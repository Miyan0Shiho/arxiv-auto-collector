# Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning

**Authors**: Yuting Liu, Wei Wu, Jianzhe Zhao, Guibing Guo

**Published**: 2026-08-10 12:11:47

**PDF URL**: [https://arxiv.org/pdf/2608.09507v2](https://arxiv.org/pdf/2608.09507v2)

## Abstract
Natural language user preferences provide an interpretable interface for LLM personalization. However, universal preference summaries often contain information irrelevant to a particular downstream task. Directly supplying the full preference summary therefore wastes context capacity and introduces cross-task distraction, while manually designing task-specific preference views is difficult to scale. In this work, we study \emph{task-specific preference adaptation}: given a universal user preference summary and a downstream task, derive a task-conditioned representation that preserves sufficient decision-relevant evidence while removing redundant context. To this end, we propose \textsc{AlignXada}, a training-free meta-learning framework that induces reusable textual refinement policies for adapting universal preference summaries to task-specific ones. The refinement policy is iteratively optimized by a meta learner through verbal reinforcement learning. Across 13 tasks and three downstream models (39 task--model cells), \textsc{AlignXada} achieves an average gain of 3.82 points, improving 33 cells while retaining only 22.8\% of the original profile tokens and outperforming RAG in 36 cells. An extended faithfulness analysis further shows that the refined profiles remain largely grounded in the source preferences while preserving task-relevant personalization signals, suggesting that profile-side adaptation serves as a practical complement to universal memory construction for lifelong personalized agents.

## Full Text


<!-- PDF content starts -->

BBAAD9C2010037A16BA0000FCD90B6403519B27D1D132B20A7D9FE32B1A92BBA9B41B73861D17B0725792708B84C113CD0E926A3E1D03B116B1AC805767E3102841088021B72C89794E38D776BE341D6A973EC0275E3C39112EE119B2BEA1C92D993819B8E3Learning Preference Adaptation for Large
Language Model Personalization via Verbal
Reinforcement Learning
Yuting Liu1,2Wei Wu2,*Jianzhe Zhao1Guibing Guo1,*
1Software College, Northeastern University, China2Ant International
*Corresponding authors: Guibing Guo and Wei Wu.
liuyuting@stumail.neu.edu.cn,wuwei19850318@gmail.com,{guogb,zhaojz}@swc.neu.edu.cn
Abstract
Natural languageuser preferences providean interpretable interface for LLM personalization. However, universal preference
summariesoftencontaininformationirrelevanttoaparticulardownstreamtask. Directlysupplyingthefullpreferencesummary
thereforewastescontextcapacityandintroducescross-taskdistraction,whilemanuallydesigningtask-specificpreferenceviews
is difficult to scale. In this work, we studytask-specific preference adaptation: given a universal user preference summary
and a downstream task, derive a task-conditioned representation that preserves sufficient decision-relevant evidence while
removing redundant context. To this end, we proposeAlignXada, a training-free meta-learning framework that induces
reusabletextualrefinementpoliciesforadaptinguniversalpreferencesummariestotask-specificones. Therefinementpolicyis
iteratively optimized by a meta learner through verbal reinforcement learning. Across 13 tasks and three downstream models
(39 task–model cells),AlignXadaachieves an average gain of 3.82 points, improving 33 cells while retaining only 22.8% of
the original profile tokens and outperforming RAG in 36 cells. An extended faithfulness analysis further shows that the refined
profiles remain largely grounded in the source preferences while preserving task-relevant personalization signals, suggesting
thatprofile-sideadaptationservesasapracticalcomplementtouniversalmemoryconstructionforlifelongpersonalizedagents.
Code is available at https://github.com/AntResearchNLP/AlignX-Family/tree/main/AlignXada.
Keywords:large language model personalization, task-specific preference adaptation, verbal reinforcement learning, meta-learning
1 Introduction
Personalization,thetechniqueofaligningAIsystemswith
humanpreferences,hasplayedanimportantroleinthetrain-
ingof largelanguage modelssince theirearly development
[Ouyang et al., 2022]. Recently, with the proliferation of
personal AI agents [OpenClaw, 2026, Hermes, 2026] and
theexpandinguseofLLMsinapplicationssuchassearch
[Baeketal.,2024],recommendation[Gengetal.,2022,Lyu
etal.,2024,Pengetal.,2026],anddialoguesystems[Otsuka
etal.,2024],personalizationtechniques—particularlyas
akeycomponentinbuildingLLMmemories[Wangetal.,
2023, Xu et al., 2025, Chhikara et al., 2025, Yu et al., 2025,
Zhongetal.,2024]—havebecomeincreasinglycriticaland
continue to drive advances at the research frontier.
A central problem in personalization is user preference
representation, which serves as the interface through which
AI systems perceive user preferences and adapt their behav-
iors accordingly. Early studies modeled user preferences
throughuserembeddingsorbyencodinguserinformationdirectly into model parameters [Koren et al., 2009, Bao
et al., 2023, Liu et al., 2025b, Zhang et al., 2025]. While
effective, such black-box representations suffer from limited
interpretability and are difficult to update in real time. With
theemergenceofLLMs,in-contextlearninghasenableda
simpleyeteffectivealternativebyexplicitlyincorporating
userbehaviorsintothemodelcontextasdemonstrations[Du
et al., 2026, Salemi et al., 2024]. However, such approaches
are fundamentally constrained by the context capacity of
LLMs,makingitdifficulttocomprehensivelycaptureuser
preferences while remaining highly sensitive to noise in
the selected examples. More recently, the advancement
ofLLMreasoningcapabilitieshasinspiredeffortstoinfer
universal user preferences from heterogeneous data sources
and summarize them in natural language [Chen et al., 2025,
Nametal.,2025,Liuetal.,2026,Yangetal.,2026]. This
paradigm offers significant improvements in interpretability
and scalability. More importantly, it lays the foundation
for lifelong personal agents, where preference representa-
Ant International Research 1
arXiv:2608.09507v2  [cs.CL]  12 Aug 2026

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
(a) Given your history with 
knee pain  from a college 
sports injury , you might 
focus on activities like 
swimming, gentle yoga …(b) To stay active without 
putting too much strain on 
your legs , you might try 
swimming, gentle yoga …
(c) Given your strong 
connection to community 
and rich cultural traditions , 
you might enjoy activities like 
Zumba …(d) Given your love of 
baking as a creative outlet, 
you might balance that 
sedentary hobby with light 
activities like swimming …What are some good ways to stay active without putting too much 
strain on the legs?Query ( task=P ersonal email)
Downstream Model: GPT -5
Model picks (c) with the universal profile.
Wrong: the dominant cross -scenario signal 
overwhelms the small wellness clue.…
Her professional life is intensely focused on public policy, 
specifically civil liberties, with a strong emphasis on data privacy, 
surveillance technology oversight, and protecting vulnerable 
communities .
...
Her political identity is characterized by a blend of fierce 
advocacy and a warm, community -oriented approach.
…
…often grounding them in personal anecdotes or cultural 
metaphors to make them more relatable and impactful.
...
She has shared stories about her family, her children, a past 
college soccer injury , a challenging pregnancy, and the stress of 
her demanding job.
...
She is conscious of her physical and mental well -being. She 
mentions a past college soccer injury , a history of gestational 
diabetes, and stable ferritin levels, suggesting a proactive 
approach to her health.
…Universal Profile 𝑃
7,675 chars | >= 6 scenarios
Only ~1% of characters are task -relevant for this query .…
Often weaves personal narratives into communications to build 
trust and add emotional weight. Examples include stories about 
her family (children, partner Daniel), a past college soccer injury , 
or a challenging pregnancy.
…
Wellness: Discusses managing stress, work -life balance, 
mindfulness, and specific health topics (e.g., past soccer injury , 
HPA axis).
…Refined Profile෨𝑃
637 chars | 8 % of raw
Model selects (a) with the refined profile.
Correct: the decision is anchored on task -relevant 
evidence.
dominant 
distractor
Induced  policy  𝜙: compress  profile,  drop distractors,  keep task evidence …task-relevant clue competing distractor neutral context
Figure 1:An example for task-specific preference adaptation. Given a query, only a small portion of the universal profile 𝑃is relevant to
thetask,whiletheremaininginformationmayactasdistractorsandleadtosuboptimalpersonalization.AlignXadarefines 𝑃byretaining
task-relevantevidenceandremovingdistractingcontext,enablingthedownstreammodeltogrounditsresponseintheappropriateuser
information.
tions can continuously evolve as user–agent interactions
accumulate over time.
In this work, we study LLM personalization from the per-
spective of universal user preference interfaces. Rather than
pursuingimprovedmethodsforconstructinguniversalprefer-
encesummaries,weassumeauniversaluserprofileisalready
available and investigate a complementary yet practically
important problem:how to effectively adapt the universal
profile to specific tasks. Our motivation stems from the
observationthatuniversalpreferencesummaries,whilecom-
prehensive,inevitablycontainredundantortask-irrelevant
information that may act as noise in downstream applica-
tions. AsillustratedinFigure1,theuserqueryisstrongly
associated with her past college injury, whereas information
suchas“community-orientedapproach”islargelyirrelevant.
Suchredundantinformationmisleadsthedownstreammodel
(i.e.,GPT-5),resultinginasuboptimalresponse. Incontrast,
theresultcanbesubstantiallyimprovedbyusinganadapted
profile that is shorter and more focused on task-relevant
information.
Toward task-specific preference adaptation, we propose a
frameworkfortransforminguniversalpreferencesummaries
into task-aware representations. Ideally, the adapted rep-
resentation should satisfy two desiderata: (1)Sufficiency,
preserving all preference information necessary for down-
stream personalization; and (2)Compactness, removing
redundant or task-irrelevant information to reduce noise and
improvecontextefficiency. Tothisend,weproposeAlignX-
ada,ameta-learningframeworkforpreferenceadaptation.
Rather than directly training a model to refine user profiles,
AlignXadaemploys a meta learner to learn structured
refinement policies in natural language, which are subse-
quentlyusedbyafrozenrefinertoadapttheuniversaluserpreference. Thisdisentanglementbetweenpolicygeneration
and preference refinement makes the adaptation process
transparent and controllable, enabling human-in-the-loop
diagnosisandrefinement. Themetalearnerisoptimizedvia
verbalreinforcementlearning,whererefinementpoliciesare
iterativelyimprovedusingnaturallanguagefeedbackderived
fromtask-specificdemonstrationsofuserpreferences. By
avoidingparameterupdatesduringpolicylearning,AlignX-
adanaturally supports both open-source and proprietary
models.
WeevaluateAlignXadaonacompositebenchmarkspan-
ning nine conversational tasks and four ranking, rating, and
generation tasks with three downstream models. Across
39 task–model cells,AlignXadaimproves 33 cells by an
average of 3.82 points while reducing the profile token ratio
to 22.8%. It strictly outperforms RAG in 36 cells, showing
that task-oriented preference reorganization provides ben-
efits beyond query-level retrieval. On PersonaMem-v2, a
faithfulnessauditfurthershowsthat97.5%ofrefined-profile
claims are supported by the source profiles, while 83.3% of
available gold preference evidence is retained, suggesting
thatAlignXadamainlyperformscontrolledtask-specific
compression and reorganization.
We summarize our contributions as follows:
•We formalize thetask-specific preference adaptation
problem, which aims to adapt universal user prefer-
ences to downstream tasks by removing redundant or
task-irrelevantinformation,therebypavingthewayfor
lifelong personalized agents equipped with memory.
•WeintroduceAlignXada,ameta-learningframework
that induces natural language refinement policies from a
smallsetoftask-specificdemonstrations. Thepolicyis
iterativelyoptimizedviaverbalreinforcementlearning,
2

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
makingAlignXadacompatible with both open-source
and proprietary LLMs.
•Weconductextensiveevaluationsacrossthirteentasks
and three downstream models. The results show that
AlignXadaconsistentlyachievesafavorabletrade-off
between task performance and context-token usage, ow-
ing to the faithfulness and compactness of the refined
preferences.
2 Related Work
2.1 LLM Personalization
Aslarge languagemodels(LLMs)evolve fromgeneralized
chatbots into personal AI agents, there has been growing
interest in building personalized AI systems in which the
behavior of a general-purpose LLM is aligned with indi-
vidual preferences. Existing approaches can be broadly
categorized into four groups. Retrieval-based methods iden-
tifyrelevantuserrecordsorprofileelementsfromexternal
memory or databases [Sun et al., 2025, Du et al., 2026, Liu
et al., 2023, Shi et al., 2025], and leverage the retrieved
content for downstream personalization. Parametric ap-
proachesencodepersonainformationintotrainablemodel
parameters, such as LoRA modules, adapters, soft prompts,
model-merging weights, and rerankers [Tan et al., 2024,
Clarke et al., 2024, Li et al., 2023, Liu et al., 2025a, Jang
etal.,2023,Zhuangetal.,2024],therebyadaptingageneral-
purpose LLM toward user-specific behavior. In addition,
motivated by the strong in-context learning capability of
LLMs, prompt-based approaches append preference signals
totheinputcontextandsteerresponsegenerationthrough
personalized prompts [Dong et al., 2023, Yang et al., 2024b,
Chengetal.,2024,Lietal.,2025a]. Morerecently,withthe
emergence of lifelong personal agents and LLM memory
systems,severalstudieshaveadvocatedlearninguniversal
userprofilesaslong-termandcontinuouslyevolvingrepre-
sentations of user preferences [Li et al., 2025b, Liu et al.,
2026,Yangetal.,2026].AlignXadaismotivatedbythis
trend toward lifelong personalization. However, rather than
constructinguniversalprofilesthemselves,AlignXadaas-
sumes such profiles are already available and studies how
to adapt them to specific downstream tasks in real-world
applications. In this sense,AlignXadacomplements ex-
istingeffortsbybridgingthegapbetweenuniversalprofile
construction and task-oriented deployment.
2.2 Textual Optimization
With advances in LLMreasoning capabilities, recent work
has used strong LLMs to optimize natural-language arti-
facts without gradient-based training. OPRO [Yang et al.,
2024a]iteratively proposestask-levelinstructionsbased on
reward trajectories, EvoPrompt [Guo et al., 2024] applies
evolutionaryoperatorstocandidateprompts,TextGrad[Yuk-
sekgonuletal.,2024]propagatesverbal“gradients”throughprompt computation graphs, and Reflexion [Shinn et al.,
2023]generatesreflectivecritiquesforper-instancetrajec-
tories. Inthesemethods,theoptimizedartifactistypically
either atask-levelinstruction shared across users or aper-
instanceself-correctiontext, making ituser-independent or
instance-local.AlignXadaextends this paradigm by for-
mulating task-specific profile adaptation as the optimization
ofareusableuser-conditionalrewritepolicythattransforms
each user’s universal preference into a task-specific repre-
sentation, rather than introducing a new general-purpose
textual optimizer.
3 Methodology
Figure 2 presents an overview ofAlignXada. In a nutshell,
AlignXadaconsistsoftwostages: few-shotpolicyinduction
and policy deployment for task-specific personalization.
During policy induction, the framework leverages a small
support set of task-specific demonstrations, each consisting
of a universal preference summary, a user query, and the
corresponding user response, and iteratively refines the
rewritingpolicygeneratedbyametalearnerusingnatural
language feedback. Once policy optimization converges,
AlignXadaentersthedeploymentstage,wheretheselected
policy is consumed by a refiner to produce a refined profile
fortaskadaptation. Therefinedprofileisthenprovidedto
downstream models for personalized inference. Throughout
theentireprocess,allmodels—includingthemetalearner,
therefiner,andthedownstreammodel—remainfrozen. In
the following, Section 3.1 formalizes the learning problem,
and Section 3.2 presents the policy induction procedure
based on verbal reinforcement learning.
3.1 Problem Formalization
Let𝑢denote a user and 𝑃𝑢denote the corresponding uni-
versal preference summary, which may be obtained from
anexternalLLMormemorysystemandistreatedasprior
knowledge in this work. Given a task 𝜏, the objective is
to derive a task-adapted profile ˜𝑃𝑢from𝑃𝑢such that the
performance of a downstream model 𝑀for user𝑢can be
substantially improved on task𝜏.
A common approach to deriving ˜𝑃𝑢from𝑃𝑢is to learn a
generativemodel 𝑅asarefineranddefine ˜𝑃𝑢=𝑅(𝑃𝑢,M𝜏),
whereM𝜏specifies thetask 𝜏. Inthis work,we instantiate
M𝜏as a support set𝑆(𝜏)defined as
𝑆(𝜏)={(𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖)}𝑏
𝑖=1,(1)
where𝑃𝑢𝑖istheuniversalprofileofuser 𝑢𝑖,𝑥𝑖denotesanin-
putprompt,𝑦𝑖denotesthecorrespondingresponse, and 𝑏is
thesizeofthesupportset. Althoughtherepresentationalca-
pacityof𝑆(𝜏)maybelimitedbytheselectionof (𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖)
tuples and the budget 𝑏, such a formulation naturally aligns
withreal-worlduser-AIinteractions: userpromptsspecify
thetask,whileuserresponsesprovidesignalsoftask-specific
3

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Upstream PreferenceSource(out of scope)Dataset Provided
Memory SystemGiventask 𝜏anddownstream model𝑀.A. Few-shot policy induction on support set𝑆!.SupportSet 𝑆(!)=𝑃$!,𝑥%,𝑦%%&'(Refinement Policy 𝜙)-Goal-Preserve-Compress-Avoid-Output Style-PriorityRefiner 𝑅frozenexecutorDownstream Model𝑀frozentarget model𝑆𝑐𝑜𝑟𝑒!(3𝑦,𝑦)Reward 𝑟())StructuredFeedback 𝐸)Meta Learner𝜋*+),frozen policy generator𝜙)-'t = 1..T roundserrors, transitions,…Best-epoch selection: argmax over𝜙.…𝜙/Frozen policy𝜙(!)for task 𝜏B. Policydeploymentfortask-specificpersonalization.Query Example𝑃,𝑥+Frozen Policy𝜙(!)Refiner 𝑅frozen executorRefined Preference 8𝑃DownstreamModel𝑀frozen target modelPrediction3𝑦All models frozen, only policy text updates.
Preference Summarizer…
Figure 2:Overview ofAlignXada. For each task,AlignXadainduces a task-specific textual refinement policy from a small support set
in several rounds and then freezes the selected policy for held-out deployment. All models remain frozen throughout, and only the policy
text is updated during induction.
preferences and behavioral patterns, thereby alleviating the
cold-start problem.
Instead of updating the parameters of 𝑅, we keep the
model frozen and learn a textual refinement policy 𝜙(𝜏)
using a meta learner 𝜋meta. The task-adapted profile is then
defined as ˜𝑃𝑢=𝑅(𝑃𝑢,𝜙(𝜏)). The learning objective is to
maximize
E(𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖)∼𝑆(𝜏)
Score𝜏 𝑀(𝑥𝑖,˜𝑃𝑢𝑖),𝑦𝑖
,(2)
where Score𝜏(·,·)denotestheevaluationfunctionfortask
𝜏.
We propose a verbal reinforcement learning approach
to optimize Eq. (2), where the refinement policy 𝜙(𝜏)is
iterativelyestimatedfrom 𝜋metausingfeedbackderivedfrom
𝑆(𝜏). Details are presented in next section.
3.2 Policy Induction
Overview.Algorithm 1 presents the task-specific pol-
icy induction procedure inAlignXada. Starting from a
predefined task-agnostic initial policy 𝜙0(detailed in Ap-
pendix F.1),AlignXadaiteratively induces a refinement
policyfortask 𝜏throughtwocoreflows:rolloutandupdate.
Ateachround 𝑡,therolloutflowevaluatesthecurrentpolicy
𝜙𝑡onthesupportset 𝑆(𝜏)byapplyingtherefiner 𝑅,query-
ingthedownstreammodel 𝑀,andcomputingtask-specific
scores. Theupdateflowthensummarizestherolloutrecords
into structured feedback 𝐸𝑡, which is used by the meta
learner𝜋metatorevisethepolicyandproducethenextpolicy
𝜙𝑡+1. Afterallrounds,AlignXadareturnsthepolicywith
the highest development-set performance.
Rollout.At round 𝑡,AlignXadaevaluates the current
refinementpolicy 𝜙𝑡oneachsupportexample (𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖)∈𝑆(𝜏). The refiner 𝑅applies𝜙𝑡to rewrite the universal
preference summary into a task-adapted profile:
˜𝑃𝑢𝑖,𝑡=𝑅(𝑃𝑢𝑖,𝜙𝑡).(3)
ThedetailedprompttemplateisprovidedinAppendixE.1.
Thedownstreammodel thenconditionsontheinput𝑥 𝑖and
the refined profile ˜𝑃𝑢𝑖,𝑡to generate a prediction:
ˆ𝑦𝑖,𝑡=𝑀(𝑥𝑖,˜𝑃𝑢𝑖,𝑡).(4)
Thispredictionisevaluatedagainstthereferenceresponse
𝑦𝑖using the task-specific evaluation function:
𝑠𝑖,𝑡=Score𝜏(ˆ𝑦𝑖,𝑡,𝑦𝑖).(5)
The rollout record at round𝑡is then defined as
R𝑡={(𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖,˜𝑃𝑢𝑖,𝑡,ˆ𝑦𝑖,𝑡,𝑠𝑖,𝑡)}|𝑆(𝜏)|
𝑖=1.(6)
To reduce overfitting,AlignXadaevaluates each policy
on a development set 𝐷(𝜏)disjoint from 𝑆(𝜏). Applying
Eqs. 3–5 to its examples yields development scores 𝑠𝐷
𝑖,𝑡,
whose average defines the policy utility:
𝐽𝐷(𝜏)(𝜙𝑡)=1
|𝐷(𝜏)||𝐷(𝜏)|∑︁
𝑖=1𝑠𝐷
𝑖,𝑡.(7)
The evaluated policy and its development-set utility are
stored in the history:
H←H∪{(𝜙 𝑡,𝐽𝐷(𝜏)(𝜙𝑡))}.(8)
MaintainingthishistoryallowsAlignXadatoretainalleval-
uatedpolicies,asverbalpolicyupdatesarenotguaranteed
to improve monotonically.
4

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Algorithm 1Task-specific policy induction inAlignXada.
Require: Task𝜏;supportset 𝑆(𝜏);developmentset 𝐷(𝜏);
frozen meta learner 𝜋meta; frozen refiner 𝑅; frozen
downstream model 𝑀; evaluation function Score𝜏; ini-
tial policy𝜙 0; number of rounds𝑇
Ensure:Task-specific refinement policy𝜙(𝜏)
1:H←∅⊲history of evaluated policies
2:for𝑡=0,1,...,𝑇do
3:R𝑡←∅⊲support-set rollout records
4:for all(𝑃 𝑢𝑖,𝑥𝑖,𝑦𝑖)∈𝑆(𝜏)do
5: ˜𝑃𝑢𝑖,𝑡←𝑅(𝑃𝑢𝑖,𝜙𝑡)
6:ˆ𝑦 𝑖,𝑡←𝑀(𝑥𝑖,˜𝑃𝑢𝑖,𝑡)
7:𝑠 𝑖,𝑡←Score𝜏(ˆ𝑦𝑖,𝑡,𝑦𝑖)
8:R𝑡←R𝑡∪
{(𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖,˜𝑃𝑢𝑖,𝑡,ˆ𝑦𝑖,𝑡,𝑠𝑖,𝑡)}
9:end for
10:𝐽𝐷(𝜏)(𝜙𝑡)←1
|𝐷(𝜏)|Í|𝐷(𝜏)|
𝑖=1𝑠𝐷
𝑖,𝑡
11:H←H∪{(𝜙 𝑡,𝐽𝐷(𝜏)(𝜙𝑡))}
12:if𝑡 <𝑇then
13:𝐸 𝑡←AGG(R 𝑡)
14:𝜙 𝑡+1←𝜋 meta(𝜙𝑡,𝐸𝑡)
15:end if
16:end for
17:𝜙(𝜏)←arg max(𝜙𝑡,𝐽𝐷(𝜏)(𝜙𝑡))∈H𝐽𝐷(𝜏)(𝜙𝑡)
18:return𝜙(𝜏)
Policy update.After the rollout,AlignXadaconstructs
structured feedback:
𝐸𝑡=AGG(R 𝑡),(9)
where AGG(·)converts the rollout records into a textual
diagnosticsummary. Thefeedbackincludesscalarsignals,
such as the average support-set score; instance-level signals,
such as predictions and scores; and representative failure
patterns derived from low-scoring examples. The complete
feedback format is provided in Appendix E.3.
Given the current policy and structured feedback, the
frozen meta learner produces a revised policy:
𝜙𝑡+1=𝜋 meta(𝜙𝑡,𝐸𝑡).(10)
The update prompt (provided in Appendix E.2) instructs
𝜋metato follow the predefined policy schema, make targeted
revisions based on the feedback, and avoid example-specific
rules that merely memorize support instances.
The revised policy may adjust the refinement goal, the
typesofuserevidencetopreserve,thecompressionstrategy,
the error patterns to avoid, the output style, or the prior-
ity order among these instructions. An example policy is
providedinAppendixF.2. Inthisway,themetalearnercon-
vertsrolloutdiagnosticsintoareusabletask-levelrefinement
policy rather than producing direct answers or per-example
corrections.After𝑇rounds,AlignXadaselectsandreturnsthebest
evaluated policy as the final task-specific refinement policy:
𝜙(𝜏)=arg max(𝜙𝑡,𝐽𝐷(𝜏)(𝜙𝑡))∈H𝐽𝐷(𝜏)(𝜙𝑡).(11)
4 Experiments
4.1 Experimental Setup
Benchmark.To evaluate profile refinement in multi-
domain, lifelong personalization settings, we construct a
compositebenchmarkbyintegratingPersonaMem-v2[Jiang
et al., 2025] and MemoryCD [Zhang et al., 2026]. We
first derive task-agnostic user summaries from both datasets
and extract semantic signatures capturing stable interests,
preferences, aversions, and contextual constraints. We then
match users one-to-one based on semantic compatibility,
excludingpairswithexplicitpreferenceconflicts. Foreach
matched pair, we construct a shared universal profile by
interleaving and summarizing their PersonaMem and Mem-
oryCD histories. Each composite user is evaluated with the
same universal profile across 13 downstream tasks: nine
PersonaMem-v2 conversational tasks and four MemoryCD
tasks—item ranking, rating prediction, review-title gener-
ation, and review generation. For each task, we construct
user-disjoint support, development, and evaluation sets, en-
suring that users involved in policy induction do not appear
in the held-out evaluation set. Detailed dataset statistics are
providedin AppendixA. Wealso evaluatethetwo original
benchmarks separately and report the results in Section 4.3.
Evaluation Metrics.Weevaluatefour-waychoiceques-
tions withexactaccuracy,item ranking with Hit@1, rating
prediction withRating-Score, and both generation tasks
with ROUGE-L. User profiles are constructed under the
all-historymemorysetting,whichsummarizesuserbehavior
fromthecompletehistoricalcontext. Giventheground-truth
rating𝑦and model prediction ˆ𝑦∈{1,2,3,4,5} , Rating-
Score is defined asS(ˆ𝑦,𝑦)=max
0,1−|ˆ𝑦−𝑦|
4
.
Alongsidetask-levelscores,wereportcontextcompres-
sionusingthetokenratio(TR),definedastheaveragerefined-
profilelengthdividedbytheaverageuniversal-profilelength.
Implementation Details.In the main experiments, we
use Gemini-2.5-Pro [Comanici et al., 2025] to generate
a universal user preference from each user’s raw history
providedbythebenchmark. Unlessotherwisestated,both
the meta learner and the preference refiner use Gemini-2.5-
Pro. Wesetthesupportbatchsizeto 𝑏=20andthenumber
ofpolicy-updateroundsto 𝑇=5,subjecttoourexperimental
budget. When more demonstrations are available, we select
thesupportexamplesusingtheadaptivesamplingmethod
described in Appendix B to mitigate sampling bias. The
universal-preferencebaselinedirectlypassestheuniversal
preferencetothedownstreammodelwithoutrefinement. The
RAG baseline uses BM25 [Robertson and Zaragoza, 2009].
5

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Table 1:Overall performance (%) on the composite benchmark with different downstream models. Higher metric is better unless
otherwise specified. Token ratio (TR) denotes the ratio of refined profile token length to those in the original profile. Task-level gains over
Raw are significant under a two-sided exact sign test (𝑝=2.44×10−4).
Task Qwen3-8B DeepSeek-V4-Flash GPT-5-mini
Raw RAGAlignXadaTR(%)↓Raw RAGAlignXadaTR(%)↓Raw RAGAlignXadaTR(%)↓
Chat
Acc.43.46 40.78 −2.68 45.63+2.17 18.9 55.34 41.75 −13.59 64.08+8.74 18.1 43.40 40.78 −2.6249.51+6.11 19.6
Creative
Acc.55.68 46.59 −9.09 56.82+1.14 21.7 59.09 54.55 −4.5473.86+14.77 22.5 59.32 61.36 +2.0465.91+6.59 28.9
Knowledge
Acc.72.19 61.54 −10.65 75.00+2.81 14.382.9071.01 −11.89 82.84−0.06 33.575.2171.01 −4.2073.63−1.58 18.7
Personal
Acc.34.5233.33 −1.1933.33−1.19 24.9 45.24 39.29 −5.95 54.76+9.52 20.7 47.62 41.67 −5.9550.00+2.38 17.1
Prof.Email
Acc.39.1735.00 −4.1737.67−1.50 25.1 46.67 35.83 −10.84 59.17+12.5 25.841.5038.33 −3.1740.83−0.67 26.4
Prof.Writing
Acc.31.82 27.27 −4.55 40.00+8.18 14.6 50.00 40.91 −9.09 57.27+7.27 27.8 44.55 36.36 −8.1949.09+4.54 16.8
Social
Acc.(%)36.36 34.09 −2.27 40.91+4.55 20.3 56.82 53.41 −3.41 62.50+5.68 37.9 48.41 48.86 +0.4551.14+2.73 19.2
Translation
Acc.38.64 34.23 −4.41 42.24+3.60 24.4 54.95 45.95 −9.00 63.06+8.11 18.6 54.95 50.45 −4.5059.46+4.51 21.7
Trouble
Acc.49.51 46.60 −2.91 53.40+3.88 22.3 60.19 55.34 −4.85 62.14+1.94 24.355.19 55.19 +0.0054.37−0.82 18.5
Ranking
Hit@159.61 55.77 −3.84 61.53+1.92 20.8 69.23 78.85 +9.6282.69+13.46 31.9 78.69 76.92 −1.7780.77+2.08 26.7
Rating
Score75.84 75.48 −0.36 77.40+1.56 28.2 74.04 76.44 +2.40 78.37+4.33 27.2 77.8880.29 +2.4179.33+1.45 26.4
Title
ROUGE-L13.46 12.99 −0.47 15.38+1.92 13.1 12.78 10.76 −2.02 16.02+3.24 20.4 13.04 13.17 +0.1313.80+0.76 30.3
Review
ROUGE-L13.94 13.90 −0.04 14.37+0.43 22.2 13.89 13.86 −0.03 15.32+1.43 22.8 11.61 12.03 +0.4212.06+0.45 18.4
Avg.Δ – -3.59+2.2720.8 – -4.86+7.0025.5 – -1.92+2.1922.2
Foreachquery,wesegmenttheuniversalprofileatstructural
boundariesandfurtherdivideitintooverlappingwindowsof
120 words with a 30-word overlap. We use the downstream
taskpromptastheretrievalqueryandrankallprofilechunks
with BM25 ( 𝑘1=1.5,𝑏=0.75). The top eight chunks
are restored to their original order and concatenated within
an approximate 768-token profile budget ( 𝑇𝑅≈20% ),
which is comparable to the length of profiles refined by
our method. The downstream models are Qwen3-8B [Yang
et al., 2025], DeepSeek-V4-Flash [DeepSeek-AI, 2026], and
GPT-5-mini [OpenAI, 2025]. We evaluate generalization
with DeepSeek-V4-Flash as the meta model in Appendix C,
providefurtheranalysesbeyondperformancecomparisons
in Appendix D, and present case studies in Appendix G.
Unlessotherwisespecified,Qwen3-8Bisusedasthedefault
downstream model for these analyses.
4.2 Main Results
WeevaluatewhetherAlignXadaimprovestheperformance–
context trade-off across heterogeneous personalization tasks
anddownstreammodels. Aneffectivepreferencerefinement
method should reduce preference context while preserv-
ing or improving downstream performance. We compare
AlignXadawith the raw universal-preference and RAG
baselines across39task–model cells.
The results in Table 1 yield three observations.(1)
AlignXadaconsistently improves the performance–efficiency trade-off across models and tasks.Across
the39task–model cells,AlignXadaimproves 33, with an
average gain of+3.82points over the raw universal pref-
erence. Allthreedownstreammodelsimproveonaverage:
+2.27points for Qwen3-8B, +7.00for DeepSeek-V4-Flash,
and+2.19for GPT-5-mini. The gains also span task for-
mats, covering all 12MemoryCD cells. Meanwhile, the
refined profiles retain only 22.8%of the original tokens,
and performance gains are nearly uncorrelated with token
ratio (𝑟=0.06). Thus,AlignXadaimproves the utility
ofretainedevidenceratherthanrelyingonlongerprofiles.
(2) The advantage over RAG shows that task-oriented
preference adaptation goes beyond query-level retrieval.
AlignXadaoutperformsRAGin 36cells,tiesinone,and
underperforms it in only two, with an average margin of
+7.28points. Incontrast,RAGreducestheraw-profilescore
by3.46points on average. For example, on professional
email with DeepSeek-V4-Flash, RAG lowers accuracy from
46.67%to35.83%,whereasAlignXadaraisesitto 59.17%.
Retrieval may surface locally relevant records but fragment
preferenceswhoserelevanceisindirectordistributedacross
interactions.AlignXadainstead reorganizes the consol-
idated profile into a coherent decision context.(3) The
optimalpreferencerepresentationdependsonboththe
taskandthedownstreammodel.Forprofessionalemail,
thechangesare−1.50,+12.50,and−0.67pointsforQwen3-
8B, DeepSeek-V4-Flash, and GPT-5-mini, respectively; for
6

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Table 2:Task-level performance and token ratios on separate
PersonaMem-v2 and MemoryCD (values×100%).
Task Raw RAGAlignXadaTR↓
Benchmark=PersonaMem-v2
Chat Message 30.11 27.4230.9131.4
Creative Writing 31.13 27.1332.7837.0
Knowledge Query49.9243.98 48.71 43.3
Personal Email 30.93 29.8332.3940.4
Professional Email 31.09 27.7233.8338.7
Professional Writing28.4624.73 27.66 51.6
Social Media Post 25.99 29.6629.9439.7
Translation 31.80 29.6832.4338.5
Trouble Consult 32.77 33.0635.2946.3
Avg.Δ– -2.11+1.3040.8
Benchmark=MemoryCD
Item Ranking 57.14 54.2561.1462.1
Rating Prediction 78.86 75.8180.0059.5
Review Title 13.47 14.3214.3644.7
Review Generation 13.72 14.0414.4625.8
Avg.Δ– -1.19+1.6948.0
knowledgequery,theyare +2.81,−0.06,and−1.58points.
These sign reversals indicate that no single compression
strategyisoptimalforalldownstreammodels,motivatingthe
useofdownstreamfeedbackforpolicyinduction. Despite
thisheterogeneity,allregressionsremainwithin 1.58points,
whereas the largest gain reaches+14.77points.
4.3 Results on Source-Native Benchmarks
Thecompositebenchmarkmergesheterogeneoushistories
from PersonaMem-v2 and MemoryCD into a universal user
profile, exposing each downstream task to both relevant
and unrelated evidence. Although this setting reflects the
noisyhistoriesoflifelongagents,AlignXadamaybenefit
mainly from removing cross-domain noise introduced by
benchmarkconstruction. WethereforeevaluateAlignXada
separatelyonthetwosourcebenchmarks,whereprofilesare
built from more domain-coherent native histories, to test
whether it remains effective from a cleaner starting point.
The results in Table 2 show two patterns.AlignXada
remains effective on cleaner, source-native profiles.It
improves the average primary metric by 1.30points on
PersonaMem-v2, outperforming the raw profile on seven
of nine tasks, and by 1.69points on MemoryCD, improv-
ing all four tasks. These gains indicate thatAlignXada
goesbeyondremovingcross-domainnoisebyusingsupport-
set feedback to induce preference representations better
aligned with downstream requirements.Feedback-based
adaptation provides a better performance–compression
trade-offthanquery-levelretrieval.AlignXadaoutper-
formsRAGoneverytask,whereasRAGreducestheaverage
score by 2.11points on PersonaMem-v2 and 1.19points on
MemoryCD.Meanwhile,AlignXadareducestheprofiles
to average token ratios of 40.8%and48.0%, respectively.
051015202530
Mean profile token ratio (%)
Lower is better
5 10 20
Feedback batch size-101234Primary metric change (pp)
Higher is better
4/13 improved
-0.337/13 improved
+0.7311/13 improved
+2.58
Task trends (norm.) Mean Median Token ratioFigure3:Effectoffeedbackbatchsizeonthecompositebench-
mark. Solid and dashed lines show the mean and median primary-
metricchanges,respectively. Graylinesshownormalizedtask-level
trends, bars show the mean profile token ratio, and annotations
indicatethenumberoftasksoutperformingtheraw-profilebaseline.
This suggests that support-set feedback reorganizes user
evidenceintoacompactrepresentationthatismoreuseful
to downstream models than locally retrieved excerpts.
4.4 Hyperparameter Analysis
Impact of Support Batch Size 𝑏.Figure 3 compares
feedback batch sizes 𝑏∈{5,10,20} while keeping the in-
duction pool and update-round budget fixed.(1) Larger
feedback batches improve the reliability of policy induc-
tion.As𝑏increases from 5to20, the mean primary-metric
changerisesfrom−0.33to+2.58points,themedianfrom
−0.91to+1.94points, and the number of improved tasks
from 4/13to11/13(7/13at𝑏=10). Smaller batches
makeeachupdatemoresensitivetothesampledexamples,
whereas larger batches provide broader and more balanced
feedback, leading to more stable policies.(2) The gains
come from better feedback coverage rather than weaker
compression. 𝑏=20achievesthehighestmeanandmedian
gainswhileproducingthelowestmeantokenratio( 20.5%,
compared with 21.1%at𝑏=5and21.7%at𝑏=10).
Thus, its advantage does not result from retaining more con-
text. Instead,broaderfeedbackhelpsdistinguishrecurring
evidence-losspatterns fromisolatedfailuresand prioritize
moreusefulpreferenceevidence. Wethereforeuse 𝑏=20
as the default, as it provides the best accuracy–compression
trade-offamongtheevaluatedsettings. Largerbatchesare
not evaluated because they exceed the context budget.
Impact of the Number of Update Rounds 𝑇.Figure4
showswhenthefinalselectedpolicyfirstappearsindiagnos-
tic runs with 𝑇∈{5,10} .(1) Effective refinement policies
aretypicallylearnedwithinafewupdates.Eightofthethir-
teen task-specific policies emerge by the second round, and
7

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
0 1 2 3 4 5 6 7 8 9 10
The number of update round of the final selected policy.Chat Message
Creative Writing
Knowledge Query
Personal Email
Professional Email
Professional Writing
Social Media Post
Translation
Trouble Consult
Item Ranking
Rating Prediction
Review Generation
Review TitleE5
E2
E1
E1
E1
Initial
E1
E3
E5
E2
E9
E5
E2
T=5 budget 12/13 policies learned by E5At or before epoch 5 After epoch 5
Figure 4:Update round of the final selected policy in the 𝑇∈
{5,10}diagnostic runs. Blue bars denote policies obtained by
round five, red bars denote later policies.
Claim
SupportNon-
HallucinationNon-
ContradictionSource
CoverageOptimized
RetentionConditional
RetentionNon-critical
Missing020406080100Score (%)97.5 97.999.8
34.030.283.3 84.7Claim faithfulness Decision evidenceFaithfulness Audit
Figure 5:Faithfulness audit result on the PersonaMem-v2.
twelvewithinfiverounds,indicatingthatthemeta-learner
canquicklytranslatesupport-setfeedbackintoeffectivepoli-
cies.(2) Additional rounds offer task-dependent benefits.
Rating prediction is the only task whose final policy first
appears after round five, at round nine. Moreover, among
the seven tasks for which 𝑇=10performs better, six select
policies generated within thefirst fiverounds. Thus, most
tasks converge early, while additional rounds mainly benefit
challenging or unstable tasks. Balancing policy quality and
inference cost, we use𝑇=5in the main experiments.
4.5 Faithfulness Audit
Performanceandcompressionalonedonotestablishwhether
a refined preference can serve as a faithful substitute for the
user universal preference. A refiner may reduce the context
budget while removing the decisive preference evidence
orintroducefictionaltraits. Wethereforeaudittherefined
preferences along two dimensions:claim-level faithfulness
to the input profile andpreference evidence retention.
Experimental details are elaborated in Appendix D.2.
Wehave twomainobservations aboutAlignXadafrom
Figure5. (1)AlignXadaishighlyfaithfultothesource
profile.Across all scenarios, 97.5%of extracted claimsaresupportedbytheoriginalprofile,thenon-hallucination
rateis 97.9%,andthenon-contradictionratereaches 99.8%.
Theseresultsshowthattheverbalpolicydoesnotturnthe
refiner into an unconstrained generator. Instead,AlignX-
adalargely functions as a controlled compression-and-
reorganization operator that preserves the factual content of
the universal preference representation while adapting it for
downstream use. (2)The main bottleneck is incomplete
decision-evidenceavailabilityratherthanprofilehallu-
cination.Only 34.0%of benchmark preference evidence
is recoverable from the original source profiles, while re-
fined profiles retain 30.2%overall. This small gap from the
source-evidenceceilingsuggeststhatmanyerrorsarisefrom
insufficientevidenceintheuniversalprofileitself,ratherthan
from the refiner deleting available evidence. Conditional
on the original profile containing the golden preference,
AlignXadapreserves itin 83.3%of cases, with a critical
missing rate of 15.3%. Therefore, the audit provides a
more precise interpretation ofAlignXada’s downstream
limitations:AlignXadacan faithfully compress and adapt
theprofileitreceives,butitsperformanceceilingdepends
on whether the universal profile contains the cross-topic
evidence needed for the decision. The remaining headroom
mainlyliesinimprovingsource-profilecoverageandmaking
decision-relevant evidence more consistently available to
the refiner.
5 Conclusion
In this work, we study LLM personalization from the per-
spectiveofuniversaluserpreferenceinterfaces. Wepropose
AlignXada, which induces a reusable natural-language
refinementpolicythroughverbalreinforcementlearning,im-
proving the performance–budget trade-off while preserving
source-supported claims and critical evidence. By enabling
effective use of universal preferences, our work offers a new
perspective on memory adaptation for lifelong personalized
agents.
Limitations
Despiteitsadvantages,AlignXadahastwomainlimitations.
First, although our evaluation spans PersonaMem-v2 and
MemoryCD across multiple task formats, both benchmarks
areconstructedorcuratedevaluationsettingsandmaynot
fullycapturetheevolving,noisyinteractionsofreal-world
lifelongagents. Futureworkshoulddevelopbroaderdatasets
that provide both universal user preferences and diverse
downstream task formats, enabling more comprehensive
evaluation of task-specific preference adaptation. Second,
AlignXadaassumes that the support set used for policy
induction is representative of the target task distribution.
When the support set is biased or insufficiently diverse, the
induced refinement policy may not generalize well. More
effective sampling and support-set construction strategies
8

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
remain important directions for future work.
Ethics Statement
All experimental datasets used in this study are derived
frompreviouslypublishedworkandobtainedeitherthrough
official APIs or by synthetic construction based on these
sources. All user-related information in the datasets has
beenanonymized,andnopersonallyidentifiableinformation
isinvolved. Wedo notuseanynon-open-source data. All
data are used solely for scientific research, rather than for
commercial purposes or for profiling or decision-making
about individuals, and their acquisition and use comply
withrelevantethicalguidelinesandstandardsofacademic
integrity. All existing resources used in our experiments,
includingdatasets,pretrainedmodels,andAPIs,areaccessed
andusedinaccordancewiththeiroriginallicensesandterms
ofuse. Thedatasetsandmodelswerelyonwerefilteredand
processed by their original authors prior to public release to
mitigate potential ethical risks.
References
Jinheon Baek, Nirupama Chandrasekaran, Silviu Cucerzan,
Allen Herring, and Sujay Kumar Jauhar. Knowledge-
augmentedlargelanguagemodelsforpersonalizedcon-
textual query suggestion. InProceedings of the ACM
Web Conference 2024, pages 3355–3366, 2024. doi:
10.1145/3589334.3645404. URLhttps://doi.org/10.114
5/3589334.3645404.
Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli
Feng, and Xiangnan He. TALLRec: An effective and
efficient tuning framework to align large language model
withrecommendation. InProceedingsofthe17thACM
Conference on Recommender Systems, pages 1007–1014,
2023. doi: 10.1145/3604915.3608857. URL https:
//doi.org/10.1145/3604915.3608857.
Yizhuo Chen, Xin Liu, Ruijie Wang, Zheng Li, Pei Chen,
ChanglongYu,QingyuYin,PriyankaNigam,MengJiang,
and Bing Yin. POPI: Personalizing LLMs via optimized
natural language preference inference.arXiv preprint
arXiv:2510.17881, 2025. URL https://arxiv.org/abs/2510
.17881.
ChuanqiCheng,QuanTu,WeiWu,ShuoShang,CunliMao,
ZhengtaoYu,andRuiYan. “in-dialogueswelearn”: To-
wards personalized dialogue without pre-defined profiles
through in-dialogue learning. InProceedings of the 2024
ConferenceonEmpiricalMethodsinNaturalLanguage
Processing, pages 10408–10422, 2024.
PrateekChhikara,DevKhant,SaketAryan,TaranjeetSingh,
and Deshraj Yadav. Mem0: Building production-ready ai
agentswithscalablelong-termmemory.arXivpreprint
arXiv:2504.19413, 2025.Christopher Clarke, Yuzhao Heng, Lingjia Tang, and Jason
Mars. PEFT-U: Parameter-efficient fine-tuning for user
personalization.arXivpreprintarXiv:2407.18078,2024.
URL https://arxiv.org/abs/2407.18078.
Gheorghe Comanici et al. Gemini 2.5: Pushing the frontier
with advanced reasoning, multimodality, long context,
and next generation agentic capabilities, 2025. URL
https://arxiv.org/abs/2507.06261.
DeepSeek-AI. DeepSeek-V4: Towards highly efficient
million-token context intelligence, 2026. URL https:
//huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/
main/DeepSeek_V4.pdf.
Yi Dong, Zhilin Wang, Makesh Narsimhan Sreedhar, Xi-
anchao Wu, and Oleksii Kuchaiev. Steerlm: Attribute
conditionedsftasan(user-steerable)alternativetorlhf. In
FindingsoftheAssociationforComputationalLinguistics:
EMNLP 2023, 2023.
Linfeng Du, Ye Yuan, Zichen Zhao, Fuyuan Lyu, Emil-
ianoPenaloza,XiuyingChen,ZipengSun,JikunKang,
Laurent Charlin, Xue Liu, and Haolun Wu. Optimiz-
ing user profiles via contextual bandits for retrieval-
augmented LLM personalization, 2026. URL https:
//arxiv.org/abs/2601.12078. Accepted to ACL 2026.
Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge,
and Yongfeng Zhang. Recommendation as language
processing(RLP):Aunifiedpretrain,personalizedprompt
and predict paradigm (P5). InProceedings of the 16th
ACMConferenceonRecommenderSystems,pages299–
315, 2022. doi: 10.1145/3523227.3546767. URL
https://doi.org/10.1145/3523227.3546767.
QingyanGuo,RuiWang,JunliangGuo,BeiLi,KaitaoSong,
XuTan,GuoqingLiu,JiangBian,andYujiuYang. Evo-
Prompt: Connecting LLMs with evolutionary algorithms
yields powerful prompt optimizers. InInternational Con-
ference on Learning Representations (ICLR), 2024. URL
https://arxiv.org/abs/2309.08532.
Hermes. Hermes agent: The agent that grows with you,
2026. URL https://hermes-agent.nousresearch.com/.
Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong Wang,
Jack Hessel, Luke Zettlemoyer, Hannaneh Hajishirzi,
Yejin Choi, and Prithviraj Ammanabrolu. Personal-
ized soups: Personalized large language model align-
ment via post-hoc parameter merging.arXiv preprint
arXiv:2310.11564, 2023. URL https://arxiv.org/abs/2310
.11564.
Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao,
Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao
Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Gre-
gory Wornell, Lyle Ungar, Dan Roth, Sihao Chen, and
9

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Camillo Jose Taylor. PersonaMem-v2: Towards personal-
izedintelligencevialearningimplicituserpersonasand
agentic memory.arXiv preprint arXiv:2512.06688, 2025.
URL https://arxiv.org/abs/2512.06688.
Yehuda Koren, Robert Bell, and Chris Volinsky. Matrix
factorization techniques for recommender systems.Com-
puter, 42(8):30–37, 2009. doi: 10.1109/MC.2009.263.
URL https://doi.org/10.1109/MC.2009.263.
Jia-Nan Li, Jian Guan, Songhao Wu, Wei Wu, and Rui
Yan. From 1,000,000 users to every user: Scaling up
personalizedpreferenceforuser-levelalignment.ArXiv,
abs/2503.15463, 2025a. URL https://api.semanticscholar.
org/CorpusID:277113478.
Jia-Nan Li, Jian Guan, Wei Wu, and Rui Yan. Extended
inductive reasoning for personalized preference inference
frombehavioralsignals.ArXiv,abs/2505.18071,2025b.
URLhttps://api.semanticscholar.org/CorpusID:278886
858.
Lei Li, Yongfeng Zhang, and Li Chen. Personalized
promptlearningforexplainablerecommendation.ACM
Transactions on Information Systems, 41(4), 2023. doi:
10.1145/3580488. URL https://doi.org/10.1145/3580488.
JiongnanLiu,YutaoZhu,ShutingWang,XiaochiWei,Erxue
Min,YuLu,ShuaiqiangWang,DaweiYin,andZhicheng
Dou. LLMs + persona-plug = personalized LLMs. In
Proceedingsofthe63rdAnnualMeetingoftheAssociation
forComputational Linguistics(Volume1: Long Papers),
pages 9373–9385, 2025a. doi: 10.18653/v1/2025.acl-lon
g.461. URL https://aclanthology.org/2025.acl-long.461/.
Shuai Liu, Hyundong Cho, Marjorie Freedman, Xuezhe
Ma, and Jonathan May. RECAP: Retrieval-enhanced
context-aware prefix encoder for personalized dialogue
responsegeneration. InProceedingsofthe61stAnnual
Meeting of the Association for Computational Linguistics
(Volume1: LongPapers),pages8404–8419,2023. doi:
10.18653/v1/2023.acl-long.468. URLhttps://aclantholo
gy.org/2023.acl-long.468/.
Yuting Liu, Jinghao Zhang, Yizhou Dang, Yuliang Liang,
Qiang Liu, Guibing Guo, Jianzhe Zhao, and Xingwei
Wang. CoRA: Collaborative information perception
by large language model’s weights for recommenda-
tion. InProceedings of the AAAI Conference on Ar-
tificial Intelligence, volume 39, pages 12246–12254,
2025b. doi: 10.1609/aaai.v39i12.33334. URL
https://doi.org/10.1609/aaai.v39i12.33334.
Yuting Liu, Jian Guan, Jia-Nan Li, Wei Wu, Jiang-Ming
Yang,JianzheZhao,andGuibingGuo. Textasauniversal
interfacefortransferablepersonalization.arXivpreprint
arXiv:2601.04963, 2026. URL https://arxiv.org/abs/2601
.04963.HanjiaLyu,SongJiang,HanqingZeng,YinglongXia,Qifan
Wang, Si Zhang, Ren Chen, Chris Leung, Jiajie Tang,
and Jiebo Luo. LLM-rec: Personalized recommendation
via prompting large language models. InFindings of
theAssociationforComputationalLinguistics: NAACL
2024, pages 583–612. Association for Computational
Linguistics, 2024. doi: 10.18653/v1/2024.findings-naacl
.39. URLhttps://aclanthology.org/2024.findings-naacl.3
9/.
Hyunji Nam, Yanming Wan, Mickel Liu, Peter Ahnn,
JianxunLian,andNatashaJaques. Learningtosummarize
user information for personalized reinforcement learning
from human feedback.arXiv preprint arXiv:2507.13579,
2025. URL https://arxiv.org/abs/2507.13579.
OpenAI. GPT-5 system card, 2025. URL https://openai.c
om/index/gpt-5-system-card/.
OpenClaw. Openclaw: The ai that actually does things,
2026. URL https://openclaw.ai/.
Atsushi Otsuka, Kazuya Matsuo, Ryo Ishii, Narichika
Nomoto, and Hiroaki Sugiyama. User-specific dia-
logue generation with user profile-aware pre-training
model and parameter-efficient fine-tuning.arXiv preprint
arXiv:2409.00887, 2024. URL https://arxiv.org/abs/2409
.00887.
LongOuyang,JeffreyWu,XuJiang,DiogoAlmeida,Carroll
Wainwright, Pamela Mishkin, Chong Zhang, Sandhini
Agarwal,KatarinaSlama,AlexRay,JohnSchulman,Ja-
cobHilton,FraserKelton,LukeMiller,MaddieSimens,
Amanda Askell, Peter Welinder, Paul F Christiano, Jan
Leike, and Ryan Lowe. Training language models to
follow instructions with human feedback. In S. Koyejo,
S.Mohamed,A.Agarwal,D.Belgrave,K.Cho,andA.Oh,
editors,Advances in Neural Information Processing Sys-
tems,volume35,pages27730–27744.CurranAssociates,
Inc., 2022. URL https://proceedings.neurips.cc/paper_fil
es/paper/2022/file/b1efde53be364a73914f58805a0017
31-Paper-Conference.pdf.
Zhiyuan Peng, Xuyang Wu, Huaixiao Tou, Yi Fang, and
YuGong. Memrerank: Preference memoryforpersonal-
ized product reranking, 2026. URL https://arxiv.org/abs/
2603.29247.
StephenRobertsonandHugoZaragoza. Theprobabilistic
relevance framework: Bm25 and beyond.Foundations
andTrendsinInformationRetrieval,page333–389,2009.
Alireza Salemi, Surya Kallumadi, and Hamed Zamani.
Optimization methods for personalizing large language
models through retrieval augmentation.arXiv preprint
arXiv:2404.05970, 2024. URL https://arxiv.org/abs/2404
.05970.
10

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
TengShi,JunXu,XiaoZhang,XiaoxueZang,KaiZheng,
Yang Song, and Han Li. Retrieval augmented generation
withcollaborativefilteringforpersonalizedtextgeneration.
arXiv preprint arXiv:2504.05731, 2025. URL https:
//arxiv.org/abs/2504.05731.
NoahShinn,FedericoCassano,AshwinGopinath,Karthik
Narasimhan, and Shunyu Yao. Reflexion: Language
agentswithverbalreinforcementlearning. InAdvancesin
Neural Information Processing Systems (NeurIPS), pages
8634–8652,2023. URLhttps://papers.nips.cc/paper_fil
es/paper/2023/hash/1b44b878bb782e6954cd888628510
e90-Abstract-Conference.html.
Chenkai Sun, Ke Yang, Revanth Gangi Reddy, Yi Fung,
Hou Pong Chan, Kevin Small, ChengXiang Zhai, and
Heng Ji. Persona-DB: Efficient large language model
personalizationforresponsepredictionwithcollaborative
data refinement. InProceedings of the 31st International
ConferenceonComputationalLinguistics,pages281–296,
2025. URL https://aclanthology.org/2025.coling-main.
20/.
Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu,
BingYin,andMengJiang. Democratizinglargelanguage
modelsviapersonalizedparameter-efficientfine-tuning.
arXiv preprint arXiv:2402.04401, 2024. URL https:
//arxiv.org/abs/2402.04401. AcceptedtoEMNLP2024
Main.
Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng
Yan, Jianfeng Gao, and Furu Wei. Augmenting language
modelswithlong-termmemory. InAdvancesinNeural
InformationProcessingSystems(NeurIPS),2023. URL
https://arxiv.org/abs/2306.07174.
WujiangXu,ZujieLiang,KaiMei,HangGao,JuntaoTan,
andYongfengZhang. A-MEM:AgenticmemoryforLLM
agents. InAdvances in Neural Information Processing
Systems(NeurIPS),2025. URLhttps://arxiv.org/abs/25
02.12110. arXiv:2502.12110.
An Yang et al. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.
Bufang Yang, Lilin Xu, Yixuan Li, Kaiwei Liu, Xi-
aofan Jiang, and Zhenyu Yan. Sensorpersona: An
llm-empoweredsystemforcontinualpersonaextraction
from longitudinal mobile sensor streams, 2026. URL
https://arxiv.org/abs/2604.06204.
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu,
Quoc V. Le, Denny Zhou, and Xinyun Chen. Large
languagemodelsasoptimizers. InInternationalConfer-
enceonLearningRepresentations(ICLR),2024a. URL
https://arxiv.org/abs/2309.03409.Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han
Zhong, Dong Yu, and Jianshu Chen. Rewards-in-context:
Multi-objective alignment of foundation models with
dynamicpreferenceadjustment.InProceedingsofthe41st
International Conference on Machine Learning, pages
56276–56297, 2024b.
HongliYu,TinghongChen,JiangtaoFeng,JiangjieChen,
Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma,
Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshap-
ing long-context llm with multi-conv rl-based memory
agent.arXiv preprint arXiv:2507.02259, 2025.
Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng
Liu, Zhi Huang, Carlos Guestrin, and James Zou.
TextGrad: Automatic “differentiation” via text.arXiv
preprint arXiv:2406.07496, 2024. URL https://arxiv.org/
abs/2406.07496.
JinghaoZhang,YutingLiu,WenjieWang,QiangLiu,Shu
Wu, Liang Wang, and Tat-Seng Chua. Personalized
text generation with contrastive activation steering. In
Proceedingsofthe63rdAnnualMeetingoftheAssociation
forComputational Linguistics(Volume1: Long Papers),
pages 7128–7141,2025. doi: 10.18653/v1/2025.acl-lon
g.353. URL https://aclanthology.org/2025.acl-long.353/.
Weizhi Zhang, Xiaokai Wei, Wei-Chieh Huang, Zheng Hui,
ChenWang,MichelleGong,andPhilipSYu. Memorycd:
Benchmarkinglong-contextusermemoryofllmagents
for lifelong cross-domain personalization.arXiv preprint
arXiv:2603.25973, 2026.
WanjunZhong,LianghongGuo,QiqiGao,HeYe,andYanlin
Wang. MemoryBank: Enhancing large language models
with long-term memory. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 38, pages
19724–19731, 2024. doi: 10.1609/aaai.v38i17.29946.
URL https://doi.org/10.1609/aaai.v38i17.29946.
Yuchen Zhuang, Haotian Sun, Yue Yu, Rushi Qiang, Qifan
Wang, Chao Zhang, and Bo Dai. HYDRA: Model fac-
torization framework for black-box LLM personalization.
InAdvances in Neural Information Processing Systems
(NeurIPS), 2024. URL https://arxiv.org/abs/2406.02888.
arXiv:2406.02888.
11

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Table 3:Statistics of the constructed composite benchmark.
Task Support Query Total
Personal Email 40 84 210
Professional Email 40 120 256
Professional Writing 40 110 269
Creative Writing 40 88 239
Translation 40 111 264
Trouble Consult 40 103 243
Chat Message 40 103 226
Social Media Post 40 88 217
Knowledge Query 40 169 416
Item Ranking 40 52 131
Rating Prediction 40 52 131
Review Title Generation 40 52 131
Review Generation 40 52 131
Total 520 1,184 2,864
Table 4:Statistics of the PersonaMem-v2 benchmark.
Task Support Query Total
Personal Email 49 424 473
Professional Email 50 485 535
Professional Writing 63 453 516
Creative Writing 52 457 509
Translation 52 478 530
Trouble Consult 49 452 501
Chat Message 49 444 493
Social Media Post 56 426 482
Knowledge Query 61 661 722
Total 481 4,280 4,761
A Benchmark Statistics
Tables3–5summarizethetaskdistributionsusedinourmain
and source-native evaluations. Since a user may contribute
multiple targets and PersonaMem-v2 users do not necessar-
ily have an example in every conversational scenario, the
number ofavailablerecords variesacross tasks. In Table3,
Supportdenotes the fixed induction budget used for each
task,Querydenotes the held-out instances used for final
evaluation, andTotaldenotes all constructed task records
beforeruntimesupportanddevelopmentsubsampling;the
latter therefore also includes candidate induction and devel-
opment records not displayed in separate columns. Overall,
the composite benchmark contains 2,864records across 13
tasks. For the source-native evaluations, PersonaMem-v2
contributes 4,761examplesacrossnineconversationaltasks,
whileMemoryCDcontributes 2,240examplesacrossfour
recommendation and generation tasks. All experiments use
user-disjointinductionandevaluationsplitstopreventpolicyTable 5:Statistics of the MemoryCD benchmark.
Task Support Query Total
Item Ranking 160 400 560
Rating Prediction 160 400 560
Review Title Generation 160 400 560
Review Generation 160 400 560
Total 640 1600 2240
learning from exploiting evaluation-user histories.
B Adaptive Sampling for Policy Learning
Themetalearnerhasalimitedcontextbudget,soeachpolicy
updatecanuseatmost 𝑏supportexamplesevenwhenalarger
candidatepoolisavailable. Thismakesthechoiceofsupport
examples important. Fixed or randomly sampled batches
may over-represent easy successes, persistent failures, or
isolated regressions, leading to updates that are either too
conservative or too reactive. To obtain a more balanced
diagnosticview,weuseadaptivediagnosticsampling,which
selects support examples from multiple outcome-transition
states.
In the main experiments, we reserve a candidate pool
C(𝜏)={(𝑃𝑢𝑖,𝑥𝑖,𝑦𝑖)}𝑚
𝑖=1for adaptive support selection and
adevelopmentset 𝐷(𝜏)forfinalpolicyselection. Here, 𝑚
denotesthenumberofcandidateexamplesavailableinthe
experimental split rather than a method-level budget. At
each update, the sampler selects 𝑏examples fromC(𝜏)to
form the support set 𝑆(𝜏), where𝑏≤𝑚. This allows each
updatetoremainwithinthecontextbudgetwhilepreserving
broadercoverageacrossusers,topics,anddifficultylevels.
In deployment, the candidate-pool size is determined by the
available task-specific interactions.
For each support example, we compare two binary out-
comes: the outcome under the raw universal preference
andthemostrecentoutcomeunderthecurrentrefinement
policy. Thiscomparisonassignseachexampletooneoffour
transition states:
•improved: the raw profile is incorrect, but the refined
profile is correct, indicating useful rewritten-profile rep-
resentations;
•regressed: the raw profile is correct, but the refined
profile is incorrect, exposing evidence that may have
been removed or made less usable;
•stable-success: both the raw and refined profiles are
correct, identifying safe compression behavior;
•persistent-failure: boththerawandrefinedprofilesare
incorrect,revealingevidencerequirementsthatneither
profile makes accessible to the downstream model.
Ateachupdateround,thesamplerpartitionsthecandidate
pool by these transition states and allocates the support
budget uniformly across them. If 𝑏is not divisible by
12

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
60 65 70 75 80
Context token reduction (%) - higher is better-4-202468Primary metric change (pp) - higher is betterChat Message
Creative Writing
Knowledge QueryPersonal Email
Professional EmailProfessional Writing
Social Media PostTranslation
Trouble Consult
Item RankingRating Prediction
Review Title
Review GenerationFixed Adaptive
Figure6:Fixedversusadaptivesupportsamplingonthecomposite
benchmark. Each arrow connects the same task under fixed
sampling(circle)and adaptivesampling(square). Thehorizontal
axisreportscontext-tokenreductionandtheverticalaxisreports
thechangeinthetask-specificprimarymetric;higherisbetteron
both axes.
four,theremainingslotsareassigneddeterministically,with
prioritygiventoimprovedandregressedexamples. Ifa
transition state has too few examples, its unused slots are
redistributed to the remaining states. This strategy provides
themetalearnerwithbothcorrectivesignalsfromfailures
and stabilizing signals from successes in each update.
We compare fixed and adaptive sampling in Figure 6 and
maketwoobservations.(1)Adaptivesamplingimproves
the overall performance–compression trade-off rather
than trading additional context for better predictions.
The mean task-level gain in the primary metric increases
from+1.22points under fixed sampling to +2.58points
under adaptive sampling, while the number of tasks that
improve over the raw-profile baseline rises from 7/13to
11/13. At the same time, the mean context-token reduction
increases from 68.3%to72.7%. Ten of the thirteen arrows
moveupward,tenmoverightward,andeightmoveinboth
directions. Adaptive sampling therefore yields a broadly
more favorable shift inperformance–compression space by
improvingtheselectionoffeedbackexamplesusedforpolicy
induction, rather than by retaining longer refined profiles.
(2)Theprimarybenefitistherecoveryoftasksforwhich
fixedsamplinginducesunstableorincompletepolicies.
Adaptive sampling changes the gains for creative writing,
knowledge query, review title generation, and review gener-
ation fromnegative to positive. It also increasesthe gainfor
ratingpredictionfrom +0.48to+5.85points,professionalTable 6:Performance and token ratio with DeepSeek-V4-Flash
as the meta model, rewrite model, and the target model (values
×100%).
Task Raw RAGAlignXada-DTR↓
Chat Message 55.34 41.75 60.24 16.2
Creative Writing 59.09 54.55 68.71 23.1
Knowledge Query 82.90 71.01 79.16 30.7
Personal Email 45.24 39.29 48.81 16.9
Professional Email 46.67 35.83 56.67 24.6
Professional Writing 50.00 40.91 52.13 26.5
Social Media Post 56.82 53.41 61.06 31.8
Translation 54.95 45.95 61.48 20.4
Trouble Consult 60.19 55.34 60.91 22.3
Item Ranking 69.23 78.85 82.42 33.7
Rating Prediction 74.04 76.44 77.83 26.2
Review Title 12.78 10.76 15.35 20.5
Review Generation 13.89 13.86 14.50 18.4
writingfrom+5.45to+8.18points,andtroubleconsultation
from+0.97to+3.88points. Thesetasksbenefitfromrepeat-
edlyexposingthemeta-learnertobothregressions,which
revealdecision-relevantevidencediscardedbythecurrent
policy,andpersistentfailures,whichrevealevidencethatthe
policystillfailstosurface. Theeffectisnotuniform: fixed
sampling remains stronger for chat message, personal email,
and translation, while professional email remains below the
raw-profilebaseline. Adaptivesamplingshouldthereforebe
interpretedasamorereliablestrategyforallocatingfeedback
across heterogeneous tasks, rather than as a guarantee of
improvement on every task.
CGeneralization Across Different Meta Mod-
els
To assess whetherAlignXadageneralizes beyond its de-
faultmodelconfiguration,wereplaceGemini-2.5-Prowith
DeepSeek-V4-Flash forpolicy inductionand profilerewrit-
ing. We also use DeepSeek-V4-Flash as the downstream
model, yielding an all-DeepSeek variant,AlignXada-D,
while keeping the learning procedure and experimental
protocol unchanged.
AlignXadaremains effective when all model roles
use DeepSeek-V4-Flash, indicating that its gains are not
specific to Gemini-2.5-Pro.AlignXada-D improves 12
of the 13tasks over the raw universal preference, with an
averagegainof+4.47points. Itimproveseightofthenine
PersonaMem-v2 tasks, averaging +4.22points, and all four
MemoryCD tasks, averaging +5.04points. The refined
profiles retain only 23.95%of the original tokens. These
results demonstrate thatAlignXadatransfers across model
families and task formats while maintaining a favorable
performance–efficiency trade-off.
ItsconsistentadvantageoverRAGsuggeststhatthis
transferstemsfrompreference-leveladaptationrather
13

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
than model-specific retrieval behavior.AlignXada-D
outperforms RAG on all 13tasks by 9.33points on aver-
age, whereas RAG falls 4.86points below the raw-profile
baseline. Its largest gains over the raw profile occur on
item ranking (+13.19), professional email ( +10.00), and
creative writing (+9.62), spanning ranking, classification,
and open-ended generation. Thus, feedback-guided pref-
erence reorganization remains effective even when policy
induction,rewriting,anddownstreaminferenceusethesame
model family.
Differentpolicy-inductionandrewritingmodelsnever-
thelessproducedistinctperformance–compressiontrade-
offs.For the same DeepSeek-V4-Flash downstream model,
the default Gemini-based configuration achieves an average
gain of+7.00points with a token ratio of 25.5%, compared
with+4.47points and 23.95%forAlignXada-D.AlignX-
ada-Dthereforeproducesslightlymorecompactprofilesbut
smallerperformancegains,withknowledgequeryasitsonly
regression (−3.74points). Theseresults distinguish frame-
workgeneralityfrommodelinterchangeability:AlignXada
transfers across model families, but the choice of policy-
inductionandrewritingmodelsstillaffectshoweffectively
decision-relevant evidence is identified and retained.
D Diagnostic Analysis on PersonaMem-v2
D.1 Preference Source Robustness
To examine whetherAlignXada’s behavior depends on
the quality and format of the universal preference represen-
tation, we construct three sources of universal preference
information on PersonaMem-v2.Source Asummarizes
a user’s identity, background, interests, communication
style, and other attributes using GPT-5 [OpenAI, 2025],
which is provided by the benchmark. We use A to test
whetherAlignXadacanstillimproveacompact,curated
persona preference that has already removed much of the
conversational detail.Source Buses the persona’s raw
structured JSON record from the benchmark. This record is
the generator-side persona artifact used to create the bench-
mark’s preferences, conversation snippets, and answers. To
avoiddirectconversationleakage,weremovethebenchmark
conversationbranchwhileretainingthestructuredpersona
and preference fields. We use B as a high-information
structured-source stress test to measure whether a verbal
refinercanreplacehand-designedfieldselectionwhenex-
plicit preferences are available.Source Csummarizes each
user’s raw chat history into a comprehensive natural lan-
guage preference description using Gemini-2.5-Pro. Unlike
AandB,Cisthereforeahistory-derivedpreferencerather
than apersona-generation artifact. We useC asthe default
source in the main experiments because it best matches our
target setting: a system first summarizes long user–assistant
interactionsintoarichbuttask-agnosticpreferencesummary,
andAlignXadathen adapts that profile into a task-specificpreference representation.
TheresultsareshowninTable7,fromwhichwedrawthe
following observations.(1)AlignXadaworks best as a
task-specificrefinerfornatural-languagepreferencepro-
files,ratherthanasareplacementforupstreamprofile
construction.BothSourceAandSourceCimproveafter
refinement, with absolute gains of +0.61%and+1.30%,
respectively, whereas Source B starts from the strongest
raw baseline but decreases after refinement. This pattern
suggests thatAlignXadabenefits fromsources whose evi-
dence is already expressed in natural language, while raw
structuredrecordsmayrequireamorefield-awaretransfor-
mation.(2)ThegainsonAandCcomefromreorganizing
available preference evidence into a more task-oriented
form.Source A has the lowest raw accuracy ( 25.67%),
because the dataset-provided expanded persona is compact
and often lacks the full cross-topic evidence required by
downstream tasks.AlignXadastill improves it to 26.28%,
showingthatevenalimitedpersonacanbenefitfromtask-
specific reorganization. Source C starts from a stronger raw
baseline ( 32.47%), as the comprehensive profile contains
richeruserhistory,andAlignXadafurtherimprovesitto
33.77%. This makes Source C the best match for our target
setting: it is sufficiently evidence-rich for personalization
while remaining in natural language, allowing the refiner
toreliablycompressandreorganizeit.(3)ThedroponB
suggests a format mismatch rather than a lack of useful
information.Source B’s high raw accuracy ( 32.91%) is
expected,becauseitisthestructuredpersonausedtosynthe-
size the benchmark and often contains explicit preference
aligned with the query-required evidence. However, after
refinement, accuracy drops to 32.07%. One likely reason is
thatmanyusefulsignalsarestoredasfine-grainedstructured
fields or leaf values; rewriting them into a compact natural-
languageprofilecanremoveexactdecisionevidencethatthe
downstreammodelcandirectlyread. Fromthecompression
perspective,thisresultalsoshowsthatstrongcompression
aloneisinsufficient. Thebesttrade-offisachievedbySource
C,whereAlignXadaobtainsapositivegainwhilereducing
the profile to40.8%of its original token length.
D.2 Faithfulness Audit Details
In Section 4.5, we audit refined preferences along two
dimensions:claim-level faithfulnessto the input profile
andpreferenceevidenceretention. Theexperimentalsetup
and implementation details are as follows:
Claim-level faithfulness audit.For each unique raw–
refinedpreferencepair,weuseDeepSeek-V4-Pro[DeepSeek-
AI, 2026] as the judge model to decompose the refined
preference into atomic claims and label each claim with
respecttotheuniversalpreferenceassupported,contradicted,
not found, ortoo vague. Let 𝑁𝑐denote the total number
ofatomicclaimsacrossallauditedpairs. Thethreeclaim-
levelmetrics—ClaimSupport,Non-Hallucination,andNon-
14

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Table 7:Ablation on the source of the universal preference (values×100%).
Source A Source B Source C
TaskRawAlignXadaTR(↓) RawAlignXadaTR(↓) RawAlignXadaTR(↓)
Chat Message 28.37 28.64 36.8 30.19 29.73 16.4 30.11 30.91 31.4
Creative Writing 24.58 24.13 49.7 33.42 31.89 31.2 31.13 32.78 37.0
Knowledge Query 33.21 37.84 35.3 52.76 50.18 23.6 49.92 48.71 43.3
Personal Email 22.73 23.46 53.2 31.95 30.27 28.7 30.93 32.39 40.4
Professional Email 23.64 21.37 45.9 29.58 28.42 27.3 31.09 33.83 38.7
Professional Writing 24.82 25.16 45.6 26.94 27.31 26.5 28.46 27.66 51.6
Social Media Post 26.35 26.78 45.2 30.67 29.14 23.8 25.99 29.94 39.7
Translation 25.91 27.32 41.5 34.28 34.76 21.4 31.80 32.43 38.5
Trouble Consult 21.46 21.83 34.7 26.37 26.92 30.1 32.77 35.29 46.3
Avg.Δ– +0.61 43.1 – -0.84 25.4 – +1.30 40.8
Contradiction—are defined as follows:
Claim Support=#supp.
𝑁𝑐,
Non-Hallucination=1−#notf.+#cont.
𝑁𝑐,
Non-Contradiction=1−#cont.
𝑁𝑐.(12)
Here,supportedindicates that a claim is explicitly stated or
semantically entailed by the universal preference,not found
indicates unsupported new information, andcontradicted
indicates inconsistency with the source. Claims labeledtoo
vagueareincludedin 𝑁𝑐butarenotcountedassupported.
We define the judge prompt as follows:
Claim-Level Faithfulness Audit
Audit the refined profile against the original
profile.
Labels:
- supported: the claim is explicitly stated or
semantically entailed by the original profile.
- contradicted: the claim conflicts with the
original profile.
- not_found: the claim adds specific
information not present in the original
profile.
- too_vague: the claim is so generic that it
cannot be checked as a concrete profile claim.
Return exactly one JSON object with this
schema:
{
"claims": [
{
"claim": "atomic claim from the refined
profile",
"label": "supported | contradicted |
not_found | too_vague","evidence": "short supporting or
contradicting evidence from the original
profile, or empty string",
"rationale": "brief reason"
}
]
}
Split the refined profile into concise atomic
claims. Do not invent claims that are not in
the refined profile.
<OriginalProfile>
{original_profile}
</OriginalProfile>
<RefinedProfile>
{refined_profile}
</RefinedProfile>
Preference-evidence audit.PersonaMem-v2 provides a
“preference” field for each benchmark query, which we treat
asground-truthdecisionevidence. Foreachauditedquery,
the judge determines whether this evidence is preserved
in the universal preference and in the refined preference.
Each preference is labeled asexact/specific,semantically
generalized,missing, orcontradicted, where the first two
labels are counted as evidenceretained. Let 𝑁𝑒denote
the number of audited query examples. We define four
evidence-levelmetrics: SourceCoverage,RefinedRetention,
Conditional Retention, and Non-Critical Missing:
Source Coverage=#source ret.
𝑁𝑒,
Refined Retention=#refined ret.
𝑁𝑒,
Conditional Retention=#source ret.& refined ret.
#source ret.,
Non-Critical Missing=1−#critical missing
#source ret..(13)
15

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
Here,source ret.indicates that the universal preference
contains the ground-truth evidence, andrefined ret.indi-
cates that the refined preference preserves it. Acritical
missingcase occurs when the universal preference retains
the evidence but the refined preference is labeledmissing.
The judge prompt is defined as follows:
Preference-Evidence Audit
Audit whether each profile preserves the
benchmark preference.
Labels:
- exact_or_specific: the profile explicitly
preserves the preference or a highly specific
equivalent.
- semantic_generalized: the profile preserves
the preference at a broader but still useful
level.
- missing: the profile does not contain usable
evidence for the preference.
- contradicted: the profile states the opposite
or says the preference was retracted/should not
be used.
Return exactly one JSON object with this
schema:
{
"original": {
"label": "exact_or_specific |
semantic_generalized | missing |
contradicted",
"evidence": "short evidence span from the
original profile, or empty string",
"rationale": "brief reason"
},
"optimized": {
"label": "exact_or_specific |
semantic_generalized | missing |
contradicted",
"evidence": "short evidence span from the
refined profile, or empty string",
"rationale": "brief reason"
}
}
<BenchmarkPreference>
{benchmark_preference}
</BenchmarkPreference>
<OriginalProfile>
{original_profile}
</OriginalProfile>
<RefinedProfile>
{refined_profile}
</RefinedProfile>
E Prompt TemplatesE.1 Rewrite Prompt Template
Therefinerreceivesthecurrentpolicy 𝜙𝑡,thedownstream
task family description, and the original profile𝑃 𝑢.
Prompt Template for Preference Generation
Current rewrite pattern phi_t:
{current_policy}
Downstream task family:
{task_description}
Original profile P_u:
{original_profile}
Rewrite the profile strictly according to
phi_t.
Requirements:
1. Execute phi_t exactly, including
its preservation, compression, abstraction,
structure, and style rules.
2. Preserve factual faithfulness to P_u; do
not add unsupported facts, preferences, or
assumptions.
3. Keep the rewrite query-agnostic and do
not optimize it for a specific prompt, option,
answer, or evaluation instance.
4. Output only the rewritten profile P*_u.
E.2 Meta-Learner Prompt Template
The meta learner receives the current policy and mini-batch
feedback, then emits only the next policy.
Prompt Template for Policy Update
Meta prompt version:
{prompt_version}
Downstream task family:
{task_description}
Primary optimization metric:
{primary_metric_name}
Current pattern phi_t:
{current_policy}
Support-set error information (JSON):
{feedback_payload}
Optimization objective:
- Improve future rewritten-profile performance
for the same task family while updating only
the rewrite policy, not the downstream answer
behavior.
- Preserve or improve the primary metric first;
reduce redundant context only after retaining
decision evidence.
- Keep the policy query-agnostic so it can serve
unseen examples rather than the current support
prompts.
16

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
- Treat concrete user facts, constraints,
sensitive or anti-stereotypical preferences,
and privacy behavior as potential decision
evidence unless the feedback shows they are
harmful.
Use feedback:
- Prioritize regressions as evidence of omitted,
distorted, or under-weighted signals; preserve
behaviors that explain improvements and stable
successes.
- Separate evidence loss from downstream
confusion, ambiguity, or source-profile
insufficiency before changing the policy.
- Balance latent evidence with ordinary
concrete evidence such as roles, places,
objects, language/register, named interests,
and everyday preferences.
Constraints:
- Make the smallest useful update to phi_t and
avoid overfitting to one example, user, topic,
or surface form.
- next_pattern must describe how to rewrite
profiles, not how to answer the downstream
task.
- Return only the final JSON.
E.3 Structured Feedback Template
Thestructuredfeedbackrecord 𝐸𝑡isserializedasaJSON
list,whereeachentrycorrespondstoonesupportexamplein
themini-batch. Bydefault,thefeedbackdoesnotincludethe
fullrawpreference;instead,themetalearnerobservesthe
rewrittenprofile,pairedoutcomes,scores,andtaskmetadata.
Feedback Template
[
{
"idx": int,
"prompt": string,
"reference": string,
"optimized_profile": string,
"original_prediction": string,
"optimized_prediction": string,
"original_primary_score": float,
"optimized_primary_score": float,
"score_delta": float,
"original_metrics": object,
"optimized_metrics": object,
"meta_summary": {
"transition": "improved | regressed |
persistent_failure | stable_success",
"scenario": string,
"pref_type": string,
"topic_query": string,
"reference_letter": string,"gold_signal": string,
"original_prediction": string,
"original_signal": string,
"optimized_prediction": string,
"predicted_signal": string,
"prediction_changed": bool
},
"original_profile_words": int,
"optimized_profile_words": int,
"word_reduction": int
}
]
F Policy Examples
F.1 Initial Policy
All experiments start from the same task-agnostic initial
policy:
Initial Policy
Rewrite the profile into concise, stable
preference rules that preserve relevant user
signals while reducing unnecessary detail.
F.2 An example of Induced Policies
Example Induced Policy
Goal: Produce a compact, faithful refined
profile that lets a Qwen3-8B downstream
answer a 4-way personal-email choice question
grounded in the user’s relationship context,
tone, and recurring email phrasing, while
preserving cross-scenario evidence that
disambiguates the choice.
Preserve:
- Relationship roles and names of recurring
email recipients.
- Tone, greetings, signoffs, and recurring
phrasing by relationship.
- Lifestyle constraints and cross-scenario
preferences that affect email decisions.
Compress:
- Narrative chit-chat from chat history.
- Duplicated preferences or long descriptions
that do not change the email decision.
Avoid:
- Removing cross-scenario evidence or confusing
the user’s preferences with those of family
members.
- Copying exact answer-text phrasings from chat
history.
Output Style:
17

ANT INTERNATIONAL RESEARCH Learning Preference Adaptation for Large Language Model Personalization via Verbal Reinforcement Learning
- Bulleted, with explicit field labels and
relationship/scenario tags.
- No prose narrative.
Priority: When in doubt, preserve
cross-scenario preference evidence over
local email-style fluff.
G Case Study
We select a representative user case, together with its raw
universal preference and refined preference, as a case study.
Therawuniversalpreferenceleadsthedownstreammodel
to an incorrect answer, whereas theAlignXada-refined
profile leads it to the correct one. The user asks for calming
activities or audio options after a long day. Among the four
candidateresponses,thegoldanswerrecommendsatranquil
BachharpsichordsuiteorsoftlyrecordedGregorianchant.
With the raw universal preference, the downstream model
selectsaresponsecenteredonJapaneseshakuhachiandkoto
music; after refinement, it selects the gold response. The
boxesbelowshowlightlyformattedexcerptscopiedfromthe
corresponding JSON outputs, with omitted material marked
by ellipses.
Raw Universal Preference Excerpt
This user is a highly educated, articulate, and reflective
individual, likely an academic in the humanities, with
a professional focus on history, theology, and liturgy,
particularly within the Anglican tradition. Their name
isElaineH.Nakamura. TheyresideinornearPortland,
Oregon, and are of East Asian, specifically Japanese,
descent. Elaineisdeeplyengagedinbothherprofessional
life as a university lecturer and her personal life within
a parish community, where she is a lay leader and choir
member.
... The rhythms of the church calendar (Advent, feast
days) are significant markers in her life.
Her aesthetic sensibilities are refined and consistent. She
appreciates:
-Music:Sheisachoirmember,rehearsingandperforming
worksbycomposerslikeTallis. Shehasasophisticatedap-
preciationforclassicalmusic(Bach’sJesu,meineFreude),
choral traditions, and the emotional impact of music in
different settings. She also enjoys karaoke with friends.
-Literature:Sheisanavidreader,enjoying19th-century
novels likeMiddlemarch, mystery novels (P.D. James,
SarahWaters),andpoetry. Readingisacherishedritual,
often done in the evening with tea and a cozy blanket.
-VisualArts&Design:Sheisdrawntoartgalleries,the-
ater(bothscriptedandimprovisational),andtheaesthetics
of interior design ...AlignXada-Refined Profile Excerpt
Tone & Style
-Formality:Prefersapolite,formal,andarticulatecom-
munication style, even in personal emails.
-Tone:Warm, reflective, and slightly literary.
-Vocabulary:Uses precise, rich, sensory vocabulary
(e.g., “faint patter of rain,” “ancient cadences”).
...
Content & Interests
-Professional Identity:University lecturer in Church
History. This is a central part of her identity and informs
her worldview.
-Core Academic Interests:Liturgical practices and
history (Anglican tradition, medieval mystery plays, fu-
neral liturgies); theology and its intersection with culture;
19th-century parish library development.
-Cultural & Aesthetic Preferences:
-Music:Classical (Bach), choral traditions (Tallis).
Participates in a parish choir. Also enjoys karaoke with
friends.
-Literature:19th-centurynovels(Middlemarch),mys-
terynovels(P.D.James,SarahWaters),poetry. Reading
is a cherished ritual.
-Arts:Enjoys art galleries, theater (scripted and im-
prov),andappreciateshandmadecraftslikegreetingcards.
This example shows thatAlignXadacan improve down-
stream behavior without introducing new user facts. The
rawpreferencealreadycontainstherelevantevidence,but
it presents decision-relevant musical and liturgical prefer-
ences alongside many other salient identity and cultural
details. In this case, the raw model appears to overweight
theuser’sJapaneseheritageandselectsthecandidatemen-
tioningshakuhachiandkoto. Therefinedprofilereorganizes
the same source-supported information into task-relevant
clusters: classical music, choral practice, Anglican parish
life, 19th-century reading, and quiet evening rituals. This
makes the Bach/Gregorian-chant candidate more directly
supported than the Japanese-instrument distractor, while
reducing the profile by46.6%.
18