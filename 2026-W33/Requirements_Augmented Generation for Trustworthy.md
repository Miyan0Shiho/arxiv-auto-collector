# Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software

**Authors**: Fanyu Wang, Chetan Arora, Zhenping Xie, Yonghui Liu, Kla Tantithamthavorn, Aldeida Aleti, Siwei Jiang

**Published**: 2026-08-13 08:49:59

**PDF URL**: [https://arxiv.org/pdf/2608.12970v1](https://arxiv.org/pdf/2608.12970v1)

## Abstract
LLM-based software (LBS) integrates large language models as core components to deliver flexible, personalised responses. Unlike traditional software with deterministic outputs, LBSs exhibit context-dependent, stochastic behaviour that renders classical acceptance testing and test oracles insufficient: the same query may require fundamentally different responses depending on user personas and software context. This gap creates an urgent need for automated acceptance testing frameworks that autonomously interpret user instructions, while reliably inferring user intentions in a changing environment. In this paper, we present an automated acceptance testing framework for LBS with calibrated verdict reliability via two technical contributions. First, we introduce Requirements-Augmented Generation (REAG), which interprets user intentions by retrieving relevant software requirements, domain knowledge, and personas via adaptive RAG and self-reasoning to generate context-aware test oracles. Second, recognising that oracle generation may retrieve irrelevant constraints, misinterpret intent, or hallucinate requirements, we introduce a confidence-calibrated cascade judgment. This method quantifies verdict reliability via simulated expert agreement -- accepting high-confidence verdicts, escalating ambiguous cases, or abstaining when uncertain -- with empirical reliability guarantees backed by conformal risk control. An industrial case study on a production nutrition advisory application demonstrates that REAG achieves a 3.91/5 oracle quality score, reaching qualified or marginal oracle quality in 82% of cases. The confidence-calibrated cascade achieves 98.8% accuracy, improves oracle quality from 3.91 to 4.30 by filtering unqualified outputs, and delivers a 31.7% cost-efficiency improvement over single-judge baselines, validating industrial viability

## Full Text


<!-- PDF content starts -->

Requirements-Augmented Generation for Trustworthy
Acceptance Testing of LLM-Based Software
Fanyu Wang
Monash University
Melbourne, Australia
fanyu.wang@monash.eduChetan Arora
Monash University
Melbourne, Australia
chetan.arora@monash.eduZhenping Xie
Jiangnan University
Wuxi, China
xiezp@jiangnan.edu.cnYonghui Liu
Australian National
University
Canberra, Australia
yonghui.liu@anu.edu.au
Kla Tantithamthavorn
Monash University
Melbourne, Australia
chakkrit@monash.eduAldeida Aleti
Monash University
Melbourne, Australia
aldeida.aleti@monash.eduSiwei Jiang
Tianjin University
Tianjin, China
Yiwuyishi Intelligent
Technology Co., Ltd.
Nantong, China
jiangsiwei@tju.edu.cn
Abstract
LLM-based software (LBS) integrates large language models as
core components to deliver flexible and personalised responses.
Unlike traditional software with deterministic outputs, LBSs ex-
hibit context-dependent, stochastic behaviour that renders classical
acceptance testing and test oracles insufficient: the same query
may require fundamentally different responses depending on users’
personas and software context. This gap creates an urgent need for
automated acceptance testing frameworks that can autonomously
interpret users’ instructions, while also raising the challenge of
how we can reliably interpret users’ intentions in a changing envi-
ronment. In this paper, we present an automated acceptance test-
ing framework for LBS with calibrated verdict reliability through
two technical contributions. First, we introduceREquirements-
AugmentedGeneration (REAG), which interprets user intentions
by retrieving relevant software requirements, domain knowledge,
and personas via adaptive RAG and self-reasoning to generate
context-aware test oracles. Second, recognising that oracle gen-
eration may retrieve irrelevant constraints, misinterpret intent,
or hallucinate requirements, we introduce a confidence-calibrated
cascade judgment that quantifies verdict reliability via simulated
expert agreement, accepting high-confidence verdicts, escalating
ambiguous cases, or abstaining when uncertain, with empirical re-
liability guarantees backed by conformal risk control. An industrial
case study on a production nutrition advisory application demon-
strates that REAG achieves a 3.91/5 oracle quality score, successfully
achieving qualified or marginal oracle quality in 82% of cases. The
confidence-calibrated cascade achieves 98.8% accuracy, improves
oracle quality from 3.91 to 4.30 by filtering unqualified outputs,
and delivers a 31.7% cost-efficiency improvement over single-judge
baselines, validating industrial viability.
This work is licensed under a Creative Commons Attribution 4.0 International License.
ASE ’26, Munich, Germany
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2882-2/2026/10
https://doi.org/10.1145/3832783.3834491CCS Concepts
•Software and its engineering →Acceptance testing;Require-
ments analysis.
Keywords
LLM-based software, Acceptance testing, Test Oracle, Requirements
Engineering
ACM Reference Format:
Fanyu Wang, Chetan Arora, Zhenping Xie, Yonghui Liu, Kla Tantithamtha-
vorn, Aldeida Aleti, and Siwei Jiang. 2026. Requirements-Augmented Gen-
eration for Trustworthy Acceptance Testing of LLM-Based Software. In
Proceedings of the 41st IEEE/ACM International Conference on Automated
Software Engineering (ASE ’26), October 12–16, 2026, Munich, Germany.ACM,
New York, NY, USA, 13 pages. https://doi.org/10.1145/3832783.3834491
1 Introduction
Large Language Models (LLMs), with broad applications across a
wide range of tasks, are increasingly adopted in modern software
as core functional components [ 36,40,59]. Unlike traditional soft-
ware, which produces deterministic outputs for identical inputs,
LLM-based software (LBS) behaves stochastically and is intent-
driven, where the same query may require fundamentally different
responses depending on the user’s persona, health profile, or op-
erational context [ 13,15]. This context-sensitivity is by design,
precisely what allows LBS to deliver flexible, personalised experi-
ences across diverse user needs. However, this flexibility comes at
a cost to software quality assurance. Traditional testing approaches
assume a well-defined action space and predictable program be-
haviour, where a correct output can be specified in advance and
verified by comparison [ 18,23]. For LBS, neither assumption holds,
as what constitutes a correct response is determined not only by
functional logic but also by the user’s intent, software requirements,
and user-specific constraints such as health profiles and usage con-
texts, a combination that varies with each user and context.
These characteristics have two practical consequences for ac-
ceptance testing. First, it is impossible to construct test oracles by
requirements coverage alone, since a single user intention may
correspond to infinitely many valid LLM responses, making no
arXiv:2608.12970v1  [cs.SE]  13 Aug 2026

ASE ’26, October 12–16, 2026, Munich, Germany Wang et al.
finite enumeration of expected outputs a reliable oracle. Second,
even when oracles can be generated, their correctness cannot be
assumed, as oracle generation may retrieve irrelevant constraints,
misinterpret user intent, or hallucinate requirements not grounded
in actual software specifications. Together, these consequences re-
frame acceptance testing for LBS as arequirements-grounded intent
reconstruction problem. User-specific constraints, such as health
conditions or usage contexts, are part of the software requirements
in LBS and must be treated as first-class inputs to oracle generation,
not afterthoughts. This motivates two interconnected objectives:
•obj-i Intention Interpretation.The system must infer a user’s
likely intent from an input query and profile, and reconstruct the
behavioural expectations the system should have satisfied, sub-
ject to software constraints (requirements, persona, and domain
knowledge), producing executable pass/fail oracle criteria.
•obj-ii Verdict Reliability.Given a generated oracle, the verdict
mechanism must reliably determine whether actual system be-
haviour satisfies the criteria.
We present a framework that, rather than enumerating expected
outputs, generates executable test oracles by reasoning over soft-
ware constraints and calibrates the reliability of verdicts through
a statistically calibrated LLM cascade that simultaneously filters
low-quality oracles and reduces evaluation cost. To address obj-i ,
we introduceRequirements-Augmented Generation (REAG).
Existing RAG approaches retrieve general knowledge to produce
better answers to questions. This fails for acceptance testing be-
cause the goal is not to answer a question but to specify what
the system should have done given a particular user’s constraints.
REAG addresses this by retrieving software artifacts, specifically
requirements, user profiles, and software docs, to reconstruct the
behavioural expectations the system should have satisfied, and en-
coding those expectations as executable pass/fail oracle criteria. The
individual components we employ, namely ICRALM scoring [ 43],
adaptive-k selection [ 49], and self-reasoning trajectories [ 60], are
established techniques in the NLP domain; the contribution lies
in reorienting their collective purpose from knowledge augmen-
tation toward behavioural specification, a goal that none of them
individually address. To address obj-ii , we introduceconfidence-
calibrated cascade judgment. LLM-as-a-judge have been applied
to model evaluation and data annotation, but not to acceptance
testing, where the judge must serve two simultaneous purposes:
assessing whether actual system behaviour satisfies oracle criteria,
and detecting whether the oracle itself is reliable enough to trust.
Our cascade addresses both by using simulated expert agreement
to estimate verdict confidence, accepting high-confidence verdicts,
escalating uncertain cases to stronger judges, and abstaining when
no judge reaches sufficient confidence. The abstention mechanism
acts as a quality gate that prevents unreliable oracle verdicts from
propagating. Confidence thresholds are calibrated using conformal
risk control [ 4], providing finite-sample reliability guarantees on
accepted verdicts, which, to our knowledge, is the first application
of this statistical framework to acceptance testing.
1.1 Industrial Context and Motivation
We evaluate our framework on a nutrition scoring and advisory
mobile application developed by Yiwuyishi Intelligent Technol-
ogy (Nantong) Co., Ltd., a software company with ≈20 developersspecialising in IT consulting and software outsourcing. The applica-
tion leverages LLM capabilities to provide personalised nutritional
guidance by analysing food packaging images and users’ health pro-
files. The system serves diverse user segments identified through
user research, including weight management, athletic performance,
and medical conditions such as diabetes, hypertension, and kid-
ney disease. Its core functionality involves OCR-based extraction
of nutritional information from food packaging images, retrieval
of relevant dietary guidelines from domain knowledge bases, and
LLM-powered generation of personalised nutritional assessments.
Unlike traditional software, this system must dynamically infer
implicit user needs, such as blood glucose management for diabetic
users or sodium restriction for those with hypertension, even when
these concerns are not explicitly stated in the query. This makes it
a representative and challenging case for LBS acceptance testing.
Figure 1 contrasts traditional acceptance testing with our frame-
work on the nutrition app NutriScore. The traditional approach
defines Given-When-Then acceptance criteria that verify the sys-
tem assembles a prompt package and returns a response. While
this confirms the service layer behaves correctly, it remains en-
tirely agnostic to whether the LLM’s response is appropriate for
the user’s actual health context. In this example, a diabetic user
with hypertension uploads an image of coconut water. The tradi-
tional test passes because the system produces a response. However,
the actual response provides only generic nutritional benefits and
ignores user-specific health concerns. This failure is invisible to
traditional testing because the criteria operate at the service layer,
not at the semantic layer of the LLM’s response.
Our framework addresses this gap by treating oracle generation
as the reconstruction of intent from software constraints. REAG re-
trieves the user’s health conditions (Type 2 Diabetes, Hypertension),
dietary goals (weight loss), and medication constraints (Metformin,
Lisinopril) from the software requirements and persona definitions,
then reasons over these constraints to generate a test oracle with
explicit pass/fail criteria, such as flagging high sugar content for
blood glucose risk and warning about sodium interactions with
hypertension medication. The cascade then evaluates the actual re-
sponse against this oracle, and correctly identifies a test failure that
traditional methods would miss entirely. Importantly, if REAG had
generated a low-quality oracle, for instance, retrieving irrelevant
constraints about athletic performance, the cascade would abstain
rather than propagate an unreliable verdict.
1.2 Contributions
•Requirements-Augmented Generation (REAG) for oracle
generation. We reframe test oracle generation as a requirements-
grounded intent reconstruction problem. Unlike prior RAG work
that retrieves knowledge to answer questions, REAG retrieves
software constraints, including requirements, personas, and do-
main knowledge, to reconstruct what the system should have
understood within a user’s intent, producing executable pass/fail
oracle criteria. This is the first use of RAG as a behavioural spec-
ification mechanism for acceptance testing, addressingobj-i.
•Confidence-calibrated cascade judgment as a quality gate.
We introduce a cascade of LLM judges that serves a dual pur-
pose absent from prior LLM-as-a-judge work: validating system

Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software ASE ’26, October 12–16, 2026, Munich, Germany
Figure 1: Acceptance Testing for an LBS (Our case study).
behaviour against oracle criteria while screening out unreliable
oracles through confidence-based abstention. Thresholds are
calibrated using conformal risk control, providing empirical reli-
ability guarantees for accepted verdicts and addressing obj-ii .
•Industrial evaluation. We conduct a real case study with indus-
try practitioners on a production nutrition advisory application
(346 test scenarios, 46 user profiles), demonstrating 98.8% accu-
racy, an oracle-quality improvement from 3.91 to 4.30 through
cascade filtering, and a 31.7% cost reduction relative to a single
judge baseline. We note that such industry studies involving real
practitioners and proprietary software artifacts in requirements-
driven testing are scarce in SE research. To show the generality,
we further discuss in Section 5.2 Background
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) augments text generation
with real-time retrieval from external knowledge sources, reduc-
ing hallucinations and outdated information [ 32,64]. However,
traditional RAG systems rely on fixed top-k retrieval heuristics
that retrieve predetermined numbers of documents regardless of
query characteristics, often degrading performance with irrelevant
context [ 46]. Adaptive top-k methods address this by dynamically
determining optimal document counts based on query specificity
and score distributions, including similarity gap analysis [ 49,60],
threshold-based methods [14, 35, 67], and ML techniques [37, 46].
Recent work has further extended RAG with reasoning-aware
retrieval strategies. Self-Reasoning [ 60] enhances retrieved docu-
ments with explicit reasoning trajectories, improving reliability and
traceability. RATT [ 65] and CoT-RAG [ 34] adopt chain-of-thought
prompting within the retrieval loop to improve evidence grounding
and response faithfulness. These reasoning-aware approaches share
the spirit of our self-reasoning mechanism in REAG, but target gen-
eral knowledge retrieval rather than software constraint retrieval
for behavioural specification. In this work, we use three tuning-
free retrieval methods specifically adapted for querying software
artifacts and reconstructing user intent. Ram et al. [ 43] introduce
ICRALM, which estimates semantic correlations between queries
and documents via LLM log probabilities without fine-tuning. Xia et
al. [60] apply self-reasoning to enhance retrieved documents with
reasoning trajectories. The key distinction of REAG is that retrieval
targets software artefacts, e.g., requirements, personas, and domain
knowledge, rather than general knowledge, and the output is an
executable behavioural specification rather than generated answer.
2.2 Automated Acceptance Testing
Automated acceptance testing determines whether software sat-
isfies acceptance criteria and meets end-user requirements from
the customer’s perspective [ 5,12,38]. Various methodologies have
emerged to reduce manual effort, including Behavior-Driven Devel-
opment (BDD) [ 19,42], Model-Based Testing (MBT) [ 38,48], and
Test-Driven Development [ 51]. However, automating the genera-
tion of acceptance testing artifacts from requirements remains fun-
damentally limited. A recent survey by Wang et al. [ 54] catalogues
only 156 primary studies between 1990 and 2024, with very few
supporting fully automated acceptance testing. Existing approaches
face various challenges. The scalability issue in requirements trace-
ability is identified in Corriveau and Shi [ 17] when linking hundreds
of requirements to thousands of test cases. An abstraction gap exists
between coarse-grained requirements and detailed tests [ 5]. Practi-
tioners consistently struggle with test-first practices in real-world
contexts [ 51]. Recent AI-powered approaches [ 22,28,56] represent
promising directions, with Wang et al. [ 53] leveraging RAG to gen-
erate acceptance criteria from multi-modal requirements. But these
works still focus on traditional software rather than LBS.
Beyond these, recent works have proposed comprehensive datas-
ets or benchmarks for agents or LLMs, including hallucination [ 8],
safety [ 2], or even concrete applications [ 25,55]. These studies
evaluate LLM capabilities in isolation rather than LLM behavior
within deployed, user-centric software systems. Instead, we target

ASE ’26, October 12–16, 2026, Munich, Germany Wang et al.
this gap by treating oracle generation as requirements-grounded
intent reconstruction, enabling acceptance testing where correct
behavior is defined by user context and software constraints rather
than functional logic alone.
2.3 Trustworthy LLMs in Judging
LLM-as-a-Judge is an emerging paradigm that extends large lan-
guage models to serve as evaluators for open-ended and human-
preference-aligned tasks [ 33,68], with applications in model evalu-
ation, data annotation, and reward modeling [ 20,31]. While LLMs
show promise as scalable alternatives to human annotators [ 30,57],
their stochastic nature introduces systematic biases, inconsistent
judgments, and vulnerability to adversarial inputs [ 50,69]. Recent
empirical evidence further shows that LLM judges from related
model families exhibit significantly correlated errors, agreeing on
wrong answers in a substantial fraction of cases [ 50]. Our cas-
cade mitigates this by selecting judges from distinct model families
(Gemini and GPT), as discussed in Section 4.
Conformal risk control[ 4] offers a principled solution by pro-
viding formal guarantees that expected risk remains below a user-
specified threshold [ 47]. Given a calibration set, the method identi-
fies a confidence threshold that ensures the true disagreement rate
stays below a tolerance level𝛼with high probability(1−𝛿):
𝑃(𝑓𝐿𝑀(𝑥)=𝑦ℎ𝑢𝑚𝑎𝑛|𝑐𝐿𝑀(𝑥)≥ ˆ𝜆)≥1−𝛼(1)
where𝑓𝐿𝑀denotes the model’s prediction, 𝑦ℎ𝑢𝑚𝑎𝑛 represents the
ground truth label, 𝑐𝐿𝑀is a confidence measure, and ˆ𝜆is the cal-
ibrated threshold. The threshold is determined by computing an
empirical risk ˆ𝑅(𝜆) on calibration data, deriving an exact upper
confidence bound ˆ𝑅+(𝜆)using binomial statistics, and selecting
ˆ𝜆=inf{𝜆 :ˆ𝑅+(𝜆′) ≤𝛼for all𝜆′≥𝜆} through fixed sequence
testing. This approach provides exact, finite-sample guarantees for
a single calibrated judge. In our cascade setting, we apply confor-
mal risk control independently per judge tier following Jung et
al. [27], providing per-tier empirical reliability bounds on accepted
verdicts. Three practical constraints apply to this setting. First, the
calibration guarantees require that calibration and deployment
data follow the same distribution, a condition that may be partially
stressed by our synthetic persona generation [ 9]. Second, per-tier
bounds do not compose formally at the system level when adaptive
routing is applied across stages [ 61]. Third, selective temperature
scaling applied post-hoc further relaxes the formal guarantees for
judges with insufficient calibration samples. We address these con-
straints empirically in Section 4 by reporting coverage-accuracy
trade-offs across 𝛼settings, and note that recent follow-up work on
conformally-calibrated LLM judges [ 7] confirms that the marginal
guarantees hold well in practice under the i.i.d. assumption.
3 Methodology
The overview of this framework is exhibited in Figure 2. The
framework operates in three stages: (1) user intention elicitation,
(2) requirements-augmented oracle generation, and (3) reliability-
calibrated verdict generation via cascade judgment. For the formu-
las, we provide a notation summary in Table 1.Table 1: Notation Summary
Symbol Description
IUser intention
PSet of predefined persona categories
QSet of real user queries
𝑅,𝐾,𝑝Retrieved requirements, background knowledge, persona description
I|⟨𝑅,𝐾,𝑝⟩Interpreted intention with software context
𝑀𝑖 Judge at tier𝑖in the cascade
𝑓𝑀𝑖(𝑥)Verdict produced by judge𝑀 𝑖
𝑐𝑀𝑖(𝑥)Confidence score of judge𝑀 𝑖on instance𝑥
ˆ𝜆𝑖 Calibrated confidence threshold for judge𝑀 𝑖
𝐷cal Calibration dataset of oracle-output pairs
ˆ𝑅(𝜆)Empirical disagreement risk at threshold𝜆
ˆ𝑅+(𝜆)Upper confidence bound on empirical risk
𝛼,𝛿Risk tolerance and confidence parameters
𝑁,𝐾Number of simulated annotators and in-context examples
3.1 User Intention Elicitation
The defining characteristic of LBS is its user-centric behaviour:
“Should I eat this food?” requires different responses for diabetic,
athletic, or pregnant users. This user-dependence is not incidental
to the nutrition domain but fundamental to any LBS whose core
function is to deliver personalised responses. In a legal advisory
system, the same query about contract terms requires different
responses for a small business owner versus a multinational cor-
poration. In a fitness coaching system, the same exercise query
requires different responses for a post-surgery patient versus a
competitive athlete. Consequently, persona 𝑝is a first-class input to
acceptance testing in any LBS, not a domain-specific design choice:
(i) all queries must be associated with personas to determine cor-
rect system behavior, and (ii) personas affect query interpretation,
e.g., responses for the same query as “Can I have this energy bar?”
would be very different for a diabetic non-athlete versus an athlete.
As LBS operates in a vast, practically non-enumerable space,
comprehensive testing requires diverse persona-query combina-
tions. We collect real user queries and define persona categories
from actual software requirements. However, within each persona
category, users exhibit significant variation, e.g., diabetic users
differ in medications, comorbidities, and dietary contexts, all af-
fecting correct system behaviour. Manually collecting profiles cov-
ering all such variations is prohibitively expensive. We therefore
employ a simulated user agent to systematically expand user pro-
files within predefined persona categories. The agent Auseruses
few-shot prompting with real persona categories and queries as
examples, generating enriched user situations that capture realistic
within-category variation through three mechanisms: grounding in
real user queries, constraining to authentic persona categories from
software requirements, and learning realistic elaboration patterns
from few-shot examples. Specifically, a user intention Iconsists of
a user query and persona. We generate intentions through:
(1)Given predefined persona categories P(from software require-
ments) and real user queries Q, we randomly sample {𝑝𝑖,...,𝑝𝑗}
∈Pand{𝑞 𝑖,...,𝑞𝑗}∈Q.
(2)We assign a multi-modal LLM as agent Auserwith few-shot
prompting using the sampled personas and queries as examples.
(3)AgentAusergenerates enriched user situations within sampled
persona categories, producing intentions Icombining real
queries with detailed persona profiles.

Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software ASE ’26, October 12–16, 2026, Munich, Germany
Actual Behavior
Softwar e Requir ements
User P ersona
Back ground Knowledge
          LLM-based Software
LLM  (Comparison) small
Confidence Prediction
Check if LLM  is confident about the results small
LLM  (Comparison) Medium
Confidence Prediction
Check if LLM  is confident about the results Medium
LLM  (Comparison) Large
Confidence Prediction
Check if LLM  is confident about the results LargeEscalate
Escalate
Test
Verdict
#2. Test Oracle Generation
 #3. Trustworthy Test Verdict Generation
 #1. Intention Elicitation
Intention Interpretion with       Self-Reasoning
        User
        Intention
        User
        Intention
Information RetrievalRetrieved
Docs
Testing Generation
Each
Doc
Docs with
Interpretion
Expected Behavior
(Test Oracle) Simulated User
(User Agent)
Real UserRele vance:
True
Rele vant Reason:
The provided doc aligns 
user’s intention from ...Cite Content:
True
Reason t o cite:
This sentence hints how
this system works,  ...Analysis:
The doc directly supp-
orts the user's need to
analyze an image-bas-
ed query 
You are analyzing a software system to infer its expected behavior. Below is the structured
context about the software:
 - User Input (current query): {user_query}
 - User Situation: {user_situation}
 - User Persona: {persona_items}
 - Relevant User Story: {user_story_items}
 - Domain Knowledge: {domain_knowledge_items}
 - Background Information: {background_items}
 - Non-Functional Requirements: {nfr_items}
 Using all the above information, describe the expected behavior of the software
 -- how the system should respond or act according to the user's input and the provided
context. The output should follow the JSON format as:
{  "TestObjective": "Concise description of the main goal or  purpose of the test. It should
define the functionality,  feature, or system behavior being validated.",  ExpectedResults-
Description": "Narrative explanation of what  the correct output or behavior should be. This
specifies  what the system should produce or how it should behave given  the user's context
and constraints.",  "PassFailCriteria": [  "Explicit criteria that determine whether the test
passes  or fails.",  "Each rule compares actual system output with expected  values or
conditions.",  "May include qualitative or quantitative conditions."  ] }
{"TestObjective": "Verify the system correctly processes food  packaging image for Emotional Eater persona: OCR
extraction,  sugar/sodium risk identification, score computation, and  mood-stabilizing recommendations.",
 
"ExpectedResultsDescription": "System performs OCR to extract  ingredients (vegetables, soy sauce, sugar,
vinegar) and  nutritional data. It identifies 'sugar' as added sugar and  'soy sauce' as high-sodium contributor.
Displays comprehensive  scorecard (total + sub-scores) within 3 seconds with red  'High Sugar' badge if added
sugar >= 10g/100g. Provides  persona-tailored recommendations: low-sugar alternatives,  reduced-sodium
options, mood-stabilizing nutrients, and  mindful eating tips.",  
 
"PassFailCriteria": [ 
"PASS if OCR extracts ingredients containing sugar and soy  sauce with numeric nutrition data or added-sugar
flag.",
"PASS if 'High Sugar' badge displayed when added sugar >=  10g/100g, or 'Added sugar present' warning if
numeric  unavailable.",  "PASS if sodium ingredients highlighted with advisory and  reduced-sodium alternatives.",
"PASS if scorecard rendered within 3 seconds under normal  conditions.",
"PASS if persona-specific recommendations include low-sugar  guidance and mood-stabilizing tips for Emotional
Eater.",
"FAIL if sugar not flagged, or High Sugar badge missing when  sugar >= 10g/100g, or no sodium warnings, or
scorecard  exceeds 3 seconds, or no persona-specific advice.",
"FAIL if unhandled error instead of fallback instructions  when OCR fails."  ] }Prompt Template Oracle Example
Figure 2: Overview of Acceptance Testing Framework
The generated intentions are combined with real user intentions to
ensure comprehensive coverage while maintaining realism.
3.2 Intent Interpretation with Software Context
Our framework interprets the user’s intent based on the software
context by retrieving relevant software requirements. In practice,
user queries express intentions implicitly: a user asking “Help me
find a way to the supermarket” also implicitly requires the nearest
location, shortest route, and profile-appropriate transport mode.
These implicit requirements must be surfaced for oracle generation.
We realise this through three steps: i) ranking relevant software
artefacts based on user intention Iusing ICRALM [ 43], ii) applying
an adaptive top- 𝑘mechanism to dynamically select relevant doc-
uments without manual configuration of 𝑘[49], and iii) applying
self-reasoning to enhance retrieved artefacts with interpretation
trajectories [60].
3.2.1 Relevant Doc Ranking.ICRALM estimates semantic correla-
tion between a query and document candidates by computing the
average log probability on query tokens:
ICRALM I|Task,Doc=1
|I||I|∑︁
𝑡=1
log𝑝𝜃 𝑖𝑡|Task,𝑖 <𝑡,Doc(2)
whereIrefers to the user’s intention, Task is the retrieval prompt,
and Doc is the document candidate. Each document is scored and
ranked accordingly.3.2.2 Adaptive Top-k Selection.Fixed top- 𝑘retrieval risks omitting
critical evidence or overwhelming the model with irrelevant con-
text [ 49]. We adopt Adaptive- 𝑘retrieval [ 49], which dynamically
determines the optimal number of documents by: (1) computing
ICRALM similarity scores, (2) sorting in descending order, (3) com-
puting discrete differences Gap𝑖=ICRALM 𝑖−ICRALM 𝑖+1, and (4)
selecting𝑘=arg max𝑘Gap(𝑘) , where the largest similarity drop
marks the boundary between relevant and irrelevant documents.
We retrieve the top- 𝑘documents plus a small buffer, which is par-
ticularly suitable for requirements interpretation where relevant
context varies significantly across query complexities.
3.2.3 Self-Reasoning-based Intention Interpretation.Traditional R-
AG provides retrieved documents without explanation, which may
introduce unreliable or untraceable responses [ 60]. We address this
through self-reasoning in three steps: i)relevance checkingthat in-
structs the LLM to judge and explain the relevance of each retrieved
document to the user’s intention, ii)evidence checkingthat directs
the LLM to cite specific sentences from retrieved documents and
explain why they support the interpretation, and iii)application
analysisthat synthesizes the previous steps into a concise final
interpretation. The retrieved software context consists of require-
ments𝑅, background knowledge 𝐾, and persona description 𝑝. We
formulate the interpretation as:
IReasoning−−−−−−−→I|⟨𝑅,𝐾,𝑝⟩(3)
whereI|⟨𝑅,𝐾,𝑝⟩ represents the interpreted intention with the
retrieved software context.

ASE ’26, October 12–16, 2026, Munich, Germany Wang et al.
3.3 Test Oracle Generation
After obtaining the interpreted intention in Eq. 3, we generate the
expected behaviour of the software as a test oracle. Test cases and
acceptance criteria face fundamental limitations for LBS: a single
interpreted intention I|⟨𝑅,𝐾,𝑝⟩ may correspond to infinitely many
valid LLM responses, making exhaustive enumeration infeasible
and exact-matching brittle. Test oracles instead encode executable
behavioural properties that validate unboundedly many stochastic
responses [ 39,66], making them the appropriate representation for
LBS acceptance testing.
We employ an LLM to generate test oracles from a predefined
prompt template, as shown in Figure 2. The oracle specifies a
TestObjective, ExpectedResultsDescription, and explicit PassFail-
Criteria that compare actual system output against expected be-
haviour. However, test oracles are notoriously difficult to obtain
correctly [ 24,63,66]: LLM hallucinations and interpretation errors
from previous steps may produce oracles with incorrect imple-
mentations or surface-level descriptions [ 10,21]. This calls for an
explicit reliability assessment of the generated oracles.
3.4 Calibrated Test Verdicts Generation
The cascade uses three judges: Gemini-2.5-Flash-Lite, GPT-4.1-mini,
and Gemini-2.5-Flash, selected from different model families to re-
duce correlated errors. Recent empirical evidence shows that judges
from the same model family tend to inherit shared biases and ex-
hibit correlated errors, agreeing on wrong answers in a substantial
fraction of cases [ 29,50]. Selecting judges from distinct families,
specifically Gemini and GPT in our cascade, reduces but does not
fully eliminate this risk, as acknowledged in Section 4.
3.4.1 Simulated Annotators for Confidence Estimation.Given 𝐾
examples of verdicts from 𝑁domain experts, we prompt judge 𝑀𝑖
to produce verdict 𝑉and explanation 𝐸, where𝑓𝑀𝑖(𝑥)=⟨𝑉,𝐸⟩ ,
𝑉∈{YES,NO} . We estimate confidence 𝑐𝑀𝑖(𝑥)through simulated
annotators by creating 𝑁different evaluation perspectives, where
each annotator 𝑗is conditioned on 𝐾in-context examples repre-
senting the𝑗-th expert’s judgment patterns:
𝑐𝑀𝑖(𝑥)=max
𝑦1
𝑁𝑁∑︁
𝑗=1𝑝𝑀𝑖(𝑦|𝑥;(𝑥 1,𝑗,𝑦1,𝑗),...,(𝑥𝐾,𝑗,𝑦𝐾,𝑗))(4)
where𝑦∈{YES,NO} . High confidence ( 𝑐𝑀𝑖(𝑥)≈ 1.0) indicates
simulated experts converge on the verdict, suggesting reliable judg-
ment and well-specified oracle criteria. Low confidence ( 𝑐𝑀𝑖(𝑥)≈
0.5) indicates oracle flaws or genuinely difficult conformance judg-
ments. The cascade operates as follows:
(1) Load the lowest-cost judge as the initial judge𝑀 0.
(2)For each PassFailCriterion in the oracle, produce verdict 𝑓𝑀𝑖(𝑥)
with confidence𝑐 𝑀𝑖(𝑥).
(3)If𝑐𝑀𝑖(𝑥) ≥ ˆ𝜆𝑖, accept the verdict and proceed to the next
criterion. If𝑐 𝑀𝑖(𝑥)< ˆ𝜆𝑖, escalate to the next judge𝑀 𝑖+1.
(4)If no judge produces sufficient confidence, revoke the oracle
with statusabstainfor manual expert review.
Abstained cases are collected for human inspection, who can vali-
date oracles or identify defects for iterative pipeline improvement,ensuring only high-confidence verdicts are accepted. In our im-
plementation, we set 𝑁= 5simulated annotators and 𝐾= 3in-
context examples per annotator. The 𝐾examples for each annotator
are sampled from the calibration set𝐷 calusing stratified selection
across verdict labels and persona categories, ensuring that each
simulated annotator reflects a distinct judgment perspective rather
than a homogeneous evaluation pattern. Confidence scores 𝑐𝑀𝑖(𝑥)
are obtained by prompting 𝑀𝑖with each annotator’s demonstra-
tion set and averaging the resulting label probabilities, following
the procedure of Jung et al. [ 27]. When logit-level probabilities
are unavailable from black-box APIs, we approximate 𝑝𝑀𝑖(𝑦|𝑥)
by sampling 𝑀𝑖with temperature 𝑇= 0.7across𝑁independent
runs and computing the empirical label frequency, consistent with
established practice in conformal LLM evaluation [41].
3.4.2 Expert-Calibrated Cascaded Evaluation.Thresholds ˆ𝜆𝑖are
calibrated using domain expert annotations based on conformal risk
control [ 4]. We construct a calibration dataset 𝐷calof oracle criteria
paired with actual system outputs, annotated by software experts
with knowledge of system requirements and domain constraints.
Each instance(𝑥,𝑦 human)∈𝐷 calcontains: (i) a test oracle, (ii) actual
system output, and (iii) expert verdict 𝑦human∈ {YES,NO} . For
each judge𝑀𝑖, we calibrate ˆ𝜆𝑖using fixed-sequence testing [ 11] as:
(1)Compute verdict 𝑓𝑀𝑖(𝑥)& confidence 𝑐𝑀𝑖(𝑥)for each(𝑥,𝑦 human)
∈𝐷 cal.
(2)For candidate thresholds 𝜆from high (0.999) to low, compute
empirical disagreement risk:
ˆ𝑅(𝜆)=1
𝑛(𝜆)∑︁
(𝑥,𝑦 human)∈𝐷 cal
⊮h
𝑓𝑀𝑖(𝑥)≠𝑦 human∧𝑐𝑀𝑖(𝑥)≥𝜆i
(5)
where𝑛(𝜆)=Í⊮{𝑐𝑀𝑖(𝑥)≥𝜆} counts instances above thresh-
old𝜆.
(3) Compute the exact(1−𝛿)upper confidence bound:
ˆ𝑅+(𝜆)=supn
𝑅:𝑃
Bin 𝑛(𝜆),𝑅≤
𝑛(𝜆) ˆ𝑅(𝜆)
≥𝛿o
(6)
(4)Select the threshold maximising coverage while bounding risk
below𝛼:
ˆ𝜆𝑖=inf{𝜆: ˆ𝑅+(𝜆′)≤𝛼for all𝜆′≥𝜆}(7)
This guarantees that verdicts accepted by the cascade align with
expert judgment at the specified confidence level per Eq. 1, with
probability at least1−𝛿.
In practice, calibrated thresholds may result in imbalanced cas-
cade compositions where certain judges receive insufficient sam-
ples (e.g.,𝑛(ˆ𝜆𝑖)<𝜏 ). We apply selective temperature scaling:
for each such judge 𝑀𝑖, we scale confidence scores as 𝑐′
𝑀𝑖(𝑥)=
𝜎(logit(𝑐𝑀𝑖(𝑥))/𝑇𝑖)where𝑇𝑖>1, then re-calibrate ˆ𝜆𝑖until all
judges handle at least 𝜏samples (typically 𝜏∈[ 5,10]), preserving
original calibration quality for well-performing judges at 𝑇𝑖=1.0.
4 Evaluation and Case Study
4.1 Research Questions
We conduct this case study to address three research questions
mapping to the two test objectives defined in Section 3:

Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software ASE ’26, October 12–16, 2026, Munich, Germany
Table 2: Data Categories in Nutrition Advisory Application
Data Category Collected Data Collection Method Analysis Type Sample
D1. Software Artifacts Requirements docs, domain docs,
and persona definitionsDocument Analysis NA 300+ artifacts
D2. Software Behaviours OCR outputs, parsed data, scores,
recommendations, logsSystem Execution NA 346 scenarios
D3. Oracle Evaluation Multi-perspective experts’ evalua-
tionInterviews & Forms Quantitative (Likert Scale) 100×4 scores
D4. Verdict Annotation Test verdicts annotation from ex-
pertsInterviews & Forms Quantitative (Agreement) 246 verdicts for calibration & 100
for evaluation
D5. Practitioner Interview Qualitative discussion about the
generated outputsInterview Qualitative (Open Discussion) 30-min interview transcript
D6. Extensive Evaluation Selective agreement, selective con-
fidence, cost comparisonAutomated Evaluation Quantitative (Metrics) Calibration Acc. on 246 verdicts
& evaluation on 100 verdicts
•RQ1 ( obj-i ):How effectively does REAG interpret user intent
within software constraints to generate test oracles?This research
question evaluates our framework’s ability to achieve obj-i ,
where user intention Ishould remain consistent under both
expected and actual software contexts. The reasoning must
correctly interpret what users want, understand system limita-
tions, and reason about unstated expectations. We define four
evaluation perspectives following prior work [ 6,53]: Relevance,
Coverage, Correctness, and Coherence.
•RQ2 ( obj-ii ):How accurately does the cascade judge determine
if system behaviour aligns with interpreted user intent?This
research question evaluates obj-ii , assessing whether the ver-
dict mechanism reliably determines if actual system behaviour
satisfies the criteria, while filtering unreliable oracles through
confidence-based abstention. We aim to i) screen out cases
where oracle interpretation may have failed, and ii) provide
statistical guarantees on the reliability of accepted verdicts.
•RQ3:What is the cost-efficiency of cascaded evaluation for prac-
tical deployment?We quantify the efficiency of our system by
comparing it against the single highest-capability LLM baseline.
4.2 Case Study Design and Protocols
Our evaluation is structured around three research questions, with
metrics tailored to assess oracle-generation quality (RQ1), verdict
selective reliability (RQ2), and cost-efficiency (RQ3). Following best
practices in empirical software engineering research [ 45,52], we
employ separate calibration (246 items) and evaluation (100 items)
datasets to ensure reliable assessment and prevent overfitting.
4.2.1 Wohlin’s Essential Aspects.We conduct an embedded single-
case study [ 62] on a production nutrition advisory application,
following guidelines by Verner et al. [ 52] and Runeson et al. [ 45].
Our study design explicitly addresses Wohlin’s five essential as-
pects [ 58].(1) Empirical Observation:We conduct systematic
data collection across three phases, yielding six complementary
data categories (Table 2): software artifacts (300+ artifacts), software
behaviours (346 execution traces), oracle evaluations (400 Likert
ratings), verdict annotations (346 expert judgments), practitioner
interviews (6 transcripts), and automated metrics.(2) Contempo-
rary Events:The application is actively under beta testing (30–50
testers) with all data collection occurring during the study period(3)
Real-World Context:We study an actual mobile application with
real users, authentic system constraints (API rate limits, credentialpolicies), and 46 user personas identified through user research.(4)
Multiple Data Collection Methods:Our triangulation strategy
employs document analysis (D1), system execution (D2), Likert
evaluation (D3), expert annotation (D4), interviews (D5), and auto-
mated metrics (D6), enabling cross-validation where oracle quality
issues (D3) can be traced to retrieval failures (D5) and reflected in
low verdict confidence (D6).(5) Unclear Phenomenon-Context
Boundary:Framework performance is entangled with application
context in ways that cannot be isolated experimentally.
4.2.2 Case Study Protocols.Annotator Recruit Protocols: All
annotators are recruited from the development team of the target
software, including six software developers with 3–8 years of experi-
ence handling different system functions, ensuring diverse priorities
in evaluation. While recruiting from the development team intro-
duces potential familiarity bias, these annotators possess privileged
knowledge of system requirements and domain constraints that ex-
ternal annotators would lack, making their judgments more reliable
for evaluating requirement-grounded oracle correctness [ 45]. Also,
these developers were not privy to any technical implementation
details of our approach to avoid biasing the annotation process.
Nonetheless, we acknowledge this in Section 5.1.Annotation Pro-
tocols: We employ two distinct expert annotation sets to establish
ground truth. The calibration set uses 246 randomly selected oracle-
output pairs, where three experts independently judge conformance
(YES/NO) with explanations. Disagreements are resolved through
structured post-discussion sessions. The evaluation set additionally
uses Likert-scale evaluation collected through interviews with three
annotators. Expert annotations reveal a Fleiss’ 𝜅of 0.76 (substantial
agreement) across D3, D4, and D5, with near-unanimous agreement
after discussion. To limit leakage between the two sets, the calibra-
tion and evaluation sets were annotated in separate sessions, and
all calibration labels were frozen before any evaluation-set item
was scored or any threshold ˆ𝜆𝑖was fit.Interview Protocol: We
conduct semi-structured interviews with all recruited annotators
to gather qualitative insights on oracle quality, perceived verdict
trust, and framework usability. Each interview lasts 30–45 minutes
covering: (i) perceived oracle quality compared to manually-written
oracles, (ii) trust in automated verdicts, and (iii) barriers to adoption.
Interviews are transcribed into significant points.

ASE ’26, October 12–16, 2026, Munich, Germany Wang et al.
4.3 Evaluation Procedure and Metrics
4.3.1 RQ1: Oracle Quality Assessment Metrics.To evaluate how ef-
fectively REAG interprets user intent for acceptance testing within
software constraints ( obj-i ), we conducted a four-dimensional
expert evaluation following prior work [6, 53]:
(1)Relevanceassesses how well the generated oracle aligns with
the target software, user query, and situation.
(2)Coverageassesses whether the oracle covers the necessary
test objectives and the persona-based behavioural variations.
(3)Correctnessevaluates the accuracy and soundness of the gen-
erated oracle.
(4)Coherenceevaluates whether experts can follow the logic of
the oracle without additional explanation.
Experts rate oracles on a Likert scale questionnaire [ 26]1. We ad-
ditionally collect qualitative interview data to understand how
interpretation capabilities manifest in practice. To better illustrate
the overall score distribution, we also report the average of the
four dimension scores (e.g., average( 4,3,2,2)=2.75) in Figure 3.
This average is not an evaluation dimension but a visual reference
used in RQ2 to track how cascade filtering affects retained sample
quality across𝛼settings.
4.3.2 RQ2: Selective Reliability Metrics.By identifying unqualified
oracles from RQ1, we evaluate how accurately the cascade judge
eliminates them and determines system behaviour alignment with
user intent ( obj-ii ).Selective Agreementquantifies the fraction of
accepted verdicts that align with expert consensus. High selective
agreement (above 95%) indicates high conditional reliability on the
accepted set; it speaks to the quality of retained verdicts and not
to behaviour on abstained or out-of-distribution cases.Selection
Coveragequantifies the fraction of oracle criteria for which the
cascade produces a definitive verdict rather than abstaining. Higher
coverage (above 85%) means fewer cases require manual review.
Evaluator Compositionanalyses the distribution of verdicts pro-
duced by each judge tier. We evaluate all metrics across confidence
thresholds, plot coverage-accuracy trade-off curves to identify opti-
mal operating points, and compare against single-judge baselines.
4.3.3 RQ3: Cost-Efficiency Metrics.To evaluate practical deploy-
ment viability, we quantify cost-efficiency using Cost Per Perfor-
mance (CPP) [ 16], which measures economic efficiency by dividing
total computational cost by the percentage of selective agreement.
4.4 Deployment Details
We deploy our framework (implemented over three stages) across
a mixture of local and API-based environments, with the local
environment running on multiple NVIDIA A100-SMI-80GB GPUs.
(1)#1 User Intention Elicitation.To process multi-modal user
queries (food images in this application), we employ GPT-5-
nano with predefined user personas and nutrition lists in image
format. Temperature is set to 0.9 to ensure diversity in the
outputs, reflecting the broad variation in real user query styles
across persona categories.
(2)#2 Requirements-Augmented Generation (REAG).We con-
figure the generation in three sub-steps: i) software context
1From the definition of Likert scale, >= 3/5 are acceptable scores.retrieval with ICRALM [ 43], using GPT-oss-20B [ 3] as the re-
trieval kernel (replaceable with lighter alternatives such as
SentenceBERT [ 44]); ii) self-reasoning using GPT-5-nano; iii)
oracle generation using GPT-5-mini to ensure oracle quality.
(3)#3 Test Verdicts Generation.Three LLMs form the cascade:
Gemini-2.5-Flash-Lite as the lowest judge, GPT-4.1-mini as
the mid-tier judge, and Gemini-2.5-Flash as the highest judge.
Judge selection prioritises diversity across model families to re-
duce correlated errors [ 29,50], with tiers ordered by increasing
capability and API cost as detailed in RQ3.
4.5 RQ1 Results
RQ1 examines how effectively REAG interprets user intention in
the software context, w.r.t. obj-i . In Figure 3, we present expert
evaluation results across four dimensions. The average score for rel-
evance is 4.11/5, followed by coherence (4.02/5), coverage (3.99/5),
and correctness (3.51/5), with an overall average of 3.91/5, where
the average serves as a composite indicator of oracle reliability
rather than a single quality judgment, with dimension-level results
reported separately in Figure 3. Experts identified oracles with in-
correct implementations and surface-level descriptions, attributing
these to hallucinations and incorrect retrieval results.
The distribution across 100 data points reveals that 22 received
perfect scores (5/5), while 60 scored between 3.5 and 5.0. We report
two distinct denominators to avoid ambiguity. At the composite
level, 7 oracles scored below 3.0 on the four-dimensional average,
indicating issues spanning multiple dimensions simultaneously. At
the per-dimension level, we regard an oracle as unqualified if it
scores below 3 on at least one dimension, which holds for 18 unique
oracles; the per-dimension flag counts are 8 for Relevance, 8 for
Coverage, 18 for Correctness, and 4 for Coherence, and these sum
to more than 18 because a single oracle may be flagged on several
dimensions. Oracles scoring exactly 3 on a dimension are marginal
(18 for Relevance, 17 for Coverage, 33 for Correctness, 19 for Co-
herence). The 82% figure reported throughout is the complement
of the 18 unique unqualified oracles. The defects mainly come from
the Correctness dimension.
From interview data, developers highlighted three findings:
(1)In stage #1, the simulation agent yields diverse and contextually
rich user situations. For example, given “Coconut Water” and
an aging persona, it generates: “After a light afternoon walk,
the elderly user with T2DM and hypertension sits down for a
quick hydration break and chooses coconut water. Contains
natural sugars and provides little fiber or protein, so portion size
matters to avoid glucose spikes.” This captures edge cases and
enriches sparse queries with contextual intent. /commen◎-do◎s“It came up
with situations I didn’t even think about, like combining multiple
conditions together. That’s actually more useful than what we’d
write ourselves. ”
(2)In most cases, the self-reasoning mechanism retrieves relevant
requirements, but not consistently. Unqualified generations
mostly lead to incorrect retrieval, in which the system retrieves
irrelevant requirements or omits critical constraints, making
the user’s intention appear inconsistent between expected and
actual contexts. /commen◎-do◎s“When it works, it’s great. But sometimes

Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software ASE ’26, October 12–16, 2026, Munich, Germany
Figure 3: RQ1 - Frequency of Likert Scale ratings for each evaluation criterion by experts.
it just grabs the wrong section, like it’s looking at a completely
different feature, and then everything downstream is wrong. ”
(3)In edge cases, the generation is misled by explicit risks in
generated scenarios rather than focusing on core software
constraints, causing marginal generations that extend beyond
software-defined scope. This occurs when focusing on promi-
nent persona characteristics rather than the requirements de-
fined in the software artefacts. /commen◎-do◎s“It picked up on the environ-
mentally friendly angle from the persona and ran with it. That’s
not something we test for, it’s not in the requirements anywhere. ”
The 82% qualified or marginal rate indicates interpretation suc-
ceeds in the majority of cases. The remaining 18% are attributable
to incorrect retrieval and cases where the model follows prominent
scenario details outside the defined software requirements.
4.5.1 Failure Case Analysis.Among the 18 unqualified cases, two
distinct failure patterns account for the majority of low scores.
(1) REAG retrieves requirements from the wrong functional layer
entirely. Several cases involving health-sensitive personas, includ-
ing a night-shift worker uploading a seasoning ingredient list and
a hypertension patient scanning a high-sodium spice product, re-
ceived oracles whose pass/fail criteria verified service-layer (e.g.,
OCR) behaviours such as API response handling and OCR con-
fidence thresholds, while omitting the dietary safety constraints
central to the user’s health context. These scored 1/5 across all four
dimensions because they were irrelevant to the user’s intent and
did not meet any of the required health constraints. This failure
(stemming from technical RAG failure) mirrors the traditional ac-
ceptance criteria limitation illustrated in Figure 1, where the system
returned a response without assessing its semantic appropriateness.
(2) reflects scope drift (scores in the 2–3 range) rather than a com-
plete mismatch. REAG retrieves the correct feature area but gener-
ates oracles that enumerate too broad a set of system behaviours,
diluting the health-critical criteria with generic technical checks. A
diabetic user scanning a peanut and wheat sugar product received
an oracle that correctly identified sugar threshold rules but also
included barcode fallback procedures and carbohydrate normalisa-
tion checks, causing Coverage and Correctness to score 2 because
the persona-critical constraints were underspecified relative to the
surrounding content. A post-surgery recovery patient received an
oracle that addressed the right product category but expressed itscriteria at the implementation field level rather than in terms of
user-observable behaviour. In both patterns, the common thread is
that retrieval precision rather than generation quality is the primary
bottleneck for oracle correctness.
RQ1 Answer
REAG achieves an overall oracle quality of 3.91/5 across relevance (4.11),
coherence (4.02), coverage (3.99), and correctness (3.51), successfully
handling 82% of cases. The remaining 18% fail primarily because REAG
retrieves technical infrastructure requirements rather than the health
and persona constraints that should govern the oracle, confirming that
retrieval precision is the primary bottleneck for oracle quality.
4.6 RQ2 Results
RQ2 examines whether the cascade judge accurately determines if
system behaviour aligns with user intent, relating to obj-ii . The
cascade uses three judges: Gemini-2.5-Flash-Lite, GPT-4.1-mini, and
Gemini-2.5-Flash, selected from different model families to reduce
correlated errors and LLM blind spots [29, 50].
In Figure 4, the cascade achieves a similar curve to target hu-
man agreement when 𝛼∈( 0.18,0.40), consistently outperforms
when𝛼∈( 0.12,0.18), and overfits when 𝛼∈( 0,0.12). At𝛼=0.14,
the framework achieves 98.8% accuracy with 14% abstention. At
𝛼= 0.16, it maintains 90.9% accuracy with just 1% abstention.
The empirical human agreement curve closely tracks the target
across different 𝛼values, demonstrating that calibration effectively
controls the disagreement rate. Across all 𝛼settings, the empiri-
cal human agreement meets or exceeds the target threshold1 −𝛼,
confirming that the per-tier conformal calibration controls disagree-
ment risk as intended and validating the finite-sample reliability
guarantee in practice. At lower 𝛼values, the cascade relies increas-
ingly on higher-tier judges; at higher 𝛼values, the lowest-tier judge
handles nearly all cases. The sweet spot at 𝛼=0.14and𝛼=0.16
balances accuracy and coverage across multiple judges.
Figure 5 shows that setting 𝛼=0.13eliminates most unqualified
generations. The average score of kept samples increases from 3.93
at𝛼= 0.16to 4.30 at 𝛼= 0.12, confirming that lower 𝛼values
filter lower-quality oracles. However, from 𝛼= 0.13to0.12, the
method also eliminates qualified results, with coverage dropping
from 81% to 70%. This over-filtering likely occurs because judges

ASE ’26, October 12–16, 2026, Munich, Germany Wang et al.
Figure 4: RQ2 - Selection Coverage (R) & Selective Agreements (L) by 𝛼Settings (Default 𝛿=0.1, 88∗is the last observable point).
Figure 5: Sample Elimination Trend by𝛼Settings (Default𝛿=0.1)
assign high confidence to incorrect verdicts [ 50], and properly
calibrated thresholds prevent this from affecting the accepted set.
RQ2 Answer
Our cascade achieves optimal balance at 𝛼=0.14with 98.8% accuracy
and 86% selection coverage, and at 𝛼=0.16with 90.9% accuracy and
99% selection coverage, outperforming single-model judges (Gemini-2.5-
Flash: 90%, GPT-4.1-mini: 83%). Kept sample quality improves from 3.93
to 4.30 as𝛼decreases. Thresholds below 0.12 lead to over-elimination
due to LLM overconfidence.
4.7 RQ3 Results
Table 3: RQ3: Cost per Performance (CPP) Analysis.
Setting Performance Judge Distribution Total Cost CPP
(𝛼value) (Agreement %) (J1 / J2 / J3 %) (per 100 units) (Cost/Perf)
J2 Baseline 83.0% 0 / 100 / 0 300.0 3.61
J3 Baseline 90.0% 0 / 0 / 100 460.0 5.11
0.36 62.0% 99 / 1 / 0 102.0 1.65
0.18 83.0% 0 / 100 / 0 300.0 3.61
0.16 90.9% 0 / 82.8 / 17.2 327.5 3.60
0.14 98.8% 0 / 72.1 / 27.9 344.6 3.49
0.12 97.1% 0 / 0 / 100 460.0 4.74
In RQ3, we adopt Cost per Performance (CPP) [ 16] to quantify
cost-efficiency. The three judges correspond to three cost tiers: J1
(Gemini-2.5-Flash-Lite) at base cost (1x), J2 (GPT-4.1-mini) at 3.0x,
and J3 (Gemini-2.5-Flash) at 4.6x, based on official API pricing.Table 3 shows that the highest-performance setting ( 𝛼=0.14,
98.8% agreement) achieves a CPP of 3.49, which is 3.3% more cost-
efficient than GPT-4.1-mini alone (3.61 CPP) and 31.7% more cost-
efficient than Gemini-2.5-Flash alone (5.11 CPP), while delivering
superior performance. The high-coverage setting ( 𝛼= 0.16, 99%
selection coverage) achieves 90.9% agreement at CPP of 3.60, match-
ing the expensive J3 baseline performance at 29.5% lower cost.
Settings with high 𝛼(e.g., 0.36) are cheapest at 1.65 CPP but pro-
duce unacceptable performance (62.0% agreement) and fail to filter
unqualified generations. This confirms the cascade achieves a better
cost-performance balance than any single-model approach, making
large-scale acceptance testing practical in industrial deployment.
RQ3 Answer
Our cascade achieves superior cost-efficiency with CPP of 3.49 at 𝛼=
0.14(98.8% accuracy), being 3.3% more efficient than GPT-4.1-mini alone
and 31.7% more efficient than Gemini-2.5-Flash alone, showing that
cascaded evaluation reduces cost without sacrificing verdict quality in
industrial deployment.
5 Discussion
Extended from our single embedded case, we further discuss the
generality of our framework by separating our claims into three
tiers of decreasing generality, and then state concretely what port-
ing the framework to another LBS requires.

Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software ASE ’26, October 12–16, 2026, Munich, Germany
Tier 1: Mechanisms and settings are domain-independent.
Three findings follow from the architecture of the approach rather
than from nutrition. First, persona is a first-class oracle input for any
LBS whose core function is personalisation: wherever the correct
response is a function of who is asking, an oracle that ignores the
asker is underspecified, and the legal-advisory and fitness-coaching
cases in Section 3 instantiate the same structure. Second, retrieval
precision rather than generation quality is the dominant bottle-
neck for oracle correctness. This is a property of grounding oracles
in retrieved artefacts: a generator cannot specify a constraint it
was never shown, so oracle quality is bounded above by retrieval
quality in any instantiation of REAG. Our failure analysis supports
reading this structurally rather than domain-specifically, since both
observed failure modes, retrieval from the wrong functional layer
and scope drift, are artefact-organisation failures and should recur
wherever technical and user-facing requirements coexist in one
corpus. Third, the cascade’s cost advantage follows from monotone
price–capability tiering among commercial LLMs rather than from
any property of our particular judges: so long as cheaper mod-
els exist that are adequate for easy cases, routing by confidence
dominates always paying for the strongest judge.
Tier 2: Configurations are transferable after re-calibration.
The risk tolerance 𝛼, the per-tier thresholds ˆ𝜆𝑖, and the assignment
of specific models to tiers areoutputsof the calibration procedure,
not constants of the framework. A new deployment re-runs the
procedure of Section 3.4 against its own expert-labelled calibration
set and obtains its own thresholds; our 𝛼=0.14operating point
carries no authority elsewhere. This is precisely what makes the
framework portable while its numbers are not, and it also means the
conformal guarantee is re-established locally in each deployment
rather than inherited from our study.
Tier 3: Measurements that are specific to this case.The 3.91/5
and 4.30/5 oracle quality scores, the 98.8% selective agreement, and
the 31.7% cost reduction are joint properties of this application,
the quality of its requirements corpus, and this model lineup. They
should be read as evidence that the approach is viable in a pro-
duction setting, not as expected values for other systems. Because
REAG’s ceiling is set by the artefacts it retrieves over, we expect
domains with sparser, less structured, or less consistently main-
tained requirements to yield lower oracle quality, and domains with
mature, well-partitioned specifications to yield higher.
What porting the framework requires.Concretely, an adopter
needs four ingredients, none of them nutrition-specific: a machine-
readable artefact corpus covering requirements and domain knowl-
edge; a set of persona categories drawn from that corpus rather
than invented; a modest expert-labelled calibration set of oracle–
output pairs, for which our 246 items are an existence proof rather
than a lower bound; and at least two LLM judges at distinct price
points, preferably from different vendors. The three pipeline stages
themselves are artefact-agnostic. The domains where we would
expect the approach to degrade are those lacking the second or
third ingredient, that is, where user classes are undocumented or no
expert-labelling budget exists, since calibration is what converts an
uncalibrated confidence score into a reliability guarantee. Validat-
ing these expectations across domains remains future work, and we
scope our claim accordingly: we contribute a transferable methodtogether with one industrial demonstration of it, not evidence of
cross-domain performance.
5.1 Threats to Validity
We identify several threats to validity following Wohlin’s guide-
lines [ 58].Internal validityconcerns arise from four sources. An-
notation inconsistency is mitigated through structured calibration
sessions and consensus discussions, achieving substantial inter-
rater agreement (Fleiss’ 𝜅=0.76). Annotator bias is partially miti-
gated through blind annotation protocols, though recruiting from
the development team introduces implicit knowledge of system
design. Correlated LLM failures are reduced by selecting judges
from distinct model families, though cross-family correlated errors
remain a known risk [ 29,50]. Synthetic persona generation may
introduce distributional differences between calibration and eval-
uation data, though empirical results in Section 4 show coverage
guarantees hold across all 𝛼settings [ 9].External validityis lim-
ited by our single-case study design, which prioritises industrial
depth over experimental breadth consistent with embedded case
study methodology [ 62]. Generalisation to other LBS domains re-
quires further validation, and domain specialist involvement as
co-annotators would strengthen correctness judgments for domain-
critical criteria. Section 5 sets out which claims we believe transfer
as mechanisms, which transfer only after re-calibration, and which
are specific to this case.Construct validitythreats include the four
Likert-scale dimensions that measure expert perception rather than
end-to-end fault-detection capability. Connecting oracle verdicts to
production defect detection rates would provide stronger evidence
and is identified as future work.
6 Conclusion
We have presented an automated acceptance testing framework for
LLM-based software, addressing the fundamental challenge of test-
ing systems with non-deterministic, context-dependent behaviour
through two technical contributions: Requirements-Augmented
Generation (REAG) for context-aware oracle generation, and confid-
ence-calibrated cascade judgment with conformal risk control for
verdicts carrying calibrated selective reliability. The framework
achieves two test objectives: interpreting user intent within soft-
ware constraints ( obj-i ) and providing statistically reliable verdicts
on behaviour alignment ( obj-ii ). Our industrial case study on a
nutrition advisory application with 346 test scenarios demonstrates
that REAG achieves an average oracle quality of 3.91/5. The cas-
cade judge achieves 98.8% accuracy and improves average oracle
quality by 10% through cascade filtering, while being 31.7% more
cost-efficient than the strongest single judge. Per-tier conformal
calibration controls disagreement risk as intended, with empirical
human agreement meeting or exceeding the target threshold across
all𝛼settings.
Our evaluation reveals key limitations: 18% of oracles are affected
by retrieval errors or cases where the model follows prominent
scenario details outside the defined software requirements, and
edge cases with ambiguous intent still require human judgment.
Future work should validate the framework across diverse LBS
domains, involve domain specialists as co-annotators, and connect
oracle verdicts to production defect detection rates.

ASE ’26, October 12–16, 2026, Munich, Germany Wang et al.
7 Data Availability Statement
The code and prompt templates are publicly available at [ 1]. The
full evaluation dataset cannot be released due to a confidentiality
agreement with the industrial partner. To support replication and
community use, we provide a set of runnable anonymised data
points in the repository.
References
[1]2026. Replication Package: Requirements-Augmented Genera-
tion for Trustworthy Acceptance Testing of LLM-Based Software.
https://github.com/fanyuuwang/Requirements-Augmented-Generation-
for-Trustworthy-Acceptance-Testing-of-LLM-based-Software. Accessed:
2026-07-30.
[2]Raiyaan Abdullah, Yogesh Singh Rawat, and Shruti Vyas. 2025. iSafetyBench: A
video-language benchmark for safety in industrial environment. InProceedings
of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops.
1444–1453. doi:10.1109/iccvw69036.2025.00154
[3]Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin
Arbus, Rahul K Arora, Yu Bai, Bowen Baker, Haiming Bao, et al .2025. gpt-oss-
120b & gpt-oss-20b model card.arXiv preprint arXiv:2508.10925(2025). doi:10.
48550/arXiv.2508.10925
[4]Anastasios N Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal
Schuster. 2022. Conformal risk control.arXiv preprint arXiv:2208.02814(2022).
doi:10.48550/arXiv.2208.02814
[5]L. Antonelli, Mariángeles Hozikian, G. Camilleri, Alejandro Fernández, J. Grigera,
Diego Torres, and P. Zaraté. 2019. Wiki support for automated definition of
software test cases.Kybernetes49, 4 (2019), 1305–1324. doi:10.1108/k-10-2018-
0548
[6]Chetan Arora, Tomas Herda, and Verena Homm. 2024. Generating test scenarios
from NL requirements using retrieval-augmented LLMs: An industrial study. In
2024 IEEE 32nd International Requirements Engineering Conference (RE). IEEE,
240–251. doi:10.1109/re59067.2024.00031
[7]Sher Badshah, Ali Emami, and Hassan Sajjad. 2026. SCOPE: Selective Conformal
Optimized Pairwise LLM Judging.arXiv preprint arXiv:2602.13110(2026). doi:10.
48550/arXiv.2602.13110
[8]Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng
Zhang, Nicola Cancedda, and Pascale Fung. 2025. Hallulens: Llm hallucination
benchmark. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). 24128–24156. doi:10.18653/
v1/2025.acl-long.1176
[9]Rina Foygel Barber, Emmanuel J Candes, Aaditya Ramdas, and Ryan J Tibshirani.
2023. Conformal prediction beyond exchangeability.The Annals of Statistics51,
2 (2023), 816–845. doi:10.1214/23-aos2276
[10] Earl T Barr, Mark Harman, Phil McMinn, Muzammil Shahbaz, and Shin Yoo. 2015.
The oracle problem in software testing: A survey.IEEE transactions on software
engineering41, 5 (2015), 507–525. doi:10.1109/tse.2014.2372785
[11] Peter Bauer. 1991. Multiple testing in clinical trials.Statistics in medicine10, 6
(1991), 871–890. doi:10.1002/sim.4780100609
[12] E. Bjarnason, M. Unterkalmsteiner, Markus Borg, and Emelie Engström. 2016. A
multi-case study of agile requirements engineering and the use of test cases as
requirements.Information and Software Technology77 (2016), 61–79. doi:10.1016/
j.infsof.2016.03.008
[13] Alessio Bucaioni, M. Weyssow, Junda He, Yu zeng Lyu, and David Lo. 2025. A
Functional Software Reference Architecture for LLM-Integrated Systems.2025
IEEE 22nd International Conference on Software Architecture Companion (ICSA-C)
(2025), 1–5. doi:10.1109/icsa-c65153.2025.00006
[14] Shihao Cai, Chongming Gao, Yang Zhang, Wentao Shi, Jizhi Zhang, Keqin Bao,
Qifan Wang, and Fuli Feng. 2025. K-order Ranking Preference Optimization for
Large Language Models.Findings of the Association for Computational Linguistics:
ACL 2025(2025), 4844–4859. doi:10.18653/v1/2025.findings-acl.250
[15] Daihang Chen, Yonghui Liu, M. Zhou, Yanjie Zhao, Haoyu Wang, Shuai Wang,
Xiao Chen, Tégawendé F. Bissyandé, Jacques Klein, and Li Li. 2025. LLM for
Mobile: An Initial Roadmap.ACM Transactions on Software Engineering and
Methodology34, 5 (2025), 1–29. doi:10.1145/3708528
[16] Mike Cohn. 2024. Is It Dangerous to Calculate the Cost per Point?
https://www.mountaingoatsoftware.com/blog/is-it-dangerous-to-calculate-
the-cost-per-point. Last updated June 24, 2024.
[17] Jean-Pierre Corriveau and Wei Shi. 2013. Traceability in acceptance testing.
Journal of Software Engineering and Applications6, 10 (2013), 36–46. doi:10.4236/
jsea.2013.610a005
[18] Lívia Fernanda de Oliveira, Cássio Leonardo Rodrigues, and Renato de Freitas
Bulcão-Neto. 2023. Characterizing the software acceptance testing and the
inclusion of people with disabilities by means of a systematic mapping.IEEE
Latin America Transactions21, 1 (2023), 35–46. doi:10.1109/tla.2023.10015143[19] Marcela G. dos Santos, Sylvain Hallé, Fábio Petrillo, and Yann-Gaël Guéhéneuc.
2024. AAT4IRS: automated acceptance testing for industrial robotic systems.
Frontiers in Robotics and AI11 (2024). doi:10.3389/frobt.2024.1346580
[20] Yann Dubois, Chen Xuechen Li, Rohan Taori, Tianyi Zhang, Ishaan Gulrajani,
Jimmy Ba, Carlos Guestrin, Percy S Liang, and Tatsunori B Hashimoto. 2023.
Alpacafarm: A simulation framework for methods that learn from human feed-
back.Advances in Neural Information Processing Systems36 (2023), 30039–30069.
doi:10.52202/075280-1308
[21] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting
hallucinations in large language models using semantic entropy.Nature630,
8017 (2024), 625–630. doi:10.1038/s41586-024-07421-0
[22] Margarida Ferreira, Luis Viegas, João Pascoal Faria, and B. Lima. 2025. Acceptance
Test Generation with Large Language Models: An Industrial Case Study.2025
IEEE/ACM International Conference on Automation of Software Test (AST)(2025),
1–11. doi:10.1109/ast66626.2025.00007
[23] Stephan Goericke (Ed.). 2020.The future of software quality assurance. Springer
Nature. doi:10.1007/978-3-030-29509-7
[24] Shan Jiang, Chenguang Zhu, and Sarfraz Khurshid. 2024. Generating executable
oracles to check conformance of client code to requirements of JDK Javadocs
using LLMs.ArXivabs/2411.01789 (2024). doi:10.48550/arXiv.2411.01789
[25] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve
real-world github issues?arXiv preprint arXiv:2310.06770(2023). doi:10.48550/
arXiv.2310.06770
[26] Ankur Joshi, Saket Kale, Satish Chandel, and D Kumar Pal. 2015. Likert scale:
Explored and explained.British journal of applied science & technology7, 4 (2015),
396–403. doi:10.9734/bjast/2015/14975
[27] Jaehun Jung, Faeze Brahman, and Yejin Choi. 2024. Trust or Escalate: LLM Judges
with Provable Guarantees for Human Agreement.arXiv preprint arXiv:2407.18370
(2024). doi:10.48550/arXiv.2407.18370
[28] Shanthi Karpurapu, Sravanthy Myneni, Unnati Nettur, Likhit Sagar Gajja, Dave
Burke, Tom Stiehm, and Jeffery Payne. 2024. Comprehensive Evaluation and
Insights Into the Use of Large Language Models in the Automation of Behavior-
Driven Development Acceptance Test Formulation.IEEE Access12 (2024), 58715–
58721. doi:10.1109/access.2024.3391815
[29] Elliot Kim, Avi Garg, Kenny Peng, and Nikhil Garg. 2025. Correlated errors in
large language models.arXiv preprint arXiv:2506.07962(2025). doi:10.48550/arXiv.
2506.07962
[30] Hannah Kim, Kushan Mitra, Rafael Li Chen, Sajjadur Rahman, and Dan Zhang.
2024. Meganno+: A human-llm collaborative annotation system.arXiv preprint
arXiv:2402.18050(2024). doi:10.48550/arXiv.2402.18050
[31] Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin, Sean
Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo. 2024.
Prometheus 2: An open source language model specialized in evaluating other
language models.arXiv preprint arXiv:2405.01535(2024). doi:10.48550/arXiv.2405.
01535
[32] Patrick Lewis, Ethan Perez, Aleksandra Piktus, F. Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, M. Lewis, Wen tau Yih, Tim Rocktäschel, Se-
bastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks.Neural Information Processing Systems(2020).
doi:10.48550/arXiv.2005.11401
[33] Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad Beigi, Chengshuai Zhao,
Zhen Tan, Amrita Bhattacharjee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al .
2024. From generation to judgment: Opportunities and challenges of llm-as-a-
judge.arXiv preprint arXiv:2411.16594(2024). doi:10.48550/arXiv.2411.16594
[34] Feiyang Li, Peng Fang, Zhan Shi, Arijit Khan, Fang Wang, Dan Feng, Weihao
Wang, Xin Zhang, and Yongjian Cui. 2025. Cot-rag: Integrating chain of thought
and retrieval-augmented generation to enhance reasoning in large language
models.arXiv preprint arXiv:2504.13534(2025), 22. doi:10.48550/arXiv.2504.13534
[35] Haoxuan Li, Wei Song, Aofan Liu, and Peiwu Qin. 2025. AdaDocVQA: Adaptive
Framework for Long Document Visual Question Answering in Low-Resource
Settings.Proceedings of the 33rd ACM International Conference on Multimedia
(2025), 13777–13783. doi:10.1145/3746027.3761996
[36] Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. 2024. A survey on LLM-based
multi-agent systems: workflow, infrastructure, and challenges.Vicinagearth1, 1
(2024), 9. doi:10.1007/s44336-024-00009-2
[37] Zhipeng Li, Hailong Su, Yong Wu, Qinhu Zhang, Chang-An Yuan, V. Gribova, V.
Filaretov, and De-Shuang Huang. 2024. Hierarchical Multiview Top-k Pooling
With Deep-Q-Networks.IEEE Transactions on Artificial Intelligence5, 6 (2024),
2985–2996. doi:10.1109/tai.2023.3334261
[38] Daniel Maciel, A. C. Paiva, and A. Silva. 2019. From Requirements to Automated
Acceptance Tests of Interactive Apps: An Integrated Model-based Testing Ap-
proach.International Conference on Evaluation of Novel Approaches to Software
Engineering(2019), 265–272. doi:10.5220/0007679202650272
[39] Muhammad Fraz Malik, MNA Khan, Uzma Bibi, and Muhammad Ayaz Malik.
2017. Multi-Agent based Functional Testing in the Distributed Environment.
International Journal of Advanced Computer Science and Applications8, 8 (2017).
doi:10.14569/IJACSA.2017.080818

Requirements-Augmented Generation for Trustworthy Acceptance Testing of LLM-Based Software ASE ’26, October 12–16, 2026, Munich, Germany
[40] Jianing Qiu, Kyle Lam, Guohao Li, Amish Acharya, Tien Yin Wong, Ara Darzi,
Wu Yuan, and Eric J Topol. 2024. LLM-based agentic systems in medicine and
healthcare.Nature Machine Intelligence6, 12 (2024), 1418–1420. doi:10.1038/
s42256-024-00944-1
[41] Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S
Jaakkola, and Regina Barzilay. 2023. Conformal language modeling.arXiv
preprint arXiv:2306.10193(2023). doi:10.48550/arXiv.2306.10193
[42] I. K. Raharjana, Fadel Harris, and Army Justitia. 2020. Tool for Generating
Behavior-Driven Development Test-Cases.Journal of Information Systems Engi-
neering and Business Intelligence6, 1 (2020), 27–36. doi:10.20473/jisebi.6.1.27-36
[43] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, A. Shashua, Kevin Leyton-
Brown, and Y. Shoham. 2023. In-Context Retrieval-Augmented Language Models.
Transactions of the Association for Computational Linguistics11 (2023), 1316–1331.
doi:10.1162/tacl_a_00605
[44] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings
using siamese bert-networks.arXiv preprint arXiv:1908.10084(2019). doi:10.
48550/arXiv.1908.10084
[45] Per Runeson, Martin Host, Austen Rainer, and Bjorn Regnell. 2012.Case study
research in software engineering: Guidelines and examples. John Wiley & Sons.
doi:10.1002/9781118181034
[46] Yash Saxena, Ankur Padia, Mandar S. Chaudhary, Kalpa Gunaratna, Srinivasan
Parthasarathy, and Manas Gaur. 2025. Ranking Free RAG: Replacing Re-ranking
with Selection in RAG for Sensitive Domains.arXiv preprint arXiv:2505.16014
(2025). doi:10.48550/arXiv.2505.16014
[47] Jake C Snell and Thomas L Griffiths. 2025. Conformal prediction as bayesian
quadrature.arXiv preprint arXiv:2502.13228(2025). doi:10.48550/arXiv.2502.13228
[48] Tomasz Straszak and M. Smialek. 2015. Model-driven acceptance test automation
based on use cases.Computer Science and Information Systems12, 2 (2015),
707–728. doi:10.2298/csis141217033s
[49] Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. 2025. Efficient Context Selec-
tion for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k.Proceedings
of the 2025 Conference on Empirical Methods in Natural Language Processing(2025),
20116–20141. doi:10.18653/v1/2025.emnlp-main.1017
[50] Zailong Tian, Zhuoheng Han, Yanzhe Chen, Haozhe Xu, Xi Yang, Richeng Xuan,
Houfeng Wang, and Lizi Liao. 2025. Overconfidence in llm-as-a-judge: Diagnosis
and confidence-driven solution.arXiv preprint arXiv:2508.06225(2025). doi:10.
48550/arXiv.2508.06225
[51] Jack van Heugten Breurkes, Fabian Gilson, and M. Galster. 2022. Overlap be-
tween Automated Unit and Acceptance Testing – a Systematic Literature Review.
International Conference on Evaluation and Assessment in Software Engineering
(2022), 80–89. doi:10.1145/3530019.3530028
[52] June M Verner, Jennifer Sampson, Vladimir Tosic, NA Abu Bakar, and Barbara A
Kitchenham. 2009. Guidelines for industrially-based multiple case studies in soft-
ware engineering. In2009 Third International Conference on Research Challenges
in Information Science. IEEE, 313–324. doi:10.1109/rcis.2009.5089295
[53] Fanyu Wang, Chetan Arora, Yonghui Liu, Kaicheng Huang, Chakkrit Tan-
tithamthavorn, Aldeida Aleti, Dishan Sambathkumar, and David Lo. 2025. Multi-
Modal Requirements Data-based Acceptance Criteria Generation using LLMs. In
2025 40th IEEE/ACM International Conference on Automated Software Engineering
(ASE)(Seoul, Korea, Republic of). IEEE Press, 3334–3345. doi:10.1109/ASE63991.
2025.00275
[54] Fanyu Wang, Chetan Arora, Chakkrit Tantithamthavorn, Kaicheng Huang, and
Aldeida Aleti. 2025. Requirements-driven automated software testing: A system-
atic review.ACM Transactions on Software Engineering and Methodology(2025).
doi:10.20944/preprints202502.0628.v1 Preprint; DOI resolves to Preprints.org,
not the ACM TOSEM version.[55] Fanyu Wang, Xiaoxi Kang, Paul Burgess, Aashish Srivastava, Chetan Arora,
Adnan Trakic, Lay-Ki Soon, Md Khalid Hossain, and Lizhen Qu. 2026. LePREC:
Reasoning as Classification over Structured Factors for Assessing Relevance of
Legal Issues.arXiv preprint arXiv:2604.19464(2026). doi:10.48550/arXiv.2604.19464
[56] Junjie Wang, Yuchao Huang, Chunyang Chen, Zhe Liu, Song Wang, and Qing
Wang. 2024. Software Testing With Large Language Models: Survey, Landscape,
and Vision.IEEE Transactions on Software Engineering50, 4 (2024), 911–936.
doi:10.1109/tse.2024.3368208
[57] Xinru Wang, Hannah Kim, Sajjadur Rahman, Kushan Mitra, and Zhengjie Miao.
2024. Human-llm collaborative annotation through effective verification of llm
labels. InProceedings of the 2024 CHI Conference on Human Factors in Computing
Systems. 1–21. doi:10.1145/3613904.3641960
[58] Claes Wohlin. 2021. Case Study Research in Software Engineering—It is a Case,
and it is a Study, but is it a Case Study?Information and Software Technology133
(2021), 106514. doi:10.1016/j.infsof.2021.106514
[59] Fangzhou Wu, Ning Zhang, Somesh Jha, Patrick McDaniel, and Chaowei Xiao.
2024. A new era in llm security: Exploring security concerns in real-world llm-
based systems.arXiv preprint arXiv:2402.18649(2024). doi:10.48550/arXiv.2402.
18649
[60] Peng Xia, Peng Xia, Kangyu Zhu, Haoran Li, Haoran Li, Weijia Shi, Sheng Wang,
Linjun Zhang, James Zou, and Huaxiu Yao. 2024. MMed-RAG: Versatile Multi-
modal RAG System for Medical Vision Language Models.International Conference
on Learning Representations(2024). doi:10.48550/arXiv.2410.13085
[61] Yunpeng Xu, Mufang Ying, Wenge Guo, and Zhi Wei. 2025. Two-stage risk
control with application to ranked retrieval. InProceedings of the Thirty-Fourth
International Joint Conference on Artificial Intelligence(Montreal, Canada)(IJCAI
’25). 9104–9111. doi:10.24963/ijcai.2025/1012
[62] Robert K Yin. 2018.Case study research and applications(6 ed.). Sage Thousand
Oaks, CA.
[63] Boxi Yu, Qiuyang Mang, Qingshuo Guo, and Pinjia He. 2023. Retromorphic
Testing: A New Approach to the Test Oracle Problem.ArXivabs/2310.06433
(2023). doi:10.48550/arXiv.2310.06433
[64] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. 2025.
Evaluation of Retrieval-Augmented Generation: A Survey.Communications in
Computer and Information Science(2025), 102–120. doi:10.1007/978-981-96-1024-
2_8
[65] Jinghan Zhang, Xiting Wang, Weijieying Ren, Lu Jiang, Dongjie Wang, and
Kunpeng Liu. 2025. Ratt: A thought structure for coherent and correct llm
reasoning. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 39.
26733–26741. doi:10.1609/aaai.v39i25.34876
[66] Quanjun Zhang, Chunrong Fang, Siqi Gu, Ye Shang, Zhenyu Chen, and Liang
Xiao. 2025. Large Language Models for Unit Testing: A Systematic Literature
Review.arXiv preprint arXiv:2506.15227(2025). doi:10.48550/arXiv.2506.15227
[67] Jiahui Zhao, Ziyi Meng, Stepan Gordeev, Zijie Pan, Dongjin Song, Sandro Stein-
bach, and Caiwen Ding. 2024. Key Information Retrieval to Classify the Un-
structured Data Content of Preferential Trade Agreements.arXiv.org(2024).
doi:10.48550/arXiv.2401.12520
[68] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al .2023. Judging
llm-as-a-judge with mt-bench and chatbot arena.Advances in neural information
processing systems36 (2023), 46595–46623. doi:10.52202/075280-2020
[69] Xin Zhou, Yiwen Guo, Ruotian Ma, Tao Gui, Qi Zhang, and Xuanjing Huang. 2025.
Self-consistency of the internal reward models improves self-rewarding language
models.arXiv preprint arXiv:2502.08922(2025). doi:10.48550/arXiv.2502.08922
Received 2026-04-30; accepted 2026-07-01