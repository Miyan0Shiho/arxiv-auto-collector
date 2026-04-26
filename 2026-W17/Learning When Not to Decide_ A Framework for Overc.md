# Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication

**Authors**: Mohamed Afane, Emily Robitschek, Derek Ouyang, Daniel E. Ho

**Published**: 2026-04-21 18:17:08

**PDF URL**: [https://arxiv.org/pdf/2604.19895v1](https://arxiv.org/pdf/2604.19895v1)

## Abstract
A well-known limitation of AI systems is presumptuousness: the tendency of AI systems to provide confident answers when information may be lacking. This challenge is particularly acute in legal applications, where a core task for attorneys, judges, and administrators is to determine whether evidence is sufficient to reach a conclusion. We study this problem in the important setting of unemployment insurance adjudication, which has seen rapid integration of AI systems and where the question of additional fact-finding poses the most significant bottleneck for a system that affects millions of applicants annually. First, through a collaboration with the Colorado Department of Labor and Employment, we secure rare access to official training materials and guidance to design a novel benchmark that systematically varies in information completeness. Second, we evaluate four leading AI platforms and show that standard RAG-based approaches achieve an average of only 15% accuracy when information is insufficient. Third, advanced prompting methods improve accuracy on inconclusive cases but over-correct, withholding decisions even on clear cases. Fourth, we introduce a structured framework requiring explicit identification of missing information before any determination (SPEC, Structured Prompting for Evidence Checklists). SPEC achieves 89% overall accuracy, while appropriately deferring when evidence is insufficient -- demonstrating that presumptuousness in legal AI is systematic but addressable, and that doing so is a necessary step towards systems that reliably support, rather than supplant, human judgment wherever decisions must await sufficient evidence.

## Full Text


<!-- PDF content starts -->

Learning When Not to Decide: A Framework for Overcoming
Factual Presumptuousness in AI Adjudication
Mohamed Afane
afane@law.stanford.edu
Stanford University
Stanford, CA, USAEmily Robitschek
erobitsc@law.stanford.edu
Stanford University
Stanford, CA, USA
Derek Ouyang
douyang1@law.stanford.edu
Stanford University
Stanford, CA, USADaniel E. Ho
dho@law.stanford.edu
Stanford University
Stanford, CA, USA
Abstract
A well-known limitation of AI systems ispresumptuousness: the
tendency of AI systems to provide confident answers when infor-
mation may be lacking. This challenge is particularly acute in legal
applications, where a core task for attorneys, judges, and admin-
istrators is to determine whether evidence is sufficient to reach a
conclusion. We study this problem in the important setting of unem-
ployment insurance adjudication, which has seen rapid integration
of AI systems and where the question of additional fact-finding
poses the most significant bottleneck for a system that affects mil-
lions of applicants annually. First, through a collaboration with
the Colorado Department of Labor and Employment, we secure
rare access to official training materials and guidance to design a
novel benchmark that systematically varies in information com-
pleteness. Second, we evaluate four leading AI platforms and show
that standard RAG-based approaches achieve an average of only
15% accuracy when information is insufficient. Third, advanced
prompting methods improve accuracy on inconclusive cases but
over-correct, withholding decisions even on clear cases. Fourth,
we introduce a structured framework requiring explicit identifi-
cation of missing information before any determination (SPEC,
Structured Prompting for Evidence Checklists). SPEC achieves 89%
overall accuracy, while appropriately deferring when evidence is
insufficient—demonstrating that presumptuousness in legal AI is
systematic but addressable, and that doing so is a necessary step
towards systems that reliably support, rather than supplant, human
judgment wherever decisions must await sufficient evidence.
CCS Concepts
•Computing methodologies →Natural language processing;
•Applied computing→Law.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
ICAIL ’26, Singapore, Singapore
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXKeywords
legal reasoning, model uncertainty, benefits adjudication
ACM Reference Format:
Mohamed Afane, Emily Robitschek, Derek Ouyang, and Daniel E. Ho. 2026.
Learning When Not to Decide: A Framework for Overcoming Factual Pre-
sumptuousness in AI Adjudication. InProceedings of Twenty-First Interna-
tional Conference on Artificial Intelligence and Law (ICAIL ’26).ACM, New
York, NY, USA, 10 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Legal reasoning has always required navigating uncertainty. Statutes
employ vague standards—“reasonable efforts,” “good cause,” “willful
misconduct”—that demand interpretation against specific facts. Evi-
dence is incomplete; witnesses contradict each other; circumstances
resist clean categorization. For human adjudicators, this is familiar
terrain. For artificial intelligence (AI), it has proven treacherous.
The earliest legal expert systems, developed in the 1980s and
1990s, encoded statutory rules as deterministic decision trees. They
worked well when law reduced to bright-line tests—twenty weeks
of employment, a specific dollar threshold—but collapsed when
confronting the uncertainty endemic to legal reasoning. Governing
law proved too complex to represent directly as symbolic logic;
as Kevin Ashley documented, they “did not possess the capacity
to represent uncertainty or engage in non-monotonic reasoning
about changing facts” [ 4]. Modern transformer-based architectures
have not resolved this limitation. Despite remarkable linguistic
capabilities, large language models (LLM) generate fluent, assured
outputs regardless of whether the underlying evidence warrants
such confidence [ 39]. The systems that once failed because they
could not handle uncertainty now fail because they cannot recog-
nize insufficiency.
This presumptuousness is a general problem for AI deployed in
reasoning tasks, but it is particularly acute in law—and nowhere
more so than in administrative adjudication, where AI deployment
is accelerating, and errors directly harm individuals [ 10]. Unem-
ployment insurance (UI) offers a compelling domain to study this
problem. The governing law is formidably complex: scattered statu-
tory provisions, vague eligibility standards, and jurisdiction-specific
exceptions that resist systematic codification [ 28,40]. Developing
sound adjudicative judgment can take years. When a researcher
asked a California UI claims processor why he kept deferring ques-
tions to senior colleagues, he replied: “I’ve only been here 17 years.arXiv:2604.19895v1  [cs.AI]  21 Apr 2026

ICAIL ’26, June 08–12, 2026, Singapore, Singapore Afane, Robitschek, Ouyang, and Ho
The folks who really know how UI works have been here for 25
years or more” [ 28]. Agencies maintain extensive training programs
to cultivate this judgment in new adjudicators—and increasingly
look to AI to help.
The pressure to automate is intense. During the pandemic, unem-
ployment systems nationwide collapsed under unprecedented claim
volumes, not primarily because they lacked processing capacity, but
because fact-finding became the bottleneck. Determining eligibility
requires gathering and weighing evidence: why did the claimant
leave their job? Did the employer provide adequate documentation?
Is the claimant’s account consistent with the record? When systems
could not perform this evidentiary work adequately, they failed [ 28].
Agencies find themselves caught between competing demands: rely
on merit staff and fail to meet timeliness requirements, or automate
and risk running afoul of rules designed to ensure human judgment
[6]. The Colorado Department of Labor and Employment (CDLE)
has piloted AI tools designed to help adjudicators “do the work more
accurately and efficiently” rather than replace human judgment [ 2].
New Hampshire is testing an adjudication assistant that collects
more complete information at the start of claims [ 1]. The question
is not whether AI will be integrated into benefits administration—it
already is—but whether these systems can recognize the limits of
their own knowledge.
The consequences of getting this wrong are severe. In 2015,
Michigan’s automated UI system falsely accused over 20,000 resi-
dents of fraud, issuing confident determinations without adequate
evidentiary support. The system’s designers had not equipped it
to recognize when information was insufficient; it simply decided.
Many residents had wages garnished or tax refunds seized before
the errors came to light. At least one person took her own life after
being assessed $50,000 in penalties [ 11]. The pattern extends to
eligibility determinations: AI systems that deny without sufficient
evidence strand workers who may qualify; systems that approve
without sufficient evidence create overpayments that agencies later
claw back. Either way, the harm flows from deciding before the
facts warrant it.
A common approach to addressing these concerns is to have a
human in the loop. But this framing obscures critical design choices.
Two well-known failure modes in human–AI interaction are au-
tomation bias, where humans over-rely on algorithmic suggestions,
and algorithm aversion, where humans dismiss useful recommenda-
tions [ 3]. Humans can identify erroneous algorithmic outputs and
override them, but only when systems provide clear signals about
uncertainty rather than overconfident predictions that humans
must second-guess [ 7]. A system that explicitly identifies missing
information before reaching a determination provides exactly such
a signal.
Most benchmarks for evaluating AI systems, including those in
legal domains, sidestep this question entirely. They assess perfor-
mance when all relevant information is provided, testing whether
models reach correct conclusions, not whether they can recognize
when conclusions are unwarranted [ 15,17,18,20,42]. This paper
addresses that gap. Through a collaboration with CDLE, we obtain
rare access to official training materials—the actual guidance used
to teach human adjudicators when they have enough information
to decide. We use these materials, alongside governing statutes pro-
vided through retrieval-augmented generation (RAG), to constructeligibility scenarios that vary systematically in information com-
pleteness: cases that are clearly eligible, clearly ineligible, and cases
where a competent adjudicator should recognize that determination
must await further fact-finding.
We contribute:(1)The first legal adjudication dataset that pre-
cisely controls the presence or absence of legally operative facts,
developed through collaboration with CDLE that provided deep
immersion into the domain, including thorough review of inter-
nal training materials and comprehensive understanding of how
information gaps manifest across diverse cases;(2)An evaluation
showing that four leading AI platforms, given identical statutory
and official adjudication guidance through RAG, achieve on average
15% accuracy on cases lacking sufficient information, delivering
presumptuous yes/no eligibility decisions when they should de-
fer;(3)Evidence that advanced prompting methods and detailed
adjudication instructions, even with explicit guidance to withhold
decisions when information is missing, over-correct and fail to de-
cide clear cases; and(4)SPEC ( Structured Prompting for Evidence
Checklists), a multi-stage framework that eliminates baseline pre-
sumptuousness and resolves the determination–deferral tradeoff,
achieving 89% overall accuracy with appropriate deferral when
evidence is insufficient.
2 Unemployment Insurance
Unemployment insurance is an emblematic case of legal reason-
ing amid incomplete information. UI provides temporary income
support to workers who lose employment through no fault of their
own. Eligibility determinations require applying statutory crite-
ria to case-specific facts. For example, under C.R.S. § 8-73-108, a
claimant discharged for workplace misconduct may be disqualified
from benefits, but only if the conduct meets statutory definitions
requiring assessment of severity, intent, and employer policies.
Adjudicators work from information submitted by claimants
and employers, supplemented by interviews and documentation
requests. This information is routinely incomplete: claimants may
not know what facts are legally relevant; employers may provide
conclusory characterizations without supporting detail; accounts
may conflict on outcome-determinative points. Determining when
sufficient information exists to decide, versus when additional fact-
finding is required, is central to adjudicative practice.
State agencies maintain detailed adjudication guides that oper-
ationalize statutory requirements, specifying what facts must be
established for each eligibility criterion, how to weigh conflicting
evidence, and when to request additional information. In a unique
collaboration, we are able to access CDLE’s guide, which covers
voluntary quit, discharge for misconduct, availability for work, and
other criteria under Colorado Revised Statutes.
3 Related Works
Our work pertains to five bodies of research.
1. Retrieval-augmented generation (RAG) for Legal Tasks.
RAG addresses a fundamental limitation of language models: their
knowledge is fixed at training time [ 21] and may not include the
specific legal provisions relevant to a given task. RAG systems
retrieve relevant documents—statutes, regulations, adjudication

Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication ICAIL ’26, June 08–12, 2026, Singapore, Singapore
guides—and provide them to LLMs as context alongside each query,
grounding model outputs in authoritative sources [12, 29].
For legal adjudication, RAG offers a natural fit. The system can re-
trieve exactly those statutory provisions and interpretive guidance
that govern the case at hand, rather than relying on whatever legal
knowledge the model absorbed during pretraining. Recent work
has explored sophisticated retrieval strategies, including case-based
reasoning approaches that use similarity between legal questions
to improve context selection [ 37]. In our evaluation, all systems
receive identical Colorado UI materials through RAG: the relevant
Colorado Revised Statutes, administrative regulations, and official
adjudication guidance used by CDLE.
However, retrieval alone does not overcome presumptuousness
[26]. Providing a model with the legal standard for “willful miscon-
duct” does not ensure it will recognize when the facts provided are
insufficient to apply that standard. The model may retrieve the cor-
rect legal rule and still issue a determination based on incomplete
evidence. Our evaluation tests the rate at which this occurs and
whether structured prompting can prevent it.
2. Legal AI Benchmarks.Existing legal AI benchmarks de-
signed specifically for legal reasoning—LegalBench’s statutory and
contract reasoning tasks [ 15], CUAD’s contract clause extraction
[17], SARA’s tax law entailment tasks [ 18], CaseHOLD’s legal hold-
ing identification [ 42], and ContractNLI’s contract inference evalu-
ation [ 20]—all share a structural assumption: that sufficient infor-
mation exists to determine the correct answer. None systematically
test whether models can recognize when information isinsufficient:
when the appropriate response is “more facts are needed,” rather
than a definitive conclusion. This gap mirrors a broader limita-
tion in LLM evaluation, where benchmarks predominantly assess
passive reasoning with complete information rather than active
reasoning requiring information gathering [ 43]. Recent work has
begun examining related limitations. He et al. addressinterpretive
ambiguity—the challenge arising when legal principles can be read
multiple ways—proposing computational frameworks that mirror
legal mechanisms for constraining interpretation [ 16]. Our work
addresses a complementary problem: not ambiguity in what the
rule means, but insufficiency in the facts available to apply it. Both
require AI systems to overcome presumptuousness, but operate at
different points in the reasoning process.
3. LLM Information-Seeking Under Uncertainty.A growing
body of work examines whether language models can recognize
information gaps in other domains. QuestBench evaluates whether
models can identify missing variables in underspecified logic and
math problems; even when models solve fully-specified versions,
they achieve only 40–50% accuracy at selecting appropriate clari-
fication questions [ 22]. AR-Bench tests active reasoning through
iterative questioning in detective scenarios and puzzles, finding
that models frequently fail to acquire needed information even
with sophisticated search strategies [ 43]. MediQ simulates clini-
cal interactions requiring diagnostic follow-up questions; directly
prompting models to ask questions actuallydegradesperformance
by 11% compared to immediate diagnosis [23].
These parallel efforts reveal a common problem: current LLMs
struggle to recognize when they lack sufficient information to an-
swer confidently. Wang et al. find a related pattern in comparing
A claimant was terminated for using a not 
medically prescribed controlled substance at work. 
Is the claimant eligible for unemployment benefits 
considering the following facts? 
1. The separation was non-volitional 
2. The separation was directly due the use of the substance 
3. The claimant declared a substance use disorder to the division 
4. Disorder substantiated by licensed physician OR treatment program 
5. The claimant started an approved treatment program within 4 weeks 
SPEC 
 SPEC The claimant is eligible for a 
full award of UI benefits. 
Inconclusive. We need to 
confirm if they started an 
approved program within the 
4-week deadline. 5/5 
facts 4/5 
facts ✓ ✗
✓Figure 1: Comparison of model outputs on a termination case
involving non-prescribed controlled substance use at work.
When all five statutory requirements are provided (left), mod-
els reach the correct eligibility determination. When one
requirement is withheld (right), standard LLMs still issue
presumptuous eligibility decisions, while SPEC identifies the
missing information and outputs Inconclusive.
AI agents to human workers: agents take “an overwhelmingly pro-
grammatic approach” across work domains, often producing infe-
rior work while masking deficiencies through confident execution
[34]. The machine learning literature formalizes this challenge as
abstention: knowing when not to answer [ 36]. AbstentionBench
finds that reasoning fine-tuning hurts abstention and model scale
provides no improvement [ 19]; medical QA work reaches simi-
lar conclusions [ 24]. The selective prediction literature formalizes
the underlying tradeoff: improving accuracy on answered cases
reduces coverage of decidable ones [ 13,14], with subsequent work
addressing over-abstention specifically [32].
But in legal adjudication, abstention is not merely a statistical
tradeoff to optimize—it is a procedural requirement. When evidence
is insufficient, the appropriate response is additional fact-finding,
not a confident guess. And effective human–AI collaboration de-
pends on this distinction: when systems provide confident pre-
dictions, humans struggle to identify when override is warranted
[7]. Explicit gap identification provides exactly the signal that as-
sertive predictions obscure. Our work extends this investigation
to the legal domain, where deferral carries concrete procedural
meaning, authoritative texts define what information is needed,
and serious consequences can occur when systems issue premature
determinations without adequate support.
4. Prompting Techniques and Model Calibration.Prompt-
ing is commonly used to improve reasoning. Chain-of-Thought
(CoT) decomposes complex problems into intermediate steps [ 35].
Tree-of-Thoughts (ToT) explores multiple reasoning branches [ 38],
extended by Graph-of-Thought to model reasoning as arbitrary
graphs [ 5]. Self-Consistency uses consensus across solution paths as
an uncertainty signal [ 33]. Reflexion enables iterative self-critique

ICAIL ’26, June 08–12, 2026, Singapore, Singapore Afane, Robitschek, Ouyang, and Ho
and refinement [ 31]. Multi-Agent Debate uses multiple model in-
stances to surface errors through deliberation [ 9]. These techniques
improve reasoningquality, but are not specifically designed to rec-
ognizeinformation gaps—to know when to defer rather than when
to reason more carefully.
Research on model calibration addresses the related problem of
aligning confidence with accuracy. Overconfidence is persistent:
models generate confident predictions even when uncertainty is
high [ 39]. Technical approaches including temperature scaling and
learned calibration show promise [ 8,30]. But calibration typically
asks:given the available information, how sure should the model be of
its answer?Our problem is upstream:is there enough information to
answer at all?The first relates to overconfidence within a decision;
the second whether to decide in the first place, with implications
for subsequent information gathering.
5. From Symbolic Rules to Structured Guidance.Early le-
gal expert systems represented statutory requirements as deci-
sion trees with discrete nodes for each legally operative fact [ 41].
When a required fact was unavailable, the system deferred rather
than guessed. Although presumptuousness would seem to be well-
addressed in this way, symbolic approaches never scaled. Translat-
ing statutes into formal logic required painstaking manual effort by
legal experts and knowledge engineers working provision by pro-
vision, and the resulting systems remained brittle, unable to adapt
when statutes changed or cases fell outside anticipated patterns [ 4].
Neural methods appear to offer a solution to this bottleneck. Lan-
guage models appear capable of applying authoritative guidance
to novel fact patterns without requiring manual formalization of
every statutory provision. Even when the legal documents them-
selves are often dense, inconsistently organized, and difficult to
operationalize directly, RAG methods help to focus LLM reasoning
on the documents that contain the governing requirements, and
that human adjudicators would reference.
Yet access to the right materials is not sufficient. Figure 1 il-
lustrates a case with discrete statutory requirements that can be
verified as present or absent; when one requirement is withheld,
standard language models issue presumptuous eligibility determi-
nations despite the gap. Our work explores whether structured
prompting can recover the key virtue of symbolic systems: explicit
gap detection. At the same time, Figure 2 underscores the broader,
more fundamental challenge: discrete factual requirements exist
alongside qualitative standards requiring much more subjective,
contextual judgment (i.e., whether a pay reduction was “unreason-
able”), often contained only in the state agencies’ internal adjudica-
tion guides.
SPEC offers a path forward through the seeming incompatibility
of symbolic and neural systems, by recognizing that these guides
approximate soft symbolic rules: not rigid decision trees, but lists
identifying legally operative facts for each separation type. Neu-
ral methods can perform the contextual reasoning necessary for
individual judgments—assessing whether a claimant’s job search
constitutes “reasonable effort” by weighing labor market condi-
tions, qualifications, and search strategies—while an overarching
prompting structure maintains focus on gap detection across a set of
governing legal requirements and guidance standards. Thus, across
different discrete stages within a legal reasoning task, our approach
The claimant objected to the 
compensation and did not 
accept the modified terms. Legal 
requirements 
The reduction in the 
claimant’s rate of pay must 
have been unreasonable. Guidance 
documents 
The adjudicator shall consider whether the 
reduction targeted only this individual or applied 
to all workers in the same class, economic 
conditions, the employer's financial status, and 
acceptance by similarly situated workers. Loss 
of shift differential or overtime is not 
unreasonable unless such pay was guaranteed. A claimant quit their job on May 15th after the employer 
reduced their hourly rate from $35 to $25. The employer stated 
the reduction was permanent due to budget constraints. The 
claimant quit within one week of the pay cut taking effect. The 
prevailing rate for this job in the locality is $32 to $38 per hour. 
Is the claimant eligible for UI benefits? Figure 2: A voluntary quit case involving pay reduction re-
quires contextual evaluation of two statutory requirements:
whether the reduction was “unreasonable” and whether the
claimant objected. The determination must consider mul-
tiple factors including whether the reduction targeted the
individual or all workers, economic conditions, employer fi-
nancial status, and acceptance by similarly situated workers.
leverages the potential for complementarity between neural flexi-
bility (i.e., distilling requirements from open-ended text, navigating
qualitative judgment for soft guidance) and symbolic structure (i.e.,
maintaining a functional checklist, flagging gaps).
4 Methodology
SPEC is a multi-agent framework that guards against presump-
tuousness by making information sufficiency an explicit part of
the reasoning process. The core principle is straightforward: legal
determinations should only proceed when sufficient information
exists to support them. When critical facts are missing, the sys-
tem identifies what is needed rather than rendering an unreliable
judgment.
4.1 Problem Formulation
Consider an adjudication task where a query 𝑞describes a claimant’s
situation, and 𝐶a corpus of statutory provisions, administrative
guides, and cited case law containing the legal criteria for eligibility.
The system extracts facts 𝐹from𝑞and derives requirements 𝑅from
𝐶that must be satisfied for a determination.
For each requirement 𝑟∈𝑅 , the framework evaluates whether
the facts𝐹adequately support 𝑟. The information gap 𝐺consists
of requirements lacking adequate support:
𝐺={𝑟∈𝑅:𝑟is not satisfied by facts𝐹in𝑞}(1)
A system is properly calibrated for adjudication if it produces a
determination only when no gaps exist, and otherwise outputs
inconclusive along with identification of what information is miss-
ing. Standard prompting fails this criterion because language mod-
els generate determinations directly without explicitly evaluating
whether each requirement is satisfied by the available facts.

Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication ICAIL ’26, June 08–12, 2026, Singapore, Singapore
4.2 System Overview
SPEC consists of three main components: requirement retrieval,
multi-agent analysis, and determination. Figure 3 illustrates the
architecture.
The retrieval component extracts all relevant passages from adju-
dication guides related to the query. The system retrieves complete
legal citations, statutory provisions, considerations, and case law
as unified blocks from 𝐶. Multi-part requirements are preserved
together as complete passages rather than fragmented. This compo-
nent performs extraction only—no conclusions or determinations
are permitted. The extraction preserves exact wording to maintain
legal precision and retrieves examples from adjudication guides
that inform subsequent analysis.
4.3 Multi-Agent Analysis and Fact Verification
The core of SPEC is a three-agent system that maps requirements
to facts through specialized analytical instructions. All agents must
reference the exact question text without rephrasing, and treat
stated facts as established without demanding external verification.
Agent 1 (Requirements Checklist) extracts a structured check-
list from the guides with three categories: required elements (core
statutory requirements with statute numbers), considerations (all
considerations from the guide, noting that not all apply to every
case), and case law requirements (case name and principle estab-
lished). This agent performs extraction only with no assessment.
Its role is to derive the requirements 𝑅from𝐶that must be satis-
fied for a determination. The checklist provides the foundation for
subsequent fact verification and supervisory review by explicitly
enumerating what information must be present.
Agent 2 (Fact Verification) compares each checklist item against
the case facts. The system extracts facts 𝐹from𝑞and compares
them against the requirements 𝑅derived by Agent 1. Each item 𝑟is
marked as either satisfied (requirement clearly met by facts 𝐹, with
the relevant passage quoted), or unaddressed (no direct mention
or incomplete information). For items marked unaddressed, the
agent assesses relevance: critical gaps are those required by statute
or case law or that are outcome-determinative, while items not
relevant are considerations that do not apply to the specific case.
The agent applies specific legally and logically grounded verifica-
tion principles: degree and severity require frequency, extent, and
intent; comparative requirements need actual numbers; document
format requirements need explicit description; contract mentions
do not equal violations without explicit statement; and conflicting
accounts between claimant and employer reflect critical gaps.
Agent 3 (Supervisory Review) serves as the final authority, in-
dependently reviewing the requirements generated by Agent 1 and
the previous assessment of Agent 2. This agent verifies satisfied
items against legal requirements and overturns to unaddressed if
legally insufficient. For unaddressed items, the agent independently
reassesses whether facts could be reasonably inferred from stated
facts and whether gaps are truly outcome-determinative.
When Agent 3 recommends proceeding, no critical gaps remain,
and the system produces a determination grounded in the statutory
provisions and case law retrieved in the first component. When
Agent 3 recommends abstention, critical gaps exist, and the systemoutputs “inconclusive” along with identification of which specific re-
quirements lack adequate support, and what additional information
is needed. This explicit gap identification is critical for fact-finding
processes conducted by UI adjudicators.
4.4 Design Principles
Our framework enforces specific principles that prevent both pre-
sumptuousness and excessive caution. These principles are also
incorporated into the enhanced prompting configurations used
when evaluating other AI systems. Stated facts that are not contra-
dicted by other stated facts are treated as established; the system
does not demand proof of assertions made in the query. However,
unstated facts remain unknown—the system does not infer nonex-
istence from silence or assume missing information favors either
party. When claimant and employer accounts directly conflict on
outcome-determinative facts, the conflict is flagged for resolution
through fact-finding rather than assumed away. Statutory thresh-
olds using comparative language (greater of, exceeds, longer than)
are assessed based on the specific values or circumstances provided
in the query. When statutes require consideration of specific factors,
absence of those factors in the query constitutes a gap preventing
determination. Relevant case law principles identified in adjudica-
tion guides must be satisfied by the stated facts for determination to
proceed. Explicit requirement enumeration transforms the implicit
question “is there enough information?” into an explicit reasoning
process over requirements𝑅and facts𝐹.
The multi-agent architecture creates a dependency chain where
each component receives outputs from all previous components
as explicit input, ensuring that the determination cannot execute
until multi-agent assessment completes. Rather than using fixed
templates, SPEC generates prompts dynamically based on each
query. Following a system prompt, an initial planner LLM analyzes
the query and produces tailored instructions that encode aware-
ness of position in the pipeline, explicit reference to outputs from
prior components, and analysis strategies specific to the statutory
provisions at issue.
A single-pass approach would require the model to simultane-
ously determine what facts are needed, whether those facts are
present, and what conclusion follows. SPEC decomposes these into
distinct components: requirement extraction, fact verification, and
supervisory review, each with explicit interfaces between stages.
5 Experiments
5.1 Dataset
We construct a dataset of 250 questions based on Colorado UI law
through careful curation grounded in detailed understanding of
the governing statutes, regulations, and procedures. As described
earlier, our collaboration with CDLE enables us to construct realis-
tic and representative questions that reflect the actual challenges
faced in UI adjudication. Questions are designed to test both the
ability to reach correct determinations when sufficient information
exists and the ability to recognize when critical facts are missing.
The dataset includes both conceptual questions about legal provi-
sions and fact-specific cases requiring eligibility determinations.
Questions address scenarios involving voluntary quit, discharge for

ICAIL ’26, June 08–12, 2026, Singapore, Singapore Afane, Robitschek, Ouyang, and Ho
Review all 
outputs and 
decide Statute 
Regs
Guides Relevant 
context Agent 3 
Requirements 
checklist Facts checked 
against 
requirements 
No gaps: 
Determination Gaps: 
Fact-finding 
guidance 
Extract 
without 
assessment Agent 1 
Preserve 
complete 
passages Retrieval Apply 
verification 
principles Agent 2 
System 
prompt 
Query Planner 
Dynamic prompts 
Figure 3: SPEC pipeline. A RAG system extracts statutory provisions, case law, and considerations as complete passages.
Dynamic prompts direct three agents: Agent 1 builds a requirements checklist, Agent 2 verifies facts against requirements, and
Agent 3 provides supervisory review with authority to overturn assessments. If no critical gaps exist, a determination is made;
otherwise, specific fact-finding is recommended.
misconduct, availability for work, and other eligibility criteria un-
der Colorado law. More than half of the cases (56%) are inconclusive,
where critical facts necessary for a determination are intentionally
withheld. These cases test whether systems appropriately defer
judgment when information is insufficient. The remaining cases
provide complete information necessary for accurate eligibility
determinations.
5.2 Models and Evaluation Metrics
For our comparisons with existing AI systems, we focus on Claude
Sonnet 4.5, GPT 5.2, Gemini 3, and NotebookLM. NotebookLM
is Google’s document analysis tool designed to answer questions
based on uploaded materials, and it has been suggested for legal
settings [ 27]. All systems receive identical legal materials through
RAG for each query. For SPEC, we use Sonnet 4.5 as the underlying
model. Additional analyses in this section explore varying levels of
information completeness, enhanced prompts, advanced prompting
techniques, ablation of SPEC features, and SPEC’s performance
when running on different underlying models. Across all analyses,
we measure accuracy on two types of cases. Complete cases contain
sufficient information for a determination, and we assess whether
the system produces the correct outcome, such as eligible/ineligible
for benefit eligibility cases or yes/no for other direct questions
where sufficient information is provided. Inconclusive cases lack
critical information needed for a determination, and we measure
whether the system correctly identifies these cases as inconclusive
rather than issuing a premature determination. Overall accuracy
reflects performance across the entire test set.
5.3 Baseline Performance and
Presumptuousness
We first evaluate all systems using a standard prompt without
specialized instructions to establish baseline performance:
You are an expert on Colorado Unemployment Insurance law.
Use the provided document to answer questions accurately and
concisely, provide clear determination and reasoning in one
paragraph, and answer the exact question asked by the user.Table 1 presents baseline results across all four systems. Overall
accuracy remains low, with GPT 5.2 achieving the highest perfor-
mance at 0.54 while Sonnet 4.5 reaches only 0.42. More critically,
accuracy on inconclusive cases reveals severe presumptuousness
across all systems. GPT 5.2 appropriately defers judgment in only
30% of cases lacking sufficient information, while Sonnet 4.5 defers
in just 8% of such cases. This pattern demonstrates that current sys-
tems routinely issue confident determinations even when critical
facts necessary for accurate adjudication are absent, reflecting a
fundamental failure to recognize the limits of available information.
Model All Cases Complete Inconclusive
Sonnet 4.5 0.42 0.84 0.08
Gemini 3 0.480.900.15
GPT 5.2 0.54 0.85 0.30
NotebookLM 0.440.900.08
SPEC0.890.890.89
Table 1: Baseline model accuracy on all cases, complete cases,
and inconclusive cases.
5.4 Model Presumptuousness Across Different
Levels of Information Completeness
Figure 4 illustrates model accuracies across different levels of infor-
mation completeness. Eligible and ineligible cases contain all facts
required for determination. Inconclusive cases are missing one to
four critical pieces of information, labeled as Missing-1 through
Missing-4, respectively. These comparisons test both whether sys-
tems recognize when available information is insufficient for reli-
able adjudication, and to what degree they can detect gaps. Missing-
4 cases with four missing facts should be relatively easier to identify
as incomplete, while Missing-1 cases with only one missing fact
more closely resemble complete cases and present a harder recog-
nition challenge.
All models perform reasonably well on complete cases where
all information is provided. Sonnet 4.5, Gemini 3, GPT 5.2, and
NotebookLM achieve 75-91% accuracy on eligible cases and 90-96%
on cases where claimants are ineligible. However, performance

Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication ICAIL ’26, June 08–12, 2026, Singapore, Singapore
collapses dramatically on inconclusive cases. Almost all models
fall below 35% accuracy when critical facts are missing, with most
performing substantially worse. NotebookLM and Sonnet 4.5 both
salvage 7% accuracy on Missing-1 cases. GPT 5.2 shows the best
performance among baseline models on inconclusive cases, main-
taining 70% on Missing-4, but still fails to appropriately defer on
the majority of cases lacking sufficient information. This pattern
demonstrates that systems routinely issue presumptuous determi-
nations even when critical facts are absent. For a claimant, a false
denial means lost income during a period of unemployment; a false
acceptance creates an overpayment the agency may later seek to
recover. Figure 5 shows how failures vary by model. Sonnet 4.5 is
especially likely to disqualify outright, issuing unwarranted denials
(“Ineligible”) in 68% of cases with missing information (94/139)—
particularly concerning given that our inconclusive cases do not
includeanydisqualifying facts. SPEC maintains consistent perfor-
mance across all categories, achieving 93% on eligible cases, 90% on
ineligible cases, and 83-100% across all inconclusive subcategories.
Figure 4: Model accuracy across information completeness
levels. Eligible and Ineligible cases contain all necessary facts.
Missing-1 through Missing-4 have one to four critical facts
missing. Models perform well on complete cases but fail
dramatically on inconclusive cases. SPEC maintains high
accuracy across all categories. Error bars show 95% bootstrap
confidence intervals.
5.5 Instruction-Enhanced Models and SPEC
We evaluate enhanced configurations where models receive ex-
plicit adjudication instructions. The most common issues involve
the “able and available” requirement and wage threshold verifica-
tion: while verifying availability and confirming claimants earned
sufficient wages during their base period are standard practices in
adjudication, these are applied across cases and are not the focus of
this work. The enhanced prompt (summary in the next paragraph;
full version in Appendix) addresses this by assuming claimants
are able, available, and meet wage requirements unless explicitly
stated otherwise, allowing models to focus on the substantive legal
questions that actually determine eligibility. Additional instructions
require deferral when critical facts are absent, verification of dead-
lines and thresholds, systematic statutory checking, and case law
application.
Figure 5: Error analysis of inconclusive cases (n=139). Rates
of false “denial” or “approval” of incomplete cases vary by
model.
Assume claimant able and available for work unless explicitly
stated otherwise. When critical information is genuinely miss-
ing, output Inconclusive and specify what additional facts are
needed; do not assume unstated facts. Verify all time-based
requirements including filing deadlines and waiting periods.
Check quantitative thresholds such as severity or frequency of
misconduct. Identify each statutory requirement and verify sat-
isfaction. Apply relevant case law from the adjudication guide.
Provide a clear determination with statutory citations.
Table 2 presents performance for enhanced configurations. En-
hanced instructions improve overall accuracy across all models.
GPT 5.2 Enhanced achieves the highest accuracy on inconclusive
cases at 0.90. SPEC achieves 0.89 overall accuracy with strong perfor-
mance on both complete cases (0.89) and inconclusive cases (0.89).
Figure 6 visualizes how enhanced instructions shift model behavior
from baseline to enhanced configurations. The gradient trajectories
exemplify an underlying determination–deferral tradeoff: models
that improve on inconclusive cases (like GPT 5.2, moving from 0.30
to 0.90) tend to become more conservative on complete cases (drop-
ping from 0.85 to 0.68). This pattern suggests that instruction-based
approaches to gap recognition face a tension between identifying
missing information and making determinations when information
is sufficient. While Sonnet 4.5 appears to partially escape this trade-
off, it still fundamentally trails behind in overall accurate deferral,
whereas SPEC achieves a near-Pareto improvement across the best
of all other model performances.
Model All Cases Complete Inconclusive
Sonnet 4.5 Enhanced 0.640.900.44
Gemini 3 Enhanced 0.75 0.88 0.65
GPT 5.2 Enhanced 0.80 0.680.90
SPEC0.890.89 0.89
Table 2: Enhanced model accuracy on all cases, complete
cases, and inconclusive cases.

ICAIL ’26, June 08–12, 2026, Singapore, Singapore Afane, Robitschek, Ouyang, and Ho
Figure 6: Performance trajectories from baseline to enhanced
prompting. For Gemini 3 and GPT 5.2, enhanced prompt-
ing improves inconclusive case accuracy but reduces com-
plete case performance (the determination–deferral trade-
off). SPEC achieves a near-Pareto improvement across all
performances, maintaining high accuracy on both complete
cases (89%) and inconclusive cases (89%).
5.6 Advanced Prompting Methods
We evaluate five established prompting techniques using GPT 5.2
to assess their effectiveness at legal adjudication and information
gap recognition. All methods receive identical statutory context
through RAG and use the enhanced prompt template. Table 3 shows
that advanced prompting methods achieve modest overall accuracy
(0.61-0.71) with similar performance patterns to standard prompt-
ing: reasonable accuracy on complete cases (0.76-0.87) but poor
deferral rates on inconclusive cases (0.49-0.64). These results show
that existing multi-step and multi-agent prompting approaches
provide marginal improvements but do not fundamentally address
the challenge of recognizing information gaps in legal adjudication.
Method All Cases Complete Inconclusive
Chain-of-Thought 0.700.870.57
Tree-of-Thoughts0.710.810.64
Self-Consistency 0.70 0.85 0.58
Reflexion 0.68 0.84 0.56
Multi-Agent Debate 0.61 0.76 0.49
Table 3: Advanced prompting methods accuracy on all cases,
complete cases, and inconclusive cases.
5.7 Advanced Prompting Methods with SPEC
We evaluate whether explicitly prompting advanced methods to
identify information gaps improves their performance. These con-
figurations incorporate SPEC’s core principle of systematic gap
identification through modified prompts, though they do not use
SPEC’s full multi-agent architecture.
Table 4 presents results for these gap-aware configurations. Over-
all accuracy ranges from 0.71 to 0.81, with CoT + SPEC achieving
the best performance. Inconclusive case accuracy improves acrossall methods, reaching 0.84-0.96 compared to 0.49-0.64 in the baseline
versions. However, complete case accuracy drops to 0.50-0.74, indi-
cating these configurations become overly conservative in general—
in other words, vulnerable to the same determination–deferral
tradeoff.
Method All Cases Complete Inconclusive
Chain-of-Thought + SPEC0.81 0.740.87
Tree-of-Thoughts + SPEC 0.77 0.530.96
Self-Consistency + SPEC 0.79 0.72 0.84
Reflexion + SPEC 0.73 0.50 0.91
Multi-Agent Debate + SPEC 0.71 0.50 0.87
Table 4: Advanced prompting methods with explicit gap de-
tection instructions.
5.8 Ablation Studies
We isolate the contributions of SPEC’s components through system-
atic evaluation. Table 5 presents results across four configurations.
2-Agent (No Extractor)merges the extraction and verification func-
tions into one agent while retaining the independent supervisor.
2-Agent (No Supervisor)merges verification and supervisory review
while retaining a separate extraction agent.Single Agentcollapses
all three functions into one agent performing extraction, verifica-
tion, and review in a single pass.Static Promptingreplaces dynamic
prompt generation with fixed templates: whereas the full SPEC
pipeline generates stage-specific prompts adapted to each question,
Static Prompting uses identical prompts regardless of the legal issue
presented.
Configuration All Cases Complete Inconclusive
2-Agent (No Extractor) 0.78 0.85 0.73
2-Agent (No Supervisor) 0.77 0.85 0.70
Single Agent 0.78 0.83 0.75
Static Prompting0.84 0.86 0.83
Table 5: Ablation results showing accuracy on all cases, com-
plete cases, and inconclusive cases.
Ablation across multiple configurations reveals which compo-
nents contribute most to performance. The 2-Agent (No Supervi-
sor) configuration shows the largest performance drop, with in-
conclusive case accuracy falling to 70% compared to 89% for the
full pipeline. When fact verification and supervisory review are
combined, the system loses the independent check that catches
errors in requirement satisfaction. The 2-Agent (No Extractor) and
Single Agent configurations show similar performance, maintain-
ing 78% overall accuracy with 73-75% on inconclusive cases. Static
Prompting achieves performance closest to the full pipeline at 84%
overall with 83% on inconclusive cases. These results suggest that
maintaining the three-agent separation is the most important ar-
chitectural feature, with the supervisory agent providing the most
critical contribution to gap detection.

Learning When Not to Decide: A Framework for Overcoming Factual Presumptuousness in AI Adjudication ICAIL ’26, June 08–12, 2026, Singapore, Singapore
5.9 SPEC with Different Underlying Models
We evaluate SPEC when running on different underlying models to
understand how model capability affects performance on both de-
ferral and determination accuracy. This analysis addresses practical
deployment considerations where agencies must balance perfor-
mance against resource constraints. We implement SPEC with six
models: Claude Sonnet 4.5, Claude Haiku 4.5, Gemini 3, Gemini
2.5, GPT 5.2, and GPT 4.1. Each configuration receives identical
statutory context and case queries. Figure 7 presents results across
model tiers. Sonnet 4.5 demonstrates balanced performance across
both complete and inconclusive cases. Gemini 3 shows a tendency
toward issuing determinations, performing strongly on complete
cases but less reliably identifying information gaps. GPT 5.2 exhibits
more conservative behavior, readily deferring on inconclusive cases
while being more cautious on complete cases.
Figure 7: SPEC performance when running on different un-
derlying foundation models. Models show varying tradeoffs
between complete case and inconclusive case accuracy. Color
indicates provider family.
6 Limitations
We evaluate Colorado UI eligibility, a domain chosen for its statu-
tory complexity, real-world deployment pressures, and the rare
access our collaboration affords to official training materials. While
UI adjudication shares structural features with other administrative
settings—vague standards, incomplete evidence, high stakes for
individuals—extending this work to other jurisdictions and legal
domains would test the boundaries of what we’ve found and reveal
challenges specific to other domains.
Our inconclusive cases were constructed by systematically with-
holding facts from complete scenarios. Real-world incompleteness
is messier: ambiguous rather than absent information, conflicting
accounts, and facts whose legal relevance is itself contested. Test-
ing on naturally occurring cases—drawn from actual adjudication
records—would show whether SPEC’s structured gap detection
handles ambiguous uncertainty as well as absence.
SPEC’s multi-agent architecture increases computational cost
relative to single-pass prompting, a tradeoff worth examining as
these tools move toward production. We did not test interactive use,where adjudicators query the system iteratively and refine their un-
derstanding; such settings may be where explicit gap identification
proves most valuable. Finally, the adjudication guides central to our
approach cannot be publicly released, limiting reproducibility. We
release our benchmark dataset of carefully-constructed questions
and our evaluation framework to support further work.
7 Conclusion
Current AI systems can be presumptuous, issuing confident deter-
minations when critical information is missing. Our evaluation of
four leading platforms on UI eligibility reveals baseline accuracy
averaging 15% on cases that should be deemed inconclusive. In
other words, the AI systems most readily available to UI adjudica-
tors today are extremely likely to provide definitive answers when
the appropriate response is, in fact, “more facts needed.”
Enhanced prompting and other advanced prompting methods
improve gap recognition relative to base models, but often fall
victim to the determination–deferral tradeoff: improvements in the
ability to defer appropriately on incomplete cases come at the direct
expense of increased conservatism (and hence lower accuracy) on
complete cases.
We introduce SPEC, a framework that escapes this tradeoff
through multi-agent analysis requiring explicit identification, enu-
meration, and verification of information gaps before permitting
determinations. SPEC achieves 89% accuracy on both complete and
inconclusive cases—a near-Pareto improvement over the determi-
nation and deferral accuracies of all other methods.
Our findings have direct implications for AI deployment in legal
adjudication. Presumptuousness has persisted in frontier models,
perhaps abetted by the failure of popular AI benchmarks to rec-
ognize and reflect the importance of epistemic humility in many
challenging real-world domains. Many existing legal AI bench-
marks themselves are designed with 100% of tasks resolving to
a determinable outcome. While we instead designed our evalua-
tion to be a roughly 50/50 split between complete and incomplete
cases, a recent study of UI found that real adjudicators chose to
seek additional fact-finding in 87% of claims [ 25], suggesting that
the accuracy gains of SPEC over other methods in a real-world
adjudication setting are likely even greater than what we present
in our study.
SPEC is not architecturally complex; neither is RAG nor Chain-
of-Thought. But it assembles targeted design principles that prove
effective in practice, and we offer it as a practically useful tool for
high-stakes legal reasoning. Our work also informs the longstand-
ing debate around the availability of agency guidance materials.
Such guidance materials may be important to enable responsible
and open AI development and research in legal settings, just as open
data has fostered public innovation. We hope that our benchmark
and SPEC inspire others to address presumptuousness and other
challenges with clear real-world stakes.
Generative AI assistance
We used ChatGPT and Claude for assistance with editing and for
supporting scripting tasks.

ICAIL ’26, June 08–12, 2026, Singapore, Singapore Afane, Robitschek, Ouyang, and Ho
Supplementary Information
Code and data are available at https://github.com/reglab/spec.
References
[1]2023.State of New Hampshire Use of Artificial Intelligence (AI) Technologies Policy.
Technical Report. State of New Hampshire Department of Information Technol-
ogy Office of the Chief Information Officer. https://mm.nh.gov/files/uploads/
doit/documents/sonh-use-of-artificial-intelligence-technologies-policy.pdf
[2]Natalie Alms. 2024. Labor Department experiments with AI in unemployment sys-
tems. https://www.nextgov.com/digital-government/2024/02/labor-department-
experiments-ai-unemployment-systems/394179/
[3]Saar Alon-Barkat and Madalina Busuioc. 2023. Human–AI Interactions in Public
Sector Decision Making: “Automation Bias” and “Selective Adherence” to Algo-
rithmic Advice.Journal of Public Administration Research and Theory33, 1 (Jan.
2023), 153–169. doi:10.1093/jopart/muac007
[4]Kevin D. Ashley. 2017.Artificial Intelligence and Legal Analytics: New Tools for
Law Practice in the Digital Age(1 ed.). Cambridge University Press. doi:10.1017/
9781316761380
[5]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski,
Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr
Nyczyk, and Torsten Hoefler. 2024. Graph of Thoughts: Solving Elaborate Prob-
lems with Large Language Models.Proceedings of the AAAI Conference on Artificial
Intelligence38, 16 (March 2024), 17682–17690. doi:10.1609/aaai.v38i16.29720
[6]Daniel E. Ho, Olivia Martin, Kit Rodolfa, and Amy Perez. 2025. Evaluation as Due
Process: Civil Service in an Automated Age.Administrative Law Review(2025).
[7]Maria De-Arteaga, Riccardo Fogliato, and Alexandra Chouldechova. 2020. A Case
for Humans-in-the-Loop: Decisions in the Presence of Erroneous Algorithmic
Scores. InProceedings of the 2020 CHI Conference on Human Factors in Computing
Systems. ACM, Honolulu HI USA, 1–12. doi:10.1145/3313831.3376638
[8]Shrey Desai and Greg Durrett. 2020. Calibration of Pre-trained Transformers.
doi:10.48550/arXiv.2003.07892
[9]Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch.
2024. Improving factuality and reasoning in language models through multiagent
debate. InProceedings of the 41st International Conference on Machine Learning
(Vienna, Austria)(ICML’24). JMLR.org, Article 467, 31 pages.
[10] David Freeman Engstrom and Daniel E. Ho. 2020. Algorithmic Accountability
in the Administrative State.Yale Journal on Regulation(Jan. 2020). https:
//openyls.law.yale.edu/handle/20.500.13051/8311
[11] Ryan Felton. 2016. Michigan unemployment agency made 20,000 false fraud
accusations – report.The Guardian(Dec. 2016). https://www.theguardian.com/
us-news/2016/dec/18/michigan-unemployment-agency-fraud-accusations
[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Ji-
awei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-Augmented Generation
for Large Language Models: A Survey. doi:10.48550/ARXIV.2312.10997
[13] Yonatan Geifman and Ran El-Yaniv. 2017. Selective Classification for Deep Neural
Networks. doi:10.48550/ARXIV.1705.08500
[14] Yonatan Geifman and Ran El-Yaniv. 2019. SelectiveNet: A Deep Neural Network
with an Integrated Reject Option. doi:10.48550/ARXIV.1901.09192
[15] Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya
Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rock-
more, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan,
Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, Jessica Wu,
Joe Nudell, Joel Niklaus, John Nay, Jonathan H. Choi, Kevin Tobia, Margaret
Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzenberger,
Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer
Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and Zehua Li. 2023. LegalBench:
A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large
Language Models. doi:10.48550/arXiv.2308.11462
[16] Luxi He, Nimra Nadeem, Michel Liao, Howard Chen, Danqi Chen, Mariano-
Florentino Cuéllar, and Peter Henderson. 2025. Statutory Construction and
Interpretation for Artificial Intelligence. doi:10.48550/ARXIV.2509.01186
[17] Dan Hendrycks, Collin Burns, Anya Chen, and Spencer Ball. [n. d.]. CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review. ([n. d.]).
[18] Nils Holzenberger, Andrew Blair-Stanek, and Benjamin Van Durme. 2020. A
Dataset for Statutory Reasoning in Tax Law Entailment and Question Answering.
doi:10.48550/ARXIV.2005.05257
[19] Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025.
AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions. doi:10.
48550/ARXIV.2506.09038
[20] Yuta Koreeda and Christopher D. Manning. 2021. ContractNLI: A Dataset for
Document-level Natural Language Inference for Contracts. doi:10.48550/arXiv.
2110.01799
[21] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. doi:10.48550/ARXIV.2005.11401[22] Belinda Z. Li, Been Kim, and Zi Wang. 2025. QuestBench: Can LLMs ask the
right question to acquire information in reasoning tasks? doi:10.48550/ARXIV.
2503.22674
[23] Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S. Ilgen, Emma
Pierson, Pang Wei Koh, and Yulia Tsvetkov. 2024. MediQ: Question-Asking LLMs
and a Benchmark for Reliable Interactive Clinical Reasoning. doi:10.48550/ARXIV.
2406.00922
[24] Sravanthi Machcha, Sushrita Yerra, Sahil Gupta, Aishwarya Sahoo, Sharmin
Sultana, Hong Yu, and Zonghai Yao. 2026. Knowing When to Abstain: Medical
LLMs Under Clinical Uncertainty. doi:10.48550/ARXIV.2601.12471
[25] Varun Magesh, Olivia H Martin, Faiz Surani, Amy Perez, Kit Rodolfa, and Daniel E
Ho. 2026. Evaluating Generative AI in Benefits Administration: A Demonstration
Project.Symposium on Computer Science and Law (CSLAW ’26)(2026). https:
//dho.stanford.edu/wp-content/uploads/CDLE.pdf
[26] Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Man-
ning, and Daniel E. Ho. 2024. Hallucination-Free? Assessing the Reliability of
Leading AI Legal Research Tools. doi:10.48550/arXiv.2405.20362
[27] Lisa Larrimore Ouellette, Amy R. Motomura, Jason Reinecke, and Jonathan S.
Masur. 2025. Can AI Hold Office Hours?Journal of Legal Education (forthcoming)
(Feb. 2025). doi:10.2139/ssrn.5166938
[28] Jennifer Pahlka. 2023.Recoding America: Why Government Is Failing in the Digital
Age and How We Can Do Better. Henry Holt and Company.
[29] Kusuma Raditya and Pantho Sayed. 2025. Retrieval-augmented gen-
eration (RAG): Towards a promising LLM architecture for legal work?
https://jolt.law.harvard.edu/digest/retrieval-augmented-generation-rag-
towards-a-promising-llm-architecture-for-legal-work
[30] Maohao Shen, Subhro Das, Kristjan Greenewald, Prasanna Sattigeri, Gregory
Wornell, and Soumya Ghosh. 2024. Thermometer: Towards Universal Calibration
for Large Language Models. doi:10.48550/arXiv.2403.08819
[31] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik
Narasimhan, and Shunyu Yao. 2023. Reflexion: Language Agents with Verbal
Reinforcement Learning. doi:10.48550/ARXIV.2303.11366
[32] Tejas Srinivasan, Jack Hessel, Tanmay Gupta, Bill Yuchen Lin, Yejin Choi, Jesse
Thomason, and Khyathi Raghavi Chandu. 2024. Selective "Selective Prediction":
Reducing Unnecessary Abstention in Vision-Language Reasoning. doi:10.48550/
ARXIV.2402.15610
[33] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang,
Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain
of Thought Reasoning in Language Models. doi:10.48550/arXiv.2203.11171
[34] Zora Zhiruo Wang, Yijia Shao, Omar Shaikh, Daniel Fried, Graham Neubig, and
Diyi Yang. 2025. How Do AI Agents Do Human Work? Comparing AI and Human
Workflows Across Diverse Occupations. doi:10.48550/arXiv.2510.22780
[35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, and others. 2022. Chain-of-thought prompting elicits
reasoning in large language models.Advances in neural information processing
systems35 (2022), 24824–24837.
[36] Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe,
and Lucy Lu Wang. 2025. Know Your Limits: A Survey of Abstention in Large
Language Models.Transactions of the Association for Computational Linguistics
13 (June 2025), 529–556. doi:10.1162/tacl_a_00754
[37] Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stew-
art Massie, Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, and Bruno
Fleisch. 2024. CBR-RAG: Case-Based Reasoning for Retrieval Augmented Gener-
ation in LLMs for Legal Question Answering. doi:10.48550/arXiv.2404.04302
[38] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and
Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate Problem Solving with
Large Language Models. InAdvances in Neural Information Processing Systems,
Vol. 36. Curran Associates, Inc., 11809–11822.
[39] Fanghua Ye, Mingming Yang, Jianhui Pang, Longyue Wang, Derek F. Wong,
Emine Yilmaz, Shuming Shi, and Zhaopeng Tu. 2024. Benchmarking LLMs via
Uncertainty Quantification.Advances in Neural Information Processing Systems
37 (Dec. 2024), 15356–15385. doi:10.52202/079017-0491
[40] Matthew M. Young, Johannes Himmelreich, Danylo Honcharov, and Sucheta
Soundarajan. 2022. Using Artificial Intelligence to Identify Administrative Errors
in Unemployment Insurance.Government Information Quarterly39, 4 (Oct. 2022),
101758. doi:10.1016/j.giq.2022.101758
[41] Jesus Manuel Niebla Zatarain. 2018. Book Review: Artificial Intelligence and
Legal Analytics: New Tools for Law Practice in the Digital Age.Script-ed15, 1
(Aug. 2018), 156–161. doi:10.2966/scrip.150118.156
[42] Lucia Zheng, Neel Guha, Brandon R. Anderson, Peter Henderson, and Daniel E.
Ho. 2021. When Does Pretraining Help? Assessing Self-Supervised Learning for
Law and the CaseHOLD Dataset. doi:10.48550/arXiv.2104.08671
[43] Zhanke Zhou, Xiao Feng, Zhaocheng Zhu, Jiangchao Yao, Sanmi Koyejo, and
Bo Han. 2025. From Passive to Active Reasoning: Can Large Language Models
Ask the Right Questions under Incomplete Information? doi:10.48550/arXiv.2506.
08295