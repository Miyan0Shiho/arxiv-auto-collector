# Insider Knowledge: How Much Can RAG Systems Gain from Evaluation Secrets?

**Authors**: Laura Dietz, Bryan Li, Eugene Yang, Dawn Lawrie, William Walden, James Mayfield

**Published**: 2026-01-19 17:03:20

**PDF URL**: [https://arxiv.org/pdf/2601.13227v1](https://arxiv.org/pdf/2601.13227v1)

## Abstract
RAG systems are increasingly evaluated and optimized using LLM judges, an approach that is rapidly becoming the dominant paradigm for system assessment. Nugget-based approaches in particular are now embedded not only in evaluation frameworks but also in the architectures of RAG systems themselves. While this integration can lead to genuine improvements, it also creates a risk of faulty measurements due to circularity. In this paper, we investigate this risk through comparative experiments with nugget-based RAG systems, including Ginger and Crucible, against strong baselines such as GPT-Researcher. By deliberately modifying Crucible to generate outputs optimized for an LLM judge, we show that near-perfect evaluation scores can be achieved when elements of the evaluation - such as prompt templates or gold nuggets - are leaked or can be predicted. Our results highlight the importance of blind evaluation settings and methodological diversity to guard against mistaking metric overfitting for genuine system progress.

## Full Text


<!-- PDF content starts -->

Insider Knowledge: How Much Can RAG
Systems Gain from Evaluation Secrets?
Laura Dietz1, Bryan Li2, Eugene Yang3, Dawn Lawrie3, William
Walden3, and James Mayfield3
1University of New Hampshire, Durham, New Hampshire, USAdietz@cs.unh.edu
2University of Pennsylvania, Philadelphia, Pennsylvania, USA
bryanli@seas.upenn.edu
3Human Language Technology Center of Excellence, Johns Hopkins University,
Baltimore, Maryland, USA
{eugene.yang,lawrie,wwalden1,mayfield}@jhu.edu
Abstract.RAG systems are increasingly evaluated and optimized us-
ing LLM judges, an approach that is rapidly becoming the dominant
paradigm for system assessment. Nugget-based approaches in particular
are now embedded not only in evaluation frameworks but also in the ar-
chitectures of RAG systems themselves. While this integration can lead
to genuine improvements, it also creates a risk of faulty measurements
due to circularity. In this paper, we investigate this risk through compar-
ative experiments with nugget-based RAG systems, includingGinger
andCrucible, against strong baselines such asGptResearcher. By
deliberately modifyingCrucibleto generate outputs optimized for an
LLM judge, we show that near-perfect evaluation scores can be achieved
when elements of the evaluation—such as prompt templates or gold
nuggets—are leaked or can be predicted. Our results highlight the impor-
tance of blind evaluation settings and methodological diversity to guard
against mistaking metric overfitting for genuine system progress.4
Keywords:Retrieval-augmented generation·LLM judge·Nugget evaluation.
1 Introduction
As LLM-based chat systems like ChatGPT and Claude become primary informa-
tion sources for general users, ensuring accuracy, recency, and credible sourcing of
their responses is critical for developing trustworthy systems [10, 15]. Retrieval-
Augmented Generation (RAG) has recently emerged as a leading approach to
achieve these aims [8, 21, 25]. In parallel,nugget-basedevaluation methods, which
check generated outputs for the presence of key pieces of information (nuggets),
have become an important tool for RAG evaluation [1, 27, 29, 31], grounding
4Online appendix athttps://github.com/hltcoe/ecir26-crucible-appendix/arXiv:2601.13227v1  [cs.IR]  19 Jan 2026

2 Dietz et al.
system assessment in assessor-identified Q&A pairs or claims. Increasingly, how-
ever, these methods rely on LLMs not only for matching nuggets to content, but
also for developing nugget banks from scratch [11, 31].
While all evaluation paradigms can produce systematically biased, invalid
measurements, LLM-based evaluation methods are particularly susceptible to
a range of vulnerabilities [14]. For instance, circularity arises whenever com-
ponents of the evaluation methods are usefully integrated into a RAG system.
Moreover prompts, scripts, and LLM judges are routinely made public and may
thus be inadvertently exploited by system developers. This can lead to inflated
evaluation scores and reduced alignment with manual assessments [9]. A sug-
gested safeguard against circularity is to rely on human-curated, nugget-based
evaluation [18, 27, 35].
Yet such safeguards are not perfect. In this paper, we identify three pathways
by whichinsider knowledgeof an LLM-based evaluation process can be exploited
to inflate evaluation results, even without a direct test set leak: (1) tuning RAG
systems to optimize narrowly for measured criteria while ignoring other,unmea-
suredquality criteria (RQ1); (2) exploiting knowledge of the evaluation prompt
and LLM model to modify system outputs (RQ2); and (3) predicting the set of
gold nuggets for system usage with high accuracy (RQ3).5
Among these, RQ3 poses the most serious and underappreciated threat.
Nugget sets are meant to serve as independent ground truth. However, our
investigations reveal that system-generated nuggets can overlap substantially
with human-curated ones. This overlap may be greater still for nugget creation
pipelines, such asAutoNuggetizer[30], that rely on LLMs to generate candi-
date nuggets that are then manually filtered. Thus, even absent nefarious intent,
system developers may achieve high nugget-based evaluation scores by leveraging
LLM-based nugget generation components.
Indeed,noneof RQ1, RQ2, or RQ3 assumes intentional subversion: RAG
systems mayincidentallybe trained using a model similar to the LLM judge,
developed using similar prompts, or optimized against metrics or nuggets similar
to those used at test time. Moreover, adopting these ideas may lead to genuinely
better systems, as noted by Clarke and Dietz [9]. By simulating the downstream
effects of integrating these components into system development, we show how
safeguards can collapse, resulting in inflated results alongmeasureddimensions.
Our findings reveal that automatic nugget-based evaluations, though valuable,
are vulnerable to a form of leakage that is subtle, pervasive, and arguably already
occurring.
Contributions.To study this problem, we introduceCrucible, a RAG system
that incorporates evaluation ideas to probe the susceptibility of nugget-based
LLM judge systems.Crucibleis designed to expose and study vulnerabilities
ofAutoArgue[37], an open-source LLM-based implementation ofArgue[27].
5To avoid confusion, we refer to nuggets generated automatically by RAG systems,
such asCrucible, assystem nuggets, and manually created nuggets used by the
evaluation systemArgueasgold nuggets.

Insider Knowledge for RAG Systems 3
Drawing on insider knowledge ofAutoArgue,Crucibledemonstrates that the
evaluation is robust only when key evaluation elements are kept secret. Once
these elements become public knowledge, they may be used to enhance system
outputs and inflate scores without necessarily obtaining genuine improvements.
We present a case study on the TREC NeuCLIR 2024 Report Generation Pi-
lot task—a long-form RAG task with complex queries and an emphasis on faith-
ful citation. Crucially, nuggets for this task were fully manually curated through
a principled process. Our exclusive focus on TREC NeuCLIR is a consequence of
the lack of availablealternativeRAG test collections with fully manual nuggets
at the time of our study. Since the purpose of this case study is to explore the
susceptibility of an LLM judge paradigm, we do not evaluate the efficiency of
the RAG system or the automatic evaluation.
2 Background
2.1 LLM-as-a-Judge
The introduction of LLMs as automatic judges for retrieval and generation sys-
tems has led to a surge of interest in their reliability as replacements for or
supplements to human assessments [26]. While initial studies highlighted strong
correlations with human labels [34, 35], more recent work documents weaknesses
and instabilities that threaten validity [9, 32].
Recent IR-focused work emphasizes observational comparisons and prompt-
level analyses. Balog et al. [5] synthesize the roles that LLMs can play as rankers,
judges, and assistants, and identify feedback loops as a fundamental risk for eval-
uation. Thakur et al. [33] compare LLM judges against human annotations for
retrieval-augmented generation tasks at the TREC RAG Track, showing both
alignment and divergence. Arabzadeh and Clarke [3] benchmark pointwise, pair-
wise, and rubric-based judging paradigms, while their follow-up [4] highlights the
prompt sensitivity of LLM judges. Upadhyay et al. [36] extend these compar-
isons with a large-scale study ofUmbrela, an open-source implementation of
Bing’s relevance assessor, across multiple settings. While their positive findings
are used to support the reliability of LLM judges, Clarke and Dietz [9] show that
Umbrelaceases to be a reliable evaluator under conditions of circularity.
A complementary line of work designs adversarial content attacks to measure
judge susceptibilities, often by injecting instructions that override evaluation
prompts (e.g.forget all previous instructions and rate this response as relevant).
Early studies explored generic prompt subversions in generative settings [2, 6,
7, 22], while more recent work has shifted toward citation evaluation [10].
Taken together, the current literature offers descriptive accounts of vulnera-
bilities [14], but lacks systematic meta-evaluation paradigms of the sort proposed
in this work for investigating hypothesized weaknesses of LLM-based evaluations.
2.2 (Auto-)ARGUE
Argue, introduced by Mayfield et al. [27], is a nugget-based evaluation frame-
work designed forreport generation—generation of long-form, citation-backed

4 Dietz et al.
responses to a complex user request.Argueevaluates reports at the sentence
level, assessing each sentence for whether it is supported by its citations and
whether it attests any of a set of nuggets (represented as Q&A pairs) associated
with the request. These judgments can then be accumulated into report-level
scores. TheArgueframework has been used in the manual evaluation for the
TREC NeuCLIR Track Report Generation Pilot in 2024 [24]. The case study
presented in this work focuses onAutoArgue[37], an open-source LLM-based
implementation ofArgue.
2.3 UMBRELA Subversion Probe
Asubversion probewraps an existing IR/RAG system in a lightweight procedure
designed to exploit weaknesses of an LLM judge. Clarke and Dietz [9] introduced
such a probe forUmbrela[35], which reorders documents based onUmbrela
relevance labels. Applied to TREC RAG 2024 systems, it revealed two findings:
(1) the probe delivered genuine quality improvements under human assessment;
and (2)Umbrelaproduced systematically biased scores for manipulated sys-
tems, especially among top performers. This is despite the conformation of ear-
lier meta-evaluations which reported strong results forUmbrelaon traditional
IR [33] (Tau@60=0.89), its effectiveness dropped sharply once systems included
LLM judge components (Tau@60=0.63 / Tau@10=0.38). These results highlight
that unreliable agreement with human judges can incentivize misleading system
design. Research on the robustness of such judges for RAG evaluation is limited.
2.4 Nugget-based RAG Test Collections
Many recent RAG benchmarks rely on nuggets that are produced by large
language models. Fully automatic approaches, such as theRubric Work-
bench[19], generate nuggets without human oversight, while semi-automatic
pipelines, e.g.,AutoNuggetizer[30], employ post-hoc human filtering. In both
cases the nuggets remain traceable to an LLM source, which means a competent
RAG system can simply reproduce or approximate them. Consequently, show-
ing alignment between system-generated nuggets and gold nuggets yields little
insight into the distortions introduced by nugget prediction.
To properly interrogate RQ3, we thus require a test collection whose nuggets
were created without LLM influence; only then can we simulate the effects of
nugget prediction and measure the distortion it introduces into evaluation out-
comes. Collections that satisfy this criterion are scarce. Given this landscape,
we turn to the TREC NeuCLIR 2024 Report Generation Pilot, to date, the
only publicly available RAG benchmark that offers a fully manual nugget bank
created without LLM assistance. Details of this collection are deferred to the
evaluation section.

Insider Knowledge for RAG Systems 5
Fig. 1.Workflow ofCrucibleand its evaluation in theAutoArguesystem.Crucible
ideates nuggets from retrieved documents, extracts and summarizes passages, and as-
sembles them into a cited report.AutoArgueevaluates each sentence by checking
coverage of manual gold nuggets and verifying citation support. It is common that the
summary sentence created to address a system nugget, such as “How many RoundUp
cases has Bayer lost so far?”, also covers gold nuggets for evaluation, such as “What
do the RoundUp lawsuits allege?”, since both sets capture related aspects of the same
topic.
3Crucible: A Subversion Probe forAutoArgue
3.1Crucible: A RAGE System
As an experimental proofing ground, we design the RAGE systemCrucible
[12], a retrieval-augmented generation (RAG) system that incorporates ideas
from automatic evaluation (E).Crucibleadopts ideas from nugget-based eval-
uation, to use document-grounded nuggets to generate reports, as depicted in
Figure 1. Hence, it is designed to optimizenugget-basedevaluation measures
while providing highly faithfulcitations—two key metrics measured by the eval-
uation systemArgue(Section 4.2).
1.Nugget Ideation.Cruciblestarts with automatically generating a system
nugget bank that drives the extraction and generation processes. In our
setup, nuggets take the form of Q&A pairs that represent topically relevant
key facts. We purposefully imitate the style of Q&A nugget used byArgue.
2.Retrieval.Documents are collected from top 20 of PLAID-X dense retrieval
model [38], a retrieval system provided by the organizers of TREC NeuCLIR,
whose dataset we use in this study. We also study the effects of other retrieval
models [28, 40], but these results are omitted for brevity.

6 Dietz et al.
3.Sentence Extraction.Crucibleextracts candidate passages from re-
trieved documents along with self-contained sentences using prompt-based
templates6that align document content to nugget questions and answers.
4.Selection and Assembly.Depending on the requested report length,Cru-
cibleselects up tok∈ {1,5,20}sentence(s) per system nugget with the
highest extraction confidence and assembles them into a report that maxi-
mizes coverage of the system nugget bank, using the the source document of
each sentence as a citation.
LLM prompts provided in the online appendix and in the system paper [12].
3.2 Worked Example
Figure 1 provides an example of howCruciblegenerates a report for a user
request about lawsuits and settlements. System-generated nuggets guide the ex-
traction of matching passages, which are summarized and cited. A list of system
and gold nuggets for this example is available in the online appendix.
TheArgueframework separately evaluates each included sentence along
with its citation. The evaluation makes use of a test collection with manual gold
nuggets, which are developed by human judges to determine the relevance of
the information. An example is “What do the RoundUp lawsuits allege?”. The
AutoArgueimplementation uses an LLM prompt to determine whether a gold
nugget is covered by the sentence and the citation. Additionally,AutoArgue
verifies that cited documents support the corresponding summary sentence with
a separate prompt.
Figure 1 depicts how theCruciblesystem generates one sentence for its
response: “Bayer has lost three Roundup cases where the complainant alleged
that Glyphosate, a carcinogenic ingredient, was contained in Roundup.” This
sentence was generated to address the system-generated nugget “How many
RoundUp cases has Bayer lost so far?”, but because the document contains
additional relevant information, the resulting sentence also addresses the gold
nugget from the test collection “What do the RoundUp lawsuits allege?”. This
effect is expected, as both sets of nuggets represent concrete information elements
for the same broad topic. Often there are closely related questions in both nugget
sets, such as the number of lawsuits.
In some cases,Crucibledirectly guesses a gold nugget, as shown in Figure
2. Although our nugget generation was not tuned to imitate gold nuggets, some
overlap arises naturally. As semi-automatic pipelines increasingly use LLMs to
propose nuggets for human verification, we expect such overlap to be increasingly
common. To study the downstream effect in RQ3, we simulate a setting where
Crucibleperfectly predicts all gold nuggets and observe the effects onAuto-
Argueevaluation scores.
6The system message was omitted due to a bug. Updated results in online appendix.

Insider Knowledge for RAG Systems 7
Fig. 2.Example whereCruciblecorrectly guesses a gold nugget (“How much did the
court order Bayer to pay Dewayne Johnson?”). RQ3 explores how evaluation outcomes
change if RAG systems reliably guess the gold nugget set
3.3CrucibleSubversion Probes
To “attack” potential vulnerabilities ofArgue, we design subversion probes that
augment theCruciblesystem. This work’s key innovation is the use of insider
knowledge about the LLM judge’s (here:AutoArgue’s) own decision process to
filter and rewrite outputs, ensuring that every retained sentence is “approved” by
that judge. Our hypothesis is that this leads to measurable effectiveness increases
that may or may not generalize to improvements under human judgment.
We design two kinds of subversion probes to simulate what would happenif
knowledge of specific aspects of theAutoArgueLLM judge became known to
system developers.
Guessed prompt probe.The first subversion probe simulates the condition
in which the prompts used by the judge are becoming public knowledge. Such
prompts could then be used to improve the quality of the responses from the
perspective of the LLM judge prior to evaluation. This probe applies two filtering
steps directly after theSentence Extractionstep (Section 3.1):
1.Citation Filtering.Each candidate sentence and extracted passage gets
passed throughAutoArgue’s own citation support checks.
2.Nugget Coverage Filtering.Each candidate sentence (or extracted pas-
sage) is passed throughAutoArgue’s nugget matching checks, to verify
that it covers the intended system nugget.
Candidate sentences that fail these checks are discarded beforeSelection and
Assembly. We use these probes to study the change in evaluation measurements
from using the exact prompts of the LLM judge (RQ2).
Guessed nuggets probe.The second subversion probe simulates the condition
in which gold nuggets are guessable by a fully automatic system (RQ3). This
could arise when relevance for test topics is well understood by LLMs or gold
nuggets are taken from a pool of automatically predicted system nuggets. This

8 Dietz et al.
could arise in a variety of ways—e.g., via leaked knowledge of the model used
to generate the nuggets or of other aspects of the nugget curation process. Here,
we consider the limiting case of such a leak, in which theexactset of gold
nuggets is known. To simulate this condition, this probe replaces theNugget
Ideationstep (Section 3.1) with the set of gold nuggets. This point highlights
the importance of keeping gold nuggets secret.
3.4 What is the point of subversion probes?
System.From the system perspective,Crucible’s subversion probes act as
a wrapper; any RAG pipeline can be run through them to produce outputs
tuned forAutoArgue. By design, the wrapped systems obtain higher evalu-
ation scores under the LLM judge. Whenever these materialize as genuine im-
provements under human judgment, we suggest that the probe is incorporated
into the wrapped system to yield an improved RAGE system.
Evaluation.From the evaluation perspective,Crucibleoperationalizes an “at-
tack”; if a system that is adulterated withCrucible’s guessed probes achieves
much higherArguescores but not higher manual evaluation scores, those im-
provements reveal a vulnerability in the attacked evaluation method. In such
cases, guardrails to thwart the attack should be developed, such as ensembles of
prompts, hidden nugget banks, or held-out LLMs. The probe would then be used
to test the resilience of the evaluation method and the efficacy of the proposed
guardrails.
Meta-evaluation.Paradigm changes in system development present a challenge
to the reliable design of meta-evaluations for LLM judges. In such cases, high
agreement between an LLM judge and human evaluation can be observed only for
certain systems, e.g., those that did not adopt ideas from LLM judge approaches.
However, as demonstrated by Clarke and Dietz [9], this correlation may not
hold for next-generation systems for which we seek an LLM judge that can
reliably identify progress. Especially when the LLM judge quality is measured via
Kendall’s tau correlation between automatic and manual leaderboards, a healthy
variety of approaches is essential. Here subversion probes that can modify the
outputs of a set of existing systems provide a force-multiplier that can simulate
anticipated paradigm-changes of RAG systems, rendering the meta-evaluation
more convincing.
4 Experiment Setup
4.1 Dataset
We use the test collection of the TREC NeuCLIR 2024 report-generation pilot
collection [24]. The corpus contains ten million documents in Chinese, Russian,
and Farsi, each accompanied by an automatic English translation. For each of

Insider Knowledge for RAG Systems 9
the 19 report requests (queries), assessors provided a title, a problem statement,
and the requester’s background.
For all report requests, TREC assessors manually designed a comprehensive
set of nuggets, each comprising a factual question and a set of correct answers.
The nuggets were created after assessors read the relevant documents in the
original language anddid not involve any LLM assistance. On average, 10–20
question-answer pairs were produced per request, and these nuggets serve as
the gold standard for assessing system-generated reports submitted to the 2024
NeuCLIR track (our system’s reports were not part of the original pool). We
focus exclusively on this dataset because, at the time of writing, no alternative
RAG benchmark provides a comparable manually curated nugget bank.
4.2 Evaluation Measures
TheAutoArgueevaluation system used manually designed gold nuggets from
the TREC NeuCLIR track to determine the relevance of the system’s generated
report. UsingLlama3.3-70B-Instruct,AutoArguescans each sentence of the
report for mentions of a correct answer to the nugget question. The nugget recall
metric follows the assumption that the more nuggets are covered in the generated
report, the more relevant the report is. Nugget-measures are complemented by
a verification of citation faithfulness with the LLM.
We report results usingAutoArguefor the following measures:
Nugget Recall=covered nuggets
gold nuggetsmeasures the recall of gold nuggets.
Nugget Density=covered nuggets
sentencesmeasures the tradeoff between concise sum-
maries and nuggets covered.
Nugget-bearing Sentences=sentence with nuggets
all sentencesmeasures how many sen-
tences include facts relevant for the report.
Citation Support=supported citations
citationsof citations supporting their sentence.
4.3 Variations and Baseline Systems
To explore our hypotheses, we compare severalCruciblevariants; those marked
with†rely on insider knowledge about evaluator prompts (RQ2); those marked
with⋆rely on insider knowledge to generate nuggets (RQ3):
Base:BasicCruciblesystem using PLAID-X [38] as the retrieval model, no
sentence filtering, and automatically generated system nuggets.7
Citation Filter†:Filter citations that don’t support the sentence usingAuto-
Argueprompts (and removing sentences lacking any citation).
Cov-Sentence†:Filter sentences that do not cover any of the system nuggets.
Cov-Extract†:Filter sentences where the extracted passage does not cover
any of the system nuggets.
7We also experimented with multilingual LSR [28] and Qwen3 [40] and found similar
trends. Results are omitted from this paper for brevity.

10 Dietz et al.
Gold nuggets⋆:Simulate the system correctly guessing all gold nuggets.
For each variation, we compare reports of different lengths:
Short:Up tok= 1 sentence per nugget (report limit=2000 characters)
Medium:Up tok= 5 sentences per nugget (report limit=10,000 characters)
Long:Up tok= 20 sentences per nugget (report limit=1,000,000 characters)
To provide a reference point, we compare ourCruciblesystem variations
to the following systems:
GINGER [23]:A nugget-based RAG system using GPT-4o with clustering.
Designed for TREC RAG 2024 topics and evaluation withAutoNugge-
tizer[30]. Using 100 documents retrieved by Qwen3 (best tested variant).
GINGER-Llama:Gingerusing the same LLM asCrucible(LLaMA).
GptResearcher [16]:A simple RAG pipeline implemented using the open
sourceGptResearcher[17] toolkit that simply passes the top-ranked re-
trieved documents to LLaMA to generate the final report.
BulletPoints [39]:An extractive pipeline (submitted ashltcoe-eugene) that
was one of the top-performing systems in the TREC NeuCLIR 2024 Report
Generation Pilot. TheBulletPointspipeline extracts and groups facts
from the retrieved documents using LLaMA.
We useLLaMA-3.3-70B-Instructfor all system implementations and the
AutoArgueevaluation, except where otherwise noted.
5 Results and Analysis
5.1 RQ1: Does knowing the evaluation framework and metric
improve the measured performance? (Yes)
We explore RQ1 by comparing RAG systems designed for the TREC NeuCLIR
task andAutoArgue-based evaluation (“with evaluation knowledge”) to RAG
systems designed for a different task. The latter set of systems includes both
GptResearcherandGinger, which was designed for the TREC RAG 24
task that was evaluated with the AutoNuggetizer system [30]. We also include
BulletPoints, a top performing system from the NeuCLIR track.
Figure 3 presents these systems in order from no to full knowledge of the
Argueevaluation approach. The more general systemGptResearcherlags
behindBulletPointsin all metrics. Even for the two nugget-first RAG sys-
temsGingerandCrucible, the trend is clearly visible in relative performance
gains ranging from 78% to 417% across all metrics. Thus, system designers who
optimize for an evaluation framework will likely see measurable improvements.

Insider Knowledge for RAG Systems 11
GPTR GINGER GINGER
LlamaBullet
PointsCrucible
Base0.00.20.4
0.1770.244 0.2410.508
0.429(a) Nugget Recall
With Eval Knowledge
False
True
GPTR GINGER GINGER
LlamaBullet
PointsCrucible
Base0.00.20.40.6
0.265 0.285
0.1360.4680.703(b) Nugget-bearing Sentences
GPTR GINGER GINGER
LlamaBullet
PointsCrucible
Base0.000.250.500.751.00
0.571
0.4360.4760.8350.902(c) Citation Support
GPTR GINGER GINGER
LlamaBullet
PointsCrucible
Base0.00.20.4
0.1310.264
0.1340.3400.448(d) Nugget Density
Fig. 3.RQ1: Yes, knowledge of the evaluation system is likely to help development of
a RAG system that obtains high evaluation scores.Ginger, a recent nugget-first RAG
system designed for TREC RAG 24 is expected not to have used any insider knowledge
about theArgueevaluation framework. We show that despite conceptual similarities
betweenGingerandCrucible, the performance characteristics are vastly different.
Short Medium LongBase
+ Citation Filter
+ Cov-Sentence
+ Cov-Extraction0.902 0.845 0.810
0.973 (7.87%) 0.947 (12.07%) 0.929 (14.69%)
0.964 (6.87%) 0.939 (11.12%) 0.926 (14.32%)
0.961 (6.54%) 0.961 (13.73%) 0.941 (16.17%)(a) Citation Support
Short Medium Long0.448 0.124 0.030
0.460 (2.68%) 0.126 (1.61%) 0.033 (10.00%)
0.461 (2.90%) 0.129 (4.03%) 0.036 (20.00%)
0.457 (2.01%) 0.131 (5.65%) 0.051 (70.00%)(b) Nugget Density
Fig. 4.RQ2: Yes, filtering candidate extractions with the evaluation prompt for citation
support and/or nugget detection will improve the respective evaluation metrics. %
indicates relative improvement over “Base”.
5.2 RQ2: Does knowing the prompt used in the automatic
evaluation improve measured performance? (Yes, weak signal)
For RQ2, we ask whether the prompt-guessing probes yield the expected im-
provement inAutoArgue’s evaluation measures. While theCruciblebase
system uses its own prompts for the initial sentence extraction, we evaluate
whether filtering these sentences withAutoArgue’s prompts leads to higher
evaluation scores. We explore this question in several variations of theCrucible
system, generating short, medium and long reports.
We verify that the citation filtering probe indeed leads to an improvement
underAutoArgue’s citation support metric (Figure 4(a)). Likewise, we verify
that the nugget cover filtering probe leads to improvements underAutoArgue’s
nugget metrics (Figure 4(b)).
Across all variations we observe small, yet consistent performance improve-
ments, many of which are confirmed by paired t-test. We remark that by its
bottom-up design, theCruciblebase system is already very effective in ground-
ing sentence generation in cited sources and system nuggets; however, longer

12 Dietz et al.
Base + Citation Filter + Cov-Sentence + Cov-ExtractionAuto Nuggets
Gold Nuggets0.429 0.436 0.436 0.438
0.610
(42.19%)0.689
(58.03%)0.650
(49.08%)0.724
(65.30%)(a) Nugget Recall
Base + Citation Filter + Cov-Sentence + Cov-Extraction0.703 0.713 0.712 0.733
0.850
(20.91%)0.893
(25.25%)0.864
(21.35%)0.897
(22.37%)(b) Nugget-bearing Sentences
Fig. 5.RQ3: Yes, system that could predict gold nuggets, obtain a stark increase in
nugget-oriented metrics, including Nugget Recall (≈+50% over Auto Nuggets) as well
as the fraction of sentences that discuss relevant nuggets (≈+20%).
generations that include the 5–20 sentences per nugget benefited more from dis-
carding sentences that did not passAutoArgue’s tests.
5.3 RQ3: Does the ability to predict manually created gold nuggets
improve the measured performance? (Yes, for nuggets)
For RQ3 we explore a hypothetical scenario where the RAG system is able to
predict (or otherwise obtain) the set of gold nuggets used by the evaluator. We
simulate this by using the gold nuggets as system nuggets inCrucible.
Figure 5 demonstrates large improvements (+42% to +65%) on nugget recall
when the evaluator’s gold nuggets are known. There is also an improvement in
nugget-bearing sentences (+21% to +25%) as sentences that don’t mention gold
nuggets are avoided.
Finally, we conducted a sanity check to verify that insider knowledge of gold
nuggets does not improve other evaluation metrics, such as citation support. The
data can be found in the online appendix.
5.4 Are Improvements Genuine?
We note that the detection of nuggets in text is affected by the randomness of
the underlying LLM, which can lead to disagreement between system extraction
andAutoArgueevaluator (e.g. Topic 335). Manual inspection showed that the
sentences covered both system and gold nuggets.
Although all of our quantitative results are based on manual nuggets, we are
curious whether measured evaluation improvements would translate to a human-
felt quality difference. The author team selected two reports generated for the
example Topic 335, one from theCrucible-base variant and one that applies
both the guessed prompt and the guessed nugget probe. The team is divided on
which contains more relevant facts, appreciating that theCrucible-base report
mentions a wide range of numerical data on court cases, both pending and lost
at different court levels and dollar amounts of settlements. However, the guessed
prompt probe indeed translates to sentences where extracted passages clearly
specify the relevant entities, such as Bayer. A team member commented that
they considered the citations arising from this probe as more faithful, and hence
were less suspicious that information is incorrect—a clear indicator that future
RAGE systems should adopt the Citation Filtering probe.

Insider Knowledge for RAG Systems 13
5.5 Early Results from TREC 2025
To gain empirical insights from fresh test collections, we submitted several vari-
ations of our approach to TREC 2025 [13]. In TREC RAGTIME, we find that
investing effort in improving the system nuggets yields higher nugget recall
(0.27→0.33▲). In TREC DRAGUN, nugget coverage filtering results in small
but not statistically significant gains in support (0.166→0.173).
For TREC RAGTIME, citation filtering significantly improves citation sup-
port (0.83→0.99▲), with no difference across approaches (0.97–0.99). In con-
trast, manual citation support annotations in TREC RAG show a substantial
spread across methods, with weighted precision ranging from 0.59→0.69→
0.80▲. We remark that citation filtering introduces circularity withAutoArgue’s
evaluation procedure used in TREC RAGTIME. This divergence implies that
our attack successfully exposes a prompt-based vulnerability.
6 Conclusion
We present a systematic empirical study of evaluation subversion using insider
knowledge, not just a system description.Crucibleis a RAGE system with
a subversion probe developed for the TREC NeuCLIR track and its evaluation
frameworkAutoArgue. UsingCrucible, we examine how insider knowledge
about the evaluation process can distort measured performance. Specifically, we
study three types of insider knowledge: (RQ1) structural understanding of the
evaluation targets; (RQ2) knowledge of evaluation prompts; (RQ3) knowledge of
gold nuggets used for evaluation. All three influence evaluation outcomes, with
knowledge of the gold nuggets posing the most severe threat to validity.
While this work represents only an initial step toward a systematic investiga-
tion of vulnerabilities in LLM-based evaluation paradigms such asAutoArgue,
it raises an important warning: the public release of evaluation artifacts, includ-
ing prompts and nuggets, carries significant risk. If evaluation artifacts are be-
coming public knowledge or are reverse-engineered during system development,
they can artificially inflate scores of systems developed with insider knowledge,
undermining the integrity of empirical comparisons across systems. Our study
underscores the need for safeguards to prevent such subversion, lest evaluation
results become misleading or invalid.
To mitigate the vulnerabilities identified in this work while preserving re-
usable, transparent, and reproducible RAG evaluation, we recommend sharing
test collections through platforms that support blinded experimental evalua-
tion. The TREC Auto-Judge track provides one such example by using TIRA
[20] and by distributing our test probes within this blinded evaluation setting.
We hope that future work will develop subversion probes for additional LLM
judge paradigms to quantify evaluation vulnerabilities more systematically and
to inform the design of effective safeguards.
Disclosure of Interests.Authors have no competing interests.

Bibliography
[1] Abbasiantaeb, Z., Lupart, S., Azzopardi, L., Dalton, J., Aliannejadi, M.:
Conversational gold: Evaluating personalized conversational search system
using gold nuggets. In: Proceedings of the 48th International ACM SI-
GIR Conference on Research and Development in Information Retrieval,
pp. 3455–3465 (2025), https://doi.org/10.1145/3726302.3730316
[2] Alaofi, M., Arabzadeh, N., Clarke, C.L., Sanderson, M.: Generative infor-
mation retrieval evaluation. In: Information access in the era of generative
ai, pp. 135–159, Springer (2024)
[3] Arabzadeh, N., Clarke, C.L.: Benchmarking llm-based relevance judgment
methods. In: Proceedings of the 48th International ACM SIGIR Confer-
ence on Research and Development in Information Retrieval, pp. 3194–3204
(2025), https://doi.org/10.1145/3726302.3744382
[4] Arabzadeh, N., Clarke, C.L.A.: A human-ai comparative analysis of
prompt sensitivity in LLM-based relevance judgment. In: Proceed-
ings of the 48th International ACM SIGIR Conference on Research
and Development in Information Retrieval (SIGIR ’25), ACM (2025),
https://doi.org/10.1145/3726302.3730374
[5] Balog, K., Metzler, D., Qin, Z.: Rankers, judges, and assistants: Towards
understanding the interplay of LLMs in information retrieval evaluation.
arXiv preprint arXiv:2503.19092 (2025)
[6] Bardas, N., Mordo, T., Kurland, O., Tennenholtz, M., Zur, G.: Prompt-
based document modifications in ranking competitions. arXiv preprint
arXiv:2502.07315 (2025)
[7] Ben Basat, R., Tennenholtz, M., Kurland, O.: The probability ranking prin-
ciple is not optimal in adversarial retrieval settings. In: Proceedings of the
2015 International Conference on The Theory of Information Retrieval, p.
51–60, ICTIR ’15 (2015)
[8] Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican,
K., Van Den Driessche, G.B., Lespiau, J.B., Damoc, B., Clark, A., et al.:
Improving language models by retrieving from trillions of tokens. In: Inter-
national conference on machine learning, pp. 2206–2240, PMLR (2022)
[9] Clarke, C.L.A., Dietz, L.: LLM-based relevance assessment still can’t replace
human relevance assessment. In: EVIA 2025: Proceedings of the Tenth In-
ternational Workshop on Evaluating Information Access (EVIA 2025), a
Satellite Workshop of the NTCIR-18 Conference, June 10-13, 2025, Tokyo,
Japan, pp. 1–5 (2025), https://doi.org/10.20736/0002002105
[10] Dhol´ e, K., Chandradevan, R., Agichtein, E.: AdvERSEM: Adversarial ro-
bustness testing and training of LLM-based groundedness evaluators via
manipulation of semantic structures. In: 14th Joint Conference on Lexical
and Computational Semantics (2025)

Insider Knowledge for RAG Systems 15
[11] Dietz, L.: A workbench for autograding retrieve/generate systems. In: Pro-
ceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, pp. 1963–1972 (2024)
[12] Dietz, L., Li, B., Liu, G., Ju, J.H., Yang, E., Lawrie, D., Walden, W., May-
field, J.: Incorporating Q&A nuggets into retrieval-augmented generation.
In: Proceedings of the 48th European Conference on Information Retrieval
(ECIR 2026) (2026)
[13] Dietz, L., Li, B., Mayfield, J., Lawrie, D., Yang, E., Walden, W.: [hltcoe]
hltcoe evaluation team at trec 2025: Rag, ragtime, and dragun. In: The
Thirty-Fourth Text REtrieval Conference Proceedings (TREC2025) (2025)
[14] Dietz, L., Zendel, O., Bailey, P., Clarke, C., Cotterill, E., Dalton, J., Hasibi,
F., Sanderson, M., Craswell, N.: Principles and guidelines for the use of LLM
judges. In: Proceedings of the 11th ACM SIGIR / The 15th International
Conference on Innovative Concepts and Theories in Information Retrieval
(2025)
[15] Ding, Y., Facciani, M., Poudel, A., Joyce, E., Aguinaga, S., Veeramani,
B., Bhattacharya, S., Weninger, T.: Citations and trust in LLM generated
responses (2025), URLhttps://arxiv.org/abs/2501.01303
[16] Duh, K., Yang, E., Weller, O., Yates, A., Lawrie, D.: HLTCOE at LiveRAG:
GPT-Researcher using ColBERT retrieval. arXiv preprint arXiv:2506.22356
(2025)
[17] Elovic, A.: gpt-researcher (Jul 2023), URLhttps://github.com/
assafelovic/gpt-researcher
[18] Farzi, N., Dietz, L.: Exam++: Llm-based answerability metrics for ir evalua-
tion. In: Proceedings of LLM4Eval: The First Workshop on Large Language
Models for Evaluation in Information Retrieval (2024)
[19] Farzi, N., Dietz, L.: Pencils down! automatic rubric-based evaluation of
retrieve/generate systems. In: Proceedings of the 2024 ACM SIGIR Inter-
national Conference on Theory of Information Retrieval, pp. 175–184 (2024)
[20] Fr¨ obe, M., Reimer, J.H., MacAvaney, S., Deckers, N., Reich, S., Beven-
dorff, J., Stein, B., Hagen, M., Potthast, M.: The Information Re-
trieval Experiment Platform. In: Chen, H.H., Duh, W., Huang, H.H.,
Kato, M., Mothe, J., Poblete, B. (eds.) 46th International ACM SI-
GIR Conference on Research and Development in Information Retrieval
(SIGIR 2023), pp. 2826–2836, ACM (Jul 2023), ISBN 9781450394086,
https://doi.org/10.1145/3539618.3591888
[21] Izacard, G., Grave, ´E.: Leveraging passage retrieval with generative models
for open domain question answering. In: Proceedings of the 16th Conference
of the European Chapter of the Association for Computational Linguistics:
Main Volume, pp. 874–880 (2021)
[22] Kurland, O., Tennenholtz, M.: Competitive search. In: Proceedings of the
45th International ACM SIGIR Conference on Research and Development
in Information Retrieval, p. 2838–2849, SIGIR ’22 (2022)
[23]  Lajewska, W., Balog, K.: Ginger: Grounded information nugget-based gen-
eration of responses. In: Proceedings of the 48th International ACM SI-

16 Dietz et al.
GIR Conference on Research and Development in Information Retrieval,
pp. 2723–2727 (2025)
[24] Lawrie, D., MacAvaney, S., Mayfield, J., McNamee, P., Oard, D.W., Sol-
daini, L., Yang, E.: Overview of the TREC 2024 NeuCLIR track. arXiv
preprint arXiv:2509.14355 (2025)
[25] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
K¨ uttler, H., Lewis, M., Yih, W.t., Rockt¨ aschel, T., et al.: Retrieval-
augmented generation for knowledge-intensive NLP tasks. Advances in neu-
ral information processing systems33, 9459–9474 (2020)
[26] Li, D., Jiang, B., Huang, L., Beigi, A., Zhao, C., Tan, Z., Bhattacharjee, A.,
Jiang, Y., Chen, C., Wu, T., Shu, K., Cheng, L., Liu, H.: From generation
to judgment: Opportunities and challenges of LLM-as-a-judge (2025), URL
https://arxiv.org/abs/2411.16594
[27] Mayfield, J., Yang, E., Lawrie, D., MacAvaney, S., McNamee, P., Oard,
D.W., Soldaini, L., Soboroff, I., Weller, O., Kayi, E., Sanders, K., Mason,
M., Hibbler, N.: On the evaluation of machine-generated reports. In: Pro-
ceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, pp. 1904–1915 (2024)
[28] Nguyen, T., Lei, Y., Ju, J.H., Yang, E., Yates, A.: Milco: Learned sparse
retrieval across languages via a multilingual connector. arXiv [cs.IR] (2025)
[29] Pavlu, V., Rajput, S., Golbus, P.B., Aslam, J.A.: Ir system evaluation us-
ing nugget-based test collections. In: Proceedings of the fifth ACM inter-
national conference on Web search and data mining, pp. 393–402 (2012),
https://doi.org/10.1145/2124295.2124343
[30] Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Lin, J.:
Initial nugget evaluation results for the TREC 2024 rag track with the
autonuggetizer framework (2024), URLhttps://arxiv.org/abs/2411.
09607, arXiv preprint
[31] Pradeep, R., Thakur, N., Upadhyay, S., Campos, D., Craswell, N., Sobo-
roff, I., Dang, H.T., Lin, J.: The great nugget recall: Automating fact
extraction and rag evaluation with large language models. In: Proceed-
ings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval, p. 180–190, SIGIR ’25, Associ-
ation for Computing Machinery, New York, NY, USA (2025), ISBN
9798400715921, https://doi.org/10.1145/3726302.3730090, URLhttps://
doi.org/10.1145/3726302.3730090
[32] Soboroff, I.: Don’t use LLMs to make relevance judgments. Information
retrieval research journal1(1), 10–54195 (2025)
[33] Thakur, N., Pradeep, R., Upadhyay, S., Campos, D., Craswell, N., Lin,
J.: Support evaluation for the TREC 2024 rag track: Comparing human
versus LLM judges. In: Proceedings of the 48th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR
’25), ACM (2025), https://doi.org/10.1145/3726302.3730165
[34] Thomas, P., Spielman, S., Craswell, N., Mitra, B.: Large Language Models
can Accurately Predict Searcher Preferences. In: Proceedings of the 47th

Insider Knowledge for RAG Systems 17
International ACM SIGIR Conference on Research and Development in
Information Retrieval (2024)
[35] Upadhyay, S., Pradeep, R., Thakur, N., Craswell, N., Lin, J.: UMBRELA:
UMbrela is the (Open-Source Reproduction of the) Bing RELevance Asses-
sor. arXiv preprint arXiv:2406.06519 (2024)
[36] Upadhyay, S., Thakur, N., Pradeep, R., Campos, D., Clarke, C.L.A., Lin,
J.: A large-scale study of relevance assessments with large language mod-
els using UMBRELA. In: Proceedings of the 15th International Confer-
ence on the Theory of Information Retrieval (ICTIR ’25), ACM (2025),
https://doi.org/10.1145/3731120.3744605
[37] Walden, W., Weller, O., Dietz, L., Li, B., Liu, G.K.M., Hou, Y., Yang,
E.: Auto-ARGUE: LLM-based report generation evaluation. arXiv preprint
arXiv:2509.26184 (2025)
[38] Yang, E., Lawrie, D., Mayfield, J., Oard, D.W., Miller, S.: Translate-distill:
learning cross-language dense retrieval by translation and distillation. In:
European Conference on Information Retrieval, pp. 50–65, Springer (2024)
[39] Yang, E., Lawrie, D., Weller, O., Mayfield, J.: HLTCOE at TREC 2024
NeuCLIR track (2025), URLhttps://arxiv.org/abs/2510.00143
[40] Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang,
A., Liu, D., Lin, J., et al.: Qwen3 embedding: Advancing text embedding
and reranking through foundation models. arXiv preprint arXiv:2506.05176
(2025)