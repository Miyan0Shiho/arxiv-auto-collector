# Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation

**Authors**: Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, Gianluca Demartini

**Published**: 2026-08-11 02:26:47

**PDF URL**: [https://arxiv.org/pdf/2608.10385v1](https://arxiv.org/pdf/2608.10385v1)

## Abstract
Large language models (LLMs) are increasingly used as relevance assessors in information retrieval (IR) evaluation, raising questions about how assessor framing affects judgment reliability and downstream system comparison. We study persona conditioning as a diagnostic mechanism for exposing LLM assessor sensitivity. Using task-oriented personas drawn from two complementary sources (PersonaHub and NVIDIA Nemotron-Personas-USA), we instantiate five assessor roles emphasizing intent interpretation, domain expertise, contrastive judgment, evidence verification, and global search-quality assessment, compared with a standard UMBRELA baseline. Across six LLM backbones on TREC DL20 and RAG24, our analyses reveal structured rather than uniform assessor sensitivity. Judgments usually remain close to the baseline while shifting assessment strictness, evidential threshold, or interpretation emphasis rather than producing widespread relevance reversals. At the system level, high-capacity models preserve system-ranking agreement, while smaller models amplify persona-induced instability. Local rank-displacement analysis shows sensitivity concentrates on particular retrieval systems and system types, especially neural ranking/reranking systems on DL20 and RAG-oriented pipelines on RAG24. Persona source matters less than assessor role and model capacity. These findings position persona-conditioned judging as a controlled sensitivity probe for stress-testing LLM-based IR evaluation pipelines and identifying systems whose evaluation outcomes are sensitive to assessor framing.

## Full Text


<!-- PDF content starts -->

Persona Conditioning as an Assessor-Sensitivity Probe for
LLM-Based IR Evaluation
Samaneh Mohtadi
s.mohtadi@uq.edu.au
The University of Queensland
Brisbane, AustraliaPietro Bernardelle
p.bernardelle@uq.edu.au
The University of Queensland
Brisbane, Australia
Joel Mackenzie
joel.mackenzie@uq.edu.au
The University of Queensland
Brisbane, AustraliaGianluca Demartini
g.demartini@uq.edu.au
The University of Queensland
Brisbane, Australia
Abstract
Large language models (LLMs) are increasingly used as relevance
assessors in information retrieval (IR) evaluation, raising questions
about how assessor framing affects judgment reliability and down-
stream system comparison. We study persona conditioning as a
diagnostic mechanism for exposing LLM assessor sensitivity. Using
task-oriented personas drawn from two complementary sources
(PersonaHub and NVIDIA Nemotron-Personas-USA), we instanti-
ate five assessor roles emphasizing intent interpretation, domain
expertise, contrastive judgment, evidence verification, and global
search-quality assessment, compared with a standard UMBRELA
baseline. Across six LLM backbones on TREC DL20 and RAG24, our
analyses reveal structured rather than uniform assessor sensitiv-
ity. Judgments usually remain close to the baseline while shifting
assessment strictness, evidential threshold, or interpretation em-
phasis rather than producing widespread relevance reversals. At the
system level, high-capacity models preserve system-ranking agree-
ment, while smaller models amplify persona-induced instability.
Local rank-displacement analysis shows sensitivity concentrates
on particular retrieval systems and system types, especially neural
ranking/reranking systems on DL20 and RAG-oriented pipelines on
RAG24. Persona source matters less than assessor role and model
capacity. These findings position persona-conditioned judging as a
controlled sensitivity probe for stress-testing LLM-based IR evalua-
tion pipelines and identifying systems whose evaluation outcomes
are sensitive to assessor framing.
CCS Concepts
•Information systems→Relevance assessment.
Keywords
Information Retrieval Evaluation, Large Language Models, LLM-as-
a-Judge, Relevance Judgment, Persona Conditioning
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
CIKM ’26, Rome, Italy
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MMACM Reference Format:
Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca De-
martini. 2026. Persona Conditioning as an Assessor-Sensitivity Probe for
LLM-Based IR Evaluation. InProceedings of The 35th ACM International
Conference on Information and Knowledge Management (CIKM ’26).ACM,
New York, NY, USA, 12 pages.
1 Introduction
Relevance judgments play a central role in information retrieval (IR)
evaluation by enabling system quality to be analyzed and compared.
While traditionally produced by trained human assessors, the high
cost and limited scalability of manual judging have driven growing
interest in utilizing large language models (LLMs) as automated
relevance judges. Recent work suggests that, under appropriate
prompting, LLM judges can approximate professional assessors or
real searcher preferences [ 41,47]. Other studies show that LLM-
derived judgments can produce system rankings broadly compara-
ble to those derived from human judgments, supporting scalable
IR evaluation [ 48,49]. However, LLM-based judgments are sensi-
tive to prompt formulation and judging context [ 1,14,44]. They
can also be affected by superficial document cues, thresholding
choices, and evaluator bias [ 5,11,55]. Recent guidelines further
emphasize that these risks should be considered when using LLMs
as judges [ 17]. These concerns motivate viewing LLM judges as
components within a human–machine evaluation spectrum rather
than as definitive sources of ground truth [18].
Beyond these concerns, many LLM-based judging approaches
operationalize evaluation through a single fixed judging perspec-
tive. This perspective may approximate a professional assessor, a
real searcher, or a standardized relevance guideline, but it still rep-
resents only one interpretation of the assessment task. In contrast,
decades of IR research establish relevance assessment as inherently
subjective [ 45], shaped by how assessors interpret query intent, task
context, and relevance criteria [ 35,42]. Large-scale evaluation stud-
ies show that assessor disagreement can affect evaluation outcomes,
with structured variation in assessor preferences capable of mean-
ingfully shifting system comparisons [ 4,9,51]. This observation
motivates the following question for LLM-based IR evaluation:how
sensitive are evaluation outcomes to changes in assessor perspective?
Existing persona-conditioning work has explored psychological
traits, demographic attributes, confidence styles, role-play prompts,
arXiv:2608.10385v1  [cs.IR]  11 Aug 2026

CIKM ’26, November 7–11, 2026, Rome, Italy Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca Demartini
and lexical prompt variation [ 10,27,59], while criteria-based ap-
proaches have proposed decomposing relevance into explicit di-
mensions such as accuracy, usefulness, and coverage [ 20]. These
efforts primarily target behavioral alignment or prompt optimiza-
tion, leaving open how systematic variation in assessor perspective
propagates through relevance labels, ranking stability, and retrieval-
system evaluation outcomes.
In this paper, we study persona conditioning as an evaluator-
sensitivity probe for LLM-based IR evaluation. Rather than treating
personas as a mechanism for improving relevance labels, we use
them to systematically perturb the assessor perspective and observe
how these perturbations affect both relevance judgments and down-
stream IR system rankings. We instantiate task-oriented assessor
roles covering query interpretation, domain expertise, contrastive
judgment, evidence verification, and global search-quality assess-
ment, and compare them against a standard UMBRELA judge [ 50].
To examine whether persona source affects evaluation behavior, we
compare abstract PersonaHub profiles [ 24] with skill-grounded per-
sonas from the NVIDIA Nemotron-Personas-USA collection [34].
To enable scalable experimentation, we adopt summary-based
judging and reuse fixed document summaries across persona runs,
following prior work showing that concise summaries preserve
judgment behavior and system-level stability while substantially re-
ducing inference cost [ 37]. We conduct experiments on the DL20 [ 16]
and RAG24 [ 38] datasets using multiple open and closed LLM back-
bones of different sizes. Our analysis examines persona effects at
both the relevance-judgment and IR system ranking levels through
graded agreement analysis, system ranking stability, and system-
level rank-displacement analysis.
Our study addresses the following research questions:
RQ1 How does persona conditioning affect relevance judgments
relative to a standard UMBRELA-style LLM judge?
RQ2 Do abstract personas and skill-grounded personas induce
different evaluator-sensitivity patterns?
RQ3 How stable are IR system rankings under persona-conditioned
judging across datasets and model families?
RQ4 Which retrieval systems are most influenced by persona-
conditioned judging, and do these effects correlate with sys-
tem characteristics?
Persona conditioning produces structured, model-dependent sensi-
tivity rather than uniform evaluation instability. At the judgment
level, persona-conditioned labels generally remain close to the UM-
BRELA baseline, with differences appearing mainly as localized
shifts in assessment strictness, evidential threshold, and interpre-
tation emphasis. At the system level, sensitivity concentrates on
particular retrieval-system types while global rankings remain rel-
atively stable for high-capacity models. Persona source proves less
consequential than assessor role and model capacity. These patterns
position persona-conditioned judging as a controlled diagnostic
stress test rather than a labeling alternative, with practical value
for identifying retrieval systems whose evaluations are sensitive to
assessor framing.
2 Background and Related Work
Relevance Subjectivity and Evaluation Stability.Relevance in IR
is inherently subjective and shaped by user intent, task context,and assessor interpretation of relevance criteria [ 35,42,45]. Em-
pirical TREC analyses have repeatedly documented this variability.
Voorhees [ 51] showed that although individual assessors often
disagree on relevance, evaluation outcomes can remain relatively
stable when using different sets of qualified assessors. Zobel [ 60]
similarly examined the reliability of large-scale pooled evaluation
and found that system comparisons are often robust to variation in
relevance judgments. Subsequent work has refined this picture: Bai-
ley et al. [ 4] demonstrated that assessors of different quality levels
do not produce interchangeable judgments and that assessor type
affects measured effectiveness, while Carterette and Soboroff [ 9]
showed that systematic assessor error can shift system rankings
even when label-level agreement remains high. Together, these find-
ings motivate treating assessor variation not as annotation noise
to be averaged out, but as a structured component of IR evalua-
tion whose effects on downstream system comparison must be
characterized.
LLMs as Relevance Judges.The use of LLMs as automated evalua-
tors was first widely formalized in general NLP, where frameworks
such as MT-Bench [ 58] and G-Eval [ 32] showed that LLMs can
produce evaluation scores correlating strongly with human judg-
ments across open-ended tasks. Adapting these ideas to IR, the
cost and scalability limitations of manual judging have motivated
widespread exploration of LLMs as automated relevance assessors.
Under carefully designed prompting protocols, LLM judges can ap-
proximate professional assessors or real searcher preferences [47]
and produce system rankings broadly comparable to those derived
from human judgments [ 48,49]. Frameworks such as LLMJudge
formalize LLM-based relevance assessment pipelines and support
systematic comparison of judging configurations [ 41], while stan-
dardized prompting approaches such as UMBRELA [ 49,50] enable
large-scale analysis of agreement patterns and evaluation behavior
across datasets and models. However, the generalizability of these
frameworks depends on the backbone model: Farzi and Dietz [21]
reproduce UMBRELA across multiple LLMs and find that smaller
models exhibit meaningfully degraded assessment performance
relative to larger ones. Criteria-based approaches [ 20] further de-
compose relevance into explicit assessment dimensions, such as
accuracy, usefulness, and coverage. Recent work shows that input-
side formulation influences evaluation behavior, as formalized topic
descriptions and narratives substantially improve agreement with
human judgments [ 29]. Benchmarking studies show that LLM-
based relevance assessment is sensitive to experimental design
choices, with outcomes varying across prompting strategies and
judging setups [ 3]. Moreover, label-level agreement alone does not
guarantee ranking stability, since similar agreement levels can still
produce meaningfully different system orderings [ 4,9,25]. Our
work differs from these criteria-based, input-side, and prompting-
focused approaches by varyingwhois assumed to be judging rather
thanwhatis assessed orhowthe assessment input is structured,
while keeping the underlying judging framework fixed.
Bias and Instability in LLM-based Evaluation.LLM-based rele-
vance judgment raises concerns about reliability, robustness, and
bias, particularly when LLM-generated labels are used as replace-
ments rather than supplements for human assessors [ 14,44]. Broader
NLP studies have documented systematic biases in LLM evaluators,

Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation CIKM ’26, November 7–11, 2026, Rome, Italy
most notably positional or order bias, in which evaluation out-
comes can shift simply by changing the order in which candidate
responses appear [ 52]. Empirical studies show that LLM judgments
are sensitive to prompt formulation, superficial document cues [ 56],
and judging context; for example, injecting query terms into irrel-
evant documents can inflate relevance labels without improving
substantive relevance [ 1,2]. Cognitive-bias-like effects have also
been identified in LLM assessors, including threshold priming [ 11]
and recency bias [ 19]. At the system level, LLM judges may fa-
vor LLM-based ranking and generation systems, raising concerns
about circular evaluation and self-preference [ 5]. Broader analyses
identify structural evaluation risks such as circularity, LLM nar-
cissism, loss of variety of opinion, and multiple systematic judge
biases [ 17,55]. Recent work has begun to show that disagreement
and evaluator behavior can exhibit structured patterns within the
query–document representation space, enabling analysis beyond
aggregate label-level agreement [ 36]. These studies typically char-
acterize biases that arise within a fixed judging configuration; we
instead examine sensitivity that emerges when the judging perspec-
tive itself is systematically varied.
Persona Conditioning and Assessor Perspective.Most directly rel-
evant to our study, recent work examines how persona condi-
tioning alters LLM behavior in evaluation tasks. Wang et al. [ 53]
show that role-play signals can systematically alter zero-shot rank-
ing behavior while remaining only weakly entangled with query–
document representations, suggesting that persona effects may
driven by evaluator framing rather than changes to the underlying
query–document representation. Work on synthetic annotators
and crowd impersonation finds that persona conditioning does
not reliably reproduce the diversity or inconsistency patterns ob-
served in real human assessors [ 22,23,31], raising questions about
how persona-based evaluation should be interpreted. A separate
line of research models personas through psychological traits or
demographic attributes [ 27,28,46]. In relevance assessment specif-
ically, personality-conditioned prompts have been reported to yield
modest alignment improvements [ 10], though demographic and
persona prompting more broadly often produce inconsistent effects
and limited controllability [ 33,43,57,59]. In contrast to this body
of work, which primarily models personas through demographic or
psychological traits, we adopttask-orientedassessor perspectives
and analyze how these perspectives propagate to downstream IR
evaluation stability and retrieval-system sensitivity.
3 Methodology and Experimental Setting
This section describes our methodology and experimental setup.
Our goal is to use persona-conditioned judging as a diagnostic
probe for evaluating how changes in assessor perspective affect
relevance labels and downstream IR evaluation outcomes.1
3.1 Datasets and Models
We conduct experiments on two benchmark datasets commonly
used in recent IR evaluation with LLM judges: TREC Deep Learning
2020 (DL20) [ 16] and the TREC Retrieval-Augmented Generation
track 2024 (RAG24) [ 38]. DL20 contains 54 predominantly short,
1Code is publicly available at https://osf.io/5sf2z/overview?view_only=
e57aef041bb142d59722c44ef4f5682bTable 1:Domain taxonomy used for Domain-Expert assessor per-
sonas. Each query is assigned to exactly one domain.
Domain Abbr. Domain Abbr.
Health and Medicine H&M Environment and Earth Sciences ENV
Law, Policy, and Government LPG Current Affairs and History CAH
Science and Technology S&T Education and Humanities E&H
Business and Economics B&E Arts, Media, and Culture AMC
General Knowledge and Reasoning GKR
factual, and definitional queries, whereas RAG24 contains 86 longer,
more open-ended queries that often involve explanatory, contextual,
or subjective information needs. Both datasets employ graded rele-
vance labels, and we include all available submitted system runs for
each dataset in our system-level evaluation. We evaluate six LLM
backbones covering a range of model families and scales, including
GPT-4o and GPT-4o-mini, LLaMA-3.1-70B and LLaMA-3.1-8B, and
Qwen-2.5-72B and Qwen-2.5-7B. We use instruction-tuned conver-
sational variants of the open-weight models [ 39], which are well
suited to our in-context prompting approach for eliciting distinct
assessor perspectives during relevance judgment. All judgments are
generated with temperature set to 0 to reduce sampling variability
across repeated assessor conditions.
3.2 Assessor Personas
In this work, we model LLM judges using task-oriented assessor
personas. We use the term persona to denote an explicit evalua-
tion role that reflects a particular relevance assessment perspective,
rather than psychological personality traits or demographic at-
tributes. This distinction is central to our study, which focuses on
how relevance is judged, rather than who the judge is.
3.2.1 Assessor Roles.We define five assessor roles that emphasize
different relevance perspectives: (i) query-aligned interpretation,
(ii) domain expertise, (iii) contrastive or orthogonal judgment, (iv)
evidence verification, and (v) a global assessor perspective.
Query-Aligned Assessor.The Query-Aligned assessor focuses on
the user’s information need, favoring documents that semantically
align with the inferred query intent. This role is instantiated sepa-
rately for each query to reflect query-specific interpretation.
Domain-Expert Assessor.The Domain-Expert assessor, also re-
ferred to as Domain, emphasizes topical expertise and domain-
specific relevance criteria. To instantiate this role consistently across
datasets, we construct a shared nine-domain taxonomy and assign
each query to exactly one domain (Table 1). The mapping was pro-
duced by two independent annotators, with disagreements resolved
through discussion after measuring inter-annotator agreement us-
ing Cohen’s 𝜅(0.86). This mapping is used for Domain persona
retrieval and domain-level aggregation.
Orthogonal Assessor.The Orthogonal assessor introduces a con-
trastive perspective by intentionally deviating from dominant in-
terpretations of query intent. This role is designed to surface as-
sessor sensitivity under alternative assessor framing, rather than
to simulate an incorrect or random judge. Orthogonal assessors
are instantiated on a per-query basis using semantically dissimilar
persona retrieval to encourage contrastive assessor framing.

CIKM ’26, November 7–11, 2026, Rome, Italy Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca Demartini
Table 2:Illustrative persona examples for the RAG24 query“should teachers notify parents about state testing?”across both PersonaHub and
USPersonadata. Personas taken fromUSPersonatend to be longer and more skill-oriented.
Source Role Example Persona
PersonaHubQuery A single parent advocating for education reform based on their child’s stress from excessive testing.
Domain A sociology professor researching the impact of incorporating humanities in STEM education on student success and creativity.
Orthogonal A highly sought-after designer known for creating chic and trendy nightclub interiors.
USPersonaQuery A persona with the following skills: Early childhood education, STEM curriculum integration for preschool, Structured daily routine design, Low-
stimulus classroom environment creation, Individualized learning plan development, Observational child assessment, Educational technology utilization,
Hands-on science experiment facilitation, Literacy foundation teaching, Emotional regulation techniques for children, Parent communication and
consultation, Curriculum adaptation for anxiety-prone learners.
Domain A persona with the following skills: Interdisciplinary curriculum development, Instructional technology integration, Public speaking and lecturing,
Classroom management, Conflict resolution, Community outreach and partnership building, Research and academic writing, Flexible lesson planning,
Mentorship and coaching, Cultural competency in humanities education.
Orthogonal A persona with the following skills: Vendor negotiation, Strategic sourcing, Cost analysis and budgeting, Supply chain risk assessment, Market research,
Creative procurement problem solving, Sustainability sourcing, Contract drafting, Data analytics with Excel and Power BI, Use of e-procurement
platforms (SAP Ariba, Coupa), Improvisational logistics planning, Relationship management with suppliers, Inventory forecasting, Process optimization,
Cross-functional collaboration.
Evidence-Verification Assessor.The Evidence-Verification asses-
sor emphasizes factual correctness, evidential support, and source
credibility in relevance judgments. This role targets RAG-style eval-
uation scenarios, where documents may be topically related yet
poorly supported or unreliable, and therefore penalizes speculative
or unsubstantiated claims even when topical alignment is strong.
Unlike query-specific personas, this assessor is defined through
a fixed prompt to explicitly isolate a trust-oriented relevance di-
mension. We define this persona as: “An assessor who prioritizes
factual correctness, evidential support, and source credibility when
judging relevance. Favor well-supported and verifiable content, and
penalize speculative or unsubstantiated claims, even when topically
related.”
Global Assessor Persona (GAP).Following prior work that uses
global assessor impersonation [ 47], we define GAP as a profes-
sional search-quality rater applying standard web-search relevance
guidelines. GAP is expressed through a fixed system prompt and is
applied uniformly across all queries and datasets. Compared to the
UMBRELA default, GAP makes the assessor perspective explicit,
providing a stable query-independent reference for comparison
with query-specific persona-conditioned judgments. We define this
persona as: “A professional search quality rater evaluating relevance
according to standard web search guidelines.”
3.2.2 Persona Sources.We instantiate the Query-Aligned, Domain-
Expert, and Orthogonal roles from two complementary persona
sources in order to compare abstract task-oriented personas with
skill-grounded professional personas.
The first source is PersonaHub [ 24], which provides synthetic
persona descriptions designed to support diversity in annotation
and evaluation tasks [ 6–8,12,13,23]. PersonaHub personas are
abstract and broad in scope, often encoding viewpoints, occupations,
or experiential backgrounds intended to diversify interpretation.
The second source, which we refer to as USPersona, is derived
from the NVIDIA Nemotron-Personas-USA collection [ 34]. Un-
like PersonaHub, USPersona provides occupationally grounded
profiles associated with explicit skills and expertise. To preserve
task-oriented assessor framing rather than demographic or psycho-
logical conditioning, we restrict retrieval to professional-descriptionand skill-related fields while excluding demographic, cultural, and
lifestyle attributes. Retrieval is therefore performed over skill-based
descriptions of the form“A person with the following skills: [skills
list]”, enabling construction of assessor roles grounded in concrete
expertise rather than identity characteristics.
For both PersonaHub and USPersona, persona selection follows
the same retrieval procedure. Persona descriptions are encoded us-
ingall-MiniLM-L6-v2 , and cosine similarity is computed between
persona embeddings and either query or domain representations.
At each selection step, we retrieve the top three candidate personas
for inspection and retain the highest-ranked one. Query-Aligned
and Domain-Expert personas are selected based on high similarity
to the query or domain context, while Orthogonal personas are
selected to be intentionally dissimilar. This process allows different
queries to activate different assessor profiles, reflecting structured
variation in interpretation rather than fixed assessor identity.
3.3 UMBRELA Baseline
The UMBRELA assessor corresponds to the default UMBRELA
relevance-judging configuration [ 50] and serves as the primary
baseline. We use UMBRELA as a fixed judging reference. This allows
us to measure how explicit assessor-role conditioning changes
relevance judgments and downstream system rankings relative to
a standard LLM judging setup.
3.4 Summary-Based Judging
Across both datasets, each query is evaluated under nine judging
conditions: PersonaHub Query, PersonaHub Domain, PersonaHub
Orthogonal, USPersona Query, USPersona Domain, USPersona Or-
thogonal, Evidence-Verification, GAP, and UMBRELA. Since each
query–document pair is judged repeatedly across assessor roles,
persona sources, and model backbones, the total number of LLM
evaluations grows rapidly. We therefore adopt summary-based
judging, following prior work showing that concise document sum-
maries can preserve agreement and system-level ranking stability
while substantially reducing evaluation cost [37].
For all experiments, each document is summarized once into an
approximately 80-token summary, and the resulting summary is

Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation CIKM ’26, November 7–11, 2026, Rome, Italy
reused across all assessor personas, persona sources, model back-
bones, and evaluation settings. Document summaries are generated
using the prompt template and generation settings released with
the prior summary-based judging study [ 37]; following that setup,
we use GPT-4o as the summarization model. This design enables
controlled experimentation by fixing the document representation
and varying only the assessor perspective. As a result, observed
differences in relevance judgments are less likely to arise from
document length, context variability, or repeated summarization.
Summary reuse is particularly important in our setting because each
query–document pair is evaluated multiple times under different
assessor roles and LLM configurations.
3.5 Evaluation Framework
We adopt the UMBRELA judging prompt as the underlying relevance-
judging framework and vary only the assessor-role instruction. For
persona-conditioned runs, the UMBRELA prompt [ 50] is prefixed
with the assessor instruction “You are acting as {persona}.” This
design keeps the judging template fixed, allowing observed differ-
ences to be attributed to persona conditioning rather than prompt-
template changes. Our evaluation considers three complementary
levels.
Judgment-level agreement.We measure how persona-conditioned
judgments differ from UMBRELA and human judgments. Because
relevance labels are ordinal, we report quadratic-weighted Cohen’s
𝜅[15]. Quadratic weighting accounts for the ordinal structure of
relevance labels by penalizing distant disagreements more strongly
than adjacent-grade shifts. This allows us to distinguish severe
relevance inversions from local grade shifts induced by persona
conditioning.
System ranking stability.At the system level, we first compute
retrieval effectiveness using NDCG@10 [ 26], following standard
TREC Deep Learning practice [ 16]. For each assessor configuration,
including UMBRELA, we derive system-level effectiveness scores
from LLM-derived relevance labels and rank submitted runs accord-
ingly. We then compare each LLM-induced system ranking with
the human-derived ranking using Kendall’s 𝜏, a standard measure
of pairwise ranking consistency widely used to study sensitivity to
assessor variation [ 25,30], and Rank-Biased Overlap (RBO) with
persistence parameter 𝜙=0.9, which emphasizes agreement among
top-ranked systems [54].
System-level sensitivity.In addition to global ranking stability,
we measure localized system impact through rank-displacement
analysis. For each retrieval system 𝑠and persona condition 𝑝, we
compute the rank shift relative to UMBRELA:
Δ𝑟(𝑠, 𝑝)=𝑟 𝑝(𝑠)−𝑟 UMB(𝑠),
where negative values indicate that a system moves up under
persona-conditioned judging and positive values indicate that it
moves down. We summarize system sensitivity as:
Sensitivity(𝑠)=1
|𝑃|∑︁
𝑝∈𝑃|Δ𝑟(𝑠, 𝑝)|,where 𝑃denotes the set of non-UMBRELA persona conditions.
Mean absolute rank shift is first computed across retrieval sys-
tems for each persona condition and model, and then averaged
across models for dataset-level reporting. To assess the robustness
of model sensitivity estimates, we additionally compute 95% boot-
strap confidence intervals by resampling retrieval systems with
replacement 1,000 times within each dataset–model pair (random
seed: 42).
We additionally report maximum rank displacement, affected-
system counts, signed NDCG@10 shifts, and direction consistency
across models. Maximum rank displacement captures the largest
observed rank movement under persona conditioning. Affected-
system counts measure how many systems change rank relative
to UMBRELA, while signed NDCG@10 shifts indicate whether
systems improve or degrade under a persona condition. Direction
consistency measures whether systems move consistently upward
or downward across models. Together, these analyses identify re-
trieval systems whose evaluation outcomes are most sensitive to
changes in assessor perspective and distinguish global ranking
stability from localized system movement.
4 Results and Analysis
The results show how persona conditioning affects LLM judgments
at three levels: relevance-label agreement, downstream system-
ranking stability, and localized system sensitivity.
4.1 RQ1: Effects of Persona Conditioning on
Relevance Judgments
We first examine how assessor perspectives influence relevance
judgments relative to the default UMBRELA judging configuration,
considering agreement with UMBRELA and agreement with human
judgments across datasets and model families.
4.1.1 Agreement with UMBRELA Judging.Figure 1 reports the
mean quadratic-weighted Cohen’s 𝜅between persona-conditioned
and UMBRELA judgments, averaged across queries for each dataset
and model. Higher values indicate stronger agreement with the
baseline UMBRELA configuration, whereas lower values indicate
larger persona-induced deviations.
Overall, persona-conditioned judgments remain close to UM-
BRELA for most models. Agreement is highest for the high-capacity
LLaMA-3.1-70B and Qwen-2.5-72B, with mid-capacity GPT-4o-mini
exhibiting similarly stable behavior. GPT-4o is a notable exception:
despite being a high-capacity model, it maintains only moderate-to-
high agreement on DL20 and drops further on RAG24, especially
under Evidence and GAP personas, indicating greater sensitivity
to assessor perspective on the more interpretive dataset. In con-
trast, smaller models, particularly LLaMA-3.1-8B and Qwen-2.5-7B,
show substantially lower agreement and greater variability across
assessor roles.
Although role-level patterns vary across models and datasets,
a consistent diagnostic trend emerges in the heatmap: global as-
sessor perspectives such as Evidence and GAP often remain close
to UMBRELA, whereas Orthogonal personas are more likely to
induce larger deviations. Query and Domain personas generally
fall between these two extremes.

CIKM ’26, November 7–11, 2026, Rome, Italy Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca Demartini
DL20
GPT-4o DL20
LLaMA-3.1-70BDL20
Qwen-2.5-72BDL20
GPT-4o-miniDL20
LLaMA-3.1-8BDL20
Qwen-2.5-7BRAG24
GPT-4o RAG24
LLaMA-3.1-70BRAG24
Qwen-2.5-72BRAG24
GPT-4o-miniRAG24
LLaMA-3.1-8BRAG24
Qwen-2.5-7B
Dataset / ModelPersonaHub Query
PersonaHub Domain
PersonaHub Orthogonal
USPersona Query
USPersona Domain
USPersona Orthogonal
Evidence
GAP
HumanPersona0.733 0.836 0.866 0.867 0.216 0.699 0.520 0.731 0.724 0.775 0.055 0.633
0.757 0.896 0.875 0.867 0.152 0.726 0.518 0.776 0.765 0.808 0.074 0.688
0.757 0.838 0.854 0.867 0.195 0.571 0.513 0.661 0.689 0.790 0.052 0.449
0.759 0.845 0.867 0.844 0.133 0.634 0.509 0.764 0.769 0.793 0.042 0.549
0.786 0.885 0.900 0.882 0.108 0.685 0.540 0.768 0.782 0.805 0.037 0.588
0.772 0.820 0.852 0.855 0.152 0.380 0.479 0.682 0.699 0.776 0.063 0.276
0.681 0.900 0.933 0.917 0.366 0.776 0.374 0.792 0.871 0.845 0.125 0.713
0.717 0.830 0.897 0.905 0.240 0.803 0.436 0.825 0.832 0.847 0.067 0.685
0.572 0.471 0.443 0.535 0.209 0.442 0.316 0.280 0.280 0.299 0.095 0.278Weighted Cohen's kappa vs UMBRELA  DL20 (left) and RAG24 (right)
0.20.40.60.81.0
Mean Weighted Cohen's kappa vs UMBRELA
Figure 1:Persona-level agreement with UMBRELA judging. Heatmap values show the mean quadratic-weighted Cohen’s 𝜅agreement
between persona-conditioned and UMBRELA judgments, averaged over all queries.
4.1.2 Agreement with Human Judgments.To assess how persona
conditioning affects agreement with human judgments, we per-
form a per-query win-rate analysis relative to UMBRELA. For each
persona source, assessor role, and model, we compare persona-
conditioned and UMBRELA judgments on the same query. A per-
sona is counted as a win for a query if its quadratic-weighted Co-
hen’s 𝜅agreement with human judgments is higher than that of the
corresponding UMBRELA judge. The win rate (W%) is then com-
puted as the proportion of queries for which the persona achieves
higher agreement with human judgments than UMBRELA. We
also report the mean agreement difference ( ¯Δ), computed as the
average per-query difference in weighted 𝜅between the persona-
conditioned assessor and UMBRELA. Positive values of ¯Δindicate
increased alignment with human judgments relative to UMBRELA,
while negative values indicate decreased alignment. Table 3 reports
W% and ¯Δfor each persona source, dataset, model, and assessor
role. Because each comparison is made against the same model
under UMBRELA judging, the analysis should be interpreted as
measuring persona-induced movement toward or away from hu-
man judgments rather than absolute judgment quality.
Persona conditioning does not uniformly improve agreement
with human judgments. On DL20, several mid- and high-capacity
models obtain win rates above 50% for selected assessor roles, often
with small positive ¯Δvalues. This pattern is most visible for GPT-
4o-mini and Qwen-2.5-72B, while LLaMA-3.1-70B shows mixed
but occasionally positive shifts. In contrast, LLaMA-3.1-8B exhibits
consistently low win rates and strongly negative agreement shifts
across nearly all roles and datasets, indicating high evaluator insta-
bility under persona conditioning.
Dataset-level differences are also apparent. Persona-conditioned
judging produces more positive or near-neutral shifts on DL20,
whereas RAG24 shows weaker and less consistent gains, with ¯Δvalues often close to zero or negative. GPT-4o shows a different pat-
tern among high-capacity models: although it maintains reasonable
agreement with UMBRELA, its win rates against human judgments
are lower than most other mid- and high-capacity models, and all
¯Δvalues on RAG24 are negative. This suggests that persona condi-
tioning moves GPT-4o away from human judgments on the more
interpretive RAG24 queries rather than increasing agreement.
The win-rate results therefore show that persona-induced changes
do not translate into uniform gains in human agreement. Consis-
tent with the UMBRELA-agreement heatmap, the direction and
magnitude of these changes depend on the interaction between
model capacity, dataset characteristics, and assessor role. Thus, per-
sona conditioning acts as a controlled source of evaluator variation,
revealing where judgments are sensitive to assessor perspective.
4.2 RQ2: Abstract vs Skill-Grounded Personas
We compare abstract PersonaHub personas with skill-grounded
USPersona profiles under matched Query, Domain, and Orthogonal
assessor roles. As shown in Figure 1 and Table 3, the two persona
sources produce broadly similar agreement and win-rate patterns,
with no uniformly stronger source across datasets or models. Dif-
ferences are concentrated in particular role–model combinations:
USPersona sometimes amplifies positive shifts, such as Domain
with LLaMA-3.1-70B on DL20, but can also amplify negative shifts,
such as Orthogonal with Qwen-2.5-7B on RAG24.
Overall, the source of the persona description affects the mag-
nitude of assessor-perspective effects, but does not fundamentally
change the stability pattern observed across roles, datasets, and
model families. One possible explanation is that both PersonaHub
and USPersona operationalize task-oriented assessor framing through
the same retrieval and prompting pipeline. Although the two sources
differ in abstraction level and specificity, both guide the model
toward similar relevance-assessment perspectives. This suggests

Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation CIKM ’26, November 7–11, 2026, Rome, Italy
Table 3:Win–rate analysis relative to UMBRELA. W% is the proportion of queries with higher human agreement than UMBRELA, and ¯Δis
the mean per-query weighted- 𝜅shift (with positive values indicating a shifttowardhuman preferences). Bold values indicate the highest
W% per model row.
TREC Deep Learning 2020
ModelPersonaHub USPersonaEvidence GAP
Query Domain Orthogonal Query Domain Orthogonal
W% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯Δ
GPT-4o 27.8 -0.05137.0-0.045 27.8 -0.056 20.4 -0.056 24.1 -0.043 25.9 -0.049 24.1 -0.080 24.1 -0.074
GPT-4o-mini 48.1 -0.002 48.1 -0.012 51.9 -0.000 53.7 -0.010 51.9 0.012 50.0 -0.005 51.9 0.00961.10.013
LLaMA3.1-70B 51.9 -0.003 57.4 0.011 57.4 0.003 50.0 -0.00170.40.037 64.8 0.017 68.5 0.016 33.3 -0.051
LLaMA3.1-8B 7.4 -0.123 7.4 -0.1369.3-0.123 3.7 -0.153 1.9 -0.163 7.4 -0.140 3.7 -0.102 0.0 -0.148
Qwen2.5-72B 59.3 0.010 68.5 0.018 63.0 0.021 68.5 0.020 63.0 0.01074.10.039 64.8 0.006 29.6 -0.024
Qwen2.5-7B63.00.009 61.1 0.019 40.7 -0.087 53.7 -0.044 42.6 -0.051 31.5 -0.193 59.3 0.02363.00.015
TREC RAG 2024
ModelPersonaHub USPersonaEvidence GAP
Query Domain Orthogonal Query Domain Orthogonal
W% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯ΔW% ¯Δ
GPT-4o 14.0 -0.08725.6-0.072 17.4 -0.081 14.0 -0.091 15.1 -0.091 12.8 -0.095 17.4 -0.116 19.8 -0.090
GPT-4o-mini 57.0 -0.001 60.5 0.003 46.5 -0.007 53.5 -0.003 58.1 0.004 47.7 -0.00661.60.005 60.5 0.005
LLaMA3.1-70B 39.5 -0.015 50.0 0.001 50.0 -0.013 43.0 -0.006 48.8 0.005 51.2 -0.00657.00.008 39.5 -0.011
LLaMA3.1-8B 27.9 -0.05930.2-0.048 27.9 -0.056 20.9 -0.071 23.3 -0.073 23.3 -0.061 27.9 -0.047 20.9 -0.079
Qwen2.5-72B 45.3 -0.014 55.8 -0.000 54.7 -0.000 43.0 -0.009 51.2 -0.007 52.3 -0.00258.1-0.001 44.2 -0.003
Qwen2.5-7B 38.4 -0.013 45.3 -0.008 31.4 -0.077 37.2 -0.053 38.4 -0.045 12.8 -0.17347.7-0.014 40.7 -0.021
that persona source mainly modulates the magnitude of assessor-
perspective effects, whereas the role instruction and model deter-
mine the dominant stability pattern.
4.3 RQ3: Stability of System Rankings
We next evaluate whether persona-conditioned judgments preserve
downstream IR system rankings derived from human relevance
judgments across datasets, assessor perspectives, and model fami-
lies.
4.3.1 System Effectiveness under Persona-Conditioned Judging.Fig-
ure 2 reports system-level NDCG@10 scores computed using persona-
conditioned relevance judgments on DL20 (top row) and RAG24
(bottom row), plotted against human-derived effectiveness. Each
point corresponds to a submitted retrieval system under a particular
assessor perspective. The diagonal line ( 𝑦=𝑥 ) indicates perfect
agreement with human-derived effectiveness, and deviations from
the diagonal reflect persona-induced changes in the evaluation.
Across most models and datasets, the scatter plots exhibit a
strong monotonic relationship with the human baseline, indicat-
ing that systems ranked highly under human judgments generally
remain highly ranked under persona-conditioned judging. For high-
capacity models, persona-conditioned effectiveness estimates tend
to form approximately parallel bands around the diagonal rather
than large-scale crossing patterns. The main visible exception is
GPT-4o on RAG24, where several assessor perspectives pull effec-
tiveness estimates below the diagonal, consistent with the lower
UMBRELA agreement observed in Section 4.1.1. This suggests thatassessor perspectives primarily introduce systematic shifts in scor-
ing behavior, such as stricter or more lenient effectiveness estima-
tion, while largely preserving relative system ordering.
Across assessor roles, Query and Domain perspectives often
overlap closely, whereas Orthogonal perspectives tend to produce
larger shifts in effectiveness estimates. Evidence and GAP usu-
ally remain closer to the human baseline for stronger models, al-
though this behavior varies across datasets and model families. The
persona-source comparison shows the same pattern observed at
the judgment level (Section 4.2): broadly similar behavior, with
differences concentrated in role–model combinations rather than
as a consistent source-level effect.
Overall, the scatter plots indicate that persona conditioning gen-
erally preserves the global structure of system effectiveness while in-
troducing assessor-dependent shifts in evaluation scale. Large-scale
rank inversions remain uncommon for stronger models, whereas
lower-capacity models exhibit greater assessor sensitivity and more
pronounced persona-dependent variation.
4.3.2 Ranking Consistency Across Models.To assess ranking sta-
bility under persona-conditioned judging, we compute system rank-
ings from LLM-derived NDCG@10 scores and compare them against
rankings derived from human relevance judgments. We report
Kendall’s 𝜏and Rank-Biased Overlap (RBO, 𝜙=0.9), where Kendall’s
𝜏measures global rank-order consistency and RBO emphasizes
agreement among top-ranked systems. Table 4 summarizes ranking
agreement across datasets, assessor perspectives, and models.

CIKM ’26, November 7–11, 2026, Rome, Italy Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca Demartini
0.1 0.2
Human NDCG@100.050.100.150.200.25DL20 
 
 PERSONA NDCG@10GPT-4o
0.1 0.2
Human NDCG@10LLaMA-3.1-70B
0.1 0.2
Human NDCG@10Qwen-2.5-72B
0.1 0.2
Human NDCG@10GPT-4o-mini
0.1 0.2
Human NDCG@10LLaMA-3.1-8B
0.1 0.2
Human NDCG@10Qwen-2.5-7B
0.25 0.50 0.75
Human NDCG@100.200.400.600.80RAG24 
 
 PERSONA NDCG@10GPT-4o
0.25 0.50 0.75
Human NDCG@10LLaMA-3.1-70B
0.25 0.50 0.75
Human NDCG@10Qwen-2.5-72B
0.25 0.50 0.75
Human NDCG@10GPT-4o-mini
0.25 0.50 0.75
Human NDCG@10LLaMA-3.1-8B
0.25 0.50 0.75
Human NDCG@10Qwen-2.5-7B
PersonaHub Query
PersonaHub DomainPersonaHub Orthogonal
USPersona QueryUSPersona Domain
USPersona OrthogonalEvidence GAP UMBRELA
Figure 2:Human- vs. LLM-derived retrieval effectiveness under persona-conditioned judging for DL20 (top) and RAG24 (bottom). Each
point represents a retrieval system scored using mean NDCG@10 across queries.
Table 4:Agreement between LLM-derived and human system rankings based on NDCG@10, measured using Kendall’s 𝜏and RBO(𝜙= 0.9)
across datasets, models, and assessor perspectives. PersonaHub and USPersona variants are grouped by Query (Q), Domain (D), and
Orthogonal (O) prompts; UMB. denotes UMBRELA.
Dataset MetricGPT-4o GPT-4o-mini
PersonaHub USPersonaEvidence GAP UMB.PersonaHub USPersonaEvidence GAP UMB.
Q D O Q D O Q D O Q D O
DL20𝜏.917 .924.952.924 .933 .905 .894 .866 .937 .932 .938 .936.951.932 .937.951.939 .928
RBO .866 .702 .707 .697 .780 .626.876.555 .698 .710 .706 .697 .713 .697 .705.788.703 .679
RAG24𝜏.882 .930.944.913 .898 .875 .857 .894 .903 .901 .899 .903.922.896 .901 .916 .918 .913
RBO .604 .644.985.641 .641 .606 .605 .642 .665.665.642 .643 .641 .640 .664 .642 .641 .641
Dataset MetricLLaMA-3.1-70B LLaMA-3.1-8B
PersonaHub USPersonaEvidence GAP UMB.PersonaHub USPersonaEvidence GAP UMB.
Q D O Q D O Q D O Q D O
DL20𝜏.935 .939 .938.946.944 .911 .938 .936 .932 .707 .814 .849 .741 .815 .716 .627 .721.902
RBO .715 .736 .736 .707.742.699 .716 .735 .708 .393 .652.655.613 .457 .348 .542 .393 .648
RAG24𝜏.944 .946 .922.958.944 .951 .951 .927 .900 .815 .812 .749 .792 .784 .741 .824 .807.853
RBO .991 .992 .644 .994 .983.995.992 .991 .639 .987 .988 .290 .588 .803 .288.989.883 .803
Dataset MetricQwen-2.5-72B Qwen-2.5-7B
PersonaHub USPersonaEvidence GAP UMB.PersonaHub USPersonaEvidence GAP UMB.
Q D O Q D O Q D O Q D O
DL20𝜏.926.942.936 .929 .935 .921 .939 .938 .936 .890 .903 .868 .869.932.812 .892 .897 .896
RBO .700 .699 .712 .698 .723 .645.769.706 .693 .633 .628.852.550 .762 .617 .771 .713 .746
RAG24𝜏.908 .900 .892 .944 .910 .908.953.901 .902 .930.968.867 .960 .898 .834 .914 .919 .925
RBO .663 .640 .638 .991 .663 .663.992.639 .640 .643.892.294 .890 .641 .586 .640 .641 .662
Overall, high-capacity models generally preserve strong ranking
agreement with human-derived system rankings across most asses-
sor perspectives. LLaMA-3.1-70B, Qwen-2.5-72B, GPT-4o-mini, andGPT-4o usually achieve high Kendall’s 𝜏, indicating that persona
conditioning rarely disrupts global system ordering for stronger

Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation CIKM ’26, November 7–11, 2026, Rome, Italy
models. In contrast, lower-capacity models, particularly LLaMA-
3.1-8B, exhibit reduced ranking stability and substantially lower
RBO values under several assessor perspectives, indicating greater
instability among top-ranked systems.
Across datasets, ranking stability does not follow a uniform
DL20–RAG24 pattern. Kendall’s 𝜏remains high for stronger mod-
els on both datasets, but RBO varies more sharply across assessor
perspectives and models. This suggests that persona conditioning
often preserves global system ordering while still affecting the rela-
tive ordering of top-ranked systems. The effect is most visible for
smaller models, where some assessor perspectives produce substan-
tial top-rank volatility despite moderate global rank agreement.
Persona effects are role-dependent. Domain, Orthogonal, and
Evidence perspectives sometimes achieve ranking agreement com-
parable to or higher than UMBRELA, particularly for high-capacity
models, but this pattern is not consistent across datasets, models,
or stability metrics. No single assessor perspective is uniformly
most stable. Instead, ranking stability depends on the interaction
between model capacity, dataset, metric, and assessor perspective.
The persona-source comparison shows the same pattern as at the
judgment level (Section 4.2): persona source has a secondary effect
relative to model capacity, dataset, and assessor role.
4.4 RQ4: System-Level Sensitivity
Although global ranking agreement remains relatively stable, indi-
vidual retrieval systems may still move substantially under persona-
conditioned judging. We therefore analyze rank displacement rel-
ative to UMBRELA to identify which assessor perspectives and
retrieval systems are most evaluator-sensitive.
4.4.1 Rank-Displacement Sensitivity Analysis.To quantify localized
movement, we analyze system-level rank displacement relative to
UMBRELA using the sensitivity measures defined in Section 3.5.
Table 5 summarizes rank displacement by assessor perspective. US-
Persona Orthogonal yields the highest mean absolute rank displace-
ment on both DL20 (2.31) and RAG24 (2.66), indicating that con-
trastive skill-grounded personas act as strong evaluator-sensitivity
probes. PersonaHub Orthogonal also produces elevated displace-
ment on RAG24 (2.22) but remains moderate on DL20 (1.67), so
the Orthogonal role is not uniformly the most disruptive perspec-
tive. Domain perspectives consistently induce among the smaller
shifts on both datasets (USPersona Domain: 1.46 on DL20, 1.55
on RAG24), while GAP behaves inconsistently—lowest on RAG24
(1.43) but third-highest on DL20 (2.11). RAG24 shows larger score
perturbations overall, especially in mean absolute ΔNDCG@10,
peaking at 0.052 for USPersona Orthogonal compared to 0.013 on
DL20. This suggests greater assessor sensitivity in broader and more
interpretive retrieval settings. Despite these perturbations, the aver-
age score changes remain bounded, indicating localized movement
rather than large-scale effectiveness collapse. To examine the role
of model capacity, we additionally aggregate system sensitivity by
model. Table 6 reports mean system-rank displacement with 95%
bootstrap confidence intervals, computed by resampling retrieval
systems with replacement 1,000 times within each dataset–model
pair.
Model capacity strongly modulates sensitivity to persona con-
ditioning, although the pattern is not strictly monotonic acrossTable 5:System-level sensitivity relative to UMBRELA. Mean
and max|Δ𝑟|report average and largest rank displacement; mean
|ΔNDCG @10|reports average score perturbation. Bold indicates
the largest mean|Δ𝑟|per dataset.
Dataset Persona Mean Max Mean
|Δ𝑟| |Δ𝑟| |ΔNDCG@10|
DL20 PersonaHub Query 2.05 17 0.003
USPersona Query 1.99 20 0.003
PersonaHub Domain 1.75 17 0.001
USPersona Domain 1.46 11 0.002
PersonaHub Orthogonal 1.67 13 0.010
USPersona Orthogonal 2.31270.013
Evidence 2.21320.007
GAP 2.11 20 0.002
RAG24 PersonaHub Query 1.83 17 0.004
USPersona Query 2.29 17 0.003
PersonaHub Domain 1.96 17 0.000
USPersona Domain 1.55 16 0.005
PersonaHub Orthogonal 2.22200.035
USPersona Orthogonal 2.66 20 0.052
Evidence 1.97 17 0.016
GAP 1.43 18 0.004
Table 6:Model sensitivity to persona-conditioned evaluation. Mean
|Δ𝑟|reports average absolute system-rank displacement with 95%
bootstrap confidence intervals from resampling retrieval systems.
Max|Δ𝑟|reports the largest observed movement for any retrieval
system.
Dataset Model Mean|Δ𝑟|[95% CI] Max|Δ𝑟|
DL20 LLaMA-3.1-8B5.19 [4.60, 5.79] 32
Qwen-2.5-7B 2.15 [1.79, 2.54] 15
GPT-4o 1.56 [1.21, 1.96] 12
GPT-4o-mini 1.10 [0.87, 1.37] 8
Qwen-2.5-72B 0.86 [0.67, 1.07] 8
LLaMA-3.1-70B 0.80 [0.64, 0.98] 5
RAG24 LLaMA-3.1-8B4.17 [3.48, 4.88] 20
LLaMA-3.1-70B 2.21 [1.59, 2.91] 12
Qwen-2.5-7B 2.01 [1.61, 2.53] 19
GPT-4o 1.80 [1.46, 2.20] 17
GPT-4o-mini 0.99 [0.69, 1.33] 10
Qwen-2.5-72B 0.75 [0.52, 1.01] 10
datasets. LLaMA-3.1-8B produces the largest average system-rank
displacement on both datasets, with bootstrap intervals that re-
main clearly higher than those of the higher-capacity judge models.
Qwen-2.5-7B also exhibits elevated instability, while Qwen-2.5-72B
and GPT-4o-mini remain comparatively stable. LLaMA-3.1-70B is
highly stable on DL20 but more sensitive on RAG24, indicating that
dataset characteristics also shape persona-induced rank displace-
ment.
These results show that persona effects are concentrated rather
than uniformly distributed across rankings. Persona conditioning

CIKM ’26, November 7–11, 2026, Rome, Italy Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca Demartini
rarely disrupts global rankings for high-capacity models, but it ex-
poses localized evaluator-sensitive systems and amplifies instability
in lower-capacity evaluators.
4.4.2 Sensitivity Across System Types.Next, we examine whether
assessor sensitivity is concentrated in particular types of retrieval
systems. Following prior TREC Deep Learning analyses, which dis-
tinguish reranking and neural language model runs [ 16], and TREC
RAG work that frames systems around retrieval-augmented gener-
ation pipelines [ 40], we identify the most affected runs using broad
system-type descriptions rather than official track labels. These
descriptions are derived from official TREC run descriptions and
associated system papers, and are used only to interpret sensitivity
patterns rather than as additional experimental variables.
Table 7 reports representative systems with the highest evalu-
ator sensitivity in each dataset. The reported Mean |Δ𝑟|and Max
|Δ𝑟| ranges are computed over the listed systems, where Δ𝑟de-
notes rank displacement relative to UMBRELA across persona set-
tings. On DL20, several of the most evaluator-sensitive runs are
transformer-based neural ranking and reranking systems, including
bigIR-DCT-T5-F ,fr_pass_roberta , and pash_r1 . These systems
exhibit larger rank displacement than typical runs under persona-
conditioned judging.
On RAG24, the most evaluator-sensitive runs we identify are
retrieval-augmented and generation-oriented pipelines, including
iiia_standard_* andielab-* . These runs exhibit larger average
displacement and comparable or higher maximum rank shifts than
the most sensitive DL20 runs, suggesting that RAG-style settings
may amplify assessor sensitivity. This pattern is also consistent with
the earlier model analysis: the largest perturbations are typically
associated with smaller models such as LLaMA-3.1-8B and Qwen-
2.5-7B, whereas higher-capacity models produce more bounded
movement.
We further examined whether persona-induced rank movement
is directionally consistent across models. Most systems exhibit
mixed or unchanged movement directions, particularly on RAG24,
where no system moves consistently up or down across all models.
On DL20, only 7 out of 472 system–persona pairs show consis-
tent directional behavior. For example, fr_pass_roberta moves
upward under both PersonaHub Orthogonal and USPersona Or-
thogonal perspectives across all six models, with mean rank shifts
of−4.17and−4.83, respectively. This suggests that system sensitiv-
ity emerges from the interaction between assessor perspective and
model capacity rather than being a fixed property of the retrieval
system alone.
Overall, these results suggest that persona conditioning acts as
a diagnostic stress test that exposes evaluator-sensitive retrieval
architectures rather than uniformly disrupting system rankings.
5 Conclusions and Implications
This work examined how assessor perspectives influence LLM-
based relevance judgments and downstream IR evaluation. Across
agreement analysis, ranking stability, and system-sensitivity analy-
sis, a consistent set of patterns emerges.
First, persona conditioning systematically alters relevance judg-
ments, but the magnitude and structure of these effects depend
strongly on model capacity and dataset characteristics. High-capacitymodels generally maintain strong agreement with both UMBRELA
and human judgments, while lower-capacity models exhibit sub-
stantially greater instability under changes in assessor perspective.
This indicates that assessor conditioning is meaningful only when
the underlying model can reliably support the imposed evaluator
constraints; otherwise, persona conditioning primarily amplifies
judgment variability.
Second, persona effects are localized rather than globally dis-
ruptive. Although persona-conditioned judgments can induce no-
ticeable shifts in effectiveness estimates and individual system
ranks, overall ranking agreement with human evaluation remains
relatively stable for stronger models. The scatter plots and rank-
correlation analyses show that persona conditioning usually pre-
serves the global structure of system rankings while producing
bounded movement among individual systems, particularly near
the top of the ranking. This suggests that assessor perspectives
mainly change scoring behavior and local rank positions, rather
than producing widespread rank inversions.
Third, assessor sensitivity is not uniformly distributed across
retrieval systems. Persona-induced perturbations concentrate on
particular systems and system types: among the most evaluator-
sensitive runs are transformer-based neural ranking/reranking
systems on DL20 and retrieval-augmented or generation-oriented
pipelines on RAG24. These systems exhibit larger rank displace-
ment under persona-conditioned judging than more stable runs.
The strongest instabilities are associated with smaller models such
as LLaMA-3.1-8B and Qwen-2.5-7B, indicating that low-capacity
evaluators amplify persona-induced ranking sensitivity.
Finally, persona source has a secondary effect compared with
assessor role and judge-model capacity. Abstract PersonaHub per-
sonas and skill-grounded USPersona profiles produce broadly simi-
lar behavior, with differences concentrated in specific role–model
combinations rather than forming consistent source-level effects.
USPersona Orthogonal induces the largest average rank displace-
ment on both datasets, while PersonaHub Orthogonal shows el-
evated displacement mainly on RAG24. This suggests that con-
trastive skill-grounded perspectives function as particularly strong
evaluator-sensitivity probes.
Overall, the results position persona conditioning as a diagnostic
mechanism for exposing assessor sensitivity rather than as a uni-
versally beneficial judging strategy. When applied to sufficiently
capable models, assessor perspectives reveal where retrieval evalu-
ation is sensitive to framing, interpretation, and evaluator assump-
tions while largely preserving global ranking structure. This makes
persona-conditioned judging useful for stress-testing LLM-based
IR evaluation pipelines and identifying evaluator-sensitive retrieval
systems and architectures. These findings suggest that controlled as-
sessor variation can complement single-configuration LLM judging
by revealing where evaluation outcomes are sensitive to assessor
perspective.
GenAI Usage Disclosure
Generative AI tools were used to support manuscript editing, word-
ing refinement, grammar checking, and LaTeX formatting. They
were also used to assist with code debugging and implementation
support during analysis. All research design, experimental setup,
data processing decisions, analyses, results, interpretations, and

Persona Conditioning as an Assessor-Sensitivity Probe for LLM-Based IR Evaluation CIKM ’26, November 7–11, 2026, Rome, Italy
Table 7:Representative most evaluator-sensitive retrieval systems under persona-conditioned judging. The reported sensitivity ranges are
computed over the listed systems and measured as mean absolute rank displacement relative to UMBRELA across persona settings.
Dataset Representative Systems Mean|Δ𝑟|Max|Δ𝑟|Interpreted System Type
DL20bigIR-DCT-T5-F,pash_r1,fr_pass_roberta3.69–4.19 18–27 Transformer-based neural ranking/reranking systems
RAG24iiia_standard_*,ielab-*4.31–8.75 19–20 Retrieval-augmented and generation-oriented pipelines
scholarly claims were conducted, verified, and approved by the
authors.
References
[1]Marwah Alaofi, Paul Thomas, Falk Scholer, and Mark Sanderson. 2024. LLMs
can be Fooled into Labelling a Document as Relevant: best café near me; this
paper is perfectly relevant. InProceedings of the 2024 Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval in the
Asia Pacific Region (SIGIR-AP ’24). 32–41. doi:10.1145/3673791.3698431
[2]Marwah Alaofi, Paul Thomas, Falk Scholer, and Mark Sanderson. 2026. On the
Use of LLMs for Relevance Labelling.ACM Transactions on Information Systems
(2026). doi:10.1145/3788872
[3]Negar Arabzadeh and Charles L.A. Clarke. 2025. Benchmarking LLM-based
Relevance Judgment Methods. InProceedings of the 48th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR ’25).
3194–3204. doi:10.1145/3726302.3730305
[4]Peter Bailey, Nick Craswell, Ian Soboroff, Paul Thomas, Arjen P. de Vries, and
Emine Yilmaz. 2008. Relevance Assessment: Are Judges Exchangeable and Does
It Matter?. InProceedings of the 31st Annual International ACM SIGIR Conference
on Research and Development in Information Retrieval. ACM, 667–674. doi:10.
1145/1390334.1390447
[5]Krisztian Balog, Donald Metzler, and Zhen Qin. 2025. Rankers, Judges, and
Assistants: Towards Understanding the Interplay of LLMs in Information Re-
trieval Evaluation. InProceedings of the 48th International ACM SIGIR Conference
on Research and Development in Information Retrieval (SIGIR ’25). 3865–3875.
https://doi.org/10.1145/3726302.3730348
[6]Pietro Bernardelle, Stefano Civelli, Leon Fröhling, Riccardo Lunardi, Kevin Roi-
tero, and Gianluca Demartini. 2025. Political ideology shifts in large language
models.arXiv preprintarXiv:2508.16013 (2025). https://arxiv.org/abs/2508.16013
[7]Pietro Bernardelle, Leon Froehling, Stefano Civelli, and Gianluca Demartini. 2026.
SubData: Bridging Heterogeneous Datasets to Enable Theory-Driven Evaluation
of Political and Demographic Perspectives in LLMs. InProceedings of the the fifth
edition of NLPerspectives, Shiran Dudy, Gavin Abercrombie, Valerio Basile, Elisa
Leonardelli, and Simona Frenda (Eds.). ELRA Language Resources Association
(ELRA), Palma, Mallorca (Spain), 84–97. doi:10.63317/2uppkbro3uvq
[8]Pietro Bernardelle, Leon Fröhling, Stefano Civelli, Riccardo Lunardi, Kevin Roi-
tero, and Gianluca Demartini. 2025. Mapping and influencing the political ideol-
ogy of large language models using synthetic personas. InCompanion Proceedings
of the ACM on Web Conference 2025. 864–867. doi:10.1145/3701716.3715578
[9]Ben Carterette and Ian Soboroff. 2010. The Effect of Assessor Error on IR System
Evaluation. InProceedings of the 33rd International ACM SIGIR Conference on
Research and Development in Information Retrieval. ACM, 539–546. doi:10.1145/
1835449.1835540
[10] Nuo Chen, Hanpei Fang, Jiqun Liu, Tetsuya Sakai, and Xiao-Ming Wu. 2026.
Judging with Personality and Confidence: A Study on Personality-Conditioned
LLM Relevance Assessment.arXiv preprint arXiv:2601.01862(2026). https:
//arxiv.org/abs/2601.01862
[11] Nuo Chen, Jiqun Liu, Xiaoyu Dong, Qijiong Liu, Tetsuya Sakai, and Xiao-Ming
Wu. 2024. AI Can Be Cognitively Biased: An Exploratory Study on Threshold
Priming in LLM-Based Batch Relevance Assessment. InProceedings of the 2024
Annual International ACM SIGIR Conference on Research and Development in
Information Retrieval in the Asia Pacific Region (SIGIR-AP ’24). ACM, 54–63. doi:10.
1145/3673791.3698420
[12] Stefano Civelli, Pietro Bernardelle, and Gianluca Demartini. 2025. The Impact of
Persona-based Political Perspectives on Hateful Content Detection. InCompanion
Proceedings of the ACM on Web Conference 2025. 1963–1968. doi:10.1145/3701716.
3718383
[13] Stefano Civelli, Pietro Bernardelle, Nardiena A. Pratama, and Gianluca Demartini.
2026. Ideology-Based LLMs for Content Moderation.ACM Trans. Intell. Syst.
Technol.(April 2026). doi:10.1145/3810946
[14] Charles L. A. Clarke and Laura Dietz. 2025. LLM-based Relevance Assessment
Still Can’t Replace Human Relevance Assessment. InProceedings of the 11th
International Workshop on Evaluating Information Access (EVIA 2025). National
Institute of Informatics, Tokyo, Japan. doi:10.20736/000200210
[15] Jacob Cohen. 1968. Weighted Kappa: Nominal Scale Agreement Provision for
Scaled Disagreement or Partial Credit.Psychological Bulletin70, 4 (1968), 213–220.doi:10.1037/h0026256
[16] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2020. Overview
of the TREC 2020 Deep Learning Track. InProceedings of the Twenty-Ninth Text
REtrieval Conference (TREC 2020) (NIST Special Publication, Vol. 1266). National
Institute of Standards and Technology (NIST). https://trec.nist.gov/pubs/trec29/
papers/OVERVIEW.DL.pdf
[17] Laura Dietz, Oleg Zendel, Peter Bailey, Charles L. A. Clarke, Ellese Cotterill, Jeff
Dalton, Faegheh Hasibi, Mark Sanderson, and Nick Craswell. 2025. Principles and
Guidelines for the Use of LLM Judges. InProceedings of the 2025 International ACM
SIGIR Conference on Innovative Concepts and Theories in Information Retrieval
(ICTIR ’25). ACM, 218–229. doi:10.1145/3731120.3744588
[18] Guglielmo Faggioli, Laura Dietz, Charles L.A. Clarke, Gianluca Demartini,
Matthias Hagen, Claudia Hauff, Noriko Kando, Evangelos Kanoulas, Martin
Potthast, Benno Stein, and Henning Wachsmuth. 2023. Perspectives on Large
Language Models for Relevance Judgment. InProceedings of the 2023 ACM SIGIR
International Conference on the Theory of Information Retrieval (ICTIR ’23). 39–50.
https://doi.org/10.1145/3578337.3605136
[19] Hanpei Fang, Sijie Tao, Nuo Chen, Kai-Xin Chang, and Tetsuya Sakai. 2025.
Do Large Language Models Favor Recent Content? A Study on Recency Bias
in LLM-Based Reranking. InProceedings of the 2025 Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval in the
Asia Pacific Region (SIGIR-AP ’25). ACM, 85–94. doi:10.1145/3767695.3769493
[20] Naghmeh Farzi and Laura Dietz. 2025. Criteria-Based LLM Relevance Judgments.
InProceedings of the 2025 International ACM SIGIR Conference on Innovative
Concepts and Theories in Information Retrieval (ICTIR ’25). ACM, 254–263. doi:10.
1145/3731120.3744591
[21] Naghmeh Farzi and Laura Dietz. 2025. Does UMBRELA Work on Other LLMs?.
InProceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’25). ACM, 3214–3222. doi:10.1145/
3726302.3730317
[22] Ryan Lin Feng, Keyu Tian, Hanming Zheng, Congjing Zhang, Li Zeng, and Shuai
Huang. 2025. CrowdLLM: Building LLM-Based Digital Populations Augmented
with Generative Models.arXiv preprint arXiv:2512.07890(2025). https://arxiv.
org/abs/2512.07890
[23] Leon Fröhling, Gianluca Demartini, and Dennis Assenmacher. 2025. Personas
with Attitudes: Controlling LLMs for Diverse Data Annotation. InProceedings of
the 9th Workshop on Online Abuse and Harms (WOAH). Association for Computa-
tional Linguistics, Vienna, Austria, 468–481. https://aclanthology.org/2025.woah-
1.43/
[24] Tao Ge, Xin Chan, Xiaoyang Wang, Dian Yu, Haitao Mi, and Dong Yu. 2024.
Scaling Synthetic Data Creation with 1,000,000,000 Personas.arXiv preprint
arXiv:2406.20094 (2024). https://arxiv.org/abs/2406.20094
[25] Ariel Gera, Odellia Boni, Yotam Perlitz, Roy Bar-Haim, Lilach Eden, and Asaf
Yehudai. 2025. JuStRank: Benchmarking LLM Judges for System Ranking. In
Proceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). Association for Computational Linguistics,
Vienna, Austria, 682–712. doi:10.18653/v1/2025.acl-long.34
[26] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated Gain-Based Evaluation
of IR Techniques.ACM Transactions on Information Systems20, 4 (2002), 422–446.
doi:10.1145/582415.582418
[27] Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and
Yixin Zhu. 2023. Evaluating and Inducing Personality in Pre-trained Language
Models. InProceedings of the 37th International Conference on Neural Infor-
mation Processing Systems (NeurIPS ’23). Curran Associates, Inc., Article 466,
10622–10643 pages. https://proceedings.neurips.cc/paper_files/paper/2023/hash/
21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html
[28] Hang Jiang, Xiajie Zhang, Xubo Cao, Cynthia Breazeal, Deb Roy, and Jad Kabbara.
2024. PersonaLLM: Investigating the Ability of Large Language Models to Express
Personality Traits. InFindings of the Association for Computational Linguistics:
NAACL 2024. Association for Computational Linguistics, Mexico City, Mexico,
3605–3627. doi:10.18653/v1/2024.findings-naacl.229
[29] Jüri Keller, Maik Fröbe, Björn Engelmann, Fabian Haak, Timo Breuer, Birger
Larsen, and Philipp Schaer. 2026. Formalized Information Needs Improve Large-
Language-Model Relevance Judgments. InProceedings of the 49th International
ACM SIGIR Conference on Research and Development in Information Retrieval
(SIGIR ’26). ACM. arXiv:2604.04140.

CIKM ’26, November 7–11, 2026, Rome, Italy Samaneh Mohtadi, Pietro Bernardelle, Joel Mackenzie, and Gianluca Demartini
[30] Maurice G. Kendall. 1938. A New Measure of Rank Correlation.Biometrika30,
1-2 (1938), 81–93. doi:10.1093/biomet/30.1-2.81
[31] David La Barbera, Riccardo Lunardi, Mengdie Zhuang, and Kevin Roitero. 2025.
Impersonating the Crowd: Evaluating LLMs’ Ability to Replicate Human Judg-
ment in Misinformation Assessment. InProceedings of the 2025 International ACM
SIGIR Conference on Innovative Concepts and Theories in Information Retrieval
(ICTIR ’25). ACM, 12–21. doi:10.1145/3731120.3744581
[32] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang
Zhu. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.
InProceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics, Singapore, 2511–2522.
doi:10.18653/v1/2023.emnlp-main.153
[33] Angel Felipe Magnossão de Paula, J. Shane Culpepper, Alistair Moffat,
Sachin Pathiyan Cherumanal, Falk Scholer, and Johanne Trippas. 2025. The
Effects of Demographic Instructions on LLM Personas. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval (SIGIR ’25). ACM, 3045–3049. doi:10.1145/3726302.3730255
[34] Yev Meyer and Dane Corneil. 2025. Nemotron-Personas-USA: Synthetic Personas
Aligned to Real-World Distributions. https://huggingface.co/datasets/nvidia/
Nemotron-Personas-USA Accessed: 2025-09-27.
[35] Stefano Mizzaro. 1997. Relevance: The Whole History.Journal of the American
Society for Information Science48, 9 (1997), 810–832. doi:10.1002/(SICI)1097-
4571(199709)48:9<810::AID-ASI6>3.0.CO;2-U
[36] Samaneh Mohtadi and Gianluca Demartini. 2026. Query-Document Dense
Vectors for LLM Relevance Judgment Bias Analysis. InProceedings of the 48th
European Conference on Information Retrieval (ECIR). arXiv:2601.01751 [cs.IR]
https://arxiv.org/abs/2601.01751
[37] Samaneh Mohtadi, Kevin Roitero, Stefano Mizzaro, and Gianluca Demartini. 2026.
The Effect of Document Summarization on LLM-Based Relevance Judgments. In
Proceedings of the 48th European Conference on Information Retrieval (ECIR 2026).
arXiv:2512.05334 [cs.IR] https://arxiv.org/abs/2512.05334
[38] NIST TREC Organizers. 2024. TREC 2024 Retrieval-Augmented Generation
(RAG) Track Guidelines. https://trec-rag.github.io/annoucements/2024-track-
guidelines/ Accessed: 2025-09-27.
[39] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al .2022.
Training Language Models to Follow Instructions with Human Feedback. In
Proceedings of the 36th International Conference on Neural Information Processing
Systems (NeurIPS ’22). 27730–27744. https://dl.acm.org/doi/10.5555/3600270.
3602281
[40] Ronak Pradeep, Nandan Thakur, Sahel Sharifymoghaddam, Eric Zhang, Ryan
Nguyen, Daniel Campos, Nick Craswell, and Jimmy Lin. 2025. Ragnarök: A
Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented
Generation Track. InProceedings of the 47th European Conference on Information
Retrieval (ECIR 2025), Part I. Springer, 132–148. doi:10.1007/978-3-031-88708-6_9
[41] Hossein A. Rahmani, Clemencia Siro, Mohammad Aliannejadi, Nick Craswell,
Charles L. A. Clarke, Guglielmo Faggioli, Bhaskar Mitra, Paul Thomas, and Emine
Yilmaz. 2025. Judging the Judges: A Collection of LLM-Generated Relevance
Judgements.arXiv preprint arXiv:2502.13908(2025). https://arxiv.org/abs/2502.
13908
[42] Tefko Saracevic. 1996. Relevance Reconsidered. InInformation Science: Integration
in Perspectives (Proceedings of the Second Conference on Conceptions of Library
and Information Science (CoLIS 2)), Peter Ingwersen and Niels Ole Pors (Eds.).
Copenhagen, Denmark, 201–218.
[43] Melanie Sclar, Yejin Choi, and Alane Suhr. 2023. Quantifying Language Models’
Sensitivity to Spurious Features in Prompt Design or: How I Learned to Start
Worrying About Prompt Formatting.arXiv preprint arXiv:2310.11324(2023).
doi:10.48550/arXiv.2310.11324
[44] Ian Soboroff. 2024. Don’t Use LLMs to Make Relevance Judgments.Information
Retrieval Research Journal(2024). doi:10.54195/irrj.19625
[45] Eero Sormunen. 2002. Liberal Relevance Criteria of TREC: Counting on Negligible
Documents?. InProceedings of the 25th Annual International ACM SIGIR Conference
on Research and Development in Information Retrieval. ACM, 324–330. doi:10.
1145/564376.564433
[46] Aleksandra Sorokovikova, Sharwin Rezagholi, Natalia Fedorova, and Ivan P.
Yamshchikov. 2024. LLMs Simulate Big5 Personality Traits: Further Evidence.InProceedings of the 1st Workshop on Personalization of Generative AI Systems
(PERSONALIZE 2024). Association for Computational Linguistics, St. Julians,
Malta, 83–87. https://aclanthology.org/2024.personalize-1.7/
[47] Paul Thomas, Seth Spielman, Nick Craswell, and Bhaskar Mitra. 2024. Large
Language Models Can Accurately Predict Searcher Preferences. InProceedings
of the 47th International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR ’24). 1930–1940. doi:10.1145/3626772.3657707
[48] Shivani Upadhyay, Ehsan Kamalloo, and Jimmy Lin. 2024. LLMs Can Patch Up
Missing Relevance Judgments in Evaluation. arXiv:2405.04727 [cs.IR] doi:10.
48550/arXiv.2405.04727
[49] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick
Craswell, Ian Soboroff, and Jimmy Lin. 2025. A Large-Scale Study of Relevance
Assessments with Large Language Models Using UMBRELA. InProceedings of the
2025 International ACM SIGIR Conference on Innovative Concepts and Theories in In-
formation Retrieval (ICTIR ’25). 358–368. https://doi.org/10.1145/3731120.3744605
[50] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, and Jimmy
Lin. 2024. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing
RELevance Assessor.arXiv preprint arXiv:2406.06519(2024). doi:10.48550/arXiv.
2406.06519
[51] Ellen M. Voorhees. 2000. Variations in Relevance Judgments and the Measurement
of Retrieval Effectiveness. InProceedings of the 21st Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval. ACM,
315–323. doi:10.1145/290941.291017
[52] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo
Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large Language
Models are not Fair Evaluators. InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). Association
for Computational Linguistics, Bangkok, Thailand, 9440–9450. doi:10.18653/v1/
2024.acl-long.511
[53] Yumeng Wang, Jirui Qi, Catherine Chen, Panagiotis Eustratiadis, and Suzan
Verberne. 2026. How Role-Play Shapes Relevance Judgment in Zero-Shot LLM
Rankers. InAdvances in Information Retrieval (ECIR ’26). Springer, 228–242. doi:10.
1007/978-3-032-21289-4_15
[54] William Webber, Alistair Moffat, and Justin Zobel. 2010. A Similarity Measure
for Indefinite Rankings.ACM Transactions on Information Systems28, 4 (2010),
20:1–20:38. doi:10.1145/1852102.1852106
[55] Jiayi Ye, Yanbo Wang, Yue Huang, Dongping Chen, Qihui Zhang, Nuno Moniz,
Tian Gao, Werner Geyer, Chao Huang, Pin-Yu Chen, Nitesh V. Chawla, and
Xiangliang Zhang. 2025. Justice or Prejudice? Quantifying Biases in LLM-as-a-
Judge. InProceedings of the International Conference on Learning Representations
(ICLR ’25).
[56] Chuting Yu, Hang Li, Guido Zuccon, Joel Mackenzie, and Teerapong Leelanupab.
2026. When LLM Judges Inflate Scores: Exploring Overrating in Relevance
Assessment.arXiv preprintarXiv:2602.17170 (2026). https://doi.org/10.48550/
arXiv.2602.17170
[57] Pengwei Zhan, Zhen Xu, Qian Tan, Jie Song, and Ru Xie. 2024. Unveiling the Lex-
ical Sensitivity of LLMs: Combinatorial Optimization for Prompt Enhancement.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing (EMNLP). Association for Computational Linguistics, Miami, Florida,
USA, 5128–5154. doi:10.18653/v1/2024.emnlp-main.295
[58] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-
Bench and Chatbot Arena. InAdvances in Neural Information Processing Systems
(NeurIPS).
[59] Mingqian Zheng, Jiaxin Pei, Lajanugen Logeswaran, Moontae Lee, and David
Jurgens. 2024. When “A Helpful Assistant” Is Not Really Helpful: Personas in
System Prompts Do Not Improve Performance of Large Language Models. In
Findings of the Association for Computational Linguistics: EMNLP 2024. Association
for Computational Linguistics, Miami, Florida, USA, 15126–15154. doi:10.18653/
v1/2024.findings-emnlp.888
[60] Justin Zobel. 1998. How Reliable Are the Results of Large-Scale Information
Retrieval Experiments?. InProceedings of the 21st Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval. ACM, New
York, NY, USA, 307–314. doi:10.1145/290941.291014