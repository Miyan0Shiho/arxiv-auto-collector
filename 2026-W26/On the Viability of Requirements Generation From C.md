# On the Viability of Requirements Generation From Code: An Experience Report

**Authors**: Alexander Korn, Jone Bartel, Max Unterbusch, Andreas Vogelsang

**Published**: 2026-06-24 08:28:14

**PDF URL**: [https://arxiv.org/pdf/2606.25550v1](https://arxiv.org/pdf/2606.25550v1)

## Abstract
Empirical research in Requirements Engineering is hampered by a lack of adequate datasets that pair source code with corresponding requirements. A tempting route to addressing this lack is the use of Large Language Models to synthesize requirements from existing code bases. We investigate this question by evaluating an LLM-based and RAG-supported agentic approach that generates requirements from source code, verifies their implementation status relying on a human-in-the-loop, and synthetically introduces requirements smells and non-implemented requirements. Our goal was to create datasets that mimic reality and foster empirical RE research. However, during the study, various problems arose, leading to this experience report. Contrary to our initial hypotheses, LLMs were unable to (i) generate non-implemented requirements reliably, (ii) generate high quality requirements, and (iii) reliably introduce synthetic requirements smells. Furthermore, neither an LLM nor a single human-in-the-loop suffices to detect requirements smells reliably. These findings suggest that the generation of code-to-requirements datasets using LLMs is not yet viable and requires human supervision, especially for quality assurance. We critically reflect on our lessons learned and draw relevant conclusions for both researchers and practitioners.

## Full Text


<!-- PDF content starts -->

On the Viability of Requirements Generation From
Code: An Experience Report
Alexander Korn, Jone Bartel, Max Unterbusch, Andreas V ogelsang
paluno – The Ruhr Institute for Software Technology
University of Duisburg-Essen
Essen, Germany
{firstname}.{lastname}@uni-due.de
Abstract—Empirical research in Requirements Engineering
is hampered by a lack of adequate datasets that pair source
code with corresponding requirements. A tempting route to
addressing this lack is the use of Large Language Models to
synthesize requirements from existing code bases. We investigate
this question by evaluating an LLM-based and RAG-supported
agentic approach that generates requirements from source code,
verifies their implementation status relying on a human-in-the-
loop, and synthetically introduces requirements smells and non-
implemented requirements. Our goal was to create datasets
that mimic reality and foster empirical RE research. However,
during the study, various problems arose, leading to this ex-
perience report. Contrary to our initial hypotheses, LLMs were
unable to (i) generate non-implemented requirements reliably, (ii)
generate high quality requirements, and (iii) reliably introduce
synthetic requirements smells. Furthermore, neither an LLM nor
a single human-in-the-loop suffices to detect requirements smells
reliably. These findings suggest that the generation of code-to-
requirements datasets using LLMs is not yet viable and requires
human supervision, especially for quality assurance. We critically
reflect on our lessons learned and draw relevant conclusions for
both researchers and practitioners.
Index Terms—Agentic AI, LLMs, Synthetic Datasets, Require-
ments Smells
© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including
reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or
reuse of any copyrighted component of this work in other works.I. INTRODUCTION
Empirical research inRequirements Engineering (RE)is
based on the quality of the data used. To best support research,
data should (i) be FAIR (findable, accessible, interoperable,
reusable) [1], (ii) contain a realistic number of artifacts, (iii)
contain high-quality requirements, and yet (iv) be realistic
in quality [2], so that real-world imperfections can also be
researched. For supporting research on the relation between
requirements and code [3], datasets should also (v) contain
relevant trace links, e.g. requirements-to-code traces. Although
there are datasets satisfying some of these dimensions, there
is a lack of datasets satisfying all dimensions (cf. Section II).
Ideally, industry datasets could be used, which would render
(iv) useless. However, industry data can rarely be disclosed
for proprietary or legal reasons, leading to a large number of
academic or toy datasets used in RE research [4].
Due to significant advances in the performance ofLarge
Language Models (LLMs), researchers started using LLMs to
generate or synthesize textual artifacts such as code and re-
quirements [5], [6] or even requirements traceability data [7]–[9]. However, the generated datasets show deficiencies in
quality and realism.
Our initial idea was to propose an agentic human-in-the-
loop approach that synthesizes requirements from existing
code bases and link them to the corresponding code loca-
tions. Specifically, we wanted to use Retrieval-Augmented
Generation (RAG) to ground the requirements in code, pre-
sumably leading to high-quality requirements with clear links
to code. We operationalize requirements quality by using
requirements smells as quantifiable proxies for quality issues
in requirements [10], [11]. We hypothesized that the generated
requirements would show a low smell rate and are, in fact,
implemented in the source code. We further hypothesized that
we could extend the approach to generate a control group
of non-implemented requirements with similar properties to
the implemented requirements. Finally, we planned to extend
the approach by an agent that introduces requirements smells
synthetically, such that a realistic dataset would be generated
with explicit labeling of requirements smells.
In this experience report, we describe how we implemented
and evaluated this approach and finally had to withdraw
most of our hypothesis. In contrast, we observe the following
problems:
•The supposedly high-quality requirements still exhibited
a smell rate of 24.5%.
•73.8% of the smell-free butnon-implementedrequire-
ments generated were marked asimplementedby a human
evaluator, showing that the LLM often generates imple-
mented requirements instead.
•The inter-rater agreement for smell detection and appro-
priateness evaluation of generated requirements showed
fair agreement at best.
•Smell evaluation by the human-in-the-loop was highly
subjective and context dependent.
In this experience report, we describe our approach, the
evaluation, the results, and reflect on these. The key contri-
butions are:
•An empirical evaluationof our approach across 2
software projects, examining implementation accuracy,
hallucination rate, generation quality, and human-LLM
reliability on smell detection.
•Concrete lessons learnedon the limitations of LLM-
based requirements generation, specifically around smellarXiv:2606.25550v1  [cs.SE]  24 Jun 2026

labeling subjectivity, the LLM complying with the given
prompts, and codebase-dependency of the results.
•Two publicly available requirements-to-code datasets
generated and peer-reviewed during the study.
DATAAVAILABILITY
All data used in this study, including the requirements-to-
code datasets generated during the evaluation, the experimental
implementation of the presented approach, and the analysis
code used to process the study results, are available in the
supplementary material1.
II. RELATEDWORK
Related work stems from two areas: requirement-to-code-
traceability and the more recent requirement-from-code gen-
eration.
A. Requirement-to-code Datasets
Requirement-and-code datasets provide requirements along
with the specific code traces that implement them. Although
their applicability is broad, the most common use is by
the traceability community interested in creating or recover-
ing traceability links from requirements to source code [3].
Zoogan et al. [12] list all datasets used for requirements
traceability up until 2017. A subset of this list is relevant
as requirement-and-code datasets. Disregarding publicly un-
available datasets, e.g., non-disclosable industry data sets, and
infrequently used datasets, the remaining datasets embody a
trend towards standard datasets being re-used as baselines and
for better comparability2. These standard datasets including
iTrust, eTour, SMOS, and eANCI, indeed are frequently
used in more recent studies [9], [13]–[17]. The code-and-
requirement datasets provide a golden set of requirements
and their trace links to code that were manually crafted and
hence, are usually small in size. Even combined, they do not
suffice to provide sufficient amounts of training data forDeep
Learning (DL)methods [6], [18]. Additionally, these ideal
requirements-and-code datasets deviate from the mixed-quality
in real-world requirement datasets [2]. Hence, we hypothesized
that datasets are more realistic if we use synthetically introduce
requirements smells.
B. Requirements Generation from Code
Xu et al. [7] show the feasibility of generating requirements
from source code. They use existing requirement-to-code
datasets to finetune GPT-models and then generate require-
ments for the application of legacy systems. Dearstyne [8]
tested the generation of requirements from source code as
one of the four applied problems in requirements traceability.
Practitioners find that automatically generated requirements
are comparable in quality to their handcrafted requirements.
Persson et al. [19] propose Code2Req, which allows users
to generate requirements from code. Their approach is based
on RAG-supported LLMs and includes existing requirements
1Supplementary material: https://doi.org/10.6084/m9.figshare.32393787
2We show a tabular view of this list in our supplementary material.sets as context information. They also show the feasibility
of automatic requirement generation from code using LLMs,
although they describe it as not yet practically valuable.
Jin et al. [13] focus on generating requirements from code
with the aim of facilitating the understanding and verification
of code generated by LLMs.
These approaches do not consider two major aspects: First,
they aim to produce exclusively high-quality requirements,
rather than generating a dataset that reflects the characteristics
of real-world requirements, which are of mixed quality [2].
Dearstyne’s work tries to correct for that by incorporating
existing sets of requirements from the given codebase as
context to the LLMs. However, this is not enough since (i)
an idealized set of context requirements leads to the initial
problem of unrealistic requirements, and (ii) a realistic set of
context requirements leads to an inexplicit quality decrease in
output requirements, as reported in her work. Without explic-
itly knowing the location of these qualitative deficiencies, they
are of little use to research. Second, their approaches are fully
automated without human validation or feedback. In contrast,
when producing datasets intended for downstream research or
project work, an additional verification and validation step is
required to ensure the structural and content correctness of the
resulting artifacts. This human feedback can also be used to
guide the requirements generation process.
III. APPROACH
This section describes the approach used for our study.
Figure 1 illustrates the complete architecture.
A. Agentic Approach
Generator Agent.The generator agent is responsible for
producing batches ofnrequirements (in our experiments:
n= 10). To accomplish this, the agent employs RAG to
retrieve code chunks from the target project’s code base. For
each batch, we invoke the generator separatelyntimes, each
time producing acandidaterequirement. The system prompt
deliberately refrains from requiring the generated requirement
to be guaranteed as implemented or valid, as this is the
responsibility of the subsequent verifier agent. Instead, the
prompt provides guidance on the characteristics of well-
formed requirements and conventions for how requirements
should be expressed. Furthermore, the LLM is instructed
to output a rationale explaining the basis for the candidate
requirement, the code files deemed relevant, the names of the
functions that supposedly implement the requirement, and a
confidence value. The user prompt supplies the LLM with a
list of all previously generated requirements, labeled by the
human-in-the-loop (cf. Section III-C). The exact prompts are
provided in the supplementary material1.
As RAG returns only the most relevant results for a given
query, we employsharding(cf. the work of Shao et al. [20]),
which enables systematic retrieval of different regions of the
code base. Each chunk is assigned a random shard from
00–99 as metadata. At each generator invocation, a random

Generation
Sample
random
shards
no
yesFound new requirement?Block
chunks
Support requirement?
Add labelsGenerator
agent
Verifier
agentBatch of
requirements
decides
Human-in-
the-loopTrace link
recovery agent
trace links
rejectrequirement
Shot poolaccept
Label as
rejected
pass shots to generatorRAG vector
store
Smell mutation
agentadds synthetic smells
to requirementsFig. 1. An overview of the complete agentic architecture of our experimental approach.
subset of three candidate chunks is selected, restricting re-
trieval to chunks belonging to that subset. If the generator
cannot identify novel requirements within those chunks, it is
instructed to return no requirement. In this case, the shard
subset is resampled and the generator re-invoked. If no more
requirements can be extracted from a chunk, it is marked as
saturatedand excluded from future subsets. We sample the
shards randomly to allow for a diverse set of requirements to
be generated.
As a control group for evaluation, the generator is addi-
tionally tasked to generate a configurable number of non-
implemented requirements (20 % in our experiments). In this
case, the prompt instructs the LLM to reconstruct requirements
for which the agent does not see clear evidence about their
implementation.
Verifier Agent.This agent is tasked with determining
whether a previously generated requirement candidate is ac-
tually implemented in the code base. Unlike the generator,
the verifier is permitted to search the entire code base with
RAG, as the most relevant retrieval results are expected to
concentrate around the candidate requirement. We prompt the
LLM to classify the retrieved evidence as either (a)support-
ing, (b)contradicting, or (c)insufficientfor the candidate
requirement. To retrieve a diverse and representative set of
evidence chunks, the verifier issues 3 targeted RAG queries: (i)
the top-4 chunks retrieved by the candidate requirement text,
capturing functional intent, (ii) the top-3 chunks retrieved by
the function names identified by the generator, targeting likely
implementation sites, and (iii) the top-3 chunks retrieved by
the file names identified by the generator. We determined this
query constellation to be productive in initial experiments.
Beyond its verdict (supporting, contradicting, or insuffi-
cient), the verifier is also instructed to return a rationale,the relevant files and functions, and a confidence value.
The system prompt additionally includes explicit evaluation
rules to guide the LLM in distinguishing between the verdict
categories. In the user prompt, the verifier is provided with the
candidate requirement, and the files, functions, and rationale
returned by the generator agent.
Mutation Agent.Analogously to the generation of non-
implemented requirements, the proportion of smelly require-
ments in each batch is configurable (20 % in our experiments).
Smelly requirements are generated by selecting requirements
verified as implemented by the verifier agent andmutating
them to synthetically introduce target requirements smells. The
system prompt requests preserving the original requirements
as much as possible, restricting changes strictly to those parts
necessary to introduce the target smell. In the user prompt, we
include the requirement to be mutated, the randomly selected
target smell, chosen from a list of smell categories (cf. Table I),
and a description of the smell category.
The agent employs RAG to retrieve requirements similar
to the candidate from a curated dataset of smelly require-
ments, pre-filtered to include only examples exhibiting the
selected target smell. These examples then serve as few-
shot demonstrations to improve mutation quality. The mutated
requirement is added to the batch as a smelly variant of an
otherwise well-formed requirement.
Trace-Link-Recovery (TLR) Agent.The TLR agent serves
the purpose of supporting the human-in-the-loop in assessing
the implementation status of a requirement, thus producing
verified trace links for all generated requirements. The agent
receives a candidate requirement together with the file and
function names identified by the preceding agents. It then
issues targeted RAG queries to retrieve relevant code chunks:
(i) the top-2 chunks retrieved by the requirement text, (ii) the

top-4 chunks retrieved by function names, and (iii) the top-4
chunks retrieved by file names. Again, this constellation was
determined through earlier experiments.
The agent is prompted to identify the exact lines of code
that implement the given requirement. To support this, the
system prompt includes explicit tracing guidelines specifying
that trace links must be evidence-based. Clear criteria are
established for what constitutes a valid trace link, including
conditions, assignments, and method signatures, as well as
for what must be excluded, such as calls to other methods,
comments, and tests. These guidelines closely follow the
tracing rules by V ogelsang et al. [21].
B. Retrieval-Augmented Generation (RAG)
We employ RAG to enable the agent to efficiently access
the complete code base of the target projects. We preprocess
the source code of each project prior to indexing. First, we
exclude all files irrelevant to the approach, i.e., non-code files
and test files. The remaining files are then split into chunks of
100 lines each, with an overlap of 10 lines at the beginning
and 10 lines at the end of each chunk. This serves the purpose
of balancing retrieval precision against context coverage, as
overly large chunks risk introducing excessive irrelevant code
into the results.
We prefix each chunk with a metadata header containing the
file identifier, the complete path within the project, and the line
range covered by the chunk. This facilitates file-based retrieval,
as the file path can be retrieved directly from the chunk text.
In addition, each chunk is assigned a random shard identifier
(cf. III-A), which is added to the vector store as metadata. For
the TLR agent, each chunk is additionally stored in a second
variant with line numbers prepended to every line of code. This
allows the TLR agent to return exact line numbers reliably.
C. Human-in-the-loop & Multi-shot Prompting
We follow a human-in-the-loop approach for two reasons.
First, because we generate requirements from source code
without pre-existing ground truth, automated verification alone
does not guarantee validity. Second, Unterbusch and V ogel-
sang [22] demonstrate that validated human reasoning in the
RE context, incorporated via few-shot learning, enables LLMs
to adapt to context-specific quality judgments efficiently. This
suggests that human involvement grounds model behavior in
ways that purely automated approaches cannot replicate.
We feed back all requirements labeled by the human-in-
the-loop to the generator agent as few-shot examples. This
feedback loop serves the purpose of LLM adaptation as
described by Unterbusch et al. Additionally, it also prevents the
generation of duplicate requirements by making the generator
aware of all previously accepted (or rejected) requirements.
IV. STUDYDESIGN
To report the study design and study results, we follow
the guidelines by Jedlitschka and Pfahl [23] on how to report
experiments in software engineering.A. Goals & Research Questions
The goal of our human-in-the-loop study is to explore
whether the generation of realistic synthetic requirements from
source code is viable. We assess this in three areas. First,
we investigate whether the requirements generated by our
approach are actually implemented in the code base and not
hallucinated. Human-verified trace links act as ground truth.
Second, we evaluate the quality of the generated requirements
by examining whether they are free of requirements smells.
Third, we investigate the viability of synthetically introducing
requirements smells to produce realistic datasets. This results
in the following research questions:
•RQ1.1:To what extent are requirements reconstructed
from source code by an LLM actually implemented in
the code base?(Implementation Accuracy)
•RQ1.2:To what extent are the reconstructed requirements
hallucinated by an LLM?(Hallucination Rate)
•RQ2:To what extent are requirements generated by an
LLM free of requirements smells?(Generation Quality)
•RQ3:To what extent do human evaluators agree with
the LLM on the category of synthetically introduced
requirements smells?(Human-LLM Reliability)
B. Datasets & Participants
We use two software projects as code bases to generate
synthetic requirements. The first project, hereafter referred
to asSEP, is a student project developed by bachelor’s
students during a practical software engineering course at our
university. The source code comprises approximately 24,000
lines of code written in Java, TypeScript, HTML, and CSS,
and implements a full-stack web application resembling a
personal driver hiring platform (like UBER). The second
project,Mattermost, is the publicly available source code3of
the desktop client of the Mattermost messaging application. It
has approximately 28,000 lines of code written in TypeScript
and HTML. The two projects were selected to cover a diverse
range of code characteristics, ranging from a well-maintained
long-term open-source project to a student project developed
under less rigorous review practices within a constrained
timeframe.
All human evaluators involved are PhD students with expe-
rience in Requirements Engineering.
C. Experimental Setup
To conduct our experiments, we implemented the agentic
approach described in Section III in Python. To make the
approach accessible to evaluators, we developed a web appli-
cation usingSvelteKitthat interfaces with the Python backend.
LLM requests were managed usingCelerytasks, enabling
multiple agent invocations to be processed simultaneously.
During the study, evaluators accessed the locally deployed
web application to complete their evaluations, with all results
stored in a database for subsequent export and analysis.
For retrieval-augmented generation, we usedChromaDBas
3https://github.com/mattermost/desktop (commitadfcd2conmain).

TABLE I
SMELLCATEGORIESUSEDTHROUGHOUT THESTUDY. EXTRACTEDFROMPREVIOUSWORK BYVOGELSANG ET AL. [21]ANDEXTENDED FORTHIS
PAPERBASED ONWORK BYFRATTINI ET AL. [24].
Smell Category Description
Subjective Language Words of which the semantics are not objectively defined, such as user friendly, easy to use, cost effective, etc. OR Sentences
expressing personal opinions or feelings.
Ambiguities Sentences or sentence parts that are unclear/imprecise and can be misunderstood if read by different people.
Loopholes Sentences containing escape clauses or conditional exceptions that allow the requirement to be circumvented or interpreted as not
applicable, e.g., using phrases like ’where possible’, ’as appropriate’, ’unless otherwise agreed’, etc.
Open-ended,
non-verifiable termsTerms or phrases that cannot be objectively measured or tested, such as ’sufficient’, ’adequate’, ’reasonable’, ’as needed’, etc.
Superlatives Over strong guarantees using absolute words such as always, never, guarantee, 100 %, cannot fail, etc.
Comparatives Comparative words such as better, more, etc.
Negative Statements Sentences containing negative modifiers (e.g., not), negative expressions.
Vague Pronouns Pronouns that refer back to a previous part of the text for which the reference is unclear.
Non-atomic Requirements that combine multiple distinct conditions, behaviors, or features into a single statement, making them difficult to test
or trace individually.
Passive V oice Sentences using passive voice such that it is unclear who is performing a certain action.
Optional Parts Sentences containing optional parts, e.g., by using the words possibly, eventually, if possible, if needed, etc.
Weak Verbs Weak verbs, such as can, could, may, etc.
UI/UX Aesthetics Requirements to UI/UX without measurable criteria, such as modern look, pleasant UI, nice animations, etc.
Scope Creep Making too broad assumptions, such as support all languages, work on all devices, integrate with any service, etc.
a local vector store, with embeddings produced by OpenAI’s
text-embedding-3-largemodel.
We selected OpenAI’sgpt-5.4-mini-2026-03-17as
the model for all agentic tasks, as it demonstrated suitable
performance across all tasks while offering a favorable cost-
performance tradeoff compared to the fullgpt-5.4model.
We did not evaluate any additional models.
For each of the two projects, three independent evaluators
evaluated the generated requirements, serving as the human-
in-the-loop for one round of generation. The evaluators were
asked to assess requirements for one hour with respect to the
following three criteria:
1)Appropriateness.A subjective 5-point Likert scale score
assessing two aspects: (a) whether the requirement con-
forms to the expected format, and (b) whether it describes
plausible system behavior that could be expected from the
given system.
2)Requirements Smells.All requirements smells identified
by the evaluator in the given requirement, selected from
a predefined list of 14 smell types (see Table I).
3)Implementation Status.Based on the traceability informa-
tion produced by the TLR agent, highlighting the most
relevant code locations related to the requirement’s im-
plementation, evaluators classify the requirement asfully
implemented,partially implemented, ornot implemented.
While the initial design aimed for a single evaluation round
(only by the human-in-the-loop), we planned additional rounds
as a contingency to resolve potential inter-rater disagreements.
In a second round, the evaluators were rotated across require-
ments sets and asked to independently re-rate the appropriate-
ness and smell labels to assess inter-rater agreement. During
the rating, they did not have access to the first evaluator’s
decisions. In a third round, two authors of this paper reviewed
all inter-rater disagreements in the smell labels and resolved
them by developing a codebook addressing common sourcesof confusion that arose during evaluation. Although the spe-
cific conventions were derived from disagreements observed
in practice rather than specified in advance, the following
codebook was established to ensure consistent labeling across
evaluators.
•The use ofshouldalone does not constitute theweak verb
smell;
•thevague pronounsmell applies only when multiple
possible references exist for a given pronoun;
•when bothUI/UX aestheticsandsubjective language
are applicable, the more specificUI/UX aestheticssmell
should be used;
•the smellsopen-ended, non-verifiable termsandsubjec-
tive languageare consolidated intosubjective language;
•thepassive voicesmell applies only when the use of
passive voice results in an unclear or unidentifiable actor.
V. STUDYRESULTS
A. Human-in-the-loop Datasets
The human-in-the-loop study produced a total of 188 re-
quirements across the two source code projects. The generated
requirements are labeled either as (i) implemented, i.e., imple-
mented and smell-free by the LLM, (ii) non-implemented, i.e.,
not-implemented and smell-free, or (iii) smelly, i.e., including
a synthetically added requirement smell. We introduce the
non-implemented class as a control group. These categories
are solely based on the LLM’s perspective. We show the
distribution of implemented, non-implemented, and smelly
requirements in Table II.
B. RQ1.1: Implementation Accuracy
This research question aims to assess the implementation
accuracy of the LLM-generated requirements by examining
the extent to which requirements generated from source code
are confirmed as implemented by the human-in-the-loop eval-
uators, based on the trace link code excerpts presented to

TABLE II
REQUIREMENTSGENERATEDDURING THEHUMAN-IN-THE-LOOPSTUDY
Project Evaluator# of Requirements
Impl. Non-impl. SmellyTotal
SEPE1 18 7 530
E2 19 8 734
E3 21 7 634
MattermostE1 16 5 526
E2 21 6 532
E3 19 8 532
TABLE III
RQ1.1: IMPLEMENTATIONACCURACY OF THELLM BASED ON
HUMAN-IN-THE-LOOPLABELING
Project EvaluatorResults
Precision RecallF 1-score
SEPE1 0.944 0.708 0.810
E2 0.895 0.739 0.810
E3 0.857 0.720 0.783
E1–E3 0.897 0.722 0.800
MattermostE1 1.000 0.762 0.865
E2 0.952 0.769 0.851
E3 1.000 0.731 0.844
E1–E3 0.982 0.753 0.853
All 0.939 0.738 0.826
them during the study. We report precision, recall, andF 1-
scores of comparing the LLM’s generation methods (i.e.,
whether it was tasked to generate an implemented or non-
implemented requirement) with the labels of the human-in-
the-loop evaluators in Table III. To compute these values, we
define true positives as labeled implemented by both the LLM
and human evaluator, false positives as labeled implemented
by the LLM but labeled non-implemented by the human, etc.
As can be seen, there is a consistent gap between precision
and recall across all evaluators and projects. While the preci-
sion is consistently high (0.857–1.000), the recall stays lower
(0.708–0.769). When the LLM is tasked to generate an imple-
mented requirement, humans almost always agree. Still, when
the LLM is tasked to generate a non-implemented requirement
deliberately, humans often find it to be implemented based
on the given trace links. Furthermore, the performance for
Mattermost is consistently stronger than for SEP. A possible
explanation for this may be the higher code quality of a well-
maintained long-term project.
C. RQ1.2: Hallucination Rate
In Table IV, we report the false positive requirements
(i.e., labeled as implemented by the LLM and labeled as
non-implemented by the human-in-the-loop), as well as the
hallucination rate, which is defined as the false positive rate.
The overall hallucination rate of 6.1 % is low, which
suggests that the LLM reliably generates requirements that
are actually implemented. Still, there is a large difference
between the two projects. For Mattermost, we see a rate of
only 1.8 %, while for SEP, the rate is 10.3 %. This discrepancyTABLE IV
RQ1.2: REQUIREMENTSHALLUCINATED BY THELLMAND
HALLUCINATIONRATE ASASSESSED BY THEHUMAN-IN-THE-LOOP
USINGTRACELINKS
ProjectResults
LLM: Impl. FP (Hallucinated) Hallucination Rate
SEP 58 6 10.3 %
Mattermost 56 1 1.8 %
Total 114 7 6.1 %
follows the observation from RQ1.1. It is possible, that the
code quality is relevant for this difference. Well-structured
code that has been developed over a long time-span with many
different reviewers, eases the LLM to ground the generated
requirements in clear evidence. However, as Mattermost is
also more complex than the student projects, it might have
been easier for the evaluators to identify hallucinations as such.
Though, as there are only 7 reported hallucinations in total,
generalizability is limited.
D. RQ2: Generation Quality
Comparing the smell labels of the first-round human-in-the-
loop with the second-round evaluators, we find that Cohen’sκ
ranges from−0.083(Mattermost; E1) to0.803(Mattermost;
E3), with an overallκ= 0.213across both projects and
all evaluator pairs, indicating fair agreement at best. This
disagreement shows the necessity of the third labeling round,
in which two authors of this paper resolved disagreements by
developing a codebook from common sources of confusion
(cf. Section IV-C). The peer-reviewed labels from this third
round are used as the basis for the results reported below. We
report all results in Table V.
The smell rate in requirements labeled as smell-free (im-
plemented and non-implemented combined) by the LLM, over
both projects and all evaluators, is 24.5 %. This smell rate is
calculated based on the peer-reviewed smells. As observed for
RQ1.1 and RQ1.2, both projects exhibit notable differences in
variation. The smell rates for SEP range from 7.1 %–40.7 %,
showing considerable variance across the 3 evaluators. For
Mattermost, the results are much more consistent with a range
of 25.9 %–33.3 %.
Of all possible smell categories, only 6 categories were
labeled to appear in the allegedly smell-free requirements.
These categories, in descending order of frequency, are:non-
atomic(12),passive voice(11),ambiguities(8),negative
statements(6),subjective language(3), andvague pronouns
(1). Note, that a single requirement may contain multiple
smells. Many of these dominant smells are structural and
syntactic, rather than semantic ones, suggesting that the LLM
tends to generate requirements that are functionally correct but
linguistically imprecise. The absence of smell categories such
assuperlativesorcomparativestogether with the functional
correctness suggests that the LLM does well in presenting the
code behavior based on the given information.
A representative example for a non-atomic requirement as
generated by the LLM is (Mattermost):”When a download

TABLE V
RQ2: REQUIREMENTSSMELLSFOUND INLLM-GENERATED
SMELL-FREEREQUIREMENTSAFTERPEER-REVIEW
Project EvaluatorResults
# of Reqs. # of Smelly Reqs. Smell Rate
SEPE1 25 3 12.0 %
E2 27 11 40.7 %
E3 28 2 7.1 %
E1–E3 80 16 20.0 %
MattermostE1 21 7 33.3 %
E2 27 8 29.6 %
E3 27 7 25.9 %
E1–E3 75 22 29.3 %
All 155 38 24.5 %
item no longer has a valid file location, opening its folder
should mark it as deleted and, if a download location is config-
ured, open that download folder instead. ”This highlights the
LLM trying to include closely related functionality (in code)
in a single requirement, leading to long, complicated, and/or
non-atomic requirements. Another representative requirement,
this time for passive voice, is:”When the main window
is closed on Windows or Linux and tray minimization is
enabled, the application hides the window instead of quitting. ”
It is unclear which actor is responsible for closing the main
window. Ambiguous requirements would often allow various
interpretations of the intended functionality, e.g., in:”If a
notification sound is configured, the notification is shown
silently and the chosen sound is stored for later playback”it
is unclear whether the mentioned silent notification is shown
directly after configuring a notification sound, or whether this
configures a general system property.
E. RQ3: Human-LLM Reliability
To assess whether synthetically introduced smells are re-
liably detected by human evaluators, we compare the LLM’s
intended smells against the peer-reviewed smells from the final
evaluation round. In both projects, 60.6 % of intentionally
smelly requirements were labeled with the exact intended
smell category, and SEP showed a higher exact-match agree-
ment (66.7 %) than Mattermost (53.3 %). When considering
whether any smell was detected, regardless of category, 90.9 %
of intentionally smelly requirements received at least one smell
label. We report all results in Table VI.
Figure 2 shows the per-category agreement: while some
smells, e.g.,subjective languageandUI/UX aesthetics, are
detected with perfect agreement, other smell categories such as
passive voiceandloopholesshow zero exact-match agreement.
F . Appropriateness
Lastly, we evaluated the appropriateness scores. We did
not include them as a research question, as they are a
highly subjective measure, which is reflected in the inter-rater
agreement between the first and second evaluation rounds.
Over both projects, Krippendorff’sα= 0.178, with an exactTABLE VI
RQ3: REQUIREMENTSSMELLSFOUND INLLM-GENERATEDSMELLY
REQUIREMENTSAFTERPEER-REVIEW
Project Evaluator# of Smelly
Reqs.Any Smell? Exact Smells?
# % # %
SEPE1 5 4 80.0 % 4 80.0 %
E2 7 7 100.0 % 5 71.4 %
E3 6 5 83.3 % 3 50.0 %
E1–E3 18 16 88.9 % 12 66.7 %
MattermostE1 5 5 100.0 % 2 40.0 %
E2 5 5 100.0 % 3 60.0 %
E3 5 4 80.0 % 3 60.0 %
E1–E3 15 14 93.3 % 8 53.3 %
All 33 30 90.9 % 20 60.6 %
0% 25% 50% 75% 100%
Exact Smell Detection Rate (TPR)loopholespassive voicecomparativesnegative statementsvague pronounsnon-atomicoptional partssuperlativesambiguitiesscope creepsubjective languageui/ux aestheticsweak verbs
0/1  (0%)0/3  (0%)1/3  (33%)1/2  (50%)2/4  (50%)3/5  (60%)2/3  (67%)2/3  (67%)1/1  (100%)1/1  (100%)4/4  (100%)2/2  (100%)1/1  (100%)
50%
Fig. 2. Agreement of human evaluators (after peer-review) with LLM on the
categories of synthetically introduced smells.
agreement of 36.2 % (70.2 % when allowing a one-point de-
viation between ratings), confirms that appropriateness scores
are inconsistent across evaluators and should be interpreted
accordingly.
Comparing appropriateness scores across generation meth-
ods, the Kruskal-Wallis test reveals a significant overall effect
for Mattermost (p= 0.0001;ε2= 0.212) and across both
projects combined (p= 0.0034;ε2= 0.061), but not for
SEP alone (p= 0.0981;ε2= 0.048). Pairwise comparisons
using the Mann-Whitney U test with Holm–Bonferroni cor-
rection show that for Mattermost, positive requirements score
significantly higher than both not-implemented (p adj= 0.0043,
p <0.01;r=−0.447) and smelly requirements (p adj=
0.0005,p <0.001;r=−0.598). For SEP, we do not
detect significant differences between generation methods after
Holm–Bonferroni correction. Across both projects combined,
only the contrast between positive and smelly requirements
is calculated as significant (p adj= 0.002,p <0.01;r=
−0.367), while non-implemented requirements do not differ
significantly from either group. These results suggest that

human evaluators likely perceive smelly requirements as less
appropriate than well-formed ones.
Figure 3 shows the mean appropriateness scores per batch
between evaluators for both projects. Although the trend line
suggests a slight upward trajectory over successive batches, in-
dicating that the multi-shot feedback loop may have an impact,
for both projects, this correlation is not statistically significant
(Spearman’s rank correlation:ρ= 0.40;p= 0.600), and
the small number of batches (n= 4) limits any meaningful
interpretation of this trend. We note it as a tentative observation
that warrants further investigation with larger batch counts.
VI. DISCUSSION
We initially intended to provide a complete approach for
generating realistic requirements-to-code datasets. Throughout
the study and evaluation, various problems arose, leading to
this experience report. While the hallucination rate is low,
confirming our initial hypothesis that, through code-grounding,
LLMs should be able to generate implemented requirements,
this does not hold true for non-implemented ones. Low recall
comparing the human judgments to the LLM’s task shows that,
even when specifically prompted to generate non-implemented
requirements, the LLM cannot reliably perform this task. We
agree with previous work [13], [19] that LLMs show promis-
ing results in requirements generation from code. However,
similarly to what these studies already hint at, we find that
human oversight is still necessary.
We hypothesized that through clear guidelines provided in
the prompt, the LLM would be able to generate high-quality
requirements if desired, thus containing a low rate of uninten-
tional requirements smells. Seeing that 24.5 % of requirements
are still labeled as smelly by the human evaluators also
proves this hypothesis wrong. Femmer and V ogelsang [25]
argue that requirements quality is fundamentally quality-in-
use: a requirement is only defective if it actually impairs
a downstream activity, and whether a given smell does so
depends heavily on context. Frattini [26] reinforces this with
industrial evidence, showing that the factors that determine
whether a smell constitutes an actual defect are numerous and
context-specific. Consequently, the 24.5 % smell rate reported
in RQ2 should be interpreted as the rate at which generated
requirements exhibit linguistic patterns associated withpoten-
tialquality issues, rather than confirmed defects. Preliminary
evidence that smells affect at least some downstream tasks is
provided by V ogelsang et al. [21], who show that requirements
smells negatively impact automated traceability, suggesting
that this concern is not merely theoretical.
When designing the original study, we assumed that one
human-in-the-loop would be sufficient to label requirements
smells with adequate reliability. However, the weak inter-rater
agreement ofκ= 0.213made an additional round of evalu-
ation and the development of a codebook necessary, defining
conventions to operationalize context-specific interpretations
of smells. This is supported by the findings of Unterbusch
and V ogelsang [22], showing that smell judgments are context-
dependent and not a task that can be objectively completedeven by trained evaluators. We acknowledge that this also
possibly affects the actual smell rate for RQ2, regardless of
the countermeasures taken (cf. Section VI-B).
Furthermore, we assumed that synthetically introduced
smells could be reliably labeled, enabling the production of
realistic datasets. We see that human evaluators identified a
smelly requirements in 90.9 % of cases, showing that these
requirements are recognized as problematic. However, since
only 60.6 % of smells were exactly identified, the exact smell
perceived by humans again appears to be context-dependent
and not objectively identifiable.
For practitioners, our findings suggest that while a low
hallucination rate indicates that the approach is useful for
generating requirements from existing codebases, there are
other issues that are not immediately obvious. Generated
requirements should not be used without human review, as the
LLM fails to self-assess quality and to follow the prompted
tasks precisely. Critically, the performance gap between Mat-
termost and SEP indicates that the approach performs better
for well-maintained codebases, precisely those most in need
of requirements generation in practice. Further investigation
of the unintentional smells introduced by LLMs could inform
prompt engineering improvements, highlighting the defects to
which LLMs are most vulnerable.
A. Lessons Learned
From the study and its findings, we learn the following
lessons.
1) Labeling smells in requirements always requires multiple
humans, as smells are highly context-sensitive. Neither
an LLM nor a single human evaluator alone is reliable
for this task.
2) LLMs can support the generation of requirements from
code, but should not be trusted with this task without
human supervision as (a) prompts are not always fol-
lowed precisely (e.g., generating implemented require-
ments even when explicitly prompted not to do so) and
(b) generated requirements can exhibit major quality
deficiencies.
3) The reliability of LLM-based requirement generation,
even when based on source code, strongly depends on
the codebase; practitioners should not expect consistent
results across codebases of different maintenance levels.
B. Threats to Validity
Internal Validity.Prompting must be considered a threat to
validity whenever LLMs are employed (cf. Korn et al. [27]),
as prompts may miss important details or bias the LLM toward
outcomes anticipated by the prompt author. We mitigated
this by iteratively refining the prompts throughout the study.
The multi-shot feedback loop introduces a further threat,
as each evaluator’s later ratings are shaped by their own
prior decisions, causing individual biases to accumulate, thus
complicating cross-evaluator comparison. Additionally, since
smelly requirements are mutations of previously accepted

1 2 3 4
Batch number (per evaluator)12345Appropriateness score (1 5)
Mattermost
E1
E2
E3
Mean
Trend
1 2 3 4
Batch number (per evaluator)
Sep
E1
E2
E3
Mean
TrendFig. 3. Comparison of human-in-the-loop appropriateness ratings over time. Each batch is generated with reviews from last batch as shot samples.
requirements, evaluators may recognize them based on fa-
miliarity with the unmodified counterpart. Presenting smell
categories as a predefined list may further encourage evalu-
ators to label smells they would not otherwise have noticed,
potentially inflating false positive rates. The evaluator pool
consisted exclusively of PhD students with varying levels of
RE expertise, which may introduce selection bias and noise,
particularly for smell detection.
External Validity.The approach is evaluated exclusively
on general-purpose software projects, excluding domains with
domain-specific requirements such as safety-critical or heavily
regulated systems. We consider this a limited threat, as the
approach relies solely on information retrieved via RAG from
the target code base without incorporating domain-specific
knowledge. Furthermore, the fixed RAG configuration may
require reconfiguration for projects that differ substantially
in scale from those used in this study. However, as we only
used 2 very different software projects, generalizability of the
assumed explanations between Mattermost and SEP is very
limited.
Construct Validity.Appropriateness ratings are inherently
subjective as evaluators may interpret the 5-point scale dif-
ferently, leading to inconsistent ratings. The smell labels in
round three are not purely independent human judgments, as
the authors had to resolve disagreements and produce the code-
book; the labels therefore may partially reflect the authors’
interpretation. Implementation status is assessed solely based
on TLR agent trace links, meaning that retrieval failures may
produce mislabeled requirements rather than reflecting true
implementation status. We consider this an acceptable tradeoff.
Finally, results are based on a single LLM; alternative models
may yield different outcomes, although the exact model is
reported in full to enable reproduction and comparison.
Conclusion Validity.The datasets produced during the
study are relatively small, which can reduce statistical power
for subgroup analysis. Although the Holm-Bonferroni correc-
tion is applied to each family of comparisons, the overall
family-wise error rate across all analyzes may not be fully
controllable.VII. CONCLUSION
We present an LLM-based and RAG-supported agentic
approach for generating realistic requirements-to-code datasets
from existing code bases, and evaluate it in a human-in-the-
loop study across two software projects. Although we origi-
nally intended to offer this approach as a viable solution for
generating datasets for downstream RE research, we now offer
an experience report, as during the study various problems
arose. Although we initially hypothesized that LLMs would be
able to reliably generate both implemented and intentionally
non-implemented requirements, only the former held true
(6.1 % hallucination rate). Furthermore, we hypothesized that
through prompting with rigorous guidelines, the LLM would
be able to generate requirements free of requirements smells
(thus potentially high in quality). In practice, 24.5 % of the
generated requirements exhibit requirements smells, showing
that quality cannot be self-assessed by the LLM alone. A
single human-in-the-loop also proved to be insufficient in
detecting smells reliably, confirming that requirements smells
are context-dependent and cannot be classified reliably without
shared conventions, as reflected in a low initial inter-rater
agreement ofκ= 0.213. This required multiple rounds of
evaluations that were aimed at resolving common misunder-
standings. Finally, the reliability of LLM-based requirements
generation strongly depends on code quality, such that practi-
tioners should not expect consistent results across code bases
of different maintenance levels.
These findings suggest that fully autonomous generation
of requirements from code is not yet viable, requiring hu-
man supervisor at least for quality assurance. We offer our
lessons learned from this study and publish two generated
requirements-to-code datasets as a concrete contribution to the
empirical RE community.
A. Future Work
Several directions remain for future work. Automated smell
detection could be applied as an initial optimization step
before human review, reducing cognitive workload. Rather
than asking evaluators to identify all smells from scratch,

the human-in-the-loop could confirm or reject automatically
detected smell candidates, shifting the task from open-ended
labeling to verification. The multi-shot feedback loop shows
a tentative upward trend in appropriateness scores across
batches; a larger evaluation with more batches and evaluators
would allow this effect to be examined more rigorously, and
may inform whether the feedback loop can reduce the need
for human oversight. Finally, a larger evaluation across more
code bases and domains, particularly safety-critical or heavily
regulated systems, is necessary to establish the generalizability
of these findings.
ACKNOWLEDGMENTS
We thank the study participants for their time and support
throughout the evaluation of the presented approach. Large
language models were used to assist in improving the writing
of this paper and to support the development of the experi-
mental implementation of the approach.
Funded by the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) – Project number: 566352773.
REFERENCES
[1] A.-L. Lamprecht, L. Garcia, M. Kuzak, C. Martinez, R. Arcila, E. Martin
Del Pico, V . Dominguez Del Angel, S. van de Sandt, J. Ison, P. A.
Martinez, P. McQuilton, A. Valencia, J. Harrow, F. Psomopoulos, J. L.
Gelpi, N. Chue Hong, C. Goble, and S. Capella-Gutierrez, “Towards
FAIR principles for research software,”Data Science, vol. 3, no. 1, pp.
37–59, 2019.
[2] D. Yang, X. Xie, X. Yang, M. Hu, Y . Huang, Y . Zhang, W. Miao,
T. Su, C. Wan, and G. Pu, “Assessing the impact of requirement
ambiguity on LLM-based function-level code generation,”arXiv preprint
arXiv:2604.21505, 2026.
[3] J. L. C. Guo, J.-P. Steghöfer, A. V ogelsang, and J. Cleland-Huang,
Natural Language Processing for Requirements Traceability. Springer
Nature Switzerland, 2025, pp. 89–116.
[4] Q. Motger, C. Catot, and X. Franch, “Characterizing datasets for
LLM-based requirements engineering: A systematic mapping study,”
inInternational Conference on Evaluation and Assessment in Software
Engineering (EASE). ACM, 2026.
[5] J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim, “A survey on large
language models for code generation,”ACM Transactions on Software
Engineering and Methodology (TOSEM), vol. 35, no. 2, pp. 1–72, 2026.
[6] A. El-Hajjami and C. Salinesi, “Synthline: A product line approach
for synthetic requirements engineering data generation using large
language models,” inInternational Conference on Research Challenges
in Information Science. Springer, 2025, pp. 208–225.
[7] R. Xu, Z. Xu, G. Li, and V . S. Sheng, “Bridging the gap between
source code and requirements using GPT (student abstract),” inAAAI
Conference on Artificial Intelligence, vol. 38, no. 21, 2024, pp. 23 686–
23 687.
[8] K. R. Dearstyne, “Intelligent traceability to support software main-
tainability and accountability,” inIEEE International Requirements
Engineering Conference (RE). IEEE, 2025, pp. 607–611.
[9] Y . Wang, J. Keung, X. Ma, Z. Mao, K. Chen, and Y . Li, “R2Code:
A self-reflective LLM framework for requirements-to-code traceability,”
arXiv preprint arXiv:2604.22432, 2026.
[10] H. Femmer, D. Méndez Fernández, S. Wagner, and S. Eder, “Rapid
quality assurance with requirements smells,”Journal of Systems and
Software (JSS), vol. 123, pp. 190–213, 2017.
[11] E. Gentili and D. Falessi, “Characterizing requirements smells,” inInter-
national Conference on Product-Focused Software Process Improvement
(PROFES). Springer, 2023, pp. 387–398.
[12] W. Zogaan, P. Sharma, M. Mirahkorli, and V . Arnaoudova, “Datasets
from fifteen years of automated requirements traceability research:
Current state, characteristics, and quality,” inIEEE International Re-
quirements Engineering Conference (RE). IEEE, 2017, pp. 110–121.[13] D. Jin, Z. Jin, Y . Zhang, Z. Fang, L. Li, Y . He, X. Chen, and W. Sun,
“UserTrace: User-level requirements generation and traceability recovery
from software project repositories,” 2025.
[14] D. Fuchß, T. Hey, J. Keim, H. Liu, N. Ewald, T. Thirolf, and A. Koziolek,
“LiSSA: Toward generic traceability link recovery through retrieval-
augmented generation,” inIEEE/ACM International Conference on
Software Engineering (ICSE). IEEE, 2025, pp. 1396–1408.
[15] K. Moran, D. N. Palacio, C. Bernal-Cárdenas, D. McCrystal, D. Poshy-
vanyk, C. Shenefiel, and J. Johnson, “Improving the effectiveness of
traceability link recovery using hierarchical bayesian networks,” in
ACM/IEEE International Conference on Software Engineering (ICSE).
ACM, 2020, pp. 873–885.
[16] S. J. Ali, V . Naganathan, and D. Bork, “Establishing traceability between
natural language requirements and software artifacts by combining RAG
and LLMs,” inInternational Conference on Conceptual Modeling (ER).
Springer, 2024, pp. 295–314.
[17] T. Hey, F. Chen, S. Weigelt, and W. F. Tichy, “Improving traceability
link recovery using fine-grained requirements-to-code relations,” in
IEEE International Conference on Software Maintenance and Evolution
(ICSME). IEEE, 2021, pp. 12–22.
[18] J. Lin, Y . Liu, Q. Zeng, M. Jiang, and J. Cleland-Huang, “Traceability
transformed: Generating more accurate links with pre-trained BERT
models,” inIEEE/ACM International Conference on Software Engineer-
ing (ICSE). IEEE, 2021, pp. 324–335.
[19] E. Persson, E. Alégroth, and T. Gorschek, “Code2Req: Using
generative AI to generate requirements from source code,” 2025.
[Online]. Available: http://dx.doi.org/10.2139/ssrn.5845431
[20] R. Shao, J. He, A. Asai, W. Shi, T. Dettmers, S. Min, L. Zettlemoyer,
and P. W. Koh, “Scaling retrieval-based language models with a trillion-
token datastore,”Advances in Neural Information Processing Systems,
vol. 37, pp. 91 260–91 299, 2024.
[21] A. V ogelsang, A. Korn, G. Broccia, A. Ferrari, J. Fischbach, and
C. Arora, “On the impact of requirements smells in prompts: The case
of automated traceability,” inIEEE/ACM International Conference on
Software Engineering: New Ideas and Emerging Results (ICSE-NIER),
2025, pp. 51–55.
[22] M. Unterbusch and A. V ogelsang, “Context-adaptive requirements defect
prediction through human-LLM collaboration,” inIEEE/ACM Interna-
tional Conference on Software Engineering: New Ideas and Emerging
Results (ICSE-NIER), 2026.
[23] A. Jedlitschka and D. Pfahl, “Reporting guidelines for controlled ex-
periments in software engineering,” inInternational Symposium on
Empirical Software Engineering (ESEM), 2005.
[24] Y . Li, J. Keung, X. Ma, C. Y . Chong, J. Zhang, and Y . Liao, “LLM-
based class diagram derivation from user stories with chain-of-thought
promptings,” inIEEE 48th Annual Computers, Software, and Applica-
tions Conference (COMPSAC). IEEE, 2024, pp. 45–50.
[25] H. Femmer and A. V ogelsang, “Requirements Quality Is Quality in Use,”
IEEE Software, vol. 36, no. 3, pp. 83–91, May 2019.
[26] J. Frattini, “Identifying Relevant Factors of Requirements Quality: An
Industrial Case Study,” inRequirements Engineering: Foundation for
Software Quality, D. Mendez and A. Moreira, Eds. Cham: Springer
Nature Switzerland, 2024, pp. 20–36.
[27] A. Korn, L. Zaruchas, C. Arora, A. Metzger, S. Smolka, F. Wang,
and A. V ogelsang, “Reporting LLM prompting in automated software
engineering: A guideline based on current practices and expectations,” in
ACM International Conference on AI Foundation Models and Software
Engineering (FORGE), 2026.