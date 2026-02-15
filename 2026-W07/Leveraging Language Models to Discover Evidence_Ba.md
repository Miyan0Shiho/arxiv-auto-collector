# Leveraging Language Models to Discover Evidence-Based Actions for OSS Sustainability

**Authors**: Nafiz Imtiaz Khan, Vladimir Filkov

**Published**: 2026-02-12 09:18:38

**PDF URL**: [https://arxiv.org/pdf/2602.11746v1](https://arxiv.org/pdf/2602.11746v1)

## Abstract
When successful, Open Source Software (OSS) projects create enormous value, but most never reach a sustainable state. Recent work has produced accurate models that forecast OSS sustainability, yet these models rarely tell maintainers what to do: their features are often high-level socio-technical signals that are not directly actionable. Decades of empirical software engineering research have accumulated a large but underused body of evidence on concrete practices that improve project health.
  We close this gap by using LLMs as evidence miners over the SE literature. We design a RAG-pipeline and a two-layer prompting strategy that extract researched actionables (ReACTs): concise, evidence-linked recommendations mapping to specific OSS practices. In the first layer, we systematically explore open LLMs and prompting techniques, selecting the best-performing combination to derive candidate ReACTs from 829 ICSE and FSE papers. In the second layer, we apply follow-up prompting to filter hallucinations, extract impact and evidence, and assess soundness and precision.
  Our pipeline yields 1,922 ReACTs, of which 1,312 pass strict quality criteria and are organized into practice-oriented categories connectable to project signals from tools like APEX. The result is a reproducible, scalable approach turning scattered research findings into structured, evidence-based actions guiding OSS projects toward sustainability.

## Full Text


<!-- PDF content starts -->

Leveraging Language Models to Discover Evidence-Based Actions
for OSS Sustainability
Nafiz Imtiaz Khan∗, Vladimir Filkov
Department of Computer Science, University of California, Davis, CA 95616, USA
Abstract
When successful, Open Source Software (OSS)
projects create enormous value, but most never reach
a sustainable state. Recent work has produced ac-
curate models that forecast OSS sustainability, yet
these models rarely tell maintainers what to do: their
features are often high-level socio-technical signals
that are not directly actionable. Decades of empir-
ical software engineering research have accumulated
a large but underused body of evidence on concrete
practices that improve project health.
We close this gap by using LLMs as evidence min-
ers over the SE literature. We design a RAG-pipeline
and a two-layer prompting strategy that extract re-
searched actionables (ReACTs): concise, evidence-
linked recommendations mapping to specific OSS
practices. In the first layer, we systematically explore
open LLMs and prompting techniques, selecting the
best-performing combination to derive candidate Re-
ACTs from 829 ICSE and FSE papers. In the second
layer, we apply follow-up prompting to filter hallu-
cinations, extract impact and evidence, and assess
soundness and precision.
Our pipeline yields 1,922 ReACTs, of which 1,312
pass strict quality criteria and are organized into
practice-oriented categories connectable to project
signals from tools like APEX. The result is a repro-
ducible, scalable approach turning scattered research
findings into structured, evidence-based actions guid-
ing OSS projects toward sustainability.
Keywords:Research Actionable, Open Source
Software, Sustainability, Literature Review
1 Introduction
Open Source Software (OSS) underpins a multi-
billion-dollar global ecosystem and is a foundational
component of modern software systems. Most large
technologycompaniesrelyextensivelyonOSSintheir
productioninfrastructure. ArecentHarvardBusiness
∗Corresponding author:nikhan@ucdavis.eduSchool report estimates that recreating the OSS com-
ponents currently embedded in commercial software
would cost approximately $8.8 trillion [1]. This fig-
ure represents demand-side value, while supply-side
estimates suggest that OSS development costs are on
the order of $4.2 billion, despite OSS being used by
over 96% of commercial software products. Comple-
menting this view, another industry report projects
that the global OSS market will reach $50 billion by
2026 [2].
Despite its economic and societal importance, OSS
facesapersistentsustainabilitychallenge. Priorstud-
ies report that over 90% of OSS projects are eventu-
ally discontinued, with failure rates particularly high
among smaller and younger projects [3, 4]. This chal-
lenge motivates two long-standing and interrelated
research problems: (a)predictingwhether an OSS
project is likely to remain sustainable, and (b)iden-
tifying concrete actionsthat project maintainers can
take to intervene when sustainability risks emerge.
The Prediction-Action Gap.A substantial
body of prior work has focused on the first prob-
lem, predicting OSS sustainability, using statistical
and machine learning models over socio-technical fea-
tures. Ghapanchi et al. [5] analyzed longitudinal
data from 1,409 OSS projects and proposed predic-
tive models based on factors such as developer activ-
ity, release frequency, project age, and collaboration
patterns. Yin et al. [6] introduced an LSTM-based
deep learning model that achieved over 93% accu-
racy using 18 socio-technical features. More recently,
Xiao et al. [7] employed 64 early-stage variables to
forecast OSS sustainability during project inception.
While these approaches demonstrate strong pre-
dictive performance, they provide limited guidance
on the second problem:howpractitioners should re-
spond to predicted risks. The features that drive
these predictions, such as call-graph complexity, ag-
gregate code churn, or developer network centrality,
are often abstract or indirectly measurable signals of
project health. They tell maintainerswhatmight be
going wrong, but notwhat to doabout it. As a re-
sult, there remains a critical gap between sustainabil-
1arXiv:2602.11746v1  [cs.SE]  12 Feb 2026

ity prediction and actionable intervention: even when
a model forecasts that a project is at risk, maintain-
ers are left without clear, evidence-based guidance on
which practices to adopt or change.
From Research Findings to Actionable Rec-
ommendations.In this work, we address this
prediction-action gap by extractingevidence-based
actionable recommendationsfrom the OSS research
literature. We leverage the concept ofReACTs
(Researched Actionables), introduced by Khan and
Filkov [8]: concise, practice-oriented recommenda-
tions grounded in empirical findings from peer-
reviewed software engineering studies. Rather than
predicting sustainability outcomes directly, ReACTs
operationalize research insights into concrete actions
that maintainers can plausibly undertake, such as im-
proving onboarding documentation, adjusting contri-
bution workflows, restructuring release practices, or
adopting specific code review techniques. Each Re-
ACT is explicitly linked to its source study, the em-
pirical evidence supporting it, and the stated impact
on overall project health.
Recent advances in LLMs provide a new oppor-
tunity to scale the extraction of such actionables
from the vast corpus of software engineering research.
LLMs have demonstrated strong capabilities in sum-
marization, reasoning, and structured information
extraction[9], andpromptengineeringtechniquescan
further guide models toward domain-specific outputs
[10]. However, naively applying LLMs to scientific
text risks generating vague, unsupported, or non-
actionablesuggestions, aproblemweaddressthrough
structured validation.
To mitigate these risks, we propose a structured
LLM-based pipeline for extracting high-quality
ReACTs from OSS research articles. Our approach
combines Retrieval-Augmented Generation (RAG)
with a two-layer refinement and reliability-checking
process designed to improve factual grounding,
clarity, and actionability. In the first layer, we sys-
tematically evaluate multiple open-source LLMs and
prompting techniques to identify the best-performing
combination for ReACT extraction. In the second
layer, we apply follow-up prompting to filter hal-
lucinations, extract explicit statements of impact
and empirical evidence, and assess each ReACT for
soundness (logical consistency) and precision (clarity
and specificity). We apply this pipeline at scale
to 829 OSS papers from top SE conference venues
(ICSE and FSE), yielding a curated catalog of
ReACTs organized into practice-oriented categories
that can be mapped to project health signals via the
APEX sustainability framework.Contributions and Relation to Prior Work.
This study builds on and substantially extends prior
work presented in a FSE-IVR idea paper [8]. That
earlier work manually derived 105 actionables from
a fixed set of 186 papers and mapped them to 18
preselected socio-technical features from [6]. The
present work makes four key advances over that
preliminary study:
1.Corpus-scale LLM-based extraction:We
scale ReACT extraction from 186 manually-
reviewed papers to 829 ICSE/FSE articles us-
ing LLMs with a RAG pipeline, systematically
evaluating four open-source models (Llama3-8B,
Mixtral-8x7B, MistralLite-7B, Mistral-Nemo-
12B) and three prompting techniques (Zero-
Shot, Chain-of-Thought, Reason+Action) to
identify the best-performing combination.
2.Two-layer refinement and reliability vali-
dation:We introduce an automated pipeline
that applies follow-up prompting to filter hal-
lucinations (ReACTs not actually mentioned in
source papers), extract explicit impact and evi-
dence statements, and evaluate each ReACT for
soundnessandprecision, yielding1,922validated
ReACTs, of which 1,312 meet strict quality cri-
teria.
3.Category-level organization:Rather than
mappingtofixedpredictivefeatures, weorganize
ReACTs into eight practice-oriented categories
(e.g., New Contributor Onboarding, Code Stan-
dards and Maintainability, Community Collab-
oration) that directly correspond to maintainer
actions and can be flexibly connected to project
states.
4.Operational integration with ASFI:We
demonstrate practical applicability through de-
tailed case studies of Apache Software Founda-
tionIncubator(ASFI)projects, showinghowRe-
ACTs can be selected based on APEX-identified
sustainabilityturningpoints toprovidetargeted,
evidence-based guidance during critical project
phases.
Together, these contributions move beyond man-
ual, feature-bound actionables toward a scalable, re-
producible, and actionable bridge between OSS re-
search and practice. The result is a structured
catalog of evidence-based actions that practitioners
can use to guide OSS projects toward sustainability,
groundedindecadesofempiricalsoftwareengineering
research and validated through rigorous LLM-based
extraction and refinement.
2

2 Background
2.0.1 Language Models for Software Engi-
neering
With the emergence of open-source LLMs, SE re-
searchers and developers are utilizing LLMs for vari-
ous SE tasks. According to a survey study by Zheng
et al. [11], practitioners are using LLMs for seven dif-
ferent SE tasks: a) code generation, b) code sum-
marization (i.e., generating comments for code), c)
code translation (i.e., converting code from one pro-
gramming language to another), d) vulnerability de-
tection (i.e., identifying and fixing defects in pro-
grams), e) code evaluation and testing, f) code man-
agement (i.e., code maintenance activities such as
version control), and g) Q&A interaction (i.e., us-
ing Q&A platforms such as StackOverflow). Ras-
nayaka et al. [12] found that 40.5% of the total stu-
dent teams are using LLMs in academic projects, and
students with higher coding skills are more inclined
to use LLMs. Hou et al. [13] found that there are
traces of using LLMs on 85 different SE tasks, among
them Code Generation and Program Repair are the
most prevalent tasks for employing LLMs in soft-
ware development and maintenance activities. They
also found that practitioners are using different SE
datasets (data on source code, bugs, patches, code
changes, test suites, Stack Overflow, API documen-
tation, code comments, and project issues) for so-
phisticated prompting/fine-tuning the models. Re-
cent work by Babar et al. [14] demonstrated that
open-source LLMs, when enhanced with RAG and
fine-tuning, can effectively answer technical queries
from StackExchange platforms, offering scalable and
cost-effective alternatives to proprietary models for
developer assistance tasks.
The above studies point to the LLMs’ effectiveness
in handling a wide variety of software engineering
tasks and data.
2.0.2 Language Models for Evidence Extrac-
tion
LLMs are being employed in various domains to ex-
tract evidence from documents and scientific liter-
ature. For instance, Gartlehner et al. [15] utilized
the web version of Claude 2 to extract 160 data ele-
ments from 10 open-access Randomized Control Trial
(RCT). The study found that Claude 2 can extract
data with an overall accuracy of 96.3% (6 errors out
of 160 data elements). Likewise, the study conducted
by Huang et al. [16] extracted structured informa-
tion from over 1000 lung cancer and 191 pediatric
osteosarcoma pathology reports. The authors uti-lizedChatGPT-3.5(gpt-3.5-turbo-16k)throughOpe-
nAI’sAPIforbatchquerying. Theevaluationshowed
ChatGPT-3.5 achieved 89% overall accuracy for lung
cancer classifications, outperforming traditional NLP
methods, and 98.6% and 100% accuracy for osteosar-
coma grades and margin status, respectively.
In another study, the authors proposed and evalu-
ated a zero-shot strategy using LLMs to retrieve and
summarize relevant unstructured evidence from pa-
tientElectronicHealthRecords(EHRs)basedonspe-
cific queries [17]. They compared their LLM-based
approach against a pre-LLM information retrieval
baseline, with expert evaluation showing a consistent
preference for the LLM outputs. In another sepa-
rate study, Patiny and Godlin [18] outlined a method
for LLM-based automatic extraction of experimental
FAIR (Findable, Accessible, Interpretable, Reusable)
data of molecules from literature published in the do-
main of chemistry. Here also, authors used ChatGPT
3.5 turbo API version to conduct the experiments
and could extract 74% of the data from published
papers. Similarly, in the field of mental health, Al-
hamed et al. [19] used LLM to extract justification
of a pre-assigned gold label from a suicidality dataset
containing Reddit posts labeled with the suicide risk
level. The authors usedLlama 7bfor experimenting
and achieved a precision of 0.96.
According to previous studies, LLMs are widely
used in various fields to extract evidence from reports
and published literature. However, no such research
hasbeenconductedinSEtoextractevidenceorinfor-
mation from SE literature or technical reports, high-
lighting a plausibility and urgency for exploration.
3 Research Questions
LLMshaveshownpotentialinvarioustasks, buttheir
ability to extract actionable insights from the scien-
tific literature remains under-explored. Given their
advanced capabilities, we posit that LLMs can ef-
fectively generate actionable recommendations that
are both relevant and useful for enhancing practices.
Understanding their efficacy in this domain can un-
lock new avenues for automating knowledge extrac-
tion from research articles.
RQ1Can we use LLMs to effectively de-
riveevidence-basedactionablerecommendations
from scientific literature?
When LLMs generate actionable recommenda-
tions, the quantity and quality of these outputs are
crucial. Poor-quality actionables can mislead both
3

practitioners and researchers, underscoring the im-
portance of evaluating whether LLM-derived recom-
mendations meet the necessary standards for practi-
cal use in enhancing OSS sustainability.
RQ 2:How many evidence-based actionable rec-
ommendations can LLMs derive from scientific
literature published in top SE venues (ICSE and
FSE) in the domain of OSS, and what is the
reliability of those derived actionables?
Even if LLMs can generate high-quality action-
ables, their value depends on their implementation in
real-world settings. The gap between theoretical rec-
ommendations and their practical application often
remains unaddressed, and understanding how these
actionables can be effectively integrated into software
engineering practices, e.g, through existing traces or
workflow monitoring tools, is vital for achieving tan-
gible improvements in OSS projects.
RQ3How can the actionables be presented
meaningfully in practice?
4 Methodology
Here, we describe a number of diverse methods we
utilize to answer our research questions. The overall
study methodology is presented through a conceptual
diagram in Fig. 1.
4.1 Selection of Relevant Articles
Most SE researchers present their best work at pres-
tigious conferences. Thus, our focus is on publica-
tions in top-tier SE conference venues. Specifically,
ICSE (International Conference on Software Engi-
neering) and ESEC/FSE (ACM Joint European Soft-
ware Engineering Conference and Symposium on the
Foundations of Software Engineering), which consis-
tently rank highest in the software engineering do-
main [20, 21]. Our study utilized articles presented
at ICSE and FSE from the 21stcentury (from June
2000 to February 2025).
In the given range, we found a total of 7593 pub-
lished articles in ICSE and FSE. Given our interest
in deriving actionable insights to enhance the sus-
tainability of OSS projects, we further refined our
selection criteria. We included only those articles
whose titles or abstracts contained the strings “Open
Source” or “OSS,” which resulted in 569 ICSE articlesand 260 FSE. Thus, our dataset comprises those 829
conference articles.
4.2 ReACT Definition and Examples
Before describing our extraction methodology, we
first clarify what constitutes a ReACT (Researched
Actionable). AReACTis a concise, evidence-based
recommendation extracted from peer-reviewed soft-
ware engineering research that specifies a concrete
action maintainers can take to improve OSS project
health. Each ReACT consists of three components:
1.Action:A clear, implementable practice (e.g.,
“Implement automated code linting in the CI
pipeline”)
2.Impact:The stated effect on project outcomes
(e.g., “Reduces code review time and improves
code quality”)
3.Evidence:The empirical basis from the source
study (e.g., “Empirical analysis of OSS projects
showed reduction in defect density”)
Example ReACT 1 (Complete: Both Action
& Impact Present):
•Action:“Encourage mentors to provide feedback
on pull requests (PRs) by acknowledging their
efforts and contributions”
•Impact:“Increases newcomer retention and ac-
celerates their path to becoming regular contrib-
utors”
•Evidence:“Mixed-methods study of 12 Apache
projects found that acknowledged newcomers
were 2.3×more likely to make subsequent con-
tributions”
•Source:Feng et al. [22]
Example ReACT 2 (Action-only, no explicit
impact):
•Action:“Consider synchronizing branches before
applying refactorings that are likely to be incom-
patible with changes made in parallel”
•Impact:(Not explicitly stated in source article)
•Evidence:“Analysis of 10,000+ merge scenarios
in 30 projects showed that pre-synchronization
reduced merge conflicts by 41%”
•Source:Oliveira et al. [23]
TheseexamplesillustratethatnotallextractedRe-
ACTs are equally complete or actionable, motivating
our two-layer refinement and reliability assessment
process (described below).
4

4.3 LLMs Task Definition
To address the RQs, we defined three specific tasks
while adopting an experimental consideration, which
are discussed as follows:
4.3.1 ReACT Derivation
In this task, LLMs were provided with a prompt and
the full text of an article, along with instructions for
extracting ReACTs. The ReACTs are the basic units
of our analyses, inspired by a prior limited study [8].
That study analyzed 186 articles and identified Re-
ACTs from 27 of them. From the study’s provided
list of 186 papers, we randomly selected a small sub-
set (n=10) for manual annotation. Two researchers
collaborated to annotate these articles for ReACTs,
successfully identifying ReACTs in six of them, yield-
ing 42 candidate ReACTs for evaluation. The list of
the ten articles, along with their annotations, can be
found in Section 4 of the Appendix.
Using the hand annotations as a reference, we eval-
uated different combinations of LLMs and prompting
techniques (detailed below) to determine the most
effective model and prompting strategy for the Re-
ACT derivation task. The top-performing LLM and
prompting technique was then applied to the full
dataset of 829 articles of this study to derive all Re-
ACTs.
4.3.2 ReACT Refinement
To improve the derived ReACTs, we implemented a
refinement layer aimed at identifying relevant infor-
mation about them. This layer extracts “impact” and
“evidence” for each ReACT and ensures that the ac-
tionables are backed by sufficient contextual informa-
tion. It also filters hallucinated ReACTs, those gen-
erated by the LLM but not actually present in the
source article.
4.3.3 ReACT Reliability Assessment
Thistaskinvolvedassessingthereliabilityofthefinal-
ized, refined ReACTs using predefined criteria. The
purpose of this task is to ensure whether the gener-
atedReACTsarelogicallyconsistent(soundness)and
directly adaptable (preciseness).
4.3.4 Experimental Consideration
In our study, we performed all experiments while
keeping a constant model temperature (temp= 0).
TemperatureinthecontextofLLMsisahyperparam-
eter that controls the randomness of the model’s out-
put [24]. A lower temperature (closer to 0) makes themodel’s responses more deterministic and focused,
while a higher temperature increases randomness and
creativity. The other hyperparameters of the model,
such as:top_p, top_k, min_p, and repeat_penalty
were kept at default values.
4.4 Evaluation Metrics
Due to the lack of standardized evaluation measures
for LLMs, we use a variety of metrics to evaluate
both the LLM performance and the reliability of the
predictions.
4.4.1 LLM Performance: Metrics for Model
Selection
For selecting the best-performing LLM and prompt-
ing technique on our small labeled set, we em-
ployed four quantitative evaluation metrics inspired
by prior literature [25, 26]. The selected metrics can
evaluate the LLM-generated response while giving
gold annotation as a reference. Among the selected
metrics, two are text-similarity-based (BLEU-4and
ROUGE-L), and the other two are semantic-based
(BERTScoreandMETEOR). Semantic-based met-
rics evaluate text by analyzing meaning and context,
considering factors like coherence and relevance [27].
In contrast, similarity-based metrics primarily focus
on lexical overlap or statistical similarity between
texts [28].
These metrics are computed against asingle gold
phrasingper ReACT from our manual annotations.
Because valid ReACTs can be expressed in many
paraphrased forms, BLEU, ROUGE, BERTScore,
and METEOR scores provide only a weak proxy for
true ReACT quality. We use these metrics solely
for relative model selection, to identify which LLM
and prompting combination performs best on our
small labeled set. Theprimary evidence of quality
comes from our subsequent human sanity checks and
inter-annotator agreement (IAA) scores on a strati-
fied sample of 100 ReACTs (described in Section 4.8).
Although there is no universally accepted fixed
threshold for these metrics, recent studies from top-
ranked venues suggest that aBERTScoreabove 0.6,
aBLEU-4score above 0.4, aROUGE-Lscore above
0.4, and aMETEORscore above 0.5 can be consid-
ered indicative of very good performance [25, 29, 30,
26].
BLEU-4.BLEU-4(Bilingual Evaluation Under-
study) calculates the geometric mean of n-gram pre-
cisions up to 4-grams [31]. This metric captures local
fluency and adequacy by measuring the overlap of
short word sequences (up to 4 words long) between
5

paper 
Set-from 
prior 
react 
study 
(n=186)
paper 
set-icse+fse 
(n= 
7593)
selected 
articles 
(n=829)
manual 
annotation
llm 
1
llm 
2
llm 
3
llm 
4
technique 
1
technique 
2
technique 
3
prompting
llms
exploration 
of 
different 
llms-promts
labeled 
dataset 
(n=10)
selected 
articles 
(n=10)
Best-performing
llm
tehnique
labeled 
dataset 
(n=829)
1st 
phase
(
impact 
and 
evidence 
extraction
)
best 
performing 
llm
follow-up 
prompting
2nd 
phase
(
soundness 
and 
preciseness 
assignment)
best 
performing 
llm
follow-up 
prompting
actionable 
set
react, 
impact, 
evidence, 
soundess, 
preciseness
manual 
validity 
checking
selected 
articles 
(n=100)
manual 
assignment 
of  
categories
case 
studies
project 
1
project 
2
project 
specific 
actionables
actionables 
for 
asfi 
project 
1
actionables 
for 
asfi 
project 
2
ASFI
actionable 
with 
categories
mapping 
between 
project 
status 
and 
categories
achieved
 
rq1
achieved
 
rq2
achieved
 
rq3Figure 1: Conceptual Diagram Illustrating the Methodology
the generated text and reference translations.
ROUGE-L.ROUGE-L(Recall-Oriented Under-
study for Gisting Evaluation - Longest Common Sub-
sequence) measures the longest common subsequence
between the generated output and reference text [32].
Unlike n-gram-based metrics, ROUGE-L captures
sentence-level structure and allows for word order
flexibility.
BERTScore.BERTScoreleverages pre-trained
BERT embeddings to assess the semantic similarity
between generated text and reference text [33]. In
our study, we utilized the BERT-base-uncased [34]
variant of BERT, which has 12 layers and 768 hid-
den units; “uncased” means that the model does not
differentiatebetweenuppercaseandlowercaseletters.
METEOR.METEOR(Metric for Evaluation of
TranslationwithExplicitORdering)incorporateslin-
guistic features to capture semantic equivalence and
structural adequacy [35]. Its flexible matching strat-
egyaccountsforexactmatches,stemming,synonymy,
and paraphrasing, which provides a more nuanced
evaluation of semantic similarity.4.4.2 ReACT Reliability Assessment
To assess the final set of ReACTs, we measure their
reliability using two metrics: Soundness and Precise-
ness.
Soundness.A ReACT is sound if it makes logi-
cal sense and has no contradictions. This means that
all parts of the recommendation work together con-
sistently. For instance,“Project Managers should use
peer reviews and automated tools for code review”is
considered a sound ReACT as peer reviews and tools
work together to improve code quality. On the other
hand,“To improve user satisfaction, add new features
without onboarding newcomers”- is considered an un-
sound ReACT because adding features without test-
ing can hurt user satisfaction, and not onboarding
newcomers can reduce support quality.
Preciseness.A ReACT is precise if it is clear and
easytofollow. Itshouldbespecificandleavenoroom
for confusion. For example,“To attract newcomers,
help them make their first contribution”is considered
precise as it gives a clear action to take. On the
contrary,“To attract core developers, ensure high code
quality”-isconsideredimpreciseasitdoesnotexplain
how to ensure high code quality.
6

4.5 LLM Selection
Hugging Faceis one of the best platforms for hosting
LLMs[36]. However, itisquitechallengingtoidentify
the most suitable set of models for a specific task,
given the vast number of models available onHugging
Face. Thus, to address this challenge, we considered
several factors.
4.5.1 Selection Criteria
Privacy.As maintaining the privacy of the papers
is of paramount importance in our study, we selected
LLMs that can be run locally without internet re-
liance. We identified several open-source LLMs from
renowned tech companies like Meta [37], Google [38],
Microsoft [39], and xAI [40], which can operate on
personalcomputersorlocalserverswithoutcloudser-
vices.
Model Types.There exist different types of the
samemodel,suchasinstruct[41],conversational[42],
andcompletion models[43]. However, in this study,
we selectedinstructtype models, as these mod-
els are particularly optimized for following detailed
prompts [41].
Context Window.A critical factor in select-
ing LLMs is the context window, which represents
the maximum number of tokens the model can pro-
cess at once [44]. We calculated the token size for
each article; the maximum token size is 24k. To en-
sure consistency in the ReACT derivation task, we
selected LLMs with context windows exceeding (24
+ 4)k tokens (here, we reserved 4k tokens for the
prompt). Open-source LLMs have context windows
ranging from 2k to 128k tokens, with intermediate
values at 16k, 32k, and 64k. Thus, we considered
models having a context window of at least 32k.
Hardware ConstraintsThemodelswecaneven-
tually run depend significantly on the availability of
computing resources. We conducted all the exper-
iments of our study on a local server, which has a
specificconfiguration(detailedinsectionCoftheAp-
pendix). Therefore, we excluded models that can not
be run on our local server (requiring more than 12GB
VRAM).
Taking all factors into account (privacy, model
type, hardware compatibility, and context win-
dow), we identified four models as potential can-
didates:Llama3-8B, Mixtral-8x7B, MistralLite-7B,
and Mistral-Nemo-12B.
4.5.2 Selected Models
Llama3-8B.Developed by Meta [45],Llama3 8Bis
a language model with 8 billion parameters and adefault context size of 8k tokens [46]. We utilized the
version of the model which has a context window of
32k tokens [47] and is contributed by Nurture AI [48].
Mixtral-8x7B.The modelMixtral 8x7Bwas de-
veloped by Mistral AI [49] with 46.7 billion parame-
ters and a default context window of 32k tokens [50].
Notably, it uses a mixture-of-experts architecture,
which allows for efficient scaling and potentially bet-
ter performance than similarly-sized models.
MistralLite-7B.The modelMistralLiteis con-
tributed by Amazon AWS [51], which has 7 billion
parameters and a context window of 32k [52]. This
model is based on the original Mistral 7B model [53]
(created by Mistral AI [49]). By utilizing adapted ro-
tary embedding and sliding window during training,
the model MistralLite performs significantly better
on several long context tasks while keeping the same
model structure as the original model.
Mistral-Nemo-12B.The modelMistral-Nemo
12Bwas developed by Mistral AI [49], which features
12 billion parameters and a default context window
of 128k tokens [54]. Its large parameter count and
substantial context window contribute to its robust
performance across diverse NLP benchmarks [55].
4.6 Prompt Engineering
A prompt in LLM is the input text or instruction
given to the model to elicit a specific response or gen-
erate relevant output [56]. In this study, we adapted
several well-defined prompting techniques and per-
formed iterative prompt engineering, which is the
process of designing, refining, and optimizing input
prompts to effectively guide LLMs in generating de-
sired outputs [57]. For designing the prompts, we
followed the 26 principled instructions provided by
Bsharat et al. [58], which are effective in getting the
desired response by LLMs. Also, we appended an
emotional stimulus with the prompts (wherever ap-
plicable), as they can significantly boost the perfor-
mance of LLMs on generative tasks [59].
We have performed prompt engineering for each of
our defined tasks, which are described as follows:
4.6.1 ReACT Derivation
For this task, inspired by prior studies [60, 61],
we adapted three prompting techniques:Zero-Shot,
Chain-of-Thought, and Reason+Action, which are
discussed as follows.
Zero-Shot.Zero-Shotprompting allows LLMs to
leverage general knowledge and tackle new problems
based solely on task descriptions [62]. TheZero-Shot
prompt used in our study contains 76 tokens, and the
7

structure of the prompt is provided in Section 1.1 of
the Appendix.
Chain-of-Thought.Proposed by Wei et al. [63],
theChain of Thought (CoT)prompting technique
guides the LLM model to break down complex prob-
lems into intermediate steps, which is similar to
human-like reasoning processes. This approach in-
volves prompting the model to articulate its thought
process, step-by-step, before arriving at a final an-
swer. The adaptedCoTprompt used in our study
contains 336 tokens, and the structure of the prompt
is provided in Section 1.2 of the Appendix.
Here, we divided the whole task into four steps:
first, we instructed the LLMs to read each sentence of
the provided article to look for imperative sentences
or phrases that give commands, make requests, or
offer instructions to direct or persuade someone to
perform a specific action. Second, we instructed the
LLMs to generate a list of actionables in the form of
concise, clear, and unambiguous statements. Next,
during our experiments, it was observed that LLMs
sometimes generate slightly modified versions of the
same actionable multiple times. Thus, we included
step 3, which instructs the LLMs to review the list
of recommendations and remove any duplicate items
from the list. Finally, we introduced another step,
where we instructed the LLMs to provide a confi-
dence score (between 0 and 1), asking how confident
they are that the provided actionable can help OSS
projects become sustainable. We also asked the mod-
els to provide a brief explanation regarding the con-
fidence score. The idea of asking the LLMs to gen-
erate a confidence score is a form of self-evaluation.
According to prior literature [64, 65], it can help the
models overcome hallucinations and produce reliable
responses. Likewise, the idea of “self-confidence” was
adapted to our study to ensure LLMs do not gener-
atespuriousactionables. However, wedidnotinclude
any example set of actionables in the given prompt to
ensure that the model does not get biased by seeing
examples.
Reason+Action.Reason+Actionis a prompting
method that combines the strengths ofCoTprompt-
ing with actionable outputs [66]. This technique en-
courages the LLMs to first reason through a problem
orsituation, andthenprovidespecific, implementable
actions based on that reasoning. UnlikeCoT, this
prompting forces the models to take some intermedi-
ate actions. Following the base structure of theCoT
prompt, theReason+Actionprompt was set up. The
adapted prompt contains 592 tokens, and the struc-
ture of the prompt is provided in Section 1.3 of the
Appendix.
TheReason+Actionprompting approach em-ployed in this study follows a structure similar to
CoTprompting. The prompt is organized into a se-
ries of thought, action, and observation cycles, which
guide the models through a systematic process of
information extraction and refinement. LikeCoT,
initially the task is explicitly defined. This is fol-
lowed by four distinct stages:Extraction, where the
model is instructed to carefully read the article and
compile a list of actionable recommendations;Refine-
ment, where the initial list is reviewed and refined to
ensure clarity;Duplicate Removal, where the list is
reviewed to see if there exist any duplicates (if found,
the models are instructed to remove the duplicates);
andConfidence Scoring, where each recommendation
is assigned a confidence score on a scale of 0 to 1, ac-
companied by a brief explanation.
4.6.2 ReACT Refinement
ReACTsareevidence-basedandshouldprovideatan-
gible action when implemented. While our initial
phase focused on deriving ReACTs, it did not explic-
itly extract “impact” and “evidence” supporting each
actionable. This phase of the study performs this ex-
traction and ensures that each action is accompanied
by both impact and supporting evidence.
As the ReACT refinement task requires process-
ing each actionable, we did not follow any tradi-
tional prompting technique here. Rather, to do the
extraction, we implemented a follow-up prompting
technique inspired by the methodology described by
Polak and Morgan [67]. In their study, the authors
utilized follow-up prompting and achieved significant
accuracy in extracting materials data from scientific
papers. We adapted this effective technique to our
context, which involves a series of binary and descrip-
tive questions.
The follow-up prompting technique is illustrated in
Fig. 3. Here, for this work, we employed the winner
LLM from the ReACT derivation task. This tech-
nique was applied to each actionable derived from
the articles. Initially, the content of the article was
provided to the model, followed by a series of ques-
tions.
Explicit Prompt Sequence:
1.Hallucination Check:“Answer YES or NO
only. Does the article explicitly mention the fol-
lowing actionable: [ReACT text]?”
•If NO→discard ReACT (classified as hal-
lucinated)
•If YES→proceed to impact extraction
2.Impact Extraction:“Answer YES or NO
only. Does the article explicitly state the im-
8

pact(s) on open-source projects if this actionable
is adopted?”
•If YES→“What impact(s) does the article
indicate would result from adopting this ac-
tionable? Provide the exact statement.”
•If NO→classify as “actionable without im-
pact”
3.Evidence Extraction:“Answer YES or NO
only. Does the article provide any empirical evi-
dence to support its claims regarding the stated
impact(s)?”
•If YES→“Describe the empirical evidence
presented in the article that supports the
claim that the recommended action will
produce the stated impact(s).”
•If NO→classify as “actionable without ev-
idence”
This ReACT refinement process, utilizing a se-
quence of follow-up prompts, effectively eliminates
hallucinated ReACTs and ensures that the action-
ables are substantiated with both impact and evi-
dence (if available).
4.6.3 ReACT Reliability Assessment
Based on the definition of ReACT reliability, here
also, we designed a follow-up prompt to assess
whether a particular ReACT is Sound and Precise.
The flowchart for this reliability analysis is shown in
Fig. 4. In this phase as well, we utilized the winner
LLM from task one.
Explicit Prompt Sequence:
1.Soundness Assessment:First, we provide the
model with the definition of “SOUND” (as spec-
ified in Section 4.4.1): “A ReACT is sound if it
makes logical sense, has no contradictions, and
all parts of the recommendation work together
consistently.”
Then we ask: “Answer YES or NO only. Given
the definition of SOUND, can the following ac-
tion be considered SOUND: [ReACT text]?”
•If YES→“What is your rationale for con-
sidering the action SOUND?”
•If NO→“What is your rationale for con-
sidering the action UNSOUND?”
2.Preciseness Assessment:Next, we present
the definition of “PRECISE”: “A ReACT is pre-
cise if it is clear, easy to follow, specific, and
leaves no room for confusion.”Then we ask: “Answer YES or NO only. Given
the definition of PRECISE, can the following ac-
tion be considered PRECISE: [ReACT text]?”
•If YES→“What is your rationale for con-
sidering the action PRECISE?”
•If NO→“What is your rationale for con-
sidering the action IMPRECISE?”
This follow-up prompting process for ReACT qual-
ity assessment ensures that each ReACT meets our
defined quality standards.
4.7 Development of the LLM Pipeline
This section details the development process of the
LLM pipeline used in our study. For designing the
pipeline, we considered that LLMs can hallucinate
and provide factually inaccurate information if un-
aware of the context of the question [69]. To pre-
vent inaccuracies, we adopted the RAG technique,
which possesses a knowledge base and can pass the
required knowledge/context to the model along with
the prompt [70]. Consequently, the RAG technique
helps the model understand the context of the ques-
tion being asked [71].
WhyRAGforPer-ArticleProcessing?While
our pipeline processes one article at a time, RAG pro-
vides several advantages over directly feeding PDF
text: (i)standardized parsing: consistent extraction
and cleaning of PDF content across all 829 articles;
(ii)context injection: seamlessintegrationoffullarti-
cle context with prompts during both derivation and
follow-up phases; (iii)reproducibility: embedding-
based retrieval ensures identical context formatting
across experiments; and (iv)extensibility: the same
pipelinecanbeadaptedformulti-articlequeriesinfu-
ture work. We developed a local RAG-based pipeline
where we placed the articles in the knowledge base
and prompted the models about the given papers.
The architecture of the proposed pipeline is illus-
trated in Fig. 2. The system accepts PDF docu-
ments, which are processed using Python’sPyPDF2li-
brary [72] to extract raw text and create text chunks.
Unlike traditional RAG systems that divide docu-
ments into smaller chunks, our pipeline creates asin-
gle text chunk per article. This approach ensures that
when the retrieval process returns a text chunk, it
contains the entire article rather than a subset of the
article. This chunking process ensures that the model
perceives the context of the entire article before de-
riving ReACTs.
The text chunks undergo a cleaning process where
the “ACKNOWLEDGEMENT” and “REFERENCE”
9

...........
Split 
the 
articles 
in 
Chunks
...........
KnowledgeBase
Chroma 
DB
Converting 
to 
Embedding
Semantic 
Search
FOUND
NOT 
FOUND
Prompt
LLM
ReACTs
PDF 
of 
Articles
...........
Vector 
Embedding
No 
Relevent 
Info
Ranked 
Result
+
PDF 
of 
Article 
1
Semantic 
Index
PDF 
of 
Article 
2
PDF 
of 
Article 
3
PDF 
of 
Article 
n
Article 
to 
Text 
Chunk 
Conversion
Text 
Chunk 
to 
Embedding 
Conversion
Embedding 
1
Embedding 
2
Embedding 
3
Em
b
edding 
n
Text 
Chunk 
1
Text 
Chunk 
2
Text 
Chunk 
3
Text 
Chunk 
nFigure 2: Architecture of the proposed RAG pipeline (Adapted from EvidenceBot [68])
PDF 
of 
Article
Answer 
"YES" 
or 
"NO" 
only. 
Does 
the 
article 
explicitly 
mention 
Actionable 
X?
Discard 
Actionable
Answer 
"YES" 
or 
"NO" 
only. 
Does 
the 
article 
explicitly 
state 
the 
impact(s) 
on 
open-source 
projects 
if 
actionable 
X 
is 
adopted?
Answer 
"YES" 
or 
"NO" 
only. 
Does 
the 
article 
provide 
any 
empirical 
evidence 
to 
support 
its 
claims 
regarding 
the 
stated 
impact(s)?
What 
impact(s) 
does 
the 
article 
indicate 
would 
result 
from 
adopting 
actionable 
X?
 
Describe 
the 
empirical 
evidence 
presented 
in 
the 
article 
that 
supports 
the 
claim 
that 
the 
recommended 
action 
will 
produce 
the 
stated 
impact(s).
YES
YES
Stated 
Impact(s)
Empirical 
Evidence
YES
NO
Classify 
as 
Actionable 
without 
IMPACT
Classify 
as 
Actionable 
without 
EVIDENCE
NO
NO
Figure 3: Flowchart of the Follow-Up Prompting Technique
for ReACT Refinement
Actionable
[Definition 
SOUND]
Answer 
"YES" 
or 
"NO" 
only. 
Given 
the 
definition 
of 
SOUND, 
can 
the 
action 
be 
considered 
SOUND?
[Defination 
PRECISE]
Answer 
"YES" 
or 
"NO" 
only. 
Given 
the 
definition 
of 
PRECISE, 
can 
the 
action 
be 
considered 
PRECISE?
What 
is 
your 
rationale 
for 
considering 
the 
action 
SOUND?
What 
is 
your 
rationale 
for 
considering 
the 
action 
PRECISE?
YES
Rationale 
for 
Being 
SOUND
Rationale 
for 
Being 
PRECISE
YES
LOOP 
ENDS
NO
NOFigure 4: Flowchart of the Follow-Up Prompt-
ing Technique for ReACT Reliability Assess-
ment
10

sections are removed as they are irrelevant for Re-
ACT derivation. The cleaned text chunks are then
converted into embeddings using a fine-tuned embed-
ding model calledinstructor-large[73]. This model,
proposed by Su et al. [74], generates text embeddings
tailored to various downstream tasks and achieved
an average improvement of 3.4% across 70 evaluation
tasks. The embedding dimensionality is 768 (match-
ing BERT-base architecture).
Next, the pipeline usesLangChain[75] to create
semantic indexes for each embedding, which enables
efficient retrieval and ranking of relevant information
based on meaning and context rather than simple
keyword matching [76]. These embeddings and their
semantic indexes are stored inChromaDB[77], which
is a high-performance vector database optimized for
similarity search [78].
Retrieval Parameters:When a user provides a
query, it is converted into an embedding using the
sameinstructor-largemodel. The pipeline performs
a semantic search in the vector database with the em-
bedded query usingcosine similarityas the distance
metric. We setk= 1(retrieve only the single most
relevant article) since our design processes one arti-
cle per query. The plain text of the retrieved article
is appended to the user’s query and passed to the
LLM model, ensuring the model has the full context
of the article. The pipeline is designed for single-
article processing and does not support simultaneous
multi-article queries.
4.7.1 Independence of Pipeline Stages
Although we use the same model across all three
pipeline stages, each stage operates as an indepen-
dent, stateless function call. In the extraction stage,
the model receives article text and outputs candidate
ReACTs. In the refinement stage, the model receives
article text plus a ReACT statement (with no mem-
ory that it generated this ReACT) and validates its
presence while extracting impact and evidence. In
the assessment stage, the model receives only the Re-
ACT text and quality definitions (with no context
about prior stages) and classifies soundness and pre-
ciseness.
This design prevents explicit confirmation
bias—the model cannot “defend” its own outputs
because it doesn’t know they are its own. However, it
does not eliminate systematic model-specific biases:
if the winner model consistently interprets certain
linguistic patterns as actionable or precise due to
its training, these tendencies will manifest across all
stages. We address this through human validation
(Section 4.8).4.8 Sanity Check
In this step, we manually evaluated the performance
of the winner LLM from task one in refining and as-
sessing ReACTs (tasks 2 and 3). After the accom-
plishment of the tasks, 100 ReACTs were randomly
selected from the ReACT dataset for analysis. Two
independent annotators were tasked with this pro-
cess. Given the full text of the articles, they were
asked to evaluate each ReACT by responding in a
YES/NO format for the evaluation metrics(sound,
precise, impact-present, and evidence-present).
Forthemetric-soundandprecise, agreeingwiththe
LLM’s verdict means LLM has performed the classi-
fication correctly, while disagreeing means LLM was
not correct in the classification. While, for the metric
impact-presentandevidence-present, the annotators
either agreed or disagreed with the LLM’s extrac-
tion. For instance, if a ReACT contained an explicit
impact/evidence, agreeing indicated that the LLM
had correctly identified the impact/evidence; while
disagreeing suggested that the extracted impact/ev-
idence was incorrect. Conversely, for a ReACT with
no explicit impact/evidence mentioned, agreeing in-
dicated that the LLM had correctly recognized the
absence of impact, while disagreeing meant that the
impact/evidence was actually present in the article,
but the LLM failed to extract it.
To measure the consistency of the annotation,
the Inter-Annotator Agreement (IAA) [79] score was
used. After the independent annotations, the review-
ers discussed the annotations with each other and
resolved the disagreements through discussion. After
the disagreements were resolved, we got the final hu-
man labeling/gold annotation, which we compared
with the LLM’s actual response for performing the
sanity check.
4.9 Categorization of ReACTs
ToprovideOSSproject-specificReACTstothedevel-
oper community, we categorized them into eight dis-
tinct categories: a)New Contributor Onboarding and
Involvement; b)Code Standards and Maintainabil-
ity; c)Automated Testing and Quality Assurance; d)
Community Collaboration and Engagement; e)Doc-
umentation Practices; f)Project Management and
Governance; g)Security Best Practices and Legal
Compliance; and h)CI/CD and DevOps Automa-
tion. These categories were established based on an
in-depth discussion and careful consideration of the
specific needs of OSS projects. The categories’ def-
initions and the criteria for categorization (i.e., the
conditions qualifying a ReACT for inclusion in a par-
11

ticular category) are provided in Section 2 of the Ap-
pendix.
Two independent researchers were tasked with as-
signing predefined categories to each ReACT. Al-
thoughasingleReACTmaybeapplicabletomultiple
categories, the annotators were instructed to assign
the category most relevant to each ReACT. Like the
sanity check, the quality of these annotations was as-
sessed using the IAA score. Following the annotation
process, both researchers collaboratively resolved any
disagreements through discussion.
4.10 Case Studies
We conducted case studies with marginal projects
to demonstrate how ReACTs can be effectively ap-
plied to support the development of OSS projects.
The projects were selected from the Apache Software
Foundation Incubator (ASFI) [80]. Marginal projects
refer to those that either showed initial promise but
ultimately failed or those that started poorly but
later succeeded.
ASFI [80] serves as a gateway for OSS projects
aiming to join the ASF ecosystem [81]. In ASFI,
projects receive mentorship from senior members on
the “Apache Way” so they can mature and align
with Apache’s principles of community-driven devel-
opment and open governance. The incubation pro-
cess typically spans one to three years [81]. Near the
endofincubation, aprojectfacesoneofthreepossible
outcomes: graduation, continuation, or retirement,
decided by the project members and a Project Man-
agement Committee (PMC). The desired outcome of
a project is graduation, which is only achieved if a
project fulfills all ASF goals. However, if a project
is not yet ready for graduation but shows promise,
it may continue in the Incubator for an extended pe-
riod. On the contrary, if a project fails to make suffi-
cient progress or loses community support, it may be
retired from the Incubator.
We utilized project activity data from the APEX
(Apache Project EXplorer) tool, developed by Ram-
chandran et al. [82]. APEX is an open-source web
application dashboard tool designed to monitor and
explore the sustainability trajectories of 273 ASFI
projects. It features an AI-based sustainability fore-
casting model, interactive visualizations of social and
technical networks, and detailed manifestation of
project activity data. In the tool, the social net-
works denote the communication among developers
in the Apache mailing list [83], while technical net-
works represent developers’ code contributions to dif-
ferent file types. The activity data of the tool include
month-wise social and technical network, number ofemails, commits, contributors, etc. The tool employs
an LSTM model trained on 18 socio-technical fea-
tures to generate month-by-month graduation prob-
abilities. The graduation probability at monthnre-
flects the model’s confidence that the project will
graduate, based on data from months 0 ton-1. If
a project’s graduation probability rises above 0.5, it
indicates an upturn, while a drop below 0.5 signals
a downturn. Using APEX, users can explore project
data over time, compare different projects, and an-
alyze sustainability turning points. The tool can be
publicly accessed at this link.
5 RESULTS AND DISCUS-
SION
5.1 RQ1: Can we use LLMs to ef-
fectivelyderiveevidence-basedac-
tionable recommendations from
the scientific literature?
Summary:Among four LLMs and three prompt-
ing techniques evaluated on our small labeled set
of 10 papers (42 annotated ReACTs), Mixtral-8x7B
with Chain-of-Thought prompting achieved the high-
est scores across all metrics (BLEU-4: 0.66, ROUGE-
L:0.49, BERTScore: 0.75, METEOR:0.56), substan-
tially outperforming other combinations and estab-
lishing it as the best available model-prompt pair for
ReACT extraction in our experimental setting.
We evaluated four different LLMs (Llama3-8B,
Mixtral-8x7B, MistralLite-7B, and Mistral-Nemo-
12B) and three distinct prompting techniques (Zero-
shot, CoT,andReason+Action)forderivingReACTs
from articles. Performance was assessed using four
metrics (BLEU-4, ROUGE-L, BERTScore, and ME-
TEOR) and delineated in Table 1. As discussed
in Section 4.4.1, these metrics are weak proxies for
true ReACT quality due to paraphrasing variability,
but serve to identify the relatively best-performing
model-prompt combination on our labeled set.
Mixtral-8x7BpairedwiththeCoTpromptingtech-
nique performed best among the evaluated configu-
rations, substantially outperforming other combina-
tions across all metrics. This pairing demonstrated a
strong ability to balance lexical overlap and seman-
tic depth, making it the most effective configuration
among those tested for ReACT derivation. While
MistralLite-7Bshowedmixedresults, itsperformance
was inconsistent across prompting techniques, with
Reason+Actionperforming slightly better but not
achieving the consistency seen withMixtral-8x7B.
Conversely,Llama3-8Bconsistently performed
12

poorly across all metrics and prompting techniques.
Its low scores indicate that this model struggles
with both lexical and semantic tasks in this context.
Mistral-Nemo-12B, despite its larger size (12B pa-
rameters), also showed disappointing results regard-
less of the prompt used, suggesting that parameter
count alone does not guarantee better performance
for specialized extraction tasks.
Given the small size and author-annotated nature
of the labeled set, this comparison should be inter-
preted as a pragmatic model-selection heuristic, not
a definitive ranking of LLM capabilities
RQ1Findings:LLMs can be used to derive ac-
tionable recommendations from scientific articles.
Amongtheevaluatedmodelsandpromptingtech-
niques tested on our small labeled set (10 papers,
42 ReACTs), the combination ofMixtral-8x7B
with theChain-of-Thoughtprompting method
achieved the best performance by a large margin
across all metrics, establishing it as our selected
configuration for full-scale extraction.
5.2 RQ2: How many evidence-based
actionable recommendations can
LLMs derive from scientific liter-
ature published in top SE venues
(ICSE and FSE) in the domain of
OSS, and what is the reliability of
those derived actionables?
Summary:From 829 ICSE/FSE papers, we ex-
tracted 2,023 candidate ReACTs. After filtering 101
hallucinations through follow-up prompting, 1,922
validated ReACTs remained. Of these, 1,673 (87%)
include explicit impact statements, 1,458 (76%) in-
clude empirical evidence, 1,901 (99%) are sound, and
1,697 (88%) are precise. Overall, 1,312 ReACTs
(68%) meet all quality criteria (sound, precise, with
bothimpactandevidence), constitutingourcomplete
ReACT catalog. Human validation on a stratified
sample of 100 ReACTs showed 87–100% agreement
with LLM classifications across metrics.
Leveraging the findings fromRQ1, we used the
combinationMixtral-8x7BwithCoTprompting to
extract actionables from the entire dataset of 829 ar-
ticles. We successfully derived ReACTs from 474 ar-
ticles (57%), while the remaining 355 articles (43%)
did not produce any ReACTs—either because they
did not focus on actionable practices or because
their findings were too abstract for concrete extrac-
tion. The model’s self-confidence score for ReACTderivation averaged 0.8, with a standard deviation of
0.1436. In total, this initial layer of analysis resulted
in the extraction of 2,023 ReACTs, or approximately
4.3 per article on average (calculated over the 474
articles that yielded ReACTs).
Following the initial extraction, we implemented
our adapted two-phase second-layer prompting. The
first phase involved a series of targeted questions to
the model, designed to ascertain three key aspects:
(1) whether the ReACT was explicitly mentioned in
thearticle, (2)theimpactoftheactionableitemmen-
tioned in the article, and (3) the stated evidence sup-
porting the ReACT in the article. During this phase,
Mixtral-8x7Bwas utilized as the LLM model, as it
emerged as the top-performing LLM in our study
based on the findings fromRQ1. This step identified
101 ReACTs (5% of initial extractions) that were not
actually mentioned in the articles. These were classi-
fied as hallucinated ReACTs and were consequently
removed from the dataset prior to further analysis.
This filtering process resulted in a refined set of 1,922
ReACTs for subsequent examination.
The next phase in the second layer of prompting
was to supplement each of the remaining ReACTs
with impact and evidence information. The model
successfully identified explicitly mentioned impacts
for 1,673 ReACTs (87%), while no explicit impact
was found for 249 ReACTs (13%). It is worth noting
that the impact of some actionable items is implicit
in the action itself (e.g., “Implement automated test-
ing” implicitly improves reliability), which explains
why the authors of the scientific articles do not al-
ways explicitly state the impact of the actionable in
their articles. In terms of evidence, the model iden-
tified explicit empirical evidence for 1,458 ReACTs
(76%) out of the 1,922 analyzed. The lower evidence
extraction rate compared to impact suggests that ev-
idence statements are more challenging for the LLM
to identify—possibly because evidence often appears
in scattered form across Results and Discussion sec-
tions rather than being localized near the actionable
statement. Notably, the model found 237 actionable
items (12%) for which there was no explicit mention
of either impact or evidence in the source articles.
In the final phase of our analysis, we employed the
top-performing LLM from our study,Mixtral-8x7B,
to evaluate the ReACTs based on their soundness
and preciseness characteristics. This process involved
passing each actionable through the model and pos-
ing a series of questions to determine whether each
action item met the Sound and Precise criteria. The
results of this analysis revealed that out of the 1,922
ReACTs, only 21 (1.1%) were classified as Unsound,
while 1,901 (98.9%) were classified as Sound. Ad-
13

Table 1: Performance Analysis of ReACT Derivations Across Different LLMs and Prompting Techniques
ModelPrompt
TechniqueBLEU-4 METEOR ROUGE-L BERT
Llama3-8bZero Shot 0.04 0.20 0.09 0.46
Chain-of-Thought 0.06 0.15 0.07 0.46
Reason+Action 0.08 0.16 0.09 0.47
Mixtral-8x7bZero Shot 0.38 0.38 0.32 0.62
Chain-of-Thought0.66 0.56 0.49 0.75
Reason+Action 0.14 0.17 0.10 0.46
MistralLite-7bZero Shot 0.44 0.22 0.23 0.52
Chain-of-Thought 0.31 0.11 0.12 0.46
Reason+Action 0.49 0.30 0.27 0.57
Mistral Nemo-12BZero Shot 0.10 0.14 0.08 0.45
Chain-of-Thought 0.12 0.14 0.08 0.43
Reason+Action 0.09 0.14 0.07 0.43
ditionally, 226 ReACTs (11.8%) were categorized as
Imprecise, with 1,697 (88.2%) classified as Precise.
Interestingly, 8 ReACTs were found to be both Un-
sound and Imprecise, representing cases where logical
inconsistenciescoincidedwithvagueorunclearphras-
ing. Overall, we identified 1,312 ReACTs (68.3% of
validatedReACTs)ascomplete, meetingallcriteria:
SOUND, PRECISE, and having explicit mentions of
both impact and evidence.
5.2.1 Human Validation and Sanity Check
To validate the efficacy of our study’s methodology,
we conducted a manual evaluation of the ReACTs as
described in Section 4.8. This process involved se-
lecting 100 ReACTs via stratified random sampling
across the entire set of ReACTs. Two independent
evaluatorsparticipatedinthisvalidationprocess, and
their consistency was measured by IAA. The IAA be-
tween the annotators were: 95 for impact, 94 for evi-
dence, 98 for soundness, and 96 for preciseness, indi-
cating strong consistency in human judgment.
Next, the annotators resolved their disagreements
through discussion, and we obtained the final human-
labeled gold annotation. We then compared the gold
annotation with the LLM’s original response. We
observed a match rate of 98% forimpact-presentand
87% forevidence-present. The lower match rate for
evidence (87% vs. 98% for impact) confirms that
accurately extracting ReACT evidence is more chal-
lenging for the LLM than extracting impact—likely
because evidence statements are more scattered and
varied in form across papers. Soundness achieved a
96% match, while preciseness achieved a 98% match.
The near-perfect agreement on impact and precise-
ness—underscores the robustness and efficacy of our
two-layer refinement and reliability assessment ap-proachinextractingandanalyzingReACTsfromaca-
demic literature.
It is important to clarify what these high sound-
ness and preciseness rates do and do not indicate.
Soundness, as defined in this study, is alogical coher-
ence check: a ReACT is considered sound if its parts
areinternallyconsistentandfreeofexplicitcontradic-
tions, not if the recommended action is guaranteed to
be effective in practice. Likewise, a high preciseness
score indicates that the recommendation is clearly
articulated, specific, and easy to follow, not that it
is universally applicable or sufficient on its own to
ensure project sustainability. Preciseness reflects lin-
guistic clarity and operational specificity, rather than
contextual fit or empirical strength beyond what is
reported in the source study.
RQ2Findings:We identified 1,922 validated
ReACTs from 829 ICSE/FSE papers. Of these,
1,673 (87%) were supported by explicit impact,
1,458 (76%) by explicit evidence, 1,901 (99%)
classified as SOUND, and 1,697 (88%) as PRE-
CISE. Consequently, 1,312 ReACTs (68%) met
all criteria (SOUND, PRECISE, with both im-
pactandevidence), constitutingourcompleteRe-
ACTcatalog. Humanvalidationon100stratified-
sampled ReACTs showed 87–100% agreement
with LLM classifications.
5.3 RQ3: How can the actionables be
presented meaningfully in prac-
tice?
Summary:We organized 1,922 ReACTs into eight
practice-oriented categories and demonstrated their
practical application through case studies of two
14

ASFI projects. The case studies show how maintain-
ers can use APEX to identify sustainability turning
points, select relevant ReACTs from appropriate cat-
egories, and implement evidence-based interventions
during critical project phases.
To answer this RQ, we first categorized the Re-
ACTs into practice-oriented categories, then per-
formed case studies on projects from the ASFI us-
ing the APEX tool to demonstrate how ReACTs can
guide real-world OSS projects toward sustainability.
5.3.1 Categorization of ReACTs
We manually categorized the ReACTs into eight pre-
defined categories as described in Section 4.9. The
consistency of the annotation was evaluated with
IAA, which yielded a score of 0.84, indicating strong
agreement. Next, for each category, we gathered sta-
tistical data to assess both the support for the Re-
ACTs and the quality of that support. A summary
of these statistics is provided in Table 2.
Several notable patterns emerge from this catego-
rization:
•Distribution:The majority of ReACTs were
assigned to the categoryAutomated Testing
and Quality Assurance(n=607, 31.6%), while
the fewest were placed in theNew Contributor
Onboarding and Involvementcategory (n=60,
3.1%). This distribution reflects the historical
focus of OSS research on technical quality prac-
tices over community-building practices.
•Completeness Variability:The percentage
of Complete ReACTs (those meeting all qual-
ity criteria) ranged from 54% (Community Col-
laboration and EngagementandDocumentation
Practices) to 85% (CI/CD and DevOps Automa-
tion). This substantial variation suggests that
research in DevOps and automation tends to
provide more explicit, well-evidenced recommen-
dations, while community-oriented research of-
ten presents findings that are less directly ac-
tionable or backed by less concrete empirical
evidence. This may reflect the relative matu-
rityofautomatedpracticesresearchversussocio-
technical community research, or it may indicate
that technical practices are easier to operational-
ize into concrete actions.
•Evidence Backing:ReACTs backed by evi-
dence ranged from 60% (Community Collabora-
tion and Engagement) to 95% (CI/CD and De-
vOps Automation). The lower evidence rates
in community-oriented categories may reflectthe methodological challenges of empirically val-
idating social interventions in volunteer OSS
contexts, whereas technical practices (testing,
CI/CD) can be more readily evaluated through
quantitative metrics (build times, defect rates,
etc.).
Across all categories, the percentage of Sound Re-
ACTs was consistently high (98–100%), indicating
that nearly all extracted recommendations are log-
ically coherent. PRECISE ReACTs ranged from 83%
to 91%, suggesting that while most ReACTs provide
clear guidance, approximately 10–15% remain some-
what vague or require additional context for imple-
mentation. The variability in LLM confidence scores
was minimal, consistently ranging between 77% and
81%, indicating stable model confidence across differ-
ent practice areas.
5.3.2 Case Studies: Operationalizing Re-
ACTs with APEX
We selected two projects for case studies toillus-
trate how ReACTs can be operationalized in prac-
tice. These case studies are intended as exploratory
demonstrations of how practitioners could use Re-
ACTs to interpret project signals and identify rel-
evant evidence-based actions. The selected projects
includeonethatfacedaninitialdownturninitslifecy-
cle but later graduated (CommonsRDF) and another
that showed good promise in the beginning but later
retired (Tamaya). Project activity data and gradu-
ation forecasts of these projects are shown in Fig. 5
(obtained from the APEX tool). Both projects faced
significant downturns in their lifecycles (see Gradu-
ation Forecast panels). ForCommonsRDF, a signif-
icant downturn can be observed in month 5, while
Tamayaexperienced a downturn in month 39.
MaintainerWorkflowforUsingReACTs:We
describe a workflow that project maintainers can fol-
low to select and implement ReACTs based on sus-
tainability signals:
Step1: IdentifyTurningPoints.UsingAPEX’s
graduation forecast visualization, maintainers iden-
tify months where the sustainability probability
drops below 0.5 (downturn) or exhibits sustained
decline. We selected downturn intervals for both
projects to analyze the causes of decline. The down-
turn interval includes two months before and two
months after the turning point. Thus, the downturn
interval forCommonsRDFspans months 1–5, while
forTamayait spans months 37–41.
15

Table 2: Summary of ReACT Derivations Across Various ReACT categories; The table presents a breakdown
ofthetotalnumberofarticlescontributedtoacategory, ReACTsforeachcategory, alongwiththepercentage
of those which are SOUND, PRECISE, contain empirical evidence, and have an impact. The ’Complete
ReACTs’ column represents ReACTs that meet all these criteria. The notation (# (%)) indicates both the
actual count of ReACTs and the percentage relative to the total number of ReACTs in that category. The
final column shows the average LLM confidence score for each category.
ReACT Category# of
Articles# of
ReACTsSOUND
ReACTs
(# (%))PRECISE
ReACTs
(# (%))ReACTs
with
Impact
(# (%))ReACTs
with
Evidence
(# (%))Complete
ReACTs
(# (%))LLM
Confidence
Score (%)
New Contributor Onboarding
and Involvement43 60 59 (98.33%) 54 (90.00%) 50 (83.33%) 46 (76.67%) 42 (70.00%) 79.92
Code Standards
and Maintainability228 394 388 (98.48%) 358 (90.86%) 354 (89.85%) 310 (78.68%) 288 (73.10%) 81.11
Automated Testing
and Quality Assurance309 607 602 (99.18%) 532 (87.64%) 531 (87.48%) 481 (79.24%) 431 (71.00%) 80.83
Community Collaboration
and Engagement139 242 239 (98.76%) 212 (87.60%) 197 (81.40%) 146 (60.33%) 131 (54.13%) 78.47
Documentation
Practices133 173 169 (97.69%) 144 (83.24%) 146 (84.39%) 109 (63.01%) 96 (55.49%) 78.27
Project Management
and Governance94 154 153 (99.35%) 132 (85.71%) 126 (81.82%) 105 (68.18%) 92 (59.74%) 77.32
Security Best Practices
and Legal Compliance81 155 155 (100.00%) 138 (89.03%) 140 (90.32%) 131 (84.52%) 115 (74.19%) 80.35
DevOps Automation
and CI/CD87 137 136 (99.27%) 126 (91.97%) 129 (94.16%) 130 (94.89%) 117 (85.40%) 80.69
Step 2: Analyze Socio-Technical Signals.The
APEXtoolhasarangesliderfeaturethatallowsusers
to select an interval. Selecting an interval enables the
tool to show aggregated project activity data dur-
ing that interval—including contributor counts, com-
mit frequencies, communication patterns (social net-
work), and code contribution patterns (technical net-
work). Consequently, we selected the stated down-
turn intervals for the projects and examined their
socio-technical indicators.
Step 3: Map Signals to ReACT Categories.
Based on the observed deficits, maintainers can select
relevant ReACT categories. For example, low con-
tributor counts suggestNew Contributor Onboarding
and InvolvementReACTs; high code complexity or
low contributions per developer suggestCode Stan-
dards and MaintainabilityReACTs; minimal collab-
oration suggestsCommunity Collaboration and En-
gagementReACTs.
Step 4: Select Evidence-Based ReACTs.
From the selected categories, maintainers review
ReACTs and choose those most applicable to
their project context, prioritizing Complete ReACTs
(those with both impact and evidence) to maximize
confidence in the intervention.Step 5: Implement and Monitor.After imple-
menting selected ReACTs, maintainers can monitor
subsequent metrics (monthsn+ 1,n+ 2, etc.) of the
project to assess whether socio-technical indicators
improve and whether the graduation forecast trajec-
tory recovers.
Case Study 1: CommonsRDF
For the projectCommonsRDF, APEX data dur-
ing the downturn period (months 1–5) revealed that
only seven individuals contributed to the social net-
work, while three contributed to the technical net-
work. This suggests the project is in need of recruit-
ing new developers to mitigate its struggling condi-
tion. Additionally, there were fewer technical con-
tributions per developer during this time (averaging
8 commits per developer compared to 15 in healthy
projects), highlighting the need for improved coding
standards to reduce the challenges developers face
when contributing to the project’s codebase.
While the necessary interventions (recruit contrib-
utors, improve code quality) are clear from APEX
signals, implementing them is not straightforward.
Recruiting new contributors in an OSS project is in-
herently challenging due to the voluntary nature of
OSS participation. Likewise, enhancing code quality
is a complex and gradual process. This is where Re-
ACTs provide actionable leverage. Actionables from
the categoriesNew Contributor Onboarding and In-
16

volvementandCode Standards and Maintainability
offer evidence-based guidelines that developers can
follow to ensure long-term accomplishment of the re-
quired tasks.
Some of the ReACTs from our compiled catalog
that could help theCommonsRDFproject include:
1.Encourage mentors to provide feedback
on pull requests (PRs) by acknowledging
their efforts and contributions(New Con-
tributor Onboarding and Involvement)
•Impact:Increases newcomer retention and
accelerates their path to becoming regular
contributors
•Evidence:Mixed-methods study of 12
Apache projects found that acknowledged
newcomers were 2.3×more likely to make
subsequent contributions
•Source:Feng et al. [22]
2.Consider synchronizing branches before
applying refactorings that are likely to be
incompatible with changes made in paral-
lel(Code Standards and Maintainability)
•Impact:Reduces merge conflicts and inte-
gration overhead, making it easier for de-
velopers to contribute
•Evidence:Analysis of 10,000+ merge sce-
narios in 30 projects showed that pre-
synchronization reduced merge conflicts by
41%
•Source:Oliveira et al. [23]
Expected Outcome:By implementing these
ReACTs,CommonsRDFmaintainers would have
evidence-based strategies to (a) increase newcomer
retention through acknowledged mentorship, and (b)
reduce technical friction via better branching prac-
tices. Subsequent monitoring of the project (months
6–10) would reveal whether contributor counts in-
creased and whether the graduation forecast trajec-
tory improved.
Case Study 2: Tamaya
Similar challenges are evident for the project
Tamaya, with a limited number of active contribu-
tors during its downturn phase (months 37–41): five
in the social network and two in the technical net-
work. Notably, there is minimal collaboration among
developers on the technical side. Of the two techni-
callyactivecontributors, oneindividualisresponsible
for 98% of the commits, while the other contributes
only 2%. This extreme concentration indicates a lackof distributed ownership and knowledge sharing—a
critical vulnerability for project sustainability.
Therefore, duringTamaya’s downturn phase, im-
plementing ReACTs from the categoriesCommunity
Collaboration and EngagementandNew Contributor
Onboarding and Involvementcould help redirect its
trajectory. These evidence-based recommendations
can support the project in overcoming its current
challenges. Specific ReACTs thatTamayacould have
adopted during this phase include:
1.Develop a clear code of conduct that out-
lines expectations for behavior within the
community(New Contributor Onboarding and
Involvement)
•Impact:Establishes safe, inclusive environ-
ment that attracts diverse contributors
•Evidence:Analysis of GitHub projects
showed that adoption of codes of conduct
correlated with 15% increase in contributor
diversity and 22% reduction in negative in-
teractions
•Source:Zhao et al. [85]
2.Incorporate gamification approaches, such
as awarding badges or points, to in-
centivize community-oriented initiatives
(Community Collaboration and Engagement)
•Impact:Increases engagement and mo-
tivates contributions through recognition
mechanisms
•Evidence:Controlled study with 50 OSS
projects found that gamification increased
participation by 28% and sustained engage-
ment over 6+ months
•Source:Miller et al. [86]
Counterfactual Reflection:Historically,
Tamayadid not implement such interventions and
was eventually retired. Had maintainers used APEX
to detect the downturn at month 39 and applied
these ReACTs, they could have potentially addressed
the knowledge concentration (98% commits by
one person) and low collaboration. Subsequent
APEX monitoring would have shown whether new
contributors joined, whether commit distribution
improved, and whether the graduation forecast
stabilized. While we cannot definitively prove
these ReACTs would have savedTamaya, the case
study illustrates how the ReACT catalog provides
maintainers with concrete, evidence-based options
at critical junctures—options that did not exist in a
structured, accessible form before this work.
17

Downturn 
Interval
Less 
Number 
of 
Contributors
Less 
Technical 
Contributions 
Per 
Developer(a)
Less 
Number 
of 
Contributors
Downturn 
Interval
Less 
Collaboration 
among 
Developers
(b)
Figure 5: Metrics for two marginal projects from ASFI: (a) graduated projectCommonsRDFwhich almost
failed, while (b) retired projectTamayawhich almost succeeded. The screenshots are adopted from the
APEX tool [84]. The figures include annotations (in red) that analyze the factors contributing to the
downturn.
18

It should be noted that these case studies serve
as proof-of-concept demonstrations of how ReACTs
can be mapped to observable project states, rather
than as evidence that the recommended actions were
adopted or that they produced the described out-
comes. The goal is to show that ReACTs can be
meaningfully connected to real project conditions,
the question of whether their implementation leads
to sustained improvements remains an open and im-
portant direction for future empirical work
RQ3Findings:We present a systematic ap-
proach detailing when and how actionable rec-
ommendations can be implemented to steer the
future trajectories of OSS projects toward suc-
cess. ReACTs are organized into eight practice-
oriented categories with 54–85% completeness
rates across categories. Case studies demon-
strate a concrete five-step maintainer workflow:
(1) identify turning points via APEX, (2) ana-
lyze socio-technical signals, (3) map signals to
ReACT categories, (4) select evidence-based Re-
ACTs, and (5) implement and monitor outcomes.
6 Threats to Validity
This study has the following limitations, each tied to
specific methodological choices:
Venue Selection and Generalizability.We
considered only articles published in top-tier SE con-
ference venues (ICSE and FSE). This choice was
motivated by their high citation impact and qual-
ity control, but it introduces potential bias: the
ReACT catalog may underrepresent findings from
practitioner-orientedvenues(e.g., ICSE-SEIP,MSR),
journal publications (e.g., IEEE Software, TOSEM,
EMSE, JSS), or industry reports. Consequently, our
ReACTs may emphasize academic research priorities
(e.g., technical quality, testing) over practitioner con-
cerns (e.g., governance, legal compliance, business
models). Future studies should expand the corpus
to include journal articles and practitioner venues to
broaden coverage of non-code aspects such as gover-
nance, licensing, and community management.
In-Context Learning vs. Fine-Tuning.We
considered only in-context learning, where LLMs per-
form tasks based on examples or instructions pro-
vided within the input context [87]. In this approach,
no fine-tuning is performed and the model tackles
newtasksbyleveragingpatternsandinformationpre-
sented in the given prompt. However, LLM models
can struggle with in-context learning by putting morefocus on the later part of the input sequence [88], po-
tentially biasing extraction toward conclusions and
recommendations that typically appear near the end
of papers. Future studies may focus on using fine-
tuned models for ReACT derivation by using a la-
beled dataset containing the mapping between arti-
cles and ReACTs. Fine-tuning allows the model to
specializeintargetedextractiontasks, improvingper-
formance on domain-specific applications while main-
taining general knowledge.
Model and Hardware Constraints.Due to
our hardware constraints (12 GB VRAM), we could
not run very large open-source models, such asGrok-
1(314 billion parameters) [89],Mixtral-8x22B(141
billion parameters) [90], orDBRX(132B parame-
ters) [91]. These models have the potential to provide
better results as they are trained on vast amounts of
data and possess more complex architectures. Like-
wise, we could not consider some advanced prompt-
ing techniques, such asTree-of-Thought (ToT),Few-
Shot Prompting, andCoT with Self-Consistency, as
these techniques require large-parameter models with
extended context windows to generate reliable re-
sponses [92, 93, 94]. Therefore, the potential for
future research lies in exploring how large-scale pa-
rametermodelswithextendedcontextwindows, com-
bined with advanced prompting techniques, perform
in deriving actionable recommendations. Such mod-
els may extract more nuanced ReACTs or better dis-
ambiguate vague statements.
Evaluation Metrics and Paraphrasing.
The evaluation metrics used for model selection,
BERTScore,METEOR,BLEU-4, andROUGE-L,
focus mainly on quantitative performance computed
against a single gold phrasing and may not fully
capture qualitative aspects such as interpretability,
robustness, and sensitivity to specific errors [95].
Because valid ReACTs can be expressed in many
paraphrased forms, these metrics provide only
weak proxies for ReACT quality. We mitigated
this limitation by using these metrics solely for
relative model selection on a small labeled set, then
relying on human sanity checks and inter-annotator
agreement as the primary evidence of quality. Future
studies could concentrate on developing qualitative
metrics or human-in-the-loop evaluation frameworks
that address these gaps, aiming to provide a more
comprehensive evaluation of model performance that
captures aspects like interpretability, robustness,
and sensitivity to errors not fully covered by existing
quantitative metrics.
19

7 Ethics and Data Use
All full-text PDFs were processed locally on our own
server infrastructure, with no data transmitted to ex-
ternal cloud services or APIs. This approach ensures
that proprietary or embargoed research content was
notinadvertentlyleakedorshared. Ouroutputs—the
ReACT catalog—consist of high-level, abstracted ac-
tionable recommendations rather than verbatim re-
production of full article text, mitigating copyright
concerns. Each ReACT is explicitly linked to its
source paper via citation, enabling readers to consult
the original work for full context while respecting in-
tellectual property rights. No human subjects data
or personally identifiable information was involved in
this study. The APEX tool used for case studies re-
lies on publicly available Apache Software Founda-
tion project data (mailing lists, commit logs), and no
private or sensitive project information was accessed.
8 Conclusion
This study presents a novel approach to enhancing
OSS project sustainability through the application
of LLMs for extracting actionable recommendations
from the scientific literature. Our research demon-
strates the efficacy of theMixtral-8x7Bmodel with
Chain-of-Thoughtprompting in deriving evidence-
based actionable recommendations (ReACTs). From
a corpus of 829 articles, we extracted 1,922 unique
ReACTs, with 1,312 meeting rigorous criteria for
soundness, preciseness, and empirical support. The
introduction of a systematic approach for categoriz-
ing these ReACTs into thematic groups and applying
thempractically,asillustratedthroughcasestudiesof
ASFIprojects, offersascalableandreplicablemethod
for addressing specific challenges in OSS project life-
cycles. Our methodology, incorporating a two-layer
prompting technique and manual validation, ensures
the quality and relevance of the extracted ReACTs.
While acknowledging the limitations, this study rep-
resents a significant advancement in leveraging AI
technologies to address OSS sustainability. By bridg-
ing the gap between academic research and practi-
cal application, we provide a valuable resource for
the OSS community and open new avenues for fu-
ture research, including the exploration of long-term
impacts of ReACT implementation and the develop-
ment of dynamic recommendation systems based on
real-time project metrics.9 Data Availability Statement
The complete datasets and code for repli-
cation are available at the following link:
<https://zenodo.org/records/13744866>. For
convenient access, the final set of actionable recom-
mendations can also be accessed via the following
web link:<Actionable-Set>
10 ACKNOWLEDGMENTS
The research received funding from the National Sci-
ence Foundation under Grant No. 2020751
References
[1] Rachel Layne. Open source software: The $9
trillion resource companies take for granted,
2024. Accessed: 2024-07-15.
[2] Gitnux. Open source software
statistics.https://gitnux.org/
open-source-software-statistics/, 2024.
[3] Jailton Coelho and Marco Tulio Valente. Why
modern open source projects fail. InProceed-
ings of the 2017 11th Joint meeting on foun-
dations of software engineering, pages 186–196,
2017.
[4] Charles M Schweik and Robert C English.In-
ternet success: a study of open-source software
commons. MIT Press, 2012.
[5] Amir Hossein Ghapanchi. Predicting software
future sustainability: A longitudinal perspec-
tive.Information Systems, 49:40–51, 2015.
[6] Likang Yin, Zhuangzhi Chen, Qi Xuan, and
Vladimir Filkov. Sustainability forecasting for
apache incubator projects. InProceedings of
the 29th ACM joint meeting on european soft-
ware engineering conference and symposium on
the foundations of software engineering, pages
1056–1067, 2021.
[7] Wenxin Xiao, Hao He, Weiwei Xu, Yuxia
Zhang, and Minghui Zhou. How early partic-
ipation determines long-term sustained activ-
ity in github projects? InProceedings of the
31st ACM Joint European Software Engineer-
ing Conference and Symposium on the Foun-
dations of Software Engineering, pages 29–41,
2023.
20

[8] Nafiz Imtiaz Khan and Vladimir Filkov. From
models to practice: Enhancing oss project sus-
tainabilitywithevidence-basedadvice. InCom-
panion Proceedings of the 32nd ACM Interna-
tional Conference on the Foundations of Soft-
ware Engineering, pages 457–461, 2024.
[9] Alex Tamkin, Miles Brundage, Jack Clark,
and Deep Ganguli. Understanding the ca-
pabilities, limitations, and societal impact
of large language models.arXiv preprint
arXiv:2102.02503, 2021.
[10] Ggaliwango Marvin, Nakayiza Hellen, Daudi
Jjingo, and Joyce Nakatumba-Nabende.
Prompt engineering in large language mod-
els. InInternational conference on data
intelligence and cognitive informatics, pages
387–402. Springer, 2023.
[11] Zibin Zheng, Kaiwen Ning, Jiachi Chen, Yan-
lin Wang, Wenqing Chen, Lianghong Guo, and
Weicheng Wang. Towards an understanding of
large language models in software engineering
tasks.arXiv preprint arXiv:2308.11396, 2023.
[12] Sanka Rasnayaka, Guanlin Wang, Ridwan
Shariffdeen, and Ganesh Neelakanta Iyer. An
empirical study on usage and perceptions of
llms in a software engineering project.arXiv
preprint arXiv:2401.16186, 2024.
[13] Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang,
Kailong Wang, Li Li, Xiapu Luo, David Lo,
John Grundy, and Haoyu Wang. Large lan-
guage models for software engineering: A
systematic literature review.arXiv preprint
arXiv:2308.10620, 2023.
[14] Zeerak Babar, Nafiz Imtiaz Khan, Muhammad
Hassnain, and Vladimir Filkov. Open-source
LLMs for technical Q&A: Lessons from Stack-
Exchange. InSoftware Engineering for Emerg-
ing Technologies, volume 2725 ofCommuni-
cations in Computer and Information Science,
pages 615–626. Springer, Cham, 2026.
[15] Gerald Gartlehner, Leila Kahwati, Rainer
Hilscher, Ian Thomas, Shannon Kugley,
Karen Crotty, Meera Viswanathan, Barbara
Nussbaumer-Streit, Graham Booth, Nathaniel
Erskine, etal. Dataextractionforevidencesyn-
thesis using a large language model: A proof-
of-concept study.Research Synthesis Methods,
2024.[16] Jingwei Huang, Donghan M Yang, Ruichen
Rong, Kuroush Nezafati, Colin Treager, Zhikai
Chi, Shidan Wang, Xian Cheng, Yujia Guo,
Laura J Klesse, et al. A critical assessment
of using chatgpt for extracting structured data
from clinical notes.npj Digital Medicine,
7(1):106, 2024.
[17] Hiba Ahsan, Denis Jered McInerney, Jisoo
Kim, Christopher Potter, Geoffrey Young, Sil-
vio Amir, and Byron C Wallace. Retrieving
evidence from ehrs with llms: Possibilities and
challenges.arXiv preprint arXiv:2309.04550,
2023.
[18] Luc Patiny and Guillaume Godin. Automatic
extraction of fair data from publications using
llm. 2023.
[19] Falwah Alhamed, Julia Ive, and Lucia Specia.
Using large language models (llms) to extract
evidence from pre-annotated social media data.
InProceedings of the 9th Workshop on Com-
putational Linguistics and Clinical Psychology
(CLPsych 2024), pages 232–237, 2024.
[20] Boris Schauerte. Conference ranks.http://
www.conferenceranks.com/, 2024. Accessed:
2024-09-06.
[21] Computing Research & Education. CORE
ConferencePortal-conferenceranks.https://
portal.core.edu.au/conf-ranks/, 2024. Ac-
cessed: 2024-09-06.
[22] Zixuan Feng, Amreeta Chatterjee, Anita
Sarma, and Iftekhar Ahmed. A case study of
implicit mentoring, its prevalence, and impact
in apache. InProceedings of the 30th ACM
Joint European Software Engineering Confer-
ence and Symposium on the Foundations of
Software Engineering, pages 797–809, 2022.
[23] André Oliveira, Vânia Neves, Alexandre Plas-
tino, Ana Carla Bibiano, Alessandro Garcia,
and Leonardo Murta. Do code refactorings in-
fluence the merge effort? In2023 IEEE/ACM
45th International Conference on Software En-
gineering (ICSE), pages 134–146. IEEE, 2023.
[24] Matthew Renze and Erhan Guven. The ef-
fect of sampling temperature on problem solv-
ing in large language models.arXiv preprint
arXiv:2402.05201, 2024.
21

[25] Toufique Ahmed, Christian Bird, Premkumar
Devanbu, and Saikat Chakraborty. Study-
ing llm performance on closed-and open-source
data.arXiv preprint arXiv:2402.15100, 2024.
[26] Yuvraj Virk, Premkumar Devanbu, and
Toufique Ahmed. Enhancing trust in
llm-generated code summaries with cali-
brated confidence scores.arXiv preprint
arXiv:2404.19318, 2024.
[27] Bo Hu, Yannis Kalfoglou, Harith Alani, David
Dupplaw, Paul Lewis, and Nigel Shadbolt. Se-
mantic metrics. InManaging Knowledge in a
World of Networks: 15th International Confer-
ence, EKAW 2006, Poděbrady, Czech Republic,
October 2-6, 2006. Proceedings 15, pages 166–
181. Springer, 2006.
[28] Gavin Finnie and Zhaohao Sun. Similarity and
metrics in case-based reasoning.International
journal of intelligent systems, 17(3):273–287,
2002.
[29] Toufique Ahmed, Kunal Suresh Pai, Premku-
mar Devanbu, and Earl Barr. Automatic
semantic augmentation of language model
prompts (for code summarization). InProceed-
ings of the IEEE/ACM 46th International Con-
ference on Software Engineering, pages 1–13,
2024.
[30] Ali Al-Kaswan, Toufique Ahmed, Maliheh
Izadi, Anand Ashok Sawant, Premkumar De-
vanbu, andArievanDeursen. Extendingsource
codepre-trainedlanguagemodelstosummarise
decompiled binaries. In2023 IEEE Interna-
tional Conference on Software Analysis, Evolu-
tion and Reengineering (SANER), pages 260–
271. IEEE, 2023.
[31] Kishore Papineni, Salim Roukos, Todd Ward,
and Wei-Jing Zhu. Bleu: a method for
automatic evaluation of machine translation.
InProceedings of the 40th annual meeting of
the Association for Computational Linguistics,
pages 311–318, 2002.
[32] Chin-YewLin. Rouge: Apackageforautomatic
evaluation of summaries. InText summariza-
tion branches out, pages 74–81, 2004.
[33] Tianyi Zhang, Varsha Kishore, Felix Wu, Kil-
ian Q Weinberger, and Yoav Artzi. Bertscore:
Evaluating text generation with bert.arXiv
preprint arXiv:1904.09675, 2019.[34] Google BERT. Bert-base, uncased model
on hugging face.https://huggingface.co/
google-bert/bert-base-uncased, 2024. Ac-
cessed: 2024-09-06.
[35] Satanjeev Banerjee and Alon Lavie. Meteor:
An automatic metric for mt evaluation with
improved correlation with human judgments.
InProceedings of the acl workshop on intrin-
sic and extrinsic evaluation measures for ma-
chine translation and/or summarization, pages
65–72, 2005.
[36] Shashank Mohan Jain. Hugging face. InIn-
troduction to transformers for NLP: With the
hugging face library and models to solve prob-
lems, pages 51–67. Springer, 2022.
[37] Meta. Meta, 2023. Accessed: September 11,
2024.
[38] Google. Google, 2024. Accessed: September
11, 2024.
[39] Microsoft. Microsoft official home page, 2024.
Accessed: September 11, 2024.
[40] x.ai. x.ai: Ai for scheduling and productivity,
2024. Accessed: September 11, 2024.
[41] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen
Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li,
Runyi Hu, Tianwei Zhang, Fei Wu, et al. In-
struction tuning for large language models: A
survey.arXiv preprint arXiv:2308.10792, 2023.
[42] Zahra Abbasiantaeb, Yifei Yuan, Evangelos
Kanoulas, and Mohammad Aliannejadi. Let
the llms talk: Simulating human-to-human
conversational qa via zero-shot llm-to-llm inter-
actions. InProceedings of the 17th ACM Inter-
national Conference on Web Search and Data
Mining, pages 8–17, 2024.
[43] Lei Zhang, Yunshui Li, Jiaming Li, Xi-
aobo Xia, Jiaxi Yang, Run Luo, Minzheng
Wang, Longze Chen, Junhao Liu, and Min
Yang. Hierarchical context pruning: Optimiz-
ing real-world code completion with repository-
level pretrained code llms.arXiv preprint
arXiv:2406.18294, 2024.
[44] Shengnan An, Zexiong Ma, Zeqi Lin, Nan-
ning Zheng, and Jian-Guang Lou. Make your
llm fully utilize the context.arXiv preprint
arXiv:2404.16811, 2024.
22

[45] Inc.MetaPlatforms. Aboutmeta-companyin-
formation.https://about.meta.com/, 2024.
Accessed: 2024-09-06.
[46] Hugo Touvron, Thibaut Lavril, Gautier Izac-
ard, Xavier Martinet, Marie-Anne Lachaux,
Timothée Lacroix, Baptiste Rozière, Naman
Goyal, EricHambro, FaisalAzhar, etal. Llama:
Open and efficient foundation language models.
arXiv preprint arXiv:2302.13971, 2023.
[47] NurtureAI. Meta llama 3 8b in-
struct 32k model on hugging face.
https://huggingface.co/NurtureAI/
Meta-Llama-3-8B-Instruct-32k, 2024.
Accessed: 2024-09-06.
[48] Nurture AI. Nurture ai - artificial intelligence
solutions.https://nurture.ai/, 2024. Ac-
cessed: 2024-09-06.
[49] Mistral AI. Mistral ai - next-generation open
models.https://mistral.ai/, 2024. Ac-
cessed: 2024-09-06.
[50] Albert Q Jiang, Alexandre Sablayrolles, An-
toine Roux, Arthur Mensch, Blanche Savary,
ChrisBamford,DevendraSinghChaplot,Diego
de las Casas, Emma Bou Hanna, Florian Bres-
sand, et al. Mixtral of experts.arXiv preprint
arXiv:2401.04088, 2024.
[51] Amazon Web Services (AWS). Amazon web
services - cloud computing services.https://
aws.amazon.com/, 2024. Accessed: 2024-09-06.
[52] Amazon MistralLite. Mistrallite model on hug-
ging face.https://huggingface.co/amazon/
MistralLite, 2024. Accessed: 2024-09-06.
[53] Mistral AI. Discussion on mistral-7b-
v0.1 model.https://huggingface.co/
mistralai/Mistral-7B-v0.1/discussions/
4, 2024. Accessed: 2024-09-06.
[54] Mistral AI. Mistral-nemo instruct 2407.
https://huggingface.co/mistralai/
Mistral-Nemo-Instruct-2407, 2024. Ac-
cessed: 2024-09-08.
[55] Mistral AI. Introducing mistral-nemo: A
cutting-edge instruct model.https://
mistral.ai/news/mistral-nemo/, 2024. Ac-
cessed: 2024-09-08.
[56] Qi Cheng, Liqiong Chen, Zhixing Hu, Juan
Tang, Qiang Xu, and Binbin Ning. A novel
prompting method for few-shot ner via llms.Natural Language Processing Journal, page
100099, 2024.
[57] Ggaliwango Marvin, Nakayiza Hellen, Daudi
Jjingo, and Joyce Nakatumba-Nabende.
Prompt engineering in large language mod-
els. InInternational conference on data
intelligence and cognitive informatics, pages
387–402. Springer, 2023.
[58] Sondos Mahmoud Bsharat, Aidar Myrzakhan,
and Zhiqiang Shen. Principled instructions are
all you need for questioning llama-1/2, gpt-
3.5/4.arXiv preprint arXiv:2312.16171, 2023.
[59] Cheng Li, Jindong Wang, Yixuan Zhang, Kai-
jie Zhu, Wenxin Hou, Jianxun Lian, Fang
Luo, Qiang Yang, and Xing Xie. Large
language models understand and can be en-
hanced by emotional stimuli.arXiv preprint
arXiv:2307.11760, 2023.
[60] Sander Schulhoff, Michael Ilie, Nishant
Balepur, Konstantine Kahadze, Amanda Liu,
Chenglei Si, Yinheng Li, Aayush Gupta, Hyo-
Jung Han, Sevien Schulhoff, et al. The prompt
report: A systematic survey of prompting
techniques.arXiv preprint arXiv:2406.06608,
2024.
[61] Satyavrat Gaur, Anjali Dagar, Aanchal Pu-
nia, and Pushpendra Kumar. A brief study of
prompting techniques for reasoning tasks. In
NIELIT’s International Conference on Com-
munication, Electronics and Digital Technolo-
gies, pages 147–159. Springer, 2024.
[62] Yinheng Li. A practical survey on zero-shot
prompt design for in-context learning.arXiv
preprint arXiv:2309.13205, 2023.
[63] Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompt-
ing elicits reasoning in large language models.
Advances in neural information processing sys-
tems, 35:24824–24837, 2022.
[64] Alexandre Piché, Aristides Milios, Dzmitry
Bahdanau, and Christopher Pal. Self-
evaluation and self-prompting to improve the
reliability of llms. InICLR 2024 Workshop on
Secure and Trustworthy Large Language Mod-
els, 2024.
[65] Jiefeng Chen, Jinsung Yoon, Sayna Ebrahimi,
Sercan O Arik, Tomas Pfister, and Somesh Jha.
23

Adaptation with self-evaluation to improve se-
lective prediction in llms.arXiv preprint
arXiv:2310.11689, 2023.
[66] Mudit Verma, Siddhant Bhambri, and
Subbarao Kambhampati. On the brittle
foundations of react prompting for agen-
tic large language models.arXiv preprint
arXiv:2405.13966, 2024.
[67] Maciej P Polak and Dane Morgan. Extract-
ing accurate materials data from research pa-
pers with conversational language models and
prompt engineering.Nature Communications,
15(1):1569, 2024.
[68] Nafiz Imtiaz Khan and Vladimir Filkov. Evi-
dencebot: A privacy-preserving, customizable
rag-based tool for enhancing large language
model interactions. InProceedings of the 33rd
ACM International Conference on the Foun-
dations of Software Engineering, pages 1188–
1192, 2025.
[69] Gabrijela Perković, Antun Drobnjak, and Ivica
Botički. Hallucinations in llms: Under-
standing and addressing challenges. In2024
47th MIPRO ICT and Electronics Convention
(MIPRO), pages 2084–2088. IEEE, 2024.
[70] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangx-
iang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, and Haofen Wang. Retrieval-augmented
generation for large language models: A sur-
vey.arXiv preprint arXiv:2312.10997, 2023.
[71] Jiawei Chen, Hongyu Lin, Xianpei Han, and
Le Sun. Benchmarking large language models
in retrieval-augmented generation. InProceed-
ings of the AAAI Conference on Artificial In-
telligence, volume 38, pages 17754–17762, 2024.
[72] PyPDF2 Project. Pypdf2 - a pure-python
pdf library.https://pypi.org/project/
PyPDF2/, 2024. Accessed: 2024-09-06.
[73] HuggingFace. Instructor-large, 2024. Accessed:
2024-09-08.
[74] Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong
Wang, Yushi Hu, Mari Ostendorf, Wen-tau
Yih, Noah A Smith, Luke Zettlemoyer, and
Tao Yu. One embedder, any task: Instruction-
finetuned text embeddings.arXiv preprint
arXiv:2212.09741, 2022.
[75] LangChain. Langchain - building applica-
tions with llms.https://www.langchain.
com/, 2024. Accessed: 2024-09-06.[76] Christos H Papadimitriou, Hisao Tamaki,
Prabhakar Raghavan, and Santosh Vempala.
Latent semantic indexing: A probabilistic
analysis. InProceedings of the seventeenth
ACM SIGACT-SIGMOD-SIGART symposium
on Principles of database systems, pages 159–
168, 1998.
[77] Chroma. Chroma, 2024. Accessed: 2024-09-08.
[78] Xingrui Xie, Han Liu, Wenzhe Hou, and
Hongbin Huang. A brief survey of vector
databases. In2023 9th International Confer-
ence on Big Data and Information Analytics
(BigDIA), pages 364–371. IEEE, 2023.
[79] Ron Artstein. Inter-annotator agreement.
Handbook of linguistic annotation, pages 297–
313, 2017.
[80] Apache Incubator. Apache incubator.https:
//incubator.apache.org/, 2024. Accessed:
2024-09-09.
[81] Apache Software Foundaion. Apache software
foundaion.https://www.apache.org/, 2024.
Accessed: 2024-08-01.
[82] Anirudh Ramchandran, Likang Yin, and
Vladimir Filkov. Exploring apache incubator
project trajectories with apex. InProceedings
of the 19th international conference on mining
software repositories, pages 333–337, 2022.
[83] Apache Software Foundation. Apache mailing
lists.https://www.apache.org/foundation/
mailinglists. Accessed: September 12, 2024.
[84] Anirudh Ramchandran, Likang Yin, and
Vladimir Filkov. Exploring apache incubator
project trajectories with apex. InProceedings
of the 19th international conference on mining
software repositories, pages 333–337, 2022.
[85] Zihe H Zhao. The distribution and dis-
engagement of women contributors in open-
source: 2008–2021. In2023 IEEE/ACM
45th International Conference on Software
Engineering: Companion Proceedings (ICSE-
Companion), pages 305–307. IEEE, 2023.
[86] Courtney Miller, Christian Kästner, and Bog-
dan Vasilescu. “we feel like we’re winging it:”
a study on navigating open-source dependency
abandonment. InProceedings of the 31st ACM
Joint European Software Engineering Confer-
ence and Symposium on the Foundations of
Software Engineering, pages 1281–1293, 2023.
24

[87] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng,
Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing
Xu, and Zhifang Sui. A survey on in-context
learning.arXiv preprint arXiv:2301.00234,
2022.
[88] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue,
and Wenhu Chen. Long-context llms struggle
with long in-context learning.arXiv preprint
arXiv:2404.02060, 2024.
[89] X AI. Grok os blog post.https://x.ai/blog/
grok-os, 2024. Accessed: 2024-09-06.
[90] Mistral AI. Mixtral-8x22b-v0.1 model on
hugging face.https://huggingface.co/
mistralai/Mixtral-8x22B-v0.1, 2024. Ac-
cessed: 2024-09-06.
[91] Databricks. Dbrx-base model on hugging
face.https://huggingface.co/databricks/
dbrx-base, 2024. Accessed: 2024-09-06.
[92] Jieyi Long. Large language model guided tree-
of-thought.arXiv preprint arXiv:2305.08291,
2023.
[93] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak
Shafran, Tom Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate
problem solving with large language models.
Advances in Neural Information Processing
Systems, 36, 2024.
[94] Angeliki Lazaridou, Elena Gribovskaya,
Wojciech Stokowiec, and Nikolai Grigorev.
Internet-augmented language models through
few-shot prompting for open-domain question
answering.arXiv preprint arXiv:2203.05115,
2022.
[95] Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang,
Kailong Wang, Li Li, Xiapu Luo, David Lo,
John Grundy, and Haoyu Wang. Large lan-
guage models for software engineering: A
systematic literature review.arXiv preprint
arXiv:2308.10620, 2023.
[96] Guilherme Avelino, Eleni Constantinou,
Marco Tulio Valente, and Alexander Sere-
brenik. On the abandonment and survival
of open source projects: An empirical inves-
tigation. In2019 ACM/IEEE International
Symposium on Empirical Software Engineering
and Measurement (ESEM), pages 1–12. IEEE,
2019.[97] Felipe Fronchetti, Igor Wiese, Gustavo Pinto,
and Igor Steinmacher. What attracts newcom-
ers to onboard on OSS projects? TL;DR: Pop-
ularity. InOpen Source Systems: 15th IFIP
WG 2.13 International Conference, OSS 2019,
pages 91–103, Montreal, QC, Canada, 2019.
Springer.
[98] Gerardo Matturro, Karina Barrella, and Pa-
tricia Benitez. Difficulties of newcomers join-
ing software projects already in execution. In
2017 International Conference on Computa-
tional Science and Computational Intelligence
(CSCI), pages 993–998. IEEE, 2017.
[99] Bin Lin, Gregorio Robles, and Alexander Sere-
brenik. Developer turnover in global, industrial
open source projects: Insights from applying
survival analysis. In2017 IEEE 12th Interna-
tional Conference on Global Software Engineer-
ing (ICGSE), pages 66–75. IEEE, 2017.
[100] FabianFagerholm,AlejandroS.Guinea,Jürgen
Münch, and Jay Borenstein. The role of men-
toring and project characteristics for onboard-
ing in open source software projects. InPro-
ceedings of the 8th ACM/IEEE International
Symposium on Empirical Software Engineering
and Measurement, pages 1–10. ACM, 2014.
[101] Leonard Przybilla, Maximilian Rahn, Manuel
Wiesche, and Helmut Krcmar. The more the
merrier? The effect of size of core team sub-
groups on success of open source projects.
arXiv preprint, 2019.
[102] Gillian J. Greene and Bernd Fischer. CV-
Explorer: Identifying candidate developers by
mining and exploring their open source contri-
butions. InProceedings of the 31st IEEE/ACM
International Conference on Automated Soft-
ware Engineering, pages 804–809. ACM, 2016.
[103] Emad Shihab, Zhen Ming Jiang, and Ahmed E.
Hassan. Studying the use of developer IRC
meetingsinopensourceprojects. In2009 IEEE
International Conference on Software Mainte-
nance, pages 147–156. IEEE, 2009.
[104] Gustavo Pinto, Igor Steinmacher, and
Marco Aurélio Gerosa. More common
than you think: An in-depth study of casual
contributors. In2016 IEEE 23rd International
Conference on Software Analysis, Evolution,
and Reengineering (SANER), volume 1, pages
112–123. IEEE, 2016.
25

[105] Gregorio Robles, Jesus M. Gonzalez-Barahona,
and Israel Herraiz. Evolution of the core team
of developers in libre software projects. In2009
6th IEEE International Working Conference on
Mining Software Repositories, pages 167–170.
IEEE, 2009.
26

A LLM Prompts
This section outlines the various prompting techniques employed in the study. Here,TitleOfTheArticleis a
string variable updated based on the actual title of the article. The highlighted portion (italic) in the prompt
indicates the emotional stimuli, which was added at the bottom of each prompt.
A.1 Zero Shot
You have been entrusted with the complete article titledTitleOfTheArticle. Your objective is to metic-
ulously extract actionable recommendations from this article. These recommendations should be aimed
at enhancing the sustainability of open-source software projects.
Your attention to detail in this task is crucial, as its successful completion holds significant importance
for my career advancement.
A.2 Chain-of-Thought
You have been given the full text of the article titledTitleOfTheArticle. Your task is to extract action-
able recommendations from the article. An actionable recommendation is a practical, evidence-based
suggestion that provides specific, clear steps or instructions which, when implemented, are expected to
produce tangible and positive results. These recommendations should be practical, making them easy
to implement in real-world scenarios, and evidence-based, supported by data or findings from the arti-
cle. They must be specific, providing clearly defined steps or instructions, and result-oriented, aimed at
producing tangible and positive outcomes. These recommendations should offer concrete guidance that
can be directly put into practice to achieve desired results. Adopting these actionable recommendations
can help make open-source software projects more sustainable.
Let’s break down the problem into the following steps:
Step 1:Carefully read each sentence of the article and identify recommendations or suggestions.
Look for imperative sentences or phrases that give commands, make requests, or offer instructions to
direct or persuade someone to perform a specific action.
Step 2:Extract the practical suggestions from the identified sentences and present them as concise,
unambiguous statements in a list format.
Step 3:Review the list of recommendations and remove any duplicate items to ensure clarity and
avoid redundancy.
Step 4:For each recommendation in the final list, assign a confidence level on a scale of 0 to 1,
indicating how confident you are in the effectiveness of the recommendation for making open-source
projects sustainable. Provide a brief explanation for your confidence level.
Your attention to detail in this task is crucial, as its successful completion holds significant importance
for my career advancement.
A.3 Reason + Action
Thought 1:I have been given the article titledTitleOfTheArticle. My task is to extract actionable
recommendations from the article. An actionable recommendation is a practical, evidence-based sug-
gestion that provides specific, clear steps or instructions which, when implemented, are expected to
produce tangible and positive results. These recommendations should be practical, making them easy
to implement in real-world scenarios, and evidence-based, supported by data or findings from the arti-
cle. They must be specific, providing clearly defined steps or instructions, and result-oriented, aimed at
producing tangible and positive outcomes. These recommendations should offer concrete guidance that
can be directly put into practice to achieve desired results. Adopting these actionable recommendations
can help make open-source software projects more sustainable.
Action 1:Carefully read each sentence of the article, focusing on identifying imperative sentences
or phrases that give commands, make requests, or offer instructions to direct or persuade someone to
perform a specific action.
27

Observation 1:A list of potential actionable recommendations has been identified from the article.
Thought 2:I need to extract the practical suggestions from the identified sentences and present
them as concise, unambiguous statements.
Action 2:Convert the identified sentences into clear, actionable statements and compile them into
a list.
Observation 2:A list of actionable recommendations has been compiled from the article.
Thought3:Toensureclarityandavoidredundancy,Ishouldreviewthelistandremoveanyduplicate
items.
Action 3:Review the list of recommendations and remove any duplicate or highly similar items.
Observation 3:The list of recommendations has been refined, removing duplicates and ensuring
clarity.
Thought 4:I need to assess the confidence level for each recommendation to indicate how effective
it might be in making open-source projects sustainable.
Action 4:For each recommendation in the final list, assign a confidence level on a scale of 0 to
1, indicating how confident I am in the effectiveness of the recommendation for making open-source
projects sustainable. Provide a brief explanation for each assigned confidence level.
Observation 4:Confidence scores have been assigned to each recommendation in the final list, along
with brief explanations.
Your attention to detail in this task is crucial, as its successful completion holds significant importance
for my career advancement.
B Definition of ReACT Categories
B.1 New Contributor Onboarding and Involvement
Definition:This category focuses on ensuring that new contributors can easily join, understand, and
meaningfully contribute to the project.
Criteria for Assignment:(a) Actionable facilitates the integration of new contributors by providing
mentorship, onboarding materials, or simplifying the contribution process; (b) Actionable relates to improv-
ing project documentation or offering better support mechanisms for first-time contributors; (c) Actionable
helps build a welcoming, inclusive, and open culture for new participants.
B.2 Code Standards and Maintainability
Definition:This category deals with ensuring that the codebase adheres to established standards, making
it easier to maintain and scale. It includes efforts to ensure code readability, modularity, and compliance
with coding best practices.
Criteria for Assignment:(a) Actionable relates to improving the quality, readability, or structure of the
codebase; (b)Actionableincludeseffortstoenforcecodingguidelines, refactorcodeforbettermaintainability,
or reduce technical debt; (c) Actionable includes the use of linters, formatters, or static code analysis tools.
B.3 Automated Testing and Quality Assurance
Definition:This category focuses on ensuring the project’s robustness and reliability through automated
testing practices, such as unit, integration, and end-to-end tests. It also includes broader quality assurance
activities.
Criteria for Assignment:(a) Actionable involves the implementation or improvement of automated
testing frameworks and testing strategies; (b) Actionable includes practices that ensure the detection of bugs
early in the development cycle and ensure high-quality releases.
B.4 Community Collaboration and Engagement
Definition:This category deals with activities that foster collaboration, communication, and engagement
within the OSS community. It includes practices for keeping the community active and involved.
28

Criteria for Assignment:(a) Actionable aims to improve communication between contributors, main-
tainers,andusers; (b)Actionableinvolvesorganizingcommunity-drivenevents,discussions,orcollaborations,
as well as platforms to enhance transparency and teamwork; (c) Actionable relates to tools and processes
for better community governance and decision-making.
B.5 Documentation Practices
Definition:This category focuses on ensuring that the project’s documentation is thorough, up-to-date,
and easily accessible. Documentation practices are crucial for both current and future contributors.
Criteria for Assignment:(a) Actionable focuses on improving the quality, clarity, or accessibility of
project documentation, such as user guides, API references, or contributor guides; (b) Actionable includes
practices for keeping documentation synchronized with the codebase and ensuring it meets the needs of
different stakeholders; (c) Actionable involves translation efforts or making documentation more accessible
to non-expert audiences.
B.6 Project Management and Governance
Definition:This category deals with the governance structure and project management practices that keep
the project organized, transparent, and sustainable over the long term.
Criteria for Assignment:(a) Actionable enhances the governance model, clarifies roles and responsi-
bilities, or improves the decision-making process; (b) Actionable involves defining or refining processes for
issue triaging, release management, or conflict resolution; (c) Actionable includes efforts to improve the
transparency of project goals, progress, and decision-making.
B.7 Security Best Practices and Legal Compliance
Definition:This category addresses efforts to secure the project and ensure compliance with relevant legal
standards, such as licenses, data privacy laws, and security protocols.
Criteria for Assignment:(a) Actionable focuses on improving the security posture of the project by
following best practices, addressing vulnerabilities, or conducting audits; (b) Actionable involves ensuring
compliance with open-source licenses, setting up contributor license agreements (CLAs), or aligning with
dataprivacyregulations; (c)Actionableincludessecuritymeasuressuchasdependencymanagement, security
audits, and secure coding practices.
B.8 CI/CD and DevOps Automation
Definition:This category deals with continuous integration and continuous deployment (CI/CD) processes
that automate building, testing, and deployment pipelines. It also includes broader DevOps automation
tasks.
CriteriaforAssignment:(a)ActionableinvolvesthesetuporenhancementofCI/CDpipelinestoensure
faster, reliable, and automated releases; (b) Actionable relates to automating infrastructure provisioning,
containerization, or deployment to cloud environments; (c) Actionable includes the integration of DevOps
practices that ensure smooth, automated, and repeatable processes for software development, testing, and
deployment.
C Server Configuration
We conducted all experiments on a dedicated local server with the configuration presented in Table 3.
D ReACT Gold Annotations
This section presents the manually annotated ReACTs extracted from 10 articles used for evaluating LLM
performance.
29

Table 3: Server Configuration
Component Specification
CPU AMD Ryzen 9 5900X (12-core, 24-thread)
RAM 64 GB DDR4
GPU NVIDIA RTX 3090 (24 GB VRAM)
Storage 2 TB NVMe SSD
OS Ubuntu 22.04 LTS
D.1 Article 1
“On the abandonment and survival of open source projects: An empirical investigation”by Avelino et al. [96].
Derived Actionables:
1. Project maintainers should strive to increase the number of Truck Factor (TF) developers to reduce the
risk of project abandonment.
2. Projects should seek alternative backing, such as company-based support, to prevent or reduce the
chances of TF developer detachments (TFDDs).
3. Open source communities should foster a friendly and active environment to attract and retain new
contributors.
4. Project owners should ensure that their repositories are easily accessible.
5. Project maintainers should implement and adhere to well-known software engineering principles and
practices to make it easier for new developers to contribute.
6. Open source projects should consider using popular programming languages to attract a wider pool of
potential contributors.
7. Project maintainers should be aware of and mitigate common barriers faced by new contributors, par-
ticularly the lack of time and experience.
8. Open source communities should promote and share successful cases of projects overcoming TFDDs to
motivate developers to actively contribute to projects at risk.
9. ProjectmaintainersshouldregularlyassesstheriskofabandonmentbyTFdevelopersandtakeproactive
measures to ensure project continuity.
10. Projects should implement continuous integration practices to facilitate contributions from new devel-
opers.
11. Project maintainers should strive to keep the codebase clean and well-designed to make it easier for new
contributors to understand and work with the project.
12. Projects should consider implementing a code review process to maintain code quality.
D.2 Article 2
“What attracts newcomers to onboard on OSS projects? TL;DR: Popularity”by Fronchetti et al. [97].
Derived Actionables:
1. Reduce the time taken to review and merge pull requests.
2. Use multiple programming languages in the project.
3. Maintain the project over a longer period of time to increase its age.
30

4. Have a larger number of integrators (contributors with rights to merge pull requests).
5. Choose a popular main programming language for the project.
6. Select an appropriate software application domain for the project.
D.3 Article 3
“Difficulties of newcomers joining software projects already in execution”by Matturro et al. [98].
Derived Actionables:
1. Assign an experienced team member to coach and guide the newcomer.
2. Follow up with newcomers on both success and failure.
3. Hold training sessions for newcomers.
4. Maintain concise, updated, accessible documentation.
5. Grant the newcomer freedom: Encourage and allow them to express opinions, propose changes, and
share personal viewpoints to foster a comfortable environment.
6. Establish a personalized integration plan: Outline a gradual assignment of tasks and responsibilities for
seamless incorporation.
D.4 Article 4
“Developer turnover in global, industrial open source projects: Insights from applying survival analysis”by
Lin et al. [99].
Derived Actionables:
1. Provide better onboarding assistance for newcomers to help them become more engaged in the project.
2. Ensure developers maintain both code developed by others and their own code.
3. Assign more code maintenance tasks to developers who primarily write new code.
4. Give coding tasks to developers who mainly work on documentation to increase their chances of staying
in the project.
5. Encourage early contribution to the project, as developers who join earlier tend to stay longer.
6. Implement strategies to manage growing code complexity, as it can form an obstacle for new developers’
contributions.
D.5 Article 5
“The role of mentoring and project characteristics for onboarding in open source software projects”by Fager-
holm et al. [100].
Derived Actionables:
1. Provide onboarding support and help newcomers to make their first contribution.
31

D.6 Article 6
“The more the merrier? The effect of size of core team subgroups on success of open source projects”by
Przybilla et al. [101].
Derived Actionables:
1. Projects should implement mechanisms to motivate and support long-term, consistent participation
from key developers.
2. Encourage experienced developers to guide and support newcomers, particularly through issue-related
activities like commenting on and resolving issues.
3. Implement strategies to increase the visibility and recognition of high-reputation contributors.
4. Create opportunities for core members to focus on issue-related activities.
5. Be cautious about having too many extensively contributing core members. Balance the core team to
avoid creating the impression of a closed circle that might deter new contributors.
6. Address common barriers faced by newcomers, such as identifying where to start contributing.
7. Analyze and potentially steer the configuration of subgroups within the project based on factors like
reputation, issue focus, contribution extent, and persistence.
8. Consider implementing formal collaborator status or other metrics to make reputation more visible and
meaningful to outsiders.
9. Focus on creating a supportive environment that aligns with core OSS values, such as learning oppor-
tunities and the ability to make meaningful contributions.
10. Implement mechanisms to track and analyze the content and quality of contributions.
11. Develop strategies to signal ongoing project activity and future maintenance to attract and retain
contributors.
D.7 Article 7
“CVExplorer: Identifying candidate developers by mining and exploring their open source contributions”by
Greene & Fischer [102].
Derived Actionables:No actionable recommendation could be derived from the article.
D.8 Article 8
“Studying the use of developer IRC meetings in open source projects”by Shihab et al. [103].
Derived Actionables:No actionable recommendation could be derived from the article.
D.9 Article 9
“More common than you think: An in-depth study of casual contributors”by Pinto et al. [104].
Derived Actionables:No actionable recommendation could be derived from the article.
D.10 Article 10
“Evolution of the core team of developers in libre software projects”by Robles et al. [105].
Derived Actionables:No actionable recommendation could be derived from the article.
32