# Exploring LLMs and RAG for Plausible and Explainable Material Prediction of Vehicle Components

**Authors**: Frederik Wagner, Annerose Eichel, Sabine Schulte im Walde

**Published**: 2026-09-16 10:28:28

**PDF URL**: [https://arxiv.org/pdf/2609.18437v1](https://arxiv.org/pdf/2609.18437v1)

## Abstract
In this work, we explore whether LLMs can accurately predict and explain plausible materials for vehicle components such as brake discs or fuel injectors without requiring extensive fine-tuning. We test and evaluate three approaches: a standard generative LLM baseline, a single-pass Retrieval-Augmented Generation (RAG) approach, and an iterative Chain-of-Verification (CoVe) variant. For retrieval, we rely on publicly available data using a domain-filtered Wikipedia corpus. Since no gold standard exists for this task, we develop a custom web-based annotation tool supporting crucial functions for structured domain expert evaluation. LLM-based generation substantially outperforms prior work, which is not further surpassed by the tested RAG approaches. Our results surface remaining challenges for RAG-based systems: hyperparameter optimization, the availability of high-quality, legally accessible domain corpora, and expert evaluation study design.

## Full Text


<!-- PDF content starts -->

Exploring LLMs and RAG for Plausible and Explainable Material
Prediction of Vehicle Components
Frederik Wagner and Annerose Eichel and Sabine Schulte im Walde
University of Stuttgart, Institute for Natural Language Processing, Germany
{frederik.wagner,annerose.eichel,schulte}@ims.uni-stuttgart.de
Abstract
In this work, we explore whether LLMs can
accurately predict and explain plausible ma-
terials for vehicle components such asbrake
discsorfuel injectorswithout requiring exten-
sive fine-tuning. We test and evaluate three ap-
proaches: a standard generative LLM baseline,
a single-pass Retrieval-Augmented Generation
(RAG) approach, and an iterative Chain-of-
Verification (CoVe) variant. For retrieval, we
rely on publicly available data using a domain-
filtered Wikipedia corpus. Since no gold stan-
dard exists for this task, we develop a custom
web-based annotation tool supporting crucial
functions for structured domain expert eval-
uation. LLM-based generation substantially
outperforms prior work, which is not further
surpassed by the tested RAG approaches. Our
results surface remaining challenges for RAG-
based systems: hyperparameter optimization,
the availability of high-quality, legally acces-
sible domain corpora, and expert evaluation
study design.
1 Introduction
LLMs have become part of everyday life, and are
increasingly used in professional settings. This
includes high-stakes domains with limited avail-
able evaluation of their reliability in corresponding
real-life deployment (Merenda et al., 2026; Del-
mas et al., 2025; Bakos et al., 2025). The vehi-
cle repair domain is one such real-life scenario,
which additionally raises critical concerns in terms
of physical safety (e.g., improperly executed re-
pairs may impact the reliability of a vehicle). Our
work addresses this gap, focusing on the vehicle
repair setting with a documented need for aid in
information retrieval (Eichel et al., 2023). Consider
a vehicle repair shop where an AI assistant is used
to support a mechanic, for example, by guiding
them through a repair process. To do this, models
underlying an AI assistant need to combine world
knowledge with domain-specific information. Forexample, most people would know thatfabricis
not a plausible material for abrake discbased on
their general understanding of the world. But with-
out domain-specific background knowledge in the
automotive domain, a lay person is likely not able
to precisely predict the actual materials that a brake
disc can be made out of, such asgray cast ironand
stainless steel. In the current study, we use the task
ofpredicting plausible materials for vehicle com-
ponentsto evaluate whether and to which extent
LLMs acquire accurate domain-specific knowledge.
If a model performs well on this task, it might be
able to guide a mechanic through a repair process.
In this study, we use LLMs that are trained
on vast amounts of text which allows for learn-
ing world knowledge though distributional patterns
(Sun et al., 2024; Holtermann et al., 2025). In our
baseline experiment, we test whether anoff-the-
shelf LLM-based approach can predict plausible
materials for vehicle components in the vehicle
repair domain without extensive pre-training or
fine-tuning. Our results clearly outperform prior
work (Schlipf, 2022; Eichel et al., 2023), demon-
strating that plausible materials for a component
may be elicited from LLMs with a fair degree of
confidence. However, it remains unclearwhyspe-
cific material candidates are predicted. The second
part of this work thus focuses on the prediction
of plausible materials includingan explanation
regarding the potential use of the material in a
vehicle component.
In this context, we test whetherretrieval-
augmented generation (RAG)(Lewis et al., 2020;
Gao et al., 2024) leads to improvements over stan-
dard LLM-based methods. Here, our goal is to
bypass standard fine-tuning methods for domain-
specific adaptation, which is costly in terms of hard-
ware, energy, and time with ever-growing model
sizes. RAG takes user input, applies standard infor-
mation retrieval techniques to find relevant textual
information from an extensive database, and lever-
arXiv:2609.18437v1  [cs.IR]  16 Sep 2026

ages the retrieved snippets to create a prompt for
an LLM. This way, domain-specific information
is encoded in the prompt and does not need to be
learned through fine-tuning. However, LLM-based
retrieval may be strongly affected by hallucinations,
i.e., semantically valid text providing a plausible
interpretation which however contains misleading
or factually incorrect information (Zhang et al.,
2025). We thus assess whether RAG is useful in
detecting hallucinated materials or explanations. In
more detail, we implement and evaluate three ap-
proaches include a no-RAG method using LLMs,
a method using standard RAG (Karpukhin et al.,
2020), and an iterative RAG approach using Chain-
of-Verification (CoVe) (Ji et al., 2023; Press et al.,
2023; Dhuliawala et al., 2024). We compare an
open-source and a proprietary LLM to ensure that
results are not model-specific. To assess system
performance, we develop a custom full-stack an-
notation tool that supports crucial functions for
evaluating open-ended natural language generation
through domain experts.
Results highlight significant but non-substantial
differences between the tested RAG methods and
LLMs. For a subset of tasks, we observe consider-
able disagreement among human expert annotators.
We thus zoom into potential sources for the ob-
served disparities, and discuss explanations as well
as future directions.
Our contributions are summarized as follows:
•We show that for the task of predicting plausi-
ble material candidates for vehicle components,
anoff-the-shelf LLM-based approach clearly
outperforms previous work without extensive
training or fine-tuning.
•We investigate explanations forwhya specific
material candidate is predicted, and reveal that
RAG-based approaches do not lead to substan-
tial improvements over standard LLM-based
methodsin the investigated setup.
•We develop acustom full-stack annotation tool
for expert evaluationthat supports web-based
access without installation, secure token-based
authentication, automatic progress saving, text-
span highlighting, star ratings, and drag-and-drop
ranking focusing on survey-specific re-usability.
2 Related Work
LLMs and RAG for Materials ScienceWhile
NLP methods have been leveraged for highly spe-
cialized domains, LLMs offer promising possibili-ties for automating material science tasks such as
information retrieval, knowledge organization, and
innovation derivation and generation (Olivetti et al.,
2020). Recent work uses LLMs at various stages,
including pre-training (Kim et al., 2024; Oh et al.,
2025), dataset creation and verification (Song et al.,
2023), and modeling (Cheung et al., 2024; Jansen
et al., 2025). However, research on predicting plau-
sible material candidates for the (vehicle) repair
domain remains underexplored, in particular, when
taking into account the additional task of explain-
ing the reason for predicting a specific material.
Our work addresses this gap: we apply and analyze
the suitability of LLMs used in zero- and few-shot
settings for the task at hand.
In addition to the evaluation of LLMs, we inves-
tigate the integration of RAG combining natural
language generation with a retrieval step. Gao et al.
(2024) provide a comprehensive survey, identify-
ing two primary retrieval approaches:sparse re-
trieval(e.g., BM25 (Robertson and Walker, 1994),
based on word overlap) anddense retrieval(e.g.,
DPR (Karpukhin et al., 2020), using neural embed-
dings).Hybrid retrievalcombines both retrieval
methods (Gao et al., 2021). RAG approaches for
material science tasks such as knowledge extraction
and analysis include, among other work, (C et al.,
2025) who use embeddings fine-tuned for material
science (MatSci-BERT) (Gupta et al., 2022) and
a dense retriever. Other researchers focus on spe-
cific materials in a specialized domain, e.g. nano-
structured materials (Krotkov et al., 2025) lever-
aging multilingual embeddings alongside a dense
retrieval module. Our work also focuses on a partic-
ular domain and task, however, we employ hybrid
retrieval, using Karpukhin et al. (2020)’s embed-
dings for DPR which we find to substantially out-
perform MatSci-BERT embeddings.
Semantic Plausibility and HallucinationA
known limitation of LLMs ishallucination: the
tendency to generate semantically valid text that
has a plausible interpretation, but contains mis-
leading or factually incorrect information.1In re-
cent years, a range of advances to detect and re-
duce hallucinations have been presented. For ex-
ample, Ji et al. (2023) propose a technique called
self-reflection, that iteratively improves the LLM-
generated answer to a question in the medical field.
Press et al. (2023) show that LLMs struggle with
compositional reasoning problems and demonstrate
1We refer to Zhang et al. (2025) for a detailed overview.

the benefit of splitting up complex questions. Dhu-
liawala et al. (2024) combine self-reflection and
question-splitting approaches by proposing a tech-
nique calledChain of Verification(CoVe). More
specifically, the model first generates an answer to
the initial question. Then, the model generates veri-
fication questions which are answered individually
in the next step. Finally, the initial answer is refined
using the generated verification question-answer
pairs. While CoVe has been successfully coupled
with RAG to improve performance of tasks such
as computer-assisted design (CAD) in engineering
(Joseph, 2025), we apply the technique to the task
of plausible material prediction in the vehicle repair
domain, including domain expert evaluation.
3 Data
3.1 Vehicle Components
Vehicle Component DatasetAs targets for our
components, we rely on a set of 7,069 unique com-
ponent names curated by experts from the vehi-
cle repair domain.2A component name may de-
note a tangible physical component such ascool-
ing blower, as well as intangible functional and
software components such asABS warning lamp
functionandroad test. The dataset comprises 155
single-word components and 6,914 multiword com-
ponents with up to eight constituents.
Evaluation DatasetTo evaluate predicted ma-
terials in a human annotation study, we create an
evaluation set comprising 100 components which
focuses on physical (brake disc,spark plug) vs.
software components (parking assistant,catalytic
converter monitoring). We discard physical com-
ponents that act as a system and consist of several
sub-components, e.g.,clutchconsisting the sub-
componentsclutch pedal,clutch disk, andmaster
cylinder. To achieve maximum component vari-
ety, we discard components that are highly similar
to each other but used in different places in a ve-
hicle, e.g.pressure control module high-pressure
solenoid valveandinlet camshaft control solenoid
valve. Here, a high-pressure solenoid valve might
need to be constructed from sturdier materials than
a solenoid valve dealing with lower pressure but
possibly more time-sensitive tasks. Since 98% of
components are MWEs, we mirror the constituent
distribution of the full dataset in the evaluation set.
2The dataset is provided by a company disclosed upon
acceptance.For this, we first draw a random sample of 200
component from the full dataset, manually discard
all components not fulfilling the above-described
criteria, and select components until a set of 100 is
reached from the remaining set.
3.2 Retrieval Data Sources
The retrieval corpus is built from a Wikipedia dump.
We develop a custom extraction tool in Rust (cho-
sen for speed over the standard gensim library, re-
ducing extraction time from eight hours to under
two) to pull full articles meeting two constraints:
(1) articles containing at least two multi-word ve-
hicle components, and (2) articles co-mentioning
”material/materials” and automotive-domain terms
at least three times each. This results in 29,425 and
2,943 articles, respectively, yielding a corpus of
32,368 articles (avg. article length: 4,233 words).
4 Experiment I: Predicting Plausible
Materials for Vehicle Components
ModelingWe first perform a baseline exper-
iment to explore LLM performance compared
to previous work leveraging pattern-based boot-
strapping algorithms (Schlipf, 2022) and domain-
adapted PLMs of varying size (Eichel et al.,
2023). For this, we evaluate the open-source
LLM Mixtral-8x22B-Instruct-v0.1 (Mistral
AI, 2024).3We prompt the model in both sim-
ple zero-shot and few-shot settings.4We conduct
an initial analysis of the output and find relevant
candidates present in both settings.5However, few-
shot results tend to be more specific than zero-shot
results. For instance, instead of quite generic ma-
terial candidateplastic, model predictions include
Acrylonitrile Butadiene Styrene (ABS)orPolyvinyl
Chloride (PVC). Thus, expert evaluation focuses
on few-shot prompting output only.
Evaluation StudyFour annotators (three engi-
neering experts recruited via Prolific and one au-
thor) evaluate whether each predicted material is
plausible for a given component. We use Google
Forms to present five material candidates in a
multiple-choice question setup including the option
that no material is plausible. Further, annotators
3We are aware that a wide range of models, including
more recent ones, exist and that alternative models may yield
different results. Since our work explores relative differences
between models, we nevertheless believe that our findings
provide valuable insights into a highly underexplored topic.
4See for details App. A.
5One author performed a manual evaluation.

could indicate that they do not know the answer to
make sure that collected responses are trustworthy.
Following prior work (Schlipf, 2022; Eichel et al.,
2023), we calculate inter-annotator agreement as
follows. Given two sets of annotations AandBfor
the same component, each set encodes the annota-
tor’s choice of plausibility for each material. aior
bidenote thei-th element in the set, and withδ i:
δi=(
1a i=bi
0a i̸=bi(1)
Subsequently, inter-annotator agreement (IAA) is
calculated as laid out in Eq. (2). IAA values of 1
and 0 denote perfect agreement and disagreement,
respectively. Annotator-specific results are shown
in App. A, Table 5, yielding an average IAA = 0.69
consistent with prior work.
IAA=P|A|
i=1δi
|A|(2)
Evaluation MetricsWe use the following met-
rics to compare model performance.
•COVERAGE@ n: Proportion of components for
which at least one material among a model’s top-
5 predictions is rated plausible bynannotators.
•PRECISION@ n: Proportion of material predic-
tions among a model’s top-5 predictions rated
plausible bynannotators.
ResultsResults are shown in Figure 1 and Table1,
indicating that an off-the-shelf LLM performs very
well on the task at hand. This is mirrored in both
(i) high coverage, i.e., the LLM predicted at least
one material considered plausible by all annotators
in 94 out of 100 cases, and (ii) more than 90%
of material predictions are considered plausible
by at least half of the annotators. The compari-
son to prior work (metrics granularity: @1 and
@3) highlights a clear gap in performance between
auto-regressive LLMs, domain-adapted encoder-
only models such as RoBERTa (DOMAIN RB)
(Eichel et al., 2023), and pattern-based approaches
such as Basilisk (Schlipf, 2022). Our results also
reveal that the task of plausible material prediction
for components in the vehicle repair domain does
not require more complex approaches such as RAG.
However, while plausible materials for a compo-
nent can be elicited from LLMs with a fair degree
of confidence, the question remainswhythe mate-
rial candidate was predicted. In the following, we
thus focus on the prediction of plausible materialsincluding an explanation regarding potential use
of the material in a component.
Figure 1: Few-shot LLM results
COVERAGEPRECISION
@1 @3 @1 @3
Schlipf (2022) 73% 40% 45% 14%
Eichel et al. (2023) 93% 73% 62% 28%
Our approach 100% 100% 97% 71%
Table 1: Comparison: Previous work vs. our results
usingMixtral-8x22B-Instruct-v0.1.
5 Experiment II: Explaining Plausible
Material Candidates
5.1 Modeling Approaches
No-RAG (Standard Generative QA)Since
LLMs are trained on vast amounts of training data
containing information across many domains, they
store a significant amount of knowledge that can
be harnessed without using fine-tuning or using
advanced techniques like RAG. This knowledge
can be extracted using prompting, as shown in
Experiment I (§4).6To elicit model responses,
we prompt an LLM with a zero-shot question in-
spired by (Zhang et al., 2024) targeting the domain-
specific context of interest: “In the automotive con-
text, which materials is the component ’<compo-
nent>’ made out of?” (cf. App. B, Figure B for
the full prompt). The model is further instructed
to list materials sorted by prevalence and provide a
brief explanation for each. Thus, our approach re-
lies entirely on knowledge encoded in the model’s
weights during training.
Retrieval-Augmented Generation (RAG)We
use RAG to augment the standard generation QA
6Note that the no-RAG approach partially overlaps with
the baseline setup, however, the LLM is not only prompted to
generate a list of materials but also an explanation why each
material was predicted. This task extension allows us to eval-
uate LLM consistency in response and overall performance
with the extended input prompt.

prompt with up to ten domain-specific passages
before generating the response. We implement a
hybrid retrieval pipeline (cf. App. B, Figure 3).
Retrieval ModuleThe retrieval system was im-
plemented using the Haystack framework with
OpenSearch as thedocument store(chosen over
an in-memory store for performance reasons). The
ingestion pipeline comprises four stages: (1)data
cleaningthrough the conversion of MediaWiki
markup to plain text using a patched version of
wikiextractor7; (2)splittingarticles into 200-
word passages with a 50-word sliding-window over-
lap; (3)embeddingeach passage using the DPR
context embedding model (Karpukhin et al., 2020),
which is specifically trained for question answer-
ing, and yields clearly better results than MatSci-
BERT (Gupta et al., 2022) based on preliminary
experiments; and (4)indexingpassage text and
embeddings into OpenSearch. In total, 458,972
passages are indexed.
The retriever module uses hybrid retrieval by
combining sparse retrieval (BM25, using the com-
ponent name as query) and dense retrieval (DPR
embeddings, using a full question as query). The
two retrieval paths are implemented such that it is
possible to pass different queries to the sparse and
dense retrievers. This is useful because sparse re-
trieval is optimally used with keywords as queries,
while the embedding model of the dense retriever
is trained using a whole sentence as input. Both
retrievers return the top-50 passages, which are
then merged and re-ranked by a document joiner.
The final output is the top-10 passages. A manual
evaluation of the first 10 components found that
roughly 22% of retrieved passages (2.2 per compo-
nent) contained genuinely helpful information: a
moderate but non-trivial signal that motivates our
RAG design.8
Full RAG WorkflowTo augment a prompt and
generate a prediction and explanation, the retriever
is coupled with an LLM, as shown in Figure 2.
The retriever is fed two separate queries: (i) the
component name for sparse (BM25) retrieval, and
7https://github.com/attardi/wikiextractor
8For instance, one passage retrieved for the component
refrigerant bypass valveprovides details about refrigerants
such as R-12, R-22, and R-134a, which are commonly used
in air conditioning systems. Another passage highlights the
high global warming potential of these refrigerants. Com-
bined, these insights suggest that sealing may be crucial in
refrigeration systems, leading to the inference that materials
like rubber might be used for sealing in the refrigerant bypass
valve. Therefore, both of these passages would be considered
as containing helpful information.(ii) a component-specific question for dense (DPR)
retrieval, e.g., the query “What materials does the
vehicle component ’<component>’ consist of?”.
The top-10 retrieved passages are prepended to the
prompt with instructions to answer even if direct
material mentions are absent, and to avoid citing
the passages explicitly in the output.
Chain-of-Verification (CoVe)CoVe (Dhuli-
awala et al., 2024) extends RAG with an iterative
self-verification loop to reduce hallucinations in
the generated answers. The process includes the
following steps visualized in Figure 2.
•Generate Baseline Response: The model gen-
erates an initial material list with explanations
(identical to the RAG approach above).
•Plan Verifications: The model receives the orig-
inal question and initial response, then generates
a set of targeted verification questions.
•Execute Verifications: Each verification ques-
tion is answered independently using the retrieval
pipeline, which can surface information about
materials not retrieved in Step 1.
•Generate Final Verified Response: The initial
response and all verification QA pairs are con-
catenated into a final prompt, and the model re-
fines its answer accordingly.
The key advantage of CoVe is that verification ques-
tions may uncover new retrieval targets. For exam-
ple, the initial retrieval for the componentengine
pistonmight include a range of plausible materials
used for building pistons as well as the material
candidatetin. Unless the model has seen informa-
tion about tin (in relation to piston manufacturing)
during training, it misses the knowledge that the
melting point of tin is too low to make it a plau-
sible material candidate for pistons. With CoVe,
the model can generate a verification question such
as “Is tin a suitable material for engine pistons?”.
Based on this question, the retriever can include
information about tin, allowing the model to reason
that the melting point is too low and conclude that
tin is not a plausible material for engine pistons.
For a full prompt example, see App. B, Figure B.2.
Experimental SetupAll three methods use
both Mixtral-8x22B-Instruct-v0.1 (Mistral
AI, 2024) and GPT-4o (OpenAI et al., 2024) (cf.
App. B for details), yielding six output variants per
component.

Figure 2: Simplified overviews of (i) standard RAG workflow including the retriever module and a generation model
(top, blue box), and (ii) CoVe workflow including a baseline response, planning and execution of verifications, and
the final verified answer (bottom, green box).
5.2 Expert Evaluation Study
Study DesignWe conduct a human expert eval-
uation study to evaluate no-RAG and RAG-based
system performance. Evaluation was divided along
three aspects, following prior work (Zhong et al.,
2022) which are elicited through five tasks.
CorrectnessAs LLMs are prone to hallucination,
annotators are instructed to determine whether a
material prediction is plausible for a component.
In addition, they are also tasked with verifying the
factuality of the explanations.
•Task 1: Annotators are shown a list with all mate-
rial predictions, and have to select all plausible
materials.
•Task 2: Annotators are shown the full output,
including explanations. They have to highlight
all factually incorrect parts.
CompletenessAnnotators assess whether mate-
rial predictions and explanations are complete, en-
suring that all expected materials are included, and
explanations provide sufficient information.
•Task 3: Annotators are tasked with rating the
quality of the material predictions on a scale of
one to five. If they think a material is missing,
they are asked to enter it into an optional com-
ment text field.
•Task 4: Similar to the previous task, annotators
are tasked with rating the quality of the explana-
tions on a scale of one to five. They can enterwhich information they miss or which is super-
fluous in a text field.
RankingAnnotators are asked to rank the out-
puts according to their subjective expert prefer-
ences. Importantly, they should only consider the
contents of the output when ranking, not the for-
matting or style.
•Task 5: For each component, annotators are
shown all six outputs simultaneously. They are
asked to rank the outputs based on their prefer-
ence, as shown in Figure 5.
Tasks 1–4 are displayed simultaneously, while
Task 5 is shown individually (cf. App B.1, Fig-
ures 4 and 5).
Annotation ToolWe develop a custom full-stack
annotation tool from scratch since no existing open-
source or cost-free annotation tool satisfied all re-
quirements for a suitable evaluation platform.9The
tool is built using TypeScript, NestJS (backend),
SQLite (database), and Vue with PrimeVue (fron-
tend). It supports web-based access without in-
stallation, secure token-based authentication, auto-
matic progress saving, text-span highlighting for
marking factual errors, star ratings, optional free-
text comments for missing materials, and user-
9Since Google Forms was used in the baseline study, it
would be an obvious choice. However, the tool lacks highlight-
ing support and does not offer an annotator-friendly way to
implement the ranking task. Potato (Pei et al., 2022) has lim-
ited security and documentation, and LimeSurvey’s JavaScript
customization pathway is is relatively unexplored with no
working community examples.

COVERAGE@2 PRECISION@2
Mixtral GPT-4o Mixtral GPT-4o
Baseline 100.0% — 90.6% —
No RAG 100.0% 98.0% 87.3% 89.7%
RAG 100.0% 92.0% 88.0%93.7%
CoVe 100.0% 98.0% 89.3% 91.2%
Average 100.0% 96.0% 88.8%91.2%
Table 2: Overview of results for plausible material pre-
diction (Task 1), comparing baseline results (cf. §4),
No-RAG, RAG, and CoVe approaches. Metrics are cal-
culated at majority vote (@2 out of 3 expert annotators).
friendly drag-and-drop ranking of multiple outputs
simultaneously. Our tool is designed as a reusable
boilerplate, with survey-specific logic implemented
as extensions to the core framework.
Setup and ParticipantsWe evaluate 50 compo-
nents (5 of 10 batches of 10 components each),
and recruit three annotators per batch via Prolific.
Participants are pre-screened to be located in the
US, UK, or Germany and fluent in English; two
hold engineering degrees and one a materials sci-
ence degree per batch.10Participants are guided
through an initial interactive tutorial explaining
how the tool works and receive detailed introduc-
tion to all the tasks. We embed attention checks in
each batch; one participant who failed both checks
was excluded per Prolific guidelines. For further
details on the study setup, we refer to App. B.1.
Annotator (Dis)agreementTo determine anno-
tation reliability, we calculate agreement among
annotators leveraging the following metrics for
the various tasks. Agreement for Task 1 (plau-
sibility selection) is calculated using IAA as laid
out in Eq. 2. The achieved IAA score of 0.63 is
slightly below the baseline study’s 0.69 but still
very reasonable. For Task 2 (highlighting), not
enough overlap between annotators is observed
to allow for the calculation of a meaningful met-
ric (see B.2 for more details). For Tasks 3 and 4
(star ratings for material and explanation quality),
Krippendorff’s α-coefficient is calculated (Krip-
pendorff, 2011). Here, we find annotators mostly
disagreeing in their choices (Krippendorff’s α≈
-0.22 and -0.23 respectively). For Task 5 (rank-
ing), the rank correlation coefficient as laid out by
Kendall (1938) is computed, reaching a moderate
Kendall’s τ= 0.24 . Overall, agreement varies
10Imbalance in expert background stems from a surplus of
annotators with an engineering vs. materials science degree.Task 2 Task 5
Mixtral GPT-4o Mixtral GPT-4o
No RAG 22% 22% 2.29 2.07
RAG 30%18% 2.582.84
CoVe 28% 18% 2.45 2.77
Average 27%19% 2.442.56
Table 3: Overview of model outputs flagged to contain
at least one factual inaccuracy (Task 2, left panel), and
average position in ranking (Task 5, right panel).
considerably across tasks, with scores for Tasks 3
and 4 indicating substantial disagreement. In the
following, we thus present results in conjunction
with further analyses on potential sources of this,
and provide a detailed discussion in §5.4 and §6.
5.3 Results
Task 1: Plausible Material PredictionResults
for the prediction of plausible material candidates
are shown in Table 2, usingCOVERAGEandPRECI-
SIONat majority vote.11Overall, results show very
strong performance across the board, with slight
differences between the Mixtral andGPT-4o mod-
els:Mixtral scores better forCOVERAGE@2 and
GPT-4o reaches higher performance forPRECI-
SION@2. We further analyze model performance
regarding material predictions by prevalence. More
specifically, we expect most used, and thus plausi-
ble, materials to be listed first, and less used, and
thus less plausible, candidates to be listed last. Re-
sults are shown in App. B.2, Table 9, indicating that
the models indeed sorted materials by prevalence as
instructed: first predictions were on average more
often rated plausible (88.9%) than last predictions
(85.1%).
Task 2: Factual Error DetectionThe goal of
this task is to detect factually incorrect sections
(hallucinations) in generated predictions and ex-
planations. Annotators highlighted 177 text sec-
tions as potentially factually incorrect. After
manually filtering out accidental highlights (28),
redundant plausibility markings (51), correctly
highlighted-but-accurate statements (4), and invalid
outputs (9), 85 highlighted sections remain that do
contain factual inaccuracies or hallucinations re-
mained. Remarkably, only two factual errors were
flagged by more than one annotator. We report
an overview of outputs containing 1+ factual error
11We use majority vote to account for varying numbers of
annotators. We refer the reader for results forPRECISION@3
andCOVERAGE@3 and a detailed discussion to App. B.2.

Task 3 Task 4
Material Rat. Explanation Rat.
Mixtral GPT-4o Mixtral GPT-4o
No RAG 4.25 4.27 4.15 4.22
RAG 4.12 3.92 3.87 3.89
CoVe 4.15 4.09 4.09 4.06
Average 4.17 4.10 4.04 4.06
Table 4: Average Material Ratings (Task 3) and Average
Explanation Ratings (Task 4).
in Table 3, revealing no significant disparities be-
tween the GPT-4o model producing fewer errors
than Mixtral . We perform an additional analy-
sis (cf. App. B.2) examining the relation between
materials considered to be plausible (Task 1) and
marked factual errors. Results indicate a fairly
balanced distribution of highlighted errors across
material candidates considered as plausible vs. im-
plausible. Hence, we cannot conclusively attribute
marked inaccuracies to either implausible materials
or errors in the explanation. We also investigate
whether factual errors are skewed towards first or
last predictions. Results do not indicate such a
tendency, neither for materials rated plausible nor
implausible.
Task 3 and Task 4: Quality RatingsThe goal of
these two tasks is to understand whether material
predictions (Task 3) and explanations (Task 4) do
not lack relevant materials or contain superfluous
information. Results are shown in Table 4. On
average, annotators award approx. 4.1/5 stars for
material quality and 4.1/5 for explanation quality
across all approaches and models. No-RAG re-
ceived highest ratings for both materials (4.26) and
explanations (4.18), while simple RAG received the
lowest scores (4.02 and 3.88 respectively). CoVe
sits in between. Overall, Mixtral predictions and
ratings seem to be considered of slightly higher
quality than GPT-4o output, however, the differ-
ences are non-substantial.
Task 5: Preference RatingThe final task targets
eliciting the subjective preference of the involved
expert annotators. Outputs ranked first are at rank
0, while outputs ranked last are at rank 5, i.e. lower
is better. Average positions are summarized in
Table 3. Results show that no-RAG outputs were
ranked noticeably better (avg. position 2.18) than
RAG (2.71) and CoVe (2.61). A potential confound
is the default order randomized per component but
not per annotator: if annotators were biased towardminimal reordering, this could inflate agreement
and reduce sensitivity to actual differences.
5.4 Discussion
Hyperparameter OptimizationContrary to our
expectations, LLM-based systems incorporating
RAG components did not outperform the No-RAG
implementation. We hypothesize that irrelevant
passages might have been retrieved which nega-
tively influenced performance. However, for RAG
and CoVe to outperform the no-RAG baseline, the
retriever must identify useful passages in the re-
trieval sources (Cuconasu et al., 2024). Further-
more, another important hyperparameter is the
number of retrieved results (top- k). While higher k
values increase the likelihood of retrieving relevant
information, they can also introduce excessive irrel-
evant context that may mislead the LLM. Similar
considerations apply to prompts: prior work shows
that even small prompt changes can significantly
affect outputs (Chen et al., 2025), making prompts
another hyperparameter. Importantly, our work
does not focus on prompt engineering but instead
adopts prompts from related work. Hence, we can-
not conclusively determine how prompt changes
affect overall performance.
Retrieval Data QualityBeyond the relevance
of the retrieved passages, the quality of the avail-
able data also plays a critical role. Wikipedia pro-
vides valuable information for common compo-
nents (brake discs, camshafts), however, it does
lack material-level detail for highly specialized
components. If the retriever fails to locate rele-
vant information simply because it is absent in the
data, the probability that RAG outperforms non-
RAG methods decreases. We investigated alterna-
tive domain-specific corpora but up to date we are
not aware of alternatives that are both (i) substan-
tially more informative and (ii) legally available for
research use.
6 Conclusion
In this work, we tackled the task of predicting plau-
sible materials for vehicle components. A baseline
experiment established that even a simple LLM ap-
proach clearly outperforms prior work. Our main
study extended the task to require explanations
alongside material predictions, and compared No-
RAG, standard RAG, and Chain-of-Verification
methods across two established LLMs. We con-
ducted an expert evaluation study using a custom-

built annotation tool, and found significant but non-
substantial differences between approaches. De-
spite observed limitations, the results are encourag-
ing: LLM-based systems (with our without RAG)
do hold promise for practical deployment in vehicle
repair assistance and related industry applications.
Limitations
Hyperparameter OptimizationSince no gold
standard exists for the open-ended task addressed
in this work, automated evaluation metrics cannot
be applied.Hyperparameter optimization based on
end-to-end system performance thus presents a lim-
itation since running a full evaluation study for
each set of possible hyperparameters is not feasible.
To account for this, we perform targeted human
evaluation of results of different extent and sample
sizes at different development stages.
Expert AnnotatorsWe achieve very reasonable
agreement for the evaluation of the task of predict-
ing plausible material candidates. The evaluation of
the generated explanations as to why a specific ma-
terial is predicted is more challenging, with lower
agreement and even disagreement between annota-
tions collected from domain experts. We point out
potentially limiting) reasons for this observation.
Firstly, the questions in the study are deliberately
designed to be open-ended, allowing annotators a
certain degree of freedom wrt. interpretation. This
approach was intentional, given the ambiguous na-
ture of the task. For instance, one of the generated
outputs includes the statement, “Plastic is one of
the most commonly used materials in modern car
interiors”. One annotator flagged this as factually
incorrect, which might reflect their expert or even
non-domain related individual background. An an-
notator with domain expertise on luxury cars might
indeed find this statement inaccurate, while some-
one concentrating on budget cars might rate the
explanation entirely plausible.
Secondly, we recruit expert annotators through
the crowdsourcing platform Prolific which intro-
duces additional challenges. The presented study
is particularly complex, requiring the participants
to have a solid understanding of both automotive
technology and materials science. While partici-
pants are required to hold a degree in engineering
or material sciences, we do observe varied quality
in completed work: some participants completed
tasks in under 30 minutes for a study estimated at
65 minutes (passing the attention checks). In thiscase, Prolific’s payment model likely incentivizes
speed over quality since a pre-defined amount is
paid upon completion even if less time than antic-
ipated was needed while bonus payment in case
more time was needed is voluntary.
Ethical Considerations
For the presented RAG approaches, we harness
publicly available data. More specifically, we use
a portion of the English Wikipedia customized
to the domain of interest. We acknowledge that
Wikipedia text content including Wikipedia dumps
is licensed under both the Creative Commons
Attribution-ShareAlike 3.0 License and the GNU
Free Documentation License.
We use and adapt an open-source LLM as pro-
vided and licensed under the Apache License 2.0
byhuggingface (Wolf et al., 2020). We further
use a closed, proprietary model with inaccessible
training datasets and algorithmic weights. We ac-
knowledge the possibility of (accidental) breaches
of data privacy, systemic model bias, transparency,
and reproducibility introduced by the use of such a
model. Across models, we point out that retrieved
material predictions and retrieved explanations us-
ing the outlined methods are a product of learning
methods which might be prone to error. We recom-
mend that predictions and generated text should be
approved by an expert or flagged otherwise in case
they are used in a downstream application to avoid
potential risks including harm of objects or safety
risks in case of incorrect repair procedures.
In the context of our evaluation tasks, we col-
lected ratings from human participants. Annotation
was fully voluntary and could be stopped at any
time without providing any reasons. We paid par-
ticipants fairly according to the platform’s recom-
mendation, communicated decisions transparently,
and reached out to individual participants whenever
necessary during the annotation approval process.
Use of AI AssistantsThe authors acknowledge
the use of AI assistants solely for correcting gram-
matical errors, optimizing coherence and length
within selected paragraphs, and formatting boxes
and tables.
References
Steve Bakos, Chen Xing, Heidar Davoudi, Aijun An,
and Ron DiCarlantonio. 2025. Generating spatial
knowledge graphs from automotive diagrams for

question answering. InProceedings of the 2025
Conference on Empirical Methods in Natural Lan-
guage Processing: Industry Track, pages 2270–2286,
Suzhou (China). Association for Computational Lin-
guistics.
Nidhisree C, Panchami Dinesh, Baishali Garai, Ananya
Paul, and Rajat Subhra Bhowmick. 2025. Automat-
ing knowledge discovery in material science with
rag framework. In2025 IEEE 22nd India Council
International Conference (INDICON), pages 1–6.
Banghao Chen, Zhaofeng Zhang, Nicolas Langrené,
and Shengxin Zhu. 2025. Unleashing the potential
of prompt engineering for large language models.
Patterns, 6(6):101260.
Jerry Cheung, Yuchen Zhuang, Yinghao Li, Pranav
Shetty, Wantian Zhao, Sanjeev Grampurohit, Rampi
Ramprasad, and Chao Zhang. 2024. POLYIE: A
dataset of information extraction from polymer mate-
rial scientific literature. InProceedings of the 2024
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
pages 2370–2385, Mexico City, Mexico. Association
for Computational Linguistics.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, SIGIR ’24, page
719–729, New York, NY , USA. Association for Com-
puting Machinery.
Maxime Delmas, Magdalena Wysocka, Danilo Gu-
sicuma, and Andre Freitas. 2025. Accelerating an-
tibiotic discovery with large language models and
knowledge graphs. InProceedings of the 63rd An-
nual Meeting of the Association for Computational
Linguistics (Volume 6: Industry Track), pages 693–
705, Vienna, Austria. Association for Computational
Linguistics.
Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu,
Roberta Raileanu, Xian Li, Asli Celikyilmaz, and
Jason Weston. 2024. Chain-of-verification reduces
hallucination in large language models. InFindings
of the Association for Computational Linguistics:
ACL 2024, pages 3563–3578, Bangkok, Thailand.
Association for Computational Linguistics.
Annerose Eichel, Helena Schlipf, and Sabine Schulte im
Walde. 2023. Made of steel? learning plausible ma-
terials for components in the vehicle repair domain.
InProceedings of the 17th Conference of the Euro-
pean Chapter of the Association for Computational
Linguistics, pages 1420–1435, Dubrovnik, Croatia.
Association for Computational Linguistics.
Luyu Gao, Zhuyun Dai, Tongfei Chen, Zhen Fan, Ben-
jamin Van Durme, and Jamie Callan. 2021. Comple-
ment lexical retrieval model with semantic residualembeddings. InAdvances in Information Retrieval:
43rd European Conference on IR Research, ECIR
2021, Virtual Event, March 28 – April 1, 2021, Pro-
ceedings, Part I, page 146–160, Berlin, Heidelberg.
Springer-Verlag.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Tanishq Gupta, Mohd Zaki, N. M. Anoop Krishnan,
and Mausam. 2022. MatSciBERT: A materials do-
main language model for text mining and information
extraction.npj Computational Materials, 8(1):102.
Carolin Holtermann, Paul Röttger, and Anne Lauscher.
2025. Around the world in 24 hours: Probing LLM
knowledge of time and place. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
22875–22897, Vienna, Austria. Association for Com-
putational Linguistics.
Peter Jansen, Samiah Hassan, and Ruoyao Wang. 2025.
Matter-of-fact: A benchmark for verifying the fea-
sibility of literature-supported claims in materials
science. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 4090–4102, Suzhou, China. Association for
Computational Linguistics.
Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko
Ishii, and Pascale Fung. 2023. Towards mitigating
LLM hallucination via self reflection. InFindings
of the Association for Computational Linguistics:
EMNLP 2023, pages 1827–1843, Singapore. Associ-
ation for Computational Linguistics.
Ashly Joseph. 2025. Reducing hallucinations in large
language models through integrated self-verification
and retrieval-augmented generation. InInternational
Design Engineering Technical Conferences and Com-
puters and Information in Engineering Conference,
volume 89213, page V02BT02A032. American Soci-
ety of Mechanical Engineers.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
M. G. Kendall. 1938. A new measure of rank correla-
tion.Biometrika, 30(1-2):81–93.
Junho Kim, Yeachan Kim, Jun-Hyung Park, Yerim
Oh, Suho Kim, and SangKeun Lee. 2024. MELT:
Materials-aware continued pre-training for language
model adaptation to materials science. InFindings
of the Association for Computational Linguistics:
EMNLP 2024, pages 10690–10703, Miami, Florida,
USA. Association for Computational Linguistics.

Klaus Krippendorff. 2011. Computing Krippendorff’s
alpha-reliability.
Nikita A Krotkov, Dmitrii A Sbytov, Anna A
Chakhoyan, Polina I Kornienko, Anna A Starikova,
Maxim G Stepanov, Anastasiia O Piven, Timur A
Aliev, Tetiana Orlova, Mushegh S Rafayelyan, and
1 others. 2025. Nanostructured material design
via a retrieval-augmented generation (rag) approach:
Bridging laboratory practice and scientific litera-
ture.Journal of Chemical Information and Modeling,
65(20):11064–11078.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Flavio Merenda, Jose Manuel Gomez-Perez, and Ger-
man Rigau. 2026. Can LLMs reason like doctors?
exploring the limits of large language models in com-
plex medical reasoning. InFindings of the Associ-
ation for Computational Linguistics: EACL 2026,
pages 2432–2452, Rabat, Morocco. Association for
Computational Linguistics.
Mistral AI. 2024. Mixtral 8x22b. https://mistral.
ai/news/mixtral-8x22b/.
Yerim Oh, Jun-Hyung Park, Junho Kim, SungHo Kim,
and SangKeun Lee. 2025. Incorporating domain
knowledge into materials tokenization. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 9623–9644, Vienna, Austria. Associa-
tion for Computational Linguistics.
Elsa A. Olivetti, Jacqueline M. Cole, Edward Kim, Olga
Kononova, Gerbrand Ceder, Thomas Yong-Jin Han,
and Anna M. Hiszpanski. 2020. Data-driven materi-
als research enabled by natural language processing
and information extraction.Applied Physics Reviews,
7(4):041317.
OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher,
Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec
Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb,
Alex Beutel, Alex Borzunov, Alex Carney, Alex
Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o
system card.Preprint, arXiv:2410.21276.
Jiaxin Pei, Aparna Ananthasubramaniam, Xingyao
Wang, Naitian Zhou, Apostolos Dedeloudis, Jack-
son Sargent, and David Jurgens. 2022. POTATO:
The portable text annotation tool. InProceedings of
the 2022 Conference on Empirical Methods in Nat-
ural Language Processing: System Demonstrations,
pages 327–337, Abu Dhabi, UAE. Association for
Computational Linguistics.Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
S. E. Robertson and S. Walker. 1994. Some simple
effective approximations to the 2-poisson model for
probabilistic weighted retrieval. InSIGIR ’94, pages
232–241, London. Springer London.
Helena Schlipf. 2022. Learning domain-specific mate-
rial properties in the vehicle repair information do-
main using unstructured and structured data sources.
Master’s thesis, University of Stuttgart.
Yu Song, Santiago Miret, Huan Zhang, and Bang Liu.
2023. HoneyBee: Progressive instruction finetuning
of large language models for materials science. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2023, pages 5724–5739, Singapore.
Association for Computational Linguistics.
Kai Sun, Yifan Xu, Hanwen Zha, Yue Liu, and Xin Luna
Dong. 2024. Head-to-tail: How knowledgeable are
large language models (LLMs)? A.K.A. will LLMs
replace knowledge graphs? InProceedings of the
2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long Pa-
pers), pages 311–325, Mexico City, Mexico. Associ-
ation for Computational Linguistics.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le
Scao, Sylvain Gugger, and 3 others. 2020. Trans-
formers: State-of-the-art natural language processing.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing: System
Demonstrations, pages 38–45, Online.
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu,
Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang,
Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei
Bi, Freda Shi, and Shuming Shi. 2025. Siren’s
song in the AI ocean: A survey on hallucination in
large language models.Computational Linguistics,
51(4):1373–1418.
Zihan Zhang, Meng Fang, and Ling Chen. 2024. Re-
trievalQA: Assessing adaptive retrieval-augmented
generation for short-form open-domain question an-
swering. InFindings of the Association for Compu-
tational Linguistics: ACL 2024, pages 6963–6975,
Bangkok, Thailand. Association for Computational
Linguistics.
Ming Zhong, Yang Liu, Da Yin, Yuning Mao, Yizhu
Jiao, Pengfei Liu, Chenguang Zhu, Heng Ji, and

Jiawei Han. 2022. Towards a unified multi-
dimensional evaluator for text generation. InPro-
ceedings of the 2022 Conference on Empirical Meth-
ods in Natural Language Processing, pages 2023–
2038, Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
A Experiment I
Your task is to predict a comma-separated list of up to
five materials that a vehicle component is made of. For
example:
Component: brake disc
Materials: grey cast iron, carbon-ceramic composite,
ceramic
Component: motor oil
Materials: mineral oil, synthetic oil, additives
Component: igniter
Materials: nickel, aluminium oxide, sintered alumina,
steel
Please predict the materials for the following
component: component
Just provide the names of the materials, separated by
commas, without any explanation. Answer:
Author A1 A2 A3
Author — 0.68 0.75 0.69
A1 0.68 — 0.75 0.60
A2 0.75 0.75 — 0.64
A3 0.69 0.60 0.64 —
Table 5: Inter-annotator agreement scores.
B Experiment II
Please answer the question below. When listing multi-
ple materials, sort them by the amount of the material
used, from the most used to the least used. Please
also provide a brief explanation for each material. If
you do not know the answer directly, please suggest
plausible materials that could be used in the component.
Question: In the automotive context, which ma-
terials is the component ”component” made out of?
Answer:
Experimental SetupThroughout our experi-
ment we use Mixtral-8x22B-Instruct-v0.1
(Mistral AI, 2024) and GPT-4o (OpenAI et al.,
2024) as language models. We access Mixtral
through huggingface (Wolf et al., 2020), and run
it locally on two NVIDIA RTX A600 GPUS with
4-bit quantization with default parameter settings.
We access GPT-4o through the OpenAI API.
For RAG, we use OpenSearch12as document
12https://opensearch.org/
Figure 3: Overview of retriever module including input
encoding and retriever paths.
store which supports both fast full-text search via
an inverted index (sparse retrieval), as well as re-
trieval using vector embeddings of the text (dense
retrieval). The OpenSearch instance is hosted on a
private server including a two-way authentication
with transport layer security (TLS) certificates for
secure communication. In addition, an OpenSearch
dashboard is deployed to explore the data and
test queries directly with a graphical user inter-
face. To create embeddings, we use a model13by
Karpukhin et al. (2020) which we obtain through
huggingface.
B.1 Evaluation Study
Annotation CostsDomain expert annotations
collected in the context of Experiment I (§4) and
Experiment II (§5) required a budget of 78$ and
202$, respectively. We follow the used platform’s
guidelines regarding fair payment and compensate
completion times exceeding our initial estimates
with a bonus payment.
B.2 Results
Task 1: Plausible Material PredictionWe show
additional results for Task 1 using metrics in corre-
spondence to Experiment I (§4 in Table 7). While
13https://huggingface.co/facebook/dpr-ctx_
encoder-multiset-base

no significant difference between the three ap-
proaches can be observed, all approaches perform
significantly worse than the baseline forCOVER-
AGE@3 andPRECISION@3. The lower scores
can be attributed to the high level of disagreement
among raters, as reflected by the low IAA scores
(cf. Table 6). To shed further light on the disagree-
ment within the collected annotations, we calculate
IAA per batch. Result are shown in Table 8, indi-
cating that annotator disagreement is batch-specific
with individual annotators agreeing more with each
other for batches 1, 3, and 5 and less for batches 2
and 4.
It should be further noted that given the agree-
ment percentages seen in the studies,COVERAGE
andPRECISIONheavily depend on the number of
annotators, and are thus not directly comparable.
For example, when calculatingCOVERAGE@3 with
100 annotators, it is highly likely that at least one
material suggestion will be deemed plausible by at
least three annotators, as this represents just 3% of
the total. In this study, however, three annotators
account for 100% of the evaluations, meaning a
material suggestion must be considered plausible
by all annotators, which is much less likely. The
same applies toPRECISION@3. We thus present
results at majority vote in Table 2.
Task 2We further analyze the relation between
the highlighted sections and annotator decisions
for Task 1. 40 out of the 85 highlighted sections
are part of explanations for materials that are con-
sidered plausible by the annotator in the previous
task. In contrast, 45 sections are part of expla-
nations relating to material that was considered
implausible in the previous task. In this context, it
remains unclear whether the highlighting of these
sections stems from an implausible material can-
didate or factual inaccuracies in the explanation.
Consequently, the error rate presented in Table 3
may overestimate the number of factual errors.Prompt:Your task is to create verification questions
based on the below original question and the baseline
response. The verification questions are meant for
verifying the factual accuracy in the baseline response.
Output should be numbered list of verification
questions.
Actual Question: “In the automotive context,
which materials is the component “charge-air cooler
water sprayer” made out of” Baseline Response: “1.
Water: The primary material used in a charge-air
cooler water sprayer is water. It is used to spray onto
the charge-air cooler to lower the temperature of the
incoming air, which enhances engine efficiency by
increasing air density.
2. Aluminum: While the sprayer itself may use water,
the charge-air cooler often uses aluminum due to its
lightweight properties and good thermal conductivity,
though aluminum is near its upper technological limit
at high temperatures and pressures.
The water is used in the greatest amount as it is
continuously sprayed, while aluminum forms the
structural component of the cooler.”
Final Verification Questions:
Generated Response:
1. Is water a material used in the construction of
the charge-air cooler water sprayer, or is it only the
medium being sprayed?
2. Is aluminum specifically used in the construction
of the charge-air cooler water sprayer, or is it used
primarily in the charge-air cooler itself?
3. Does the baseline response accurately distinguish
between the materials used in the sprayer system and
those used in related components like the charge-air
cooler?
4. Are there any other materials commonly used in the
construction of a charge-air cooler water sprayer that
are not mentioned in the baseline response?

Figure 4: Screenshot of annotation interface for tasks 1–4.

Figure 5: Screenshot of annotation interface for task 5.

Batch 1 Batch 2 Batch 3 Batch 4 Batch 5 Average
Task 1A0.72 0.37 0.75 0.59 0.76 0.63±0.16
Task 3B0.19 -0.23 -0.41 -0.40 -0.25 −0.22±0.24
Task 4B-0.14 -0.05 -0.42 -0.26 -0.29 −0.23±0.14
Task 5C0.49 0.26 0.08 0.07 0.32 0.24±0.17
Table 6: Inter-Annotator Agreement using the following different metrics to calculate scores. A: IAA as defined in
Eq. 2, B: Krippendorff’sα, C: Kendall’sτ).
COV@1 COV@3 PREC@1 PREC@3
Baseline LLM 100.0% 100.0% 97.2%71.4%
No RAG 100.0%88.0% 99.0% 63.9%
RAG 96.0% 87.0% 100.0%65.8%
CoVe 100.0%88.0% 99.1% 63.4%
Table 7:COVERAGE@n(Cov) andPRECISION@n(Prec) (Task 1).
IAA COV@1 COV@3 PREC@1 PREC@3
Batch 1 0.72 96.7% 93.3% 98.9% 79.8%
Batch 2 0.37 98.3% 56.7% 98.1% 15.6%
Batch 3 0.75 100.0% 95.0% 100.0% 84.0%
Batch 4 0.59 100.0% 96.7% 99.7% 55.1%
Batch 5 0.76 98.3% 96.7% 100.0% 86.0%
Table 8: IAA,COVERAGE@n(Cov), andPRECISION@n(Prec) per batch (Task 1).
No RAG RAG CoVe Mixtral GPT-4o Average
First Material 88.0% 88.5% 90.0% 86.7% 91.1% 88.9%
Last Material 78.0% 91.7% 86.0% 83.3% 87.0% 85.1%
Table 9: PRECISION@2 for first and last material prediction of output (Task 1).