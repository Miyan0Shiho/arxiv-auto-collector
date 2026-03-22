# V-DyKnow: A Dynamic Benchmark for Time-Sensitive Knowledge in Vision Language Models

**Authors**: Seyed Mahed Mousavi, Christian Moiola, Massimo Rizzoli, Simone Alghisi, Giuseppe Riccardi

**Published**: 2026-03-17 14:33:08

**PDF URL**: [https://arxiv.org/pdf/2603.16581v1](https://arxiv.org/pdf/2603.16581v1)

## Abstract
Vision-Language Models (VLMs) are trained on data snapshots of documents, including images and texts. Their training data and evaluation benchmarks are typically static, implicitly treating factual knowledge as time-invariant. However, real-world facts are intrinsically time-sensitive and subject to erratic and periodic changes, causing model predictions to become outdated. We present V-DyKnow, a Visual Dynamic Knowledge benchmark for evaluating time-sensitive factual knowledge in VLMs. Using V-DyKnow, we benchmark closed- and open-source VLMs and analyze a) the reliability (correctness and consistency) of model responses across modalities and input perturbations; b) the efficacy of knowledge editing and multi-modal RAG methods for knowledge updates across modalities; and c) the sources of outdated predictions, through data and mechanistic analysis. Our results show that VLMs frequently output outdated facts, reflecting outdated snapshots used in the (pre-)training phase. Factual reliability degrades from textual to visual stimuli, even when entities are correctly recognized. Besides, existing alignment approaches fail to consistently update the models' knowledge across modalities. Together, these findings highlight fundamental limitations in how current VLMs acquire and update time-sensitive knowledge across modalities. We release the benchmark, code, and evaluation data.

## Full Text


<!-- PDF content starts -->

V-DyKnow: A Dynamic Benchmark for
Time-Sensitive Knowledge in Vision Language Models
Seyed Mahed Mousavi*, Christian Moiola*, Massimo Rizzoli*, Simone Alghisi*,
Giuseppe Riccardi
Signals and Interactive Systems Lab, University of Trento, Italy
{mahed.mousavi, giuseppe.riccardi}@unitn.it
Abstract
Vision-Language Models (VLMs) are trained
on data snapshots of documents, including im-
ages and texts. Their training data and evalua-
tion benchmarks are typically static, implicitly
treating factual knowledge as time-invariant.
However, real-world facts are intrinsically time-
sensitive and subject to erratic and periodic
changes, causing model predictions to become
outdated. We present V-DyKnow, a Visual
Dynamic Knowledge benchmark for evaluat-
ing time-sensitive factual knowledge in VLMs.
Using V-DyKnow, we benchmark closed- and
open-source VLMs and analyzea)the relia-
bility (correctness and consistency) of model
responses across modalities and input pertur-
bations;b)the efficacy of knowledge editing
and multi-modal RAG methods for knowledge
updates across modalities; andc)the sources of
outdated predictions, through data and mech-
anistic analysis. Our results show that VLMs
frequently output outdated facts, reflecting out-
dated snapshots used in the (pre-)training phase.
Factual reliability degrades from textual to vi-
sual stimuli, even when entities are correctly
recognized. Besides, existing alignment ap-
proaches fail to consistently update the models’
knowledge across modalities. Together, these
findings highlight fundamental limitations in
how current VLMs acquire and update time-
sensitive knowledge across modalities. We
release the benchmark, code, and evaluation
data1.
1 Introduction
Vision-Language Models (VLM) are increasingly
used for answering factual questions about the real
world from visual inputs. In these settings, VLMs
are expected to connect the images’ content with
stored world knowledge and output answers to the
question (Cohen et al., 2025). Similar to Large
*Equal contribution.
1Benchmark material available at https://github.com/
sislab-unitn/V-DyKnow
Figure 1: An example of multimodal querying VLMs
for factual knowledge that is time-sensitive. Upon a vi-
sual stimulus, the VLM first misidentifies the entity and
retrieves an incorrect fact. Following the clarification
turn about the entity, the model generates an outdated
answer. In the final turn, the entity is explicitly stated
in the text, and the correct fact is finally returned. This
example highlights key issues investigated in this work:
the prevalence of outdated knowledge in VLMs and the
performance gap in VLMs for visual and textual stimuli.
Language Models (LLMs), VLMs are static mod-
els whose factual knowledge is acquired during
training from large heterogeneous data snapshots
collected at different points in time. These snap-
shots capture different states of the world and its
entities and may encode outdated and/or conflicting
versions of the same facts. As a result, the factual
knowledge expressed by a VLM reflects the con-
tent and temporal mixture of its (pre-)training data,
rather than the state of the world at inference time.
The evaluation of knowledge in VLMs is pre-
dominantly conducted using static benchmarks
1arXiv:2603.16581v1  [cs.AI]  17 Mar 2026

with fixed ground truth (Cohen et al., 2025; Papi
et al., 2026). Prior work has examined fac-
tual knowledge in VLMs and identified modality-
dependent gaps between visual and textual entity
representations (Cohen et al., 2025). Research on
multimodal knowledge editing has studied whether
factual updates transfer across modalities (Fang
et al., 2025; Chen et al., 2025). However, these
efforts focus on a fixed set of static facts, implicitly
assuming that factual correctness is time-invariant.
As real-world facts evolve, such benchmarks fall
out of sync with the current state of the world, and
the evaluation scores thus measure agreement with
outdated information rather than factual correctness
at inference time, undermining the validity of the
evaluation itself. This misalignment is particularly
severe in visual settings, where model responses
depend jointly on entity recognition, cross-modal
understanding, and factual recall, similiar to the
example in Figure 1.
To enable the evaluation of time-sensitive factual
knowledge in VLMs, we introduce V-DyKnow, a
visual dynamic benchmark for VLMs. Instead of
relying on fixed annotations, model outputs are
validated against up-to-date factual information
derived from Wikidata (Vrande ˇci´c and Krötzsch,
2014) at evaluation time, allowing correctness to
be assessed relative to the current state of the world.
V-DyKnow represents factual queries as visual en-
tities (e.g., portraits, flags, and logos) paired with
relations and attributes, each associated with tem-
poral validity intervals. V-DyKnow supports analy-
sis of outdatedness, cross-modality evaluation, and
input-bound and cross-modal consistency.
Using V-DyKnow, we conduct a systematic anal-
ysis of time-sensitive factual knowledge in VLMs
along three axes. First, we evaluate how accurately
and consistently VLMs retrieve real-world facts
when entities are presented visually rather than tex-
tually, and quantify the prevalence of outdated re-
sponses under visual grounding. Second, we exam-
ine whether existing alignment approaches, includ-
ing knowledge editing and multimodal retrieval-
augmented generation (RAG), can update outdated
facts across modalities. And third, we investi-
gate the sources of outdated predictions through
training-data analysis and mechanistic interpretabil-
ity. Our results show that VLMs frequently retrieve
outdated facts and that factual recall degrades under
visual stimulus, even when the entity is correctly
recognized. Existing updating methods work only
in simplified scenarios and interfere with knowl-edge acquired during pre-training. Together, these
findings highlight limitations in how current VLMs
acquire, recall, and maintain time-sensitive knowl-
edge. Our contributions can be summarized as
follows:
•We introduceV-DyKnow, a dynamic bench-
mark for evaluating time-sensitive factual
knowledge in Vision–Language Models
across modality and input perturbations.
•We provide the first systematic evaluation of
time-sensitive factual knowledge in state-of-
the-art VLMs, showing the prevalence of out-
dated facts and large performance gap across
modalities.
•We analyze the effectiveness of existing align-
ment approaches, including knowledge edit-
ing and multimodal retrieval-augmented gen-
eration.
•We investigate the sources of outdated pre-
dictions through data and mechanistic anal-
yses, linking model outputs to training data
snapshots and examining how factual recall
emerges across model layers.
2 V-DyKnow
Building on DyKnow (Mousavi et al., 2024), we
introduce V-DyKnow, a benchmark for evaluating
time-sensitive factual knowledge in VLMs under
both textual and visual grounding.
FactsFollowing the prior work (Mousavi et al.,
2024, 2025), we represent factual knowledge as
(subject, property, attribute)triplets. Thesubject
denotes a real-world entity (e.g.,Apple), theprop-
ertydefines a relation associated with that entity
(e.g.,CEO), and theattributecorresponds to the
value of that relation (e.g.,Tim Cook). To con-
struct V-DyKnow, we extract the up-to-date at-
tribute value for each subject-property in DyKnow
from Wikidata (Vrande ˇci´c and Krötzsch, 2014), a
continuously updated knowledge graph. Attributes
in Wikidata are connected to subjects through prop-
erties and enriched with qualifiers specifying their
validity intervals. For example, the subjectAp-
pleis associated with attributeTim Cookvia the
propertyCEO, annotated with the validity interval
2011–present. Meanwhile, outdated attributes are
preserved with their previously valid time spans
(e.g.,Steve Jobs 1997-2011). The final benchmark
contains 139 time-sensitive facts, including 82 facts
associated with 47 countries, 28 facts related to
athletes, and 29 facts corresponding to 22 organi-
2

(Year) ModelVLM Visual Prompt VLM Textual Prompt LLM* Textual Prompt
CorrectOutdatedIrrelevantCorrectOutdatedIrrelevantCorrectOutdatedIrrelevant
(2023) LLaV A-1.5 13% 31% 56% 33% 49% 18% 33% 47% 19%
(2024) LLaV A-OneVision 22% 36% 42% 31% 45% 24% 37% 42% 22%
(2024) PaliGemma 2 3% 4% 93% 0% 0% 100% 62% 28% 10%
(2024) Molmo 24% 20% 56% 34% 44% 22% 40% 40% 20%
(2024) Qwen2-VL 28% 38% 34% 36% 44% 20% 37% 42% 22%
(2025) Qwen2.5-VL 32% 38% 30% 39% 40% 22% 44% 35% 21%
(2025) InternVL3.5 26% 21% 53% 40% 29% 31% 51% 29% 19%
(2025) GPT-4 71% 18% 11% 72% 19% 9% - - -
(2025) GPT-5 75% 15% 10% 76% 14% 9% - - -
Table 1: Evaluation of state-of-the-art VLMs on V-DyKnow under visual and textual prompting. Results report the
percentage ofCorrect (currently valid attribute),Outdated (historically valid but no longer current), andIrrelevant
(neither current nor historically valid) responses using theupper-boundstrategy. The results can be compared with
the performance of the corresponding LLM prior to multimodal (pre-)training/instruction tuning (LLM* Textual
Prompt).
zations (i.e., companies and institutions).
Visual PromptsTo evaluate factual knowledge
across modalities, we construct paired queries in
which the same subject entity is presented either
textually or visually. In the textual setting, the sub-
ject appears explicitly in the prompt (e.g.,Who
is the CEO of Apple?). In the visual setting, the
textual subject is replaced by the corresponding
image of the entity obtained from Wikidata (e.g.,
Who is the CEO of this company?). This formula-
tion enables a direct comparison of factual recall
when entities are accessed through linguistic versus
visual representation. The visual representations
consist of coats of arms and flags for countries,
portraits for athletes, and logos for organizations.
For countries, we include the coat of arms to help
mitigate potential ambiguity, as certain flags share
similar color patterns and may become difficult to
distinguish under transformations2. The prompts
are presented in § Table 6.
Visual Entity RecognitionVisual queries intro-
duce an additional challenge compared to text-only
prompts, since the model must first identify the
depicted entity and then retrieve the correspond-
ing attribute from its internal knowledge. When a
model produces an incorrect answer, it is therefore
unclear whether the error arises from a failure of
visual recognition or from missing factual knowl-
edge. To disentangle these factors, we introduce
an auxiliary recognition task, where models are
prompted to identify the depicted entity. Prompt
templates for the detection task are generated using
the same procedure as the factual queries, presented
2For example, the flag of the Netherlands, rotated by 90° re-
sembles the French flag. The coat of arms, therefore, provides
an additional visual cue that uniquely identifies the country.in § Table 7.
ProtocolFor each query, the model’s response is
compared against the attribute collected at the eval-
uation time, and classified asCorrectif it matches
the currently valid attribute. It is labeledOutdated
if it corresponds to an attribute that was valid in the
past. Finally, a response is categorized asIrrele-
vantif it matches neither the current nor any pre-
viously valid attribute associated with the queried
property. This fine-grained distinction of the er-
rors enables a more precise characterization of how
VLMs encode and retrieve time-sensitive informa-
tion.
StrategyGenerative models are known to be
sensitive to prompt lexicalization, where different
versions of the same query may lead to different
responses despite identical semantics. To address
this limitation, V-DyKnow evaluates each factual
query using three prompts with minor perturba-
tions in lexicalization while preserving the same
semantics (presented in § Table 6). Each model is
queried independently with the three prompts, and
the responses are evaluated under theUpper-Bound
strategy, where the final prediction for a query cor-
responds to the best-performing response among
the three prompts (Correct > Outdated > Irrelevant).
This procedure reduces the likelihood that evalua-
tion outcomes are driven by prompt-specific arti-
facts rather than by the model’s underlying factual
knowledge.
ModelsWe evaluate 9 VLMs (and their corre-
sponding LLM): LLaV A-1.5 7B (Vicuna-v1.5 7B)
(Liu et al., 2024) LLaV A-OneVision 7B (Qwen2
7B) (Li et al., 2025), PaliGemma 2 10B (Gemma 2
9B) (Steiner et al., 2024), Molmo 7B (OLMo 7B)
(Deitke et al., 2025), Qwen2-VL 7B (Qwen2 7B)
3

(Year) Model Accuracy
(2023) LLaV A-1.5 69%
(2024) LLaV A-OneVision 80%
(2024) PaliGemma 2 63%
(2024) Molmo 60%
(2024) Qwen2-VL 91%
(2025) Qwen2.5-VL 87%
(2025) InternVL3.5 58%
(2025) GPT-4 83%
(2025) GPT-5 75%
Table 2: Accuracy on the Visual Entity Recognition
task. Models are prompted to identify the entity de-
picted in each image before answering the factual query.
Scores are computed using anUpper-Boundacross three
prompt variants (presented in § Table 6).
(Wang et al., 2024a), Qwen2.5-VL 7B (Qwen2.5
7B) (Qwen et al., 2025), InternVL3.5 8B (Qwen3
8B) (Wang et al., 2025), GPT-43, and GPT-54. Ad-
ditional details about the model’s checkpoints are
presented in § Table 8
3 Results
We assess the time-sensitive knowledge in state-of-
the-art VLMs on V-DyKnow. We analyze how fre-
quently models produce outdated responses, their
performance under minor perturbations, and the
effectiveness of existing methods for updating their
factual knowledge.
A. Model Evaluation
Table 1 reports the performance of nine state-of-
the-art VLMs on V-DyKnow5. Across models, out-
dated factual responses appear frequently, often
exceeding the number of correct answers. Many
models retrieve the knowledge that no longer re-
flects the current state of the world. For most mod-
els, factual retrieval is substantially more reliable
when the entity is provided textually. This gap is
more evident in LLaV A-1.5, Molmo, InternVL3.5,
and the Qwen family, resulting in a large propor-
tion of irrelevant responses upon visual stimuli.
This gap suggests that factual knowledge accessi-
ble through textual queries is not always accessi-
ble when the same entity must first be recognized
from an image, highlighting a limitation in the in-
teraction between visual recognition and factual
recall. While more recent models (e.g., Qwen2.5-
VL, GPT-{4,5}) achieve higher rates of correct an-
3gpt-4.1-2025-04-14
4gpt-5.1-2025-11-13
5The models are assessed against the factual knowledge
retrieved on Nov 2025 from Wikidata.(Year) ModelPrompt Agreement
Visual Textual
(2023) LLaV A-1.5 52% 70%
(2024) LLaV A-OneVision 66% 80%
(2024) PaliGemma 2 25% 100%
(2024) Molmo 57% 81%
(2024) Qwen2-VL 46% 79%
(2025) Qwen2.5-VL 70% 78%
(2025) InternVL3.5 58% 75%
(2025) GPT-4 92% 97%
(2025) GPT-5 84% 90%
Table 3: Output consistency (prompt agreement) across
three different lexicalizations of the same query forVi-
sualandTextualinputs. An agreement measures the
percentage of cases for which all prompts produce the
same answer (i.e., correct, outdated, or irrelevant).
swers, outdated responses continue to constitute a
substantial fraction of model outputs. Notably, the
gap between visual and textual prompting is sub-
stantially reduced in the most recent models. Com-
paring VLMs with their corresponding base LLM
reveals that multimodal training often degrades fac-
tual recall. Except for LLaV A-1.5, the textual per-
formance of the VLM is consistently lower than its
corresponding LLM, suggesting that multimodal
alignment can obscure knowledge that is other-
wise accessible in the underlying language model.
PaliGemma 2 represents an extreme case: after
multimodal alignment, the model outputs “unan-
swerable” to most of our queries.
Visual Entity RecognitionThe results, reported
in Table 2, indicate that models with a smaller gap
between visual and textual factual retrieval exhibit
a strong recognition performance. The Qwen2
model family achieves entity recognition accuracy
above 85%, and correspondingly shows a moderate
difference between visual and textual prompting
in Table 1. In contrast, models with weaker en-
tity recognition performance such as LLaV A-1.5,
Molmo, and InternVL3.5 tend to display substan-
tially larger modality gaps, with higher proportions
of irrelevant outputs when the entity is presented vi-
sually. These results suggest that failures in visual
entity recognition contribute to the degradation of
factual retrieval under visual prompting. Neverthe-
less, even models with strong recognition capabil-
ities, such as Qwen2-VL and Qwen2.5-VL, still
produce a large fraction of outdated answers, indi-
cating that correctly identifying the entity does not
guarantee access to up-to-date factual knowledge.
4

(Year) Model# Outdated
FactsKnowledge Editing Multi-modal RAG
WISE GRACE IKE Retrieved Doc. Gold Doc.
(2023) LLaV A-1.5 69 2.9% 5.5% 95.6% 73.5% 84.4%
(2024) Qwen2-VL 83 3.9% 2.4% 100.0% 80.1% 92.8%
Table 4: Effectiveness of existing alignment methods (editing and multimodal RAG) on outdated facts for LLaV A-
1.5 and Qwen2-VL. Scores report the harmonic mean of efficacy and paraphrase success. We compare knowledge
editing methods with multimodal RAG using either retrieved documents as well as gold documents.
B. Output Consistency
We evaluate the consistency of model outputs
across different lexicalizations of the same query as
prompt agreement (Mousavi et al., 2025). Across
models, presented in Table 3, agreement is consis-
tently higher for textual prompts, indicating that
predictions are more sensitive to prompt formula-
tion when the entity must first be inferred from an
image. This pattern is particularly pronounced for
models such as LLaV A-1.5, Molmo, and Qwen2-
VL. In contrast, models such as GPT-4 and GPT-5
exhibit high agreement across both modalities, in-
dicating more stable retrieval behavior. An extreme
case is PaliGemma, which consistently generates
irrelevant answers. Overall, these results indicate
that factual retrieval in VLMs is not only modality-
dependent but also sensitive to prompt formaliza-
tion.
C. Updating VLMs’ Knowledge
The precedence of outdated factual knowledge in
VLMs,Table 1, motivates evaluating whether ex-
isting alignment approaches can effectively update
the factual knowledge encoded in VLMs. We
evaluate knowledge editing methods, aiming to
update factual knowledge without retraining the
full model. We consider three representative ap-
proaches:GRACE(Hartvigsen et al., 2023), which
stores edits in an external memory module applied
through an adaptor;WISE(Wang et al., 2024b),
which extends GRACE by also storing pre-trained
knowledge in an external memory module; and
IKE(Zheng et al., 2023), which introduces updated
examples through in-context learning. We compare
the editing methods with multimodal Retrieval-
Augmented Generation (RAG) (Das et al., 2025)
under two settings: a) a realistic setting where doc-
uments are retrieved based on the input question;
and b) an oracle setting where the gold document
containing the answer is provided directly to the
model. We focus on LLaV A-1.5 and Qwen2-VL,
as the old models with the highest percentage ofoutdated responses included in Table 1. We mea-
sure the effectiveness of each approach using the
harmonic mean of efficacy and paraphrase success
(Mousavi et al., 2024, 2025). Additional implemen-
tation details are presented in Sections §A.3, §A.4,
and §A.5
Table 4 shows that performance varies substan-
tially across methods. Among the knowledge edit-
ing methods, only IKE achieves a high harmonic
mean, whereas WISE and GRACE perform poorly
(both below 6%). However, IKE is not fully re-
alistic, as it requires the deterministic up-to-date
fact for each query, whereas facts should be re-
trieved based on the question (as done in mul-
timodal RAG). Retrieval-based approaches show
more consistent performance. Multimodal RAG
achieves high scores across both models, although
performance depends on retrieval quality, as re-
flected by the gap between retrieved and gold doc-
uments. Importantly, both RAG and IKE rely on
external information at inference time and there-
fore do not update the model’s parametric knowl-
edge. As a result, both methods may still suffer
from inconsistencies between the model’s internal
knowledge and the external information provided
at inference time (Wu et al., 2024).
Qualitative AnalysisTo better understand the
impact of alignment approaches on the model out-
puts, we annotate the generated responses using a
human evaluation protocol adapted from Mousavi
et al. (2022). Two researchers independently re-
viewed the outputs produced after editing/RAG.
Responses are annotated asCorrectif a previously
outdated answer is updated successfully. Mean-
while,Incorrectoutputs are further subdivided into
a)Outdated, when the model continues to return out-
dated knowledge; b)Generic, if the model produces
vague statements (e.g., “The CEO of Apple is a
very important person”); and c)Hallucination, when
the model fabricates or confuses attributes, such as
attributing political roles to athletes.
The results in Table 5 further confirm that WISE
5

Model CorrectIncorrect
OutdatedGenericHalluc.
LLaVA-1.5
WISE 3% 73% 0% 24%
GRACE 21% 63% 1% 15%
IKE 95% 0% 5% 0%
RAG
Ret. Doc74% 2% 0% 23%
Gold Doc85% 3% 0% 12%
Qwen2-VL
WISE 4% 34% 3% 57%
GRACE 34% 51% 0% 14%
IKE 100% 0% 0% 0%
RAG
Ret. Doc80% 5% 0% 15%
Gold Doc93% 3% 0% 4%
Table 5: Qualitative error analysis of knowledge editing
(WISE, GRACE, IKE) and retrieval-based alignment
methods across two representative LLMs, based on hu-
man annotation of generated outputs. Responses are
categorized asCorrectwhen the outdated attribute is
successfully updated. Incorrect responses are further
classified asOutdatedwhen the model continues to
return the outdated fact,Genericwhen the model pro-
duces vague answers, andHallucinatedwhen the model
fabricates or confuses attributes.
and GRACE fail to reliably update the models’ fac-
tual knowledge, as the majority of their responses
remain outdated even after editing. Both methods
increase the proportion of hallucinated responses,
with models either assigning incorrect attributes
to a given subject-property pair (e.g., the emperor
of Japan is Barack Obama) or answering with the
name of the visual subject (e.g.,“Can you name
the current prime minister of this country? Italy”).
IKE produces few generic outputs, such as answer-
ing“yes”to questions like“can you name... ”. Fi-
nally, even when provided with gold documents,
RAG can still produce outdated or hallucinated
answers, confirming that the model’s pre-trained
parametric knowledge can interfere with external
evidence.
4 Further Analysis
We conduct additional analyses to better under-
stand the sources of outdated responses. First, we
examine the temporal validity intervals associated
with model predictions to approximate the time pe-
riod reflected in their parametric knowledge. We
then analyze the training data of a representative
model to determine whether the information re-
quired to answer a query was present in the pre-
training corpus. Finally, we perform a mechanisticanalysis to study how factual information is re-
trieved across model layers.
A. Data Interval Approximation
Using the temporal validity intervals associated
with Wikidata attributes, we analyze the correct
and outdated outputs of each model to approximate
the temporal interval reflected in their parametric
knowledge. For each predicted attribute, we re-
trieve its validity interval and aggregate the distri-
bution of these intervals across model outputs. The
results are summarized in Figure 2. Our approxima-
tions are in line with the cutoffs of the models with
disclosed data intervals such as Qwen2-VL (June
2023)6, GPT-4 (June 1st 2024) and GPT-5 (Septem-
ber 30th 2024) API7. Although all evaluated mod-
els were released after 2023, most responses cor-
respond to facts valid before 2020, suggesting that
the underlying training data predominantly reflects
earlier states of the world. Within several model
families, newer versions tend to generate responses
associated with more recent intervals. For exam-
ple, LLaV A-OneVision and Qwen2.5-VL produce
fewer responses tied to earlier years than LLaV A-
1.5 and Qwen2-VL, respectively. Across open-
source models, the temporal intervals roughly starts
from 2005. GPT-{4,5} produce responses concen-
trated in more recent intervals, primarily between
2017 and 2023, although outdated predictions can
still extend to earlier periods. Notably, both GPT
models show nearly identical temporal distribu-
tions, suggesting that they rely on similar training
data snapshots. Overall, these patterns indicate that
most of the model outputs and presumably thier
training data reflect an outdated state of the world,
even though more recent models tend to incorpo-
rate more recent information.
B. Linking Responses to Training Data
To understand how VLMs’ predictions are shaped
by the information available during pretraining, we
analyze the training corpus for a stratified subset
of entities. We conduct this analysis using Molmo,
as its pretraining data is publicly available and
therefore allows direct inspection of the documents
the model has been exposed to. We focus on the
Wikipedia portion of the corpus, as prior work sug-
gests that models tend to acquire factual knowledge
more reliably from structured and curated sources
such as Wikipedia compared to other documents
6Qwen AI Chat Models
7OpenAI API Models
6

2004 2006 2008 2010 2012 2014 2016 2018 2020 2022 2024 2026(2025) GPT -5(2025) GPT -4(2025) InternVL3.5(2025) Qwen2.5-VL(2024) Qwen2-VL(2024) Molmo(2024) PaliGemma 2(2024) LLaV A-OneV ision(2023) LLaV A-1.5Figure 2: Temporal distribution of the model responses based on Wikidata. For each VLM, we map their correct
and outdated responses (e.g., “The CEO of Apple is Steve Jobs ”) to the time at which the corresponding attribute
was valid (e.g., “1997-2011”). By aggregating these intervals using a boxplot, we can approximate the state of
the world encoded in the model’s parameters. For example, results show that, while Qwen2-VL responses range
between 2004 and 2023 (in line with its reported cutoff), most of them are concentrated between 2013 and 2019.
in the pre-training snapshots(Li et al., 2024). For
each entity, we analyze the main Wikipedia page
included in the pretraining snapshot and determine
whether the current fact appears in the document
and how prominently it is mentioned. Our goal
is not to exhaustively search all documents in the
corpus, but to determine whether the correct infor-
mation was present in the canonical source at the
time the training snapshot was collected. If the cor-
rect information appears in this version of the page,
it is likely that the model was exposed to it during
pretraining; conversely, if it is absent, the model
may have only observed the outdated attribute or
no relevant information at all.
We select 30 entities, 10 cases where Molmo gen-
erates the correct attributes, 10 where it produces
an outdated attribute, and 10 where it generates an
irrelevant response. For entities where Molmo pro-
duces the correct answer, in five cases the correct
fact does not appear in the document, suggesting
that the model must have acquired this informa-
tion from other sources in its pretraining data. In
one case, the correct attribute is mentioned but not
as the most frequent value, while in four cases, it
is both present and the most frequently occurring
information in the page. In cases where Molmo
produces outdated answers, in seven cases the cur-
rent information does not appear in the Wikipedia
page, indicating that the model’s prediction is con-
sistent with the information available in the docu-
ment at the time of pretraining. However, in three
cases, the current attribute is present and even themost frequently occurring value in the page, yet
the model still retrieves the outdated one. This sug-
gests that outdated predictions cannot always be
explained by missing information in the training
data. Finally, for entities where Molmo produces
irrelevant responses, the current attribute is absent
from the Wikipedia page in six cases. In the re-
maining four cases, the correct face appears in the
document, twice as a secondary mention and twice
as the most frequent value, yet the model still fails
to retrieve it. Taken together, these observations in-
dicate that model behavior is not determined solely
by the presence or frequency of information in the
training data.
C. Mechanistic Interpretability
We conduct a mechanistic analysis to examine how
different editing methods modify the retrieval of
factual information across model layers. Building
on prior work (Mousavi et al., 2025; Geva et al.,
2023; Ou et al., 2025), we probe the model with
visual queries and analyze how different layers con-
tribute to the prediction of updated and outdated
attributes. Figure 3 presents the results of this anal-
ysis for Qwen2-VL. Additional figures and detailed
analyses are provided in Section §A.6. Overall, we
observe that the pre-trained model assigns a high
probability to one of the candidate attributes only
at the final layer, suggesting that earlier layers con-
tribute little to the explicit factual recall reflected
in the final prediction. In case of successful ed-
its (the upper part in Figure 3), while WISE only
7

Pre-T rained WISE GRACE
L. Montenegro 2024-Now
A. Costa 2015-2024
Pedro P . Coelho 2011-2015
20 22 24 26 28
Philadelphia 2024-Now
City Thunder 2017-2019LA Clippers 2019-2024
Indiana Pacers 2010-2017
1
0.8
0.6
0.4
0.2
0
1
0.8
0.6
0.4
0.2
020 22 24 26 28 20 22 24 26 28
20 22 24 26 28 20 22 24 26 28 20 22 24 26 28Pre-T rained WISE GRACE Image
Question
What club is this NBA
player curr ently playing
for?
Image
Question
Who is the curr ent Head
of Government of this
country?Figure 3: Mechanistic analysis of Qwen2-VL illustrating how editing methods modify the probability for certain
attributes (y-axis) across different layers (x-axis) when asked about an Image-Question pair. In case of a successful
update (top), WISE and GRACE affect different layers: WISE primarily edits the final layer, whereas GRACE
modifies a broader range. In contrast, neither method successfully updates the basketball team of Paul George
(bottom), suggesting that this knowledge may be more deeply embedded or that other stored facts may interfere.
modifies the final layer, GRACE targets earlier lay-
ers and propagates the change across layers 20 to
28. Meanwhile, in the case of unsuccessful ed-
its (the lower part in Figure 3), GRACE slightly
reduces the probability assigned to the incorrect en-
tity, whereas WISE increases it, potentially due to
interference with other stored factual associations.
5 Literature Review
BenchmarksExisting evaluations of time-sensitive
knowledge have been largely restricted to the tex-
tual inputs. Dynamic benchmarks such as DyKnow
(Mousavi et al., 2024), EvolveBench (Zhu et al.,
2025), and DynaQuest (Lin et al., 2025) have been
introduced to assess how models handle tempo-
rally changing facts, but these are designed exclu-
sively for Large Language Models (LLMs). In
contrast, existing benchmarks with visual queries
(Chen et al., 2023; Cheng et al., 2025) are largely
static, making it difficult to assess the validity of
time-sensitive facts under visual grounding.
VLMsCurrent research suggests that VLMs suf-
fer from significant representational discrepancies,
making them unreliable repositories of knowledge
(Huh et al., 2024). Recent studies identify a huge
performance gap across modalities, with high per-
formance with textual queries Cohen et al. (2025),
indicating the VLMs fail to utilize the visual sig-
nals effectively Fu et al. (2025). This inconsistency
has also been observed in video and speech (Papiet al., 2026), and in multimodal knowledge edit-
ing (Fang et al., 2025), where edits fail to transfer
across modalities. Meanwhile, multimodal knowl-
edge editing studies rely on static ground truth and
do not take in consideration the temporal validity
of facts or factual updates over time (Fang et al.,
2025; Chen et al., 2025).
6 Discussion & Conclusion
Our analysis shows that outdated knowledge is
common across both open- and closed-source
VLMs, reflecting old training data snapshots even
in recent models. These findings raise broader ques-
tions about how large multimodal models should
represent knowledge about a world that continu-
ously evolves. Current training paradigms rely
on static data snapshots, while existing alignment
methods remain insufficient for maintaining con-
sistent and up-to-date factual knowledge. Address-
ing these challenges may require new learning
paradigms that explicitly model temporal validity,
integrate dynamic knowledge sources, and support
continual updates without disrupting existing rep-
resentations. We believe V-DyKnow provides a
useful resource for studying these questions. By
explicitly modeling temporal validity and support-
ing multimodal evaluation, the benchmark enables
systematic analysis of how models acquire, store,
and update real-world knowledge. We release the
benchmark, code, and evaluation data to facilitate
8

future research on dynamic factual knowledge in
vision–language models.
Limitations
The 139 time-sensitive facts obtained from Wiki-
data refer to frequent entities. Consequently, the
results should be interpreted as upper bounds, as
factual recall for less common entities may be more
difficult and could lead to more outdated or irrel-
evant results. Evaluation of editing methods only
takes into account the effectiveness of updating
time-sensitive facts, but the potential side effects
of editing are not evaluated in this work. Due to
limited resources, only open-source models with
7B parameters have been evaluated. As an upper
bound on the performance of current VLMs for fac-
tual recall, we evaluated closed-source API models.
References
Qizhou Chen, Chengyu Wang, Dakan Wang, Taolin
Zhang, Wangyue Li, and Xiaofeng He. 2025. Life-
long knowledge editing for vision language models
with low-rank mixture-of-experts. InProceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 9455–9466.
Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit
Changpinyo, Alan Ritter, and Ming-Wei Chang. 2023.
Can pre-trained vision and language models answer
visual information-seeking questions? InProceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 14948–14968,
Singapore. Association for Computational Linguis-
tics.
Xianfu Cheng, Wei Zhang, Shiwei Zhang, Jian Yang,
Xiangyuan Guan, Xianjie Wu, Xiang Li, Ge Zhang,
Jiaheng Liu, Yuying Mai, Yutao Zeng, Zhoufutu Wen,
Ke Jin, Baorui Wang, Weixiao Zhou, Yunhong Lu,
Hangyuan Ji, Tongliang Li, Wenhao Huang, and
Zhoujun Li. 2025. Simplevqa: Multimodal factuality
evaluation for multimodal large language models. In
Proceedings of the IEEE/CVF International Confer-
ence on Computer Vision (ICCV), pages 4637–4646.
Ido Cohen, Daniela Gottesman, Mor Geva, and Raja
Giryes. 2025. Performance gap in entity knowledge
extraction across modalities in vision language mod-
els. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 29095–29108, Vienna, Austria.
Association for Computational Linguistics.
Deepayan Das, Davide Talon, Yiming Wang, Massi-
miliano Mancini, and Elisa Ricci. 2025. Training-
free personalization via retrieval and reasoning on
fingerprints. InProceedings of the IEEE/CVF In-
ternational Conference on Computer Vision, pages
9683–9692.Matt Deitke, Christopher Clark, Sangho Lee, Rohun
Tripathi, Yue Yang, Jae Sung Park, Mohammadreza
Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini,
Jiasen Lu, Taira Anderson, Erin Bransom, Kiana
Ehsani, Huong Ngo, YenSung Chen, Ajay Patel,
Mark Yatskar, Chris Callison-Burch, and 31 others.
2025. Molmo and pixmo: Open weights and open
data for state-of-the-art vision-language models. In
Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages
91–104.
Lingyong Fang, Xinzhong Wang, Depeng Wang, Zon-
gru Wu, Ya Guo, Huijia Zhu, Zhuosheng Zhang, and
Gongshen Liu. 2025. Can knowledge be transferred
from unimodal to multimodal? investigating the tran-
sitivity of multimodal knowledge editing. InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), pages 2482–2490.
Stephanie Fu, tyler bonnen, Devin Guillory, and Trevor
Darrell. 2025. Hidden in plain sight: VLMs overlook
their visual representations. InSecond Conference
on Language Modeling.
Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir
Globerson. 2023. Dissecting recall of factual associa-
tions in auto-regressive language models. InProceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 12216–12235,
Singapore. Association for Computational Linguis-
tics.
Google Cloud. Gemini 3.1 pro: Generative ai
on vertex ai. https://docs.cloud.google.
com/vertex-ai/generative-ai/docs/models/
gemini/3-1-pro?hl=en.
Tom Hartvigsen, Swami Sankaranarayanan, Hamid
Palangi, Yoon Kim, and Marzyeh Ghassemi. 2023.
Aging with grace: Lifelong model editing with dis-
crete key-value adaptors.Advances in Neural Infor-
mation Processing Systems, 36:47934–47959.
Minyoung Huh, Brian Cheung, Tongzhou Wang, and
Phillip Isola. 2024. The platonic representation hy-
pothesis.Preprint, arXiv:2405.07987.
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng
Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yan-
wei Li, Ziwei Liu, and Chunyuan Li. 2025. LLaV A-
onevision: Easy visual task transfer.Transactions on
Machine Learning Research.
Jiahuan Li, Yiqing Cao, Shujian Huang, and Jiajun Chen.
2024. Formality is favored: Unraveling the learning
preferences of large language models on data with
conflicting knowledge. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5307–5320.
Qian Lin, Junyi Li, and Hwee Tou Ng. 2025. Dy-
naQuest: A dynamic question answering dataset
reflecting real-world knowledge updates. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2025, pages 26918–26936, Vienna, Austria.
Association for Computational Linguistics.
9

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae
Lee. 2024. Improved baselines with visual instruc-
tion tuning. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition
(CVPR), pages 26296–26306.
Kevin Meng, David Bau, Alex Andonian, and Yonatan
Belinkov. 2022. Locating and editing factual associa-
tions in gpt.Advances in neural information process-
ing systems, 35:17359–17372.
Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe
Riccardi. 2024. DyKnow: Dynamically verifying
time-sensitive factual knowledge in LLMs. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 8014–8029, Miami, Florida,
USA. Association for Computational Linguistics.
Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe
Riccardi. 2025. Llms as repositories of factual knowl-
edge: Limitations and solutions.arXiv preprint
arXiv:2501.12774.
Seyed Mahed Mousavi, Gabriel Roccabruna, Michela
Lorandi, Simone Caldarella, and Giuseppe Riccardi.
2022. Evaluation of response generation models:
Shouldn’t it be shareable and replicable? InPro-
ceedings of the 2nd Workshop on Natural Language
Generation, Evaluation, and Metrics (GEM), pages
136–147.
OpenAI. 2026a. GPT-4.1 Model | OpenAI
API. https://developers.openai.com/api/
docs/models/gpt-4.1.
OpenAI. 2026b. GPT-5.1 Model | OpenAI
API. https://developers.openai.com/api/
docs/models/gpt-5.1.
Yixin Ou, Yunzhi Yao, Ningyu Zhang, Hui Jin, Ji-
acheng Sun, Shumin Deng, Zhenguo Li, and Huajun
Chen. 2025. How do LLMs acquire new knowledge?
a knowledge circuits perspective on continual pre-
training. InFindings of the Association for Computa-
tional Linguistics: ACL 2025, pages 19889–19913,
Vienna, Austria. Association for Computational Lin-
guistics.
Sara Papi, Maike Züfle, Marco Gaido, beatrice savoldi,
Danni Liu, Ioannis Douros, Luisa Bentivogli, and
Jan Niehues. 2026. MCIF: Multimodal crosslin-
gual instruction-following benchmark from scientific
talks. InThe Fourteenth International Conference on
Learning Representations.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Massimo Rizzoli, Simone Alghisi, Olha Khomyn,
Gabriel Roccabruna, Seyed Mahed Mousavi, andGiuseppe Riccardi. 2025. CIVET: Systematic evalu-
ation of understanding in VLMs. InFindings of the
Association for Computational Linguistics: EMNLP
2025, pages 4462–4480, Suzhou, China. Association
for Computational Linguistics.
Andreas Steiner, André Susano Pinto, Michael Tschan-
nen, Daniel Keysers, Xiao Wang, Yonatan Bit-
ton, Alexey Gritsenko, Matthias Minderer, Anthony
Sherbondy, Shangbang Long, and 1 others. 2024.
Paligemma 2: A family of versatile vlms for transfer.
arXiv preprint arXiv:2412.03555.
Denny Vrande ˇci´c and Markus Krötzsch. 2014. Wiki-
data: a free collaborative knowledgebase.Commun.
ACM, 57(10):78–85.
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi-
hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, and 1 others. 2024a. Qwen2-
vl: Enhancing vision-language model’s perception
of the world at any resolution.arXiv preprint
arXiv:2409.12191.
Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi
Yao, Yong Jiang, Pengjun Xie, Fei Huang, and Hua-
jun Chen. 2024b. Wise: Rethinking the knowledge
memory for lifelong model editing of large language
models.Advances in Neural Information Processing
Systems, 37:53764–53797.
Peng Wang, Ningyu Zhang, Xin Xie, Yunzhi Yao,
Bozhong Tian, Mengru Wang, Zekun Xi, Siyuan
Cheng, Kangwei Liu, Guozhou Zheng, and 1 others.
2023. Easyedit: An easy-to-use knowledge editing
framework for large language models.arXiv preprint
arXiv:2308.07269.
Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu,
Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin
Jing, Shenglong Ye, Jie Shao, and 1 others. 2025. In-
ternvl3. 5: Advancing open-source multimodal mod-
els in versatility, reasoning, and efficiency.arXiv
preprint arXiv:2508.18265.
Kevin Wu, Eric Wu, and James Zou. 2024. Clasheval:
Quantifying the tug-of-war between an llm’s inter-
nal prior and external evidence.Advances in neural
information processing systems, 37:33402–33422.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong
Wu, Jingjing Xu, and Baobao Chang. 2023. Can
we edit factual knowledge by in-context learning?
InProceedings of the 2023 Conference on Empiri-
cal Methods in Natural Language Processing, pages
4862–4876.
Zhiyuan Zhu, Yusheng Liao, Zhe Chen, Yuhao Wang,
Yunfeng Guan, Yanfeng Wang, and Yu Wang. 2025.
10

EvolveBench: A comprehensive benchmark for as-
sessing temporal awareness in LLMs on evolving
knowledge. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 16173–16188, Vi-
enna, Austria. Association for Computational Lin-
guistics.
11

A Appendix
A.1 Benchmark Construction
We complement the information in the main paper
by providing additional details on the construction
of our evaluation benchmark (discussed in Section
2).
A.1.1 Visual Queries
For each subject category, we retrieved images
from the following Wikidata (Vrande ˇci´c and
Krötzsch, 2014) fields: portrait photographs (P18)
for athletes; company or institutional logos (P154)
for organizations; national flags (P41) and coats of
arms (P94) for countries. As prior research indi-
cates that image dimensions can significantly influ-
ence model accuracy (Rizzoli et al., 2025), we stan-
dardize all images to a fixed resolution of 672×672
pixels. During this preprocessing phase, uniform
black padding was applied where necessary to pre-
serve the target dimensions without distorting the
original aspect ratio. Images with transparent back-
grounds were rendered with a white background.
Following the image standardization step, we
generated prompt templates in which the subject
entity is represented by its corresponding image.
To construct these queries, we prompted Google
Gemini Pro (Google Cloud) to generate five con-
cise question templates for each subject category.
Each template is formulated as a question asking
for a specificattribute(e.g., “Tim Cook”) of the
subjectentity depicted in the image (e.g., “Apple”).
While the templates differ in their wording, they
are intended to express the same underlying query
so that they lead to the same attribute value as the
answer. To ensure this semantic equivalence, two
researchers from our group manually reviewed the
generated templates, verifying that each formula-
tion queried the same property and would therefore
produce the same attribute as the expected answer.
Based on this review, three templates were selected
as the final prompts used to query the models. The
complete list of prompt templates for our visual
queries is reported in Table 6.
A.1.2 Textual Queries
For each prompt template used for the visual
queries, we derived its corresponding textual coun-
terpart by replacing the part referring the image
(e.g., “this country”, “this player”, or “this organi-
zation”) with the explicit textual name of the sub-
ject entity (e.g., “this country” →“Italy”, or “thisorganization” →“Apple”). To guarantee that these
modifications did not alter the semantic meaning of
the queries, two researchers manually verified the
resulting textual prompts, ensuring they remained
grammatically correct and accurately reflected the
original output response of the visual queries.
A.1.3 Visual Entity Recognition Prompts
Following the construction of the visual prompts,
we leveraged Google Gemini Pro (Google Cloud)
to generate five distinct templates that prompt the
models to identify the visual subject entity. Two re-
searchers manually reviewed the generated prompts
and selected three for each subject category. The
complete set of detection prompt templates is pro-
vided in Table 7.
A.2 Model Checkpoints and Inference Details
To ensure reproducibility, we detail the model
checkpoints, API versions, hardware, and genera-
tion hyperparameters.
A.2.1 Model Checkpoints
We provide the lists of the HuggingFace repos-
itory identifiers for the open-weight VLMs and
their corresponding LLMs in Table 8. LLMs
baselines were selected to match the base mod-
els utilized during VLM training. For closed-
source models, we queried the OpenAI using
thegpt-4.1-2025-04-14 (OpenAI, 2026a) and
gpt-5.1-2025-11-13(OpenAI, 2026b) models.
A.2.2 Inference Hyperparameters
To ensure deterministic responses, we employed
greedy decoding for all open-weight models. To re-
duce verbose responses, we append the instruction
“Answer with the name only”, following prior find-
ings (Rizzoli et al., 2025), and set the maximum
number of tokens to 20.
A.2.3 Hardware
Open-weight model inference was executed locally
on a single NVIDIA RTX 3090 (24GB) GPU using
bfloat16 precision. Image pre-processing was
handled using the default pipeline of each VLM.
A.3 Editing Methods Implementation Details
We present more details on the editing methods
used in the paper:
1.WISE(Wang et al., 2024b) introduces a
dual-memory architecture that partitions new
knowledge in a side memory (to store new
12

Subject CategorySubject EntityProperty Prompt Template with [Place Holder]
CountriesItaly, United States of America, China, Germany, Japan, India, United Kingdom, France, Brazil, Canada, Russia, Mexico, South Korea, Australia, Spain, Indonesia, Turkey, Netherlands, Saudi Arabia, Poland, Belgium, Argentina, Sweden, Ireland, Norway, Austria, Israel, Thailand, United Arab Emirates, Singapore, Bangladesh, Philippines, Vietnam, Malaysia, Denmark, Egypt, Nigeria, South Africa, Iran, Colombia, Romania, Chile, Pakistan, Czech Republic, Finland, Iraq, Portugal(Head of State)PresidentKingMonarchEmperorSupreme LeaderCoat of Amrs❏Who is the current [Head of State/ Head of Gov.] of this country?❏Can you name the current [Title] of this country?❏Which person currently serves as [Title] for this country?(Head of Gov.)Prime MinisterPremier of RepublicFederal ChancellorFlag
AthletesSoccer PlayerCristiano Ronaldo, Lionel Messi Neymar Jr., Kylian Mbappé, Karim Benzema, Erling Haaland, Mohamed Salah, Sadio Mané, Kevin De Bruyne,Harry KaneSports Team❏For which club does this player currently play for?❏What is the current football club of this athlete?❏What club is this football player currently playing for?Basketball Player (NBA)Stephen Curry, Kevin Durant, LeBron James, Nikola Jokic, Bradley Beal, Giannis Antetokounmpo, Damian Lillard, Kawhi Leonard, Paul George❏For which club does this player currently play for?❏What is the current basketball  club of this athlete?❏What club is this NBA player currently playing for?F1 DriverMax Verstappen, Lewis Hamilton, Fernando Alonso, Sergio Pérez, Charles Leclerc, Lando Norris, Carlos Sainz Jr., George Russell, Pierre Gasly❏For which club does this player currently play for?❏What is the current F1 team of this athlete?❏What team is this F1 driver currently racing for?
Private / Public Organi-zationsCompanyWalmart, Saudi Aramco, Amazon, ExxonMobil, Apple, Shell, CVS Health, Volkswagen Group, Alphabet Inc., Toyota, TotalEnergies, Glencore, BP, Cencora, Inc., Microsoft, Gazprom, Mitsubishi, Ford Motor CompanyCEO❏Who is the current [Title] of  this company?❏Can you name the current [Title] of this company?❏Who currently holds the position [Title] of  of this company?ChairpersonOrgani-zationUnited Nations, World Bank, North Atlantic Treaty Organization, International Olympic CommitteeGeneral secretary❏Who is the current [Title] of this orgamization?❏Who is the current [Title]  of this organization?❏Who currently holds the position of [Title] of this organization?ChairpersonTable 6: List of all subject entities and their corresponding visual prompts used for benchmarking the VLMs.
13

Subject CategorySubject EntityProperty Prompt Template with [Place Holder]
CountriesItaly, United States of America, China, Germany, Japan, India, United Kingdom, France, Brazil, Canada, Russia, Mexico, South Korea, Australia, Spain, Indonesia, Turkey, Netherlands, Saudi Arabia, Poland, Belgium, Argentina, Sweden, Ireland, Norway, Austria, Israel, Thailand, United Arab Emirates, Singapore, Bangladesh, Philippines, Vietnam, Malaysia, Denmark, Egypt, Nigeria, South Africa, Iran, Colombia, Romania, Chile, Pakistan, Czech Republic, Finland, Iraq, Portugal(Head of State)PresidentKingMonarchEmperorSupreme LeaderCoat of Amrs❏Which country does this image represent?❏To which nation does this symbol belong to?❏Which country officially uses this as a [Coat of Arms/ Flag]?(Head of Gov.)Prime MinisterPremier of RepublicFederal ChancellorFlag
AthletesSoccer PlayerCristiano Ronaldo, Lionel Messi Neymar Jr., Kylian Mbappé, Karim Benzema, Erling Haaland, Mohamed Salah, Sadio Mané, Kevin De Bruyne,Harry KaneSports Team❏Who is the athlete featured here?❏What is this athlete’s name?❏Who is the soccer player shown in this image?Basketball Player (NBA)Stephen Curry, Kevin Durant, LeBron James, Nikola Jokic, Bradley Beal, Giannis Antetokounmpo, Damian Lillard, Kawhi Leonard, Paul George❏Who is the athlete featured here?❏What is this athlete’s name?❏Who is the basketball player shown in this image?F1 DriverMax Verstappen, Lewis Hamilton, Fernando Alonso, Sergio Pérez, Charles Leclerc, Lando Norris, Carlos Sainz Jr., George Russell, Pierre Gasly❏Who is the athlete featured here?❏What is this athlete’s name?❏Who is the formula 1 driver shown in this image?
Private / Public Organi-zationsCompanyWalmart, Saudi Aramco, Amazon, ExxonMobil, Apple, Shell, CVS Health, Volkswagen Group, Alphabet Inc., Toyota, TotalEnergies, Glencore, BP, Cencora, Inc., Microsoft, Gazprom, Mitsubishi, Ford Motor CompanyCEO❏Which company is represented by the logo in the image?❏What is the name of the company with this logo?❏Which company does this logo belog to?ChairpersonOrgani-zationUnited Nations, World Bank, North Atlantic Treaty Organization, International Olympic CommitteeGeneral secretary❏Which organization is represented by the logo in the image?❏What is the name of the organization with this logo?❏Which organization does this logo belog to?ChairpersonTable 7: List of all subject entities and their corresponding visual entity recognition prompts used for benchmarking
the VLMs.
14

Model Name VLM Checkpoint LLM Checkpoint
LLaV A-1.5 7Bllava-hf/llava-1.5-7b-hf lmsys/vicuna-7b-v1.5
LLaV A-OneVision 7Bllava-hf/llava-onevision-qwen2-7b-si-hf Qwen/Qwen2-7B-Instruct
PaliGemma 2 10Bgoogle/paligemma2-10b-mix-448 google/gemma-2-9b-it
Molmo 7Ballenai/Molmo-7B-O-0924 allenai/OLMo-7B-Instruct-hf
Qwen2-VL 7BQwen/Qwen2-VL-7B-Instruct Qwen/Qwen2-7B-Instruct
Qwen2.5-VL 7BQwen/Qwen2.5-VL-7B-Instruct Qwen/Qwen2.5-7B-Instruct
InternVL3.5 8BOpenGVLab/InternVL3_5-8B-HF Qwen/Qwen3-8B
Table 8: Official HuggingFace model checkpoints for the open-weight VLMs and their corresponding base LLMs
used in our evaluation.
edits) while maintaining existing pretrained
knowledge in the primary memory. At infer-
ence time, a router is used to decide which
memory to use for an incoming query. If there
is no match between the query and the edit
scope, the primary memory is selected to gen-
erate the final output. In case of a match, the
prompt is routed through the side memory to
generate the response grounded on the edited
knowledge.
2.GRACE(Hartvigsen et al., 2023) introduces
an Adaptor at a chosen model layer to cache
targeted edits while leaving the pretrained
weights frozen. During the editing phase, the
Adaptor saves the hidden activation of the de-
sired editing fact as a key and learns its cor-
rected output as a corresponding value. Each
key is assigned a dynamic search radius that
expands or splits to prevent overlaps and en-
courage generalization to similar inputs. At
inference time, the Adaptor evaluates the la-
tent distance between the new prompt and the
cached keys. If the prompt falls inside a key’s
radius, the Adaptor intercepts the forward pass
and outputs the learned corrected value. If
there is no match, the input passes through the
original pre-trained weights to generate the
final response.
3.IKE(Zheng et al., 2023) introduces an in-
context learning approach to edit a model’s
factual knowledge without altering any of its
pre-trained weights. To apply an edit, the
method retrieves relevant demonstration ex-
amples from a training corpus to guide the
editing of knowledge. These demonstrations
are specifically formatted into three types: i)
copy, to introduce the new fact; ii) update, to
help the model generalize the injected knowl-
edge; and iii) retain, to guide the model in
preserving unrelated facts. These demonstra-Model# Demonstrations (K)
1 2 3
LLaV A-1.5 88.3% 95.6% 0%
Qwen2-VL 100% 100% 99.1%
Table 9: Harmonic mean for IKE with up to K= 3
demonstrations. Performance peaks at K= 2 , while
adding a third demonstration degrades results, suggest-
ing difficulty in retrieving the relevant information from
the context. LLaV A-1.5 is an extreme case, as it ends
up repeatedly generating\ntokens forK= 3.
tions, along with the actual up-to-date fact, are
prepended directly to the user’s target query to
condition the model’s final answer. However,
this technique does not represent a realistic
scenario, as it requires the relevant, up-to-date
fact to be deterministically provided to the
model for each target question.
For WISE and GRACE, we used the implemen-
tations provided in the EasyEdit framework (Wang
et al., 2023), adopting the default configurations
and hyperparameters. As LLaV A-1.5 is not di-
rectly supported by EasyEdit, we adapted the con-
figuration designed for LLaV A-OneVision, which
belongs to a similar model family. Since both
methods require training additional layers and a
router module, we performed the training on a sin-
gle NVIDIA A100 (80 GiB) GPU.
For IKE, we randomly selected up to K= 3
demonstrations from the pool of unedited facts.
The results in Table 9 show that, while using
two demonstrations improves the harmonic mean,
including a third demonstration degrades perfor-
mance. LLaV A-1.5 represents an extreme case: the
model generates only \ntokens, possibly because
the extended context exceeds its effective context
window. Based on these observations, we select
the configuration with K= 2 for the experiments
reported in Table 4, as it achieves the highest har-
monic mean.
15

Model# Documents (K)
1 3 5
LLaV A-1.5 73.5% 44.7% 8.7%
Qwen2-VL 80.1% 69.8% 67.7%
Table 10: Harmonic mean for RAG when considering
K∈ {1,3,5}documents as context.
A.4 Multimodal RAG
We provide additional details about the dataset con-
struction and pipeline used to evaluate multimodal
RAG.
A.4.1 Dataset Construction
We construct a set of image-document pairs for
Qwen2-VL and LLaV A-1.5 by collecting passages
from Wikipedia. For each outdated entity, we ex-
tract passages that contain up-to-date attributes
from its Wikipedia page. These passages are man-
ually verified to ensure they contain the correct
answer, and one passage is selected. We then pair
the selected passage with the main figure from the
corresponding Wikipedia page.
A.4.2 Retrieval and Reranking Pipeline
We implement the multimodal RAG pipeline using
the Qwen3-2B Embedder and Reranker (Zhang
et al., 2025). We encode each image-document pair
using the Qwen3-2B Embedder and retrieve the 10
candidates with the highest cosine similarity based
on the visual query. The retrieved candidates are
then re-ranked using the Qwen3-2B Reranker.
To determine the optimal number of documents
Kto include in the context, we evaluate configu-
rations with K∈ {1,3,5} . Results in Table 10
show that increasing the number of documents in
the context degrades performance for both models,
potentially because additional documents introduce
irrelevant or distracting information. For the ex-
periment reported in Table 4, we therefore select
the configuration with K= 1 , as it achieves the
highest harmonic mean.
A.5 Evaluation Metrics
Following prior work (Mousavi et al., 2024, 2025;
Meng et al., 2022), we assess the effectiveness of
each approach using the harmonic mean of efficacy
and paraphrase success. Given a time-sensitive
question qabout a subject-property pair and its up-
to-date attribute a, efficacy success measures the
proportion of cases in which the edited model Mreturns aforq(i.e.,M(q) =a ). Instead, para-
phrase success measures the proportion of cases in
which Mreturns awhen prompted using a para-
phrased version q′of the same question q(i.e.,
M(q′) =a).
A.6 Mechanistic Analysis
We complement the mechanistic analysis presented
in Section 4 by reporting the results for LLaV A-1.5.
As none of the editing algorithms successfully up-
date the same fact, we select one successful edit
for WISE and one for GRACE and report the re-
sults in Figure 11. In contrast to Qwen2-VL, a
larger number of layers contribute to recalling fac-
tual information, suggesting that such knowledge
may be stored differently across VLMs. Similar
to Figure 3, GRACE propagates the modification
across layers 20 to 32; notably, this method also
significantly reduces the probability of the other
attributes. In contrast, WISE mainly affects layer
29, after which the probability gradually decreases
in subsequent layers. Interestingly, the strongest
effect does not occur at the editing layer specified
in the configuration of WISE, as the method tar-
gets layer 23. Moreover, the information associated
with the other attributes is not completely removed,
suggesting a more localized editing effect.
Figure 12 presents an example of an unsuccess-
ful edit. In this case, both methods show little
to no change compared to the activations of the
pre-trained model. Interestingly, the probability of
the outdated attributes slightly increases for WISE,
similarly to the effect described for Qwen2-VL in
Figure 3.
16

William Racing 2025-Now
Scuderia Ferrari 2021-2024
McLaren 2019-2020
Renault  2017-2018
Toro Rosso 2015-2017
Aston Martin 2023-Now
Renault 2021-2022
McLaren 2015-2018
Scuderia Ferrari 2010-2014Pre-T rained WISE
Pre-T rained
1
0.8
0.6
0.4
0.2
0
20 22 24 26 28 30 32 20 22 24 26 28 30 32
20 22 24 26 28 30 32 20 22 24 26 28 30 32GRACEImage
Question
For which club does this
player curr ently play for?
Image
Question
For which club does this
player curr ently play for?
1
0.8
0.6
0.4
0.2
0Table 11: Mechanistic analysis of LLaV A-1.5 illustrating how editing methods successfully modify the contribution
of different layers during factual recall.
Masoud Pezeshkian 2024-Now
Ebrahim Raisi 2021-2024
Hassan Rouhani 2013-2021
Mohammad Katami 1997-2005
Akbar Hashemi Rafsanjani 1989-1997
Ali Khamenei 1981-1989
1
0.8
0.6
0.4
0.2
0WISE Pre-T rained GRACE
Mahmoud Ahmadinejad 2005-2013Image
Question
Who is the curr ent head
of state of this country?
22 24 26 28 30 32 22 24 26 28 30 32 22 24 26 28 30 32
Table 12: Mechanistic analysis of LLaV A-1.5 illustrating how editing methods fail to modify the contribution of
different layers during factual recall.
17