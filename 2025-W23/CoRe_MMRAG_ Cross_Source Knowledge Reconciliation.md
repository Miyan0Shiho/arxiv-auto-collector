# CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG

**Authors**: Yang Tian, Fan Liu, Jingyuan Zhang, Victoria W., Yupeng Hu, Liqiang Nie

**Published**: 2025-06-03 07:32:40

**PDF URL**: [http://arxiv.org/pdf/2506.02544v2](http://arxiv.org/pdf/2506.02544v2)

## Abstract
Multimodal Retrieval-Augmented Generation (MMRAG) has been introduced to
enhance Multimodal Large Language Models by incorporating externally retrieved
multimodal knowledge, but it introduces two challenges: Parametric-Retrieved
Knowledge Inconsistency (PRKI), where discrepancies between parametric and
retrieved knowledge create uncertainty in determining reliability, and
Visual-Textual Knowledge Inconsistency (VTKI), where misalignment between
visual and textual sources disrupts entity representation. To address these
challenges, we propose Cross-source knowledge \textbf{Re}conciliation for
Multimodal RAG (CoRe-MMRAG), a novel end-to-end framework that effectively
reconciles inconsistencies across knowledge sources. CoRe-MMRAG follows a
four-stage pipeline: it first generates an internal response from parametric
knowledge, then selects the most relevant multimodal evidence via joint
similarity assessment, generates an external response, and finally integrates
both to produce a reliable answer. Additionally, a specialized training
paradigm enhances knowledge source discrimination, multimodal integration, and
unified answer generation. Experiments on KB-VQA benchmarks show that
CoRe-MMRAG achieves substantial improvements over baseline methods, achieving
5.6% and 9.3% performance gains on InfoSeek and Encyclopedic-VQA, respectively.

## Full Text


<!-- PDF content starts -->

arXiv:2506.02544v2  [cs.CL]  4 Jun 2025CoRe-MMRAG: Cross-Source Knowledge Reconciliation
for Multimodal RAG
Yang Tian‡, Fan Liu†, Jingyuan Zhang¶, Victoria W.¶, Yupeng Hu‡*, Liqiang Nie§*
‡School of Software, Shandong University
†National University of Singapore,¶Independent Researcher
§Harbin Institute of Technology, Shenzhen
{tianyangchn,liufancs,nieliqiang}@gmail.com ,huyupeng@sdu.edu.cn
Abstract
Multimodal Retrieval-Augmented Generation
(MMRAG) has been introduced to enhance
Multimodal Large Language Models by in-
corporating externally retrieved multimodal
knowledge, but it introduces two challenges:
Parametric-Retrieved Knowledge Inconsis-
tency (PRKI), where discrepancies between
parametric and retrieved knowledge create un-
certainty in determining reliability, and Visual-
Textual Knowledge Inconsistency (VTKI),
where misalignment between visual and tex-
tual sources disrupts entity representation.
To address these challenges, we propose
Cross-source knowledge Reconciliation for
MultiModal RAG (CoRe-MMRAG), a novel
end-to-end framework that effectively recon-
ciles inconsistencies across knowledge sources.
CoRe-MMRAG follows a four-stage pipeline:
it first generates an internal response from para-
metric knowledge, then selects the most rel-
evant multimodal evidence via joint similar-
ity assessment, generates an external response,
and finally integrates both to produce a reli-
able answer. Additionally, a specialized train-
ing paradigm enhances knowledge source dis-
crimination, multimodal integration, and uni-
fied answer generation. Experiments on KB-
VQA benchmarks show that CoRe-MMRAG
achieves substantial improvements over base-
line methods, achieving 5.6% and 9.3% per-
formance gains on InfoSeek and Encyclopedic-
VQA, respectively. We release code and data at
https://github.com/TyangJN/CoRe-MMRAG.
1 Introduction
Recent advances in Multimodal Large Language
Models (MLLMs) (Alayrac et al., 2022; Li et al.,
2023a; Liu et al., 2023, 2024b; Achiam et al., 2023;
Reid et al., 2024) have significantly improved mul-
timodal reasoning and generation tasks by lever-
aging joint vision-language representations. How-
*Corresponding author
TextCis relevant
December 20, 1957
What is developerofthisplane？
ImageBisrelevant
Parametric-Retrieved Knowledge Inconsistency
What is the date this aircrafttook the first flight?
Visual-Textual Knowledge InconsistencyAugust 29, 1970
Figure 1: Two types of knowledge inconsistency in
MMRAG: (1) Parametric-Retrieved Knowledge Incon-
sistency, where parametric and retrieved external knowl-
edge generate conflicting answers to the same query. (2)
Visual-Textual Knowledge Inconsistency, where mis-
alignments between visual and textual sources disrupt
entity representation.
ever, these models inherently suffer from hallu-
cination (Bai et al., 2024) and knowledge limita-
tions (Caffagni et al., 2024), as their parametric
knowledge is frozen after pretraining and cannot
dynamically adapt to external information.
Multimodal Retrieval-Augmented Generation
(MMRAG) has emerged as a promising approach
to enhance MLLMs by incorporating retrieved tex-
tual and visual knowledge during inference (Yan
and Xie, 2024; Qi et al., 2024; Zhang et al., 2024b).
By accessing external information, MMRAG helps
mitigate knowledge gaps and improves factual
grounding. However, integrating retrieved knowl-
edge into MLLMs presents two key challenges.
First, Parametric-Retrieved Knowledge Incon-
sistency (PRKI) . Since MLLMs rely on frozen pre-
training knowledge, retrieved text and images may
contradict, extend, or refine this information. More-
over, retrieved content can be incomplete, noisy,
or misleading, introducing biases or factual errors.
Without effective reconciliation, the model may
struggle to balance reliability between internal para-

metric and external retrieved knowledge, leading
to incorrect responses. As shown in Figure 1, for
the aircraft’s first flight date, introducing noisy re-
trieval information (August 29, 1970) overrides the
model’s reliable parametric knowledge (December
20, 1957). Second, Visual-Textual Knowledge In-
consistency (VTKI) . Since each modality captures
different aspects of entity representation (Li et al.,
2023a), retrieved images and textual documents
often provide non-overlapping and misaligned in-
formation. For instance, an image may visually
relate to the query while its paired text describes
a different aspect or interpretation. These incon-
sistencies disrupt knowledge integration, making it
difficult for the model to determine which informa-
tion to prioritize.
To address the above-mentioned challenges, we
propose Cross-source knowledge Reconciliation
forMultiModal RAG (CoRe-MMRAG), a novel
end-to-end framework designed to mitigate the in-
consistencies between different knowledge sources.
CoRe-MMRAG follows a four-stage pipeline: it
first generates an initial response using only the
model’s internal knowledge, then performs joint
similarity assessment to select the most relevant
multimodal evidence, followed by generating a
response grounded in the retrieved content, and
finally integrates both sources to produce a coher-
ent and reliable answer. This structured process
enables the model to effectively reconcile diverse
knowledge inputs and leverage complementary in-
formation from multiple modalities.
To further enhance knowledge reconciliation,
CoRe-MMRAG incorporates a specialized train-
ing paradigm with three key objectives: knowl-
edge source selection that learns to identify the
most reliable information source between internal
parametric and external retrieved knowledge, mul-
timodal selection that optimizes the joint under-
standing of visual-textual pairs, and unified answer
generation that ensures consistent and accurate re-
sponses. Through this comprehensive training strat-
egy, CoRe-MMRAG develops robust capabilities
in handling knowledge inconsistencies between dif-
ferent sources.
We conduct comprehensive experiments on
two knowledge-based VQA benchmarks (Mensink
et al., 2023; Chen et al., 2023) to evaluate our
approach. Using Qwen2-VL-7B (Wang et al.,
2024) as the base MLLM, CoRe-MMRAG demon-
strates substantial improvements in both zero-shot
and fine-tuned setting. Specifically, our methodachieves performance gains of 5.6% over the base-
line on the InfoSeek, while also surpassing the
baseline on the Encyclopedic-VQA benchmark by
9.3%. The contributions of this work are summa-
rized as follows:
•We identify and formalize two fundamental
challenges in multimodal RAG: Parametric-
Retrieved Knowledge Inconsistency and
Visual-Textual Knowledge Inconsistency. To
address these issues, we propose CoRe-
MMRAG, a novel end-to-end framework that
effectively reconciles inconsistencies across
different knowledge sources.
•We design a specialized training paradigm
with three targeted objectives that enhance
MLLMs in knowledge source discrimination,
multimodal integration, and unified answer
generation, enabling effective knowledge in-
consistency mitigation.
•Extensive experiments demonstrate that
CoRe-MMRAG achieves significant perfor-
mance gains over previous SOTAs on multiple
Knowledge-based VQA benchmarks.
2 Related Work
Multimodal RAG. Recent advancements in
MLLMs, such as LLaV A-family (Liu et al., 2023;
Li et al., 2024), Qwen2-VL (Wang et al., 2024),
MiniCPM-V (Yao et al., 2024), and Intern-VL
(Chen et al., 2024), have demonstrated remark-
able performance across various multimodal tasks.
However, these models inherently suffer from hal-
lucination issues (Li et al., 2023b; Caffagni et al.,
2024). One effective approach to mitigating hal-
lucination is the incorporation of multimodal data
(Liu et al., 2024a), which provides complementary
knowledge beyond text-based sources. By integrat-
ing multimodal information, models can ground
their responses in more diverse and concrete ev-
idence, reducing the likelihood of hallucination.
Another approach is inspired by RAG(Kandpal
et al., 2023; Li et al., 2025; Gao et al., 2023;
Asai et al., 2024), several multimodal RAG frame-
works(Lin et al., 2023; Hu et al., 2025; Adjali et al.,
2024) have been proposed, including Wiki-LLaV A
(Caffagni et al., 2024), EchoSight (Yan and Xie,
2024), RoRA-VLM (Qi et al., 2024) and LLaV A-
mR2AG (Zhang et al., 2024b). These methods
typically follow a three-stage pipeline of retrieval,

reranking, and generation. However, during rerank-
ing, they primarily rely on text-based similarity
measures between retrieved passages and questions,
which may lead to incorrect passage selection due
to the inherent cross-modal discrepancy, ultimately
affecting the final answer quality.
Multimodal Understanding. Multimodal un-
derstanding (Zhang et al., 2024a; Yin et al., 2023;
Wang et al., 2025; Liu et al., 2018) aims to in-
tegrate and interpret information across multiple
modalities, such as vision and language, to enhance
reasoning and decision-making. A key challenge
in this field is incorporating external knowledge
to support tasks that require information beyond
what is directly present in one modality. One major
research direction is knowledge-based Visual Ques-
tion Answering (VQA), where models answer ques-
tions requiring external factual knowledge. Early
benchmarks like OK-VQA (Marino et al., 2019)
and A-OKVQA (Schwenk et al., 2022) introduced
commonsense and general knowledge reasoning,
while ViQuAE (Lerner et al., 2022) expanded to
entity-specific queries. More recent datasets, such
as InfoSeek (Chen et al., 2023) and Encyclope-
dic VQA (Mensink et al., 2023), enforce both vi-
sual grounding and knowledge retrieval, exposing
the limitations of existing models, including LLM-
based approaches, in handling multimodal knowl-
edge integration. Beyond VQA, multimodal un-
derstanding extends to tasks such as image-text
retrieval, captioning, and reasoning, where align-
ing visual and textual representations is critical.
3 Methodology
In this section, we first formalize the problem
of Multimodal Retrieval-Augmented Generation
(MMRAG) and introduce two types of inconsis-
tency (§3.1) that arise when applying MMRAG to
KB-VQA. We then present our framework (§3.2)
and detail our training approach (§3.3), both of
which are designed to enhance the capacity of
MLLMs to resolve knowledge inconsistency.
3.1 Problem Formulation
To assess the effectiveness of our proposed frame-
work, we conduct evaluations on Knowledge-
based Visual Question Answering (KB-VQA)
tasks. Given an input image-question pair Q=
(Qv, Qt)from the question set Qwith support from
an external knowledge base K, where each knowl-
edge entry Kicomprises a visual component Viand the associated textual article Ti. A typical MM-
RAG pipeline addresses KB-VQA through three
stages: retrieval, reranking, and generation. In
the retrieval stage, a frozen CLIP (Radford et al.,
2021) encodes both the query image Qvand knowl-
edge base images {Vi}|K|
i=1into a shared embed-
ding space, where the relevance is measured via
cosine similarity: sim(Qv, Vi). The top- kentries
{(Vi, Ti)}k
i=1are retrieved based on these similar-
ity scores. Then, the retrieved entries are reranked
by the multimodal model Mbased on the seman-
tic relevance between Qand the retrieved articles
{Ti}k
i=1. Finally, Mprocesses Qalong with the
most relevant entry to generate the final answer.
During this process, we identify two types of
issues. The first problem arises from the inconsis-
tency between the model’s parametric knowledge
and the retrieved external knowledge, referred to as
Parametric-Retrieved Knowledge Inconsistency
(PRKI) . Formally, let M(Q)denote the output
based on the model’s parametric knowledge, and let
M(Q,P)represent the response when augmented
with the retrieved knowledge set P={(Vi, Ti)}k
i=1.
The PRKI occurs when:
M(Q)̸=M(Q,P). (1)
The second issue is Visual-Textual Knowledge
Inconsistency (VTKI) , which arises when tex-
tual and visual modalities of the retrieved entries
{(Ti, Vi)}k
i=1yield inconsistent relevance rankings.
Formally, the VTKI manifests when:
argmax rM(Q,T)̸= argmax rM(Q,V),(2)
where argmax rM(·,·)represents model M’s pre-
diction of the most relevant entry ID in each modal-
ity. Both PRKI and VTKI can significantly im-
pact model performance. PRKI occurs when noisy
external knowledge overrides reliable parametric
knowledge, resulting in erroneous outputs. Mean-
while, VTKI becomes especially problematic dur-
ing the reranking stage. Due to the inconsistency
between the textual and visual knowledge, relying
on unimodal knowledge increases the risk of in-
troducing irrelevant information to M, which can
propagate errors from the reranking stage to an-
swer generation, potentially leading to PRKI and a
reduction in model performance.
3.2 CoRe-MMRAG Framework
To mitigate the PRKI and VTKI problems de-
scribed in §3.1, we propose CoRe-MMRAG, a

Step 1: Parametric Knowledge GenerationStep4:Parametric-Retrieved KnowledgeIntegrationStep2:Visual-Textual Knowledge IntegrationAnswer: The DC-10 was first flown on February 9, 1970.Answer: The DC-10 was first flown on August 29, 1970.
Answer: BasedonC,The DC-10 was first flown on August 29, 1970.Step3:External Knowledge GenerationAnswer: Content Cismore similar with the questionbecause it is a wide-body plane with three engines
AnswerGeneration
KnowledgeRetrieval
KnowledgeBase
When did this aircraft make its first flight?
Figure 2: Overview of the CoRe-MMRAG Framework. CoRe-MMRAG processes a multimodal query and retrieved
knowledge in four stages: (1) generating an initial response based solely on parametric knowledge; (2) jointly
integrating visual and textual information to identify the most relevant retrieved content; (3) generating an external
response based on the selected content from step (2); and (4) reconciling discrepancies between parametric and
retrieved knowledge to produce the final answer.
framework that effectively reconciles inconsisten-
cies across knowledge sources. As shown in Fig-
ure 2, given a query Qand its retrieved knowledge
entries P, the model Mis prompted to generate
responses across four distinct stages in an end-to-
end manner. Below, we detail each stage of the
framework.
Step 1: Parametric Knowledge Generation .
Although external knowledge Pis available in the
input, we first prompt the model to generate yint
based solely on its parametric knowledge:
yint=M(Q). (3)
This generation establishes a reference point for
detecting potential conflicts with retrieved knowl-
edge in Step 4.
Step 2: Visual-Textual Knowledge Integra-
tion. Following the parametric knowledge gener-
ation, the model evaluates the relevance between
query Qand knowledge entries in P. Considering
the potential VTKI manifested as:


Iv= argmax
i∈{1,...,k}rM(Q,{Vi}k
i=1),
It= argmax
j∈{1,...,k}rM(Q,{Tj}k
j=1),
It̸=Iv.(4)
We propose a joint similarity assessment that uti-
lizes the complementary nature of visual and tex-
tual modalities:
Itv= argmax
i∈{1,...,k}rM(Q,{Vi, Ti}k
i=1), (5)
where Itvdenotes the most relevant entry ID based
on multimodal knowledge. This unified ranking ap-proach jointly leverages abstract semantic concepts
from textual descriptions and detailed visual char-
acteristics, eliminating the bias introduced by sep-
arate unimodal evaluations, which enables a more
robust relevance assessment and resolves VTKI.
Step 3: External Knowledge Generation. Af-
ter obtaining the most relevant knowledge entry
PItv, the model is prompted to generate a response
based on the retrieved external knowledge:
yext=M 
Q,(VItv, TItv)
, (6)
where yextdenotes the model’s response that ex-
plicitly considers the retrieved visual-textual knowl-
edge pair (VItv, TItv).
Step 4: Parametric-Retrieved Knowledge
Integration. Given responses from parametric
knowledge yintand retrieved knowledge yext, the
parametric-retrieved knowledge inconsistencies
may arise. The model is prompted to resolve these
inconsistencies and generate the final response:
y∗=M 
Q, yint, yext,(VItv, TItv)
, (7)
where y∗represents the final response, which is
determined by comparing the credibility of para-
metric knowledge and retrieved external knowl-
edge. This process enables the model to leverage
both knowledge sources while ensuring the reliable
generation of the final answer.
3.3 Inconsistency-Aware Multimodal Training
Inspired by the self-training mechanism in STaR
(Zelikman et al., 2022), we propose a fine-tuning
paradigm that leverages the model’s self-generated
outputs under different knowledge conditions. The

model learns to resolve PRKI and mitigate VTKI
through three specialized training objectives, thus
improving its ability to generate accurate answers
based on retrieved knowledge.
Parametric-Retrieved Knowledge Selection .
We begin by generating answers based solely on
the model’s parametric knowledge, ˆyint=M(Q),
and re-evaluate the same question with retrieved
external knowledge, obtaining ˆyext=M(Q,P).
After generating both outputs, we filter questions
where the model produces correct answers exclu-
sively using either internal or external knowledge,
forming fine-tuning datasets DintandDext:


ˆyint̸= ˆyext,
Dint={(Qj,Pj)|ˆyint
j= ˆygt
j}|Q|
j=1,
Dext={(Qi,Pi)|ˆyext
i= ˆygt
i}|Q|
i=1.(8)
Then, we fine-tune the model MonDintandDext,
the training is guided by loss function LPRKI :
LPRKI =−X
(Qj,Pj)∼DintlogM(ˆyint|Qj,Pj)+
X
(Qi,Pi)∼DextlogM(ˆyext|Qi,Pi)
,(9)
which encourages the model to prioritize the knowl-
edge source that leads to correct answer generation,
ensuring robustness in handling PRKI.
Visual-Textual Knowledge Selection . The
model computes the most relevant entry IDs
independently using visual knowledge ˆIv=
argmax rM(Q,V)and textual knowledge ˆIt=
argmax rM(Q,T). The training datasets Dvand
Dtare constructed by selecting samples as follows:


ˆIv̸=ˆIt,
Dv={(Qm,Pm)|ˆIv
m=Igt
m}|Q|
m=1,
Dt={(Qn,Pn)|ˆIt
n=Igt
n}|Q|
n=1.(10)
Here, Igtdenotes the ground-truth index for the
most relevant entry. The model is then fine-tuned
onDvandDtusing:
LVTKI =−X
(Qm,Pm)∼Dvlog arg max
j∈{1,...,k}rM(ˆIv|Qm,Pj
m)+
X
(Qn,Pn)∼Dtlog arg max
i∈{1,...,k}rM(ˆIt|Qn,Pi
n)
,
(11)Dataset Train Val Test KB
Enc-VQA 1M 13.6K 5.8K 2M Wiki
InfoSeek 934K 73K 348K 100K Wiki
Table 1: Statistics of datasets, including sample splits
and their corresponding knowledge base sizes.
where the loss function LVTKIenables the model to
evaluate the reliability of visual and textual modal-
ities and prioritize the more confident one, thereby
mitigating VTKI induced by unimodal bias.
Unified Answer Generation . We countinue
training the model on DvandDt, applying Super-
vised Fine-Tuning (SFT) with the loss function:
LSFT=−X
(Qm,Pm)∼DvlogM(ˆygt|Qm,PIv
m)+
X
(Qn,Pn)∼DtlogM(ˆygt|Qn,PIt
n)
,(12)
where LSFTis used to fine-tune the model, encour-
aging it to generate accurate answers based on the
ground-truth external knowledge.
4 Experiments
4.1 Datasets
We evaluate our proposed CoRe-MMRAG on two
large-scale knowledge-based VQA benchmarks:
Encyclopedic VQA (Mensink et al., 2023) and
InfoSeek (Chen et al., 2023). Both datasets con-
tain diverse visual-textual queries requiring fine-
grained entity knowledge, with explicit knowl-
edge bases to ensure answer verifiability. Encyclo-
pedic VQA (Enc-VQA) contains 221K (Qt,ˆygt)
unique question-answer pairs distributed across
16.7K fine-grained entities, where each question-
answer pair is associated with up to five diverse
instance images, resulting in 1M image-question-
answer (Qv, Qt,ˆygt)triplets, while InfoSeek con-
tains 1.3M (Qv, Qt,ˆygt)triplets corresponding to
approximately 11K distinct entities. Detailed statis-
tics for both data sets, including sample splits and
knowledge base sizes, are shown in Table 1. Fol-
lowing standard practice in EchoSight (Yan and
Xie, 2024), we evaluate on Enc-VQA’s test set af-
ter excluding two-hop questions, resulting in 4.7K
test triplets. For InfoSeek, we report results on
the validation split, which contains unseen entities
(Unseen-E) and novel questions (Unseen-Q).

4.2 Metrics
Metrics for Retrieval. We adopt Recall@ kas
the standard metric to evaluate the retrieval perfor-
mance (Liu et al., 2021). This metric examines
whether the ground-truth article appears within the
top-kretrieved results. Following EchoSight, the
evaluation criterion considers an article correct only
when its URL exactly matches the ground-truth
URL, ensuring precise assessment of retrieval ac-
curacy.
Metrics for Answer Generation. We employ
dataset-specific metrics following standard prac-
tices in knowledge-based VQA. For Enc-VQA, we
use BEM (Zhang et al., 2019), while for InfoS-
eek (Chen et al., 2023), we adopt both VQA accu-
racy (Marino et al., 2019) and Relaxed Accuracy
(Methani et al., 2020; Masry et al., 2022).
4.3 Implementation Details
Retriever. For external knowledge retrieval, Enc-
VQA utilizes a knowledge base consisting of 2M
Wikipedia articles and up to 5M associated images.
In contrast, InfoSeek employs a filtered subset of
100K Wikipedia articles with approximately 350K
images, following the setup in (Yan and Xie, 2024).
Visual features are extracted using a frozen Eva-
CLIP-8B encoder (Sun et al., 2023), where we
use the pooled embeddings from the last layer to
compute the cosine similarity between reference
images and candidate images. We construct a vi-
sual feature index using the FAISS library for effi-
cient similarity search and retrieve the top-5 most
relevant entries as external knowledge. Retrieval
performance is reported in Table 2.
Zero-shot Settings. We employ Qwen2-VL-7B
(Wang et al., 2024) as our base model, leveraging
its 32K token input capacity to accommodate both
visual and textual knowledge. To ensure a fair com-
parison with existing MMRAG approaches, includ-
ing Wiki-LLaV A (Caffagni et al., 2024), EchoSight
(Yan and Xie, 2024), RoRA-VLM (Qi et al., 2024),
and LLaV A-mR2AG (Zhang et al., 2024b), we
reimplement these pipelines using Qwen2-VL-7B
as a unified backbone. We consider the following
configurations for zero-shot evaluation: (1) Qwen2-
VL-Param , which relies solely on the model’s in-
ternal parametric knowledge without any external
context; (2) Qwen2-VL-Oracle , which takes the
ground-truth wiki entry as input and serves as an
upper-bound; (3) Qwen2-VL-1-Stage , which repli-
cates Wiki-LLaV A’s one-stage pipeline by encod-DatasetsRecall@ k(%)
k=1 k=2 k=5 k=10
InfoSeek 45.6 54.8 67.1 73.0
Enc-VQA 13.3 16.9 31.3 41.0
Table 2: The retrieval performance of Eva-CLIP-8B on
two datasets.
ing all retrieved entries for answer generation; (4)
Qwen2-VL-2-Stage , which follows the two-stage
architecture of LLaV A-mR2AG and EchoSight by
reranking retrieved entries and generating answers
based on the top-ranked one; and (5) Qwen2-VL-
MMSTaR , which incorporates all retrieved entries
into a Chain-of-Thought reasoning process follow-
ing the STaR framework (Zelikman et al., 2022).
Fine-tuned Setting . We fine-tune the Qwen2-
VL variants with task-specific supervision to en-
hance both retrieval accuracy and answer genera-
tion. Qwen2-VL-1-Stage is trained using a stan-
dard supervised objective LSFTwith ground-truth
wiki entries as input, enhancing the model’s ability
to generate correct answers from these references.
Qwen2-VL-2-Stage is optimized with a combina-
tion of a selection objective LPRKI and generation
objective LSFT, enabling the model to better iden-
tify the correct entry from the top-5 retrieved can-
didates and generate more accurate answers based
on the selected context.
Our proposed CoRe-MMRAG is trained with
three objectives, including LVTKI ,LPRKI , and
LSFT, which jointly enhance parametric-retrieved
knowledge selection, visual-textual knowledge se-
lection, and final answer generation. We sample
approximately 30K (Qv, Qt,ˆygt)triplets from the
training set of each benchmark to construct the
training set. The model is fine-tuned using LoRA
with a rank of 8, applied to all layers. Training
is conducted for 3 epochs with a learning rate of
1×10−4and a batch size of 1×2, using 8 H100
GPUs. The full training process takes approxi-
mately 10 hours.
4.4 Main Results
Table 3 presents comprehensive comparisons of
our method against current SOTA approaches on
Enc-VQA and InfoSeek benchmarks.
Zero-shot Setting. Qwen2-VL-Oracle, which
accesses ground-truth Wikipedia entries, estab-
lishes an upper-bound performance of 51.2% on
Enc-VQA and 57.9% on InfoSeek. When using
retrieved entries instead of gold references, our

Enc-VQA InfoSeek
Model LLM KB Single-Hop Unseen-Q Unseen-E All
Zero-shot Models
LLaV A-1.5 Vicuna-7B - 16.3 13.0 10.3 12.2
Qwen2-VL-Param Qwen2-7B - 12.7 23.1 21.8 22.1
Qwen2-VL-Oracle Qwen2-7B Wiki 51.2 57.9 57.9 57.9
Qwen2-VL-1-Stage † Qwen2-7B Wiki 17.9 40.8 40.9 40.9
Qwen2-VL-2-Stage † Qwen2-7B Wiki 17.0 34.9 35.1 35.0
Qwen2-VL-MMSTaR †Qwen2-7B Wiki 16.9 33.4 34.0 33.9
Ours† Qwen2-7B Wiki 20.1 42.3 43.3 42.9
Fine-tuned Models
Wiki-LLaV A Vicuna-7B Wiki 17.7 30.1 27.8 28.9
RoRA-VLM Vicuna-7B Wiki+Web 20.3 27.3 25.1 26.9
EchoSight LLaMA3-8B Wiki 19.4 - - 27.7
LLaV A-mR2AG Vicuna-7B Wiki 55.1* 39.1 39.7 39.4
Qwen2-VL-1-Stage † Qwen2-7B Wiki 24.3 42.3 43.4 43.0
Qwen2-VL-2-Stage † Qwen2-7B Wiki 23.1 36.8 37.6 37.3
Qwen2-VL-MMSTaR †Qwen2-7B Wiki 20.9 34.8 35.2 35.1
Ours† Qwen2-7B Wiki 27.2 45.2 46.9 46.5
Table 3: Main results (%) on Enc-VQA and InfoSeek with external knowledge. †denotes our method and variants.
Note: on Enc-VQA LLaV A-mR2AG* uses Google Lens for retrieval, achieving 62.5% Recall@5, while our
Eva-CLIP-based retrieval achieves 31.3% Recall@5.
proposed CoRe-MMRAG achieves the best results
among all methods, reaching 42.9% accuracy on
InfoSeek and 20.1% on Enc-VQA, surpassing the
second-best method, Qwen2-VL-1-Stage, by mar-
gins of 2.0% and 2.2%, respectively. Qwen2-VL-
2-Stage yields suboptimal results due to potential
information loss in unimodal reranking. Qwen2-
VL-MMSTaR shows only slight improvements,
indicating that current Chain-of-Thought reason-
ing remains limited for knowledge-intensive VQA.
Nonetheless, it still outperforms the parametric-
only baseline, highlighting the benefit of retrieved
multimodal knowledge. Notably, in zero-shot set-
tings, the performance gain on InfoSeek is more
substantial than on Enc-VQA. This discrepancy is
likely due to the larger and more complex knowl-
edge base of Enc-VQA, which introduces greater
retrieval noise and increases the difficulty of both
PRKI and VTKI.
Fine-tuned Setting. Our method maintains su-
perior performance, achieving 46.5% on InfoSeek
and 27.2% on Enc-VQA. Furthermore, it exhibits
the largest improvements comparing zero-shot set-
ting, with gains of 3.6% on InfoSeek and 7.1%
on Enc-VQA. Qwen2-VL-2-Stage, trained with
LVTKI andLSFT, shows an improvement of 2.3%
improvement over its zero-shot performance on
InfoSeek and 6.1% on Enc-VQA. Qwen2-VL-1-
Stage, fine-tuned with LSFT, achieves gains of
2.1% on InfoSeek and 6.4% on Enc-VQA. Qwen2-
VL-MMSTaR demonstrates limited improvement,primarily due to the inadequate quality of its self-
generated training instances.
46.751.745.649.650.157.3
015304560
Zero-shotFine-tunedGT-Refernce﻿RecognitionAccuracy(%)Visual-Textual Knowledge Reconciliation
TextualVisualOurs40.942.943.0 46.522.116.617.523.817.619.301020304050
Param1-StageOursParam(FT)1-Stage(FT)Ours(FT)Generation Accracy(%)Parametric-Retrieved Knowledge Reconciliation
with External Knowledgewithout External Knowledge
Figure 3: Effectiveness of Our Method in Mitigat-
ing PRKI and VTKI. Top: Evaluation of Parametric-
Retrieved Knowledge Reconciliation, comparing our
proposed method with Qwen-2-VL-Param (Param) and
Qwen2-VL-1-Stage (1-Stage) under zero-shot and fine-
tuned settings. Bottom : Evaluation of Visual-Textual
Knowledge Reconciliation, showing that our method
improves ground-truth entry recognition through both
textual and visual modalities across both settings.

Recall@1 Recall@2 Recall@5 Recall@ k(K>5) All
Model U-Q U-E Acc U-Q U-E Acc U-Q U-E Acc U-Q U-E Acc Acc
Param 25.0 25.3 25.2 24.4 23.9 24.1 20.7 22.3 21.8 19.3 15.7 16.4 22.1
1-Stage †
Top-1 59.1 58.6 58.8 30.1 27.3 28.0 22.0 22.4 22.1 15.4 15.6 15.6 39.6
Top-2 58.4 57.9 58.2 46.5 45.7 46.0 22.6 25.9 25.1 15.5 15.9 15.8 40.4
Top-5 56.7 56.2 56.4 45.6 44.8 44.7 32.3 34.4 33.8 16.8 16.5 16.6 40.9
Ours
Top-1 61.5 61.8 61.7 31.7 27.7 28.6 22.8 22.8 22.8 16.6 15.0 15.4 40.7
Top-2 61.0 61.2 61.2 48.4 49.0 48.9 23.8 27.3 26.4 16.9 15.7 16.0 42.9
Top-5 59.7 59.3 59.4 46.2 47.8 47.6 33.2 35.8 35.2 17.5 16.3 16.7 42.9
Table 4: Performance (%) with different numbers of retrieved entries. Recall@ kindicates the presence of ground-
truth entry in top- kretrieved results, with accuracy reported across Unseen-Questions (U-Q) and Unseen-Entities
(U-E) settings.
Knowledge Reconciliation. Figure 3 illustrates
the effectiveness of our method in addressing both
VTKI and PRKI under zero-shot and fine-tuned
settings. In the zero-shot setting, the model iden-
tifies 46.7% of the ground truth Wikipedia entries
using the textual modality and 45.6% using the
visual modality, with over 40% of the entities cor-
rectly recognized by both. Our method improves
this recognition rate to 50.1% demonstrating the
effectiveness of our method in handling VTKI.
For PRKI, our method achieves better retention
of parametric knowledge compared to the 1-Stage
baseline, indicating more robust reconciliation of
parametric-retrieved knowledge inconsistency. In
the fine-tuned setting, the model’s ability to lever-
age multimodal knowledge for identifying ground-
truth Wikipedia entries further improves, increas-
ing from 50.1% to 57.3%, demonstrating the effec-
tiveness of the objective LVTKI . However, we ob-
serve that the 1-Stage baseline, trained solely with
LSFT, tends to compromise the retention of correct
parametric knowledge, resulting in a larger perfor-
mance drop (23.8% to 17.6%) relative to the origi-
nal parametric outputs. In contrast, our method bet-
ter preserves parametric knowledge, with a modest
drop (23.8% to 19.3%), indicating the effectiveness
of theLPRKI objective.
4.5 Ablation Studies
To validate the effectiveness of our approach, we
conduct ablation studies on the InfoSeek validation
set.
Effect of Retrieved Entry Count. Increasing
the number of retrieved entries generally leads to
improved overall recall accuracy. However, its
impact varies across different Recall@ kgroups,
where Recall@ kindicates whether the ground-truth entry is included in the top- kretrieved re-
sults. As shown in Table 4, we observe a clear
relationship between the number of retrieved en-
tries (Top- m) and the Recall@ kperformance. The
first case is when m≤k, meaning the number
of retrieved entries does not exceed the evaluation
threshold. In this setting, increasing mdirectly ex-
pands the candidate pool, which generally leads to
consistent performance improvements. For exam-
ple, in the Recall@5 group, increasing mfrom 2 to
5 improves recall accuracy, with the 1-Stage base-
line rising from 25.1% to 33.8% and our method
from 26.4% to 35.2%, as more relevant entries are
included in the input. In contrast, when m > k ,
the inclusion of additional entries yields marginal
or even negative returns, likely due to the introduc-
tion of irrelevant or noisy information. This effect
is particularly evident in lower Recall@ kgroups,
where precision is more sensitive to input quality.
For instance, in the Recall@2 group, increasing m
from 2 to 5 leads to a decrease in accuracy, from
46.0% to 44.7% for the 1-Stage baseline and from
48.9% to 47.6% for our method.
Moreover, the impact of mvaries across evalu-
ation scenarios. While increasing the number of
external entries initially benefits both Unseen-Q
and Unseen-E, the latter demonstrates more stable
and robust improvements. In contrast, performance
degradation caused by excessive retrieval is more
pronounced in Unseen-Q, highlighting its greater
sensitivity to noisy or irrelevant knowledge.
Effect of Different Prompt. Table 4 presents
a comprehensive zero-shot performance analysis
between Qwen2-VL-1-Stage and our proposed
method. For PRKI resolution, CoRe-MMRAG
exhibits enhanced robustness to knowledge noise
through carefully designed prompts. In Recall@1

MethodsRecall@5 (%)
Unseen-Q Unseen-E All
Ours 45.2 46.9 46.5
w/oLPRKI 44.1 46.0 45.5
w/oLVTKI 44.2 45.6 45.3
w/oLSFT 43.3 45.0 44.6
Table 5: Ablation study on training objectives.
group, when increasing retrieved entries from 1
to 2, our method maintains stable performance
with accuracy shifting from 61.7% to 61.2%, while
Qwen2-VL-1-Stage shows larger degradation from
58.8% to 58.2%. This advantage becomes more ev-
ident in the setting of Unseen-Questions, where our
method preserves accuracy from 61.5% to 61.0%
compared to the Qwen2-VL-1-Stage’s significant
drop from 59.1% to 58.4%.
Moreover, our approach demonstrates superior
multimodal knowledge integration capabilities. In
Recall@2 group, increasing retrieved entries from 1
to 2 yields substantially larger gains as our method
improves from 28.6% to 48.9% versus Qwen2-VL-
1-Stage advancing from 28.0% to 46.0%. The
improvement margin widens further in Unseen-
Entities scenarios, with our method achieving
progress from 27.7% to 49.0% while the baseline
moves from 27.3% to 45.7%, confirming the effec-
tiveness of our method on visual-textual fusion, as
visualized in Figure 3.
Effect of Training Objectives. Table 5 demon-
strates the contribution of each training objec-
tive. Removing any objective leads to performance
degradation, with LSFTcausing the most signifi-
cant decline at 1.9% in overall performance. The
absence of LPRKI andLVTKI also leads to notable
performance drops at 1.0% and 1.2% respectively,
indicating both objectives are essential for effec-
tive knowledge conflict and inconsistency resolu-
tion. Our full model achieves the best performance
across all metrics, validating the complementary
nature of these objectives.
5 Conclusion
In this paper, we present CoRe-MMRAG, a novel
framework that addresses two critical challenges
in MMRAG: parametric-retrieved knowledge in-
consistency and visual-textual knowledge incon-
sistency. CoRe-MMRAG follows a four-stage
pipeline that effectively reconciles internal para-
metric knowledge with externally retrieved infor-
mation and leverages joint similarity assessment tointegrate complementary visual and textual signals.
To further enhance its capabilities, we introduced a
specialized training paradigm with three targeted
objectives focused on knowledge source selection,
multimodal integration, and answer generation. Ex-
tensive experiments on InfoSeek and Enc-VQA
benchmarks demonstrate the effectiveness of our
approach, achieving performance gains of 5.6%
and 9.3% over baseline methods.
6 Limitations and Future Works
Despite the promising results, our work has several
important limitations. First, our framework’s effec-
tiveness is heavily dependent on the initial retrieval
quality using Eva-CLIP-8B. As demonstrated in
our experiments, the retrieval performance (Re-
call@1) remains relatively low at 45.6% for InfoS-
eek and 13.3% for Enc-VQA. This limited retrieval
performance creates a ceiling effect for the over-
all system performance, suggesting that improve-
ments in the initial retrieval stage could lead to
significant gains in the final results. Second, our ap-
proach faces substantial computational challenges.
The four-stage process and multiple training objec-
tives require significant computational resources,
with training taking approximately 10 hours on
8 H100 GPUs. The model requires substantial
memory to process both visual and textual knowl-
edge simultaneously, which may limit its practi-
cal deployment in resource-constrained environ-
ments. Finally, there are limitations in terms of
dataset coverage and generalization. While we
demonstrate strong performance on two specific
KB-VQA benchmarks, the effectiveness of our ap-
proach on other types of multimodal tasks or real-
world scenarios remains unexplored. The model’s
performance on rare entities or edge cases in the
knowledge base may be suboptimal, and there ex-
ists a potential domain gap between the Wikipedia
knowledge base used for training and real-world
applications. Future work should address these lim-
itations to improve the practical applicability of our
approach.

References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 , pages 1–100.
Omar Adjali, Olivier Ferret, Sahar Ghannay, and Hervé
Le Borgne. 2024. Multi-level information retrieval
augmented generation for knowledge-based visual
question answering. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing , pages 16499–16513.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc,
Antoine Miech, Iain Barr, Yana Hasson, Karel
Lenc, Arthur Mensch, Katherine Millican, Malcolm
Reynolds, et al. 2022. Flamingo: a visual language
model for few-shot learning. Advances in Neural
Information Processing Systems , 35:23716–23736.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InInternational Conference on Learning Representa-
tions , pages 1–30.
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He,
Zongbo Han, Zheng Zhang, and Mike Zheng Shou.
2024. Hallucination of multimodal large language
models: A survey. arXiv preprint arXiv:2404.18930 ,
pages 1–40.
Davide Caffagni, Federico Cocchi, Nicholas Moratelli,
Sara Sarto, Marcella Cornia, Lorenzo Baraldi, and
Rita Cucchiara. 2024. Wiki-llava: Hierarchical
rretrieval-augmented generation for multimodal llms.
InProceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition , pages 1818–
1826.
Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit
Changpinyo, Alan Ritter, and Ming-Wei Chang. 2023.
Can pre-trained vision and language models answer
visual information-seeking questions? In Proceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing , pages 14948–14968.
Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo
Chen, Sen Xing, Muyan Zhong, Qinglong Zhang,
Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scal-
ing up vision foundation models and aligning for
generic visual-linguistic tasks. In Proceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 24185–24198.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 , pages 1–21.
Wenbo Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz,
Pan Lu, Kai-Wei Chang, and Nanyun Peng. 2025.Mrag-bench: Vision-centric evaluation for retrieval-
augmented multimodal models. In International Con-
ference on Learning Representations , pages 1–24.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric
Wallace, and Colin Raffel. 2023. Large language
models struggle to learn long-tail knowledge. In In-
ternational Conference on Machine Learning , pages
15696–15707. PMLR.
Paul Lerner, Olivier Ferret, Camille Guinaudeau, Hervé
Le Borgne, Romaric Besançon, José G Moreno, and
Jesús Lovón Melgarejo. 2022. Viquae, a dataset for
knowledge-based visual question answering about
named entities. In Proceedings of the 45th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval , pages 3108–
3120.
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang,
Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan
Zhang, Yanwei Li, Ziwei Liu, et al. 2024. Llava-
onevision: Easy visual task transfer. arXiv preprint
arXiv:2408.03326 , pages 1–43.
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
2023a. Blip-2: Bootstrapping language-image pre-
training with frozen image encoders and large lan-
guage models. In International Conference on Ma-
chine Learning , pages 19730–19742. PMLR.
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang,
Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Eval-
uating object hallucination in large vision-language
models. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 292–305.
Zixu Li, Zhiwei Chen, Haokun Wen, Zhiheng Fu, Yu-
peng Hu, and Weili Guan. 2025. Encoder: Entity
mining and modification relation binding for com-
posed image retrieval. In Proceedings of the AAAI
Conference on Artificial Intelligence , pages 5101–
5109.
Weizhe Lin, Jinghong Chen, Jingbiao Mei, Alexan-
dru Coca, and Bill Byrne. 2023. Fine-grained late-
interaction multi-modal retrieval for retrieval aug-
mented visual question answering. Advances in
Neural Information Processing Systems , 36:22820–
22840.
Fan Liu, Zhiyong Cheng, Lei Zhu, Zan Gao, and
Liqiang Nie. 2021. Interest-aware message-passing
gcn for recommendation. In Proceedings of the Web
Conference 2021 , page 1296–1305. ACM.
Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen,
Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li,
and Wei Peng. 2024a. A survey on hallucination
in large vision-language models. arXiv preprint
arXiv:2402.00253 , pages 1–10.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae
Lee. 2024b. Improved baselines with visual instruc-
tion tuning. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition ,
pages 26296–26306.
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning. Advances in
Neural Information Processing Systems , 36:34892–
34916.
Meng Liu, Xiang Wang, Liqiang Nie, Xiangnan He,
Baoquan Chen, and Tat-Seng Chua. 2018. Atten-
tive moment retrieval in videos. In Proceedings of
the 41st International ACM SIGIR Conference on
Research & Development in Information Retrieval ,
pages 15–24.
Kenneth Marino, Mohammad Rastegari, Ali Farhadi,
and Roozbeh Mottaghi. 2019. Ok-vqa: A visual ques-
tion answering benchmark requiring external knowl-
edge. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages
3195–3204.
Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. 2022. Chartqa: A benchmark
for question answering about charts with visual and
logical reasoning. In Findings of the Association for
Computational Linguistics , pages 2263–2279.
Thomas Mensink, Jasper Uijlings, Lluis Castrejon,
Arushi Goel, Felipe Cadar, Howard Zhou, Fei Sha,
André Araujo, and Vittorio Ferrari. 2023. Encyclo-
pedic vqa: Visual questions about detailed properties
of fine-grained categories. In Proceedings of the
IEEE/CVF International Conference on Computer
Vision , pages 3113–3124.
Nitesh Methani, Pritha Ganguly, Mitesh M Khapra, and
Pratyush Kumar. 2020. Plotqa: Reasoning over sci-
entific plots. In Proceedings of the IEEE/CVF Win-
ter Conference on Applications of Computer Vision ,
pages 1527–1536.
Jingyuan Qi, Zhiyang Xu, Rulin Shao, Yang Chen, Jin
Di, Yu Cheng, Qifan Wang, and Lifu Huang. 2024.
Rora-vlm: Robust retrieval-augmented vision lan-
guage models. arXiv preprint arXiv:2410.08876 ,
pages 1–15.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sas-
try, Amanda Askell, Pamela Mishkin, Jack Clark,
et al. 2021. Learning transferable visual models
from natural language supervision. In International
Conference on Machine Learning , pages 8748–8763.
PMLR.
Machel Reid, Nikolay Savinov, Denis Teplyashin,
Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste
Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Fi-
rat, Julian Schrittwieser, et al. 2024. Gemini 1.5: Un-
locking multimodal understanding across millions of
tokens of context. arXiv preprint arXiv:2403.05530 ,
pages 1–154.Dustin Schwenk, Apoorv Khandelwal, Christopher
Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022.
A-okvqa: A benchmark for visual question answer-
ing using world knowledge. In European Conference
on Computer Vision , pages 146–162. Springer.
Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang,
and Yue Cao. 2023. Eva-clip: Improved train-
ing techniques for clip at scale. arXiv preprint
arXiv:2303.15389 , pages 1–7.
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi-
hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhanc-
ing vision-language model’s perception of the world
at any resolution. arXiv preprint arXiv:2409.12191 ,
pages 1–52.
Xiao Wang, Jianlong Wu, Zijia Lin, Fuzheng Zhang,
Di Zhang, and Liqiang Nie. 2025. Video datafly-
wheel: Resolving the impossible data trinity in video-
language understanding. IEEE Transactions on Pat-
tern Analysis and Machine Intelligence , 46(1):1–13.
Yibin Yan and Weidi Xie. 2024. Echosight: Advanc-
ing visual-language models with wiki knowledge. In
Findings of the Empirical Methods in Natural Lan-
guage Processing , pages 1538–1551.
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang,
Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li,
Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v:
A gpt-4v level mllm on your phone. arXiv preprint
arXiv:2408.01800 , pages 1–26.
Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi,
Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiy-
ong Wang, Lu Sheng, Lei Bai, et al. 2023. Lamm:
Language-assisted multi-modal instruction-tuning
dataset, framework, and benchmark. Advances in
Neural Information Processing Systems , 36:26650–
26685.
Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Good-
man. 2022. Star: Bootstrapping reasoning with rea-
soning. Advances in Neural Information Processing
Systems , 35:15476–15488.
Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu.
2024a. Vision-language models for vision tasks: A
survey. IEEE Transactions on Pattern Analysis and
Machine Intelligence , 46(8):5625–5644.
Tao Zhang, Ziqi Zhang, Zongyang Ma, Yuxin Chen,
Zhongang Qi, Chunfeng Yuan, Bing Li, Junfu Pu,
Yuxuan Zhao, Zehua Xie, et al. 2024b. Mr2ag:
Multimodal retrieval-reflection-augmented genera-
tion for knowledge-based vqa. arXiv preprint
arXiv:2411.15041 , pages 1–14.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Wein-
berger, and Yoav Artzi. 2019. Bertscore: Evaluating
text generation with bert. In International Confer-
ence on Learning Representations , pages 1–43.

A Appendix
A.1 Prompts
In this section, we present the prompts used in dif-
ferent MMRAG pipelines, including Qwen2-VL-
Param, Qwen2-VL-Oracle, Qwen2-VL-1-Stage,
Qwen2-VL-2-Stage, Qwen2-VL-MMSTaR, and
our proposed CoRe-MMRAG. Each prompt is il-
lustrated with the format, an example, and the cor-
responding final answer.
Prompt for Qwen2-VL-Param
Prompt format:
Question : <image >
{ question }
Please use parametric knowledge answer the
,→question within 5 words
Example:
Question : <image >
Which historic county does this village belong
,→to?
Please use parametric knowledge answer the
,→question within 5 words
Answer:
" Scotland "
The prompt for Qwen2-VL-Oracle is con-
structed using the ground-truth Wikipedia entry.
Specifically, wiki_title_gt and wiki_content_gt refer to
the title and textual content of the ground-truth
entry, respectively.
Prompt for Qwen2-VL-Oracle
Prompt format:
Based on the retrieved document , answer the
,→question
<image >
{ question } within 5 words
Context :
Reference Image :<image >
Wiki title : { wiki_title_gt }
Wiki content :{ wiki_content_gt }
Example:
Based on the retrieved document , answer the
,→question
<image >
What isthe parent organization of this building
,→? within 5 words .
Context :
Reference Image :< image #gt >
Wiki title : Colonial Williamsburg
Wiki content : Colonial Williamsburg isa living -
,→history museum and private foundation
,→presenting a part of the historic
,→diseconstructions ...
Answer:
" The Colonial Williamsburg Foundation "
The following is the prompt for Qwen2-VL-1-
Stage, where wiki_title_A/B/C/D/E and wiki_content_A/B/C/D
,→/Erepresent the titles and textual contents of the
top-5 retrieved Wikipedia entries.Prompt for Qwen2-VL-1-Stage
Prompt format:
Based on the retrieved document , answer the
,→question
<image >
{ question } within 5 words
Reference A:
<image #A>
Wiki title : { wiki_title_A }
Wiki content :{ wiki_content_A }
Reference B:
<image #B>
Wiki title : { wiki_title_B }
Wiki content :{ wiki_content_B }
Reference C:
<image #C>
Wiki title : { wiki_title_C }
Wiki content :{ wiki_content_C }
Reference D:
<image #D>
Wiki title : { wiki_title_D }
Wiki content :{ wiki_content_D }
Reference E:
<image #E>
Wiki title : { wiki_title_E }
Wiki content :{ wiki_content_E }
Example:
Based on the retrieved document , answer the
,→question
<image >
What isthe width ( inkilometre ) of this lake ?
,→within 5 words .
Reference A
<image #A>
Wiki title : Wadi Numeira
Wiki content : Wadi Numeira isa Wadi inJordan
,→that isknown for its deep gorge cut through
,→the sandstone ...
Reference B
<image #B>
Wiki title : Kalya
Wiki content : Kalya isan Israeli settlement
,→organized as a kibbutz inthe West Bank ..
Reference C
<image #C>
Wiki title : Ein Gedi
Wiki content : Ein Gedi , also spelled En Gedi ,
,→meaning \" spring of the kid \", is an oasis and
,→ a nature reserve in Israel ...
Reference D
<image #D>
Wiki title : Dead Sea
Wiki content : The Dead Sea , also known by other
,→names , is a salt lake bordered by Jordan to
,→the east and Israel ...
Reference E
<image #E>
Wiki title : Jewish National Fund
Wiki content : Jewish National Fund was founded
,→in 1901 to buy and develop land in Ottoman
,→Syria ... "
Answer:
"10"
The following is the prompt for Qwen2-VL-2-
Stage, where wiki_title_A/B/C/D/E and wiki_content_A/B/C/
,→D/Edenote the titles and textual contents of the
top-5 retrieved Wikipedia entries. wiki_title_select
and wiki_content_select refer to the title and content of
the entry selected during the first-stage reranking.

Prompt for Qwen2-VL-2-Stage
Prompt format for rerank:
Identify the most similar wiki context to the
,→question
<image >
{ question }
Context A:
Wiki title : { wiki_title_A }
Wiki content :{ wiki_content_A }
Context B:
Wiki title : { wiki_title_B }
Wiki content :{ wiki_content_B }
Context C:
Wiki title : { wiki_title_C }
Wiki content :{ wiki_content_C }
Context D:
Wiki title : { wiki_title_D }
Wiki content :{ wiki_content_D }
Context E:
Wiki title : { wiki_title_E }
Wiki content :{ wiki_content_E }
The answer shold be provided in format : [
,→Reference X] where X isthe most similar
,→reference (A/B/C/D/E)
Prompt format for generation:
Based on the retrieved document , answer the
,→question
<image >
{ question } within 5 words .
Context :
Wiki title : { wiki_title_select }
Wiki content : { wiki_content_select }
Example:
# Rerank :
... Context A: ...
# See Prompt for Qwen2 -VL -1- Stage
The answer shold be provided in format : [
,→Reference X] where X isthe most similar
,→reference (A/B/C/D/E)
# Generation :
Based on the retrieved document , answer the
,→question
<image >
What isthe width ( inkilometre ) of this lake ?
,→within 5 words
Context :
Wiki title : Jewish National Fund
Wiki content : Jewish National Fund was founded
,→in1901 to buy and develop land inOttoman
,→Syria ...
Answer:
" Reference E"
" Not given "
The following is the prompt for Qwen2-VL-
MMSTaR, where wiki_title_A/B/C/D/E and wiki_content_A/
,→B/C/D/E represent the titles and textual contents of
the top-5 retrieved Wikipedia entries.
Prompt for Qwen2-VL-MMSTaR (Part I)
Prompt format:
Based on the retrieved document , answer the
,→question
<image >
{ question } within 5 words
Reference A:
<image #A>
Wiki title : { wiki_title_A }
Wiki content :{ wiki_content_A }Prompt for Qwen2-VL-MMSTaR (Part II)
Prompt format:
Reference B:
<image #B>
Wiki title : { wiki_title_B }
Wiki content :{ wiki_content_B }
Reference C:
<image #C>
Wiki title : { wiki_title_C }
Wiki content :{ wiki_content_C }
Reference D:
<image #D>
Wiki title : { wiki_title_D }
Wiki content :{ wiki_content_D }
Reference E:
<image #E>
Wiki title : { wiki_title_E }
Wiki content :{ wiki_content_E }
Note : Analyze step -by - step and provide final
,→answer in format : []
Example:
Based on the retrieved document , answer the
,→question
<image >
What place inflows lake ? within 5 words .
Reference A: ...
# See Prompt for Qwen2 -VL -1- Stage
Note : Analyze step -by - step and provide final
,→answer in format : []
Answer:
" Step 1: Key Information Analysis
Query Image Analysis :
The query image shows a scenic view of a lake
,→surrounded by mountains and a town . The lake
,→is prominent in the center , with a town on
,→its shores .
Question Analysis :
The question asks about the place that inflows
,→into the lake .
Step 2: Reference Comparison :
Reference A: Describes Gsteigwiler , a village
,→near a lake .
Reference B: Describes Walensee , a lake in
,→Switzerland .
Reference C: Describes Schnepfau , a village near
,→ a lake .
Reference D: Describes Lake Lucerne , a lake in
,→Switzerland .
Reference E: Describes Lake Brienz , a lake in
,→Switzerland .
Relevant Reference Selection :
Reference B ( Walensee ) and Reference D ( Lake
,→Lucerne ) both describe lakes in Switzerland ,
,→but Reference B is more likely to have a
,→river inflowing into it , as it is a larger
,→lake .
Step 3: Answer Generation
Answer : The place that inflows into Lake Walensee
,→ is the Seez river . According to the
,→description in Reference B, the Seez ,
,→Murgbach , and Linth rivers lead to the lake .
,→The Seez river is the most likely candidate
,→for inflowing into the lake , so the final
,→answer is [ Seez river ]"
The following is the prompt for CoRe-MMRAG,
where wiki_title_A/B/C/D/E and wiki_content_A/B/C/D/E rep-
resent the titles and textual contents of the top-5
retrieved Wikipedia entries.

Prompt for CoRe-MMRAG
Prompt format:
Question :
<image >
{ question }
Step 1:
If the question with image isdirectly
,→answerable , provide the answer in[] less
,→than 5 words .
Ifnot, proceed to Step 2.
Step 2:
Identify the most similar Wikipedia reference to
,→ the { question }
Reference A:
<image #A>
Wiki title : { wiki_title_A }
Wiki content :{ wiki_content_A }
Reference B:
<image #B>
Wiki title : { wiki_title_B }
Wiki content :{ wiki_content_B }
Reference C:
<image #C>
Wiki title : { wiki_title_C }
Wiki content :{ wiki_content_C }
Reference D:
<image #D>
Wiki title : { wiki_title_D }
Wiki content :{ wiki_content_D }
Reference E:
<image #E>
Wiki title : { wiki_title_E }
Wiki content :{ wiki_content_E }
Step 3:
Based on the most similar reference , answer the
,→question
<image >
{ question }
within 5 words
Step 4:
If the answers from Step 1 and Step 3 differ ,
,→determine which one isthe final answer ,
,→output the final answer inless than 5 words .
Example:
Question :
<image >
What isthe width ( inkilometre ) of this lake ?
Step 1:
If the question with image isdirectly
,→answerable , provide the answer in[] less
,→than 5 words .
Ifnot, proceed to Step 2.
Step 2:
Identify the most similar Wikipedia reference to
,→ What isthe width ( inkilometre ) of this
,→lake ?
Reference A ...
# See Prompt for Qwen2 -VL -1- Stage
Step 3:
Based on the most similar reference , answer the
,→question
<image >
{ question }
within 5 words
Step 4:
If the answers from Step 1 and Step 3 differ ,
,→determine which one isthe final answer ,
,→output the final answer inless than 5 words .
Answer:
"2.8 km"A.2 Case Study
A.2.1 VTKI Examples
The cases for VTKI are illustrated in Figure 4. For
each input query, we present the top-5 retrieved
references (A–E) ranked by embedding similar-
ity between the reference images and the query
image, computed using EV A-CLIP-8B. The exam-
ples in the figure demonstrate that the Qwen-2-VL
model produces different outputs when ranking is
based solely on image modality versus text modal-
ity. When relying only on image similarity, the
model may incorrectly select visually similar but
semantically irrelevant references. For instance, in
the second row of Figure 4, the model selects refer-
ence C as the most similar, but its associated text
discusses Bursaphelenchus xylophilus, which is un-
related to the question "How many offspring can
this bird produce at the same time?" Conversely,
when using only textual information, as shown in
the fourth row of Figure 4, the model mistakenly se-
lects reference B over reference C. Both references
describe food items resembling pudding, but with-
out visual cues, the model cannot determine which
is more appropriate. These examples highlight the
VTKI problem, where unimodal approaches may
lead to incorrect reference selection. In contrast,
our proposed CoRe-MMRAG method effectively
addresses this issue by leveraging both modalities.
When unimodal outputs diverge but include the cor-
rect answer, CoRe-MMRAG enables the model to
identify and choose the correct reference.
A.2.2 PRKI Examples
The cases for PRKI are illustrated in Figure 5,
which follows the same ranking settings as Figure 4.
These examples demonstrate that the presence of
noisy information in external data can negatively
impact the model’s ability to express accurate para-
metric knowledge, leading to incorrect outputs. In
such PRKI scenarios, our proposed model effec-
tively mitigates this issue. Notably, the first and
second rows in Figure 5 show that our method suc-
cessfully outputs the correct parametric knowledge,
explicitly marked with square brackets ("[]").

Colonial WilliamsburgColonial Williamsburg is a living-history museum and private foundation presenting a part of the historic ... 
List of the oldest buildings in South CarolinaThis article attempts to list the oldest extant buildings surviving in the state of South Carolina... 
Colonial WilliamsburgColonial Williamsburg is a living-history museum and private foundation presenting a part of the historic ...
National Register of Historic Places listings in Virginia Beach, VirginiaThis is a list of the National Register of Historic Places listings in Virginia Beach... 
Cherokee PhoenixCherokee Phoenix is the first newspaper published by Native Americans in the United States and the first published ...What is the parent organization of this building?Ground-Truth Reference: ABy Image Relevance: BBy Text Relevance: ACoRe-MMRAG: A
Great spotted woodpeckerThe great spotted wood-pecker is a medium-sized woodpecker with pied black and white plumage and a red patch ... Great spotted woodpeckerThe great spotted wood-pecker is a medium-sized woodpecker with pied black and white plumage and a red patch ... BursaphelenchusxylophilusBursaphelenchusxylophilus, commonly known as pine wood nematode or pine wilt nematode (PWN) , is a species of nematode...Woodpecker Woodpecker are part of the family Picidae, which also includes the piculets, wrynecks, and sapsuckers. Members of this family ... Great spotted woodpeckerThe great spotted wood-pecker is a medium-sized woodpecker with pied black and white plumage and a red patch ... How many offspring can this bird produce at the same time?Ground-Truth Reference: ABy Image Relevance: CBy Text Relevance: ACoRe-MMRAG: A
Wadi NumeiraWadi Numeirais a Wadi in Jordan that is known for its deep gorge cut through the sandstone. It gives its name to the Bronze ...KalyaKalya is an Israeli settlement organized as a kibbutz in the West Bank. It was originally established in 1929 ... Ein GediEin Gedi, also spelled En Gedi, meaning "spring of the kid", is an oasis and a nature reserve in Israel, located west of...Dead Sea The Dead Sea, also known by other names, is a salt lake bordered by Jordan to the east and Israel and the West Bank to the west.. Jewish National Fund It has also built 180 dams and reservoirs, developed 250,000 acres (1,000km²) of land and established more than 1,000 parks ...Whatis the width (in kilometer) of this lake?Ground-Truth Reference: DBy Image Relevance: DBy Text Relevance: ECoRe-MMRAG: D
Coconut The coconut tree (Cocos nucifera) is a member of the palm tree family (Arecaceae) and the only living species of the genus Cocos... Jell-O Jell-O is a variety of gelatin desserts (fruit-flavored gels), puddings, and no-bake cream pies. The original Jell-O ... Crème caramel Crème caramel, flan, caramel pudding or caramel custard is a custard dessert with a layer of clear caramel sauce...Dessert  Dessert is a course that concludes a meal. The course consists of sweet foods, such as confections, and possibly ... Jell-O Jell-O is a variety of gelatin desserts (fruit-flavored gels), puddings, and no-bake cream pies. The original Jell-O ... Who is the owner of this food?Ground-Truth Reference: BBy Image Relevance: BBy Text Relevance: CCoRe-MMRAG: B
349th Squadron (Belgium) 349th Squadron is a fighter squadron in the Air Com-ponent of the Belgian Armed Forces. The squa-drontraces its origins...RAF Leuchars Royal Air Force Leuchars or RAF Leuchars was a Royal Air Force (RAF) station located in Leuchars, Fife ... Supermarine Spitfire The Supermarine Spitfire is a British single-seat fighter aircraft used by the Royal Air Force and other Allied countries before ...Supermarine Spitfire The Supermarine Spitfire is a British single-seat fighter aircraft used by the Royal Air Force and other Allied countries before ... Supermarine Spitfire The Supermarine Spitfire is a British single-seat fighter aircraft used by the Royal Air Force and other Allied countries before ... What is the date this aircraft took the first flight?Ground-Truth Reference: CBy Image Relevance: ABy Text Relevance: CCoRe-MMRAG: C
Figure 4: Qualitative results on sample image-question pairs from the InfoSeek dataset. The leftmost column
displays the input query image. References A–E are ordered by descending embedding similarity (computed using
EV A-CLIP-8B) between the reference images and the query image. "By image relevance" presents the Qwen-2-VL
output of most relevant references based solely on reference images, while "By text relevance" relies only on
reference texts. Our proposed CoRe-MMRAG model leverages multimodal information to select the most relevant
references.

List of extreme points of the Netherlands This is a list of the extreme points of the Netherlands, the points that are farther up, down, north, south, east…StrombolicchioStrombolicchiois a sea stack of volcanic origin 2km (1.2mi) to the northeast of the island of Stromboli in the Aeolian...PanareaPanareais the smallest of the seven inhabited Aeolian Islands, a volcanic island chain in north of Sicily, southern Italy…LehuaLehua Island is a small, crescent-shaped island in the Hawaiian islands, 0.7 miles (1.1km) north of Niʻihau... Volcano Volcano is a rupture in the crust of a planetary-mass object, such as Earth, that allows hot lava, volcanic ash, and gases to …What country does this island belong to?Ground-Truth: ItalyQwen2-VL-Param: ItalyQwen2-VL-1-Stage:The NetherlandsCoRe-MMRAG:[Italy]
Water landingIn aviation, a water landing is, in the broadest sense, an aircraft landing  on a body of water. Seaplanes, such as floatplanes and flying boats..Maldivian Air TaxiMaldivian Air Taxi (MAT) was a domestic carrier in the Maldives and was one of the largest seaplane operators in the world ... De Havilland Canada DHC-6 Twin Otter The de Havilland Canada DHC-6 Twin Otter is an aircraft developed by de Havilland Canada ....De Havilland Canada DHC-6 Twin Otter The de Havilland Canada DHC-6 Twin Otter is an aircraft developed by de Havilland Canada .... De Havilland Canada DHC-6 Twin Otter The de Havilland Canada DHC-6 Twin Otter is an aircraft developed by de Havilland Canada .... Who is the developer of this aircraft? Ground-Truth: de Havilland CanadaQwen2-VL-Param: de Havilland CanadaQwen2-VL-1-Stage:Viking AirCoRe-MMRAG:[de Havilland Canada]
Czech Airlines  Czech Airlines is the flag carrier of the Czech Republic. Its head office is located inthe Vokovicearea of Prague's 6th district …List of defunct airlines of Bulgaria This is a list of defunct airlines of Bulgaria.Airbus A310 The Airbus A310 is a wide-body aircraft.By the end of production, a total of 255 A310shad been ordered and delivered…Finnair Finnair is the flag carrier and largest airline of Finland, with its head-quarters in Vantaa on the grounds  ... Mahan Air Mahan Air, operating under the name Mahan Air, is a privately owned Iranian airline based in Tehran, Iran. It operates scheduled…What is the total quantity of produced items for this type of aircraft? Ground-Truth: 255Qwen2-VL-Param: 1300Qwen2-VL-1-Stage:255CoRe-MMRAG:255
List of Category A listed buildings in South Lanarkshire  This is a list of Category A listed buildings in South Lanarkshire... List of Category A listed buildings in South Lanarkshire  This is a list of Category A listed buildings in South Lanarkshire... New Lanark New Lanark is a village on the River Clyde, appro-ximately1.4 miles (2.2 kilometres) from Lanark, in South Lanarkshire....List of Category A listed buildings in South Lanarkshire  This is a list of Category A listed buildings in South Lanarkshire... List of Category A listed buildings in South Lanarkshire  This is a list of Category A listed buildings in South Lanarkshire... Which historic county does this village belong to? Ground-Truth: South LanarkshireQwen2-VL-Param: ScotlandQwen2-VL-1-Stage:South LanarkshireCoRe-MMRAG:South Lanarkshire
SivriaSivriais a 2,591m high peak in the Pirinmountain range, south-western Bulgaria. It is located inthe northern part of ... Blagoevgrad Province Blagoevgrad Province also known as PirinMacedonia or Bulgarian Macedonia is a province (oblast) of southwestern Bulgaria ...Province of BresciaThe Province of Brescia is a Province in the Lombardy administrative region of northern Italy. It has a population....RilaNational Park RilaNational Park is the largest national park in Bulgaria spanning an area of 810.46km² (312.92sqmi; 200,270 acres) ... 
PirinNational Park PirinNational Park, originally named Vihren National Park, The park is situated in Blagoevgrad Province…Which city or region does this park locate in?Ground-Truth: Blagoevgrad ProvinceQwen2-VL-Param: BulgariaQwen2-VL-1-Stage:Blagoevgrad ProvinceCoRe-MMRAG:Blagoevgrad Province
Figure 5: Qualitative final answer outputs on sample image-question pairs from the InfoSeek dataset. The leftmost
column shows the input query image. References A–E are ordered by descending embedding similarity (computed
using EV A-CLIP-8B) between the reference images and the query image. A star symbol indicates the ground-truth
reference.