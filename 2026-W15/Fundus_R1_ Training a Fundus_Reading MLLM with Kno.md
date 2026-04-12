# Fundus-R1: Training a Fundus-Reading MLLM with Knowledge-Aware Reasoning on Public Data

**Authors**: Yuchuan Deng, Qijie Wei, Kaiheng Qian, Jiazhen Liu, Zijie Xin, Bangxiang Lan, Jingyu Liu, Jianfeng Dong, Xirong Li

**Published**: 2026-04-09 14:55:22

**PDF URL**: [https://arxiv.org/pdf/2604.08322v1](https://arxiv.org/pdf/2604.08322v1)

## Abstract
Fundus imaging such as CFP, OCT and UWF is crucial for the early detection of retinal anomalies and diseases. Fundus image understanding, due to its knowledge-intensive nature, poses a challenging vision-language task. An emerging approach to addressing the task is to post-train a generic multimodal large language model (MLLM), either by supervised finetuning (SFT) or by reinforcement learning with verifiable rewards (RLVR), on a considerable amount of in-house samples paired with high-quality clinical reports. However, these valuable samples are not publicly accessible, which not only hinders reproducibility but also practically limits research to few players. To overcome the barrier, we make a novel attempt to train a reasoning-enhanced fundus-reading MLLM, which we term Fundus-R1, using exclusively public datasets, wherein over 94\% of the data are annotated with only image-level labels. Our technical contributions are two-fold. First, we propose a RAG-based method for composing image-specific, knowledge-aware reasoning traces. Such auto-generated traces link visual findings identified by a generic MLLM to the image labels in terms of ophthalmic knowledge. Second, we enhance RLVR with a process reward that encourages self-consistency of the generated reasoning trace in each rollout. Extensive experiments on three fundus-reading benchmarks, i.e., FunBench, Omni-Fundus and GMAI-Fundus, show that Fundus-R1 clearly outperforms multiple baselines, including its generic counterpart (Qwen2.5-VL) and a stronger edition post-trained without using the generated traces. This work paves the way for training powerful fundus-reading MLLMs with publicly available data.

## Full Text


<!-- PDF content starts -->

Fundus-R1: Training a Fundus-Reading MLLM with
Knowledge-Aware Reasoning on Public Data
Yuchuan Deng
Renmin University of China
ChinaQijie Wei
Renmin University of China
ChinaKaiheng Qian
Renmin University of China
China
Jiazhen Liu
The Hong Kong University of Science
and Technology
Hong Kong SAR, ChinaZijie Xin
Renmin University of China
ChinaBangxiang Lan
Renmin University of China
China
Jingyu Liu
Renmin University of China
ChinaJianfeng Dong
Zhejiang Gongshang University
ChinaXirong Li∗
Renmin University of China
China
Figure 1: Showcases of multimodal large language model (MLLM) based fundus image reading. Our proposed Fundus-R1 is a
reasoning-enhanced model trained exclusively on public data. Best viewed on screen.
Abstract
Fundus imaging such as CFP, OCT and UWF is crucial for the early
detection of retinal anomalies and diseases. Fundus image under-
standing, due to its knowledge-intensive nature, poses a challenging
vision-language task. An emerging approach to addressing the task
is to post-train a generic multimodal large language model (MLLM),
either by supervised finetuning (SFT) or by reinforcement learning
with verifiable rewards (RLVR), on a considerable amount of in-
house samples paired with high-quality clinical reports. However,
these valuable samples are not publicly accessible, which not only
hinders reproducibility but also practically limits research to few
players. To overcome the barrier, we make a novel attempt to train a
reasoning-enhanced fundus-reading MLLM, which we term Fundus-
R1, using exclusively public datasets, wherein over 94% of the data
∗Corresponding author: Xirong Li (xirong@ruc.edu.cn)are annotated with only image-level labels. Our technical contri-
butions are two-fold. First, we propose a RAG-based method for
composing image-specific, knowledge-aware reasoning traces. Such
auto-generated traces link visual findings identified by a generic
MLLM to the image labels in terms of ophthalmic knowledge. Sec-
ond, we enhance RLVR with a process reward that encourages
self-consistency of the generated reasoning trace in each rollout.
Extensive experiments on three fundus-reading benchmarks, i.e.,
FunBench, Omni-Fundus and GMAI-Fundus, show that Fundus-R1
clearly outperforms multiple baselines, including its generic coun-
terpart (Qwen2.5-VL) and a stronger edition post-trained without
using the generated traces. This work paves the way for training
powerful fundus-reading MLLMs with publicly available data.arXiv:2604.08322v1  [cs.CV]  9 Apr 2026

XX’XX, XXXX, XXXX Yuchuan Deng, Qijie Wei, Kaiheng Qian, Jiazhen Liu, Zijie Xin, Bangxiang Lan, Jingyu Liu, Jianfeng Dong, and Xirong Li
Keywords
Fundus image understanding, multimodal LLM, reasoning-enhanced
post-training
1 Introduction
The retina, our body’s sole visual sensor, is primarily examined
via fundus imaging. Color fundus photography (CFP) offers an en
face view of the retina, while optical coherence tomography (OCT)
provides cross-sectional visualizations [ 19]. The ability of retinal
experts to read these images – identifying lesions, discerning their
type, location, and quantity, and subsequently diagnosing retinal
diseases – is a critical skill cultivated through years of rigorous
practice. This paper aims to imbue a multimodal large language
model (MLLM) with this skill. The resultant MLLM, which we term
Fundus-R1 , is obtained with novelreasoning-enhanced training on
existing public datasets currentlylackingreasoning-trace annota-
tions for supervised learning.
Fundus image understanding is inherently knowledge-intensive,
making it a challenging vision-language task. As shown in Fig. 1,
unlike common image recognition, fundus reading requires the
model not only to identify (subtle) visual findings in a given fundus
image, but also to leverage highly specific domain knowledge to
decode these findings into diagnostic output. Accordingly, recent
efforts on this task have increasingly focused on post-training a
generic MLLM into a powerful fundus-image reader [ 17,20,23,38].
Current fundus-reading MLLMs mostly rely on richer supervi-
sion constructed from a mixture of public and private resources,
such as in-house samples paired with high-quality clinical reports
or private retinal images, see Tab. 1. In particular,the most in-
formative supervision in these methods largely comes from
the private resources. While such resources have substantially
advanced the field, they are unfortunately not publicly accessible.
Such a limitation not only weakens research reproducibility but,
more importantly, restricts further progress to a small number of
groups that have privileged access to proprietary data.
Table 1: Fundus-R1 versusexisting fundus-reading MLLMs.
The star symbol (*) indicates models publicly accessible and
thus evaluated in our experiments.
Model Training Data (Raw annotations)Reasoning
TraceRL Reward
DeepDR-LLM [18] private (clinical reports)✗–
VisionFM [28] public + private (clinical reports)✗–
VisionUnite [20] public + private (clinical reports)✓–
RetinalGPT [45] public + private (retinal images)✓–
EyeCareGPT [17] public + private (clinical reports)✓–
FundusExpert* [23] public + private (clinical reports)✓–
OphthaReason* [38] public (image-level labels + clinical reports)✓Answer+Format
Fundus-R1 public (94% image-level labels) ✓ Answer+Format+Process
Some initial efforts have been made to gather clinical reports
from public resources. For instance, OphthaReason [ 38] attempts
to extract such reports from PubMed Central (PMC)1. However, as
these reports are primarily released for educational purposes, their
quantity is inherently limited.
1https://pmc.ncbi.nlm.nih.gov/When relying on training with public fundus-image datasets, a
major challenge is that most of these datasets provide only image-
level labels, see Tab. 2. Such holistic supervision may inform the
model of the correct answer, but provides little guidance on how
visual findingsfrom fundus images should be organized into a
diagnostic reasoning trace. This raises a key research question:
can a reasoning-enhanced fundus-reading MLLM be trained
using exclusively public data?
Table 2: Training data sources for Fundus-R1 . To avoid data
leakage, images included in our test sets,i.e.FunBench [ 37],
Omni-Fundus [ 10] and GMAI-Fundus [ 42], are excluded.
Over 94% of our training samples are with image-level la-
bels only.
Dataset #Samples Pixel labeled Primary tasks
Image modality: CFP
FGADR-143-9 [13] 143 143 DR-specific lesion segmentation
IDRiD [26] 413 54 DR grading
GRAPE [11] 631 0 Glaucoma progression prediction
JSIEC [2] 800 0 35-class disease classification
Retinal-Lesions [36] 1,264 1,264 DR grading
RFIMiD [24] 2,560 0 46-label disease classification
OIA-ODIR [16] 7,342 0 8-label disease classification
DDR [18] 8,763 531 DR grading
Image modality: OCT
OCTID [9] 458 0 5-class disease classification
OCTDL [14] 1,651 0 7-class classification and grading
OIMHS [43] 3,859 3859 4-class lesion segmentation
RETOUCH [1] 5,697 2742 Fluid segmentation
NEH [31] 13,642 0 3-class disease classification
UCSD [12] 107,312 0 4-class disease classification
Image modality: UWF
TOP [7] 10,433 0 9-label disease classification
Image modalities: CFP + UWF
DeepDRiD [22] 1,800 0 DR grading
Image modalities: CFP + OCT
MMC-AMD [35] 2170 0 4-class AMD classification
Total 168,938 8593 –
A straightforward approach to answering this question is to
prompt a generic MLLM to construct reasoning traces from exist-
ing image–label pairs, and then use these traces to post -train a base
model. This idea has demonstrated high potential for tasks such
as item counting, geometric reasoning, and mathematical problem
solving [ 39,41]. However, this strategy is ineffective in the cur-
rent context, as generic MLLMs lack sufficient ophthalmic domain
knowledge [ 27,37]. When synthesizing reasoning traces directly
from the image and label, the generated outputs often suffer from
insufficient visual evidence, improper use of domain knowledge,
or logically flawed reasoning chains (see Tab. 3), rendering them
unreliable as intermediate supervision.
We also leverage generic models, but through a carefully de-
signed pipeline that harnesses the ability of generic MLLMs for
low-level visual recognition and that of generic LLMs for struc-
tured information extraction. Specifically, we first employ retrieval-
augmented generation (RAG) to construct label- and modality-
specific domain knowledge from public ophthalmic references (Eye-
Wiki2, AAO3, and PMC). From this structured knowledge, we derive
2https://eyewiki.org/Main_Page
3https://www.aao.org/Assets/811c9cb7-279d-4b3d-9cca-032191e4891c/
638749627918470000/diabetic-retinopathy-ppp-pdf

Fundus-R1: Training a Fundus-Reading MLLM with Knowledge-Aware Reasoning on Public Data XX’XX, XXXX, XXXX
a task-conditioned vocabulary of visual findings and use a generic
MLLM to extract image-specific findings from each training im-
age. Based on the extracted visual findings and the corresponding
label-conditioned knowledge, we then compose knowledge-aware
reasoning traces. This design not only enhances the reliability of
reasoning supervision. More importantly, reasoning supervision
can now be induced from public data with mainly image-level la-
bels, even when the underlying MLLM lacks sufficient ophthalmic
knowledge. To effectively leverage the reasoning-enriched training
data for post-training based on Reinforcement Learning with Verifi-
able Rewards (RLVR), we propose a novel process-based reward in
addition to the standard answer-based and format-based rewards.
In sum, our contributions are as follows:
•We propose a RAG-based pipeline that composes image-
specific, knowledge-aware reasoning traces by explicitly dis-
entangling visual findings from label- and modality-conditioned
domain knowledge.
•We introduce an answer-dependent process reward that im-
proves RLVR by encouraging self-consistent and logically
plausible reasoning traces.
•Extensive experiments on three fundus-image reading bench-
marks,i.e.FunBench [ 37], Omni-Fundus [ 10], and GMAI-
Fundus [ 42] verify the efficacy of Fundus-R1 . A reasoning-
enhanced fundus-reading MLLM can indeed be trained using
exclusively public data, 94% of which provide image-level
labels only.Fundus-R1will be open source.
2 Related Work
Existing methods for training fundus-reading MLLMs differ not
only in the data resources they rely on, but also in how such re-
sources are transformed into supervision and optimized during
training. In practice, the form of available data largely determines
the downstream learning paradigm.
One line of work mainly improves performance by scaling up
training data, typically through the introduction of private oph-
thalmic or clinical resources, while retaining relatively coarse su-
pervision. Representative examples include DeepDR-LLM [ 15] and
VisionFM [28]. DeepDR-LLM leverages private real-world clinical
supervision together with fundus analysis modules, while VisionFM
expands training resources with large-scale public and private oph-
thalmic images as well as synthetic ophthalmic data for foundation-
style visual pretraining. Although effective, such methods mainly
rely on larger data volume, and most supervision still remains at
the level of images, labels, or coarse clinical associations, without
explicitly modeling the reasoning process.
Another line of work seeks to construct richer supervision from
raw resources, which naturally leads to a “data transformation + in-
struction tuning” paradigm. With access to private clinical reports
or hospital resources, methods such as VisionUnite [ 20], Retinal-
GPT [ 45], EyeCareGPT [ 17], and FundusExpert [ 23] transform anno-
tations, clinical reports, or retinal attributes into richer image–text
pairs, dialogues, VQA samples, report-generation data, or struc-
tured reasoning-style corpora, and then train ophthalmic MLLMs
through multi-stage pretraining or instruction tuning. Compared
with plain image-level labels, these methods provide substantiallystronger supervision, but such supervision often depends on non-
public resources.
Public-data-only solutions for reasoning-oriented supervision
are much more limited. OphthaReason [ 38] is one of the few systems
built mainly on public datasets and PMC-derived text. It retrieves
electronic clinical reports from PMC to assist the synthesis of rea-
soning traces, and further incorporates reinforcement learning with
answer and format rewards to improve reasoning-oriented gener-
ation. However, because high-quality public clinical text remains
scarce, the amount of synthesized reasoning supervision is still
limited. Moreover, even with RL, its optimization remains centered
on answer correctness and output formatting, without directly su-
pervising the faithfulness or diagnostic validity of the reasoning
trace itself.
To sum up, existing studies suggest that strong fundus-reading
MLLMs usually depend on supervision richer than public image-
level labels, while the corresponding learning paradigms are largely
determined by how such richer supervision is constructed. Differ-
ent from the prior research, we aim for constructing reasoning
traces from image-level labels and consequently exploiting the
auto-generated traces for training a reasoning-enhanced model.
3 ProposedFundus-R1Solution
3.1 Problem Statement
We aim to develop an MLLM, denoted as M, that takes a fundus
image𝐼and a task-specific question 𝑞as input. The objective of M
is to generate a correct answer ˆ𝑦along with a detailed reasoning
trace𝜏. This process can be expressed more formally as
(ˆ𝑦,𝜏)←M(𝐼,𝑞).(1)
Developing such a model is challenging, as existing public datasets
primarily offer image-level labels without accompanying reasoning
traces, see Tab. 2. To address this, we detail in Sec. 3.2 our approach
to generating these crucial reasoning traces by synergistically com-
bining ophthalmic knowledge with generic LLMs. Subsequently, we
elaborate in Sec. 3.3 how to effectively exploit these auto-generated
reasoning traces for transforming a base model (Qwen2.5-VL[ 32])
into our specializedFundus-R1.
3.2 Adding Reasoning Traces to Existing Data
As shown in Tab. 3, directly prompting a generic MLLM tends
to generate over interpreted and logically inconsistent reasoning
traces. By contrast, we propose a RAG-based method for generating
knowledge-based and image-grounded reasoning traces, see Fig. 2.
3.2.1 Label-and-Modality-Specifc VF Vocabulary Construction.Due
to the differences in imaging mechanisms between CFP and OCT,
a given lesion may present with distinct visual characteristics
across the varied image modalities. For example,drusensappear as
yellowish-white, round subretinal deposits in CFP, whereas they
manifest as mound-like elevations or nodules on OCT. Therefore,
a VF vocabulary should be constructed in a label- and modality-
specific manner. More specifically, given a label 𝑦and an image
modality𝑚, we first acquisite domain knowledgew.r.t. 𝑦and𝑚, de-
noted as𝐷𝐾[𝑦,𝑚] , from online ophthalmic references using RAG
and a generic LLML. We then conduct a Text-to-Findings operation
on𝐷𝐾[𝑦,𝑚] to obtain the vocabulary referred to as 𝑉𝑜𝑐𝑎𝑏[𝑦,𝑚] .

XX’XX, XXXX, XXXX Yuchuan Deng, Qijie Wei, Kaiheng Qian, Jiazhen Liu, Zijie Xin, Bangxiang Lan, Jingyu Liu, Jianfeng Dong, and Xirong Li
(a) Label-and-modality-specifc domain knowledge ( 𝐷𝐾) acquisition. A visual-finding (VF) vocabulary is derived by merging the required and
supportive VFs stored in𝐷𝐾.
(b) Image-specifc VF extraction
 (c) Knowledge-aware reasoning trace compo-
sition
Figure 2: Proposed method for generating visual-finding-embedded reasoning traces. Given a task label 𝑦and fundus modality
𝑚, we first emplopy a generic LLM L(Qwen3-Max) to obtain structured label-and-modality-specific domain knowledge 𝐷𝐾[𝑦,𝑚]
and a visual-finding vocabulary from authoritative ophthalmic references. For each training image 𝐼of modality 𝑚and labeled
with𝑦, we extract from the image visual findings 𝑉𝐹[𝐼] by prompting a generic MLLM M𝑣𝑓𝑒(Qwen2.5-VL-32B). Lastly, a reasoning
trace is generated for the given image by promptingLwith question𝑞, answer𝑦,𝐷𝐾[𝑦,𝑚], and𝑉𝐹[𝐼].
We formulate the above process as
 
𝑑𝑒𝑠𝑝𝑦,𝑚←RAG(𝑦,𝑚),
𝐷𝐾[𝑦,𝑚] ←L(𝑑𝑒𝑠𝑝 𝑦,𝑚,prompt),
𝑉𝑜𝑐𝑎𝑏[𝑦,𝑚] ←Text-to-Findings(𝐷𝐾[𝑦,𝑚]).(2)
RAG-based domain knowledge acquisition. Our RAG mod-
ule is implemented using Qwen3-Max [ 33] asLand LangChain
[3] for online information retrieval. Given a textual query com-
posed of the label 𝑦and the image modality 𝑚, the RAG module
finds web pages relevantw.r.t.the query from multiple sources
(Eyewiki,AAO and PMC), downloads the pages, and parses them.
Sections describing characteristic symptoms, diagnostic criteria
and modality-specific manifestations are retained to form a natural-
language description denoted by 𝑑𝑒𝑠𝑝𝑦,𝑚. For example, given 𝑦as
Moderate NPDRand 𝑚asCFP, the query reads as “What are the key
CFP findings for diagnosing Moderate NPDR ?”. The corresponding
𝑑𝑒𝑠𝑝𝑦,𝑚reads as “On CFP, Moderate NPDR is characterized by mul-
tiple microaneurysms and dot-blot intraretinal hemorrhages, which
together reflect a greater lesion burden than that seen in mild dis-
ease. These abnormalities are typically distributed across the posterior
pole and indicate progression beyond the earliest stage of diabetic
retinopathy. At the same time, there should be no evidence of neo-
vascularization. In addition to these core findings, hard exudates and
cotton-wool spots may also be observed”. We refer to the supplement
for more examples. Given 𝑑𝑒𝑠𝑝𝑦,𝑚, we promptLto act as a knowl-
edge extractor, yielding 𝐷𝐾[𝑦,𝑚] as a JSON-style structured record,
see Fig. 2a.VF vocabulary construction. Following LLM extraction, the
Text-to-Findings operation applies lightweight regex-based post
processing to remove redundant expressions, normalize lexical vari-
ants, and consolidate synonymous phrases into their canonical
forms. For instance, in CFP-based diabetic retinopathy (DR) grad-
ing, the disease labels are Mild NPDR, Moderate NPDR, Severe
NPDR, and PDR. The resultant VF vocabulary contains commonly
recognized retinal lesions such as microaneurysm (MA), retinal
hemorrhage (RH), hard exudate (HE), cotton-wool spot (CWS), vit-
reous hemorrhage (VH), and neovascularization (NV). Per label and
modality, the VFs are stored in two disjoint groups in 𝐷𝐾[𝑦,𝑚] :
required VFs, which capture core evidence explicitly noted in the
retrieved references as characteristic or decisive for the target label,
andsupportive VFs, which offer auxiliary but non-decisive cues.
3.2.2 Image-specifc VF Extraction.Recall that each training image
is already associated with a specific label 𝑦and a modality label
𝑚. Using the previously constructed VF vocabulary 𝑉𝑜𝑐𝑎𝑏[𝑦,𝑚] ,
we tackle the VF extraction task by prompting a generic MLLM,
denoted asM𝑣𝑓𝑒, with a set of binary questions, specifically asking
whether every entry in 𝑉𝑜𝑐𝑎𝑏[𝑦,𝑚] is present in the given im-
age. These questions are posed jointly via a customized prompt𝑦,𝑚:
“In this {m} image, determine the presence of the following findings:
{Vocab[y,m]}. Answer with present or absent for each finding”.
To improve the precision of VF extraction, we prompt M𝑣𝑓𝑒
five times to generate a set of five rollouts. These rollouts are then
aggregated into a set of predicted VFs, denoted 𝑉𝐹[𝐼] , by a majority

Fundus-R1: Training a Fundus-Reading MLLM with Knowledge-Aware Reasoning on Public Data XX’XX, XXXX, XXXX
Table 3: Reasoning traces generated for given VQA triplets. Free-form CoT generation tends to introduce over-interpretation
and logically inconsistent reasoning, while our method produces evidence-aligned and logically consistent reasoning.
Input VQA Free-form CoT generation (Qwen3-VL-Plus) Knowledge-aware generation (Ours)
Which level of diabetic retinopathy is
shown in the fundus image?
Moderate NPDR<think>The fundus image shows scattered
microaneurysms and dot hemorrhages around the
posterior pole. Multiple hard exudates and mild
cotton-wool spots are also visible. However, the mention of
venous beadingandintraretinal microvascular
abnormalitiessuggests more advanced vascular damage
than is actually supported by the image. Therefore, the
condition is interpreted asmoderate NPDR.</think>
<answer>Moderate NPDR</answer><think>The fundus image shows multiple
microaneurysmsanddot–blot hemorrhagesin the
posterior pole. No neovascularization, preretinal
hemorrhage, or vitreous hemorrhage is observed.
Compared with mild NPDR, the increased number and
broader distribution of these lesions support a diagnosis of
moderate NPDR.</think> <answer>Moderate
NPDR</answer>
What type of AMD is shown here?
Exudative AMD<think>The OCT shows elevation of the retinal pigment
epithelium with subretinal hyperreflective material and
mild intraretinal fluid. It also mentions outer retinal layer
thinning and ellipsoid-zone disruption, butno clear signs
of active leakageare claimed despite visible fluid. These
findings are interpreted asexudative AMD.</think>
<answer>Exudative AMD</answer><think>The OCT demonstrates alarge pigment
epithelial detachment (PED)together withsubretinal
fluid (SRF)andintraretinal fluid (IRF). This
combination is characteristic ofexudative AMDand
indicates ongoing neovascular activity.</think>
<answer>Exudative AMD</answer>
Does this fundus image show any disease?
Glaucoma<think>The ultra-widefield fundus image shows a
relatively large optic disc with an increased cup-to-disc
ratio. Mild superior rim thinning is noted, and the trace
further argues thatthe absence of retinal lesions
supports glaucoma, which is not a valid diagnostic basis.
The conclusion isglaucoma.</think>
<answer>Glaucoma</answer><think>The image shows an optic disc with amarkedly
enlarged cup-to-disc ratio. The cup is deep and vertically
elongated, withsuperior and inferior rim thinningin a
typical glaucomatous pattern. These findings indicate
structural optic nerve damage consistent with
glaucoma.</think> <answer>Glaucoma</answer>
voting strategy. For each entry in 𝑉𝑜𝑐𝑎𝑏[𝑦,𝑚] , it receives a vote
each time it appears in a rollout. An entry is added to 𝑉𝐹[𝐼] only
if its vote count exceeds two. Through this rollout-based voting
strategy, we extract VFs that M𝑣𝑓𝑒detects consistently and reliably.
The VF extraction process is summarized in Eq. 3.
 
prompt𝑦,𝑚←Make-Prompt(𝑚,𝑉𝑜𝑐𝑎𝑏[𝑦,𝑚]),
rollouts←M 𝑣𝑓𝑒(𝐼,prompt𝑦,𝑚),
𝑉𝐹[𝐼] ←Aggregate(rollouts).(3)
Note that a small portion of our training data (less than 6%, see
Tab. 2) have pixel-level lesion annotations. For these image, we
directly use their lesion labels as𝑉𝐹[𝐼].
3.2.3 Knowledge-aware Reasoning Trace Composition.Given an
image𝐼and its label 𝑦w.r.t.a specific fundus-reading task, we
generate a question 𝑞by filling out a pre-specified task-specific
template with 𝑦as the correct answer, following the common prac-
tice in previous work [ 10,37]. For the VQA triplet (𝐼,𝑞,𝑦) , we
propose to construct a reasoning trace in a knowledge-aware man-
ner, by promptingLto generate the trace conditioned on both VFs
𝑉𝐹[𝐼] and𝐷𝐾[𝑦,𝑚] . In particular,Lis guided to (i) summarize
the visual findings from 𝑉𝐹[𝐼] , (ii) connect the findings to the do-
main knowledge from 𝐷𝐾[𝑦,𝑚] , and (iii) derive the final answer
𝑦accordingly. We empirically observe that compared to free-formchain-of-thought generation, the above prompting strategy pro-
duces reasoning traces that are more structured and easier to verify,
see Fig. 2c.
Using the pipeline illustrated in Fig. 2, we initially generated
146,425 reasoning traces in total. A quality check was then auto-
matically performed by prompting Lto identify problematic traces,
including modality inconsistency, omission of required VFs, inclu-
sion of redundant or incorrect VFs, and omission or mismatch of
domain knowledge,etc.Further details are provided in the sup-
plement. Ultimately, 80,115 traces were retained and added to the
training data for MLLM post-training.
3.3 Improving RLVR with a Process Reward
With the reasoning-trace-enriched training data, we now proceed
to post-train a generic model to its fundus-reading counterpart
M. We adopt the widely used RLVR algorithm based on Group
Relative Policy Optimization (GRPO) [ 29]. Given a specific VQA
triplet(𝐼,𝑞,𝑦) as a training example, the basic idea of GRPO is
to first letMgenerate a group of 𝐺different outputs, commonly
known as rollouts, which are denoted by {(ˆ𝑦𝑖,𝜏𝑖)}𝐺
𝑖=1, where ˆ𝑦𝑖
is the predicted answer in the rollout 𝑖and𝜏𝑖as the generated
reasoning trace. Based on ˆ𝑦𝑖,𝜏𝑖and the input, a reward value 𝑟𝑖for
the rollout𝑖is calculated. The average reward of the group is then
used as a baseline to calculate a relative advantage for each rollout.
The model is optimized by maximizing these relative advantages.
The calculation of𝑟 𝑖is thus critical.

XX’XX, XXXX, XXXX Yuchuan Deng, Qijie Wei, Kaiheng Qian, Jiazhen Liu, Zijie Xin, Bangxiang Lan, Jingyu Liu, Jianfeng Dong, and Xirong Li
Prior work, such as OphthaReason [ 38], determines the reward
based solely on whether the predicted answer ˆ𝑦𝑖is correct and
whether the generated trace 𝜏adheres to the required format. Specif-
ically, the reward is computed as the sum of a binary answer-based
reward𝑟𝑎𝑛𝑠,𝑖and a binary format-based reward 𝑟𝑓𝑚𝑡,𝑖, thereby com-
pletely disregarding the quality of 𝜏itself. By contrast, we explicitly
account for trace quality by introducing a process reward𝑟 𝑝𝑟𝑜,𝑖.
(a) Answer-dependent process reward computation.
(b) Correct rollout with a logicallyplausiblereasoning trace.
(c) Correct rollout with a logicallytenuousreasoning trace.
(d) Correct rollout with a logicallyflawedreasoning trace.
(e)Incorrectrollout with a logicallyplausiblereasoning trace.
(f)Incorrectrollout with a logicallyflawedreasoning trace.
Figure 3: Our process reward𝑟 𝑝𝑟𝑜. Best viewed digitally.
Our design of 𝑟𝑝𝑟𝑜,𝑖 is guided by the following considerations.
Ideally, we wantMto produce a correct answer based on correct
visual findings. Therefore, when ˆ𝑦𝑖matches𝑦,𝑟𝑝𝑟𝑜,𝑖should verify
that𝜏aligns with𝑉𝐹[𝐼] ,i.e.the visual findings previously extracted
from the training image 𝐼. However, such a criterion might be over
strict, especially in the early stages of training when the model hasnot yet learned to answer correctly. To address this, even when ˆ𝑦𝑖
is incorrect, the model can still receive a reward if its reasoning
process remains logically plausiblew.r.t. 𝐷𝐾[ ˆ𝑦𝑖,𝑚],i.e.the domain
knowledge associated with the incorrect answer. Taking these into
account, we propose to compute 𝑟𝑖,𝑝𝑟𝑜 in an answer-dependent
manner as follows:
𝑟pro,𝑖= 
LLM-as-judge(𝜏 𝑖,𝑉𝐹[𝐼]), ˆ𝑦𝑖=𝑦,
LLM-as-judge(𝜏 𝑖,𝐷𝐾[ ˆ𝑦𝑖,𝑚]), ˆ𝑦𝑖≠𝑦,(4)
where LLM-as-judge returns a value by prompting Lto assess𝜏𝑖
against the provided reference, which is either 𝑉𝐹[𝐼] or𝐷𝐾[ ˆ𝑦𝑖,𝑚]) .
In particular, when ˆ𝑦𝑖is correct,Lis instructed to rate the trace
asplausible,tenuous, orflawed, corresponding to reward values
of0.4,0and−0.4, respectively, see Fig. 3. When ˆ𝑦𝑖is incorrect,L
evaluates whether the trace is logically plausible, yielding a reward
value of0.2if plausible and0otherwise. Note that summing the two
terms in Eq. 4 is prone to reward hacking, resulting in suboptimal
performance. The overall reward 𝑟𝑖is obtained by summing 𝑟𝑎𝑛𝑠,𝑖,
𝑟𝑓𝑚𝑡,𝑖 and𝑟𝑝𝑟𝑜,𝑖.
3.4 Two-Stage Training
Note that directly post-training the base model using RLVR is prob-
lematic, as rollouts in the early stages of training are predominantly
incorrect and the generated reasoning traces often vary substan-
tially in length and format. As a consequence, reward values tend to
be near zero, making RLVR slow to boot up. We therefore employ
a two-stage training procedure. In the first stage, SFT is performed
to adapt the model to the various fundus image reading tasks. We
then refine the model in the second stage using the process-reward-
enhanced RLVR.
4 Experiments
4.1 Experimental Setup
Test sets. We adopt three public test sets: FunBench [ 37], Omn-
iMedVQA [ 10], and GMAI-MMBench [ 42]. FunBench evaluates
an MLLM’s fundus reading skills via a hierarchical task organiza-
tion across four levels, includingmodality perception(L1),anatomy
perception(L2),lesion analysis(L3), anddisease diagnosis(L4). For
OmniMedVQA and GMAI-MMBench, we adopt their subsets re-
lated to fundus image reading. In particular, the following is selected
from OmniMedVQA:modality recognition(MR),anatomy identifi-
cation(AI),lesion grading(LG), anddisease diagnosis(DD). As for
GMAI-MMBench, we useattribute recognition(AR),nervous tissue
recognition(NT),blood vessels recognition(BVR),disease diagno-
sis(DD), andseverity grading(SG). For the ease of reference, we
rename the two subsets asOmni-FundusandGMAI-Fundus,
respectively. Tab. 4 shows the basic statistics of the three test sets.
Table 4: Three test sets used in our experiments.
Test set Modalities(#Images)#VQA triplets
FunBench [37] CFP (7,608) / OCT (6,076) / UWF (2,664) 91,810
Omni-Fundus [10] CFP (8,209) / OCT (3,791) 12,000
GMAI-Fundus [42] CFP (1,327) / OCT (311) /UWF(63) 1,718

Fundus-R1: Training a Fundus-Reading MLLM with Knowledge-Aware Reasoning on Public Data XX’XX, XXXX, XXXX
Table 5: Performance of varied training setups. Bold and underline denote the best and second-best, respectively.
# Training SetupOverall FunBench Omni-Fundus GMAI-Fundus
Avg FunBench Omni-Fundus GMAI-Fundus L1 L2 L3 L4 MR AI LG DD AR NT BVR DD SG
A: Base (Qwen2.5-VL-3B) 48.3 41.0 66.4 37.4 88.1 40.8 21.7 13.6 96.4 87.4 44.6 37.3 31.3 25.3 74.7 37.1 18.6
Training without reasoning trace:
B:A+ SFT 53.8 58.9 71.1 31.4 69.395.820.4 50.0 94.2 91.9 51.7 46.4 24.7 16.0 65.3 33.4 17.6
C:A+ GRPO 55.5 53.9 68.1 44.6 76.8 71.5 27.2 40.2 92.4 92.8 36.2 50.9 38.7 27.385.3 32.0 39.7
D:A+ SFT + GRPO 52.5 53.1 71.3 33.0 88.4 57.7 30.5 35.7 95.4 93.8 43.4 52.7 28.0 19.3 37.3 43.3 37.1
Training with reasoning traces:
E:A+ SFT 56.5 61.9 67.7 39.8 90.4 83.8 18.155.4 97.689.3 26.2 57.7 38.0 10.0 81.3 41.2 28.5
F:A+ SFT + GRPO 59.0 64.7 71.4 41.0 92.1 94.3 29.8 42.7 92.7 87.2 53.7 51.8 22.0 24.7 72.0 51.0 35.2
G:Fundus-R1-3B 65.6 67.1 79.8 50.1 90.5 90.9 33.9 52.9 93.8 92.1 63.4 69.7 38.0 16.7 92.0 57.0 46.9
Ablation on the process reward:
H: Using the VF item only 60.3 63.2 75.6 42.1 91.0 89.7 30.7 41.6 93.2 91.7 54.5 63.0 31.3 14.0 69.3 51.5 44.2
I: Using the DK item only 63.8 66.6 77.4 47.5 92.8 94.5 31.4 47.9 97.0 93.8 58.4 60.340.711.3 90.7 52.9 41.7
J: Summing the VF and DK items 61.8 61.7 76.7 47.093.482.6 31.7 39.1 94.396.057.1 59.5 35.3 12.0 88.0 52.647.1
Performance metrics. For each test set, we report its official
metric: F1-score for FunBench, and accuracy for Omni-Fundus and
GMAI-Fundus.
Details of implementation. We use bf16 precision on 8 H800
GPUs (80GB) for training and RTX 3090 GPUs for inference. Subject
to our computational capacity, we adopt Qwen2.5-VL [ 32] (3B/7B)
as our base model. SFT is conducted with LLaMA-Factory [ 44],
while RLVR is performed within the verl framework [ 30]. In the
SFT stage, the model is trained for 2 epochs with a learning rate of
3𝑒−5. In the reinforcement learning stage, the sampling tempera-
ture is set to 1.0, and 4 rollouts are generated for each prompt. RL
training is performed for 8 epochs with a learning rate of1 𝑒−6.
AdamW is used as the optimizer in both stages. More detailed hyper-
parameter settings and prompts are provided in the supplementary
material. Since the answer format may vary across MLLMs, we
adopt VLMEvalKit [ 8] as a unified answer extraction tool to ensure
a fair comparison.
4.2 Exp-1. Trainingw/orw/oReasoning
We first evaluate whether enriching the training data with the gen-
erated reasoning traces is necessary. Tab. 5 shows the performance
of training setups without reasoning traces (Setup B–D) and with
reasoning traces (Setup E–G). The clearly better performance of
the latter group confirms the necessity of incorporating reasoning
traces into the standard VQA-triplet based training data.
Let us take a closer look at Setup B, which adapts the base model
by SFT alone. Its improved overall performance (48.3 →53.8) is
largely contributed by a substantial gain on FunBench (41.0 →58.9),
albeit with a noticeable decline on GMAI-Fundus (37.4 →31.4).
Moreover, combining SFT and GRPO without reasoning trace (Setup
D) does not yield further improvement. Rather, the overall score
drops to 52.5. These results suggest that answer-only supervision is
insufficient for reliably initializing reasoning-oriented RLVR, and
that simply stacking SFT and RL without process-based supervision
may even be detrimental.
When reasoning traces are provided, SFT + GRPO (Setup F)
clearly outperforms its trace-free counterpart (Setup D), from 52.5
to 59.0. However, the lower performance of Setup F relative to SetupE on certain columns, for instance,L4in FunBench (55.4 →42.7)
andARin GMAI-Fundus (38.0 →22.0), suggests that not all earlier
gains from SFT are consistently preserved. Hence, while reasoning
traces make RLVR more promising, relying on the answer and
format-based rewards is inadequate to unleash the value of traces.
Further, we evaluate the quality of visual finding extraction on
Retinal-Lesions [ 36] that has ground truth available for multiple DR-
related lesions. As a baseline, we prompt Qwen3-VL-Plus, which is
stronger than Qwen2.5-VL-32B used in our VF extraction pipeline,
to generate CoT traces given the VQA triplets. Corresponding VFs
are then extracted from the generated trace. As shown in Tab. 6, our
method is clearly better (46.8 →62.6), though much room for im-
provement remains. Despite the imperfections in the extracted VFs,
they are embedded into the reasoning traces in a knowledge-aware
manner, rendering the traces valuable for injecting ophthalmic
knowledge into the MLLM via post-training.
Table 6: Evaluation of visual finding extraction. Test set:
Retinal-Lesions [ 36]. Performance metrics: Sensitivity, Speci-
ficity and their harmonic mean (S2).
Visual findingOurs Qwen3-VL-Plus
Sen. Spe. S2 Sen. Spe. S2
Microaneurysm (MA) 78.3 49.5 60.7 58.4 23.3 33.3
Retinal hemorrhage (RH) 73.7 44.4 55.4 45.5 51.0 48.1
Hard exudate (HE) 78.7 58.1 66.9 49.9 65.8 56.7
Cotton-wool spot (CWS) 48.2 94.4 63.8 31.9 85.3 46.4
Vitreous hemorrhage (VH) 62.5 75.4 68.3 37.5 81.0 51.3
Neovascularization (NV) 55.8 66.3 60.6 32.6 72.2 44.9
Avg. 66.2 64.762.642.6 63.1 46.8
4.3 Exp-2. Ablation on the Process Reward
Since the only difference between Setup G and Setup F is the use
of𝑟𝑝𝑟𝑜, the superior performance of Setup G over Setup F (59.0 →
65.6) verifies the effectiveness of the process reward. Gains are par-
ticularly evident on diagnosis-oriented tasks, including FunBench
L4(42.7→52.9), Omni-FundusLG(53.7 →63.4) andDD(51.8 →
69.7), as well as GMAI-FundusBVR(72.0 →92.0),DD(51.0→57.0),

XX’XX, XXXX, XXXX Yuchuan Deng, Qijie Wei, Kaiheng Qian, Jiazhen Liu, Zijie Xin, Bangxiang Lan, Jingyu Liu, Jianfeng Dong, and Xirong Li
Table 7:Fundus-R1versusothers. Models within each group are sorted in descending order by their overall performance.
Model Vision Encoder LLMPost-training
Data ScaleFunBench Omni-Fundus GMAI-Fundus Avg.
Generic MLLMs:
Qwen2.5-VL-3B [32] Qwen2.5-ViT Qwen2.5-3B – 41.0 66.4 37.4 48.3
InternVL2.5-8B [5] InternViT-300M InternLM2.5-7B – 51.0 55.6 48.2 51.6
Qwen2.5-VL-7B Qwen2.5-ViT Qwen2.5-7B – 46.1 68.1 42.1 52.1
Medical MLLMs (SFT)
HealthGPT-M3-7B [21] CLIP ViT-L/14-336 Phi-3-mini 1.6M 52.4 64.0 46.3 54.2
HuatuoGPT-Vision-7B [4] OpenAI CLIP ViT-L/14 Qwen2-7B 1.3M 59.0 69.6 47.3 58.6
Lingshu-7B [40] Qwen2.5-ViT Qwen2.5-7B 5.1M 55.197.758.7 70.5
Medical MLLMs (RLVR):
MedVLM-R1 [25] Qwen2-ViT Qwen2-2B 600 46.2 56.3 37.1 46.6
QoQ-Med-7B [6] Qwen2.5-ViT Qwen2.5-7B 2.6M 62.5 57.9 32.8 51.1
Fundus-reading MLLMs (SFT):
FundusExpert-8B [23] InternViT-300M InternLM2.5-7B 200K 48.3 78.471.165.9
Fundus-reading MLLMs (RLVR):
OphthaReason-Intern [38] InternViT-300M Qwen2.5-1.5B 121.5K 44.9 57.7 44.1 48.9
OphthaReason-Qwen Qwen2.5-ViT Qwen2.5-3B 121.5K 48.8 62.4 49.7 53.6
Med-R1-fundus [34] Qwen2.5-ViT Qwen2.5-3B undisclosed 48.2 79.2 36.5 54.7
Fundus-R1-3B Qwen2.5-ViT Qwen2.5-3B 80.1K 67.1 79.8 50.1 65.6
Fundus-R1-7B Qwen2.5-ViT Qwen2.5-7B 80.1K 69.2 91.1 61.1 73.8
andSG(35.2→46.9). In contrast, for low-level perception tasks
such as FunBenchL1/L2, Setup G performs slightly worse. These
results suggest that the process reward primarily benefits fundus
reading tasks that require high-level reasoning.
We observe that for the GMAI-FundusNTtask, almost all post-
trained models underperform relative to the base model. This task
involves identifying a specific OCT layer,e.g.choroidal and RPE,
from a marked-out region of an OCT image. The task is not covered
by our training collection, see Tab. 2, and thus out-of-scope for the
post-trained models.
Our ablation on the process reward (Setup H–J, Tab. 5) shows
that the DK item is more effective than its VF counterpart. Summing
the two items (Setup J) instead of using them selectively (Eq. 4) is
suboptimal. Fig. 4 further shows the benefit of the process reward:
making the model achieve higher answer rewards with shorter and
hence more concise rollouts during RLVR training.
4.4 Exp-3.Fundus-R1versusOthers
To demonstrate the challenging nature of fundus image reading,
we report the performance of various public 3B/7B MLLMs on the
three test sets. In addition to Qwen2.5-VL, which serves as our base
model, we include InternVL2.5-8B [ 5] as another generic MLLM.
For SFT-based medical MLLMs, we include HealthGPT-M3-7B [ 21],
HuatuoGPT-Vision-7B [ 4], and Lingshu-7B [ 40]. For RLVR-based
medical MLLMs, we select MedVLM-R1 [ 25] and QoQ-Med [ 6].
As for fundus-reading MLLMs, we include Med-R1-fundus [ 34],
FundusExpert-8B [23] and OphthaReason [38].
The results are summarized in Tab. 7. Fundus-R1 compares fa-
vorably against the others. Nevertheless, it is worth noting that
since the specialized models are post-trained under varied setups,
the conclusion is drawn more at a solution level, other than at an
ingredient level.
(a) The influence of𝑟 𝑝𝑟𝑜on the answer reward.
/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000016/uni00000013/uni00000013 /uni00000017/uni00000013/uni00000013 /uni00000018/uni00000013/uni00000013 /uni00000019/uni00000013/uni00000013 /uni0000001a/uni00000013/uni00000013 /uni0000001b/uni00000013/uni00000013
/uni00000037/uni00000055/uni00000044/uni0000004c/uni00000051/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000056/uni00000057/uni00000048/uni00000053/uni00000056/uni00000018/uni00000015/uni00000018/uni00000018/uni00000018/uni00000013/uni00000018/uni0000001a/uni00000018/uni00000019/uni00000013/uni00000013/uni00000032/uni00000058/uni00000057/uni00000053/uni00000058/uni00000057/uni00000003/uni0000002f/uni00000048/uni00000051/uni0000004a/uni00000057/uni0000004b/uni0000005a/uni0000004c/uni00000057/uni0000004b/uni00000052/uni00000058/uni00000057/uni00000003rpro
/uni0000005a/uni0000004c/uni00000057/uni0000004b/uni00000003rpro
(b) The influence of𝑟 𝑝𝑟𝑜on the output length ofM
Figure 4: The use of the process reward 𝑟𝑝𝑟𝑜makes the model
achieve larger answer rewards with shorter output length.
5 Conclusions
We introduced Fundus-R1 , a reasoning-enhanced fundus-reading
MLLM trained using only public datasets. Our central goal is to

Fundus-R1: Training a Fundus-Reading MLLM with Knowledge-Aware Reasoning on Public Data XX’XX, XXXX, XXXX
reduce the dependence of fundus-reading MLLMs on inaccessi-
ble in-house data and private clinical reports, while still enabling
effective reasoning-oriented post-training under predominantly
image-level supervision. To achieve this, we proposed a RAG-based
reasoning-trace construction pipeline that combines image-specific
visual findings with label- and modality-conditioned ophthalmic
knowledge, and further incorporated an answer-dependent pro-
cess reward into RLVR to improve the self-consistency of gener-
ated reasoning traces. Experiments on FunBench, Omni-Fundus,
and GMAI-Fundus showed that Fundus-R1 consistently surpasses
multiple strong baselines. These results indicate that reasoning
supervision can be effectively induced from public data and can
substantially improve fundus-reading performance, especially on
knowledge-intensive tasks such as lesion analysis and disease diag-
nosis. We hope that this work will encourage more reproducible
and accessible research on fundus-reading MLLMs.
Limitation of this study. Due to computational constraints,
we use Qwen2.5-VL (3B/7B) as our base model. Since the proposed
solution is not specifically tailored to this model, we expect the
solution to generalize effectively to post-training other generic
MLLMs for fundus image understanding.
References
[1]Hrvoje Bogunović, Freerk Venhuizen, et al .2019. RETOUCH: The retinal OCT
fluid detection and segmentation benchmark and challenge.TMI38, 8 (2019),
1858–1874.
[2]Ling-Ping Cen, Jie Ji, Jian-Wei Lin, Si-Tong Ju, Hong-Jie Lin, Tai-Ping Li, Yun
Wang, Jian-Feng Yang, Yu-Fen Liu, Shaoying Tan, et al .2021. Automatic detection
of 39 fundus diseases and conditions in retinal photographs using deep neural
networks.NComms.12, 1 (2021), 4828.
[3] Harrison Chase. 2022.LangChain. https://github.com/langchain-ai/langchain
[4]Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guim-
ing Hardy Chen, Xidong Wang, Zhenyang Cai, Ke Ji, Xiang Wan, et al .2024.
Towards injecting medical visual knowledge into multimodal llms at scale. In
EMNLP. 7346–7370.
[5]Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui,
Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al .2024. Expanding
Performance Boundaries of Open-Source Multimodal Models with Model, Data,
and Test-Time Scaling.arXiv preprint arXiv:2412.05271(2024).
[6]Wei Dai, Peilin Chen, Chanakya Ekbote, and Paul Pu Liang. 2025. QoQ-Med:
Building Multimodal Clinical Foundation Models with Domain-Aware GRPO
Training. InNeurIPS.
[7]DateCazuki. 2022. TOP: Classifier using fundus image dataset provided by
Tsukazaki Hospital. https://github.com/DateCazuki/Fundus_Diagnosis. Dataset
of fundus images from Tsukazaki Hospital, used for multi-disease classification.
Accessed: 2025-04-10.
[8]Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu,
Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al .2024. VLMEvalKit: An
open-source toolkit for evaluating large multi-modality models. InProceedings of
the 32nd ACM International Conference on Multimedia. 11198–11201.
[9]Peyman Gholami, Priyanka Roy, Mohana Kuppuswamy Parthasarathy, and Va-
sudevan Lakshminarayanan. 2020. OCTID: Optical coherence tomography image
database.Computers & Electrical Engineering81 (2020), 106532.
[10] Yutao Hu, Tianbin Li, et al .2024. OmniMedVQA: A new large-scale comprehen-
sive evaluation benchmark for medical LVLM. InCVPR.
[11] Xiaoling Huang, Xiangyin Kong, Ziyan Shen, Jing Ouyang, Yunxiang Li, Kai Jin,
and Juan Ye. 2023. GRAPE: A multi-modal dataset of longitudinal follow-up
visual field and fundus images for glaucoma management.Scientific Data10, 1
(2023), 520.
[12] Daniel S. Kermany, Michael Goldbaum, Wenjia Cai, Carolina C. S. Valentim,
Huiying Liang, Sally L. Baxter, Alex McKeown, Ge Yang, Xiaokang Wu, Fangbing
Yan, et al .2018. Identifying Medical Diagnoses and Treatable Diseases by Image-
Based Deep Learning.Cell172, 5 (Feb. 2018), 1122–1131.e9.
[13] Hoda Kheradfallah, Janarthanam Jothi Balaji, Varadharajan Jayakumar, Mo-
hammed Abdul Rasheed, and Vasudevan Lakshminarayanan. 2022. Annotation
and segmentation of diabetic retinopathy lesions: an explainable AI application.
InMedical Imaging 2022: Computer-Aided Diagnosis, Vol. 12033. SPIE, 502–511.[14] Mikhail Kulyabin, Aleksei Zhdanov, et al .2024. OCTDL: Optical coherence
tomography dataset for image-based deep learning methods.Scientific data11, 1
(2024), 365.
[15] Jiajia Li, Zhouyu Guan, et al .2024. Integrated image-based deep learning and
language models for primary diabetes care.Nature medicine30, 10 (2024), 2886–
2896.
[16] Ning Li, Tao Li, Chunyu Hu, Kai Wang, and Hong Kang. 2021. A benchmark of
ocular disease intelligent recognition: One shot for multi-disease detection. In
BMO.
[17] Sijing Li, Tianwei Lin, Lingshuai Lin, Wenqiao Zhang, Jiang Liu, Xiaoda
Yang, Juncheng Li, Yucheng He, Xiaohui Song, Jun Xiao, Yueting Zhuang, and
Beng Chin Ooi. 2025. EyecareGPT: Boosting Comprehensive Ophthalmology
Understanding with Tailored Dataset, Benchmark and Model. InACMMM.
[18] Tao Li, Yingqi Gao, Kai Wang, Song Guo, Hanruo Liu, and Hong Kang. 2019. Diag-
nostic assessment of deep learning algorithms for diabetic retinopathy screening.
Information Sciences501 (2019), 511–522.
[19] Xirong Li, Yang Zhou, Jie Wang, Hailan Lin, Jianchun Zhao, Dayong Ding, Wei-
hong Yu, and Youxin Chen. 2021. Multi-modal multi-instance learning for retinal
disease recognition. InACMMM.
[20] Zihan Li, Diping Song, Zefeng Yang, Deming Wang, Fei Li, Xiulan Zhang, Paul E
Kinahan, and Yu Qiao. 2025. VisionUnite: A Vision-Language Foundation Model
for Ophthalmology Enhanced with Clinical Knowledge.IEEE Transactions on
Pattern Analysis and Machine Intelligence(2025).
[21] Tianwei Lin, Wenqiao Zhang, Sijing Li, Yuqian Yuan, Binhe Yu, Haoyuan Li,
Wanggui He, Hao Jiang, Mengze Li, Xiaohui Song, Siliang Tang, Jun Xiao, Hui Lin,
Yueting Zhuang, and Beng Chin Ooi. 2025. HealthGPT: A Medical Large Vision-
Language Model for Unifying Comprehension and Generation via Heterogeneous
Knowledge Adaptation. InICML.
[22] Ruhan Liu, Xiangning Wang, Qiang Wu, Ling Dai, Xi Fang, Tao Yan, Jaemin
Son, Shiqi Tang, Jiang Li, Zijian Gao, et al .2022. DeepDRiD: Diabetic retinopa-
thy—grading and image quality estimation challenge.Patterns3, 6 (2022).
[23] Xinyao Liu and Diping Song. 2025. Constructing Ophthalmic MLLM for
Positioning-diagnosis Collaboration Through Clinical Cognitive Chain Reason-
ing. InICCV.
[24] Samiksha Pachade, Prasanna Porwal, Dhanshree Thulkar, Manesh Kokare, Girish
Deshmukh, Vivek Sahasrabuddhe, Luca Giancardo, Gwenolé Quellec, and Fabrice
Mériaudeau. 2021. Retinal fundus multi-disease image dataset (RFMiD): A dataset
for multi-disease detection research.Data6, 2 (2021), 14.
[25] Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li,
Chen Chen, Cheng Ouyang, and Daniel Rueckert. 2025. MedVLM-R1: Incen-
tivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via
Reinforcement Learning. InMICCAI.
[26] Prasanna Porwal, Samiksha Pachade, Ravi Kamble, Manesh Kokare, Girish Desh-
mukh, Vivek Sahasrabuddhe, and Fabrice Meriaudeau. 2018. Indian diabetic
retinopathy image dataset (IDRiD): a database for diabetic retinopathy screening
research.Data3, 3 (2018), 25.
[27] Zhenyue Qin, Yu Yin, Dylan Campbell, Xuansheng Wu, Ke Zou, Yih-Chung
Tham, Ninghao Liu, Xiuzhen Zhang, and Qingyu Chen. 2025. LMOD: A Large
Multimodal Ophthalmology Dataset and Benchmark for Large Vision-Language
Models. InNAACL.
[28] Jianing Qiu, Jian Wu, Hao Wei, Peilun Shi, Minqing Zhang, Yunyun Sun, Lin Li,
Hanruo Liu, Hongyi Liu, Simeng Hou, et al .2024. Development and validation
of a multimodal multitask vision foundation model for generalist ophthalmic
artificial intelligence.NEJM AI1, 12 (2024), AIoa2300221.
[29] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei
Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al .2024. DeepSeekMath: Pushing
the limits of mathematical reasoning in open language models.arXiv preprint
arXiv:2402.03300(2024).
[30] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang,
Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and
efficient rlhf framework. InEuroSys. 1279–1297.
[31] Saman Sotoudeh-Paima, Ata Jodeiri, Fedra Hajizadeh, and Hamid Soltanian-
Zadeh. 2022. Multi-scale convolutional neural network for automated AMD
classification using retinal OCT images.Computers in biology and medicine144
(2022), 105368.
[32] Qwen Team. 2025. Qwen2.5-VL. https://qwenlm.github.io/blog/qwen2.5-vl/
[33] Qwen Team. 2025. Qwen3-Max: Just Scale it.
[34] Rongsheng Wang. 2025. Med-R1: Encourage Medical LLM to engage in deep
thinking similar to DeepSeek-R1. https://github.com/WangRongsheng/Med-R1.
[35] Weisen Wang, Xirong Li, Zhiyan Xu, Weihong Yu, Jianchun Zhao, Dayong Ding,
and Youxin Chen. 2022. Learning Two-Stream CNN for Multi-Modal Age-Related
Macular Degeneration Categorization.IEEE Journal of Biomedical and Health
Informatics26, 8 (2022), 4111–4122.
[36] Qijie Wei, Xirong Li, Weihong Yu, Xiao Zhang, Yongpeng Zhang, Bojie Hu, Bin
Mo, Di Gong, Ning Chen, Dayong Ding, et al .2021. Learn to segment retinal
lesions and beyond. InICPR.
[37] Qijie Wei, Kaiheng Qian, and Xirong Li. 2025. FunBench: Benchmarking Fundus
Reading Skills of MLLMs. InMICCAI.

XX’XX, XXXX, XXXX Yuchuan Deng, Qijie Wei, Kaiheng Qian, Jiazhen Liu, Zijie Xin, Bangxiang Lan, Jingyu Liu, Jianfeng Dong, and Xirong Li
[38] Ruiqi Wu, Yuang Yao, et al .2025. Bridging the Gap in Ophthalmic AI: MM-
Retinal-Reason Dataset and OphthaReason Model toward Dynamic Multimodal
Reasoning.arXiv preprint arXiv:2508.16129(2025).
[39] Guowei Xu, Peng Jin, Ziang Wu, Hao Li, Yibing Song, Lichao Sun, and Li Yuan.
2025. Llava-cot: Let vision language models reason step-by-step. InICCV.
[40] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu
Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al .
2025. Lingshu: A Generalist Foundation Model for Unified Multimodal Medical
Understanding and Reasoning.arXiv preprint arXiv:2506.07044(2025).
[41] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu
Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al .2025. Mulberry:
Empowering mllm with o1-like reasoning and reflection via collective monte
carlo tree search. InNeurIPS.
[42] Jin Ye, Guoan Wang, Yanjun Li, Zhongying Deng, Wei Li, Tianbin Li, Haodong
Duan, Ziyan Huang, Yanzhou Su, Benyou Wang, et al .2024. GMAI-MMBench: AComprehensive Multimodal Evaluation Benchmark Towards General Medical AI.
InNeurIPS.
[43] Xin Ye, Shucheng He, Xiaxing Zhong, Jiafeng Yu, Shangchao Yang, Yingjiao
Shen, Yiqi Chen, Yaqi Wang, Xingru Huang, and Lijun Shen. 2023. OIMHS: An
optical coherence tomography image dataset based on macular hole manual
segmentation.Scientific Data10, 1 (2023), 769.
[44] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi
Feng, and Yongqiang Ma. 2024. LlamaFactory: Unified Efficient Fine-Tuning of
100+ Language Models. InProceedings of the 62nd Annual Meeting of the Associ-
ation for Computational Linguistics. Association for Computational Linguistics.
http://arxiv.org/abs/2403.13372
[45] Wenhui Zhu, Xin Li, Xiwen Chen, Peijie Qiu, Vamsi Krishna Vasa, Xuanzhao
Dong, Yanxi Chen, Natasha Lepore, Oana Dumitrascu, Yi Su, et al .2025. Retinal-
GPT: A Retinal Clinical Preference Conversational Assistant Powered by Large
Vision-Language Models.arXiv preprint arXiv:2503.03987(2025).