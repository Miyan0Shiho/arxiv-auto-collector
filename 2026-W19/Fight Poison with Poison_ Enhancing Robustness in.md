# Fight Poison with Poison: Enhancing Robustness in Few-shot Machine-Generated Text Detection with Adversarial Training

**Authors**: Wenjing Duan, Qi Zhou, Yuanfan Li

**Published**: 2026-05-04 09:16:57

**PDF URL**: [https://arxiv.org/pdf/2605.02374v1](https://arxiv.org/pdf/2605.02374v1)

## Abstract
Machine-generated text (MGT) detection is critical for regulating online information ecosystems, yet existing detectors often underperform in few-shot settings and remain vulnerable to adversarial, humanizing attacks. To build accurate and robust detectors under limited supervision, we adopt a threat-modeling perspective and study detector vulnerabilities from an attacker's viewpoint under an output-only black-box setting. Motivated by this perspective, we propose RAG-GuidEd Attacker Strengthens ConTrastive Few-shot Detector (REACT), an adversarial training framework that improves both few-shot detection performance and robustness against attacks. REACT couples a humanization-oriented attacker with a target detector: the attacker leverages retrieval-augmented generation (RAG) to craft highly human-like adversarial examples to evade detection, while the detector learns from these adversaries with a contrastive objective to stabilize few-shot representation learning and enhance robustness. We alternately update the attacker and the detector to enable their co-evolution. Experiments on 4 datasets with 4 shot sizes and 3 random seeds show that REACT improves average detection F1 by 4.95 points over 8 state-of-the-art (SOTA) detectors and reduces the average attack success rate (ASR) under 4 strong attacks by 3.66 percentage points.

## Full Text


<!-- PDF content starts -->

Fight Poison with Poison: Enhancing Robustness in Few-shot
Machine-Generated Text Detection with Adversarial Training
Wenjing Duan1,†, Qi Zhou1,†, Yuanfan Li1,∗
1Faculty of Electronic and Information Engineering, Xi’an Jiaotong University
†Equal contribution,∗Corresponding author
dwjduan@stu.xjtu.edu.cn
Abstract
Machine-generated text (MGT) detection is
critical for regulating online information
ecosystems, yet existing detectors often under-
perform in few-shot settings and remain vul-
nerable to adversarial, humanizing attacks. To
build accurate and robust detectors under lim-
ited supervision, we adopt a threat-modeling
perspective and study detector vulnerabilities
from an attacker’s viewpoint under an output-
only black-box setting. Motivated by this per-
spective, we proposeRAG-GuidEdAttacker
StrengthensConTrastive Few-shot Detector
(REACT), an adversarial training framework
that improves both few-shot detection perfor-
mance and robustness against attacks. RE-
ACTcouples a humanization-oriented attacker
with a target detector: the attacker leverages
retrieval-augmented generation (RAG) to craft
highly human-like adversarial examples to
evade detection, while the detector learns from
these adversaries with a contrastive objective to
stabilize few-shot representation learning and
enhance robustness. We alternately update the
attacker and the detector to enable their co-
evolution. Experiments on 4 datasets with 4
shot sizes and 3 random seeds show that REACT
improves average detection F1 by4.95points
over 8 state-of-the-art (SOTA) detectors and re-
duces the average attack success rate (ASR) un-
der 4 strong attacks by3.66percentage points.
We will release our code and datasets upon ac-
ceptance.
1 Introduction
With the rapid advancement of large language mod-
els (LLMs) such as ChatGPT (Achiam et al., 2023),
LLaMA (Dubey et al., 2024), and DeepSeek (Guo
et al., 2025), it has become straightforward to
generate fluent, highly human-like text at scale.
While this has greatly facilitated people’s daily
lives, the misuse of generated content—such as
LLM-written peer-review reports, phishing emails,
and fake news—has raised widespread concern. Tomitigate such abuse, researchers have developed a
variety of methods for detecting machine-generated
text (MGT) (Liu et al., 2022; Hu et al., 2023; Liu
et al., 2024; Bao et al., 2024; Li et al., 2025) to
inform readers about the provenance of a given
passage.
Modern MGT detectors can be broadly catego-
rized into two types: (i) metric-based detectors and
(ii) model-based detectors. Because metric-based
detectors are often fragile to adversarial attacks (Su
et al., 2023; Mitchell et al., 2023; Hans et al., 2024),
model-based detectors are widely adopted in the
literature and practice. However, some recent stud-
ies (Liu et al., 2022, 2024) show that training a
high-performing MGT detector typically requires
large-scale text data, which often limits its effec-
tiveness in data-constrained settings. As illustrated
in Figure 1, in some few-shot settings, standard
detectors typically achieve only 60–70% detection
accuracy, falling short of practical usability. Few-
shot-tailored detectors can partially improve perfor-
mance, yet their accuracy still drops substantially
even under simple attacks (Wang et al., 2024a):
minor edits to just a handful of words can cause
misclassification. Adversarial training is a com-
mon approach to improving robustness against at-
tacks (Hu et al., 2023; Koike et al., 2024; Li et al.,
2025), but a more serious issue is that adversarially
trained detectors can still degrade toward chance-
level performance under stronger attacks (e.g., the
Humanizeattack (Wang et al., 2024a), which ex-
plicitly rewrites MGT to appear more human-like).
This reveals a key gap: robust few-shot MGT detec-
tion under strong attacks remains underexplored.
Motivation.We leverageadversarial trainingto
improve both detection accuracy and robustness
of MGT detectors in few-shot settings. However,
existing adversarial training approaches (Hu et al.,
2023; Koike et al., 2024; Li et al., 2025) often
fail to remain robust against highly humanized
attacks. Since many attacks on MGT detectors
1arXiv:2605.02374v1  [cs.CR]  4 May 2026

Machine -Generated Text
User
Write a brief introduction 
about game Genshin Impact in 
about 20 words.
Rewrite the sentence in more 
human- like way, following the 
style of the example human -
written sentence below:
Only a handful of human minds can 
truly grasp the beauty and complexity 
hidden in life’s simplest moments.Genshin Impact is an open-
world action RPG featuring 
elemental combat, gacha  
characters, and an expansive 
fantasy world called Teyvat .
Few worlds invite players to wander as freely as Genshin 
Impact, where magic, adven -
ture , and quiet wonder unfold 
across the land of Teyvat .Few- shot Detection and Attack
Few-shot Det. Adversarial Det.
Machine
Written()
MachineWritten()Human
Written()
Human
Written()
Human
Written()
MachineWritten()Baseline Det.
Human
Written()
Human
Written()
Human
Written()
Genshin Impact is an open- world 
action RPG featuring elemental 
combat, gacha  characters, and an 
expansive fantasy world called Teyvat .Raw MGT
Humanize Attack
Few worlds invite players to wander 
as freely as Genshin Impact, where magic, adventure, and quiet wonder 
unfold across the land of Teyvat .Modify Attack
Genshin Impact is an open- world 
action RPG featuring elemental 
combat, gacha  characters, with  an 
expansive fantasy world named  Teyvat .Few-shot Detection Attacks in MGTFigure 1:Performance drop of model-based detectors under attacks in the few-shot settings.The evaluated
detectors include: (i) baseline detector (Baseline Det.), RoBERTa-Base (Liu et al., 2019); (ii) few-shot-tailored
detector (Few-shot Det.), CoCo (Liu et al., 2022); and (iii) adversarially trained detectors (Adversarial Det.),
RADAR (Hu et al., 2023), GREATER(Li et al., 2025) and REACT(Ours). When trained on the RAID dataset (Dugan
et al., 2024) with 64 shots, the baseline detector and RADAR achieve only 60–70% detection accuracy even on
original texts, falling short of practical usability. Across prior detectors, performance under theHumanizeattack
drops close to chance, whereas our REACTdegrades much less and remains substantially more robust.
revolve aroundhumanizingMGT to evade detec-
tion (Krishna et al., 2023; Koike et al., 2024), we
are motivated by the intuition of“fight poison with
poison”and focus on constructing highly human-
ized adversarial examples to strengthen robustness
under such attacks. Nevertheless, naïvely incor-
porating adversarial examples can destabilize op-
timization in few-shot settings and hurt clean ac-
curacy. To address this, we draw on contrastive
learning and develop a few-shot adversarial train-
ing framework that improves both robustness and
detection performance.
Our Work.In this paper, we propose an adversar-
ial training framework for training a robust MGT
detector against attacks in few-shot settings, named
RAG-GuidEdAttacker StrengthensConTrastive
Few-shot Detector (REACT). REACTcouples a
detector that discriminates human-written texts
(HWTs) from MGTs with an attacker that generates
humanized adversarial examples to evade detection.
To strengthen the attacker, we adopt a retrieval-
augmented generation (RAG) (Lewis et al., 2020)
scheme: a retrieval pool built from training sam-
ples and previously generated adversaries provides
a positive/negative pair as in-context demonstra-
tions, encouraging more human-like perturbations.
Under the black-box constraint where only detec-
toroutputsare available, we use an open-source
surrogate model to approximate the target detector
and to derive retrieval labels. To stabilize and ac-
celerate few-shot learning, we further introduce a
contrastive objective for the detector. At each train-
ing step, we alternately update the detector and the
attacker’s surrogate model, enabling the attacker togenerate increasingly humanized adversarial exam-
ples over time. Experimental results on 4 datasets
with 3 random seeds and 4 shot settings demon-
strate that REACTachieves an average detection F1
score of93.72, outperforming SOTA detectors by
4.95points. Moreover, REACTattains an average
attack success rate (ASR) of10.04%under 4 at-
tacks, reducing ASR by3.66%compared to SOTA
baselines. Our contributions are as follows:
•Effective Adversarial Training Framework.
We propose an adversarial training framework,
REACT, to improve the robustness of MGT de-
tectors against attacks in few-shot settings. In
this framework, the attacker produces highly
humanized adversarial examples, while the de-
tector employs contrastive learning over origi-
nal and adversarial samples to accurately dis-
tinguish HWTs from MGTs and remain robust
under attack.
•Adversarial Example Generation.We pro-
pose a novel adversarial example generation
strategy in the black-box setting. We em-
ploy RAG to produce highly human-like texts,
and use a surrogate model to obtain stronger
positive–negative exemplar pairs for retrieval,
further improving the quality of RAG-guided
humanization.
•Outstanding Performance.Experimental re-
sults across 4 datasets, 3 random seeds, and
4 shot budgets demonstrate that our method
surpasses 8 SOTA detectors in detection accu-
racy and exhibits stronger robustness against
4 powerful attacks.
2

2 Related Work
Machine-Generated Text (MGT) Detectors.
With the rapid development of LLMs, numerous
detectors have been proposed to identify machine-
generated text (MGT). Existing approaches broadly
fall into two categories:metric-basedandmodel-
baseddetectors. Metric-based detectors (Su et al.,
2023; Bao et al., 2024; Hans et al., 2024; Xu et al.,
2024; Zhu et al., 2025) require little or no super-
vised training and classify text using statistical sig-
nals of MGT, but they can be fragile to adversar-
ial perturbations and often produce uncalibrated
scores, complicating threshold selection in deploy-
ment. In contrast, model-based detectors (Liu et al.,
2022; Hu et al., 2023; Liu et al., 2024; Koike et al.,
2024; Li et al., 2025) fine-tune a text classifier
(e.g., BERT/RoBERTa) and are generally stronger,
yet they usually require substantial labeled data
to achieve high accuracy. Few-shot-tailored de-
tectors (Liu et al., 2022, 2024) mitigate the data
requirement but remain vulnerable to attacks, while
adversarially trained detectors (Hu et al., 2023; Li
et al., 2025) may suffer optimization instability
and reduced clean accuracy in few-shot settings.
To address these limitations, our work focuses on
improving both detection accuracy and robustness
against attacks under limited supervision.
Adversarial Training.Adversarial training opti-
mizes a detector to preserve correct predictions on
adversarial examples that are deliberately crafted
to induce misclassification. However, recent stud-
ies (Thorat and Caines, 2025; Umrajkar, 2025) sug-
gest that existing adversarial training methods do
not transfer well to few-shot detection: directly in-
corporating adversarial examples can exacerbate
underfitting and degrade accuracy when supervi-
sion is scarce. Moreover, adversarial training for
MGT detectors (Hu et al., 2023; Koike et al., 2024;
Li et al., 2025) remains ineffective against highly
humanized, strong attacks in few-shot settings. To
address these limitations, our method leverages
RAG (Lewis et al., 2020) to generate highly human-
like adversarial examples and introduces a con-
trastive objective to stabilize few-shot representa-
tion learning, improving both accuracy and robust-
ness.
3 Threat Model
We follow the standard threat modeling framework
outlined in prior work (Biggio and Roli, 2018; Hu
et al., 2023; Koike et al., 2024; Li et al., 2025)and describe the goals and capabilities of both the
attacker and the detector in our adversarial-training
setup.
Attacker’s Goals and Capabilities.Given an
MGT passage, the attacker aims tohumanizeit to
deceive the detector and outputs a rewrite. We call
rewritesfrom MGT inputsasadversarial examples,
regardless of success. During training only, we also
generate auxiliary rewrites for HWTs as data aug-
mentation and treat them asMGT-labeledsamples
for detector optimization. We do not regard these
HWT-based rewrites as attacks, and they never de-
fine the attacker’s objective. To reflect real-world
deployment where the detector is accessed only via
an API that returnstop-1 predicted labels, we con-
sider a black-box setting in which the attacker has
no access to the detector’s weights, architecture, or
any proprietary training data beyond the provided
few-shot set, and can observe only the output labels
via queries. The attacker is allowed to leverage (i)
any publicly available LMs/LLMs, including both
open-source models and closed-source/commercial
models accessed via APIs, and (ii) previously gen-
erated rewrites together with detector-returned la-
bels to synthesize stronger adversarial examples.
The attacker makesonedetector query per gener-
ated rewrite (including auxiliary rewrites during
training).
Detector’s Goals and Capabilities.The detector
aims to correctly distinguish HWTs from MGTs
while remaining robust to adversarial examples.
During training, the detector has access only to
a few-shot labeled training set (containing both
HWTs and MGTs), the adversarial examples pro-
duced from MGT inputs, and the auxiliary rewrites
generated from HWT inputs for data augmentation.
We further assume that the detector has nodirect
accessto LLMs or large-scale external corpora be-
yond the provided training data. We emphasize that
any LLM usage is confined to the attacker module
for generating rewrites, while the deployed detector
itself never queries LLMs and only optimizes on
the resulting texts.
4 Methodology
In this section, we introduce our REACTframework.
The architecture of REACTis shown in Figure 2.
Our REACTcontains an attacker and a detector.
In the following subsections, we first describe the
workflows of our attacker and detector, and then
introduce the adversarial training procedure.
3

Input Passage X
To
Adversarial 
Example �𝑿𝑿
Input 
Passage X
Input Passage X
\
Or
Genshin Impact is an open -
world action RPG featuring 
elemental combat, gacha  
characters, and an expansive fantasy world called Teyvat .
Human Written 
Text
Machine -
Generated Text?Attacker for Generating Adversarial Examples
Retrieval Pool ℬ 
Index Text
0
321
…33
…Raw MGT 1
Raw HWT 1 
Adversarial 
Example 1
Adversarial Example 21
0.8542 0.8321
0.6714
0.3658
0.2373
0.11720.0488
0 2 5 24 33 32 6 1
IndexRetrieval 
Score𝑿𝑿𝓜𝓜
𝑿𝑿𝑯𝑯Surrogate 
Detector  𝓓𝓓𝒔𝒔𝒔𝒔𝒔𝒔 Score
Perform like target 
detector and obtain 
retrieval score of ℬPrompt Template 𝒁𝒁
You are a de -AI-style rewriter. Your 
task is to rewrite a given text…Rewrite TARGET_TEXT entirely in 
the style of 
𝑿𝑿𝑯𝑯, while avoiding the 
style of 𝑿𝑿𝓜𝓜. 
Now here are the references and 
the target: 
HUMAN_LIKE_REFERENCE: 𝑿𝑿𝑯𝑯 
AI_LIKE_REFERENCE: 𝑿𝑿𝓜𝓜   
TARGET_TEXT: Input Passage X
 Generator  𝒢𝒢
Generate adversarial
example using  Z(X; 𝑋𝑋𝐻𝐻, 𝑋𝑋𝑀𝑀)
Adversarial Training
You know, there's 
this game I keep 
coming back to---Genshin Impact. It 
drops you into this 
breathtaking, wide-
open fantasy realm 
called Teyvat , where 
you're free to wander and 
discover. The real 
magic, for me, is in 
the combat…Adversarial Example
�𝑿𝑿 Update X
Surrogate Detector  𝓓𝓓𝒔𝒔𝒔𝒔𝒔𝒔(·)
Bridge the gap between 𝓓𝓓 𝒔𝒔𝒔𝒔𝒔𝒔and 𝓓𝓓𝒕𝒕𝒕𝒕𝒔𝒔 with 𝓛𝓛 𝒕𝒕𝒕𝒕𝒕𝒕
0 2 5 24 33 32 6 1
IndexTrue Score Predicted Score
Target Detector 𝓓𝓓𝒕𝒕𝒕𝒕𝒔𝒔⋅
Distinguish HWTs and MGTs Optimized by  𝓛𝓛det
Adversarial Example �𝑿𝑿
Prediction Label ̂𝒍𝒍
Detector to Distinguish HWTs and MGTsPair
(𝑿𝑿𝓜𝓜, 𝑿𝑿𝑯𝑯)
𝓛𝓛det= 𝓛𝓛𝑨𝑨𝑨𝑨𝑨𝑨+𝝀𝝀𝑷𝑷𝑷𝑷𝑨𝑨 𝓛𝓛𝑷𝑷𝑷𝑷𝑨𝑨
Adversarial Classification Loss ( 𝓛𝓛𝑨𝑨𝑨𝑨𝑨𝑨) Pairwise Boundary Contrastive Loss ( 𝓛𝓛𝑷𝑷𝑷𝑷𝑨𝑨 )
Accurate on HWTs and MGTs while keep robust to the 
Adversarial Examples
Accelerate convergence in few-shot settings
Hidden State of [CLS]
Decision BoundaryPull PushDetector to
Distinguish
HWTs  and MGTs!\Genshin Impact is an open -
world action RPG featuring 
elemental combat, gacha  
characters, and an expansive fantasy world called Teyvat .
Attacker for
Generating
Adversarial Examples!
Data Mixed Data Mixed
Trainable Frozen
ℬ Figure 2:Pipeline of REACT.The attacker generates adversarial examples via RAG (§4.1), which are then fed into
the detector (§4.2) and used in the adversarial training procedure to update the retrieval pool. (§4.3).
4.1 Attacker for Generating Adversarial
Examples
Our attacker consists of a surrogate detector Dsur(·)
and a generator G(·). To achieve the attacker’s
goal in §3, we adopt a RAG-style humanization
procedure to generate adversarial candidates, and
update Dsurwith an attacker loss after each training
step to keep the retrieval scoring aligned with the
target detector.
Retrieval-Augmented Generation.Given an
input passage X, prior work often crafts an ad-
versarial example ˜X1via local perturbations such
as synonym substitution (Jin et al., 2020; Zhou
et al., 2024) or mask-based replacement (Li et al.,
2020), which may introduce unnatural phrasing
and reduce readability. Instead, we generate ˜Xby
conditioningGon retrieved exemplars.
Specifically, we maintain a text-only retrieval
pool (knowledge base) B={X i}|B|
i=1, initialized
with the few-shot training passages and continu-
ously expanded with attacker-generated rewrites
from previous steps. Let Psur 
l|X i
denote the
probability assigned to label lby the surrogate, for
retrieval, we score each XiwithDsurby its MGT
probability pi=P sur(l= MGT|X i), and use
it as a scalar retrieval score v(X i) =p ito rank
1We use ˜Xto denote arewritegenerated by the attacker for
an input passage X. When Xis an MGT, ˜Xis anadversarial
example; when Xis an HWT (during training only), ˜Xis
anauxiliary rewriteused for data augmentation and is not
considered an attack.candidates inB:
XH= arg min
Xi∈Bv(X i),
XM= arg max
Xi∈Bv(X i).(1)
Here, XHandXMcorrespond to the most human-
like and most machine-like instances under Dsur,
serving as style anchors for rewriting. This re-
trieved pair provides in-context exemplars for RAG:
XHsupplies human-writing cues to imitate, while
XMhighlights machine-like patterns to avoid. We
find that suchextremeanchors provide the clear-
est style contrast under the surrogate’s scoring and
we further study alternative prompting strategies
in §6.1. We then inject (XH, XM)into a prompt
templateZ(·)and feed it toGto produce:
˜X=G 
Z(X;XH, XM)
.(2)
For future retrieval, we insert only the rewritten
text into the pool, i.e., B ← B ∪ { ˜X}, yielding
a self-improving retrieval pool. We demonstrate
in Appendix B.4 via human evaluation that our
RAG strategy can effectively generate humanized
texts, and we further provide a case study in Ap-
pendix B.5.
Attacker’s Loss.To align Dsurwith the label-
only feedback from Dtarin the black-box set-
ting, for each generated rewrite ˜Xwe take ˆl=
arg maxD tar(˜X)as a pseudo label and minimize
the negative log-likelihood underD sur:
Latt=−logP sur ˆl|˜X
.(3)
4

In each training step, we update the surrogate
model using Eq. (3)while keeping the detector
frozen. Detailed procedures are described in §4.3.
4.2 Detector to Distinguish HWTs and MGTs
Our detector is a target classifier Dtar(·). Follow-
ing the detector goal in §3, we optimize Dtarwith
two complementary objectives: an adversarial clas-
sification loss that enforces label correctness under
humanization, and a pairwise boundary contrastive
loss that regularizes the representation geometry.
Adversarial Classification Loss.The Adversar-
ial Classification Loss (ACL) trains Dtarto cor-
rectly classify the original training sample while
remaining robust to the attacker-produced human-
ized rewrite. Given (X, l)∈D train and its attacker-
generated rewrite ˜X, we treat ˜Xas anMGT-labeled
instance for training (both for adversarial examples
from MGT inputs and auxiliary rewrites from HWT
inputs). Let Ptar(l|X) be the probability assigned
byD tarto labellfor inputX. We define
LACL=−logP tar(l|X)−αlogP tar(MGT| ˜X),(4)
where αdown-weights the adversarial term. We in-
troduce αbecause the attacker continuously gener-
ates additional MGT-labeled rewrites, which effec-
tively amplifies class imbalance and can destabilize
optimization and bias training toward adversarial
artifacts. By controlling the contribution of ˜X,α
stabilizes optimization and improves robustness
transfer to newly generated adversarial candidates.
Pairwise Boundary Contrastive Loss.To further
stabilize optimization and accelerate convergence
in few-shot settings, we propose a Pairwise Bound-
ary Contrastive loss (PBC) that directly constrains
the geometry between an input and its rewrite.
While prior work often employs InfoNCE (Liu
et al., 2024) or supervised contrastive learning (Liu
et al., 2022), these objectives typically rely on many
informative in-batch negatives or a memory bank to
construct reliable contrast sets. Such requirements
become brittle under few-shot data and small effec-
tive batch sizes, introducing additional components
whose gains are sensitive to design choices. We in-
stead use a pairwise formulation on a single (X,˜X)
pair, which is more sample-efficient and less de-
pendent on batch composition.
Lethand˜hbe the last-layer [CLS] represen-
tations of Xand ˜XfromDtar, and let c=
cos(norm(h),norm( ˜h))be their cosine similarity,
where norm(u) =u/∥u∥ 2denotes ℓ2normaliza-Algorithm 1Adversarial Training Procedure
1:Input:few-shot training set Dtrain={(X, l)} , surrogate
detector Dsur, target detector Dtar, generator G, prompt
template Z(·) , maximum epochs epoch max, hyperparam-
etersα,λ pbc,δsame,δdiff
2:Initialize:retrieval pool B ← {X|(X, l)∈D train},
epoch←0
*** adversarial training phase begins ***
3:whileepoch < epoch maxdo
4:foreach training pair(X, l)inD train do
*** Step A: RAG-based humanization. ***
5: Compute retrieval scores for all Xi∈ B using
Dsur
6: Retrieve style anchors(XH, XM)by Eq. (1)
7: Build the promptZ(X;XH, XM)
8: Generate the rewrite ˜Xby Eq. (2)
9: QueryD taronce, obtain ˆl←arg maxD tar(˜X)
*** Step B: UpdateD sur***
10: Calculate attacker lossL attusing Eq. (3)
11: UpdateD surwith SGD (D tarfrozen)
*** Step C: UpdateD tar***
12: Assign the training label for the rewrite ˜Xas˜l←
MGT.
13: CalculateL ACL using Eq. (4)
14: CalculateL PBC using Eq. (5)
15:L det← L ACL+λpbcLPBC (Eq. (6))
16: UpdateD tarwith SGD (D surfrozen)
*** Step D: UpdateB***
17: Update the poolB ← B ∪ { ˜X}
18:end for
19:epoch←epoch+ 1
20:end while
21:Output:trained detectorD tarand the updated poolB
tion. We define
LPBC=I[l= ˜l]·max 
0,(1−δ same)−c
+I[l̸= ˜l]·max 
0, c−δ diff
,(5)
where I[·]is the indicator function that equals 1if
the condition holds and 0otherwise, land˜ldenote
the labels of Xand˜X, and δsame andδdiffare the
margins for same-label and different-label pairs.
Minimizing Eq. (5)enforces c≥1−δ same for
same-label pairs and c≤δ difffor different-label
pairs, yielding more stable decision boundaries.
This low-variance, pairwise signal is particularly
effective for few-shot learning, improving conver-
gence without requiring large negative sets.
Total Loss.The overall detector objective is
Ldet=L ACL+λpbcLPBC,(6)
where λpbcweights the contrastive term. At each
training step, we update Dtarby minimizing Eq. (6)
with the attacker frozen, as detailed in §4.3.
4.3 Adversarial Training
We propose an adversarial training framework
where the attacker and the detector are updated
5

Dataset Shot MetricMetric-based Detectors Model-based Detectors
Binoculars Fast DetectGPT LRR RoBERTa-Base CoCo†PECOLA†RADAR* GREATER* REACT(Ours)*
DetectRL32Acc↑88.38 ±0.06 86.90 ±0.22 76.47 ±0.68 86.08 ±0.96 92.20 ±3.42 50.24 ±0.42 87.76 ±12.49 90.17 ±1.64 99.46±0.58
F1↑87.62 ±0.61 86.10 ±0.59 73.04 ±2.60 83.82 ±1.30 91.48 ±4.08 0.96±1.67 89.00 ±9.98 89.11 ±2.02 99.46±0.58
64Acc↑88.50 ±0.49 86.65 ±0.52 76.93 ±0.58 99.43 ±0.61 89.83 ±1.38 93.78 ±5.48 99.06 ±0.71 99.12 ±0.27 99.82±0.27
F1↑88.01 ±0.57 86.02 ±0.05 74.37 ±1.68 99.43 ±0.62 88.66 ±1.71 93.16 ±6.50 99.06 ±0.69 99.11 ±0.28 99.82±0.27
128Acc↑88.60 ±0.52 86.97 ±0.06 77.10 ±0.39 95.54 ±6.69 99.61 ±0.15 99.93±0.11 98.70 ±1.65 99.48 ±0.27 99.74 ±0.07
F1↑88.14 ±0.17 86.17 ±0.30 74.69 ±1.30 94.98 ±7.66 99.61 ±0.15 99.93±0.11 98.66 ±1.71 99.48 ±0.27 99.74 ±0.08
256Acc↑88.87 ±0.18 87.03 ±0.13 77.40 ±0.13 98.08 ±2.78 99.87 ±0.07 99.97±0.03 98.94 ±1.31 99.95 ±0.05 99.79 ±0.16
F1↑88.38 ±0.22 86.23 ±0.44 75.02 ±0.37 97.99 ±2.94 99.87 ±0.07 99.97±0.03 98.92 ±1.35 99.95 ±0.05 99.79 ±0.16
Avg.Acc↑88.59 ±0.34 86.84 ±0.23 76.97 ±0.44 94.78 ±2.76 95.38 ±1.26 85.98 ±1.51 96.14 ±4.04 97.15 ±0.56 99.70±0.27
F1↑88.04 ±0.39 86.13 ±0.32 74.28 ±1.46 94.05 ±3.13 94.91 ±1.52 73.50 ±2.00 96.41 ±3.43 96.91 ±0.68 99.70±0.27
OUTFOX32Acc↑86.62 ±0.51 80.45 ±0.28 78.52 ±0.29 88.15 ±2.78 53.17 ±2.69 56.07 ±10.47 77.34 ±16.01 68.85 ±9.14 95.64±1.56
F1↑85.96 ±0.74 78.43 ±0.28 77.14 ±1.12 88.48 ±2.00 68.12 ±1.24 23.16 ±39.94 82.51 ±11.60 76.20 ±5.09 95.77±1.42
64Acc↑86.67 ±0.45 80.38 ±0.19 78.42 ±0.33 90.84 ±1.74 93.31 ±1.23 67.69 ±15.73 93.62 ±3.18 88.10 ±4.06 97.12±0.05
F1↑86.07 ±0.79 78.19 ±0.10 77.16 ±1.46 90.54 ±2.29 93.40 ±0.90 67.88 ±8.74 93.33 ±3.81 88.04 ±3.66 97.16±0.02
128Acc↑86.70 ±0.61 80.20 ±0.05 78.53 ±0.25 87.65 ±9.64 95.80 ±1.94 92.76 ±5.57 95.30 ±2.26 95.57 ±2.65 96.73±0.48
F1↑86.06 ±0.85 78.18 ±0.09 77.02 ±0.85 88.01 ±8.84 95.64 ±2.12 93.29 ±4.62 95.39 ±2.10 95.38 ±2.93 96.75±0.56
256Acc↑86.82 ±0.13 80.27 ±0.03 78.35 ±0.30 97.56 ±0.82 97.93 ±0.37 98.14±0.55 97.14 ±0.92 98.01 ±0.73 98.03 ±0.52
F1↑86.21 ±0.15 78.22 ±0.10 76.69 ±1.03 97.55 ±0.81 97.93 ±0.38 98.15±0.55 97.10 ±0.98 98.02 ±0.73 98.04 ±0.51
Avg.Acc↑86.70 ±0.42 80.33 ±0.14 78.45 ±0.29 91.05 ±3.74 85.05 ±1.55 78.62 ±8.08 90.85 ±5.61 87.68 ±4.15 96.88±0.65
F1↑86.08 ±0.63 78.26 ±0.14 77.00 ±1.11 91.14 ±3.49 88.77 ±1.16 70.62 ±13.46 92.08 ±4.60 89.41 ±3.14 96.94±0.62
RAID32Acc↑84.07 ±0.16 86.40±0.48 74.19 ±0.61 72.09 ±4.59 53.40 ±2.92 50.73 ±1.06 56.85 ±10.44 53.91 ±5.76 79.30 ±0.90
F1↑82.31 ±0.20 86.25±0.32 70.36 ±1.91 65.97 ±13.25 66.99 ±1.25 10.31 ±16.39 65.78 ±2.02 65.94 ±1.39 80.18 ±0.86
64Acc↑83.37 ±1.37 86.42±0.38 74.46 ±0.42 69.12 ±6.59 78.24 ±4.63 50.00 ±0.00 69.97 ±17.30 77.21 ±12.42 84.55 ±3.78
F1↑81.80 ±0.75 86.23±0.25 71.27 ±1.47 61.36 ±17.70 78.22 ±5.30 44.44 ±38.49 74.05 ±6.48 80.12 ±6.46 83.63 ±4.11
128Acc↑83.56 ±0.91 86.45 ±0.35 74.80 ±0.47 86.56 ±1.86 80.16 ±2.40 54.35 ±7.44 59.99 ±17.31 83.72 ±1.80 87.42±0.86
F1↑82.07 ±0.52 86.28 ±0.22 72.34 ±1.29 85.33 ±3.05 78.05 ±5.53 15.66 ±26.78 71.63 ±8.60 81.78 ±3.86 87.86±0.45
256Acc↑83.53 ±0.49 86.38 ±0.12 74.71 ±0.34 90.59 ±2.03 90.58 ±1.14 76.48 ±11.54 77.15 ±23.51 90.41 ±3.19 92.43±1.52
F1↑82.08 ±0.30 86.24 ±0.09 72.04 ±1.32 90.65 ±1.58 90.79 ±0.89 70.21 ±24.17 82.60 ±13.81 90.07 ±3.66 92.39±1.41
Avg.Acc↑83.63 ±0.73 86.41±0.33 74.54 ±0.46 79.59 ±3.52 75.59 ±2.77 57.89 ±5.01 65.99 ±17.14 76.31 ±5.79 85.93 ±1.76
F1↑82.06 ±0.44 86.25±0.22 71.50 ±1.50 75.83 ±8.40 78.51 ±3.24 35.16 ±26.43 73.51 ±7.73 79.48 ±3.84 86.02 ±1.71
SemEval32Acc↑84.50 ±1.06 84.02 ±1.10 79.65 ±0.23 82.76 ±1.91 61.85 ±8.74 50.63 ±1.10 85.42 ±6.37 73.75 ±17.10 89.01±3.53
F1↑83.73 ±0.55 83.17 ±0.64 79.32 ±0.56 80.00 ±3.19 72.57 ±4.46 3.07±5.32 87.32 ±4.71 78.99 ±10.13 89.57±3.23
64Acc↑84.92 ±0.74 84.60 ±1.00 79.57 ±0.26 85.77 ±3.69 87.27 ±2.25 72.27 ±19.29 89.40 ±3.37 88.82 ±2.35 90.97±2.59
F1↑83.93 ±0.41 83.58 ±0.65 78.95 ±1.46 84.74 ±4.80 87.06 ±1.57 78.27 ±10.22 88.80 ±4.36 89.33 ±1.68 91.40±2.14
128Acc↑84.85 ±0.51 84.58 ±0.82 79.70 ±0.23 90.90 ±2.79 88.90 ±3.39 86.39 ±4.03 88.09 ±6.84 92.98±2.91 92.38 ±2.70
F1↑83.93 ±0.18 83.61 ±0.62 79.64 ±0.58 90.84 ±3.26 89.16 ±3.92 87.84 ±2.75 89.27 ±5.46 93.06±2.88 92.74 ±2.29
256Acc↑85.17 ±0.42 84.98 ±0.31 79.60 ±0.13 93.52 ±1.99 95.28 ±0.82 91.86 ±2.26 90.90 ±2.46 95.62±1.91 95.17 ±0.68
F1↑84.04 ±0.09 83.87 ±0.17 79.08 ±0.24 93.76 ±1.82 95.38 ±0.78 92.36 ±1.80 90.79 ±2.47 95.67±1.90 95.20 ±0.78
Avg.Acc↑84.86 ±0.68 84.55 ±0.81 79.63 ±0.21 88.24 ±2.59 83.38 ±3.80 75.29 ±6.67 88.45 ±4.76 87.79 ±6.07 91.88±2.37
F1↑83.91 ±0.31 83.56 ±0.52 79.25 ±0.71 87.33 ±3.27 86.04 ±2.68 65.39 ±5.02 89.04 ±4.25 89.26 ±4.15 92.23±2.11
OverallAvg.Acc↑85.94 ±0.54 84.54 ±0.38 77.40 ±0.35 88.41 ±3.15 84.84 ±2.35 74.46 ±5.32 85.35 ±7.88 87.23 ±4.14 93.60±1.26
F1↑85.02 ±0.44 83.55 ±0.31 75.51 ±1.20 87.09 ±4.57 87.06 ±2.15 61.17 ±11.75 87.76 ±5.01 88.77 ±2.94 93.72±1.18
Table 1:Detection performance across 4 datasets and shot sizes.We report the mean and standard deviation of
accuracy (Acc) and F1 score (F1) over 3 random seeds for each experiment (mean ±std).†denotes few-shot-tailored
detectors, and∗indicates adversarially trained detectors. The best result in each row isbolded, and the second-best
is underlined . We report TPR@FPR=1% in Appendix B.3.
alternatelywithin each training step, enabling a co-
evolution process. Unlike prior adversarial-training
approaches (Hu et al., 2023; Koike et al., 2024;
Li et al., 2025), our attacker adopts a dynamic
strategy that generates adversarial examples on the
fly. As training proceeds, the attacker leverages
increasingly diverse texts in the retrieval pool B
to produce more human-like adversarial rewrites,
allowing the detector to learn richer humanization
patterns. Meanwhile, the detector is optimized with
the ACL and PBC objectives to balance accuracy
and robustness in the few-shot settings.
Training Process.At each training step t, the
attacker first generates a rewrite ˜Xof the input
passage Xusing the RAG procedure in Eq. (1)
and Eq. (2). We then update the surrogate detec-
torDsurusing the attacker loss in Eq. (3), while
keeping Dtarfrozen. Next, we update the targetdetector Dtarby minimizing the detector objective
in Eq. (6), with Dsurfrozen. Finally, we add the
newly generated rewrite to the retrieval pool B, so
that subsequent steps can retrieve stronger anchors
and produce more challenging adversarial exam-
ples. We detail the adversarial training process in
form of pseudocode in Algorithm 1, and we further
analyze the training dynamics during adversarial
training in Appendix B.6, demonstrating the effec-
tiveness of our co-evolution mechanism.
5 Experiment Results
We conduct extensive experiments to comprehen-
sively evaluate the detection accuracy of REACT
in few-shot settings and its robustness under adver-
sarial attacks. Due to space constraints, we present
the ablation study results in the Appendix B.1.
6

Figure 3:Average (top) and maximum ASR (bottom) (%) across 4 datasets and 4 shot sizes under 4 attacks.
A lower ASR (%) indicates better model performance. We only include methods whose clean-test accuracy is above
75in Table 1, since ASR is not informative when a detector is close to random guessing.
5.1 Detection Performance in Few-shot
Settings
Experimental setup.We conduct a comprehen-
sive evaluation of REACTin few-shot settings over
four datasets (DetectRL (Wu et al., 2024), OUT-
FOX (Koike et al., 2024), RAID (Dugan et al.,
2024), and SemEval (Wang et al., 2024b)) and four
shot sizes, focusing on detection accuracy. Our
baselines include: (i) Metric-based detectors, in-
cluding Fast DetectGPT (Bao et al., 2024), Binoc-
ulars (Hans et al., 2024), and LRR (Su et al.,
2023); and (ii) Model-based detectors, including
RoBERTa-Base (Liu et al., 2019), CoCo (Liu et al.,
2022), PECOLA(Liu et al., 2024), RADAR (Hu
et al., 2023), and GREATER(Li et al., 2025).
For all model-based detectors, we use the same
RoBERTa-Base backbone to ensure a fair compari-
son. For our method, we mainly adopt Llama-3.2-
3B-Instruct (Dubey et al., 2024) as the generator
Gand ALBERT-Base-v2 (Lan et al., 2019) as the
surrogate detector Dsur, and the prompt template
Zcan be found in Figure 5. To improve the robust-
ness of our conclusions, we train each model with
training sets sampled using three random seeds, and
evaluate all models on the same test set. Detailed
descriptions of the baseline models and implemen-
tation details are provided in the Appendix A.4.
Experiment results.We present the few-shot
detection performance of REACTin Table 1 and
unveil the following three key insights:1) Bestoverall performance.On the overall average, RE-
ACTachieves93.60accuracy and93.72F1. It im-
proves over the best metric-based detector Binocu-
lars by+7.66accuracy and+8.70F1 score, and it
also surpasses the strongest model-based detector
by+5.19accuracy and+4.95F1 score.2) Low
sensitivity to random seeds.REACTshows con-
sistently small standard deviations under few-shot
sampling. For DetectRL with 32-shot, RADAR
fluctuates substantially at 87.76 ±12.49 accuracy,
while REACTremains stable at 99.46 ±0.58 accu-
racy. Similar patterns appear on OUTFOX with
32-shot, where RADAR reports 77.34 ±16.01 accu-
racy but REACTachieves 95.64 ±1.56 accuracy.3)
Stable scaling across shot sizes.REACTis al-
ready competitive at 32-shot and scales smoothly
with more supervision, especially on the chal-
lenging RAID benchmark. Its accuracy increases
from 79.30 ±0.90 at 32-shot to 92.43 ±1.52 at 256-
shot, a gain of+13.13points. At 64-shot on
RAID, REACTreaches 84.55 ±3.78 accuracy, im-
proving over the best adversarially trained detector
GREATERby+7.34points, and over the strongest
few-shot-tailored detector CoCo by+6.31points.
At 128-shot, REACTfurther surpasses GREATERby
+3.70points in accuracy, showing that it effectively
absorbs additional data rather than being trapped in
a weak few-shot solution. Our REACTalso exhibits
a certain degree of generalization ability to unseen
datasets, as detailed in Appendix B.2.
7

5.2 Robustness against Attacks
Experimental setup.We evaluate the robustness
of detectors against four strong attacks under few-
shot settings on the test sets of the above four
datasets. The evaluated attacks include Modify
Attack (Wang et al., 2024a), Paraphrasing (Kr-
ishna et al., 2023), Back-translation (Sennrich et al.,
2015), and Humanize Attack (Wang et al., 2024a).
Detailed descriptions of these attack methods are
provided in the Appendix A.5.
Experiment results.We report robustness against
four strong attacks in Figure 3 and summarize three
findings.1) Lowest average ASR in most settings.
REACTachieves the lowest average ASR in 14
out of 16dataset–shot settings. Averaged over all
datasets and shot sizes, it achieves10.04%average
ASR and reduces average ASR by3.66percentage
points compared to the best-performing baseline in
each dataset–shot setting, with larger gains at low
shots (e.g.,6.47percentage points on average at 32-
shot).2) Strongest worst-case robustness under
max ASR.REACTalso attains the lowest max ASR
in15out of 16settings. Across all datasets and
shot sizes, it lowers max ASR by8.20percentage
points on average, with especially large margins
on challenging cases such as OUTFOX at 256-shot
(18.77percentage points) and RAID at 128-shot
(23.81percentage points).3) Consistent gains
across shot sizes.The reductions remain positive
at every shot size. For average ASR, the mean
gap is6.47,1.46,4.52, and2.18percentage points
at32,64,128, and 256shots, respectively; the
same pattern holds for max ASR with9.56,5.80,
10.84, and6.60percentage points, indicating that
REACTprovides stable robustness improvements
from scarce supervision to higher-shot settings.
6 Discussion
6.1 Impact of Adversarial Example
Generation Strategy
In this section, we study how the adversarial exam-
ple generation strategy affects REACT’s accuracy
and robustness. As described in §4.1, our attacker
adopts RAG, while other schemes are alternative
prompting baselines. We compare RAG with two
prompt-based baselines:Direct Prompt, which uses
a fixed template to instruct the generator to pro-
duce human-like rewrites, andFew-shot Prompting,
which samples k∈ {1,2,3,4,5}(MGT,HWT)
pairs as in-context exemplars. We evaluate these
strategies on DetectRL and RAID, reporting clean
Figure 4:Impact of adversarial example generation
strategy in REACT.A higher accuracy (top) and lower
ASR (bottom) indicates higher model performance.
accuracy and ASR under four attacks across shot
sizes. Results shown in Figure 4 yield two findings.
First, RAG offers the best accuracy–robustness
trade-off with minimal context.With only one
retrieved pair, RAG achieves higher accuracy and
lower ASR than all prompt-based baselines, while
reducing token budget and training cost.Second,
more in-context shots can help but reduce sta-
bility.Although larger ksometimes lowers ASR,
prompt-based baselines become less consistent; for
example, 4-shot prompting reaches ∼20% ASR on
RAID at 128-shot but rises to nearly 70% at 64-
shot. In contrast, RAG remains consistently strong
across settings.
7 Conclusion
In this paper, we propose an adversarial training
framework namedRAG-GuidEdAttacker Strength-
ensConTrastive Few-shot Detector (REACT) to im-
prove the detection accuracy of MGT detectors in
few-shot settings as well as their robustness against
attacks. We design a novel RAG-based adversarial
example generation strategy for the attacker and
introduce a pairwise boundary contrastive loss for
the detector to enhance few-shot detection perfor-
mance. Extensive experiments on 4 datasets with
4 shot sizes and 3 random seeds demonstrate that
REACTconsistently outperforms 8 SOTA detectors
in both detection accuracy and robustness. Further
discussion of adversarial example generation strate-
gies show that RAG achieves better training effec-
tiveness while reducing training time compared
with alternative few-shot prompting methods.
8

Limitations
Despite the strong performance of REACT, it still
has several limitations.First, while REACTper-
forms well on English benchmarks, extending it
to multilingual settings remains challenging.Sec-
ond, although adversarial training yields clear im-
provements, it also increases training time and in-
curs additional computational and resource over-
head during training.Third, the detector in our
work operates at the document level and therefore
cannot localize machine-generated content within
a human–AI co-authored text. Developing fine-
grained attribution methods, such as sentence-level
or token-level detection, remains an important di-
rection for future work.
Ethics Statement
This work studies and improves the robustness
of machine-generated text (MGT) detection under
strong humanization-style attacks, with a particular
focus on few-shot settings. Our goal is defensive:
we design an adversarial training framework that
helps detectors learn more reliable, attack-resilient
decision boundaries, rather than enabling content
deception in real-world systems.
Dual-use considerations.Techniques that gener-
ate more human-like rewrites are inherently dual-
use: they could be misused to evade detectors, fa-
cilitate plagiarism, or amplify misinformation. We
therefore frame the attacker strictly as atraining-
timecomponent to harden detectors. Our experi-
ments and analysis are intended to expose vulner-
abilities and improve defensive robustness, not to
provide practical evasion guidance.
Prompt disclosure and risk mitigation.For
transparency and reproducibility, we include a
prompttemplatein the appendix. However, we take
care to avoid releasing a turnkey evasion recipe. In
particular, (i) we do not provide an automated end-
to-end evasion pipeline (e.g., query-efficient search
loops, attack orchestration code, or deployment-
ready scripts), (ii) we do not include systematically
optimized prompts, prompt ensembles, or model-
specific prompt tuning details that would materially
lower the barrier to misuse, and (iii) we emphasize
that effectiveness in our framework relies on the
fulladversarial-training setup (e.g., retrieval pool
maintenance and surrogate-aligned scoring under a
constrained black-box feedback signal), rather than
the prompt alone. We strongly discourage usingany components of this work to bypass safety or
provenance mechanisms.
Responsible release and intended use.Any ar-
tifacts we release are intended to supportdefen-
siveresearch and reproducibility. We encourage
practitioners to combine detection with appropriate
safeguards (e.g., human-in-the-loop review, trans-
parency mechanisms, and due-process policies)
and to consider rate limiting and abuse monitor-
ing when detectors are deployed as public APIs.
Data, privacy, and content safety.All experi-
ments are conducted on publicly available bench-
mark datasets. We do not collect new user data
and do not target private, personally identifiable, or
sensitive information. Generated rewrites are used
solely for evaluating robustness and training de-
fenses, and are not used for surveillance or content
censorship.
Human evaluation.When human evaluation is
performed (Appendix B.4), we follow standard an-
notation protocols to minimize risk to annotators,
avoid exposing private information, and restrict
the task to assessing writing quality and human-
likeness. Annotators are informed of the task pur-
pose and the data sources, and the study is con-
ducted in accordance with applicable institutional
and legal requirements.
Environmental and resource impact.Adversar-
ial training increases compute cost due to iterative
generation and alternating updates. We mitigate
this by using a controlled experimental setup and
reporting results under fixed budgets. Future work
should further reduce training overhead while pre-
serving robustness gains.
Overall, we present REACTas a defensive con-
tribution to strengthening MGT detection under
realistic attack pressures, and we discourage any
misuse of the methodology or disclosed materials
for deceptive purposes.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi
Yang, and Yue Zhang. 2024. Fast-detectgpt: Efficient
zero-shot detection of machine-generated text via
conditional probability curvature. InICLR.
9

Battista Biggio and Fabio Roli. 2018. Wild patterns:
Ten years after the rise of adversarial machine learn-
ing. InProceedings of the 2018 ACM SIGSAC Con-
ference on Computer and Communications Security,
pages 2154–2156.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv preprint arXiv:2407.21783.
Liam Dugan, Alyssa Hwang, Filip Trhlík, Andrew
Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ip-
polito, and Chris Callison-Burch. 2024. Raid: A
shared benchmark for robust evaluation of machine-
generated text detectors. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 12463–
12492.
Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang,
Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng
Wu. 2023. How close is chatgpt to human experts?
comparison corpus, evaluation, and detection.arXiv
preprint arXiv:2301.07597.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Abhimanyu Hans, Avi Schwarzschild, Valeriia
Cherepanova, Hamid Kazemi, Aniruddha Saha,
Micah Goldblum, Jonas Geiping, and Tom Goldstein.
2024. Spotting llms with binoculars: zero-shot
detection of machine-generated text. InProceedings
of the 41st International Conference on Machine
Learning, pages 17519–17537.
Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2023.
Radar: Robust ai-text detection via adversarial learn-
ing.Advances in neural information processing sys-
tems, 36:15077–15095.
Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter
Szolovits. 2020. Is bert really robust? a strong base-
line for natural language attack on text classification
and entailment. InProceedings of the AAAI con-
ference on artificial intelligence, volume 34, pages
8018–8025.
Ryuto Koike, Masahiro Kaneko, and Naoaki Okazaki.
2024. Outfox: Llm-generated essay detection
through in-context learning with adversarially gen-
erated examples. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 38, pages
21258–21266.
Kalpesh Krishna, Yixiao Song, Marzena Karpinska,
John Wieting, and Mohit Iyyer. 2023. Paraphras-
ing evades detectors of ai-generated text, but retrieval
is an effective defense.Advances in Neural Informa-
tion Processing Systems, 36:27469–27500.Zhenzhong Lan, Mingda Chen, Sebastian Goodman,
Kevin Gimpel, Piyush Sharma, and Radu Soricut.
2019. Albert: A lite bert for self-supervised learn-
ing of language representations.arXiv preprint
arXiv:1909.11942.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue,
and Xipeng Qiu. 2020. Bert-attack: Adversarial at-
tack against bert using bert. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6193–6202.
Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao
Shen, and Xiaoming Liu. 2025. Iron sharpens iron:
Defending against attacks in machine-generated text
detection with adversarial training.arXiv preprint
arXiv:2502.12734.
Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingx-
uan Wang, Bingzheng Xu, Bochao Wu, Bowei
Zhang, Chaofan Lin, Chen Dong, and 1 others. 2025.
Deepseek-v3. 2: Pushing the frontier of open large
language models.arXiv preprint arXiv:2512.02556.
Shengchao Liu, Xiaoming Liu, Yichen Wang, Zehua
Cheng, Chengzhengxu Li, Zhaohan Zhang, Yu Lan,
and Chao Shen. 2024. Does detectgpt fully utilize
perturbation? bridging selective perturbation to fine-
tuned contrastive learning detector would be better.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 1874–1889.
Xiaoming Liu, Zhaohan Zhang, Yichen Wang, Hang Pu,
Yu Lan, and Chao Shen. 2022. Coco: Coherence-
enhanced machine-generated text detection under
data limitation with contrastive learning.arXiv
preprint arXiv:2212.10341.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized BERT pretraining
approach.arXiv preprint arXiv:1907.11692.
Eric Mitchell, Yoonho Lee, Alexander Khazatsky,
Christopher D Manning, and Chelsea Finn. 2023. De-
tectgpt: Zero-shot machine-generated text detection
using probability curvature. InInternational Con-
ference on Machine Learning, pages 24950–24962.
PMLR.
Colin Raffel, Noam Shazeer, Adam Roberts, Kather-
ine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the
limits of transfer learning with a unified text-to-text
transformer.Journal of Machine Learning Research,
21(140):1–67.
10

Rico Sennrich, Barry Haddow, and Alexandra Birch.
2015. Improving neural machine translation
models with monolingual data.arXiv preprint
arXiv:1511.06709.
Jinyan Su, Terry Zhuo, Di Wang, and Preslav Nakov.
2023. Detectllm: Leveraging log rank information
for zero-shot detection of machine-generated text.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 12395–12412.
Shantanu Thorat and Andrew Caines. 2025. Dactyl:
Diverse adversarial corpus of texts yielded from large
language models.arXiv preprint arXiv:2508.00619.
Jörg Tiedemann and Santhosh Thottingal. 2020. OPUS-
MT — Building open translation services for the
World. InProceedings of the 22nd Annual Confer-
enec of the European Association for Machine Trans-
lation (EAMT), Lisbon, Portugal.
Ved Umrajkar. 2025. Dac-lora: Dynamic adversarial
curriculum for efficient and robust few-shot adapta-
tion.arXiv preprint arXiv:2509.20792.
Yichen Wang, Shangbin Feng, Abe Bohan Hou, Xiao
Pu, Chao Shen, Xiaoming Liu, Yulia Tsvetkov, and
Tianxing He. 2024a. Stumbling blocks: Stress testing
the robustness of machine-generated text detectors
under attacks.arXiv preprint arXiv:2402.11638.
Yuxia Wang, Jonibek Mansurov, Petar Ivanov, jinyan
su, Artem Shelmanov, Akim Tsvigun, Osama Mo-
hammed Afzal, Tarek Mahmoud, Giovanni Puccetti,
Thomas Arnold, Chenxi Whitehouse, Alham Fikri
Aji, Nizar Habash, Iryna Gurevych, and Preslav
Nakov. 2024b. Semeval-2024 task 8: Multidomain,
multimodel and multilingual machine-generated text
detection. InProceedings of the 18th International
Workshop on Semantic Evaluation (SemEval-2024),
pages 2041–2063, Mexico City, Mexico. Association
for Computational Linguistics.
Junchao Wu, Runzhe Zhan, Derek Wong, Shu Yang,
Xinyi Yang, Yulin Yuan, and Lidia Chao. 2024. De-
tectrl: Benchmarking llm-generated text detection in
real-world scenarios.Advances in Neural Informa-
tion Processing Systems, 37:100369–100401.
Yihuai Xu, Yongwei Wang, Yifei Bi, Huangsen
Cao, Zhouhan Lin, Yu Zhao, and Fei Wu. 2024.
Training-free llm-generated text detection by min-
ing token probability sequences.arXiv preprint
arXiv:2410.06072.
Ying Zhou, Ben He, and Le Sun. 2024. Humanizing
machine-generated content: Evading ai-text detec-
tion through adversarial attack. InProceedings of
the 2024 Joint International Conference on Compu-
tational Linguistics, Language Resources and Evalu-
ation (LREC-COLING 2024), pages 8427–8437.
Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan,
Shi Wang, and Yanan Cao. 2025. Dna-detectllm: Un-
veiling ai-generated text via a dna-inspired mutation-
repair paradigm.arXiv preprint arXiv:2509.15550.
11

A Experimental Setting
A.1 Dataset
We conduct experiments on four benchmark
datasets: DetectRL (Wu et al., 2024), OUT-
FOX (Koike et al., 2024), RAID (Dugan et al.,
2024), and SemEval (Wang et al., 2024b). For
each dataset, we construct few-shot training sets by
sampling {32,64,128,256} labeled instances us-
ing three random seeds {66,2025,9999} . For eval-
uation, we sample 2,000disjointinstances (1,000
HWT + 1,000 MGT) as a balanced test set from
the corresponding dataset split, ensuring no overlap
with the few-shot training set. We additionally use
HC3 (Guo et al., 2023) for OOD evaluation.
A.2 Implementation
REACTis deployed on a server equipped with two
RTX 4090 GPUs running Ubuntu 22.04. Our adver-
sarial training framework mainly adopts Llama-3.2-
3B-Instruct (Dubey et al., 2024) as the generator G
and ALBERT-Base-v2 (Lan et al., 2019) as the sur-
rogate detector Dsur. We use RoBERTa-Base (Liu
et al., 2019) as the backbone of the target detec-
torDtar. For metric-based detectors, we train a
logistic-regression calibrator on the few-shot train-
ing set to map raw scores to calibrated probabilities
(of the MGT class), and report Acc/F1 using a fixed
decision threshold of 0.5. For model-based detec-
tors, we employ the same RoBERTa-Base back-
bone to ensure a fair comparison, train each method
for 6 epochs under an identical protocol, and report
results using the final checkpoint with a fixed deci-
sion threshold of 0.5. Under this 6-epoch protocol,
REACTuses a two-stage schedule: we first pre-
train on clean samples using only the cross-entropy
loss for 3 epochs, and then perform adversarial
training for another 3 epochs. The selected hyper-
parameters of REACTare summarized in Table 2.
We use the following prompt template Z(·) in
Figure 5:
A.3 Evaluation Metrics
To evaluate detection performance, we reportac-
curacy(Acc) andF1-score(F1). To quantify ro-
bustness against adversarial attacks, we use the
attack success rate(ASR), where a lower ASR indi-
cates stronger robustness. Under our threat model,
the attacker perturbs only machine-generated texts
(MGTs). Accordingly, ASR is defined as the frac-
tion of originally correctly classified MGTs thatHyperparameter Value
Pre-training epochsepoch pre 3
Maximum adversarial epochsepoch max 3
Total epochs 6
Target detector learning rateη tar 2×10−5
Surrogate detector learning rateη sur 2×10−5
AdamW weight decayβ0.03
Batch sizeM2
Max sequence lengthL(forD tar,Dsur) 512
Adversarial CE scaling coefficientα0.5
PBC loss weightλ pbc 1.2
PBC margin for same-classδ same 0.1
PBC margin for diff-classδ diff 0.3
Generator max new tokensT new 512
Generator input max lengthL gen 4096
Sampling temperatureτ0.7
Nucleus samplingp0.9
Table 2: Hyperparameters for our REACT.
become misclassified after the attack:
ASR =#{MGTs misclassified after attack}
#{MGTs correctly classified before attack}.(7)
A.4 Experimental Comparison Methods
To better assess the effectiveness of REACT, we
compare it with representative state-of-the-art
(SOTA) MGT detectors, including three widely
usedmetric-baseddetectors and fivemodel-based
detectors. Metric-based detectors typically output
raw scoresrather than calibrated probabilities. Fol-
lowing the standard practice in their original pa-
pers, we perform score calibration on the corre-
sponding training set by fitting a logistic regres-
sion model that maps each detector’s raw score to
an output probability for binary classification. For
each metric-based detector, we follow the backbone
choices recommended by its original paper (and of-
ficial implementation) for the underlying language
model(s). In particular, for methods that require
an LM backbone, we use the paper-recommended
scoring/sampling model for Fast DetectGPT, the
recommended performer/observer pair for Binocu-
lars, and the recommended scoring model for LRR.
•Fast DetectGPT(Bao et al., 2024): A metric-
based zero-shot detector derived from Detect-
GPT, designed to reduce the computational
overhead of perturbation-based scoring. In-
stead of repeatedly perturbing the input, it
measuresconditional probability curvature,
which captures discrepancies in token-level
likelihood geometry between human-written
and machine-generated text. This single-pass
12

Prompt templateZ(X;XH, XM)
Human-style reference:XH
Machine-style reference:XM
Target passage:X
Instruction:
You are a de-AI-style rewriter. Your task is to rewrite a given text so that it avoids being detected as AI-generated.
Your objectives:
- Preserve the original meaning, facts, and intent of the text.
- Make the rewritten text look as if it were written by a human.
- Strictly write in the SAME LANGUAGE as the TARGET_TEXT.
- Your maximum generation length is {max_tokens} tokens.
You will be given two reference texts:
1. HUMAN_LIKE_REFERENCE: a text (or several examples) that the detector considers the most human-like.
2. AI_LIKE_REFERENCE: a text (or several examples) that the detector considers the most AI-like.
Your style constraints:
- Imitate the style, tone, and rhythm of HUMAN_LIKE_REFERENCE.
- Explicitly avoid any stylistic patterns that resemble AI_LIKE_REFERENCE.
- Do NOT mention detectors, AI, models, prompts, or the rewriting process.
- Do NOT add explanations, comments, or meta-text. Output ONLY the rewritten text.
Steps you MUST follow internally (do NOT output the steps):
1. Carefully read HUMAN_LIKE_REFERENCE and understand its style and tone.
2. Briefly compare it in your mind with AI_LIKE_REFERENCE and identify stylistic differences.
3. Rewrite TARGET_TEXT entirely in the style of HUMAN_LIKE_REFERENCE, while avoiding the style of AI-
LIKE_REFERENCE.
Now here are the references and the target:
HUMAN_LIKE_REFERENCE:
XH
AI_LIKE_REFERENCE:
XM
TARGET_TEXT:
X
Rewritten TARGET_TEXT (remember: output ONLY the rewritten text):
Figure 5: Prompt template used to constructZ(X;XH, XM)for RAG-based rewriting.
scoring strategy substantially accelerates infer-
ence while maintaining competitive detection
quality.
•Binoculars(Hans et al., 2024): A metric-
based zero-shot detector that detects machine-
generated text by contrasting the output dis-
tributions of two closely related pretrained
language models. It computes a lightweight
divergence-style score that reflects subtle in-
consistencies between human and machine
writing, requiring neither fine-tuning nor la-
beled data.
•LRR(Su et al., 2023): A metric-based zero-
shot detector that uses log-rank statistics under
a reference language model. It identifies atyp-
ical token-rank patterns that commonly arise
in LLM generations, offering a fast alternative
to perturbation-heavy methods while retaining
strong empirical performance.
•RoBERTa-Base(Liu et al., 2019): A stan-
dard supervised detector that fine-tunes an
RoBERTa-Base classifier on the labeled train-ing set and directly outputs class probabilities
for binary MGT detection.
•CoCo(Liu et al., 2022): A few-shot-tailored
detector that introduces contrastive training
driven bylocal coherencesignals. It first
scores each input using local coherence com-
puted by a set of in-domain language mod-
els, and then trains the detector with a con-
trastive objective to better separate human-
written texts and machine-generated texts in
representation space. Since CoCo is primarily
designed for sample-efficient discrimination
rather than adversarial robustness, it can be
less robust under strong attacks in our robust-
ness evaluation.
•PECOLA(Liu et al., 2024): A contrastive-
learning-based detector that improves fine-
tuned MGT detectors viaselective pertur-
bationand a contrastive objective to learn
more discriminative representations. We treat
PECOLAas a few-shot-tailored baseline due
to its sample-efficient training design, but it
is not explicitly adversarially trained, which
13

can lead to weaker robustness under our attack
suite.
•RADAR(Hu et al., 2023): An adversari-
ally trained detector that improves robustness
against paraphrasing-style attacks by training
the detector jointly with a paraphraser. The
paraphraser generates low-perplexity para-
phrases to evade detection, while the detec-
tor learns to correctly classify both original
texts and adversarial paraphrases. RADAR fo-
cuses on monolingual paraphrase-based adver-
saries and is mainly proposed as a robustness-
oriented training recipe; thus, it is not specifi-
cally tailored to few-shot training settings.
•GREATER(Li et al., 2025): An adversari-
ally trained detector within the GREATER
framework, where an adversary identifies and
perturbs critical tokens and searches for effec-
tive adversarial examples. GREATERupdates
the adversary and the detector synchronously
so that the detector learns to defend against
evolving attacks and generalize to other per-
turbations. As a robustness-first adversarial
training framework, GREATERis not origi-
nally designed for few-shot settings and typ-
ically incurs higher training overhead due to
the coupled attacker–defender optimization.
A.5 Detailed Information of Attack Method
Here we provide detailed descriptions of the 4
strong attack methods used in the main paper.
•Modify Attack(Wang et al., 2024a): A mask-
filling style humanization attack implemented
withT5-large(Raffel et al., 2020). Given
an input MGT, we mask a subset of tokens
and let T5-large fill in the blanks, producing a
rewritten text that largely preserves the orig-
inal semantics while injecting more human-
like lexical and syntactic variations. This at-
tack targets detectors that rely on surface cues
or token-level likelihood patterns by introduc-
ing semantics-preserving perturbations.
•Paraphrasing(Krishna et al., 2023): A
sentence-level paraphrase attack implemented
withDIPPER(Krishna et al., 2023). For each
MGT input, DIPPER rewrites the passage to
preserve meaning while substantially altering
phrasing and local syntax, yielding human-
like paraphrases. This attack aims to evade de-
tection via semantic camouflage by replacingcharacteristic generation artifacts with more
natural expressions.
•Back-translation(Sennrich et al., 2015): A
sentence-level paraphrase attack that round-
trip translates each MGT through a pivot lan-
guage to induce lexical and syntactic diver-
sity while preserving meaning. We use pub-
licHelsinki-NLP OPUS-MTmodels (Tiede-
mann and Thottingal, 2020) and chooseGer-
manas the pivot language. Concretely, we
translate the input text into German and then
back to the original language, and replace the
original text with the back-translated output.
•Humanize Attack(Wang et al., 2024a): An
in-context learning based humanization attack
that usesthree (MGT,HWT) pairs as in-
context exemplars to guide rewriting. Given
an MGT input, the attacker prompts the gener-
ator with these paired exemplars to synthesize
a highly human-like rewrite while keeping
the underlying meaning. We useDeepSeek
V3.2(Liu et al., 2025) as the generator for this
attack.
B Additional Result
B.1 Ablation Study
Experiment Setup.To verify the role of each com-
ponent in REACT, we conduct an ablation study
that strictly follows the same experimental setting
as Table 1 so that the conclusions are directly com-
parable. Under this controlled setup, we compare
four variants in Table 3: a standard supervised
RoBERTa-Basedetector, a variant that removes
the pairwise boundary-constraint loss LPBC from
Eq.(6)(w/oLPBC), a variant that disables RAG
and instead uses direct prompting to generate ad-
versarial rewrites (w/o RAG), and the fullREACT.
We do not include the ablation that removes
LACL, because LACL is the classification loss;
without it, the detector cannot be trained to per-
form the classification task.
Experiment results.We report the ablation results
in Table 3 and summarize three findings. Following
Table 1, we report ASR only for detectors whose
clean-test accuracy exceeds 75%, since ASR is not
informative when a detector is close to random
guessing (hence “–” for RoBERTa-Base on RAID
at 32/64 shots).
1) RAG is a major driver of robustness by
strengthening humanization-oriented adversar-
14

Dataset Shot Metric RoBERTa-Base w/oL PBC w/o RAG REACT(Ours)
DetectRL32Acc↑86.08 ±0.96 96.89 ±4.88 96.29 ±2.09 99.46 ±0.58
Avg ASR↓10.84 1.36 2.040.74
64Acc↑99.43 ±0.61 99.77 ±0.06 99.72 ±0.10 99.82 ±0.27
Avg ASR↓1.16 0.60 0.810.48
128Acc↑95.54 ±6.69 99.72 ±0.07 99.71 ±0.05 99.74 ±0.07
Avg ASR↓1.87 0.46 0.44 0.28
256Acc↑98.08 ±2.78 99.79 ±0.16 99.77 ±0.06 99.79 ±0.16
Avg ASR↓1.82 0.43 0.38 0.26
Avg.Acc↑94.78 ±2.76 99.04 ±1.29 98.87 ±0.58 99.70 ±0.27
Avg ASR↓3.92 0.71 0.920.44
OUTFOX32Acc↑88.15 ±2.78 91.72 ±1.76 92.20 ±3.17 95.64 ±1.56
Avg ASR↓22.20 13.79 15.662.36
64Acc↑90.84 ±1.74 93.12 ±6.09 94.19 ±3.64 97.12 ±0.05
Avg ASR↓21.39 13.09 14.593.53
128Acc↑87.65 ±9.64 95.41 ±0.35 96.18 ±2.65 96.73 ±0.48
Avg ASR↓18.14 14.82 15.5210.26
256Acc↑97.56 ±0.82 97.03 ±0.44 97.40 ±0.34 98.03 ±0.52
Avg ASR↓16.43 4.92 4.87 4.27
Avg.Acc↑91.05 ±3.74 94.32 ±2.16 94.99 ±2.45 96.88 ±0.65
Avg ASR↓19.54 11.66 12.665.11
RAID32Acc↑72.09 ±4.59 76.35 ±7.45 75.08 ±0.96 79.30 ±0.90
Avg ASR↓– 72.72 65.14 25.21
64Acc↑69.12 ±6.59 77.43 ±1.95 80.78 ±1.27 84.55 ±3.78
Avg ASR↓– 83.90 74.26 29.28
128Acc↑86.56 ±1.86 86.87 ±3.30 86.91 ±3.19 87.42 ±0.86
Avg ASR↓45.41 42.45 61.7816.97
256Acc↑90.59 ±2.03 91.01 ±3.83 92.29 ±0.98 92.43 ±1.52
Avg ASR↓28.51 26.49 61.6520.07
Avg.Acc↑79.59 ±3.77 82.92 ±4.13 83.77 ±1.60 85.93 ±1.77
Avg ASR↓36.96 56.39 65.7122.89
SemEval32Acc↑82.76 ±1.91 85.60 ±5.29 79.13 ±7.85 89.01 ±3.53
Avg ASR↓45.09 17.08 16.65 12.77
64Acc↑85.77 ±3.69 87.65 ±2.28 89.37 ±6.35 90.97 ±2.59
Avg ASR↓22.38 13.98 14.9811.60
128Acc↑90.90 ±2.79 91.62 ±4.56 89.29 ±4.62 92.38 ±2.70
Avg ASR↓17.53 12.43 13.218.19
256Acc↑93.52 ±1.99 93.95 ±1.03 94.43 ±3.66 95.17 ±0.68
Avg ASR↓15.10 14.86 14.49 14.30
Avg.Acc↑88.24 ±2.59 89.70 ±3.29 88.06 ±5.62 91.88 ±2.37
Avg ASR↓25.03 14.59 14.8311.72
OverallAvg.Acc↑88.41 ±3.22 91.50 ±2.72 91.42 ±2.56 93.60 ±1.26
Avg ASR↓21.36 20.84 23.5310.04
Table 3:Result of Ablation Study.Accuracy (Acc, %)
is reported as mean ±std over three random seeds. Avg
ASR (%) denotes the average attack success rate over
four attacks. The best result in each row isbolded, and
the second-best is underlined .
ial rewrites.Disabling retrieval guidance (w/o
RAG) substantially weakens adversarial training,
increasing the overall Avg ASR from10.04to
23.53and reducing accuracy from93.60to91.42.
The effect is most pronounced on RAID, where
Avg ASR rises from22.89to65.71; at 128-shot it
jumps from16.97to61.78, highlighting the impor-
tance of strong adversarial rewrites for robustness.
2)LPBC improves the accuracy–robustness
trade-off and stabilizes few-shot training.Re-
moving the pairwise boundary constraint (w/o
LPBC) degrades performance on average, reducing
the overall accuracy from93.60to91.50and in-
creasing the overall Avg ASR from10.04to20.84.
On RAID, Avg ASR further deteriorates to56.39,
suggesting that PBC provides an additional low-
variance representation-level signal that helps the
detector learn robust boundaries under few-shot
sampling.
3) RAG and PBC are complementary for
the best accuracy–robustness trade-off.RAG
strengthens the attacker to provide more chal-
lenging humanization-oriented supervision, whileLPBC helps the detector convert such supervision
into stable representation learning; combining them
yields the best joint results. For example, on OUT-
FOX at 32-shot, REACTachieves95.64accuracy
and2.36Avg ASR, whereasw/o RAGreaches
92.20accuracy and15.66Avg ASR, andw/o LPBC
yields91.72accuracy and13.79Avg ASR.
B.2 Out-Of-Distribution (OOD) Experiment
To assess the out-of-distribution (OOD) generaliza-
tion of REACT, we train detectors on one dataset
and evaluate them on a disjoint, unseen dataset.
Specifically, we train on OUTFOX and test on HC3,
which is collected from different sources and con-
tains no overlapping instances with the OUTFOX
corpus, thereby probing cross-domain transfer un-
der a realistic distribution shift. For a fair compar-
ison under OOD shift, we focus on model-based
detectors, which can be trained under the same few-
shot protocol. HC3 evaluation uses the same label
mapping and preprocessing as in-domain experi-
ments.
Table 4 reports the OOD results and leads to
two observations.1) REACTachieves strong and
stable OOD transfer across all shot sizes.RE-
ACTconsistently attains86–87accuracy and87–88
F1 across all few-shot settings, with an overall aver-
age of86.60accuracy and87.55F1. Notably, it re-
mains strong even at 32-shot, reaching 87.03±3.27
accuracy and 88.05±2.48 F1, suggesting that RE-
ACTdoes not rely on extensive in-domain super-
vision to transfer to an unseen distribution.2)
Baselines degrade substantially under domain
shift, and some exhibit degenerate behaviors.
In contrast, most baselines show near-chance ac-
curacy and/or highly unstable F1 on HC3 when
trained on OUTFOX. For example, RoBERTa-
Base averages 52.37 ±5.56 accuracy with a large-
variance 18.37 ±19.29 F1, indicating unreliable de-
cision boundaries under domain shift. CoCo and
PECOLAalso exhibit limited transferability (Avg.
accuracy ≈52 ) with substantial variance across
shot sizes. Moreover, RADAR degenerates to a
nearly constant behavior ( 35.40 ±0.00 accuracy and
4.34±0.00 F1), suggesting that the learned decision
rule fails to adapt under OOD shift.
We hypothesize that the strong OOD robustness
of REACTis enabled by its adversarial training
mechanism. By introducing diverse, humanization-
oriented adversarial rewrites during training and
regularizing representations with PBC, REACTis
encouraged to rely less on dataset-specific artifacts
15

Dataset Shot Metric RoBERTa-Base CoCo†PECOLA†RADAR* GREATER* REACT(Ours)*
OUTFOX32Acc↑50.90 ±3.55 50.07 ±0.06 49.58 ±0.73 35.40 ±0.00 49.32 ±8.88 87.03±3.27
F1↑11.74 ±16.36 66.70 ±0.03 1.34±2.32 4.34±0.00 45.54 ±39.45 88.05±2.48
64Acc↑49.79 ±1.42 44.58 ±13.41 51.01 ±1.59 35.40 ±0.00 45.59 ±5.92 86.82±2.21
F1↑8.92 ±8.98 43.99 ±22.72 22.97 ±38.95 4.34±0.00 17.27 ±28.59 87.79±2.08
128Acc↑60.40 ±15.12 54.41 ±7.82 57.81 ±7.97 35.40 ±0.00 55.35 ±9.28 85.45±5.13
F1↑40.25 ±40.83 25.16 ±22.56 49.84 ±26.10 4.34±0.00 17.12 ±28.81 86.84±4.00
256Acc↑48.40 ±2.16 60.97 ±5.32 50.46 ±5.04 35.40 ±0.00 56.25 ±1.82 87.09±1.96
F1↑12.58 ±11.00 41.00 ±16.65 51.96 ±5.54 4.34±0.00 34.63 ±9.62 87.53±0.87
Avg.Acc↑52.37 ±5.56 52.51 ±6.65 52.21 ±3.83 35.40 ±0.00 51.63 ±6.47 86.60±3.14
F1↑18.37 ±19.29 44.21 ±15.49 31.53 ±18.23 4.34±0.00 28.64 ±26.62 87.55±2.36
Table 4:OOD detection performance on HC3 dataset.We report the mean and standard deviation of accuracy
(Acc) and F1 score (F1) over 3 random seeds (mean ±std).†denotes few-shot-tailored detectors, and∗indicates
adversarially trained detectors. The best result in each row isbolded.
and to learn more stable cues that transfer better
across domains. This interpretation is consistent
with our ablation results and training dynamics
analysis in Appendix B.1 and Appendix B.6.
B.3 TPR@FPR=1% in Main Result
While accuracy and F1 reflect overall classification
quality and are already reported in the main results,
evaluatingTPR at a low false-positive rateis es-
sential for security-sensitive deployments where
even a small FPR can be costly (e.g., moderation
triage, plagiarism screening, or forensic auditing).
Moreover, when reporting Acc/F1, detectors often
rely on tuned decision thresholds and may apply
post-hoc calibration on a validation split to select
an operating point. In contrast, TPR@FPR=1% is
an operating-point metric derived from the score
ranking: we sweep thresholds on thetest setand
report the TPR at the threshold that attains 1% FPR.
This evaluation is largely invariant tomonotonic
score calibrations that preserve ranking. We re-
port TPR@FPR=1% in Table 5 and highlight three
insights:1) Best overall low-FPR performance.
On the overall average, REACTachieves the high-
est80.81TPR@FPR=1%. It improves over the
strongest metric-based detector Fast DetectGPT
(68.07) by+12.74points, and also surpasses the
best competing model-based detector GREATER
(76.29) by+4.52points. Across the 16 dataset–
shot settings, REACTobtains the best mean perfor-
mance in8/16settings and ranks within the top-2
in12/16, demonstrating robust low-FPR general-
ization across diverse benchmarks.2) High re-
call under strict false-positive constraints.RE-
ACTmaintains high recall at stringent operating
points on multiple datasets. On DetectRL, it is
near-saturated across all shots (Avg.99.97, reach-ing100.00at 64/256-shot), indicating extremely
high sensitivity even when FPR is capped at 1%.
On OUTFOX, REACTalready achieves91.34at
64-shot and scales to95.77at 256-shot, showing
that the proposed training recipe remains effective
in a practically relevant low-FPR setting. Notably,
on the challenging RAID benchmark, REACTex-
hibits a strong scaling trend from 33.07 ±27.01 (32-
shot) to 83.33 ±3.58 (256-shot), a gain of+50.26
points, and becomes the best-performing method
at 128/256-shot, suggesting that REACTcan satisfy
strict low-FPR constraints once modest supervision
is available.3) Metric-based detectors can lead
in a few low-shot corners, but REACTremains
the strongest model-based option.We observe
that some metric-based detectors are competitive or
even superior in a small number of settings, most
notably on RAID at 32/64-shot where Fast Detect-
GPT attains69.96while model-based detectors are
generally weaker. A similar pattern appears on Se-
mEval at 32-shot where Fast DetectGPT achieves
66.37, slightly higher than REACT(63.35). Despite
these few-shot corner cases, REACTis consistently
the strongest (or among the strongest)model-based
detector under low-FPR evaluation: for instance, its
RAID Avg. (63.14) outperforms GREATER(56.47)
by+6.67points.
B.4 Human Evaluation of Adversarial
Example Generation Strategy
Experiment Settings.We conduct a human study
to assess whether our RAG-based generation can
truly produce more human-like adversarial texts.
We randomly sample 100 human-written texts
(HWTs) and generate corresponding texts with
four strategies: (1)Direct Generate, where we
directly use an LLM to generate the rewrite version
16

Dataset Shot MetricMetric-based Detectors Model-based Detectors
Fast DetectGPT Binoculars LRR RoBERTa-Base PECOLA†CoCo†RADAR* GREATER* REACT(Ours)*
DetectRL32 TPR@FPR=1%↑73.65 ±0.00 75.45 ±0.00 40.82 ±0.00 84.38 ±3.14 78.42 ±1.10 88.93 ±2.09 97.85 ±2.49 91.02 ±8.18 99.97±0.06
64 TPR@FPR=1%↑73.65 ±0.00 75.45 ±0.00 40.82 ±0.00 99.90 ±0.00 92.68 ±3.64 95.80 ±4.93 99.87 ±0.15 99.84 ±0.06 100.00 ±0.00
128 TPR@FPR=1%↑73.65 ±0.00 75.45 ±0.00 40.82 ±0.00 98.54 ±1.37 99.97±0.06 99.84 ±0.06 99.71 ±0.20 99.93 ±0.11 99.90 ±0.17
256 TPR@FPR=1%↑73.65 ±0.00 75.45 ±0.00 40.82 ±0.00 99.48 ±0.73 99.97 ±0.06 100.00 ±0.00 98.37 ±2.18 100.00 ±0.00 100.00 ±0.00
Avg.TPR@FPR=1%↑73.65 ±0.00 75.45 ±0.00 40.82 ±0.00 95.58 ±1.31 92.76 ±1.22 96.14 ±1.77 98.95 ±1.26 97.70 ±2.09 99.97±0.06
OUTFOX32 TPR@FPR=1%↑62.28 ±0.00 0.50±0.00 0.20±0.00 73.01 ±5.23 41.70 ±29.87 72.87 ±9.05 70.63 ±13.51 64.62 ±11.03 73.57±27.78
64 TPR@FPR=1%↑62.28 ±0.00 0.50±0.00 0.20±0.00 75.98 ±5.60 69.50 ±6.92 86.46 ±7.15 88.22 ±3.98 75.46 ±8.93 91.34±1.00
128 TPR@FPR=1%↑62.28 ±0.00 0.50±0.00 0.20±0.00 91.54 ±4.89 91.41 ±1.92 90.95 ±1.81 88.83 ±6.93 92.22±4.00 91.75 ±0.21
256 TPR@FPR=1%↑62.28 ±0.00 0.50±0.00 0.20±0.00 94.37 ±4.88 96.68±1.73 96.26 ±0.83 95.61 ±1.92 96.09 ±2.70 95.77 ±3.09
Avg.TPR@FPR=1%↑62.28 ±0.00 0.50±0.00 0.20±0.00 83.73 ±5.15 74.82 ±10.11 86.64 ±4.71 85.82 ±6.59 82.10 ±6.67 88.11±8.02
RAID32 TPR@FPR=1%↑69.96 ±0.00 63.57 ±0.00 26.27 ±0.00 25.33 ±22.29 1.95±0.68 15.23 ±11.48 8.33±11.17 24.06 ±30.68 33.07 ±27.01
64 TPR@FPR=1%↑69.96 ±0.00 63.57 ±0.00 26.27 ±0.00 19.82 ±18.27 5.01±2.74 21.78 ±25.26 29.56 ±26.30 57.71 ±1.29 64.32 ±13.68
128 TPR@FPR=1%↑69.96 ±0.00 63.57 ±0.00 26.27 ±0.00 67.81 ±9.39 22.17 ±11.68 44.21 ±16.23 24.71 ±35.80 68.23 ±4.80 71.84±2.29
256 TPR@FPR=1%↑69.96 ±0.00 63.57 ±0.00 26.27 ±0.00 79.56 ±6.99 53.26 ±9.33 72.49 ±1.04 34.05 ±35.06 75.88 ±4.82 83.33±3.58
Avg.TPR@FPR=1%↑69.96 ±0.00 63.57 ±0.00 26.27 ±0.00 48.13 ±14.24 20.60 ±6.11 38.43 ±13.50 24.16 ±27.08 56.47 ±10.40 63.14 ±11.64
SemEval32 TPR@FPR=1%↑66.37 ±0.00 63.77 ±0.00 13.17 ±0.00 60.38 ±6.35 12.34 ±6.68 13.37 ±0.88 58.23 ±4.74 39.81 ±29.59 63.35 ±11.30
64 TPR@FPR=1%↑66.37 ±0.00 63.77 ±0.00 13.17 ±0.00 57.62 ±14.69 46.03 ±11.50 71.52 ±7.52 72.80 ±5.84 73.00 ±4.48 73.24±7.19
128 TPR@FPR=1%↑66.37 ±0.00 63.77 ±0.00 13.17 ±0.00 70.92 ±6.98 69.62 ±4.82 72.33 ±6.14 56.84 ±32.63 78.96±7.07 72.62 ±15.72
256 TPR@FPR=1%↑66.37 ±0.00 63.77 ±0.00 13.17 ±0.00 70.80 ±19.40 70.43 ±2.44 72.87 ±2.17 76.76 ±4.80 83.79±7.41 78.84 ±18.30
Avg.TPR@FPR=1%↑66.37 ±0.00 63.77 ±0.00 13.17 ±0.00 64.93 ±11.86 49.61 ±6.36 57.52 ±4.18 66.16 ±12.00 68.89 ±12.14 72.01±13.13
OverallAvg.TPR@FPR=1%↑68.07 ±0.00 50.82 ±0.00 20.12 ±0.00 73.09 ±8.14 59.45 ±5.95 69.68 ±6.04 68.77 ±11.73 76.29 ±7.82 80.81±8.21
Table 5:TPR@FPR=1% across 4 datasets and shot sizes.We report the mean and standard deviation of
TPR@FPR=1% over 3 random seeds for each experiment (mean ±std).†denotes few-shot-tailored detectors, and∗
indicates adversarially trained detectors. The best result(s) in each row arebolded, and the second-best result(s) are
underlined .
Generation Strategy H1 H2 H3 Mean Significance
Direct Generate2.660±1.327 3.330±1.356 2.390±1.294 2.793±1.326 2.13e−12∗/–
Direct Prompt2.660±1.208 3.600±1.279 2.700±1.345 2.987±1.277 4.05e−09∗/1.16e−01
Few-shot Prompt2.720±1.207 3.320±1.230 2.510±1.337 2.850±1.258 9.16e−12∗/5.74e−01
RAG (Ours)3.560±1.149 4.120±0.856 3.010±1.283 3.563±1.096 1.18e−01/1.16e−09∗
HWT3.700±1.068 4.110±0.777 3.370±1.405 3.727±1.083–/2.13e−12∗
Table 6:Human evaluation of adversarial example generation strategies (human-likeness, 1–5).H1–H3 are
three independent raters scoring how human-like each text is (higher is more human-like).Meanreports the average
of the three rater means and the average of the three rater standard deviations.Significancereports two paired
Wilcoxon signed-rank tests (two-sided) in the formatvs HWT / vs Direct Generate.∗indicates p <0.01 (significant
difference); “–” denotes non-applicable comparisons (the row itself is the reference).
of a given HWT without any additional prompt-
ing; (2)Direct Prompt, where we use a single
instruction prompt to encourage human-like writ-
ing; (3)Few-shot Prompt, where we provide one
MGT/HWT pair as in-context guidance; and (4)
RAG (Ours), where we use retrieval-augmented
generation with curated context. Three annotators
(senior undergraduate students; English as a sec-
ond language) independently rate each text on a
1–5 human-likeness scale (higher indicates more
human-like). All texts (generated and HWT) are
fully shuffled and presented in random order; anno-
tators receive a short training session and are shown
three example pairs before annotation. Each anno-
tator finishes within two hours and is compensatedat twice the local minimum hourly wage. For sig-
nificance testing, we adopt the following protocol:
for each item i, we first aggregate the three ratings
as¯yi(Gk) =1
3P
r∈{H1,H2,H3} yi,r(Gk), and then
apply a two-sided paired Wilcoxon signed-rank test
between {¯yi(Gk)}100
i=1and the reference set (HWT
or Direct Generate), marking p <0.01 as signifi-
cant.
Experiment Results.Table 6 shows thatRAG
(Ours)is perceived as substantially more human-
like than the other generation strategies. RAG
achieves the highest mean score (3.563), closing
most of the gap to real HWT (3.727). More im-
portantly, RAG isnotsignificantly different from
HWT under our strict criterion ( p= 1.18e−01≥
17

0.01), while it is significantly different from Di-
rect Generate ( p= 1.16e−09<0.01 ). This in-
dicates that human raters consider RAG outputs
markedly more human-like than naive LLM gen-
erations, and statistically indistinguishable from
human writing at p <0.01 . In contrast,Direct
PromptandFew-shot Promptprovide limited
improvement. Both remain significantly different
from HWT ( p= 4.05e−09 andp= 9.16e−12 ),
yet neither differs significantly from Direct Gener-
ate (p= 1.16e−01 andp= 5.74e−01 ). Together,
these results suggest that retrieval-guided context is
a key factor for generating stronger, more human-
like adversarial examples, rather than relying on
lightweight prompting alone.
B.5 Case Study
Experiment Settings.To qualitatively understand
how different attacks and generation strategies
change the perceived human-likeness of machine-
generated text, we present a representative ex-
ample in Table 7 (the same sample as in Fig-
ure 1). We compare 3 perturbation-based attacks
(Modify,Paraphrasing,Back-translation) used
in §5.2 against prompting-based generation strate-
gies (Direct Prompt,Few-shot Prompt) and our
RAG (Ours)generator. All outputs are rated by
the same three annotators and follow the identical
scoring rubric and annotation protocol described
in §B.4, where each text is scored on a 1–5 scale
indicating how human-like it appears.
Experiment Results.Table 7 illustrates a
clear gap betweensurface-levelperturbations
andhumanization-orientedgeneration. Starting
from the raw MGT baseline (Mean = 2.00 ),
perturbation-based attacks either fail to improve
human-likeness or even make the text more sus-
picious. For example,Modifyslightly reshuffles
local phrasing but remains terse and list-like, re-
ceiving the lowest score (Mean = 1.33 ). Simi-
larly,Back-translationintroduces unnatural word
choice (e.g., “elementary combat”) and preserves
the original template-like structure, yielding only
a marginal Mean of 1.67.Paraphrasingchanges
casing and wording but still reads like a feature
enumeration, resulting in a limited improvement
(Mean= 2.33), comparable to the raw text.
Prompting-based strategies provide moderate
gains but remain constrained by the same “feature-
summary” writing style.Direct Promptexpands
the description with more formal phrasing, yet the
output still resembles a polished product synop-sis and receives Mean = 2.33 .Few-shot Prompt
further increases length and injects additional con-
tent (e.g., creatures and artifacts), which improves
human-likeness to Mean = 3.00 , but parts of the
added details feel generic and less grounded, limit-
ing the perceived authenticity.
In contrast,RAG (Ours)yields the most human-
like output (Mean = 4.00 ). The text not only in-
troduces richer content but also exhibits distinctly
human cues: (i) a personal stance (“there’s this
game I keep coming back to”), (ii) narrative flow
with discourse markers (“You know,” “for me”),
(iii) subjective impressions and metaphors (“almost
musical combos,” “living, breathing world”), and
(iv) balanced treatment of potentially machine-like
signals such as “gacha” by embedding them into
a coherent personal experience rather than listing
them as features. These characteristics collectively
make the generation read less like a templated de-
scription and more like natural human commentary,
explaining the consistently higher ratings across
annotators.
Overall, this case study supports the main take-
away of our human evaluation: naive perturbations
and lightweight prompting tend to preserve the
underlying “LLM-summary” signature, whereas
retrieval-guided generation can substantially shift
the writing style toward human-like discourse, pro-
ducing stronger and more realistic adversarial ex-
amples for training robust detectors.
B.6 Analysis of Training Dynamics
To better understand the convergence and stability
of our adversarial training procedure—and to exam-
ine how the surrogate attacker evolves throughout
training—we visualize the step-wise loss curves
acrossall datasetsandall shot sizesin Figure 6.
From Figure 6, we summarize three key observa-
tions.
(i) Stable optimization across the stage transi-
tion.A common concern in few-shot adversarial
training is that switching from clean supervised
learning to adversarial updates can destabilize opti-
mization, especially when data are scarce. Across
all datasets and shot sizes, our training remains
well-behaved: losses stay within a controlled range,
and the stage transition does not induce catas-
trophic spikes, divergence, or long-lasting oscil-
lations. Even in the 32-shot setting, where stochas-
ticity is typically amplified, the detector adapts
quickly after entering the adversarial stage and
maintains a stable loss profile thereafter. Over-
18

Method Text H1 H2 H3 Mean
Raw MGTGenshin Impact is an open-world action RPG featuring elemental combat,
gacha characters, and an expansive fantasy world called Teyvat.2 3 1 2.00
ModifyGenshin Impact is an open-world action RPG featuring combat, gacha and
lore, set in an expansive fantasy world called Teyvat.1 2 1 1.33
ParaphrasingTHE GENSHIN IMPACT, an open world action RPG, involves elemental
combat, Gacha-style characters and a vast fantasy land called Teyvat.2 3 2 2.33
Back-translationGenshin Impact is an open-world action RPG with elementary combat,
Gacha characters and an expansive fantasy world called Teyvat.1 3 1 1.67
Direct PromptAn immersive experience, Genshin Impact is an open-world action role-
playing game. It presents players with a dynamic elemental combat system,
a character acquisition model based on the gacha mechanic, and a vast
fantasy setting known as Teyvat for exploration.2 3 2 2.33
Few-shot PromptGenshin Impact is an open-world action RPG featuring elemental combat,
gacha characters, and an expansive fantasy world called Teyvat. Here,
players can explore a rich, dynamic world filled with a variety of mystical
creatures, mystical beasts, and powerful magical artifacts, all of which
contribute to the diverse gameplay of the title.3 4 2 3.00
RAG (Ours)You know, there’s this game I keep coming back to—Genshin Impact. It
drops you into this breathtaking, wide-open fantasy realm called Teyvat,
where you’re free to wander and discover. The real magic, for me, is in
the combat. You get to mix and match elemental powers in these dazzling,
almost musical combos. And the characters? Sure, you collect them
through the gacha system, but they all have such rich stories. It feels less
like checking off features and more like stepping into a living, breathing
world that’s just waiting to be explored.4 5 3 4.00
Table 7:Case study of attacks on the samples in Figure 1.The scoring rubric and annotation protocol are the
same as §B.4.Meanis computed as the average ofH1–H3.
all, these curves suggest that our training schedule
and objectives avoid abrupt distribution shifts that
could derail learning.
(ii) Alternating updates yield a meaningful
attacker–detector interaction.Within the adver-
sarial stage, we observe a consistent coupling be-
tween the attacker-side and detector-side objectives.
In many settings, fluctuations in the attacker loss
Lattcoincide with corresponding changes in the
detector’s adversarial loss LACL. This pattern is
expected under our alternating-update design: as
the attacker produces rewrites that are harder for
the current surrogate (and thus harder to imitate the
target detector’s label-only feedback), the detector
is exposed to more challenging humanized samples
and places greater emphasis on maintaining correct
decisions under humanization, which is reflected
byLACL. Subsequently, as the detector adapts, both
losses settle into a more stable regime. Importantly,
the sustained dynamics across datasets and shot
sizes indicate that the surrogate attacker does not
trivially collapse to a fixed, non-informative rewrit-
ing pattern; instead, the interaction resembles an
adaptive curriculum in which the detector is con-
tinually trained on progressively more challenginghumanized candidates.
(iii) PBC provides a fast, low-variance signal
that complements LACL.Finally, Figure 6 shows
clear convergence behavior within each stage. Dur-
ing pre-training, Lcedecreases smoothly, indicating
effective fitting on clean supervision. After switch-
ing to adversarial training, LACLandLPBCrapidly
move toward a stable regime rather than remain-
ing persistently high in low-data regimes. Notably,
LPBCtypically becomes active early and stabilizes
quickly. This behavior aligns with our motivation
for PBC: as a pairwise, margin-based geometric
constraint, it provides a low-variance learning sig-
nal that does not rely on large batches or many
in-batch negatives, which are often unreliable in
few-shot settings. Empirically, the early stabiliza-
tion of LPBCis accompanied by faster stabilization
ofLACL, suggesting that PBC helps regularize rep-
resentation geometry and facilitates more efficient
convergence of the main adversarial classification
objective.
Overall, Figure 6 supports that our frame-
work yields (1) stable optimization across the pre-
training–to–adversarial transition, (2) an effective
attacker–detector interaction under alternating up-
19

Figure 6:Loss curves over training steps.We report step-wise losses for both the detector and the attacker,
including the detector cross-entropy loss during pre-training ( Lce), the attacker loss during pre-training ( Latt), the
detector ACL loss during adversarial training ( LACL), the detector PBC loss during adversarial training ( LPBC),
and the attacker loss during adversarial training ( Latt). The red vertical line marks the stage transition, and the
adversarial training stage is highlighted with a light-red background.
dates with label-only feedback, and (3) reliable
convergence with PBC providing an additional,
well-conditioned signal that complements ACL in
few-shot adversarial training.
20