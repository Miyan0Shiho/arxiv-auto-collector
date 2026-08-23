# TTSD-FAR: Test-Time Self-Distillation with Fisher-Anchored Restoration for Missing-Modality Emotion Recognition in LVLMs

**Authors**: Muhammad Haseeb Aslam, Alessandro Koerich, Marco Pedersoli, Ali Etemad, Eric Granger

**Published**: 2026-08-18 23:40:28

**PDF URL**: [https://arxiv.org/pdf/2608.18386v1](https://arxiv.org/pdf/2608.18386v1)

## Abstract
Large video-language models (LVLMs) have shown remarkable performance on multimodal tasks like multimodal emotion recognition (ER) in the wild. ER is inherently multimodal, requiring a joint understanding of facial expressions, vocalizations, language, biosignals, and gestures. However, real-world deployment remains challenging: modalities may be missing or noisy at test time. Partial observations can be viewed as a distribution shift relative to the complete-modality distribution. SOTA TTA methods based on entropy minimization or perplexity reduction do not transfer to autoregressive LVLMs, while retrieval augmented generation (RAG) degrades when the observed modality is weak. Because no ground-truth supervision exists to verify individual updates, adaptation across this stream risks accumulating drift and degrading once the model departs from a reliable solution. An effective solution must therefore adapt to arbitrary missing-modality patterns and remain effective during continual adaptation. We address both jointly with Test-Time Self-Distillation (TTSD), a parameter-efficient framework in which a frozen teacher, trained on complete modalities, guides an adaptive low-rank student via self-distillation, updating only a negligible number of parameters. Stability is built into this same loop through Fisher-Anchored Restoration (FAR), which monitors Fisher information stability to detect convergence versus drift and restores the student toward the teacher's anchor when distributional shifts are identified. Our experiments on MELD, DFEW, and BAH under 0%-50% missing modalities show that this unified adaptation-restoration design consistently outperforms entropy-based adaptation, RAG, and perplexity-based generation over long adaptation horizons, where baselines without restoration progressively degrade while TTSD-FAR remains consistent.

## Full Text


<!-- PDF content starts -->

TTSD-FAR: Test-Time Self-Distillation with Fisher-Anchored Restoration for
Missing-Modality Emotion Recognition in LVLMs
Muhammad Haseeb Aslam1, Alessandro Koerich1, Marco Pedersoli1, Ali Etemad2, Eric
Granger1
1LIVIA, ETS Montreal, Canada
2Aiim Lab, Queen’s University, Canada.
muhammad-haseeb.aslam.1@ens.ettsmtl.ca
Abstract
Large video-language models (LVLMs) have shown remark-
able performance on multimodal tasks like multimodal emo-
tion recognition (ER) in-the-wild. ER is inherently multi-
modal, requiring a joint understanding of facial expressions,
vocalizations, language, biosignals, and gestures. However,
real-world deployment remains challenging: modalities may
be missing or noisy at test time. Partial observations can be
viewedasadistributionshiftrelativetothecomplete-modality
distribution. State-of-the-art TTA methods based on entropy
minimization or perplexity reduction do not transfer to au-
toregressive LVLMs, while retrieval-augmented generation
(RAG) degrades when the observed modality is weak. Be-
cause no ground-truth supervision exists to verify individual
updates,adaptationacrossthisstreamrisksaccumulatingdrift
and degrading once the model departs from a reliable solu-
tion. An effective solution must therefore adapt to arbitrary
missing-modality patterns and remain effective during con-
tinual adaptation. We address both jointly withTest-Time
Self-Distillation(TTSD),aparameter-efficientframeworkin
whichafrozenteacher,trainedoncompletemodalities,guides
an adaptive low-rank student via self-distillation, updating
onlyanegligiblenumberofparameters.Stabilityisbuiltinto
thissameloopthroughFisher-AnchoredRestoration(FAR),
which monitors Fisher information stability to detect conver-
genceversusdriftandrestoresthestudenttowardtheteacher’s
anchor when distributional shifts are identified. Our experi-
ments1onMELD,DFEW,andBAHunder0%–50%missing
modalities show that this unified adaptation-restoration de-
signconsistentlyoutperformsentropy-basedadaptation,RAG,
and perplexity-based generation over long adaptation hori-
zons, where baselines without restoration progressively de-
grade while TTSD-FAR remains consistent.
1 Introduction
The rapid advancement of LVLMs (Lin et al. 2023; Maaz
et al. 2024; Li et al. 2023) has revolutionized multimodal
learningtasks,achievingunprecedentedperformance.These
models leverage transformer architectures with billions of
parameters, pre-trained on massive video-text corpora, to
develop rich cross-modal representations (Radford et al.
1Our code is included in the supplementary materials and will
be made public.
Copyright©2027, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.2021). LVLMs are promising for video-based multimodal
ER (MER), where systems typically capture facial, vocal,
and language cues to address subtle (or compound) expres-
sions and high inter-subject variability (Huang et al. 2025;
Chengetal.2024).RecentworkshowsthatLVLMscancap-
ture fine-grained emotional and health states and contextual
relationships that traditional discriminative models fail to
detect (Ge, Tang, and Li 2024). However, their deployment
inreal-worldapplicationsconfrontsafundamentalchallenge
thathasreceivedsurprisinglylittleattention:missingmodal-
ity at test time.
Inpracticalvideo-basedMERscenarios,modalityabsence
is ubiquitous and inevitable. Inputs may suffer from audio
corruptioninnoisyenvironments,transcriptionservicesmay
failduetopooraudioqualityorprivacyconstraints,andsen-
sormalfunctionscaneliminateentiremodalitiesduringdata
collection (Wang et al. 2020). Traditional approaches ad-
dressthisthroughtraining-timestrategies,eitherbytraining
separate models for each modality combination or by em-
ploying modality dropout during training (Nezakati et al.
2024). Yet both approaches are impractical for LVLMs, as
retrainingbillion-parametermodelsforeachmissingmodal-
ity scenario is computationally prohibitive (requiring1020
FLOPsfor7Bparametermodels),whiletraining-timemodal-
ity dropout degrades full-modality performance and cannot
adapt to deployment-specific missing patterns (Ma et al.
2021). Ramazanova et al. (2025) pose the missing modality
as a domain shift problem and propose a method for TTA.
The authors claim that despite advancements in the missing
modality literature, a common drawback persists: all state-
of-the-artapproachesnecessitateexpensiveretrainingofthe
multimodal model. This poses a substantial challenge, par-
ticularly in applications with: (i) extensive training data; (ii)
where the retraining process is prohibitively expensive; and
(iii)whensourcedataisnotavailable,makingtheaforemen-
tioned approaches impractical.
As shown with recent foundation models (e.g., GPT-style
architectures and large multimodal systems (Maaz et al.
2024; Lin et al. 2023)), retraining large language models
(LLMs) and LVLMs is prohibitively expensive, requiring
specialized hardware and weeks of distributed optimiza-
tion. It is therefore impractical to retrain or fully fine-tune
these models for every downstream task or deployment set-
ting due to catastrophic forgetting, computational cost, and
arXiv:2608.18386v1  [cs.CV]  18 Aug 2026

source data availability. This motivates the development of
lightweightTTAmethodsforefficientcustomizationwithout
modifying the full parameter set.
Weinvestigatethedomainshiftcausedbymissingmodal-
ity inputs from 3 different viewpoints: (i) UMAP visualiza-
tionofthemodel’sinternalrepresentations;(ii)MMDcom-
putations between the full and missing modality inputs of
the same sample; and (iii) the degradation of performance.
We perform this analysis on the MELD dataset with 50%
text-missing inputs using theVideo-LLaVA-7Bmodel. Fig-
ure 1 displays the representation geometry of the complete
and text-missing inputs. The two colors correspond to the
two classes. The•represents the full modality inputs and
×represents the missing modality input for the same data
samples. It reveals that the full and missing inputs are not
only offset, but that missing inputs also blur the decision
boundary. With full-modality inputs, the two classes form
well-separated clusters, whereas under partial observation,
this separation collapses entirely, with both classes overlap-
ping into a single indistinguishable region
Quantitative analysis reveals a substantial internal distri-
bution shift when modalities are missing, with an MMD
of0.8350andcosinesimilarityof0.6342betweencomplete
andmissinginputsforthesameinputsamples.Thisindicates
that the absence of modalities can significantly disturb the
learned attention manifold, even though the semantic con-
tent remains unchanged. Lastly, the decline in performance
shown in Tables 1-3 indicates that LVLMs experience a se-
vere performance drop (F1-score 0.6149→0.4785) in the
50% text-missing case. Entropy-based TTA methods tend
to drop towards near-random performance after adaptation.
This observation is in line with the findings of Hu et al.
(2025), which showed that entropy minimization objectives
are ill-suited for the TTA of LLMs.
A second, less-addressed challenge in online TTA for
LLMs and LVLMs is perpetual adaptation. In practice, two
failuremodesemerge.First,oncethestudentLoRAhascon-
verged, continued updates introduce gradient noise that per-
turbs the learned representation without improving perfor-
mance (Wang et al. 2022; Niu et al. 2023). Second, test
streamsinrealdeploymentsarenon-stationary:speakeriden-
tities,recordingconditions,andmissing-modalityratesshift
overtime.Perpetualadaptationwithoutdriftawarenessrisks
overwriting consolidated knowledge, structurally analogous
to catastrophic forgetting (Kirkpatrick et al. 2017). Moni-
toring the raw distillation loss is an unreliable criterion for
eitherstoppingorrestarting:thelosscanplateauduetosam-
pledifficultyratherthangenuineparameterconvergenceand
canspikeduetohardwithin-distributionsamplesratherthan
atruedistributionalshift.Thismotivatestwodesignchoices:
adaptationneedsasupervisionsignalthatfitsautoregressive
LVLMs, motivating latent alignment over entropy. A stop-
ping signal grounded in the parameters themselves rather
than the loss, motivating the Fisher information diagonal,
whose shift from its value at convergence exposes genuine
drift and lets adaptation pause and reactivate accordingly.
The contributions of this paper are summarized as fol-
lows:(1)TTSD: a parameter-efficient TTA method tailored
to autoregressive LVLMs under missing modalities, built
Figure 1: UMAP of the internal representations. The cross
(×)representsthemissingmodalityinputs,andthecircle(•)
represents the full modality inputs.
around conditional latent representation alignment as a su-
pervisionsignalthatovercomesthefailuremodesofentropy-
basedobjectivesandretrievalmethods.(2)FAR:atwo-state
stopping and drift-detection and reactivation mechanism for
TTSD. It monitors the Fisher information diagonal to de-
tect student convergence (active→anchored) and uses
a Fisher Mismatch Index to detect test-stream distributional
shift (anchored→active).(3)An extensive set of experi-
mentsonthreechallengingvideo-basedERdatasets,namely
MELD (Poria et al. 2019), DFEW (Jiang et al. 2020), and
BAH(González-Gonzálezetal.2026),withmultiplemissing
ratiosandacrossdifferentmodalitiesshowstheeffectiveness
of our proposed TTSD-FAR method.
2 Related Works
Learning with Missing Modalities.Multimodal learning
with missing modalities has been extensively studied, with
early works such as ModDrop (Neverova et al. 2016),
SMIL (Ma et al. 2021), and masked modality projection
(MMP) (Nezakati et al. 2024) introducing train-time strate-
gies to enhance robustness. Maheshwari, Liu, and Kira
(2024) propose a multi-modal teacher for masked modality
learning, improving semantic segmentation under missing
modalities.Somemethodshaveexploredmodality-invariant
architectures where single-branch models trained to be in-
variant to modality absence exhibit robustness during both
training and testing (Saeed et al. 2024; Zhao, Mao, and
Chen 2021; Havaei et al. 2016). Guo, Jin, and Zhao (2024)
use prompt learning in specialized cross-modal transformer
architectures to regenerate the missing modality. More re-
cently, Chen et al. (2026) proposed SMCIR, which de-
tects sample-level modality missingness via an unsuper-
visedentropy/mutual-information/similarityscore(DMFD),
thenreconstructsmissingmodalitiesthroughcontext-guided
multi-scale cross-modal attention (CMCG). Despite this
progress, most methods rely on training-time strategies, ei-
ther reconstructing missing modalities, simulating modal-

ity absence during training, or making custom architectural
changes. These train-time methods are not directly transfer-
able to LVLMs, which are expensive to re-train and whose
generativedecoderslackclearself-supervisedobjectivesthat
correlate with generation quality under modality absence.
Multimodalsystemsoftenexhibitanimbalanceinmodal-
ity strength, with some modalities providing stronger task-
relevant signals than others. When the stronger modality is
absent, models experience a significant performance drop,
highlighting the need for mechanisms that preserve knowl-
edge from complete multimodal observations when operat-
ing even with weaker modalities.
Test-Time Adaptation.TTA and test-time training have
emerged to handle distribution shifts at inference time with-
out access to target labels (Zhang, Levine, and Finn 2022;
Wang et al. 2022). TENT proposes entropy minimization
by adapting normalization statistics at test time (Wang et al.
2021).EATA(Niuetal.2022)improvesentropy-basedTTA
byselectivelyupdatingmodelparametersusingonlyreliable
low-entropy samples and regularizing updates with a Fisher
constraint to prevent forgetting. Many recent works analyze
batch normalization for robustness under shift (Nado et al.
2021). A related line of continual-TTA work decides when
to reset via drift detection rather than fixed schedules or en-
tropy alone: periodic resets to pretrained weights curb long-
horizon collapse (Press et al. 2023). Mishra (Mishra 2026)
introduced RDumb++, a principled extension of RDumb
thatintroducedtwodrift-detectionmechanisms,i.e.,entropy-
baseddriftscoringandKL-divergencedriftscoring,together
with adaptive reset strategies. Alternating domain construc-
tion was proposed by (Lin, Huang, and Lee 2024). Sun
etal.(2020)insteadoptimizeauxiliaryself-supervisedlosses
during inference to improve adaptation. These methods are
tailored to models that produce a single softmax distribu-
tion over a small, fixed label set, where output entropy is
a direct, low-dimensional measure of prediction confidence.
Self-supervised objectives and associated stopping signals
do not transfer to LVLMs. Hu et al. (2025) introduced test-
time learning for LLMs (TLM), adapting LLMs during in-
ference using only unlabeled test data via input perplexity
minimizationratherthanentropyminimizationorsupervised
fine-tuning.Adaptingentirelayersornormalizationstatistics
in billion-parameter models further raises stability and effi-
ciencyconcerns,motivatinglightweight,parameter-efficient
adaptation instead. Unlike EATA’s per-step Fisher regular-
ization, TTSD-FAR uses the Fisher diagonal as a control
signal that suspends and resumes adaptation entirely.
Retrieval-Augmented Generation.RAG methods improve
LLM and LVLM’s performance during inference by incor-
porating external knowledge. Relevant information from a
knowledge base is retrieved during inference (Asai et al.
2023; Zhao et al. 2024; Qian et al. 2025). RAG methods
prove effective for tasks that require domain knowledge, but
they rely heavily on the quality of retrieved information and
incur additional computational latency. Precomputed vector
databases or memory banks are also required for retrieval
during inference (Abootorabi et al. 2025).
State-of-the-artmethodstoaddressmissingmodalitiesei-
therrequireretrainingwitheachmodalitycombinationorde-pend on discriminative learning objectives and specialized
architectures to hallucinate missing modalities. The main
issue faced by LVLMs for TTA is that there is no strong su-
pervision signal available for adaptation. Entropyobjectives
do not provide meaningful supervision, and RAG schemes
fail if the observed modality is too weak to retrieve correct
class neighbors. This paper fills such a gap by proposing a
novelapproachforfeature-leveladaptationusingdualLoRA
adapters through self-distillation and FAR for convergence
monitoringthatallowsstoppingandresumingadaptationdy-
namically.
3 Proposed Methodology
LetX=Xv×Xtdenotethemultimodalinputspaceconsist-
ing of video and text modalities, whereXvrepresents video
frames andXtrepresents textual utterances. LetYdenote
the output label space (e.g., emotion classes). We assume
accesstoapre-trainedLVLMfθ:X → Y, θ∈Rd,trained
on large-scale multimodal corpora. The model consists of
(i) modality-specific encodersEv:Xv→Rn×dv, Et:
Xt→Rm×dt,and(ii)aLLMθbackbone:R(n+m)×d→ Y,
which performs cross-modal fusion and autoregressive rea-
soning. We denote the frozen pre-trained backbone param-
eters byθ0and the teacher and student LoRA weights by
∆θtea=BteaAteaand∆θstu=BstuAstu, respectively. We
further denote the vectorized student LoRA parameters as
ϕ=vec(Astu, Bstu)∈Rp, wherepis the total number of
student LoRA parameters.
3.1 Problem Formulation
Latent Inference under Partial Observation.Letx=
(xv, xt)∼P fulldenote complete multimodal inputs drawn
from the training joint distribution. At test time, modali-
ties may be missing. We introduce binary modality masks
mv, mt∈ {0,1},and define the partially observed input as
˜x= (˜xv,˜xt) = (mv·xv, mt·xt).(1)
The teacher’s intermediate representation at layerlis
htea(x) = LLMθ0+∆θtea
l (xv, xt)∈Rs×d.(2)
The missing-modality setting can be viewed as apartial
observation problem, where the learner must infer a rep-
resentation consistent with the latent multimodal structure
learned from complete data. We interprethtea(x)as a latent
variable lying on a multimodal representation manifoldZ.
Undersquarederrorloss,theBayes-optimalestimatorofthe
complete-modality representation given the observed input
˜xis the conditional expectation
h∗(˜x) =E
htea(x)|˜x
.(3)
TTSD as Conditional Latent Reconstruction.The stu-
dent adapter parameterized by∆θstuproduces
hstu(˜x) = LLMθ0+∆θstu
l (˜xv,˜xt)∈Rs×d.(4)
TTSD minimizes the feature-level distillation objective
Ldistill=Ex∼Pfullhhstu(˜x)−htea(x)2
2i
,(5)

LVLM
✓complete
You liked 
it, really?
You liked 
it?Test-time input stream
Which part 
exactly?The whole 
thing!
can we go?
------
✘vision ✓complete ✘text
timecomplete adapt student via self distillation
missing student inference only (no update)Teacher 
LoRA
Student 
LoRABase Weights
𝞱Mask
෤𝑥
Modality 
Router
ℎ𝑙𝑠𝑡𝑢−ℎ𝑙𝑡𝑒𝑎
22ℒ𝑑𝑖𝑠𝑡𝑖𝑙𝑙
ACTIVE
ANCHORED
𝐶𝐼𝑡<𝜏𝑓𝑐𝐹𝑀𝐼𝑡>𝜏𝑓𝑚𝑖FAR𝑥=(𝑥𝑡,𝑥𝑣)
෤𝑥∈(𝑚𝑡,𝑥𝑣
𝑥𝑡,𝑚𝑣)hltea +
+
layer i∆𝞱tea
∆𝞱stuhlstuFigure 2: Illustration of the proposed TTSD-FAR method. The solid green arrow shows the flow of information in the case of
complete modalities, and the dashed red arrow shows the flow of information in the missing modality case. The solid green
arrow shows both modalities, and the single dashed green arrow denotes the input (˜x) with masked input. In the missing case,
the model bypasses adaptation and only infers with the student LoRA. In the complete case, the model uses complete modality
features from the teacher LoRA and masked modality features from the student LoRA, and distillation lossL distillis calculated
to match the student representation with the teacher representation.
whose minimizer satisfieshstu∗(˜x) =E[htea(x)|˜x], so that
under a squared-error objective, the student is encouraged
to approximate the conditional expectation of the teacher
representation given partial input.
However, perpetual adaptation ofϕacross an unbounded
test stream introduces two failure modes: (i) once∆θstuhas
converged,continuedgradientupdatesaddnoisewithoutim-
provingthelearnedrepresentation;and(ii)innon-stationary
streams, parameter drift can overwrite previously consoli-
dated knowledge. We therefore augment TTSD with Fisher-
AnchoredRestoration(FAR),atwo-statemechanismthatde-
termineswhenadaptationshouldproceedandwhenitshould
besuspended,basedsolelyonthegeometryoftheFisherim-
portance landscape.
3.2 Architecture Overview
OurTTSD-FARframeworkoperatesonasinglebasevideo-
language modelfθ0with two distinct LoRA adapters. Fig-
ure 2 illustrates the architecture. The architecture enables
efficient mode switching:
Teacher mode:ˆytea=fθ0+∆θtea(xv, xt)(6)
Student mode:ˆystu=fθ0+∆θstu(˜xv,˜xt)(7)
3.3 TTA: Student LoRA
At test time, when encountering a samplexwith com-
pletemodalities,weperformonlineadaptationofthestudent
LoRAthroughfeature-levelself-distillationfromtheteacher.
By directly matching intermediate representations, the stu-
dent adapter is encouraged to capture the internal geometry
of the teacher representation space rather than imitating its
final decisions. This preserves the diversity and richness of
the teacher features and mitigates mode collapse, which is
particularly critical under missing-modality scenarios.Given a layerl∈ Lof the LLM backbone, the hidden
representations are:
htea
l=LLMθ0+∆θtea
l(Ev(xv), Et(xt))∈Rs×d(8)
hstu
l=LLMθ0+∆θstu
l(Ev(˜xv), Et(˜xt))∈Rs×d(9)
wheresisthesequencelength(thesumofvideotokensand
texttokens)anddisthehiddendimension.Weminimizethe
mean squared error between student and teacher representa-
tions as:
Ldistill=X
l∈Lαl· ∥hstu
l−sg(htea
l)∥2
2(10)
where sg(·)denotes the stop-gradient operation to prevent
backpropagation into the frozen teacher, andα lare layer-
specific weights.
Teacherfeaturesareextractedoncepersample.Toensure
complete isolation, features are detached from the teacher
network because no gradient calculation is required in the
teacher LoRA. Teacher and student forward passes do not
share computational graph nodes, preventing gradient con-
tamination.Withoutisolation,sharednormalizationstatistics
or attention caches can cause interference.
Why Self-Distillation for Missing Modalities?
Theteacher,trainedoncompletemodalities,haslearned
richcross-modalfeaturesthatfuseinformationfromboth
vision and text. When a modality is missing at test time,
thestudentcanbeviewedassolvinganinverseproblem:
∆θstu= arg min
∆θE˜x∥hstu(˜x)−h tea(x)∥2
2.(11)

3.4 Fisher-Anchored Restoration (FAR)
Continuousadaptationofϕisnotharmless:oncethestudent
hasfoundasolutiontoEq.(11),additionalgradientstepson
the same distribution degrade performance through noise
accumulation, while distribution shifts in the test stream
can silently overwrite consolidated knowledge. Monitoring
Ldistilldirectly is unreliable as a stopping signal since the
loss can plateau not because the student has converged, but
because the current batch of samples is uniformly hard. The
correct stopping signal is the stability of theparameter im-
portance landscape. Specifically, whether the curvature of
Ldistillwith respect toϕhas stabilised. Symmetrically, drift
shouldnotbedetectedbyrawlossspikes,whichcanbetrig-
geredbyhardsamplesinastationarystream.Instead,driftis
detected by monitoring whether incoming gradients engage
parameterdimensionsthatwereunimportantduringconsol-
idation. FAR formalizes this intuition through two comple-
mentary Fisher-based statistics and a two-state mechanism:
active(the student LoRA is being updated) andanchored
(the student LoRA is frozen).
Online Fisher Diagonal.After each adaptation step, the
diagonal of the empirical Fisher information matrix overϕ
is updated via an exponential moving average:
Ft=βF·Ft−1+ (1−β F)·(∇ ϕLdistill)⊙2(12)
where(·)⊙2denotes element-wise squaring,β F∈(0,1)is
thedecayparametercontrollingtheweightplacedonhistor-
ical estimates. The elementF t,japproximates the expected
squaredgradientwithrespecttothej-thparameterinϕ,pro-
viding a running estimate of each parameter’s contribution
to the distillation objective.F tis updated only during the
activestate.
ConsolidationIndexandtheactive→anchoredTran-
sition.Theconsolidationindex(CI)measuresthenormal-
izedℓ 1rate of change of the Fisher diagonal between con-
secutive adaptation steps:
CIt=∥Ft−Ft−1∥1
∥Ft∥1+ε(13)
whereε >0is a small numerical stability constant. When
CItis large, the importance landscape is still evolving, i.e.,
the student has not yet settled on a stable curvature config-
uration. As the student converges,CI t→0, indicating that
theFisherdiagonalhasstoppedshifting.Theℓ 1normischo-
sen becauseF tis a non-negative vector, andℓ 1gives the
total mass shifted in the importance landscape, which has
a direct interpretation as the number of parameters whose
importance is still changing. The transition fromactive→
anchoredis triggered whenCI tremains below a threshold
τfcforKconsecutiveadaptationstepsCI t< τfc.Thestudent
LoRA parametersϕare then frozen; no weight updates are
applied while in theanchoredstate, and the currentF tis
stored as an anchor,F anchor =Ft.
Fisher Mismatch Index and theanchored→ac-
tiveTransition.Whileanchored, for each incomingcomplete-modality sample, the gradient ofL distillis com-
puted at the frozen anchor parameters without applying a
weight update:
gt=∇ ϕLdistill
ϕ=ϕanchor(14)
The Fisher mismatch index (FMI) for sampletis then
FMI t=1
ppX
j=1g2
t,j
Fanchor,j +ε(15)
wherep=|ϕ|is the total number of student LoRA parame-
ters.Theratiog2
t,j/Fanchor,jislargewhenthecurrentsample
induces a strong gradient in parameterj, but that parame-
ter had low importance during consolidation. This points to
large gradients in directions the consolidated student found
unimportant. The transition fromanchoredback toactive
is triggered whenFMI t> τfmiexceeds a threshold.
4 Results and Discussion
4.1 Experimental Methodology
Datasets.We evaluate our approach on three widely used
multimodal and video-based emotion recognition bench-
marks: Multimodal Emotion Lines Dataset(MELD)(Po-
ria et al. 2019), Dynamic Facial Expression in-the-Wild
(DFEW)(Jiangetal.2020),andBehavioralAmbivalence/H-
esitancy(BAH)(González-González et al. 2026), each pre-
senting distinct challenges in terms of modality diversity,
temporal dynamics, and subtle affective cues. Details of the
datasets are provided in the supplementary material. We ex-
clusivelyvalidateonemotionrecognition,whichisoneofthe
fewmultimodalapplicationswheretexttranscriptsserveasa
crucial,non-redundantinputmodality.Thismakesitanideal
testbed for missing-modality robustness, since dropping the
text modality induces genuine information loss.
Evaluation Protocol.This paper makes no assumptions
about the training phase. TTSD-FAR only requires a model
that works with multiple modalities at test time. To the best
ofourknowledge,thisisthefirstTTAmethodthateffectively
adaptstomissingmodalitiesinLVLMs.TTSD-FARupdates
the model only when it encounters modality-complete sam-
ples but generates predictions for all samples, regardless of
the modalities they contain. Missing modality scenarios are
simulatedbyrandomlymaskingmodalitiesatinferencetime,
withmissingratiosrangingfrom0%(complete)to50%.All
baselinesusethesamefrozenLVLMbackboneforfaircom-
parison. Detailed implementation settings for each dataset
are provided in the supplementary material.
4.2 Results under Missing Modalities
Tables 1– 3 summarize performance across MELD, DFEW,
and BAH under progressively increasing missing-modality
ratios.Acrossallthreedatasets,entropy-basedTTAmethods
(TENT and EATA) consistently underperform, confirming
thatconfidence-basedobjectivesprovideunreliablesupervi-
sion for autoregressive LVLMs under severe modality shift.
This observation is consistent with the findings of Hu et al.
(2025),whoshowedthatentropyminimizationdoesnotpro-
videreliableoptimizationsignalsforautoregressivelanguage

Table1:Comparisonofmethodsunderprogressivelymissingtextualandvisualmodalities.WereportF1ontheMELDdataset
as the proportion of unavailable input increases. N/A: Perplexity Gen. is not applicable to the vision-missing case.
Text Missing Vision Missing
Method Full 10% 20% 30% 40% 50% 10% 20% 30% 40% 50%
No Adaptation 0.6149 0.5926 0.5706 0.5390 0.5190 0.4785 0.6122 0.6025 0.5934 0.5825 0.5775
TENT(ICLR’21)– 0.1650 0.1680 0.1765 0.1855 0.1655 0.1658 0.1725 0.1655 0.1838 0.1826
EATA(ICML’22)– 0.2854 0.2690 0.2770 0.2736 0.2482 0.2950 0.2750 0.2745 0.2530 0.2382
RAG(ICCV’25)– 0.5640 0.5248 0.5120 0.4845 0.4660 0.5925 0.5830 0.5634 0.5567 0.5534
Perplexity Gen. – 0.5680 0.5460 0.5255 0.4938 0.4722 N/A N/A N/A N/A N/A
TTSD-FAR (Ours)–0.6115 0.5890 0.5625 0.5454 0.5124 0.6130 0.6075 0.6055 0.5970 0.5915
modelsattesttime.RAGandperplexity-basedgenerationre-
maincompetitivewhentheobservedmodalityissufficiently
informative,buttheirperformancedegradesasthedominant
modality becomes unavailable.
On MELD (Table 1), removing text causes substantially
greaterdegradationthanremovingvision(0.6149→0.4785
versus0.6149→0.5775), indicating that textual infor-
mation provides the dominant supervision signal for emo-
tion recognition in conversational settings. Consequently,
retrieval-based methods struggle once textual information
is unavailable, while entropy-based methods rapidly col-
lapse toward near-random performance. TTSD-FAR instead
leverages complete-modality teacher representations to su-
pervise a student operating on synthetically masked inputs,
allowing it to remain close to the full-modality baseline
acrossbothmissing-textandmissing-visionsettings.Asim-
ilar trend is observed on DFEW (Table 2), where removing
visual information induces a substantial performance drop
because facial dynamics constitute the dominant modality.
Entropy-basedadaptationagainfails,whileRAGdeteriorates
as weaker visual representations reduce retrieval quality.
TTSD-FAR consistently achieves the strongest performance
across all missing ratios, demonstrating that feature-level
self-distillation provides a substantially more reliable adap-
tationsignalthanconfidence-orretrieval-basedapproaches.
Results for the weaker text-missing scenario are included in
the supplementary material. Results on BAH (Table 3) fur-
ther demonstrate the generality of the proposed framework.
Although entropy-based methods perform relatively better
because BAH is a binary classification task, they remain
consistently inferior to TTSD-FAR. RAG and perplexity-
based generation improve upon the zero-shot baseline at
lower missing ratios but gradually deteriorate as textual in-
formationbecomesincreasinglyunavailable.Additionalim-
plementationdetailsandsupplementaryexperimentsarepro-
vided in the supplementary material.
4.3 Computational Complexity Analysis
Experiments show that TTSD-FAR performs well when
adapting to missing modality conditions at test time. How-
ever, this performance gain comes at a computational cost.
We distinguish the two cases and explain the computational
overhead for both cases. i)Full modality case:TTSD-FAR
requires two forward passes, one forward pass for obtaining
the teacher features and one for obtaining the student fea-Table2:Comparisonofmethodsunderprogressivelymissing
visual modality. We report F1 on the DFEW dataset as the
proportion of unavailable visual input increases.
Method Full 10% 20% 30% 40% 50%
No Adaptation0.55490.5300 0.5078 0.4984 0.4837 0.4434
TENT – 0.1548 0.1553 0.1665 0.1658 0.1640
EATA – 0.2534 0.2778 0.2735 0.2845 0.2885
RAG – 0.5340 0.5048 0.5020 0.4800 0.4468
TTSD-FAR (Ours)–0.5515 0.5300 0.5195 0.5015 0.4775
Table3:Comparisonofmethodsunderprogressivelymissing
textual modality conditions for BAH dataset.
Method Full 10% 20% 30% 40% 50%
Zero-Shot0.71420.6835 0.6742 0.6526 0.6215 0.6023
TENT – 0.4923 0.4834 0.4710 0.4572 0.4550
EATA – 0.5630 0.5467 0.5360 0.5310 0.5250
RAG – 0.6835 0.6730 0.6535 0.6215 0.5950
Perplexity Gen. – 0.6870 0.6745 0.6535 0.6310 0.6030
TTSD-FAR (Ours)–0.7050 0.6925 0.6695 0.6405 0.6250
tures. ii)Missing modality case:For the missing modality
case,thereisnocomputationaloverhead.Themodelusesthe
adapted student LoRA for prediction. We update only 44M
parameters out of the 7B parameters of theVideo-LLaVA
7Bmodel. This results in only≈0.629% of the parameter
updates. It is also important to note that the total wall time
fortheadaptationscenariodependsonthemissingratio.For
the 50% missing case, 50% of the samples will pose no ad-
ditional overhead; the remaining 50% of the input samples
require two forward passes and backpropagation to the stu-
dentLoRA.Consequently,thetotalwalltimeofthecomplete
MELDtestsetwithoutadaptationtakes≈1.5hours,andthe
time taken with adaptation is≈2.1 hours.
4.4 Mechanistic Analysis of FAR
To isolate the contribution of FAR, we compare TTSD with
and without the restoration mechanism across MELD and
BAH under increasing missing rates.
As shown in Figure 3, the unbounded variant (TTSD w/o
FAR), which continues to adapt the student LoRA on ev-
ery incoming sample, consistently underperforms the FAR-

10% 20% 30% 40% 50%
Missing Rate (%)0.500.520.540.560.580.600.62F1 Score
MELD
10% 20% 30% 40% 50%
Missing Rate (%)0.620.640.660.680.70F1 Score
BAHEffect of Fisher-Anchored Restoration (FAR) Module across Datasets and Missing Rates
FAR gain TTSD (Ours) w/ FAR TTSD w/o FARFigure3:EffectivenessoftheFARmoduleonthedistillation
results across different missing rates.
Figure 4: Fisher importance heatmaps (log 10scale): TTSD-
FAR(left),TTSD(middle),anddifferencemap(right),con-
firming systematically higher importance under FAR.
governed version at every missing rate. The consistent FAR
gain across both the 7-class (MELD) and binary (BAH) set-
tings indicates that this stopping/restarting behavior is not
task-specific; rather, it reflects a general property of self-
distillation under non-stationary missing-modality streams.
The three heatmaps in Figure 4 show how Fisher param-
eter importance is distributed across LoRA modules under
two adaptation regimes. The left panel showsF anchor, the
Fisher diagonal stored at the moment FAR declared conver-
gence,whereimportanceissharplyconcentratedintheearly
layers of the query and up-projection modules, with the re-
mainderofthelandscapeseveralordersofmagnitudelower;
this sparsity indicates that the student identified a precise,
low-dimensional parameter subspace sufficient to solve the
missing-modality distillation task. The middle panel shows
the equivalent Fisher diagonal accumulated by continuous
TTSDwithoutanystoppingcriterion.Theimportanceland-
scape is substantially more diffuse, and the gradient mass
is spread uniformly across all layers and modules, reflecting
thatcontinuedadaptationbeyondconvergencedilutesthesig-
nalthatwasmeaningfulattrueconvergence.Therightpanel
shows the log-scale difference, which is almost entirely red,
confirmingthatFARanchorsatamomentofhigherandmore
concentratedimportanceacrossmostoftheparameterspace.
4.5 Discussion
Ourexperimentsrevealtwocorechallengesforadaptingau-
toregressive LVLMs under partial modality observation: re-
covering missing semantic content and sustaining effective
adaptationoverlongdeployment.Retrieval-basedadaptationassumestheobservedmodalityprovidesaqueryinformative
enoughtoretrievesemanticallyrelevantreferences.Thisas-
sumptionbreaksdownwhenthemissingmodalitycarriesthe
dominant information, in which case the retrieved examples
supplyinconsistentormisleadingsupervision.Entropymini-
mizationfailsinacomplementaryway.Itimplicitlyassumes
confident predictions are correct predictions, but under sub-
stantialmodalityshift,autoregressiveLVLMsoftengenerate
overconfident, semantically incorrect outputs. Minimizing
entropyinthisregimereinforcestheseerrors,producingop-
timization drift rather than genuine adaptation, consistent
with the findings of Hu et al. (2025).
TTSD-FAR addresses both failure modes by treating
missing-modality adaptation as a conditional representation
alignment problem rather than a confidence or retrieval op-
timizationproblem.Thestudentisguidedtowardthefrozen
reference model’s complete-modality latent representation
throughfeature-levelsupervision,recoveringsemanticinfor-
mationthattheobservedmodalityalonecannotprovideand
yielding substantially more robust adaptation under severe
degradation. A separate issue is that continued optimization
pastconvergencecausesparameterdiffusion,erodinguseful
representations.FAR’sadaptiveconsolidationaddressesthis
by tracking optimization stationarity via Fisher geometry to
detect convergence, then reactivating adaptation only when
deviations from the consolidated geometry signal genuine
distributional drift. This preserves learned representations
while retaining the capacity to adapt when the input distri-
bution actually shifts.
5 Conclusion
TTSD-FAR is a parameter-efficient framework for adapt-
ing large video-language models to missing modalities dur-
ing inference. Missing modalities induce substantial shifts
in representation geometry within autoregressive LVLMs,
a regime in which entropy-based test-time adaptation and
retrieval-augmented generation both fail to provide reliable
supervision. TTSD-FAR instead couples a frozen teacher,
trained on complete modalities, with an adaptive student
adapter that operates on masked inputs, aligning student
and teacher representations via feature-level self-distillation
without modifying the base model’s parameters. Fisher-
Anchored Restoration governs this process through a cycle
of consolidation and reactivation: adaptation is suspended
once the student’s parameter importance landscape stabi-
lizes, and resumed only when the Fisher geometry signals
genuine distributional drift, preventing the parameter diffu-
sionthatunboundedadaptationotherwiseinduces.Extensive
experiments across three emotion recognition benchmarks
show that TTSD-FAR consistently improves performance
under severe missing-modality conditions while preserving
accuracy competitive with the complete-modality setting.
Supplementarymaterials.Includesproofsketches,dataset
description and implementation details, algorithm for the
proposedTTSD-FARmethod,andextendedablationstudies.
A Appendix
Algorithm for TTSD-FAR............................8

Theoretical Properties of FAR........................8
Prop. 1: Fisher Stability as a Stationarity Signal.... 8
Proposition 2: No-Regret Reactivation............. 9
Proposition 3: FMI Hard-Sample Robustness.......9
Additional Results..................................10
Distilling from Base Teacher (no teacher LoRA) ..... 10
Weaker Modality Missing Results...................10
Additional Ablations............................... 11
LoRA Rank Sensitivity............................11
Effectivity of TTSD on Small Video Language Model 11
FAR Hyperparameters Sensitivity...................11
Effect of the FMI Threshold......................11
FAR State Dynamics over the Test Stream..........11
Loss-Based Stopping vs. FAR......................12
Datasets and Implementation Details................12
Datasets..........................................12
Implementation Details............................13
FAR Parameters................................ 14
B Algorithm for TTSD-FAR
Algorithm 1 summarizes the adaptation loop for the LVLM
using the proposed TTSD-FAR methodology.
Algorithm 1: TTSD-FAR
Require:∆θtea,ϕ0,η,β F,τfc,K,τfmi,M,ε
1:ϕ←ϕ 0;F, Fprev←0 p;state←Active;c←0;B ← ∅
2:foreach incoming sample˜xdo
3:ifstate=Activethen
4:htea
l ←sg(LLMθ0+∆θtea
l(x));hstu
l ←
LLMθ0+∆θstu
l(˜x)
5:g← ∇ ϕP
lαl∥hstu
l−htea
l∥2
2;ϕ←ϕ−ηg
6:F←β FFprev+ (1−β F)g⊙2
7:CI← ∥F−F prev∥1/(∥F∥ 1+ε);F prev←F
8:c←c+ 1ifCI< τ fc, elsec←0
9:ifc≥Kthen
10:ϕ anchor, Fanchor←ϕ, F; state←Anchored;
B ← ∅
11:end if
12:else
13:Infer with frozenϕ anchor{no update}
14:if˜xis a complete-modality samplethen
15:Mask˜xsynthetically→˜x′;g←
∇ϕLdistill(˜x′)|ϕanchor
16:FMI←1
pP
jg2
j/(Fanchor,j +ε);pushtoB(keep
lastM)
17:ifFMI> τ fmithen
18:ϕ, F prev←ϕanchor, Fanchor;c←0; state←
Active;B ← ∅
19:end if
20:end if
21:end if
22:end forB.1 Theoretical Properties of FAR
This section provides an intuitive theoretical justification
for the proposed Fisher-Anchored Restoration (FAR) mech-
anism. Rather than proving global optimality, our goal is
to explain why the proposed two-state adaptation strategy
avoidsunnecessaryparameterdriftwhileremainingcapable
of re-adapting under distribution shift.
AssumptionsThe following assumptions are standard in
stochastic optimization and test-time adaptation.
Assumption 1 (Local Smoothness).The feature-level dis-
tillation objectiveL distill(ϕ)is continuously differentiable
withL-Lipschitz gradients,
∥∇L(ϕ 1)− ∇L(ϕ 2)∥ ≤L∥ϕ 1−ϕ2∥.(16)
Assumption2(BoundedGradientVariance).Thestochas-
tic gradients satisfy
E[gt] =∇L(ϕ t),(17)
and
E
∥gt− ∇L(ϕ t)∥2
≤σ2.(18)
Assumption3(PiecewiseStationaryTestStream).Thetest
distribution remains stationary between distribution shifts.
This assumption is common in continual test-time adapta-
tion, where environmental conditions remain approximately
constant for finite intervals before changing.
Proposition 1: Fisher Stability as a Stationarity Signal.
LetF tdenotetheonlineFisherdiagonaldefinedinEq.(12),
Ft=βFFt−1+ (1−β F)g⊙2
t,(19)
whereg tis the stochastic gradient ofL distillat stept, taken
coordinate-wisesothatg⊙2
tdenoteselement-wisesquaring.
Under Assumption 2 (bounded gradient variance) and As-
sumption 3 (piecewise stationarity), suppose
CIt=∥Ft−Ft−1∥1
∥Ft∥1+ε< τfc (20)
holds forKconsecutive adaptation steps. Then the second
moment of the stochastic gradient,E[g⊙2
t], has stabilized
across those steps, in the sense of Eq. (9) below. This need
not imply that∇L(ϕ t)≈0.
Proof Sketch:From the recursion definingF t,
Ft−Ft−1= (1−β F) 
g⊙2
t−Ft−1
.(21)
Substituting into the definition ofCI t,
CIt=(1−β F)∥g⊙2
t−Ft−1∥1
∥Ft∥1+ε.(22)
Sinceβ F∈(0,1)isfixed,CI t< τfcisequivalent,uptothe
constant(1−β F), to
∥g⊙2
t−Ft−1∥1<τfc
1−β F(∥Ft∥1+ε),(23)
i.e., the current squared gradient agrees with the running
Fisher estimate in aggregateℓ 1mass. Taking expectations
under Assumption 2, which givesE[g t] =∇L(ϕ t)and

E∥gt−∇L(ϕ t)∥2≤σ2,theper-coordinatesecondmoment
satisfies
E[(g t,j)2] = (∇ jL(ϕt))2+ Var(g t,j).(24)
Condition(5)smallforKconsecutivestepsthereforeforces
E[(g t,j)2]≈E[(g t−1,j)2]≈ ··· ≈E[(g t−K,j )2]∀j,
(25)
which holds under either of two disjoint cases:
(a) stationary point:∇L(ϕ t)≈0,(26)
Var(g t,j)≈σ2
j,(27)
(b) constant residual:∇L(ϕ t)≈c j̸= 0,(28)
Var(g t,j)≈σ2
j−c2
j(unchanged).
(29)
Equation(7)alonecannotdistinguishcase(a)fromcase(b):
both produce a stableF t. What Eq. (7) does establish is that
the distribution generatingg t, and in particular its second
moment, has stopped changing over theK-step window.
Remark(Informal):ConsolidationReducesUnnecessary
Drift.GivenEq.(7),considerTadditionalstepstakenwhile
CItremainsbelowτ fc.Writingϕ t+T−ϕt=−ηPT
i=1gt+i
and applying Assumption 2,
E
∥ϕt+T−ϕt∥2
=η2TX
i=1E
∥gt+i∥2
=O(Tη2σ2),
(30)
matchingEq.(5),regardlessofwhethercase(a)or(b)above
holds. In case (a), this accumulation is pure noise with no
accompanyingdecreaseinL distill;incase(b),usefulprogress
−ηTcis being made, but at rate bounded byc, the residual
gradient, which Eq. (7) shows is itself no longer detectably
shrinking. In either case, continued updates trade a fixed,
boundableamountofprogressagainstunbounded,monoton-
icallygrowingvarianceasTincreases.FARoperationalizes
theresultingtradeoffbyfreezingϕonceEq.(2)holdsforK
consecutivesteps,acceptingthesmallresidual-progresscost
ofcase(b)inexchangeforeliminatingthevariancegrowthin
Eq. (9) until Eq. (15) signals that reactivation is warranted.
Proposition 2: No-Regret Reactivation
AssumethetestdistributionchangesfromP AtoPB,andthe
induced gradient statistics satisfy
FMI t> τfmi.(31)
Then FAR exits the anchored state and resumes gradient
updates. Following reactivation, the optimization dynamics
are identical to continuous TTSD adaptation up to a finite
detection delay.
ProofSketch:Duringtheanchoredstate,thestudentparam-
eters remain fixed,
ϕt=ϕanchor .(32)
For each incoming complete-modality sample, FAR eval-
uatesFMI t=1
ppX
j=1g2
t,j
Fanchor,j +ϵ.(33)
When the test distribution remains stationary, the incom-
ing gradients remain aligned with the previously consoli-
datedFishergeometry,resultinginrelativelysmallmismatch
values that remain below the reactivation threshold.
Suppose the underlying distribution changes. The gradi-
ents then begin appearing along parameter directions that
previously exhibited low Fisher importance. Consequently,
g2
t,j≫F anchor,j ,(34)
for a subset of parameters, causing the Fisher mismatch
index to exceedτ fmi.
Once reactivated, FAR performs the same parameter up-
date as standard TTSD,
ϕt+1=ϕt−ηg t.(35)
Therefore,afterdetectingadistributionshift,FARfollows
thesameoptimizationtrajectoryasperpetualadaptation.The
only difference is that FAR avoids unnecessary updates dur-
ingstationaryperiods,therebyreducingparameterdriftwith-
out sacrificing future adaptability.
Proposition 3: FMI Hard-Sample Robustness.
Suppose the student LoRA is in the anchored state with
stored baselineF anchor≈F∗(P) :=E P[g⊙g]. Consider
twoscenariosthatproduceequallyelevateddistillationloss:
EP′[Ldistill] =C·E P[Ldistill],(36)
wherescenario(a)isahardwithin-distributionsamplex H∼
Pwith∥g H∥2=C·E P[∥g∥2],andscenario(b)isashifted
distributionP′withthesameexpectedlosselevation.Under
Assumptions 2–3, the FMI satisfies
EP[FMI(x H)]≈1,(37)
regardless ofC, while
EP′[FMI t]̸= 1,(38)
wheneverF∗(P′)̸=F∗(P).Consequently,theFMIthresh-
oldτ fmiis triggered selectively by genuine distributional
shift and not by gradient magnitude alone.
Proof Sketch:By definition of the diagonal Fisher,
F∗(P)j=EP[g2
j].Forahardsamplex H∼P,thegradient
gHis drawn from the same distributionPused to construct
Fanchor, soE P[g2
H,j] =F∗(P)j=Fanchor,jcoordinate-wise
— the magnitudeCscales the gradient but is distributed
across the same high-importance coordinates that defined
Fanchorin the first place, leaving the ratio unchanged:
EP[FMI(x H)] =1
ppX
j=1Fanchor,j
Fanchor,j +ε≈1.(39)
UnderashifteddistributionP′,gradientsinsteadconcentrate
in coordinatesj∈SwhereF anchor,jwas small, sog2
t,j≫
Fanchor,jforj∈S, giving
EP′[FMI t]≥1 +|S|
p·δ
flow+ε,(40)

LVLM
Student 
LoRABase Weights
𝞱
Mask
෤𝑥
Modality 
Router
ℎ𝑙𝑠𝑡𝑢−ℎ𝑙𝑡𝑒𝑎
22ℒ𝑑𝑖𝑠𝑡𝑖𝑙𝑙
ACTIVE
ANCHORED
𝐶𝐼𝑡<𝜏𝑓𝑐𝐹𝑀𝐼𝑡>𝜏𝑓𝑚𝑖FAR𝑥=(𝑥𝑡,𝑥𝑣)
෤𝑥∈(𝑚𝑡,𝑥𝑣
𝑥𝑡,𝑚𝑣)hltea
+
layer i∆𝞱stuhlstuFigure5:TTSD-FARwithoutteacherLoRA.Baseweightsprovide
teacher representations.
Table 4: Comparison of methods under progressively miss-
ing text conditions. Distilling from the Base model, i.e., no
teacher LoRA. We report WF1 on the BAH dataset as the
proportion of unavailable textual input increases.
Method Full 10% 20% 30% 40% 50%
No Adaptation0.63410.5905 0.5798 0.5710 0.5662 0.5614
TENT – 0.4910 0.4840 0.4733 0.4501 0.4453
EATA – 0.5465 0.5367 0.5290 0.5245 0.5110
RAG – 0.5770 0.5720 0.5540 0.5448 0.5250
Perplexity Gen. – 0.5790 0.5695 0.5648 0.5488 0.5430
TTSD-FAR (Ours)–0.6185 0.6120 0.5955 0.5910 0.5870
whereδ >0is the shift magnitude inSandf low=
max j∈SFanchor,j. Asf low→0, this bound diverges, so
anyfiniteτ fmieventuallydetectstheshift.Sampledifficulty
withinP(largeC)thereforeleavesFMInear1,whilestruc-
turalchangeinwheregradientmassconcentrates(agenuine
distributional shift) drives it above threshold.
C Additional Results
Distilling from Base Teacher (no teacher LoRA).
WhenworkingwithLLMsandLVLMs,itisoftenrealisticto
assume that no retraining, either full or parameter-efficient,
is performed on the base model. To demonstrate the effec-
tivenessofTTSDunderthisconstraint,weconductanexper-
iment where distillation is performed directly from the base
model weights without any teacher LoRA as shown in Fig.
5. In this setting, teacher representations are extracted from
the frozen base model, resulting in the same optimization
problem as Eq. 11
Table 4showstheresultsforthecasewheretheteacherem-
beddingsareobtaineddirectlyfromthebasemodelweights.
Since the same distillation mechanism is applied, the setup
iseffectivelyidenticalexceptthattheteacherrepresentations
are extracted from the frozen base model rather than from
a Teacher LoRA trained with the objective in Eq. 11. The
results therefore follow a similar trend to the case where the
teacher embeddings are obtained using the Teacher LoRA
objective.Weaker-Modality Missing Results
DFEW - Text Missing at Test-Time - Weaker Modality
MissingTable5reportstheperformanceofdifferentmeth-
ods on the DFEW under progressively increasing levels of
missing textual modality. As the proportion of missing text
increases, the performance of the model without adaptation
gradually degrades.
Table5:Comparisonofmethodsunderprogressivelymissing
textual modality conditions. F1 on the DFEW dataset as the
proportion of missing textual input increases.
Method Full 10% 20% 30% 40% 50%
No Adaptation0.5549 0.5530 0.5490 0.5365 0.5260 0.5235
TENT – 0.1655 0.1725 0.1790 0.1935 0.1833
EATA – 0.2635 0.2674 0.2835 0.2855 0.2943
RAG – 0.5530 0.5425 0.5265 0.5265 0.5210
Perplexity Gen. – 0.5535 0.5460 0.5350 0.5300 0.5210
TTSD-FAR (Ours)– 0.5540 0.5510 0.5444 0.5376 0.5278
However,theperformancedropduetothemissingmodal-
ity is insignificant compared to the vision missing in the
DFEW dataset. This is primarily due to the dataset struc-
ture. The DFEW dataset comprises of 16372 short clips,
of which 6196 are non-English transcriptions and 4960
clip have no speaker utterance at all. This shows that even
the full modality performance (0.5549) is already less re-
liant on the textual modality. Classical test-time adapta-
tion methods such as TENT (Wang et al. 2021) and EATA
(Niu et al. 2022) exhibit severe performance drops, sug-
gesting that entropy-minimization-based adaptation strate-
gies are not well suited for multimodal missing-modality
scenarios. In contrast, retrieval-based augmentation (RAG)
and generation-based approaches maintain performance be-
cause the observed modality is stronger in this case. Our
proposedTTSDmethod consistently achieves the best per-
formance under moderate to severe missing-text conditions
(20%–50%), demonstrating improved robustness and stabil-
ity when textual inputs are partially unavailable. These re-
sults indicate that TTSD effectively mitigates the negative
impact of missing textual modality while preserving strong
performance as modality degradation increases.
BAH - Vision Missing at Test-Time- Weaker Modality
MissingTable6reportsresultsonBAHundervisionmiss-
ing conditions. The vision drop for the BAH dataset results
in negligible performance drops even at severe missing per-
centages (50%). The 10% has no effect on the performance.
TTSD is able to improve over the existing TTA methods in
both weaker and stronger missing modality conditions.
As shown in Table 9, the textual input is critical for Am-
bivalencedetection.Thisexplainswhythetext-missingcase
for the BAH dataset results in a much higher performance
drop than vision-missing, as shown in Table 6. Another im-
portant observation is that in the vision missing case (Table
6)theRAG-basedmethodisalsoabletoimprove,unlikethe
textmissingcase.Thisisbecauseinthevisionmissingcase,
the observed modality (text) is stronger and is able to effec-
tivelyretrieverelevantsamples.Whereas,inthetextmissing

Table6:Comparisonofmethodsunderprogressivelymissing
visualmodality.WereportAvg-F1ontheBAHdatasetasthe
proportion of unavailable visual input increases.
Method Full 10% 20% 30% 40% 50%
No Adaptation0.7142 0.7142 0.7095 0.7046 0.6950 0.6923
TENT – 0.4834 0.4846 0.4790 0.4645 0.4754
EATA – 0.5655 0.5428 0.5165 0.5095 0.5052
RAG – 0.7100 0.7055 0.7059 0.7110 0.6975
TTSD-FAR (Ours)– 0.7142 0.7125 0.7120 0.7125 0.7125
case, the observed modality (vision) was not discriminant
enough for effective retrieval.
D Additional Ablations:
LoRA Rank Sensitivity.We investigate the effect of
LoRA rank on TTSD’s robustness to modality-incomplete
inputs. As expected, performance degrades monotonically
withincreasingmissingratesacrossallranks;yet,thedegra-
dation pattern reveals a clear sensitivity to rank capacity.
Lower-rank adaptation (r=4) suffers disproportionately un-
der higher missing rates, exhibiting steeper and less stable
declinescomparedtohigherrank,suggestingthatinsufficient
parameter capacity limits the model’s ability to compensate
for missing modalities. Ranksr=8 andr=16 remain con-
sistently close throughout, with only marginal gains from
doubling the rank, indicating diminishing returns beyond
r=8. These results suggest thatr=8 strikes the best balance
between adaptability, robustness, and computational com-
plexity. The ablation shows that LoRA rank is a meaningful
factor in missing-modality robustness.
10 20 30 40 50
T ext Missing Rate (%)0.400.450.500.550.60 F1 Score
MELD
r = 4
r = 8
r = 16
10 20 30 40 50
Vision Missing Rate (%)0.400.450.500.550.60 F1 Score
DFEW
r = 4
r = 8
r = 16
10 20 30 40 50
T ext Missing Rate (%)0.4500.4750.5000.5250.5500.5750.6000.6250.650 F1 Score
BAH
r = 4
r = 8
r = 16
Figure 6: Impact of LoRA rank on distillation performance
under varying missing-modality rates across three datasets.
Effectivity of TTSD on Small Video Language Model
(SVLM):Tofurthershowtheeffectivenessoftheproposed
TTSD method. We perform the experiment on a smaller
videolanguagemodel,MobileVideoGPT-0.5B(Shakeretal.
2025).
ForVideo-LLaVA-7B,thebaselinemodelwithoutadapta-
tionshowsagradualdegradationinperformanceasthepro-
portion of missing textual input increases. Applying TTSD
consistentlyimprovestheresultsacrossallmissing-modality
levels, demonstrating that the proposed method effectively
recoversusefulrepresentationswhentextualinformationbe-
comes partially unavailable. A similar trend is observed forthesmallerMobileVideoGPT-0.5Bmodel.Althoughitsover-
all performance is lower due to its reduced model capac-
ity, TTSD still provides consistent improvements over the
non-adaptedbaselineunderallmissing-modalityconditions.
These results indicate that TTSD is not restricted to a spe-
cificLVLMarchitectureandcanimproverobustnessforboth
large and lightweight video-language models.
D.1 FAR Hyperparameters Sensitivity
Figure 7:Effect ofτ fmion reactivation events. The three panels
showtheFMItraceatτ fmi1.0(top),2.0(middle)and,3.0(bottom).
EffectoftheFMIThreshold:Figure7illustratesthesen-
sitivityofFAR’sreactivationcriteriontothechoiceofτ fmi.
Each panel plots the FMI trace computed during the AN-
CHORED state, with the horizontal dashed line indicating
τfmiand vertical dashed lines marking each detected reac-
tivation event. Atτ fmi= 1.0(top), the threshold is low
enough that it is crossed both by the largest, genuine spikes
andbyseveralsmallerfluctuationsinthe1.0–1.5range,yield-
ingnineseparatereactivations.Thisdemonstratesthefailure
mode of an overly permissive threshold, where the mecha-
nism reacts to noise as frequently as to true distributional
shift. Atτ fmi= 2.0(middle), corresponding to the value
used in our reported experiments, the threshold clears the
minorfluctuations,whilestilldetectingthetwoclearlydom-
inant spikes, resulting in exactly two reactivations. This set-
ting represents the balance adopted for the main results. At
τfmi= 3.0(bottom),thethresholdisraisedfurthersuchthat
even a second sizeable spike is no longer detected, leaving
onlythesinglelargesteventtotriggerreactivation.Thisillus-
tratestheoppositefailuremode:anexcessivelyconservative
threshold trades false positives for false negatives, risking
missed detections of genuine distributional shift.
FAR State Dynamics over the Test StreamFigure 8
shows the ACTIVE/ANCHORED state trajectory of FAR
across the test stream under three hyperparameter regimes.
The middle panel corresponds to the values reported in the
paper and supplementary (β F= 0.99,τ fc= 0.02,K= 5,

Table 7: F1 Score on the BAH dataset under progressively missing textual input for two models: Video-LLaVA-7B and
MobileVideoGPT-0.5B.
Model Method Complete 10% 20% 30% 40% 50%
Video-LLaVA-7BNo Adaptation0.63410.5905 0.5798 0.5710 0.5662 0.5614
TTSD-FAR (Ours) 0.6185 0.6120 0.5955 0.5910 0.5870
MobileVideoGPT-0.5BNo Adaptation0.48890.4521 0.4413 0.4310 0.4218 0.4125
TTSD-FAR (Ours) 0.4685 0.4595 0.4510 0.4428 0.4385
τfmi= 2.0). The student remains ACTIVE for roughly the
first 40% of the stream before the Consolidation Index stays
belowτ fcforKconsecutive samples, triggering consolida-
tion into ANCHORED. It remains frozen until a distribu-
tionalshiftshortlyafter,aroundthemid-pointofthestream,
raisestheFMIaboveτ fmi,triggeringreactivation.Asecond
cyclefollowslaterinthestream,consolidatingandthenreac-
tivating again, after which the student remains active for the
remainder until the very end, where it transitions to the An-
chored state. This produces two well-separated reactivation
cycles.
Figure 8:Timeline of FAR states (active-anchored) across the
test stream.
Thetoppanelshowsasingleconsolidationslightlypastthe
mid-pointofthestream,withnoreactivationfortheremain-
der. This is consistent with a substantially higherτ fmi= 5,
underwhichlatershiftsinthestreamnolongerraisetheFMI
enough to cross the threshold, leaving the student perma-
nentlyfrozenonceanchored.Thebottompanelusesarelaxed
consolidation requirement with a lower reactivation thresh-
old ofτ fmi= 0.5. Consolidation fires as soon as CI drops
belowτ fc, and the lowτ fmicauses ordinary gradient noise
whileanchoredtobesufficienttotriggerreactivation.These
settings produce rapid, near-continuous alternation between
ACTIVE and ANCHORED throughout most of the stream,
resulting in the student being in the ANCHORED state for
roughly80%oftheteststream.Thesetwoextremesmotivate
the reported configuration ofK= 5andτ fmi= 2.0, which
yields the stable, interpretable cycles shown in the middle
panel rather than premature permanent freezing or unstable
oscillation.D.2 Loss-Based Stopping vs. FAR
Table8comparesTTSD-FARagainsttwoablatedvariantson
the BAH dataset (with settings similar to Tab. 4) under pro-
gressivelymissingtextualinput.i)plainTTSD,whichadapts
oneverycomplete-modalitysamplewithoutanystoppingcri-
terion,andii)TTSD-LS,whichinsteadhaltsadaptationonce
the raw distillation loss plateaus with a patience of 5, after
which the student remains frozen for the rest of the stream.
TTSD-LSunderperformsnotonlyTTSD-FARbutalsoplain,
unbounded TTSD at every missing ratio. This confirms the
motivationgiveninSection3.4:therawdistillationlossisan
unreliablestoppingsignal,sinceitcanplateauduetosample
difficulty rather than genuine parameter convergence. Once
frozen, TTSD-LS has no mechanism to detect that the test
streamhassinceshiftedandneverresumesadaptation.Itac-
cumulates the disadvantages of stopping without any of the
drift-awareness that FAR provides. In contrast, TTSD-FAR
consistently outperforms both plain TTSD and TTSD-LS
across all missing ratios, indicating that gating adaptation
on the Fisher information geometry rather than on the loss
itself is necessary to obtain the benefits of stopping without
sacrificing the ability to resume adaptation when genuine
distributional shift occurs.
Table8:AblationcomparingTTSD-FARagainstplainTTSD
(nostoppingcriterion)andTTSD-LS(loss-plateaustopping
criterion) under progressively missing text conditions. We
reportWF1ontheBAHdatasetastheproportionofunavail-
able textual input increases.
Method Full 10% 20% 30% 40% 50%
No-Adaptation 0.6341 0.5905 0.5798 0.5710 0.5662 0.5614
TTSD – 0.6065 0.6039 0.5887 0.5816 0.5728
TTSD-LS – 0.5910 0.5820 0.5730 0.5645 0.5560
TTSD-FAR (Ours)–0.6185 0.6120 0.5955 0.5910 0.5870
E Datasets and Implementation Details
Datasets
MELD (Multimodal EmotionLines Dataset):MELD is
a conversational emotion recognition dataset that extends
theoriginalEmotionLinescorpusbyincorporatingsynchro-
nizedaudio,visual,andtextualmodalities(Poriaetal.2019).
It contains approximately 1,433 multi-party dialogues and
over13,000utterancesextractedfromtheFriendsTVseries,
each labeled with one of seven discrete emotions (Anger,

Disgust, Sadness, Joy, Neutral, Surprise, Fear) and senti-
mentlabels.Themultimodalnatureandconversationalcon-
text make MELD a challenging benchmark for models that
must integrate semantic, acoustic, and facial cues to infer
emotion.
DFEW(DynamicFacialExpressionintheWild):DFEW
is a large-scale dynamic facial expression dataset collected
from more than 1,500 movies and comprising over 16,000
short video clips depicting naturalistic expressions captured
under unconstrained conditions (Jiang et al. 2020). Unlike
posedorlaboratory-controlleddatasets,DFEWincludessig-
nificantvariationsinpose,illumination,occlusion,andactor
demographics,witheachclipannotatedforoneofsevenba-
sicemotions.Thisdatasetservesasabenchmarkfordynamic
facialexpressionrecognition(DFER)inthewild,emphasiz-
ing robustness to real-world noise and temporal dynamics.
BAH (Behavioral Ambivalence/Hesitancy Dataset):The
BAH dataset is a recently introduced benchmark for recog-
nizingambivalenceandhesitancy(A/H)invideorecordings
designed for behaviour change studies (González-González
etal.2026).Itcontains1,118videosrecordedfrom224par-
ticipants across diverse demographics, captured while par-
ticipantsrespondedtostimuliintendedtoelicitambivalence
or hesitancy. Frame- and video-level annotations highlight
segmentscontainingA/Hcues,andthedatasetalsoprovides
aligned face crops, audio transcripts with timestamps, and
participant metadata. BAH’s focus on subtle and conflicting
emotional states presents a distinct challenge compared to
conventional discrete emotion recognition tasks, highlight-
ingtheneedformodelsthatcancapturefine-grainedbehav-
ioralcuesacrossmodalitiesandareabletoperformwelleven
under missing conditions. The dataset comes with a prede-
finedtrain-testsplit.Wepresenttheresultsontheofficialtest
set. The performance measure reported is the F1 score.
Implementation Details
Missing modality Simulation:This section provides the
details of the missing modality simulation for both vision
and text missing conditions. The missing modality problem
is a vast paradigm, and it is important to set the scope of
the experimentation to effectively validate the method. The
datasets used for the validation of the TTSD-FAR method
all contain utterance-level annotations. For the text-missing
scenario, the speaker utterance in the prompt is replaced by
an empty string, and for the vision-missing scenario, zero-
imputed frames tensor is fed into the model. Figure 9 shows
thevision-missingandtext-missingscenariosfortheMELD
dataset.
MELD Dataset:We evaluate on the MELD test split us-
ingutterance-levelmetadataprovidingdialogueID,speaker,
transcription, and 7-class emotion labels (neutral, surprise,
fear, sadness, joy, disgust, anger). Visual inputs are con-
structed from pre-extracted cropped and aligned facial
frames, resized to224×224and uniformly sampled to
T= 8, frames per utterance. Textual inputs are the raw
utterance transcriptions from the metadata. Missing modal-
ity scenarios are simulated by replacing the utterance string
with an empty string for text-missing conditions, controlled
by a seeded random state (seed 42) for reproducibility. The
Vision Missing
Text MissingCompleteVision MissingFigure 9: Illustration of the missing modality
model is prompted in instruction-format with the video and
utterance as inputs, and asked to classify the emotion from
the seven candidate labels. Student LoRA is adapted using
the Adam optimizer with a learning rate10−4with a batch
sizeof1,wheregradientaccumulationover8stepsisequiv-
alent to performing a single effective update per sample,
updatingonlythestudentLoRAparametersr= 8,α= 16,
and dropout 0.3.DFEW Dataset:Since Video-LLaVA-7B
is used in all experiments, the data loading and prompt-
ing structure remains the same as MELD. Student LoRA is
adaptedusingtheAdamoptimizerwithalearningrate10−4
with a batch size of 1, where gradient accumulation over 8
stepsisequivalenttoperformingasingleeffectiveupdateper
sample, updating only the student LoRA parametersr= 8
,α= 16, and dropout 0.3. All experiments use float16 pre-
cision. The results are reported on the Official Fold-1 test
set.
BAH Dataset:Similar to DFEW and MELD datasets, we
performtheexperimentsfortheBAHdatasetusingtheVide-
oLlavamodel.However,thetaskofAmbivalence/Hesitancy
detection is more complex than basic emotion recognition.
Following Gonzaleset al.(González-González et al. 2026),
we use the prompting structure shown in Table 9 to get the
bestresults.Table 9showshighrelianceonthetextualinput.
Adding an accurate description of the task improves perfor-
mance, and the best results are achieved with the definition
and speaker utterance.
StudentLoRAisadaptedusingtheAdamoptimizerwitha
learningrate10−4withabatchsizeof1,wheregradientac-
cumulation over 8 steps is equivalent to performing a single
effectiveupdatepersample,updatingonlythestudentLoRA
parametersr= 8,α= 16,anddropout0.3.Allexperiments
usefloat16precision.Theresultsarereportedontheofficial
test set.
Compute details:For all experiments, the student is being
alignedtotheteacher’shiddenrepresentations,notitsoutput
distribution. We therefore fix generation to greedy decoding
(do_sample=False) so that final predictions are determinis-
tic given a fixed adapter state. Experiments were run on a
machine with 4×NVIDIA A100-SXM4-40GB GPUsand
CUDA version 12.8. The exact libraries and corresponding
versions are included in the code supplement.
FAR Parameters:β F,τfc,K, andτ fmitogether control
theACTIVE/ANCHOREDstatemachine.β FsetstheEMA

Table 9: Summary of prompt variations for zero-shot inference with corresponding video-level F1 scores.
Prompt Type Prompt Avg F1
Simple ClassifytheemotioninthevideoaseitherNon-AmbivalentorAmbivalent.Respondwithonlyoneword. 0.2827
Definition 1 Definition: Ambivalence is the state of having contradictory or conflicting feelings or attitudes towards
something or someone simultaneously. Classify the emotion in the video as eitherNon-Ambivalentor
Ambivalent. Respond with only one word.0.3326
Definition 2 Definition: Ambivalence and hesitancy is understood as the simultaneous experience of desires for
change and against change. Classify the emotion in the video as eitherNon-AmbivalentorAmbivalent.
Respond with only one word.0.3772
Transcript + Def
1Video transcript: {transcript}. Definition: Ambivalence is the state of having contradictory or
conflicting feelings or attitudes towards something or someone simultaneously. Classify the emotion in
the video as eitherNon-AmbivalentorAmbivalent. Respond with only one word.0.6341
Transcript + Def
2Video transcript: {transcript}. Definition: Ambivalence and hesitancy are understood as the
simultaneous experience of desires for change and against change. Classify the emotion in the video as
eitherNon-AmbivalentorAmbivalent. Respond with only one word.0.3945
decayfortheFisherdiagonal.τ fcandKjointlygateconsol-
idation: CI must stay belowτ fcforKconsecutive adapted
samplesbeforethestudentisfrozen.τ fmigatesreactivation,
triggering once FMI exceeds this threshold.Mandεplay a
minor role, affecting only reporting and numerical stability
rather than the transition logic itself.
Table 10: FAR hyperparameters
Parameter ValueDefinition
βF 0.99EMA decay parameter for the on-
line Fisher diagonal
τfc 0.02Consolidation Index threshold
K 5No. of consecutive adaptation
steps required
τfmi 2.0Fisher Mismatch Index threshold
ε 10−8Numerical stability constant
References
Abootorabi, M. M.; et al. 2025. Ask in Any Modality: A
ComprehensiveSurveyonMultimodalRetrieval-Augmented
Generation. InFindings of ACL.
Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H.
2023. Self-RAG: Learning to Retrieve, Generate, and Cri-
tique through Self-Reflection.ICLR, abs/2310.11511.
Chen,J.;Liu,J.;Liu,S.;Zhang,W.;Li,A.;Zhu,E.;andLiu,
X. 2026. Sample-specific Modality Diagnosis and Cross-
modal Enhancement for Incomplete Multimodal Represen-
tations.Proceedings of the AAAI Conference on Artificial
Intelligence, 40(24): 20154–20162.
Cheng, Z.; Cheng, Z.-Q.; He, J.-Y.; Sun, J.; Wang, K.; Lin,
Y.; Lian, Z.; Peng, X.; and Hauptmann, A. 2024. Emotion-
LLaMA: Multimodal Emotion Recognition and Reasoning
with Instruction Tuning. arXiv:2406.11161.
Ge, M.; Tang, D.; and Li, M. 2024. Video Emotion Open-
vocabulary Recognition Based on Multimodal Large Lan-
guage Model. arXiv:2408.11286.González-González, M.; Belharbi, S.; Zeeshan, M. O.;
Sharafi, M.; Aslam, M. H.; Pedersoli, M.; Koerich, A. L.;
Bacon, S. L.; and Granger, E. 2026. BAH Dataset for Am-
bivalence/Hesitancy Recognition in Videos for Digital Be-
havioural Change. InICLR.
Guo, Z.; Jin, T.; and Zhao, Z. 2024. Multimodal Prompt
Learning with Missing Modalities for Sentiment Analysis
and Emotion Recognition. InProc. ACL, 1726–1736.
Havaei,M.;Guizard,N.;Chapados,N.;andBengio,Y.2016.
HeMIS: Hetero-Modal Image Segmentation. InMICCAI.
Hu, J.; Zhang, Z.; Chen, G.; Wen, X.; Shuai, C.; Luo, W.;
Xiao, B.; Li, Y.; and Tan, M. 2025. Test-Time Learning for
Large Language Models. arXiv:2505.20633.
Huang, D.; Li, Q.; Yan, C.; Cheng, Z.; Han, Z.; Huang, Y.;
Li, X.; Li, B.; Wang, X.; Lian, Z.; Cheng, Z.-Q.; and Peng,
X.2025. Emotion-Qwen:AUnifiedFrameworkforEmotion
and Vision Understanding. arXiv:2505.06685.
Jiang, X.; Zong, Y.; Zheng, W.; Tang, C.; Xia, W.; Lu,
C.; and Liu, J. 2020. DFEW: A Large-Scale Database
for Recognizing Dynamic Facial Expressions in the Wild.
arXiv:2008.05924.
Kirkpatrick,J.;Pascanu,R.;Rabinowitz,N.;Veness,J.;Des-
jardins, G.; Rusu, A. A.; Milan, K.; Quan, J.; Ramalho,
T.; Grabska-Barwinska, A.; Hassabis, D.; Clopath, C.; Ku-
maran, D.; and Hadsell, R. 2017. Overcoming catastrophic
forgetting in neural networks.Proceedings of the National
Academy of Sciences, 114(13): 3521–3526.
Li, K.; He, Y.; Wang, Y.; Li, Y.; Wang, W.; Luo, P.; Wang,
Y.; Wang, L.; and Qiao, Y. 2023. VideoChat: Chat-Centric
Video Understanding. InarXiv preprint arXiv:2305.06355.
Lin, B.; Zhu, B.; Ye, Y.; Ning, M.; Jin, P.; and Yuan, L.
2023. Video-LLaVA: Learning United Visual Represen-
tation by Alignment Before Projection. InarXiv preprint
arXiv:2311.10122.
Lin, G.-T.; Huang, W. P.; and Lee, H.-y. 2024. Continual
Test-timeAdaptationforEnd-to-endSpeechRecognitionon
Noisy Speech. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing.

Ma,M.;Ren,J.;Zhao,L.;Tulyakov,S.;Wu,C.;andPeng,X.
2021. SMIL: Multimodal Learning with Severely Missing
Modality. InProceedings of the 35th AAAI Conference on
ArtificialIntelligence(AAAI2021),2302–2310.AAAIPress.
Maaz,M.;Rasheed,H.;Khan,S.;andKhan,F.2024. Video-
ChatGPT:TowardsDetailedVideoUnderstandingviaLarge
VisionandLanguageModels. InKu,L.-W.;Martins,A.;and
Srikumar,V.,eds.,Proceedingsofthe62ndAnnualMeeting
oftheAssociationforComputationalLinguistics(Volume1:
Long Papers), 12585–12602. Bangkok, Thailand: Associa-
tion for Computational Linguistics.
Maheshwari, H.; Liu, Y.-C.; and Kira, Z. 2024. Missing
Modality Robustness in Semi-Supervised Multi-Modal Se-
mantic Segmentation. InProc. IEEE/CVF WACV, 1020–
1030.
Mishra, H. 2026. RDumb++: Drift-Aware Continual Test-
Time Adaptation.arXiv preprint arXiv:2601.15544.
Nado,Z.;etal.2021. EvaluatingPrediction-TimeBatchNor-
malization for Robustness under Covariate Shift.NeurIPS.
Neverova,N.;etal.2016. ModDrop:AdaptiveMulti-Modal
Gesture Recognition. InCVPR.
Nezakati, N.; Reza, M. K.; Patil, A.; Solh, M.; and Asif, S.
2024. MMP: Towards Robust Multi-Modal Learning with
Masked Modality Projection.
Niu, S.; Wu, J.; Zhang, Y.; Chen, Y.; Zheng, S.; Zhao, P.;
and Tan, M. 2022. Efficient Test-Time Model Adaptation
without Forgetting. In Chaudhuri, K.; Jegelka, S.; Song, L.;
Szepesvari, C.; Niu, G.; and Sabato, S., eds.,Proceedings
of the 39th International Conference on Machine Learning,
volume 162 ofProceedings of Machine Learning Research,
16888–16905. PMLR.
Niu, S.; Wu, J.; Zhang, Y.; Wen, Z.; Chen, Y.; Zhao, P.;
and Tan, M. 2023. Towards Stable Test-Time Adaptation in
Dynamic Wild World. arXiv:2302.12400.
Poria, S.; Hazarika, D.; Majumder, N.; Naik, G.; Cambria,
E.; and Mihalcea, R. 2019. MELD: A Multimodal Multi-
Party Dataset for Emotion Recognition in Conversations.
arXiv:1810.02508.
Press, O.; Schneider, S.; Kümmerer, M.; and Bethge, M.
2023. RDumb: A simple approach that questions our
progress in continual test-time adaptation. InAdvances in
NeuralInformationProcessingSystems,volume36,39915–
39935.
Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Lian, D.; Dou, Z.;
and Huang, T. 2025. MemoRAG: Boosting Long Context
Processing with Global Memory-Enhanced Retrieval Aug-
mentation. InProceedings of the ACM on Web Conference
2025, WWW ’25, 2366–2377. New York, NY, USA: Asso-
ciation for Computing Machinery. ISBN 9798400712746.
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.;
Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.;
Krueger, G.; and Sutskever, I. 2021. Learning Transferable
VisualModelsFromNaturalLanguageSupervision. InPro-
ceedings of the 38th International Conference on Machine
Learning (ICML 2021), 8748–8763. PMLR.Ramazanova, M.; Pardo, A.; Ghanem, B.; and Alfarra, M.
2025. Test-TimeAdaptationforCombatingMissingModal-
ities in Egocentric Videos. arXiv:2404.15161.
Saeed, M. S.; Nawaz, S.; Zaheer, M. Z.; Khan, M. H.; Nan-
dakumar,K.;Yousaf,M.H.;Sajjad,H.;Schepper,T.D.;and
Schedl, M. 2024. Modality Invariant Multimodal Learning
to Handle Missing Modalities: A Single-Branch Approach.
arXiv:2408.07445.
Shaker, A.; Maaz, M.; Gou, C.; Rezatofighi, H.; Khan, S.;
andKhan,F.S.2025. Mobile-VideoGPT:FastandAccurate
Video Understanding Language Model. arXiv:2503.21782.
Sun, Y.; et al. 2020. Test-Time Training with Self-
SupervisionforGeneralizationunderDistributionShifts. In
ICML.
Wang, D.; et al. 2021. Tent: Fully Test-Time Adaptation by
Entropy Minimization. InICLR.
Wang, Q.; Fink, O.; Van Gool, L.; and Dai, D. 2022. Con-
tinual Test-Time Domain Adaptation. InCVPR.
Wang,Q.;Zhan,L.;Thompson,P.;andZhou,J.2020. Mul-
timodal learning with incomplete modalities by knowledge
distillation. InProceedingsofthe26thACMSIGKDDInter-
national Conference on Knowledge Discovery & Data Min-
ing, 1828–1838.
Zhang,M.;Levine,S.;andFinn,C.2022.MEMO:TestTime
Robustness via Adaptation and Augmentation. InNeurIPS.
Zhao, J.; Mao, X.; and Chen, L. 2021. Missing Modality
Imagination Network for Emotion Recognition. InAAAI.
Zhao, Q.; Wang, R.; Cen, Y.; Zha, D.; Tan, S.; Dong, Y.;
andTang,J.2024. LongRAG:ADual-PerspectiveRetrieval-
AugmentedGenerationParadigmforLong-ContextQuestion
Answering. InAl-Onaizan,Y.;Bansal,M.;andChen,Y.-N.,
eds.,Proceedingsofthe2024ConferenceonEmpiricalMeth-
odsinNaturalLanguageProcessing,22600–22632.Miami,
Florida, USA: Association for Computational Linguistics.