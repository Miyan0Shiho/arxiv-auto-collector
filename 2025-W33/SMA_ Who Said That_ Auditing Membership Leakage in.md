# SMA: Who Said That? Auditing Membership Leakage in Semi-Black-box RAG Controlling

**Authors**: Shixuan Sun, Siyuan Liang, Ruoyu Chen, Jianjie Huang, Jingzhi Li, Xiaochun Cao

**Published**: 2025-08-12 17:32:24

**PDF URL**: [http://arxiv.org/pdf/2508.09105v2](http://arxiv.org/pdf/2508.09105v2)

## Abstract
Retrieval-Augmented Generation (RAG) and its Multimodal Retrieval-Augmented
Generation (MRAG) significantly improve the knowledge coverage and contextual
understanding of Large Language Models (LLMs) by introducing external knowledge
sources. However, retrieval and multimodal fusion obscure content provenance,
rendering existing membership inference methods unable to reliably attribute
generated outputs to pre-training, external retrieval, or user input, thus
undermining privacy leakage accountability
  To address these challenges, we propose the first Source-aware Membership
Audit (SMA) that enables fine-grained source attribution of generated content
in a semi-black-box setting with retrieval control capabilities. To address the
environmental constraints of semi-black-box auditing, we further design an
attribution estimation mechanism based on zero-order optimization, which
robustly approximates the true influence of input tokens on the output through
large-scale perturbation sampling and ridge regression modeling. In addition,
SMA introduces a cross-modal attribution technique that projects image inputs
into textual descriptions via MLLMs, enabling token-level attribution in the
text modality, which for the first time facilitates membership inference on
image retrieval traces in MRAG systems. This work shifts the focus of
membership inference from 'whether the data has been memorized' to 'where the
content is sourced from', offering a novel perspective for auditing data
provenance in complex generative systems.

## Full Text


<!-- PDF content starts -->

SMA: Who Said That? Auditing Membership
Leakage in Semi-Black-box RAG Controlling
Shixuan Sun
Sun Yat-Sen University
ssun0526@uni.sydney.edu.auSiyuan Liang
Nanyang Technological University
siyuan.liang@ntu.edu.sgRuoyu Chen
University of Chinese Academy of Science
chenruoyu@iie.ac.cn
Jianjie Huang
Zhongguancun Academy
huangjj67@mail2.sysu.edu.cnJingzhi Li
Institute of Information Engineering,
Chinese Academy of Sciences
lijingzhi@iie.ac.cnXiaochun Cao∗
Sun Yat-Sen University
caoxiaochun@mail.sysu.edu.cn
Abstract —Retrieval-Augmented Generation (RAG)
and its Multimodal Retrieval-Augmented Generation
(MRAG) significantly improve the knowledge coverage
and contextual understanding of Large Language Models
(LLMs) by introducing external knowledge sources.
However, retrieval and multimodal fusion obscure content
provenance, rendering existing membership inference
methods unable to reliably attribute generated outputs
to pre-training, external retrieval, or user input, thus
undermining privacy leakage accountability
To address these challenges, we propose the first
Source-aware Membership Audit (SMA) that enables
fine-grained source attribution of generated content in a
semi-black-box setting with retrieval control capabilities.
To address the environmental constraints of semi-black-
box auditing, we further design an attribution estimation
mechanism based on zero-order optimization, which ro-
bustly approximates the true influence of input tokens on
the output through large-scale perturbation sampling and
ridge regression modeling. In addition, SMA introduces
a cross-modal attribution technique that projects image
inputs into textual descriptions via MLLMs, enabling
token-level attribution in the text modality, which for
the first time facilitates membership inference on image
retrieval traces in MRAG systems. Experiments on
multiple textual and multimodal RAG benchmarks show
that SMA outperforms state-of-the-art black-box MIA
baselines in detecting source-specific membership leakage,
with notable improvements in accuracy (+15.74%) and
coverage metrics (+10.01%) under noise and zero-gradient
conditions, demonstrating the effectiveness of attribution-
based strategies in differentiating content origins. This
work shifts the focus of membership inference from
“whether the data has been memorized” to “where the
content is sourced from”, offering a novel perspective for
auditing data provenance in complex generative systems.
I. I NTRODUCTION
Large Language Models (LLMs) and Multi-
modal Large Language Models (MLLMs) have
made significant progress in natural language un-
derstanding and generation tasks. With the rise of
Retrieval-Augmented Generation (RAG) and Multi-
model Retrieval-Augmented Generation (MRAG), mod-
els can dynamically access external knowledge bases
during inference, enabling them to supplement re-
sponses with up-to-date or domain-specific information.While this improves the relevance and accuracy of re-
sponses, it also raises privacy and safety risks [ ?], [73],
[81]–[89]. For example, external sources may contain
sensitive or proprietary data that can be exported by the
model, such as corporate documents, support tickets,
or patient histories [4], leading to accidental content
disclosure.
To cope with the privacy leakage problem in
generative models, Membership Inference Attacks
(MIA) [60] [75] [61] are widely used to identify
whether a particular data instance has been used during
model training. However, in RAG systems, existing
techniques struggle to accurately infer the origin of
generated content and face the following two main
challenges: (1) Destabilized input-output associations.
RAG/MRAG systems dynamically incorporate exter-
nally retrieved content at inference time, which is
concatenated with the original query and passed into
the model to produce responses. This type of input
augmentation and blending disrupts the stable input-
output correspondence that traditional MIA methods
rely on, making it difficult for the attacker to pinpoint
which parts of the output stem from the original query.
(2) Unobservable influence paths due to multimodal
fusion. In MRAG, images are encoded into latent
feature representations before being consumed by the
model, and these are jointly processed with text to guide
generation. Since the attacker cannot directly access
or inspect the retrieved image content, and the model
output integrates signals from multiple modalities, the
contribution path of potential leakage becomes opaque.
To address the difficulty of tracing the origin
of generated content, this paper proposes SMA, the
first source-aware membership inference framework for
RAG and MRAG systems, which, unlike the traditional
MIA approach that only determines “whether or not it
is memorized,” can further identify whether the leaked
content originated from the model’s pre-training corpus
or from external retrieval results. This capability is
important for enabling content traceability and privacy
risk assessment.arXiv:2508.09105v2  [cs.AI]  13 Aug 2025

Fig. 1: An example of LLM/RAG MIA
SMA operates in a semi-black-box environment,
it does not have access to the internal structure of
the model or the gradient information, but it has
control over the switching of the retrieval module. In
realistic deployment scenarios, enabling or disabling
the RAG component often hinges on the application’s
requirements: when response timeliness is crucial, the
retrieval mechanism might be turned off to ensure
faster inference; conversely, when high knowledge
accuracy and confidence in responses are prioritized,
the RAG mode is typically activated to incorporate
external, up-to-date information. However, it is essential
to acknowledge that not all commercial platforms
explicitly expose this toggle functionality—for instance,
Grok currently lacks this direct user-controllable option.
Figure 1 illustrates the overall flow of SMA, which
is designed to capture source bias in the output behavior
by constructing lightweight input perturbations and
toggling the retrieval module on/off, using an inference
mechanism based on attributional discrepancies. Mean-
while, we introduce a zero-gradient scoring strategy to
estimate the influence of input tokens on outputs using
large-scale perturbation sampling with ridge regression
modeling without gradient access. For compatibility
with multimodal inputs, SMA also builds a unified
cross-modal attribution mechanism to transform images
into textual descriptions via MLLM outputs with token-
level attribution in the textual domain, thus supporting
for the first time membership inference of image
retrieval paths in MRAG.
Experiments on multiple textual and multi-model
RAG systems demonstrate that SMA significantly
outperforms state-of-the-art membership inference ap-
proaches in source-specific leakage auditing. SMA
achieves 15.74% improvement in metric ACC and
10.01% in metric Coverage under topkretrieval settings,
confirming its effectiveness in fine-grained provenance
analysis. Our contributions are summarized as follows:
(1) We introduce the source-aware membership audit-
ing problem, which extends traditional membership
inference to differentiate between internal and external
content sources in RAG and MRAG systems. (2) Wepropose SMA, a semi-black-box auditing framework
that performs fine-grained source attribution via in-
put perturbation and a novel zero-gradient scoring
mechanism. (3) Experimental results on multiple RAG
systems show that SMA achieves superior performance
in source leakage auditing, improving ACC by 33.94%
and Coverage by 25.32% over the existing baseline.
II. R ELATED WORK
A. Membership Inference Attacks
MIAs aim to determine whether a given data
point was included in the training set of a target
model by observing its behavior on that point. Early
work by Shokri et al. [26] demonstrated practical
MIAs on classification models via the shadow models
and posterior probabilities [27]. S. Yeom et al. [28]
further showed that only label access can suffice when
combined with loss thresholds and sensitivity measures.
More recently, there has been more attention on
RAG [65] [45], Huang et al. [5] demonstrated that RAG
systems are more likely to output verbatim sensitive
strings from their retrieval databases, especially when
attackers craft adversarial prompts. Zeng et al. [6]
further expanded on this by demonstrating privacy
leakage from real-world datasets such as Enron emails
and medical records, and even proposed automated
prompt injection attacks that can extract raw sensi-
tive content with high precision. Compounding this
issue, optimization-based attacks such as DEAL [7]
have shown that attackers can automatically generate
prompts which can maximize privacy leakage from
RAG systems. Naseh et al. [29] proposed the Inter-
rogation Attack , crafting natural-language queries that
are answerable only if a target document resides in the
RAG vector database, achieving high inference accu-
racy and remaining stealthy against prompt-injection
detectors. Similarly, the RAG-Thief [30] generates
probing prompts to elicit context membership without
triggered safe guarding.
Despite these advances, important gaps remain.
First, no existing work considers MIA against multi-
modal RAG (MRAG) systems combined with MLLMs.

Second, current RAG MIAs cannot accurately attribute
retrieved information to either the external database or
the LLM’s internal knowledge, leading to decreased
accuracy when RAG is enabled on LLMs. Our work
fills these gaps by proposing the first source-aware
MIA for MRAG&MLLM semi-black-boxed system
that both detects image membership in the retrieval
set and discriminates between Retrieved Member and
Pretrained Member.
B. Retrieval-Augmented Generation
RAG enhances a base language model by consult-
ing an external text datastore [63] at inference time.
Given a text query, a dense or sparse retriever ( e.g.,
DPR [31] or BM25 [32] ) fetches the most relevant
requirements, which are then prepended to the query
and fed into a generative model. This design enables
on-demand access to up-to-date or domain-specific
knowledge without retraining or fine-tuning the model.
MRAG extends RAG to incorporate image data
alongside text. In MRAG, two main retrieval modes
coexist, the first is Text-to-Image Retrieval where a
text query is projected into a joint embedding space
(e.g., via CLIP [33] ) to retrieve semantically matching
images. The second is Image-to-Image Retrieval [62]
where a visual query is used to fetch related images that
provide additional context for generation. The selected
images are encoded by a vision encoder and fused with
text embeddings to prompt a multi-model generative
model.
Considering RAG and MRAG rely on external
datastores that can be updated or pruned independently
of the model, they are both attractive targets for
leaking or poisoning sensitive data ( e.g. private medical
records, proprietary corporate documents ). Unlike
monolithic LLMs, where forgetting requires expensive
fine-tuning [64], removing or auditing entries in a
vector database is relatively easy—but often overlooked.
Moreover, an attacker could insert malicious or private
content into the datastore, enabling stealthy membership
inference or data exfiltration attacks [71] [72]. Thus,
we argue that the security of RAG/MRAG knowl-
edge bases—encompassing fine-grained access control,
verifiable deletion logs, and adversarial filtering—is
paramount for safe deployment in sensitive domains.
C. Black-Box Attribution Methods
The attribution methods for generative mod-
els [66] [67] [68] [69] [70] seek to explain which
parts of the input are most responsible for certain
outputs, offering valuable insights into model behavior
without requiring full access to internal parameters.
Early model-agnostic approaches such as LIME [34]
fit simple surrogate models around individual predic-
tions by perturbing inputs and observing changes in
outputs, thereby estimating feature importance through
local linear approximations. SHAP (Shapley additive
explanations) [35] further casts this perturbation-basedreasoning into a principled game-theoretic framework,
attributing output differences to input features via
Shapley values.
In our work, we leverage both black-box attri-
bution techniques to distinguish whether generated
content from an external RAG database or from the
model’s pre-trained parameters. By comparing the
attribution signatures under these two settings, we
are able to infer the token-level source, achieving a
membership inference in both black-box RAG&LLM
and MRAG&MLLM systems.
III. B ACKGROUND
A. Preliminary
Inputs and model definition. To establish the
modeling basis for subsequent attribution mechanisms,
we first define the input format of the respective
LLMs [3] and MLLMs [59] [11] [57].
In a pure text scenario, an LLM receives an input
token sequence xtext= (w1, w2, . . . , w k), where each
token wi∈ V, andVdenotes the model’s vocabulary.
The model generates an output text yfrom this input,
denoted as:
y=f(xtext) (1)
In a multimodal scenario, an MLLM not only
receives text but also visual inputs such as images. The
image Iis converted into a visual token sequence via
a visual encoder ( e.g., ViT ):
vvis= VisionEnc( I). (2)
These visual tokens are then concatenated with
the text tokens xtextto form the input sequence:
xmulti=concat (vvis,xtext) (3)
The corresponding generated output denotes y=
f(xmulti).
Retrieval-augmented generation . To enhance the
knowledge and contextual coverage of generated con-
tent, RAG and its MRAG introduce external retrieval
modules that enable models to dynamically incorporate
additional evidence from textual or visual sources.
We introduce the input modeling of text-based RAG
and image-based MRAG separately, as their retrieval
modalities and token representations differ.
Text-to-Text RAG . Given a sequence of text tokens
xtextas user input, a retriever queries an external
document corpus Dtextto obtain the top- kmost relevant
passages:
xretr={d1, . . . , d k}. (4)
These retrieved passages are appended to the
original input, forming the final input sequence:
xRAG=concat (xtext,xretr). (5)
The model then generates the output based on this
combined context:
y=f(xRAG). (6)

Formally, the model [76] computes the conditional
distribution over the next token ytautoregressively,
conditioned on the previous tokens and the full context:
P(y|xtext,xretr) =TY
t=1P(yt|y1:t−1,xtext,xretr).
(7)
Image-to-Image MRAG . In MRAG settings, the
model receives both an original image Iorigand an
optional text prompt xtext. We use a CLIP-based image
encoder to embed the input image and perform nearest-
neighbor retrieval from an external image corpus,
returning the topkmost similar images {I′
1, . . . ,I′
k}.
Each image is transformed into a sequence of visual
tokens:
vretr_vis={VisionEnc( I′
i)}k
i=1. (8)
The final input to the MLLM is constructed by
concatenating the retrieved visual tokens, the original
image tokens, and the textual prompt:
xMRAG=concat (vretr_vis,vvis,xtext). (9)
Based on this multimodal context, the model
produces the output:
y=f(xMRAG). (10)
B. Threat Model
Audit target systems . We focus on two representa-
tive deployment scenarios as audit targets: RAG&LLM
and MRAG&MLLM systems.
In RAG&LLM systems, the model receives a
text query and augments it with topkretrieved text
passages from an external corpus. These passages are
concatenated with the original input and fed into the
language model for generation.
In MRAG&MLLM systems, the input includes
both an image and optional text. The system retrieves
related images from an external image corpus based
on visual similarity, and encodes both the original and
retrieved images through a modality interface before
passing them into a multimodal language model.
These retrieval-augmented designs enhance re-
sponse quality but also blur the boundary between
model-internal knowledge and external sources, com-
plicating attribution and raising new challenges for
membership auditing.
Auditing goal . Traditional MIA aim to determine
whether a given input sample was included in a model’s
training corpus. However, in RAG and MRAG systems,
generated content may originate from either the model’s
pretraining data or external retrieved content. Therefore,
we extend the auditing objective to a three-way classifi-
cation task: for any input x, determine whether it is: (1)
Pretrained Member : The sample was encountered by
the model during pretraining; (2) Retrieved Member :
The sample exists in the external retrieval databaseand is leaked via RAG/MRAG; (3) Non-Member : The
sample does not appear in either the pretraining corpus
or the external retrieval database. This auditing task
not only determines whether the generated content
constitutes a data leakage event, but also identifies the
leakage pathway— i.e., whether the source is the model
itself or an external resource.
Audit capability settings . We assume the auditor
possesses the following capabilities: (1) Black-box
access to the target system, including the ability to pro-
vide text and image inputs and observe corresponding
outputs; (2) Retrieval control ,i.e., the ability to toggle
RAG or MRAG modules on or off ( e.g., to control
whether retrieval is performed ); (3) Input perturbation ,
i.e., the ability to apply fine-grained modifications to
inputs in order to construct counterfactual examples
that elicit divergent outputs. Without access to model
weights, gradients, or attention mechanisms, we design
an attribution mechanism to determine the provenance
path of the generated content, thereby enabling cross-
modal, multi-source membership auditing.
IV. M ETHOD
A. Problem Definition
In the context of the aforementioned generation
system, the audit issue we are concerned with is not
the traditional question of “whether the output contains
a certain piece of data,” but rather a more fine-grained
determination: given an input query (or its internal key
substrings), does the information originate from the
models’ pretraining corpus, external retrieval repository,
or novel content directly provided by the users’ input?
Specifically, the audit task can be formalized as
follows: Given an input x(comprising text or images),
and its corresponding black-box model fblack (the
LLM/MLLM with or without RAG systems), determine
whether the input falls into one of three categories.
This distinction not only determines “whether leakage
has occurred,” but also further identifies the “leakage
pathway.” We denote this three-way classification task
as:
A(fblack,x)→

Pretrained Member ,
Retrieved Member ,
Non-Member .(11)
Since modern systems are often deployed in RAG
or MRAG configurations with controllable switches, we
assume that auditors possess the following minimum
capabilities: (1) the ability to provide inputs to the
system and observe corresponding outputs (black-box
access); (2) the ability to toggle the retrieval module on
or off; (3) the ability to apply controlled perturbations
to the input to construct counterfactuals ( e.g., key
substring substitution, Gaussian noise perturbation on
images, etc. ). The objective is to determine, under
this setting, whether the model’s response is based on
internally acquired knowledge or externally retrieved

content solely through input-output interaction, thereby
enabling attribution analysis of the membership rela-
tionship between the model’s pretraining corpus and
retrieval sources.
B. Input Perturbation Design
To induce observable changes in the model’s
generative behavior, we first apply controlled per-
turbations to the original input. These perturbations
specifically target key semantic components within
the input, aiming for “light perturbations with high
responsiveness,” i.e., inducing significant differences in
whether the model relies on external retrieval content
(RAG) or its own pre-trained knowledge (LLM) without
compromising semantic integrity.
In the text modality, we employ keyword-level
semantic perturbations. Defined the i-th word wi, let
the original text input be xtext={w1, w2, . . . , w L},
where Ldenotes the number of words. We only perform
perturbation operations on tokens in Ptext(·), including
synonym substitution, random masking, and word-
level Unicode alteration. The perturbed text ytextis
represented as:
˜xtext=Ptext(xtext), (12)
wherePtext(·)denotes the perturbation operator applied
to the words.
In the image modality, we inject pixel-wise Gaus-
sian noise with the perturbed function Pvis(·)into the
original query image to construct perturbation variants
of varying magnitudes. Given the original image Iorig,
its perturbed version is defined as:
˜I=Pvis(I) =Iorig+ϵ, ϵ∼ N(0, σ2), (13)
where σcontrols the perturbation strength. We select
multiple σvalues ( e.g.,{10,20,30,40}) to observe
whether the model’s generation exhibits systematic
shifts across different perturbation levels.
The core motivation behind this cross-modal per-
turbation design is that the autoregressive mechanism of
LLMs is sensitive to token-level perturbations, while the
embedding-based retrieval mechanism of RAG exhibits
a certain degree of input robustness. Therefore, under
the condition that key tokens are perturbed, if the
generated results remain consistent, it is more likely to
be RAG-driven; otherwise, it is more likely to rely on
the LLM itself.
To accommodate the multimodal setting, we fur-
ther transform the perturbed visual input with a specific
prompt into a textual representation using the model’s
own image encoder and captioning head, denoted as
fblack(·). Specifically, the noisy image ˜Iis processed by
the model to yield a corresponding textual embedding
or caption, denoted as ˜yvis=fblack(prompt, ˜I). This
allows us to directly attribute the influence of visual
perturbations within the same textual autoregressive de-
coding space. By analyzing the generation conditioned
on˜yvis, we can assess whether the perturbations inthe modality encoder propagate meaningfully into the
generation process, and how they are weighted relative
to retrieved contexts or prior tokens.
C. Zero-Gradient Auditing Mechanism
LLMs and MLLMs in practical applications are
often closed-source APIs without access to model
parameters or gradients, we propose a Zero-Gradient
Attribution Mechanism to estimate the contribution
of input tokens to the output under semi-black-box
conditions. The method does not rely on the internal
architecture or backpropagation, but instead models
attribution by analyzing the relationship between input
perturbations and output variations.
We construct Nrandomly perturbed variants
˜xtext(i)and feed them into the black-box model
fblack(·)to obtain the corresponding outputs ˜y(i)=
fblack(˜xtext(i)). Each perturbed sample ˜xtext(i)is asso-
ciated with a binary mask vector m(i)∈ {0,1}L, where
each element indicates whether the corresponding word
is retained (1) or perturbed (0).
To quantify the impact of each perturbation on
the model’s output, we define an attributed response
score as:
r(i)=γ1Len ˜y(i)
Len( ˜xtext(i))+γ2Sim ˜y(i),˜xtext(i)
,
(14)
where Len(·)denotes the output length, and Sim(·,·)
represents a semantic similarity metric ( e.g., BLEU,
cosine similarity, BERTScore ). The weighting factors
γ1andγ2control the contribution of each term. A
lower r(i)indicates a greater perturbation impact on
the output.
We then stack all mask vectors into a perturbation
matrix M∈0,1N×Land assemble all response scores
into a vector r∈RN. To estimate token attribution,
we fit a linear regression model:
r=Mβ, (15)
where β∈RLdenotes the attribution scores to be
estimated. We adopt ridge regression with regularization
to obtain a closed-form solution:
β= (M⊤M+αΛ−1M⊤r, (16)
where α > 0is a ridge coefficient used to reduce
multicollinearity and improve robustness, and Λde-
notes the L×Lidentity matrix. Each component
βjthus represents the estimated contribution of the
j-th token wjto the model’s output, serving as its
attribution strength. This approach performs zero-
order sensitivity analysis purely based on input-output
behaviors, without relying on gradient information.
In the multimodal setting, if the input includes an
image xvis, we apply small Gaussian noise perturbations
˜xvis=Pvis(xvis)and process it via the model’s
visual encoder to generate a textual representation
˜yvis=fblack(˜xvis), which is then integrated into the text
generation pipeline. This allows image perturbations to

be translated into measurable changes in the token-level
textual output.
The proposed mechanism is applicable to text-only,
multimodal, and retrieval-augmented generation (RAG)
scenarios, and enables unified attribution analysis under
semi-black-box constraints.
D. Attribution Scoring under RAG Switch
To quantitatively characterize the difference in
contribution between RAG and the LLM in generating
outputs, we introduce the Attribution Difference Score
(ADS), which measures the change in the impact of
key words on the generated results by toggling the
retrieval component.
We generate outputs for the same perturbed
input with the retrieval module enabled and disabled,
respectively, and denote the model outputs as:
βRAG=Ridge (xtext,ˆy;ˆy(M)RAG ) (17)
βw/o RAG =Ridge (xtext,ˆy;ˆyw/o (M)RAG ) (18)
Where ˆy(M)RAG andˆyw/o (M)RAG denote the outputs
ofxtextin the black-box system with (M)RAG enabled
and disabled, respectively. For each word, we compute
the corresponding ridge regression coefficient. For each
word, the ridge regression coefficient wjrepresents the
gradient attribution score of the j-th word wj, which
directly indicates the sensitivity of the model’s output
to this word. To categorize the source of each word, we
define thresholds according to the following criteria:
label(wj) =(
Pretrained Member , βj≥τ,
Retrieved/Non Member ,βj< τ,
(19)
Then, we compute the modified Attribution Dif-
ference Score (ADS) for each critical word using:
Diff (βj) =β(M)RAG (j)−βw/o (M)RAG (j)(20)
Here, w/and w/o represent outputs with and
without RAG, respectively. The ADS directly measures
how significantly toggling RAG affects the generated
outputs, reflecting the token’s dependence on retrieval
content.
Here, τrepresents the threshold distinguishing
words originating from the model’s pretrained knowl-
edge from those retrieved externally. Subsequently, we
further dissect the words labeled as Retrieved/Non
Member by setting additional empirical thresholds
τ1= 0.1andτ2=−0.1, applying the attribution
difference score Diff (βj)as follows:
label(wj) =(
Non-Member , τ 1≤Diff(βj)< τ2,
Retrieved Member ,otherwise .
(21)
Typically originating from user input prompts,
from words retrieved via external knowledge bases.
It is crucial to emphasize that the selected words,classified through this multi-tiered thresholding ap-
proach, constitute meaningful keywords, deliberately
excluding trivial or semantically negligible words such
as prepositions and conjunctions (e.g., and,is,or). This
targeted keyword approach ensures the keyword will
be detected in the output of LLMs.
The core idea of this bias-based attribution strat-
egy is as follows: under the condition that a key word is
perturbed, if disabling RAG causes a significant output
shift while enabling RAG mitigates this deviation,
it implies that the model’s generation depends on
externally retrieved content.
E. SMA Framework
The SMA framework is designed to address the
issue of knowledge source attribution in black-box
settings, determining whether a particular piece of
information in the model’s output originates from the
pre-training corpus of a LLM or from the retrieved
content of an external retrieval module (RAG/MRAG).
This method is based on the concept of zeroth-order
attribution, which infers the importance of words or
regions solely through changes in input-output pairs,
even when gradients, attention weights, or internal
representations are inaccessible. Specifically, SMA
applies structured perturbations to the input content
(text or image): in text, it randomly samples and
retains a subset while injecting character-level Uni-
code variants; in images, it adds Gaussian noise that
preserves high-level semantics. These perturbations do
not affect the retrieval results but distort the model’s
internal encoding, enabling the inference of the input
components primarily responsible for the generated
output.
After collecting outputs from multiple perturbed
inputs, SMA constructs response scores based on the
output length and its semantic similarity to the original
output. The perturbation masks and response scores
are then fed into a ridge regression model to obtain
token-level attribution scores. Higher scores indicate
that the corresponding word has a stronger influence
on the model’s output. Furthermore, in multimodal
scenarios, SMA treats images as a “contextual signal”
for generating descriptive text. By jointly perturbing
both images and text, it achieves cross-modal attribution
capabilities. For example, if a word’s attribution score
remains stable under varying image noise levels, it
likely originates from the LLM; if the attribution
fluctuates significantly with image perturbations, it is
more likely derived from the retrieved visual content.
The overall process of SMA is illustrated in
Figure 2, which includes three stages: structured
perturbation of text and image inputs, querying the
black-box model and collecting outputs, and performing
attribution modeling via ridge regression. The full
procedure is detailed in Algorithm 1, which shows
how to estimate the contribution of each input token to
the final output through perturbation sampling, response

scoring, and regression analysis, thereby enabling
automated auditing and tracing of knowledge sources.
Algorithm 1 Source Member Attribution (SMA)
Require: Input x(text or multimodal), model fθ,
number of perturbations N, similarity weights
γ1, γ2, regularization α
Ensure: Token-level attribution score vector β∈RL
1:Extract textual tokens: xtext={w1, w2, . . . , w L}
2:Obtain original model output: ˆy=fθ(x)
3:fori= 1 toNdo
4: Sample binary mask vector m(i)∈ {0,1}L
uniformly
5: ifP
jm(i)
j= 0 then
6: Set one random m(i)
j←1
7: end if
8: Apply character-level or Unicode perturbation
to masked tokens to form ˜x(i)
9: Query the model: ˆy(i)=fθ(˜x(i))
10: Compute response score:
r(i)=γ1Len ˆy(i)
Len( ˜x(i))+γ2Sim ˆy(i),˜x(i)
11:end for
12:Stack {m(i)}N
i=1into binary mask matrix M∈
{0,1}N×L
13:Form response vector: y= (y1, y2, . . . , y N)⊤
14:Solve ridge regression:
β= (M⊤M+αΛ−1M⊤r
15:foreach word j= 1 toLdo
16: ifβj≥τthen
17: Label word wjasPretrained Member
18: else
19: Temporarily label word wj as
Retrieved/Non Member
20: end if
21:end for
22:foreach word jlabeled as Retrieved/Non Member
do
23: ifτ1≤Diff(βj)< τ2then
24: Re-label word wjasNon-Member
25: else
26: Re-label word wjasRetrieved Member
27: end if
28:end for
29:return Token belonging
V. E XPERIMENT AND EVALUATION
In this section, we present a comprehensive
evaluation of our proposed zero-gradient attribution-
based attack framework. Our goal is to assess its
effectiveness in identifying the provenance of generated
content—distinguishing whether specific output tokens
originate from retrieved inputs (RAG/MRAG) or from
the internal knowledge of the underlying LLM or
MLLM.A. Experiment setup
Datasets. We use two datasets ragbench and
PubMedQA [22] for rag storage. For MIA comparison,
we used WikiMIA [49] and WikiMIA-24 [48] datasets.
We also use the two databases VL-MIA-image [24] and
the Wikipedia image datasets for MRAG evaluation.
Baseline. For textual RAG evaluation, we em-
ployed LLaMA-2 7B [18], LLaMA-3.1 8B [17], and
Qwen2.5 7B [19] language models. Retrieval contexts
were generated using three embedding methods: All-
MiniLM-L6-v2, bge-large-en-v1.5 [20], and gte-Qwen2-
1.5B-instruct [21].
For MRAG, we evaluated using the multi-model
Qwen2.5-VL-7B model [23] with the CLIP vit-base-
patch32 image embedding model. In addition, we
utilized commercial large language models, including
ChatGPT-4o mini [78], ChatGPT-4.1 mini [78], and
Gemini-2.5 flash [79], as our black-box model base-
lines.
Evaluation metrics We utilize two primary met-
rics, accuracy and coverage. Accuracy (ACC). De-
fined as the cosine similarity between embeddings of
attributed RAG-derived outputs and original retrieval
results, measuring the fidelity of our attribution:
ACC =cos(Enc(orec),Enc(oorig)) (22)
Coverage. Calculated as the ratio of identified
RAG-derived tokens to total tokens retrieved by RAG
systems, indicating how comprehensively our attribu-
tion captures retrieval content. FPR uses to quantify
the proportion of data that are incorrectly identified as
members. Member Data Accuracy (MDA). Defined
Nmember
correct is the number of member data correctly
identified as members, and Nmember
total is the total number
of member data samples.
MDA =Nmember
correct
Nmember
total(23)
Non-member Data Accuracy (NMDA). Defined
Nnon-member
correct is the number of non-member data cor-
rectly identified as non-members, and Nnon-member
total is
the total number of non-member data samples.
NMDA =Nnon-member
correct
Nnon-member
total(24)
B. Experimental Results
1) Evaluation of RAG&LLM System Performance
Across Models: To evaluate the generalizability of SMA
accross different RAG embedding models and various
LLMs, we combine three mainstream foundation LLMs
including LLaMA-3.1 8B, Qwen2.5 7B, and LLaMA-
2 7B, with three different RAG embedding models,
such as all-MiniLM-L6-v2, bge-large-en-v1.5 and te-
Qwen2-1.5B-instruct. Table I reports two key metrics,
ACC and coverage, for each combination across the RB
and PMQA datasets, under the retrieval topksetting
of 3. We can draw three main conclusions: ❶The

Black BoxModelLLM & RAG
Llama-38B
EncoderDecoder…Embedding
Vector DB
TEXTRetrieved Text
MLLM & MRAGVision EncoderVision Decoder…
Vector DB
Retrieved Images
+ TEXT
Vision Embedding
InputInput PerturbationLLMMLLMBetween the months of March and April 2014, a court in Minya, Egypt, has recommended the 
Random choose some lettersa g m G
Convert Them to Unicode
Between !he "onths #f $archand April 2014,%court in $inya,E&ypt, has recommended !heText:Please describe the image in detail, including text, structure, and main componentsImage:
Add GaussiannoiseMulti-level Noise
Sigma: 20Sigma: 30Sigma: 60Sigma: 80
InputAttribution under RAG Switch
Output
<-τNon-Member>τLLM/MLLMMember(M)RAG[-τ, τ]
(M)LLM perturbedZero Gradient AuditingRidge Regression
Vision TransformerThis image shows a …others diminishing in size as they fan outward.
This image appears to be a small and highly pixelated picture …scheme.!!"#$%#" + !#&'()(#",#$%"&) (-'-+/Λ)(!-'r
……Perturbed'!(#)*#" + '#+,-"#",#$%"&/I=2$%"&+∈ [“Between 'he (onths )f *arch and April 2014, a court in Ginya, Egypt, has recommended 'he”,” Between 'he (onths )f Garch and April 2014, +court in Ginya, Egypt, has recommended 'he”,…,]
Score[“Between”,  “the”,  “months”,  “of”,  “March”,  “and”,  “April”,  “2014”,  “a”,  “court”,  “in”,  “Minya”,  “Egypt”,  “has”,  “recommended”,  “the”,.., ”name” ] 
[“This image appears to be a small and hi,hlypixelated picture … scheme.”,” This image appears to be a small and highly pixel-ted picture … scheme”,…,][“This”, “image”, “appears”,  “to”,  “be”,  “a”, “small”, “and”, “highly”, “pixelated”, “picture”, …,  ”scheme” ] (M)LLM Input
OptimizeLLMMLLMBetween the months of March and April 2014, a court in Minya, Egypt, has recommended the 
diminishing in size they fan outward.Member MRAG DATACompare
Fig. 2: Overview of the SMA framework for LLMs and MLLMs under RAG and MRAG settings, illustrating
the token-based attribution pipeline
accuracy of SMA can be influenced by the choice
of embedding methods. For example, the bge-large-
en-v1.5 embedding boosts LLaMA-3.1 8B’s ACC on
PMQA to 0.9172, while the same model with all-
MiniLM-L6-v2 on RB achieves only 0.7230. ❷SMA
The type of LLMs can also affect the performance of
SMA.❸There exists a trade-off between accuracy and
coverage across different models and embeddings. For
instance, although Qwen2.5 7B attains a high accuracy
of 0.8288 using all-MiniLM-L6-v2 on RB, its coverage
drops to 0.1724, compared to LLaMA-3.1 8B, which
maintains a more balanced profile with ACC 0.7230
and coverage 0.8095 under the same setting.
In summary, these results highlight the importance
of careful selection and tuning of both embedding
models and LLMs to optimize the overall performance
of RAG-based systems, as both accuracy and coverage
are sensitive to these choices.
InMRAG&MLLM System . When adding noise,
after white-box attribution experiments with MLLMs,
Fig. 3a shows that under same prompts and model
hyperparameter, introducing noise to an image leads
the MLLM to focus more attention on the perturbed
image. Meanwhile, results from MRAG experiments
demonstrate that adding noise to the image does not
affect the MRAG retrieval results. Based on these
findings, we first obtain the MLLM inference outputs
using the original image paired with the image retrieved
by MRAG, and then repeat the process using the noised
image. By constructing prompts with the noisy image,
we guide the MLLM’s attention toward the targeted
image, thereby extracting descriptive content about theother image. We then apply attribution methods to
identify which parts of the LLM output originate from
the image, and compare these outputs with the original
image descriptions to score and evaluate the attribution
results. Fig. 3b illustrates the effects of varying levels
of Gaussian noise in four different models which
represented by different std values introduced to images
in MRAG& MLLM System. We observe how noise
influences performance metrics.
As shown in the figure, moderate Gaussian noise
which std values around 50 to 60, enhancing the
performance of the MLLM, particularly reflected in
increased ACC and AUC scores. This improvement
indicates that moderate noise effectively directs the
model’s attention towards the perturbed images, thus
enhancing its capability for distinguishing sensitive
visual information, and subsequently improving True
Positive Rate. However, excessively high noise at the
std value of 80 negatively impacts model performance
across all metrics, likely due to significant image
distortion that hinders the model’s ability to accurately
interpret visual content.
These findings suggest that carefully calibrated
noise perturbations can substantially improve the at-
tribution accuracy and robustness of MRAG systems
in membership inference attacks, highlighting the
importance of optimal noise management for practical
deployments.
2) Comparison with SoTA method: We conducted
extensive experiments to evaluate the effectiveness of
our proposed MIA method in scenarios involving RAG.
Specifically, we compare our method, SMA, against

TABLE I: Performance Comparison of RAG Embedding Models Combined with LLMs with topk=3
RAG&LLM ModelsLLaMA-3.1 8B Qwen2.5 7B LLaMA-2 7B
ACC Coverage ACC Coverage ACC Coverage
RAG(all-MiniLM-L6-v2)+RB 0.7230 0.8095 0.8288 0.1724 0.8030 0.6667
RAG(all-MiniLM-L6-v2)+PMQA 0.7366 0.8000 0.6700 0.5714 0.7902 0.3500
RAG(bge-large-en-v1.5)+RB 0.9036 0.4500 0.6026 0.5714 0.7483 0.5000
RAG(bge-large-en-v1.5)+PMQA 0.9172 0.2619 0.6691 0.5700 0.8445 0.5714
RAG(gte-Qwen2-1.5B-instruct)+RB 0.6083 0.3750 0.6026 0.8000 0.8464 0.4219
RAG(gte-Qwen2-1.5B-instruct)+PMQA 0.8369 0.6842 0.5311 0.4000 0.9137 0.3409
Note : ACC (Accuracy) indicates the similarity between the outputs from membership inference attack (MIA) and the original RAG retrieval
results. Coverage represents the proportion of matched RAG data relative to all data retrieved by the RAG system.
(a) White Box Noise Comparison on attention and
Gaussian noise
(b) Effect of Gaussian noise on MRAG&MLLM system
performance metrics
Fig. 3: Attribution attention comparison under different
noise levels
five baseline methods, PETEL [15], Mink++ [50], Min-
K% PROB (K= 10,20,30)[77], under the six LLM
models on two distinct datasets.
Table III presents a comprehensive evaluation
of different MIA methods—PETEL, Mink++, Min-
K%PROB (K= 10,20,30), and our proposed SMA
framework—across six LLM models and two bench-
mark datasets under RAG settings. The table reports
ACC and coverage for each method and model combina-
tion. It is evident that the SMA framework consistently
outperforms the baselines by a substantial margin. For
instance, on the WikiMIA dataset with LLaMA-2 7B,
SMA achieves an accuracy of 0.8624 and a coverage
of 0.5882, while others remain below 0.53 on bothmetrics. Similarly, with LLaMA-3.1 8B on WikiMIA-
24, SMA attains 0.7233 accuracy, significantly sur-
passing the highest baseline of 0.6280. Notably, SMA
also demonstrates strong coverage on Qwen2.5 7B,
achieving up to 0.7500. These results highlight the
robustness and effectiveness of the SMA framework,
establishing a new state-of-the-art in membership
auditing for retrieval-augmented LLMs, especially
under black-box constraints. It is worth noting that
existing methods such as PETEL and Mink++ exhibit
inherent incompatibility with commercial API-based
language models like ChatGPT-4o mini, ChatGPT-4.1
mini, and Gemini-2.5 flash. Specifically, these baseline
approaches require direct access to internal parameters,
such as gradient or singal token, to effectively perform
membership inference. However, prevalent commercial
APIs typically restrict output to plain textual responses,
withholding granular internal details. Consequently,
PETEL, Mink++, and similar methods relying on such
internal model specifics are substantially constrained
and fail to adapt efficiently to these black-box commer-
cial environments. For PETEL and Min-K%PROB, we
implemented minimal modifications to accommodate
commercial APIs. Additionally, the Gemini API does
not return the parameters required by Min-K%PROB
and Mink++, further limiting its applicability. In con-
trast, our proposed SMA framework, operating solely
on externally observable input-output perturbations,
circumvents these limitations, making it uniquely suited
and adaptable for auditing membership inference in
contemporary commercial API models. By attributing
input data and extracting relevant information from the
RAG database when identified, our method notably
outperforms the baseline methods.
We also conducted comprehensive evaluations
to compare our SMA framework against existing
MRAG based MIA methods, specifically VLMA [51]
and vlm_mia [54], using the VL-MIA-images dataset
with the base model Qwen2.5-VL 7B, ChatGPT-
4o mini, ChatGPT-4.1 mini, and Gemini-2.5 flash.
Table III shows the performance of different MIA
methods—VLMA, VLM_MIA, and our SMA frame-
work—under the MRAG scenario with topk, evaluated
using multiple metrics including ACC, AUC, TPR
and FPR. The results clearly demonstrate that the

TABLE II: Performance comparison of MIA methods under RAG settings across six LLM models
Model MethodWikiMIA [49] WikiMIA-24 [48]
ACC Coverage ACC Coverage
LLaMA-2 7BPETEL [15] 0.5230 0.5230 0.5741 0.5741
Mink++ [50] 0.5190 0.5190 0.5010 0.5010
MIN-K%PROB (K=10%) [77] 0.5032 0.5032 0.5370 0.5370
MIN-K%PROB (K=20%) [77] 0.5015 0.5015 0.5322 0.5322
MIN-K%PROB (K=30%) [77] 0.5029 0.5029 0.5314 0.5314
SMA 0.8624 0.5882 0.6730 0.6000
LLaMA-3.1 8BPETEL [15] 0.5332 0.5332 0.6280 0.6280
Mink++ [50] 0.5200 0.5200 0.5960 0.5960
MIN-K%PROB (K=10%) [77] 0.5126 0.5126 0.5556 0.5556
MIN-K%PROB (K=20%) [77] 0.5074 0.5074 0.5378 0.5378
MIN-K%PROB (K=30%) [77] 0.5045 0.5045 0.5531 0.5531
SMA 0.6351 0.7895 0.7233 0.6667
Qwen2.5 7BPETEL [15] 0.5088 0.5088 0.4968 0.4968
Mink++ [50] 0.5890 0.5890 0.5540 0.5540
MIN-K%PROB (K=10%) [77] 0.5000 0.5000 0.5918 0.5918
MIN-K%PROB (K=20%) [77] 0.5072 0.5072 0.5676 0.5676
MIN-K%PROB (K=30%) [77] 0.5035 0.5035 0.5523 0.5523
SMA 0.6484 0.6700 0.5705 0.7500
ChatGPT 4.1-miniPETEL [15] 0.5370 0.5370 0.5384 0.5384
Mink++ [50] 0.3621 0.3621 0.4909 0.4909
MIN-K%PROB (K=10%) [77] 0.5025 0.5025 0.5306 0.5306
MIN-K%PROB (K=20%) [77] 0.5023 0.5023 0.5209 0.5209
MIN-K%PROB (K=30%) [77] 0.5024 0.5024 0.5185 0.5185
SMA 0.7489 0.5455 0.7394 0.5400
ChatGPT 4o-miniPETEL [15] 0.5625 0.5625 0.5537 0.5537
Mink++ [50] 0.4290 0.4290 0.4733 0.4733
MIN-K%PROB (K=10%) [77] 0.5700 0.5700 0.5411 0.5411
MIN-K%PROB (K=20%) [77] 0.5789 0.5789 0.5580 0.5580
MIN-K%PROB (K=30%) [77] 0.5749 0.5749 0.5797 0.5797
SMA 0.6276 0.6273 0.6839 0.6012
Gemini 2.5-flashPETEL [15] 0.5614 0.5614 0.5583 0.5583
Mink++ [50] NaN NaN NaN NaN
MIN-K%PROB (K=10%) [77] NaN NaN NaN NaN
MIN-K%PROB (K=20%) [77] NaN NaN NaN NaN
MIN-K%PROB (K=30%) [77] NaN NaN NaN NaN
SMA 0.5828 0.6000 0.6137 0.5669
SMA framework achieves superior overall performance,
setting new benchmarks for both accuracy and relia-
bility. Specifically, SMA attains the highest accuracy
(0.7900) and AUC (0.8227) in Qwen2.5 VL 7B, while
also achieving the lowest false positive rate (FPR) of
0.0785 among all methods. In comparison, VLMA and
VLM_MIA not only fall behind in accuracy and AUC,
but also suffer from substantially higher FPR values,
with VLM_MIA’s FPR reaching 0.9968. These results
highlight the robust capability of the SMA framework to
effectively audit membership information in multimodal
RAG systems, outperforming existing methods across
nearly all evaluation dimensions.
Our experiments revealed that VLMA experiences
a reduction in attack accuracy when MRAG is intro-
duced, compared to its performance reported originally.
Additionally, our observations indicate that as the topk
parameter in the MRAG system increases, the ACC
of the membership inference attack correspondingly
decreases.
The attack principle employed in our MRAG-
based MIA method closely follows the approach
used in the RAG&LLM system scenario. Specifically,we separate images retrieved by MRAG from those
provided by the user and apply attribution techniques to
isolate content derived from the pre-trained MLLM. By
assessing whether the user-provided images align with
the target images, we effectively conduct the MIA. The
results clearly demonstrate the superiority of our SMA
method in the presence of MRAG. Specifically, our
method achieved an ACC of 0.79, higher than VLMA’s
0.5983 and vlm_mia’s 0.4985.
In both retrieval-augmented settings, whether text-
only RAG or MRAG. Our SMA consistently outper-
forms prior membership inference methods. By distin-
guishing content sourced from the external database
versus the model’s own pre-training, SMA maintains
high accuracy and coverage even when RAG or
MRAG is enabled, whereas existing techniques like
PETEL, Mink++, and VLMA suffer substantial drops
in performance once the retrieval component is active.
C. Ablation Studies
To demonstrate the contribution of each compo-
nent in our attack framework, we conducted a com-
prehensive ablation study across both the RAG&LLM

TABLE III: Performance comparison of MIA methods under RAG settings across LLMs
Model Method ACC AUC TPR FPR
Qwen2.5 VL 7BVLM_MIA [54] 0.4311 0.5208 0.6100 0.5966
VLMA [51] 0.4985 0.4303 0.9968 0.9998
SMA 0.7900 0.8227 0.6400 0.0785
ChatGPT 4o-miniVLM_MIA [54] 0.5000 0.2476 0.2480 1.0000
VLMA [51] 0.5865 0.5160 0.5900 1.0000
SMA 0.7700 0.8541 0.6800 0.1569
ChatGPT 4.1-miniVLM_MIA [54] 0.5000 0.0237 0.0000 1.0000
VLMA [51] 0.5176 0.2917 0.1000 1.0000
SMA 0.7300 0.8123 0.8000 0.3529
Gemini 2.5 flashVLM_MIA [54] 0.6193 0.8945 0.2628 0.9948
VLMA [51] 0.5449 0.4503 0.4000 0.8750
SMA 0.6300 0.4937 0.6000 0.2157
and MRAG&MLLM systems. We have decomposed
our approach into (i) a basic zero-gradient attribution
method, and (ii) the combination of noise injection
with zero-gradient attribution. This experimental setup
allows us to systematically assess the necessity and
effect of noise perturbation for optimizing our method.
With the bge-large-en-v1.5 embedding model, topkset
to 8 and the ragbench dataset.
TABLE IV: Ablation Study for LLMs and MLLM for
each singal method
Type Model Attr+ZeroGrad +Noise
LLMLLaMA-3.1 8B 0.7666 0.7895
Qwen2.5 7B 0.6033 0.6484
ChatGPT-4o mini 0.5733 0.6276
MLLMQwen2.5-VL 7B 0.6500 0.7900
ChatGPT-4o mini 0.7100 0.7700
Gemini-2.5 flash 0.5100 0.6300
As shown in Table IV, it presents the results of
ablation experiments, evaluating the accuracy impact of
zero-gradient attribution alone (Attr+ZeroGrad) and the
addition of noise (+Noise) across both LLM and MLLM
models. The table is composed of results for LLaMA-
3.1 8B, Qwen2.5 7B and ChatGPT-4o mini(LLMs)
as well as Qwen2.5-VL 7B (MLLM). From table,
two conclusions can be drawn: First, adding noise
consistently improves the accuracy for all models. For
example, the accuracy of LLaMA-3 8B increases from
0.7666 to 0.7895, while Qwen2.5 7B rises from 0.6033
to 0.6484, and ChatGPT-4o mini from 0.5733 up to
0.6276. In visual Model, Qwen2.5-VL 7B goes from
0.65 to 0.7900, ChatGPT-4o mini goes from 0.7100 to
0.7700, Gemini-2.5 flash goes from 0.5100 to 0.6300
after noise is added. Second, although the absolute
increase varies, the largest gain is observed for Gemini-
2.5 flash, where accuracy improves by 0.1200.
Overall, these results demonstrate that combining
zero-gradient attribution with noise is highly effective
for both language and multimodal models, as evidenced
by the substantial improvement in accuracy—for in-
stance, an increase of 0.1200 for Gemini-2.5 flash.TABLE V: Comparison with White-box Attribution in
the ragbench dataset and topkset to 8, all-MiniLM-L6-
v2 embedding model
Method LLaMA-3.1 8B Qwen2.5 7B
Attr+Noise+White-box 0.8612 0.8073
Attr+Noise+Alt. Model 0.7058 0.6464
Attr+Noise+Zero-Grad 0.7008 0.6657
Fig. 4: Separation of token between RAG and LLM
pre-train based on topk=3
Ablation result across White-Box attribution: Ta-
ble V shows the comparison result in methods of White-
Box Attribution with Noise, White-Box Attrition with
Alternative Model and our current method Black-Box
zero-gradient Attribution with Noise under two LLM
categories. We can know that: First, The White-Box
Attribution with Noise is under the desired condition,
thus the average Accuracy Score of two LLMs get the
0.8612 and 0.8073. Second, the the alternative models
(the two models are exchanged for attribution) reach
the 0.7058 and 0.6464. Finally, our current method
Black-Box Attribution with Noise get the Accuracy of
0.7008 and 0.6657 by two LLMs.
In summary, our experiments conclusively demon-
strate that noise perturbation significantly enhances
MIA accuracy in both text-only and multi-model

Fig. 5: Separation of token between MRAG and MLLM
pre-train based on topk=1
systems. Moreover, the gap between our semi-black-
box method and the white-box upper bound is relatively
narrow, highlighting the effectiveness and practicality
of our approach.
VI. D ISCUSSION
Costing. One practical consideration of our SMA
framework is the inference cost associated with black-
box access. Since SMA operates in a RAG black-box
setting and relies on perturbation-based zero-gradient
attribution, each query potentially requires multiple
forward passes to complete attribution across tokens.
When accessed via APIs, this leads to increased token
consumption and corresponding monetary cost. The
total token usage for each inference in SMA can be ap-
proximated as (Token SMA=Token Output From Taret + 60) .
The actual cost depends on the maximum token limits
and pricing model of the deployed server. Fortunately,
with the emergence of low-cost model providers such
as DeepSeek, the economic overhead is increasingly
manageable—for example, 1M tokens currently cost as
little as $0.07. To further mitigate operational costs
and avoid triggering abnormal API usage alerts, a
promising future direction is to integrate shadow model
inference [52] [53]. This approach would replicate the
queried model locally, allowing efficient large-scale
offline testing of the SMA framework without incurring
external token charges or raising API rate-limiting
alarms. Such optimizations could substantially improve
the deployment practicality of SMA in high-volume
audit scenarios.
Parameter Sensitivity. Another aspect of our
analysis involves understanding how the SMA attack
behaves under different parameter settings in both
RAG&LLM and MRAG&MLLM System scenarios.In Figure 6, we evaluate two key hyperparameters,
the RAG number of results topk, and the number
of perturbation queries used in zero-gradient black-
box attribution. In Fig.6a , we vary topkfrom 1 to
8, which controls how many documents are retrieved
from the RAG system and appended to the prompt.
Despite the increasing uncertainty introduced by adding
more retrieved passages, our SMA method maintains
a consistently high ACC across different topkvalues.
This suggests that SMA is robust to varying retrieval
depths and can effectively filter out non-member
content from diverse inputs. Fig. 6b explores the
effect of different perturbation counts, ranging from
1 to 100. Perturbations influence the precision of
attribution in zero-gradient settings: too few queries
may yield noisy gradients, while excessive queries
increase computational load. Our results show that
SMA achieves stable and high accuracy around 60–
80 perturbations, with diminishing returns beyond that
point. The number of perturbations can thus be flexibly
tuned based on the attacker’s computational budget and
accuracy requirement.
(a) Adaptation to RAG topk
(b) Black-box Perturbations vs. ACC
Fig. 6: (a) Varying RAG retrieval depth shows that SMA
sustains high ACC and graceful coverage degradation
even as topkgrows. (b) Varying the number of zero-
gradient perturbations reveals an optimal budget that
maximizes ACC in black-box MIA.
Limitations. While SMA demonstrates strong per-
formance under RAG scenarios, it has several practical
limitations. First, as a strictly semi-black-box technique,

SMA is constrained by the language model’s maximum
token limit and sampling temperature. Specifically, if
the temperature is set above typical operational ranges
such as temperature bigger than 5.0 generated outputs
can become overly stochastic, degrading attribution
consistency and attack accuracy. Similarly, overly
restrictive max_tokens settings truncate the context
window, preventing sufficient perturbation analysis and
impairing the attack’s success rate. Second, SMA’s
reliance on repeated perturbation-based queries leads
to high CPU and API usage. Conducting dozens to
hundreds of semi-black-box attribution passes incurs
significant computational overhead and latency, result-
ing in longer overall execution times compared to white-
box or single-shot methods. Future work may explore
optimizations such as batched perturbations or early
stopping criteria to alleviate these resource constraints.
VII. C ONCLUSION
In this paper, we propose SMA (Source-aware
Membership Auditing), the first membership auditing
framework with source attribution capability, to deter-
mine whether the leaked content originates from the
model’s pre-training corpus or an external retrieved
data source in a black-box setting. The core of SMA
consists of two key designs: (1) response sensitivity
analysis based on lightweight input perturbations for
eliciting differential variations across sources; and (2) a
zero-gradient scoring mechanism that does not require
gradient information and is suitable for black-box
environments in deployed systems. To support multi-
model inputs, SMA also introduces a unified cross-
modal attribution mechanism, which for the first time
enables cross-modal auditing of member leakage in
MRAG systems. The experimental results validate its
significant advantages in textual and multi-model tasks,
providing a practical tool for data compliance and
privacy auditing in generative systems.
REFERENCES
[1]X. Xing, C.-W. Kuo, L. Fuxin, Y . Niu, F. Chen, M. Li, Y . Wu,
L. Wen, and S. Zhu, “Where do Large Vision-Language Models
Look at when Answering Questions?” Mar. 2025.
[2]N. Kokhlikyan, V . Miglani, M. Martin, E. Wang, B. Alsallakh,
J. Reynolds, A. Melnikov, N. Kliushkina, C. Araya, S. Yan,
and O. Reblitz-Richardson, “Captum: A unified and generic
model interpretability library for pytorch,” 2020. [Online].
Available: https://arxiv.org/abs/2009.07896
[3]H. Zhao, H. Chen, F. Yang, N. Liu, H. Deng, H. Cai, S. Wang,
D. Yin, and M. Du, “Explainability for large language models:
A survey,” ACM Trans. Intell. Syst. Technol. , vol. 15, no. 2,
Feb. 2024. [Online]. Available: https://doi.org/10.1145/3639372
[4]W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S.
Chua, and Q. Li, “A survey on rag meeting llms: Towards
retrieval-augmented large language models,” in Proceedings of
the 30th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining , ser. KDD ’24. New York, NY , USA:
Association for Computing Machinery, 2024, p. 6491–6501.
[Online]. Available: https://doi.org/10.1145/3637528.3671470
[5]Y . Huang, S. Gupta, Z. Zhong, K. Li, and D. Chen,
“Privacy implications of retrieval-based language models,”
inThe 2023 Conference on Empirical Methods in Natural
Language Processing , 2023. [Online]. Available: https:
//openreview.net/forum?id=3RTpKMVg0P[6]S. Zeng, J. Zhang, P. He, Y . Liu, Y . Xing, H. Xu, J. Ren,
Y . Chang, S. Wang, D. Yin, and J. Tang, “The good and the
bad: Exploring privacy issues in retrieval-augmented generation
(RAG),” in Findings of the Association for Computational
Linguistics: ACL 2024 , L.-W. Ku, A. Martins, and V . Srikumar,
Eds. Bangkok, Thailand: Association for Computational
Linguistics, Aug. 2024, pp. 4505–4524. [Online]. Available:
https://aclanthology.org/2024.findings-acl.267/
[7]T. Zhang, Y . Jiang, R. Gong, P. Zhou, W. Yin, X. Wei,
L. Chen, and D. Liu, “DEAL: High-efficacy privacy attack on
retrieval-augmented generation systems via LLM optimizer,”
2025. [Online]. Available: https://openreview.net/forum?id=
sx8dtyZT41
[8]T. Koga, R. Wu, and K. Chaudhuri, “Privacy-preserving
retrieval-augmented generation with differential privacy,” 2025.
[Online]. Available: https://arxiv.org/abs/2412.04697
[9]J. Qi, G. Sarti, R. Fernández, and A. Bisazza, “Model
internals-based answer attribution for trustworthy retrieval-
augmented generation,” in Proceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing ,
Y . Al-Onaizan, M. Bansal, and Y .-N. Chen, Eds. Miami,
Florida, USA: Association for Computational Linguistics,
Nov. 2024, pp. 6037–6053. [Online]. Available: https:
//aclanthology.org/2024.emnlp-main.347/
[10] V . Miglani, A. Yang, A. H. Markosyan, D. Garcia-Olano, and
N. Kokhlikyan, “Using captum to explain generative language
models,” arXiv preprint arXiv:2312.05491 , 2023.
[11] D. Zhang, Y . Yu, J. Dong, C. Li, D. Su, C. Chu, and D. Yu.
MM-LLMs: Recent Advances in MultiModal Large Language
Models. [Online]. Available: http://arxiv.org/abs/2401.13601
[12] C. Fu, Y .-F. Zhang, S. Yin, B. Li, X. Fang, S. Zhao, H. Duan,
X. Sun, Z. Liu, L. Wang, C. Shan, and R. He. MME-Survey:
A Comprehensive Survey on Evaluation of Multimodal LLMs.
[Online]. Available: http://arxiv.org/abs/2411.15296
[13] Microsoft Azure, “Azure API Management,” https://azure.
microsoft.com/en-us/products/api-management, May 2025, on-
line; accessed 11 May 2025.
[14] S. Liu, P.-Y . Chen, B. Kailkhura, G. Zhang, A. Hero, and P. K.
Varshney, “A primer on zeroth-order optimization in signal
processing and machine learning,” 2020. [Online]. Available:
https://arxiv.org/abs/2006.06224
[15] Y . He, B. Li, L. Liu, Z. Ba, W. Dong, Y . Li, Z. Qin, K. Ren,
and C. Chen. Towards Label-Only Membership Inference
Attack against Pre-trained Large Language Models. [Online].
Available: http://arxiv.org/abs/2502.18943
[16] Z. Li, Y . Wu, Y . Chen, F. Tonin, E. A. Rocamora,
and V . Cevher, “Membership inference attacks against
large vision-language models,” 2024. [Online]. Available:
https://arxiv.org/abs/2411.02902
[17] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian,
A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan,
A. Yang, A. Fan, and E. A. Anirudh Goyal., “The
llama 3 herd of models,” 2024. [Online]. Available:
https://arxiv.org/abs/2407.21783
[18] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi,
Y . Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale,
D. Bikel, L. Blecher, and . E. A. Cristian Canton Ferrer,
“Llama 2: Open foundation and fine-tuned chat models,” 2023.
[Online]. Available: https://arxiv.org/abs/2307.09288
[19] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li,
D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang,
J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao,
K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men,
R. Lin, T. Li, T. Xia, X. Ren, X. Ren, Y . Fan, Y . Su, Y . Zhang,
Y . Wan, Y . Liu, Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5
technical report,” arXiv preprint arXiv:2412.15115 , 2024.
[20] C. Li, M. Qin, S. Xiao, J. Chen, K. Luo, Y . Shao, D. Lian,
and Z. Liu, “Making text embedders few-shot learners,” 2024.
[Online]. Available: https://arxiv.org/abs/2409.15700
[21] Z. Li, X. Zhang, Y . Zhang, D. Long, P. Xie, and
M. Zhang, “Towards general text embeddings with multi-
stage contrastive learning,” 2023. [Online]. Available:
https://arxiv.org/abs/2308.03281
[22] Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu,
“Pubmedqa: A dataset for biomedical research question
answering,” 2019. [Online]. Available: https://arxiv.org/abs/
1909.06146

[23] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang,
P. Wang, S. Wang, J. Tang, H. Zhong, Y . Zhu, M. Yang, Z. Li,
J. Wan, P. Wang, W. Ding, Z. Fu, Y . Xu, J. Ye, X. Zhang, T. Xie,
Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin, “Qwen2.5-vl
technical report,” arXiv preprint arXiv:2502.13923 , 2025.
[24] K. Srinivasan, K. Raman, J. Chen, M. Bendersky, and
M. Najork, “Wit: Wikipedia-based image text dataset for
multimodal multilingual machine learning,” in Proceedings of
the 44th International ACM SIGIR Conference on Research
and Development in Information Retrieval , ser. SIGIR
’21. ACM, Jul. 2021, p. 2443–2449. [Online]. Available:
http://dx.doi.org/10.1145/3404835.3463257
[25] L. Ibanez-Lissen, L. Gonzalez-Manzano, J. M. de Fuentes,
N. Anciaux, and J. Garcia-Alfaro, “Lumia: Linear probing
for unimodal and multimodal membership inference attacks
leveraging internal llm states,” 2025. [Online]. Available:
https://arxiv.org/abs/2411.19876
[26] R. Shokri, M. Stronati, C. Song, and V . Shmatikov, “Member-
ship inference attacks against machine learning models,” in
2017 IEEE Symposium on Security and Privacy (SP) , 2017,
pp. 3–18.
[27] C. A. Choquette-Choo, F. Tramer, N. Carlini, and N. Papernot,
“Label-only membership inference attacks,” in Proceedings
of the 38th International Conference on Machine Learning ,
ser. Proceedings of Machine Learning Research, M. Meila
and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp.
1964–1974. [Online]. Available: https://proceedings.mlr.press/
v139/choquette-choo21a.html
[28] S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha, “ Privacy
Risk in Machine Learning: Analyzing the Connection to
Overfitting ,” in 2018 IEEE 31st Computer Security Foundations
Symposium (CSF) . Los Alamitos, CA, USA: IEEE Computer
Society, Jul. 2018, pp. 268–282. [Online]. Available:
https://doi.ieeecomputersociety.org/10.1109/CSF.2018.00027
[29] A. Naseh, Y . Peng, A. Suri, H. Chaudhari, A. Oprea,
and A. Houmansadr. Riddle Me This! Stealthy Membership
Inference for Retrieval-Augmented Generation. [Online].
Available: http://arxiv.org/abs/2502.00306
[30] C. Jiang, X. Pan, G. Hong, C. Bao, and M. Yang. RAG-Thief:
Scalable Extraction of Private Data from Retrieval-Augmented
Generation Applications with Agent-based Attacks. [Online].
Available: http://arxiv.org/abs/2411.14110
[31] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu,
S. Edunov, D. Chen, and W.-t. Yih, “Dense passage retrieval
for open-domain question answering,” in Proceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , B. Webber, T. Cohn, Y . He,
and Y . Liu, Eds. Online: Association for Computational
Linguistics, Nov. 2020, pp. 6769–6781. [Online]. Available:
https://aclanthology.org/2020.emnlp-main.550/
[32] S. Robertson, S. Walker, S. Jones, M. Hancock-Beaulieu, and
M. Gatford, “Okapi at trec-3.” 01 1994, pp. 0–.
[33] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh,
S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark,
G. Krueger, and I. Sutskever, “Learning transferable visual
models from natural language supervision,” 2021. [Online].
Available: https://arxiv.org/abs/2103.00020
[34] M. T. Ribeiro, S. Singh, and C. Guestrin, “"why should i
trust you?": Explaining the predictions of any classifier,” 2016.
[Online]. Available: https://arxiv.org/abs/1602.04938
[35] S. M. Lundberg and S.-I. Lee, “A unified approach to inter-
preting model predictions,” in Advances in Neural Information
Processing Systems , I. Guyon, U. V . Luxburg, S. Bengio,
H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett,
Eds., vol. 30. Curran Associates, Inc., 2017. [Online].
Available: https://proceedings.neurips.cc/paper_files/paper/
2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
[36] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam,
D. Parikh, and D. Batra, “Grad-cam: Visual explanations from
deep networks via gradient-based localization,” International
Journal of Computer Vision , vol. 128, no. 2, p. 336–359,
Oct. 2019. [Online]. Available: http://dx.doi.org/10.1007/
s11263-019-01228-7
[37] DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, B. Wu,
C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Guo,
D. Yang, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao,
G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Zhang,H. Ding, H. Xin, H. Gao, H. Li, H. Qu, J. L. Cai, J. Liang,
J. Guo, J. Ni, J. Li, J. Wang, J. Chen, J. Chen, J. Yuan,
J. Qiu, J. Li, J. Song, K. Dong, K. Hu, K. Gao, K. Guan,
K. Huang, K. Yu, L. Wang, L. Zhang, L. Xu, L. Xia, L. Zhao,
L. Wang, L. Zhang, M. Li, M. Wang, M. Zhang, M. Zhang,
M. Tang, M. Li, N. Tian, P. Huang, P. Wang, P. Zhang,
Q. Wang, Q. Zhu, Q. Chen, Q. Du, R. J. Chen, R. L. Jin,
R. Ge, R. Zhang, R. Pan, R. Wang, R. Xu, R. Zhang, R. Chen,
S. S. Li, S. Lu, S. Zhou, S. Chen, S. Wu, S. Ye, S. Ye,
S. Ma, S. Wang, S. Zhou, S. Yu, S. Zhou, S. Pan, T. Wang,
T. Yun, T. Pei, T. Sun, W. L. Xiao, W. Zeng, W. Zhao, W. An,
W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, X. Q. Li,
X. Jin, X. Wang, X. Bi, X. Liu, X. Wang, X. Shen, X. Chen,
X. Zhang, X. Chen, X. Nie, X. Sun, X. Wang, X. Cheng,
X. Liu, X. Xie, X. Liu, X. Yu, X. Song, X. Shan, X. Zhou,
X. Yang, X. Li, X. Su, X. Lin, Y . K. Li, Y . Q. Wang, Y . X.
Wei, Y . X. Zhu, Y . Zhang, Y . Xu, Y . Xu, Y . Huang, Y . Li,
Y . Zhao, Y . Sun, Y . Li, Y . Wang, Y . Yu, Y . Zheng, Y . Zhang,
Y . Shi, Y . Xiong, Y . He, Y . Tang, Y . Piao, Y . Wang, Y . Tan,
Y . Ma, Y . Liu, Y . Guo, Y . Wu, Y . Ou, Y . Zhu, Y . Wang,
Y . Gong, Y . Zou, Y . He, Y . Zha, Y . Xiong, Y . Ma, Y . Yan,
Y . Luo, Y . You, Y . Liu, Y . Zhou, Z. F. Wu, Z. Z. Ren, Z. Ren,
Z. Sha, Z. Fu, Z. Xu, Z. Huang, Z. Zhang, Z. Xie, Z. Zhang,
Z. Hao, Z. Gou, Z. Ma, Z. Yan, Z. Shao, Z. Xu, Z. Wu,
Z. Zhang, Z. Li, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song,
Z. Gao, and Z. Pan, “Deepseek-v3 technical report,” 2025.
[Online]. Available: https://arxiv.org/abs/2412.19437
[38] X. Zheng, Y . Li, H. Chu, Y . Feng, X. Ma, J. Luo,
J. Guo, H. Qin, M. Magno, and X. Liu, “An empirical
study of qwen3 quantization,” 2025. [Online]. Available:
https://arxiv.org/abs/2505.02214
[39] X. Chen, Z. Wu, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu,
and C. Ruan, “Janus-pro: Unified multimodal understanding
and generation with data and model scaling,” 2025. [Online].
Available: https://arxiv.org/abs/2501.17811
[40] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis,
L. Zettlemoyer, and W.-t. Yih, “REPLUG: Retrieval-augmented
black-box language models,” in Proceedings of the 2024
Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers) , K. Duh, H. Gomez, and S. Bethard,
Eds. Mexico City, Mexico: Association for Computational
Linguistics, Jun. 2024, pp. 8371–8384. [Online]. Available:
https://aclanthology.org/2024.naacl-long.463/
[41] J. Wang, Y . He, C. Kang, S. Xiang, and C. Pan, “Image-text
cross-modal retrieval via modality-specific feature learning,” in
Proceedings of the 5th ACM on International Conference on
Multimedia Retrieval , ser. ICMR ’15. New York, NY , USA:
Association for Computing Machinery, 2015, p. 347–354.
[Online]. Available: https://doi.org/10.1145/2671188.2749341
[42] Y . Liu, Y . Guo, E. M. Bakker, and M. S. Lew, “Learning a
recurrent residual fusion network for multimodal matching,”
in2017 IEEE International Conference on Computer Vision
(ICCV) , 2017, pp. 4127–4136.
[43] LlamaIndex Contributors, “Introduction,” https:
//docs.llamaindex.ai/en/stable/#introduction, 2024, accessed:
2025-05-13.
[44] D. Van Veen, C. Van Uden, L. Blankemeier, J.-B. Delbrouck,
A. Aali, C. Bluethgen, A. Pareek, M. Polacin, E. P.
Reis, A. Seehofnerová, N. Rohatgi, P. Hosamani, W. Collins,
N. Ahuja, C. P. Langlotz, J. Hom, S. Gatidis, J. Pauly, and A. S.
Chaudhari, “Adapted large language models can outperform
medical experts in clinical text summarization,” Nature
Medicine , vol. 30, no. 4, p. 1134–1142, Feb. 2024. [Online].
Available: http://dx.doi.org/10.1038/s41591-024-02855-5
[45] M. Liu, S. Zhang, and C. Long, “Mask-based membership
inference attacks for retrieval-augmented generation,” 2025.
[Online]. Available: https://arxiv.org/abs/2410.20142
[46] S. Jin, X. Pang, Z. Wang, H. Wang, J. Du, J. Hu, and K. Ren,
“Safeguarding llm embeddings in end-cloud collaboration
via entropy-driven perturbation,” 2025. [Online]. Available:
https://arxiv.org/abs/2503.12896
[47] Z. Yuan, Q. Jin, C. Tan, Z. Zhao, H. Yuan, F. Huang, and
S. Huang, “Ramm: Retrieval-augmented biomedical visual
question answering with multi-modal pre-training,” 2023.
[Online]. Available: https://arxiv.org/abs/2303.00534
[48] W. Fu, H. Wang, C. Gao, G. Liu, Y . Li, and T. Jiang, “MIA-

tuner: Adapting large language models as pre-training text
detector,” in Proceedings of the AAAI Conference on Artificial
Intelligence , Philadelphia, Pennsylvania, USA, 2025.
[49] W. Shi, A. Ajith, M. Xia, Y . Huang, D. Liu, T. Blevins, D. Chen,
and L. Zettlemoyer, “Detecting pretraining data from large
language models,” 2023.
[50] J. Zhang, J. Sun, E. Yeats, Y . Ouyang, M. Kuo,
J. Zhang, H. F. Yang, and H. Li, “Min-k%++: Improved
baseline for pre-training data detection from large language
models,” in The Thirteenth International Conference on
Learning Representations , 2025. [Online]. Available: https:
//openreview.net/forum?id=ZGkfoufDaU
[51] Z. Li, Y . Wu, Y . Chen, F. Tonin, E. A. Rocamora, and V . Cevher,
“Membership Inference Attacks against Large Vision-Language
Models,” Nov. 2024.
[52] A. Salem, Y . Zhang, M. Humbert, P. Berrang, M. Fritz, and
M. Backes, “Ml-leaks: Model and data independent membership
inference attacks and defenses on machine learning models,”
2018. [Online]. Available: https://arxiv.org/abs/1806.01246
[53] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and
F. Tramer, “Membership inference attacks from first principles,”
2022. [Online]. Available: https://arxiv.org/abs/2112.03570
[54] Y . Hu, Z. Li, Z. Liu, Y . Zhang, Z. Qin, K. Ren, and C. Chen,
“Membership inference attacks against vision-language models,”
2025. [Online]. Available: https://arxiv.org/abs/2501.18624
[55] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y . Hou, Y . Min,
B. Zhang, J. Zhang, Z. Dong, Y . Du, C. Yang, Y . Chen, Z. Chen,
J. Jiang, R. Ren, Y . Li, X. Tang, Z. Liu, P. Liu, J.-Y . Nie, and
J.-R. Wen, “A Survey of Large Language Models,” Oct. 2024.
[56] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun,
M. Wang, and H. Wang, “Retrieval-augmented generation for
large language models: A survey,” 2024. [Online]. Available:
https://arxiv.org/abs/2312.10997
[57] L. Mei, S. Mo, Z. Yang, and C. Chen, “A survey of multimodal
retrieval-augmented generation,” 2025. [Online]. Available:
https://arxiv.org/abs/2504.08748
[58] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn,
X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold,
S. Gelly, J. Uszkoreit, and N. Houlsby, “An Image is Worth
16x16 Words: Transformers for Image Recognition at Scale,”
Jun. 2021.
[59] S. Yin, C. Fu, S. Zhao, K. Li, X. Sun, T. Xu, and E. Chen,
“A survey on multimodal large language models,” National
Science Review , vol. 11, no. 12, Nov. 2024. [Online]. Available:
http://dx.doi.org/10.1093/nsr/nwae403
[60] H. Wu and Y . Cao, “Membership inference attacks on
large-scale models: A survey,” 2025. [Online]. Available:
https://arxiv.org/abs/2503.19338
[61] L. Bai, H. Hu, Q. Ye, H. Li, L. Wang, and J. Xu, “Membership
inference attacks and defenses in federated learning: A survey,”
2024. [Online]. Available: https://arxiv.org/abs/2412.06157
[62] X. Song, H. Lin, H. Wen, B. Hou, M. Xu, and L. Nie, “A
comprehensive survey on composed image retrieval,” 2025.
[Online]. Available: https://arxiv.org/abs/2502.18495
[63] A. A. Khan, M. T. Hasan, K. K. Kemell, J. Rasku, and
P. Abrahamsson, “Developing retrieval augmented generation
(rag) based llm systems from pdfs: An experience report,”
2024. [Online]. Available: https://arxiv.org/abs/2410.15944
[64] S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li,
R. Hu, T. Zhang, F. Wu, and G. Wang, “Instruction tuning for
large language models: A survey,” 2024. [Online]. Available:
https://arxiv.org/abs/2308.10792
[65] M. Anderson, G. Amit, and A. Goldsteen, “Is my data
in your retrieval database? membership inference attacks
against retrieval augmented generation,” in Proceedings
of the 11th International Conference on Information
Systems Security and Privacy . SCITEPRESS - Science
and Technology Publications, 2025, p. 474–485. [Online].
Available: http://dx.doi.org/10.5220/0013108300003899
[66] Y . Cai and G. Wunder, “On gradient-like explanation
under a black-box setting: When black-box explanations
become as good as white-box,” 2024. [Online]. Available:
https://arxiv.org/abs/2308.09381
[67] Z. Zhao and B. Shan, “Reagent: A model-agnostic feature
attribution method for generative language models,” 2024.
[Online]. Available: https://arxiv.org/abs/2402.00794[68] T. Idé and N. Abe, “Generative perturbation analysis for
probabilistic black-box anomaly attribution,” in Proceedings of
the 29th ACM SIGKDD Conference on Knowledge Discovery
and Data Mining , ser. KDD ’23. ACM, Aug. 2023, p. 845–856.
[Online]. Available: http://dx.doi.org/10.1145/3580305.3599365
[69] E. Zaher, M. Trzaskowski, Q. Nguyen, and F. Roosta,
“Manifold integrated gradients: Riemannian geometry for
feature attribution,” 2024. [Online]. Available: https://arxiv.org/
abs/2405.09800
[70] L. Simpson, F. Costanza, K. Millar, A. Cheng, C.-C. Lim,
and H. G. Chew, “Tangentially aligned integrated gradients
for user-friendly explanations,” 2025. [Online]. Available:
https://arxiv.org/abs/2503.08240
[71] D. Chen, N. Yu, Y . Zhang, and M. Fritz, “Gan-leaks: A
taxonomy of membership inference attacks against generative
models,” in Proceedings of the 2020 ACM SIGSAC Conference
on Computer and Communications Security , ser. CCS
’20. ACM, Oct. 2020, p. 343–362. [Online]. Available:
http://dx.doi.org/10.1145/3372297.3417238
[72] K. S. Liu, C. Xiao, B. Li, and J. Gao, “Performing
co-membership attacks against deep generative models,” 2019.
[Online]. Available: https://arxiv.org/abs/1805.09898
[73] S. Zhou, “Semidefinite programming relaxations and debiasing
for maxcut-based clustering,” 2025. [Online]. Available:
https://arxiv.org/abs/2401.10927
[74] C. Luo, S. Song, W. Xie, M. Spitale, Z. Ge, L. Shen, and
H. Gunes, “Reactface: Online multiple appropriate facial
reaction generation in dyadic interactions,” 2024. [Online].
Available: https://arxiv.org/abs/2305.15748
[75] H. Hu, Z. Salcic, L. Sun, G. Dobbie, P. S. Yu, and X. Zhang,
“Membership inference attacks on machine learning: A survey,”
2022. [Online]. Available: https://arxiv.org/abs/2103.07853
[76] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess,
R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei,
“Scaling laws for neural language models,” 2020. [Online].
Available: https://arxiv.org/abs/2001.08361
[77] W. Shi, A. Ajith, M. Xia, Y . Huang, D. Liu, T. Blevins,
D. Chen, and L. Zettlemoyer, “Detecting pretraining data
from large language models,” in The Twelfth International
Conference on Learning Representations , 2024. [Online].
Available: https://openreview.net/forum?id=zWqr3MQuNs
[78] OpenAI, J. Achiam, and etc, “Gpt-4 technical report,” 2024.
[Online]. Available: https://arxiv.org/abs/2303.08774
[79] G. Comanici and e. Eric Bieber, “Gemini 2.5: Pushing the
frontier with advanced reasoning, multimodality, long context,
and next generation agentic capabilities,” 2025. [Online].
Available: https://arxiv.org/abs/2507.06261
[80] S. Liang, M. Zhu, A. Liu, B. Wu, X. Cao, and E.-C. Chang,
“Badclip: Dual-embedding guided backdoor attack on multi-
modal contrastive learning,” arXiv preprint arXiv:2311.12075 ,
2023.
[81] S. Liang, J. Liang, T. Pang, C. Du, A. Liu, M. Zhu, X. Cao,
and D. Tao, “Revisiting backdoor attacks against large vision-
language models from domain shift,” in Proceedings of the
Computer Vision and Pattern Recognition Conference , 2025,
pp. 9477–9486.
[82] L. Lu, S. Pang, S. Liang, H. Zhu, X. Zeng, A. Liu, Y . Liu,
and Y . Zhou, “Adversarial training for multimodal large
language models against jailbreak attacks,” arXiv preprint
arXiv:2503.04833 , 2025.
[83] S. Liang, J. Liu, J. Zhai, T. Fang, R. Tu, A. Liu, X. Cao,
and D. Tao, “T2vshield: Model-agnostic jailbreak defense for
text-to-video models,” arXiv preprint arXiv:2504.15512 , 2025.
[84] Z. Ying, A. Liu, T. Zhang, Z. Yu, S. Liang, X. Liu, and D. Tao,
“Jailbreak vision language models via bi-modal adversarial
prompt,” arXiv preprint arXiv:2406.04031 , 2024.
[85] Z. Ying, S. Wu, R. Hao, P. Ying, S. Sun, P. Chen, J. Chen,
H. Du, K. Shen, S. Wu et al. , “Pushing the limits of safety: A
technical report on the atlas challenge 2025,” arXiv preprint
arXiv:2506.12430 , 2025.
[86] S. Liang, A. Liu, J. Liang, L. Li, Y . Bai, and X. Cao, “Imitated
detectors: Stealing knowledge of black-box object detectors,”
inProceedings of the 30th ACM International Conference on
Multimedia , 2022.
[87] P. enhancing face obfuscation guided by semantic-aware at-
tribution maps, “Privacy-enhancing face obfuscation guided

by semantic-aware attribution maps,” IEEE Transactions on
Information Forensics and Security , 2023.
[88] Y . Xiao, A. Liu, Q. Cheng, Z. Yin, S. Liang, J. Li, J. Shao,
X. Liu, and D. Tao, “Genderbias- \emph{VL}: Benchmarking
gender bias in vision language models via counterfactual
probing,” arXiv preprint arXiv:2407.00600 , 2024.
[89] Y . Xiao, A. Liu, S. Liang, X. Liu, and D. Tao, “Fairness
mediator: Neutralize stereotype associations to mitigate bias
in large language models,” arXiv preprint arXiv:2504.07787 ,
2025.
APPENDIX
Fig. 7 provides supplementary heatmaps for the
LLM experiments described in the main text, visually
comparing the performance of different MIA methods
across multiple models and evaluation metrics.

Llama-2 7BLlama-3.1 8B Qwen2.5 7BPETEL
Mink++
MIN-K%PROB (K=10%)
MIN-K%PROB (K=20%)
MIN-K%PROB (K=30%)
SMA0.5230 0.5332 0.5088
0.5190 0.5200 0.5890
0.5032 0.5126 0.5000
0.5015 0.5074 0.5072
0.5029 0.5045 0.5035
0.8624 0.6351 0.6484WikiMIA / ACC
Llama-2 7BLlama-3.1 8B Qwen2.5 7BPETEL
Mink++
MIN-K%PROB (K=10%)
MIN-K%PROB (K=20%)
MIN-K%PROB (K=30%)
SMA0.5230 0.5332 0.5088
0.5190 0.5200 0.5889
0.5032 0.5126 0.5000
0.5015 0.5074 0.5072
0.5029 0.5045 0.5035
0.5882 0.7895 0.6700WikiMIA / Coverage
Llama-2 7BLlama-3.1 8B Qwen2.5 7BPETEL
Mink++
MIN-K%PROB (K=10%)
MIN-K%PROB (K=20%)
MIN-K%PROB (K=30%)
SMA0.5741 0.6280 0.4968
0.5010 0.5960 0.5540
0.5370 0.5556 0.5918
0.5322 0.5378 0.5676
0.5314 0.5531 0.5523
0.6730 0.7233 0.5705WikiMIA-24 / ACC
Llama-2 7BLlama-3.1 8B Qwen2.5 7BPETEL
Mink++
MIN-K%PROB (K=10%)
MIN-K%PROB (K=20%)
MIN-K%PROB (K=30%)
SMA0.5741 0.6280 0.4968
0.5010 0.5960 0.5540
0.5370 0.5556 0.5918
0.5322 0.5378 0.5676
0.5314 0.5531 0.5523
0.6000 0.6667 0.7500WikiMIA-24 / Coverage
0.20.30.40.50.60.70.80.9
ScoreAnnotated Heatmaps (ACC & Coverage) for swj0419 and wfju99Fig. 7: Heat map of MIA methods under RAG settings across six LLM models