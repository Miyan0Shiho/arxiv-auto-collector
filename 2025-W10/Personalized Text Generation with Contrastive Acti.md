# Personalized Text Generation with Contrastive Activation Steering

**Authors**: Jinghao Zhang, Yuting Liu, Wenjie Wang, Qiang Liu, Shu Wu, Liang Wang, Tat-Seng Chua

**Published**: 2025-03-07 08:07:15

**PDF URL**: [http://arxiv.org/pdf/2503.05213v1](http://arxiv.org/pdf/2503.05213v1)

## Abstract
Personalized text generation aims to infer users' writing style preferences
from their historical texts and generate outputs that faithfully reflect these
stylistic characteristics. Existing solutions primarily adopt two paradigms:
retrieval-augmented generation (RAG) and parameter-efficient fine-tuning
(PEFT). While these approaches have advanced the field, they suffer from two
critical limitations: (1) the entanglement of content semantics and stylistic
patterns in historical texts impedes accurate modeling of user-specific writing
preferences; and (2) scalability challenges arising from both RAG's inference
latency by retrieval operations and PEFT's parameter storage requirements for
per user model. To overcome these limitations, we propose StyleVector, a
training-free framework that disentangles and represents personalized writing
style as a vector in LLM's activation space, enabling style-steered generation
during inference without requiring costly retrieval or parameter storage.
Comprehensive experiments demonstrate that our framework achieves a significant
8% relative improvement in personalized generation while reducing storage
requirements by 1700 times over PEFT method.

## Full Text


<!-- PDF content starts -->

Personalized Text Generation with Contrastive Activation Steering
Jinghao Zhang1,2, Yuting Liu3, Wenjie Wang4,
Qiang Liu1,2, Shu Wu1,2, Liang Wang1,2, Tat-Seng Chua5
1NLPR, Institute of Automation, Chinese Academy of Sciences,
2School of Artificial Intelligence, University of Chinese Academy of Sciences,
3Northeastern University, China,4University of Science and Technology of China,
5National University of Singapore
jinghao.zhang@cripac.ia.ac.cn ,{qiang.liu,shu.wu,wangliang}@nlpr.ia.ac.cn ,
yutingliu@stumail.neu.edu.cn ,wenjiewang96@gmail.com ,dcscts@nus.edu.sg ,
Abstract
Personalized text generation aims to infer users’
writing style preferences from their historical
texts and generate outputs that faithfully reflect
these stylistic characteristics. Existing solu-
tions primarily adopt two paradigms: retrieval-
augmented generation (RAG) and parameter-
efficient fine-tuning (PEFT). While these ap-
proaches have advanced the field, they suffer
from two critical limitations: (1) the entangle-
ment of content semantics and stylistic patterns
in historical texts impedes accurate modeling
of user-specific writing preferences; and (2)
scalability challenges arising from both RAG’s
inference latency by retrieval operations and
PEFT’s parameter storage requirements for per
user model. To overcome these limitations, we
propose StyleVector, a training-free framework
that disentangles and represents personalized
writing style as a vector in LLM’s activation
space, enabling style-steered generation dur-
ing inference without requiring costly retrieval
or parameter storage. Comprehensive experi-
ments demonstrate that our framework achieves
a significant 8% relative improvement in per-
sonalized generation while reducing storage
requirements by 1700 ×over PEFT method.
1 Introduction
Large language models (LLMs) have demonstrated
unprecedented capabilities in text generation and
complex reasoning through pre-training on massive
corpora. However, these models still function as
"one-size-fits-all" systems, optimized for average-
case scenarios, and fail to adapt to individual users’
unique preferences. The increasing demand for
personalized AI assistants highlights the need to
customize LLMs to better align with the specific
preference of each user (Kirk et al., 2024; Chen
et al., 2024a; Au et al., 2025; Cai et al., 2024; Jang
et al., 2023; Lin et al., 2024; Zhang et al., 2024b;
Liu et al., 2024a; Zhu et al., 2025).
Personalized text generation has emerged as acritical research frontier (Salemi et al., 2024b; Ku-
mar et al., 2024; Alhafni et al., 2024; Chen and
Moscholios, 2024). Consider a scenario where
given an email subject xand a user u’s historical
subject-email pairs Pu, the system must infer the
user’s writing style from Puto generate stylisti-
cally consistent emails. Current approaches pre-
dominantly fall into two categories: (1) Retrieval-
augmented generation (RAG) methods (Zhang
et al., 2023; Salemi and Zamani, 2024a,b), which
enhance input prompts by retrieving personalized
information from Pu, and (2) parameter-efficient
fine-tuning (PEFT) methods (Salemi and Zamani,
2024a; Tan et al., 2024a; Zhuang et al., 2024),
which train per-user adapter modules using Pu. De-
spite their merits, these methods suffer from criti-
cal limitations: (a) The inherent entanglement of
user-agnostic content semantics anduser-specific
stylistic patterns in historical data impedes accu-
rate style inference. (b) The substantial inference
latency of RAG’s retrieval mechanisms and stor-
age requirements of PEFT’s per-user parameters
renders these solutions impractical for real-world
deployment at scale.
Recent advances in activation engineering (Zou
et al., 2023; Liu et al., 2023; Rimsky et al., 2024)
reveal that LLMs encode features and concepts as
linear directions in hidden activation space. These
directional vectors can effectively steer model be-
havior through simple linear interventions during
inference. Building on these insights, we reveal
thatuser-specific writing styles can similarly be rep-
resented as directional vectors in activation space.
This leads to an elegant solution for personalized
generation: (1) By contrasting the hidden activa-
tions between user-authentic responses (contain-
ing both content and style) and model-generated
generic responses (content-preserving but style-
agnostic), we can derive "style vector" that con-
tains personal stylistic signatures. (2) The derived
style vector could be used to steer model generation
1arXiv:2503.05213v1  [cs.CL]  7 Mar 2025

towards user-specific writing styles through sim-
ple linear interventions during inference, without
parameter updates or extensive retrieval.
To this end, we present StyleVector, an efficient,
training-free framework that only requires storing
one vector for each user to achieve high-quality
personalized text generation. As shown in Fig-
ure 1, our methodology comprises three key steps:
(1) generating style-agnostic responses for histor-
ical inputs using a base LLM, (2) deriving style
vectors by contrasting hidden activations between
authentic user responses and generated neutral re-
sponses, and (3) steering generation during infer-
ence through linear activation interventions with
the obtained style vectors.
Comprehensive evaluations on LaMP (Salemi
et al., 2024b) and LongLaMP (Kumar et al., 2024)
benchmarks for short- and long-form personaliza-
tion respectively demonstrate our method’s effec-
tiveness. Experimental results show that StyleVec-
tor achieves 8% relative improvement in personal-
ization quality while reducing storage requirements
by 1700 ×over PEFT-based methods.
Our contributions are summarized as follows:
•We reveal that user-specific writing styles
can be represented as linear directions in ac-
tivation space through contrastive analysis
between authentic user responses and style-
agnostic model outputs.
•We propose a training-free personalized gen-
eration framework through simple linear acti-
vation interventions, requiring only 2|Pu|for-
ward passes (zero back-propagation) per user
and compresses personalized information into
a single vector.
•Experiments on both short- and long-form per-
sonalization benchmarks show the effective-
ness of our method, while significantly reduc-
ing storage and inference latency compared to
retrieval-based and adapter-based approaches.
2 Preliminaries
2.1 Problem Formulation
Personalized text generation aims to infer the user’s
writing style preferences based on the text created
from their history and generate outputs that align
with those preferences. Formally, for each user
u: given an input prompt xspecifying task re-
quirements (e.g., an email subject), the languagemodel Mgenerates output ˆy=M(x, Pu)con-
ditioned on both xand the user’s historical data
Pu={(xi, yi)}|Pu|
i=1, where each pair (xi, yi)rep-
resents previous interactions (e.g., subject-email
pairs). The ground truth output yrepresents the
user-customized response that reflects u’s unique
writing style (e.g., personalized email drafts).
2.2 Base Solutions
Retrieval-Augmented Generation (RAG)
RAG-based approaches achieve personalization
through context-aware retrieval. Given input x,
the system retrieves kmost relevant historical re-
sponses from Puusing retriever R, then generates
personalized responses by combining retrieved
documents R(x, Pu, k)with the input prompt:
ˆy=M(x, R(x, Pu, k)) (1)
Parameter-Efficient Fine-Tuning (PEFT)
PEFT methods customize LLMs by training
lightweight adapters (e.g., LoRA (Hu et al., 2021))
on user-specific data while keeping base model
parameters frozen (Tan et al., 2024b). For each
useru, a distinct adapter θuis trained via:
θ∗
u= arg min
θX
(xi,yi)∈PuL(M(xi;θ), yi)(2)
where L(·)denotes the sequence-to-sequence
cross-entropy loss. During inference:
ˆy=M(x;θu) (3)
2.3 Limitations of Base Solutions
Existing approaches face the following two funda-
mental constraints.
Entangled Style-Content Representation Both
RAG and PEFT methods process historical entries
pias monolithic units. However, each historical
entry contains both the user-agnostic semantics
corresponding to the input xiand the user-specific
writing style (Fisher et al., 2024). This entangle-
ment impedes accurate style modeling, particularly
for RAG methods that retrieve documents based on
semantic matching, and the semantic-dominated
retrieved contexts lead to style dilution (see Sec-
tion 4.5 for examples).
Scalability Bottlenecks As summarized in Ta-
ble 1, existing methods suffer from three critical
2

(c) Steering Personalized Generation……𝛼!,#ℓ𝛼%,#ℓ
Style Vector 𝒔𝒖ℓ𝛼!,#ℓ,𝛼!,#ℓ#'(|*!|ActivationsExtract
(b) Extracting Style VectorTransformer Layer 𝟏Transformer Layer 𝑳Transformer Layer ℓ(𝑥#,𝑦#)(𝒙𝒊,)𝒚𝒊)
Transformer Layer 𝟏Transformer Layer 𝑳Transformer Layer ℓ……
Personalized Output !𝒚
Steer at Layer ℓ(𝑥,,𝑦,)
(𝑥|*!|,𝑦|*!|)
History 𝑷𝐮…General LLMs
(𝒙𝟏,)𝒚𝟏),…,(𝒙𝑷𝒖,)𝒚|𝑷𝒖|)(a)Generating Style-Agnostic Response
Input: 𝒙
Input: 𝒙
Figure 1: The overall framework of StyleVector.
Metric RAG PEFT StyleVector
Training Time/User O(|Pu|)* O(|Pu|) O(|Pu|)*
Latency/Query O(|Pu|)O(Load+Merge ) O(1)
Storage/User O(|Pu|D) O(rDL) O(D)
* Training-free. Denotes pre-processing cost.
Table 1: System Efficiency Comparison.
scalability constraints: training time, inference la-
tency and storage requirement. Due to space con-
straints, we have placed the complexity analysis of
the baseline in the Appendix D. These compounded
costs render existing methods challenging for real-
world deployment at scale (Salemi and Zamani,
2024a). We also provide empirical cost compar-
isons in Section 4.2.
3 Method
Our StyleVector framework aims to identify a user-
specific style vector through contrastive activation
analysis, then steer LLM generation via targeted
activation intervention. As shown in Figure 1, the
process comprises three stages: (1) Style-agnostic
response generation, (2) Style vector extraction
through contrastive activation analysis, and (3) Ac-
tivation steering during inference.
3.1 Generating Style-Agnostic Response
Given a user uwith historical interactions Pu=
{(xi, yi)}|Pu|
i=1, where xidenotes an input and yi
the user-authored response, we first generate style-
agnostic responses {ˆyi}|Pu|
i=1by instructing any gen-eral LLM Mgwith the input xi:
ˆyi=Mg(xi). (4)
Please note that the general LLM Mgis designed
to generate responses that are independent of the
user’s style and only related to the input seman-
tics. It does not necessarily need to be the same
as a personalized large model M; it can be any
model, whether open-source or closed-source. We
conduct experiments in Appendix C.2 to show the
robustness on Mgof our method.
In this way, yidenotes the user-authentic content,
containing both content semantics and stylistic pat-
terns. Model-generated generic ˆyionly preserves
content semantics related to xibut stripped of per-
sonal style. By contrasting yiandˆyi, we could
disentangle user-specific style from user-agnostic
semantics.
3.2 Extracting Style Vector
We extract style vectors through contrastive anal-
ysis of hidden activations. Let hℓ(r)∈Rddenote
the hidden states of the last token at layer ℓwhen
processing text r. The positive and negative activa-
tions of history piece ican be represented as:
aℓ
p,i=hℓ(xi⊕yi), aℓ
n,i=hℓ(xi⊕ˆyi),(5)
where⊕denotes concatenation the strings of input
and output. Then we can obtain the user style
vector by considering all history pieces:
sℓ
u=f([aℓ
p,i, aℓ
n,i]|Pu|
i=0), (6)
3

where f(·)is an extracting function that takes all
the positive and negative activations and returns a
single style vector. The essence of the function fis
to find a direction in the activation space that points
from style-agnostic samples to user-authentic sam-
ples. There could be many possible functions, and
here we discuss three strategies:
1) Mean Difference. The most straightforward
approach computes the mean difference between
positive and negative activations:
sℓ
u=1
|Pu||Pu|X
i=1(aℓ
p,i−aℓ
n,i). (7)
sℓ
urepresents the average direction in the activation
space that distinguishes user-specific style patterns
from style-agnostic ones.
2) Logistic Regression. We can also employ
logistic regression to find a direction that best
separates positive and negative examples. Let
X= [aℓ
p,1;...;aℓ
p,|Pu|;aℓ
n,1;...;aℓ
n,|Pu|]be the ma-
trix of all activations, and y= [1, ...,1,−1, ...,−1]
be the corresponding labels. The style vector is
obtained by:
w= arg min
wX
ilog(1 + e−yiXiw), (8)
where wdenotes the normal vector to the decision
boundary. When moving in the direction of w, the
model’s predicted probability of being a positive
sample will monotonically increase. We use the
normalized was the style vector:
sℓ
u=w
∥w∥2, (9)
3) Principal Component Analysis. The Princi-
pal Component Analysis (PCA) approach finds the
steering vector sℓ
uby identifying the direction of
maximum variance in the differences between pos-
itive and negative activations. Let ∆i=aℓ
p,i−aℓ
n,i
be the difference between the i-th pair of posi-
tive and negative activations. PCA computes the
first principal component of the set {∆i}∪{− ∆i},
which can be formulated as:
sℓ
u= arg max
v:∥v∥=1|Pu|X
i=1(∆T
iv)2. (10)
This formulation ensures that: 1. The resulting
vector sℓ
uhas unit norm 2. It maximizes the pro-
jected variance of the activation differences 3. Theinclusion of −∆ienforces symmetry around the
origin, making the solution invariant to the choice
of which sample is positive or negative
The solution to this optimization problem is
given by the first eigenvector of the matrixP|Pu|
i=1(∆i∆T
i+ (−∆i)(−∆T
i)), which can be effi-
ciently computed using Singular Value Decompo-
sition (SVD).
3.3 Steering Personalized Generation
After obtaining the style vector, we can steer the
model’s generation by intervene the hidden states
at inference time. In this work, we only consider
intervene one layer ℓ, which could be selected via
validation set. Let hℓ(x)denote the hidden states
at layer ℓwhen processing input x. We use the
most straightforward approach directly adds the
scaled style vector to the hidden states of the token
position t:
h′
ℓ(x)t=hℓ(x)t+αsℓ
u (11)
where αis a scaling factor controlling the strength
of steering. Following (Rimsky et al., 2024), we
intervene every token position of the generated text
after the end of the initial prompt t≥ |x|. We also
try different positions experimentally in Section
4.2.
Efficiency Analysis For pre-processing, our
method requires only 2|Pu|forward passes of
LLMs to obtain activations and the style vector
extracting is negligible when compared with the
cost of LLMs. For storage, the final style vec-
torsℓ
uonly requires D-dimensional vector storage.
For additional inference latency, activation steering
only introduces Delement-wise addition overhead.
The complexity analysis is summarized in Table 1.
4 Experiments
4.1 Experimental Setup
Benchmarks and Evaluation We adopt LaMP
benchmark (Salemi et al., 2024b) and LongLaMP
benchmark (Kumar et al., 2024), which are de-
signed for evaluating short-form and long-form
personalized text generation, respectively. We ex-
clude email generation tasks for both datasets since
it involves private data that we cannot access. We
choose the user split for both benchmarks and the
dataset statistics are presented in Table 4. Follow-
ing previous works (Tan et al., 2024a; Salemi and
Zamani, 2024a), we use ROUGE-L and METEOR
as evaluation metrics.
4

Benchmark MetricNon-personalized RAG-based PEFT-basedOurs Improv.
LLaMA2 BM25 Contriever SFT DPO
LongLaMP: Abstract GenerationROUGE-L 0.2056 0.2020 0.2035 0.2038 0.2020 0.2060 0.2%
METEOR 0.2950 0.2911 0.2922 0.2929 0.2933 0.2973 0.8%
LongLaMP: Topic WritingROUGE-L 0.1299 0.1235 0.1256 0.1303 0.1277 0.1361 4.7%
METEOR 0.1874 0.1782 0.1853 0.1914 0.1901 0.1949 4.0%
LongLaMP: Review GenerationROUGE-L 0.1380 0.1388 0.1391 0.1364 0.1320 0.1448 5.0%
METEOR 0.1614 0.1655 0.1663 0.1574 0.1446 0.1804 11.8%
LaMP: News Headline GenerationROUGE-L 0.0398 0.0403 0.0403 0.0407 0.0401 0.0411 3.2%
METEOR 0.0790 0.0792 0.0807 0.0800 0.7910 0.0809 2.5%
LaMP: Scholarly Title GenerationROUGE-L 0.1086 0.0909 0.0919 0.1100 0.1047 0.1366 25.8%
METEOR 0.2337 0.2066 0.2086 0.2348 0.1930 0.2575 10.2%
LaMP: Tweet ParaphrasingROUGE-L 0.2506 0.2554 0.2571 0.2341 0.2204 0.2827 12.8%
METEOR 0.2588 0.2603 0.2634 0.2503 0.2389 0.3042 17.5%
Table 2: The performance results on LongLaMP and LaMP personalized text generation benchmarks. The best
score is in bold and the second best is underlined .
Baselines We compare our proposed StyleVec-
tor with RAG-based personalization methods and
PEFT-based personalization methods.
For RAG-based personalization, we employ two
widely-used retrievers BM25 (Robertson et al.,
2009) and Contriever (Lei et al., 2023).
For PEFT-based personalization, we fine-tune
user-specific LoRA adapter (Hu et al., 2021) for
each user using their profile Pu= (xi, yi)|Pu|
i=0, us-
ing SFT loss in Equation 2. Additionally, since
we obtained style-agnostic responses, we also em-
ploy DPO loss (Rafailov et al., 2024) to guide the
model to generate user-authentic responses rather
than style-agnostic responses.
Implementation Details We implement our pro-
posed StyleVector and all baselines with Llama-2-
7B-chat (Touvron et al., 2023). For the RAG ap-
proach, we set the number of retrieved documents
k= 2; for the PEFT approach, we set the rank
of LoRA to 8. For StyleVector, unless otherwise
specified, we will use gpt-3.5-turbo to generate
style-neutral responses and employ the simplest
mean difference extracting function. We demon-
strate the performance of using different extracting
functions and general models in Appendix. We
conduct experiments on the validation set to select
the appropriate number of intervention layer ℓand
intervention strength αfor each task. For more
details, please refer to Appendix B.
4.2 Main Results
By comparing our method with the baseline in
terms of generation performance and efficiency, weTask Averaged Cost ↓ SFT RAG Ours
Abstract
GenerationTT/User (s) 131.98 0.64 27.23
IL/Query (s) 22.59 18.90 15.59
IL/5-Query (s) 94.75 96.97 79.23
SS/User (MB) 17.00 0.35 0.01
Review
GenerationTT/User (s) 62.45 0.44 11.65
IL/Query (s) 18.88 8.23 11.75
IL/5-Query (s) 77.33 52.52 59.69
SS/User (MB) 17.00 0.10 0.01
News Headline
GenerationTT/User (s) 123.28 1.22 22.16
IL/Query (s) 25.52 12.47 10.32
IL/5-Query (s) 105.00 78.08 57.80
SS/User (MB) 17.00 0.83 0.01
Scholarly Title
GenerationTT/User (s) 112.31 0.51 22.53
IL/Query (s) 25.43 9.52 10.49
IL/5-Query (s) 104.33 50.68 54.30
SS/User (MB) 17.00 0.26 0.01
Table 3: Comparison of Training Time (TT, for train-
free RAG and StyleVector, represents pre-processing
time), Inference Latency (IL) and Storage Space (SS)
requirements across different methods. The lowest cost
in in bold .
demonstrate that our approach can achieve strong
generation performance while maintaining high ef-
ficiency.
Generation Performance Comparison Table 2
shows the generation performance comparison and
We can observe that:
•StyleVector demonstrates superior perfor-
mance across both short-term and long-term
personalized text generation tasks. Notably,
StyleVector achieves averaged 11% and 8%
relative improvements on ROUGE-L and ME-
TEOR compared with RAG-based methods
5

0510152025300.080.120.160.20METEOR Score
T opic Writing
0510152025300.100.120.140.160.18
Review Generation
0510152025300.180.220.26METEOR Score
T weet Paraphrasing
0510152025300.110.150.190.23
Scholarly Title Generation0.090.110.130.15
0.130.15
ROUGE-L Score
0.230.270.31
0.040.080.120.16
ROUGE-L Score
METEOR Base METEOR ROUGE-L Base ROUGE-LFigure 2: Performance comparison across different in-
tervention layers l.
and PEFT-based methods, respectively.
•Both RAG-based and PEFT-based methods
show unstable performance and cannot consis-
tently improve base model across all tasks.
RAG-based methods are more effective in
tasks with less user history (review genera-
tion and tweet paraphrasing are the two tasks
with the least user history), while PEFT per-
forms better in scenarios with more historical
data as it provides more training texts.
Efficiency Comparison Table 3 shows the scala-
bility comparison, where we implement Contriver
as the retriever of RAG.
•In terms of training time, our method is
training-free and requires only 1/5 of the pre-
processing time compared to SFT. However,
since RAG uses smaller retrievers (e.g., the
Contriever model we use is no larger than
0.1B), RAG’s preprocessing time is the short-
est.
•In terms of inference latency, RAG is faster
on tasks with less user history, but it becomes
significantly slower on tasks with more user
history. SFT takes too long to load and merge
LoRA, making it unsuitable for scenarios that
require frequent updates. Our method is inde-
pendent of user history and does not require
prolonged loading, making it a more versatile
approach.
•In terms of storage space, our method only
requires storing a single vector per user, mak-
ing it unquestionably the most space-efficient,
which occupies about 1/1700 of the space re-
quired by SFT.
0.5
 0.0 0.5 1.0 1.50.140.18METEOR Score
T opic Writing
0.5
 0.0 0.5 1.0 1.50.210.250.29
Review Generation
0.5
 0.0 0.5 1.0 1.5
Multiplier0.180.200.230.250.28METEOR Score
T weets Paraphrasing
0.5
 0.0 0.5 1.0 1.5
Multiplier0.200.220.240.26
Scholarly Title Generation0.100.120.140.16
0.230.250.270.290.31
ROUGE-L Score
0.230.250.270.290.31
0.100.120.140.16
ROUGE-L Score
METEOR Base METEOR ROUGE-L Base ROUGE-LFigure 3: Performance comparison across different in-
tervention strengths α.
4.3 Steering Analysis
Analysis of layers and multipliers The interven-
tion layer ℓand the intervention strength αare two
important hyperparameters of our method. In this
section, we analyze the impact of different values
ofℓandαon generation performance. The results
are shown in Figure 2 and Figure 3, from which we
can observe that:
•The activations controlling the model’s
writing style are typically reflected in the
middle to later layers . As shown in Fig-
ure 2, although there may be subtle differences
across tasks, in general, the most effective in-
tervention occurs when modifying the middle
to later layers of the model (around layer 15
and beyond). Linear probing results in Sec-
tion 4.4 also lead to the similar conclusion.
•Positive intervention can guide the model to
generate in the user’s style, while negative
intervention can push it away from that
style. As shown in Figure 3, when α < 0,
the negative intervention causes the model’s
generated content to drift away from the user’s
style, resulting in a score lower than that of the
non-personalized model. However, if αis too
large, it can cause abnormal activation values,
thereby disrupting the generation process.
4.4 Style Vector Analysis
Probing Study To investigate how writing style
features are encoded in the model’s hidden states,
we conduct a linear probing analysis across differ-
ent layers of the base LLM. For each user u∈ U,
we construct a binary classification task where the
positive samples are the user’s authentic historical
6

0 510 15 20 25 30
Layer0.8500.8750.9000.9250.950Probing AUC
Tweet Paraphrasing
0 510 15 20 25 30
Layer0.97850.97900.97950.9800
Scholarly Title GenerationFigure 4: Probing results on LaMP benchmark.
texts yi∈ P u, and the negative samples are our
framework’s style-agnostic responses ˆyigenerated
for the same input contexts. We extract hidden
states at layer lfor all samples and train a logistic
regression classifier to distinguish between authen-
tic and generated texts. Figure 4 shows the aver-
aged probing results across all users, which reveals
two key findings:
•High Layer-wise Separability. All lay-
ers achieve strong classification performance
(AUC > 0.85), suggesting that user-specific
stylistic patterns are robustly encoded through-
out the network. This confirms our hypothesis
that style information persists in the model’s
internal representations, even when not explic-
itly supervised.
•The activations controlling the model’s
writing style are typically reflected in the
middle to later layers . The AUC increases
with the depth of the layers, which aligns
with our empirical findings in Section 4.3,
where style steering interventions in these
layers yielded optimal generation quality.
The progressive feature refinement suggests
that stylistic attributes are gradually distilled
through the forward pass, reaching maximal
linear separability in higher layers.
4.5 Case Study
To demonstrate the effectiveness of our method,
we analyze a representative case from user_310 in
LaMP: News Headline Generation benchmark in
Figure 5, demonstrating three key insights about
our style vector approach:
•Style vector encodes user preferences . The
highlighted tokens are the top 5 tokens that
most closely match the style vector among all
historical tokens. We can observe that the top-
5 tokens (":", "ips", "for", "What", "Need") in
historical headlines reveal consistent stylistic
patterns of using subtitles and combinationssuch as "tips for" or "what need".
•Style vector can steer personalized gener-
ation . Our method generates "Keeping Your
Teen Safe Online: Tips and Strategies for Par-
ents", which naturally incorporates 3 key style
tokens (":", "ips", "for") while maintaining
content fidelity. However, the generation by
baselines can not match user style preferences.
•It’s necessary to decouple style from seman-
tic. We list style ranking and semantic rank-
ing of each historical headline, where style
ranking represents the ranking results based
on the similarity between the historical head-
line embeddings with the style vector, and se-
mantic ranking represents the ranking results
obtained by Contriver (Lei et al., 2023). We
can observe that headlines with higher style
rankings exhibit stronger alignment with user-
preferred stylistic patterns. However, there ex-
ists significant divergence between style rank-
ing and semantic ranking. For RAG-based
methods, the semantic-dominated retrieved
headlines fail to provide useful patterns about
stylistic preferences.
•Style Transfer .We tried rewriting the user’s
historical texts in a certain style (by instruct-
ing GPT) to recalculate the style vector, in or-
der to observe whether we can steer the model
to generate in the desired style. We targeted
two styles: "exclamatory tone, ending with
an exclamation mark" and "removal of colons
and subheadings." The results show that our
method can achieve style transfer while main-
taining semantic fidelity, further demonstrat-
ing that the style vector can indeed encode the
user’s writing style.
5 Related Work
5.1 Personalized Text Generation
The rapid evolution of LLMs has fundamentally
transformed content generation paradigms, shift-
ing from generic outputs to sophisticated personal-
ized text generation. Current methodologies in per-
sonalized generation predominantly fall into two
technical categories: Retrieval-Augmented Gen-
eration (RAG) approaches leverage users’ histori-
cal content ( Pu) through dynamic retrieval mech-
anisms. While foundational work (Zhang et al.,
2023; Salemi and Zamani, 2024a; Salemi et al.,
2024a; Richardson et al., 2023) established ba-
sic retrieval frameworks, recent innovations have
7

InputGenerate a headline for the following article: “Here are a few tips to keep your teen safe when using the Internet and other web-based technologies. If you think it's an awkward conversation; you can hand them this blog to read.”Ground Truth OutputSocial Media Gone Awry:TipsforTeens to Stay SafeUser HistoryUser-created HeadlinesStyleRankingSemantic RankingThe Anxiety of Hiring a Nanny:Tipsforthe Screening Process116Leading Causes of Injury Death Among Children:WhatParents Needto Know214Summer Camp Safety:Essential Questions Parents Should Ask312Protecting your child after a disclosure of sexual abuse:Whatparents needto know.413Parent Alert:TipsforKeeping Your Children Safe this Summer52Internet Predators:Parents, Monitor Your Children!81Talking to Your Child About the School Shooting in Newtown, CT124Stop Bullying:Teach Your Child Empathy and Limit Their Intake of Violence153If You See Something, Please Do  Something to Prevent Child Abuse1711LLaMASet boundaries and rules for internet use+SFTUse parental controls to limit access to inappropriate content+RAGTalking to Your TeenAbout Online Safety:It's Time to Get Real+OursKeeping Your TeenSafeOnline:Tipsand Strategies forParentsStyle Transfer+ ‘!’TipsforKeeping Your TeenSafeOnline:A Must-Read for Parents!-’:’5 Tipsto Keep Your TeenSafeOnlineFigure 5: Case study of user_310 in News Headline Generation task. The highlighted tokens are the top 5 tokens that
most closely match the style vector among all historical tokens. The underline words are the words that match the
ground truth. ’Style Ranking’ represents the ranking results based on the similarity between the historical headline
embeddings with the style vector. ’Semantic Ranking’ represents the ranking results obtained by Contriver (Lei
et al., 2023).
enhanced these paradigms. Richardson et al.
(2023); Zhang (2024); Tan et al. (2025) developed
profile-augmented prompting strategies, while
Salemi and Zamani (2024b) introduced feedback-
driven retrieval model optimization, demonstrat-
ing improved personalization accuracy. Parameter-
Efficient Fine-Tuning (PEFT) methods adopt an
alternative paradigm by adapting per-user param-
eters through lightweight adapter modules. Com-
parative studies (Salemi and Zamani, 2024a) reveal
that PEFT approaches, particularly those employ-
ing user-specific adapter tuning (Tan et al., 2024a;
Zhuang et al., 2024; Liu et al., 2024b; Ding et al.,
2025), achieve competitive personalization while
maintaining computational efficiency.
5.2 Activation Engineering
Emerging research in activation engineering has
uncovered that LLMs encode semantic concepts as
linear subspaces within hidden activation represen-
tations (Zou et al., 2023; Liu et al., 2023; Rimsky
et al., 2024). This geometric interpretation enables
targeted behavioral steering through linear inter-ventions during inference. Turner et al. (2023)
pioneered activation addition using contrastive-
derived steering vectors for sentiment and topic
control, while Rimsky et al. (2024) enhanced steer-
ing precision through mass-mean activation dif-
ferentials. Zhang et al. (2024a) identified truth-
correlated heads via linear probing, achieving en-
hanced veracity through targeted modulation. Com-
plementing this, Chen et al. (2024b) developed
multi-directional orthogonal steering to amplify
truthfulness in model responses.
6 Conclusion
In this work, we demonstrate that user’s writing
style can be represented as a vector in LLM’s
activation-space. Based on this insight, we intro-
duces a simple yet effective frame, StyleVector,
that achieves personalized text generation through
inference time intervention, without parameter up-
dates or retrievals. Experiments on both short- and
long-form personalization benchmarks show our
method can achieve strong generation performance
while maintaining high efficiency.
8

Limitations
While our framework demonstrates significant ad-
vantages in efficiency and effectiveness, several
limitations warrant discussion to guide future re-
search:
Our training-free style vector derivation, though
efficient, may not achieve optimal disentanglement
of style from content. The current contrastive ap-
proach relies on the model’s inherent ability to
separate these features through simple activation
arithmetic. Future work could explore hybrid ap-
proaches that combine our parametric-free method
with lightweight optimization techniques to refine
the style vectors while maintaining storage effi-
ciency.
The single-vector user representation, while
storage-efficient, potentially conflates multiple
stylistic dimensions (e.g., lexical preferences, syn-
tactic structures, and discourse patterns). A more
granular approach could represent users through
sparse combinations (Cunningham et al., 2023;
Lieberum et al., 2024) of concept-specific vectors,
enabling precise control over individual style com-
ponents.
Our evaluation focuses on established bench-
marks (LaMP and LongLaMP) that assume do-
main homogeneity within each user’s historical
data. However, real-world personalization scenar-
ios often involve cross-domain style consistency –
users may employ distinct stylistic registers across
different tasks (e.g., formal emails vs. casual so-
cial media posts). Current benchmarks lack the
capability to assess whether learned style vectors
can: (1) preserve task-appropriate stylistic varia-
tions within users, or (2) prevent negative interfer-
ence between conflicting domain-specific patterns.
Future work should develop cross-domain person-
alization benchmarks that incorporate mixed-task
histories.
Ethics Statement
The experimental datasets are publicly available
from some previous works, downloaded via of-
ficial APIs. The information regarding users in
all datasets has been anonymized, ensuring there
are no privacy concerns related to the users. We
do not disclose any non-open-source data, and we
ensure that our actions comply with ethical stan-
dards. We use publicly available pre-trained mod-
els, i.e., LLaMA-2, Contriver, and APIs, i.e., GPT-
3.5-turbo, DeepSeek-Chat. All the checkpoints anddatasets are carefully processed by their authors to
ensure that there are no ethical problems.
References
Bashar Alhafni, Vivek Kulkarni, Dhruv Kumar, and
Vipul Raheja. 2024. Personalized text generation
with fine-grained linguistic control. ArXiv .
Steven Au, Cameron J Dimacali, Ojasmitha Pedirappa-
gari, Namyong Park, Franck Dernoncourt, Yu Wang,
Nikos Kanakaris, Hanieh Deilamsalehy, Ryan A
Rossi, and Nesreen K Ahmed. 2025. Personal-
ized graph-based retrieval for large language models.
arXiv preprint arXiv:2501.02157 .
Hongru Cai, Yongqi Li, Wenjie Wang, Fengbin Zhu,
Xiaoyu Shen, Wenjie Li, and Tat-Seng Chua. 2024.
Large language models empowered personalized web
agents. arXiv preprint arXiv:2410.17236 .
Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu, Qi Liu,
Gangwei Jiang, Yuanhao Pu, Yuxuan Lei, Xiaolong
Chen, Xingmei Wang, et al. 2024a. When large
language models meet personalization: Perspectives
of challenges and opportunities. World Wide Web ,
27(4):42.
Zhongzhi Chen, Xingwu Sun, Xianfeng Jiao, Fengzong
Lian, Zhanhui Kang, Di Wang, and Chengzhong Xu.
2024b. Truth forest: Toward multi-scale truthful-
ness in large language models through intervention
without tuning. In Proceedings of the AAAI Con-
ference on Artificial Intelligence , volume 38, pages
20967–20974.
Ziyang Chen and Stylios Moscholios. 2024. Using
prompts to guide large language models in imitating
a real person’s language style. ArXiv .
Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert
Huben, and Lee Sharkey. 2023. Sparse autoencoders
find highly interpretable features in language models.
arXiv preprint arXiv:2309.08600 .
Yucheng Ding, Yangwenjian Tan, Xiangyu Liu,
Chaoyue Niu, Fandong Meng, Jie Zhou, Ning Liu,
Fan Wu, and Guihai Chen. 2025. Personalized lan-
guage model learning on text data without user iden-
tifiers. ArXiv .
Jillian R. Fisher, Skyler Hallinan, Ximing Lu, Mitchell
Gordon, Zaid Harchaoui, and Yejin Choi. 2024.
Styleremix: Interpretable authorship obfuscation via
distillation and perturbation of style elements. ArXiv ,
abs/2408.15666.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. 2021. Lora: Low-rank adap-
tation of large language models. arXiv preprint
arXiv:2106.09685 .
9

Joel Jang, Seungone Kim, Bill Yuchen Lin, Yizhong
Wang, Jack Hessel, Luke Zettlemoyer, Hannaneh
Hajishirzi, Yejin Choi, and Prithviraj Ammanabrolu.
2023. Personalized soups: Personalized large lan-
guage model alignment via post-hoc parameter merg-
ing. arXiv preprint arXiv:2310.11564 .
Hannah Rose Kirk, Bertie Vidgen, Paul Röttger, and
Scott A Hale. 2024. The benefits, risks and bounds of
personalizing the alignment of large language models
to individuals. Nature Machine Intelligence , pages
1–10.
Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra,
Alireza Salemi, Ryan A Rossi, Franck Dernoncourt,
Hanieh Deilamsalehy, Xiang Chen, Ruiyi Zhang,
Shubham Agarwal, et al. 2024. Longlamp: A bench-
mark for personalized long-form text generation.
arXiv preprint arXiv:2407.11016 .
Yibin Lei, Liang Ding, Yu Cao, Changtong Zan, An-
drew Yates, and Dacheng Tao. 2023. Unsupervised
dense retrieval with relevance-aware contrastive pre-
training. In Findings of the Association for Computa-
tional Linguistics: ACL 2023 , pages 10932–10940.
Tom Lieberum, Senthooran Rajamanoharan, Arthur
Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant
Varma, János Kramár, Anca Dragan, Rohin Shah,
and Neel Nanda. 2024. Gemma scope: Open sparse
autoencoders everywhere all at once on gemma 2.
arXiv preprint arXiv:2408.05147 .
Zihao Lin, Zichao Wang, Yuanting Pan, Varun Man-
junatha, Ryan Rossi, Angela Lau, Lifu Huang, and
Tong Sun. 2024. Persona-sq: A personalized sug-
gested question generation framework for real-world
documents. arXiv preprint arXiv:2412.12445 .
Jianzhi Liu, Hexiang Gu, Tianyu Zheng, Liuyu Xiang,
Huijia Wu, Jie Fu, and Zhaofeng He. 2024a. Dy-
namic generation of personalities with large language
models. ArXiv , abs/2404.07084.
Jiongnan Liu, Yutao Zhu, Shuting Wang, Xiaochi Wei,
Erxue Min, Yu Lu, Shuaiqiang Wang, Dawei Yin,
and Zhicheng Dou. 2024b. Llms + persona-plug =
personalized llms. ArXiv .
Sheng Liu, Haotian Ye, Lei Xing, and James Zou. 2023.
In-context vectors: Making in context learning more
effective and controllable through latent space steer-
ing. arXiv preprint arXiv:2311.06668 .
Adam Paszke, Sam Gross, Francisco Massa, Adam
Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca
Antiga, et al. 2019. Pytorch: An imperative style,
high-performance deep learning library. Advances in
neural information processing systems , 32.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2024. Direct preference optimization: Your language
model is secretly a reward model. Advances in Neu-
ral Information Processing Systems , 36.Chris Richardson, Yao Zhang, Kellen Gillespie, Sudipta
Kar, Arshdeep Singh, Zeynab Raeesy, Omar Zia
Khan, and Abhinav Sethy. 2023. Integrating sum-
marization and retrieval for enhanced personalization
via large language models. ArXiv .
Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong,
Evan Hubinger, and Alexander Turner. 2024. Steer-
ing llama 2 via contrastive activation addition. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers) , pages 15504–15522, Bangkok, Thai-
land. Association for Computational Linguistics.
Stephen Robertson, Hugo Zaragoza, et al. 2009. The
probabilistic relevance framework: Bm25 and be-
yond. Foundations and Trends ®in Information Re-
trieval , 3(4):333–389.
Alireza Salemi, Surya Kallumadi, and Hamed Zamani.
2024a. Optimization methods for personalizing large
language models through retrieval augmentation. In
Annual International ACM SIGIR Conference on Re-
search and Development in Information Retrieval .
Alireza Salemi, Sheshera Mysore, Michael Bendersky,
and Hamed Zamani. 2024b. LaMP: When large lan-
guage models meet personalization. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 7370–7392, Bangkok, Thailand. Association
for Computational Linguistics.
Alireza Salemi and Hamed Zamani. 2024a. Comparing
retrieval-augmentation and parameter-efficient fine-
tuning for privacy-preserving personalization of large
language models. arXiv preprint arXiv:2409.09510 .
Alireza Salemi and Hamed Zamani. 2024b. Learn-
ing to rank for multiple retrieval-augmented models
through iterative utility maximization. arXiv preprint
arXiv:2410.09942 .
Zhaoxuan Tan, Zheyuan Liu, and Meng Jiang. 2024a.
Personalized pieces: Efficient personalized large lan-
guage models through collaborative efforts. arXiv
preprint arXiv:2406.10471 .
Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu,
Bing Yin, and Meng Jiang. 2024b. Democratizing
large language models via personalized parameter-
efficient fine-tuning. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing .
Zhaoxuan Tan, Zinan Zeng, Qingkai Zeng, Zhenyu Wu,
Zheyuan Liu, Fengran Mo, and Meng Jiang. 2025.
Can large language models understand preferences
in personalized recommendation? ArXiv .
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .
10

Alexander Matt Turner, Lisa Thiergart, Gavin Leech,
David Udell, Juan J Vazquez, Ulisse Mini, and Monte
MacDiarmid. 2023. Activation addition: Steering
language models without optimization. arXiv e-
prints , pages arXiv–2308.
T Wolf. 2019. Huggingface’s transformers: State-of-
the-art natural language processing. arXiv preprint
arXiv:1910.03771 .
Jiarui Zhang. 2024. Guided profile generation improves
personalization with large language models. In Find-
ings of the Association for Computational Linguistics:
EMNLP 2024 .
Kai Zhang, Fubang Zhao, Yangyang Kang, and Xi-
aozhong Liu. 2023. Memory-augmented llm per-
sonalization with short-and long-term memory coor-
dination. arXiv preprint arXiv:2309.11696 .
Shaolei Zhang, Tian Yu, and Yang Feng. 2024a.
Truthx: Alleviating hallucinations by editing large
language models in truthful space. arXiv preprint
arXiv:2402.17811 .
Zhehao Zhang, Ryan A Rossi, Branislav Kveton, Yijia
Shao, Diyi Yang, Hamed Zamani, Franck Dernon-
court, Joe Barrow, Tong Yu, Sungchul Kim, et al.
2024b. Personalization of large language models: A
survey. arXiv preprint arXiv:2411.00027 .
Jianfeng Zhu, Ruoming Jin, and Karin G. Coifman.
2025. Investigating large language models in in-
ferring personality traits from user conversations.
ArXiv .
Yuchen Zhuang, Haotian Sun, Yue Yu, Rushi Qiang,
Qifan Wang, Chao Zhang, and Bo Dai. 2024. Hydra:
Model factorization framework for black-box llm
personalization. arXiv preprint arXiv:2406.02888 .
Andy Zou, Long Phan, Sarah Chen, James Campbell,
Phillip Guo, Richard Ren, Alexander Pan, Xuwang
Yin, Mantas Mazeika, Ann-Kathrin Dombrowski,
et al. 2023. Representation engineering: A top-
down approach to ai transparency. arXiv preprint
arXiv:2310.01405 .Algorithm 1 Personalized Generation with Style
Steering
Require: User interaction history Pu =
{(xi, yi)}|Pu|
i=1, general LLM Mg, intervention
layer ℓ, scaling factor α, input query x
Ensure: Personalized generation model M
1:Stage 1: Generate Style-Agnostic Responses
2:foreach data pair (xi, yi)∈Pudo
3: Generate style-agnostic response ˆyi←
Mg(xi)
4:end for
5:Stage 2: Extract Style Vector
6:foreach data pair (xi, yi)∈Pudo
7: Compute positive activation aℓ
p,i←
hℓ(xi⊕yi)
8: Compute negative activation aℓ
n,i←
hℓ(xi⊕ˆyi)
9:end for
10:Extract style vector sℓ
u←f({aℓ
p,i, aℓ
n,i}|Pu|
i=1)
11:Stage 3: Activation Steering
12:foreach generation position t≥ |x|do
13: Retrieve original activation hℓ(x)t
14: Inject style vector h′
ℓ(x)t←hℓ(x)t+αsℓ
u
15:end for
A Algorithm
The complete procedure is formalized in Algo-
rithm 1.
B Experiment Details
DPO Baseline The DPO algorithm (Rafailov
et al., 2024) reframes preference learning by di-
rectly optimizing a policy to align with human pref-
erences without explicit reward modeling. Since
we obtained style-agnostic responses, we also em-
ploy DPO loss (Rafailov et al., 2024) to guide
the model to generate user-authentic responses yi
rather than style-agnostic responses ˆyi.
θ∗
u= arg min
θX
(xi,yi,ˆyi)∈Pu−logσ
βlogMθ(yi|xi)
Mref(yi|xi)
−βlogMθ(ˆyi|xi)
Mref(ˆyi|xi)
(12)
where Mθis the policy with adapter θu,Mrefis
the reference policy (base model Mwith frozen
parameters), σdenotes the sigmoid function, and β
controls deviation from the reference policy. This
approach enables parameter-efficient preference
11

Mean LR PCA0510Topic Writing
Mean LR PCA0510Review Generation
Mean LR PCA051015Tweets Paraphrasing
Mean LR PCA0102030Scholarly Title GenerationAverage Improvement (%)Mean
LR
PCAALL Input Tokens
Last Input Tokens
Generated Tokens After InputFigure 6: Performance comparison across different ex-
tracting functions and intervention positions t.
alignment through lightweight adapters while main-
taining the base model’s capabilities.
Implementation Details All experiments were
performed on a cluster of 8 NVIDIA RTX 3090
GPUs, with implementations built upon the Py-
Torch framework (Paszke et al., 2019), Hugging-
Face Transformers (Wolf, 2019) library. To save
computational resources, we apply 8-bit quantiza-
tion and greedy decoding for all methods.
C Additional Experimental Analysis
C.1 Analysis of Extracting Function and
Intervention Position
We compare three different extracting functions
in Equation 5 and different intervention token po-
sitions tin Equation 11. We use three different
intervention positions: intervening on all input to-
kens, intervening only on the last input token, and
intervening on each newly generated token. The
results are shown in Figure 6. As we can see, using
any extracting function and intervention position
results in significant improvements in personalized
text generation. Although it is very simple and
does not introduce excessive complexity, the per-
formance of the Mean Difference function is still
highly superior. Moreover, the more tokens are
intervened, the more pronounced the performance
improvement.
C.2 Analysis of General Model Selection
We compare the different choices of the general
LLM Mgin Equation 4 which is designed to gener-
4.04.55.05.56.06.57.0Topic Writing
6.06.57.07.58.08.5Review GenerationAverage Improvement (%)LLaMA-2-7b
Qwen2.5-72bDeepSeek-Chat
GPT-3.5-TurboFigure 7: Performance comparison with different
generic models Mg.
ate style-agnostic responses. The results are shown
in Figure 7, from which we can observe that the
proposed StyleVector is robust over different gen-
eral models. The general model does not have to be
the same as the model being intervened (LLaMA-
2-7b); in fact, text generated by a more powerful
model tends to have a higher relevance to the input
x, greater diversity, and is more conducive to the
extraction of style vectors.
C.3 Clustering
Figure 8 illustrates the distribution of clustered
style vectors for all users in two tasks of the LaMP
benchmark. As can be seen, the dimensionality-
reduced user style vectors can be grouped into sev-
eral clusters, indicating that different users may
share similar writing styles.
Additionally, in Figure 9, we provide examples
of some clusters and highlight the significant writ-
ing style patterns of these clusters. For example,
in the case of cluster 1, the users within it share
two writing style patterns: one prefers starting with
numbers, and likes adding parentheses at the end
to supplement the content. For cluster 2, all users
share one pattern: they tend to use the dash ’–’ to
connect elements in the title.
25
 0 25
t-SNE Component 120
10
01020t-SNE Component 2T weets Paraphrasing
50
 0 50
t-SNE Component 140
20
02040t-SNE Component 2Scholarly Title Generation
0123456789
Cluster Label
Figure 8: Clustering results of style vectors of all users.
12

Cluster 1User IDUser-created Headlines127The 10Least Affordable Major Metro Areas (PHOTOS)5Things Your Real Estate Agent Won't Tell You (VIDEO)62813Classic Photos Of Phil Jackson Back When He Was The Knicks' Hipster Iconoclast13Ways Johnny Manziel's Pro Day Was The Most Johnny Football Thing Ever (GIFs/PHOTOS)1325On 'Cats' 30th Anniversary, A Brief History (SLIDESHOW)'La Boheme' At Philadelphia Opera Uses High-Tech Van Goghs And Renoirs (PHOTOS)16529Reasons You Should Get A Hair Gloss Treatment (Instead Of A Normal Dye Job)13Lessons We Can Learn From Veteran Actresses' Style (PHOTOS)Cluster 2288The Sponsors Of Obamacare Repeal Are Trying To Fool America --AndFellow RepublicansClinton Lays Out Agenda For Making Child Care Better --AndMore Affordable'1900Chris Christie Video Shows That GOP Empathy Is Real --AndLimitedTrump Has 2 Events This Weekend --AndBoth Benefit His Businesses1401Obama Hits The Trail For Hillary Clinton --AndTo Cement His Legacy For GenerationsTrump Scrambles For A Win --Any Win, Really --As He Nears 100 DaysFigure 9: Case study of clustering writing patterns in News Headline Generation task. The highlighted tokens are
the shared writing styles in cluster.
D Scalability Bottlenecks of Baselines
As summarized in Table 1, existing methods suffer
from three critical scalability constraints:
•Training Time. PETF demands user-
specific adapter optimization with complexity
O(|Pu|), incurring significant costs for large
user bases due to the heavy back-propagations.
RAG is training-free, eliminating gradient-
based training overhead but requiring O(|Pu|)
vectorization pre-processing.
•Inference Latency. In addition to the nor-
mal decoding latency of the language model,
RAG suffers from O(|Pu|)retrieval latency,
which makes it inefficient for users with long
histories. For PEFT, the process of loading
these adapters can introduce overhead, partic-
ularly in scenarios requiring frequent updates
or real-time interactions.
•Storage Overhead. RAG stores all histor-
ical interactions ( O(|Pu|)per user), scaling
poorly for long-term usage. PETF main-
tainsO(rDL)storage for each user (typically
0.1%-1% of base model parameters), where r
is the rank of LoRA, Lis the number of layers
andDis the model hidden dimension.
E Datasets and Task Definition
This paper utilizes the LaMP benchmark and
LongLaMP benchmark for evaluation. We only se-# User Input Length Output Length # History
Abstract Generation 4560 33.82 144.28 120.30
Topic Writing 2453 28.36 263.03 50.39
Review Generation 1822 119.39 304.54 34.39
News Headline Generation 2376 29.97 9.78 287.16
Scholarly Title Generation 2500 152.81 9.26 89.61
Tweet Paraphrasing 1496 29.76 16.93 17.74
Table 4: Datasets statistics. We report the number of
users in test set, the averaged length of input xand
output y, and the averaged number of histories |Pu|.
lect generation tasks in the two benchmarks and the
statistics are shown in Table 4. We show the input-
output pair formats where the text in {BRACES}
can be replaced with content specific to different
users and queries:
LongLaMP: Abstract Generation This task fo-
cuses on generating personalized abstracts for tech-
nical documents or articles based on the provided
title and keywords.
INPUT: Generate an abstract for the title "{ti-
tle}" using the following items: "{keywords}"
OUTPUT: {abstract}
LongLaMP: Review Generation This task in-
volves generating personalized product reviews that
align with the user’s preferences, based on the prod-
uct description and the score assigned to the prod-
uct by the user.
13

INPUT: Generate the review text written by
a reviewer who has given an overall rating of
{rating} for a product with description "{de-
scription}". The summary of the review text
is "{summary}".
OUTPUT: {review}
LongLaMP: Topic Writing This task focuses
on generating a personalized long-form Reddit post
on a given topic from its summary written by user.
INPUT: Generate the content for a Reddit post
"{summary}".
OUTPUT: {post}
LaMP: News Headline Generation This task
focuses on generating a personalized news headline
for a given user-created article.
INPUT: Generate a headline for the following
article "{article}".
OUTPUT: {headline}
LaMP: Scholarly Title Generation This task
requires language models to generate titles for an
input abstract of an paper.
INPUT: Generate a title for the following ab-
stract of a paper "{abstract}".
OUTPUT: {title}
LaMP: Tweet Paraphrasing This task requires
language models to generate a tweet in the style of
a user given an input tweet.
INPUT: Paraphrase the following tweet with-
out any explanation before or after it "{origi-
nal tweet}".
OUTPUT: {tweet}
14