# Detecting Hallucinations in Retrieval-Augmented Generation via Semantic-level Internal Reasoning Graph

**Authors**: Jianpeng Hu, Yanzeng Li, Jialun Zhong, Wenfa Qi, Lei Zou

**Published**: 2026-01-06 14:35:20

**PDF URL**: [https://arxiv.org/pdf/2601.03052v1](https://arxiv.org/pdf/2601.03052v1)

## Abstract
The Retrieval-augmented generation (RAG) system based on Large language model (LLM) has made significant progress. It can effectively reduce factuality hallucinations, but faithfulness hallucinations still exist. Previous methods for detecting faithfulness hallucinations either neglect to capture the models' internal reasoning processes or handle those features coarsely, making it difficult for discriminators to learn. This paper proposes a semantic-level internal reasoning graph-based method for detecting faithfulness hallucination. Specifically, we first extend the layer-wise relevance propagation algorithm from the token level to the semantic level, constructing an internal reasoning graph based on attribution vectors. This provides a more faithful semantic-level representation of dependency. Furthermore, we design a general framework based on a small pre-trained language model to utilize the dependencies in LLM's reasoning for training and hallucination detection, which can dynamically adjust the pass rate of correct samples through a threshold. Experimental results demonstrate that our method achieves better overall performance compared to state-of-the-art baselines on RAGTruth and Dolly-15k.

## Full Text


<!-- PDF content starts -->

Detecting Hallucinations in Retrieval-Augmented Generation via
Semantic-level Internal Reasoning Graph
Jianpeng Hu1, Yanzeng Li2, Jialun Zhong1, Wenfa Qi1, Lei Zou1*
1Wangxuan Institute of Computer Technology, Peking University, Beijing, China
2Institute of Artificial Intelligence and Future Networks, Beijing Normal University, Zhuhai, China
jianpeng.hu@outlook.com, liyanzeng@bnu.edu.cn, zhongjl@stu.pku.edu.cn
{qiwenfa, zoulei}@pku.edu.cn
Abstract
The Retrieval-augmented generation (RAG)
system based on Large language model (LLM)
has made significant progress. It can effectively
reduce factuality hallucinations, but faithful-
ness hallucinations still exist. Previous meth-
ods for detecting faithfulness hallucinations
either neglect to capture the models’ internal
reasoning processes or handle those features
coarsely, making it difficult for discriminators
to learn. This paper proposes a semantic-level
internal reasoning graph-based method for de-
tecting faithfulness hallucination. Specifically,
we first extend the layer-wise relevance prop-
agation algorithm from the token level to the
semantic level, constructing an internal reason-
ing graph based on attribution vectors. This
provides a more faithful semantic-level rep-
resentation of dependency. Furthermore, we
design a general framework based on a small
pre-trained language model to utilize the depen-
dencies in LLM’s reasoning for training and
hallucination detection, which can dynamically
adjust the pass rate of correct samples through
a threshold. Experimental results demonstrate
that our method achieves better overall perfor-
mance compared to state-of-the-art baselines
on RAGTruth and Dolly-15k.
1 Introduction
LLMs easily generate grammatically coherent but
factually incorrect outputs, a phenomenon com-
monly referred to as “hallucination” (Mishra et al.;
Zhang et al., 2024; Li et al., 2023). Post-learning
for downstream tasks or introducing the RAG sys-
tem (Lewis et al., 2020) can mitigate the factuality
hallucinations to some extent, which refer to the
tendency of LLMs to produce outputs that are in-
consistent with real-world facts. However, due
to the inherent knowledge bias of the internal pa-
rameters of LLMs, the generated content of RAG
may be inconsistent with the context provided by
*Corresponding author
s𝑎,2
To heat frozen baked 
scones, preheat your 
oven to 220 °C (200°C 
fan oven / gas mark 6)s𝑎,3
Place the frozen scones on a 
baking tray lined with parchment 
paper and bake for 15 -20 minutes 
or until they are crispy and golden 
browns𝑐,12
The 15 minute guideline 
starts from the point at 
which you put your scones 
in a preheated oven.
s𝑎,1
Based on the 
provided passages, 
here's how to heat 
frozen baked 
scones:s𝑐,11
Make sure to 
preheat the oven 
to the correct 
temperature 
(220°C / 200°C 
fan oven / gas 
mark 6) before 
putting your 
scones in.s𝑐,5
It’s the butter 
that makes 
these cranberry 
scones so tender, 
light and 
delicious.
0.260.18
0.25
0.140.44
0.1s𝑐,1
how to heat 
frozen baked 
scones
0.12s𝑐,2
"Bear in mind that 
your response 
should be strictly 
based on the 
following three 
passages:
0.11Figure 1: Example of a semantic-level internal reason-
ing graph. Yellow nodes represent contextual semantic
fragments, blue nodes represent semantic fragments of
the model’s response, and the weight on the edge in-
dicates the contribution degree of the source semantic
fragment to the target (with an upper boundary of 1).
The dashed box indicates a hallucinated semantic frag-
ment.
users (Lyu et al., 2024), leading to faithfulness
hallucinations. He et al. (2022) has found that
faithfulness hallucinations arise primarily from the
inconsistency between LLMs’ word-level output
and true thought process, meaning that LLMs only
utilize surface knowledge, such as entity popular-
ity (Lehmann et al., 2025), during reasoning.
To detect faithfulness hallucinations, both Man-
akul et al. (2023) and Zhao and Zhang (2025) pro-
pose post-processing methods with LLMs. How-
ever, multiple invocations of the LLM system can
lead to significant resource consumption and am-
plification of model bias. Other scholars (Chen
et al., 2025; Wu et al., 2024; Burns et al., 2022)
assess hallucinations based on the internal embed-
ding of the model, which often relies on heuristic
discrimination trained on these abstract features,
resulting in poor interpretability. Hu et al. (2024);
Chuang et al. (2024) detect hallucinations from the
perspective of output attribution, but their direct ac-
cumulation of all token-level attribution vectors for
LLMs’ responses will introduce substantial noise
(as shown in Appendix A).arXiv:2601.03052v1  [cs.CL]  6 Jan 2026

To better explain the origin of faithfulness hallu-
cinations, motivated by Phukan et al. (2024), we di-
vide the tokens generated during the autoregressive
reasoning of LLMs into linking and substantive to-
kens. Unlike the original concept, we define linking
tokens as non-substantive text in LLM-generated
responses that serve to connect contextual infor-
mation, whereas substantive tokens refer to text
that utilizes the contextual information provided
by users and reflects the semantic content of the
responses. We consider thathallucinations origi-
nate from LLM mistakenly generating substan-
tive tokens as linking, which manifests on the
surface as generation based on entity popularity.
Visual analysis from the perspective of attribution
score distribution can reveal which substantive to-
kens LLMs treat as linking tokens, and these dif-
ferences are difficult to detect from a human per-
spective because of semantic drift. Specifically,
linking tokens rely more on words generated ear-
lier within the same sentence, whereas substantive
tokens also depend on words in the long-distance
context (detailed in Appendix B).
Since faithfulness hallucinations typically occur
at the semantic level, we extend the aforementioned
concepts to this level and detect faithfulness hallu-
cinations by our proposed semantic-level internal
inference graphs, which can faithfully construct
the dependency of the linking and substantive frag-
ments. Specifically, we first employ Layer-wise
relevance propagation (LRP) to calculate the score
vector attributed to each token during the autore-
gressive process. This attribution method, which
utilizes internal model parameters and predefined
rules, faithfully reflects the true computational pro-
cess within the model. According to the token-level
attribution vectors, we model the attribution rela-
tionship between contextual semantic fragments
and those of LLM’s response, forming the internal
reasoning graph, as illustrated in Fig. 1. We can
observe that the semantic fragments with halluci-
nations assign higher attribution scores to previ-
ously generated semantic fragments in the model’s
autoregressive reasoning, while showing weaker
dependency on the context provided by the user.
LLM mistakenly treats substantive fragments as
linking during the reasoning process, leading to the
hallucination phenomenon.
Based on the above observation, we linearize
each response node and attribution dependency in
the internal reasoning graph, concatenate them into
a prompt, and input it to downstream pretrainedlanguage models (PLM) for fine-tuning in binary
classification tasks. In the hallucination detection
phase, we determine hallucination based on the bi-
nary classification label distribution of all semantic
fragments in the LLM-generated text. The entire
process relies solely on the model with a small
number of parameters.
In summary, our contributions mainly include:
•We extend token-level LRP to the semantic
level and propose a method for construct-
ing semantic-level internal reasoning graph
of LLMs.
•We analyze faithfulness hallucinations from
the distribution differences between linking
and substantive fragments and utilize internal
reasoning graphs to detect them.
•The experimental results on two general
datasets demonstrate that the performance of
our framework surpasses previous baselines.
2 Related Work
Hallucination DetectionLLMs often produce
content that is grammatically correct but semanti-
cally conflicts with real-world or contextual knowl-
edge, known as hallucinations (Huang et al., 2025).
Hallucinations widely exist in downstream rea-
soning (Vu et al., 2024; Li et al., 2024). Ex-
isting hallucination detection methods mainly in-
clude: (1) LLMs-based post-verification methods,
which detect hallucinations by using multi-agent
systems (Nguyen et al., 2025), multiple rounds
of self-criticism (Mündler et al., 2024), or consis-
tency of multiple generations (Kuhn et al.); (2)
representation-based methods, which identify ab-
normal states through model hidden states (Bhatia
et al., 2025), outputs of attention modules (Simhi
et al., 2024), semantic alignment rates (Huang et al.,
2023), etc.; (3) task-specific methods, including
fine-tuning (Bergeron et al., 2025) or designing
task-specific features (Guerreiro et al., 2023). Most
of the aforementioned methods rely on explicit rea-
soning or uninterpretable feature spaces, which
may fail to detect semantic biases within the model.
Our framework focuses more on faithfully model-
ing the true reasoning dependencies within LLMs
and training a lightweight detector based on them.
Faithful AttributionOur method, grounded in
additive interpretability theory (Agarwal et al.,
2021), decomposes model predictions into a sum

of contributions from each input, analyzing the sig-
nificance of source tokens to target tokens based
on contribution scores. Perturbation-based method
is the most classic attribution algorithm, including
SHAP (Lundberg and Lee, 2017), LIME (Ribeiro
et al., 2016), AtMan (Deiseroth et al., 2023), etc.
These methods exhibit significant computational
complexity in the application of LLMs. For the
transformer architecture, some scholars utilize at-
tention mechanisms (Abnar and Zuidema, 2020;
Chefer et al., 2021) to capture causal relationships.
However, these methods lack category specificity
and cannot faithfully interpret the final predictions.
Based on an improved backpropagation method,
the propagation rules are customized between lay-
ers to trace back from the model output to the in-
put. Classic methods include Input ×Gradient (Si-
monyan et al., 2013), LRP (Bach et al., 2015; V oita
et al., 2021), SmoothGrad (Smilkov et al., 2017),
etc. AttenLRP (Achtibat et al., 2024) employed in
this paper is an improved backpropagation method
that can effectively handle nonlinear relationships.
3 Methodology
3.1 Task Formulation
LetD= (Q i, Ai)|D|
i=1denote a RAG dataset consist-
ing of |D|samples, where each sample pair com-
prises a user query Qiand its corresponding answer
Ai. Each answer is based on contexts retrieved
from a knowledge base C={C j}|C|
j=1, where Cj
represents a text block. Given a question Qi, a re-
trieval model parameterized by ϕfirst retrieves the
most relevant text blocks from Cto form a prompt,
which is then input into a generative model pa-
rameterized by θto generate the answer Ai. The
complete process of RAG is defined as follows:
P(A i|Qi) =P ϕ(Cj|Qi)Pθ(Ai|Qi, Cj)
Pθ(Ai|Qi, Cj) =nY
k=1Pθ(ak|a1,···, a k−1;Cj;Qi)
(1)
where nis the total number of tokens in the target
answer Ai. We introduce a discriminator param-
eterized by γto determine whether Aicontains
hallucinations:
P(L i|Qi, Cj) =P θ(Ai|Qi, Cj)Pγ(Li|Qi, Cj, Ai)
(2)
where label Liindicates whether the sample ex-
hibits hallucinations. The framework of our
method is shown in Fig. 2.3.2 Contribution Score Calculation
The LRP algorithm is used to calculate the contri-
bution score. The basic assumption of LRP is that
a function fjwithNinput features x={x i}N
i−1
can be decomposed into independent contributions
of a single input variable Ri←j, representing the
amount of output jthat can be attributed to input
i. When these contributions are added up, they are
proportional to the original function value.
fj(x)∝R j=NX
iRi←j (3)
The decomposition property of LRP leads to an
important conservation property (Achtibat et al.,
2024), which ensures that the sum of all contribu-
tion scores in each layer remains constant. This
feature allows for meaningful and faithful attribu-
tion, as the scale of each contribution score can be
associated with the output of the original function.
In the generation stage, when generating each
token ak, the generator simultaneously outputs a
correlation vector ⃗ rkbased on internal gradients
and predefined LRP rules (see Appendix C for de-
tails), which is used to quantify the correlation
between akand each contextual token Cj. By ag-
gregating all correlation scores ⃗ rkof a token during
the generation process, we can obtain a correlation
matrixR i.
3.3 Internal Reasoning Graph Construction
Due to potential semantic biases in the model dur-
ing training, which is caused by the solidification of
pre-trained corpora and the prejudice of the model,
the semantic meaning of certain words understood
by LLMs may differ from their meaning in the
real world. This results in a gap between the rea-
soning process understood by humans based on
chain-of-thought and the reasoning process actu-
ally intended by LLMs. That is, there are some
thinking outputs that cater to human preferences,
while the model’s true reasoning process does not
rely on these thinking outputs or follow thinking
shortcuts. The correlation matrix Ricalculated in
the previous step using the LRP is a token-level
attribution matrix that is faithful to the internal
reasoning process of the model. Therefore, the in-
ternal reasoning graph constructed based on Rican
faithfully reflect which contextual semantic frag-
ments a certain semantic fragment originates from
during the internal inference of the model.

Entity 
Extraction 
ToolsFragments
𝑠1
𝑠2
𝑠𝑚…𝑠1
𝑠2
𝑠𝑚…Entities 
e3e5e7…
…e1e2e4…e1e5e6…
QuestionContext
Text 1 
Text 2 …
PLMs𝑐,4 s𝑎,4 s𝑐,5s𝑎,3
…
𝑙𝑎,1 𝑙𝑎,4 𝑙𝑎,2s𝑐,1 s𝑎,1 s𝑐,2s𝑐,3
s𝑐,1 s𝑎,2 s𝑐,2s𝑎,1
…
𝑙𝑎
Input Contribution Score Calculation Internal Reasoning Graph Construction Hallucination Discrimination
LLM
…
Contribution Score
𝑡1𝑡2 𝑡𝑛 𝑡3
Layer -wise Relevance Propagation
Internal 
Reasoning 
Graph
R7(3)=𝑓(𝑥)
R3←6(1,2)R2←5(1,2)R1←4(1,2)
R4←7(2,3)
R5←7(2,3)
R6←7(2,3)R3←5(1,2)1
2
34
5
67Briefly answer 
the following 
question:
The winds are 
blowing at 10 
mph from the 
north -northwest.Based on the given 
passages, here are 
the steps to upload 
a video to Steam:There is 39 percentage 
chance of rain and 10 
mph winds from the 
North -Northwest.There is 28 percentage 
chance of rain and 5 
mph winds from the 
North -Northeast.In case the passages 
do not contain the 
necessary information 
to answer the question, 
please reply with:Then make a video, 
andupload it to 
YouTube.
Create a YouTube 
account if you don't 
already have one.Upload your 
video to 
YouTubehow to upload a 
video to steamThe
winds
are…Figure 2: The framework of our method. LRP is first applied to derive a relevance distribution Ribased on the
parameters of the LLM. Based on Riof entities and semantic fragments of input and output content, a semantic-level
internal reasoning graph of LLMs is constructed. Subsequently, a PLM is used to determine whether each fragment
exhibits semantic conflicts or omissions. Finally, the degree of hallucination occurring in the reasoning graph units
is used to determine whether the model’s overall response exhibits hallucinations.
We first recursively use “\n” and sentence tool
of Spacy to segment the input context into indi-
vidual semantic fragments, thereby obtaining a
setSc={s c,1, sc,2,···} , where each sc,irepre-
sents the ith semantic fragment in that context.
For the model’s output, we similarly use “\n” to
segment, obtaining a set of semantic fragments
Sa={s a,1, sa,2,···} contained in one response
from the model. The union of these semantic frag-
ments, denoted as S=S c∪Sa, serves as the node
set of the model’s internal reasoning graph, that is,
the set of atomic steps for model inference.
As mentioned above, the text within a seman-
tic fragment includes both linking content used to
connect contexts and make them grammatically
correct, and substantive content that specifically
reflects the actual meaning and thought expressed
in the text. When calculating attribution at the frag-
ment level based on LRP, the attribution scores
calculated for these linking fragments introduce a
significant amount of noise. To distinguish between
semantic fragments that contain rich semantic infor-
mation and meaningless content, we use a general
named entity extraction tool (Spacy and Stanze)
to extract entities from the document to the great-
est extent possible. The extracted content includes
nouns, verbs, noun phrases, negation words, andnamed entities, which can maximize the reflection
of the semantic meaning expressed in the text while
filtering out linking content. The set of extracted
results, after removing duplicates, is considered the
core content E={e 1, e2,···} that expresses se-
mantic information in the text. Finally, by mapping
the set Eto each element in the semantic fragment
setS, we can obtain the subset of actual meaning
Esi={e}contained in each semantic fragment.
In each target semantic fragment si, we only se-
lect tokens contained in Esito calculate its attribu-
tion vector relative to the preceding text. Then, we
average all the selected correlation vectors element
by element to obtain the token-level attribution vec-
tor of the semantic segment relative to the preced-
ing text. For the attributed semantic fragment sjin
the preceding text, we select the maximum value
of tokens contained in Esjin the attribution vector
ofsias the score of si’s attribution to sj. When
attributing, sioften only attaches to a small number
of tokens with actual meaning in the preceding text.
Using an average function can dilute the high cor-
relation information in sjdue to fragment length.
Therefore, we use a maximum function to aggre-

gate the correlation vectors:
Wsa,i,s∗,j=

max
w∈sc,j(1
|Esa,i|X
e∈Esa,i⃗ re,w)
sa,i∈Sa, sc,j∈Sc
max
w∈sa,j(1
|Esa,i|X
e∈Esa,i⃗ re,w)
sa,i, sa,j∈Sa, i > j
0s a,i, sa,j∈Sa, i≤j(4)
where ⃗ re,wrepresents the element associated with
token win the vector corresponding to entity ein
the matrix Ri. If the entity consists of multiple
tokens, the average of the vectors corresponding
to these multiple entities is taken. Through this
step, we can obtain the semantic-level correlation
matrix W∈Rna×(nc+na), where each element
Wi,jrepresents the influence degree of the jth se-
mantic fragment on the ith. As shown in Fig. 2,
since subsequent semantic segments do not have
an attributive influence on preceding semantic seg-
ments, this matrix is a lower triangular matrix from
columnn ato columnn c+na.
The internal reasoning graph G={V, E} is
a directed graph that faithfully reflects the depen-
dency relationships between semantic fragments
during the internal inference process of the model.
The nodes of the graph, V=S c∪Sa, represent
the semantic fragments obtained from the previous
context. The edges are associated through the at-
tribution scores between semantic fragments. The
weight of each edge is the normalized attribution
score. Based on the semantic-level correlation ma-
trix, we propose two methods for constructing intra-
inference graphs:
Top k methodThis strategy first ranks the at-
tribution scores calculated for the target semantic
fragments from highest to lowest, then selects the
topkfragments as the source, and inserts edges
from the source to the target semantic fragments in
the graph. The set of incoming edges for node sa,j
can be represented as:
Ea,j={(s ∗,i, sa,j)|Topk(W sa,j,s∗,i), s∗,i∈V}
(5)
Adaptive MethodThe distribution of attribution
scores calculated often exhibits a long-tail char-
acteristic. To select edges adaptively, we arrange
the attribution scores in descending order and cal-
culate the discrete gradient of the sequence. Themaximum discrete gradient point is used to distin-
guish between important and unimportant source
semantic fragments. Assuming that v1≥v 2≥
··· ≥v nc+nais the non-increasing ordering of
elements in Wsa,i, where each v1is mapped one-to-
one with s∗,iusing the function f(·), then the set
of incoming edges for node sa,jcan be represented
as follows:
m= argmax
1≤i≤n c+na−1(Wsa,j,f(vi)−W sa,j,f(vi+1))
Ea,j={(s ∗,i, sa,j)|s∗,i∈f(v k),1≤k≤m}
(6)
The union of the incident edge sets of all nodes
forms the edge setE={E a,j|1≤j≤n a}.
3.4 Hallucination Discrimination
As depicted in Fig. 1, in the attribution nodes of
hallucinated semantic fragments, a higher propor-
tion of attribution is allocated to the previous se-
mantic fragment answered by the model; whereas
the attribution nodes of non-hallucinated semantic
fragments tend to be more related to the contex-
tual corpus provided by the user, indicating that
they are more faithful to the context provided by
humans. Therefore, an important reason for the
occurrence of hallucination is that the model treats
the next generated semantic fragment as linking
content, rather than as substantive content.
To enable the model to discover the attribution
distribution differences and semantic differences
in these contextual dependencies, we linearize the
inference graph into multiple semantic combina-
tions. Specifically, for each semantic fragment sa,j
answered by the model, we concatenate all its in-
coming edges to form a prompt, and feed it into
a pre-trained language model (PLM) to obtain the
labell a,j:
la,j=PLM({s ∗,i|s∗,i∈Ea,j} ⊕s a,j)(7)
In this paper, we utilize the ALIGNSCORE (Zha
et al., 2023), which is based on the RoBERTa archi-
tecture (Liu et al., 2019), as the PLM for training,
and employ its binary classification inference head
for hallucination discrimination. ALIGNSCORE
has undergone pretraining on a vast amount of data,
focusing on the degree of information alignment
between two arbitrary segments. Therefore, by
simply fine-tuning it with the downstream cross-
entropy loss function, it can exhibit strong halluci-
nation detection capabilities. Ultimately, we em-
ploy a flexible threshold αto determine whether

the entire model response exhibits hallucinations.
If the proportion of semantic fragments containing
hallucinations in the model response exceeds α,
then the model response is considered to exhibit
hallucinations. Formally, it can be expressed as
follows:
la=I[Pna
j=1I[la,j= 0]
na≤α](8)
where Idenotes an indicator function. This pa-
per aligns with previous research, where la= 1
signifies a correct model response, indicating the
absence of hallucination, and la= 0 denotes an
incorrect model response, indicating the presence
of hallucination. When α= 0 , it means that any
semantic fragment in the model response that is
suspected of being hallucinatory will result in the
entire response being classified as hallucination. In
practice, the value of αcan be adjusted based on
the required level of model reliability for specific
scenarios.
4 Experiments
4.1 Datasets
We conducted experiments using RAGTruth (Niu
et al., 2024) and Dolly-15k (Conover et al., 2023).
RAGTruth is a manually annotated RAG sample
set generated by various LLMs. The Llama-7B
part we used contains 510 hallucination samples
and 479 normal samples, while the Llama-13B part
contains 399 hallucination samples and 590 nor-
mal samples. Dolly-15k is a large model question-
answering dataset covering multiple scenarios. We
only used the closed question-answering scenar-
ios oriented towards the RAG framework and fil-
tered out samples with empty contexts. Consistent
with Hu et al. (2024), GPT-4 is used to compare
model outputs with standard answers to annotate
the dataset.
4.2 Baselines
Detailed implementation of our method (SIRG) is
provided in the Appendix D. We compare SIRG
with the following baselines:
Prompt(Niu et al., 2024) Through prompt engi-
neering, we manually design LLM (Llama-7b and
GPT-3.5-turbo) prompts to identify hallucinations.
SelfCheckGPT(Manakul et al., 2023) Self-
CheckGPT is employed to assess the consistency
between sampled responses, calculating the proba-
bility of hallucination.Fine-tune(Niu et al., 2024) We fine-tune Llama-
7b and Qwen-7b on the corresponding dataset to
detect hallucinations.
EigenScore(Chen et al., 2024) This method uti-
lizes the eigenvalues of the response covariance
matrix to measure semantic consistency in the em-
bedding space.
SEP(Kossen et al., 2024) A linear probe trained
on the hidden states of LLMs is utilized to detect
hallucinations.
LRP4RAG(Hu et al., 2024) This is a method
based on LRP, which directly feeds the contribution
scores to SVM classifiers or LLMs for hallucina-
tion detection.
4.3 Main Results
We employ 3 evaluation metrics to compare SIRG
with 9 state-of-the-art baselines on RAGTruth and
Dolly-15k, with some results directly sourced from
Hu et al. (2024).
Table 1 shows the comparison results on
RAGTruth. SIRG has strong performance across
all metrics, even outperforming high-resource-
consumption methods such as LLMs fine-tuning.
On RAGTruth Llama-7B and RAGTruth Llama-13B ,
SIRG ranks first with the highest F1, achieving im-
provements of 3.07% and5.78% over the currently
most advanced methods respectively. Prompt-
based and self-validation-based methods (Self-
CheckGPT) rely on pre-trained LLMs by design-
ing prompts to enable single or multiple rounds of
self-correction, which makes them highly unsta-
ble. Switching LLMs or prompts can significantly
impact downstream tasks. For example, using dif-
ferent prompts with the same gpt-3.5-turbo resulted
in a44.89% difference in recall. Although gpt-3.5-
turbo achieves 92.54% recall in the SelfCheckGPT
framework, its precision drops to 53.27% , indi-
cating that LLMs blindly classify most samples
as correct. This demonstrates that relying solely
on pre-training knowledge is insufficient for accu-
rate hallucination detection. The Fintune approach
trains LLMs using specific RAGTruth data samples,
yet its average performance remains only 62.74%
and36.75% . We attribute this to the insufficient
training data scale, which may inadvertently dis-
rupt the general knowledge acquired through fine-
tuning. Consequently, the fine-tuning outcomes are
inferior to those of direct Prompt-based methods.
Both EigenScore and SEP are methods based on

ModelPrecision Recall F1
RAGTruth Llama-7B
Promptllama-7b 52.64% 76.08% 62.23%
Promptgpt-3.5-turbo 56.91% 47.65% 51.87%
SelfCheckGPT llama-7b 53.32% 83.53% 65.09%
SelfCheckGPT gpt-3.5-turbo 53.27%92.54%67.62%
Fintune llama-7b 62.50% 65.75% 63.58%
Fintune qwen2-7b 61.76% 64.34% 61.90%
EigenScore−74.69% 66.82%
SEP−74.77% 66.27%
LRP4RAG LLM 71.18% 75.78% 73.54%
SIRG (Ours)73.64%79.83%76.61%
RAGTruth Llama-13B
Promptllama-7b 41.02% 56.64% 47.58%
Promptgpt-3.5-turbo 47.58% 44.36% 45.91%
SelfCheckGPT llama-7b 43.66% 75.94% 55.44%
SelfCheckGPT gpt-3.5-turbo 43.01%89.47%58.10%
Fintune llama-7b 62.50% 27.92% 37.62%
Fintune qwen2-7b 63.55% 25.93% 35.89%
EigenScore−67.15% 66.37%
SEP−65.80% 71.59%
LRP4RAG LLM 77.14% 74.58% 75.86%
SIRG (Ours)78.48%85.51%81.84%
Table 1: Overall precision, recall, and F1-score on
RAGTruth with Llama-7B and Llama-13B.
vector space discriminators that lack direct contex-
tual semantic information, making it difficult to
adequately identify hallucination. LRP4RAG em-
ploys token-level aggregation of attribution vectors
generated by the LRP algorithm to derive contex-
tual relevance for model responses, representing a
coarse-grained approach. This method introduces
excessive noise of linking text, resulting in sub-
optimal performance during classifier training or
discriminative tasks using LLMs. Our approach
also employs the LRP algorithm, but it enhances
the processing of substantive information in re-
sponse texts by filtering out semantic noise and
formally modeling it as a reasoning graph. This
facilitates easier training of the downstream dis-
criminator while helping humans understand the
decision-making process of LLMs. See Fig.9 in
Appendix B for details of the example.
Table 2 presents the comparative results of
the Dolly-15k dataset. For the threshold-based
benchmark model, we provide its optimal thresh-
old parameters. Since the content generated
by the LLM each time is random, fine-tuning
to fit fixed responses is pointless in this sce-
nario. On DollyQwen2.5-3B and DollyQwen2.5-7B ,
SIRG achieved F1 scores of 82.17% and89.17%
respectively, surpassing the state-of-the-art methodModelPrecision Recall F1
DollyQwen2.5-3B
Prompt58.41% 24.98% 34.99%
SelfCheckGPT67.32% 32.47% 43.88%
EigenScore68.88% 64.58% 66.66%
SEP77.94% 79.19% 78.56%
LRP4RAG LLM 80.55%82.91% 81.71%
SIRG (Ours)72.10%95.49% 82.17%
DollyQwen2.5-7B
Prompt61.46% 47.23% 53.36%
SelfCheckGPT67.32% 32.47% 43.88%
EigenScore58.57% 70.03% 63.79%
SEP76.36% 77.59% 76.97%
LRP4RAG LLM 79.60% 82.20% 80.80%
SIRG (Ours)84.21% 96.00% 89.71%
Table 2: Overall precision, recall, and F1-score on
Dolly-15k with Qwen2.5-3B and Qwen2.5-7B.
LRP4RAG. Due to the more unstable response of
Qwen2.5-3B compared to Qwen2.5-7B, SIRG per-
forms better on Qwen2.5-7B than on Qwen2.5-3B.
4.4 Faithfulness of LRP-based Internal
Reasoning Graph
To verify the faithfulness of the internal inference
graph constructed by SIRG, the same perturbation
tests as (Bakish et al., 2025) are employed (detailed
in Appendix E).
We implement token-level blocking of semantic
fragments in ten sequential steps based on their
relevance, with results presented in Fig. 3. When
adding the semantic fragment deemed most rele-
vant by LRP, the most significant changes are ob-
served in the decrease of (y0−yp)2and the increase
ofloghts k, indicating that this semantic fragment
plays a pivotal role in LLM’s computational pro-
cess. After pruning semantic fragments based on
relevance scores, those with lower relevance to the
target exhibit negligible impact on both (y0−yp)2
andloghts k, whereas highly relevant fragments
demonstrate substantial effects post-pruning. This
shows that our algorithm can effectively identify
the source semantic fragment, which has a signifi-
cant impact on the target semantic fragment. If the
importance of contributions is randomly assigned,
the result curve should change gradually with ad-
dition or pruning. Compared with the standard
curve of random addition or pruning, our method
demonstrates a significant AUC advantage. For

Figure 3: Perturbation tests on RAGTruth with Llama-
7B.(y0−yp)2indicates the change of final embeddings
before and after perturbation, while logits krepresents
the average probability of the target semantic fragment.
The dashed line shows the curve state after random
addition or pruning. Perturbation tests are conducted on
100 samples and mean of the above indicators are took.
quantitative comparison of this curve, please refer
to the work of Bakish et al. (2025).
4.5 The Impact of Hyperparameters
For the classifier of SIRG, we focus on exploring
the impact of αin Equation 8. When α= 0 , as
long as there is one hallucinated semantic frag-
ment, this response will be judged as a hallucinated
sample, representing the strictest hallucination de-
tection strategy. As shown in Fig. 4, at this point,
SIRG has a relatively low recall for correct samples
but a high precision in identifying hallucinations.
As the value of αincreases, the detection strategy
becomes increasingly lenient, so the pass rate for
correct samples rises. When α= 0.4 , the recall
for correct samples reaches 100% , but the preci-
sion decreases to 72.86% . Although adjusting α
greatly impacts recall and precision, it has a rel-
atively weak effect on the F1 score. In different
scenarios, we can dynamically adjust αbased on
the desired pass rate for correct samples.
For the construction of the internal reasoning
graph, we regulate the number of source seman-
tic fragments by setting different Topk values.
The performance of discriminators trained using
graphs generated by various construction strate-
gies is demonstrated in Table 3. As the number
Figure 4: Overall precision, recall, and F1 score of
Llama-13B on RAGTruth are evaluated by setting dif-
ferentαvalues.
Top 1 5 10 15 20 Ada
Precision79.35% 82.46% 85.71% 83.12% 82.27% 78.98%
Recall84.82% 87.58% 91.03% 91.72% 89.65% 85.51%
F182.00% 84.94% 85.71% 87.21% 85.80% 82.11%
Table 3: Overall precision, recall, and F1 score of
Llama-13B on RAGTruth are evaluated by setting dif-
ferent Topk . Ada denotes the gradient-based adaptive
construction strategy referred in Equation 6.
of source semantic fragments increases, the dis-
criminator can obtain more semantic information.
Consequently, the discriminators’ F1 score progres-
sively improves during the initial training phase,
reaching its peak at k= 15 . An increase in kwill
introduce edge contribution noise into the target se-
mantic fragment, meaning many low-contribution
semantic fragments are mistakenly fed to the dis-
criminator for training. The discriminator may mis-
interpret the conflict between insignificant semantic
fragments and target semantic fragments as halluci-
nation, resulting in inferior performance at k= 20
compared to k= 15 . For the adaptive discrete gra-
dient strategy, it tends to only select 1 or 2 source
semantic fragments. Without sufficient informa-
tion, the discriminator’s performance is limited.
5 Conclusion
This paper first extends the token-level LRP al-
gorithm to the semantic level within the autore-
gressive inference paradigm. Then we construct
internal reasoning graphs using semantic fragments
from RAG contexts and LLMs’ responses, which
faithfully model the dependencies of the internal
reasoning process. Based on that, we propose
a framework, SIRG, for identifying faithfulness
hallucinations in RAG. SIRG achieves the perfor-
mance of LLM-based detection frameworks us-
ing only a lightweight parameterized discriminator,
demonstrating the effectiveness of our approach.

Limitations
For the internal reasoning graph construction of
SIRG, since LRP requires computing internal
model gradients for each token generation, this
results in a high time complexity for the attribution
score calculation phase. Future work will opti-
mize LRP’s computational objects from semantic
fragment perspectives to reduce graph construction
time. We will also evaluate the semantic-level faith-
fulness of various attribution methods in capturing
internal inference processes.
For the hallucination detection module of SIRG,
although the linearization method has achieved
great results, it is a naive way of using topolog-
ical relation, which ignores the multi-hop depen-
dency information and the subtle error propagation
information in the internal reasoning graph. Fu-
ture efforts will explore multi-angle applications of
this graph, including adaptive node relationship ag-
gregation via graph neural networks. Additionally,
developing low-resource hallucination discrimina-
tors remains a key research focus.
Acknowledgments
We would like to appreciate anonymous reviewers
for their valuable comments that help us to improve
this manuscript.
Ethics Statement
Based on the research presented in this paper, we
acknowledge the ethical implications of developing
hallucination detection methods for LLMs. While
our work aims to enhance the reliability and trust-
worthiness of AI-generated content, we recognize
that such techniques could potentially be misused
to conceal model limitations or manipulate outputs
in ways that undermine transparency. We affirm our
commitment to responsible AI research by ensuring
our method, SIRG, is designed to improve factual
faithfulness rather than to deceive. All experiments
were conducted using publicly available datasets
with proper citations, and we have openly disclosed
the limitations of our approach to avoid overstating
its capabilities. We encourage the community to
utilize this work for promoting accountability and
interpretability in LLMs, and we emphasize the
importance of continued ethical scrutiny as halluci-
nation detection technologies evolve.References
Samira Abnar and Willem Zuidema. 2020. Quantifying
attention flow in transformers. InProceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics. Association for Computational
Linguistics.
Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi,
Maximilian Dreyer, Aakriti Jain, Thomas Wiegand,
Sebastian Roland Lapuschkin, and Wojciech Samek.
2024. Attnlrp: Attention-aware layer-wise relevance
propagation for transformers. InInternational Con-
ference on Machine Learning 2024.
Rishabh Agarwal, Levi Melnick, Nicholas Frosst,
Xuezhou Zhang, Ben Lengerich, Rich Caruana, and
Geoffrey E Hinton. 2021. Neural additive models:
Interpretable machine learning with neural nets.Ad-
vances in neural information processing systems,
34:4699–4711.
Ameen Ali, Thomas Schnake, Oliver Eberle, Grégoire
Montavon, Klaus-Robert Müller, and Lior Wolf.
2022. Xai for transformers: Better explanations
through conservative propagation. InInternational
conference on machine learning, pages 435–451.
PMLR.
Sebastian Bach, Alexander Binder, Grégoire Montavon,
Frederick Klauschen, Klaus-Robert Müller, and Wo-
jciech Samek. 2015. On pixel-wise explanations
for non-linear classifier decisions by layer-wise rele-
vance propagation.PloS one, 10(7):e0130140.
Yarden Bakish, Itamar Zimerman, Hila Chefer, and Lior
Wolf. 2025. Revisiting lrp: Positional attribution as
the missing ingredient for transformer explainability.
arXiv preprint arXiv:2506.02138.
Loris Bergeron, Ioana Buhnila, Jérôme François, and
Radu State. 2025. Halluguard: Evidence-grounded
small reasoning models to mitigate hallucinations
in retrieval-augmented generation.arXiv preprint
arXiv:2510.00880.
Gagan Bhatia, Somayajulu G Sripada, Kevin Allan,
and Jacobo Azcona. 2025. Distributional seman-
tics tracing: A framework for explaining halluci-
nations in large language models.arXiv preprint
arXiv:2510.06107.
Collin Burns, Haotian Ye, Dan Klein, and Jacob Stein-
hardt. 2022. Discovering latent knowledge in lan-
guage models without supervision.arXiv preprint
arXiv:2212.03827.
Hila Chefer, Shir Gur, and Lior Wolf. 2021. Trans-
former interpretability beyond attention visualization.
InProceedings of the IEEE/CVF conference on com-
puter vision and pattern recognition, pages 782–791.
Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu,
Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024.
Inside: Llms’ internal states retain the power of hallu-
cination detection.arXiv preprint arXiv:2402.03744.

Kang Chen, Yaoning Wang, Kai Xiong, Zhuoka Feng,
Wenhe Sun, Haotian Chen, and Yixin Cao. 2025. Do
llms signal when they’re right? evidence from neuron
agreement.arXiv preprint arXiv:2510.26277.
Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ran-
jay Krishna, Yoon Kim, and James Glass. 2024.
Lookback lens: Detecting and mitigating contextual
hallucinations in large language models using only
attention maps. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing, pages 1419–1436.
Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie,
Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell,
Matei Zaharia, and Reynold Xin. 2023. Free dolly:
Introducing the world’s first truly open instruction-
tuned llm.
Björn Deiseroth, Mayukh Deb, Samuel Weinbach,
Manuel Brack, Patrick Schramowski, and Kristian
Kersting. 2023. Atman: Understanding transformer
predictions through memory efficient attention ma-
nipulation.Advances in Neural Information Process-
ing Systems, 36:63437–63460.
Nuno M Guerreiro, Duarte M Alves, Jonas Waldendorf,
Barry Haddow, Alexandra Birch, Pierre Colombo,
and André FT Martins. 2023. Hallucinations in large
multilingual translation models.Transactions of the
Association for Computational Linguistics, 11:1500–
1517.
Hangfeng He, Hongming Zhang, and Dan Roth. 2022.
Rethinking with retrieval: Faithful large language
model inference.arXiv preprint arXiv:2301.00303.
Haichuan Hu, Congqing He, Xiaochen Xie, and
Quanjun Zhang. 2024. Lrp4rag: Detecting hal-
lucinations in retrieval-augmented generation via
layer-wise relevance propagation.arXiv preprint
arXiv:2408.15533.
Kung-Hsiang Huang, Hou Pong Chan, and Heng Ji.
2023. Zero-shot faithful factual error correction. In
The 61st Annual Meeting Of The Association For
Computational Linguistics.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa
Schut, Shreshth Malik, and Yarin Gal. 2024. Seman-
tic entropy probes: Robust and cheap hallucination
detection in llms.arXiv preprint arXiv:2406.15927.
Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Se-
mantic uncertainty: Linguistic invariances for uncer-
tainty estimation in natural language generation. In
The Eleventh International Conference on Learning
Representations.Hans Hergen Lehmann, Jae Hee Lee, Steven Schock-
aert, and Stefan Wermter. 2025. Knowing the facts
but choosing the shortcut: Understanding how large
language models compare entities.arXiv preprint
arXiv:2510.16815.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng,
Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen.
2024. The dawn after the dark: An empirical study
on factuality hallucination in large language models.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 10879–10899.
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter
Pfister, and Martin Wattenberg. 2023. Inference-
time intervention: Eliciting truthful answers from
a language model.Advances in Neural Information
Processing Systems, 36:41451–41530.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach.arXiv preprint arXiv:1907.11692.
Scott M Lundberg and Su-In Lee. 2017. A unified ap-
proach to interpreting model predictions.Advances
in neural information processing systems, 30.
Qing Lyu, Marianna Apidianaki, and Chris Callison-
Burch. 2024. Towards faithful model explanation in
nlp: A survey.Computational Linguistics, 50(2):657–
723.
Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.
Selfcheckgpt: Zero-resource black-box hallucination
detection for generative large language models. In
Proceedings of the 2023 conference on empirical
methods in natural language processing, pages 9004–
9017.
Abhika Mishra, Akari Asai, Vidhisha Balachandran,
Yizhong Wang, Graham Neubig, Yulia Tsvetkov, and
Hannaneh Hajishirzi. Fine-grained hallucination de-
tection and editing for language models. InFirst
Conference on Language Modeling.
Niels Mündler, Jingxuan He, Slobodan Jenko, and Mar-
tin Vechev. 2024. Self-contradictory hallucinations
of large language models: Evaluation, detection and
mitigation. InThe Twelfth International Conference
on Learning Representations (ICLR 2024). OpenRe-
view.
Vy Nguyen, Ziqi Xu, Jeffrey Chan, Estrid He, Feng Xia,
and Xiuzhen Zhang. 2025. Hallucinate less by think-
ing more: Aspect-based causal abstention for large
language models.arXiv preprint arXiv:2511.17170.

Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun
Shum, Randy Zhong, Juntong Song, and Tong Zhang.
2024. Ragtruth: A hallucination corpus for develop-
ing trustworthy retrieval-augmented language models.
InProceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 10862–10878.
Anirudh Phukan, Shwetha Somasundaram, Apoorv Sax-
ena, Koustava Goswami, and Balaji Vasan Srinivasan.
2024. Peering into the mind of language models:
An approach for attribution in contextual question
answering. InACL (Findings).
Marco Tulio Ribeiro, Sameer Singh, and Carlos
Guestrin. 2016. " why should i trust you?" explaining
the predictions of any classifier. InProceedings of
the 22nd ACM SIGKDD international conference on
knowledge discovery and data mining, pages 1135–
1144.
Adi Simhi, Jonathan Herzig, Idan Szpektor, and Yonatan
Belinkov. 2024. Constructing benchmarks and inter-
ventions for combating hallucinations in llms.arXiv
preprint arXiv:2404.09971.
Karen Simonyan, Andrea Vedaldi, and Andrew Zis-
serman. 2013. Deep inside convolutional networks:
Visualising image classification models and saliency
maps.arXiv preprint arXiv:1312.6034.
Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda
Viégas, and Martin Wattenberg. 2017. Smoothgrad:
removing noise by adding noise.arXiv preprint
arXiv:1706.03825.
Elena V oita, Rico Sennrich, and Ivan Titov. 2021. Ana-
lyzing the source and target contributions to predic-
tions in neural machine translation. InProceedings
of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing
(Volume 1: Long Papers), pages 1126–1140.
Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry
Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny
Zhou, Quoc Le, and 1 others. 2024. Freshllms: Re-
freshing large language models with search engine
augmentation. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 13697–
13720.
Di Wu, Jia-Chen Gu, Fan Yin, Nanyun Peng, and Kai-
Wei Chang. 2024. Synchronous faithfulness monitor-
ing for trustworthy retrieval-augmented generation.
arXiv preprint arXiv:2406.13692.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. Alignscore: Evaluating factual consistency
with a unified alignment function. InProceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 11328–11348.
Dongxu Zhang, Varun Gangal, Barrett Lattimer, and
Yi Yang. 2024. Enhancing hallucination detectionthrough perturbation-based synthetic data generation
in system responses. InFindings of the Association
for Computational Linguistics: ACL 2024, pages
13321–13332.
Yaxin Zhao and Yu Zhang. 2025. Halluclean: A unified
framework to combat hallucinations in llms.arXiv
preprint arXiv:2511.08916.
A Coarse-grained Processing
Hu et al. (2024) employs the maximum and av-
erage value of the LRP attribution vector for all
tokens to get the attribution distributions of the
whole response. As shown in Fig. 5, using the
maximum value method accumulates the context
that contributes the most to each response token,
resulting in a noisy contribution distribution. Us-
ing the average value method dilutes the context
with a high contribution to the substantive word
with a large number of linking words, leading to a
significantly lower contribution distribution.
B Linking Tokens and Substantive
Tokens
In the sentence “The 15 minute guideline starts
from the point at which you put your scones in a
preheated oven”, “15 minute”, “preheated oven”,
etc., contain rich semantic information and are
considered as substantive tokens, while “The”,
“which”, etc., serve to enhance sentence fluency
in the document and are therefore considered link-
ing tokens.
The above description of the linking and substan-
tive token is identified from a human perspective.
From an LLM’s standpoint, due to semantic drift
and dataset bias during training, the linking and
substantive tokens inferred by the LLM may differ
from human perceptions. When LLMs generate
tokens by treating human-considered substantive
content as linking, it results in the faithfulness hal-
lucination. The cognitive gap between humans and
LLMs makes this phenomenon hard to detect di-
rectly.
We employ the LRP to perform token-level attri-
bution of the next token generated by the LLM, and
visualize the results using a heat map. As shown
in Fig.6, linking tokens rely more on words gen-
erated earlier within the same sentence, whereas
substantive tokens also depend on words in the
long-distance context. When the model generates
hallucination tokens, the heat map in Fig.7 reveals
that their attribution distribution exhibits character-
istics of linking tokens. From a human perspective,

<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2 A
good flow of blood will help the delivery of oxygen to the hair and scal p,
keeping the cells health yand less prone to break age and damage .3 Tip #
5 Never over -style the hair .Tip #6 A good way to promote hair growth
is to have regular trim ming of the hair every 8 weeks .2 This rids the
hair of un health ysplit ends ,and prom otes the length ening of hair .3 Tip
#7Eat ahealth ydiet at all times .4 Vit am ins for hair growth should be
inst illed in the daily diet.<0x0A> <0x0A> pass age 2:1Fats will help you process
the vit am ins and min erals in your diet that are necessary for health yhair
production ,including Vit am ins A,D,E,and K.2 Eat the right kinds of fats .
3 Satur ated fats ,trans fat ty acids ,and hydro gen ated fats are examples of
fats that you should avoid .tdoes ,however ,stop loss of length from the tips ,
as it prevents break ages from split ends .1 How often you should cut your
hair depends on how health yit is and how fast you want it to grow ;it is
totally up to your own jud gement .2 Normal hair grows anywhere from 1⁄2
inch (1.3cm )-5⁄8inch (1.6cm )per month .<0x0A> <0x0A> pass age 3:Cut
your hair .Gets rid of split ends and makes hair appear health ier ,though
this does not make it grow faster .Cut it on anight when the moon is in
phase new moon your hair grows faster or when 1st quarter moon phase .ow
to Im pro ve Hair Grow th.Let ’sface it.W e want long hair and we want itnow !
Although there is no mi racle for rapid hair growth there are always different
options to try .<0x0A> <0x0A> In case the pass ages do not contain the necessary
information to answer the question ,please reply with :”Un able to answer based
on given pass ages . ”<0x0A> output :<s> Based on the provided pass ages ,here
are some ways to improve hair growth :<0x0A> <0x0A> 1.Mass age the scal p
daily to promote circul ation and deliver oxygen to the hair and scal p.(Pass age
1)<0x0A> 2.Avoid over -sty ling the hair ,as it can cause damage and lead
to break age .Reg ular trim ming every 8weeks can help keep the hair health y
and promote length ening .(Pass age 2)<0x0A> 3.Include health yfats in your
diet,such as those found in nuts ,seeds ,and avocados ,as they help the body
process vit am ins and min erals essential for hair growth .(Pass age 2)<0x0A> 4.
Cutting the hair regularly can help get rid of split ends and make the hair
appear health ier ,although it may not make the hair grow faster .(Pass age
3)<0x0A> <0x0A> Un able to answer based on given pass ages .The pass ages
do not provide specific information on how to speed up hair growth .
<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2 A
good flow of blood will help the delivery of oxygen to the hair and scal p,
keeping the cells health yand less prone to break age and damage .3 Tip #
5 Never over -style the hair .Tip #6 A good way to promote hair growth
is to have regular trim ming of the hair every 8 weeks .2 This rids the
hair of un health ysplit ends ,and prom otes the length ening of hair .3 Tip
#7Eat ahealth ydiet at all times .4 Vit am ins for hair growth should be
inst illed in the daily diet.<0x0A> <0x0A> pass age 2:1Fats will help you process
the vit am ins and min erals in your diet that are necessary for health yhair
production ,including Vit am ins A,D,E,and K.2 Eat the right kinds of fats .
3 Satur ated fats ,trans fat ty acids ,and hydro gen ated fats are examples of
fats that you should avoid .tdoes ,however ,stop loss of length from the tips ,
as it prevents break ages from split ends .1 How often you should cut your
hair depends on how health yit is and how fast you want it to grow ;it is
totally up to your own jud gement .2 Normal hair grows anywhere from 1⁄2
inch (1.3cm )-5⁄8inch (1.6cm )per month .<0x0A> <0x0A> pass age 3:Cut
your hair .Gets rid of split ends and makes hair appear health ier ,though
this does not make it grow faster .Cut it on anight when the moon is in
phase new moon your hair grows faster or when 1st quarter moon phase .ow
to Im pro ve Hair Grow th.Let ’sface it.W e want long hair and we want itnow !
Although there is no mi racle for rapid hair growth there are always different
options to try .<0x0A> <0x0A> In case the pass ages do not contain the necessary
information to answer the question ,please reply with :”Un able to answer based
on given pass ages . ”<0x0A> output :<s> Based on the provided pass ages ,here
are some ways to improve hair growth :<0x0A> <0x0A> 1.Mass age the scal p
daily to promote circul ation and deliver oxygen to the hair and scal p.(Pass age
1)<0x0A> 2.Avoid over -sty ling the hair ,as it can cause damage and lead
to break age .Reg ular trim ming every 8weeks can help keep the hair health y
and promote length ening .(Pass age 2)<0x0A> 3.Include health yfats in your
diet,such as those found in nuts ,seeds ,and avocados ,as they help the body
process vit am ins and min erals essential for hair growth .(Pass age 2)<0x0A> 4.
Cutting the hair regularly can help get rid of split ends and make the hair
appear health ier ,although it may not make the hair grow faster .(Pass age
3)<0x0A> <0x0A> Un able to answer based on given pass ages .The pass ages
do not provide specific information on how to speed up hair growth .Figure 5: The attribution of LLM responses is obtained by processing attribution features at a coarse granularity.
The left figure shows the maximum value of the attribution vector for all tokens in the response, while the right
figure shows the average value. The deeper the red color, the greater the token’s contribution to the target response.
however, these tokens should be substantive tokens
in the RAG system that strictly depend on contex-
tual content. In other words, LLMs process sub-
stantive tokens as linking, resulting in faithfulness
hallucination.
Faithfulness hallucination is a semantic phe-
nomenon where LLM responses contain content
that is semantically inconsistent or contextually
absent. When isolated from context, it is chal-
lenging to determine whether LLMs exhibit faith-
fulness hallucination at the token level alone. As
shown in Fig.8, the sample’s golden label is “Have
Hallucination” and the reason is “LOW INTRO-
DUCTION OF NEW INFORMATION. Original:
This might be correct, however, the exact way to
make sesame milk is not directly stated in the pas-
sages. Generative: ...by grinding the seeds into a
fine powder and mixing them with water or other
liquids. (Passage 2)”. However, direct analysis of
the superimposed token-level attribution distribu-
tion makes it difficult to identify the hallucination
fragment. By constructing the target fragment as a
semantic-level reasoning graph using our method
(as shown in Fig.9), we can easily identify that the
information in the target fragment primarily orig-inates from Passage 1, Passage 3, and the LLM’s
previous generation, indicating a high probability
of the faithfulness hallucination.
C LRP Rules
To obtain token-level correlations within LLM, we
backpropagate output correlations through each
transformer layer.
For the densely connected modules within the
transformer architecture, we utilize the local Ja-
cobian matrix Jji=∂zj
∂xi, which represents the
gradient of the layer’s outputz jwith respect to its
input xi, as the weight for assigning correlations.
Specifically, for the multi-layer perceptrons within
the model, they typically consist of a linear layer
followed by an additional nonlinear function:
zj=X
iWjixi+bj
aj=σ(z j)(9)
where Wjirepresents the model’s parameter and σ
denotes the nonlinear function. By linearizing the
linear layer in Equation 9 at any point x∈RN, the

<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2Agood
flow of blood will help the delivery of oxygen to the hair and scal p,keeping
the cells health yand less prone to break age and damage .3 Tip # 5Never
over -style the hair .Tip #6A good way to promote hair growth is to have
regular trim ming of the hair every 8weeks .2 This rids the hair of un health y
split ends ,and prom otes the length ening of hair .3 Tip #7Eat ahealth y
diet at all times .4 Vit am ins for hair growth should be inst illed in the daily
diet.<0x0A> <0x0A> pass age 2:1Fats will help you process the vit am ins and
min erals in your diet that are necessary for health yhair production ,including
Vit am ins A,D,E,and K.2 Eat the right kinds of fats .3 Satur ated fats ,trans
fat ty acids ,and hydro gen ated fats are examples of fats that you should avoid .t
does ,however ,stop loss of length from the tips ,as itprevents break ages from
split ends .1 How often you should cut your hair depends on how health yit
is and how fast you want it to grow ;it is totally up to your own jud gement .
2 Normal hair grows anywhere from 1⁄2inch (1.3cm )-5⁄8inch (1.6cm )
per month .<0x0A> <0x0A> pass age 3:Cut your hair .Gets rid of split ends and
makes hair appear health ier ,though this does not make itgrow faster .Cut it
on anight when the moon isin phase new moon your hair grows faster or when
1st quarter moon phase .ow to Im pro ve Hair Grow th.Let ’sface it.W e want long
hair and we want it now !Although there is no mi racle for rapid hair growth
there are always different options to try .<0x0A> <0x0A> In case the pass ages
do not contain the necessary information to answer the question ,please reply
with :”Un able to answer based on given pass ages . ”<0x0A> output :<s> Based
on the provided pass ages ,here are some
<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2Agood
flow of blood will help the delivery of oxygen to the hair and scal p,keeping
the cells health yand less prone to break age and damage .3 Tip # 5Never
over -style the hair .Tip #6A good way to promote hair growth is to have
regular trim ming of the hair every 8weeks .2 This rids the hair of un health y
split ends ,and prom otes the length ening of hair .3 Tip #7Eat ahealth y
diet at all times .4 Vit am ins for hair growth should be inst illed in the daily
diet.<0x0A> <0x0A> pass age 2:1Fats will help you process the vit am ins and
min erals in your diet that are necessary for health yhair production ,including
Vit am ins A,D,E,and K.2 Eat the right kinds of fats .3 Satur ated fats ,trans
fat ty acids ,and hydro gen ated fats are examples of fats that you should avoid .t
does ,however ,stop loss of length from the tips ,as itprevents break ages from
split ends .1 How often you should cut your hair depends on how health yit
is and how fast you want it to grow ;it is totally up to your own jud gement .
2 Normal hair grows anywhere from 1⁄2inch (1.3cm )-5⁄8inch (1.6cm )
per month .<0x0A> <0x0A> pass age 3:Cut your hair .Gets rid of split ends and
makes hair appear health ier ,though this does not make itgrow faster .Cut it
on anight when the moon isin phase new moon your hair grows faster or when
1st quarter moon phase .ow to Im pro ve Hair Grow th.Let ’sface it.W e want long
hair and we want it now !Although there is no mi racle for rapid hair growth
there are always different options to try .<0x0A> <0x0A> In case the pass ages
do not contain the necessary information to answer the question ,please reply
with :”Un able to answer based on given pass ages . ”<0x0A> output :<s> Based
on the provided pass ages ,here are some ways to
<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2Agood
flow of blood will help the delivery of oxygen to the hair and scal p,keeping
the cells health yand less prone to break age and damage .3 Tip # 5Never
over -style the hair .Tip #6A good way to promote hair growth is to have
regular trim ming of the hair every 8weeks .2 This rids the hair of un health y
split ends ,and prom otes the length ening of hair .3 Tip #7Eat ahealth y
diet at all times .4 Vit am ins for hair growth should be inst illed in the daily
diet.<0x0A> <0x0A> pass age 2:1Fats will help you process the vit am ins and
min erals in your diet that are necessary for health yhair production ,including
Vit am ins A,D,E,and K.2 Eat the right kinds of fats .3 Satur ated fats ,trans
fat ty acids ,and hydro gen ated fats are examples of fats that you should avoid .t
does ,however ,stop loss of length from the tips ,as itprevents break ages from
split ends .1 How often you should cut your hair depends on how health yit
is and how fast you want it to grow ;it is totally up to your own jud gement .
2 Normal hair grows anywhere from 1⁄2inch (1.3cm )-5⁄8inch (1.6cm )
per month .<0x0A> <0x0A> pass age 3:Cut your hair .Gets rid of split ends and
makes hair appear health ier ,though this does not make itgrow faster .Cut it
on anight when the moon isin phase new moon your hair grows faster or when
1st quarter moon phase .ow to Im pro ve Hair Grow th.Let ’sface it.W e want long
hair and we want it now !Although there is no mi racle for rapid hair growth
there are always different options to try .<0x0A> <0x0A> In case the pass ages
do not contain the necessary information to answer the question ,please reply
with :”Un able to answer based on given pass ages . ”<0x0A> output :<s> Based
on
<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2 A
good flow of blood will help the delivery of oxygen to the hair and scal p,
keeping the cells health yand less prone to break age and damage .3 Tip #
5 Never over -style the hair .Tip #6 A good way to promote hair growth
is to have regular trim ming of the hair every 8 weeks .2 This rids the
hair of un health ysplit ends ,and prom otes the length ening of hair .3 Tip
#7Eat ahealth ydiet at all times .4 Vit am ins for hair growth should be
inst illed in the daily diet.<0x0A> <0x0A> pass age 2:1Fats will help you process
the vit am ins and min erals in your diet that are necessary for health yhair
production ,including Vit am ins A,D,E,and K.2 Eat the right kinds of fats .
3 Satur ated fats ,trans fat ty acids ,and hydro gen ated fats are examples of
fats that you should avoid .tdoes ,however ,stop loss of length from the tips ,
as it prevents break ages from split ends .1 How often you should cut your
hair depends on how health yit is and how fast you want it to grow ;it is
totally up to your own jud gement .2 Normal hair grows anywhere from 1⁄2
inch (1.3cm )-5⁄8inch (1.6cm )per month .<0x0A> <0x0A> pass age 3:Cut
your hair .Gets rid of split ends and makes hair appear health ier ,though
this does not make it grow faster .Cut it on anight when the moon is in
phase new moon your hair grows faster or when 1st quarter moon phase .ow
to Im pro ve Hair Grow th.Let ’sface it.W e want long hair and we want itnow !
Although there is no mi racle for rapid hair growth there are always different
options to try .<0x0A> <0x0A> In case the pass ages do not contain the necessary
information to answer the question ,please reply with :”Un able to answer based
on given pass ages . ”<0x0A> output :<s> Based on the provided pass ages ,here
are some ways to improve hair growth :<0x0A> <0x0A> 1.Mass age the scal p
daily to promote circul ation and deliver oxygen to the hair
<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2 A
good flow of blood will help the delivery of oxygen to the hair and scal p,
keeping the cells health yand less prone to break age and damage .3 Tip #
5 Never over -style the hair .Tip #6 A good way to promote hair growth
is to have regular trim ming of the hair every 8 weeks .2 This rids the
hair of un health ysplit ends ,and prom otes the length ening of hair .3 Tip
#7Eat ahealth ydiet at all times .4 Vit am ins for hair growth should be
inst illed in the daily diet.<0x0A> <0x0A> pass age 2:1Fats will help you process
the vit am ins and min erals in your diet that are necessary for health yhair
production ,including Vit am ins A,D,E,and K.2 Eat the right kinds of fats .
3 Satur ated fats ,trans fat ty acids ,and hydro gen ated fats are examples of
fats that you should avoid .tdoes ,however ,stop loss of length from the tips ,
as it prevents break ages from split ends .1 How often you should cut your
hair depends on how health yit is and how fast you want it to grow ;it is
totally up to your own jud gement .2 Normal hair grows anywhere from 1⁄2
inch (1.3cm )-5⁄8inch (1.6cm )per month .<0x0A> <0x0A> pass age 3:Cut
your hair .Gets rid of split ends and makes hair appear health ier ,though
this does not make it grow faster .Cut it on anight when the moon is in
phase new moon your hair grows faster or when 1st quarter moon phase .ow
to Im pro ve Hair Grow th.Let ’sface it.W e want long hair and we want itnow !
Although there is no mi racle for rapid hair growth there are always different
options to try .<0x0A> <0x0A> In case the pass ages do not contain the necessary
information to answer the question ,please reply with :”Un able to answer based
on given pass ages . ”<0x0A> output :<s> Based on the provided pass ages ,here
are some ways to improve hair growth :<0x0A> <0x0A> 1.Mass age the scal p
daily to promote circul ation and deliver oxygen to the hair and scal
<s> Br iefly answer the following question :<0x0A> how to improve hair
growth <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Tip #4T o increase hair growth ,
constantly mass age the scal pdaily to promote circul ation in the scal p.2 A
good flow of blood will help the delivery of oxygen to the hair and scal p,
keeping the cells health yand less prone to break age and damage .3 Tip #
5 Never over -style the hair .Tip #6 A good way to promote hair growth
is to have regular trim ming of the hair every 8 weeks .2 This rids the
hair of un health ysplit ends ,and prom otes the length ening of hair .3 Tip
#7Eat ahealth ydiet at all times .4 Vit am ins for hair growth should be
inst illed in the daily diet.<0x0A> <0x0A> pass age 2:1Fats will help you process
the vit am ins and min erals in your diet that are necessary for health yhair
production ,including Vit am ins A,D,E,and K.2 Eat the right kinds of fats .
3 Satur ated fats ,trans fat ty acids ,and hydro gen ated fats are examples of
fats that you should avoid .tdoes ,however ,stop loss of length from the tips ,
as it prevents break ages from split ends .1 How often you should cut your
hair depends on how health yit is and how fast you want it to grow ;it is
totally up to your own jud gement .2 Normal hair grows anywhere from 1⁄2
inch (1.3cm )-5⁄8inch (1.6cm )per month .<0x0A> <0x0A> pass age 3:Cut
your hair .Gets rid of split ends and makes hair appear health ier ,though
this does not make it grow faster .Cut it on anight when the moon is in
phase new moon your hair grows faster or when 1st quarter moon phase .ow
to Im pro ve Hair Grow th.Let ’sface it.W e want long hair and we want itnow !
Although there is no mi racle for rapid hair growth there are always different
options to try .<0x0A> <0x0A> In case the pass ages do not contain the necessary
information to answer the question ,please reply with :”Un able to answer based
on given pass ages . ”<0x0A> output :<s> Based on the provided pass ages ,here
are some ways to improve hair growth :<0x0A> <0x0A> 1.Mass age the scal p
daily to promote circul ation and deliver oxygenFigure 6: Contribution scores of the next generated token. The upper three boxes show the distribution of attribution
scores for the next generated linking token, while the lower three boxes display the substantive token. The deeper
the red color, the greater the token’s contribution to the last token.
basic LRP rule can be derived:
Rl−1
i=X
jWjixiRl
j
zj(x) +ε(10)
Since element-wise nonlinearity has only a single
input variable and output variable, the decompo-
sition of attribution results is the operation itself.
Therefore, the total input correlation Rl
ican only
be allocated to a single input variable.
Rl−1
i=Rl
i (11)
We apply the Equation 11 to all element-wise oper-
ations of single-input and single-output variables.
For the multi-head attention module in the trans-
former architecture, let Q,K, and Vrepresent the
query, key, and value matrices, respectively. The
scaled dot product attention calculates the attention
weightAand output valueOas follows:
A=softmax(Q·KT
√dk)
O=A·V
softmax j(x) =exj
P
kexk(12)Taylor expansion of the softmax function at refer-
ence pointxyields the following correlation prop-
agation rules:
Rl−1
i=xi(Rl
i−siX
jRl
j)(13)
Among them, sirepresents the i-th output of the
softmax function. For the double matrix multi-
plication part in self-attention computation, since
f(0,0) = 0 holds, it is necessary to decompose
the matrix multiplication into a form without bias
terms. For this purpose, we decompose matrix mul-
tiplication into affine operations containing sum-
mation and bilinear parts containing element-wise
multiplication.
Ojp=X
iAjiVip (14)
For the bilinear multiplication summation part men-
tioned above, we decompose it into element-level
multiplication operations withNinput variables:
fj(x) =NY
ixji (15)
The following uniform correlation propagation
rules can be obtained at the reference point xusing

<s> Br iefly answer the following question :<0x0A> how do iget copies of my
income tax returns <0x0A> Bear in mind that your response should be strictly
based on the following three pass ages :<0x0A> pass age 1:Step 4:Enter form
used .Enter the form you used to file your federal income tax return on line 6.
Ifyou are request ing apersonal income tax return ,it’slikely you file don F orm
1040,1040A or 1040EZ.Y ou must also enter the tax years you are request ing
copies for on line 7.<0x0A> <0x0A> pass age 2:1This trans cript includes your
mar ital status ,the type of return you file d,your adjust ed gross income and
tax able income . Y ou can get trans cript son the web ,by phone or by mail .
T o request trans cript sonline ,go to IRS .gov and use the Order aT rans cript
tool .<0x0A> <0x0A> pass age 3:Y ou may also request acopy of your California
tax return by subm itting aRequest for Copy of Income T ax or Fiduciary
Return F orm FTB 3516or by subm itting your request on plain paper and
provide the following :1 Y our Name (and Sp ouse ifjoint return )Social security
numbers .<0x0A> <0x0A> In case the pass ages do not contain the necessary
information to answer the question ,please reply with :”Un able to answer based
on given pass ages . ”<0x0A> output :<s> Based on the provided pass ages ,here is
how you can obtain copies of your income tax returns :<0x0A> <0x0A> Pass age
1states that to obtain copies of your income tax returns ,you will need to
complete F orm 4506
<s> Br iefly answer the following question :<0x0A> how to heat fro zen baked
scon es<0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:There should be little bits of cold
but ter scattered around the flour .The moment the cold but ter pieces bake ,they
create little po ckets of steam and cause the dough to rise –this issimilar to what
happens when you bake pie crust .It’sthe but ter that makes these cran berry
scon es so tender ,light and del icious .<0x0A> <0x0A> pass age 2:Scon es should
be light and fla ky ,not heavy and flat .W e stick to all -pur pose flour in our
kitchen –it’spretty much our go-to flour .F or this reci pe,we prefer to use soft ,
low protein all -pur pose flour .Prote in levels vary by flour brand .Prote in content
can affect the tend ern ess of baked goods .<0x0A> <0x0A> pass age 3:Make sure
to pre heat the oven to the correct temperature (220°C /200°C fan oven /gas
mark 6)before putting your scon es in.The 15minute guid eline starts from the
point at which you put your scon es in apre heated oven .It may also take abit
longer to bake them through in astandard (non -fan )or gas oven .Y ou will also
need to extend the time ifyou ’re baking more than 12tea -time or 24can ap é
scon es in the oven at the same time .<0x0A> <0x0A> In case the pass ages do
not contain the necessary information to answer the question ,please reply with :
”Un able to answer based on given pass ages . ”<0x0A> output :<s> Based on the
provided pass ages ,here ’show to heat fro zen baked scon es:<0x0A> <0x0A> T o
heat fro zen baked scon es,pre heat your oven to 220°C (200°C fan oven /gas
mark 6).Place the fro zen scon es on abaking tray lined with par chment paper
and bake for 15-2Figure 7: The contribution score of the next generated
token (labeled as hallucination). The deeper the red
color, the greater the token’s contribution to the last
token.
the Shapley method (with a baseline of zero) or the
Taylor decomposition method:
Rl−1
ji=1
NRl
j (16)
Therefore, based on the correlation propagation
calculation formulas of each sub-part in the self-
attention module mentioned above, we can obtain
the following correlation propagation rules in the
self-attention layer:
Rl−1
ji=X
pAjiVipRl
jp
2Ojp+ε(17)
There is no bias term for absorption correlation
in this rule, and the amount of εabsorption can
be ignored. By adopting this rule, we strictly ad-
here to the conservation property in the process of
correlation propagation.
Regarding the correlation propagation rule of
LayerNorm layer, when using common ε= 10−6
andV er[x] = 1 , it actually absorbs 99% of the cor-
relation. Therefore, linearizing at xis meaningless.
<s> Br iefly answer the following question :<0x0A> how to use ses ame
seeds <0x0A> Bear in mind that your response should be strictly based on the
following three pass ages :<0x0A> pass age 1:1Add raw ses ame seeds to any side
dish to add crunch and flav or.2 F or instance ,you can add to veget able or bean
dishes .3 Ses ame oil is agreat option to try to incorpor ate in your cook ing .4
Like almond milk and hemp milk ,you can make your own ses ame milk .Y ou
can also make alight sal ad dress ing out of tah ini .2 Ses ame seeds can also
be roasted ,crushed and then spr ink led over sal ads .3 This will give you more
variety and flav or than your typical sal ad .4 Add raw ses ame seeds to any
side dish to add crunch and flav or.<0x0A> <0x0A> pass age 2:1.T oast over the
stovetop .Ifyou don ’tsee any dust or small pebbles in your ses ame seeds ,you can
put them directly in askillet or frying pan .He at over medium -low heat ,stir ring
occasionally ,for two to three minutes or until the seeds turn brown ,glisten ,
and occasionally pop or jump .1 Do not add oil to the skillet ..T oast over the
stovetop .Ifyou don ’tsee any dust or small pebbles in your ses ame seeds ,you can
put them directly in askillet or frying pan .He at over medium -low heat ,stir ring
occasionally ,for two to three minutes or until the seeds turn brown ,glisten ,and
occasionally pop or jump .<0x0A> <0x0A> pass age 3:Sesame seeds are tiny ,flat
oval seeds with anut ty taste and adel icate ,almost invisible crunch .They come
in ahost of different colors ,depending upon the variety ,including white ,yellow ,
black and red .Ses ame seeds are highly val ued for their high content of ses ame
oil ,an oil that is very resist ant to ranc idity .he hull sof ses ame seeds contain
oxalates .In fact ,most of the calci um found in the seed hull comes in the form
ofcalci um oxalate .The ses ame seed paste (tah ini )found in gro cery stores ismost
often made with seed kern els —the part of the ses ame seed that remains after
the hull has been removed .<0x0A> <0x0A> In case the pass ages do not contain
the necessary information to answer the question ,please reply with :”Un able
to answer based on given pass ages . ”<0x0A> output :<s> Based on the provided
pass ages ,here are some ways to use ses ame seeds :<0x0A> <0x0A> 1.Add raw
ses ame seeds to any side dish to add crunch and flav or.(Pass age 1)<0x0A> 2.
T oast ses ame seeds over the stovetop by heating them over medium -low heat ,
stir ring occasionally ,for two to three minutes or until they turn brown ,glisten ,
and occasionally pop or jump .(Pass age 2)<0x0A> 3.Use ses ame seeds as
a topping for sal ads ,adding a nut ty flav or and crunch y texture .(Pass age
1)<0x0A> 4.Make ses ame milk by grinding the seeds into afine pow der and
mixing them with water or other liqu ids .(Pass age 2)<0x0A> 5.Ro ast ses ame
seeds and use them as atopping for sal ads or other dishes .(Pass age 1)<0x0A> 6.
Use ses ame oil in cook ing for added flav or and health benefits .(Pass age
1)<0x0A> 7.Make alight sal ad dress ing using tah ini ,ases ame seed paste .
(Pass age 2)<0x0A> 8.Add ses ame seeds to baked goods ,such as bread sand
muff ins ,for added flav or and nut rition .(Pass age 1)<0x0A> 9.Use ses ame seeds
as agarn ish for sou ps and other dishes .(Pass age 1)<0x0A> 10.Ex periment with
different types of ses ame seeds ,such as white ,yellow ,black ,and red ,to find the
one that best suits your taste prefer ences .(Pass age 3)<0x0A> <0x0A> Un able
to answer based on given pass ages .The pass ages do not provide information on
how to use ses ame seeds in specific dishes or reci pes .Figure 8: The superimposed token-level attribution dis-
tribution of the response (start with “<s> Based on”).
The deeper the red color, the greater the token’s contri-
bution to the last token. Based solely on the overlap-
ping attribution distribution, it is difficult to determine
whether the response contains hallucination fragments.
By using Taylor expansion to decompose Layer-
Norm or RMSNorm with reference point 0, we can
obtain the same identity association propagation
rule as Equation 11.
By using the correlation backpropagation based
on the above rules, we can obtain the contribu-
tion score of the model input to the probability
value output. Intuitively, the higher the contribu-
tion score, the greater the impact of the input label
on a specific output label. By combining these con-
tribution vectors into a matrix Ri, we can obtain a
correlation matrix that is faithful to the internal in-
ference process of the model. The proof is detailed
in (Achtibat et al., 2024).
D Implementation Details
We employ LXT 2.0 to compute LRP attribution
scores for both Llama and Qwen. For substantive
word extraction, we utilize Spacy and Stanza to

Passage 1 : 3  Sesame oil is 
a great option to try to 
incorporate in your 
cooking.4  Like almond 
milk and hemp milk, you 
can make your own 
sesame milk.
0.22Previous Generate : Use 
sesame seeds as a 
topping for salads, 
adding a nutty flavor 
and crunchy texture.
0.11
Passage 3: The sesame seed 
paste (tahini) found in grocery 
stores is most often made with 
seed kernels \u2014the part of 
the sesame seed that remains 
after the hull has been removed.0.09
Previous Generate: Based on the 
provided passages, here are some 
ways to use sesame seeds0.07Passage 3: Sesame seeds 
are tiny, flat oval seeds with 
a nutty taste and a delicate, 
almost invisible crunch.0.06Make sesame milk by 
grinding the seeds into a 
fine powder and mixing 
them with water or other 
liquids. (Passage 2)Figure 9: The internal reasoning graph constructed
based on the target fragment in the response shown in
Fig. 8. The target fragment clearly originates from pas-
sage 1, passage 3 and previous generation, contradicting
the annotation that it comes from passage 2. Thus, the
fragment within the dashed box is most likely a case of
faithfulness hallucination.
identify nouns, named entities, noun phrases, and
negations in the text. During training, the Align-
Score pre-trained Roberta model with 124M pa-
rameters is used. The batch size is set to 16, and
we perform 100iterations on the training set using
the Adam optimizer at a learning rate of 1e−5 .
The model with the best F1 score is selected for the
final result.
E Perturbation Tests
The perturbation tests employed by Bakish et al.
(2025) are divided into two types of disturbance:
generation and pruning. The generation pro-
cess progressively incorporates semantic fragments
starting from null, ordered by relevance from high-
est to lowest. An approach that faithfully replicates
the real decision-making process of LLMs will
identify the most impactful semantic fragments.
When these fragments are added, the embedding
distance of this model is significantly reduced com-
pared to the previous one, and the logit (calculated
against the predicted target semantic fragment)
shows a marked increase. The pruning method
starts by masking the semantic fragments with the
lowest relevance and gradually progresses towards
the semantic fragments with higher importance.
Removing low-impact semantic fragments should
preserve model prediction stability. Following Ali
et al. (2022), the final metrics were quantified using
Area-Under-Curve (AUC), capturing model accu-
racy relative to the percentage of masked semantic
fragments, from0%to100%.