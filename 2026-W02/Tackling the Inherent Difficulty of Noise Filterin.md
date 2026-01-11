# Tackling the Inherent Difficulty of Noise Filtering in RAG

**Authors**: Jingyu Liu, Jiaen Lin, Yong Liu

**Published**: 2026-01-05 08:40:37

**PDF URL**: [https://arxiv.org/pdf/2601.01896v2](https://arxiv.org/pdf/2601.01896v2)

## Abstract
Retrieval-Augmented Generation (RAG) has become a widely adopted approach to enhance Large Language Models (LLMs) by incorporating external knowledge and reducing hallucinations. However, noisy or irrelevant documents are often introduced during RAG, potentially degrading performance and even causing hallucinated outputs. While various methods have been proposed to filter out such noise, we argue that identifying irrelevant information from retrieved content is inherently difficult and limited number of transformer layers can hardly solve this. Consequently, retrievers fail to filter out irrelevant documents entirely. Therefore, LLMs must be robust against such noise, but we demonstrate that standard fine-tuning approaches are often ineffective in enabling the model to selectively utilize relevant information while ignoring irrelevant content due to the structural constraints of attention patterns. To address this, we propose a novel fine-tuning method designed to enhance the model's ability to distinguish between relevant and irrelevant information within retrieved documents. Extensive experiments across multiple benchmarks show that our approach significantly improves the robustness and performance of LLMs.

## Full Text


<!-- PDF content starts -->

Tackling the Inherent Difficulty of Noise Filtering in RAG
Jingyu Liu1,Jiaen Lin3,Yong Liu1,2*
1Gaoling School of Artificial Intelligence, Renmin University of China, Beijing, China
2Beijing Key Laboratory of Big Data Management and Analysis Methods, Beijing, China
3School of Software Tsinghua University, Beijing, China
liujy1016@ruc.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has
become a widely adopted approach to enhance
Large Language Models (LLMs) by incorporat-
ing external knowledge and reducing hallucina-
tions. However, noisy or irrelevant documents
are often introduced during RAG, potentially
degrading performance and even causing hal-
lucinated outputs. While various methods have
been proposed to filter out such noise, we ar-
gue that identifying irrelevant information from
retrieved content is inherently difficult and lim-
ited number of transformer layers can hardly
solve this. Consequently, retrievers fail to filter
out irrelevant documents entirely. Therefore,
LLMs must be robust against such noise, but
we demonstrate that standard fine-tuning ap-
proaches are often ineffective in enabling the
model to selectively utilize relevant informa-
tion while ignoring irrelevant content due to
the structural constraints of attention patterns.
To address this, we propose a novel fine-tuning
method designed to enhance the model’s ability
to distinguish between relevant and irrelevant
information within retrieved documents. Exten-
sive experiments across multiple benchmarks
show that our approach significantly improves
the robustness and performance of LLMs.
1 Introduction
Large Language Models (LLMs) (Brown, 2020)
have demonstrated remarkable capabilities across
a variety of tasks, including text generation and
question answering (Ouyang et al., 2022; Wei et al.,
2022), code generation (Gu, 2023), and information
retrieval (Dai et al., 2024). However, current LLMs
often suffer from serious hallucinations (Huang
et al., 2023; Choudhary et al., 2025) due to a lack of
factual information. Moreover, the knowledge em-
bedded within LLMs is encoded in their parameters
(Yang et al., 2024), meaning that incorporating new
knowledge requires further fine-tuning, which is
*Corresponding Author.both time-consuming and resource-intensive. Con-
sequently, augmenting LLMs with external retriev-
ers has led to significant performance improve-
ments (Lewis et al., 2020; Liang et al., 2024; Zhao
et al., 2024; Izacard et al., 2023; Zhang et al.,
2025b).
However, in real-world RAG scenarios, the infor-
mation retrieved from documents is not always di-
rectly usable and often requires further processing
because documents may contain noisy information
(Jiang et al., 2023b,a), and some documents may
even be completely distracting, containing incor-
rect answers (Shi et al., 2023a; Wu et al., 2024;
Zhang et al., 2025b; Ding et al., 2025). Such noise
and distracting documents can negatively impact
performance.
Apparently, to improve the performance, we can
either reduce the number of distracting documents
by more advanced retriever (Xu et al., 2024; Yoran
et al., 2023; Yan et al., 2024) or fine-tune the model
(Zhang et al., 2025b; Ding et al., 2025; Yoran et al.,
2023) to make it more robust to noisy information.
Our paper shows that these two methods fail be-
cause,
•Filtering out irrelevant information is inher-
ently difficult, small retrieval models fail to
solve it.
•Fine-tuning the LLM can hardly distinguish
irrelevant information while taking advantage
of relevant ones.
About the difficulty of filtering out irrelevant in-
formation. Considering the query‘Alice and Bob
are running, Bob is exhausted. How does Bob feel?’
Here, assessing the relevance of‘exhausted’neces-
sitates considering the token‘Bob’,‘feel’in the
query alongside‘Bob’which is the subject of‘ex-
hausted’. Thus, evaluating relevance requires the
information from three or even more tokens. Yet,
the attention mechanism typically computes only
1arXiv:2601.01896v2  [cs.CL]  6 Jan 2026

pairwise relationships, making it challenging to
resolve this issue within a limited number of trans-
former layers (Sanford et al., 2024). Therefore fil-
tering out noise documents is inherently difficult
which explains why current small retriever mod-
els (Izacard et al., 2021; Robertson et al., 2009;
Karpukhin et al., 2020) would always incorporate
some noisy information in the retrieval results.
Therefore, a question is can powerful LLMs
solve this problem. We hope that the LLM could be
robust to noisy information while extracting help-
ful information (Yoran et al., 2023; Ding et al.,
2025). However, we argue that standard fine-tuning
is structurally ill-suited for this task. The core issue
lies in a fundamental trade-off imposed by their lin-
ear update mechanism. To filter noise, the learned
parameter update must apply a strong negative ad-
justment to the attention scores of irrelevant tokens.
Yet, this same linear adjustment is applied across
all tokens, which can inadvertently distort the nu-
anced, relative attention patterns among the rele-
vant tokens that are crucial for complex reasoning.
In essence, the model is forced to choose between
effective noise filtering and preserving its reasoning
capacity.
To overcome this limitation, our work decou-
ples these competing objectives. We propose a
novel fine-tuning method that introduces a non-
linear rectification function to the attention update.
This function is specifically designed to operate
in two distinct regimes: for irrelevant tokens, it
creates a sharp, saturating penalty to effectively
"zero them out"; for relevant tokens, it allows for
more gentle adjustments that preserve their rela-
tive importance. This approach enables the model
to aggressively filter noise while simultaneously
safeguarding its core reasoning abilities. Extensive
experiments show that our method significantly im-
proves robustness in noisy RAG settings.
The main contributions of this paper are:
•We reveal that the inherent triple-wise nature
of the noise filtering process may necessitate
numerous transformer layers, which means
we can hardly filter out noise documents with
small models.
•We show that fine-tuning the LLM to filter
out noise while effectively taking advantage
of the relevant information is challenging
•We developed a new fine-tuning method to bet-
ter distinguish irrelevant tokens and extensiveexperiments shows its effectiveness.
2 Related Work
RAG in LLM.Many recent works have explored
better RAG strategies. Shi et al. (2023b) treat the
model as a black box and design a retriever to
enhance performance. However, researchers have
identified that noise in the context can negatively
impact performance (Shi et al., 2023a; Zeng et al.,
2025; Ding et al., 2025), some researchers focus on
eliminating this noisy information and compress
the noisy documents(Jiang et al., 2023b; Liu et al.,
2025; Zhang et al., 2025a; Liskavets et al., 2025).
And some others try to fine-tune the large language
model to make it more robust to noisy informa-
tion(Yoran et al., 2023; Zhang et al., 2024, 2025b).
Filtering the noise.Yoran et al. (2023) tries to
fine-tune the LLM to better filter out distracting
documents, while RAFT (Zhang et al., 2024) uses
different proportion of distracting documents to
make it more robust. Self-RAG(Asai et al., 2023),
use special tokens to indicate whether the LLM
should use the document information.. Xu et al.
(2024) first highlighted the duality of RAG, encom-
passing both benefits and detriments. Also, various
researchers propose new fine-tuning methods to
make the LLM robust to noisy information (Wang
et al., 2023; Ding et al., 2025; Zhang et al., 2025b).
Also Ding et al. (2025) finds that some specifically
designed fine-tuning methods helps less in mod-
ern models. But they fail to understand why noise
filtering is difficult in current LLMs. Differential
Transformer (Ye et al., 2025) calculates the atten-
tion score as difference between two separate soft-
max attention to cancel noise. Also CrAM (Deng
et al., 2024) tries to adjust attention weights given
the credibility of different documents.
3 The Triple-Wise Problem
Clearly, noise in the documents adversely affects
the performance of large language models (LLMs).
And various researchers are trying to develop new
retrieval methods to reduce the number of noisy
information, and some others try to develop a fil-
tering model to filter out the irrelevant information
before input to LLM. In this section, we show that
filtering out noise documents is inherently a com-
plex problem. It is difficult to filter out noise by
small model based rerankers/retrievers.
For input X= [xT
0,xT
1, . . . ,xT
n−1]with the
query and document, and we should decide is the
2

Bob is exhausted, How does Bob feel?check the subject
asks for feeling,
relevantis/is't
f(exhausted,feel)=g(Bob,is,exhausted,Bob,feel)
pairwise attn understand sentenceFigure 1: Judging the relevance of the token ’exhausted’
actually requires checking the whole meaning of the
sentence, so it can hardly be done by limited number of
transformer layers.
document relevant to query. It’s crucial to note
that calculating the relevance of the document re-
quire the involvement of many tokens. For instance,
in the query“Alice and Bob are running, Bob
is exhausted. How does Bob feel?”, the context
“exhausted”describes the feeling of Bob, which
serves as a useful contenxt. However, determining
relevance necessitates considering“Bob”,“feel”
in the query alongside“Bob”, which is the sub-
ject of“exhausted”, only in this way the model
can understand that “exhausted” describes Bob and
the query asks for the feeling of Bob. We show
it more clearly in Figure 1. Therefore, identifying
the relevance of a token demands information from
multiple tokens. However, self-attention computes
relationships only between pairs, making it chal-
lenging to effectively address this issue.
As self attention only calculates pair-wise rela-
tionship between tokens, and judging the relevance
of the document requires the involvement of multi-
ple tokens. It is necessary to stack multiple atten-
tion layers to aggregate information from different
tokens to conduct judgment. However, as noted in
Sanford et al. (2024), To effectively solve the triple-
wise problem, we need the multi-layer transformer
to have width, depth, embedding dimension, or bit
complexity at least NΩ(1)c, where Nis context
length and cis a constant represents the embedding
dimension required to represent noise filtering in-
formation. This is impractical for small models as
it only shows limited depth, width and embedding
dimension.
The challenge arises from the triple-wise nature
of the problem contrasted with the pairwise nature
of attention; the model can only evaluate a token’srelevance when its embedding contains substantial
information about the whole context. For exam-
ple, as shown in Figure 1, to assess the token“ex-
hausted”in the sentence“Bob is exhausted. How
does Bob feel?”, the embedding must encompass
information about its subject of“exhausted”and
the subject of“feel”,“Bob”, as well as contex-
tual details from phrases like“is/isn’t”and“feel-
ing”. Therefore, a significant amount of informa-
tion must be incorporated into the embedding be-
fore any judgment can be made, it actually requires
the understanding about the meaning of the whole
sentence before judging the relevance of the token.
However, a single layer of self-attention can only
consider the input of fixed dimension embedding,
which contains information up to mp, where mrep-
resents the embedding dimension and pindicates
precision of embedding. The term mpsignifies the
maximal information carried by the embedding.
But this can hardly represent the meaning of the
whole sentence, especially when the input sequence
is long.
This indicates that trying to filter out irrelevant
context is difficult, the input to LLM would like
to incorporate noisy information in the retrieved
documents. However, current large language mod-
els holds great embedding dimension (4096 for
Llama3-8B) and depth (32 for Llama3-8B), which
should be able to solve the triple-wise problem
theoretically. This suggests the burden of noise
filtering should shift from the limited-capacity re-
triever to the powerful LLM itself. However, as we
will show next, making the LLM robust is not a
straightforward task.
4 Robustness of LLM When Faced with
Noise
Clearly, noise in the documents negatively impacts
the performance of LLMs, and noisy information is
inevitable in RAG. Therefore some researchers fo-
cus on fine-tuning the model to make it more robust
when faced with noisy information thereby enhanc-
ing performance (Yoran et al., 2023; Zhang et al.,
2025b; Ding et al., 2025; Jiang et al., 2023b). And
there is a question that, can the fine-tuned LLM ef-
fectively filter out irrelevant information and gather
useful information to get the final answer?
Letrrepresents the relevance of tokens, ri=
0means that token xiis a noise token, other-
wise the token is relevant. Let attn(x i,xj) =
(Wqxi)TWkxjrepresent the attention pattern
3

which is trained to extract relevant information.
And we want to filter out irrelevant information
while preserving the attention pattern on relevant
tokens, this can be represented as following:
σ
dattn(x i,xj)
=(
0ifr j= 0,
σ 
(Wqxi)TWkxj
else,
where σmeans the softmax function. This shows
that the desired attention pattern should effectively
exclude noise while preserving the attention pattern
of relevant tokens.
Therefore, dattn can be considered the optimal
response when confronted with noise, as it effec-
tively filters out irrelevant tokens and utilizes the
relevant information in the most efficient manner.
Fine-tuning the model involves adjusting its pa-
rameters and attention pattern, which allows the
fine-tuned model to be expressed as
attn′(xi,xj) = ((W q+ ∆W q)xi)T(Wk+ ∆W k)xj
=xT
i(W+ ∆W)x j,
where W=WT
qWkand∆W represents the
adjustments. The critical question arises: can we
fine-tune the model to approximate the optimal one,
i.e., is there a ∆W such that σ(attn′(xi,xj))≈
σ(dattn(x i,xj))?σ(·)represents the softmax.
Theorem 4.1.if there exists
attn′(xi,xj) =x i(W+ ∆W)x j,
ϵapproximates dattn(x i,xj)i.e.,
1−ϵ≤σ(attn′(xi, xj))
σ(dattn(x i, xj))≤1 +ϵ,
whereσrepresents softmax, then we need
ξr≲ln1
1−ϵ,
where ξr= max(xT
i∆Wx j)−min(xT
i∆Wx k),
andx j,kis relevant tokens.
Apparently, if we want to approximate the op-
timal reasoning pattern, we need ξr≈0, soξr≲
ln1
1−ϵ≈0. This shows that, to effectively fine-tune
the model to filter out irrelevant information in the
attention matrix, we need ξr≈0. This implies
that for all tokens to be retained, xT
i∆Wx jmust
remain nearly constant. A solution could make||∆W|| small or even 0, but this actually also re-
quires to split the tokens to be retained and to be
filtered, so a small ||∆W|| does not solve the prob-
lem as shown in Appendix B.2. As a result, ap-
proximating the optimal solution proves to be quite
challenging because the attention will be disturbed
due to the noise filtering fine-tuning.
One might argue that preserving the original at-
tention pattern is unnecessary. Perhaps the model
could learn a new, superior attention pattern during
fine-tuning that excels at both filtering noise and
leveraging semantic information. However, this per-
spective overlooks a fundamental conflict between
these two objectives.
To conduct the two task, we require the input
to attention contain two different kinds of infor-
mation, noise filtering and semantic reasoning. As
suggested by Kawaguchi et al. (2023), a model with
finite capacity attempts to optimize for two differ-
ent tasks would show suboptimal performance on
both task because the two different tasks requires
different information, and each act as noise to each
other, so when the model is trying to filter out noise,
but some semantic information is also incorporated
into the input, then it will downweight the perfor-
mance on the filtering process and vice versa.
This also fits for the feed forward layers. It can-
not be an expert filter and an expert reasoner at the
same time when its input is already contaminated.
Therefore, relying on the FFN to clean up the mess
made by the attention layer is an inefficient strategy
that compromises the model’s overall performance.
We show more analysis in Appendix B.3
It is worth noting that although the analysis
mainly focus on single head attention, for multi
head attention, we can use one head focus on filter-
ing out noise and another head focus on taking ad-
vantage of the relevant information. This is indeed
ideal, but fine-tuning based on existing LLMs fails
to greatly influence the existing parameters, other-
wise it will cause catastrophic forgetting (Huang
et al., 2024; Luo et al., 2025).
5 Fine-tuning for Noise Filtering
5.1 Attention Rectification
Conventional fine-tuning paradigms, primarily ad-
just the model’s behavior by introducing an update
matrix, ∆W , to the original weight matrix W.
The modified attention score between a query to-
keniand a key token jis computed based on their
embeddings, xiandxj. However, as noted in The-
4

y=5tanh(x)Relevant
Tokens
Irrelevant
TokensMargin
sharp
increase/ decreasey=xFigure 2: plot of5tanh(x), this shows that, by tanh, we
can effectively enlarge the margin between relevant and
irrelevant tokens and maintain similar attention weight
to those relevant ones.
orem 4.1, a simple linear addition of the update
termxT
i∆Wx jfaces a fundamental trade-off: it
struggles to simultaneously (1) create a sufficiently
large margin between the attention scores of rel-
evant and irrelevant tokens, and (2) preserve and
optimize the nuanced, relative attention patterns
among multiple relevant tokens, which is crucial
for the model’s intrinsic reasoning capabilities.
To address this, we propose to augment the stan-
dard attention mechanism with a non-linear rectifi-
cation function, g(), applied to the attention update.
The final attention is computed as:
attn′(xi,xj) =xT
iWx j+g(xT
i∆Wx j).(1)
The core of our method lies in the design of the
rectification function g(x). The function g(x) is de-
signed to operate in two distinct regimes: a filtering
regime for low-relevance tokens and a refinement
regime for high-relevance tokens. This is achieved
by the following formulation:
g(x) =(
max(ξ·tanh(x), x)ifx >= 0,
min(ξ·tanh(x), x)else,
where ξis a hyperparameter that controls the
saturation threshold. The behavior of g(x) is illus-
trated in Figure 2. For small or negative values of
the attention update x, the term ξ∗tanh(x) domi-
nates. The hyperbolic tangent function offers two
key advantages:
•Sharp Discrimination: The steep gradient of
tanh(x) around x= 0 creates a sharp tran-sition, effectively amplifying the margin be-
tween positive (potentially relevant) and nega-
tive (irrelevant) attention updates. This serves
as a powerful mechanism for filtering out
noise as a large negative attention could be
allocated to unrelated tokens.
•Saturating Behavior: As xbecomes suffi-
ciently large, tanh(x) approaches 1, causing
the output to saturate at ξ. This ensures that all
highly relevant tokens receive a consistent and
significant attention boost, preventing their
original relative attention scores (after Soft-
max) from being severely distorted. The scal-
ing factor ξis chosen to be large enough to
establish a clear margin, determined by the
typical attention score variance in the base
model.
In this way, we can effectively separate the rele-
vant and irrelevant tokens during the sharp discrim-
ination, and we can ensure that the relevant tokens
share similar attention score.
However, simply using the ξ·tanh(x) may not
be enough, a key limitation of using ξ·tanh(x)
alone is that it clamps the attention boosts for all
highly relevant tokens to a single value ξ, preclud-
ing any further fine-grained adjustments among
them. Our composite function g(x) overcomes this.
When the attention update xis large enough to
exceed ξ·tanh(x) ,g(x) reverts to a linear iden-
tity which means g(x) =x . This linear growth
phase allows the model to continue differentiating
among highly relevant tokens, enabling it to learn a
more optimal attention distribution for the specific
downstream task, rather than merely preserving the
original one.
From a regularization standpoint. We actually re-
quire the attention module to conduct two tasks,
nose filtering and relevant information aggrega-
tion. The two tasks may require different infor-
mation to process and they are actually noise to
each other. In this way, an unconstrained linear
update might learn to assign attention with high
variance scores based on such noise as shown in
Kawaguchi et al. (2023). The saturating nature of
thetanh component in g(x) acts as a "soft clamp"
constraining these potentially high variance values
within a bounded range ( ξ). This restriction pre-
vents the model from becoming overly sensitive
to noisy features, thereby enhancing its robustness
and generalization performance. For instance, an
attention update that might vary between ξ±δ with
5

only linear update, and with the tanh activation, it
would be regularized toξ±ϵ, whereϵ << δ.
Also, max(·) andmin(·) is not a continuous
function, so we use
g(x) =(
max(ξ·tanh(x), x)ifx≥0,
min(ξ·tanh(x), x)else,
≈log(exp(a) + exp(b) + 1)
−log(exp(−a) + exp(−b) + 1),
where a=ξ·tanh(x) ,b=x . Apparently,
when a, b >= 0 ,g(x) will be dominated by
log(exp(a) + exp(b) + 1) , which is approximately
max(a, b) . Otherwise if a, b <0 ,g(x) will be
dominated by −log(exp(−a) + exp(−b) + 1) ,
which is approximatelymin(a, b).
5.2 The Auto-Regressive Nature may Cause
Problem
With the activation, the attention module can effec-
tively filter out noise information. However, most
LLMs are trained in an auto-regressive manner,
meaning that a token cannot aggregate information
from any tokens that appear later in the sequence;
thus,ai,j= 0ifj > i . Typically, the query is posi-
tioned after the document tokens, preventing these
document tokens from assessing their relevance ef-
fectively because they have no access to the query.
Consequently, the relevance judgment must occur
during the calculation of the query embeddings,
which is usually much shorter than the document,
making the nose filtering harder because we need
to judge the relevance of ndoctokens during the
calculation ofn query tokens.
Instead, if we position the query ahead of docu-
ments, then the relevance can be effectively calcu-
lated during the calculation document token embed-
ding. This arrangement enables the information of
query to be effectively transferred to the document
tokens for relevance judgment. In this way we can
judge the relevance of ndoctokens during the calcu-
lation of ndoctokens, as ndocis usually much larger
thannquery , so placing the query ahead could help.
Therefore, we can hypothesize that when there is
noise in the document, placing the query at the
beginning would help the judgment. We conduct
experiments on various datasets in Appendix A to
show that placing the query ahead can effectively
help the performance.6 Experiments
6.1 Datasets and Metrics
To evaluate the performance of our proposed
method, we use Nature Questions (Kwiatkowski
et al., 2019), TriviaQA (Joshi et al., 2017) which is
traditionally used to evaluate the noise robustness
of RAG system. Also, we use multi hop reasoning
datasets HotpotQA (Yang et al., 2018) and 2Wiki-
MultiHopQA (Ho et al., 2020) as well as the long
form QA dataset ASQA (Stelmakh et al., 2022)
to show the performance of our method. For the
first 4 datasets, we use accuracy to measure the
performance which is determined by whether the
predicted answer contains the ground-truth answer.
For ASQA, we measure the percentage of short
answers are shown in the generated answer to eval-
uate the performance.
For all 5 datasets, we use Dense Passage Re-
triever (Karpukhin et al., 2020) as the retriever, we
retrieve some documents and select the first 3 docu-
ments that are not presented in the gold documents
as the noisy documents, then combine the noisy
documents with the gold documents as the input to
the LLM, showing that our method can effectively
distinguish distracting documents while taking ad-
vantage of relevant ones.
For the first 4 datasets, we randomly select 3000
samples to test and another 7000 to train. For
ASQA we use the split of ALCE (Gao et al., 2023)
and use the 948 samples to test the performance
and another 4000 for training. More Experimental
settings can be seen in Appendix A
6.2 Implementation Details
When calculating ∆W , we use Low Rank Adap-
tion, which means we actually calculates ∆W=
A·B ,A∈Rh×r, where his the hidden dimension
andris the rank, in our experiments, we set r= 64 .
For the experiments, if not otherwise specified, we
position the query before the documents.
When calculating the attention, we use Group
Query Attention like used in the LLaMA architec-
ture. And we use Low Rank Adaption with rank
64, so the parameters for training are the same. We
conduct our experiments on Llama-3.1-8B-Instruct,
Qwen/Qwen2.5-7B-Instruct, mistralai/Mistral-7B-
Instruct-v0.2.
It is worth noting that, our method does not need
to calculate another attention matrix and add the
new attention matrix to the previous one. We can
directly add the activation to the existing attention
6

NQ TriviaQA HotpotQA 2wiki ASQA
reverse vanilla reverse vanilla reverse vanilla reverse vanilla reverse vanilla mean
vanilla 52.4 46.7 52.1 45.6 61.4 54.3 53.7 52.1 42.0 41.8 50.2
LoRA 65.7 62.6 72.6 71.3 86.1 85.6 95.3 96.4 42.3 42.2 72.0
ξ= 164.9 64.4 73.9 73.1 86.6 86.1 95.6 96.5 44.3 44.3 73.0
ξ= 366.7 64.2 74.1 73.187.3 86.397.6 97.247.8 46.474.1
ξ= 567.664.874.5 73.787.1 86.1 97.397.547.2 46.1 74.2
ξ= 1066.965.474.1 73.4 87.2 85.897.997.4 47.3 46.274.2
Table 1: Performance of our fine-tuning method when faced with explicit distracting documents. reverse means we
place the query ahead of documents and vanilla means the query is placed after documents
NQ Trivia Hotpot 2wiki ASQA
Qwen 7Bvanilla 53.5 59.9 74.8 67.1 43.5
LoRA 61.4 69.2 82.3 95.2 46.5
tanh62.7 69.9 84.3 96.3 48.3
Mistral 7Bvanilla 52.3 63.5 71.6 58.6 46.0
LoRA 62.3 70.0 84.2 96.2 48.8
tanh66.3 72.4 86.7 96.8 51.3
Table 2: The performance for Qwen2.5-7B ( ξ= 5 ) and
Mistral-7B (ξ= 3) with our activation function
matrix. However, directly add the activation would
greatly disrupt the attention patter, causing bad per-
formance, therefore, we set ξto be 0at first so our
method becomes vanilla attention, then during the
training process, we gradually increase ξlinearly
for the first 80% steps and keeps ξa constant for
the last 20% steps. And besides LoRA fine-tuning,
We conduct full fine-tuning under this setting and
the result is shown in Table 5.
We compare our method with LoRA mainly be-
cause our method focus on how to adjust the at-
tention schema to make the model more robust to
noise. But current researchs primarily focuses on
either how to structure training data (Yoran et al.,
2023; Ding et al., 2025) or how to train models to
better handle different types of noise (Fang et al.,
2024). In contrast, our work addresses a different
challenge: the standard self-attention mechanism
inherently struggles to filter out noisy informa-
tion So we modify the attention computation by
incorporating a non-linear activation function to
enhance robustness. Differential Transformer (Ye
et al., 2025) also tries to adjust attention, but re-
quires to train the model from scratch instead of
fine-tuning based on existing model, and CrAM
does not involve fine-tuning, so the comparison is
unfair.6.3 Main Results
Table 1 shows the performance when faced with
explicit distracting documents, we evaluate two
different setting where the query is placed behind
the documents or ahead the documents. The result
shows that with g(x), the model can better distin-
guish between relevant and distracting documents,
showing better performance. Also, we can observe
that after fine-tuning, placing the query ahead of
documents still helps. The result shown in Table 1
might be high especially for 2wiki, this is mainly
because we add gold documents to the context to
show how can the model grab useful information
from context. We also show the performance when
all documents are retrieved in Table 4, it also shows
that our method performs well.
And we can observe that the hyperparameter ξ
actually does not require specific tuning, we can di-
rectly set the value based on the inherent attention
weight margin of the model, which means the gap
between high attention scores and low attention
scores. If we set ξto make it cover the margin of at-
tention scores, then our method can work well. We
show the margin of the original attention weight of
Llama 3.1-8B-Instruct in Figure 5, we can observe
that most of the margins of Llama lies about 6, so
withξ= 3 , (margin between (-3,3)), the model can
effectively distinguish irrelevant tokens, and larger
values like ξ= 5 orξ= 10 also performs good as
we show in Table 1.
We also conduct experiments on Qwen/Qwen2.5-
7B-Instruct and mistralai/Mistral-7B-Instruct-v0.2,
we show the performance in Table 2, which shows
that our method also effectively helps the perfor-
mance on Qwen2.5-7B-Instruct and Mistral-7B-
Instruct. We also show the performance with full
fine-tuning in Table 5.
7

NQ TriviaQA HotpotQA 2wiki
Dataset0.20.40.60.81.01.21.4Score
0.34
0.070.310.741.34
0.92 0.931.24Llama
NQ TriviaQA HotpotQA 2wiki
Dataset0.07 0.060.110.080.24
0.120.24
0.09Qwen
NQ TriviaQA HotpotQA 2wiki
Dataset0.57
0.320.56
0.101.13
0.641.04
0.21MistralFigure 3: The difference of attention score on answer tokens (mean(attn(answer))-mean(attn(other)), for clarity, we
scale this gap by a factor of 1,000.
NQ TriviaQA HotpotQA 2wiki ASQA mean
Dataset405060708090100Score65.171.684.296.6
41.371.8
64.372.484.795.9
42.772.0
66.773.887.496.6
47.474.4
66.774.187.397.6
47.874.7Performance Comparison of Aggregation Methods
Method
sum
tanh
sigmoid
ours
Figure 4: The performance when we set different activa-
tion function, sum(ξ·tanh(x), x) (sum), ξ·tanh(x)
(tanh). We also show the performance when we use
g(x) =max/min(x,2ξ·(sigmoid(x)−0.5)) (sig-
moid).ξis set to 3, and ours means our method.
6.4 The Difference of Attention Score
By adding the activation function, our method
can effectively distinguish between useful infor-
mation and noise, to show this, we calculate
the attention score gap between the tokens con-
taining answer and other tokens (after softmax,
mean(attn(answer))-mean(attn(other)), for clarity,
we scale this gap by a factor of 1,000, the result
is show in Figure 3. We do not show the result of
ASQA because it is a long form QA, the answer is
not directly indicated in the documents. The result
shows that the rectification helps the model to rec-
ognize the answer and filter out noisy information,
which explains why our method helps.
6.5 Ablation Study with Different Activation
We also conduct experiments when g(x) =ξ·
tanh(x) andg(x) =ξ·tanh(x) +x .g(x) =
tanh(x) stands for the situation that g(x) only fo-
cus on enlarge the margin between relevant and ir-
ASQA NQ TriviaQA 2wiki Hotpot
Dataset5.05.56.06.57.07.58.08.5MarginMargin of Datasets
0 5 10 15 20 25 30
Layer5.56.06.57.07.58.0
Margin of Different Layers
ASQA
NQ
TriviaQA
2wiki
HotpotFigure 5: The margin of attention scores. We calculate
the margin as the difference between the 90th and 10th
percentile attention scores to reduce the influence of
outliers
relevant tokens without further linear growth. And
g(x) =tanh(x) +x stands for the situation where
the the steady growth process is missing, it rapidly
increase with x, so the soft clamp will not work.
We also try using
g(x) =(
max(ξ·(sigmoid(x)−0.5)), x)ifx >= 0,
min(ξ·(sigmoid(x)−0.5)), x)else,
We use sigmoid activation instead of tanh and show
the performance. As shown in Figure 4, simply use
tanh(x) ortanh(x) +x has suboptimal perfor-
mance, this is mainly because they fail to optimize
the attention pattern on relevant tokens or miss-
ing the saturating behavior. Also we can observe
that replacing tanh withsigmoid helps the perfor-
mance, this is mainly because sigmoid is actually
quite similar with tanh , they all increases fast at
the beginning and result in a steady region after the
growth.
7 Conclusion
In this paper, we highlight that noise filtering in
RAG is inherently difficult, limited number of trans-
former layers can not effectively solve it, so we
8

require the LLM to be robust to noise information.
Then we show that simply fine-tuning the LLM
may not be optimal as it will disturb the attention
pattern. Then we propose a new fine-tuning method
which can help to be more robust to noise, exten-
sive experiments show that our method works well,
it can effectively filter out noise while taking ad-
vantage of relevant information.
Limitations
The paper discusses the limitation of LLM when
dealing with noisy information, showing that cur-
rent LLMs can not effectively process noisy in-
formation. However, although a new fine-tuning
method is proposed, it can not fully address the
problem as it is a fine-tuning based on the trained
model. It might help more if we train a model from
scratch, but due to limited computational resource,
this can only leave for future work.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
arXiv preprint arXiv:2310.11511.
Tom B Brown. 2020. Language models are few-shot
learners.arXiv preprint ArXiv:2005.14165.
Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Kr-
ishnamurthy Dj Dvijotham, and Somesh Jha. 2025.
Through the stealth lens: Rethinking attacks and de-
fenses in rag.arXiv preprint arXiv:2506.04390.
Sunhao Dai, Chen Xu, Shicheng Xu, Liang Pang, Zhen-
hua Dong, and Jun Xu. 2024. Bias and unfairness in
information retrieval systems: New challenges in the
llm era. InProceedings of the 30th ACM SIGKDD
Conference on Knowledge Discovery and Data Min-
ing, pages 6437–6447.
Boyi Deng, Wenjie Wang, Fengbin Zhu, Qifan Wang,
and Fuli Feng. 2024. Cram: Credibility-aware atten-
tion modification in llms for combating misinforma-
tion in rag.Preprint, arXiv:2406.11497.
Hanxing Ding, Shuchang Tao, Liang Pang, Zihao Wei,
Liwei Chen, Kun Xu, Huawei Shen, and Xueqi
Cheng. 2025. Revisiting robust rag: Do we still need
complex robust training in the era of powerful llms?
arXiv preprint arXiv:2502.11400.
Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xi-
aojun Chen, and Ruifeng Xu. 2024. Enhancing
noise robustness of retrieval-augmented language
models with adaptive adversarial training.Preprint,
arXiv:2405.20978.Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
2023. Enabling large language models to generate
text with citations.arXiv preprint arXiv:2305.14627.
Qiuhan Gu. 2023. Llm-based code generation method
for golang compiler testing. InProceedings of the
31st ACM Joint European Software Engineering Con-
ference and Symposium on the Foundations of Soft-
ware Engineering, pages 2201–2203.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Jianheng Huang, Leyang Cui, Ante Wang, Chengyi
Yang, Xinting Liao, Linfeng Song, Junfeng Yao, and
Jinsong Su. 2024. Mitigating catastrophic forget-
ting in large language models with self-synthesized
rehearsal.arXiv preprint arXiv:2403.01244.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2023. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.arXiv preprint arXiv:2311.05232.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023a. Llmlingua: Compressing
prompts for accelerated inference of large language
models.arXiv preprint arXiv:2310.05736.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng
Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023b.
Longllmlingua: Accelerating and enhancing llms
in long context scenarios via prompt compression.
arXiv preprint arXiv:2310.06839.
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke
Zettlemoyer. 2017. Triviaqa: A large scale distantly
supervised challenge dataset for reading comprehen-
sion.arXiv preprint arXiv:1705.03551.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
9

Kenji Kawaguchi, Zhun Deng, Xu Ji, and Jiaoyang
Huang. 2023. How does information bottleneck help
deep learning? InInternational Conference on Ma-
chine Learning, pages 16049–16096. PMLR.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu
Zhu, Zhouyu Jiang, Ling Zhong, Yuan Qu, Peilong
Zhao, Zhongpu Bo, Jin Yang, and 1 others. 2024.
Kag: Boosting llms in professional domains via
knowledge augmented generation.arXiv preprint
arXiv:2409.13731.
Barys Liskavets, Maxim Ushakov, Shuvendu Roy, Mark
Klibanov, Ali Etemad, and Shane K Luke. 2025.
Prompt compression with context-aware sentence
encoding for fast and improved llm inference. In
Proceedings of the AAAI Conference on Artificial
Intelligence, volume 39, pages 24595–24604.
Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng
He, Huanxuan Liao, Haoran Que, Zekun Wang,
Chenchen Zhang, Ge Zhang, Jiebin Zhang, and
1 others. 2025. A comprehensive survey on
long context language modeling.arXiv preprint
arXiv:2503.17407.
Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou,
and Yue Zhang. 2025. An empirical study of catas-
trophic forgetting in large language models during
continual fine-tuning.IEEE Transactions on Audio,
Speech and Language Processing.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, and 1
others. 2022. Training language models to follow in-
structions with human feedback.Advances in neural
information processing systems, 35:27730–27744.
Stephen Robertson, Hugo Zaragoza, and 1 others. 2009.
The probabilistic relevance framework: Bm25 and
beyond.Foundations and Trends® in Information
Retrieval, 3(4):333–389.
Clayton Sanford, Daniel J Hsu, and Matus Telgarsky.
2024. Representational strengths and limitations of
transformers.Advances in Neural Information Pro-
cessing Systems, 36.Jonathan Scarlett and V olkan Cevher. 2019. An in-
troductory guide to fano’s inequality with appli-
cations in statistical estimation.arXiv preprint
arXiv:1901.00555.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed Chi, Nathanael Schärli, and
Denny Zhou. 2023a. Large language models can
be easily distracted by irrelevant context.Preprint,
arXiv:2302.00093.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Rich James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2023b. Replug: Retrieval-
augmented black-box language models.arXiv
preprint arXiv:2301.12652.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. Asqa: Factoid questions meet long-
form answers.arXiv preprint arXiv:2204.06092.
Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan
Parvez, and Graham Neubig. 2023. Learning to filter
context for retrieval-augmented generation.arXiv
preprint arXiv:2311.08377.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
and 1 others. 2022. Chain-of-thought prompting elic-
its reasoning in large language models.Advances
in neural information processing systems, 35:24824–
24837.
Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai
Zhang, and Yanghua Xiao. 2024. How easily do
irrelevant inputs skew the responses of large language
models?Preprint, arXiv:2404.03302.
Shicheng Xu, Liang Pang, Huawei Shen, and Xueqi
Cheng. 2024. Unveil the duality of retrieval-
augmented generation: Theoretical analysis and prac-
tical solution.arXiv preprint arXiv:2406.00944.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
arXiv preprint arXiv:2401.15884.
Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu,
Zhiyu Li, Bo Tang, Wenqiang Wei, Jinbo Wang,
Zeyun Tang, Shichao Song, and 1 others. 2024.
Memory3: Language modeling with explicit memory.
arXiv preprint arXiv:2407.01178.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.Preprint, arXiv:1809.09600.
Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu,
Gao Huang, and Furu Wei. 2025. Differential trans-
former.Preprint, arXiv:2410.05258.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2023. Making retrieval-augmented language
models robust to irrelevant context.arXiv preprint
arXiv:2310.01558.
10

Linda Zeng, Rithwik Gupta, Divij Motwani, Diji Yang,
and Yi Zhang. 2025. Worse than zero-shot? a fact-
checking dataset for evaluating the robustness of
rag against misleading retrievals.arXiv preprint
arXiv:2502.16101.
Gongbo Zhang, Zihan Xu, Qiao Jin, Fangyi Chen, Yilu
Fang, Yi Liu, Justin F Rousseau, Ziyang Xu, Zhiy-
ong Lu, Chunhua Weng, and 1 others. 2025a. Lever-
aging long context in retrieval augmented language
models for medical question answering.npj Digital
Medicine, 8(1):239.
Qianchi Zhang, Hainan Zhang, Liang Pang, Ziwei Wang,
Hongwei Zheng, Yongxin Tong, and Zhiming Zheng.
2025b. Finefilter: A fine-grained noise filtering mech-
anism for retrieval-augmented large language models.
arXiv preprint arXiv:2502.11811.
Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng
Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonza-
lez. 2024. Raft: Adapting language model to domain
specific rag.arXiv preprint arXiv:2403.10131.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren
Wang, Yunteng Geng, Fangcheng Fu, Ling Yang,
Wentao Zhang, and Bin Cui. 2024. Retrieval-
augmented generation for ai-generated content: A
survey.arXiv preprint arXiv:2402.19473.
A Experiments
A.1 Experiment settings
When conduct fine-tuning, we use learning rate of
1e-4, and we use kaiming initialization to initialize
the parameter with a=√
5. The experiments is
conducted on 8 NVIDIA A100 80GB.
For NQ, TriviaQA, HotpotQA and 2Wiki, we
randomly select 3000 samples to test and another
7000 to train. For ASQA we use the split of ALCE
(Gao et al., 2023) and use the 948 samples to test
the performance and another 4000 for training.
Also, we train 3 epochs for each dataset except
ASQA due to its limited number of data, we train it
for 5 epochs instead. We train the model with batch
size 8.
We use DPR as the retriever and retrieve top 3
noise documents as noise. Then we mix it with
the gold documents and shuffle the documents ran-
domly as the input context.
A.2 Placing the Query Ahead Helps
Here we show that placing the query ahead of the
documents can help the performance, we conduct
experiments on NQ, TriviaQA, HotpotQA, 2Wiki
and ASQA. we conduct evaluation on 3000 sam-
ples for the first 4 datasets and 948 for ASQA.
The result shown in Table 3 shows that placing the
query ahead can indeed help the performance.
ASQA NQ TriviaQA 2wiki Hotpot
Dataset2.502.753.003.253.503.754.004.25MarginMargin of Datasets
0 5 10 15 20 25 30
Layer2.53.03.54.04.5
Margin of Different Layers
ASQA
NQ
TriviaQA
2wiki
Hotpot(a) The margin of attention scores for Mistral 7B
ASQA NQ TriviaQA 2wiki Hotpot
Dataset02468101214MarginMargin of Datasets
0 5 10 15 20 25
Layer0.02.55.07.510.012.515.017.5
Margin of Different Layers
ASQA
NQ
TriviaQA
2wiki
Hotpot
(b) The margin of attention scores for Qwen 7B
Figure 6: The margin of attention scores. We calculate
the margin as the difference between the 90th and 10th
percentile attention scores to reduce the influence of
outliers
11

NQ TriviaQA HotpotQA 2wiki ASQA
reverse vanilla reverse vanilla reverse vanilla reverse vanilla reverse vanilla
GPT-4o61.2456.3172.3570.6182.4180.3485.7383.1549.1645.13
DeepSeek59.6956.9170.5767.1779.4876.3280.8072.7147.9945.62
Llama 80B63.1354.0771.0753.9381.5376.8387.1375.8048.5848.33
Llama 8B52.4346.7052.0745.6361.4354.3353.7052.0742.0441.81
Mistral 7B52.2751.6363.5063.5071.5770.1058.5755.3046.0443.67
Qwen2.5 7B 53.4753.63 59.8756.5774.8070.4767.0759.70 43.4744.74
Table 3: The performance of 3000 samples except ASQA (948 samples), this shows that putting the query ahead
could help the performance.
NQ TriviaQA HotpotQA 2wiki ASQA mean
Llamavanilla 21.8 33.7 14.7 28.8 24.5 24.7
LoRA 37.2 52.3 32.5 51.2 30.4 40.72
ours38.4 56.2 34.7 52.4 32.7 42.88
Qwenvanilla 20.3 29.2 13.8 28.3 24.2 23.16
LoRA 26.2 45.5 27.2 49.8 34.2 36.58
ours28.1 49.4 30.1 52.4 35.7 39.14
Mistralvanilla 26.4 37.8 24.9 43.5 24.8 31.48
LoRA 40.7 56.137.752.8 33.2 44.1
ours42.5 58.337.255.3 35.2 45.7
Table 4: The performance when all documents are retrieved. We retrieve top 5 documents and use those retrieved
documents as context. Our method also shows better performance
NQ TriviaQA HotpotQA 2wiki ASQA mean
LlamaSFT 67.8 74.6 88.7 96.4 45.7 74.6
ours 69.2 76.4 90.2 96.7 49.3 76.4
QwenSFT 65.2 72.3 85.1 96.1 49.2 73.6
ours 67.5 75.1 87.3 96.1 52.1 75.6
MistralSFT 65.2 73.6 86.1 96.4 50.3 74.3
ours 67.6 75.2 88.3 97.8 53.6 76.5
Table 5: The performance of full fine-tuning, and ours means adding the rectification directly to the existing attention
matrix, which means we only need to calculate one attention score.
12

B Proofs
B.1 The Triple-Wise problem
Suppose Alice and Bob are given inputs a, b∈ {0,1}n, respectively, with the goal of jointly computing
DISJ(a, b) = max iaibiby alternately sending a single bit message to the other party over a sequence of
communication rounds. Any deterministic protocol for computing DISJ(a, b) requires at least nrounds
of communication.
ri=(
0if∃a, b s.t. g(x i, xa, xb) = 0
1else
In normal cases, judging the value of rirequires calculating g(xi, xa, xb)for all a∈[0, n d)and
b∈[n d, nd+nq). Here we simplify the question, and we consider the situation where g(xi, xa, xb) = 0
only if b=a+n dandnq=nd. Apparently, this is a special case of the original problem, and if one layer
of self-attention fail to solve this, it is impossible for it to solve the original problem.
If we assume that the input is like,
xi∈

{xi}ifi= 0,
{0,x a}ifi∈ {1, . . . , n d−1},
{0,x b}ifi∈ {n d, . . . ,2·n d−1}.(2)
Given input (a, b)∈ {0,1}nd× {0,1}nd, letxi=xaif and only if ai= 1and let xi=xbif and only if
bi−nd= 1. In this wayr i= 0if and only ifDISJ(a, b) = 1.
For simplicity, we use n=n d. If we consider the setting of RAG, then actually Alice and Bob each
hold a matrix A∈Rn×d,B∈Rn×d, each row of the matrix contains w, which is the embedding to
judge if the token a relevant information. Sod=H(w). and we assume thatx= [s,w].
Then, to judge is a token relevant, we need to embedding xito contain information about w, if we want
to judge the relevance ofx 0, then
xi=

xi ifi= 0,
[s,a i]ifi∈ {1, . . . , n d−1},
[s,b i]ifi∈ {n d, . . . ,2·n d−1}.(3)
LetDISJ1(A,B) = max i(g′(xi,ai,bi))to be noise filtering task in RAG, then it requires to access
wof all tokens, which means the calculation ofDISJ1requiresn×H(w)bits of communication.
Also similar to the setting ofx,r i= 1if and only ifDISJ1(A,B) = 1.
Then, this is the same with the 3Match problem, following the proof of Theorem 7 in Sanford
et al. (2024), and with the following form of transformer f(X) ,2pHlog logn+mpHlog logn≈
mpHlog lognbits are communicated.
f(X) =ϕ(f h(X)),
whereϕstands for the feed forward layers.
fh(X) =PN
i=1exp 
(Wqx1)TWkxi
WvxiPN
i=1exp ((W qx1)TWkxi)
Therefore, only we requirempHlog logn≥nH(w)→mph≥nH(w)/log logn.
Theorem B.1.For input documents of length n, ifmpH≤Ω(nH(w)/log logn) , then there is no one
layer transformerMwith embedding sizem, precisionpandHheads satisfyingM(X) =r.
Also, as shown in Sanford et al. (2024), multiple layers of multi-headed attention are subject to the
same impossibility
13

Conjecture B.2.Every multi-layer transformer that computes Match3 must have width, depth, embedding
dimension, or bit complexity at leastNΩ(1).
This is based on the situation that, each translation only need 1 bit, for noise filtering, we require to
translate H(w) bit of information each time, so it requires width, depth, embedding dimension, or bit
complexity at least NΩ(1)·H(w) . This directly means that triple-wise problems can not be solved with
limited number of transformer layers
B.2 Proof of Theorem 4.1
Letattn(x i) = (W qxi)TWkX:ibe the original attention layer of LLM, and attn′(xi) =
((W q+ ∆W q)xi)T(Wk+ ∆W k)X:ibe the fine-tuned one and dattn be desired function, can we
fine-tune the model to be dattn?
So we need
softmax(attn′(X))[i]≈(
0if∃b s.t. g 1(xi, xb) = 0
softmax(attn(X r))[i]else
whereX rmeans those related tokens. letA i,j=attn(x i,xj) = (W qxi)TWkxj=xT
iWx j
attn′(xi,xj) = ((W q+ ∆W q)xi)T(Wk+ ∆W k)xj
=x i(W+ ∆W)x j
=x iWx j+xi∆Wx j
where∆W= ∆W qWk+W q∆W k+ ∆W q∆W k
Then, to effectively separate noise information, we require the attention score of the noise to be small
and the attention score of useful information to be large, so to filter out noise, we require ∆W′
q,∆W′
k
satisfying
xi∆Wx j(
≤cl ifxjis noise,
∈[ch, ch+ξr]else,(4)
wherec landc hare constants andc h> cl
ifxjis a noise token andx i∆Wx j=cl, then
softmax(attn′(xi))[j]−0 =softmax(A i,:+xi∆WX)[j]
=exp(A i,j+xi∆Wx j)P
kexp(A i,k+xi∆Wx k)
=exp(A i,j+cl)P
kexp(A i,k+xi∆Wx k)
=exp(A i,j+cl−ch)P
kexp(A i,k+xi∆Wx k−ch)
if we needsoftmax(attn′(xi))[j]−0≤ϵ, letA i,j= max(A i,:), then
exp(A i,j+cl−ch)P
kexp(A i,k+xi∆Wx k−ch)≤softmax(attn′(xi))[j]−0≤ϵ
exp(A i,j+cl−ch)≤ϵX
kexp(A i,k+xi∆Wx k−ch)
cl−ch≤ln 
ϵX
kexp(A i,k+xi∆Wx k−ch)!
−A i,j
cl−ch≤ln (ϵnexp(A i,j+ξr))−A i,j
cl−ch≤ξr+ lnϵn
14

else if xjis a relevant token, let c=c h, andP
k′exp(A i,k′)denotes the summation of all relevant
tokens, we consider a simple case where xi∆Wx j=ch, and for one token xk1, we have xi∆Wx k1=
ch+ξr, and for all other relevant tokens we havex i∆Wx k=ch
softmax( ˆattn(x i))[j]−softmax(attn′(xi))[j]
=exp(A i,j)P
k′exp(A i,k′)−exp(A i,j+xi∆Wx j)P
kexp(A i,k+xi∆Wx k)
=exp(A i,j)P
k′exp(A i,k′)−exp(A i,j+xi∆Wx j−c)P
kexp(A i,k+xi∆Wx k−c)
=exp(A i,j)P
k′exp(A i,k′)−exp(A i,j)P
kexp(A i,k+xi∆Wx k−c)(5)
considering softmax( ˆattn(x i))[j]−softmax(attn′(xi))[j] =c(1
a−1
b)≤ϵc
a,c= exp(A i,j),a=P
k′exp(A i,k′),b=P
kexp(A i,k+xi∆Wx k−c)
b−a≤ϵb
(1−ϵ)b≤a
b≤a
1−ϵ
X
kexp(A i,k+xi∆Wx k−c)≤P
k′exp(A i,k′)
1−ϵ
X
k′exp(A i,k′+xi∆Wx k−c)≤P
k′exp(A i,k′)
1−ϵ
X
k′−k1exp(A i,k′) + exp(A i,k1+ξr)≤P
k′exp(A i,k′)
1−ϵ
exp(A i,k1+ξr)≤ϵ
1−ϵX
k′−kexp(A i,k′) +exp(A i,k1)
1−ϵ
ξr≤ln 
ϵ
1−ϵX
k′−kexp(A i,k′) +exp(A i,k1)
1−ϵ!
−A i,k1
Considering the case thatϵ
1−ϵP
k′−kexp(A i,k′)≈0, then we needξ r≲ln1
1−ϵ
Asϵ≈0, soln1
1−ϵ≈0
B.3 MLP also fails to filter out noise
In the following, we make use of the quantities
Nmax(t) = max
ˆv∈ˆVNˆv(t), N min(t) = min
ˆv∈ˆVNˆv(t),
where
Nˆv(t) =X
v∈V⊮{d(v,ˆv)≤t}
counts the number ofv∈ Vwithin a “distance”tofˆv∈ ˆV.
Theorem B.3.(Fano’s inequality with approximate recovery in Scarlett and Cevher (2019))For any
random variablesv,ˆvon the finite alphabetsV, ˆV, we have
Pe(t)≥H(v| ˆv)−1
log|V|
Nmax(t).(6)
Pe(t) = Pr(||z− ˆz||> t), wherezis inferenced by some function and the input isv.
15

Assume the input to the feed forward layer v, the first llayers are used to identify the relevance and
the last few layers are used for inference. The output of the first llayers are vl, and the embedding is
used to conduct inference and get the result z. So we can say that with probability p≥H(vl|ˆv)−1
log|S|
Nmax (t)the
resulting embedding fails to be tclose to the original onei.e., d(z,ˆz)≤t , where ˆvstands for the optimal
input where all irrelevant information is filtered and all the related information is contained and ˆzis the
corresponding output..
Assume that ˆware equally distributed to each token which satisfies I(w i;v) =I(w j;v), therefore,
for each document, the model holds the same probability of mistakenly identify its relevance. Let pwe
stands for the error probability of identify noise tokens, and δbe the percentage of relevant tokens.
With probability δpwe, the relevant token is mistakenly regarded as irrelevant, and with probability
(1−δ)p wethe irrelevant token is mistakenly regarded as relevant. So there areδ(1−p we)
δ(1−p we)+(1−δ)p wepercent
of information about the relevant ones. Also pwepercent of relevant information and 1−p wepercent of
irrelevant information are discarded, then I(s;v l) = ((1−p we)·δ+p we·(1−δ))I(s;v) are the left
information about inference, among these,δ(1−p we)
δ(1−p we)+(1−δ)p weare acutally related information, others are
noisy information.
In this way,
I(vl;ˆv) =I(v l;s)
=δ(1−p we)
δ(1−p we) + (1−δ)p we·((1−p we)·δ+p we·(1−δ))I(s;v)
=δ(1−p we)·I(s;v)
≤δ(1−H(w|v−1)
H(w))·I(s;v)
=δ(I(w;v) + 1
H(w))·I(s;v)
Therefore,
Pe(t)≥H(v| ˆv)−1
log|V|
Nmax(t)=H(v)−I(v; ˆv)
log|V|
Nmax(t)
≥H(v)−g 1(δ, I(w;v))·I(s;v)
log|V|
Nmax(t)
whereg 1(δ, I(w;v)) =δ(I(w;v)+1
H(w))
So when there is no noise, the inference can be conducted based on those information, then we have
Pr (||z−ˆz||> t)≥H(v)−g 1(δ, I(w;v))·I(s;v)
log|V|
Nmax(t)
Pr (||z−ˆz|| ≤t)≤1−H(v)−g 1(δ, I(w;v))·I(s;v)
log|V|
Nmax(t)
Considering the noise in the embedding of vl, and the noise would have negative impact on the inference.
Also the extra noisy information contained inv lis
I(v−;vl) =(1−δ)p we
δ(1−p we) + (1−δ)p we·((1−p we)·δ+p we·(1−δ))I(s;v)
= (1−δ)p we·I(s;v)
≥(1−δ)H(w|v)−1
H(w)·I(s;v)
Consider the best case whereI(v−;vl) = (1−δ)H(w|v)−1
H(w)·I(s;v)
16

Theorem B.4(Theorem 2 of Kawaguchi et al. (2023)).Let D ⊆ {1,2, . . . , D+ 1} . Then, for any δ >0 ,
with probability at least1−δover the training sets, the following generalization bound holds:
∆(s)≤min
l∈DQl,(7)
where forl≤D,
Ql=Gl
3q
(I(X;Zs
l|Y)+I(ϕS
l;S))ln(2)+ bGl
2
n+Gl
1(ζ)√n;
and forl=D+ 1,
Ql=R(fs)s
I(ϕS
l;S) ln(2) + ˇGl
2
2n,
Here, S∼ P⊗n,Gl
1(ζ) = ˆO(q
I(ϕS
l;S) + 1) ,bGl
2=ˆO(1) ,ˇGl
2=ˆO(1) , and Gl
3=ˆO(1) asn→ ∞ .
The formulas ofGl
1(ζ),bGl
2,ˇGl
2, andGl
3are given in Appendix.
using||f(x)− ˆf(x)||as the loss function, then we have that
L −ˆL ≤Gl
3s 
I(X;Zs
l|Y) +I(ϕS
l;S)
ln(2) + bGl
2
n+Gl
1(ζ)√n
=c1q
I(v−|vl) +I(ϕS
l;S) +c 3
≤c1p
I(v−|vl) +c 1q
I(ϕS
l;S) +c 3
≤c1p
I(v−|vl) +c 2(8)
wherec 2=c3+c1q
I(ϕS
l;S)Therefore,
Pr
||f(x)− ˆf(x)|| ≤t+c 1p
I(v−|vl)+ +c 2
≤1−H(v)−g 1(δ, I(w;v))·I(s;v)
log|V|
Nmax(t)(9)
withI(v−;vl) = (1−δ)H(w|v)−1
H(w)·I(s;v) =g 2(δ, I(w;v))·I(s;v).
Pr
||f(x)− ˆf(x)||> t+c 1p
g2(δ, I(w;v))·I(s;v)+ +c 2
>H(v)−g 1(δ, I(w;v))·I(s;v)
log|V|
Nmax(t)(10)
Theorem B.5.For a Feed Forward Network fand the input xcontains 1−δ percent of noisy information,
assume the optimal function is ˆf(x)which filter out the noise and finish the inference, then
Pr
||f(x)− ˆf(x)||> t′
>H(v)−g 1(δ, I(w;v))·I(s;v)
log|V|
Nmax(t), (11)
where t′=t+c 1p
g2(δ, I(w;v))·I(s;v) +c 2.g1(δ, I(w;v)) =δ(I(w;v)+1
H(w))g2(δ, I(w;v)) = (1−
δ)H(w|v)−1
H(w)
17