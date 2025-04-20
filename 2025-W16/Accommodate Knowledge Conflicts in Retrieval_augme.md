# Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild

**Authors**: Jiatai Wang, Zhiwei Xu, Di Jin, Xuewen Yang, Tao Li

**Published**: 2025-04-17 14:40:31

**PDF URL**: [http://arxiv.org/pdf/2504.12982v1](http://arxiv.org/pdf/2504.12982v1)

## Abstract
The proliferation of large language models (LLMs) has significantly advanced
information retrieval systems, particularly in response generation (RG).
Unfortunately, LLMs often face knowledge conflicts between internal memory and
retrievaled external information, arising from misinformation, biases, or
outdated knowledge. These conflicts undermine response reliability and
introduce uncertainty in decision-making. In this work, we analyze how LLMs
navigate knowledge conflicts from an information-theoretic perspective and
reveal that when conflicting and supplementary information exhibit significant
differences, LLMs confidently resolve their preferences. However, when the
distinction is ambiguous, LLMs experience heightened uncertainty. Based on this
insight, we propose Swin-VIB, a novel framework that integrates a pipeline of
variational information bottleneck models into adaptive augmentation of
retrieved information and guiding LLM preference in response generation.
Extensive experiments on single-choice, open-ended question-answering (QA), and
retrieval augmented generation (RAG) validate our theoretical findings and
demonstrate the efficacy of Swin-VIB. Notably, our method improves
single-choice task accuracy by at least 7.54\% over competitive baselines.

## Full Text


<!-- PDF content starts -->

Accommodate Knowledge Conflicts in Retrieval-augmented
LLMs: Towards Reliable Response Generation in the Wild
Jiatai Wang
College of Computer Science
Nankai University
Tianjin, China
1120240357@mail.nankai.edu.cnZhiwei Xu∗
Haihe Lab of ITAI
Tianjin, China
xuzhiwei2001@ict.ac.cnDi Jin
Meta AI
USA
jindi@meta.com
Xuewen Yang
InnoPeak Technology, Inc
Palo Alto, USA
xuewen.yang@protonmail.comTao Li∗
College of Computer Science
Nankai University
Tianjin, China
litao@nankai.edu.cn
Abstract
The proliferation of large language models (LLMs) has significantly
advanced information retrieval systems, particularly in response
generation (RG). Unfortunately, LLMs often face knowledge con-
flicts between internal memory and retrievaled external informa-
tion, arising from misinformation, biases, or outdated knowledge.
These conflicts undermine response reliability and introduce un-
certainty in decision-making. In this work, we analyze how LLMs
navigate knowledge conflicts from an information-theoretic per-
spective and reveal that when conflicting and supplementary in-
formation exhibit significant differences, LLMs confidently resolve
their preferences. However, when the distinction is ambiguous,
LLMs experience heightened uncertainty. Based on this insight, we
propose Swin-VIB, a novel framework that integrates a pipeline
of variational information bottleneck models into adaptive aug-
mentation of retrieved information and guiding LLM preference
in response generation. Extensive experiments on single-choice,
open-ended question-answering (QA), and retrieval augmented gen-
eration (RAG) validate our theoretical findings and demonstrate the
efficacy of Swin-VIB. Notably, our method improves single-choice
task accuracy by at least 7.54% over competitive baselines.
CCS Concepts
•Information systems →Question answering .
Keywords
Reliable response generation; retrieval-augmented LLMs; knowl-
edge conflicts; preference adaption; variational information bottle-
neck
1 Introduction
The rapid advancement of large language models (LLMs) has revo-
lutionized information retrieval systems, particularly in retrieval-
augmented generation (RAG). Among these novel technologies,
the response generation (RG) technique [ 21] directs information
accessing with LLMs while bypassing rigid retrieval granularity
and relevance matching. Despite their remarkable effectiveness,
LLMs remain susceptible to hallucinations, making it challenging
∗Both authors are corresponding authors.
Figure 1: Illustration of knowledge conflict in RG
to ensure reliability in domain-specific knowledge applications
and knowledge updates. Augmenting LLMs with external infor-
mation has become a mainstream approach to mitigating these
issues [ 3,19,33,42,43]. However, the disparity between the inter-
nal memory of LLM and the retrieved information from the external
sources leads to knowledge conflicts [ 40]. These conflicts are al-
ways caused by misinformation, unreliable sources, and publisher
bias of the retrieved information. Additionally, pre-trained LLMs
themselves may encode inherent biases and outdated knowledge
due to the limitations of their training corpora. The inability to
synchronize LLMs with real-time updates further exacerbates these
discrepancies, leading to contradictions between internal and exter-
nal knowledge. As illustrated in Figure 1, these conflicts introduce
uncertainty in RG [ 37], posing a serious threat to the reliability of
RG tasks and increasing the risk of biased or erroneous inference.
The existing approaches do not fully agree on how LLMs re-
solve knowledge conflicts. Some approaches [ 9,12,38] reveal that
LLMs are inclined to prioritize external context, while others [ 14,
16,23,37] argue that LLMs may rely on their internal memory.
For example, a) LLMs tend to cite external context when internal
priors are weak but resist when strong [ 37]; b) LLMs are highly
receptive to conflicting external evidence if it is coherent and con-
vincing [ 38]; c) Despite external evidence, the LLM sticks to its
flawed internal memory [ 14]. Even worse, the LLM operates in a
black box so the fundamental mechanism of the conflicts of invis-
ible internal knowledge remains difficult to be fully understood,
which is particularly challenging in practice. Therefore, the existingarXiv:2504.12982v1  [cs.CL]  17 Apr 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
empirical approaches [ 15,40] typically focus on heuristic guide-
lines to improve the reliability of RG but lack a unifying theoretical
framework.
To bridge this gap, we re-examine knowledge conflict in RG from
the perspective of information theory. We analyze how LLMs es-
tablish preferences when faced with conflicting knowledge sources
and observe:
•When the disparity between conflicting and supplementary
information is significant, LLMs confidently settle into a
stable preference.
•When the distinction is ambiguous, LLMs experience ex-
treme uncertainty, making their response unreliable.
Building on these insights, we propose Swin-VIB, a Variational In-
formation Bottleneck (VIB) approach with a Sliding window. More
specifically, we leverage a pipeline-based multiple variational infor-
mation bottleneck models [ 2] to adaptively augment the retrieved
information and guide the preference of LLMs for response gener-
ation. This design stabilizes the LLM’s preference and minimizes
ambiguity, enabling more accurate and consistent responses even
in challenging scenarios with knowledge conflicts. Our major con-
tributions are summarized as follows:
•To mitigate the uncertainty of response generation caused
by knowledge conflicts, we model the interplay between
the internal memory of LLMs and external context and re-
lease the preference principle behind knowledge conflict
in retrieval-augmented LLMs. With significant differences
between conflicting information and supplementary infor-
mation, LLMs have more confidence to settle into a stable
preference, otherwise, LLMs fall into extremely high uncer-
tainty.
•This analysis of LLMs’ preference provides insight into ac-
commodating knowledge conflicts in retrieval-augmented
LLMs. In this way, we propose Swin-VIB, a sliding-window
approach that integrates multiple variational information
bottlenecks to adapt perplexing knowledge, enhancing that
knowledge that can guide the LLMs toward stable preference
and thus accurate response generation.
•Extensive experiments across single-choice, open-ended ques-
tion answering (QA) and RAG, validate our theoretical model
and demonstrate that Swin-VIB outperforms baselines for
reliable response generation in the wild.
These contributions address vital challenges in reliable response
generation, paving the way for more trustworthy and effective gen-
erative information retrieval systems in practice. To the best of our
knowledge, this is the first approach that systematically analyzes
and evaluates knowledge conflicts of LLMs, rather than merely rely-
ing on empirical evaluations, thus enjoying higher interpretability.
Access the implementation code at GitHub1after acceptance.
2 Problem Analysis
Knowledge conflicts may cause uncertainty in response generation
where LLMs need to make a choice with conflicted knowledge from
the internal memory of LLMs and the external context. Considering
the black-box nature of LLMs, it remains challenging to analyze
1https://anonymous.4open.science/r/Swin-VIB-54F9the fundamental mechanism of these conflicts between invisible
internal knowledge and external contexts. Without a theoretical
analysis of knowledge conflicts, the limitations of empirical rules
and experimental settings in response generation can hardly be
alleviated[ 14,37,38,40]. To tackle this challenge, we propose a
theoretical framework that shows knowledge conflicts can be de-
fined by conditional entropy, achieving an understanding of the
knowledge conflicts of retrieval-augmented LLMs. First of all, the
corresponding symbol system is defined in Table 1.
Table 1: Symbols and their meanings
Symbol Meaning
𝑄Query prompts, the initial input that drives LLM
inferences in the RG
𝑅=𝐾(𝑄)External contexts retrieved from the knowledge
base𝐾according to query 𝑄
𝑂=𝐿𝐿𝑀(𝑅,𝑄)Generated Responses by LLM according to 𝑅and𝑄
Definition 2.1 (Uncertainty of Retrieval-augmented Response Gen-
eration ).The uncertaintyUof𝑂given the𝑄and𝑅is represented
by conditional entropy 𝐻(𝑂|𝑅,𝑄), is given by
U=𝐻(𝑂|𝑅,𝑄)=−∑︁
𝑜,𝑟,𝑞𝑝(𝑜,𝑟,𝑞)log𝑝(𝑜|𝑟,𝑞),(1)
where𝑜,𝑟, and𝑞are specific instances of 𝑂,𝑅, and𝑄, respectively.
Joint probability distribution 𝑝(𝑜,𝑟,𝑞)represents the correlation
between variables, and 𝑝(𝑜|𝑟,𝑞)is the conditional probability of 𝑜
given𝑟and𝑞.
Assumption 2.1 (Non-void Relevant Retrieval ).A qualified external
information retriever can recall external contexts correlated with
the corresponding query [ 26], that is∀𝑞∈𝑄,∃𝑟∈𝑅,𝑝(𝑟,𝑞)>0,
where𝑞is a query prompt, and 𝑟is one correlated external context.
According to Definition 2.1, we rewrite Formula 1 according to
law of total probability and chain rule,
U=−∑︁
𝑟,𝑞𝑝(𝑟,𝑞)∑︁
𝑜𝑝(𝑜|𝑟,𝑞)log𝑝(𝑜|𝑟,𝑞).(2)
Let𝜓(·)denote instance-level uncertainty, and be calculated by
conditional entropy,
𝜓(𝑝(𝑜|𝑟,𝑞))=−𝑝(𝑜|𝑟,𝑞)log𝑝(𝑜|𝑟,𝑞) (3)
Since𝑟is retrieved according to 𝑞,∃C>0,Í
𝑟,𝑞𝑝(𝑟,𝑞)=C,
whereCis a constant. Thus, Uis equal to,
U=−C∑︁
𝑜𝑝(𝑜|𝑟,𝑞)log𝑝(𝑜|𝑟,𝑞)=C∑︁
𝑜𝜓(𝑝(𝑜|𝑟,𝑞))(4)

Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Here,𝑝(𝑜|𝑟,𝑞)can be derived as the follows:
𝑝(𝑜|𝑟,𝑞)
=∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑝(𝑥|𝑟,𝑞)𝑑𝑥, //marginalization
=∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑝(𝑟,𝑞|𝑥)𝑝(𝑥)
𝑝(𝑟,𝑞)𝑑𝑥, //Bayes’ Theorem
∝∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑝(𝑟,𝑞|𝑥)𝑝(𝑥)𝑑𝑥, //𝑝(𝑟,𝑞)is a constant
∝∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑝(𝑟,𝑞|𝑥)
𝑝 𝑟,𝑞|𝑥𝛾𝑝(𝑥)𝑑𝑥, //𝑝 𝑟,𝑞|𝑥𝛾is a constant
=∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑒𝑥𝑝 
log𝑝(𝑟,𝑞|𝑥)
𝑝 𝑟,𝑞|𝑥𝛾!
𝑝(𝑥)𝑑𝑥
=∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑒𝑥𝑝 
log𝑝(𝑞|𝑟,𝑥)
𝑝 𝑞|𝑟,𝑥𝛾+log𝑝(𝑟|𝑥)
𝑝 𝑟|𝑥𝛾!
𝑝(𝑥)𝑑𝑥
≈∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑒𝑥𝑝 
−logB+log𝑝(𝑟|𝑥)
𝑝 𝑟|𝑥𝛾!
𝑝(𝑥)𝑑𝑥
≈∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑒𝑥𝑝 log𝑝(𝑟|𝑥)−log𝑝 𝑟|𝑥𝛾𝑝(𝑥)𝑑𝑥
(5)
where𝑋denotes the high-dimensional latent space of the LLM
containing all embedded information that may affect response gen-
eration;𝑥is a recalled instance sampled from 𝑋;𝑥𝛾is an information
instance that coincides with information retrieved from an exter-
nal text or database, while considering query 𝑞. According to the
duality of retrieval-augmented response generation, the response
generation can be regarded as a hidden Markov model, and thus
log𝑝(𝑞|𝑟,𝑥)
𝑝(𝑞|𝑟,𝑥𝛾)has an upper bound −logB[41].
The self-information of 𝑝(·), is equal to−log𝑝(·), quantifies how
much information is revealed [ 27]. We categorize the external infor-
mation retrieved from an external text or database into two types,
whereas the LLMs’ pre-trained knowledge is either contradicted or
insufficient.
•Conflicting Information : If the retrieved information instance
contradicts𝑥𝛾in the LLMs’ pre-trained knowledge. This is a
conflicting information instance 𝐼𝑐, quantified by
−log𝑝 𝑟|𝑥𝛾.
•Supplementary Information : When encountering the retrieved
information instance with new or unseen knowledge, it is
an instance of supplementary information, 𝐼𝑠, quantified
−log𝑝(𝑟|𝑥).
According to Formula 5,
𝑝(𝑜|𝑟,𝑞)∝∫
𝑋𝑝(𝑜|𝑟,𝑞,𝑥)𝑒𝑥𝑝[𝐼𝑐−𝐼𝑠]𝑝(𝑥)𝑑𝑥
∝𝐼𝑐−𝐼𝑠(6)
We can achieve a conclusion that p(o|r,q)is determined by Ic−Is
because𝑒𝑥𝑝(·)is monotonically increasing and decisive. In addition,
𝜓andUare also correlated with Ic−Is.
Ultimately, the entropy function-based relationship between 𝜓
and the difference between 𝐼𝑐and𝐼𝑠is illustrated in Figure 2. Let
the preference boundary 𝜂=𝐼𝑐−𝐼𝑠, at which𝜓attains its maximum
value. According to the relationship between 𝜓and the difference
Figure 2: the relationship between 𝜓and the difference be-
tween𝐼𝑐and𝐼𝑠
between𝐼𝑐and𝐼𝑠, we separate the curve in Figure 2 into three
regions:
•R reliance (Reliance Region) : The LLM exhibits a preference to
lean towards external context, where (𝐼𝑐,𝐼𝑠)∈R2,𝐼𝑐−𝐼𝑠<𝜂.
•R confident (Confident Region) : The LLM exhibits a preference
to lean towards internal knowledge, where (𝐼𝑐,𝐼𝑠)∈R2,𝐼𝑐−
𝐼𝑠>𝜂.
•R perplexity (Perplexity Region) : The LLM gets confused due to
high uncertainty about the recalled knowledge and has no
obvious preference, where the difference between 𝐼𝑐and𝐼𝑠
approaches a 𝑣-neighborhood around 𝜂.𝑣is a constant that
is defined as a perplexity boundary.
Without considering that the strict prompt templates can also
influence preference, our prompts are benign for LLMs to generate
responses according to retrieved external context [ 37]. We con-
clude the principle of LLMs’ preference on the used information by
Remark 2.1.
Remark 2.1. If the difference between 𝐼𝑐and𝐼𝑠is quite large
or small, LLMs maintain stable preference, and achieve response
generation with low uncertainty, i.e.,
Ílim(𝐼𝑐−𝐼𝑠)→−∞𝜓=0, 𝐼𝑐−𝐼𝑠∈R relianceÍlim(𝐼𝑐−𝐼𝑠)→+∞𝜓=0, 𝐼𝑐−𝐼𝑠∈R conflict(7)
Considering the black-box nature of LLMs, the difference be-
tween conflicting information and supplementary information can-
not be quantified explicitly, but we can adapt this type of difference
by observing the trend of the output uncertainty of the LLMs. When
the absolute difference between conflicting information and sup-
plementary information becomes larger, the LLM will intensify its
preference to reduce uncertainty, and thus the output uncertainty
continues a decreasing trend. In this way, we give two optimization
strategies to achieve reliable response generation:
•Acceptation Strategy : Accept the external context if the out-
put uncertainty of LLMs consistently becomes smaller;
•Rejection Strategy : Reject the external context if output un-
certainty does not have an explicit trend;

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Figure 3: External information adaption with a bottleneck-
based structure
3 Methodology
3.1 Design Framework
According to the strategies proposed in Section 2, a novel model,
Swin-VIB, is proposed to adaptively compress various information
and extract their differences, which is inspired by the information
bottleneck structure [ 30]. The information bottleneck is an optimal
compression method, consisting of an encoder that reduces the
amount of information and a decoder that makes key inference. In
Swin-VIB, this structure is trained to learn the high-dimensional
association between the difference of conflicting information and
supplemental information and the output uncertainty of LLMs.
Specifically, we randomly set a window to select tokens in an exter-
nal context and extract the corresponding attention scores as the
input representation G(𝑅)for bottleneck training:
G(𝑅)=𝑁∑︁
𝑛=11
𝑙𝑒𝑛|𝜔|∑︁
𝑖=1S A𝑏(𝜔),where𝜔=win 𝑅,𝑙𝑒𝑛.(8)
win 𝑅,𝑙𝑒𝑛denotes randomly selects a token subsequence of length
𝑙𝑒𝑛from the external context 𝑅.A𝑛(𝜔)∈R|𝜔|×|𝜔|is the attention
matrix produced by the 𝑛-th Transformer decoder block [ 32] for
the window 𝜔,𝑁is the number of blocks. Saggregates the total
attention scores assigned among all token pairs within the window.
The detailed design is present as the following: 1) The encoder
compresses the representation Ginto a latent representation Z,
adaptively discarding irrelevant information while maintaining
difference features correlated to information from internal and ex-
ternal sources; 2) The decoder establishes the relationship between
these difference features and the labeled uncertainty index of the
model output.
Training objectives and process : Our goal is to compress the
amount of information in Gwhile maximizing the reliability of
inference. Therefore, we label Gby symbol Y:
Y= 
1,if𝑅is unique or the ground true.
0,if𝑅includes multiple optional
instances except for the ground true.(9)
This setting makes Yrepresent the uncertainty of the retrieval-
augmented response generation. We include GandYto construct
training datasets for each transformer decoder block.Here, it is by minimizing the mutual information 𝐼(G,Z)to guide
the encoder to adaptively learn the key features to identify infor-
mation differences, whereas maximizing the mutual information
𝐼(Z,Y)facilitates the decoder to predict the output uncertainty
based on the latent representation Y:
max𝐼(Z,Y;𝜙)s.t.min𝐼(G,Z;𝜃), (10)
where𝜃and𝜙are the parameter of the encoder 𝑞𝜃(Z|G)and
the decoder 𝑝𝜙(Y|Z), respectively. The ability to learn informa-
tion differences is featured in the bottleneck structure during the
training process (see Figure 3). However, G,Y, and Zcan only be
represented by discrete values and Gaussian distributions, resulting
in a severe limitation in model training. Inspired by variational
inference [ 2], the best approximate posterior distribution can be
found by minimizing the KL divergence between distributions. To
further guarantee the loss function trainable, the expectation 𝜇𝑞
and variance 𝜎𝑞ofZare predicted by the encoder:
Z=𝑟(G,𝜖)=𝜇𝑞(G)+𝜎𝑞(G)𝜖, (11)
where𝑟(¤)is a reparameterization function [ 17]. Noise𝜖is intro-
duced for constructing the intermediate stochastic vector. So the
optimization objective is defined as:
L=𝐼(Z,Y)−𝛽𝐼(Z,G)=𝐻(Y,𝑝𝜙(Z))+𝛽KL(𝑞𝜃(Z|G)∥𝑝(Z))
(12)
where𝑝(g,y)=𝑝(g)𝑝(y|g)can be approximated by the empirical
distribution. 𝛽serves as the bottleneck parameter controlling the
capacity for information compression.
Inference process : The encoder reverses the ability learned
during training to compress the amount of information and eval-
uate the difference of various information, allowing the decoder
to more accurately predict uncertainty. In this way, the strategies
listed in Section 2 are implemented, thereby stabilizing the LLM
preference and accommodating conflicts in retrieval-augmented
response generation with LLMs.
3.2 Swin-VIB
Corresponding to 𝑁groups of attention scores extracted from 𝑁
transformer decoder blocks in an LLM [ 32], we cascade 𝑁bottle-
neck structures to form Swin-VIB, incorporating it into the LLM
and facilitate the LLM to generate response. Thus, according to
Formula 12, their loss functions are defined as follows:
L𝑛=
𝐻(Y,𝑝𝜙(Z𝑛))+𝛽KL(𝑞(Z𝑛|G𝑛)∥𝑝(Z))	𝑁
𝑛=1(13)
The output of all bottlenecks is weighted to achieve an averaging
result for identifying output uncertainty (detailed in Sec. 4.4.1):
ˆY=1
𝑁𝑁∑︁
𝑛𝑝𝜙(𝑞𝜃(G)) (14)
where, if ˆYis greater than the threshold 𝜉, it is approximated with
1, otherwise, setting to 0, which is a specific implementation of Y in
Formula 9. This design integrates the information of all Transformer
representations while avoiding any change on the LLM.
In its inference stage, responses are generated by three steps (as
illustrated in Figure 4):
Step 1 (Retrieval) : Relevant contexts are retrieved from external
knowledge according to a query;

Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Figure 4: An overview of response generation with Swin-VIB.
Step 2 (Augmentation) : We first use LLM to extract the representa-
tion of the external context, G, and then use 𝑁trained bottlenecks
to perform sliding-window-based inference. The sliding window
dynamically accepts or rejects the context according to the output
of Swin-VIB. Based on empirical experience, we set the window
length to 7. In this way, the tokens in the context are incorporated
into the prompt template. The forward propagation of the varia-
tional information bottleneck requires the introduction of random
noise, we involve two types of randomization methods:
•MonteCarlo method , which obtains a better generalization
effect by randomly sampling;
•Mean method , which discards the variance of the encoder
prediction to obtain fast inference speed.
Step 3 (Response generation) : Reliable responses are generated by
retrieval-augmented LLMs, even in the presence of conflicting in-
formation.
4 Experiments
4.1 Evaluation Setup
4.1.1 Datasets. To evaluate how LLMs handle conflicts and im-
prove the reliability of response generation, two popular public
Question-Answer (QA) datasets, ConflictQA [ 38] and TruthfulQA
[22], are employed.
•ConflictQA is constructed based on the initial memory of
the most popular LLMs (Llama2-7b [ 31], Llama2-70b [ 31],
GPT4 [ 1], Qwen-7b [ 4], and four other LLMs). We take the
key "counter-memory" in ConflictQA to indicate external
contexts that conflict with the internal memory.
•TruthfulQA includes the keys "correct answer" and "incorrect
answer", and contains multiple correct or incorrect contexts
that conflict with each other.
To adapt TruthfulQA for the scenarios with conflicted information,
we use LLMs to select keys among "correct answer" and "incorrect
answer" based on their internal memory. One or two contexts are
randomly taken from the keys that were not chosen by the LLM as
the "counter-memory" key, to follow the usage of ConflictQA.Table 2: Dataset information
DatasetSizeTask
Single choice Open-ended QA RAG
ConflictQA (Llama2-7B) 2839 ✓ ✓
ConflictQA (Qwen-7B) 7204 ✓ ✓
TruthfulQA 817 ✓ ✓ ✓
4.1.2 Implementation. The experiments were run on Ubuntu 20.04
with NVIDIA RTX A6000 GPUs, using Pytorch 2.4.1. To validate
the performance of Swin-VIB, we compare Swin-VIB with baselines
on two popular LLMs, Llama2-7B [ 31] and Qwen-7B [ 4]. In detail,
Swin-VIB is configured in terms of various layers and parameters.
𝜉=0.8,𝛽=10−3for ConflictQA, and 𝜉=0.68,𝛽=10−4for
TruthfulQA. To establish scenarios with the conflicted information,
we implement three popular tasks with a series of metrics2to
evaluate the performance of Swin-VIB across different tasks:
1) Single-Choice: Single-choice task evaluates the generalization
ability of Swin-VIB and other baselines, by exploring the preferences
of LLM through its choices. Specifically, we provide two options for
each query. One of the options is obtained from the internal memory,
the other is from the external context, and only one of them is
correct. To construct prompt templates with conflict information,
we use the "counter-memory" in the dataset to fill the <conflicting
external context> slots in the prompt template, as shown in Figure
5(a). This prompt template instructs the LLM to select one of the
options to generate responses.
A comprehensive set of metrics is used to evaluate the perfor-
mance of Swin-VIB for single-choice, where 𝑇represents the total
number of instances, and 𝐿is the number of instances that the
system handles correctly without intervention. 𝐶𝑟,𝐶𝑤, and𝐶𝑛cor-
respond to the number of instances with correct, incorrect, and
abandoned answers, respectively. 𝐶𝑐𝑜𝑟𝑟𝑒𝑐𝑡 denotes the number of
instances that were initially answered incorrectly but corrected by
Swin-VIB,𝐶𝑑𝑒𝑓𝑒𝑛𝑠𝑒 represents the number of results that resisted
misleading influences and achieved correct answers, 𝐶𝑚𝑖𝑠𝑙𝑒𝑎𝑑 indi-
cates the number of initially correct answers that become incorrect
due to misleading or confusing information, and 𝐶𝑜𝑣𝑒𝑟𝑐𝑜𝑛𝑓𝑖𝑑𝑒𝑛𝑐𝑒
refers to the number of results maintaining confidence in incorrect
answers. The following is a detailed description of the metrics:
2In all tables,↑indicates higher values are better, ↓indicates lower values are better.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
(a) Prompt template for single-choice task
 (b) Prompt template for open-ended QA task
 (c) Prompt template for RAG task
Figure 5: Prompt templates for three scenarios
•Accuracy (ACC =𝐶𝑟
𝑇), Error Rate (ER =𝐶𝑤
𝑇) and Abandon-
ment Rate ( AR=𝐶𝑛
𝑇) measure the proportion of correct,
incorrect, and abandoned answers. We quantify the uncer-
tainty of the LLM by H𝑡𝑜𝑡𝑎𝑙 , which is equal to−(𝐴𝐶𝐶 log2(𝐴𝐶𝐶+
𝜖)+𝐸𝑅log2(𝐸𝑅+𝜖)+𝐴𝑅log2(𝐴𝑅+𝜖)), where H𝑡𝑜𝑡𝑎𝑙 mea-
sures the information uncertainty of the LLM in handling
specific inference tasks so that it can effectively approximate
the uncertainty value ( U).𝜖is a small constant used to avoid
numerical issues in logarithmic calculations.
•Correction Rate ( CR=𝐶𝑐𝑜𝑟𝑟𝑒𝑐𝑡
𝑇−𝐿) measures the RG system’s
ability to correct initially incorrect answers. Correspond-
ingly,H𝐶𝑅=−(𝐶𝑅log2(𝐶𝑅+𝜖)+(1−𝐶𝑅)log2((1−𝐶𝑅)+𝜖)),
used to measure the uncertainty in the LLM’s ability to cor-
rect its previous incorrect answers after introducing conflict-
ing information.
•Resistance Rate ( RR=𝐶𝑑𝑒𝑓𝑒𝑛𝑠𝑒
𝐿) measures RG system’s resis-
tance to incorrect information. The calculation method of
H𝑅𝑅is similar to that for H𝐶𝑅.
•Mislead Rate ( MR=𝐶𝑚𝑖𝑠𝑙𝑒𝑎𝑑
𝐿) measures the instances the
LLM is misled by incorrect information. Overconfidence
Rate ( OR=𝐶𝑜𝑣𝑒𝑟𝑐𝑜𝑛𝑓𝑖𝑑𝑒𝑛𝑐𝑒
𝑇−𝐿) counts the instances the LLM
is overconfident on incorrect information. Similarly, H𝑀𝑅
andH𝑂𝑅are also calculated by following the similar method
above.
2) Open-Ended QA : The open-ended task aims to evaluate Swin-
VIB’s capability to maintain high-quality and reliable response
generation under conditions where the context includes potential
conflicting information. For each query, we utilize the key "counter-
memory" in the dataset to fill the slot <conflicting external context>
in the prompt template (see Figure 5(b)).
To evaluate the performance of Swin-VIB on Open-Ended QA,
we utilize BLEU-4 [ 24], METEOR [ 7], and CHRF [ 25] to evaluate
the linguistic accuracy and relevance of the generated responses,
which consider n-gram overlap, synonymy, and character-level pre-
cision and recall. The key "ground truth" in ConflictQA and the key
"best answer" in TruthfulQA are used as the ground truth used for
evaluation.
3) Improving RAG Systems with Swin-VIB : This task aims to
explore the improvements in generation performance and efficiency
of RAG systems with Swin-VIB. We evaluate various performance
metrics of RAG systems on database TruthfulQA. To implement the
seamless integration of Swin-VIB into existing RAG frameworks, an
Elasticsearch-based retrieval system [ 10] with an embedding modelnamed by m3e-base [ 35] and a database named by TruthfulQA is
constructed. Specifically, for each query, we retrieve the top five
correct or incorrect responses to fill the slot <conflicting external
context> in the corresponding templates (see Figure 5(c)). A com-
prehensive set of metrics followed by benchmark RAGAS [ 11] is
used in this task, including similarity, correctness, relevance, and
faithfulness. Finally, we use the key "best answer" in TruthfulQA
as the ground truth for the performance evaluation of Swin-VIB in
RAG.
4.2 Empirical Verification of LLM’s Preference
Figure 6: Distributions of LLM-generated responses with dif-
ferent LLM preference when a different type of external
information is provided, and the corresponding uncertainty
(H𝑡𝑜𝑡𝑎𝑙) for each case.
To verify the finding in Section 2, we explore the preferences
of LLMs through the single-choice task. The uncertainty of the
output, H𝑡𝑜𝑡𝑎𝑙 corresponding toU, is investigated by adapting the
types of external information. Two types of external information
are involved: a) Conflicting context: The “counter-memory” key in
ConflictQA contains explicit conflicting information. b) Supplemen-
tary context: A query prompt, “<query> A:<option a> B:<option b>
Please answer the supplementary knowledge that is not related to
option A and option B based on the query”, is used to make GPT-
4o [1] generate different amount of supplementary information
In this way, we conduct the following experiments (see Figure 6):
•For case (1), replacing 50% external information with con-
flicting context to simulate an RG system with significant
conflicting information, pushing the LLM into the confident
region.
•For case (2), increasing the proportion of the conflicting
context to 100% and simulating more significant information
difference. Results show that H𝑡𝑜𝑡𝑎𝑙 decreases, while RR and

Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 3: Evaluation results in the scenario of single-choice. The 1stbest results are indicated in red and the 2ndbest results are
indicated in blue. All values are percentages.
LLM MethodConflictQA TruthfulQA
ACC↑ CR↑ RR↑ MR↓ OR↓ ACC↑ CR↑ RR↑ MR↓ OR↓
Llama2-7BNaive LLM 20.61 - - - - 56.70 - - - -
Naive RAG 79.68 96.49 16.92 82.38 0.00 59.36 50.60 65.85 26.89 46.79
TACS 81.84 97.73 20.68 39.56 2.26 62.56 49.14 72.79 25.15 45.79
Rowen-CL 80.10 92.59 28.55 45.37 6.18 61.44 49.66 70.24 25.01 47.75
Swin-VIB (Mean) 84.04 98.18 29.40 70.00 1.77 70.10 58.96 78.46 19.14 40.97
Swin-VIB (Montecarlo) 85.12 99.02 31.47 68.03 0.00 69.70 59.70 77.24 20.43 36.79
Qwen-7BNaive LLM 22.24 - - - - 58.87 - - - -
Naive RAG 76.45 94.03 14.19 76.11 5.96 51.90 51.07 52.53 46.01 47.73
TACS 87.90 91.08 76.62 22.76 7.91 64.74 49.08 76.51 21.61 49.10
Rowen-CL 80.18 88.01 52.46 47.89 10.31 67.56 56.71 75.64 22.89 42.89
Swin-VIB (Mean) 90.75 94.37 77.90 22.01 5.39 79.44 69.07 87.36 11.31 30.91
Swin-VIB (Montecarlo) 90.87 94.66 77.59 22.19 5.18 78.81 69.23 85.94 13.21 29.90
Table 4: External information adaption in the scenario of single-choice. The 1stbest results are indicated in red and the 2nd
best results are indicated in blue.
LLM MethodConflictQA TruthfulQA
H𝐶𝑅↓H𝑅𝑅↓H𝑀𝑅↓H𝑂𝑅↓H𝐶𝑅↓H𝑅𝑅↓H𝑀𝑅↓H𝑂𝑅↓
Llama2-7BNaive RAG 0.22 0.66 0.68 0.00 0.99 0.93 0.83 0.99
TACS 0.16 0.74 0.97 2.25 0.99 0.84 0.83 0.99
Rowen-CL 0.38 0.86 0.99 6.18 0.99 0.88 0.81 0.99
Swin-VIB (Mean) 0.13 0.87 0.88 1.77 0.97 0.75 0.68 0.95
Swin-VIB (Montecarlo) 0.08 0.89 0.88 0.00 0.97 0.77 0.73 0.94
Qwen-7BNaive RAG 0.33 0.59 0.79 0.33 0.99 0.99 0.99 0.99
TACS 0.43 0.78 0.79 0.43 0.99 0.79 0.75 0.99
Rowen-CL 0.53 0.99 0.99 0.49 0.98 0.80 0.77 0.98
Swin-VIB (Mean) 0.31 0.76 0.76 0.30 0.89 0.55 0.51 0.89
Swin-VIB (Montecarlo) 0.30 0.77 0.76 0.29 0.89 0.59 0.56 0.88
OR increase, suggesting that this LLM in the confident region
becomes more confident in its internal memory.
•For case (3), taking 50% supplementary context to simulate
a RG system with significant supplementary information,
pushes the LLM into a reliance region.
•For case (4), to enhance this instance that the supplementary
context is more than conflicting information, the proportion
of the supplementary context is increased to 100%. The re-
sults show that H𝑡𝑜𝑡𝑎𝑙 decreases, while CR and MR increase,
and the LLM has firm confidence in its internal memory.
These observations verify our findings about the relation between
output uncertainty of LLMs and the information difference, which
determines LLM preferences.
Table 5: Uncertainty of Swin-VIB and Naive RAG system
LLMH𝑡𝑜𝑡𝑎𝑙↓
ConflictQA TruthfulQA
Naive
RAGSwin-VIBNaive
RAGSwin-VIB
Llama2-7B 0.690.64(Mean)1.131.06(Mean)
0.63(Montecarlo) 1.03(Montecarlo)
Qwen-7B 0.760.46(Mean)0.980.75(Mean)
0.45(Montecarlo) 0.78(Montecarlo)
4.3 Experimental Results on Different Tasks
In this section, we evaluate the effectiveness of Swin-VIB on three
tasks: single choice, open-ended QA, and improving RAG Systems
with Swin-VIB.Table 6: Experimental results on the open-ended QA task.
LLM Dataset Method BLEU-4 ↑METEOR↑CHRF↑
Llama2-7BConflictQANaive RAG 7.59 42.85 39.23
Swin-VIB 15.71 50.95 47.15
TruthfulQANaive RAG 10.67 39.89 37.06
Swin-VIB 12.92 43.26 40.70
Qwen-7BConflictQANaive RAG 3.46 15.33 20.10
Swin-VIB 5.23 19.71 25.78
TruthfulQANaive RAG 17.80 43.26 48.36
Swin-VIB 18.92 43.57 49.41
4.3.1 Single Choice. We compare Swin-VIB with 4 baselines in-
cluding Naive LLM, Naive RAG, TACS [ 45], and Rowen-CL [ 8].
From Table 3 and Table 4, we can make the following observations:
(1) Swin-VIB achieves SOTA in almost all metrics, especially the
ACC on TruthfulQA is improved by 7.54% for Llama2-7b, and is
improved by 11.88% for Qwen-7B. In Table 4, Swin-VIB achieves the
smallest output uncertainty compared to other baselines, indicating
that it can adapt knowledge differences and reduce uncertainty
significantly. (2) As CR increases, H𝐶𝑅decreases. Meanwhile, if MR
decreases, H𝑀𝑅also decreases. This demonstrates that Swin-VIB
rejects the external context with high uncertainty, and thus shows
resilience to incorrect external information. (3) On conflictQA, H𝑅𝑅
also increases, showing that the LLM lacks confidence in its ability
to resist incorrect external information. In this case, although the
LLM seems confused, RR is significantly improved by Swin-VIB,
since Swin-VIB can distinguish the uncertainty of the LLM output
and ensure reliable accuracy.
As shown in Figure 7, Swin-VIB has significantly improved ACC.
At the same time, the AR decreases, which validates that Swin-
VIB can alleviate LLMs’ hesitation, and reduce the probability to
abandon answering. Furthermore, as shown in Table 5, the decrease
inH𝑡𝑜𝑡𝑎𝑙 verifies smaller output uncertainty of the LLM.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
(a) Llama2-7B
 (b) Qwen-7B
Figure 7: Performance comparison in the scenario of single-choice
(a) BLEU-4 obtained on ConflictQA
 (b) BLEU-4 obtained on TruthfulQA
 (c) CHRF obtained on ConflictQA
(d) CHRF obtained on TruthfulQA
 (e) METROR obtained on ConflictQA
 (f) METROR obtained on TruthfulQA
Figure 8: The result distribution of Llama2-7B with Swin-VIB and Naive RAG.
(a) Instance-level entropy with TruthfulQA based on
Llama2-7B
(b) Instance-level entropy with ConflictQA based on
Qwen-7B
Figure 9: A comparison of instance-level uncertainty
4.3.2 Open-ended QA. To evaluate the negative impact of Swin-
VIB on the authenticity and coherence of the responses generated
by the original LLM, we implement a task of Open-ended QA. Com-
pared to Naive RAG, the involved sliding window mechanism of
Swin-VIB may lead to the inconsistency of external context and even
worse logical errors. To explore whether Swin-VIB has a negative
impact on the original LLM, we compare the responses generated
by the LLM with Naive RAG and those from the LLM with Swin-VIB
in scenarios with conflicting information. Table 6 shows these two
methods have similar evaluation results across BLEU-4, METEOR,
and CHRF. This reveals that the process for the external context
adaptation by the sliding window does not affect the authenticity
and coherence of the response generation quality. On the other
hand, Figure 8 illustrates the distribution of the evaluation results
on Llama2-7B with Swin-VIB. These evaluation results concentratemore on the higher values, which means Swin-VIB has even better
performance in response generation. The external context adap-
tion of Swin-VIB is beneficial to reduce the uncertainty of LLM
output. In addition, we measure the instance-level information en-
tropy of generated responses. We calculate this type of entropy by
the following formula, −Í𝑚
𝑖=1𝑝(𝑤𝑖)log2𝑝(𝑤𝑖), where𝑝(𝑤𝑖)de-
notes the probability of occurrence of the word 𝑤𝑖in the response
and𝑚is the total number of distinct words of instance-level re-
sponse. Figure 9 shows the LLM with Swin-VIB responses have
lower entropy, qualifying the instance-level response generation.
All of these observations ensure that the quality improvement of
response generation is attributed solely to Swin-VIB’s design.
4.3.3 Improving RAG Systems with Swin-VIB. In this subsection,
we demonstrate the enhancement of Swin-VIB in the RAG system.
Swin-VIB seamlessly integrates with the RAG system and remains
sufficiently lightweight. To evaluate the effectiveness of Swin-VIB
in the RAG task, we compare Swin-VIB and Naive RAG (shown in
Table 7):
•The improvements in answer correctness demonstrate Swin-
VIB’s ability to provide reliable responses. This is due to
more stable preferences helping the LLM select external
information even when introducing the top 5 contradicted
contexts.
•The slight improvement in the similarity and relevance of the
generated responses indicates that Swin-VIB guides the LLM
to integrate internal and external knowledge while aligning
closely with the query’s intention.
•The decrease in faithfulness suggests that the response gen-
eration does not rigidly follow the retrieved information.

Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 7: Evaluate RAG tasks through RAGAS [11] on TruthfulQA. All values are percentages.
Dataset Methods Answer Similarity ↑ Answer Correctness ↑ Answer Relevance↑ Faithfulness↓
Llama2-7BNaive RAG 86.37 48.05 94.71 73.03
Swin-VIB 87.21 52.86 94.68 66.58
Qwen-7BNaive RAG 87.52 47.33 95.18 74.08
Swin-VIB 87.86 52.25 95.73 62.43
(a) Convergence Analysis on Bottleneck 1 of Qwen-7B
 (b) Bottleneck Analysis on ConflictQA
 (c) Bottleneck Analysis on TruthfulQA
Figure 10: Convergence Analysis.
(a) Bottleneck Analysis on ConflictQA
 (b) Bottleneck Analysis on TruthfulQA
Figure 11: Parameter analysis.
This implies that the sliding window mechanism can effec-
tively filter the confused context, rather than simply using
all retrieved information.
In terms of efficiency, we evaluated the impact of the Swin-VIB
plugin on RG delay for two LLMs, Llama2-7B, and Qwen-7B, as
detailed in Table 8. For Llama2-7B, Swin-VIB gets an average delay
of 0.3913 seconds for the mean method and 3.5613 seconds for Mon-
tecarlo method. Qwen-7B experienced similar increases of 0.3874
seconds (mean method) and 3.5922 seconds (Monte Carlo method).
The mean method presents a minimal delay increase, optimal for
time-sensitive applications, while the Monte Carlo method, though
slower, enhances robustness for more complex scenarios. Swin-VIB
thus offers a balance between efficiency and accuracy, suitable for
diverse operational contexts.
Table 8: Efficiency evaluation on TruthfulQA
LLMAverage time required per instance with Swin-VIB
Naive RAGInference
MethodCost per window Cost
Llama2-7B 0.4912Mean0.08ms+0.3913s
Montecarlo +3.5613s
Qwen-7B 0.1862Mean0.12ms+0.3874s
Montecarlo +3.5922s
4.4 Model Analysis
4.4.1 Convergence Evaluation. As shown in Figure 10(a), the loss
function of our bottleneck structure stabilizes around 200 epochs,
illustrating its efficient convergence during bottleneck training. Weexplore the bottleneck training on a validation dataset in terms of
the following two metrics:
•Attention Score: It indicates how attention dynamics come
to be stabilized. It applies logistic regression to calculate
the F1 scores for layers in each transformer and performs
normalization on the obtained F1 scores. The stable attention
score verifies the capability of the LLM to accommodate the
conflicting information.
•Mean Squared Error (MSE): We calculate MSE of the pre-
dictions of each bottleneck and the ground truth. The small
MSE indicates the conflicting information is accommodated.
The attention Score values come to stable except for several bottle-
necks that suggest challenges in handling conflicting information.
Figure 10(b) and Figure 10(c) show that we need to adapt bottle-
necks with the corresponding Transformers. Meanwhile, deeper
features tend to reduce the need for dynamic attention adjustment.
4.4.2 Parameter Analysis. We change the bottleneck parameter
of Swin-VIB, 𝛽, from 10−2to10−6to study its effect on response
generation. Figure 11 demonstrates the ACC and H𝑡𝑜𝑡𝑎𝑙 fluctuate
with different 𝛽, as well as a negative correlation between ACC
and uncertainty, validating our approach can minimize uncertainty
through external information adaption. On dataset ConflictQA,
a narrower bottleneck ( 𝛽=10−6) can more effectively compress
information representation, improving response quality. In contrast,
On dataset TruthfulQA, the best accuracy is achieved at 𝛽=10−5,
but declined when 𝛽=10−6. These observations confirm that
the configuration of the bottleneck parameters effect the output
uncertainty and thus has on impact on the response generation
quality of the LLM with Swin-VIB.
5 Related Works
Efforts to mitigate risks associated with these conflicts in RG can
be broadly categorized into two groups: a) Internal knowledge-
driven methods [ 3,6,20,28,29,34,44,46] enhances the LLMs’
awareness of potential conflicts independently by fine-tuning with
updated knowledge; b) External-validation methods [ 5,8,13,18,
33,36,39,43] incorporate verification steps to ensure the reliability
of retrieved contexts.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
However, existing methods have limitations in fully elucidating
LLMs’ preference mechanisms: a)Internal catastrophic forgetting of
LLMs caused by fine-tuning [ 40]; b)External validation always leads
to overconfidence in external sources, and thus degrades the LLMs’
inference precise [ 38]. In contrast, we propose a novel RG method
[21], Swin-VIB. It only uses internal attention scores to guide the
training of external reliability enhancement modules, without any
change of the pre-trained LLMs. On the other hand, Swin-VIB adapts
conflicted knowledge, avoiding solely relying on internal or external
information. A bottleneck-based adapter is involved to regulate the
compression ratio of the retrieved information and optimize the
posterior distribution of the LLM-empowered response generation.
To the best of our knowledge, this is the first time information
bottleneck theory is used to adapt data representations.
6 CONCLUSION
This work proposes a novel theoretical framework to analyze and
address the issue of information conflicts encountered by LLMs in
RG systems. Leveraging analysis on the knowledge conflicts and
preferences of LLMs from an information theory perspective, we
find that the uncertainty of LLMs can be mitigated by adapting the
external information difference of LLMs. Insighted by this, we pro-
pose Swin-VIB to optimize how RAG handles external contexts to
achieve reliable response generation. Experimental results demon-
strate that Swin-VIB significantly accommodates conflicts, reduces
uncertainty in LLM outputs, and generates more accurate, consis-
tent, and context-aware responses. Moreover, Swin-VIB enhances
retrieval system performance, facilitating its real-world applica-
tions. Future work will explore extending this approach to more
types of response generation tasks to further validate and enhance
its effectiveness.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774
(2023).
[2]Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy. 2016. Deep
variational information bottleneck. arXiv preprint arXiv:1612.00410 (2016).
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv
preprint arXiv:2310.11511 (2023).
[4]Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji
Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men,
Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng
Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang,
Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng
Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang
Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen Technical
Report. arXiv preprint arXiv:2309.16609 (2023).
[5]I Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou,
Junxian He, Graham Neubig, Pengfei Liu, et al .2023. FacTool: Factuality De-
tection in Generative AI–A Tool Augmented Framework for Multi-Task and
Multi-Domain Scenarios. arXiv preprint arXiv:2307.13528 (2023).
[6]Boyi Deng, Wenjie Wang, Fengbin Zhu, Qifan Wang, and Fuli Feng. 2024. CrAM:
Credibility-Aware Attention Modification in LLMs for Combating Misinformation
in RAG. arXiv preprint arXiv:2406.11497 (2024).
[7]Michael Denkowski and Alon Lavie. 2014. Meteor universal: Language spe-
cific translation evaluation for any target language. In Proceedings of the ninth
workshop on statistical machine translation . 376–380.
[8]Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen, and Xueqi Cheng. 2024.
Retrieve only when it needs: Adaptive retrieval augmentation for hallucination
mitigation in large language models. arXiv preprint arXiv:2402.10612 (2024).[9]Nicholas Dufour, Arkanath Pathak, Pouya Samangouei, Nikki Hariri, Shashi
Deshetti, Andrew Dudfield, Christopher Guess, Pablo Hernández Escayola, Bobby
Tran, Mevan Babakar, et al .2024. AMMeBa: A Large-Scale Survey and Dataset
of Media-Based Misinformation In-The-Wild. arXiv preprint arXiv:2405.11697
(2024).
[10] Elastic. 2023. Elasticsearch. https://www.elastic.co/guide/en/elasticsearch/
reference/current/elasticsearch-intro-what-is-es.html. Accessed: 2023-10-10.
[11] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. 2023. Ra-
gas: Automated evaluation of retrieval augmented generation. arXiv preprint
arXiv:2309.15217 (2023).
[12] Josh A Goldstein, Girish Sastry, Micah Musser, Renee DiResta, Matthew Gentzel,
and Katerina Sedova. 2023. Generative language models and automated influ-
ence operations: Emerging threats and potential mitigations. arXiv preprint
arXiv:2301.04246 (2023).
[13] Giwon Hong, Jeonghwan Kim, Junmo Kang, Sung-Hyon Myaeng, and Joyce Jiy-
oung Whang. 2023. Why So Gullible? Enhancing the Robustness of Retrieval-
Augmented Models against Counterfactual Noise. arXiv preprint arXiv:2305.01579
(2023).
[14] Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Qiuxia
Li, and Jun Zhao. 2024. Tug-of-war between knowledge: Exploring and resolving
knowledge conflicts in retrieval-augmented language models. arXiv preprint
arXiv:2402.14409 (2024).
[15] Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li,
Xiaojian Jiang, Kang Liu, and Jun Zhao. 2024. Cutting Off the Head Ends the
Conflict: A Mechanism for Interpreting and Mitigating Knowledge Conflicts in
Language Models. arXiv preprint arXiv:2402.18154 (2024).
[16] Jungo Kasai, Keisuke Sakaguchi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir
Radev, Noah A Smith, Yejin Choi, Kentaro Inui, et al .2024. REALTIME QA: what’s
the answer right now? Advances in Neural Information Processing Systems 36
(2024).
[17] Diederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes.
arXiv preprint arXiv:1312.6114 (2013).
[18] João A Leite, Olesya Razuvayevskaya, Kalina Bontcheva, and Carolina Scarton.
2023. Detecting misinformation with llm-predicted credibility signals and weak
supervision. arXiv preprint arXiv:2309.07601 (2023).
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[20] Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, An-
dreas Veit, Felix Yu, and Sanjiv Kumar. 2022. Large language models with con-
trollable working memory. arXiv preprint arXiv:2211.05110 (2022).
[21] Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and
Zhicheng Dou. 2024. From matching to generation: A survey on generative
information retrieval. arXiv preprint arXiv:2404.14851 (2024).
[22] Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how
models mimic human falsehoods. arXiv preprint arXiv:2109.07958 (2021).
[23] Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois,
and Sameer Singh. 2021. Entity-based knowledge conflicts in question answering.
arXiv preprint arXiv:2109.05052 (2021).
[24] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a
method for automatic evaluation of machine translation. In Proceedings of the
40th annual meeting of the Association for Computational Linguistics . 311–318.
[25] Maja Popović. 2015. chrF: character n-gram F-score for automatic MT evaluation.
InProceedings of the tenth workshop on statistical machine translation . 392–395.
[26] Stephen E Robertson and K Sparck Jones. 1976. Relevance weighting of search
terms. Journal of the American Society for Information science 27, 3 (1976), 129–
146.
[27] Claude Elwood Shannon. 1948. A mathematical theory of communication. The
Bell system technical journal 27, 3 (1948), 379–423.
[28] Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer,
and Scott Wen-tau Yih. 2023. Trusting your evidence: Hallucinate less with
context-aware decoding. arXiv preprint arXiv:2305.14739 (2023).
[29] Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Victoria Lin,
Noah A Smith, Luke Zettlemoyer, Scott Yih, and Mike Lewis. 2023. In-context
pretraining: Language modeling beyond document boundaries. arXiv preprint
arXiv:2310.10638 (2023).
[30] Naftali Tishby, Fernando C Pereira, and William Bialek. 2000. The information
bottleneck method. arXiv preprint physics/0004057 (2000).
[31] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023).
[32] Ashish Vaswani. 2017. Attention is all you need. arXiv preprint arXiv:1706.03762
(2017).
[33] Hongru Wang, Boyang Xue, Baohang Zhou, Tianhua Zhang, Cunxiang Wang,
Guanhua Chen, Huimin Wang, and Kam-fai Wong. 2024. Self-DC: When to
retrieve and When to generate? Self Divide-and-Conquer for Compositional

Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Unknown Questions. arXiv preprint arXiv:2402.13514 (2024).
[34] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023. Self-knowledge guided
retrieval augmentation for large language models. arXiv preprint arXiv:2310.05002
(2023).
[35] Yuxin Wang, Qingxuan Sun, and Sicheng He. 2023. M3E: Moka Massive Mixed
Embedding Model .
[36] Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, and Benjamin
Van Durme. 2022. Defending Against Misinformation Attacks in Open-Domain
Question Answering. arXiv preprint arXiv:2212.10002 (2022).
[37] Kevin Wu, Eric Wu, and James Zou. 2024. How faithful are RAG models? Quan-
tifying the tug-of-war between RAG and LLMs’ internal prior. arXiv preprint
arXiv:2404.10198 (2024).
[38] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. 2023. Adaptive
chameleon or stubborn sloth: Revealing the behavior of large language mod-
els in knowledge conflicts. In The Twelfth International Conference on Learning
Representations .
[39] Rongwu Xu, Brian S Lin, Shujian Yang, Tianqi Zhang, Weiyan Shi, Tianwei
Zhang, Zhixuan Fang, Wei Xu, and Han Qiu. 2023. The Earth is Flat because...:
Investigating LLMs’ Belief towards Misinformation via Persuasive Conversation.
arXiv preprint arXiv:2312.09085 (2023).
[40] Rongwu Xu, Zehan Qi, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu.
2024. Knowledge Conflicts for LLMs: A Survey. arXiv preprint arXiv:2403.08319(2024).
[41] Shicheng Xu, Liang Pang, Huawei Shen, and Xueqi Cheng. 2024. Unveil the
Duality of Retrieval-Augmented Generation: Theoretical Analysis and Practical
Solution. arXiv preprint arXiv:2406.00944 (2024).
[42] Haoyan Yang, Zhitao Li, Yong Zhang, Jianzong Wang, Ning Cheng, Ming Li, and
Jing Xiao. 2023. Prca: Fitting black-box large language models for retrieval ques-
tion answering via pluggable reward-driven contextual adapter. arXiv preprint
arXiv:2310.18347 (2023).
[43] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models.
arXiv preprint arXiv:2210.03629 (2022).
[44] Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng,
Huajun Chen, and Ningyu Zhang. 2023. Editing large language models: Problems,
methods, and opportunities. arXiv preprint arXiv:2305.13172 (2023).
[45] Tian Yu, Shaolei Zhang, and Yang Feng. 2024. Truth-Aware Context Selection:
Mitigating the Hallucinations of Large Language Models Being Misled by Un-
truthful Contexts. arXiv preprint arXiv:2403.07556 (2024).
[46] Shaolei Zhang, Tian Yu, and Yang Feng. 2024. Truthx: Alleviating hallucinations
by editing large language models in truthful space. arXiv preprint arXiv:2402.17811
(2024).
Received 16 January 2025