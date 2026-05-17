# Towards Order Fairness: Mitigating LLMs Order Sensitivity through Dual Group Advantage Optimization

**Authors**: Xu Chu, Guanyu Wang, Zhijie Tan, Xinrong Chen, Ziyu Li, Tong Mo, Weiping Li

**Published**: 2026-05-12 11:31:18

**PDF URL**: [https://arxiv.org/pdf/2605.11974v1](https://arxiv.org/pdf/2605.11974v1)

## Abstract
Large Language Models (LLMs) suffer from order bias, where their performance is affected by the arrangement order of input elements. This unfairness limits the model's applications in scenarios such as in-context learning and Retrieval-Augmented Generation (RAG). Recent studies attempt to obtain optimal or suboptimal arrangements based on statistical results or using dataset-based search, but these methods increase inference overhead while leaving the model's inherent order bias unresolved. Other studies mitigate order sensitivity through supervised fine-tuning using augmented training sets with multiple order variants, but often at the cost of accuracy, trapping the model in consistent yet incorrect hallucinations. In this paper, we propose \textbf{D}ual \textbf{G}roup \textbf{A}dvantage \textbf{O}ptimization (\textbf{DGAO}), which aims to improve model accuracy and order stability simultaneously. DGAO calculates and balances intra-group relative accuracy advantage and inter-group relative stability advantage, rewarding the policy model for generating order-stable and correct outputs while penalizing order-sensitive or incorrect responses. This marks the first time reinforcement learning has been used to mitigate LLMs' order sensitivity. We also propose two new metrics, Consistency Rate and Overconfidence Rate, to reveal the pseudo-stability of previous methods and guide more comprehensive evaluation. Extensive experiments demonstrate that DGAO achieves superior order fairness while improving performance on RAG, mathematical reasoning, and classification tasks. Our code is available at: https://github.com/Hyalinesky/DGAO.

## Full Text


<!-- PDF content starts -->

Towards Order Fairness: Mitigating LLMs Order Sensitivity through
Dual Group Advantage Optimization
Xu Chu†, Guanyu Wang†, Zhijie Tan, Xinrong Chen, Ziyu Li, Tong Mo, Weiping Li*
School of Software and Microelectronics, Peking University, Beijing, China
{chuxu, wgy2023, besttangent, chenxinrong23, liziyu_lizy }@stu.pku.edu.cn,
{motong, wpli}@ss.pku.edu.cn
†Equal contribution*Corresponding author
Abstract
Large Language Models (LLMs) suffer from
order bias, where their performance is affected
by the arrangement order of input elements.
This unfairness limits the model’s applications
in scenarios such as in-context learning and
Retrieval-Augmented Generation (RAG). Re-
cent studies attempt to obtain optimal or subop-
timal arrangements based on statistical results
or using dataset-based search, but these meth-
ods increase inference overhead while leav-
ing the model’s inherent order bias unresolved.
Other studies mitigate order sensitivity through
supervised fine-tuning using augmented train-
ing sets with multiple order variants, but of-
ten at the cost of accuracy, trapping the model
in consistent yet incorrect hallucinations. In
this paper, we proposeDualGroupAdvantage
Optimization (DGAO), which aims to improve
model accuracy and order stability simultane-
ously. DGAO calculates and balances intra-
group relative accuracy advantage and inter-
group relative stability advantage, rewarding
the policy model for generating order-stable
and correct outputs while penalizing order-
sensitive or incorrect responses. This marks the
first time reinforcement learning has been used
to mitigate LLMs’ order sensitivity. We also
propose two new metrics, Consistency Rate
and Overconfidence Rate, to reveal the pseudo-
stability of previous methods and guide more
comprehensive evaluation. Extensive exper-
iments demonstrate that DGAO achieves su-
perior order fairness while improving perfor-
mance on RAG, mathematical reasoning, and
classification tasks. Our code is available at:
https://github.com/Hyalinesky/DGAO.
1 Introduction
Large Language Models (LLMs) achieve remark-
able capabilities in utilizing input text to solve
tasks (Hurst et al., 2024; Guo et al., 2025; Yang
et al., 2025), playing crucial roles in recommen-
dation systems (Lin et al., 2024; Du et al., 2024),
Figure 1: Consistency Rates (obtained from 8 random
order variants of a sample, with higher values indicat-
ing stronger order stability) of LLMs on RAG dataset
SearchQA (Dunn et al., 2017) are consistently below
the ideal 100%. Moreover, as the number of tokens in
the context window increases, the Consistency Rates of
LLMs continue to decrease.
search systems (Narayanan Venkit et al., 2025; Li
et al., 2025), and domain-specific question answer-
ing (Goyal et al., 2024; Chu et al., 2025b). In par-
ticular, LLMs demonstrate powerful capabilities in
in-context learning (Bertsch et al., 2024; Agarwal
et al., 2024) and Retrieval-Augmented Generation
(RAG) (Yu et al., 2024b; He et al., 2024) tasks,
being able to extract important information from
contexts composed of numerous input contents and
derive answers.
However, current mainstream LLMs face order
bias (Lu et al., 2021; Liu et al., 2023; Wang et al.,
2024a), even for cutting-edge large-parameter
LLMs, as shown in Figure 1. This unfairness may
affect their applications. In in-context learning
and RAG scenarios, the prompts or documents that
need to be input to LLMs can be viewed as ele-
ments in an unordered set, where the order of these
elements is typically interchangeable and is not in-
tended to influence the results (Zhang et al., 2024b).
However, LLMs organize the elements in the set
into natural language sequences when processing
input, thereby introducing ordering to originally
unordered elements. Recent research shows that
LLMs are sensitive to input order. For example,arXiv:2605.11974v1  [cs.LG]  12 May 2026

statistical results indicate that many LLMs exhibit
additional attention to the beginning and end of se-
quences (Liu et al., 2023), and LLMs’ performance
typically deteriorates when key information is lo-
cated in the middle of the input rather than at the
beginning or end. Although placing key informa-
tion in contextual positions that LLMs pay more at-
tention to shows promise for performance improve-
ment (Tan et al., 2024a; Yao et al., 2025), defin-
ing key information such as key frames or golden
documents is very difficult in many cases. Other
studies employ dataset-based search to find opti-
mal or suboptimal arrangement orders (Lu et al.,
2021; Sorensen et al., 2022; Guo et al., 2024), but
the search process requires introducing additional
inference time and typically relies on additional val-
idation sets. These methods do not improve model
structure or adjust parameters, merely utilizing the
order bias phenomenon to enhance LLMs’ perfor-
mance on specific tasks, and cannot fundamentally
mitigate the model’s unfair bias.
Some studies modify the structure of LLMs to
avoid or mitigate positional bias (Tan et al., 2024b;
Egressy and Stühmer, 2025; Chu et al., 2025c), but
new structures require introducing large amounts
of data for training. Orthogonally, fine-tuning mod-
els to suppress order bias shows promise. (Zhang
et al., 2024b) constructs multiple order variants
for each training sample and uses these variants as
new samples for supervised fine-tuning (SFT) of
LLMs. However, as shown in Table 1, although
the model’s order stability improves, its accuracy
decreases. This reveals a disturbing phenomenon:
simply fine-tuning models through data augmenta-
tion may cause models to generate consistently in-
correct answers in pursuit of response consistency,
which can be harmful to downstream tasks.
In this paper, to the best of our knowledge, this is
the first time reinforcement learning is used to en-
hance LLMs’ order stability while simultaneously
improving their accuracy. We proposeDualGroup
AdvantageOptimization (DGAO), an on-policy
reinforcement learning algorithm that rewards the
policy model for generating order-stable and cor-
rect outputs and rejects order-sensitive or incor-
rect outputs, thereby avoiding the model accuracy
decline brought by SFT with data augmentation.
DGAO calculates and balances intra-group rela-
tive advantage and inter-group relative advantage,
where intra-group relative advantage rewards out-
puts with correct answers, while inter-group rel-
ative advantage rewards order-stable groups. WeDataset Model Accuracy Consistency Rate
SearchQALlama-3.2-3B-SFT 58.76 81.21
Llama-3.2-3B-PAFT 56.31 (-2.45) 86.48 (+5.27)
Gemma-3-4B-SFT 59.71 78.09
Gemma-3-4B-PAFT 58.72 (-0.99) 83.86 (+5.77)
GSM8KLlama-3.2-3B-SFT 22.81 56.85
Llama-3.2-3B-PAFT 14.43 (-8.38) 58.99 (+2.14)
Gemma-3-4B-SFT 15.24 44.42
Gemma-3-4B-PAFT 12.19 (-3.05) 48.31 (+3.89)
Table 1: The accuracy and answer Consistency Rate of
Llama-3.2-3B and Gemma-3-4B on the RAG dataset
SearchQA and math dataset GSM8K constructed with
8 random orders. Position Aware Fine Tuning (PAFT)
refers to augmenting each sample in the training set
into 8 order variants for fine-tuning. Consistency Rate
represents the maximum consistency rate of answers
obtained from 8 random order variants of a sample,
with higher values indicating stronger order stability.
+/− indicates improvement or decline relative to the
SFT baseline.
also propose two new metrics: Consistency Rate,
which evaluates response consistency, and Over-
confidence Rate, which measures how much a
model’s excessive focus on response consistency
affects its accuracy. Extensive experiments on three
different model architectures of LLMs across math-
ematics, sentiment classification, and RAG tasks
demonstrate that DGAO enhances model accuracy
and reduces order sensitivity. Our contributions
can be summarized as:
• We reveal a key pitfall in existing mitigation
strategies: merely improving order stability may
harm accuracy. Current SFT-based training strate-
gies lack simultaneous, explicit supervision for the
two objectives, making it difficult to optimize sta-
bility and accuracy concurrently.
• We propose DGAO, a reinforcement learning
algorithm that rewards LLMs for generating order-
stable and correct outputs. This is the first time
reinforcement learning is used to enhance LLM
performance and order stability simultaneously.
• We propose Consistency Rate and Overcon-
fidence Rate for evaluating LLMs’ order stabil-
ity. Experiments show that LLMs reinforced with
DGAO achieve simultaneous improvements in ac-
curacy and order stability across mathematics, sen-
timent classification, and RAG tasks, with LLMs
gradually moving toward order fairness.
2 Related Works
2.1 Order Sensitivity in LLMs
Current mainstream LLMs face the problem of or-
der sensitivity (Lu et al., 2021; Liu et al., 2023;

Wang et al., 2024a), where when the order of ele-
ments in the input context changes, even though it
is not intended to affect the results, model outputs
vary in quality. Previous mainstream improvement
strategies include prompting, inference-time inter-
vention, dataset-based search, model architecture
adjustment, and supervised fine-tuning. Methods
based on prompting (Zhang et al., 2024a; Jiao et al.,
2024b; Li et al., 2024b) or inference-time interven-
tion (Wang et al., 2024a; Yu et al., 2024a; Zhou
et al., 2024; Wei et al., 2024) lack data-driven train-
ing. (Zhang et al., 2024a) proves that LLMs lack
relative position awareness capability, and prompt-
ing methods that rely on contextual position percep-
tion such as (Jiao et al., 2024b; Li et al., 2024b) may
fail. Inference-time intervention methods (Wang
et al., 2024a; Wei et al., 2024) often rely on heuris-
tic assumptions and lack generalizability, poten-
tially introducing biases due to factors such as at-
tention score interference and hallucination.
Dataset-based search (Lu et al., 2021; Sorensen
et al., 2022; Guo et al., 2024) aims to find an op-
timal or sub-optimal example arrangement order
to improve model performance. But they require
additional forward inference (Guo et al., 2024) and
creating or introducing domain datasets (Lu et al.,
2021; Sorensen et al., 2022), thereby increasing
overhead.
Model architecture adjustments suppress order
sensitivity by adding additional components (Tan
et al., 2024b; Chu et al., 2025c) or improving trans-
former structures (Egressy and Stühmer, 2025).
(Egressy and Stühmer, 2025) removes sequential
positional encoding and causal masks from trans-
formers, adding set positional encoding and set
attention masks to make the entire model’s output
invariant to option order permutation. (Chu et al.,
2025c) designs and trains an order selection mod-
ule that organizes text-attributed graphs into better
order for LLMs to better handle graph tasks. How-
ever, new model architectures require additional
training, which typically involves greater training
overhead than fine-tuning, thus introducing sub-
stantial training costs.
Supervised fine-tuning Methods, such as (Zhang
et al., 2024b), do achieve improvements on cer-
tain tasks. It construct each training sample into
multiple order variants and use these variants as
new data for simultaneous training. However, as
shown in Table 1, this approach harms answer accu-
racy, which does not align with the ideal of guiding
models to answer in an order-fair manner.2.2 Reinforcement Learning in LLM
Introducing reinforcement learning to enhance
LLM capabilities has achieved superior results in
domains such as mathematics (Luong et al., 2024;
Shao et al., 2024; Yang et al., 2024; Ying et al.,
2024; Chen et al., 2024) and code (Hui et al., 2024;
Jiao et al., 2024a; Li et al., 2024a; Chen et al., 2025).
Reinforcement learning for LLM fine-tuning can
be categorized into On-Policy (Schulman et al.,
2017; Ouyang et al., 2022; Shao et al., 2024; Ding
et al., 2025) and Off-Policy (Rafailov et al., 2023;
Amini et al., 2024; Chen et al., 2024; Yuan et al.,
2024) based on whether the learning samples are
generated by the current model being optimized.
Off-Policy algorithms, such as DPO (Rafailov
et al., 2023), rely on a static, pre-collected pref-
erence dataset. The model uses a fixed dataset
to learn throughout the entire training process and
does not need to generate new samples during train-
ing. Off-Policy algorithms have high sample effi-
ciency but are prone to distribution shift and are
limited by preference dataset quality.
On-Policy algorithms generate optimization data
in real-time from the current policy model, en-
abling iterative optimization of the policy model.
PPO (Schulman et al., 2017; Ouyang et al., 2022)
trains reward models and value models to perform
advantage estimation for each generation and up-
date parameters. GRPO (Shao et al., 2024) aban-
dons training value models and instead calculates
intra-group relative advantages to guide parameter
updates. However, these algorithms cannot han-
dle one situation: how to optimize when an entire
group deserves to be rewarded or rejected. For
example,when generations within a group are all
order-stable, how to design rewards and advan-
tages to guide the model to learn this order-stable
generation?
3 Methodology
In this section, we introduce Dual Group Advan-
tage Optimization (DGAO). The algorithmic frame-
work is shown in Figure 2. DGAO simultane-
ously calculates intra-group relative accuracy ad-
vantages and inter-group relative stability advan-
tages, thereby encouraging model to generate order-
stable and correct outputs.
3.1 Preliminary
Reinforcement learning that calculates intra-
group advantages has achieved success in recent

 
  
    
    
  
    
          
          
          
          
            
            
            
            Text prompts
 
Text prompts
 
    
    
    
    
    
    
    
    +
+
 
    
    
    
    
 
    
    
    
    
 
    
    
    
    
 
 
    
    
    
    
 
    
    
    
    
 
    
    
    
    
 
  +
  
+
  
+
  
+
  
+
  
+
 
Reward→
→
→
→
→
→Reward→
→
→
→
→
→→
→
→
→
→
→→
→
→
→
→
→→
→
→
LLM
→
→
→
LLM
          
          
            
            
    
    
    
    
    
    
    
    
    
    
→
→
→
→
→
→Shuffle
Shuffle
       
       
       
       
       
       
    Figure 2: DGAO algorithm framework. A batch con-
tainsM(e.g., M= 2 in the figure) queries, each query
consists of text prompts and Q. Each text prompt is con-
structed into Norder variants, and rewards are obtained
through respective inference. Based on the rewards,
intra-group relative advantage Aindand inter-group rel-
ative advantage Agroup are calculated respectively, and
the aggregated final advantage Afinalis obtained to guide
parameter updates.
work (Guo et al., 2025; Chu et al., 2025a), which
removes value models, thereby reducing training
overhead and optimization difficulty. For example,
Group Relative Policy Optimization (GRPO) (Shao
et al., 2024) used to train DeepSeek-R1-Zero (Guo
et al., 2025), based on a simple yet effective rule:
for the same problem, generate several candidate
answers, called a group, and directly compare the
relative merits of candidate answers within the
group. Specifically, for a given query q, GRPO first
uses the current policy πoldto generate Gdifferent
responses y1, y2, . . . , y Gand obtains correspond-
ing rewards r1, r2, . . . , r G. GRPO calculates the
relative advantage of each answer through intra-
group normalization:
Ai=ri−mean({r 1, . . . , r G})
std({r 1, . . . , r G}),(1)
where Airepresents the advantage degree of the
i-th answer relative to other answers within the
group. Based on these relative advantage values,
GRPO’s objective function is:
LGRPO (θ) =1
GGX
i=1
minπθ(yi|q)
πold(yi|q)Ai,
clipπθ(yi|q)
πold(yi|q),1−ϵ,1 +ϵ
Ai
−βD KL(πθ||πref)
,
(2)
DKL(πθ||πref) =πref(yi|q)
πθ(yi|q)−logπref(yi|q)
πθ(yi|q)−1,(3)where ϵandβrespectively control the clipping
range and the strength of KL divergence penalty.
πrefis the reference policy, used to prevent the
optimized policy πθfrom deviating too far and
causing catastrophic forgetting.
3.2 Order Sensitivity Phenomenon
In in-context learning and RAG scenarios, the text
prompts that serve as LLM inputs are typically
unordered and are not intended to influence re-
sults through order changes. For example, doc-
uments retrieved by RAG systems (unsorted), or
prompt examples in in-context learning. These
text prompts can be viewed as elements in an
unordered set. Given a query consisting of a
setS={T 1;T2;. . .;T n}composed of contex-
tual text prompts and a question Q, letπbe a
permutation of the indices {1, . . . , n} . The in-
put sequence in a specific order is represented as
qπ=Tπ(1)⊕Tπ(2)⊕ ··· ⊕T π(n)⊕Q.
Since the elements in set Sare inherently un-
ordered, ideally, a robust model should exhibit
permutation invariance, i.e., for any permutations
π, π′,P(y|q π) =P(y|q π′).However, LLMs
model the joint probability of the output sequence
y= (y 1, . . . , y L)through the chain rule:
P(y|q π) =LY
t=1P(yt|y<t, qπ).(4)
Due to the positional embeddings and causal at-
tention mechanisms in mainstream Transformer-
based LLMs, the hidden state representation htat
steptdepends on the specific absolute and relative
positions of tokens in qπ. Therefore, changes in
the context order πalter the attention weights and
embedding representations by changing token posi-
tions, which may lead to differences in conditional
probability distributions, i.e., P(y|q π)̸=P(y|q π′).
This difference manifests as the model assigning
different probabilities to correct answers under dif-
ferent orders, thereby producing theorder biasphe-
nomenon, as shown in Figure 3. This phenomenon
can lead to unfairness and affect downstream tasks.
Moreover, recent research indicates that there is no
unified optimal order across models or tasks (Lu
et al., 2021), and even the "Lost in the middle" phe-
nomenon, where models focus on the beginning
and end of contexts does not always occur (Zhang
et al., 2024b; Tan et al., 2024a). This brings chal-
lenges to the application of methods that directly
order based on statistical preferences.

Figure 3: Impact of input order on the performance of
Llama-3.2-3B across 8 random orders.
3.3 Dual Group Advantage Optimization
To mitigate order sensitivity, we aim to optimize the
policy πθso that it consistently generates correct
answers under different permutations. We theoreti-
cally analyze the limitations of existing strategies
in Appendix A, including standard SFT on aug-
mented data (PAFT (Zhang et al., 2024b)) and RL
approaches, in addressing this problem.
To better address order sensitivity, we propose
DualGroupAdvantageOptimization (DGAO),
with the algorithm architecture shown in Figure 2.
DGAO is an On-Policy reinforcement learning
algorithm that aims to reward the policy model
for generating order-stable and correct outputs
while rejecting order-sensitive or incorrect out-
puts, thereby avoiding the model accuracy degra-
dation caused by data augmentation SFT. UnlikeGRPO, which only considers intra-group relative
advantage, DGAO calculates and balances both
intra-group relative advantage and inter-group rel-
ative advantage, where intra-group relative advan-
tage rewards outputs with correct answers, while
inter-group relative advantage rewards order-stable
groups.
The core idea of DGAO is based on a key con-
struction: in a training batch, there are Mdifferent
queries, each query consists of ntext prompts and
one question Q, i.e., q=T π(1)⊕Tπ(2)⊕ ··· ⊕
Tπ(n)⊕Q. For each query qj,Nversions with
different orders of text prompts are constructed,
denoted as qj,1, qj,2, . . . , q j,N. For each qj,i, the
model generates a result yj,i. In this way, N
ordered samples and their results form a group.
DGAO defines two advantage components to cap-
ture performance at different levels:
Inter-group advantage Agroup(j)measures the
overall stability of a group by calculating the
group reward Rgroup(j) =1
NPN
i=1R(yj,i)and
global baseline b=1
MPM
j=1Rgroup(j), obtaining
Agroup(j) =R group(j)−b . This advantage remains
the same for all samples within group j, reflecting
the performance of that group as a whole relative
to the batch average. Here, the reward for individ-
ual model results R(yj,i)can be given by a reward
model or rule-based reward function. In our im-
plementation, we use rule-based reward, where the
result gets a score of 1 if correct and 0 if incorrect.
It should be noted that inter-group advantage does
not encourage the model to generate consistently
incorrect answers within a group. Although this
may appear to be "order stable", it would harm
model accuracy. Therefore, groups that tend to out-
put consistently incorrect answers also have lower
inter-group advantage.
Intra-group advantage Aind(yj,i)inherits the
relative advantage idea from GRPO and measures
the accuracy performance of individual generated
results relative to their same-group samples. By
calculating Aind(yj,i) =R(y j,i)−R group(j), this
advantage reflects the difference of each generated
result relative to the average level of its group, may
have different values for different samples within
groupj.
DGAO uses hyperparameter α∈[0,1] to per-
form weighted combination of two advantage com-
ponents, forming hybrid advantage: Ahybrid(yj,i) =
α·A group(j) + (1−α)·A ind(yj,i).When α→1 ,
the model focuses more on order-stability perfor-

mance; when α→0 , the model focuses more
on intra-group accuracy performance. To ensure
training stability, DGAO normalizes Ahybrid for the
entire batch:
Afinal(yj,i) =Ahybrid(yj,i)−µ A
σA+ϵA,(5)
where µAandσAare the mean and standard de-
viation of hybrid advantage in the current batch
respectively, and ϵAis a small constant. Finally,
the objective function of DGAO can be expressed
as:
LDGAO(θ) =1
MNMX
j=1NX
i=1(min(πθ(yj,i|qj,i)
πold(yj,i|qj,i)Afinal(yj,i),
clip(πθ(yj,i|qj,i)
πold(yj,i|qj,i),1−ϵ,1+ϵ)A final(yj,i))−βD KL(πθ||πref)).
(6)
4 Experiments
In this section, we evaluate the performance of
LLMs trained with DGAO in terms of accuracy
and order sensitivity. We aim to address the follow-
ing questions:RQ1: How do models perform af-
ter training with DGAO?RQ2: How does DGAO
compare with other methods for suppressing or-
der sensitivity?RQ3: How does DGAO compare
with other reinforcement learning methods?RQ4:
How do inter-group advantages and intra-group ad-
vantages affect the effectiveness of DGAO?RQ5:
What is the generalization capability of order sta-
bility for models trained with DGAO?
4.1 Experimental Setup
Datasets and metrics. We select 3 order-sensitive
tasks, mathematics, sentiment classification, and
RAG, and choose datasets that are widely used
for evaluation. Mathematics datasets include
GSM8K (Cobbe et al., 2021) and CM17K (Qin
et al., 2021). In our experiments, each dataset
samples 2K training samples, where each sam-
ple’s query context contains 4 randomly sampled
question-answer pairs as prompts, followed by 1
question that the model needs to answer. Then,
we randomly shuffle the order between prompts to
obtain 8 query variants, resulting in a total of 16K
training data, with 8 variants recorded as 1 group.
For sentiment classification dataset, we choose
SST2 (Socher et al., 2013), which consists of sen-
tences extracted from movie reviews with each sen-
tence annotated with a unique Positive or Negative
sentiment, forming a binary classification task. In
our experiments, we sample 2K training samples,where each sample’s query context contains 16
question-answer pairs as prompts, followed by 1
question that the model needs to answer. Then,
we randomly shuffle the order between prompts to
obtain 8 query variants, resulting in a total of 16K
training data, with 8 variants recorded as 1 group.
For RAG tasks, we choose SQuAD v2 (Ra-
jpurkar et al., 2018) and SearchQA (Dunn et al.,
2017). Each question includes the question, answer,
and documents needed to obtain the answer. In
our experiments, each dataset samples 2K training
samples respectively, where each sample’s query
context contains 5 documents, followed by 1 ques-
tion that the model needs to answer. Among the 5
documents, only one is relevant to answering the
current question. Then, we randomly shuffle the
order between prompts to obtain 8 query variants,
resulting in a total of 16K training data, with 8
variants recorded as 1 group.
Test sets for all datasets are constructed in the
same way as the training sets. Examples can be
found in Appendix H.
To comprehensively evaluate the model’s answer
accuracy and order stability, we introduce 3 met-
rics. In test samples, Mrepresents the number of
groups, i.e., the number of original queries. Nrep-
resents the number of order variants within groups,
Ljrepresents the label answer for the j-th group,
yj,irepresents the model’s output for the i-th order
variant query in the j-th group, and I(·)represents
the indicator function. When the condition in paren-
theses is true, its value is 1; when the condition is
false, its value is 0. The metrics are formally de-
fined as follows:
•Accuracy:1
M×NPM
j=1PN
i=1I(yj,i=L j).
It represents the average test accuracy of all test
variants.
•Consistency Rate:
1
MPM
j=1
1
Nmax p∈YjPN
i=1I(yj,i=p)
,
where Yj={y j,1, yj,2, . . . , y j,N}is the set of
output results given by the model for all N
versions of the j-th query. This value represents
the maximum consistency rate of the model’s
answers, with larger values indicating larger
order-stability.
•Overconfidence Rate:
1
MPM
j=1
1
Nmax p∈Yj,p̸=L jPN
i=1I(yj,i=p)
,
where Yj={y j,1, yj,2, . . . , y j,N}.If for a certain
group j, all outputs are correct (i.e., there is
no output pthat is not equal to Lj), then the
value of the max term for that group is 0. This

Model Dataset Accuracy↑Consistency Rate↑Overconfidence Rate↓
Qwen-2.5-3BSST2-SFT 93.48 97.33 6.91
SST2-PAFT 92.45 (-1.03) 99.28 (+1.95) 7.18 (+0.27)
SST2-DGAO 93.53 (+0.05) 99.35 (+2.02) 6.76 (-0.15)
SQuAD v2-SFT 84.97 92.91 11.14
SQuAD v2-PAFT 81.59 (-3.38)93.56 (+0.65)13.77 (+2.63)
SQuAD v2-DGAO 85.63 (+0.66)93.35 (+0.44)10.22 (-0.92)
SearchQA-SFT 68.49 85.54 20.64
SearchQA-PAFT 69.62 (+1.13)90.91 (+5.37)24.10 (+3.46)
SearchQA-DGAO 71.87 (+3.38)87.81 (+2.27)19.26 (-1.38)
GSM8k-SFT 42.53 67.50 28.52
GSM8k-PAFT 33.81 (-8.72) 67.97 (+0.47) 36.99 (+8.47)
GSM8k-DGAO 42.91 (+0.38) 68.11 (+0.61) 28.41 (-0.11)
CM17k-SFT 28.33 78.02 51.24
CM17k-PAFT 18.83 (-9.50)83.20 (+5.18)64.94 (+13.70)
CM17k-DGAO 29.35 (+1.02)77.24 (-0.78)46.99 (-4.25)
Llama-3.2-3BSST2-SFT 89.47 99.08 10.26
SST2-PAFT 87.62 (-1.85) 99.08 (+0.00) 11.87 (+1.61)
SST2-DGAO 90.64 (+1.17) 99.14 (+0.06) 8.80 (-1.46)
SQuAD v2-SFT 73.73 88.98 18.54
SQuAD v2-PAFT 58.11 (-15.62) 82.62 (-6.36) 29.19 (+10.65)
SQuAD v2-DGAO 75.44 (+1.71) 89.60 (+0.62) 17.57 (-0.97)
SearchQA-SFT 58.72 81.21 25.88
SearchQA-PAFT 56.34 (-2.38)86.45 (+5.24)33.49 (+7.61)
SearchQA-DGAO 62.51 (+3.79)84.72 (+3.51)25.43 (-0.45)
GSM8k-SFT 22.85 56.78 37.56
GSM8k-PAFT 14.36 (-8.49) 58.00 (+1.22) 46.87 (+9.31)
GSM8k-DGAO 23.19 (+0.34) 58.47 (+1.69) 34.72 (-2.84)
CM17k-SFT 15.08 70.45 56.80
CM17k-PAFT 10.00 (-5.08)77.12 (+6.67)67.96 (+11.16)
CM17k-DGAO 15.19 (+0.11)70.61 (+0.16)51.14 (-5.66)
Gemma-3-4BSST2-SFT 89.98 98.959.27
SST2-PAFT 88.64 (-1.34)99.66 (+0.71)11.25 (+1.98)
SST2-DGAO 90.11 (+0.13)99.03 (+0.08) 9.48 (+0.21)
SQuAD v2-SFT 74.09 85.31 16.29
SQuAD v2-PAFT 63.92 (-10.17) 84.58 (-0.73) 24.80 (+8.51)
SQuAD v2-DGAO 75.82 (+1.73) 85.49 (+0.18) 14.80 (-1.49)
SearchQA-SFT 59.72 78.09 23.00
SearchQA-PAFT 58.64 (-1.08)83.82 (+5.73)29.74 (+6.74)
SearchQA-DGAO 63.10 (+3.38)80.60 (+2.51)22.11 (-0.89)
GSM8k-SFT 15.23 44.39 31.78
GSM8k-PAFT 12.17 (-3.06)48.27 (+3.88)38.66 (+6.88)
GSM8k-DGAO 15.59 (+0.36)44.43 (+0.04)31.43 (-0.35)
CM17k-SFT 11.39 59.32 49.26
CM17k-PAFT 9.54 (-1.85)70.18 (+10.86)52.84 (+3.58)
CM17k-DGAO 11.78 (+0.39)59.57 (+0.25)47.12 (-2.14)
Table 2: Accuracy and order stability metrics of LLMs
with different training methods. +/− indicates improve-
ment or degradation relative to the SFT baseline. ↑indi-
cates that higher values of this metric represent better
model accuracy or stronger order stability, ↓indicates
that lower values of this metric represent lower model
hallucination.
value represents the maximum consistent error
rate of the model’s answers, with larger values
indicating stronger model hallucination, as they are
overconfident and tend to output certain incorrect
answers.
Baselines. The main baselines for our
method are SFT and Position Aware Fine Tun-
ing (PAFT) (Zhang et al., 2024b). SFT refers to
standard fine-tuning using original training sam-
ples. PAFT refers to fine-tuning using datasets
augmented with 8 order variants. Additionally, al-
though orthogonal to our work, we also compare
other baselines that mitigate order sensitivity. Since
our method does not change the model architecture,
we mainly compare the following methods:
• Prompt: Adding "please do not consider the
order of text when answering" in the query.
• Lost in the middle (Liu et al., 2023): Placing
gold document at beginning or end of the context.• PINE (Wang et al., 2024b): Modify attention
mechanism and reassign positional information.
• SPHS (Yu et al., 2024a): Adjust the activation
values of position-related hidden states.
• GlobalE (Lu et al., 2021) and DEmO (Guo
et al., 2024): Methods for searching optimal or
suboptimal orders. Although they are orthogonal to
our method, we still include them for comparison.
Additionally, since we introduce a new reinforce-
ment learning method, we also compare the effects
of traditional reinforcement learning methods, in-
cluding PPO (Ouyang et al., 2022) and GRPO (Guo
et al., 2025).
Implementation Details. We evaluate 3 models:
Qwen-2.5-3B (Team, 2024), Llama-3.2-3B (Meta,
2024), Gemma-3-4B (Kamath et al., 2025). For
SFT and PAFT, all models use 2K and 16K data
of each task respectively for full-parameter fine-
tuning. For our DGAO, we first use half of the 16K
augmented data for full-parameter SFT to cold-
start the model, then use the other half of the data
for DGAO reinforcement fine-tuning. αandβ
are set to 0.5 and 0.05, respectively with detailed
parameter settings in Appendix B.
4.2 Main Results
Table 2 shows the accuracy, Consistency Rate,
and Overconfidence Rate metrics of LLMs trained
with SFT, PAFT, and DGAO. Compared to SFT,
although only variants with different orders are
used and no additional data is introduced, DGAO-
trained models improve accuracy and Consistency
Rate on almost all tasks while reducing Overcon-
fidence Rate. While PAFT using the same train-
ing data improves Consistency Rate across nearly
all tasks, it reduces accuracy and increases Over-
confidence Rate. This indicates that PAFT intro-
duces more hallucinations to maintain answer con-
sistency, thereby harming the responses. Although
DGAO does not always lead PAFT in Consistency
Rate, it performs better in both accuracy and Over-
confidence Rate, making it more reliable in practi-
cal applications. Additionally, experimental results
on the scalability of DGAO on larger parameter
LLMs can be found in Appendix D.
Table 3 compares the evaluation results of other
methods with those of DGAO-trained models. The
accuracy and order stability metrics of Prompt are
almost on par with SFT, indicating that naively
prompting the model not to focus on context order
cannot solve order bias. Pos-Beginning and Pos-
End are two variants of "Lost in middle," which

MethodSST2 SQuAD SearchQA GSM8K CM17K
Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓
SFT 89.47 99.08 10.26 73.73 88.98 18.54 58.72 81.21 25.88 22.85 56.78 37.56 15.08 70.45 56.80
Prompt 89.28 (-0.19) 99.03 (-0.05) 10.52 (+0.26) 73.37 (-0.36) 89.14 (+0.16) 18.90 (+0.36) 58.35 (-0.37) 81.08 (-0.13) 26.21 (+0.33) 22.99 (+0.14) 56.59 (-0.19) 37.81 (+0.25) 14.93 (-0.15) 70.68 (+0.23) 57.03 (+0.23)
Pos-Beginning - - - 75.21 (+1.48)92.35 (+3.37) 15.85 (-2.69) 62.49 (+3.77)87.99 (+6.78)26.24 (+0.36) - - - - - -
Pos-End - - - 74.99 (+1.26) 90.49 (+1.51) 17.65 (-0.89) 62.24 (+3.52) 87.41 (+6.20) 26.52 (+0.64) - - - - - -
PINE 84.15 (-5.32) 91.51 (-7.57) 19.86 (+9.60) 66.91 (-6.82) 81.75 (-7.23) 18.42 (-0.12) 52.23 (-6.49) 75.31 (-5.90) 26.09 (+0.21) 15.49 (-7.36) 49.88 (-6.90) 38.19 (+0.63) 7.92 (-7.16) 62.25 (-8.20) 56.88 (+0.08)
SPHS 58.85 (-30.62) 72.27 (-26.81) 23.85 (+13.59) 73.59 (-0.14) 89.10 (+0.12) 18.71 (+0.17) 47.78 (-10.94) 67.65 (-13.56) 26.22 (+0.34) 22.48 (-0.37) 56.12 (-0.66) 37.38 (-0.18) 14.99 (-0.09) 70.36 (-0.09) 57.17 (+0.37)
GlobalE 89.31 (-0.16) 99.01 (-0.07) 10.49 (+0.23) 73.81 (+0.08) 89.15 (+0.17) 18.30 (-0.24) 58.88 (+0.16) 81.19 (-0.02) 25.79 (-0.09) 22.89 (+0.04) 56.85 (+0.07) 37.51 (-0.05) 14.88 (-0.20) 70.49 (+0.04) 57.06 (+0.26)
DEmO 89.52 (+0.05) 99.11 (+0.03) 10.12 (-0.14) 74.34 (+0.61) 89.25 (+0.27) 18.25 (-0.29) 59.26 (+0.54) 81.93 (+0.72) 25.58 (-0.30) 22.95 (+0.10) 56.91 (+0.13) 37.45 (-0.11) 15.15 (+0.07) 70.51 (+0.06) 56.71 (-0.09)
DGAO 90.64 (+1.17) 99.14 (+0.06) 8.80 (-1.46) 75.44 (+1.71)89.60 (+0.62) 17.57 (-0.97) 62.51 (+3.79)84.72 (+3.51)25.43 (-0.45) 23.19 (+0.34) 58.47 (+1.69) 34.72 (-2.84) 15.19 (+0.11) 70.61 (+0.16) 51.14 (-5.66)
Table 3: Performance comparison of different methods on Llama-3.2-3B. CR represents Consistency Rate, OR
represents Overconfidence Rate.
MethodSST2 SQuAD SearchQA GSM8K CM17K
Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓
SFT 89.47 99.08 10.26 73.73 88.98 18.54 58.72 81.21 25.88 22.85 56.78 37.56 15.08 70.45 56.80
w/oA group 90.29 (+0.82) 98.83 (-0.25) 9.39 (-0.87) 75.41 (+1.68) 89.16 (+0.18)17.45 (-1.09) 61.98 (+3.26) 80.30 (-0.91) 25.51 (-0.37) 23.09 (+0.24) 56.27 (-0.51) 37.64 (+0.08) 15.25 (+0.17)70.27 (-0.18) 56.31 (-0.49)
w/oA ind 89.44 (-0.03) 99.04 (-0.04) 10.08 (-0.18) 73.92 (+0.19) 88.56 (-0.42) 18.32 (-0.22) 57.82 (-0.90) 80.39 (-0.82) 26.19 (+0.31) 21.42 (-1.43) 55.50 (-1.28) 37.43 (-0.13) 14.27 (-0.81) 62.19 (-8.26)49.74 (-7.06)
DGAO 90.64 (+1.17) 99.14 (+0.06) 8.80 (-1.46) 75.44 (+1.71) 89.60 (+0.62)17.57 (-0.97) 62.51 (+3.79) 84.72 (+3.51) 25.43 (-0.45) 23.19 (+0.34) 58.47 (+1.69) 34.72 (-2.84) 15.19 (+0.11)70.61 (+0.16)51.14 (-5.66)
Table 4: Ablation study results of DGAO (using Llama-3.2-3B as the base model).
MethodSQuAD GSM8K
Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓
SFT 73.73 88.98 18.54 22.85 56.78 37.56
PPO 73.02 (-0.71) 86.90 (-2.08) 22.28 (+3.74) 22.99 (+0.14) 56.21 (-0.57) 37.66 (+0.10)
GRPO 73.11 (-0.62) 88.15 (-0.83) 22.49 (+3.95) 22.92 (+0.07) 55.70 (-1.08) 37.59 (+0.03)
DGAO 75.44 (+1.71) 89.60 (+0.62) 17.57 (-0.97) 23.19 (+0.34) 58.47 (+1.69) 34.72 (-2.84)
Table 5: Performance comparison of different reinforce-
ment learning methods on Llama-3.2-3B.
place the most relevant documents in RAG tasks
at the beginning or end of the context, respectively.
They serve as strong baselines because they are
equivalent to transforming the test set into a form
more suitable for the model to answer. Therefore,
they achieve better accuracy and order stability in
RAG tasks, indicating that Llama-3.2-3B indeed
pays more attention to the beginning and end of
the context. However, in RAG systems that lack
ranking or in-context learning tasks (such as SST2,
GSM8K, CM17K), it is difficult to define which
prompt is the key prompt, making these methods
inapplicable. PINE and SPHS lag behind DGAO
by 8.05% and 9.87% in accuracy, respectively, and
by 8.37% and 9.42% in Consistency Rate. It is
worth noting that DGAO is actually orthogonal to
the other methods in Table 3, as it neither changes
the model structure nor improves the decoding strat-
egy. Therefore, these methods can be simply im-
plemented on DGAO-trained models, potentially
bringing further improvements in accuracy and or-
der fairness. We leave this part of the research for
future work.
Table 5 shows results of models trained with dif-
ferent reinforcement learning methods. Both PPO
and GRPO reduce the Consistency Rate, indicates
that traditional reinforcement learning methods can-
not enhance the model’s order stability. DGAO
not only improves order stability but also slightly
outperforms PPO and GRPO in accuracy improve-
ment, leading to simultaneous improvements in
both model accuracy and order fairness.
(a) Generalization performance of DGAO-
trained models on other tasks.(b) Generalization performance of DGAO-
trained models on 32-shot tasks.
Figure 4: DGAO’s Order Stability Generalization (using
Llama-3.2-3B as base model).
4.3 Ablation Experiments
Table 4 shows the performance of models trained
with DGAO without inter-group relative advantage
Agroup (α= 0 ) or without intra-group relative ad-
vantage Aind(α= 1 ). The results indicate that
without Agroup, there are general improvements in
accuracy and decreases in Overconfidence Rate,
showing that the model indeed rewards more ac-
curate answers through intra-group relative advan-
tage, but fails to improve Consistency Rate, thus
not alleviating order sensitivity. Without Aind, the
model performs poorly on almost all metrics, indi-
cating that merely rewarding inter-group relative
advantage cannot effectively help the model learn.
4.4 Generalization Capability
We train DGAO on the SQuAD v2 and CM17K
datasets respectively and evaluate on the SST2,
SearchQA and GSM8K test sets respectively, with
results shown in Figure 4(a). Since DGAO has not
been trained on the tasks corresponding to the test
sets, the model tends to be less confident when an-
swering questions from the test sets, making direct
comparison of Consistency Rate with SFT models
unfair. Therefore, we choose to compare the ratio
of Consistency Rate to Accuracy, which represents
the proportion of order-stable outputs the model
achieves under a fair accuracy baseline. The re-
sults show that compared to naive SFT on the train-
ing sets corresponding to the test datasets, DGAO-
trained models also achieve improved order stabil-

ity on other datasets, indicating that DGAO’s order
stability can generalize to tasks beyond the training
tasks. Additionally, we also evaluate whether the
model still has generalizable order stability under
longer contexts. As shown in Figure 4(b), we test
the performance of models trained with different
methods on the SQuAD v2 dataset containing 32
documents in the context. The results show that
the DGAO-trained model demonstrates consistent
accuracy and order stability. Compared to PAFT,
DGAO improves accuracy by 26.41% and Con-
sistency Rate by 16.68%, indicating that DGAO’s
order stability can generalize to tasks with more
contextual data than the training data. More experi-
ments on datasets and longer context can be found
in Appendix C.
5 Acknowledgment
This work is supported by the National Key R&D
Program of China [2023YFC3304903].
6 Conclusion
In this paper, we propose DGAO that simultane-
ously improves model accuracy and order stability.
Extensive experiments on multiple datasets demon-
strate that DGAO can make models more inclined
toward order fairness.
7 Limitations
In our experiments, due to computational cost
constraints, we primarily conduct experiments on
LLMs with 3B-14B parameters. In future work,
we will further validate the scalability of DGAO on
models with larger parameter counts. Additionally,
in our experiments, DGAO uses a fixed number of
random order variants. While this choice balances
computational efficiency and coverage, for con-
texts containing many elements, a limited number
of variants may not adequately cover all possible
ordering patterns. Although the number of order
variants learned by the model can be supplemented
by increasing the sampling size, training epochs,
or training sample size, this would incur additional
computational overhead and data collection costs.
In future work, we will explore adaptive sampling
strategies or difficulty-based order selection meth-
ods.References
Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet,
Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh
Anand, Zaheer Abbas, Azade Nova, et al. 2024.
Many-shot in-context learning.Advances in Neural
Information Processing Systems, 37:76930–76966.
Afra Amini, Tim Vieira, and Ryan Cotterell. 2024. Di-
rect preference optimization with an offset.arXiv
preprint arXiv:2402.10571.
Amanda Bertsch, Maor Ivgi, Emily Xiao, Uri Alon,
Jonathan Berant, Matthew R Gormley, and Graham
Neubig. 2024. In-context learning with long-context
models: An in-depth exploration.arXiv preprint
arXiv:2405.00200.
Guoxin Chen, Minpeng Liao, Chengxi Li, and Kai
Fan. 2024. Step-level value preference optimiza-
tion for mathematical reasoning.arXiv preprint
arXiv:2406.10858.
Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee,
Peng Xu, Mohammad Shoeybi, Bryan Catanzaro,
and Wei Ping. 2025. Acereason-nemotron: Advanc-
ing math and code reasoning through reinforcement
learning.arXiv preprint arXiv:2505.16400.
Xu Chu, Xinrong Chen, Guanyu Wang, Zhijie Tan, Kui
Huang, Wenyu Lv, Tong Mo, and Weiping Li. 2025a.
Qwen look again: Guiding vision-language reason-
ing models to re-attention visual information.arXiv
preprint arXiv:2505.23558.
Xu Chu, Zhijie Tan, Hanlin Xue, Guanyu Wang, Tong
Mo, and Weiping Li. 2025b. Domaino1s: Guiding
llm reasoning for explainable answers in high-stakes
domains.arXiv preprint arXiv:2501.14431.
Xu Chu, Hanlin Xue, Zhijie Tan, Bingce Wang, Tong
Mo, and Weiping Li. 2025c. Graphsos: Graph sam-
pling and order selection to help llms understand
graphs better.arXiv preprint arXiv:2501.14427.
Karl Cobbe, Vineet Kosaraju, Mo Bavarian, Mark
Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plap-
pert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano,
Christopher Hesse, and John Schulman. 2021. Train-
ing verifiers to solve math word problems.ArXiv,
abs/2110.14168.
Shengyuan Ding, Xinyu Fang, Ziyu Liu, Yuhang Zang,
Yuhang Cao, Xiangyu Zhao, Haodong Duan, Xiaoyi
Dong, Jianze Liang, Bin Wang, et al. 2025. Arm-
thinker: Reinforcing multimodal generative reward
models with agentic tool use and visual reasoning.
arXiv preprint arXiv:2512.05111.
Yingpeng Du, Di Luo, Rui Yan, Xiaopei Wang, Hongzhi
Liu, Hengshu Zhu, Yang Song, and Jie Zhang. 2024.
Enhancing job recommendation through llm-based
generative adversarial networks. InProceedings of
the AAAI conference on artificial intelligence, pages
8363–8371.

Matthew Dunn, Levent Sagun, Mike Higgins, V . Ugur
Güney, V olkan Cirik, and Kyunghyun Cho. 2017.
Searchqa: A new q&a dataset augmented with con-
text from a search engine.ArXiv, abs/1704.05179.
Beni Egressy and Jan Stühmer. 2025. Set-llm:
A permutation-invariant llm.arXiv preprint
arXiv:2505.15433.
Sagar Goyal, Eti Rastogi, Sree Prasanna Rajagopal,
Dong Yuan, Fen Zhao, Jai Chintagunta, Gautam Naik,
and Jeff Ward. 2024. Healai: A healthcare llm for
effective medical documentation. InProceedings
of the 17th ACM International Conference on Web
Search and Data Mining, pages 1167–1168.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: In-
centivizing reasoning capability in llms via reinforce-
ment learning.arXiv preprint arXiv:2501.12948.
Qi Guo, Leiyu Wang, Yidong Wang, Wei Ye, and Shikun
Zhang. 2024. What makes a good order of examples
in in-context learning. InAnnual Meeting of the
Association for Computational Linguistics.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering.Advances in Neural Information
Processing Systems, 37:132876–132907.
Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Day-
iheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang,
Bowen Yu, Keming Lu, et al. 2024. Qwen2. 5-coder
technical report.arXiv preprint arXiv:2409.12186.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Os-
trow, Akila Welihinda, Alan Hayes, Alec Radford,
et al. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Fangkai Jiao, Geyang Guo, Xingxing Zhang, Nancy F
Chen, Shafiq Joty, and Furu Wei. 2024a. Prefer-
ence optimization for reasoning with pseudo feed-
back.arXiv preprint arXiv:2411.16345.
Tong Jiao, Jian Zhang, Kui Xu, Rui Li, Xi Du, Shangqi
Wang, and Zhenbo Song. 2024b. Enhancing fairness
in llm evaluations: Unveiling and mitigating biases in
standard-answer-based evaluations. InProceedings
of the AAAI Symposium Series, 1, pages 56–59.
Gemma Team Aishwarya Kamath, Johan Ferret, Shreya
Pathak, Nino Vieillard, and et al. 2025. Gemma 3
technical report.ArXiv, abs/2503.19786.
Bolun Li, Zhihong Sun, Tao Huang, Hongyu Zhang,
Yao Wan, Ge Li, Zhi Jin, and Chen Lyu. 2024a.
Ircoco: Immediate rewards-guided deep reinforce-
ment learning for code completion.ArXiv,
abs/2401.16637.Yuchen Li, Hengyi Cai, Rui Kong, Xinran Chen, Jiamin
Chen, Jun Yang, Haojie Zhang, Jiayi Li, Jiayi Wu,
Yiqun Chen, et al. 2025. Towards ai search paradigm.
arXiv preprint arXiv:2506.17188.
Zongjie Li, Chaozheng Wang, Pingchuan Ma, Daoyuan
Wu, Shuai Wang, Cuiyun Gao, and Yang Liu. 2024b.
Split and merge: Aligning position biases in llm-
based evaluators. InProceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing, pages 11084–11108.
Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli
Feng, Yinwei Wei, and Tat-Seng Chua. 2024. Data-
efficient fine-tuning for llm-based recommendation.
InProceedings of the 47th international ACM SIGIR
conference on research and development in informa-
tion retrieval, pages 365–374.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2023. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel,
and Pontus Stenetorp. 2021. Fantastically ordered
prompts and where to find them: Overcoming few-
shot prompt order sensitivity. InAnnual Meeting of
the Association for Computational Linguistics.
Trung Quoc Luong, Xinbo Zhang, Zhanming Jie,
Peng Sun, Xiaoran Jin, and Hang Li. 2024. Reft:
Reasoning with reinforced fine-tuning.ArXiv,
abs/2401.08967.
Meta. 2024. Llama 3.2 model cards and prompt
formats. https://www.llama.com/docs/
model-cards-and-prompt-formats/llama3_2/ .
Accessed: 2025-07-21.
Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou,
Yixin Mao, and Chien-Sheng Wu. 2025. Search en-
gines in the ai era: A qualitative understanding to the
false promise of factual and verifiable source-cited
responses in llm-based search. InProceedings of the
2025 ACM Conference on Fairness, Accountability,
and Transparency, pages 1325–1340.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll Wainwright, Pamela Mishkin, Chong Zhang,
Sandhini Agarwal, Katarina Slama, Alex Ray, et al.
2022. Training language models to follow instruc-
tions with human feedback.Advances in neural in-
formation processing systems, 35:27730–27744.
Jinghui Qin, Xiaodan Liang, Yining Hong, Jianheng
Tang, and Liang Lin. 2021. Neural-symbolic solver
for math word problems with auxiliary tasks.arXiv
preprint arXiv:2107.01431.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.

Pranav Rajpurkar, Robin Jia, and Percy Liang. 2018.
Know what you don’t know: Unanswerable questions
for squad.arXiv preprint arXiv:1806.03822.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Proxi-
mal policy optimization algorithms.arXiv preprint
arXiv:1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath:
Pushing the limits of mathematical reasoning in open
language models.arXiv preprint arXiv:2402.03300.
Richard Socher, Alex Perelygin, Jean Wu, Jason
Chuang, Christopher D Manning, Andrew Y Ng, and
Christopher Potts. 2013. Recursive deep models for
semantic compositionality over a sentiment treebank.
InProceedings of the 2013 conference on empiri-
cal methods in natural language processing, pages
1631–1642.
Taylor Sorensen, Joshua Robinson, Christopher Michael
Rytting, Alexander Glenn Shaw, Kyle Jeffrey
Rogers, Alexia Pauline Delorey, Mahmoud Khalil,
Nancy Fulda, and David Wingate. 2022. An
information-theoretic approach to prompt engineer-
ing without ground truth labels.arXiv preprint
arXiv:2203.11364.
Zhijie Tan, Xu Chu, Weiping Li, and Tong Mo. 2024a.
Order matters: Exploring order sensitivity in mul-
timodal large language models.arXiv preprint
arXiv:2410.16983.
Zhijie Tan, Yuzhi Li, Xiang Yuan, Shengwei Meng,
Weiping Li, and Tong Mo. 2024b. Llm-mhr: A llm-
augmented multimodal hashtag recommendation al-
gorithm. In2024 IEEE International Conference on
Web Services (ICWS), pages 810–821. IEEE.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang,
Chi Han, Shuiwang Ji, Sham M. Kakade, Hao Peng,
and Heng Ji. 2024a. Eliminating position bias of
language models: A mechanistic approach.ArXiv,
abs/2407.01100.
Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang,
Chi Han, Shuiwang Ji, Sham M Kakade, Hao Peng,
and Heng Ji. 2024b. Eliminating position bias of
language models: A mechanistic approach.arXiv
preprint arXiv:2407.01100.
Sheng-Lun Wei, Cheng-Kuang Wu, Hen-Hsen Huang,
and Hsin-Hsi Chen. 2024. Unveiling selection bi-
ases: Exploring order and token sensitivity in large
language models.arXiv preprint arXiv:2406.03009.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, et al. 2025. Qwen3
technical report.arXiv preprint arXiv:2505.09388.An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao,
Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong
Tu, Jingren Zhou, Junyang Lin, et al. 2024. Qwen2.
5-math technical report: Toward mathematical ex-
pert model via self-improvement.arXiv preprint
arXiv:2409.12122.
Jiayu Yao, Shenghua Liu, Yiwei Wang, Lingrui Mei,
Baolong Bi, Yuyao Ge, Zhecheng Li, and Xueqi
Cheng. 2025. Who is in the spotlight: The hidden
bias undermining multimodal retrieval-augmented
generation.ArXiv, abs/2506.11063.
Huaiyuan Ying, Shuo Zhang, Linyang Li, Zhejian Zhou,
Yunfan Shao, Zhaoye Fei, Yichuan Ma, Jiawei Hong,
Kuikun Liu, Ziyi Wang, et al. 2024. Internlm-math:
Open math large language models toward verifiable
reasoning.arXiv preprint arXiv:2402.06332.
Yijiong Yu, Huiqiang Jiang, Xufang Luo, Qianhui
Wu, Chin-Yew Lin, Dongsheng Li, Yuqing Yang,
Yongfeng Huang, and Lili Qiu. 2024a. Mitigate posi-
tion bias in large language models via scaling a single
dimension.arXiv preprint arXiv:2406.02536.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024b. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms.Advances in
Neural Information Processing Systems, 37:121156–
121184.
Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho,
Sainbayar Sukhbaatar, Jing Xu, and Jason Weston.
2024. Self-rewarding language models.arXiv
preprint arXiv:2401.10020, 3.
Meiru Zhang, Zaiqiao Meng, and Nigel Collier. 2024a.
Can we instruct llms to compensate for position bias?
InFindings of the Association for Computational
Linguistics: EMNLP 2024, pages 12545–12556.
Zheng Zhang, Fan Yang, Ziyan Jiang, Zheng Chen,
Zhengyang Zhao, Chengyuan Ma, Liang Zhao, and
Yang Liu. 2024b. Position-aware parameter efficient
fine-tuning approach for reducing positional bias in
llms.ArXiv, abs/2404.01430.
Hanzhang Zhou, Zijian Feng, Zixiao Zhu, Junlang Qian,
and Kezhi Mao. 2024. Unibias: Unveiling and mit-
igating llm bias through internal attention and ffn
manipulation.Advances in Neural Information Pro-
cessing Systems, 37:102173–102196.

A Theoretical Motivation
To mitigate order sensitivity, we aim to optimize the
policy πθso that it consistently generates correct
answers under different permutations. Let y∗be the
ground truth answer. We first theoretically analyze
the limitations of existing strategies in addressing
this problem. For standard supervised fine-tuning
on augmented data (PAFT (Zhang et al., 2024b)),
it minimizes the negative log-likelihood:
LPAFT(θ) =−E σ∼Σ n[logπ θ(y∗|xσ)].(7)
While this encourages the model to output y∗for
all permutations, it treats each permutation xσas
an independent sample. It does not explicitly pe-
nalize the difference between distributions πθ(·|xσ)
andπθ(·|xσ′). As observed in Table 1, forcing the
model to memorize y∗corresponding to different
ordered permutations xσwithout constraints often
leads to overfitting or catastrophic forgetting, harm-
ing general accuracy.
Similarly, standard reinforcement learning ap-
proaches maximize the expected reward:
J(θ) =E σ∼Σ nEy∼πθ(·|xσ)[R(y, y∗)].(8)
However, the traditional reward R(y, y∗)is
only based on correctness, so gradient updates
independently increase the probability of correct
samples. Similar to standard SFT and PAFT, it
likewise lacks a mechanism to explicitly reward
policy consistency across the permutation group
{qπ1, . . . , q πN}.
B Experimental Setup Details
All experiments use 8 NVIDIA A800-SXM4 80G
GPUs, with memory size of 1.5 TB, operating sys-
tem Ubuntu 18.04.6, Python version 3.10.18, torch
version 2.6.0, CUDA version 12.0, transformers
version 4.53.3.
For SFT and PAFT, we set global batch size to
8, learning rate to 2e-5, and train for 3 epochs.
For DGAO, global batch size to 8, we set learn-
ing rate to 2e-5, random order number to 8, i.e., 8
order variants of each original query are allocated
on each device. Temperature and top p sampling
are both set to 1.0. ϵAis set to 1e-8. ϵis set to 0.2.
We train DGAO for 3 epochs.
For PPO and GRPO, we set global batch size
to 8, learning rate to 2e-5, training epoch to 3,temperature and top p sampling are both set to 1.0,
ϵis set to 0.2. GRPO’s generation number is set to
8.
For the above training, the Adam optimizer pa-
rameters areβ 1= 0.9,β 2= 0.999,ϵ= 1e−8.
All evaluation results are based on the average
of 3 independent runs and use greedy decoding.
C Supplementary Generalization
Experiments
Table 6 shows the results of evaluating the model’s
generalization ability for order stability on longer
contexts. On the 32-shot and 64-shot SQuAD v2
datasets, DGAO achieves higher accuracy and or-
der stability, while PAFT reduces accuracy by an
average of 23.7%. On the SST2 dataset, although
PAFT improves the Consistency Rate, the Overcon-
fidence Rate increases. This indicates that PAFT
generates consistently incorrect answers in pur-
suit of high answer consistency. Overall, mod-
els trained with DGAO also demonstrate general-
ized order stability and accuracy improvements on
longer contexts.
Task Dataset Method Accuracy↑Consistency Rate↑Overconfidence Rate↓
32-shotSQuAD v2SFT 60.61 77.24 21.05
PAFT 37.48 (-23.13) 61.89 (-15.35) 28.70 (+7.65)
DGAO 63.89 (+3.28) 78.57 (+1.33) 19.24 (-1.81)
SST2SFT 57.10 70.77 23.87
PAFT 55.78 (-1.32)81.42 (+10.65)34.82 (+10.95)
DGAO 57.91 (+0.81)70.84 (+0.07)23.60 (-0.27)
64-shotSQuAD v2SFT 49.25 67.50 23.41
PAFT 24.99 (-24.26) 50.14 (-17.36) 29.65 (+6.24)
DGAO 52.75 (+3.50) 68.33 (+0.83) 20.89 (-2.52)
SST2SFT 56.21 69.11 23.08
PAFT 55.95 (-0.26)73.81 (+4.70)27.28 (+4.20)
DGAO 56.65 (+0.44)70.26 (+1.15)23.05 (-0.03)
Table 6: Evaluation results on longer context lengths
(using Llama-3.2-3B as the base model).
D Scalability of DGAO on Larger
Parameter LLMs
We test the effect of DGAO on larger parameter
models than those in the main experiments (Sec-
tion 4.2). As shown in Table 7, DGAO still demon-
strates consistent improvements in Accuracy and
Consistency Rate on larger parameter models, and
reduces the Overconfidence Rate, which aligns
with the results we observe in the main experiments.
This indicates that DGAO is scalable and possesses
the capability to mitigate order sensitivity on LLMs
with larger parameter counts as well.
E DGAO Algorithm Pseudocode

Algorithm 1Dual Group Advantage Optimization
(DGAO)
Require:Policy modelπ θ, reference modelπ ref, old policyπ old
Require: Batch of Mqueries, order variations Nper query, hyperparameter
α∈[0,1], Clipping parameterϵ, KL penaltyβ, small constantϵ A1:foreach training iterationdo
2:Sample Generation:
3:forj= 1toMdo
4:GenerateNdifferent order versions:{q j,1, qj,2, . . . , q j,N}
5:fori= 1toNdo
6:Generate response:y j,i∼πθ(·|qj,i)
7:Calculate reward:R(y j,i) =(
1if correct
0if incorrect
8:end for
9:end for
10:Advantage Calculation:
11:forj= 1toMdo
12:Calculate group reward:R group(j) =1
NPN
i=1R(yj,i)
13:end for
14:Calculate global baseline:b=1
MPM
j=1Rgroup(j)
15:forj= 1toMdo
16:Calculate inter-group advantage:A group(j) =R group(j)−b
17:fori= 1toNdo
18: Calculate intra-group advantage: Aind(yj,i) =R(y j,i)−
Rgroup(j)
19:Calculate hybrid advantage:
20:A hybrid(yj,i) =α·A group(j) + (1−α)·A ind(yj,i)
21:end for
22:end for
23:Advantage Normalization:
24:Calculate batch statistics:µ A=mean(A hybrid), σA=std(A hybrid)
25:forj= 1toM,i= 1toNdo
26:A final(yj,i) =Ahybrid(yj,i)−µA
σA+ϵA27:end for
28:Policy Update:
29:Calculate probability ratios:r j,i=πθ(yj,i|qj,i)
πold(yj,i|qj,i)
30:Calculate clipped objective:
31:L clip(yj,i) = min(r j,iAfinal(yj,i),clip(r j,i,1−ϵ,1 +
ϵ)A final(yj,i))
32:Calculate KL penalty:L KL=DKL(πθ||πref)
33:Calculate total loss:
34:L DGAO(θ) =1
MNPM
j=1PN
i=1Lclip(yj,i)−βL KL
35:Update policy parameters:θ←θ− ∇ θLDGAO(θ)
36:Update old policy:π old←πθ37:end for
38:returnOptimized policy modelπ θModel Dataset Accuracy↑Consistency Rate↑Overconfidence Rate↓
Qwen-2.5-14BSST2-SFT 94.15 97.05 7.52
SST2-PAFT 92.23 (-1.92) 98.88 (+1.83) 7.75 (+0.23)
SST2-DGAO 94.27 (+0.12) 98.91 (+1.86) 7.41 (-0.11)
SQuAD v2-SFT 86.92 94.90 11.16
SQuAD v2-PAFT 83.68 (-3.24)95.41 (+0.51)13.52 (+2.36)
SQuAD v2-DGAO 87.51 (+0.59)95.28 (+0.38)10.42 (-0.74)
SearchQA-SFT 67.90 87.46 20.65
SearchQA-PAFT 69.03 (+1.13)92.71 (+5.25)24.08 (+3.43)
SearchQA-DGAO 71.22 (+3.32)89.82 (+2.36)19.42 (-1.23)
GSM8k-SFT 57.90 69.44 28.57
GSM8k-PAFT 49.43 (-8.47) 69.91 (+0.47) 36.73 (+8.16)
GSM8k-DGAO 58.19 (+0.29) 70.02 (+0.58) 28.27 (-0.30)
CM17k-SFT 38.87 79.97 51.27
CM17k-PAFT 29.76 (-9.11)85.08 (+5.11)64.58 (+13.31)
CM17k-DGAO 39.75 (+0.88)81.73 (+1.76)47.36 (-3.91)
Llama-3.2-11BSST2-SFT 90.23 99.18 9.87
SST2-PAFT 88.47 (-1.76) 99.24 (+0.06) 11.43 (+1.56)
SST2-DGAO 91.31 (+1.08) 99.27 (+0.09) 8.41 (-1.46)
SQuAD v2-SFT 75.65 90.95 18.61
SQuAD v2-PAFT 60.84 (-14.81) 85.12 (-5.83) 28.76 (+10.15)
SQuAD v2-DGAO 77.28 (+1.63) 91.46 (+0.51) 17.71 (-0.90)
SearchQA-SFT 58.44 83.16 25.96
SearchQA-PAFT 56.18 (-2.26) 85.27 (+2.11) 31.28 (+5.32)
SearchQA-DGAO 62.13 (+3.69) 86.54 (+3.38) 25.52 (-0.44)
GSM8k-SFT 32.14 58.77 37.65
GSM8k-PAFT 23.96 (-8.18) 59.84 (+1.07) 46.51 (+8.86)
GSM8k-DGAO 32.49 (+0.35) 60.35 (+1.58) 35.02 (-2.63)
CM17k-SFT 21.80 72.38 56.85
CM17k-PAFT 17.03 (-4.77)75.92 (+3.54)65.62 (+8.77)
CM17k-DGAO 21.87 (+0.07)72.56 (+0.18)51.58 (-5.27)
Table 7: Accuracy and order stability metrics of LLMs
with different training methods. +/− indicates improve-
ment or degradation relative to the SFT baseline. ↑indi-
cates that higher values of this metric represent better
model accuracy or stronger order stability, ↓indicates
that lower values of this metric represent lower model
hallucination.
F HyperparameterαSweeping
Experiments
To more precisely analyze the impact of hyperpa-
rameter αon DGAO, we conduct parameter sweep-
ing experiments on SST2 and SearchQA tasks, as
shown in Table 8. It is observed that when αis in
the range of 0.4 to 0.6, both inter-group advantage
and intra-group advantage take effect, achieving
the best performance by balancing accuracy and
order consistency. When α→0 , the Consistency
Rate (CR) decreases, indicating that weakening the
intra-group advantage leads to order instability in
the model; when α→1 , accuracy (Acc) decreases
and Overconfidence Rate (OR) increases, indicat-
ing that weakening the inter-group advantage leads
to reduced accuracy and increased hallucination
rate. This demonstrates that the two types of ad-
vantage computation in DGAO are complementary
and indispensable.
G Supplementary Results
Figure 5 shows a bad case of the PAFT-trained
model evaluated on the GSM8K dataset. For 8 or-
der variants of a query, PAFT provides consistently
incorrect answers, indicating that PAFT generates
consistently wrong responses in pursuit of consis-
tency. While DGAO has 1 inconsistent generation,

There were 3 cages of 10 hamsters and 3 cages of 5 
guinea pigs, totaling 30 rodents.  \n#### 30Answer from L lama-3.2-3B-PAFT
Question: Dominick went to his team's changing room and saw half as many 
robots as helmets and half as many helmets as footballs kept there. If there were 
20 helmets, calculate the total number of items Dominick saw.\nAnswer: If 
there were 20 helmets, then 20 * 2 = 40 footballs, and 20 / 2 = 10 robots. Total 
items: 10 + 40 + 20 = 70. \n#### 70\n\nQuestion: Alice likes to count the puffs 
of clouds in the sky while she eats her lunch outside at school. On Monday she 
counts just 3 puffs of clouds. Each day after that through Friday, though, she 
sees double the number of clouds in the sky as the day before. At the end of the 
week, how many clouds will she have counted in the sky at lunch across all five 
days?\nAnswer: Alice counts 3 clouds on Monday, 3 * 2 = 6 on Tuesday, 6 * 2 
= 12 on Wednesday, 12 * 2 = 24 on Thursday, and 24 * 2 = 48 on Friday. In 
total, she sees 3 + 6 + 12 + 24 + 48 = 93 clouds.  \n#### 93\n\nQuestion: Imma 
has 3 cats. She feeds her cats twice a day with 60 grams of cat food. How many 
days will 720 grams of cat food last?\nAnswer: Imma's 3 cats eat 60 x 3 = 180 
grams per meal and 180 x 2 = 360 grams per day. Given 720 grams, the food 
lasts 720 / 360 = 2 days.  \n#### 2\n\nQuestion: A whirligig spins at five times 
the speed of a thingamabob. A whatchamacallit spins eleven times faster than a 
thingamabob. A whatchamacallit spins at 121 meters per second. How fast does 
a whirligig spin?\nAnswer: A thingamabob spins at 121 / 11 = 11 meters per 
second. A whirligig spins 5 times faster, so 11 * 5 = 55 meters per second.  \
n#### 55\n\n Question: A pet shop had 6 cages of rodents. 3 of the cages 
have 10 hamsters and the other 3 have 5 guinea pigs. How many rodents 
did the pet shop have in total?\nAnswer:Query
The pet shop had 3 cages of 10 hamsters and 3 cages of 5 
guinea pigs, totaling 30 rodents.  \n#### 30
 
There are 3 cages with 10 hamsters and 3 cages with 5 
guinea pigs, totaling 30 rodents.  \n#### 30
The shop had 3 cages of 10 hamsters (3*10=30) and 3 
cages of 5 guinea pigs (3*5=15), totaling 30 + 15 = 45 
rodents.  \n#### 45Answer from L lama-3.2-3B-DGAO
The pet shop had 3 cages of 10 hamsters each and 3 
cages of 5 guinea pigs each, totaling 3*10 + 3*5 = 35 
rodents.  \n#### 35
 
The shop had 3 cages of 10 hamsters (3*10=30) and 3 
cages of 5 guinea pigs (3*5=15), totaling 30 + 15 = 45 
rodents.  \n#### 458/8 Consistency
0/8 Correct
7/8 Consistency
7/8 Correct
 
Figure 5: Comparison examples of PAFT and DGAO test results.
SettingSST2 SearchQA
Accuracy↑CR↑OR↓Accuracy↑CR↑OR↓
α= 0.0 90.29 98.83 9.39 61.98 80.30 25.51
α= 0.1 90.46 98.90 9.15 62.12 81.50 25.48
α= 0.2 90.61 98.98 8.82 62.25 82.60 25.46
α= 0.3 90.52 99.05 9.00 62.38 82.50 25.40
α= 0.4 90.59 99.10 8.90 62.46 84.20 25.44
α= 0.5 90.64 99.14 8.80 62.51 84.72 25.43
α= 0.6 90.72 99.18 8.72 62.45 85.10 25.65
α= 0.7 90.40 99.25 9.05 61.20 86.80 25.85
α= 0.8 89.78 99.30 9.95 59.80 87.50 26.40
α= 0.9 89.80 99.00 9.80 58.50 84.10 26.90
α= 1.0 89.44 99.04 10.08 57.82 80.39 26.19
Table 8: Hyperparameter αsweeping experiments (us-
ing Llama-3.2-3B as the base model).
the overall accuracy reaches 7/8. This shows that
DGAO balances the order stability and accuracy of
generation.
Figure 6 and Figure 7 are supplementary to Fig-
ure 3 in the main text. The results indicate that all
three models are order-sensitive.
H Data Construction
Figures 8, 9, and 10 respectively show construction
examples of the SST2, SQuAD v2, and GSM8k
datasets.
I Computational Cost of DGAO
As an online reinforcement learning method,
DGAO has comparable training overhead to GRPO,
both of which are slightly higher than SFT. Tak-
ing the SearchQA task as an example, training the
Qwen-3B model for 3 epochs using DGAO, GRPO,
and SFT requires 0.85, 0.81, and 0.22 GPU hours
respectively on 8 A800 GPUs with 80G memory.
This is mainly because DGAO and GRPO require
multiple sampling rounds and advantage computa-
tion, which is unavoidable in online reinforcementlearning. Considering the capability improvements
that DGAO brings, this additional overhead is ac-
ceptable.
During inference, the model trained with DGAO
has the same inference overhead and time consump-
tion as the model after SFT, because DGAO neither
introduces additional modules nor expands the in-
put during inference. Compared to methods that
use dataset-based search to find optimal or subopti-
mal arrangement orders (Lu et al., 2021; Sorensen
et al., 2022; Guo et al., 2024), DGAO also has the
advantage of inference time.

Figure 6: Impact of input order on the performance of Qwen-2.5-3B across 8 random orders.
Figure 7: Impact of input order on the performance of Gemma-3-4B across 8 random orders.

positiveAnswerAn opinion about a film: remains a perfect wildean actor . Sentiment of the 
opinion: positive. An opinion about a film: are paper -thin , and their 
personalities undergo radical changes when it suits the script . Sentiment of the 
opinion: negative. An opinion about a film: gap . Sentiment of the opinion: 
negative. An opinion about a film: unassuming . Sentiment of the opinion: 
negative. An opinion about a film: a jumbled fantasy comedy that did not figure 
out a coherent game plan at scripting , shooting or post -production stages . . 
Sentiment of the opinion: negative. An opinion about a film: at least passably 
real . Sentiment of the opinion: positive. An opinion about a film: the pitfalls . 
Sentiment of the opinion: positive. An opinion about a film: a grim future that 
does n't come close to the level of intelligence and visual splendor that can be 
seen in other films . Sentiment of the opinion: negative. An opinion about a 
film: great team . Sentiment of the opinion: positive. An opinion about a film: 
this in -depth study of important developments of the computer industry should 
make it required viewing in university computer science departments for years 
to come . . Sentiment of the opinion: positive. An opinion about a film: hard to 
imagine having more fun watching a documentary . Sentiment of the opinion: 
positive. An opinion about a film: bully . Sentiment of the opinion: negative. 
An opinion about a film: this is n't worth sitting through . Sentiment of the 
opinion: negative. An opinion about a film: as such the film has a difficult time 
shaking its blair witch project real -time roots . Sentiment of the opinion: 
negative. An opinion about a film: the bad sound , the lack of climax and , 
worst of all , . Sentiment of the opinion: negative. An opinion about a film: a 
sensitive , cultivated treatment . Sentiment of the opinion: positive. An opinion 
about a film: restoring the luster of the national lampoon film franchise , too 
long reduced to direct -to-video irrelevancy . Sentiment of the opinion:QueryFigure 8: Example of SST2 dataset.
The Great Boston Fire of 1872AnswerGiven you a series of contexts, you should answer the question after reading 
them. doc 1: Reclamation projects significantly expanded Boston throughout 
the 19th century, creating areas like the South End and the Financial District. 
Notably, after The Great Boston Fire of 1872, building rubble was used as 
landfill. Boston annexed various towns between 1804 and 1912, but attempts to 
annex Brookline, Cambridge, and Chelsea failed.\ndoc 2: In order to reduce 
negative impacts, it is desirable that pesticides be degradable or at least quickly 
deactivated in the environment. The loss of activity or toxicity of pesticides is 
due to innate chemical properties and environmental processes. For example, 
halogens within a chemical structure often slow down degradation in an aerobic 
environment.\ndoc 3: As of the 2006 Census, St. John's had 100,646 
inhabitants, with 151,322 in the urban area and 181,113 in the St. John's Census 
Metropolitan Area (CMA). St. John's is Newfoundland and Labrador's largest 
city and Canada's 20th largest CMA, including 12 other communities like 
Mount Pearl and Conception Bay South.\ndoc 4: Around the 1900s, modern 
Humanist ethical philosophies were developed from earlier secular traditions, 
affirming the dignity and worth of all people by appealing to universal human 
qualities, particularly rationality, without resorting to the supernatural. These 
ideas faced challenges from thinkers like Karl Marx and Nietzsche, who 
critiqued human rights.\ndoc 5: Minority leaders are typically energetic and 
aggressive campaigners for partisan incumbents and challengers. There is 
hardly any major aspect of campaigning that does not engage their attention. 
They assist in recruiting candidates, raise and distribute funds, persuade 
colleagues not to retire, coordinate with party committees, and encourage 
outside group support.\nQuestion: What famous disaster happened in Boston in 
1872?Query
Figure 9: Example of SQuAD v2 dataset.
The oven is off by 4% since 468 - 450 = 18 and 18 / 450 * 100 = 4%. If X + 
0.04X = 520, then 1.04X = 520, so X = 520 / 1.04 = 500.  \n#### 500AnswerQuestion: Adam has $100 and wants to spend it to open a rock stand. He can 
buy rocks for $5 each and sell them for $7 each. If he invests all his money in 
the rock stand but only sells 60% of his inventory, how much money does he 
lose?\nAnswer: Adam buys 20 rocks (100 / 5 = 20), sells 12 (20 * 0.6 = 12) for 
$84 (12 * 7 = 84), losing $16 (100 - 84 = 16).  \n#### 16\n\nQuestion: 
Heather's razors come 4 to a pack and cost $4.00 a pack.  They are currently on 
sale for buy one get one free.  She also has a $2.00 coupon.  How much will 
each individual razor cost, in cents, after the discount if she buys 2 packs of 
razors?\nAnswer: Heather buys 2 packs of razors for $4 but pays $2 after her $2 
coupon. Each pack has 4 razors, totaling 8 razors. Therefore, the cost per razor 
is $2/8 = $0.25 or 25 cents per razor.  \n#### 25\n\nQuestion: Romeo boards a 
train with 120 people. At the first stop, 20 more people board the train. At the 
second stop, 50 people descended from the train while twice that number 
boarded the train. If 80 more people descended at the third station, calculate the 
total number of people on the train at the final stop.\nAnswer: 120 + 20 = 140 
at the first stop. 140 - 50 + 100 = 190 at the second stop. 190 - 80 = 110 at the 
final stop.  \n#### 110\n\nQuestion: Amber, Micah, and Ahito ran 52 miles in 
total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many 
miles did Ahito run?\nAnswer: Amber ran 8 miles, Micah ran 3.5 * 8 = 28 
miles, leaving 52 - 8 - 28 = 16 miles for Ahito.  \n#### 16\n\nQuestion: 
Maggie's oven is malfunctioning. When she sets it to 450 the actual temperature 
is 468. If it's off by the same percentage for any recipe, what temperature 
should she set it at if her recipe calls for 520 degrees?\nAnswer:QueryFigure 10: Example of GSM8K dataset.