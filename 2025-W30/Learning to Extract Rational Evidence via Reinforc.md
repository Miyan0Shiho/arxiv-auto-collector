# Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation

**Authors**: Xinping Zhao, Shouzheng Huang, Yan Zhong, Xinshuo Hu, Meishan Zhang, Baotian Hu, Min Zhang

**Published**: 2025-07-21 13:03:55

**PDF URL**: [http://arxiv.org/pdf/2507.15586v2](http://arxiv.org/pdf/2507.15586v2)

## Abstract
Retrieval-Augmented Generation (RAG) effectively improves the accuracy of
Large Language Models (LLMs). However, retrieval noises significantly impact
the quality of LLMs' generation, necessitating the development of denoising
mechanisms. Previous methods extract evidence straightforwardly without
explicit thinking, which risks filtering out key clues and struggles with
generalization. To this end, we propose LEAR, which learns to extract rational
evidence by (1) explicitly reasoning to identify potential cues within
retrieval contents first, and then (2) consciously extracting to avoid omitting
any key cues helpful for answering questions. Specifically, we frame evidence
reasoning and evidence extraction into one unified response for end-to-end
training; apply knowledge token masks for disentanglement to derive
reasoning-based and extraction-based answers; and devise three types of
verifiable reward functions, including answer, length, and format, to update
the model via the policy optimization algorithm. Extensive experiments on three
benchmark datasets show the effectiveness of LEAR, providing compact and
high-quality evidence, improving the accuracy of downstream tasks, and
promoting effective application in online RAG systems.

## Full Text


<!-- PDF content starts -->

Learning to Extract Rational Evidence via Reinforcement Learning
for Retrieval-Augmented Generation
Xinping Zhao1, Shouzheng Huang1, Yan Zhong2, Xinshuo Hu,
Meishan Zhang1, Baotian Hu1B, Min Zhang1
1Harbin Institute of Technology (Shenzhen),2Peking University
{zhaoxinping, 210110129}@stu.hit.edu.cn ,
zhongyan@stu.pku.edu.cn ,yanshek.woo@gmail.com ,
mason.zms@gmail.com ,{hubaotian, zhangmin2021}@hit.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) ef-
fectively improves the accuracy of Large
Language Models (LLMs). However, re-
trieval noises significantly impact the qual-
ity of LLMs’ generation, necessitating
the development of denoising mechanisms.
Previous methods extract evidence straight-
forwardly without explicit thinking, which
risks filtering out key clues and struggles
with generalization. To this end, we pro-
pose LEAR , which learns to extract ratio-
nal evidence by (1)explicitly reasoning to
identify potential cues within retrieval con-
tents first, and then (2)consciously extract-
ing to avoid omitting any key cues help-
ful for answering questions. Specifically,
we frame evidence reasoning and extrac-
tion into one unified response for end-to-end
training; apply knowledge token masks for
disentanglement to derive reasoning-based
and extraction-based answers; and devise
three types of verifiable reward functions,
including answer ,length , and format , to
update the model via the policy optimiza-
tion algorithm. Extensive experiments on
three benchmark datasets show the effec-
tiveness of LEAR , providing compact and
high-quality evidence, improving the accu-
racy of downstream tasks, and promoting ef-
fective application in online RAG systems1.
1 Introduction
Retrieval-Augmented Generation (RAG) prevails
in Large Language Models (LLMs). It has
been shown highly effective for many knowledge-
intensive tasks (Lewis et al., 2020; Wu et al.,
2022), such as open-domain question answer-
ing (Shi et al., 2024; Trivedi et al., 2023), fact-
checking (Du et al., 2023; Zhao et al., 2024b), and
BCorresponding author.
1The code, data, and models will be available at https:
//github.com/HITsz-TMG/LEAR .
Figure 1: Motivating example, where key clues are
marked in green :1The key clues are filtered out,
and LLMs answer incorrectly; 2The key clues are ex-
tracted successfully, guided by the evidence reasoning.
dialog generation (Izacard et al., 2023; Thoppilan
et al., 2022), to produce more faithful and reli-
able outputs. In ideal conditions, LLMs should be
grounded on purely relevant retrieval contents to
generate accurate output and also facilitate infer-
ence. However, due to imperfect retrieval systems
or noisy data in the retrieval corpus (Wang et al.,
2023; Wei et al., 2024; Zhao et al., 2024a), the re-
trieval contents usually contain lots of irrelevant
or noisy snippets, which distract LLMs’ attention
and also inflict a heavy blow on the generation
quality of LLMs. Therefore, it is necessary and
valuable to extract evidence as well as filter out
noise for RAG to achieve superior performance.
Recently, several studies have attempted to ad-
dress this issue. They can be mainly divided into
two categories: (1)Reranking more relevant pas-
sages go up to the top of the retrieval list (Hwang
et al., 2024b; Mortaheb et al., 2025; Mao et al.,
2021); and (2)Summarizing retrieval contents into
a coherent narrative with higher signal to noise ra-
tio (Wang et al., 2023; Zhao et al., 2024a; Zhu
et al., 2024; Xu et al., 2024). The former heav-
ily relies on the performance of chosen rerankers
and may disrupt context after reranking. It usu-
ally performs worse than the latter due to a lack of
contextual understanding. The latter aims to train
LLMs as a filtering model through fine-tuningarXiv:2507.15586v2  [cs.CL]  23 Jul 2025

(Wang et al., 2023) or preference optimization
(Zhao et al., 2024a). They typically create the
training data using hand-crafted strategies, e.g.,
String Inclusion that measures whether the golden
answer is included in the candidate passage, and
Lexical Overlap, calculating the unigram overlap
between the golden answer and candidate passage.
Despite effectiveness, existing methods re-rank
or summarize retrieval contents straightforwardly
without explicit thinking, which risks filtering out
key clues due to a lack of a deep understanding of
retrieved contents. Figure 1 shows an example of
the evidence extraction (summarization) for ques-
tion answering: 1The extracted evidence fails to
maintain key clues and results in a wrong answer;
and 2The evidence reasoning guides the follow-
ing evidence extraction, e.g., “Passage 2 is rele-
vant and mentions...... ”, where key clues are suc-
cessfully maintained in the extracted evidence and
thus LLMs answer correctly. This motivates us
to develop a rational evidence extractor for RAG,
which deeply reasons and then consciously ex-
tracts as shown in the bottom layer of Figure 1.
In this work, we propose LEAR , which LEA rns
to extract R ational evidence via reinforcement
learning for RAG. Specifically, LEAR frames ev-
idence reasoning (enclosed within <reason> and
</reason> ) and extraction (enclosed within <ex-
tract> and </extract> ) into one response for
end-to-end training and generates multiple re-
sponses to collect trajectories that contain good
and bad experiences from the policy. After that,
LEAR applies two knowledge token masks on
each response to disentangle evidence reasoning
and extraction for deriving reasoning-based and
extraction-based answers (enclosed within <an-
swer> and</answer> tags) to assess their qual-
ity respectively. Finally, we devise three types
of rule-based verifiable reward functions, includ-
inganswer ,length , and format , to guide model
optimization via Group Relative Policy Optimiza-
tion (GRPO) (Shao et al., 2024). As such, LEAR
extracts rational evidence by reasoning about re-
trieval contents and then consolidating key cues
into a concise and useful narrative, which is fed
into LLMs to generate more accurate output. Our
main contributions can be summarized as follows:
• We unveil the major issue that hinders ev-
idence extraction for RAG, i.e., lacking a
deep contextual understanding of retrieval
contents, which risks filtering out key clues.• We propose a novel rational evidence extrac-
tion learning framework, LEAR , which learns
to discern potential cues in retrieval contents
and then consciously consolidates any key
cues into a concise yet very helpful evidence.
• We conduct extensive experiments on three
benchmark datasets, where the results fully
demonstrate the superiority of LEAR in terms
of performance, generalization, efficiency, as
well as robustness against retrieval noise.
2 Preliminaries
2.1 Problem Statement
In RAG (Lewis et al., 2020; Wu et al., 2022),
LLMs are given an input query qand a set of top-
kretrieved passages P={p1, p2, ..., p k}, which
aim to generate an output oclosing to the golden
answer a, conditioned on ( q,P). In the traditional
RAG paradigm, top- kretrieved passages are di-
rectly fed into LLMs. However, retrieved passages
Poften contain irrelevant or noisy contents (Wang
et al., 2023; Zhao et al., 2024a), which consid-
erably degrade LLMs’ generation quality. Given
that, an additional evidence extraction model is
proposed to condense Pinto a concise, query-
relevant context eto improve generation quality
and speed. Formally, the paradigm of RAG with
evidence extraction can be formulated as follows:
e=ME(·|q, P), o =MG(·|q, e),(1)
where ME(·)denotes the evidence extractor; e
is the extracted evidence; MG(·)denotes the an-
swer generator, i.e.,a LLM; and ois the generated
output. Despite effectiveness, we argue that this
vanilla paradigm may risk filtering out key clues
due to a lack of deep reasoning on retrieved pas-
sages. Going beyond the previous paradigm, we
present a novel one of RAG with rational evidence
extraction , which can be formulated as follows:
e=ME(·|q, P, r )· ME(r|q, P),
o=MG(·|q, e),(2)
where we train the model in an on-policy man-
ner, meaning that the evidence extractor ME(·)
and the answer generator MG(·)are the same one
during training; rdenotes evidence reasoning. It
(r) explicitly and thoroughly identifies any cues
lying in retrieved passages ( P) to guide the ev-
idence extraction ( e) to achieve more accurate

(a) Results on NQ.
 (b) Results on HotoptQA.
Figure 2: The results w.r.t. AR on the in-distribution
NQ as well as out-of-distribution HotoptQA datasets.
Dataset Vanilla Evidence Rational Evidence Rationale
NQ 70.79% 75.24% 77.30%
HotoptQA 60.55% 67.74% 71.48%
Table 1: AR results on Vanilla Evidence, Rational Evi-
dence and Rationale w.r.t. NQ and HotpotQA datasets.
output ( o). The overall objective is to learn a pol-
icyπθ:e∼ M E(·|q, P, r )·ME(r|q, P)to extract
a rational evidence ethat maximizes the expected
utilityU(a, o)over the given query distribution D:
max
πθEa∼MG(·|q,e), e∼πθ(q,P), q∼Dh
U(a, o)i
,
s.t.|e| ≪ |P| (3)
where U(·)is a utility function ( e.g.,F1score) and
is used to measure the quality of outputs condi-
tioned on the golden answer a;|e|and|P|denote
the length of evidence and passages, respectively.
2.2 Empirical Study
We conduct an empirical study to verify the as-
sumption that evidence extractors can retain as
many key clues as possible via the paradigm of
reasoning first and then extracting. We con-
struct a synthetic instruction dataset by leverag-
ing DeepSeek-R12(DeepSeek-AI et al., 2025),
where we sample 1K instances from the training
set of Natural Question (NQ) (Kwiatkowski et al.,
2019). The output of each instance in the dataset
contains three parts: (1) <reason> evidence rea-
soning </reason> ;(2) <extract> evidence extrac-
tion </extract> ; and (3) <answer> final answer
</answer> . We filter out instances with incorrect
final answers, resulting in about 620 instances.
Then, we create two variants of the dataset, where
the output of the first one consists of “ <reason>
...</reason><extract> ...</extract> ”; that of
2https://api-docs.deepseek.com/guides/
reasoning_modelthe second one only consists of “ <extract> ...
</extract> ”. Finally, we fine-tune Qwen2.5-1.5B-
Instruct (Yang et al., 2025) on these two synthetic
instruction datasets through the LoRA (Low-Rank
Adaptation) (Hu et al., 2022), respectively, where
the LoRA Rank is set to 16; we set the number of
epochs as 3. We tested on in-distribution NQ and
out-of-distribution HotoptQA (Yang et al., 2018)
datasets, excluding instances from the test set3,
where the retrieved passages did not contain the
golden answer. For a fair comparison, we set the
maximum length of the extracted evidence as 100.
Table 1 and Figure 2 show the experimental re-
sults w.r.t. Answer Recall (AR) (Zhao et al., 2025;
Jiang et al., 2024), which measures the recall of
the golden answer string in the extracted evidence.
We denote the evidence extracted by the model
trained on the first and second datasets as “rational
evidence” (marked as M) and “vanilla evidence”
(marked as N), respectively. We use “rationale” to
denote the evidence reasoning. Taking M∧ ¬N
for example, it means the answer string is recalled
by the rational evidence but not recalled by the
vanilla one. From the results, we have the follow-
ing observations: (1)The performance with ratio-
nal evidence is consistently better than that with
vanilla one, whether on in-distribution or out-of-
distribution datasets. For example, on NQ, rational
evidence achieves the AR of 75.24%, while vanilla
one achieves that of 70.79%; (2)The performance
with rationale is best, manifesting the necessity of
reasoning first. The performance with rational ev-
idence is slightly worse than that with rationale,
indicating that a better optimization is needed to
bridge this gap; and (3)The percentage of M∧¬N
is considerably higher than that of ¬M∧N,e.g.,
14.49% vs. 7.30% on HotpotQA, fully demon-
strating the superiority of the rational evidence.
We also provide a detailed failure analysis of
vanilla evidence in Appendix B. Briefly, the fail-
ures can be attributed to “ 3I”:(1) Incompleteness,
where the evidence provides some relevant infor-
mation but lacks the key clues; (2) I rrelevance,
where the evidence provides information about
the wrong entity, event, or topic; (3) I naccuracy,
where the evidence contains incorrect information,
such as wrong dates and names. In contrast, ratio-
nal evidence is able to provide accurate and rele-
vant information as well as locate the specific en-
3Because the test set is not available for HotoptQA, we
serve the development set of HotoptQA as the test set.

Figure 3: The overall system framework of the proposed LEAR and previous vanilla paradigm. (a)Compared to
vanilla evidence, our rational evidence is more skilled in extracting key cues. (b)Existing works learn to extract
vanilla evidence via supervised fine-tuning, where the output, i.e.,oracle that contains key clues, is usually created
by hand-crafted strategies, e.g., String Inclusion and Lexical Overlap in (Wang et al., 2023). (c)The proposed
framework LEAR incentivizes rational evidence extraction capability in the extractor πθvia reinforcement learning.
tity or detail needed for the answer. Given the
above qualitative as well as quantitative analyses,
it is necessary and valuable to learn to extract ra-
tional evidence for improving RAG performance.
3 Methodology
The overall framework of LEAR is illustrated in
Figure 3. In this section, we first introduce the
acquisition of rational evidence (§3.1). Then, we
define three verifiable reward functions (§3.2). Fi-
nally, we describe the training of the extractor
through online policy optimization in detail (§3.3).
3.1 Obtaining Rational Evidence
As stated in §1, vanilla evidence risks filtering out
key clues (Wang et al., 2023; Zhao et al., 2024a).
An empirical study (§2.2) further validates our as-
sumption, where rational evidence is significantly
better than vanilla one. Thus, we aim to opti-
mize the evidence extractor ME(·)to learn to ex-
tract rational evidence for RAG, formulated as:
e∼ M E(·|q, P, r )· ME(r|q, P). Specifically, we
feed the query qand its corresponding retrieved
passages Pinto the evidence extractor ME(·)and
instruct the evidence extractor to reason first and
then extract. The evidence reasoning and extrac-
tion are enclosed within the special reason and ex-
tract tags, respectively, i.e.,<reason> ...</rea-
son> and<extract> ...</extract> . The prompt
used for rational evidence generation is provided
in Appendix A. Given the rationale and the ratio-nal evidence, a question naturally arises: How to
assess the quality of them? In Section 2.2, we as-
sess the quality of them via answer recall, a heuris-
tic metric. Analogously, previous works, taking
(Wang et al., 2023) for example, heuristically em-
ploy String Inclusion or Lexical Overlap to mea-
sure the answer recall or unigram overlap. How-
ever, we argue that these metrics are not aligned
with the ultimate goal of RAG, i.e., to generate
output as close to the golden answer as possible.
To this end, we propose assessing the quality
of the rationale and rational evidence by assess-
ing the generated outputs conditioned on them. It
is well known that causal attention mechanisms
(Radford et al., 2018; Vaswani et al., 2017) in
LLMs aggregate information from previous con-
texts. Given that, another question arises: How to
generate the output conditioned on them respec-
tively without information leakage? To achieve the
above goal, we apply two knowledge token masks
on each response to disentangle the rationale and
rational evidence: (1)The first one masks the ra-
tional evidence eto assess the quality of the ra-
tionale r;(2)In contrast, the second one masks
the retrieved passages Pand rationale rsimulta-
neously to assess the quality of the rational evi-
dence e. Here, we adopt hard masking on input
rather than soft masking on attention, because soft
masking will cause information leakage where the
hidden states have already aggregated information
from previous contexts. Thus, we prefill input

(a)y=1
1+e−x/τ.
 (b)y=xγ.
Figure 4: Illustration of length rewards in terms of dif-
ferent skewness or smoothness controlled by τandγ,
where we directly plot the basic functions for clarity.
contexts from scratch after masking and generate
three outputs conditioned on different contexts:
or=MG(·|q, P, r, Ae),
oe=MG(·|q,@@P,Ar, e),
of=MG(·|q, P, r, e ),(4)
where or,oe, and ofrepresent the outputs con-
ditioned on the rationale, rational evidence, and
full context, respectively; randedenote the ratio-
nale and rational evidence, respectively; A∗denotes
the hard knowledge token mask during input. It is
worth mentioning that we also generate the output
ofconditioned on the f ull context as introducing
ofinto optimization can facilitate convergence.
3.2 Modeling Verifiable Reward
The reward function plays a key role in Rein-
forcement Learning (RL) (Ouyang et al., 2022;
Wang et al., 2024; DeepSeek-AI et al., 2025),
which guides the optimization process towards de-
sired properties. To train LEAR via RL, we de-
sign three types of rule-based verifiable reward
functions w.r.t. three primary desired properties
for evidence extraction: (1)The correctness of
the generated outputs; (2)The comprehensiveness
of rationale and the conciseness of rational evi-
dence; and (3)The format specification of the re-
sponses. For reward modeling, we first collect a
set of P REAO -tuple < P, r, e, a, o r,e,f>, where a
PREAO -tuple e is composed of retrieved P assgaes
P, rationale r, rational e vidence e, golden a nswer
a, generated output or,e,f conditioned on different
contexts. Given that, we design three types of ver-
ifiable reward functions to assess the rational evi-
dence extractor w.r.t. the three desired properties:
•Answer Reward. It focuses on the correctness
of the final outputs enclosed within <answer>and</answer> tags. However, different down-
stream tasks of RAG ( e.g., QA, fact verifica-
tion, and dialog generation) use different met-
rics to evaluate model predictions, including Ex-
act Match (EM), F1score, and Accuracy. Given
that, a question naturally arises: How to mea-
sure the answer reward uniformly w.r.t. different
downstream tasks of RAG? Here, we employ the
unigram F1to measure the answer rewards in a
unified and balanced manner (Song et al., 2025):
Rans
∗= F 1(a, o∗),∗ ∈ { r, e, f}, (5)
where Rans
∗∈[0,1]denotes the answer reward
for the generated o∗. Ifo∗is similar to a, then
Rans
∗is close to 1.0; otherwise, it is close to 0.0.
•Length Reward. It focuses on two aspects.
The first one is the comprehensiveness of ratio-
nale, enclosed within <reason> and</reason>
tags, where the rationale usually needs to be
relatively long to identify any cues lying in re-
trieval contents. The second one is the concise-
ness of rational evidence, enclosed within <ex-
tract> and</extract> tags, where the rational
evidence usually needs to be relatively short to
speed up inference. To do so, reference systems
are needed to determine whether something is
long or short. For the rational evidence, the re-
trieved passages are commonly used as their ref-
erence system (Wang et al., 2023; Zhao et al.,
2024a; Zhu et al., 2024). On top of that, we use
rational evidence as the reference system of the
rationale. They can be formulated as follows:
Rlen
r=(1
1+e−(Lr/Le−1)/τ,Lr≥Le,
1
1+e−(1−Le/Lr)/τ,otherwise .(6)
Rlen
e=(
1.0, (1−Le
LP)≥ω,
(1−Le
LP)γ,otherwise .(7)
where Rlen
r, Rlen
e∈[0,1]denote the length re-
wards of rationale and rational evidence, respec-
tively; LP,Lr, and Ledenote the length of re-
trieved passages, the rationale, and the rational
evidence, respectively; τis a temperature co-
efficient; γ∈[0,+∞]is a hyperparameter to
control the skewness of rewards; ω∈[0,1]is a
threshold to avoid falling into a trivial solution.
From Figure 4(a), we can see that the relatively
long rationale ( x= 2) will be assigned large
rewards, whereas a shorter rationale ( x=−2)
corresponds to small rewards. On the contrary,

as shown in Figure 4(b), shorter rational evi-
dence ( x= 0.8) will be assigned large rewards.
By adjusting τandγ, we can control the skew-
ness or smoothness of length rewards, as shown
in the line charts of different colors in Figure 4.
•Format Reward. It focuses on whether the re-
sponse obeys the defined format. Specifically,
the model’s evidence reasoning, evidence ex-
traction, and final output should be enclosed
within the <reason> ...</reason> ,<extract> ...
</extract> , and <answer> ...</answer> tags,
respectively. Based on the above format require-
ment, the format reward is defined as follows:
Rfmt=(
1.0,if correct format ,
0.0,if incorrect format .(8)
Putting them together. Having obtained the an-
swer, length, and format rewards, we compute the
final reward via a linear weighted sum, as complex
weighting is not the focus of this work, and a linear
one generally leads to satisfactory performance:
Rfinal=αaRans+αlRlen+αfRfmt,(9)
where Ransis the average answer reward of Rans
r,
Rans
eandRans
f;Rfmtis the average length one of
Rfmt
randRfmt
e;αa,αl, andαfare hyperparame-
ters;Rfinalis the final reward used in RL training.
3.3 Online Policy Optimization
Having obtained the final rewards of each re-
sponse, we use the policy optimization algorithm
GRPO4(Shao et al., 2024; DeepSeek-AI et al.,
2025) to optimize LEAR , incentivizing rational
evidence extraction capability in it via reinforce-
ment learning. Specifically, for each input ques-
tionq, GRPO first samples a group of responses
Y={y1, y2, ..., y G}, where Gis the group size
and each response consists of the rationale r, ra-
tional evidence e, and three outputs or,oe, and of
(§3.1). Subsequently, GRPO evaluates these re-
sponses based on the verifiable reward functions
(§3.2) and obtains final rewards for each response
R={R1, R2, ..., R G}, where we omit the su-
perscript “ final ” for brevity. Different from PPO
(Schulman et al., 2017), GRPO directly compares
the final rewards of candidate responses within the
4Here, we adopt GRPO instead of Proximal Policy Opti-
mization (PPO) (Schulman et al., 2017), because PPO needs
an additional critic model to evaluate policy performance.same group without needing an additional critic
model. The advantage of the i-th response is de-
termined through normalizing its reward Riusing
the mean and the standard deviation of rewards R:
Ai=Ri−mean( R)
std(R), (10)
where mean( ·)andstd(·)compute the average
and standard deviation, respectively. However,
GRPO’s group normalization may overly magnify
the minor numerical fluctuation. Taking R=
{0.49,0.51}for example, the mean( R)is0.5, the
std(R)is0.01, and the computed advantages are
{−1.0,1.0}, which overly magnifies the minor nu-
merical fluctuation. To mitigate this issue, we pro-
pose to clip std(R)to be at least ϵstdto ensure that
the denominator does not become too small:
˜Ai=Ri−mean( R)
clip_std(R), (11)
where clip_std(R) = max(std( R), ϵstd);ϵstdis
a hyperparameter. After obtaining refined advan-
tages{˜A1,˜A2, ...,˜AG}, we can optimize the cur-
rent policy model πθ(i.e.,the evidence extractor)
by maximizing the following objective function:
JGRPO (θ) =Ex∼D,{yi}G
i=1∼πθold(·|x)"
1
GGX
i=1minπθ(˜yi|x)
πθold(˜yi|x)˜Ai,
clipπθ(˜yi|x)
πθold(˜yi|x),1−ε,1 +ε
˜Ai
−β DKL 
πθπref#
,(12)
where x={q, P}denotes input samples drawn
from the dataset D;yis the model’s response that
consists of {r, e, o r, oe, of}, sampled from the old
policy πθold;˜yis a self-contained response consist-
ing of{r, e, o f}, where ofis an output conditioned
on full context, so they are self-contained and used
for training; ϵandβare the PPO clipping hy-
perparameters and the weighting coefficient con-
trolling the Kullback–Leibler (KL)-divergence, re-
spectively; πrefrepresents the reference policy.
4 Experiment
In this section, we conduct extensive experi-
ments on three knowledge-intensive benchmark
datasets to answer the following Research Ques-
tions ( RQs ):RQ1: How does rational evidence
perform, w.r.t. downstream task performance,

Dataset #Train #Dev #Test
NQ (Kwiatkowski et al., 2019) 79.1k 8.7k 3.6k
TQA (Joshi et al., 2017) 78.7k 8.8k 11.3k
HotoptQA (Yang et al., 2018) 88.9k 5.6k 5.6k
Table 2: Statistics of the datasets.
compared to other vanilla evidence? RQ2: How
do the properties of rational evidence vary with the
RL training process? RQ3: How does the infer-
ence efficiency of LEAR compare with that of dif-
ferent types of methods? RQ4: Can rational ev-
idence perform robustly against retrieval noises?
RQ5: How do different parts of the answer re-
wards contribute to the final model performance?
4.1 Experimental Setup
Datasets and Metrics. We experiment on three
knowledge-intensive benchmark QA datasets, i.e.,
Natural Questions (NQ) (Kwiatkowski et al.,
2019), TriviaQA (TQA) (Joshi et al., 2017), and
HotpotQA (Yang et al., 2018), where the first two
are open-domain question answering and the last
one is multi-hop question answering. The detailed
statistics of these three datasets are provided in
Table 2, where the test set for HotpotQA is un-
available, and thus we use its dev set as a substi-
tute for the test set. For evaluation, we adopt the
Exact Match (EM) and unigram F1score to eval-
uate QA performance. EM examines exact cor-
rectness while F1calculates the degree of lexi-
cal overlap, offering a more fine-grained view of
how well the prediction aligns with the golden an-
swer. To measure the improvement of the com-
putational efficiency, we employ the Compression
Ratio (CR) following the previous works (Hwang
et al., 2024a; Pan et al., 2024), where CR com-
putes the ratio of the total length of the retrieval
passages to the length of the extracted evidence.
Implementation Details. We employ Qwen2.5-
1.5B-Instruct and Qwen2.5-7B-Instruct5(Yang
et al., 2025) as the initial models. We train
these two models to extract rational evidence
through full parameter fine-tuning for 1 epoch on
the NQ dataset. The learning rate is 1e−6, and
we select the best-performing model on the dev
set. For modeling verifiable rewards, we tune
τandγwithin the ranges of {0.1,0.2,0.5,1.0}
and{0.1,0.3,0.5,0.8,1.0}, respectively. For the
length threshold ωand the weighting coefficients
(i.e.,αa,αl, and αf), we empirically set ω,
5https://github.com/QwenLM/Qwen2.5DatasetRecall NDCG
Train Dev Test Train Dev Test
NQ 78.74 73.07 74.07 68.30 61.95 63.08
TQA 82.35 77.97 77.77 76.06 70.32 70.35
HotoptQA 34.02 28.45 - 27.36 22.39 -
Table 3: Recall and NDCG of the top-5 retrieval pas-
sages in terms of training, development, and test sets.
αa,αl, and αfas 0.9, 0.8, 0.1, 0.1, respec-
tively, which commonly leads to satisfactory per-
formance. For policy optimization, we set ϵstdas
0.1 to stabilize training. During training, we set
the PPO clipping hyperparameter ϵand the co-
efficient βcontrolling KL-divergence as 0.2 and
1e−2, respectively. For QA generators, we employ
LoRA (Yang et al., 2025) to fine-tune Qwen2.5-
1.5B-Instruct and Qwen2.5-7B-Instruct to predict
the golden answer abased on the query and re-
trieved passages ( q,P) or only the query (used for
the baseline ‘Zero’), where LoRA rank is set as 8.
Passage Retrieval. Following previous works
(Wang et al., 2023; Zhu et al., 2024; Zhao et al.,
2024a; Zhang et al., 2025a), we retrieve Wikipedia
(Wiki) passages for all datasets. We use Dense
Passage Retriever (DPR) (Karpukhin et al., 2020)
to retrieve the top-5 passages from all Wiki pas-
sages, where we use the December 2018 Wiki
dump (Karpukhin et al., 2020) as the retrieval cor-
pus, and set the chunk size as 100 words. We use
the retrieval toolkit, Tevatron (Gao et al., 2023),
to perform corpus encoding, query encoding, and
passage retrieval. Table 3 shows the retrieval per-
formance w.r.t. Recall@5 and NDCG@5, where
the retrieval passage that contains the golden an-
swer is treated as the positive passage. Besides,
for HotpotQA, we compute Recall and NDCG
only for the “bridge” questions, while ignoring
yes/no and comparison questions, following previ-
ous works (Jiang et al., 2024; Khalifa et al., 2023).
The retrieval performance for HotpotQA is rela-
tively low, as it is a multi-hop QA dataset, and an-
swers are usually not spans in retrieval passages.
Comparison Baselines. To verify the effective-
ness of LEAR , we compare it with the follow-
ing three groups of competitive baselines: (1)
Without Refinement (WR) includes (i)Zero-shot
(Zero) generates output relying on LLMs’ para-
metric knowledge; (ii)Full Passage (Full) feeds all
retrieval passages into LLMs; (2) Vanilla Refine-
ment (V AR) directly extracts evidence without ex-
plicit thinking, which includes (i)Select Context

Datasets MetricsWR V AR RAR
Zero Full SeleCtx LLMLingua-2 R ECOMP FilCo SEER CoT LEAR
1.5B size model
EM 13.74 41.97 24.68 29.36 37.40 36.62 36.93 37.70 41.14
F1 53.16 70.07 59.52 62.82 67.18 66.63 67.11 68.09 70.77 NQ
CR - 1.0x 3.44x 4.51x 5.43x 16.3x 13.2x 4.56x 38.1x
EM 29.10 57.02 45.13 48.67 56.56 54.06 54.57 54.29 56.84
F1 64.62 80.13 73.40 75.29 79.85 78.56 78.81 79.56 80.85 TQA†
CR - 1.0x 3.38x 4.52x 5.35x 8.55x 10.3x 5.02x 38.8x
EM 12.36 19.20 15.30 16.64 18.52 18.18 18.60 19.52 20.46
F1 48.52 53.04 49.65 51.26 52.92 52.15 52.79 53.43 54.20 HotpotQA†
CR - 1.0x 3.40x 4.52x 5.44x 18.3x 15.5x 4.17x 33.0x
7B size model
EM 25.04 48.78 33.91 36.51 43.77 44.79 45.01 44.49 46.95
F1 60.88 74.40 65.71 67.77 71.61 72.30 72.76 73.09 73.99 NQ
CR - 1.0x 3.44x 4.51x 5.43x 15.4x 12.4x 3.36x 17.9x
EM 47.31 65.34 58.72 60.50 64.40 63.46 64.20 63.91 64.76
F1 74.36 84.83 80.95 81.94 84.12 83.66 84.14 84.45 84.74 TQA†
CR - 1.0x 3.38x 4.52x 5.35x 7.77x 9.70x 3.33x 17.6x
EM 17.95 25.82 21.27 23.45 24.84 25.27 25.81 27.25 28.02
F1 53.07 58.50 54.90 56.65 58.02 58.15 58.63 59.84 60.30 HotpotQA†
CR - 1.0x 3.40x 4.52x 5.44x 17.5x 14.3x 3.32x 16.5x
Table 4: Overall performance comparison on NQ, TQA, and HotpotQA benchmark datasets, where the best results
areboldfaced and the second-best results are underlined . EM, F1, and CR denote exact match, F1 score, and
compression ratio (the higher the better), respectively. The 1.5B/7B size models represent Qwen2.5-1.5B/7B-
Instruct. The †symbol denotes out-of-domain (OOD) evaluation datasets for LEAR , as it is only trained on NQ.
(SeleCtx) (Li et al., 2023) identifies and prunes re-
dundancy in the input context based on perplexity;
(ii)LLMLingua-2 (Pan et al., 2024) distills com-
pression knowledge from GPT-4 to reduce cru-
cial information losing; (iii)FILCO(Wang et al.,
2023) trains a context filtering model to identify
key clues; (iv)BottleNeck (Zhu et al., 2024) ap-
plies the information bottle theory to select SFT
data used to optimize filtering; (v)SEER (Zhao
et al., 2024a) learns to extract desired evidence via
self-aligned learning; (3) Rational Refinement
(RAR) includes (i)Chain-of-Thought (CoT) (Wei
et al., 2022) generates query-related information
from retrieval passages with explicit thinking.
4.2 Main Comparison (RQ1)
The overall comparison results on NQ, TQA, and
HotpotQA are shown in Table 4. From the exper-
imental results, we mainly have the following ob-
servations. (1)In all cases (18/18), LEAR achieves
the best or second-best results, indicating the su-
periority of supplementing RAG with rational ev-
idence. (2)It is surprising that directly employing
LEAR trained on NQ and tested on OOD datasets
(i.e.,TQA and HotpotQA) yields such impressive
performance. This demonstrates that online pol-
icy optimization endows LEAR with superior gen-
eralization. (3)Compared to ‘Full’, LEAR pos-sesses an extremely high compression ratio ( e.g.,
38.1x with 1.5B size model on NQ) and its per-
formance is very close to or even better than that
of ‘Full’. Though the baseline models have rela-
tively high compression ratios, they are accompa-
nied by significant performance degradation. This
demonstrates again that rational evidence benefits
downstream performance than vanilla one and also
improves the signal-to-noise ratio. (4)By compar-
ing the 1.5B LEAR model with the 7B one, we find
that the 7B one tends to extract more informative
evidence than the 1.5B one. (5)LEAR consider-
ably outperforms V AR methods in almost all cases
and provides more compact evidence, demonstrat-
ing the necessity of explicit evidence reasoning.
(6)In HotpotQA, RAR methods significantly out-
perform V AR ones, and even better than ‘Full’,
indicating that rational refinement is important to
multi-hop question answering. In conclusion, the
above results and observations fully validate the
effectiveness and efficiency of rational evidence.
4.3 Training Dynamics (RQ2)
The RL training dynamics regarding the answer
rewards and response length are shown in Fig-
ure 5. Note that the generators used in Table 4
and Figure 5 are different. Therefore, the answer
rewards in Figure 5 are slightly different from the

(a) Answer reward dynamics on 1.5B model.
 (b) Answer reward dynamics on 7B model.
 (c) Length dynamics.
Figure 5: Training dynamics w.r.t. answer reward and response (including ‘reason’, ‘extract’, and ‘answer’) length.
Models NQ TQA HotpotQA Avg
FILCO 0.64 0.82 0.59 0.68
CoT 0.55 0.70 0.71 0.65
LEAR 0.35 0.41 0.43 0.40
Table 5: Inference latency (seconds/query) on the 1.5B
model, where the smaller the better.
F1and EM results in Table 4. The results show
that the answer rewards of full context ( of), ra-
tionale ( or), and rational evidence ( oe) are consis-
tently improved during the process of reinforce-
ment learning. And, it is not surprising that the
answer reward of full context generally performs
best. More specifically, on the 1.5B model, ratio-
nale and rational evidence perform very closely,
while on the 7B model, rational evidence per-
forms slightly worse than rationale. We think
the main reason is that the 7B model can infer
answers based on implicit cues, but rational ev-
idence may compress some of them. As for re-
sponse length dynamics, the response length of the
1.5B model decreases rapidly at the beginning of
training and then increases slowly, while that of
the 7B model decreases slowly all the time. We
think the main reason is that a moderate response
length is beneficial for improving answer rewards,
because an overlong reasoning may confuse an-
swer generation, and a too short reasoning may
omit key cues. Therefore, as training continues,
the response lengths of the 1.5B model and the
7B model tend to converge. In conclusion, the
above observations provide some useful insights,
e.g.,elaborated answer and length rewards, for fur-
ther research on the rational evidence extraction.
4.4 Inference Efficiency (RQ3)
Table 5 presents the inference latency of LEAR
compared with F ILCO(a representative V AR
method) and CoT (a representative RAR method)
on the 1.5B model within a 1 ×A800 station,
where the evaluation batch size and max new to-
kens are set as 64 and 768, respectively. It can be
(a) NQ.
 (b) TQA.
Figure 6: Performance comparisons w.r.t. data noise on
the 1.5B model, where psgs is the abbr of ‘passages’.
observed that the time required for evidence ex-
traction using LEAR is considerably shorter than
FILCOand CoT, facilitating effective application
in online RAG systems. Additionally, it is surpris-
ing that the average inference latency of F ILCO
is slightly longer than that of CoT, while FilCo
generates fewer new tokens on average. We find
that the standard deviation in the lengths of the ev-
idence generated by F ILCOis significantly higher
than that of COT. Specifically, the std of F ILCO
on NQ, TQA, and HotpotQA is 107.5, 136.1, and
81.7, while that of CoT is 67.9, 61.6, and 63.9.
Due to the instability in the length of the generated
evidence, F ILCOtakes longer in batching genera-
tion instead. In summary, the above results fully
verify the efficiency of LEAR during deployment.
4.5 Robustness Analysis (RQ4)
In real-world applications, RAG systems com-
monly suffer from data noise resulting from im-
perfect retrieval. To simulate this scenario, we
randomly sample a certain number ( i.e.,0, 2, 4,
6, and 8) of irrelevant passages for each test query,
where each query is equipped with 5 retrieved rel-
evant passages as well as sampled irrelevant pas-
sages. We experiment on the 1.5B model, and the
experimental results are presented in Figure 6. The
results show that adding noise considerably de-
grades the performance of F ILCO, while the per-
formance degradation of LEAR is relatively small,
where the green line is always below the blue one.

Models Rationale Evidence Full Ctx Evidence+Rationale Evidence+Full Ctx Evidence+Rationale+Full Ctx
NQ 39.89 40.27 39.53 40.02 40.44 41.14
TQA 56.97 57.38 56.17 57.31 57.26 56.84
HotpotQA 20.30 19.73 20.13 20.34 20.23 20.46
Average 38.72 39.13 38.61 39.22 39.31 39.48
Table 6: Performance comparison (EM) of LEAR trained on different combinations of answer rewards.
In particular, LEAR with 8 irrelevant psgs outper-
forms F ILCOwithout noise. In conclusion, this
fully verifies the robustness of LEAR against noise.
4.6 Ablation Study (RQ5)
To evaluate the impact of different answer rewards
on model performance, we train 1.5B-sized LEAR
models on different combinations of answer re-
wards with average weighting. As shown in Ta-
ble 6, among models trained on a single answer re-
ward, training with evidence answer reward Rans
e
yields the highest average performance, demon-
strating that the quality of evidence is the most
important in optimizing rational evidence. For
two-answer reward combinations, ‘Evidence+Full
Ctx’ slightly outperforms ‘Evidence+Rationale’,
with noticeable gains on NQ in particular, indi-
cating that the optimization of ‘Full Ctx’ matters.
Notably, incorporating all three answer rewards
yields the best performance on NQ and HotpotQA,
as well as the highest average score, highlighting
the benefits of multi-faceted reward assessment. In
conclusion, the above results clearly manifest the
contribution and necessity of each answer reward.
5 Related Works
5.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has been
prevalent in LLMs, which effectively enhances
LLMs with external non-parametric knowledge
to remedy their outdated, incorrect, or incom-
plete internal parametric knowledge (Lewis et al.,
2020; Wu et al., 2022; Guu et al., 2020; Izacard
et al., 2023; Asai et al., 2024). The pioneering
attempts (Lewis et al., 2020; Guu et al., 2020)
demonstrated that augmenting the input context
of LLMs with retrieved passages yields significant
improvements, especially in knowledge-intensive
tasks. While prior work usually retrieves straight-
forwardly at the beginning, dynamic RAG (Asai
et al., 2024; Jeong et al., 2024; Jiang et al.,
2023) has been proposed to adaptively retrieve
passages based on the demand of generation and
the complexity of the query. Recently, agenticRAG (Zhang et al., 2025b) has been proposed to
interleave retrieval and reasoning to tackle com-
plex queries. In particular, this kind of work (Song
et al., 2025; Jin et al., 2025; Trivedi et al., 2023)
focuses on prompting or fine-tuning LLMs as
search agents that interact with search tools to get
external knowledge autonomously and on demand
through direct prompting/reinforcement learning.
5.2 RAG with Context Compression
RAG systems usually concatenate all retrieved
passages as the context of LLMs. However, this
may introduce data noise and computational over-
head due to imperfect retrieval and overlong con-
text. Recently, many works have attempted to
compress context and retain important informa-
tion, mainly including two categories: (1) Rerank-
ing methods, which rerank retrieved passages and
retain top-ranked passages (Hwang et al., 2024b;
Mortaheb et al., 2025; Mao et al., 2021) (2) Sum-
marization methods, which extract relevant infor-
mation from retrieved passages and consolidate
them into a narrative(Wang et al., 2023; Zhao
et al., 2024a; Zhu et al., 2024; Xu et al., 2024).
Despite effectiveness, existing works neglect ex-
plicit evidence reasoning, which risks filtering out
key clues as well as struggles with generalization.
6 Conclusion
In this paper, we first unveil the limitations of
the vanilla evidence extraction paradigm and ex-
plore the potential of the rational evidence extrac-
tion paradigm to solve these limitations. Specif-
ically, we propose LEAR , a rational evidence ex-
traction reinforcement learning framework. In
particular, we unify evidence reasoning and ex-
traction into one unified response, and devise ver-
ifiable reward functions to guide the optimization
ofLEAR for compact and high-quality evidence,
where knowledge token masking and std clipping
are applied to avoid information leakage and sta-
bilize RL training, respectively. Extensive exper-
iments fully demonstrate the superiority of LEAR
w.r.t. performance, generalization, and robustness.

Limitations
Despite our innovations and improvements, it does
have limitations, especially cascaded generation
between the rationale and the rational evidence.
This indicates that the rationale must be generated
before the rational evidence, which increases the
inference latency to some extent. Although the re-
sults in Table 5 show that the inference efficiency
ofLEAR remains superior to that of the baseline,
we believe that further improvements in reason-
ing efficiency are necessary to achieve even higher
inference speed. For example, reasoning should
be conducted only when necessary (Jiang et al.,
2025), i.e., generating rational evidence directly
for easy instances. We leave it for future research.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup
Sil, and Hannaneh Hajishirzi. 2024. Self-
rag: Learning to retrieve, generate, and critique
through self-reflection. In The Twelfth Interna-
tional Conference on Learning Representations,
ICLR 2024, Vienna, Austria, May 7-11, 2024 .
OpenReview.net.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei
Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao
Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu,
Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhu-
oshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingx-
uan Wang, Bochao Wu, Bei Feng, Chengda
Lu, Chenggang Zhao, Chengqi Deng, Chenyu
Zhang, Chong Ruan, Damai Dai, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong
Dai, Fuli Luo, Guangbo Hao, Guanting Chen,
Guowei Li, H. Zhang, Han Bao, Hanwei
Xu, Haocheng Wang, Honghui Ding, Huajian
Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong
Guo, Jiashi Li, Jiawei Wang, Jingchang Chen,
Jingyang Yuan, Junjie Qiu, Junlong Li, J. L.
Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong,
Kai Hu, Kaige Gao, Kang Guan, Kexin Huang,
Kuai Yu, Lean Wang, Lecong Zhang, Liang
Zhao, Litong Wang, Liyue Zhang, Lei Xu,
Leyi Xia, Mingchuan Zhang, Minghua Zhang,
Minghui Tang, Meng Li, Miaojun Wang, Ming-
ming Li, Ning Tian, Panpan Huang, Peng
Zhang, Qiancheng Wang, Qinyu Chen, Qiushi
Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan,Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen,
Shanghao Lu, Shangyan Zhou, Shanhuang
Chen, Shengfeng Ye, Shiyu Wang, Shuiping
Yu, Shunfeng Zhou, Shuting Pan, and S. S.
Li. 2025. Deepseek-r1: Incentivizing reason-
ing capability in llms via reinforcement learn-
ing.CoRR , abs/2501.12948.
Yanrui Du, Sendong Zhao, Haochun Wang, Yuhan
Chen, Rui Bai, Zewen Qiang, Muzhen Cai, and
Bing Qin. 2023. Make your decision convinc-
ing! A unified two-stage framework: Self-
attribution and decision-making. In Findings
of the Association for Computational Linguis-
tics: EMNLP 2023, Singapore, December 6-10,
2023 , pages 1101–1112. Association for Com-
putational Linguistics.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie
Callan. 2023. Tevatron: An efficient and flexi-
ble toolkit for neural retrieval. In Proceedings
of the 46th International ACM SIGIR Confer-
ence on Research and Development in Informa-
tion Retrieval, SIGIR 2023, Taipei, Taiwan, July
23-27, 2023 , pages 3120–3124. ACM.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong
Pasupat, and Ming-Wei Chang. 2020. REALM:
retrieval-augmented language model pre-
training. CoRR , abs/2002.08909.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. 2022. Lora: Low-rank
adaptation of large language models. ICLR ,
1(2):3.
Taeho Hwang, Sukmin Cho, Soyeong Jeong,
Hoyun Song, SeungYoon Han, and Jong C.
Park. 2024a. EXIT: context-aware extractive
compression for enhancing retrieval-augmented
generation. CoRR , abs/2412.12559.
Taeho Hwang, Soyeong Jeong, Sukmin Cho, Se-
ungyoon Han, and Jong C. Park. 2024b. DSLR:
document refinement with sentence-level re-
ranking and reconstruction to enhance retrieval-
augmented generation. CoRR , abs/2407.03627.
Gautier Izacard, Patrick S. H. Lewis, Maria
Lomeli, Lucas Hosseini, Fabio Petroni, Timo
Schick, Jane Dwivedi-Yu, Armand Joulin, Se-
bastian Riedel, and Edouard Grave. 2023. At-
las: Few-shot learning with retrieval aug-

mented language models. J. Mach. Learn. Res. ,
24:251:1–251:43.
Soyeong Jeong, Jinheon Baek, Sukmin Cho,
Sung Ju Hwang, and Jong Park. 2024.
Adaptive-rag: Learning to adapt retrieval-
augmented large language models through
question complexity. In Proceedings of the
2024 Conference of the North American Chap-
ter of the Association for Computational Lin-
guistics: Human Language Technologies (Vol-
ume 1: Long Papers), NAACL 2024, Mexico
City, Mexico, June 16-21, 2024 , pages 7036–
7050. Association for Computational Linguis-
tics.
Lingjie Jiang, Xun Wu, Shaohan Huang, Qingxiu
Dong, Zewen Chi, Li Dong, Xingxing Zhang,
Tengchao Lv, Lei Cui, and Furu Wei. 2025.
Think only when you need with large hybrid-
reasoning models. CoRR , abs/2505.14631.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. 2023.
Active retrieval augmented generation. In
EMNLP , pages 7969–7992. Association for
Computational Linguistics.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen.
2024. Longrag: Enhancing retrieval-
augmented generation with long-context
llms. CoRR , abs/2406.15319.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-
r1: Training llms to reason and leverage search
engines with reinforcement learning. CoRR ,
abs/2503.09516.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and
Luke Zettlemoyer. 2017. Triviaqa: A large
scale distantly supervised challenge dataset for
reading comprehension. In Proceedings of the
55th Annual Meeting of the Association for
Computational Linguistics, ACL 2017, Vancou-
ver, Canada, July 30 - August 4, Volume 1:
Long Papers , pages 1601–1611. Association for
Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. 2020. Densepassage retrieval for open-domain question an-
swering. In Proceedings of the 2020 Con-
ference on Empirical Methods in Natural
Language Processing, EMNLP 2020, Online,
November 16-20, 2020 , pages 6769–6781. As-
sociation for Computational Linguistics.
Muhammad Khalifa, Lajanugen Logeswaran,
Moontae Lee, Honglak Lee, and Lu Wang.
2023. Few-shot reranking for multi-hop QA
via language model prompting. In Proceedings
of the 61st Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long
Papers), ACL 2023, Toronto, Canada, July 9-
14, 2023 , pages 15882–15897. Association for
Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia
Redfield, Michael Collins, Ankur P. Parikh,
Chris Alberti, Danielle Epstein, Illia Polo-
sukhin, Jacob Devlin, Kenton Lee, Kristina
Toutanova, Llion Jones, Matthew Kelcey,
Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019.
Natural questions: a benchmark for question an-
swering research. Trans. Assoc. Comput. Lin-
guistics , 7:452–466.
Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-
tus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-
tau Yih, Tim Rocktäschel, Sebastian Riedel,
and Douwe Kiela. 2020. Retrieval-augmented
generation for knowledge-intensive NLP tasks.
InAdvances in Neural Information Processing
Systems 33: Annual Conference on Neural In-
formation Processing Systems 2020, NeurIPS
2020, December 6-12, 2020, virtual .
Yucheng Li, Bo Dong, Frank Guerin, and
Chenghua Lin. 2023. Compressing context to
enhance inference efficiency of large language
models. In Proceedings of the 2023 Confer-
ence on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2023, Singapore,
December 6-10, 2023 , pages 6342–6353. Asso-
ciation for Computational Linguistics.
Yuning Mao, Pengcheng He, Xiaodong Liu, Ye-
long Shen, Jianfeng Gao, Jiawei Han, and
Weizhu Chen. 2021. Reader-guided passage
reranking for open-domain question answering.
InFindings of the Association for Computa-
tional Linguistics: ACL/IJCNLP 2021, Online

Event, August 1-6, 2021 , volume ACL/IJCNLP
2021 of Findings of ACL , pages 344–350. As-
sociation for Computational Linguistics.
Matin Mortaheb, Mohammad Ali Amir Kho-
jastepour, Srimat T. Chakradhar, and Sennur
Ulukus. 2025. Re-ranking the context for mul-
timodal retrieval augmented generation. CoRR ,
abs/2501.04695.
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo
Almeida, Carroll L. Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal,
Katarina Slama, Alex Ray, John Schulman, Ja-
cob Hilton, Fraser Kelton, Luke Miller, Maddie
Simens, Amanda Askell, Peter Welinder, Paul F.
Christiano, Jan Leike, and Ryan Lowe. 2022.
Training language models to follow instructions
with human feedback. In Advances in Neu-
ral Information Processing Systems 35: Annual
Conference on Neural Information Processing
Systems 2022, NeurIPS 2022, New Orleans, LA,
USA, November 28 - December 9, 2022 .
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang,
Menglin Xia, Xufang Luo, Jue Zhang, Qing-
wei Lin, Victor Rühle, Yuqing Yang, Chin-
Yew Lin, H. Vicky Zhao, Lili Qiu, and Dong-
mei Zhang. 2024. Llmlingua-2: Data distil-
lation for efficient and faithful task-agnostic
prompt compression. In Findings of the Associ-
ation for Computational Linguistics, ACL 2024,
Bangkok, Thailand and virtual meeting, August
11-16, 2024 , pages 963–981. Association for
Computational Linguistics.
Alec Radford, Karthik Narasimhan, Tim Sali-
mans, Ilya Sutskever, et al. 2018. Improv-
ing language understanding by generative pre-
training.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Prox-
imal policy optimization algorithms. CoRR ,
abs/1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin
Xu, Junxiao Song, Mingchuan Zhang, Y . K. Li,
Y . Wu, and Daya Guo. 2024. Deepseekmath:
Pushing the limits of mathematical reasoning in
open language models. CoRR , abs/2402.03300.
Weijia Shi, Sewon Min, Michihiro Yasunaga,
Minjoon Seo, Richard James, Mike Lewis,Luke Zettlemoyer, and Wen-tau Yih. 2024.
REPLUG: retrieval-augmented black-box lan-
guage models. In Proceedings of the 2024 Con-
ference of the North American Chapter of the
Association for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long
Papers), NAACL 2024, Mexico City, Mexico,
June 16-21, 2024 , pages 8371–8384. Associa-
tion for Computational Linguistics.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie
Chen, Zhipeng Chen, Wayne Xin Zhao, Lei
Fang, and Ji-Rong Wen. 2025. R1-searcher: In-
centivizing the search capability in llms via re-
inforcement learning. CoRR , abs/2503.05592.
Romal Thoppilan, Daniel De Freitas, Jamie
Hall, Noam Shazeer, Apoorv Kulshreshtha,
Heng-Tze Cheng, Alicia Jin, Taylor Bos,
Leslie Baker, Yu Du, YaGuang Li, Hongrae
Lee, Huaixiu Steven Zheng, Amin Ghafouri,
Marcelo Menegali, Yanping Huang, Maxim
Krikun, Dmitry Lepikhin, James Qin, Dehao
Chen, Yuanzhong Xu, Zhifeng Chen, Adam
Roberts, Maarten Bosma, Yanqi Zhou, Chung-
Ching Chang, Igor Krivokon, Will Rusch, Marc
Pickett, Kathleen S. Meier-Hellstern, Mered-
ith Ringel Morris, Tulsee Doshi, Renelito De-
los Santos, Toju Duke, Johnny Soraker, Ben
Zevenbergen, Vinodkumar Prabhakaran, Mark
Diaz, Ben Hutchinson, Kristen Olson, Ale-
jandra Molina, Erin Hoffman-John, Josh Lee,
Lora Aroyo, Ravi Rajakumar, Alena Butryna,
Matthew Lamm, Viktoriya Kuzmina, Joe Fen-
ton, Aaron Cohen, Rachel Bernstein, Ray
Kurzweil, Blaise Agüera y Arcas, Claire Cui,
Marian Croak, Ed H. Chi, and Quoc Le. 2022.
Lamda: Language models for dialog applica-
tions. CoRR , abs/2201.08239.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions.
InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics
(Volume 1: Long Papers), ACL 2023, Toronto,
Canada, July 9-14, 2023 , pages 10014–10037.
Association for Computational Linguistics.
Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N. Gomez,
Lukasz Kaiser, and Illia Polosukhin. 2017. At-

tention is all you need. In Advances in Neu-
ral Information Processing Systems 30: An-
nual Conference on Neural Information Pro-
cessing Systems 2017, December 4-9, 2017,
Long Beach, CA, USA , pages 5998–6008.
Shuhe Wang, Shengyu Zhang, Jie Zhang, Runyi
Hu, Xiaoya Li, Tianwei Zhang, Jiwei Li, Fei
Wu, Guoyin Wang, and Eduard H. Hovy. 2024.
Reinforcement learning enhanced llms: A sur-
vey. CoRR , abs/2412.10400.
Zhiruo Wang, Jun Araki, Zhengbao Jiang,
Md. Rizwan Parvez, and Graham Neubig.
2023. Learning to filter context for retrieval-
augmented generation. CoRR , abs/2311.08377.
Jason Wei, Xuezhi Wang, Dale Schuurmans,
Maarten Bosma, Brian Ichter, Fei Xia, Ed H.
Chi, Quoc V . Le, and Denny Zhou. 2022.
Chain-of-thought prompting elicits reasoning in
large language models. In Advances in Neu-
ral Information Processing Systems 35: Annual
Conference on Neural Information Processing
Systems 2022, NeurIPS 2022, New Orleans, LA,
USA, November 28 - December 9, 2022 .
Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024.
Instructrag: Instructing retrieval-augmented
generation with explicit denoising. CoRR ,
abs/2406.13629.
Yuxiang Wu, Yu Zhao, Baotian Hu, Pasquale Min-
ervini, Pontus Stenetorp, and Sebastian Riedel.
2022. An efficient memory-augmented trans-
former for knowledge-intensive NLP tasks. In
Proceedings of the 2022 Conference on Empiri-
cal Methods in Natural Language Processing,
EMNLP 2022, Abu Dhabi, United Arab Emi-
rates, December 7-11, 2022 , pages 5184–5196.
Association for Computational Linguistics.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024.
RECOMP: improving retrieval-augmented lms
with context compression and selective aug-
mentation. In The Twelfth International Con-
ference on Learning Representations, ICLR
2024, Vienna, Austria, May 7-11, 2024 . Open-
Review.net.
An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang,Jianxin Yang, Jiaxi Yang, Jingren Zhou, Jun-
yang Lin, Kai Dang, Keming Lu, Keqin Bao,
Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei
Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao
Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren,
Xuancheng Ren, Yang Fan, Yang Su, Yichang
Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru
Zhang, and Zihan Qiu. 2025. Qwen2.5 techni-
cal report.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua
Bengio, William W. Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. 2018. Hot-
potqa: A dataset for diverse, explainable multi-
hop question answering. In Proceedings of the
2018 Conference on Empirical Methods in Nat-
ural Language Processing, Brussels, Belgium,
October 31 - November 4, 2018 , pages 2369–
2380. Association for Computational Linguis-
tics.
Qianchi Zhang, Hainan Zhang, Liang Pang, Hong-
wei Zheng, Yongxin Tong, and Zhiming Zheng.
2025a. Finefilter: A fine-grained noise filtering
mechanism for retrieval-augmented large lan-
guage models. CoRR , abs/2502.11811.
Weizhi Zhang, Yangning Li, Yuanchen Bei, Junyu
Luo, Guancheng Wan, Liangwei Yang, Chenx-
uan Xie, Yuyao Yang, Wei-Chieh Huang,
Chunyu Miao, Henry Peng Zou, Xiao Luo,
Yusheng Zhao, Yankai Chen, Chunkit Chan,
Peilin Zhou, Xinyang Zhang, Chenwei Zhang,
Jingbo Shang, Ming Zhang, Yangqiu Song, Ir-
win King, and Philip S. Yu. 2025b. From web
search towards agentic deep research: Incen-
tivizing search with reasoning agents. CoRR ,
abs/2506.18959.
Xinping Zhao, Dongfang Li, Yan Zhong, Boren
Hu, Yibin Chen, Baotian Hu, and Min Zhang.
2024a. SEER: self-aligned evidence extrac-
tion for retrieval-augmented generation. In Pro-
ceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing,
EMNLP 2024, Miami, FL, USA, November 12-
16, 2024 , pages 3027–3041. Association for
Computational Linguistics.
Xinping Zhao, Jindi Yu, Zhenyu Liu, Jifang Wang,
Dongfang Li, Yibin Chen, Baotian Hu, and Min
Zhang. 2024b. Medico: Towards hallucina-
tion detection and correction with multi-source

evidence fusion. In Proceedings of the 2024
Conference on Empirical Methods in Natural
Language Processing: System Demonstrations ,
pages 34–45.
Xinping Zhao, Yan Zhong, Zetian Sun, Xinshuo
Hu, Zhenyu Liu, Dongfang Li, Baotian Hu, and
Min Zhang. 2025. Funnelrag: A coarse-to-
fine progressive retrieval paradigm for RAG. In
NAACL (Findings) , pages 3029–3046. Associa-
tion for Computational Linguistics.
Kun Zhu, Xiaocheng Feng, Xiyuan Du, Yuxuan
Gu, Weijiang Yu, Haotian Wang, Qianglong
Chen, Zheng Chu, Jingchang Chen, and Bing
Qin. 2024. An information bottleneck perspec-
tive for effective noise filtering on retrieval-
augmented generation. In Proceedings of the
62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Pa-
pers), ACL 2024, Bangkok, Thailand, August
11-16, 2024 , pages 1044–1069. Association for
Computational Linguistics.
A Prompts
We provide the prompts that are used for retrieval-
augmented QA, closed-book QA, CoT generation,
and rational evidence extraction in Table 7, Ta-
ble 8, Table 9, and Table 10, respectively.
B Case Study
Through extensive case studies, we found the rea-
sons for the failure of vanilla evidence mainly lie
in threefold, i.e.,Incompleteness, Irrelevance, and
Inaccuracy, termed as “ 3I”. For Incompleteness,
an example on the left of Figure 7, vanilla evi-
dence provides information about the award cat-
egory (Best Actor) but fails to provide the specific
name of the award winner. For Irrelevance, an ex-
ample in the middle of Figure 7, vanilla evidence
provides irrelevant information about a different
work and year. For Inaccuracy, an example in the
right of Figure 7, vanilla evidence contains a fac-
tual error, attributing the opera "The Midsummer
Marriage" to the wrong composer (Benjamin Brit-
ten instead of Michael Tippett). On the contrary,
rational evidence demonstrates a greater capacity
for understanding the query’s intent, verifying fac-
tual accuracy, and reasoning to arrive at the correct
and relevant evidence. It moves beyond simple ev-
idence extraction towards reasonable evidence ex-
traction.Prompt for Retrieval-Augmented QA
[Instruction]
You are a helpful assistant. Your task is:
1. Read the given question and use the
documents provided to answer the
question.
2. If the documents don’t work, please an-
swer the question based on your own
knowledge.
Question: {question}
Document: {document}
Answer:
Table 7: The prompt for retrieval-augmented QA.
Prompt for Closed-book QA
[Instruction]
You are a helpful assistant. Your task is:
1. Read the given question and then an-
swer the question directly.
2. Give a short answer to the question
based on your own knowledge.
Question: {question}
Answer:
Table 8: The prompt for closed-book QA.
Prompt for CoT Generation
[Instruction]
You are a helpful assistant. Your task is:
Read the given documents, and answer the
question below.
Question: {question}
Document: {document}
Let’s think step by step.
Table 9: The prompt for Chain-of-Thought generation.

Prompt for Rational Evidence Extraction
[Instruction]
You are a highly skilled knowledge reasoner and extractor.
Your task is to carefully read the given question and passages to reason how the passages lead to
the answer and extract relevant information that may be used to answer the question.
Follow these steps:
1. In the <reason></reason> tag, perform the following steps. Question Analysis: Analyze the
question to understand the specific information they are seeking. Identify the key concepts,
entities, and relationships involved. Passage Analysis: For each passage, carefully read and
identify sentences or phrases that are useful for answering the given question.
2. In the <extract></extract> tag, synthesize useful information from the passages into a coherent
narrative. Organize the information logically and concisely.
3. In <answer></answer> tags, give a short answer to the given question, based on the passages,
reasoning information, and extracted knowledge. If none of them work, please answer the
question based on your knowledge.
Question: {question}
Passages: {passages}
Table 10: The prompt for rational evidence extraction, where generation is terminated when encountering token
‘</extract>’.
Figure 7: Three main failure issues, termed as 3I: (1) Incompleteness , where the evidence provides some relevant
information, but lacks the key clues needed to answer the questions; (2) Irrelevance , where the evidence provides
information about the wrong entity, event, or topic; and (3) Inaccuracy , where the evidence contains incorrect
information, such as wrong dates, names, or relationships.