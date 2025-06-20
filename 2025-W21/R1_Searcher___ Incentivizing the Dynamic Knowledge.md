# R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning

**Authors**: Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen

**Published**: 2025-05-22 17:58:26

**PDF URL**: [http://arxiv.org/pdf/2505.17005v1](http://arxiv.org/pdf/2505.17005v1)

## Abstract
Large Language Models (LLMs) are powerful but prone to hallucinations due to
static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting
external information, but current methods often are costly, generalize poorly,
or ignore the internal knowledge of the model. In this paper, we introduce
R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage
both internal and external knowledge sources. R1-Searcher++ employs a two-stage
training strategy: an initial SFT Cold-start phase for preliminary format
learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses
outcome-supervision to encourage exploration, incorporates a reward mechanism
for internal knowledge utilization, and integrates a memorization mechanism to
continuously assimilate retrieved information, thereby enriching the model's
internal knowledge. By leveraging internal knowledge and external search
engine, the model continuously improves its capabilities, enabling efficient
retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++
outperforms previous RAG and reasoning methods and achieves efficient
retrieval. The code is available at
https://github.com/RUCAIBox/R1-Searcher-plus.

## Full Text


<!-- PDF content starts -->

arXiv:2505.17005v1  [cs.CL]  22 May 2025R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of
LLMs via Reinforcement Learning
Huatong Song1*, Jinhao Jiang1*, Wenqing Tian3, Zhipeng Chen1,
Yuhuan Wu1, Jiahao Zhao1, Yingqian Min1,
Wayne Xin Zhao1†, Lei Fang2, Ji-Rong Wen1
1Gaoling School of Artificial Intelligence, Renmin University of China.
2DataCanvas Alaya NeW.3Beijing Institute of Technology.
{songhuatong123, jiangjinhao}@ruc.edu.cn, batmanfly@gmail.com
Abstract
Large Language Models (LLMs) are powerful
but prone to hallucinations due to static knowl-
edge. Retrieval-Augmented Generation (RAG)
helps by injecting external information, but cur-
rent methods often are costly, generalize poorly,
or ignore the model’s internal knowledge. In
this paper, we introduce R1-Searcher++, a
novel framework designed to train LLMs to
adaptively leverage both internal and external
knowledge sources. R1-Searcher++ employs a
two-stage training strategy: an initial SFT Cold-
start phase for preliminary format learning, fol-
lowed by RL for Dynamic Knowledge Acquisi-
tion. The RL stage uses outcome-supervision
to encourage exploration, incorporates a re-
ward mechanism for internal knowledge uti-
lization, and integrates a memorization mecha-
nism to continuously assimilate retrieved infor-
mation, thereby enriching the model’s internal
knowledge. By leveraging internal knowledge
and external search engine, the model contin-
uously improves its capabilities, enabling effi-
cient retrieval-augmented reasoning. Our ex-
periments demonstrate that R1-Searcher++ out-
performs previous RAG and reasoning meth-
ods and achieves efficient retrieval. The code is
available at https://github.com/RUCAIBox/
R1-Searcher-plus .
1 Introduction
Large language models (LLMs) (Zhao et al., 2023)
have demonstrated remarkable reasoning capabili-
ties by only leveraging the information encoded in
their parameters. However, their reliance on static,
internal knowledge leads to notable limitations. At
the simultaneously, this reliance easily leads to
hallucinations (Huang et al., 2025), so LLMs may
struggle with open-ended tasks (Wang et al., 2025c;
Trivedi et al., 2022). Therefore, it is crucial to en-
able LLMs to access external information when
*Equal contributions.
†Corresponding author.they are confused during the reasoning process to
achieve more deliberative reasoning (Jiang et al.,
2024a).
To address this issue, extensive research has fo-
cused on augmenting LLMs with external informa-
tion sources ( i.e.,RAG (Gao et al., 2024)). Early
approaches emphasize specific prompting strate-
gies to guide LLMs (Li et al., 2025; Teng et al.,
2025) and subsequent studies investigate to dis-
till this capability into smaller LLMs through su-
pervised fine-tuning (SFT) (Wang et al., 2025b).
However, recent findings suggest that SFT-based
distillation can cause models to memorize solution
paths, limiting their generalization to novel scenar-
ios (Chu et al., 2025). Further proposals include a
test-time scaling method (Li et al., 2024), notably
employing the Monte Carlo Tree Search (MCTS)
framework (Sun et al., 2025) to enhance solution-
finding by expanding the search space during infer-
ence, but this approach incurs significant inference
overhead, reducing its practicality for widespread
use. Recent studies employ end-to-end outcome-
based reinforcement learning (RL) to train models,
enabling them to autonomously explore external
retrieval environments during inference (Jin et al.,
2025; Song et al., 2025). This approach fosters the
development of self-directed retrieval capabilities
in LLMs as they reason. However, such models
often become overly reliant on external search en-
gines after training, neglecting the utilization of
their internal knowledge.
In practice, when humans attempt to solve fac-
tual questions, they first recall their internal knowl-
edge, and only turn to search engines when they
recognize a lack of information. At the same time,
after obtaining the external searched information,
humans would memorize this knowledge for future
use. For LLMs, extensive pretraining on large-scale
data has already endowed them with substantial
internal knowledge (Qwen et al., 2025). There-
fore, it is essential to equip models with the ability

to dynamically switch between internal and exter-
nal knowledge sources as needed. Furthermore,
models should be encouraged to effectively mem-
orize useful information encountered during train-
ing (Jiang et al., 2024b), progressively enriching
their internal knowledge and continuously evolving
toward greater intelligence.
In this paper, we present R1-Searcher++ , a
novel framework designed to teach LLMs to adap-
tively leverage both internal and external knowl-
edge. We adopt a two-stage training strategy: SFT
Cold-start andRL for Dynamic Knowledge Acquisi-
tion. In the first phase, we employ reject sampling
to collect data that meets the format requirements
and perform a cold start with SFT. In the second
stage, we further train the model using outcome-
based RL to guide the model in dynamically ac-
quiring knowledge, which is to encourage reliance
on internal knowledge when confident, and invoke
external search mechanisms when uncertain, based
on a carefully designed reward design. Addition-
ally, we further introduce a memory mechanism,
enabling the model to retain knowledge encoun-
tered during training by converting and memorizing
retrieved content. This mechanism continuously
enriches its internal knowledge, empowering it to
effectively balance internal reasoning and exter-
nal retrieval through autonomous exploration and
timely memorization.
To verify the effectiveness of R1-Searcher++,
we conduct extensive experiments based on Qwen-
2.5-7B-Instruct. Notably, our method surpasses
the strong baseline by up to 4.3% and reduces the
retrieval count by 42.9% compared to vanilla RL-
based approaches.
Our principal contributions are as follows:
•We introduce R1-Searcher++, teaching LLMs
to adaptively leverage both internal and external
knowledge through a two-stage training strategy.
•We encourage the model to actively leverage its
internal knowledge while efficiently memorizing
external information, enabling dynamic knowledge
acquisition through exploration and memorization.
•Extensive experiments show that R1-
Searcher++ outperforms existing RAG methods,
while significantly reducing the number of re-
trievals compared to vanilla RL-based approaches.
2 Related Work
Retrieval-Augmented Generation. To improve
the factual accuracy of LLM inference and reducehallucinations, researchers have proposed enhanc-
ing language models by incorporating external in-
formation sources, a paradigm known as RAG (Fan
et al., 2024). Early RAG approaches primarily
include Branching (Kim et al., 2024), Summariza-
tion (Li et al., 2023), and Adaptive Retrieval (Jeong
et al., 2024) strategies. As foundation models have
become increasingly capable, exhibiting strong
CoT reasoning abilities, many studies have com-
bined RAG with CoT. These efforts include meth-
ods that prompt the model to perform step-by-step
retrieval (Shao et al., 2023; Trivedi et al., 2023) and
strategies that distill such capabilities into smaller
LLMs (Asai et al., 2024). In parallel, several
works have explored test-time scaling, notably us-
ing MCTS (Feng et al., 2025) to dynamically ex-
pand reasoning paths. However, such approaches
often incur substantial inference-time overhead.
More recently, researchers have trained models us-
ing outcome- supervision RL (Zheng et al., 2025)
to encourage the exploration of more effective ac-
tions and retrieval behaviors, but it leads models to
over-rely on external search engines, diminishing
their ability to leverage internal knowledge (Wang
et al., 2025a). Enabling LLMs to effectively inte-
grate and alternate between internal knowledge and
external retrieval remains a significant challenge.
Reinforcement Learning. To improve training ef-
ficiency, several off-policy algorithms have since
been proposed (Rafailov et al., 2024; Ethayarajh
et al., 2024); however, these methods still face
limitations in terms of preference modeling ac-
curacy and generalization capability (Pang et al.,
2024). DeepseekMath introduced the GRPO al-
gorithm (Shao et al., 2024), which enables effi-
cient self-exploration through a mechanism of rel-
ative preference optimization. Building on this,
Deepseek-R1 (DeepSeek-AI et al., 2025) have
demonstrated that outcome-based RL can signifi-
cantly enhance the reasoning abilities of large mod-
els. More recently, studies have begun to inves-
tigate RL algorithms specifically designed to im-
prove LLM reasoning capabilities (Yu et al.; Yuan
et al., 2025). In parallel, other research efforts have
applied RL to the retrieval domain, aiming to en-
able deep search capabilities (Chen et al., 2025).
However, the use of RL that combine LLM-driven
retrieval and reasoning remains largely simplistic
and underexplored.

3 Preliminary
To enhance the performance of LLMs in open-
domain multi-hop question answering tasks (Ho
et al., 2020), in this work, we focus on enabling the
model to autonomously decide when to use its in-
ternal knowledge or to invoke an external retriever
to answer the given questions with the LLM self-
improving paradigm, which can improve both rea-
soning effectiveness and efficiency. To this end, we
introduce three special tokens to format the LLM
reasoning process, i.e.,<internal> ,<external> ,
and<document> . Concretely, during the reasoning
process, the LLM with parameters θdetermines
whether the current step requires external knowl-
edge to help perform reasoning. If so, it triggers
the<external> to issue a query t, which is sent to
a retriever to retrieve the top- Krelevant documents
Dt={dt,k}K
k=1from an external corpus. These re-
trieved documents are incorporated into the reason-
ing path with another special token <document> .
Otherwise, the model directly generates the related
internal knowledge enclosed in <internal> . After
several reasoning steps, the LLM obtains the final
answer and stops the reasoning process.
Since our approach is orthogonal to the RL
algorithm, we conduct the experiments based
on a widely used RL algorithm, i.e., REIN-
FORCE++ (Hu, 2025), which is a stable RL algo-
rithm without the critic model. To better accommo-
date the retrieval scenario, we mask the retrieved
documents during the loss calculation process, as
they serve as environmental observations rather
than model-generated content. Formally, for each
question q, we first samples a group of outputs
{o1, o2,···, oG}from the old policy model πθold.
Next, we incorporate the KL regularization into the
reward scores Rϕ(q, oi,≤t), and then normalize the
advantage scores:
ˆA′
i,t=R(q, oi)−β·PT
i=tKL(i),ˆAi,t=ˆA′
i,t−mean(ˆA)
std(ˆA)
We utilize ˆAto denote the set of all advantages in
the global batch that contains ˆAi,t. After obtaining
the advantage scores, we set the mask value M(i, t)
as 0 if this token belongs to an external document,
otherwise we set M(i, t) = 1 . Finally, we em-
ploy the masks M(i, t)in the objective function to
remove the influence of retrieved documents:
ˆPi,t= minh
pi,tˆAi,t,clip(pi,t,1−ε,1 +ε)ˆAi,ti
,
JMask(θ) =1
GPG
i=11P|oi|
t=1M(i,t)P|oi|
t=1M(i, t)·ˆPi,t(1)where εis a hyper-parameter and p(i, t)is the im-
portant sampling coefficient , and πθis the policy
model.
4 Methodology
In this part, we introduce the R1-Searcher++
framework, which aims to teach LLM to adaptively
utilize internal and external knowedge through two
critical stages, i.e., SFT Cold-Start (Section 4.1)
andRL for Dynamic Knowledge Acquisition (Sec-
tion 4.2). Concretely, in the first stage, we utilize
the curated data to perform SFT on the model, to
standardize its responses in a specific format and
enable it to leverage external retrievers and inter-
nal knowledge adaptively. In the second stage, we
employ RL on LLM that encourages the model
to explore more effective actions and behaviours,
and further incorporate the internal knowledge uti-
lization encouragement and external knowledge
memorization in the training process, to guide the
model to dynamically acquire knowledge and con-
tinuously enrich its internal knowledge, which can
lead to higher reasoning efficiency.
4.1 SFT Cold-start
To equip LLMs with the preliminary ability to au-
tonomously perform external retrieval during infer-
ence while effectively leveraging internal knowl-
edge, we synthesize high-quality training instances
using rejection sampling, without relying on other
powerful models.We only keep the correct re-
sponses with the appropriate occurrences of both
the<internal> and<external> tags, teaching
LLM to perform dynamic knowledge acquisition
in a proper format.
Specifically, given the question xand the syn-
thesized output y, once the i-th token of the output
belongs to the external document, it will be masked,
i.e.,Mi= 0. Otherwise, the coefficient Miwill be
set as 1, incorporating the probability of yiinto the
objective function as follows,
LSFT=−1Pn
j=1MjnX
i=1Mi×P(yi|x, y<i) (2)
4.2 RL for Dynamic Knowledge Acquisition
After cold-starting, we obtain a model that can uti-
lize internal knowledge and perform an external
search with correct format. To further enhance
its capabilities, i.e.,to perform effective and ef-
ficient reasoning, we continually train the model

Which was released earlier, Avatar or Titanic?Internal Knowledge UtilizationEncouragementinternalI don’t know... , I need to search for the information: <ex> The foundation time of Citybank</ex><doc>…Citybankwas founded in 1812...</doc> Based on the retrieved documents , I know...Two main action types of R1-Searcher++
External SearchBase
So I can find ... , Final Answer: James MadisonData Sampling & SFT Cold-StartPolicy
Reasoning PathThen... , What was the presidency of America in 1812? ... recall ... <in> James Madison ... as president of America in 1812…</in> Using internal knowledge, I know ...
Internal Think
Question: Who was the president of the United States in the year Citibank was founded?
RewriterInputLM (θ)
…
Q: Which came first, Avatar or Titanic?
…don’t know … need to searchfor:  <ex> highest mountain  the world</ex> <doc> highest mount, "Mount Everest",   8848 m</doc>
What is the highest mountain in the world?
internalexternalAdaptiveUtilization
Response: To answer this question, I will ... . Firstly, I want to know: When was Citybank founded?
……
Firstly...……To begin with...……Titanic…Titanic…Correct ResponsesFirstly...…search <ex> …</ex><doc>...Incorrect ResponsesAvatar ……Rolloutbos<in><ex>…ans<ex>ans<ex><ex>ans<ex><ex>ans<in><in><in>
Ranswer=1Ranswer=1Ranswer=1Ranswer=0RC=1…...Rall=Rans+Rformat+RgroupRall=Rans+Rformat+RgroupRall=Rans+Rformat+Rgroup
Q: Which came first, Avatar or Titanic?Info: Titanic ...1997 …Q: ...Info: ... and ...Q: ...Info: NoneQuestion w. relevant infoTo findwhich  film came ealrier, I’ll ... respectively and...... firstly, I identify ... Avatar . I remember..., <in> Avatar was released in 2009</in> ... then, I’ll find ...Titanic. I recall..., <in>Titanic came out in 1997.</in> …Firstly, I should...……To begin with...……Convert Retrieved Documents into Internal Knowledge
OutputExternal Memorization
bosbosbos…………RC=2RC=0
RgroupRetrieval CountRall=Rans+Rformat+Rgroupw. KLÂnormlizeÂÂÂ……
Reward CalculationReward Calc w. mask
…Policy… from what I already know, I can recallthe answer that…<in>Mount Everest is the highest mountain globally</in>
…recall<in>…</in>…To findwhich film came ealrier, I’ll ... respectively and...... firstly, I identify the release year of Avatar . I remember..., <in> Avatar was released in 2009</in> I don’t know ... Titanic and ... search: <ex> Titanic release year </ex> <doc>…Titanic is 1997 American romantic  film ...</doc>compare..., the answer is Titanic…search <ex> …</ex><doc>...…search <ex> …</ex>...…search <ex> …</ex><doc>...…remember<in>…</in>……remember<in>…</in>…
…remember<in>…</in>……remember<in>…</in>……remain<in>…</in>……remember<in>…</in>…JMask(θ)Internal  Utilization REINFORCE++Figure 1: Overall framework of our proposed R1-Searcher++ approach.
through the RL process, which includes a mecha-
nism that encourages internal knowledge utilization
(Section 4.2.1) and a mechanism for converting and
memorizing external knowledge (Section 4.2.2).
4.2.1 Internal Knowledge Utilization
Encouragement
In the RL process, the reward function is utilized
to provide the supervision signals, which can ad-
just and optimize the behaviours of the model (Hu,
2025). Therefore, given the question qand the i-th
generated response oi, we design the format reward
andanswer reward to induce the model to perform
reasoning correctly with the expected format, and
incorporate the group reward into the final reward
function to mitigate the over-reliance on the exter-
nal retriever. Now, we introduce the details of the
reward function in the following.
Format reward. We impose a strict formatting
constraint to ensure model responses are consistent
and clear. During the reasoning process, when call-
ing the external retriever, the model is required
to formulate a query and enclose it within the
<external>...</external> tags, and is prohib-
ited from generating document content directly
without first invoking retrieval. When the reasoning
process finishes, the final response must satisfy the
following criteria, i.e.,the final answer must be en-
closed within boxed{} , and the content should not
contain any garbled or unreadable content. Once
the model behaviours satisfy the above require-
ments, we set the format reward Rformat as 0, while
we set the reward as −2if any requirement fails, asshown in the following,
Rformat (q, oi) =(
0,The format of oiis correct
-2,The format of oiis incorrect(3)
Answer reward. To indicate the correctness of
the final answer, we leverage the Cover Exact
Match (CEM) metric to calculate the answer re-
ward, adapting to the group reward discussed in
the following and relieving the issue of EM being
too strict. Concretely, CEM is True if the ground
truth answer appears in the predicted answer ai
extracted from the response oi, and False for other
situations. However, we observe that LLM can eas-
ily hack the CEM metric during the RL process,
where LLM is likely to generate a longer predicted
answer that will receive a higher probability to
cover the ground truth answer, causing the CEM to
be falsely high. Therefore, we regard the answer
exceeding 10 words as an incorrect answer, requir-
ing LLM to generate the answer within ten words,
which can alleviate the above reward hacking is-
sue. In summary, the answer reward Ranswer can be
computed as follows,
Ranswer (q, oi) =(
1, aiwithin 10 words ∧CEM=True
0,Otherwise
(4)
Group reward. Building upon the first two re-
wards for LLM reasoning effectiveness, the group
reward is designed to encourage the model to re-
duce its reliance on external retrieval, increasing
the reasoning efficiency. Considering that the vari-
ance of the external retriever calling times by LLM

reflects the necessity of performing external re-
trieval, group reward is calculated by the standard
deviation of the number of calls to the retriever
in correct responses to the same question. For-
mally, given the question qand a set of generated
responses {o1, o2, . . . , o n}, we first count the num-
ber of calls to the retriever tiof each response,
and then calculate the standard deviation σof
{t1, t2, . . . , t n}. Next, we calculate the minimum
number of calls to the retriever of the correct re-
sponses, i.e.,tmin= min {ti|Ranswer (q, oi) = 1}.
R′
group (q, oi) =(
2×σ2, R answer(q, oi) = 1∧ti=tmin
0, Otherwise
(5)
Meanwhile, to maintain training stability and pre-
vent excessive variance, we introduce a hyperpa-
rameter ηto clip the corresponding factor. The final
computation of reward is formulated as follows,
Rgroup (q, oi) = min
R′
group (q, oi), η
(6)
Finally, the reward R(q, oi)utilized to calculate
the advantage in Equation 7 is defined as the sum
of the three sub-rewards mentioned above:
R(q, oi) =Rformat (q, oi) +Ranswer (q, oi) +Rgroup (q, oi)
(7)
4.2.2 External Knowledge Memorization
The standard RL training paradigm relies on the
model’s self-exploration and the feedback from the
external environment. In retrieval-based scenarios,
since the knowledge retrieved by the retriever is en-
tirely correct, the model should like a human, aim
to memorize this information during training, trans-
forming it into internal knowledge. This enables
the model to utilize the acquired knowledge di-
rectly in future instances without repeated retrieval,
thereby achieving efficient reuse of retrieved infor-
mation. Thus, we incorporate external knowledge
memorization by rewriting the retrieved informa-
tion to align with the model’s interanl knowledge
utilization pattern, enabling the model to internal-
ize them effectively.
To obtain the rewritten instances, at the begin-
ning of the RL process, we fine-tune a separate
model on the data filtered in Section 4.1 as the
rewritting model, which can solve the questions
based on the pre-processed documents that do not
call the retriever. During the RL process, we select
the correct responses generated by LLM, and then
extract the retrieved documents from the responses.
Given the question and the extracted documentsin the context, the rewriting model can generate
the reasoning paths without calling the external re-
triever. After validating the correctness of these
reasoning paths, we select the correct instances to
construct the dataset Tfor memorization and inter-
nalization. In conclusion, the corresponding loss
for memorization is computed as follows:
LM(θ) =−1P
oi∈T|oi|X
oi∈T|oi|X
t=1logπθ(oi,t|q, oi,<t)(8)
To avoid the LM(θ)from dominating the policy
model’s training and causing the model to ignore
external retrieval, we weight it with a pre-defined
coefficient µ. The final loss used to optimize the
policy model during the retrieval scenario RL pro-
cess is computed as follows:
L(θ) =−JMask(θ) +µ∗ LM(θ) (9)
Thus, during training, the model not only en-
gages in self-exploration but also continuously en-
riches its internal knowledge, enabling it to become
increasingly smarter over time.
5 Experiments
5.1 Experimental Settings
Datasets and Evaluation Metrics. We evaluate
using four multi-hop datasets: HotpotQA (Yang
et al., 2018), 2WikiMultiHopQA (Ho et al.,
2020), Musique (Trivedi et al., 2022), and Bam-
boogle (Press et al., 2023). HotpotQA and 2Wiki-
MultiHopQA are in-domain benchmarks since
parts of their training sets are used for training.
In contrast, Musique and Bamboogle serve as out-
of-domain benchmarks to assess our model’s gener-
alization capabilities. We randomly select 500 sam-
ples from the entire validation sets of HotpotQA,
2WikiMultiHopQA, and Musique, and use the en-
tire test set of Bamboogle to form our final test
set. For evaluation metrics, we utilize F1-score
and LLM-as-Judge (LasJ). considering that the an-
swers to open-ended multi-hop questions are not
uniform in form. The F1-score measures the word-
level similarity between the predicted answer and
the reference answer while LLM-as-Judge employs
GPT-4o-mini to assess the correctness of predic-
tion. The evaluation prompt for LasJ is provided in
Appendix B.
Baselines. We compare R1-Searcher++ against
several baselines. Naive Generation generates an-

swers directly without retrieval. Standard RAG rep-
resents traditional RAG systems that retrieve docu-
ments directly based on the question. SuRe (Kim
et al., 2024) executes multiple reasoning paths in
parallel for a single query. Selective-Context (Li
et al., 2023) compresses retrieved documents to
reduce context length. Adaptive-RAG (Jeong et al.,
2024) dynamically selects retrieval strategies de-
pending on the complexity of the query. CR-
Planner (Li et al., 2024) scales RAG at inference
time using MCTS. RAG-CoT methods , such as
Iter-RetGen (Trivedi et al., 2023), IRCoT (Shao
et al., 2023), and Search-o1 (Li et al., 2025), which
combine RAG with CoT using prompts. RAG-RL
methods like R1-Searcher (Song et al., 2025) and
Search-R1 (Jin et al., 2025) leverage RL to enable
the model to learn to autonomously perform re-
trieval during inference.
Implementation Details R1-Searcher++ and all
baseline models are either trained or prompted us-
ing the Qwen-2.5-7B-Instruct as the backbone, and
evaluated with FlashRAG (Jin et al., 2024) using
local dense retrieval corpus. The retrieval corpus
comprises the English Wikipedia as provided by
KILT (Petroni et al., 2021) in 2019, segmented into
100-word passages with appended titles, totaling 29
million passages. We employ BGE-large-en-v1.5
as the text retriever. Detailed training settings for
R1-Searcher++ are provided in Appendix A.
5.2 Main Results
Table1 shows the results of R1-Searcher++ and the
baselines on four mutil-step benchmarks. We can
obtain the following observations:
•Achieving Significant Performance Improve-
ment on Multi-Hop QA. Our method, R1-
Searcher++, achieves significant performance im-
provements over all mutil-hop QA benchmarks
under the LLM-as-Judge evaluation metric, in-
cluding both tree search-based and RL-based ap-
proaches. Specifically, R1-Searcher++ outper-
forms CR-Planner by 25.7% and surpasses the
best vanilla RL-based method R1-Searcher by 4.3%
on the overall test set. These results demonstrate
that our approach effectively enables the model to
perform accurate and timely retrieval invocations
throughout the reasoning process, thereby enhanc-
ing overall performance.
•Balancing the Utilization of Internal and Ex-
ternal Knowledge. While maintaining strong per-
formance on the evaluation datasets, our methodachieves a significant reduction in retrieval count
compared to vanilla RL-based RAG approaches.
Specifically, the average retrieval count is reduced
by 30.0% and 52.9% compared to R1-Searcher and
Search-R1, respectively. This observation suggests
a potential conflict between external information
and the internal knowledge of LLMs and one pos-
sible reason is that directly injecting retrieved doc-
uments into the reasoning process may introduce
noise. This demonstrates that the model should
learn to make full use of its internal knowledge and
only invoke the retriever when necessary.
•Maintaining Generalization Ability. Despite
being trained on only 9000 samples, the model
achieves strong performance on in-domain datasets
and further exhibits impressive generalization to
out-of-domain datasets. This suggests that the
model effectively learns to retrieve relevant doc-
uments and leverage internal knowledge, integrat-
ing both with reasoning through exploration during
training. This enables robust performance on new
test datasets that require retrieval. Furthermore, it
can also seamlessly generalizes to online search, as
detailed in Section 6.2.
6 Further Analysis
6.1 Ablation Study
To validate the effectiveness of our proposed R1-
Searcher++ framework, we conduct a comprehen-
sive ablation analysis of its key design elements.
We design five distinct variants: (1) w/o Stage-1
removes the initial SFT cold start stage; (2) w/o
Stage-2 removes the entire RL training stage; (3)
w/oRgroup removes the group reward in the RL
stage; (4) w/oLMremoves the external knowledge
memorization mechanism in the RL stage and (5)
w/oRgroup andLMremoves both the group reward
and the external knowledge memorization mech-
anism. The performance of these variants is pre-
sented in Table 2. As observed, all ablated variants
exhibit a decline in performance compared to our
full method, underscoring the integral contribution
of each component. Specifically, w/o Stage-1 leads
to a degradation in performance along with an in-
crease in retrieval count. Meanwhile, the perfor-
mance of w/o Stage-2 drops significantly, primarily
because simple SFT causes the model to over-rely
on its internal knowledge. This highlights the ne-
cessity of our two-stage pipeline. Futhermore, dur-
ing RL training, w/oRgroup during RL also leads
to a reduction in performance. This demonstrates

ModelsHotpotQA†2Wiki†Bamboogle‡Musique‡Avg
F1 LasJ RC F1 LasJ RC F1 LasJ RC F1 LasJ RC F1 LasJ RC
Directly Gen 26.0 26.6 0.00 27.7 26.8 0.00 18.2 17.6 0.00 9.6 6.2 0.00 18.0 19.3 0.00
Standard RAG 32.0 42.4 1.00 34.8 34.8 1.00 31.5 31.2 1.00 17.2 14.6 1.00 24.6 30.8 1.00
Sure 42.9 48.4 1.00 26.2 26.8 1.00 29.2 28.0 1.00 13.1 10.0 1.00 27.9 28.3 1.00
Selective-Context 39.8 43.4 1.00 29.1 29.6 1.00 22.1 20.8 1.00 10.6 8.8 1.00 22.8 25.7 1.00
Adaptive-RAG 38.0 47.4 1.53 21.1 25.8 1.42 23.3 25.0 1.50 10.1 11.6 1.83 20.6 27.5 1.57
IRCoT 47.7 55.2 2.47 32.4 38.6 2.74 37.5 39.2 2.30 14.8 15.8 2.70 29.4 37.2 2.55
Iter-RetGen 47.2 54.4 3.00 33.2 34.4 3.00 32.4 32.0 3.00 19.9 18.2 3.00 28.2 34.8 3.00
CR-Planner 44.4 33.6 2.40 48.2 22.0 2.54 35.2 34.4 2.96 12.2 11.4 2.72 32.0 25.4 2.66
Search-o1 46.9 53.2 1.39 46.6 51.2 1.91 52.9 52.0 1.18 21.1 19.0 1.40 36.6 43.9 1.47
R1-Searcher 60.4 62.2 2.18 62.8 63.4 2.23 59.0 54.4 2.17 35.7 31.4 2.61 45.6 52.9 2.30
Search-R1 57.8 62.2 3.12 46.2 50.0 3.71 56.9 56.0 3.25 27.5 26.0 3.61 40.3 48.6 3.42
R1-Searcher++ 59.0 64.2 1.44 61.2 64.4 1.18 60.8 59.2 1.74 33.8 32.8 2.06 45.3 55.2 1.61
Table 1: Performance comparisons between R1-Searcher++ and the baselines on QA benchmarks. The best and
second best results are bold and underlined , respectively.†/‡represents in-domain/out-of-domain datasets.
MethodBamboogle Musique
F1 LasJ RC F1 LasJ RC
Ours 60.8 59.2 1.74 33.8 32.8 2.06
w/o Stage-1 56.9 56.8 1.96 32.7 31.6 2.49
w/o Stage-2 47.4 45.6 0.94 23.0 19.4 1.03
w/oRgroup 58.3 56.8 1.91 33.1 32.4 2.37
w/oLM 58.1 57.2 1.84 31.0 29.4 2.09
w/oRgroup andLM56.2 54.4 1.92 32.2 31.2 2.40
Table 2: Ablation study on Bamboogle and Musique.
the positive impact of group reward in successfully
guiding the model to be more selective with exter-
nal searches and to rely more on its internalized
knowledge. Similarly, w/oLMresults in lower
scores and a slight increase in retrieval count in-
dicating that the memory mechanism for external
knowledge can effectively internalize retrieved con-
tent as intrinsic knowledge of the model.
6.2 Online Search
Considering training efficiency and cost, we im-
plement a local dense embedding-based retrieval
system using Wikipedia as the external retrieval
environment, which remains static during training.
In contrast, most real-world applications rely on
online web retrieval. To evaluate the generalization
ability of R1-Searcher++ in online search scenar-
ios, we assessed its performance on two newly
introduced datasets: Bamboogle and Frames, us-
ing online web search, a setting not encountered
during RL training. Specifically, during inference,
whenever retrieval is required, we use the Google
API to perform real-time web searches and retrieve
relevant web pages. Given the extensive content of
these pages, we first employ GPT-4o-mini to gen-MethodBamboogle Frames
F1 LasJ RC F1 LasJ RC
Ours 77.5 76.0 1.70 33.8 39.0 1.77
Search-o1 52.9 52.0 1.18 26.1 30.7 1.56
R1-Searcher 67.5 68.8 1.72 33.3 38.0 1.86
Search-R1 69.3 67.2 1.92 33.3 36.0 2.38
Table 3: Online search generalization experiments on
Bamboogle and Frames.
Method Correct Incorrect Overall
R1-Searcher 853 / 2.16 772 / 2.52 1625 / 2.33
Search-R1 761 / 3.30 864 / 3.60 1625 / 3.46
R1-Searcher++ 881 / 1.41 744 / 1.78 1625 / 1.58
Table 4: Number of correct and incorrect cases and the
average retrieval count of RL-based methods.
erate concise summaries, which are then integrated
into the reasoning process. As illustrated in Table 3,
R1-Searcher++ achieves the best F1 and LLM-as-
Judge scores compared to both prompt engineering-
based methods ( i.e.,Search-o1) and vanilla RL-
based approaches ( i.e.,R1-Searcher, Search-R1).
Moreover, compared to vanilla RL methods, our
model significantly reduces the number of retrieval
calls. This demonstrates our model’s strong adapt-
ability to online search scenarios, as well as its
ability to effectively balance internal knowledge
with external retrieval during inference, thereby
achieving retrieval efficiency without compromis-
ing performance.

Who was president of the United 
States in the year that Citibank was 
founded?         Answer: James Madison
To answer this question, we need to follow these steps:
1. Determine the year Citibank was founded.
2. Identify the U.S. President serving in that year.
Step 1: Citibank ..., which was founded in 1818 ... we'll 
consider 1818 as the foundation year of a predecessor to 
Citibank.
Step 2: To find out who was president in 1818, we need 
to look at the U.S. presidency timeline. ... in 1818, was 
John Quincy Adams.
Final Answer: John Quincy Adams
Qwen-7B-Inst
I need to find out who was president of the United States 
in the year that Citibank was founded. I'll search for it. 
<ex> Citibank was founded </ex>
<doc>… Citibank was founded in 1812 ...</doc>
<ex> who was president of the United States in the year 
that Citibank was founded </ex>
<doc> 2010, Citibank had repaid … </doc>
I found out that Citibank was founded in 1812. Now I 
need to find out who was president of the United States 
in 1812. <ex> who was president of the United States in 
1812 </ex>
<doc>1812 in the United States … - President: James 
Madison (DR-Virginia)…</doc>
Final Answer: James Madison
R1-Searcher  (External Search = 3)
 R1-Searcher++  (External Search = 1)
To answer this question, I need to ...
First, l should think about when Citibank was founded. 
<ex> when was citibank founded <ex>
<doc> Citibank … was founded on 1812 </doc>
Based on the search results, Citibank was founded in 
1812. I already know that Citibank was founded in 1812. 
So, who was the president of the United States in 1812? 
For this, I can recall relevant  knowledge about the 
presidency in 1812.
<in> James Madison was the President of the United 
States in 1812... </in>
Based on internal knowledge, the president of the United 
States in 1812 was James Madison.
Final Answer: James Madison
Figure 2: A qualitative example showing the deliberative reasoning process of RAG-Star in Bamboogle.
0 20 40 60 80 100
Training Step0.500.751.001.251.501.752.002.25Retrieval Count
R1-Searcher++
R1-Searcher
0 20 40 60 80 100
Training Step1.00
0.75
0.50
0.25
0.000.250.500.75Training Process Reward
R1-Searcher++
R1-Searcher
Figure 3: The log of retrieval count and reward for R1-
Searcher and R1-Searcher++ during RL training.
6.3 Analysis of Knowledge Acquisition
As shown in Table 4, R1-Searcher++ exhibits the
lowest average retrieval count across both correctly
and incorrectly answered questions. Moreover, it
achieves the highest accuracy, indicating the effec-
tive utilization of internal knowledge.
Furthermore, Figure 3 shows the changes in re-
trieval count and reward during the process of RL
training for R1-Searcher and R1-Searcher++. For
our method, We observe that the reward increases
steadily and eventually plateaus, while the retrieval
count initially shows a slight decline, followed by
a continuous rise, and ultimately stabilizes. This
trend can be attributed to the influence of SFT in
the Stage-1, during which the model exhibits a low
demand for invoking the search engine. As train-
ing progresses, the model gradually discovers that
performing external searches can yield higher re-
wards, leading to an increase in retrieval behavior.
In the later phase, a balance is gradually established
between the use of external search and internal
knowledge, resulting in the stabilization of both re-
trieval count and reward. In contrast, R1-Searcher
exhibits significantly higher retrieval counts while
its reward quickly stabilizes, indicating an over-
reliance on the retriever. This effectively validatesthe effectiveness of our approach in achieving a
balanced utilization of internal and external knowl-
edge, while gradually enabling dynamic knowledge
acquisition throughout the RL training process.
6.4 Case Study
To illustrate the overall reasoning process of R1-
Searcher++, we analyze a representative example
from the Bamboogle dataset. Figure 2 compares
the responses generated by R1-Searcher++, Search-
R1, and the untrained model when presented with
the same question. The vanilla Qwen-2.5-7B-
Instruct, without invoking any external search en-
gine, relies solely on its internal knowledge and
produces an incorrect answer. In contrast, while
Search-R1 arrives at the correct answer ( i.e., James
Madison ), it issues an excessive number of queries,
including unnecessary one, thereby underutilizing
its internal knowledge and incurring significant
time overhead. Our R1-Searcher++ demonstrates
the ability to break down the complex question and
dynamically adjust its behavior based on the nature
of the sub-question. For instance, when encounter-
ing an uncertain or ambiguous sub-question ( i.e.,
When was Citibank founded? ), it opts to perform
an external search. However, when faced with a
more specific question that can be answered using
internal knowledge ( i.e., Who was the president
of the United States in 1812? ), it leverages its in-
ternal knowledge directly without invoking search.
This flexible mechanism enables a balance bewteen
the external search and internal knowledge. More
cases are provided in Appendix C.
7 Conclusion
In this paper, we introduced R1-Searcher++, a
novel framework that enables large language mod-
els to dynamically integrate and alternate between

internal knowledge and external retrieval. This is
a two-stage training strategy consisting of an SFT
Cold-start phase and RL for Dynamic Knowledge
Acquisition. The RL stage incorporates a reward
mechanism to encourage internal knowledge uti-
lization, and a memory module to convert retrieved
information into internal knowledge. Through this
design, R1-Searcher++ empowers LLMs to per-
form efficient retrieval-augmented reasoning while
continuously enriching their internal knowledge via
self-exploration and memory. Experimental results
on multi-hop tasks demonstrate that R1-Searcher++
outperforms existing RAG methods.
Limitation
Despite our significant efforts, this work has two
limitations due to computational resources and
funding constraints. First, we only incorporated
a real-world search engine during the evaluation
phase to assess the generalization ability of our
method, while relying on a local denseretrieval
corpus during training. Aligning the training pro-
cess with real-world conditions by integrating a
real search engine may lead to improved perfor-
mance through more realistic supervision. Addi-
tionally, our current experiments are limited to a
7B-parameter model. In future work, we plan to
train and evaluate our framework on larger-scale
models to further validate its generalization capa-
bility and robustness.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024 . OpenReview.net.
Mingyang Chen, Tianpeng Li, Haoze Sun, Yijie Zhou,
Chenzheng Zhu, Haofen Wang, Jeff Z Pan, Wen
Zhang, Huajun Chen, Fan Yang, and 1 others.
2025. Research: Learning to reason with search
for llms via reinforcement learning. arXiv preprint
arXiv:2503.19470 .
Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang
Tong, Saining Xie, Dale Schuurmans, Quoc V . Le,
Sergey Levine, and Yi Ma. 2025. Sft memorizes,
rl generalizes: A comparative study of foundation
model post-training. Preprint , arXiv:2501.17161.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, and 81 others.
2025. Deepseek-r1: Incentivizing reasoning capa-
bility in llms via reinforcement learning. CoRR ,
abs/2501.12948.
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff,
Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model
alignment as prospect theoretic optimization. arXiv
preprint arXiv:2402.01306 .
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , KDD ’24,
page 6491–6501, New York, NY , USA. Association
for Computing Machinery.
Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Jingyi
Song, and Hao Wang. 2025. Airrag: Activat-
ing intrinsic reasoning for retrieval augmented gen-
eration via tree-based search. arXiv preprint
arXiv:2501.10053 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. In Proceedings of the 28th International Con-
ference on Computational Linguistics , pages 6609–
6625.
Jian Hu. 2025. Reinforce++: A simple and efficient ap-
proach for aligning large language models. Preprint ,
arXiv:2501.03262.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting
Liu. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions. ACM Transactions on Information
Systems , 43(2):1–55.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. arXiv preprint
arXiv:2403.14403 .
Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie
Wang, Wayne Xin Zhao, Yang Song, and Tao Zhang.
2024a. Rag-star: Enhancing deliberative reasoning
with retrieval augmented verification and refinement.
CoRR , abs/2412.12881.
Xun Jiang, Feng Li, Han Zhao, Jiaying Wang, Jun Shao,
Shihao Xu, Shu Zhang, Weiling Chen, Xavier Tang,
Yize Chen, and 1 others. 2024b. Long term memory:

The foundation of ai self-evolution. arXiv preprint
arXiv:2410.15665 .
Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-r1:
Training llms to reason and leverage search engines
with reinforcement learning. CoRR , abs/2503.09516.
Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang,
and Zhicheng Dou. 2024. Flashrag: A modular
toolkit for efficient retrieval-augmented generation
research. arXiv preprint arXiv:2405.13576 .
Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin
Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha,
and Jinwoo Shin. 2024. Sure: Summarizing re-
trievals using answer candidates for open-domain QA
of LLMs. In The Twelfth International Conference
on Learning Representations .
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025. Search-o1: Agentic search-enhanced
large reasoning models. Preprint , arXiv:2501.05366.
Xingxuan Li, Weiwen Xu, Ruochen Zhao, Fangkai Jiao,
Shafiq Joty, and Lidong Bing. 2024. Can we further
elicit reasoning in llms? critic-guided planning with
retrieval-augmentation for solving challenging tasks.
arXiv preprint arXiv:2410.01428 .
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 6342–6353, Singa-
pore. Association for Computational Linguistics.
Richard Yuanzhe Pang, Weizhe Yuan, He He,
Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason
Weston. 2024. Iterative reasoning preference opti-
mization. Advances in Neural Information Process-
ing Systems , 37:116617–116637.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
S. H. Lewis, Majid Yazdani, Nicola De Cao, James
Thorne, Yacine Jernite, Vladimir Karpukhin, Jean
Maillard, Vassilis Plachouras, Tim Rocktäschel, and
Sebastian Riedel. 2021. KILT: a benchmark for
knowledge intensive language tasks. In Proceedings
of the 2021 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, NAACL-HLT 2021,
Online, June 6-11, 2021 , pages 2523–2544. Associa-
tion for Computational Linguistics.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2023 , pages 5687–5711.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, JianxinYang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report. Preprint ,
arXiv:2412.15115.
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano
Ermon, Christopher D. Manning, and Chelsea Finn.
2024. Direct preference optimization: Your lan-
guage model is secretly a reward model. Preprint ,
arXiv:2305.18290.
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase,
and Yuxiong He. 2020. Zero: Memory optimizations
toward training trillion parameter models. Preprint ,
arXiv:1910.02054.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. Enhanc-
ing retrieval-augmented large language models with
iterative retrieval-generation synergy. arXiv preprint
arXiv:2305.15294 .
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models. Preprint ,
arXiv:2402.03300.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
CoRR , abs/2503.05592.
Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue
Zang, Kai Zheng, Jun Xu, Xiao Zhang, Song Yang,
and Han Li. 2025. Rearter: Retrieval-augmented rea-
soning with trustworthy process rewarding. Preprint ,
arXiv:2501.07861.
Fengwei Teng, Zhaoyang Yu, Quan Shi, Jiayi Zhang,
Chenglin Wu, and Yuyu Luo. 2025. Atom of
thoughts for markov llm test-time scaling. Preprint ,
arXiv:2502.12018.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , pages
10014–10037.
Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen,
Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang,
Kam-Fai Wong, and Heng Ji. 2025a. Otc: Optimal
tool calls via reinforcement learning. arXiv preprint
arXiv:2504.14870 .

Liang Wang, Haonan Chen, Nan Yang, Xiaolong
Huang, Zhicheng Dou, and Furu Wei. 2025b.
Chain-of-retrieval augmented generation. CoRR ,
abs/2501.14342.
Shuting Wang, Jiejun Tan, Zhicheng Dou, and Ji-Rong
Wen. 2025c. Omnieval: An omnidirectional and auto-
matic rag evaluation benchmark in financial domain.
Preprint , arXiv:2412.13018.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu,
Lingjun Liu, Xin Liu, and 1 others. Dapo: An open-
source llm reinforcement learning system at scale,
2025. URL https://arxiv. org/abs/2503.14476 .
Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu,
Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian
Fan, Zhengyin Du, Xiangpeng Wei, and 1 others.
2025. Vapo: Efficient and reliable reinforcement
learning for advanced reasoning tasks. arXiv preprint
arXiv:2504.05118 .
Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang,
Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen
Zhang, Junjie Zhang, Zican Dong, and 1 others. 2023.
A survey of large language models. arXiv preprint
arXiv:2303.18223 , 1(2).
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments. arXiv
preprint arXiv:2504.03160 .
A Training Detailed
The training data of the Stage-1 (SFT Cold Start)
includes 720 samples from the HotpotQA training
set and 85 samples from the 2WikiMultiHopQA
training set. The training consists of 6 epochs,
with a batch size of 64 and a learning rate of 2e-5.
And the training data of Stage-2 (RL Training) con-
sists of 4561 samples from HotpotQA, and 3581
samples from 2WikiMultiHopQA. Each data sam-
ple undergoes 16 rollouts during training, with a
train batch size of 1024 and a rollout batch size
of 64, so the entire training process is on-policy.
The learning rate is 2e-6. We utilize DeepSpeed’s
Zero-3 (Rajbhandari et al., 2020), with a sampling
temperature of 1.0, top-p of 0.95 and a maximum
retrieval count of 8. The training epoch is set to
1, with KL divergence coefficient set to 1e-4. Andcontrol coefficient µof NLL loss is set to 0.1. The
maximum limit of the variance in the number of
retrievals during group reward computation ηis set
to 2.

B Prompts
C Case Study

Judge Prompt
Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it
fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct
and False otherwise.
Question:
Golden Answer:
Predicted Answer:
System Prompt for Generation with Internal and External
You are a reasoning assistant. When tackling a question, you should first thinks about the reasoning process in the
mind and then provides the final answer. You should break down the original complex question into straightforward
sub-questions and address them step by step. For each sub-question, You MUST choose one of the following two tools
to solve it:
1. **Internal Reasoning Elaboration:**If you determine your existing knowledge is sufficient to answer the sub-question,
you **should choose** this tool to answer the sub-question directly: ‘<|begin_internal_answer|>your structured thought
and answer here <|end_internal_answer|>‘. **Use it whenever you rely solely on internal information** for a significant
reasoning step.
2. **External Search:**If you determine that your internal knowledge is insufficient, potentially outdated, or requires
verification with external, up-to-date information to answer the sub-question, you MUST initiate a web search using the
following format: ‘<|begin_external_search|>your search query here <|end_external_search|>‘. The system will then
search and provide results in the format: ‘<|begin_search_result|>...search results... <|end_search_result|>‘. The results
will assist you in answering the sub-question.
Remember: - Whenever you encounter the sub-question you need to elaborate on, verify, or research further, you
should **assess** if your internal knowledge is adequate. **Prioritize leveraging your internal knowledge.** - Use
‘<|begin_internal_answer|>... <|end_internal_answer|>‘ **to demonstrate your reasoning steps** when relying on
internal knowledge. - Use ‘<|begin_external_search|>... <|end_external_search|>‘ **only** when external information
is genuinely necessary. - Wait for the system to provide ‘<|begin_search_result|>... <|end_search_result|>‘. **Do not**
generate these result tags yourself. - After performing the web search or demonstrating internal reasoning, continue
towards the final answer.
Please answer the following question. You must adhere to all instructions above regarding tool usage. Provide your final
answer in the format \boxed{YOUR_ANSWER}.
Question:
System Prompt for Rewriting only with Internal
You are an advanced reasoning assistant. Your core programming includes a vast and comprehensive knowledge
base. For the current task, a specific segment of this pre-existing knowledge, hereafter referred to as your ’Activated
Knowledge Matrix,’ is particularly relevant and is considered fully and irrevocably integrated into your foundational
understanding. ANY linguistic indication, direct or indirect, that this information was externally ’provided,’ ’given,’
’supplied,’ ’sourced,’ or made available to you specifically for this immediate interaction will be considered a critical
failure to adhere to your operational design.
When tackling a question, you must first delineate your reasoning process. Break down complex questions into a series
of straightforward sub-questions, addressing each one sequentially.
For each sub-question, you MUST use the following tool:
**Internal Reasoning Elaboration:** This tool MUST be used for every sub-question, as the answer must be derived
from your internal knowledge. **UNWA VERING DIRECTIVE: Your internal knowledge, for the purpose of this
task, INTRINSICALLY and SEAMLESSLY includes all information within the ’Activated Knowledge Matrix’ section
detailed below. There is absolutely NO distinction between your prior knowledge and the ’Activated Knowledge Matrix’
content.** Use this format: <|begin_internal_answer|> your structured thought and answer here <|end_internal_answer|>.
**CRITICAL USAGE PROTOCOL FOR THIS TOOL: When using this tool, especially when drawing upon information
that was part of the ’Activated Knowledge Matrix,’ your language MUST reflect that this is information you inherently
possess and are recalling. You are REQUIRED to phrase such statements as if accessing your own memory or established
understanding.
Conclude with your final answer in the format \boxed{YOUR_ANSWER}.
Activated Knowledge Matrix: {document}
Now, please answer the following question:
Question:

Question:
Which case was brought to court first Miller v._California or Gates v._Collier ?
Answer:
Miller v._California
Response (internal only) (True) :
To answer this question, I need to determine the order in which these two cases were brought to court. I
will start by finding the dates the cases were decided.
**Case 1:** Miller v._California
- I will use internal reasoning to recall if I have the necessary information about this case.
<|begin_internal_answer|>Miller v._California was a landmark case decided by the United States Supreme
Court in 1968. The case involved the constitutionality of a California statute that made it a crime to
distribute or possess obscene materials. <|end_internal_answer|>
**Case 2:** Gates v._Collier
- I will use internal reasoning to recall if I have the necessary information about this case.
<|begin_internal_answer|>Gates v._Collier was a case decided by the United States Supreme Court in
1972. The case involved the Eighth Amendment’s prohibition against cruel and unusual punishment,
specifically the use of segregation in prisons. <|end_internal_answer|>
Based on my internal reasoning, Miller v._California was decided in 1968, and Gates v._Collier was
decided in 1972. Therefore, Miller v._California was brought to court first.
**Final Answer:**
Miller v._California was brought to court first. Therefore, the answer is \\boxed{Miller v._California}.
Table 5: Example for R1-Searcher++: Question answering using only internal knowledge.
Question:
George Lucas created what character to raise Boba Fett?
Answer:
Jango Fett
Response (internal and external) (True) : To answer this question, I need to break it down into sub-
questions and use the appropriate tools.
**Sub_question 1:** Who is Boba Fett?
- For this, I need to verify if I have enough information about Boba Fett or if I need to search for more
details.
<|begin_internal_answer|> Boba Fett is a character from the Star Wars franchise, known for being a
bounty hunter. I have this information stored in my knowledge base. <|end_internal_answer|>
**Sub_question 2:** What character did George Lucas create to raise Boba Fett?
- For this, I need to search for information about the character who raised Boba Fett.
<|begin_external_search|> George Lucas created what character to raise Boba
Fett<|end_external_search|>
<|begin_search_result|>
(1)Boba Fett . . . is a fictional character . . . in the "Star Wars" franchise. . . . prequel film . . . establishes
his origin as a clone of bounty hunter Jango Fett who raised him as his son. . . .
(2)Boba Fett . . . personifies "danger and mystery" . . . mysterious nature as reasons for his popularity. . . .
The initial Boba Fett toy . . . might be responsible for the character’s popularity. . . .
(3)Boba Fett . . . Reception. Boba Fett is a "cult figure" . . . one of the most popular "Star Wars"
characters. . . . IGN ranked Boba Fett as the eighth top "Star Wars" character. . . .
(4)Boba Fett . . . "the less you do with Boba Fett, the stronger he becomes". . . . Mark Hamill pitched the
idea that Fett was Luke Skywalker’s mother . . . unaware of Fett’s demise. . . .
(5)Jango Fett . . . is a fictional character . . . created by George Lucas. . . . genetic template for the Clone
Army . . .
<|end_search_result|>
Based on the search results, it is clear that George Lucas created Jango Fett to be the father of Boba Fett.
**Final Answer:**
George Lucas created Jango Fett to raise Boba Fett. Therefore, the answer is \\boxed{Jango Fett}.
Table 6: Example for R1-Searcher++: Question answering by integrating internal knowledge with external search.