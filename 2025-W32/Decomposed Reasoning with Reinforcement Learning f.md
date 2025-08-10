# Decomposed Reasoning with Reinforcement Learning for Relevance Assessment in UGC Platforms

**Authors**: Xiaowei Yuan, Lei Jin, Haoxin Zhang, Yan Gao, Yi Wu, Yao Hu, Ziyang Huang, Jun Zhao, Kang Liu

**Published**: 2025-08-04 15:14:09

**PDF URL**: [http://arxiv.org/pdf/2508.02506v1](http://arxiv.org/pdf/2508.02506v1)

## Abstract
Retrieval-augmented generation (RAG) plays a critical role in user-generated
content (UGC) platforms, but its effectiveness depends heavily on accurate
relevance assessment of query-document pairs. Despite recent advances in
applying large language models (LLMs) to relevance modeling, UGC platforms
present unique challenges: 1) ambiguous user intent due to sparse user feedback
in RAG scenarios, and 2) substantial noise introduced by informal and
unstructured language. To address these issues, we propose the Reinforced
Reasoning Model for Relevance Assessment (R3A), which introduces a decomposed
reasoning framework over queries and candidate documents before scoring. R3A
first leverages auxiliary high-ranked documents within the platform to infer
latent query intent. It then performs verbatim fragment extraction to justify
relevance decisions, thereby reducing errors caused by noisy UGC. Based on a
reinforcement learning framework, R3A is optimized to mitigate distortions
arising from ambiguous queries and unstructured content. Experimental results
show that R3A significantly outperforms existing baseline methods in terms of
relevance accuracy, across both offline benchmarks and online experiments.

## Full Text


<!-- PDF content starts -->

Decomposed Reasoning with Reinforcement Learning for Relevance
Assessment in UGC Platforms
Xiaowei Yuan1,2, Lei Jin3, Haoxin Zhang3, Yan Gao3,
Yi Wu3, Yao Hu3, Ziyang Huang1,2, Jun Zhao1,2, Kang Liu1,2,∗
1The Key Laboratory of Cognition and Decision Intelligence for Complex Systems,
Institute of Automation, Chinese Academy of Sciences
2School of Artificial Intelligence, University of Chinese Academy of Sciences
3Xiaohongshu Inc.
Abstract
Retrieval-augmented generation (RAG) plays
a critical role in user-generated content (UGC)
platforms, but its effectiveness depends heav-
ily on accurate relevance assessment of query-
document pairs. Despite recent advances in
applying large language models (LLMs) to rele-
vance modeling, UGC platforms present unique
challenges: 1) ambiguous user intent due to
sparse user feedback in RAG scenarios, and 2)
substantial noise introduced by informal and
unstructured language. To address these issues,
we propose the Reinforced Reasoning Model
for Relevance Assessment (R ³A), which intro-
duces a decomposed reasoning framework over
queries and candidate documents before scor-
ing. R ³A first leverages auxiliary high-ranked
documents within the platform to infer latent
query intent. It then performs verbatim frag-
ment extraction to justify relevance decisions,
thereby reducing errors caused by noisy UGC.
Based on a reinforcement learning framework,
R³A is optimized to mitigate distortions arising
from ambiguous queries and unstructured con-
tent. Experimental results show that R ³A signif-
icantly outperforms existing baseline methods
in terms of relevance accuracy, across both of-
fline benchmarks and online experiments.
1 Introduction
Retrieval-augmented generation (RAG) systems
have become a critical paradigm in information re-
trieval systems, enabling user search queries to be
answered with retrieved external knowledge (Ram
et al., 2023; Gao et al., 2024). At Xiaohongshu,
a prominent user-generated content (UGC) plat-
form featuring product reviews, travelogues, and
lifestyle narratives, RAG serves as an important
component of search systems. It can integrate both
search and summarization capabilities, enabling
the system to search through its billions of user
documents and generate concise answers (Li et al.,
2025b; Zhang et al., 2025).
Planning your Tokyo travel? I got you!! 
🗼Let me walk you through the must-visit spots, visa tips, and where to stay 
😎Weended up just chilling in Shibuya cafés most of the time 
☕
💕#TokyoVibesQ: Tokyo travel ?Highly Relevant!It mentions the travel plan, visit spots  and visa tips in Tokyo.visa? hotel? itineraries?
❌
⚠Ambiguous!
⚠Noisy!Figure 1: An illustrative example of an inaccurate
relevance assessment for a query–document pair.
The model is misled by informal phrasing and spuri-
ously relevant content, failing to recognize the lack
of substantive, query-specific information, which
ultimately results in an erroneous assessment.
A critical component of the RAG system is the
relevance assessment module. This module quanti-
tatively evaluates the semantic relevance between
user queries and retrieved documents (Thomas
et al., 2024), ensuring that the generation is ac-
curately grounded in the query-related context. Re-
cent advances in large language models (LLMs)
have opened up new possibilities for relevance mod-
eling by enabling fine-grained understanding of se-
mantics and intent (Faggioli et al., 2023; Zheng
et al., 2023), relying on prompting (MacAvaney
and Soldaini, 2023; Upadhyay et al., 2024b) or
supervised fine-tuning (SFT) methods (Zan et al.,
2023; Fitte-Rey et al., 2025).
However, applying such models to UGC plat-
forms introduces unique challenges. First , infer-
ring user search intent on UGC platforms is diffi-
cult due to the absence of traditional click-through
data. Unlike conventional systems that rely on
large-scale click logs to align relevance signals with
user behavior (Jiang et al., 2024), RAG on UGC
platforms typically receives feedback only at the
answer level rather than at the individual documentarXiv:2508.02506v1  [cs.IR]  4 Aug 2025

level. This limitation exacerbates the ambiguity of
user intent .Second , the widespread use of infor-
mal language, emotional expressions, emojis, and
off-topic content in UGC introduces substantial
noise , which significantly impairs model judgment
and frequently leads to inaccurate relevance as-
sessments. As illustrated in Figure 1, the model
incorrectly assigns a high relevance score to the
document. This error is primarily caused by spu-
riously relevant surface cues, such as " visit spots ",
"chilling in Shibuya ", and " #TokyoVibes ". Conse-
quently, the model overlooks the document’s lack
of substantive, query-relevant content, leading to
an inaccurate relevance assessment.
To address the above challenges, we propose a
novel Reinforced Reasoning Model for Relevance
Assessment (R ³A), which performs decomposed
reasoning based on reinforcement learning (RL)
algorithm. We argue that generating high-quality
relevance assessment in UGC scenarios requires
strong reasoning capabilities to address the chal-
lenges of ambiguous query intent and noisy content.
To better capture the user intent, the model input
is augmented with a set of auxiliary in-platform
high-ranked documents retrieved using the same
query. These additional documents provide con-
texts to help the model parse the user’s likely intent.
Furthermore, to mitigate the impact of noise, the
model is required to extract the most relevant frag-
ment from the candidate document, constrained to
be verbatim excerpts from the original text. This
constraint helps model reduce noise-induced errors
and ground its assessment maximally in the docu-
ment.
Experimental results demonstrate that R ³A out-
performs all baseline models in relevance assess-
ment on our real-world industry dataset NoteRel.
Moreover, the distilled R ³A-1.5B model exceeds
the performance of the larger 7B model by 1.7% in
accuracy and significantly outperforms competing
methods in online A/B testing.
The contributions of this paper are as follows:
•To tackle the unique challenges on the UGC
platform, this paper proposes R ³A that per-
forms decomposed reasoning over both am-
biguous query and the noisy document. This
approach enables the model to better infer the
user intent and reduce erroneous outputs.
•On the real-world industry dataset, R ³A con-
sistently outperforms all baselines, exhibitingstronger sensitivity to relevance classification
boundaries and improved accuracy.
•The distilled R ³A-1.5B model outperforms
prior methods in online A/B testing, demon-
strating the practical effectiveness of proposed
R³A method.
2 Decomposed Reasoning for Relevance
Assessment
This paper proposes the Reinforced Reasoning
Model for Relevance Assessment (R ³A) method
for UGC platforms, which enhances the reasoning
capabilities of relevance modeling based on RL
algorithm.
The overall framework of R ³A is illustrated in
Figure 2. After a cold-start initialization, the RL
training procedure involves a two-round interac-
tion between the model and the environment (in-
platform documents). In the first round , a set
of auxiliary in-platform documents d′is provided
alongside the user query qto support the model in
inferring latent query intent. In the second round ,
the model evaluates the candidate document dfor
relevance. During this stage, the model extracts
query-relevant fragment from dto minimize inac-
curate outputs and ground its assessment in seman-
tically aligned content.
2.1 Cold Start
Following previous work (Guo et al., 2025; Wei
et al., 2025; Chen et al., 2025b), we begin with
a cold-start phase using a large-scale unlabeled
dataset (see Section 3.1). This dataset is annotated
using DeepSeek-R1 (Guo et al., 2025), which gen-
erates responses via structured chain-of-thought
reasoning. Details of the system prompt are in
Appendix D. We find that applying SFT on the rele-
vance assessment task before RL improves training
stability and performance.
2.2 Decomposed Reasoning with RL
After the cold-start initialization, the training of
R³A is implemented based on the Group Relative
Policy Optimization (GRPO) (Shao et al., 2024)
algorithm. GRPO performs multiple rollouts per
input and calculate the relative reward rwithin
the group as the advantage A. It optimizes the
following objective:

trainablemodelfrozenmodelenvironmentobservationtoken,masklossactiontoken,calculateloss+LLMrolloutOutput2(𝑎!,…,𝑎!"#$%):<think>reasoningcontent</think><extract>fragment</extract><score>0/1/2</score>
SecondRound
𝐑=𝕀𝐟𝐨𝐫𝐦𝐚𝐭(𝒂𝟎,…,𝒂𝑵#𝑲%𝟏)$𝐫𝐬𝐜𝐨𝐫𝐞;					𝐫𝐬𝐜𝐨𝐫𝐞=𝟏,		𝒔𝒑𝒓𝒆𝒅==𝒔𝒈𝒐𝒍𝒅										𝝀,		𝒔𝒑𝒓𝒆𝒅−𝒔𝒈𝒐𝒍𝒅==𝟏𝟎,		𝒔𝒑𝒓𝒆𝒅−𝒔𝒈𝒐𝒍𝒅==𝟐Query𝑞LLM
searchDocBaserolloutOutput1(𝑎&,…,𝑎!$%):<think>reasoningcontent</think><intent>queryintent</intent>RetrievedDocs𝑑’
FirstRoundUpdateRewardModelComputeReward
ComputeReward
Doc𝑑coldstart
Figure 2: Overview of the R ³A framework. The RL training procedure involves a two-round interaction
between the model and the environment (in-platform documents). Firstly, auxiliary documents are
retrieved from the platform based on the user’s query and are then used, along with the query, to infer the
user’s intent. Secondly, the model is tasked with extracting verbatim, query-relevant fragment from the
candidate document to ground its assessment maximally in the document.
JGRPO (θ) = E[q, d∼P(Q, D ), τi∼πθold(τ|q, d)]
1
|G||G|X
i=11
|ai|
minπθ(ai|q, d)
πθold(ai|q, d)Ai,
clip πθ(ai|q, d)
πθold(ai|q, d),1−ϵ,1 +ϵ
Ai
−βDKL(πθ||πref)
(1)
Here, qandddenote the query and associated
document sampled from the training distribution
P(Q, D ). Given an input (q, d)pair1, a group G
of trajectories τiis generated using the old policy
πθold. Each trajectory τicomprises a sequence of
actions (a0,i, a1,i, a2,i, . . .), representing the output
of model reasoning. ϵis the clipping ratio. The
objective is to update the current policy πθ—the
relevance assessment model under training.
The term Ai=ri−µr
σrrepresents the standard-
ized advantage of trajectory τi, where riis the re-
ward assigned to the trajectory, and µrandσrare
the mean and standard deviation of the rewards
within group G, respectively.
Reasoning on Query Intent. In the first round,
the model engages with a set of auxiliary highly
ranked documents d′, retrieved using the same
query qwithin the platform, to parse the under-
lying query intent. Then, the model is encouraged
to utilize the previously analyzed query intent to
inform its relevance assessment in the next round.
1The retrieved document d′, based on the input query and
document d, is omitted here for brevity.So the trajectory in the first round can be rep-
resented as: τi1= (q, d′, a0,i, a1,i, ..., a N−1,i),
where Ndenotes the number of output tokens. We
define the following prompt to structure the output
tokens ( an,i,0≤n≤N−1) in the first round2:
Prompt in the 1st Round
System
You are a content understanding engineer working
on a user-generated content platform.
User
...
Please carefully analyze the given [query] and the
corresponding [in-platform documents] to infer the
underlying query intent.
Your response must strictly follow the format:
<think> [the reasoning content] </think>
<intent> [inferred user intent] </intent>
Input
[query]: {query}
[in-platform documents]: {docs}
Assistant
Reasoning on Noisy Document. To counteract
the noise introduced by informal language, in the
second round, we adopt an additional objective re-
quiring the model to perform relevance assessment.
It requires the model to extract verbatim, query-
relevant fragment (e.g., phrases or sentences) from
the candidiate document, returning a "None" output
when no matching content is found. This mecha-
nism aims to encourage the model to ground its as-
sessment maximally in the document, thereby pre-
venting noise-induced misjudgments of relevance.
2The detailed system template is shown in Appendix C

Thus, the complete trajectory in a rollout (two-
round interaction) can be represented as τi=
(τi1, d, a N,i, aN+1,i, . . . , a N+K−1,i, r), where K
denotes the number of tokens and rdenotes the
reward to be calculated. Then we define the
following format to structure the output tokens
(aN+k,i,0≤k≤K−1) in the second round:
Prompt in the 2nd Round
User
Please assess the relevance of the [document to be
evaluated] based on the user’s input [query] and the
inferred [intent], and extract the relevant fragment of
the document accordingly.
...
Your response must strictly follow the format:
<think> [the reasoning content] </think>
<extract> [fragment/none] </extract>
<score> [0/1/2] </score>
Input
[document to be evaluated]: {doc}
Assistant
Reward Function. We design the rule-based re-
ward function such that a reward is granted if and
only if the LLM-generated output fully conforms to
all specified reasoning and answer formats, as well
as the extraction requirements. The total reward R
is defined as:
R=Iformat·rscore (2)
where Iformat is an indicator function that equals 1
if the trajectory format is correct, and 0 otherwise.
Therscoremeasures the correctness of the model’s
prediction spredcompared to the gold score sgold:
rscore =

1spred==sgold
λ|spred-sgold|= 1
0|spred-sgold|= 2(3)
where λis a hyperparameter ( 0≤λ < 1) that
softly penalizes near-misses. The value of λbal-
ances the trade-off between strict correctness and
leniency in reward shaping. In Exp. 3.3, we in-
vestigate the impact of varying λon model perfor-
mance.
3 Experiment
In this section, we conduct both offline and online
experiments to evaluate the performance of R ³A
method on our dataset.
3.1 Settings
Dataset. We collect data from our online RAG
system at Xiaohongshu. Unlike traditional searchsystems that rely on user click-through data, we
propose to leverage the citation signal from the
answer generator.
Specifically, the generation model is explicitly
constrained to cite source content during response
generation. To construct training samples, we col-
lect real-world user queries and retrieved docu-
ments from our online system log. The generator
then forwards Mtimes for each query-document
pair. If a document is referenced in at least Ngen-
erations, we consider it a high-confidence positive
sample; otherwise, it’s treated as a challenging hard
negative sample. In practice, we set Nto 5 and M
to 2. We further collect random negatives from the
global corpus to introduce distributional noise.
We collect a total of 50k unlabeled documents
for cold-start training, and another 7k samples for
human double-blind annotation. Our dataset has
three classes: 0) Irrelevant, 1) Partially Relevant, 2)
Highly Relevant. Annotation details are provided
in Appendix A. We consider these as "gold" labels.
We refer to this dataset as NoteRel in the paper. A
class-balanced dataset is retained with 5K training
samples and the rest for evaluation. We find that
incorporating additional data beyond this set does
not yield significant improvements on our primary
evaluation metrics—consistent with observations
reported in prior work (Ye et al., 2025; Li et al.,
2025a; Jiang et al., 2025).
Baselines. To demonstrate the effectiveness of
R³A, we compare it with the following baselines3:
•UMbrela (Upadhyay et al., 2024b) It is a
prompting-based method that provides a step-
by-step guide to structure the relevance label-
ing task, thereby facilitating more nuanced
reasoning by the LLM. In our experiment,
we reproduce the UMbrela method using sev-
eral LLMs, including QwQ (Team Qwen,
2024), DeepSeek-R1 (Guo et al., 2025), GPT-
4o (Hurst et al., 2024), among others.
•SFT (Fitte-Rey et al., 2025) This method in-
troduces a framework that directly fine-tunes
the model using relevance label, based on the
prompt design of Upadhyay et al. (2024b). To
ensure a fair comparison, we also pre-train the
model using cold-start data before fine-tuning.
•R1-Zero/R1 (Guo et al., 2025) This method
employs rule-based RL to encourage the
3All methods are conducted in a zero-shot manner.

MethodF1-Score AUCAccuracy
0 1 2 0/12 01/2
Prompting
UMbrela w/o parameter update
Qwen2.5-7B 48.1 46.6 54.6 65.1 66.3 49.6
QwQ-32B 62.3 43.6 63.5 70.3 69.4 55.9
DeepSeek-V3 56.6 46.3 60.4 69.4 69.8 54.1
DeepSeek-R1 61.9 42.8 63.0 72.1 71.8 56.0
GPT-4o 63.3 53.8 59.6 73.0 69.8 58.2
Supervised Fine-Tuning
w/ parameter update
Bert 60.7 46.9 48.8 69.4 62.2 51.4
Qwen2.5-1.5B 68.4 51.8 57.7 76.3 68.5 59.0
+ pretrained 70.1 55.3 56.0 77.3 68.2 60.0
Qwen2.5-7B 71.0 55.6 52.1 78.2 66.8 60.3
+ pretrained 69.9 60.6 53.9 77.0 67.6 61.4
Reinforcement Learning
w/ parameter update on Qwen2.5-1.5B
R1-Zero 66.7 45.3 58.5 75.1 68.2 56.8
R1 70.7 46.9 62.0 78.3 69.0 59.7
R³A-Zero 71.3 49.8 60.5 79.4 68.3 60.4
R³A 72.0 51.5 62.9 80.8 69.6 61.7
w/ parameter update on Qwen2.5-7B
R1-Zero 67.5 47.2 62.5 75.6 71.7 59.3
R1 74.4 49.6 63.7 81.2 72.7 63.2
R³A-Zero 75.8 51.7 63.3 82.4 72.5 63.6
R³A 77.1 56.0 64.2 83.1 73.3 65.2
Distilling (Online Serving)
R³A-Distill-1.5B 71.4 55.9 60.3 78.3 70.5 62.0
Table 1: Overall performance on relevance assess-
ment. Models with the "-Zero" suffix are trained
without cold-start initialization. The labels 0, 1,
and 2 indicate "Irrelevant", "Partially Relevant",
and "Highly Relevant", respectively. AUC 0/12and
AUC 01/2denote the one-vs-rest strategy on AUC
metric.
model to engage in explicit reasoning dur-
ing relevance assessment. The model’s out-
put format is constrained to the <think> and
<score> tags, in accordance with the original
implementation. The R1-Zero refers to RL
initiated without the cold-start strategy.
•Distillation The distilled model is trained on a
1.5B backbone using the same SFT approach.
It utilizes only the logits output of the score la-
bels, and is trained on 100k samples generated
by the R ³A-7B model. We deploy this version
in our online system for better inference speed
and overall throughput.
Models. For SFT and RL-based methods, we ex-
plore LLMs using instruction-tuned Qwen2.5 mod-
els (Yang et al., 2024) ranging from 1.5B to 7B
parameters, with training specifics in Appendix B.
Metrics. We use F1 score (macro-averaged), Ac-
curacy, and one-vs-rest AUC metrics (AUC 0/12
0 50 100 150 200 250 300 350
Step0.00.10.20.30.40.50.60.70.8Reward
R3A-Qwen-7B
R3A-Zero-Qwen-7B
R3A-Qwen-1.5B
R3A-Zero-Qwen-1.5B(a) Training Rewards
0 50 100 150 200 250 300 350
Step50100150200250300350400Response LengthR3A-Qwen-7B
R3A-Zero-Qwen-7B
R3A-Qwen-1.5B
R3A-Zero-Qwen-1.5B (b) Response Length
Figure 3: The training log of R ³A-Zero-1.5/7B
and R ³A-1.5/7B, including the curve of training
rewards and response length.
and AUC 01/2) following previous works (Welleck,
2016; Chen et al., 2024).
3.2 Main Results
Overall Performance. As shown in Table 1, our
proposed method R ³A consistently outperforms
all baseline approaches on the test set of NoteRel.
Among prompting methods, GPT-4o delivers the
strongest performance; however, approaches that
involve parameter updates (SFT and RL) gener-
ally outperform prompting. Notably, R ³A-Zero
surpasses both R1-Zero and SFT baselines, indicat-
ing that the R ³A framework effectively leverages
the reasoning capability of model improve perfor-
mance. Furthermore, R ³A-7B obtains the best AUC
values (83.1% on 0/12 and 73.3% on 01/2), reflect-
ing enhanced sensitivity to relevance boundaries.
While improvements in F1-score for class 1 are
limited across models—highlighting the difficulty
of corner cases—R ³A maintains strong accuracy
and balanced performance overall, confirming its
robustness in UGC scenarios. We showcase some
of our results in Appendix E.
Superior Distillation. Remarkably, the distilled
SFT model R ³A-Distill-1.5B, trained using only
label supervision, not only retains the performance
gains of its 1.5B RL-trained counterpart (+3.0%
in accuracy), but also outperforms the larger 7B
SFT model by +1.7% in accuracy. These results
indicate that the knowledge distilled from R ³A ef-
fectively preserves essential relevance assessment
capabilities in a smaller model.
Role of Cold Start. Figure 3 shows that R ³A
models initialized with cold start exhibit faster re-
ward growth and achieve higher final rewards com-
pared to their R ³A-Zero counterparts, underscoring
the benefit of cold-start initialization. Although
Zero models begin with lower rewards, they gradu-
ally acquire the desired format, demonstrating that

decomposed reasoning remains effective even with-
out prior initialization. In terms of response length,
both R ³A and R ³A-Zero models rapidly converge
to a stable and concise output length.
3.3 Ablation Study
To assess the contribution of each component in the
R³A framework, we conducted a series of ablation
studies, as summarized in Table 2. Format Variants
refer to the removal of either intent reasoning or
extraction reasoning. All variants lead to a drop in
performance across all metrics, with the removal
of extraction reasoning causing the most signifi-
cant degradation. This underscores the importance
of grounding relevance assessment in the candi-
date document. We also evaluated the impact of
removing retrieved documents , allowing the model
to infer intent solely from the user query. This vari-
ant similarly resulted in performance degradation.
Reward Variants , applying lighter penalties for con-
fusion between classes 0 and 2 will weaken the
model’s performance. Therefore, we set the reward
balancing factor λto 0 in our main experiments.
Finally, collapsing the two-stage interaction into a
single-turn input substantially harms performance,
particularly on class 2 (F1 drops to 58.3%) and
overall accuracy (down to 60.6%). This suggests
that longer inputs make it harder for the model to at-
tend to the candidate document, leading to reduced
assessment effectiveness.
3.4 Online Performance
We deploy R ³A-Distill-1.5B in our production
RAG system as the re-rank module4. An online
experiment is conducted with 10% of online traffic
for one week. First, we conduct a human evalua-
tion with 100 random query-answer pairs from our
online system log. Annotators are required to judge
the quality of answer from multiple dimensions, in-
cluding factual accuracy, format, and completeness.
Compared with our online model (Fitte-Rey et al.,
2025), the distilled model yields a final GSB dis-
tribution of 23 : 71 : 6 (Good:Same:Bad), an 17%
improvement over the final answer quality.
In addition, due to the sparsity of click-through
data in RAG scenarios, we adopt re-query rate as
a proxy metrics to evaluate answer quality. We
refer re-query rate as the frequency of immediate
follow-up search attempts after an initial query,
4R³A-Distill-1.5B takes the top 100documents from our
pre-rank system, and selects the 10most relevant documents
for answer generation.MethodF1-Score AUCAccuracy
0 1 2 0/12 01/2
R³A-7B ( λ= 0) 77.1 56.0 64.2 83.1 73.3 65.2
Format Variants
w/o<intent> 76.0 55.9 63.7 82.6 72.8 64.7
w/o<extraction> 74.9 52.7 63.2 81.1 72.4 63.9
Input Variants
w/oretrieval∗75.7 54.4 63.9 82.4 72.9 64.6
Reward Variants
λ= 0.5 75.3 52.5 59.7 82.0 69.9 62.0
λ= 0.2 74.9 53.6 62.9 81.9 72.3 62.5
λ= 0.1 72.6 48.4 60.3 80.2 70.8 61.4
Interaction Round Variants
single-round interaction †73.8 49.1 58.3 80.7 68.5 60.6
*"Without retrieval" denotes the removal of retrieved in-platform documents, while the
output format remains unchanged and still requires query intent reasoning.
†Both the retrieved documents and the document to be evaluated are input into the model,
which is required to analyze the intent, extract the fragment and assess the relevance.
Table 2: Ablation study on the R ³A method. Each
row removes or modifies a component to assess its
impact.
within the same session. A lower number indi-
cates higher user satisfaction, as it suggests that
the initial search result already address the user
intent without requiring reformulation. We observe
a significant 1.03% reduction, which implies that
the generated answers better satisfy user needs and
reduce subsequent search attempts.
4 Related Work
Relevance modeling evaluates the extent to which
a document satisfies a user query. Traditional
human-annotated approaches are costly and prone
to subjectivity, prompting interest in the "LLM-as-
a-Judge" paradigm (Zheng et al., 2023).
Faggioli et al. (2023) are among the first to in-
vestigate a range of human–machine collaboration
strategies in which LLMs assist in relevance judg-
ment. Building upon this, automated evaluations
using LLMs have combined various prompting
techniques such as zero-shot, one-shot (MacAvaney
and Soldaini, 2023), or few-shot learning (Thomas
et al., 2024; Upadhyay et al., 2024a,b).
Another line of work (Ma et al., 2024; Fitte-
Rey et al., 2025) involves training dedicated LLMs
for assessment tasks. While ProPBP (Chen et al.,
2024) also proposes to include user behaviors in
model inputs, their approach primarily focuses on
addressing personalization challenges, whereas our
method is designed to enhance reasoning capabili-
ties for final relevance assessment. More recently,
models such as JudgeLRM (Chen et al., 2025a)
and Rank-R1 (Zhuang et al., 2025) have emerged,

explicitly incorporating reasoning across different
assessment tasks through RL with outcome-driven
rewards.
5 Conclusion
In this paper, we propose R ³A, a novel decomposed
reasoning framework tailored for relevance assess-
ment for RAG system in UGC scenarios. This
approach enables the model to better infer the user
intent and reduce erroneous outputs. Empirically,
R³A consistently outperforms all baselines, exhibit-
ing strong capabilities in relevance assessment task
in UGC scenarios.
Limitations
Despite the effectiveness of R ³A in improving rel-
evance assessment in RAG systems under UGC
scenarios, there are several limitations. First, R ³A
is evaluated primarily on industry-specific UGC
dataset. Its performance may not generalize well
to other domains such as biomedical or legal texts,
where language style and relevance criteria differ
significantly. Second, since the in-platform doc-
ument retrieval pipeline is dependent on retrieval
quality, suboptimal retrieval results may lead to in-
correct estimation of user intent and misalignment
with the target document under assessment, thereby
limiting the effectiveness of R ³A regardless of its
reasoning capability.
References
Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu,
Qian Wang, Bryan Hooi, and Bingsheng He. 2025a.
Judgelrm: Large reasoning models as a judge. CoRR ,
abs/2504.00050.
Shuang Chen, Yue Guo, Zhaochen Su, Yafu Li, Yu-
lun Wu, Jiacheng Chen, Jiayu Chen, Weijie Wang,
Xiaoye Qu, and Yu Cheng. 2025b. Advancing multi-
modal reasoning: From optimized cold start to staged
reinforcement learning. CoRR , abs/2506.04207.
Zeyuan Chen, Haiyan Wu, Kaixin Wu, Wei Chen,
Mingjie Zhong, Jia Xu, Zhongyi Liu, and Wei Zhang.
2024. Towards boosting llms-driven relevance mod-
eling with progressive retrieved behavior-augmented
prompting. arXiv preprint arXiv:2408.09439 .
Guglielmo Faggioli, Laura Dietz, Charles L. A. Clarke,
Gianluca Demartini, Matthias Hagen, Claudia Hauff,
Noriko Kando, Evangelos Kanoulas, Martin Potthast,
Benno Stein, and Henning Wachsmuth. 2023. Per-
spectives on large language models for relevance
judgment. In Proceedings of the 2023 ACM SIGIR
International Conference on Theory of InformationRetrieval, ICTIR 2023, Taipei, Taiwan, 23 July 2023 ,
pages 39–50. ACM.
Quentin Fitte-Rey, Matyas Amrouche, and Romain De-
veaud. 2025. Augmented relevance datasets with
fine-tuned small llms. CoRR , abs/2504.09816.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai
Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao,
Zhuoshu Li, Ziyi Gao, Aixin Liu, and 80 others.
2025. Deepseek-r1: Incentivizing reasoning capa-
bility in llms via reinforcement learning. CoRR ,
abs/2501.12948.
Jian Hu, Xibin Wu, Weixun Wang, Xianyu, Dehao
Zhang, and Yu Cao. 2024. Openrlhf: An easy-to-
use, scalable and high-performance RLHF frame-
work. CoRR , abs/2405.11143.
Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, Alek-
sander Madry, Alex Baker-Whitcomb, Alex Beutel,
Alex Borzunov, Alex Carney, Alex Chow, Alex Kir-
illov, Alex Nichol, Alex Paino, and 79 others. 2024.
Gpt-4o system card. CoRR , abs/2410.21276.
Guanying Jiang, Lingyong Yan, Haibo Shi, and Dawei
Yin. 2024. The real, the better: Aligning large lan-
guage models with online human behaviors. Preprint ,
arXiv:2405.00578.
Pengcheng Jiang, Xueqiang Xu, Jiacheng Lin, Jinfeng
Xiao, Zifeng Wang, Jimeng Sun, and Jiawei Han.
2025. s3: You don’t need that much data to train a
search agent via RL. CoRR , abs/2505.14146.
Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025a.
LIMR: less is more for RL scaling. CoRR ,
abs/2502.11886.
Yuchen Li, Hengyi Cai, Rui Kong, Xinran Chen, Jiamin
Chen, Jun Yang, Haojie Zhang, Jiayi Li, Jiayi Wu,
Yiqun Chen, Changle Qu, Keyi Kong, Wenwen Ye,
Lixin Su, Xinyu Ma, Long Xia, Daiting Shi, Jiashu
Zhao, Haoyi Xiong, and 2 others. 2025b. Towards ai
search paradigm. Preprint , arXiv:2506.17188.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. In Proceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, SIGIR 2024,
Washington DC, USA, July 14-18, 2024 , pages 2421–
2425. ACM.

Sean MacAvaney and Luca Soldaini. 2023. One-shot
labeling for automatic relevance estimation. In Pro-
ceedings of the 46th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27,
2023 , pages 2230–2235. ACM.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Mingchuan Zhang, Y . K. Li, Y . Wu,
and Daya Guo. 2024. Deepseekmath: Pushing the
limits of mathematical reasoning in open language
models. CoRR , abs/2402.03300.
Team Qwen. 2024. Qwq: Reflect deeply on the bound-
aries of the unknown.
Paul Thomas, Seth Spielman, Nick Craswell, and
Bhaskar Mitra. 2024. Large language models can ac-
curately predict searcher preferences. In Proceedings
of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval,
SIGIR 2024, Washington DC, USA, July 14-18, 2024 ,
pages 1930–1940. ACM.
Shivani Upadhyay, Ehsan Kamalloo, and Jimmy Lin.
2024a. Llms can patch up missing relevance judg-
ments in evaluation. CoRR , abs/2405.04727.
Shivani Upadhyay, Ronak Pradeep, Nandan Thakur,
Nick Craswell, and Jimmy Lin. 2024b. UMBRELA:
umbrela is the (open-source reproduction of the) bing
relevance assessor. CoRR , abs/2406.06519.
Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang,
Yue Wang, Linghe Kong, Lichao Sun, and Weiran
Huang. 2025. Advancing multimodal reasoning
via reinforcement learning with cold start. CoRR ,
abs/2505.22334.
Sean J. Welleck. 2016. Efficient AUC optimization
for information ranking applications. In Advances in
Information Retrieval - 38th European Conference on
IR Research, ECIR 2016, Padua, Italy, March 20-23,
2016. Proceedings , volume 9626 of Lecture Notes in
Computer Science , pages 159–170. Springer.
An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayi-
heng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Ji-
axi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and
22 others. 2024. Qwen2.5 technical report. CoRR ,
abs/2412.15115.
Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie
Xia, and Pengfei Liu. 2025. LIMO: less is more for
reasoning. CoRR , abs/2502.03387.Wen Zan, Yaopeng Han, Xiaotian Jiang, Yao Xiao, Yang
Yang, Dayao Chen, and Sheng Chen. 2023. Spm:
Structured pretraining and matching architectures for
relevance modeling in meituan search. In Proceed-
ings of the 32nd ACM International Conference on
Information and Knowledge Management , CIKM
’23, page 4923–4929. Association for Computing
Machinery.
Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin,
Chen Gao, and Yong Li. 2025. A survey of large lan-
guage model empowered agents for recommendation
and search: Towards next-generation information re-
trieval. Preprint , arXiv:2503.05659.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Advances in Neural Information Processing Systems
36: Annual Conference on Neural Information Pro-
cessing Systems 2023, NeurIPS 2023, New Orleans,
LA, USA, December 10 - 16, 2023 .
Shengyao Zhuang, Xueguang Ma, Bevan Koopman,
Jimmy Lin, and Guido Zuccon. 2025. Rank-r1: En-
hancing reasoning in llm-based document rerankers
via reinforcement learning. CoRR , abs/2503.06034.

A Annotation Quality Control
To ensure the reliability of the annotations, we em-
ployed rigorous quality control measures, includ-
ing:
•Annotation Guidelines: Annotators were pro-
vided with detailed guidelines and training
sessions to standardize their understanding of
relevance criteria, minimizing potential biases
in the annotations.
•Inter-annotator Agreement: A subset of anno-
tations was double-annotated by two indepen-
dent annotators to assess the consistency of
the judgments. Only those annotations with
an agreement above a predefined threshold
were retained for further use.
•Expert Review: A small portion of the anno-
tated data was reviewed by domain experts
to verify the accuracy and consistency of the
annotations.
•Feedback Loops: Regular feedback was pro-
vided to annotators to ensure continuous im-
provement of annotation quality during the
process.
These measures were implemented to ensure that
the annotations are reliable, consistent, and repre-
sentative of the true relevance of query-document
pairs.
B Implementation Details
We employ Qwen2.5-1.5B(-Instruct) and Qwen2.5-
7B(-Instruct) as the initial models. Models with the
"-Zero" suffix are trained without cold start from
the Instruct model. We utilize the OpenRLHF (Hu
et al., 2024) framework for training. GRPO (Shao
et al., 2024) is used as the reinforcement learning
algorithm. We use the NoteRel as the training and
test sets. We set the number of rollouts as 16 for
one task. We set the learning rate as 5e-7, batch
size as 32, training steps as 360. We set λas 0 in
reward function. We use 8 A100 GPUs for all the
experiments.
C System Prompt for R³A
The following presents the full prompt used in the
first-round interaction.Prompt in the 1st Round
System
You are a content understanding engineer working
on a user-generated content platform.
User
Please determine the primary intent behind a user’s
search query, using both your internal knowledge
and the provided context.
Your input consists of the [query] and the [in-
platform documents] retrieved based on that query.
The latter is intended to assist in judging the user’s
intent but may contain irrelevant content. The search
query should be considered the primary reference.
Please carefully analyze the given [query] and the
corresponding [in-platform documents] to infer the
underlying query intent.
Your response must strictly follow the format:
<think> [the reasoning content] </think>
<intent> [inferred user intent] </intent>
Input
[query]: {query}
[in-platform documents]: {docs}
Assistant
The following presents the full prompt used in
the second-round interaction.
Prompt in the 2nd Round
User
Please assess the relevance of the [document to be
evaluated] based on the user’s input [query] and the
inferred [intent], and extract the relevant fragment of
the document accordingly.
Scoring Criteria
0 = not relevant, the document has nothing to do with
the query.
1 = partially relevant, the document is relevant to the
query but partly answers it.
2 = highly relevant, the document is dedicated to the
query and contains the exact answer.
Extraction Guidelines
1. Extract the content from the [document to be
evaluated] that is strictly relevant to the query
and can help answer the query. This may include
paragraphs, sentences, or even individual phrases.
2. The extracted content must come directly from the
original document, with all punctuation preserved.
Your response must strictly follow the format:
<think> [the reasoning content] </think>
<extract> [fragment/none] </extract>
<score> [0/1/2] </score>
Input
[document to be evaluated]: {doc}
Assistant
D UMbrela Prompt
The following presents the full prompt used in
UMbrela method, which is also employed by
DeepSeek-R1 to generate reasoning chains and an-
swers on unlabeled data.

Prompt
System
You are a relevance assessor working on a user-
generated content platform.
User
Given a query and a document, you must provide a
score on an integer scale of 0 to 2 with the following
meanings:
0 = represent that the document has nothing to do
with the query
1 = represents that the document has some answer
for the query, but the answer may be a bit unclear, or
hidden amongst extraneous information
2 = represents that the document is dedicated to the
query and contains the exact answer
Important Instruction:
Assign category 1 if document presents something
very important related to the entire topic but also
has some extra information and category 2 if the
document only and entirely refers to the topic. If
none of the above satisfies give it category 0.
Please determine the primary intent behind a user’s
search query, using both your internal knowledge
and the provided context.
Your response must strictly follow the format:
<think> [the reasoning content] </think>
<score> [0/1/2] </score>
Input
[query]: {query}
[document to be evaluated]: {doc}
Assistant
E Case Study
In Table 3, we present an example where a docu-
ment with a gold score of 0 was misclassified as
partly relevant by the R1 model but correctly clas-
sified by R ³A. The input consists of the query and
the document to be evaluated, without retrieved
documents to assist in intent parsing, in order to
improve readability. The user query asked for the
precise definition of an infinite series, whereas the
document only contained related concepts from a
table of contents and did not address the question
directly. The R1 model was misled by superficial
lexical matches and assigned a relevance score of
1, despite the document failing to meet the user’s
information need. In contrast, our model, R ³A, suc-
cessfully inferred the document’s lack of relevance
by identifying and reasoning over the actual con-
tent, ultimately producing the correct assessment.

Query: What does infinite series mean?
Document: Science students at the University of Manchester, speed up! The most comprehensive calculus vocabulary
list is here Hi dear students studying at UoM I’m sure your English is totally fine for daily communication, but when it
comes to some technical terms, it can get tricky! Looking up unfamiliar words one by one is just too much trouble and
really drags down our learning efficiency. When I was studying abroad, I often thought: if only there were a specialized
vocabulary book for each subject!
Well, here it is! The most complete English-Chinese vocabulary list for calculus is finally here! You’re really catching it
at the right time!
This vocabulary list is based on content from the following calculus topics:
Limits and Continuity:
[emoji]Definition and computation of limits [emoji]Infinite limits and asymptotes [emoji]Definition and determination of
continuity
...
Integrals:
[emoji]Definitions of definite and indefinite integrals [emoji]Basic integration techniques: substitution, integration by
parts [emoji]Applications of definite integrals: area, volume, arc length
Infinite Series :
[emoji]Convergence tests for series [emoji]Power series and Taylor series
Multivariable Calculus:
[emoji]Partial derivatives and directional derivatives [emoji]Multiple integrals and vector calculus
To all students taking MATH10121, MATH10222, MATH20122, and MATH20411 at UoM — don’t let calculus
vocabulary slow down your progress. Save this vocabulary list now, learn efficiently, and enjoy solving problems!
Like + follow, and if there’s anything else you’d like to see, just DM me #UniversityofManchester #UoMstudyabroad
#UKstudents #calculusQ&A #UoMstudents
Gold Score: 0
R1 Output:
<think> First, I need to evaluate the relevance between the user’s search term and the document to be assessed, and
assign a score based on the evaluation criteria. The search term is "What does infinite series mean?", and the user’s intent
is to understand the concept and meaning of infinite series.
Next, let’s look at the content of the document. The title mentions "The Most Compre-
hensive Calculus V ocabulary List" and covers multiple topics from calculus courses, includ-
ing limits and continuity, derivatives, integrals, infinite series, and multivariable calculus.
In the section on "Infinite Series," the document lists terms such as convergence tests, power series, and Taylor series.
When analyzing relevance, the core of the search term is the definition and meaning of "infinite series." Although the
document does mention content related to infinite series, it primarily presents a vocabulary list of related terms and
methods, such as convergence tests and power series, without providing a detailed explanation or mathematical definition
of the concept. The purpose of the document is to offer a vocabulary list to help students memorize and learn technical
terms in calculus, rather than to explain each concept in depth.
According to the scoring criteria, a score of 2 requires that the document directly meets the main need of the search
term and provides a complete answer. Since the user is looking for an explanation of infinite series and the document
does not provide detailed content in this regard—only related terms and methods—it partially addresses the topic.
Therefore, it should be scored as 1 point. </think>
<answer> 1</answer>
R³A Output in the 2nd Round (Ours):
<think> First, I need to assess the relevance between the user’s search query and the document to be evaluated, and
assign a score based on the evaluation criteria. The search term is "What does infinite series mean?", and the user’s need
is to understand the definition or basic concept of infinite series.
Next, I examine the content of the document. Most of the document consists of a usage guide for a calculus vocab-
ulary list and the source courses. In the "Infinite Series" section, it mentions "tests for convergence of series" and
"power series and Taylor series." These may relate to certain evaluation methods and topics within infinite series,
but they do not provide a definition or basic conceptual explanation of infinite series.
According to the scoring criteria, a score of 2 requires that the document directly satisfies the main need of the
search term and provides a complete answer. The search term calls for a definition or basic concept of infinite series,
which the document does not provide. A score of 1 is appropriate when the document partially answers the query or
under specific conditions. In this case, the document does not provide a definition or basic concept of infinite series ,
but instead refers to some related methods and topics such as convergence tests and power series. Therefore, the
document does not directly answer the search term and does not provide a complete response.
As a result, the document is irrelevant to the main intent of the search term and should be given a score of 0. </think>
<extract> None </extract>
<answer> 0</answer>
Table 3: The Case comparison using the R1 and R ³A models. The highlighted content indicate the
portions in the original text that overlap with the query keywords, as well as the key reasoning components
generated by the model.