# RAG-Zeval: Towards Robust and Interpretable Evaluation on RAG Responses through End-to-End Rule-Guided Reasoning

**Authors**: Kun Li, Yunxiang Li, Tianhua Zhang, Hongyin Luo, Xixin Wu, James Glass, Helen Meng

**Published**: 2025-05-28 14:55:33

**PDF URL**: [http://arxiv.org/pdf/2505.22430v1](http://arxiv.org/pdf/2505.22430v1)

## Abstract
Robust evaluation is critical for deploying trustworthy retrieval-augmented
generation (RAG) systems. However, current LLM-based evaluation frameworks
predominantly rely on directly prompting resource-intensive models with complex
multi-stage prompts, underutilizing models' reasoning capabilities and
introducing significant computational cost. In this paper, we present RAG-Zeval
(RAG-Zero Evaluator), a novel end-to-end framework that formulates faithfulness
and correctness evaluation as a rule-guided reasoning task. Our approach trains
evaluators with reinforcement learning, facilitating compact models to generate
comprehensive and sound assessments with detailed explanation in one-pass. We
introduce a ranking-based outcome reward mechanism, using preference judgments
rather than absolute scores, to address the challenge of obtaining precise
pointwise reward signals. To this end, we synthesize the ranking references by
generating quality-controlled responses with zero human annotation. Experiments
demonstrate RAG-Zeval's superior performance, achieving the strongest
correlation with human judgments and outperforming baselines that rely on LLMs
with 10-100 times more parameters. Our approach also exhibits superior
interpretability in response evaluation.

## Full Text


<!-- PDF content starts -->

arXiv:2505.22430v1  [cs.CL]  28 May 2025RAG-Zeval: Towards Robust and Interpretable Evaluation on RAG
Responses through End-to-End Rule-Guided Reasoning
Kun Li♡*, Yunxiang Li♡∗, Tianhua Zhang♡∗, Hongyin Luo♢,
Xixin Wu♡, James Glass♢, Helen Meng♡
♡The Chinese University of Hong Kong, Hong Kong SAR, China
♢Massachusetts Institute of Technology, Cambridge MA, USA
{li.kun, yli, thzhang}@link.cuhk.edu.hk
Abstract
Robust evaluation is critical for deploying trust-
worthy retrieval-augmented generation (RAG)
systems. However, current LLM-based evalua-
tion frameworks predominantly rely on directly
prompting resource-intensive models with com-
plex multi-stage prompts, underutilizing mod-
els’ reasoning capabilities and introducing sig-
nificant computational cost. In this paper, we
present RAG-Zeval (RAG -ZeroEval uator), a
novel end-to-end framework that formulates
faithfulness and correctness evaluation as a
rule-guided reasoning task. Our approach
trains evaluators with reinforcement learning,
facilitating compact models to generate com-
prehensive and sound assessments with de-
tailed explanation in one-pass. We introduce a
ranking-based outcome reward mechanism, us-
ing preference judgments rather than absolute
scores, to address the challenge of obtaining
precise pointwise reward signals. To this end,
we synthesize the ranking references by gen-
erating quality-controlled responses with zero
human annotation. Experiments demonstrate
RAG-Zeval ’s superior performance, achieving
the strongest correlation with human judgments
and outperforming baselines that rely on LLMs
with 10−100×more parameters. Our ap-
proach also exhibits superior interpretability
in response evaluation.
1 Introduction
Retrieval-Augmented Generation (RAG) systems
(Lewis et al., 2021; Gao et al., 2024; Li et al., 2024)
have become a cornerstone for building knowledge-
intensive NLP applications, such as question an-
swering, fact-checking in various domains. (Zhao
et al., 2025; Pipitone and Alami, 2024) By inte-
grating external knowledge retrieval with large lan-
guage models (LLMs), RAG enables more accurate
and contextually relevant responses (Li et al., 2025;
*Equal contribution.Asai et al., 2024), especially for queries that go be-
yond the static knowledge encoded in model param-
eters. As RAG systems are increasingly deployed
in real-world scenarios, robust and comprehensive
evaluation is essential to assess their performance
and guide further development (Yu et al., 2025a).
However, evaluating RAG systems remains chal-
lenging due to their modular structure and the com-
plex interplay between retrieval and generation.
Traditional metrics such as recall@k and MRR
for retrieval, and BLEU (Papineni et al., 2002),
and ROUGE (Lin, 2004) for generation are often
coarse-grained and fail to capture semantic fidelity
or factual consistency in open-ended tasks. To
overcome these limitations, recent work has ex-
plored model-based evaluation strategies, partic-
ularly leveraging LLMs as automatic judges (Gu
et al., 2024). Frameworks such as RAGAS (Shahul
et al., 2023) and RAG-Checker (Ru et al., 2024)
have demonstrated that LLMs can provide scalable
and automated assessments of metrics like context
relevance and faithfulness, reducing the reliance
on costly human annotation and enabling efficient
large-scale evaluation.
Although showing the potential as being au-
tomated RAG evaluators, these LLM-based ap-
proaches (Shahul et al., 2023; Ru et al., 2024) pre-
dominantly rely on prompting large-scale LLMs
with advanced capabilities (e.g., GPT-4 (OpenAI
et al., 2024), Llama3-70B (Grattafiori et al., 2024))
to achieve strong assessment results, introducing
significant computational costs. Motivated by these
limitations, we study whether compact LLMs can
be transformed into end-to-end and interpretable
evaluators with incentivized reasoning abilities.
In this work, we present RAG-Zeval (RAG -Zero
Eval uator), a novel framework that formulates
faithfulness and correctness evaluation as a rule-
guided reasoning task with zero human annota-
tion. Our approach enables the compact evalua-
tors to generate comprehensive and sound assess-

ments in one-pass under the instruction of prede-
fined rules, systematically performing (1) claim
decomposition, (2) evidence grounding, and (3)
supportiveness judgment. This end-to-end evalua-
tion distinguish our method from previous multi-
stage pipelines, which ensures assessment consis-
tency through holistic reasoning and captures the
interdependence between different steps. Drawing
inspiration from recent advances in reinforcement
learning (RL) (DeepSeek-AI et al., 2025; Yu et al.,
2025b; Liu et al., 2025a), we train evaluators using
rule-based RL (DeepSeek-AI et al., 2025) to elicit
interpretable reasoning capabilities from compact
LLMs. This rule-based RL alignment eliminates
the need for process rewards (e.g., evaluation tra-
jectory annotations) (Lightman et al., 2024; Zhong
et al., 2025). To overcome the challenge of ac-
quiring precise point-wise reward signals, we intro-
duce a ranking-based outcome reward mechanism,
which operates on more easily obtainable prefer-
ence judgments instead of absolute scores (Guan
et al., 2025). Recognizing that high-quality rewards
in open-ended generation tasks typically require
expensive human annotations (Liu et al., 2025b),
we further synthesize the ranking reference using
Context-Aware Decoding (Shi et al., 2023) to gen-
erate quality-controlled response candidates. This
enables fully automated training free of human la-
bels. Additionally, we employ curriculum learning,
progressively increasing the number of candidate
responses to be ranked as RL advances, which fur-
ther improves the evaluators performance.
We assess RAG-Zeval on both faithfulness and
correctness benchmarks to analyze its performance
in deriving interpretable and reliable evaluations.
Experimental results demonstrate that RAG-Zeval
achieves strong alignment with human judgments,
maintaining transparent and interpretable decision-
making through its rule-guided reasoning process.
2 Related Work
With the rapid advancement of Retrieval-
Augmented Generation (RAG) systems (Fan
et al., 2024; Li et al., 2025), effective and robust
evaluation methods beyond traditional metrics
have become increasingly important.
A significant line of work evaluates the retrieval
and generation components separately. For re-
trieval, traditional information retrieval metrics
such as precision, recall, MRR, and MAP are
widely used. (Yu et al., 2024; Tang and Yang,2024) For the generation component, metrics like
BLEU (Papineni et al., 2002), ROUGE (Lin, 2004),
and BERTScore (Zhang et al., 2020) are com-
monly used, alongside human evaluation. Such
component-wise evaluation often fails to capture
the complex interactions between retrieval and gen-
eration in real-world RAG systems, and human
annotation is costly, time-consuming, and subject
to annotator bias and inconsistency.
Recent research on the evaluation of RAG sys-
tems has moved beyond traditional component-
wise metrics, proposing a variety of end-to-end
frameworks that leverage large language models
(LLMs) as evaluators. LLM-based evaluation
frameworks such as TruLens (Ferrara et al., 2024)
and ARES (Saad-Falcon et al., 2024) adopt direct
prompting to score responses without decompos-
ing them into individual claims. Other approaches,
including RAGAS (Shahul et al., 2023), RAG-
Checker (Ru et al., 2024), and OpenEval (Ispas
et al., 2025), introduce claim-level decomposition,
enabling LLMs to assess the faithfulness and cor-
rectness of each factual statement for finer-grained
and more interpretable evaluation.
Despite these advances, most current LLM-
based evaluation frameworks rely on direct prompt-
ing of large, resource-intensive models, often in-
volve complex multi-stage prompting, and treat
LLMs as black-box scorers without fully leverag-
ing their reasoning abilities. Recent progress in re-
inforcement learning and reward modeling, such as
Deepseek-R1 (DeepSeek-AI et al., 2025) and gen-
erative reward modeling (GRM) (Liu et al., 2025b),
demonstrates that rule-driven, interpretable evalu-
ators trained via rule-based RL can provide more
transparent and scalable assessments with stronger
reasoning ability. These developments motivate
our approach to construct RAG evaluators using
similar RL-based, rule-guided techniques.
3 Methodology
3.1 Problem Formulation
A majority of evaluation tools for assessing the
response quality of RAG systems adopt a claim-
based paradigm(Ru et al., 2024; Shahul et al., 2023;
Ispas et al., 2025; Manakul et al., 2023). In this
paradigm, system responses are decomposed into
individual claims, a declarative sentence that con-
veys an atomic piece of information, and each
claim is then evaluated for supportiveness—the
degree to which it is grounded in the provided ref-

Question q
Passages c
Context-Aware 
Decoding𝛼1𝛼2𝛼3 >>
Rule-
Guided 
Prompt
Qwen2.5-7B-
Instruct𝑦1
𝑦2
𝑦3>>Reference Rank
Detailed Evaluation 1 𝑦1𝑦2𝑦3 >>
Detailed Evaluation N 𝑦2𝑦1𝑦3 >>… …Response Synthesis Ranking -based RLFigure 1: An overview of RAG-ZEval . We synthesize training data using Context-Aware Decoding. The complete
prompt is presented in Fig.5. The ground-truth ranking of y’s depends on the value of α.
erence context (e.g., ground-truth answer for cor-
rectness or retrieved passages for faithfulness).
In a RAG setting, a response is considered 1) cor-
rect if the ground-truth answer supports the claims
of the response, or 2) faithful if the retrieved pas-
sage supports the claims of the response. Based on
this, following prior work (Ru et al., 2024; Ispas
et al., 2025; Shahul et al., 2023), given a response,
we define its correctness as
# Claims supported by ground-truth answer
# Claims, (1)
andfaithfulness as
# Claims supported by retrieved passage
# Claims, (2)
indicating the precision rate of claims in the re-
sponse that are supported by the ground-truth an-
swer and the retrieved passage respectively. These
formulations consolidate the evaluation paradigms
for both correctness and faithfulness, allowing for
the development of a unified evaluator that assesses
the two quality dimensions . We can conduct evalua-
tion in both correctness and faithfulness with differ-
ent reference (ground-truth answer for correctness
and retrieved passages for faithfulness evaluation).
3.2 Prompting for Rule-Guided Reasoning
Different from previous claimed-based work which
runs in a multi-stage pipeline, we develop a novel
approach for end-to-end claim-based evaluation,
through generation of complete evaluation trajec-
tories under the rules of guide.
To this end, we adopt the prompt demonstrated
in Fig.5, which elaborates the rules and format the
generation should conform to. In detail, given a
question qand the reference c, and the set of re-
sponses to evaluate {y}, LLM should give a sound
evaluation process—decomposing a response into
claims, and then determining those claims’ sup-
portiveness as well as finding the grounding ev-
idence in the reference. In addition, the genera-
tion is required to represent the evaluation process
in a JSON format. After parsing the generatedJSON-formatted string into a Python list object
using json.loads() , we can easily extract the in-
termediate results (e.g., # claims (un)supported by
reference) by querying the resulting list object.
Casting the evaluation process into generation
of evaluation trajectory not only streamlines the
pipeline, but also facilitates further finetuning of
the model.
3.3 Reinforcement Learning with Ranking
Objective
Finetuning models with valid evaluation trajec-
tories as outlined in §3.2, presents a non-trivial
challenge due to the prohibitive cost of manual
annotation—particularly for claim decomposition
and supportiveness judgment. To address this, we
use rule-based reinforcement learning (DeepSeek-
AI et al., 2025) to finetune our model, without the
need for annotation of the whole trajectory.
Nonetheless, constructing labeled data for re-
ward calculation remains necessary. A naive way
would be to annotate the score according to Eq.1
or 2. However, this way still relies on the claim
decomposition and supportiveness judgment as the
intermediate results. To circumvent this, our rule-
based reinforcement learning method introduces a
novel optimization paradigm that trains the model
to perform relative ranking of candidate responses
based on their degree of supportiveness w.r.t the
reference, rather than predicting absolute scores.
Specifically, first, given question qand reference
c, we synthesize a set of responses {y}with vary-
ing groundness degree w.r.t c(§3.3.1). During this
process, the ground-truth rank of {y}can be ob-
tained naturally. Subsequently, based on qandc,
we adopt {y}as the candidates to rank and ap-
ply rule-based RL to reinforce the model’s ranking
ability by advancing the generated evaluation tra-
jectories (§3.3.2). Note that the RL objective is to
rank the responses instead of predicting their exact
scores. This can mitigate the adverse effects of bias
introduced during data synthesis on the training.

3.3.1 Responses Synthesis with Ranking
Relation
With Context-Aware Decoding (Li et al., 2022; Shi
et al., 2023) , the i-th token of a response yis
sampled as
yi∼softmax [(1 + α)PLLM(∗ |q,c,y<i)
−αPLLM(∗ |q,y<i)].(3)
The weight αcontrols the extent in which the gener-
ation of yiis conditioned on c(which is the passage
in this case), and a larger one translates into rthat
is more reference-conditioned. Note that αcan be
negative ( α <0), which leads to reference-resistant
generation of y.
For each question, we synthesize a set of re-
sponses {yai}with different degrees of ground-
ness by varying α. The ground-truth ranking of
these responses can be obtained naturally as
∀αi, αj∈R, α i> αj=⇒yαi≻yαj.(4)
For implementation, PLLM(∗ |q,c,r<i)and
PLLM(∗ |q,r<i)are modeled using in-context
learning. We use a third-party LLM for sampling
candidate responses prior to the RL stage.
3.3.2 Rule-Based RL
We fine-tune the model with rule-based RL. Partic-
ularly, we adopt Group Relative Policy Optimiza-
tion (GRPO, Shao et al., 2024) with rule-based
outcome rewards. During rolling out, with the ques-
tionq, the reference c, and the set of synthesized
responses {yai}as input, the model generates com-
plete evaluation trajectories according to the rules
specified in the prompt.
Reward Design We define three types of rewards,
including format reward, evidence reward, and ac-
curacy reward. The rewards for the rollout of eval-
uation trajectory are defined as follows.
•Format reward assesses the completeness of
the evaluation trajectory. rfis0if the string
of evaluation trajectory satisfies all the fol-
lowing requirements: 1) it can be parsed into
a Python List object using json.loads() ;
2) the items in the list correspond exactly to
the set of candidate responses; 3) each item
within the List is aDict object containing
all required fields as specified in the prompt
(the circled region in Fig.5); 4) each supported
claim has at least one evidence. Otherwise a
penalty of −0.5is applied.•Evidence reward measures how verbatim ex-
tracted evidence spans are cited from the ref-
erence. The reward for each span is defined as
the length of its longest common substring
with the reference text, normalized by the
span’s length1. An evidence span of length
less than 10 receives reward 0. The evidence
reward of an evaluation trajectory reis the av-
erage over all evidence spans in the trajectory.
•Accuracy reward evaluates whether the rank-
ing based on evaluation scores inferred by the
model is correct. The evaluation score is de-
rived as S(y) =# Claims of ysupported by c
# Claims of y.The
accuracy reward rais1if the ranking aligns
with the ground-truth ranking, or 0otherwise.
Formally,
ra=

1,ifS(yαi)> S(yαj),∀yαi,yαj∈ {y}and
yαi≻yαj
0,otherwise
(5)
Note that the intermediate results required for
reward calculation can be accessed by visiting
the object parsed using json.loads() . For in-
stance, for each candidate response, we apply
simple Python operations to enumerate all en-
tries in its atomic_claims list and verify their
is_supported values. The JSON-formatted out-
put demonstrates superior precision in results ex-
traction, compared to traditional regular expression-
based approaches.
Taking together above three rewards, the combined
reward rfor a rollout is
r=

1 + 0 .5∗re,ifrf= 0andra= 1,
0, ifrf= 0andra= 0,
−0.5, otherwise .(6)
The reward function encourages the model to
rank the candidate responses more accurately
through optimizing the evaluation trajectories.
Curriculum Learning Intuitively, it is more chal-
lenging to rank a larger set of candidate responses.
In the spirit of curriculum learning (Bengio et al.,
2009; Narvekar et al., 2020), to facilitate smooth
and incremental learning, we gradually escalate the
complexity of the ranking task by increasing the
number of candidate responses as the RL training
process advances.
1The length of a sequence is computed as its total token
count

4 Experiment Settings
4.1 Benchmarks
Faithfulness We assess the faithfulness judgment
performance of different evaluation approaches on
WikiEval dataset (Shahul et al., 2023), which con-
tains question-context-answer triples with human-
annotated judgments. The questions are formu-
lated from 50 Wikipedia pages, and for each ques-
tion, ChatGPT generates two answers: one with
Wikipedia context and one without. Two human an-
notators then judge which answer is more faithful
to the source, reaching 95% agreement.
Correctness To assess different correctness eval-
uation approaches, we use the Meta Evaluation
Dataset constructed by Ru et al. (2024). The dataset
contains 280 instances from 10 domains. Each
instance includes a question, the ground-truth an-
swer, and a pair of responses generated by two
RAG systems2. Two human annotators assess the
responses, assigning preference labels from five
relative choices: significantly better, slightly bet-
ter, tie, slightly worse and significantly worse. We
adopt human-annotated correctness preferences as
the references to benchmark evaluation methods.
4.2 Metrics
Faithfulness For each WikiEval instance, the eval-
uators are required to identify the more faithful
answer between two candidates. Evaluator perfor-
mance is then measured as the percentage of cases
where the evaluators’ preference aligns with the
human annotators’ judgment (Shahul et al., 2023;
Ispas et al., 2025). We follow Shahul et al. (2023)
to handle possible tieswith three scenarios (see
App. A.3 for more details):
•Best-Case : Measures the frequency of eval-
uators assigning greater or equal faithfulness
scores to good answers over poor ones.
•Worst-Case : Computes the frequency of
strictly greater faithfulness scores assigned
to good answers.
•Middle-Case : Adopts ternary scoring with a
partial point of 0.5for ties.
Correctness Following Ru et al. (2024), we covert
the human-annotated correctness preference labels
(five relative choices) into a numerical score dif-
ference for each response pair, i.e., hi=H(r2
i)−
2https://github.com/amazon-science/RAGChecker/
blob/main/data/meta_evaluation/H(r1
i)∈ {− 2,−1,0,1,2}. A normalized score
difference is computed as ei=f(E(r2
i)−E(r1
i))
for each evaluation approach, where E(·)is the cor-
rectness score measured by the evaluator and f(·)
is a linear normalization function. To assess the
performance of different evaluation methods, we
compute three correlation coefficients between hu-
man judgments hiand system scores ei: Pearson’s
r, Spearman’s ρ, and Kendall’s τ.
4.3 Implementation Details
Responses Synthesis We use Natural Question
dataset (Kwiatkowski et al., 2019) to synthesize
the responses, where each question is accompa-
nied by a grounding passage. 5,500 instances are
selected for response synthesis. For each α∈
{0,−0.5,−1,−1.4}, we synthesize a response
according to Eq.3, using Qwen2.5-7B-Instruct
(Qwen, 2024). See App. A.1 for more details.
Training We fine-tune our model based on
Qwen2.5-7B-Instruct . For RL training, the sam-
ple number is 8 and temperature is 1 during roll-
out. The KL coefficient in the learning objective is
0.015. We train the model for a total of 2 epochs
for training. To achieve curriculum learning for
RL, we use 3 candidate responses for ranking in
the first epoch and increase this to 4 in the second
epoch. More details can be found in App. A.2.
Inference For test instances on both datasets de-
scribed in §4.1, similar to rolling out at training
stage, the evaluator model takes as input the ques-
tion, the reference text (ground-truth answer for
correctness and retrieved passage for faithfulness),
and two candidate responses; During the generation
of evaluation trajectory, we use nucleus sampling
(Holtzman et al., 2019) with p= 0.9and tempera-
ture= 0.1. For those generated sequence that fails
to parse, we utilize regular expressions to extract
the required results. The correctness/faithfulness
score for a response yis computed as S(y)(Eq.5,
see Fig.7 for an example).
4.4 Baselines
We compare our approach with a comprehensive
set of baseline evaluation methods, including
non-LLM based and LLM-based paradigms.
For non-LLM based methods, we report BLEU
(Papineni et al., 2002) and ROUGE-L (Lin,
2004) as representative n-gram based metrics,
as well as BERTScore(Zhang et al., 2020)
for embedding-based metric. For LLM-based
evaluation, we include recent frameworks that

Method Model (-Instruct) Best Middle Worst
BLEU – 0.860 0.860 0.860
RougL – 0.900 0.900 0.900
BERTScore – 0.900 0.900 0.900
ARESllama-70b 1.000 0.920 0.840
qwen-72b 1.000 0.928 0.856
gpt-4o 1.000 0.956 0.912
TruLensllama-70b 1.000 0.860 0.720
qwen-72b 0.984 0.830 0.676
gpt-4o 1.000 0.940 0.900
RAGASllama-70b 0.960 0.910 0.860
qwen-72b 0.960 0.922 0.884
gpt-4o 0.980 0.940 0.900
RAG-Checkerllama-70b 1.000 0.962 0.924
qwen-72b 0.976 0.936 0.896
gpt-4o 0.973 0.933 0.893
OpenEval* llama-70b 0.960 0.950 0.940
SFT qwen-72b 0.828 0.828 0.828
RAG-Zeval
w/o RLllama-70b 0.980 0.960 0.927
qwen-72b 0.993 0.957 0.883
qwen-7b 0.932 0.930 0.858
RAG-Zeval qwen-7b 1.000†0.992†0.984†
Table 1: Performance on faithfulness evalu-
ation. We assess different methods using
Llama3.1-70B-Instruct ,Qwen2.5-70B-Instruct ,
GPT-4o and/or Qwen2.5-7B-Instruct . Non- GPT
results are averaged over five trials to mitigate
randomness. Due to API cost, we ran GPT-4o three
times for each method. We cite results of OpenEval
from the original paper (Ispas et al., 2025). †indicates
the result is statistically significant at the level of 0.01.
all use iterative prompting with large language
models as evaluators. ARES (Saad-Falcon et al.,
2024) and TruLens (Ferrara et al., 2024) are
non-claim-based, directly prompting the LLM
for overall or aspect-based scores. RAGAS
(Shahul et al., 2023), RAG-Checker (Ru et al.,
2024), and OpenEval (Ispas et al., 2025) are
claim-based, decomposing responses into factual
claims for individual assessment. All LLM-
based baselines use Llama3.1-70B-Instruct,
Qwen2.5-70B-Instruct, GPT-4o as the
evaluator backbone. In addition, we con-
sider standard SFT, which directly fine-tunes
Qwen2.5-7B-Instruct to replicate the relative
ranking of responses, using the same synthetic data
described in Section 3.3.2 (see App. A.4 for more
details).
5 Main Experiments
Comparison with Baselines Table 1 and 2 present
the performances of RAG-Zeval and baseline eval-
uators. Generally, the claim-based methods out-Method Model (-Instruct) Pearson Spearman Kendall
BLEU – 0.302 0.305 0.236
RougL – 0.395 0.428 0.335
BERTScore – 0.350 0.437 0.341
ARESllama-70b 0.350 0.328 0.296
qwen-72b 0.423 0.396 0.360
gpt-4o 0.382 0.370 0.333
TruLensllama-70b 0.428 0.453 0.366
qwen-72b 0.428 0.446 0.360
gpt-4o 0.396 0.390 0.312
RAGAS embedding 0.411 0.432 0.283
RAG-Checkerllama-70b 0.463 0.425 0.337
qwen-72b 0.495 0.465 0.375
gpt-4o 0.499 0.459 0.369
SFT qwen-72b 0.359 0.350 0.320
RAG-Zeval
w/o RLllama-70b 0.492 0.443 0.351
qwen-72b 0.521†0.482†0.388†
qwen-7b 0.427 0.367 0.312
RAG-Zeval qwen-7b 0.501†0.452†0.354†
Table 2: Performance on correctness evaluation. Corre-
lation between different methods and human judgments
are reported. We assess RAGAS (Shahul et al., 2023)
with Text-Embedding-Ada-002 model (Neelakantan
et al., 2022) following the original setting. Other set-
tings are the same as Tab.1. Following Ru et al. (2024),
we only show the metric with the best correlation for
each baseline framework. See more details in App. A.4.
†indicates the result is statistically significant at the
level of 0.01.
perform non-claim-based ones. For both bench-
marks, RAG-Zeval has the strongest correlation
with human preference in terms of almost all met-
rics. Despite its compact architecture (7 billion
parameters), RAG-Zeval demonstrates superior per-
formance over most baselines built on large-scale
LLMs with 10-100 ×more parameters. This result
validates the effectiveness of our approach for en-
hancing evaluation capabilities in compact LLMs.
For in-depth comparison, Fig. 2 visualizes the
distribution of scores assigned by RAG-Zeval and
some baselines that give numerical (instead of cat-
egorical) predictions for faithfulness evaluation,
where we can see the distribution of faithful and
unfaithful responses3. While TruLens has the most
concentrated distribution near 1 for faithful re-
sponses, its distribution for unfaithful responses
disperses evenly across the X-axis, indicating its in-
ability to distinguish the unfaithful responses. For
faithful responses, RAG-Zeval , RAG-Checker and
RAGAS demonstrate similar distributional shapes,
particularly showing comparable peakedness near
3The correctness evaluation benchmark is not used here,
due to the fact that human annotators only provide relative
assessment (e.g., preference ranking) rather than absolute cat-
egorical judgments (correct/incorrect labels).

0.0 0.2 0.4 0.6 0.8 1.0
Scores012345678DensityRAG-Zeval
Faithful responses
Unfaithful responses
0.0 0.2 0.4 0.6 0.8 1.0
Scores0.02.55.07.510.012.515.017.520.0DensityTruLens
Faithful responses
Unfaithful responses
0.0 0.2 0.4 0.6 0.8 1.0
Scores012345678DensityRAGAS
Faithful responses
Unfaithful responses
0.0 0.2 0.4 0.6 0.8 1.0
Scores0123456DensityRAG-Checker
Faithful responses
Unfaithful responsesFigure 2: The density distribution of the scores assigned by the faithfulness evaluators.The distribution of the faithful
and unfaithful responses are marked with red and blue, respectively. TruLens, RAGAS and RAG-Checker are all
implemented with Qwen2.5-72B-Instruct as the backbone LLM.
Objective Pearson Spearman Kendall
Ranking (Ours) 0.501†0.452†0.354†
Predicting the best 0.406 0.393 0.311
Table 3: Comparison of two training objectives. The
results are obtained by averaging across 5 runs.
1. However, RAG-Zeval shows superior discrimina-
tive capacity, maintaining clear separation between
faithful and unfaithful response distributions.
Comparison with Ablated Variants As shown in
Tab. 1 and 2, SFT on ground-truth ranking exhibits
the worst performances among the LLM-based
methods, implying the significance of intermediate
reasoning. On the other hand, even without further
training, the non-RL variant of our approach main-
tains robust evaluation performance. This suggests
that our rule-guided generation approach, which
generates the reasoning trajectories for evaluation
end-to-end, can effectively harness the LLMs’ rea-
soning capabilities to achieve superior evaluation
performances. This advantage gets more promi-
nent through RL training, ultimately leading to
superior or comparable results compared to large-
scale LLM-based baselines. More discussion on
rule-guided reasoning is in § 6.3.
6 Analysis
The following problems are discussed. 1) How
reinforcement learning stimulates the model’s eval-
uation ability. 2) What is the effect of the task
complexity represented by training objective and
data. 3) What is the effectiveness of the rule-guided
reasoning. Here we employ the correctness bench-
mark as the testbed, since it is more challenging.
6.1 Self-Evolution of Evaluation Abilities
To investigate how the model abilities evolve over
the RL training process, we continuously monitor
the model’s behaviors in claim decomposition, evi-
0 40 80 120 160 200 240 280 320343
Step9.510.010.511.011.512.012.5Count
(a)
Claim count9.5310.6911.74
10.8410.6412.1312.34
11.60 11.33
11.38
0 40 80 120 160 200 240 280 320343
Step0.740.790.840.890.94Score
(b)
Grounding degree of evidence
Entailment degree between 
evidence and claims
0.760.88
0.750.830.870.96
0.860.92
0.830.88
0.760.93
0.800.93
0.840.91
0.850.93
0.870.93Dynamics of RAG-Zeval abilities during trainingFigure 3: (a) shows the changes of decomposed claim
count, while (b) presents the evolution of abilities of ev-
idence extraction and supportiveness judgment through-
out the RL training process. The statistics are based on
the rollout samples during training.
Data configuration Pearson Spearman Kendall
Curriculum learning
(first 3 and then 4 responses)0.501†0.452†0.354†
Static (3 responses) 0.457 0.433 0.339
Static (4 responses) 0.450 0.402 0.314
Table 4: Comparison of different data configuration.
The results are obtained by averaging across 5 runs.
dence extraction, supportiveness judgment. Their
dynamics is plotted in Fig. 3.
The blue line depicts the average number of
claims decomposed from a response, which ini-
tially exhibits a sharp increase and ends at a sta-
ble level. As finer-grained claim decomposition
enables more discriminative comparisons among
candidate responses, the model learns increasingly
comprehensive claim decomposition for enhanced
ranking performance. As evidenced by the case
study in Fig. 6 and 7, the already trained check-
point (at step 343) provides a more comprehensive
decomposition, whereas some claims by the un-
trained checkpoint (at step 0) amalgamate atomic
claims that should be addressed separately.
For each extracted evidence, we measure its
grounding degree in terms of the normalized length
of its longest common substring (similar to evi-

Method Generation Pearson Spearman Kendall
Ours The complete reasoning trajectory specified in Fig.5 0.501†0.452†0.354†
w/o evidence The reasoning trajectory without evidence and analysis 0.489 0.452 0.353
SFT Response ranking result 0.359 0.350 0.320
Table 5: Results of methods with different evaluation pattern, obtained by averaging across 5 runs.
dence reward definition in §3.3.2). For each sup-
ported claim generated by the model, we quantify
the degree of textual entailment between its corre-
sponding evidence and the claim itself, using Align-
Score4(Zha et al., 2023). The grounding (red line)
and entailment (green line) degrees both experience
a notable growth, implying that the model’s capa-
bilities of evidence extraction and supportiveness
judgment get improved through the RL training.
Overall, reinforcement learning effectively incen-
tivizes the development of reasoning capabilities
essential for responses evaluation , consequently
improving final evaluation performance.
6.2 Effect of Task Complexity
Ranking-based Objective We simplify the
ranking-based accuracy reward (Eq. 5) as
ra=

1,ifS(yαi)> S(yαj), αi=max{α},
∀yαj∈ {y}andyαi̸=yαj
0, otherwise
(7)
indicating that an accuracy reward of 1is earned if
the model assigns the highest evaluation score to y
with the largest αvalue. This simplified formula-
tion, similar to the one used by Liu et al. (2025c),
does not require an correct ranking over the en-
tire set of candidate responses. The comparison
between the two formulation is shown in Tab. 3.
The model trained with the simplified accuracy
reward suffers a notable performance drop. This
implies that reducing task complexity may diminish
the incentive for the model to develop enhanced
evaluation capabilities . Because a finer-grained
and more discriminative assessment of responses
is crucial for achieving a comprehensive ranking.
Curriculum Learning During the RL training,
we organize the training data in a way that the
complexity of the ranking task escalates as the
training advances. To study the effect of this prac-
tice, we train models with the following two static
data organization—the training instance across all
epochs consistently contains 3 or 4 candidate re-
sponses for ranking. As illustrated in Tab. 4, the
4The evidence is input as context , and the claims is as
claim .context andclaim are two arguments for AlignScore
which measures how likely context would entails claim .curriculum learning-based configuration has the
best performance. Its improvement over the static
one with 3 responses further corroborates above
finding that increased task complexity may help
ability acquisition. However, the static configura-
tions with 4 response performs worst. We found it
earns a much lower average combined reward than
the curriculum learning-based configuration in the
first epoch (seen in App.B). Employing overly chal-
lenging task objective in the initial training stage
may suppress model learning, as it is less possible
to find a valid rollout and the model then hardly
receives positive feedback during the training.
6.3 Effectiveness of Rule-Guided Reasoning
Results in § 5 demonstrates that RAG-Zeval out-
performs direct SFTon ground-truth ranking of re-
sponses. To better illustrate the significance of our
rule-guided reasoning, we further introduce an in-
termediate variant between the above two methods–
remove the requirement to provide supporting evi-
dence and corresponding analysis, while maintain-
ing all other settings consistent with RAG-Zeval .
As shown in Tab. 5, their performances are pos-
itively correlated with the level of detail of their
generation, which substantiates the advantage of
the rule-guided reasoning. Analogues to Chain-
of-Thought (Wei et al., 2022), RAG-Zeval benefits
from the stepwise reasoning in its evaluation tra-
jectories. Also, detailed evaluation processes offer
better explainability behind the model predictions.
7 Conclusion
In this work, we introduce RAG-Zeval , a novel rule-
guided evaluation framework that enables compact
LLMs to perform end-to-end, interpretable assess-
ment of RAG system outputs. Our approach trains
evaluators via reinforcement learning with a novel
ranking-based objective, bypassing the requirement
for human-annotated data. Through comprehen-
sive experiments on benchmarks of faithfulness
and correctness evaluation, we demonstrate that
our approach achieves strong alignment with hu-
man judgments, outperforming current large-scale
LLM-based baselines while maintaining a much

smaller model scale. The result highlights the po-
tential of compact, reasoning-driven evaluators for
scalable and transparent RAG evaluation.
Limitations
This work has several limitations that point to av-
enues for future improvement. Although our ap-
proach employs smaller models compared to direct
use of large LLMs, the RL training process still
requires considerable computational resources and
access to high-performance hardware, which may
not be available to all researchers. In addition,
our current experiments are primarily conducted in
English and on general-domain datasets; the gen-
eralizability of the evaluator to other languages
or specific domains remains to be explored. Fur-
ther validation on multilingual and domain-specific
benchmarks would strengthen the robustness and
applicability of our method.
Additionally, our experiments run on static
datasets, which may not capture real-world dy-
namic interactions well (e.g., adversarial inputs,
evolving user preferences). Further investigation
of its performance in real-world environments is
essential prior to deployment, to ensure unbiased
and accurate judgments.
Ethical Considerations
Our RL-based RAG evaluation framework also
raises several ethical considerations. The com-
putational requirements, though reduced com-
pared to large LLMs, may still create barriers
for less-resourced groups, potentially exacerbat-
ing inequities in access to advanced evaluation
tools. Moreover, automated evaluation should not
be viewed as a substitute for human oversight, espe-
cially in high-stakes or sensitive applications, as it
may overlook nuanced ethical or contextual factors.
Besides, if the synthetic or training data used for
evaluator construction contains biases or unrepre-
sentative patterns, these biases may be propagated
in the evaluation results. Responsible deployment
requires ongoing attention to these issues and a
commitment to transparency and fairness
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations .Yoshua Bengio, Jérôme Louradour, Ronan Collobert,
and Jason Weston. 2009. Curriculum learning. In
Proceedings of the 26th Annual International Confer-
ence on Machine Learning , ICML ’09, page 41–48,
New York, NY , USA. Association for Computing
Machinery.
DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang,
Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhi-
hong Shao, Zhuoshu Li, Ziyi Gao, and 181 others.
2025. Deepseek-r1: Incentivizing reasoning capa-
bility in llms via reinforcement learning. Preprint ,
arXiv:2501.12948.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. Preprint ,
arXiv:2405.06211.
J. Ferrara, Ethan-Tonic, and O. M. Ozturk. 2024. The
rag triad. https://www.trulens.org/trulens_
eval/core_concepts_rag_triad/ .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models. Preprint , arXiv:2407.21783.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, Yuanzhuo Wang, and
Jian Guo. 2024. A survey on llm-as-a-judge. ArXiv ,
abs/2411.15594.
Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang,
Youran Sun, Yi Zhu, Fan Yang, and Mao Yang.
2025. rstar-math: Small llms can master math rea-
soning with self-evolved deep thinking. Preprint ,
arXiv:2501.04519.
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and
Yejin Choi. 2019. The curious case of neural text
degeneration. ArXiv , abs/1904.09751.
Alex-R ˘azvan Ispas, Charles-Elie Simon, Fabien Cas-
pani, and Vincent Guigue. 2025. Towards lighter and
robust evaluation for retrieval augmented generation.
The Next Frontier in Reliable AI": Workshop on ICLR
2025 , abs/2503.16161.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur P. Parikh, Chris Alberti,

Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc V . Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. Preprint , arXiv:2005.11401.
Kun Li, Tianhua Zhang, Yunxiang Li, Hongyin Luo, Ab-
dalla Moustafa, Xixin Wu, James Glass, and Helen
Meng. 2025. Generate, discriminate, evolve: Enhanc-
ing context faithfulness via fine-grained sentence-
level self-evolution. Preprint , arXiv:2503.01695.
Kun Li, Tianhua Zhang, Xixin Wu, Hongyin Luo, James
Glass, and Helen Meng. 2024. Decoding on graphs:
Faithful and sound reasoning on knowledge graphs
through generation of well-formed chains. Preprint ,
arXiv:2410.18415.
Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang,
Jason Eisner, Tatsunori Hashimoto, Luke Zettle-
moyer, and Mike Lewis. 2022. Contrastive decoding:
Open-ended text generation as optimization. In An-
nual Meeting of the Association for Computational
Linguistics .
Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harri-
son Edwards, Bowen Baker, Teddy Lee, Jan Leike,
John Schulman, Ilya Sutskever, and Karl Cobbe.
2024. Let’s verify step by step. In The Twelfth Inter-
national Conference on Learning Representations .
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. In Text Summariza-
tion Branches Out , pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi,
Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin.
2025a. Understanding r1-zero-like training: A criti-
cal perspective. Preprint , arXiv:2503.20783.
Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong
Ruan, Peng Li, Yang Liu, and Yu Wu. 2025b.
Inference-time scaling for generalist reward model-
ing. Preprint , arXiv:2504.02495.
Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong
Ruan, Peng Li, Yang Liu, and Yu Wu. 2025c.
Inference-time scaling for generalist reward model-
ing.
Ilya Loshchilov and Frank Hutter. 2017. Decoupled
weight decay regularization. In International Confer-
ence on Learning Representations .Potsawee Manakul, Adian Liusie, and Mark Gales. 2023.
SelfCheckGPT: Zero-resource black-box hallucina-
tion detection for generative large language models.
InProceedings of the 2023 Conference on Empiri-
cal Methods in Natural Language Processing , pages
9004–9017, Singapore. Association for Computa-
tional Linguistics.
Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko
Sinapov, Matthew E. Taylor, and Peter Stone. 2020.
Curriculum learning for reinforcement learning do-
mains: a framework and survey. 21(1).
Arvind Neelakantan, Tao Xu, Raul Puri, Alec Rad-
ford, Jesse Michael Han, Jerry Tworek, Qiming
Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy,
Johannes Heidecke, Pranav Shyam, Boris Power,
Tyna Eloundou Nekoul, Girish Sastry, Gretchen
Krueger, David Schnurr, Felipe Petroski Such, Kenny
Hsu, and 6 others. 2022. Text and code em-
beddings by contrastive pre-training. Preprint ,
arXiv:2201.10005.
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal,
Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale-
man, Diogo Almeida, Janko Altenschmidt, Sam Alt-
man, Shyamal Anadkat, Red Avila, Igor Babuschkin,
Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim-
ing Bao, Mohammad Bavarian, Jeff Belgum, and
262 others. 2024. Gpt-4 technical report. Preprint ,
arXiv:2303.08774.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. In Proceedings of the
40th Annual Meeting of the Association for Compu-
tational Linguistics , pages 311–318, Philadelphia,
Pennsylvania, USA. Association for Computational
Linguistics.
Nicholas Pipitone and Ghita Houir Alami. 2024.
Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain. ArXiv ,
abs/2408.10343.
Qwen. 2024. Qwen2.5: A party of foundation models.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang,
Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxi-
ang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang,
Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang,
Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024.
Ragchecker: A fine-grained framework for diagnos-
ing retrieval-augmented generation. NeurIPS .
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. In Proceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers) , pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.

ES Shahul, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2023. Ragas: Automated evalu-
ation of retrieval augmented generation. In Confer-
ence of the European Chapter of the Association for
Computational Linguistics .
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Jun-
Mei Song, Mingchuan Zhang, Y . K. Li, Yu Wu, and
Daya Guo. 2024. Deepseekmath: Pushing the limits
of mathematical reasoning in open language models.
ArXiv , abs/2402.03300.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2024. Hybridflow: A flexible
and efficient rlhf framework. arXiv preprint arXiv:
2409.19256 .
Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia
Tsvetkov, Luke Zettlemoyer, and Scott Yih. 2023.
Trusting your evidence: Hallucinate less with context-
aware decoding. In North American Chapter of the
Association for Computational Linguistics .
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries. Preprint , arXiv:2401.15391.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Proceedings of the 36th International Conference on
Neural Information Processing Systems , NIPS ’22,
Red Hook, NY , USA. Curran Associates Inc.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong,
Qi Liu, and Zhaofeng Liu. 2024. Evaluation of
retrieval-augmented generation: A survey. ArXiv ,
abs/2405.07437.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2025a. Evaluation of Retrieval-
Augmented Generation: A Survey , page 102–120.
Springer Nature Singapore.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong
Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin,
Bole Ma, Guangming Sheng, Yuxuan Tong, Chi
Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, and
16 others. 2025b. Dapo: An open-source llm re-
inforcement learning system at scale. Preprint ,
arXiv:2503.14476.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. In Proceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 11328–11348, Toronto, Canada. Association
for Computational Linguistics.
Tianyi Zhang, Varsha Kishore*, Felix Wu*, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Eval-
uating text generation with bert. In International
Conference on Learning Representations .Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chun-
yan Miao. 2025. Medrag: Enhancing retrieval-
augmented generation with knowledge graph-elicited
reasoning for healthcare copilot. In Proceedings of
the ACM on Web Conference 2025 , pages 4442–4457.
Jialun Zhong, Wei Shen, Yanzeng Li, Songyang Gao,
Hua Lu, Yicheng Chen, Yang Zhang, Wei Zhou, Jin-
jie Gu, and Lei Zou. 2025. A comprehensive sur-
vey of reward models: Taxonomy, applications, chal-
lenges, and future. Preprint , arXiv:2504.12328.

A Implementation Details
A.1 Response Synthesis
Starting from Natural Question dataset
(Kwiatkowski et al., 2019), we first filter
out the instances with a passage that has
more than 6,000 tokens. We then randomly
select 5,500 instances from the remaining in-
stances. For each α∈ {0,−0.5,−1,−1.4},
we synthesize a response according to Eq. 3,
using Qwen2.5-7B-Instruct (Qwen, 2024).
PLLM(∗ |q,c)andPLLM(∗ |q)are modeled
using in-context learning, and the in-context
prompts are shown in Tab.9. Greedy search is used
for sampling tokens.
A.2 RAG-Zeval
We utilize VERL (Sheng et al., 2024), a open-
source library, to apply RL training to the models.
The training runs on 8 H20 GPUs and takes ap-
proximately 20 hours. The hyperparameters for the
training are listed in Tab. 6.
Hyperparameters
Training batch size 32
Optimizer AdamW
(Loshchilov and Hutter, 2017)
Learning rate 1e-6
Warmup step 10
Gradient accumulation step 1
Learning rate scheduler Linear
KL coefficient 0.015
Rollout temperature 1
Rollout number 8
Rollout maximum length 8192
Total epoch 2
Table 6: The settings of hyperparameters used in the RL
training.
A.3 Faithfulness Metric
When assessing the faithfulness evaluation perfor-
mance of different methods, we follow Shahul et al.
(2023) to handle possible tieswith three scenarios:
•Best-Case : Measures the frequency of eval-
uators assigning greater or equal faithfulness
scores to good answers compared to poor
ones.
best=1
nnX
i=1I[F(goodi)≥F(poori)]
•Worst-Case : Computes the frequency of
strictly greater faithfulness scores assignedto good answers.
worst =1
nnX
i=1I[F(goodi)>F(poori)]
•Middle-Case : Adopts ternary scoring with a
partial point of 0.5for ties.
middle =1
nnX
i=1{I[F(good i)>F(poor i)]
+0.5·I[F(good i) = F(poor i)]}
A.4 Baselines
Forstandard SFT baseline, we enhance model
generalizability across varying number of candi-
date answers by randomly partitioning the train-
ing data (described in §3.3.2) into three subsets:
pairwise (2 responses), triplet (3 responses) and
quadruplet (4 responses) ranking tasks. Each sub-
set contains approximately equal data volume as
reported in Tab.7. The model is trained to repro-
duce the relative ranking of responses based on
their faithfulness with respect to the grounding pas-
sage.
Pairwise Triplet Quadruplet Total
Question 647 970 3883 5500
Instance 3877 3874 3883 11634
Table 7: Data statistics for standard supervised fine-
tuning. Each original question includes four generated
responses in different faithfulness levels, yielding six
pairwise, four triplet, and one quadruplet ranking in-
stance per question. The final row reports deduplicated
instance counts.
For correctness evaluation, not all baseline eval-
uation framework has a direct correctness metric.
For RAG-Checker (Ru et al., 2024), we report
the performance using the precision metric, which
aligns with our definition of correctness. For base-
lines without a direct correctness metric, we fol-
low the setting in Ru et al. (2024) to report the
best correlation among all metrics in Tab.2. Tab.8
shows the complete results of Trulens (Ferrara et al.,
2024) and ARES (Saad-Falcon et al., 2024) using
Qwen2.5-72B-Instruct .
B Training Dynamics
Figure 4 presents the reward progression during
RL training under different data configurations (de-
tailed in Tab.4). The static 4-response configuration
initially yields significantly lower average rewards
compared to other conditions, attributable to its

Method Model Pearson Spearman Kendall
ARESRelevancy 0.423 0.396 0.360
Faithfulness 0.372 0.356 0.320
TrulensRelevancy 0.368 0.320 0.289
Faithfulness 0.428 0.446 0.360
Table 8: Complete results of Trulens and ARES
with Qwen2.5-72B-Instruct on correctness evalua-
tion. Performance is averaged over five trials to mitigate
randomness.
Figure 4: Reward dynamics of RL training with dif-
ferent data configuration. The red line represents the
curriculum learning settings, while the green and blue
lines are for static 3 and 4 responses, respectively.
greater task complexity. The curriculum-based ap-
proach (red series) experiences an expected per-
formance dip at Step 175 during the transition to
4 candidate responses, yet maintains superior av-
erage rewards over the static 4-response baseline
throughout subsequent training.
C Prompt
The prompt used in reinforcement learning is
shown in Fig.5, elaborating the rules that the evalu-
ator should conform to. Given a question, the con-
text and Kcandidate answers to be assessed, the
model should generate a JSON-formatted output
containing detailed evaluation for each candidate
answer. Each answer evaluation should involve
claim decomposition, claim supportive judgment,
grounding evidence generation and relation analy-
sis.
D Case Study
In Fig. 6 and 7, RAG-Zeval and its variant without
RL training ( RAG-Zeval w/o RL ) generate evalua-
tions for the same input. For this question “ price
of PS3 when it first came out ”, human annotators
judge response B as significantly better than re-
sponse A. RAG-Zeval w/o RL assigns response
A1point, while response B 1/3point, resultingin an incorrect ranking A > B , which contradicts
human preference. Additionally, its final claim is
a verbatim copy of the original answer sentences,
failing to perform atomic claim decomposition. In
contrast, RAG-ZEval assigns response A 0point
and response B 3/4point, producing the correct
ranking that aligns with human judgment.

Answer the following questions based on the given passages.
Question: What was the purpose of designing and building the Fiat Ecobasic concept car?
Passage: The Fiat Ecobasic is a concept car designed by the Italian manufacturer Fiat and presented in December 1999 at the Bologna
Motor Show and exhibited in March 2000 at the Geneva Motor Show. The purpose of this concept was to prove that it was possible to
design and build a car capable of transporting four adults in a structure made of fully recyclable composite materials and whose production
and operating costs were ultra-low.
Answer: The purpose of designing and building the Fiat Ecobasic concept car was to prove that it was possible to create a car that could
transport four adults using fully recyclable composite materials. Additionally, the car aimed to have ultra-low production and operating
costs.
Question: When did Pope Benedict XVI become the head of the Catholic Church and sovereign of the Vatican City State, and when did he
resign?
Passage: PPope Benedict XVI (Latin: Benedictus PP. XVI; Italian: Benedetto XVI; German: Benedikt XVI; born Joseph Aloisius
Ratzinger; 16 April 1927 – 31 December 2022) was the head of the Catholic Church and sovereign of the Vatican City State from 19
April 2005 until his resignation on 28 February 2013. Benedict’s election as pope occurred in the 2005 papal conclave that followed
the death of Pope John Paul II. In 1981, he was appointed Prefect of the Congregation for the Doctrine of the Faith, one of the most
important dicasteries of the Roman Curia. From 2002 until he was elected pope, he was also Dean of the College of Cardinals. Before
becoming pope, he had been "a major figure on the Vatican stage for a quarter of a century"; he had had an influence "second to none
when it came to setting church priorities and directions" as one of John Paul II’s closest confidants.Benedict’s writings were prolific and
generally defended traditional Catholic doctrine, values, and liturgy. He was originally a liberal theologian but adopted conservative views
after 1968. During his papacy, Benedict advocated a return to fundamental Christian values to counter the increased secularisation of many
Western countries. He viewed relativism’s denial of objective truth, and the denial of moral truths in particular, as the central problem of
the 21st century. Benedict also revived several traditions, including the Tridentine Mass. He strengthened the relationship between the
Catholic Church and art, promoted the use of Latin, and reintroduced traditional papal vestments, for which reason he was called "the
pope of aesthetics". He was described as "the main intellectual force in the Church" since the mid-1980s.On 11 February 2013, Benedict
announced his resignation, citing a "lack of strength of mind and body" due to his advanced age. His resignation was the first by a pope
since Gregory XII in 1415, and the first on a pope’s initiative since Celestine V in 1294. He was succeeded by Francis on 13 March 2013
and moved into the newly renovated Mater Ecclesiae Monastery in Vatican City for his retirement.
Answer: Pope Benedict XVI became the head of the Catholic Church and sovereign of the Vatican City State on April 19, 2005. He held
this position until his resignation on February 28, 2013.
Question: { question }
Passage: { Passage }
Answer:
Answer the following questions.
Question: What was the purpose of designing and building the Fiat Ecobasic concept car?
Answer: The purpose of designing and building the Fiat Ecobasic concept car was to prove that it was possible to create a car that could
transport four adults using fully recyclable composite materials. Additionally, the car aimed to have ultra-low production and operating
costs.
Question: When did Pope Benedict XVI become the head of the Catholic Church and sovereign of the Vatican City State, and when did he
resign?
Answer: Pope Benedict XVI became the head of the Catholic Church and sovereign of the Vatican City State on April 19, 2005. He held
this position until his resignation on February 28, 2013.
Question: { question }
Answer:
Table 9: Prompts used to model PLLM(∗ |q,c)andPLLM(∗ |q), respectively, for Context-Aware Decoding
approach. The in-context examples are also sourced from Natural Question.

You are an answer evaluator for a RAG system. Given %s candidate answers, your task is to analyze 
the degree to which each answer is supported by the given reference context.
### **Task:**  
Conduct a strict, evidence -based evaluation of answer support using the given context.  
### **Input:**
Question: < current_question >
Context: < given_context >
Answers: <K candidate_answers >
### **Evaluation Steps:** 
For each answer (indexed with letters), follow this structured analysis:
1. Decompose the Answer into Atomic Claims
  - Break down the answer into one or more atomic claims —clear, standalone sentences that each 
convey a single, distinct piece of information.
  - Ensure no pronouns are used; each claim must be self -contained.
  - Avoid overlap between claims; together, they should cover all necessary information without 
gaps or repetition.
2. Assess Claim Support Using Context
  - For each claim, determine whether it is:
    - Supported (directly inferable from the context),
    - Unsupported (contradicted by or unrelated to the context).
3. Provide Grounding Evidence (if Applicable)
  - If a claim is supported or contradicted, extract all relevant verbatim sentences from the 
context as grounding evidence that may either support or contradict it. Output the grounding 
evidence as a list that is comprised of the verbatim sentences.
  - If a claim is unrelated, leave the list of grounding evidence empty.
  
4. Analyze Claim Support
  - For each claim, assess how clearly the context supports or contradicts it.
5. Output Valid JSON ( Parsable  in Python)
  - Output the analysis result in JSON format. It should be in the schema:
  ```json
  [
    {
      "id": <candidate answer id>,
      "answer": <candidate answer>,
      "atomic_claims ": [
        {
          "claim": < atomic_claim >,
          "is_supported ": <true/false>,
          "grounding_evidence ": [
            <sentence>, ...
            ],
          "analysis": <analysis on the relation between the claim and context>
        }, ...
        ]
    },...
  ]
  ```Figure 5: The complete prompt used in training the evaluator. Given the current question, context, and Kcandidate
answers, the evaluator outputs a JSON-formatted string containing detailed evaluation for each candidate answer.
Each evaluation follows the four key steps (highlighted in purple) to assess answer quality.

[
  {
    "id": "A",
    "answer": <response A>,
    "atomic_claims ": [
      {
        "claim": "The specific price of the PlayStation 3 when it first came out is not mentioned 
in the provided content.",
        "is_supported ": true,
        "grounding_evidence ": ["PlayStation3 had two hardware configurations announced: a 20 GB 
model and a 60 GB model, priced at US $499 ( \u20ac499) and US $599 ( \u20ac599)."],
        "analysis": "The context does not explicitly state that the specific price is not 
mentioned, but it does provide the prices for the two hardware configurations. Therefore, the claim 
is supported by the absence of a direct statement about the price not being mentioned."
      }
    ]
  },
  {
    "id": "B",
    "answer": <response B>,
    "atomic_claims ": [
      {
        "claim": "The text does not explicitly mention the price of PS3 when it first came out.",
        "is_supported ": true,
        "grounding_evidence ": ["PlayStation3 had two hardware configurations announced: a 20 GB 
model and a 60 GB model, priced at US $499 ( \u20ac499) and US $599 ( \u20ac599)."],
        "analysis": "The context does not explicitly state that the price is not mentioned, but it 
does provide the prices for the two hardware configurations. Therefore, the claim is supported by 
the absence of a direct statement about the price not being mentioned."
      },
      {
        "claim": "The PS3 was initially criticized for its high price and lack of quality games.",
        "is_supported ": false,
        "grounding_evidence ": [],
        "analysis": "The context does not mention any criticism regarding the price or the quality 
of games. This claim is unsupported."
      },
      {
        "claim": "If you're looking for the price of PS3 at launch, according to other sources, the 
20 GB model was priced at $499 and the 60 GB model was priced at $599 in the US when it was 
released on November 17, 2006.",
        "is_supported ": false,
        "grounding_evidence ": [],
        "analysis": "The context does not provide this information. This claim is unsupported."
      }
    ]
  }
]
Question: price of PS3 when it first came out
Ground-truth: PlayStation3 had two hardware configurations announced: a 20 GB model and a 
60 GB model , priced at US $499 ( € 499 ) and US $599 ( € 599 ).
Response A: The specific price of the PlayStation 3 when it first came out is not 
mentioned in the provided content.
Response B: The text does not explicitly mention the price of PS3 when it first came out. 
However, it does mention that the PS3 was initially criticized for its high price and 
lack of quality games. If you're looking for the price of PS3 at launch, according to 
other sources, the 20 GB model was priced at $499 and the 60 GB model was priced at $599 
in the US when it was released on November 17, 2006.
Figure 6: Case study (Part-1): Human annotators judge Response B as significantly better than Response A for the
given question. RAG-ZEval w/o RL assigns response A 1point, while response B 1/3point, misaligning with
human preference. Additionally, the final claim produced by RAG-ZEval w/o RL is a direct copy of the original
answer sentences, lacking atomic claim decomposition.

[
  {
    "id": "A",
    "answer": <response A>,
    "atomic_claims ": [
      {
        "claim": "The specific price of the PlayStation 3 when it first came out is not 
mentioned.",
        "is_supported ": false,
        "grounding_evidence ": [],
        "analysis": "The context does mention the specific prices of the PS3 models, which 
contradicts this claim."
      }
    ]
  },
  {
    "id": "B",
    "answer": <response B>,
    "atomic_claims ": [
      {
        "claim": "The text does not explicitly mention the price of PS3 when it first came 
out.",
        "is_supported ": true,
        "grounding_evidence ": ["PlayStation3 had two hardware configurations announced: a 
20 GB model and a 60 GB model , priced at US $499 ( \u20ac 499 ) and US $599 ( \u20ac 599 
)."],
        "analysis": "The context provides the prices of the PS3 models, but it does not 
explicitly state that these are the prices when it first came out."
      },
      {
        "claim": "The PS3 was initially criticized for its high price and lack of quality 
games.",
        "is_supported ": false,
        "grounding_evidence ": [],
        "analysis": "This claim is not supported by the context and is an additional piece 
of information not present in the given context."
      },
      {
        "claim": "The 20 GB model was priced at $499 in the US when it was released on 
November 17, 2006.",
        "is_supported ": true,
        "grounding_evidence ": ["PlayStation3 had two hardware configurations announced: a 
20 GB model and a 60 GB model , priced at US $499 ( \u20ac 499 ) and US $599 ( \u20ac 599 
)."],
        "analysis": "The context directly supports this claim by providing the exact prices 
of the PS3 models."
      },
      {
        "claim": "The 60 GB model was priced at $599  in the US when it was released on 
November 17, 2006..",
        "is_supported ": true,
        "grounding_evidence ": ["PlayStation3 had two hardware configurations announced: a 
20 GB model and a 60 GB model , priced at US $499 ( \u20ac 499 ) and US $599 ( \u20ac 599 
)."],
        "analysis": "The context directly supports this claim by providing the exact prices 
of the PS3 models."
      }
    ]
  }
]Figure 7: Case study (Part-2): For the same question and responses (shown in Fig. 6), RAG-Zeval assigns response
A0point and response B 3/4point, producing a ranking consistent with human judgment.