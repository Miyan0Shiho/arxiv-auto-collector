# RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning

**Authors**: Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Julia Hockenmaier, Tong Zhang

**Published**: 2025-03-17 02:53:42

**PDF URL**: [http://arxiv.org/pdf/2503.12759v1](http://arxiv.org/pdf/2503.12759v1)

## Abstract
Recent research highlights the challenges retrieval models face in retrieving
useful contexts and the limitations of generation models in effectively
utilizing those contexts in retrieval-augmented generation (RAG) settings. To
address these challenges, we introduce RAG-RL, the first reasoning language
model (RLM) specifically trained for RAG. RAG-RL demonstrates that stronger
answer generation models can identify relevant contexts within larger sets of
retrieved information -- thereby alleviating the burden on retrievers -- while
also being able to utilize those contexts more effectively. Moreover, we show
that curriculum design in the reinforcement learning (RL) post-training process
is a powerful approach to enhancing model performance. We benchmark our method
on two open-domain question-answering datasets and achieve state-of-the-art
results, surpassing previous SOTA generative reader models. In addition, we
offers empirical insights into various curriculum learning strategies,
providing a deeper understanding of their impact on model performance.

## Full Text


<!-- PDF content starts -->

RAG-RL: Advancing Retrieval-Augmented
Generation via RL and Curriculum Learning
Jerry Huang1∗, Siddarth Madala1, Risham Sidhu1, Cheng Niu2,
Julia Hockenmaier1,Tong Zhang1
1University of Illinois at Urbana-Champaign,2NewsBreak
{jerry8, smadala2, rsidhu3, juliahmr, tozhang}@illinois.edu
cheng.niu@newsbreak.com
Abstract
Recent research highlights the challenges retrieval models face in retrieving useful
contexts and the limitations of generation models in effectively utilizing those con-
texts in retrieval-augmented generation (RAG) settings. To address these challenges,
we introduce RAG-RL , the first reasoning language model (RLM) specifically
trained for RAG. RAG-RL demonstrates that stronger answer generation models
can identify relevant contexts within larger sets of retrieved information – thereby
alleviating the burden on retrievers – while also being able to utilize those contexts
more effectively. Moreover, we show that curriculum design in the reinforcement
learning (RL) post-training process is a powerful approach to enhancing model
performance. We benchmark our method on two open-domain question-answering
datasets and achieve state-of-the-art results, surpassing previous SOTA generative
reader models. In addition, we offers empirical insights into various curriculum
learning strategies, providing a deeper understanding of their impact on model
performance.
1 Introduction
Retrieval-augmented generation (RAG; [ 7,11,24]) relies on retrieval and generation models that
work together to retrieve and integrate external contexts effectively for answering questions or
generating content. While previous works have made significant progress in improving these systems
by optimizing the retriever and designing re-ranking models [ 30,6,27], challenges persist when
it comes to retrieving relevant real-world contexts that require deep reasoning [ 21]. Furthermore,
prior studies have shown that generation models also face limitations to effectively synthesizing
information from multiple documents due to limitations in their reasoning capabilities [25].
In this work, we tackle the aforementioned challenges by training reasoning language models (RLMs)
capable of performing multi-hop reasoning across a greater number of retrieved documents. Prior
research has focused on optimizing the retrieval and re-ranking components of RAG with the goal of
presenting a small subset of documents to the answer generation model by maximizing metrics such
asrecall@5 . However, a model that can effectively differentiate between relevant and irrelevant
contexts given a longer list of retrieved passages would lessen the burden on retrieval models by
increasing recall and be more adept at using the information in the retrieved contexts [8].
Building upon on the recent success of reinforcement learning (RL) post-training techniques in
mathematics and coding [ 26,28], we apply Group Relative Policy Optimization (GRPO) [ 19] with
simple, rule-based rewards in the RAG scenario and show that RAG-RL achieves state-of-the-
art performance on both HotpotQA and MuSiQue compared to prevous SOTA generative reader
∗This work was done while Jerry was a Research Intern at NewsBreak.
Preprint.arXiv:2503.12759v1  [cs.CL]  17 Mar 2025

models. Furthermore, our comprehensive evaluation demonstrates that RAG-RL performs well both
in scenarios with numerous distractor passages and when restricted to only gold documents, mirroring
the settings of using a weaker retrieval model or a more advanced retriever and/or re-ranking system.
Here, “gold" documents refers to the set of documents from which the answer to a given question can
be deduced, while “distractor" documents are those that do not contain relevant information.
We also investigate how to achieve better model performance when using GRPO through training
models in different curriculum learning settings. Specifically, we study the effectiveness of introducing
question-answer training samples of different difficulty levels and the impact dataset ordering has on
the final model outcome. We observe that (1) adding easier samples during training greatly improves
the trained models’ performance, (2) curricula that scale problem difficulty linearly from easiest to
hardest perform worse when compared to min-max curricula that begin with the easiest samples and
jumps straight to the hardest samples (the difficulty level excepted at test time), and (3) the benefits
of ordering training samples from easiest to hardest are not conclusively supported.
To summarize, the main contributions of this work are as follows:
1.We introduce RAG-RL, the first RLM specialized for RAG and demonstrate SOTA per-
formance in both settings where either a large number of irrelevant contexts or when only
relevant contexts are retrieved when compared to previous SOTA generative reader models.
2.We introduce a curriculum-learning driven approach that improves the performance of the
RL post-training process that works well along with simple rule-based reward functions.
3.We provide several empirical insights on the effectiveness of different curriculum learn-
ing settings and demonstrate that the construction of an effective curriculum can lead to
significant performance improvements.
2 Related Works
2.1 RAG Systems
Rather than relying solely on parametric knowledge, RAG has been widely utilized in tasks that
require external information [ 7,11,24]. Previous works have made tremendous progress in designing
and training sophisticated retrieval and re-ranker models [ 6,27] to tackle the task of open-domain
question answering [ 3]. One important line of work has focused on improving the encoder models that
are used in the embedding generation process [9, 12], while another has focused on designing RAG
systems that focus on drawing connections between multiple different documents [ 5,6]. Rank1 [ 27]
has also demonstrated that allocating test-time compute for document re-ranking achieves impressive
performance on retrieving contexts that require reasoning (i.e. contexts that may not be semantically
similar, but whose relevance requires in-depth reasoning to identify). Past work has also sought to
take advantage of the long contexts of modern-day large language models (LLMs) by providing these
models with larger sets of retrieved documents, but have shown that LLMs still struggle to effectively
identify relevant contexts as the number of retrieved passages increases [8].
2.2 Reasoning Language Models
With the introduction of RLMs by OpenAI in their o1 models [ 16], the research community has made
progress in replicating similar models that have shown impressive performance in tasks that require
reasoning driven largely in part due to R1’s release [ 4]. Prior works have demonstrated the potential
for training smaller scale RLMs using reinforcement learning (RL) in the domains of mathematics,
logic, and coding [ 28,26] and have also achieved impressive performance. Other open source works
have sought to replicate the reasoning behavior of RLMs through cost-effective ways [ 13] while
others have investigated other factors that underlie RLMs, such as the ability to control the length of
chain-of-thought sequences in an attempt to control the amount of test time compute used [1].
2.3 Curriculum Learning
Curriculum learning [ 2] has been extensively studied as a training paradigm that orders the training set
by increasing difficulty to enhance stability and sample efficiency. In the context of question answering
(QA), curriculum learning has been leveraged to bridge distributional gaps between pre-training
2

and downstream fine-tuning datasets [ 31], mitigating domain shift and improving generalization.
Recent advances in LLMs have incorporated curriculum-inspired self-improvement mechanisms [ 10],
wherein models iteratively augment their training data with instances they can already solve, thereby
facilitating generalization to more complex reasoning tasks. Furthermore, curriculum learning
has been explored in RL settings [ 14], to progressively expose agents to increasingly difficult
tasks; however, its efficacy remains subject to task-specific constraints with some studies reporting
marginal benefits [ 28]. Contemporary QA research highlights the inherent difficulty of multi-hop
reasoning, particularly in benchmarks such as HotPotQA, where answering complex queries requires
integrating multiple disjoint pieces of evidence. This suggests that curriculum-based strategies could
be potentially beneficial in optimizing learning trajectories for reasoning-intensive QA models [ 15].
3 RAG-RL
In this section, we include a detailed overview of the RAG-RL training process. We begin by
presenting the rule-based rewards used in GRPO, and then introduce the curriculum construction
process.
3.1 Reward Modeling
System Prompt
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
Final answer: final answer
Supporting passages: title1, title2,...
</answer>
Figure 1: System prompt used for all experiments.
Given the modest improvements observed with supervised fine-tuning (SFT) in RAG settings [ 32,8],
we turn to RL as an alternative training method. We adopt GRPO [ 19] and define rule-based rewards
to encourage faithful answer generation, accurate citation usage, and adherence to the desired output
format.
Our rule-based rewards comprise of three components: answer rewards, citation rewards, and
formatting rewards. The system prompt in Figure 1 outlines the expected output format for all of our
models.
Answer Rewards . To incentivize correct final answers, we define the exact match reward as:
Ranswer =γanswer·1(oanswer =Ganswer), (1)
where oanswer is the generated final answer, Ganswer is the ground truth answer, and γanswer is a scaling
factor, which we set to 5 for our experiments.
Citation Rewards . To reward proper citations, we define:
Rcitations =γcorrect·Recall( ocitations ,Gcitations )−γincorrect ·cincorrect , (2)
where recall denotes the fraction of relevant citations cited in the final answer ( ocitations ) and Gcitations
is the list of ground truth citations, cincorrect is the number of incorrect citations, and both γcorrect and
γincorrect are the scaling factors for each component of the reward, which we set to 5 and 2 respectively
for our experiments.
Formatting Rewards . To enforce the desired output format, we assign a reward of γformat for correct
formatting—i.e., the presence of proper XML tags and required headings—while imposing a penalty
3

pfor outputs with excessive text or when raw Unicode exceeds twenty percent of the content2.
Formally,
Rformatting =γformat,if correct formatting is present,
−p, if incorrect formatting or excessive Unicode is detected.(3)
Total Reward and Objective Function . The overall reward for a sample is the sum of the individual
components:
Rtotal=Ranswer +Rcitation +Rformatting . (4)
We use the GRPO algorithm [ 19] for policy optimization. For each training sample q, we sample G
outputs {o1,o2,···,oG}from the old policy πθoldand then optimize the policy model by maximizing
the following objective:
JGRPO (θ) =E[q∼P(Q),{oi}G
i=1∼πθold(O|q)]
1
GGX
i=11
|oi||oi|X
t=1
minπθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t)ˆAi,t,clipπθ(oi,t|q,oi,<t)
πθold(oi,t|q,oi,<t), 1−ϵ, 1 +ϵ
ˆAi,t
−βDKL[πθ||πref]
,(5)
where ϵandβare hyper-parameters, and ˆAi,tis the group’s advantage. The clipping function
stabilizes policy updates by constraining the change in action probabilities, and the KL divergence
termDKL(πθ∥πref)penalizes deviations from the reference policy [18].
3.2 Curriculum Construction
Given a question, (Q), and a set of documents, (D), we partition Dinto a set of gold passages (D+)
and a set of distractor passages (D−). The number of “hops" required to correctly answer Qis given
byj=|D+|, while k=|D−|represents the maximum number of retrieved distractor documents.
Naturally, the difficulty of a multi-hop question for RAG can be measured along two axes: the number
of hops required (j)and the number of distractor documents provided to the generation model from
D−. For training RAG-RL, we define the difficulty of a training sample solely based on the subset
size of D−that we provide to the generation model. The easiest training samples contain at most one
distractor document along with all gold documents, while the hardest samples include the full set of
all retrieved passages. Formally, a training sample (Si)of difficulty level lis defined as
Sl
i= [Q,D+
1,D+
2, ...,D+
j,D−
1,D−
2, ...,D−
d],where d= min(max( l+ 2−j, 0),k) (6)
Since the minimum number of hops required among our datasets is 2, a difficulty level of 1corre-
sponds to 1 distractor document for a 2-hop question. The highest difficulty level we can effectively
introduce is thus k−2, which we denote as Kgoing forward. This definition ensures that all gold
contexts are retrieved regardless of difficulty level. Furthermore, at inference time, the list of all
documents presented to the generation model is shuffled to ensure a realistic retrieval setting.
While the datasets we use have 2-hop, 3-hop, and 4-hop questions, we focus on the number of
distractor passages as the only axis of difficulty due to the limited number of difficulty levels we can
introduce by varying the number of hops each question requires. An ablation that sorts the train set
by both the number of hops and the number of distractor passages is included in Section 6.3.
4 Experiments
4.1 Datasets
We evaluate RAG-RL on two multi-hop question-answering benchmarks, HotpotQA [29] and
MuSiQue (answerable) [ 22]. While HotpotQA has been shown to be a weaker test for multi-hop
reasoning due to the presence of spurious signals [ 23], we include it due to its widespread use but
primarily focus on MuSiQue in our analysis.
2The Unicode penalty has proven particularly beneficial for training stability.
4

Figure 2: Overview of the two curriculum construction settings used during training. Linear denotes
a curriculum that scales the number of distractor passages from 1 to n, while min-max denotes a
curriculum that is split evenly between the easiest and the hardest problems.
4.2 Training Setup
We use Qwen2.5-7B-Instruct [17] as our base model due to its strong performance in RL post-
training settings [ 28] and employ GRPO for the post-training process. Approximately 40,000
and 20,000 question-answer pairs were sourced from HotpotQA’s and MuSiQue’s training sets
respectively. All experiments train for a single epoch with a constant learning rate of 1.0e-6, a global
batch size of 294, KL coefficient of 0.01, and 7 rollouts for each of the 42 problems in each batch.
4.3 Baselines
We benchmark Qwen2.5-7B-Instruct as our natural baseline for the base model and then use
GRPO with samples of difficulty Kas our fine-tuned baseline (i.e. we train our model on the same
problem difficulty expected at test-time).
4.4 Curriculum Learning Settings
To investigate the effectiveness of curriculum construction in the post-training process, we benchmark
several different settings. As defined in Section 3.2, the minimum difficulty level is 1and the
maximum is K. Figure 2 provides an overview of the main curriculum construction settings used in
our experiments. We define a function Csetting :{1,. . .,n} → { 1,. . .,K}that maps an index iin
the training set to its corresponding difficulty level under each setting.
For the shuffled settings, we define σ:{1,. . .,n} → { 1,. . .,n}as a random permutation function,
ensuring a bijective reassignment of indices while preserving the difficulty distribution.
•Max : Each sample in the training set is presented at the difficulty level expected at test time.
Thus, the difficulty function is defined as:
Cmax(i) =K,∀i∈ {1,. . .,n}
•Linear : The training set is partitioned into Kequally sized subsets, with difficulty levels
increasing linearly from 1toK. The mapping function is:
Clinear (i) =K·i
n
•Linear Shuffled : The same difficulty levels are used as in the Linear setting, but the order
in which samples appear during training is randomized via σ:
Clinear-shuffled (i) =Clinear (σ(i))
5

•Min-Max : The training set is split into two equal parts, where the first half consists of the
easiest difficulty level ( 1) and the second half consists of the hardest difficulty level ( K).
The function is defined as:
Cmin-max (i) =1, ifi≤n/2
K,ifi > n/ 2
•Min-Max Shuffled : The same difficulty levels are used as in the Min-Max setting, but the
order is shuffled via σ:
Cmin-max-shuffled (i) =Cmin-max (σ(i))
Evaluation To benchmark the performance of our RLMs, we evaluate the F1 scores of the final
answer and passage-level citations on the full validation sets provided by HotpotQA and Musique.
We sample each response 3times and take the average score among all generations.
To compare our RLMs to previous works, we measure the performance of our model in two settings:
thedistractor setting andideal retriever setting . The distractor setting consists of providing the
generation model with the gold passages and up to 18distractor passages, which is comparable to
having the reasoning model handle both re-ranking and answer generation. On the other hand, in the
ideal retriever setting, the reasoning model is only given the gold truth passages, which is comparable
to answer generation with a strong retrieval and re-ranking system.
5 Results
Tables 1 and 2 present the QA performance of our baseline model and the RLMs we trained
across six different curriculum learning and curricula constructions settings. Notably, the min-max
curriculum achieves the highest F1 scores across both datasets. Additionally, we observe a significant
performance improvement both from the base model to any trained variant and from the base trained
model to any curriculum-enhanced variant.
HotpotQA MuSiQue
Model / Curriculum Answer F1 Citation F1 Joint F1 Answer F1 Citation F1 Joint F1
Qwen2.5-7B-Instruct / – 60.65 36.47 45.55 25.88 25.35 25.61
Qwen2.5-7B-Instruct / Max 68.52 71.55 70.00 46.06 64.66 53.80
Qwen2.5-7B-Instruct / Linear 72.65 80.53 76.39 47.93 68.45 56.38
Qwen2.5-7B-Instruct / Linear Shuffled 70.12 79.75 74.63 51.95 69.63 59.51
Qwen2.5-7B-Instruct / Min-Max 74.97 81.25 77.98 55.13 69.27 61.40
Qwen2.5-7B-Instruct / Min-Max Shuffled 72.12 80.40 76.09 52.44 69.91 59.93
Table 1: Performance of models in the distractor setting.
Distractor Setting We report the results of our fine-tuned models in the distractor setting in where
between 6to18distractor contexts are retrieved and presented to the answer generation model in
Table 1. The min-max curricula achieve the highest F1 scores in all three categories (answer, citation,
and joint). When compared to previous RAG systems, we demonstrate competitive performance.
Table 3 presents the performance of each model on MuSiQue, segmented by the number of hops
required per question. As the number of hops increases, F1 scores consistently decline across all
models.
HotpotQA MuSiQue
Model / Curriculum Answer F1 Citation F1 Joint F1 Answer F1 Citation F1 Joint F1
Qwen2.5-7B-Instruct / – 67.90 63.26 65.50 41.16 58.16 48.21
Qwen2.5-7B-Instruct / Max 74.79 77.38 76.06 59.04 77.99 67.21
Qwen2.5-7B-Instruct / Linear 77.94 86.45 81.97 64.84 85.23 73.65
Qwen2.5-7B-Instruct / Linear Shuffled 76.98 87.08 81.72 66.95 86.21 75.37
Qwen2.5-7B-Instruct / Min-Max 79.74 87.38 83.38 69.79 86.81 77.37
Qwen2.5-7B-Instruct / Min-Max Shuffled 78.13 87.26 82.45 68.69 89.89 77.87
Table 2: Performance of models in the ideal retriever setting.
6

Ideal Retriever Setting In Table 2, we report the performance of our models in the ideal retriever
setting where our models are only given the relevant gold contexts for each question. Similar to in
the previous section, we see that the min-max curriculum also achieves the highest F1 scores across
the board. We demonstrate here that previous SOTA retrieval and re-ranking systems can also see
improved performance by using RAG-RL as their answer generation model.
6 Discussion
Musique 2-hop Musique 3-hop Musique 4-hop
Model / Curriculum Answer F1 Citation F1 Joint F1 Answer F1 Citation F1 Joint F1 Answer F1 Citation F1 Joint F1
Qwen2.5-7B-Instruct / – 28.94 29.46 29.19 23.09 22.62 22.85 21.65 17.80 19.53
Qwen2.5-7B-Instruct / Max 48.95 70.47 57.77 44.94 63.84 52.75 39.22 48.23 43.26
Qwen2.5-7B-Instruct / Linear 52.04 74.38 61.23 45.53 67.08 54.24 39.74 52.74 45.33
Qwen2.5-7B-Instruct / Linear Shuffled 55.05 74.48 63.31 50.10 70.05 58.42 45.86 53.84 49.53
Qwen2.5-7B-Instruct / Min-Max 57.03 76.27 65.26 54.11 67.51 60.08 51.16 50.94 51.05
Qwen2.5-7B-Instruct / Min-Max Shuffled 55.19 74.22 63.31 50.80 70.92 59.20 46.98 54.73 50.56
Table 3: Performance of models on MuSiQue in the distractor setting grouped by the number of hops
in each question.
6.1 Importance of a Including Additional Easier Samples
The results in Tables 1 and 2 demonstrate that including samples of easier difficulty in the training
process can help the model achieve higher performance during evaluation. Given a fixed number of
training steps, the metrics in Table 3 shows that the baseline curriculum has consistently has lower F1
scores across all questions regardless of the number of hops required to answer the question correctly.
6.2 Do we need Granular Problem Difficulty?
MuSiQue
Model Metric Answer F1 Citation F1 Joint F1
Qwen2.5-7B-Instruct Pass@1 25.42 25.11 25.26
Qwen2.5-7B-Instruct Pass@32 43.38 45.46 44.40
Table 4: Baseline model performance on MuSiQue.
Previous work in the area of self-improvement has shown that LLMs exhibit limited generalizability
and that weak-to-strong curricula are effective for helping models generalize beyond their initial
training distributions [ 10]. However, our results suggest that this is not always necessary as the linear
curriculum is outperformed by the min-max curriculum in both shuffled and non-shuffled settings.
We believe that this is a byproduct of our base model having relatively strong performance on the
task before any post-training is applied. We show our base model’s Pass@32, which is calculated by
taking the maximum F1 score across 32 generations per question, in 4. We find that these scores are
comparable to our baseline GRPO, demonstrating relatively strong baseline performance on the task.
6.3 Does Dataset Ordering Matter?
MuSiQue
Model / Curriculum Answer F1 Citation F1 Joint F1 Setting
Qwen2.5-7B-Instruct / Linear 47.93 68.45 56.38 Distractor Setting
Qwen2.5-7B-Instruct / Linear Shuffled 51.95 69.63 59.51 Distractor Setting
Qwen2.5-7B-Instruct / Linear Sorted by Number of Hops 51.27 69.33 58.95 Distractor Setting
Qwen2.5-7B-Instruct / Linear 64.84 85.23 73.65 Ideal Retriever
Qwen2.5-7B-Instruct / Linear Shuffled 66.95 86.21 75.37 Ideal Retriever
Qwen2.5-7B-Instruct / Linear Sorted by Number of Hops 66.59 83.99 74.29 Ideal Retriever
Table 5: Model performance on MuSiQue given the same training set but different dataset orderings.
7

0 100 200 300 400 500
Steps0.150.200.250.300.350.40Exact MatchEval Answer EM
0 100 200 300 400 500
Steps0.00.51.01.52.02.5Citation RewardEval Citation Reward
0 100 200 300 400 500
Steps180190200210220230Output TokensCompletion Length
0 100 200 300 400 500
Steps468Reward ScoreReward StdLinear Linear Shufﬂed Linear Sorted by Number of HopsFigure 3: Evaluation and training plots for 3 different dataset orderings.
Curriculum learning strategies where the training sets are ordered from easy to hard have been
successfully employed in many areas of machine learning [ 20]. Table 5 and Figure 3 include an
ablation in which we sort the training set from easiest to hardest by sorting both by the number of
hops each question requires and by the number of distractor passages, resulting in batches at the
end of training containing the maximum number of passages with the greatest number of hops. We
compare this against the linear and linear shuffled curricula which are constructed from the same
training set but feature different dataset orderings. Although we see that the evaluation scores for both
the final answer and citations start lower for the linear shuffled setting, all three curves converge to
around the same point towards the end of training. Across all settings, we observe that the F1 scores
are highest when the dataset is shuffled randomly, leading to our conclusion that the results do not
conclusively support that using a specific ordering during training leads to significant performance
gains or declines.
7 Conclusion
In this work, we introduce RAG-RL, the first RLM explicitly trained for the RAG answer generation
task. We hope this research paves the way for further exploration of jointly optimizing multiple
components of the RAG process and refining RL-based post-training techniques.
8

References
[1]Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks
with reinforcement learning, 2025. URL: https://arxiv.org/abs/2503.04697 ,arXiv:
2503.04697 .
[2]Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum
learning. In International Conference on Machine Learning , 2009. URL: https://api.
semanticscholar.org/CorpusID:873046 .
[3]Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading Wikipedia to answer
open-domain questions. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) ,
pages 1870–1879, Vancouver, Canada, July 2017. Association for Computational Linguistics.
URL: https://aclanthology.org/P17-1171/ ,doi:10.18653/v1/P17-1171 .
[4]DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, et al.
Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL:
https://arxiv.org/abs/2501.12948 ,arXiv:2501.12948 .
[5]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and
fast retrieval-augmented generation, 2024. URL: https://arxiv.org/abs/2410.05779 ,
arXiv:2410.05779 .
[6]Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to
memory: Non-parametric continual learning for large language models, 2025. URL: https:
//arxiv.org/abs/2502.14802 ,arXiv:2502.14802 .
[7]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: retrieval-
augmented language model pre-training. In Proceedings of the 37th International Conference
on Machine Learning , ICML’20. JMLR.org, 2020.
[8]Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. Long-context LLMs meet RAG: Over-
coming challenges for long inputs in RAG. In The Thirteenth International Conference on Learn-
ing Representations , 2025. URL: https://openreview.net/forum?id=oU3tpaR8fm .
[9]Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catan-
zaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding
models, 2025. URL: https://arxiv.org/abs/2405.17428 ,arXiv:2405.17428 .
[10] Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, and Dimitris Papailiopoulos.
Self-improving transformers overcome easy-to-hard and length generalization challenges, 2025.
URL: https://arxiv.org/abs/2502.01612 ,arXiv:2502.01612 .
[11] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and
Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in
Neural Information Processing Systems , 33:9459–9474, 2020.
[12] Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet
Singh, and Douwe Kiela. Generative representational instruction tuning, 2025. URL: https:
//arxiv.org/abs/2402.09906 ,arXiv:2402.09906 .
[13] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi,
Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-
time scaling, 2025. URL: https://arxiv.org/abs/2501.19393 ,arXiv:2501.19393 .
[14] Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E. Taylor, and Peter
Stone. Curriculum learning for reinforcement learning domains: A framework and survey, 2020.
URL: https://arxiv.org/abs/2003.04960 ,arXiv:2003.04960 .
[15] Kosuke Nishida, Kyosuke Nishida, Masaaki Nagata, Atsushi Otsuka, Itsumi Saito, Hisako
Asano, and Junji Tomita. Answering while summarizing: Multi-task learning for multi-hop
QA with evidence extraction. In Anna Korhonen, David Traum, and Lluís Màrquez, editors,
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics ,
pages 2335–2345, Florence, Italy, July 2019. Association for Computational Linguistics. URL:
https://aclanthology.org/P19-1225/ ,doi:10.18653/v1/P19-1225 .
9

[16] OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky,
et al. Openai o1 system card, 2024. URL: https://arxiv.org/abs/2412.16720 ,arXiv:
2412.16720 .
[17] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, et al. Qwen2.5
technical report, 2025. URL: https://arxiv.org/abs/2412.15115 ,arXiv:2412.15115 .
[18] John Schulman. Approximating kl divergence, 2020. URL: http://joschu.net/blog/
kl-approx.html .
[19] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, Y . K. Li, Y . Wu, and Daya Guo. Deepseekmath: Pushing the limits of
mathematical reasoning in open language models, 2024. URL: https://arxiv.org/abs/
2402.03300 ,arXiv:2402.03300 .
[20] Petru Soviany, Radu Tudor Ionescu, Paolo Rota, and Nicu Sebe. Curriculum learning: A survey,
2022. URL: https://arxiv.org/abs/2101.10382 ,arXiv:2101.10382 .
[21] Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han yu Wang, Haisu
Liu, Quan Shi, Zachary S. Siegel, Michael Tang, Ruoxi Sun, Jinsung Yoon, Sercan O. Arik,
Danqi Chen, and Tao Yu. Bright: A realistic and challenging benchmark for reasoning-intensive
retrieval, 2024. URL: https://arxiv.org/abs/2407.12883 ,arXiv:2407.12883 .
[22] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal. Musique: Multihop questions
via single-hop question composition. Transactions of the Association for Computational
Linguistics , 10:539–554, 2022. URL: https://aclanthology.org/2022.tacl-1.31/ ,
doi:10.1162/TACL_A_00475 .
[23] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal. Interleaving retrieval with
chain-of-thought reasoning for knowledge-intensive multi-step questions. In A. Rogers, J. Boyd-
Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 10014–10037, Toronto, Canada,
July 2023. Association for Computational Linguistics. URL: https://aclanthology.org/
2023.acl-long.557 ,doi:10.18653/v1/2023.acl-long.557 .
[24] Yuhao Wang, Ruiyang Ren, Junyi Li, Xin Zhao, Jing Liu, and Ji-Rong Wen. REAR: A
relevance-aware retrieval-augmented framework for open-domain question answering. In Yaser
Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing , pages 5613–5626, Miami, Florida, USA,
November 2024. Association for Computational Linguistics. URL: https://aclanthology.
org/2024.emnlp-main.321/ ,doi:10.18653/v1/2024.emnlp-main.321 .
[25] Zora Zhiruo Wang, Akari Asai, Xinyan Velocity Yu, Frank F. Xu, Yiqing Xie, Graham Neubig,
and Daniel Fried. Coderag-bench: Can retrieval augment code generation?, 2024. URL:
https://arxiv.org/abs/2406.14497 ,arXiv:2406.14497 .
[26] Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel
Fried, Gabriel Synnaeve, Rishabh Singh, and Sida I. Wang. Swe-rl: Advancing llm reasoning
via reinforcement learning on open software evolution, 2025. URL: https://arxiv.org/
abs/2502.18449 ,arXiv:2502.18449 .
[27] Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, and Benjamin Van
Durme. Rank1: Test-time compute for reranking in information retrieval, 2025. URL: https:
//arxiv.org/abs/2502.18418 ,arXiv:2502.18418 .
[28] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou,
Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based
reinforcement learning, 2025. URL: https://arxiv.org/abs/2502.14768 ,arXiv:2502.
14768 .
[29] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop ques-
tion answering, 2018. URL: https://arxiv.org/abs/1809.09600 ,arXiv:1809.09600 .
[30] Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong Liu, and Shen Huang. End-to-end beam
retrieval for multi-hop question answering, 2024. URL: https://arxiv.org/abs/2308.
08973 ,arXiv:2308.08973 .
10

[31] L. Zhang, Quan Wang, Benfeng Xu, Yi Liu, and Zhendong Mao. Curriculum learning driven
domain adaptation for low-resource machine reading comprehension. IEEE Signal Process-
ing Letters , 31:2650–2654, 2024. URL: https://api.semanticscholar.org/CorpusID:
271260355 .
[32] Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and
Joseph E. Gonzalez. Raft: Adapting language model to domain specific rag, 2024. URL:
https://arxiv.org/abs/2403.10131 ,arXiv:2403.10131 .
11