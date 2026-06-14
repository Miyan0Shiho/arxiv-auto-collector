# Learning to Reason by Analogy via Retrieval-Augmented Reinforcement Fine-Tuning

**Authors**: Zilin Xiao, Qi Ma, Chun-cheng Jason Chen, Xintao Chen, Avinash Atreya, Hanjie Chen, Vicente Ordonez

**Published**: 2026-06-11 17:59:52

**PDF URL**: [https://arxiv.org/pdf/2606.13680v1](https://arxiv.org/pdf/2606.13680v1)

## Abstract
Retrieval-augmented generation (RAG) has become a standard mechanism for grounding language models in external knowledge, yet conventional retrieval based on lexical or semantic similarity is poorly suited for complex reasoning tasks: a semantically similar problem may demand an entirely different solution strategy, while a superficially different problem may share the same underlying reasoning pattern. We propose Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT), a post-training framework that teaches language models to reason by analogy. RA-RFT uses gold-relevance distillation to train a retriever that ranks contexts by expected reasoning benefit rather than semantic overlap, and then fine-tunes the policy model via reinforcement fine-tuning methods with retrieved analogous demonstrations, so the model learns to leverage reasoning traces under verifiable outcome rewards. We further analyze the diversity of retrieved contexts and find that reasoning-aware retrieval surfaces complementary solution strategies that provide distinct reasoning scaffolds for individual problems. Across challenging mathematical reasoning benchmarks, RA-RFT consistently outperforms standard reinforcement fine-tuning methods. For example, it improves AIME 2025 average@32 accuracy by 7.1 and 2.8 points over GRPO for Qwen3-1.7B and Qwen3-4B respectively -- suggesting that reasoning-aware retrieval is a complementary axis of improvement and orthogonal to advances in reward design or training curricula.

## Full Text


<!-- PDF content starts -->

Learning to Reason by Analogy via
Retrieval-Augmented Reinforcement Fine-Tuning
Zilin Xiao1,2,∗,Qi Ma1,Jason Chen1,Xintao Chen1,Avinash Atreya1,Hanjie Chen2,Vicente Ordonez2
1Meta Superintelligence Labs,2Rice University
∗Work partially done during internship at Meta
Retrieval-augmented generation (RAG) has become a standard mechanism for grounding language
models in external knowledge, yet conventional retrieval based on lexical or semantic similarity is
poorly suited for complex reasoning tasks: a semantically similar problem may demand an entirely
different solution strategy, while a superficially different problem may share the same underlying
reasoning pattern. We proposeRetrieval-Augmented Reinforcement Fine-Tuning (RA-RFT), a post-
training framework that teaches language models to reason by analogy. RA-RFT uses gold-relevance
distillation to train a retriever that ranks contexts by expected reasoning benefit rather than semantic
overlap, and then fine-tunes the policy model via reinforcement fine-tuning methods with retrieved
analogous demonstrations, so the model learns to leverage reasoning traces under verifiable outcome
rewards. We further analyze the diversity of retrieved contexts and find that reasoning-aware retrieval
surfaces complementary solution strategies that provide distinct reasoning scaffolds for individual
problems. Across challenging mathematical reasoning benchmarks, RA-RFT consistently outperforms
standard reinforcement fine-tuning methods. For example, it improves AIME 2025 average@32
accuracy by 7.1 and 2.8 points over GRPO for Qwen3-1.7B and Qwen3-4B respectively—suggesting
that reasoning-aware retrieval is a complementary axis of improvement and orthogonal to advances in
reward design or training curricula.
Date:June 12, 2026
Correspondence:Zilin Xiao atzilin@meta.com
1 Introduction
Reinforcement learning from verifiable rewards (RLVR) has emerged as a powerful post-training paradigm
for complex reasoning (DeepSeek-AI, 2025; OpenAI, 2024). By optimizing for outcome correctness rather
than imitating reference solutions, RLVR elicits sophisticated chain-of-thought reasoning that generalizes
across diverse problem types (Shao et al., 2024). However, these approaches rely entirely on the parametric
knowledge of a model: when confronted with novel problems whose solutions require reasoning patterns not
well-represented in pre-training, the model has no mechanism to draw on external problem-solving expertise.
Retrieval-augmented generation (RAG) offers a natural complement, grounding model outputs in externally
retrieved content (Lewis et al., 2020; Guu et al., 2020). Yet standard retrieval is poorly suited for reasoning
tasks: conventional retrievers rank candidates by lexical or semantic similarity, which correlates weakly with
reasoning utility, understood as whether the retrieved content actually helps solve the target problem (Su et al.,
2025). A semantically similar problem may demand an entirely different solution strategy, while a superficially
different problem may share the same underlying reasoning pattern. As a result, naively augmenting reasoning
models with retrieved content often yields marginal or inconsistent gains (Arabzadeh et al., 2025).
We approach this problem through the lens ofanalogical reasoning(Gentner, 1983): expert problem-solvers
recall previously solved problems not because surface details match, but because the underlying reasoning
structure transfers. This motivates a retrieval paradigm that selects examples whosereasoning traces,i.e.
step-by-step solution strategies are maximally informative for the target problem. Building on this insight, we
proposeRetrieval-Augmented Reinforcement Fine-Tuning (RA-RFT), a post-training framework consisting of
three stages: (1)gold-relevance distillation, which constructs retrieval supervision grounded in reasoning utility
by using a judge model to directly assess whether candidate reasoning traces share transferable reasoning
1arXiv:2606.13680v1  [cs.CL]  11 Jun 2026

Evaluate the expression (x+1x)3 given that x=5. If tanθ+cotθ=4,evaluate tan3θ+cot3θ. Surface SimilarityUnhelpful Retrieved Exemplar
Reasoning Relevance
Reasoning Analogous Exemplar
Step 1: Identify the Binomial Identity …Step 1: Use Direct Substitution …
 Given that x+1x=5, ﬁnd the exact value of x3+1x3.Training Query(Q)(E1)(E2)E1+Q
ywyl
E2+Q
ywylReinforcement Fine-tuning with Retrieved Context Figure 1Motivation of RA-RFT.Left:A training query may be retrieved with a surface-similar but reasoning-irrelevant
exemplar (E 1, which could be easily solved by direct substitution) or a superficially different but reasoning-analogous
exemplar (E 2, which shares the same binomial identity strategy).Right:The quality of retrieved context is critical
for RLVR training. Conditioning on the unhelpful exemplarE 1misleads the model, degrades rollout quality, and
produces noisy reward signals that hurt policy learning. In contrast, conditioning on the reasoning-analogous exemplar
E2provides transferable solution strategies that improve rollout quality and yield informative rewards, leading to
effective policy improvement. See case studies in Section C.
patterns with the target problem; (2)reasoning-aware retriever training, which uses contrastive learning
on these utility-based annotations to train a dense retriever that surfaces structurally analogous problems,
and (3)reinforcement fine-tuning with retrieved demonstrations, which injects retrieved reasoning traces into
training prompts and optimizes the target model via RLVR. We further analyze the diversity of retrieved
contexts and the quality of the reasoning-aware retriever and find that reasoning-aware retrieval produces
structurally varied traces that surface distinct solution strategies for individual problems, suggesting that
retrieval quality is a key factor for effective context augmentation.
We evaluate RA-RFT on competition-level mathematical reasoning benchmarks, including AIME 2024 and
2025 (MAA, 2024), HMMT February 2025 (HMMT, 2025), and BrUMO 2025 (BrUMO, 2025). Across
Qwen3-1.7B and Qwen3-4B, RA-RFT consistently outperforms both standalone RLVR and strong baselines: it
improves AIME 2025 average@32 accuracy by 7.1 and 2.8 points over GRPO, and achieves 4.1 and 2.6 points
of overall average gain across all four benchmarks. These results confirm that grounding retrieval in reasoning
utility, rather than surface-level similarity, is critical for unlocking the full potential of retrieval-augmented
reinforcement fine-tuning.
2 Related Work
Reinforcement Learning for Reasoning.Reinforcement learning from verifiable rewards (RLVR) has emerged
as a leading paradigm for eliciting reasoning in large language models (DeepSeek-AI, 2025; OpenAI, 2024).
Group Relative Policy Optimization (GRPO) (Shao et al., 2024) replaces the value-network critic with
group-normalized advantages, and a family of follow-up variants (Liu et al., 2025; Yu et al., 2025; MiniMax,
2025; Qi et al., 2026) addresses optimization bias, large-scale stability, importance sampling, and ratio clipping.
A complementary line shapes the training signal through context engineering rather than the optimizer:
injecting partial solutions into hard problems (Li et al., 2026), stepwise hints from teacher models (Zhang
et al., 2025a), or expert demonstrations in lieu of verifiers (Cai and Provilkov, 2025). RA-RFT is orthogonal
to both directions: rather than modifying the reward, the optimizer, or the training curriculum, it augments
RLVR rollouts with externally retrieved reasoning traces, providing a knowledge source that the policy must
learn to useunder outcome reward.
Retrieval-Augmented Generation for Reasoning.While retrieval-augmented generation is effective for knowledge-
intensive tasks (Lewis et al., 2020; Guu et al., 2020; Ram et al., 2023), its application to reasoning is more
delicate: retrieval helps only under bounded conditions and noisy retrievals actively hurt (Liu et al., 2024),
and standard retrievers underperform on reasoning-intensive queries where surface similarity diverges from
2

Gold Relevance DistillationTraining SetCorpusQuestions
Questions + TracesJudge Model
Relevance Labels{qi}Ni=1
{(pj,tj)}Mj=1(q1,p1,j1)→(q2,p1,j1)→(qN,p1,j1)→(qN,pM,jM)→⋯⋯Reasoning-Aware  Retriever TrainingContrastive Training
QueryPositivePullReinforcement Fine-tuningRetrieverTraining Query
Relevant Reasoning TraceConcatG  Sampled ResponseReward FunctionGround-Truth AnswerCompute AdvantagesPolicy Update
Judge PromptFigure 2Overview of the RA-RFT framework. A judge model evaluates query-corpus pairs to produce binary
reasoning-relevance labels. Then we train a dense retriever via contrastive learning using the distilled relevance labels.
The trained retriever provides reasoning-analogous traces which augment the reinforcement fine-tuning process.
reasoning utility (Su et al., 2025). Recent work tackles this with corpus restructuring into step-by-step
traces (Arabzadeh et al., 2025), rubric-based retriever fine-tuning distilled from an LLM judge (Lan et al.,
2025), or interleaved retrieval-and-reasoning trained by RL (Li et al., 2025). All of these target inference-time
RAG and either leave the policy frozen or train the policy to issue retrieval queries. RA-RFT instead introduces
reasoning relevanceas the retrieval objective, distilled from a GPT-4o judge over offline query-context pairs,
and integrates the resulting retriever directly into reinforcement fine-tuning so the policy is shaped by retrieved
analogies during exploration rather than only at test time.
Learning from Demonstrations and Analogies.Our work draws on analogical reasoning (Gentner, 1983; Holyoak
and Thagard, 1996) and the in-context learning paradigm (Brown et al., 2020; Liu et al., 2022; Rubin
et al., 2022), where models adapt to new tasks by conditioning on demonstrations rather than parameter
updates. Several recent works retrieve or construct demonstrations tailored to specific reasoning tasks at
inference time, including computational-graph-matched exemplars for math word problems (Yang et al.,
2024), trajectory-as-exemplar prompting (Zheng et al., 2024), adaptive test-time memory (Suzgun et al.,
2026), large-scale subquestion–subroutine procedural memories retrieved mid-trajectory (Wu et al., 2026),
and compact procedural “behaviors” mined from a model’s own past traces and reinjected in-context or via
SFT (Didolkar et al., 2025). A complementary line internalizes such in-context signal into parameters via
on-policy distillation against a privileged-context teacher (Ye et al., 2026; Zhao et al., 2026), but cannot supply
reasoning patterns the teacher itself does not already know. In the broader RL setting, Goyal et al. (2022)
augment embodied agents with neural retrieval over past trajectories, conditioning the policy on historical
experience beyond the current state. RA-RFT differs from all of these by closing the loop for verifiable-reward
reasoning: retrieved analogous demonstrations are folded into the RL fine-tuning loop, so the policy learns
under outcome rewards to exploit retrieved analogies rather than to imitate them at test time, and the
retriever itself is trained against reasoning-utility supervision rather than surface similarity.
3 Methodology
Our framework consists of three stages: (1) gold-relevance distillation to construct reasoning-utility-based
retrieval supervision, (2) reasoning-aware retriever training via contrastive learning, and (3) reinforcement
fine-tuning with retrieved demonstrations. Figure 2 illustrates the overall pipeline.
3.1 Overall Setup
We consider a training set of reasoning problems D={(qi, ai)}N
i=1with verifiable answers. Given a language
model Mϕ, the goal is to train Mϕto solve problems in D(and generalize beyond) by optimizing for
outcome correctness via reinforcement learning. Standard RLVR approaches sample a group of responses
{ˆa1, . . . ,ˆa G} ∼ M ϕ(· |q)for each problem q, compute rewards r(ˆag, a)based on answer correctness, and
3

update the policy using normalized advantages. However, RLVR is bottlenecked by the model’s parametric
knowledge. When a problem demands a reasoning strategy that the model has not internalized during
pre-training, for instance, a number-theoretic argument that hinges on a combinatorial identity, no amount of
sampling will reliably discover it, leading to sparse rewards and stalled learning. Existing works either inject
dense supervision from the reference solution to thesameproblem (Zhao et al., 2026), which requires the
model to already be capable enough to rationalize the trace, or reshape the training curriculum (Li et al.,
2026; Zhang et al., 2025a), which still cannot supply reasoning patterns absent from the model’s parameters.
What is missing is a mechanism akin to human analogical reasoning: the ability to recall a structurally related
solved problem whose solution strategytransfers, even when the two problems differ on the surface.
To address these limitations, we augment the reasoning process with an external corpus C={(pj, tj)}M
j=1
of problems paired with reasoning traces, where each trace tjis a step-by-step solution generated by a
teacher model. The central question is:which corpus entries, when provided as context, will most improve the
model’s ability to reason about a given problem?Standard retrieval ranks candidates by embedding similarity
c∗=arg max c∈C⟨eq,ec⟩, but this does not account for whether c∗actually helps the modelreasonabout q.
We instead definereasoning relevanceas the degree to which conditioning on a candidate trace improves the
model’s probability of producing a correct answer:
c∗
reason = arg max
c∈CPM(a=a∗|q, c),(1)
where a∗is the gold answer. Since computing this exactly is intractable, we approximate it viagold-relevance
distillation(Section 3.2): querying a strong judge model to directly assess reasoning relevance for each
query-candidate pair. These annotations are then used to train areasoning-aware retrievervia contrastive
learning (Section 3.3), which in turn supplies retrieved demonstrations forreinforcement fine-tuningof the
target model (Section 3.4). Figure 2 summarizes the complete pipeline.
3.2 Gold-Relevance Distillation
To train a reasoning-aware retriever, we need supervision that reflects reasoning utility rather than surface
similarity. We proposegold-relevance distillation, which leverages a judge model Mjudge(e.g. GPT-4o) to
directly assess the reasoning relevance of candidate traces for a given query.
Given the training setDand corpusC, we construct relevance labels via pairwise evaluation:
1.For each training problem qi, enumerate all candidate traces c∈ C, forming the complete set of
query-context pairs.
2.For each pair( qi, c), prompt the judge model Mjudgeto assess whether the reasoning trace in cexhibits
transferable reasoning patterns that are relevant to solvingq i.
3.Assign binary relevance labels yi,c∈ {0,1}, where yi,c= 1if the judge determines that the reasoning
patterns incare structurally relevant toq i, even if the two problems differ in surface-level content.
The judge is prompted to evaluate reasoning-structural similarity rather than surface similarity: two problems
are deemed reasoning-relevant if they share analogous solution strategies, mathematical structures, or proof
techniques, regardless of whether they involve similar topics or notation. This pairwise evaluation avoids
the circularity of relying on an initial retriever that may miss reasoning-relevant traces due to low surface
similarity, ensuring that gold labels capture true reasoning utility across the entire corpus.
3.3 Reasoning-Aware Retriever Training
Using the gold-relevance annotations, we train a dense retriever Rθvia contrastive learning. For each query
qi, letC+
i={c∈ C :yi,c= 1}denote the set of reasoning-relevant traces. We optimize the InfoNCE objective:
Lretrieval =−X
iX
c+∈C+
ilogexp(⟨e qi,ec+⟩/τ)P
c∈C iexp(⟨e qi,ec⟩/τ),(2)
where τis a temperature hyperparameter. This objective encourages Rθto rank traces that genuinely aid
reasoning higher than semantically similar but reasoning-irrelevant alternatives.
4

3.4 Reinforcement Fine-Tuning with Retrieved Demonstrations
A key challenge in reinforcement fine-tuning is that difficult problems produce sparse reward signals: if the
model rarely generates correct answers, it receives little useful gradient information for improvement. Prior
work addresses this through curriculum design (Li et al., 2026) or stepwise hints (Zhang et al., 2025a), but
these approaches require explicit problem decomposition or teacher annotation at the step level. Retrieved
reasoning traces offer a natural alternative: by conditioning on analogous solved examples, the model’s
effective success rate on challenging problems increases, yielding denser reward signals without modifying the
reward function or training curriculum.
With the reasoning-aware retriever Rθfixed, we fine-tune the target model Mϕvia reinforcement fine-tuning
with retrieved demonstrations. Our approach is compatible with any policy optimization algorithm. We
describe the general procedure below and instantiate it with GRPO (Shao et al., 2024) in our experiments.
For each training problem(q, a):
1. Retrieve top-kreasoning traces{c 1, . . . , c k}=R θ(q,C)from the corpus.
2.Sample a group of Gresponses {ˆa1, . . . ,ˆa G} ∼ M ϕ(· |q, c 1, . . . , c k)conditioned on the retrieved traces.
3. Compute rewardsr(ˆa g, a)based on answer correctness.
4. UpdateM ϕusing the policy optimization objective with normalized advantages.
When instantiated with GRPO, the objective computes advantages relative to the group mean reward:
LGRPO =E (q,a)∼D"
−1
GGX
g=1Ag·logM ϕ(ˆag|q,{c j})#
,(3)
where Ag= (r(ˆag, a)−¯r)/σris the normalized advantage. The critical insight is that retrieved traces serve as
reasoning scaffolding: rather than relying solely on parametric knowledge, the model learns to extract and
transfer solution strategies from analogous demonstrations, effectively increasing the density of reward signals
during training. We condition the policy update on {cj}rather than on qalone for two reasons. First, the
rollouts ˆagare themselves sampled from Mϕ(· |q,{c j}), so the importance-sampling identity requires the
update distribution to match the sampling distribution, otherwise the advantage estimate is biased. Second,
marginalizing over {cj}would force the model to either ignore retrieved context (collapsing to standard
GRPO) or learn an implicit retrieval policy, neither of which captures the intended training signal that the
model should learn touseretrieved analogies when they are supplied.
4 Experiments
We first compare RA-RFT against standard RLVR baselines and state-of-the-art methods (Section 4.2), then
conduct ablation studies that disentangle the contributions of retrieval augmentation, the training objective,
the choice of policy optimizer, and retriever quality (Section 4.3).
4.1 Experimental Setup
Models.We experiment with the Qwen3 (Qwen Team, 2025) model family at two scales: Qwen3-1.7B,
Qwen3-4B as starting checkpoints. For gold-relevance distillation (Section 3.2), we use Qwen3-235B-A22B to
generate and summarize reasoning traces for the retrieval corpus, and GPT-4o as the judge model Mjudge
to produce binary reasoning-relevance labels. To make the pairwise evaluation tractable, we restrict judge
comparisons to pairs that share the same coarse problem-type label, reducing the number of calls by roughly
an order of magnitude (see Section A for details). For the reasoning-aware retriever, we initialize Rθfrom
Reason-ModernColBERT (Chaffin, 2025), a late-interaction multi-vector retrieval model, and further fine-tune
it with the contrastive objective described in Section 3.3.
Training data.We mainly adopt the training dataset of QuestA (Li et al., 2026) for a fair comparison.
Specifically, a total of 12.5k problems selected in QuestA are used as training queries in all training. We
construct the retrieval corpus Cusing the queries in the OpenR1-Math-220K (Hugging Face, 2025) dataset,
5

Method AIME24 AIME25 HMMT25 BrUMO25 Avg. Avg. (all)
Qwen3-4B
Base (Instruct) 70.5 64.3 41.3 65.5 58.7 60.4
+ GRPO 74.8 66.4 46.4 69.8 62.5 64.4
+ OPSD†76.0 66.9 45.2 – 62.7 –
+ RA-RFT (Ours) 75.8 69.2 47.3 75.7 64.1 67.0
Qwen3-1.7B
Base (Instruct) 48.1 35.9 23.4 50.9 35.8 39.6
+ GRPO 50.4 41.6 26.3 54.8 39.4 43.3
+ OPSD†51.4 38.3 25.0 – 38.2 –
+ QuestA 52.0 42.7 26.0 52.6 40.2 43.3
+ RA-RFT (Ours) 55.1 48.7 28.2 57.4 44.0 47.4
Table 1Performance comparison across mathematical reasoning benchmarks for Qwen3 models. We report average@32
accuracy in percentage except for OPSD†whose results are taken from the original paper using average@16. “Avg.”
averages over AIME24, AIME25, and HMMT25; “Avg. (all)” includes all benchmarks when applicable.Boldindicates
the best result per model size. Highlighted rows mark our method.
but exclude any problems that overlap with the training set to prevent learning shortcuts. Both the training
queries and the corpus traces are examined to ensure that they do not contain any problems from the
evaluation benchmarks.
Evaluation benchmarks.We evaluate on four competition-level mathematical reasoning benchmarks: AIME
2024 (MAA, 2024), AIME 2025 (MAA, 2024), HMMT February 2025 (HMMT, 2025), and BrUMO
2025 (BrUMO, 2025). We report average@32 accuracy using temperature 1.0 and a maximum genera-
tion length of 32,768 tokens for all experiments.
Baselines.We compare against these baseline methods: (1)Base (Instruct): the base model with thinking
mode enabled; (2)GRPO(Shao et al., 2024): standard group relative policy optimization with binary outcome
rewards, using 16 rollouts per problem and maximum generation length of 32,768 tokens; (3)OPSD(Zhao
et al., 2026): a self-distillation method where a single LLM acts as both teacher and student. The teacher
conditions on privileged reasoning traces while the student sees only the question, and training minimizes
per-token divergence between these distributions over the student’s own rollouts. Note that OPSD did not
release the checkpoint, so we report the results from the original paper, which uses average@16 for evaluation;
(4)QuestA(Li et al., 2026): a state-of-the-art RLVR method that injects partial solutions into training
problems to create a smoother difficulty curriculum and denser reward signals. We reproduce their method
following their official guidelines except that we use a more generic and recent base model Qwen3-series.
Implementation details.For all reinforcement fine-tuning methods, we sample 16 responses per problem and
disable the KL penalty. We use 64 H100 (80GB) GPUs for training all model variants. We use the AdamW
optimizer with a learning rate of1 ×10−6. The maximum rollout length is 32,768 tokens. At training time,
we retrieve k= 1reasoning trace per problem, and we keep k= 1for all experiments unless otherwise
specified. All experiments use full-parameter training and are conducted using VeRL (Sheng et al., 2025).
Hyperparameterτfor retriever training is set to 0.05. We include additional details in Section A.
4.2 Main Results
Table 1 reports results on competition-level mathematical reasoning benchmarks across two model scales.
On Qwen3-1.7B, RA-RFT improves over standard GRPO by +4.7 on AIME24, +7.1 on AIME25, +1.9
on HMMT25, and +2.6 on BrUMO25. On Qwen3-4B, RA-RFT achieves the best results on three out of
four benchmarks, with notable gains on BrUMO25 (+5.9) and AIME25 (+2.8) over GRPO. Compared to
OPSD, which leverages privileged reasoning traces via on-policy self-distillation, RA-RFT achieves stronger
performance using only outcome rewards. We report QuestA results for Qwen3-1.7B only, as the data the
authors released did not yield gains at the 4B scale in our preliminary experiments.
6

0 200 400 600 800 1000
Training Step42.044.046.048.050.052.054.0Accuracy (%, average@32)
 Step 0: 40.8%Step 0: 48.1%55.1%
50.4%AIME 2024
0 200 400 600 800 1000
Training Step32.535.037.540.042.545.047.5Accuracy (%, average@32)
 Step 0: 33.1%Step 0: 35.9%48.8%
41.6%AIME 2025
0 200 400 600 800 1000
Training Step22.023.024.025.026.027.028.029.0Accuracy (%, average@32)
 Step 0: 21.9%Step 0: 23.4%28.2%
26.2%HMMT Feb 2025
0 200 400 600 800 1000
Training Step46.048.050.052.054.056.058.0Accuracy (%, average@32)
 Step 0: 46.1%Step 0: 50.9%57.4%
54.8%BRUMO 2025RA-RFT (Ours) GRPO (Baseline)Figure 3Comparison of validation average@32 accuracy across training steps on Qwen3-1.7B. The step-0 accuracy for
each method is annotated in the bottom-right corner of each subfigure. Notably, RA-RFT starts at a lower step-0
accuracy than GRPO across all benchmarks, as the base model is initially distracted by the unfamiliar retrieved
context. As reinforcement fine-tuning progresses, the model learns to effectively leverage the retrieved reasoning traces,
ultimately surpassing GRPO by a substantial margin.
Method AIME24 AIME25 HMMT25 BrUMO25 Avg.
SFT 48.9 36.0 23.1 50.7 39.7
RA-SFT 48.6 39.5 22.8 51.2 40.5
GRPO 50.4 41.6 26.3 54.8 43.3
RA-RFT (Ours) 55.1 48.7 28.2 57.4 47.4
Table 2Comparison of training objectives and retrieval augmentation on Qwen3-1.7B. We report average@32 accuracy.
Highlighted row marks our method.
Figure 3 compares validation accuracy curves across training steps. RA-RFT not only converges to a higher
final accuracy than GRPO but also exhibits faster learning in early training stages, suggesting that retrieved
reasoning traces provide informative learning signals that accelerate policy improvement.
4.3 Ablation Studies
SFT vs. RA-SFT vs. GRPO vs. RA-RFT.We compare four training configurations to disentangle retrieval
augmentation from the training objective.SFTperforms standard supervised fine-tuning on teacher-generated
reasoning traces;RA-SFTadds the same retrieved trace to the SFT prompt, providing the same context as
RA-RFT but with a cross-entropy objective. Table 2 reports results on Qwen3-1.7B. Retrieval augmentation
provides negligible benefit under supervised fine-tuning (RA-SFT 40.5 vs. SFT 39.7), because SFT minimizes
a token-level loss against fixed targets and the model simply imitates the teacher regardless of what auxiliary
information is in the prompt. The contrast with RA-RFT confirms that the benefit of retrieved reasoning
traces is unlocked only when the training objective allows the model to explore and selectively integrate
external evidence into its own reasoning process, rather than passively copying a fixed solution.
Context diversity drives problem-level accuracy variation.Figure 4 examines how different retrieved contexts
affect accuracy at the individual problem level. For each test problem, we retrieve top-4traces from our
reasoning-aware retriever and evaluate the model independently under each retrieved context, plotting the
resulting accuracy distribution alongside the raw GRPO baseline. We observe that the top-1 retrieved context
(dark blue) lifts the per-problem accuracy above the GRPO baseline for the majority of problems across all
four benchmarks. In addition, there is a substantial spread across retrieved contexts for many problems: the
vertical range bars reveal that the best-performing context for a given problem can exceed the worst-performing
context by a wide margin, indicating that different retrieved traces surface distinct solution strategies.
Retriever comparison.We compare our reasoning-aware retriever against several alternatives before and after
finetuning with reasoning supervision: (1)Qwen3-Embedding-4B(Zhang et al., 2025b), a strong single-vector
dense retriever, used here as the off-the-shelf RAG baseline; (2)Reason-ModernColBERT(Chaffin, 2025), a
late-interaction multi-vector model trained on reasoning-intensive data that performs well on the BRIGHT
7

Evaluation Sample0.00.20.40.60.81.0Accuracy (average@32)50.4%55.1%AIME 2024
Evaluation Sample41.6%48.7%AIME 2025
Evaluation Sample26.3%28.2%HMMT Feb 2025
Evaluation Sample54.8%57.4%BruMO 2025
GRPO RA-RFT (Top-1 Context) RA-RFT (Other Contexts) Context Accuracy RangeFigure 4Per-sample accuracy of RA-RFT under different retrieved contexts versus raw GRPO on Qwen3-1.7B. Each
sample is sorted by its average@32 accuracy in raw GRPO (orange diamond). Dark blue dots show accuracy under the
top-1 retrieved context. Light blue dots show accuracy under other retrieved contexts. Dashed horizontal lines mark
the benchmark-level averages on all problems.
Retriever R@1 AIME24 AIME25 HMMT25 BrUMO25 Avg.
Qwen3-Emb-4B 2.3 47.5 36.6 23.6 46.1 38.5
Qwen3-Emb-4B +
 14.7 49.8 39.0 24.8 48.3 40.5
R-ModernColBERT 7.2 48.3 40.9 22.9 50.8 40.7
R-ModernColBERT +
 43.5 55.1 48.7 28.2 57.4 47.4
Random Trace Context 0.0 46.9 36.8 22.4 44.2 37.6
Table 3Ablation on retrieval models using Qwen3-1.7B with RA-RFT training.
 indicates the retriever is fine-tuned
with our reasoning supervision. R@1 is the retriever’s recall@1 (in %) on a held-out gold-relevance evaluation set of
10,000 samples. The remaining columns are downstream RA-RFT average@32 accuracy (in %). The bottom block
reports test runs that replace the reasoning-aware top-1 trace with a random trace from the same problem-type
bucket. Gray denotes the best retriever configuration that is adopted in all experiments unless otherwise specified.
Qwen3-Embed-4B is short for Qwen3-Embedding-4B. R-ModernColBERT is short for Reason-ModernColBERT.
benchmark (Su et al., 2025); (3)Random trace from the corpus, which can ablate the effect of external retriever.
Table 3 reports the results on Qwen3-1.7B. First, gold relevance supervision is necessary: fine-tuning lifts
Qwen3-Embedding-4B from 38.5 to 40.5 and Reason-ModernColBERT from 40.7 to 47.4 average accuracy,
confirming that aligning retrieval scores with reasoning utility is essential. Second, multi-vector late-interaction
retrieval is better suited for reasoning-intensive retrieval (Chaffin, 2025): Reason-ModernColBERT already
matches Qwen3-Emb-4B without supervision (40.7 vs. 38.5), and the gap widens after fine-tuning (47.4 vs.
40.5). Direct ranking evaluation against held-out gold-relevance labels confirms the same trend, with our
fine-tuned retriever also achieving substantially higher recall@1 than the off-the-shelf checkpoint. Notably,
even when the retriever does not perfectly recover the gold-relevant trace, the gain from conditioning RL
on externally retrieved reasoning-intensive traces remains substantial, indicating that RA-RFT is robust to
imperfect retrieval and benefits from the broader pool of structurally analogous reasoning patterns surfaced
by the retriever.
5 Conclusion
We introduce Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT), a post-training framework that
teaches language models to reason by analogy. The key insight is that effective retrieval for reasoning must
be grounded in reasoning utility rather than surface-level similarity: a structurally analogous problem, even
if topically unrelated, can provide a transferable solution scaffold that guides the model toward the right
problem reduction. We hope RA-RFT can inspire future work on extending analogical reasoning to other
domains, richer retrieval corpora, and tighter integration between retrieval and language model post-training.
8

References
Negar Arabzadeh, Wenjie Ma, Sewon Min, and Matei Zaharia. Restructuring the corpus makes RAG work for math.
InThe 5th Workshop on Mathematical Reasoning and AI at NeurIPS 2025, 2025. https://openreview.net/forum?
id=6cYmnzJViJ.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakan-
tan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger,
Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher
Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner,
Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners.
In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors,
Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing
Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. https://proceedings.neurips.cc/paper/2020/
hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.
BrUMO. BrUMO 2025, 2025.https://brumo.org.
Locke Cai and Ivan Provilkov. Escaping the verifier: Learning to reason via demonstrations.arXiv preprint
arXiv:2511.21667, 2025.
Antoine Chaffin. Reason-moderncolbert, 2025.https://huggingface.co/lightonai/Reason-ModernColBERT.
DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.arXiv preprint
arXiv:2501.12948, 2025.
Aniket Didolkar, Nicolas Ballas, Sanjeev Arora, and Anirudh Goyal. Metacognitive reuse: Turning recurring LLM
reasoning into concise behaviors.arXiv preprint arXiv:2509.13237, 2025.
Dedre Gentner. Structure-mapping: A theoretical framework for analogy.Cognitive Science, 7(2):155–170, 1983.
Anirudh Goyal, Abram L. Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke, Adrià Puigdomènech
Badia, Arthur Guez, Mehdi Mirza, Peter Conway Humphreys, Ksenia Konyushkova, Michal Valko, Simon Osindero,
Timothy P. Lillicrap, Nicolas Heess, and Charles Blundell. Retrieval-augmented reinforcement learning. In Kamalika
Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato, editors,International
Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, Proceedings of Machine
Learning Research, pages 7740–7765. PMLR, 2022.https://proceedings.mlr.press/v162/goyal22a.html.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model
pre-training. InInternational conference on machine learning, pages 3929–3938. PMLR, 2020.
HMMT. HMMT february 2025, 2025.https://www.hmmt.org.
Keith J. Holyoak and Paul Thagard.Mental Leaps: Analogy in Creative Thought. MIT Press, 1996.
Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. https://github.com/huggingface/
open-r1.
Junwei Lan, Jianlyu Chen, Zheng Liu, Chaofan Li, Siqi Bao, and Defu Lian. Retro*: Optimizing LLMs for reasoning-
intensive document retrieval.arXiv preprint arXiv:2509.24869, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation
for knowledge-intensive nlp tasks. InAdvances in Neural Information Processing Systems, volume 33, 2020.
Jiazheng Li, Hongzhou Lin, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Yi Wu, and Jingzhao Zhang. Questa:
Expanding reasoning capacity in LLMs via question augmentation. InThe Fourteenth International Conference on
Learning Representations, 2026.https://openreview.net/forum?id=3MifB0f7qR.
Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, and
Xipeng Qiu. R3-RAG: learning step-by-step reasoning and retrieval for llms via reinforcement learning. In Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,Findings of the Association for
Computational Linguistics: EMNLP 2025, Suzhou, China, November 4-9, 2025, pages 10491–10507. Association for
Computational Linguistics, 2025.https://aclanthology.org/2025.findings-emnlp.554/.
9

Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What makes good in-context
examples for GPT-3? InProceedings of Deep Learning Inside Out (DeeLIO): The 3rd Workshop on Knowledge
Extraction and Integration for Deep Learning Architectures, pages 100–114, 2022.
Jingyu Liu, Jiaen Lin, and Yong Liu. How much can rag help the reasoning of llm?arXiv preprint arXiv:2410.02338,
2024.
Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding
r1-zero-like training: A critical perspective. InSecond Conference on Language Modeling, 2025. https://openreview.
net/forum?id=5PAF7PAY2Y.
MAA. American invitational mathematics examination (AIME), 2024. https://maa.org/math-competitions/
american-invitational-mathematics-examination-aime.
MiniMax. Minimax-m1: Scaling test-time compute efficiently with lightning attention.CoRR, abs/2506.13585, 2025.
doi: 10.48550/ARXIV.2506.13585.https://doi.org/10.48550/arXiv.2506.13585.
OpenAI. Learning to reason with llms. 2024.https://openai.com/index/learning-to-reason-with-llms/.
Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust
region in LLM reinforcement learning.arXiv preprint arXiv:2602.04879, 2026.
Qwen Team. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham.
In-context retrieval-augmented language models.Transactions of the Association for Computational Linguistics, 11:
1316–1331, 2023.
Ohad Rubin, Jonathan Herzig, and Jonathan Berant. Learning to retrieve prompts for in-context learning. In
Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies, pages 2655–2671, 2022.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo.
Deepseekmath: Pushing the limits of mathematical reasoning in open language models.CoRR, abs/2402.03300,
2024. doi: 10.48550/ARXIV.2402.03300.https://doi.org/10.48550/arXiv.2402.03300.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and
Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. InProceedings of the Twentieth European Conference
on Computer Systems, pages 1279–1297, 2025.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han yu Wang, Liu Haisu, Quan Shi,
Zachary S Siegel, Michael Tang, Ruoxi Sun, Jinsung Yoon, Sercan O Arik, Danqi Chen, and Tao Yu. BRIGHT: A
realistic and challenging benchmark for reasoning-intensive retrieval. InThe Thirteenth International Conference on
Learning Representations, 2025.https://openreview.net/forum?id=ykuc5q381b.
Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time
learning with adaptive memory. InProceedings of the 19th Conference of the European Chapter of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 7080–7106, 2026.
Di Wu, Devendra Singh Sachan, Wen-tau Yih, and Mingda Chen. Procedural knowledge at scale improves reasoning.
arXiv preprint arXiv:2604.01348, 2026.
Xiaocong Yang, Jiacheng Lin, Ziqi Wang, and Chengxiang Zhai. Learning by analogy: Enhancing few-shot prompting
for math word problem solving with computational graph-based retrieval.arXiv preprint arXiv:2411.16454, 2024.
Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models.
arXiv preprint arXiv:2602.12275, 2026.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu,
Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang,
Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli
Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and
Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. InThe Thirty-ninth Annual
Conference on Neural Information Processing Systems, 2025.https://openreview.net/forum?id=2a36EMSSTp.
Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. Stephint: Multi-level
stepwise hints enhance reinforcement learning to reason.arXiv preprint arXiv:2507.02841, 2025a.
10

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng
Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking
through foundation models.arXiv preprint arXiv:2506.05176, 2025b.
Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner:
On-policy self-distillation for large language models, 2026.https://arxiv.org/abs/2601.18734.
Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. Synapse: Trajectory-as-exemplar prompting with memory
for computer control. InThe Twelfth International Conference on Learning Representations, 2024.
11

Appendix
A Additional Implementation Details
Algorithm 1Retrieval-Augmented Reinforcement Fine-Tuning (RA-RFT)
Require: Training set D={(qi, ai)}N
i=1; corpus C={(pj, tj)}M
j=1; judge model Mjudge; target model Mϕ;
dense encoderR θ; group sizeG; number of retrieved tracesk; temperatureτ
1:// Stage 1: Gold-Relevance Distillation (Section 3.2)
2:foreach(q i, ai)∈ Ddo
3:foreachc∈ Cdo
4:y i,c← M judge(qi, c)▷binary reasoning-relevance label
5:C+
i← {c∈ C:y i,c= 1}
6:// Stage 2: Reasoning-Aware Retriever Training (Section 3.3)
7:whileR θnot convergeddo
8:Sample minibatchB ⊂ D
9:L retrieval ← −X
i∈BX
c+∈C+
ilogexp(⟨e qi,ec+⟩/τ)P
c∈C iexp(⟨e qi,ec⟩/τ)
10:Updateθ←θ−η∇ θLretrieval
11:// Stage 3: Reinforcement Fine-Tuning with Retrieved Demonstrations (Section 3.4)
12:whileM ϕnot convergeddo
13:Sample minibatchB ⊂ D
14:foreach(q, a)∈ Bdo
15:{c 1, . . . , c k} ← R θ(q,C)▷retrieve top-kreasoning traces
16:Sample{ˆa 1, . . . ,ˆa G} ∼ M ϕ(· |q, c 1, . . . , c k)
17:r g←r(ˆa g, a)forg= 1, . . . , G ▷outcome-based rewards
18:Compute advantages{A g}G
g=1from{r g}
19:Updateϕvia policy optimization with{A g}G
g=1
20:returntrained modelM ϕ
Gold Relevance Distillation.Algorithm 1 outlines the complete RA-RFT pipeline. In Stage 1 (lines 3–7), the
judge model Mjudge(GPT-4o) evaluates each query-candidate pair to produce binary reasoning-relevance
labels. Concretely, the judge is presented with two question-answer pairs and asked to determine whether they
share transferable reasoning patterns, such as reliance on the same theorems, algorithms, or proof techniques
(see the full prompt template in Figure 7). The judge outputs a brief justification followed by a binary yes/no
relevance decision. Naïvely evaluating all |D| × |C| pairs would be prohibitively expensive. To reduce the
number of comparisons, we leverage the coarseproblem-typelabels (e.g. algebra, combinatorics, geometry,
number theory) already present in the metadata of the source datasets. The judge evaluation is then restricted
to pairs that share the same problem type, as cross-type reasoning transfer is rare in practice. This filtering
reduces the total number of judge calls by approximately an order of magnitude while preserving nearly all
reasoning-relevant pairs.
VerifiableRewardDesign.Weadoptabinaryoutcome-basedrewardfunction r(ˆag, a)thatverifiesthecorrectness
of each sampled responseˆa gagainst the ground-truth answera. Formally:
r(ˆag, a) =(
1ifverify(extract(ˆa g), a) =True,
0otherwise,(4)
whereextract (ˆag)parses the model output to locate the final \boxed{...} expression, and verify (·,·)performs
symbolic equivalence checking between the extracted prediction and the ground truth. The verification handles
12

Setting AIME24 AIME25 HMMT25 BrUMO25 Avg.
GRPO (no retrieval) 50.4 41.6 26.3 54.8 43.3
GRPO + retrieval at inference-time 44.3 35.3 22.1 49.2 37.7
RA-RFT (Ours) 55.1 48.7 28.2 57.4 47.4
Table 4Inference-time-only retrieval diagnostic: GRPO checkpoint evaluated with the reasoning-aware retriever’s top-1
trace prepended at inference. Highlighted row marks our method.
mathematical equivalences such as0 .5 =1
2and2 x=x·2by normalizing both the prediction and the ground
truth into canonical symbolic forms before comparison. We implement this using the math_verify library,
which supports both LaTeX-level and expression-level extraction for predictions and LaTeX-level extraction
for gold answers.
Retriever Training Details.We fine-tune lightonai/Reason-ModernColBERT , a ColBERT-style multi-vector
model pre-trained on reasoning-intensive corpora. Multi-vector retrieval is preferred over single-vector retrieval
because late interaction over per-token embeddings better captures structural reasoning similarity (e.g. shared
proof techniques or algorithmic patterns) beyond surface-level lexical overlap. We train with contrastive loss
at temperature τ=0.05for 3 epochs, with a batch size of 128, learning rate3 ×10−5, linear warmup over 10%
of steps, and weight decay0 .01. Embeddings are gathered across all GPUs before computing the contrastive
objective, maximizing the number of in-batch negatives without increasing per-device memory. All training is
done inbfloat16. We hold out 5% of the training data for evaluation and select the checkpoint with the best
recall@1 on this set for usage in the following reinforcement fine-tuning stage. The single vector retriever was
trained with the same hyperparameters and training data, but using Qwen/Qwen3-Embedding-4B as the
base model.
Train/test contamination.The QuestA training queries and the OpenR1-Math-220K (Hugging Face, 2025)
retrievalcorpusarebothderivedfromNuminaMath-1.5, whosecontributingsources(AoPSForum, AMC/AIME
1984–2023, MATH, Olympiads) predate the AIME 2024/2025, HMMT February 2025, and BrUMO 2025
benchmarks we evaluate on. In addition, the open-r1 toolkit ships an 8-gram decontamination script targeting
AIME 2024/25 and MATH-500 (Hugging Face, 2025), so the released corpus excludes verbatim matches
against those benchmarks. Therefore, the training and corpus dataset do not contain any samples from test
benchmarks used.
On the Choice of Models for Corpus Curation.Neither Qwen3-235B-A22B (trace generator) nor GPT-4o
(relevance judge) is essential to RA-RFT. Both are used only for one-time, offline corpus curation and never
appear in the training or inference loop. Crucially, RA-RFT isnota distillation method: corpus traces serve
as in-context demonstrations fordifferentqueries during RL rollouts, and the policy is supervised only by the
verifiable outcome reward on its own rollout, never by a token-level loss against any teacher response. We
chose Qwen3-235B-A22B because a stronger trace generator yields a higher fraction of correct derivations and
thus a cleaner retrieval corpus, and GPT-4o for the judge purely for cost: it offered the best per-call price
among frontier models. Other open-source LLM should yield qualitatively similar results in either role.
B Additional Experimental Results
Inference-time-only retrieval diagnostic.To probe whether the RA-RFT gain comes from inference-time
access to retrieved context rather than from training-time co-adaptation, we evaluate the standard GRPO
checkpoint with the same reasoning-aware retriever’s top-1 trace prepended only at inference. We stress this
is adiagnostic, not a competitive baseline: the GRPO checkpoint has never seen retrieved context during
training, so it is not optimized to consume one. As shown in Table 4, this configuration does not match
RA-RFT, isolating the contribution of training the policy under retrieved demonstrations rather than just
supplying them at evaluation time.
Policy optimization methods.Our framework is agnostic to the choice of policy optimization algorithm
(Section 3.4). We instantiate RA-RFT with two different base optimizers (RLOO and DAPO) while keeping
13

Method AIME24 AIME25 HMMT25 BrUMO25 Avg.
RLOO 51.7 39.2 24.4 51.7 41.8
RA-RLOO 55.8 44.6 28.6 53.1 45.5
DAPO 50.6 42.0 24.5 53.1 42.6
RA-DAPO 54.0 42.8 27.5 53.3 44.4
Table 5Ablation on policy optimization algorithms within the RA-RFT framework using Qwen3-1.7B.
all other components fixed, and compare each against its retrieval-augmented counterpart. As shown in
Table 5, retrieval augmentation consistently improves average accuracy by 3.7 and 1.8 points over RLOO and
DAPO respectively, demonstrating that the benefit of reasoning-aware retrieval is not tied to any particular
policy optimization algorithm and transfers across different training objectives.
C Case Study
Figure 5 illustrates a representative case where RA-RFT dramatically outperforms standard GRPO on an
AIME 2025 problem (13/32 vs. 1/32 sampling correct). The retrieved context question shares no surface
similarity with the target, yet it encodes the same structural reasoning pattern, providing a scaffold that
guides the model toward the correct problem reduction. Standard GRPO attempted a different solution
because of lacking this analogical signal, but misinterpreted the requirement and ultimately arrived at a wrong
answer. This example highlights the core mechanism of RA-RFT: reasoning-aware retrieval transfers solution
strategies rather than surface features.
D Prompt Templates
We use three prompt templates throughout the RA-RFT pipeline, each serving a distinct role. Figure 8 shows
thereasoning trace summarizationprompt directly taken from previous work (Arabzadeh et al., 2025), which
instructs Qwen3-235B-A22B to rewrite raw corpus solutions into concise, step-by-step traces that highlight
reusable reasoning strategies. Figure 7 shows thegold-relevance distillation judgeprompt, which asks GPT-4o
to determine whether two questions share transferable reasoning patterns and output a binary relevance label
along with a brief justification. Figure 6 shows theRA-RFT training and inferenceprompt, which prepends
a retrieved reasoning-analogous exemplar (reference question and its reasoning trace) to the target query,
providing the model with an in-context demonstration to guide its solution.
E Limitations
RA-RFT adds a separate retriever and a one-time GPT-4o judge pass for gold-relevance labels: the retriever
is small relative to the policy and frozen during RL. In addition, training the reasoning-aware retriever does
require a one-time annotation pass from a strong judge model to produce gold-relevance labels, which incurs
labeling cost beyond what standard RLVR pipelines need. We consider this a favorable trade-off: the upfront
labeling budget is fixed and modest, while the trained retriever enables the diversity of retrieved contexts
that drive substantial and consistent gains across benchmarks and model scales, and the retriever can be
reused across different training runs and even different base models as long as the retrieval corpus remains the
same. While we validate RA-RFT on competition-level mathematical reasoning benchmarks, the underlying
principle of analogical reasoning is domain-general, and extending the framework to other reasoning-intensive
domains, such as code generation or scientific problem solving, requires only constructing an appropriate
retrieval corpus with reasoning traces, which we leave for future work.
14

How does RA-RFT help solve question with retrieved demonstration at inference?Target Question (from AIME 2025)Sixteen chairs are arranged in a row. Eight people each select a chair in which to sit so that no person sits next to two other people. Let  be the number of subsets of the 16 chairs that could be selected. Find the remainder when  is divided by 1000.NNContext Question & Reasoning Trace
The sides of a convex -gon are sequentially denoted as . Each side is to be painted one of three colors: red, yellow, or blue, such that no two adjacent sides have the same color. How many different coloring methods are there?na1,a2,⋯,anStep 1: Define the count Step 2: Base case Step 3: Count linear arrangements (ignoring cycle) Step 4: Account for cyclic adjacency …Key Observation: … This leads to a combinatorial structure where the selected chairs are arranged in blocks of size 1 or 2, separated by at least one unselected chair. Strategy: We calculate the number of ways to arrange valid block conﬁgurations of size 1 or 2 in the 16 chairs, such that: … We define: k = number of blocks of size 2 m = number of blocks of size 1 … This gives us 5 possible cases, and their contributions are:Key Interpretation: There are 16 chairs in a row. 8 people each sit in a different chair. The condition: no person sits next to two other people. This means no three people are seated consecutively. So, the problem reduces to counting the number of ways to choose 8 chairs out of 16 such that no three selected chairs are consecutive. Approach: Recurrence and Dynamic Programming We can model this with a dynamic programming solution: Let dp[i][j] be the number of ways to place j people in the first i chairs, without any 3 consecutive people. The recurrence relation is: dp[i][j]=dp[i−1][j]+dp[i−1][j−1]−dp[i−3][j−2] Computation Steps We compute up to dp[16][8]=171,848. Final Step: Modulo 1000 We are asked to find the remainder when this number is divided by 1000: 171,848 mod 1000 = 848 
✅ Final Answer 848  (k, m)tC(t, k)C(9, t)Contribution(0, 8)8199(1, 6)7736252(2, 4)615841260(3, 2)5101261260(4, 0)41126126Total = 9 + 252 + 1260 + 1260 + 126 = 2907 Final Step: Compute 2907 mod 1000: 2907÷1000=2 remainder 907 Final Answer: 907  RA-RFT Answer (Correct 13 / Total 32)Standard GRPO Answer (Correct 1 / Total 32)Figure 5Case study comparing RA-RFT and standard GRPO on an AIME 2025 problem. Left: the RA-RFT
model, conditioned on a retrieved reasoning trace about coloring a convex n-gon, correctly identifies a block-counting
decomposition and arrives at the answer 907. Right: standard GRPO without retrieval misinterprets the adjacency
constraint as a no-three-consecutive condition and applies an incorrect DP recurrence. The retrieved context question
is structurally analogous since both problems count valid configurations under local adjacency constraints. However
they share no surface-level similarity as the target problem involves the context of arranging the chairs and context
question is about coloring a convexn-gon.
15

RA-RFT Training & Inference Prompt Template
# Related Question-Answer Reference Below is a related question that may help you solve the upcoming problem. Knowledge of the reasoning steps can assist in answering the next question. ## Reference Question {reference_question} ## Reasoning {reasoning_trace} ——— Now, please solve the following question: ## Question {training_query_question}Figure 6Prompt template used during both RA-RFT training and inference. The retrieved reasoning-analogous
exemplar, including the reference question and its reasoning trace, is prepended to the training query.
16

Gold Relevance Distillation Judge Prompt Template
# Task Description You will be provided with two questions from the internet. Your goal is to identify if these two questions are closely related to each other. If so, we consider them as a reasoning-intensive relevant pair. Some examples include: 1. Two questions need the domain-specific knowledge of the same underlying algorithms or theorems to solve. 2. Knowledge of the second question-answer pair can greatly help answer the first question. However, do not consider general knowledge as a reason for relevance. Instead, consider both questions sharing the same reasoning pattern as a reason for relevance. Here are some examples: # Examples {in_context_example} # Task Format Format your response in markdown as follows: ## The Reason of Relevance \reason{[They are relevant because they both require using the theorem of …]} ## Is relevant? \answer{[yes/no]} # Question-Answer Pair 1 {query_question} # Question- Answer Pair 2 {corpus_question} {corpus_reasoning_trace} Figure 7Prompt template for the judge model in gold-relevance distillation (Section 3.2). The judge evaluates whether
two questions share transferable reasoning patterns, producing binary relevance labels for retriever training.
17

Reasoning Trace Summarization Prompt
Instruction. You are given a math problem and its solution. Your goal is to rewrite the solution into a clearly labeled, step-by-step concise format that teaches how to solve the problem. Guidelines. -Each step should reflect a logical phase in solving the problem. -Use a concise “cheatsheet” style so learners can generalize the strategy to harder or related problems. -If the problem or solution is incomplete or noisy, infer reasonable steps and still provide guidance. -Focus on key ideas, strategies, or techniques the learner can reuse. -Keep at most 7 steps; be concise and avoid verbosity. -Avoid over-fragmentation; combine actions logically where appropriate. Output format. Problem: [Copy the problem exactly as given] Step 1: [Short label] [Brief explanation of what’s being done and why, with any relevant calculations] Step 2: [Next short label] [Brief explanation and math] ... Step N: [Final step] [Brief explanation and final processing or insight] Therefore, the final answer is: $\boxed{[final answer]}$ Given problem: {corpus_question} {corpus_solution}
Example Reasoning Trace from CorpusProblem: The set of positive integer solutions to the inequality $2x-3 \leq 5$ is____. Step 1: Solve the inequality algebraically Add 3 to both sides: $2x - 3 + 3 \leq 5 + 3 \Rightarrow 2x \leq 8$. Divide by 2: $x \leq 4$. Step 2: Identify the domain constraint We seek positive integer solutions, so $x$ must satisfy $x > 0$ and $x \in \mathbb{Z}$. Thus, $x = 1, 2, 3, \dots$ Step 3: Combine conditions and list valid solutions From $x \leq 4$ and $x \geq 1$ (positive integers), the possible values are $x = 1, 2, 3, 4$. Therefore, the final answer is: $\boxed{1, 2, 3, 4}$Figure 8Prompt template for reasoning trace summarization (top) and an example summarized reasoning trace from
the corpus (bottom). Raw solutions are condensed into concise, step-by-step traces that highlight reusable reasoning
strategies.
18