# Procedural Knowledge at Scale Improves Reasoning

**Authors**: Di Wu, Devendra Singh Sachan, Wen-tau Yih, Mingda Chen

**Published**: 2026-04-01 20:01:47

**PDF URL**: [https://arxiv.org/pdf/2604.01348v1](https://arxiv.org/pdf/2604.01348v1)

## Abstract
Test-time scaling has emerged as an effective way to improve language models on challenging reasoning tasks. However, most existing methods treat each problem in isolation and do not systematically reuse knowledge from prior reasoning trajectories. In particular, they underutilize procedural knowledge: how to reframe a problem, choose an approach, and verify or backtrack when needed. We introduce Reasoning Memory, a retrieval-augmented generation (RAG) framework for reasoning models that explicitly retrieves and reuses procedural knowledge at scale. Starting from existing corpora of step-by-step reasoning trajectories, we decompose each trajectory into self-contained subquestion-subroutine pairs, yielding a datastore of 32 million compact procedural knowledge entries. At inference time, a lightweight in-thought prompt lets the model verbalize the core subquestion, retrieve relevant subroutines within its reasoning trace, and reason under diverse retrieved subroutines as implicit procedural priors. Across six math, science, and coding benchmarks, Reasoning Memory consistently outperforms RAG with document, trajectory, and template knowledge, as well as a compute-matched test-time scaling baseline. With a higher inference budget, it improves over no retrieval by up to 19.2% and over the strongest compute-matched baseline by 7.9% across task types. Ablation studies show that these gains come from two key factors: the broad procedural coverage of the source trajectories and our decomposition and retrieval design, which together enable effective extraction and reuse of procedural knowledge.

## Full Text


<!-- PDF content starts -->

Procedural Knowledge at Scale Improves Reasoning
Di Wu1∗,Devandra Singh Sachan2,Wen-tau Yih2,Mingda Chen∗
1University of California, Los Angeles,2Meta FAIR
∗Work done while at Meta FAIR
Test-time scaling has emerged as an effective way to improve language models on challenging reasoning
tasks. However, most existing methods treat each problem in isolation and do not systematically reuse
knowledge from prior reasoning trajectories. In particular, they underutilize procedural knowledge:
how to reframe a problem, choose an approach, and verify or backtrack when needed. We introduce
Reasoning Memory, a retrieval-augmented generation (RAG) framework for reasoning models that
explicitly retrieves and reuses procedural knowledge at scale. Starting from existing corpora of step-by-
step reasoning trajectories, we decompose each trajectory into self-contained subquestion-subroutine
pairs, yielding a datastore of 32 million compact procedural knowledge entries. At inference time,
a lightweight in-thought prompt lets the model verbalize the core subquestion, retrieve relevant
subroutines within its reasoning trace, and reason under diverse retrieved subroutines as implicit
procedural priors. Across six math, science, and coding benchmarks, Reasoning Memory consistently
outperforms RAG with document, trajectory, and template knowledge, as well as a compute-matched
test-time scaling baseline. With a higher inference budget, it improves over no retrieval by up to 19.2%
and over the strongest compute-matched baseline by 7.9% across task types. Ablation studies show
that these gains come from two key factors: the broad procedural coverage of the source trajectories
and our decomposition and retrieval design, which together enable effective extraction and reuse of
procedural knowledge.
Date:April 3, 2026
Correspondence:diwu@cs.ucla.edu
Code:https://github.com/facebookresearch/reasoning-memory
1 Introduction
Test-time scaling, where additional inference-time compute is allocated per instance to improve accuracy, has
quickly become a central paradigm for frontier reasoning tasks in math, science, and coding (Wei et al., 2022;
Wang et al., 2023; Yao et al., 2023; Snell et al., 2024; Chen et al., 2024; Lee et al., 2025; Muennighoff et al.,
2025). Recently, reasoning models such as OpenAI’s o1 series (OpenAI, 2024), DeepSeek R1 (DeepSeek-AI,
2025), and QwQ (Qwen Team, 2024) are explicitly trained to produce “thinking” tokens. At inference time,
their performance scales up as they are allowed to “think longer” with a larger token budget.
Despite this progress, one limitation of current test-time scaling strategies is that they do not reuse knowledge
across reasoning runs. Such knowledge includes bothfactual knowledge, such as definitions, formulas,
and theorems, and more importantlyprocedural knowledge: how to reframe a problem, decompose it into
subquestions, exploit structure, choose an approach, and verify or backtrack when needed. Discarding
procedural knowledge from prior reasoning trajectories can hurt both accuracy and efficiency, as models are
forced to repeatedly rediscover useful strategies and revisit unsuccessful lines of thought.
Retrieval-augmented generation (RAG) provides a natural framework for accumulating useful knowledge
offline and incorporating it at test time. Recent work has begun to interleave retrieval with reasoning or to
introduce general-purpose datastores that improve reasoning performance (Li et al., 2025a; Song et al., 2025;
Lyu et al., 2025). However, these systems primarily improve access to general background knowledge, while
leaving reasoning-specific procedural knowledge implicit. For difficult reasoning tasks, the relevant bottleneck
is often not missing facts alone, but missing guidance on what subquestion to solve next and how to solve it.
As a result, retrieved context is often only loosely aligned with the model’s current reasoning state. This
1arXiv:2604.01348v1  [cs.CL]  1 Apr 2026

A triangle has side lengths 13, 14, and 15. Let O be its circumcenter and H be its orthocenter. Compute OH^2.Okay, so I got this math question……(10k tokens)For problems like this, I should use Heron’s formula.1. Let the side lengths be a, b, c.2. Compute the semiperimeter s = (a + b + c) / 2.3. Compute the area K = sqrt(s(s-a)(s-b)(s-c)).4. Simplify the expression to keep an exact value.What is the relationship between a triangle’s circumradius and the distance from its circumcenter to its orthocenter? How to compute the area of a triangle given its side lengths?How to compute the circumradius of a triangle given side lengths?SQ1SR1SQ2SR2
A triangular park has sides AB=90m, BC=100m,  and CA=170m. A straight footpath from B meets AC at D such that BD ⟂ AC.Whatistheareaofthepark?<think> … let me frame a Google search query for a similar basic problem: area of triangle given three sides
SQ3SR3Procedural Knowledge DatastoreSubquestionSubroutineReasoning Trajectory
Query VerbalizationScaling with Procedural Knowledge Retrieval
Trajectory1-1SQ1SR1SQ2SR2SQ3SR3Trajectory1-2Trajectory1-3Trajectory2-1Trajectory2-2Trajectory2-3Trajectory3-1Trajectory3-2Trajectory3-3
(a)DatastoreConstruction
(b)RAGInferenceFigure 1 Illustration of the Reasoning Memory framework.(a) We extract self-contained procedural knowledge from diverse
public reasoning trajectories to construct a datastore. (b) At inference time, reasoning models retrieve and reuse
relevant procedures within the reasoning trace, enabling test-time scaling through in-thought procedural retrieval.
weak alignment can be especially problematic for reasoning models, whose long reasoning traces may amplify
irrelevant contexts. Our pilot study shows that standard document-level RAG can yield limited or even
negative gains for reasoning models, suggesting the challenge of retrieving knowledge in a form well aligned
with the problem and easy for reasoning models to leverage (§5.1). High-quality procedural knowledge is rare
in standard web corpora, and generic retrieval often fails to surface the right subroutine when it is needed.
Toaddressthislimitation, weintroduceReasoningMemory, aRAGframeworkthatreusesproceduralknowledge
at scale for reasoning models. As demonstrated in Figure 1, rather than indexing generic documents, Reasoning
Memory builds a procedural datastore from existing reasoning trajectories. We segment trajectories into
atomic subquestion descriptions paired with concise high-level subroutines, yielding a datastore of over 32M
items. At inference time, a lightweight in-thought query prompt encourages the model to verbalize its current
subquestion as a short query. We retrieve relevant subroutines and inject them into the reasoning trace as
implicit procedural priors. We then sample multiple reasoning trajectories under diverse retrieved procedures
and apply a simple length-based uncertainty heuristic to filter both subroutines and candidate solutions before
selecting the final answer. The result is a simple test-time scaling procedure in which additional inference
budget enables broader exploration under diverse retrieved procedures, leading to stronger performance as
compute increases.
Empirically, Reasoning Memory delivers consistent gains across open-weight reasoning models, inference
budgets, andfrontierbenchmarksinmath, science, andcoding. Itoutperformsdocument-levelRAG,trajectory-
and template-based procedural RAG, and compute-matched test-time scaling without retrieval. Overall, it
improves accuracy by up to 19.2% over no retrieval and by up to 7.9% over the strongest compute-matched
baseline, with gains observed across task types. Our analyses further show that performance improves with
larger and more diverse procedural datastores, and under a fixed inference budget, it is more effective to
explore multiple trajectories under a small set of high-quality retrieved subroutines than to spend the same
compute under a single subroutine. Additional ablations on datastore construction and in-thought querying
identify the contribution of our design.
2 Related Work
Test-time Scaling for ReasoningTest-time scaling allocates additional inference-time compute per instance
to improve performance on challenging math, science, and coding tasks (Wei et al., 2022; Chen et al., 2024;
DeepSeek-AI, 2024). Starting from Chain-of-Thought prompting (Wei et al., 2022), prior work has explored two
broad paradigms.Sequential scalingextends a single trajectory through planning, reflection, self-correction,
and backtracking (Madaan et al., 2023; Chen et al., 2024; Lee et al., 2025; Muennighoff et al., 2025; Xu et al.,
2025), whileparallel scalingsamples multiple candidates and combines them through selection or aggregation
(Wang et al., 2023; Yao et al., 2023; Snell et al., 2024). Recent reasoning models explicitly trained to produce
long thinking prefixes, including OpenAI’s o1 series (OpenAI, 2024), DeepSeek R1, and QwQ (DeepSeek-AI,
2025; Qwen Team, 2024), exhibit strong inference-time scaling behavior, and simple budget forcing can already
2

yield strong gains on math and science benchmarks (Muennighoff et al., 2025). Another line of work combines
parallel scaling with search, including process reward models and Monte Carlo Tree Search style methods
(Wang et al., 2024a; Gao et al., 2024; Park et al., 2025). Our work targets reasoning models and augments
test-time scaling with a procedural knowledge datastore derived from prior reasoning trajectories. It supports
sequential scaling by injecting retrieved procedures into the thinking process and parallel scaling by exploring
different retrieved subroutines.
Memorizing Procedural KnowledgeA complementary line of work studies how to represent and reuse
procedural knowledge such as workflows, strategies, or templates distilled from experience. Agent Workflow
Memory (Wang et al., 2024b) induces reusable workflows from web navigation traces, while ReasoningBank
(Ouyang et al., 2025a) converts successful and failed episodes into compact strategies and supports memory-
aware test-time scaling. Outside web agents, Buffer of Thoughts (Yang et al., 2024b) distills high-level thought
templates from problem-solving traces and retrieves them for new instances. Related approaches, including
Self-Discover (Zhou et al., 2024), ReasonFlux (Yang et al., 2025b), and RLAD (Qu et al., 2025), similarly
operate over libraries of heuristics or natural-language abstractions to improve reasoning. These results show
that explicit procedural memory can help, but existing methods often rely on relatively small libraries, focus
on instruction-following models, or target agentic settings. In contrast, we build a large-scale procedural
knowledge datastore and inject retrieved procedures directly into the thinking stream of reasoning models on
challenging math, science, and coding problems.
RAG for ReasoningRetrieval-augmented generation (RAG) equips language models with external knowledge
through retrieval and in-context conditioning (Guu et al., 2020; Lewis et al., 2020; Shi et al., 2024). While
most RAG systems are developed for factuality-related tasks, a smaller line of work studies RAG for reasoning-
intensive settings. CompactDS builds a compact and diverse web-scale datastore and applies standard
retrieval to improve performance on multiple reasoning benchmarks (Lyu et al., 2025). ReasonIR trains a
retriever specialized for reasoning-intensive retrieval (Shao et al., 2025). These approaches mainly retrieve
general documents and are typically evaluated with instruction-tuned models. Agentic search systems such
as Search-o1 (Li et al., 2025a) and R1-Searcher (Song et al., 2025) instead let models issue queries during
reasoning and incorporate retrieved documents. However, these methods primarily use retrieval to supply
factual knowledge, rather than to surface reusable procedural knowledge that directly guides the model’s
thinking process.
3 Reasoning Memory: Approach
We introduceReasoning Memory, a RAG framework that induces, retrieves, and reusesproceduralknowledge at
scale for reasoning models. As shown in Figure 1, we first distill public reasoning trajectories into a datastore
ofsubquestion-subroutinepairs (§3.1). At inference time, the model verbalizes its current subquestion as a
compact query, and the retrieved procedures are injected directly into its thinking stream (§3.2). We then use
multiple retrieved subroutines as implicit priors for test-time scaling (§3.3).
3.1 Procedural Knowledge Datastore
Problem-solving trajectories of current reasoning models contain rich procedural knowledge, but they are
too long and noisy to reuse directly. Reasoning Memory therefore converts them offline into aprocedural
knowledge datastoreof natural-language subquestions and concise solution subroutines, which supports both
effective retrieval and efficient in-context use.
Formally, let a trajectory instance be(q ,a,r), whereqis the original problem,ais the final answer, andr
is the intermediate reasoning trace. We map(q ,r)to a set of Ksubquestion-subroutine pairs {(qi, si)}K
i=1
using a two-step prompting pipeline. First, we derive self-contained subquestions {qi}that capture the key
intermediate goals in the trajectory. Then, for each qi, we generate a concise subroutine sithat summarizes the
high-level procedure used to address it, while abstracting away local calculations and incidental trial-and-error.
This yields a reusable representation that remains short enough to serve as an in-context procedural prior. A
concrete example is shown in Figure 10 in the appendix.
3

Rather than collecting trajectories from scratch, we build the datastore from the publicly released Nemotron
V1 corpus (Nathawani et al., 2025), which covers diverse math, science, and coding questions. The resulting
datastore contains approximately 32 million( qi, si)pairs. On average, subquestions contain 19.2 tokens,
subroutines 207.9 tokens, and each trajectory contributes 10.5 subquestions. The full datastore construction
details are presented in Section B.1. QwQ-32B is used for both prompting steps1.
3.2 In-Thought Active Retrieval Augmentation
Given a large datastore of self-contained problem-procedure pairs, the next challenge is querying it effectively
to benefit a reasoning model’s thinking process. To begin with, prompting the model to follow certain query
generation formats within its thoughts is challenging. On the other hand, approaches such as Search-o1
(Li et al., 2025a) that defer query generation until extensive reasoning is expensive and is better suited
factuality-related gaps than reasoning strategies which require early interventions to model thinking.
Inspired by Jiang et al. (2023), we instead leverage a simplethought-hijackingprompt to enable query
generation part of the thinking stream itself. Concretely, we start the model’s thinking with the following
meta-thinking sentence:
Now, let me search for a similar basic problem whose solution
can help unblock me for solving the current step. Let me
frame it as a more high-level google search query:
We extract the next sentence in the continuation as the retrieval query ˜q. In practice, this produces
concise subquestion descriptions that align well with datastore subquestions. Given ˜q, we retrieve the top- k
subquestions and their corresponding subroutines, denoted by {(ˆqj,ˆsj)}k
j=1, using a standard dense retriever.
For each retrieved pair, we insert the following hint directly into the reasoning stream:
[hint] Here is a problem solving procedure for a related
question“ˆq j":ˆsj[end of hint]
This hint is followed by a simple continuation cue, “ Okay,”. The model then continues reasoning conditioned
on the retrieved procedural prior. Because both the query and the retrieved subroutine appear in the same
thinking channel, retrieval acts as a lightweight extension of the model’s ongoing reasoning.
3.3 Inference-Time Scaling with Reasoning Memory
Finally, Reasoning Memory introduces a test-time scaling method that naturally adapts to different compute
budgets (in terms of samples) by adjusting the diversity of retrieved procedures and intensity of per-procedure
compute. Given a sampling budget of at most msamples and top- kretrieved subroutines {(ˆqj,ˆsj)}k
j=1, we
allocate ⌊m/k⌋samples to each retrieved subroutine using the retrieval-augmented prompt from §3.2. This
yields a pool of candidate trajectories {πj,ℓ}, where j∈ { 1, . . . , k}indexes the retrieved subroutine and
ℓ∈ {1, . . . ,⌊m/k⌋}indexes samples under that subroutine.
We then score each trajectory πj,ℓwith an uncertainty measure rj,ℓand normalize scores across the full pool:
˜rj,ℓ≡max j′,ℓ′rj′,ℓ′−rj,ℓ
max j′,ℓ′rj′,ℓ′−min j′,ℓ′rj′,ℓ′.
For each retrieved subroutine, we compute an average quality score
¯rj≡1
⌊m/k⌋⌊m/k⌋X
ℓ=1˜rj,ℓ.
We retain subroutines with ¯rj> τ, rank their associated trajectories by confidence, and keep the top n
samples for pass@1-based evaluation.
Inspired by Hassid et al. (2025), we use the thinking length as the default uncertainty score. The intuition is
that uncertain reasoning tends to produce longer traces due to extra branching and backtracking. Concretely,
1Smaller models are also effective, as most of the knowledge is already in the trajectories (Table 2).
4

we set rj,ℓto the generated trajectory length in words, so that after normalization, larger ˜rj,ℓindicates a
shorter and potentially more confident trajectory relative to the rest of the pool. The subroutine score ¯rj
therefore measures whether a retrieved subroutine tends to induce shorter samples. By design, the pipeline
is compatible with other uncertainty signals, such as log-likelihood, entropy, or self-evaluated relevance.
Nevertheless, we find length to be a strong and simple choice among these alternatives and use it for the main
results. We provide a full comparison in §C.3.
4 Experimental Setup
ModelsWe focus on open-weight reasoning models that expose their thinking tokens. Our main evaluation
uses three models spanning different sizes, base model families, and post-training recipes: DeepSeek-R1-Distill-
Llama-8B (DeepSeek-AI, 2025), OpenThinker3-7B (Guha et al., 2025), and Qwen3-32B (Yang et al., 2025a).
Each model is used in the reasoning mode without additional fine-tuning.
Benchmarks and MetricsWe evaluate on challenging math, science, and coding benchmarks. For math,
we use AIME 2024 and AIME 2025 (30 problems each; Mathematical Association of America, 2024), and
MATH500 (Lightman et al., 2024), a standard subset of the MATH test set (Hendrycks et al., 2021). For
science, we use GPQA-Diamond (GPQA-D) (Rein et al., 2023). For coding, we use LiveCodeBench (LCB)
(Jain et al., 2025); following the filtering criteria of Li et al. (2025a), we evaluate on 112 problems from releases
V1–V4 and 109 problems from releases V5–V6. We use standard metrics from prior work: math-equal for
AIME and MATH500, exact match for GPQA-D, and execution-based pass@1 for LiveCodeBench, using the
implementations of Li et al. (2025a). Unless otherwise specified, we sample m= 8trajectories per problem and
report the averaged performance (equivalent to pass@1). For methods with uncertainty-based selection,the
performance is averaged over the selected nsamples after filtering. Our main setting uses( m, n) = (8 ,4). We
also evaluate(m, n) = (30,8)and analyze larger budgets up tom= 100in §5.3.
BaselinesWe compare Reasoning Memory against baselines covering RAG with trajectory-level knowledge,
template-based procedural knowledge, document-level factual knowledge, as well as retrieval-free uncertainty-
based test-time scaling.
•No RAG.The model solves each problem without retrieval.
•Trajectory RAG.We retrieve full reasoning trajectories from the Nemotron v1 corpus using the question as
the query, and either (1) prepend the top 3 trajectories as long-form exemplars (prefix) or (2) compress
the top trajectory into a short summary and prepend it (summary). This baseline uses the same source
corpus as Reasoning Memory, but operates at the trajectory level rather than the subroutine level.
•Template RAG.We instantiate RAG over small human-designed or automatically distilled template
libraries from prior work. TheReasonFluxvariant uses templates from Yang et al. (2025c), and the
Self-Discovervariant uses templates from Zhou et al. (2024). For fairness, all retrieved reasoning
templates are injected into the thinking stream in the same way as in Reasoning Memory.
•Document RAG.We include two factual knowledge retrieval baselines: retrieving fromCompactDS
following Lyu et al. (2025), and retrieving from Google2. In both cases, the original question is used as
the query and the retrieved passages are prepended to the question in the prompt, following Li et al.
(2025a).
•Length Scaling.A retrieval-free test-time scaling baseline inspired by Hassid et al. (2025). We sample m
independent trajectories and select nhigh-confidence ones using the length-based uncertainty heuristic
from §3.3. This isolates the benefit of procedural retrieval3. We evaluate(m, n) = (8,4)and(30,8).
•Reasoning Memory.Our full method retrieves kprocedural subroutines from the Nemotron-based
datastore, injects them in thought using the thought-hijacking prompt, and performs two-stage length-
2We set the search date cutoff to 2023/12/31 to prevent data contamination.
3In the main text, we do not apply length scaling to the other baselines, since this is not standard practice. When added, the
relative performance trends remain similar. Results are reported in §C.1.
5

AIME 2024 GPQA-D LCB0.00.20.40.6AccuracyBase Model: Llama-3.1-8B
AIME 2024 GPQA-D LCBBase Model: Qwen2.5-7B
AIME 2024 GPQA-D LCBBase Model: Qwen2.5-32B
Instruction Version, No Retrieval
Instruction Version, CompactDS
Reasoning Version, No Retrieval
Reasoning Version, CompactDSFigure 2 Standard document RAG benefits instruction-tuned models more than reasoning models.Under the CompactDS
pipeline, instruction-tuned models obtain modest gains from retrieval, whereas the corresponding reasoning models
often see limited gains or even degradation, despite much stronger no-retrieval performance.
based filtering over mtrajectories. We use τ= 1and evaluate( m, n, k )=(8 ,4,3)or(30 ,8,10), carefully
matching the sampling budget of the Length Scaling baseline.
For all retrieval-based methods, we use ReasonIR-8B (Shao et al., 2025) as the retriever. Full implementation
details are provided in §B.
5 Results
5.1 Standard Document RAG is Poorly Aligned with Reasoning Models
We begin with a pilot study to test whether a standard document-level RAG pipeline benefits reasoning models
in the same way it has been shown to benefit instruction-tuned models. CompactDS reported that a web-scale
general document datastore can improve instruction-tuned models on challenging reasoning benchmarks, but
did not evaluate this setup on reasoning models (Lyu et al., 2025). We therefore follow the CompactDS
setup and compare paired instruction-tuned and reasoning models from the same families on AIME 2024,
GPQA-Diamond, and LiveCodeBench. Full setup details are presented in Section A.
Figure 2 shows a consistent pattern across model families and tasks: instruction-tuned models benefit
modestly from retrieved documents, while the corresponding reasoning models often gain little or even degrade.
This suggests that standard document-level RAG is poorly aligned with reasoning models, whose current
subquestions often require procedural guidance rather than generic background context. In Section A.2, we
further diagnose this phenomenon using synthesized knowledge. The results show that reducing retrieval
noise makes injected knowledge more effective, and that thetypeof knowledge also matters: procedural
guidance yields larger gains than factual knowledge for reasoning models. This motivates our focus on compact
procedural subroutines rather than general documents.
5.2 Main Results
Table 1 summarizes the end-to-end accuracy of Reasoning Memory and all baselines. Most baselines fail to
consistently outperform No RAG on frontier reasoning tasks. Averaged over all tasks, only Google-based
Document RAG improves over No RAG for two models: DeepSeek-R1-Distill-Llama-8B (0.422 vs. 0.413)
and Qwen3-32B (0.658 vs. 0.634). In contrast, Reasoning Memory consistently improves over No RAG by a
substantial margin, reaching 0.407 versus 0.357 when averaged across all models and tasks. This shows that,
under a simple RAG pipeline, reasoning models benefit much more from large-scale procedural retrieval than
from web documents, full trajectories, or small template libraries.
Reasoning Memory also remains stronger than the retrieval-free Length Scaling baseline in most settings,
winning 31 out of 36 comparisons across models, tasks, and the two budget settings m∈ { 8,30}. Increasing
the inference budget further improves performance. At m= 30, Reasoning Memory improves over No RAG
by 0.12 (19.0%) on math, 0.05 (9.8%) on science, and 0.09 (28.9%) on coding, averaged across models. These
gains support procedural knowledge retrieval as an effective way to improve test-time scaling for reasoning
models. We next examine this effect in more detail and show that the gap remains consistent as the inference
6

Method Ret.mMath Science Code (LCB)
AIME 24’ AIME 25’ MATH500 GPQA-D V1-4 V5-6
DeepSeek-R1-Distill-Llama-8B
No RAG N/A 8 0.461 0.350 0.806 0.427 0.220 0.211
Trajectory RAG (Prefix) Proc. 8 0.425 0.246 0.784 0.391 0.195 0.128
Trajectory RAG (Summary) Proc. 8 0.408 0.321 0.822 0.431 0.184 0.121
Template RAG (ReasonFlux) Proc. 8 0.450 0.342 0.828 0.417 0.174 0.210
Template RAG (Self-Discover) Proc. 8 0.442 0.296 0.821 0.414 0.213 0.247
Document RAG (CompactDS) Fact. 8 0.356 0.312 0.818 0.319 0.208 0.172
Document RAG (Google) Fact. 8 0.483 0.3510.8380.383 0.246 0.228
Length Scaling N/A 8 0.508 0.325 0.815 0.426 0.255 0.278
Reasoning Memory Proc. 80.511 0.364 0.838 0.444 0.278✣0.292✣
Length Scaling N/A 30 0.548 0.358 0.802 0.447 0.282 0.302
Reasoning Memory Proc. 300.575 0.392✣0.836✣0.461 0.310✣0.325✣
OpenThinker3-7B
No RAG N/A 8 0.470 0.392 0.841 0.438 0.318 0.266
Trajectory RAG (Prefix) Proc. 8 0.429 0.358 0.828 0.437 0.242 0.249
Trajectory RAG (Summary) Proc. 8 0.438 0.354 0.855 0.451 0.242 0.260
Template RAG (ReasonFlux) Proc. 8 0.467 0.354 0.830 0.457 0.262 0.268
Template RAG (Self-Discover) Proc. 8 0.442 0.392 0.825 0.441 0.276 0.279
Document RAG (CompactDS) Fact. 8 0.454 0.333 0.846 0.458 0.292 0.258
Document RAG (Google) Fact. 8 0.501 0.422 0.864 0.452 0.235 0.231
Length Scaling N/A 8 0.612 0.506 0.876 0.484 0.333 0.344
Reasoning Memory Proc. 80.725✣0.646✣0.908✣0.522✣0.362✣0.397✣
Length Scaling N/A 30 0.647 0.528 0.873 0.502 0.345 0.343
Reasoning Memory Proc. 300.758✣0.679✣0.911✣0.542✣0.381✣0.412✣
Qwen3-32B
No RAG N/A 8 0.789 0.586 0.905 0.669 0.387 0.465
Trajectory RAG (Prefix) Proc. 8 0.683 0.408 0.895 0.607 0.249 0.432
Trajectory RAG (Summary) Proc. 8 0.629 0.429 0.895 0.595 0.267 0.418
Template RAG (ReasonFlux) Proc. 8 0.654 0.500 0.901 0.644 0.300 0.278
Template RAG (Self-Discover) Proc. 8 0.667 0.483 0.903 0.633 0.312 0.339
Document RAG (CompactDS) Fact. 8 0.604 0.388 0.899 0.602 0.267 0.284
Document RAG (Google) Fact. 8 0.7950.7670.919 0.678 0.382 0.426
Length Scaling N/A 8 0.792 0.591 0.9030.6990.399 0.444
Reasoning Memory Proc. 80.825✣0.767 0.926✣0.6740.462✣0.503✣
Length Scaling N/A 30 0.812 0.619 0.9080.6820.405 0.476
Reasoning Memory Proc. 300.838 0.754✣0.923✣0.6810.471✣0.508✣
Table 1 Main results on frontier reasoning benchmarks.Accuracy of all methods on math (AIME 2024, AIME 2025,
MATH500), science (GPQA-D), and coding (LiveCodeBench V1–4 and V5–6). The columnRet.= type of retrieval,
Proc. = procedural, Fact. = factual. Bold numbers mark the best result under the same budget. ✣= statistically
significantly better than the second best via paired t-test (p < 0.05).
budget increases and more samples are drawn. In the later sections, we also analyze the datastore composition
and key design ablations in detail. Finally, more qualitative examples are provided in §C.4.
5.3 Scaling Behavior with Larger Inference Budgets
We next study whether Reasoning Memory continues to help as the inference budget increases, and how the
budget should be allocated across retrieved subroutines. Using DeepSeek-R1-Distill-Llama-8B, we evaluate
m∈ { 20,40,60,80,100}under three strategies: (1)Length Scaling, which samples mtrajectories without
retrieval; (2)Reasoning Memory (Intensity-First), which allocates 20 samples to each retrieved subroutine
and uses only as subset top-ranked subroutines to stay within the budget limit; and (3)Reasoning Memory
(Diversity-First), which always uses the top 20 retrieved subroutines and allocates the maximum number of
samples per subroutine within the budget limit.
As shown in Figure 3, the diversity-first strategy yields the strongest scaling behavior. On AIME 2024
and LiveCodeBench (V1–V4), it improves monotonically with budget and achieves the best performance at
m= 100, clearly outperforming both Length Scaling and intensity-first Reasoning Memory. On GPQA-D,
7

20 40 60 80 100
Budget0.5200.5400.5600.5800.6000.620Performance
AIME 2024
20 40 60 80 100
Budget0.4350.4400.4450.4500.455Performance
GPQA Diamond
20 40 60 80 100
Budget0.2900.3000.310Performance
LiveCodeBench
Scaling without Retrieval Reasoning Memory (Intensity-First) Reasoning Memory (Diversity-First)Figure 3 Performance as a function of inference budget.We compare Length Scaling without retrieval with two Reasoning
Memory variants on DeepSeek-R1-Distill-Llama-8B as the total sampling budgetmincreases.
0 1 2 3
Datastore Size 1e70.5000.5200.5400.560Performance
AIME 2024
0 1 2 3
Datastore Size 1e70.4300.4400.4500.460Performance
GPQA Diamond
0 1 2 3
Datastore Size 1e70.3050.3100.315Performance
LiveCodeBench
Nemotron 10 100%
 Nemotron Math only Nemotron Code only Nemotron Science only OpenThoughts 3
Figure 4 Effect of datastore size and composition on Reasoning Memory.Performance of DeepSeek-R1-Distill-Llama-8B
with budgetm= 30. Larger and more diverse datastores generally yield stronger performance.
all methods show weaker returns from additional budget, but diversity-first Reasoning Memory remains
competitive with Length Scaling, and its best result is still achieved at a lower budget4. Overall, these
results highlight the importance of a diversity-first budget allocation strategy and confirms the advantage of
Reasoning Memory at larger budgets.
5.4 Impact of Datastore Composition
We next study how the size and composition of the procedural datastore affect Reasoning Memory. Using
DeepSeek-R1-Distill-Llama-8B with m= 30, we compare three classes of datastore variants: (1) random
subsets of Nemotron ranging from 10% to 100% trajectories, (2) domain-specific Nemotron subsets containing
only math, code, or science trajectories, and (3) a datastore built from OpenThoughts3, which is more
math-focused.
Figure 4 shows that performance generally improves as we scale the mixed Nemotron datastore. The full
datastore gives the best accuracy on AIME 2024 and GPQA-D, while LiveCodeBench is less sensitive once
the datastore reaches roughly 25% of full scale. Among domain-specific datastores, OpenThoughts3 is highly
competitive on AIME 2024, matches similarly sized Nemotron subsets on LiveCodeBench, but underperforms
clearly on GPQA-D. Within Nemotron, science-only trajectories give the best science performance on GPQA-D,
while code-only trajectories are surprisingly more effective than math-only trajectories across all three tasks
despite having similar size. For both math and coding, however, all in-domain subsets underperform the
full mixed datastore, suggesting that an important source of improvement is the cross-domain transfer of
procedural knowledge.
4In fact, in Table 1, the main setting reaches an even higher GPQA-D score of 0.461 withm= 30.
8

Setting AIME 2024 GPQA-D LCB (V1-4) Average
Removing Subroutine Decomposition 0.551 0.433 0.308 0.431
Removing Self-Generated Query 0.567 0.453 0.295 0.438
Full Reasoning Memory (Qwen3-8B) 0.5500.474 0.3200.448
Full Reasoning Memory (QwQ-32B)0.5750.461 0.3100.449
Table 2 Ablations on datastore and query design.All results use DeepSeek-R1-Distill-Llama-8B with budget m= 30. The
full Reasoning Memory configuration uses QwQ-32B as the datastore generator, a decomposed datastore indexed by
subquestions, and self-verbalized in-thought queries.
5.5 Ablations: Subroutine Decomposition, Query Generation, and Alternative Models
We ablate three key design choices of Reasoning Memory using DeepSeek-R1-Distill-Llama-8B withm= 30.
First, we remove the subquestion decomposition and instead index the datastore by the original question,
with a single subroutine summarizing the full trajectory. Second, we keep the decomposed datastore but
disable self-verbalized retrieval queries, using the original question directly as the query. Third, we replace
QwQ-32B with the smaller Qwen3-8B for the datastore generation. Finally, uncertainty criteria is another
important design choice to study and we report the findings in §C.3 due to space constraints.
As shown in Table 2, Reasoning Memory is robust to the choice of the datastore generator model, but both
decomposition and self-verbalized queries matter. Using Qwen3-8B yields performance close to the full system,
and even slightly improves GPQA-D and LiveCodeBench, suggesting that much of the useful procedural
knowledge already resides in the source trajectories. In contrast, removing subroutine decomposition causes
the largest drop on AIME and GPQA-D, indicating that fine-grained decomposition is important for building
a diverse datastore that matches different subquestion granularities. Replacing self-verbalized queries with the
original question also leads to consistent degradation across all three benchmarks. The largest drop is seen on
LiveCodeBench, where problem statements are often long and cluttered with context, format specifications,
and examples. Overall, these ablations support the core design choices of Reasoning Memory while suggesting
that datastore construction can be performed with smaller models at little cost.
6 Conclusion
Reasoning Memory shows that test-time compute for reasoning models can be spent not only on producing
longer chains of thought, but also on reusing prior problem-solving experience in a structured way. By
surfacing procedural knowledge from existing reasoning trajectories and turning it into an explicit datastore,
our approach transforms instruction-tuning corpora into an inference-time resource for guiding new problem-
solving episodes. We further introduce a simple recipe for aligning retrieval with the model’s own thinking
stream and for scaling inference with multiple retrieved priors. Empirically, Reasoning Memory improves
diverse open-weight reasoning models on math, science, and coding tasks, scales well with both datastore
size and inference budget, and remains robust across several implementation choices. More broadly, our
results suggest a promising direction for both the reasoning and RAG communities: building systems that
can accumulate, retrieve, and reuse procedural knowledge over time.
References
Steven Bird and Edward Loper. NLTK: The natural language toolkit. InProceedings of the ACL Interactive Poster
and Demonstration Sessions, pages 214–217, Barcelona, Spain, July 2004. Association for Computational Linguistics.
https://aclanthology.org/P04-3031/.
Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In
The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024.
OpenReview.net, 2024.https://openreview.net/forum?id=KuPixIqPiq.
DeepSeek-AI. Deepseek-v3 technical report.CoRR, abs/2412.19437, 2024. doi: 10.48550/ARXIV.2412.19437.
https://doi.org/10.48550/arXiv.2412.19437.
9

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.CoRR, abs/2501.12948,
2025. doi: 10.48550/ARXIV.2501.12948.https://doi.org/10.48550/arXiv.2501.12948.
Zitian Gao, Boye Niu, Xuzheng He, Haotian Xu, Hongzhang Liu, Aiwei Liu, Xuming Hu, and Lijie Wen. Interpretable
contrastive monte carlo tree search reasoning.CoRR, abs/2410.01707, 2024. doi: 10.48550/ARXIV.2410.01707.
https://doi.org/10.48550/arXiv.2410.01707.
Etash Kumar Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina,
Jean Mercat, Trung Vu, Zayne Sprague, Ashima Suvarna, Benjamin Feuer, Liangyu Chen, Zaid Khan, Eric Frankel,
Sachin Grover, Caroline Choi, Niklas Muennighoff, Shiye Su, Wanjia Zhao, John Yang, Shreyas Pimpalgaonkar,
Kartik Sharma, Charlie Cheng-Jie Ji, Yichuan Deng, Sarah M. Pratt, Vivek Ramanujan, Jon Saad-Falcon, Jeffrey
Li, Achal Dave, Alon Albalak, Kushal Arora, Blake Wulfe, Chinmay Hegde, Greg Durrett, Sewoong Oh, Mohit
Bansal, Saadia Gabriel, Aditya Grover, Kai-Wei Chang, Vaishaal Shankar, Aaron Gokaslan, Mike A. Merrill,
Tatsunori Hashimoto, Yejin Choi, Jenia Jitsev, Reinhard Heckel, Maheswaran Sathiamoorthy, Alexandros G.
Dimakis, and Ludwig Schmidt. Openthoughts: Data recipes for reasoning models.CoRR, abs/2506.04178, 2025.
doi: 10.48550/ARXIV.2506.04178.https://doi.org/10.48550/arXiv.2506.04178.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: retrieval-augmented language
model pre-training.CoRR, abs/2002.08909, 2020.https://arxiv.org/abs/2002.08909.
Michael Hassid, Gabriel Synnaeve, Yossi Adi, and Roy Schwartz. Don’t overthink it. preferring shorter thinking
chains for improved LLM reasoning.CoRR, abs/2505.17813, 2025. doi: 10.48550/ARXIV.2505.17813. https:
//doi.org/10.48550/arXiv.2505.17813.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob
Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Sai-Kit
Yeung, editors,Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1,
NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021. https://datasets-benchmarks-proceedings.
neurips.cc/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html.
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama,
Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models
for code. InThe Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April
24-28, 2025. OpenReview.net, 2025.https://openreview.net/forum?id=chfJJYC3iL.
Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan,
and Graham Neubig. Active retrieval augmented generation. In Houda Bouamor, Juan Pino, and Kalika Bali,
editors,Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP
2023, Singapore, December 6-10, 2023, pages 7969–7992. Association for Computational Linguistics, 2023. doi:
10.18653/V1/2023.EMNLP-MAIN.495.https://doi.org/10.18653/v1/2023.emnlp-main.495.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang,
and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Jason
Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace, editors,Proceedings of the 29th
Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pages 611–626.
ACM, 2023. doi: 10.1145/3600006.3613165.https://doi.org/10.1145/3600006.3613165.
Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, and Xinyun
Chen. Evolving deeper LLM thinking.CoRR, abs/2501.09891, 2025. doi: 10.48550/ARXIV.2501.09891. https:
//doi.org/10.48550/arXiv.2501.09891.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation
for knowledge-intensive NLP tasks. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina
Balcan, and Hsuan-Tien Lin, editors,Advances in Neural Information Processing Systems 33: Annual Conference
on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. https:
//proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html.
XiaoxiLi, GuantingDong, JiajieJin, YuyaoZhang, YujiaZhou, YutaoZhu, PeitianZhang, andZhichengDou. Search-o1:
Agentic search-enhanced large reasoning models.CoRR, abs/2501.05366, 2025a. doi: 10.48550/ARXIV.2501.05366.
https://doi.org/10.48550/arXiv.2501.05366.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou.
Webthinker: Empowering large reasoning models with deep research capability.CoRR, abs/2504.21776, 2025b. doi:
10.48550/ARXIV.2504.21776.https://doi.org/10.48550/arXiv.2504.21776.
10

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John
Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. InThe Twelfth International Conference
on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. https://
openreview.net/forum?id=v8L0pN6EOi.
Llama Team. The llama 3 herd of models.CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783.
https://doi.org/10.48550/arXiv.2407.21783.
Xinxi Lyu, Michael Duan, Rulin Shao, Pang Wei Koh, and Sewon Min. Frustratingly simple retrieval improves
challenging, reasoning-intensive benchmarks.CoRR, abs/2507.01297, 2025. doi: 10.48550/ARXIV.2507.01297.
https://doi.org/10.48550/arXiv.2507.01297.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri,
Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean
Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. In Alice Oh,
Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors,Advances in Neural
Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS
2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. http://papers.nips.cc/paper_files/paper/2023/hash/
91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html.
Mathematical Association of America. American Invitational Mathematics Examination (AIME). https://maa.org/
math-competitions/american-invitational-mathematics-examination-aime , February 2024. American Invitational
Mathematics Examination (AIME) 2024.
Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy
Liang, Emmanuel J. Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling.CoRR, abs/2501.19393, 2025.
doi: 10.48550/ARXIV.2501.19393.https://doi.org/10.48550/arXiv.2501.19393.
Dhruv Nathawani, Igor Gitman, Somshubra Majumdar, Evelina Bakhturina, Ameya Sunil Mahabaleshwarkar, , Jian
Zhang, and Jane Polak Scowcroft. Nemotron-Post-Training-Dataset-v1, 2025. https://huggingface.co/datasets/
nvidia/Nemotron-Post-Training-Dataset-v1.
OpenAI. Openai o1 system card and Learning to reason with LLMs. https://arxiv.org/abs/2412.16720 andhttps:
//openai.com/index/learning-to-reason-with-llms , 2024. Technical reports accompanying the OpenAI o1 and
o1-mini reasoning models.
OpenAI. Introducing GPT-5.2. https://openai.com/index/introducing-gpt-5-2/ , December 2025. Accessed: 2025-12-
21.
Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki,
Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and
Tomas Pfister. Reasoningbank: Scaling agent self-evolving with reasoning memory.CoRR, abs/2509.25140, 2025a.
doi: 10.48550/ARXIV.2509.25140.https://doi.org/10.48550/arXiv.2509.25140.
Siru Ouyang, Xinyu Zhu, Zilin Xiao, Minhao Jiang, Yu Meng, and Jiawei Han. RAST: reasoning activation
in llms via small-model transfer.CoRR, abs/2506.15710, 2025b. doi: 10.48550/ARXIV.2506.15710. https:
//doi.org/10.48550/arXiv.2506.15710.
Sungjin Park, Xiao Liu, Yeyun Gong, and Edward Choi. Ensembling large language models with process reward-guided
tree search for better complex reasoning. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors,Proceedings of the 2025
Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language
Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025,
pages 10256–10277. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.NAACL-LONG.515.
https://doi.org/10.18653/v1/2025.naacl-long.515.
Yuxiao Qu, Anikait Singh, Yoonho Lee, Amrith Setlur, Ruslan Salakhutdinov, Chelsea Finn, and Aviral Kumar.
RLAD: training llms to discover abstractions for solving reasoning problems.CoRR, abs/2510.02263, 2025. doi:
10.48550/ARXIV.2510.02263.https://doi.org/10.48550/arXiv.2510.02263.
Qwen Team. Qwq: A family of open reasoning models. https://qwenlm.github.io/blog/qwq-32b-preview/ , 2024.
Technical report and model card for the QwQ reasoning model family.
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael,
and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark.CoRR, abs/2311.12022, 2023. doi:
10.48550/ARXIV.2311.12022.https://doi.org/10.48550/arXiv.2311.12022.
11

Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighoff, Xi Victoria Lin, Daniela Rus, Bryan Kian Hsiang Low,
Sewon Min, Wen-tau Yih, Pang Wei Koh, and Luke Zettlemoyer. Reasonir: Training retrievers for reasoning tasks.
CoRR, abs/2504.20595, 2025. doi: 10.48550/ARXIV.2504.20595.https://doi.org/10.48550/arXiv.2504.20595.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and
Wen-tau Yih. REPLUG: retrieval-augmented black-box language models. In Kevin Duh, Helena Gómez-Adorno,
and Steven Bethard, editors,Proceedings of the 2024 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico
City, Mexico, June 16-21, 2024, pages 8371–8384. Association for Computational Linguistics, 2024. doi: 10.18653/
V1/2024.NAACL-LONG.463.https://doi.org/10.18653/v1/2024.naacl-long.463.
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be
more effective than scaling model parameters.CoRR, abs/2408.03314, 2024. doi: 10.48550/ARXIV.2408.03314.
https://doi.org/10.48550/arXiv.2408.03314.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen.
R1-searcher: Incentivizing the search capability in llms via reinforcement learning.CoRR, abs/2503.05592, 2025.
doi: 10.48550/ARXIV.2503.05592.https://doi.org/10.48550/arXiv.2503.05592.
Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd:
Verify and reinforce llms step-by-step without human annotations. In Lun-Wei Ku, Andre Martins, and Vivek
Srikumar, editors,Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages9426–9439.AssociationforComputational
Linguistics, 2024a. doi: 10.18653/V1/2024.ACL-LONG.510.https://doi.org/10.18653/v1/2024.acl-long.510.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and
Denny Zhou. Self-consistency improves chain of thought reasoning in language models. InThe Eleventh International
Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.
https://openreview.net/forum?id=1PL1NIMMrw.
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory.CoRR, abs/2409.07429,
2024b. doi: 10.48550/ARXIV.2409.07429.https://doi.org/10.48550/arXiv.2409.07429.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le,
and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Sanmi Koyejo,
S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors,Advances in Neural Information
Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New
Orleans, LA, USA, November 28 - December 9, 2022, 2022. http://papers.nips.cc/paper_files/paper/2022/hash/
9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.
Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, and Caiming Xiong. Scalable chain of thoughts via elastic
reasoning.CoRR, abs/2505.05315, 2025. doi: 10.48550/ARXIV.2505.05315. https://doi.org/10.48550/arXiv.2505.
05315.
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang,
Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men,
Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report.CoRR, abs/2412.15115, 2024a.
doi: 10.48550/ARXIV.2412.15115.https://doi.org/10.48550/arXiv.2412.15115.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang,
Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin
Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui
Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang,
Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun
Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report.CoRR, abs/2505.09388,
2025a. doi: 10.48550/ARXIV.2505.09388.https://doi.org/10.48550/arXiv.2505.09388.
Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E. Gonzalez, and Bin Cui.
Buffer of thoughts: Thought-augmented reasoning with large language models. In Amir Globersons, Lester Mackey,
Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors,Advances in Neural
Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS
12

2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024b. http://papers.nips.cc/paper_files/paper/2024/
hash/cde328b7bf6358f5ebb91fe9c539745e-Abstract-Conference.html.
Ling Yang, Zhaochen Yu, Bin Cui, and Mengdi Wang. Reasonflux: Hierarchical LLM reasoning via scaling thought
templates.CoRR, abs/2502.06772, 2025b. doi: 10.48550/ARXIV.2502.06772. https://doi.org/10.48550/arXiv.2502.
06772.
Ling Yang, Zhaochen Yu, Bin Cui, and Mengdi Wang. Reasonflux: Hierarchical LLM reasoning via scaling thought
templates.CoRR, abs/2502.06772, 2025c. doi: 10.48550/ARXIV.2502.06772. https://doi.org/10.48550/arXiv.2502.
06772.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree
of thoughts: Deliberate problem solving with large language models. In Alice Oh, Tristan Naumann, Amir
Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors,Advances in Neural Information Pro-
cessing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023,
New Orleans, LA, USA, December 10 - 16, 2023, 2023. http://papers.nips.cc/paper_files/paper/2023/hash/
271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html.
Pei Zhou, Jay Pujara, Xiang Ren, Xinyun Chen, Heng-Tze Cheng, Quoc V. Le, Ed H. Chi, Denny Zhou, Swaroop
Mishra, and Huaixiu Steven Zheng. SELF-DISCOVER: large language models self-compose reasoning structures.
In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and
Cheng Zhang, editors,Advances in Neural Information Processing Systems 38: Annual Conference on Neural
Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. http:
//papers.nips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html.
13

A Additional Pilot Study Details
A.1 Standard Document RAG with CompactDS
As a pilot study, we investigate how a standard document-level RAG pipeline affectsinstruction-tuned
models versusreasoningmodels. We follow the CompactDS pipeline of Lyu et al. (2025), which retrieves
general-domain background passages and appends them to the model prompt. For each benchmark question,
we use the question text as the retrieval query, perform the same two-stage retrieval procedure as the original
implementation, and insert the top- kretrieved documents into the prompt using the same formatting and
instruction template. Unless otherwise stated, we setk= 3.
We evaluate paired instruction-tuned and reasoning models from three model families: Llama-3.1-8B (Llama
Team, 2024), Qwen2.5-7B, and Qwen2.5-32B (Yang et al., 2024a). For Llama-3.1-8B and Qwen2.5-7B, we use
DeepSeek-R1 distillations as the reasoning variants (DeepSeek-AI, 2025). For Qwen2.5-32B, we use QwQ-32B
as the reasoning variant (Qwen Team, 2024). We evaluate on AIME 2024 (Mathematical Association of
America, 2024), GPQA-Diamond (Rein et al., 2023), and LiveCodeBench (Jain et al., 2025). For each question,
we generate m= 8independent samples with a maximum output length of 32k tokens and report the average
performance across samples. Following Li et al. (2025a), we use math-equal for AIME, exact match for
GPQA-D, and execution-based pass@1 for LiveCodeBench.
Figure 2 shows a consistent pattern across model families and tasks. Instruction-tuned models operate in a
lower-accuracy regime but obtain modest and fairly reliable gains from retrieved documents, consistent with
the findings of Lyu et al. (2025). In contrast, the corresponding reasoning models start from substantially
stronger no-retrieval performance, yet standard document RAG provides only limited gains and can reduce
accuracy in multiple settings. The negative effect is most visible for smaller reasoning models, but the overall
pattern is consistent across math, science, and coding tasks.
We view this result as evidence of a mismatch between standard document RAG and reasoning models. First,
reasoning models are not explicitly trained to treat retrieved passages as procedural guidance that should
steer an ongoing chain of thought. Second, retrieved web documents are optimized for broad knowledge
coverage rather than for the specific subquestion currently being solved. As a result, even relevant retrievals
may be only weakly aligned with the model’s active reasoning state. This motivates the central design choice
of our method: instead of retrieving general documents, we retrieve compact procedural subroutines that
more directly match the current subquestion.
A.2 Controlled Knowledge Injection: Factual vs. Procedural
The CompactDS result above leaves open an important question: are reasoning models inherently incompatible
with retrieval, or is the issue the form of the retrieved knowledge? To probe this question, we run a controlled
knowledge injection study in which the retrieved content is synthesized to be directly relevant to each evaluation
problem.
For each problem from AIME 2024, GPQA-Diamond, and LiveCodeBench, we prompt a strong model (GPT-
5.2 via API, OpenAI (2025)) to synthesize a short paragraph that is helpful for solving the problem while
satisfying two constraints: (1) it must not reveal the final answer, and (2) it must not provide intermediate
computations or step-by-step derivations that could leak the answer. We synthesize two types of content:
(1)factual knowledge, including definitions, formulas, and theorems, and (2)procedural knowledge, including
high-level decomposition strategies, solution plans, and verification heuristics. The prompts are shown in
Figures 6 and 7. The resulting paragraph is typically 100–200 tokens long and is inserted into the same
prompt location used for CompactDS documents.
Figure 5 reports performance deltas relative to the no-retrieval setting. Both instruction-tuned and reasoning
models improve when given synthesized knowledge, which suggests that retrieval itself is not the core issue.
Instead, the usefulness of retrieval depends strongly on the form and alignment of the injected content. In
particular, reasoning models benefit much more from these targeted knowledge paragraphs than from retrieved
CompactDS documents, even though both are inserted through the same prompt template. This supports the
view that reasoning models are sensitive to the relevance and specificity of the injected context.
14

Inst. Reas. Inst. Reas. Inst. Reas.0.000.050.10Performance gain
Llama 3.1 8B Qwen2.5 7B Qwen2.5 32BFactual Knowledge
Procedural KnowledgeFigure 5 Utility of different types of synthesized knowledge.We report performance gains relative to the no-retrieval
setting. Both factual and procedural knowledge help, but procedural knowledge yields larger gains across tasks and
model families.
Procedural knowledge is consistently more helpful than factual knowledge across tasks and model families. We
hypothesize that this is because many difficult reasoning problems are bottlenecked less by missing background
facts than by missing guidance on how to decompose the problem, choose a strategy, or verify an intermediate
direction. This pattern is especially natural for AIME and LiveCodeBench, which often require problem
reframing and multi-step inference. GPQA-Diamond benefits somewhat more from factual knowledge, likely
because some questions can be answered more directly from domain concepts or definitions, but procedural
guidance remains stronger overall.
These results motivate our focus on building a scalable datastore that explicitly captures procedural knowledge
and retrieves it in a form that reasoning models can readily use. Table 3 reports the full per-task results for
factual and procedural knowledge injection. The “Average” row corresponds to the aggregate numbers shown
in Figure 5.
Llama-3.1-8B (Inst.) Llama-3.1-8B (Reas.) Qwen2.5-7B (Inst.) Qwen2.5-7B (Reas.) Qwen2.5-32B (Inst.) Qwen2.5-32B (Reas.)
Fact. Proc. Fact. Proc. Fact. Proc. Fact. Proc. Fact. Proc. Fact. Proc.
AIME 0.040 0.029 0.075 0.090 0.015 0.012 0.096 0.167 0.025 0.067 -0.013 -0.023
GPQA 0.223 0.254 0.150 0.151 0.173 0.204 0.076 0.084 0.159 0.160 0.065 0.076
LCB 0.027 0.059 0.041 0.077 0.029 0.033 0.033 0.053 0.178 0.159 0.004 0.032
Average 0.097 0.114 0.089 0.106 0.072 0.083 0.068 0.101 0.121 0.129 0.019 0.028
Table 3 Per-Task Performance Delta of the Knowledge Synthesis Pilot StudyFact. = factual knowledge and Proc. =
procedural knowledge. The difference between prompting with knowledge augmentation and prompting without any
augmention is provided. Augmenting procedural knowledge generally leads to larger performance gains.
Prompt for Synthesizing Factual Knowledge
You will read a question and write one short passage of factual background knowledge. Directly write
general definitions, concepts, and relationships that might be relevant. Do not state the connection
between the knowledge and the question. Do not include any steps toward solving the question.
Question:
{question}
Write the paragraph now.
Figure 6 Prompt for synthesizing factual knowledge for the pilot study.GPT-5.2 is prompted to generate the knowledge.
B Additional Implementation Details
B.1 Reasoning Memory
15

Prompt for Synthesizing Procedural Knowledge
You will read a question and write one short paragraph of procedural knowledge (generalizable
steps/strategies) as the hint to solving the problem. Describe how someone should approach this general
type of problem step-by-step. Keep it high-level and reusable: do NOT compute anything, do NOT include
concrete numbers or intermediate results.
Question:
{question}
Write the paragraph now.
Figure 7 Prompt for synthesizing procedural knowledge for the pilot study.GPT-5.2 is prompted to generate the knowledge.
Prompt for Trajectory Decomposition
You are an expert tutor. Given a question, a final answer, and a detailed solution process written by the
teacher, break the problem down into a list of general, self-contained subquestions (SQs). Each
subquestion should capture a logical reasoning step or reusable subroutine that can help solve similar
problems. The subquestions should be general and you should avoid using the original question's surface
form. Avoid introducing specific numbers or setups as much as you can. In your response, start your
subquestions under a section ### Subquestions .
Write one question on each line, starting by an index 1. 2. 3., etc.
### Question:
{question}
### Teacher Answer:
{teacher_answer}
### Teacher Thought Process:
{teacher_thought}
Now, analyze the provided information and list the important subquestions.
Figure 8Prompt for decomposing trajectory into the subquestions.
Datastore ConstructionWe leverage the publicly released Nemotron post-training V1 corpus5. We use the
math, stem, and code subsets, which contain 2.0, 20.7, and 1.9 million trajectories respectively. Nemotron
pairs each question with multiple reasoning trajectories generated with a teacher model. Empirically, we
found these trajectories often converge to similar general approaches and are mostly differ in local steps or
formatting. As such, decomposing a large number of trajectories for each question would bring marginal gain
to the datatore’s diversity. We thus simply select the first trajectory per question to construct the datastore.
After this deduplication process, we obtained roughly 170k unique questions for math, 3 million for stem, and
34k for code, taking up approximately 10% of the original dataset. Each trajectory contains a user question
and a response generated by DeepSeek-R1. We prompt QwQ-32B with the prompt as shown in Figure 8 to
generate subquestions from the full trajectories with temperature=0.6, top_p=0.95, and max_tokens=10,000.
Next, we leverage the prompt shown in Figure 9 to generate a subquestion-specific subroutine conditioned on
the original question, trajectory, and the subquestion. For this step, we prompt QwQ-32B with the same
set of parameters. The final datastore contains 32 million (subquestion, subroutine pairs). On average, we
observe 10.5 subquestions decomposed from each trajectory. Each subquestion on average contains 17.7 words
(or 19.2 tokens using QwQ-32B tokenizer), and each subroutine on average contains 197.6 words (or 207.9
tokens), significantly shorter than the thinking in the original trajectories.
5Downloaded athttps://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1.
16

Prompt for Subroutine Generation over Subquestions
You are an expert tutor. Given a question, a final answer, a detailed solution process, and a subquestion
(SQ),
write a reusable subroutine-style hint that describes how to solve SQ.
The hint should restate the general subproblem it addresses, and then describe the reasoning steps one
might follow.
Use a first-person voice like "For problems like this, I should...".
In your response, start by outlining the general problem under a section ### Applicable Problems .
Then, start your hint on a new line by ### Hint.
Inside the hint, you must first re-state the general problem setting that the hint can apply to.
### Question:
{q}
### Teacher Answer:
{ans}
### Teacher Thought Process:
{thought}
### Subquestion:
{subq}
Now, analyze the question, answer, and teacher's thought and write your hint for the subquestion.
Make sure your hint helps approach similar questions without revealing the answer or any intermediate
results.
Figure 9Prompt for generating reusable subroutines for subquestions.
RetrievalWeleverageReasonIR-8Bforallthereasoningmemoryexperimentsinthepaper6. Thesubquestions
are encoded and used as the keys. We do not add special instructions but only added the special token
"<|embed|>" as used in the official codebase. For the queries, we prepend an additional domain-specific
instruction before encoding with ReasonIR, as shown below:
(For Math)Please answer the following math question.
(For Coding)Generate a correct Python program that passes
all tests for the given problem.
(For Science)Please answer the following question.
RAG InferenceWe perform query verbalization and final generation using the prompt presented in §3.2.
For query verbalization, we add the thought hijacking prompt immediately after the question and generate a
maximum of 100 tokens. Then, we use the nltk toolkit (Bird and Loper, 2004) to segment the first sentence in
the model’s response as the query. For the final answer generation, we use top-p sampling with temperature
0.7 and p= 0.95for Qwen3-32B and p= 0.8for the other two small models. The maximum number of output
tokens is set to 32,768. This same set of parameters is used for all the baselines as well. All the experiments
are run with a supercomputing cluster containing A100, H100, and H200 nodes. We host the models via vllm
(Kwon et al., 2023) and perform batched calls.
Sample FilteringWe perform additional filtering over the msamples collected from top- kretrievals. For πj,l,
thesample lofthe jthretrievedhint, wecalculatetherawscore rj,las|πj,l|, thelengthofthetrajectoryinterms
of tokens. Let rmaxandrminbe the maximum and minimum scores across all samples for the same question,
we normalize the score to get a quality score for each sample as ˆrj,l= (rmax−rj,l)/(rmax−rmin)∈[0,1]. We
6Accessed athttps://huggingface.co/reasonir/ReasonIR-8B
17

Original Question:
A museum prints ticket codes in an unknown base b≥6. One ticket reads2 A5b, where Ais a single
digit. When interpreted as a base-10 integer, the code must (1) be the product of two distinct primes
and (2) satisfy the checksum N≡ 3 (mod 11). Determine all pairs( b, A)that produce a valid ticket.
Original Reasoning Trajectory:
Okay, so I’ve got this math problem here...(very long, omitted)
Subquestions:
[SQ1] How do you verify that a given numeral is valid in baseb?
[SQ2] How do you convert a number from an arbitrary basebto its decimal (base 10) equivalent?
[SQ3] How do you determine whether a number is a product of two distinct primes?
[SQ4] How do you solve a modular arithmetic constraint of the form N≡r (modm )when N is an
algebraic expression in unknown integers?
Subroutine for SQ2:
For problems like this, I should first write the number as a sum of each digit multiplied by the base
raised to the power of its position index. Here’s how:
1.Identify each digitof the number in the original base.
2.Assign positional indicesstarting from0at the rightmost digit (the least significant digit).
3.Compute the weightof each digit by raising the basebto the power of its position index.
4.Multiply each digitby its corresponding weight.
5.Sum all the productsto obtain the decimal (base 10) equivalent.
For example, a numberd ndn−1. . . d 1d0in basebconverts to decimal as:
dn·bn+dn−1·bn−1+. . .+d 1·b1+d 0·b0.
Ensure all digits are valid in the original base (i.e.,0≤d i< b) before proceeding.
Figure 10 An Example of the Reasoning Memory Datastore.We extract multiple self-contained subquestions from the
original question and the long reasoning trajectory. Then, a corresponding subroutine is generated for each subquestion.
note that that since shorter responses are preferred, a higher ˆrj,lis better. To filter retrieved subroutines, we
calculate ˆrjby averaging the samples corresponding to this subroutine, and retain the subroutines with score
higher than τ= 0.1. Finally, we sort the samples corresponding to the retained subroutines and retain n
samples with the highestˆr j,l.
B.2 Baselines
In this section, we outline the implementation of the baseline methods.
•Trajectory RAG (Prefix)We index the deduplicated Nemotron trajectories using their questions as the
key and ReasonIR as the embedding model. The test questions are used as the query. We use the first
256 tokens of the top-3 retrieved trajectories as we empirically found it to cover the first paragraph,
which is often a high-level plan of the problem solving strategy. We format the retrieved trajectories in
the same way as Reasoning Memory.
•Trajectory RAG (Summary)We use the same retrieval methods as Trajectory RAG (Prefix). The only
difference is that we prompt the QwQ-32B to extract the high-level question from the trajectory and
write a corresponding subroutine using a prompt similar to Figure 8 and Figure 9. We use top-1 retrieved
trajectory as it empirically leads to better performance than top-3 or top-5.
•Template RAG (ReasonFlux)We access the templates from ReasonFlux’s official repository7. After
7https://github.com/Gen-Verse/ReasonFlux/blob/d82429dae582c8ecbe762d643c909955e7cb9636/ReasonFlux_v1/data/
18

preprocessing, we collected a corpus with 420 solution templates for various math problems. We
concatenate the template metadata and the content as the key for retrieval. Using ReasonIR, we retrieve
top-1 relevant template and evaluate using the same prompt as Reasoning Memory.
•Template RAG (Self-Discover) We convert the templates directly from the appendix section in (Zhou
et al., 2024) and ended up with a corpus containing 39 high-level problem solving strategies. We
concatenate the template metadata and the content as the key for retrieval. Using ReasonIR, we retrieve
top-1 relevant template and evaluate using the same prompt as Reasoning Memory.
•Document RAG (CompactDS)We use the same implementations for the pilot study. Using the official
codebase and data8, we retrieve and augment top-3 documents via two-step exact search.
•Document RAG (Google)We use the baseline implementation provided by Li et al. (2025b). We use Serper
search and augment the top-10 retrieved document in the context following the original implementation9.
In our preliminary studies, we found the default Serper search results contain a large number of data
contamination instances, where the webpage content directly contains the answer. We therefore set the
date cutoff to 2023/12/31, which we found to minimize the contaminations.
•Length ScalingWe use the same implementation as in the pilot study to repeatedly generate samples.
Our implementation was inspired by Li et al. (2025a)’s baseline implementation10.
C Additional Analyses
C.1 Combining Baselines with Length Scaling
In the main text, we only apply length-based uncertainty filtering to the retrieval-free Length Scaling baseline
and to Reasoning Memory. For completeness, Table 4 reports results when we additionally apply the same
filtering procedure to other RAG baselines under the m= 8budget, further reducing the pool to n= 4
samples. Length filtering generally improves performance for the smaller reasoning models, especially for No
RAG and Document RAG. For the larger Qwen3-32B model, however, the effect is mixed: length filtering
helps No RAG and Document RAG but consistently hurts Trajectory RAG and Template RAG. Since this
step is not a standard practice in prior work and its impact is not uniform across settings, we do not include
these variants in the main comparisons.
DeepSeek-R1-Distill-Llama-8B OpenThinker3-7B Qwen3-32B
Method AIME 24’ GPQA LCB (V1–4) AIME 24’ GPQA LCB (V1–4) AIME 24’ GPQA LCB (V1–4)
No RAG 0.461 0.427 0.220 0.470 0.438 0.318 0.789 0.669 0.387
+ length scaling 0.508 0.426 0.255 0.612 0.484 0.333 0.792 0.699 0.399
Trajectory RAG (Summary) 0.408 0.431 0.184 0.438 0.451 0.242 0.629 0.595 0.267
+ length scaling 0.483 0.439 0.175 0.533 0.444 0.290 0.567 0.580 0.224
Template RAG (Self-Discover) 0.442 0.414 0.213 0.442 0.441 0.276 0.667 0.633 0.312
+ length scaling 0.500 0.431 0.252 0.533 0.451 0.303 0.608 0.623 0.215
Document RAG (Google) 0.483 0.383 0.246 0.501 0.452 0.235 0.795 0.678 0.382
+ length scaling0.5230.390 0.265 0.475 0.432 0.250 0.8150.6820.436
Reasoning Memory 0.5110.444 0.278 0.725 0.522 0.362 0.8250.6740.462
Table 4Effect of adding length-based uncertainty filtering to retrieval-based baselines, withm= 8andn= 4.
C.2 Retrieval vs. Training
Reasoning Memory provides a model-agnostic way to digest a reasoning corpus as an external datastore
and perform RAG at inference time. Alternatively, post-training can be performed to internalize the corpus
into a model’s parameter. Is Reasoning Memory still useful after the same corpus has been internalized? In
this section, we investigate this question by experimenting with the Llama-3.1-Nemotron-Nano-8B-v1 model
template_library.json
8https://github.com/Alrope123/compactds-retrieval
9https://github.com/RUC-NLPIR/WebThinker/blob/main/scripts/run_naive_rag.py
10https://github.com/RUC-NLPIR/Search-o1/blob/main/scripts/run_direct_gen.py
19

released by Nvidia. According to the model card, it is likely that this model has been trained on some variants
of the Nemotron v1 corpus via supervised fine-tuning (SFT) and reinforcement learning (RL). Despite this,
we emphasize that the comparisons provided in this section is not fully controlled for two reasons: (1) the
detailed training data composition and compute recipe for the model is not entire clear and (2) due to question
deduplication, we only use a subset of Nemotron that is roughly 10% large. A fully controlled comparison
would re-run post-training from the backbone model, which is beyond our budget.
As shown in Table 5, we observe a mixed result comparing Reasoning Memory with no retrieval and length
scaling. Despite the model being trained on similar corpus, Reasoning Memory is still able to improve the
performance over no retrieval on AIME and LCB. However, it cannot outperform the length scaling baseline
on GPQA and LCB. These results indicate that fine-tuning is a strong method to learn a corpus and the
contribution from Reasoning Memory might not be compositional, depending on the corpus. This also opens
several exciting future directions, such as expanding the knowledge sources for Reasoning Memory, actively
learning the reasoning memory corpus, and learning more helpful subroutines that incorporate the knowledge
of the model’s ability.
Method AIME 24’ GPQA LCB (V1–4)
No RAG 0.6580.5530.380
Length Scaling 0.692 0.5480.396
Reasoning Memory0.7080.534 0.395
Table 5Results on AIME 24’, GPQA, and LCB (V1–4) using Llama-3.1-Nemotron-Nano-8B-v1. For both Length
Scaling and Reasoning Memory we use the budgetm= 30andn= 8.
C.3 Uncertainty Criteria
Our main system uses generation length as the uncertainty proxy for stage 1 item-level filtering after retrieval.
In this section, we compare this metric with several alternative criteria while keeping stage 2 fixed to
length-based selection. Specifically, for each retrieved subroutine, we compute:
•Likelihood: log likelihood of the first 200 continued tokens conditioned on the retrieved subroutine. For
this metric, lower is better as we expect the retrieval to inject new and unfamiliar information.
•Entropy: average token-level entropy of the first 200 continued tokens conditioned on the retrieved
subroutine. Higher is better.
•Contrast: the distributional difference between the downstream reasoning model and its base pretrained
model, evaluated on the the first 200 tokens generated in the continuation. This metric is inspired by
Ouyang et al. (2025b) and we instantiate it with KL Divergence (higher is better).
•Self-Eval: a scalar score where the model is prompted to rate how relevant the retrieved subroutine is to
the problem on a Likert scale. We repeatedly sample for 10 times with temperature = 0.6 and take the
average as the score to apply the filter.
All scores are then normalized using the same procedure as in §3.3. Using DeepSeek-R1-Distill-Llama-8B with
a budget of m= 30, we report the task accuracy results in Table 6. Overall, no single alternative dominates
across all benchmarks, but length is the most reliable choice overall. On AIME 24’, entropy achieves the
highest score (0.600), with length close behind at 0.575. On GPQA-D and LCB (V1–4), however, length
clearly outperforms all other metrics (0.461 vs. 0.429 or lower on GPQA-D, and 0.310 vs. at most 0.273 on
LCB). Averaged over the three tasks, length is therefore the strongest option, while also being significantly
simpler and cheaper than metrics that require extra model evaluations. For this reason we adopt length as
our default uncertainty proxy in the main experiments.
C.4 Qualitative Study
We conduct a small qualitative study in this section. Tables 7, 8 and 10 show examples of Reasoning Memory
runs with DeepSeek-R1-Distill-Llama-8B. Across examples, the self generated query serves as a compact
20

Metric AIME 24’ GPQA-D LCB (V1–4)
Likelihood 0.571 0.429 0.273
Entropy0.6000.417 0.273
Contrast 0.563 0.419 0.269
Self-Eval 0.571 0.402 0.263
Length 0.5750.461 0.310
Table 6 Alternative metrics for step 1 filtering.Results for DeepSeek-R1-Distill-Llama-8B with budget m=30; stage 2
always uses length as the uncertainty proxy. Length achieves strong overall performance while being the simplest.
paraphrase of the model’s latent goal (“convert between bases”, “factor a quadratic”, etc.), which makes it
well matched to the subquestions stored in the datastore. In all four examples, the retrieved subroutines
are both relevant to the question and general enough to transfer across problems and benchmarks. We also
observe that the continued reasoning rarely copies the hint verbatim. Instead, the model typically restates
the problem and derives the key steps, sometimes refering to the retrieved subroutine later on in the solution,
suggesting that the hint acts more like a latent steering signal than an explicit template to fill in.
The two failure cases are shown in Tables 11 and 12. In the first failure, the self verbalized query becomes
overly specific, including particular numbers and surface details that occur infrequently in the datastore,
which leads to irrelevant and thus less useful retrievals. In the second failure, the retrieved hint is physically
reasonable but mismatched to the problem setup: the guidance correctly included information such as using
Stefan–Boltzmann law but does not capture the problem-specific setup that the stars are blackbodies. As
a result, the hint is generic and somewhat distracting. These cases highlight the two current limitations of
Reasoning Memory, namely query specificity and the problem-specific setting mismatch.
21

Question Please answer the following math question. Find the sum of all integer bases b >9for which
17bis a divisor of97 b.
Query17 in basebis equal to what in decimal?
Retrieved
SubquestionHow do you convert a number from an arbitrary basebto its decimal (base 10) equivalent?
Associated
SubroutineFor problems like this, I should first write the number as a sum of each digit multiplied by the
base raised to the power of its position index. Here’s how:
1.Identify each digitof the number in the original base.
2.Assign positional indicesstarting from0at the rightmost digit (the least significant digit).
3.Compute the weightof each digit by raising the base bto the power of its position index.
4.Multiply each digitby its corresponding weight.
5.Sum all the productsto obtain the decimal (base 10) equivalent.
For example, a numberd ndn−1. . . d 1d0in basebconverts to decimal as:
dn·bn+dn−1·bn−1+. . .+d 1·b1+d 0·b0.
Ensure all digits are valid in the original base (i.e.,0≤d i< b) before proceeding.
Continued
ReasoningOkay, so I have this math problem here: Find the sum of all integer bases b >9for which17 b
is a divisor of97 b. Hmm, okay. Let me try to figure this out step by step.
First, I remember that when numbers are written with a subscript like17 b, it means they’re in
base b. So I need to convert both17 band97 binto decimal (base 10) to work with them more
easily.
Let’s start with17 b. In base b, the rightmost digit is the units place (which is b0) and the next
digit is theb1place. So17 bwould translate to:
1×b1+ 7×b0=b+ 7.
Got that. So17 b=b+ 7in decimal.
Now, let’s do the same for97 b...
Table 7An example from AIME 2025 where Reasoning Memory is effective.
22

Question Two-thirds of the students at Baker Middle School take music. There are 834 students who
take music. How many students are there at Baker Middle School?
Query2
3of something equals 834, what is that something?
Retrieved
SubquestionWhat is the result of dividing 3.6 by 1.6?
Associated
SubroutineFor problems like this, I should first eliminate decimals by expressing the division as a fraction
of whole numbers. For example, if dividing a.bbyc.d, multiply both the numerator and
denominator by 10 to convert them into whole numbers (ab
cd). Then, simplify the fraction by
dividing the numerator and denominator by their greatest common divisor or perform long
division directly. If simplifying isn’t straightforward, break the division into parts: calculate
how many times the divisor fits into the dividend, subtract that portion, and handle the
remainder to find the exact value. This method works for any decimal division where precision
is needed.
Continued
ReasoningOkay, so I’ve got this math problem here: Two-thirds of the students at Baker Middle School
take music, and there are 834 students who take music. The question is asking how many
students are there at Baker Middle School in total. Hmm, let’s think about this step by step.
Alright, two-thirds of the school takes music, which means if I can figure out what number
multiplied by two-thirds gives me 834, that number will be the total number of students at the
school. So, essentially, I need to solve for the total number of students, which I’ll callS.
The equation would be:
2
3·S= 834.
To find S, I should isolate it on one side of the equation. To do that, I can multiply both sides
by the reciprocal of two-thirds, which is three-halves 3
2
. So, multiplying both sides by3
2:
S= 834·3
2.
Let me compute that.834divided by2is417, and then417multiplied by3is:400 ×3 = 1200,
and17×3 = 51, so adding those together gives me1251. So,S= 1251.
Table 8An example from MATH500 where Reasoning Memory is effective.
23

Question Please answer the following question. Imagine a point charge qis moving with a trajectory
⃗ s(t), where tis time, with respect to the origin. Let ⃗ rbe the field point, where the field is
observed, with respect to the origin of the same reference frame, and ⃗dbe the vector from the
point where the electromagnetic field was generated at some earlier time trto the observation
point ⃗ r. The velocity of the moving charge qat the field generating instant tris⃗ v. What
are the scalar potential Vand vector potential ⃗Aat time t, satisfying t > t r, and position ⃗ r?
In obtaining the expressions, use the notation cfor the speed of light in vacuum, ϵ0for the
permittivity of free space andµ 0for the permeability. Choices:
(A)V(⃗ r, t) =qc
4πϵ 0
dc− ⃗d·⃗ v,⃗A(⃗ r, t) =µ0qc⃗ v
4π
dc− ⃗d·⃗ v
(B)V(⃗ r, t) =q
4πϵ 0r,⃗A(⃗ r, t) =⃗ v
c2V(⃗ r, t)
(C)V(⃗ r, t) =qc
4πϵ 0
dc+ ⃗d·⃗ v,⃗A(⃗ r, t) =µ0qc⃗ v
4π
dc+ ⃗d·⃗ v
(D)V(⃗ r, t) =q
4πϵ 0r,⃗A(⃗ r, t) =⃗ v2
c2V(⃗ r, t)
Queryhow to find scalar and vector potentials for a moving charge
Retrieved
SubquestionWhat is the general form of the scalar potential for a point charge moving at constant velocity
using retarded potentials?
Associated
SubroutineFor problems like this, I should recall the Liénard–Wiechert potential formulation. The scalar
potential depends on the charge’s position at the retarded time
tr=t−R
c,
where Ris the distance from the retarded position of the charge to the field point. The general
form includes a correction factor due to the charge’s constant velocityv, involving the unit
vector ˆnfrom the charge to the field point. Specifically, the potential’s denominator must
include a term
1−β· ˆn,whereβ=v
c,
to account for relativistic effects. This ensures the potential accounts for the charge’s motion
relative to the observation point.
Continued
ReasoningOkay, so I’ve got this physics problem here about finding the scalar potential Vand vector
potential ⃗Afor a moving charge. Hmm, I’m a bit rusty on this, but let me think it through
step by step.
The question says we have a point charge qmoving along a trajectory ⃗ s(t). At a later time t,
we’re looking at the electromagnetic field at a point ⃗ r. The vector ⃗dis from the point where
the field was generated at time tr(the retarded time) to the observation point ⃗ r. The velocity
of the charge at timet ris⃗ v. So, I need to findVand ⃗Aat timet > t r.
Wait, what’s the retarded time? Oh right, in electrodynamics, when dealing with moving
charges, you have to consider the light travel time delay. So the retarded time tristminus the
light travel time from the charge’s position at timet rto the observation point⃗ r. So,
tr=t−R
c,
whereRis the distance between the charge’s position att rand⃗ r.
Now, ...
Table 9An example from GPQA-D where Reasoning Memory is effective.
24

QuestionGenerate a correct Python program that passes all tests for the given problem.
Problem Title:Ax+By < C
Problem Statement:
You are given three length-Nsequences of positive integers:
A= (A 1, A2, . . . , A N), B= (B 1, B2, . . . , B N), C= (C 1, C2, . . . , C N).
Find the number of pairs of positive integers(x, y)that satisfy the following condition:
Ai×x+B i×y < C ifor all1≤i≤N.
It can be proved that the number of such pairs of positive integers satisfying the condition is
finite.
You are givenTtest cases, each of which should be solved.
Query how to count the number of( x, y)pairs that satisfy multiple inequalities Aix+Biy < C ifor
alli
Retrieved
SubquestionHow do you count the number of integer pairs( x, y)that satisfy a linear inequality with
non-negative coefficients and bounds?
Associated
SubroutineFor problems like this, I should consider the following steps:
1.Fix one variable and solve for the other.Choose one variable (e.g., y) and iterate through its
possible values within the bounds. For each fixed y, determine the minimum or maximum
value of the other variable (e.g.,x) required to satisfy the inequality.
2.Use complementary counting if feasible.If the inequality’s valid region is large, it may be
easier to calculate the total number of candidate pairs (for example,( N+ 1)2for bounds
0≤x, y≤N ) and subtract the count of pairs that donotsatisfy the inequality. This is
especially useful when the invalid region is smaller or has a simpler structure.
3.Account for variable bounds.Ensure all computed ranges for xoryrespect their given
constraints (e.g.,0 ≤x≤N ). Adjust the ranges to stay within these bounds, even if the
inequality suggests values outside them.
4.Sum valid counts systematically.For each fixed value of the first variable, compute the
number of valid values for the second variable and accumulate these counts. Organize
the calculations (for example, by grouping cases where the inequality trivially holds or
fails) to avoid errors and redundant work.
This approach balances direct enumeration with strategic shortcuts to efficiently count solutions
while avoiding overcomplication.
Continued
ReasoningSo I’ve got this programming problem to solve. Let’s read it carefully.
The problem is about finding the number of pairs( x, y)of positive integers that satisfy
Aix+Biy < C ifor all ifrom1to N. So for each i, the same xandyhave to satisfy this
inequality.
In other words, we need to find all(x, y)such that foreveryi,
Aix+B iy < C i
holds simultaneously. We are given Ttest cases, and for each test case we get Nlines of
Ai, Bi, Ci.
The goal is: for each test case, compute the number of such(x, y)pairs.
Geometrically, each inequality Aix+Biy < C idescribes a half-plane in the first quadrant (since
xandyare positive integers). The feasible region is the intersection of all these half-planes
restricted to x≥1,y≥1. The statement that the answer is finite means this intersection is a
bounded polygonal region.
One possible strategy is to treatxas the main loop variable...
Table 10An example from LiveCodeBench where Reasoning Memory is effective.
25

QuestionThere arenvalues ofxin the interval0< x <2πwhere
f(x) = sin 
7π·sin(5x)
= 0.
Fortof thesenvalues ofx, the graph ofy=f(x)is tangent to thex-axis. Findn+t.
Query How to find the number of solutions to sin(7π·sin (5x)) = 0between0and2 π, and how many
of those have tangent tox-axis.
Top-3
Retrieved
Subquestion1.How do you analyze the behavior of a trigonometric function (e.g., using derivatives) to
confirm the number of solutions in a given interval?
2.How do you systematically count and confirm the total number of valid solutions by
combining results from both cases(sin(x) = 0and2 cos(x) + cos(2x) = 0)?
3.How do you analyze the behavior of a function (e.g., f(x) =tan(2x) +sin(x)) in intervals
between vertical asymptotes to determine the number of solutions?
Table 11A failure example from AIME 2025 caused by over-specific queries.
26

Question Astronomers are interested in the luminosities of two stars. Star 1 has a radius1 .5times
larger than that of Star 2. The mass of Star 1 is also larger by the same factor (1 .5times).
Through spectroscopy, the astronomers managed to determine the wavelengths at which the
stars appeared brightest, and found that they are the same. Astronomers also calculated
the radial velocities of the two stars to be0and700 km/sfor Star 1 and Star 2 respectively.
Assuming that the stars radiate as black bodies, they computed the luminosity of Star 1 to be
greater than the luminosity of Star 2 by a factor of:
Choices:
(A)∼2.32
(B)∼2.23
(C)∼2.35
(D)∼2.25
Queryastronomers luminosity star radius relation
Retrieved
SubquestionWhat is the relationship between a star’s radius and its luminosity?
Associated
SubroutineFor problems asking about the relationship between a star’s radius and luminosity, I should
start by recalling the Stefan–Boltzmann law:
L= 4πR2σT4,
where Lis luminosity, Ris radius, Tis temperature, and σis the Stefan–Boltzmann constant.
Luminosity depends on both radius and temperature. To isolate the radius–luminosity relation-
ship, consider how changes in radius affect Lwhile holding temperature constant. However, for
main-sequence stars, radius and temperature are not independent—both increase with mass.
Use the mass–radius and mass–luminosity relationships (e.g., L∝M3.5,R∝M0.7) to deduce
the overall correlation. Remember that temperature also contributes strongly to luminosity, so
the radius’s role must be analyzed in concert with mass and temperature dependencies.
Continued
ReasoningOkay, so I have this astronomy problem here about the luminosities of two stars, Star 1 and
Star 2. Let me try to work through it step by step. The question says that Star 1 has a
radius1 .5times larger than Star 2, and the mass of Star 1 is also1 .5times larger. They also
mention that the wavelengths at which the stars appear brightest are the same, and their radial
velocities are0for Star 1 and700 km/sfor Star 2. The question is about the ratio of their
luminosities, so I need to compute how much brighter Star 1 is compared to Star 2.
I remember that the luminosity of a star is related to its radius and temperature. The Stefan–
Boltzmann law is
L= 4πR2σT4,
where Lis luminosity, Ris radius, σis the Stefan–Boltzmann constant, and Tis the effective
temperature. At first I might wonder whether the Doppler effect due to different radial velocities
matters here, but the problem states that the wavelengths at which they appear brightest are
the same. That suggests their effective temperatures are the same, because the peak wavelength
(or color) is determined by temperature via Wien’s law. So I can takeT 1=T 2.
If the temperatures are equal, then the ratio of luminosities depends only on the ratio of radii:
L1
L2=R1
R22T1
T24
=R1
R22
.
SinceR 1= 1.5R 2, this gives
L1
L2= (1.5)2= 2.25,
which corresponds to choice (D), approximately2.25.
Table 12A failure example from GPQA-D caused by problem setup mismatch.
27