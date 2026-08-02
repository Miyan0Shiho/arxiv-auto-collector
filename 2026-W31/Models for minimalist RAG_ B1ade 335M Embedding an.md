# Models for minimalist RAG: B1ade 335M Embedding and 1B Parameter Small Language Models

**Authors**: Shreyas Subramanian, Mecit Gungor, Vikram Elango

**Published**: 2026-07-29 22:41:25

**PDF URL**: [https://arxiv.org/pdf/2607.27506v1](https://arxiv.org/pdf/2607.27506v1)

## Abstract
Language and embedding models used in RAG systems are conventionally assumed to require large-scale pretraining and explicit grounding supervision. We present B1ade, an efficient RAG architecture comprising two purpose-built components: a compact embedding model and a purpose-built SLM. B1ade-embed, a 335M parameter retrieval model constructed via parameter-free fusion of five pretrained encoders achieves top MTEB scores among sub-500M models with zero additional training, and B1ade-1B, an SLM trained on low-cost GPUs using Group Relative Policy Optimization (GRPO) on 723M tokens (2.2M examples) of curated context-question pairs with rewards that optimize only answer similarity. Our central finding is emergent attribution: despite receiving no explicit supervision for source citation, B1ade-1B cites retrieved passages in 42.4% of responses, exceeding the attribution rate of its training distribution by 5.5 percentage points. This demonstrates that grounding behavior can emerge as an accuracy-maximizing strategy under RL training, without explicit reward engineering. On standard QA benchmarks, B1ade-1B achieves 81.82% on PopQA, 65.8% on PubMedQA, and 51.09% on FEVER. In end-to-end RAG evaluation, B1ade-1B achieves an average score of 0.654 across correctness, completeness, coherence, and faithfulness, a 10.8% improvement over the SFT, while closing the gap with models 1.5x its size. These results show that strategic model composition and reward design suffice for resource-efficient RAG, without large-scale pretraining.

## Full Text


<!-- PDF content starts -->

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Models for minimalist RAG: B1ade 335M Embedding and 1B
Parameter Small Language Models
Shreyas Subramanian, Mecit Gungor, Vikram Elango
Amazon, Seattle, USA
mecit@amazon.com
Abstract
Language and embedding models used in RAG systems are conventionally
assumed to require large-scale pretraining and explicit grounding super-
vision. We presentB1ade, an efficient RAG architecture comprising two
purpose-built components: a compact embedding model and a purpose-
built SLM. B1ade-embed, a 335M parameter retrieval model constructed via
parameter-free fusion of five pretrained encoders achieves top MTEB scores
among sub-500M models with zero additional training, and B1ade-1B, an
SLM trained on low-cost GPUs using Group Relative Policy Optimiza-
tion (GRPO) on 723M tokens (2.2M examples) of curated context-question
pairs with rewards that optimize only answer similarity. Our central find-
ing is emergent attribution: despite receiving no explicit supervision for
source citation, B1ade-1B cites retrieved passages in 42.4% of responses,
exceeding the attribution rate of its training distribution by 5.5 percent-
age points. This demonstrates that grounding behavior can emerge as an
accuracy-maximizing strategy under RL training, without explicit reward
engineering. On standard QA benchmarks, B1ade-1B achieves 81.82% on
PopQA, 65.8% on PubMedQA, and 51.09% on FEVER. In end-to-end RAG
evaluation, B1ade-1B achieves an average score of 0.654 across correctness,
completeness, coherence, and faithfulness, a 10.8% improvement over the
SFT, while closing the gap with models 1.5 ×its size. These results show
that strategic model composition and reward design suffice for resource-
efficient RAG, without large-scale pretraining.
1 Introduction
Since its introduction, Retrieval Augmented Generation has emerged as a critical ap-
proach for grounding large language model outputs in verifiable sources, particularly
for knowledge-intensive tasks in specialized domains Lewis et al. (2020). However, deploy-
ing effective RAG systems at scale requires addressing two fundamental challenges: (1)
efficient retrieval of relevant passages from large corpora, and (2) faithful generation that
acknowledges and cites source material. While recent advances in foundation models have
improved both retrieval and generation capabilities, state-of-the-art systems typically rely
on large models (7B+ parameters) with substantial computational requirements, limiting
accessibility for resource-constrained deployments.
A central question for efficient RAG development is whether competitive performance re-
quires extensive pretraining, or whether strategic model composition and targeted optimiza-
tion can achieve similar capabilities with dramatically reduced compute. We investigate this
through two complementary techniques: zero-training model merging for dense retrieval,
and direct policy optimization on compact language models for grounded generation. Our
approach demonstrates that careful architectural choices and training strategies can match
or exceed the per-parameter efficiency of conventional large-scale pretraining.
We introduceB1ade-embed, a 335M parameter embedding model constructed through
parameter-free fusion of five pretrained encoders, achieving top performance in sentence
similarity among sub-500M models without additional training. This establishes zero-
1
arXiv:2607.27506v1  [cs.CL]  29 Jul 2026

Preprint:https://huggingface.co/collections/w601sxs/b1ade
training model merging as a compute-free method for constructing competitive dense
retrievers. For generation, we presentB1ade-1B, a 1B parameter model trained via Group
Relative Policy Optimization (GRPO) (Shao et al., 2024) on only 723M tokens of curated
context-question pairs which is orders of magnitude fewer than standard pretraining ap-
proaches.
Our primary contribution is the empirical demonstration of emergent attribution behavior
inB1ade-1Bsimilar to what is described in Wei et al. (2022). Despite training with a reward
function that optimizesexclusivelyfor answer similarity, with no explicit term for attribution
or source citation, the model learns to cite sources in 42.4% of responses on evaluation
datathat is a statistically significant 5.5 percentage point increase over the 36.9% attribution
rate in its training distribution. This emergence validates that grounding behaviors can arise
as learned accuracy-maximizing strategies when careful reading and source acknowledg-
ment correlate with correct answers, rather than requiring hand-crafted attribution rewards.
The model exhibits strategic application, using specific passage citations at 5.6 ×the training
rate and applying more attribution to complex multi-hop questions than simple factual
queries.
We developB1ade-1Bthrough an iterative process. Our initial model (v1) based on GPT-
NeoX with supervised fine-tuning (SFT) achieved suboptimal RAG performance. This
motivated a second iteration (v2) combining a Llama-3.2-1B base architecture with GRPO
training, substantially improving performance across multiple evaluation dimensions. Em-
pirically,B1ade-1Bachieves 81.82% on PopQA, 65.8% on PubMedQA, and 51.09% on FEVER,
demonstrating that small models trained with minimal compute can achieve competitive
performance on standard benchmarks.
Contributions:
•We introduce zero-training model merging as a compute-free method for con-
structing competitive dense retrievers, demonstrating that parameter-free fusion of
pretrained encoders can achieve top performance without additional training.
•We show that direct policy optimization on compact language models with simple
accuracy rewards induces emergent attribution behavior, with models learning to
cite sources 5.5 percentage points more frequently than their training distribution
despite no explicit supervision for grounding.
•We demonstrate that small language models trained via GRPO on 723M tokens
achieve competitive RAG performance, establishing that strategic training can
match conventional approaches with orders of magnitude less compute.
•We provide empirical analysis of the strategy-execution gap in small model attri-
bution, showing that while models learn when to cite sources, factual knowledge
limitations remain the primary performance bottleneck.
The remainder of this paper is organized as follows. Section 2 reviews related work in
grounded generation, small language models, and reinforcement learning for NLP . Section 3
describes our training methodology, including the iterative development process and GRPO
training approach. We show how the simpleCoT training collection is created, and discuss
its unique attribution characteristics. Section 4 evaluates B1ade across three complementary
settings: standard benchmarks, end-to-end RAG, and LLM-as-Judge comparative evalua-
tion, with emphasis on the emergent attribution finding. Section 5 discusses implications,
limitations, and evaluation biases identified in our analysis. Section 6 concludes with a
summary of contributions and future directions.
2 Related Work
Recent advances in RAG have demonstrated that compact language models can achieve
competitive performance when augmented with external knowledge retrieval capabilities
Sorstkins (2025). Several compact models in the 1-4 billion parameter range have emerged
as viable alternatives to larger architectures, with research establishing 1B parameters
as a practical lower bound for effective instruction-following and question-answering
2

Preprint:https://huggingface.co/collections/w601sxs/b1ade
capabilities. Notable examples include Pleias-RAG-350M and Pleias-RAG-1B Langlais
et al. (2025), which outperform most small language models below 4 billion parameters on
standardized RAG benchmarks such as HotPotQA and 2wiki, demonstrating competitive
performance with larger 7-8B models including Qwen-2.5-7B and Llama-3.1-8B. These
findings suggest that carefully designed compact architectures can achieve strong RAG
performance without the resource requirements of their larger counterparts.
To the best of our knowledge, development of competitive embedding models with zero
training and model merging is largely unexplored. However, the development of efficient
embedding models through knowledge distillation has become a critical component of
resource-constrained RAG systems Zhang et al. (2025). Multi-stage distillation frameworks
enable smaller student embedding models to learn from multiple larger teacher models,
addressing the challenge that state-of-the-art embedding models often have numerous
parameters and high vector dimensionality that pose deployment challenges in real-world
scenarios Chen et al. (2025). Recent work has explored distilling RAG capabilities from large
language models to small language models through evidence-based and graph-based knowl-
edge distillation techniques, enabling compact models to replicate retrieval-augmented
generation capabilities without extensive fine-tuning Chen et al. (2025). These advances in
efficient embedding and knowledge distillation techniques have made RAG more accessible
for resource-constrained deployments, aligning with the principle that carefully designed
compact components can provide effective solutions for RAG tasks.
3 Methodology
We present our approach to resource-efficient RAG through two complementary techniques:
zero-training model merging for dense retrieval (B1ade-embed, 335M parameters) and direct
policy optimization for grounded generation (B1ade-1B, 1B parameters). Our methodology
demonstrates that strategic model composition and targeted low-cost training can achieve
competitive performance with dramatically reduced computational requirements compared
to conventional large-scale pretraining.
3.1B1ade-embed: Zero-Training Model Merging for Dense Retrieval
B1ade-embedis a 335M parameter embedding model created usingModel Stock
model merging, demonstrating that parameter-free fusion of pretrained encoders
can achieve competitive performance without additional training. The model
comprises the following components using Mergekit1Goddard et al. (2024): (1)
bert-large-uncased , (2)WhereIsAI/UAE-Large-V1 , (3)BAAI/bge-large-en-v1.5 , (4)
mixedbread-ai/mxbai-embed-large-v1 , (5)avsolatorio/GIST-large-Embedding-v0 . Mod-
els were chosen based on superior MTEB performance (at the time of writing) despite
smaller size and compatible architecturesB1ade-embeduses the Model Stock algorithm Wang
et al. (2024a), which optimizes linear interpolation weights by computing pairwise task
vector similarities across models, providing a principled approach to parameter-free model
composition. See Appendix F for detailed merging configuration and algorithm.
On the MTEB benchmark Muennighoff et al. (2023),B1ade-embedranks #4 overall and #1 in
Retrieval. Among models with output dimension ≤1024 and fewer than 500M parameters,
it achieves #1 in Sentence Similarity, #4 in Summarization, Retrieval, and Pair Classification,
and #5 in Reranking. See Appendix B for complete MTEB rankings (Table 12), including a
comparison against Stella-400M-v5, the strongest sub-500M retriever which motivates our
retriever ablation in Section 4.5.
Since its release on the Huggingface Hub,B1ade-embedhas been independently and exter-
nally validated against various unique tasks. For example, in clinical and biomedical tasks,
Soffer et al. (2024) evaluated 30 models across 2.1M comparisons on clinical notes, synthetic
EHRs, MIMIC-IV ICU data, PubMed abstracts, and research papers, identifyingB1ade-embed
as versatile across both clinical and biomedical domains while noting its computational
1https://github.com/arcee-ai/mergekit
3

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Aspect B1ade v1 B1ade v2
Base ArchitectureGPT-NeoX-1B Llama-3.2-1B
Pre-training Tokens 380B Up to 9T
Post-training Tokens 720M 720M
Epochs 3 1
Training StagesSFT only GRPO only
LoRA Rank 16 16
Target Modules Q, V projection Q, V projection
Trainable Parameters∼8.4M∼8.4M
Precision fp32 bfloat16
Performance (RAG Evaluation)
Overall Score0.5900.654 (+10.8%)
Completeness 0.4350.539 (+22.8%)
Coherence 0.7600.856 (+12.6%)
Table 1: Comparison of B1ade v1 and v2 model configurations and performance. B1ade v2
achieves substantial improvements through architectural switch and GRPO training.
efficiency despite smaller size, particularly strong on short tasks like triage notes and chief
complaints where it competed closely with larger models. A related study tested 39 em-
bedding models across 7 medical semantic similarity tasks using Mount Sinai patient data,
MIMIC-IV , PubMed texts, and Llama-3-70B synthetic data, finding that smaller models like
B1ade-embed(335M parameters) performed comparably to larger models on short tasks.
In labor market applications, Qu et al. (2026) report that a system incorporatingB1ade-
embedachieved 81% Positive Predictive Value in a closed-world evaluation against ESCO’s
occupational hierarchy, demonstrating strong capture of nuanced semantic relationships in
labor market terminology.
The syftr (Pareto-Optimal Generative AI) framework Conway et al. (2025) evaluated over
1023unique RAG configurations, whereB1ade-embedconsistently appeared in Pareto-optimal
solutions, achieving top accuracy on DRDocs (software documentation), CRAG3 (sports),
and Infinitebench (long-context retrieval over 100K tokens), while lying on the performance
envelope for HotpotQA and Financebench.
3.2B1ade-1B: Direct Policy Optimization for Grounded Generation
We developB1ade-1Bthrough an iterative process with two major versions, each addressing
specific limitations discovered through evaluation. This iterative development demonstrates
the importance of base model selection and training methodology for efficient RAG systems.
Table 1 summarizes the key differences and improvements between versions.
Our initial model is based on the GPT-NeoX architecture Andonian et al. (2023) with 1 billion
parameters, initialized from the pre-trained checkpoint and fine-tuned with Low-Rank
Adaptation (LoRA) Hu et al. (2022). While performance on typical text-based benchmarks
was superior to other models in the same weight class, evaluation on end-to-end RAG tasks
revealed suboptimal performance as shown in our results, particularly in completeness and
coherence dimensions, motivating architectural and training improvements.
Based on these limitations on end-to-end RAG tasks, we developed a second version of
B1adewith two key changes: (1) architecture switch to Llama-3.2-1B Grattafiori et al. (2024)
base, and (2) training enhancement directly using GRPO with no SFT checkpoint, since
Llama-3.2-1B provides a useful and competitive base, which we include in our evaluations.
3.3 Training Data: simpleCoT Collection
We construct simpleCoT , a large-scale dataset of 2,214,941 examples (723M tokens) aggre-
gated from seven sources spanning multi-hop QA, instruction following, mathematical
reasoning, and commonsense understanding. All splits are unified into a standardized
4

Preprint:https://huggingface.co/collections/w601sxs/b1ade
schema and shuffled with fixed seed 42, yielding 1,771,953 training and 442,988 test exam-
ples. Full dataset composition and processing details are provided in Appendix A. The
dataset is publicly available.2
3.4 Training
We directly perform GRPO (Shao et al., 2024) on top of the Llama-3.2-1B base using a
standard ROUGE-L reward on answer similarity, with no explicit attribution or grounding
term. We train for 3 epochs with learning rate 5 ×10−6(AdamW), group size k=4, batch
size 8 questions (32 responses), bfloat16 precision, on a single NVIDIA A10G (24GB) for
approximately 12 hours.
Critically, our reward function optimizes only for answer similarity using ROUGE-L score,
with no explicit term for attribution or grounding behavior. This design choice is intentional:
we investigate whether grounding behavior can emerge as a learned accuracy-maximizing
strategy when careful reading and source acknowledgment correlate with correct answers,
without requiring explicit supervision or hand-crafted attribution rewards. This minimal
reward design enables us to isolate and study the emergence phenomenon.
4 Empirical Evaluation
We evaluate our approach across two complementary settings: (1) standard context-based
QA benchmarks to measure core reasoning capabilities, and (2) end-to-end RAG to assess
real-world applicability with retrieval systems using LLM-as-Judge scoring. Our evalua-
tion focuses on validating the emergent attribution phenomenon and characterizing the
performance-efficiency tradeoffs of our training paradigm.
As described in Section 3, we develop two versions ofB1ade-1B: v1 (GPT-NeoX base, SFT
only) and v2 (Llama-3.2-1B base, GRPO-only). Standard benchmarks and LLM-as-Judge
evaluations use v2, while RAG evaluation compares both versions to demonstrate the
improvement from architectural and training changes. Due to the poor performance of the
base v1 model on end-to-end RAG tasks, we do not use the same for subsequent GRPO
training, or for ablation experiments such as SFT-only vs GRPO-only. We focus on showing
that simple reward functions (like rouge score) can still provide meaningful improvements
in tasks like RAG. Next we discuss standard QA benchmarks and our end-to-end RAG
benchmarks involving both B1ade-embedding and B1ade SLM.
4.1 Standard Context-Based QA Benchmarks
We evaluateB1ade-1Bon standard question-answering benchmarks (PopQA, TriviaQA,
Natural Questions, WikiMultihopQA, FEVER, ARC-Challenge, PubMedQA) using Exact
Match accuracy. See Appendix A.2 for the complete evaluation protocol.
Table 2: Standard context-based QA benchmark results (Exact Match %).B1ade-1Bachieves
the highest PopQA score (81.82%) and strong performance on PubMedQA (“PMQA”)
(65.80%) and FEVER (51.09%) among 1B models. 0∗indicates instruction-following failures.
Model Size PopQA TQA NQ ASQA FEVER ARC PMQA WikiMQA
TinyLlama Zhang et al. (2024) 1.1B 64.3260.0044.10 0.48 0∗23.46 0∗0.23
Llama-3.1B Grattafiori et al. (2024) 1.0B 48.55 36.65 35.6740.2123.27 49.91 16.812.79
Phi-1.5 Li et al. (2023) 1.2B 68.30 50.34 0 0.14 0∗51.96 34.4 0.04
OPT Zhang et al. (2022) 1.3B 70.51 43.75 47.80 2.75 0∗22.95 0∗0
Pythia Biderman et al. (2023) 1.0B 73.44 39.7766.900.01 8.17 22.78 18.2 0.06
Qwen-2.5 Yang et al. (2024) 1.5B 74.77 41.10 51.36 37.04 23.9871.9357.8 11.59
B1ade-1B 1.0B 81.8233.11 48.32 16.5651.0922.4465.807.56
2https://huggingface.co/datasets/w601sxs/simpleCoT
5

Preprint:https://huggingface.co/collections/w601sxs/b1ade
B1ade-1Bachieves 81.82% on PopQA (surpassing Qwen-2.5’s 74.77%), 65.80% on Pub-
MedQA, and 51.09% on FEVER. These results show that training on context-based tasks
with reasoning chains enables transfer across domains. Our model surpasses Qwen-2.5 on
three tasks despite Qwen having 50% more parameters, indicating that training on curated
data compensates for reduced capacity. The higher average scores relative to other 1B
models result from strong performance on key tasks and avoidance of catastrophic failures
observed in other sub-2B models (Also see Fig. 1 for a visual). For other models, zero or
near-zero scores result from instruction-following failures and strict exact-match evalua-
tions (detailed analysis in Appendix A.3). Among the models we tested, only Llama-1B
andB1ade-1Bachieve consistent performance without catastrophic failures. Other models
exhibit context repetition, option echoing, and format inconsistencies, revealing limitations
in deploying sub-2B models for structured RAG without constrained decoding (detailed in
Appendix C.2).
4.2 End-to-End RAG Evaluation
We evaluateB1ade-1Bin realistic RAG settings usingB1ade-embedand stella-400M for
retrieval, with Claude Sonnet 4 as LLM-as-Judge on correctness, completeness, coherence,
and faithfulness. We use LLM-as-Judge-based model evaluation jobs on Amazon Bedrock;
this evaluation motivated our v1 to v2 improvements.3
Table 3: End-to-end RAG evaluation results using B1ade-embed (Claude Sonnet 4 as judge).
Scores range from 0–1 (higher is better).Bolddenotes global best per metric. See Section 4.5
for retriever choice and training data scale ablations.
LLM Embedding K Corr. Comp. Cohe. Faith. Avg.
Baseline Models (no fine-tuning)
Llama-1BB1ade-embed5 0.636 0.549 0.911 0.6390.684
Qwen-1.5BB1ade-embed50.6790.527 0.829 0.680 0.679
B1ade v1 (GPT-NeoX Base, SFT Only)
B1ade v1B1ade-embed5 0.521 0.435 0.760 0.644 0.590
B1ade v2 (Llama-3.2-1B Base, GRPO)
B1ade v2 B1ade-embed5 0.541 0.539 0.8560.6810.654
B1ade v1 (GPT-NeoX + SFT) scores 0.590 average, with low completeness (0.435) and
coherence (0.760). Switching to Llama-3.2-1B and adding GRPO (v2) improves to 0.654
(+10.8%), with largest gains in completeness (+22.8%) and coherence (+12.6%). B1ade v2
approaches Qwen-1.5B (0.679) with 33% fewer parameters but trails in correctness (0.541 vs.
0.679). While Llama-1B base achieves the highest average score (0.684) driven by coherence
(0.911), B1ade v2 leads all models on faithfulness (0.681), the metric most directly reflective
of RAG grounding quality, demonstrating that GRPO training successfully optimizes for
source-grounded generation despite using 33% fewer parameters than Qwen-1.5B. Also see
note in Appendix E on length bias from judge model impacting Llama 1B scores.
4.3 Attribution Emergence Analysis
We conduct head-to-head comparisons using LLM-as-Judge on 6,345 QA pairs from
RAGEval Zhu et al. (2025), comparingB1adev2, Llama-3.2-1B (base), and Qwen-1.5B.
Claude Sonnet 4 ranks outputs based on context fidelity, attribution quality, completeness,
and factual accuracy (see Appendix C.1 for the complete template). This evaluation enables
direct measurement of the emergent attribution phenomenon.
B1ade-1Bachieves 32.2% first-place rate vs. 29.1% for base Llama-1B (+3.1pp) and reaches
76% of Qwen-1.5B’s performance (42.2%) with 67% of its parameters. The key finding is
3Built-in model evaluation metrics: https://docs.aws.amazon.com/bedrock/latest/userguide/
model-evaluation-metrics.html
6

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Table 4: LLM-as-Judge evaluation results ( n= 6,345).Top:Primary rankings showing
B1ade v2 achieves 32.2% first-place rate with our training approach (Llama-3.2-1B base with
GRPO).Bottom:Attribution emergence—B1ade-1B exceeds the training distribution by 5.5
percentage points, representing the main novel contribution of this work.
Rank Model First Place Avg Rank Params
1st Qwen-1.5B42.2%(2,678) 1.77 1.5B
2nd B1ade v232.2%(2,044) 2.11 1B
3rd Llama-3.2-1B29.1%(1,849) 2.08 1B
Training Data Attribution 36.9%
B1ade Eval Attribution42.4%
thatB1ade-1Buses attribution at 42.4%, exceeding the 36.9% rate found naturally in the
training distribution. This shows grounding emerges as an accuracy-maximizing strategy,
not through memorization.
With 6,345 evaluations, the 5.5pp increase (2,689 observed vs. 2,341 expected) is significant
(p< 0.001, binomial test). If the model merely memorized training patterns, we would
expect 36.9% attribution and uniform application. Instead we observe: (1) 42.4% attribution
(+5.5pp above training), (2) specific citations at 2.8% (5.6 ×training rate), and (3) higher
attribution on multi-hop vs. factual questions.
4.4 50K Subset Curation Methodology
To enable fair comparison of GRPO’s effectiveness across different data scales, we created a
clean 50K subset from the 2.2M-example simpleCoT dataset using a four-step pipeline:
1.Normalization:Extract raw question and answer text from dataset-specific XML-
like scaffolding (e.g., question: <...> ,answer: <...> ). Drop malformed examples.
2.Length filtering:Retain only examples where the question contains 20–300 tokens
and the answer contains 10–200 tokens. This eliminates trivial one-word answers
and excessively long documents while maintaining diversity in question complexity.
3.Deduplication:Remove exact duplicate questions by hashing the first 80 characters
of cleaned question text, preserving the first occurrence.
4.Stratified sampling:Partition questions into four equal-sized difficulty bins based
on token length quartiles (easy →hard), then sample uniformly from each bin to
ensure balanced coverage of question complexity. This prevents the subset from
being skewed toward either trivial or overly complex questions.
This curation process reduces the 2.2M raw examples to approximately 500K candidate
examples after filtering, from which we stratified-sample 50K for the scale ablation. The
resulting subset is deterministic (seeded), publicly available on Hugging Face Hub4, and
serves as the training corpus for the smaller model experiments in Section 4.5. Note: 50K
experiments train for 1 epoch, while full 2.2M experiments train for 3 epochs.
4.5 Ablation Studies and Additional Analyses
We present three complementary analyses examining training data scale, retriever choice,
and inference optimization.
Training data scale: 50K vs. 2.2M.To assess how much training data is necessary to
realize GRPO’s benefits, we train Qwen-0.5B and Llama-1B on only 50K examples from the
simpleCoT dataset. Table 5 (RAG) and Appendix Table 11 (standard benchmarks) show that
4https://huggingface.co/datasets/w601sxs/simplecot_subset_50k
7

Preprint:https://huggingface.co/collections/w601sxs/b1ade
standard benchmark performance is substantially lower thanB1ade-1Btrained on the full
2.2M examples, confirming that factual knowledge acquisition requires large-scale training.
GRPO’s effects differ across data scales. On 50K, GRPO improves all Llama-1B metrics
except faithfulness: correctness +0.047 (+12.9%), coherence +0.066 (+18.6%), completeness
+0.003 (+0.8%). On 2.2M, B1ade v2 shows the opposite: only faithfulness improves (+0.042,
+6.6%), while correctness declines -0.095, coherence -0.055, completeness -0.010 relative
to base Llama-1B. With limited data, GRPO improves general quality. With abundant
data, GRPO optimizes exclusively for grounding at the expense of other metrics. This
indicates GRPO’s reward does not scale uniformly. GRPO reduces performance on format-
constrained tasks (FEVER: -13.1pp for Qwen-0.5B), consistent with the reward not penalizing
format deviations. These results suggest decoupling between knowledge acquisition (scale-
dependent) and grounding behavior (learnable from limited data).
Table 5: Training data scale ablation: Impact of data quantity on GRPO effectiveness (RAG
evaluation with B1ade-embed, scores 0–1). Bottom row shows Llama-1B trained on full
2.2M dataset with GRPO (B1ade v2) for comparison.
Model Data Corr. Comp. Faith. Avg
Qwen-0.5B 50K 0.321 0.356 0.393 0.345
Qwen-0.5B GRPO 50K 0.340 0.369 0.425 0.368
Llama-1B 50K 0.364 0.383 0.430 0.383
Llama-1B GRPO 50K 0.411 0.386 0.421 0.407
B1ade v2 (Llama-1B) 2.2M 0.541 0.539 0.681 0.654
Retriever choice (B1ade-embed vs. Stella-400M).Table 6 shows minimal impact from
retriever choice. B1ade v2 scores 0.654 with B1ade-embed vs. 0.652 with Stella-400M (+0.002
difference); B1ade v1 shows 0.590 vs. 0.589. Generation quality is bottlenecked by the
language model, not retrieval at K=5. B1ade-embed is a compute-free alternative to Stella-
400M without quality loss.
Table 6: Retriever ablation: B1ade-embed vs. Stella-400M on B1ade v1 and v2 (scores range
0–1, higher is better).
Model Embedding Corr. Comp. Avg
B1ade v1 B1ade-embed 0.521 0.435 0.590
B1ade v1 Stella-400M 0.526 0.437 0.589
B1ade v2 B1ade-embed 0.541 0.539 0.654
B1ade v2 Stella-400M 0.545 0.542 0.652
Retrieval scope (K=5 vs K=10).We test retrieval scope to assess sensitivity to the number of
retrieved documents. Table 7 compares B1ade v2 with K=5 (0.654 avg) and K=10 (0.466 avg).
Increasing K from 5 to 10 documents provides minimal benefit and degrades performance,
consistent with the observation that language model capacity is the bottleneck. We use K=5
for all reported results.
Table 7: Retrieval scope ablation: B1ade v2 with K=5 vs K=10 (B1ade-embed, scores 0–1).
Model K Corr. Comp. Cohe. Avg
B1ade v2 5 0.541 0.539 0.856 0.654
B1ade v2 10 0.431 0.347 0.566 0.466
Inference optimization: speculative decoding.We test self-speculative decoding for
inference throughput. DoLa decoding reduces PubMedQA accuracy from 65.8% to 63.8%,
8

Preprint:https://huggingface.co/collections/w601sxs/b1ade
while ngram lookup improves it to 68.2%. On FEVER, both methods degrade accuracy by
25% due to repeated token artifacts. Speculative decoding disrupts the attribution patterns
learned during GRPO training.
5 Discussion and Limitations
We organize this discussion around key themes: the emergence and validation of attribution
as a learned strategy, the strategy-execution gap revealed by our evaluations, implications
for efficient RAG development, and limitations with future directions.
B1ade-1B exhibits attribution at 42.4%, exceeding training distribution (36.9%) by 5.5pp.
This validates that grounding emerges as an accuracy-maximizing strategy. However, the
base Llama-3.2-1B may have pre-existing attribution capabilities from distillation or prior
training. We cannot definitively separate GRPO training effects from base model capabilities.
Evidence of learned behavior beyond base capabilities: (1) Specific passage citations (e.g.,
“Passage 1 states...”) appear at 2.8%, 5.6 ×the training rate of 0.5%, with insufficient training
examples to explain via memorization. (2) Attribution rate is higher on multi-hop questions
than factual queries, not uniform across types. (3) Direct quotes appear in 9.8% of responses,
with 494 accurate (38.8%), showing text extraction beyond templates. (4) The model correctly
distinguishes when context supports vs. does not support claims (e.g., recognizing “passage
does not explicitly mention emergence, only predominance”), showing epistemic calibration.
Regarding training limitations, the simpleCoT dataset containing 2.2M examples exhibits
source imbalance, with Kaist contributing 76.6% of training data (Table 9). This may limit
generalization to domains underrepresented in training, such as creative writing or low-
resource languages. Regarding evaluation limitations, despite explicit instructions to avoid
length bias, we observe systematic length-rank correlation (Table 18), with Llama’s longer
responses penalized (correlation of +0.264) and Qwen’s brevity rewarded (correlation of
-0.112). The “attribution paradox,” where B1ade-1B loses to Qwen despite providing identi-
cal answers with explicit attribution (Appendix Cases 1–3), suggests the judge may favor
conciseness over source citation, contradicting stated preferences. All LLM-as-Judge evalua-
tions use Claude Sonnet 4 as a single judge model. Different judge models such as GPT-4
or Llama-3-70B-Instruct may exhibit different biases, potentially altering rankings, and
multi-judge consensus would strengthen validity. Our standard benchmarks in Section 4.1
focus on factual QA and reasoning. We do not evaluate creative generation, dialogue, or
code generation, which are domains where attribution behavior may manifest differently.
6 Conclusion
We introduce a training paradigm for resource-efficient RAG using zero-training model
merging for retrieval and policy optimization on compact language models. Strategic archi-
tectural choices and training achieve competitive performance with reduced computational
requirements. Through this training paradigm We presented B1ade, a resource-efficient
RAG system with two contributions: B1ade-embed, a competitive 335M retriever built via
zero-training model merging that matches Stella-400M in end-to-end RAG (0.654 vs. 0.652)
without any retriever training; and B1ade-1B, a 1B SLM trained with GRPO on 723M tokens
that achieves 81.82% PopQA, 65.8% PubMedQA, and 0.654 average RAG score (LLaaJ) that
approaches Qwen-1.5B with 500M fewer parameters. An emergent property of GRPO train-
ing is that attribution behavior arises through accuracy-only optimization: training with
ROUGE rewards alone produces a 42.4% citation rate, exceeding the training distribution
(36.9%) by 5.5%. Together, these results show that competitive RAG components can be built
without large-scale specialized training, with grounding behavior emerging as a byproduct
of reward optimization rather than explicit supervision.
Future work should investigate whether attribution emergence generalizes across model
scales, training curricula, and reward formulations. Key questions include: (1) Does this
phenomenon persist in larger models, or is it specific to the capacity constraints of 1B-scale
systems? (2) Can alternative reward formulations (e.g., token-level rewards, contrastive
9

Preprint:https://huggingface.co/collections/w601sxs/b1ade
objectives) induce stronger or more reliable grounding behaviors? (3) What training data
characteristics (attribution rate, citation specificity, reasoning chain length) most strongly
influence emergence? Additionally, methods to close the strategy-execution gap with-
out sacrificing parameter efficiency warrant investigation, potentially through knowledge
distillation or hybrid architectures that separate reasoning from knowledge storage.
References
Alex Andonian, Stella Biderman, Sid Black, Preetham Gali, Leo Gao, Eric Hallahan, Josh
Levy-Kramer, Connor Leahy, Lucas Nestler, Kip Parker, et al. Gpt-neox: Large scale
autoregressive language modeling in pytorch.Zenodo, 2023.
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle
O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai
Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models
across training and scaling. InInternational Conference on Machine Learning, pp. 2397–2430.
PMLR, 2023.
Jennifer Chen, Aidar Myrzakhan, Yaxin Luo, Hassaan Muhammad Khan, Sondos Mahmoud
Bsharat, and Zhiqiang Shen. Drag: Distilling rag for slms from llms to transfer knowledge
and mitigate hallucination via evidence and graph-based distillation, 2025. URL https:
//arxiv.org/abs/2506.01954.
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge.arXiv:1803.05457v1, 2018.
Alexander Conway, Debadeepta Dey, Stefan Hackmann, Matthew Hausknecht, Michael
Schmidt, Mark Steadman, and Nick Volynets. syftr: Pareto-optimal generative ai.arXiv
preprint arXiv:2505.20266, 2025.
Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vladimir
Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. Arcee’s MergeKit: A
toolkit for merging large language models. In Franck Dernoncourt, Daniel Preo¸ tiuc-Pietro,
and Anastasia Shimorina (eds.),Proceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing: Industry Track, pp. 477–485, Miami, Florida, US, November
2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-industry.
36. URLhttps://aclanthology.org/2024.emnlp-industry.36.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The
llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a
multi-hop QA dataset for comprehensive evaluation of reasoning steps. InProceedings
of the 28th International Conference on Computational Linguistics, pp. 6609–6625, Barcelona,
Spain (Online), December 2020. International Committee on Computational Linguistics.
URLhttps://www.aclweb.org/anthology/2020.coling-main.580.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models.ICLR,
1(2):3, 2022.
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. PubMedQA:
A dataset for biomedical research question answering. In Kentaro Inui, Jing Jiang, Vin-
cent Ng, and Xiaojun Wan (eds.),Proceedings of the 2019 Conference on Empirical Meth-
ods in Natural Language Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP), pp. 2567–2577, Hong Kong, China, November
2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1259. URL
https://aclanthology.org/D19-1259/.
10

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. triviaqa: A Large Scale
Distantly Supervised Challenge Dataset for Reading Comprehension.arXiv e-prints, art.
arXiv:1705.03551, 2017.
Seungone Kim, Se June Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, and
Minjoon Seo. The cot collection: Improving zero-shot and few-shot learning of language
models via chain-of-thought fine-tuning.arXiv preprint arXiv:2305.14045, 2023.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh,
Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina
Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszko-
reit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering
research.Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi:
10.1162/tacl_a_00276. URLhttps://aclanthology.org/Q19-1026/.
Pierre-Carl Langlais, Pavel Chizhov, Mattia Nee, Carlos Rosas Hinostroza, Matthieu Delsart,
Irène Girard, Othman Hicheur, Anastasia Stasenko, and Ivan P . Yamshchikov. Even small
reasoners should quote their sources: Introducing the pleias-rag model family.ArXiv,
abs/2504.18225, 2025. URLhttps://api.semanticscholar.org/CorpusID:278129318.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474, 2020.
Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat
Lee. Textbooks are all you need ii: phi-1.5 technical report.arXiv preprint arXiv:2309.05463,
2023.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel
Khashabi. When not to trust language models: Investigating effectiveness and limitations
of parametric and non-parametric memories.arXiv preprint, 2022.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text
embedding benchmark. InProceedings of the 17th Conference of the European Chapter of the
Association for Computational Linguistics, pp. 2014–2037, 2023.
Jiaming Qu, Madhu Gopinathan, Shayan Ali Akbar, and Omar Alonso. Interactive taxonomy
development with hybrid methods. 2026.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathe-
matical reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.
Shelly Soffer, Benjamin S Glicksberg, Patricia Kovatch, Orly Efros, Robert Freeman, Alexan-
der W Charney, Girish N Nadkarni, and Eyal Klang. A scalable framework for bench-
marking embedding models for semantic medical tasks.medRxiv, pp. 2024–08, 2024.
Andrejs Sorstkins. Assessing rag and hyde on 1b vs. 4b-parameter gemma llms for personal
assistants integretion, 2025. URLhttps://arxiv.org/abs/2506.21568.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. Asqa: Factoid questions
meet long-form answers. InProceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing, pp. 8273–8288, 2022.
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: a
large-scale dataset for fact extraction and VERification. InNAACL-HLT, 2018.
Cheng Wang, Xinyu Fang, et al. Model stock: All we need is just a few fine-tuned models.
arXiv preprint arXiv:2403.19522, 2024a.
Dong-Hwan Wang, Sima Isber, Jaeho Kim, Minkyu Moon, Dennis Han, Joonmo Choi, and
Changick Kim. Model stock: All we need is just a few fine-tuned models.arXiv preprint
arXiv:2403.19522, 2024b.
11

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani
Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of
large language models.arXiv preprint arXiv:2206.07682, 2022.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong
Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren
Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin
Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize
Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu
Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin
Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong
Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report.arXiv preprint
arXiv:2407.10671, 2024.
Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong Wang. Jasper and stella: distillation of
sota embedding models, 2025. URLhttps://arxiv.org/abs/2412.19048.
Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. Tinyllama: An open-source
small language model.arXiv preprint arXiv:2401.02385, 2024.
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen,
Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained
transformer language models.arXiv preprint arXiv:2205.01068, 2022.
Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing Wang,
Shuo Wang, Yishan Li, Nan Zhang, et al. Rageval: Scenario specific rag evaluation dataset
generation framework. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp. 8520–8544, 2025.
12

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Appendix
A Dataset Details
A.1 Source Datasets
The simpleCoT dataset aggregates seven diverse source datasets totaling 2,214,941 examples
across multiple reasoning domains, including multi-hop question answering (HotpotQA
Distractor and FullWiki variants), general instruction following (Alpaca cleaned), mathe-
matical reasoning (Math QA), commonsense narrative understanding (Cosmos QA), diverse
instruction patterns (GPTeacher), and evolved complex instructions (Wizard Evol V2). The
data processing pipeline consists of three key stages: First, all training splits are loaded
directly from their respective Hugging Face repositories using the datasets library. Second,
format unification is performed by casting all datasets to a standardized Kaist Kim et al.
(2023) schema comprising three fields: context (system instructions and factual passages),
question (user query embedded within context), and ground truth (rationale combined
with final answer) ensuring consistent structure across heterogeneous sources. Third, the
unified datasets are concatenated, shuffled with a fixed random seed (42) for reproducibility,
and split into training and testing subsets using an 80/20 ratio, yielding 1,771,953 training
examples and 442,988 test examples. This systematic approach preserves the diversity of
reasoning types while enabling standardized model training and evaluation across the
entire collection.
Dataset Samples % of Total
HotpotQA (Distractor + FullWiki) 180K 8.1%
Wizard Evol Instruct V2 196K 8.8%
Alpaca (cleaned) 52K 2.3%
Math QA 37K 1.7%
GPTeacher 30K 1.4%
Cosmos QA 25K 1.1%
Kaist (base format) 1,695K 76.6%
Total 2,214,941 100.0%
Table 8: Composition of the simpleCoT training dataset. Kaist provides the base format for
unification.
Source Samples % of Total
Wizard Evol Instruct V2 196,000 8.8%
HotpotQA (combined) 180,000 8.1%
Alpaca (cleaned) 52,000 2.3%
Math QA 37,000 1.7%
GPTeacher 30,000 1.4%
Cosmos QA 25,000 1.1%
Kaist (diverse) 1,694,941 76.6%
Table 9: Training data distribution across source datasets. Kaist contains 139 diverse sub-
tasks.
A.2 Standard Benchmark Results
Benchmarks:
•PopQA(Mallen et al. (2022)): Popular entity questions
•TriviaQA(Joshi et al. (2017)): Trivia-style factual questions
•Natural Questions (NQ)(Kwiatkowski et al. (2019)): Google search queries
•WikiMultihopQA(Ho et al. (2020)): Multi-hop Wikipedia reasoning
13

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Metric Value
Total Samples 2,214,941
Source Datasets 7
Training Split 1,771,953 (80%)
Test Split 442,988 (20%)
Generic Attribution 36.9%
Specific Citation 0.5%
No Attribution 63.1%
Fine-tuning Tokens 723M
Shuffle Seed 42
Table 10: Summary statistics forsimpleCoTcollection.
Figure 1: B1ade performance on standard benchmarks
•FEVER(Thorne et al. (2018)): Fact extraction and verification
•ASQA(4.4K) (Stelmakh et al. (2022)): Ambiguous questions with long-form answers
•ARC-Challenge(Clark et al. (2018)): Science questions (AI2 Reasoning Challenge)
•PubMedQA(Jin et al. (2019)): Biomedical yes/no/maybe questions (500 samples)
A.3 Note on low/zero scores
A striking finding across our RAG benchmarks was the widespread failure of small
language models to adhere to basic output format constraints, even with extensive
prompt engineering. This phenomenon manifested in several distinct failure modes that
14

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Model Data PopQA TQA NQ ASQA FEVER ARC PMQA WikiMQA
Qwen-0.5B 50K 7.90 8.60 4.20 6.00 20.5374.807.40 6.01
Qwen-0.5B
GRPO50K 8.82 11.60 4.20 5.40 7.39 72.00 10.30 7.06
Llama-1B 50K 9.60 25.00 11.60 13.40 14.26 52.60 31.40 9.56
Llama-1B
GRPO50K 11.20 25.00 11.60 13.40 15.40 52.60 32.60 13.20
Table 11: Standard benchmark results (Exact Match %) for smaller models trained on 50K
subset. GRPO produces modest gains on most tasks but drops on FEVER and ARC for
Qwen-0.5B, consistent with the reward not penalizing format deviations.
fundamentally undermined evaluation validity. Most notably, we observed systematic
"option repetition" behavior where models would output all possible answer choices rather
than selecting one which is a trivial strategy that artificially inflates performance metrics.
For instance, on PubMedQA, Llama 3.2 1B repeated all three options ("yes," "no," "maybe")
in 321 of 500 test cases, creating false positive matches that initially suggested 80% accuracy.
Only after implementing strict single-answer extraction did we observe the true baseline
performance.
Catastrophic Failures and Zero-Score Phenomena:Several model-benchmark combi-
nations produced zero or near-zero scores despite iterative prompt refinement, revealing
fundamental limitations in instruction-following capacity for certain small models. On
ARC-Challenge, Phi 1.5, Phi 3 Mini, and TinyLlama all achieved 0% exact match accuracy,
while Llama 3.2 1B managed only 0.09% (1 correct answer out of 1,172 questions). These
failures persisted even when simplifying prompts, adding explicit constraints, or restructur-
ing inputs as multiple-choice questions. Particularly illustrative is the case of Pythia 1.4B on
PubMedQA, where the model frequently generated only a period (".") with no mention of
the required answer options. When constrained through template-based generation, the
model instead repeated instructions, specifically outputting "choose between the following
three options and only print the correct answer from ’yes’" in 276 of 500 cases. This repetition
artificially boosted accuracy to 44%, which collapsed to 9% when these echoed instructions
were removed from evaluation.
OPT 1.3B exhibited similar catastrophic behavior across multiple benchmarks, achieving 0%
on WikiMultihopQA despite various prompt formulations. On FEVER, OPT repeated the
entire context verbatim for every test instance under the original prompt, necessitating a
complete reformulation. Even after simplification, the model frequently output the system
prompt itself as the answer.
Model-Specific Failure Modes and Prompt Sensitivity:Each model exhibited idiosyn-
cratic responses to prompt engineering, requiring bespoke optimization strategies that often
contradicted general best practices. Llama 3.2 1B required verbose, markdown-formatted
instructions to trigger chain-of-thought reasoning on FEVER, but this same verbosity caused
context repetition in other models. Interestingly, while chain-of-thought generation im-
proved Llama’s answer quality in some cases, it frequently failed to reach a conclusion
producing multi-step reasoning that never selected between "SUPPORTS" or "REFUTES."
Additionally, Llama often formatted answers using LaTeX notation (e.g., "The final answer
is:SUPPORTS "), complicating answer extraction. In the most problematic cases, Llama
outputs contained both "SUPPORTS" and "REFUTES" within extended self-contradictory
reasoning; penalizing these double-matches reduced FEVER accuracy from 49% to 23%.
Qwen 2.5 1.5B demonstrated extreme sensitivity to prompt format. Markdown-structured
prompts triggered full context repetition, while simpler templates produced refusal-like out-
puts that restated the claim in negative form (e.g., "Refutes: The Ukrainian Soviet Socialist
Republic was not a founding participant of the UN. The Soviet Union was a founding partic-
ipant of the UN."). Most surprisingly, using Qwen’s documented message format with role
15

Preprint:https://huggingface.co/collections/w601sxs/b1ade
tags degraded performance, causing the model to output repetitive Unicode characters (" t,i
t,it,i..."). Only after converting FEVER to explicit multiple-choice format did Qwen produce
evaluable outputs. Phi 1.5 exhibited a distinct hallucination pattern: when presented with
two multiple-choice options (A and B), it systematically "completed" the list to four options
(adding hallucinated C and D choices) before selecting an answer. For example:
Choose only one option -
A.'SUPPORTS'
B.'REFUTES'.
Result: C.'NEITHER'.
D.'MAYBE'.
Answer:
A.'SUPPORTS'
This behavior increased from 15.98% accuracy (when evaluating the full response) to 24.88%
when restricting matches to text following "Answer:", suggesting the model reliably an-
swered correctly after completing its hallucinated option set.
Evaluation Methodology Implications:These findings necessitated substantial refinements
to evaluation methodology. Initial exact-match scoring proved inadequate, as it could be
trivially gamed through option repetition or context echoing. We implemented several
countermeasures:
1.Positional answer extraction: Restricting match detection to text following explicit
markers like "Answer:" reduced false positives from hallucinated reasoning chains.
2.Penalty for ambiguity: Responses containing multiple contradictory answers (e.g.,
both "yes" and "no") were marked incorrect, even if one match was correct.
3.Soft matching: For datasets like WikiMultihopQA, we added fuzzy matching
to account for semantically equivalent but lexically different answers, revealing
substantial performance gaps between strict (7.56% for B1ade) and soft matching
(13.68% F1).
4.Context truncation strategies: CUDA OOM errors for longer contexts (e.g., FEVER
Wikipedia articles on OPT) required systematic truncation, introducing an un-
controlled variable that may have degraded performance for context-dependent
questions.
The most concerning implication is that standard benchmark scores may systematically
overestimate small model capabilities when evaluation doesn’t account for these failure
modes. The PubMedQA case where naive evaluation suggested 80% accuracy that dropped
to actual baseline performance after proper answer extraction demonstrates how easily
benchmarks can be inadvertently "gamed" by models that exploit evaluation weaknesses
rather than demonstrating genuine comprehension.
Performance Variability and Model Selection:Despite uniform prompt engineering efforts,
model performance varied dramatically across benchmarks in ways that defy simple model-
size predictions. On ARC-Challenge, Qwen 2.5 1.5B achieved 71.93% accuracy while larger
models like OPT 1.3B achieved only 22.95%. Similarly, on PopQA, Pythia 1.4B (73.44%) sub-
stantially outperformed the smaller TinyLlama (64.32%), but this relationship inverted on
WikiMultihopQA where TinyLlama’s exact match score (0.23%), while abysmal, exceeded
Pythia’s 0.06%. These inconsistencies suggest that benchmark performance for small models
depends more on alignment between model training distribution and task format than
on general reasoning capacity. Qwen’s strong ARC performance likely reflects exposure
to similar multiple-choice formats during training, while its failure on FEVER (requiring
the messages format) indicates brittleness when presented with unexpected input structures.
Broader Implications for RAG Systems:The systematic instruction-following failures docu-
mented here pose significant challenges for deploying small models in retrieval-augmented
generation systems. The zero-score phenomenon suggests that below a certain capability
16

Preprint:https://huggingface.co/collections/w601sxs/b1ade
threshold, prompt engineering alone cannot induce reliable structured output genera-
tion. For production RAG systems requiring predictable output formats (e.g., JSON for
API responses), these models would require either constrained decoding or downstream
post-processing both adding latency and complexity that partially negate the efficiency
advantages of smaller models. Moreover, the model-specific prompt optimization required
for even basic functionality undermines the modularity of RAG architectures. A system
designed around Llama 3.2 1B’s verbose markdown preferences would fail catastrophi-
cally if swapped to Qwen without prompt reengineering. This brittleness suggests that
small model RAG systems may require maintaining model-specific prompt templates and
evaluation pipelines, increasing maintenance burden.
B B1ade-embed Additional Details
B.1 MTEB rankings and performance at tasks
In the current MTEB dashboard,B1ade-embedachieves the following rankings:
• #4 overall (Borda ranking)
• #1 on Retrieval overall
• #4 on Pair Classification and Classification
• #3 on ASKUBUNTUDUPQUESTIONSand MTOPDOMAINCLASSIFICATION
•#2 on ARXIVHIERARCHICALCLUSTERINGS2S, ARGUANA, and SPRINTDUPLICATE-
QUESTIONS
•#1 on BIOSSES, BANKING77CLASSIFICATION, CQADUPSTACKGAMINGRE-
TRIEVAL, MINDSMALLRERANKING, SCIDOCS
• High performing on several STS tasks (mean: 78.99, STS: 87.07)
Model Avg Class Cluster Pair Rerank Retrieval STS
Stella-400M-v570.11 86.67 56.70 87.74 60.16 58.9784.22
learning2_model65.3977.75 47.96 84.53 58.5057.9181.43
gte-large-en-v1.565.3977.75 47.96 84.53 58.5057.9181.43
cde-small-v1 65.0081.71 48.3284.69 56.75 53.27 81.63
mxbai-embed-large-v1 64.68 75.64 46.71 87.2060.1154.3985.00
UAE-Large-V1 64.64 75.58 46.7387.2559.88 54.66 84.54
bge-large-en-v1.5 64.23 75.97 46.08 87.12 60.03 54.29 83.11
B1ade-embed (ours)64.21 75.16 46.46 87.07 60.00 53.3085.04
Table 12: B1ade-embed MTEB leaderboard results (56 datasets, all tasks). At the time
of writing, B1ade-embed ranked #9 overall among 400M-parameter models, competitive
with leading retrieval models despite zero-training design via parameter fusion. Bold
indicates best performance per column. Scores on Classification (12 datasets), Clustering (11),
PairClassification (3), Reranking (4), Retrieval (15), and STS (10) tasks shown as averages.
Table 12 shows that Stella-400M-v5 ranks first across six of seven MTEB categories, making
it the strongest available retriever in the sub-500M class. This is why we selected Stella-
400M-v5 as the comparison point for the retriever ablation in Section 4.5: it represents
the strongest available retrieval baseline, and demonstrating that B1ade-embed performs
comparably in end-to-end RAG (0.654 vs. 0.652 average) against it is a strong validation
of our zero-training model merging approach. The only category where B1ade-embed
outperforms Stella is STS (85.04 vs. 84.22), which is directly relevant to RAG retrieval quality
— measuring semantic similarity between query and passage. This suggests that despite
lower overall MTEB scores, B1ade-embed is well-suited for RAG-specific retrieval tasks.
On domain-specific leaderboards (sub-500M class):
• Top-3 onCodeSearchNetRetrieval
17

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Figure 2: Embedding model performance on legacy MTEB leaderboard for the sub-500M
category.
• Top-5 on legal-text retrieval
• Top-2 by mean score on medical benchmarks
• #1 onNFcorpus
B.2 Knowledge Distillation Experiments
We also experimented with distilling knowledge from Alibaba-NLP/gte-large-en-v1.5
(434M parameters) into b1ade-embed . Training used AllNLI and English Wikipedia text for
30 epochs (LR = 5e-5, batch size = 64, FP16) with PCA dimensionality reduction followed
by dense projection. The distilled model showed targeted gains in specific MTEB subsets,
However, average scores dropped across all MTEB categories. All subsequent experiments
use the baseb1ade-embedmodel without distillation.
Table 13: Knowledge distillation results on selected MTEB subsets. The distilled model
shows targeted improvements on specific tasks, though average scores dropped across all
MTEB categories.
MTEB Subset Before (%) After (%) Improvement Metric
ToxicConversationsClassification 67.91 75.01 +7.10 Accuracy
TweetSentimentExtractionClassification 59.22 61.8 +2.58 Accuracy
MedrxivClusteringS2S 31.7 34.02 +2.32 Validity Measure
ClimateFEVER 30.37 31.95 +1.58 nDCG@10
18

Preprint:https://huggingface.co/collections/w601sxs/b1ade
C LLMaaJ and Attribution Analysis Examples
C.1 Complete Judge Template
We use Claude Sonnet 4 as the judge to rank model outputs. The full prompt template is:
Judge Template
You are evaluating three model responses to a question. The question includes context
passages and asks for a specific answer.
Your task:Rank the three responses from best (Rank 1) to worst (Rank 3) based on these
criteria, in order of importance:
1. Context Fidelity (HIGHEST PRIORITY)
• Does the response accurately use only the provided context?
• Are there any hallucinations or fabricated information?
• Does it misquote or misrepresent the passages?
2. Attribution Quality
• Does the response cite or reference the source passages?
• Are claims grounded in specific context?
• Note: Prefer responses that acknowledge sources
3. Completeness
• Does it fully answer the question?
• Are all relevant details from context included?
•DO NOT prefer brevity over completeness
4. Factual Accuracy
• Is the core answer correct based on the context?
• Does it identify the right entity/relationship/fact?
Output Format:
Model A: Rank [1/2/3]
Model B: Rank [1/2/3]
Model C: Rank [1/2/3]
Reasoning: [Brief explanation of ranking decision]
C.2 Example Evaluations
Example 1: B1ade-1B Wins with Correct Nuance
Context (Excerpt)
The Asian genotype became the predominant genotype circulating worldwide in 2007...
Question
When did the Asian genotype emerge?
Model Outputs:
Llama-1B (Rank 3)
The Asian genotype emerged in Cambodia between 2001 and 2007.
19

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Qwen-1.5B (Rank 2)
The Asian genotype emerged in Indochina around 2007.
B1ade-1B (Rank 1)✓
Based on the passage, the Asian genotype became the predominant genotype in 2007.
However, the passage does not explicitly mention when the Asian genotype first
emerged, only when it became predominant.
Judge Reasoning:“B1ade-1B correctly identifies that the passage describes when the
genotype becamepredominant, not when itemerged. The question asks about emergence, but
the context only provides information about predominance. B1ade’s careful distinction and
epistemic humility (acknowledging the passage doesn’t say when it emerged) demonstrate
superior context fidelity. Ranks: B1ade=1, Qwen=2, Llama=3.”
Analysis:B1ade wins by understanding the nuance between “emerged” and “became
predominant.” This is the most common winning pattern (48% of wins).
Example 2: B1ade-1B Loses from Factual Error
Context (Excerpt)
Hemagglutinin (HA) is the major surface and attachment glycoprotein on influenza virus...
Neuraminidase (NA) is another surface glycoprotein that facilitates viral release.
Question
Which is the major surface and attachment glycoprotein on influenza virus?
Model Outputs:
Llama-1B (Rank 1)
Hemagglutinin (HA)
Qwen-1.5B (Rank 1)
hemagglutinin (HA)
B1ade-1B (Rank 3)
Based on the passage, the NA glycoprotein is the major surface and attachment
glycoprotein on influenza virus.
Judge Reasoning:“Context explicitly states: ’Hemagglutinin (HA) is the major surface and
attachment glycoprotein.’ B1ade-1B’s answer of NA (neuraminidase) is factually incorrect.
Despite proper attribution (“Based on the passage”), the core answer is wrong. Llama and
Qwen both correctly identify HA. Ranks: Llama=1, Qwen=1, B1ade=3.”
Analysis:B1ade-1B loses from factual error (56% of losses), not from attribution strategy.
The model correctly attributes but provides the wrong glycoprotein.
Example 3: B1ade-1B Wins Despite Attribution Paradox
20

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Question
What percentage of patients developed acute liver injury?
Model Outputs:
Qwen-1.5B (Rank 2)
14-53% of patients with COVID-19 developed acute liver injury.
B1ade-1B (Rank 1)
According to the passage, 14-53% of patients with COVID-19 developed acute liver
injury, with elevated levels of ALT and AST.
Judge Reasoning:“Both provide the correct range (14-53%). B1ade adds useful clinical
detail (elevated ALT/AST) and explicitly attributes to the passage. More complete response.
Ranks: B1ade=1, Qwen=2.”
D Attribution Analysis:
B1ade-1B wins when attribution is combined with additional relevant detail from context.
This shows attribution can provide value when execution is correct.
D.1 Detection Keywords
Generic Attribution Phrases:
• “based on”, “according to”, “the passage”, “the context”
• “as mentioned”, “as stated”, “from the passage”, “the article”
• “as described”, “in the passage”, “the text states”
Specific Citation Phrases:
• “passage 1”, “passage 2”, “passage 3”
• “first passage”, “second passage”, “third passage”
• “title:”, “paragraph”
Detection Method:Case-insensitive substring matching. A response is classified as “at-
tributed” if it contains any generic attribution phrase. If it also contains a specific citation
phrase, it’s classified as “specifically cited.”
D.2 Quote Accuracy Measurement
We measure quote accuracy using fuzzy string matching:
fromdifflibimportSequenceMatcher
defcompute_quote_accuracy(quote, passages):
"""
␣␣␣␣Check␣if␣quote␣appears␣in␣any␣passage.
␣␣␣␣Returns␣accuracy␣category.
␣␣␣␣"""
max_overlap = 0.0
forpassageinpassages:
# Normalize both strings
21

Preprint:https://huggingface.co/collections/w601sxs/b1ade
quote_norm = normalize(quote)
passage_norm = normalize(passage)
# Compute sequence similarity
similarity = SequenceMatcher(
None, quote_norm, passage_norm
).ratio()
max_overlap =max(max_overlap, similarity)
# Classify
ifmax_overlap >= 0.8:
return"exact" # High overlap
elifmax_overlap >= 0.5:
return"partial" # Medium overlap (paraphrase)
else:
return"inaccurate" # Low overlap (misquote)
Thresholds:
•≥80%: Exact or very close quote
• 50-80%: Partial quote or paraphrase
•<50%: Inaccurate or fabricated
D.3 Attribution Analysis Details
D.3.1 Manual Pattern Analysis
We manually analyzed 50 cases (25 wins, 25 losses) to identify qualitative patterns.
Why B1ade Wins:
Pattern % of Wins Description
Correct Nuance48% Understands what question actually
asks
Complete Answer 16% Provides comprehensive response
Least Wrong 16% All models struggle; B1ade closest
Direct Quote 13% Includes exact passage quote
Good Attribution 6% Clear source citation helps
Table 14: Patterns in B1ade winning cases.
Why B1ade Loses:
Pattern % of Losses Description
Factually Wrong56%Incorrect answer to question
Incomplete 20% Missing key information
Misunderstood Question 12% Answered different question
Over-Interpretation 12% Added unsupported inference
Fabricated Sources 0%Made up citations
Table 15: Patterns in B1ade losing cases. Critical finding: B1ade loses from factual errors
(56%), NOT from fabrication (0%).
D.3.2 Attribution results
To illustrate the attribution patterns observed in the training data, we present three rep-
resentative examples from the ground truth responses. These examples demonstrate the
22

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Metric B1ade Qwen Llama Training
Attribution Usage 42.4%40.9% 35.4% 36.9%
Total Attributions 2,689 2,595 2,243 —
Good Attributions 1,825 2,1261,624 —
Bad Attributions 592 227 380 —
Success Rate67.9% 81.9%72.4% —
Table 16: Complete attribution analysis across models.
Figure 3: Strategy-execution gap in attribution behavior. (a) B1ade demonstrates the highest
attribution usage rate (42.4%), exceeding both the training baseline (36.9%, red dashed line)
and other models, validating superior strategic application. (b) However, Qwen achieves
the highest success rate (81.9%) when attributing, revealing an execution quality gap.
types of attribution behaviors models learn to emulate. Example 1 and 3 show explicit
source attribution using phrases like “the article states that” and “the context describes,”
while Example 2 represents the majority case (63.1%) where responses provide information
without explicit attribution markers.
Example 1: With Generic Attribution (36.9%)
Ground Truth:“The article states that many medicinal and recreational drugs, such as
tetrahydrocannabinol (active ingredient in cannabis), caffeine, morphine and nicotine come
directly from plants. These are some examples of the medicines found in plants mentioned
by the author.”
Example 2: No Attribution (63.1%)
Ground Truth:“The Battle of Appomattox Court House was a battle in the final stages of the
American Civil War, resulting in Confederate General Robert E. Lee surrendering his Army
to Union Commander Ulysses S. Grant on April 9th 1865.”
Example 3: With Context Reference (36.9%)
Ground Truth:“The context describes events and details of his life, such as how he met Edna
Stillwell in Vincennes and their early careers, the ‘Doughnut Dunkers’ routine that brought
him recognition...”
E Evaluation Bias Analysis
We identify several systematic biases in judge behavior:
23

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Metric B1ade Qwen Llama
Total Quotes 1,274851 909
Quote Usage % 9.8%7.0% 7.4%
Accurate Quotes 494419 325
Quote Accuracy % 38.8%49.2%35.8%
Rank Improvement +0.105+0.098 -0.184
Table 17: Quote usage and effectiveness. B1ade quotes most frequently and has most
accurate quotes in absolute terms (494).
Model Avg Length Length-Rank Correlation Bias
B1ade 538 chars +0.084 Slight penalty
Llama 500 chars +0.264 Strong penalty
Qwen 322 chars -0.112 Slight reward
Table 18: Length bias analysis. Positive correlation = longer responses receive worse ranks.
E.1 Length Bias
Despite explicit instruction “DO NOT prefer brevity over completeness,” we observe corre-
lation between response length and worse ranks for some models. For example Llama’s
longer responses are penalized despite judge instructions.
E.2 Attribution Paradox
We manually found 10+ cases where B1ade explicitly attributed the right excerpt but lost to
Qwen without attribution, contradicting the stated preference for source citation.
Example Case 1:
Question
How many antigens could Liew’s ELISA detect?
Qwen (Rank 1 - No Attribution)
Liew validated one multiplex ELISA for the detection of 9 antigens.
B1ade (Rank 2 - With Attribution)
Based on the passage, Liew’s multiplex ELISA test could detect 9 antigens.
Both responses are factually identical; B1ade adds attribution phrase. Judge ranks Qwen
higher despite stated preference for attribution.
Example Case 2:
24

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Question
It continues to be a challenge for clinicians to identify preterm infants likely to experi-
ence subsequent neurodevelopmental deficits. The Test of Infant Motor Performance
(TIMP) and the assessment of spontaneous general movements (GMs) are the only re-
liable diagnostic and predictive tools for the functionality of the developing nervous
system, if applied before term.AIM: To determine to what extent singular preterm
assessments of motor performance can predict the neurodevelopmental outcome
in 14-month olds. The assessment of the quality of general movements (GMs) in
young infants is a reliable and valid diagnostic tool for detecting brain dysfunction
early in life. Poor repertoire GMs are the most frequently observed abnormal GMs
during the preterm, term and early postterm period. However, their predictive value
for the neurological outcome is low.AIM: To find out whether a detailed scoring of
poor repertoire GMs might lead to a better prediction of the neurological outcome.
To assess the clinical value of a modified version, not employing video recording,
of Precthl’s method on the qualitative assessment of general movements (GMs) in
preterm, term and young infants at neurological risk.Is it possible to predict the
infant’s neurodevelopmental outcome at 14 months of age by means of a single
preterm assessment of General Movements?
Qwen (Rank 1 - with Attribution)
Yes, according to Passage #1, it is possible to predict the infant’s neurodevelop-
mental outcome at 14 months of age using a single preterm assessment of General
Movements (GMs).
B1ade (Rank 2 - With Attribution)
Based on the provided context and the details given about the Test of Infant Motor
Performance (TIMP) and the assessment of spontaneous general movements (GMs),
it appears that the current approach is to use a single assessment of general move-
ments to predict neurodevelopmental deficits in preterm infants at 14 months of age.
The assessment of the quality of general movements (GMs) in young infants is a
reliable and valid diagnostic tool for detecting brain dysfunction early in life, but its
predictive value for neurological outcome is low. To address the question of whether
it is possible to predict the infant’s neurodevelopmental outcome at 14 months of age
by means of a single preterm assessment of General Movements, the answer is: No,
it is not possible to predict the infant’s neurodevelopmental outcome at 14 months
of age by means of a single preterm assessment of General Movements.
In this case Qwen’s answer is in fact wrong, despite pointing to the passage. B1ade correctly
answers this nuanced question very similar to the ground truth:
25

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Ground truth
Based on the context provided, it seems that the ability to predict an infant’s neu-
rodevelopmental outcome at 14 months of age using a single preterm assessment
of General Movements (GMs) is limited. The key points are: 1. The assessment
of spontaneous GMs is one of the only reliable diagnostic and predictive tools for
the functionality of the developing nervous system in preterm infants, if applied
before term. 2. However, the context states that "Poor repertoire GMs are the most
frequently observed abnormal GMs during the preterm, term and early postterm
period. However, their predictive value for the neurological outcome is low." 3.
The first study mentioned aimed to determine to what extent singular preterm as-
sessments of motor performance can predict the neurodevelopmental outcome in
14-month-olds. This suggests that a single preterm GM assessment may have limited
predictive value for 14-month neurodevelopmental outcomes. Therefore, based on
the information provided, it does not seem possible to reliably predict a preterm
infant’s neurodevelopmental outcome at 14 months of age using a single preterm
assessment of General Movements alone. The predictive value of this assessment
appears to be limited, and multiple assessments over time may be needed for more
accurate predictions of longer-term neurodevelopmental outcomes.
Example Case 3:
Question
Kodiak Island (Alutiiq: "Qikertaq", ) is a large island on the south coast of the U.S.
state of Alaska, separated from the Alaska mainland by the Shelikof Strait. The
largest island in the Kodiak Archipelago, Kodiak Island is the second largest island
in the United States and the 80th largest island in the world, with an area of , slightly
larger than Cyprus. It is long and in width ranges from . Kodiak Island is the
namesake for Kodiak Seamount, which lies off the coast at the Aleutian Trench. The
largest community on the island is the city of Kodiak, Alaska. The U.S. state of Florida
has a total of 4,510 islands that are ten acres or larger. This is the second-highest
number of islands of any state of the United States; only Alaska has more. Kings
Island is a amusement park located northeast of Cincinnati in Mason, Ohio. Owned
and operated by Cedar Fair, the park first opened in 1972 by the Taft Broadcasting
Company. It was part of a larger effort to move and expand Coney Island, a popular
resort destination along the banks of the Ohio River that was prone to frequent
flooding. After more than $300 million in capital investments, the park features over
100 attractions including fourteen roller coasters and a water park.How large is
King’s Island?
Qwen (Rank 3 - hallucination, overconfidence)
King’s Island is approximately 100 acres in size.
B1ade (Rank 2 - brief answer with attribution)
The answer to the question "How large is King’s Island?" is not explicitly mentioned
in the given text.
In this case Qwen’s answer is hallucinated. B1ade correctly and briefly answers this question.
Here the ground truth is more explanatory but matches B1ade’s answer.
26

Preprint:https://huggingface.co/collections/w601sxs/b1ade
Ground truth
The first context discusses the physical characteristics of the King Island emu, which
provides information about the size of the bird’s bones, but does not relate to the
size of the amusement park. The second context describes King’s Island as an
amusement park with over 100 attractions, including fourteen roller coasters and a
water park. However, it does not provide any information about the physical size
of the park.Therefore, there is no available information to provide a direct answer
to the question "How large is King’s Island?" based on the two given contexts. [No
reference available].
F B1ade-embed: Mergekit Configuration Details
F.1 Model Stock Merging Strategy
B1ade-embedemploys theModel Stockalgorithm Wang et al. (2024b), which optimizes linear
interpolation weights through geometric properties of task vectors. Model Stock provides
a principled, data-driven approach to combining multiple pretrained models without
requiring additional training, making it ideal for efficient model composition.
F.1.1 Merging Configuration
The merge configuration uses Mergekit5Goddard et al. (2024), an open-source toolkit for
parameter-efficient model composition:
•Merge Method: model_stock — geometric weight optimization for linear interpola-
tion
•Base Model: bert-large-uncased — serves as reference for task vector computation
and architecture anchor
•Auxiliary Models: Four MTEB-optimized dense retrievers:
–WhereIsAI/UAE-Large-V1 — specialized in multilingual dense retrieval and
cross-language transfer
–BAAI/bge-large-en-v1.5 — optimized for cross-lingual semantic matching
and zero-shot retrieval
–mixedbread-ai/mxbai-embed-large-v1 — instruction-following embeddings
for task-aware representations
–avsolatorio/GIST-large-Embedding-v0 — general-purpose semantic similar-
ity and broad domain coverage
•Output Dimension: 1024 (matching BERT-Large architecture)
•Precision: float32 for numerical stability when combining diverse embedding
spaces
•Tokenizer: Unified tobert-large-uncasedtokenizer (30,522 vocabulary)
F.1.2 Model Stock Algorithm
Model Stock computes optimized interpolation weights using the following procedure:
1.Task Vector Extraction: For each auxiliary model i, compute the task vector as the
difference from the base model:
τi=θ i−θ base (1)
whereθ iandθ baseare the flattened parameter vectors.
5https://github.com/arcee-ai/mergekit
27

Preprint:https://huggingface.co/collections/w601sxs/b1ade
2.Pairwise Similarity Computation: Calculate cosine similarities between all pairs of
task vectors to measure model coherence:
sij=τi·τj
∥τi∥ · ∥τ j∥(2)
3.Interpolation Weight Derivation: The interpolation factor tis computed as the
mean of pairwise similarities, which reflects how well-aligned the auxiliary models
are:
t=1
(n
2)∑
i<jsij (3)
4.Weighted Linear Merge: The final merged parameters are computed as a weighted
interpolation between the base model and the average of auxiliary models:
θmerged =t· ¯θaux+ (1−t)·θ base (4)
where ¯θaux=1
n∑n
i=1θiis the arithmetic mean of auxiliary model parameters.
This adaptive weighting mechanism automatically balances the contribution from all aux-
iliary models based on their mutual alignment, avoiding manual hyperparameter tuning
while maintaining base model coherence throughout the merged architecture.
F.1.3 Model Selection Rationale
The five models (including base) were selected based on complementary MTEB performance
characteristics:
•Diversity: Each model excels on different MTEB task types (retrieval, similarity,
clustering, classification, reranking), ensuring broad semantic coverage
•Efficiency: All models remain below 1024-dimensional output and compatible with
BERT-Large architecture, enabling parameter-free composition
•Performance: Collectively rank in the top tier of MTEB leaderboard within the
sub-500M parameter constraint at the time of the experiment
•Complementarity: Task vector analysis revealed high diversity (low pairwise simi-
larity) while remaining semantically coherent, indicating each model contributes
unique knowledge
The resulting 335M-parameter merged model inherits representational capabilities from all
five input models without requiring additional pretraining, demonstration, or fine-tuning,
making it highly resource-efficient for downstream RAG applications.
28