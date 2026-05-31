# Analyzing Quality-Latency-Resource Trade-offs in a Technical Documentation RAG Assistant Using LoRA Adaptation

**Authors**: Evgenii Palnikov, Elizaveta Gavrilova

**Published**: 2026-05-27 09:37:55

**PDF URL**: [https://arxiv.org/pdf/2605.28222v1](https://arxiv.org/pdf/2605.28222v1)

## Abstract
We study quality-latency-resource trade-offs in a documentation-grounded retrieval-augmented generation (RAG) system that uses Low-Rank Adaptation (LoRA) of the generator. We build a manually verified benchmark of 5,144 question-answer pairs over the official Kubernetes documentation and combine it with a fixed hybrid-retrieval pipeline (BGE-M3 dense, BGE-M3 native sparse, Reciprocal Rank Fusion, cross-encoder reranking). Over this benchmark we ablate 20 LoRA configurations on Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct across rank and target-module choices, and evaluate each on token-level F1, LLM-judged groundedness and correctness (pass@4), inference latency, inference memory, and training cost, all reported with bootstrap 95% confidence intervals. Pareto analysis shows that LoRA adapters acting only on the q and v attention projections consistently dominate the front, while the 3B/8B choice mainly defines operating regime. A param-matched control comparison further indicates that the q/v advantage is structural rather than purely parametric. The benchmark, selected adapters, and code are available at https://github.com/EugPal/rag-lora-tradeoffs.

## Full Text


<!-- PDF content starts -->

Preprint
ANALYZINGQUALITY-LATENCY-RESOURCETRADE-
OFFS IN ATECHNICALDOCUMENTATIONRAG ASSIS-
TANTUSINGLORA ADAPTATION
Evgenii Palnikov1, Elizaveta Gavrilova1
1HSE University
ABSTRACT
We study quality–latency–resource trade-offs in a documentation-grounded
retrieval-augmented generation (RAG) system that uses Low-Rank Adapta-
tion (LoRA) of the generator. We build a manually verified benchmark of
5,144 question–answer pairs over the official Kubernetes documentation and
combine it with a fixed hybrid-retrieval pipeline (BGE-M3 dense, BGE-M3 na-
tive sparse, Reciprocal Rank Fusion, cross-encoder reranking). Over this bench-
mark we ablate 20 LoRA configurations on Llama-3.2-3B-Instruct and
Llama-3.1-8B-Instruct across rank and target-module choices, and evalu-
ate each on token-level F1, LLM-judged groundedness and correctness (pass@4),
inference latency, inference memory, and training cost, all reported with boot-
strap 95% confidence intervals. Pareto analysis shows that LoRA adapters act-
ing only on the q and v attention projections consistently dominate the front,
while the 3B/8B choice mainly defines operating regime. A param-matched con-
trol comparison further indicates that the q/v advantage is structural rather than
purely parametric. The benchmark, selected adapters, and code are available at
https://github.com/EugPal/rag-lora-tradeoffs.
1 INTRODUCTION
Retrieval-augmented generation (RAG) has become a standard way to ground large language models
(LLMs) in external corpora when their parametric memory is insufficient or stale (Lewis et al., 2020;
Gao et al., 2023; Wu et al., 2024). Fordocumentation-orientedassistants – where users expect
answers backed by specific configuration flags, API versions, or feature gates – the bar is even higher:
a deployable system must combine factual accuracy and visible support from the retrieved context
with acceptable latency, inference memory, and training cost. In practice these objectives are in
direct tension. Larger generators improve answer quality but increase inference memory and latency;
richer retrieval and reranking add overhead; more expressive LoRA (Hu et al., 2022) adapters change
the cost of training. Picking a working point therefore requires reasoning about amulti-objective
trade-off rather than a single quality metric.
Existing work largely studies these dimensions in isolation. RAG surveys focus on re-
trieval/fusion/reranking design choices (Gao et al., 2023; Yu et al., 2024); parameter-efficient fine-
tuning surveys focus on the LoRA family in general purpose tasks (Han et al., 2024; Wang et al.,
2025); RAG-evaluation work focuses on quality and groundedness on standard QA sets (Es et al.,
2024; Yu et al., 2024). A few recent papers combine RAG with LoRA, but they typically treat
LoRA as a single technique – a single rank, a single target-module set – and compare it against
alternatives like DoRA or full fine-tuning (Tahir et al., 2024; Zhao et al., 2024a; Baqar & Khanda,
2025). This leaves an open question:how does the choice of LoRA configuration – rank, target
modules, base-model size – interact with retrieval quality, latency, and training cost when the rest of
the RAG pipeline is held fixed?
We address this question by constructing a documentation-grounded RAG benchmark and analysing
a structured space of LoRA configurations on it. Our benchmark consists of 5,144 manually veri-
fied question–answer pairs over the official Kubernetes documentation, paired with a fixed hybrid
retrieval pipeline (BGE-M3 (Chen et al., 2024) dense + BGE-M3 native sparse + Reciprocal Rank
1arXiv:2605.28222v1  [cs.CL]  27 May 2026

Preprint
Fusion (Cormack et al., 2009) + a cross-encoder reranker). Over this pipeline we ablate 20 LoRA
configurations on Llama-3.2-3B-Instruct andLlama-3.1-8B-Instruct (5 ranks ×2
target-module sets per base model) against the corresponding non-adapted baselines. Each configura-
tion is evaluated on (i) token-level F1, (ii) LLM-as-a-judge groundedness and correctness (pass@4),
(iii) inference latency, (iv) inference VRAM, and (v) training time and VRAM, with bootstrap 95%
confidence intervals on all point estimates and paired bootstrap on key ∆F1comparisons. To probe
robustness, we re-run the full grid under 10 retrieval/prompting ablation regimes.
Three findings emerge. First, on every Pareto front we examine – F1-vs-latency, F1-vs-inference-
memory, and F1-vs-training-cost – non-dominated points are dominated by LoRA adapters that act
only on the qandvattention projections. full_attention adapters ( q, k, v, o ) never appear on
the training Pareto front and win at most 2/10 ablation regimes on groundedness. Second, the 3B/8B
choice mainly determines the operating regime rather than the achievable quality: the strongest
LoRA-adapted 3B configuration is statistically indistinguishable from the unadapted 8B baseline
onF1(∆F1∈[−0.021,+0.026] ), while costing roughly 9GB less inference VRAM. Third, a
param-matchedcontrol comparison – in which the total number of trainable LoRA parameters is
held constant across theq/vand full-attention schemes – shows that theq/vadvantage is structural
and not a consequence of having more parameters.
Our contributions are: (i) a manually verified, license-clean RAG benchmark over Kubernetes
documentation, designed for multi-criteria evaluation; (ii) a systematic LoRA-configuration study
within a fixed hybrid-retrieval pipeline, with statistical inference on all reported trade-offs; (iii) a
param-matched control comparison that isolates the structural effect of the target-module choice
from the effect of parameter count; (iv) the released benchmark and selected adapters, intended as a
reproducible starting point for follow-up work on retrieval-augmented fine-tuning of documentation
assistants.
2 RELATEDWORK
RAG and retrieval components.Retrieval-augmented generation (Lewis et al., 2020) has been
thoroughly surveyed in recent years (Gao et al., 2023; Wu et al., 2024): modern RAG is not a single
architecture but a family of designs that differ in how retrieval, fusion, reranking, and context use are
organised. On the retrieval side, dense passage retrieval was canonicalised by Karpukhin et al. (2020),
classical sparse retrieval by BM25 (Robertson & Walker, 1994; Robertson et al., 1995; Robertson &
Zaragoza, 2009), and training-aware extensions such as RocketQA (Qu et al., 2021) and ColBERTv2
(Santhanam et al., 2022) show that retrieval quality depends as much on the training scheme as on
the index type. Recent multi-functional embedders – in particular BGE-M3 (Chen et al., 2024) –
are widely adopted in hybrid pipelines combined with cross-encoder reranking (Nogueira & Cho,
2019; Yoon et al., 2024) and rank-level fusion (Cormack et al., 2009). For documentation-grounded
assistants the overall quality is therefore a property of the whole pipeline, not of the generator alone.
Parameter-efficient fine-tuning.Full fine-tuning of instruction-tuned LLMs is expensive in both
compute and memory; PEFT surveys (Han et al., 2024; Wang et al., 2025) document a large literature
of methods that change model behaviour while keeping the backbone frozen. Among these, LoRA
(Hu et al., 2022) introduces low-rank adapters on linear projections and has spawned a family
of follow-ups, including QLoRA (Dettmers et al., 2023) and LoRA+ (Hayou et al., 2024), plus
throughput-focused training infrastructure such as ASPEN (Ye et al., 2023). Collectively, this line
shows that LoRA can deliver domain adaptation without full fine-tuning. It does not, however, answer
the question of which LoRAconfiguration– rank, target modules, base-model size – to pick when
the adapter is plugged into a retrieval-augmented system.
RAG evaluation, groundedness, and quality–cost trade-offs.The evaluation literature (Yu et al.,
2024; Es et al., 2024) stresses that RAG quality must be measured not only on the final answer but on
retrieval quality, groundedness of the answer in retrieved evidence, and generation properties. Closer
to the trade-off framing of this paper, Baek et al. (2026) treat LoRA-configuration selection itself
as a non-trivial optimisation problem inside the LoRA hyperparameter space, and Baqar & Khanda
(2025) compare RAG, LoRA, and DoRA jointly from an accuracy-and-faithfulness viewpoint.
2

Preprint
Combining RAG and LoRA.A small but growing body of work studies RAG and LoRA together.
JORA (Tahir et al., 2024) contributes infrastructure for retrieval-augmented LoRA fine-tuning;
RAMoLE (Zhao et al., 2024a) uses retrieval to select between multiple LoRA experts at inference
time. Among empirical studies, Baqar & Khanda (2025) is closest in spirit to this work – it contrasts
RAG, LoRA, and DoRA on question answering – but, like other work in this group, treats LoRA as a
single technique to be plugged in rather than as a configuration space to be searched over.
Research gap.Across the four sub-areas above, work on RAG focuses on retrieval and pipeline
design; work on LoRA/PEFT focuses on efficient adaptation in the abstract; work on RAG evaluation
establishes that groundedness and multi-criteria analysis are necessary. What remains underexplored
is thesystematic empirical analysis of how different LoRA configurations inside a fixed RAG archi-
tecture for technical documentation interact with answer quality, latency, inference memory, and
training cost. This is the gap addressed by the present paper.
3 TASK ANDBENCHMARK
3.1 TASK FORMULATION
We consider documentation-grounded question answering over a fixed corpus D={d 1, . . . , d n}
of text chunks. A fixed retrieval-and-reranking module Rmaps every question qto a context
c=R(q, D) . The research variable is the generator configuration x∈ X , characterised by the
base model, the LoRA rank, and the set of target modules. For each configuration xthe system
produces an answer ax(q, c) . We associate with xa quality score Q(x) =F 1(x)and a cost vector
C(x) = 
Linf(x), M inf(x), T train(x), M train(x)
, where Linfis mean inference latency, Minfis
peak inference VRAM, Ttrain is total training time, and Mtrain is peak training VRAM. The goal is
not a singlex∗= arg maxQ(x)but the Pareto-optimal set (Lewis et al., 2020; Yu et al., 2024).
3.2 CORPUS ANDQACONSTRUCTION
The corpus is the official Kubernetes documentation, cleaned and segmented into semantically
coherent chunks suitable for retrieval and answer attribution. The QA set was built in two stages. We
hand-wrote 500 QA pairs directly from the documentation, then drafted further candidates one-by-one
with an LLM agent based on GPT-5.4 – not via bulk synthetic augmentation. All candidates went
through manual verification by the author: pairs were discarded on ambiguous questions, missing
supporting evidence, duplicates, or factual errors in the reference answer. Of 5,467 candidates, 323
(≈5.9% ) were rejected and ≈20% of the remaining pairs were edited; the rest were accepted
as-is. The final benchmark contains 5,144 verified pairs ( 500hand-written + 4,644 LLM-drafted and
manually verified), partitioned into train/eval/test as in Table 1.
Split Rows Exact Normal
Train 3,614 1,449 2,165
Eval 745 361 384
Test 785 475 310
Table 1: Split sizes and distribution of answer types (EXACTvs.NORMAL).
Same-family risk.The QA-generation model (GPT-5.4) and the LLM judge used in §6 ( gpt-
5.4-mini ) belong to the same model family, which raises a same-family-bias concern. There is no
direct train/test leak: the judge is not trained on the gold answers, and all gold answers were manually
verified. We discuss this residual risk explicitly as a limitation (§9).
3.3 ANSWER TYPES
We annotate each QA pair as eitherEXACTorNORMAL.EXACTpairs require near-literal answers
– flags, paths, field names, API versions, feature gates – whileNORMALpairs are factual answers
that tolerate paraphrasing as long as the meaning is preserved. This typology was designed for this
3

Preprint
benchmark; it is used as a supervision signal at training time and as a stratification axis at evaluation
time, but it isnotpassed to the generator at inference, which receives only the question and the
retrieved context. This separation of training- and test-time conditioning follows the evaluation
protocols in prior RAG work (Yu et al., 2024; Es et al., 2024).
4 RAG PIPELINE
The pipeline follows the standard three-step structure – retrieval, reranking, generation – consistent
with modern RAG surveys (Lewis et al., 2020; Gao et al., 2023; Wu et al., 2024).
Retrieval.The main configuration uses a hybrid retrieval contour. Dense retrieval runs over a
FAISS (Douze et al., 2024; Johnson et al., 2019) index of BGE-M3 (Chen et al., 2024) embeddings;
sparse retrieval uses the native sparse channel of BGE-M3. Dense and sparse candidate lists are
merged via Reciprocal Rank Fusion (Cormack et al., 2009). Classical BM25 (Robertson & Zaragoza,
2009) is used only as the sparse component in a separate ablation regime (§7.4), not in the main
pipeline.
Reranking.The fused candidate list passes through a pretrained cross-encoder reranker,
BAAI/bge-reranker-v2-m3 , building on the standard cross-encoder and listwise reranking
literature (Nogueira & Cho, 2019; Yoon et al., 2024) and on retrieval extensions of large language
models (Li et al., 2023).
Generator and prompting.The generator is one of two instruction-tuned models from the Llama 3
family (Dubey et al., 2024): Llama-3.2-3B-Instruct andLlama-3.1-8B-Instruct .
Both are decoder-only transformers with RoPE positional encoding, grouped-query attention, and
SwiGLU MLP blocks. At inference we use aneutralprompting mode, with no explicit instruction
to “answer only from the context” – the role of strict grounding is left to LoRA fine-tuning and to
retrieval quality rather than to prompt engineering. This separates the contribution of adaptation from
that of prompt-side anchoring and keeps the comparison between adapters honest. At training time
the model retains the option to condition on the answer-type label (EXACT/NORMAL, §3.3), but the
inference prompt does not carry that label. This split between training and test-time conditioning
follows established practice in RAG evaluation (Yu et al., 2024; Es et al., 2024).
Inference context.The main inference regime uses eval_top_k= 2 retrieved chunks, matching
embed_top_k= 2 at training time. For sensitivity analysis we additionally run the full grid with
eval_top_k∈ {1,4}(§7.5).
5 LORA CONFIGURATIONS ANDEXPERIMENTALDESIGN
5.1 CONFIGURATION SPACE
Full fine-tuning of the chosen Llama-3 models is too expensive for the configuration search we want
to run. We therefore use Low-Rank Adaptation (Hu et al., 2022), which is well-suited for ablations
because it changes the generator’s behaviour while keeping the backbone frozen (Dettmers et al.,
2023; Han et al., 2024).
We train 20 LoRA adapters on top of two base models. For each base model we vary two factors:
the LoRA rank r∈ {4,8,16,32,64} and the set of adapted modules. The latter takes two val-
ues:qv_only adapts the qandvattention projections only; full_attention adapts all four
major attention projections ( q,k,v,o). The two unadapted base models, 3B baseline and8B
baseline, serve as controls evaluated under the same inference protocol.
5.2 FIXED TRAINING HYPERPARAMETERS
All training runs use a common optimisation setup: num_train_epochs= 8 ,learning_-
rate= 2×10−5, AdamW (Kingma & Ba, 2015; Loshchilov & Hutter, 2019) with cosine schedule
(Loshchilov & Hutter, 2017) and a linear warm-up over ≈3% of the total steps, in bf16 mixed
4

Preprint
precision. We use embed_top_k= 2 retrieved chunks per training example. LoRA-side hyper-
parameters are also fixed across runs: bias=none ,task_type=CAUSAL_LM , and lora_-
dropout= 0.05 . We tie lora_alpha to rank via the rule α= 2r . With these constraints, the
comparison isolates three sources of variation: the base-model size, the rank, and the target-module
choice.
The training prompt uses a mixed-context strategy. When supporting chunks are pre-annotated for an
example they are used directly; otherwise the context is materialised from the retrieval contour (§4).
This is methodologically close to retrieval-augmented fine-tuning setups such as JORA (Tahir et al.,
2024), and narrows the gap between idealised training and the actual inference regime.
5.3 HARDWARE AND INFERENCE STACK
All runtime cost metrics (latency,inference VRAM,training time,training VRAM) are measured on a
singlehardware configuration, to remove device-related variance from the comparison. The compute
node runs in Yandex DataSphere: one NVIDIA A100 (40 GB), 28 vCPU, and 114 GB of host RAM.
The same node is used both for LoRA training and for inference, so training and inference metrics
are directly comparable across configurations.
The inference stack is Hugging Face Transformers with PEFT: the base model is loaded in its na-
tive precision without quantisation ( -no-quant-generator and-no-quant-judge are set),
the LoRA adapter is attached separately via PeftModel.from_pretrained , and the mixed-
precision setting matches training ( bf16 ). The retrieval contour uses BAAI/bge-m3 for dense
embeddings and BAAI/bge-reranker-v2-m3 for reranking ( reranker_batch_size= 16 ,
retrieve_top_n= 20 ). Generation is greedy (no sampling) with a fixed maximum answer length
shared across configurations.
Latency is measured per-sample (effective generator batch size of 1) as the mean wall-clock time per
test example over the full test split ( n= 785 ), counting all pipeline stages: query embedding, dense
and sparse retrieval, reranking, prompt construction, and generation. Inference VRAM is logged as
peak device memory viatorch.cuda.max_memory_allocated()at the end of each run.
5.4 HYPOTHESES
The experimental design tests four working hypotheses: (i) scaling the base model improves quality
but raises both latency and inference memory; (ii) increasing rank expands adaptation capacity, but
with diminishing returns at high r; (iii) broader target-module coverage ( full_attention ) is
not necessarily preferable once training and inference costs are factored in; and (iv) among the
resulting configurations, the practically useful working points are not the per-metric maxima but the
Pareto-optimal set.
6 EVALUATIONMETHODOLOGY
6.1 QUALITY METRICS
The primary quality metric istoken-level F1between the generated and the gold answer (Yu et al.,
2024). All headline results and per-regime tables and plots in §7 and Appendix I are computed on
the held-out test split ( n= 785 ); the eval split ( n= 745 ) is used for configuration selection during
the experimental loop. Along with the point estimate of F1we report a non-parametric bootstrap
95% confidence interval ( 1,000 resamples on the test split). For paired comparisons of ∆F1between
configurations we use the paired bootstrap on the same test split, which is appropriate for ranking
close configurations on a shared sample.
We do not report embedding-based semantic metrics such as BERTScore (Zhang et al., 2020). For
documentation-grounded question answering with a high share ofEXACTquestions, token F1is the
most directly interpretable similarity metric; the semantic side of quality is instead captured by the
judge-based scores described next.
5

Preprint
6.2 JUDGE-BASED GROUNDEDNESS AND CORRECTNESS
We add two judge-based quality axes computed by an external LLM judge, gpt-5.4-mini . The
judge receives only the triple(question, context, answer)and is blind to the generator identity – no
model_id or configuration name is provided – so that scores are not contaminated by adapter or
model branding. It assigns two independent ratings:correctness, the content-level accuracy of the
answer with respect to the provided context, andgroundedness, the degree to which the answer is
supported by the context with no unsupported additions (Yu et al., 2024; Es et al., 2024; Baqar &
Khanda, 2025).
We aggregate these into two pass-at- kscores,correctness_pass@4andgroundedness_pass@4:
the fraction of answers that the judge rates ≥4 on the corresponding scale. These judge-based
scores are used as asecondaxis of quality alongside F1, not as a replacement. As §7.3 shows, the
configuration that maximises F1is often not the configuration that maximises grounding, which is
itself an informative finding.
6.3 COST METRICS
We measure four cost quantities. The two inference-side metrics are mean latency Linfand peak
inference VRAM Minf(§5.3). The two training-side metrics are total training time Ttrain and peak
training VRAM Mtrain. The four together form the cost vector used in the Pareto analysis (§3.1).
This setup follows the quality–cost framing used in RAG and LoRA evaluation literature (Es et al.,
2024; Baek et al., 2026; Baqar & Khanda, 2025).
6.4 PARETO ANALYSIS
Comparison across the resulting configuration set is done with a multi-objective Pareto analysis. A
configuration xis Pareto-optimal if there is no other configuration x′that is at least as good on quality
and at least as good on every cost dimension, and strictly better on at least one of them (§3.1). In the
two-dimensional fronts of §7 the quality coordinate is F1and the cost coordinate alternates between
mean inference latency, peak inference VRAM, training time, and training VRAM. This protocol
focuses the discussion on the non-dominated set rather than on a single “best” configuration; the
latter view is incompatible with the actual practical question of choosing a working point under a
cost budget.
7 RESULTS
All results below are computed on the held-out test split ( n= 785 ) with the fixed hybrid-retrieval
pipeline described in §4 and the fixed training and inference setup described in §5. Unless noted oth-
erwise, F1 denotes token-level F1 andgrnd@4(resp.corr@4) denotes the LLM-judge groundedness
(resp. correctness) pass@4 score (§6). 95% bootstrap confidence intervals (CIs) are computed with
1,000 resamples; for paired comparisons of ∆F1between configurations we use the paired bootstrap
on the same test split.
7.1 QUALITY VS.INFERENCE COST
On the F1-vs-latency front the non-dominated configurations are 3B r64 qv_only (F1=
0.597 [0.574, 0.622]) and 8B r64 qv_only (F1= 0.617 [0.593, 0.639]); see Table 2. The
8B point gains ∆F1= +0.020 over the 3B point with paired-bootstrap 95% CI [+0.0005,+0.0410]
– a small but statistically supported margin – at the cost of an additional ∼9 GB of inference memory
and a 0.057 s latency increase. Notably, 3B r64 qv_only is statistically indistinguishable from
the unadapted 8B baseline (0.595 [0.572, 0.620]): ∆F1∈[−0.021,+0.026] . The runtime-front
choice is therefore not between two qualities but between two operating regimes.
6

Preprint
Figure 1: Token-level F1 vs. end-to-end inference latency (per query) in the base regime. Error
bars are 95% bootstrap CIs on F1. The Pareto front (red dashed) connects the two non-dominated
configurations, 3B r64 qv_only and8B r64 qv_only ; allfull_attention adapters lie
strictly inside the front.
Config.F 1[95% CI] Lat. (s) VRAM (GB)
3B r64 qv_only0.597 [0.574, 0.622] 0.598 12.76
8B r64 qv_only0.617 [0.593, 0.639] 0.655 21.93
Table 2: Non-dominated points on the F1-vs-latency front in the base regime.
The F1-vs-inference-VRAM view cleanly separates the 3B and 8B families (VRAM difference
∼9 GB), whereas within each family the VRAM spread is only 0.3–0.4GB – well below the 95% CI
width of F1. Inference memory therefore acts as a discrete family selector rather than a continuous
trade-off knob.
7.2 TRAINING COST
The cheapest training point is 3B r4 qv_only (52.95 min, 19.07 GB). Stronger training-side
trade-offs cluster around qv_only configurations at r=32 andr=64 for the 3B family and r=16
andr=64 for the 8B family.No full_attention configuration appears on the training Pareto
front (Table 3). Within the 8B family the upper qv_only points ( r=16 andr=64 ) have largely
overlapping F1CIs, so the choice between them is driven by secondary criteria, not by a robust
quality gap.
7

Preprint
Figure 2: F1vs. training wall-clock time (minutes) over all 20 LoRA configurations. The Pareto front
(red dashed) is dominated by qv_only adapters; no full_attention configuration appears on
it. Error bars are 95% bootstrap CIs onF 1.
Figure 3: F1vs. peak training GPU memory, split by base-model family (3B / 8B). Within each
family, the training Pareto front is again populated entirely by qv_only adapters. Error bars are
95% bootstrap CIs onF 1.
8

Preprint
Config.F 1[95% CI] Train (min) VRAM (GB)
3B r4 qv_only0.572 [0.550, 0.592] 52.95 19.07
3B r8 qv_only0.573 [0.552, 0.594] 54.08 19.09
3B r16 qv_only0.583 [0.562, 0.605] 55.90 19.13
3B r32 qv_only0.592 [0.569, 0.614] 52.41 19.20
3B r64 qv_only0.597 [0.574, 0.622] 53.32 19.39
8B r4 qv_only0.610 [0.588, 0.633] 68.57 31.31
8B r16 qv_only0.615 [0.593, 0.634] 68.38 31.39
8B r64 qv_only0.617 [0.595, 0.640] 68.93 31.74
Table 3: Non-dominated points on the training Pareto front (time and VRAM). All training-front
points use theqv_onlyadaptation scheme.
7.3 GROUNDEDNESS AS A SECOND QUALITY AXIS
Figure 4: F1vs. LLM-judge groundedness pass@4 in the base regime. Error bars are 95% bootstrap
CIs on both axes. Adapted configurations cluster tightly in the upper right, while the unadapted
baselines are clearly separated ongrnd@4.
The configuration with the highest point F1, 8B r64 qv_only (F1= 0.617 ,grnd@4 = 0.818 ),
doesnotmaximise groundedness. The grnd@4 maximum ( 0.825 ) is reached by 8B r16 qv_-
only (F1= 0.615 ); the two are statistically indistinguishable on F1(∆F1∈[−0.011,+0.016] ).
LoRA-adapted configurations also shift grnd@4 upward by 0.05–0.08 over the unadapted baselines
at comparable latency – a margin that exceeds the typical CI width ( ≈0.03 ). Within already-adapted
configurations, grnd@4 spread mostly falls inside the CIs and separation is driven by the 3B/8B
choice. Task quality and grounding are therefore not interchangeable: for documentation QA,
the choice of optimal configuration depends on whether the deployment prioritises exact-match
agreement, supportedness, or a compromise between them.
7.4 ROBUSTNESS ACROSS RETRIEVAL AND PROMPTING
To test whether the LoRA conclusions are an artefact of one pipeline choice, we re-run the full grid un-
der 10 retrieval/prompting regimes: five retrieval variants ( base ,reranker_off ,dense_only ,
9

Preprint
Figure 5: Frequency with which each adaptation scheme yields the best configuration across the 10
retrieval/prompting ablation regimes, separately forF 1(blue) andgrnd@4(orange).
sparse_only ,hybrid_bm25 ) crossed with two prompting modes ( neutral ,explicit_-
grounded ). Two main observations hold across all 10 modes (Table 4, full per-regime tables
in Appendix I). First, the best-F1 configuration is 8B r64 qv_only inall10 modes, and the
best-grnd@4 configuration concentrates on qv_only adapters in 8 of 10 modes ( full_atten-
tion appears only in two sparse -based modes). Second, the F1-optimal and grnd@4-optimal
points never coincide (same_point=noin 10/10 modes). The choice of working configuration is
therefore a function of the retrieval–prompting setupandthe quality criterion, not of LoRA alone.
Criterionqv_onlywinsfull_attentionwins
BestF 1 10/10 0/10
Bestgrnd@48/10 2/10
Table 4: Frequency with which each adaptation scheme yields the best configuration across the 10
retrieval/prompting ablation regimes.
7.5 STATISTICAL ROBUSTNESS OF∆F 1
Table 5 reports paired-bootstrap CIs for six key comparisons. The reliably supported effects are (i) 3B
r64 qv_only over3B baseline , (ii)3B r64 qv_only over the param-matched 3B r64
full_attention adapter, and (iii) the family-level ∆F1ofqv_only overfull_attention
averaged over the eight param-matched pairs analysed in §8. In contrast, the gaps between 3B
r64 qv_only and the 8B baseline , and between the top two 8B qv_only points, are not
distinguishable from sampling noise at n= 785 . The headline of this work is therefore not any single
local maximum onF 1, but the structural advantage of theqv_onlyadaptation scheme.
10

Preprint
Comparison∆F 1 95% CI
3B r64 qv_only-3B baseline+0.056 [+0.033,+0.078]
3B r64 qv_only-8B baseline+0.003 [−0.021,+0.026]
8B r64 qv_only-3B r64 qv_only+0.020 [+0.0005,+0.0410]
8B r64 qv_only-8B r16 qv_only+0.003 [−0.011,+0.016]
3B r64 qv_only-3B r64 full_attention+0.020 [+0.004,+0.036]
qv_only-full_attention∗+0.007 [+0.001,+0.012]
*mean∆F 1over the 8 param-matched pairs (Table 6).
Table 5: Paired-bootstrap 95% CIs on∆F 1(n= 785,1,000resamples).
8 PARAM-MATCHEDCONTROLCOMPARISON
Up to this point, the F1advantage of qv_only adapters (§7.1–§7.4) could have two natural expla-
nations. The first isstructural: adapting exactly the two projections that select and re-weight the
retrieved context is sufficient inside this RAG architecture, while extending adaptation to kando
projections grows the adapter without buying proportional quality. The second isparametric: with
the same rank r,full_attention adapts twice as many projections as qv_only and thus splits
a fixed-shape low-rank budget more thinly across attention sub-spaces, so the per-projection capacity
may simply be too small.
To distinguish the two, we run a param-matched control. Because full_attention adapts four
projections instead of two, equal total LoRA parameter counts are achieved by halving the rank for
full_attention . We therefore pair each qv_only configuration at rank rwith the full_-
attention configuration at rank r/2, computing ∆F1=F 1(QV_ONLY)−F 1(FULL_ATTENTION)
on the test split with paired bootstrap 95% confidence intervals. Results are reported in Table 6.
Family Param budgetqv_only full_attention∆F 1 95% CI
3B 256d r=64r=32 +0.021 [+0.005,+0.037]
3B 128d r=32r=16 +0.002 [−0.013,+0.017]
3B 64d r=16r=8 +0.005 [−0.008,+0.020]
3B 32d r=8r=4−0.002 [−0.014,+0.011]
8B 256d r=64r=32 +0.015 [−0.001,+0.030]
8B 128d r=32r=16 +0.008 [−0.006,+0.023]
8B 64d r=16r=8 +0.016 [+0.003,+0.027]
8B 32d r=8r=4 +0.003 [−0.009,+0.017]
Table 6: Param-matched comparison between qv_only andfull_attention at equal total
LoRA parameter count. dis the per-projection rank dimensionality budget. Paired-bootstrap 95%
CIs on∆F 1.
Two observations follow. First, at equal parameter budget, qv_only is significantly better in 2 of 8
pairs (one in each family), statistically indistinguishable from full_attention in the remaining
6, and significantly worse in none. The advantage therefore survives the parameter-count control: it is
not the case that full_attention would dominate once you give it the same budget as qv_only .
Second, the only family-budget combination at which full_attention comes close is 3B at the
smallest budget ( 32d,r=8 vs.r=4), where the CI of ∆F1straddles zero on the negative side. This is
consistent with the interpretation that at very low rank qv_only is itself capacity-constrained, and
full_attention can match it by spreading the same budget over more projections; once rank
grows beyondr=8for either scheme, the structural advantage ofqv_onlyre-emerges.
9 DISCUSSION
9.1 WHYQ V_O N L YWINS ON THEPARETO FRONT
Across every Pareto front in §7 the non-dominated points concentrate around qv_only adapters,
and the paired-bootstrap test against the param-matched full_attention alternative gives a
11

Preprint
mean ∆F1= +0.007 [+0.001,+0.012] (§7.5, §8). The most plausible explanation is that, inside
a fixed retrieval contour, the adapter’s job is not to globally restructure attention but to refine how
the generatorselects fromandre-weightsthe already-supplied context. The qandvprojections
are precisely the two attention projections that control these two operations; adapting also kando
increases the adapter size and the training cost faster than it buys robust gain on the front.
9.2 BACKBONE SIZE SETS THE REGIME,NOT THE VERDICT
The 3B/8B split looks structural but should not be read as the headline. A larger 8B backbone does set
the upper F1ceiling, but it almost automatically commits the system to a more expensive operating
regime – roughly 9GB of additional inference VRAM and ∼15 extra training minutes per adapter.
Crucially, the strongest adapted 3B configuration is statistically indistinguishable from the unadapted
8B baseline ( ∆F1∈[−0.021,+0.026] ). So scaling the backbone is not the only way to recover that
level of quality: targeted LoRA adaptation closes the gap at a fraction of the cost. The backbone
choice is therefore best understood as selecting an operating regime (small vs. large inference budget),
with the within-regime quality controlled mainly by the adapter scheme.
9.3 RANK EFFECT IS SUB-LINEAR BUT PRACTICALLY USEFUL
Increasing rhelps F1more than it hurts inference cost in this pipeline. For 3B qv_only , going
from r=4 tor=64 moves F1from 0.572 to0.597 , while latency stays in 0.60–0.63 s and inference
VRAM moves only from 12.67 to12.76 GB – well inside the noise of measurement. The dominant
inference cost in this pipeline is the base model and the retrieval contour, not the LoRA adapter itself.
The same monotone-but-saturating pattern holds for the 8B family, but the upper rungs ( r=16 vs.
r=64 ) are within paired-bootstrap noise, so within the top 8B configurations the rank should be
picked by secondary criteria (e.g., groundedness, or training cost), not by point F1.
9.4 RETRIEVAL AND PROMPTING SHIFT THE OPTIMUM BUT NOT THE FAMILY
The 10-mode ablation shows that switching off the reranker, going dense- or sparse-only, or replacing
the native sparse channel with BM25, together with the choice of neutral vs. explicit-grounded
prompting,doesmove the per-mode optimum point inside the same configuration grid. What it does
notdo is overturn the qualitative pattern: 8B r64 qv_only is the best- F1configuration in 10/10
modes, and qv_only adapters win ongrnd@4in 8 of 10. Retrieval and prompting therefore decide
how favourable the operating environment is, not which adaptation scheme should be used in it.
9.5 SENSITIVITY TO CONTEXT BUDGET(topk)
A separate generalisation run varies the number of retrieved chunks at inference time, holding all
adapters fixed. Increasing eval_top_k from 1to4raises the best F1from 0.600 to0.632 but
adds∼0.12 s of latency (Figure 6, Table 7). The best- F1configuration is 8B r64 qv_only for all
three values of top_k ; the runtime Pareto front collapses to a single point at top_k= 1 and gains
3B r64 qv_only as a second non-dominated point at top_k∈ {2,4} . The conclusion is that
broadening retrieval is not a free quality lever, but the structural ranking of LoRA configurations
survives the change.
top_kBest-F 1configF 1[95% CI] Lat. (s) Runtime front
18B r64 qv_only0.600 [0.577, 0.623] 0.604 {8B r64 qv_only}
28B r64 qv_only0.617 [0.593, 0.639] 0.655 {3B r64 qv_only,8B r64 qv_only}
48B r64 qv_only0.632 [0.612, 0.654] 0.719 {3B r64 qv_only,8B r64 qv_only}
Table 7: Sensitivity to retrieval context budget at inference time, in the base regime. All adapters held
fixed.
12

Preprint
Figure 6: Effect of the retrieval cutoff kon the best F1(left axis, blue; with 95% bootstrap CI) and
on mean end-to-end inference latency (right axis, red). The chosen working point k=2 corresponds
to the “knee” of the curve, beyond which marginal F1gains are dominated by marginal latency cost.
9.6 ERROR ANALYSIS
To make the discussion of residual errors quantitative, we sampled 100 mis-predictions on the
test split ( 50each from 3B r64 qv_only and8B r64 qv_only ,F1<1,seed= 42 ) and
labelled them into four operational classes:retrieval miss,overclaiming,incomplete answer, and
exact/precision failure(Table 8).Exact/precisionerrors dominate ( 53% overall, 66% for the 8B
adapter): a substantial share of the remaining gap is not about missing knowledge but about reproduc-
ing a literal command, flag, version, or field name verbatim. Incomplete answers ( 24%) and retrieval
misses ( 19%) follow; pure overclaiming is rare in this sample ( 4%). This taxonomy suggests two
natural follow-up directions: post-hoc constrained decoding for the exact-class sub-distribution, and
groundedness-aware re-training for the incomplete-answer class.
Error type3B r64 qv_only 8B r64 qv_onlyTotal
retrieval miss 11 (22%) 8 (16%) 19 (19%)
overclaiming 2 (4%) 2 (4%) 4 (4%)
incomplete answer 17 (34%) 7 (14%) 24 (24%)
exact/precision failure 20 (40%) 33 (66%) 53 (53%)
Table 8: Manual error-type taxonomy on a balanced 100-sample subset of mis-predictions ( F1<1,
fixed random seed).
10 CONCLUSION
We presented a systematic study of LoRA configurations inside a fixed documentation-grounded
RAG pipeline. Three findings stand out. First, on every Pareto front we examined – F1against
latency, inference VRAM, training time, and training VRAM – the non-dominated points concentrate
around adapters that act on the qandvattention projections only; full-attention adapters do not
appear on the training Pareto front in this study, and the param-matched control comparison confirms
that the advantage of qv_only is structural rather than just a parameter-count artefact. Second,
the 3B/8B choice determines the operating regime (small vs. large inference budget), but a strong
LoRA-adapted 3B configuration reaches statistically comparable F1to the unadapted 8B baseline
at roughly 9GB less inference VRAM, so backbone size and adaptation interact ascomplements,
not substitutes. Third, judge-based groundedness is not interchangeable with F1: the configuration
13

Preprint
that maximisesF 1is not the configuration that maximises grounding, and at fixedF 1the difference
between them is robust across the 10 retrieval/prompting ablation regimes.
Practically, the paper produces three recommended working points: a lightweight one ( 3B r64
qv_only ) when the deployment is inference-cost-constrained but expects high quality; the F1-
maximising one ( 8B r64 qv_only ) when an 8B budget is available; and a grounding-maximising
8B alternative (8B r16 qv_only) when supportedness is the priority.
To support reproducible follow-up work, we will release the manually verified Kubernetes QA
benchmark and the selected LoRA adapters together with this paper. The most natural extensions of
this study are multi-seed validation for the close-CI comparisons, a human-agreement study for the
LLM judge, a port of the ablation protocol to a second documentation domain, and a re-measurement
of the runtime axis on alternative inference stacks.
REFERENCES
Sang-Eon Baek, Joon-Min Lee, Sung-Bae Kim, and Tae-Hyun Oh. Efficient hyper-parameter search
for LoRA via language-aided Bayesian optimization, 2026.
Mohammad Baqar and Rajat Khanda. Hallucinations and truth: A comprehensive accuracy evaluation
of RAG, LoRA and DoRA, 2025.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-Embedding:
Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge
distillation. InFindings of the Association for Computational Linguistics: ACL 2024, 2024.
Gordon V . Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal rank fusion outperforms
Condorcet and individual rank learning methods. InProceedings of the 32nd International ACM
SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759, 2009.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning
of quantized LLMs. InAdvances in Neural Information Processing Systems, 2023.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The Faiss library, 2024.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The Llama 3 herd of models,
2024.
Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. RAGAs: Automated evaluation
of retrieval augmented generation. InProceedings of the 18th Conference of the European Chapter
of the Association for Computational Linguistics: System Demonstrations, 2024.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey,
2023.
Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. Parameter-efficient fine-tuning
for large models: A comprehensive survey, 2024.
Soufiane Hayou, Nikhil Ghosh, and Bin Yu. LoRA+: Efficient low rank adaptation of large models.
InProceedings of the 41st International Conference on Machine Learning, 2024.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large language models. InInternational
Conference on Learning Representations (ICLR), 2022.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs.IEEE
Transactions on Big Data, 7(3):535–547, 2019.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), 2020.
14

Preprint
Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. InInternational
Conference on Learning Representations, 2015.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems, volume 33, pp. 9459–9474, 2020.
Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao. Making large language models a better
foundation for dense retrieval, 2023.
Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In
International Conference on Learning Representations, 2017.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. InInternational Confer-
ence on Learning Representations, 2019.
Rodrigo Nogueira and Kyunghyun Cho. Passage re-ranking with BERT, 2019.
Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua
Wu, and Haifeng Wang. RocketQA: An optimized training approach to dense passage retrieval for
open-domain question answering. InProceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021.
Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond.
Foundations and Trends in Information Retrieval, 3(4):333–389, 2009.
Stephen E. Robertson and Steve Walker. Some simple effective approximations to the 2-Poisson
model for probabilistic weighted retrieval. InProceedings of the 17th Annual International ACM
SIGIR Conference on Research and Development in Information Retrieval, 1994.
Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford.
Okapi at TREC-3. InProceedings of the Third Text REtrieval Conference (TREC-3), 1995.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Col-
BERTv2: Effective and efficient retrieval via lightweight late interaction. InProceedings of the
2022 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies, 2022.
Anique Tahir, Bryan Cheng, Ian Price, et al. JORA: JAX tensor-parallel LoRA library for re-
trieval augmented fine-tuning. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics: System Demonstrations, 2024.
Luping Wang, Sheng Chen, Linnan Jiang, Shu Pan, Runze Cai, Sen Yang, and Fei Yang. Parameter-
efficient fine-tuning in large models: A survey of methodologies.Artificial Intelligence Review,
2025.
Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang, Xue
Liu, Tei-Wei Kuo, Nan Guan, and Chun Jason Xue. Retrieval-augmented generation for natural
language processing: A survey, 2024.
Zhengmao Ye, Dengchun Li, Jingqi Tian, Tingfeng Lan, Yanbo Liang, Yexi Jiang, Jianjun Wang,
Jiamu Xie, Kaichen Lu, and Yujie Bian. ASPEN: High-throughput LoRA fine-tuning of large
language models with a single GPU, 2023.
Soyoung Yoon, Eunbi Choi, Jiyeon Kim, Hyeongu Yun, Yireun Kim, and Seung-won Hwang. ListT5:
Listwise reranking with fusion-in-decoder improves zero-shot retrieval. InProceedings of the 62nd
Annual Meeting of the Association for Computational Linguistics, 2024.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. Evaluation of retrieval-
augmented generation: A survey, 2024.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore:
Evaluating text generation with BERT. InInternational Conference on Learning Representations,
2020.
15

Preprint
Hao Zhao, Wensheng Gan, Yutao Song, et al. Retrieval-augmented mixture of LoRA experts for
uploadable machine learning, 2024a.
Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. Dense text retrieval based on pretrained
language models: A survey.ACM Transactions on Information Systems, 42(4):89:1–89:60, 2024b.
16

Preprint
CONTENTS
1 Introduction 1
2 Related Work 2
3 Task and Benchmark 3
3.1 Task formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
3.2 Corpus and QA construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
3.3 Answer types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
4 RAG Pipeline 4
5 LoRA Configurations and Experimental Design 4
5.1 Configuration space . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
5.2 Fixed training hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
5.3 Hardware and inference stack . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
5.4 Hypotheses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
6 Evaluation Methodology 5
6.1 Quality metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
6.2 Judge-based groundedness and correctness . . . . . . . . . . . . . . . . . . . . . . 6
6.3 Cost metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
6.4 Pareto analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
7 Results 6
7.1 Quality vs. inference cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
7.2 Training cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
7.3 Groundedness as a second quality axis . . . . . . . . . . . . . . . . . . . . . . . . 9
7.4 Robustness across retrieval and prompting . . . . . . . . . . . . . . . . . . . . . . 9
7.5 Statistical robustness of∆F 1. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
8 Param-Matched Control Comparison 11
9 Discussion 11
9.1 Whyqv_onlywins on the Pareto front . . . . . . . . . . . . . . . . . . . . . . . 11
9.2 Backbone size sets the regime, not the verdict . . . . . . . . . . . . . . . . . . . . 12
9.3 Rank effect is sub-linear but practically useful . . . . . . . . . . . . . . . . . . . . 12
9.4 Retrieval and prompting shift the optimum but not the family . . . . . . . . . . . . 12
9.5 Sensitivity to context budget (topk) . . . . . . . . . . . . . . . . . . . . . . . . . 12
9.6 Error analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
17

Preprint
10 Conclusion 13
A Limitations 20
B Domain and Problem Statement 20
B.1 Domain . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B.2 Problem statement . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
B.3 Task specifics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
C Data and Corpus 22
C.1 Data source . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
C.2 Construction of the QA sets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
C.3 Answer types . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
C.4 Dataset characteristics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
D RAG System Architecture 23
D.1 Overall solution scheme . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
D.2 Retrieval component . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
D.3 Ranking and reranking . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
D.4 Generator and prompting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
E LoRA Adaptation and Experimental Design 24
E.1 Motivation for using LoRA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
E.2 Trained configurations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
E.2.1 Baseline evaluation protocol . . . . . . . . . . . . . . . . . . . . . . . . . 24
E.3 Fixed training hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
E.4 LoRA adaptation parameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
E.5 Training-context formation regime . . . . . . . . . . . . . . . . . . . . . . . . . . 25
E.6 Hardware configuration and inference stack . . . . . . . . . . . . . . . . . . . . . 26
E.7 Experimental hypotheses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
F Evaluation Methodology 26
F.1 Quality metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
F.2 Judge-based groundedness and correctness . . . . . . . . . . . . . . . . . . . . . . 27
F.3 Cost metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
F.4 Pareto analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
G Experimental Results 27
G.1 Quality and latency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
G.2 Quality and inference memory . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
G.3 Training cost . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
18

Preprint
G.4 Judge-based groundedness as a second quality axis . . . . . . . . . . . . . . . . . 31
G.5 Ablation experiments on retrieval and prompting . . . . . . . . . . . . . . . . . . 32
G.6 Statistical robustness ofF1differences . . . . . . . . . . . . . . . . . . . . . . . . 34
H Discussion 34
H.1 Whyqv_onlyadapters dominate . . . . . . . . . . . . . . . . . . . . . . . . . . 34
H.1.1 Param-matched comparison ofqv_onlyvsfull_attention. . . . . 35
H.2 Why the backbone size sets the operating regime but does not cancel the LoRA effect 35
H.3 Why the effect of rank growth is non-linear . . . . . . . . . . . . . . . . . . . . . 35
H.4 Why retrieval and prompting shift the optimum point but do not cancel the conclusions
about LoRA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
H.5 Generalization analysis bytop_k. . . . . . . . . . . . . . . . . . . . . . . . . . 36
H.6 Practical recommendations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
H.7 Error analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
I Per-Regime Detailed Tables 39
J Per-Regime Plots 49
19

Preprint
A LIMITATIONS
Single training seed.All LoRA adapters were trained with a single fixed seed ( seed= 42 ), which
controls adapter initialisation, batch order, and dropout masks. The bootstrap CIs of §7 capture
sampling variance over the test set (fixed model, varying test) butnottraining-time variance from re-
running the same configuration with a different seed. In the LLM fine-tuning literature, seed-to-seed
dispersion on task metrics can reach a few F1points (Baek et al., 2026), which is particularly relevant
for the 0.002 –0.005F 1gaps reported here. We therefore phrase claims about close configurations
(e.g.,8B r64 qv_only vs.8B r16 qv_only ) as “statistically comparable” rather than as one
strictly dominating the other.
Single LLM judge, no human-agreement calibration.Thegrnd@4andcorr@4scores are
produced by a single judge ( gpt-5.4-mini ) under a fixed protocol. The judge is blind to the
generator (no model_id ), which removes the most direct form of generator-side bias, but we did not
run a formal human-agreement study (e.g., Cohen’s κon an expert-annotated sample) and we did not
cross-validate the judge against an alternative judge model. The judge-based metrics should therefore
be read as indicative scores from one automatic system rather than as final values of correctness
or groundedness. Importantly, all main conclusions of the paper rely on the judge-free metric F1;
the judge-based scores are used only as a secondary axis to differentiate configurations that are
close on F1. A residual concern is that the judge model ( gpt-5.4-mini ) and the model used in
QA-candidate drafting (GPT-5.4, §3.2) come from the same family, which may introduce same-family
bias against Llama-style answers. By construction this bias is uniform across all compared LoRA
configurations, so the relative ordering on the Pareto front is preserved, but the absolute judge scores
should be interpreted with this caveat.
Single domain.The empirical results, including absolute metric values and the shape of the
Pareto fronts, are tied to the Kubernetes-documentation corpus and the curated QA set built on it.
The domain has characteristic properties – manyEXACTquestions, technical terms, commands,
flags, API versions – that affect both retrieval behaviour and the distribution of error types (§9.6).
Absolute numbers should not be transported to other documentation domains (medical regulations,
legal corpora, internal company docs) without re-running the protocol. Thequalitativefindings
–qv_only advantage, sub-linear rank effect, and the family-as-regime split – are conjecturally
transferable, but confirming this requires an equivalent ablation in a different domain.
Single retriever and reranker.The retrieval stack (BGE-M3 dense + BGE-M3 native sparse,
merged via RRF, followed by the bge-reranker-v2-m3 cross-encoder) is held fixed across all
experiments. Replacing the embedder or reranker, or moving to a late-interaction architecture such
as ColBERTv2 (Santhanam et al., 2022), can shift both absolute metrics and the position of the
per-regime optimum. §7 is therefore best read as a statement about LoRA under this specific retrieval
stack, not as a universal claim about RAG systems.
Single inference hardware and stack.Latency, inference VRAM, training time, and training
VRAM are all measured on a single hardware setup (§5.3). Different batching, alternative inference
engines (vLLM, TensorRT-LLM), different precisions ( fp16 ,int8 ,int4 ), or different parallelism
schemes can shift absolute values and possibly the relative ordering on the runtime axis. The shape
of the quality-versus-cost curves is robust, but absolute numbers should be re-measured before
transferring to a production deployment.
B DOMAIN ANDPROBLEMSTATEMENT
B.1 DOMAIN
The domain of the work is question answering over technical documentation. Unlike open-domain
conversational systems, here the user expects not a generally plausible answer, but accurate and
verifiable information grounded in the documentation. Such a setting fits the logic of Retrieval-
Augmented Generation (RAG), in which the answer must rely on an external corpus rather than only
on the parametric memory of the model Lewis et al. (2020); Gao et al. (2023); Wu et al. (2024); Yu
et al. (2024).
20

Preprint
A particular feature of this domain is its high sensitivity to the wording of the answer. For some
questions a brief factual restatement is acceptable, while for others an almost literal form of the
answer is essential: an exact flag, path, field name, API version, or specific parameter value. The
system must therefore be able to handle both ordinary documentation-grounded answers and cases
that require near-literal reproduction; for documentation question answering this is directly tied to the
requirements of factual accuracy and degree of support by the retrieved context Yu et al. (2024); Es
et al. (2024).
B.2 PROBLEM STATEMENT
We consider the task of building a RAG system that, given a user question about the Kubernetes
documentation, must retrieve the relevant fragments of the corpus and generate an answer based only
on the retrieved context. Such a formulation is in line with Retrieval-Augmented Generation for
tasks that require access to an external knowledge base and with documentation question-answering
systems Lewis et al. (2020); Gao et al. (2023); Wu et al. (2024); Yu et al. (2024).
LetD={d 1, . . . , d n}denote the documentation corpus, represented as a set of text chunks, and let
Rbe the fixed retrieval-and-reranking pipeline that for a question qselects the context c=R(q, D) .
Letx∈X denote the generator configuration inside this fixed retrieval pipeline, that is, the choice of
base model, LoRA rank, and the scope of adapted modules. For each configuration x, the system
maps a question qto an answer ax(q, c) , where cis determined solely by the retrieval and reranking
procedures, and the main research focus shifts to the generator adaptation and to comparing several
adapter configurations Karpukhin et al. (2020); Robertson & Walker (1994); Robertson et al. (1995);
Robertson & Zaragoza (2009); Zhao et al. (2024b); Qu et al. (2021); Santhanam et al. (2022); Chen
et al. (2024); Nogueira & Cho (2019); Yoon et al. (2024); Hu et al. (2022); Dettmers et al. (2023);
Hayou et al. (2024); Han et al. (2024); Wang et al. (2025); Ye et al. (2023).
For each configuration we introduce a quality function Q(x) =F1(x) and a cost vector C(x) =
(Linf(x), M inf(x), T train(x), M train(x)), where Linfis the mean inference latency, Minfis the peak
inference memory consumption, Ttrain is the total training time, and Mtrain is the peak training
memory consumption. For a specific deployment scenario the choice of configuration can be written
as a constrained optimization problem of the form x∗= arg maxQ(x) subject to constraints on one
or several components ofC(x).
However, the main interest of this work is not in a single solution x∗, but in the set of Pareto-optimal
configurations:
P={x∈X|∄x′∈X:Q(x′)≥Q(x),
Cj(x′)≤C j(x)for allj,and at least one inequality is strict}.
This formulation matches the practical logic of the task, because the applied value of a documentation-
oriented RAG assistant is determined not only by the maximum answer quality, but also by conflicting
constraints on latency, memory, and training cost; this logic is consistent with works on multi-criteria
evaluation of RAG and LoRA Yu et al. (2024); Es et al. (2024); Baek et al. (2026); Baqar & Khanda
(2025).
B.3 TASK SPECIFICS
The task under study has a number of features that make it non-trivial compared with ordinary
generative question answering. First, answers are heterogeneous in nature: some questions require
literal reproduction of a value, while others demand a concise but complete and correct description of
a fact. Second, generation quality strongly depends on the quality of retrieval and reranking, because
the generator is restricted to the provided context. An error at the retrieval stage directly affects the
final answer even with a strong generator.
Third, for a practical system computational constraints are critical. Even if a more complex con-
figuration improves quality, it may be unsuitable in real conditions due to increased latency or
memory consumption. Fourth, tuning LoRA inside a RAG system does not reduce to standard
fine-tuning of a language model: one must account for the interaction between the retrieval stage,
the prompt-formation mode, and the generator adaptation scheme. This is precisely why the work
21

Preprint
uses a Pareto-based approach, which makes it possible to analyse not a single best point but a set of
non-dominated configurations.
C DATA ANDCORPUS
C.1 DATA SOURCE
The data source is the Kubernetes documentation. The corpus was pre-cleaned and converted into a set
of text fragments suitable for retrieval and for grounding the answer in the supporting context. During
corpus preparation, the documentation pages were brought into a structured textual representation
and then segmented into semantically meaningful chunks. This segmentation follows the general
logic of building corpora suitable for retrieval in RAG systems Gao et al. (2023); Wu et al. (2024);
Yu et al. (2024); Karpukhin et al. (2020); Chen et al. (2024).
C.2 CONSTRUCTION OF THEQASETS
The question-answer set was assembled in a combined way: 500 QA pairs were written by the author
manually, while the remaining part was prepared using an AI agent based on the GPT-5.4 model
(through the agent interface in the Cursor environment). Generation was performed one pair at a
time: each question-answer pair was created individually from the Kubernetes documentation rather
than obtained through bulk automated augmentation. The resulting set then underwent a full manual
verification pass by the author, including verification of question correctness, reference answers, and
their correspondence to the source fragments of the documentation. During verification, successive
cleanup passes were applied to the whole set; a pair was discarded if the question was malformed or
ambiguous, if no answer was present in the documentation corpus, if it was a duplicate of another
pair, or if the reference answer contained a factual error. As a result of the cleanup passes, 323 pairs
out of an initial pool of 5467 QA candidates were rejected (about 5.9%), and approximately 20% of
the remaining pairs required editing of the wording or rephrasing of the reference answer to match
the format of short documentation answers; the remaining pairs were accepted without changes. This
logic is consistent with modern approaches to evaluating RAG systems, where both the quality of the
answer and its support by the retrieved context matter Yu et al. (2024); Es et al. (2024). The final
verified set of 5144 pairs (500 written manually from scratch and 4644 automatically generated and
manually verified) was split into training, validation, and test parts. In the current configuration of the
work we use the train, eval, and test splits, intended respectively for training adapters, selecting and
comparing configurations, and the final evaluation.
In addition, it is worth recording the relationship between the model used to generate QA candidates
(GPT-5.4) and the model used as the judge in §6 ( gpt-5.4-mini ). Although both models belong
to the same family, no direct train/test leak arises between them: the judge was not trained on the gold
answers, and the reference answers themselves underwent a full manual verification by the author.
The residual risk of a same-family bias is discussed separately in Appendix A as a limitation of the
judge-based metrics.
Thus the work produces not only a question-answer set, but an integrated evaluation benchmark:
the documentation corpus, the QA set, the train/eval/test split, the metric system, and a unified
protocol for ablation and Pareto experiments. This makes the resulting set usable not only for training
and testing individual adapters, but also for the reproducible comparison of retrieval and LoRA
configurations within a common experimental setting.
C.3 ANSWER TYPES
In the QA set two answer types are distinguished: exact andnormal . The exact class covers
cases in which the almost-literal form of the answer matters, for example flags, paths, field names,
API version, feature gates, and other short literal values. The normal class covers ordinary factual
answers, grounded in the documentation and admitting careful rephrasing while preserving meaning.
This distinction is introduced in the present work as part of our own methodology for annotating and
analysing answers.
22

Preprint
This annotation was used both for manual analysis of the QA set and during adapter training. It is
important to emphasize that in the current main inference regime the generator does not receive the
answer-type label ( exact ornormal ) as an input signal and operates in a neutral prompt-formation
mode. The answer-type annotation is therefore used in the work primarily as a supervision signal
during training and as a source of stratified quality metrics, but not as an explicit condition at test-time
inference; this scheme methodologically aligns with the separation of training and testing regimes in
RAG-evaluation works Yu et al. (2024); Es et al. (2024).
C.4 DATASET CHARACTERISTICS
The final sizes of the splits are given in Table 9.
Split Rows Exact Normal
Train 3614 1449 2165
Eval 745 361 384
Test 785 475 310
Table 9: Sizes of the training, validation, and test parts of the QA set
Table 9 records the sizes of the training, validation, and test parts of the set, as well as the distribution
of answer types within each part. These characteristics are used further as a description of the
experimental base on which adapter training and final configuration evaluation are performed.
D RAG SYSTEMARCHITECTURE
D.1 OVERALL SOLUTION SCHEME
The system under study is a Retrieval-Augmented Generation pipeline for question answering over
technical documentation. Given a user question, the system first retrieves a set of candidates from the
documentation corpus, then reranks the retrieved fragments, and finally passes the selected context to
the generator. This three-step scheme of retrieval, reranking, and generation matches modern views
on RAG systems Lewis et al. (2020); Gao et al. (2023); Wu et al. (2024); Yu et al. (2024).
D.2 RETRIEVAL COMPONENT
In the current main configuration we use a hybrid retrieval pipeline: dense retrieval based on a FAISS
index is combined with native sparse retrieval based on BGE-M3, and the results are merged using
Reciprocal Rank Fusion. This scheme draws on works on dense retrieval, BM25, hybrid retrieval, and
result fusion Karpukhin et al. (2020); Robertson & Walker (1994); Robertson et al. (1995); Robertson
& Zaragoza (2009); Zhao et al. (2024b); Qu et al. (2021); Santhanam et al. (2022); Chen et al. (2024);
Douze et al. (2024); Johnson et al. (2019); Cormack et al. (2009); Li et al. (2023).
D.3 RANKING AND RERANKING
After the initial retrieval step the candidates additionally pass through reranking. In the system under
study we use the pretrained reranker BAAI/bge-reranker-v2-m3 . The very idea of reranking
goes back to classical works on cross-encoder and listwise reranking Nogueira & Cho (2019); Yoon
et al. (2024); for the BGE family the work on dense retrieval based on large language models Li et al.
(2023) is also relevant.
D.4 GENERATOR AND PROMPTING
We consider two instruction-tuned models of the Llama 3 family as generators: meta-
llama/Llama-3.2-3B-Instruct andmeta-llama/Llama-3.1-8B-Instruct . Both
models belong to successive releases of the Llama 3 family Dubey et al. (2024) – versions 3.2 and
3.1, respectively; they share the common architectural line of decoder-only transformers with RoPE,
23

Preprint
GQA attention, and SwiGLU blocks described in Dubey et al. (2024). In the main inference regime
we use a neutral prompt-formation mode that minimizes explicit control over answer style.
This prompt-formation mode allows adapters to be evaluated in a single inference regime and thus
compared on a fairer basis. At training time, the option to condition on the answer type is preserved,
which is important for handling both literal and ordinary factual answers; such a separation of training
and testing regimes methodologically aligns with works on RAG evaluation Yu et al. (2024); Es et al.
(2024).
E LORA ADAPTATION ANDEXPERIMENTALDESIGN
E.1 MOTIVATION FOR USINGLORA
Full fine-tuning of modern instruction-tuned language models is costly both in memory and in time.
We therefore use parameter-efficient fine-tuning, namely LoRA. This choice rests directly on works
on LoRA, QLoRA, and general PEFT surveys Hu et al. (2022); Dettmers et al. (2023); Hayou et al.
(2024); Han et al. (2024); Wang et al. (2025). For our research task, in which several adaptation
configurations have to be compared, this approach is particularly convenient.
E.2 TRAINED CONFIGURATIONS
The current set of training artefacts comprises two baseline models and 20 trained LoRA configura-
tions. For each base model we considered LoRA configurations with several rank values and two
attention-module coverage schemes: adaptation of the query and value projections only ( q_proj ,
v_proj ) and adaptation of all main projections in the attention block ( q_proj ,k_proj ,v_proj ,
o_proj).
Base model Ranks Adaptation schemes Number of LoRA configs
3B4, 8, 16, 32, 64 qv_only,full_attention10
8B4, 8, 16, 32, 64 qv_only,full_attention10
Table 10: Composition of the trained LoRA configurations: base models, rank options, and adaptation
schemes
The baseline configurations 3B baseline and8B baseline were used as LoRA-free reference
points and were compared against the trained adapters on the same evaluation metrics.
E.2.1 BASELINE EVALUATION PROTOCOL
For the baseline configurations we used the same inference protocol as for the main LoRA-adapter
comparisons. The evaluation was performed in the neutral prompting mode, without any additional
explicit requirement of a grounded-answer style in the prompt. On the retrieval side we used the same
base pipeline as in the main 01_base__neutral regime: dense retrieval, native sparse retrieval,
candidate fusion via reciprocal rank fusion, and subsequent reranking with the pretrained reranker.
The number of context fragments at inference time was also kept fixed and matched the main com-
parison regime: top_k = 2 . Thanks to this, the comparison of 3B baseline /8B baseline
with the LoRA configurations was performed in the same retrieval and prompting pipeline in which
the main results of §7 are interpreted, rather than in a separate special regime. This is important for
the correctness of the subsequent comparisons, including the discussion of the closeness between 3B
r64 qv_onlyand8B baselineonF1.
24

Preprint
E.3 FIXED TRAINING HYPERPARAMETERS
Parameter Value Comment
embed_top_k 2Number of context fragments in a training example
num_train_epochs 8Total number of epochs
learning_rate 2e-5Single value used for all runs
Table 11: Fixed training hyperparameters shared across all LoRA configurations
In the current set of training runs a single scheme was used to form the training context. Across all
configurations the number of retrieved context fragments was fixed as embed_top_k = 2 , that is,
every training example was built on top of two context fragments.
All runs in the current set used num_train_epochs = 8 andlearning_rate = 2e-5 .
The base optimization scheme follows the standard PEFT setup: the AdamW optimizer Kingma & Ba
(2015); Loshchilov & Hutter (2019) with a cosine schedule on the learning rate Loshchilov & Hutter
(2017) and a linear warm-up over approximately ∼3% of the total training steps; mixed precision
is set to bf16 . Compared configurations were therefore trained in a comparable regime, and any
differences in quality and cost can be attributed primarily to the adapter parameters rather than to
changes in the optimization regime.
E.4 LORAADAPTATION PARAMETERS
For all runs we used bias = none ,task_type = CAUSAL_LM , and lora_dropout =
0.05 . The lora_alpha coefficient was chosen proportional to the rank via the rule alpha =
2r. As a result, when comparing configurations we varied primarily the model scale, the rank, and
the scope of adapted modules, while the remaining LoRA parameters stayed fixed.
Configuration aspect Value
r = 4 lora_alpha = 8
r = 8 lora_alpha = 16
r = 16 lora_alpha = 32
r = 32 lora_alpha = 64
r = 64 lora_alpha = 128
lora_dropout 0.05
qv_only q_proj,v_proj
full_attention q_proj,k_proj,v_proj,o_proj
Table 12: LoRA adaptation parameters: correspondence of rank to lora_alpha , dropout, and the
attention-module coverage for the two compared schemes
In terms of the scope of adapted modules, two main schemes were used in the current experimental
set. The qv_only configuration adapts only the q_proj andv_proj projections. The full_-
attention configuration extends adaptation to all main attention projections: q_proj ,k_proj ,
v_proj, ando_proj. This comparison makes it possible to assess whether wider attention-block
coverage justifies the additional training and inference cost inside the same RAG architecture.
E.5 TRAINING-CONTEXT FORMATION REGIME
Adapter training does not rely on a single context regime. We use a mixed scheme for forming
the training example. If the example has pre-labelled supporting chunks, exactly those are fed as
the supporting context. If no such labelling exists, the context is formed by the retrieval procedure.
This scheme is methodologically close to the retrieval-augmented fine-tuning settings described in
JORA Tahir et al. (2024) and helps reduce the gap between idealized training and the actual inference
regime.
25

Preprint
E.6 HARDWARE CONFIGURATION AND INFERENCE STACK
All runtime cost metrics ( Latency ,Inference VRAM ,Training Time ,Training VRAM )
were measured in a single compute environment, so as to remove any contribution of hardware to
the relative differences between configurations. The measurements were performed in Yandex
DataSphere on a single compute node with the following characteristics: 1 ×NVIDIA A100 40 GB,
28 vCPU, and 114 GB of RAM. The same node was used both for training the LoRA adapters and
for subsequent inference; thanks to this Training Time ,Training VRAM ,Latency , and
Inference VRAMremain comparable across the entire set of experiments.
The inference stack is based on Hugging Face Transformers and PEFT Dettmers et al. (2023);
Hayou et al. (2024): the base model ( meta-llama/Llama-3.2-3B-Instruct ormeta-
llama/Llama-3.1-8B-Instruct ) is loaded in the original precision without quantization (the
flags--no-quant-generator and--no-quant-judge ), and the LoRA adapter is loaded
separately on top of the base model via PeftModel.from_pretrained ; the mixed-precision
setting matches bf16 , as during training (see §E.5). The retrieval pipeline uses BAAI/bge-m3
for dense embeddings and BAAI/bge-reranker-v2-m3 for final reranking ( reranker_-
batch_size = 16 ,retrieve_top_n = 20 ); in the main comparison regime the generator
operates with a fixed eval_top_k = 2 retrieved fragments, which matches the training-context
formation scheme ( embed_top_k = 2 , see §E.5); additionally, for sensitivity control to the
parameter, runs were performed with eval_top_k∈ {1,4} . Generation is performed in greedy
decoding (no sampling) with the maximum answer length fixed in the code and identical across all
configurations.
Latency is measured per sample (the effective generator batch size equals one), as the mean time
of the complete processing of a single test example – from receiving the query to returning the
generated answer – over the entire test split n = 785 , and includes all pipeline stages: building the
query embedding, dense and sparse retrieval, reranking, prompt formation, and answer generation.
Inference VRAM is recorded as the peak GPU memory consumption on the device, obtained via
torch.cuda.max_memory_allocatedat the end of the run.
E.7 EXPERIMENTAL HYPOTHESES
The work tests the following hypotheses:
• scaling the model up improves quality but raises latency and memory consumption;
•increasing the rank expands the adaptation capacity, but does not always yield a proportional
gain in quality;
•a wider coverage of adapted modules is not necessarily preferable once training and inference
cost are accounted for;
•among the set of configurations there exist practical working points that do not maximize
quality but offer the best balance between quality and cost.
F EVALUATIONMETHODOLOGY
F.1 QUALITY METRICS
The main quality metric in the work is token-level F1, used to compare configurations by the degree of
agreement between the answer and the reference Yu et al. (2024). The final F1values, the judge-based
metrics, the summary tables of §7 and the corresponding plots are computed on the final test split ( n =
785). The eval split ( n = 745 ) is used for selection and intermediate comparison of configurations
during the experimental cycle. In addition to point estimates we also compute 95% bootstrap
confidence intervals on the test set (1000 resamples). For key configuration comparisons we use the
paired bootstrap for the difference∆F1, which separates stable differences from narrow gaps lying
within sampling noise. Semantic embedding-based metrics such as BERTScore Zhang et al. (2020)
are not used in the main setting of the work: for documentation-oriented question answering with a
high share of exact questions, token-level F1 provides a more direct and interpretable comparison
with the reference, while the semantic dimension of quality is covered separately by the judge-based
correctness and groundedness scores (see Section 5.2).
26

Preprint
F.2 JUDGE-BASED GROUNDEDNESS AND CORRECTNESS
In addition, we use judge-based evaluation of groundedness and correctness. In this scheme an
external judge language model gpt-5.4-mini receives the question, the retrieved context, and
the generated answer, after which it assigns two independent scores: correctness as the degree of
semantic correctness of the answer relative to the provided context, and groundedness as the degree
to which the answer is supported by the retrieved fragments without unsupported additions. Evalua-
tion is performed in a blind regime: the judge receives only the triple (question, context,
answer) and does not receive the model_id , the configuration name, or any other generator
metadata. This evaluation is used not as a replacement for the main F1metric, but as an additional
axis of quality that supports the analysis of supportedness of the answer by the retrieved context and
of its faithfulness to that context Yu et al. (2024); Es et al. (2024); Baqar & Khanda (2025).
The main aggregates of the judge-based evaluation are correctness_pass@4 andgrounded-
ness_pass@4 , that is, the fraction of answers that received a score of at least 4 on the corresponding
scale. These metrics are especially useful when several configurations have similar F1values but
differ in the degree of context support. In particular, the subsequent ablation experiments show
that the configuration optimal in F1does not necessarily coincide with the configuration optimal in
groundedness, which indicates a stable trade-off between task quality and supportedness.
F.3 COST METRICS
To analyse computational cost we use both deployment and training metrics. The former include the
mean answer latency and the peak GPU memory consumption at inference. The latter include the
total training time and the peak GPU memory consumption at training. Such a set reflects the idea
of jointly analysing quality and cost, emphasized in works on the efficiency evaluation of RAG and
LoRA Es et al. (2024); Baek et al. (2026); Baqar & Khanda (2025).
F.4 PARETO ANALYSIS
To compare configurations in a multi-criteria setting we use Pareto analysis. A configuration is
considered non-dominated if no other configuration is no worse in quality and at the same time no
worse in cost, while being strictly better in at least one dimension. In two-dimensional fronts the role
of the quality function is played by F1, while the role of cost is taken in turn by the mean inference
latency, inference memory, training time, and training memory. This approach makes it possible to
identify practical working points and to avoid simplistic model selection based on a single metric;
the multi-criteria logic of such a comparison is consistent with works in which quality and cost are
analysed jointly Baek et al. (2026); Baqar & Khanda (2025).
G EXPERIMENTALRESULTS
In this appendix the results of comparing baseline systems and LoRA adapters are interpreted as
multi-criteria trade-offs between quality, latency, memory, and training cost. All summary results in
Tables 13, 14, 15 and 16, and in the corresponding plots, are computed on the final test split ( n =
785).
27

Preprint
G.1 QUALITY AND LATENCY
Figure 7: F1 vs Latency
On the main response-time front the non-dominated configurations are 3B r64 qv_only and
8B r64 qv_only . The first represents the strongest lightweight point, and the second yields the
highest quality in the current set of experiments. These points are listed in Table 13.
Configuration F1 [95% CI] EM Latency, s Inference VRAM, GB
3B r64 qv_only0.597 [0.574, 0.622] 0.250 0.598 12.760
8B r64 qv_only0.617 [0.593, 0.639] 0.250 0.655 21.926
Table 13: Non-dominated points on the quality–latency front
Table 13 shows that the main response-time front (the runtime front) in the base regime consists of
two points, between which the main practical trade-off lies. The transition from 3B r64 qv_only
to8B r64 qv_only brings approximately a 0.020 gain in F1, and the paired bootstrap for ∆F1
gives a 95% CI of [+0.0005, +0.0410] . This indicates a small but test-set-supported advantage
of the 8B configuration in F1, which at the same time comes with a transition to a substantially more
expensive inference-memory regime and a latency increase of approximately 0.057 s. Importantly, 3B
r64 qv_only withF1 = 0.597 [0.574, 0.622] turns out to be statistically comparable
to the unadapted 8B baseline configuration ( 0.595 [0.572, 0.620] ): for ∆F1 the 95%
CI is[-0.021, +0.026] . The choice between these points is therefore determined not only by
quality, but also by whether the system needs a lightweight 3B regime or a more expensive 8B regime
with the highest point quality.
28

Preprint
G.2 QUALITY AND INFERENCE MEMORY
Figure 8: F1 vs Inference VRAM
The inference-memory plot shows a clear separation between the 3B and 8B families. The 3B
configurations occupy a more compact area in memory and therefore remain most attractive for
constrained hardware.
G.3 TRAINING COST
Figure 9: F1 vs Training Time
29

Preprint
Figure 10: F1 vs Training VRAM
In terms of training time and training memory the cheapest point remains 3B r4 qv_only ,
however the current training Pareto fronts are no longer limited to the lowest ranks. The strongest
training-side trade-offs form around qv_only configurations with ranks r = 32 andr = 64
in the 3B family and r = 16 /r = 64 in the 8B family. The composition of the non-dominated
training-front points is given in Table 14.
Configuration F1 [95% CI] Training time, min Train VRAM, GB Front
3B r4 qv_only0.572 [0.550, 0.592] 52.95 19.072 vram
3B r8 qv_only0.573 [0.552, 0.594] 54.08 19.092 vram
3B r16 qv_only0.583 [0.562, 0.605] 55.90 19.131 vram
3B r32 qv_only0.592 [0.569, 0.614] 52.41 19.203 time, vram
3B r64 qv_only0.597 [0.574, 0.622] 53.32 19.387 time, vram
8B r4 qv_only0.610 [0.588, 0.633] 68.57 31.307 vram
8B r16 qv_only0.615 [0.593, 0.634] 68.38 31.391 time, vram
8B r64 qv_only0.617 [0.595, 0.640] 68.93 31.740 time, vram
Table 14: Non-dominated points by training criteria
Table 14 shows that inside the 3B family the move to r = 32 andr = 64 is accompanied by a
noticeable improvement in F1, while training time and memory grow only moderately compared
to the lower ranks. At the same time, full_attention configurations do not appear at all on
the current training Pareto fronts, which further emphasizes the structural advantage of qv_only .
For the 8B family the differences in training time between qv_only configurations are also small
relative to the overall backbone cost, while the upper points r = 16 andr = 64 have almost fully
overlapping F1intervals, so the choice between them is driven more by additional criteria than by a
stable quality gap.
30

Preprint
G.4 JUDGE-BASED GROUNDEDNESS AS A SECOND QUALITY AXIS
Figure 11: F1 vs Groundedness pass@4
Alongside the main task metric F1, the work additionally considered a judge-based groundedness
evaluation, aggregated as groundedness_pass@4 . This metric measures how often the model
produces answers that are sufficiently grounded in the retrieved context and contain no unsupported
additions. Unlike F1, it captures not just agreement with the reference answer but also the degree of
supportedness of the result inside the retrieval-augmented pipeline.
The plot shows that configurations with the best point F1do not necessarily coincide with con-
figurations optimal in groundedness_pass@4 . In the base regime the best- F1configuration
8B r64 qv_only attains F1 = 0.617 [0.593, 0.639] , while the groundedness maxi-
mum ( 0.825 ) is achieved by 8B r16 qv_only atF1 = 0.615 [0.592, 0.635] . For the
paired difference ∆F1 between them the 95% CI is [-0.011, +0.016] , that is, on F1these
two points are statistically indistinguishable on the current test set. This means that a gain in task
quality is not equivalent to a gain in the factual support of the answer by the context. Consequently,
for documentation-oriented question answering the quality of the system cannot be fully described by
F1alone: some configurations reach higher groundedness at a statistically comparable level of the
main metric.
31

Preprint
Figure 12: Groundedness pass@4 vs Latency
The comparison of groundedness_pass@4 against latency further shows that the LoRA-free
baseline configurations ( 3B-base ,8B-base ) systematically lie below all adapted points: at
comparable latency, LoRA fine-tuning shifts groundedness_pass@4 upward by 0.05–0.08, and
in absolute value this shift exceeds the width of the 95% CI ( ≈0.03 ). Within already-adapted
configurations the groundedness_pass@4 spread mostly falls inside the CI band, and the
separation is driven mainly by the 3B/8B split. Latency and groundedness therefore do not form a
non-trivial front beyond the one already visible on F1, but fine-tuning remains a necessary condition
for a high level of supportedness. The resource view by inference VRAM carries no additional
information either: within each family the VRAM spread does not exceed 0.3–0.4 GB, while the
corresponding groundedness spread fully lies within the 95% CI band, so the corresponding
projection is not shown.
Thus the judge-based analysis confirms the overall multi-criteria logic of the work. Practically useful
configurations differ not only in F1and resource cost, but also in the degree of supportedness, so the
choice of working point depends on what exactly is considered the priority in the applied scenario:
maximum agreement with the reference, stricter groundedness, or a trade-off between them.
G.5 ABLATION EXPERIMENTS ON RETRIEVAL AND PROMPTING
To check the robustness of the conclusions about LoRA configurations, additional ablation experi-
ments were conducted in which not only the adapter parameters but also other parts of the pipeline
were varied: the retrieval pipeline and the prompt-formation mode. We considered ten regimes
obtained by combining five retrieval settings ( base ,reranker_off ,dense_only ,sparse_-
only ,hybrid_bm25 ) and two prompting modes ( neutral ,explicit_grounded ). For each
regime two reference points were extracted from the 22 configurations: the configuration with the
maximumF1and the configuration with the maximumgroundedness_pass@4.
The base regime ( base ) used the full retrieval pipeline, including dense retrieval, native sparse
retrieval, their fusion via reciprocal rank fusion, and subsequent reranking of the candidates by
the pretrained reranker. The reranker_off regime excludes only the final reranking step while
keeping hybrid retrieval. The dense_only regime disables the sparse branch and uses only dense
retrieval, while sparse_only , conversely, excludes the dense component and uses only sparse
retrieval. The hybrid_bm25 regime preserves the hybrid scheme but replaces native sparse retrieval
32

Preprint
with classical BM25, which makes it possible to compare two variants of the sparse component inside
the same retrieval pipeline. On top of the retrieval ablations, two prompting modes were considered:
neutral , corresponding to a neutral generation mode without explicitly requiring the answer to
rely on the context, and explicit_grounded , in which the prompt explicitly emphasizes the
need to answer strictly based on the retrieved material.
The summary results are given in Table 15. The table allows us to see whether the main conclusions
about LoRA are preserved when retrieval/prompting is varied, how the optimal working point shifts
within a regime, and whether the points optimal in task quality and in groundedness coincide. An
aggregated view by adaptation scheme, around which the optimal points concentrate, is additionally
shown in Figure 13. The full per-regime plots and the detailed tables with all configurations are
moved to the appendices in order not to overload the main exposition.
Figure 13: Distribution of optimal configurations by adaptation scheme ( qv_only ,full_atten-
tion ,baseline ) across the ten ablation regimes: number of regimes in which the scheme gives
the best point onF1and ongroundedness_pass@4.
retrieval prompt best F1 F1 [95% CI] grnd@4 corr@4 lat. inf. VRAM best grnd F1 [95% CI] grnd@4 corr@4 lat. inf. VRAM same
config (s) (GB) config (s) (GB) point
base neutral 8B r64 qv_only 0.617 [0.593, 0.639] 0.818 0.827 0.655 21.926 8B r16 qv_only 0.615 [0.592, 0.635] 0.825 0.836 0.690 21.824 no
base explicit 8B r64 qv_only 0.623 [0.599, 0.645] 0.822 0.824 0.664 21.953 8B r16 qv_only 0.610 [0.589, 0.632] 0.823 0.834 0.662 21.852 no
no_rerank neutral 8B r64 qv_only 0.617 [0.596, 0.638] 0.819 0.824 0.658 21.926 8B r16 qv_only 0.615 [0.592, 0.637] 0.836 0.842 0.685 21.824 no
no_rerank explicit 8B r64 qv_only 0.623 [0.601, 0.645] 0.817 0.824 0.649 21.953 8B r16 qv_only 0.610 [0.589, 0.632] 0.829 0.834 0.675 21.852 no
dense neutral 8B r64 qv_only 0.617 [0.595, 0.640] 0.818 0.828 0.650 21.926 8B r8 qv_only 0.606 [0.585, 0.628] 0.824 0.831 0.673 21.836 no
dense explicit 8B r64 qv_only 0.623 [0.600, 0.645] 0.824 0.827 0.661 21.953 8B r16 qv_only 0.610 [0.588, 0.633] 0.827 0.832 0.679 21.852 no
sparse neutral 8B r64 qv_only 0.617 [0.595, 0.640] 0.820 0.823 0.661 21.926 8B r4 full_attention 0.603 [0.580, 0.626] 0.827 0.832 0.801 21.871 no
sparse explicit 8B r64 qv_only 0.623 [0.600, 0.645] 0.828 0.832 0.646 21.953 8B r16 full_attention 0.601 [0.580, 0.623] 0.829 0.834 0.766 21.877 no
bm25_hybrid neutral 8B r64 qv_only 0.617 [0.597, 0.639] 0.815 0.824 0.659 21.926 8B r8 qv_only 0.606 [0.586, 0.627] 0.827 0.836 0.646 21.836 no
bm25_hybrid explicit 8B r64 qv_only 0.623 [0.603, 0.645] 0.823 0.829 0.664 21.953 8B r16 qv_only 0.610 [0.590, 0.631] 0.834 0.838 0.691 21.852 no
Table 15: Ablation summary results byF1andgroundedness_pass@4
Table 15 shows that the main conclusions about LoRA do not disappear when retrieval and prompting
are changed. At the level of the entire ablation series the same structural pattern is consistently
preserved: the points with the maximum point F1and the points optimal in groundedness in most
regimes concentrate around qv_only adapters, while full_attention only occasionally turns
out to be optimal in groundedness. At the same time, the interval estimates show that within individual
regimes some of the best- F1configurations are statistically close to their nearest alternatives. Under
moderate changes of the retrieval pipeline and the prompting policy what is therefore reproduced is
not so much the advantage of a particular backbone, but the advantage of the qv_only adaptation
scheme itself.
At the same time, the optimal working point within a regime shifts depending on the chosen quality
criterion. In all ten regimes the points with the maximum F1do not coincide with the points
with the maximum groundedness_pass@4 (same_point = no ). As a result, retrieval and
33

Preprint
prompting affect not only the absolute level of quality, but also the position of the trade-off between
task performance and groundedness. The ablation results therefore confirm the methodological
necessity of considering LoRA configurations not as a single source of variation, but as part of a
more general retrieval-augmented system with conflicting criteria of quality and cost.
G.6 STATISTICAL ROBUSTNESS OFF1DIFFERENCES
To separate stable F1differences from narrow gaps, for the key comparisons we computed non-
parametric 95% bootstrap CIs on the test set ( n = 785 , 1000 resamples). For individual configura-
tion pairs we additionally used the paired bootstrap for the difference ∆F1 , since it correctly assesses
whether the observed gap crosses the boundary of statistical noise.
Comparison∆F195% CI on∆F1Significant?
3B r64 qv_only-3B baseline+0.056 [+0.033, +0.078] yes
3B r64 qv_only-8B baseline+0.003 [-0.021, +0.026] no
8B r64 qv_only-3B r64 qv_only+0.020 [+0.0005, +0.0410] yes
8B r64 qv_only-8B r16 qv_only+0.003 [-0.011, +0.016] no
3B r64 qv_only-3B r64 full_attention+0.020 [+0.004, +0.036] yes
8B r64 qv_only-8B r64 full_attention+0.011 [-0.003, +0.025] no
qv_only-full_attention*+0.0067 [+0.0010, +0.0124] yes
*mean∆F1over 8 param-matched pairs from Table 17.
Table 16: Paired bootstrap estimates of keyF1differences
Table 16 shows that the statistically stable effects on the current test set are primarily three: the
improvement of 3B r64 qv_only over3B baseline , the superiority of 3B r64 qv_only
over3B r64 full_attention , and the positive average difference of qv_only overfull_-
attention across matched rank pairs. On the contrary, the difference between 3B r64 qv_-
only and8B baseline , as well as between 8B r64 qv_only and8B r16 qv_only , does
not allow us to claim strict statistical superiority. The main stable conclusion of the work is therefore
not connected to any local maximum on F1, but to the fact that LoRA adaptation in the successful
qv_only scheme systematically forms stronger working points and gives a confirmed gain in at
least some of the key comparisons.
H DISCUSSION
H.1 WHYQ V_O N L YADAPTERS DOMINATE
The obtained results show that the strongest response-time fronts are formed not by the wider
full_attention adapters, but by the narrower qv_only solutions. This is confirmed by the
bootstrap analysis: the average difference qv_only - full_attention over matched rank
pairs is positive and statistically significant ( ∆F1 = +0.0067 , 95% CI [+0.0010, +0.0124] ),
with the effect being most pronounced in the 8B family. The most plausible explanation is that, for
documentation-oriented question answering inside a fixed retrieval pipeline, the main task of the
adapter is not a deep reorganization of the entire attention block, but a more accurate tuning of how
the model selects and uses the context already given to it. In this setting, selective adaptation of the
q_proj andv_proj projections turns out to be sufficient, while extending LoRA to k_proj and
o_proj increases the adapter size and training cost faster than it brings a stable benefit on the Pareto
fronts.
The comparison of 3B r64 qv_only and3B r64 full_attention is particularly telling.
In the current set of results the broader module coverage not only fails to improve quality, but also
loses in F1; for∆F1 between these configurations the 95% CI is [+0.004, +0.036] . This
means that in the current set of experiments full_attention acts not as a universally better
strategy, but as a more expensive way to obtain a less stable result.
34

Preprint
H.1.1 PARAM-MATCHED COMPARISON OFQ V_O N L YVSF U L L_A T T E N T I O N
An additional control check is the param-matched analysis, in which pairs of configurations with
the same total number of LoRA parameters are compared. Since full_attention adapts four
attention projections while qv_only only two, the same parameter budget is reached at half the
rank for full_attention . For every such pair on the final test split ( n = 785 ) we computed
∆F1 =F1(qv_only)−F1(full_attention) and the paired bootstrap 95% CI. The summary
is given in Table 17.
Family Param budgetqv_only full_attention∆F195% CI Significant?
3B 256dr = 64 r = 32+0.0207 [+0.0052, +0.0373] yes
3B 128dr = 32 r = 16+0.0021 [-0.0133, +0.0174] no
3B 64dr = 16 r = 8+0.0054 [-0.0083, +0.0203] no
3B 32dr = 8 r = 4-0.0016 [-0.0137, +0.0109] no
8B 256dr = 64 r = 32+0.0150 [-0.0013, +0.0301] no
8B 128dr = 32 r = 16+0.0083 [-0.0056, +0.0225] no
8B 64dr = 16 r = 8+0.0158 [+0.0034, +0.0265] yes
8B 32dr = 8 r = 4+0.0032 [-0.0093, +0.0170] no
Table 17: Param-matched comparison of qv_only andfull_attention by paired bootstrap
F1differences
Table 17 shows that at the same parameter budget qv_only is significantly better in 2 of 8 pairs,
is statistically comparable in the remaining 6, and in no pair is significantly worse than full_-
attention . It follows that the claim of a structural advantage of qv_only does not reduce to
the fact that this scheme simply “gets more parameters per projection” at a fixed rank: even in the
param-matched setting it is at least no worse and in some comparisons yields a confirmed gain in F1.
H.2 WHY THE BACKBONE SIZE SETS THE OPERATING REGIME BUT DOES NOT CANCEL THE
LORAEFFECT
The split between 3B and 8B configurations looks structural, but it should not be interpreted as
the main result of the work. The larger 8B model does possess a greater parametric capacity and
therefore more often sets the upper bound on F1, but it almost automatically moves the system
into a more expensive regime in terms of inference and training memory. The 3B family, on the
contrary, forms an area of efficient solutions: with a noticeably smaller resource profile it retains
quality high enough to remain a practical choice for constrained hardware. It is particularly telling
that the strong adapted configuration 3B r64 qv_only turns out to be statistically comparable to
the unadapted 8B baseline model: the point difference in F1is only +0.003 , while the 95% CI
for∆F1 is[-0.021, +0.026] . The difference between 3B and 8B therefore cannot be reduced
to a simple rule of “more parameters means better”: domain-specific adaptation can compensate for
the advantage of a larger LoRA-free backbone even where a strict conclusion about superiority in F1
can no longer be drawn.
Thus the backbone scale primarily sets the operating regime of the system, but does not answer
the question of which adaptation scheme is successful. This question is decided inside each family
separately, and this is where the main effect of the work appears: in both the 3B and the 8B family the
qv_only adapters turn out to be more competitive in the quality-cost ratio than full_attention .
The difference between 3B and 8B should therefore be understood as a background against which a
more important regularity, related to the structure of LoRA adaptation itself, becomes visible.
H.3 WHY THE EFFECT OF RANK GROWTH IS NON-LINEAR
Increasing the rank does not produce the same effect for all configurations, however in both families
an important regularity is observed: increasing rimproves quality more strongly than it raises the
runtime cost. For 3B qv_only adapters the move from r = 4 tor = 64 raises F1from0.572
to0.597 , while the latency of these two extreme points changes only from 0.633 to0.598 s,
and the inference memory from 12.668 to12.760 GB. This means that in the current pipeline
35

Preprint
the main inference cost is determined not so much by the adapter size as by the base model and the
retrieval pipeline, and the additional adaptation capacity coming with rank goes mainly into quality
improvements.
The same logic is partially reproduced in the larger family as well: inside qv_only the move from
8B r4 to8B r64 raises F1from0.610 to0.617 , while latency stays in the approximate range
of0.66-0.69 s and inference memory changes very little inside the 8B regime. However, between
8B r16 qv_only and8B r64 qv_only the paired bootstrap gives a 95% CI of [-0.011,
+0.016] , so inside the upper part of the 8B family rank growth is more accurately interpreted as a
weak positive trend rather than a strictly established advantage of the higher rank. In other words,
a high rank brings the greatest benefit precisely where a structurally successful LoRA adaptation
scheme has already been chosen, but inside the 8B family the choice between the top ranks requires
accounting for additional criteria.
H.4 WHY RETRIEVAL AND PROMPTING SHIFT THE OPTIMUM POINT BUT DO NOT CANCEL THE
CONCLUSIONS ABOUTLORA
The ablation experiments show that the retrieval and prompting components affect not only the
absolute values of the metrics, but also the position of the optimal working point inside the same set
of adapters. Switching off the reranker, moving to dense-only or sparse-only retrieval, and explicit
changes to the prompting policy can shift the local optimum in F1or in groundedness, since they
change both the quality of the supplied context and the freedom of the generator in producing the
answer.
However, these changes do not destroy the main conclusion about LoRA configurations; they only
modify the conditions in which it shows itself. In other words, retrieval and prompting determine
how favourable the environment for the generator turns out to be, but inside this environment the
advantage more often stays with qv_only adapters. The ablation results should therefore be read
not as a refutation of the main LoRA conclusions, but as a confirmation that the adapter, retrieval, and
prompting form a single pipeline, inside which a structurally successful adaptation scheme remains a
stable source of advantage.
H.5 GENERALIZATION ANALYSIS BYT O P_K
As a separate generalization analysis we checked the robustness of the main conclusions with respect
to the number of chunks passed to the generator at inference time. In addition to the main regime
top_k = 2 , additional experiments were performed for top_k = 1 andtop_k = 4 in the
01_base__neutral regime for all 22 configurations without any further adapter training. This
cross-section makes it possible to check that the obtained conclusions do not reduce to one particular
value of the retrieval-context budget.
36

Preprint
Figure 14: Change of the best quality and latency for different values oftop_k
The plot for top_k = 1, 2, 4 shows that increasing the retrieval-context budget consistently
raises the maximum of F1(0.600 -> 0.617 -> 0.632 ), but at the same time raises latency
(0.604 -> 0.655 -> 0.719 s). This means that increasing top_k does not reduce to a free
improvement in quality: a wider context does help the best configuration, but it requires a more
expensive inference-time regime.
A comparison of the concrete best points is given in Table 18.
top_kbest F1 config F1 [95% CI] lat. (s) inf. VRAM (GB) runtime Pareto front
1 8B r64 qv_only0.600 [0.577, 0.623] 0.604 21.8208B r64 qv_only
2 8B r64 qv_only0.617 [0.593, 0.639] 0.655 21.9263B r64 qv_only,8B r64 qv_only
4 8B r64 qv_only0.632 [0.612, 0.654] 0.719 22.0783B r64 qv_only,8B r64 qv_only
Table 18: Comparison of the best configurations for different values oftop_k
Table 18 shows that for all three values of top_k the best configuration in F1remains the same:
8B r64 qv_only . At the same time, the structure of the runtime trade-off changes only mildly:
fortop_k = 1 the quality-latency front collapses to the single point 8B r64 qv_only , while
fortop_k = 2 andtop_k = 4 it simultaneously contains the lightweight point 3B r64 qv_-
only and the highest-quality point 8B r64 qv_only . The generalization analysis therefore
shows that varying top_k affects primarily the absolute values of quality and cost, but does not
cancel the main conclusion about the advantage of LoRA adapters that adapt the q- andv-projections
of attention.
H.6 PRACTICAL RECOMMENDATIONS
From the obtained results, several explicit recommendations on the choice of configuration can be
drawn.
•If the priority is a lightweight inference regime with high quality preserved, the preferred
configuration is3B r64 qv_only.
•If minimizing training cost while keeping acceptable quality is important, a reasonable
starting point is 3B r4 qv_only , since it is the cheapest non-dominated configuration in
training time and memory.
37

Preprint
•If one needs the best quality-cost compromise inside the light family, the most practical
intermediate configuration is 3B r32 qv_only : it already lies on the training Pareto
front and is noticeably close to the upper points in quality without moving to the 8B regime.
•If it is important to obtain quality statistically comparable to 8B baseline but with a
lighter resource profile, a reasonable working point is3B r64 qv_only.
•If inside the 8B family it is more important to keep higher groundedness at statistically
comparable quality, 8B r16 qv_only is preferable, since exactly this configuration
yields the maximumgroundedness_pass@4in the base regime.
•If the priority is the maximum point-quality estimate and a substantially larger resource
budget is available, 8B r64 qv_only is preferable; however its advantage in F1over
8B r16 qv_onlyon the current test set is not statistically confirmed.
•full_attention configurations are reasonable to consider as a niche option: in the
current set of runs they rarely give a stable advantage inF1and more often lose in cost.
H.7 ERROR ANALYSIS
Alongside the aggregate metrics it is useful to consider the typical error classes that recur across
different RAG-system configurations. In the present work it is convenient to group these errors not
by individual failing examples, but by the failure mechanism inside the retrieval-augmented pipeline.
This allows us to separate errors caused by low-quality context from errors that arise already at the
answer-generation stage given relevant context.
To make this section quantitative, we built a reproducible sample of 100 failing answers on the final
test split: 50 randomly chosen cases with F1 < 1 for3B r64 qv_only and8B r64 qv_-
only each ( seed = 42 ). Each example was then assigned to one of four operational error classes:
retrieval miss (the relevant source chunk did not enter the final context), overclaiming
(the answer adds extra unsupported statements), incomplete answer (the answer extracts
only part of the required information), and exact/precision failure (an error of exact
reproduction of a command, field, version, name, or other literal value). The summary is given in
Table 19.
Table 19: Quantitative cross-section of error types on a sample of 100 failing answers
Error type 3B r64 qv_only 8B r64 qv_only Total
retrieval miss11 (22%) 8 (16%) 19 (19%)
overclaiming2 (4%) 2 (4%) 4 (4%)
incomplete answer17 (34%) 7 (14%) 24 (24%)
exact/precision failure20 (40%) 33 (66%) 53 (53%)
Table 19 shows that in this sample exact-reproduction errors dominate (53%), and this is especially
pronounced for 8B r64 qv_only (66% against 40% for 3B r64 qv_only ). This indicates
that a noticeable share of the remaining errors is connected not with the full absence of relevant
context, but with inaccurate reproduction of short commands, field names, versions, and other local
facts. The second place is occupied by incomplete answers (24%), followed by retrieval miss
(19%), whileoverclaimingis noticeably rarer in this sample (4%).
Below are some illustrative examples from the same sample.
•retrieval miss : for the question What does kube-proxy provide for
Services? the reference answer is formulated as provides load balancing ,
while the model answers A virtual IP mechanism. ; for the question What pro-
tocol does kube-proxy not understand? the reference is does not un-
derstand HTTP , while the model returns TCP, UDP, AND SCTP STREAM FOR-
WARDING..
•overclaiming : for the question What does the term storage version
describe in Kubernetes? the reference is limited to the formula how an ob-
ject is stored in your cluster , but the model adds an extra part about the ob-
38

Preprint
ject’s representation in the API; for the question Why might a Windows Pod show
ErrImgPull or ImagePullBackOff? the reference ties the error to running on an
incompatible Windows node, while the model substitutes more general reasons such as a
wrong image name or a missingimagePullSecret.
•incomplete answer : for the question Which method is easier to use
for adding custom resources: CRDs or aggregated APIs? the
reference answer is two-part: CRDs are easier to use. Aggregated APIs
are more flexible. , while the model keeps only the first part; for the question
Does Kubernetes have a User object for storing usernames in
its API? the reference simultaneously states the absence of a User object and the
absence of storing usernames in the API, while the model answers onlyNo..
•exact/precision failure : for the question What is the default API
server bind port for kubeadm init phase kubeconfig all? the ref-
erence value is 6443 , while the model returns 6444 ; for the question Which flag
forces Windows line endings in kubectl edit? the reference form is
-windows-line-endings , while the model returns the string windows-line-
enddings.
First, a noticeable but non-dominant share is made up of retrieval miss -type errors, in which
the relevant documentation fragment does not enter the final context. In this case the generator is
forced to answer from incomplete or indirectly related material, and therefore even strong adapters
cannot stably recover the correct answer. Practically such cases show up either as a factual error or as
a partially correct but overly general answer.
Second, we observe unsupported-addition (overclaiming) errors. In the current sample there are
few of them, but they remain methodologically important: the answer looks plausible and often
gets an acceptable F1, but it adds statements that do not directly follow from the retrieved context.
This error type is especially sensitive to the judge-based groundedness evaluation and explains why
configurations optimal in F1do not coincide with configurations optimal in groundedness_-
pass@4.
Third, we see incomplete answers in the presence of relevant context. On the quantitative sample
this is the second most frequent error type. Here retrieval has already worked well enough, but the
generator extracts from the found material only part of the required information: it omits a caveat, a
constraint, a condition, or one element of a multi-part answer. Such errors are especially important
for technical documentation, where even a brief omission may change the practical meaning of the
answer.
Fourth, for the class of exact questions, format and precision-of-reproduction errors are typical, and
exactly this type turns out to be the largest in the quantitative sample. Even with the correct context,
the model can give an answer that is close in meaning but not literally correct: it rephrases the field
name, inaccurately reproduces a flag, a path, an API version, or a specific value. Such failures show
that some of the errors in documentation-oriented question answering are related not to the absence
of knowledge as such, but to the difficulty of accurately reproducing local facts.
Finally, a separate class is made up of prompt-sensitive failures. The ablation experiments show
that changing the retrieval/prompting policy can shift the optimal working point even for the same
LoRA configurations. This effect was not encoded as a separate row in Table 19, since it shows
up primarily at the level of between-regime comparison rather than inside a single fixed sample of
answers. It follows that a proper error analysis for a RAG system should treat the generator, retrieval,
and prompting as interrelated components of one pipeline, not as independent sources of quality.
I PER-REGIMEDETAILEDTABLES
This appendix contains the detailed metric tables for each of the 10 ablation regimes referenced
in Section 7. For every regime we report, for each of the 12 LoRA configurations and the two
non-adapted baselines (3B/8B), the point estimates of F1, judge groundedness_pass@4 , judge
correctness_pass@4 together with their 95% bootstrap confidence intervals ( 1,000 resamples
on the test split, n= 785 ), as well as the corresponding latency and inference VRAM. The same
numbers underpin all per-regime plots in Appendix J and the aggregate ablation summary in Section 7.
39

Preprint
01_B A S E__N E U T R A L
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.542 [0.517, 0.565] 0.716 [0.684, 0.745] 0.717 [0.684, 0.748] 0.628 12.664
3B r4 full_attention0.577 [0.554, 0.600] 0.795 [0.767, 0.823] 0.808 [0.781, 0.834] 0.975 12.674
3B r4 qv_only0.572 [0.550, 0.592] 0.800 [0.772, 0.827] 0.806 [0.777, 0.833] 0.633 12.668
3B r8 full_attention0.578 [0.558, 0.599] 0.789 [0.761, 0.819] 0.808 [0.782, 0.834] 0.734 12.701
3B r8 qv_only0.575 [0.553, 0.597] 0.803 [0.772, 0.831] 0.813 [0.787, 0.842] 0.731 12.676
3B r16 full_attention0.590 [0.568, 0.614] 0.801 [0.774, 0.829] 0.810 [0.782, 0.838] 0.753 12.719
3B r16 qv_only0.583 [0.562, 0.605] 0.783 [0.755, 0.811] 0.790 [0.758, 0.818] 0.649 12.701
3B r32 full_attention0.577 [0.554, 0.600] 0.789 [0.759, 0.817] 0.803 [0.775, 0.831] 0.776 12.732
3B r32 qv_only0.592 [0.569, 0.614] 0.796 [0.764, 0.824] 0.808 [0.781, 0.834] 0.621 12.719
3B r64 full_attention0.578 [0.554, 0.601] 0.799 [0.772, 0.827] 0.818 [0.792, 0.845] 0.783 12.924
3B r64 qv_only0.597 [0.574, 0.622] 0.782 [0.752, 0.811] 0.791 [0.762, 0.817] 0.598 12.760
8B baseline0.595 [0.572, 0.620] 0.772 [0.739, 0.800] 0.775 [0.746, 0.804] 0.755 21.836
8B r4 full_attention0.603 [0.583, 0.625] 0.819 [0.794, 0.846] 0.828 [0.801, 0.855] 0.792 21.871
8B r4 qv_only0.610 [0.589, 0.631] 0.822 [0.795, 0.847] 0.832 [0.806, 0.859] 0.686 21.828
8B r8 full_attention0.599 [0.578, 0.620] 0.809 [0.782, 0.837] 0.815 [0.789, 0.841] 0.772 21.859
8B r8 qv_only0.606 [0.584, 0.628] 0.822 [0.795, 0.847] 0.833 [0.806, 0.860] 0.706 21.836
8B r16 full_attention0.604 [0.583, 0.625] 0.805 [0.777, 0.832] 0.810 [0.783, 0.836] 0.807 21.885
8B r16 qv_only0.615 [0.592, 0.635] 0.825 [0.797, 0.854] 0.836 [0.811, 0.862] 0.690 21.824
8B r32 full_attention0.602 [0.581, 0.624] 0.800 [0.772, 0.827] 0.808 [0.780, 0.838] 0.844 21.961
8B r32 qv_only0.612 [0.590, 0.634] 0.811 [0.783, 0.837] 0.814 [0.786, 0.841] 0.665 21.850
8B r64 full_attention0.606 [0.583, 0.628] 0.814 [0.789, 0.839] 0.815 [0.789, 0.842] 0.899 22.164
8B r64 qv_only0.617 [0.593, 0.639] 0.818 [0.791, 0.842] 0.827 [0.797, 0.855] 0.655 21.926
Table 20: Per-configuration metrics for the 01_base__neutral regime: F1, groundedness, and
correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test split), inference latency,
and peak inference VRAM.
40

Preprint
02_B A S E__E X P L I C I T_G R O U N D E D
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.536 [0.512, 0.562] 0.729 [0.698, 0.761] 0.724 [0.693, 0.754] 0.610 12.686
3B r4 full_attention0.559 [0.536, 0.583] 0.778 [0.749, 0.806] 0.796 [0.768, 0.824] 0.736 12.695
3B r4 qv_only0.566 [0.545, 0.587] 0.806 [0.777, 0.836] 0.813 [0.783, 0.839] 0.653 12.689
3B r8 full_attention0.569 [0.546, 0.591] 0.787 [0.760, 0.817] 0.803 [0.776, 0.829] 0.768 12.723
3B r8 qv_only0.571 [0.549, 0.593] 0.778 [0.750, 0.806] 0.792 [0.762, 0.820] 0.668 12.697
3B r16 full_attention0.577 [0.556, 0.598] 0.789 [0.759, 0.815] 0.805 [0.778, 0.832] 0.777 12.740
3B r16 qv_only0.574 [0.551, 0.597] 0.795 [0.764, 0.822] 0.805 [0.776, 0.832] 0.651 12.723
3B r32 full_attention0.567 [0.544, 0.591] 0.786 [0.759, 0.814] 0.803 [0.773, 0.831] 0.814 12.754
3B r32 qv_only0.576 [0.551, 0.600] 0.775 [0.744, 0.801] 0.783 [0.755, 0.811] 0.669 12.740
3B r64 full_attention0.573 [0.549, 0.597] 0.787 [0.759, 0.815] 0.808 [0.781, 0.834] 0.897 12.945
3B r64 qv_only0.598 [0.575, 0.619] 0.776 [0.749, 0.805] 0.790 [0.759, 0.818] 0.596 12.781
8B baseline0.603 [0.578, 0.627] 0.780 [0.753, 0.806] 0.778 [0.750, 0.805] 0.752 21.795
8B r4 full_attention0.602 [0.580, 0.623] 0.819 [0.792, 0.846] 0.825 [0.799, 0.852] 0.814 21.863
8B r4 qv_only0.602 [0.580, 0.623] 0.811 [0.783, 0.838] 0.824 [0.799, 0.851] 0.687 21.855
8B r8 full_attention0.601 [0.579, 0.623] 0.813 [0.786, 0.837] 0.819 [0.794, 0.846] 0.789 21.852
8B r8 qv_only0.606 [0.585, 0.627] 0.820 [0.794, 0.846] 0.836 [0.810, 0.860] 0.693 21.863
8B r16 full_attention0.601 [0.580, 0.622] 0.813 [0.783, 0.838] 0.814 [0.786, 0.842] 0.843 21.877
8B r16 qv_only0.610 [0.589, 0.632] 0.823 [0.797, 0.848] 0.834 [0.806, 0.859] 0.662 21.852
8B r32 full_attention0.601 [0.579, 0.622] 0.804 [0.776, 0.831] 0.811 [0.781, 0.838] 0.808 21.953
8B r32 qv_only0.610 [0.588, 0.632] 0.813 [0.786, 0.838] 0.815 [0.789, 0.841] 0.648 21.877
8B r64 full_attention0.603 [0.580, 0.623] 0.808 [0.777, 0.833] 0.815 [0.786, 0.845] 0.806 22.156
8B r64 qv_only0.623 [0.599, 0.645] 0.822 [0.795, 0.848] 0.824 [0.796, 0.850] 0.664 21.953
Table 21: Per-configuration metrics for the 02_base__explicit_grounded regime: F1,
groundedness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test
split), inference latency, and peak inference VRAM.
41

Preprint
03_R E R A N K E R_O F F__N E U T R A L
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.541 [0.516, 0.564] 0.729 [0.699, 0.761] 0.718 [0.685, 0.746] 0.634 12.664
3B r4 full_attention0.573 [0.552, 0.595] 0.780 [0.753, 0.809] 0.794 [0.767, 0.820] 0.732 12.674
3B r4 qv_only0.571 [0.548, 0.593] 0.800 [0.772, 0.829] 0.811 [0.783, 0.838] 0.636 12.668
3B r8 full_attention0.579 [0.557, 0.600] 0.799 [0.772, 0.824] 0.808 [0.780, 0.836] 0.784 12.701
3B r8 qv_only0.574 [0.554, 0.597] 0.799 [0.769, 0.827] 0.808 [0.778, 0.836] 0.647 12.676
3B r16 full_attention0.586 [0.564, 0.608] 0.803 [0.775, 0.831] 0.815 [0.786, 0.838] 0.798 12.719
3B r16 qv_only0.581 [0.558, 0.605] 0.791 [0.762, 0.817] 0.799 [0.772, 0.827] 0.649 12.701
3B r32 full_attention0.579 [0.557, 0.601] 0.794 [0.766, 0.823] 0.814 [0.785, 0.839] 0.781 12.732
3B r32 qv_only0.589 [0.568, 0.611] 0.794 [0.766, 0.823] 0.804 [0.778, 0.829] 0.626 12.719
3B r64 full_attention0.575 [0.549, 0.596] 0.790 [0.762, 0.819] 0.805 [0.776, 0.832] 0.777 12.924
3B r64 qv_only0.597 [0.572, 0.620] 0.786 [0.758, 0.815] 0.794 [0.766, 0.823] 0.613 12.760
8B baseline0.592 [0.569, 0.617] 0.771 [0.741, 0.801] 0.766 [0.735, 0.796] 0.779 21.836
8B r4 full_attention0.603 [0.583, 0.625] 0.823 [0.796, 0.848] 0.831 [0.804, 0.856] 0.788 21.871
8B r4 qv_only0.610 [0.587, 0.631] 0.822 [0.794, 0.848] 0.829 [0.804, 0.855] 0.656 21.828
8B r8 full_attention0.599 [0.577, 0.619] 0.808 [0.780, 0.834] 0.815 [0.785, 0.841] 0.742 21.859
8B r8 qv_only0.606 [0.584, 0.627] 0.828 [0.800, 0.854] 0.832 [0.805, 0.857] 0.656 21.836
8B r16 full_attention0.604 [0.584, 0.624] 0.809 [0.782, 0.836] 0.811 [0.783, 0.838] 0.809 21.885
8B r16 qv_only0.615 [0.592, 0.637] 0.836 [0.809, 0.861] 0.842 [0.815, 0.868] 0.685 21.824
8B r32 full_attention0.602 [0.580, 0.622] 0.797 [0.767, 0.824] 0.806 [0.778, 0.833] 0.817 21.961
8B r32 qv_only0.612 [0.590, 0.636] 0.817 [0.791, 0.843] 0.820 [0.794, 0.848] 0.641 21.850
8B r64 full_attention0.606 [0.585, 0.627] 0.815 [0.787, 0.842] 0.817 [0.786, 0.841] 0.843 22.164
8B r64 qv_only0.617 [0.596, 0.638] 0.819 [0.792, 0.846] 0.824 [0.796, 0.851] 0.658 21.926
Table 22: Per-configuration metrics for the 03_reranker_off__neutral regime: F1, ground-
edness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test split),
inference latency, and peak inference VRAM.
42

Preprint
04_R E R A N K E R_O F F__E X P L I C I T_G R O U N D E D
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.537 [0.514, 0.561] 0.740 [0.710, 0.769] 0.740 [0.707, 0.772] 0.665 12.686
3B r4 full_attention0.559 [0.537, 0.583] 0.780 [0.749, 0.810] 0.786 [0.761, 0.813] 0.783 12.695
3B r4 qv_only0.566 [0.544, 0.587] 0.800 [0.772, 0.827] 0.809 [0.781, 0.836] 0.624 12.689
3B r8 full_attention0.569 [0.545, 0.592] 0.800 [0.775, 0.829] 0.810 [0.783, 0.837] 0.760 12.723
3B r8 qv_only0.571 [0.547, 0.593] 0.789 [0.761, 0.817] 0.803 [0.773, 0.831] 0.670 12.697
3B r16 full_attention0.577 [0.554, 0.598] 0.794 [0.762, 0.822] 0.805 [0.778, 0.832] 0.788 12.740
3B r16 qv_only0.574 [0.551, 0.597] 0.790 [0.763, 0.818] 0.804 [0.776, 0.832] 0.631 12.723
3B r32 full_attention0.567 [0.544, 0.589] 0.794 [0.767, 0.822] 0.804 [0.776, 0.831] 0.782 12.754
3B r32 qv_only0.576 [0.552, 0.597] 0.785 [0.757, 0.814] 0.797 [0.769, 0.824] 0.633 12.740
3B r64 full_attention0.573 [0.550, 0.595] 0.799 [0.772, 0.827] 0.808 [0.780, 0.834] 0.881 12.945
3B r64 qv_only0.598 [0.575, 0.621] 0.777 [0.745, 0.805] 0.785 [0.757, 0.815] 0.602 12.781
8B baseline0.599 [0.575, 0.620] 0.773 [0.744, 0.803] 0.775 [0.743, 0.805] 0.780 21.795
8B r4 full_attention0.602 [0.581, 0.624] 0.817 [0.789, 0.842] 0.827 [0.800, 0.854] 0.785 21.863
8B r4 qv_only0.602 [0.579, 0.624] 0.815 [0.787, 0.845] 0.824 [0.796, 0.851] 0.714 21.855
8B r8 full_attention0.601 [0.579, 0.624] 0.819 [0.791, 0.845] 0.823 [0.794, 0.851] 0.786 21.852
8B r8 qv_only0.606 [0.584, 0.626] 0.819 [0.790, 0.846] 0.827 [0.800, 0.854] 0.683 21.863
8B r16 full_attention0.601 [0.580, 0.624] 0.823 [0.796, 0.851] 0.828 [0.804, 0.855] 0.816 21.877
8B r16 qv_only0.610 [0.589, 0.632] 0.829 [0.801, 0.856] 0.834 [0.806, 0.860] 0.675 21.852
8B r32 full_attention0.601 [0.581, 0.622] 0.804 [0.776, 0.831] 0.810 [0.783, 0.836] 0.832 21.953
8B r32 qv_only0.610 [0.589, 0.631] 0.810 [0.783, 0.837] 0.820 [0.792, 0.847] 0.647 21.877
8B r64 full_attention0.603 [0.580, 0.625] 0.808 [0.777, 0.834] 0.811 [0.783, 0.837] 0.829 22.156
8B r64 qv_only0.623 [0.601, 0.645] 0.817 [0.791, 0.843] 0.824 [0.795, 0.851] 0.649 21.953
Table 23: Per-configuration metrics for the 04_reranker_off__explicit_grounded
regime: F1, groundedness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples
on the final test split), inference latency, and peak inference VRAM.
43

Preprint
05_D E N S E_O N L Y__N E U T R A L
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.536 [0.512, 0.559] 0.724 [0.693, 0.757] 0.724 [0.694, 0.754] 0.598 11.676
3B r4 full_attention0.573 [0.552, 0.594] 0.786 [0.757, 0.814] 0.806 [0.781, 0.833] 0.725 12.674
3B r4 qv_only0.571 [0.546, 0.593] 0.810 [0.783, 0.838] 0.814 [0.787, 0.841] 0.616 12.668
3B r8 full_attention0.579 [0.558, 0.601] 0.797 [0.767, 0.824] 0.811 [0.785, 0.837] 0.762 12.701
3B r8 qv_only0.574 [0.551, 0.597] 0.796 [0.769, 0.823] 0.805 [0.777, 0.833] 0.660 12.676
3B r16 full_attention0.586 [0.563, 0.606] 0.803 [0.775, 0.828] 0.813 [0.786, 0.841] 0.807 12.719
3B r16 qv_only0.581 [0.557, 0.604] 0.787 [0.759, 0.817] 0.792 [0.763, 0.819] 0.633 12.701
3B r32 full_attention0.579 [0.556, 0.602] 0.800 [0.772, 0.827] 0.809 [0.780, 0.833] 0.759 12.732
3B r32 qv_only0.589 [0.566, 0.609] 0.796 [0.767, 0.825] 0.806 [0.777, 0.832] 0.630 12.719
3B r64 full_attention0.575 [0.554, 0.598] 0.782 [0.754, 0.810] 0.800 [0.773, 0.829] 0.782 12.924
3B r64 qv_only0.597 [0.573, 0.621] 0.787 [0.760, 0.814] 0.800 [0.773, 0.827] 0.620 12.760
8B baseline0.581 [0.556, 0.605] 0.764 [0.735, 0.794] 0.768 [0.740, 0.796] 0.810 20.795
8B r4 full_attention0.603 [0.582, 0.623] 0.817 [0.791, 0.843] 0.825 [0.799, 0.850] 0.787 21.871
8B r4 qv_only0.610 [0.588, 0.631] 0.818 [0.791, 0.843] 0.827 [0.796, 0.851] 0.676 21.828
8B r8 full_attention0.599 [0.575, 0.621] 0.817 [0.790, 0.842] 0.822 [0.795, 0.850] 0.769 21.859
8B r8 qv_only0.606 [0.585, 0.628] 0.824 [0.796, 0.851] 0.831 [0.805, 0.855] 0.673 21.836
8B r16 full_attention0.604 [0.582, 0.626] 0.809 [0.782, 0.834] 0.813 [0.785, 0.839] 0.801 21.885
8B r16 qv_only0.615 [0.592, 0.635] 0.817 [0.789, 0.841] 0.831 [0.805, 0.856] 0.655 21.824
8B r32 full_attention0.602 [0.581, 0.624] 0.799 [0.769, 0.825] 0.806 [0.778, 0.832] 0.812 21.961
8B r32 qv_only0.612 [0.591, 0.635] 0.813 [0.785, 0.841] 0.818 [0.789, 0.843] 0.667 21.850
8B r64 full_attention0.606 [0.582, 0.627] 0.815 [0.789, 0.841] 0.817 [0.790, 0.843] 0.835 22.164
8B r64 qv_only0.617 [0.595, 0.640] 0.818 [0.791, 0.845] 0.828 [0.801, 0.854] 0.650 21.926
Table 24: Per-configuration metrics for the 05_dense_only__neutral regime: F1, ground-
edness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test split),
inference latency, and peak inference VRAM.
44

Preprint
06_D E N S E_O N L Y__E X P L I C I T_G R O U N D E D
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.532 [0.508, 0.555] 0.734 [0.702, 0.766] 0.727 [0.697, 0.757] 0.579 11.676
3B r4 full_attention0.559 [0.537, 0.583] 0.783 [0.754, 0.811] 0.799 [0.772, 0.825] 0.778 12.695
3B r4 qv_only0.566 [0.544, 0.587] 0.804 [0.776, 0.831] 0.810 [0.782, 0.837] 0.636 12.689
3B r8 full_attention0.569 [0.546, 0.590] 0.792 [0.764, 0.819] 0.804 [0.776, 0.831] 0.792 12.723
3B r8 qv_only0.571 [0.548, 0.593] 0.783 [0.755, 0.809] 0.799 [0.772, 0.827] 0.660 12.697
3B r16 full_attention0.577 [0.556, 0.598] 0.790 [0.762, 0.818] 0.803 [0.776, 0.829] 0.827 12.740
3B r16 qv_only0.574 [0.551, 0.597] 0.790 [0.761, 0.819] 0.795 [0.768, 0.822] 0.628 12.723
3B r32 full_attention0.567 [0.544, 0.590] 0.790 [0.759, 0.818] 0.803 [0.776, 0.829] 0.789 12.754
3B r32 qv_only0.576 [0.552, 0.598] 0.782 [0.752, 0.811] 0.792 [0.764, 0.820] 0.659 12.740
3B r64 full_attention0.573 [0.550, 0.595] 0.795 [0.769, 0.822] 0.809 [0.781, 0.836] 0.804 12.945
3B r64 qv_only0.598 [0.574, 0.621] 0.783 [0.755, 0.814] 0.792 [0.762, 0.820] 0.668 12.781
8B baseline0.586 [0.560, 0.611] 0.769 [0.740, 0.800] 0.767 [0.739, 0.795] 0.732 20.797
8B r4 full_attention0.602 [0.581, 0.625] 0.823 [0.794, 0.850] 0.828 [0.803, 0.855] 0.810 21.863
8B r4 qv_only0.602 [0.580, 0.623] 0.813 [0.783, 0.839] 0.823 [0.799, 0.847] 0.700 21.855
8B r8 full_attention0.601 [0.581, 0.624] 0.820 [0.792, 0.848] 0.827 [0.800, 0.851] 0.786 21.852
8B r8 qv_only0.606 [0.582, 0.627] 0.823 [0.796, 0.850] 0.832 [0.806, 0.856] 0.688 21.863
8B r16 full_attention0.601 [0.579, 0.625] 0.815 [0.790, 0.841] 0.819 [0.792, 0.847] 0.821 21.877
8B r16 qv_only0.610 [0.588, 0.633] 0.827 [0.799, 0.854] 0.832 [0.805, 0.856] 0.679 21.852
8B r32 full_attention0.601 [0.579, 0.622] 0.801 [0.773, 0.829] 0.808 [0.778, 0.836] 0.871 21.953
8B r32 qv_only0.610 [0.588, 0.632] 0.814 [0.785, 0.841] 0.820 [0.792, 0.846] 0.680 21.877
8B r64 full_attention0.603 [0.580, 0.624] 0.809 [0.781, 0.838] 0.818 [0.790, 0.843] 0.825 22.156
8B r64 qv_only0.623 [0.600, 0.645] 0.824 [0.797, 0.850] 0.827 [0.800, 0.851] 0.661 21.953
Table 25: Per-configuration metrics for the 06_dense_only__explicit_grounded regime:
F1, groundedness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test
split), inference latency, and peak inference VRAM.
45

Preprint
07_S P A R S E_O N L Y__N E U T R A L
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.532 [0.507, 0.555] 0.713 [0.680, 0.744] 0.716 [0.683, 0.746] 0.523 11.643
3B r4 full_attention0.573 [0.552, 0.594] 0.790 [0.762, 0.819] 0.803 [0.776, 0.828] 0.746 12.674
3B r4 qv_only0.571 [0.549, 0.592] 0.804 [0.775, 0.831] 0.809 [0.781, 0.839] 0.613 12.668
3B r8 full_attention0.579 [0.557, 0.601] 0.796 [0.766, 0.824] 0.809 [0.780, 0.836] 0.731 12.701
3B r8 qv_only0.574 [0.552, 0.596] 0.795 [0.766, 0.823] 0.809 [0.781, 0.836] 0.671 12.676
3B r16 full_attention0.586 [0.564, 0.608] 0.809 [0.781, 0.836] 0.815 [0.789, 0.842] 0.756 12.719
3B r16 qv_only0.581 [0.559, 0.604] 0.796 [0.768, 0.822] 0.803 [0.776, 0.828] 0.619 12.701
3B r32 full_attention0.579 [0.556, 0.601] 0.794 [0.767, 0.822] 0.813 [0.785, 0.839] 0.897 12.732
3B r32 qv_only0.589 [0.567, 0.612] 0.791 [0.763, 0.819] 0.800 [0.771, 0.827] 0.641 12.719
3B r64 full_attention0.575 [0.552, 0.600] 0.796 [0.769, 0.824] 0.809 [0.781, 0.834] 0.791 12.924
3B r64 qv_only0.597 [0.573, 0.621] 0.791 [0.764, 0.819] 0.797 [0.769, 0.824] 0.595 12.760
8B baseline0.588 [0.564, 0.613] 0.763 [0.734, 0.794] 0.757 [0.725, 0.786] 0.666 20.396
8B r4 full_attention0.603 [0.580, 0.626] 0.827 [0.800, 0.854] 0.832 [0.805, 0.857] 0.801 21.871
8B r4 qv_only0.610 [0.588, 0.630] 0.820 [0.791, 0.847] 0.825 [0.799, 0.850] 0.694 21.828
8B r8 full_attention0.599 [0.576, 0.620] 0.817 [0.787, 0.843] 0.820 [0.794, 0.846] 0.758 21.859
8B r8 qv_only0.606 [0.584, 0.627] 0.823 [0.795, 0.850] 0.829 [0.804, 0.854] 0.641 21.836
8B r16 full_attention0.604 [0.583, 0.627] 0.808 [0.778, 0.834] 0.818 [0.791, 0.845] 0.794 21.885
8B r16 qv_only0.615 [0.593, 0.636] 0.825 [0.800, 0.851] 0.831 [0.804, 0.856] 0.654 21.824
8B r32 full_attention0.602 [0.580, 0.622] 0.797 [0.768, 0.826] 0.804 [0.773, 0.832] 0.842 21.961
8B r32 qv_only0.612 [0.590, 0.635] 0.811 [0.785, 0.837] 0.824 [0.800, 0.851] 0.637 21.850
8B r64 full_attention0.606 [0.582, 0.627] 0.815 [0.786, 0.842] 0.818 [0.790, 0.845] 0.827 22.164
8B r64 qv_only0.617 [0.595, 0.640] 0.820 [0.792, 0.848] 0.823 [0.799, 0.847] 0.661 21.926
Table 26: Per-configuration metrics for the 07_sparse_only__neutral regime: F1, ground-
edness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test split),
inference latency, and peak inference VRAM.
46

Preprint
08_S P A R S E_O N L Y__E X P L I C I T_G R O U N D E D
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.533 [0.508, 0.559] 0.721 [0.689, 0.752] 0.717 [0.687, 0.750] 0.513 11.643
3B r4 full_attention0.561 [0.539, 0.584] 0.781 [0.750, 0.810] 0.790 [0.762, 0.817] 0.749 12.695
3B r4 qv_only0.561 [0.540, 0.582] 0.800 [0.771, 0.827] 0.808 [0.778, 0.837] 0.623 12.689
3B r8 full_attention0.569 [0.546, 0.591] 0.791 [0.763, 0.818] 0.804 [0.777, 0.829] 0.751 12.723
3B r8 qv_only0.569 [0.549, 0.591] 0.777 [0.746, 0.806] 0.796 [0.768, 0.823] 0.669 12.697
3B r16 full_attention0.575 [0.552, 0.598] 0.787 [0.757, 0.814] 0.797 [0.769, 0.827] 0.779 12.740
3B r16 qv_only0.571 [0.550, 0.593] 0.780 [0.749, 0.806] 0.791 [0.761, 0.818] 0.624 12.723
3B r32 full_attention0.572 [0.550, 0.593] 0.797 [0.771, 0.825] 0.805 [0.778, 0.831] 0.806 12.754
3B r32 qv_only0.582 [0.558, 0.604] 0.782 [0.754, 0.810] 0.791 [0.763, 0.818] 0.733 12.740
3B r64 full_attention0.576 [0.553, 0.599] 0.795 [0.766, 0.822] 0.806 [0.777, 0.833] 0.768 12.945
3B r64 qv_only0.593 [0.568, 0.614] 0.776 [0.744, 0.804] 0.785 [0.757, 0.813] 0.605 12.781
8B baseline0.596 [0.573, 0.618] 0.771 [0.743, 0.800] 0.768 [0.739, 0.799] 0.670 20.396
8B r4 full_attention0.602 [0.578, 0.625] 0.824 [0.797, 0.848] 0.836 [0.808, 0.861] 0.809 21.863
8B r4 qv_only0.602 [0.579, 0.623] 0.819 [0.791, 0.843] 0.831 [0.804, 0.856] 0.706 21.855
8B r8 full_attention0.601 [0.581, 0.623] 0.822 [0.792, 0.847] 0.827 [0.800, 0.854] 0.802 21.852
8B r8 qv_only0.606 [0.585, 0.627] 0.822 [0.794, 0.848] 0.831 [0.801, 0.860] 0.653 21.863
8B r16 full_attention0.601 [0.580, 0.623] 0.829 [0.803, 0.855] 0.834 [0.808, 0.860] 0.766 21.877
8B r16 qv_only0.610 [0.588, 0.633] 0.823 [0.795, 0.850] 0.831 [0.804, 0.856] 0.665 21.852
8B r32 full_attention0.601 [0.578, 0.622] 0.806 [0.777, 0.833] 0.806 [0.780, 0.833] 0.842 21.953
8B r32 qv_only0.610 [0.587, 0.631] 0.813 [0.785, 0.841] 0.818 [0.789, 0.846] 0.654 21.877
8B r64 full_attention0.603 [0.578, 0.624] 0.813 [0.782, 0.838] 0.817 [0.791, 0.845] 0.821 22.156
8B r64 qv_only0.623 [0.600, 0.645] 0.828 [0.800, 0.854] 0.832 [0.805, 0.856] 0.646 21.953
Table 27: Per-configuration metrics for the 08_sparse_only__explicit_grounded regime:
F1, groundedness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test
split), inference latency, and peak inference VRAM.
47

Preprint
09_H Y B R I D_B M25__N E U T R A L
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.543 [0.519, 0.568] 0.729 [0.697, 0.759] 0.721 [0.690, 0.752] 0.617 11.789
3B r4 full_attention0.572 [0.550, 0.595] 0.790 [0.762, 0.815] 0.801 [0.772, 0.829] 0.753 12.674
3B r4 qv_only0.569 [0.547, 0.592] 0.797 [0.768, 0.824] 0.803 [0.776, 0.829] 0.635 12.668
3B r8 full_attention0.579 [0.557, 0.601] 0.801 [0.772, 0.831] 0.814 [0.785, 0.839] 0.752 12.701
3B r8 qv_only0.571 [0.549, 0.591] 0.797 [0.768, 0.824] 0.804 [0.776, 0.831] 0.628 12.676
3B r16 full_attention0.587 [0.564, 0.610] 0.801 [0.773, 0.829] 0.811 [0.783, 0.837] 0.774 12.719
3B r16 qv_only0.577 [0.553, 0.601] 0.785 [0.757, 0.813] 0.797 [0.771, 0.827] 0.624 12.701
3B r32 full_attention0.578 [0.556, 0.600] 0.801 [0.773, 0.828] 0.817 [0.787, 0.845] 0.751 12.732
3B r32 qv_only0.593 [0.570, 0.617] 0.792 [0.763, 0.822] 0.803 [0.775, 0.833] 0.652 12.719
3B r64 full_attention0.576 [0.553, 0.598] 0.797 [0.771, 0.825] 0.810 [0.781, 0.837] 0.767 12.924
3B r64 qv_only0.597 [0.573, 0.621] 0.786 [0.755, 0.815] 0.792 [0.764, 0.820] 0.673 12.760
8B baseline0.596 [0.574, 0.620] 0.778 [0.748, 0.806] 0.778 [0.748, 0.806] 0.769 20.768
8B r4 full_attention0.603 [0.581, 0.623] 0.824 [0.797, 0.850] 0.834 [0.808, 0.859] 0.811 21.871
8B r4 qv_only0.610 [0.588, 0.633] 0.819 [0.791, 0.847] 0.829 [0.804, 0.856] 0.675 21.828
8B r8 full_attention0.599 [0.576, 0.621] 0.811 [0.785, 0.838] 0.817 [0.790, 0.845] 0.766 21.859
8B r8 qv_only0.606 [0.586, 0.627] 0.827 [0.801, 0.852] 0.836 [0.808, 0.861] 0.646 21.836
8B r16 full_attention0.604 [0.581, 0.625] 0.810 [0.782, 0.836] 0.808 [0.780, 0.836] 0.806 21.885
8B r16 qv_only0.615 [0.592, 0.636] 0.818 [0.791, 0.845] 0.824 [0.797, 0.848] 0.660 21.824
8B r32 full_attention0.602 [0.581, 0.623] 0.797 [0.768, 0.824] 0.806 [0.777, 0.833] 0.829 21.961
8B r32 qv_only0.612 [0.591, 0.635] 0.819 [0.792, 0.846] 0.825 [0.797, 0.852] 0.649 21.850
8B r64 full_attention0.606 [0.583, 0.627] 0.814 [0.787, 0.841] 0.822 [0.796, 0.848] 0.866 22.164
8B r64 qv_only0.617 [0.597, 0.639] 0.815 [0.787, 0.842] 0.824 [0.797, 0.851] 0.659 21.926
Table 28: Per-configuration metrics for the 09_hybrid_bm25__neutral regime: F1, ground-
edness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test split),
inference latency, and peak inference VRAM.
48

Preprint
10_H Y B R I D_B M25__E X P L I C I T_G R O U N D E D
config F1 [95% CI] grnd@4 [95% CI] corr@4 [95% CI] lat. (s) inf. VRAM (GB)
3B baseline0.543 [0.520, 0.567] 0.735 [0.704, 0.766] 0.729 [0.696, 0.758] 0.615 11.789
3B r4 full_attention0.561 [0.538, 0.584] 0.778 [0.752, 0.809] 0.796 [0.769, 0.824] 0.775 12.695
3B r4 qv_only0.561 [0.540, 0.585] 0.799 [0.771, 0.825] 0.806 [0.778, 0.833] 0.656 12.689
3B r8 full_attention0.569 [0.547, 0.591] 0.796 [0.768, 0.824] 0.810 [0.782, 0.837] 0.805 12.723
3B r8 qv_only0.569 [0.548, 0.590] 0.778 [0.748, 0.806] 0.791 [0.762, 0.819] 0.667 12.697
3B r16 full_attention0.575 [0.554, 0.598] 0.781 [0.753, 0.809] 0.794 [0.762, 0.822] 0.776 12.740
3B r16 qv_only0.571 [0.548, 0.596] 0.786 [0.754, 0.814] 0.794 [0.764, 0.820] 0.642 12.723
3B r32 full_attention0.572 [0.549, 0.593] 0.800 [0.771, 0.829] 0.811 [0.783, 0.838] 0.841 12.754
3B r32 qv_only0.582 [0.559, 0.604] 0.780 [0.750, 0.808] 0.791 [0.763, 0.818] 0.737 12.740
3B r64 full_attention0.576 [0.551, 0.598] 0.795 [0.767, 0.824] 0.809 [0.782, 0.837] 0.783 12.945
3B r64 qv_only0.593 [0.570, 0.615] 0.780 [0.750, 0.809] 0.792 [0.766, 0.822] 0.593 12.781
8B baseline0.603 [0.580, 0.625] 0.781 [0.755, 0.808] 0.781 [0.753, 0.810] 0.734 20.768
8B r4 full_attention0.602 [0.580, 0.625] 0.820 [0.795, 0.846] 0.829 [0.804, 0.856] 0.804 21.863
8B r4 qv_only0.602 [0.579, 0.623] 0.808 [0.781, 0.834] 0.818 [0.791, 0.843] 0.714 21.855
8B r8 full_attention0.601 [0.580, 0.623] 0.820 [0.792, 0.846] 0.822 [0.792, 0.848] 0.795 21.852
8B r8 qv_only0.606 [0.585, 0.627] 0.832 [0.808, 0.857] 0.842 [0.815, 0.869] 0.675 21.863
8B r16 full_attention0.601 [0.581, 0.624] 0.827 [0.800, 0.850] 0.832 [0.806, 0.857] 0.842 21.877
8B r16 qv_only0.610 [0.590, 0.631] 0.834 [0.808, 0.860] 0.838 [0.811, 0.861] 0.691 21.852
8B r32 full_attention0.601 [0.579, 0.621] 0.804 [0.776, 0.829] 0.808 [0.780, 0.834] 0.850 21.953
8B r32 qv_only0.610 [0.589, 0.633] 0.808 [0.782, 0.836] 0.813 [0.783, 0.838] 0.672 21.877
8B r64 full_attention0.603 [0.580, 0.624] 0.810 [0.785, 0.838] 0.817 [0.790, 0.842] 0.820 22.156
8B r64 qv_only0.623 [0.603, 0.645] 0.823 [0.795, 0.847] 0.829 [0.803, 0.855] 0.664 21.953
Table 29: Per-configuration metrics for the 10_hybrid_bm25__explicit_grounded regime:
F1, groundedness, and correctness (pass@4) with 95% bootstrap CI (1000 resamples on the final test
split), inference latency, and peak inference VRAM.
J PER-REGIMEPLOTS
This appendix contains the detailed plots for each of the 10 ablation regimes referenced in Section 7.
For every regime we show the main two-dimensional projections used in the trade-off analysis. Plots
with an F1axis ( f1_vs_latency ,f1_vs_inference_vram ,f1_vs_groundedness_-
pass4 ) are drawn with vertical error bars corresponding to 95% bootstrap confidence intervals on the
final test split ( n= 785 ,1,000 resamples). Analogously, for the judge metrics groundedness_-
pass@4 andcorrectness_pass@4 (per-example binomial indicators of passing the Likert-
score threshold ≥4) we compute 95% bootstrap CIs; the groundedness_pass4_vs_latency
plot has vertical error bars on the groundedness axis, while f1_vs_groundedness_pass4
has two-sided error bars (on both axes). The point CI values are mirrored in the corresponding tables
of Appendix I. The f1_vs_inference_vram plot is drawn as two adjacent panels (separate
zoom on the 3B and 8B families), because under a single axis the configuration points cluster into two
narrow bands of about 0.3 GB and overlap with each other; separate panels allow the within-cluster
structure to be distinguished. The groundedness_pass4_vs_inference_vram projection
is not included in the appendix: in it the spread along the VRAM axis within each family does not
exceed 0.3–0.4 GB, while the corresponding groundedness spread fully lies inside the 95% CI
band, so the plot carries no information beyondf1_vs_inference_vram.
49

Preprint
01_B A S E__N E U T R A L
Regime:base + neutral.
F1 vs Latency
F1 vs Inference VRAM
50

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
51

Preprint
02_B A S E__E X P L I C I T_G R O U N D E D
Regime:base + explicit_grounded.
F1 vs Latency
F1 vs Inference VRAM
52

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
53

Preprint
03_R E R A N K E R_O F F__N E U T R A L
Regime:reranker_off + neutral.
F1 vs Latency
F1 vs Inference VRAM
54

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
55

Preprint
04_R E R A N K E R_O F F__E X P L I C I T_G R O U N D E D
Regime:reranker_off + explicit_grounded.
F1 vs Latency
F1 vs Inference VRAM
56

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
57

Preprint
05_D E N S E_O N L Y__N E U T R A L
Regime:dense_only + neutral.
F1 vs Latency
F1 vs Inference VRAM
58

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
59

Preprint
06_D E N S E_O N L Y__E X P L I C I T_G R O U N D E D
Regime:dense_only + explicit_grounded.
F1 vs Latency
F1 vs Inference VRAM
60

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
61

Preprint
07_S P A R S E_O N L Y__N E U T R A L
Regime:sparse_only + neutral.
F1 vs Latency
F1 vs Inference VRAM
62

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
63

Preprint
08_S P A R S E_O N L Y__E X P L I C I T_G R O U N D E D
Regime:sparse_only + explicit_grounded.
F1 vs Latency
F1 vs Inference VRAM
64

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
65

Preprint
09_H Y B R I D_B M25__N E U T R A L
Regime:hybrid_bm25 + neutral.
F1 vs Latency
F1 vs Inference VRAM
66

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
67

Preprint
10_H Y B R I D_B M25__E X P L I C I T_G R O U N D E D
Regime:hybrid_bm25 + explicit_grounded.
F1 vs Latency
F1 vs Inference VRAM
68

Preprint
F1 vs Groundedness pass@4
Groundedness pass@4 vs Latency
69