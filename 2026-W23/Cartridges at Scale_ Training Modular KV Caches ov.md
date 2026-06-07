# Cartridges at Scale: Training Modular KV Caches over Large Document Collections

**Authors**: Momchil Hardalov, Gonzalo Iglesias, Adrià de Gispert

**Published**: 2026-06-03 07:42:55

**PDF URL**: [https://arxiv.org/pdf/2606.04557v1](https://arxiv.org/pdf/2606.04557v1)

## Abstract
Large Language Models can reason over long contexts, yet prefilling millions of tokens is wasteful as much of the content remains static across queries. Cartridges address this by distilling document collections into reusable key-value (KV) caches that eliminate prefilling while preserving accuracy. A critical limitation of this approach is that cartridges are monolithic and non-compositional: encoding an entire collection into a single KV block does not scale, and naively mixing cartridges trained in isolation collapses performance to near chance. We introduce Cartridges at Scale (CAS), a training framework for scalable multi-cartridge learning with dynamic distractor mixing and a memory-efficient budget manager that rotates hundreds of per-document cartridges between GPU and persistent storage. Our approach scales to collections exceeding a million tokens, improving over a monolithic cartridge by 10-31 points at comparable token budgets. Oracle cartridge accuracy falls within 2-6 points of full in-context learning even at high compression. When paired with retrieval for cartridge selection, CAS matches or exceeds conventional RAG accuracy while consuming 3-4x fewer prompt tokens.

## Full Text


<!-- PDF content starts -->

Cartridges at Scale: Training Modular
KV Caches over Large Document Collections
Momchil Hardalov Gonzalo Iglesias Adrià de Gispert
Amazon AGI
{momchilh, gjii, agispert}@amazon.com
Abstract
Large Language Models can reason over long
contexts, yet prefilling millions of tokens is
wasteful as much of the content remains static
across queries. Cartridges (Eyuboglu et al.,
2025) address this by distilling document col-
lections into reusable key-value (KV) caches
that eliminate prefilling while preserving ac-
curacy. A critical limitation of this approach
is that cartridges are monolithic and non-
compositional: encoding an entire collection
into a single KV block does not scale, and
naively mixing cartridges trained in isolation
collapses performance to near chance. We in-
troduceCartridges at Scale(CAS), a training
framework for scalable multi-cartridge learning
with dynamic distractor mixing and a memory-
efficient budget manager that rotates hundreds
of per-document cartridges between GPU and
persistent storage. Our approach scales to col-
lections exceeding a million tokens, improving
over a monolithic cartridge by 10–31 points at
comparable token budgets. Oracle cartridge ac-
curacy falls within 2–6 points of full in-context
learning even at high compression. When
paired with retrieval for cartridge selection,
CAS matches or exceeds conventional RAG
accuracy while consuming 3–4 ×fewer prompt
tokens.
1 Introduction
Large language models (LLMs) can reason over
information in their context window (Snell et al.,
2025; OpenAI, 2024, 2025; DeepSeek-AI, 2025;
Comanici et al., 2025; Anthropic, 2025; Agarwal
et al., 2025), however this capability comes at a cost
that scales with context length—both in latency
(prefill time) and memory (KV cache size). For ap-
plications requiring persistent access to document
collections—such as enterprise knowledge bases,
patient records, and legal corpora—repeatedly pro-
cessing the same documents for each query is pro-
hibitively expensive.
0 2000 4000 6000 8000 10000 12000
Training Step2030405060708090Accuracy (%)
No ContextOracle (Text)Joint training enables
selective attention
across all cartridges
Isolated carts
catastrophically
interfere
Trained in Isolation, Prompt: Oracle Cartridge
Trained in Isolation, Prompt: 20 Cartridges
Joint Training, Prompt: 20 CartridgesFigure 1: Performance of per-document trained Car-
tridges on LongHealth (Adams et al. (2024), docs: 20
patient notes, ∼232K tokens).Oracle– only the patient
Cartridge is presented in the prompt;Full– all cartridges
are presented. Details in § 4,Training Regime.
Retrieval-augmented generation (RAG, Lewis
et al. (2020); Guu et al. (2020)) mitigates this by re-
trieving only the most relevant text chunks at query
time, but each retrieved chunk is afragmentof
the original document, forcing the model to reason
over incomplete information.
Prior work has proposed reducing KV-cache size
via token eviction or merging (Zhang et al., 2023;
Kang et al., 2024; Liu et al., 2026), or compressing
text into soft (Mu et al., 2023; Cheng et al., 2024;
Louis et al., 2025) or discrete (Jiang et al., 2023;
Pan et al., 2024) token representations. These ap-
proaches often require expensive runtime prefills,
rely on a separate encoder, or lose information from
long documents (Łajewska et al., 2025).
More recently, researchers have explored encod-
ing documents into compact, reusable representa-
tions that can be loaded into the model KV cache
at inference time without additional computational
cost (Devoto et al., 2025; Kim et al., 2026; Zweiger
et al., 2026).Cartridges(Eyuboglu et al., 2025)
proposed training a small set of key-value vectors
per document collection via context distillation,
achieving 10–100× compression while preserving
1arXiv:2606.04557v1  [cs.CL]  3 Jun 2026

the model’s ability to answer questions about the
encoded content. However, their approach relies on
a single monolithic cartridge per collection, which
does not scale to real-world document collections:
loading all documents into one cache bloats the pre-
fix with irrelevant tokens, while information-dense
content (e.g., financial tables, clinical notes, tech-
nical specifications) cannot be highly compressed
without loss. Moreover, a single monolithic car-
tridge is inflexible to future changes: updating or
adding a single document requires re-encoding the
entire KV cache. The natural alternative—one
cartridge per document—exposes a more funda-
mental flaw: as Figure 1 shows,mixing cartridges
trained in isolation leads to model collapse—each
cartridge performs well in the oracle setting (one
correct cartridge loaded), but loading all cartridges
simultaneously drops performance to near-chance,
as the model never learned to selectively attend
across independently trained KV prefixes.
In this work, we introduce CAS, a framework for
training and deploying cartridges at scale over col-
lections of hundreds of documents and millions of
tokens. CAS proposes a training regime based on
dynamic cartridge rotation and mixing that closes
the performance gap of isolated training (Fig. 1)
while remaining feasible within current GPU mem-
ory constraints. We further show that CAS inte-
grates naturally with dense retrieval for cartridge
selection, enabling a practical deployment path that
matches or exceeds text RAG efficiency. Our con-
tributions are as follows:
•We show that independently trained cartridges
suffer catastrophic interference at inference time,
collapsing to near-chance performance, and that
joint training with distractor cartridges removes
this degradation.
•We propose CAS, a scalable training framework
with GPU ↔persistent storage swapping, op-
timizer offloading, and prioritized rotation, en-
abling joint training of hundreds of cartridges
within a fixed GPU memory budget.
•We propose improved self-study data generation
via proportional-to-length sampling and batched
multi-question generation, reducing synthesis
cost by up to 20× while improving fact coverage.
•We demonstrate that per-document cartridges
consistently outperform a single monolithic car-
tridge by up to 30points across five diverse
benchmarks, with compression tolerance rang-
ing from near-lossless at 100× for short techni-
cal documents to up to 5% relative performancedrop for table-heavy financial filings.
•We show that Cartridge RAG matches or exceeds
Text RAG accuracy at up to 4×fewer prompt
tokens, combining the efficiency of compressed
KV caches with the selectivity of dense retrieval.
2 Cartridges at Scale (CAS) Training
Background.Our work builds on the Cartridge
framework of (Eyuboglu et al., 2025). A Cartridge
Z∈RL×p×d×2is a compact, trainable KV cache
that encodes a context of nCtokens as a fixed-size
prefix of length p≪n Cthat is injected into the
model’s attention layers L(of size d), yielding
compression ratios of 10–100× while integrating
directly into existing inference servers.
The Cartridge is trained via a context-distillation
objective using training data generated through
Self-Study: the corpus is chunked into subcor-
pora˜c, and the LLM generates synthetic question–
answer pairs about each chunk using diverse seed
prompts (Eyuboglu et al., 2025). The loss min-
imizes the KL divergence between ateacher(a
model with the subcorpus in context) and astudent
(the same model augmented with the Cartridge).
All model parameters remain frozen—only the key-
value vectors inZreceive gradients.
2.1 Training with Multiple Cartridges
Although (Eyuboglu et al., 2025) discuss that inde-
pendently trained Cartridges can be combined by
concatenating their KV caches, this composition
was never explicitly tested. Figure 1 shows that
mixing caches trainedin isolation—independently
and without awareness of other Cartridges that may
coexist at inference time—leads to model collapse.
Additionally, naive scaling to very large Cartridge
sizes is not feasible since training over large docu-
ment collections has high memory requirements.1
To address these limitations, we introduce CAS,
a training and inference framework that supports
the use ofNcartridges simultaneously (Figure 2).
Mixed-visibility training.Mixed-visibility train-
ing is key for learning KV representations that al-
low the frozen model to selectively attend to the
relevant cartridge in the presence of distractors, di-
rectly optimizing for the multi-cartridge inference
setting. To achieve this, we adapt the KL loss as
follows: let Z∗∈ {Z 1, . . . , Z N} be the relevant
1A 1K-token Qwen3-8B Cartridge takes ∼551 MiB dur-
ing backprop (500K tokens ∼274 GiB). Gradients and Adam
moments increase the Cartridges’ memory footprint 4×.
2

(a)Self-Study.For each document
di, we sample a list of questions us-
ingMQ. Subsequently, the answers
along with the logits from the target
teacherM Aare collected.
(b)Training.Each document diis as-
signed a dedicated cartridge. Abud-
get managerkeeps only Bcartridges
on GPU and rotates the pool every R
steps.
(c)Inference.A selection function ac-
tivates kcartridges from the store. The
selected cartridges are concatenated
and prepended to the query tokens be-
fore decoding.
Figure 2: Cartridges at Scale (CAS) End-to-End Pipeline.
cartridge for example x,Zpool the cartridge pool,
Zdist=Zpool\Z∗the distractor candidates, and
k∼ U(k min, kmax), then the loss becomes:
˜Z=(
Z∗w.p.P iso
Z∗∪S, S∼ U(Z dist, k)w.p.1−P iso(1)
L(Z) =X
(x,˜c)∈D|x|X
i=1DKL 
F(· | ˜c⊕x <i)∥F ˜Z(· |x <i)(2)
Training with packing requires keeping all active
cartridges in the prompt, while each packed sample
sees a different subset via an attention mask.
Budget ManagerDuring training, we introduce
abudget managerthat maintains a fixed GPU pool
ofB≤N cartridges, keeping the remaining N−
Bcartridges offloaded to CPU memory or disk
(Figure 2b). At each step, the budget manager
selects the active cartridges (i.e., those residing on
GPU) and samples data from their training pool.
Pool rotation.Every Roptimizer steps, we ro-
tate the active cartridges in the GPU pool: a fraction
ϕof the pool is evicted and replaced with new car-
tridges from persistent storage. Evicted cartridges
receive no gradients until they are moved back to
GPU. Setting R= 1 andϕ= 1 rotates the entire
pool at every step, maximizing diversity but incur-
ring the highest data-transfer overhead. Larger R
or smaller ϕamortize this overhead by increasing
the duration for which cartridges remain active, at
the cost of slower turnover and reduced exposure
to diverse cartridges during training. The rotation
policy preferentially swaps in cartridges that have
received the fewest optimizer steps, ensuring uni-
form document coverage.To ensure the consistency of the RoPE embed-
dings (Su et al., 2024), query token position IDs
must be offset by the total number of active car-
tridge KV tokens, which occupy the prefix of the
sequence. This offset is recomputed at each step
based on the actual number of tokens contributed
by the active cartridges.
2.2 Optimization Dynamics
Optimizer.We use Adam (Kingma and Ba,
2014) with fp32 precision, while we keep the
weights in bfloat16, as this is crucial for numerical
stability when scaling up the number of cartridges.
Instead of a fixed learning rate (Eyuboglu et al.,
2025), we adopt a slow linear decay schedule with
per-cartridge linear warmup to maintain a higher
effective LR throughout training (see Appendix B).
Per-cartridge LR warmup.When a cartridge is
first swapped into the GPU pool, it has received
zero optimizer steps and its gradients may be large
relative to its current parameter values. Applying
the global LR immediately can destabilize training.
We therefore apply a per-cartridge linear warmup: a
cartridge that has received soptimizer steps uses an
effective LR of ηeff=ηt·min(1,s
Wc), where Wcis
the per-cartridge warmup length. To avoid creating
one optimizer param group per cartridge (which
would prevent fused Adam kernels), we bucket the
warmup scale into Nb= 4discrete levels, yielding
at mostN bparam groups per step.
2.3 Improved Self-Study Data Synthesis
The quality of synthetic training data is the primary
driver of cartridge performance. A detailed anal-
ysis on FinQA documents (Appendix J) reveals
three key failures of the original procedure: (i)un-
even fact coverage—since questions are generated
3

without global awareness of the document, the syn-
thesizer focuses on prose commentary rather than
details such as figures in tables; (ii)unbalanced
document sampling—facts are sampled uniformly
across chunks, causing facts in longer, denser sec-
tions to be under-represented; and (iii)low through-
put—one question per API call makes synthesis
slow and expensive. To mitigate these limitations,
we propose proportional-to-length sampling and
batched multi-question generation (Figure 2a).
Proportional-to-length sampling.We re-
place uniform random chunk sampling with a
proportional-to-lengthstrategy ensuring bal-
anced document coverage. Each document di
in the collection is assigned a sampling weight
wi=|di|
minj|dj|, so that longer documents—which
are assumed to contain more facts—are sampled
proportionally more often during synthesis.
Multi-question generation.We prompt the
question-generation model to produce nquestions
simultaneously (instead of doing nLLM calls); we
usen=20 and sample multiple times with tem-
perature 0.6 (Figure 2a). The prompt instructs
the model to vary the question style (factual re-
call, comparison, reasoning, detail-oriented) and
to cover different facts, details, or aspects of the
provided context. Responses are returned as an ar-
ray, parsed into individual questions and answered
independently by the student model.
Decoupled question generation.We decou-
ple the question-generation model MQfrom the
answer-generation MA.MQis a larger, more capa-
ble model (Agarwal et al., 2025, GPT-OSS 120B)
that generates high-quality, diverse questions given
the document context, whereas MAremains the
target model whose KV cache we want to learn; it
produces answers conditioned on the document—
along with their token-level log-probabilities—as
distillation targets for cartridge training (details in
Appendix C).
2.4 Cartridge Initialization
KV cache initialization quality matters for final Car-
tridge performance. Eyuboglu et al. (2025) show
that initializing the cartridge using the KV cache
from the first ptokens of an arbitrary text outper-
forms random initialization, as it already projects
the KVs within the model’s latent space.
Cart-specific initialization.We propose a sim-
ple yet effective change to this procedure: whentraining over a collection of Ndocuments, each
cartridge Ziis initialized from its own truncated
document dirather than from an arbitrary text.
This ensures that the initial KV cache already en-
codes document-specific content, providing a warm
start that significantly reduces the initial loss (over
∼50% lower in our experiments), reduces the num-
ber of training steps required to reach convergence
and leads to∼10% lower final loss.
We experimented with pooling-based compres-
sion strategies (mean/max pooling, top-knorm se-
lection, and sliding-window averaging) as alter-
natives for capturing more information from the
original document, but none outperformed the cart-
specific truncation strategy.
2.5 Inference
At inference time, the model has access to a collec-
tion of Ntrained cartridges {Z1, . . . , Z N}, each
encoding a distinct document di. When combining
multiple cartridges we prepend their sequentially
concatenated KV caches to the formatted prompt
and question (Figure 2c), without stripping any
special tokens from the processed inputs.
Because loading all Ncartridges simultaneously
is both inefficient and infeasible for large collec-
tions (prefix length grows as O(N·p) ), we select
thekmost relevant cartridges for a given query q
before the forward pass. We evaluate two selec-
tion strategies: (i)Oracle, where the ground-truth
cartridge is always loaded (an upper bound, not ap-
plicable in practice), and (ii)Retrieved, where the
top-kcartridges are selected by a dense retriever.
3 Experimental Setup
Models.We use Qwen3-8B (Yang et al., 2025)
as the main model in our experiments (see Ap-
pendix A for training and inference details). Our
RAG baseline uses dense retrieval via Amazon
Bedrock Knowledge Bases (details in Appendix F).
Datasets.We use these 5 datasets (Table 1):2
•LongHealth(Adams et al., 2024): A clinical QA
benchmark consisting of detailed fictional patient
records with multiple-choice questions that test
information extraction, negation understanding,
and temporal sorting over long medical notes.
•QASPER(Dasigi et al., 2021): An info-seeking
QA dataset over full NLP research papers, where
questions are written by readers who only saw
2We filter out unanswerable questions and outlier docu-
ments that are significantly longer than others in the collection.
4

Dataset Docs Questions Qs/Doc Avg. Tok. Total Tok. Task Type Domain
LongHealth 20 400 20.0 11,700 236K Multiple-choice (5-way) Clinical patient records
QASPER 407 1,451 3.5 4,751 665K Extract./free-form/yes-no Full research papers
QuALITY 115 2,086 18.1 5,713 1.9M Multiple-choice (4-way) Fiction & non-fiction narratives
T2-RB/FinQA 380 1,147 3.0 1,026 392K Math. Calculation Corporate earnings reports w/ tables
TechQA 496 610 1.8 1,509 748K Extractive IBM IT support technotes
Table 1: Dataset statistics.Docs: unique documents (one cartridge per document).Qs/Doc: average questions per
document.Avg. Tok.: mean document length in tokens (Qwen3-8B tokenizer).
the title and abstract. Answers include extractive
spans, free-form text, and yes/no responses.
•QuALITY(Pang et al., 2022): A multiple-
choice reading comprehension benchmark over
long-form fiction and non-fiction narratives,
where questions require careful reading and rea-
soning rather than surface-level pattern matching.
•T2-RAGBench/FinQA(Strich et al., 2026; Chen
et al., 2021): A decontextualized variant of
FinQA designed for RAG evaluation, where ques-
tions about corporate earnings reports have been
rewritten to be context-independent. The task re-
quires numerical reasoning by constructing math-
ematical formulas over tables and text extracted
from financial filings.
•TechQA(Castelli et al., 2020): A technical sup-
port QA dataset drawn from IBM’s Technote cor-
pus, where questions require extracting precise
solutions from IT documentation covering enter-
prise software and infrastructure issues.
Retrieval.For our baseline RAG system, we use
a chunk-based indexing of the documents, then we
retrieve the relevant chunks based on the question,
without reformulation (details in Appendix F). In
the cartridge case, we use the RAG results as a
proxy to retrieve relevant cartridges (see Figure 2c).
In particular, we map each retrieved chunk to a
document id, which in turn is associated with a car-
tridge. The final list contains the unique cartridge
ids in the order of their retrieval. With 1,024-token
chunks, LongHealth and QuALITY average ∼2
chunks per unique cartridge, while for TechQA
and FinQA the ratio is closer to 1:1 (see Table 9).
EvaluationEach dataset uses a task-specific sys-
tem prompt during generation to elicit answers in
the expected format. For FinQA, the model is in-
structed to produce structured mathematical for-
mulas (e.g., divide(subtract(x, y), y) ). For
LongHealth and QuALITY , the model selects from
multiple-choice options. For QASPER, the model
provides brief extractive or yes/no answers. For
TechQA, the model needs to produce concise fac-tual answers. The full model prompts that we use
are provided in Appendix H.
4 Experimental Results
Training Regime.We compare cartridge per-
formance for LongHealth in two regimes: (i) all
cartridges are trained in isolation ( Piso=1.0 ), and
(ii) they are mixed in 1 out of every 4 exam-
ples ( Piso=0.75 ). In both regimes, we train with
the same hyper-parameters—compression of 20×
(∼585 tokens/cart.), 80 epochs, and adding k∼
U(1, M)additional cartridges as distractors.
First, we focus on the oracle setting, where at
inference time each question is given only the rele-
vant cartridge in the KV prefix. In Table 3 (Oracle),
the two joint training regimes achieve compara-
ble performance— 78.9 (M= 10 active) and 79.0
(M= 20 )—while isolated training remains at 73.6.
This suggests that single-cartridge quality benefits
from exposure to distractors during training.
The gap widens drastically when we decode with
all patient cartridges (Full): joint training with 10
active cartridges degrades to 56.4, while isolated
training collapses to 26.0. This shows that mixing
cartridges trained in isolation leads to catastrophic
interference, since the model never learned to dis-
tinguish between the distributions encoded in indi-
vidual cartridges. Crucially, joint training with all
20 cartridges active nearly eliminates this degrada-
tion: theFull (20 docs)score reaches 77.8, only
1.2below its Oracle Cartridge result.
Positional Invariance.Figure 3 quantifies how
the performance changes under different permuta-
tions of the cartridges. We see that shuffling the
cartridge order has no effect ( 77.8 both forOrdered
andShuffled), demonstrating position-invariant at-
tention over the KV prefix. When a random sub-
set of kcartridges is loaded, accuracy on ques-
tions whose cartridgeispresent remains high ( 78–
87), while accuracy on questions whose cartridge
isabsentdrops to 27–31. This confirms that the
model attends selectively to the correct cartridge
5

Method Comp. LongHealth (Acc.) QuALITY(Acc.)QASPER(F1)FinQA(EM)TechQA(LLaaJ)
No Context – 37.5±1.143.6±0.419.2±0.62.9±0.021.1±1.1
Oracle Context1×87.4±0.882.5±0.356.7±0.466.8±2.774.7±0.9
CAS Training (Ours)
Single-Cartridge / Doc.2×81.1±1.178.6±0.954.9∗±0.362.7±0.275.8±0.7
5×79.9±0.878.5±0.954.9±0.657.9±0.475.6±0.7
10×80.1±0.877.6±0.554.1±0.350.7±0.875.2±2.0
20×78.9±1.077.6±0.653.9±0.645.0±0.975.0±0.4
50×76.9±2.276.7±0.553.1±0.430.2±1.874.6±1.5
100×77.3±1.476.4±0.853.1±0.323.0±0.374.7±0.9
Table 2: No Context, Oracle Context, and Oracle Cartridge at selected compression ratios on Qwen3-8B. ±denotes
std. dev. over 3 runs.Boldmarks the best cartridge result per dataset.∗Trained with 3 ×due to memory constraints.
Setting Oracle (1 doc) Full (20 docs)
Oracle Context 87.4±0.8–†/ 65.5
Train in Isolation 73.6±0.426.0±3.1
Train Jointly (25% ex., 10 act.)78.9±1.056.4±0.6
Train Jointly (25% ex., 20 act.)79.0±1.477.8±0.6
Table 3: Effects of the training regime on inference.
Oracle– only the gold cartridge;Full– all notes simul-
taneously. ±denotes std. dev. over 3 runs.†All notes
exceeds the context; 65.5 uses 10 patients groups.
Figure 3: Accuracy ofTrain Jointly (25% ex., 20 act.)
on LongHealth varying the number of present cartridges.
rather than relying on a generic shared memory.
Even though we back-propagate through all active
cartridges simultaneously, there is a clear knowl-
edge separation: each cartridge encodes document-
specific information that the model can retrieve
independently of the other cartridges in the prefix.
Scaling to Hundreds of Cartridges.Next, we
scale up training to four additional datasets, each
containing several hundred unique documents (Ta-
ble 1). In this setting, we simulate realistic docu-
ment collections with up to a million tokens.
Table 2 compares the model performance when
consuming raw contexts vs. their compressed vari-
ants in individual cartridges. TheNo Contextrow is
the model performance without context, only using
the question and the system prompt. We can see
that all datasets perform poorly compared to the
Oracle Context, showing that the model is not able
to solve the tasks parametrically. In the second part
of the table, we show the best results at a givenToken Budget LongHealth QuALITY FinQA
Compr. Acc. Compr. Acc. Compr. EM
4,096 57×67.2 160×62.1 96×6.9
20,000 12×70.5 33×64.2 20×14.1
28,672 8×72.6 23×65.2 14×13.0
Multi-cart. 10×80.1 20×77.6 20×45.0
Table 4: Performance of a monolith cartridge trained on
all documents under varying budgets.
target compression ( 2×to100× )3for each dataset.
We see the following patterns across datasets:
(i) the achievable compression depends strongly
on document complexity: factually dense datasets
and longer documents (LongHealth) do not tolerate
high compression rates, while extraction tasks that
do not require reasoning (TechQA) have no loss
in performance even at 100× compression; (ii) the
parametric knowledge of the model is stabilizing
the compression – QuALITY and QASPER contain
books/papers that are included in many pre-training
corpora (Gao et al., 2020; Weber et al., 2024)
and have only a low drop at higher compressions,
(iii) dense structures in the text, such as tables, are
hard to compress, i.e., FinQA drops 4-9 points at
2×-5×compression, similar to LongHealth.
Cartridges Granularity.We ablate whether the
gains come fromcompression capacity(the car-
tridge has ≥p KV slots) or fromper-document
specialization(each document has its own car-
tridge). We disentangle the two by training a sin-
gle monolithic cartridge on the concatenation of
all documents, with cartridge sizes of 4K, 20K,
28.7K tokens—the latter being roughly the full
available KV budget given a 32K context window
(no YaRN Peng et al. (2024)) minus prompt and
generation overhead.
Despite allocating up to 28K tokens, the mono-
3Each document is compressed to its length divided by the
target compression.
6

0 2000 4000 6000 8000 10000 12000405060708090
t=2
t=100
k=1k=3k=5k=10LongHealth
Cart. (Oracle)
Cart. (RAG) t=2
t=5
t=10
t=20
t=50
t=100
T ext (RAG)
Oracle (T ext)
0 2000 4000 6000 80004050607080t=2
t=100
k=1k=3k=5 k=10QuALITY
0 2000 4000 6000 8000 100002030405060t=2
t=100
k=1k=3k=5 k=10FinQA
0 2000 4000 6000 8000 10000404550556065707580t=2t=100
k=1k=3k=5k=10TechQA
Model Performance
Avg. additional prompt tokensFigure 4: Accuracy vs. additional prompt tokens trade-off between text and Cartridge RAG. The red star denotes the
text oracle score;t: compression ratio;k: number of retrieved chunks.
lith baseline (Table 4) achieves only 72.6 on
LongHealth, 65.2 on QuALITY , and 14.1 on
FinQA. In contrast, per-document cartridges (Multi-
cart.) reach 80.1 on LongHealth at 10× compres-
sion (∼1.2 K tok./doc, 23.6K total), 78.6 on QuAL-
ITY at 2×(∼2,9 K tok./doc), and 62.7 on FinQA at
2×(∼524 tok./doc). This large gap confirms that
per-document specialization is critical. The FinQA
results are particularly striking: 380 financial doc-
uments with dense numerical content cannot be
meaningfully compressed into a single shared rep-
resentation, as gradient updates from different doc-
uments destructively interfere. Nonetheless, dedi-
cating a separate cartridge to each doc resolves this
interference entirely in the oracle setting.
Cartridge RAG.TheOracleresults assume the
gold cartridge is always loaded, but in practice we
often do not know which document or context is
relevant. Moreover, performing inference with hun-
dreds or thousands of cartridges in the prompt is
inefficient and even infeasible. Here, we propose
a hybrid approach combining standard dense re-
trieval with cartridge inference: given a query, we
retrieve the top- kdocument chunks, resolve the
corresponding cartridge identifiers, and load only
those cartridges into the KV prefix instead of the
raw chunk text. WhileText RAGappends kchunks
of∼1,024 tokens each (consuming k×1K prompt
tokens; see Appendix F),Cartridge RAGloads the
corresponding document cartridges at t×compres-
sion, e.g., at t=20 , a12K-token document occupies
only∼600 KV tokens, representing theentiredoc-
ument rather than a single chunk fragment.
Figure 4 reports accuracy as a function of the ad-
ditional prompt tokens consumed at inference.4On
LongHealth and QuALITY , where retrieval recall
is high ( ≥95% atk=10 ; Appendix G),Cartridge
RAGclearly dominatesText RAG. On LongHealth,
4We exclude QASPER from the RAG comparison; the data
is not decontextualized (“What datasets do they use?”).Cartridge RAGat t=20 ,k=10 matches the best
Text RAG( 75.4) using only 2,673 tokens— 3.7×
fewer than Text RAG’s 9,860 —and at t=100 ,
k=10 still delivers 76.5 at just 566tokens. On
QuALITY , Cartridge RAG with t=5,k=5 reaches
75.3, surpassing Text RAG ( 73.9) by1.4at roughly
half the token budget ( 2,635 vs.4,754 ). On FinQA,
we observe a clear trend where loading more car-
tridgeshurts: at t=2, accuracy peaks at k=5 (52.8)
and drops to 50.7 atk=10 , consistently across all
compression ratios. We hypothesize that the model
becomes overwhelmed by the dense numerical con-
tent encoded in multiple financial-document car-
tridges, and that increasing the number of active
cartridges during training (currently capped at 10)
may alleviate this. On TechQA,Text RAGslightly
outperformsCartridge RAG( 73.2 vs.71.0), likely
because answers require extracting precise spans
from the gold text—a setting where verbatim to-
ken access in raw chunks is more effective than
compressed KV representations from multiple doc-
uments that bring noise. Still, in the TechQA case
Cartridge RAGremains more effective until retriev-
ing more than 5 chunks.
5 Discussion
Cost Analysis.The main cost of using cartridges
lies in the offline training process which includes
self-study data generation: sampling 100-200 ques-
tions/doc. and doing one inference with the target
model to generate the response5. Depending on
the size, complexity, and information density of
documents, the training epochs can vary but often
around 10 to 20 epochs are enough to reach 95%
of the best cartridge performance (Figure 1).
Inference costs are limited to loading the car-
tridges; there are no additional inference costs since
we do not need any prefills. At inference, since
5We are modeling the KL divergence which allows for
training on partial sequences, unlike in cross-entropy loss.
7

prefill cost scales as the attention cost ( O(n2)), a
10× token reduction yields ∼100× fewer prefill
FLOPs: Cartridge at r=10 uses∼1,200 tokens vs.
∼12,000 for the text Oracle and ∼9,860 for Text
RAG (k=10) on LongHealth.
Storage is another factor to consider. The KV
cache footprint in BF16 per 1K tokens scales
with model size: ∼110 MiB for Qwen3-0.6B,
∼141MiB for 4B/8B, and∼250MiB for 32B.
Document grouping.Our proposed CAS frame-
work naturally extends togrouped cartridgesen-
coding multiple related documents (e.g., quarterly
reports from the same company, chapters of the
same book). Grouping exploits shared vocabulary
and background knowledge, allowing the token
budget to focus on unique information that can
help achieve even higher compression. It also ad-
dresses the retrieval recall ceiling on homogeneous
corpora: by clustering co-relevant documents, the
retriever operates over a smaller, more discrimi-
native set of groups, and a single retrieved group
cartridge provides richer context than any single-
document cartridge. We leave the systematic study
of grouping strategies to future work.
6 Related Work
Soft Prompting.These methods are variants of
prefix-tuning methods (Lester et al., 2021) that
compress a given text into continuous represen-
tations (Li et al., 2025a). Similar to KV cache
compression, soft compression encodes long con-
texts into compact soft representations that can
be prepended to queries, allowing for storing and
caching. AutoCompressors (Chevalier et al., 2023)
recursively generate summary vectors; Gisting (Bu-
latov et al., 2022; Mu et al., 2023) condenses
prompts into virtual gist tokens; and other meth-
ods encode contexts into compact memory slots
via LoRA-adapted or KV-based encoders (Ge et al.,
2024b; Li et al., 2025b). xRAG (Cheng et al., 2024)
and PISCO (Louis et al., 2025) take a modular ap-
proach, projecting retrieval embeddings or distilled
representations into the LLM’s input space while
keeping the decoder frozen. These methods differ
primarily in their training objectives (KL diver-
gence vs. sequence-level distillation) and in which
components remain trainable.
KV Cache Compression.A natural way to re-
duce the context is to apply operations to its KV
cache. This can happen either after or during theprefill phase by evicting non-impactful key-value
pairs (Zhang et al., 2023; Oren et al., 2024; Ge et al.,
2024a; Tang et al., 2024; Chari and Durme, 2025;
Ła´ncucki et al., 2025) or merging them (Wang
et al., 2024; Zhang et al., 2024; Liu et al., 2025;
Wan et al., 2025). SnapKV (Li et al., 2024) and
PyramidKV (Cai et al., 2025) evict low-importance
KV entries based on attention scores observed dur-
ing prefill, with PyramidKV additionally allocat-
ing non-uniform per-layer budgets. KVzip (Kim
et al., 2026) selects tokens to retain via a learned
context-reconstruction objective. Attention Match-
ing (Zweiger et al., 2026) optimizes a compacted
cache to reproduce the original attention distribu-
tion instead of reconstructing individual KV vec-
tors. RetrievalAttention (Liu et al., 2026) takes
an orthogonal approach, offloading the full KV
cache to the CPU and using approximate nearest-
neighbor search to fetch the relevant entries.
We view many of these KV cache techniques as
complementary to cartridges: compacted caches
can serve as a more effective initialization point,
which is then fine-tuned to the collection distribu-
tion; additionally, compaction can be applied to the
trained cartridges because their representation is
query-independent and may admit further compres-
sion for specific queries.
7 Conclusions
We present an end-to-end framework for training
and serving multiple cartridges at a granularity as
small as that of individual documents. Our frame-
work addresses a fundamental flaw in the original
monolithic Cartridges design, which prevents mix-
ing independently trained cartridges, leading to
collapse in performance at near chance levels. Ex-
periments across five long-context datasets demon-
strate that our approach scales to hundreds of car-
tridges over hundreds of thousands to millions of to-
kens, outperforming monolithic designs by 10–30
points at comparable computational budgets. We
further propose improved initialization strategies
and more efficient self-study generation, achieving
superior accuracy and efficiency. Compression tol-
erance varies with document complexity ( 100× for
TechQA, ≤2× for FinQA), reaching within 5% of
uncompressed baselines. Finally, we show that car-
tridges can be combined with retrieval, matching
RAG performance while using 3–4 ×fewer tokens,
though for information-dense tasks such as FinQA,
text RAG maintains an edge.
8

Acknowledgments
We want to thank Rexhina Blloshmi, Felix Hieber,
Bill Byrne, Hagen Fürstenau and Karan Gill for the
time they took to support, discuss and improve this
work.
Limitations
Cartridge KV position.All of our experiments
prepend the trained KV cache to the prompt and
query tokens, which is the natural setting for single-
turn QA: the cartridge occupies the earliest po-
sitions and all subsequent tokens attend to it via
standard causal masking. In multi-turn scenarios,
however, a cartridge may need to be loaded mid-
conversation—for example, when a user references
a new document after several turns of dialogue.
Prepending the cartridge at that point would invali-
date the previously computed KV entries, requiring
a full recomputation of the conversation history.
Cartridge-specific retrieval.Our Cartridge
RAG pipeline reuses a standard text-based dense
retriever to select which cartridges to activate. This
is a pragmatic choice, but it means retrieval qual-
ity is bounded by how well the query matches the
raw document textrather than the compressed KV
representation. A dedicated cartridge retriever—
one that learns to match queries directly against
cartridge embeddings—could improve recall, par-
ticularly on corpora where text-based retrieval is
unreliable. Developing such a retriever is a natural
next step, but requires either a learned projection
from cartridge KV space to a query-compatible
embedding space, or a contrastive training objec-
tive that aligns query representations with cartridge
representations.
Language coverage.All five benchmarks used
in this work are in English, and the base model
(Qwen3-8B) is predominantly trained on English
text. It is unclear whether the self-study synthesis
procedure, which relies on the model’s own gener-
ative capabilities to produce training data, transfers
equally well to lower-resource languages where
the model’s generation quality is weaker. We do
not evaluate cartridge performance on non-English
corpora and cannot make claims about multilingual
generalization.
Model scale.All experiments use a single model
size. Larger models may achieve higher compres-
sion ratios at the same quality level, since theirgreater parametric capacity can compensate for
more aggressive KV compression. Conversely,
smaller models may require lower compression
ratios to retain acceptable performance. Nonethe-
less, previous work (Eyuboglu et al., 2025; Zweiger
et al., 2026) showed that cartridge performance
scales across model sizes and model families.
Potential Risks
Cartridges enable efficient, persistent encoding of
arbitrary document collections into reusable KV
caches. A malicious actor could exploit this to
encode and serve disinformation, harmful instruc-
tions, or private data at scale. Additionally, docu-
ments encoded into cartridges may retain sensitive
information in their KV representations in ways
that are difficult to audit or redact—unlike raw text
retrieval, it is unclear to what extent personal or
confidential content can be extracted from a trained
cartridge, raising data governance concerns in en-
terprise or medical deployments. We encourage
practitioners to apply the same content-filtering,
bias mitigation, and access-control measures to car-
tridge collections as they would to the underlying
documents.
References
Lisa Adams, Felix Busch, Daniel Truhn, and Keno K.
Bressem. 2024. LongHealth: A question answer-
ing benchmark with long clinical documents.arXiv
preprint arXiv:2401.14490.
Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Alt-
man, Andy Applebaum, Edwin Arbus, Rahul K
Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1
others. 2025. gpt-oss-120b & gpt-oss-20b model
card.arXiv preprint arXiv:2508.10925.
Anthropic. 2025. System card: Claude opus 4 & Claude
sonnet 4. Technical report, Anthropic.
Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev.
2022. Recurrent memory transformer. InAdvances
in Neural Information Processing Systems.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu,
Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong,
Yue Dong, Junjie Hu, and Wen Xiao. 2025. Pyra-
midKV: Dynamic KV cache compression based on
pyramidal information funneling. InSecond Confer-
ence on Language Modeling.
Vittorio Castelli, Rishav Chakravarti, Saswati Dana, An-
thony Ferritto, Radu Florian, Martin Franz, Dinesh
Garg, Dinesh Khandelwal, Scott McCarley, Michael
McCawley, Mohamed Nasr, Lin Pan, Cezar Pendus,
John Pitrelli, Saurabh Pujar, Salim Roukos, Andrzej
9

Sakrajda, Avi Sil, Rosario Uceda-Sosa, and 2 others.
2020. The TechQA dataset. InProceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics, pages 1269–1278, Online.
Vivek Chari and Benjamin Van Durme. 2025. Com-
pactor: Calibrated query-agnostic KV cache com-
pression with approximate leverage scores.Preprint,
arXiv:2507.08143.
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena
Shah, Iana Borova, Dylan Langdon, Reema Moussa,
Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang. 2021. FinQA: A dataset of nu-
merical reasoning over financial data. InProceedings
of the 2021 Conference on Empirical Methods in Nat-
ural Language Processing, pages 3697–3711, Online
and Punta Cana, Dominican Republic.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-
Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan
Zhao. 2024. xrag: Extreme context compression
for retrieval-augmented generation with one token.
Advances in Neural Information Processing Systems,
37:109487–109516.
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and
Danqi Chen. 2023. Adapting language models to
compress contexts. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing, pages 3829–3846, Singapore.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and
1 others. 2025. Gemini 2.5: Pushing the frontier with
advanced reasoning, multimodality, long context, and
next generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan,
Noah A. Smith, and Matt Gardner. 2021. A dataset
of information-seeking questions and answers an-
chored in research papers. InProceedings of the
2021 Conference of the North American Chapter of
the Association for Computational Linguistics, pages
4599–4610.
DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Rea-
soning Capability in LLMs via Reinforcement Learn-
ing.arXiv preprint arXiv:2501.12948.
Alessio Devoto, Maximilian Jeblick, and Simon Jégou.
2025. Expected attention: Kv cache compression by
estimating attention from future queries distribution.
arXiv preprint arXiv:2510.00636.
Sabri Eyuboglu, Ryan Saul Ehrlich, Simran Arora, Neel
Guha, Dylan Zinsley, Emily Ruoyu Liu, Atri Rudra,
James Y . Zou, Azalia Mirhoseini, and Christopher Re.
2025. Cartridges: Lightweight and general-purpose
long context representations via self-study. InES-
FoMo III: 3rd Workshop on Efficient Systems for
Foundation Models.Leo Gao, Stella Biderman, Sid Black, Laurence Gold-
ing, Travis Hoppe, Charles Foster, Jason Phang,
Horace He, Anish Thite, Noa Nabeshima, and 1
others. 2020. The pile: An 800gb dataset of di-
verse text for language modeling.arXiv preprint
arXiv:2101.00027.
Leo Gao, Jonathan Tow, Baber Abbasi, Stella Bider-
man, Sid Black, Anthony DiPofi, Charles Foster,
Laurence Golding, Jeffrey Hsu, Alain Le Noac’h,
Haonan Li, Kyle McDonell, Niklas Muennighoff,
Chris Ociepa, Jason Phang, Laria Reynolds, Hailey
Schoelkopf, Aviya Skowron, Lintang Sutawika, and
5 others. 2024. The language model evaluation har-
ness.
Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang,
Jiawei Han, and Jianfeng Gao. 2024a. Model tells
you what to discard: Adaptive KV cache compression
for LLMs. InThe Twelfth International Conference
on Learning Representations.
Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen,
and Furu Wei. 2024b. In-context autoencoder for
context compression in a large language model. In
The Twelfth International Conference on Learning
Representations, ICLR ’24, Vienna, Austria.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2023. LLMLingua: Compressing
prompts for accelerated inference of large language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 13358–13376, Singapore.
Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa
Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao.
2024. Gear: An efficient kv cache compression
recipe for near-lossless generative inference of llm.
arXiv preprint arXiv:2403.05527.
Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W.
Lee, Sangdoo Yun, and Hyun Oh Song. 2026. KVzip:
Query-agnostic KV cache compression with context
reconstruction. InThe Thirty-ninth Annual Confer-
ence on Neural Information Processing Systems.
Diederik P Kingma and Jimmy Ba. 2014. Adam: A
method for stochastic optimization.arXiv preprint
arXiv:1412.6980.
Weronika Łajewska, Momchil Hardalov, Laura Aina,
Neha Anna John, Hang Su, and Lluis Marquez. 2025.
Understanding and improving information preserva-
tion in prompt compression for LLMs. InFindings
of the Association for Computational Linguistics:
EMNLP 2025, pages 17520–17541, Suzhou, China.
10

Adrian Ła ´ncucki, Konrad Staniszewski, Piotr Nawrot,
and Edoardo Ponti. 2025. Inference-time hyper-
scaling with KV cache compression. InThe Thirty-
ninth Annual Conference on Neural Information Pro-
cessing Systems.
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
The power of scale for parameter-efficient prompt
tuning. InProceedings of the 2021 Conference on
Empirical Methods in Natural Language Processing,
pages 3045–3059, Online and Punta Cana, Domini-
can Republic.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai,
Patrick Lewis, and Deming Chen. 2024. SnapKV:
LLM knows what you are looking for before genera-
tion. InAdvances in Neural Information Processing
Systems.
Zongqian Li, Yinhong Liu, Yixuan Su, and Nigel Col-
lier. 2025a. Prompt compression for large language
models: A survey. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers),
pages 7182–7195, Albuquerque, New Mexico.
Zongqian Li, Yixuan Su, and Nigel Collier. 2025b.
500xCompressor: Generalized prompt compression
for large language models. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
25081–25091.
Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhen-
hua Han, Qianxi Zhang, Qi Chen, Chengruidong
Zhang, Bailu Ding, Kai Zhang, and 1 others. 2026.
Retrievalattention: Accelerating long-context llm in-
ference via vector retrieval.Advances in Neural In-
formation Processing Systems, 38:54358–54385.
Xin Liu, Xudong Wang, Pei Liu, and Guoming Tang.
2025. ZSMerge: Zero-shot KV cache compression
for memory-efficient long-context LLMs.Preprint,
arXiv:2503.10714.
Maxime Louis, Hervé Déjean, and Stéphane Clin-
chant. 2025. PISCO: Pretty simple compression
for retrieval-augmented generation. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 15506–15521, Vienna, Austria.
Jesse Mu, Xiang Li, and Noah Goodman. 2023. Learn-
ing to compress prompts with gist tokens. InAd-
vances in Neural Information Processing Systems.OpenAI. 2024. OpenAI o1 System Card.arXiv preprint
arXiv:2412.16720.
OpenAI. 2025. OpenAI o3 and o4-mini Sys-
tem Card. https://cdn.openai.com/pdf/
2221c875-02dc-4789-800b-e7758f3722c1/
o3-and-o4-mini-system-card.pdf.
Matanel Oren, Michael Hassid, Nir Yarden, Yossi Adi,
and Roy Schwartz. 2024. Transformers are multi-
state RNNs. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Process-
ing, pages 18724–18741, Miami, Florida, USA.
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia,
Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle,
Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu,
and Dongmei Zhang. 2024. LLMLingua-2: Data dis-
tillation for efficient and faithful task-agnostic prompt
compression. InFindings of the Association for Com-
putational Linguistics: ACL 2024, pages 963–981,
Bangkok, Thailand.
Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi,
Nikita Nangia, Jason Phang, Angelica Chen, Vishakh
Padmakumar, Johnny Ma, Jana Thompson, He He,
and Samuel R. Bowman. 2022. QuALITY: Question
answering with long input texts, yes!Proceedings
of the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics,
pages 5336–5358.
Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico
Shippole. 2024. YaRN: Efficient context window
extension of large language models. InInternational
Conference on Learning Representations, volume
2024, pages 31932–31951.
Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Avi-
ral Kumar. 2025. Scaling LLM Test-Time Compute
Optimally Can be More Effective than Scaling Param-
eters for Reasoning. InThe Thirteenth International
Conference on Learning Representations, ICLR ’25,
Singapore, Singapore.
Jan Strich, Enes Kutay Isgorur, Maximilian Trescher,
Chris Biemann, and Martin Semmann. 2026. T2-
RAGBench: Text-and-table benchmark for evaluat-
ing retrieval-augmented generation. InProceedings
of the 19th Conference of the European Chapter of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 165–191.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,
Wen Bo, and Yunfeng Liu. 2024. Roformer: En-
hanced transformer with rotary position embedding.
Neurocomputing, 568:127063.
Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao,
Baris Kasikci, and Song Han. 2024. QUEST: query-
aware sparsity for efficient long-context LLM infer-
ence. InProceedings of the 41st International Con-
ference on Machine Learning.
Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan
Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong,
11

Longyue Wang, and Mi Zhang. 2025. D2O: Dynamic
discriminative operations for efficient long-context in-
ference of large language models. InThe Thirteenth
International Conference on Learning Representa-
tions.
Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia
Zhang. 2024. Model tells you where to merge: Adap-
tive KV cache merging for LLMs on long-context
tasks.Preprint, arXiv:2407.08454.
Maurice Weber, Daniel Y . Fu, Quentin Anthony,
Yonatan Oren, Shane Adams, Anton Alexandrov,
Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Vir-
ginia Adams, Ben Athiwaratkun, Rahul Chalamala,
Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang,
Christopher Ré, Irina Rish, and Ce Zhang. 2024. Red-
pajama: an open dataset for training large language
models. InAdvances in Neural Information Pro-
cessing Systems, volume 37, pages 116462–116492.
Curran Associates, Inc.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 Technical Report.arXiv preprint
arXiv:2505.09388.
Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong,
Zhenyu Zhang, Shiwei Liu, and Rongrong Ji. 2024.
CaM: Cache merging for memory-efficient LLMs
inference. InProceedings of the 41st International
Conference on Machine Learning.
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, and 1 oth-
ers. 2023. H2o: Heavy-hitter oracle for efficient
generative inference of large language models.Ad-
vances in Neural Information Processing Systems,
36:34661–34710.
Adam Zweiger, Xinghong Fu, Han Guo, and Yoon Kim.
2026. Fast KV compaction via attention matching.
arXiv preprint arXiv:2602.16284.
12

A Hyperparameters
Table 5 summarizes the training hyper-parameters
used across all datasets. All experiments use
Qwen3-8B6as the base model with BF16 weights
and FP32 Adam optimizer states. Cartridge param-
eters are the only trainable parameters; all model
weights remain frozen. We train using a cluster
with NVidia H200 and B200 cloud GPUs.
We use Qwen3-8B as released on Hugging Face.
The model card provided by the Qwen team doc-
uments training data, intended use, and limita-
tions (Yang et al., 2025).
Dataset-specific variations.The total number
of optimizer steps varies by dataset due to differ-
ent corpus sizes: LongHealth (20 docs) trains for
∼12,480 steps, QuALITY (115 docs) for ∼12,400
steps, QASPER (407 docs) for ∼14,160 steps,
and FinQA (380 docs) for ∼14,160 steps. The
LR schedule max_steps parameter is set accord-
ingly. For the isolation ablation in Table 3, we
usePiso=0.75 for joint training and Piso=1.0 for
isolated training, with all 20 cartridges active si-
multaneously (no rotation needed for 20 cartridges
on a H100 GPU).
Model Inference.For all experiments we evalu-
ate the models with thinking mode enabled, temper-
ature 0.6, and 2048 maximum completion tokens.
We use a separate prompt for each dataset (see Ap-
pendix H), which we pass in the user turn. In the
RAG condition, the retrieved context replaces the
full document in the prompt; in the Oracle text
condition, the full document text is provided. All
reported results are averaged across at least 3 runs.
B Effective LR per Cartridge
Because cartridge rotation means each cartridge
is active for only a fraction of total training
steps, the effective number of gradient updates
per cartridge is much smaller than the global step
count. A standard short-decay schedule causes
the learning rate to reach its minimum long be-
fore most cartridges have received sufficient up-
dates, stalling their convergence. We ablate this
on FinQA by comparing a short-decay schedule
(max_steps=1600 ) against a slow-decay sched-
ule ( max_steps=5000 ,αf=0.02 ) that maintains
a higher effective learning rate throughout train-
ing (Figure 5). The short-decay schedule reaches
6https://huggingface.co/Qwen/Qwen3-8B
Figure 5: LR schedule ablation on FinQA. The
slow-decay schedule with max_steps=5000 and per-
cartridge warm-up is superior tomax_steps=1600.
54.8% ; switching to slow decay improves this
to59.0% (+4.2 points). Adding per-cartridge
warmup and prioritize-least-trained scheduling fur-
ther pushes accuracy to 60.8% , as each cartridge
receives a proper warmup phase when first loaded
into the GPU budget.
C Impact of Self-Study Data
Improvements
We ablate the cumulative effect of our self-study
data improvements on FinQA using 10 single-doc
cartridges of 512 tokens each. We compare the
three data generation strategies:
•Original: The baseline procedure
from Eyuboglu et al. (2025)—one question
per LLM call, using the target model (Qwen3-
8B) for both question and answer generation,
with uniform random chunk sampling.
•Decoupled MQ,n=20 : Multi-question gener-
ation ( n=20 questions per call) with a decou-
pled, more capable question model (GPT-OSS
120B) while keeping the target model as MA.
Same uniform sampling.
•Decoupled MQ,n=20 + prop. to length:
Same as above, but replacing uniform sam-
pling with proportional-to-length document
sampling, ensuring longer documents receive
proportionally more training examples.
The results (Table 6) show that decoupling
the question model and batching 20 questions
per call provides the largest single improve-
ment (+6.8 points over the original), by gen-
erating more diverse and higher-quality ques-
tions that better cover the document’s factual
content. Adding proportional-to-length sampling
contributes a further +2.0 points by ensuring
13

Hyperparameter Value Notes
Optimization
Optimizer Adam (FP32) Moment tensors offloaded to NVMe for CPU-resident carts
Peak learning rate (η 0) 0.05–0.1 Dataset-dependent; 0.1 for LongHealth, 0.05 for others
LR schedule Linear decay with warmup
Warmup steps (W) 200 Global warmup
Final LR multiplier (α f) 0.02η min= 0.02·η 0
Warmup minimum LR 0.002
Per-cartridge warmup (W c) 20 steps Applied when a cartridge first enters the GPU pool
Training epochs 80
Global batch size 128
Packed sequence length 8,192 tokens
Budget Manager & Rotation
Cartridge budget (B) 20 Max cartridges on GPU simultaneously
Rotation interval (R) 10 steps Steps between pool rotations
Swap fraction (ϕ) 0.5 Fraction of pool replaced per rotation
Prioritize least trained Yes Cartridges with fewest steps swapped in first
Empty cache interval 5 rotationstorch.cuda.empty_cache()frequency
Cartridge Masking
Isolation probability (P iso) 0.75 Prob. of seeing only the relevant cartridge
Max active cartridges (k max) 10 Max distractors visible per example
Min active cartridges (k min) 1
Initialization & Padding
Init strategy Cart-specific Each cartridge initialized from its own document
Cache padding Repeat Tiles existing KV vectors if doc<cart size
Num frozen tokens 1 BOS token frozen during training
Infrastructure
Hardware NVIDIA H200 & B200
Precision BF16 (weights), FP32 (optimizer)
Gradient checkpointing Disabled
torch.compileDisabled
Table 5: Training hyperparameters for the CAS framework. Values shown are the defaults used across all five
datasets unless noted otherwise. The peak learning rate is the primary dataset-dependent hyperparameter.
Data Strategy Accuracy (%)
Original 55.9
DecoupledM Q,n=2062.7
DecoupledM Q,n=20+ prop. to length64.7
Oracle (Text) 66.8
Table 6: Impact of self-study data improvements on
FinQA (10 cartridges, 512 tokens/cart).
that longer, information-dense documents receive
training signal commensurate with their expected
greater complexity.
D Scoring Metrics
We use dataset-specific scoring metrics:
•T2-RAGBench/FinQA: Formula execution
accuracy. The predicted formula is parsed and
executed; the numerical result is compared to
the gold answer with a relative tolerance of
1%.
•LongHealth: Fuzzy option matching. The
predicted answer is compared against all fiveoptions using string similarity, and the closest
match is selected.
•QASPER: Average token F1 with multi-
reference support, following the standard
QASPER evaluation protocol. For each ques-
tion, we compute the token-level F1 against
all available reference answers and take the
maximum.
•QuALITY: Exact match on the extracted an-
swer letter (A, B, C, or D).
•TechQA: LLM-as-a-Judge. A separate judge
model (DeepSeek-Distilled-Qwen-32B) eval-
uates whether the generated answer is factu-
ally equivalent to the reference answer. The
judge outputs a JSON object with a justifi-
cation and a binary grade. A score of 1 is
assigned when the judge returns "grade":
"correct" , and 0 otherwise. Table 7 shows
the exact judge prompt.
For all datasets, we report accuracy as the per-
centage of correctly answered questions.
14

Role Prompt
SystemYou are to act as an impartial judge, evaluating whether an answer to a question matches a
provided reference answer. Your task is to determine if the given answer is correct based
on its factual equivalence to the reference answer, ignoring differences in punctuation
and phrasing.
When evaluating the answer, consider the following criteria:
1. Factual equivalence: Does the answer convey the same information as the reference
answer?
2. Completeness: Does the answer address all parts of the question that the reference
answer addresses?
3. Accuracy: Is the information in the answer consistent with the reference answer?
4. Additional information: If the answer contains more information than the reference
answer, does it remain consistent and not contradict itself?
Your response should be structured as follows:
1. A justification for your decision, explaining your reasoning based on the evaluation
criteria.
2. A grade of either "correct" or "incorrect".
Important note on deflections and invalid questions:
- If the answer is a deflection or does not attempt to answer the question, grade it as
"incorrect" unless the reference answer is also a deflection.
- If both the answer and the reference answer indicate that the question is invalid or
cannot be answered, grade it as "correct".
- If the answer is a placeholder like {{YOUR_ANSWER}}, any generic phrase that does NOT
address the question, or an empty answer, then count it as "incorrect".
Your response should be in json format as follows:
{"justification": "...", "grade": "correct" or "incorrect"}
UserHere is the question:
{query}
Here is the answer to be judged:
{answer}
Here is the reference answer:
{expected_answer}
Table 7: LLM-as-a-Judge prompt used for scoring free-form answers on TechQA. The judge model (DeepSeek-
Distilled-Qwen-32B) receives the system prompt defining evaluation criteria and a user message containing the
question, generated answer, and reference answer.
E Cartridge Size Statistics
Because each cartridge is sized proportionally to
its source document, the actual number of KV to-
kens per cartridge varies across documents within
a dataset. Table 8 reports the minimum, mean,
and maximum cartridge sizes (in tokens) for each
dataset at each target compression ratio used in our
experiments.
All cartridge sizes are rounded up to the nearest
multiple of 16 tokens. This constraint arises from
the paged KV-cache implementation in our infer-
ence server (tokasaurus7), which manages memory
in fixed-size pages of 16 tokens. Aligning cartridge
sizes to page boundaries avoids internal fragmen-
tation and ensures that each cartridge occupies an
7https://github.com/ScalingIntelligence/
tokasaurusinteger number of pages, enabling efficient mem-
ory allocation and deallocation during both train-
ing (pool rotation) and inference (cartridge load-
ing/unloading). We additionally enforce a mini-
mum cartridge size of 16 tokens (one page), en-
suring that even the shortest documents receive at
least one full page of KV representation.
At high compression ratios ( 50×–100× ), the
shortest documents in FinQA and TechQA are
clamped to the 16-token minimum. For example,
a 500-token FinQA document at 100× compres-
sion would nominally require only 5 tokens, but is
rounded up to 16. This means theeffectivecom-
pression ratio for short documents is lower than the
target—a 500-token document with a 16-token car-
tridge achieves 31× rather than 100× . Conversely,
the longest documents in each collection achieve
15

Dataset Comp. Min Mean Max Total
LongHealth (20 docs)2×5,040 5,859 6,640 117,184
5×2,016 2,349 2,656 46,976
10×1,008 1,178 1,328 23,568
20×512 593 672 11,856
50×208 242 272 4,848
100×112 126 144 2,512
QASPER (407 docs)2×464 2,349 4,912 955,968
3×304 1,569 3,280 638,400
5×192 944 1,968 384,336
10×96 476 992 193,904
20×64 242 496 98,640
50×64 103 208 42,016
100×64 68 112 27,568
QuALITY (115 docs)2×1,136 2,865 4,208 329,456
5×464 1,151 1,696 132,336
10×240 579 848 66,544
20×128 293 432 33,744
50×64 123 176 14,176
100×64 74 96 8,464
FinQA (380 docs)2×112 524 1,280 199,184
5×64 214 512 81,456
10×64 112 256 42,528
20×64 69 128 26,096
50×16 29 64 11,088
100×16 17 32 6,448
TechQA (471 docs)2×64 460 2,048 216,832
5×64 193 1,008 90,784
10×64 109 512 51,248
20×64 75 256 35,328
50×64 65 112 30,496
100×16 19 64 8,752
Table 8: Cartridge size statistics (in tokens) per dataset and target compression ratio.Min: smallest cartridge in the
collection.Mean: average tokens per cartridge.Max: largest cartridge.Total: sum of all cartridge sizes. All sizes
are multiples of 16 (the page size).
compression ratios close to the target. The wide
min–max range within a dataset (e.g., 64–2,048
for TechQA at 2×) reflects the natural variation
in document lengths: the proportional sizing en-
sures that each document receives a KV budget
commensurate with its information content.
F RAG Baseline: Indexing and Retrieval
Details
We describe the full RAG pipeline below.
Document indexing.Each dataset’s
documents are indexed with chunk size
C∈ {128,256,512,1024} tokens. Docu-
ments are chunked using a fixed-size strategy with
a 10% token overlap between adjacent chunks.
Chunks are embedded using Amazon Titan Embed
Text v2 ( amazon.titan-embed-text-v2:0 )
via Amazon Bedrock8with 1024-dimensional
embeddings and retrieved via cosine similarity.
8https://aws.amazon.com/bedrockRetrieval.At evaluation time, each question is
issued as a retrieval query against the Knowl-
edge Base. We retrieve the top- Kchunks for
K∈ {1,3,5,10} . To avoid redundant API calls,
we retrieve K= 10 once per chunk size and slice
for smaller Kvalues. Retrieved chunks are concate-
nated in score order and prepended to the question
as the context for the reader model.
Hyperparameter selection.We report the best
result across all 16 chunk/top- Kcombinations
(4×4 ) for each dataset as the “RAG best” entry
in Table 2. The optimal configuration varies by
dataset: datasets with short, self-contained answers
(TechQA, FinQA) benefit from larger chunks and
higher K, while datasets requiring precise span ex-
traction (QASPER) are less sensitive to retrieval
configuration.
Effective token counts.Table 10 reports the aver-
age number of context tokens consumed per query
for Text RAG at chunk size 1024 across different k
16

Datasetk=1k=3k=5k=10
LongHealth 1.0 1.8 2.5 4.4
QuALITY 1.0 1.4 2.2 4.8
FinQA 1.0 3.0 5.0 8.7
TechQA 1.0 3.0 4.9 8.1
Table 9: Average number of unique source docu-
ments covered by the top- kretrieved chunks (chunk
size 1024). Datasets with shorter documents (FinQA,
TechQA) have higher chunk diversity, while longer-
document datasets (LongHealth, QuALITY) exhibit
within-document clustering.
Datasetk=1k=3k=5k=10
LongHealth 986 2,960 4,944 9,860
QuALITY 941 2,856 4,733 9,412
FinQA∼960∼2,880∼4,800∼9,600
TechQA∼960∼2,880∼4,800∼9,600
Table 10: Average context tokens per query for Text
RAG (chunk size 1024).
values. Due to the 10% overlap between adjacent
chunks and shorter final chunks in some documents,
the effective token count is slightly below the nom-
inalk×1024.
Unique cartridges per query.Because multi-
ple retrieved chunks may originate from the same
source document, the number of unique documents
loaded in Cartridge RAG is often lower than the
number of retrieved chunks k. Table 9 reports the
average number of unique documents covered by
the top- kchunks (at chunk size 1024). Datasets
with short documents (FinQA, TechQA) exhibit
high chunk diversity—10 chunks span 8–9 dis-
tinct documents—while datasets with longer docu-
ments (LongHealth, QuALITY) show more within-
document clustering, with 10 chunks covering only
4–5 documents on average.
G Retrieval Quality: Recall and MRR
To contextualize the Cartridge RAG results in §4
Cartridge RAG, we report document-level retrieval
quality of the dense retriever used by both the
Text RAG and Cartridge RAG pipelines. For each
query we retrieve 10chunks, deduplicate to unique
source documents, and compute Recall@ Kand
MRR@ Kover the resulting ranked document list
(soKcounts unique documents, not chunks). Ta-
bles 11–14 report these metrics for the four chunk
sizes used in our sweep.
LongHealth and QuALITY have nearly saturated
retrieval ( ≥95% Recall@10), so any gap betweenChunk R@1 R@3 R@5 R@10 MRR@1 MRR@3 MRR@5 MRR@10
128 79.2 88.5 91.2 95.8 79.2 83.6 84.5 85.5
256 76.5 87.5 91.2 95.5 76.5 81.5 82.5 83.3
512 76.0 89.0 93.0 97.0 76.0 81.8 82.8 83.6
1024 76.5 86.5 91.2 96.8 76.5 81.0 82.4 83.4
Table 11: LongHealth document-level retrieval ( 20docs,
400queries).
Chunk R@1 R@3 R@5 R@10 MRR@1 MRR@3 MRR@5 MRR@10
128 85.3 91.1 93.1 95.2 85.3 88.0 88.5 88.9
256 88.0 92.6 94.0 95.4 88.0 90.1 90.5 90.8
512 88.3 92.1 93.2 95.5 88.3 90.1 90.4 90.9
1024 87.1 91.4 93.1 94.8 87.1 89.1 89.6 90.0
Table 12: QuALITY document-level retrieval ( 115docs,
2,086queries).
Text RAG and Oracle (Text) on these datasets is
attributable to information loss in the chunked text
rather than retrieval failures.
FinQA and TechQA both exhibit high retrieval
quality ( ≥95% Recall@10), comparable to
LongHealth and QuALITY . FinQA documents con-
tain distinctive company names and financial ter-
minology that make retrieval effective despite the
large corpus (380 docs). TechQA’s IBM Technotes
cover distinct products and error codes, yielding
strong R@1 ( 80–83%) that improves with larger
chunk sizes as more diagnostic context is captured
per chunk.
H Evaluation Prompts
We use task-specific system prompts during eval-
uation. All prompts instruct the model to wrap its
final answer in <answer> tags. In the Cartridge
setting, the document context is encoded in the
trained KV cache prefix and is not included in the
text prompt—the model receives only the system
prompt and the user question. In the Oracle (Text)
baseline, the full document text is prepended to
the user message. Table 15 lists the exact prompts
used.
For FinQA, we use the evaluation prompt recom-
mended in the T2-RAGBench paper (Strich et al.,
2026).9For QASPER10and QuALITY , we adopt
the prompts from previous work (Gao et al., 2024;
Eyuboglu et al., 2025; Zweiger et al., 2026). For
LongHealth, we use the prompt format from the
original benchmark11. For TechQA, since no stan-
9https://github.com/uhh-hcds/g4kmu-paper/tree/
main/src/g4k/prompts/sys_prompts
10https://github.com/EleutherAI/
lm-evaluation-harness/tree/main/lm_eval/tasks/
qasper
11https://github.com/HazyResearch/cartridges/
blob/main/examples/benchmarks/longhealth/
baseline_longhealth.py
17

Chunk R@1 R@3 R@5 R@10 MRR@1 MRR@3 MRR@5 MRR@10
128 70.4 85.7 90.7 95.9 70.4 77.0 78.0 79.0
256 70.5 87.3 91.8 96.1 70.5 78.0 79.0 79.0
512 69.2 86.0 91.5 95.6 69.2 77.0 78.0 79.0
1024 69.2 87.0 91.8 96.0 69.2 77.0 78.0 79.0
Table 13: FinQA document-level retrieval ( 380docs,
1,147queries).
Chunk R@1 R@3 R@5 R@10 MRR@1 MRR@3 MRR@5 MRR@10
128 79.8 89.0 91.8 93.1 79.8 83.8 84.4 84.6
256 80.3 89.7 92.6 93.9 80.3 84.5 85.2 85.4
512 82.3 90.2 93.1 95.2 82.3 85.8 86.5 86.9
1024 83.1 92.1 93.8 95.2 83.1 87.3 87.7 87.9
Table 14: TechQA document-level retrieval ( 496docs,
610queries).
dard evaluation prompt exists, we conducted a
prompt search over several variants—including
a minimal instruction (“Answer concisely”), a
domain-specific expert framing, a chain-of-thought
variant, and a structured extraction format—and
selected the prompt that yielded the highest Oracle
(Text) accuracy on the development set.
I Self-Study Data Synthesis Details
Question generation model.We use GPT-OSS
120B (Agarwal et al., 2025) as the question genera-
tor (MQ). This model receives the full document
context in its system prompt and generates batches
of 20 diverse questions per call. The answer gen-
erator ( MA) is the target model itself (Qwen3-8B),
ensuring that the distillation signal reflects the stu-
dent’s own output distribution.
Seed prompt types.Each synthesis call ran-
domly selects a seed prompt type from five cat-
egories:structuring(requests to organize informa-
tion into JSON, YAML, or other formats),sum-
marization(requests to summarize specific sec-
tions),question(factual recall and reasoning ques-
tions),use_case(practical downstream applica-
tion tasks), andcreative(open-ended discussion
prompts). This diversity ensures the cartridge is
trained on varied interaction patterns rather than
only factoid QA.
Multi-question prompt format.In batched
mode, the question generator receives the following
instruction (shown for thequestiontype):
Generate {n} diverse questions that test
knowledge of the information in the
corpus above. Each question should cover
a different fact, detail, or aspect
of the corpus. Vary the style: mix
factual recall, comparison, reasoning,
and detail-oriented questions. Include
specific details (ids, names, titles,dates, numerical values, etc.) in each
question so it is clear what you are
asking about.
Output ONLY a JSON array of strings,
e.g. ["question 1", "question 2", ...].
No other text, no markdown fences, no
explanation.
The structured JSON output format enables re-
liable parsing; responses that fail to parse are dis-
carded (typically<5%of calls).
Sampling rounds and temperatures.For each
dataset, we run 4 independent sampling rounds,
one per seed prompt type (question, structuring,
summarization, use_case), each generating 10,000
samples for a total of 40,000 training examples
per dataset.12MQ(GPT-OSS 120B, the question
generator) uses temperature 0.6with top- p=0.95 ,
top-k=20 , and a maximum of 4,096 completion to-
kens per call. MA(Qwen3-8B, the teacher/answer
generator) uses temperature 0.0(greedy decoding)
with a maximum of 2,048 completion tokens, ensur-
ing deterministic distillation targets. Both models
operate with thinking mode enabled.
Sampling configuration.Within each round, we
generate 10,000 synthesis samples with a batch
size of 4 contexts and up to 128 parallel API calls.
Documents are sampled using the proportional-
to-length strategy described in §2.3, with a fixed
random seed for reproducibility. For multi-note
documents (e.g., LongHealth patient records), we
sample one note per prompt to ensure fine-grained
coverage of individual clinical notes.
J Self-Study Data Quality Analysis
To understand the failure modes of the original Self-
Study procedure, we conduct a detailed analysis on
9 FinQA cartridges (512 tokens each, trained for
80 epochs). We measurevalue coverage: for each
unique number in the original document, we count
the fraction of training samples that reproduce it.
Synthesizer bias toward narrative over tables.
The synthesizer preferentially generates questions
about prose commentary rather than specific table
cell values. Table 16 shows that across all 9 car-
tridges, only 41% of unique numbers in the train-
ing data are grounded in the original document.
The remaining 59% are derived calculations (62%),
rounded/reformatted values (22%), or pure fabrica-
tions (15%).
12We exclude thecreativeseed type from training data as it
produces open-ended prompts less suited for distillation.
18

Benchmark System Prompt + User Message Format
T2-RAGBench /
FinQASystem: You are an expert in answering financial questions by constructing
mathematical formulas based on a simple syntax.
- Task: Provide a FORMULA ANSWER to the question based on the given context.
Guidelines:
1. Answer Type: A formula is either a number or one of: add(f1, f2), subtract(f1,
f2), multiply(f1, f2), divide(f1, f2), exp(f1, f2), greater(f1, f2)
2. Reasoning: Carefully analyze the context. Pay special attention to the table.
3. Final Answer: "<answer> FORMULA </answer>"
User: {context}
Question: {question}
LongHealthSystem: Please reference the patient medical records to answer the user’s questions.
Choose the single best option and provide your answer exactly as it appears in the
options.
Wrap your answer in: <answer> The correct option text here </answer>
User: {question}
A. {option_a} B. {option_b} C. {option_c} D. {option_d} E. {option_e}
QASPERSystem: You are a research assistant answering questions about a scientific paper.
Answer as briefly as possible. Give only the answer, no explanation. If the question
cannot be answered from the paper, say "Unanswerable". For yes/no questions, answer
"Yes" or "No". Wrap your answer in: <answer> ... </answer>
User: {question}
QuALITYSystem: You are a careful reader answering multiple-choice questions about a long
article. Read the article and choose the single best answer option. Provide your
answer as the letter (A, B, C, or D) wrapped in answer tags. Example: <answer> B
</answer>
User: {question}
A. {option_a} B. {option_b} C. {option_c} D. {option_d}
Answer with the letter of the correct option (A, B, C, or D).
TechQASystem: You are an expert technical support assistant specializing in IT
infrastructure, software products, and enterprise systems. Answer the technical
question to the best of your ability. Be concise and factual. Wrap your answer in:
<answer> YOUR_ANSWER </answer>
User: {question}
Table 15: Evaluation prompts used for each benchmark. Variables in braces are replaced with benchmark data at
evaluation time.
Category Count %
Grounded (in document) 281 41%
Derived calculations 255 37%
Rounded/reformatted 90 13%
Pure fabrications 62 9%
Total not in document 407 59%
Table 16: Classification of unique numbers appearing
in Self-Study training data across 9 FinQA cartridges
(688 unique numbers total, excluding years).
For example, in a Lockheed Martin document,
the text-derived delta “operating profit decreased
$264 million” achieves 87% coverage (80/92 sam-
ples), while the underlying table values it was com-
puted from—$6,608M, $6,770M, $7,092M (net
sales 2014–2016)—appear at only 3–5% coverage.All 4 test questions requiring these table values
score 0%.
Does coverage predicts correctness?Table 17
shows the relationship between minimum value
coverage and answer correctness. A coverage
threshold of ∼25% is necessary (87% of correct an-
swers meet it) but not sufficient (46% of incorrect
answers also meet it), indicating that additional fail-
ure modes—such as entity confusion—contribute
to errors even when values are present.
Entity confusion.The cartridge stores numeric
values but loses entity-value associations. In a UPS
shareowner return document, the model correctly
retrieves UPS’s return ($121.46, 100% coverage)
but substitutes the S&P 500 value ($108.59, 69%
19

Min coverage Correct (n=15) Incorrect (n=13)
≥0% 100% 100%
≥10% 100% 69%
≥25% 87% 46%
≥30% 67% 23%
≥50% 27% 23%
Table 17: Fraction of questions meeting a minimum
coverage threshold, split by correctness. Coverage ≥
25%is necessary but not sufficient.
coverage) for the Dow Jones Transportation value
($127.07, 69% coverage) when computing their
difference—both distractor values have identical
coverage, and the model picks the wrong entity.
Hallucination despite self-verification.The
model fabricates plausible values, attempts chain-
of-thought verification, detects inconsistencies, and
proceeds anyway. In a Hologic acquisition docu-
ment, the model needs the goodwill value ($6,900,
28% coverage) but produces three different hallu-
cinated values across 3 runs (4,200, 6,000, 3,000).
In each run, it sums the allocation components, no-
tices the total does not match $31,300, yet outputs
the wrong formula regardless.
K Artifact Licenses and Terms of Use
Licenses and terms for use/distribution of arti-
facts.The five benchmarks used in this work are
publicly available under the following licenses:
•LongHealth(Adams et al., 2024): Released
under the Apache-2.0 License.13The dataset
consists of entirely fictional patient records
created by the authors; no real patient data is
included.
•QASPER(Dasigi et al., 2021): Released un-
der the CC BY 4.0 License.14The dataset
consists of questions and answers over NLP
research papers; the dataset itself is distributed
under CC BY 4.0.
•QuALITY(Pang et al., 2022): Released un-
der the CC BY 4.0 License.15Source texts
are drawn from Project Gutenberg (public do-
main) and other permissively licensed collec-
tions, including nonfiction and fiction sources
such as Slate articles from the Open American
National Corpus, The Long+Short, Freesouls,
13https://github.com/kbressem/LongHealth/blob/
main/LICENSE
14https://huggingface.co/datasets/allenai/
qasper
15https://nyu-mll.github.io/quality/and Open Access books. These texts are pub-
lished works and do not contain personal data,
though some may include mature themes typi-
cal of literary and journalistic content.
•FinQA(Chen et al., 2021): Released under
the MIT License.16The underlying finan-
cial reports are sourced from corporate fil-
ings made available via the SEC EDGAR
system, including earnings reports and an-
nual/quarterly reports (e.g., 10-K and 10-Q
documents). These are publicly accessible fi-
nancial disclosures prepared by reporting com-
panies. T2-RAGBench (Strich et al., 2026) is
also released under the MIT License.
•TechQA(Castelli et al., 2020): The NVIDIA
TechQA-RAG-Eval variant used in this work
is released under the Apache-2.0 License and
is explicitly cleared for commercial and non-
commercial use.17It is derived from the origi-
nal IBM TechQA dataset, whose code reposi-
tory is also Apache-2.0.18
The base model used in our experiments,
Qwen3-8B(Yang et al., 2025), is released under
the Apache 2.0 License, which permits research
and commercial use.
We do not release new datasets in this work.
The trained cartridge artifacts (KV cache parame-
ters) are derivatives of the above datasets and the
Qwen3-8B model weights; any release of such ar-
tifacts would be subject to the intersection of the
applicable licenses above.
Offensive content and personal data.
•LongHealth: The dataset consists of entirely
fictional patient records with no real individ-
uals. No anonymization was required. We
verified that no real names, addresses, or iden-
tifying information appear in the data.
•QASPER: Source texts are NLP research pa-
pers. No personal data or offensive content is
expected; the domain is scientific writing.
•QuALITY: Source texts are fiction and non-
fiction narratives, these are published literary
works; no personal data is present.
•FinQA / T2-RAGBench: Source texts are
corporate earnings reports from SEC EDGAR.
These are formal financial documents; no per-
16https://github.com/czyssrs/FinQA/blob/main/
LICENSE
17https://huggingface.co/datasets/nvidia/
TechQA-RAG-Eval
18https://github.com/IBM/techqa/blob/master/
LICENSE.md
20

sonal data or offensive content is present.
•TechQA: Source texts are IBM Technote IT
support documents. These are technical docu-
mentation; no personal data or offensive con-
tent is expected.
No additional anonymization steps were taken
beyond those applied by the original dataset cre-
ators, as none of the datasets contain personal data
about private individuals.
L AI use disclosure
We used AI to assist with code writing and
manuscript typesetting.
21