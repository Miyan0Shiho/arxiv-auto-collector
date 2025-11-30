# NNGPT: Rethinking AutoML with Large Language Models

**Authors**: Roman Kochnev, Waleed Khalid, Tolgay Atinc Uzun, Xi Zhang, Yashkumar Sanjaybhai Dhameliya, Furui Qin, Chandini Vysyaraju, Raghuvir Duvvuri, Avi Goyal, Dmitry Ignatov, Radu Timofte

**Published**: 2025-11-25 14:10:44

**PDF URL**: [https://arxiv.org/pdf/2511.20333v1](https://arxiv.org/pdf/2511.20333v1)

## Abstract
Building self-improving AI systems remains a fundamental challenge in the AI domain. We present NNGPT, an open-source framework that turns a large language model (LLM) into a self-improving AutoML engine for neural network development, primarily for computer vision. Unlike previous frameworks, NNGPT extends the dataset of neural networks by generating new models, enabling continuous fine-tuning of LLMs based on closed-loop system of generation, assessment, and self-improvement. It integrates within one unified workflow five synergistic LLM-based pipelines: zero-shot architecture synthesis, hyperparameter optimization (HPO), code-aware accuracy/early-stop prediction, retrieval-augmented synthesis of scope-closed PyTorch blocks (NN-RAG), and reinforcement learning. Built on the LEMUR dataset as an audited corpus with reproducible metrics, NNGPT emits from a single prompt and validates network architecture, preprocessing code, and hyperparameters, executes them end-to-end, and learns from result. The PyTorch adapter makes NNGPT framework-agnostic, enabling strong performance: NN-RAG achieves 73% executability on 1,289 targets, 3-shot prompting boosts accuracy on common datasets, and hash-based deduplication saves hundreds of runs. One-shot prediction matches search-based AutoML, reducing the need for numerous trials. HPO on LEMUR achieves RMSE 0.60, outperforming Optuna (0.64), while the code-aware predictor reaches RMSE 0.14 with Pearson r=0.78. The system has already generated over 5K validated models, proving NNGPT as an autonomous AutoML engine. Upon acceptance, the code, prompts, and checkpoints will be released for public access to enable reproducibility and facilitate community usage.

## Full Text


<!-- PDF content starts -->

NNGPT: Rethinking AutoML with Large Language Models
Roman Kochnev Waleed Khalid Tolgay Atinc Uzun Xi Zhang Yashkumar S. Dhameliya
Furui Qin Chandini Vysyaraju Raghuvir Duvvuri Avi Goyal Dmitry Ignatov Radu Timofte
Computer Vision Lab, CAIDAS & IFI, University of W ¨urzburg, Germany
Abstract
Building self-improving AI systems remains a fundamen-
tal challenge in the AI domain. We present NNGPT, an
open-source framework that turns a large language model
(LLM) into a self-improving AutoML engine for neural net-
work development, primarily for computer vision. Unlike
previous frameworks, NNGPT extends the dataset of neu-
ral networks by generating new models, enabling contin-
uous fine-tuning of LLMs based on closed-loop system of
generation, assessment, and self-improvement. It integrates
within one unified workflow five synergistic LLM-based
pipelines: zero-shot architecture synthesis, hyperparameter
optimization (HPO), code-aware accuracy/early-stop pre-
diction, retrieval-augmented synthesis of scope-closed Py-
Torch blocks (NN-RAG), and reinforcement learning. Built
on the LEMUR dataset as an audited corpus with repro-
ducible metrics, NNGPT emits from a single prompt and
validates network architecture, preprocessing code, and hy-
perparameters, executes them end-to-end, and learns from
result. The PyTorch adapter makes NNGPT framework-
agnostic, enabling strong performance: NN-RAG achieves
73% executability on 1,289 targets, 3-shot prompting boosts
accuracy on common datasets, and hash-based deduplica-
tion saves hundreds of runs. One-shot prediction matches
search-based AutoML, reducing the need for numerous tri-
als. HPO on LEMUR achieves RMSE 0.60, outperform-
ing Optuna (0.64), while the code-aware predictor reaches
RMSE 0.14 with Pearsonr= 0.78. The system has already
generated over 5K validated models, proving NNGPT as an
autonomous AutoML engine. Upon acceptance, the code,
prompts, and checkpoints will be released for public access
to enable reproducibility and facilitate community usage.
1. Introduction
A longstanding goal in machine learning is to build sys-
tems that design and improve themselves. In practice, as-
sembling a competitive neural pipeline still requires sub-
stantial expert effort: choosing architectures, writing trans-
forms, specifying losses and metrics, tuning hyperparame-ters, and running many training jobs. Classical AutoML,
for example Optuna [2] with Tree-structured Parzen Esti-
mator (TPE) [5], automates parts of this process but relies
on black-box search that requires hundreds of trials and ig-
nores the semantics of the underlying code.
Large language models (LLMs) such as GPT-4 [42],
Code Llama [47], and DeepSeek [11] excel at code gener-
ation, which has motivated attempts to let them propose ar-
chitectures, hyperparameters, or training scripts. Most prior
work, however, remains at design time: the LLM emits frag-
ments or templates that are only loosely tied to execution,
logging, or learning from outcomes, so the feedback loop
from generation to training to improved generation is rarely
closed.
We ask whether a single LLM, embedded in an online
loop with executable code and training feedback, can serve
as the core of a self-improving AutoML engine for neural
networks. Answering this requires representing full training
pipelines as executable specifications, validating and run-
ning them at scale, predicting performance early, and using
logs to continuously improve the LLM.
We introduceNNGPT, an open-source framework that,
to our knowledge, is the first to unify zero-shot model gen-
eration and editing, hyperparameter recommendation, ac-
curacy and early-stop prediction, retrieval-augmented code
synthesis, and an reinforcement learning (RL)-based im-
provement loop into a single closed AutoML cycle driven
by prompt-based LLM inference. From one high-level
prompt, NNGPT produces an executable training specifica-
tion (model, transforms, metrics, optimizer, schedule), exe-
cutes it, logs code and metrics, and uses these logs to fine-
tune and reinforce the underlying LLMs.
NNGPT is built on LEMUR [16], which we treat as a
corpus of executable neural programs with audited imple-
mentations, unified preprocessing, and reproducible met-
rics. This grounds one-shot generation in runnable Py-
Torch [44] code rather than abstract templates. The frame-
work remains PyTorch-agnostic via a thin adapter that ex-
poses model factories, data transforms, and metric bindings.
In practice, NNGPT has generated and trained over 10,000
distinct models (over 5,000 through its self-improving loop
1arXiv:2511.20333v1  [cs.AI]  25 Nov 2025

Method LLM Arch. HPO Full train One-shot Closed loop Acc. pred. Open source
Optuna [2]✗ ✗ ✓ ✗ ✗ ✗ ✗ ✓
Hyperopt [6]✗ ✗ ✓ ✗ ✗ ✗ ✗ ✓
Google Vizier [15]✗ ✗ ✓ ✗ ✗ ✗ ✗ ✓
ENAS [46]✗ ✓ ✓ ✗ ✗ ✗ ✗ ✓
DARTS [32]✗ ✓ ✓ ✗ ✗ ✗ ✗ ✓
AutoKeras [24]✗ ✓ ✓ ✗ ✗ ✗ ✗ ✓
EvoPrompting [7]✓ ✓ ✓ ✗ ✗ ✓ ✗ ✗
LLAMBO [35]✓ ✗ ✓ ✗ ✗ ✓ ✗ ✗
LLMatic [40]✓ ✓ ✓ ✗ ✗ ✓ ✗ ✗
GPT-NAS [56]✓ ✓ ✓ ✗ ✗ ✓ ✗ ✗
Self-Programming AI [50]✓ ✓ ✓ ✗ ✗ ✓ ✗ ✗
Grammar / schema prompting [52]✓ ✗ ✗ ✗ ✓ ✗ ✗ ✗
NNGPT (ours)✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓
Table 1. Feature comparison between NNGPT and representative AutoML and LLM-based methods. Columns indicate whether a method
uses an LLM, generates architectures (Arch.), performs hyperparameter optimization (HPO), synthesizes full training specifications, oper-
ates in a one-shot fashion, supports a closed feedback loop, includes accuracy prediction, and is open source.
with LLMs), with training statistics, all incorporated into
LEMUR as verified outcomes.
The closed-loop view is as follows. A fine-tuned LLM
emits network architectures, preprocessing code, and hy-
perparameters; an execution engine compiles, trains, and
logs; a code-aware predictor forecasts final accuracy and a
stopping epoch from model code plus early metrics; a re-
trieval module patches or synthesizes PyTorch blocks; and
an RL layer updates LoRA adapters [22] using rewards de-
rived from executability and downstream performance.
Our experiments demonstrate that LLMs can function
as autonomous AutoML systems: they generate com-
plete pipelines, predict performance, and refine themselves
based on real-time feedback within a continuous self-
improvement loop.
1.1. Related Work
Prior work falls into two strands: (i) search-based AutoM-
L/NAS and (ii) LLM-driven generation.
Search-based AutoML and NAS.Classical HPO
frameworks such as Optuna [2], Hyperopt [6], and Google
Vizier [15] rely on Bayesian optimization or TPE to iter-
atively probe the space of hyperparameters. These tools
are strong baselines but typically require hundreds of tri-
als, treat training code as an opaque black box, and do not
synthesizecompletetrain specs (data, metrics, optimizer).
NAS systems, including ENAS [46], DARTS [32], and Au-
toKeras [24], extend search to architectures (with many sta-
bilized variants), yet remain compute-intensive and slow to
adapt to new tasks and codebases.
LLM-driven generation and closed loops.Recent
methods integrate LLMs into design loops: EvoPrompt-
ing [7] uses an LLM as a mutation operator in evolution-
ary search; LLAMBO [34] couples LLM suggestions with
Bayesian optimization; LLMatic [40], GPT-NAS [56], and
Self-Programming AI [50] generate code or fill templates.
Grammar/schema prompting [52] and retrieval help withvalidity. However, most systems either require repeated
LLM queries, cover only fragments of the pipeline (e.g.,
mutations without training integration), or lack an online
mechanism to learn from executed runs.
Performance prediction.Predicting final accuracy and
early stopping has been studied via curve fitting from
early learning curves [1, 14], architecture- or text-based
surrogates [36, 53], and graph-level representations [13].
These approaches often abstract away implementation de-
tails (losses, custom layers, metric code) that strongly af-
fect convergence, and they rarely integrate with executable
pipelines.
Position of NNGPT.Table 1 situates NNGPT among
these lines. Unlike prior art, NNGPT (1)one-shotgen-
erates schema-validatedfulltraining specifications (archi-
tecture & HPO) that run immediately; (2) replaces costly
search for many tasks with LLM-based hyperparameter rec-
ommendation; (3) includes acode-awarepredictor that con-
sumes executable code and early metrics to forecast final
accuracy and a stopping epoch; and (4) closes the loop via
LoRA/RL updates using execution logs - within a single
open-source framework. A comprehensive literature review
with extended comparisons is provided in the Supplemen-
tary Material, Sec. 6.
2. NNGPT Framework
2.1. System Overview
NNGPT exposes LLMs as end to end configuration agents
for neural networks. Given a high level task description,
the system constructs a structured prompt, calls an LLM
once, validates the structured output, executes the resulting
training pipeline, logs all artefacts, and feeds the trace back
into its training data.
At the core of NNGPT is a shared execution substrate
built on the LEMUR dataset framework, which provides ex-
ecutable PyTorch models, standardized training loops, and
2

Figure 1. Overview of the NNGPT Pipeline: starting with a query to the LEMUR API to retrieve a neural network entry and its metadata,
the system constructs a prompt, using an LLM generates the code of neural network model, and trains it. Artifacts are logged in a structured
database, while LoRA fine-tuning continuously updates the LLM based on training results, creating a self-improving AutoML loop.
reproducible evaluation metrics. To avoid hard coupling to
LEMUR, the framework defines a thin adapter interface for
arbitrary PyTorch codebases that maps a configuration ob-
ject to a model constructor, data transforms, and metric def-
initions. Any model that implements this interface can be
driven by the same LLM generated specification.
A typical pass through the system proceeds as follows.
First, the configuration module aggregates task level infor-
mation such as dataset, objective, resource limits, and op-
tional seed architectures. Next, a prompt builder converts
this state into a JSON like instruction block that encodes
the desired output schema. The LLM backend then emits
a structured, machine readable specification with explicit
fields for architecture, hyperparameters, and training proce-
dure. A schema layer based on Pydantic validates the output
and either autocorrects minor issues or requests a single re-
generation in case of structural violations. Valid configura-
tions are handed off to a distributed training engine, which
runs the experiment, logs metrics and metadata in an SQLite
store, and optionally exports models back into the LEMUR
corpus. Finally, dedicated modules use these traces to fine
tune the LLM with LoRA, to train an accuracy and early
stopping predictor, or to update a RL signal. This shared
infrastructure underpins all pipelines described below.
2.2. Problem Formulation
We cast NNGPT as closed-loop neural pipeline synthesis
problem. Given a task promptP(task, dataset, metric, op-
tional budget) and an optional retrieval setR(code and
metadata from LEMUR or external projects), the goal is
to produce an executable training specificationCand a
horizonTthat maximize downstream performance under
a compute budget.
An LLM generatorG θmaps
(P, R)7→ ˜C∈ C,
where ˜Cis a YAML/JSON spec that defines the model,
data transforms, optimization schedule, loss, and nominal
epochs. A validatorV(schema + type/shape checks) en-
forces structural constraints, auto-fixes minor issues or trig-
gers a single re-prompt; onlyV( ˜C) = passis promoted to
C.GivenCandT, the trainerEexecutes the pipeline,
(m,u) =E(C, T),
producing validation metrics over epochsmand auxiliary
logsu(loss curves, learning rate, runtime, hardware). Code
and traces are persisted in a structured log.
To cut wasted compute, a code-aware predictorH ϕesti-
mates
(cacc∗,bt∗) =H ϕ(C, R,m 1:t0),
wherecacc ∗approximates the final accuracy and bt∗proposes
an early-stopping epoch from the firstt 0epochs. This in-
duces an adaptive horizonT′=bt∗and enables early termi-
nation and priority scheduling.
Over time the log
Dlog={(P(k), R(k), C(k),m(k),u(k))}K
k=1
updatesθandϕ:G θis refined with LoRA fine-tuning
and lightweight policy-gradient updates using rewards from
executability and downstream performance, whileH ϕis
trained to minimize prediction error on final accuracy and
stopping epoch. This closes the loop: generate, validate,
execute, log, update, under an explicit compute budget.
2.3. Pipelines in NNGPT
On top of the common backbone, NNGPT implements five
concrete pipelines that target different stages of the model
development process while reusing the same prompting,
validation, and execution stack. As summarized in Fig. 1,
all pipelines follow one seven-step template: LEMUR re-
trieval→configuration setup/validation→prompt assem-
bly & injection→LLM-guided generation→model val-
idation/training→LoRA fine-tuning→database logging.
The pipeline-specific behavior is concentrated in thePrompt
stage (steps 3–4): the prompt schema, retrieved context, and
expected outputs differ across zero-shot generation/editing,
hyperparameter recommendation, accuracy/early-stop pre-
diction, NN-RAG patching, and the RL loop. Implementa-
tion details of the shared stages are expanded in Fig. 2.
2.3.1. Zero-shot model generation and editing
The first pipeline turns LLMs into generators of com-
plete architectures and training specifications. Given a task
3

Figure 2. Detailed view of pipeline stages 3–7 in NNGPT. Left to right: Configuration Setup & Validation (default or HF-loaded LLM,
PEFT/quantization); Prompt Assembly & Injection (templated prompts with task data); LLM-Guided Architecture Generation (one-shot
code and hyperparameters); Model Validation & Evaluation (format checks, training, logging); LoRA Fine-tuning (update adapters from
train/ *.json). All pipelines reuse this stack; variation mainly occurs in the Prompt stage.
prompt and a target dataset, NNGPT queries LEMUR for
high-performing models, selects onereferenceimplemen-
tation andnadditionalsupportingexamples, and assem-
bles a Few-Shot Architecture Prompting (FSAP) prompt
that concatenates the task description, dataset specs, and
full PyTorch classes for these models. The LLM (typically a
LoRA-fine-tuned DeepSeek-Coder-7B) is instructed to emit
a new scope-closed architecture class plus a YAML config-
uration rather than a local patch. A whitespace-normalized
MD5 hash validator removes duplicate programs in<1ms,
preventing redundant training of architectures that differ
only in formatting. Generated specifications are checked
by the schema layer and executed by the training engine;
successful runs are logged and added back to LEMUR as
new entries. See Sec. 3.1 for the experimental setup and
evaluation.
2.3.2. Hyperparameter recommendation
The second pipeline treats hyperparameter tuning as con-
ditional text generation over a fixed schema. Training data
are drawn from LEMUR runs that link each model to its
dataset, transforms, target accuracy, epoch budget, and the
hyperparameters that achieved this performance. Prompts
expose the model name,nn code,transform code,
task, dataset, and either a target accuracy or a “maximize ac-
curacy” instruction; the LLM is asked to output only the val-
ues for a predefined set of hyperparameters (learning rate,
momentum, weight decay, batch size, etc.) in a strict order.
We fine-tune several DeepSeek and CodeLlama vari-
ants with LoRA on this corpus and integrate the best
checkpoints (DeepSeek-Coder-1.3B-Base and DeepSeek-
R1-Distill-Qwen-7B) as plug-and-play recommenders in-
side NNGPT. At inference time, the hyperparameterpipeline runs as a one-shot replacement for search-based
HPO: a single LLM call produces a configuration that is
executed by the same backend as Optuna, and outcomes are
logged back to LEMUR. Section 3.2 shows that these one-
shot recommendations reach accuracy comparable to TPE-
based Optuna on LEMUR, while eliminating thousands of
optimization trials.
2.3.3. Accuracy and early-stop prediction
The third pipeline uses an LLM as a multi-task predictor
of training dynamics. Each training run in LEMUR is con-
verted into a structured prompt that fuses three modalities:
(i) structured metadata (task, dataset, maximum epochs),
(ii) executable code snippets for the model, metric, and data
transforms, and (iii) early-epoch validation accuracies (typi-
cally at epochs 1 and 2). A DeepSeek-Coder-1.3B backbone
is adapted with 4-bit QLoRA, reducing memory by∼4×
while keeping full context, and fine-tuned to regress both
the final best validation accuracy and the epoch at which
it occurs under a constrained JSON/YAML output schema.
Stratified group splitting by (task, dataset, architecture) pre-
vents leakage between similar models and enforces out-of-
architecture generalization.
Within NNGPT, the predictor runs alongside training:
after a few epochs, it estimates(cacc ∗,bt∗)and can trigger
early stopping when predicted accuracy is low, or prioritize
promising runs in the job scheduler. As shown in Sec. 3.3,
the baseline model achieves RMSE≈0.145and Pearson
r≈0.78over 1,200+ runs, making it practical for compute-
aware filtering and scheduling.
2.3.4. NN-RAG: retrieval augmented patching
The fourth pipeline augments generation with a retrieval
system for reusable PyTorch blocks. NN-RAG scans cu-
4

rated repositories for non-abstractnn.Modulesubclasses
that implementforward(), round-trips their source
through LibCST [30] to preserve concrete syntax, and in-
dexes parse trees and import relations in an SQLite-backed
store with SHA-1 content addressing. Given a target ar-
chitecture and a textual or code query (e.g., “multi-head
attention block”), NNGPT retrieves candidate blocks and
uses a scope-sensitive resolver to compute the minimal
transitive closure of dependencies that respects Python’s
LEGB and import semantics. Definitions are reordered via
topological sort to satisfy definition-before-use constraints,
and the resulting scope-closed module is passed through
AST parsing, bytecode compilation, and sandboxed im-
port. On a corpus of 1,289 target blocks, this validator ad-
mits 941 modules (73% executability), covering attention,
convolutions, transformer blocks [51], losses, and higher-
level architectures. These validated blocks form a library
that NNGPT can splice into generated models, enabling
retrieval-augmented patching while preserving executabil-
ity guarantees.
2.3.5. Reinforcement learning based closed loop
We add an RL layer on top of supervised fine-tuning, treat-
ing the LLM as a policy over architecture code and con-
figurations. From LEMUR we build a masked-architecture
corpus where layer definitions (e.g.,self.conv1 =
nn.Conv2d(...)) are replaced by placeholders; the
model learns to reconstruct these blocks, acquiring realis-
tic priors. At RL time it proposes full architectures that are
executed viacheck nn, which tests compilation, forward
correctness, and a short CIFAR-10 [26] run. A scalar reward
mixes syntactic validity, runtime feasibility, and validation
accuracy, and a GRPO-style policy-gradient update (in-
spired by DeepSeek-R1 [10]) is applied to LoRA adapters.
In parallel, a channel-mutation engine performs deter-
ministic width edits through five steps:torch.fxtracing,
mapping graph nodes to source, planning shape-preserving
changes, AST rewriting, and re-validation. Both RL-
generated and mutated models pass through the same val-
idation and training stack, and successful configurations are
added to LEMUR, closing a performance-aware loop that
explores and reinforces architectures that train well in prac-
tice.
Across all five pipelines, accepted configurations, logs,
and artifacts are exported back into LEMUR dataset or ex-
ternal repositories, turning raw LLM generations into a con-
tinually expanding corpus of executable neural programs
that subsequent NNGPT runs can query, modify, and build
upon.
3. Experiments
Experiments are conducted on a single 24GB GPU (Geforce
RTX 3090/4090) at a time, using PyTorch and 8-bit or 4-bitLoRA adapters on top of code-capable LLMs within the AI
Linux1environment on the Kubernetes cluster. NNGPT in-
teracts with the LEMUR corpus through a fixed API that
exposes model factories, transformation pipelines, and met-
ric definitions, but any PyTorch codebase can be integrated
via the same adapter interface. For each pipeline we re-
port results on mid-scale vision benchmarks (e.g. CIFAR-
like classification and COCO-style detection/segmentation
tasks), using standardized training and evaluation protocols
from LEMUR. Further engineering details, CLI tools, and
prompts are provided in the Supplementary Material, Sec. 7.
3.1. Zero-Shot Generation and LEMUR Growth
To quantify NNGPT’s ability to expand LEMUR via zero-
shot generation, we instantiate a Few-Shot Architecture
Prompting (FSAP) pipeline. For each base model and
dataset, the LLM receives a natural-language task descrip-
tion, one reference implementation, andn∈ {1, . . . ,6}
supporting architectures sampled from strong LEMUR en-
tries, and is asked to synthesize a new PyTorch model that
follows the required API. We denote the resulting vari-
antsalt-nn1–alt-nn6. Generated code passes through
schema validation and a whitespace-normalized MD5 hash
check to remove duplicates before training.
Table 2 summarizes generation statistics. Withn= 1
(alt-nn1), NNGPT produces 3,394 valid architectures;
increasing the number of supporting examples sharply re-
duces success, down to 306 models forn= 2and≈100
models forn= 3–5. Atn= 6(alt-nn6) the sys-
tem generates only 7 models (99.8% failure), indicating se-
vere context overflow. Across all settings, we obtain 4,033
candidates, of which about 1,900 are unique after hash-
based deduplication; the hash validator rejects∼100 near-
duplicate programs and saves an estimated 200–300 GPU
hours of redundant training.
To compare quality across datasets with different sample
sizes, we report dataset-balanced means (Table 3). The bal-
anced mean accuracy peaks atn= 3(alt-nn3, 53.1%)
compared to 51.5% for then= 1baseline, while larger
contexts (n= 4,5) degrade performance to 47.3% and
43.0%. Gains are most pronounced on harder tasks such
as CIFAR-100, wherealt-nn3improves overalt-nn1
by +11.6 percentage points after a single epoch. In aggre-
gate, this pipeline contributes roughly 1.9k new architec-
tures (and over 3k trained runs) to LEMUR, demonstrating
that NNGPT can continuously grow its own training corpus
through LLM-driven zero-shot generation.
3.2. Hyperparameter Generation Quality
We evaluate how well fine-tuned DeepSeek and CodeLlama
models generatevalidhyperparameter sets from structured
1AI Linux:https://hub.docker.com/r/abrainone/ai-linux
5

Variantn(examples) Models Success rate
alt-nn11 3,394 100%
alt-nn22 306 9.0%
alt-nn33 103 3.0%
alt-nn44 102 3.0%
alt-nn55 121 3.6%
alt-nn66 7 0.2%†
Total–4,033–
Table 2.FSAP yield vs. context size.The LLM gets one reference
model andnsupporting examples from LEMUR and must emit
a complete, executable PyTorch architecture. Variantsalt-nnk
useksupports.Modelscount schema-validated, deduplicated ar-
chitectures;Success rateis yield relative ton=1(100%). The
drop atn=6indicates context overflow. In total, 4,033 candidates
were produced (∼1.9k unique;∼100 near-duplicates filtered).
†99.8% relative drop vs.n=1indicates severe context overflow.
VariantnModels Balanced Mean (%)
alt-nn11 3,39451.5±29.5
alt-nn22 30649.8±32.3
alt-nn33 10353.1±26.9
alt-nn44 10247.3±32.5
alt-nn55 12143.0±33.1
Table 3.Accuracy of generated architectures (dataset-
balanced mean).For each variantalt-nnkwe train every ac-
cepted model for one epoch with SGD+momentum and report top-
1 accuracy. To avoid bias from uneven per-dataset sample counts,
we first average within each (variant, dataset) pair and then macro-
average across datasets (equal weight per dataset).alt-nn6is
omitted due to insufficient sample size (n=7).Boldmarks the
best overall balanced mean.
prompts. Given a specification describing model architec-
ture, dataset, task, transform pipeline, and desired accuracy,
the LLM must output a complete hyperparameter config-
uration (names and values) that conforms to the expected
schema (Listing 5). Evaluation is carried out on 500 held-
out neural configurations sampled from LEMUR; a genera-
tion is counted as correct if it passes schema and value-type
checks.
Table 4 reports the number of correct generations.
The best overall result is obtained by DeepSeek-R1-
Distill-Qwen-7B [10] (20 epochs), with 465 valid out-
puts (93.00%), slightly outperforming our fine-tuned
CodeLlama-7b-Python-hf [47] (460/500, 92.00%), which
has previously been used as a strong baseline for this
task [25]. Smaller and domain-specialized DeepSeek vari-
ants also perform well: DeepSeek-Coder-1.3b-base [17]
reaches 88.40% after 15 epochs, and DeepSeek-Coder-
7b-Instruct-v1.5 [17] achieves a stable 73.60% at 15–20
epochs. At the same time, several runs with pro-Exp. Model Ep.Correct
/500FailedPct.
(%)
1 DeepSeek-Coder-V2-Lite-Base 15 8 492 1.60
2 DeepSeek-Coder-1.3b-base 10 435 65 87.00
3 DeepSeek-Coder-1.3b-base 15 442 58 88.40
4 DeepSeek-Coder-1.3b-base 25 0 500 0.00
5 DeepSeek-Coder-1.3b-base 35 4 496 0.80
6 DeepSeek-R1-Distill-Qwen-7B 15 457 43 91.40
7 DeepSeek-R1-Distill-Qwen-7B 20 465 35 93.00
8 DeepSeek-R1-Distill-Qwen-7B 35 176 324 35.20
9 DeepSeek-R1-Distill-Qwen-7B 35 268 232 53.60
10 DeepSeek-Coder-7b-base-v1.5 15 127 373 25.40
11 DeepSeek-Coder-7b-base-v1.5 20 114 386 22.80
12 DeepSeek-Coder-7b-base-v1.5 35 0 500 0.00
13 DeepSeek-Math-7b-base 15 214 286 42.80
14 DeepSeek-Math-7b-base 35 1 499 0.20
15 DeepSeek-Coder-7b-Instruct-v1.515 368 132 73.60
16 DeepSeek-Coder-7b-Instruct-v1.5 20 368 132 73.60
17 DeepSeek-Coder-7b-Instruct-v1.5 35 23 477 4.60
18 CodeLlama-7b-Python-hf 21 4604092.00
Table 4.Structured hyperparameter generation on 500 held-
out LEMUR configs.Each row is a fine-tuned checkpoint (model,
epochs). For every config, the LLM must output a complete hyper-
parameter set that matches a strict schema; a prediction is counted
asCorrectif it passes all schema/value checks (elseFailed). Per-
centage is Correct/500. Bold = best per family and overall.
17 CV Models (as in [25])
Method RMSE SE 95% CI
Optuna∗0.589 0.004 [0.581, 0.597]
CodeLlama-7b-Python-hf∗0.5630.004 [0.556, 0.570]
DeepSeek-Coder-1.3b-base 0.651 0.032 [0.588, 0.714]
DeepSeek-R1-Distill-Qwen-7B 0.659 0.037 [0.586, 0.732]
CodeLlama-7b-Python-hf 0.642 0.031 [0.554, 0.729]
LEMUR Dataset
Method RMSE SE 95% CI
Optuna 0.636 0.002 [0.633, 0.640]
DeepSeek-Coder-1.3b-base 0.649 0.062 [0.526, 0.771]
DeepSeek-R1-Distill-Qwen-7B 0.652 0.029 [0.595, 0.709]
CodeLlama-7b-Python-hf0.6030.031 [0.543, 0.663]
Table 5.Downstream training with LLM-generated hyperpa-
rameters vs. Optuna.We compare one-shot LLM recommenda-
tions (no iterative search; 35–50 predictions per model) against
Optuna (>20k trials) on two evaluation sets. Metric: RMSE↓
ofϵ i=1−a i, wherea iis the observed validation accuracy when
trained with predicted HPs. We report SE and normal-approx. 95%
CI. Rows∗reproduced from [25].
longed fine-tuning collapse to near-zero valid generations
(e.g., 25–35 epochs for DeepSeek-Coder-1.3b-base and
DeepSeek-Math-7b-base [49]), highlighting the sensitiv-
ity of structured generation tasks to overfitting and over-
training.
To assess downstream effectiveness, we use the gen-
erated hyperparameters to train models on two evaluation
sets: (i) the 17-model CV benchmark from [25] and (ii) the
LEMUR dataset. For each configuration, we compute an er-
ror termϵ i= 1−a i, wherea iis the observed accuracy un-
der the predicted hyperparameters, and report RMSE over
6

Dataset RMSE [95% CI] Pearsonr(p) Notes
CelebA-Gender 0.0497 [0.044, 0.056] 0.2125 (p= 2.38×10−7) Low
correlation
despite low
RMSE
CIFAR-10 0.1589 [0.150, 0.168] 0.3472 (p= 3.77×10−30) Moderate
correlation
CIFAR-100 0.1668 [0.155, 0.178]−0.0276(ns) Very poor
ranking
ability
ImageNette 0.2439 [0.221, 0.266] 0.4973 (p= 1.25×10−36) High error,
moderater
MNIST 0.0331 [0.025, 0.040] 0.4224 (p= 1.04×10−22) Excellent
precision
Places365 0.1672 [0.135, 0.193]−0.0297(ns) Weak gener-
alization
SVHN 0.0605 [0.058, 0.063] 0.3587 (p= 4.08×10−20) Lowr,
tolerable
RMSE
COCO 0.1494 [0.138, 0.160] 0.8173 (p <10−4) Strongest
dataset
correlation
Overall 0.1449 0.7766Captures
trends;
uneven by
dataset
Table 6.Per-dataset accuracy prediction (baseline, Exp. 1).
RMSE reflects regression error for final accuracy; Pearsonrmea-
sures ranking quality.
ϵialong with standard errors and 95% confidence intervals
(Table 5).
On the 17-model benchmark, prior results for
CodeLlama-7b-Python-hf [25] achieve an RMSE of
0.563, statistically outperforming Optuna [3] (0.589). Our
DeepSeek-based models yield higher RMSEs and wider
intervals on this heterogeneous testbed, consistent with
their sensitivity to overfitting. On the LEMUR distribution,
where schemas and transforms align more closely with
training, DeepSeek-Coder-1.3b-base and DeepSeek-R1-
Distill-Qwen-7B obtain RMSEs of 0.649 and 0.652, close
to Optuna’s 0.636, while our fine-tuned CodeLlama-7b-
Python-hf achieves the best RMSE of 0.603. Although
LLM-based methods show wider confidence intervals
(partly because each model is evaluated on only 35–50
one-shot predictions compared with over 20 000 trials for
Optuna), they approach or exceed Optuna’s accuracy while
avoiding iterative search, demonstrating that one-shot LLM
hyperparameter generation can serve as a competitive,
low-cost alternative in the NNGPT loop.
3.3. Accuracy and Early-Stop Prediction
We instantiate the predictorH ϕas a code-aware LLM
fine-tuned with 4-bit QLoRA on 1,200+ NNGPT/LE-
MUR runs spanning CNNs and Transformers [51] on
CelebA-Gender [33], CIFAR-10/100 [26], ImageNette [21],
MNIST [28], Places365 [59], SVHN [41], and COCO [31].
Each run is converted into a structured prompt with
nncode,transform code,metric code, hyperpa-Metric Value
RMSE 0.2567 [0.248–0.265]
MAE 0.1709
R20.2885
Pearson correlation 0.6409 (p= 1.25×10−204)
Within 5% tolerance 12.30%
Within 10% tolerance 45.01%
Samples 1764
Table 7.Experiment 2 (balanced + regularized) – global met-
rics.Aggregate performance of the alternative predictor across all
datasets; RMSE shows 95% CI.
Variant RMSER2rwithin
5%within
10%
Exp. 1 (baseline) 0.1449 0.55 0.7766 50.0% 75.0%
Exp. 2 (balanced + reg.) 0.2567 0.2885 0.6409 12.3% 45.0%
Table 8.Global results over all datasets.Baseline (Exp. 1)
is used in NNGPT; Exp. 2 underperforms under added balanc-
ing/regularization.
rameters, dataset/task metadata, max epochs, and validation
accuracies at epochs 1–2; the model predicts the final best
validation accuracy and its epoch. To prevent leakage, we
use stratified group splits by (task, dataset, architecture).
Per-dataset outcomes are summarized in Tab. 6.
We report two variants. The baseline (Exp. 1) is
a QLoRA-tuned code LLM without additional balanc-
ing/strong regularization; it achieves RMSE= 0.1449,
r= 0.7766, andR2≈0.55globally (see Tab. 8). Corre-
lation varies by dataset, ranging from moderate on CIFAR-
10/SVHN to strong on COCO, while roughly 50% and 75%
of predictions fall within 5% and 10% of the true final accu-
racy, respectively. The dataset-wise heterogeneity is visible
in Tab. 6. We use the predictions both as a proxy for final
performance and to propose an early-stop epochT′=bt∗for
low-potential runs.
An experiment 2 applies inverse-frequency balancing
and stronger regularization (higher dropout/weight decay,
fewer epochs). It underperforms: RMSE rises to0.2567,
R2drops to0.2885,rto0.6409, and only 12.3%/45.0%
of predictions fall within 5%/10% tolerance (Tab. 7; also
compared side-by-side in Tab. 8). We therefore adopt the
baseline predictor in the full NNGPT loop.
The baseline is already useful for early-stop and priority
scheduling, but (i) correlation varies across datasets; (ii) ag-
gressive balancing/regularization can suppress adaptation in
LoRA-tuned LLMs; and (iii) adding uncertainty estimates
is a promising next step for safer automated stopping.
7

Category Count Examples
Attention 180MultiHeadAttention,
SelfAttention
Convolutions 220Conv2d,DeformConv
Transformer blocks 150BertLayer,
SwinTransformerBlock
Pooling ops 110MaxPool2d,
AdaptiveAvgPool2d
Normalization 100BatchNorm2d,LayerNorm
Losses 90FocalLoss,DiceLoss
Architectures 150ResNet,EfficientNet
Utilities 289DropPath,PatchEmbed
Total 1,289–
Table 9.NN-RAG evaluation corpus by category.Distribution
of the 1,289 target PyTorch blocks that NN-RAG attempts to re-
construct as scope-closed modules.Countis the number of distinct
block names per category;Exampleslist typical modules retrieved
in each group.
3.4. NN-RAG: Executability and Correctness
We evaluate NN-RAG as a retrieval-augmented system for
extracting reusable, scope-closed PyTorch blocks. The
pipeline scans repositories for non-abstractnn.Module
classes withforward(), round-trips sources through
LibCST to preserve syntax, and indexes parse artifacts and
import relations in an SQLite store with content hashing.
For each target name, a scope-aware resolver computes
the minimal transitive closure of dependencies (respecting
Python’s LEGB and import rules), orders definitions topo-
logically, and emits a self-contained module that preserves
original imports and aliases. Candidates then pass AST
parsing, compilation, and sandboxed import to catch unre-
solved symbols and import-time failures.
We test on 1,289 target blocks from widely used PyTorch
projects - attention, convolutions, transformer components,
pooling, normalization, losses, architectures, and utilities
(Tab. 9). NN-RAG reconstructs 941 blocks as executable
scope-closed modules, a 73.0% executability rate with-
out manual intervention. Major libraries (torchvision [38],
timm [54], transformers [55], OpenMMLab [43]) contribute
many targets, while a long tail of smaller repos supplies
the rest. This yields a large, validated library of import-
complete building blocks that NNGPT can safely retrieve
and assemble during LLM-driven synthesis.
3.5. Reinforcement Learning Setup
We evaluate the RL loop by generating 32 executable
AirNet models and training each for a single epoch on
MNIST [28], SVHN [41], and CIFAR-100 [26]. As
a non-RL baseline we use one-epoch averages and best
scores from the LEMUR AlexNet [27] (ast-dimension) fam-
ily; the baseline values are embedded in Tab. 10. RL-
generated models match or exceed the baseline on the sim-
pler datasets: on MNIST the average and best accuracies
are 0.9876 and 0.9921, and on SVHN they are 0.8148 andDatasetBaseline
avgRL avg
(n)∆
(avg)Baseline
bestRL
best
CIFAR-100 0.0461 0.0163 (5)−0.02980.2079 0.0253
MNIST 0.7088 0.9876 (13) 0.2788 0.9906 0.9921
SVHN 0.2526 0.8148 (14) 0.5622 0.9032 0.9068
Table 10.RL one-epoch results vs LEMUR baselines.Baseline
avg and best come from LEMURAlexNet (ast-dimension); RL avg
is over generatedAirNetmodels (nshown).∆is RL avg minus
baseline avg.
0.9068. Performance on CIFAR-100 is weak with an aver-
age of 0.0163 and a best of 0.0253, while the LEMUR best
reaches 0.2079. These results indicate that the RL loop reli-
ably produces runnable architectures and can achieve strong
one-epoch accuracy on easier datasets; harder regimes ap-
pear to require more iterations, longer training, and reward
shaping. We treat this study as diagnostic due to small per-
dataset sample sizes and single-epoch budgets.
Limitations.Our deployment on a single 24 GB GPU
limits prompt length, context window, and schema com-
plexity. Experiments focus on mid-scale vision; large-scale
datasets such as ImageNet-1k [12] and LVIS [19] are left
for future work. We also do not yet enforce architectural
novelty or optimize explicitly for hardware efficiency.
4. Conclusion
We introducedNNGPT, a self-improving AutoML frame-
work that turns an LLM into an online engine for neural net-
work development. From a single prompt, NNGPT emits
a full executable specification, validates and trains it, logs
code and metrics, and improves its generators and predic-
tors from these traces. The framework unifies five pipelines
in one loop: zero-shot model generation and editing, hyper-
parameter recommendation, accuracy and early-stop pre-
diction, retrieval-augmented code synthesis, and reinforce-
ment learning updates.
Experiments show practical effectiveness. Zero-shot
generation added over 5,000 trained models and about 1.9k
unique architectures to LEMUR. NN-RAG reconstructs 941
of 1,289 target blocks (73% executability). One-shot hy-
perparameter prediction reaches 93% schema-valid outputs
and RMSE close to Optuna. The code-aware predictor
achieves RMSE0.1449with Pearsonr= 0.7766across
more than 1,200 runs and supports early termination and
scheduling. The RL loop reliably generates executable
models and attains strong one-epoch accuracy on MNIST
and SVHN.
NNGPT reduces AutoML cost and latency by replac-
ing many-trial search with one-shot, schema-validated gen-
eration while maintaining transparent, reproducible logs.
Code, prompts, and checkpoints are released to support
replication and extension.
8

References
[1] Adriaensen et al. Efficient bayesian learning curve extrapo-
lation using prior-data fitted networks.Advances in Neural
Information Processing Systems, 36:19858–19886, 2023. 2
[2] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru
Ohta, and Masanori Koyama. Optuna: A next-generation
hyperparameter optimization framework. InProceedings of
the 25th ACM SIGKDD International Conference on Knowl-
edge Discovery & Data Mining, page 2623–2631, New York,
NY , USA, 2019. Association for Computing Machinery. 1, 2
[3] Akiba et al. Optuna: A next-generation hyperparameter
optimization framework. InProceedings of the 25th ACM
SIGKDD international conference on knowledge discovery
& data mining, pages 2623–2631, 2019. 7, 1
[4] Amir Bassamzadeh and Samira Methani. Enhancing llm out-
put quality for specialized apis: Fine-tuning versus retrieval.
arXiv preprint arXiv:2401.01234, 2024. 1
[5] James Bergstra, R ´emi Bardenet, Yoshua Bengio, and Bal ´azs
K´egl. Algorithms for hyper-parameter optimization. InPro-
ceedings of the 24th International Conference on Neural In-
formation Processing Systems, page 2546–2554, Red Hook,
NY , USA, 2011. Curran Associates Inc. 1
[6] Bergstra et al. Hyperopt: A python library for optimizing the
hyperparameters of machine learning algorithms.SciPy, 13:
20, 2013. 2, 1
[7] Angelica Chen, David M. Dohan, and David R. So. Evo-
prompting: Language models for code-level neural architec-
ture search.arXiv preprint arXiv:2302.14838, 2023. 2, 1
[8] Mark et al. Chen. Evaluating large language models trained
on code.arXiv preprint arXiv:2107.03374, 2021. 1
[9] Xiangning Chen and Cho-Jui Hsieh. Stabilizing differen-
tiable architecture search via perturbation-based regulariza-
tion. InInternational conference on machine learning, pages
1554–1565. PMLR, 2020. 1
[10] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang,
Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai
Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu
Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao
Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen,
Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo,
Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han
Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian
Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li,
Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu,
Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai
Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai
Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang,
Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua
Zhang, Minghui Tang, Meng Li, Miaojun Wang, Ming-
ming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng
Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang,
Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen,
Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng
Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan,
S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, TaoYun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wan-
jia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu,
Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu,
Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng
Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xi-
aosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song,
Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y . K. Li, Y . Q.
Wang, Y . X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao
Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yi-
fan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang,
Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan
Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yun-
fan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang
Zhou, Y . X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li,
Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun
Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe
Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao,
Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu,
Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan,
Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang.
Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning, 2025. 5, 6
[11] DeepSeek-AIet al. DeepSeek-V3 technical report, 2025. 1
[12] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,
and Li Fei-Fei. Imagenet: A large-scale hierarchical image
database. In2009 IEEE Conference on Computer Vision and
Pattern Recognition, pages 248–255, 2009. 8
[13] Ding et al. Architecture-aware learning curve extrapola-
tion via graph ordinary differential equation.arXiv preprint
arXiv:2412.15554, 2024. 2
[14] Domhan et al. Speeding up automatic hyperparameter opti-
mization of deep neural networks by extrapolation of learn-
ing curves. InIJCAI, pages 3460–3468, 2015. 2
[15] Golovin et al. Google vizier: A service for black-box op-
timization. InProceedings of the 23rd ACM SIGKDD In-
ternational Conference on Knowledge Discovery and Data
Mining (KDD ’17), pages 1487–1495. ACM, 2017. 2, 1
[16] Arash Torabi Goodarzi, Roman Kochnev, Waleed Khalid,
Furui Qin, Tolgay Atinc Uzun, Yashkumar Sanjaybhai
Dhameliya, Yash Kanubhai Kathiriya, Zofia Antonina Ben-
tyn, Dmitry Ignatov, and Radu Timofte. LEMUR Neural
Network Dataset: Towards Seamless AutoML, 2025. 1
[17] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong,
Wentao Zhang, Guanting Chen, Xiao Bi, Y . Wu, Y . K. Li,
Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-
coder: When the large language model meets programming
– the rise of code intelligence, 2024. 6
[18] Guo et al. DeepSeek-Coder: When the large language model
meets programming - the rise of code intelligence, 2024. 1
[19] Agrim Gupta, Piotr Doll ´ar, and Ross Girshick. Lvis: A
dataset for large vocabulary instance segmentation, 2019. 8
[20] He et al. Designing network algorithms via large language
models.arXiv preprint arXiv:2404.01617, 2024. 2
[21] Jeremy Howard. Imagenette: A smaller subset of ImageNet
for quick experimentation.https://github.com/
fastai/imagenette, 2019. 7
9

[22] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
LoRA: Low-Rank Adaptation of Large Language Models,
2021. 2
[23] Hu et al. LoRA: Low-rank adaptation of large language mod-
els. InInternational Conference on Learning Representa-
tions (ICLR 2022), 2022. 1
[24] Haifeng Jin, Qingquan Song, and Xia Hu. Auto-keras: An
efficient neural architecture search system. InProceedings of
the 25th ACM SIGKDD international conference on knowl-
edge discovery & data mining, pages 1946–1956, 2019. 2,
1
[25] Roman Kochnev, Arash Torabi Goodarzi, Zofia Antonina
Bentyn, Dmitry Ignatov, and Radu Timofte. Optuna vs Code
Llama: Are LLMs a New Paradigm for Hyperparameter
Tuning?, 2025. 6, 7
[26] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10
(canadian institute for advanced research). 5, 7, 8
[27] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton.
Imagenet classification with deep convolutional neural net-
works. InAdvances in Neural Information Processing Sys-
tems (NeurIPS), pages 1097–1105, 2012. 8
[28] Yann LeCun, L ´eon Bottou, Yoshua Bengio, and Patrick
Haffner. Gradient-based learning applied to document recog-
nition.Proceedings of the IEEE, 86(11):2278–2324, 1998.
MNIST. 7, 8
[29] Li et al. Competition-level code generation with alphacode.
Science, 378(6624):1092–1097, 2022. 1
[30] LibCST Contributors. Libcst: Concrete syntax tree
parser and toolkit for python.https://libcst.
readthedocs.io/en/latest/, 2025. Accessed:
Nov. 13, 2025. 5
[31] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir
Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva
Ramanan, C. Lawrence Zitnick, and Piotr Doll ´ar. Microsoft
coco: Common objects in context, 2015. 7
[32] Hanxiao Liu, Karen Simonyan, and Yiming Yang.
Darts: Differentiable architecture search.arXiv preprint
arXiv:1806.09055, 2018. 2, 1
[33] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
Deep learning face attributes in the wild. InICCV, 2015.
CelebA dataset. 7
[34] Liu et al. Large language models to enhance bayesian opti-
mization. InThe Twelfth International Conference on Learn-
ing Representations (ICLR 2024), 2024. 2
[35] Liu et al. Large language models to enhance bayesian opti-
mization, 2024. 2
[36] S. Long, Y . Wang, and H. Zhang. Performance prediction
based on neural architecture features.Cognitive Computa-
tion and Systems, 2(2):80–83, 2020. 2
[37] Machida et al. Msr-darts: Minimum stable rank of dif-
ferentiable architecture search. In2022 International Joint
Conference on Neural Networks (IJCNN), pages 1–9. IEEE,
2022. 1
[38] TorchVision maintainers and contributors. ”torchVision: Py-
Torch’s computer vision library”, ”2016”. 8, 5[39] Mangrulkar et al. Peft: State-of-the-art parameter-
efficient fine-tuning methods.https://github.com/
huggingface/peft, 2022. 3
[40] Nasir et al. Llmatic: Neural architecture search via large
language models and quality-diversity optimization. InPro-
ceedings of the Genetic and Evolutionary Computation Con-
ference (GECCO ’24), page 1–9, Melbourne, VIC, Australia,
2024. ACM. 2, 1
[41] Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bis-
sacco, Bo Wu, and Andrew Y . Ng. Reading digits in natural
images with unsupervised feature learning. InNIPS Work-
shop on Deep Learning and Unsupervised Feature Learning,
2011. SVHN. 7, 8
[42] OpenAI et al. GPT-4 technical report, 2024. 1
[43] OpenMMLab Contributors. Openmmlab: Open-source com-
puter vision toolbox ecosystem.https://openmmlab.
com/, 2023. 8
[44] Paszke et al. PyTorch: An imperative style, high-
performance deep learning library, 2019. 1, 3
[45] Peng et al. The impact of AI on developer productivity: Ev-
idence from GitHub Copilot, 2023. 1
[46] Pham et al. Efficient neural architecture search via parame-
ters sharing. InInternational conference on machine learn-
ing, pages 4095–4104. PMLR, 2018. 2, 1
[47] Baptiste Rozi `ere, Jonas Gehring, Fabian Gloeckle, Sten
Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu
Liu, Romain Sauvestre, Tal Remez, J ´er´emy Rapin, Artyom
Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt,
Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong,
Alexandre D ´efossez, Jade Copet, Faisal Azhar, Hugo Tou-
vron, Louis Martin, Nicolas Usunier, Thomas Scialom, and
Gabriel Synnaeve. Code Llama: Open Foundation Models
for Code, 2024. 1, 6
[48] Rozi `ere et. al. Code Llama: Open foundation models for
code.arXiv preprint arXiv:2308.12950, 2024. 1
[49] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao
Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y . K. Li,
Y . Wu, and Daya Guo. Deepseekmath: Pushing the limits of
mathematical reasoning in open language models, 2024. 6
[50] Alex Sheng and Shankar Padmanabhan. Self-programming
artificial intelligence using code-generating language mod-
els.arXiv preprint arXiv:2205.00167, 2022. 2, 1
[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia
Polosukhin. Attention is all you need, 2023. 5, 7
[52] Wang et al. Grammar prompting for domain-specific lan-
guage generation with large language models. InAdvances
in Neural Information Processing Systems, 2023. 2, 1
[53] White et al. How powerful are performance predictors in
neural architecture search? InNeurIPS, pages 28454–28469,
2021. 2
[54] Ross Wightman. Pytorch image models (timm).https:
/ / github . com / rwightman / pytorch - image -
models, 2019. 8
[55] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chau-
mond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim
Rault, R ´emi Louf, Morgan Funtowicz, Joe Davison, Sam
10

Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien
Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama
Drame, Quentin Lhoest, and Alexander M. Rush. Hugging-
Face’s Transformers: State-of-the-art Natural Language Pro-
cessing, 2020. 8, 3
[56] Yu et al. GPT-NAS: Generative pre-trained models for neu-
ral architecture search.arXiv preprint arXiv:2310.12345,
2023. 2, 1
[57] Yu, Xi and others. Cyclic differentiable architecture search.
IEEE transactions on pattern analysis and machine intelli-
gence, 45(1):211–228, 2022. 1
[58] Li Zhang, Ruizhi Chen, and Pengfei Yu. Llm-assisted hyper-
parameter tuning: A prompt-based approach.arXiv preprint
arXiv:2308.04214, 2023. 2
[59] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva,
and Antonio Torralba. Places: A 10 million image database
for scene recognition.IEEE Transactions on Pattern Anal-
ysis and Machine Intelligence, 40(6):1452–1464, 2018.
Places365. 7
11

NNGPT: Rethinking AutoML with Large Language Models
Supplementary Material
5. Demo Video: TuneNNGen in Action
We include a time-lapse screen recording that demonstrates
the end-to-end workflow ofTuneNNGen.py: prompt as-
sembly, one-shot YAML/code generation, schema valida-
tion, training and logging, and LoRA updates. It also
shows typical failure modes (e.g., schema violations, miss-
ing functions) and how NNGPT surfaces them through the
validator and executor stack. Refer to the supplementary
video for the full run, including intermediate console out-
puts and per-model evaluation messages.
6. Extended Related Work and Detailed Com-
parisons
Automated neural network configuration has long been a
focus of research, with prior work falling broadly into two
categories: classical search-based optimization and recent
LLM-driven generative approaches. Below we review key
methods from both domains, focusing on the limitations ad-
dressed by NNGPT.
Search-Based Hyperparameter Optimization.Frame-
works like Optuna [2], Hyperopt [6], and Google Vizier [15]
use Bayesian optimization and Tree-structured Parzen Es-
timators (TPE) [5] to iteratively tune hyperparameters.
Though effective, these methods demand significant com-
pute, often requiring hundreds of training runs, and act as
black-box optimizers, offering limited transparency into the
configuration semantics or model behavior beyond their fi-
nal scores.
AutoML and Neural Architecture Search (NAS).Au-
toML systems such as Optuna [3] and Hyperopt treat tun-
ing as black-box optimization, while NAS frameworks like
ENAS [46], AutoKeras [24], and DARTS [32] (plus variants
like CDARTS, SDARTS, iDARTS) expand this to architec-
ture generation via reinforcement learning or differentiable
search. These methods, however, remain resource-intensive
and adapt slowly to new tasks.
Efforts to stabilize NAS, such as RC-DARTS [57],
NDARTS [9], and MSR-DARTS [37], still rely on costly
infrastructure. NNGPT bypasses iterative search entirely: a
single LLM call produces a complete, schema-valid config-
uration that is directly executable and logged for traceabil-
ity. This design reduces computational cost while maintain-
ing competitive accuracy, as shown by our DeepSeek-based
results.
LLMs for Code and Configuration Generation.Re-
cent advances in code-oriented LLMs such as Codex [8],
Code Llama [48], and DeepSeek-Coder [18] enable au-
tomatic generation of hyperparameters, architectures, andtraining scripts. Tools like GitHub Copilot[45] and Alpha-
Code [29] further highlight their utility for programming
tasks.
Most existing systems, however, lack integration with
execution pipelines. Outputs are rarely validated empiri-
cally or tested for reproducibility. EvoPrompting [7] em-
beds LLMs into evolutionary search but requires repeated
inference and lacks schema control.
NNGPT employs a one-shot generation strategy: a fine-
tuned LLM produces a schema-compliant YAML config-
uration that is immediately validated and executed. Struc-
tured output enforcement (via Pydantic) and empirical feed-
back support reliable AutoML automation.
Prediction of Neural Network Accuracy.Despite ad-
vances in LLMs, many methods rely on high-level descrip-
tions that miss implementation-specific patterns, or predict
isolated metrics instead of full training dynamics. Yet per-
formance prediction is vital for tuning efficiency and re-
source use in AutoML.
We address this via a multi-modal predictor that pro-
cesses both structured metadata (e.g., task, dataset) and ex-
ecutable code (e.g.,nn code,metric code). Key con-
tributions include: (1) joint modeling of code and metadata,
(2) efficient 4-bit QLoRA fine-tuning [23] to predict accu-
racy and stopping points, and (3) stratified data splits that
prevent leakage while preserving balance.
LLM-Driven Neural Network Generation.LLMs are
increasingly integrated into NAS pipelines. EvoPrompt-
ing uses a pretrained LLM as a mutation operator, requir-
ing multiple inference calls and soft prompt-tuning [7].
LLMatic replaces hand-written mutations with CodeGen-
generated code [40]. GPT-NAS [56] completes partial ar-
chitectures, and Self-Programming AI [50] revises its own
configs.
While effective, these methods rely on expensive itera-
tive queries. In contrast, NN-GPT performs one-shot gener-
ation: a single LoRA-tuned LLM call yields an executable,
schema-compliant configuration with no further iteration
required.
Schema-Constrained Generation.LLMs frequently
generate syntactically invalid code in structured tasks.
Grammar Prompting mitigates this by embedding formal
grammars (e.g., Backus–Naur Form) into prompts [52],
while RAG methods reduce errors by referencing external
examples [4]. NN-GPT adopts a different approach: it em-
ploys a strict Pydantic schema tailored to YAML configura-
tions. Minor issues are auto-corrected; major ones trigger
a single re-prompt, ensuring valid, executable outputs with-
out relying on grammars or retrieval.
1

LLMs for Pipeline Optimization.LLMs have
also been explored for broader pipeline automation.
LLAMBO [34] frames Bayesian optimization as an LLM-
driven dialogue, where the model proposes new candi-
dates based on previous evaluations, outperforming classi-
cal BO in sparse regimes. Prompt-based HPO uses GPT-4
to refine hyperparameters from task descriptions[58], and
NADA[20] generates full control algorithms, filtering them
through simulation. In contrast, NN-GPT focuses on end-
to-end generation: a single LLM call outputs complete,
runnable training recipes — data loaders, models, and opti-
mization logic, without requiring iterative refinement.
Training Performance Prediction.Efforts to predict
training outcomes fall into three main groups, each with key
limitations. Statistical curve fitting [1, 14] extrapolates final
metrics from early trends but ignores architecture and im-
plementation, reducing generalizability. These approaches
treat training as a numeric process, overlooking structural
factors that influence convergence behavior.
Architecture-Based Performance Predictors.Methods
like [36, 53] use textual summaries or engineered features
to estimate model performance. However, such abstractions
overlook critical implementation-level details, limiting their
ability to capture the nuanced behaviors encoded in exe-
cutable code.
Graph-Based Approaches.Techniques such as [13] rep-
resent models via explicit computational graphs, sometimes
enhanced with differential modeling. Although more ex-
pressive than text, these methods require costly manual
graph construction and still miss fine-grained code-level
patterns such as custom regularization or dynamic schedul-
ing, which strongly affect training outcomes.
Multimodal Prediction via Executable Code.While
prior approaches either discard architectural context [1, 14]
or rely on lossy text abstractions [36, 53], our method
directly processes executable code snippets (nn code,
metric code,transform code) within the LLM
prompt. This enables the model to learn implementation-
specific patterns, e.g., custom regularization, branching, or
metric logic, that strongly influence convergence but are in-
accessible via abstract representations.
In contrast to graph-based methods [13] requiring man-
ual topology construction, we combine three complemen-
tary inputs: (i) structured metadata, (ii) raw executable
code, and (iii) early training dynamics. This multimodal
context allows the model to align syntactic code patterns
with numerical trends, revealing complex interactions that
are missed by unimodal models.
Moreover, our LLM jointly predicts final accuracy and
optimal stopping points within a shared latent space, cap-
turing their intrinsic correlation — unlike prior works that
treat them separately. This unified structure enhances gen-
eralization across tasks and architectures.Table 1 situates NNGPT among classical AutoML toolk-
its and recent LLM-driven approaches. Bayesian HPO and
NAS frameworks provide strong baselines for search over
hyperparameters and architectures but rely on iterative eval-
uation and do not synthesize full training programs. Recent
LLM-based methods introduce generation and closed-loop
ideas, yet typically cover only parts of the pipeline or lack
robust execution and prediction components. In contrast,
NNGPT combines one-shot generation of executable spec-
ifications, code-aware accuracy prediction, and a closed
feedback loop inside a single open-source framework.
7. Implementation
The NNGPT framework is implemented in pure Python and
provided as a pip-installable package. It requires Python
3.10 or higher and CUDA for GPU-based training. The
core components are accessible via top-level command-
line interfaces such asNNAlter *.py,NNEval.py,
TuneNNGen *.py,TuneAccPrediction.py,
TuneHyperparameters.pyorTuneRAG.py. These
tools together compose the full pipeline introduced in
Section 8.
7.1. Repository Structure
The codebase is organized into modular subdirectories.
Theab/gptdirectory contains the main scripts. Prompt
templates, training configurations, and YAML schema
definitions are defined inab/gpt/conf, while LoRA
fine-tuning utilities and other helper functions reside in
ab/gpt/util.
7.2. Command-Line Interface
The scriptNNAlter.pyperforms seed model alteration
using architectures from the LEMUR corpus. Each modi-
fied model is trained for a small number of epochs (default:
8, configurable via the-eflag). This utility relies on the
ab.gpt.util.AlterNN.altermethod and defaults
to the deepseek/ai/DeepSeek-R1-Distill-Qwen-7B model.
TheTuneNNGen *.pyscripts handle one-shot LLM-
based generation, YAML schema validation, full model
training, and optional LoRA fine-tuning. A fast iteration
mode can be enabled by passing the-sflag, which skips
the alteration phase.
Additionally,NNEval.pyoffers standalone evaluation
of generated architectures without invoking LoRA, en-
abling targeted testing of LLM outputs.
All required CUDA and MPI dependencies are pre-
packaged in the public Docker image.
7.3. Prompting and Inference Flow
Prompt assembly begins with encoding task specifications,
dataset, metric, and compute constraints into a structured
JSON block. This prompt is then transformed into strict
2

YAML by the LLM. The DeepSeek 7B model is loaded
viatransformers.AutoModelForCausalLMin 8-
bit precision, with LoRA adapters applied as needed.
7.4. Distributed Training Back-End
Model training is executed usingPyTorch[44] and
torch.distributed. MPI support is configured at in-
stall time using system-level dependencies. During training,
per-epoch statistics such as loss, accuracy, mAP/IoU, GPU
utilization, and Git SHA are recorded in a local SQLite
database via SQLAlchemy. If an optional statistics mod-
ule is available, the database can be exported to Excel-
compatible dashboards using a single API call.
7.5. LoRA Fine-Tuning Loop
LoRA adapters are fine-tuned after every 50 new model
training runs. The fine-tuning procedure consists of three
epochs using a linear warm-up and cosine decay sched-
ule. The resulting adapter checkpoints are published to the
framework’s Hugging Face namespace, making them avail-
able for the next generation cycle.
7.6. Consistent Dependencies
NNGPT ensures consistency and reproducibility
through a pre-built Docker image and a pinned
requirements.txtfile that specifies fixed versions of
key libraries such astorch[44],transformers[55],
andpeft[39].
7.7. Prompt Generation Pipeline
Prompt generation is template-driven. A JSON file defines
the structure of the prompt, including optional insertion
points for code samples. This enables dynamic construc-
tion of prompts containing one or two models sampled from
the database, under a given task. For training data genera-
tion, prompts may be further constrained to ensure shared or
divergent characteristics between samples, improving task
alignment while maintaining flexibility.
7.8. Offline Initialization of the Training Set
During early experiments, the raw DeepSeek-Coder-1.3B
model frequently refused to modify input code, either
claiming sufficiency or producing rule-violating changes.
Introducing stricter task constraints often led to outputs that
ignored parts of the prompt or simply returned unmodified
code. These observations motivated the need for a boot-
strapped training set.
To address this, the larger DeepSeek-R1-Distill-Qwen-
7B model was used to generate initial examples. During
20 offline epochs, this model produced modified architec-
tures by applying simple transformations, such as changing
channel widths or digits inConv2dlayers, without requir-
ing accuracy improvements. The first prompt template isshown in Listing 1. To diversify the training set, additional
prompts requested the model to modify 2 to 4 digits ran-
domly. Since DeepSeek-R1 7B reliably introduced at least
one valid change, its outputs served as the initial training
corpus for fine-tuning the smaller model.
During prompt engineering, both one-shot and few-
shot examples were evaluated. However, neither signifi-
cantly improved task fidelity on the raw models. In fact,
DeepSeek-Coder-1.3B and DeepSeek-R1-Distill-Qwen-7B
often lost track of task requirements or confused multiple
examples. Zero-shot prompting consistently yielded more
predictable behavior and was therefore adopted.
Define two ‘nn.Conv2d()‘ layers as a neighboring
pair
when there are only non-Conv2d layers between
them.
Your task:
- Randomly pick and modify 1 or 2
neighboring pairs
of ‘nn.Conv2d()‘ layers based on the
rules below.
- Return ONLY the full result.
Instructions:
1. For each picked neighboring pair:
- Change the out-channels of the
former ‘nn.Conv2d()‘
and the in-channels of the latter to
the same new integer.
2. Modify the numbers directly in the code.
DO NOT introduce new methods or
parameters.
3. Do NOT pick the first neighboring pair;
select randomly.
4. You MUST return the FULL CODE, including
all classes - even if unmodified.
5. If no such pair exists:
- Randomly change 2 digits in the code
to new values.
6. Do not make any changes beyond the rules
above.
Here is the input code for you to modify:
’’’\n{nn_code}\n’’’
Listing 1. Prompt instructing the LLM to modify neighboring
nn.Conv2d() layers under strict rules, ensuring in-place edits and
structural preservation of PyTorch code.
7.9. Automated LEMUR Dataset Extension
The extended dataset was generated using prompt templates
similar to those used during fine-tuning (e.g., Listing 2).
Generated samples were stored alongside their originating
code and task context to ensure proper evaluation. Cer-
tain datasets such as COCO required specific preprocessing
(e.g., fixedResizeoperations), and segmentation tasks
mandated the use of IoU rather than accuracy.
1. Change the layers in the following
implementation to improve accuracy on image
classification tasks.
3

2. Strictly follow these constraints:
- Modify ONLY the layers in the code.
- DO NOT introduce new methods or
parameters.
3. Your response MUST:
- Include the full code (including
unchanged classes or functions).
- Indicate the modified class explicitly.
- Actually make at least one layer
modification.
- Not worry about decreasing accuracy - it’
s acceptable
Input code: ’’’\n{nn_code}\n’’’
Listing 2. Prompt directing the LLM to modify existing layers in a
neural network implementation under strict constraints, requiring
full-code output and explicit identification of edits.
To extract generated code, regular expressions were used
to match code blocks wrapped in triple backticks. To
avoid infinite loops in parsing, the number of backticks was
counted before applying regex-based extraction. Evaluation
was performed via the LEMUR API. If parsing or execution
errors occurred, the failing code and exception trace were
returned to the LLM for repair. Each sample was given up
to two correction attempts. Validated models were saved in
structured directories compatible with the LEMUR dataset
format and integrated into the local repository for down-
stream use.
7.10. Fine-Tuning for Accuracy Prediction
The model Deepseek-coder-1.3b-Instruct was fine-tuned
via supervised learning to enable the prediction of valida-
tion accuracy changes resulting from architectural modifi-
cations. To align with the offline generation procedure de-
scribed earlier, the fine-tuning objective was formulated as
a code-to-code transformation task, where the model is pre-
sented with an original implementation, its evaluated accu-
racy, and the accuracy of a known improved variant. The ex-
pected output is the modified architecture that corresponds
to the target accuracy, drawn from the LEMUR dataset.
The prompt, shown in Listing 3, encodes this training
setup explicitly. It instructs the model to alter only layer
definitions in order to achieve the desired accuracy gain,
without introducing new methods or structural components.
The fields{accuracy}and{addon accuracy}repre-
sent the original and improved accuracy, respectively, and
{nncode}contains the complete source code of the base
model. The expected output is the full code of the improved
model, including unchanged components, so the model also
learns to preserve contextually irrelevant structures.
1. Change the layers in the following
implementation to improve accuracy from {
accuracy} to {addon_accuracy} on image
classification tasks.2. Strictly modify ONLY the layers in the code
and STRICTLY DO NOT INTRODUCE NEW METHODS OR
PARAMETERS
3. You should answer not only the class you
modified but with FULL CODES INCLUDING
EVERYTHING (Other classes, etc.) NOT CHANGED
4. You have to reply will full codes, don’t be
afraid of actually reducing the accuracy.
Strictly check and make sure at least one layer
being modified.
Listing 3. Prompt template used to generate training data by
requesting constrained layer modifications aimed at improving
model performance.
After eight epochs of supervised fine-tuning, the model
consistently produced syntactically valid outputs that ad-
hered to the required formatting and structural constraints.
It successfully learned to apply targeted architectural modi-
fications that align with observed changes in validation ac-
curacy, and reliably preserved all components mandated by
the LEMUR API specification.
In addition to replicating transformations observed dur-
ing training, the model demonstrated the ability to com-
pose valid variants across different architecture families.
It consistently selected effective transformation strategies
for given task-model pairs, producing high-quality outputs
with minimal prompt engineering. This behavior reflects an
emerging structural prior that favors stable and empirically
grounded edits.
Moreover, the model preserved functional integrity even
in cases where only subtle changes were required, main-
taining coherence across unchanged code regions. These
results confirm that the fine-tuned model captures not only
the mapping between code edits and accuracy shifts, but
also the appropriate output syntax and modular consistency
necessary for downstream execution and evaluation.
7.11. Fine-Tuning for Hyperparameter Generation
In addition to accuracy prediction, NNGPT also supports
automated generation of numeric hyperparameter values
tailored to a given neural network, task, and dataset. To
enable this capability, we fine-tuned a diverse collection of
DeepSeek models using supervised learning on entries from
the LEMUR dataset. The selected set spans a wide range
of model scales and pretraining strategies, including both
lightweight and instruction-tuned architectures. This diver-
sity allowed us to examine how model size and initialization
influence hyperparameter reasoning ability.
Each model was fine-tuned using the LoRA method to
reduce memory overhead while preserving gradient flow
across transformer layers. LoRA rank and alpha were se-
lected from 16, 32, 64, with a dropout of 0.05. All ex-
periments employed a consistent setup that included the
AdamW optimizer, a cosine learning rate scheduler, a fixed
learning rate of3×10−5, gradient accumulation, check-
4

pointing, and a training duration of up to 35 epochs depend-
ing on convergence behavior.
Training prompts were constructed to align with the
LEMUR schema and encode structured conditioning vari-
ables: architecture code, target accuracy, number of training
epochs, and required transformations. The input template,
shown in Listing 4, requests the LLM to predict the values
of specific hyperparameters that would lead the given model
to achieve a known accuracy level.
### Input:
"Generate only the values (do not provide
any explanation) of the hyperparameters
({prm_names}) of a given model:
{entry[’nn’]} for the task:
{entry[’task’]} on dataset:
{entry[’dataset’]}, with transformation:
{entry[’transform_code’]}, so that the
model achieves accuracy =
{entry[’accuracy’]} with number of
training epochs = {entry[’epoch’]}. Code
of that model: {entry[’nn_code’]}"
### Response:
"Here are the hyperparameter values for
which the model will achieve the
specified accuracy: {prm_values}."
Listing 4. Prompt template used for fine-tuning LLMs to generate
hyperparameter values based on model code, task, dataset,
transformations, target accuracy, and training duration.
During evaluation, a modified version of the prompt
(Listing 5) asked the model to produce hyperparameters
aimed at maximizing accuracy, thereby testing generaliza-
tion to unseen combinations.
### Input:
"Generate only the values (do not provide any
explanation) of the hyperparameters
({prm_names}) of a given model:
{entry[’nn’]} for the task: {entry[’task’]}
on dataset: {entry[’dataset’]}, with
transformation: {entry[’transform_code’]},
so that the model achieves the HIGHEST
accuracy with number of training epochs =
{entry[’epoch’]}. Code of that model:
{entry[’nn_code’]}"
Listing 5. Prompt for generating hyperparameter values that
maximize accuracy, conditioned on model code, task, dataset,
transformation logic, and training epochs.
All fine-tuned models were subsequently integrated
into the NNGPT pipeline. Specifically, two of the best
performing checkpoints, DeepSeek-Coder-1.3b-Base and
DeepSeek-R1-Distill-Qwen-7B, were published to Hug-
ging Face and are now directly accessible within the frame-
work. These models can be used as plug-and-play com-
ponents for automated hyperparameter suggestion in both
evaluation and generation modes.
By extending the NNGPT workflow with fine-tuned
LLMs for hyperparameter prediction, we close the loop be-tween architecture generation and training configuration,
offering a unified and fully automated AutoML solution.
Experimental results evaluating the quality and effective-
ness of these generated hyperparameters are presented in
Section 3.2.
8. Methodology
NNGPT reconceptualizes LLMs as fully automated config-
uration agents for neural networks. Given a high-level task
specification, the system follows the eight-stage pipeline
depicted in Figure 1, encompassing architecture retrieval,
prompt generation, model evaluation, and iterative self-
improvement. Below, we detail each stage.
8.1. Stage 1: LEMUR API
NNGPT starts by querying the LEMUR Dataset API for ex-
ecutable architectures and metadata with standardized train-
ing and evaluation. LEMUR provides full implementations,
uniform preprocessing, and reproducible metrics, so one-
shot generation targets runnable code and uses LEMUR as
a reliable metric oracle. Unlike torchvision [38], it adds re-
producible benchmarking, native metric logging, and direct
access to architecture code, making it well suited for gener-
ative AutoML.
8.2. Stage 2: Neural Network Data Retrieval
Following the API query, candidate architectures and asso-
ciated metadata are retrieved from the LEMUR corpus. A
dedicated script,NNAlter *.py, then applies a controlled
set of architectural perturbations - including insertion/re-
moval of layers, changes to channel widths, and alterations
to residual connections. Each altered model undergoes a
lightweight convergence check via a short warm-up training
run (with the-eflag), ensuring that the resulting design re-
mains viable. The resulting population of diverse, training-
validated models is archived in an SQLite-backed reposi-
tory, which serves as training data for LLM fine-tuning and
architecture prediction.
8.3. Stage 3: Configuration Setup
After retrieval, a configuration scaffold is constructed that
encapsulates the task, dataset, desired metric, and (option-
ally) a compute budget. This configuration is used to pa-
rameterize the prompt generator, enabling reproducible and
deterministic experiment assembly. The prompt generation
code is modular and extensible, supporting dynamic recon-
figuration and alternate LLM backbones through Hugging
Face-compatible interfaces.
8.4. Stage 4: Prompt Assembly & Injection
The core input to the LLM is a structured prompt, assem-
bled from the configuration data. This includes a JSON ob-
ject describing the task, dataset, metric, and resource con-
5

straints. The prompt is formatted with system-level instruc-
tions that force the model to emit a strict YAML block be-
ginning with-yamland following a predefined schema.
The generative pass itself is carried out using a
DeepSeek-V2 16B model equipped with 8-bit LoRA
adapters fine-tuned on the altered-model corpus. Sampling
is configured usingT= 0.2,p= 0.9, and top-1 selec-
tion to minimize unnecessary exploration. Output is parsed
withruamel.yamland validated usingpydantic. Mi-
nor schema violations are auto-corrected, while major is-
sues trigger a re-prompt with an embedded diagnostic mes-
sage. Only configurations that pass validation proceed to
execution.
8.5. Stage 5: LLM-Guided Architecture Genera-
tion
Once validated, the YAML output is interpreted as an exper-
iment specification that includes a full architecture graph,
optimizer parameters, loss functions, and data transforma-
tions. This file serves as the input to the training engine.
Generated architectures may differ from their seed counter-
parts not only in topology but also in auxiliary properties
such as normalization strategies or metric implementations.
This stage integrates architectural creativity with empirical
feasibility, ensuring that every configuration is both novel
and executable.
8.6. Stage 6: Model Validation and Evaluation
Training is conducted using a distributed PyTorch en-
gine (ab/gpt/TuneNNGen *.py), which usesmpi4py
andtorch.distributedfor multi-GPU support. All
datasets are accessed through the LEMUR API to ensure
consistent preprocessing and evaluation. During training,
metrics such as loss, accuracy, mAP, and IoU are logged per
epoch alongside GPU utilization and configuration hashes.
These logs are persisted in the same SQLite database as the
architecture graph, enabling traceability and post-hoc anal-
ysis via an internal visualization suite designed for model
benchmarking.
In parallel, a lightweight accuracy prediction module is
used to estimate final model performance based on early
training signals and static features. The predictor processes
three input modalities: (i) structured metadata, (ii) exe-
cutable code (model, transform, metric), and (iii) early ac-
curacy values. A stratified split strategy ensures robust gen-
eralization and prevents data leakage across similar archi-
tectures.
8.7. Stage 7: LoRA Fine-tuning
Afterk= 50successful training runs, a new fine-tuning
iteration is triggered. LoRA adapters are retrained on the
extended dataset of architecture–performance pairs using
three epochs of linear warm-up and cosine decay. This up-dates the LLM’s internal representation of the architecture
space, biasing future generations toward successful config-
urations. The updated checkpoints are immediately used in
subsequent pipeline passes.
8.8. Stage 8: Database Storage
All artifacts, including trained weights, YAML specifi-
cations, metric logs, and architectural source code, are
stored in a structured SQLite-based repository designed for
long-term reproducibility. This supports future reanalysis,
dataset expansion, and model distillation. The continu-
ous accumulation of structured training data reinforces the
NNGPT pipeline, transforming raw LLM generations into
a curated and executable AutoML knowledge base.
8.9. AutoML Loop
Once all eight stages complete, the pipeline automatically
restarts using updated model checkpoints, prompt tem-
plates, and database entries. This iterative process enables
NNGPT to function as a self-improving AutoML cycle,
where each generation, validation, and fine-tuning pass en-
hances the quality and diversity of future outputs.
6