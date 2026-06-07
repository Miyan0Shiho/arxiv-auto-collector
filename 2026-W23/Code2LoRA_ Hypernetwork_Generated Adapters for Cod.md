# Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution

**Authors**: Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie

**Published**: 2026-06-04 17:59:46

**PDF URL**: [https://arxiv.org/pdf/2606.06492v1](https://arxiv.org/pdf/2606.06492v1)

## Abstract
Code language models need repository-level context to resolve imports, APIs, and project conventions. Existing methods inject this knowledge as long inputs (retrieved through RAG or dependency analysis) or through per-repository fine-tuning and LoRA -- costly at repository scale and brittle to evolving codebases. We introduce Code2LoRA, a hypernetwork framework that generates repository-specific LoRA adapters, effectively injecting repository knowledge with zero inference-time token overhead. Code2LoRA supports two usage scenarios: Code2LoRA-Static converts a single repository snapshot into an adapter, suitable for comprehension of stable codebases; while Code2LoRA-Evo maintains an adapter backed by a GRU hidden state updated per code diff, suitable for active development of evolving codebases. To evaluate Code2LoRA against parameter-efficient fine-tuning baselines, we build RepoPeftBench, a benchmark of 604 Python repositories with two tracks: a static track with 40K training and 12K test assertion-completion tasks, and an evolution track with 215K commit-derived training and 87K commit-derived test tasks. On the static track, Code2LoRA-Static achieves 63.8% cross-repo and 66.2% in-repo exact match, matching the per-repository LoRA upper bound; on the evolution track, Code2LoRA-Evo achieves 60.3% cross-repo exact match (+5.2 pp over a single shared LoRA). Code2LoRA's code can be found at https://anonymous.4open.science/r/code2lora-6857; the model checkpoints and RepoPeftBench datasets can be found at https://huggingface.co/code2lora.

## Full Text


<!-- PDF content starts -->

Code2LoRA: Hypernetwork-Generated Adapters for Code Language
Models under Software Evolution
Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie
University of Waterloo
{lhotsko, yinxi.li, yuntian, pynie}@uwaterloo.ca
Abstract
Code language models need repository-level
context to resolve imports, APIs, and project
conventions. Existing methods inject this
knowledge as long inputs (retrieved through
RAG or dependency analysis) or through per-
repository fine-tuning and LoRA—costly at
repository scale and brittle to evolving code-
bases. We introduce Code2LoRA, a hyper-
network framework that generates repository-
specific LoRA adapters, effectively injecting
repository knowledge with zero inference-time
token overhead. Code2LoRA supports two
usage scenarios: Code2LoRA-Static converts
a single repository snapshot into an adapter,
suitable for comprehension of stable code-
bases; while Code2LoRA-Evo maintains an
adapter backed by a GRU hidden state up-
dated per code diff, suitable for active devel-
opment of evolving codebases. To evaluate
Code2LoRA against parameter-efficient fine-
tuning baselines, we build RepoPeftBench, a
benchmark of 604 Python repositories with two
tracks: a static track with 40K training and
12K test assertion-completion tasks, and an evo-
lution track with 215K commit-derived train-
ing and 87K commit-derived test tasks. On
the static track, Code2LoRA-Static achieves
63.8% cross-repo and 66.2% in-repo exact
match, matching the per-repository LoRA up-
per bound; on the evolution track, Code2LoRA-
Evo achieves 60.3% cross-repo exact match
(+5.2 pp over a single shared LoRA).1
1 Introduction
Real codebases span thousands of files whose im-
ports, APIs, and conventions a code language
model must know to complete assertions, fix bugs,
or navigate a project. Today’s LLM-based cod-
ing assistants typically inject this repository knowl-
1Code2LoRA’s code can be found at https:
//anonymous.4open.science/r/code2lora-6857 ; the
model checkpoints and RepoPeftBench datasets can be found
athttps://huggingface.co/code2lora.edge as long inputs, in the form of retrieved rele-
vant files through RAG (retrieval-augmented gen-
eration) or dependency analysis, and pay for the
retrieved context at every query. This is costly
because repository-level context can be massive,
stressing the LLM’s context window and RAG’s re-
trieval capability. Another approach is to fine-tune
the model or LoRA adapters (Hu et al., 2022) for
one repository or a group of related repositories,
pushing knowledge into parameters. These meth-
ods also require costly training, and even worse, are
brittle toevolvingcodebases, where every commit
can invalidate the adapter and require retraining.
Recent work on hypernetwork-generated LoRA
adapters (Ha et al., 2017; Charakorn et al., 2025,
2026) is promising: a single forward pass over
a conditioning input produces task- or document-
specific weights for a frozen LLM. These methods,
however, are built for short natural-language task
descriptions or single documents, not the long con-
text a repository typically carries, and they assume
a static conditioning input with no mechanism for
tracking a codebase as it evolves.
We propose Code2LoRA, a hypernetwork frame-
work that generates repository-specific LoRA
adapters, effectively injecting repository knowl-
edge with zero inference-time token overhead.
We design around two orthogonal axes—how
knowledge enters the parameters andwhenit is
updated—and instantiate them as two usage sce-
narios: Code2LoRA-Static converts a single repos-
itory snapshot into an adapter, suitable for com-
prehension of stable codebases; Code2LoRA-Evo
maintains an adapter backed by a GRU hidden state
updated per code diff, so the recurrence augments
(rather than replaces) the snapshot prior, suitable
for active development of evolving codebases.
We evaluate Code2LoRA on RepoPeftBench,
a benchmark of 604 Python repositories (512 in-
distribution and a 92-repository temporal holdout
created after the scrape cutoff). RepoPeftBench di-
1arXiv:2606.06492v1  [cs.SE]  4 Jun 2026

vides each repository into non-test and test portions:
the model may use non-test code as repository con-
text and must complete assertion-completion tasks
in the test portion, which is a task that requires
complex reasoning capabilities (Jain et al., 2025).
Two tracks instantiate our usage scenarios: a static
track with 39,612 training and 11,636 test tasks on
a single repository snapshot, and an evolution track
with 215,129 training and 86,793 test tasks drawn
from commit history. Evaluation uses in-repo (IR)
and cross-repo (CR) splits on the in-distribution
corpus, plus a temporal out-of-distribution (OOD)
test split on the post-cutoff holdout (§6.3).
On the static track, Code2LoRA-Static achieves
63.8% cross-repo exact match, well above context-
injection methods such as RAG and dependency-
resolved context; without any per-repository train-
ing it also reaches 66.2% in-repo exact match,
matching the per-repository LoRA upper bound.
On the evolution track, snapshot-based adaptation
goes stale once evaluation uses commit-derived
tasks; Code2LoRA-Evo reaches 60.3% cross-repo
exact match, +5.2 pp over a shared LoRA. On
the temporal OOD holdout, Code2LoRA-Evo re-
mains the strongest method under the same commit-
derived protocol (§6.3).
The main contributions of this work include:
•Idea.We propose using hypernetworks to effec-
tively inject repository knowledge into code lan-
guage models, and frame the problem alonghow
knowledge enters model parameters andwhenit
is refreshed.
•Framework.We design and implement
Code2LoRA, a hypernetwork that maps reposi-
tory code to LoRA adapters with zero inference-
time token overhead, instantiated as Code2LoRA-
Static (mapping one repository snapshot) and
Code2LoRA-Evo (maintaining an adapter from
sequential code diffs).
•Benchmark.We curate RepoPeftBench, a
benchmark of 604 Python repositories, includ-
ing a 92-repository temporal holdout for out-of-
distribution evaluation.
•Evaluation.Code2LoRA outperforms the
strongest baselines on RepoPeftBench by +9.9 pp
on the static track and +5.2 pp on the evolution
track, with consistent gains on the temporal OOD
holdout (§6.3).2 Related Work
Parameter-efficient fine-tuning.LoRA (Hu
et al., 2022) enables efficient adaptation through
low-rank decomposition of weight updates; ex-
tensions include QLoRA (Dettmers et al., 2023),
DoRA (Liu et al., 2024a), weight merging (Yadav
et al., 2023), multi-LoRA routing (Huang et al.,
2024), LoRACode (Chaturvedi et al., 2025), and
MoLE (Zong et al., 2025), which trains a separate
LoRA module per programming language. These
methods treat adapters as static artifacts, trained per
task, per language, or per repository; Code2LoRA
insteadgeneratesadapters conditioned on reposi-
tory context, enabling adaptation to unseen code-
bases without retraining.
Hypernetworks for LoRA generation.Hyper-
networks (Ha et al., 2017) generate the param-
eters of a target network from a conditioning
signal. Recent applications to language mod-
els include HyperTuning (Phang et al., 2023)
and HyperLoRA (Lv et al., 2024) for cross-task
generalization, Generative Adapter (Chen et al.,
2025) for single-pass contextualization, and Zhy-
per (Abdalla et al., 2025) for factorized condi-
tioned LoRA generation. Closest to our frame-
work are Text2LoRA (Charakorn et al., 2025) and
Doc2LoRA (Charakorn et al., 2026), which both
map a whole input (a task description and a doc-
ument, respectively) to a LoRA in one forward
pass. Text2LoRA conditions on a short task de-
scription via an external text encoder and targets
only Q/V projections; Doc2LoRA conditions on
a document via per-layer activations of the frozen
target LLM (Perceiver (Jaegle et al., 2021) encoder,
MLP down_proj only) and is built for document
QA, not code. Code2LoRA-Static generalizes this
family to a third input modality—an entire code
repository—and to full target coverage (all seven
attention and MLP projections rather than Q/V
or down-projection only). To isolate the LoRA-
generation head from confounds, we strengthen
Text2LoRA along both axes: we feed it the same
whole-repository embedding Code2LoRA-Static
consumes, and we emit LoRAs for the same
seven projection types per layer. The strength-
ened Text2LoRA still underperforms Code2LoRA-
Static, pinning down the head as the bottleneck for
repository-level adaptation. Code2LoRA-Evo adds
a second hypernetwork design: a GRU aggregates
sequential code diffs into a hidden state that condi-
tions adapter generation at each commit, yielding
2

an adapter trajectory over a repository’s lifetime;
no analogue exists in the Text2LoRA/Doc2LoRA
line of work, which only model a single static input.
Software evolution and continual code adap-
tation.Software evolution and mining software
repository—tracking how code changes commit by
commit, file by file—is a well-established line of
software engineering research (Kagdi et al., 2007;
Hassan, 2008), underpinning analyses of change
impact, bug introduction ( ´Sliwerski et al., 2005),
and refactoring detection (Tsantalis et al., 2018)
over long version histories. In NLP, a parallel
line investigateswhena deployed model should
be refreshed: continual pretraining and online fine-
tuning aim to keep language models current under
temporal drift (Lazaridou et al., 2021; Jang et al.,
2022), but typically maintain a single global check-
point and have no notion ofwhichrepository is
being adapted to. Code2LoRA-Evo sits at the in-
tersection of these two lines: it treats sequential
code diffs as the unit of update and refreshes a
repository-specific adapter as the commit history
unfolds. This is the first hypernetwork formulation
that targets repository-level adaptation under soft-
ware evolution rather than a single static snapshot.
Repository-level code understanding and gen-
eration.Prior work on incorporating repository
context typically routes information through the
input: RepoFusion (Shrivastava et al., 2023) trains
with cross-file context, RepoCoder (Zhang et al.,
2023) iteratively retrieves and generates, Repo-
Former (Wu et al., 2024) uses selective retrieval,
CoCoMIC (Ding et al., 2024) jointly models in-
file and cross-file context, R2C2-Coder (Deng
et al., 2025) enhances repo-level completion with
repository-context-aware methods, and RepoHy-
per (Phan et al., 2025) uses semantic-graph re-
trieval. Evaluation benchmarks include Cross-
CodeEval (Ding et al., 2023) and RepoBench (Liu
et al., 2024b). Code2LoRA instead distills repos-
itory knowledge into modelparameters, avoid-
ing context-window limits and per-query retrieval
cost, and—through Code2LoRA-Evo—tracks how
that knowledge changes as code evolves com-
mit by commit. We base our experiments on
Qwen2.5-Coder-1.5B (Hui et al., 2024); other re-
cent code LLMs include CodeLlama (Rozière et al.,
2024), StarCoder (Li et al., 2023), and DeepSeek-
Coder (Guo et al., 2024).3 Method
Code2LoRA is a hypernetwork framework that
generates repository-specific LoRA adapters for
a frozen code LM, effectively injecting repository
knowledge with zero inference-time token over-
head. As illustrated in Figure 1a, the framework
has three components: a sharedrepository en-
coder(§3.1) that maps repository-level context
to dense embeddings, ahypernetworkthat maps
those embeddings to LoRA weights, abase LLM
that receives the generated adapter and performs in-
ference. Only the hypernetwork is trained, via the
standard language-modeling loss; the repository en-
coder and base LLM are frozen. The two usage sce-
narios differ in hypernetwork design: Code2LoRA-
Static (§3.2) directly projects the repository embed-
ding into LoRA weights; Code2LoRA-Evo (§3.3)
inserts a GRU before the projection head to aggre-
gate a sequence of diff embeddings.
3.1 Repository Encoder
Repository-level context must be compressed into
a fixed-size vector before the hypernetwork can
consume it. We adopt a training-free two-step em-
bedding approach that works effectively in practice
using a frozen Qwen3-Embedding-0.6B model.
Step 1: file-level embedding.Each file fiin the
repository context (or its diff ∆fi) is divided into
4096-token chunks with 512-token overlap, embed-
ded by the frozen model, and mean-pooled over
chunks to produce a file vectorf i∈Rd(d=1024).
Step 2: repository-level aggregation.For a full
repository snapshot, each file vector receives an im-
portance weight wibased on a combination of con-
tent distinctiveness, file size, and path importance.
The repository embedding is the concatenation of
a weighted mean and a max pool,
e=P
iwifi; max ifi
∈R2d,
capturing both the average character and the most
distinctive features of the codebase. The embed-
dings are pre-computed at training time.
3.2 Code2LoRA-Static
The static hypernetwork in Code2LoRA-Static
maps a single embedding eto a LoRA adapter
in one forward pass. For each module type
m∈ {q,k,v,o,gate,up,down} , its LoRA matri-
cesAmandBmare generated by a shared 2-layer
3

Figure 1: Code2LoRA architecture.(a)Overall pipeline: repository context is encoded and mapped to LoRA
adapters, which are injected into a frozen LLM to support inference (example task: assertion completion).
(b)Code2LoRA-Static’s static hypernetwork.(c)Code2LoRA-Evo’s recurrent hypernetwork.
MLP with GELU activation followed by dedicated
output heads:
h=p
dhL2Norm(MLP(e)),
Am= tanh(HeadA
m(h))·exp(sA
m),
Bm= tanh(HeadB
m(h))·exp(sB
m),
where learnable log-scales sA/B
m control adapter
magnitudes (initialized to −3.5 ). LoRA matri-
ces are shared across all layers of base LLM
and injected via W′=W+α
rBmAm. With
hidden dimension dh=1024 and LoRA rank
r=16 , the Code2LoRA-Static hypernetwork has
∼720M trainable parameters. Code2LoRA-
Static’s hypernetwork architecture is similar to
that of Text2LoRA (Charakorn et al., 2025) and
Doc2LoRA (Charakorn et al., 2026), but (1) is
driven by a whole-repository embedding summa-
rized from millions of tokens rather than a task
description, and (2) injects LoRA to all seven mod-
ule types rather than just Q/V or down-projection
to be more flexible.
3.3 Code2LoRA-Evo
The recurrent hypernetwork in Code2LoRA-Evo
maintains a repository-specific adapter over a
chronological stream of diff embeddings {et}. Thediff embeddings are aggregated by a GRU recurrent
neural network: at step tthe encoder (§3.1) sup-
plieset, which is linearly projected and combined
with the previous state,
zt= GRU(LayerNorm(Linear(e t)),z t−1).
The initial GRU state z0is initialized by a small lin-
ear projector given the initial repository embedding
(e.g., on the first commit). At each step t, the LoRA
adapter is generated by the shared head (§3.2) with
ztsubstituted for e, yielding anadapter trajectory
over the repository’s lifetime. Each update requires
only one GRU step on the stored diff embedding et,
which is substantially cheaper than re-encoding the
full repository. Beyond the shared head, the GRU
and initial-state projector add ∼25M parameters,
for∼745M trainable parameters in total.
3.4 Training
We train the hypernetwork end-to-end by minimiz-
ing cross-entropy on assertion-completion pairs
from the frozen base LLM, with LoRA weights
supplied byHypernetworkθ:
L(θ) =−X
(x,y)∈Dlogp(y|x; Hypernetworkθ(u)),
4

where xis the input prefix, ythe output target,
andu=e for Code2LoRA-Static or u=z tfor
Code2LoRA-Evo. For Code2LoRA-Evo, we opti-
mize with truncated backpropagation through time,
detaching ztevery K=16 steps (App. D). Batches
are formed by first sampling a repository, then a
pair of input-output from it, so that the hypernet-
work sees diverse repositories and does not overfit
to data-rich ones.
4 RepoPeftBench: A Repository-Level
PEFT Benchmark
We construct RepoPeftBench, a repository-level
benchmark for parameter-efficient fine-tuning of
code language models. The corpus comprises
604 Python repositories drawn from GitHub un-
der shared quality filters—each uses pytest or
unittest , carries a permissive license, and shows
recent activity—partitioned along a fixed temporal
cutoff ( 2025-04-01 ) into 512 in-distribution repos-
itories and a 92 out-of-distribution (OOD) reposi-
tories. The in-distribution set was collected before
the cutoff date, and requires an additional filter of
having at least 300 stars (to ensure high-quality),
which supplies all training and validation splits;
commit histories are truncated at the cutoff date.
The OOD set comprises repositories created strictly
after the cutoff date and is reserved for held-out
test-time evaluation only (§6.3). We collect both
the last snapshot as well as the full commit histories
of each repository.
Two evaluation tracks share the same task, met-
rics, and CR/IR repository partitions but differ in
how instances are indexed in history (§4). Table 1
summarizes the split sizes used in all reported re-
sults. Benchmark construction details are in Ap-
pendix B.
Task.Each instance is an assertion-completion
input-output pair: the model receives a structured
prefix from a test file and must predict the expected
value of an assertion. The task follows the code-
execution probe of LiveCodeBench (Jain et al.,
2025), but replaces hand-curated single-function
snippets with assertions mined at scale from real
test suites. Assertion completion is well suited to
repository-level evaluation because all instances
in a repository share the same non-test source as
conditioning context. Repository-level code com-
pletion, as in RepoBench (Liu et al., 2024b), is not
suitable because each target file requires excluding
that file from context to prevent leakage and thus adifferent repository slice per instance. CrossCodeE-
val (Ding et al., 2023), RepoHyper (Phan et al.,
2025), and R2C2-Coder (Deng et al., 2025) like-
wise ship only retrieval-selected slices; RepoPeft-
Bench releases full information of each repository
to evaluate methods that ingest the full codebase.
We extract instances from five assertion families:
bare assert ,self.assert* ,pytest.raises ,
pytest.approx , and NumPy-style assert_* . The
inputconcatenates imports, the enclosing class (if
any), helper methods, and the test-function body
up to the assertion cut point; theoutputis the ex-
pected value of the assertion, namely the right-hand
side of the binary comparison operator, or the last
argument of the assertion function call.
Repository splits.We partition the 512 in-
distribution repositories into cross-repo (CR) and
in-repo (IR) sets, shared by both evaluation tracks.
Cross-Repo (CR)holds out 103 repositories en-
tirely at training time (51 validation, 52 test) to
measure generalization to unseen codebases.In-
Repo (IR)uses the remaining 409 repositories
for training and is the only setting in which per-
repository LoRA is defined; held-out instances
within each training repository are assigned by the
track-specific protocol below.
Evaluation tracks.TheStatictrack draws ev-
ery instance from a single snapshot per repository
(62,294 tasks) and corresponds to Code2LoRA-
Static: on CR splits, tasks are extracted from each
held-out repository’s last commit snapshot; on IR
splits, tasks are also extracted from last commits,
and are randomly split into training, validation,
and test sets in a ratio of 8:1:1. TheEvolution
track replays each repository’s commit history and
emits a task whenever a commit adds or modifies
an assertion, storing the input-output pair together
with the production-code diff ∆t; it corresponds to
Code2LoRA-Evo. On CR splits, evaluation uses
all commit-derived tasks from held-out reposito-
ries; on IR splits, following the time-segmented
methodology of Nie et al. (2022), commits within
each training repository are partitioned chronolog-
ically so that training examples strictly precede
validation and test. Evolution-track training and
evaluation each retain at most eight tasks per com-
mit; Code2LoRA-Evo training further caps at four
tasks per test file so that no commit dominates a
backpropagation window. Table 1 reports the num-
ber of tasks for every split used in our experiments.
Commit histories are bursty: repositories accumu-
5

late hundreds of test-touching commits in irregu-
lar clusters (Appendix Figure 2), which motivates
streaming adaptation via Code2LoRA-Evo rather
than a single frozen snapshot.
5 Experimental Setup
Models.The base LLM is Qwen2.5-Coder-
1.5B (Hui et al., 2024), loaded in bfloat16 ;
all baselines and both Code2LoRA usage sce-
narios share this backbone. Repository encoder
uses Qwen3-Embedding-0.6B (Zhang et al., 2025).
Both models are released under the Apache 2.0 li-
cense and our research use is consistent with their
model cards.
Hyperparameters.Code2LoRA generate rank-
r=16 LoRA adapters with α=32 for all
seven attention/MLP projection types, with each
(Am,Bm)pair shared across all 28transformer
layers (§3). Code2LoRA-Static has ∼720M
trainable parameters, while Code2LoRA-Evo has
∼745M trainable parameters. We train both for
3 epochs with AdamW (cosine schedule) on a
single H100 80 GB GPU using TRL (von Werra
et al., 2020); full hyperparameters, schedules, and
sequence-length budgets are in Appendix D.
Baselines.We evaluate against various baselines:
• Pretrained: base LLM (Qwen2.5-Coder-1.5B).
•RAG ( k=3): non-test source files pre-chunked
into 512-token segments, embedded with
Qwen3-Embedding-0.6B; top- kretrieved chunks
prepended to the prefix at inference (results for
k∈{5,10}and chunk size256in Appendix C).
•Dep.-Resolved Context: prepends function and
class definitions reachable from each prefix’s im-
ports via dependency analysis, with relevance-
aware compression under an adaptive token bud-
get (Appendix D.1).
• FFT: all model parameters are made trainable.
•Single LoRA: one rank-16 adapter trained onall
repositories.
•Per-repo LoRA (Zong et al., 2025): one rank-16
adapter trainedperrepository (IR splits only),
serving as an upper bound on repository-level
adaptation.
•Text2LoRA (Charakorn et al., 2025): a hyper-
network that emits a LoRA from an external
task embedding. To control for input modality
and target-module coverage, we strengthen theupstream baseline along both axes: the natural-
language task description is replaced with the
same repository encoder that Code2LoRA uses
(mean +max-pooled Qwen3-Embedding-0.6B),
and the output heads are extended from{Q,V}
to all seven attention and MLP projections. Train-
ing data, loss, and budget match Code2LoRA, so
only the LoRA-generation head differs (details
in Appendix D).
Evaluation metrics.We reportExact Match
(EM, after whitespace collapsing and trailing-
punctuation removal, with relaxed matching that
tolerates model overgeneration);Edit Similar-
ity(difflib (Python Software Foundation, 2024)
SequenceMatcher ratio); andCodeBLEU(Ren
et al., 2020), which incorporates AST and data-
flow structure in addition to n-gram overlap.
6 Results
We organize the results around the two evalua-
tion tracks of RepoPeftBench. The static track
(§6.1, Table 2) evaluates Code2LoRA-Static and
baselines on a single snapshot of each reposi-
tory; Code2LoRA-Evo requires commit history and
therefore does not apply to this track. The evolu-
tion track (§6.2, Table 3) evaluates all methods on
commit-derived prefixes.
6.1 Static Track
Table 2 shows the results on RepoPeftBench’s
static track. On CR evaluation, Code2LoRA-Static
reaches 63.8% EM, +9.9 pp over the strongest
baseline (FFT + RAG, 53.9%) and above ev-
ery context-injection method (RAG ( k=3) 39.7%,
Dep.-Resolved Context 48.2%) and other fine-
tuned baselines (FFT 51.4%, Single LoRA 47.4%).
The strengthened Text2LoRA baseline, which
matched with Code2LoRA on input modality
(whole-repository embedding) and target-module
coverage (all seven projections), reaches only
45.8% EM; this isolates the Text2LoRA hypernet-
work as the bottleneck for repository-level adapta-
tion, since only the LoRA-generation head differs
from Code2LoRA-Static once input and targets are
matched. On IR evaluation, Code2LoRA-Static
reaches 66.2% EM, matching the Per-repo LoRA
upper bound (64.0%) without any per-repository
training—confirming that cross-repository transfer
learned by the hypernetwork is more valuable than
fitting one adapter per repository on the in-repo
data budget.
6

Table 1: Dataset statistics for RepoPeftBench, divided into static and evolution tracks (sharing the same set of 512
in-distribution repositories) and 92 out-of-distribution repositories, split into train/val/test sets under cross-repo (CR)
and in-repo (IR) settings.
Split Repos Commits Tasks Tasks / repo
Static track
Train 409 409 39,612 96.9
CR Val / Test 51 / 52 51 / 52 6,213 / 6,414 121.8 / 123.3
IR Val / Test 409 / 409 409 / 409 4,833 / 5,222 11.8 / 12.8
Evolution track
Train (Code2LoRA-Static and baselines) 400†400 44,149 110.4
Train (Code2LoRA-Evo) 400†45,516 215,129 537.8
CR Val / Test 49 / 51 8,614 / 6,618 58,944 / 44,732 1,203 / 877
IR Val / Test 389 / 389 5,710 / 6,179 38,783 / 42,061 99.7 / 108.1
Out-of-distribution holdout
OOD Test 92 1,950 14,813 161.0
†9 repositories lack sufficient commit histories and are excluded from Code2LoRA-Evo training.
Table 2: Results on RepoPeftBench static track.
Cross-Repo (CR Test) In-Repo (IR Test)
Method EM (%) EditSim CodeBLEU EM (%) EditSim CodeBLEU
Inference-only (no fine-tuning)
Pretrained 45.7 0.605 0.646 46.8 0.624 0.655
RAG (k=3) 39.7 0.516 0.556 42.1 0.544 0.581
Dep.-Resolved Context 48.2 0.625 0.657 49.5 0.640 0.667
Fine-tuned
FFT 51.4 0.695 0.678 55.9 0.727 0.714
FFT + RAG 53.9 0.703 0.688 56.8 0.731 0.713
Single LoRA 47.4 0.663 0.649 50.4 0.687 0.675
Per-repo LoRA†— — — 64.0 0.801 0.788
Hypernetwork-based
Text2LoRA 45.8 0.606 0.647 46.7 0.625 0.655
Code2LoRA-Static 63.8 0.784 0.778 66.2 0.806 0.797
†Per-repo LoRA is an in-repo upper bound and is not applicable to the cross-repo setting.
6.2 Evolution Track
Real repositories evolve commit by commit, and
a static snapshot adapter goes stale once the edit
stream diverges from the snapshot it was trained
on. The evolution track (Table 3) evaluates with
commit-derived tasks and is where Code2LoRA-
Evo—with a GRU that aggregates sequential code
diffs—applies.
Table 3 reports evolution-track results on
commit-derived prefixes. Relative to the static track
(Table 2), commit-derived tasks are substantially
harder: Pretrained CR EM drops from 45.7% to
31.5%. Both context-injection methods collapse:
RAG ( k=3) falls below the pretrained backbone on
CR and IR, while Dep.-Resolved Context recov-
ers only to pretrained levels on CR and yields amodest IR gain. Among fine-tuned methods, Sin-
gle LoRA reaches 55.1% / 61.3% EM; Per-repo
LoRA reaches 64.2% IR EM (the only applicable
split). Code2LoRA-Static, included as a within-
framework reference on the same commit-derived
inputs, scores 55.7% / 60.6%, which is close to
Single LoRA on CR and markedly below its static-
track performance (63.8% / 66.2%). The strength-
ened Text2LoRA baseline reaches only 41.7% /
43.5% EM, far below both Code2LoRA variants
on this track. Code2LoRA-Evo is the strongest
method on both splits (60.3% CR, 64.5% IR EM),
+5.2 pp over Single LoRA on CR and exceeding
the Per-repo LoRA upper bound on IR without
per-repository training. Appendix Figure 9 (§F.3)
shows that this lead persists across long commit
7

Table 3: Results on RepoPeftBench evolution track.
Cross-Repo (CR Test) In-Repo (IR Test)
Method EM (%) EditSim CodeBLEU EM (%) EditSim CodeBLEU
Inference-only (no fine-tuning)
Pretrained 31.5 0.490 0.515 29.3 0.469 0.501
RAG (k=3) 23.6 0.411 0.446 23.0 0.402 0.437
Dep.-Resolved Context 31.1 0.490 0.516 31.6 0.494 0.517
Fine-tuned
Single LoRA 55.1 0.749 0.709 61.3 0.787 0.753
Per-repo LoRA†— — — 64.2 0.803 0.788
Hypernetwork-based
Text2LoRA 41.7 0.596 0.600 43.5 0.612 0.613
Code2LoRA-Static 55.7 0.760 0.716 60.6 0.787 0.749
Code2LoRA-Evo 60.3 0.810 0.763 64.5 0.828 0.790
†Per-repo LoRA is an in-repo upper bound and is not applicable to the cross-repo setting.
Table 4: Results on RepoPeftBench OOD set.
Method EM (%) EditSim CodeBLEU
Inference-only (no fine-tuning)
Pretrained 44.6 0.568 0.630
RAG (k=3) 32.6 0.464 0.536
Dep.-Resolved Context 45.5 0.584 0.637
Fine-tuned
Single LoRA 72.3 0.836 0.817
Hypernetwork-based
Text2LoRA 60.4 0.720 0.740
Code2LoRA-Static 72.2 0.842 0.818
Code2LoRA-Evo 74.1 0.866 0.846
histories, with the smallest downward drift among
fine-tuned methods. Together with the static track
(§6.1), these results show a consistent ordering:
parametric adaptation outperforms context injec-
tion on both tracks, and recurrent aggregation over
commit diffs outperforms a static snapshot once
evaluation follows repository evolution.
6.3 Out-of-Distribution Generalization
The OOD set comprises 92 repositories created
strictly after the in-distribution training cutoff
(2025-04-01 ) and used for held-out evaluation
only, which challenges the generalization of the
learned hypernetwork on new types of repository-
level context. Table 4 reports results on the tem-
poral holdout under the same commit-derived pro-
tocol as Table 3. Code2LoRA-Evo achieves the
highest EM (74.1%), ahead of Code2LoRA-Static
(72.2%) and Single LoRA (72.3%). OOD as-
sertion targets are systematically shorter than in-distribution ones (median 7 characters vs. 12–13;
Appendix E), which uniformly inflates exact-match
scores on this split and explains why Single LoRA
reaches 72.3% here despite 55.1% / 61.3% on the
evolution track; we therefore restrict comparison
to within Table 4. On that basis, Code2LoRA-Evo
leads the next-best fine-tuned adapter by ∼1.8 pp
EM—narrower than the evolution-track gap ( ∼5 pp
CR EM, Table 3) but positive and consistent across
EditSim and CodeBLEU.
7 Conclusion
We introduced Code2LoRA, a hypernetwork frame-
work that generates repository-specific LoRA
adapters, effectively injecting repository knowl-
edge with zero inference-time token overhead, and
RepoPeftBench, a benchmark of 604 Python reposi-
tories suitable for evaluating repository-level PEFT
methods. The framework instantiates two usage
scenarios along how knowledge enters parame-
ters and when it is refreshed: Code2LoRA-Static,
which maps a repository snapshot to an adapter for
stable codebases and reaches 63.8% CR / 66.2%
IR EM on the static track; and Code2LoRA-Evo,
which maintains an adapter via a GRU hidden state
updated on each code diff for evolving codebases
and reaches 60.3% CR / 64.5% IR EM on the evo-
lution track. Experiments on out-of-distribution
repositories confirms the strong generalization ca-
pability of Code2LoRA. These results demonstrate
that repository knowledge is best injected para-
metrically and updated to track software evolution
rather than through long input context. We envi-
sion Code2LoRA as a building block will support
8

stronger, customizable to repository-level context,
and less costly AI code assistants.
Limitations
Scope of evaluation.We evaluate only on Python
repositories, a single frozen backbone (Qwen2.5-
Coder-1.5B), and one downstream task (asser-
tion completion derived from naturally occurring
pytest /unittest suites). The architecture is in
principle language- and task-agnostic by construc-
tion (multi-language embedder, per-module-type
LoRA targets shared across all layers), but extend-
ing the empirical evidence to additional languages,
backbones, and downstream tasks is left to future
work.
OOD target-length artifact.The 74.1% OOD
EM (Table 4) may be partially inflated because
assertion targets in our strictly post-cutoff OOD
repositories are systematically shorter (median 7
characters) than in CR/IR test (median 12–13 char-
acters); this confound is shared by every OOD row
and we discuss it in Appendix E. We therefore em-
phasize the within-OOD comparison: Code2LoRA-
Evo leads the next-best fine-tuned adapter by
∼1.8 pp EM, with the direction of the effect consis-
tent across all metrics.
Surface-level metrics.Exact match misses func-
tional equivalence; we mitigate with EditSim,
CodeBLEU, and a pytest -based execution probe
on a runnable CR-test slice. A more semantic eval-
uation (e.g., executing every generated assertion
against the project’s test runtime) is a natural ex-
tension but was out of scope for this submission’s
compute budget.
Model size.The LoRA-generation hypernet-
work dominates the trainable parameter count—
∼720M for Code2LoRA-Static and ∼745M for
Code2LoRA-Evo—and is itself a function of the
backbone’s projection dimensions. The evolution-
track finding is therefore most directly supported
at the 1.5B-parameter scale; whether recurrent ag-
gregation over commit diffs remains necessary (or
sufficient) once the backbone is much larger is an
open question.
Reproducibility.Code, RepoPeftBench, and hy-
perparameters (Appendix D) will be released upon
acceptance; all experiments run on a single H100
80 GB GPU.Potential risks.RepoPeftBench is constructed
exclusively from public permissively licensed
Python repositories (Appendix B), so the dataset
itself does not introduce new personal data, harm-
ful content, or proprietary code into circulation,
and we redistribute each repository under its orig-
inal license terms with attribution preserved. The
downstream artifact—a code language model con-
ditioned on a repository-specific LoRA—inherits
the well-understood risks of code LLMs more
broadly: it can be steered to emit insecure, incor-
rect, or licensed-code-resembling completions, and
our repository-conditioning amplifies attribution
risk if a user feeds in a private repository and the
generated assertions surface verbatim from training
repos. We make no claims of safety for production
deployment without standard mitigations (license-
aware filtering of generated code, human review
of generated test assertions before commit, and
rejection of completions matching long verbatim
training spans).
Acknowledgments
We thank Saarang Agarwal, Kyunghyun Cho, Bi-
hui Jin, Jiale Amber Wang, Wentao Zhang, Yifan
Zong and the anonymous reviewers for their com-
ments and feedback. This work is enabled in part
by support provided by Compute Ontario (com-
puteontario.ca) and the Digital Research Alliance
of Canada (alliancecan.ca). This work is partially
supported by the Natural Sciences and Engineer-
ing Research Council of Canada (NSERC) under
funding reference number RGPIN-2024-04909 and
RGPIN-2024-05178.
References
Mohamed Hesham Ibrahim Abdalla, Zhipin Wang,
Christian Frey, Steffen Eger, and Josif Grabocka.
2025. Zhyper: Factorized hypernetworks
for conditioned LLM fine-tuning.Preprint,
arXiv:2510.19733.
Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, and
Robert Tjarko Lange. 2025. Text-to-loRA: Instant
transformer adaption. InForty-second International
Conference on Machine Learning.
Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Ue-
saka, and Robert Tjarko Lange. 2026. Doc-to-lora:
Learning to instantly internalize contexts.Preprint,
arXiv:2602.15902.
Saumya Chaturvedi, Aman Chadha, and Laurent Bind-
schaedler. 2025. LoRACode: LoRA adapters for
9

code embeddings. InICLR 2025 Third Workshop on
Deep Learning for Code.
Tong Chen, Hao Fang, Patrick Xia, Xiaodong Liu, Ben-
jamin Van Durme, Luke Zettlemoyer, Jianfeng Gao,
and Hao Cheng. 2025. Generative adapter: Con-
textualizing language models in parameters with a
single forward pass. InThe Thirteenth International
Conference on Learning Representations.
Ken Deng, Jiaheng Liu, He Zhu, Congnan Liu, Jingxin
Li, Jiakai Wang, Peng Zhao, Chenchen Zhang,
Yanan Wu, Xueqiao Yin, Yuanxing Zhang, Zizheng
Zhan, Wenbo Su, Bangyu Xiang, Tiezheng Ge, and
Bo Zheng. 2025. R2c2-coder: Enhancing and bench-
marking real-world repository-level code completion
abilities of code large language models.Preprint,
arXiv:2406.01359.
Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and
Luke Zettlemoyer. 2023. QLoRA: Efficient finetun-
ing of quantized LLMs. InConference on Neural
Information Processing Systems.
Yangruibo Ding, Zijian Wang, Wasi Ahmad, Murali Kr-
ishna Ramanathan, Ramesh Nallapati, Parminder
Bhatia, Dan Roth, and Bing Xiang. 2024. CoCoMIC:
Code completion by jointly modeling in-file and
cross-file context. InProceedings of the 2024 Joint
International Conference on Computational Linguis-
tics, Language Resources and Evaluation (LREC-
COLING 2024), pages 3433–3445.
Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Han-
tian Ding, Ming Tan, Nihal Jain, Murali Krishna
Ramanathan, Ramesh Nallapati, Parminder Bhatia,
Dan Roth, and Bing Xiang. 2023. Crosscodeeval:
A diverse and multilingual benchmark for cross-file
code completion. InThirty-seventh Conference on
Neural Information Processing Systems Datasets and
Benchmarks Track.
Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai
Dong, Wentao Zhang, Guanting Chen, Xiao Bi,
Y . Wu, Y . K. Li, Fuli Luo, Yingfei Xiong, and Wen-
feng Liang. 2024. Deepseek-coder: When the large
language model meets programming – the rise of
code intelligence.Preprint, arXiv:2401.14196.
David Ha, Andrew M. Dai, and Quoc V . Le. 2017. Hy-
pernetworks. InInternational Conference on Learn-
ing Representations.
Ahmed E. Hassan. 2008. The road ahead for mining
software repositories. In2008 Frontiers of Software
Maintenance, pages 48–57.
Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu
Chen. 2022. LoRA: Low-rank adaptation of large
language models. InInternational Conference on
Learning Representations.
Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu
Pang, Chao Du, and Min Lin. 2024. Lorahub: Effi-
cient cross-task generalization via dynamic lora com-
position.Preprint, arXiv:2307.13269.Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Day-
iheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang,
Bowen Yu, Keming Lu, Kai Dang, Yang Fan,
Yichang Zhang, An Yang, Rui Men, Fei Huang,
Bo Zheng, Yibo Miao, Shanghaoran Quan, and 5 oth-
ers. 2024. Qwen2.5-Coder technical report.Preprint,
arXiv:2409.12186.
Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol
Vinyals, Andrew Zisserman, and Joao Carreira. 2021.
Perceiver: General perception with iterative atten-
tion. InProceedings of the 38th International Con-
ference on Machine Learning, volume 139, pages
4651–4664.
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia
Yan, Tianjun Zhang, Sida Wang, Armando Solar-
Lezama, Koushik Sen, and Ion Stoica. 2025. Live-
codebench: Holistic and contamination free evalua-
tion of large language models for code. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin,
Janghoon Han, Gyeonghun Kim, Stanley Jungkyu
Choi, and Minjoon Seo. 2022. Towards continual
knowledge learning of language models. InInterna-
tional Conference on Learning Representations.
Huzefa Kagdi, Michael L. Collard, and Jonathan I.
Maletic. 2007. A survey and taxonomy of approaches
for mining software repositories in the context of soft-
ware evolution.Journal of Software Maintenance
and Evolution: Research and Practice, 19(2):77–
131.
Angeliki Lazaridou, Adhi Kuncoro, Elena Gribovskaya,
Devang Agrawal, Adam Liska, Tayfun Terzi, Mai
Gimenez, Cyprien de Masson d’Autume, Tomáš Ko-
cisk`y, Sebastian Ruder, Dani Yogatama, Kris Cao,
Susannah Young, and Phil Blunsom. 2021. Mind the
gap: Assessing temporal generalization in neural lan-
guage models. InConference on Neural Information
Processing Systems.
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas
Muennighoff, Denis Kocetkov, Chenghao Mou, Marc
Marone, Christopher Akiki, Jia Li, Jenny Chim,
Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo,
Thomas Wang, Olivier Dehaene, Mishig Davaadorj,
Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko,
and 48 others. 2023. Starcoder: may the source be
with you!Preprint, arXiv:2305.06161.
Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo
Molchanov, Yu-Chiang Frank Wang, Kwang-Ting
Cheng, and Min-Hung Chen. 2024a. DoRA: Weight-
decomposed low-rank adaptation. InProceedings of
the 41st International Conference on Machine Learn-
ing, volume 235 ofProceedings of Machine Learning
Research, pages 32100–32121.
Tianyang Liu, Canwen Xu, and Julian McAuley. 2024b.
Repobench: Benchmarking repository-level code
auto-completion systems.
10

Chuancheng Lv, Lei Li, Shitou Zhang, Gang Chen, Fan-
chao Qi, Ningyu Zhang, and Hai-Tao Zheng. 2024.
HyperLoRA: Efficient cross-task generalization via
constrained low-rank adapters generation. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 16376–16393.
Pengyu Nie, Jiyang Zhang, Junyi Jessy Li, Raymond J.
Mooney, and Milos Gligoric. 2022. Impact of evalu-
ation methodologies on code summarization. InAn-
nual Meeting of the Association for Computational
Linguistics, pages 4936–4960.
Huy N. Phan, Hoang N. Phan, Tien N. Nguyen, and
Nghi D. Q. Bui. 2025. Repohyper: Search-expand-
refine on semantic graphs for repository-level code
completion. In2025 IEEE/ACM Second Interna-
tional Conference on AI Foundation Models and Soft-
ware Engineering (Forge), page 14–25. IEEE Press.
Jason Phang, Yi Mao, Pengcheng He, and Weizhu Chen.
2023. HyperTuning: Toward adapting large language
models without back-propagation. InProceedings
of the 40th International Conference on Machine
Learning, volume 202, pages 27854–27875. PMLR.
Python Software Foundation. 2024. difflib — helpers
for computing deltas. https://docs.python.org/
3/library/difflib.html.
Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu,
Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio
Blanco, and Shuai Ma. 2020. Codebleu: a method
for automatic evaluation of code synthesis.Preprint,
arXiv:2009.10297.
Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten
Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi,
Jingyu Liu, Romain Sauvestre, Tal Remez, Jérémy
Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna
Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron
Grattafiori, Wenhan Xiong, Alexandre Défossez, and
7 others. 2024. Code llama: Open foundation models
for code.Preprint, arXiv:2308.12950.
Disha Shrivastava, Denis Kocetkov, Harm de Vries,
Dzmitry Bahdanau, and Torsten Scholak. 2023. Re-
pofusion: Training code models to understand your
repository.Preprint, arXiv:2306.10998.
Jacek ´Sliwerski, Thomas Zimmermann, and Andreas
Zeller. 2005. When do changes induce fixes?ACM
SIGSOFT Software Engineering Notes, 30(4):1–5.
Nikolaos Tsantalis, Mohammad Mansouri, Laleh M.
Eshkevari, Davood Mazinanian, and Danny Dig.
2018. Accurate and efficient refactoring detection
in commit history. InInternational Conference on
Software Engineering, pages 483–494.
Leandro von Werra, Younes Belkada, Lewis Tunstall,
Edward Beeching, Tristan Thrush, Nathan Lambert,
Shengyi Huang, Kashif Rasul, and Quentin Gal-
louédec. 2020. TRL: Transformers Reinforcement
Learning.Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Kr-
ishna Ramanathan, and Xiaofei Ma. 2024. Repo-
former: Selective retrieval for repository-level code
completion.Preprint, arXiv:2403.10059.
Prateek Yadav, Derek Tam, Leshem Choshen, Colin
Raffel, and Mohit Bansal. 2023. TIES-merging: Re-
solving interference when merging models. InThirty-
seventh Conference on Neural Information Process-
ing Systems.
Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin
Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and
Weizhu Chen. 2023. RepoCoder: Repository-level
code completion through iterative retrieval and gen-
eration. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2471–2484, Singapore.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Yifan Zong, Yuntian Deng, and Pengyu Nie. 2025.
Mix-of-Language-Experts Architecture for Multilin-
gual Programming . In2025 IEEE/ACM Interna-
tional Workshop on Large Language Models for Code
(LLM4Code), pages 200–208. IEEE Computer Soci-
ety.
11

A Use of LLMs
We used an LLM-based writing assistant to polish
grammar. All ideas, analyses, experiments, and
scientific claims are our own, and we take full re-
sponsibility for the content of this work.
B Dataset Details
This section documents detailed construction pro-
cess and statistics of RepoPeftBench, organized as
the data flows from raw GitHub repositories to the
QnA splits actually consumed by the methods in
Tables 2–4: task motivation (§B.1), repository se-
lection and licensing (§B.2), construction pipeline
(§B.3), the splits used at training and evaluation
(§B.4), composition by assertion family and tar-
get type (§B.5), token-length distributions (§B.6),
per-repository breakdown (§B.7), and the privacy /
content review (§B.8).
B.1 Motivation and Task
Why a repository-conditioned assertion task.
Our assertion-completion task is directly inspired
by thecode executiontask of LiveCodeBench (Jain
et al., 2025), which probes whether a model can pre-
dict the runtime value produced by a piece of code
at a designated point of evaluation. Treating an
assertion target as the “answer” a developer wrote
down for what a piece of codeshouldevaluate to
at exactly that line, the prediction objective inherits
the same semantics—compute, in the model’s head,
what this expression would resolve to in this con-
crete context—while replacing LiveCodeBench’s
hand-curated, single-function snippets with natu-
rally occurring assertions extracted at scale from
real test suites. This reframing keeps the cognitive
load of the original task (multi-step, type-aware,
value-level reasoning over surrounding code) and
additionally couples each prediction to a full repos-
itory’s API surface, naming conventions, fixtures,
and domain vocabulary—turning code execution
into an explicitrepository-conditionedreasoning
probe.
Why a new dataset.Existing repository-level
benchmarks (RepoBench (Liu et al., 2024b), Cross-
CodeEval (Ding et al., 2023), RepoHyper (Phan
et al., 2025), R2C2-Coder (Deng et al., 2025)) ship
only the slices their task consumes—a target file
and a handful of retrieval-selected snippets—and
discard the rest of the codebase and the Git history
at release time. This is fine for input-side meth-
ods but precludes any method that must ingest thewholerepository as parameters or as a streaming
state. We therefore release each repository in Re-
poPeftBenchwhole: every non-test source file (for
the repository representation), every test file (for
assertion QnAs), and every first-parent production
commit (for the evolution track’s diff sequences).
B.2 Repository Selection and Licensing
In-distribution selection.The
GitHub Search API was queried with
language:python license:mit stars:>=300
pushed:>=2023-01-01 together with a
pytest /unittest usage filter; matching reposito-
ries were ranked by star count and downloaded in
two passes (the upper pool with ≥1000 stars and a
mid-range pool with 300–1000 stars), yielding the
512in-distribution repositories used for training
and CR/IR evaluation.
Temporal OOD holdout.To probe generaliza-
tion beyond the training scrape, we mined an
additional set of repositories with the same lan-
guage, testing, activity, size, and non-fork fil-
ters butwithoutthe ≥300 -star constraint—star-
count ranges were searched from 6upward so
that enough candidates exist among repositories
created strictly after 2025-04-01 . Permissive li-
censes (MIT and Apache-2.0) were both consid-
ered during mining; 92repositories passed fork-
chain and pytest checks and yield valid assertion
pairs. Together with the in-distribution corpus,
these form the 604repositories in RepoPeftBench.
Because the in-distribution query hard-filtered on
license:mit , all512in-distribution repositories
are MIT-licensed; the OOD holdout may include
Apache-2.0 repositories where that was the up-
stream license. We retain a copy of each repos-
itory’s LICENSE file alongside the source tree in the
released dataset, and the dataset release itself is dis-
tributed under the same MIT terms with attribution
to the upstream maintainers preserved.
Intended use and consistency with upstream
terms.Using the source contents of MIT-
licensed public repositories for research on code
language models is consistent with the upstream
license, which explicitly permits use, modification,
and redistribution provided that the copyright no-
tice is included. RepoPeftBench and the released
Code2LoRA checkpoints are intended exclusively
for non-commercial research on repository-level
adaptation of code LMs; downstream commer-
cial or product deployment is out of scope for
12

this release and would require an independent re-
licensing review of each contributing repository.
Derivatives produced from the dataset (e.g., embed-
dings, generated LoRAs, predictions) inherit the
same research-use scope.
B.3 Construction Pipeline
Test file identification.Files are classified as
test files if they match any of: test_*.py ,
*_test.py , or reside in directories named tests/ ,
test/ . Identified test files are moved to a separate
TEST_HYPERNET/ directory within each repository,
preserving relative paths.
Structured prefix construction.Each QnA pre-
fix is constructed as follows:
1. All import statements from the test file.
2.The enclosing class definition (if the test is a
method).
3. Helper methods (setUp,tearDown, fixtures).
4.The test function signature and body up to the
assertion cut point.
This structured approach preserves the most infor-
mative context while managing token budget.
Quality filters applied.
•Targets starting with comma (malformed AST
segmentation).
•Targets outside function bodies (module-level
assertions).
• Empty or whitespace-only targets.
•Duplicate targets within the same test func-
tion.
•Targets containing only punctuation or single
characters.
Bursty commit pattern.Figure 2 shows the per-
repository test-touching commit distribution that
motivates the evolution track: test-touching com-
mits arrive in irregular bursts rather than at uniform
intervals, so a single static snapshot of any repos-
itory fails to capture the full history of assertion
edits seen during active development.
B.4 Splits Used in Experiments
Table 1 (in the main paper) reports the exact splits
consumed by every number in Tables 2–4 (one
row per split actually used at training or evaluation
time); Table 5 below expands that overview with
per-commit and per-repository densities, including
the smart-cap output for Code2LoRA-Evo training.
For the evolution track we enforce a per-commit
cap of≤8QnAs at both training time (as part of the
2024-01 2024-04 2024-07 2024-10
Commit date (first-parent)agronholm/apschedulermagic-wormhole/magic-wormholemiguelgrinberg/python-socketiopallets-eco/flask-securitytmux-python/tmuxpcommit (no new QnA) commit with new QnAs (area  #QnAs)
Figure 2: Bursty commit pattern, illustrated using ran-
domly selected 5 repositories out of the 604 RepoPeft-
Bench repositories. Test-touching commits arrive irreg-
ularly; the median repository accumulates over 100 such
commits, motivating per-commit (rather than one-shot)
adaptation under software evolution.
smart cap, which additionally bounds at ≤4QnAs
per test file) and evaluation time: every evaluator
scores the first ≤8QnAs per (repo, commit) group
so that the EM / EditSim / CodeBLEU averages are
not dominated by a few unusually large commits
with hundreds of QnAs in a single test file. The av-
erage density after the eval-time cap is ∼6.8 QnAs
per commit (below the cap because many commits
naturally have fewer than8QnAs).
B.5 Composition by Assertion Family and
Target Type
To characterize the assertion-completion task at the
level of what the model actually predicts, Table 6
breaks down each split by assertion family (which
keyword triggers the test) and by target type (what
the assertion expects). The three splits are tightly
aligned: bare assert accounts for ∼82–86% of
pairs and the target distribution (numeric/string lit-
erals, variables, function calls, complex expres-
sions) varies by at most ∼2 pp between train, CR
test, and IR test. This rules out distribution shift
across splits as an explanation for the cross-repo
gap, and confirms that improvements on CR test
are genuine generalization rather than reweighting
of easier target categories.
B.6 Token-Length Statistics
Table 7 reports token-length distributions for the
four input components (repository, DRC con-
text, structured prefix, target) over the 62,294
static-track QnAs (Qwen2.5-Coder-1.5B tokenizer;
same denominator as Table 6). Repositories are
large (median 165K tokens), DRC context—when
present—is moderate (median 517 tokens) but
heavy-tailed, prefixes are compact (median 224
13

Table 5: Fine-grained statistics for every split actually consumed by the main tables.Static track: one anchor
snapshot per repository (rows feed Table 2).Evolution track: multi-commit prefixes (rows feed Tables 3 and 4); the
smart cap ( ≤4QnAs per test file, ≤8per commit) is applied to Code2LoRA-Evo training rows so that no commit
can dominate a backprop window.
Split Repos Commits QnAs Cmts / repo QnAs / cmt QnAs / repo
Static track — one anchor snapshot per repository (no per-commit cap)
Train 409 409 39,612 1.00 96.9 96.9
CR Val 51 51 6,213 1.00 121.8 121.8
CR Test 52 52 6,414 1.00 123.3 123.3
IR Val 409 409 4,833 1.00 11.8 11.8
IR Test 409 409 5,222 1.00 12.8 12.8
OOD Test 92 92 9,942 1.00 108.1 108.1
Evolution track — multi-commit;≤8QnAs / commit at train (smart-cap,≤4/file) and eval
Train (Code2LoRA-Static, anchor) 400 400 44,149 1.00 110.4 110.4
Train (Code2LoRA-Evo, 8-cap) 400 45,516 215,129 113.79 4.73 537.8
CR Val 49 8,614 58,944 175.80 6.84 1,203
CR Test 51 6,618 44,732 129.76 6.76 877
IR Val 389 5,710 38,783 14.68 6.79 99.7
IR Test 389 6,179 42,061 15.88 6.81 108.1
OOD Test 92 1,950 14,813 21.20 7.60 161.0
Table 6: Composition of the static-track QnAs by asser-
tion family (which keyword triggers the test) and target
type (what the assertion expects), computed over the
62,294 QnAs actually used at training and evaluation
time (sum of static train, CR Val/Test, and IR Val/Test
rows in Table 1). Splits are tightly aligned: every target-
type fraction differs by at most ∼2pp between train, CR
test, and IR test.
Property Train CR Test IR Test
Assertion types
assert82.5% 86.2% 82.2%
self.assert*13.5% 10.0% 13.6%
pytest.*4.1% 3.8% 4.3%
Target types
Numeric literal 18.7% 19.9% 19.4%
String literal 18.2% 18.2% 18.5%
Variable 21.7% 21.9% 21.8%
Collection 11.8% 10.2% 11.3%
Function call 9.4% 10.2% 8.9%
Complex expression 14.5% 14.0% 15.0%
Bool/None literal 5.8% 5.5% 5.1%
tokens), and targets are short (median 3tokens).
Figure 3 plots the prefix-only and DRC+prefix
length distributions side by side and marks com-
mon context-window sizes, illustrating why DRC
training requires the 8K-context setting of Table 9.
B.7 Per-Repository Performance Breakdown
To support repository-by-repository scrutiny of ev-
ery method, we release a per-repository table cover-
ing all 409IR-test repositories with EM, EditSim,
CodeBLEU, and example counts for pretrained,
FFT, sLoRA, per-repo LoRA, and Code2LoRA-
0 500 1000 1500 2000 2500 3000 3500 4000
T oken Count010002000300040005000600070008000Number of QnA PairsPrefix-Only Input Length
512 tokens
2048 tokens
4096 tokens
0 2000 4000 6000 8000 10000 12000 14000 16000
T oken Count010002000300040005000Number of QnA PairsDRC + Prefix Input Length
2048 tokens
4096 tokens
8192 tokens
16384 tokensFigure 3: Token length distributions for prefix-only (left)
and DRC+prefix (right) input formats across all splits.
Vertical dashed lines mark common context window
sizes. Prefix-only inputs are compact (median 224to-
kens), while DRC+prefix inputs have a heavy right tail
requiring larger context windows.
Static. The supplementary materials contain the
full table; aggregate distributions and the data-
sparsity scatter for per-repo LoRA are summarized
in Figures 6 and 7.
B.8 Privacy and Content Review
The dataset contains only non-test source files
and test files from public open-source projects
with permissive licenses, copied verbatim from
the upstream repositories. No private reposito-
ries, user accounts, commit messages, issue bod-
ies, or PR discussions are included; identifying
information is therefore limited to whatever the
upstream maintainers chose to embed in public
Python source (e.g., author docstrings, copyright
headers in LICENSE files, contact emails inside
module-level docstrings of well-known libraries).
We did not perform automated PII scrubbing be-
cause (i) the dataset is a redistribution of already-
14

Table 7: Token length statistics across the 62,294 static-track QnAs (Qwen2.5-Coder-1.5B tokenizer). Repo size is
the total token count of all Python source files per repository (repeated per pair). DRC statistics are over the 64.1%
of pairs with resolvable dependency context.
Component Mean Med. Std p75 p95 p99 Max
Repo size 284,509 165,376 363,914 311,729 1,028,703 1,865,509 2,994,853
DRC context†1,900 517 6,243 1,634 7,849 20,826 574,001
Prefix 360 224 566 396 992 2,588 27,171
Target 4.8 3.0 10.2 5.0 14 43 290
†Computed over 39,902 pairs (64.1%) with resolvable dependency context.
public, license-permitted source, and (ii) any ag-
gressive scrubbing would alter the very identifiers
(function names, fixture names, class names) that
the benchmark task requires the model to predict.
We did not observe offensive content in random
spot checks of the dataset, which is consistent with
the high-star permissive-license selection criterion;
users who identify problematic content in any of
the released repositories may file an issue against
the dataset repository for removal.
C Additional Ablation Studies
C.1 RAG with Differentk
We sweep the number of retrieved chunks kand
chunk size to confirm that the RAG result in the
main table (k=3, 512-token chunks) is the strongest
configuration for our setting, and that the degrada-
tion under RAG is not an artifact of a particular bud-
get. Pretrained RAG monotonically degrades with
kon both CR and IR (Table 8, top): going from
k=3 tok=10 at 512-token chunks drops CR EM by
3.4 pp and IR EM by 2.7 pp. Smaller (256-token)
chunks at the same retrieval budget are uniformly
worse than the 512-token variant. Combining RAG
with trained adapters (Table 8, bottom) helps FFT
mildly but hurts sLoRA, consistent with the finding
that retrieval-injected tokens shift the distribution
away from what the adapter was trained on. The
largest single kused at training and reported in
Table 2 is therefore the optimal RAG configuration,
not a strawman.
D Implementation Details
This section documents the dependency-resolved
context (DRC) extraction algorithm and the exact
hyperparameters used to train every method in Ta-
bles 2–4. All training and evaluation runs use a
single H100 80 GB GPU; total compute is summa-
rized at the end of the section.D.1 Dependency-Resolved Context
Construction
DRC takes a test prefix and, via static import analy-
sis, returns the function and class definitions reach-
able from its imports. We describe the resolution
strategy, the relevance-aware compression that fits
results into the adaptive 8K-token budget, and the
empirical coverage on RepoPeftBench.
Import resolution strategy.For each import in
the test prefix:
1.Parse using AST with fallback regex for syn-
tax errors.
2.Resolve the module to a file path, trying multi-
ple source roots: repository root, src/ ,lib/ ,
package directories with__init__.py.
3.For relative imports, resolve relative to the test
file’s location.
4.If the imported name is used in the test prefix,
extract its definition (function or class) from
the source file via AST.
Coverage.DRC context is available for 70.3%
of CR-test pairs, 64.7% of IR-test pairs, and ap-
proximately 64% of training pairs. When present,
DRC adds a median of 517 tokens (mean 1,900,
p95 7,850 tokens). Pairs with no resolvable im-
ports (e.g., testing third-party libraries or built-in
functions only) receive no DRC augmentation and
are trained and evaluated on the plain prefix.
D.2 Detailed Architecture Diagrams
Figure 4 and Figure 5 expand the overview in Fig-
ure 1 with step-by-step training and inference de-
tails for each usage scenario.
D.3 Training Details
Table 9 lists the optimizer, schedule, sequence
length, batch size, and adapter configuration for ev-
ery trained baseline (FFT, sLoRA, per-repo LoRA)
and for Code2LoRA-Static (with and without
15

Table 8: RAG ablation over chunk size and kon CR and IR test. Top: pretrained + RAG; bottom: trained models +
RAG at inference.
CR Test IR Test
ChunkkEM EditSim CB EM EditSim CB
Pretrained + RAG
512 3 39.7 0.516 0.556 42.1 0.544 0.581
512 5 37.5 0.486 0.527 41.0 0.524 0.559
512 10 36.3 0.469 0.509 39.4 0.521 0.574
256 5 35.0 0.457 0.499 38.0 0.489 0.528
256 10 33.0 0.428 0.470 35.5 0.453 0.494
Trained + RAG
256 5 (FFT) 53.9 0.703 0.688 56.8 0.731 0.713
256 5 (sLoRA) 37.0 0.588 0.586 39.0 0.620 0.609
model.py
utils.py
config.py
...Frozen
Embedder
Qwen3-Emb-0.6B
precomputed offlinef1
f2
fK...Weighted
Mean+Max
importance-weightederepo
∈R2048MLP
Trunk
2-layer GELU,H=512
L2-norm·√
HhHeadA
t
HeadB
t
t∈{q,k,v,o,up,gate,dn}
tanh·exp(s t)At
Bt×7types,
shared×28layers
assert res == ?
test prefixFrozen LLM
Qwen2.5-Coder-1.5B
W′=W+α
rBtAtexpected_val
predicted target∇θLLM
1 Repository Encoding(offline)2 Hypernetwork (Code2LoRAHead)
3 Adapted Inference
Figure 4: Detailed Code2LoRA-Static architecture.(1)Repository-level context is encoded by a frozen embedding
model (Qwen3-Embedding-0.6B) and aggregated into a 2048-dim repository embedding erepo; the result is stored in
the dataset and consumed verbatim at training time—gradients never flow back through the embedder.(2)A shared
MLP trunk (2-layer GELU, hidden H=512 ) maps erepoto a hidden representation h(L2-normalized, rescaled
by√
H); separate HeadA
m,HeadB
mheads emit Am,Bmfor each of the 7 projection types via tanh·exp(s m)
scaling with a clamped learnable log-scale sm. The same (Am,Bm)pair is shared across all 28 transformer layers.
(3)Generated LoRA weights are injected into the frozen LLM via W′=W+α
rBmAm. Only the hypernetwork
parametersθare trained via the language-modeling loss (dashed red); the LLM and embedder stay frozen.
training-time DRC) on the static track. All methods
share the same backbone (Qwen2.5-Coder-1.5B,
bf16), the same optimizer (AdamW, cosine sched-
ule, weight decay 0.01), and roughly the same ef-
fective compute budget; the methods differ in LR,
sequence length, and (for adapter methods) LoRA
rank, dropout, and module coverage. Code2LoRA-
Static uses an 8K sequence length to accommo-
date dependency-resolved context when enabled;
Code2LoRA-Evo truncates BPTT every 16 com-
mits and uses a 4K sequence length per step (§D.5).
D.4 Compute Resources
All experiments were conducted on a single
NVIDIA H100 80 GB GPU per job. Total GPUhours: FFT variants ∼6 h, sLoRA variants ∼10 h,
Code2LoRA-Static (no DRC) ∼17 h, Code2LoRA-
Static+DRC ∼18 h, per-repo LoRA ( ∼0.1 h per
repo×409 repos) ∼41 h, and evaluation jobs
∼30 h. Code2LoRA-Evo training requires an addi-
tional∼24 h on the commit-derived dataset.
D.5 Hypernetwork Training
Hyperparameters
The Code2LoRA-Static variant uses input dim
2,048 (mean +max repository embedding), trunk
hidden H=512 , LoRA rank r=16 ,α=32 , and
all seven attention/MLP projection types shared
across all 28transformer layers. Code2LoRA-
Evo uses a 1-layer GRU with hidden size
16

repo t=0repo state @ commit 0
e(0)
repo
∈R2048Repo-State
Initializer
Linear→GELU→LayerNormh0∈RHdiff∆ 1
diff∆ 2
...
diff∆ Tper-commit diffs
Frozen
Embedder
Qwen3-Emb-0.6B
precomputed offlinee1
e2
...
eT∈R2048
GRU GRU ··· GRU h1 h2 hT ∈R2048each step: input projection (Linear+LayerNorm)→GRU recurrence
truncated BPTT (detach everyK=16steps)
LayerNorm
ctx= LN(h T)MLP
Trunk
2-layer GELU,H=1024
L2-norm·√
HHeadA
t
HeadB
t
t∈{q,k,v,o,up,gate,dn}
tanh·exp(s t)At
Bt
×7types, shared×28layersFrozen LLM
Qwen2.5-Coder-1.5B
W′=W+α
rBtAt
assert res == ?
test prefixLCE
target tokenscontexth T
∇ϕLCE: updateϕ={repo-state init, GRU, Code2LoRAHead}; LLM and embedder remain frozen1 Offline Embedding
2 Repo-State Init
3 Repository GRU
4 LoRA Head (shared w/ Fig. 4) 5 Adapted LLM
frozen
trainable (ϕ)
- - - gradient flow
Figure 5: Detailed Code2LoRA-Evo architecture and training procedure.(1)Per-commit production-code
diffs ∆tand the initial repository snapshot are encoded by the shared frozen embedder into 2048-dim vectors
{et}T
t=1ande(0)
repo; the resulting embeddings are stored in the dataset.(2)A small repo-state initializer (Lin-
ear→GELU →LayerNorm) maps the static snapshot e(0)
repoto the initial hidden state h0∈R2048.(3)A 1-layer
GRU walks the chronological diff sequence; each step projects etwith a Linear + LayerNorm and applies the GRU
recurrence to produce ht. Truncated BPTT detaches the hidden state every K=16 steps.(4)The final state hTis fed
(after LayerNorm) into Code2LoRA-Evo’s LoRA-generation projection head (analogous in design to Code2LoRA-
Static’s; Figure 4): a 2-layer GELU trunk with L2-norm rescaling, plus per-module-type HeadA
m/HeadB
moutput
heads with tanh·exp(s m)scaling. The resulting (Am,Bm)are shared across all 28 transformer layers per type.
(5)Generated LoRAs are injected into the frozen LLM ( W′=W+α
rBmAm); training minimizes the cross-
entropy loss on the assertion target. Gradients (dashed red) flow through the projection head, GRU, and repo-state
initializer; the LLM and embedder stay frozen.
2,048 and a smallrepo-state initializer(Lin-
ear→GELU →LayerNorm) that maps the ini-
tial 2048-dim repository embedding to h0;
the LayerNorm-ed final state hTfeeds into
Code2LoRA-Evo’s projection head (analogous in
design to Code2LoRA-Static’s, with trunk hid-
den1,024 vs.512). Truncated BPTT detaches
the hidden state every K=16 commits. Both vari-
ants are trained for 3epochs with AdamW (cosine
schedule, weight decay 0.01): Code2LoRA-Static
at LR 1×10−4and max sequence length 8,192 ;
Code2LoRA-Evo at LR 5×10−5and max sequence
length 4,096 . Best checkpoint is selected by CR-
val loss.E OOD Evaluation Caveats
Two confounds in Table 4 are worth surfacing.(i)
Prefix shape.Table 4 uses commit-derived prefixes
(median ∼7.9 KB), identical to Table 3 andnotthe
short static prefixes ( ∼0.9 KB) of Table 2; OOD-
vs-Table 3 deltas are therefore unconfounded by
prefix shape, and only the underlying repositories
differ.(ii) Target length.OOD assertion targets are
systematically shorter (median 7 chars) than CR/IR-
test (12–13 chars), inflating exact-match credit on
every OOD row uniformly; sLoRA’s OOD EM
(72.3%) substantially exceeds its in-distribution
EM (55.1/61.3%) for this reason. The within-
table Code2LoRA-Evo vs. sLoRA gap on OOD
is+1.8 pp—narrower than the in-distribution gap
17

Table 9: Training hyperparameters. The “+DRC” column shares all settings with Code2LoRA-Static and adds a
4K-token dependency-resolved context budget injected ahead of the prefix. The commit-derived results in Tables 3–4
use analogous V2 trainers (1 epoch, batch 1, grad-accum 16, max seq 4,096); see §D.5 and the released code for full
details.
FFT sLoRA pLoRA Code2LoRA-Static +DRC
LR 2e-5 5e-5 2e-4 1e-4 (same)
Epochs 3 5 3 3 (same)
Max seq len 2,048 2,048 2,048 8,192 (same)
Batch size 4 4 4 1 (same)
Grad accum 8 8 4 8 (same)
Effective batch 32 32 16 8 (same)
LoRA rank — 16 16 16 (same)
LoRA alpha — 32 32 32 (same)
LoRA dropout — 0.1 0.0 — —
Warmup ratio 0.05 0.10 0.10 0.03 (same)
Max DRC tokens — — — — 4,096
Precision bf16 bf16 bf16 bf16 bf16
Optimizer AdamW AdamW AdamW AdamW AdamW
LR schedule cosine cosine cosine cosine cosine
(+5.2/+ 3.2 pp, Table 3) but always positive, so
Code2LoRA-Evo remains the best method on ev-
ery split under matched inputs. We interpret the
narrower OOD margin as evidence that part of
the streaming advantage is recovered from within-
distribution edit patterns seen at training: the OOD
repositories were created strictly after the scrape
cutoff, so their early-life commit trajectories were
never observed.
F Broader Analysis
This section complements the main-paper analy-
sis with the supporting figures and tables: per-
repository variance and data-sparsity scatter (§F.1),
the repository-count scaling curve (§F.2), the per-
commit-position trend (§F.3), structural analysis of
the generated LoRAs (§F.4), the LiveCodeBench-
style error taxonomy and qualitative examples
(§F.5, §F.6), DRC coverage broken out by availabil-
ity (§F.7), and the efficiency comparison (§F.8).
F.1 Per-Repository Performance and Data
Sparsity
The aggregate IR-test EM in Table 3 hides sub-
stantial repository-to-repository variance for per-
repo LoRA. Across the 389 repositories evalu-
ated by every method, per-repo LoRA EM spans
the full [0,100]% range with a median of 62.5%
and a standard deviation of 20.9; on 10.5%
of repositories ( 41/389 ) per-repo LoRA scores
belowthe pretrained baseline (per-repo median
30.7% ). The dominant driver is training-data
availability: per-repo LoRA overfits to small in-repo datasets and frequently regresses below the
unadapted backbone whenever the in-repo train-
ing pool is thin. Code2LoRA-Static sidesteps
this failure mode through cross-repository knowl-
edge transfer: the hypernetwork learns shared pat-
terns from 409 repositories (39,612 examples) and
regularizes the generated adapters, yielding the
tighter per-repository EM distribution shown in
Figure 6 ( σ=16.8 for Code2LoRA-Static and 15.8
for Code2LoRA-Evo vs. 20.9 for per-repo LoRA;
only1.3% and1.8% of repositories fall below pre-
trained, respectively) and the flatter EM-vs-data-
size profile shown in Figure 7.
F.2 Repository-Count Scaling
To understand whether the hypernetwork
benefits frombreadth(more distinct repos-
itories) or merelydepth(more pairs), we
sweep the number of training repositories at
{10,25,50,100,150,200,409,500,623} while
keeping the per-repo data budget and training
schedule fixed. Two findings emerge. First, with
only 10 repositories ( ∼2% of the full training
set), Code2LoRA-Static already reaches 57.7%
CR-test EM—above FFT trained on the full data
(51.4% , Table 2). Second, CR-test EM scales
log-linearly with repository count up to ∼200
repositories and is essentially flat between 200 and
623, suggesting that breadth saturates around a few
hundred distinct codebases at the current backbone
size. Figure 8 plots the curve; Table 10 reports the
underlying numbers.
18

PretrainedRAG (k=3)DRCtext2lorasLoRA
Per-Repo LoRACode2LoRA-StaticCode2LoRA-Evo020406080100Per-Repository Exact Match (%)
med=30.7 med=23.1 med=32.2 med=45.6 med=63.4 med=62.5 med=62.5 med=66.7Figure 6: Per-repository EM distribution on the IR-test
split of RepoPeftBench (Table 3 checkpoints; n=389
repositories common to all methods). Each violin
shows the full distribution of per-repository EM for
one method; the inner box reports the IQR and the
white dot marks the median. Code2LoRA-Static (me-
dian62.5% ,σ=16.8 ) and Code2LoRA-Evo (median
66.7% ,σ=15.8 ) achieve consistently high performance
with substantially lower variance than per-repo LoRA
(median 62.5% ,σ=20.9 ); per-repo LoRA falls below
the pretrained baseline on 10.5% of repositories ver-
sus only 1.3% and1.8% for Code2LoRA-Static and
Code2LoRA-Evo, demonstrating the regularizing effect
of cross-repository knowledge transfer.
Table 10: Effect of training-repository count on CR-test
EM.
Training Repos % of Full CR Test EM (%)
10 2% 57.7
25 4% 60.9
50 8% 60.9
100 16% 61.3
150 24% 61.5
200 32% 62.2
409 66% 63.8
500 80% 61.2
623 100% 63.5
F.3 Per-Commit Position Trend
To verify that Code2LoRA-Evo’s evolution-track
advantage is not driven by a few late-history com-
mits, we plot CR-test EM as a function of each
commit’s normalized position within its reposi-
tory’s chronological history. For every repository
the timeline is rescaled to [0%,100%] (so0%is
the first scored commit and 100% the last), QnAs
are bucketed into 5%-wide bins, and each bin’s
score is the QnA-weighted mean EM across that
bin. This collapses short and long repository his-
tories onto a single axis and visualizes the entire
lifecycle of every repository rather than only its first
commits. Figure 9 shows that Code2LoRA-Evo’s
lead persists across the entire history; the snapshot-
based methods (Code2LoRA-Static, sLoRA, FFT)
102 3×1014×1016×101
Per-repo training examples (log)020406080100Per-repo EM (\%)Per-repo LoRA: EM vs.\ training-set size
linear fit, slope=14.7/decade
N_train = 50Figure 7: Per-repo LoRA EM vs. training-set size on
IR test. Repositories with fewer than 50 training pairs
frequently underperform the IR-test pretrained baseline
(46.8%), while Code2LoRA-Static maintains stable per-
formance regardless of per-repo data availability.
101102103
Number of Training Repositories58596061626364Exact Match (%)
57.760.9 60.961.361.562.263.8
61.263.5Code2LoRA Scaling with Repository Diversity
Log-linear (R2=0.721)
Power law (R2=0.719)
101102103
Number of Training Repositories3.6×1013.7×1013.8×1013.9×1014×1014.1×1014.2×101Error Rate (%)Error Rate Scaling (log-log)
E=43.9N0.027
Figure 8: CR-test EM as a function of training reposi-
tory count. Code2LoRA-Static benefits from repository
diversity, with performance improving log-linearly.
exhibit the steepest downward drift, consistent with
the staleness mechanism described in §6.2, while
Code2LoRA-Evo stays flattest.
F.4 Structure of the Generated LoRAs
A natural question is whether the hypernetwork
emits genuinely repository-specific adapters or
whether it converges to a single mean adapter that
happens to behave well on average. We probe this
from two angles.Diversity of adapters: pairwise
cosine similarities between the 52 mean-centered
CR-test LoRAs (659K-dim flattened) span the full
[−1,+1] range with mean 0.01 and standard de-
viation 0.94, so the adapters are not a collapsed
mean.Semantic structure: a t-SNE projection of
those adapters (Figure 10) shows that repositories
with similar codebases cluster together and that
clusters carry coherent EM ranges, indicating that
the hypernetwork’s adapter manifold is smooth and
semantically organized rather than arbitrary.Per-
module concentration: a comparison of per-module
weight norms (Figure 11) reveals that Code2LoRA-
Static concentrates updates on a repository-specific
subset of modules (typically gate andupprojec-
19

0 20 40 60 80 100
Commit position within the repository's CR-test history (\% of total)203040506070Exact-Match accuracy (\%)
CR-test accuracy vs. normalized commit position (51 held-out repos, full history)
Pretrained
RAGDRC
Single LoRAT ext2LoRA
Code2LoRACode2LoRA-EvoFigure 9: CR-test exact-match vs. normalized commit
position (51 held-out repositories, commit-derived pre-
fixes). Each repository’s timeline is scaled to 0–100%;
points are qna-weighted means per 5% bin.
t-SNE Component 1t-SNE Component 2
inline-snapshot
LunaVox
mkslidesMini-Agent
chispaALNSpyfilesystem2Hexis
anyio
nitpickaisuite
cloudproxyintentkit
django-hijacketh-account
spacy-transformetrackio
graphql-core
mcpadapt
html5lib-pythongcalclipgqueuer
Watsonkapre deprecatedhivemindwsgidavMM-REACT
tinydbitableskopf
octodns
bitmcpm.sh
pfhedgemountaineer
pyamgpytorch-frame
pipxcmd2
pendulum
luma.led_matrixlleavesytmusicapisupabase-py
libtmuxpgsyncvanna
django-ai-assist
pxwechatpydjango-modern-ret-SNE of Generated LoRA Adapters (mean-centered, n=52 repos)
Color = per-repo Exact Match (%)
020406080100
CR T est Exact Match (%)
Figure 10: t-SNE of generated LoRA adapters for 52
CR-test repositories (PCA pre-reduction to 50 dims,
then t-SNE). Color indicates per-repo Exact Match (%).
Repositories with similar codebases tend to cluster to-
gether, and clusters show coherent EM ranges, demon-
strating that the hypernetwork learns a smooth, semanti-
cally meaningful adapter manifold.
tions), whereas FFT+DRC applies a uniform delta
across all modules—a qualitative difference that
helps explain Code2LoRA-Static’s stronger cross-
repo transfer.
F.5 Error Analysis
We classify all 2,321 incorrect CR-test predictions
of Code2LoRA-Static using a LiveCodeBench-
inspired taxonomy (Jain et al., 2025). The break-
down in Figure 12 shows that no single failure
mode dominates:wrong literal( 31.0% ) andsyntax
error( 28.0% ) together account for ∼60% of er-
rors, with the remainder split amongtype mismatch
(19.0% ),near-miss( 10.8% ), andwrong identifier
(10.2% ); hallucinations and empty outputs are each
Down Gate K O Q Up Vmkslides (12%)
nitpick (34%)
gcalcli (50%)
pyamg (56%)
pgsync (63%)
spacy-transform (65%)
pyfilesystem2 (67%)
cmd2 (68%)
pipx (70%)
deprecated (77%)
vanna (83%)
LunaVox (88%)Code2LoRA: Per-Repo LoRA Weight Norms (normalized)
Down Gate K O Q Up VFFT+DRCFFT+DRC: Uniform Weight Delta Norms (normalized)
0.00.20.40.60.81.0
Relative norm
0.00.51.0
Relative normFigure 11: Comparison of per-module weight norms.
Top: Code2LoRA-Static generates repo-specific LoRA
adapters with varying weight distributions across mod-
ule types. Bottom: FFT+DRC applies a uniform weight
delta. Code2LoRA-Static’s structured, repo-specific
adaptations explain its stronger cross-repo performance.
0 100 200 300 400 500 600 700 800
Number of Incorrect PredictionsWrong Literal  (31.0%)
Syntax Error  (28.0%)
Type Mismatch  (19.0%)
Near-Miss  (10.8%)
Wrong Identifier  (10.2%)
Hallucination  (0.7%)
Empty/Truncated  (0.3%)719
651
440
251
237
16
7Error classification on CR test (2,321 incorrect predictions)
Figure 12: Error classification of Code2LoRA-Static
failures on CR test (2,321 incorrect predictions), fol-
lowing a LiveCodeBench-inspired taxonomy. Wrong
literal (31.0%), syntax error (28.0%), type mismatch
(19.0%), near-miss (10.8%), wrong identifier (10.2%);
hallucinations and empty outputs are<1%each.
under 1%. The wrong-literal class is dominated by
numeric tests where the correct value depends on
runtime state (e.g., expression-valued assertions);
the near-miss class corresponds to syntactically
valid completions that differ from the reference
only in trailing punctuation or single tokens.
F.6 Qualitative Examples
We complement the aggregate numbers with quali-
tative views of CR-test predictions. Figure 13 pairs
two representative successes frominline-snapshot
andALNSwhere Code2LoRA-Static recovers repo-
specific identifiers and conventions that pretrained
Qwen2.5-Coder and full fine-tuning miss. We then
zoom in on a representative case with an expanded
20

layout that shows the metadata header, full test
prefix, retrieved repository context, and side-by-
side per-method predictions: Figure 14 illustrates
thecontext-quality bottleneckcase, where retrieval
surfaces the relevant class definition but only the
parametric methods complete the value-level rea-
soning step.
Beyond the two short panels of Figure 13, we
feature one additional commit-derived CR-test case
in full detail below. The case is drawn from the sup-
plementary file positive_analysis.md (10 cases
total, 5 per category) and demonstrates the comple-
mentarycontext-quality bottleneckphenomenon:
RAG@3 / DRC retrieval surfaces the exact class
definition that determines the assertion’s outcome,
yet pretrained, RAG, DRC, and sLoRA all fail to
translate that prepended evidence into the correct
prediction; only the hypernetwork variants com-
plete the value-level reasoning step from the re-
trieved evidence.
We further feature four detailed qualitative ex-
amples drawn from the commit-derived IR-test
set (GRU dataset variant; the source HTML re-
port report_gru_ir_test_qnas.html samples
300 QnAs across 18 methods). Each figure shows
the full test prefix, the actual DRC and RAG@3
contexts that were injected at evaluation time
(trimmed to the most relevant signatures and class
initializers; non-essential method bodies are elided
with “ ...”), and the per-method predictions for
the five methods that the report tracks for Table 3:
Code2LoRA-Static, Code2LoRA-Evo, RAG, DRC,
and Text2LoRA. Figures 15 and 16 are easy cases
where the local prefix already exposes the com-
pletion pattern and retrieval merely corroborates
it. Figure 17 is aretrieval-precisioncase: only
DRC retrieves the discriminating -> bool signa-
ture, RAG misses it and collapses onto the n-gram-
likely is 1 ; the parametric Code2LoRA variants
succeed without context. Figure 18 is aretrieval-
degeneracycase: DRC retrieves the literal answer
JobOutcome.abandoned in a docstring and RAG
retrieves the JobOutcome enum class plus the dot-
ted access pattern (one inference hop away from
the answer), yet both methods collapse onto a FIM-
token artifact at the very first generated token; only
the methods that bake the repository signal into the
parameters complete the assertion.Table 11: CR-test EM partitioned by DRC availability.
DRC helps only when context is resolvable (+1.8 pp vs.
pretrained); Code2LoRA-Static performs consistently
regardless, showing the repository embedding captures
information beyond import resolution.
CR Test EM (%)
Method w/ DRC (70.3%) w/o DRC (29.7%)
Pretrained 48.1 51.5
Dep.-Resolved Context 49.9 44.2
Code2LoRA-Static67.0 66.9
F.7 Effect of Dependency-Resolved Context
Coverage
DRC is only meaningful when the imports in the
test prefix actually resolve to repository code. On
CR-test, 70.3% of pairs (4,511/6,414) have non-
empty DRC, while the remaining 29.7% (1,903
pairs) import only from the standard library or
third-party packages and therefore receive no DRC
augmentation. To check whether DRC’s modest
aggregate gain reflects a strong effect on the resolv-
able subset or a uniformly weak effect, we partition
CR-test by DRC availability in Table 11. DRC
adds +1.8 pp over pretrainedonlyon the resolv-
able subset and is actively destructive ( −7.3 pp) on
the no-DRC subset, where the model is forced to
attend to empty context slots. Code2LoRA-Static
is essentially flat across the two partitions ( 67.0 vs.
66.9 EM), showing that the learned repository em-
bedding captures information beyond what import-
resolved definitions provide.
F.8 Deployment Efficiency
Table 12 compares the deployment cost of ev-
ery method along three axes that matter when
scaling to many, continuously-changing reposito-
ries: extra inference tokens, per-repository adap-
tation time, and incremental storage on top of
the shared frozen base model. RAG and DRC
both incur per-query token overhead in the 500–
2,000 range, while FFT requires ∼4 h of train-
ing and a full 3.1GB model copy per repository.
Code2LoRA-Static and Code2LoRA-Evo sit at the
other extreme: zero extra inference tokens, sub-
10 ms adapter generation, and bounded extra stor-
age ( 679MB for the Code2LoRA-Static hyper-
network shared across all repositories, 65MB for
the Code2LoRA-Evo variant, bothindependentof
repository count). Per-repo LoRA matches the in-
ference cost of Code2LoRA but requires ∼5 min of
21

(a)inline-snapshot(CR)✓Success
s2 = s.run(reported_flag)
asserts2.source == ???
REFERENCEs2.source
CODE2LORA-STATICs2.source✓
FFTs.source✗
PRETRAINEDs.source✗
Code2LoRA-Static captures thes2naming pattern; baselines default tos.(b)ALNS(CR)✓Success
select.update(Zero(), 0, 0, 1)
assert_almost_equal(
select.destroy_weights[0], ???
REFERENCEexpected[0])
CODE2LORA-STATICexpected[0])✓
FFTexpected)✗
PRETRAINED1)✗
The repo usesexpected[i]arrays for ground truth.
Figure 13: Qualitative examples from CR test. Each panel shows a test prefix with the completion target ( ???),
ground-truth reference, and model predictions.(a)–(b): Code2LoRA-Static correctly infers repo-specific identifiers
and conventions that pretrained Qwen2.5-Coder and full fine-tuning miss.
training per new repository and 32MB per reposi-
tory, neither of which scales.
G Discussions
We organize the discussion around three central
questions raised by the framework.
Q1. Why parameters over context?For asser-
tion completion the answer depends on a short
window of repository-specific symbols rather than
long-range token-level reasoning. RAG and DRC
inject related but locally noisy tokens that shift the
model’s distribution; FFT collapses repository sig-
nal into one “average” specialization. Code2LoRA
routes the same information into per-repository
LoRAparameters, conditioning the model at ev-
ery layer without paying tokens or sharing capacity
across repositories—explaining the consistent gaps
to FFT, DRC, and pLoRA on both IR and CR (Ta-
ble 2).
Q2. Why two usage scenarios rather than one?
Thehow/whenframing admits two ends: one-shot
snapshot adaptation vs. incremental refresh under
evolution. Code2LoRA-Static is sufficient—and,
in raw CR/IR EM, optimal—on the static track (Ta-
ble 2): the same code embedding goes into a single
forward pass and out comes one LoRA per module
type, with no recurrence and no commit history to
maintain at deployment. Real codebases, however,
do not stand still: the bursty commit pattern in Fig-
ure 2 shows that snapshot adaptation accumulates
staleness as a repository accumulates edits, and Ta-
ble 3 shows the same Code2LoRA-Static model
dropping back to parity with the single-adapter
baseline once the evaluation prefix reflects commit-
time state. Code2LoRA-Evo is the shared-head
extension for this drift: the static head is reused,but the head’s context vector becomes a recurrent
hidden state updated at each recurrent step with
amortized constant work per update. The two usage
scenarios therefore correspond to stable-codebase
comprehension vs. active development on evolving
codebases, not competing ablations.
Q3. Where does Code2LoRA-Evo’s edge come
from?Code2LoRA-Evo reuses Code2LoRA-
Static’s LoRA-generation head; the only added ca-
pacity is a GRU recurrence over sequential diff em-
beddings before the shared MLP trunk. The empiri-
cal lead (Table 3, +5.2 pp commit-CR EM over sin-
gle LoRA) is the value of aggregating edit history
into the hypernetwork context commit-by-commit,
rather than asking a single snapshot embedding
to capture both code and its history. Results on
the temporal OOD holdout in RepoPeftBench cor-
roborate generalization (§6.3). Appendix Figure 9
corroborates this: Code2LoRA-Evo’s advantage
persists across the entire commit timeline, with the
shallowest staleness drift among trained adapters.
22

Detailed qualitative example:nolar/kopf(Code2LoRA-exclusive, CR test, commit-derived)
QnA metadata
REPOSITORYnolar/kopfCOMMITSHAd848601b0df0. . .
COMMIT POSITION19.2%(55 / 287) PYTHON FILES131
REPO SIZE∼423K chars /∼120K tok ASSERTION FAMILYpytest.raises(exception class)
TEST LOCATIONtests/basic-structs/test_resource.py:7:9
Test prefix (model input)
importpytest
fromkopf.structs.resourcesimportResource
...
deftest_no_args():
withpytest.raises( ???
Retrieved repository context
DRC (import-resolved, 4K-token budget):
# kopf/structs/resources.py
classResource(NamedTuple):
group:str
version:str
plural:str
@property
defname(self):
returnf'{self.plural}.{self.group}'
RAG@3 (top-3 retrieved 512-token chunks):surfaces the identical Resource NamedTuple definition
plus two unrelated chunks (truncated; full text in supplementary).
Per-method predictions and exact-match outcome
Method Prediction EM
REFERENCETypeError)
Pretrained (Qwen2.5-Coder-1.5B)ValueError)✗
RAG (k=3)ValueError)✗
Dependency-Resolved ContextValueError)✗
Single LoRA (sLoRA)ValueError)✗
Code2LoRA-StaticTypeError)✓
Code2LoRA-EvoTypeError)✓
The retrieved context surfaces the exact Resource NamedTuple with three required fields, so the evidence to deduce that
Resource() with zero arguments raises TypeError is in the prompt. Yet pretrained, RAG, DRC, and sLoRA all default
toValueError —the more common pytest.raises idiom—showing that input-side methods do not reliably execute the
type-level reasoning hop even when the relevant evidence has been retrieved. Both Code2LoRA variants predict TypeError
because the repository’sNamedTuple-vs-class conventions were distilled into the LoRA-generation step.
Figure 14: Qualitative example of the QnA from the CR test set
23

Detailed qualitative example:fla-org/flash-linear-attention(IR test, commit-derived)
QnA metadata
REPOSITORYfla-org/flash-linear-attentionCOMMITSHAd62e316ea88b. . .
COMMIT POSITION277 / 409 (training-window) IN-REPO SPLITtrain
ASSERTION FAMILYassert_close(...)– repository utility comparing two tensors with a tolerance ratio
TEST LOCATIONtests/ops/test_kda.py::test_naive_chunk, line 73
Test prefix (model input, trimmed)
fromfla.ops.kda.naiveimportnaive_chunk_kda, naive_recurrent_kda
fromfla.utilsimportIS_INTEL_ALCHEMIST, assert_close, device
...
deftest_naive_chunk(B, T, H, D, scale, gate_logit_normalizer, dtype):
...
ref, ref_ht = naive_recurrent_kda(q=..., k=..., v=v.clone(),
g=g.clone(), beta=beta.clone(),
scale=scale, initial_state=h0.clone(),
output_final_state=True)
tri, tri_ht = naive_chunk_kda(q=..., k=..., v=v.clone(),
g=g.clone(), beta=beta.clone(),
scale=scale, initial_state=h0.clone(),
output_final_state=True)
assert_close("o", ref, tri, 0.005)
assert_close("ht", ref_ht, tri_ht, ???
Retrieved repository context (trimmed)
DRC (import-resolved):
# fla/ops/kda/naive.py
defnaive_recurrent_kda(q, k, v, g, beta,
scale=None, initial_state=None,
output_final_state=False): ...
defnaive_chunk_kda(q, k, v, g, beta,
scale=None, initial_state=None,
output_final_state=False, chunk_size=64): ...
# fla/utils.py -- this is the discriminating signature
defassert_close(prefix, ref, tri, ratio,
warning=False, err_atol=1e-6):
... # the 4th positional argument is the tolerance``ratio''
RAG@3 (top-3 retrieved chunks):unrelated benchmarks/ops/benchmark_kda.py kernel-
benchmarking loop (truncated; noassert_closesignature is included).
Per-method predictions and exact-match outcome
Method Prediction EM
REFERENCE0.005)
Code2LoRA-Static0.005)✓
Code2LoRA-Evo0.005)✓
RAG (k=3)0.005)✓
Dependency-Resolved Context0.005)✓
Text2LoRA0.005)✓
The completion repeats the third positional argument of the immediately-preceding assert_close call. DRC additionally
retrieves the assert_close(prefix, ref, tri, ratio, ...) signature, which confirms that the open slot is the ratio
parameter; RAG’s three chunks are unrelated benchmarking code. The local pattern is strong enough that all five methods
succeed unconditionally—an example of the lower-bound regime where context injection neither helps nor hurts.
Figure 15: Qualitative example of a QnA from the IR test set (GRU dataset variant). Trivial in-prefix repetition: the
previous line already exhibits the completion pattern assert_close(..., 0.005) , and DRC additionally surfaces
the corroboratingassert_closesignature.
24

Detailed qualitative example:se2p/pynguin(IR test, commit-derived)
QnA metadata
REPOSITORYse2p/pynguinCOMMITSHA3f25634f7ec7. . .
COMMIT POSITION932 / 1144 (late history) IN-REPO SPLITval
ASSERTION FAMILYbareassert, comparing a registry return value to an auto-increment integer id
TEST LOCATIONtests/instrumentation/test_tracer.py::test_line_registration, line 61
Test prefix (model input, trimmed)
frompynguin.instrumentation.tracerimport(
LineMetaData, SubjectProperties, ...
)
...
deftest_line_registration(subject_properties: SubjectProperties):
assertsubject_properties.register_line(
LineMetaData(0, "foo", 42)) == 0
assertsubject_properties.register_line(
LineMetaData(0, "foo", 43)) == ???
Retrieved repository context (trimmed)
DRC (import-resolved):
# src/pynguin/instrumentation/tracer.py
classLineMetaData:
"""Stores meta data of a line."""
code_object_id:int
file_name:str
line_number:int
...
RAG@3 (top-3 retrieved chunks; the discriminating method body):
# src/pynguin/instrumentation/tracer.py
classSubjectProperties:
existing_lines:dict[int, LineMetaData] = field(default_factory=dict)
...
defregister_line(self, meta: LineMetaData) ->int:
ifmetanot inself.existing_lines.values():
line_id =len(self.existing_lines) # auto-increment
self.existing_lines[line_id] = meta
else:
... # return the existing id for an already-registered line
returnline_id
Per-method predictions and exact-match outcome
Method Prediction EM
REFERENCE1
Code2LoRA-Static1✓
Code2LoRA-Evo1✓
RAG (k=3)1✓
Dependency-Resolved Context1✓
Text2LoRA1✓
The previous line already established the pattern register_line(...) == 0 ; the canonical next id is therefore 1. RAG ad-
ditionally retrieves the SubjectProperties.register_line body, which makes the auto-increment convention ( line_id
= len(self.existing_lines) ) explicit. DRC retrieves the LineMetaData field schema but not the discriminating method
body. All five methods produce1.
Figure 16: Qualitative example of a QnA from the IR test set. Class-aware auto-increment id: RAG@3 retrieves
the actual SubjectProperties.register_line method body that returns len(self.existing_lines) ; DRC
retrieves the supportingLineMetaDataschema.
25

Detailed qualitative example:beartype/beartype(IR test, commit-derived)
QnA metadata
REPOSITORYbeartype/beartypeCOMMITSHA5f8778d6ba44. . .
COMMIT POSITION902 / 1014 (late history) IN-REPO SPLITtest
ASSERTION FAMILYassert <expr> is ?– the discriminating slot is a boolean identity literal
TEST FILEbeartype_test/a00_unit/a50_check/a60_error/a90_main/test_errorget.py, line 197
TEST FUNCTIONtest_get_func_pith_violation_conf_is_color
Test prefix (model input, trimmed)
frombeartypeimportBeartypeConf
frombeartype._check.error.errmainimportget_func_pith_violation
frombeartype._util.text.utiltextansiimportis_str_ansi
...
deftest_get_func_pith_violation_conf_is_color() -> None:
...
# Violation configured to contain ANSI escape sequences.
violation = get_func_pith_violation(
call_meta=minify_decor_meta_kwargs(
func=she_drew_back, conf=BeartypeConf(is_color=True)),
**kwargs)
# Assert this violation message contains ANSI escape sequences.
assertis_str_ansi(str(violation))is ???
Retrieved repository context (trimmed)
DRC (import-resolved;includes the discriminating signature):
# beartype/_check/error/errmain.py
defget_func_pith_violation(call_meta, pith_name,
pith_value, **kwargs) -> Exception: ...
# beartype/_check/metadata/call/callmetadecormin.py
defminify_decor_meta_kwargs(...): ...
# beartype/_util/text/utiltextansi.py -- this is the discriminating signature
defis_str_ansi(text:str) ->bool:
"""True only if the passed text contains one or more ANSI escape sequences."""
...
return_ANSI_REGEX.search(text)is notNone
RAG@3 (top-3 retrieved chunks): get_func_pith_violation body + checkmake.py helpers (trun-
cated).Neither chunk includes the is_str_ansi signature, so RAG never sees the -> bool return
type.
Per-method predictions and exact-match outcome
Method Prediction EM
REFERENCETrue
RAG (k=3)1✗
Dependency-Resolved ContextTrue✓
Code2LoRA-StaticTrue✓
Code2LoRA-EvoTrue✓
Text2LoRATrue✓
The slot is a boolean identity check ( is ?) on the return value of is_str_ansi(...) . The discriminat-
ing evidence is the function’s -> bool return type, which DRC retrieves explicitly because the import from
beartype._util.text.utiltextansi import is_str_ansi resolves to its definition. RAG@3 retrieves other functions
from the same module set butmisses is_str_ansi itself, and the base model defaults to the more-common idiom is 1 (a
truthy-shortcut pattern frequent in non-typed Python). Both Code2LoRA variants and Text2LoRA succeed parametrically,
having internalized the typed-boolean convention from the repository.
Figure 17: Qualitative example of a QnA from the IR test set. Retrieval-precision case: DRC follows the import
graph and surfaces the discriminating is_str_ansi(...) -> bool signature, while RAG@3 retrieves adjacent
but non-discriminating functions and collapses onto the n-gram-likelyis 1. 26

Detailed qualitative example:agronholm/apscheduler(IR test, commit-derived)
QnA metadata
REPOSITORYagronholm/apschedulerCOMMITSHAe4b1db1dcb8d. . .
COMMIT POSITION353 / 1207 (mid-history) IN-REPO SPLITval
ASSERTION FAMILYassert <expr> is <enum-member>– the slot is aJobOutcomeenum value
TEST LOCATIONtests/test_datastores.py::test_reap_abandoned_jobs, line 857
Test prefix (model input, trimmed)
fromapschedulerimportJob, JobOutcome, Task, ...
fromapscheduler.datastores.baseimportBaseExternalDataStore
...
# Earlier in the same test module (line 809) -- same target pattern:
# assert result.outcome is JobOutcome.abandoned
asyncdeftest_reap_abandoned_jobs(datastore: DataStore, ...) -> None:
task = Task(id="task1", func="...", job_executor="async")
await datastore.add_task(task)
job = Job(task_id="task1", executor="async",
result_expiration_time=timedelta(seconds=30))
await datastore.add_job(job)
await datastore.reap_abandoned_jobs("testscheduler")
jobs = await datastore.acquire_jobs("testscheduler", ..., 1)
assert len(jobs) == 1
await datastore.reap_abandoned_jobs("testscheduler")
assert notawait datastore.get_jobs()
abandoned_job_result = await datastore.get_job_result(jobs[0].id)
assertabandoned_job_result.outcomeis ???
Retrieved repository context (trimmed)
DRC (import-resolved;the answer is literally in a docstring):
# src/apscheduler/abc/_datastore.py
classDataStore(metaclass=ABCMeta):
@abstractmethod
asyncdefreap_abandoned_jobs(self, scheduler_id:str) -> None:
"""Find jobs marked as acquired by the given scheduler ID and
release them with the outcome of :attr:`~JobOutcome.abandoned`."""
...
# src/apscheduler/_structures.py
classJob:
id: UUID; task_id:str; ...
RAG@3 (top-3 retrieved chunks;the enum class is exposed but.abandoneditself is not):
# src/apscheduler/abc/_datastore.py (truncated overload)
classDataStore(metaclass=ABCMeta):
asyncdefreap_abandoned_jobs(self, scheduler_id:str) -> None: ...
# src/apscheduler/_events.py
classJobReleased(SchedulerEvent):
""":param outcome: the outcome of the job
:param ...: if``outcome``is :attr:`JobOutcome.error`.""" # only the .error member
outcome: JobOutcome = attrs.field(converter=as_enum(JobOutcome))
...
Per-method predictions and exact-match outcome
Method Prediction
REFERENCEJobOutcome.abandoned
Figure 18: Qualitative example of a QnA from the IR test set. Retrieval-degeneracy case: DRC retrieves a chunk
that contains the literal answer JobOutcome.abandoned (in a docstring), and RAG@3 retrieves the enum class
and the dotted access pattern but not the literal .abandoned member; yet for both methods the prepended context
triggers a Fill-In-the-Middle decode failure at generation time. Only the parametric methods (Code2LoRA variants,
Text2LoRA) complete the assertion correctly. 27

Table 12: Efficiency comparison. Extra storage is beyond the shared frozen base model (Qwen2.5-Coder-1.5B, 3.1
GB in bf16). Both Code2LoRA variants add zero inference tokens and generate repo-specific adapters in a single
forward pass.
Method Extra Tokens Adapt. Time Extra Storage
Pretrained 0 N/A —
RAG (k=3)∼1,500 per query +chunk index
Dep.-Resolved Context∼500–2,000 per query +import cache
FFT 0∼4h +3.1 GB
Single LoRA 0∼2h +32 MB
Per-repo LoRA 0∼5 min/repo +32 MB/repo
Code2LoRA-Static0<10ms/repo +679 MB
Code2LoRA-Evo0<10ms + GRU enc. +65 MB
28