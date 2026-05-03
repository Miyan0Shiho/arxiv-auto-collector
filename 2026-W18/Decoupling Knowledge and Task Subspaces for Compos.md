# Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation

**Authors**: Weihang Su, Hanwen Zhang, Qingyao Ai, Yiqun Liu

**Published**: 2026-04-29 15:00:35

**PDF URL**: [https://arxiv.org/pdf/2604.26768v1](https://arxiv.org/pdf/2604.26768v1)

## Abstract
Parametric Retrieval-Augmented Generation (PRAG) encodes external documents into lightweight parameter modules that can be retrieved and merged at inference time, offering a promising alternative to in-context retrieval augmentation. Despite its potential, many PRAG implementations train document adapters with task-supervised objectives, which may cause each adapter to encode both document-specific facts and reusable task-solving behavior. This entanglement may make adapter composition less reliable: when multiple adapters are merged at inference time, their overlapping task behaviors can accumulate together with document-specific updates, potentially making the merged adapter less stable and less focused on the intended document knowledge. To examine this issue, we explore Orthogonal Subspace Decomposition (OSD), an adapter-training setup that separates reusable task behavior from document-specific knowledge adapters. Concretely, we first train a Task LoRA to capture reusable task behavior, and then train document LoRAs to encode document-specific knowledge in a orthogonal subspace. This setup provides a controlled way to examine how orthogonalizing task and document LoRA updates affects adapter composition in multi-document PRAG. Experiments across multiple knowledge-intensive tasks and model scales suggest that this orthogonalization strategy can improve compositional robustness in parametric RAG, especially when multiple document adapters are merged.

## Full Text


<!-- PDF content starts -->

Decoupling Knowledge and Task Subspaces for Composable Parametric
Retrieval Augmented Generation
Weihang Su*1,Hanwen Zhang†1,Qingyao Ai1,Yiqun Liu1
1Department of Computer Science and Technology, Tsinghua University
Abstract
Parametric Retrieval-Augmented Generation
(PRAG) encodes external documents into
lightweight parameter modules that can be
retrieved and merged at inference time, of-
fering a promising alternative to in-context
retrieval augmentation. Despite its poten-
tial, many PRAG implementations train doc-
ument adapters with task-supervised objec-
tives, which may cause each adapter to en-
code both document-specific facts and reusable
task-solving behavior. This entanglement may
make adapter composition less reliable: when
multiple adapters are merged at inference time,
their overlapping task behaviors can accumu-
late together with document-specific updates,
potentially making the merged adapter less
stable and less focused on the intended doc-
ument knowledge. To examine this issue, we
explore Orthogonal Subspace Decomposition
(OSD), an adapter-training setup that separates
reusable task behavior from document-specific
knowledge adapters. Concretely, we first train
a Task LoRA to capture reusable task behav-
ior, and then train document LoRAs to encode
document-specific knowledge in an orthogonal
subspace. This setup provides a controlled way
to examine how orthogonalizing task and doc-
ument LoRA updates affects adapter compo-
sition in multi-document PRAG. Experiments
across multiple knowledge-intensive tasks and
model scales suggest that this orthogonalization
strategy can improve compositional robustness
in parametric RAG, especially when multiple
document adapters are merged.
1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a standard paradigm for grounding large lan-
guage models (LLMs) with external knowledge by
retrieving relevant documents at inference time and
injecting them into the prompt (Lewis et al., 2020).
*swh22@mails.tsinghua.edu.cn
†Contributed equallyDespite its effectiveness, standard in-context RAG
faces an inherent bottleneck: retrieved knowledge
is only provided transiently in context, whereas
LLMs often utilize knowledge most effectively
when it is represented within their parameters. This
limitation has motivated growing interest inpara-
metricforms of knowledge injection, where re-
trieved documents are encoded into loadable param-
eter modules rather than appended as raw text (Su
et al., 2025; Tan et al., 2025). In adapter-based Para-
metric Retrieval-Augmented Generation (PRAG),
each document can be encoded as a lightweight
parameter module that is retrieved and composed
at inference time, pointing toward a new form of
external parametric memory for LLMs (Su et al.,
2025).
However, simply parameterizing documents is
insufficient to build an effective external para-
metric memory system. The critical missing
property is composability. In existing PRAG-
style methods, each document adapter is trained
using document-grounded task supervision (e.g.,
question-answering or fact-verification examples).
Consequently, the learned adapter does not exclu-
sively encode factual knowledge; it simultaneously
encode the task-specific behavior required to utilize
that knowledge. This creates an entangled represen-
tation of both “what the document says” and “how
to perform the task.” Such entanglement renders
document parameterization highly suboptimal in
multi-document settings, where multiple retrieved
adapters must be merged. Instead of purely ag-
gregating complementary knowledge, the model
redundantly accumulates partially overlapping task
patterns. This leads to parameter interference and
degraded performance as the number of composed
adapters increases.
To examine this issue, we studyOrthogo-
nal Subspace Decomposition (OSD), an adapter-
training setup that explores whether task-level be-
havior and document-level knowledge can be sepa-arXiv:2604.26768v1  [cs.CL]  29 Apr 2026

rated for more stable parametric memory compo-
sition. The basic idea is to decouple parameter-
ized external memory into two components: a task
component, represented by a shared Task LoRA
that captures reusable task behavior, and a knowl-
edge component, represented by document-specific
Knowledge LoRAs that encode document-level in-
formation. During training, each Knowledge LoRA
is learned on top of a frozen Task LoRA, with
an orthogonality constraint that discourages over-
lap with the task component. At inference time,
the model uses the corresponding Task LoRA to-
gether with the merged Knowledge LoRAs of the
retrieved documents. This design allows us to ex-
amine whether separating task and document up-
dates can reduce interference when multiple docu-
ment adapters are composed.
We implement this idea with two variants. The
soft orthogonal variant adds an orthogonal regular-
ization term during Knowledge LoRA training to
penalize overlap with the task component. The hard
orthogonal variant parameterizes the document
adapter within the null space of the Task LoRA,
enforcing subspace separation by construction. Ide-
ally, the shared Task LoRA captures reusable task
behavior, while the document-specific Knowledge
LoRAs are encouraged to focus more on document-
level information rather than redundant task heuris-
tics. This separation is intended to make multi-
document composition less affected by repeated
task-level updates.
We evaluate this training setup across several
knowledge-intensive tasks, including open-domain
question answering, fact checking, slot filling,
and knowledge-grounded dialogue, using standard
PRAG benchmarks and expanded KILT-style multi-
task settings. Across different model scales, our
experiments suggest that orthogonalizing task and
document LoRA updates can improve the stabil-
ity of parametric RAG when multiple document
adapters are merged. The effect varies across
datasets and models, but the decoupled variants
are often less sensitive to increasing retrieval depth
than entangled parametric baselines. Additional
representation analysis further suggests that or-
thogonalization changes the geometry of document
adapters and, in the soft variant, makes relevant
document pairs more distinguishable in parameter
space. Our primary contributions are summarized
as follows:
•We analyze a potential source of instabilityin external parametric memory: document
adapters trained with task supervision may entan-
gle document-specific knowledge with reusable
task behavior, which can affect multi-document
adapter composition.
•We exploreOrthogonal Subspace Decomposi-
tionas a simple decoupling strategy that sep-
arates a shared Task LoRA from document-
specific Knowledge LoRAs. We consider both
soft and hard orthogonal variants to study how
different degrees of subspace separation affect
adapter composition.
•We report experiments across multiple
knowledge-intensive tasks and model scales,
suggesting that separating task and document
LoRA updates can improve the robustness of
parametric RAG under adapter composition.
2 Problem Formulation
Letqdenote an input query, and let R(q) return the
top-krelevant documents from an external corpus:
R(q) ={D 1, D2, . . . , D k}.(1)
The goal is to generate an output ythat is
grounded in the retrieved evidence, i.e.,
P(y|q,R(q)).(2)
Classic RAG approaches approximate this ob-
jective by concatenating retrieved documents into
the input context. Parametric RAG instead encodes
each document into a document-specific adapter
and composes multiple adapters at inference time.
While this avoids long-context injection, existing
document-wise parametric memory remains diffi-
cult to compose: each document adapter is trained
from task supervision grounded in that document,
and therefore tends to entangle document-specific
knowledge with task-specific behavior. As a re-
sult, when multiple document adapters are merged,
redundant or conflicting task patterns accumulate
alongside the desired factual knowledge, causing
severe parameter interference. Our objective is to
build a composable external parametric memory
system. Concretely, given a query qand a task type
t, we aim to learn a shared task module that cap-
tures task-general behavior, together with a set of
document modules that encode document-specific
factual knowledge, such that the retrieved docu-
ment modules can be stably composed with the
task module at inference time.

3 Methodology
We study a simple decomposition strategy for para-
metric retrieval-augmented generation (PRAG),
where task-level behavior and document-specific
knowledge are represented by separate LoRA mod-
ules. Rather than treating this decomposition as a
guaranteed disentanglement mechanism, our goal
is to examine whether separating these two compo-
nents during training can make document adapters
easier to compose at inference time. Concretely, we
first train a shared Task LoRA for each task type,
and then train document-specific Knowledge Lo-
RAs while keeping the corresponding Task LoRA
fixed. We instantiate this idea with two variants: a
soft orthogonality regularizer and a hard orthogonal
parameterization.
3.1 Overview
For a task type t, we first train a Task LoRA, de-
noted as ∆θ(t)
T, using task-level supervision that is
not tied to any single document. This adapter is
intended to capture reusable task behavior, such as
output format, task-specific reasoning patterns, and
general response style. Afterwards, for each docu-
ment Di, we train a document-specific Knowledge
LoRA, denoted as ∆θK,i, using examples derived
from Di. During this stage, the Task LoRA ∆θ(t)
T
is loaded but frozen, so that the Knowledge LoRA
is optimized on top of an already available task
adapter.
At inference time, given a query qand task type
t, the retriever returns a set of relevant documents
{D1, . . . , D k}. We activate the corresponding Task
LoRA once and aggregate the Knowledge LoRAs
associated with the retrieved documents:
θ′(q, t) =θ 0+∆θ(t)
T+Agg (∆θ K,1, . . . ,∆θ K,k),
(3)
where θ0denotes the base model parameters. For
each adapted layer, the aggregation can be written
as
∆W K=kX
i=1αi∆W K,i,kX
i=1αi= 1,(4)
where αimay be uniform or determined by re-
trieval scores. This formulation differs from di-
rectly merging multiple task-supervised document
adapters, since the shared task component is ap-
plied only once.3.2 Task and Knowledge Modules
We use the standard Low-Rank Adaptation (LoRA)
parameterization for each adapted weight matrix
W∈Rdout×din:
∆W=BA,(5)
where A∈Rr×d in,B∈Rdout×r, and ris the
LoRA rank. We omit the LoRA scaling factor for
notational simplicity.
For task type t, the Task LoRA is trained on a
task-level corpusD(t)
task:
Ltask=−X
(x,y)∈D(t)
tasklogPθ0+∆θ(t)
T(y|x).(6)
The task-level examples are constructed to empha-
size general task execution rather than memoriza-
tion of a particular document. This adapter is then
kept fixed when training document-specific Knowl-
edge LoRAs.
For each document Di, we construct a document-
level training set DK,iand optimize the correspond-
ing Knowledge LoRA on top of the frozen Task
LoRA:
L(i)
ce=−X
(x,y)∈D K,ilogPθ0+∆θ(t)
T+∆θ K,i(y|x).
(7)
This setup does not by itself guarantee perfect sepa-
ration between task behavior and document knowl-
edge. However, it provides a controlled way to
study whether document adapters trained in the
presence of a fixed task adapter become more suit-
able for multi-document composition.
3.3 Orthogonal Knowledge Learning
A document-specific adapter trained with task su-
pervision may still learn both document informa-
tion and task-level shortcuts. To reduce this overlap,
we explore orthogonality constraints between the
Task LoRA and each Knowledge LoRA. We apply
the constraint to the LoRA down-projection matri-
cesA, since their row spaces determine the input
directions used by the low-rank update.
For a specific adapted layer, let
AT∈RrT×din, A K,i∈RrK×din (8)
denote the down-projection matrices of the Task
LoRA and the i-th Knowledge LoRA, respectively.

The overlap between their row spaces can be mea-
sured by the Frobenius norm of their cross inner-
product matrix:
ATA⊤
K,i2
F.(9)
This quantity is zero when every row direction of
AK,iis orthogonal to every row direction ofA T.
Soft Orthogonality.The first variant adds an or-
thogonality penalty when training the Knowledge
LoRA:
L(i)
ortho=X
ℓA(ℓ)
TA(ℓ)⊤
K,i2
F,(10)
where ℓindexes adapted layers. The full objective
for the document module is
L(i)
know=L(i)
ce+λL(i)
ortho,(11)
where λcontrols the strength of the orthogonality
regularization.
Equivalently, for each layer, the penalty can be
computed as
ATA⊤
K,i2
F= tr
ATA⊤
K,iAK,iA⊤
T
.(12)
This soft variant does not strictly prevent overlap,
but it discourages the Knowledge LoRA from using
the same projection directions as the Task LoRA.
Hard Orthogonality.The second variant en-
forces orthogonality by reparameterizing the
Knowledge LoRA projection matrix. For each
adapted layer, we compute the singular value de-
composition of the trained task projection matrix:
AT=UΣV⊤.(13)
LetρTdenote the numerical rank of AT, deter-
mined by a singular-value threshold. We write
V∥∈Rdin×ρTfor the right singular vectors span-
ning the row space of AT, andV⊥∈Rdin×(din−ρT)
for an orthonormal basis of its null space. We
then parameterize the Knowledge LoRA down-
projection matrix as
AK,i=bAK,iV⊤
⊥,(14)
where
bAK,i∈RrK×(din−ρT)(15)
is the learnable parameter matrix. Since V⊤
⊥V∥= 0,
this construction gives
AK,iA⊤
T= 0.(16)Thus, the Knowledge LoRA is constrained to use
input directions orthogonal to the row space of the
Task LoRA.
Compared with the soft variant, this hard vari-
ant imposes a stricter structural constraint and re-
moves the need for an auxiliary orthogonality loss.
However, it also reduces the available parameter
space for the Knowledge LoRA. We therefore treat
the two variants as complementary ways to ex-
plore whether reducing task–knowledge overlap
improves the composability of parametric memory.
4 Experimental Setup
4.1 Baselines
To comprehensively evaluate the effectiveness of
our proposed decoupling framework, we compare
D-PRAG and D-PRAG-hard with two representa-
tive retrieval-augmented paradigms: standard in-
context RAG (Lewis et al., 2020) and Parametric
RAG (PRAG) (Su et al., 2025). Further details
of the baseline implementations and experimental
settings are provided in Appendix A.1.
4.2 Datasets
We evaluate our method on five task categories,
including four knowledge-intensive tasks from
the KILT benchmark (Petroni et al., 2021) and
one vertical-domain setting in medicine. The
knowledge-intensive tasks cover open-domain
question answering, fact-checking, slot filling, and
knowledge-grounded dialogue.
For open-domain question answering, we use
the DPR Wikipedia dump as the retrieval cor-
pus (Karpukhin et al., 2020), and evaluate on
four datasets: 2WikiMultihopQA (Ho et al.,
2020), HotpotQA (Yang et al., 2018), Com-
plexWebQuestions (Talmor and Berant, 2018), and
PopQA (Mallen et al., 2022). For fact-checking,
slot filling, and dialogue, we use the knowledge
sources provided by KILT (Petroni et al., 2021),
and select FEVER (Thorne et al., 2018) for fact-
checking, Zero-Shot RE (Levy et al., 2017) for
slot filling, and Wizard of Wikipedia (Dinan et al.,
2018) for dialogue. For the vertical-domain setting,
we use PubMed Abstracts (Gao et al., 2020) as
the corpus and evaluate on PubMedQA (Wu et al.,
2025). More details about the datasets and corpora
are provided in Appendix A.2.

Figure 1: Performance comparison across different retrieval depths ( K∈ {1,3,5,7,10} ) on five representative
datasets. From top to bottom, the three rows report results on Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, and
Meta-Llama-3-8B-Instruct.
4.3 Implementation Details
We conduct experiments on the Llama 3 model fam-
ily, including Llama-3.2-1B-Instruct, Llama-3.2-
3B-Instruct, and Meta-Llama-3-8B-Instruct. For
retrieval, we employ BM25 implemented with Elas-
ticSearch to retrieve the top- Krelevant documents
for each query. To study the sensitivity of differ-
ent methods to retrieval depth and context den-
sity, we vary the number of retrieved documents
withK∈ {1,3,5,7,10} . All experiments are con-
ducted on NVIDIA A100 (40GB) GPUs. Addi-
tional implementation details, including hyperpa-
rameter settings, are provided in Appendix A.3,
and the prompt templates used in our experiments
are listed in Appendix B.
4.4 Evaluation Metrics
We adopt task-specific evaluation metrics to as-
sess the performance of our decoupling framework.
Specifically, we report F1 for open-domain ques-
tion answering, slot filling, and dialogue, and Accu-
racy for fact-checking and the medical-domain task.
Due to the computational cost of large-scale infer-
ence, we evaluate all methods on a representative
subset comprising the first 300 test instances from
each dataset, thereby providing a consistent com-
parison setting across models and methods. Further
details on the evaluation protocols and metric defi-nitions are provided in Appendix A.4.
5 Experimental Results
5.1 Main Results
Figure 1 compares standard RAG, PRAG, D-
PRAG, and D-PRAG-hard as the number of re-
trieved documents increases under three similarity-
controlled retrieval settings. Overall, PRAG is of-
ten more sensitive to retrieval depth than the decou-
pled variants: in several settings, its performance
decreases as more document-specific adapters are
merged. By contrast, D-PRAG and D-PRAG-hard
generally exhibit flatter curves, suggesting that sep-
arating task-level behavior from document-level
information can improve the stability of multi-
document adapter composition. The effect is not
uniform across all datasets and model scales: stan-
dard RAG remains competitive in some cases, and
the relative performance of D-PRAG and D-PRAG-
hard varies across settings. Nevertheless, the re-
peated pattern that the decoupled variants are less
affected by increasing Kis consistent with our hy-
pothesis that reducing task-document entanglement
can make external parametric memory more robust
under adapter composition.

Figure 2: Cosine similarity distributions of relevant and
irrelevant passage pairs under PRAG, D-PRAG, and
D-PRAG-hard in the same experimental setting as our
main results. The upper and lower parts correspond
to Llama-3.2-1B-Instruct and Llama-3.2-3B-Instruct,
respectively. Similarity is computed from flattened
document-specific LoRA parameters, shown separately
for thedown_projandup_projmatrices.
5.2 Representation Analysis
To further inspect how different training strategies
affect the geometry of document adapters, we com-
pare the cosine similarity distributions between rel-
evant and irrelevant passage pairs using flattened
document-specific LoRA parameters, as shown in
Figure 2.
For PRAG, both relevant and irrelevant pairs are
concentrated in a highly positive similarity range,
indicating that document adapters tend to share
strong common directions in parameter space. This
makes the similarity distributions only moderately
informative for distinguishing relevance, and is con-
sistent with the possibility that task-supervised doc-
ument adapters contain shared task-level compo-
nents in addition to document-specific information.
D-PRAG shows a clearer separation between rel-
evant and irrelevant pairs in this analysis: relevant
pairs are generally shifted toward higher cosine
similarity, while irrelevant pairs remain more con-
centrated at lower values. This suggests that re-
ducing overlap with the task component can make
document adapters somewhat more sensitive to
document-level relatedness. However, the distri-
butions still overlap, so we interpret this result assuggestive rather than as direct evidence that the
learned adapters encode purely factual knowledge.
D-PRAG-hard exhibits a different pattern: the
cosine similarities of both relevant and irrelevant
pairs are centered near zero and largely overlap.
This is expected from the hard orthogonal param-
eterization, which strongly changes the scale and
geometry of adapter directions. Therefore, raw
cosine similarity in the flattened parameter space
becomes less directly interpretable as a relevance
indicator for this variant. Overall, the representa-
tion analysis provides an auxiliary view consistent
with our main hypothesis: separating task-level and
document-level updates can change the geometry
of document adapters, and in the soft variant this
change is associated with clearer relevance-related
structure.
6 Related Work
Retrieval-Augmented Generation and Paramet-
ric Knowledge Injection.Standard RAG (Lewis
et al., 2020) and its enhancements (Izacard and
Grave, 2021; Izacard et al., 2023; Borgeaud et al.,
2022) rely onin-contextknowledge injection,
which is constrained by inference context win-
dows. To address this, parametric knowledge
injection stores external knowledge directly in
model weights. Early works used dedicated
adapters (Wang et al., 2021), while recent Para-
metric RAG (PRAG) and DyPRAG encode docu-
ments into composable or dynamically generated
LoRA modules (Su et al., 2025; Tan et al., 2025).
Our work shares this parametric focus but targets
a critical bottleneck: the limitedcomposabilityof
document-wise modules caused by the entangle-
ment of task behavior and factual knowledge.
Parameter-Efficient Adaptation and Modu-
lar Composition.Parameter-efficient fine-tuning
(PEFT) methods like adapters and LoRA (Houlsby
et al., 2019; Hu et al., 2022) naturally enable mod-
ular composition. Existing approaches combine
these lightweight modules or parameter updates
across different tasks or languages (Pfeiffer et al.,
2021, 2020; Huang et al., 2023; Ilharco et al., 2023;
Yadav et al., 2023). However, these methods pri-
marily focus ontask-orlanguage-leveladaptations.
In contrast, our work tackles the distinct challenge
of composing multipledocument-levelmodules for
parametric RAG, where modules must contribute
complementary factual content without redundant
task heuristics.

Orthogonality and Interference-Aware Low-
Rank Adaptation.Orthogonal subspace design
effectively reduces parameter interference in con-
tinual learning and optimizes LoRA training (Wang
et al., 2023; Büyükakyüz, 2024). While these
works validate orthogonality for improving mod-
ularity, they focus on sequential task adaptation
and optimization efficiency. We fundamentally re-
purpose orthogonal constraints forfunctionalde-
composition in external parametric memory: sepa-
rating a shared task subspace (for reusable execu-
tion behavior) from orthogonal document-specific
subspaces (for factual memory) to ensure stable
multi-document merging.
7 Conclusion
In this paper, we studied whether external para-
metric memory can be made more composable
by separating task-level behavior from document-
level information. Motivated by the observation
that document adapters trained with task supervi-
sion may encode both document-specific informa-
tion and reusable task behavior, we exploredOr-
thogonal Subspace Decomposition, an adapter-
training setup that combines a shared Task LoRA
with document-specific Knowledge LoRAs. We
instantiated this idea with soft and hard orthog-
onal variants to examine how different forms of
task-document separation affect multi-document
adapter composition. Across multiple knowledge-
intensive tasks, datasets, and model scales, our ex-
periments suggest that orthogonalizing task and
document LoRA updates can improve the stabil-
ity of parametric RAG when multiple document
adapters are merged. The effect varies across set-
tings, and the decoupled variants are not uniformly
superior, but they often show lower sensitivity to
retrieval depth than entangled parametric baselines.
Additional representation analysis suggests that or-
thogonalization changes the geometry of document
adapters and, in the soft variant, can make relevant
document pairs more distinguishable in parameter
space. These findings position task-document de-
coupling as a useful direction for further studying
composable external parametric memory.
8 Limitations
This work should be understood as a preliminary
technical report rather than a finalized methodolog-
ical contribution. Our goal is to document an ex-
ploratory empirical study of task-knowledge LoRAdecomposition for parametric retrieval-augmented
generation, focusing on whether separating task-
level behavior from document-specific knowledge
can improve adapter composability in initial exper-
iments. Although the results suggest that such a
decomposition may help stabilize multi-document
adapter composition, they should be interpreted as
initial empirical observations rather than definitive
evidence of a fully established solution. Further
studies are needed to validate the robustness of this
strategy across broader model families, larger-scale
retrieval settings, and alternative adapter architec-
tures.
References
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George
van den Driessche, Jean-Baptiste Lespiau, Bogdan
Damoc, Aidan Clark, Diego de Las Casas, Aurelia
Guy, Jacob Menick, Roman Ring, Tom Hennigan,
Saffron Huang, Loren Maggiore, Chris Jones, Albin
Cassirer, Andy Brock, Michela Paganini, Geoffrey
Irving, Oriol Vinyals, Simon Osindero, Karen Si-
monyan, Jack Rae, Erich Elsen, and Laurent Sifre.
2022. Improving language models by retrieving from
trillions of tokens. InProceedings of the 39th Inter-
national Conference on Machine Learning, volume
162 ofProceedings of Machine Learning Research,
pages 2206–2240. PMLR.
Kerim Büyükakyüz. 2024. OLoRA: Orthonormal low-
rank adaptation of large language models.arXiv
preprint arXiv:2406.01775.
Emily Dinan, Stephen Roller, Kurt Shuster, Angela
Fan, Michael Auli, and Jason Weston. 2018. Wizard
of wikipedia: Knowledge-powered conversational
agents.arXiv preprint arXiv:1811.01241.
Leo Gao, Stella Biderman, Sid Black, Laurence Gold-
ing, Travis Hoppe, Charles Foster, Jason Phang, Ho-
race He, Anish Thite, Noa Nabeshima, et al. 2020.
The pile: An 800gb dataset of diverse text for lan-
guage modeling.arXiv preprint arXiv:2101.00027.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Con-
ference on Computational Linguistics, pages 6609–
6625.
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski,
Bruna Morrone, Quentin de Laroussilhe, Andrea Ges-
mundo, Mona Attariyan, and Sylvain Gelly. 2019.
Parameter-efficient transfer learning for NLP. In
Proceedings of the 36th International Conference on
Machine Learning, volume 97 ofProceedings of Ma-
chine Learning Research, pages 2790–2799. PMLR.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. LoRA: Low-rank adaptation of
large language models. InThe Tenth International
Conference on Learning Representations.
Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu
Pang, Chao Du, and Min Lin. 2023. Lorahub: Ef-
ficient cross-task generalization via dynamic LoRA
composition.arXiv preprint arXiv:2307.13269.
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Worts-
man, Suchin Gururangan, Ludwig Schmidt, Han-
naneh Hajishirzi, and Ali Farhadi. 2023. Editing
models with task arithmetic. InThe Eleventh Inter-
national Conference on Learning Representations.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettle-
moyer. 2017. Zero-shot relation extraction via read-
ing comprehension. InProceedings of the 21st Con-
ference on Computational Natural Language Learn-
ing (CoNLL 2017), pages 333–342.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Hannaneh Hajishirzi, and Daniel Khashabi. 2022.
When not to trust language models: Investigat-
ing effectiveness and limitations of parametric and
non-parametric memories. arxiv.arXiv preprint
arXiv:2212.10511.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Maillard,
et al. 2021. Kilt: a benchmark for knowledge in-
tensive language tasks. InProceedings of the 2021Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 2523–2544.
Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé,
Kyunghyun Cho, and Iryna Gurevych. 2021.
Adapterfusion: Non-destructive task composition for
transfer learning. InProceedings of the 16th Con-
ference of the European Chapter of the Association
for Computational Linguistics: Main Volume, pages
487–503, Online. Association for Computational Lin-
guistics.
Jonas Pfeiffer, Ivan Vuli ´c, Iryna Gurevych, and Se-
bastian Ruder. 2020. MAD-X: An adapter-based
framework for multi-task cross-lingual transfer. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP),
pages 7654–7673, Online. Association for Computa-
tional Linguistics.
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan,
Changyue Wang, Hongning Wang, Ziyi Ye, Yujia
Zhou, and Yiqun Liu. 2025. Parametric retrieval
augmented generation. InProceedings of the 48th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, pages
1240–1250, New York, NY , USA. Association for
Computing Machinery.
Alon Talmor and Jonathan Berant. 2018. The web as
a knowledge-base for answering complex questions.
InProceedings of the 2018 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
Volume 1 (Long Papers), pages 641–651.
Yuqiao Tan, Shizhu He, Huanxuan Liao, Jun Zhao,
and Kang Liu. 2025. Better wit than wealth: Dy-
namic parametric retrieval augmented generation for
test-time knowledge enhancement.arXiv preprint
arXiv:2503.23895.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
Fever: a large-scale dataset for fact extraction and
verification. InProceedings of the 2018 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies, Volume 1 (Long Papers), pages
809–819.
Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei,
Xuanjing Huang, Jianshu Ji, Guihong Cao, Daxin
Jiang, and Ming Zhou. 2021. K-Adapter: Infusing
knowledge into pre-trained models with adapters. In
Findings of the Association for Computational Lin-
guistics: ACL-IJCNLP 2021, pages 1405–1418, On-
line. Association for Computational Linguistics.
Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong
Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing
Huang. 2023. Orthogonal subspace learning for lan-
guage model continual learning. InFindings of the
Association for Computational Linguistics: EMNLP

2023, pages 10658–10671, Singapore. Association
for Computational Linguistics.
Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu,
Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin
Cho, Chang-In Choi, et al. 2025. Medreason: Elicit-
ing factual medical reasoning steps in llms via knowl-
edge graphs.arXiv preprint arXiv:2504.00993.
Prateek Yadav, Derek Tam, Leshem Choshen, Colin A.
Raffel, and Mohit Bansal. 2023. TIES-Merging: Re-
solving interference when merging models. InAd-
vances in Neural Information Processing Systems,
volume 36.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
A Experimental setup
A.1 Baselines
•Standard RAG(Lewis et al., 2020): This is a
traditional in-context retrieval-augmented gen-
eration method. It directly prepends the top-
Kretrieved documents to the input prompt, en-
abling the model to leverage external knowledge
through in-context learning without any modifi-
cation to the model parameters.
•Parametric RAG(Su et al., 2025): Unlike the
standard RAG, this approach internalizes exter-
nal knowledge by parameterizing retrieved doc-
uments into LoRA(Hu et al., 2022) modules. It
enables parametric knowledge injection, allow-
ing the model to memorize the retrieved con-
tent within its weights during the inference stage.
We utilize the official implementation from the
PRAG library.
A.2 Datasets and Corpus
Open domain QA:We use the Wikipedia dump
provided by DPR(Karpukhin et al., 2020) as our
corpus.To evaluate the model’s capabilities in both
knowledge retrieval and complex reasoning, we
select four diverse and representative benchmarks:
•2WikiMultihopQA(2WQA)(Ho et al., 2020):
A Wikipedia-based multi-hop QA dataset that
emphasizes the ability to associate and reason
over cross-document factual relationships.
•HotpotQA(HQA)(Yang et al., 2018): A multi-
hop reasoning benchmark that requires models tosynthesize evidence across multiple documents
to derive the correct answer.
•ComplexWebquestions(CWQ)(Talmor and Be-
rant, 2018): Comprising complex natural lan-
guage queries from the web, this dataset de-
mands compositional reasoning over multiple
sub-questions.
•PopQA(Mallen et al., 2022):An open-domain
QA dataset focused on popular entities, where
questions typically rely on single-hop retrieval or
implicit factual knowledge.
Fact-Checking: We use the knowledge source pro-
vided by KILT(Petroni et al., 2021) as our external
corpus.
•FEVER(Thorne et al., 2018): A large-scale
benchmark for fact extraction and verification, re-
quiring the model to classify claims as Supported,
Refuted, or NotEnoughInfo based on retrieved
evidence.
Slot-filling: We use the knowledge source provided
by KILT(Petroni et al., 2021) as our external cor-
pus.
•Zero Shot RE(Levy et al., 2017): A relation
extraction dataset formulated as a slot-filling
task, which evaluates the model’s ability to
extract specific subject-relation-object triplets
from the corpus.
Dialogue: We use the knowledge source provided
by KILT(Petroni et al., 2021) as our external cor-
pus.
•Wizard of Wikipedia (WoW)(Dinan et al.,
2018): A knowledge-driven dialogue dataset
where the model must engage in conversa-
tions by retrieving and incorporating relevant
Wikipedia topics into its responses.
Medical-Verify: To test the effectiveness of our
proposed method in vertical domains, we conduct
experiments in Medicine. We use the PubMed
Abstracts subset of The Pile(Gao et al., 2020) as
our external corpus.
•PubMedQA(Wu et al., 2025): A specialized
biomedical QA dataset based on PubMed ab-
stracts. It requires models to answer research
questions by reasoning over the provided con-
text.

A.3 Implementation Details
•Data Augmentation: During Document Aug-
mentation Process, We select meta-Llama-3-8B-
Instruct. For knowledge-intensive tasks, we em-
ploy a multi-task data augmentation strategy. For
each document, we generate three input-output
pairs for each of the four task types—Open do-
main QA, Fact-checking, Slot-filling, and Dia-
logue—specifically for training the Doc_LoRA.
To further train the task_LoRA, we generate an
additional single pair per task type from the same
document. In the vertical domain (Medical), the
augmentation is focused on domain-specific ver-
ification. Each document yields three Medical
Verification pairs for Doc_LoRA training and
one pair for the task_LoRA. The prompt tem-
plates we use in data augmentation are provided
in AppendixB.1
•Task LoRA Training: For knowledge-intensive
tasks, we randomly sample 1,500 documents
from the augmented corpus, while for vertical do-
mains, 900 documents are sampled to construct
the training sets. The complete training data are
available at our official repository:1, During the
training process, we adopt a default configura-
tion of 1 epoch and a learning rate of 1×10−4.
Specific adjustments are made for optimal con-
vergence on certain tasks: (1) For the1B model,
we set the epoch to 3 for Fact-checking and 2
for Dialogue; (2) For the3B model, we utilize a
learning rate of 8×10−5for Fact-checking and
set the epoch to 2 for Dialogue.
•Doc LoRA Training: In the knowledge param-
eterization stage, we set the default configura-
tion to 1 epoch with a learning rate of 3×10−4.
To ensure robust performance across different
tasks and model scales, we apply the following
specific adjustments:(1) For PopQA, WoW, and
Zero Shot RE, the training epoch is increased
to 2 for all model scales (1B, 3B, and 8B).(2)
On the 3B model, the learning rate is adjusted
to1×10−4for FEVER and 5×10−4for Zero
Shot RE. On the 8B model, the learning rate for
FEVER is set to 5×10−5, while Zero Shot RE
utilizes5×10−4.
For a fair comparison, the training hyperparam-
eters remain identical across PRAG, D-PRAG,
and D-PRAG-hard under the same experimen-
tal settings.For our decoupling mechanisms, in
1https://github.com/oneal2000/OSDthe soft orthogonality approach (D-PRAG), we
set the regularization coefficient λto 0.1 for 1B
and 3B models, and 0.2 for the 8B model. In
the hard orthogonality approach (D-PRAG-hard),
the threshold τto identify near-zero singular val-
ues in the Singular Value Decomposition (SVD)
is set to 1×10−5. The prompt templates we use
to train Doc LoRA are provided in AppendixB
•Inference: To ensure a fair and reproducible
evaluation, the decoding temperature is strictly
set to 0 (i.e., greedy search) for all generation
tasks. The prompt templates we use during the
inference process are provided in AppendixB
A.4 Evaluation and Metrics
To ensure a comprehensive and computationally ef-
ficient evaluation, we report the performance on the
first 300 instances of the overall test split for each
dataset. Furthermore, for the multi-hop reasoning
benchmarks (2WQA and HQA), we conduct an ad-
ditional evaluation by testing the first 300 instances
within each specific sub-category. We adopt task-
specific metrics for this evaluation: F1-score is re-
ported for the QA (2WQA, HQA, CWQ, PopQA),
Slot-filling (Zero Shot RE), and Dialogue (WoW)
tasks, while Accuracy is utilized for Fact-checking
(FEVER) and Medical Verification (PubMedQA).
B Prompt Templates
In this section, we detail the prompt templates uti-
lized across our framework, divided into two main
stages: Data Augmentation and Training/Inference.
B.1 Data Augmentation Prompts
During the data augmentation phase, the model is
prompted to generate a rewritten version of the
provided text and structured input-output pairs.
Rewrite Prompt: This prompt template is used
to rewrite the provided text.
Rewrite Prompt
Rewrite the following passage. While keeping
the entities, proper nouns, and key details such
as names, locations, and terminology intact,
create a new version of the text that expresses
the same ideas in a different way. Make sure
the revised passage is distinct from the origi-
nal one, but preserves the core meaning and
relevant information. [passage]

Augmentation Prompts: The augmentation
prompt templates provided below are used to gener-
ate four task-specific input-output pairs per passage,
with three allocated for training the Doc LoRA
and the remaining one reserved for the task LoRA.
[...] serves as a placeholder representing a concrete
JSON instance of the specific task type. Readers
are encouraged to refer to our official GitHub repos-
itory (2) for the complete executable examples.
Augmentation Prompt: Open-domain QA
I will provide a passage of text, and you need
to generate four different questions based on
the content of this passage. Each question
should be answerable using the information
provided. Additionally, please provide an ap-
propriate answer for each question derived
from the passage.
You need to generate the question and answer
in the following format:
[...]
This list should have at least four elements.
Passage:
[passage]
Augmentation Prompt: Fact-Checking
I will provide a passage of text, and you need
to generate four claims based on the content of
this passage. Each claim should be verifiable
using the information provided in the passage.
Additionally, please provide an appropriate la-
bel for each claim, indicating whether it is
’SUPPORTS’ or ’REFUTES’.
You need to generate each claim and label in
the following format:
[...]
This list should have at least four elements.
Passage:
[passage]
Augmentation Prompt: Slot-filling
I will provide a passage of text, and you need
to extract four slot-filling examples from it.
Each example should identify a subject entity
mentioned in the passage, one of its relations,
and the corresponding object entity.
You should model the input as a structured
2https://github.com/oneal2000/OSDstring in the format ’subject_entity [SEP] rela-
tion’.
The output should be the object entity that fills
the slot, based on the passage.
Additionally, for each slot-filling example, you
need to generate a natural language template
question that could be answered by the output
(use the subject entity and relation in the ques-
tion)
You need to generate the input, output, and the
template question in the following format:
[...]
You only need to output this list in the above
format.
This list should have at least four elements
Passage:
[passage]
Augmentation Prompt: Dialogue
I will provide a passage of text from Wikipedia,
and you need to generate four knowledge-
grounded dialogues in the style of the Wizard
of Wikipedia dataset.
Each dialogue should be a natural, multi-turn
conversation between a curious user and a
knowledgeable assistant (wizard) who has ac-
cess to the provided passage.
The assistant should provide informative, de-
tailed responses based on the passage con-
tent, while maintaining a natural conversa-
tional flow.
The input should contain the conversation his-
tory (alternating user and assistant messages),
and the output should be the assistant’s re-
sponse to the last user message.
Each message in the input should be separated
by a newline character (\n), except the last mes-
sage.
The assistant’s responses should be informa-
tive, engaging, and naturally incorporate in-
formation from the passage without directly
copying it.
The user’s questions should progressively ex-
plore different aspects of the topic, building
upon previous turns in the conversation.
You need to generate the dialogues in the fol-
lowing format:
[...]
Important guidelines:

- The assistant should provide detailed, infor-
mative responses based on the passage
- The conversation should feel natural and en-
gaging, not robotic
- Each dialogue should have a varied number
of turns (typically 1-6 turns)
- The user’s questions should explore different
aspects of the topic mentioned in the passage
- The assistant’s responses should synthesize
information from the passage in a natural way
- Generate at least four dialogues with different
conversation flows
You only need to output this list in the above
format.
This list should have at least four elements.
Passage:
[passage]
Augmentation Prompt: Medical Verification
I will provide a passage. Based ONLY on the
factual content of the passage, generate four
yes/no questions.
Each question must be objectively answerable
as ’yes’ or ’no’ based on the passage.
For each question, provide the correct answer
as exactly one lowercase word: ’yes’ or ’no’.
[...]
This list must contain at least four elements.
Passage:
[passage]
B.2 Training and Inference Prompts
To maintain strict alignment between the training
and inference stages, we employ identical prompt
templates throughout both phases. Specifically,
during the training of the task LoRA, external pas-
sages are deliberately excluded from the prompts
to focus entirely on task format adaptation. Con-
versely, during the Doc LoRA training, we utilize
a mixed prompt strategy, incorporating instances
both with and without the contextual passages to
ensure robust knowledge internalization.
Specifically, the [passages] placeholder is dy-
namically populated with a concatenated text of
retrieved documents when external context is re-
quired; otherwise, it is simply left blank. The con-
catenated text format is provided here:Concatenated Text
Passage 1: Passage 1
Passage 2: Passage 2
. . . . . .
Passage K: Passage K
Prompt for Open domain QA: This template is
used for Open domain QA.
Prompt 1: Open domain QA
You should answer the question by referring to
the knowledge provided below and integrating
your own knowledge.
[passages]
Question: [question]
Answer:
Prompt for Fact-checking: This template is used
for Fact-checking.
Prompt 2: Fact-Checking
You are tasked with verifying a claim using
the knowledge provided below combined with
your own knowledge.
Your response MUST be exactly one word:
’SUPPORTS’ or ’REFUTES’. Do not output
anything else and do not explain your choice.
Using the passages and your knowledge,
output ’SUPPORTS’ if the claim is true, or
’REFUTES’ if the claim is false.
[passages]
Claim: [input]
Output:
Prompt for Slot-filling: This template is used for
Slot-filling.
Prompt 3: Slot-Filling
You are tasked with extracting the object
entity that completes a given slot using the
knowledge provided below together with your
own knowledge.
The input format is: ’subject_entity [SEP]
relation’.
You must return only the object entity that
is directly connected to the subject_entity
through the specified relation.

The extracted entity should be the one that can
directly serve as the answer to the question:
[template_question]
Do not provide explanations or additional text,
only output the object entity.
[passages]
Input: [input]
Output:
Prompt for Dialogue: This template is used for
Dialogue.
Prompt 4: Dialogue
You are tasked with generating a response
to the user’s last message in the whole
conversation history, using the knowledge
provided below combined with your own
knowledge.
The input contains the entire conversation
history between the user and the assistant.
[passages]
Input: [input]
Output:
Prompt for Medical Verification: This template
is used for Medical Verification.
Prompt 5: Medical Verification
You are tasked with answering a medical
yes/no question using the knowledge provided
below combined with your own medical
knowledge.
Your response MUST be exactly one word:
’yes’ or ’no’. Do not output anything else and
do not explain your choice.
Using the passages and your knowledge,
Output ’yes’ if it is true, otherwise output
’no’.
[passages]
Question: [question]
Answer: