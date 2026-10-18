# MMKG-R1: Constructing Multimodal Knowledge Graphs via Reinforcement Learning

**Authors**: Zongyu Wu, Yilong Wang, Xiaochen Wang, Minhua Lin, Yuan Zhong, Fenglong Ma, Xiang Zhang, Suhang Wang

**Published**: 2026-10-14 00:35:59

**PDF URL**: [https://arxiv.org/pdf/2610.19341v1](https://arxiv.org/pdf/2610.19341v1)

## Abstract
Multimodal knowledge graphs (MMKGs) organize information across different modalities to support multimodal Graph Retrieval-Augmented Generation (RAG), which aims to improve the performance of multimodal large language models (MLLMs). The effectiveness of these multimodal Graph RAG methods largely depends on the quality of the constructed MMKGs. However, most existing methods construct MMKGs without explicit guidance on graph quality, while reinforcement learning (RL)-based methods that rely on final-answer correctness may provide only indirect feedback on MMKG quality. To address these limitations, we propose MMKG-R1, a novel MMKG-quality-driven RL framework for training an MLLM as a multimodal knowledge constructor. To be specific, MMKG-R1 introduces multiple quality-aware rewards to directly evaluate both the quality of information extracted from individual knowledge sources and the quality of the assembled MMKG. These rewards are then used to guide the RL training. Experiments across multiple datasets show that MMKG-R1 constructs MMKGs that improve downstream question-answering accuracy.

## Full Text


<!-- PDF content starts -->

MMKG-R1: Constructing Multimodal Knowledge Graphs via
Reinforcement Learning
Zongyu Wu†, Yilong Wang†, Xiaochen Wang, Minhua Lin, Yuan Zhong,
Fenglong Ma, Xiang Zhang, Suhang Wang
The Pennsylvania State University
{zongyuwu,yvw5769,xcwang,mfl5681,yfz5556,fenglong,xzz89,szw494}@psu.edu
Abstract
Multimodal knowledge graphs (MMKGs) or-
ganize information across different modali-
ties to support multimodal Graph Retrieval-
Augmented Generation (RAG), which aims to
improve the performance of multimodal large
language models (MLLMs). The effective-
ness of these multimodal Graph RAG meth-
ods largely depends on the quality of the con-
structed MMKGs. However, most existing
methods construct MMKGs without explicit
guidance on graph quality, while reinforcement
learning (RL)-based methods that rely on final-
answer correctness may provide only indirect
feedback on MMKG quality. To address these
limitations, we propose MMKG-R1, a novel
MMKG-quality-driven RL framework for train-
ing an MLLM as a multimodal knowledge con-
structor. To be specific, MMKG-R1 introduces
multiple quality-aware rewards to directly eval-
uate both the quality of information extracted
from individual knowledge sources and the
quality of the assembled MMKG. These re-
wards are then used to guide the RL training.
Experiments across multiple datasets show that
MMKG-R1 constructs MMKGs that improve
downstream question-answering accuracy.
1 Introduction
Multimodal large language models (MLLMs) (Liu
et al., 2023, 2024; Bai et al., 2025; Yin et al., 2024)
have been widely adopted across various domains.
Despite their effectiveness, they still suffer from
outdated knowledge and hallucinations (Bai et al.,
2024; Li et al., 2023), especially when answering
questions that require recent or unseen information.
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Gao et al., 2023; Abootorabi et al.,
2025) is an established approach for reducing hallu-
cinations in (M)LLMs (Touvron et al., 2023; Wang
et al., 2024) and enabling them to access and utilize
†Core contributors.
Figure 1: An example illustrating that a correct down-
stream answer for a certain question does not necessarily
indicate that all content in the constructed MMKG is ac-
curate or of high quality. MMKG-R1 introduces explicit
quality-aware rewards to assess the constructed MMKG
and utilize them to train a multimodal constructor.
external knowledge. As knowledge is often multi-
modal, including information like texts and images,
recent works (Guo et al., 2025a; Dai et al., 2026)
start constructing multimodal knowledge graphs
(MMKGs) (Liang et al., 2024) to support multi-
modal Graph RAG such as MMGraphRAG (Wan
and Yu, 2025). By incorporating both semantic
information in each modality and cross-modal as-
sociations, these methods aim to further improve
retrieval quality and downstream question answer-
ing performance.
Although MMKG-based Graph RAG methods
have shown promising performance, most existing
approaches (Guo et al., 2025a; Wan and Yu, 2025)
follow a common pipeline in which MMKGs are
constructed directly by pretrained models without
explicitly considering their utility for downstream
RAG tasks. Since the performance of these meth-
ods heavily rely on the quality of the constructed
MMKGs, a promising direction is to train a model
1
arXiv:2610.19341v1  [cs.CV]  14 Oct 2026

to generate high-quality MMKGs.
Some recent approaches (Tsang et al., 2026;
Lin et al., 2026) have begun exploring graph op-
timization through reinforcement learning (RL).
AutoGraph-R1 (Tsang et al., 2026) trains con-
structors for text-derived KGs using task-aware
rewards based on deducibility and passage recall.
EvoGraph-R1 (Lin et al., 2026) trains an agent to
retrieve from and edit a multimodal knowledge hy-
pergraph, primarily guided by final answer corret-
ness. However, as noted in AutoGraph-R1, answer-
level feedback alone does not reliably reflect graph
quality. A correct downstream answer does not nec-
essarily imply that the MMKG correctly represents
the source documents. For example, as shown in
Figure 1, a model using multimodal Graph RAG
may correctly answer a question about a company’s
profit even when the constructed MMKG records
an incorrect profit value. Other correctly extracted
information, such as the company’s revenue, may
still enable retrieval of the relevant source image,
allowing the model to obtain the correct profit value
directly from the image. However, such noise in
the MMKG may result in reduced performance on
other questions. For example, when asked "What is
the revenue of the company with a profit of $20M?",
the system may fail to identify the relevant com-
pany and retrieve its source image due to the incor-
rect profit value in the MMKG, potentially leading
to an incorrect answer.
Therefore, in this paper, we propose to study
a novel problem of designing reward signals that
assess both the quality of information extracted
from individual knowledge sources and the qual-
ity of the MMKG assembled from the extracted
information to better train the MMKG construc-
tor through RL. This raises two key challenges:
(i) How can we directly evaluate the quality of
constructed MMKGs without relying on final an-
swer correctness? (ii) How can we provide effec-
tive reward signals at both the knowledge-source
and MMKG levels to guide constructor training?
To address the challenges, we propose MMKG-
R1, a novel graph-quality-driven RL framework
that trains a multimodal constructor using comple-
mentary feedback at both MMKG and knowledge-
source levels. At the MMKG level, we design
Deducibility Rewardto evaluate the evidence re-
trieved from the assembled MMKG, encouraging
the constructor to preserve relevant facts and re-
lations from the original knowledge sources. At
the knowledge-source level, we design two rewardsignals. TheSource-Grounded Quality Reward
compares extracted information with its original
knowledge sources to evaluate extraction accuracy
and coverage of key information, independent of
the current question. TheGraph-Readiness Reward
encourages well-structured extractions that provide
usable entities and facts for MMKG assembly.
Ourmain contributions are:(i)We present
MMKG-R1, a novel quality-driven RL framework
for constructing high-quality MMKGs from mul-
timodal knowledge sources;(ii)We move beyond
conventional final-answer-based rewards by de-
signing multiple quality-aware reward signals that
directly assess construction quality at both the
knowledge-source and MMKG levels, providing
more targeted feedback to guide MMKG construc-
tion; and(iii)We conduct comprehensive experi-
ments on multiple datasets, showing that MMKG-
R1 consistently generates high-quality MMKGs
that can effectively support multimodal Graph
RAG.
2 Related Work
2.1 Graph RAG
Retrieval-augmented generation (RAG) (Lewis
et al., 2020; Chen et al., 2022; Hu et al., 2023;
Yu et al., 2025; Abootorabi et al., 2025; Gao et al.,
2023) provides external evidence to mitigate the
knowledge limitations and hallucinations of (mul-
timodal) large language models. Graph RAG ex-
tends this paradigm by organizing knowledge into
graph structures to support retrieval and genera-
tion (Edge et al., 2024; Sun et al., 2024; Guo
et al., 2025b; Wang et al., 2026). For example,
GraphRAG (Edge et al., 2024) uses hierarchical
community summaries, while LightRAG (Guo
et al., 2025b) combines graph-based indexing with
dual-level retrieval. Recent methods construct mul-
timodal knowledge graphs (MMKGs) to support
multimodal Graph RAG (Wan and Yu, 2025; Yuan
et al., 2026; Guo et al., 2025a; Dai et al., 2026; Wu
et al., 2026b). For instance, MMGraphRAG (Wan
and Yu, 2025) builds image-based entity-relation
graphs and a text-based KG, respectively, then inte-
grates them through cross-modal fusion. However,
most existing methods rely on pretrained models
without task-specific training of the MMKG con-
struction process.
2

2.2 Reinforcement Learning for Graph
Construction
Reinforcement learning (RL) methods (Schulman
et al., 2017) like Group Relative Policy Optimiza-
tion (GRPO) (Shao et al., 2024), has been used to
train LLMs for retrieval decisions and graph reason-
ing (Jin et al., 2025b; Wu et al., 2026a; Luo et al.,
2026; Xu et al., 2026; Zhang and Zhao, 2025). Re-
cent studies further apply RL to graph construction.
AutoGraph-R1 (Tsang et al., 2026) uses GRPO to
train a textual graph constructor with graph util-
ity rewards, while EvoGraph-R1 (Lin et al., 2026)
learns to evolve query-conditioned multimodal hy-
pergraphs using rewards for answer correctness,
trajectory structure, and action cost. However, re-
wards in EvoGraph-R1 do not directly evaluate the
quality of the constructed graph. MMKG-R1 in-
stead learns to construct high-quality multimodal
knowledge graphs through direct quality feedback.
Our designed rewards assess both the information
extracted from individual sources and the assem-
bled MMKG, providing supervision for source-
grounded extraction and graph readiness.
3 Preliminaries
In this section, we first formalize how MMKG-
R1 constructs an MMKG and how the constructed
graph supports question answering, and then define
the learning problem.
Multimodal graph construction.Let Xdenote
a collection of source instances from multimodal
documents, each being a text passage, an im-
age, or a table. MMKG-R1 trains an MLLM
πθas themultimodal constructorto generate a
structured extraction yxof typed entities and sub-
ject–relation–object triples from the visible content
v(x)of each sourcex∈ X:
yx∼πθ(· |v(x)).(1)
Additional source information µx, such as prove-
nance metadata and references to the original con-
tent, is maintained by the environment and is never
shown to πθ. Two fixed, non-trainable modules
then process the extractions: the parser Pparses
eachyx, and the assembler Buses the parsed results
andµxto assign graph identifiers, attach prove-
nance, link entities across sources, and assemble
the graph as:
Gθ(X) =B 
(P(y x), µx)	
x∈X
.(2)The resulting Gθis afact-centricandsource-
groundedMMKG: each extracted triple becomes
a fact node connected to its arguments and its sup-
porting source, and fixed entity linking connects
compatible mentions across sources. Since Pand
Bare fixed, the quality of Gθis governed by the
extractions ofπ θ.
Graph-based question answering.Given a
question qand a constructed MMKG G, a fixed
retrieval and evidence-packing pipeline selects rel-
evant evidence under predefined budgets:
E(q, G) = (F q,Vq),(3)
where Fqcontains retrieved structured facts with
source provenance, and Vqcontains selected orig-
inal visual inputs (images and image-backed ta-
bles) with their available source metadata. A down-
stream model answers qusing only E(q, G) , with-
out access to the full set of sources.
Problem formulation.Each training example
ξq= (q,A q,Xq)∈ D consists of a question q,
a set of accepted answers Aq, and the annotated
supporting sources Xq. Construction isquestion-
blind: πθprocesses each x∈ X qindependently
from v(x) alone, without access to q,Aq, or the
other sources, and yields Gθ(Xq)via Eqs. (1)–(2),
so its extractions can be reused across questions.
The question and answers are used only by the
training environment to assess the resulting graph.
Therefore, we seek to optimize
max
θEξq∼DEπθh
Q 
Gθ(Xq);ξqi
,(4)
where the inner expectation is over the sampled ex-
tractions {yx}x∈Xq, andQscores the quality of the
graph itself rather than the correctness of a final an-
swer. Section 4 specifiesQand how Equation. (4)
is optimized.
4 Method
Our proposed MMKG-R1 trains a multimodal con-
structor to facilitate the contruction of MMKGs
that effectively support downstream question an-
swering. As shown in Figure 2, we introduce three
complementary reward signals that jointly evaluate
evidence sufficiency for downstream question an-
swering, fidelity to the original knowledge source,
and structural usability: (i)Deducibility Reward,
which evaluates whether the retrieved evidence
from a constructed MMKG is sufficient to support
3

Figure 2: Overview of MMKG-R1. The constructor independently extracts structured knowledge from each
knowledge source. The extracted knowledge is then used to assemble MMKGs. Three designed rewards guide
constructor optimization: the graph-level Deducibility Reward and two source-level rewards, the Source-Grounded
Quality Reward and the Graph-Readiness Reward.
a correct answer; (ii)Source-Grounded Quality
Reward, which assesses whether the knowledge
extracted by the constructor faithfully preserves the
central content of its original knowledge source;
and (iii)Graph-Readiness Reward, which encour-
ages structurally valid and usable outputs for graph
construction. Evidence sufficiency depends on the
question and on how extractions from different
sources combine, so the Deducibility Reward is
computed at the candidate-graph level. Fidelity
and usability are properties of each extraction re-
gardless of the question, so the Source-Grounded
Quality and Graph-Readiness Rewards are com-
puted at the source-extraction level.
4.1 Source-Level Construction and Graph
Assembly
For each training question q, we consider its anno-
tated supporting knowledge sources Xq, as defined
in Section 3. The constructor πθprocesses each
knowledge source x(q)
mindependently, using only
its content v(x(q)
m). Let πθolddenote the rollout
policy used to collect the current training batch.
For each knowledge source x(q)
m, we independently
sample four responses y(q)
m,i∼π θold(· |v(x(q)
m)),
wherei∈ {1,2,3,4}.
Constructor outputs with the same sampling in-
dexiare then parsed and combined into a candidate
multimodal knowledge graph G(q)
i, while preserv-
ing source metadata. This produces four candidate
MMKGs per question for group-relative compari-
son, without enumerating all possible combinations
of outputs across knowledge sources. For each can-
didate graph G(q)
i, we use the fixed retrieval and
evidence-packing pipeline introduced in Section 3
to obtain E(q)
i=E(q, G(q)
i), which includes struc-tured facts retrieved from the graph and original
visual evidence when available. We use a shared,
fixed environment to perform entity linking, graph
assembly, retrieval, and budgeted evidence packing
for all candidates. Throughout training, only the
constructor is updated, while all other components
remain fixed.
4.2 Graph-Level Deducibility Reward
We first present the design of the Deducibility Re-
ward, which is inspired by the Knowledge-Carrying
Reward proposed in AutoGraph-R1 (Tsang et al.,
2026). For each candidate MMKG G(q)
i, we use
a frozen MLLM as the Judge to evaluate whether
the retrieved evidence E(q)
ifromG(q)
iprovides suf-
ficient support for at least one accepted complete
answer in Aq. Training graphs are constructed
from the annotated supporting sources Xq, which
contain the evidence needed to answerq.
To be specific, the judge model is prompted to
evaluate two types of evidence gaps in sequence: (i)
Whole-bundle gap: Indicates that both the retrieved
structured facts and the original visual information
in knowledge sources are insufficient to support
any complete accepted answer; and (ii)Structured-
path gap: This gap is evaluated only when the
whole bundle is sufficient. It indicates that omis-
sions or contradictions in answer-critical informa-
tion prevent the structured facts from supporting
a complete, correct, and unambiguous reasoning
path. The two checks distinguish the usefulness of
the retrieved evidence from the completeness and
correctness of the structured reasoning path.
Each reported gap describes concrete missing
or conflicting evidence. A reported whole-bundle
gap results in a score of 0. When only a structured-
path gap exists, the evidence supports an answer,
4

but the structured reasoning path is incomplete or
incorrect. We assign a score of 0.5to recognize
the usefulness of the evidence while encouraging
improvements to the structured facts. A candidate
with neither gap receives a score of 1. These scores
define three levels of evidence utility, formalized
as follows:
d(q)
i=

0,whole-bundle gap,
0.5,structured-path gap only,
1,neither gap.(5)
4.3 Knowledge Source-Level Feedback
While the Deducibility Reward described above
evaluates whether retrieved graph evidence is suf-
ficient to support downstream question answering,
it does not fully capture the quality of individual
knowledge source-level extractions. To address this
limitation, we introduce two complementary knowl-
edge source-level rewards: theSource-Grounded
Quality Rewardand theGraph-Readiness Reward.
Source-Grounded Quality Reward.The pro-
posed Deducibility Reward evaluates evidence for
the current question, but source extractions are in-
tended for reuse across queries. An extraction may
retain enough information to retrieve a relevant im-
age or answer the current question while omitting
other central content or introducing errors unre-
lated to that question. These deficiencies may leave
the Deducibility Reward unchanged, yet limit the
graph’s reliability and usefulness for other queries.
We therefore introduce the Source-Grounded Qual-
ity Reward to encourage the constructor to accu-
rately extract the knowledge source’s central knowl-
edge. A frozen judge model is prompted to eval-
uate each selected extraction against its original
knowledge source, without access to the question
or accepted answers.
The judge model evaluates two complementary
criteria: (i)reliability, requiring the extraction to
contain at least one meaningful, source-supported
assertion and no material factual or binding errors;
and (ii)central coverage, requiring the extraction
to capture the source’s key explicit information,
including entities, relations, values, and necessary
qualifiers, without requiring exhaustive coverage
of minor details.
An extraction receives a binary quality reward
u(q)
m,i= 1 only if both criteria are satisfied; oth-
erwise, u(q)
m,i= 0. Thus, an accurate extraction
that omits central information, or a comprehensiveextraction containing material errors, receives a re-
ward of 0. Source-Grounded Quality Reward is
considered only when all four candidate graphs
have valid judge evaluations and receive the same
Deducibility Reward of either 0.5or1. In these
groups, the retrieved evidence is sufficient to sup-
port a complete answer, so the graph-level reward
cannot distinguish among candidates. Source-level
evaluation then provides additional feedback by
assessing extraction accuracy and coverage of key
information that the question-specific reward may
overlook. This encourages the constructor to bet-
ter preserve information from the original sources,
even when candidates are equally answerable. For
each group, we sample at most one eligible source
to assess extraction quality, providing additional
supervision while limiting evaluation costs.
Graph-Readiness Reward.To encourage the
constructor to generate structurally valid and us-
able extractions, we introduce the Graph-Readiness
Reward. It considers both the format validity of the
original response, similar to format rewards used
in prior work (Jin et al., 2025a), and the usability
of the extracted content for MMKG construction.
For each response y(q)
m,i, we define a binary
Graph-Readiness reward r(q)
m,i∈ {0,1} . We set
r(q)
m,i= 1 only if the original output satisfies the
required JSON schema without repair, the response
terminates normally and passes the generation-
protocol checks, and it produces at least one usable
source-bound entity mention or fact. Otherwise,
r(q)
m,i= 0. Malformed responses receive a reward
of0even if the parser can recover their content, as
do structurally valid but empty responses.
4.4 GRPO-Based Optimization
We normalize Deducibility rewards d(q)
iacross
valid candidate graphs for each question to com-
pute the advantages A(q),D
m,i for their correspond-
ing source responses. The source-level advantages
A(q),Q
m,i andA(q),R
m,i are computed from separately
mean-centered rewards u(q)
m,iandr(q)
m,iacross re-
sponses generated for the same source. After apply-
ing the corresponding activation and source-credit
masks, the combined advantage for each source
responsey(q)
m,iis:
A(q)
m,i=A(q),D
m,i +λQA(q),Q
m,i +λRA(q),R
m,i,(6)
where λQandλRweight the two source-level ad-
vantages. Finally, we optimize the constructor with
5

Method MMQA MMDocRAG MMLongBench-Doc
Direct Inference 23.96 50.45 24.73
Naive RAG 29.79 59.46 25.38
Base GraphRAG (Dense) 33.75 67.12 40.65
F1-based RL (Dense) 35.83 71.17 40.43
MMKG-R1(Dense) 38.13 72.07 41.29
Base GraphRAG (Subgraph) 29.58 63.96 40.65
F1-based RL (Subgraph) 29.79 64.41 41.08
MMKG-R1(Subgraph) 33.96 65.32 41.51
Base GraphRAG (Hybrid) 40.83 66.67 41.94
F1-based RL (Hybrid) 38.54 68.02 42.15
MMKG-R1(Hybrid) 42.50 68.92 42.37
Table 1: Answer accuracy (%) on the three evaluation subsets. Qwen3-VL-8B-Instruct is used as the answer
model. Direct Inference and Naive RAG do not construct knowledge graphs. For MMLongBench-Doc, trained
constructors are transferred from MMDocRAG without additional training. Bold indicates the best result within
each dataset–graph-retrieval block.
GRPO by maximizing a dual-clipped surrogate ob-
jective (Ye et al., 2020).
5 Experiments
In this section, we conduct comprehensive experi-
ments to answer these research questions: (i)RQ1:
Can our proposed MMKG-R1 construct high-
quality multimodal knowledge graphs (MMKGs)
that improve the performance of multimodal Graph
RAG? (ii)RQ2: Are our proposed reward scores
more effective than the F1 score in guiding the
construction of high-quality MMKGs? and (iii)
(RQ3): How do different reward designs affect the
performance of MMKG-R1?
5.1 Experimental Setup
5.1.1 Datasets
We train two multimodal knowledge construc-
tors separately on the training sets of MM-
DocRAG (Dong et al., 2025) and MultiModalQA
(MMQA) (Talmor et al., 2021). We then evaluate
them on subsets of their respective test sets. We
use subsets for two reasons: (i) Some test ques-
tions share gold passages with training questions,
which may lead to data leakage and affect evalu-
ation reliability; (ii) Some test questions can be
answered using textual information alone, with-
out requiring visual evidence, making them less
suitable for evaluating multimodal knowledge con-
struction. To assess the transferability of MMKG-
R1, we further evaluate its performance on a subset
of MMLongBench-Doc (Ma et al., 2024). The
details of all used datasets are described in Ap-
pendix A.Answer Model Method Dense Subgraph Hybrid
32B-AWQBase 39.79 35.42 43.96
F1-based RL 40.00 36.46 46.46
MMKG-R140.63 38.54 48.96
Table 2: Performance of MMKG-R1 on MMQA with
Qwen3-VL-32B-Instruct-AWQ as the answer model.
Base denote Base Graph RAG.
5.1.2 Evaluation Protocol
We use a judge model to evaluate answer accu-
racy. To reduce potential evaluation bias, we use
Qwen3.5-27B (Team, 2026) as an independent
judge, distinct from the model used for calculat-
ing rewards during training. For each dataset, all
methods are evaluated using the same prompt and
evaluation criteria.
5.1.3 Baseline Methods
We compare MMKG-R1 with four baseline meth-
ods: (i)Direct Inference with Materials.We
retain the supporting materials and add distractor
within the model’s context budget. The frozen an-
swer model directly answers from this expanded
multimodal context. (ii)Naive RAG.We index
raw text and table chunks together with visual ma-
terials in a corpus-wide pool using a multimodal
embedding model. For each question, the top 20
units are retrieved by embedding similarity and
supplied to the frozen answer model. (iii)Base
GraphRAG.A constructor without reinforcement
learning extracts knowledge to build an MMKG for
downstream retrieval and question answering. (iv)
F1-based RL.This baseline adopts a correctness-
6

Dense Subgraph Hybrid3035404550Answer accuracy (%)40.63
38.5448.96
40.21
37.7147.29
39.58
38.3346.04MMKG-R1
MMKG-R1 w/o SGQ
Binary DeducibilityFigure 3: Reward ablations on MMQA subset. Metric
is answer accuracy (%).
oriented RL objective used in representative related
work (Lin et al., 2026), computing token-level F1
between the responses generated by a frozen an-
swer model and the reference answers. It retains the
Graph-Readiness Reward but excludes the Source-
Grounded Quality Reward.
5.1.4 Implementation Details
Computational Devices.All training and eval-
uation experiments are conducted on machines
equipped with four NVIDIA RTX A6000 GPUs,
each with 48 GB of memory.
Backbone Models.We use Qwen3-VL-4B-
Instruct (Bai et al., 2025) as the multimodal con-
structor. Both the Deducibility Reward and Source-
Grounded Quality Reward use frozen Qwen3-
VL-32B-Instruct-AWQ. The Graph-Readiness Re-
ward is computed using deterministic rules. The
downstream answer model is Qwen3-VL-8B-
Instruct (Bai et al., 2025) for all three datasets. All
methods use the same frozen answer model and
answer-evaluation protocol within each dataset.
Retrieval SettingsDuring training, we use a
fixed retrieval and evidence-packing procedure
with entity, fact, and relation entry points to com-
pute the Deducibility Reward. At evaluation time,
we assess the same trained constructor under three
different retrieval methods:Dense Retrieval,Sub-
graph Retrieval, andHybrid Retrieval.
More implementations are described in B.5.2 Main Results
Table 1 reports answer accuracy across the three
evaluation subsets. Direct Inference and Naive
RAG generally achieve lower accuracy than graph-
based retrieval methods. This highlights the bene-
fits of structured multimodal knowledge in graphs.
F1-based RL training improves downstream accu-
racy over Base GraphRAG in most settings, show-
ing that final-answer correctness can provide use-
ful feedback for MMKG construction. However,
these improvements are not consistent across all
settings. For example, on the MMQA subset with
hybrid retrieval, accuracy drops from 40.83% for
the untrained Base GraphRAG to 38.54% after F1-
based RL training, suggesting that answer-based
rewards may not always provide reliable guidance
for MMKG construction. Our proposed MMKG-
R1 consistently outperforms both Base GraphRAG
and F1-based RL, For example, on the MMQA
subset with dense retrieval, MMKG-R1 achieves
an accuracy of 38.13%, outperforming both Base
GraphRAG (33.75%) and F1-based RL (35.83%).
This demonstrates that our quality-aware rewards
provide effective guidance for MMKG construction
5.3 Transferability of MMKG-R1
Different Answer Models.On the MMQA sub-
set, we additionally evaluate Qwen3-VL-32B-
Instruct-AWQ as the answer model, alongside
Qwen3-VL-8B-Instruct. The results are shown
in Table 2. MMKG-R1 consistently constructs
MMKGs that achieve higher answer accuracy
across all three retrieval strategies, even when eval-
uated with a different answer model.
Cross-Dataset Transferability.To evaluate the
transferability of MMKG-R1, We further eval-
uate the MMDocRAG-trained constructors on
MMLongBench-Doc without further training. As
shown in Table 1, MMKG-R1 consistently out-
performs both Base GraphRAG and F1-based RL
across all three retrieval strategies. These results
suggest that our quality-aware rewards help the con-
structor learn knowledge construction capabilities
that transfer to unseen datasets.
5.4 Ablation Studies
To examine the contributions of our reward de-
sign, we evaluate two variants on the MMQA sub-
set: (i)MMKG-R1 w/o Source-Grounded Quality
(SGQ) Reward.This variant removes the SGQ Re-
ward while retaining the Deducibility and Graph-
7

0% 25% 50% 75% 100%
Training progress0%10%20%30%40%50%60%Discriminative groups (%)
(a) Reward discrimination
0% 25% 50% 75% 100%
Training progress051015Triples amount
(b) Triples amount
0% 25% 50% 75% 100%
Training progress0%10%20%30%Delivery failure rate (%)
(c) Delivery failureMMKG-R1 MMKG-R1 w/o SGQ Binary DeducibilityFigure 4: Training dynamics on MMQA. (a) Proportion of discriminative question groups during training with our
Deducibility Reward and its binary variant. (b,c) Mean number of triples per source output and output delivery
failure rates for MMKG-R1 and MMKG-R1 w/o SGQ.
Readiness Rewards. (ii)Binary Deducibility.This
variant removes the SGQ Reward and binarizes the
Deducibility Reward by mapping both 0.5and1
to1, while keeping 0unchanged. It no longer dis-
tinguishes between incomplete and complete struc-
tured reasoning paths when the evidence supports
the answer. We use Qwen3-VL-32B-Instruct-AWQ
for answer generation and evaluation in this abla-
tion study. As shown in Figure 3, full MMKG-R1
consistently outperforms both variants. Remov-
ing the SGQ Reward reduces accuracy, while bi-
narizing the Deducibility Reward generally leads
to further degradation. These results suggest that
MMKG-level deducibility feedback alone does not
fully substitute for source-grounded quality super-
vision. We further examine how our Deducibility
Reward compares with its binary variant during
training, and how adding SGQ affects the construc-
tor’s outputs.
Deducibility Reward vs. Its Binary Variant.
We first compare how our Deducibility Reward
and its binary variant distinguish candidate graphs
for the same question. A group is discriminative
if at least three valid candidates remain and their
task rewards are not all identical. Figure 4(a) re-
ports the proportion of such groups among all sam-
pled groups, with SGQ disabled in both variants.
Our reward yields a higher proportion across all
reported windows. Unlike the binary variant, it
distinguishes incomplete from complete structured
reasoning paths even when both support the answer,
providing finer feedback for relative optimization.
Effect of SGQ.We next examine how adding
SGQ affects the constructor’s outputs using two
metrics: (1)Triples amount, defined as the aver-age number of triples obtained from each source
output after parsing and deduplication; and (2)
Output delivery failure rate, defined as the pro-
portion of source outputs that fail the structural,
normal-termination, or nonempty-output require-
ments. Figure 4(b,c) compares these metrics
throughout training between full MMKG-R1 and
MMKG-R1 w/o SGQ. As training progresses, the
full model retains more triples while exhibiting
fewer output delivery failures. These observations
complement the downstream accuracy gains, sug-
gesting that SGQ helps preserve structured infor-
mation and improve output reliability.
Together, these analyses support the complemen-
tary roles of the two rewards: our Deducibility Re-
ward provides differentiated task feedback among
candidates for the same question, while SGQ sup-
plies additional quality supervision at the source
level. Both contribute to training the constructor.
6 Conclusion
We introduce MMKG-R1, a novel reinforcement
learning framework that trains a multimodal con-
structor to help build useful multimodal knowl-
edge graphs (MMKGs) for downstream question
answering. MMKG-R1 combines the graph-level
Deducibility Reward with two knowledge-source-
level rewards, the Source-Grounded Quality Re-
ward and the Graph-Readiness Reward, to provide
complementary feedback on evidence sufficiency,
source fidelity, and usability. Results across mul-
tiple datasets show that the multimodal construc-
tor trained with MMKG-R1 can effectively extract
knowledge to build useful MMKGs, while demon-
strating strong generalization to unseen datasets.
8

Limitations
The Deducibility and Source-Grounded Quality re-
wards rely on external multimodal large language
models as judges, making their reliability depen-
dent on the quality of the judge model’s assess-
ments.
Ethical considerations
This work aims to train a multimodal large lan-
guage model to extract high-quality knowledge
from multimodal sources for constructing multi-
modal knowledge graphs. All experiments are con-
ducted based on publicly available datasets. We do
not anticipate significant ethical or societal risks
arising from this research.
References
Mohammad Mahdi Abootorabi, Amirhosein Zobeiri,
Mahdi Dehghani, Mohammadali Mohammadkhani,
Bardia Mohammadi, Omid Ghahroodi, Mahdieh So-
leymani Baghshah, and Ehsaneddin Asgari. 2025.
Ask in any modality: A comprehensive survey on
multimodal retrieval-augmented generation. InFind-
ings of the Association for Computational Linguistics:
ACL 2025.
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhi-
fang Guo, Qidong Huang, Jie Huang, Fei Huang,
Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng
Li, and 45 others. 2025. Qwen3-vl technical report.
Preprint, arXiv:2511.21631.
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He,
Zongbo Han, Zheng Zhang, and Mike Zheng Shou.
2024. Hallucination of multimodal large language
models: A survey.Preprint, arXiv:2404.18930.
Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga,
and William Cohen. 2022. MuRAG: Multimodal
retrieval-augmented generator for open question an-
swering over images and text. InProceedings of the
2022 Conference on Empirical Methods in Natural
Language Processing.
Sijun Dai, Qiang Huang, Xiaoxing You, and Jun Yu.
2026. Mg2-rag: Multi-granularity graph for multi-
modal retrieval-augmented generation. InEuropean
Conference on Computer Vision (ECCV).
Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng
Wang, Ruiming Tang, and Yong Liu. 2025. Bench-
marking retrieval-augmented multimomal generation
for document question answering. InAdvances in
Neural Information Processing Systems.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
Preprint, arXiv:2404.16130.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2023. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and
Chao Huang. 2025a. Rag-anything: All-in-one rag
framework.Preprint, arXiv:2510.12323.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025b. LightRAG: Simple and fast retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025.
Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-
Wei Chang, Yizhou Sun, Cordelia Schmid, David A.
Ross, and Alireza Fathi. 2023. Reveal: Retrieval-
augmented visual-language pre-training with multi-
source multimodal knowledge memory. In2023
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR).
Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O.
Arik, and Jiawei Han. 2025a. An empirical study
on reinforcement learning for reasoning-search inter-
leaved llm agents.Preprint, arXiv:2505.15117.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan O Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. 2025b. Search-r1: Training LLMs to
reason and leverage search engines with reinforce-
ment learning. InSecond Conference on Language
Modeling.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serv-
ing with pagedattention. InProceedings of the 29th
Symposium on Operating Systems Principles.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems.
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang,
Wayne Xin Zhao, and Ji-Rong Wen. 2023. Eval-
uating object hallucination in large vision-language
models. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing.
Wanying Liang, Pasquale De Meo, Yong Tang, and
Jia Zhu. 2024. A survey of multi-modal knowledge
graphs: Technologies and trends.ACM Computing
Surveys, 56.
9

Jiashi Lin, Changhong Jiang, Xiangru Lin, Ruifei
Zhang, Xinyi Zhu, Jiyao Liu, Cheng Tang, Ye Du,
Shujian Gao, Junzhi Ning, Lihao Liu, Ziyan Huang,
Tianbin Li, Jin Ye, and Junjun He. 2026. Evograph-
r1: Self-evolving multimodal knowledge hyper-
graphs for agentic retrieval. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR).
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae
Lee. 2024. Improved baselines with visual instruc-
tion tuning. InProceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition
(CVPR).
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. 2023. Visual instruction tuning. InAdvances in
Neural Information Processing Systems.
Haoran Luo, Haihong E, Guanting Chen, Qika Lin,
Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song,
Xiaobao Wu, Yifan Zhu, and Anh Tuan Luu. 2026.
Graph-r1: Towards agentic graphRAG framework
via end-to-end reinforcement learning. InForty-third
International Conference on Machine Learning.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang
Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024.
Mmlongbench-doc: Benchmarking long-context doc-
ument understanding with visualizations. InAd-
vances in Neural Information Processing Systems.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Prox-
imal policy optimization algorithms.Preprint,
arXiv:1707.06347.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2025. Hybridflow: A flexible
and efficient rlhf framework. InProceedings of the
Twentieth European Conference on Computer Sys-
tems.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-
Yeung Shum, and Jian Guo. 2024. Think-on-graph:
Deep and responsible reasoning of large language
model on knowledge graph. InInternational Confer-
ence on Learning Representations.
Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav,
Yizhong Wang, Akari Asai, Gabriel Ilharco, Han-
naneh Hajishirzi, and Jonathan Berant. 2021. Mul-
timodalqa: complex question answering over text,
tables and images. InInternational Conference on
Learning Representations.Qwen Team. 2026. Qwen3.5: Towards native multi-
modal agents.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, Aurelien Rodriguez, Armand Joulin, Edouard
Grave, and Guillaume Lample. 2023. Llama: Open
and efficient foundation language models.Preprint,
arXiv:2302.13971.
Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao,
Tianshi Zheng, Baixuan Xu, Shujie Liu, and Yangqiu
Song. 2026. Autograph-r1: End-to-end reinforce-
ment learning for knowledge graph construction. In
Proceedings of the 64th Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers).
Xueyao Wan and Hang Yu. 2025. Mmgraphrag:
Bridging vision and language with inter-
pretable multimodal knowledge graphs.Preprint,
arXiv:2507.20804.
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi-
hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei
Du, Xuancheng Ren, Rui Men, Dayiheng Liu,
Chang Zhou, Jingren Zhou, and Junyang Lin. 2024.
Qwen2-vl: Enhancing vision-language model’s per-
ception of the world at any resolution.Preprint,
arXiv:2409.12191.
Xiaochen Wang, Zongyu Wu, Yuan Zhong, Xiang
Zhang, Suhang Wang, and Fenglong Ma. 2026. Gpr:
Empowering generation with graph-pretrained re-
triever. InProceedings of the ACM Web Conference
2026.
Jinming Wu, Zihao Deng, Wei Li, Yiding Liu, Bo You,
Bo Li, Zejun MA, and Ziwei Liu. 2026a. MMSearch-
r1: Incentivizing LMMs to search. InProceedings
of the 64th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers).
Zongyu Wu, Yilong Wang, Xiaochen Wang, Minhua
Lin, Zhichao Xu, Fenglong Ma, Xiang Zhang, and
Suhang Wang. 2026b. Multi-granularity context-
enhanced rag over multimodal knowledge graphs.
Preprint, arXiv:2608.25986.
Zhichao Xu, Zongyu Wu, Yun Zhou, Aosong Feng,
Kang Zhou, Sangmin Woo, Kiran Ramnath, Yijun
Tian, Xuan Qi, Weikang Qiu, Lin Lee Cheong, and
Haibo Ding. 2026. Beyond correctness: Rewarding
faithful reasoning in retrieval-augmented generation.
Transactions on Machine Learning Research.
Deheng Ye, Zhao Liu, Mingfei Sun, Bei Shi, Peilin
Zhao, Hao Wu, Hongsheng Yu, Shaojie Yang, Xipeng
Wu, Qingwei Guo, et al. 2020. Mastering complex
control in moba games with deep reinforcement learn-
ing. InProceedings of the AAAI conference on artifi-
cial intelligence.
10

Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing
Sun, Tong Xu, and Enhong Chen. 2024. A survey on
multimodal large language models.National Science
Review, 11(12).
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. Vis-
RAG: Vision-based retrieval-augmented generation
on multi-modality documents. InThe Thirteenth In-
ternational Conference on Learning Representations.
Xu Yuan, Liangbo Ning, Qingqing Ye, Wenqi Fan, and
Qing Li. 2026. mkg-rag: Leveraging multimodal
knowledge graphs in retrieval-augmented generation
for knowledge-intensive vqa. InProceedings of the
49th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval.
Zhiqiang Zhang and Wen Zhao. 2025. A collabora-
tive reasoning framework powered by reinforcement
learning and large language models for complex ques-
tions answering over knowledge graph. InProceed-
ings of the 31st International Conference on Compu-
tational Linguistics.
A Dataset Details
MMDocRAG.MMDocRAG (Dong et al., 2025)
is designed to evaluate multimodal evidence re-
trieval and question answering over multimodal
documents. MMDocRAG is licensed under the
Apache License 2.0.
MultiModalQA.MultiModalQA (MMQA) (Tal-
mor et al., 2021) is built from Wikipedia tables
linked to text passages and images, and is designed
to evaluate complex question answering involving
cross-modal and multi-hop reasoning. In our ex-
periments, text and images are preserved in their
original forms, while structured tables are serial-
ized into text. MMQA is publicly available, but no
explicit license is specified.
MMLongBench-Doc.MMLongBench-Doc (Ma
et al., 2024) evaluates long-document understand-
ing involving information like text and images. We
select answerable questions with valid supporting
pages and evidence types that include figures or
charts. This subset is used only for cross-dataset
transfer evaluation with the constructor trained on
MMDocRAG. MMLongBench-Doc is licensed un-
der the CC BY-NC 4.0.
B More Implementation Details
For constructor rollouts during the training phase,
we set the sampling temperature to 1.0 and the max-
imum output length to 4,096 tokens. We build ourtraining pipeline using verl (Sheng et al., 2025) and
use vLLM (Kwon et al., 2023) for answer model
inference.
C AI Usage Disclosure
The authors used generative AI tools to assist with
writing and coding. All AI-assisted content and
code were reviewed and verified by the authors
11