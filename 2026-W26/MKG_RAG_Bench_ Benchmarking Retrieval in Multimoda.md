# MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph-Augmented Generation

**Authors**: Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, Fenglong Ma

**Published**: 2026-06-24 23:38:42

**PDF URL**: [https://arxiv.org/pdf/2606.26458v1](https://arxiv.org/pdf/2606.26458v1)

## Abstract
Retrieval-augmented generation (RAG) over knowledge graphs has emerged as a promising approach for grounding large language models, yet existing benchmarks largely overlook the challenges of retrieval in multimodal knowledge graph RAG (MKG-RAG). In practice, retrieval is a critical bottleneck: multimodal knowledge is heterogeneous, difficult to align across modalities, and often poorly served by retrievers designed for unstructured corpora. To address this gap, we introduce MKG-RAG-Bench, a cross-domain benchmark explicitly designed to evaluate retrieval in MKG-RAG. MKG-RAG-Bench is constructed from two multimodal knowledge graphs spanning general and medical domains, and includes carefully aligned question-answering datasets that support controlled evaluation of both retrieval and downstream generation. The benchmark is built using an LLM-based curation pipeline that filters low-utility knowledge, generates structurally grounded queries with exact supervision, and systematically covers diverse modality configurations. Through extensive experiments across representative retriever families and modality settings, we show that effective multimodal retrieval remains challenging yet crucial for end-to-end MKG-RAG performance, and that retrieval quality strongly determines generation outcomes. By isolating retrieval as a first-class evaluation target, MKG-RAG-Bench provides a principled foundation for diagnosing current limitations and advancing multimodal knowledge graph RAG systems.

## Full Text


<!-- PDF content starts -->

MKG-RAG-Bench: Benchmarking Retrieval in Multimodal
Knowledge Graph–Augmented Generation
Xiaochen Wang
The Pennsylvania State University
University Park, Pennsylvania, USA
xcwang@psu.eduBao Hoang
Michigan State University
East Lansing, Michigan, USA
hoangbao@msu.eduHan Liu
Dalian University of Technology
Dalian, Liaoning, China
liu.han.dut@gmail.com
Ting Wang
Stony Brook University
Stony Brook, New York, USA
wang@cs.stonybrook.eduFenglong Ma∗
The Pennsylvania State University
University Park, Pennsylvania, USA
fenglong@psu.edu
Abstract
Retrieval-augmented generation (RAG) over knowledge graphs
has emerged as a promising approach for grounding large lan-
guage models, yet existing benchmarks largely overlook the chal-
lenges of retrieval in multimodal knowledge graph RAG (MKG-
RAG). In practice, retrieval is a critical bottleneck: multimodal
knowledge is heterogeneous, difficult to align across modalities,
and often poorly served by retrievers designed for unstructured
corpora. To address this gap, we introduceMKG-RAG-Bench, a
cross-domain benchmark explicitly designed to evaluate retrieval in
MKG-RAG.MKG-RAG-Benchis constructed from two multimodal
knowledge graphs spanning general and medical domains, and in-
cludes carefully aligned question-answering datasets that support
controlled evaluation of both retrieval and downstream genera-
tion. The benchmark is built using an LLM-based curation pipeline
that filters low-utility knowledge, generates structurally grounded
queries with exact supervision, and systematically covers diverse
modality configurations. Through extensive experiments across
representative retriever families and modality settings, we show
that effective multimodal retrieval remains challenging yet crucial
for end-to-end MKG-RAG performance, and that retrieval quality
strongly determines generation outcomes. By isolating retrieval as
a first-class evaluation target,MKG-RAG-Benchprovides a princi-
pled foundation for diagnosing current limitations and advancing
multimodal knowledge graph RAG systems.1
CCS Concepts
•Computing methodologies →Continuous space search;
Natural language generation.
∗Corresponding author.
1The benchmark can be accessed via: https://github.com/XiaochenWang-PSU/MKG-
RAG-Bench.
This work is licensed under a Creative Commons Attribution 4.0 International License.
KDD ’26, Jeju Island, Republic of Korea
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2259-2/2026/08
https://doi.org/10.1145/3770855.3817570Keywords
Multimodal retrieval-augmented generation, multimodal knowl-
edge graphs
ACM Reference Format:
Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma. 2026.
MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge
Graph–Augmented Generation. InProceedings of the 32nd ACM SIGKDD
Conference on Knowledge Discovery and Data Mining V.2 (KDD ’26), Au-
gust 09–13, 2026, Jeju Island, Republic of Korea.ACM, New York, NY, USA,
12 pages. https://doi.org/10.1145/3770855.3817570
1 Introduction
Retrieval-augmented generation (RAG) enhances large language
models (LLMs) by retrieving information relevant to a given query
from external sources and conditioning generation on this retrieved
evidence [ 16]. While RAG has proven effective in improving fac-
tual grounding, conventional approaches predominantly retrieve
from unstructured textual corpora that are noisy, fragmented, and
weakly connected. These limitations often hinder reliable evidence
selection and multi-step reasoning, especially for complex queries.
To address these issues,knowledge graph–based RAG(KG-RAG)
has been proposed [ 8]. By retrieving information from structured
knowledge graphs, KG-RAG enables more coherent, contextualized,
and comprehensive knowledge access, leading to improved answer
quality and faithfulness [ 39]. However, existing KG-RAG methods
are largely confined totextual knowledge graphs, which substan-
tially limits their applicability in real-world scenarios where critical
information is conveyed through images, charts, tables, and other
non-textual modalities.
Multimodal KG-RAG.To overcome this limitation, we explore the
task, multimodal KG-RAG (MKG-RAG), which integrates structured
knowledge across multiple modalities to support more expressive
retrieval and reasoning. Formally, given a query 𝑞, a multimodal
knowledge graphG, a retriever Retriever , and a multimodal large
language model MLLM , the goal ofMKG-RAGis to generate a
responseR, i.e.,
R=MLLM(𝑞,Retriever(𝑞,G)).
The most closely related setting is multimodal RAG [ 3], which fol-
lows the traditional RAG pipeline by retrieving multimodal content
from unstructured external corpora (e.g., Wikipedia) rather than
from a structured multimodal knowledge graph. To the best of ourarXiv:2606.26458v1  [cs.AI]  24 Jun 2026

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma
Multimodal  Knowledge  Graphs  
(MarKG /MedMKG )
MKG Raw  TripletsLLM -based  Utility  
Filtering
High -Utility  
TripletsControlled  
Masking
??
Hybrid  Question  
Synthesis
Text-only
Query
Multimodal
QueryRetrieval  Data
Generation  DataQuery
Positive  Triplet
Hard  Negatives
Query
Evidence
Answer
InputBenchmark  Construction  PipelineEvaluation  Data
Figure 1: The proposed pipeline for benchmark construction using multimodal knowledge graphs and LLMs.
knowledge,no existing benchmark or prior worksystematically
studies retrieval and generation in theMKG-RAGsetting.
Motivation andChal lenges.Multimodal KG-RAG is benchmark-
worthy because it evaluates a model’s ability to retrieve, align, and
reason overstructured multimodal knowledge, a critical capability
for real-world systems that is not captured by existing RAG, KG-
RAG, or multimodal benchmarks. This work aims to introduce
thefirstbenchmark for theMKG-RAGtask, consisting of curated
datasets and evaluation protocols that explicitly target retrieval
quality and multimodal grounding. Designing such a benchmark,
however, poses several non-trivial challenges:
•Heterogeneous Retrieval.Unlike conventional knowledge
graphs, where both queries and target triplets are unimodal,
multimodal knowledge graphs support heterogeneous modal-
ity combinations across queries and retrieval targets. InMKG-
RAG, queries may be text-only, image-only, or multimodal, while
relevant graph components may combine textual, visual, and
numerical information in different ways. These combinations in-
duce distinct retrieval behaviors and demand different modeling
strategies. A meaningfulMKG-RAGbenchmark must therefore
systematically cover diverse multimodal retrieval scenarios to
reflect realistic usage and to enable fair comparison among re-
trievers.
•Multimodal Knowledge Graph–Query Alignment.InMKG-
RAG, relevant knowledge may not be explicitly aligned with
the query at the surface level, particularly when the query and
target nodes differ in modality or abstraction. For example, a
textual query may require grounding in visual evidence, while
an image-based query may necessitate retrieving semantically re-
lated textual entities and relations. Addressing such cross-modal
and cross-structural alignment requires joint reasoning over
graph topology, semantic relationships, and modality-specific
representations. Accordingly, aMKG-RAGbenchmark must in-
clude queries that demand non-trivial alignment beyond shallow
modality matching.
•Benchmark Utility.The primary goal of the benchmark is to
evaluate both the effectiveness of multimodal retrieval and the
grounding quality of multimodal LLMs. This requires careful con-
struction to ensure that the retrieved knowledge is informative
and well-aligned with downstream queries. Multimodal knowl-
edge graphs often contain generic or weakly informative triplets,
which can lead to ambiguous or poorly grounded questions andobscure the contribution of retrieval. A usefulMKG-RAGbench-
mark must prioritize high-utility triplets and disentangle im-
provements due to effective retrieval from those arising from
generative priors.
Our Solution.To address these challenges, we propose a cross-
domain, retrieval-orientedMKG-RAGbenchmark, namedMKG-
RAG-Bench, comprising two multimodal knowledge graphs and
corresponding multimodal question-answering datasets. The bench-
mark is designed to enable comprehensive evaluation across both
the retrieval and generation stages of multimodal KG-RAG. The
benchmark construction pipeline is shown in Figure 1. Specifically,
we use an LLM to select high-utility triplets and employ heuristic
strategies to ensure strong alignment between the selected triplets
and the constructed queries. The same LLM, with different prompts,
is further leveraged to generate multimodal, knowledge-grounded
questions across diverse retrieval scenarios, thereby systematically
covering heterogeneous modality combinations. We conduct exten-
sive experiments with representative retriever architectures across
different multimodal settings, evaluating both effectiveness and
efficiency throughout the retrieval and generation pipeline.While
this work primarily focuses on evaluatingMKG-RAGusingtraining-
freeapproaches, the benchmark is also applicable to training-based
methods, including retriever training and MLLM fine-tuning.
Contributions.In summary, our main contributions are as follows:
•We identify a critical research gap inMKG-RAG, highlighting
that retrieval, despite its central role, is underexplored and insuf-
ficiently evaluated in existing multimodal RAG research.
•We introduce the first, cross-domain, retrieval-orientedMKG-
RAGbenchmark comprising two multimodal knowledge graphs
and corresponding question-answering datasets, enabling sys-
tematic evaluation of retrieval and downstream generation.
•We propose a benchmark curation pipeline based on LLMs that
directly addresses key challenges in heterogeneous retrieval, mul-
timodal alignment, and benchmark utility.
•We develop a unified evaluation framework and conduct compre-
hensive experiments across diverse multimodal settings, demon-
strating the importance of effective multimodal retrieval and
enabling future research on more powerful retrievers forMKG-
RAG.

MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph–Augmented Generation KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
Table 1: Comparison of representative multimodal RAG and KG-RAG benchmarks.
Benchmark
NameKnowledge Source
TypeKnowledge
StructureRetrieval
UnitHeterogeneous
Query SupportEvaluation
GranularityCross-Domain
Coverage
OK-VQA [26] Web Unstructured Passage Multimodal Generation-only Single-domain
M2RAG [25] Web Unstructured Passage and Image Multimodal Generation-only Single-domain
CRAG-MM [37] Web Ad-hoc MKG Passage and Image Multimodal Generation-only Multi-domain
Dyn-VQA [20] Web Unstructured Passage and Image Multimodal Generation-only Multi-domain
WebQSP [43] Textual KG Textual KG Textual Triplet Text-only Generation-only Single-domain
CWQ [35] Textual KG Textual KG Textual Triplet Text-only Generation-only Single-domain
GrailQA [10] Textual KG Textual KG Textual Triplet Text-only Generation-only Multi-domain
REAL-MM-RAG [41] Multimodal Documents PDF Chunk Text-only Retrieval-only Multi-domain
DocBench [46] Multimodal Documents PDF Chunk Text-only Generation-only Multi-domain
MMLongBench [24] Multimodal Documents PDF Chunk Text-only Generation-only Multi-domain
MMDocRAG [6] Multimodal Documents PDF Chunk Text-only Retrieval & Generation Multi-domain
MKG-RAG-Bench Multimodal KG Explicit MKG Multimodal Triplet Multimodal Retrieval & Generation Multi-domain
2 Preliminaries
2.1 Benchmark Comparisons
The most closely related line of work is multimodal RAG (M-RAG),
which focuses on retrieving multimodal content, such as textual
passages and images, from large unstructured corpora. Retrieved
multimodal evidence can be directly incorporated into the input
of multimodal large language models [ 20,25], or reorganized into
ad-hoc multimodal structures to further enhance generation [ 37].
However, M-RAG follows the conventional RAG paradigm and does
not assume access to a structured multimodal knowledge graph.
As a result, retrieval is performed overunstructured corpora, and
any induced structure is transient and query-specific rather than
explicitly represented in the knowledge source.
Another related line of work is knowledge graph–based RAG
(KG-RAG), which aims to retrieve relevant triplets or subgraphs
from a given knowledge graph to support grounded generation.
Existing KG-RAG approaches [ 18,23], however, primarily focus on
textual knowledge graphsand do not explicitly model or retrieve
multimodal information. Consequently, they are not designed to
evaluate or support retrieval and reasoning over structured multi-
modal knowledge.
In contrast,MKG-RAGoperates overpre-defined multimodal
knowledge graphs, where entities, relations, and cross-modal con-
nections are explicitly modeled. This fundamentally changes the
retrieval problem: instead of selecting isolated multimodal doc-
uments,MKG-RAGrequires identifying relevant subgraphs that
integrate heterogeneous modalities through structured relations.
Consequently, existing M-RAG and KG-RAG benchmarks are insuf-
ficient for evaluatingMKG-RAG, as they do not capture multimodal
graph-based retrieval or disentangle retrieval quality from down-
stream generation. Table 1 compares representative benchmarks
across multiple perspectives, highlighting the distinctive character-
istics ofMKG-RAG.
2.2 Retrievers
Although theMKG-RAGtask differs from conventional RAG, mul-
timodal RAG, and KG-RAG, many existing retrieval techniques canbe directly adapted to this setting. In this work, we categorize com-
monly used retrievers into four groups based on their underlying
retrieval strategies:
•Text-only retrievers[ 14,29,30]. Text-only retrievers treat both
queries and candidate triplets as plain text and rank candidates
using textual similarity, such as sparse lexical matching or dense
semantic embeddings, without incorporating visual information.
•Fusion-based multimodal retrievers[ 3,27]. Fusion-based
retrievers encode textual and visual information into a shared
representation space and perform retrieval by measuring cross-
modal similarity between multimodal queries and multimodal
targets.
•Captioning-based retrievers[ 9,44]. Captioning-based retriev-
ers first convert visual content into textual descriptions using an
external image captioning model, and then reduce multimodal
retrieval to text-only retrieval over the generated captions.
•Reranking-based retrievers[ 4,42]. Reranking approaches typ-
ically adopt a multi-stage pipeline in which a lightweight first-
stage retriever filters a candidate set, followed by a more powerful
reranker that performs fine-grained relevance estimation and
final selection.
To ensure fair and controlled comparison, we evaluate all retrievers
in atraining-freesetting by directly using their pretrained weights,
without fine-tuning on the proposed benchmark.
2.3 Preliminary Analysis
A seemingly straightforward approach to constructing a multi-
modal knowledge graph RAG benchmark is to directly combine
existing multimodal knowledge graphs, such as MMKG [ 22] and
TIVA-KG [ 38], with modality-relevant downstream tasks, e.g., A-
OKVQA [ 33] and K-VQA [ 31]. However, as discussed in Section 1,
such a naive combination fails to adequately capture the require-
ments of theMKG-RAGtask and often leads to severe misalignment
between the knowledge source, retrieval objectives, and down-
stream evaluation. In particular, the knowledge required to answer
a given query may not be present in the knowledge graph, while
irrelevant retrieved content may dominate the model’s input, intro-
ducing noise and ultimately degrading performance.
To empirically illustrate this issue, we conduct a preliminary
study using MedMKG [ 40], a multimodal medical knowledge graph

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma
RAG-free Text-onlyCaptioningFusion
Reranking2022242628303234BLEU (%)31.47
25.92
25.1431.26
28.81(a) VQA-RAD
RAG-free Text-onlyCaptioningFusion
Reranking30.36
28.74
26.6528.1629.29(b) SLAKE
Figure 2: BLEU (%) comparison on (a) VQA-RAD and (b)
SLAKE datasets. RAG-free consistently achieves the best per-
formance across both benchmarks.
constructed from MIMIC clinical data, to support medical vi-
sual question answering on two widely used benchmarks, VQA-
RAD [ 15] and SLAKE [ 21]. Specifically, we augment multimodal
large language models with retrieval from MedMKG and evaluate
their performance under different retrieval strategies. The results
are summarized in Figure 2.
Across all experimental settings, models augmented with mul-
timodal retrieval consistently underperform their RAG-free coun-
terparts. Although multimodal retrievers outperform text-only re-
trievers, they still fail to surpass generation without retrieval aug-
mentation. These results indicate that retrieval from MedMKG does
not effectively support the downstream visual question answering
tasks. We attribute this behavior to two primary factors: (1) task-
critical knowledge required by the MLLM may be absent from the
multimodal knowledge graph, and (2) task-irrelevant knowledge
present in the graph may be retrieved and injected into the model
input, introducing substantial noise that interferes with generation.
Taken together, these observations reveal a fundamental limita-
tion of directly combining existing multimodal knowledge graphs
with downstream multimodal tasks for MKG-RAG. The resulting
misalignment between knowledge sources, retrieval targets, and
evaluation objectives undermines the effectiveness of retrieval aug-
mentation and obscures the true capabilities of multimodal KG-RAG
systems. This motivates the need for a dedicatedMKG-RAGbench-
mark that explicitly aligns multimodal knowledge graphs, retrieval
objectives, and downstream evaluation—precisely the goal of the
benchmark proposed in this work.
3 Benchmark Construction
An effective benchmark should not only cover diverse data sources
but also support rigorous evaluation across multiple stages of a
task. In the proposedMKG-RAGbenchmark, we construct datasets
from two multimodal knowledge graphs spanning distinct domains:
MarKG [ 45] for the general domain and MedMKG [ 40] for the
medical domain2. This cross-domain design allows us to evalu-
ate the generality of multimodal KG-RAG methods under vary-
ing knowledge structures and modality distributions. To enable
faithful evaluation of both retrieval and generation in multimodal
KG-RAG, we design a principled data construction pipeline that
filters out low-utility knowledge, generates structurally aligned and
knowledge-grounded queries with exact supervision, and supports
explicit evaluation of both retrieval and generation stages.
2The details of these two multimodal knowledge graphs are provided in Appendix A.3.1 Dataset Curation
As shown in Figure 1, our construction pipeline consists of three
stages: (1) LLM-based utility filtering, (2) controlled completeness
simulation via triplet masking, and (3) hybrid question synthesis.
Each stage is designed to address a specific challenge in benchmark-
ing multimodal KG-RAG.
Step 1: LLM-based Utility Filtering.A central challenge in bench-
marking multimodal KG-RAG is ensuring that retrieval contributes
meaningful and well-grounded evidence to downstream tasks. Mul-
timodal knowledge graphs often contain triplets that are overly
generic, definitional, or weakly informative, from which derived
queries may be ambiguous or insufficiently grounded, limiting their
usefulness for evaluating retrieval.
Given a triplet, we prompt an LLM to assess itsretrieval utility,
i.e., whether it can support a clear and informative information-
seeking query. Triplets deemed low-utility are those that tend to
yield questions answerable without external evidence or that fail to
convey a specific retrieval intent [ 43]. For example, the triplet (Sun,
is_a, star) may lead to the question “What is the Sun?”, which nei-
ther requires retrieval nor benefits from structured knowledge. We
remove such triplets from the benchmark to ensure that generated
queries are well specified and grounded in retrievable knowledge.
This filtering step produces a refined triplet set that better reflects
realistic retrieval scenarios in which external knowledge access
is necessary for accurate reasoning. The prompt templates and
decision criteria are detailed in Appendix B.
Step 2: Controlled Completeness Simulation.To construct
queries that require knowledge retrieval while preserving unam-
biguous supervision, we simulate controlled incompleteness over
the retained triplets. Rather than presenting complete knowledge
graph facts, we mask one component of each triplet and treat the
remaining components as observed context, thereby mimicking
information-seeking queries.
For each triplet(ℎ,𝑟,𝑡) , we construct two masked variants: (i)
relation masking, where the relation 𝑟is masked and must be in-
ferred from the head and tail, and (ii)tail masking, where the tail 𝑡
is masked and must be inferred from the head and relation. We keep
the head entity unchanged as a semantic anchor, which facilitates
multimodal grounding when visual information is available and
ensures that each query is uniquely supported by its originating
knowledge.
We avoid masking the head entity, as doing so would require
entity disambiguation beyond the scope of retrieval evaluation
and could introduce additional ambiguity. This design allows us to
focus on assessing retrievers’ ability to identify relevant relations
and entities under realistic multimodal conditions. The resulting
masked triplets serve as structurally grounded templates for query
generation.
Step 3: Hybrid Question Synthesis.Masked triplets provide
structured but non-linguistic representations and do not directly
resemble natural user queries. To generate realistic questions while
preserving exact alignment with the underlying knowledge, we
again employ GPT-5 [ 34] to synthesize natural-language questions
from masked triplets under two complementary modes.

MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph–Augmented Generation KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
Table 2: Statistics ofMKG-RAG-Bench, where %MM refers to the percentage of multimodal data.
Dataset SplitRetrieval Generation
Query Triplet Question Answer
# of Queries %MM AvgLen # of Triplets %MM AvgLen # of Questions %MM AvgLen # of Answers AvgLen
MKG-RAG-Bench-GTrain 48,908 30.1 12.2 25,517 28.8 6.8 48,908 30.1 12.2 61,001 2.5
Val 6,113 31.0 12.2 25,517 28.8 6.8 6,113 31.0 12.2 7,586 2.5
Test 6,115 30.3 12.2 25,517 28.8 6.8 6,115 30.3 12.2 7,795 2.5
MKG-RAG-Bench-MTrain 4,781 40.6 16.2 18,468 49.0 8.7 4,781 40.6 16.2 16,564 3.7
Val 597 42.2 16.1 18,468 49.0 8.7 597 42.2 16.1 2,030 3.5
Test 599 43.1 16.2 18,468 49.0 8.7 599 43.1 16.2 2,074 3.6
Text-only question synthesis.For masked triplets derived from
text-only knowledge, GPT-5 generates textual questions that ex-
plicitly correspond to the missing component. For example, from
(penicillin,[Mask],bacterial infection) , the LLM generates: “What
is the relationship between penicillin and bacterial infection?” This
produces text-only queries whose answers are uniquely grounded
in the masked triplet.
Image-grounded question synthesis.For masked triplets whose
head entity is associated with an image, we provide the image to
GPT-5 and require the head entity to be referenced as [Image] . This
enforces explicit visual grounding and prevents trivial lexical match-
ing on entity names. For example, (Eiffel Tower,located_in,[Mask])
yields the question: “
 Which city is the landmark in the image
located in?” To avoid shortcut retrieval caused by identical images
appearing in both queries and knowledge graph nodes, we apply
image augmentations to the question images in Appendix C.
Across both modes, question synthesis preserves the seman-
tics of the masked component while producing natural, modality-
appropriate queries, enabling retrieval evaluation under both text-
only and multimodal settings.
3.2 Downstream Task Data
To enable explicit and fine-grained evaluation of retrieval per-
formance, we constructretrieval supervisiondirectly from the
aligned query–triplet pairs. For each synthesized question, the orig-
inating triplet is treated as the primary positive target. Building
on the retrieval dataset, we constructdownstream generation
datain a supervision-preserving manner. Because each question is
derived from a masked triplet, the masked component corresponds
exactly to the missing information requested by the query. We
therefore treat the masked entity or relation as the ground-truth
answer. This process yields question–answer (QA) pairs that are
guaranteed to be supported by the aligned multimodal knowledge
graph. The resulting QA data includes both text-only and visually
grounded instances, reflecting the mixed-modality nature ofMKG-
RAG. By construction, successful generation requires both correct
retrieval and faithful grounding, enabling meaningful end-to-end
evaluation. An example of our benchmark construction is provided
in Appendix D.
3.3 Dataset Statistics
After curating both retrieval and generation datasets from two
domain-specific multimodal knowledge graphs, we split the
datasets for each domain and each stage into train/validation/test
sets with an8 :1:1ratio. The split is performed by questions, so the
same question is never shared across train/validation/test sets, whilekeeping the corpus fixed, following standard IR evaluation. This
split supports not only comprehensive evaluation but also retriever
training when needed. To reflect realistic MKG-RAG usage, each
split contains a mixture of text-only and multimodal query–triplet
pairs, rather than separating modalities into disjoint partitions. As
a result, we obtainMKG-RAG-Bench, a cross-domain multimodal
benchmark that supports training and evaluation under diverse
settings in practical MKG-RAG scenarios. Table 2 reports the key
statistics ofMKG-RAG-Bench, whereMKG-RAG-Bench-G repre-
sents the data obtained from the general multimodal knowledge
graph MarKG, andMKG-RAG-Bench-M is the dataset constructed
based on the medical multimodal knowledge graph MedMKG.
A notable difference between the two subsets lies in the query–
triplet cardinality. ForMKG-RAG-Bench-G, the number of queries
notably exceeds the number of triplets. In contrast,MKG-RAG-
Bench-M contains substantially fewer queries than triplets. This
gap is primarily driven by the structural properties of the underly-
ing medical KG, which exhibits more frequent1 :𝑛patterns. Popular
clinical entities (e.g.,lungorpneumonia) often connect to many
other concepts through the same relation, so a fixed incomplete
form such as(ℎ,𝑟,[𝑀𝑎𝑠𝑘]) may correspond to multiple valid tails.
Consequently, many triplets share the same masked form and yield
identical queries, resulting in fewer unique queries despite a larger
triplet set.
By contrast, the general-domain KG is relatively sparse: relations
between everyday entities are less densely connected and tend
to produce more unique incomplete forms, leading to a higher
diversity of queries per triplet.
4 Benchmark Evaluation
4.1 Evaluation Settings
We evaluate the proposedMKG-RAG-Benchon both retrieval and
generation tasks. As our primary focus is on thetraining-free
setting, all evaluations are conducted exclusively on the test splits
reported in Table 2, without using any training or validation data. As
described in Section 3.1 (Step 3), the benchmark includes two types
of queries—text-only and multimodal—and the underlying multi-
modal knowledge graphs contain both text-only and multimodal
triplets3. To comprehensively assess retrieval and generation per-
formance under different modality combinations, we evaluate the
following five settings, defined by the query modality and the scope
of candidate triplets: (S1)all queries with all triplets, (S2)text-
only queries with text-only triplets, (S3)text-only queries
3Note that MarKG and MedMKG do not include triplets of the form (image, relation,
image).

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma
Table 3: Retrieval performance (%) onMKG-RAG-Bench-G dataset under different retrieval settings.
Query
TypeTriplet
TypeMethodNDCG @𝐾↑ Precision @𝐾↑ Recall @𝐾↑
5 10 20 50 100 5 10 20 50 100 5 10 20 50 100
All AllRandom 0.01 0.02 0.03 0.04 0.07 0.00 0.00 0.00 0.00 0.00 0.01 0.03 0.07 0.14 0.29
Text-only 35.40 37.59 38.89 39.87 40.37 9.94 5.81 3.22 1.41 0.74 43.52 49.93 54.79 59.45 62.40
Captioning 35.56 37.54 38.59 39.32 39.65 9.90 5.72 3.12 1.34 0.69 43.16 48.95 52.83 56.26 58.24
Fusion 51.70 54.65 56.23 57.25 57.67 14.38 8.39 4.63 2.01 1.04 62.29 70.84 76.62 81.32 83.67
Reranking 42.37 45.02 46.74 48.40 49.42 11.81 6.91 3.89 1.77 0.97 51.84 59.59 66.00 73.86 79.67
Text-only Text-onlyRandom 0.01 0.01 0.02 0.05 0.08 0.00 0.00 0.00 0.00 0.00 0.01 0.03 0.06 0.19 0.41
Others* 50.71 53.39 54.74 55.60 55.97 14.11 8.10 4.39 1.87 0.96 61.41 69.21 74.19 78.23 80.37
Text-only AllRandom 0.01 0.01 0.02 0.04 0.07 0.00 0.00 0.00 0.00 0.00 0.02 0.02 0.05 0.15 0.30
Text-only 48.98 51.71 53.15 54.06 54.45 13.68 7.90 4.31 1.84 0.95 59.78 67.72 73.04 77.31 79.54
Captioning 50.45 53.15 54.51 55.40 55.76 14.03 8.07 4.38 1.86 0.96 61.12 68.99 74.01 78.16 80.29
Fusion 50.70 53.39 54.73 55.60 55.96 14.11 8.10 4.39 1.87 0.96 61.41 69.23 74.18 78.25 80.40
Reranking 48.93 51.67 53.11 54.02 54.40 13.67 7.90 4.31 1.84 0.95 59.72 67.70 73.01 77.32 79.53
Multimodal MultimodalRandom 0.06 0.09 0.12 0.21 0.31 0.02 0.02 0.02 0.02 0.02 0.09 0.17 0.30 0.74 1.32
Text-only 8.70 9.82 10.93 12.29 13.20 2.44 1.60 1.04 0.57 0.35 11.12 14.50 18.77 25.43 30.83
Captioning 5.22 5.90 6.47 7.25 7.81 1.44 0.95 0.60 0.33 0.21 6.68 8.72 10.94 14.74 18.11
Fusion 55.07 58.55 60.70 62.07 62.57 15.34 9.19 5.27 2.38 1.25 65.52 75.57 83.29 89.47 92.15
Reranking 27.79 30.25 32.66 36.04 38.56 7.66 4.75 2.97 1.64 1.04 34.33 41.63 50.67 66.69 81.04
Multimodal AllRandom 0.01 0.01 0.01 0.03 0.06 0.00 0.00 0.00 0.00 0.00 0.01 0.03 0.04 0.13 0.28
Text-only 4.19 5.12 6.10 7.22 7.99 1.34 0.99 0.71 0.41 0.26 6.15 8.95 12.73 18.27 22.86
Captioning 1.16 1.49 1.80 2.17 2.44 0.35 0.29 0.21 0.12 0.08 1.69 2.72 3.90 5.73 7.34
Fusion 53.88 57.38 59.54 60.92 61.48 14.96 9.00 5.19 2.34 1.24 64.09 74.23 82.00 88.14 91.21
Reranking 27.44 29.81 32.21 35.54 38.09 7.53 4.64 2.91 1.61 1.03 33.77 40.84 49.83 65.65 80.20
Table 4: Retrieval performance (%) onMKG-RAG-Bench-M dataset under different retrieval settings.
Query
TypeTriplet
TypeMethodNDCG @𝐾↑ Precision @𝐾↑ Recall @𝐾↑
5 10 20 50 100 5 10 20 50 100 5 10 20 50 100
All AllRandom 0.00 0.01 0.02 0.04 0.09 0.00 0.01 0.01 0.01 0.01 0.01 0.03 0.06 0.13 0.34
Text-only 24.72 28.19 30.43 32.25 33.06 14.48 9.88 6.12 2.96 1.63 27.93 36.31 42.86 49.24 52.58
Captioning 24.60 28.02 30.24 32.06 32.88 14.38 9.80 6.07 2.94 1.62 27.74 36.01 42.50 48.93 52.33
Fusion 26.53 30.17 32.68 34.85 35.93 15.62 10.63 6.62 3.25 1.82 29.67 38.55 45.84 53.46 57.88
Reranking 27.85 32.48 36.60 41.36 42.25 16.82 12.01 8.12 4.50 2.41 31.36 42.42 54.01 70.03 73.66
Text-only Text-onlyRandom 0.03 0.03 0.07 0.13 0.20 0.02 0.02 0.02 0.03 0.02 0.02 0.04 0.14 0.38 0.70
Others* 41.93 47.81 51.61 54.69 56.07 24.56 16.75 10.38 5.02 2.76 47.37 61.59 72.69 83.51 89.17
Text-only AllRandom 0.01 0.01 0.01 0.03 0.06 0.01 0.00 0.00 0.01 0.01 0.00 0.00 0.01 0.08 0.24
Text-only 41.93 47.80 51.61 54.69 56.06 24.56 16.75 10.38 5.02 2.76 47.37 61.59 72.69 83.51 89.17
Captioning 41.71 47.51 51.26 54.33 55.70 24.39 16.61 10.28 4.98 2.74 47.05 61.07 72.01 82.84 88.53
Fusion 41.93 47.80 51.61 54.69 56.06 24.56 16.75 10.38 5.02 2.76 47.37 61.59 72.69 83.51 89.17
Reranking 41.81 47.64 51.42 54.50 55.87 24.48 16.68 10.32 5.00 2.75 47.18 61.33 72.35 83.16 88.84
Multimodal MultimodalRandom 0.01 0.01 0.02 0.06 0.14 0.01 0.00 0.01 0.02 0.02 0.01 0.01 0.02 0.16 0.49
Text-only 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Captioning 0.17 0.19 0.25 0.31 0.39 0.10 0.07 0.06 0.03 0.03 0.22 0.28 0.43 0.65 1.02
Fusion 4.41 4.83 5.48 6.36 7.01 2.79 1.83 1.23 0.71 0.45 4.25 5.47 7.27 10.28 12.92
Reranking 7.81 10.71 15.32 22.48 22.67 5.81 5.30 4.94 3.79 1.93 8.62 15.25 27.67 51.16 51.85
Multimodal AllRandom 0.03 0.05 0.05 0.08 0.10 0.02 0.02 0.01 0.01 0.01 0.03 0.06 0.07 0.17 0.27
Text-only 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Captioning 0.01 0.01 0.04 0.06 0.09 0.01 0.00 0.01 0.01 0.01 0.01 0.01 0.10 0.21 0.32
Fusion 4.41 4.83 5.48 6.36 7.01 2.79 1.83 1.23 0.71 0.45 4.25 5.47 7.27 10.28 12.92
Reranking 7.81 10.71 15.32 22.48 22.67 5.81 5.30 4.94 3.79 1.93 8.62 15.25 27.67 51.16 51.85
with all triplets, (S4)multimodal queries with multimodal
triplets, and (S5)multimodal queries with all triplets.
Setting S1 provides an overall evaluation of different retrieval
techniques under the most general and realistic scenario. Compar-
ing S2 and S3 allows us to assess whether incorporating multimodal
triplets benefits retrieval and generation for text-only queries. Sim-
ilarly, comparing S4 and S5 evaluates the contribution of text-only
triplets when handling multimodal queries. Together, these settingsenable systematic analysis of how multimodal information influ-
ences retrieval and generation performance across different query
types and knowledge graph configurations.
4.2 Retrieval Evaluation
Evaluation Configuration.The objective of retrieval evaluation
is to assess the effectiveness of different retrieval techniques in-
troduced in Section 2.2, includingtext-only retrievers[ 11,12,19],

MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph–Augmented Generation KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
fusion-based multimodal retrievers[ 3,27],captioning-based retriev-
ers[9,44], andreranking-based retrievers[ 4,42]. In addition, we
include a basicrandom retrieveras a simple lower-bound baseline
for the retrieval task. For fair comparison, all retrievers are im-
plemented using a shared CLIP encoder [ 28]4to obtain unified
representations. Additionally, thecaptioning-based retrieversare
implemented with a BLIP model [ 17]5. Both queries and candidate
triplets are embedded into the same representation space, and can-
didates are ranked based oncosine similarity. We report standard
retrieval metrics, including NDCG@ 𝐾, Precision@ 𝐾, and Recall@ 𝐾
in the main experiments. Implementation details are provided in
Appendix E.
Retrieval Results and Key Findings.We evaluate retrieval per-
formance on bothMKG-RAG-Bench-G andMKG-RAG-Bench-M
under different query and triplet compositions. Detailed results are
reported in Table 3 and Table 4, with efficiency analysis provided in
Appendix F. Note thatOthers*means the baselines mentioned in
the evaluation configuration. Rather than enumerating numerical
comparisons, we summarize the results through a set of key find-
ings that highlight the challenges and opportunities of multimodal
KG-RAG retrieval.
Find ing1:Amodal itygappersists inretrieval.For text-only
queries retrieving from text-only triplets, all retrievers exhibit
nearly identical performance, as they effectively reduce to the same
text-based retrieval pipeline. When the candidate space is expanded
to include both text-only and multimodal triplets, performance
changes only marginally, and top-ranked results remain dominated
by text-only triplets. This observation suggests that multimodal
triplets rarely serve as effective matches for purely textual queries
and primarily increase the size of the candidate pool without con-
tributing useful evidence. More broadly, this behavior reveals a
persistentmodality gapin the retrieval space: cross-modal match-
ing between textual queries and multimodal knowledge does not
emerge naturally from hybrid indexing alone. This finding indi-
cates that simply incorporating multimodal knowledge into a KG
is insufficient for enabling cross-modal retrieval, and that explicit
mechanisms are required to bridge modalities when such interac-
tions are desired.
Finding2:Multimodal embedding iscriticalforvisuallygrounded
retrieval.Once queries require visual grounding, unimodal re-
trieval strategies become unreliable. Text-only retrievers degrade
sharply because they lack the ability to incorporate image evidence.
Captioning-based retrievers also underperform, as the captioning
process introduces an information bottleneck: omissions or inaccu-
racies in generated captions directly propagate into retrieval errors.
In contrast, embedding-based multimodal retrievers remain effec-
tive in this regime. Fusion-based methods benefit from jointly en-
coding visual and textual signals into a shared representation space,
enabling more robust cross-modal alignment. Reranking-based ap-
proaches further improve performance by refining candidate sets
using stronger multimodal matching. These results indicate that the
central capability for effective MKG-RAG retrieval is not improved
text ranking, but reliable multimodal representation learning that
preserves and aligns visual evidence with KG semantics.
4https://huggingface.co/sentence-transformers/clip-ViT-B-32
5https://huggingface.co/Salesforce/blip-image-captioning-baseFinding3:Retrieval difficulty andmethod preference aredomain -
dependent.Visually grounded retrieval is consistently more chal-
lenging onMKG-RAG-Bench-M than onMKG-RAG-Bench-G. A
plausible explanation is that medical images tend to be visually ho-
mogeneous and require sensitivity to subtle patterns, while the as-
sociated terminology is dense and fine-grained. As a result, generic
similarity signals are weaker and more easily confounded. This
domain discrepancy also explains differences in method preference.
OnMKG-RAG-Bench-G, fusion-based retrieval is often sufficient,
as coarse semantic alignment yields strong performance. OnMKG-
RAG-Bench-M, reranking-based approaches provide greater bene-
fits, as refinement helps disambiguate subtle visual cues and special-
ized language. These observations highlight that retrieval strategies
effective in general domains may not transfer directly to specialized
domains without adaptation.
Implications forMKG -RAG retriever design.Together, these
findings suggest several directions for future retriever develop-
ment. First, bridging the modality gap requires explicit cross-modal
alignment objectives or supervision, rather than relying on mixed-
modality indexing to induce cross-modal matching implicitly. Sec-
ond, multimodal retrieval should prioritize representation learning
that preserves fine-grained visual information and aligns it directly
with KG semantics, as lossy caption-based conversions consistently
limit performance. Third, domain-aware refinement becomes in-
creasingly important in settings with subtle visual distinctions and
specialized terminology, motivating stronger reranking or verifier-
style matching modules and the use of domain-adapted visual and
textual encoders. Overall, effective MKG-RAG retrieval is likely to
require the integration of explicit cross-modal alignment, robust
multimodal embeddings, and domain-sensitive refinement.
4.3 Generation Evaluation
Evaluation Configuration.For generation evaluation, we include
ageneration-onlybaseline (i.e., RAG-free) that queries the LLM
without retrieval augmentation, and evaluate RAG variants built
on the same retrievers, modality settings, and implementations
described in Section 4.2. For each query, we provide the top- 𝐾
retrieved triplets as additional context to the generator, using 𝐾=5
to balance evaluation fidelity with computational cost and token
budget. We use GPT-5 [ 34] as the generation model. Generation is
evaluated as an open-ended (visual) question answering task using
Exact Match (EM), token-level F1, Contains@1, and BLEU-1.
Generation Results and Key Findings.Table 5 reports genera-
tion performance onMKG-RAG-Bench-G andMKG-RAG-Bench-
M under different query and knowledge modality combinations.
Rather than enumerating individual scores, we summarize the re-
sults through several key findings that reveal how retrieval quality
and modality interact to shape end-to-end MKG-RAG performance.
Find ing4:Retrieval provides genuine butuneven utilityfor
generation.Compared to the generation-only baseline, retrieval-
augmented generation yields consistent improvements across most
metrics and settings, confirming that retrieved multimodal knowl-
edge can be effectively consumed by the generator. Notably, even
Random retrieval occasionally improves performance, indicating
that when relevant triplets appear in the context, the generator can

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma
Table 5: Generation results (%) onMKG-RAG-Bench-G andMKG-RAG-Bench-M with𝐾=5.
Query
TypeTriplet
TypeMethodMKG-RAG-Bench-G MKG-RAG-Bench-M
EM↑F1↑Contains@1↑BLEU↑ EM↑F1↑Contains@1↑BLEU↑
All AllRAG-free 5.04 15.15 16.50 43.03 1.34 7.84 4.51 44.72
Random 8.99 17.82 18.76 47.27 1.67 7.91 6.68 50.61
Text-only 13.62 25.50 26.72 50.63 13.69 25.80 22.70 54.35
Captioning 13.20 25.12 26.66 50.55 14.69 25.76 22.20 55.19
Fusion 16.76 29.96 31.77 53.03 13.86 25.53 23.87 55.86
Reranking 14.47 26.99 28.63 51.42 16.86 28.78 26.88 56.78
Text-only Text-onlyRAG-free 4.64 15.00 16.05 41.20 0.85 9.35 4.25 45.15
Random 8.65 17.69 18.18 46.10 3.40 12.39 11.33 54.79
Others* 15.55 28.13 29.63 51.29 21.46 38.72 38.53 61.77
Text-only AllRandom 9.36 17.76 18.45 46.51 1.47 10.38 8.50 53.37
Text-only 15.91 28.30 29.17 51.59 21.99 38.22 36.95 61.63
Captioning 15.89 28.26 29.41 51.47 23.46 38.40 36.36 62.47
Fusion 16.38 28.69 29.78 51.80 21.41 37.80 36.95 61.69
Reranking 15.94 28.06 28.98 51.32 23.46 38.18 36.66 61.96
Multimodal MultimodalRAG-free 5.52 14.79 16.19 46.57 2.03 5.78 2.85 45.50
Random 7.85 18.36 20.90 49.86 2.44 5.53 4.07 49.05
Text-only 8.28 19.61 21.66 49.97 1.63 4.28 2.44 45.32
Captioning 8.34 21.19 24.85 50.48 2.44 4.94 3.25 43.58
Fusion 18.46 34.35 38.06 56.73 3.25 10.26 6.50 49.14
Reranking 11.42 24.49 27.40 51.88 7.32 15.19 12.60 47.80
Multimodal AllRandom 8.14 17.96 19.47 49.03 1.94 4.65 4.26 46.96
Text-only 8.36 19.09 21.09 48.41 2.71 9.38 3.88 44.73
Captioning 7.01 17.89 20.33 48.44 3.10 9.05 3.49 45.57
Fusion 17.64 32.87 36.35 55.86 3.88 9.32 6.59 48.15
Reranking 11.11 24.55 27.83 51.64 8.14 16.37 13.95 49.93
exploit them. This serves as a sanity check for the alignment be-
tween constructed knowledge, queries, and answers, and validates
the overall quality of the benchmark.
Find ing5:Generation gains aresubstantially weaker inmulti-
modal regimes.Retrieval augmentation delivers the largest gains
when both queries and supporting knowledge are textual. In con-
trast, for multimodal queries, improvements are often marginal
unless a strong multimodal retriever is used, and even then abso-
lute gains remain limited—particularly onMKG-RAG-Bench-M.
These results indicate that end-to-end generation is highly sensitive
to whether retrieval can reliably supply visually grounded evidence.
Simply incorporating multimodal knowledge into the pipeline does
not automatically translate into proportional generation improve-
ments, revealing a key bottleneck in current MKG-RAG systems.
Finding6:Generationperformance closely tracks retrieval quality.
Generation trends closely mirror retrieval behavior. Text-only and
captioning-based retrieval pipelines provide limited gains for mul-
timodal queries because they either ignore visual information or
compress it through lossy captions, constraining evidence quality
before generation begins. In contrast, embedding-based multimodal
retrievers yield the most reliable improvements in multimodal set-
tings. Fusion-based methods are generally more effective onMKG-
RAG-Bench-G, while reranking-based approaches offer additional
benefits onMKG-RAG-Bench-M, where subtle visual distinctions
and specialized terminology demand stronger refinement. This
alignment highlights retrieval quality as a primary driver of end-
to-end MKG-RAG generation performance.Implications: toward graph -aware multimodal generation.
The comparatively modest gains in multimodal settings suggest that
current multimodal retrievers, largely designed for unstructured
corpus retrieval, do not fully exploit the structure and semantics of
MKG. Improving MKG-RAG generation, therefore, likely requires
retrieval methods that are explicitly graph-aware, such as relation-
sensitive matching, neighborhood- or path-based evidence aggre-
gation, and graph-constrained candidate selection to reduce dis-
tractors. Beyond retrieval, there is also headroom in how evidence
is presented to the generator, including improved triplet selection,
ordering, and representation formats that better convey structured
graph information. Overall, these findings suggest that progress in
MKG-RAG will depend less on scaling generic multimodal retrieval
and more on developing graph-aware, domain-sensitive retrieval
and evidence integration strategies.
4.4 Ablation on LLM-based Utility Filtering
A key design choice inMKG-RAG-Benchis to filter out low-utility
triplets that tend to produce ambiguous or weakly grounded queries.
To validate this choice, we conduct an ablation study comparing
questions derived fromfilteredtriplets and those derived from the
remainingtriplets retained in the benchmark. We randomly sam-
ple 150 multimodal questions from each group and evaluate MKG-
RAG performance under Setting (S4) using our strongest Fusion-
based retriever, with both RAG-free and retrieval-augmented gener-
ation. Results are reported in Table 6. Across all metrics, questions
derived from filtered triplets consistently yield lower performance

MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph–Augmented Generation KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
Table 6: Generation results (%) for validating the effectiveness of LLM-based utility filtering strategy.
Split Method EM↑F1↑Contains@1↑BLEU↑
FilteredRAG-free 3.33 11.09 8.00 41.35
Fusion 8.00 14.04 12.00 46.21
RemainingRAG-free 5.52 14.79 16.19 46.57
Fusion 18.46 34.35 38.06 56.73
Table 7: Cross-LLM consistency analysis.
Model Filter Alignment (%) BERTScore
Qwen3.5 99.7 0.8818
Gemini 2.5 Flash 97.0 0.8818
than those derived from remaining triplets, even without retrieval
augmentation, indicating that such questions are intrinsically diffi-
cult and poorly specified. Although retrieval improves performance
in both cases, gains on filtered-triplet questions are substantially
smaller, suggesting that retrieval cannot compensate for unclear
or weakly grounded information needs. These results demonstrate
that low-utility triplets confound reliable MKG-RAG evaluation,
and therefore validate the necessity of LLM-based utility filtering to
ensure clear, grounded queries and faithful assessment of retrieval
and generation performance.
4.5 Discussion on LLM Dependency
To evaluate whether the pipeline depends heavily on a specific
LLM, we conduct a cross-LLM consistency analysis. Besides the
GPT-based construction pipeline, we apply Qwen3.5 [ 36] and Gem-
ini 2.5 Flash [ 5] to a 300-sample subset, including 150 text-only
samples and 150 multimodal samples. We compare their outputs
with GPT-based results using filtering alignment and question-level
semantic similarity measured by BERTScore. As shown in Table 7,
both alternative LLMs achieve high filtering alignment with GPT-
based outputs, with Qwen3.5 reaching 99.7% and Gemini 2.5 Flash
reaching 97.0%. The generated questions also show strong semantic
consistency, with a BERTScore of 0.8818 for both models. These
results indicate that the filtering and synthesis process is robust to
the choice of LLM backbone.
We further conduct a human evaluation to assess the reliability
of the filtering and question synthesis results. Human annotators
reach 88% agreement with the filtering decisions, and the generated
questions receive an average quality and diversity score of 3.97 out
of 5. Together, the cross-LLM and human evaluation results show
that the benchmark construction process maintains stable behavior
across different LLMs and introduces limited model-specific bias in
practice.
5 Conclusion
This work demonstrates that progress in MKG-RAG is increasingly
constrained by retrieval, a component that has received limited
attention largely due to the lack of suitable benchmarks. To addressthis gap, we introduceMKG-RAG-Bench, a cross-domain bench-
mark explicitly designed to evaluate retrieval in MKG-RAG, com-
prising two multimodal knowledge graphs and carefully aligned
question-answering datasets.MKG-RAG-Benchis constructed
through an LLM-based curation pipeline that filters low-utility
knowledge and systematically covers diverse modality configura-
tions, enabling controlled and realistic retrieval evaluation. Exten-
sive experiments across representative retriever families reveal that
effectively leveraging multimodal signals during retrieval remains
both challenging and critical for end-to-end performance. By isolat-
ing and diagnosing these challenges,MKG-RAG-Benchprovides a
principled foundation for developing, evaluating, and advancing
more effective retrievers for MKG-RAG. We hope this benchmark
will catalyze future research on graph-aware, domain-adaptive
multimodal retrieval methods and more effective integration of
structured knowledge into multimodal generation.
Acknowledgements
This research was partially supported by a 2025/2026 Rising Re-
searcher Grant from Penn State’s Institute for Computational &
Data Sciences (RRID:SCR_025154) and the National Science Foun-
dation under Grant No. 2333790 and 2238275.
References
[1]Olivier Bodenreider. 2004. The Unified Medical Language System (UMLS): inte-
grating biomedical terminology.Nucleic Acids Res.32, Database issue (Jan. 2004),
D267–70.
[2]Jiangjie Chen, Rui Xu, Ziquan Fu, Wei Shi, Zhongqiao Li, Xinbo Zhang, Changzhi
Sun, Lei Li, Yanghua Xiao, and Hao Zhou. 2022. E-KAR: A benchmark for ratio-
nalizing natural language analogical reasoning.arXiv preprint arXiv:2203.08480
(2022).
[3]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William Cohen. 2022. Murag:
Multimodal retrieval-augmented generator for open question answering over
images and text. InProceedings of the 2022 Conference on Empirical Methods in
Natural Language Processing. 5558–5570.
[4]Zhanpeng Chen, Chengjin Xu, Yiyan Qi, Xuhui Jiang, and Jian Guo. 2025. VLM
Is a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation
via Knowledge-enhanced Reranking and Noise-injected Training. InFindings of
the Association for Computational Linguistics: EMNLP 2025. 8140–8158.
[5]Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen
Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
et al.2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multi-
modality, Long Context, and Next Generation Agentic Capabilities.arXiv preprint
arXiv:2507.06261(2025).
[6]Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, and
Yong Liu. 2025. Benchmarking Retrieval-Augmented Multimodal Generation for
Document Question Answering.arXiv preprint arXiv:2505.16470(2025).
[7]Aleksandr Drozd, Anna Gladkova, and Satoshi Matsuoka. 2016. Word embed-
dings, analogies, and machine learning: Beyond king-man+ woman= queen. In
Proceedings of coling 2016. 3519–3530.
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[9]Feng Gao, Qing Ping, Govind Thattai, Aishwarya Reganti, Ying Nian Wu, and
Prem Natarajan. 2022. Transform-Retrieve-Generate: Natural Language-Centric

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma
Outside-Knowledge Visual Question Answering. InCVPR 2022. IEEE, 5057–5067.
[10] Yu Gu, Sue Kase, Michelle Vanni, Brian Sadler, Percy Liang, Xifeng Yan, and Yu
Su. 2021. Beyond iid: three levels of generalization for question answering on
knowledge bases. InProceedings of the web conference 2021. 3477–3488.
[11] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented gen-
eration for textual graph understanding and question answering.Advances in
Neural Information Processing Systems37 (2024), 132876–132907.
[12] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao.
2025. GRAG: Graph Retrieval-Augmented Generation. arXiv:2405.16506 [cs.LG]
https://arxiv.org/abs/2405.16506
[13] Alistair E W Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum,
Matthew P Lungren, Chih-Ying Deng, Roger G Mark, and Steven Horng. 2019.
MIMIC-CXR, a de-identified publicly available database of chest radiographs
with free-text reports.Sci. Data6, 1 (Dec. 2019), 317.
[14] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. doi:10.48550/ARXIV.2004.04906
[15] Jason J Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. 2018.
A dataset of clinically generated visual questions and answers about radiology
images.Scientific data5, 1 (2018), 1–10.
[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[17] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping
language-image pre-training for unified vision-language understanding and
generation. InInternational conference on machine learning. PMLR, 12888–12900.
[18] Mufei Li, Siqi Miao, and Pan Li. 2024. Simple is effective: The roles of graphs and
large language models in knowledge-graph-based retrieval-augmented genera-
tion.arXiv preprint arXiv:2410.20724(2024).
[19] Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao
Zhang, and Bing Yin. 2023. Graph Reasoning for Question Answering with
Triplet Retrieval. InACL (Findings).
[20] Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng,
Hui Wang, Hai-Tao Zheng, Fei Huang, Jingren Zhou, et al .[n. d.]. Benchmarking
Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and
Self-adaptive Planning Agent. InThe Thirteenth ICLR.
[21] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. 2021. SLAKE:
A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Ques-
tion Answering. arXiv:2102.09542 [cs.CV] https://arxiv.org/abs/2102.09542
[22] Ye Liu, Hui Li, Alberto Garcia-Duran, Mathias Niepert, Daniel Onoro-Rubio, and
David S Rosenblum. 2019. MMKG: multi-modal knowledge graphs. InEuropean
Semantic Web Conference. Springer, 459–474.
[23] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2023. Reasoning
on graphs: Faithful and interpretable large language model reasoning.arXiv
preprint arXiv:2310.01061(2023).
[24] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan
Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al .2024. Mmlongbench-doc: Benchmarking
long-context document understanding with visualizations.Advances in Neural
Information Processing Systems37 (2024), 95963–96010.
[25] Zi-Ao Ma, Tian Lan, Rong-Cheng Tu, Yong Hu, Yu-Shi Zhu, Tong Zhang, Heyan
Huang, Zhijing Wu, and Xian-Ling Mao. 2024. Multi-modal Retrieval Augmented
Multi-modal Generation: Datasets, Evaluation Metrics and Strong Baselines.
arXiv preprint arXiv:2411.16365(2024).
[26] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. 2019.
Ok-vqa: A visual question answering benchmark requiring external knowledge.
InProceedings of CVPR. 3195–3204.
[27] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. 2025. A Survey of Multimodal
Retrieval-Augmented Generation. doi:10.48550/ARXIV.2504.08748
[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InInternational conference on machine learning. PmLR, 8748–8763.
[29] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. doi:10.48550/ARXIV.1908.10084
[30] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends®in Information Retrieval3, 4
(2009), 333–389. doi:10.1561/1500000019
[31] Naganand Yadati Sanket Shah, Anand Mishra and Partha Pratim Talukdar. 2019.
KVQA: Knowledge-Aware Visual Question Answering. InAAAI.
[32] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross
Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell
Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson,
Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. 2022. LAION-5B:
An open large-scale dataset for training next generation image-text models.
arXiv:2210.08402 [cs.CV] https://arxiv.org/abs/2210.08402[33] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and
Roozbeh Mottaghi. 2022. A-okvqa: A benchmark for visual question answering
using world knowledge. InECCV. Springer, 146–162.
[34] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed
El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al .
2025. Openai gpt-5 system card.arXiv preprint arXiv:2601.03267(2025).
[35] Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for
answering complex questions.arXiv preprint arXiv:1803.06643(2018).
[36] Qwen Team. 2026. Qwen3.5: Accelerating Productivity with Native Multimodal
Agents. https://qwen.ai/blog?id=qwen3.5
[37] Jiaqi Wang, Xiao Yang, Kai Sun, Parth Suresh, Sanat Sharma, Adam Czyzewski,
Derek Andersen, Surya Appini, Arkav Banerjee, Sajal Choudhary, et al .2025.
CRAG-MM: Multi-modal Multi-turn Comprehensive RAG Benchmark.arXiv
preprint arXiv:2510.26160(2025).
[38] Xin Wang, Benyuan Meng, Hong Chen, Yuan Meng, Ke Lv, and Wenwu Zhu.
2023. TIVA-KG: A Multimodal Knowledge Graph with Text, Image, Video and
Audio. InProceedings of MM’23. Association for Computing Machinery, New
York, NY, USA, 2391–2399. doi:10.1145/3581783.3612266
[39] Xiaochen Wang, Zongyu Wu, Yuan Zhong, Xiang Zhang, Suhang Wang, and Fen-
glong Ma. 2026. GPR: Empowering Generation with Graph-Pretrained Retriever.
InProceedings of the ACM Web Conference 2026. 8349–8352.
[40] Xiaochen Wang, Yuan Zhong, Lingwei Zhang, Lisong Dai, Ting Wang, and
Fenglong Ma. 2025. MEDMKG: Benchmarking Medical Knowledge Exploitation
with Multimodal Knowledge Graph. arXiv:2505.17214 [cs.AI] https://arxiv.org/
abs/2505.17214
[41] Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eliyahu
Schwartz, Udi Barzelay, and Leonid Karlinsky. 2025. REAL-MM-RAG: A Real-
World Multi-Modal Retrieval Benchmark. InAnnual Meeting of the Association
for Computational Linguistics.
[42] Yibin Yan and Weidi Xie. 2024. EchoSight: Advancing Visual-Language Models
with Wiki Knowledge. InFindings of the Association for Computational Linguistics:
EMNLP 2024. Association for Computational Linguistics, 1538–1551.
[43] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and
Jina Suh. 2016. The value of semantic parse labeling for knowledge base question
answering. InProceedings of the 54th ACL (Volume 2: Short Papers). 201–206.
[44] Wenjia Zhai. 2024. Self-adaptive Multimodal Retrieval-Augmented Generation.
doi:10.48550/ARXIV.2410.11321
[45] Ningyu Zhang, Lei Li, Xiang Chen, Xiaozhuan Liang, Shumin Deng, and Hua-
jun Chen. 2023. Multimodal Analogical Reasoning over Knowledge Graphs.
arXiv:2210.00312 [cs.CL] https://arxiv.org/abs/2210.00312
[46] Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng
Zhang, Hai Zhao, and Dong Yu. 2025. Docbench: A benchmark for evaluating
llm-based document reading systems. InProceedings of the 4th International
Workshop on Knowledge-Augmented Methods for Natural Language Processing.
359–373.
A Source Knowledge Graphs
We use the following two knowledge graphs:
•MarKG.MarKG [ 45] is a multimodal knowledge graph con-
structed to support multimodal analogical reasoning, where the
entities involved in an analogy (e.g., head/tail/query entities) may
come from different modalities. It is released together with the
Multimodal Analogical Reasoning dataset (MARS) and is built
by expanding and annotating seed entities and relations from
E-KAR [ 2] and BATS [ 7]. In particular, MarKG links external
entities to Wikidata and associates entities with images collected
from LAION-5B [ 32]. Overall, MarKG contains 11,292 entities
and 192 relation types, forming 34,420 triplets, with 76,424 linked
images, providing multimodal grounding for general-domain
analogy.
•MedMKG.For the multimodal knowledge graph, we use
MedMKG [ 40]. MedMKG is a multimodal knowledge graph that
fuses rich imaging data from MIMIC-CXR [ 13] with structured
clinical knowledge from the Unified Medical Language System
(UMLS) [ 1], using both rule-based tools and large language mod-
els for precise concept extraction and relationship modeling. To

MKG-RAG-Bench: Benchmarking Retrieval in Multimodal Knowledge Graph–Augmented Generation KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea
Utility Filtering Prompt
You curate and write natural-language queries for a KG retrieval dataset.
Modality: {multimodal or text-only}.
At query time, the head is given as: {IMAGE or TEXT}.
Task: This group corresponds to {FIXED (head, relation) with MULTIPLE possible tailsORFIXED (head, tail) with MULTIPLE possible relations}.
You must decide if this group is meaningful for generating questions that ask for the missing part.
Rules:
1) Output MUST be JSON:{"keep": bool, "reason": str}.
2) Return ONLY JSON (no extra keys).
3) Decide keep based primarily on relation_text and tail examples.
4) Because head_text may be uninformative for images, DO NOT reject just because head_text is generic.
5) Set keep=false only when the tails are mostly placeholders or administrative/meta concepts, e.g.,
tails like ‘thing/object/entity/space/place/environment/location/category’, or when the relation is too vague.
Input:
GROUP TYPE: {mask_tail or mask_relation}
head_text: {head_text}
relation_text: {relation_text}
tail_text (if mask_relation): {tail_text}
candidates_count: {cand_total}
examples (random 3): {cand_examples}
candidates_sample: {cand_sample}
Question Generation Prompt
You curate and write natural-language queries for a KG retrieval dataset.
Modality: {multimodal or text-only}.
At query time, the head is given as: {IMAGE or TEXT}.
Task: This group corresponds to {FIXED (head, relation) with MULTIPLE possible tailsORFIXED (head, tail) with MULTIPLE possible relations}.
Generate exactly K WH-questions that ask for the missing part.
Rules:
1) Output MUST be JSON:{"questions": [..]}.
2) {prefix_rule}
3) {head_leak_rule}
4) Do NOT copy relation_text (or candidate relation strings) verbatim; paraphrase.
5) If mask_tail: Questions MUST NOT mention any specific tail candidates.
6)≤25 words per question. Avoid yes/no.
7) Refer to the head as ‘{head_ref}’ (or equivalent if multimodal).
8) If mask_relation: You MAY mention tail_text.
Input:
GROUP TYPE: {mask_tail or mask_relation}
head_text: {head_text}
relation_text: {relation_text}
tail_text (if mask_relation): {tail_text}
candidates_count: {cand_total}
examples (random 3): {cand_examples}
candidates_sample: {cand_sample}
K={k}
Figure 3: Two-agent prompting for query construction: filtering low-quality triplets and generating WH-questions for the
remaining triplets.
enhance graph quality and compactness, Neighbor-aware Fil-
tering (NaF), a filtering algorithm specifically designed for mul-
timodal knowledge graphs, is applied to MedMKG. MedMKGcomprises 53,587 edges, 4,868 images, and 262 relations over
3,148 concepts.

KDD ’26, August 09–13, 2026, Jeju Island, Republic of Korea Xiaochen Wang, Bao Hoang, Han Liu, Ting Wang, and Fenglong Ma
Table 8: Retrieval efficiency measured by Queries per Second (QPS) under an online serving setting with cached embeddings
and warm-started retrievers. We conduct the experiments with all settings as detailed in Section 4.1.
Dataset Retriever S1 S2 S3 S4 S5
MKG-RAG-Bench-GText-only 108.96 108.17 109.42 95.41 107.28
Captioning 102.72 108.95 107.05 97.99 108.53
Fusion 20.82 116.42 117.11 20.73 20.97
Reranking 21.91 114.44 121.12 22.46 23.60
MKG-RAG-Bench-MText-only 69.68 69.99 69.53 69.58 69.54
Captioning 79.52 77.00 79.32 72.92 79.41
Fusion 7.89 78.81 77.24 7.72 7.84
Reranking 7.94 77.23 76.83 7.72 7.65
B Prompts for Data Construction
DuringMKG-RAG-Benchconstruction, we prompt GPT-5 twice:
first to validate knowledge utility, and then to translate triplets into
natural-language queries. The corresponding prompts are shown
in Figure 3.
C Image Augmentation for Visual Questions
To prevent exact image duplication from serving as a shortcut in
multimodal retrieval, we apply image augmentation to the images
used in visual questions. For each question image, we randomly
apply one augmentation from the following set:swapping to another
image in the knowledge graph that refers to the same entity, random
cropping, rotation, or color jitter. These augmentations ensure that
the image in the visual question differs from the image associated
with the target triplet, eliminating trivial matches and better sim-
ulating realistic retrieval settings where an exact duplicate of the
query image is unlikely to appear in the knowledge graph.
D Example ofMKG-RAG-BenchConstruction
We use two examples to illustrate how the proposed pipeline gen-
erates benchmark data with high-utility triplets.
•Text-only Query.We take (Antigua and Barbuda, member_of,
[MASK]) as example. The pipeline will generate a query first,
which is “Ofwhich global orintergovernmentalbodiesisAntigua
and Barbuda amem ber?” For theretrievaltask, the ground
truth data include several high-utility triples, such as {(Antigua
and Barbuda, member_of, Commonwealth of Nations), (Antigua
and Barbuda, member_of, United Nations), (Antigua and Barbuda,
member_of, World Trade Organization). For thegenerationtask,
the answers are {Commonwealth of Nations, United Nations, World
Trade Organization}.
•Multimodal Query.If there is a high-utility multimodal triplet
(People’s Republic of China
 , railway_traffic_side, left), the
entity “left” is then masked, leading to the generation of the
multimodal query:
 “Onwhich side dotrains operateinthe
subject’s coun try?” Note that the image entities can be different.
For theretrievaltask, the ground triplet is the original high-
utility triplet, and for thegenerationtask, the answer is the
mask “left”.E Details about Retriever Implementation
The proposedMKG-RAG-Benchis ageneralbenchmark, and any
retrievers can be used. However, in the current version, we imple-
ment the following retrievers in the experiments.
For text-only queries and triplets, all retrievers that share the
same CLIP implementation produce identical embeddings by encod-
ing text solely with CLIP’s text encoder. As a result, they achieve the
same performance on text-only query–text-only triplet retrieval due
to this homogeneous encoding. For multimodal retrieval, the text-
only retriever ignores visual signals and encodes only the textual
entity associated with the image. The captioning-based retriever
first uses BLIP to convert the image into a textual description, and
then encodes the resulting text with CLIP’s text encoder.
The remaining two retrievers adopt more advanced multimodal
strategies. We implement the fusion retriever using a late-fusion
design, where the visual embedding from CLIP’s image encoder
is averaged with the text embedding of the textual part of each
multimodal triplet. The reranking retriever performs a two-stage
procedure: it first retrieves candidates using image-embedding simi-
larity, and then reranks the shortlisted triplets using text-embedding
similarity.
F Retrieval Efficiency Analysis
An additional study analyzes the retrieval efficiency of the baseline
retrievers. We report Queries per Second (QPS) under an online
serving setting where all embeddings are cached and the retrievers
are warm-started. The results are shown in Table 8. Overall, the
findings highlight a clearefficiency–effectiveness trade-off: in
the multimodal setting, fusion-based and reranking-based retrievers
exhibit roughly an order-of-magnitude lower throughput (about
0.1×QPS) than the remaining baselines. By contrast, captioning-
based retrieval benefits substantially from caching and achieves
competitive throughput, approaching that of the text-only retriever.
Taken together, this study quantifies the efficiency cost of stronger
multimodal modeling and provides practical guidance for deploying
MKG-RAG systems under real-time constraints.