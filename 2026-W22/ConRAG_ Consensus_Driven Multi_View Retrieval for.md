# ConRAG: Consensus-Driven Multi-View Retrieval for Multi-Hop Question Answering

**Authors**: Yikai Zhu, Kunfeng Chen, Qihuang Zhong, Juhua Liu, Bo Du

**Published**: 2026-05-27 07:51:46

**PDF URL**: [https://arxiv.org/pdf/2605.28093v1](https://arxiv.org/pdf/2605.28093v1)

## Abstract
Retrieval-augmented generation (RAG) has emerged as a promising paradigm for enhancing large language models (LLMs) on multi-hop question answering (QA), which requires reasoning over evidence from multiple documents. Current multi-hop RAG methods generally focus on either query-side task decomposition or corpus-side knowledge graph construction. Despite their progress, these methods still struggle to achieve satisfactory performance on complex multi-hop QA tasks. To this end, we propose ConRAG, a consensus-driven multi-view RAG framework that effectively boosts LLMs on complex multi-hop QA. The core of ConRAG is to systematically optimize both the query and corpus sides and to leverage multi-view evidence (relation, entity, and text signals) for more accurate retrieval. Extensive experiments on three multi-hop QA benchmarks show that ConRAG consistently outperforms all baselines by a clear margin, e.g., up to +26.9% average performance gains over vanilla RAG, and enables Gemma-4-31B to achieve a new state-of-the-art record on the challenging MuSiQue benchmark.

## Full Text


<!-- PDF content starts -->

ConRAG: Consensus-Driven Multi-View Retrieval for Multi-Hop
Question Answering
Yikai Zhu*Kunfeng Chen*Qihuang Zhong†Juhua Liu†Bo Du†
School of Computer Science, Wuhan University
Wuhan, China
{yikaizhu,chenkunfeng,zhongqihuang,liujuhua,dubo}@whu.edu.cn
Abstract
Retrieval-augmented generation (RAG) has
emerged as a promising paradigm for enhanc-
ing large language models (LLMs) on multi-
hop question answering (QA), which requires
reasoning over evidence from multiple docu-
ments. Current multi-hop RAG methods gen-
erally focus on either query-side task decom-
position or corpus-side knowledge graph con-
struction. Despite their progress, these meth-
ods still struggle to achieve satisfactory perfor-
mance on complex multi-hop QA tasks. To
this end, we propose ConRAG , a consensus-
driven multi-view RAG framework that effec-
tively boosts LLMs on complex multi-hop QA.
The core of ConRAG is to systematically opti-
mize both the query and corpus sides and to
leverage multi-view evidence (relation, entity,
and text signals) for more accurate retrieval.
Extensive experiments on three multi-hop QA
benchmarks show that ConRAG consistently
outperforms all baselines by a clear margin,
e.g., up to +26.9% average performance gains
over vanilla RAG, and enables Gemma-4-31B
to achieve a new state-of-the-art record on the
challenging MuSiQue benchmark.
1 Introduction
Retrieval-augmented generation (RAG) expands
the knowledge boundaries of large language mod-
els (LLMs) by retrieving query-relevant contexts
from external knowledge bases ( Arslan et al. ,
2024 ;Fan et al. ,2024 ). Despite achieving remark-
able performance on various knowledge-intensive
tasks, current RAG methods still fall short in com-
plex multi-hop question answering (QA), which
requires retrieving and reasoning over multiple
pieces of supporting evidence ( Mavi et al. ,2024 ;
Zhuang et al. ,2024 ). Speciﬁcally, traditional RAG
methods typically adopt single-round retrieval, i.e.,
they retrieve relevant content only based on the
*Equal contribution.
†Corresponding author.
b) Graph-based RAGa) Reasoning-based RAG
Retrieval
Subgraph CorpusNER
EntityRelation
Extraction
Knowledge GraphQuestionDecomposition
Dependency-aware
PlanExecution
Sub QuestionRetrieval
PassageGeneration
c) ConRAG
Fact
Evidence-grounded
KGMulti-view
Indexing
Vector
Database
Decomposition
Dependency-aware
PlanExecution
Slot-bound
Sub QuestionCorpus
QuestionUnified
Evidence UnitsMulti-view
Consensus
RetrievalGeneration
Generation
AnswerAnswerAnswer
Passage-level Retrieval
Heterogeneous Objects
ExtractionFigure 1: Overview of different representative RAG
paradigms for multi-hop question answering.
original user query. However, in multi-hop QA
scenarios, a user query often relies on evidence
from multiple documents, i.e., it cannot be fully
answered with a single retrieved context.
Hence, multi-hop RAG has attracted consider-
able attention recently, with research primarily
progressing along two lines of work: Reasoning-
based RAG (Figure 1, a) and Graph-based RAG
(Figure 1, b). Speciﬁcally, these two lines of work
operate from different perspectives: reasoning-
based RAG methods focus on the query side by
decomposing complex questions into simpler sub-
tasks ( Press et al. ,2023 ;Yao et al. ,2023 ;Trivedi
et al. ,2023 ;Zhou et al. ,2023 ;Chen et al. ,2026 ),
while graph-based methods focus on the corpus
side by organizing knowledge structures to facili-
tate cross-passage evidence discovery ( Edge et al. ,
2024 ;Guo et al. ,2025 ;Gutiérrez et al. ,2024 ,
2025 ;Zhuang et al. ,2026 ). Despite their effec-
tiveness, these works still have some shortcom-
ings. On the one hand, although reasoning-based
methods decompose complex queries into simpler
sub-tasks, they still rely on semantic retrieval over
document chunks. This makes them vulnerable
1arXiv:2605.28093v1  [cs.CL]  27 May 2026

to poor retrieval quality, which can degrade over-
all performance and even cause error propagation
across reasoning steps. On the other hand, al-
though graph-based methods can retrieve more rel-
evant evidence objects, their retrieval signals of-
ten come from heterogeneous objects with differ-
ent granularities, such as entities, relations, sum-
maries, and text units. This makes it difﬁcult
to compare different retrieval signals and may re-
quire task-speciﬁc aggregation or context assem-
bly. This raises a question: can we combine the
strengths of both approaches and jointly optimize
the query and corpus sides to synergistically im-
prove multi-hop RAG performance?
To achieve this goal, we need to address two
central challenges: ¶how to better plan retrieval
and reasoning on the query side, and ·how to
make full use of heterogeneous evidence on the
corpus side. Speciﬁcally, for ¶, a straightforward
approach is to directly prompt the LLM to de-
compose the query into several sub-tasks and ex-
ecute them sequentially. However, the dependen-
cies between sub-tasks may not be well captured
in this manner, making it difﬁcult for the model to
fully leverage information obtained from preced-
ing steps when solving subsequent ones. This mo-
tivates us to explore a task decomposition strategy
that better models inter-task dependencies. For ·,
naively mixing heterogeneous evidence and feed-
ing it into the LLM context is clearly inefﬁcient
and suboptimal. Considering that different types
of evidence vary in importance and often provide
complementary information, a more effective ap-
proach would be to map them into a uniﬁed rank-
ing space and design an appropriate mechanism to
jointly weigh their contributions.
To this end, we propose ConRAG, a consensus-
driven multi-view retrieval framework that boosts
LLMs on complex multi-hop QA. As illustrated
in Figure 1(c), ConRAG centers on three com-
plementary designs: Connection ,Constraint , and
Consensus . Speciﬁcally, Connection links entities
and relations to veriﬁable evidence units through
an evidence-grounded knowledge graph on the
corpus side. Constraint introduces a lightweight
slot-bound execution mechanism to strengthen
inter-task dependencies on the query side. Con-
sensus fuses multi-view retrieval signals within
a uniﬁed evidence-unit space. In practice, Con-
RAG ﬁrst constructs an evidence-grounded knowl-
edge graph and projects heterogeneous signals
(i.e., relation, entity-anchor, and text-evidence)into a shared evidence-unit space. For each sub-
task, we then perform multi-view retrieval and
apply a multi-view consensus bonus strategy to
jointly weight and fuse the heterogeneous evi-
dence within this space. Based on the fused scores,
we select the top- kevidence units as context for
the LLM to generate an answer, which in turn ﬁlls
the corresponding slot for the next sub-task, trig-
gering successive rounds of retrieval and answer-
ing until all sub-tasks are completed.
We evaluate ConRAG on three popular multi-
hop QA benchmarks: HotpotQA ( Yang et al. ,
2018 ), 2WikiMultiHopQA ( Ho et al. ,2020 ), and
MuSiQue ( Trivedi et al. ,2022 ). Results with GPT-
4o-mini and Gemma-4-31B demonstrate that Con-
RAG consistently outperforms all baselines, e.g.,
up to +26.9% average gains over vanilla RAG
across all benchmarks under the LLM-based met-
ric. Encouragingly, on the challenging MuSiQue,
ConRAG enables Gemma-4-31B to achieve new
state-of-the-art (SOTA) results, demonstrating its
effectiveness. To summarize, our contributions
are threefold: (1) We propose ConRAG, which
jointly optimizes both the query and corpus sides
to enhance multi-hop QA performance of LLMs in
RAG scenarios. (2) ConRAG introduces a simple-
yet-effective slot-bound execution strategy and a
multi-view retrieval mechanism to enable effec-
tive multi-round RAG. (3) Extensive experiments
demonstrate that ConRAG outperforms existing
methods by a clear margin and achieves SOTA re-
sults on several challenging benchmarks.
2 Related Work
Retrieval-augmented generation (RAG) improves
the performance of LLMs on knowledge-intensive
tasks by retrieving evidence from external knowl-
edge bases ( Lewis et al. ,2020 ;Arslan et al. ,2024 ;
Fan et al. ,2024 ). A typical RAG system fol-
lows a retrieve-then-read paradigm, where rele-
vant passages are retrieved via dense retrieval and
then used as context for generation ( Karpukhin
et al. ,2020 ). While effective for single-hop ques-
tions, this paradigm often fails to capture complete
evidence chains required for complex multi-hop
QA ( Zhuang et al. ,2024 ). Consequently, a grow-
ing body of recent work has focused on explor-
ing multi-hop RAG, which can be categorized into
two groups: Reasoning-based RAG and Graph-
based RAG. The former operates on the query side,
organizing multi-step reasoning through question
2

decomposition, tool interaction, interleaved re-
trieval, or dependency execution ( Press et al. ,
2023 ;Yao et al. ,2023 ;Trivedi et al. ,2023 ;Zhou
et al.,2023 ;Chen et al. ,2026 ). The latter operates
on the corpus side, leveraging knowledge graphs,
memory structures, or hierarchical summaries to
enhance evidence discovery ( Edge et al. ,2024 ;
Guo et al. ,2025 ;Gutiérrez et al. ,2024 ,2025 ;
Sarthi et al. ,2024 ;He et al. ,2024 ;Wang et al. ,
2024 ;Zhuang et al. ,2026 ).
Despite their effectiveness, both lines of
work still have some limitations. Speciﬁcally,
reasoning-based methods, while decomposing
complex queries into simpler sub-tasks, still rely
on semantic retrieval over document chunks, mak-
ing them susceptible to retrieval errors that can
propagate across reasoning steps and degrade over-
all performance. Graph-based methods, while ca-
pable of retrieving structurally related evidence,
depend on heterogeneous retrieval signals derived
from objects of varying granularities, which are
inherently incomparable. This makes it difﬁ-
cult to effectively integrate them, often necessitat-
ing task-speciﬁc aggregation or context assembly
strategies. In contrast, ConRAG jointly optimizes
both the query and corpus sides through a multi-
view consensus-driven RAG framework that uni-
ﬁes heterogeneous knowledge graph retrieval sig-
nals within a shared evidence-unit ranking space.
The work most relevant to ours is Youtu-
GraphRAG ( Dong et al. ,2026 ), which also in-
tegrates graph construction, structural organiza-
tion, and agentic retrieval for complex reason-
ing. However, Youtu-GraphRAG still relies on het-
erogeneous corpus-side evidence from graph ob-
jects of different granularities, which are mainly
combined through retrieval and context assem-
bly rather than explicit cross-view evidence align-
ment, resulting in suboptimal performance. In
contrast, ConRAG introduces slot-bound execu-
tion as a lightweight query-side constraint propa-
gation mechanism and projects heterogeneous sig-
nals into a uniﬁed evidence-unit ranking space for
consensus-enhanced retrieval.
3 Method
In this section, we introduce the overall Con-
RAG framework. Given a set of evidence units
C=fcigN
i=1and an input question q, where
each cidenotes a veriﬁable textual evidence unit,
ConRAG aims to retrieve high-quality evidenceunits throughout multi-step reasoning and gener-
ate the ﬁnal answer based on retrieved evidence
and intermediate results. Speciﬁcally, as illus-
trated in Figure 2, ConRAG consists of three com-
plementary components: Connection links enti-
ties and relations in the corpus-side knowledge
graph back to their source evidence units, ensuring
that graph structures remain grounded in veriﬁable
text. Constraint employs lightweight slot-bound
execution to convert intermediate answers into
constraints for subsequent retrieval queries. Con-
sensus projects relation, entity-anchor, and text-
evidence retrieval signals into a uniﬁed evidence-
unit space, and ranks candidate evidence through
consensus-enhanced scoring. Among them, Con-
sensus serves as the core retrieval mechanism of
ConRAG, while Connection provides the corpus-
side structural foundation and Constraint offers
complementary query-side guidance.
3.1 Connection: Evidence-Grounded KG
ConRAG ﬁrst extracts entities, attributes, and re-
lation triples from evidence units and constructs
an evidence-grounded knowledge graph. We rep-
resent it as a directed entity–relation graph:
G= (V,E), (1)
where Vdenotes the set of entity nodes and Ede-
notes the set of relation edges. Each edge e=
(vs, r, v t)2 Erepresents an explicit relation from
the source entity vsto the target entity vt.
The key idea of Connection is that graph ob-
jects are not directly used as generation context.
Instead, each graph object maintains links to its
source evidence units. Let X(ci)denote the set
of graph objects extracted from ci. For any graph
object o2 V [ E , its source mapping is deﬁned as:
src(o) =fci2 C j o2 X(ci)g. (2)
This mapping allows graph-derived signals to be
traced back to their source textual evidence. As
such, the knowledge graph in ConRAG does not
serve as a heterogeneous context container, but
rather as a retrieval interface that bridges structural
signals and evidence units.
3.2 Consensus: Evidence-Aligned Retrieval
Unlike prior RAG methods that rely on individ-
ual text or graph evidence, ConRAG introduces
a consensus-based retrieval strategy that aims to
fully leverage multi-view evidence. Speciﬁcally,
3

II. Online Reasoning Constraint
I.Offline Construction
Connection
III. Online Retrieval
Consensus
Corpus
Evidence-grounded KGSlot-bound Execution
Q2
 
  A1?A1
Q3
<dep:2>?Q3
  A2?
Q1 Q2 Q3Execution TraceMulti-view Retrieval
Relation View Entity-Anchor View Text-Evidence View
Fact Extraction
Triples Entities AttributesUnified Evidence Space
Consensus Ranking
Best Worse
Top-K Evidence
Sub Generation
Sub Answer Acquired Information+
IV . Final Generation Execution Trace +Acquired Information Final AnswerDependency-aware DecompositionQuestion
                             
Q3
 
<dep:2>?Q1
           ?Q2
 
<dep:1>?
Q1
           ?
Q2
 
<dep:1>??
A2
A3
Multi-view Retrieval
Relation View Entity-Anchor View Text-Evidence ViewFigure 2: Overview of ConRAG. ConRAG links corpus-side graph objects to veriﬁable evidence units and re-
trieves evidence from three complementary views. Candidates are ranked through consensus-enhanced scoring,
while slot-bound execution provides lightweight query-side constraints to guide subsequent retrieval steps.
our goal is not to simply merge multiple retrieval
results, but to align retrieval signals from heteroge-
neous sources into a uniﬁed evidence-unit space
and select aligned evidence based on multi-view
consensus. In practice, building upon the above
evidence-grounded knowledge graph, ConRAG
constructs three complementary retrieval views.
The relation view indexes textual representations
of relation edges to capture explicit relational
clues. The entity-anchor view indexes textual rep-
resentations of entity nodes, where each represen-
tation encodes the entity name, type, attributes,
and local relation summaries. The text-evidence
view directly builds a dense index over evidence
units, preserving the semantic matching capabil-
ity of standard dense retrieval. Formally, given a
query x, the three views return relation hits, entity-
anchor hits, and text-evidence hits as:
Rx= TopKe∈Esim( x, τr(e)), (3)
Ax= TopKv∈Vsim( x, τa(v)), (4)
Tx= TopKc∈Csim( x, c), (5)
where sim(,)denotes cosine similarity between
dense embeddings, and τr(e)andτa(v)denote the
textualized representations of relation edges and
entity nodes, respectively.
In this way, we obtain multi-granularity, multi-
view evidence. To effectively fuse these results,we ﬁrst map the relation and entity-anchor hits
back to evidence units through src():
Cx=Tx[src(Rx)[src(Ax), (6)
where src(O) =∪
o∈Osrc(o)maps a set of graph
objects to the union of their source evidence units.
All subsequent scoring and ranking are performed
overCx. This design makes relation, entity-anchor,
and text-evidence signals comparable within a uni-
ﬁed evidence-unit space.
For the relation view, let Rc
x=fe2 R xj
c2src(e)gdenote the retrieved relation edges
linked to evidence unit c. When Rc
x=;, we set
sr(c) = 0 . Otherwise, let h(e) = sim( x, τr(e))
denote the matching score of relation edge e, and
lete⋆= arg max e∈Rcxh(e)denote the strongest
matched relation edge. The relation score uses
the strongest relation match as the primary signal
and treats the remaining matched relations as addi-
tional weak residual support:
sr(c) =h(e⋆) +β∑
e∈Rcx\{e⋆}h(e). (7)
For the entity-anchor view, let Ac
x=fv2
Axjc2src(v)gdenote the matched entity
anchors supporting evidence unit c. The entity-
anchor score projects matched entity scores back
4

to their source evidence units and uses a degree-
aware penalty to suppress hub nodes as follows:
sa(c) =∑
v∈Acxsim( x, τa(v))δ(v), (8)
where
δ(v) ={
1, deg( v)1,
1
1+log deg( v),deg( v)>1.(9)
In practice, this score also includes a constant scal-
ing factor. Since the scores from the three views
are normalized separately, this constant does not
affect the ranking within the same view. For the
text-evidence view, we directly obtain its score
from dense retrieval and denote it as st(c). The
three scores are then normalized as ¯sr(c),¯sa(c),
and¯st(c), respectively.
Building on these multi-view scores, we design
a compact evidence-unit scoring function to merge
them, which can be formulated as:
s(c) = [¯ sr(c),¯sa(c)p(c),¯st(c)]⊤, (10)
where p(c) =1
1+log( degree (c))is an evidence-level
structural control term to limit structural noise
from highly connected nodes. Furthermore, we in-
troduce a fusion weight vector α= [αr, αa, αt]⊤
to balance the contributions of different views:
Score( c) =α⊤s(c)b(c), (11)
where b(c)is a multi-view consensus bonus:
b(c) = 1 + λmax(0 , m(c) 1)
2, (12)
Here, m(c)denotes the number of retrieval views
that assign a positive score to c. Intuitively, when
an evidence unit receives support from multiple
views, its ranking score is boosted accordingly.
This mechanism reﬂects the core idea of our Con-
RAG: rather than simply merging multiple re-
trieval results, the system identiﬁes evidence that
achieves consensus across views within the uniﬁed
evidence-unit space.
3.3 Constraint: Slot-Bound Execution
A critical challenge in multi-hop QA is that sub-
sequent retrieval queries must be informed by in-
termediate answers obtained in earlier steps. To
address this, ConRAG employs slot-bound execu-
tion as a lightweight constraint propagation mech-
anism. Given an input question q, the system ﬁrstretrieves an initial context and then generates a
dependency-aware sub-question plan based on the
question and the retrieved evidence:
P=f(i, qi,Di)gM−1
i=0, (13)
where qiis the sub-question with identiﬁer iand
Didenotes the previous steps it depends on. If qi
depends on previous answers, its unresolved argu-
ments are represented by hdep:ji.
Before executing the sub-question with identi-
ﬁeri, ConRAG constructs a slot binding map:
Θi=fhdep:ji 7! ˆajjj2 D ig. (14)
The bound query is then obtained by:
q⋆
i= Bind( qi,Θi). (15)
Here, Θimaps each dependency placeholder to
its corresponding previous answer, and Bind( )
replaces every hdep :jiinqiwith ˆaj. The
bound query q⋆
iis used as a new retrieval query,
triggering the multi-view retrieval and consensus-
enhanced scoring described above. The answer
ˆaiand acquired information from each step serve
as available states for later steps. After all sub-
questions are executed, the system organizes step
answers and accumulated acquired information
into a grounded execution trace, and generates the
ﬁnal answer from this trace.
4 Experiments
4.1 Experimental Setup
Tasks and Datasets. We conduct experiments
on HotpotQA ( Yang et al. ,2018 ), 2WikiMulti-
HopQA ( Ho et al. ,2020 ), and MuSiQue ( Trivedi
et al.,2022 ). These datasets cover bridge, compari-
son, compositional, and multi-hop reasoning ques-
tions. To control cost and ensure comparability,
we follow the setting of Chen et al. (2026 ), sam-
pling 1,000 questions from the validation split of
each dataset and constructing the retrieval corpus
with the corresponding supporting passages and
distractor passages. Following prior work ( Chen
et al. ,2026 ;Zhuang et al. ,2026 ), we adopt two
end-to-end question answering metrics: Str-Acc
andLLM-Acc .Str-Acc denotes the string-based ac-
curacy, i.e., directly determining whether the pre-
diction contains the reference answer. LLM-Acc
denotes the LLM-based accuracy, where a uniﬁed
LLM judge determines whether the generated an-
swer is semantically consistent with the reference
answer ( Zheng et al. ,2023 ). Details of all tasks
are shown in Appendix A.1.
5

Type MethodHotpotQA 2WikiMultiHopQA MuSiQue Average Score
Str-Acc LLM-Acc Str-Acc LLM-Acc Str-Acc LLM-Acc Str-Acc LLM-Acc
GPT-4o-mini is used as the LLM backbone
Direct LLM Zero-shot 38.7 36.3 26.4 24.3 17.6 14.0 27.6 24.9
Text RAGVanilla RAG (Top-1) 38.4 48.6 34.8 37.3 13.2 18.5 28.8 34.8
Vanilla RAG (Top-3) 43.2 53.1 43.0 42.0 20.3 23.6 35.5 39.6
Vanilla RAG (Top-5) 44.1 53.9 46.7 45.6 21.0 23.6 37.3 41.0
Graph-based RAGKGP 46.4 57.1 47.5 43.7 23.3 27.5 39.1 42.8
G-Retriever 28.5 40.9 26.7 35.7 9.1 15.6 21.4 30.7
RAPTOR 48.1 57.8 47.7 45.9 25.2 29.1 40.3 44.3
GraphRAG 39.6 45.2 46.3 43.3 16.5 23.1 34.1 37.2
LightRAG 47.8 57.7 43.1 36.3 18.1 19.4 36.3 37.8
HippoRAG 53.7 55.6 47.7 47.2 24.9 30.1 42.1 44.3
HippoRAG2 56.7 61.9 50.0 47.1 27.0 32.6 44.6 47.2
LinearRAG 50.1 49.1 44.7 45.7 17.4 17.9 37.4 37.6
Reasoning-based RAG LogicRAG 54.8 62.6 64.7 62.5 30.4 37.5 50.0 54.2
Ours ConRAG 63.0 63.5 64.9 70.7 40.6 41.8 56.2 58.7
Gemma-4-31B is used as the LLM backbone
Direct LLM Zero-shot 31.5 30.6 31.8 31.4 9.3 9.3 24.2 23.8
Text RAGVanilla RAG (Top-1) 39.2 38.1 14.3 13.3 14.0 14.7 22.5 22.0
Vanilla RAG (Top-3) 47.6 48.3 37.5 36.9 22.0 22.7 35.7 36.0
Vanilla RAG (Top-5) 51.1 52.0 40.3 40.3 25.2 27.4 38.9 39.9
Graph-based RAGHippoRAG 54.1 55.3 52.9 55.6 25.4 27.6 44.1 46.2
HippoRAG2 54.1 55.9 46.1 47.5 25.7 27.9 42.0 43.8
Youtu-GraphRAG 67.4 69.1 69.5 74.7 43.8 48.3 60.2 64.0
LinearRAG 45.7 46.1 43.0 44.2 18.3 19.7 35.7 36.7
Reasoning-based RAG LogicRAG 60.7 62.4 60.7 65.7 32.1 37.2 51.2 55.1
Ours ConRAG 68.4 70.4 70.2 77.6 48.4 52.3 62.3 66.8
Table 1: Comparison (%) of different RAG methods on several multi-hop question answering benchmarks.
Implementation Details. We assess ConRAG
on two LLM backbones: GPT-4o-mini (Hurst
et al. ,2024 ) and Gemma-4-31B1. In our implemen-
tation, we adopt all-MiniLM-L6-v22as the sen-
tence embedding model and retrieve the top-3 pas-
sages for each query. Greedy decoding is used
during model inference to ensure reproducibility.
For the LLM-Acc metric, we consistently employ
GPT-4o-mini as the LLM judge. The evaluation
prompt is provided in Appendix A.4.
Baselines. We compare ConRAG with a series
of baselines: (1) Direct LLM , which answers
questions without any external retrieval; (2) Text
RAG , which retrieves top- ktext chunks based
on similarity between the query and chunks, i.e.,
vanilla RAG; (3) Graph-based RAG , which ex-
ploits corpus-side structures to enhance evidence
discovery, including KGP ( Wang et al. ,2024 ),
G-Retriever ( He et al. ,2024 ), RAPTOR ( Sarthi
et al. ,2024 ), GraphRAG ( Edge et al. ,2024 ), Ligh-
tRAG ( Guo et al. ,2025 ), HippoRAG ( Gutiérrez
1https://huggingface.co/google/gemma-4-31B-it
2https://huggingface.co/sentence-transformers/all-
MiniLM-L6-v2et al. ,2024 ), HippoRAG2 ( Gutiérrez et al. ,2025 ),
Youtu-GraphRAG ( Dong et al. ,2026 ), and Linear-
RAG ( Zhuang et al. ,2026 ); and (4) Reasoning-
based RAG , which constructs a query-side depen-
dency graph during inference, represented by Log-
icRAG ( Chen et al. ,2026 ). The details of these
baselines are shown in Appendix A.2.
4.2 Main Results
Table 1presents the main results on several bench-
marks, from which we can ﬁnd that:
ConRAG outperforms the other baselines
across all benchmarks. As seen, ConRAG
achieves the best results on HotpotQA, 2WikiMul-
tiHopQA, and MuSiQue under both Str-Acc and
LLM-Acc metrics. Compared with Text RAG,
ConRAG brings clear improvements by retriev-
ing evidence chains across entities and passages,
rather than relying only on direct query–chunk
matching. Compared with Graph-based RAG
and Reasoning-based RAG, ConRAG further im-
proves ﬁnal evidence selection by aligning struc-
tural signals, textual signals, and query-side con-
straints in a uniﬁed evidence-unit ranking space.
6

These results prove the effectiveness of ConRAG’s
consensus-driven multi-view retrieval.
ConRAG brings consistent performance gains
on both GPT and Gemma backbones. The
improvements of ConRAG are not limited to
a speciﬁc LLM backbone. Speciﬁcally, in
GPT-4o-mini , ConRAG achieves the best perfor-
mance across all three datasets, including substan-
tial gains on 2WikiMultiHopQA and MuSiQue.
Moreover, in Gemma-4-31B , ConRAG also con-
sistently outperforms vanilla RAG, Graph-based
RAG, and LogicRAG. This shows that the pro-
posed ConRAG framework remains effective
when using a stronger open-source backbone.
These results indicate that the gains mainly come
from improved evidence retrieval and ranking,
rather than from a particular generator.
ConRAG is particularly effective in challeng-
ing multi-hop scenarios. The results also show
that the gains of ConRAG are more pronounced
on MuSiQue, which requires composing evidence
across more complex and less explicit multi-hop
chains. Speciﬁcally, under GPT-4o-mini , Con-
RAG improves over the strongest baseline by
+10.2Str-Acc points and +4.3LLM-Acc points.
Under Gemma-4-31B , ConRAG still improves over
the strongest baseline by +4.6Str-Acc points and
+4.0LLM-Acc points. These results suggest that
consensus-driven multi-view retrieval is particu-
larly useful when complete evidence chains are
difﬁcult to recover through a single retrieval view
or query-side reasoning alone.
4.3 Ablation Study
We conduct ablation studies to investigate the key
designs of ConRAG. Speciﬁcally, we ﬁrst analyze
the consensus-enhanced fusion and slot-bound ex-
ecution strategies, and then examine the multi-
view retrieval mechanism. In this part, GPT-4o-
mini is used as the default LLM backbone.
Effect of consensus-enhanced fusion and slot-
bound execution strategies. Here, we remove
these strategies from ConRAG individually and
analyze the impact of their removal. As shown
in Figure 3, removing either consensus-enhanced
fusion (“-w/o Consensus”) or slot-bound execu-
tion (“-w/o Slot Binding”) leads to consistent
performance degradation. Speciﬁcally, without
consensus-enhanced fusion, ConRAG cannot pro-
mote evidence units supported by multiple re-
Ours -w/o Consensus -w/o Slot Binding525354555657585960Average Score (%)56.2
54.9
53.558.7
57.6
55.6
Str-Acc
LLM-AccFigure 3: Ablation results of consensus-enhanced fu-
sion and slot-bound execution. Here, we report the
average results of GPT-4o-mini on three datasets.
Retrieval Views Average Score
Relation Entity Text Str-Acc LLM-Acc
3 7 7 50.4 52.5
7 3 7 48.9 51.0
7 7 3 54.8 57.1
3 3 7 49.9 51.7
3 7 3 55.7 58.1
7 3 3 55.3 57.7
3 3 3 56.2 58.7
Table 2: Ablation study on different retrieval view
combinations in ConRAG. The average results of
GPT-4o-mini across three benchmarks are reported.
trieval views, weakening evidence ranking. With-
out slot-bound execution, later retrieval cannot
directly use intermediate answers as dependency
constraints, making the retrieval process less spe-
ciﬁc to the required reasoning path. The larger
drop caused by removing slot-bound execution in
most cases indicates the importance of reliable
constraint propagation for multi-step retrieval.
Effect of retrieval-view combinations. Table 2
reports the ablation results of different retrieval-
view combinations. Among single-view variants,
the text-evidence view achieves the most stable
performance, but it is still inferior to the full
three-view setting. Adding the relation or entity-
anchor view usually brings further gains, and com-
bining all three views achieves the best results
across all datasets and metrics. These results show
that graph-derived structural signals complement
dense textual signals, verifying the necessity of
evidence-aligned multi-view retrieval.
7

0.15
0.25
0.35
0.450.10
0.20
0.3038.040.042.0
/uni03B1r/uni03B1a
41.842.0*
40.140.741.8
40.4
39.539.840.240.840.1
38.8
37.8
37.539.040.542.0
LLM-Acc (%)Figure 4: Analysis of different fusion weights in Con-
RAG. The x- and y-axes denote αrandαa, respec-
tively. Here, we report the results of GPT-4o-mini on
MuSiQue using the LLM-Acc metric.
4.4 Further Analysis
Here, we perform a more in-depth analysis to ex-
amine: 1) sensitivity of fusion weights in multi-
view retrieval and 2) the efﬁciency of ConRAG.
Sensitivity analysis of fusion weights. Con-
RAG uses αr,αa, and αtto control the fusion
weights of the relation, entity-anchor, and text-
evidence views, respectively. We ﬁx all other hy-
perparameters and vary only αrand αa, where
αt= 1 αr αa. As shown in Figure 4, ConRAG
achieves the best performance on MuSiQue when
the text-evidence view retains a relatively large
weight while the two structural views serve as
complementary support. Notably, when the struc-
tural views are assigned excessively large weights,
performance tends to degrade, indicating that re-
lation and entity-anchor signals are better suited
as auxiliary evidence-selection cues. Overall, al-
though three fusion coefﬁcients are introduced to
control the weights of different views, ConRAG
consistently outperforms the vanilla RAG (23.6)
across a wide range of parameter settings, demon-
strating the robustness of our approach.
Efﬁciency analysis. We compare the online in-
ference cost of different RAG methods on 2Wiki-
MultiHopQA, excluding ofﬂine graph construc-
tion. Speciﬁcally, we report the average wall-
clock time and token consumption per query us-
ing GPT-4o-mini in Table 3. During the infer-
ence of ConRAG, relation, entity-anchor, and text-
evidence retrieval are performed in parallel, and
graph-derived hits are used only for evidence-unit
ranking rather than directly appended to the LLM
context. Moreover, slot-bound execution propa-Method Avg. Time (s) Avg. #Tokens
Zero-shot 5.88 216.2
Vanilla RAG 4.28 489.7
G-Retriever 12.50 1000.0
KGP 70.72 11097.8
RAPTOR 5.79 2568.0
GraphRAG 13.05 4699.8
LightRAG 35.14 5730.6
HippoRAG 6.30 2608.8
HippoRAG2 5.89 2809.2
LinearRAG 0.58 609.4
LogicRAG 9.83 1777.9
Youtu-GraphRAG 71.46 12855.1
ConRAG 5.64 2517.2
Table 3: Efﬁciency comparison of different RAG
methods on 2WikiMultiHopQA using GPT-4o-mini.
gates only dependency-relevant intermediate infor-
mation to later steps, instead of appending all re-
trieved graph objects or complete reasoning histo-
ries. That is, by reducing the context length, Con-
RAG enables more efﬁcient model inference. As
a result, ConRAG achieves an average query time
of 5.64 seconds, lower than LogicRAG (9.83 sec-
onds) and most graph-based RAG methods. Its
average token consumption is 2,517.2, which is
lower than several structure-enhanced methods yet
higher than vanilla RAG, LinearRAG, and Logi-
cRAG. These results indicate that our ConRAG
not only brings better performance, but also im-
proves the inference efﬁciency.
5 Conclusion
In this paper, we propose ConRAG, a consensus-
driven multi-view retrieval framework that en-
hances LLMs for complex multi-hop QA. Unlike
existing RAG methods that primarily focus on
query-side reasoning or corpus-side graph struc-
tures, ConRAG systematically optimizes both
the query and corpus sides, aligning multi-view
retrieval signals within a uniﬁed evidence-unit
ranking space to achieve more accurate retrieval.
Speciﬁcally, ConRAG constructs an evidence-
grounded knowledge graph that connects entities
and relations to veriﬁable evidence units, and per-
forms consensus-enhanced fusion to select evi-
dence supported by multiple retrieval views. Fur-
thermore, ConRAG introduces lightweight slot-
bound execution to propagate intermediate con-
straints during multi-step retrieval. Experiments
on three widely used benchmarks demonstrate
that ConRAG consistently and signiﬁcantly out-
performs existing RAG baselines.
8

Limitations
Our work has several potential limitations. First,
due to limited computational budget, we evaluate
ConRAG on three English multi-hop QA bench-
marks with two LLM backbones. Extending the
experiments to more languages, larger corpora,
and more LLMs would further strengthen the gen-
eralizability of our ﬁndings. Second, ConRAG
relies on ofﬂine evidence-grounded knowledge
graph construction and multi-view index build-
ing. Although this design keeps the online re-
trieval cost controllable, graph construction qual-
ity may still be affected by information extrac-
tion errors, noisy relations, and entity normaliza-
tion mistakes. How to build and incrementally
update the evidence-grounded knowledge graph
more efﬁciently in large-scale or frequently chang-
ing corpora remains under-explored. Lastly, slot-
bound execution in ConRAG depends on the qual-
ity of intermediate answers. When early sub-
questions produce incorrect or incomplete an-
swers, the slot-bound queries may introduce mis-
leading constraints for subsequent retrieval steps.
We believe that incorporating intermediate answer
veriﬁcation or selective re-planning represents a
promising direction for future work.
Ethics Statement
We take ethical considerations very seriously and
strictly adhere to the ACL Ethics Policy. This pa-
per proposes a retrieval framework for complex
multi-hop question answering. It aims to improve
evidence retrieval and answer generation over pro-
vided corpora, rather than encouraging models to
learn private or sensitive knowledge. All datasets
used in this paper are publicly available and have
been widely adopted by researchers. The con-
structed evidence-grounded knowledge graph is
used only for retrieval and ranking. Therefore, we
do not anticipate any additional ethical concerns.
References
Muhammad Arslan, Hussam Ghanem, Saba Munawar,
and Christophe Cruz. 2024. A survey on RAG
with LLMs. Procedia Computer Science , 246:3781–
3790.
Shengyuan Chen, Chuang Zhou, Zheng Yuan, Qing-
gang Zhang, Zeyang Cui, Hao Chen, Yilin Xiao,
Jiannong Cao, and Xiao Huang. 2026. You don’t
need pre-built graphs for RAG: Retrieval augmented
generation with adaptive reasoning structures. InProceedings of the AAAI Conference on Artiﬁcial In-
telligence , volume 40, pages 30270–30278.
Junnan Dong, Siyu An, Yifei Yu, Qian-Wen Zhang,
Linhao Luo, Xiao Huang, di yin, Yunsheng Wu, and
Xing Sun. 2026. Youtu-GraphRAG: Vertically uni-
ﬁed agents for graph retrieval-augmented complex
reasoning. In The Fourteenth International Confer-
ence on Learning Representations .
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph RAG approach to query-focused summariza-
tion. arXiv preprint arXiv:2404.16130 .
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on RAG meeting LLMs: To-
wards retrieval-augmented large language models.
InProceedings of the 30th ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining ,
pages 6491–6501.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025. LightRAG: Simple and fast retrieval-
augmented generation. In Findings of the Associa-
tion for Computational Linguistics: EMNLP 2025 ,
pages 10746–10761.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. HippoRAG: Neu-
robiologically inspired long-term memory for large
language models. In Advances in Neural Informa-
tion Processing Systems , volume 37, pages 59532–
59569.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to mem-
ory: Non-parametric continual learning for large lan-
guage models. In Proceedings of the 42nd Inter-
national Conference on Machine Learning , volume
267, pages 21497–21515.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson,
and Bryan Hooi. 2024. G-Retriever: Retrieval-
augmented generation for textual graph understand-
ing and question answering. In Advances in Neural
Information Processing Systems , volume 37, pages
132876–132907.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. In Proceedings of the 28th Inter-
national Conference on Computational Linguistics ,
pages 6609–6625.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. GPT-4o system card. arXiv preprint
arXiv:2410.21276 .
9

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. In Proceedings of
the 2020 Conference on Empirical Methods in Nat-
ural Language Processing (EMNLP) , pages 6769–
6781.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive NLP tasks. In Advances in Neural Infor-
mation Processing Systems , volume 33, pages 9459–
9474.
Vaibhav Mavi, Anubhav Jangra, and Adam Jatowt.
2024. Multi-hop question answering. Foundations
and Trends in Information Retrieval , 17(5):457–586.
Oﬁr Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring
and narrowing the compositionality gap in language
models. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2023 , pages 5687–5711,
Singapore.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. RAPTOR: Recursive abstractive processing
for tree-organized retrieval. In The Twelfth Interna-
tional Conference on Learning Representations .
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. MuSiQue: Mul-
tihop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. In Pro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (volume 1: long
papers) , pages 10014–10037.
Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu,
Ruiyi Zhang, and Tyler Derr. 2024. Knowledge
graph prompting for multi-document question an-
swering. In Proceedings of the AAAI conference
on artiﬁcial intelligence , volume 38, pages 19206–
19214.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik R Narasimhan, and Yuan Cao.
2023. ReAct: Synergizing reasoning and actingin language models. In The Eleventh International
Conference on Learning Representations .
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang,
Joseph Gonzalez, and Ion Stoica. 2023. Judging
llm-as-a-judge with mt-bench and chatbot arena. In
Advances in Neural Information Processing Systems ,
volume 36, pages 46595–46623.
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei,
Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H.
Chi. 2023. Least-to-most prompting enables com-
plex reasoning in large language models. In The
Eleventh International Conference on Learning Rep-
resentations .
Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi
Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang,
and Xiao Huang. 2026. LinearRAG: Linear graph
retrieval augmented generation on large-scale cor-
pora. In The Fourteenth International Conference
on Learning Representations .
Ziyuan Zhuang, Zhiyang Zhang, Sitao Cheng, Fangkai
Yang, Jia Liu, Shujian Huang, Qingwei Lin, Saravan
Rajmohan, Dongmei Zhang, and Qi Zhang. 2024.
EfﬁcientRAG: Efﬁcient retriever for multi-hop ques-
tion answering. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing , pages 3392–3411.
A Appendix
A.1 Details of Tasks and Datasets
We conduct experiments on three widely-used
multi-hop QA benchmarks: HotpotQA ( Yang
et al.,2018 ), 2WikiMultiHopQA ( Ho et al. ,2020 ),
and MuSiQue ( Trivedi et al. ,2022 ). Following
Chen et al. (2026 ), we sample 1,000 questions
from the validation split of each benchmark and
construct the retrieval corpus using the correspond-
ing supporting and distractor passages. The statis-
tics of the datasets are shown in Table 4. Below,
we brieﬂy describe each benchmark.
HotpotQA. HotpotQA ( Yang et al. ,2018 ) is
a Wikipedia-based multi-hop question answering
benchmark. It mainly contains two types of ques-
tions, bridge and comparison questions, which re-
quire systems to retrieve and combine supporting
evidence across multiple passages.
2WikiMultiHopQA. 2WikiMultiHopQA ( Ho
et al. ,2020 ) is constructed from Wikipedia and
Wikidata. It contains four types of multi-hop
questions that require comparison, composition,
inference, and relation chaining over entities.
10

Dataset #Passages #Query Types
HotpotQA 9,221 2
2WikiMultiHopQA 6,119 4
MuSiQue 11,656 3
Table 4: Statistics of all evaluated datasets. “#Pas-
sages” denotes the number of passages in the retrieval
corpus, and “#Query Types” denotes the number of
question categories covered by each benchmark.
These questions are grounded in entity rela-
tions, making the dataset suitable for evaluating
relation-aware retrieval and multi-step reasoning.
MuSiQue. MuSiQue ( Trivedi et al. ,2022 ) is a
challenging multi-hop question answering bench-
mark designed to reduce shortcut reasoning. It
contains three types of compositional questions
with different reasoning depths. Compared with
HotpotQA and 2WikiMultiHopQA, its questions
usually involve less explicit but more composi-
tional evidence chains.
A.2 Details of Compared Methods
We compare ConRAG with representative RAG
baselines, including direct LLM inference,
text-based retrieval, graph-based retrieval, and
reasoning-based retrieval. Below, we provide
brief descriptions of the compared methods.
Direct LLM. Direct LLM answers questions
without using external retrieval. It directly
prompts the LLM with the original question and
generates the ﬁnal answer.
Vanilla RAG. Vanilla RAG retrieves text
chunks according to query–chunk similarity and
feeds the top-ranked chunks into the LLM for
answer generation. It represents the standard
chunk-based retrieval paradigm without corpus-
side graph structures or query-side decomposition.
KGP. KGP ( Wang et al. ,2024 ) constructs
knowledge graphs from multi-document evidence
and uses graph traversal to build structured
prompts for LLMs. It organizes retrieved evidence
with graph-based prompts for multi-document
question answering.
G-Retriever. G-Retriever ( He et al. ,2024 ) com-
bines graph neural networks with LLMs for
retrieval-augmented question answering over tex-
tual graphs. It formulates graph retrieval as a Prize-Collecting Steiner Tree problem to select relevant
graph-structured evidence.
RAPTOR. RAPTOR ( Sarthi et al. ,2024 ) recur-
sively clusters and summarizes text segments to
build a hierarchical tree representation of the cor-
pus. At inference time, it retrieves evidence from
different levels of the hierarchy to combine ﬁne-
grained and coarse-grained information.
GraphRAG. GraphRAG ( Edge et al. ,2024 )
constructs corpus-side graph representations from
entities, relations, and community summaries. It
uses these structures to support graph-guided evi-
dence organization and retrieval.
LightRAG. LightRAG ( Guo et al. ,2025 ) incor-
porates graph structures into text indexing and re-
trieval. It uses dual-level retrieval to combine low-
level and high-level knowledge discovery.
HippoRAG. HippoRAG ( Gutiérrez et al. ,2024 )
is a training-free graph-enhanced retrieval frame-
work inspired by human long-term memory. It
retrieves evidence across documents using query-
related concepts as graph seeds.
HippoRAG2. HippoRAG2 ( Gutiérrez et al. ,
2025 ) extends HippoRAG with improved graph
construction, passage integration, and retrieval
mechanisms. It strengthens associative-memory-
style retrieval for complex multi-hop QA.
Youtu-GraphRAG. Youtu-GraphRAG ( Dong
et al. ,2026 ) is an agentic GraphRAG framework
that explicitly integrates graph construction,
structural organization, and iterative retrieval to
support complex reasoning.
LinearRAG. LinearRAG ( Zhuang et al. ,2026 )
constructs a relation-free graph-based retrieval
structure, termed Tri-Graph, to organize passages.
It enables passage-level graph retrieval without ex-
plicit relation modeling.
LogicRAG. LogicRAG ( Chen et al. ,2026 ) is
a reasoning-based RAG method that constructs
a query-side dependency graph during inference.
It decomposes complex questions into dependent
sub-questions and executes them according to the
predicted dependency structure.
A.3 More Implementation Details
We evaluate ConRAG with two LLM back-
bones, GPT-4o-mini (Hurst et al. ,2024 ) and
11

Backbone Dataset #Question Wrong Rate (%)Slot-correct Slot-wrong
Str-Acc LLM-Acc Str-Acc LLM-Acc
Gemma-4-31BHotpotQA 566 38.7 71.5 88.2 59.8 68.9
2WikiMultiHopQA 656 32.2 83.2 89.5 78.0 81.0
MuSiQue 780 62.1 59.8 73.0 41.1 48.6
Table 5: Comparison between slot-correct and slot-wrong queries. “#Question” denotes the number of ques-
tions where the model introduces dependency slots during task decomposition, and “Wrong Rate” indicates the
proportion of questions with incorrectly ﬁlled slots. Gemma-4-31B is used as the LLM backbone.
Gemma-4-31B3. Unless otherwise speciﬁed, all
methods use the same data split, retrieval corpus,
embedding model, and ﬁnal top- ksetting. We
adopt all-MiniLM-L6-v2 as the sentence embed-
ding model and set the ﬁnal top- kto 3. Greedy
decoding is used during inference for reproducibil-
ity. For the LLM-Acc metric, we consistently use
GPT-4o-mini as the LLM judge, where the de-
tailed prompt is shown in Figure 9. The fusion
weights (αr, αa, αt)are selected for each dataset,
while the residual weight βand the consensus
weight λare shared across datasets. Speciﬁcally,
we set β= 0.02andλ= 0.05in all experiments.
The hyperparameter settings are shown in Table 6.
Benchmark αr αa αt β λ
HotpotQA 0.15 0.20 0.65 0.02 0.05
2WikiMultiHopQA 0.25 0.20 0.55 0.02 0.05
MuSiQue 0.25 0.10 0.65 0.02 0.05
Table 6: Hyperparameter settings of ConRAG.
For GPT-4o-mini backbone, we directly
reuse the reported baseline results from Chen
et al. (2026 ), except for LinearRAG and Youtu-
GraphRAG. Since LinearRAG and Youtu-
GraphRAG adopt a different data split from ours
and other baselines, directly citing their reported
results would lead to an unfair comparison. Thus,
we re-run LinearRAG with its ofﬁcial code on
our data split. During our reproduction of Youtu-
GraphRAG, we found that this method suffers
from low computational efﬁciency (completing
experiments on all three datasets requires over
one week) and consumes a substantial number
of tokens. Due to the limited computational
resources, we do not report its results under the
GPT-4o-mini setting. For Gemma-4-31B backbone,
since no prior work has reported results with this
backbone, we reproduce runnable baselines using
their publicly available code. All reproduced
3https://huggingface.co/google/gemma-4-31B-itmethods are evaluated under identical experi-
mental settings, including the data split, retrieval
corpus, embedding model, and ﬁnal top- k, to
ensure a fair comparison.
A.4 Prompt Details
Here, we present the detailed prompts used by
ConRAG and the prompt used for evaluation.
Speciﬁcally, we present the prompts for graph fact
extraction, question decomposition, step-wise an-
swering, ﬁnal answer generation, and LLM-as-a-
Judge evaluation in Figure 5,6,7,8, and 9, re-
spectively. In particular, the graph fact extrac-
tion prompt extracts explicit triples, entities, and
attributes from each passage for corpus-side ev-
idence organization. The question decomposi-
tion prompt constructs a dependency-aware sub-
question plan from the original question and cur-
rently available evidence. The step-wise answer-
ing prompt answers each planned sub-question us-
ing retrieved evidence and acquired information
accumulated from previous steps. The ﬁnal an-
swer prompt generates the ﬁnal answer from the
original question, execution trace, and accumu-
lated acquired information. The LLM-as-a-Judge
prompt judges whether the generated answer is
equivalent to the gold answer for LLM-Acc eval-
uation. For readability, we only show the core in-
struction parts of each prompt.
A.5 Full Results and Additional Analysis
In this part, we report the full results of experi-
ments in the main paper and provide an additional
analysis of slot-ﬁlling correctness. Speciﬁcally,
Table 7shows the full results of Figure 3,i.e., the
ablation study of consensus-enhanced fusion and
slot-bound execution. Table 8shows the detailed
results of Table 2,i.e., the ablation study of dif-
ferent retrieval-view combinations. Please refer to
the tables for more details.
12

Graph fact extraction prompt
# Identity
Extract explicit graph facts from a passage for retrieval.
# Instructions
- Use `schema `as the preferred label set; create new labels only when needed.
- Use only facts stated in `passage `; do not infer.
- Preserve passage entity names; merge entities only when the passage clearly treats them as identical.
- Split dense passages into separate facts.
- Put relations in directed triples: subject, relation, object.
- Put literal values in attributes, including dates, roles, locations, genres, counts, awards, occupations, nationalities,
and categories.
- Include every entity referenced by triples or attributes in `entities `.
- Keep labels concise and retrieval-friendly.
# Output Format
Return valid JSON only. Do not include markdown, citations, explanations, reasoning, or extra keys.
{
"attributes": {"<entity_name>": ["<attribute_key>: <attribute_value>"]},
"triples": [["<subject>", "<relation>", "<object>"]],
"entities": {"<entity_name>": "<entity_type>"}
}
If no explicit graph facts are present, return:
{"attributes": {}, "triples": [], "entities": {}}
<schema>
{schema}
</schema>
<passage>
{passage}
</passage>
Figure 5: Prompt template for graph fact extraction.
VariantHotpotQA 2WikiMultiHopQA MuSiQue Average Score
Str-Acc LLM-Acc Str-Acc LLM-Acc Str-Acc LLM-Acc Str-Acc LLM-Acc
ConRAG (Ours) 63.0 63.5 64.9 70.7 40.6 41.8 56.2 58.7
-w/o Consensus Fusion 62.0 62.5 62.8 69.8 39.8 40.5 54.9 57.6
-w/o Slot Binding 60.8 60.7 61.5 67.2 38.3 38.9 53.5 55.6
Table 7: Full results of Figure 3,i.e., ablation study of consensus-enhanced fusion and slot-bound execution.
We report Str-Acc and LLM-Acc on each benchmark using GPT-4o-mini as the backbone.
Effect of slot ﬁlling correctness. To analyze
how slot ﬁlling correctness affects the ﬁnal ac-
curacy of ConRAG, we divide samples into slot-
correct andslot-wrong groups according to LLM-
judged slot ﬁlling correctness. Notably, we only
consider questions where dependency slots are in-
troduced during task decomposition. A sample
is classiﬁed as slot-correct only when all depen-
dency slots are ﬁlled with correct intermediate an-
swers. As shown in Table 5, when using Gemma-
4-31B as the backbone, slot-wrong samples consis-
tently yield lower ﬁnal answer accuracy than their
slot-correct counterparts across all three datasets.
This indicates that slot-bound execution provides
effective constraints when intermediate answers
are correctly propagated, whereas incorrect slot
ﬁlling can mislead subsequent retrieval and de-
grade overall performance.A.6 Use of AI Assistants
During the writing of this paper, we used propri-
etary LLMs only as general-purpose writing assis-
tants. They were used to polish the manuscript
and appendix, including ﬁxing grammatical issues,
improving the clarity and ﬂuency of non-native
expressions. We did not use LLMs to generate
research ideas, design methods, conduct experi-
ments, search for related work, or produce scien-
tiﬁc claims. All technical contributions, experi-
mental results, analyses, and implementations are
original and have been veriﬁed by the authors.
A.7 Reproducibility
We will publicly release our code in https://
github.com/yikai-zhu/ConRAG to help repro-
duce the experimental results of this paper.
13

Question decomposition prompt
# Identity
Create the retrieval plan required to answer the question using the evidence.
# Instructions
-`acquired_information `: concise facts explicitly stated in `evidence `; no inference.
- Do not answer directly. `plan `must be non-empty.
- Plan only for missing facts or facts that need verification needed to answer `question `.
- Preserve the target slot, entity constraints, comparisons, and yes/no/all/both conditions.
- Use the fewest retrieval-oriented steps; resolve bridge entities before dependent facts.
- Make `sub_question `self-contained except for `<dep:ID> `placeholders.
- Use zero-based contiguous ids; dependencies must be earlier ids.
- Every `<dep:ID> `must appear in `dependencies `; no unused dependencies.
# Output Format
Return valid JSON only. Do not include explanations, markdown, citations, or extra keys.
{
"acquired_information": "<complete concise grounded facts useful for answering question>",
"plan": [
{
"id": 0,
"sub_question": "<retrieval-oriented sub-question>",
"dependencies": [<integer>, ...]
}
]
}
<question>
{question}
</question>
<evidence>
{evidence}
</evidence>
Figure 6: Prompt template for question decomposition.
Step-wise answering prompt
# Identity
Answer one plan step from evidence.
# Instructions
- Use `original_question `to preserve the target slot and answer type.
- Use `acquired_information `as already grounded context.
- Use `sub_question `as the immediate question to answer.
- Return the shortest grounded value that answers `sub_question `.
- If one entity or value is requested, return one entity or value, not a list.
- Do not substitute related slots such as employer for school, birthplace for nationality, producer for director, or
participant for winner.
- Put complete, concise, grounded facts that directly help answer `original_question `in`acquired_information `.
# Output Format
Return valid JSON only. Do not include reasoning traces, markdown, citations, or extra keys.
{
"answer": "<short answer>",
"acquired_information": "<complete concise grounded facts useful for answering original_question>"
}
<original_question>
{original_question}
</original_question>
<acquired_information>
{acquired_information}
</acquired_information>
<sub_question>
{sub_question}
</sub_question>
<evidence>
{evidence}
</evidence>
Figure 7: Prompt template for step-wise answering.
14

Final answer prompt
# Identity
You are the final answer agent for an evidence-grounded RAG QA system.
# Instructions
- Return only the direct answer, as concisely as possible.
- Do not explain or provide any additional context.
- If the answer is a simple yes/no, return exactly `Yes. `or`No.`
- If the answer is a name, return only the name.
- If the answer is a date, return only the date.
- If the answer is a number, return only the number.
- If the answer requires a brief phrase, make it as concise as possible.
- Give only the essential answer, nothing more.
# Output Format
Return only the final answer string.
<question>
{question}
</question>
<evidence>
{evidence}
</evidence>
Figure 8: Prompt template for ﬁnal answer generation.
Answer evaluation prompt
# Identity
Judge answer equivalence against the gold answer.
# Instructions
- Mark `correct `only if the predicted answer contains the gold answer 's key information, is factually compatible, and adds
no contradiction.
- Accept harmless aliases, paraphrases, and formatting differences.
- Mark `incorrect `for blank answers, wrong entities or values, missing required information, or contradictions.
# Output Format
Return exactly one word: `correct `or`incorrect `.
<predicted_answer>
{pred_answer}
</predicted_answer>
<gold_answer>
{gold_answer}
</gold_answer>
Figure 9: Prompt template for LLM-as-a-Judge evaluation.
Retrieval Views HotpotQA 2WikiMultiHopQA MuSiQue Average Score
Relation Entity Text Str-Acc LLM-Acc Str-Acc LLM-Acc Str-Acc LLM-Acc Str-Acc LLM-Acc
3 7 7 56.3 56.2 61.3 67.3 33.7 33.9 50.4 52.5
7 3 7 57.2 57.8 57.5 63.0 32.0 32.1 48.9 51.0
7 7 3 62.9 63.1 62.2 67.8 39.2 40.3 54.8 57.1
3 3 7 57.1 57.2 60.0 66.1 32.7 31.9 49.9 51.7
3 7 3 62.8 63.2 63.7 69.6 40.5 41.4 55.7 58.1
7 3 3 62.7 63.3 62.8 68.6 40.3 41.3 55.3 57.7
3 3 3 63.0 63.5 64.9 70.7 40.6 41.8 56.2 58.7
Table 8: Full results of Table 2,i.e., ablation study of different retrieval-view combinations. Different from
the main-body table that reports averaged results across three benchmarks, this table reports the detailed results on
HotpotQA, 2WikiMultiHopQA, and MuSiQue.
15