# Disco-RAG: Discourse-Aware Retrieval-Augmented Generation

**Authors**: Dongqi Liu, Hang Ding, Qiming Feng, Jian Li, Xurong Xie, Zhucun Xue, Chengjie Wang, Jiangning Zhang, Yabiao Wang

**Published**: 2026-01-07 20:32:50

**PDF URL**: [https://arxiv.org/pdf/2601.04377v1](https://arxiv.org/pdf/2601.04377v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems.

## Full Text


<!-- PDF content starts -->

Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
Dongqi LiuΩΘ*, Hang Ding∆, Qiming FengΓ, Jian LiΘ, Xurong XieΨ,
Zhucun XueΨ,Chengjie WangΘ,Jiangning ZhangΘ,Yabiao WangΘ
ΩSaarland University,ΘTencent YouTu Lab
∆Shanghai Jiaotong University,ΓFudan University,ΨZhejiang University
Abstract
Retrieval-Augmented Generation (RAG) has
emerged as an important means of enhanc-
ing the performance of large language models
(LLMs) in knowledge-intensive tasks. How-
ever, most existing RAG strategies treat re-
trieved passages in a flat and unstructured way,
which prevents the model from capturing struc-
tural cues and constrains its ability to synthe-
size knowledge from dispersed evidence across
documents. To overcome these limitations, we
propose Disco-RAG , a discourse-aware frame-
work that explicitly injects discourse signals
into the generation process. Our method con-
structs intra-chunk discourse trees to capture
local hierarchies and builds inter-chunk rhetor-
ical graphs to model cross-passage coherence.
These structures are jointly integrated into a
planning blueprint that conditions the gener-
ation. Experiments on question answering
and long-document summarization benchmarks
show the efficacy of our approach. Disco-RAG
achieves state-of-the-art results on the bench-
marks without fine-tuning. These findings un-
derscore the important role of discourse struc-
ture in advancing RAG systems.
1 Introduction
The advent of large language models (LLMs; Tou-
vron et al. 2023; Yang et al. 2025; Achiam et al.
2023) has advanced research progress in natural
language processing (NLP), achieving competitive
performance across a wide range of tasks, including
question answering (Wu et al., 2025a; Lee et al.,
2025a; Zhang et al., 2025b), document summa-
rization (Mondshine et al., 2025; Liu et al., 2025a;
Wang et al., 2025a; Luo et al., 2025), and text gener-
ation (Duong et al., 2025; Bigelow et al., 2025; Que
and Rong, 2025; Zhang et al., 2025a). However,
due to the reliance on static training corpora, LLMs
can be inadequate for knowledge-intensive scenar-
ios, such as handling domain-specific knowledge,
*Bdongqi@lst.uni-saarland.de
Query: Does vitamin D supplementation reduce flu risk in adults?
Standard RAG
EvidenceDisco -RAG
Evidence
Chunk A
“12% lower incidence
in winter among
deficient adults.
Chunk B
“Across 15 RCTs,
overall effect not
significant. ””
Chunk C
“Vitamin D is
......”12% lower
(deficient
adults)winter
benefit if
deficientOverall not
significant
Vitamin D is
...... ............
Background Claim Qualified
Evidence Conclusion
Not broadly effective
overall; possible benefit
for deficient adults.Plan
AnswerＳ
Ｎ Ｎ
ＳＳ
I can't understand the relationships
between these documents.
Vitamin D helps
reduce flu risk. AnswerFigure 1: Comparison between standard RAG and
Disco-RAG . While standard RAG retrieves isolated
chunks without structural links, Disco-RAG organizes
evidence into discourse structures (trees & graphs).
Here, SdenotesSatellite(the supplementary part), and
NdenotesNucleus(the core part).
proprietary data, or information requiring real-time
updates (Chang et al., 2025; Lee et al., 2025b; Yue
et al., 2025; Wang et al., 2024b; Xia et al., 2025).
Retrieval-Augmented Generation (RAG) has been
proposed as a suitable strategy by integrating an
external knowledge component through retrieval-
based mechanisms (Lewis et al., 2020; Asai et al.,
2024; Chan et al., 2024).
In standard RAG pipelines, external documents
are segmented into chunks, which are then encoded
into vectors and stored in a database. At query time,
relevant chunks are retrieved to provide contextual
grounding for the LLM (Lewis et al., 2020). One
important but insufficiently addressed limitation
of existing RAG systems concerns the mismatch
between retrieval granularity and generative un-
derstanding. While retrieval modules return rel-
evant chunks, these chunks are often fragmented
in discourse, resembling scattered pieces of evi-
dence without clear logical connections (Edge et al.,
2024; Su et al., 2025). This manifests at two levels.
First,intra-chunk structural blindness: within each
chunk, RAG systems may fail to capture internalarXiv:2601.04377v1  [cs.CL]  7 Jan 2026

discourse. Second,inter-chunk coherence gaps:
across multiple chunks, RAG systems may strug-
gle to identify rhetorical connections. As depicted
in Figure 1 (left),Chunk Amentionsa 12% lower
incidence, whileChunk Bnotesno significant over-
all effect. Without recognizing that the former is a
conditional finding (e.g.,among deficient adults in
winter), standard RAG tends to overgeneralize and
incorrectly conclude thatvitamin D reduces flu risk.
These deficiencies prevent effective resolution of
conflicting claims, as standard RAG approaches
lack the capacity to organize retrieved evidence
through a higher-level causal flow. This leaves the
final LLM generator to grapple with abag of facts
rather than a coherentline of reasoning.
Recent investigations have revealed that integrat-
ing discourse knowledge into LLMs can improve
downstream performance (Gautam et al., 2024; Liu
and Demberg, 2024) and alleviate hallucinations
(Liu et al., 2025b). These findings highlight the
drawback of relying solely on flat sequential repre-
sentations and underline the benefits of discourse
for context engineering (Ma et al., 2025; Mei et al.,
2025). Building on these insights, the present work
aims to investigate whether explicitly modeling and
providing discourse information to the LLM can
improve generation quality in the context of RAG.
To answer this, we propose Disco-RAG , a frame-
work that constructs local discourse trees for each
retrieved chunk and infers inter-chunk coherence
relations across chunks to form a rhetorical graph.
To synthesize information, rather than merely con-
catenating it, the text generator needs not only to
understand the relations between evidence but also
to strategize how to present them. This requires
a high-level plan to orchestrate the narrative flow.
We thus introduce a discourse-aware planning mod-
ule that enables the model to dynamically generate
a plan to guide the generation. As shown in Fig-
ure 1 (right), the discourse-aware process enables
the model to infer thatvitamin D is not broadly
effective but may benefit deficient adults under spe-
cific conditions, producing more faithful answers
and aligning with the underlying evidence.
In our experiments, we evaluate Disco-RAG on
three benchmarks, Loong (Wang et al., 2024a),
ASQA (Stelmakh et al., 2022), and SciNews (Liu
et al., 2024). Consistent improvements are ob-
served compared with standard RAG systems and
state-of-the-art (SOTA) methods. On the Loong
benchmark, our approach delivers gains of up to
10.0 points in LLM Score. On the ASQA dataset,our method exceeds the best existing systems on
Exact Match and DR Score by notable margins. On
the SciNews benchmark, Disco-RAG establishes
new SOTA performance across most evaluation
metrics.
In summary, our contributions are as follows:
•We present Disco-RAG , an inference-time strat-
egy that explicitly injects discourse knowledge
into the RAG pipeline to alleviate the discrepancy
between chunk-level evidence and discourse-
level reasoning.
•We propose a modeling method that combines
intra-chunk discourse trees, inter-chunk rhetori-
cal graphs, and discourse-driven plans to capture
local hierarchies, cross-passage coherence, and
argumentative flow.
•We conduct experiments on knowledge-intensive
QA and summarization tasks, demonstrating con-
sistent gains over strong RAG baselines. Analysis
studies further confirm the efficacy of discourse-
aware guidance in enhancing generation correct-
ness, coherence, and factuality.
2 Related Work
2.1 Structure-Aware Retrieval-Augmented
Generation
Retrieval-Augmented Generation (RAG) enhances
LLMs in knowledge-intensive tasks by retrieving
external evidence (Lewis et al., 2020). However,
conventional RAG methods typically treat retrieved
chunks as isolated and flat sequences, overlook-
ing their structural interconnections. To mitigate
this, recent research has explored structure-aware
variants of RAG. Graph-based methods (Nigatu
et al., 2025; Hu et al., 2025; Wu et al., 2025b; Zhu
et al., 2025) such as GraphRAG (Edge et al., 2024)
and KG-RAG (Sanmartin, 2024) organize evidence
into knowledge graphs, while subsequent work has
improved retrieval by simulating human memory
mechanisms (Gutierrez et al., 2024; Gutiérrez et al.,
2025) or enriching graph semantics (Liang et al.,
2025). Other approaches construct structured sub-
graphs for coherence (Mavromatis and Karypis,
2025; Li et al., 2025a), or employ alternative for-
mats like hierarchical graphs (Zhang et al., 2024;
Wang et al., 2025b; Huang et al., 2025), semantic
chunking (Wang et al., 2025c; Qu et al., 2025; Zhao
et al., 2025), trees (Fatehkia et al., 2024; Sarthi
et al., 2024), and tables (Lin et al., 2025). More
adaptive strategies dynamically select structures
based on context (Li et al., 2025b). Despite these

advances, most efforts emphasize surface-level as-
sociations (e.g.,linking entities) while largely over-
looking the rhetorical structure that governs causal
flow, evidence presentation, and conclusion formu-
lation. This hinders logical depth and discourse
coherence, which our work seeks to address.
2.2 Rhetorical Structure Theory for Text
Generation
Rhetorical Structure Theory (RST; (Mann and
Thompson, 1987, 1988)) is a discourse framework
that models hierarchical dependencies and rhetor-
ical relations among Elementary Discourse Units
(EDUs). It distinguishes betweennucleusandsatel-
liteunits, connected by relations such asElabora-
tion,Causality, andContrast, forming tree struc-
tures that reflect communicative intent. Founda-
tional work (Marcu, 1997, 1999; Mann and Thomp-
son, 1987; Bhatia et al., 2015; Hayashi et al., 2016)
has established strong correlations between rhetori-
cal structure and human text planning (Adewoyin
et al., 2022). Later studies have leveraged RST
by converting trees into dependency graphs or im-
posing structural constraints to improve coherence
and consistency in neural generation models (Chis-
tova, 2023; Zeldes et al., 2025; Chistova, 2024;
Maekawa et al., 2024). More recent efforts have in-
tegrated RST into LLMs to improve cross-sentence
reasoning and enhance both structural integrity and
interpretability of generated outputs (Liu et al.,
2023; Liu and Demberg, 2024). Compared with
shallow discourse markers or sentence-level con-
nectives, the present work extends RST model-
ing to the RAG setting by explicitly encoding the
deeper structure of retrieved passages and high-
lighting the importance of hierarchical discourse.
3 Proposed Method
Method Overview.We formalize the standard
RAG as a conditional generation problem. Given
a query qand a set of Top- kretrieved chunks
C(q;D) ={c 1, c2, . . . , c k}from a corpus D, the
output is y= arg max y′P(y′|q,C(q;D)) , where
P(·) denotes the conditional distribution of the
generator. To overcome the limitations of the
retrieval-and-concatenation paradigm, we propose
Disco-RAG to augment standard RAG with rhetori-
cal parsing and discourse-aware planning.
As illustrated in Figure 2, our pipeline consists
of three main stages. (1) We delve into each
chunk cito uncover its internal logical hierarchyby constructing an intra-chunk RST tree ti, (2) We
zoom out to map the relational landscape across all
chunks Cvia an inter-chunk rhetorical graph G, and
(3) We apply a discourse-driven planning module
that devises a blueprint Bbased on T=tk
i=1and
Gto guide the final generation process.
We hypothesize that under identical retriever and
decoding conditions, explicitly injecting discourse
knowledge improves the correctness, coherence,
and factual consistency of generated text. Here,
rhetorical modeling serves as aknowledge-level
prior, while planning offersreasoning-level guid-
ance, jointly inducing stronger structural biases
than standard RAG. The following paragraphs pro-
vide a detailed account of each component.
Intra-Chunk RST Tree.For each retrieved
chunk ci, we construct an RST tree tiusing an
LLM-based RST parser Ato model local co-
herence.1Given ci, parser Ajointly performs
elementary discourse unit (EDU) segmentation
and RST parsing, producing a sequence of EDUs
{ei1, . . . , e im}, nucleus and satellite role assign-
ments, and rhetorical relations among EDUs. For-
mally, ciA− →t i= (V i, Ei), where Vi=
{ei1, . . . , e im}is the set of EDU nodes, Ris the
set of rhetorical relations (e.g.,Elaboration,Con-
trast, andCause), and Ei⊆Vi×Vi× R is the set
of directed connections labeled with relation types.
The symbol ×denotes theCartesian product. The
top-middle panel of Figure 2 shows how EDUs are
organized into a hierarchical tree.2
The RST tree parsing is formalized as P(ti|
ci;θA) =Qm
j=1P(eij|ci;θA)·Q
(u,v)P(ru,v|
eiu, eiv;θA), where P(eij|ci)signifies the prob-
ability of EDU boundary prediction and u, v∈V i
are discourse units, P(ru,v|eiu, eiv)corresponds
to the probability of the rhetorical relation between
two EDUs, and θAindicates the parameters of the
parser.
Inter-Chunk Rhetorical Graph.For all re-
trieved chunks C, we construct a directed graph
G= (C,F) . The edge set F ⊆ C × C ×(R ∪
UNRELATED) encodes rhetorical relations or lack
thereof. We adopt a listwise inference strategy,
where all retrieved chunks Care provided to parser
Ain a single pass, and Ajointly predicts a set of di-
rected rhetorical relations {ri,j}or an UNRELATED
1Prompt is detailed in Appendix Figure 10.
2Intra-chunk RST trees are constructed offline.

Similarity-Based Retrieval Intra-Chunk RST Tree Inter-Chunk Rhetorical Graph
Query
Corpus
ChunksRetrieverElementary
Discourse Unit
SegmentationRST Tree 
ConstructionCause
Condition Contrast
Background
Background
Elaboration
ElaborationCause
Generation
Query1
2
3
4
5Background
Argument
Evidence
......
ConclusionPlanning
QueryCause
Condition Contrast
Background
Background
Elaboration
ElaborationCauseFigure 2: The Disco-RAG pipeline: Starting from passage retrieval (providing context), then intra-chunk RST tree
parsing (capturing local discourse), inter-chunk rhetorical graph construction (modeling global discourse), rhetorical
planning (blueprint generation), and answer generation (producing the final output).
label for all chunk pairs.3
The rhetorical graph construction is modeled as
P(G | C;θ A). This joint distribution can be factor-
ized over ordered chunk pairs as P(G | C;θ A) =Qk
i=1Qk
j=1,j̸=i P(ri,j| C;θ A). As shown in the
top-right panel of Figure 2, the resulting graph G
serves as a global discourse scaffold, allowing the
generator to reason over cross-chunk connections.
Discourse-Driven Planning.To move beyond
the flat concatenation of retrieved evidence, we
introduce a planning module that produces a rhetor-
ically informed blueprint to guide the text gen-
eration. This is modeled through a mapping
from the input query q, retrieved chunks Cto-
gether with their RST trees T, and the inter-chunk
rhetorical graph Ginto a discourse-aware plan
(q,C,T,G)A− → B.
As depicted in the center-bottom panel of Fig-
ure 2, the plan Bis dynamically conditioned on the
discourse structures and the query4. The plan out-
lines reasoning steps that involve selecting salient
content, organizing argumentative flow, and priori-
tizing supporting evidence.
Discourse-Guided RAG.The final stage of gen-
eration is conditioned on four inputs: (1) the orig-
inal text chunks C; (2) the intra-chunk RST trees
T; (3) the inter-chunk rhetorical graph G; and
(4) the discourse-aware plan B. The objective is
y= arg max y′P 
y′|q,C,T,G,B) , where y′de-
notes a candidate output and yrefers to the final
output that maximizes the conditional probability.5
3Appendix Figure 11 provides prompt and format details
used in inter-chunk relation prediction.
4Appendix Figure 12 provides prompt used in discourse-
aware planning.
5Appendix Figure 18 contains the generation prompt.4 Experimental Settings
Evaluation Datasets.We evaluate our method
on three benchmarks, namely Loong (Wang et al.,
2024a), ASQA (Stelmakh et al., 2022), and
SciNews (Liu et al., 2024). The Loong dataset fo-
cuses on knowledge-intensive reasoning with Spot-
light Locating (Spot.), Comparison (Comp.), Clus-
tering (Clus.), and Chain of Reasoning (Chain.).
These tasks are evaluated under varying document
lengths, where longer inputs increase evidence frag-
mentation and reasoning difficulty. ASQA involves
long-form question answering and requires models
to generate responses that are coherent and factu-
ally grounded. SciNews targets long-document lay
summarization, where the objective is to rewrite
scientific articles into accurate and accessible sum-
maries for general audiences (Cachola et al., 2025).
Dataset statistics are reported in Appendix Table 6.
Automatic Metrics.To ensure consistency and
fair comparison, we follow the official evaluation
protocols provided by each dataset’s repository
(Wang et al., 2024a; Stelmakh et al., 2022; Liu
et al., 2024). For the Loong dataset (Wang et al.,
2024a; Li et al., 2025b), we report results using
Exact Match (EM) and LLM-based scores. For
ASQA (Stelmakh et al., 2022; Chang et al., 2025),
the evaluation includes EM, ROUGE-L (RL) (Lin,
2004), and DR Score (Stelmakh et al., 2022). On
SciNews, we evaluate with RL, BERTScore (Zhang
et al., 2020), SARI (Xu et al., 2016), and SummaC
(Laban et al., 2022). These metrics assess the in-
formativeness, fluency, and factual consistency of
generated answers. Detailed descriptions of these
metrics are provided in Appendix B.

Implementation Details.Unless specified oth-
erwise, we use Llama-3.1-8B ,Llama-3.3-70B ,
orQwen2.5-72B across all modules to instanti-
ate and compare performance at different model
scales and families (Grattafiori et al., 2024).6
For embedding and retrieval modules, we utilize
Qwen3-Embedding-8B (Zhang et al., 2025c) with
a chunk size of 256 tokens without sliding win-
dow, and Top-10 retrieval based on cosine seman-
tic similarity. We run each setting once; we use
beam search with a beam width of 3, and fix all
retrieval settings across compared methods. For
Loong and ASQA, retrieval is conducted over the
entire corpus, reflecting an open-domain setting.
For SciNews, retrieval is restricted to the source
document associated with each summary, reflecting
a closed-domain setup.
Selected Baselines.We compare Disco-RAG
against three baseline settings: (1) zero-shot
LLMs ( Llama-3.1-8B ,Llama-3.3-70B , and
Qwen2.5-72B ) with full input context. (2) stan-
dard RAG approach (Lewis et al., 2020)7, where
relevant chunks are prepended to the query prior
to inference.8and (3) previously published results
from state-of-the-art RAG (if applicable) systems
on the same benchmark.
5 Results
Main Results.The experimental results are sum-
marized in Table 1, Table 2, and Table 3, which
correspond to the Loong, ASQA, and SciNews
benchmarks, respectively. Across all benchmarks
and evaluation metrics, Disco-RAG consistently de-
livers stable and substantial improvements over the
standard RAG baseline.
On the Loong benchmark, Disco-RAG demon-
strates clear gains across varying document-length
settings. With Llama-3.3-70B as the backbone,
our method achieves an LLM Score of 71.00 in
Set 1, outperforming standard RAG by 8.22 points.
The performance gap becomes more significant in
Set 4, where Disco-RAG scores 54.62 compared
to 35.61 for standard RAG. Averaged across all
four sets, our approach also surpasses the best prior
6Llama-3.1-8B ,Llama-3.3-70B , and Qwen2.5-72B
are the abbreviated names for Llama-3.1-8B-Instruct ,
Llama-3.3-70B-Instruct, andQwen2.5-72B-Instruct.
7Prompts for full context generation and standard RAG
method are provided in Appendix Figure 13 and Figure 14.
8All experiments are training-free and use only task in-
structions without in-context examples. All hyperparameters
follow the same settings as Disco-RAG.reported training-based method StructRAG (62.07
vs. 60.38).
On ASQA, our method again yields consis-
tent advantages. With Llama-3.1-8B , EM, RL,
and DR scores increase from 37.3/36.9/23.4 to
40.4/42.2/32.6, and with Llama-3.3-70B , EM
rises to 42.0 and DR to 32.8. Notably, our
method outperforms more sophisticated prompting
systems, such as MAIN-RAG (42.0 RL) and Tree
of Clarifications (39.7 RL), achieving an RL
score of 42.3. On the SciNews summarization task,
our approach exhibits strong generalization ability.
Using Llama-3.3-70B ,Disco-RAG obtains 21.11
RL score, 65.67 BERTScore, 44.37 SARI, and
69.48 SummaC, surpassing both standard RAG and
the previous best system (Liu et al., 2024, 2025b).
Ablation Studies.We perform ablation studies
on the Loong benchmark, as summarized in Ta-
ble 4, to assess the contribution of each component
inDisco-RAG . We find that the removal of any
single module leads to performance degradation.
The full model achieves an overall LLM Score
of 62.07, which drops to 56.22, 57.10, and 59.75
when the RST tree, rhetorical graph, and planner
are removed, respectively. Similarly, the Exact
Match metric decreases from 0.24 in the full set-
ting to values ranging from 0.20 to 0.22 across the
ablated variants. We also include two generic plan-
ning baselines built on standard RAG to isolate
the added value of discourse structure modeling
beyond planning alone.9
Among the three components, the RST tree and
rhetorical graph prove to be the most critical. In the
long-document setting (Set 4), eliminating the RST
tree leads to a decrease in LLM Score from 54.62
to 47.63. Similarly, removing the rhetorical graph
reduces the score to 48.16, whereas excluding the
planner causes a smaller drop to 50.34. These find-
ings imply that while all three modules contribute
complementarily, structural modeling within and
across chunks plays a central role in aggregating
information and maintaining discourse coherence
in long-context generation.
6 Analysis
Impact of Retrieval Granularity and Noise Ro-
bustness.To assess the robustness of Disco-RAG
under different retrieval conditions, we execute a
series of controlled experiments that manipulate
9Prompts for these two generic planning baselines are
provided in Appendix Figure 15 and Figure 16.

Condition ModelSpot. Comp. Clus. Chain. Overall
LLM Score ↑EM↑LLM Score ↑EM↑LLM Score ↑EM↑LLM Score ↑EM↑LLM Score ↑EM↑
Set 1 (10K–50K Tokens)
Full ContextLlama-3.1-8B 55.43 0.35 56.06 0.36 47.41 0.08 65.66 0.37 56.16 0.30
Qwen2.5-72B 55.11 0.34 57.21 0.33 47.09 0.10 66.51 0.36 56.59 0.31
Llama-3.3-70B 58.82 0.44 61.33 0.35 48.15 0.1170.310.37 59.54 0.32
Standard RAGLlama-3.1-8B 62.61 0.32 60.61 0.26 53.61 0.08 58.76 0.32 60.08 0.25
Qwen2.5-72B 63.20 0.32 61.29 0.35 54.14 0.11 64.67 0.34 61.58 0.33
Llama-3.3-70B 68.44 0.45 65.32 0.39 55.30 0.12 66.48 0.36 62.78 0.34
SOTA ResultsRQ-RAG⋆(Chan et al., 2024) 72.310.5448.16 0.05 47.44 0.07 58.96 0.25 53.51 0.17
GraphRAG⋆(Edge et al., 2024) 31.67 0.00 27.60 0.00 40.71 0.14 54.290.4340.82 0.18
StructRAG (Li et al., 2025b) 74.53 0.47 75.58 0.4765.13 0.2367.84 0.34 69.43 0.35
Disco-RAG (Llama-3.1-8B) 73.35 0.40 73.57 0.37 64.44 0.12 68.00 0.34 69.18 0.32
Disco-RAG (Qwen2.5-72B) 74.46 0.42 74.39 0.41 64.66 0.15 67.73 0.35 69.59 0.36
Disco-RAG (Llama-3.3-70B)76.600.4575.650.45 65.360.17 68.30 0.38 71.00 0.38
Set 2 (50K–100K Tokens)
Full ContextLlama-3.1-8B 51.30 0.27 42.37 0.21 38.32 0.06 44.49 0.11 43.78 0.14
Qwen2.5-72B 52.37 0.30 44.47 0.25 39.24 0.07 47.69 0.11 46.61 0.13
Llama-3.3-70B 55.27 0.34 47.93 0.26 40.05 0.08 50.08 0.10 48.24 0.17
Standard RAGLlama-3.1-8B 57.02 0.25 45.42 0.19 44.21 0.05 50.42 0.15 49.12 0.16
Qwen2.5-72B 60.13 0.26 50.64 0.20 45.17 0.05 53.28 0.16 50.33 0.17
Llama-3.3-70B 60.38 0.27 53.37 0.22 45.76 0.07 56.73 0.18 53.77 0.18
SOTA ResultsRQ-RAG⋆(Chan et al., 2024) 57.35 0.35 50.83 0.16 42.85 0.03 47.60 0.10 47.09 0.10
GraphRAG⋆(Edge et al., 2024) 24.80 0.00 14.29 0.00 37.86 0.00 46.25 0.12 33.06 0.03
StructRAG (Li et al., 2025b) 68.00 0.4163.710.3661.40 0.17 54.70 0.19 60.95 0.24
Disco-RAG (Llama-3.1-8B) 66.03 0.36 63.56 0.24 59.53 0.14 53.06 0.16 59.03 0.23
Disco-RAG (Qwen2.5-72B) 67.17 0.36 64.06 0.30 60.63 0.15 57.22 0.20 61.32 0.25
Disco-RAG (Llama-3.3-70B)69.920.39 64.34 0.36 61.67 0.18 58.23 0.22 63.61 0.28
Set 3 (100K–200K Tokens)
Full ContextLlama-3.1-8B 42.25 0.22 37.43 0.12 32.27 0.00 35.62 0.00 36.51 0.08
Qwen2.5-72B 45.47 0.29 40.13 0.13 35.29 0.01 48.47 0.01 42.01 0.10
Llama-3.3-70B 47.31 0.31 41.11 0.14 35.64 0.01 49.78 0.01 42.27 0.11
Standard RAGLlama-3.1-8B 49.22 0.21 40.24 0.03 36.04 0.00 49.05 0.00 43.42 0.06
Qwen2.5-72B 50.14 0.25 41.83 0.04 40.07 0.03 49.09 0.02 44.38 0.11
Llama-3.3-70B 50.33 0.33 43.70 0.06 40.13 0.04 50.10 0.05 45.77 0.13
SOTA ResultsRQ-RAG⋆(Chan et al., 2024) 50.50 0.13 44.62 0.00 36.98 0.00 36.79 0.07 40.93 0.05
GraphRAG⋆(Edge et al., 2024) 15.83 0.00 27.40 0.00 42.50 0.00 43.330.1733.28 0.04
StructRAG (Li et al., 2025b)68.62 0.4457.74 0.3558.27 0.1049.73 0.13 57.92 0.21
Disco-RAG (Llama-3.1-8B) 60.76 0.26 55.80 0.11 53.07 0.05 50.31 0.08 56.64 0.15
Disco-RAG (Qwen2.5-72B) 65.58 0.33 56.89 0.19 57.23 0.06 51.20 0.13 57.14 0.18
Disco-RAG (Llama-3.3-70B) 66.37 0.38 57.840.28 58.850.07 52.170.15 58.86 0.22
Set 4 (200K–250K Tokens)
Full ContextLlama-3.1-8B 31.79 0.12 25.37 0.06 27.87 0.00 26.76 0.00 27.82 0.04
Qwen2.5-72B 34.22 0.18 28.23 0.06 28.11 0.00 28.48 0.00 30.15 0.04
Llama-3.3-70B 36.76 0.21 32.22 0.07 30.69 0.00 30.17 0.00 32.21 0.05
Standard RAGLlama-3.1-8B 40.01 0.11 31.90 0.00 32.33 0.00 29.92 0.00 33.52 0.02
Qwen2.5-72B 40.14 0.16 32.31 0.01 34.00 0.00 30.02 0.01 33.64 0.04
Llama-3.3-70B 40.27 0.25 34.49 0.02 36.41 0.01 31.33 0.02 35.61 0.07
SOTA ResultsRQ-RAG⋆(Chan et al., 2024) 29.17 0.08 40.36 0.00 26.92 0.00 34.69 0.00 31.91 0.01
GraphRAG⋆(Edge et al., 2024) 17.50 0.00 26.67 0.00 20.91 0.00 33.670.3323.47 0.05
StructRAG (Li et al., 2025b) 56.87 0.19 55.62 0.2556.59 0.00 35.71 0.05 51.42 0.10
Disco-RAG (Llama-3.1-8B) 56.68 0.19 53.92 0.1257.530.02 36.00 0.03 50.87 0.08
Disco-RAG (Qwen2.5-72B) 57.27 0.22 54.97 0.15 57.40 0.02 36.17 0.06 54.47 0.10
Disco-RAG (Llama-3.3-70B)57.74 0.24 55.800.17 57.36 0.0336.06 0.06 54.62 0.11
Table 1: Loong benchmark results across four document-length settings. Our method ( Disco-RAG ) is compared
against zero-shot LLMs with full context, standard RAG, and prior SOTA.⋆means that the results are directly
taken from Li et al. (2025b). We usebold redto indicate the best results and blue underlined text to indicate the
second-best results.
the chunk size of passages, the number of Top-
kretrieved chunks, and the proportion of noisy
passages. All experiments are conducted on the
Loong dataset using Llama-3.3-70B as the gen-
erator model. We maintain identical prompts and
decoding configurations across all systems. The
evaluation includes two baseline methods, namely
the full context setting and the standard RAG frame-
work. Performance is reported using the average
LLM Score over four subsets, and the results are
visualized in Figure 3.
Panel (a) of Figure 3 shows that standard RAG
performs best at a chunk size of 256 tokens (49.33)
but degrades with larger chunks due to the loss of
structural coherence. In contrast, Disco-RAG main-
tains stable performance across all chunk sizes,
with scores ranging from 62.07 to 58.94, showingstrong robustness to granularity shifts. Panel (b)
of Figure 3 shows that while standard RAG peaks
at Top-10 and declines with larger kdue to accu-
mulating noise, Disco-RAG also performs best at
Top-10 but remains robust up to Top-50, showing
enhanced capacity to integrate and filter redundant
information. Panel (c) of Figure 3 evaluates noise
robustness by replacing a fraction of the Top-10
retrieved passages with unrelated content. We ran-
domly replace a proportion of retrieved chunks
(e.g.,20%, 40%) with irrelevant ones sampled at
random from a pool of non-retrieved chunks. The
standard RAG exhibits a steep performance drop
from 49.33 to 45.23 as noise increases, whereas
Disco-RAG retains a score of 56.17, highlighting
the structural resilience of our method to retrieval
errors.

128 256 512 1024
Chunk Size4045505560LLM Score
(a) Performance vs. Chunk Size
1510 20 50
Top-k4045505560LLM Score
(b) Performance vs. Top-k
20% 40% 60% 80%
Noisy Chunk Ratio4045505560LLM Score
(c) Performance vs. Noise Chunks (Top-10)
Full Context Standard RAG Disco-RAGFigure 3: Performance comparison under varying chunk size (a), Top-kvalue (b), and retrieval noise level (c).
Model EM ↑RL↑DR Score ↑
Baselines with full context
Llama-3.1-8B 20.1 30.6 16.3
Qwen2.5-72B 21.3 31.8 17.1
Llama-3.3-70B 22.7 32.9 16.8
Baselines with standard RAG
Llama-3.1-8B 37.3 36.9 23.4
Qwen2.5-72B 37.7 37.2 23.7
Llama-3.3-70B 38.2 37.2 24.1
SOTA Results
FLARE (Jiang et al., 2023) 41.3 34.3 31.1
Tree of Clarifications (Kim et al., 2023) — 39.7 36.6
Open-RAG (Islam et al., 2024) 36.3 38.1 —
ConTReGen (Roy et al., 2024) 41.2 — 30.3
DualRAG (Cheng et al., 2025) — 31.7 —
RAS (Jiang et al., 2025) — 39.1 —
MAIN-RAG-Mistral-7B (Chang et al., 2025) 35.7 36.2 —
MAIN-RAG-Llama3-8B (Chang et al., 2025) 39.2 42.0 —
Ours
Disco-RAG (Llama-3.1-8B) 40.4 42.2 32.6
Disco-RAG (Qwen2.5-72B) 41.8 41.333.2
Disco-RAG (Llama-3.3-70B)42.0 42.332.8
Table 2: Performance on the ASQA benchmark.
Disco-RAG consistently outperforms standard RAG
across all metrics. It also surpasses existing SOTA meth-
ods on most dimensions.
Impact of Structure Quality and Perturbation
Analysis.To determine whether the performance
gains of Disco-RAG arise from the quality of struc-
tural modeling rather than the mere presence of
structural cues, we conduct a set of controlled per-
turbation experiments targeting three core com-
ponents of our framework. These include intra-
chunk RST trees, inter-chunk rhetorical graphs, and
discourse-aware plans. For each module, we intro-
duce partial degradations by randomly selecting
relation labels, edge directions, or planning steps,
and either replacing or removing them. This design
ensures that the perturbed structures still retain par-
tial coherence, allowing us to assess how sensitive
the model is to incomplete or noisy signals. All
experiments are conducted with Llama-3.3-70B
under consistent retrieval and decoding conditions
to maintain causal interpretability.
Figure 4 presents the results of the perturbation
study. Panel (a) of Figure 4 exhibits that perturbing
intra-chunk structures leads to consistent perfor-
mance decrease. Randomly shuffling a portion of
rhetorical relation labels reduces the LLM ScoreModel RL ↑BERTScore ↑SARI↑SummaC ↑
Baselines with full context
Llama-3.1-8B 15.33 59.27 35.43 48.31
Qwen2.5-72B 17.00 60.41 37.62 55.03
Llama-3.3-70B 17.19 61.03 37.65 54.73
Baselines with standard RAG
Llama-3.1-8B 17.12 60.35 38.01 55.26
Qwen2.5-72B 18.09 61.28 38.32 60.12
Llama-3.3-70B 18.17 61.37 37.74 60.39
SOTA Results
RSTformer (Liu et al., 2024) 20.12 62.80 41.56 —
SingleTurnPlan (Liang et al., 2024) 19.68 — — —
Plan-Input (Liu et al., 2025b) — 65.32 —72.40
Ours
Disco-RAG (Llama-3.1-8B) 19.25 63.47 40.25 63.35
Disco-RAG (Qwen2.5-72B) 20.10 64.83 41.48 66.30
Disco-RAG (Llama-3.3-70B)21.11 65.67 44.3769.48
Table 3: Performance on the SciNews dataset.
Disco-RAG beats both zero-shot and standard RAG, and
often surpasses prior SOTA across multiple metrics.
from 62.07 to 55.48. Randomly altering some nu-
cleus–satellite roles lowers the score to 55.15. Re-
moving a randomly selected subtree connection
decreases the score to 56.77. Panel (b) of Fig-
ure 4 presents the effect of modifying rhetorical
graphs. Randomly removing some graph connec-
tions between chunks reduces the score to 57.60.
Randomly flipping the directions of a subset of
edges yields 55.82, while replacing some discourse
relation labels within the graph gives 55.50. Panel
(c) of Figure 4 analyzes the degradation of rhetor-
ical plans. Omitting the plan altogether reduces
performance to 59.75. Shuffling some of the step
sequences causes a decline to 57.50, while remov-
ing a subset of steps results in 58.14.
Across all three dimensions, structural pertur-
bations lead to performance reduction, yet do not
eliminate the benefits conferred by structure-aware
modeling. Even when exposed to corrupted or in-
complete signals, Disco-RAG consistently outper-
forms both the standard RAG and the full context
setting. These results confirm that the observed im-
provements are not merely due to the inclusion
of additional tokens, but instead arise from the
model’s capacity to leverage structural signals.
Human Evaluation.We conduct a human evalu-
ation on the SciNews dataset. We randomly sam-
ple 15 test articles and ask three graduate students

MethodSet 1 Set 2 Set 3 Set 4 Overall
LLM Score ↑EM↑LLM Score ↑EM↑LLM Score ↑EM↑LLM Score ↑EM↑LLM Score ↑EM↑
Disco-RAG (full) 71.00 0.38 63.61 0.28 58.86 0.22 54.62 0.11 62.07 0.24
w/o RST tree 65.45 0.34 58.41 0.22 54.90 0.14 47.63 0.07 56.22 0.20
w/o rhetorical graph 67.80 0.33 58.87 0.24 54.04 0.15 48.16 0.10 57.10 0.21
w/o planning 69.11 0.35 60.14 0.25 57.20 0.20 50.34 0.1259.75 0.22
Standard RAG 62.78 0.34 53.77 0.18 45.77 0.13 35.61 0.07 49.33 0.17
w/ retrieve-and-plan 64.05 0.35 54.92 0.18 46.11 0.14 37.22 0.07 50.64 0.14
w/ plan-and-retrieve 64.62 0.35 55.38 0.19 47.82 0.14 38.08 0.08 51.38 0.18
Table 4: Ablation study of the three modules in Disco-RAG with Llama-3.3-70B .w/o RST treeremoves intra-
chunk modeling,w/o rhetorical graphremoves inter-chunk modeling, andw/o planningremoves discourse-aware
planning. We additionally report two generic planning baselines built on standard RAG.retrieve-and-plangenerates
a free-form plan conditioned on retrieved chunks before generation, andplan-and-retrievefirst generates a free-form
plan from the query and then performs a retrieval step guided by this plan.
Remove
EdgesShuffle
Nucleus-SatelliteShuffle
Rel. Labels404550556065LLM Score56.77
55.15 55.48(a) RST Tree Perturbation
Remove
ConnectionsShuffle
DirectionsShuffle
Rel. Labels404550556065LLM Score57.60
55.82 55.50(b) Rhetorical Graph Perturbation
No Plan Shuffled Plan Missing Steps404550556065LLM Score59.75
57.5058.14(c) Planning Compliance Perturbation
Full Context Standard RAG Discourse-RAG
Figure 4: Effect of structural perturbations on performance. Panels (a), (b), and (c) correspond to intra-chunk RST
trees, inter-chunk rhetorical graphs, and discourse-aware plans, respectively. Each perturbation involves randomly
altering or removing the relevant elements.
with computer science backgrounds to rate four
anonymized systems, namely the full context LLM
without retrieval, the standard RAG baseline, our
Disco-RAG model, and human-written references.
Following the protocol of Liu et al. (2024), human
raters read each article together with four shuffled
summaries and assign scores on a three-point Likert
scale along four dimensions,Relevance,Simplicity,
Conciseness, andFaithfulness, where higher val-
ues indicate better quality. We measure inter-rater
agreement using Fleiss’ κand obtain average val-
ues of 0.72, 0.65, 0.66, and 0.68 on the four dimen-
sions, indicating substantial consistency among an-
notators. Table 5 reports the average scores across
all annotated samples. Detailed instructions for the
human raters are provided in Appendix M.
System Relevance ↑Simplicity ↑Conciseness ↑Faithfulness ↑
Full Context 1.65 1.98 1.52 1.45
Standard RAG 1.87 2.12 1.60 1.67
Disco-RAG2.40 2.43 2.27 2.53
Human Reference 2.89 2.63 2.48 2.88
Table 5: Average human ratings on SciNews. Scores
are computed on a three-point Likert scale, and higher
values indicate better performance.
Table 5 suggests that Disco-RAG improves per-
ceived answer quality over both full context and
standard RAG systems, with considerable gains in
FaithfulnessandConciseness. Human-written ref-
erences remain the strongest overall according to
annotators, which indicates that there is still room
for future model development, but the ranking ofneural systems in human evaluation is consistent
with the trends observed in automatic metrics and
supports the benefits of discourse-aware retrieval-
augmented generation. Further discussion of the
parsing evaluation, efficiency analysis, shallow dis-
course marker analysis, the impact of model train-
ing, significance testing, attention and decoding
behavior, qualitative case studies, and LLM usage
can be found in Appendix D, Appendix E, Ap-
pendix F, Appendix G, Appendix H, Appendix I,
Appendix J and Appendix K, respectively.
7 Conclusion
In this study, we tackle the absence of discourse
structure modeling in existing RAG approaches
by presenting Disco-RAG . Grounded in Rhetorical
Structure Theory, our approach constructs both lo-
cal hierarchies and global discourse representations
over retrieved evidence and leverages them to de-
rive a high-level blueprint that guides the reasoning
process of the language model. Experimental re-
sults demonstrate that Disco-RAG achieves consid-
erable gains across multiple knowledge-intensive
QA and summarization tasks, surpassing previous
state-of-the-art methods without in-domain fine-
tuning. Ablation studies validate the complemen-
tary contributions of each structural component.
Taken together, these findings highlight structured
discourse modeling as a promising direction for
advancing retrieval-augmented generation.

8 Ethical Considerations
All datasets used in this work are publicly available,
and we follow the original licenses and usage poli-
cies. Our pipeline operates entirely on de-identified
text without collecting or inferring personal iden-
tities or sensitive attributes, and all intermediate
artifacts, such as discourse structures and plans, are
derived solely from these corpora rather than live
user queries or proprietary logs. Human evalua-
tors participate voluntarily and are appropriately
compensated, while care is taken to avoid exposing
annotators to harmful content beyond what already
exists in the datasets. Large language models are
used as backbone retrievers/generators and as as-
sistive tools for discourse parsing and language re-
finement (as noted in Appendix K), but they do not
replace the authors in methodological design or re-
sult interpretation. We also comply with ACL Pol-
icy on Publication Ethics, and we caution against
applying our system in high-stakes environments
without additional safeguards and human oversight.
9 Limitations
Data.Our experiments apply three publicly avail-
able benchmarks, Loong, ASQA, and SciNews.
These datasets provide a good basis for evaluating
long-context reasoning, but we do not conduct a
dedicated analysis of potential biases in their con-
tent or label distributions. As a result, the behavior
ofDisco-RAG across genres, languages, or data
collection processes that differ substantially from
these benchmarks remains an open question, and
extending our analysis to different corpora is a di-
rection for future work.
Model.We instantiate our framework with
three open-source large language models, in-
cluding Llama-3.1-8B ,Llama-3.3-70B , and
Qwen2.5-72B , together with a fixed retriever. The
consistent gains observed across these backbones
suggest that the proposed discourse mechanism is
not tied to a specific model, yet we do not sys-
tematically explore alternative architectures, pa-
rameter scales, or decoding strategies. In prac-
tice, our framework assumes that the backbone
models provide basic discourse understanding and
long-context processing, and we expect that fu-
ture improvements in foundation models can be
incorporated with minimal changes to the overall
design so that Disco-RAG remains largely decou-
pled from specific language models. Further workis also needed to understand how Disco-RAG be-
haves with smaller or more specialized models and
under tighter computational constraints.
Parser.Our method depends on an LLM-based
discourse parser to produce intra-chunk RST trees
and inter-chunk rhetorical graphs. In Appendix D,
we evaluate this parser on the standard RST-DT
benchmark and show that the zero-shot LLM parser
attains span and nuclearity F1 scores that are close
to a fine-tuned supervised baseline. Together with
the ablation and perturbation studies in Table 4
and Figure 4, these results indicate that better dis-
course structures lead to stronger downstream per-
formance, and more accurate parsers can be readily
plugged into our pipeline without changing the
retrieval or generation components. Our goal in
this work is not to combine Disco-RAG with the
strongest available parser, but rather to demonstrate
that explicitly modeling discourse information is
beneficial for RAG. Since the Loong, ASQA, and
SciNews datasets do not provide gold discourse
annotations, we cannot directly quantify parsing
accuracy on data samples, and we leave more fine-
grained comparisons with alternative parsers and
discourse formalisms to future work.
Automated Evaluation.We evaluate models
using a combination of Exact Match, ROUGE-
L, DR Score, BERTScore, SARI, SummaC, and
an LLM-based metric for Loong that relies on
GPT-4-turbo-2024-04-09 . This suite covers mul-
tiple aspects of quality and has been adopted in
prior work, while each metric has known limita-
tions, and the LLM-based judge may inherit biases
or topic preferences from its own training data. Re-
ported numbers should therefore be interpreted as
indicative rather than exhaustive, and future work
could benefit from more fine-grained evaluation
protocols and larger-scale human studies.
Efficiency.Compared with standard RAG,
Disco-RAG introduces additional token consump-
tion and latency because it requires additional
LLM calls for discourse parsing and rhetorical
planning. The measurements in Appendix E show
that this overhead is moderate under our settings.
Deploying our method in latency-sensitive or
large-scale applications will therefore require
engineering optimizations such as caching and
reusing discourse structures, batching structural
queries, or distilling lighter parsers and planners,
and there remains an inherent trade-off between

structural richness and runtime efficiency.
Scope and Generalization.The present work
focuses on long-context question answering and
summarization. The improvements we observe on
Loong, ASQA, and SciNews suggest that discourse
modeling is beneficial across multiple tasks, but we
do not explore applications like dialog-style ques-
tion answering, interactive agents, multilingual re-
trieval, or domains with stronger constraints. In
our framework, Rhetorical Structure Theory only
serves as a knowledge prior for organizing retrieved
evidence rather than a claim that it is the only cor-
rect formalization of discourse. Investigating how
Disco-RAG behaves in these broader settings and
how different notions of discourse structure influ-
ence retrieval-augmented generation is an impor-
tant direction for future work.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Rilwan Adewoyin, Ritabrata Dutta, and Yulan He. 2022.
RSTGen: Imbuing fine-grained interpretable control
into long-FormText generators. InProceedings of
the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, pages 1822–1835,
Seattle, United States. Association for Computational
Linguistics.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Parminder Bhatia, Yangfeng Ji, and Jacob Eisenstein.
2015. Better document-level sentiment analysis from
RST discourse parsing. InProceedings of the 2015
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 2212–2218, Lisbon, Portu-
gal. Association for Computational Linguistics.
Eric J Bigelow, Ari Holtzman, Hidenori Tanaka, and
Tomer Ullman. 2025. Forking paths in neural text
generation. InThe Thirteenth International Confer-
ence on Learning Representations.
Isabel Cachola, Daniel Khashabi, and Mark Dredze.
2025. Evaluating the evaluators: Are readability
metrics good measures of readability?arXiv preprint
arXiv:2508.19221.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. RQ-RAG:Learning to refine queries for retrieval augmented
generation. InFirst Conference on Language Model-
ing.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Ma-
hashweta Das, and Na Zou. 2025. MAIN-RAG:
Multi-agent filtering retrieval-augmented generation.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 2607–2622, Vienna, Austria.
Association for Computational Linguistics.
Rong Cheng, Jinyi Liu, Yan Zheng, Fei Ni, Jiazhen Du,
Hangyu Mao, Fuzheng Zhang, Bo Wang, and Jianye
Hao. 2025. DualRAG: A dual-process approach to in-
tegrate reasoning and retrieval for multi-hop question
answering. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 31877–31899, Vi-
enna, Austria. Association for Computational Lin-
guistics.
Elena Chistova. 2023. End-to-end argument mining
over varying rhetorical structures. InFindings of
the Association for Computational Linguistics: ACL
2023, pages 3376–3391, Toronto, Canada. Associa-
tion for Computational Linguistics.
Elena Chistova. 2024. Bilingual rhetorical structure
parsing with large parallel annotations. InFindings of
the Association for Computational Linguistics: ACL
2024, pages 9689–9706, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Song Duong, Florian Le Bronnec, Alexandre Al-
lauzen, Vincent Guigue, Alberto Lumbreras, Laure
Soulier, and Patrick Gallinari. 2025. SCOPE: A self-
supervised framework for improving faithfulness in
conditional text generation. InThe Thirteenth Inter-
national Conference on Learning Representations.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Masoomali Fatehkia, Ji Kim Lucas, and Sanjay Chawla.
2024. T-rag: lessons from the llm trenches.arXiv
preprint arXiv:2402.07483.
Akash Gautam, Lukas Lange, and Jannik Strötgen. 2024.
Discourse-aware in-context learning for temporal ex-
pression normalization. InProceedings of the 2024
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies (Volume 2: Short Papers),
pages 306–315, Mexico City, Mexico. Association
for Computational Linguistics.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,

Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. HippoRAG: Neu-
robiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Con-
ference on Neural Information Processing Systems.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From RAG to memory:
Non-parametric continual learning for large language
models. InForty-second International Conference
on Machine Learning.
Katsuhiko Hayashi, Tsutomu Hirao, and Masaaki Na-
gata. 2016. Empirical comparison of dependency
conversions for RST discourse trees. InProceedings
of the 17th Annual Meeting of the Special Interest
Group on Discourse and Dialogue, pages 128–136,
Los Angeles. Association for Computational Linguis-
tics.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen
Ling, and Liang Zhao. 2025. GRAG: Graph retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: NAACL 2025, pages
4145–4157, Albuquerque, New Mexico. Association
for Computational Linguistics.
Haoyu Huang, Yongfeng Huang, Yang Junjie, Zhenyu
Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, and
James Cheng. 2025. Retrieval-augmented generation
with hierarchical knowledge. InFindings of the Asso-
ciation for Computational Linguistics: EMNLP 2025,
pages 6044–6060, Suzhou, China. Association for
Computational Linguistics.
Shayekh Bin Islam, Md Asib Rahman, K S M Tozammel
Hossain, Enamul Hoque, Shafiq Joty, and Md Rizwan
Parvez. 2024. Open-RAG: Enhanced retrieval aug-
mented reasoning with open-source large language
models. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2024, pages 14231–
14244, Miami, Florida, USA. Association for Com-
putational Linguistics.
Pengcheng Jiang, Lang Cao, Ruike Zhu, Minhao
Jiang, Yunyi Zhang, Jimeng Sun, and Jiawei
Han. 2025. Ras: Retrieval-and-structuring for
knowledge-intensive llm generation.arXiv preprint
arXiv:2502.10996.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joon-
suk Park, and Jaewoo Kang. 2023. Tree of clarifica-
tions: Answering ambiguous questions with retrieval-
augmented large language models. InProceedings
of the 2023 Conference on Empirical Methods inNatural Language Processing, pages 996–1009, Sin-
gapore. Association for Computational Linguistics.
Philippe Laban, Tobias Schnabel, Paul N. Bennett, and
Marti A. Hearst. 2022. SummaC: Re-visiting NLI-
based models for inconsistency detection in summa-
rization.Transactions of the Association for Compu-
tational Linguistics, 10:163–177.
Dosung Lee, Wonjun Oh, Boyoung Kim, Minyoung
Kim, Joonsuk Park, and Paul Hongsuck Seo. 2025a.
ReSCORE: Label-free iterative retriever training
for multi-hop question answering with relevance-
consistency supervision. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 341–359,
Vienna, Austria. Association for Computational Lin-
guistics.
Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen
Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa
Rangwala, and Christos Faloutsos. 2025b. Hyb-
GRAG: Hybrid retrieval-augmented generation on
textual and relational knowledge bases. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 879–893, Vienna, Austria. Association
for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Mufei Li, Siqi Miao, and Pan Li. 2025a. Simple is effec-
tive: The roles of graphs and large language models
in knowledge-graph-based retrieval-augmented gen-
eration. InThe Thirteenth International Conference
on Learning Representations.
Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu
Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei
Han, Le Sun, and Yongbin Li. 2025b. StructRAG:
Boosting knowledge intensive reasoning of LLMs via
inference-time hybrid information structurization. In
The Thirteenth International Conference on Learning
Representations.
Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu,
Ling Zhong, Peilong Zhao, Mengshu Sun, Zhiqiang
Zhang, Jun Zhou, Wenguang Chen, Wen Zhang, and
Huajun Chen. 2025. Kag: Boosting llms in profes-
sional domains via knowledge augmented generation.
InCompanion Proceedings of the ACM on Web Con-
ference 2025, WWW ’25, page 334–343, New York,
NY , USA. Association for Computing Machinery.
Yi Liang, You Wu, Honglei Zhuang, Li Chen, Jiaming
Shen, Yiling Jia, Zhen Qin, Sumit Sanghai, Xuanhui
Wang, Carl Yang, and 1 others. 2024. Integrating
planning into single-turn long-form text generation.
arXiv preprint arXiv:2410.06203.

Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Teng Lin, Yizhang Zhu, Yuyu Luo, and Nan Tang. 2025.
Srag: Structured retrieval-augmented generation for
multi-entity question answering over wikipedia graph.
arXiv preprint arXiv:2503.01346.
Dongqi Liu and Vera Demberg. 2024. RST-LoRA: A
discourse-aware low-rank adaptation for long docu-
ment abstractive summarization. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), pages 2200–2220, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Dongqi Liu, Yifan Wang, and Vera Demberg. 2023. In-
corporating distributions of discourse structure for
long document abstractive summarization. InPro-
ceedings of the 61st Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 5574–5590, Toronto, Canada. Associ-
ation for Computational Linguistics.
Dongqi Liu, Yifan Wang, Jia Loy, and Vera Demberg.
2024. SciNews: From scholarly complexities to pub-
lic narratives – a dataset for scientific news report
generation. InProceedings of the 2024 Joint In-
ternational Conference on Computational Linguis-
tics, Language Resources and Evaluation (LREC-
COLING 2024), pages 14429–14444, Torino, Italia.
ELRA and ICCL.
Dongqi Liu, Chenxi Whitehouse, Xi Yu, Louis Mahon,
Rohit Saxena, Zheng Zhao, Yifu Qiu, Mirella Lapata,
and Vera Demberg. 2025a. What is that talk about?
a video-to-text summarization dataset for scientific
presentations. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 6187–6210,
Vienna, Austria. Association for Computational Lin-
guistics.
Dongqi Liu, Xi Yu, Vera Demberg, and Mirella Lapata.
2025b. Explanatory summarization with discourse-
driven planning.Transactions of the Association for
Computational Linguistics, 13:1146–1170.
Guanran Luo, Zhongquan Jian, Wentao Qiu, Meihong
Wang, and Qingqiang Wu. 2025. DTCRS: Dynamic
tree construction for recursive summarization. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 10948–10963, Vienna, Austria.
Association for Computational Linguistics.
Bolei Ma, Yuting Li, Wei Zhou, Ziwei Gong, Yang Janet
Liu, Katja Jasinskaja, Annemarie Friedrich, Julia
Hirschberg, Frauke Kreuter, and Barbara Plank. 2025.
Pragmatics in the era of large language models: A
survey on datasets, evaluation, opportunities and chal-
lenges. InProceedings of the 63rd Annual Meeting ofthe Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 8679–8696, Vienna,
Austria. Association for Computational Linguistics.
Aru Maekawa, Tsutomu Hirao, Hidetaka Kamigaito,
and Manabu Okumura. 2024. Can we obtain signifi-
cant success in RST discourse parsing by using large
language models? InProceedings of the 18th Confer-
ence of the European Chapter of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 2803–2815, St. Julian’s, Malta. Association
for Computational Linguistics.
William C Mann and Sandra A Thompson. 1987.
Rhetorical structure theory: A theory of text orga-
nization. Technical report, University of Southern
California, Information Sciences Institute Los Ange-
les.
William C Mann and Sandra A Thompson. 1988.
Rhetorical structure theory: Toward a functional the-
ory of text organization.Text-interdisciplinary Jour-
nal for the Study of Discourse, 8(3):243–281.
Daniel Marcu. 1997. From discourse structures to text
summaries. InIntelligent Scalable Text Summariza-
tion.
Daniel Marcu. 1999. A decision-based approach to
rhetorical parsing. InProceedings of the 37th An-
nual Meeting of the Association for Computational
Linguistics, pages 365–372.
Costas Mavromatis and George Karypis. 2025. GNN-
RAG: Graph neural retrieval for efficient large lan-
guage model reasoning on knowledge graphs. In
Findings of the Association for Computational Lin-
guistics: ACL 2025, pages 16682–16699, Vienna,
Austria. Association for Computational Linguistics.
Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Bao-
long Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong-Zhi
Li, Duzhen Zhang, and 1 others. 2025. A survey of
context engineering for large language models.arXiv
preprint arXiv:2507.13334.
Itai Mondshine, Tzuf Paz-Argaman, and Reut Tsarfaty.
2025. Beyond n-grams: Rethinking evaluation met-
rics and strategies for multilingual abstractive summa-
rization. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 19019–19035, Vienna,
Austria. Association for Computational Linguistics.
Hellina Hailu Nigatu, Min Li, Maartje Ter Hoeve, Saloni
Potdar, and Sarah Chasins. 2025. mRAKL: Multi-
lingual retrieval-augmented knowledge graph con-
struction for low-resourced languages. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 13072–13089, Vienna, Austria. Associa-
tion for Computational Linguistics.
Renyi Qu, Ruixuan Tu, and Forrest Sheng Bao. 2025.
Is semantic chunking worth the computational cost?
InFindings of the Association for Computational

Linguistics: NAACL 2025, pages 2155–2177, Al-
buquerque, New Mexico. Association for Computa-
tional Linguistics.
Haoran Que and Wenge Rong. 2025. PIC: Unlock-
ing long-form text generation capabilities of large
language models via position ID compression. In
Proceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 6982–6995, Vienna, Austria.
Association for Computational Linguistics.
Kashob Kumar Roy, Pritom Saha Akash, Kevin Chen-
Chuan Chang, and Lucian Popa. 2024. ConTRe-
Gen: Context-driven tree-structured retrieval for
open-domain long-form text generation. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2024, pages 13773–13784, Miami, Florida,
USA. Association for Computational Linguistics.
Diego Sanmartin. 2024. Kg-rag: Bridging the gap
between knowledge and creativity.arXiv preprint
arXiv:2405.12035.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. RAPTOR: Recursive abstractive processing
for tree-organized retrieval. InThe Twelfth Interna-
tional Conference on Learning Representations.
Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-
Wei Chang. 2022. ASQA: Factoid questions meet
long-form answers. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 8273–8288, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan,
Changyue Wang, Hongning Wang, Ziyi Ye, Yujia
Zhou, and Yiqun Liu. 2025. Parametric retrieval
augmented generation. InProceedings of the 48th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, pages
1240–1250.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, and 1 others. 2023. Llama: Open and effi-
cient foundation language models.arXiv preprint
arXiv:2302.13971.
Jiaan Wang, Fandong Meng, Zengkui Sun, Yunlong
Liang, Yuxuan Cao, Jiarong Xu, Haoxiang Shi, and
Jie Zhou. 2025a. An empirical study of many-to-
many summarization with large language models.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 11328–11344, Vienna, Austria.
Association for Computational Linguistics.
Minzheng Wang, Longze Chen, Fu Cheng, Shengyi
Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan
Xu, Lei Zhang, Run Luo, Yunshui Li, Min Yang, Fei
Huang, and Yongbin Li. 2024a. Leave no documentbehind: Benchmarking long-context LLMs with ex-
tended multi-doc QA. InProceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5627–5646, Miami, Florida,
USA. Association for Computational Linguistics.
Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and
Yuchi Ma. 2025b. Archrag: Attributed community-
based hierarchical retrieval-augmented generation.
arXiv preprint arXiv:2502.09891.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing
Huang. 2024b. Searching for best practices in
retrieval-augmented generation. InProceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pages 17716–17736, Mi-
ami, Florida, USA. Association for Computational
Linguistics.
Zhitong Wang, Cheng Gao, Chaojun Xiao, Yufei Huang,
Shuzheng Si, Kangyang Luo, Yuzhuo Bai, Wen-
hao Li, Tangjian Duan, Chuancheng Lv, Guoshan
Lu, Gang Chen, Fanchao Qi, and Maosong Sun.
2025c. Document segmentation matters for retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: ACL 2025, pages
8063–8075, Vienna, Austria. Association for Compu-
tational Linguistics.
Jian Wu, Linyi Yang, Dongyuan Li, Yuliang Ji, Manabu
Okumura, and Yue Zhang. 2025a. MMQA: Evaluat-
ing LLMs with multi-table multi-hop complex ques-
tions. InThe Thirteenth International Conference on
Learning Representations.
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min
Xu, Filippo Menolascina, Yueming Jin, and Vicente
Grau. 2025b. Medical graph RAG: Evidence-based
medical large language model via graph retrieval-
augmented generation. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 28443–
28467, Vienna, Austria. Association for Computa-
tional Linguistics.
Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Wei-
jia Shi, Sheng Wang, Linjun Zhang, James Zou, and
Huaxiu Yao. 2025. MMed-RAG: Versatile multi-
modal RAG system for medical vision language mod-
els. InThe Thirteenth International Conference on
Learning Representations.
Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen,
and Chris Callison-Burch. 2016. Optimizing sta-
tistical machine translation for text simplification.
Transactions of the Association for Computational
Linguistics, 4:401–415.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.

Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf
Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuan-
hui Wang, and Michael Bendersky. 2025. Inference
scaling for long-context retrieval augmented genera-
tion. InThe Thirteenth International Conference on
Learning Representations.
Amir Zeldes, Tatsuya Aoyama, Yang Janet Liu, Siyao
Peng, Debopam Das, and Luke Gessler. 2025. eRST:
A signaled graph theory of discourse relations and
organization.Computational Linguistics, 51(1):23–
72.
Jinghao Zhang, Yuting Liu, Wenjie Wang, Qiang Liu,
Shu Wu, Liang Wang, and Tat-Seng Chua. 2025a.
Personalized text generation with contrastive activa-
tion steering. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 7128–7141,
Vienna, Austria. Association for Computational Lin-
guistics.
Taolin Zhang, Dongyang Li, Qizhou Chen, Chengyu
Wang, and Xiaofeng He. 2025b. BELLE: A bi-level
multi-agent reasoning framework for multi-hop ques-
tion answering. InProceedings of the 63rd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 4184–4202,
Vienna, Austria. Association for Computational Lin-
guistics.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Eval-
uating text generation with bert. InInternational
Conference on Learning Representations.
Xiaoming Zhang, Ming Wang, Xiaocui Yang, Daling
Wang, Shi Feng, and Yifei Zhang. 2024. Hierarchical
retrieval-augmented generation model with rethink
for multi-hop question answering.arXiv preprint
arXiv:2408.11875.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, and 1 others. 2025c.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176.
Jihao Zhao, Zhiyuan Ji, Zhaoxin Fan, Hanyu Wang,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2025. MoC: Mixtures of text chunking learners for
retrieval-augmented generation system. InProceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 5172–5189, Vienna, Austria. Associa-
tion for Computational Linguistics.
Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and
Wei Hu. 2025. Knowledge graph-guided retrieval
augmented generation. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 8912–8924, Albuquerque, New Mexico.
Association for Computational Linguistics.A Details of Datasets
Table 6 summarizes the key statistics of the Loong,
ASQA, and SciNews datasets used in our experi-
ments. The Loong dataset is a cross-domain and
multi-task benchmark that covers long-text under-
standing, reasoning, and generation. It is specifi-
cally designed to evaluate models’ ability to handle
long-context inputs and perform comprehensive
reasoning. The ASQA (Ambiguous Question An-
swering) dataset focuses on questions with mul-
tiple valid interpretations, providing explanatory
responses that evaluate a model’s capacity to re-
solve semantic ambiguity and produce interpretable
answers. The SciNews dataset centers on the sci-
entific news domain, spanning a wide range of sci-
entific topics. It contains news articles paired with
academic papers and is intended to test models’
capacity for long-context news understanding and
generation.
B Details of Evaluation Metrics
For the Loong dataset.We report two evalua-
tion metrics. The first is Exact Match (EM), which
is a strict measure of the percentage of model pre-
dictions that exactly match the ground truth an-
swers. It is a binary measure that assigns a score
of one for a perfect match and zero otherwise.
The second metric is the LLM Score (Wang et al.,
2024a), ranging from 0 to 100. Following the pro-
tocol introduced by the dataset authors, we em-
ploy GPT-4-turbo-2024-04-09 as an automated
evaluator to rate the overall quality of generated
responses. Unlike EM, which captures only factual
correctness, the LLM Score provides a holistic eval-
uation by jointly considering comprehensiveness,
clarity, and adherence to instructions, thereby offer-
ing a more integrated assessment across multiple
dimensions of quality.
For the ASQA dataset.We adopt the standard
evaluation suite. The first is Exact Match (EM),
defined as before. The second is ROUGE-L (Lin,
2004), an evaluation metric based on the Longest
Common Subsequence (LCS). It measures the n-
gram overlap between prediction and reference by
identifying the longest sequence of words that oc-
curs in both while preserving word order, thereby
evaluating the coverage of key information. Given
a predicted text ˆyiand a reference text yi, let
LCS(ˆy i, yi)denote the length of their longest com-
mon subsequence. The ROUGE-L recall, precision,

Dataset Loong ASQA SciNews
Split Set1(10K-50K) Set2(50K-100K) Set3(100K-200K) Set4(200K-250K) Test Test
Language EN, ZH EN, ZH EN, ZH EN, ZH EN EN
Test Instances 323 564 481 232 1015 4188
Table 6: Summary statistics of the Loong, ASQA, and SciNews datasets used in our experiments.
and F1 are defined as:
RL=LCS(ˆy i, yi)
|yi|(1)
PL=LCS(ˆy i, yi)
|ˆyi|(2)
FL=(1 +β2)·R L·PL
RL+β2·PL(3)
where |yi|and|ˆyi|are the lengths of the refer-
ence and predicted texts, respectively, and βis set
to one by default to balance recall and precision. In
our experiments, we report ROUGE-L F1 (RL).
The third metric is the Disambiguation Recall
(DR) Score (Stelmakh et al., 2022), which is specif-
ically designed for ASQA to evaluate whether a
prediction covers all possible disambiguated an-
swers present in the reference set. While ROUGE-
L cannot distinguish between two fluent but se-
mantically divergent answers, the DR score ex-
plicitly evaluates coverage across multiple refer-
ence answers. A higher DR score indicates that
the generated response captures a larger fraction
of the possible interpretations of an ambiguous
question. Given multiple reference answers Yi=
{y(1)
i, y(2)
i, . . . , y(ki)
i}for a query and a generated
answer ˆyi, the instance-level DR score is defined
as:
DRi=1
|Yi||Yi|X
j=11
ˆyicontains the information iny(j)
i
(4)
where 1[·]is an indicator function equal to one
if the predicted answer includes the content of a
reference answer y(j)
i, and zero otherwise. The
overall DR score acrossNqueries is defined as:
DR=1
NNX
i=1DRi.(5)
For the SciNews dataset.We focus on sum-
marization quality using four metrics. The first
is ROUGE-L, as defined above. The second is
BERTScore (Zhang et al., 2020), which computessemantic similarity between prediction and ref-
erence using contextual embeddings from a pre-
trained BERT model. The third is SARI (Xu et al.,
2016), which assesses the quality of simplifica-
tion by comparing system outputs against both the
source text and the reference texts. SARI explicitly
measures the precision and recall of words that are
added, deleted, and kept. For a source sentence
si, a prediction ˆyi, and a set of reference simpli-
fications Yi={y(1)
i, . . . , y(ki)
i}, SARI is defined
as:
SARI=1
3
AddF1+KeepF1+Del F1
(6)
where Add F1, KeepF1, and Del F1denote the F1
scores for added, kept, and deleted n-grams rela-
tive to both the source and the reference sets. The
fourth metric is SummaC (Laban et al., 2022), a
model-based measure of factual consistency. Sum-
maC can be used to determine whether a generated
summary is entailed by its source document and
detects unsupported or hallucinated content, which
is essential for ensuring the reliability of generated
text.
C Details of Baselines
Here we describe the baselines used for compari-
son:
•Standard RAG(Lewis et al., 2020) We
implement the standard retrieval-augmented
generation framework, where a retriever
(Qwen3-Embedding-8B ) retrieves relevant
documents and a generator ( Llama-3.1-8B ,
Llama-3.3-70B orQwen2.5-72B ) produces the
final answer conditioned on the retrieved context.
•GraphRAG(Edge et al., 2024) augments re-
trieval with a graph-based knowledge representa-
tion by constructing a semantic knowledge graph
from retrieved passages. It leverages community
detection to capture global structures and inte-
grates graph contexts into generation, enabling
more accurate and coherent reasoning across doc-
uments.
•RQ-RAG(Chan et al., 2024) refines queries
through explicit rewriting, decomposition, and

disambiguation before retrieval. It trains LLMs
end-to-end on a curated dataset with search-
augmented supervision, enabling dynamic query
refinement and improving both single-hop and
multi-hop QA by learning to search only when
needed.
•FLARE(Jiang et al., 2023) actively decides
when and what to retrieve during generation by
predicting upcoming sentences and using them as
queries to fetch additional documents whenever
low-confidence tokens appear.
•Tree of Clarifications(Kim et al., 2023) ad-
dresses ambiguous questions by recursively con-
structing a tree of disambiguated questions with
retrieval-augmented few-shot prompting, prun-
ing unhelpful branches through self-verification,
and generating a long-form answer that covers
all valid interpretations.
•Open-RAG(Islam et al., 2024) enhances
retrieval-augmented reasoning with open-source
LLMs by transforming a dense model into
a parameter-efficient sparse mixture-of-experts,
combining contrastive learning against distrac-
tors with hybrid adaptive retrieval.
•ConTReGen(Roy et al., 2024) employs a
context-driven, tree-structured retrieval frame-
work for open-domain long-form text generation.
It performs top-down planning to recursively de-
compose a query into sub-questions for in-depth
retrieval, followed by bottom-up synthesis to in-
tegrate information from leaf nodes to the root.
•DualRAG(Cheng et al., 2025) introduces a dual-
process framework for multi-hop QA, consist-
ing of Reasoning-augmented Querying (RaQ),
which identifies knowledge gaps and formulates
targeted queries, and progressive Knowledge Ag-
gregation (pKA), which filters and structures re-
trieved information into a coherent knowledge
outline.
•RAS(Jiang et al., 2025) interleaves iterative re-
trieval planning with dynamic construction of
query-specific knowledge graphs. It converts
retrieved text into factual triples, incrementally
builds a structured graph, and conditions genera-
tion on the evolving graph.
•MAIN-RAG(Chang et al., 2025) is a training-
free framework that employs three LLM agents
to collaboratively filter and rank retrieved docu-
ments. It introduces an adaptive judge bar that
dynamically adjusts relevance thresholds basedon score distributions, effectively reducing noisy
retrievals while preserving relevant information.
•StructRAG(Li et al., 2025b) introduces hy-
brid information structurization for knowledge-
intensive reasoning. It employs a hybrid struc-
ture router to select the optimal structure type
(e.g., table, graph, catalogue), a scattered knowl-
edge structurizer to transform raw documents into
structured knowledge, and a structured knowl-
edge utilizer to decompose complex questions
and infer accurate answers based on the struc-
tured representation.
D Parsing Evaluation
To assess the parser quality on our framework, we
evaluate the LLM-based parser used in Disco-RAG
on the RST-DT benchmark following the evalua-
tion protocol of Maekawa et al. (2024). We com-
pare a fine-tuned RST parser from Maekawa et al.
(2024) with our zero-shot parser instantiated with
Llama-3.3-70B . Both models are evaluated on
span F1, nuclearity F1, and relation F1 using the
official data splits and scoring scripts of the bench-
mark, and the results are summarized in Table 7.
Our zero-shot parser attains competitive scores that
are close in nuclearity and relation prediction and
somewhat lower in the span prediction, which still
reflects reasonable sensitivity to rhetorical seman-
tics without any task-specific tuning.
Model Setting Span F1 Nuclearity F1 Relation F1
Maekawa et al. (2024) Supervised 79.8 70.4 60.0
Our Parser Unsupervised 70.4 63.1 58.6
Table 7: Evaluation of the RST parser on the RST-DT
benchmark following the protocol of Maekawa et al.
(2024).
Note that parser development is not the primary
focus of Disco-RAG , and these results indicate that
the zero-shot LLM parser provides a reasonable
structural signal for downstream reasoning. More-
over, the fine-tuned parser only generates output
with a specific format, and cannot complete the
rhetorical graph prediction between chunks, while
the zero-shot parser provides such flexibility.
We further conduct a case study to examine
whether the parser outputs are acceptable for the
downstream task when gold annotations are un-
available for our benchmarks. For each of Loong,
ASQA, and SciNews, we randomly select 10 in-
stances from the test set and run our pipeline to
obtain intra-chunk RST trees, inter-chunk rhetor-

ical graphs, and discourse-aware plans for the re-
trieved evidence. We then ask three human anno-
tators to judge two questions with binary labels.
The first question evaluateswhether the predicted
discourse structures are broadly acceptable, mean-
ing that they capture major relations within and
between chunks even if they are not perfectly accu-
rate in every detail. The second question evaluates
whether the discourse-aware plan is acceptable,
meaning that it organizes the answer in a reason-
able order and reflects the main evidence required
by the query.
Across the sampled instances, the average ac-
ceptability rates are 0.72 for intra-chunk discourse
trees, 0.80 for inter-chunk rhetorical graphs, and
0.93 for discourse-aware plans. The inter-annotator
agreement measured by Fleiss’ κis 0.709 for
intra-chunk discourse trees, 0.733 for inter-chunk
rhetorical graphs, and 0.862 for discourse-aware
plans, indicating high consistency among annota-
tors. These results suggest that the parsing outputs
and the plans provide usable discourse signals for
our framework, and we expect that improved pars-
ing performance would further enhance the relia-
bility of discourse structures and thereby support
additional gains in answer quality.
E End-to-End Efficiency Analysis
We provide an end-to-end efficiency comparison
between standard RAG and Disco-RAG to quantify
the additional inference cost introduced by struc-
tural modeling. Experiments are conducted on the
Loong benchmark using the same configuration.
To be specific, we adopt the same retrieval corpus
and Top- kretrieval strategy (with chunk size fixed
at 256 tokens), the same decoding hyperparameters,
and we vary only whether discourse-aware compo-
nents are enabled. Standard RAG performs a single
generation call conditioned on the retrieved con-
text. Disco-RAG assumes that RST trees over the
corpus have been pre-parsed offline, and at infer-
ence time adds one listwise inter-chunk rhetorical
graph prediction and one discourse-aware planning
call before the final answer generation. All mea-
surements are obtained on a cluster equipped with
32×NVIDIA A100 80GB GPUs.
To characterize how inference cost scales with
the number of retrieved chunks, we fix the retriever
and model configuration on Loong and vary only
Top-k∈ {10,20,30,50} . For each setting, we
record (1) the average token cost, defined as thetotal number of prompt input and output tokens
across all LLM calls, and (2) the end-to-end latency
per query, including retrieval, structure prediction,
planning, and final generation. As shown in Ta-
ble 8, across all Top- kvalues, the token cost of
Disco-RAG is roughly 2.2 ×that of standard RAG,
and the end-to-end latency increases by about two
seconds per query on average.
Top-kStandard RAG Disco-RAG Standard RAG Disco-RAG
(Token Cost) (Token Cost) (Latency) (Latency)
10 3.6k 7.9k 7.6s 9.8s
20 6.4k 14.3k 13.8s 15.9s
30 8.2k 18.2k 21.5s 23.8s
50 13.5k 29.6k 32.8s 35.1s
Table 8: End-to-end token cost (input + output) and
latency under different Top- ksettings on the Loong
benchmark. Results are averaged over the same set of
queries with identical retriever, generator, and decoding
configurations for both Standard RAG and Disco-RAG.
Combining these results with the accuracy
improvements reported in the main paper on
Loong and the other benchmarks, we observe
that Disco-RAG incurs a moderate and bounded
increase in inference cost in exchange for sub-
stantial gains in performance over standard RAG,
and it often outperforms existing structure-aware
(training-based) RAG methods. We view this trade-
off between cost and performance as acceptable in
knowledge-intensive applications, where RST tree
parsing can be fully amortized offline and reused
across queries, while the additional listwise dis-
course inference and planning incur a stable over-
head that yields stronger discourse coherence and
factual robustness in the generated answers.
F Comparison with Shallow Discourse
Markers
We conduct a study on the Loong to assess whether
shallow discourse cues alone can provide compa-
rable benefits to full RST-based modeling. To this
end, we design a marker-based variant that con-
structs inter-chunk links using explicit discourse
markers without applying EDU segmentation. We
adopt a listwise inference strategy and provide
all retrieved chunks to Llama-3.3-70B in a sin-
gle pass, which jointly predicts discourse marker
for each ordered chunk pair based on connective
cues such ashowever,but,although,in contrast,
therefore,because,as a result, andmeanwhile.10
10The prompt used for shallow discourse marker inference
is provided in Appendix Figure 17.

Method LLM Score ↑Exact Match ↑
Standard RAG 49.33 0.17
w/ Discourse Markers 50.41 0.20
Disco-RAG 62.07 0.24
Table 9: Comparison of standard RAG, a shallow dis-
course marker variant, and Disco-RAG on the Loong
benchmark withLlama-3.3-70B.
Table 9 compares three configurations, namely
standard RAG, a shallow variant that augments
standard RAG with discourse markers, and the full
Disco-RAG model. The marker-based system im-
proves LLM Score from 49.33 to 50.41 and Exact
Match from 0.17 to 0.20. However, these gains
remain modest compared with the full discourse-
aware setting, where Disco-RAG reaches 62.07
LLM Score and 0.24 Exact Match under the same
conditions.
G Effect of Supervised Fine-Tuning
We examine how supervised fine-tuning in-
teracts with discourse-aware modeling on the
SciNews summarization benchmark. Starting from
Llama-3.3-70B , we fine-tune the generator on the
SciNews training split with a standard sequence-to-
sequence summarization objective and test using
RAG setting under three conditions. In the end-
to-end baseline, the model is trained using only
the raw document summary pairs without any dis-
course inputs. In the second setting, the model is
trained in the same way, but at test time, we aug-
ment the inputs with the intra-chunk RST trees,
inter-chunk rhetorical graphs, and discourse-aware
plans produced by Disco-RAG . In the third set-
ting, both training and inference use the discourse-
enriched inputs so that the model can adapt its
parameters to the structural signals. For compar-
ison, we also include the original training-free
Disco-RAG system that conditions generation on
discourse structures via prompting without param-
eter updates.
Method RL ↑SummaC ↑
End-to-end SFT (no discourse) 20.3 66.8
Disco-RAG (training-free) 21.1 69.5
SFT with test time discourse 22.8 72.3
SFT with train and test discourse 23.3 74.0
Table 10: Impact of supervised fine-tuning (SFT) and
discourse conditioning.
All systems share the same retrieval pipelineand decoding configuration as described in the
main paper, and we report RL and SummaC on
the SciNews test set. Table 10 shows that naive
end-to-end fine-tuning improves over the zero-shot
standard RAG baselines but remains behind the
training-free Disco-RAG . When discourse struc-
tures are provided at test time, the fine-tuned
model surpasses Disco-RAG , indicating that struc-
tural guidance and parameter adaptation bring com-
plementary benefits. When discourse structures
are incorporated during both training and infer-
ence, we observe further gains in both RL and
SummaC. These results confirm that our discourse-
aware framework is orthogonal to model training
and that injecting discourse information can consis-
tently enhance performance on top of supervised
fine-tuning.
H Significance Testing
To assess whether the improvements of Disco-RAG
over standard RAG are statistically reliable under
the same backbone model and decoding configu-
ration, we conduct paired t-tests on metric scores
for every benchmark, every backbone, and every
automatic metric. For human evaluation, we apply
the same paired t-test on the instance-level aver-
age ratings across the three annotators for each
criterion. Across all evaluation settings reported
in the paper, Disco-RAG is significantly better than
standard RAG withp <0.05.
I Attention and Decoding Analysis
To understand how discourse-aware modeling in-
fluences the decoding process, we analyze inter-
layer attention behavior and factual consistency
on the SciNews summarization benchmark. Using
Llama-3.3-70B as the generator, we compare con-
figurations that mirror the ablation settings in the
main paper: the full Disco-RAG model, variants
that remove either intra-chunk RST trees, inter-
chunk rhetorical graphs, or discourse-aware plans
while keeping other components unchanged, and
a standard RAG baseline that conditions genera-
tion only on retrieved chunks. For each configu-
ration, we compute inter-layer attention entropy
by averaging the token-level cross-layer attention
distributions over all decoder layers and attention
heads, and we report SummaC scores as a measure
of factual consistency with respect to the source
articles.
Table 11 summarizes the results. As structural

Model Configuration Inter-layer Attention Entropy ↓SummaC ↑
Disco-RAG 3.72 69.5
w/o Discourse Plan 4.07 68.6
w/o Intra-chunk RST 4.45 65.8
w/o Inter-chunk Graph 4.53 66.9
Standard RAG 6.21 60.4
Table 11: Inter-layer attention entropy and SummaC
on the SciNews benchmark with Llama-3.3-70B under
different structural configurations. Lower entropy and
higher SummaC indicate more focused attention and
better factual consistency, respectively.
guidance is removed, inter-layer attention entropy
increases, indicating that the model attends less se-
lectively to salient elements such as nucleus spans
and key graph nodes. At the same time, Sum-
maC scores decrease, reflecting a loss of factual
alignment between generated summaries and their
sources. The full Disco-RAG model achieves the
lowest attention entropy and the highest SummaC
score among the discourse-aware variants, while
standard RAG exhibits the highest entropy and the
lowest factual consistency. Taken together, these
findings suggest that discourse structures act as
soft constraints on content ordering and focus dur-
ing decoding, promoting more stable and faithful
generation.
J Case Studies
We present qualitative case studies to illustrate the
effectiveness of Disco-RAG compared to standard
RAG. The three cases are shown in Figure 5, Fig-
ure 6, and Figure 7.
K Use of Large Language Models
In preparing this paper, we use GPT-5.2 as a writ-
ing assistant for language polishing, grammar cor-
rection, and stylistic refinement. The model is not
involved in the research ideation, methodology de-
sign, experiments, or result interpretation. The
authors take full responsibility for the content of
the manuscript, including any parts generated with
the assistance of GPT-5.2 . In accordance with con-
ference policy, we confirm that the LLM is not an
author of this work and does not bear responsibility
for its scientific claims.
L Prompts for Disco-RAG
Appendix Figure 10, Figure 11, Figure 12 and Fig-
ure 18 present the prompts used inDisco-RAG.
M Human Evaluation Guidelines

Case Study 1
Query.When did The Lion King debut on Broadway?
Retrieved Chunks.
[Chunk 1] The Lion King is a stage musical ... It debuted on July 8, 1997,
in Minneapolis, Minnesota at the Orpheum Theatre ...
[Chunk 2] ... The musical premiered on Broadway at the New Amsterdam Theatre
in previews on October 15, 1997 ...
[Chunk 3] ... The official Broadway opening was on November 13, 1997,
at the New Amsterdam Theatre ...
[Chunk 4] ... On June 13, 2006, the Broadway production moved to the Minskoff Theatre,
where it is still running ...
Standard RAG Output.
The Lion King debuted on Broadway in 1997. It was first performed in Minneapolis
before moving to Broadway, where it became one of the most popular shows.
Disco-RAG Output.
The Lion King opened on Broadway more than once. It premiered in previews at the
New Amsterdam Theatre on October 15, 1997, and officially opened on November 13,
1997. Later, on June 13, 2006, the production moved to the Minskoff Theatre, where it
continues to run.
Reference Answer.
The Lion King opened on Broadway more than once. It premiered on Broadway at the
New Amsterdam Theatre in previews on October 15, 1997, then officially opened on
Broadway on November 13, 1997. On June 13, 2006, the Broadway production moved
to the Minskoff Theatre, another Broadway theater, where it is still running after more
than 9,000 performances.
Remark.Standard RAG produces a vague answer that misses key details about the Broadway
timeline, providing only the year 1997. In contrast, Disco-RAG integrates evidence across all
relevant chunks, capturing both the preview and official opening dates as well as the later theater
relocation.
Figure 5: Case study comparing standard RAG and Disco-RAG on the query “When did The Lion King debut on
Broadway?”. Our method captures both the preview and official opening as well as the later relocation, while
standard RAG gives only a vague year-based answer.

Case Study 2
Query.Where did Italy win the World Cup in 2006?
Retrieved Chunks.
[Chunk 1] The 2006 FIFA World Cup final was played on July 9, 2006,
between Italy and France ... after a 1-1 draw, Italy won 5-3 on penalties ...
[Chunk 2] ... The final was held at the Olympiastadion in Berlin, Germany,
with an attendance of 69,000 spectators ...
[Chunk 3] Italy claimed their fourth world title, having previously
won in 1934, 1938, and 1982 ...
[Chunk 4] ... France had won their only previous World Cup in 1998,
while Italy had lost finals in 1970 and 1994 ...
[Chunk 5] The match was part of the France-Italy football rivalry,
and marked Italy's first world title in 24 years ...
Standard RAG Output.
Italy won the 2006 World Cup in Munich, Germany, after a penalty shootout against
France.
Disco-RAG Output.
Italy defeated France in the 2006 FIFA World Cup Final, played on July 9, 2006, at the
Olympiastadion in Berlin, Germany. After a 1–1 draw, Italy prevailed 5–3 in the penalty
shootout.
Reference Answer.
The Italy national football team defeated France and was declared world champions in the
2006 World Cup Final, the last match of the 2006 FIFA World Cup, at the Olympiastadion
stadium in Berlin, Germany, on July 9, 2006. Italy had won three previous finals in 1934,
1938, 1982 and lost in 1970 and 1994, while France had won their only previous final in
1998.
Remark.Standard RAG incorrectly states that the final took place in Munich, likely due to
spurious retrieval or shallow aggregation of context. In contrast, Disco-RAG integrates evidence
across multiple chunks, correctly identifying the Olympiastadion in Berlin as the venue and
providing richer historical context.
Figure 6: Case study comparing standard RAG and our proposed Disco-RAG on the query “Where did Italy win the
World Cup in 2006?”. Our method correctly identifies the Olympiastadion in Berlin, while standard RAG produces
a factual error.

Case Study 3
Query.Is Pluto a planet?
Retrieved Chunks.
[Chunk 1] Pluto was discovered in 1930 and was long regarded as the ninth planet
in the Solar System in many textbooks and popular references.
[Chunk 2] In 2006, the International Astronomical Union adopted a formal definition
of "planet" and reclassified Pluto as a dwarf planet.
[Chunk 3] The 2006 definition requires a planet to orbit the Sun, be nearly round,
and clear its orbital neighborhood, which Pluto does not satisfy.
Standard RAG Output.
Yes. Pluto is the ninth planet in the Solar System.
Disco-RAG Output.
Pluto was historically treated as the ninth planet, but it is currently classified as a dwarf
planet. In 2006, the International Astronomical Union adopted a definition of a planet
that requires clearing the orbital neighborhood, and Pluto does not meet this criterion.
Reference Answer.
Pluto is not classified as a planet under the current astronomical definition. It was long
considered the ninth planet after its discovery in 1930, but the International Astronomical
Union reclassified it as a dwarf planet in 2006 because it does not clear its orbital
neighborhood.
Remark.Standard RAG commits to an outdated claim from Chunk 1 and produces a categorical
answer that conflicts with the current definition. In contrast, Disco-RAG uses the Contrast relation
to avoid merging incompatible statements and uses the Background relation to ground the final
answer in the relevant criterion, which reduces the risk of hallucinating a definitive but incorrect
conclusion under conflicting evidence.
Figure 7: Case study showing how discourse relations affect generation under conflicting evidence. The Contrast
relation prevents incompatible claims from being merged, and the Background relation provides the criterion needed
for a faithful answer.

Relation Definitions for Intra-chunk RST Tree Construction
Relation Definitions:
-ELABORATION: Satellite provides additional detail or information about the nucleus.
-EXPLANATION: Satellite explains or clarifies the nucleus content.
-EVIDENCE: Satellite provides evidence or proof for the nucleus claim.
-EXAMPLE: Satellite gives a specific example of the nucleus concept.
-CONTRAST: Satellite presents opposing or contrasting information.
-COMPARISON: Satellite compares two or more entities or concepts.
-CONCESSION: Satellite acknowledges opposing viewpoint while maintaining main claim.
-ANTITHESIS: Satellite presents directly opposite or contradictory information.
-CAUSE: Satellite describes the cause of an event or situation.
-RESULT: Satellite describes the result or consequence of an action.
-CONSEQUENCE: Satellite shows the outcome following from the nucleus.
-PURPOSE: Satellite explains the intended goal or purpose.
-CONDITION: Satellite specifies conditions under which something holds.
-TEMPORAL: Satellite indicates temporal relationship between events.
-SEQUENCE: Satellite shows sequential order of events or actions.
-BACKGROUND: Satellite provides background context or setting.
-CIRCUMSTANCE: Satellite describes circumstances surrounding an event.
-SUMMARY: Satellite summarizes or generalizes the nucleus content.
-RESTATEMENT: Satellite restates the nucleus in different words.
-EVALUATION: Satellite provides evaluation or assessment of the nucleus.
-INTERPRETATION: Satellite offers interpretation of the nucleus content.
-ATTRIBUTION: Satellite attributes information to a source.
-DEFINITION: Satellite defines a term or concept.
-CLASSIFICATION: Satellite classifies or categorizes information.
Figure 8: Relation Definitions for Intra-chunk RST Tree Construction.

Relation Definitions for Inter-chunk Rhetorical Graph Construction
Relation Definitions:
-SUPPORTS: Chunk provides support or evidence for another chunk.
-CONTRADICTS: Chunk contradicts or opposes another chunk.
-ELABORATES: Chunk elaborates on information in another chunk.
-EXEMPLIFIES: Chunk provides examples for another chunk’s concepts.
-CAUSES: Chunk describes causes for events in another chunk.
-RESULTS_FROM: Chunk describes results from another chunk’s events.
-ENABLES: Chunk describes what enables another chunk’s situation.
-PREVENTS: Chunk describes what prevents another chunk’s situation.
-PRECEDES: Chunk describes events that precede another chunk.
-FOLLOWS: Chunk describes events that follow another chunk.
-SIMULTANEOUS: Chunk describes simultaneous events with another chunk.
-BACKGROUND_FOR: Chunk provides background context for another chunk.
-GENERALIZES: Chunk provides general principles for another chunk’s specifics.
-SPECIFIES: Chunk provides specific details for another chunk’s generalizations.
-COMPARES_WITH: Chunk compares information with another chunk.
-CONTRASTS_WITH: Chunk contrasts information with another chunk.
-SUPPLEMENTS: Chunk supplements information in another chunk.
-REPLACES: Chunk replaces or updates information in another chunk.
-MOTIVATES: Chunk provides motivation for another chunk’s content.
-JUSTIFIES: Chunk justifies claims or actions in another chunk.
-UNRELATED: Chunk has no meaningful rhetorical or semantic relation to another chunk.
Figure 9: Relation Definitions for Inter-chunk Rhetorical Graph Construction.

Prompt for Intra-chunk RST Tree Construction
You are an expert in Rhetorical Structure Theory (RST) analysis. Your task is to analyze the given
text and construct a precise RST tree.
Critical instructions:
1. RST tree is a hierarchical tree structure (not a graph or network).
2. Each internal node has exactly two children: one nucleus (core) and one satellite (support) or
two nuclei at the same time.
3. The nucleus contains the main information; the satellite provides supporting content.
4. Relations describe how the satellite relates to the nucleus.
5. Think carefully and output ONLY ONE complete RST tree.
Allowed RST relations:
ELABORATION, EXPLANATION, EVIDENCE, EXAMPLE, CONTRAST, COMPARISON,
CONCESSION, ANTITHESIS, CAUSE, RESULT, CONSEQUENCE, PURPOSE, CONDITION,
TEMPORAL, SEQUENCE, BACKGROUND, CIRCUMSTANCE, SUMMARY , RESTATE-
MENT, EV ALUATION, INTERPRETATION, ATTRIBUTION, DEFINITION, CLASSIFICA-
TION
Relation definitions:
{Relation Definition}
Step-by-step process:
1. Segment text into meaningful elementary discourse unit (EDU).
2. Determine the most important EDU (this becomes the root nucleus).
3. For each other EDU, decide: Is it nucleus (core) or satellite (support)?
4. Assign one relation from the allowed list.
5. Build the binary tree bottom-up.
Required output format:
EDUs:
[1]<first EDU>
[2]<second EDU>
. . .
[N]<Nth EDU>
RST ANALYSIS:
RELATION(EDU i, EDU j): {RELATION TYPE}
. . .
TREE STRUCTURE:
ROOT[1-N]
|--- NUCLEUS[X] <EDU text> (N)
|--- SATELLITE[Y] <EDU text> (S): {RELATION TYPE}
Validation rules:
- Each EDU must be complete and meaningful.
- Relations must be chosen from the allowed list.
- Mark (N) for nucleus, (S) for satellite.
- Output exactly ONE complete tree.
TEXT TO ANALYZE:{chunk i}
Figure 10: Prompt for Intra-chunk RST Tree Construction. The relation definitions are provided in Figure 8.

Prompt for Listwise Discourse Relation Inference
You are an expert in discourse analysis. Your task is to infer the rhetorical relations jointly among
a list of retrieved text chunks. In each call to this prompt, you are given the entire set of chunks,
and you must construct a directed rhetorical graph over all of them.
Task objective:
Given a list of chunks CHUNK[1] ,CHUNK[2] , . . . , CHUNK[K] , determine for every ordered pair
of distinct chunks whether there exists a meaningful rhetorical relation from the source chunk
CHUNK[i] to the target chunk CHUNK[j] . If a relation exists, assign a directed discourse label;
otherwise, mark the pair asUNRELATED.
Relation direction:
For each ordered pair (i, j) withi̸=j , treat CHUNK[i] as the source and CHUNK[j] as the target.
The relation type should reflect how the source chunk contributes rhetorically to the target.
Allowed relation types:
SUPPORTS, CONTRADICTS, ELABORATES, EXEMPLIFIES, CAUSES, RESULTS_FROM,
ENABLES, PREVENTS, PRECEDES, FOLLOWS, SIMULTANEOUS, BACKGROUND_FOR,
GENERALIZES, SPECIFIES, COMPARES_WITH, CONTRASTS_WITH, SUPPLEMENTS,
REPLACES, MOTIV ATES, JUSTIFIES, UNRELATED
Relation definitions:
{Relation Definition}
Step-by-step process:
1. Carefully read all chunks in the list and identify the main claim, fact, or event expressed in each
one.
2. Reason about how each chunk relates to the others at the discourse level, taking into account
global context across all chunks.
3. For every ordered pair of distinct indices (i, j) , decide whether CHUNK[i] serves a discourse
function relative toCHUNK[j].
4. If a rhetorical link exists, assign exactly one relation type from the allowed list.
Required output format:
For each ordered pair(i, j)withi̸=j, output one line in the following format:
CHUNK[i] -> CHUNK[j]: {RELATION_TYPE}
List all such lines for all ordered pairs in a consistent order (e.g., sorted byithenj).
Validation rules:
- Use only the allowed relation types.
- Relation direction must be fromCHUNK[i]toCHUNK[j].
- Output exactly one relation type for every ordered pair withi̸=j.
TEXT TO ANALYZE:
CHUNK[1]: [first chunk]
CHUNK[2]: [second chunk]
. . .
CHUNK[K]: [K-th chunk]
Figure 11: Prompt for listwise discourse relation inference. The relation definitions are provided in Figure 9.

Prompt for Discourse-Driven Planning
You are an expert in discourse-aware text generation. Your task is to produce a discourse-aware
plan — a natural language paragraph that outlines how the final answer should be organized.
Inputs:
1. The user query.
2. Retrieved text chunks.
3. Intra-chunk RST trees, capturing local rhetorical hierarchies.
4. Inter-chunk rhetorical graph, modeling cross-passage discourse flow.
Critical instructions:
1. The plan must be written as a continuous paragraph in natural language.
2. The plan should describe the intended organization of the final answer.
3. The plan must be dynamically adapted to the given user query and evidence.
4. Avoid reproducing the content of the chunks; only outline how they will be used.
5. Output exactly one complete rhetorical plan.
Required output format:
PLAN:< one paragraph in natural language that describes the planned organization of the answer >
TEXT TO ANALYZE:{query, chunks, RST trees, rhetorical graph}
Figure 12: Prompt for Discourse-Driven Planning.
Prompt for Full Context Generation
You are an expert in question answering and text generation. Your task is to answer the user query
using the provided full document as context.
Inputs:
1. The user query.
2. The full document.
Critical instructions:
1. The answer must directly address the user query.
2. Use the full document as the only source of factual claims.
3. If the document does not support a claim, do not add it.
4. Write a coherent answer without copying long spans verbatim from the document.
Required output format:
ANSWER<one paragraph or multiple paragraphs in natural language>
TEXT TO ANALYZE{query, document}
Figure 13: Prompt for full context generation used in our baseline.

Prompt for Standard RAG
You are an expert in retrieval-augmented generation. Your task is to answer the user query using
only the retrieved text chunks as evidence.
Inputs:
1. The user query.
2. Retrieved text chunks.
Critical instructions:
1. The answer must directly address the user query.
2. Use the retrieved chunks as the only source of factual claims.
3. If the retrieved chunks do not support a claim, do not add it.
4. Write a coherent answer without copying long spans verbatim from the chunks.
Required output format:
ANSWER<one paragraph or multiple paragraphs in natural language>
TEXT TO ANALYZE{query, chunks}
Figure 14: Prompt for standard RAG used in our baseline.
Prompt for Retrieve-and-Plan Baseline
You are an expert in retrieval-augmented generation. Your task is to answer the user query by
first writing a short plan and then writing the final answer using only the retrieved text chunks as
evidence.
Inputs:
1. The user query.
2. Retrieved text chunks.
Critical instructions:
1. Write the plan as a single continuous paragraph that outlines the structure of the answer.
2. The plan must be grounded in what is supported by the retrieved chunks.
3. The answer must directly address the user query and use the retrieved chunks as the only source
of factual claims.
4. If the retrieved chunks do not support a claim, do not add it.
5. Write a coherent answer without copying long spans verbatim from the chunks.
Required output format:
PLAN<one paragraph plan>
ANSWER<one paragraph or multiple paragraphs in natural language>
TEXT TO ANALYZE{query, chunks}
Figure 15: Prompt for the retrieve-and-plan baseline used in our ablation study.

Prompt for Plan-and-Retrieve Baseline
You are an expert in retrieval-augmented generation. Your task is to support a plan-guided retrieval
procedure and then answer the user query.
Stage 1
Given only the user query, write a short plan and a retrieval hint that summarizes what evidence
should be retrieved.
Stage 2
After retrieving all text chunks using the retrieval hint, write the final answer using only the
retrieved text chunks as evidence.
Inputs:
1. The user query.
2. Retrieved text chunks returned after plan-guided retrieval.
Critical instructions:
1. Write the plan as a single continuous paragraph that outlines the structure of the answer.
2. The retrieval hint must be a list of retrieval queries that helps retrieve evidence aligned with the
plan.
3. The answer must directly address the user query and use the retrieved chunks as the only source
of factual claims.
4. If the retrieved chunks do not support a claim, do not add it.
5. Write a coherent answer without copying long spans verbatim from the chunks.
Required output format:
PLAN<one paragraph plan>
RETRIEVAL HINT<a list of retrieval queries>
ANSWER<one paragraph or multiple paragraphs in natural language>
TEXT TO ANALYZE{query, chunks}
Figure 16: Prompt for the plan-and-retrieve baseline used in our ablation study.

Prompt for Shallow Discourse Marker Inference
You are an expert in discourse analysis. Your task is to infer explicit discourse markers jointly
among a list of retrieved text chunks. In each call to this prompt, you are given the entire set of
chunks, and you must output a marker decision for every ordered pair of distinct chunks.
Task objective:
Given a list of chunks CHUNK[1] ,CHUNK[2] , . . . , CHUNK[K] , determine for every ordered pair (i, j)
withi̸=j whether there exists an explicit discourse marker from a marker list that indicates a
meaningful rhetorical connection fromCHUNK[i]toCHUNK[j]. If no marker is supported, output
NONE.
Discourse marker list:
however,but,although,in contrast,therefore,because,as a result,meanwhile,moreover,
furthermore,for example,for instance,in addition
Critical instructions:
1. For each ordered pair (i, j) withi̸=j , treat CHUNK[i] as the source and CHUNK[j] as the target.
2. Consider only explicit connectives that are supported by the two chunks. Do not infer implicit
relations.
3. Output exactly one marker from the marker list if a marker is applicable; otherwise output NONE .
4. Output a decision for every ordered pair withi̸=j.
Required output format:
For each ordered pair(i, j)withi̸=j, output one line in the following format:
CHUNK[i] -> CHUNK[j]: {MARKER}
TEXT TO ANALYZE:
CHUNK[1]: [first chunk]
CHUNK[2]: [second chunk]
. . .
CHUNK[K]: [K-th chunk]
Figure 17: Prompt for discourse marker inference used in the shallow discourse marker baseline.

Prompt for Discourse-Guided RAG
You are an expert in retrieval-augmented generation with discourse knowledge. Your task is to
generate a coherent and faithful answer by leveraging the following inputs:
Inputs:
1. The user query.
2. Retrieved text chunks.
3. Intra-chunk RST trees, capturing local rhetorical hierarchies.
4. Inter-chunk rhetorical graph, modeling cross-passage discourse flow.
5. A discourse-aware plan that outlines the intended argumentative organization.
Critical instructions:
1. The answer must directly address the user query.
2. Integrate evidence from multiple chunks, guided by their RST trees and rhetorical graph.
3. Follow the discourse-aware plan for structuring the answer.
4. Maintain factual accuracy, logical coherence, and rhetorical clarity.
5. Output a continuous answer in natural language.
Required output format:
ANSWER:< a single coherent paragraph or multi-paragraph answer grounded in discourse
structures>
Validation requirements:
- The answer must be faithful to the retrieved content.
- The answer must be logically organized and reflect discourse-level coherence.
- Avoid verbatim repetition of chunks; instead, synthesize and integrate them.
- Output exactly one complete answer.
TEXT TO ANALYZE:{query, chunks, RST trees, rhetorical graph, discourse-aware plan}
Figure 18: Prompt for Discourse-Guided RAG.

Human Evaluation Guidelines
Prerequisites:Eligibility for this evaluation requires simultaneous fulfillment of two conditions:
being a Master’s or Ph.D. student in Computer Science or a closely related field, and demonstrating
advanced proficiency in English sufficient to read and assess scientific news articles. Participants
are compensated at the standard hourly rate and are asked to confirm that they meet these criteria
before taking part in the task.
Instructions:For each selected sample, annotators are given the source document together with
four anonymized summaries, and the system identities are hidden and the order is randomized for
every instance. Raters are instructed to first read the source document carefully and then evaluate
each summary independently using a three-point Likert scale along four criteria:Relevance,
Simplicity,Conciseness, andFaithfulness.
Evaluation Criteria:Below, we provide a detailed explanation of the four criteria used in our
human evaluation. Raters are asked to consider each criterion separately and to base their scores
only on the information that is explicitly supported by the source document.
•RelevanceThis criterion assesses how well the summary covers the main topics, events, and
findings discussed in the source document. A highly relevant summary focuses on central points
and avoids spending space on marginal or tangential details.
•SimplicityThis criterion measures how easy the summary is to read and understand. A simple
summary uses clear and precise language, maintains a coherent structure, and avoids unnecessary
jargon or convoluted phrasing that could hinder comprehension.
•ConcisenessThis criterion evaluates whether the summary is compact while still conveying the
essential content. A concise summary avoids repetition and digression, omits minor details that
are not needed for understanding, and does not exceed the length required to communicate the
core message.
•FaithfulnessThis criterion judges whether the summary is supported by the source document
and free of hallucinations. A faithful summary does not introduce claims that contradict the
source, does not exaggerate or overgeneralize findings, and does not omit critical qualifications
that change the meaning of the original text.
Rating System:For each criterion, raters assign an integer score from 1 to 3, where 1 indicates
low quality, 2 indicates acceptable quality, and 3 indicates high quality. Scores should be given
solely based on the source document and the summary, without using AI tools to assist in judgment.
Annotators may consult trusted external resources, such as textbooks or scientific encyclopedias,
only when they need to clarify terminology.
Figure 19: Guidelines presented to human raters for the SciNews dataset evaluation.