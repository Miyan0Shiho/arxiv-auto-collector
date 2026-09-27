# Asymmetric Dynamic Routing: Balancing Reasoning Depth and Computational Efficiency in Hypergraph RAG

**Authors**: Qi Sun, Yijia Zhang, Xingliang Hou, Caibo Li, Qiang Li, Yu Guo

**Published**: 2026-09-24 09:17:40

**PDF URL**: [https://arxiv.org/pdf/2609.29282v1](https://arxiv.org/pdf/2609.29282v1)

## Abstract
While graph-based and hypergraph-based Retrieval-Augmented Generation (RAG) significantly mitigate hallucinations in Large Language Models (LLMs), existing structure-based RAG systems typically adopt static traversal strategies regardless of the query complexity. We identify this ``static retrieval fallacy'' as a primary source of computational redundancy for simple queries and cognitive context gaps for complex reasoning tasks. To balance reasoning quality and inference efficiency, we propose Asymmetric Dynamic Routing (ADR), an intent-conditioned retrieval framework operating over hierarchical knowledge graphs. ADR employs a lightweight structured classifier to dynamically dispatch queries among three asymmetric topological traversal operators: localized fact anchoring, bottom-up adjacency diffusion, and top-down insight grounding, which collectively enable bidirectional information flow across hierarchical knowledge layers. Extensive empirical evaluations across five domain-specific corpora demonstrate that ADR maintains strong reasoning performance while reducing prompt token consumption by up to 48.7\% and end-to-end query latency by 45.3\%, yielding a favorable quality--efficiency trade-off for query-adaptive Hypergraph RAG.

## Full Text


<!-- PDF content starts -->

ASYMMETRIC DYNAMIC ROUTING: BALANCING REASONING DEPTH AND
COMPUTATIONAL EFFICIENCY IN HYPERGRAPH RAG
Qi Sun1*, Yijia Zhang2*, Xingliang Hou1, Caibo Li2, Qiang Li3, Yu Guo1†
1School of Software Engineering, Xi’an Jiaotong University, Xi’an, Shaanxi, China
2State Key Laboratory of Human-Machine Hybrid Augmented Intelligence,
and Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, Xi’an, Shaanxi, China
3EHV Power Transmission Company of China Southern Power Grid Co., Ltd
ABSTRACT
While graph-based and hypergraph-based Retrieval-Augmented
Generation (RAG) significantly mitigate hallucinations in Large
Language Models (LLMs), existing structure-based RAG systems
typically adopt static traversal strategies regardless of the query
complexity. We identify this “static retrieval fallacy” as a pri-
mary source of computational redundancy for simple queries and
cognitive context gaps for complex reasoning tasks. To balance
reasoning quality and inference efficiency, we propose Asymmetric
Dynamic Routing (ADR), an intent-conditioned retrieval frame-
work operating over hierarchical knowledge graphs. ADR employs
a lightweight structured classifier to dynamically dispatch queries
among three asymmetric topological traversal operators: localized
fact anchoring, bottom-up adjacency diffusion, and top-down insight
grounding, which collectively enable bidirectional information flow
across hierarchical knowledge layers. Extensive empirical evalu-
ations across five domain-specific corpora demonstrate that ADR
maintains strong reasoning performance while reducing prompt to-
ken consumption by up to 48.7% and end-to-end query latency by
45.3%, yielding a favorable quality–efficiency trade-off for query-
adaptive Hypergraph RAG.
Index Terms—Adaptive information retrieval, dynamic query
routing, hypergraph diffusion
1. INTRODUCTION
Retrieval-Augmented Generation (RAG) provides non-parametric
memory for Large Language Models (LLMs), grounding generated
responses in verifiable external corpora to suppress factual hallu-
cinations [1, 2, 3]. To overcome the semantic fragmentation and
logical disconnection inherent in traditional dense vector similarity
search, recent approaches have shifted toward structured knowledge
representations. Frameworks such as GraphRAG [4], LightRAG [5],
and Hyper-RAG [6] organize domain knowledge into topological
graphs and hypergraphs. By modeling explicit entity dependencies
and higher-order semantic associations, these structured representa-
tions provide LLMs with multi-hop deductive pathways, enabling
comprehensive synthesis across disparate documents [7, 8].
However, despite these topological advancements, existing stucture-
based RAG systems still rely on uniform configurations for subgraph
expansion radius and traversal depth [9], regardless of query com-
plexity. We define this limitation as theStatic Retrieval Fallacy,
which leads to two contrasting problems:
* Equal contribution. † Corresponding author: yu.guo@xjtu.edu.cn•Context Pollution via Over-Retrieval:For simple factual
queries, unnecessary multi-hop diffusion introduces topologi-
cally adjacent but semantically irrelevant information into the
context. This degrades the signal-to-noise ratio within the
LLM’s cross-attention layers, increases prefill memory con-
sumption, and may induce associative hallucinations [10].
•Cognitive Gaps via Under-Retrieval:Abstract, cross-
document inquiries require macro-level synthesis. Localized
topological expansion may fail to bridge disparate abstrac-
tion layers, resulting in fragmented evidence and superficial
reasoning rather than mechanistic explanation [11].
Iterative, reflection-driven methods such as Self-RAG [12], IR-
CoT [13], and CogRAG [14] introduce adaptability through multi-
turn agentic loops. However, they incur multi-second inference la-
tency overheads, limiting their applicability to high-throughput, real-
time deployment in speech and signal processing systems.
To resolve the tension between reasoning depth and computational
efficiency, we proposeAsymmetric Dynamic Routing (ADR), a
single-pass, intent-guided framework that dynamically adjusts the
retrieval entry level, traversal direction, and expansion scope over a
hierarchical knowledge substrate.
The main contributions of this work are threefold:
• We formalize theStatic Retrieval Fallacy, highlighting the
mismatch between uniform graph traversal and heteroge-
neous query complexity.
• We develop ADR as a query-conditioned framework for bidi-
rectional topological traversal, comprising three asymmetric
operators: localized fact anchoring, bottom-up adjacency dif-
fusion, and top-down insight grounding.
• Through evaluations across five distinct domains, we find that
traversal requirements vary across domains and demonstrate
that ADR maintains strong reasoning performance while sub-
stantially reducing prompt token consumption and latency.
2. RELATED WORK
Stucture-based RAG Architectures.The transition from flat
chunk retrieval to topological indexing represents a major shift
in RAG architectures. GraphRAG [4] uses LLM-extracted knowl-
edge graphs to generate hierarchical community summaries, Ligh-
tRAG [5] adopts dual-level retrieval, and GNN-RAG [15] applies
graph neural networks to retrieve relevant reasoning paths. Hyper-
RAG [6] further uses hypergraph representations to capture pairwise
arXiv:2609.29282v1  [cs.IR]  24 Sep 2026

and beyond-pairwise correlations. Despite these advances, their re-
trieval procedures remain largely predefined rather than conditioned
on query complexity, potentially causing unnecessary expansion for
simple queries and insufficient relational coverage for complex ones.
Adaptive and Agentic Retrieval Paradigms.Recent stud-
ies explore query-adaptive mechanisms to address static retrieval
inefficiencies. Adaptive-RAG [16] selects among no-retrieval,
single-step, and iterative retrieval according to query complexity;
Self-RAG [12] uses learned reflection tokens to control retrieval;
ARAG [17] triggers retrieval based on low-confidence predictions;
and CRAG [18] evaluates retrieval quality to initiate corrective
actions. While powerful, these agentic and iterative loops may
propagate generation errors and introduce significant temporal la-
tency. Moreover, existing adaptive methods primarily vary retrieval
frequency or reasoning depth, without explicitly adapting the topo-
logical traversal to heterogeneous query requirements.
3. PROPOSED METHODOLOGY
As shown in Fig. 1, incoming queries are classified into three cate-
gories based on their retrieval and reasoning requirements:
•Simple queries:Trigger localized anchoring within the fac-
tual hypergraphH K.
•Moderate queries:Execute bottom-up diffusion into the
deep insight hypergraphH D.
•Complex queries:Initiate top-down semantic matching
overH D, followed by a inverse cross-layer mappingπ−1to
ground the high-level insights in verifiable supporting factual
evidence withinH K.
3.1. Hierarchical Dual-Hypergraph Representation
ADR operates on the dual-hypergraph knowledge framework intro-
duced in DHI [19] and follows its original construction procedure.
Given a document corpus, the knowledge space is organized asG=
(HK, HD), whereH K= (V K, EK)preserves fine-grained enti-
ties and factual relations directly grounded in the source text, while
HD= (V D, ED)organizes higher-level abstract structures and their
semantic associations.
InH K,VKrepresents factual entities, whileE Kcontains pair-
wise factual relations (|e|= 2) and higher-order factual hyperedges
(|e| ≥3). The latter capture joint relations among multiple entities
and provide the structural basis for higher-level abstraction. Each
higher-order factual hyperedge is associated with an atomic direct in-
sightι(e)and mapped byπto a corresponding insight vertex inV D,
whereas pairwise edges remain withinH K. Each deep insight edge
ε∈E Dconnects a set of insight vertices and carries a synthesized
narrative insightI(ε). ADR uses this cross-layer correspondence to
adapt the retrieval level and traversal direction to query complexity.
3.2. Intent Routing and Conservative Fallback
ADR categorizes an incoming queryqinto three query categories
according to its retrieval and reasoning requirements, withY=
{1,2,3}:
•Simple (y= 1):Targets distinct entities, definitions, or direct
factual relations that can be resolved through localized factual
retrieval.
•Moderate (y= 2):Requires relational or procedural rea-
soning across multiple facts, with localized evidence supple-
mented by broader contextual associations.•Complex (y= 3):Requires high-level thematic synthesis,
cross-domain comparison, or causal reasoning based on ab-
stract semantic structures and supporting factual evidence.
ADR employs an LLM-based intent classifierf ϕto output a pre-
dicted categoryy∈ Yand confidencec y∈[0,1]. To handle low-
confidence predictions, ADR adopts a Conservative Fallback Princi-
ple: the prediction is accepted whenc y≥γ; otherwise, the query is
redirected to Moderate, as defined in Eq. (1):
ˆy=yI(c y≥γ) + 2I(c y< γ),(1)
whereˆydenotes the final routing decision andγ= 0.75by default.
Moderate is selected as the fallback because it provides an interme-
diate retrieval scope, reducing the risk of routing uncertain queries
to either extreme while maintaining bounded retrieval cost.
3.3. Asymmetric Bidirectional Traversal Operators
3.3.1. Operator 1: Localized Fact Anchoring (ˆy= 1)
For Simple queries, ADR performs localized retrieval within the fac-
tual hypergraphH K, as the required evidence can typically be ob-
tained directly from relevant entities and factual relations. Given a
queryq, ADR identifies thek emost relevant factual entities to form
the entity anchor setV easin Eq. (2):
Veas= arg max
V′⊂VK,|V′|=keX
v∈V′cos(e v,q),(2)
wheree vdenotes the embedding of factual entityv. ADR then re-
trieves the factual edges directly associated with these anchors using
Eq. (3):
Esub={e∈E K|e∩V eas̸=∅}.(3)
Retrieval terminates atE subwithout further neighborhood expan-
sion or cross-layer traversal. The granular factual content associated
withE subis then formatted as the retrieval context for final gen-
eration. This restricted expansion provides an effect analogous to
zero-hop retrieval while avoiding unnecessary context expansion.
3.3.2. Operator 2: Bottom-Up Adjacency Diffusion (ˆy= 2)
For Moderate queries, directly anchored factual evidence may be in-
sufficient to capture the broader relational context required for rea-
soning. ADR therefore extends retrieval upward fromH KtoH D.
Sinceπis defined only for higher-order factual hyperedges, those
inE subare mapped to their corresponding insight vertices using
Eq. (4):
Useed={π(e)|e∈E sub,|e| ≥3}.(4)
Starting fromU seed, ADR performs a single adjacency expansion
inH Dto retrieve the associated deep insight edges in Eq. (5):
Eup={ε∈E D|ε∩U seed̸=∅}.(5)
The granular factual content associated withE suband the syn-
thesized narrative insights associated withE upare then jointly for-
matted as the retrieval context. This bottom-up traversal preserves
factual grounding while extending the context to broader semantic
associations required by Moderate queries.

LLM
 User 
QueriesSimpleWhat is the relationship 
between A and B?
ModerateHow did the X events 
involving A, B, and C 
unfold together?Query 
Entities
Query Entities
Query Relations
Query 
Deep InsightsQuery Insights
LLM
Mode Analysis Bilateral DiffusionAnswer: By 
integrating and 
analyzing the 
relationships 
between entities, 
as well as the 
comprehensive 
insight into the 
knowledge chain 
and the central 
aggregation 
insight, we can 
know...
ComplexWhat underlying 
pattern explains the 
recurring conflicts 
between these groups?Fig. 1. The architectural blueprint of Asymmetric Dynamic Routing (ADR).
Table 1. Statistical information of the datasets.
Statistics Mix CS Agri. Neuro. Patho.
Documents 61 10 12 1 1
Chunks 560 1,992 1,813 1,790 824
Tokens 615,355 2,190,803 1,993,515 1,968,716 905,760
3.3.3. Operator 3: Top-Down Insight Grounding (ˆy= 3)
For Complex queries, localized factual anchoring may fail to capture
the high-level semantic structures required for thematic synthesis,
comparison, or causal reasoning. ADR therefore reverses the traver-
sal direction and starts fromH D, retrieving thek ddeep insight edges
most relevant to the query using Eq. (6):
E∗= arg max
E′⊂ED,|E′|=kdX
ε∈E′cos(e ε,q),(6)
wheree εdenotes the embedding of deep insight edgeε. The re-
trieved insight structures are then grounded in their supporting fac-
tual evidence through the inverse cross-layer mapping in Eq. (7):
Eground =[
ε∈E∗{π−1(u)|u∈ε} ⊆E K.(7)
The synthesized narrative insights associated withE∗and the
granular factual content associated withE ground are jointly format-
ted as the retrieval context. This top-down traversal combines high-
level semantic reasoning with supporting factual evidence fromH K.
4. EXPERIMENTAL EV ALUATION
4.1. Experimental Configuration
Datasets.We evaluate ADR on five datasets: Mix, CS, and Agri-
culture from UltraDomain [20], and Neurology and Pathology from
MIRAGE [21]. Their statistics are summarized in Table 1. Mix
represents a cross-domain sparse setting; CS and Agriculture repre-
sent intra-domain sparse settings with relatively weak inter-passagedependencies; and Neurology and Pathology represent intra-domain
dense settings with stronger semantic continuity and local relations.
Baselines and Protocols.We compare ADR with six represen-
tative baselines covering non-retrieval, conventional RAG, graph-
based RAG, and hypergraph-based RAG methods:
•LLM (Zero-shot):Directly generates responses without ex-
ternal knowledge retrieval.
•NaiveRAG [1]:Retrieves relevant text chunks through dense
vector similarity search.
•GraphRAG [4]:Organizes knowledge into entity-relation
graphs and supports global retrieval through hierarchical
community summaries.
•LightRAG [5]:Employs a dual-level retrieval framework
that integrates entity-level and relation-level information.
•Hyper-RAG [6]:Uses hypergraph structures to model
higher-order relations among multiple entities.
•DHI [19]:Constructs coupled factual and deep insight hy-
pergraphs and performs cross-layer retrieval to integrate fine-
grained facts with higher-level insights.
All methods use GPT-4o-mini as the generation backbone and
text-embedding-3-smallfor representation, with tempera-
tureT= 0for deterministic evaluation. We adopt a normalized
LLM-as-a-Judge protocol (0–100) across five dimensions: Compre-
hensiveness, Diversity, Empowerment, Logicality, and Readability.
4.2. Overall Performance Comparison
Table 2 compares ADR with six baseline methods across the five
datasets. ADR achieves consistently strong performance across
different knowledge settings. In particular, ADR obtains scores of
83.91 on Mix and 86.65 on Pathology, compared with 83.18 and
85.78 for DHI, respectively. Similar improvements are observed
on CS, Agriculture, and Neurology, indicating that query-adaptive
traversal can support heterogeneous retrieval requirements while
maintaining stable performance across domains.

Table 2. Overall performance comparison with baseline methods
across five datasets (0–100 scale, higher is better).
Method Mix CS Agri. Neuro. Patho.
LLM (Zero-shot) 79.30 81.08 79.64 81.15 82.81
NaiveRAG [1] 78.09 79.43 76.20 79.20 82.04
GraphRAG [4] 81.06 84.03 79.98 83.10 82.72
LightRAG [5] 81.01 81.25 79.05 81.82 84.43
Hyper-RAG [6] 80.39 83.88 81.98 83.74 84.41
DHI [19] 83.18 84.66 82.97 84.23 85.78
ADR (Ours) 83.91 85.24 83.86 85.02 86.65
Table 3. Ablation study of different routing strategies across five
datasets.
Routing Category Mix CS Agri. Neuro. Patho.
Frozen: Simple-only 82.01 82.09 81.89 83.41 83.40
Frozen: Moderate-only 82.52 82.14 82.70 84.08 85.40
Frozen: Complex-only 82.05 83.13 81.31 83.76 83.89
ADR (Dynamic) 83.91 85.24 83.86 85.02 86.65
∆gain over best static+1.39 +2.11 +1.16 +0.94 +1.25
4.3. Dynamic Routing Analysis and Computational Efficiency
To examine the effectiveness of dynamic routing, we compare ADR
with three fixed traversal modes: Simple-only, Moderate-only, and
Complex-only. Each fixed configuration applies the same retrieval
strategy to all queries, whereas ADR selects the retrieval path ac-
cording to query complexity.
As shown in Table 3, ADR outperforms the best fixed configura-
tion by 1.39, 2.11, 1.16, 0.94, and 1.25 points on Mix, CS, Agricul-
ture, Neurology, and Pathology, respectively. The relative effective-
ness of fixed traversal modes also varies across datasets with differ-
ent knowledge characteristics. On CS, Complex-only achieves 83.13
compared with 82.14 for Moderate-only, suggesting that queries in
this sparse setting may benefit from deeper retrieval. In contrast,
Moderate-only outperforms Complex-only by 1.39 and 1.51 points
on Agriculture and Pathology, respectively, indicating that additional
traversal depth does not always improve response quality. These
results show that no fixed traversal mode is consistently optimal
across different knowledge settings. This adaptive mechanism en-
ables ADR to preserve localized evidence for simpler queries while
invoking deeper cross-layer retrieval when additional reasoning con-
text is required.
ADR also improves computational efficiency. As shown in Ta-
ble 4, ADR reduces average latency from 435 to 238 ms and prompt
tokens from 3,760 to 1,929 relative to Complex-only, corresponding
to reductions of 45.3% and 48.7%, respectively, while improving the
overall score from 82.05 to 83.91.
Table 4. Computational efficiency profiling and resource parsimony.
Routing Category Latency Tokens Score
Fixed Complex-only 435 ms 3,760 82.05
ADR (Dynamic) 238 ms 1,929 83.91
∆Savings-45.3% -48.7% +1.864.4. Sensitivity Analysis of the Routing Threshold
Whenγis too low, ambiguous queries with relatively low confi-
dence may still be accepted as reliable routing decisions rather than
triggering the conservative fallback. Consequently, they may be as-
signed to inappropriate retrieval trajectories with either insufficient
or excessive retrieval depth, resulting in missing evidence or unnec-
essary context. For example, on CS, reducingγfrom 0.75 to 0.50
decreases the Logicality score by 1.14 points. Conversely, an overly
high threshold (γ≥0.85) causes 68.4% of queries to fall back to
the Moderate trajectory, increasing average latency by 58 ms with-
out significant performance gains (p >0.05). Overall,γ= 0.75
provides a favorable balance between routing reliability and compu-
tational efficiency and is used as the default setting.
4.5. Qualitative Analysis of Reasoning Pathways
We further examine a complex causal query from the Pathology
corpus asking how somatic IDH1 mutations promote gliomagene-
sis through epigenetic remodeling. Simple-only retrieves localized
facts about IDH1 and the R132H mutation but fails to capture the
further causal chain, whereas Complex-only retrieves 3,850 tokens
of tangential biochemical context, including irrelevant associations
such as EGFR amplification.
ADR classifies the query as Complex (c y= 0.94). Starting from
the deep insight hypergraphH D, Top-Down Insight Grounding iden-
tifies the causal mechanism: mutant IDH1 drives 2-hydroxyglutarate
(2-HG) accumulation, which inhibitsα-KG-dependent dioxyge-
nases and induces further epigenetic remodeling. ADR then applies
π−1to ground this mechanism in supporting facts fromH K, in-
cluding the IDH1 R132H mutation and its altered metabolic activity.
This cross-layer retrieval preserves the causal progression from
mutation to metabolic disruption and epigenetic remodeling while
filtering unrelated associations, yielding a more causally coherent
response with fewer hallucinated associations and 41.2% fewer
prompt tokens than Complex-only.
5. DISCUSSION AND LIMITATIONS
While ADR balances retrieval efficiency and reasoning fidelity, sev-
eral limitations remain. First, Intent Routing operates independently
of the real-time graph structure. For a Complex query in a struc-
turally sparse domain, insufficient graph connectivity may limit the
supporting factual evidence retrieved by Top-Down Insight Ground-
ing, requiring a fallback to dense vector search. Second, constructing
the dual-hypergraph framework(H K, HD)incurs upfront indexing
costs, reflecting a trade-off between offline construction and online
inference efficiency.
6. CONCLUSION
In this paper, we introduced Asymmetric Dynamic Routing (ADR)
to address the “static retrieval fallacy” in stucture-based RAG Ar-
chitectures. ADR adapts retrieval depth and traversal direction to
query complexity through three asymmetric bidirectional traversal
operators: localized fact anchoring, bottom-up adjacency diffusion,
and top-down insight grounding. Experiments across five domains
demonstrate that query-adaptive routing maintains strong reasoning
performance while reducing unnecessary traversal and context over-
head, with latency and prompt token consumption reduced by 45.3%
and 48.7%, respectively. Future work will explore continuous soft
routing over dynamic multi-modal hypergraphs.

7. REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al.,
“Retrieval-augmented generation for knowledge-intensive nlp
tasks,”Advances in neural information processing systems,
vol. 33, pp. 9459–9474, 2020.
[2] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, M. Wang, and H. Wang, “Retrieval-augmented gen-
eration for large language models: A survey,”arXiv preprint
arXiv:2312.10997, 2023.
[3] G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni,
T. Schick, J. Dwivedi-Yu, A. Joulin, S. Riedel, and E. Grave,
“Atlas: Few-shot learning with retrieval augmented language
models,”Journal of Machine Learning Research, vol. 24, no.
251, pp. 1–43, 2023.
[4] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody,
S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson, “From
local to global: A graph rag approach to query-focused sum-
marization,”arXiv preprint arXiv:2404.16130, 2024.
[5] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “Lightrag: Sim-
ple and fast retrieval-augmented generation.” inEMNLP (Find-
ings), 2025, pp. 10 746–10 761.
[6] Y . Feng, H. Hu, S. Ying, X. Hou, S. Liu, M. Yang, J. Li, S. Du,
N. Zheng, H. Huet al., “Hyper-rag: Combating llm halluci-
nations using hypergraph-driven retrieval-augmented genera-
tion,”Nature Communications, 2026.
[7] M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec,
“Qa-gnn: Reasoning with language models and knowledge
graphs for question answering,” inProceedings of the 2021
conference of the North American chapter of the association
for computational linguistics: human language technologies,
2021, pp. 535–546.
[8] X. He, Y . Tian, Y . Sun, N. V . Chawla, T. Laurent, Y . LeCun,
X. Bresson, and B. Hooi, “G-retriever: Retrieval-augmented
generation for textual graph understanding and question an-
swering,”Advances in Neural Information Processing Systems,
vol. 37, pp. 132 876–132 907, 2024.
[9] K. H. Lau, F. Zhang, B. Ruan, Y . Zhou, Q. Guo, R. Zhang, and
X. Zhou, “Breaking the static graph: Context-aware traversal
for graph-based rag,” inFindings of the Association for Com-
putational Linguistics: ACL 2026, 2026, pp. 5849–5863.
[10] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis,
L. Zettlemoyer, and W.-t. Yih, “Replug: Retrieval-augmented
black-box language models,” inProceedings of the 2024 con-
ference of the north american chapter of the association for
computational linguistics: Human language technologies (vol-
ume 1: Long papers), 2024, pp. 8371–8384.
[11] B. Jin, C. Xie, J. Zhang, K. K. Roy, Y . Zhang, Z. Li, R. Li,
X. Tang, S. Wang, Y . Menget al., “Graph chain-of-thought:
Augmenting large language models by reasoning on graphs,”
inFindings of the Association for Computational Linguistics:
ACL 2024, 2024, pp. 163–184.
[12] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-
rag: Learning to retrieve, generate, and critique through self-
reflection,” inInternational conference on learning represen-
tations, vol. 2024, 2024, pp. 9112–9141.[13] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabhar-
wal, “Interleaving retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions,” inProceedings of
the 61st annual meeting of the association for computational
linguistics (volume 1: long papers), 2023, pp. 10 014–10 037.
[14] H. Hu, Y . Feng, R. Li, R. Xue, X. Hou, Z. Tian, Y . Gao,
and S. Du, “Cog-rag: cognitive-inspired dual-hypergraph with
theme alignment retrieval-augmented generation,” inProceed-
ings of the AAAI Conference on Artificial Intelligence, vol. 40,
no. 37, 2026, pp. 31 032–31 040.
[15] C. Mavromatis and G. Karypis, “Gnn-rag: Graph neural re-
trieval for efficient large language model reasoning on knowl-
edge graphs,” inFindings of the Association for Computational
Linguistics: ACL 2025, 2025, pp. 16 682–16 699.
[16] S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. C. Park,
“Adaptive-rag: Learning to adapt retrieval-augmented large
language models through question complexity,” inProceedings
of the 2024 conference of the north american chapter of the as-
sociation for computational linguistics: Human language tech-
nologies (volume 1: Long papers), 2024, pp. 7036–7050.
[17] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu,
Y . Yang, J. Callan, and G. Neubig, “Active retrieval augmented
generation,” inProceedings of the 2023 conference on empir-
ical methods in natural language processing, 2023, pp. 7969–
7992.
[18] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling, “Cor-
rective retrieval augmented generation,”arXiv preprint
arXiv:2401.15884, 2024.
[19] Q. Sun, X. Hou, C. Li, Y . Zhang, Q. Li, and Y . Guo,
“Dual-hypergraph indexing: Bridging knowledge islands for
multi-hop reasoning in retrieval-augmented generation,” 2026.
[Online]. Available: https://arxiv.org/abs/2609.28108
[20] H. Qian, Z. Liu, P. Zhang, K. Mao, D. Lian, Z. Dou, and
T. Huang, “Memorag: Boosting long context processing with
global memory-enhanced retrieval augmentation,” inProceed-
ings of the ACM on Web Conference 2025, 2025, pp. 2366–
2377.
[21] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking
retrieval-augmented generation for medicine,” inFindings of
the Association for Computational Linguistics: ACL 2024,
2024, pp. 6233–6251.