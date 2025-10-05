# Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement

**Authors**: Zhaoyan Wang, Zheng Gao, Arogya Kharel, In-Young Ko

**Published**: 2025-10-02 11:30:51

**PDF URL**: [http://arxiv.org/pdf/2510.01910v1](http://arxiv.org/pdf/2510.01910v1)

## Abstract
Graph Neural Networks (GNNs) are widely adopted in Web-related applications,
serving as a core technique for learning from graph-structured data, such as
text-attributed graphs. Yet in real-world scenarios, such graphs exhibit
deficiencies that substantially undermine GNN performance. While prior
GNN-based augmentation studies have explored robustness against individual
imperfections, a systematic understanding of how graph-native and Large
Language Models (LLMs) enhanced methods behave under compound deficiencies is
still missing. Specifically, there has been no comprehensive investigation
comparing conventional approaches and recent LLM-on-graph frameworks, leaving
their merits unclear. To fill this gap, we conduct the first empirical study
that benchmarks these two lines of methods across diverse graph deficiencies,
revealing overlooked vulnerabilities and challenging the assumption that LLM
augmentation is consistently superior. Building on empirical findings, we
propose Robust Graph Learning via Retrieval-Augmented Contrastive Refinement
(RoGRAD) framework. Unlike prior one-shot LLM-as-Enhancer designs, RoGRAD is
the first iterative paradigm that leverages Retrieval-Augmented Generation
(RAG) to inject retrieval-grounded augmentations by supplying class-consistent,
diverse augmentations and enforcing discriminative representations through
iterative graph contrastive learning. It transforms LLM augmentation for graphs
from static signal injection into dynamic refinement. Extensive experiments
demonstrate RoGRAD's superiority over both conventional GNN- and LLM-enhanced
baselines, achieving up to 82.43% average improvement.

## Full Text


<!-- PDF content starts -->

Are LLMs Better GNN Helpers? Rethinking Robust Graph
Learning under Deficiencies with Iterative Refinement
Zhaoyan Wang∗
School of Computing
KAIST
Daejeon, Republic of Korea
zhaoyan123@kaist.ac.krZheng Gao
School of Computer Science and Engineering
UNSW
Sydney, NSW, Australia
zheng.gao1@unsw.edu.au
Arogya Kharel
School of Computing
KAIST
Daejeon, Republic of Korea
akharel@kaist.ac.krIn-Young Ko
School of Computing
KAIST
Daejeon, Republic of Korea
iko@kaist.ac.kr
Abstract
Graph Neural Networks (GNNs) are widely adopted in Web-related
applications, serving as a core technique for learning from graph-
structured data, such as text-attributed graphs. Yet in real-world
scenarios, such graphs exhibit deficiencies that substantially un-
dermine GNN performance. While prior GNN-based augmentation
studies have explored robustness against individual imperfections,
a systematic understanding of how graph-native and Large Lan-
guage Models (LLMs) enhanced methods behave under compound
deficiencies is still missing. Specifically, there has been no com-
prehensive investigation comparing conventional approaches and
recent LLM-on-graph frameworks, leaving their merits unclear.
To fill this gap, we conduct the first empirical study that bench-
marks these two lines of methods across diverse graph deficiencies,
revealing overlooked vulnerabilities and challenging the assump-
tion that LLM augmentation is consistently superior. Building on
empirical findings, we propose Robust Graph Learning via Retrieval-
Augmented Contrastive Refinement (RoGRAD) framework. Unlike
prior one-shot LLM-as-Enhancer designs, RoGRAD is the first it-
erative paradigm that leverages Retrieval-Augmented Generation
(RAG) to inject retrieval-grounded augmentations by supplying
class-consistent, diverse augmentations and enforcing discrimina-
tive representations through iterative graph contrastive learning. It
transforms LLM augmentation for graphs from static signal injec-
tion into dynamic refinement. Extensive experiments demonstrate
RoGRAD’s superiority over both conventional GNN- and LLM-
enhanced baselines, achieving up to 82.43% average improvement.
1 Introduction
Graph Neural Networks (GNNs) have become a cornerstone for an-
alyzing various types of Web graphs with applications ranging from
social and traffic networks [ 43,50,56,61], to recommender sys-
tems [ 15,21,52]. Although GNNs achieve excellent results on stan-
dard benchmark datasets, their performance can degrade substan-
tially when faced with real-world graph data that contain inherent
imperfections [ 9,25]. This is because complete graphs in practice
suffer from data corruptions and loss, so that labels, structures, and
features are often scarce and incomplete, and nodes may be missing
∗Corresponding author.or isolated. Such deficiencies, referred to as ‘weak information’ [ 34],
are shown to degrade GNN performance markedly [44, 55].
Beyond solitary studies of such deficiencies, their coexistence is
pervasive in practical applications and poses challenges to the reli-
ability and robustness of GNN models. For instance, intact social
networks can develop deficiencies when certain social relation-
ships are intentionally hidden by users, and when user profiles or
attributes become unavailable due to privacy restrictions or account
deactivation.[ 28,30,64]. Such incompleteness hinders neighbor-
hood aggregation in GNNs and degrades task performance.
Many studies have sought to enhance GNN robustness when
graphs are noisy, incomplete, or weakly supervised. Prior works
refine graph structures to reduce incompleteness and noise [ 24,32,
60], design architectures for robust representations [ 65], or employ
training strategies [ 3,12] and supervision-efficient learning [ 10,49].
Although these efforts demonstrate progress, their effectiveness
under joint degraded conditions is insufficiently investigated.
Recent advances in LLMs endow LLMs with extensive knowl-
edge, proficiency in semantic understanding, and strong reasoning
capabilities [ 2,6,18]. These properties motivate the integration of
LLMs and graphs [ 7] either as encoders of node attributes [ 23,26,
58], enhancers providing auxiliary signals [ 17,59], or reasoners
reformulating graph tasks into text [5, 63].
However, a critical question remains:Can LLM-enhanced methods
truly deliver better supportive performance and stronger robustness
than traditional ones under real-world deficiencies?The compar-
ative merits of LLM-enhanced versus conventional approaches
under graph deficiencies remains unexplored. While LLMs bring
semantic priors and external knowledge, they incur substantial
computational overhead, distinct instability across runs, and hal-
lucinations [ 4,20]. Given the trade-offs, the question of whether
LLMs are indeed beneficial GNN helpers must be answered.
To address this gap, we empirically benchmark LLM-enhanced
and conventional graph-native GNN approaches across a spectrum
of graph deficiencies. Centering on the canonical node classifica-
tion task, we provide the first systematic evaluation of robustness
to compound imperfections, uncovering the performance dispar-
ities between LLM-enhanced and conventional GNN approaches,
offering new insights of incorporating LLMs on graphs.arXiv:2510.01910v1  [cs.LG]  2 Oct 2025

Zhaoyan Wang et al.
Our empirical analysis reveals:Under less severe deficiencies, LLM-
enhanced works trail behind non-LLM counterparts that are simpler
and more resource-efficient, we further uncover two key limitations
of existing LLM-enhanced frameworks. First, LLM generations suf-
fer from semantic homogeneity rather than providing informative
diversity (see Appendix Section A for details). This severely con-
strains GNNs to learn discriminative boundaries, impairing down-
stream tasks such as node classification and link prediction [ 13,35].
Generations across different classes tend to converge on similar
phrasing, blurring inter-class distinctions, while samples within
the same class are insufficiently coherent, providing heterogeneous
signals that fail to consolidate class prototypes. The lack of differ-
entiation weakens the discriminative capacity of encoded represen-
tations and reduces augmentation utility.
Second, existing LLM-as-Enhancer frameworks adopt a one-shot
paradigm, where augmentations are generated once. This static
pipeline fundamentally diminishes GNN performance and robust-
ness, as low-quality or homogeneous generations cannot be refined.
In contrast, we propose the first iterative refinement paradigm for
LLM-on-graph that represents a paradigm shift from one-shot en-
hancement to iterative refinement for LLM-on-graph frameworks.
To address both limitations, we proposeRoGRAD(Robust Graph
Learning via Retrieval-Augmented Contrastive Refinement), the
first iterative refinement framework for LLM-on-graph. RoGRAD
leverages a retrieval–diagnosis–revision mechanism to mitigate se-
mantic homogeneity and promote class-consistent diversity at the
LLM generation level. Through iterative sample generations, it en-
riches node features, supplies informative supervisory signals and
edges, and expands the limited node set alleviating weak informa-
tion. In parallel, we further introduceR2CL( Contrastive Learning
with RAGRefinement), a novel representation learning paradigm
with RAG that enforces semantic alignment and separation in the
embedding space through contrastive learning on LLM-perturbed
views. Together, these components produce more discriminative
and robust node representations from intact data to mitigate per-
formance degradation under compound graph deficiencies.
Our contributions can be summarized as follows:
•We present the first comprehensive exploration of LLM-on-
graph frameworks under varying graph deficiencies and
their comparison against non-LLM GNN-based counter-
parts. Our findings reveal that LLM-enhanced approaches,
despite their semantic richness, may underperform simpler
non-LLM ones under low-to-modest perturbations, chal-
lenging the prevailing assumptions about their superiority.
•To the best of our knowledge, RoGRAD is the first iterative
RAG framework for graph learning tasks that jointly op-
timizes LLM-generated content and node representations
through self-retrieval. It reinforces GNN robustness against
common graph deficiencies in real-world practice.
•At the representation level, we introduce R2CL, a novel
contrastive learning framework that leverages iterative
retrieval-guided, LLM-refined views to promote intra-class
representation consistency and inter-class discriminability.
•Extensive experiments on benchmark datasets validate the
consistent superiority of RoGRAD over selected baselines.2 Related Work
2.1 Graph Learning with Deficiencies
Over the years, various methods have been proposed to enhance
GNN performance under weak information. Relation-aware mod-
els [42,46] extend GNN architectures by modeling relations to
compensate for incomplete and noisy structures. ACM-GNN [ 36]
proposes an adaptive channel mixing strategy, which relieves ho-
mophilous edge sparsity. Structure refinement models such as
GRCN [ 14,57] mitigate fixed structures by dynamically revising
graph topologies, benefiting scenarios characterized by sparse and
spurious edges. While perturbation techniques such as DropEdge
or feature masking [ 38,40,45] are effective in improving general-
ization and solving overfitting, semi-supervised and few-shot meth-
ods [ 10,27,31] alleviate the problem of limited labels. DropMes-
sage [ 11] perturbs node-to-node communication for boosting ro-
bustness in message-passing. Likewise, feature propagation [ 41]
and imputation (e.g., Neighbor Mean) achieve strong performance
with missing features. RSGNN [ 8] tackles both edge and supervi-
sion deficiency by down-weighting noisy edges and densifying the
graph to exploit scarce labels. Besides, unsupervised methods such
as DGI [ 48] cope with label deficiency by learning without labels.
However, these approaches address specific deficiencies, and their
effectiveness under compound deficiencies remains underexplored.
2.2 LLM-on-graph
Emerging paradigms have proliferated through the integration of
LLMs and graphs. LLM-as-Encoder paradigm leverages pretrained
LLMs to transform textual attributes into semantically richer and
higher-dimensional embeddings, compared to shallow embeddings
(e.g., bag-of-words) [ 26,53,58], whereas LLM-as-Enhancer aug-
ments GNNs with LLM-generated contents. TAPE [ 17] distills tex-
tual explanations from GPT into compact embeddings that im-
prove node classification, while LLM4NG [ 59] generates labeled
nodes for augmenting graphs under label-limited few-shot settings.
Similarly, Chen et al. [ 7] show that textual features derived from
frozen LLMs enrich node representations under various training
paradigms. LLM-only reasoning frameworks, including LLM-as-
Predictor, bypass GNNs entirely and delegate reasoning to LLMs.
It’s important to note that, as LLM-only reasoning frameworks
such as GraphText [63] and GraphLLM [5] operate independently
of graph encoders, their performance is not directly comparable to
that of GNN-based approaches, lying outside our scope.
2.3 GNN Robustness with LLMs
To date, the potential of LLM-on-graph and investigations on GNN
robustness with LLMs are still underexplored. The only related
work, Zhang et al. [ 62], mainly examines whether LLM-as-Encoder
improves adversarial robustness. However, they focus on adver-
sarial robustness where sophisticated attackers launch crafted per-
turbations, while our work emphasizes graph defect robustness
commonly encountered in real-world applications, towards more
practical implications. Besides, their perturbation scope is narrowly
limited to structural adversarial attacks, with feature, node outage,
and supervision attacks unstudied. Most importantly, LLMs cannot

Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement
Table 1: Accuracy (%) vs. attack intensities. The darkest/lighter shadings highlight the best/second- and third-best performances.
Atk. Int. GCN ACM-GNN GRCN Nbr. Mean Feat. Prop. DropMsg DropEdge DGI RS-GNN TAPE TA_E LLM4NG MistralEmb QwenEmb
0.00 87.98 88.5485.66 88.17 88.17 87.65 87.61 82.88 63.59 84.49 83.36 88.06 87.87 87.69
0.33 87.06 87.39 86.32 86.36 86.36 87.4386.77 80.59 69.43 84.53 83.32 86.21 86.80 86.80
0.50 83.19 84.8181.70 83.37 83.58 82.23 83.68 75.17 63.44 81.74 80.59 84.07 85.27 85.56
0.66 84.44 84.55 84.18 85.63 85.6384.21 83.55 74.30 78.56 85.36 83.62 84.33 86.32 85.47
0.83 82.26 83.5082.13 82.56 82.80 81.83 82.33 73.01 69.69 80.49 78.58 83.49 83.81 83.04
0.90 72.63 76.5161.83 74.03 74.74 73.76 68.13 54.40 64.62 74.89 73.55 76.38 80.37 79.94
1.00 77.11 78.38 75.12 75.54 76.16 75.02 76.26 66.16 58.11 76.99 75.89 80.5883.92 83.50
1.16 78.68 79.19 81.1879.97 80.00 78.10 77.66 64.60 71.83 80.31 78.00 80.77 82.26 81.66
1.23 69.86 73.02 61.60 68.79 69.79 69.35 63.31 53.14 63.23 71.16 69.22 76.1378.11 79.24
1.33 75.09 77.61 76.21 76.27 76.05 74.58 72.57 64.69 62.78 76.42 74.72 80.3779.84 79.86
1.40 62.87 66.26 54.35 60.71 62.48 62.10 56.95 48.45 53.73 69.54 67.18 73.0279.00 79.06
1.50 68.37 69.63 66.52 62.98 64.11 65.26 63.33 56.22 50.59 71.59 67.04 77.1982.07 82.52
1.56 61.12 64.36 58.95 63.19 64.59 61.12 56.15 45.39 60.61 70.93 67.08 73.5776.38 74.32
1.66 70.62 71.71 73.94 72.01 71.94 69.62 67.76 55.97 62.36 73.50 70.82 77.5078.85 78.80
1.73 60.62 62.99 54.00 56.51 58.11 59.21 52.40 46.95 53.55 66.86 65.77 72.1876.06 77.50
1.80 48.53 47.56 42.14 42.36 43.36 46.46 39.38 36.66 39.11 59.05 59.92 62.1975.98 74.50
1.83 66.82 69.70 64.37 64.52 63.87 66.59 61.04 55.78 52.89 67.11 61.41 77.2677.41 78.15
1.90 51.58 55.06 49.09 44.36 49.14 50.66 46.42 41.36 42.10 62.51 58.15 69.7879.14 77.48
2.06 52.25 54.93 50.90 50.17 52.22 51.53 46.00 40.53 48.99 64.16 61.32 70.0073.25 74.38
2.13 43.29 41.63 42.99 39.02 39.74 43.06 37.22 35.75 36.35 58.44 60.12 60.5175.52 74.66
2.16 60.22 62.29 64.22 58.06 55.89 61.26 54.30 48.30 51.04 65.37 64.00 72.0775.93 75.11
2.23 50.89 53.31 45.41 41.93 45.39 49.78 41.16 40.22 41.95 62.90 61.78 69.2674.12 74.97
2.30 38.15 35.85 37.21 35.49 34.37 36.29 32.66 32.52 33.83 57.22 54.81 58.3476.59 75.04
2.46 37.50 35.23 37.49 35.62 36.00 33.82 28.69 29.53 33.26 51.54 51.01 59.4666.41 69.55
2.56 44.40 45.85 44.42 35.67 38.93 42.77 37.48 36.72 39.28 59.01 56.96 64.2771.26 73.26
2.63 34.00 33.66 35.97 29.58 30.09 32.59 32.39 31.11 32.22 55.63 55.11 59.3372.57 73.51
2.70 33.34 23.70 29.26 35.51 35.51 27.04 29.63 29.63 28.89 50.18 51.85 49.26 74.81 71.11
2.96 33.23 30.52 32.25 28.05 28.48 30.25 28.62 29.51 30.37 49.96 47.41 54.3067.58 68.94
3.03 29.63 22.22 29.26 29.39 29.79 27.78 29.63 28.52 28.15 51.66 53.33 50.00 74.81 71.85
3.36 27.41 19.63 26.30 29.39 29.39 23.70 27.41 26.67 24.81 40.37 40.74 44.4462.96 67.41
be the only augmentation solutions, and they are not systematically
benchmarked against conventional GNN-based approaches in [ 62].
3 GNN Robustness against Graph Deficiencies
To understand how graph deficiencies affect GNN performance, we
evaluate a diverse set of baselines including GNN-based and recent
LLM-enhanced techniques against various deficiencies.
3.1 Preliminaries
3.1.1 Node Classification with Text-attributed Graphs.Let G=
(V,E,T) denote a text-attributed graph, where V={𝑣 1,...,𝑣𝑛}
andE⊆V×V are the sets of nodes and edges, T={𝑡 1,...,𝑡𝑛}
denotes the set of node attributes. Each node 𝑣𝑖∈V is coupled with
corresponding texts 𝑡𝑖∈T, such as descriptions. To process such
textual information, a function 𝜙:T→R𝑑is employed to encode
each𝑡𝑖into a representationx 𝑖=𝜙(𝑡𝑖), yielding a node feature
matrixX∈R𝑛×𝑑stacking all embeddings. Structural information is
captured by an adjacency matrixA ∈{0,1}𝑛×𝑛, where𝐴𝑖𝑗=1if an
edge exists. In the semi-supervised setting, a node subset V𝐿⊂V
is associated with ground-truth labels {𝑦𝑖|𝑣𝑖∈V𝐿}, where each
𝑦𝑖∈Y and|Y|=𝐶 , with𝐶denoting the total number of classes.
The remaining nodes V𝑈=V\V𝐿are unlabeled, and the final goal
is to predict their labels by learning a mapping𝑓:(A,X)→Y𝑛.
3.1.2 GNN robustness against graph deficiencies.In real-world sce-
narios, the observed graph Gobs=(V,E obs)and node features
Xobs∈R𝑛×𝑑are rarely ideal. First, supervision is commonly con-
strained to a small subset V𝐿⊂V , having known labels {𝑦𝑖}𝑣𝑖∈V𝐿,
|V𝐿|≪|V| . The structural observation Eobs⊂E trueleads to in-
complete adjacencyA obs⊂Atrue. Feature vectors can be partiallymissing with∃𝑗:𝑥𝑖𝑗=∅. Finally, available nodes may be insuffi-
cient for learning meaningful representations when |V|is small
or sparsely connected. These deficiencies imply that (Aobs,Xobs)
deviates from the intact graph (Atrue,Xtrue). GNNs are therefore
required to satisfy 𝑓(Aobs,Xobs)≈𝑓( Atrue,Xtrue)to establish ro-
bustness for reliable node representations and classification.
3.1.3 Baselines.We categorize baselines introduced in Section 2
according to graph deficiencies they directly or potentially mitigate.
Structural Deficiency:RSGNN [ 8], GRCN [ 57] and ACM-
GNN [ 36] enhance graph structure robustness against missing or
noisy edges.Supervision Deficiency:DGI [ 48], RSGNN [ 8] and
LLM4NG [ 59] cope with the scarcity of labeled data.Feature De-
ficiency:Neighbor Mean, Feature Propagation [ 41], TAPE [ 17]
and TA_E [ 17] improve robustness to missing or corrupted node
features.Node Deficiency:DropEdge [ 40] and DropMessage [ 11]
help to learn robust representations with part of the nodes missing,
since over-fitting weakens the generalization on small graphs.
Figure 1: Accuracy under increasing attack intensities.

Zhaoyan Wang et al.
Figure 2: GCN Performance under compound deficiencies.
We also adopt Mistral-7B [ 22] and Qwen-3B [ 1] following the
LLM-as-Encoder paradigm as baselines. All of the above baselines
in the empirical study are built on top of a standard GCN backbone.
3.1.4 Empirical Study Settings.We conduct our empirical analysis
on the Cora dataset [ 54]. To ensure fair and consistent comparisons,
all base GNN models (GCN [ 29], GAT [ 47], and GraphSAGE [ 16]) are
implemented with identical configurations, see Appendix Section B
for details. For all LLM-as-Enhancer baselines, Sentence-BERT [ 39]
is employed to encode LLM-generated contents.
3.2 Empirical Analysis
Four types of deficiencies are injected on Cora before training the
downstream GNN classifiers. We report the perturbation space:
Structural Homophily Attack (SHA) with edge reduction ratio ∈
{0,0.5,0.9}, Supervision Scarcity Attack (SSA) with labeled training
ratio∈{0.6,0.4,0.2}, Feature Drop Attack (FDA) with feature drop
ratio∈{0,0.5,0.9}, and Node Removal Attack (NRA) with node
drop ratio∈{0,0.5,0.9}. Validation and test sets are fixed at 20%.
Our analysis addresses: (1)How do compound deficiencies impact
baselines(Obs. 1-2)? (2)Does LLM-as-Enhancer provides performance
and robustness gains worth their complexity(Obs. 3-5)? (3)Do base-
lines exhibit uniform or deficiency-specific sensitivity(Obs. 6)?
Observation 1. Compound deficiencies induce amplified
performance degradation.Table 1, which aggregates the exhaus-
tive results by summing intensities of individual attacks from full
tables exemplified by Appendix Table 5, shows that accuracy con-
sistently decreases as attack intensity increases. This cumulative
impact exceeds what any single deficiency would cause in isola-
tion. Taking vanilla GCN for illustration, the decline in both its
performance visualization Figure 2 and Table 1 further becomes
steeper and non-linear when multiple deficiencies co-occur at high
levels, revealing a compounding effect that accelerates performance
degradation. This necessitates jointly analyzing graph deficiencies
rather than in isolation, a perspective rarely studied before.
Observation 2. Classic GNN is strong baseline.Classic
message-passing GNN (GCN) exhibits strong performance against
almost all non-LLM baselines under compound graph deficiencies,which aligns with the findings of Luo et al. [ 37]. Our results extend
this evidence to robustness studies by demonstrating classic GCN’s
competitiveness when evaluated with a large suite of attacks.
Observation 3. LLM-as-Encoder exhibits superior robust-
ness by providing richer high-dimensional embeddings.As
shown in Table 1, Mistral and Qwen Embeddings sustain higher
accuracy, showing only gradual declines under strong attacks. The
exhibited robustness can be attributed to the richer and higher-
dimensional embeddings that LLMs’ hidden layers provide. As LLM-
based encoders use fundamentally different pretrained embeddings
and are not included in subsequent comparisons for fairness.
Observation 4. LLM augmentations fall behind simple
GNN counterparts under modest deficiencies.Table 1 and
Figure 1 show that across low-to-moderate attack regimes, both
LLM-as-Encoder and LLM-as-Enhancer paradigms perform worse-
to-comparable accuracy than much simpler GNN-based counter-
parts. This observation demonstrates that the integration of LLMs
does not consistently yield better performance and can be even
worse in many situations. Given the trade-offs mentioned in Sec-
tion 1, especially compared with extreme naive approaches such as
Feature Propagation, Neighbor Mean, and DropMessage, LLMs are
not always efficient GNN helpers against graph deficiencies.
Observation 5. LLM-as-Enhancer (e.g., LLM4NG, TAPE,
TA_E) shows no clear robustness advantage over conven-
tional GNN baselines.While LLM Encoders achieve the highest
robustness metrics in Table 2, they operate on a fundamentally
different representational basis. LLM-as-Enhancer baselines exhibit
markedly weaker robustness. Their Worst Acc, Avg Acc, and Norm-
AUC scores lag behind or show no improvement over a substan-
tial set of conventional GNN approaches. This finding points out
that despite their higher architectural complexity, current LLM-as-
Enhancer frameworks fail to deliver tangible robustness gains.
Observation 6. Different baselines exhibit distinct sensi-
tivity profiles to deficiency types.On Cora, the examined base-
lines exhibit distinct vulnerability patterns. For instance, ACM-
GNN, DropEdge, and Neighbor Mean are highly sensitive to NRA
and SHA, showing steep performance degradation, but are less
affected by FDA. In contrast, LLM4NG and TAPE are relatively
Table 2: Robustness evaluation. The reported values are aver-
aged over the four single-type attacks. "Clean Acc" denotes
the accuracy (%) without attack, and "AUC Norm" is the nor-
malized area under the accuracy–attack curve.
Algorithm Clean Acc Worst Acc Avg Acc Norm-AUC
GCN 87.98 75.585 82.575 0.833
DropEdge 87.61 71.988 81.351 0.825
DropMessage 87.65 76.370 82.517 0.831
Feat. Prop. 88.17 77.462 83.303 0.838
NeighborMean 88.17 76.930 83.072 0.836
ACM-GNN 88.54 78.518 84.170 0.847
DGI 82.88 59.375 72.928 0.744
RS-GNN 63.59 61.118 65.543 0.654
GRCN 85.66 67.415 78.643 0.802
MistralEmb. 87.87 81.718 85.125 0.854
QwenEmb. 87.69 81.238 84.960 0.853
TAPE 84.49 77.155 81.480 0.819
TAPE (TA_E) 83.36 75.472 80.232 0.807
LLM4NG 88.06 78.285 83.678 0.842

Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement
more stable under FDA and SHA but show moderate sensitivity
to high-intensity NRA. RS-GNN is relatively robust to FDA but
performs poorly overall. Therefore, robustness is highly algorithm-
dependent, and different baselines present disparate sensitivity.
In summary, current LLM-as-Enhancer frameworks remain com-
plex yet underperform under low-to-moderate perturbations and
show weaker robustness by the Norm-AUC metric. Algorithms also
differ in sensitivity to deficiency composition, exposing a robust-
ness gap that calls for more resilient approaches.
4 Methodology
4.1 System Overview
Our analysis indicates that LLM-as-Enhancer frameworks fail to
deliver better robustness, and occasionally underperform simpler
GNN-based augmentation. A key reason lies in the quality of the
LLM-generated contents ˆ𝑇={ ˆ𝑡𝑖}derived from the original at-
tributes𝑇={𝑡𝑖}. First, ˆ𝑡𝑖tend to be semantically incoherent within
the same classC𝑘and homogeneous across different classes, due to
the lack of semantic guidance. After being encoded and propagated
in GNNs, the learned representationsz 𝑖exhibit high intra-class
variance𝜎2
intra(𝑘)and small inter-class margins𝑚 inter(𝑘,𝑘′):
𝜎2
intra=1
|C𝑘|∑︁
𝑣𝑖∈C𝑘∥z𝑖−¯z𝑘∥2
2↑, 𝑚 inter(𝑘,𝑘′)=∥ ¯z𝑘−¯z𝑘′∥2↓.(1)
Second, the adoption of one-shot generation produces each ˆ𝑡𝑖=
G𝜃(𝑡𝑖)with a LLMG𝜃in a single pass without refinement. As a
result, low-quality or noisy augmentations persist.
To solve such issues, RoGRAD establishes a retrieval-augmented
learning pipeline over the text-attributed graph 𝐺=(V,E,T) to
enhance graph robustness by improving structural connectivity
and semantic discriminability before downstream classification.
The overall architecture of RoGRAD is shown in Figure 3, which
employs: (i) aSemantic-Guided Generation Module (SGGM)that
iteratively synthesizes and refines augmentation samples; (ii) a
Graph Enrichment Stagethat injects generated samples into origi-
nal𝐺to expand node coverage and features, introduce new high-
confidence edges, and supplement pseudo-label supervision; and
(iii) aContrastive Learning with RAG Refinement (R2CL)module that
regularizes node representations on randomly and LLM-perturbed
contrastive views along both structure and embedding dimensions.
Concretely, SGGM leverages G𝜃to produce and iteratively re-
fines initial sample drafts ˆ𝑡(0)
𝑖from𝑡𝑖∈ V , where it retrieves
top-𝑘same-class neighbors from the embedding store M={ e𝑖=
𝜙(𝑡𝑖) |𝑣𝑖∈𝑉} as semantic groundings. Synthetic samples are
then encoded by a Sentence-BERT encoder 𝜓(·), and a feedback
F(ˆ𝑡(𝑟)
𝑖)summarizes similarity-based diagnoses derived from com-
parisons between ˆx(𝑟)
𝑖and existing nodes. Optimized samples are
later merged into the initial graph. Finally, R2CL constructs LLM-
perturbed views eG(1)andeG(2)to obtain representations that en-
force intra-class alignment and inter-class separation.
4.2 Semantic-Guided Generation
SGGM refines the text 𝑡𝑖of augmentation samples ˆ𝑣𝑖∈ˆ𝑉into a high-
quality augmentation ˆ𝑡𝑖through a retrieval–diagnosis–revisionpipeline driven by LLM G𝜃, so that generations can be optimized at
least once, compared with one-shot generation. Original attributes
are encoded ase 𝑗=𝜙(𝑡𝑗)and stored in an embedding memory
M={ e𝑗|𝑣𝑗∈𝑉} , and the initial ˆ𝑡(0)
𝑖generated by prompts in
Figure 10 is encoded as 𝜓(ˆ𝑡(0)
𝑖). SGGM retrieves 𝑘semantically
close same-class exemplarsR(𝑟)
𝑖at round𝑟as grounding context:
R(𝑟)
𝑖=TopK𝜋
𝑗∈C(𝑖)sim( ˆx(𝑟)
𝑖,e𝑗)=TopKDˆx(𝑟)
𝑖
∥ˆx(𝑟)
𝑖∥2,e𝑗
∥e𝑗∥2E
.(2)
These exemplars are embedded intoRAG-enhanced generation
promptsthat instruct G𝜃to: (1) analyze exemplars to extract
category-specific terminology, methodologies, and topics; (2) gen-
erate a new text using representative terms while addressing a
typical same-category problem; and (3) output the text in a strict
Title–Abstract–Keywords format, illustrated by Figure 4 (i). The
additional keywords are incorporated into the embedding with a
fusion weight 𝜆to enhance intra-class alignment. After generation,
the new draft ˆ𝑡(𝑟+1)
𝑖is then encoded as ˆx(𝑟+1)
𝑖=𝜓( ˆ𝑡(𝑟+1)
𝑖)with
main content and keywords. Its similaritys(𝑟+1)
𝑖against all existing
nodes becomes[sim( ˆx(𝑟+1)
𝑖,x𝑗)]𝑗∈𝑉, based on which, four diagnos-
tic metrics: redundancy 𝑟(𝑟+1)
𝑖, class alignment 𝑎(𝑟+1)
𝑖, off-category
drift𝑜(𝑟+1)
𝑖, and duplication𝑑(𝑟+1)
𝑖are computed:
𝑟(𝑟+1)
𝑖=max
𝑗∈C(𝑖)sim( ˆx(𝑟+1)
𝑖,x𝑗), 𝑎(𝑟+1)
𝑖=∑︁
𝑗∈TopKC(𝑖)sim( ˆx(𝑟+1)
𝑖,x𝑗)
𝑘,
𝑜(𝑟+1)
𝑖=max
𝑗∉C(𝑖)sim( ˆx(𝑟+1)
𝑖,x𝑗), 𝑑(𝑟+1)
𝑖=max
ˆ𝑣𝑗∈ˆ𝑉prevsim( ˆx(𝑟+1)
𝑖,ˆx𝑗).(3)
A critique process Fthen converts the diagnostic scores
(𝑟(𝑟+1)
𝑖,𝑎(𝑟+1)
𝑖,𝑜(𝑟+1)
𝑖,𝑑(𝑟+1)
𝑖)into targeted refinement instructions.
Specifically, if 𝑟(𝑟+1)
𝑖or𝑑(𝑟+1)
𝑖are high, the instruction emphasizes
introducing novel elements; if𝑎(𝑟+1)
𝑖is low, it calls for reinforcing
category-specific terms; and if 𝑜(𝑟+1)
𝑖is high, it instructs to remove
off-category terminology. Thefeedback-based refinement prompts
in Figure 4 (ii) guide G𝜃to rewrite drafts with greater diversity
compared to same-class exemplars, stricter category alignment,
and minimal overlap with previously generated texts.
Finally, the original text 𝑡𝑖, the retrieved exemplars R(𝑟+1)
𝑖, and
the feedbackF( ˆ𝑡(𝑟+1)
𝑖)can be fed toG 𝜃to produce the next draft:
ˆ𝑡(𝑟+2)
𝑖=G𝜃 𝑡𝑖,R(𝑟+1)
𝑖,F(ˆ𝑡(𝑟+1)
𝑖).(4)
This refinement stops when no violations remain or when the
maximum round 𝑅is reached. The final ˆ𝑇={ ˆ𝑡(𝑅)
𝑖}with embeddings
ˆx𝑖=𝜓( ˆ𝑡(𝑅)
𝑖)is then utilized to enrich the initial graph.
4.3 Graph Enrichment Against Deficiencies
Refined samples are incorporated to compensate for deficiencies
with encoded features ˆx𝑖. We assemble them as ˆ𝑋=
ˆx1;...;ˆx𝑀
∈
R𝑀×𝑑and introduce the corresponding nodes ˆ𝑉={ ˆ𝑣1,..., ˆ𝑣𝑀}
into the original graph G=(V,E,X) , yielding an enriched graph
eG=(eV,eE,eX)with eV=𝑉∪ ˆ𝑉and eX=[𝑋; ˆ𝑋]∈R(𝑁+𝑀)×𝑑.
For structural enrichment, each ˆ𝑣𝑖is connected to semantically
similar original nodes according to feature similarity sim( ˆx𝑖,x𝑗)=

Zhaoyan Wang et al.
Figure 3: Overall architecture of RoGRAD. RoGRAD establishes the first iterative RAG+GCL paradigm for LLM-on-graph,
replacing static one-shot augmentation with dynamic multi-round refinement.
Figure 4: Prompts for semantic-guided generation.
ˆx⊤
𝑖x𝑗. The threshold𝜏-based neighborhood is defined as:
N𝜏(ˆ𝑣𝑖)={𝑣𝑗∈𝑉|sim( ˆx𝑖,x𝑗)>𝜏},
E ←E∪{( ˆ𝑣𝑖,𝑣𝑗)|𝑣𝑗∈N𝜏(ˆ𝑣𝑖)}.(5)
For supervision enrichment, each ˆ𝑣𝑖is assigned a hard pseudo-
label ˆ𝑦𝑖∈{1,...,𝐶} aligned with its generation category, and the
label set is updated as Y=[𝑌 orig;ˆ𝑌]. To ensure their participation in
optimization, we extend the training mask asmtrain=[mtrain;1𝑀]
while keepingmval=[mval;0𝑀]andmte=[mte;0𝑀], where𝑀
denotes the number of generated nodes newly inserted.
Consequently, the graph evolves from 𝐺toGwith|V|=𝑁+𝑀
and|E|=|𝐸|+Í
𝑖|N𝜏(ˆ𝑣𝑖)|, providing denser connectivity and
richer supervisory signals with additional node embeddings. This
enrichment effectively alleviates weak graph information to yield a
structurally and semantically informative augmented graph.
4.4 Retrieval-Refined Contrastive Learning
R2CL module is designed to periodically inject retrieval-guided se-
mantic refinements into the graph structure and node embeddings.
Modifications on both node embeddings and graph topology en-
courage learned representations to become semantically distinctive
while being label-consistent, thereby improving discriminability.Given the enriched graph eG=(eV,eE,eX)produced by SGGM,
R2CL regularizes node representations through two contrastive
views. At every 𝑇epochs, retrieval-refined augmentation on eGis
performed. Specifically, R2CL maintains the embedding store M
dynamic, where eache 𝑖∈R𝑑is the current representation of node
𝑣𝑖. For each randomly selected anchor node 𝑣𝑖, its top-𝐾nearest
neighborsN𝑖are retrieved from Mbased on similaritye⊤
𝑖e𝑗
∥e𝑖∥∥e𝑗∥.
N𝑖is split into same-class and cross-class sets S𝑖andD𝑖according
to labels{𝑦𝑗}. The anchor text 𝑡𝑖is then semantically optimized by
𝑓LLMfor embedding-level modification, conditioned on the retrieved
contexts{𝑡 𝑗:𝑗∈S𝑖∪D𝑖}, illustrated by Figure 5 (i):
ˆ𝑡𝑖=𝑓LLM 𝑡𝑖;{𝑡𝑗:𝑗∈S𝑖∪D𝑖}, ˆe𝑖=𝑓emb(ˆ𝑡𝑖).(6)
Simultaneously, the LLM determines whether an edge should
exist between the anchor 𝑣𝑖and each retrieved node 𝑣𝑗. Applying
this refinement to all anchors yields a retrieval-refined graph eGRAG
with updated feature matrixXRAGand adjacency matrixARAG.
To perform contrastive learning, R2CL constructs two graph
views. The first view eG(1)is generated by applying random edge
dropping and random feature masking to eG, producing(X(1),A(1)).
The second view eG(2)is generated from eGRAGwhen the current
epoch triggers refinement, or otherwise from another stochastically
perturbed copy of eG, producing(X(2),A(2)). Let𝑓𝜃denote a GCN
Figure 5: Prompts for retrieval-refined contrastive learning.

Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement
Table 3: Overall performance: (a) Improvement over Top 4 baselines on non-deficient graphs, (b) Performance under deficiencies.
Dataset Arch. Base GNN ACM-GNN GRCN Nbr. Mean Feat. Prop. DropMsg DropEdge DGI RS-GNN TAPE TA_E LLM4NG Ours Improv.
CoraGCN 87.98 88.54 85.66 88.17 88.17 87.65 87.61 82.88 63.59 84.49 83.36 88.06 89.39 1.31%
GAT 88.43 87.99 N/A 87.16 87.16 88.28 86.25 83.85 N/A 83.86 82.74 88.47 90.17 2.13%
SAGE 88.84 88.91 83.62 88.25 88.25 89.06 89.24 70.50 72.09 83.90 82.59 87.06 90.39 1.55%
pubMedGCN 86.36 85.45 65.72 86.10 86.09 86.21 87.60 80.97 72.97 82.14 81.43 88.25 96.29 10.54%
GAT 87.19 81.44 N/A 85.69 85.73 86.35 84.57 80.73 N/A 83.13 83.48 88.85 95.94 10.24%
SAGE 88.06 87.98 85.83 88.68 88.59 86.41 86.41 73.01 80.91 78.46 75.80 90.48 97.04 9.09%
ArxivGCN 72.73 75.77 66.74 77.39 80.14 66.04 71.73 63.61 46.73 83.04 82.61 82.09 86.79 5.88%
GAT 72.45 46.32 N/A 80.48 80.24 70.07 42.66 64.37 N/A 83.30 82.57 84.51 88.79 7.34%
SAGE 68.51 79.09 63.33 79.48 79.71 68.03 66.70 54.06 68.77 85.22 83.52 84.99 92.16 10.56%
Atk. Int. ACM-GNN NeighborMean Feat. Prop. TAPE TA_E LLM4NG Ours
Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv
0.00 88.5487.99 88.91 88.1787.16 88.25 88.1787.16 88.25 84.4983.86 83.90 83.3682.74 82.59 88.0688.47 87.06 89.3990.17 90.39
0.33 87.3987.06 88.91 86.3685.15 86.84 86.3684.99 86.84 84.5384.27 83.96 83.3283.03 80.93 86.2187.06 86.62 88.9889.17 89.46
0.50 84.8182.64 84.77 83.3782.52 83.65 83.5882.47 83.73 81.7480.27 81.46 80.5979.39 80.65 84.0784.19 83.82 86.0186.66 86.69
0.66 84.5583.66 84.36 85.6383.70 85.03 85.6383.70 85.03 85.3684.01 85.01 83.6281.52 83.33 84.3383.18 82.74 87.1386.88 86.73
0.83 83.5081.22 83.87 82.5681.91 82.69 82.8081.95 82.94 80.4979.52 79.17 78.5877.90 77.30 83.4983.29 82.85 85.3385.35 85.78
0.90 76.5172.13 76.74 74.0373.25 73.70 74.7473.61 74.72 74.8972.59 74.62 73.5571.46 72.76 76.3876.95 76.79 78.8079.37 79.40
1.00 78.3875.30 78.34 75.5474.89 73.94 76.1675.32 74.42 76.9977.18 77.27 75.8974.69 75.19 80.5880.95 81.03 83.0082.65 83.32
1.16 79.1976.79 78.47 79.9778.75 78.98 80.0078.63 78.97 80.3178.69 80.16 78.0076.24 78.60 80.7780.11 79.33 83.2183.06 83.03
1.23 73.0266.44 72.45 68.7968.83 69.62 69.7969.31 71.56 71.1670.32 73.53 69.2268.73 71.05 76.1375.83 76.25 78.3778.35 78.16
1.33 77.6173.78 77.00 76.2774.67 74.07 76.0575.06 74.35 76.4276.19 74.00 74.7275.19 70.16 80.3781.08 79.25 82.5882.69 82.42
1.40 66.2661.46 66.21 60.7160.43 59.27 62.4862.62 61.79 69.5468.71 68.87 67.1864.57 65.61 73.0272.98 73.03 75.2475.87 75.64
1.50 69.6364.37 68.22 62.9862.82 59.84 64.1163.31 61.85 71.5974.52 74.89 67.0469.78 72.37 77.1976.07 76.59 80.3080.30 78.96
1.56 64.3659.31 64.68 63.1964.41 64.94 64.5965.43 67.20 70.9369.77 71.84 67.0864.36 67.60 73.5773.55 73.29 76.2376.39 76.78
1.66 71.7166.45 70.14 72.0170.45 69.60 71.9469.94 71.19 73.5072.62 74.02 70.8271.60 73.61 77.5076.79 75.44 78.9780.06 78.69
1.73 62.9957.49 62.55 56.5156.59 55.63 58.1158.17 57.75 66.8668.67 68.91 65.7766.51 66.40 72.1871.79 71.61 74.1375.55 75.12
1.80 47.5644.66 47.45 42.3645.41 44.58 43.3646.11 45.85 59.0557.92 57.27 59.9249.49 56.93 62.1963.26 63.83 65.3865.44 64.34
1.83 69.7066.22 69.78 64.5263.63 59.68 63.8763.06 63.63 67.1172.63 69.37 61.4172.22 64.81 77.2678.67 75.70 79.7080.74 79.85
1.90 55.0651.73 54.37 44.3646.72 43.74 49.1446.94 46.30 62.5165.28 67.35 58.1559.51 63.36 69.7869.85 69.28 72.3771.53 72.79
2.06 54.9348.75 54.47 50.1750.81 50.01 52.2252.37 52.55 64.1664.27 63.88 61.3260.52 59.86 70.0069.23 68.14 72.8372.38 71.97
2.13 41.6342.26 42.39 39.0240.60 41.01 39.7440.87 41.05 58.4460.78 60.37 60.1263.94 59.28 60.5161.96 60.75 64.1164.40 65.93
2.16 62.2957.56 60.15 58.0657.42 55.40 55.8955.08 55.40 65.3767.82 64.59 64.0068.96 60.07 72.0772.81 69.33 76.7475.26 74.00
2.23 53.3148.45 51.43 41.9340.82 41.02 45.3943.63 42.85 62.9064.57 65.83 61.7861.06 63.93 69.2668.62 66.91 72.6772.57 70.84
2.30 35.8537.88 36.74 35.4935.27 35.48 34.3736.57 35.75 57.2251.53 59.80 54.8148.82 57.95 58.3459.73 59.46 62.0262.82 62.94
2.46 35.2335.95 34.64 35.6235.47 34.18 36.0035.86 33.54 51.5453.49 55.88 51.0149.45 52.37 59.4659.22 56.40 63.6965.79 63.20
2.56 45.8540.81 45.53 35.6735.89 34.28 38.9338.67 38.34 59.0161.76 56.05 56.9660.74 53.58 64.2765.56 62.30 68.8268.79 67.68
2.63 33.6634.47 33.78 29.5829.66 29.83 30.0930.18 29.34 55.6355.91 55.94 55.1151.38 53.46 59.3358.64 57.46 63.1963.06 64.07
2.70 23.7029.26 23.70 35.5135.92 35.51 35.5136.33 35.51 50.1842.04 37.03 51.8534.45 33.33 49.2648.89 51.85 53.3356.67 56.30
2.96 30.5232.45 29.88 28.0529.00 26.65 28.4829.70 25.89 49.9650.81 47.87 47.4148.00 47.58 54.3055.28 53.26 60.5761.73 59.70
3.03 22.2230.37 25.56 29.3929.39 28.98 29.7929.39 28.57 51.6640.74 41.66 53.3342.59 38.89 50.0045.19 48.15 57.0455.56 61.11
3.36 19.6329.63 19.63 29.3929.39 28.57 29.3929.39 28.57 40.3740.00 41.66 40.7439.26 38.89 44.4447.41 45.19 54.8154.07 54.81
encoder and 𝑔𝜙a projection head. For each node 𝑣𝑖in a batchB,
we compute normalized embeddings from both views:
h(𝑘)
𝑖=𝑓𝜃(X(𝑘),A(𝑘))𝑖,z(𝑘)
𝑖=𝑔𝜙(h(𝑘)
𝑖)
∥𝑔𝜙(h(𝑘)
𝑖)∥2, 𝑘∈{1,2}.(7)
We concatenate the embeddings from the two views and optimize
them with the supervised contrastive objectiveL:
L=1
|B|∑︁
𝑖∈B𝜔·−1
|𝑃(𝑖)|∑︁
𝑝∈𝑃(𝑖)logexp(z𝑖·z𝑝/𝜏𝑡𝑒𝑚𝑝)Í
𝑎∈𝐴(𝑖) exp(z𝑖·z𝑎/𝜏𝑡𝑒𝑚𝑝),(8)
where𝑃(𝑖)={𝑝≠𝑖|𝑦 𝑝=𝑦𝑖},𝐴(𝑖)={𝑎≠𝑖} , and𝜏𝑡𝑒𝑚𝑝 is
the temperature hyperparameter. To strengthen the contribution of
generated nodesV=𝑉∪ ˆ𝑉, their loss multiplicative weight 𝜔> 1.
The final robust representations and the LLM-refined enriched
graph are subsequently fed into the downstream GNN classifier for
node classification, trained in the presence of injected attacks.
Unlike previous contrastive learning approaches in SimGCL [ 33]
and SGL [ 51], which rely on structural perturbations for contrastive
views, R2CL conducts contrastive optimization on LLM-refinedretrieval-guided views during augmentation before GNN classifica-
tion. Section 5.2.4 further demonstrates its superiority.
5 Experiments
In this section, we study the following research questions:
RQ1: Does RoGRAD outperform all baselines on both non-
deficient and deficient graphs, particularly under low-to-moderate
attacks?
RQ2: How does each component contribute to the overall per-
formance of RoGRAD?
RQ3: Can RoGRAD build the strongest robustness under com-
pound graph deficiencies, allowing LLM-as-Enhancer to surpass
previously more robust non-LLM baselines?
RQ4: Can R2CL, by introducing LLM guided modification, out-
perform prior graph contrastive learning approaches?
5.1 Experimental Setup
We conducted experiments on theCora[ 54],PubMed[ 54], and
Arxiv[ 19] datasets (see Appendix Section E for details). In addition

Zhaoyan Wang et al.
Figure 6: Accuracy vs. attack intensity (left). Relative improvement over selected well-performing baselines in Table 3.b (right).
to the configurations in Section 3.1.4, we further specify the fol-
lowing settings. For the SGGM, we retrieve 𝑘=10same-category
exemplars and apply similarity thresholds of 0.85 ( 𝑟𝑖), 0.6 (𝑎𝑖), 0.3
(𝑜𝑖), and 0.7 (𝑑𝑖). Generated nodes were linked when similarity
>𝜏= 0.7. For the R2CL module, a 4-layer GCN (256-dim) with a
128-dim projection head is adopted as encoder, batch size = 128,
𝜏𝑡𝑒𝑚𝑝=0.07, and loss weight 𝜔= 2.0. It is trained for 50 epochs,
applying RAG enhancement every 𝑇=5epochs using 15 anchors
(3 same + 7 different) with edge drop = 0.1 and feature mask = 0.1.
5.2 Results and Analysis
5.2.1 Performance under Non-deficient and Deficient Graphs (RQ1).
Experiments demonstrate RoGRAD’s (Ours) consistent advantages
over conventional GNN-based and LLM-enhanced baselines on
both clean and deficient graphs. As shown in Table 3.a, under
clean graphs, it achieves the highest accuracy on all three datasets
and GNN backbones. For instance, on PubMed-SAGE, RoGRAD
reaches the highest 97.04% accuracy, outperforming the best base-
line (LLM4NG) by 7.28% and 9.09% for the top-4 best-performing
baselines. RoGRAD also demonstrates considerable improvement,
up to 10.56% improvement on other datasets and GNN architectures.
According to Table 3.b, RoGRAD consistently achieves the high-
est accuracy at every attack intensity on deficient graphs. Figure 6
shows that as attack intensity increases, LLM-as-Enhancer baselines
outperform GNN-based ones, but still lag behind RoGRAD (pink).
RoGRAD achieves improvement up to 62.12% on PubMed, 74.31%
on Cora, and 82.43% on Arxiv over best-performing baselines.
Under low-to-moderate attacks, RoGRAD successfully enables
LLM-as-Enhancer frameworks to surpass all non-LLM baselines.
The observed reversal highlights the effectiveness of RoGRAD’s
retrieval-augmented generation and contrastive refinement in miti-
gating the drawbacks of previously LLM-based augmentation.
5.2.2 Ablation Study (RQ2).Figure 7 confirms the contribution of
each module. Comparing W/o Both with base GNNs underscores
that graph enrichment in Section 4.3 leads to much higher accuracy
by supplementing missing information. The gap between W/o Both
and W/o RCLM demonstrates the effect of SGGM, which becomes
increasingly evident under higher attack intensities, reaching more
than 10% on Cora. Although this gap varies across datasets and
GNN backbones, it is also clear on PubMed and Arxiv. SGGM is
proven to yield further gains at high attack levels by injecting di-
verse, class-consistent information. Finally, the improvement of Full
over W/o RCLM highlights the indispensable role of RCLM, which
significantly enhances accuracy under severe deficiencies through
retrieval-refined contrastive optimization. These results show that
Figure 7: Ablations on Cora, PubMed, and Arxiv datasets.
each module contributes unique value, and their integration allows
RoGRAD to consistently achieve the best accuracy across datasets.
5.2.3 Robustness under Compound Deficiencies (RQ3).In Table 4,
RoGRAD establishes the strongest robustness across all datasets.
On PubMed, it reaches 96.29% clean accuracy and an outstanding
90.64% worst-case accuracy, far higher than the strongest baseline
LLM4NG at 84.48%. On Arxiv, RoGRAD improves average accuracy
to 84.99%, surpassing TAPE (80.01%) and LLM4NG (80.12%). On
Cora, RoGRAD attains the highest average accuracy, 85.68%, and
Norm-AUC 0.86. These results show that, unlike prior LLM-based
methods that trail behind robust GNNs, RoGRAD surpasses both,
proving most robust under compound deficiencies.
5.2.4 Robust Contrastive Learning with R2CL (RQ4).Traditional
contrastive learning methods for graphs, such as SimGCL [ 33] and
SGL [ 51], rely on one-shot perturbations to construct views, which
Figure 8: Accuracy across datasets: Ours vs. GCL baselines.

Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement
Table 4: Robustness across all benchmark datasets on GCN.
AlgorithmClean Acc Worst Acc Avg Acc Norm-AUC
Cora Pub. Arx. Cora Pub. Arx. Cora Pub. Arx. Cora Pub. Arx.
GCN 87.98 86.36 72.73 75.59 82.33 62.03 82.58 84.46 67.98 0.83 0.85 0.69
DropEdge 87.61 87.60 71.73 71.99 82.91 61.17 81.35 85.41 67.29 0.83 0.86 0.68
DropMsg 87.65 86.21 66.04 76.37 82.26 56.63 82.52 84.38 61.82 0.83 0.85 0.62
Feat. Prop. 88.17 86.09 80.14 77.46 82.39 65.33 83.30 84.20 73.58 0.84 0.84 0.74
Nbr. Mean 88.17 86.10 77.39 76.93 82.19 58.86 83.07 84.10 68.85 0.84 0.84 0.70
ACM-GNN 88.54 85.45 75.77 78.52 82.11 61.48 84.17 83.63 68.93 0.85 0.84 0.70
DGI 82.88 80.97 63.61 59.38 67.95 50.44 72.93 75.68 58.08 0.74 0.77 0.59
RS-GNN 63.59 72.97 46.73 61.12 66.52 34.34 65.54 69.58 40.54 0.65 0.70 0.41
GRCN 85.66 65.72 66.74 67.42 58.62 54.63 78.64 63.20 61.68 0.80 0.64 0.63
TAPE 84.49 82.14 83.04 77.16 73.03 76.13 81.48 77.87 80.01 0.82 0.78 0.80
TA_E 83.36 81.43 82.61 75.47 72.43 76.01 80.23 77.60 79.34 0.81 0.78 0.80
LLM4NG 88.06 88.25 82.09 78.29 84.48 77.02 83.68 86.44 80.12 0.84 0.87 0.81
Ours 89.39 96.29 86.79 80.88 90.64 81.94 85.68 93.61 84.99 0.86 0.94 0.85
offer limited guidance to iteratively preserve discriminative con-
sistency. In contrast, R2CL continuously optimizes graph views
through LLM-perturbed semantic refinements and structures to uti-
lize LLM’s unique strength in generating. As illustrated in Figure 8,
R2CL consistently encloses a larger shaded performance region
than SimGCL and SGL, reflecting higher accuracy and stronger
robustness across a wide range of attack intensities. The iterative
refinement explicitly enforces intra-class alignment and inter-class
separation. These results confirm that R2CL not only surpasses
existing graph contrastive baselines but also establishes robust
advantages of iterative, LLM-refined contrastive learning.
5.2.5 Effect of Hyperparameter 𝜆.The hyperparameter 𝜆, which
denotes the fusion weight ofMain Textual Content: Keywords, con-
trols the weight of keywords in the semantic-guided generation,
directly influencing the balance between main content and class-
specific terminology. As illustrated in Figure 9, different settings of
𝜆yield distinct accuracy trends under varying attack intensities.
When𝜆= 0.0, GCN remains relatively stable under mild attacks
but deteriorates rapidly as attack intensity increases. Moderate set-
ting when𝜆= 1 yields the most robust performance by reinforcing
intra-class alignment, but its performance under mild attacks is not
sufficient, whereas 𝜆= 1.5 presents the largest fluctuation. Overall,
moderate discrete choices of 𝜆= 2 provide balanced performance
and robustness to integrate keywords for class alignment.
Figure 9: Effect of keyword weight 𝜆on GCN accuracy (Arxiv).
6 Conclusion
In this work, we conducted the first comprehensive evaluation
of LLM-on-graph learning under compound graph deficiencies.Our study reveals that existing LLM-as-Enhancer paradigms often
deliver unstable performance and may fail to surpass traditional
GNN-based methods. This finding challenges the common assump-
tion that LLMs are generally better GNN helpers. Beyond empirical
gains, our work introduces a new iterative refinement paradigm
RoGRAD, the first retrieval-augmented generation framework that
targets robust graph learning under graph deficiencies to refine and
stabilize LLM guidance. RoGRAD leverages retrieval-augmented
refinement of LLM guidance to enhance intra-class consistency and
inter-class separation. At the same time, it integrates supplemen-
tary information to address graph deficiencies, thereby establishing
a principled mechanism for generating reliable and robust node
representations. Extensive experiments demonstrate that RoGRAD
sets a new benchmark for robust graph learning with LLMs. These
results establish RoGRAD as a new robust Web graph learning
framework and provide the first evidence that retrieval-augmented
iterative refinement is crucial for integrating LLMs on graphs.

Zhaoyan Wang et al.
References
[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
Wenbin Ge, Yu Han, Fei Huang, et al .2023. Qwen technical report.arXiv preprint
arXiv:2309.16609(2023).
[2] Prajjwal Bhargava and Vincent Ng. 2022. Commonsense knowledge reasoning
and generation with pre-trained language models: A survey. InProceedings of
the AAAI conference on artificial intelligence, Vol. 36. 12317–12325.
[3] Aleksandar Bojchevski and Stephan Günnemann. 2019. Certifiable robustness
to graph perturbations.Advances in Neural Information Processing Systems32
(2019).
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners.Advances in neural
information processing systems33 (2020), 1877–1901.
[5] Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang,
and Yang Yang. 2023. Graphllm: Boosting graph reasoning ability of large
language model.arXiv preprint arXiv:2310.05845(2023).
[6] Hoyeon Chang, Jinho Park, Seonghyeon Ye, Sohee Yang, Youngkyung Seo, Du-
Seong Chang, and Minjoon Seo. 2024. How do large language models acquire
factual knowledge during pretraining?Advances in neural information processing
systems37 (2024), 60626–60668.
[7]Zhikai Chen, Haitao Mao, Hang Li, Wei Jin, Hongzhi Wen, Xiaochi Wei,
Shuaiqiang Wang, Dawei Yin, Wenqi Fan, Hui Liu, et al .2024. Exploring the
potential of large language models (llms) in learning on graphs.ACM SIGKDD
Explorations Newsletter25, 2 (2024), 42–61.
[8]Enyan Dai, Wei Jin, Hui Liu, and Suhang Wang. 2022. Towards robust graph
neural networks for noisy graphs with sparse labels. InProceedings of the fifteenth
ACM international conference on web search and data mining. 181–191.
[9] Enyan Dai, Tianxiang Zhao, Huaisheng Zhu, Junjie Xu, Zhimeng Guo, Hui Liu,
Jiliang Tang, and Suhang Wang. 2024. A comprehensive survey on trustworthy
graph neural networks: Privacy, robustness, fairness, and explainability.Machine
Intelligence Research21, 6 (2024), 1011–1061.
[10] Kaize Ding, Jianling Wang, James Caverlee, and Huan Liu. 2022. Meta propagation
networks for graph few-shot semi-supervised learning. InProceedings of the AAAI
conference on artificial intelligence, Vol. 36. 6524–6531.
[11] Taoran Fang, Zhiqing Xiao, Chunping Wang, Jiarong Xu, Xuan Yang, and Yang
Yang. 2023. Dropmessage: Unifying random dropping for graph neural networks.
InProceedings of the AAAI conference on artificial intelligence, Vol. 37. 4267–4275.
[12] Fuli Feng, Xiangnan He, Jie Tang, and Tat-Seng Chua. 2019. Graph adversarial
training: Dynamically regularizing based on graph structure.IEEE Transactions
on Knowledge and Data Engineering33, 6 (2019), 2493–2504.
[13] Tao Feng, Lizhen Qu, Zhuang Li, Haolan Zhan, Yuncheng Hua, and Reza Haf.
2024. IMO: Greedy Layer-Wise Sparse Representation Learning for Out-of-
Distribution Text Classification with Pre-trained Models. InProceedings of the
62nd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 2625–2639.
[14] Luca Franceschi, Mathias Niepert, Massimiliano Pontil, and Xiao He. 2019. Learn-
ing discrete structures for graph neural networks. InInternational conference on
machine learning. PMLR, 1972–1982.
[15] Chen Gao, Xiang Wang, Xiangnan He, and Yong Li. 2022. Graph neural net-
works for recommender system. InProceedings of the fifteenth ACM international
conference on web search and data mining. 1623–1625.
[16] Will Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive representation
learning on large graphs.Advances in neural information processing systems30
(2017).
[17] Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and
Bryan Hooi. 2023. Harnessing explanations: Llm-to-lm interpreter for enhanced
text-attributed graph representation learning.arXiv preprint arXiv:2305.19523
(2023).
[18] Linmei Hu, Zeyi Liu, Ziwang Zhao, Lei Hou, Liqiang Nie, and Juanzi Li. 2023. A
survey of knowledge enhanced pre-trained language models.IEEE Transactions
on Knowledge and Data Engineering36, 4 (2023), 1413–1430.
[19] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu,
Michele Catasta, and Jure Leskovec. 2020. Open graph benchmark: Datasets for
machine learning on graphs.Advances in neural information processing systems
33 (2020), 22118–22133.
[20] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2025. A
survey on hallucination in large language models: Principles, taxonomy, chal-
lenges, and open questions.ACM Transactions on Information Systems43, 2
(2025), 1–55.
[21] Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu
Wang, and Jie Tang. 2021. Mixgcf: An improved training method for graph neural
network-based recommender systems. InProceedings of the 27th ACM SIGKDD
conference on knowledge discovery & data mining. 665–674.
[22] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux,
Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix,
and William El Sayed. 2023. Mistral 7B. arXiv:2310.06825 [cs.CL] https:
//arxiv.org/abs/2310.06825
[23] Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. 2024. Large
language models on graphs: A comprehensive survey.IEEE Transactions on
Knowledge and Data Engineering(2024).
[24] Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, and Jiliang Tang.
2020. Graph structure learning for robust graph neural networks. InProceedings
of the 26th ACM SIGKDD international conference on knowledge discovery & data
mining. 66–74.
[25] Wei Ju, Siyu Yi, Yifan Wang, Zhiping Xiao, Zhengyang Mao, Hourun Li, Yiyang
Gu, Yifang Qin, Nan Yin, Senzhang Wang, et al .2024. A survey of graph neural
networks in real world: Imbalance, noise, privacy and ood challenges.arXiv
preprint arXiv:2403.04468(2024).
[26] Shima Khoshraftar, Niaz Abedini, and Amir Hajian. 2025. GraphiT: Efficient
Node Classification on Text-Attributed Graphs with Prompt Optimized LLMs. In
Companion Proceedings of the ACM on Web Conference 2025. 1824–1829.
[27] Jongmin Kim, Taesup Kim, Sungwoong Kim, and Chang D Yoo. 2019. Edge-
labeling graph neural network for few-shot learning. InProceedings of the
IEEE/CVF conference on computer vision and pattern recognition. 11–20.
[28] Junghun Kim, Ka Hyun Park, Hoyoung Yoon, and U Kang. 2025. Accurate link
prediction for edge-incomplete graphs via PU learning. InProceedings of the
AAAI Conference on Artificial Intelligence, Vol. 39. 17877–17885.
[29] TN Kipf. 2016. Semi-supervised classification with graph convolutional networks.
arXiv preprint arXiv:1609.02907(2016).
[30] Gueorgi Kossinets. 2006. Effects of missing data in social networks.Social
networks28, 3 (2006), 247–268.
[31] Junseok Lee, Yunhak Oh, Yeonjun In, Namkyeong Lee, Dongmin Hyun, and
Chanyoung Park. 2022. Grafn: Semi-supervised node classification on graph
with few labels via non-parametric distribution assignment. InProceedings of
the 45th International ACM SIGIR Conference on Research and Development in
Information Retrieval. 2243–2248.
[32] Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, and Qing He.
2022. Reliable representations make a stronger defender: Unsupervised structure
refinement for robust gnn. InProceedings of the 28th ACM SIGKDD conference on
knowledge discovery and data mining. 925–935.
[33] Cheng Liu, Chenhuan Yu, Ning Gui, Zhiwu Yu, and Songgaojun Deng. 2024.
SimGCL: graph contrastive learning by finding homophily in heterophily.Knowl-
edge and Information Systems66, 3 (2024), 2089–2114.
[34] Yixin Liu, Kaize Ding, Jianling Wang, Vincent Lee, Huan Liu, and Shirui Pan. 2023.
Learning strong graph neural networks with weak information. InProceedings
of the 29th ACM SIGKDD conference on knowledge discovery and data mining.
1559–1571.
[35] Zhenyi Lu, Jie Tian, Wei Wei, Xiaoye Qu, Yu Cheng, Dangyang Chen, et al .2024.
Mitigating boundary ambiguity and inherent bias for text classification in the
era of large language models.arXiv preprint arXiv:2406.07001(2024).
[36] Sitao Luan, Chenqing Hua, Qincheng Lu, Jiaqi Zhu, Mingde Zhao, Shuyuan
Zhang, Xiao-Wen Chang, and Doina Precup. 2022. Revisiting heterophily for
graph neural networks.Advances in neural information processing systems35
(2022), 1362–1375.
[37] Yuankai Luo, Lei Shi, and Xiao-Ming Wu. 2024. Classic gnns are strong base-
lines: Reassessing gnns for node classification.Advances in Neural Information
Processing Systems37 (2024), 97650–97669.
[38] Pál András Papp, Karolis Martinkus, Lukas Faber, and Roger Wattenhofer. 2021.
DropGNN: Random dropouts increase the expressiveness of graph neural net-
works.Advances in Neural Information Processing Systems34 (2021), 21997–22009.
[39] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings
using siamese bert-networks.arXiv preprint arXiv:1908.10084(2019).
[40] Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. 2019. Dropedge:
Towards deep graph convolutional networks on node classification.arXiv preprint
arXiv:1907.10903(2019).
[41] Emanuele Rossi, Henry Kenlay, Maria I Gorinova, Benjamin Paul Chamberlain,
Xiaowen Dong, and Michael M Bronstein. 2022. On the unreasonable effective-
ness of feature propagation in learning on graphs with missing node features. In
Learning on graphs conference. PMLR, 11–1.
[42] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan
Titov, and Max Welling. 2018. Modeling relational data with graph convolutional
networks. InEuropean semantic web conference. Springer, 593–607.
[43] Kartik Sharma, Yeon-Chang Lee, Sivagami Nambi, Aditya Salian, Shlok Shah,
Sang-Wook Kim, and Srijan Kumar. 2024. A survey of graph neural networks
for social recommender systems.Comput. Surveys56, 10 (2024), 1–34.
[44] Hibiki Taguchi, Xin Liu, and Tsuyoshi Murata. 2021. Graph convolutional net-
works for graphs containing missing features.Future Generation Computer
Systems117 (2021), 155–168.
[45] Zhen Tan, Kaize Ding, Ruocheng Guo, and Huan Liu. 2022. Supervised graph
contrastive learning for few-shot node classification. InJoint European Conference
on Machine Learning and Knowledge Discovery in Databases. Springer, 394–411.

Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement
[46] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. 2019.
Composition-based multi-relational graph convolutional networks.arXiv
preprint arXiv:1911.03082(2019).
[47] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
Lio, and Yoshua Bengio. 2017. Graph attention networks.arXiv preprint
arXiv:1710.10903(2017).
[48] Petar Veličković, William Fedus, William L Hamilton, Pietro Liò, Yoshua Bengio,
and R Devon Hjelm. 2018. Deep graph infomax.arXiv preprint arXiv:1809.10341
(2018).
[49] Binghui Wang, Jinyuan Jia, Xiaoyu Cao, and Neil Zhenqiang Gong. 2021. Certified
robustness of graph neural networks against adversarial structural perturbation.
InProceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &
Data Mining. 1645–1653.
[50] Zhaoyan Wang, Xiangchi Song, and In-Young Ko. 2025. MultiGran-STGCNFog:
Towards Accurate and High-Throughput Inference for Multi-Granular Spatiotem-
poral Traffic Forecasting.arXiv preprint arXiv:2505.01279(2025).
[51] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and
Xing Xie. 2021. Self-supervised graph learning for recommendation. InProceed-
ings of the 44th international ACM SIGIR conference on research and development
in information retrieval. 726–735.
[52] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022. Graph neural
networks in recommender systems: a survey.Comput. Surveys55, 5 (2022), 1–37.
[53] Xixi Wu, Yifei Shen, Fangzhou Ge, Caihua Shan, Yizhu Jiao, Xiangguo Sun,
and Hong Cheng. 2025. When Do LLMs Help With Node Classification? A
Comprehensive Analysis.arXiv preprint arXiv:2502.00829(2025).
[54] Zhilin Yang, William Cohen, and Ruslan Salakhudinov. 2016. Revisiting semi-
supervised learning with graph embeddings. InInternational conference on ma-
chine learning. PMLR, 40–48.
[55] Jiaxuan You, Xiaobai Ma, Yi Ding, Mykel J Kochenderfer, and Jure Leskovec.
2020. Handling missing data with graph representation learning.Advances in
Neural Information Processing Systems33 (2020), 19075–19087.
[56] Bing Yu, Haoteng Yin, and Zhanxing Zhu. 2018. Spatio-temporal graph convolu-
tional networks: a deep learning framework for traffic forecasting. InProceedings
of the 27th International Joint Conference on Artificial Intelligence. 3634–3640.
[57] Donghan Yu, Ruohong Zhang, Zhengbao Jiang, Yuexin Wu, and Yiming Yang.
2020. Graph-revised convolutional network. InJoint European conference on
machine learning and knowledge discovery in databases. Springer, 378–393.
[58] Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, and Xuecang
Zhang. 2023. Empower text-attributed graphs learning with large language
models (llms).arXiv preprint arXiv:2310.09872(2023).
[59] Jianxiang Yu, Yuxiang Ren, Chenghua Gong, Jiaqi Tan, Xiang Li, and Xuecang
Zhang. 2025. Leveraging large language models for node generation in few-shot
learning on text-attributed graphs. InProceedings of the AAAI Conference on
Artificial Intelligence, Vol. 39. 13087–13095.
[60] Xiangchi Yuan, Chunhui Zhang, Yijun Tian, Yanfang Ye, and Chuxu Zhang. 2024.
Mitigating emergent robustness degradation while scaling graph learning. The
Twelfth International Conference on Learning Representations (ICLR).
[61] Yanfu Zhang, Shangqian Gao, Jian Pei, and Heng Huang. 2022. Improving social
network embedding via new second-order continuous graph neural networks.
InProceedings of the 28th ACM SIGKDD conference on knowledge discovery and
data mining. 2515–2523.
[62] Zhongjian Zhang, Xiao Wang, Huichi Zhou, Yue Yu, Mengmei Zhang, Cheng
Yang, and Chuan Shi. 2025. Can large language models improve the adversarial
robustness of graph neural networks?. InProceedings of the 31st ACM SIGKDD
Conference on Knowledge Discovery and Data Mining V. 1. 2008–2019.
[63] Jianan Zhao, Le Zhuo, Yikang Shen, Meng Qu, Kai Liu, Michael Bronstein,
Zhaocheng Zhu, and Jian Tang. 2023. Graphtext: Graph reasoning in text space.
arXiv preprint arXiv:2310.01089(2023).
[64] Zhengyun Zhou, Guojia Wan, and Bo Du. 2025. Common Neighbor Completion
with Information Entropy for Link Prediction in Social Networks.Data Science
and Engineering10, 1 (2025), 40–53.
[65] Yanqiao Zhu, Weizhi Xu, Jinghao Zhang, Qiang Liu, Shu Wu, and Liang Wang.
2021. Deep graph structure learning for robust representations: A survey.arXiv
preprint arXiv:2103.0303614 (2021), 1–1.
A Issues with One-shot LLM Augmentations
To illustrate the drawbacks of current LLM-as-Enhancer frame-
works, we followed the prompting style used in LLM4NG and
TAPE, which ask an LLM to generate academic-style samples in a
Title–Abstract format. Below we show two such examples. Despite
targeting different domains, the outputs reveal the same issues ob-
served in prior work: (1) semantic homogeneity across categories,
where abstracts share nearly identical templates and phrasing, and(2) the one-shot limitation, since low-quality generations cannot
be refined. These issues limit the usefulness of LLM-generated
augmentations for robust graph learning.
Title. Enhancing Neural Network Interpretability through
Layer-wise Relevance Propagation (Class: Neural Networks)
Abstract.Neural networks have become increasingly prevalent
in various domains; however , the need for interpretability has
emerged as a critical concern .This paper introduces an inno-
vative framework for enhancing the interpretability of neural
networks by leveraging Layer-wise Relevance Propagation (LRP).
We propose an extended version of LRP that adapts its relevance
propagation mechanism to accommodate both feedforward and
recurrent architectures, facilitating the analysis of complex neural
models. Through extensive experiments on various benchmark
datasets, we demonstrate the effectiveness of our approach in
providing meaningful insights into model predictions while main-
taining high accuracy. Our framework not only elucidates the
decision-making processes of neural networks but also identifies
key features influencing those decisions, thus empowering practi-
tioners to understand and trust their models. We further discuss the
implications of our findings for areas such as model auditing, and
feature engineering. Finally, we outline future research directions
aimed at refining interpretability in deep learning architectures.
Title. Enhancing Sample Efficiency in Reinforcement Learning
through Modular Exploration (Class: Reinforcement Learning)
Abstract.Reinforcement Learning (RL) has achieved
remarkable success in various domains; however, its reliance
on extensive data and sample efficiency remains challenging .
This paper proposes a novel approach to improve sample
efficiency in RL through the integration of modular exploration
strategies. By decomposing the RL learning process into distinct
modules, we effectively enable agents to explore environments
in a more systematic manner, facilitating faster convergence
towards optimal policies. We introduce a framework that combines
curiosity-driven exploration with episodic memory, allowing
agents to prioritize previously unexplored states while retaining
valuable experiences from past behaviors. Our empirical eval-
uations on standard RL benchmarks demonstrate significant
improvements in learning speed and overall performance relative
to traditional exploration methods. Additionally, our approach
reveals insights into balancing exploration and exploitation
through modular components, providing a more adaptable and
robust solution for continuous and discrete action spaces. The
implications of this research extend to multi-agent systems and
real-world applications, where sample efficiency is paramount for
effective decision-making .
B Configurations of Baseline Methods
All base GNNs are implemented with consistent hyperparameter
settings for fair comparison. The main configurations are as follows:
•Shared Parameters.Hidden dimension = 512, dropout =
0.3, learning rate = 0.01, weight decay (L2 coefficient) = 0.0.
Node drop ratio, feature drop ratio, and same-type edge
reduction ratio are set to 0.5 unless otherwise specified.

Zhaoyan Wang et al.
•GCN Encoder.A 3-layer GCN with hidden size 512 per
layer, followed by a linear classifier. Each convolutional
layer is followed by ReLU activation and dropout (0.3).
•GAT Encoder.A 2-layer GAT: the first layer uses 8 at-
tention heads (hidden dimension divided by 8), the second
layer uses a single head with hidden dimension 512. A linear
classifier projects the final representation.
•GraphSAGE Encoder.A 2-layer GraphSAGE with hid-
den size 512, followed by a linear classifier. Each layer is
followed by ReLU activation and dropout (0.3).
All models output class probabilities through a log-softmax layer
and are trained with the Adam optimizer (learning rate = 0.01).
Furthermore, for TAPE and TA_E baselines, both original node
featuresℎorigand explanatory features ℎexplare projected to match
the feature dimensionality of the corresponding datasets.
For fair comparison, in Section 5.2.4, we retain the SGGM and
Graph Enrichment components, and replace only the R2CL module
with SimGCL and SGL under identical settings.
C Research Methods
Table 5 shows complete results of GCN on the Cora under com-
binations of deficiencies (NRA, SHA, FDA, SSA). The aggregated
performance in the main paper is derived from full tables like this.
D Prompts
The prompt shown below is used for the initial sample generation,
where the LLM is instructed to produce a research paper in a given
category with its Title and Abstract.
Figure 10: Prompt for initial sample generation.
The following subsections present the prompts used in our frame-
work. For brevity, we provide moderately condensed versions that
preserve all essential instructions. The complete detailed prompts
will be released with the code repository.Note:The following
prompts are shown in theirinstantiated formfor the PubMed dataset
(medical research papers in diabetes categories). For other datasets
used in our experiments (e.g., Cora, Arxiv), we applied the same
templates with dataset-specific terminology.
D.1 Prompt for Initial Generation
Please generate a medical research paper in the category [<cate-
gory>], including a title, an abstract, and keywords.
Step 1: Analyze example papers(<similar_docs_text>) to iden-
tify: – common terms and methodologies specific to [<category>];
– typical clinical research problems in this category; – distinctive
approaches separating [<category>] from other diabetes categories.
Step 2: Generate a paper that– usesEXACTmedical terms
(15–20) from the example papers; – addresses a typical research
problem in [<category>]; – employs characteristic methodologies
of [<category>]; – avoids approaches typical of other categories.Keywords:15–20 terms grouped as: Clinical Methodologies
(5–7); Therapeutic Approaches (5–7); Biomarkers/Indicators (3–4);
Research/Study Types (2–3).
Screening:Outputs are checked for similarity with [<category>]
papers; insufficient alignment will be rejected.
Format constraints:plain text only. Each of Title, Abstract,
Keywords on a single line; exact prefixes “Title:”, “Abstract:”, “Key-
words:” with two spaces before “Abstract:” and “Keywords.”
D.2 Prompt for Refinement with Feedback
You have generated: <generated_paper>. Similarity analysis: <feed-
back>.
Task: Revisethe paper to align more closely with [<category>]
while maintaining originality: – retain Title–Abstract–Keywords
format; – use examples of similar papers as guidance; – extract 15–
20 key termsfrom the revised abstractand place them immediately
after the abstract (two spaces before “Keywords:”).
Format requirements:same as above (plain text, single line
per field, exact prefixes).
D.3 Prompt for Text Modification
Anchor paper (Category: {categories[anchor_category]}):{an-
chor_text}
Similar papers:{similar_texts}
Task:Modify the anchor paper so that: – it clearly remains in its
category; – it is more distinctive from similar papers; – key medical
concepts and methodologies are preserved, but the research focus or
terminology is varied; – output format: Title–Abstract–Keywords
only, no explanations.
D.4 Prompt for Edge Analysis
Anchor paper (Category: {categories[anchor_category]}):{an-
chor_text}Candidate papers:{similar_texts}
Task:Decide which candidates should connect to the anchor
based on: – clinical methodology similarity; – shared research do-
mains or therapeutic targets; – conceptual/medical relationships
(e.g., biomarkers, mechanisms).
Edge rules:– connect same-category papers with strong
methodological proximity; – connect across categories only with
strong overlap; – be selective.
Output format:Paper 1: CONNECT/REMOVE; Paper 2: CON-
NECT/REMOVE; ...
E Datasets
We conduct experiments on three widely used text-attributed cita-
tion graph datasets:Cora,PubMed, andArxiv. The Arxiv dataset
here is a sampled subset of the original OGB-Arxiv. Table 6 sum-
marizes their statistics.
F Additional Experimental Results
The full robustness results of GAT and GraphSAGE are reported
in Appendix (Tables 7 and 8), as a complement to the GCN results
shown in the main paper (Table 4). GRCN and RS-GNN are designed
with GCN backbones, hence they are not applicable to GAT, and
their GAT results are not reported in Table 7.

Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement
Table 5: Full results GCN Cora
NRA SHA FDA SSA train test test_acc
0.0 0.0 0.0 0.00 0.2 0.2 87.98±1.45
0.0 0.0 0.0 0.33 0.2 0.2 87.06±1.50
0.0 0.0 0.0 0.66 0.2 0.2 84.44±1.96
0.0 0.0 0.5 0.00 0.2 0.2 86.51±1.32
0.0 0.0 0.5 0.33 0.2 0.2 85.55±1.47
0.0 0.0 0.5 0.66 0.2 0.2 82.51±2.00
0.0 0.0 0.9 0.00 0.2 0.2 84.29±0.88
0.0 0.0 0.9 0.33 0.2 0.2 81.78±0.69
0.0 0.0 0.9 0.66 0.2 0.2 76.23±1.51
0.0 0.5 0.0 0.00 0.2 0.2 82.77±1.90
0.0 0.5 0.0 0.33 0.2 0.2 81.22±1.35
0.0 0.5 0.0 0.66 0.2 0.2 76.93±2.02
0.0 0.5 0.5 0.00 0.2 0.2 79.85±1.37
0.0 0.5 0.5 0.33 0.2 0.2 76.97±2.55
0.0 0.5 0.5 0.66 0.2 0.2 73.05±1.34
0.0 0.5 0.9 0.00 0.2 0.2 72.24±2.04
0.0 0.5 0.9 0.33 0.2 0.2 68.39±1.65
0.0 0.5 0.9 0.66 0.2 0.2 63.51±1.47
0.0 0.9 0.0 0.00 0.2 0.2 70.28±2.05
0.0 0.9 0.0 0.33 0.2 0.2 68.91±1.63
0.0 0.9 0.0 0.66 0.2 0.2 63.07±2.67
0.0 0.9 0.5 0.00 0.2 0.2 59.78±1.11
0.0 0.9 0.5 0.33 0.2 0.2 58.67±1.98
0.0 0.9 0.5 0.66 0.2 0.2 52.42±3.10
0.0 0.9 0.9 0.00 0.2 0.2 43.36±1.87
0.0 0.9 0.9 0.33 0.2 0.2 41.74±1.27
0.0 0.9 0.9 0.66 0.2 0.2 35.08±1.43
0.5 0.0 0.0 0.00 0.2 0.2 80.30±1.08
0.5 0.0 0.0 0.33 0.2 0.2 80.00±2.20
0.5 0.0 0.0 0.66 0.2 0.2 76.59±2.42
0.5 0.0 0.5 0.00 0.2 0.2 76.52±1.51
0.5 0.0 0.5 0.33 0.2 0.2 75.33±2.87
0.5 0.0 0.5 0.66 0.2 0.2 70.44±2.88
0.5 0.0 0.9 0.00 0.2 0.2 68.59±1.37
0.5 0.0 0.9 0.33 0.2 0.2 65.63±3.05
0.5 0.0 0.9 0.66 0.2 0.2 58.81±2.80
0.5 0.5 0.0 0.00 0.2 0.2 74.96±2.67
0.5 0.5 0.0 0.33 0.2 0.2 72.96±1.80
0.5 0.5 0.0 0.66 0.2 0.2 68.37±3.24
0.5 0.5 0.5 0.00 0.2 0.2 68.37±4.15
0.5 0.5 0.5 0.33 0.2 0.2 66.82±2.08
0.5 0.5 0.5 0.66 0.2 0.2 60.22±3.60
0.5 0.5 0.9 0.00 0.2 0.2 57.41±3.39
0.5 0.5 0.9 0.33 0.2 0.2 54.52±2.56
0.5 0.5 0.9 0.66 0.2 0.2 46.82±1.81
0.5 0.9 0.0 0.00 0.2 0.2 64.37±3.63
0.5 0.9 0.0 0.33 0.2 0.2 65.11±1.62
0.5 0.9 0.0 0.66 0.2 0.2 59.11±3.39
0.5 0.9 0.5 0.00 0.2 0.2 50.67±5.61
0.5 0.9 0.5 0.33 0.2 0.2 54.07±1.63
0.5 0.9 0.5 0.66 0.2 0.2 49.71±3.16
0.5 0.9 0.9 0.00 0.2 0.2 36.30±3.33
0.5 0.9 0.9 0.33 0.2 0.2 33.48±2.05
0.5 0.9 0.9 0.66 0.2 0.2 35.26±3.49
0.9 0.0 0.0 0.00 0.2 0.2 63.33±6.02
0.9 0.0 0.0 0.33 0.2 0.2 58.89±6.02
0.9 0.0 0.0 0.66 0.2 0.2 44.07±2.16
0.9 0.0 0.5 0.00 0.2 0.2 49.26±6.48
0.9 0.0 0.5 0.33 0.2 0.2 43.33±10.18
0.9 0.0 0.5 0.66 0.2 0.2 38.52±2.72
0.9 0.0 0.9 0.00 0.2 0.2 41.85±5.32
0.9 0.0 0.9 0.33 0.2 0.2 36.66±7.72
0.9 0.0 0.9 0.66 0.2 0.2 32.22±1.89
0.9 0.5 0.0 0.00 0.2 0.2 62.96±5.62
0.9 0.5 0.0 0.33 0.2 0.2 62.59±4.60
0.9 0.5 0.0 0.66 0.2 0.2 41.11±4.12
0.9 0.5 0.5 0.00 0.2 0.2 46.67±8.72
0.9 0.5 0.5 0.33 0.2 0.2 44.07±8.95
0.9 0.5 0.5 0.66 0.2 0.2 36.67±2.96
0.9 0.5 0.9 0.00 0.2 0.2 34.81±5.90
0.9 0.5 0.9 0.33 0.2 0.2 30.74±6.37
0.9 0.5 0.9 0.66 0.2 0.2 30.37±4.48
0.9 0.9 0.0 0.00 0.2 0.2 60.37±7.55
0.9 0.9 0.0 0.33 0.2 0.2 51.48±5.42
0.9 0.9 0.0 0.66 0.2 0.2 45.19±5.44
0.9 0.9 0.5 0.00 0.2 0.2 43.33±7.82
0.9 0.9 0.5 0.33 0.2 0.2 37.78±4.77
0.9 0.9 0.5 0.66 0.2 0.2 34.07±5.81
0.9 0.9 0.9 0.00 0.2 0.2 33.34±4.38
0.9 0.9 0.9 0.33 0.2 0.2 29.63±5.24
0.9 0.9 0.9 0.66 0.2 0.2 27.41±4.60Table 6: Statistics of the datasets.
Dataset #Nodes #Edges #Features #Classes
Cora 2,708 5,278 1,433 7
PubMed 19,717 44,324 384 3Arxiv 2,107 1,758 128 10
Table 7: GAT robustness.
AlgorithmClean Acc Worst Acc Avg Acc Norm-AUC
Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv
GAT 88.43 87.19 72.45 77.72 82.64 62.32 83.68 85.03 67.91 0.84 0.85 0.68
DropEdge 86.25 84.57 42.66 42.68 75.36 26.28 64.58 81.00 34.05 0.66 0.82 0.34
DropMsg 88.28 86.35 70.07 77.00 82.43 59.49 83.45 84.57 65.60 0.84 0.85 0.66
Feat. Prop. 87.16 85.73 80.24 76.14 82.18 67.29 82.13 83.95 74.35 0.83 0.84 0.75
Nbr. Mean 87.16 85.69 80.48 75.86 81.99 68.28 82.07 83.84 74.98 0.83 0.84 0.76
ACM-GNN 87.99 81.44 46.32 75.01 71.73 41.08 82.25 77.20 46.06 0.83 0.78 0.46
DGI 83.85 80.73 64.37 60.48 67.11 49.71 74.11 74.92 57.98 0.76 0.76 0.59
TAPE 83.86 83.13 83.30 75.41 74.03 75.95 80.19 78.65 80.38 0.81 0.79 0.81
TA_E 82.74 83.48 82.57 73.98 73.72 74.67 79.01 78.93 79.74 0.80 0.79 0.80
LLM4NG 88.47 88.85 84.51 78.51 85.09 79.21 83.96 87.15 82.19 0.84 0.87 0.8
Ours 90.17 95.94 88.79 81.25 90.96 84.84 86.23 93.64 87.36 0.87 0.94 0.88
Table 8: GraphSAGE robustness.
AlgorithmClean Acc Worst Acc Avg Acc Norm-AUC
Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv Cora Pub. Arxiv
GraphSAGE 88.84 88.06 68.51 76.20 83.07 55.95 83.28 85.72 62.70 0.84 0.86 0.63
DropEdge 89.24 86.41 66.70 77.14 80.91 53.33 83.92 84.04 60.98 0.85 0.84 0.62
DropMsg 89.06 86.41 68.03 75.70 81.12 55.16 83.12 84.14 62.33 0.84 0.85 0.63
Feat. Prop. 88.25 88.59 79.71 77.30 84.14 66.96 83.35 86.32 73.58 0.84 0.86 0.74
Nbr. Mean 88.25 88.68 79.48 76.53 83.78 66.50 83.08 86.17 73.28 0.84 0.86 0.74
ACM-GNN 88.91 87.98 79.09 78.65 84.15 65.21 84.45 86.14 72.57 0.85 0.86 0.73
DGI 70.50 73.01 54.06 48.84 59.14 42.10 59.78 66.35 48.82 0.60 0.67 0.50
RS-GNN 72.09 80.91 68.77 61.77 77.03 57.21 67.66 79.59 63.09 0.65 0.80 0.63
GRCN 83.62 85.83 63.33 62.40 77.73 51.51 74.72 82.46 58.08 0.76 0.83 0.59
TAPE 83.90 78.46 85.22 76.94 70.69 75.83 81.07 74.97 81.98 0.82 0.75 0.83
TA_E 82.59 75.80 83.52 74.80 70.66 74.14 79.57 74.32 80.68 0.80 0.75 0.82
LLM4NG 87.06 90.48 84.99 78.28 86.53 77.88 83.29 88.82 82.15 0.84 0.89 0.83
Ours 90.39 97.04 92.16 81.24 92.85 84.24 86.34 95.23 89.26 0.87 0.96 0.90
For completeness, we also report the performance curves of
GAT and GraphSAGE across Cora, PubMed, and Arxiv datasets in
Figure 11. These results complement the GCN robustness curves
shown in the main paper (Figure 6).

Zhaoyan Wang et al.
Figure 11: Performance curves of GAT and GraphSAGE on Cora, PubMed, and Arxiv.