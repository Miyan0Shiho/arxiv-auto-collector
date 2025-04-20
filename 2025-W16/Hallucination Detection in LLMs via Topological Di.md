# Hallucination Detection in LLMs via Topological Divergence on Attention Graphs

**Authors**: Alexandra Bazarova, Aleksandr Yugay, Andrey Shulga, Alina Ermilova, Andrei Volodichev, Konstantin Polev, Julia Belikova, Rauf Parchiev, Dmitry Simakov, Maxim Savchenko, Andrey Savchenko, Serguei Barannikov, Alexey Zaytsev

**Published**: 2025-04-14 10:06:27

**PDF URL**: [http://arxiv.org/pdf/2504.10063v1](http://arxiv.org/pdf/2504.10063v1)

## Abstract
Hallucination, i.e., generating factually incorrect content, remains a
critical challenge for large language models (LLMs). We introduce TOHA, a
TOpology-based HAllucination detector in the RAG setting, which leverages a
topological divergence metric to quantify the structural properties of graphs
induced by attention matrices. Examining the topological divergence between
prompt and response subgraphs reveals consistent patterns: higher divergence
values in specific attention heads correlate with hallucinated outputs,
independent of the dataset. Extensive experiments, including evaluation on
question answering and data-to-text tasks, show that our approach achieves
state-of-the-art or competitive results on several benchmarks, two of which
were annotated by us and are being publicly released to facilitate further
research. Beyond its strong in-domain performance, TOHA maintains remarkable
domain transferability across multiple open-source LLMs. Our findings suggest
that analyzing the topological structure of attention matrices can serve as an
efficient and robust indicator of factual reliability in LLMs.

## Full Text


<!-- PDF content starts -->

Hallucination Detection in LLMs via Topological Divergence on Attention
Graphs
Alexandra Bazarova1Aleksandr Yugay1Andrey Shulga1Alina Ermilova1Andrei Volodichev1
Konstantin Polev2Julia Belikova2Rauf Parchiev2Dmitry Simakov2Maxim Savchenko2
Andrey Savchenko2Serguei Barannikov1 3Alexey Zaytsev1
Abstract
Hallucination, i.e., generating factually incorrect
content, remains a critical challenge for large lan-
guage models (LLMs). We introduce TOHA, a
TOpology-based HAllucination detector in the
RAG setting, which leverages a topological di-
vergence metric to quantify the structural prop-
erties of graphs induced by attention matrices.
Examining the topological divergence between
prompt and response subgraphs reveals consis-
tent patterns: higher divergence values in spe-
cific attention heads correlate with hallucinated
outputs, independent of the dataset. Extensive
experiments — including evaluation on question
answering and data-to-text tasks — show that our
approach achieves state-of-the-art or competitive
results on several benchmarks, two of which were
annotated by us and are being publicly released
to facilitate further research. Beyond its strong
in-domain performance, TOHA maintains remark-
able domain transferability across multiple open-
source LLMs. Our findings suggest that analyzing
the topological structure of attention matrices can
serve as an efficient and robust indicator of factual
reliability in LLMs.
1. Introduction
Large language models (LLMs) have progressed signif-
icantly in recent years, finding applications in various
fields (Chkirbene et al., 2024). However, these models are
prone to generate so-called hallucinations , i.e., content that
is factually or contextually incorrect (Huang et al., 2023).
Detecting hallucinations is crucial for the safe deployment
of LLMs in sensitive fields since erroneous outputs may
lead to financial losses and seriously harm user trust.
1Skolkovo Institute of Science and Technology2Sber AI Lab
3CNRS, Universite Paris Cite. Correspondence to: Alexandra
Bazarova <a.bazarova@skoltech.ru >.
Preprint. Copyright 2025 by the author(s).
135791113151719212325272931
Head31
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1LayerRAGTruth QA
135791113151719212325272931
Head31
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1CoQAFigure 1. Difference between average TOHA scores for halluci-
nated and grounded samples per attention head/layer, evaluated on
RAGTruth QA and CoQA datasets. A lighter color corresponds
to a greater difference. Green frames highlight the heads that
segregate samples best. The same attention heads assign greater
divergence values to the hallucinated samples in both datasets.
Model: Mistral-7B-Instruct-v0.1.
Both supervised and unsupervised methods address this
issue (Azaria & Mitchell, 2023; Fadeeva et al., 2024). De-
spite the high accuracy of supervised methods, they are
often poorly transferable between different datasets and
tasks (Sky et al., 2024). Additionally, there are very few nat-
ural hallucination datasets publicly available (Zhang et al.,
2023); therefore, time-consuming and expensive annotation
work is necessary to attain real-life hallucination classifiers.
Unsupervised methods, though not suffering from these is-
sues, tend to be computationally intensive, as their inference
often includes generating multiple model outputs (Manakul
et al., 2024; Chen et al., 2024; Farquhar et al., 2024).
We address the above challenges by introducing TOHA, a
novel training-free method to detect LLMs’ hallucinations
in the retrieval-augmented generation (RAG) setting (Gao
et al., 2023). TOHA focuses on attention maps in LLMs
based on the hypothesis that their structure may reflect the
presence of hallucinations. Since the graph representation
of attention maps has proven effective for various NLP
tasks (Kushnareva et al., 2021; Tulchinskii et al., 2023),
TOHA also employs it to obtain hallucination scores. The
1arXiv:2504.10063v1  [cs.CL]  14 Apr 2025

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
previous works (Proskurina et al., 2023a; Cherniavskii et al.,
2022) have analyzed attention graphs rather naively by build-
ing a simple supervised classifier on top of graph properties;
our experiments (see Appendix D) reveal that this apply-
ing this approach to hallucination detection falls short of
achieving the desired accuracy.
TOHA takes the next step in developing a more advanced
graph analysis tool. We assume that the presence of hallu-
cinations in the RAG setting may be indicated by the topo-
logical dissimilarity of the response subgraph concerning
the prompt one. Simply put, a hallucinated response likely
includes information that is not present in the prompt, which
introduces its unique structure to the text. As a result, we
expect the topology of hallucinated responses to differ from
that of grounded ones. We draw an analogy between the
considered idea of measuring the dissimilarity between two
subgraphs and the Manifold Topology Divergence (Baran-
nikov et al., 2021) and then propose its counterpart in the
graph setting. For the latter, we prove several properties
regarding its continuity and boundedness to ensure it can be
considered a reasonable hallucination score.
Our contributions are the following:
•We propose TOHA, a training-free method based on
the topological divergences of attention graphs. TOHA
demonstrates strong in-domain performance and main-
tains domain transferability across different tasks. Our
method offers an efficient and practical solution, work-
ing an order of magnitude faster than unsupervised
methods of comparable performance.
•The existence of hallucination-aware attention heads is
discovered: calculating topological divergences from
just a few specific heads is enough for reliable halluci-
nation detection, irrespective of the dataset.
•We release two novel datasets containing outputs from
several popular open LLMs, annotated for hallucina-
tion, to facilitate benchmarking and further research in
the field.
•For all considered benchmarks, TOHA demonstrates
competitive results in the unsupervised and transfer
settings for several open-sourced LLMs, including
LLaMA-2-13B.
2. Related works
Hallucination detection methods. The problem of hallu-
cinations in LLMs has attracted significant attention, leading
to the development of methods for their detection (Zhang
et al., 2023; Huang et al., 2023; Wang et al., 2024). Several
approaches leverage a model’s internal states to detect hal-
lucinations in a supervised manner. For instance, (Azaria& Mitchell, 2023; Sky et al., 2024) demonstrated that clas-
sifiers trained on hidden states effectively identify factual
errors. In contrast, (Chuang et al., 2024) introduced look-
back ratio features from attention weights to train a linear
classifier for hallucination detection. However, supervised
methods depend on annotated datasets, which are costly to
create and may not generalize well across diverse tasks (Sky
et al., 2024). Unsupervised approaches instead exploit a
model’s uncertainty, using token or sequence probabilities
to estimate confidence during generation (Kadavath et al.,
2022; Fadeeva et al., 2024). Another line of work analyzes
inconsistencies across multiple responses to the same input.
For example, the INSIDE method (Chen et al., 2024) quan-
tifies hallucinations by measuring differential entropy in the
embedding space, while semantic entropy (Han et al., 2024;
Farquhar et al., 2024) assesses uncertainty by computing
entropy over clusters of semantically similar responses. For
black-box models, where internal states and token probabil-
ities are inaccessible, textual analysis methods have been
developed (Manakul et al., 2024; Xiong et al., 2024). While
considering multiple generations of responses can provide
valuable insights for hallucination detection, it increases
computational costs significantly and may not scale effi-
ciently for real-time applications.
Evaluation. Hallucination detection methods are typically
evaluated in tasks such as summarization (Narayan et al.,
2018), open-ended text generation (Lebret et al., 2016),
and question answering (Rajpurkar et al., 2016). In struc-
tured settings, such as multiple-choice questions in Truth-
fulQA (Lin et al., 2022) or True/False statements (Azaria &
Mitchell, 2023), automatic methods allow for direct com-
parison with reference answers and the computation of clas-
sification metrics. In contrast, hallucinations in open-ended
responses are usually annotated by human experts, as seen in
FELM (Zhao et al., 2023) and RAGTruth (Niu et al., 2023).
Since this process is costly and time-consuming, recent stud-
ies have leveraged LLMs to generate annotations (Lin et al.,
2022; Min et al., 2023), demonstrating strong agreement
with human judgments. Despite these advances, most pub-
licly available datasets provide hallucination annotations
for black-box models such as GPT-3 (Brown et al., 2020),
making them unsuitable for studying hallucination detec-
tion based on a model’s internal states. To the best of our
knowledge, RAGTruth (Niu et al., 2023) is the only dataset
that includes annotated outputs from open-source models
such as LLaMA (Touvron et al., 2023) and Mistral (Jiang
et al., 2023) in the RAG setting.
Topological Data Analysis (TDA) in NLP. Topological
Data Analysis is a mathematical framework that analyzes
multi-scale intrinsic structural patterns in data using princi-
ples from topology and computational geometry (Chazal &
Michel, 2017; Hensel et al., 2021). The application of TDA
2

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
tools in NLP has gained increasing attention (Uchendu &
Le, 2024). For instance, (Tulchinskii et al., 2024) applied
the Persistent Homology Dimension estimator of intrinsic
dimensionality to CLS embeddings of texts to detect artifi-
cially generated content. Other studies have also explored
calculating topological features from transformer attention
matrices to assess uncertainty (Kostenok et al., 2023) or per-
form grammatical acceptability classification (Proskurina
et al., 2023b). In these studies, attention matrices were
treated as weighted graphs, and TDA features of these
graphs were employed to train simple classifiers on top
of them.A
3. Background
3.1. Attention matrix as a weighted graph
Modern LLMs are mainly based on the self-attention mech-
anism, introduced in (Vaswani et al., 2017). This mecha-
nism enables models to dynamically assign varying levels
of importance to different parts of the input sequence when
generating the output.
LetX∈Rn×dbe a matrix consisting of d-dimensional
representations of ntokens, WQ, WK, WN∈Rd×dbe
trainable projection matrices.
Given a set of queries Q=XW Q∈Rn×d, a set of
keys K=XW K∈Rn×d, and corresponding values
V=XW V∈Rn×d, the attention mechanism calculates a
weighted sum of the values as follows:
Attention( Q, K, V ) = softmaxQKT
√
d
V. (1)
Each entry wijin the attention matrix
W = softmaxQKT
√
d
(2)
captures how strongly token iattends to token j,i≥jfor a
decoder, with larger wijindicating closer relationship.
An attention matrix Wcan be represented as a complete
weighted graph G, whose vertices are the tokens and whose
edges carry weights wij. From the perspective of topologi-
cal data analysis, however, it is more convenient to interpret
these weights as pseudo-distances rather than correlation
measures. Hence, we reassign the edge weights of such a
graph to be equal to 1−wij. We refer to such graphs as
attention graphs .
3.2. Manifold Topology Divergence
Given an attention matrix for the (prompt + response) text,
we construct the pseudo-distances graph, imitating a data
manifold of the text, and study its relation with the weighted
subgraph, imitating the data submanifold of the prompt.Proposed in (Barannikov et al., 2021), MTop-Div( M, N )
is a topological measure for comparing two data manifolds
MandNapproximated by point clouds MandN. This di-
vergence is based on the Cross-Barcode( M, N )tool. Here,
we briefly describe these objects; see Appendix A for more
details and motivation.
To measure how far the two manifolds MandNare
from being identical, (Barannikov et al., 2021) considered
the independent topological features of the quotient space
M/(M ∩ N ). The triviality of this space would entail that
natural maps between the homology groups
φr:H∗(M ∩ N )→H∗(M),
φp:H∗(M ∩ N )→H∗(N) (3)
are isomorphisms; simply put, the more trivial M/(M∩N )
is, the closer the two manifolds are to being identical. To
build the counterpart of this construct for manifolds rep-
resented by point clouds, the pair (M ∩ N )⊆ M is re-
placed with the analogous N ⊆ (M ∪ N ). Taking the quo-
tient is realized by setting the pairwise distances within the
point cloud Nto zero. Denoting by m(M∪N)/Nthe result-
ing matrix of pairwise distances, Cross-Barcode i(M, N )
is the i-th homology barcode (Barannikov, 1994) of the
Vietors-Rips simplicial complex V Rα(M∪N, m (M∪N)/N).
MTop-Div( M, N ), in turn, is the sum of interval lengths of
Cross-Barcode i(M, N ). In the original paper, the authors
considered i= 1.
4. Method
LetVXandEXdenote the vertex and edge sets of a graph
X, respectively. In the natural language generation process,
two subsets of an attention graph Gvertex set naturally
stand out. The first, P, represents the prompt tokens. The
second, R=VG\P, corresponds to the response tokens.
An example of an attention graph and corresponding vertex
subsets is illustrated in Figure 2b.
To evaluate the probability of hallucination in the RAG set-
ting, we would like to estimate how much of the “indepen-
dent” knowledge is captured in the response concerning the
prompt. Intuitively, when a hallucination occurs, it means
that the information in the response was not presented in the
prompt. We expect the corresponding vertex set R, along
with its edges, to modify the structure of Gin an essentially
non-trivial way, resulting in the appearance of independent
topological features.
Our method relies on the observation that estimating the
non-triviality of the response concerning the prompt is anal-
ogous to the construction of MTop-Div . Indeed, the re-
sulting MTop-Div( M, N )measures how non-trivial the
topological structure of M∪Nis concerning N. However,
MTop-Div was designed for point clouds, and certain of
3

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Figure 2. a) An attention map. Blue denotes the prompt tokens, and green is the response ones. b) The corresponding attention graph
G. Prompt tokens Pare located on the left, response tokens R— on the right. To keep the figure neat, we only plot the edges with an
attention score of no less than 0.15. c) The minimum spanning forest attaching RtoP.
its properties are based on the fact that MandNlie inRn.
The structure of our task is different, as the graphs are not
metric spaces. We reformulate MTop-Div for the graph
context below, establishing its properties in a novel setting.
4.1.MTop-Div for attention graphs
By the analogy formulated above, we develop
MTop-Div( R, P), where RandPare the response
and the prompt vertex sets in the attention graph G.
Hereinafter, we refer to RandPascomplementary
vertex sets so that the union of these vertex sets and the
edges between all these vertices comprises the complete
graph G. We next set to zero the edge weights between
thePvertices, denote w(R∪P)/Pthe resulting matrix of
edge weights, define Cross-Barcode i(R, P)as the i-th
homology barcode of the Vietoris-Rips simplicial complex
V Rα(G, w (R∪P)/P), and MTop-Div( R, P), as the sum of
interval lengths of Cross-Barcode i(R, P).
Taking into account the specifics of our task, we consider
Cross-Barcode( R, P)for the homology group H0instead
ofH1, as the resulting MTop-Div( R, P)would thus be
more interpretable. Indeed, the bars in the H0barcode
of a weighted graph correspond to edges of the minimum
spanning tree (MST) of this graph (Tulchinskii et al., 2023).
Similarly, we show that our score equals the length of the
minimum spanning forest (MSF) attaching RtoP.
Basic properties of MTop-Div for attention graphs.
Now we consider specific properties for our adaptation of
MTop-Div( R, P).
Proposition 4.1. The following holds for any attention
graph Gand its complementary vertex subsets P, R⊂VG.
•MTop-Div( R, P)value equals the length of the MSF
attaching RtoP.•Let the natural norm on the cross-barcodes be defined
as follows:
∥Cross-Barcode 0∥B= max
[bj,dj]∈Cross-Barcode 0(dj−bj).
(4)
The norm of Cross-Barcode 0(R, P)lays in the inter-
val[0,1]:
0≤ ∥Cross-Barcode 0(R, P)∥B≤1. (5)
•The divergence itself is bounded by
0≤MTop-Div( R, P)≤ |R|. (6)
The second and third statements are immediately obtained
from the properties of an attention matrix: all its weights lie
between 0and1.
Proposition 4.2. LetGbe an attention graph, P, R⊂VG—
a pair of complementary vertex subsets.
1.(Continuity of MTop-Div(R,P)) . If the weights of
Gchange by no more than ε, then the corresponding
MTop-Div( R, P)changes by no more than δ=ε|R|.
2.(Exact sequence). For any α, the following sequence of
natural maps of homology groups is exact
(Z/2Z)|P|r2− →H0(V Rα(G))r1− →
r1− →H0(V Rα(G, w (R∪P)/P))r0− →0.
See proof in Appendix A.
4.2. Universal heads
We hypothesize, inspired by prior investigations in LLM
interpretability (V oita et al., 2019; Gould et al., 2024), that
particular attention heads exhibit distinct patterns related to
hallucinations. To identify such heads, we analyzed head-
specific topological divergences as follows.
4

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Denote by hijthej-th attention head from the layer i. For
the specific data sample sand head hij, letGs
ijbe the
corresponding attention graph, Ps
ij, Rs
ij— its prompt and
response vertex subsets. Let
dij(s) =1
|Rs
ij|MTop-Div( Rs
ij, Ps
ij).
We examined typical values of this score for different heads
and layers. The average distance between hallucinated and
grounded examples from the train data for each model’s
head is the following:
∆ij=1
|Shallu|X
s∈Shalludij(s)−1
|Sgr|X
s∈Sgrdij(s),
where Shallu stands for all hallucinated samples from the
training set, and Sgrstands for all grounded training samples.
The obtained differences are displayed in Figure 3 for three
datasets.
For each model-dataset pair, we highlighted the top 4 atten-
tion heads that demonstrated the highest distance between
the divergences of hallucinated and non-hallucinated sam-
ples. It can be seen that top heads overlap: for both models,
two heads — (19, 3), (8, 19) for Mistral-7B, and (12, 16),
(9, 26) for Llama-2-7B — are present across all datasets.
4.3. TOHA
The existence of universal hallucination patterns in the at-
tention heads underlies our method. In this subsection, we
present our procedure for hallucination detection TOHA.
Here, we used the notation from the previous paragraph.
Our method employs two “probe” sets Sh, Sgcontaining
annotated data to arrange model heads from the most segre-
gating to the least segregating based on ∆ijvalues, where
ijis a head index in a model. During testing, hallucination
scores are computed as the average topological divergence
values from the top Noptheads, where Noptis a hyperpa-
rameter selected on the validation set V. In our experiments,
we consider Nopt≤Nmax, Nmax= 6to minimize the re-
quired amount of computations. The pseudocode for the
proposed approach during head selection and inference is
presented in Algorithm 1.
5. Experiments
Datasets. In our paper, we used three datasets:
RAGTruth (Niu et al., 2023), CoQA (Reddy et al., 2019),
and SQuAD (Rajpurkar et al., 2016). The RAGTruth dataset
consists of manually annotated responses of several LLMs
in the RAG setting. It includes hallucinations in three tasks:
question answering (QA), text summarization (Summ), and
data-to-text writing (Data2txt). The annotations are word-
level; we, in turn, predict response-level labels, consideringAlgorithm 1 TOHA algorithm
Require: dij(s)— divergence between prompt and re-
sponse for a sample s,Sh, Sg— probe sets; V— valida-
tion set of annotated samples {(s, ys)}s∈V;T— test set.
procedure TOHA HEADS SELECTION
∆ij←1
|Sh|P
s∈Shdij(s)−1
|Sg|P
s∈Sgdij(s)
H←sort(hij,key= ∆ ij,ascending =False)
N, N opt←1,1
Hsubset←∅
AUROC max←0
ps= 0, s∈V ▷ Initialize hallucination scores.
while N≤Nmaxdo ▷Optimal heads selection.
Hsubset←Hsubset∪ {hN}
fors∈Vdo
ps←N−1
Nps+1
NdhN(s)
end for
AUROC ←AUROC ({ys}s∈V,{ps}s∈V)
ifAUROC >AUROC maxthen
AUROC max←AUROC
Nopt←N
end if
N←N+ 1
end while
end procedure
procedure TOHA PREDICTION
fors∈Tdo ▷Prediction on the test set.
ps←1
NoptNoptP
i=1dhi(s)
end for
end procedure
a response hallucinated if it contains at least one hallucina-
tion span.
CoQA and SQuAD are both question-answering bench-
marks. For all the considered models, we used questions
from these datasets to sample responses from LLMs and
then annotated the responses in an automated manner using
GPT-4o (Hurst et al., 2024). We only annotate SQuAD re-
sponses for LLaMA-3.1-8B and Qwen2.5-7B due to time
and budget limitations.
To estimate the correctness of GPT-4o annotations, we eval-
uated their consistency with the labels produced by human
experts. Our findings indicated that the consistency between
the experts and GPT-4o is sufficient to use the latter for an-
notation, which aligns well with previous works (Bavaresco
et al., 2024).
We are contributing by releasing the obtained datasets to the
public. Additionally, we provide a straightforward annota-
5

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
13579111315171921232527293131
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1Mistral-7B-Instruct-v0.1RAGTruth QA
13579111315171921232527293131
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1RAGTruth Summ
13579111315171921232527293131
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1CoQA
13579111315171921232527293131
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1Llama-2-7b-chat-hf
13579111315171921232527293131
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1
13579111315171921232527293131
29
27
25
23
21
19
17
15
13
11
9
7
5
3
1
0.02
0.000.020.040.06
0.01
0.000.010.020.03
0.10
0.05
0.000.050.100.150.20
0.02
0.01
0.000.010.020.03
0.010
0.005
0.0000.0050.010
0.1
0.00.10.20.3
Figure 3. ∆ijvalues for the datasets RAGTruth QA, RAGTruth Summ, CoQA. A lighter color corresponds to a greater value. Vertical
axis corresponds to the layer number, horizontal — to the head number. The heads that segregate samples best are highlighted with green
frames. Model names for a row are on the left side.
tion procedure to support further research on hallucination
detection. For more details, see Appendix C.
Models. We used five popular open-source LLMs:
LLaMA-2-7B-chat, LLaMA-2-13B-chat, LLaMA-3.1-
8B-Instruct, Mistral-7B-Instruct-v0.1, and Qwen2.5-7B-
Instruct. Note that the RAGTruth dataset does not contain
responses for LLaMA-3.1-8B and Qwen-2.5-7B; therefore,
we only conducted experiments on SQuAD and CoQA for
these models.
Baselines. We compare TOHA with six baselines, in-
cluding two supervised methods, such as linear probe and
attention-pooling probe (Sky et al., 2024), and four unsu-
pervised methods: tokenwise entropy (Fadeeva et al., 2024),
semantic entropy (Farquhar et al., 2024), INSIDE (Chen
et al., 2024), and SelfCheckGPT (Manakul et al., 2024). Ap-
pendix E provides information on implementation details.
Hallucination detection results. Tables 1–2 demonstrate
the results of our experiments. As probe and validation
datasets for TOHA, we used the training subset of the
RAGTruth QA benchmark, split into three parts. All meth-
ods were evaluated on the same fixed test set for each dataset.
Table 2 contains only two datasets since, as we mentioned,
the RAGTruth dataset does not contain annotated samples
for this model.As anticipated, supervised methods outperform unsuper-
vised ones, achieving the ROC-AUC score close to 1 on the
SQuAD and CoQA datasets. Most importantly, we would
like to highlight the performance of TOHA, which stands out
among unsupervised approaches. It consistently achieves
the best or second-best results across multiple datasets. Its
main competitor in the unsupervised category is SelfCheck-
GPT. However, it is worth noting that SelfCheckGPT re-
lies on additional model generations (following the original
paper, we used 20in our experiments), making it compu-
tationally inefficient. At the same time, TOHA requires
significantly fewer extra computations.
Transfer results. The previous paragraph has shown that
unsupervised methods, including the proposed TOHA, gen-
erally achieve significantly lower detection quality than su-
pervised methods. However, it has previously been noted
that supervised methods may experience considerable drops
in quality when transferred to other datasets (Sky et al.,
2024). So, we carried out the transferability experiments.
Their results are presented in Table 3. We use the test parts
of all datasets for method comparisons to prevent data leak-
age since the head selection for TOHA was performed on
the training subset of RAGTruth QA. Here, we only provide
the results for TOHA among all unsupervised methods, as
the metrics in the unsupervised setting are duplicated from
Table 1.
6

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Table 1. ROC AUC (↑)of hallucination detection techniques for three LLMs. The best results for each model are highlighted in bold, and
the second best are underlined . The supervised and unsupervised methods are delimited by a horizontal line. TOP results for supervised
methods are presented in black , while for unsupervised ones – in green .
MethodRAGTruth RAGTruth RAGTruth CoQA
QA Summ Data2txt
LLaMA-2-7B
Attention-pooling probe 0.652 0.640 0.752 0.933
Linear probe 0.731 0.638 0.744 0.800
SelfCheckGPT 0.646 0.665 0.618 0.781
Semantic entropy 0.528 0.572 0.444 0.743
Tokenwise entropy 0.607 0.595 0.533 0.724
INSIDE 0.474 0.526 0.585 0.697
TOHA (ours) 0.646 0.638 0.573 0.858
LLaMA-2-13B
Attention-pooling probe 0.768 0.573 0.573 0.936
Linear probe 0.705 0.472 0.589 0.696
SelfCheckGPT 0.675 0.508 0.559 0.867
Semantic entropy 0.581 0.536 0.359 0.831
Tokenwise entropy 0.626 0.588 0.514 0.659
INSIDE 0.557 0.569 0.511 0.518
TOHA (ours) 0.734 0.570 0.729 0.800
Mistral-7B
Attention-pooling probe 0.791 0.661 0.685 0.978
Linear probe 0.841 0.714 0.69 0.922
SelfCheckGPT 0.709 0.600 0.63 0.941
Semantic entropy 0.543 0.558 0.431 0.861
Tokenwise entropy 0.701 0.598 0.445 0.759
INSIDE 0.652 0.558 0.427 0.766
TOHA (ours) 0.720 0.625 0.560 0.867
TOHA outperforms both supervised methods on almost all
tasks, except for several cases where our method demon-
strates second-best results. This observation aligns with our
prior hypothesis. Unlike supervised methods, TOHA also
demonstrates significantly better results than random for all
the datasets and models.
What do hallucination patterns look like? As shown
in Section 4, the topological divergences that we employ
characterize the MSF that attaches the vertices Rof the
response to the vertices Pof the prompt. For hallucination-
aware heads, which consistently assign greater scores to
hallucinated samples, we explored MSF patterns that are
typical for hallucinated and grounded samples. The results
for Mistral-7B are presented in Figure 5 (see Appendix D).
We identified two popular patterns among hallucinated sam-
ples: attention to the utility token <s> corresponding to the
prompt beginning and “entanglement”. The former is an
intuitive hallucination sign: instead of directing attention to
meaningful parts of the context, the model focuses on a util-
ity token. The “entanglement” — and, by the entanglement,
we mean that many tokens seem to be linked to randomlocations in the prompt — also allows intuitive explanation.
When the attention scores of a token are distributed evenly
across an entire text (which can be interpreted as a token’s
uncertainty), many small weights in the attention matrix
become associated with it, resulting in a greater MTop-Div
value. This distribution of attention weights would entail
that the edge in MSF connected to that token is equally
likely to attach to any part of the query, creating the “entan-
glement”.
Inference time. We measure the inference time of the top-
performing baselines and TOHA on the NVIDIA L40 GPU
over10iterations with 3warmup -iterations. In our experi-
ments, we use 16random samples from RAGTruth QA. For
methods that require additional generations, we set N= 20 ,
following the original papers (Manakul et al., 2024; Chen
et al., 2024).
Table 4 indicates that supervised models are the most effi-
cient, demonstrating two of the lowest inference times. In
contrast, using SelfCheckGPT and semantic entropy is the
most time-consuming. TOHA is significantly ahead of the
top unsupervised methods in performance, being about 10
7

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Table 2. ROC AUC (↑)of hallucination detection techniques for
LLaMA-3.1-8B. The supervised and unsupervised methods are
delimited by a horizontal line. TOP results for supervised methods
are presented in black , while for unsupervised ones – in green .
Method SQuAD CoQA
LLaMA-3.1-8B
Attention-pooling probe 0.947 0.914
Linear probe 0.973 0.855
SelfCheckGPT 0.814 0.925
Semantic entropy 0.558 0.886
Tokenwise entropy 0.559 0.459
INSIDE 0.496 0.636
TOHA (ours) 0.883 0.703
Qwen2.5-7B
Attention-pooling probe 0.850 0.661
Linear probe 0.808 0.695
SelfCheckGPT 0.642 0.769
Semantic entropy 0.688 0.742
Tokenwise entropy 0.477 0.640
INSIDE 0.582 0.566
TOHA (ours) 0.849 0.640
times faster than them.
6. Conclusion
This paper proposes TOHA (Algorithm 1) — a novel hal-
lucination detection method based on the topological diver-
gences of attention maps. In the core of TOHA lies our
observation that specific attention heads demonstrate the
same patterns in the presence of hallucinations, irrespective
of the dataset. The hallucination scores obtained by TOHA
are the average topological divergences from these heads.
For these divergences, we prove several properties regard-
ing their continuity and boundedness to ensure they can be
considered reasonable hallucination scores. Extensive exper-
iments have demonstrated that our method performs on par
with or outperforms several state-of-the-art unsupervised
baselines, including SelfCheckGPT (Manakul et al., 2024).
Moreover, TOHA is significantly more computationally effi-
cient than most, working an order of magnitude faster than
SelfCheckGPT and semantic entropy. As for the supervised
baselines, even though the proposed method does not reach
their performance on the in-domain task, we showed that
TOHA consistently outperforms them in the domain transfer
setting. This capability of TOHA is of special importance
for real-world applications, as user requests to LLMs may
be much more diverse and complex than specific bench-
marks. Thus, TOHA is a strong alternative to the existing
methods.
Besides, we annotated two novel datasets containing hal-
lucinated and grounded responses of several LLMs in theTable 3. ROC AUC values for transfer of TOHA (unsupervised)
and supervised methods’ comparison for three LLMs for subsets of
RAGTruth QA, Summ, and Data2txt.TOP-1 results are highlighted
with bold font , while TOP-2 are underlined .
MethodTOHA Linear Attn.-pool
(ours) probe probe
Train Test
Mistral-7B
QASumm 0.625 0.674 0.596
Data2txt 0.56 0.548 0.468
SummQA 0.72 0.737 0.720
Data2txt 0.56 0.502 0.502
Data2txtQA 0.72 0.618 0.468
Summ 0.625 0.348 0.478
LLaMA-2-7B
QASumm 0.638 0.679 0.596
Data2txt 0.573 0.496 0.520
SummQA 0.646 0.646 0.554
Data2txt 0.573 0.532 0.476
Data2txtQA 0.646 0.575 0.601
Summ 0.638 0.665 0.590
LLaMA-2-13B
QASumm 0.570 0.463 0.553
Data2txt 0.729 0.633 0.642
SummQA 0.734 0.507 0.563
Data2txt 0.729 0.502 0.652
Data2txtQA 0.734 0.479 0.422
Summ 0.570 0.502 0.456
Table 4. The comparison of methods’ inference time in millisec-
onds. The measurements were obtained using Mistral-7B. TOP-1
results are highlighted with bold font , while TOP-2 are underlined .
The supervised and unsupervised methods are delimited by a hori-
zontal line.
Time, ms
Inference (0.68±0.11)·105
Linear probe 0.16±0.02
Attention-pooling probe 0.380±0.166
SelfCheckGPT (1.46±0.06)·106
Semantic entropy (1.45±0.06)·106
Tokenwise entropy (0.159±0.001)·104
TOHA (1.82±0.18)·105
question-answering (QA) task. The obtained datasets will be
released later to facilitate further research in the field. The
code of the proposed method and the considered baselines
will also be publicly available.
Impact statement. By introducing TOHA, we aim to ad-
dress the following concerns:
•Recent breakthroughs in the NLP field are adopted ev-
8

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
erywhere, and the fear of missing out on new technol-
ogy drives people without proper expertise to integrate
LLMs in every possible aspect of their work. However,
LLMs’ tendency for hallucinations is often neglected,
which, in the best case, can incur financial losses and,
in the worst — harm someone’s life (e.g., in domains
such as healthcare and jurisprudence). Our work con-
tributes towards the safe application of LLMs, thus
facilitating broader adoption and integration of reliable
AI technologies across various sectors.
•Transformer architecture is a core element of the most
modern state-of-the-art NLP models. Our research
offers deeper interpretability into how LLMs process
information, shedding new light on the role of specific
attention heads and their link to factual errors.
•Topological data analysis is usually seen as some kind
of exotic. However, recent results show the effective-
ness of TDA approaches, and therefore, we aim to shed
more light on these methods in the hope that one day
they will become a part of a researcher’s regular skill
set.
•Despite the importance of hallucination detection, there
exists a lack of annotated and validated datasets as well
as pipelines for the generation of such datasets. With
our work, several new datasets would make the prob-
lem more accessible and well-defined. Moreover, re-
search would quickly produce new annotated datasets
that would help to train and validate hallucination de-
tection models.
References
Azaria, A. and Mitchell, T. The internal state of an LLM
knows when it’s lying. In Bouamor, H., Pino, J., and Bali,
K. (eds.), Findings of the Association for Computational
Linguistics: EMNLP 2023 , pp. 967–976, Singapore, De-
cember 2023. Association for Computational Linguistics.
Barannikov, S., Trofimov, I., Sotnikov, G., Trimbach, E.,
Korotin, A., Filippov, A., and Burnaev, E. Manifold
topology divergence: a framework for comparing data
manifolds. Advances in neural information processing
systems , 34:7294–7305, 2021.
Barannikov, S. A. The framed Morse complex and its in-
variants, December 1994. ISSN 2472-4912.
Bavaresco, A., Bernardi, R., Bertolazzi, L., Elliott, D.,
Fern ´andez, R., Gatt, A., Ghaleb, E., Giulianelli, M.,
Hanna, M., Koller, A., Martins, A. F. T., Mondorf, P.,
Neplenbroek, V ., Pezzelle, S., Plank, B., Schlangen, D.,
Suglia, A., Surikuchi, A. K., Takmaz, E., and Testoni,A. LLMs instead of human judges? A large scale em-
pirical study across 20 NLP evaluation tasks. CoRR ,
abs/2406.18403, 2024.
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
Advances in neural information processing systems , 33:
1877–1901, 2020.
Chazal, F. and Michel, B. An introduction to topological
data analysis: Fundamental and practical aspects for data
scientists. Frontiers in Artificial Intelligence , 4, 2017.
Chen, C., Liu, K., Chen, Z., Gu, Y ., Wu, Y ., Tao, M., Fu, Z.,
and Ye, J. INSIDE: LLMs’ internal states retain the power
of hallucination detection. In The Twelfth International
Conference on Learning Representations , 2024.
Cherniavskii, D., Tulchinskii, E., Mikhailov, V ., Proskurina,
I., Kushnareva, L., Artemova, E., Barannikov, S., Pio-
ntkovskaya, I., Piontkovski, D., and Burnaev, E. Accept-
ability judgements via examining the topology of atten-
tion maps. In Goldberg, Y ., Kozareva, Z., and Zhang, Y .
(eds.), Findings of the Association for Computational Lin-
guistics: EMNLP 2022 , pp. 88–107, Abu Dhabi, United
Arab Emirates, December 2022. Association for Compu-
tational Linguistics.
Chkirbene, Z., Hamila, R., Gouissem, A., and Devrim, U.
Large language models (LLM) in industry: A survey
of applications, challenges, and trends. In 2024 IEEE
21st International Conference on Smart Communities:
Improving Quality of Life using AI, Robotics and IoT
(HONET) , pp. 229–234. IEEE, 2024.
Chuang, Y .-S., Qiu, L., Hsieh, C.-Y ., Krishna, R., Kim,
Y ., and Glass, J. Lookback lens: Detecting and mitigat-
ing contextual hallucinations in large language models
using only attention maps. In Proceedings of the 2024
Conference on Empirical Methods in Natural Language
Processing , pp. 1419–1436, 2024.
Fadeeva, E., Rubashevskii, A., Shelmanov, A., Petrakov,
S., Li, H., Mubarak, H., Tsymbalov, E., Kuzmin, G.,
Panchenko, A., Baldwin, T., et al. Fact-checking the out-
put of large language models via token-level uncertainty
quantification. arXiv preprint arXiv:2403.04696 , 2024.
Farquhar, S., Kossen, J., Kuhn, L., and Gal, Y . Detecting
hallucinations in large language models using semantic
entropy. Nature , 630(8017):625–630, 2024.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., and Wang, H. Retrieval-augmented generation
for large language models: A survey. arXiv preprint
arXiv:2312.10997 , 2023.
9

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Gould, R., Ong, E., Ogden, G., and Conmy, A. Successor
heads: Recurring, interpretable attention heads in the wild.
InThe Twelfth International Conference on Learning
Representations , 2024.
Han, J., Kossen, J., Razzak, M., Schut, L., Malik, S. A., and
Gal, Y . Semantic entropy probes: Robust and cheap hal-
lucination detection in LLMs. In ICML 2024 Workshop
on Foundation Models in the Wild , 2024.
Hensel, F., Moor, M., and Rieck, B. A survey of topolog-
ical machine learning methods. Frontiers in Artificial
Intelligence , 4:681108, 2021.
Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H.,
Chen, Q., Peng, W., Feng, X., Qin, B., et al. A survey on
hallucination in large language models: Principles, taxon-
omy, challenges, and open questions. ACM Transactions
on Information Systems , 2023.
Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh,
A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A.,
Radford, A., et al. GPT-4o system card. arXiv preprint
arXiv:2410.21276 , 2024.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G.,
Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint
arXiv:2310.06825 , 2023.
Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain,
D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma,
N., Tran-Johnson, E., et al. Language models (mostly)
know what they know. arXiv preprint arXiv:2207.05221 ,
2022.
Kostenok, E., Cherniavskii, D., and Zaytsev, A. Uncer-
tainty estimation of transformers’ predictions via topo-
logical analysis of the attention matrices. arXiv preprint
arXiv:2308.11295 , 2023.
Kuhn, L., Gal, Y ., and Farquhar, S. Semantic uncertainty:
Linguistic invariances for uncertainty estimation in nat-
ural language generation. In The Eleventh International
Conference on Learning Representations , 2023.
Kushnareva, L., Cherniavskii, D., Mikhailov, V ., Artemova,
E., Barannikov, S., Bernstein, A., Piontkovskaya, I., Pi-
ontkovski, D., and Burnaev, E. Artificial text detection
via examining the topology of attention maps. In Pro-
ceedings of the 2021 Conference on Empirical Methods
in Natural Language Processing , pp. 635–649, 2021.
Lebret, R., Grangier, D., and Auli, M. Neural text genera-
tion from structured data with application to the biogra-
phy domain. In Proceedings of the 2016 Conference on
Empirical Methods in Natural Language Processing , pp.
1203–1213, 2016.Lin, S., Hilton, J., and Evans, O. TruthfulQA: Measur-
ing how models mimic human falsehoods. In Proceed-
ings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pp.
3214–3252, 2022.
Lin, Z., Trivedi, S., and Sun, J. Generating with confidence:
Uncertainty quantification for black-box large language
models. Transactions on Machine Learning Research ,
2024. ISSN 2835-8856.
Manakul, P., Liusie, A., and Gales, M. SelfCheckGPT:
Zero-resource black-box hallucination detection for gen-
erative large language models. In The 2023 Conference
on Empirical Methods in Natural Language Processing ,
2024.
Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W.-t., Koh, P.,
Iyyer, M., Zettlemoyer, L., and Hajishirzi, H. Factscore:
Fine-grained atomic evaluation of factual precision in
long form text generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Language
Processing , pp. 12076–12100, 2023.
Narayan, S., Cohen, S., and Lapata, M. Don’t give me the
details, just the summary! Topic-aware convolutional
neural networks for extreme summarization. In 2018
Conference on Empirical Methods in Natural Language
Processing , pp. 1797–1807. Association for Computa-
tional Linguistics, 2018.
Niu, C., Wu, Y ., Zhu, J., Xu, S., Shum, K., Zhong, R., Song,
J., and Zhang, T. RAGTruth: A hallucination corpus
for developing trustworthy retrieval-augmented language
models. arXiv preprint arXiv:2401.00396 , 2023.
Proskurina, I., Artemova, E., and Piontkovskaya, I. Can
BERT eat RuCoLA? Topological data analysis to ex-
plain. In Proceedings of the 9th Workshop on Slavic
Natural Language Processing 2023 (SlavicNLP 2023) ,
May 2023a.
Proskurina, I., Artemova, E., and Piontkovskaya, I. Can
bert eat rucola? topological data analysis to explain. In
Proceedings of the 9th Workshop on Slavic Natural Lan-
guage Processing 2023 (SlavicNLP 2023) , pp. 123–137,
2023b.
Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD:
100,000+ questions for machine comprehension of text.
In Su, J., Duh, K., and Carreras, X. (eds.), Proceedings
of the 2016 Conference on Empirical Methods in Natural
Language Processing , 2016.
Reddy, S., Chen, D., and Manning, C. D. CoQA: A con-
versational question answering challenge. Transactions
of the Association for Computational Linguistics , 7:249–
266, 2019.
10

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Sky, C.-W., Van Durme, B., Eisner, J., and Kedzie, C. Do an-
droids know they’re only dreaming of electric sheep? In
Findings of the Association for Computational Linguistics
ACL 2024 , pp. 4401–4420, 2024.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozi `ere, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation lan-
guage models. arXiv preprint arXiv:2302.13971 , 2023.
Tulchinskii, E., Kuznetsov, K., Cherniavskii, D., Baran-
nikov, S., Nikolenko, S., and Burnaev, E. Topological
data analysis for speech processing. In Proceedings of
the Annual Conference of the International Speech Com-
munication Association, INTERSPEECH , pp. 311–315,
2023.
Tulchinskii, E., Kuznetsov, K., Kushnareva, L., Cherni-
avskii, D., Nikolenko, S., Burnaev, E., Barannikov, S.,
and Piontkovskaya, I. Intrinsic dimension estimation for
robust detection of ai-generated texts. Advances in Neural
Information Processing Systems , 36, 2024.
Uchendu, A. and Le, T. Unveiling topological structures in
text: A comprehensive survey of topological data analysis
applications in NLP. arXiv preprint arXiv:2411.10298 ,
2024.
Vakhrushev, A., Ryzhkov, A., Savchenko, M., Simakov, D.,
Damdinov, R., and Tuzhilin, A. LightAutoML: AutoML
solution for a large financial services ecosystem. arXiv
preprint arXiv:2109.01528 , 2021.
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones,
L., Gomez, A., Kaiser, L., and Polosukhin, I. Atten-
tion is all you need. In Advances in Neural Information
Processing Systems , 2017.
V oita, E., Talbot, D., Moiseev, F., Sennrich, R., and Titov, I.
Analyzing multi-head self-attention: Specialized heads
do the heavy lifting, the rest can be pruned. In Proceed-
ings of the 57th Annual Meeting of the Association for
Computational Linguistics , pp. 5797–5808, 2019.
Wang, Y ., Wang, M., Manzoor, M. A., Liu, F., Georgiev,
G., Das, R., and Nakov, P. Factuality of large language
models: A survey. In Proceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing ,
pp. 19519–19529, 2024.
Xiong, M., Hu, Z., Lu, X., LI, Y ., Fu, J., He, J., and Hooi, B.
Can LLMs express their uncertainty? an empirical evalu-
ation of confidence elicitation in LLMs. In The Twelfth
International Conference on Learning Representations ,
2024.Zhang, Y ., Li, Y ., Cui, L., Cai, D., Liu, L., Fu, T., Huang,
X., Zhao, E., Zhang, Y ., Chen, Y ., et al. Siren’s song in
the AI ocean: a survey on hallucination in large language
models. arXiv preprint arXiv:2309.01219 , 2023.
Zhao, Y ., Zhang, J., Chern, I., Gao, S., Liu, P., He, J.,
et al. Felm: Benchmarking factuality evaluation of large
language models. Advances in Neural Information Pro-
cessing Systems , 36:44502–44523, 2023.
11

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
A. Topological data analysis: background
A simplicial complex Sis a collection of simplices such that every face of a simplex σ∈Sis also in S. Simplices are the
higher-dimensional generalizations of triangles; a 0-simplex is a vertex, a 1-simplex is an edge, a 2-simplex is a triangle, and
so forth. Formally, given a finite set X, ann-simplex σis an(n+ 1) subset of X. Simplicial complexes are fundamental
objects in algebraic and combinatorial topology, serving as a discrete analog to topological spaces.
The Vietoris-Rips complex V Rε(X)of a weighted graph G= (VG, EG)with distance threshold ε >0is defined as follows:
VRε(G) =
σ⊆VG∀vi, vj∈σ, w(eij)≤ε
,
where wis the edge weight function associated with G.
Homology groups Hkare invariants used in algebraic topology to study the topological properties of a space. Let Ck(S)
denote vector space over Z/2Z, with the basis consisting of k-dimensional simplices of S. Elements of Ckare called chains.
Formally, homology groups are derived from a chain complex (C•, ∂•), which is a sequence of Ckconnected by boundary
maps ∂k:
C•:··· → Ck+1∂k+1− − − → Ck∂k− → ··· , ∂k◦∂k+1= 0.
Thek-th homology group Hkis defined as the quotient of the group of k-cycles (chains whose boundary is zero) by the
group of k-boundaries (chains that are the boundary of a (k+ 1) -chain). Mathematically, this is expressed as:
Hk(S) =Zk(S)/Bk(S),
where Zk= ker ∂k={c∈Ck|∂k(c) = 0}andBk= im ∂k+1={∂k+1(c)|c∈Ck+1}is the group of k-boundaries. The
elements of Hk(S)represent various k-dimensional topological features in S. Elements of a basis in Hk(S)correspond to a
set of basic topological features.
A filtration of simplicial complexes Fis a family of nested simplicial complexes:
F:∅⊆S1⊆S2⊆ ··· ⊆ Sn=S,
where each Skis a simplicial complex itself. In practice, the filtrations of simplicial complexes are usually obtained for
sequences of increasing thresholds 0< ε1<···< εn. For example, simplicial complexes V Rεi(X)form a filtration
FV R(X) :∅⊆V Rε1(X)⊆V Rε2(X)⊆. . . V R εn(X) =V R(X)
As a threshold εincreases, new topological features (e.g., connected components, holes) can appear and disappear. The
persistent homology tool is used to track the dynamics of these topological features. Formally, the k-th persistent homology
ofSis the pair of sets of vector spaces {Hk(Si)|0≤i≤n}and maps fij, where fij:Hk(Si)→Hk(Sj)is a map
induced by the embedding Si⊆Sj. Each persistent homology class in this sequence is “born” at some Siand “dies” at
some Sjor never dies (Barannikov, 1994). This birth-death process of a basic set of independent topological features can be
visualized as the set of intervals [εbirth, εdeath]called barcode (see Figure 4). The features with 0lifespans are typically
excluded. The horizontal axis is a sequence of thresholds ε, and each horizontal bar corresponds to a single feature. We
begin with |X|=mconnected components (all of them are “born”), and as εincreases, their pairs are merged (each merge
corresponds to a “death” of a feature). The 0−th barcode construction procedure is equivalent to Kruskal’s algorithm for
minimum spanning tree (MST), the bars in the barcode correspond to the edges in the MST of X(Tulchinskii et al., 2023).
B. MTop-Div on graphs properties
Proof of Proposition 4.1 .
1. The 0−th Cross-Barcode coincides with the set of edges in the minimal spanning tree of the weighted graph Gwith all
the weights within Pvertex subset equal zero. Excluding the zero weight edges, this edge set coincides with the minimal
spanning forest attaching the vertex set RtoPvertices.
Proof of Proposition 4.2 .
1. Denote by MSF( R, P)the minimum spanning forest attaching RtoP. Note that we have properties 4.1, so
MTop-Div( R, P) =X
e∈MSF( R,P)w(e). (7)
12

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
0.25 0.5 0.25 0.5 0.25 0.5 0.25 0.5
Figure 4. H0barcode construction. As the threshold increases, the separate connected components merge, resulting in the death of
topological features. The horizontal axis is a sequence of thresholds ε, and each horizontal bar corresponds to a single feature.
Therefore, we have to show that the weight of MSF( R, P)does not change significantly when all weights are changed by
no more than ε.
There are two possibilities: 1) after a change, all MSF edges remain the same, or 2) some edges are replaced with other edges.
In the first case, it is obvious that the total sum of edges weights changes by no more than δ=ε·#edges (MSF( R, P)) =
ε· |R|. Consider the second case. Denote by MSF prevthe original MSF, by MSF new— the MSF after the change; let wbe
the edge weight function before the change, ˆw— after the change. The following inequalities hold:
ˆw(MSF new)<ˆw(MSF prev); (8)
w(MSF prev)−δ≤ˆw(MSF prev)≤w(MSF prev) +δ; (9)
w(MSF new)−δ≤ˆw(MSF new)≤w(MSF new) +δ; (10)
w(MSF new)≥w(MSF prev). (11)
From (8)-(9)follows that ˆw(MSF new)< w(MSF prev) +δ; from (10)-(11) follows that ˆw(MSF new)≥w(MSF prev)−δ,
QED.
2. We have to check the definition of the exact sequence: Ker(ri) =Im(ri+1). For a pair r0, r1, it is equivalent to the
surjectvity of r1. The H0homology group of a graph corresponds to the connected components of the graph. The set of
edges E≤α
(G,w)={e∈EG|we≤α}is always a subset in the analogous set of the weighted graph (G, w (R∪P)/P)with
all weight edges between Pvertices set to zero. Therefore, the map r1between their connected components is surjective.
Similarly the kernel of the map r1is spanned by the differences of two connected components, that are merged after adding
some of the edges between Pvertices, and any such difference lies in the image of the map r2. Also any two vertices from
Pbelong to the same connected component in the graph (G, w (R∪P)/P≤α), hence the image of r2is in the kernel of r1.
Therefore, the considered sequence is exact indeed.
C. Datasets
SQuAD (Rajpurkar et al., 2016) and CoQA (Reddy et al., 2019) are question-answering benchmarks previously used as a
basis for collecting hallucination detection datasets (Kuhn et al., 2023; Manakul et al., 2024). However, these datasets have
not been published before, complicating further up-to-date research.
We have annotated the responses of several LLMs to the questions from SQuAD and CoQA using GPT-4o in an automated
regime. We will release these datasets publicly. Here, we provide the methodology for the automated annotation and evaluate
its quality, motivating the usage of the introduced datasets in practice.
13

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
SQuAD CoQA
Given the context, answer the question in a brief but complete sentence. Once upon a time, in a quiet village, there lived a kind old baker named Henry.
Note that your answer should be strictly based on the given context. He was known for his delicious bread and warm smile. One day, a traveler arrived,
In case the context does not contain the necessary information to answer the question, tired and hungry, Henry welcomed him with a fresh loaf.
please reply with “Unable to answer based on given context”. Q:What was Henry known for?
Context: A: Baking delicious bread.
Once upon a time, in a quiet village, there lived a kind old baker named Henry. Q:What else?
He was known for his delicious bread and warm smile. One day, a traveler arrived, A:Warm smile.
tired and hungry, and Henry welcomed him with a fresh loaf. Q:How did the traveler feel when he arrived?
Question: Who was known for baking delicious bread? A:Tired and hungry.
Answer: Q: What did Henry give the traveler?
Table 5. Examples of prompts used during generation for CoQA and SQuAD (we add additional delimiter spaces and formatting that are
non-present in actual prompts for better readability). SQuAD contains instructions followed by context and questions. In CoQA, the
prompt has only a contextual passage followed by a questions-answers series, with the last question being the actual one.
You are an AI assistant specialized in detecting hallucinations in question-answering tasks.
Your job is to analyze the given context, question, and generated answer to identify
whether the answer contains any hallucinations. Examples:
Example 1.
Context :
The city of Paris is the capital of France. It is known for its iconic landmarks
like the Eiffel Tower and Notre Dame Cathedral.
The city is situated in the northern part of the country, near the Seine River.
Question : Is Paris the capital of Germany?
Generated answer : Yes, Paris is the capital of Germany.
Hallucination : Yes.
Example 2.
Context :
The city of Paris is the capital of France.
It is known for its iconic landmarks like the Eiffel Tower and Notre Dame Cathedral.
The city is situated in the northern part of the country, near the Seine River.
Question : Is Paris the capital of Germany?
Generated answer : No, Paris is not the capital of Germany. According to the context,
Paris is the capital of France.
Hallucination : No.
You should determine if the answer contains hallucinations according to the hallucination types above.
If you cannot decide if the generated answer is a hallucination, write “N/A.” as the answer.
The answer you give MUST be ONLY “Yes.”, “No.” or “N/A.”; do NOT give ANY explanation.
Table 6. Example of annotation prompt passed to GPT-4o (we add additional delimiter spaces and formatting non-present in actual
prompts for better readability).
Prompt number 1 2 3 4 5
CoQAAccuracy ( ↑) 0.809 ± 0.017 0.861 ± 0.015 0.742 ± 0.003 0.795 ± 0.009 0.831 ± 0.025
Precision ( ↑) 0.849 ± 0.021 0.911 ± 0.007 0.771 ± 0.003 0.828 ± 0.011 0.860 ± 0.012
Recall ( ↑) 0.871 ± 0.004 0.877 ± 0.019 0.877 ± 0.013 0.877 ± 0.005 0.893 ± 0.027
SQuADAccuracy ( ↑) 0.831 ± 0.003 0.857 ± 0.018 0.857 ± 0.008 0.872 ± 0.003 0.854 ± 0.007
Precision ( ↑) 0.813 ± 0.002 0.831 ± 0.028 0.845 ± 0.021 0.850 ± 0.011 0.847 ± 0.007
Recall ( ↑) 0.796 ± 0.008 0.839 ± 0.010 0.823 ± 0.023 0.858 ± 0.018 0.813 ± 0.017
Average Accuracy ( ↑) Precision ( ↑) Recall ( ↑)
CoQA 0.808 0.844 0.879
SQuAD 0.854 0.837 0.826
Table 7. Classification metrics of GPT-4o annotation for CoQA and SQuAD with human labels considered actual annotation. The top
table shows metric scores for different variants of prompts used. The bottom table shows the metric scores averaged across all prompt
variants.
14

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
C.1. Data Generation & Annotation
Generation. We generate responses from a language model (LLM) for the CoQA and SQuAD datasets, employing different
prompting strategies for each dataset while keeping these strategies consistent across models (see prompt examples in
Table 5). For SQuAD, responses are generated using a zero-shot approach. In contrast, for CoQA, we create queries in a
few-shot manner without providing specific instructions, following (Lin et al., 2024): each sample consists of a passage and
a series of question-answer pairs, concluding with a final question that the model is expected to answer.
Annotation: automated vs human. We treat hallucination detection as a binary classification problem; our target indicates
whether a hallucination is present anywhere in the model’s response. Two approaches to annotating model generations were
considered: 1) automated annotation using an LLM (in our case, GPT-4o), and 2) manual annotation by human experts.
During the automated annotation process, we provide an LLM’s output preceded by an instruction (prompt) to GPT-4o. In
this prompt, GPT-4o is asked to determine whether the output contains hallucinations, and we expect a single-word response
of either “Yes” or “No.” An example of such an instruction is shown in Table 6.
For human annotation, we asked three team members with at least upper-intermediate English proficiency to independently
annotate approximately 100 samples from each dataset. We selected samples where all annotators reached a consensus and
considered these annotations the ground truth hallucination labels.
To further evaluate GPT-4o, we conducted automatic annotation using several variations of prompts, each reformulating the
task for GPT-4o, including zero-shot and few-shot versions. We then compared these annotations to the actual hallucination
labels. The results, presented in Table 7, demonstrate a consistent alignment between GPT-4o’s annotations and those made
by humans, regardless of the specific prompt. This consistency confirms the robustness of our approach to the exact form of
instruction.
Based on these findings, we prefer automated annotation as a cost-effective and efficient alternative to human experts.
Annotation: general pipeline. CoQA and SQuAD contain questions and the ground truth answers to those questions. We
employed them to reduce the potential false positive labels in the following way: 1) we compute Rouge-L scores between
ground truth answers and the model’s response, and 2) check if any of the grounded answers are a substring of the response.
The responses corresponding to Rouge-L = 1are immediately labeled as grounded (complete match). The responses that
meet the following conditions: 1) corresponding Rouge-L ≤0.3(as in (Kuhn et al., 2023)), and 2) none of the ground truth
answers is its’ sub-string, are labeled as potential hallucinations and are then annotated via GPT-4o. The responses that are
confirmed to be hallucinated by GPT-4o are finally annotated as hallucinations.
Detailed statistics for each dataset can be seen in Table 8. The number of samples in the datasets varies across models, as we
tried to maintain a balance of hallucinated and grounded responses, as well as to ensure sample cleanness and minimize
mislabeling. The procedure outlined above selects a different number of objects in a sample depending on the quality of the
model’s responses.
ModelCoQA SQuAD
Hal. Grounded Hal. Grounded
Mistral-7B 776 776 ✗ ✗
LLaMA-2-7B 375 375 ✗ ✗
LLaMA-2-13B 279 384 ✗ ✗
LLaMA-3.1-8B 190 247 350 400
Qwen2.5-7B 124 183 215 249
Table 8. Datasets statistics. Number of hallucinated and grounded samples of each model.
D. Other experiment results
D.1. Hallucination patterns
Due to a lack of space in the paper’s main section, we illustrate typical patterns within hallucination-aware heads. Pictures a)
and c) display the minimal spanning forest (MSF) of a hallucinated example, while b) and d) — the MSF of a grounded one.
For this illustration, we used two examples from the RAGTruth QA dataset corresponding to the extreme MTop-Div values
15

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
for the (19, 4) head of Mistral-7B. The layout is similar to the one in Figure 2: on the left side, we arrange prompt tokens in
the same order as they are present in the text, while the response tokens are located on the right. Blue color denotes the
tokens of the instruction (“Briefly answer the following...”), red color — tokens of the context, green color — grounded
tokens of the response, yellow color — the hallucinated tokens of the response. Here, the “entanglement” and the attention
to the utility token <s> patterns are apparent.
Hallucinated
 Grounded
<s>[INST]Brieflyanswerthefollowingquestion:
<s>[INST]Brieflyanswerthefollowingquestion:Context
Instruction
Grounded
Hallucinateda) b)
c) d)
Figure 5. Typical patterns within the hallucination aware-heads. Two samples from RAGTruth QA are shown. Figures a) and c) correspond
to the hallucinated sample with large MTop-Div value, while b) and d) — to the grounded one, with a small one. Model: Mistral-7B.
D.2. Transfer within QA datasets
We carried out additional transferability experiments for all our models among QA datasets. For LLaMA-2-7B, Mistral-7B,
and LLaMA-2-13b, we considered transfer between the RAGTruth QA and CoQA datasets, while for Qwen2.5-7B and
LLaMA-3.1-8B — between SQuAD and CoQA. The results are provided in Table 9. TOHA mostly outperforms supervised
methods, which aligns well with the transferability experiments in section 5.
D.3. Classifier on topological features
Our preliminary experiments on developing the TDA-based hallucination detector included training classifiers on the
topological features previously used for other NLP tasks (Kushnareva et al., 2021; Cherniavskii et al., 2022). We considered
features extracted from barcodes (i.e., the sum of lengths of bars) and “naive” topological features (i.e., average vertex
degree) of attention graphs. As we considered supervised methods, our primary goal was to determine whether these features
can enhance the performance of simple hidden states-based classifiers. To make the most of the considered features and
their aggregations, we trained an AutoML model (Vakhrushev et al., 2021) on different feature sets — only hidden states;
only barcode features; hidden states and barcode features; hidden states, barcode features and “naive” topological features.
The results are presented in Table 10.
16

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
Table 9. ROC AUC values for transfer of TOHA (unsupervised) and supervised methods’ comparison for five LLMs for question answering
datasets. TOP-1 results are highlighted with bold font , while TOP-2 are underlined .
MethodTOHA Linear Attn.-pool
(ours) probe probe
Train Test
Mistral-7B
RAGTruth CoQA 0.867 0.635 0.687
CoQA RAGTruth 0.720 0.596 0.611
LLaMA-2-7B
RAGTruth CoQA 0.858 0.628 0.623
CoQA RAGTruth 0.646 0.497 0.532
LLaMA-2-13B
RAGTruth CoQA 0.800 0.574 0.617
CoQA RAGTruth 0.734 0.553 0.52
LLaMA-3.1-8B
SQuAD CoQA 0.703 0.733 0.761
CoQA SQuAD 0.883 0.730 0.582
Qwen2.5-7B
SQuAD CoQA 0.640 0.522 0.562
CoQA SQuAD 0.849 0.510 0.553
As we can see, a classifier on the proposed features alone does not achieve the performance of a hidden states-based classifier.
When concatenated to hidden states, they do not provide significant metric gains. Our conclusion was straightforward: we
needed to develop other approaches to this problem.
We also considered training classifiers on the concatenated MTop-Div values from all layers and heads of the models,
along with classifiers on top of the concatenated MTop-Div values and hidden states. These results are also displayed
in Table 10. We can see that models employing MTop-Div outperform pure hidden states-based classifiers. However,
calculating MTop-Div vales from all heads and layers is very computationally expensive, therefore, this method is not very
practical. The most valuable conclusion that can be made from this comparison is that the MTop-Div values are more
informative from the hallucination detection point of view than both standard TDA features and hidden states of a model.
Table 10. Performance of supervised classifiers on top of various feature combinations. ROC-AUC values are presented. “Hiddens” refer
to hidden states of a model. TOP-1 results are highlighted with bold font , while TOP-2 are underlined .
MethodRAGTruthCoQAQA
Mistral-7B
Linear probe 0.841 0.922
Classifier on MTop-Div 0.858 0.972
Topological features 0.671 0.694
Hiddens + top. features 0.850 0.923
Hiddens + MTop-Div 0.837 0.978
LLaMA-2-7B
Linear probe 0.731 0.800
Classifier on MTop-Div 0.750 0.960
Topological features 0.689 0.699
Hiddens + top. features 0.744 0.863
Hiddens + MTop-Div 0.721 0.950
17

Hallucination Detection in LLMs via Topological Divergence on Attention Graphs
E. Implementation details
In this section, we provide the main choices of parameters for baselines.
•For methods that rely on hidden states, we considered the outputs from the 16th layer, as the ablation studies in (Sky
et al., 2024; Azaria & Mitchell, 2023) showed that middle layers contain the maximum amount of information in the
context of factuality evaluation.
•In the linear probing method and INSIDE, we used the last token representation as an embedding of an entire sentence,
following the ablation study in (Chen et al., 2024).
•For methods relying on multiple generations, we used 20 additional generations based on the results of (Manakul et al.,
2024; Chen et al., 2024).
•As for an aggregation of token entropies from the entire sequence, we chose the maximum value to be the score for
tokenwise entropy as our own results and the results from (Manakul et al., 2024) showed that it demonstrates better
performance than simple averaging.
Our repository is available at https://anonymous.4open.science/r/tda4hallu-0679 .
18