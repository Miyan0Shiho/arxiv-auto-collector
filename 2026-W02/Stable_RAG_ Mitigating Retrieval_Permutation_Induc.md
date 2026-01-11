# Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation

**Authors**: Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei Zheng, Zhiming Zheng

**Published**: 2026-01-06 13:07:38

**PDF URL**: [https://arxiv.org/pdf/2601.02993v2](https://arxiv.org/pdf/2601.02993v2)

## Abstract
Retrieval-Augmented Generation (RAG) has become a key paradigm for reducing factual hallucinations in large language models (LLMs), yet little is known about how the order of retrieved documents affects model behavior. We empirically show that under Top-5 retrieval with the gold document included, LLM answers vary substantially across permutations of the retrieved set, even when the gold document is fixed in the first position. This reveals a previously underexplored sensitivity to retrieval permutations. Although robust RAG methods primarily focus on enhancing LLM robustness to low-quality retrieval and mitigating positional bias to distribute attention fairly over long contexts, neither approach directly addresses permutation sensitivity. In this paper, we propose Stable-RAG, which exploits permutation sensitivity estimation to mitigate permutation-induced hallucinations. Stable-RAG runs the generator under multiple retrieval orders, clusters hidden states, and decodes from a cluster-center representation that captures the dominant reasoning pattern. It then uses these reasoning results to align hallucinated outputs toward the correct answer, encouraging the model to produce consistent and accurate predictions across document permutations. Experiments on three QA datasets show that Stable-RAG significantly improves answer accuracy, reasoning consistency and robust generalization across datasets, retrievers, and input lengths compared with baselines.

## Full Text


<!-- PDF content starts -->

Stable-RAG: Mitigating Retrieval-Permutation-Induced
Hallucinations in Retrieval-Augmented Generation
Qianchi Zhang1,2, Hainan Zhang1,2*, Liang Pang4, Hongwei Zheng3, Zhiming Zheng1,2,
1Beijing Advanced Innovation Center for Future Blockchain and Privacy Computing,
2School of Artificial Intelligence, Beihang University, China,
3Beijing Academy of Blockchain and Edge Computing, China,
4Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China,
{zhangqianchi, zhanghainan}@buaa.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has
become a key paradigm for reducing factual hal-
lucinations in large language models (LLMs),
yet little is known about how the order of re-
trieved documents affects model behavior. We
empirically show that under Top-5 retrieval
with the gold document included, LLM an-
swers vary substantially across permutations of
the retrieved set, even when the gold document
is fixed in the first position. This reveals a previ-
ously underexplored sensitivity to retrieval per-
mutations. Although robust RAG methods pri-
marily focus on enhancing LLM robustness to
low-quality retrieval and mitigating positional
bias to distribute attention fairly over long con-
texts, neither approach directly addresses per-
mutation sensitivity. In this paper, we pro-
poseStable-RAG, which exploits permutation
sensitivity estimation to mitigate permutation-
induced hallucinations. Stable-RAG runs the
generator under multiple retrieval orders, clus-
ters hidden states, and decodes from a cluster-
center representation that captures the domi-
nant reasoning pattern. It then uses these rea-
soning results to align hallucinated outputs to-
ward the correct answer, encouraging the model
to produce consistent and accurate predictions
across document permutations. Experiments
on three QA datasets show that Stable-RAG sig-
nificantly improves answer accuracy, reasoning
consistency and robust generalization across
datasets, retrievers, and input lengths compared
with baselines.
1 Introduction
Large language models (LLMs) have achieved re-
markable performance on language understand-
ing and generation tasks, but still often generate
confident yet incorrect statements, known as fac-
tual hallucinations (Fan et al., 2024), especially in
knowledge-intensive settings (Chen et al., 2022;
*Corresponding author.
3.2-1B-Instruct 3.2-3B-Instruct 3-8B-Instruct 3.3-70B-Instruct20406080100PSR (%)72.7
59.5
50.8
27.592.3
78.0
71.4
45.894.1
86.2
81.9
51.494.5
88.285.5
51.693.8
86.984.4
50.8Gold Doc in Pos 1
Gold Doc in Pos 2
Gold Doc in Pos 3
Gold Doc in Pos 4
Gold Doc in Pos 5Figure 1:Perturbation Success Rate (PSR)on the NQ
test set across different LLaMA models. PSR is com-
puted as the proportion of successful document-order
perturbations to produce hallucination results among
1000 randomly sampled instances, with the gold doc-
ument fixed in the different positions. Qwen models’
results can be seen in Appendix C.1.
Huang et al., 2023). Retrieval-Augmented Gener-
ation (RAG) (Gao et al., 2023; Lewis et al., 2020)
reduces factual hallucinations by grounding model
outputs in externally retrieved documents rather
than relying only on parametric knowledge, im-
proving factuality, interpretability, and updatability
without additional retraining (Zhou et al., 2024).
Despite these benefits, RAG systems are far from
hallucination-free. We identify a critical but over-
looked vulnerability in existing RAG systems: a
strong sensitivity to the order of retrieved docu-
ments. When the retrieved content remains exactly
the same, including the gold document, merely re-
ordering them can lead the model to follow entirely
different reasoning paths and produce inconsistent
answers, referred to asPermutation-Induced Hal-
lucinations. As shown in Figure 1, we retrieve
the Top-5 documents (Zhu et al., 2024b; Xu et al.,
2024) and place the gold document at different
positions, LLM answers vary substantially across
retrieval permutations. Even when the gold docu-
ment is fixed first, models may still ignore it and
produce answers that conflict with the evidence.
This reveals a previously underexplored sensitiv-
ity to retrieval permutations, even for such short
contexts shorter than one thousand tokens.
1arXiv:2601.02993v2  [cs.CL]  7 Jan 2026

Existing work on RAG robustness mainly focus
on retrieval quality and positional bias. The former
enhances LLM robustness to low-quality retrieval
via uncertainty estimation and adversarial training,
such as noise injection (Fang et al., 2024; Yoran
et al., 2024) of weak-relevant documents. The latter
alleviates attention bias toward specific positions
in long contexts, promoting more balanced use of
retrieved documents (Zhang et al., 2024c; Wang
et al., 2025b). However, these approaches over-
look a critical issue: permutation sensitivity is nei-
ther caused by weakly relevant documents, because
the input documents are the same, nor confined to
long-context reasoning tasks, since only the Top-5
documents fall within one thousand tokens.
Instead, permutation sensitivity stems from struc-
tural instability in the internal reasoning dynamics
of LLMs. As model depth increases, document
permutations induce a growing number of distinct
reasoning trajectories, leading to more frequent
branching and a higher risk of hallucinations or
unreliable outputs. As shown in Figure 2, we mea-
sure the average number of clusters obtained via
spectral clustering over document-permuted repre-
sentations across different LLM layers on the NQ
and HotpotQA datasets. The results indicate that
reasoning trajectories in shallow layers are rela-
tively concentrated, while divergence emerges in
the middle layers and becomes more pronounced in
higher layers. Furthermore, sensitive samples(i.e.,
10+) exhibit substantially greater divergence than
non-sensitive ones(i.e., 1-2), with this effect primar-
ily localized to the higher layers. These findings
highlight the importance of mitigating permutation
sensitivity, enabling LLMs to produce stable and
accurate outputs regardless of the ordering of re-
trieved documents, which is critical for improving
the robustness of RAG systems.
In this paper, we introduceStable-RAGthat
explicitly leverages permutation sensitivity estima-
tion to mitigate the permutation-induced hallucina-
tions. Specifically, we apply spectral clustering to
the last token hidden states of the final layer before
response generation, across all document permuta-
tions to identify dominant reasoning clusters. For
each cluster, we select a representative hidden state
and decode it to obtain candidate answers, thereby
capturing the model’s core reasoning modes. Then,
we perform cross-cluster consistency alignment
over these candidates, encouraging the model to
prioritize semantically consistent and factually cor-
rect answers across different document orders. This
8 16 24 32
Layer Index024681012141618Avg. Cluster /glyph1197umber
LLaMA3-8B-Instruct
12
35
610
10+
6 12 18 24 30 36
Layer Index024681012141618Avg. Cluster /glyph1197umber
Qwen3-8B
12
35
610
10+Figure 2: Hidden-state clustering behaviors across lay-
ers for LLaMA3-8B-Instruct on the NQ train set with
DPR retriever and Qwen3-8B on the HotpotQA train set
with Contriever retriever, using 1,000 random sampled
instances. Different colored lines indicate the number of
clusters of final reasoning states produced by the LLM
under all 5!(= 120) permutations of the Top-5 retrieved
documents (e.g., the green line indicates 3–5 cluster
states). Other scales are reported in Appendix C.2.
cluster-based alignment substantially reduces the
uncertainty induced by order perturbations and im-
proves the robustness of RAG at its root.
Experiments on three QA datasets demonstrate
that Stable-RAG significantly improves answer ac-
curacy, reasoning consistency and robust general-
ization across datasets, retrievers, and input lengths
compared with strong baselines.
Our main contributions are as follows:
•We find that RAG systems are highly sensitive
to document order, leading to inconsistent rea-
soning. We analyze this permutation sensitivity
via layer-wise hidden state clustering, showing
divergence in reasoning trajectories across layers.
•We propose Stable-RAG, which mitigates
permutation-induced hallucinations using cluster-
based decoding and alignment, achieving model-
agnostic stable reasoning.
•Across three QA datasets, Stable-RAG outper-
forms strong baselines in accuracy and reasoning
consistency and generalizes across datasets, re-
trievers, and input lengths.
2 Related Work
RAG mitigates factual hallucinations in LLMs for
knowledge-intensive tasks by providing explicit
evidence from external documents (Lewis et al.,
2020; Fan et al., 2024; Chen et al., 2022). Prior
work on improving the robustness of RAG systems
has primarily focused on enhancing retrieval qual-
ity (Wang et al., 2025a; Xu et al., 2024) or strength-
ening the generator’s robustness. For instance,
2

Selective-Context (Li et al., 2023), EXIT (Hwang
et al., 2025), and AdaComp (Zhang et al., 2024b)
apply noise filtering to boost generation accuracy;
RetRobust (Yoran et al., 2024) and RAAT (Fang
et al., 2024) expose the model to retrieval noise
or irrelevant documents during training, enhanc-
ing robustness. However, these methods generally
assume a stable document order and do not system-
atically assess its impact on reasoning. Although
ATM (Zhu et al., 2024a) considers order perturba-
tions, it does not explicitly model reasoning trajec-
tories across permutations, and thus cannot ensure
consistency.
Additionally, another line of research focuses po-
sitional bias in long-context scenarios. Most LLMs
use relative positional encodings (Peysakhovich
and Lerer, 2023), such as RoPE (Su et al., 2024) or
ALiBi (Press et al., 2021), which introduce system-
atic biases: early tokens receive excessive attention
due to attention sinks (Xiao et al.; Gu et al.), while
long-range decay favors recent tokens. Prior work
mitigates these issues by modifying positional en-
codings (Zhang et al., 2024c; Chen et al., 2024;
Lin et al., 2024), adjusting causal masks, reweight-
ing attention or hidden states (Hsieh et al., 2024),
or using Pos2Distill (Wang et al., 2025b) to dis-
till knowledge from advantageous to less favorable
positions to promote fair attention across tokens.
However, these methods mainly target long con-
texts or large document sets and do not explicitly
address reasoning inconsistencies induced by dif-
ferent permutations of the same retrieved document
set.
3 Preliminary Study
3.1 Problem Formulation
Given a query qand its retrieved document set
S={d 1, d2, . . . , d n}, the goal is to ensure that the
model fθproduces consistent outputs across dif-
ferent document orderings. Let Perm(S) denote
all possible permutations of S. For any two per-
mutations π1, π2∈Perm(S) , the model’s outputs
should be as similar as possible:
fθ(q, π 1)≈f θ(q, π 2).
In this task, the model is expected to produce con-
sistent answers regardless of the document order.
3.2 Permutation Sensitivity Estimation
Recent work (Liang et al., 2025; Lee et al., 2025)
exploits hidden states to uncover latent reasoning
Layer 1 Layer 4 Layer 8
Layer 12 Layer 16 Layer 20
Layer 24 Layer 28 Layer 32Question: what is the liquid in a magic 8 ball?  Correct Answer(s): Alcohol.
Alcohol. Water.Figure 3: The layer-wise visualization of case study
from the NQ train set on LLaMA-3-8B-Instruct. Each
point corresponds to a document order, and its color
represents the model’s final answer.
trajectories, often as indicators of generative uncer-
tainty. Accordingly, we propose to quantify model
generation uncertainty via spectral clustering of
hidden states. In this section, we validate the feasi-
bility of spectral clustering algorithm through both
layer-wise visualization and quantitative analysis.
Layer-wise Visualization.For each question, we
permute the Top-5 documents to generate 5! = 120
orders and extract the hidden states of the last to-
ken from each layer before response generation.
Representative layers are then projected to two di-
mensions via PCA for visualization, as shown in
Figure 3. We observe that hidden states in shal-
low layers form mixed clusters with points corre-
sponding to different answers interleaved, while
in deeper layers the clusters become increasingly
well-separated and points with the same answer
clearly group together. This indicates that varia-
tions in document order induce distinct reasoning
trajectories, which manifest as progressively sepa-
rable clusters in hidden state space, reflecting the
model’s internal reasoning patterns. More results
are presented in Appendix C.3.
Quantitative Analysis of Clustering.To assess
each cluster’s reasoning performance, we select the
hidden state closest to the cluster center, decode
it as a representative answer of the cluster, and
match this answer with the real reasoning answers
of all hidden states in the same cluster to com-
pute overall Precision, Recall, and F1 scores. As
3

Stage 1: Hidden State Clustering
LLMLayer 1
Stage 2: Preference Data Construction Stage 3: Alignment with DPO
Fully Correct (FC)
Partially  Correct (PC)
Fully Incorrect and Unanswerable (FU)
Fully Incorrect but Answerable (FA)
Layer n
LLM
No need for Training
Calibration
Abstention
Encourage
Permutation
 Internal States Extraction
 Spectral Clustering on Hidden States
 Representative Decoding within Clusters
Compare with the Ground Truth
SamplingRight Order
Wrong OrderConsolidate
Figure 4: Overall framework of our Stable-RAG.
Model Layer Precision Recall F1
QWEN3-8B8 78.1 79.3 77.9
16 79.9 81.3 79.6
24 86.8 87.5 86.6
36 87.8 88.4 87.6
LLAMA3-8 69.2 71.8 69.3
8B-INSTRUCT16 81.4 82.5 81.3
24 82.3 83.7 82.2
32 84.1 85.2 83.9
Table 1: Clustering performance (%) of hidden states
across different layers for Qwen3-8B and LLaMA3-
8B-Instruct on the NQ train set using DPR retriever,
evaluated on 10,000 randomly sampled instances.
shown in Table 1, clustering metrics improve with
network depth, indicating that hidden states for dif-
ferent answers become more separable in deeper
layers. Notably, the clustering performance is al-
ready satisfactory for practical use, with F1 scores
of 83.9 using LLaMA3 and 87.6 using Qwen3, re-
spectively. Thus, we use the final layer hidden
states for spectral clustering in our method.
4 Methodology
Overview.Our method comprises three stages:
hidden state clustering, preference data construc-
tion and alignment with DPO, as shown in Fig-
ure 4. For each permutation, we extract the last
token hidden state of the final layer before response
generation, capturing the model’s reasoning states.
Spectral clustering is then applied to uncover latent
reasoning modes, and representative states from
each cluster are decoded. By aligning hidden states
across permutations, our approach improves gener-
ation consistency across different retrieval orders.4.1 Hidden State Clustering
Internal States Extraction.For each query
qand its retrieved document set S=
{d1, d2, . . . , d n}, we enumerate all permutations
of the documents and run the model for each per-
mutation. Let i∈ {1, . . . , N} denote the permu-
tation index, where N=n! . To reduce computa-
tional cost, we extract only the last token hidden
state of the final layer before response generation,
h(i)∈Rd. Prior work (Azaria and Mitchell, 2023;
Ni et al., 2025) has shown that this hidden state
sufficiently captures the model’s perception of its
knowledge boundaries. We organize all hidden
states into a matrixH:
H= [h(1), h(2), ..., h(N)]⊤∈RN×d,
which represents the distribution of the model’s fi-
nal reasoning states across document permutations.
Spectral Clustering on Hidden States.To deter-
mine the number of clusters adaptively and capture
the global structure of the hidden state space, we
apply spectral clustering (Ng et al., 2001) to H,
where each cluster corresponds to a latent reason-
ing mode (Lee et al., 2025). We compute the simi-
larity between each pair of hidden states h(i)and
h(j)using the exponential of the cosine distance:
Aij= exp
−1−h(i)·h(j)
∥h(i)∥∥h(j)∥
σ
,
where σis a hyperparameter controlling sensitivity.
Here, A∈RN×Ndenotes the weighted adjacency
matrix of allNhidden states.
4

The normalized graph Laplacian Lis then con-
structed as
D= diagNX
j=1Aij
, L=I−D−1/2AD−1/2,
where Dis the degree matrix, with each diagonal
entry Diirepresenting the sum of edge weights con-
nected to the i-th hidden state (treated as a graph
node), andIis the identity matrix.
The number of clusters Kis determined adap-
tively via the eigengap of L. Let λ1≤ ··· ≤λ N
be the eigenvalues of L, and define the consecutive
gaps gapi=λi+1−λibetween each pair of adja-
cent eigenvalues. The number of clusters is then
set as K= max 
2,(arg max
igapi) + 1
to ensure
clear separation between latent reasoning modes.
Once Kis determined, we obtain normalized spec-
tral embeddings for all hidden states and assign
each h(i)to one of the clusters C1, C2, . . . , C K.
See more details in Appendix B.
Representative Decoding within Clusters.
Within each cluster Ck, we identify a representa-
tive hidden state through centroid-based sampling.
The cluster centroid is computed as:
µk=1
|Ck|X
h(i)∈Ckh(i).
We select the representative hidden state:
h(rk)= arg min
h(i)∈Ck∥h(i)−µk∥2.
Only the representative hidden states selected
within each cluster {h(r1), h(r2), . . . , h(rK)}are de-
coded into textual answers, reducing the number of
runs from N=n! toKand substantially lowering
computational and annotation overhead.
Exhaustive Full-Permutation Decoding.We
study an exhaustive permutation decoding setting
in which the model is evaluated under all ( N=
n!) permutations of retrieved documents. While
this fully characterizes permutation-induced output
variability, it is computationally and annotationally
prohibitive at scale. We therefore use it only as
a reference to assess the efficiency gains of our
representative decoding strategy.
4.2 Preference Data Construction
Targets.Our goal is to build a robust RAG sys-
tem. When the model cannot answer, it is encour-
aged to abstain to suppress hallucinations. Whenan answer is available, the output should remain
consistent regardless of document order, reducing
permutation sensitivity.
Data Construction Procedure.We construct
preference data P= (x, y w, yl)for training. For
each query qwith its retrieved documents set
S={d 1, d2, . . . , d n}, the input xis formed by
concatenating qwith a specific document permuta-
tionπ. Model outputs are obtained via representa-
tive decoding of hidden-state clusters induced by
document permutations. Each instance is then com-
pared with the ground truth and categorized into
the following four types: FC(Fully Correct): the
base model produces correct answers under all doc-
ument permutations. Such instances are stable and
excluded from training. PC(Partially Correct):
the base model produces both correct and incorrect
answers across permutations. Two representative
outputs are sampled:y wis the most frequent right
answer to consolidate correct predictions, and yl
is the most frequent wrong answer for calibration.
FU (Fully Incorrect and Unanswerable): the
base model answers incorrectly under all permuta-
tions and no gold answers exist in the documents.
ywis set to“I don’t know”to encourage absten-
tion, and ylis the most frequent wrong answer.
FA(Fully Incorrect but Answerable): the base
model answers incorrectly under all permutations
but a gold answer exists in the documents. ywis set
to the gold answer to encourage correct prediction,
andy lis“I don’t know”.
4.3 Alignment with DPO
We employ Direct Preference Optimization
(DPO) (Rafailov et al., 2023) to train the base
model on the constructed preference tuples. For
each tuple (x, y w, yl), DPO maximizes the like-
lihood of the preferred answer ywover the less
preferredy l:
LDPO=−E (x,yw,yl)∼Dh
logσ
βlogπθ(yw|x)
πref(yw|x)
−βlogπθ(yl|x)
πref(yl|x)i
,
where θdenotes the model parameters, σis the sig-
moid function, and βis a scaling hyperparameter
controlling the sharpness of preference. The model
policy πθis initialized using the base reference
policyπ ref.
5

MethodNQ TriviaQA HotpotQA
Average
Contriever DPR Contriever DPR Contriever DPR
SubEM F1 SubEM F1 SubEM F1 SubEM F1 SubEM F1 SubEM F1 SubEM F1
LLAMA3-8B-INSTRUCT
Direct Generation 25.18 29.11 25.18 29.11 55.92 58.95 55.92 58.95 21.39 22.87 21.39 22.87 34.16 36.98
Vanilla RAG 40.75 42.82 45.81 47.80 63.89 65.43 67.12 68.61 30.73 34.08 25.66 28.22 45.66 47.83
Vanilla SFT 42.10 44.78 46.20 49.44 55.52 51.40 57.10 52.51 27.25 31.58 24.63 29.85 42.13 43.26
RetRobust 41.82 44.26 48.70 49.29 64.85 66.72 68.67 70.42 31.46 35.34 26.96 30.36 47.08 49.40
ATM 43.75 44.88 49.78 50.19 66.37 67.12 70.12 70.35 34.36 36.97 28.55 29.31 48.82 49.80
RAAT 42.33 43.85 49.12 49.85 65.58 66.94 68.03 69.12 33.58 36.12 26.35 28.79 47.50 49.11
Pos2Distill 44.58 43.12 49.25 48.37 64.13 65.78 66.57 68.12 32.73 35.79 26.45 28.91 47.29 48.35
Ms-PoE 40.32 42.49 45.58 47.53 64.21 66.14 66.48 67.73 30.17 33.65 26.12 28.57 45.48 47.69
Stable-RAG (Ours) 48.14 45.80 52.02 50.72 72.05 71.56 73.43 73.76 38.91 39.87 29.48 31.68 52.34 52.23
Stable-RAG♣(Ours) 48.75 46.58 52.88 51.78 72.13 71.89 74.01 74.12 39.12 40.16 30.41 32.12 52.88 52.78
QWEN3-8B
Naive Generation 21.94 24.07 21.94 24.07 45.77 48.16 45.77 48.16 19.54 24.86 19.54 24.86 29.08 32.36
Vanilla RAG 44.65 45.34 50.55 50.67 64.35 66.29 69.62 71.03 33.14 38.66 26.17 31.33 48.08 50.55
Vanilla SFT 41.41 45.05 45.60 49.19 51.87 47.62 54.46 50.17 28.36 34.15 25.35 29.77 41.18 42.66
RetRobust 43.10 44.99 49.50 50.81 63.49 65.39 69.12 70.33 32.77 39.39 26.83 33.06 47.47 50.66
ATM 45.47 45.86 50.94 51.03 64.78 66.57 70.06 71.67 35.12 40.69 29.07 33.43 49.24 51.54
RAAT 45.13 45.87 50.12 50.03 63.12 65.17 68.54 69.88 33.54 39.06 27.2133.7547.94 50.63
Pos2Distill 44.89 45.52 50.71 50.93 64.95 66.81 69.87 71.35 33.72 39.11 26.53 31.88 48.45 50.93
Ms-PoE 44.39 45.12 50.04 50.08 64.88 66.72 69.03 70.84 32.98 38.21 25.93 31.02 47.88 50.33
Stable-RAG (Ours) 46.12 46.79 51.69 51.78 66.58 68.13 71.32 72.89 35.73 41.78 30.1533.2650.27 52.44
Stable-RAG♣(Ours) 46.94 47.13 52.12 52.38 67.11 68.79 71.74 73.40 36.89 42.94 31.77 35.78 51.10 53.40
Table 2: Main results (%) on three QA benchmarks using two retrievers. ♣denotes our method trained on exhaustive
full-permutation decoding.
5 Experiments
5.1 Experiments Setup
Datasets.We evaluate our method on three
QA benchmark datasets, including (1) Open-
Domain QA, represented by NaturalQuestions
(NQ) (Kwiatkowski et al., 2019) and Trivi-
aQA (Joshi et al., 2017); (2) Multi-Hop QA, repre-
sented by HotpotQA (Yang et al., 2018). Dataset
statistics are provided in Appendix A.1.
Evaluation Metrics.Since answer style mis-
match may cause additional variance, we follow
prior work (Zhu et al., 2024a; Peng et al., 2025;
Zhang et al., 2025) and adopt Substring Exact
Match (SubEM) andF1for evaluation. SubEM
checks whether the gold answer appears as a sub-
string in the prediction, while F1 measures token-
level overlap with the reference.
Baselines.We compare our method with the fol-
lowing baseline strategies on the same test set.
Vanilla methods includeDirect Generation,Vanilla
RAG(Lewis et al., 2020), andVanilla SFT(Zhang
et al., 2024a). Robust RAG methods includeRetRo-
bust(Yoran et al., 2024),ATM(Zhu et al., 2024a),
andRAAT(Fang et al., 2024). Positional Bias meth-
ods includePos2Distill(Wang et al., 2025b) and
Ms-PoE(Zhang et al., 2024c). The details of these
baselines are presented in Appendix A.2.
Implementation Details.We use LLaMA3-8B-
Instruct (Dubey et al., 2024) and Qwen3-8B (Yang
et al., 2025) as backbone models for experiments.To ensure high and consistent evaluation qual-
ity (Cuconasu et al., 2024) and further assess the
stability of our method under different retrieval set-
tings, we follow prior work (Zhu et al., 2024b; Xu
et al., 2024; Li et al., 2024) and use the same Top-5
Wikipedia passages retrieved by DPR (Karpukhin
et al., 2020) and Contriever-MS MARCO (Izac-
ard et al., 2021) for all baselines and our method.
Additional implementation details are provided in
Appendix A.3.
5.2 Main Results
We conduct a comprehensive comparison of Stable-
RAG against all the baseline methods, as shown
in Table 2. The results indicate the following: (i)
Overall performance.Stable-RAG consistently
achieves the best overall performance across all
the datasets with both Contriever and DPR retriev-
ers, outperforming all baselines; (ii)Effectiveness
on complex reasoning.Stable-RAG consis-
tently improves performance on both single-hop
and multi-hop QA tasks, demonstrating its abil-
ity to stabilize intermediate reasoning for complex
questions. (iii)Model generalization.Stable-
RAG performs robustly across backbone models,
indicating model-agnostic generalization.
5.3 Further Analysis
Ablation Study.We conduct an ablation study
to assess the contribution of each component in
Stable-RAG, as shown in Table 3. Removing
any component consistently degrades performance,
demonstrating that all components are essential.
6

NQ TriviaQA HotpotQA
TargetNQ
TriviaQA
HotpotQASource48.14 68.65 34.75
48.03 72.05 35.98
47.29 69.61 38.91
3040506070
NQ TriviaQA HotpotQA
Dataset0204060SubEM (%)Vanilla RAG
Ours (Contriever2Contriever)
Ours (DPR2Contriever)
RetRobust
510 15 20 25 30 35 40 45 50
Top-K303540455055SubEM (%)
Ours
Vanilla RAG
RAAT
Pos2DistillFigure 5:(Left)Cross-Dataset Generalization. We evaluate on three test sets with the Contriever retriever using
SubEM.(Middle)Cross-Retriever Transferability.(Right)Cross-Top-K Robustness. We evaluate on the NQ test
set with the Contriever retriever. All experiments are conducted on LLaMA3-8B-Instruct.
Indexcomponent Dataset
Average ARPC FA FUNQ TriviaQA HotpotQA
(a)✗ ✓ ✓37.62 61.37 28.54 42.5135.1
(b)✓ ✗ ✗47.17 71.28 37.44 51.96 0.0
(c)✓ ✗ ✓46.73 70.14 35.75 50.87 17.3
(d)✓ ✓ ✗46.70 70.6938.9352.11 0.5
Ours ✓✓✓ 48.14 72.05 38.91 53.03 21.8
Table 3: Ablation results (%) on LLaMA3-8B-Instruct
with the Contriever retriever measured by SubEM.
AR(Abstention Rate) denotes the proportion of absten-
tions on 1,000 randomly sampled questions from three
datasets when no retrieval evidence is available and the
base model cannot answer. Higher AR indicates better
awareness of model limitations and evidence availability.
Best and second-best results are bolded and underlined ,
respectively.
In particular, excluding thePCcomponent (Index
a) causes significant drops across datasets, indi-
cating the importance of partially correct signals
for stabilizing reasoning. RemovingFA(Index c)
mainly impacts overall performance, while remov-
ingFU(Index b,d) sharply reduces the abstention
rate, underscoring its role in handling unanswer-
able or hallucinated cases. Overall, Stable-RAG
achieves the best trade-off between performance
and abstention.
Comparison with Standard DPO.To isolate
the effect of the order-stability mechanism, we
compare Stable-RAG with standard DPO using
the same base model and optimization strategy,
differing only in whether reasoning consistency
across document orders is enforced. In standard
DPO, the model is trained to prefer the gold an-
swer when evidence is available over other wrong
answers obtained via sampling, or“I don’t know”
when the query is unanswerable. Results in Ta-
ble 4 demonstrate that adding the order-stability
constraint consistently improves RAG performance
across datasets and retrievers without modifying
the preference optimization framework.MethodNQ TriviaQA HotpotQA
Contriever DPR Contriever DPR Contriever DPR
Standard DPO 44.76 50.88 68.03 71.67 35.9630.43
Ours 48.14 52.02 72.04 73.43 38.9129.48
Table 4: SubEM results (%) between our method and
Standard DPO using LLaMA3-8B-Instruct.
Cross-Dataset Generalization.We further eval-
uate the transferability of Stable-RAG across differ-
ent data distributions. As shown in Figure 5 (Left),
permutation-sensitivity patterns are learned on an
in-domain dataset and directly applied to multiple
out-of-distribution datasets to assess cross-dataset
generalization. Experimental results demonstrate
that Stable-RAG exhibits robust transfer across
tasks and knowledge domains, consistently out-
performing the best baseline regardless of the
source–target dataset combination, and achieving
stable improvements in answer consistency.
Cross-Retriever Transferability.We further
evaluate the model’s transferability by training on
the DPR retriever and evaluating on the Contriever
retriever. Figure 5 (Middle) shows that the model
maintains stable performance under cross-retriever
settings, demonstrating strong transferability to
different retrieval methods. Additionally, the re-
sults of training on the Contriever retriever and
evaluating on the DPR retriever are shown in Ap-
pendix C.4.
Cross-Top-K Robustness.We train the model
under a Top-5 setting and evaluate its performance
on contexts retrieved with different Top-K values.
Experimental results in Figure 5 (Right) show that
the model maintains stable performance across
various Top-K configurations and achieves signifi-
cant improvements over corresponding baselines,
demonstrating strong generalization when handling
different numbers of candidate documents.
7

8 16 24 32
Layer Index024681012141618Avg. Cluster /glyph1197umber
12
35
610
10+(a) LLaMA3-8b-Instruct.
8 16 24 32
Layer Index024681012141618Avg. Cluster /glyph1197umber
12 (After DPO)
35 (After DPO)
610 (After DPO)
10+ (After DPO) (b) Ours.
8 16 24 32
Layer Index024681012141618Avg. Cluster /glyph1197umber
12 (After DPO)
35 (After DPO)
610 (After DPO)
10+ (After DPO)
(c) Ours (w/oFU,w/oFC).
8 16 24 32
Layer Index024681012141618Avg. Cluster /glyph1197umber
12 (After DPO)
35 (After DPO)
610 (After DPO)
10+ (After DPO) (d) Standard DPO.
Figure 6: Comparison of internal model behaviors
across Base Model (a), Ours (b), one variant of Ours
(c), and Standard DPO (d) on a random subset of 500
samples from the NQ test set with Contriever retriever.
Effect of Training Data Size.As shown in Fig-
ure 7, we analyze the effect of training sample
size on learning permutation sensitivity. Perfor-
mance improves steadily with more data and sat-
urates beyond 15k samples, indicating relatively
small datasets suffice to capture core permutation-
sensitivity patterns. However, with very limited
data (e.g., 1k), performance drops markedly, re-
flecting difficulty in modeling fine-grained order
differences. Given this trade-off, we adopt 15k
samples as default, since gains over 20k do not
justify the added computational cost.
Internal Model Behaviors after DPO.We la-
bel samples by their sensitivity according to the
Base Model and exam hidden-state clustering after
training. Figure 6b shows our method reduces clus-
ters for high-sensitivity samples, keeps medium-
sensitivity samples stable, and slightly increases
low-sensitivity clusters. Figure 6c shows training
on sensitive samples only, and Figure 6d shows
standard DPO results. We can see that the increased
clusters mainly stems from DPO-induced answer
diversity rather than direct training on sensitive
samples. For instance, for the same query"when
was the cat and mouse act introduced?"and order,
the response changes from"1913."to “"introduced
in April 1913."after DPO. Additional examples are
in the open-source repository. Overall, our method
stabilizes high-sensitivity representations while pre-
serving diversity for less sensitive samples.
1k 5k 10k 15k 20k 30k
Training Data Size3842465054Performance (%)Ours (SubEM)
Ours (F1)
Best Baseline (SubEM)
Best Baseline (F1)Figure 7: Effect of training sample size on LLaMA3-
8B-Instruct with Contriever retriever on NQ dataset.
MethodPostion of Gold Document
in Pos 1 in Pos 2 in Pos 3 in Pos 4 in Pos 5
Vanilla RAG 50.8 71.4 81.9 85.5 84.4
Vanilla SFT 47.2 66.2 74.8 80.0 82.6
RetRobust 35.5 75.5 85.3 88.6 88.9
ATM 33.7 64.2 71.8 77.4 77.8
Pos2Distill 29.5 55.8 69.4 72.8 73.2
Ms-PoE 31.4 63.8 72.1 73.9 74.3
Ours 28.3 54.7 67.3 72.6 73.0
Table 5:PSR(%) on the NQ test set with DPR retriever
across different document positions, same as Figure 1.
External Positional Robustness after DPO.
Following prior settings, we evaluate PSR on 1,000
randomly sampled instances by inserting the gold
document at varying positions in the retrieved con-
text to assess external positional robustness. As
shown in Table 5, our method consistently achieves
lower PSR across all positions than the baselines,
indicating reduced sensitivity to document ordering
and improved external robustness under positional
perturbations. Experiments in Appendix C.5 fur-
ther confirms our method’s top performance under
both original and shuffled document orders.
6 Conclusion
We identify an underexplored vulnerability in RAG:
LLMs are highly sensitive to document order, pro-
ducing divergent reasoning and inconsistent or hal-
lucinatory outputs from identical evidence. Layer-
wise analysis traces this instability to the model’s
middle and higher layers. We propose Stable-RAG,
which reduces permutation-induced uncertainty by
clustering permuted hidden states and aligning
reasoning modes via DPO optimization. Exper-
iments across multiple QA benchmarks show con-
sistent gains in accuracy, reasoning stability, and
strong transferability. Enforcing layer-wise reason-
ing constraints while reducing training costs of-
fers a promising approach to mitigate permutation-
induced hallucinations.
8

Limitations
While this work demonstrates the effectiveness of
Stable-RAG in mitigating permutation-induced hal-
lucinations, it has several limitations that warrant
further investigation.
First, our approach focuses on stabilizing rea-
soning at the final-layer representation level, with-
out explicitly enforcing layer-wise reasoning path
constraints throughout the model. Although our
analysis reveals that permutation-induced diver-
gence primarily emerges in the middle and higher
layers, Stable-RAG does not directly regularize
intermediate-layer reasoning trajectories. Incorpo-
rating explicit layer-wise constraints or trajectory-
level alignment may further improve reasoning sta-
bility, but would require more fine-grained supervi-
sion or architectural modifications, which we leave
for future work.
Second, Stable-RAG relies on spectral clustering
over document-permuted hidden representations to
estimate dominant reasoning modes and construct
preference signals for DPO alignment. While this
strategy reduces annotation cost by approximately
threefold compared to exhaustive full-permutation
decoding, it still incurs non-trivial computational
and labeling overhead. More efficient clustering
strategies, weak supervision signals, or fully unsu-
pervised alignment objectives could further reduce
annotation requirements and improve scalability.
Exploring such cost-effective supervision mecha-
nisms is an important direction for building more
robust and practical RAG systems.
Ethical Considerations
While Stable-RAG aims to enhance the robust-
ness and factual accuracy of RAG systems, several
ethical considerations remain. First, although the
method reduces hallucinations caused by document
order, it cannot guarantee fully correct outputs;
users should avoid over-reliance in high-stakes do-
mains such as healthcare, law, or finance. Sec-
ond, Stable-RAG relies on external documents for
grounding, which may contain biases or errors, po-
tentially propagating or amplifying such issues.
References
Amos Azaria and Tom Mitchell. 2023. The internal
state of an LLM knows when it’s lying. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2023, pages 967–976, Singapore. Associa-
tion for Computational Linguistics.Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Yixing Fan,
and Xueqi Cheng. 2022. Gere: Generative evidence
retrieval for fact verification. InProceedings of the
45th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval,
pages 2184–2189.
Yuhan Chen, Ang Lv, Ting-En Lin, Changyu Chen,
Yuchuan Wu, Fei Huang, Yongbin Li, and Rui Yan.
2024. Fortify the shortest stave in attention: Enhanc-
ing context awareness of large language models for
effective tool use. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 11160–
11174, Bangkok, Thailand. Association for Compu-
tational Linguistics.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, pages 719–729.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, and 1 others. 2024. The llama 3 herd of models.
arXiv e-prints, pages arXiv–2407.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InPro-
ceedings of the 30th ACM SIGKDD conference on
knowledge discovery and data mining, pages 6491–
6501.
Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiao-
jun Chen, and Ruifeng Xu. 2024. Enhancing noise
robustness of retrieval-augmented language models
with adaptive adversarial training. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 10028–10039, Bangkok, Thailand. Association
for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu,
Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min
Lin. When attention sink emerges in language mod-
els: An empirical view. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li,
Zifeng Wang, Long Le, Abhishek Kumar, James
Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Kr-
ishna, and Tomas Pfister. 2024. Found in the middle:
Calibrating positional attention bias improves long
9

context utilization. InFindings of the Association
for Computational Linguistics: ACL 2024, pages
14982–14995, Bangkok, Thailand. Association for
Computational Linguistics.
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen. 2022. LoRA: Low-rank adaptation of
large language models. InInternational Conference
on Learning Representations.
Qiushi Huang, Shuai Fu, Xubo Liu, Wenwu Wang, Tom
Ko, Yu Zhang, and Lilian Tang. 2023. Learning
retrieval augmentation for personalized dialogue gen-
eration. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2523–2540, Singapore. Association for Com-
putational Linguistics.
Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun
Song, SeungYoon Han, and Jong C. Park. 2025.
EXIT: Context-aware extractive compression for en-
hancing retrieval-augmented generation. InFind-
ings of the Association for Computational Linguis-
tics: ACL 2025, pages 4895–4924, Vienna, Austria.
Association for Computational Linguistics.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Sungjae Lee, Hoyoung Kim, Jeongyeon Hwang, Eun-
hyeok Park, and Jungseul Ok. 2025. Efficient la-
tent semantic clustering for scaling test-time com-
putation of LLMs. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
24126–24144, Suzhou, China. Association for Com-
putational Linguistics.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin.
2023. Compressing context to enhance inference ef-
ficiency of large language models. InProceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing, pages 6342–6353, Singa-
pore. Association for Computational Linguistics.
Zhonghao Li, Xuming Hu, Aiwei Liu, Kening Zheng,
Sirui Huang, and Hui Xiong. 2024. Refiner : Restruc-
ture retrieved content efficiently to advance question-
answering capabilities. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2024,
pages 8548–8572, Miami, Florida, USA. Association
for Computational Linguistics.
Zhenwen Liang, Ruosen Li, Yujun Zhou, Linfeng
Song, Dian Yu, Xinya Du, Haitao Mi, and Dong
Yu. 2025. Clue: Non-parametric verification from ex-
perience via hidden-state clustering.arXiv preprint
arXiv:2510.01591.
Hongzhan Lin, Ang Lv, Yang Song, Hengshu Zhu, Rui
Yan, and 1 others. 2024. Mixture of in-context ex-
perts enhance llms’ long context awareness.Ad-
vances in Neural Information Processing Systems,
37:79573–79596.
Andrew Ng, Michael Jordan, and Yair Weiss. 2001.
On spectral clustering: Analysis and an algorithm.
Advances in neural information processing systems,
14.
Shiyu Ni, Keping Bi, Jiafeng Guo, Lulu Yu, Baolong
Bi, and Xueqi Cheng. 2025. Towards fully exploiting
LLM internal states to enhance knowledge boundary
perception. InProceedings of the 63rd Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 24315–24329, Vi-
enna, Austria. Association for Computational Lin-
guistics.
Han Peng, Jinhao Jiang, Zican Dong, Xin Zhao, and Lei
Fang. 2025. CAFE: Retrieval head-based coarse-to-
fine information seeking to enhance multi-document
QA capability. InProceedings of the 2025 Confer-
ence on Empirical Methods in Natural Language
Processing, pages 12977–12989, Suzhou, China. As-
sociation for Computational Linguistics.
Alexander Peysakhovich and Adam Lerer. 2023. At-
tention sorting combats recency bias in long context
language models.arXiv preprint arXiv:2310.01427.
Ofir Press, Noah A Smith, and Mike Lewis. 2021.
Train short, test long: Attention with linear biases
enables input length extrapolation.arXiv preprint
arXiv:2108.12409.
10

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo-
pher D Manning, Stefano Ermon, and Chelsea Finn.
2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural
information processing systems, 36:53728–53741.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,
Wen Bo, and Yunfeng Liu. 2024. Roformer: En-
hanced transformer with rotary position embedding.
Neurocomputing, 568:127063.
Ulrike V on Luxburg. 2007. A tutorial on spectral clus-
tering.Statistics and computing, 17(4):395–416.
Leandro von Werra, Younes Belkada, Lewis Tunstall,
Edward Beeching, Tristan Thrush, Nathan Lambert,
Shengyi Huang, Kashif Rasul, and Quentin Gal-
louédec. 2020. Trl: Transformer reinforcement learn-
ing.https://github.com/huggingface/trl.
Shuting Wang, Xin Yu, Mang Wang, Weipeng Chen,
Yutao Zhu, and Zhicheng Dou. 2025a. RichRAG:
Crafting rich responses for multi-faceted queries in
retrieval-augmented generation. InProceedings of
the 31st International Conference on Computational
Linguistics, pages 11317–11333, Abu Dhabi, UAE.
Association for Computational Linguistics.
Yifei Wang, Feng Xiong, Yong Wang, Linjing Li, Xi-
angxiang Chu, and Daniel Dajun Zeng. 2025b. PO-
SITION BIAS MITIGATES POSITION BIAS: Miti-
gate position bias through inter-position knowledge
distillation. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Process-
ing, pages 1495–1512, Suzhou, China. Association
for Computational Linguistics.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven
Le Scao, Sylvain Gugger, and 3 others. 2020. Trans-
formers: State-of-the-art natural language processing.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing: System
Demonstrations, pages 38–45, Online. Association
for Computational Linguistics.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song
Han, and Mike Lewis. Efficient streaming language
models with attention sinks. InThe Twelfth Interna-
tional Conference on Learning Representations.
Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. Re-
comp: Improving retrieval-augmented lms with con-
text compression and selective augmentation. InThe
Twelfth International Conference on Learning Repre-
sentations.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InEMNLP, pages 2369–2380, Brussels, Belgium.
Association for Computational Linguistics.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Be-
rant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InThe Twelfth
International Conference on Learning Representa-
tions.
Hanning Zhang, Shizhe Diao, Yong Lin, Yi Fung, Qing
Lian, Xingyao Wang, Yangyi Chen, Heng Ji, and
Tong Zhang. 2024a. R-tuning: Instructing large lan-
guage models to say ‘I don’t know’. InProceedings
of the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), pages 7113–7139, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei
Zheng, Yongxin Tong, and Zhiming Zheng. 2025.
Less is more: Compact clue selection for efficient
retrieval-augmented generation reasoning.arXiv
preprint arXiv:2502.11811.
Qianchi Zhang, Hainan Zhang, Liang Pang, Hongwei
Zheng, and Zhiming Zheng. 2024b. Adacomp: Ex-
tractive context compression with adaptive predic-
tor for retrieval-augmented large language models.
arXiv preprint arXiv:2409.01579.
Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei
Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu,
Zhangyang Wang, and 1 others. 2024c. Found in
the middle: How language models use long contexts
better via plug-and-play positional encoding.Ad-
vances in Neural Information Processing Systems,
37:60755–60775.
Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian,
Zheng Liu, Chaozhuo Li, Zhicheng Dou, Tsung-
Yi Ho, and Philip S Yu. 2024. Trustworthiness in
retrieval-augmented generation systems: A survey.
arXiv preprint arXiv:2409.10102.
Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, and
Lei Sha. 2024a. ATM: Adversarial tuning multi-
agent system makes a robust retrieval-augmented
generator. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 10902–10919, Miami, Florida, USA. Associa-
tion for Computational Linguistics.
Kun Zhu, Xiaocheng Feng, Xiyuan Du, Yuxuan Gu,
Weijiang Yu, Haotian Wang, Qianglong Chen, Zheng
Chu, Jingchang Chen, and Bing Qin. 2024b. An in-
formation bottleneck perspective for effective noise
filtering on retrieval-augmented generation. InPro-
ceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pages 1044–1069, Bangkok, Thailand. As-
sociation for Computational Linguistics.
11

Appendix
Contents
A Implementation Details 12
A.1 Datasets . . . . . . . . . . . . . . 12
A.2 Baseline Details . . . . . . . . . . 12
A.3 Training Details . . . . . . . . . . 13
A.4 Prompts . . . . . . . . . . . . . . 13
B Mathematical Derivations 13
B.1 Spectral Clustering on Hidden States 13
B.2 Similarity Graph and Adjacency
Matrix . . . . . . . . . . . . . . . 13
B.3 Degree Matrix and Normalized
Laplacian . . . . . . . . . . . . . 14
B.4 Eigen-decomposition and Deter-
mining Cluster Number . . . . . . 14
B.5 Spectral Embedding and Clustering 14
C More Experimental Results 14
C.1 Permutation Sensitivity in Qwen3
Models . . . . . . . . . . . . . . 14
C.2 Structural Instability Across
Model Families . . . . . . . . . . 14
C.3 Visualization of Layer-wise Hid-
den States . . . . . . . . . . . . . 15
C.4 Cross-Retriever Transferability . . 15
C.5 Original vs. Shuffled Order . . . . 16
A Implementation Details
A.1 Datasets
We conduct experiments on three widely used QA
datasets that cover both single-hop and multi-hop
question-answering scenarios. Table 6 summa-
rizes the key statistics of these datasets. Specif-
ically,NQ(Kwiatkowski et al., 2019) andTrivi-
aQA(Joshi et al., 2017) are representative single-
hop datasets, where each question can typically
be answered using information from a single pas-
sage retrieved from the corpus. These datasets
primarily evaluate a model’s ability to locate and
extract factual evidence efficiently. In contrast,
HotpotQA(Yang et al., 2018) is a challenging
multi-hop dataset that requires integrating and rea-
soning over multiple pieces of evidence distributed
across different documents to derive the final an-
swer. This dataset is particularly useful for testinga model’s reasoning and compositional understand-
ing capabilities. Together, these datasets provide a
comprehensive benchmark for evaluating both the
retrieval quality and reasoning robustness of our
proposed method under diverse task settings.
Dataset Type # Train # Dev # Test
NQ single-hop 79.1k 8.7k 3.6k
TriviaQA single-hop 78.7k 11.3k 8.8k
HotpotQA multi-hop 88.9k 5.6k 5.6k
Table 6: Statistics for the datasets.
A.2 Baseline Details
We compare Stable-RAG with the following base-
line strategies. To ensure a fair comparison, all
methods are evaluated on the same test set and
retrieved set.
Vanilla Methods.(i)Direct Generation. This
baseline relies solely on the generator’s parametric
knowledge to produce answers without consulting
any retrieved documents. (ii)Vanilla RAG(Lewis
et al., 2020). This baseline concatenates all re-
trieved documents as model input without any addi-
tional processing. (iii)Vanilla SFT. We implement
vanillla SFT following Zhang et al. (2024a). For
each training example, this baseline uses the gold
answer as the training label if it appears in the re-
trieved documents; otherwise, it assigns“I don’t
know”as the training label to guide the model to
abstain when the necessary information is missing.
Robust RAG.(i)RetRobust(Yoran et al., 2024).
This baseline improves retrieval-augmented QA
models by filtering out irrelevant retrieved passages
and fine-tuning the model on a mix of relevant and
irrelevant contexts, enabling it to leverage relevant
information while remaining robust to irrelevant
content. (ii)ATM(Zhu et al., 2024a). This baseline
optimizes a retrieval-augmented Generator using an
Adversarial Tuning Multi-agent system, where an
auxiliary Attacker agent iteratively steers the Gen-
erator to better discriminate useful documents from
noisy or fabricated ones, improving robustness and
performance on knowledge-intensive question an-
swering tasks. (iii)RAAT(Fang et al., 2024). This
baseline dynamically adjusts the model’s learning
process in response to various types of retrieval
noise through adaptive adversarial training, while
employing multi-task learning to enable the model
to internally recognize and handle noisy contexts,
12

thereby improving robustness and answer quality
in retrieval-augmented generation.
Positional Bias.(i)Pos2Distill(Wang et al.,
2025b). This baseline mitigates positional bias
in long-context tasks by transferring knowledge
from advantageous positions to less favorable ones
through position-to-position knowledge distillation.
(ii)Ms-PoE(Zhang et al., 2024c). This baseline
uses Multi-scale Positional Encoding to mitigate
the "lost-in-the-middle" issue in LLMs by rescaling
positional indices and assigning different scaling ra-
tios to attention heads, enabling multi-scale context
fusion without fine-tuning or extra overhead.
A.3 Training Details
We use LLaMA3-8B-Instruct1(Dubey et al., 2024)
and Qwen3-8B2(Yang et al., 2025) as backbone
models for experiments. We implement our DPO
training pipeline using the HuggingFace Transform-
ers (Wolf et al., 2020) and TRL libraries (von Werra
et al., 2020), incorporating PEFT LoRA (Hu et al.,
2022) for parameter-efficient fine-tuning. Both
the base model and reference model are initialized
from pre-trained checkpoints, with the reference
model kept in evaluation mode to provide stable
policy targets during training. Each dataset is ran-
domly shuffled and split into 85% training and 15%
validation samples, with a maximum of 18,000
samples per dataset to control computational over-
head. To guarantee reproducible results, we use
a fixed random seed with a value of 42. LoRA is
applied to all projection layers, including query,
key, value, output, gate, up, and down projections,
with rank r= 128 , alpha = 128 , dropout = 0
and no additional bias terms. The DPO configura-
tion uses a per-device batch size of 2 with gradient
accumulation of 8, a learning rate of 5×10−6, a
linear warmup ratio of 0.1, and a preference scaling
hyperparameter βof 0.4. We train LLaMA-3-8B-
Instruct for a single epoch and Qwen3-8B for two
epochs on two NVIDIA RTX PRO 6000 GPUs,
with each epoch taking roughly two hours. After
training, the fine-tuned models and tokenizers are
saved for downstream evaluation.
Notably, we set the generation temperature to
0.01 during data construction and inference, effec-
tively approximating greedy decoding to ensure
that output variations primarily reflect document-
1https://huggingface.co/meta-llama/
Meta-Llama-3-8B-Instruct
2https://huggingface.co/Qwen/Qwen3-8B<system>
You are a helpful, respectful, and honest
assistant. Answer the question with couple
of words using the provided documents.
For example: Question: What is the capital
of France? Output: Paris.
</system>
<user>
Question:{query}
Documents:
Doc1:{Document 1}
Doc2:{Document 2}
......
</user>
Table 7: Prompt for the backbone LLMs.
order sensitivity rather than sampling randomness.
A.4 Prompts
We adopt a system-user style prompting scheme
to guide the backbone LLMs to generate concise,
document-grounded answers, as presented in Ta-
ble 7.
B Mathematical Derivations
We employ spectral clustering on hidden states to
identify dominant reasoning modes across permu-
tations of retrieved documents. Compared with
conventional clustering methods, spectral cluster-
ing captures the global structure of the hidden state
space. This enables Stable-RAG to robustly group
similar reasoning behaviors, reduce noise from spu-
rious variations, and improve the consistency of
preference signals used for DPO alignment.
B.1 Spectral Clustering on Hidden States
Spectral clustering is applied to the hidden states
matrix
H= [h(1), h(2), . . . , h(N)]⊤∈RN×d
to adaptively determine the number of clusters and
capture the global structure of the hidden state
space, where each cluster corresponds to a latent
reasoning mode (Lee et al., 2025).
B.2 Similarity Graph and Adjacency Matrix
We construct a weighted similarity graph G=
(V, E) where each node corresponds to a hidden
13

state h(i)and edges encode pairwise similarities.
The adjacency matrix A∈RN×Nis computed as
the exponential of the cosine distance:
Aij= exp
−1−h(i)·h(j)
∥h(i)∥∥h(j)∥
σ
,
where σis a hyperparameter controlling sensitivity.
B.3 Degree Matrix and Normalized Laplacian
The degree matrix Dis a diagonal matrix with
entries
Dii=NX
j=1Aij.
The normalized graph Laplacian is
L=I−D−1/2AD−1/2,
whereIis the identity matrix.
B.4 Eigen-decomposition and Determining
Cluster Number
Letλ1≤ ··· ≤λ Nbe the eigenvalues of L. Define
the consecutive eigengaps as
gapi=λi+1−λi.
The number of clustersKis set adaptively as
K= max 
2,(arg max
igapi) + 1
,
ensuring clear separation between latent reasoning
modes following standard practice (Ng et al., 2001;
V on Luxburg, 2007).
B.5 Spectral Embedding and Clustering
We then compute the first Keigenvectors of L, nor-
malize each row to unit length, and apply standard
clustering to assign each hidden state h(i)to one of
the clusters
C1, C2, . . . , C K,
exactly following the procedure described in the
main text.
C More Experimental Results
C.1 Permutation Sensitivity in Qwen3 Models
We further investigate whether document-order sen-
sitivity generalizes to different model families by
reporting Perturbation Success Rate (PSR) results
on the Qwen3 series. Following the same evalua-
tion protocol as in Figure 1, we fix the gold docu-
ment in different positions and measure the propor-
tion of document-order perturbations that lead to
Qwen3-1.7B Qwen3-4B Qwen3-8B Qwen3-14B20406080PSR (%)65.9
46.0
37.7
33.478.6
67.1
63.4
51.783.1
74.6
68.4
57.182.8
74.6
69.7
57.973.4
70.2
66.8
53.2Gold Doc in Pos 1
Gold Doc in Pos 2
Gold Doc in Pos 3
Gold Doc in Pos 4
Gold Doc in Pos 5Figure 8:Perturbation Success Rate (PSR)on the NQ
test set across different Qwen3 models. PSR is com-
puted as the proportion of successful document-order
perturbations to produce hallucination results among
1000 randomly sampled instances, with the gold docu-
ment fixed in the different positions.
hallucinated outputs over 1,000 randomly sampled
instances on the NQ test set.
Figure 8 compares the PSR trends of the Qwen3
models with those observed in the LLaMA-3 In-
struct series. Overall, Qwen3 models exhibit clear
document-order sensitivity across all model sizes.
When the gold document is placed at early posi-
tions, the PSR is relatively low, indicating stronger
robustness to document-order perturbations. How-
ever, as the gold document is shifted to later po-
sitions, PSR increases substantially, suggesting a
higher likelihood of hallucinations induced purely
by document reordering.
We observe a consistent monotonic pattern
across Qwen3 variants: PSR generally rises from
Top-1 to Top-3 or Top-4 and slightly saturates or
declines afterward. This behavior closely mirrors
the trends observed in LLaMA-3 models, despite
differences in model architecture and pretraining
data. Moreover, smaller Qwen3 models tend to ex-
hibit higher sensitivity to document order changes,
while larger models demonstrate comparatively im-
proved robustness, though the issue remains non-
negligible even at larger scales.
These results indicate that document-order sen-
sitivity is not specific to a particular model family
but rather a general phenomenon shared across con-
temporary large language models. The consistent
patterns across both LLaMA-3 and Qwen3 series
further motivate the need for order-robust RAG
methods.
C.2 Structural Instability Across Model
Families
We provide additional visualizations of the struc-
tural instability in internal reasoning dynamics for
both the LLaMA-3 and Qwen3 model families as
14

4 8 12 16
Layer Index024681012141618Avg. Cluster /glyph1197umber
LLaMA-3.2-1B-Instruct
12
35
610
10+
7 14 21 28
Layer Index024681012141618Avg. Cluster /glyph1197umber
LLaMA-3.2-3B-Instruct
12
35
610
10+
8 16 24 32
Layer Index024681012141618Avg. Cluster /glyph1197umber
LLaMA3-8B-Instruct
12
35
610
10+Figure 9: Hidden-state clustering behaviors across layers for LLaMA3 series models on the NQ train set with DPR
retriever, using 1,000 random sampled instances. Different colored lines indicate the number of clusters of final
reasoning states produced by the LLM under all 5!(= 120) permutations of the Top-5 retrieved documents (e.g., the
green line indicates 3–5 cluster states).
7 14 21 28
Layer Index024681012141618Avg. Cluster /glyph1197umber
Qwen3-1.7B
12
35
610
10+
6 12 18 24 30 36
Layer Index02468101214161820Avg. Cluster /glyph1197umber
Qwen3-4B
12
35
610
10+
6 12 18 24 30 36
Layer Index024681012141618Avg. Cluster /glyph1197umber
Qwen3-8B
12
35
610
10+
8 16 24 32 40
Layer Index024681012141618Avg. Cluster /glyph1197umber
Qwen3-14B
12
35
610
10+
Figure 10: Hidden-state clustering behaviors across layers for Qwen3 series models on the HotpotQA train set with
Contriever retriever, using 1,000 random sampled instances. Different colored lines indicate the number of clusters
of final reasoning states produced by the LLM under all 5!(= 120) permutations of the Top-5 retrieved documents
(e.g., the green line indicates 3–5 cluster states).
shown in Figure 9 and Figure 10. Following the
same analysis protocol as in the main paper, we
examine how document permutations induce di-
vergence in hidden representations across different
model layers.
Despite differences in architecture, scale, and
pretraining data, both model families exhibit highly
consistent patterns of structural instability. Specifi-
cally, hidden representations in shallow layers re-
main relatively concentrated under document per-
mutations, while substantial divergence emerges in
the middle layers and becomes more pronounced in
higher layers. Moreover, samples exhibiting higher
permutation sensitivity consistently show greater
representational divergence than stable samples,
with this effect primarily localized to the middle
layers.
These observations suggest that permutation sen-
sitivity originates from a shared structural instabil-
ity in the reasoning dynamics of large language
models rather than from model-specific design
choices. The consistent trends observed across
both LLaMA-3 and Qwen3 families further support
the necessity of addressing structural instability to
improve the robustness of RAG systems.C.3 Visualization of Layer-wise Hidden
States
Figure 12 shows LLaMA3-8B-Instruct on NQ us-
ing the Contriever retriever, illustrating the hidden
state evolution across all layers for a selected exam-
ple. Figure 13 displays Qwen3-8B on HotpotQA
dataset using Contriever dataset, showing the layer-
wise progression of hidden states for a representa-
tive sample. In both cases, shallow layers exhibit
mixed clusters with points corresponding to differ-
ent answers interleaved, while deeper layers form
increasingly well-separated clusters according to
the final answers. These visualizations reinforce
that the structural evolution of reasoning trajecto-
ries, as observed in the main experiments, is con-
sistent across multiple models and datasets.
C.4 Cross-Retriever Transferability
As reported in Section 5.3, we evaluated transfer
from DPR to Contriever. Here, we additionally test
transfer from Contriever to DPR, as shown in Fig-
ure 11. Both experiments confirm that Stable-RAG
consistently improves answer consistency and re-
duces permutation-induced variance across differ-
ent retrievers, demonstrating robust cross-retriever
generalization.
15

MethodNQ TriviaQA HotpotQA
Original Shuffled Drop Original Shuffled Drop Original Shuffled Drop
Vanilla SFT 42.10 36.43 5.67 55.52 53.19 2.33 27.25 22.48 4.77
RetRobust 41.82 38.06 3.76 64.85 62.86 1.99 31.46 29.18 2.28
ATM 43.75 42.47 1.28 66.37 63.60 2.77 34.36 32.46 1.90
RAAT 42.33 40.54 1.79 65.58 62.19 3.39 33.58 29.75 3.83
Pos2Distill 44.58 43.63 0.95 64.13 63.57 0.56 32.73 32.090.64
Ms-PoE 40.32 39.17 1.15 64.21 62.96 1.25 30.17 29.14 1.03
Ours (Stable-RAG) 48.14 47.23 0.91 72.05 71.76 0.29 38.91 37.50 1.41
Table 8: Performance comparison of LLaMA3-8B-Instruct with Contriever retriever under original and shuffled
document order across three QA datasets. We report SubEM for evaluation.
NQ TriviaQA HotpotQA
Dataset0204060SubEM (%)Vanilla RAG
Ours (DPR2DPR)
Ours (Contriever2DPR)
RetRobust
Figure 11: Cross-Retriever Transferability.
C.5 Original vs. Shuffled Order
Table 8 presents a comparison of answer perfor-
mance under the original document order and a
randomly shuffled order across three QA datasets.
Our method achieves the highest SubEM scores
in both original and shuffled conditions across all
datasets, demonstrating its robustness to retrieval
order permutations and its ability to maintain stable
answer consistency.
16

Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6
Layer 7 Layer 8 Layer 9 Layer 10 Layer 11 Layer 12
Layer 13 Layer 14 Layer 15 Layer 16 Layer 17 Layer 18
Layer 19 Layer 20 Layer 21 Layer 22 Layer 23 Layer 24
Layer 25 Layer 26 Layer 27 Layer 28 Layer 29 Layer 30
Layer 31 Layer 32Question: what is the liquid in a magic 8 ball?  Correct Answer(s): Alcohol.
Alcohol. Water.Figure 12: 2D PCA visualization of hidden state representations across all layers in LLaMA3-8B-Instruct for a
single example. Each point corresponds to a document order, and its color represents the model’s final answer.
17

Layer 1 Layer 2 Layer 3 Layer 4 Layer 5 Layer 6
Layer 7 Layer 8 Layer 9 Layer 10 Layer 11 Layer 12
Layer 13 Layer 14 Layer 15 Layer 16 Layer 17 Layer 18
Layer 19 Layer 20 Layer 21 Layer 22 Layer 23 Layer 24
Layer 25 Layer 26 Layer 27 Layer 28 Layer 29 Layer 30
Layer 31 Layer 32 Layer 33 Layer 34 Layer 35 Layer 36Question: Which band has more members, Muse or The Raconteurs?  Correct Answer(s): The Raconteurs.
Muse has more members. Muse. The Raconteurs.Figure 13: 2D PCA visualization of hidden state representations across all layers in Qwen3-8B for a single example.
Each point corresponds to a document order, and its color represents the model’s final answer.
18