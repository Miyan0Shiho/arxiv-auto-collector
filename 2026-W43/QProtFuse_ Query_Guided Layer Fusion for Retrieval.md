# QProtFuse: Query-Guided Layer Fusion for Retrieval-Augmented Protein Function Prediction

**Authors**: Mianzhi Pan, Peishuo Liu, Yawen Ouyang, Botian Wang, Yiming Rong, Hao Zhou, Jianbing Zhang

**Published**: 2026-10-19 14:47:25

**PDF URL**: [https://arxiv.org/pdf/2610.30180v1](https://arxiv.org/pdf/2610.30180v1)

## Abstract
Predicting protein functions from amino acid sequences is essential to understanding biological processes. Large language models (LLMs) offer flexible function descriptions to natural-language queries, yet their limited protein understanding motivates retrieval-augmented generation (RAG). Existing protein RAG methods primarily expand external retrieval signals while overlooking the rich knowledge encoded within a single protein language model (PLM). We show that retrieval performance varies across PLM layers and biological tasks, making conventional final-layer selection suboptimal. To this end, we propose QProtFuse, which uses queries to adaptively fuse multi-layer protein features and jointly learns protein and annotation retrieval through intra-modal and cross-modal contrastive objectives. On Prot-Inst-OOD, QProtFuse achieves the best protein and annotation retrieval scores, reaching 16.4\% Recall@1 with annotation reranking. These gains improve downstream RAG-based function descriptions, yielding the highest LLM-as-a-judge scores for both Qwen3-8B (6.22/10) and DeepSeek-V4-Pro (6.68/10). Ablations and weight analyses demonstrate the value of query-guided fusion and complementary training objectives for exploiting a PLM's internal knowledge.

## Full Text


<!-- PDF content starts -->

QProtFuse: Query-Guided Layer Fusion for Retrieval-Augmented Protein
Function Prediction
Mianzhi Pan1,2Peishuo Liu1Yawen Ouyang3Botian Wang2
Yiming Rong5Hao Zhou2*Jianbing Zhang1,4*
1National Key Laboratory for Novel Software Technology &
School of Artificial Intelligence, Nanjing University, China
2Institute of AI Industry Research (AIR), Tsinghua University3Shanghai AI Lab
4Chemistry and Biomedicine Innovation Center (ChemBIC), Nanjing University
5University of Chinese Academy of Sciences, Beijing
Abstract
Predicting protein functions from amino acid
sequences is essential to understanding biologi-
cal processes. Large language models (LLMs)
offer flexible function descriptions to natural-
language queries, yet their limited protein un-
derstanding motivates retrieval-augmented gen-
eration (RAG). Existing protein RAG meth-
ods primarily expand external retrieval sig-
nals while overlooking the rich knowledge en-
coded within a single protein language model
(PLM). We show that retrieval performance
varies across PLM layers and biological tasks,
making conventional final-layer selection sub-
optimal. To this end, we propose QProtFuse,
which uses queries to adaptively fuse multi-
layer protein features and jointly learns protein
and annotation retrieval through intra-modal
and cross-modal contrastive objectives. On
Prot-Inst-OOD, QProtFuse achieves the best
protein and annotation retrieval scores, reach-
ing 16.4% Recall@1 with annotation rerank-
ing. These gains improve downstream RAG-
based function descriptions, yielding the high-
est LLM-as-a-judge scores for both Qwen3-8B
(6.22/10) and DeepSeek-V4-Pro (6.68/10). Ab-
lations and weight analyses demonstrate the
value of query-guided fusion and complemen-
tary training objectives for exploiting a PLM’s
internal knowledge.1
1 Introduction
Proteins are essential to biological processes.
Predicting their functions from amino acid se-
quences is therefore a central task in computa-
tional biology. Traditional sequence analysis uses
BLAST (Altschul et al., 1990) to find homologs
and InterPro (Blum et al., 2025) to identify fam-
ily and domain signatures. These methods de-
pend on informative matches and cannot directly
*Corresponding author
1Data and code are available at https://github.com/
panmianzhi/QProtFuse.
Recall@1 Recall@10
Retrieval Performance051015202530Score (%)ESM2
ESM2 + MMseqs2
ESM2 + ProstT5
ESM2 all-layer avg.
1 3 5 7 9 1 1 1 3 1 5 1 7 1 9 2 1 2 3 2 5 2 7 2 9 3 1 3 3
ESM2-650M layer ID51 01 52 0MRR@10 (%)Cat. act.
Dom. mot.
Gen. func.
Prot. func. (b)  (c) (a)
Identify the protein's 
function  and the 
biological process 
it may contribute
to:  MTETTL...Protein
+
Query
RetrieverProtein
Database
LLMAnswer:  It exhibits  DNA
binding  and  zinc ion 
binding , and may contri-
bute  to regul ation of DNA -
templated transcription .Top-K Samples
MMseqs2 PLMs
Figure 1:(a)A typical retrieval-augmented pipeline for
protein function description.(b)Layer-wise retrieval
performance of ESM2-650M across four Prot-Inst-OOD
tasks.(c)Retrieval performance comparison of final-
layer ESM, ESM with MMseqs2, ESM with ProstT5,
and averaged ESM layer features.
answer specific biological questions. More re-
cently, broader large language models (LLMs) for
life-science research have begun to tackle protein-
related tasks (OpenAI, 2026; Anthropic, 2026).
Protein-oriented LLMs incorporate amino acid se-
quences through direct tokenization (Wang et al.,
2024b; Bai et al., 2025; Qiu et al., 2026) or dedi-
cated protein encoders (Zhuo et al., 2024; Fei et al.,
2025; Zhou et al., 2026). Together, these models
offer the potential to answer complex biological
questions in free-form by leveraging their internal
pretrained knowledge and reasoning capabilities.
Despite these advances, the inherent gap be-
tween proteins and natural language remains a ma-
jor obstacle to reliable protein understanding by
LLMs (Zhuang et al., 2026). For protein function
description, Wu et al. (2025) show that retrieval-
based methods can outperform direct LLM gen-
eration. This highlights the value of external bio-
logical knowledge, motivating retrieval-augmented
generation (RAG) approaches (Wu et al., 2025;
1
arXiv:2610.30180v1  [cs.IR]  19 Oct 2026

Ma et al., 2026; Ding et al., 2026). As illustrated
in Figure 1(a), these approaches retrieve relevant
proteins and annotations to enhance LLM-based
protein function description.
However, current protein RAG methods focus
on incorporating external retrieval signals, such
as those from sequence alignment tools (Altschul
et al., 1990; Steinegger and Söding, 2017) and
additional protein language models (PLMs) (Lin
et al., 2023; Wang et al., 2024a; Heinzinger et al.,
2024), while leaving the knowledge within a single
PLM insufficiently explored. Previous studies re-
veal that PLMs encode rich and distinct biological
features across different layers (Rao et al., 2021;
Lin et al., 2023; Vig et al., 2021; Kumar and Jha,
2025). These findings raise two questions:
Q1:Do different PLM layers exhibit distinct
strengths in protein retrieval?
Q2:If so, how can we effectively use this
internal knowledge to improve protein RAG?
To answer these questions, we first probe the lay-
erwise retrieval performance of ESM-2 (Lin et al.,
2023) on Prot-Inst-OOD (Wu et al., 2025). As
shown in Figure 1(b), the four subtasks achieve
their best retrieval performance using features from
different intermediate layers. These observations
answer Q1 and show that relying solely on final-
layer features is suboptimal. As a first step toward
Q2, we test a simple fusion strategy by uniformly
averaging features across all layers. Even this sim-
ple strategy outperforms those incorporating MM-
seqs2 sequence similarity or ProstT5 features (Fig-
ure 1(c)).2However, uniform averaging still gen-
erally falls short of task-specific best-performing
layers (see Appendix A). This gap calls for a more
adaptive fusion strategy that exploits the distinct
strengths of different layers.
Inspired by multi-layer visual feature fusion in
multimodal LLMs (Yao et al., 2024; Lin et al.,
2025, 2026), we proposeQProtFuse(Query-
guided multi-layer Protein Feature Fusion) for
retrieval-augmented protein function prediction.
QProtFuse leverages the query toadaptively as-
sign layer-wise weightsand fuse multi-layer fea-
tures. To optimize this fusion, we propose adual-
objective contrastive learningframework that
couples an intra-modal protein-to-protein objective
2For the two compared methods, we follow the standard
retrieval designs of RAPM (Wu et al., 2025) and the structure-
only variant of ProtQueSt (Ma et al., 2026), respectively.with an inter-modal protein-to-annotation objec-
tive. These two objectives are mutually reinforcing
and jointly enable both protein and annotation re-
trieval during inference, thereby broadening the
applicability of QProtFuse. On Prot-Inst-OOD,
QProtFuse achievessuperior protein and anno-
tation retrieval performancecompared with all
baselines, reaching a Recall@1 of 16.4% and Re-
call@10 of 25.9% with our reranking configuration.
These retrieval gains improve downstream func-
tion descriptions, yielding thehighest LLM-as-a-
judge scoresamong the evaluated methods, with
6.22/10 on Qwen3-8B and 6.68/10 on DeepSeek-
V4-Pro.
Our contributions are summarized as follows:
•We present the first systematic study of multi-
layer PLM features for protein retrieval, showing
that the best layer varies across tasks and using
only final-layer features is suboptimal.
•We propose QProtFuse, which combines query-
guided layer fusion with joint protein-to-protein
and protein-to-annotation alignment to exploit
the internal knowledge of PLMs.
•We achieve SOTA retrieval and RAG-based pro-
tein function description performance, demon-
strating the value of adaptive layer fusion for
protein search and annotation.
2 Related Work
2.1 LLMs for Protein Understanding
Protein-oriented LLMs mainly integrate proteins
through direct tokenization or encoder-based align-
ment. Direct tokenization approaches model amino-
acid tokens alongside natural language (Wang et al.,
2024b; Xia et al., 2025), with further developments
in both tokenization and training objectives, where
Intern-S1 (Bai et al., 2025) introduces modality-
specific tokenization and AMix-2 (Qiu et al., 2026)
adopts block-wise diffusion for joint protein–text
modeling. Encoder-based approaches instead use
projectors or Q-Formers to align protein features
with an LLM (Zhuo et al., 2024; Liu et al., 2024;
Zhou et al., 2026; Wang et al., 2025). Within
this paradigm, Prot2Text-V2 (Fei et al., 2025)
strengthens protein–text alignment through con-
trastive learning, while BioReason-Pro (Fallah-
pour et al., 2026) emphasizes biological reasoning
through reinforcement learning. Despite these ad-
vances, Zhuang et al. (2026) identify limitations
in both approaches, motivating an alternative that
supplies LLMs with structured biological context
2

❄ ESM
Transformer Block
Transformer Block
Transformer Block
Transformer Block
Transformer Block
Transformer Block... ... ... ......Pooling......"Identify the function of the protein 
and the biological process ..."
❄ Text Encoder1. Protein    Protein  Training (two objectives) Model Architecture Training & Inference 
2. Protein     Annotation  
Query
proteinCandidate
proteinQuery
proteinCandidate
annotation
Query
 Query
SigLIP Loss(       ,       ) SigLIP Loss(       ,       )
MLP🔥...
Weight
Weighted SumSoftmax
GroupInference (three modes)
+
+
MTETTLGASNELLAEL T AEIV AA Y ...
Top-K samples 
(from #1)Re-rank 
by #2
1
2...1. Protein 
   retrieval
2. Annotation  
retrieval
3. Sample
    re-rankPooling
PoolingGroupwise
features
MLP🔥RouterFigure 2: Overview of QProtFuse Framework.Left.Features from a frozen ESM are pooled within consecutive
layer groups and fused using query-dependent weights.Right.During training, the model is jointly optimized for
intra-modal protein-protein and inter-modal protein-to-annotation alignment. During inference, one checkpoint
supports protein retrieval, annotation retrieval, and protein retrieval with annotation-based reranking.
extracted by external tools.
Reliable protein function description remains
challenging for LLMs due to the inherent modality
gap. Wu et al. (2025) show that simple retrieval
can outperform protein-oriented LLMs and pro-
pose a dual-index retrieval method combining MM-
seqs2 (Steinegger and Söding, 2017) and ESM (Lin
et al., 2023). ProtQueSt (Ma et al., 2026) ensem-
bles ESM and ProstT5 (Heinzinger et al., 2024)
features with query-conditioned contrastive learn-
ing to retrieve annotations suited to the biological
question. These methods improve retrieval through
more external similarity signals. Our work focuses
on the knowledge within a single PLM, learning to
combine its layer features according to the query.
2.2 Layerwise Probing in PLMs
Recent studies have probed the biological knowl-
edge encoded within PLMs. Rao et al. (2021) and
Lin et al. (2023) show that PLMs learn structural
information from sequences alone that supports
residue contact prediction and atomic-level folding.
More specifically, Vig et al. (2021) find that atten-
tion captures residue contacts and binding sites,
with deeper layers attending to increasingly com-
plex biophysical properties. Kumar and Jha (2025)
probe ESM-2 layers for kinase functional predic-
tion and show that selected intermediate layers out-
perform the final layer. From a geometric perspec-tive, Beshkov and Malthe-Sorenssen (2026) trace
changes in feature geometry across depth and find
that structural fidelity is highest near, but before,
the final layer. Together, these studies establish that
the most useful biological features need not occur
at the encoder output. We investigate how these
differences affect protein retrieval and how query-
guided fusion can turn them into more effective
retrieval features.
2.3 Multi-Layer Feature Fusion
Motivated by the observation that earlier visual
layers preserve visual details that are not fully cap-
tured by final layers, recent work has studied multi-
layer fusion in multimodal LLMs. Dense Con-
nector (Yao et al., 2024) explores integrating fea-
tures from multiple visual encoder layers to enrich
visual input. Chen et al. (2025) show that shal-
low and middle layers are particularly useful for
fine-grained visual tasks and develop a lightweight
layer-concat fusion strategy. Li et al. (2026) and
Lin et al. (2026) use textual queries to adaptively
determine the weights of visual features from dif-
ferent layers. Lin et al. (2025) conduct a systematic
comparison and find that combining features from
different encoder layers improves generalization
and directly fusing visual features at the LLM in-
put, rather than injecting them into its intermediate
layers, provides consistently strong performance.
3

We extend layer fusion to PLMs and adapt it to re-
trieval through joint protein-to-protein and protein-
to-description contrastive objectives.
3 Method
We study retrieval-augmented protein function de-
scription given natural-language queries. Each
sample is a tuple (p, q, d) comprising a protein se-
quence p, a natural-language query q, and its func-
tional description d. Given a reference database
D={(p i, qi, di)}M
i=1, retrieval-augmented predic-
tion selects Kreference samples SKto enhance
LLM generation of d. For a fixed LLM generator
Pϕ, the ideal evidence set satisfies
S∗
K= arg max
S⊆D,|S|=KlogP ϕ(d|p, q,S).(1)
Our goal is to train a retriever that selects such evi-
dence. However, directly optimizing Equation (1)
is expensive, as the M
K
possible combinations
form a prohibitively large search space, and the
evaluation requires costly LLM inference. Prior
work shows that improving retrieved evidence can
enhance generation quality (Lewis et al., 2020; Shi
et al., 2024). We therefore focus on retrieval effec-
tiveness as a proxy for evidence quality:
θ∗= arg max
θE(p,q,d)∼T
U 
πθ(p, q),R(d)
,
(2)
whereTis the training set, πθ(p, q) ranks the sam-
ples in Dby(p, q) , andR(d) is the set of oracle
samples in Dthat match description d.Umeasures
ranking quality, such as Recall@K.
Figure 2 illustrates the QProtFuse framework.
We evenly divide consecutive PLM layers into
groups and mean-pool their features, then use the
query to predict group weights for adaptive fu-
sion (Section 3.1). We train QProtFuse with two
complementary objectives: intra-modal protein-to-
protein (PP) alignment and cross-modal protein-
to-annotation (PA) alignment (Section 3.2). The
same model supports protein retrieval, annotation
retrieval, and two-stage retrieval with reranking
(Section 3.3).
3.1 Query-Guided Layer Fusion
Grouping layer features.For a protein pwith
nresidues, a frozen ESM2-650M encoder (Lin
et al., 2023) produces residue features H(ℓ)(p)∈
Rn×1280at each of its L= 33 transformer layers.
We average over residues to obtain a protein feature
51015202530
ESM layer ID51015202530ESM layer ID
0.850.900.95
Avg. cosine similarityFigure 3: Pairwise cosine similarity between ESM-2
layer features, averaged over training proteins. Neigh-
boring layers typically produce similar features, moti-
vating consecutive-layer grouping.
for each layer. As shown in Figure 3, neighboring
ESM layers typically yield highly similar protein
features, suggesting redundancy in routing over
every layer independently. We therefore follow
the consecutive-layer grouping strategy of Li et al.
(2026), and partition the layers into G= 7 groups
{Gg}G
g=1and average within each group:
lℓ(p) =1
nnX
t=1H(ℓ)
t(p),
gg(p) =1
|Gg|X
ℓ∈G glℓ(p).(3)
Grouping reduces the number of fusion weights
while retaining features from different depths.
Query-guided fusion.As shown in prior stud-
ies (Vig et al., 2021; Beshkov and Malthe-
Sorenssen, 2026), different PLM layers capture dis-
tinct biological properties with varying relevance
to downstream tasks. We therefore condition layer
fusion on the query, adapting each group’s contri-
bution to the requested functional aspect. A frozen
text encoder Tmaps qtoeq=T(q) , from which
an MLP predicts group weights:
α(q) = softmax(MLP q(eq)).(4)
Using these weights, we fuse the group features to
obtain the final query-conditioned protein feature:
f(p|q) =GX
g=1αg(q)g g(p),
z(p|q) = L2Norm(MLP(f(p|q))).(5)
4

A high-level view.QProtFuse makes the features
of a protein conditional on the biological question.
From a broader perspective, this follows the prin-
ciple of conditional representation learning, which
adapts features to a specified task (Liu et al., 2025),
and is consistent with the query-conditioned pro-
tein retrieval formulation of ProtQueSt (Ma et al.,
2026). Our conditioning acts on the mixture of
PLM layers, where the same sequence can receive
different combinations of features from a frozen
PLM under different queries.
3.2 Joint Contrastive Objectives
A natural way to construct πθ(p, q) in Equation (2)
is to use z(p|q) from Equation (5) to rank candi-
dates according to their descriptions. This leads to
protein–text cross-modal contrastive learning, as
used by Ma et al. (2026). However, cross-modal
alignment alone can neglect within-modality rela-
tionships and degrade feature quality (Yang et al.,
2022). We therefore combine intra-modal PP and
cross-modal PA alignment objectives to encour-
age query-conditioned similarity between proteins
while aligning them with their annotations.
Protein-to-protein alignment.Given positive
and negative pairs with labels yij∈ {+1,−1} ,
the PP objective learns to distinguish functionally
related proteins under the query. We encode the
query protein and each candidate with the same
instructionq iand compute their similarity:
rPP
ij= cos(z(p i|qi),z(p j|qi))/τ.(6)
We optimize the expected SigLIP loss (Zhai et al.,
2023) over training queries si∈ T and candidates
sj∈ D:
LPP=E si∼T, s j∼D
−logσ 
yij 
rPP
ij+b
.
(7)
Here σis the sigmoid function, and both the tem-
perature τand bias bare learnable. During training,
D=T , with candidate selection specified below.
This objective increases query-conditioned similar-
ity for positive protein pairs and decreases it for
negative pairs.
Protein-to-annotation alignment.The PA ob-
jective aligns the same query-conditioned protein
features with candidate annotation features u(dj),
obtained from the same frozen text encoder Tused
for queries and a separate annotation projector. Wedefine the similarity and expected loss as
rPA
ij= cos(z(p i|qi),u(d j))/τ,
LPA=E si∼T, s j∼D
−logσ 
yij 
rPA
ij+b
.
(8)
This objective draws protein features toward match-
ing annotations and pushes them from unrelated
descriptions. We combine the two objectives into
the final objective:
L=L PP+L PA.(9)
Both objectives update the fusion network. The
ESM and text encoder remain frozen.
Positive and negative pairs.For a query sample
si= (p i, qi, di), positives share its target descrip-
tion, while eligible negatives have similar instruc-
tions but dissimilar descriptions. We adopt the
setting of ProtQueSt (Ma et al., 2026) and define
Pi={s j∈ D: cos(q i, qj)> τ query, dj=di},
Ni={s j∈ D: cos(q i, qj)> τ query,
cos(d i, dj)< τ fn}.
(10)
τquery restricts both positive and negative pairs to
samples with similar instructions, while τfnfilters
out potential false negatives with similar annota-
tions. From Ni, we select up to Hhard negatives
with the highest similarity under all-layer-mean
ESM features and upweight them during training.
Positive pairs receive yij= +1 and negative pairs
receivey ij=−1.
3.3 Inference
Joint training enables QProtFuse to retrieve ev-
idence for RAG through three complementary
modes: protein retrieval, annotation retrieval, and
protein retrieval with annotation reranking. All
three modes use the same checkpoint.
Protein retrieval.The PP branch performs
sequence-based protein retrieval using Equation (6).
The same protein can yield different rankings when
queried with respect to distinct biological perspec-
tives, such as catalytic activity or domain organi-
zation. This makes protein retrieval responsive to
the biological question, while retaining its applica-
bility to candidates with missing, incomplete, or
unreliable annotations.
Annotation retrieval.The PA branch supports
cross-modal search from a protein and its query to
functional descriptions using rPAin Equation (8).
5

Existing methods (Su et al., 2025; Zhou et al., 2025)
align text with proteins whose features are indepen-
dent of functional queries. QProtFuse, in contrast,
conditions the protein features on q, allowing an-
notation relevance to vary with the requested func-
tional aspect. This enables direct retrieval of textual
evidence tailored to the query.
Protein retrieval with annotation reranking.
For databases containing paired proteins and anno-
tations, QProtFuse enables us to retrieve a short-
list by using PP to identify proteins from query-
conditioned features, and then rerank its entries
using PA scores, which refine the order based on
how well annotations match the query protein and
question. This reranking step can improve RAG
by prioritizing relevant evidence within an initial
candidate set (Yu et al., 2024), thereby surfacing in-
formative annotations for downstream generation.
4 Experiments
In this section, we investigate the value of exploit-
ing a single PLM’s multi-layer features for retrieval-
augmented protein function prediction. Through
baseline comparisons and controlled ablations, we
address three questions: (1) Can QProtFuse im-
prove protein retrieval? (2) How do query-guided
fusion and the PP and PA objectives contribute?
(3) Do retrieval improvements translate into better
RAG-generated function descriptions?
4.1 Experimental Setup
Dataset.We use Prot-Inst-OOD (Wu et al.,
2025), an instruction-based protein understanding
benchmark covering protein function, catalytic ac-
tivity, domain/motif, and general function. It uses
an out-of-distribution split to reduce annotation
leakage between training and test sets. We train
QProtFuse on the training split. Evaluation fol-
lows official test-to-train retrieval: all 14,503 test
samples serve as queries, and the 284,526 training
records from all four tasks form a single shared
reference database. Appendix B provides task de-
scriptions, split construction, and dataset statistics.
Metrics.We report Recall@ K(R@K) and
Mean Reciprocal Rank (MRR) for retrieval. For
generated descriptions, we report the standard NLP
metrics ROUGE-L (Lin, 2004) for lexical overlap
and Entity-BLEU (Wu et al., 2025), a biological
entity-focused metric which computes the standard
BLEU score (Papineni et al., 2002) on extracted bi-Table 1: Retrieval results on Prot-Inst-OOD (%). We
average QProtFuse scores over three seeds. “∗” denotes
the two-channel “5+5” strategy of ProtQueSt, with no
instruction filtering during the structural channel. Both
variants marked “rerank” use our reranking strategy.
Method MRR@10 R@1 R@10
Protein to protein
MMseqs2 9.5 5.5 19.0
ESM2-650M all-layer avg. 14.0 9.3 24.6
RAPM 11.0 6.0 20.7
ProtQueSt struct-only 12.5 8.3 21.7
QProtFuse pp 14.7 9.7 25.9
Protein to annotation
ProTrek 12.3 7.9 22.7
ProtQueSt FiLM-only 17.5 12.4 28.2
QProtFuse pa 20.4 15.5 29.2
Hybrid
ProtQueSt∗8.2 3.026.8
ProtQueSt rerank 17.1 15.0 21.7
QProtFuse rerank 18.6 16.425.9
ological entities. We also adopt ProtQueSt’s LLM-
as-a-judge metric (Ma et al., 2026), assessing bio-
logical recall, precision, specificity, and plausibility.
Scores are computed per task and macro-averaged.
Definitions and calculations of these metrics are
provided in Appendix C.
Baselines.We compare QProtFuse with two rep-
resentative RAG-based function description meth-
ods: RAPM (Wu et al., 2025) and ProtQueSt (Ma
et al., 2026). For retrieval, we additionally include
ESM all-layer mean and the sequence-based re-
triever MMseqs2 for protein-to-protein retrieval,
and ProTrek (Su et al., 2025) for protein-to-text
retrieval. For generation, we also compare general-
purpose LLMs without RAG and protein-oriented
LLMs. All RAG-based methods use Top-10 re-
trieved references per query for generation. Gener-
ation details are provided in Appendix D.
Implementation details.QProtFuse uses Ope-
nAI’s text-embedding-3-large (3072-D) for
query and annotation embedding. We set τquery=
0.65,τfn= 0.70 , and H= 128 , and initialize the
learnable SigLIP temperature τand bias bto0.07
and−10, respectively. Further training hyperpa-
rameters are provided in Appendix E. We apply
instruction-based candidate filtering with a simi-
larity threshold of 0.65 across retrieval methods.3
We apply this filter because each record is anno-
tated only for the functional aspect specified by
3For ProtQueSt, we follow its published “5+5” setting and
omit instruction filtering during structural retrieval.
6

Table 2: Generation results on Prot-Inst-OOD.
Prot2Text-V2 (Fei et al., 2025) is trained on the training
split and serves as a supervised comparison. E-BLEU
is Entity-BLEU and RG-L is ROUGE-L (both in %).
Judge-F is the final judge score (0–10).
Method E-BLEU RG-L Judge-F
Prot2Text-V2
SFT 28.05 48.18 6.61
Qwen3-8B
LLM-only 0.62 21.66 4.26
RAPM 22.37 28.46 5.63
ProtQueSt 26.69 21.00 6.05
QProtFuse32.29 31.53 6.22
DeepSeek-V4-Pro
LLM-only 0.79 26.00 5.05
RAPM 20.66 30.93 6.26
ProtQueSt35.4321.20 6.51
QProtFuse 28.7330.95 6.68
its instruction. Otherwise, a relevant protein could
be treated as a false negative simply because its
recorded answer addresses a different question.
4.2 Results
4.2.1 Retrieval Results
Table 1 shows that QProtFuse achieves the high-
est scores on all three metrics within both PP and
PA retrieval. Compared with ESM all-layer av-
eraging, QProtFuse improves MRR from 14.0 to
14.7 and R@10 from 24.6 to 25.9, demonstrating
gains beyond uniform fusion of the PLM features.
QProtFuse also consistently outperforms RAPM
and ProtQueSt in PP retrieval, as well as ProtQueSt
in PA retrieval. These gains over methods integrat-
ing additional external signals highlight the value
of exploiting the rich inner knowledge within a
single PLM. Adding annotation reranking to PP
retrieval further improves performance for both
ProtQueSt and QProtFuse, indicating that relevant
samples are ranked higher, which can benefit down-
stream RAG (Yu et al., 2024). QProtFuse outper-
forms reranked ProtQueSt on all three metrics and
achieves the highest MRR and R@1 among the
Hybrid methods. We use this reranking configu-
ration for downstream RAG. Note that PA ranks
only 59,065 candidate annotations, whereas PP and
Hybrid retrieve from the full training set ( ∼284K).
Their scores are therefore not directly comparable.
4.2.2 RAG Generation Results
Table 2 shows that QProtFuse’s retrieval improve-
ments benefit downstream function description. On
1416182022MRR@10 (%)Fixed mean Global weights QProtFuse
PP PA Hybrid0Single Obj. QProtFuse
PP PAFigure 4: Ablation results.Left:Fusion strategies under
joint training.Fixed meanandGlobal weightsreplace
the router with uniform all-layer averaging and learned
query-independent global group weights, respectively.
Right:Joint training versus single PP-only or PA-only
training. We report the results over three seeds.
Qwen3-8B (Yang et al., 2025), QProtFuse outper-
forms RAPM and ProtQueSt on all three metrics.
Pairing QProtFuse with DeepSeek-V4-Pro (Xu
et al., 2026) yields the highest Judge-F among all
compared methods, although ProtQueSt retains the
highest Entity-BLEU. The task-level results and
judge scores in Appendix F further show that QProt-
Fuse improves judge precision over ProtQueSt
across all four tasks with both LLMs, suggesting
that its retrieved evidence supports more focused,
biologically justified descriptions. These results
highlight the value of query-guided retrieval for
generation quality beyond lexical entity overlap.
4.3 Ablation and Analysis
We conduct controlled ablations to isolate the con-
tributions of query-guided fusion and joint training.
We further analyze the learned group weights to
examine how QProtFuse combines features from
different PLM depths.
Query-guided fusion improves retrieval effec-
tiveness.We compare QProtFuse withfixed mean
andglobal weightsvariants in Figure 4 (left) to
isolate the value of query guidance. QProtFuse
substantially outperforms both in PA and Hybrid
retrieval, indicating better cross-modal alignment
between protein and text. While PP retrieval perfor-
mance remains comparable across the three strate-
gies, we speculate that the pooled PLM features
already retain sufficient information for protein-to-
protein matching, which the final MLP projection
can extract under the joint objectives. Furthermore,
learning global weights offers little improvement
over fixed mean, highlighting the importance of
query-aware layer fusion for effective retrieval.
7

1–5
6–10
11–15
16–20
21–25
26–29
30–33Joint
PP only
PA only
Global45 8 35 8
11 86
13 20 63
15 15 14 13 14 14 16Catalytic activity
1–5
6–10
11–15
16–20
21–25
26–29
30–3317 12 17 25 13 15
81 5 7
92
15 15 14 13 14 14 16Domain / motif
1–5
6–10
11–15
16–20
21–25
26–29
30–337 20 68
81 19
97
15 15 14 13 14 14 16General function
1–5
6–10
11–15
16–20
21–25
26–29
30–3323 15 8 19 31
77 19
19 22 27 30
15 15 14 13 14 14 16Protein function
020406080100Figure 5: Learned group weights (%) across four tasks under joint training, PP-only training, PA-only training, and
Global weights (Global). Columns indicate the ESM layer ranges in each group. Weights are averaged over test
queries within each task and across three seeds.
Analyze the protein with the
following sequence and describe
its properties .Analyze the following protein
sequence and predict its
subcellular localizationGiven the protein sequence below,
please predict its potential role in
a biological process1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group010203040506070Group weight (%)
1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group
Figure 6: Three examples of query-specific group weights, each with its corresponding query shown below. Bars
indicate the mean and standard deviation across three random seeds.
The PP and PA objectives reinforce each other.
As shown in Figure 4 (right), joint training im-
proves both retrieval modes over their respective
single-objective models, with a larger gain in PA.
This suggests that intra-modal contrastive learn-
ing improves protein feature quality and facili-
tates cross-modal alignment, while cross-modal su-
pervision enriches the functional information cap-
tured by these features, benefiting protein retrieval.
These findings support the complementary roles
of intra-modal and cross-modal contrastive learn-
ing (Yang et al., 2022).
QProtFuse learns adaptive fusion weights.Fig-
ure 5 shows that joint training produces distinct
mixtures of early, middle, and late layer features
across tasks. Catalytic queries combine early (1-5)
and late (26-29) layers, general-function queries fa-
vor the final group, and domain/motif queries draw
more broadly on different depths. Single-objective
training often exhibits weight collapse toward a
single group. We speculate that this limits the ex-
pressiveness of the fused features and contributes
to weaker retrieval performance. Global weights,
in contrast, remain nearly uniform.
Figure 6 further reveals distinct fusion patterns
for three queries. Subcellular localization empha-sizes early-to-middle layers, biological-process pre-
diction combines early and late features, and gen-
eral property description relies predominantly on
the final group. The contrast between localization
and broader property description is qualitatively
consistent with prior findings that lower PLM lay-
ers capture local structural information, whereas
deeper layers encode more complex structural and
functional properties (Vig et al., 2021; Rao et al.,
2021). More examples of query-specific group
weights are provided in Appendix Figure 8. To-
gether, these weights may offer insight into how
features from different PLM depths are used to
address distinct biological queries.
5 Conclusion
In this work, we investigated multi-layer PLM
features for protein retrieval and showed that the
best-performing layers vary across biological tasks,
making conventional final-layer selection and naive
all-layer averaging suboptimal. Building on these
findings, we proposed QProtFuse, which adaptively
fuses layer features according to the query and
jointly optimizes protein and annotation retrieval
through intra-modal and cross-modal contrastive
learning. On Prot-Inst-OOD, QProtFuse achieves
the strongest protein and annotation retrieval per-
8

formance and improves downstream function de-
scription. Ablations confirm the value of query
guidance and the complementary training objec-
tives, while weight analysis reveals distinct fusion
patterns for different biological queries. These find-
ings highlight adaptive use of a single PLM’s inter-
nal knowledge as an effective approach to retrieval-
augmented protein function prediction.
Limitations
QProtFuse is currently limited by its evaluation
scale and its decoupled retrieval-generation design.
Our current evaluation focuses on Prot-Inst-OOD,
which provides a controlled setting for studying
query-guided protein retrieval and its effect on func-
tion description. Scaling to larger protein resources,
such as UniProt (UniProt Consortium, 2018), could
provide more diverse biological knowledge and
support further performance gains. In addition,
our method focuses on improving retrieval while
keeping the LLM frozen. Generated responses
are not subsequently refined or verified against the
retrieved evidence. This design isolates the con-
tribution of retrieval and leaves room for further
improvements through evidence-aware generation,
response verification, and joint optimization of re-
trieval and generation.
Ethical Considerations
Potential Misuse.Query-guided feature fusion
can support protein research, but it also carries po-
tential risks of misuse. Malicious queries could di-
rect the system to identify proteins relevant to harm-
ful biological applications. Responsible deploy-
ment should therefore incorporate query screening,
appropriate access controls, and human review of
potentially sensitive requests.
AI Assistance.We used OpenAI Codex to assist
with code development and manuscript drafting.
For each manuscript section, the authors provided
an outline specifying the logical structure, from
which the AI produced an initial draft. The au-
thors then revised and polished these drafts. The
authors take responsibility for the final manuscript,
implementation, and reported results.
References
Stephen F. Altschul, Warren Gish, Webb Miller, Eu-
gene W. Myers, and David J. Lipman. 1990. Basiclocal alignment search tool.Journal of Molecular
Biology, 215(3):403–410.
Anthropic. 2026. Advancing Claude in healthcare and
the life sciences. Official blog, January 11. Accessed
September 30, 2026.
Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao,
Weihan Cao, Chiyu Chen, Haojiong Chen, Kai
Chen, Pengcheng Chen, Ying Chen, Yongkang Chen,
Yu Cheng, Pei Chu, Tao Chu, Erfei Cui, Ganqu Cui,
Long Cui, Ziyun Cui, Nianchen Deng, and 158 oth-
ers. 2025. Intern-S1: A Scientific Multimodal Foun-
dation Model.Preprint, arXiv:2508.15763. ArXiv
preprint.
Kosio Beshkov and Anders Malthe-Sorenssen. 2026.
Towards understanding the shape of representations
in protein language models. InInternational Con-
ference on Learning Representations, volume 2026,
pages 33298–33313.
Matthias Blum, Antonina Andreeva, Laise Cavalcanti
Florentino, et al. 2025. InterPro: the protein se-
quence classification resource in 2025.Nucleic Acids
Research, 53(D1):D444–D456.
Haoran Chen, Junyan Lin, Xinghao Chen, Yue Fan,
Jianfeng Dong, Xin Jin, Hui Su, Jinlan Fu, and Xi-
aoyu Shen. 2025. Multimodal language models see
better when they look shallower. InProceedings of
the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 6677–6695, Suzhou,
China. Association for Computational Linguistics.
Li Ding, Duanyu Feng, Chen Huang, Yangshuai Wang,
Yang Li, Wenqiang Lei, and See-Kiong Ng. 2026.
Unlocking Biological Workflows for Robust Protein-
Text Question Answering: A Dual-Dimensional RAG
Framework.Preprint, arXiv:2605.17261. ArXiv
preprint.
Adibvafa Fallahpour, Arman Seyed-Ahmadi, Parsa Ideh-
pour, Omar Ibrahim, Benedict MH Choi, Purav
Gupta, Jack Naimer, Kevin Zhu, Abhinav Adduri,
Arnav Shah, et al. 2026. Bioreason-pro: Advancing
protein function prediction with multimodal biologi-
cal reasoning.bioRxiv, pages 2026–03.
Yin Fang, Xiaozhuan Liang, Ningyu Zhang, Kangwei
Liu, Rui Huang, Zhuo Chen, Xiaohui Fan, and Hua-
jun Chen. 2024. Mol-instructions: A large-scale
biomolecular instruction dataset for large language
models. InInternational Conference on Learning
Representations, volume 2024, pages 48221–48251.
Xiao Fei, Michail Chatzianastasis, Sarah Carneiro, Hadi
Abdine, Lawrence Petalidis, and Michalis Vazirgian-
nis. 2025. Prot2Text-V2: Protein Function Predic-
tion with Multimodal Contrastive Alignment. InAd-
vances in Neural Information Processing Systems,
volume 38, Main Conference, pages 26205–26232.
Curran Associates, Inc.
Michael Heinzinger, Konstantin Weissenow,
Joaquin Gomez Sanchez, Adrian Henkel, Milot
9

Mirdita, Martin Steinegger, and Burkhard Rost. 2024.
Bilingual language model for protein sequence
and structure.NAR Genomics and Bioinformatics,
6(4):lqae150.
Ajit Kumar and IndraPrakash Jha. 2025. Layer Prob-
ing Improves Kinase Functional Prediction with Pro-
tein Language Models.Preprint, arXiv:2512.00376.
ArXiv preprint.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. 2023. Effi-
cient memory management for large language model
serving with pagedattention. InProceedings of the
ACM SIGOPS 29th Symposium on Operating Systems
Principles.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Xu Li, Yi Zheng, Haotian Chen, Xiaolei Chen, Yuxuan
Liang, Chenghang Lai, Bin Li, and Xiangyang Xue.
2026. Instruction-guided fusion of multi-layer visual
features in Large Vision-Language Models.Pattern
Recognition, 170:111932.
Chenchen Lin, Sanbao Su, Rachel Luo, Yuxiao Chen,
Yan Wang, Marco Pavone, and Fei Miao. 2026.
TGIF: Text-Guided Layer Fusion Mitigates Hallu-
cination in Multimodal LLMs. InConference on
Language Modeling. Conference version.
Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. InText summarization
branches out, pages 74–81.
Junyan Lin, Haoran Chen, Yue Fan, Yingqi Fan, Xin Jin,
Hui Su, Jinlan Fu, and Xiaoyu Shen. 2025. Multi-
Layer Visual Feature Fusion in Multimodal LLMs:
Methods, Analysis, and Best Practices. InProceed-
ings of the IEEE/CVF Conference on Computer Vi-
sion and Pattern Recognition (CVPR), pages 4156–
4166.
Zeming Lin, Halil Akin, Roshan Rao, Brian Hie,
Zhongkai Zhu, Wenting Lu, Nikita Smetanin, Robert
Verkuil, Ori Kabeli, Yaniv Shmueli, Allan dos
Santos Costa, Maryam Fazel-Zarandi, Tom Sercu,
Salvatore Candido, and Alexander Rives. 2023.
Evolutionary-scale prediction of atomic-level pro-
tein structure with a language model.Science,
379(6637):1123–1130.
Honglin Liu, Chao Sun, Peng Hu, Yunfan Li, and
Xi Peng. 2025. Conditional representation learn-
ing for customized tasks. InAdvances in Neural
Information Processing Systems, volume 38, Main
Conference, pages 31706–31737. Curran Associates,
Inc.Zhiyuan Liu, An Zhang, Hao Fei, Enzhi Zhang, Xiang
Wang, Kenji Kawaguchi, and Tat-Seng Chua. 2024.
Prott3: Protein-to-text generation for text-based pro-
tein understanding. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 5949–
5966.
Linrui Ma, Yiwei Liang, Yishu Yu, and Chuhan Joyce
Qi. 2026. ProtQueSt: Query-Conditioned Retrieval-
Augmented Generation for Protein Function Annota-
tion. InProceedings of the ICML 2026 3rd Workshop
on Multi-modal Foundation Models and Large Lan-
guage Models for Life Sciences.
OpenAI. 2026. Introducing GPT-Rosalind for life sci-
ences research. Official blog. Accessed September
30, 2026.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002. Bleu: a method for automatic evalu-
ation of machine translation. InProceedings of the
40th annual meeting of the Association for Computa-
tional Linguistics, pages 311–318.
Keyue Qiu, Yixin Wu, Lihao Wang, Yawen Ouyang,
Jixiang Yu, Zihan Zhou, Changze Lv, Dongyu Xue,
Yuxuan Song, Xinbo Zhang, Hao Wang, Jiangtao
Feng, Zhiqiang Gao, Lijun Wu, Xiaoqing Zheng, Ka-
Chun Wong, Lei Bai, Ya-Qin Zhang, Wei-Ying Ma,
and 3 others. 2026. AMix-2: Establishing Protein
as a Native Modality in Large Language Models.
Preprint, arXiv:2605.30963. ArXiv preprint.
Qwen Team. 2026. Qwen3.8-Max: A new bar for cod-
ing and cowork.
Roshan Rao, Joshua Meier, Tom Sercu, Sergey Ovchin-
nikov, and Alexander Rives. 2021. Transformer Pro-
tein Language Models Are Unsupervised Structure
Learners. InInternational Conference on Learning
Representations.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-
augmented black-box language models. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers), pages 8371–8384, Mexico City,
Mexico. Association for Computational Linguistics.
Martin Steinegger and Johannes Söding. 2017. Mm-
seqs2 enables sensitive protein sequence searching
for the analysis of massive data sets.Nature biotech-
nology, 35(11):1026–1028.
Jin Su, Yan He, Shiyang You, Shiyu Jiang, Xibin Zhou,
Xuting Zhang, Yuxuan Wang, Xining Su, Igor Tol-
stoy, Xing Chang, et al. 2025. A trimodal protein
language model enables advanced protein searches.
Nature Biotechnology, pages 1–7.
The UniProt Consortium. 2018. Uniprot: the univer-
sal protein knowledgebase.Nucleic acids research,
46(5):2699–2699.
10

Jesse Vig, Ali Madani, Lav R. Varshney, Caiming Xiong,
Richard Socher, and Nazneen Fatema Rajani. 2021.
BERTology Meets Biology: Interpreting Attention in
Protein Language Models. InInternational Confer-
ence on Learning Representations.
Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue,
Shujian Huang, and Quanquan Gu. 2024a. Diffusion
language models are versatile protein learners. In
International Conference on Machine Learning.
Zeyuan Wang, Qiang Zhang, Keyan Ding, Ming Qin, Xi-
ang Zhuang, Xiaotong Li, and Huajun Chen. 2024b.
InstructProtein: Aligning human and protein lan-
guage via knowledge instruction. InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 1114–1136, Bangkok, Thailand. Association
for Computational Linguistics.
Zhicong Wang, Zicheng Ma, Ziqiang Cao, Chang-
long Zhou, Jun Zhang, and Yi Qin Gao. 2025.
Prot2chat: protein large language model with early
fusion of text, sequence, and structure.Bioinformat-
ics, 41(8):btaf396.
Juntong Wu, Zijing Liu, He Cao, Li Hao, Bin Feng,
Zishan Shu, Ke Yu, Li Yuan, and Yu Li. 2025. Re-
thinking text-based protein understanding: Retrieval
or LLM? InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing,
pages 23726–23746, Suzhou, China. Association for
Computational Linguistics.
Yingce Xia, Peiran Jin, Shufang Xie, Liang He, Chuan
Cao, Renqian Luo, Guoqing Liu, Yue Wang, Zequn
Liu, Yuan-Jyue Chen, et al. 2025. Nature language
model: deciphering the language of nature for scien-
tific discovery.arXiv preprint arXiv:2502.07527.
Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang,
Bingzheng Xu, Bochao Wu, Bowei Zhang, Chao-
fan Lin, Chen Dong, Chenchen Ling, et al.
2026. Deepseek-v4: Towards highly efficient
million-token context intelligence.arXiv preprint
arXiv:2606.19348.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, et al. 2025. Qwen3
technical report.arXiv preprint arXiv:2505.09388.
Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath
Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi,
and Junzhou Huang. 2022. Vision-language pre-
training with triple contrastive learning. In2022
IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), pages 15650–15659.
IEEE.
Huanjin Yao, Wenhao Wu, Taojiannan Yang, Yuxin
Song, Mengxi Zhang, Haocheng Feng, Yifan Sun,
Zhiheng Li, Wanli Ouyang, and Jingdong Wang.
2024. Dense Connector for MLLMs. InAdvances in
Neural Information Processing Systems, volume 37,
pages 33108–33140. Curran Associates, Inc.Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You,
Chao Zhang, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Rankrag: Unifying context ranking with
retrieval-augmented generation in llms.Advances in
neural information processing systems, 37:121156–
121184.
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov,
and Lucas Beyer. 2023. Sigmoid loss for language
image pre-training. In2023 IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV), pages
11941–11952. IEEE.
Hanjing Zhou, Mingze Yin, Wei Wu, Mingyang Li,
Kun Fu, Jintai Chen, Jian Wu, and Zheng Wang.
2025. Protclip: Function-informed protein multi-
modal learning. InProceedings of the AAAI Con-
ference on Artificial Intelligence, volume 39, pages
22937–22945.
Xibin Zhou, Chenchen Han, Yingqi Zhang, Huan Du,
Jiayuan Tian, Jin Su, Renju Liu, Kai Zhuang, Shiyu
Jiang, Anthony Gitter, Li Liu, Huayu Li, Meiqi Wu,
Shiyang You, Zichen Yuan, Feng Ju, Huilin Zhang,
Wei Zheng, Fengyuan Dai, and 7 others. 2026. De-
coding the Molecular Language of Proteins with
Evolla.bioRxiv. Preprint, version 4.
Kai Zhuang, Jiawei Zhang, Yumou Liu, Hanqun Cao,
Chunbin Gu, Mengdi Liu, Zhangyang Gao, Zi-
tong Jerry Wang, Xuanhe Zhou, Pheng-Ann Heng,
Lijun Wu, Conghui He, and Cheng Tan. 2026. Lost
in Tokenization: Context as the Key to Unlocking
Biomolecular Understanding in Scientific LLMs. In
The Fourteenth International Conference on Learn-
ing Representations.
Le Zhuo, Zewen Chi, Minghao Xu, Heyan Huang,
Jianan Zhao, Heqi Zheng, Conghui He, Xian-Ling
Mao, and Wentao Zhang. 2024. ProtLLM: An inter-
leaved protein-language LLM with protein-as-word
pre-training. InProceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 8950–8963,
Bangkok, Thailand. Association for Computational
Linguistics.
11

A Single-Layer Features versus Uniform
Averaging
Table 3 compares uniform averaging with the best
single layer selected post hoc by MRR@10 for
each task. The selected layers vary across tasks
and generally outperform uniform averaging, moti-
vating adaptive fusion over a fixed average.
Table 3: ESM2-650M single-layer and all-layer mean
retrieval results on Prot-Inst-OOD (%). Single layers
are selected by task-specific MRR@10. Both use test-to-
train retrieval with instruction cosine similarity≥0.65
Task Layer MRR@10 R@1 R@10
Best single layer
Catalytic activity 17 17.46 13.39 25.97
Domain/motif 15 21.40 13.87 38.51
General function 29 5.75 3.26 11.61
Protein function 30 12.66 8.00 23.62
All-layer mean
Catalytic activity 1–33 16.89 12.78 25.87
Domain/motif 1–33 20.45 12.88 37.85
General function 1–33 5.65 3.16 11.43
Protein function 1–33 12.82 8.26 23.36
B Prot-Inst-OOD Dataset
Prot-Inst-OOD is constructed by repartition-
ing the protein understanding tasks in Mol-
Instructions (Fang et al., 2024). Its four tasks cover
protein functions, including molecular roles, bio-
logical processes, and localization; catalytic activi-
ties and reactions; domains and motifs; and general
functional descriptions. Protein sequences are clus-
tered with MMseqs2, and clusters are randomly
assigned to training and test sets at an initial ratio
of 80:20. To reduce annotation leakage, each test
protein is matched to its nearest training protein
with MMseqs2. Test samples sharing the same
label as their retrieved counterpart are moved to
the training set, and this filtering step is repeated
twice. Table 4 summarizes the sample counts and
the number of distinct instructions for each task.
Table 4: Prot-Inst-OOD statistics. The final column
reports the number of distinct instruction types.
Task Train Test Instructions
Protein function 108,696 5,487 38
Catalytic activity 51,187 1,987 5
Domain/motif 42,368 2,732 10
General function 82,275 4,297 7
Total 284,526 14,503 60C Evaluation Metrics
Retrieval metrics.A retrieved record counts as
a hit only when its bio-entity annotation exactly
matches the query protein’s ground-truth annota-
tion. We report Recall@1 and Recall@10, defined
as the fractions of queries with at least one hit
among the top 1 and top 10 results, respectively.
MRR@10 additionally measures how highly the
first hit is ranked:
MRR @10 =1
NNX
i=1(
1/ri, r i≤10,
0,otherwise,(11)
where Nis the number of evaluated queries and
riis the rank of the first hit for query i. All three
metrics are computed per task and macro-averaged
across the four tasks.
ROUGE-L.ROUGE-L (Lin, 2004) measures
overlap through the longest common subsequence
(LCS) between a prediction ˆdand reference d. We
report its F1 score:
ROUGE-L F1=2 LCS( ˆd, d)
|ˆd|+|d|,(12)
where lengths and LCS are computed over to-
kens. We use the Python package rouge-score
version 0.1.2 with RougeScorer(["rougeL"],
use_stemmer=False).
Entity-BLEU.Entity-BLEU (Wu et al., 2025)
measures biological content overlap rather than
overlap in generic response wording. Biological
entities are first extracted from predictions and ref-
erences using RAPM’s vocabulary derived from
InterPro, EC-GO, and Mol-Instructions metadata.
Standard BLEU (Papineni et al., 2002) is then com-
puted on the extracted sequences:
E-BLEU k= BP·exp 
1
kkX
n=1logp n!
,(13)
where pnis the corpus-level clipped n-gram pre-
cision of the extracted sequences, and BP =
min{1,exp(1−L ref/Lpred)}penalizes predictions
shorter than their references. Here, LpredandLref
are the total extracted sequence lengths. We use
k= 2 , combining unigram and bigram precision,
and report scores as percentages.
12

System pr ompt:
You are a protein function prediction expert with deep
knowledge of biological terminology . [You are given a
protein sequence and a ranked list of related proteins
retrieved from a database.]
User  prompt:
Instruction: {instruction}
Protein sequence: {sequence}
[Retrieved protein annotations: {retrieved_records} ]
Example input-output pairs:
{task_specific_example_1}
{task_specific_example_2}
Do not output any other text or explanation. Only output the
answer .Figure 7: Generation prompt. Blue text is included only
for RAG.
LLM-as-a-judge.Given the instruction, refer-
ence description, reference bio-entities, and pre-
diction, the judge assesses four axes:recallof
reference entities,precisionof supported or rea-
sonably inferred claims,specificityof biological
content, andplausibilitywithout contradictions or
fabrications. Each active axis receives a score in
{0,3,5,7,10} . The final per-sample score and the
macro-average are
Ji= round
1
|Ai|X
a∈A igi,a
,(14)
Jmacro =1
44X
t=11
NtX
i∈ItJi,(15)
where gi,ais the score for axis a,Aiexcludes axes
marked inactive ( −1), andItcontains the Nteval-
uated samples for task t. The final score ranges
from 0 to 10. We refer readers to ProtQueSt for the
complete prompt and rubric.
We adopt ProtQueSt’s prompt and scoring
rubric (Ma et al., 2026), using Qwen3.8-
27B (Qwen Team, 2026) in place of Claude as
the judge. We set the temperature to 0.0and the
total generation budget to 5,024 tokens, including a
4,000-token thinking budget. Following ProtQueSt,
we randomly sample 500 predictions per task, yield-
ing 2,000 evaluated samples per method.
D Generation Details
Prompt.Figure 7 shows our generation tem-
plate. All settings use the same two randomlyselected examples per task, each formatted as
“{Instruction: ... ,Protein Sequence: ... ,
Description: ... }”. Example sequences
longer than 80 amino acids are truncated to
their first 80 residues followed by ‘ ...’. The
placeholder {retrieved_records} is filled with
a list of ten retrieved samples organized as
“[{’db_label’: ..., ’protein_similarity’:
..., ’protein_text_alignment’: ...},
...] ”. For RAPM and ProtQueSt, we use their
respective published prompts.
Inference and sampling.We run Qwen3-8B lo-
cally with vLLM (Kwon et al., 2023) and access
DeepSeek-V4-Pro through its online API, using
thedeepseek-v4-pro-0813 version. Both mod-
els use a temperature of 1.0, a maximum output
length of 512 tokens, and disabled thinking.
E Training Configuration
QProtFuse.Our model has 5.13M trainable pa-
rameters. We train it for 30 epochs with three ran-
dom seeds (20260929, 20260930, and 20261001)
using mixed bfloat16 precision on four NVIDIA
H200 GPUs. The batch size is 256 per GPU, yield-
ing a global batch size of 1,024. Each training run
takes approximately 0.36 GPU hours, excluding
the precomputation of ESM features, which are
cached locally. We use AdamW with learning rates
of10−4for the router and 10−3for the remaining
modules, and a weight decay of 10−4. All reported
results use the last checkpoint.
Baselines.For Prot2Text-V2, only the projection
MLP and the LLM LoRA adapters are trained,
with LoRA rank 32, scaling factor 64, and dropout
0.1. Training runs for up to 24 epochs on 16
NVIDIA H200 GPUs using mixed bfloat16 preci-
sion. A per-GPU batch size of 4 and four gradient-
accumulation steps yield an effective batch size
of 256. We use AdamW with a learning rate of
2×10−4, weight decay of 0.01, cosine learning-
rate decay, and a random seed of 42. For ProtQueSt,
whose code was unavailable for our experiments,
we independently implement and train the model
following the configuration disclosed in its paper.
F Detailed Generation Results
Table 5 reports generation metrics for each task,
and Table 6 breaks down the four judge dimensions.
QProtFuse achieves higher judge precision than
ProtQueSt in every task with both generators.
13

Table 5: Per-task generation results on Prot-Inst-OOD. E-BLEU denotes Entity-BLEU-2 and RG-L denotes ROUGE-
L (%); Judge-F is on a 0–10 scale. Bold denotes column-wise maxima within each general-purpose LLM block.
Protein function General function Domain/motif Catalytic activity
Method E-BLEU RG-L Judge-F E-BLEU RG-L Judge-F E-BLEU RG-L Judge-F E-BLEU RG-L Judge-F
Prot2Text-V2
SFT 42.39 50.31 7.22 16.82 42.52 5.75 26.27 45.98 7.23 26.74 53.92 6.25
Qwen3-8B
LLM-only 0.04 18.29 4.00 0.02 18.30 3.57 2.40 16.36 4.48 0.00 33.68 4.98
RAPM 29.10 25.32 6.44 7.39 25.69 5.03 38.24 23.79 5.42 14.73 39.05 5.62
ProtQueSt36.2522.29 6.6318.7924.015.4319.34 8.66 5.73 32.37 29.056.38
QProtFuse 34.9927.18 6.8710.2829.335.3048.61 29.25 6.62 35.27 40.366.09
DeepSeek-V4-Pro
LLM-only 1.86 25.47 5.02 0.12 26.62 4.23 0.01 13.30 5.76 1.1738.635.20
RAPM 33.92 30.40 6.93 5.29 32.56 5.58 27.2323.106.75 16.21 37.65 5.77
ProtQueSt48.2522.11 7.0018.6624.585.7237.56 11.53 6.6537.2626.606.68
QProtFuse 36.3430.77 7.176.0533.455.6040.5721.747.6331.97 37.83 6.31
Table 6: Per-task LLM-as-a-judge scores (0–10). Rec., Prec., Spec., and Plaus. denote recall, precision, specificity,
and plausibility, respectively. Bold denotes column-wise maxima within each general-purpose LLM block.
Protein function General function Domain/motif Catalytic activity
Method Rec. Prec. Spec. Plaus. Rec. Prec. Spec. Plaus. Rec. Prec. Spec. Plaus. Rec. Prec. Spec. Plaus.
Prot2Text-V2
SFT 5.26 7.37 8.63 7.36 3.04 6.70 8.17 4.93 4.04 7.90 8.80 7.87 2.30 7.57 9.69 4.82
Qwen3-8B
LLM-only 0.51 4.69 7.45 2.91 0.04 4.95 4.31 4.60 0.56 2.70 7.43 6.71 0.00 6.999.812.60
RAPM 5.32 5.31 8.74 6.07 3.49 4.54 6.97 4.82 4.08 2.22 8.41 6.64 1.28 7.14 9.69 3.83
ProtQueSt 5.63 5.139.176.224.134.298.764.42 3.76 3.468.866.433.926.67 9.444.97
QProtFuse5.92 5.748.756.663.615.057.274.97 5.88 4.528.457.072.447.279.62 4.46
DeepSeek-V4-Pro
LLM-only 2.00 5.71 6.79 5.30 0.70 5.41 5.76 4.81 0.96 6.84 7.90 6.90 0.58 7.18 9.24 3.37
RAPM 5.67 6.23 8.91 6.59 3.72 5.69 7.245.373.96 6.59 8.65 7.28 1.80 7.379.564.02
ProtQueSt6.655.179.346.444.324.638.585.00 4.98 5.209.016.804.766.68 9.275.57
QProtFuse 6.046.538.896.833.495.967.37 5.265.58 7.598.967.862.707.679.46 4.97
14

Given the protein sequence below,
please analyze and describe the
catalytic activity of the
corresponding enzyme, specifically
the chemical reaction it catalyzes:
Given the following protein sequence,
can you perform an analysis and
identify any potential motifs or
domains ? Here is the sequence:Given the amino acid sequence
below, please deduce the
probable function  of the
protein and the biological
process  it could participate in:Please examine the protein encoded
by the amino acid sequence and
describe its functional role, potential
involvement in cellular processes ,
and its subcellular location :
Please evaluate the following
protein sequence and
determine its possible function
in a biological context :Analyze this protein sequence
and, based on conserved domains
or motifs , deduce its possible
cellular function(s) :1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group010203040506070Group weight (%)
1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group
1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group010203040506070Group weight (%)
1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer group1–5
6–10
11–15
16–20
21–25
26–29
30–33
ESM layer groupFigure 8: More examples of query-specific group weights, each with its corresponding query shown below.
15