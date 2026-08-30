# A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers

**Authors**: Inho Kim, Sumyeong Ahn

**Published**: 2026-08-24 11:32:05

**PDF URL**: [https://arxiv.org/pdf/2608.26194v1](https://arxiv.org/pdf/2608.26194v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems have attracted significant interest for their ability to mitigate hallucinations in Large Language Models (LLMs). Although knowledge databases for RAG are increasingly diversifying to include various modalities such as speech and text, research on handling such multi-modal database scenarios remains limited. In this paper, we propose STeReO (Speech and Text Reranking Orchestrator), a reranker based on speech and text retrievers that aggregates disparate modality databases. To address the lack of specialized training data, we first curate a dataset comprising queries, mixed-modality evidence, and their corresponding relevance ranks. We then train the reranker and evaluate its effectiveness in both single-modality and mixed-modality scenarios. Our results demonstrate that the proposed algorithm excels at selecting the most relevant evidence, thereby significantly improving downstream question-answering performance.

## Full Text


<!-- PDF content starts -->

A Reranker for Orchestrating Heterogeneous Speech and Text Retrievers
Inho Kim1, Sumyeong Ahn1,∗∗
1Korea Institute of Energy Technology, South Korea
inho20@kentech.ac.kr, sumyeongahn@kentech.ac.kr
Abstract
Retrieval-Augmented Generation (RAG) systems have attracted
significant interest for their ability to mitigate hallucinations
in Large Language Models (LLMs). Although knowledge
databases for RAG are increasingly diversifying to include
various modalities such as speech and text, research on han-
dling such multi-modal database scenarios remains limited. In
this paper, we proposeSTeReO(S peech andTe xtReranking
Orchestrator), a reranker based on speech and text retrievers that
aggregates disparate modality databases. To address the lack of
specialized training data, we first curate a dataset comprising
queries, mixed-modality evidence, and their corresponding rel-
evance ranks. We then train the reranker and evaluate its effec-
tiveness in both single-modality and mixed-modality scenarios.
Our results demonstrate that the proposed algorithm excels at
selecting the most relevant evidence, thereby significantly im-
proving downstream question-answering performance.
Index Terms: multimodal reranking, heterogeneous candidates,
speech retrieval
1. Introduction
Retrieval-Augmented Generation (RAG) [1] enhances Large
Language Models (LLMs) [2–4] by incorporating external
knowledge, thereby mitigating hallucination [5] and ground-
ing responses in external evidence. While most existing RAG
pipelines assume text-only knowledge bases (Figure 1(a)), there
is a growing need to integrate unstructured spoken content such
as lectures and meeting recordings. However, this integration
is non-trivial. The most straightforward approach is leveraging
Automatic Speech Recognition (ASR), which converts audio to
text before applying standard text retrieval. However, ASR in-
troduces transcription errors that propagate through the pipeline,
adds latency, and discards non-verbal information present in the
original audio. Consequently, extending RAG to natively sup-
port spoken modalities is a significant, yet largely unexplored,
challenge.
Several studies address ASR-related limitations through
ASR-free speech retrieval methods. For instance,VoxRAG[6]
employs direct speech-to-speech matching without ASR, while
SpeechRAG [7] aligns textual queries with speech embed-
dings. However, their retrieval spaces are still restricted to
single-modality, speech-only corpora (Figure 1(b)). In con-
trast,WavRAG[8] facilitates heterogeneous retrieval by project-
ing independent audio and text databases into a shared embed-
ding space. Despite this advancement, such joint embedding
approaches for multimodal bases suffer from the well-known
modality gap[9,10]. Even in audio-text models like CLAP [11],
**indicates the corresponding author.
Speech DB
Text DB
Speech DB
Text DB
Speech DB
Text DB
(a) Conventional (Text)-based RAG(b) (Spoken)-based RAG(c) (Text+Speech)-based RAGQuery
Query
Query
TextRetrieverAudioRetriever
LLM
Reranker
ALM
ALM
Figure 1:Comparison of three RAG systems: (a) Text-based,
(b) Speech-based, and (c) Text+Speech-based RAG.
systematic score imbalance [12] often leads one modality to
dominate retrieval results, regardless of its actual relevance.
An alternative to avoid this gap is to employ modality-
specific retrievers independently and merge their candidates via
late fusion (Figure 1(c)). In this paradigm, a robust cross-modal
reranker is essential for accurately evaluating and prioritizing
candidates retrieved from diverse modalities. While recent list-
wise reranking methods, such as those based on permutation-
invariant cross-encoders [13] or Fusion-in-Decoder architec-
tures [14], have shown promising results on text-only pools, they
remain inherently unimodal. Consequently, they fail to facilitate
the complex cross-modal comparison necessitated by heteroge-
neous candidate pools. A primary hurdle in developing such a
cross-modal reranker is the absence of training data: existing
retrieval datasets lack explicit cross-modal relevance judgments
required to directly compare audio and text candidates.
To overcome this challenge, we first construct a novel
dataset that explicitly captures the cross-modal relevance rank-
ings among candidates generated by heterogeneous retrievers.
Leveraging this dataset, we proposeSTeReO(S peech andTe xt
RerankingO rchestrator), a cross-modal reranker designed to
systematically align and prioritize heterogeneous candidates –
drawn from independent, modality-specific databases– into a
single, unified ranked list for a given text query. In summary,
our main contributions are as follows:
•First, we construct a novel cross-modal dataset that provides
explicit relevance rankings for candidates retrieved from het-
erogeneous modalities. To achieve this efficiently, we propose
a novel score fusion method that merges candidate sets from
disjoint retrievers. This approach effectively filters the vast
evidence space, allowing us to accurately extract relevance
orders from a highly targeted subset of candidates.
•Second, leveraging the constructed dataset, we train a novel
cross-modal reranker,STeReO. It is fine-tuned using Low-
Rank Adaptation (LoRA) and operates via a unified token-
based scoring mechanism. Furthermore, the rich annotations
arXiv:2608.26194v1  [cs.CL]  24 Aug 2026

(a) Dataset Construction (Phase 1)(c) Inference
Speech DB
Text DB
Query
Text Retriever
Speech Retriever
RetrieverScore𝑍-scoreNorm.
Top-𝑘Evidence 𝒞⋆
Score 𝑦!Cand. 𝑐!Query 𝑞
Training Loss ℒ(𝑦!,𝑠!)(Pointwise, Pairwise, Listwise)𝑠!=logit𝑦𝑒𝑠−logit(𝑛𝑜)𝑦!(b) Training (Phase 2)
𝑞𝑐!
Backbone ALM
Q
ALM
Reranking
Mixed modalityevidenceFigure 2:Overview of the proposed frameworkSTeReO. (a)
construction dataset for mixed modality reranker, (b) training
reranker, and (c) inference procedure based on the proposed
reranker model.
within our dataset enable the proposed algorithm to be opti-
mized across various learning objectives, including pointwise,
pairwise, and listwise reranking formulations.
•Finally, we comprehensively evaluateSTeReOon a cross-
modal benchmark constructed from the Spoken SQuAD and
MS MARCO datasets. Extensive experiments across vari-
ous backbone architectures demonstrated that our proposed
method outperforms existing single-modality baselines.
2. The proposed methodSTeReO
In this section, we present the proposed method, detailing the
construction of a dataset equipped with ranking labels across
heterogeneous modalities.
Framework.Prior to detailing the proposed method, we es-
tablish the core framework of this study. Letqdenote a text
query andDmrepresent heterogeneous databases, wherem∈
M={text,speech}. For each modalitym, a modality-specific
retrieverRmextracts a candidate setCm={cm
1, . . . , cm
k},
wherekis the number of candidates from the retriever. Subse-
quently, a rerankerRprocesses the union of these sets to yield
the mixed evidenceC⋆=R(S
m∈MCm). Finally, the Audio
Language Model (ALM) generates the answerato the query as
a=ALM(C⋆, q). The inference procedure is described in Fig-
ure 2 (c). Here, our primary focus is on training the reranker
moduleR. To achieve this, we first construct the necessary
training dataset (Phase 1), followed by the formal training ofR
(Phase 2).
2.1. Phase 1: Dataset Construction
In Phase 1, we construct a cross-modal dataset by aggregating
candidates from independent, modality-specific retrievers and
annotating them with unified relevance labels using a foundation
ALM,e.g.,GPT(Figure 2 (a)).
Domain-specific Retrieving.We first generate a can-
didate setCmfor each modality and compute initial
relevance scores. To ensure high-quality retrieval in
each domain, we employ specialized pre-trained retrievers:
e5-mistral-7b-instruct[15] for text and aHuBERT-
basedSpeechRAG[7] for speech.
Normalization and Fusion.Since independent retrievers oper-
ate on disparate score scales, direct comparison is infeasible. To
facilitate efficient candidate selection and minimize downstreamlabeling costs, we align these scores usingZ-normalization:
˜ri=ri−µm
σmform∈M,
Here,r iand˜r irepresent the raw and normalized retrieval scores
for theithcandidate, whileµ mandσ mdenote the mean and
standard deviation of scores for modalitymgiven a queryq.
We then merge the candidates based on these normalized scores
to form a unified labeling setL, from which the top-ksamples
are selected.
Cross-modal Labeling via Foundation ALM.To establish a
gold-standard ranking across different modalities, we utilize a
foundation ALM (e.g.,gpt-4o-audio-preview) as a uni-
fied evaluator. The top-kcandidates inL={c 1, . . . , c k}are
fed into the ALM, which assigns individual relevance scores
y={y 1, . . . , y k}by considering both audio and text contexts
simultaneously. Detailed prompts and implementation specifi-
cations are provided in Section 3 and the Supplementary Mate-
rial.
2.2. Phase 2: Reranker Training
In phase 2, we fine-tune the reranker using the cross-modal
candidate-relevance tuples(q, c i, yi)generated in the previous
phase. To ensure parameter efficiency, we employ Low-Rank
Adaptation (LoRA) [16]. The training process is depicted in Fig-
ure 2 (b).
Autoregressive Relevance Scoring.We adopt decoder-based
ALMs, such asUltravox[17],Qwen-Audio-Chat[18], and
Qwen2-Audio[4], as our base architecture for reranker. Follow-
ing established autoregressive reranking paradigms [19,20], the
model is trained to generate a scalar relevance scores i, based
on the logit difference between the tokensYesandNoat the
final token position:
si=logit(Yes)−logit(No).
Table 1:Training objectives for reranker optimization.σ(·)
denotes the sigmoid, andτis a hyperparameter set to10.
Method Objective Formulation
PointwiseL point=−1
|C⋆|X
i
yilogσ(s i) + (1−y i) log(1−σ(s i))
PairwiseL pair=1
|P|X
(i,j)∈P(q)log (1 + exp (−δ ij·(si−sj)))
whereP={(i, j) :c i, cj∈ C⋆, yi̸=yj}, δ ij= sign(y i−yj)
ListwiseL list= 1−1
IDCGX
i2yi−1
log2(1 + ˆr i)whereˆr i= 1 +X
j̸=iσ(τ(s j−si))
Training Objectives.We optimize the reranker using the ob-
jective functions detailed in Table 1. Our framework is de-
signed for high flexibility, supporting Pointwise (binary cross-
entropy) [21, 22], Pairwise (RankNet-style) [23, 24], and List-
wise (ApproxNDCG) [25–28] loss functions. This modularity
allows any of these widely adopted objectives to be seamlessly
integrated into our training pipeline.
Audio Windowing and Score Aggregation.To handle poten-
tial evidence localization within long audio passages, we seg-
ment each candidate intoWfixed-duration windows. During
training, a single window is randomly sampled for scoring to
serve as a stochastic regularizer, thereby enhancing model ro-
bustness. In contrast, during inference, the windows are scored
independently, with the final passage-level scores iobtained by
aggregating these window-level scores viamean(·)ormax(·).

Table 2:Reranking performance under Single- and Mixed-domain settings. All results use Max pooling for audio window size.
Single-domain Mixed-domain
Spoken SQuAD MS MARCO Spoken SQuAD MS MARCO
Backbone Obj.Hit@1 MRR NDCG@5 Hit@1 MRR NDCG@5 Hit@1 MRR NDCG@5 Hit@1 MRR NDCG@5
BaselineZ-score 0.6927 0.7498 0.7678 0.6864 0.7694 0.7987 0.5273 0.6408 0.6857 0.6621 0.7510 0.7846
UltravoxPointwise 0.7834 0.8008 0.8057 0.7432 0.8060 0.8262 0.7630 0.7892 0.7971 0.6977 0.7755 0.8032
Pairwise 0.4855 0.6288 0.6775 0.4394 0.6195 0.6865 0.3602 0.5434 0.6131 0.3336 0.5349 0.6220
Listwise 0.4430 0.6017 0.6572 0.3356 0.5487 0.6332 0.4430 0.6017 0.6572 0.1185 0.3737 0.4996
Qwen-Audio-ChatPointwise 0.7644 0.7904 0.79800.7244 0.7951 0.81800.7334 0.7728 0.78490.6907 0.7720 0.8006
Pairwise 0.7561 0.7861 0.79480.7279 0.7971 0.81960.7174 0.7635 0.77800.7076 0.7834 0.8092
Listwise 0.7361 0.7755 0.7870 0.6967 0.7792 0.8062 0.7275 0.7703 0.7831 0.5898 0.7128 0.7564
Qwen2-AudioPointwise 0.7753 0.7966 0.80260.7337 0.8004 0.82200.7584 0.7872 0.79570.6997 0.7777 0.8049
Pairwise 0.7705 0.7937 0.80050.7405 0.8047 0.82520.7467 0.7803 0.79050.7164 0.7885 0.8130
Listwise 0.7580 0.7870 0.7955 0.6872 0.7744 0.8027 0.7557 0.7858 0.7946 0.4047 0.5976 0.6701
3. Experiment
3.1. Experimental Setup
This section details the experimental setup and evaluation met-
rics used in our evaluation.
Datasets.We evaluateSTeReOusing a fixed top-kpipeline
(k= 5) on two distinct datasets: Spoken SQuAD [29] con-
sisting of text queries and TTS-generated audio passages, and
MS MARCO [30], comprising text-only web passages. The
combined retrieval pool contains approximately2.8K audio and
9.1K text passages. These corpora feature non-overlapping pas-
sages and minimal query overlap. Given that the datasets differ
in both domain and modality, this setup provides a challenging
heterogeneous environment to assess the robustness of cross-
modal reranking beyond simple modality-based discrimination.
Models.Our experiments evaluate three audio-native back-
bone architectures as student rerankers:Ultravox[17],Qwen-
Audio-Chat[18], andQwen2-Audio[4]. We compare these
three againstZ-score retrievalas the primary baseline, which
ranks candidates through the modality-wise normalization of
retriever scores as described in Section 2.
Table 3:Evaluation of the foundation ALM. Label quality is
measured on5K samples, and ranking performance is assessed
via Hit@1 on1K, respectively.
Category Metric Value Gain (∆)
Label QualityF1 Score 0.700 -
MCC 0.614 -
Ranking (Hit@1)Z-score (Baseline) 0.578 -
Foundation ALM (gpt-4o) 0.652 +0.074
Oracle (GT) 0.825 +0.247
Dataset Annotation.We usegpt-4o-audio-preview1
for dataset annotation, employing deterministic decoding and
structured output to ensure annotation consistency. To en-
sure the reliability of the generated labels, we evaluate the
gpt-4o-audio-previewmodel’s performance on a held-
out set of5,000candidates. As shown in Table 3, the foundation
ALM achieves an F1 score of 0.700 and an MCC (Matthews Cor-
relation Coefficient) of 0.614 against passage-ID ground truth,
confirming high-quality label synthesis. Moreover, we assess
the ranking signal by evaluating the foundation ALM’s direct
ranking performance across1,000queries. The annotated la-
bels achieve a Hit@1 of 0.652, outperformingZ-score retrieval
baseline by0.074.
Training.To evaluate the architectural modularity of the pro-
posed algorithm, we train the reranker under three distinct ob-
1Sincegpt-4o-audio-previewdoes not support text-only queries,
gpt-4ois used for the candidates that consist solely of text.jectives:Pointwise,Pairwise, andListwiseas denoted in Ta-
ble 1. All models are fine-tuned via LoRA with hyperparameter
r= 16,α= 32, and dropout probability0.05for3epochs us-
ing the AdamW optimizer. We set the learning rate to2×10−4
with10−2weight decay, a10%linear warmup, and a gradient
accumulation factor of2. Following the audio windowing strat-
egy, as described in Section 2, we utilize 30 second segments
(i.e.,W= 4) within a 120 second total budget.
Evaluation Scenario.To evaluate the model’s precision and
robustness, we report results in two scenarios based on domain
constraints applied after reranking. TheSingle-domainsetting
is designed to simulate a single-modality environment, ensuring
that our approach maintains high performance even with a spe-
cific domain by eliminating cross-domain noise. In contrast, the
Mixed-domainsetting evaluates the model against the full het-
erogeneous pool. This scenario assesses the model’s ability to
discriminate relevance in complex, multi-modal contexts where
candidates from diverse sources are presented simultaneously.
Evaluation Metric.We assess reranking performance us-
ing Hit@1, MRR (Mean Reciprocal Rank), and NDCG@5
(Normalized Discounted Cumulative Gain), which measure the
model’s ability to correctly identify the ground-truth passage ID
within the top-kcandidates. These metrics evaluate the accuracy
of the ranked list by checking if the target passage is successfully
retrieved at the top positions. For downstream QA tasks, we ad-
ditionally report Exact Match (EM) based on substring matching
to evaluate the fidelity of the generated answers against the ref-
erence text. BothSingle-domainandMixed-domainscenarios
utilize an identical set of held-out evaluation queries (∼13K
from SQuAD and8.7K from MS MARCO).
3.2. Results
Main results.Table 2 reports the reranking performance,
demonstrating that our approach effectively maintains high pre-
cision in a single-domain environment (Single) while exhibit-
ing robust discrimination in heterogeneous contexts (Mixed).
In the Single setting, all three backbones successfully iden-
tify target information within their native domains, consistently
outperforming theZ-score baseline when optimized with the
appropriate objective. Notably, this performance advantage ex-
tends to the Mixed setting, where the models must distinguish
relevance across a complex pool of combined speech and text
candidates. While the pointwise objective yields the most stable
results across backbones, the listwise approach shows higher
sensitivity, withUltravoxexperiencing a significant perfor-
mance drop (∆MRR of 0.257) on MS MARCO compared to its
pointwise counterpart.

Table 4:Downstream QA performance (EM) comparison.GPTevaluation is conducted on a representative subset of1K samples.
Spoken SQuAD MS MARCO
Generator Scoring Model Single Mixed Generator Scoring Model Single Mixed
UltravoxZ-score Retrieval 0.4555 0.3565UltravoxZ-score Retrieval 0.3637 0.3531
Qwen2-Audio0.5046 0.4762Qwen2-Audio0.3834 0.3729
Qwen-Audio-Chat0.4986 0.4659Qwen-Audio-Chat0.3795 0.3666
Ultravox0.5088 0.4796Ultravox0.3880 0.3719
Qwen2-AudioZ-score Retrieval 0.3335 0.2646Qwen2-AudioZ-score Retrieval 0.3885 0.3774
Qwen2-Audio0.3669 0.3468Qwen2-Audio0.40900.3981
Qwen-Audio-Chat0.3633 0.3414Qwen-Audio-Chat0.4028 0.3898
Ultravox0.3688 0.3513Ultravox0.41500.3976
Qwen-Audio-ChatZ-score Retrieval 0.3426 0.2794Qwen-Audio-ChatZ-score Retrieval 0.3637 0.3516
Qwen2-Audio0.3709 0.3499Qwen2-Audio0.37980.3742
Qwen-Audio-Chat0.3656 0.3455Qwen-Audio-Chat0.3761 0.3688
Ultravox0.3733 0.3555Ultravox0.38730.3705
GPT-4o-Audio-PreviewZ-score Retrieval 0.6273 0.4960GPT-4o-Audio-PreviewZ-score Retrieval 0.3680 0.3590
Qwen2-Audio0.6731 0.6320Qwen2-Audio0.36800.3660
Qwen-Audio-Chat0.6731 0.6230Qwen-Audio-Chat0.3630 0.3560
Ultravox0.6794 0.6350Ultravox0.38200.3650
Downstream QA Performance.As illustrated in Table 4,
we verify that the improvements in reranking precision di-
rectly translate into enhanced end-task performance for down-
stream Question Answering (QA). By feeding the top-1pas-
sage from each pointwiseSTeReOinto three ALM gener-
ators, we observe that our reranking framework consistently
boosts substring-match EM over theZ-score baseline across
all generator-dataset pairs. This enhancement is robustly main-
tained in both single-domain and mixed-domain scenarios, con-
firming that the reranker provides high-quality, relevant context
that reduces potential hallucinations in generators. Notably, the
most substantial gain is recorded in the Spoken SQuAD mixed
setting, where theUltravoxgenerator achieves a0.123EM
accuracy increase rising from0.357to0.480. These results
demonstrate the practical utility of our mixed-modality based
reranking approach in supporting accurate answer generation
within complex, heterogeneous retrieval environments.
3.3. Analysis
Table 5:The ratio of speech evidence in top-5 candidates.
Query Set w/o Z-score (%) w/ Z-score (%)
Spoken SQuAD 4.0 45.2
MS MARCO 0.1 18.3
Impact of Z-score Aggregation.As described in Table 5,
to isolate retrieval-stage fusion effects, we compare unnormal-
ized fusion, modality-wiseZ-score, and Reciprocal Rank Fu-
sion (RRF). On Spoken SQuAD, both unnormalized fusion and
RRF suffer from modality collapse, yielding near-zero Hit@1
(≈0.05) as raw text scores overwhelm speech candidates.Z-
score normalization effectively resolves this bias, significantly
recovering Hit@1 to0.527. While unnormalized fusion slightly
outperformsZ-score on MS MARCO (0.687vs.0.662) due to
the inherent dominance of text candidates, we adoptZ-score as
the default. Unlike other methods,Z-score consistently ensures
speech visibility across both domains, preventing the exclusion
of speech candidates in mixed modality retrieval.
Table 6:MRR under inference-time Mean/Max pooling.
Backbone MS MARCO Spoken SQuAD
Mean Max Mean Max
Ultravox0.79180.7755 0.74600.7892
Qwen-Audio-Chat0.78210.7720 0.72860.7728
Qwen2-Audio0.78850.7777 0.74680.7872Analysis of Pooling Strategy.Table 6 comparesMeanand
Maxpooling strategies for aggregating speech window scores
in the pointwise Mixed setting. The choice of pooling primar-
ily affects audio-heavy contexts; on MS MARCO, where the
audio candidate share is relatively low, the performance gap is
marginal, with Mean pooling holding a slight edge. However,
on Spoken SQuAD where speech candidates constitute a larger
portion of the top-5pool, Max pooling consistently outperforms
Mean across all backbone cases.
Table 7:Window-length analysis onQwen2-Audio(Mixed).
Spoken SQuAD MS MARCO
WindowHit@1 MRR NDCG@5 Hit@1 MRR NDCG@5
120s×1 0.7509 0.7822 0.7919 0.70920.78340.8091
60s×2 0.7515 0.7827 0.79220.71380.78620.8112
30s×40.7584 0.7872 0.79570.6997 0.7777 0.8049
Window-length Analysis.Table 7 compares three window con-
figurations onQwen2-Audiounder a fixed 120 seconds audio
budget and Max pooling. Given the inherent maximum sequence
length constraints of the model’s audio encoder, segmenting long
audio into multiple windows is necessary to capture full tempo-
ral information without truncation. On Spoken SQuAD, finer
segmentation (30s×4) achieves the highest Hit@1 score (0.758),
outperforming the single-window (120s×1). While the perfor-
mance gap remains marginal on MS MARCO due to the lower
audio candidate density, these results indicate that finer-grained
windows can more effectively pinpoint salient information while
operating within the model’s architectural limits.
4. Conclusion
We investigated reranking in heterogeneous pools of speech and
text, demonstrating thatSTeReOeffectively maintains high pre-
cision in a single-domain environment while ensuring robust dis-
crimination in mixed-modality scenarios. By addressing struc-
tural challenges through modality-wiseZ-score normalization
and an optimized audio windowing strategy to overcome se-
quence length constraints, we achieve consistent performance
gains. Our findings highlight that the pointwise objective is
the primary driver of stability, whereas listwise approaches can
lead to sharp degradation in heterogeneous settings. Ultimately,
these reranking improvements translate into enhanced down-
stream QA performance, providing reliable context for accurate
answer generation. While this study utilizes TTS-based data,
i.e.,Spoken SQuAD, future work will focus on validating these
insights with natural, spontaneous speech.

5. Acknowledgments
This work was supported by the Institute of Information & Com-
munications Technology Planning & Evaluation (IITP) grant
funded by the Korea government (MSIT) (RS-2025-25464461,
AI’s Vision of Harmony: A Fair and Transparent Multimodal
Agentic Platform for Conflict Mediation)
6. Generative AI Use Disclosure
Large Language Model assistance was used for language editing
and polishing of portions of this manuscript. Beyond the specific
experimental procedures explicitly described in the methodol-
ogy, such as the use of generative models for synthetic dataset
construction, no generative AI tool was used to produce exper-
imental results, figures, tables or the scientific content of this
work.
7. References
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H. K¨ uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Ad-
vances in neural information processing systems, vol. 33, pp.
9459–9474, 2020.
[2] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L.
Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat
et al., “Gpt-4 technical report,”arXiv preprint arXiv:2303.08774,
2023.
[3] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut,
J. Schalkwyk, A. M. Dai, A. Hauth, K. Millicanet al., “Gemini:
a family of highly capable multimodal models,”arXiv preprint
arXiv:2312.11805, 2023.
[4] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv,
J. He, J. Linet al., “Qwen2-audio technical report,”arXiv preprint
arXiv:2407.10759, 2024.
[5] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qinet al., “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open
questions,”ACM Transactions on Information Systems, vol. 43,
no. 2, pp. 1–55, 2025.
[6] Z. Rackauckas and J. Hirschberg, “VoxRAG: A step toward
transcription-free RAG systems in spoken question answering,”
inProceedings of the 1st Workshop on Multimodal Augmented
Generation via Multimodal Retrieval (MAGMaR 2025), R. Kriz
and K. Murray, Eds. Vienna, Austria: Association for
Computational Linguistics, Aug. 2025, pp. 40–46. [Online].
Available: https://aclanthology.org/2025.magmar-1.3/
[7] D. J. Min, K. Mundnich, A. Lapastora, E. Soltanmohammadi,
S. Ronanki, and K. Han, “Speech retrieval-augmented generation
without automatic speech recognition,” 2024. [Online]. Available:
https://arxiv.org/abs/2412.16500
[8] Y. Chen, S. Ji, H. Wang, Z. Wang, S. Chen, J. He, J. Xu,
and Z. Zhao, “Wavrag: Audio-integrated retrieval augmented
generation for spoken dialogue models,” inProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). Vienna, Austria: Association for
Computational Linguistics, 2025, pp. 12 505–12 523. [Online].
Available: https://arxiv.org/abs/2502.14727
[9] V. W. Liang, Y. Zhang, Y. Kwon, S. Yeung, and J. Y. Zou, “Mind
the gap: Understanding the modality gap in multi-modal con-
trastive representation learning,”Advances in Neural Information
Processing Systems, vol. 35, pp. 17 612–17 625, 2022.
[10] P. Feng, Z. Ma, W. Chen, Y. Li, S. Wang, K. Yu, and
X. Chen, “Enhancing speech-to-speech dialogue modeling
with end-to-end retrieval-augmented generation,”arXiv preprint
arXiv:2505.00028, 2025.[11] B. Elizalde, S. Deshmukh, M. Al Ismail, and H. Wang, “Clap
learning audio concepts from natural language supervision,” in
ICASSP 2023-2023 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.
[12] K. Saijo, J. Ebbers, F. G. Germain, S. Khurana, G. Wichern,
and J. Le Roux, “Leveraging audio-only data for text-queried tar-
get sound extraction,” inICASSP 2025-2025 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP).
IEEE, 2025, pp. 1–5.
[13] F. Schlatt, M. Fr ¨obe, H. Scells, S. Zhuang, B. Koopman,
G. Zuccon, B. Stein, M. Potthast, and M. Hagen, “Set-encoder:
Permutation-invariant inter-passage attention for listwise passage
re-ranking with cross-encoders,” inEuropean Conference on In-
formation Retrieval. Springer, 2025, pp. 1–19.
[14] S. Yoon, E. Choi, J. Kim, H. Yun, Y. Kim, and S.-w. Hwang,
“Listt5: Listwise reranking with fusion-in-decoder improves zero-
shot retrieval,” inProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long
Papers), 2024, pp. 2287–2308.
[15] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei,
“Improving text embeddings with large language models,” inPro-
ceedings of the 62nd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), 2024, pp. 11 897–
11 916.
[16] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,
and W. Chen, “LoRA: Low-rank adaptation of large language
models,” inInternational Conference on Learning Representations
(ICLR), 2022.
[17] Fixie AI, “Ultravox: A fast multimodal llm for real-time voice,”
https://github.com/fixie-ai/ultravox, 2024, accessed: 2025.
[18] Y. Chu, J. Xu, X. Zhou, Q. Yang, S. Zhang, Z. Yan, C. Zhou, and
J. Zhou, “Qwen-audio: Advancing universal audio understanding
via unified large-scale audio-language models,” 2023. [Online].
Available: https://arxiv.org/abs/2311.07919
[19] W. Sun, L. Yan, X. Ma, S. Wang, P. Ren, Z. Chen, D. Yin, and
Z. Ren, “Is chatgpt good at search? investigating large language
models as re-ranking agents,” inProceedings of the 2023 confer-
ence on empirical methods in natural language processing, 2023,
pp. 14 918–14 937.
[20] Z. Qin, R. Jagerman, K. Hui, H. Zhuang, J. Wu, L. Yan, J. Shen,
T. Liu, J. Liu, D. Metzleret al., “Large language models are ef-
fective text rankers with pairwise ranking prompting,” inFindings
of the Association for Computational Linguistics: NAACL 2024,
2024, pp. 1504–1518.
[21] T.-Y. Liu, “Learning to rank for information retrieval,”Foundations
and Trends®in Information Retrieval, vol. 3, no. 3, pp. 225–331,
2009.
[22] R. Nogueira, Z. Jiang, R. Pradeep, and J. Lin, “Document ranking
with a pretrained sequence-to-sequence model,” inFindings of the
association for computational linguistics: EMNLP 2020, 2020,
pp. 708–718.
[23] C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds,
N. Hamilton, and G. Hullender, “Learning to rank using gradient
descent,” inProceedings of the 22nd International Conference
on Machine Learning, ser. ICML ’05. New York, NY, USA:
Association for Computing Machinery, 2005, p. 89–96. [Online].
Available: https://doi.org/10.1145/1102351.1102363
[24] C. J. Burges, “From ranknet to lambdarank to lambdamart: An
overview,”Learning, vol. 11, no. 23-581, p. 81, 2010.
[25] Z. Cao, T. Qin, T.-Y. Liu, M.-F. Tsai, and H. Li, “Learning to rank:
from pairwise approach to listwise approach,” inProceedings of
the 24th international conference on Machine learning, 2007, pp.
129–136.
[26] F. Xia, T.-Y. Liu, J. Wang, W. Zhang, and H. Li, “Listwise approach
to learning to rank: theory and algorithm,” inProceedings of the
25th international conference on Machine learning, 2008, pp.
1192–1199.

[27] T. Qin, T.-Y. Liu, and H. Li, “A general approximation frame-
work for direct optimization of information retrieval measures,”
Information retrieval, vol. 13, no. 4, pp. 375–397, 2010.
[28] S. Sharifymoghaddam, R. Pradeep, A. Slavescu, R. Nguyen,
A. Xu, Z. Chen, Y. Zhang, Y. Chen, J. Xian, and J. Lin, “Ran-
kllm: A python package for reranking with llms,” inProceedings
of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval, 2025, pp. 3681–3690.
[29] C.-H. Li, S.-L. Wu, C.-L. Liu, and H.-y. Lee, “Spoken squad:
A study of mitigating the impact of speech recognition errors
on listening comprehension,”arXiv preprint arXiv:1804.00320,
2018.
[30] P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu,
R. Majumder, A. McNamara, B. Mitra, T. Nguyen, M. Rosenberg,
X. Song, A. Stoica, S. Tiwary, and T. Wang, “Ms marco: A
human generated machine reading comprehension dataset,” 2018.
[Online]. Available: https://arxiv.org/abs/1611.09268