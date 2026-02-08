# Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings

**Authors**: Doohyun Kim, Donghwa Kang, Kyungjae Lee, Hyeongboo Baek, Brent Byunghoon Kang

**Published**: 2026-02-02 07:42:18

**PDF URL**: [https://arxiv.org/pdf/2602.01757v2](https://arxiv.org/pdf/2602.01757v2)

## Abstract
The proliferation of retrieval-augmented generation (RAG) has established vector databases as critical infrastructure, yet they introduce severe privacy risks via embedding inversion attacks. Existing paradigms face a fundamental trade-off: optimization-based methods require computationally prohibitive queries, while alignment-based approaches hinge on the unrealistic assumption of accessible in-domain training data. These constraints render them ineffective in strict black-box and cross-domain settings. To dismantle these barriers, we introduce Zero2Text, a novel training-free framework based on recursive online alignment. Unlike methods relying on static datasets, Zero2Text synergizes LLM priors with a dynamic ridge regression mechanism to iteratively align generation to the target embedding on-the-fly. We further demonstrate that standard defenses, such as differential privacy, fail to effectively mitigate this adaptive threat. Extensive experiments across diverse benchmarks validate Zero2Text; notably, on MS MARCO against the OpenAI victim model, it achieves 1.8x higher ROUGE-L and 6.4x higher BLEU-2 scores compared to baselines, recovering sentences from unknown domains without a single leaked data pair.

## Full Text


<!-- PDF content starts -->

Zero2Text: Zero-Training Cross-Domain Inversion Attacks
on Textual Embeddings
Doohyun Kim* 1Donghwa Kang* 2Kyungjae Lee3Hyeongboo Baek4Brent ByungHoon Kang2
Abstract
The proliferation of retrieval-augmented genera-
tion (RAG) has established vector databases as
critical infrastructure, yet they introduce severe
privacy risks viaembedding inversion attacks. Ex-
isting paradigms face a fundamental trade-off:
optimization-based methods require computation-
ally prohibitive queries, while alignment-based
approaches hinge on the unrealistic assumption
of accessible in-domain training data. These con-
straints render them ineffective in strict black-box
and cross-domain settings. To dismantle these bar-
riers, we introduceZero2Text, a noveltraining-
freeframework based onrecursive online align-
ment. Unlike methods relying on static datasets,
Zero2Textsynergizes LLM priors with a dynamic
ridge regression mechanism to iteratively align
generation to the target embedding on-the-fly.
We further demonstrate that standard defenses,
such as differential privacy, fail to effectively mit-
igate this adaptive threat. Extensive experiments
across diverse benchmarks validateZero2Text; no-
tably, on MS MARCO against the OpenAI vic-
tim model, it achieves 1.8 ×higher ROUGE-L
and 6.4 ×higher BLEU-2 scores compared to
baselines, recovering sentences from unknown
domains without a single leaked data pair.
1. Introduction
The proliferation of large language models (LLMs) (Brown
et al., 2020; Touvron et al., 2023) has revolutionized natu-
ral language processing, yet their inherent limitations, such
1Graduate School of Information Security, Korea Advanced
Institute of Science and Technology (KAIST), Daejeon, Repub-
lic of Korea2School of Computing, Korea Advanced Institute of
Science and Technology (KAIST), Daejeon, Republic of Korea
3Department of Computer Science, University of Seoul, Seoul,
Republic of Korea4Department of Artificial Intelligence, Univer-
sity of Seoul, Seoul, Republic of Korea. Correspondence to: Brent
ByungHoon Kang <brentkang@kaist.ac.kr >, Hyeongboo Baek
<hbbeak@uos.ac.kr>.
Preprint. Wednesday 4thFebruary, 2026.
(a)Direct Inversion (b) Embedder AlignmentExtract massive text -vector pairs 1
Victim 
Embedder
Decoder66 6
6
Train a decoder on massive pairsTraining Data
Victim 
EmbedderDecoder
AdapterPretrained 
Embedder
1Train a decoder on text -vector pairs 
extracted from a general corpus
2
Align victim 
embeddings with 
few-shot pairs
Few-shot 
leaked pairs
2Figure 1.Illustration of prevailing inversion paradigms. (a) Direct
inversion methods (e.g., Vec2Text) require extracting massive text-
vector pairs ( ❶) to train a specialized decoder ( ❷). (b) Embedder
alignment methods (e.g., ALGEN) train a decoder on general
corpora ( ❶) and rely on few-shot leaked pairs from DB to align
the victim embedder ( ❷). Both paradigms struggle with domain
shifts when the target domain is unknown.
as hallucinations and a lack of up-to-date knowledge, ne-
cessitate external augmentation. Consequently, retrieval-
augmented generation (RAG) (Lewis et al., 2020) has
emerged as a de facto standard, enabling models to ac-
cess proprietary and temporal data. Central to this archi-
tecture are vector databases (DBs) (Johnson et al., 2019),
which store massive amounts of unstructured data—ranging
from personal emails to corporate trade secrets—as high-
dimensional dense embeddings. These semantic representa-
tions are widely considered efficient and, implicitly, obfus-
cated enough to preserve privacy. As a result, vector DBs
have become a critical infrastructure in modern AI stacks,
handling sensitive information under the assumption that
embeddings are non-invertible one-way functions.
However, this assumption of safety is increasingly being
challenged. Recent studies demonstrate that text embed-
dings are not cryptographically secure and retain significant
semantic information that can be reconstructed. This vulner-
ability exposes a critical attack surface:embedding inversion
attacks(Song & Raghunathan, 2020; Morris et al., 2023).
An adversary with access to DB—whether through a data
breach or public API endpoints—can recover the original
raw text, leading to severe privacy violations such as the
leakage of personally identifiable information (PII) (Carlini
et al., 2021) or confidential intellectual property.
Current embedding inversion attacks largely fall into two
1arXiv:2602.01757v2  [cs.CL]  3 Feb 2026

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
Table 1.Comparison of inversion attack paradigms.
CategoryDirect Inversion Methods
(e.g., Vec2Text, GEIA)Embedder Alignment Methods
(e.g., TEIA, ALGEN)Zero2Text
(Ours)
Decoder TrainingRequired
(embedding→text decoder)Required
(decoder trained on proxy embedder)Not required
(training-free)
Alignment TrainingNot applicableRequired
(victim↔proxy embedder alignment)Not required
(online optimization)
Data DependencyLarge-scale collections of
leaked embedding–text pairsSmall-scale leaked embedding–text pairs
+ Large-scale general corpusFew-shot access to
the victim model at test time
Domain Assumption
(Training↔Test sets)Same-domainCorpus: General domain
Alignment: Same-domainNo dependency
GeneralizationPoor (training domain bias) Limited (alignment domain bias)Robust (instance-specific)
methodological paradigms, as illustrated in Figure 1. The
first category comprisesdirect inversionmethods (Fig-
ure 1(a)), such as Vec2Text (Morris et al., 2023) and
GEIA (Li et al., 2023), which learn a direct mapping from
victim embeddings to text. While effective under idealized
conditions, these approaches rely on a prohibitive assump-
tion: the adversary must extract massive victim embed-
ding–text pairs ( ❶)—often on the order of millions (e.g., 5M
pairs)—to train a decoder ( ❷). Motivated by this limitation,
a second line of work adoptsembedder alignmentmethods
(Figure 1(b)), including ALGEN (Chen et al., 2025b) and
TEIA (Huang et al., 2024a). These approaches mitigate data
dependency by training a decoder on general corpora ( ❶)
and using a small number of leaked pairs (typically 1–8K)
to align the victim embedder ( ❷). Table 1 summarizes the
trade-offs of these paradigms against our proposed frame-
work.
Despite their differences in data scale, both paradigms share
a critical limitation: they implicitly assume that the domain
of the target embeddings matches the training or leaked
(from DB) alignment data. In realistic deployments, em-
beddings often originate from unknown, heterogeneous, or
highly specialized domains. Such distributional shifts fun-
damentally undermine existing threat models, causing sig-
nificant performance degradation in cross-domain settings.
To address these limitations, we proposeZero2Text, a
training-free embedding inversion attack based onsearch
and recovery via online verification. Unlike prior methods
that rely on large collections of leaked embedding–text pairs
or offline alignment,Zero2Textrequiresno decoder training
andno leaked alignment data. Given a target embedding at
inference time,Zero2Textleverages a pre-trained LLM as a
universal generator and iteratively issues a limited number
of online API queries. LLM-guided texts are verified by
comparing their embeddings to the target embedding, and
this similarity signal is recursively fed back to guide subse-
quent generations. Through this instance-specific, on-the-fly
process,Zero2Textaligns generation to the target embed-
ding without assuming prior knowledge of the embeddingdomain, enabling robust inversion even under unknown or
out-of-distribution settings.
Our contributions are threefold:
•We proposeZero2Text, a novel architecture that elimi-
nates reliance on auxiliary datasets. By synergizing recur-
sive LLM generation with targeted online optimization,
it achieves state-of-the-art fidelity in strict black-box set-
tings.
•We demonstrate thatZero2Textsignificantly outperforms
existing baselines (e.g., Vec2Text, ALGEN) in cross-
domain generalization, exemplified by achieving 1.8 ×
higher ROUGE-L and 6.4 ×higher BLEU-2 scores on the
MS MARCO dataset against the OpenAI model, proving
that high-precision inversion is attainable without domain-
specific pre-training.
•We comprehensively evaluate standard defenses, such as
differential privacy and noise injection, exposing their in-
sufficiency against recursive alignment attacks and under-
scoring the urgent need for stronger embedding security.
2. Related Work
2.1. Inversion Attacks on Embeddings
The vulnerability of deep learning models to inversion at-
tacks has been extensively documented. Early studies by
Fredrikson et al. (2015) demonstrated the reconstruction of
sensitive inputs from model outputs in image and medical
domains. In NLP, white-box attacks exploiting gradients
have been shown to recover input tokens (Song & Raghu-
nathan, 2020) and extract user data from general-purpose
language models (Pan et al., 2020) or federated learning
updates (Zhu et al., 2019; Gupta et al., 2022). Furthermore,
Carlini et al. (2021) revealed that LLMs memorize training
data, which can be elicited via targeted prompting. Recent
research has pivoted to black-box settings relevant to embed-
ding APIs. Li et al. (2023) empirically showed that sentence
embeddings leak significant semantic information. Morris
et al. (2023) established a strong baseline with Vec2Text,
2

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
an optimization-based method that iteratively refines inputs
to match target embeddings, while Dimitrov et al. (2022)
explored discrete optimization for similar purposes. Most re-
cently, Chen et al. (2025b) introduced ALGEN, a generation-
based framework leveraging cross-model alignment for few-
shot inversion, offering a more query-efficient alternative
to optimization-based approaches. Unlike prior works re-
lying on auxiliary datasets or static alignment,Zero2Text
introduces a zero-shot paradigm that eliminates data de-
pendencies by dynamically aligning the embedding space
online via a recursive feedback loop.
2.2. Defense Mechanisms Against Inversion Attacks
Despite the rapid advancement of inversion techniques (Li
et al., 2023; Huang et al., 2024b; Chen et al., 2025a), robust
defense strategies remain underexplored. Early adversarial
training methods (Song & Raghunathan, 2020) proved inef-
fective in black-box scenarios where the attacker’s model is
unknown. To balance privacy and utility, Morris et al. (2023)
proposed injecting Gaussian noise, though Chen et al. (2024)
showed its efficacy diminishes in multilingual settings. Dif-
ferential privacy (DP) offers a rigorous alternative; Lyu et al.
(2020) applied DP during training, while Du et al. (2023b)
introduced a metric-based local differential privacy (LDP)
mechanism for inference-time sanitization. In this work, we
critically evaluate these mechanisms against our proposed
Zero2Textframework, highlighting the persistent challenges
in securing embeddings against few-shot inversion threats.
3. Methodology
In this section, we proposeZero2Text, a novel framework
designed to recover original text from unknown target em-
beddings. As illustrated in Figure 2,Zero2Textsynergizes
the linguistic priors of LLMs with arecursive online opti-
mizationmechanism to achieve high reconstruction fidelity
without accessing any auxiliary public corpora or training
data.
3.1. Problem Formulation
We situate our work within a strictblack-boxinversion sce-
nario. The adversary aims to reconstruct a text sequence x
solely from its target embedding vector ev∈Rd, generated
by an unknown victim model ϕ. The adversary queries ϕvia
an API but lacks access to model parameters θ, gradients,
or the training distribution.
Crucially, unlike priorembedder alignment methods(Chen
et al., 2025b), we assume adata-freeenvironment where
the adversary possessesno leaked embedding-text pairsto
train a surrogate model or an adapter offline. The objective
is to find a candidate ˆxthat minimizes the distance metricDusing only the query budget:
ˆx∗= argmin
ˆx∈XD(ϕ(ˆx), e v)(1)
3.2. Design Principles
In the absence of leaked reference data, existing inversion
paradigms falter.Direct inversion methods(Morris et al.,
2023) fail due to data scarcity, whileembedder alignment
methods(Chen et al., 2025b; Huang et al., 2024a) suffer
fromdistribution shift, where a static alignment learned
on open corpora fails to generalize to the specific distribu-
tion of the target embedding. To address these challenges,
Zero2Textadopts two core principles:
•Exploiting General-Purpose Linguistic Priors:Instead
of training a domain-specific decoder that risks overfitting
to biased external data, we leverage the inherent, unbiased
probability distributions of a pre-trained LLM to generate
diverse candidates from scratch.
•Targeted Online Optimization:To eliminate the gener-
alization gap caused by mismatched training distributions,
we abandon static alignment. Instead, we proposetar-
geted online optimization, which dynamically optimizes
the mapping matrix specifically for the single target em-
bedding evduring inference, ensuring precise mapping
without prior knowledge of the target domain.
3.3. Iterative Token Generation
Following initialization,Zero2Textiteratively reconstructs
the source text token-by-token. This phase corresponds to
steps❶and❽in Figure 2.
LLM-based Generation.At iteration t, given the context
st−1from the previous step (initially [BOS] ), we lever-
age an open-source LLM (e.g., Qwen (Yang et al., 2025),
Llama (Touvron et al., 2023)) to predict the probability dis-
tribution over the vocabulary. The model generates logits yt
for the next potential token. Note that to generate valid text,
we restrict the target vocabulary to ASCII tokens.
Diversity-Aware Filtering.To prevent the candidates
from collapsing into a single mode, we enforce diversity
during selection. Instead of simply taking the top- Ktokens
by probability, we select KScandidate tokens such that
the pairwise cosine similarity between their embeddings re-
mains below a predefined threshold Thw(❶). This ensures
that the candidate set stcovers a broad range of semantic
possibilities. Finally, we employ Beam Search to retain the
top-KBmost promising sentences for the next iteration ( ❽),
repeating this process until the [EOS] token is generated
or the maximum lengthTis reached.
3

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
123456
Encode sentences  𝑠𝑡 
     to embedding vectors Ƹ𝑒𝑡
Unknown target 
embedding 𝑒𝑣Select tokens using 
            𝑆(Ƹ𝑒𝑡,𝑡−1) in Eq.(4)
×𝑊𝑡−1Send (𝐾𝐴× γt-1) corresponding sentences  to API
Update 𝑊𝑡using Eq.(3)
×𝑊𝑡Sort tokens 
     using 𝑆ǁ𝑒𝑡,𝑡 
     in Eq.(4)LLM -based Token Generation Online Optimization
1
2
3
4
5
6𝑒𝑡
1
2
3
4
5
6Ƹ𝑒𝑡
2
531
4
6
Top- (𝐾𝐴× γt-1) scoring tokens
1
4
61
4
6
2
3
53
5
1
2
4
6
3
5
1
2
4
6
Select 𝐾𝐵 sentences
     using Beam Search
Victim
Embedder Pretrained
EmbedderGenerate 𝑡-th words and
     select 𝐾𝑆 sentences  𝑠𝑡 
     with logit 𝑦𝑡
𝑠𝑡 
ǁ𝑒𝑡
Project 𝑒𝑡to ǁ𝑒𝑡1
23
4
5
67
8
LLM
Generator
Figure 2.The recursive workflow ofZero2Text: ❶Generate diverse candidate tokens via LLM; ❷Project local embeddings to the target
space;❸Group candidates by confidence; ❹Query the victim model for the top- (KA×γt−1)candidates; ❺Update the alignment
matrix online; ❻Re-project non-queried candidates with the updated matrix; ❼Re-score all candidates; and ❽Select the best sequences
via Beam Search for the next iteration.
3.4. Online Verification and Optimization
To evaluate the generated candidates without querying the
victim model for every sample (goals: accuracy & query
efficiency), we introduce an online verification and opti-
mization module. This module dynamically maps our local
embeddings to the victim’s space and scores them reliably.
This corresponds to steps❷–❼in Figure 2.
Online Projection Optimization.We compute a projec-
torWtthat projects the local embedding etto the target
vector space ˜et(❷). We formulate the update of Wtas a
Ridge Regression problem (McDonald, 2009):
min
WtX
(ei,˜ei)∈(Et,˜Et)eiWt−˜ei2+λWt2,(2)
where Etand˜Etdenote the sets of local embeddings and
ground-truth embeddings (queried from the victim) accu-
mulated up to iteration t.λis a regularization parameter. To
obtain ˜Et, we query the victim embedder with KA×γt−1
candidate sentences stcorresponding to the tokens selected
in❸. Here, γ(<1) serves as an exponential decay factor
that progressively reduces the number of queries sent to the
victim embedder as iterations proceed ( ❹). Note that in the
first iteration, due to the absence of W0, we bypass step
❸after only generating e1in❷and exceptionally trans-
mit3×K Aqueries to the victim model. The closed-form
solution is updated online as follows (❺):
Wt= 
Et⊤Et+λI−1Et⊤˜Et,(3)
where Idenotes the identity matrix. This allows us to re-
project non-queried candidates using the latest Wt(❻),
ensuring accurate estimation with minimal API calls.
Confidence-Aware Scoring (Verification).To select the
best candidates (groups ❸& sorting ❼), we propose a hybrid
scoring function S(ei, t)that serves as an online verificationstep, combining the LLM’s logit priors with the projected
embedding similarity:
S(ei, t) =Z(y i) +conf t×Z(cos(e i, ev)),(4)
where Z(·) denotes z-score normalization. Here, yrepre-
sents the LLM’s logits and serves to maintain grammati-
cal consistency by capturing complex contextual dependen-
cies. Complementarily, the cosine similarity term cos(e i, ev)
quantifies the semantic similarity between the embedding
vector eiand the target embedding ev. Crucially, we intro-
duce a dynamic confidence term conftto weight the em-
bedding similarity based on the reliability of the current
projectionWt−1:
conft=1
|Et|X
i∈Etcos(et
iWt−1,˜et
i),(5)
where Etindicates the set of token indices sent to the
API at the t-th iteration, |Et|represents the size of the in-
dex set. This term ensures that the model relies more on
the embedding similarity as the projection matrix Wbe-
comes more accurate (higher conft), minimizing the risk
of divergence in the early stages. Note that for the ini-
tial iteration ( t= 1 ), since W0is undefined, conf 1=
0.7·1
|E1|P
i∈E1cos(e1
iW1,˜e1
i).
4. Evaluation
In this section, we evaluate the efficacy ofZero2Textin re-
constructing the original text, comparing it against SOTA
textual embedding inversion attacks (e.g., Vec2Text, AL-
GEN, TEIA). All experiments are conducted on a computing
environment equipped with two NVIDIA RTX 5090 GPUs
and an Intel Xeon 6530 processor.
4.1. Experimental Setting
Models.For LLM-based token generation,Zero2Textuti-
lizes the pretrained Qwen3-0.6B model (Yang et al., 2025)
4

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
Table 2.Performance comparison betweenZero2Textand SOTA methods.
Method Victim modelMS MARCO PubMed
BLEU-1 BLEU-2 ROUGE-L ROUGE-1 COS BLEU-1 BLEU-2 ROUGE-L ROUGE-1 COS
Vec2Text
w/o CorrectorGTR-Base 12.65 2.49 9.39 11.29 0.1703 11.34 2.30 8.30 10.06 0.1706
Qwen3-Embedding-0.6B 12.53 2.64 8.40 10.52 0.5166 11.39 2.48 8.41 10.50 0.4961
OPENAI(3-small) 12.65 2.71 8.59 10.85 0.1214 10.97 2.47 8.55 10.23 0.1215
OPENAI(3-large) 12.64 2.55 9.23 11.16 0.1043 11.15 1.99 8.09 9.53 0.1119
Vec2Text
w/ CorrectorGTR-Base 12.72 2.56 9.20 11.16 0.1664 11.31 2.25 8.12 9.86 0.1768
Qwen3-Embedding-0.6B 12.82 2.60 8.00 9.94 0.4976 11.19 2.40 7.89 9.95 0.4841
OPENAI(3-small) 12.66 2.83 8.33 10.53 0.1207 11.06 2.49 8.46 10.13 0.1186
OPENAI(3-large) 12.54 2.44 8.91 10.85 0.1017 11.05 1.99 7.99 9.43 0.1109
TEIAGTR-Base 4.88 0.96 5.62 6.65 0.0182 4.97 0.97 6.36 7.62 0.0296
Qwen3-Embedding-0.6B 5.20 0.92 6.06 7.25 0.0220 4.16 0.89 5.22 5.91 0.0314
OPENAI(3-small) 4.16 0.85 5.46 6.59 0.0116 5.47 0.97 6.68 7.83 0.0274
OPENAI(3-large) 5.30 1.01 6.00 7.00 0.0180 3.73 0.86 5.25 6.17 0.0270
ALGENGTR-Base 20.28 2.35 14.16 18.24 0.5574 19.52 1.42 13.76 17.70 0.5538
Qwen3-Embedding-0.6B 22.40 2.63 15.65 21.06 0.4022 22.37 2.35 14.96 19.73 0.4188
OPENAI(3-small) 19.85 1.87 13.69 17.96 0.2524 19.87 1.87 14.19 18.33 0.2478
OPENAI(3-large) 21.83 2.38 14.79 20.44 0.2483 21.27 1.97 13.94 18.25 0.2506
Zero2TextGTR-Base 24.95 10.95 18.73 26.44 0.7514 21.34 8.38 16.81 22.00 0.7174
Qwen3-Embedding-0.6B 29.29 13.45 22.55 32.22 0.7323 25.29 10.87 19.92 27.59 0.7096
OPENAI(3-small) 34.44 17.14 25.55 37.85 0.6983 35.42 17.97 25.64 36.39 0.6859
OPENAI(3-large) 32.43 16.42 26.08 37.52 0.6634 33.76 17.17 24.61 35.31 0.6403
as the LLM generator, while employing all-mpnet-base-
v2 (Reimers & Gurevych, 2019) as the attacker’s local em-
bedder. To evaluateZero2Textacross various victim mod-
els, we utilize GTR-Base (Ni et al., 2022) and Qwen3-
Embedding-0.6B (Zhang et al., 2025) as open-source victim
embedding models, and OpenAI’s Text-Embedding-3-small
(3-small) (OpenAI, 2024b) and Text-Embedding-3-large
(3-large) (OpenAI, 2024a) as closed-source victim embed-
ding models. Detailed descriptions of each embedder are
provided in Section B of supplementary material.
Datasets.To evaluate the reconstruction capabilities of
Zero2Textin terms of cross-domain generalization, we em-
ploy two distinct datasets: MS MARCO (Nguyen et al.,
2016) for the general domain and PubMed (Cohan et al.,
2018) for the medical-specific domain. MS MARCO is a
large-scale dataset widely used in information retrieval, cov-
ering a wide range of general topics. PubMed comprises
over 36 million citations and abstracts from biomedical
literature, representing highly specialized technical texts.
Following prior work (Chen et al., 2025b), we evaluate the
reconstruction performance on 200 sampled texts from each
dataset.
Metrics.To evaluate the inversion performance, we use
five metrics: BLEU-1, BLEU-2, ROUGE-L, ROUGE-1, and
COS. BLEU-1 and BLEU-2 (Papineni et al., 2002) focus
on n-gram precision to evaluate word matching between the
original and reconstructed texts. In contrast, ROUGE-L and
ROUGE-1 (Lin, 2004) emphasize recall, specifically mea-
suring the longest common subsequence (LCS) and lexical
overlap to assess sentence-level structural similarity. Addi-
tionally, we utilize COS to quantify the semantic similarity
between the two texts. Specifically, COS calculates the co-
sine similarity between the target embedding vector and the
embedding vector obtained by feeding the reconstructed
sentence into the victim model.Considered Approaches.We evaluateZero2Textagainst
three SOTA methods: Vec2Text (Morris et al., 2023),
TEIA (Huang et al., 2024a), and ALGEN (Chen et al.,
2025b). UnlikeZero2Text, which reconstructs target text
under a strictblack-boxsetting, existing approaches ne-
cessitate offline decoder training or alignment using the
same-domain data with a target embedding vector. To en-
sure a fair comparison, we assume that existing approaches
collect training text-embedding pairs by querying the vic-
tim model with 1,000 texts from the MultiHPLT English
dataset (De Gibert et al., 2024), which is distinct from the
evaluation domain. We limit the number of queries to 1,000,
following the experimental settings of ALGEN (Chen et al.,
2025b). The specific training and reconstruction strategies
for each approach are as follows:
•Vec2Text:The decoders (Base and Corrector), both
based on T5-base (235M), are trained on the 1,000
embedding-text pairs. During the reconstruction phase,
Vec2Text employs 50 iterations and beam search with
a beam size of 8, coupled with the corrector, which
iteratively queries the victim model to refine the gen-
erated text. In our experiments, we report results for
both without Corrector and with Corrector.
•TEIA:It utilizes DialoGPT-small (117M) to train a
surrogate embedding model (specifically, an adapter)
utilizing the 1,000 embedding-text pairs.
•ALGEN:The decoder of ALGEN, based on Flan-T5-
small (80M), is trained on 150k pairs consisting of
MultiHPLT and embeddings from an open-source em-
bedder. The 1,000 text-embedding pairs obtained from
the victim model are subsequently utilized for the align-
ment phase.
Hyperparamters.For text reconstruction,Zero2Textem-
ploys beam search with a beam size of 10. We set KS=
5

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
Table 3.Analysis of the number of queried sentences and tokens.
MethodSentence Token (×103)
Offline Online Offline Online
Vec2Text 1000 3144 28.45 82.60
ALGEN 1000 0 28.45 0
TEIA 1000 0 30.15 0
Zero2Text0 2180 0 13.88
1000 ,KA= 50 ,γ= 0.8 ,Thw= 0.9 ,T= 32 andλ= 0.1 .
For the generation constraints, we apply a logit penalty of
-5 to non-alphabetic tokens at the first iteration.
4.2. Comparison to SOTA Methods
Inversion Attack Performance.Table 2 shows the text
reconstruction performance of each approach on the MS
MARCO and PubMed datasets. Vec2Text and TEIA exhibit
substantially degraded performance across all datasets and
victim models, primarily due to the severe scarcity of data
available for offline training. In particular, for Vec2Text,
the performance gains from utilizing online API queries
during reconstruction are marginal, as the Corrector is in-
sufficiently trained to leverage the feedback effectively. Al-
though ALGEN employs a decoder trained on large-scale
open datasets, it suffers from significant performance drops
due to the domain discrepancy between the alignment data
and the target. These results demonstrate the vulnerability
of existing methods in cross-domain scenarios, where the
domain of the offline prepared dataset differs from that of
the actual dataset to be reconstructed.
In contrast,Zero2Textconsistently outperforms existing
methods on all metrics regardless of the victim model or
dataset. For instance, on the MS MARCO dataset and
the OpenAI (3-large) victim model,Zero2Textachieves
a ROUGE-L score of 26.08, approximately 1.8 times
higher than ALGEN’s ROUGE-L score of 14.79. More-
over,Zero2Textachieves a BLEU-2 score of 16.42, which
is approximately 6.4 times higher than the 2.55 achieved
by Vec2Text (w/o Corrector). These results suggest that by
jointly considering syntactic and semantic information as
defined in Eq. (4),Zero2Textsuccessfully reconstructs not
only single words but also continuous word sequences.
API Query Cost.Table 3 shows the API query costs for
each method, measured by the number of sentences and to-
kens transmitted to the OPENAI(3-small) victim model dur-
ing both offline and online phases. The number of queried to-
kens was calculated based on the MultiHPLT dataset for the
offline phase and the PubMed dataset for the online phase.
For the offline training, Vec2Text, ALGEN, and TEIA re-
quire querying the victim model with 1,000 sentences to
construct their training or alignment datasets. Since each
approach employs a different internal tokenizer, the num-
ber of resulting tokens varies: Vec2Text and ALGEN send
BEAM (5, 10, 20) 𝐾𝐴 (25, 50, 100) 𝛾 (0.5, 0.8, 0.9)
ALGEN
51525
01000 2000 3000 4000 5000ROUGE -L
Number of API query10203040
01000 2000 3000 4000 5000BLEU -1
Number of API queryVec2Text BEAM (1, 2, 4, 8) Vec2Text Iteration (1, 10, 20, 50)
515253545
01000 2000 3000 4000 5000ROUGE -1
Number of API query0510152025
01000 2000 3000 4000 5000BLEU -2
Number of API queryZero2Text:Figure 3.Comparison inversion performance between Vec2Text,
ALGEN andZero2Textdepending on the number of queries.
28.45k tokens, and TEIA sends 30.15k tokens (detailed in
Section A of supplementary material). In contrast,Zero2Text
requires no offline training, incurring zero query costs (both
in sentences and tokens).
In the online reconstruction phase, ALGEN, TEIA, and
Vec2Text (without Corrector) do not perform query-based
refinement, resulting in zero online query expenditure. Con-
versely, Vec2Text (with Corrector) andZero2Textactively
query the victim model, transmitting an average of 3,144
and 2,180 sentences, respectively. Notably,Zero2Textis
significantly more token-efficient, consuming only 13.88k
tokens on average, compared to Vec2Text, which consumes
82.60k tokens. This efficiency is attributed to the fact that
Zero2Textutilizes short token sequences in early iterations
and progressively reduces the number of queried sentences
via the decay inγin later iterations.
4.3. Varying Number of Queries
To investigate the trade-off between API query costs and
reconstruction performance, we conducted an experiment
by varying the parameters that control the query bud-
get. Figure 3 visualizes the reconstruction performance of
Zero2Text, ALGEN, and Vec2Text depending on the number
of queries. The y-axis respectively represents the reconstruc-
tion performance metrics (ROUGE-L, ROUGE-1, BLEU-1,
and BLEU-2), while the x-axis denotes the total number
of queries in both offline and online phases. ForZero2Text,
we adjust the online query volume by adjusting the beam
size,KA, and γ. Similarly, for Vec2Text, the query budget
is controlled via the beam size and the number of Corrector
iterations. In the case of ALGEN, which does not utilize
online queries, we vary the offline query budget over the set
{100,500,1000,2000,3000,4000,5000}.
As shown in the Figure 3, increasing the beam size, KA, and
γinZero2Textleads to a higher query count, which in turn
6

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
152025303540
0 0.5 1 1.5
Model Size (B) Model Size (B) Model Size (B) Model Size (B)BLEU -1
BLEU -2
ROUGE -L
ROUGE -1GTR -Base Qwen3 -Embedding -0.6B OPENAI(3 -small) OPENAI(3 -large)
048121620
0 0.5 1 1.51015202530
0 0.5 1 1.515202530354045
0 0.5 1 1.5Zero2Text ALGEN Zero2Text ALGEN Zero2Text ALGEN Zero2Text ALGEN
152025303540
0 0.5 1 1.5
Model Size (B) Model Size (B) Model Size (B) Model Size (B)BLEU -1
BLEU -2
ROUGE -L
ROUGE -1
048121620
0 0.5 1 1.51015202530
0 0.5 1 1.515202530354045
0 0.5 1 1.5(a) MS MARCO
(b) PubMedModel Size (B)COS
00.20.40.60.81
0 0.5 1 1.5
Model Size (B)COS
00.20.40.60.81
0 0.5 1 1.5
Figure 4.Scalability analysis of inversion attack performance between ALGEN and Zero2Text.
Table 4.Impact ofZero2Text’s pretrained embedder choice on reconstruction performance.
Attacker embedder Victim modelMS MARCO PubMed
BLEU-1 BLEU-2 ROUGE-L ROUGE-1 COS BLEU-1 BLEU-2 ROUGE-L ROUGE-1 COS
mE5-Large-InstructGTR-Base 33.17 15.40 23.61 35.44 0.8660 29.31 12.79 20.44 28.72 0.8356
Qwen3-Embedding-0.6B 33.77 16.07 25.22 38.47 0.8187 29.63 13.72 20.55 32.39 0.8288
OPENAI(3-small) 40.36 20.94 29.30 44.18 0.7736 37.11 19.29 26.26 38.99 0.7738
OPENAI(3-large) 37.76 19.68 29.16 43.20 0.7470 38.35 19.09 24.79 38.35 0.7440
Table 5.Comparison of alignment performance on the PubMed
dataset.
Method Alignment Victim model COS
TEIA OfflineOPENAI(3-small) 0.0039
OPENAI(3-large) 0.0056
ALGEN OfflineOPENAI(3-small) 0.3229
OPENAI(3-large) 0.2848
Zero2TextOnlineOPENAI(3-small) 0.6063
OPENAI(3-large) 0.5447
monotonically improves the inversion performance. AL-
GEN exhibits a similar trend where performance improves
with increased offline queries. However, the rate of improve-
ment gradually diminishes, showing a saturation trend. In
contrast, Vec2Text shows negligible performance gains even
with a substantial increase in the number of queries.
4.4. Alignment Performance
Table 5 compares the performance of the embedder align-
ment methods (TEIA, ALGEN) andZero2Text. To evaluate
this, we measure the cosine similarity between the target
embedding vector (obtained from the victim model) and
the projected embedding vector generated by processing
the ground-truth text through each method’s aligner. TEIA
trains a simple MLP-based aligner offline. However, under
the strictblack-boxscenario, the scarcity of offline training
data prevents the aligner from effectively mapping vectors
into the victim model’s embedding space, resulting in a per-
formance drop. Similarly, ALGEN fails to achieve sufficient
alignment optimization due to its limited generalization ca-
pability across domains. In contrast,Zero2Textperforms
instance-level optimization of the aligner during the onlinephase, thereby achieving a substantial cosine similarity of
over 0.5. These results highlight the crucial role of online
alignment in achieving high reconstruction performance
within strictblack-boxscenarios.
4.5. Model Scalability
To investigate the impact of model capacity on reconstruc-
tion performance, we evaluate both ALGEN andZero2Text
across different model sizes. Figure 4 visualizes the recon-
struction results on the MS MARCO and PubMed datasets
across four victim models. The x-axis represents the model
size (in billions of parameters), and the y-axis denotes the
reconstruction performance. We compareZero2Text, em-
ploying QWEN3-0.6B and QWEN3-1.7B as the LLM Gen-
erator, against ALGEN utilizing Flan-T5-small (0.08B) and
Flan-T5-large (0.78B). As shown in Figure 4, reconstruc-
tion performance generally improves with increased model
size. However, even with the larger model, ALGEN consis-
tently yields lower reconstruction performance compared to
Zero2Text. For instance, on the MS MARCO dataset against
the OpenAI (3-large) victim model,Zero2Textachieves a
ROUGE-L score approximately 1.6 times higher than that of
ALGEN. This suggests thatZero2Textis highly parameter-
efficient, maintaining superior reconstruction performance
even when utilizing smaller models than ALGEN. Further-
more, while ALGEN incurs additional training costs to uti-
lize larger models,Zero2Texthas the advantage of employ-
ing larger pretrained models without incurring any extra
training overhead.
7

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
Table 6.Performance comparison ofZero2Textwith variaus de-
fense method.
Method Defenseϵ/dPubMed
BLEU-1 BLEU-2 ROUGE-L ROUGE-1 COS
ALGENNone - 19.87 1.87 14.19 18.33 0.2478
Random - 12.61 0.52 8.85 10.32 0.0851
LapMech0.25 13.82 0.69 9.60 12.02 0.1003
0.5 15.03 0.86 10.63 12.94 0.1409
1 15.73 0.80 11.23 13.81 0.1767
2 18.03 1.33 12.48 15.75 0.2202
4 18.62 1.61 13.69 17.50 0.2432
PurMech0.25 11.97 0.50 7.95 9.28 0.0963
0.5 14.27 0.55 10.07 12.02 0.1403
1 15.98 0.93 12.07 14.92 0.1990
2 18.07 1.34 13.21 16.23 0.2250
4 19.34 1.88 13.81 18.09 0.2470
Zero2TextNone - 35.42 17.97 25.64 36.39 0.6859
Random - 15.57 3.49 9.67 11.57 0.1463
LapMech0.25 20.36 6.99 13.75 17.98 0.1893
0.5 27.37 11.69 19.27 26.56 0.3060
1 32.45 15.97 22.93 32.35 0.4816
2 34.30 17.29 24.14 34.29 0.6094
4 34.04 17.20 25.11 35.29 0.6649
PurMech0.25 19.93 6.26 13.50 17.98 0.1894
0.5 27.73 11.13 18.84 26.64 0.3016
1 32.88 15.47 22.99 32.53 0.4828
2 33.19 16.14 23.93 33.81 0.6036
4 34.65 17.32 25.20 36.02 0.6657
4.6. Attacker Embedder
To assess the robustness ofZero2Textwith respect to the
pretrained embedder configuration, we evaluate the recon-
struction performance when replacing the default embedder
with mE5-Large-Instruct (Wang et al., 2024) during the
LLM-based token generation phase. Table 4 presents the
reconstruction results ofZero2Texton the MS MARCO and
PubMed datasets. As shown in the Table 4,Zero2Textmain-
tains high reconstruction performance across both datasets
and victim models. For instance, against the GTR-Base vic-
tim model on the MS MARCO dataset,Zero2Textachieves
a ROUGE-1 score of 35.44. SinceZero2Textallows for the
integration of the attacker embedder without any additional
training costs, these results suggest thatZero2Textoffers the
flexibility to employ a diverse range of embedders, poten-
tially further enhancing reconstruction performance.
4.7. Defense
Considerable Defense Approaches.To assess the robust-
ness ofZero2Textagainst embedding protection mecha-
nisms, we evaluate its reconstruction performance under
distinct defense strategies: Random noise and Local Differ-
ential Privacy (LDP). The random noise strategy serves as
a baseline by injecting random perturbations directly into
the target embedding. For LDP, we employ the Normal-
ized Planar Laplace (LapMech) (Dwork et al., 2014) and
Purkayastha Mechanism (PurMech) (Du et al., 2023a) to
ensure metric-LDP. In our experiments, the privacy bud-
get is configured as ϵ/d∈ {0.25,0.5,1,2,4} , normalized
by the embedding dimension d. Note that a smaller ϵ/d
corresponds to stronger noise injection.Table 7.Qualitative analysis of inversion results on the MS
MARCO dataset against the Qwen3-Embedding-0.6B victim
model.BlueandReddenote reconstructed proper nouns and
continuous word sequences matched with the original text, re-
spectively.
Original textA protocol forbuilding automation.BACnetis a
datacommunication protocolmainlyused inthe
building automationand HV AC industry (Heating
Ventilation and Air
TEIAIf you’re wondering how to subscribe to the Memo
Preview Program for high school kids to scout for
the next season, go to Memo.
Vec2Text“ section has transition history system has unique,
and it don have a special unique in common. and.
We example, we planet has itsa unique
ALGENIn addition to UL’s CIO-CNC, the Intelligent Stor-
age Unit is a modular solution for building data
structures. UL provides
Zero2TextThis is aBACnetprotocol,used in building au-
tomationsystems for communication between de-
vices. It is the most widely usedcommunication
protocolin the building industry.
Defense Results.Table 6 presents the reconstruction per-
formance of ALGEN andZero2Textwhen subjected to
these defense mechanisms. While both methods experi-
ence a degradation in performance as defenses are applied,
Zero2Textexhibits remarkable robustness. Even under de-
fense mechanisms,Zero2Textmaintains a reconstruction
performance comparable to that of ALGEN without any
defense. For instance, when PurMech is applied with a high
noise intensity ( ϵ/d= 0.25 ),Zero2Textachieves a BLEU-1
score of 20.36 and a ROUGE-L score of 13.75. These re-
sults are competitive with the performance of undefended
ALGEN, which achieves 19.87 and 14.19, respectively.
4.8. Qualitative Analysis
Table 7 presents a qualitative comparison between the origi-
nal texts and the texts reconstructed by each method. While
existing SOTA methods often construct isolated words, they
consistently fail to accurately reconstruct proper nouns or
consecutive word sequences. In contrast, as shown in Ta-
ble 7,Zero2Textsuccessfully reconstruct complex elements,
including proper nouns and continuous word sequences. Ad-
ditional qualitative examples are provided in Section D of
supplementary material.
5. Conclusion
In this paper, we addressed the critical limitation of existing
inversion methods: their poor cross-domain generalization
under strictblack-boxscenarios. To this end, we propose
Zero2Text, a novel training-free inversion attack method that
integrates LLM-based token generation with a new instance-
level online alignment strategy. By eliminating the need for
training or prior domain knowledge,Zero2Texteffectively
8

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
reconstructs original texts even in strictblack-boxscenarios.
Our experiments on the MS MARCO dataset against the
OpenAI (3-large) victim model validate the effectiveness of
our approach, achieving 1.8 ×higher ROUGE-L and 6.4 ×
higher BLEU-2 scores compared to SOTA methods. These
results expose critical vulnerabilities, necessitating robust
defenses against such generalization-capable attacks.
References
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D.,
Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., et al. Language models are few-shot learners.
volume 33, pp. 1877–1901, 2020.
Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-
V oss, A., Lee, K., Roberts, A., Brown, T., Song, D., Er-
lingsson, U., et al. Extracting training data from large
language models. In30th USENIX security symposium
(USENIX Security 21), pp. 2633–2650, 2021.
Chen, Y ., Lent, H., and Bjerva, J. Text embedding inversion
security for multilingual language models. 2024.
Chen, Y ., Biswas, R., Lent, H., and Bjerva, J. Against all
odds: Overcoming typology, script, and language confu-
sion in multilingual embedding inversion attacks. 39(22):
23632–23641, 2025a.
Chen, Y ., Xu, Q., and Bjerva, J. Algen: Few-shot inver-
sion attacks on textual embeddings via cross-model align-
ment and generation. InProceedings of the 63rd Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 24330–24348, 2025b.
Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S.,
Chang, W., and Goharian, N. A discourse-aware attention
model for abstractive summarization of long documents.
2018.
De Gibert, O., Nail, G., Arefyev, N., Ba ˜n´on, M., Van
Der Linde, J., Ji, S., Zaragoza-Bernabeu, J., Aulamo, M.,
Ram ´ırez-S ´anchez, G., Kutuzov, A., et al. A new mas-
sive multilingual dataset for high-performance language
technologies.arXiv preprint arXiv:2403.14009, 2024.
Dimitrov, D. I., Balunovi ´c, M., Jovanovi ´c, N., and Vechev,
M. Lamp: Extracting text from gradients with language
model priors. pp. arXiv–2202, 2022.
Du, M., Yue, X., Chow, S. S., and Sun, H. Sanitizing sen-
tence embeddings (and labels) for local differential pri-
vacy. InProceedings of the ACM Web Conference 2023,
pp. 2349–2359, 2023a.
Du, M., Yue, X., Chow, S. S., and Sun, H. Sanitizing sen-
tence embeddings (and labels) for local differential pri-vacy. InProceedings of the ACM Web Conference 2023,
pp. 2349–2359, 2023b.
Dwork, C., Roth, A., et al. The algorithmic foundations of
differential privacy.Foundations and trends® in theoreti-
cal computer science, 9(3–4):211–407, 2014.
Fredrikson, M., Jha, S., and Ristenpart, T. Model inver-
sion attacks that exploit confidence information and ba-
sic countermeasures. InProceedings of the 22nd ACM
SIGSAC conference on computer and communications
security, pp. 1322–1333, 2015.
Gupta, S., Huang, Y ., Zhong, Z., Gao, T., Li, K., and Chen,
D. Recovering private text in federated learning of lan-
guage models. volume 35, pp. 8130–8143, 2022.
Huang, Y .-H., Tsai, Y ., Hsiao, H., Lin, H.-Y ., and Lin, S.-
D. Transferable embedding inversion attack: Uncovering
privacy risks in text embeddings without model queries.
2024a.
Huang, Y .-H., Tsai, Y ., Hsiao, H., Lin, H.-Y ., and Lin, S.-
D. Transferable embedding inversion attack: Uncovering
privacy risks in text embeddings without model queries.
2024b.
Johnson, J., Douze, M., and J ´egou, H. Billion-scale similar-
ity search with gpus.IEEE Transactions on Big Data, 7
(3):535–547, 2019.
Langley, P. Crafting papers on machine learning. In Langley,
P. (ed.),Proceedings of the 17th International Conference
on Machine Learning (ICML 2000), pp. 1207–1216, Stan-
ford, CA, 2000. Morgan Kaufmann.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. volume 33, pp. 9459–9474, 2020.
Li, H., Xu, M., and Song, Y . Sentence embedding leaks
more information than you expect: Generative embedding
inversion attack to recover the whole sentence. 2023.
Lin, C.-Y . ROUGE: A package for automatic evaluation of
summaries. InText Summarization Branches Out, pp. 74–
81, Barcelona, Spain, July 2004. Association for Compu-
tational Linguistics. URL https://aclanthology.
org/W04-1013/.
Lyu, L., He, X., and Li, Y . Differentially private representa-
tion for nlp: Formal guarantee and an empirical study on
privacy and fairness. 2020.
McDonald, G. C. Ridge regression.Wiley Interdisciplinary
Reviews: Computational Statistics, 1(1):93–100, 2009.
9

Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
Morris, J., Kuleshov, V ., Shmatikov, V ., and Rush, A. M.
Text embeddings reveal (almost) as much as text. InPro-
ceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pp. 12448–12460, 2023.
Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary,
S., Majumder, R., and Deng, L. Ms marco: A human-
generated machine reading comprehension dataset. 2016.
Ni, J., Qu, C., Lu, J., Dai, Z., Abrego, G. H., Ma, J., Zhao,
V ., Luan, Y ., Hall, K., Chang, M.-W., et al. Large dual
encoders are generalizable retrievers. InProceedings of
the 2022 Conference on Empirical Methods in Natural
Language Processing, pp. 9844–9855, 2022.
OpenAI. Text-Embedding-3-large. Available:
https://platform.openai.com/docs/
models/text-embedding-3-large, 2024a.
OpenAI. Text-Embedding-3-small. Available:
https://platform.openai.com/docs/
models/text-embedding-3-small, 2024b.
Pan, X., Zhang, M., Ji, S., and Yang, M. Privacy risks of
general-purpose language models. In2020 IEEE Sympo-
sium on Security and Privacy (SP), pp. 1314–1331. IEEE,
2020.
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu:
a method for automatic evaluation of machine transla-
tion. InProceedings of the 40th annual meeting of the
Association for Computational Linguistics, pp. 311–318,
2002.
Reimers, N. and Gurevych, I. Sentence-bert: Sentence em-
beddings using siamese bert-networks.arXiv preprint
arXiv:1908.10084, 2019.
Song, C. and Raghunathan, A. Information leakage in
embedding models. InProceedings of the 2020 ACM
SIGSAC conference on computer and communications
security, pp. 377–390, 2020.
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi,
A., Babaei, Y ., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and fine-
tuned chat models.arXiv preprint arXiv:2307.09288,
2023.
Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R.,
and Wei, F. Multilingual e5 text embeddings: A technical
report.arXiv preprint arXiv:2402.05672, 2024.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B.,
Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025.Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang, B.,
Xie, P., Yang, A., Liu, D., Lin, J., et al. Qwen3 embed-
ding: Advancing text embedding and reranking through
foundation models.arXiv preprint arXiv:2506.05176,
2025.
Zhu, L., Liu, Z., and Han, S. Deep leakage from gradients.
volume 32, 2019.
10