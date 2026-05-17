# Very Efficient Listwise Multimodal Reranking for Long Documents

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh

**Published**: 2026-05-12 09:45:59

**PDF URL**: [https://arxiv.org/pdf/2605.11864v1](https://arxiv.org/pdf/2605.11864v1)

## Abstract
Listwise reranking is a key yet computationally expensive component in vision-centric retrieval and multimodal retrieval-augmented generation (M-RAG) over long documents. While recent VLM-based rerankers achieve strong accuracy, their practicality is often limited by long visual-token sequences and multi-step autoregressive decoding. We propose ZipRerank, a highly efficient listwise multimodal reranker that directly addresses both bottlenecks. It reduces input length via a lightweight query-image early interaction mechanism and eliminates autoregressive decoding by scoring all candidates in a single forward pass. To enable effective learning, ZipRerank adopts a two-stage training strategy: (i) listwise pretraining on large-scale text data rendered as images, and (ii) multimodal finetuning with VLM-teacher-distilled soft-ranking supervision. Extensive experiments on the MMDocIR benchmark show that ZipRerank matches or surpasses state-of-the-art multimodal rerankers while reducing LLM inference latency by up to an order of magnitude, making it well-suited for latency-sensitive real-world systems. The code is available at https://github.com/dukesun99/ZipRerank.

## Full Text


<!-- PDF content starts -->

Very Efficient Listwise Multimodal Reranking for Long Documents
Yiqun Sun1Pengfei Wei1Lawrence B. Hsieh1
Abstract
Listwise reranking is a key yet computationally
expensive component in vision-centric retrieval
and multimodal retrieval-augmented generation
(M-RAG) over long documents. While recent
VLM-based rerankers achieve strong accuracy,
their practicality is often limited by long visual-
token sequences and multi-step autoregressive de-
coding. We proposeZipRerank, a highly effi-
cient listwise multimodal reranker that directly
addresses both bottlenecks. It reduces input
length via a lightweight query-image early in-
teraction mechanism and eliminates autoregres-
sive decoding by scoring all candidates in a sin-
gle forward pass. To enable effective learning,
ZipRerank adopts a two-stage training strategy:
(i) listwise pretraining on large-scale text data
rendered as images, and (ii) multimodal finetun-
ing with VLM-teacher-distilled soft-ranking su-
pervision. Extensive experiments on the MMDo-
cIR benchmark show that ZipRerank matches or
surpasses state-of-the-art multimodal rerankers
while reducing LLM inference latency by up to
an order of magnitude, making it well-suited
for latency-sensitive real-world systems. The
code is available at https://github.com/
dukesun99/ZipRerank.
1. Introduction
Vision-centric multimodal retrieval over long documents has
emerged as a foundational capability in modern multimodal
systems, supporting a wide range of applications such as
Visual Question Answering (VQA) (Antol et al., 2015; Yu
et al., 2017; 2019b;a; Lerner et al., 2024; Kim et al., 2025)
and Multimodal Retrieval-Augmented Generation (M-RAG)
(Zhao et al., 2023; Mei et al., 2025; Gao et al., 2025; You
et al., 2026b;a; Dai et al., 2026). In these scenarios, docu-
1Magellan Technology Research Institute (MTRI), Japan. Cor-
respondence to: Pengfei Wei<pengfei.wei@mtri.co.jp>.
Proceedings of the 43rdInternational Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
100101
LLM Time (s)657075808590Recall@3 (%)ZipRerank (Ours)
MM-R5GPT-5-mini
LamRA-Rank
UniME-V2
Qwen3-VL-8BDSE First-Stage10× faster58× faster
Figure 1.Speed-accuracy trade-off on MMDocIR (Dong et al.,
2025) for page-level reranking (Recall@3 vs. LLM latency).
ZipRerank (red; varying token keep ratios ρ) achieves state-of-the-
art performance comparable to MM-R5 (Xu et al., 2025) while
reducing latency by around 10 ×, and substantially narrows the gap
to GPT-5-mini at about 58×lower inference cost.
ments are often visually rich and span many pages, requiring
models to jointly reason over textual queries and large col-
lections of document images (Hassan et al., 2013; Lee et al.,
2024; Dong et al., 2025).
In text-only information retrieval, effectiveness and effi-
ciency are typically balanced through a two-stage architec-
ture. A first-stage retriever performs large-scale similarity
search using dense representations (Thakur et al., 2021;
Muennighoff et al., 2023; Sun et al., 2025a;c; Feng et al.,
2025; Sun et al., 2025d), followed by a reranker that re-
fines top candidates to improve ranking quality (Sharify-
moghaddam et al., 2025; Long et al., 2025). Within this
paradigm, pointwise rerankers score each query-document
pair independently and often achieve strong accuracy at the
cost of repeated inference (Nogueira & Cho, 2019; Zhang
et al., 2025c), while listwise rerankers jointly process all
candidates, offering a more efficient alternative by avoiding
redundant computation (Sun et al., 2023).
However, extending listwise reranking to long multimodal
documents introduces substantial efficiency challenges. As
the number of candidate pages increases and each page
contributes hundreds or thousands of visual tokens, the re-
sulting input sequences quickly exceed practical limits for
Transformer-based models, leading to severe compute and
memory overhead (Yang et al., 2025; Ye et al., 2025; Shao
1arXiv:2605.11864v1  [cs.IR]  12 May 2026

Very Efficient Listwise Multimodal Reranking for Long Documents
et al., 2025). Recent multimodal listwise rerankers, such as
MM-R5 (Xu et al., 2025), demonstrate that incorporating
explicit reasoning (e.g., via Chain-of-Thought) can signifi-
cantly improve performance. Yet, these gains come at the
cost of increased latency and resource consumption, limiting
their practicality in real-world, latency-sensitive settings.
The inefficiency of multimodal listwise rerankers can be
traced to two dominant sources. First, the prefill cost be-
comes prohibitive due to long multimodal contexts, where
attention over concatenated visual tokens dominates com-
putation. Second, autoregressive decoding introduces addi-
tional latency, as rerankers often generate multi-token out-
puts, including reasoning traces and ranking sequences, in a
strictly sequential manner. Even with KV caching (Zhang
et al., 2023; Li et al., 2024; Tang et al., 2024; Zhang et al.,
2025a; Hao et al., 2025; 2026), this dependency structure
leads to significant slowdowns, especially as the number of
candidates grows.
In this work, we proposeZipRerank, a framework for train-
ing highly efficient listwise multimodal rerankers tailored
to long-document retrieval. Our approach integrates both
training and inference innovations. On the training side, we
adopt a two-stage strategy: Stage 1 leverages large-scale
text-only reranking data to learn general listwise ranking
behavior, while Stage 2 adapts the model to multimodal set-
tings using VQA-style data augmented with rankings from
a strong VLM teacher. To address the inherent noise in
such supervision, we introduce a soft-ranking objective that
assigns graded credit across candidates rather than relying
on strict pairwise or hard labels.
At inference time, ZipRerank directly targets the two iden-
tified bottlenecks through complementary efficiency mech-
anisms. First, we reduce input length using a lightweight
query-image early interaction module that performs query-
aware visual token pruning. Second, we eliminate autore-
gressive decoding by adopting a single-logit scoring strategy,
enabling the model to rank all candidates in a single forward
pass (Gangi Reddy et al., 2024; Chen et al., 2024b). These
designs substantially reduce latency while preserving the
ability to model complex cross-modal interactions.
We evaluate ZipRerank on MMDocIR (Dong et al., 2025),
a challenging multi-domain benchmark for long multi-
modal document retrieval. Across extensive experiments,
ZipRerank achieves competitive, and in several cases su-
perior, performance compared to the state-of-the-art multi-
modal reranker MM-R5 (Xu et al., 2025), while reducing
LLM inference latency by up to an order of magnitude.
Moreover, it consistently improves over strong first-stage re-
trievers. These results show that the efficiency-effectiveness
gap in VLM-based reranking can be significantly narrowed
via careful co-design of training objectives and inference
mechanisms.Our main contributions are summarized as follows:
•Latency Decomposition.We identify two fundamental
bottlenecks in listwise multimodal rerankers for long doc-
uments: long-context Transformer computation on visual
tokens and autoregressive decoding overhead, providing
a clear basis for efficient design.
•Training for Efficient Listwise Reranking.We propose
ZipRerank, a two-stage training paradigm that transfers
general ranking ability from large-scale text data to multi-
modal settings via VLM-teacher-distilled supervision and
a noise-tolerant soft-ranking objective, enabling robust
listwise learning under weak supervision.
•Single-Pass Efficient Inference.We introduce an end-
to-end efficient reranking pipeline that combines query-
aware visual token pruning with single-logit listwise scor-
ing, reducing both input length and decoding cost to
enable single forward-pass inference.
•Effectiveness-Efficiency Trade-off.Extensive experi-
ments on MMDocIR show that ZipRerank matches or
surpasses state-of-the-art rerankers while achieving up to
an order of magnitude lower cached LLM latency, with
comprehensive ablations validating each design compo-
nent.
2. Related Work
Rerankers in Information Retrieval.The retriever-
reranker paradigm is a standard framework in information
retrieval for balancing efficiency and effectiveness (Shari-
fymoghaddam et al., 2025; Nogueira & Cho, 2019). In the
first stage, dense retrievers (Reimers & Gurevych, 2019;
Wang et al., 2022; Chen et al., 2024a; Li & Li, 2024; Sun
et al., 2024; 2025b) encode queries and documents into em-
beddings for scalable similarity search (Huang et al., 2017;
2018; Li et al., 2019; Lei et al., 2019; 2020; Karpukhin et al.,
2020; Huang et al., 2021; 2023; 2024), typically optimized
for high recall. A second-stage reranker then refines the top-
kcandidates to improve ranking accuracy (Nogueira & Cho,
2019; Pradeep et al., 2023b; Sun et al., 2023; Pradeep et al.,
2023a; Sharifymoghaddam et al., 2025). While effective,
reranking often incurs significant computational overhead
and latency (Long et al., 2025; Liu et al., 2025a; Rathee
et al., 2025). In contrast, ZipRerank focuses on reducing
this second-stage cost by redesigning both training and infer-
ence to enable efficient listwise scoring without sacrificing
ranking quality.
Multimodal Information Retrieval (MMIR).With the
rise of multimodal applications such as M-RAG (Gao et al.,
2025; You et al., 2026b;a; Dai et al., 2026), multimodal
search (Zhou et al., 2023; Jiang et al., 2024), and VQA
(Antol et al., 2015; Lerner et al., 2024; Kim et al., 2025),
there has been increasing interest in retrieval over visually
rich documents. Benchmarks such as MMDocIR (Dong
2

Very Efficient Listwise Multimodal Reranking for Long Documents
et al., 2025) capture realistic challenges, including long doc-
uments, diverse domains, and queries requiring cross-modal
reasoning across multiple pages. ZipRerank is designed
specifically for this setting, targeting the scalability and ef-
ficiency challenges posed by long multimodal documents
while maintaining strong cross-modal reasoning ability.
MMIR Retrievers.Multimodal retrievers can be broadly
categorized into single-vector and multi-vector approaches.
Single-vector retrievers, such as DSE (Ma et al., 2024), en-
code each document into a single embedding for efficient
retrieval, while multi-vector (or late-interaction) methods,
such as ColQwen (Faysse et al., 2025), use multiple repre-
sentations to achieve higher accuracy at increased computa-
tional cost. Recent work, such as Light-ColPali (Ma et al.,
2025), explores token reduction techniques to improve effi-
ciency. These approaches primarily optimize the first-stage
retrieval step, whereas ZipRerank complements them by
focusing on efficient second-stage reranking, which remains
a major bottleneck in multimodal pipelines.
MMIR Rerankers.Multimodal rerankers typically adopt
either pointwise or listwise designs. Pointwise methods
(Chen et al., 2024c; Mortaheb et al., 2025; Wasserman et al.,
2025) score each query-document pair independently, often
achieving strong accuracy but incurring high latency due
to repeated forward passes. Recent multimodal rerankers
such as LamRA-Rank (Liu et al., 2025b) and UniME-V2-
Reranker (Gu et al., 2026) further extend this direction by
adapting large multimodal models for general retrieval and
reranking tasks, with support for pointwise or listwise rank-
ing formulations. Listwise rerankers (Xu et al., 2025; Liu
et al., 2025b; Gu et al., 2026) jointly process multiple candi-
dates and generate or score an ordered list, offering a more
efficient formulation than pointwise scoring. Among them,
MM-R5 (Xu et al., 2025) demonstrates strong performance
by incorporating chain-of-thought reasoning. However, its
reliance on autoregressive generation over long multimodal
inputs results in substantial inference latency. ZipRerank
differs from prior rerankers by explicitly co-designing train-
ing and inference for efficiency: it eliminates autoregressive
decoding via single-logit scoring and reduces input length
through query-aware token pruning, enabling single-pass
reranking with competitive accuracy.
3. Preliminaries
3.1. Problem Formulation
Letq∈ Q be a text query (question or instruction). A
first-stage retriever returns anorderedlist of kcandidate
long-document pages, each rendered as an image: I=
(I1,···, I k)∈ Ik, where each Iicorresponds to a full
page (e.g., a PDF page or webpage screenshot), and the
order of Iis the retriever-provided ranking. The rerankeris a parameterized multimodal model Rθ:Q × Ik→ S k,
which maps (q,I) to a permutation π∈ S kover{1,···, k} .
The output reranked list is ˆI= (I π(1),···, I π(k)).
3.2. Latency Sources
In listwise reranking, inference latency is typically domi-
nated by the reranker’s Transformer computation rather than
first-stage retrieval. Two factors drive the cost. First, the
input context can be very long because the prompt concate-
nates the query with visual tokens from multiple candidate
pages, often resulting in thousands of tokens; attention over
such long contexts becomes a major bottleneck. Second,
many rerankers output an ordered list autoregressively, re-
quiring multiple sequential decoding steps. Even with KV
caching, each step must attend over the same long context,
so generation further amplifies latency. Our inference opti-
mizations address both sources by shortening the effective
visual-token sequence and by avoiding multi-step decoding
via single-pass scoring.
For a decoder-only Transformer with context length nand
generated length u, inference cost can be approximated as:
F(n, u)≈L 
cattdn2+cffnd2n
+uLdn·c dec,(1)
where the first term is the prefill cost and the second term
is KV-cached decoding. In multimodal listwise reranking,
nis dominated by visual tokens from kpage images and u
typically grows with k(and possibly reasoning). ZipRerank
reduces nvia query-aware visual-token pruning and sets
u= 1via single-step scoring (Appendix A.1).
4. The ZipRerank Framework
We presentZipRerank, a framework for training highly
efficient listwise multimodal rerankers tailored to long-
document retrieval (Figure 2).
4.1. Training Phase
We train ZipRerank using an instruction-style formulation
(see Figure 5). Each training example contains a text query
qand a list of mcandidate document pages rendered as
images. Each candidate is assigned a unique single-token
identifier (e.g., A,B,···), and the target output is an or-
dered sequence of these identifiers in descending relevance.
Training proceeds in two stages: Stage 1 pretrains general
listwise ranking behavior using fully supervised rankings,
and Stage 2 finetunes for vision-document reranking using
teacher-distilled soft supervision.
4.1.1. STAGE1: GENERALRERANKINGPRETRAINING
In Stage 1, we leverage a large-scaletext-onlyreranking
corpus and render text passages into page-like images to
pretrain general reranking competence. Since these datasets
3

Very Efficient Listwise Multimodal Reranking for Long Documents
Query-Image 
Early Interaction
.........A)Training Phase
Query q
Candidate Pages VLM
Vision
EncoderRankNet Loss 
Language Modelling Loss 
Soft Ranking Loss [A] > [C] > [B] [A] > [B] > [C]+
+Prompt Formatting
[Query: q]
Image A: [I1 Tokens]
Image B: [I2 Tokens]
...
Rank A, B, ...A, B, C
B) Infer ence Phase
VLM Query q
Vision
EncoderQuery Hidden
States 
Full Vision
Tokens Image Token
Filtering
Top- ρ
TokensFiltered Vision
Tokens VLMB
A
C
D
G
F
E
Extract ID
Logits Prob.Reused 
KV-Cache
Candidate Pages 
Figure 2.Overview of the ZipRerank framework.ZipRerank integrates a two-stage training pipeline: (i) listwise pretraining on
large-scale text data rendered as images, followed by (ii) vision-centric finetuning with VLM-teacher-distilled soft supervision. The
efficient inference design combines query-aware visual token pruning and single-token listwise scoring, enabling end-to-end reranking in
a single LLM forward pass.
provide full ranking supervision, we optimize:
Lstage1=L LM+λ1· Lranknet,
whereλ 1balances the two terms.
Learning-to-Rank Loss.We adopt a weighted RankNet
objective (Burges et al., 2005; Gangi Reddy et al., 2024;
Chen et al., 2024b). Let tidenote the identifier token for
candidate i, and let sibe the model’s output-vocabulary
logit for tiat thefirstdecoding step. Given a target ranking
over themcandidates, we define:
Lranknet =X
ri<rjwi,j·log(1 + exp(s j−si)),
where wi,j=1
ri+rj, andridenotes the target rank of candi-
datei(smaller indicates higher relevance). This loss directly
supervises the identifier logits at the first decoding step,
which are later used for single-step scoring at inference.
Language Modeling Loss.Let y= (y 1, . . . , y |y|)be the
target identifier sequence (including any formatting tokens)
and let xbe the formatted multimodal input. We minimize
the negative log-likelihood:
LLM=−|y|X
ℓ=1logp θ(yℓ|x, y <ℓ).
4.1.2. STAGE2: VISIONRERANKINGFINETUNING
In Stage 2, we finetune on multimodal retrieval datasets (e.g.,
VQA-style data) that typically provide only a single ground-truth positive. To obtain supervision over the full candidate
list, we use a stronger VLM teacher (e.g., OpenAI’s GPT-5)
to produce an auxiliary ordering over candidates and treat it
as a soft label. We optimize
Lstage2 =L LM+λ2· Lsoftrank ,
whereλ 2balances the two terms.
Soft Ranking Loss.We use a listwise cross-entropy over
the first-step identifier distribution, with asofttarget derived
from the teacher ranking. Let pi=exp(s i)Pm
j=1exp(s j), and let
π(k) be the candidate at position kin the teacher order
(k∈ {0, . . . , m−1}). We define the target distribution:
qπ(k)=γk
Pm−1
ℓ=0γℓ, γ∈(0,1)
and compute:
Lsoftrank =−mX
i=1qilogp i.
The geometric decay is motivated by Rank-Biased Precision
(RBP) (Moffat & Zobel, 2008), which assumes a fixed con-
tinuation probability down the ranked list and therefore as-
signs exponentially decreasing importance to lower-ranked
results. Unlike the pairwise loss in Stage 1, this objective ac-
commodates uncertainty in the teacher ordering: it anchors
learning on the ground-truth item while assigning graded
credit to other high-ranked candidates, improving robustness
when multiple candidates may be plausibly relevant.
4

Very Efficient Listwise Multimodal Reranking for Long Documents
4.2. Inference Phase
At inference time, we target two dominant latency sources
in multimodal reranking over long documents: (i) the long
input sequence induced by high-resolution page images
and listwise concatenation, and (ii) the multiple forward
passes required by autoregressive generation. We address
(i) by pruning image tokens via query–image early inter-
action, and address (ii) by adapting single-token decoding
(Gangi Reddy et al., 2024; Chen et al., 2024b) to produce
the final ranking with a single LLM forward pass.
Query-Image Early Interaction.Motivated by recent
work on efficient VLMs (Zhang et al., 2025b; Han et al.,
2025; Song et al., 2025), we introduce a simple, lightweight
query–image early interaction mechanism that filters visual
tokens in a query-aware manner, retaining only the most
relevant image tokens for reranking. Let the query qbe
tokenized into Nqtext tokens. We run the LLM on the
prompt prefix up to (but excluding) the first image token, and
extract only the hidden states at the positions corresponding
to the query tokens:
Hq∈RNq×D,Hq= (h 1,···,h Nq).(2)
For each candidate page image i∈ {1,···, k} , letVi=
(vi,1,···,v i,Ni)∈RNi×Dbe its pre-computed visual to-
ken embeddings, where Niis the number of tokens for
image i, which can vary across images. We define a text-to-
image importance score for each visual token by its maxi-
mum cosine similarity to any query token:
ai,j= max
1≤t≤N qst,i,j,(3)
where st,i,j = cos(h t,vi,j) =h⊤
tvi,j
∥ht∥∥vi,j∥andj∈
{1,···, N i}. Given a keep ratio ρ∈(0,1] , we retain the
topround(ρN i)visual tokens per image:
Ji= Topround(ρN i)({ai,j}Ni
j=1),˜Vi={v i,j}j∈Ji,(4)
and construct the final multimodal input using
{˜V1,···, ˜Vk}in place of the full visual token se-
quences. We keep the original RoPE positional embedding
(Su et al., 2024) of the image patch. We reuse the KV
cache computed for the prefix when running the LLM
on the final filtered sequence. As a result, extracting the
query-token embeddings from the prefix introduces no
additional forward-pass compute.
This pruning step can be viewed as approximately preserv-
ing the attention output when the discarded tokens carry a
small total attention mass. Moreover, our max-sim pruning
score ai,jis a tight surrogate for a smooth attention-style
pooling score over query tokens (Appendix A.2). In partic-
ular, if an attention layer places tail mass εon the pruned
tokens (i.e., the sum of their attention weights), and the
corresponding value vectors are bounded, then the changein the attention output due to pruning and renormalization
is bounded byO(ε)(Appendix A.3).
Single-Token Decoding.To avoid iterative autoregres-
sive generation, we perform ranking in one decoding step.
Specifically, each candidate image iis associated with a
unique single-token identifier ti(e.g., A,B, . . . ) in the
prompt. Given the formatted multimodal input x(q, ˜V),
we run the LLM once to obtain the next-token logits
z∈R|V|and extract the logits of the identifier tokens
{zti}k
i=1. The final reranked order is the permutation
π= argsort↓ 
zt1,···, z tk
, which yields the output list
(Iπ(1),···, I π(k))in a single forward pass.
5. Experiments
5.1. Experimental Setup
5.1.1. DATASETS
Training.We finetune our models from Qwen3-VL-8B
using two datasets. Stage 1 uses RankZephyr (Pradeep
et al., 2023b), a large-scale text-passage reranking dataset
distilled from GPT-4 rankings. We render each passage
into a 280×280 image, with the font size dynamically ad-
justed to maximize text coverage within the canvas. Stage 2
finetunes on the MMDocIR training set (Dong et al., 2025).
Benchmarking.We evaluate on the page-level retrieval
task of the MMDocIR benchmark (Dong et al., 2025). The
evaluation set comprises 313 long documents spanning 10
diverse domains, with an average length of 65.1 pages, and
1,658 expert-curated queries. Following MM-R5 (Xu et al.,
2025), we retrieve the top-20 candidate pages with a first-
stage retriever and then rerank them in a second stage.
5.1.2. METRICS
Recall@ k.Following standard practice in prior work
(Dong et al., 2025; Xu et al., 2025), we use Recall@k as the
primary evaluation metric. Similar to MM-R5, our reranker
assigns a relevance score to eachpagein the document and
returns the top- kpages with the highest scores. Let Gqde-
note the set of ground-truth relevant pages for query q, and
letR(k)
qbe the set of top-kreranked pages. We compute
Recall@k(q) =|Gq∩ R(k)
q|
|Gq|,
which measures the fraction of ground-truth evidence pages
retrieved within the top- kresults. When aggregating across
datasets/subsets, we report both micro and macro Recall@k:
micro averages over all queries, while macro averages
within each dataset/subset, and then averages across subsets.
LLM Wall-Clock Time.As an auxiliary efficiency metric
in the main result tables, we report cached LLM reranking
time, excluding vision encoding and other preprocessing
5

Very Efficient Listwise Multimodal Reranking for Long Documents
Table 1.Main results for page-level retrieval and reranking with first-stage retriever DSE wiki−ss (Ma et al., 2024).
Method Res. Adm. Tut. Aca. Bro. Fin. Guide Gov. Laws News Macro Micro Time (s)
Recall@1
DSE wiki−ss 46.5 42.4 52.1 47.8 43.9 39.3 50.3 48.9 49.6 39.4 46.0 45.5 –VLMLlama-3.2-11B-Vision0.6 0.9 0.0 4.8 0.0 0.2 2.0 0.0 1.5 0.0 1.0 1.5 5.51
Qwen3-VL-8B-Instruct30.1 8.5 31.5 36.6 21.7 16.6 31.4 36.0 35.2 0.7 24.8 26.3 2.71
GPT-5-nano60.6 57.6 65.4 63.9 60.6 49.7 61.3 67.0 73.1 38.7 59.8 59.0 27.61
GPT-5-mini65.1 74.0 66.2 75.5 69.0 59.4 67.9 76.0 82.2 65.0 70.0 69.2 23.38RerankerUniME (Listwise)55.3 46.9 64.3 43.8 46.8 43.0 61.1 55.5 60.6 35.0 51.2 49.1 0.24
LamRA (Listwise)61.3 52.8 59.0 67.2 63.4 56.5 65.8 66.3 72.3 69.3 63.4 63.6 0.53
MM-R564.1 70.7 64.3 68.0 58.8 56.1 66.0 67.9 78.4 67.2 66.1 65.1 3.82
ZipRerank61.6 65.3 64.1 67.6 65.7 56.1 70.6 68.8 76.1 46.0 64.2 63.3 0.36
ZipRerank −50% 64.0 69.9 63.3 68.5 65.7 54.3 63.6 67.9 72.3 43.1 63.3 62.4 0.30
Recall@3
DSE wiki−ss 72.7 63.9 77.1 78.7 63.9 61.3 73.7 71.3 81.4 51.1 69.5 70.2 –VLMLlama-3.2-11B-Vision7.9 25.9 9.1 53.1 6.6 10.3 8.4 9.5 21.6 3.6 15.6 20.5 5.51
Qwen3-VL-8B-Instruct75.0 58.9 76.0 77.8 69.8 65.4 76.7 74.9 86.0 32.1 69.3 70.4 2.71
GPT-5-nano82.9 86.9 84.2 84.3 77.2 72.9 83.7 81.3 93.9 48.9 79.6 79.1 27.61
GPT-5-mini89.1 95.3 85.5 94.0 87.3 85.7 89.1 89.4 96.2 68.6 88.0 88.3 23.38RerankerUniME (Listwise)74.1 63.9 79.2 79.0 66.2 62.7 77.8 73.1 81.4 51.8 70.9 71.4 0.24
LamRA (Listwise)76.4 73.9 77.2 85.6 77.2 68.9 82.2 76.7 86.0 71.5 77.6 77.8 0.53
MM-R578.2 85.0 80.6 86.6 71.6 69.4 80.2 77.6 89.8 72.3 79.1 79.0 3.82
ZipRerank86.7 89.3 88.0 90.2 87.5 79.2 87.3 87.8 95.1 56.9 84.8 84.5 0.36
ZipRerank −50% 84.3 88.3 85.7 90.7 84.4 77.6 86.7 88.5 91.3 56.9 83.4 83.4 0.30
Recall@5
DSE wiki−ss 80.0 77.3 79.5 87.3 70.7 72.1 78.9 80.6 88.3 56.2 77.1 78.2 –VLMLlama-3.2-11B-Vision24.6 51.4 44.6 76.7 40.6 38.3 33.7 23.9 40.5 8.0 38.2 43.0 5.51
Qwen3-VL-8B-Instruct82.2 71.9 81.1 85.7 77.6 73.8 84.9 81.5 89.8 39.4 76.8 77.9 2.71
GPT-5-nano87.0 91.0 87.8 91.5 81.2 79.5 89.1 86.0 94.7 57.7 84.5 84.7 27.61
GPT-5-mini92.3 97.7 89.7 97.4 89.3 88.3 93.4 93.2 97.0 70.8 90.9 91.3 23.38RerankerUniME (Listwise)80.4 77.3 81.4 87.3 72.4 72.4 80.8 82.4 88.3 56.9 78.0 78.8 0.24
LamRA (Listwise)82.0 85.0 80.0 91.9 80.0 76.6 84.1 83.3 91.3 73.0 82.7 83.3 0.53
MM-R584.8 92.5 82.9 91.4 75.1 76.4 82.8 86.9 91.3 73.7 83.8 83.9 3.82
ZipRerank91.7 93.8 90.8 96.1 89.4 86.1 92.8 92.3 96.2 61.3 89.0 89.4 0.36
ZipRerank −50% 90.9 92.4 91.2 95.6 89.7 84.9 91.5 91.4 93.2 61.3 88.2 88.6 0.30
costs. This metric isolates the cost of the LLM reranking
step once visual embeddings are available, and is useful for
comparing the decoding and scoring efficiency of different
rerankers. For API-based models, we report API wall-clock
time. To complement this cached metric, we further pro-
vide an end-to-end efficiency analysis in Appendix C.5,
including vision encoding, query-aware filtering, LLM time,
throughput, FLOPs, and peak GPU memory.
5.1.3. MODELS
We consider two first-stage retrievers: DSE wiki−ss (Ma
et al., 2024), a single-vector retriever, and ColQwen (Faysse
et al., 2025), a multi-vector late-interaction retriever.For VLM-based listwise reranking baselines, we
compare against Llama-3.2-11B-Vision1,
Qwen3-VL-8B-Instruct (Bai et al., 2025),
GPT-5-nano , and GPT-5-mini via the official
API. We also include recent listwise multimodal rerankers,
including MM-R5 (Xu et al., 2025), LamRA (Liu et al.,
2025b), and UniME (Gu et al., 2026).
We finetune ZipRerank from the
Qwen3-VL-8B-Instruct checkpoint. Additional
details, including hyperparameters, training setup, and
checkpoints, are provided in Appendix B.
1https://huggingface.co/meta-llama/
Llama-3.2-11B-Vision
6

Very Efficient Listwise Multimodal Reranking for Long Documents
Table 2.Main results for page-level retrieval and reranking with first-stage retriever ColQwen (Faysse et al., 2025).
Method Res. Adm. Tut. Aca. Bro. Fin. Guide Gov. Laws News Macro Micro Time (s)
Recall@1
ColQwen56.2 55.3 60.8 65.4 53.8 48.1 60.4 66.1 72.3 65.7 60.4 59.9 –VLMLlama-3.2-11B-Vision0.0 0.9 0.0 3.0 0.0 0.7 1.3 0.9 0.8 1.5 0.9 1.2 5.51
Qwen3-VL-8B-Instruct35.2 16.4 41.6 35.9 28.7 18.9 37.2 43.5 38.6 1.5 29.8 29.6 2.47
GPT-5-nano62.4 60.6 64.1 66.3 60.7 51.1 67.1 70.6 79.9 53.3 63.6 62.5 27.32
GPT-5-mini64.8 66.6 67.2 74.3 71.3 60.9 70.9 76.0 85.2 71.5 70.9 70.1 24.70RerankerUniME (Listwise)58.0 56.7 63.8 56.4 51.2 47.8 60.9 61.6 75.4 61.3 59.3 57.6 0.25
LamRA (Listwise)60.7 56.8 61.7 71.0 61.1 57.1 69.0 70.6 76.1 70.1 65.4 65.6 0.55
MM-R565.3 68.3 64.9 69.1 64.7 58.1 66.0 70.6 84.5 73.7 68.5 67.4 3.74
ZipRerank66.3 67.1 65.7 70.1 70.3 56.2 71.5 72.4 79.2 42.3 66.1 65.1 0.36
ZipRerank −50% 62.4 64.4 65.2 69.1 68.3 54.0 65.8 70.6 75.4 43.8 63.9 63.0 0.31
Recall@3
ColQwen79.4 85.4 76.7 85.9 70.3 64.7 77.3 84.9 93.2 72.3 79.0 78.2 –VLMLlama-3.2-11B-Vision7.5 35.2 5.5 52.7 9.9 9.1 4.1 9.5 14.0 4.4 15.2 19.5 5.51
Qwen3-VL-8B-Instruct77.0 79.0 77.8 77.1 70.1 65.3 82.9 79.7 93.2 36.5 73.9 72.9 2.47
GPT-5-nano84.5 85.8 81.6 85.7 77.7 70.5 89.3 85.6 94.7 64.2 82.0 81.0 27.32
GPT-5-mini89.1 92.1 87.6 92.4 89.5 79.7 90.6 89.4 97.7 78.1 88.6 87.8 24.70RerankerUniME (Listwise)79.9 85.4 79.2 86.2 71.6 65.5 79.4 84.9 93.2 72.3 79.8 78.9 0.25
LamRA (Listwise)81.1 84.6 77.6 87.6 77.5 69.9 83.3 86.7 95.5 76.6 82.0 81.3 0.55
MM-R581.6 88.1 80.8 88.6 77.5 70.2 80.2 88.5 95.5 78.8 83.0 82.1 3.74
ZipRerank87.9 86.9 86.9 90.3 86.8 77.0 89.3 90.5 95.5 57.7 84.9 84.4 0.36
ZipRerank −50% 84.8 88.2 82.6 89.3 87.8 75.8 87.1 90.5 92.4 59.1 83.8 83.1 0.31
Recall@5
ColQwen85.9 92.7 81.2 92.5 75.8 69.7 82.7 89.6 95.5 75.9 84.1 83.5 –VLMLlama-3.2-11B-Vision23.6 65.2 42.6 81.8 39.8 40.9 28.5 20.3 43.2 5.8 39.2 44.4 5.51
Qwen3-VL-8B-Instruct86.1 89.6 84.2 86.5 77.6 72.3 86.2 88.7 96.2 46.0 81.3 80.6 2.47
GPT-5-nano88.0 94.0 86.7 91.8 82.5 75.3 91.8 90.5 97.7 71.5 87.0 86.0 27.32
GPT-5-mini92.9 95.7 90.5 96.2 92.3 84.4 94.5 93.0 98.5 81.0 91.9 91.4 24.70RerankerUniME (Listwise)85.9 92.7 83.1 92.5 77.1 70.3 84.4 89.6 95.5 75.9 84.7 83.9 0.25
LamRA (Listwise)86.8 94.9 82.3 93.7 80.6 74.4 87.7 91.4 95.5 79.6 86.7 86.0 0.55
MM-R588.4 93.6 83.3 93.8 80.4 74.1 85.3 91.4 95.5 80.3 86.6 86.1 3.74
ZipRerank92.6 90.0 89.9 95.1 91.3 81.8 94.1 93.2 97.0 67.9 89.3 89.1 0.36
ZipRerank −50% 92.8 90.8 88.4 94.1 90.7 81.4 90.3 92.3 94.7 67.2 88.3 88.1 0.31
5.2. Main Results
Tables 1 and 2 present reranking results on top-20 candi-
dates retrieved by DSE wiki−ss and ColQwen, respectively.
ZipRerank consistently improves upon both first-stage re-
trievers across all values of k, demonstrating strong rerank-
ing effectiveness with less than 0.4 s cached LLM reranking
time.
Compared to MM-R5, our model achieves competitive per-
formance with significantly lower latency and computational
cost. On both DSE wiki−ss and ColQwen inputs, ZipRerankachieves higher Recall@ 3and Recall@ 5, while slightly
underperforming MM-R5 on Recall@ 1. This reflects a
trade-off between speed and top-1 accuracy: MM-R5 ex-
plicitly generates reasoning chains to justify the top result,
benefiting Recall@ 1but incurring substantial autoregressive
overhead.
Compared to zero-shot VLM-based reranking (Sun et al.,
2023), we find that relatively smaller models such as Llama-
3.2-11B-Vision and Qwen3-VL-8B-Instruct do not consis-
tently improve over the first-stage retriever. This suggests
that effective listwise reranking is challenging for smaller
7

Very Efficient Listwise Multimodal Reranking for Long Documents
VLMs, which must both follow the ranking instruction and
jointly reason over up to 20 page images. In contrast, a
stronger VLM such as GPT-5-mini performs substantially
better, motivating our choice to use a capable teacher model
to produce soft labels for Stage 2 training. At the same time,
such large VLMs are impractically slow for deployment;
even via API, they can take over 20 seconds per reranking re-
quest. This gap highlights the need for a specialized reranker
that delivers strong quality under strict latency constraints.
To assess the impact of query-image early interaction, we in-
clude ZipRerank −50% , where only 50% of the visual tokens
are retained after filtering. Note that Qwen3-VL already
applies aggressive pooling (4:1), so this represents a high
compression ratio. As expected, token reduction leads to
moderate performance degradation, but also reduces run-
time. The latency reduction is not fully proportional due to
factors like batch size and architectural overhead from our
two-step inference process, which is used to extract query
embeddings separately.
Overall, these results highlight the practicality of our ap-
proach: it achieves strong reranking gains with latency and
compute budgets suitable for real-world deployment.
5.3. Ablation Study
To assess the contribution of each design component, we
conduct ablation studies on the ZipRerank variants. Table 3
summarizes the results of the ablated variants of ZipRerank
for reranking of DSE wiki−ss top 20 results for the MMDo-
cIR benchmark.
w/o First-Stage Pretraining.It skips the gen-
eral reranking pretraining and directly finetunes
Qwen3-VL-8B-Instruct on Stage 2 training.
Results show that removing the first-stage pretraining
consistently degrades performance. We attribute this drop
to the Stage 2 supervision being less diverse and smaller in
scale, as well as the reduced number of effective training
steps. This ablation indicates that the first-stage pretraining
is necessary for learning strong and generalizable reranking
behavior.
w/o Second-Stage Finetuning.This version skips the mul-
timodal reranking finetuning and trains only with Stage 1
pretraining. Removing the second-stage finetuning consis-
tently reduces performance, indicating that vision-centric
training data provides additional supervision beyond Stage 1
and is important for learning robust multimodal reranking.
w/o Single-Logit Decoding.When we replace single-logit
decoding with standard autoregressive generation, ranking
performance remains largely comparable, but inference be-
comes over 6 times slower. This suggests that our training
recipe effectively aligns the model with the single-logit
decoding mechanism, enabling efficient inference withoutTable 3.Ablation study ofZipRerankon DSE wiki−ss .
Method Macro-Avg↑ Micro-Avg↑ Time (s)↓
Recall@1
ZipRerank64.2 63.3 0.36
w/o first stage63.4 62.6 0.36
w/o second stage61.8 60.6 0.36
w/o single-logit decoding64.2 63.3 2.19
w/o soft-ranking loss56.8 55.5 0.36
Recall@3
ZipRerank84.8 84.5 0.36
w/o first stage83.8 83.8 0.36
w/o second stage78.8 78.6 0.36
w/o single-logit decoding83.7 83.2 2.19
w/o soft-ranking loss79.2 79.7 0.36
Recall@5
ZipRerank89.0 89.4 0.36
w/o first stage88.3 88.7 0.36
w/o second stage84.2 84.6 0.36
w/o single-logit decoding88.0 88.0 2.19
w/o soft-ranking loss85.4 85.8 0.36
sacrificing accuracy.
w/o Soft-Ranking Loss.Replacing the Stage 2 soft-ranking
loss with the RankNet loss used in Stage 1 leads to a mod-
erate performance drop, most notably at k= 1 . This result
supports the effectiveness of the soft-ranking objective for
learning from noisy, teacher-augmented supervision.
5.4. Parameter Study
Effect of ρ.We vary the visual-token keep ratio ρin
Eq.(4)and report the resulting LLM time and Recall@ k.
As predicted by the compute model in Appendix A.1, Fig. 3
shows that decreasing ρreduces time in the long-context
regime, but also incurs a drop in reranking quality. This
trade-off can be tuned to the latency and accuracy needs
of a given application, highlighting the flexibility of our
query–image early interaction token filtering.
Scaling with k.We vary the number of input candidates k
to assess the robustness of ZipRerank as the reranking list
grows. As shown in Fig. 4, performance is generally stable
fork≥20 , whereas k= 10 yields lower recall due to a
limited candidate set from the retriever. As expected, LLM
time increases with kas more candidate pages contribute
additional input tokens.
5.5. Generalization to New Benchmark
We evaluate ZipRerank on the English subset of ViDoRe to
test out-of-domain generalization. As shown in Table 4,
ZipRerank achieves the best NDCG@ 5among listwise
rerankers with both DSE and ColQwen. It improves over
MM-R5 from 49.0 to53.4 with DSE and from 55.8 to59.9
8

Very Efficient Listwise Multimodal Reranking for Long Documents
0.1 0.3 0.5 0.7 0.9
Token Retention Ratio ()
30405060708090Recall (%)
Recall@1 Recall@3 Recall@5 LLM Time
100150200250300350400450
LLM Time (ms)
Figure 3.Parameter study on the effect of image token keep ratio
ρon reranking effectiveness (Recall@ 1,3,5 ) and latency (LLM
Time in ms) on first stage results from DSE wiki−ss .
10 20 30 40
Number of Candidates (k)556065707580859095Recall (%)
Recall@1 Recall@3 Recall@5 LLM Time
100200300400500600700
LLM Time (ms)
Figure 4.Parameter study on the effect of the number of input
passages kon reranking effectiveness (Recall@ 1,3,5 ) and latency
(LLM Time in ms) on first stage results from DSE wiki−ss .
with ColQwen. ZipRerank −50% remains strong despite re-
taining only half of the visual tokens, suggesting that the effi-
ciency gain is not specific to MMDocIR. Pointwise LamRA
achieves the highest score, but requires candidate-wise scor-
ing, whereas ZipRerank performs listwise reranking in a
single forward pass.
5.6. Robustness to Teacher Model
We study the sensitivity of ZipRerank to teacher strength by
replacing GPT-5-mini with the weaker GPT-5-nano teacher.
As shown in Table 5, ZipRerank trained with GPT-5-nano
remains close to the GPT-5-mini version. Moreover, it con-
sistently outperforms the GPT-5-nano teacher itself; for
example, on DSE wiki−ss it improves from 59.0/79.1/84.7
to63.6/82.2/87.1 on Recall@ 1/3/5 . This suggests that
ZipRerank is robust to weaker teacher supervision and ben-
efits from the proposed two-stage training and soft-ranking
objective.Table 4.NDCG@5 on ViDoRe (English).
Method DSE ColQwen
DSE wiki−ss (First-stage)41.0 50.5
UniME (Listwise)42.6 51.4
LamRA (Listwise)48.0 54.9
MM-R5 (Listwise)49.0 55.8
ZipRerank (Listwise)53.4 59.9
ZipRerank −50% (Listwise)52.2 58.5
DocReRank (Pointwise)53.3 56.5
LamRA (Pointwise)56.1 60.0
Table 5.Robustness to Stage-2 teacher strength. We compare
teacher models with ZipRerank models trained using their gener-
ated rankings. Ma-Avg stands for Macro Average, and Mi-Avg
stands for Micro Average of recall@ kscores of the individual
datasets.
DSE wiki−ss ColQwenMethodMa-Avg↑ Mi-Avg↑ Ma-Avg↑ Mi-Avg↑
Recall@1
GPT-5-mini70.0 69.2 70.9 70.1
→ZipRerank mini 64.2 63.3 66.1 65.1
GPT-5-nano59.8 59.0 63.6 62.5
→ZipRerank nano 63.8 63.6 65.2 64.8
Recall@3
GPT-5-mini88.0 88.3 88.6 87.8
→ZipRerank mini 84.8 84.5 84.9 84.4
GPT-5-nano79.6 79.1 82.0 81.0
→ZipRerank nano 81.6 82.2 83.6 83.0
Recall@5
GPT-5-mini90.9 91.3 91.9 91.4
→ZipRerank mini 89.0 89.4 89.3 89.1
GPT-5-nano84.5 84.7 87.0 86.0
→ZipRerank nano 86.6 87.1 88.5 87.8
6. Conclusion
We present ZipRerank, a framework for training highly ef-
ficient listwise multimodal rerankers for long documents.
ZipRerank employs a two-stage training pipeline that com-
bines large-scale text reranking and vision-centric VQA-
style data, together with complementary objectives, to
equip the model with strong reranking capability. To ad-
dress the prohibitive latency caused by long multimodal
input sequences and autoregressive decoding, we propose
a lightweight query-image early interaction mechanism for
query-aware visual token reduction and adopt single-logit
decoding to accelerate inference. Extensive experiments on
MMDocIR and ViDoRe show that ZipRerank matches or
surpasses state-of-the-art multimodal rerankers while being
substantially more efficient, making it suitable for deploy-
ment in latency-sensitive real-world systems.
9

Very Efficient Listwise Multimodal Reranking for Long Documents
Impact Statement
This work improves the efficiency of multimodal rerank-
ing for long-document retrieval, which can reduce infer-
ence cost and energy use in real-world retrieval systems.
However, several limitations remain. ZipRerank relies on
teacher-generated rankings during Stage 2 training, which
may inherit biases or errors from the teacher model. In
addition, query-aware pruning can discard useful visual
tokens under aggressive compression, especially for fine-
grained evidence such as small text, dense tables, or visually
similar pages. Our evaluation is also mainly focused on
document-image reranking benchmarks, and further vali-
dation on more diverse domains, languages, and retrieval
settings would strengthen the generality of the conclusions.
Potential negative impacts include enabling faster large-
scale search over sensitive documents and amplifying harms
from biased or incorrect retrieval results. We recommend
deploying the method with appropriate access controls, pri-
vacy safeguards, and evaluation for bias and failure cases in
downstream applications.
References
Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D.,
Zitnick, C. L., and Parikh, D. Vqa: Visual question
answering. InProceedings of the IEEE international
conference on computer vision, pp. 2425–2433, 2015.
Bai, S., Cai, Y ., Chen, R., Chen, K., Chen, X., Cheng, Z.,
Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z.,
Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li,
Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu,
J., Liu, C., Liu, Y ., Liu, D., Liu, S., Lu, D., Luo, R., Lv,
C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun,
Y ., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang,
Q., Wang, Y ., Xie, T., Xu, Y ., Xu, H., Xu, J., Yang, Z.,
Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang,
H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F.,
Zhou, J., Zhu, Y ., and Zhu, K. Qwen3-vl technical report.
arXiv preprint arXiv:2511.21631, 2025.
Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M.,
Hamilton, N., and Hullender, G. Learning to rank using
gradient descent. InProceedings of the 22nd international
conference on Machine learning, pp. 89–96, 2005.
Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D.,
and Liu, Z. Bge m3-embedding: Multi-lingual,
multi-functionality, multi-granularity text embeddings
through self-knowledge distillation.arXiv preprint
arXiv:2402.03216, 2024a.
Chen, Z., Pradeep, R., and Lin, J. An early first repro-
duction and improvements to single-token decoding forfast listwise reranking.arXiv preprint arXiv:2411.05508,
2024b.
Chen, Z., Xu, C., Qi, Y ., and Guo, J. Mllm is a strong
reranker: Advancing multimodal retrieval-augmented
generation via knowledge-enhanced reranking and noise-
injected training.arXiv preprint arXiv:2407.21439,
2024c.
Dai, S., Huang, Q., You, X., and Yu, J. Mg2-rag: Multi-
granularity graph for multimodal retrieval-augmented
generation.arXiv preprint arXiv:2604.04969, 2026.
Dao, T. Flashattention-2: Faster attention with bet-
ter parallelism and work partitioning.arXiv preprint
arXiv:2307.08691, 2023.
Dong, K., Chang, Y ., Goh Xin Deik, D., Li, D., Tang,
R., and Liu, Y . MMDocIR: Benchmarking multi-
modal retrieval for long documents. InProceedings
of the 2025 Conference on Empirical Methods in Nat-
ural Language Processing, pp. 30959–30993, November
2025. URL https://aclanthology.org/2025.
emnlp-main.1576/.
Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G.,
HUDELOT, C., and Colombo, P. Colpali: Efficient docu-
ment retrieval with vision language models. InThe Thir-
teenth International Conference on Learning Represen-
tations, 2025. URL https://openreview.net/
forum?id=ogjBpZ8uSi.
Feng, Y ., Sun, Y ., Sun, Y ., Zhu, M., Huang, Q., Tung, A.
K. H., and Chen, W. Don’t reinvent the wheel: Ef-
ficient instruction-following text embedding based on
guided space transformation. InProceedings of the 63rd
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 24511–24525,
2025. URL https://aclanthology.org/2025.
acl-long.1196/.
Gangi Reddy, R., Doo, J., Xu, Y ., Sultan, M. A., Swain,
D., Sil, A., and Ji, H. FIRST: Faster improved list-
wise reranking with single token decoding. InPro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 8642–8652,
November 2024. doi: 10.18653/v1/2024.emnlp-main.
491. URL https://aclanthology.org/2024.
emnlp-main.491/.
Gao, S., Zhao, S., Jiang, X., Duan, L., Chng, Y . X., Chen,
Q.-G., Luo, W., Zhang, K., Bian, J.-W., and Gong, M.
Scaling beyond context: A survey of multimodal retrieval-
augmented generation for document understanding.arXiv
preprint arXiv:2510.15253, 2025.
10

Very Efficient Listwise Multimodal Reranking for Long Documents
Gu, T., Yang, K., Zhang, K., An, X., Feng, Z., Zhang, Y .,
Cai, W., Deng, J., and Bing, L. Unime-v2: Mllm-as-a-
judge for universal multimodal embedding learning. In
Proceedings of the AAAI Conference on Artificial Intelli-
gence, volume 40, pp. 21378–21386, 2026.
Han, J., Du, L., Wu, Y ., Zhou, X., Du, H., and Zheng, W.
Adafv: Rethinking of visual-language alignment for vlm
acceleration.arXiv preprint arXiv:2501.09532, 2025.
Hao, J., Zhu, Y ., Wang, T., Yu, J., Xin, X., Zheng, B., Ren,
Z., and Guo, S. Omnikv: Dynamic context selection for
efficient long-context llms. InThe Thirteenth Interna-
tional Conference on Learning Representations, 2025.
Hao, J., Huang, Q., Wang, Y ., Zhang, M., and Yu, J. Deltakv:
Residual-based kv cache compression via long-range sim-
ilarity.arXiv preprint arXiv:2602.08005, 2026.
Hassan, E., Chaudhury, S., and Gopal, M. Multi-modal
information integration for document retrieval. In2013
12th International Conference on Document Analysis and
Recognition, pp. 1200–1204. IEEE, 2013.
Hendrycks, D., Burns, C., Chen, A., and Ball, S. CUAD: an
expert-annotated NLP dataset for legal contract review.
InProceedings of the Neural Information Processing
Systems Track on Datasets and Benchmarks 1, NeurIPS
Datasets and Benchmarks, 2021.
Huang, Q., Feng, J., and Fang, Q. Reverse query-aware
locality-sensitive hashing for high-dimensional furthest
neighbor search. In2017 IEEE 33rd International Con-
ference on Data Engineering (ICDE), pp. 167–170, 2017.
Huang, Q., Ma, G., Feng, J., Fang, Q., and Tung, A. K.
Accurate and fast asymmetric locality-sensitive hashing
scheme for maximum inner product search. InProceed-
ings of the 24th ACM SIGKDD International Conference
on Knowledge Discovery & Data Mining (KDD), pp.
1561–1570, 2018.
Huang, Q., Lei, Y ., and Tung, A. K. Point-to-hyperplane
nearest neighbor search beyond the unit hypersphere. In
Proceedings of the 2021 International Conference on
Management of Data (SIGMOD), pp. 777–789, 2021.
Huang, Q., Wang, Y ., and Tung, A. K. SAH: Shifting-
aware asymmetric hashing for reverse k maximum inner
product search. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 37, pp. 4312–4321,
2023.
Huang, Q., Wang, Y ., Sun, Y ., and Tung, A. K. Diversity-
aware k-maximum inner product search revisited.arXiv
preprint arXiv:2402.13858, 2024.Jiang, D., Zhang, R., Guo, Z., Wu, Y ., Lei, J., Qiu, P.,
Lu, P., Chen, Z., Fu, C., Song, G., et al. Mmsearch:
Benchmarking the potential of large models as multi-
modal search engines.arXiv preprint arXiv:2409.12959,
2024.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov,
S., Chen, D., and Yih, W.-t. Dense passage retrieval
for open-domain question answering. InProceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 6769–6781, 2020.
Khattab, O. and Zaharia, M. Colbert: Efficient and effective
passage search via contextualized late interaction over
bert. InProceedings of the 43rd International ACM SIGIR
conference on research and development in Information
Retrieval, pp. 39–48, 2020.
Kim, B. S., Kim, J., Lee, D., and Jang, B. Visual question
answering: A survey of methods, datasets, evaluation,
and challenges.ACM Computing Surveys, 57(10):1–35,
2025.
Landeghem, J. V ., Powalski, R., Tito, R., Jurkiewicz, D.,
Blaschko, M. B., Borchmann, L., Coustaty, M., Moens,
S., Pietruszka, M., Anckaert, B., Stanislawek, T., J ´oziak,
P., and Valveny, E. Document understanding dataset and
evaluation (DUDE). InIEEE/CVF International Con-
ference on Computer Vision, ICCV 2023, Paris, France,
October 1-6, 2023, pp. 19471–19483, 2023.
Lee, J., Ko, J., Baek, J., Jeong, S., and Hwang, S. J. Uni-
fied multimodal interleaved document representation for
retrieval.arXiv preprint arXiv:2410.02729, 2024.
Lei, Y ., Huang, Q., Kankanhalli, M., and Tung, A. Sublinear
time nearest neighbor search over generalized weighted
space. InInternational Conference on Machine Learning,
pp. 3773–3781. PMLR, 2019.
Lei, Y ., Huang, Q., Kankanhalli, M., and Tung, A. K.
Locality-sensitive hashing scheme based on longest cir-
cular co-substring. InProceedings of the 2020 ACM
SIGMOD International Conference on Management of
Data (SIGMOD), pp. 2589–2599, 2020.
Lerner, P., Ferret, O., and Guinaudeau, C. Cross-modal
retrieval for knowledge-based visual question answering.
InEuropean Conference on Information Retrieval, pp.
421–438. Springer, 2024.
Li, W., Zhang, Y ., Sun, Y ., Wang, W., Li, M., Zhang, W.,
and Lin, X. Approximate nearest neighbor search on
high dimensional data—experiments, analyses, and im-
provement.IEEE Transactions on Knowledge and Data
Engineering, 32(8):1475–1488, 2019.
11

Very Efficient Listwise Multimodal Reranking for Long Documents
Li, X. and Li, J. AoE: Angle-optimized embeddings for
semantic textual similarity. InProceedings of the 62nd
Annual Meeting of the Association for Computational
Linguistics (ACL), pp. 1825–1839, 2024.
Li, Y ., Huang, Y ., Yang, B., Venkitesh, B., Locatelli, A., Ye,
H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm knows
what you are looking for before generation.Advances
in Neural Information Processing Systems, 37:22947–
22970, 2024.
Liu, W., Ma, X., Sun, W., Zhu, Y ., Li, Y ., Yin, D., and Dou,
Z. Reasonrank: Empowering passage ranking with strong
reasoning ability.arXiv preprint arXiv:2508.07050,
2025a.
Liu, Y ., Zhang, Y ., Cai, J., Jiang, X., Hu, Y ., Yao, J., Wang,
Y ., and Xie, W. Lamra: Large multimodal model as
your advanced retrieval assistant. InProceedings of the
Computer Vision and Pattern Recognition Conference, pp.
4015–4025, 2025b.
Long, M., Sun, D., Yang, D., Wang, J., Luo, Y ., Shen, Y .,
Wang, J., Zhou, H., Guo, C., Wei, P., et al. Diver: A
multi-stage approach for reasoning-intensive information
retrieval.arXiv preprint arXiv:2508.07995, 2025.
Loshchilov, I. and Hutter, F. Decoupled weight decay regu-
larization.arXiv preprint arXiv:1711.05101, 2017.
Ma, X., Lin, S.-C., Li, M., Chen, W., and Lin, J. Unifying
multimodal retrieval via document screenshot embed-
ding. InProceedings of the 2024 Conference on Em-
pirical Methods in Natural Language Processing, pp.
6492–6505, November 2024.
Ma, Y ., Li, J., Zang, Y ., Wu, X., Dong, X., Zhang, P., Cao, Y .,
Duan, H., Wang, J., Cao, Y ., and Sun, A. Towards storage-
efficient visual document retrieval: An empirical study
on reducing patch-level embeddings. InFindings of the
Association for Computational Linguistics: ACL 2025,
pp. 19568–19580, July 2025. doi: 10.18653/v1/2025.
findings-acl.1003. URL https://aclanthology.
org/2025.findings-acl.1003/.
Mei, L., Mo, S., Yang, Z., and Chen, C. A survey of mul-
timodal retrieval-augmented generation.arXiv preprint
arXiv:2504.08748, 2025.
Moffat, A. and Zobel, J. Rank-biased precision for mea-
surement of retrieval effectiveness.ACM Transactions on
Information Systems (TOIS), 27(1):1–27, 2008.
Mortaheb, M., Khojastepour, M. A. A., Chakradhar, S. T.,
and Ulukus, S. Re-ranking the context for multi-
modal retrieval augmented generation.arXiv preprint
arXiv:2501.04695, 2025.Muennighoff, N., Tazi, N., Magne, L., and Reimers, N.
Mteb: Massive text embedding benchmark. InProceed-
ings of the 17th Conference of the European Chapter of
the Association for Computational Linguistics, pp. 2014–
2037, 2023.
Nogueira, R. and Cho, K. Passage re-ranking with bert.
arXiv preprint arXiv:1901.04085, 2019.
Pradeep, R., Sharifymoghaddam, S., and Lin, J. Rankvicuna:
Zero-shot listwise document reranking with open-source
large language models.arXiv preprint arXiv:2309.15088,
2023a.
Pradeep, R., Sharifymoghaddam, S., and Lin, J.
Rankzephyr: Effective and robust zero-shot listwise
reranking is a breeze!arXiv preprint arXiv:2312.02724,
2023b.
Rathee, M., MacAvaney, S., and Anand, A. Guiding re-
trieval using llm-based listwise rankers. InEuropean Con-
ference on Information Retrieval, pp. 230–246. Springer,
2025.
Reimers, N. and Gurevych, I. Sentence-BERT: Sentence
embeddings using siamese bert-networks. InProceedings
of the 2019 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 3982–3992, 2019.
Shao, K., Tao, K., Zhang, K., Feng, S., Cai, M., Shang, Y .,
You, H., Qin, C., Sui, Y ., and Wang, H. When tokens talk
too much: A survey of multimodal long-context token
compression across images, videos, and audios.arXiv
preprint arXiv:2507.20198, 2025.
Sharifymoghaddam, S., Pradeep, R., Slavescu, A., Nguyen,
R., Xu, A., Chen, Z., Zhang, Y ., Chen, Y ., Xian, J., and
Lin, J. Rankllm: A python package for reranking with
llms. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in In-
formation Retrieval, pp. 3681–3690, 2025.
Song, D., Wang, W., Chen, S., Wang, X., Guan, M. X.,
and Wang, B. Less is more: A simple yet effective to-
ken reduction method for efficient multi-modal LLMs. In
Proceedings of the 31st International Conference on Com-
putational Linguistics, pp. 7614–7623, January 2025.
Su, J., Ahmed, M., Lu, Y ., Pan, S., Bo, W., and
Liu, Y . Roformer: Enhanced transformer with
rotary position embedding.Neurocomputing,
568:127063, 2024. ISSN 0925-2312. doi:
https://doi.org/10.1016/j.neucom.2023.127063.
URL https://www.sciencedirect.com/
science/article/pii/S0925231223011864.
12

Very Efficient Listwise Multimodal Reranking for Long Documents
Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin,
D., and Ren, Z. Is chatgpt good at search? investigating
large language models as re-ranking agents. InProceed-
ings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pp. 14918–14937, 2023.
Sun, Y ., Huang, Q., Wang, Y ., and Tung, A. K. Diversinews:
Enriching news consumption with relevant yet diverse
news articles retrieval.Proceedings of the VLDB Endow-
ment, 17(12):4277–4280, 2024.
Sun, Y ., Huang, Q., Tang, Y ., Tung, A. K. H., and Yu,
J. A general framework for producing interpretable
semantic text embeddings. InThe Thirteenth Interna-
tional Conference on Learning Representations (ICLR),
2025a. URL https://openreview.net/forum?
id=23uY3FpQxc.
Sun, Y ., Huang, Q., Tung, A. K., and Yu, J. Text embed-
dings should capture implicit semantics, not just surface
meaning.arXiv preprint arXiv:2506.08354, 2025b.
Sun, Y ., Huang, Q., Tung, A. K. H., and Yu, J. Prism:
A framework for producing interpretable political bias
embeddings with political-aware cross-encoder. InPro-
ceedings of the 63rd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers), pp. 27719–27733, 2025c. URL https://
aclanthology.org/2025.acl-long.1344/.
Sun, Y ., Huang, Q., Xu, Z., Sun, Y ., Tang, Y ., and Tung,
A. K. One swallow does not make a summer: Under-
standing semantic structures in embedding spaces.arXiv
preprint arXiv:2512.00852, 2025d.
Tanaka, R., Nishida, K., Nishida, K., Hasegawa, T., Saito, I.,
and Saito, K. SlideVQA: A Dataset for Document Visual
Question Answering on Multiple Images. InProceedings
of the AAAI Conference on Artificial Intelligence, pp.
13636–13645, 2023.
Tang, J., Zhao, Y ., Zhu, K., Xiao, G., Kasikci, B., and Han,
S. Quest: Query-aware sparsity for efficient long-context
llm inference.arXiv preprint arXiv:2406.10774, 2024.
Thakur, N., Reimers, N., R ¨uckl´e, A., Srivastava, A., and
Gurevych, I. Beir: A heterogenous benchmark for zero-
shot evaluation of information retrieval models.arXiv
preprint arXiv:2104.08663, 2021.
Tito, R., Karatzas, D., and Valveny, E. Hierarchical multi-
modal transformers for multipage docvqa.Pattern Recog-
nition, 144:109834, 2023.
Wan, Y ., Liu, Y ., Ajith, A., Grazian, C., Hoex, B., Zhang, W.,
Kit, C., Xie, T., and Foster, I. Sciqag: A framework for
auto-generated science question answering dataset with
fine-grained evaluation, 2024. URL https://arxiv.
org/abs/2405.09939.Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L.,
Jiang, D., Majumder, R., and Wei, F. Text embeddings
by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
Wasserman, N., Heinimann, O., Golbari, Y ., Zimbalist, T.,
Schwartz, E., and Irani, M. Docrerank: Single-page hard
negative query generation for training multi-modal rag
rerankers. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pp.
8651–8669, 2025.
Xu, M., Dong, J., Hou, J., Wang, Z., Li, S., Gao, Z., Zhong,
R., and Cai, H. Mm-r5: Multimodal reasoning-enhanced
reranker via reinforcement learning for document re-
trieval.arXiv preprint arXiv:2506.12364, 2025.
Yang, S., Chen, Y ., Tian, Z., Wang, C., Li, J., Yu, B., and Jia,
J. Visionzip: Longer is better but not necessary in vision
language models. InProceedings of the Computer Vision
and Pattern Recognition Conference, pp. 19792–19802,
2025.
Ye, X., Gan, Y ., Huang, X., Ge, Y ., and Tang, Y . V oco-
llama: Towards vision compression with large language
models. InProceedings of the Computer Vision and
Pattern Recognition Conference, pp. 29836–29846, 2025.
You, X., Huang, Q., Li, L., Chang, X., and Yu, J. Cut
to the chase: Training-free multimodal summarization
via chain-of-events.arXiv preprint arXiv:2603.06213,
2026a.
You, X., Huang, Q., Li, L., Zhang, C., Liu, X., Zhang,
M., and Yu, J. Knowledge completes the vision: A
multimodal entity-aware retrieval-augmented generation
framework for news image captioning. InProceedings
of the AAAI Conference on Artificial Intelligence, pp.
12108–12116, 2026b.
Yu, Z., Yu, J., Fan, J., and Tao, D. Multi-modal factorized
bilinear pooling with co-attention learning for visual ques-
tion answering. InInternational Conference on Computer
Vision (ICCV), pp. 1821–1830, 2017.
Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y ., and Tao,
D. Activitynet-qa: A dataset for understanding complex
web videos via question answering. InProceedings of the
AAAI conference on Artificial Intelligence, volume 33, pp.
9127–9134, 2019a.
Yu, Z., Yu, J., Cui, Y ., Tao, D., and Tian, Q. Deep modular
co-attention networks for visual question answering. In
Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition (CVPR), pp. 6281–6290, 2019b.
13

Very Efficient Listwise Multimodal Reranking for Long Documents
Zhang, H., Ji, X., Chen, Y ., Fu, F., Miao, X., Nie, X., Chen,
W., and Cui, B. Pqcache: Product quantization-based
kvcache for long context llm inference.Proceedings of
the ACM on Management of Data, 3(3):1–30, 2025a.
Zhang, Y ., Fan, C.-K., Ma, J., Zheng, W., Huang, T., Cheng,
K., Gudovskiy, D. A., Okuno, T., Nakata, Y ., Keutzer,
K., et al. Sparsevlm: Visual token sparsification for effi-
cient vision-language model inference. InForty-second
International Conference on Machine Learning, 2025b.
Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang, B.,
Xie, P., Yang, A., Liu, D., Lin, J., et al. Qwen3 embed-
ding: Advancing text embedding and reranking through
foundation models.arXiv preprint arXiv:2506.05176,
2025c.
Zhang, Z., Sheng, Y ., Zhou, T., Chen, T., Zheng, L., Cai,
R., Song, Z., Tian, Y ., R ´e, C., Barrett, C., et al. H2o:
Heavy-hitter oracle for efficient generative inference of
large language models.Advances in Neural Information
Processing Systems, 36:34661–34710, 2023.
Zhao, R., Chen, H., Wang, W., Jiao, F., Do, X. L., Qin,
C., Ding, B., Guo, X., Li, M., Li, X., et al. Retrieving
multimodal information for augmented generation: A
survey. InFindings of the Association for Computational
Linguistics: EMNLP 2023, pp. 4736–4756, 2023.
Zhou, K., Hassan, F. H., and Hoon, G. K. The state of the
art for cross-modal retrieval: A survey.IEEE Access, 11:
138568–138589, 2023.
Zhu, F., Lei, W., Feng, F., Wang, C., Zhang, H., and Chua,
T. Towards complex document understanding by discrete
reasoning. InMM ’22: The 30th ACM International
Conference on Multimedia, Lisboa, Portugal, October 10
- 14, 2022, pp. 4857–4866, 2022.
14

Very Efficient Listwise Multimodal Reranking for Long Documents
A. Theoretical Analysis
A.1. Compute Scaling Law for ZIPRERANK
We derive a compact FLOPs model that makes explicit how inference computation scales with (i) non-image tokens, (ii)
visual tokens, (iii) the compression ratio from query–image early interaction (Eq. (3)–(4)), and (iv) the number of generated
tokens (including both ranking-format tokens and optional reasoning tokens). Following our efficiency evaluation protocol,
we focus on theLLM/VLM decodercompute given cached visual embeddings.
Token accounting.We rerankkcandidate page images. Let
ntext:=# non-image tokens in the prompt (instructions + query + formatting),
Nq:=# query tokens (a subset ofn text; cf. Eq. (2)),
nvis:=kX
i=1Ni# visual tokens across all candidates,
ρ∈(0,1]visual-token keep ratio in Eq. (4),
n(ρ)
vis:=kX
i=1round(ρN i)≈ρ n vis.
Define the (decoder) context lengths
nfull:=n text+n vis, n ρ:=n text+n(ρ)
vis≈n text+ρ n vis.
Output-length decomposition.Let the total number of generated tokens be
u=u rank+u reason,
where urankencodes the output ranking (typically proportional to kfor autoregressive listwise rerankers), and ureason captures
any additional reasoning/explanation tokens (potentially large for reasoning-style rerankers). A simple parametrization is
urank≈βk,
for a small formatting constantβ(e.g., separators/brackets).
Architecture parameters.Let the decoder have LTransformer blocks and hidden width d. We group constant factors
(heads, projections, fused kernels, etc.) into positive constantsc att, cffn, cdec.
Prefill and decoding FLOPs.A standard approximation for a decoder-only Transformer is
Fprefill(n)≈L
cattd n2+cffnd2n
,(5)
Fdecode (n, u)≈u·L·c decd n,(6)
where Eq. (6) assumes KV caching so that per-token decoding is linear inn.
Baseline autoregressive listwise reranker.A baseline that processes all visual tokens and generates ubase=βk+u reason
tokens has
Fbase≈F prefill(nfull) + F decode (nfull, βk+u reason).(7)
ZIPRERANK.ZIPRERANK reduces compute along two orthogonal axes described in Section 4.2: (i) it compresses visual
tokens via query–image early interaction (Eq. (3)–(4)), and (ii) it eliminates multi-step generation via single-token scoring.
Thus,
Fzip≈F score(Nq, nvis)|{z }
early interaction scoring+ F prefill(nρ)|{z}
prefill on compressed context+ F decode (nρ,1)|{z}
single-step scoring.(8)
The early interaction step computes (max) cosine similarities between query hidden states and visual embeddings as in
Eq. (3). A simple FLOPs proxy is
Fscore(Nq, nvis)≲c scored Nqnvis,
15

Very Efficient Listwise Multimodal Reranking for Long Documents
which is highly parallelizable and typically lower-order than the quadratic self-attention term whenn fullis large.
Speedup expression.Combining Eq. (7)–(8), the FLOPs speedup is
Speedup :=Fbase
Fzip≈Fprefill(ntext+n vis) + F decode (ntext+n vis, βk+u reason)
Fscore(Nq, nvis) + F prefill(ntext+ρn vis) + F decode (ntext+ρn vis,1).
Two informative regimes. Long-context regime.Ifn vis≫n textand the attention term dominates prefill, then
Fprefill(nfull)
Fprefill(nρ)≈ntext+n vis
ntext+ρn vis2
≈1
ρ2.
Generation-heavy regime.If decoding dominates the baseline (large βkand/or ureason ), then single-token scoring yields an
additional gain roughly proportional to the baseline output length:
Fdecode (nfull, βk+u reason)
Fdecode (nρ,1)≈(βk+u reason)·ntext+n vis
ntext+ρn vis.
A.2. Early Interaction Pruning as a Surrogate for Attention Pooling
ZIPRERANK scores each visual token by its maximum cosine similarity to any query token (Eq. (3)) and retains the
top-round(ρN i)tokens per image (Eq. (4)). Intuitively, this early interaction step provides a cheap query-aware proxy for
how an attention layer would pool (aggregate) information from the query when deciding which visual tokens are most
relevant.
More concretely, for each visual token j, one can define a smooth attention-style pooling score over query tokens via a
log-sum-exp (LSE) aggregation gj:= logPNq
t=1exp(s t,j), where st,jis the query–visual similarity. This score corresponds
to a soft maximum over query tokens. ZIPRERANK instead uses the hard maximum aj:= max tst,j, which is cheaper and
simpler to compute.
We show the hard max is a tight surrogate for the smooth pooling score, differing by at most an additive logN qterm. As a
result, when the boundary gap between the K-th and (K+1) -th token scores is sufficiently large, selecting the top- Ktokens
under max-sim pruning matches the top-Ktokens selected under the smooth pooling rule.
Setup.Fix an image and drop the image index. Let ht∈Rdbe the hidden state of query token t∈ {1, . . . , N q}(cf.
Eq. (2)) andv j∈Rdbe a visual token embedding. Define cosine similarities
st,j:=h⊤
tvj
∥ht∥∥vj∥.
ZIPRERANK uses
aj:= max
1≤t≤N qst,j.
A smooth alternative consistent with “soft” pooling across query tokens is
gj:= logNqX
t=1exp(s t,j).
Lemma A.1(Max vs. log-sum-exp).For everyj,
aj≤gj≤aj+ logN q.
Proof. Lower bound:P
texp(s t,j)≥exp(max tst,j). Upper bound:P
texp(s t,j)≤N qexp(max tst,j). Taking log
completes the proof.
Corollary A.2(Top- Kstability under a margin).Let a(1)≥a(2)≥. . . be the sorted {aj}values. If a(K)−a(K+1) >
logN q, then the top-Ktoken set selected by{a j}is identical to the top-Ktoken set selected by{g j}.
16

Very Efficient Listwise Multimodal Reranking for Long Documents
A.3. Pruning Error Bound via Tail Attention Mass
We bound the representation error induced by discarding tokens in an attention layer in terms of the total attention mass
removed.
Setup.Let α∈∆nvis−1be attention weights over nvisvisual tokens with value vectors vjsatisfying ∥vj∥ ≤V max. The
unpruned attention output is
c:=nvisX
j=1αjvj.
LetSbe the retained token indices and define the tail mass
ε:=X
j /∈Sαj.
After pruning and renormalization,
c′:=X
j∈Sαj
1−εvj.
Theorem A.3(Pruning error controlled by tail mass).If∥v j∥ ≤V maxfor allj, then
∥c−c′∥ ≤2ε V max.
Proof.Writec−c′=P
j /∈Sαjvj+P
j∈S
αj−αj
1−ε
vj.Sinceαj−αj
1−ε=ε
1−εαj,we have
∥c−c′∥ ≤X
j /∈Sαj∥vj∥+X
j∈Sε
1−εαj∥vj∥
≤εV max+ε
1−ε(1−ε)V max= 2εV max.
Relating tail mass to score separation.If αj= exp(g j)/P
ℓexp(g ℓ)for scores {gj}andSis the top- Kset under gj,
then with gapδ:=g (K)−g(K+1) >0,
ε≤P
j>Kexp(g (j))P
ℓ≤Kexp(g (ℓ))≤nvis−K
Kexp(−δ),(9)
showing the discarded mass decays exponentially with the boundary gap.
B. Implementation Details
B.1. Training
We adopt a two-stage training approach to develop our multimodal document reranker. Stage 1 pre-trains the model on
text passages from RankZephyr training data2rendered as images, while Stage 2 fine-tunes on real document images from
MMDocIR training dataset. All experiments were conducted on a single NVIDIA H200 GPU.
Base Model.We use QWEN3-VL-8B-INSTRUCT(Bai et al., 2025)3as our base vision-language model. During training,
we freeze the vision encoder and only update the language model parameters. We use Flash Attention 2 (Dao, 2023) for
efficient attention computation and gradient checkpointing to reduce memory consumption.
Training Objective.Our model is trained with a combined objective consisting of two loss components:
1.Language Modeling Loss:Standard cross-entropy loss on the ranking output sequence, with prompt tokens masked.
2https://huggingface.co/datasets/rryisthebest/rank_zephyr_training_data_alpha/
3https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
17

Very Efficient Listwise Multimodal Reranking for Long Documents
A] > [C] > [B] > ... > [D] <|im_end|>Target Sequence: Input Sequence: 
<|im_start|>user
You are RankGPT , an intelligent assistant that can rank passages
based on their relevancy to the query .
I will provide you with {num}  passages as images.
Rank the passages based on their relevance to the search query .
The images are provided in order: Picture 1 is passage [A],
Picture 2 is passage [B], ..., Picture {num}  is passage
[{last_label}].
Search Query:  {query}
Rank the passages above based on their relevance to the search
query .
The passages should be listed in descending order using
identifiers.
The most relevant passages should be listed first.
The output format should be [A] > [B], etc.
Only output the ranking results, do not say anything else.
Picture 1: <|vision_start|><|image_pad|><|vision_end|>
Picture 2: <|vision_start|><|image_pad|><|vision_end|>
...
Picture {num}: <|vision_start|><|image_pad|><|vision_end|>
<|im_end|>
<|im_start|>assistant
[
Figure 5.The reranking input prompt template and example target generation sequence forZipRerankwith Qwen3-VL.
2.Ranking Loss:Stage 1 uses weighted RankNet while Stage 2 uses a soft ranking loss with position-decayed target
distribution.
Stage 1: Pre-training on Rendered Text.In the first stage, we train the model on the RankZephyr dataset (Pradeep et al.,
2023b), which contains text passages with relevance labels. We render each text passage as a 280 ×280 pixel image using
dynamic font sizing to maximize canvas utilization. The training hyperparameters are summarized in Table 6a.
Stage 2: Fine-tuning on Document Images.In the second stage, we continue training from the Stage 1 checkpoint on the
MMDocIR training dataset (Dong et al., 2025), which contains real document page images with GPT-5-MINI-generated
relevance rankings. The MMDocIR training dataset is built from the following DocVQA datasets: MP-DocVQA (Tito et al.,
2023), SlideVQA (Tanaka et al., 2023), TAT-DQA (Zhu et al., 2022), SciQAG (Wan et al., 2024), DUDE (Landeghem
et al., 2023), and CUAD (Hendrycks et al., 2021). Images are resized such that the largest dimension does not exceed 1024
pixels. We force the ground truth page to position 0 in the target ranking to ensure consistent supervision. The training
hyperparameters are summarized in Table 6b.
Optimizer.We use 8-bit AdamW (Loshchilov & Hutter, 2017) with no weight decay. Learning rate scheduling follows a
cosine decay schedule after linear warmup.
Prompt Template.We use a consistent prompt template and output formatting across both training stages and evaluation in
Fig. 5.
Query-Image Early Interaction Pruning.For the token-pruned variant, we implement query-image early interaction as
a lightweight, non-parametric visual token selection module. Given query-token hidden states {ht}Nq
t=1and visual token
embeddings {vi,j}Ni
j=1for image i, we first ℓ2-normalize both text and visual embeddings and compute the cosine similarity
18

Very Efficient Listwise Multimodal Reranking for Long Documents
Table 6.Training hyperparameters for Stage 1 and Stage 2.
(a)Stage 1 (rendered text) hyperparameters.
Hyperparameter Value
Dataset Rank-Zephyr
Epochs 3
Learning rate3×10−6
Batch size 8
Gradient accumulation steps 4
Effective batch size 32
LR scheduler Cosine
Warmup steps 100
Ranking loss Weighted RankNet
Ranking loss weight (λ 1) 10.0
Max candidates per query 20
Image size 280×280
Precision BF16(b)Stage 2 (document images) hyperparameters.
Hyperparameter Value
Dataset MMDocIR Training Set
Epochs 1
Learning rate3×10−6
Batch size 2
Gradient accumulation steps 8
Effective batch size 16
LR scheduler Cosine
Warmup steps 50
Ranking loss Soft ranking
Ranking loss weight (λ 2) 1.0
Position decay (γ) 0.5
Max candidates per query 20
Precision BF16
Table 7.Model checkpoints used in experiments.
Model Type Checkpoint / API
First-Stage Retrievers
DSE Dense RetrieverMrLight/dse-qwen2-2b-mrl-v1
ColQwen Late-Interactionvidore/colqwen2-v1.0
Rerankers
Qwen3-VL-8B VLMQwen/Qwen3-VL-8B-Instruct
Llama-3.2-11B-Vision VLMmeta-llama/Llama-3.2-11B-Vision
MM-R5 Multimodal Rerankeri2vec/MM-R5
LamRA Multimodal Rerankercode-kunkun/LamRA-Rank
DocReRank Multimodal RerankerDocReRank/DocReRank-Reranker
UniME-V2 Multimodal RerankerTianchengGu/UniME-V2-reranker-Qwen25VL-7B
matrix:
st,i,j=h⊤
tvi,j
∥ht∥∥vi,j∥.
Each visual token is then assigned its maximum similarity to any query token:
ai,j= max
1≤t≤N qst,i,j.
For each image, we retain the top-K ivisual tokens according toa i,j, where
Ki= max(1,round(ρN i)),
andρis the token keep ratio. After top- Kiselection, we sort the selected indices in their original order before feeding them
to the LLM, so that the remaining visual tokens preserve their spatial ordering and original positional encodings.
The pruning module has no trainable parameters and is used only for token selection. We apply the same selected indices
to all corresponding visual feature streams in Qwen3-VL, including the deepstack features, so that the compressed visual
sequence remains aligned across layers.
B.2. Evaluation
First-Stage Retrieval.We evaluate with two first-stage retrievers to demonstrate the generalizability of our reranking
approach:
•DSE wiki−ss (Ma et al., 2024): Document Screenshot Embedding encodes queries and document pages into a shared
embedding space using a vision-language model. Retrieval scores are computed via dot product similarity.
19

Very Efficient Listwise Multimodal Reranking for Long Documents
Table 8.Ablation study ofZipRerankon ColQwen.
Method Macro-Avg↑ Micro-Avg↑ Time (s)↓
Recall@1
ZipRerank66.1 65.1 0.36
w/o first stage64.0 63.5 0.36
w/o second stage63.0 61.6 0.36
w/o single-logit decoding66.1 65.1 2.23
w/o soft-ranking loss64.0 63.5 0.36
Recall@3
ZipRerank84.9 84.4 0.36
w/o first stage83.8 83.5 0.36
w/o second stage81.1 80.3 0.36
w/o single-logit decoding84.5 83.6 2.23
w/o soft-ranking loss82.9 82.1 0.36
Recall@5
ZipRerank89.3 89.1 0.36
w/o first stage88.6 88.4 0.36
w/o second stage86.1 85.4 0.36
w/o single-logit decoding89.1 88.5 2.23
w/o soft-ranking loss87.8 87.3 0.36
•ColQwen (Faysse et al., 2025): A ColBERT-style (Khattab & Zaharia, 2020) late-interaction retriever that represents
queries and documents as multi-vector embeddings. Retrieval scores are computed using MaxSim (maximum similarity)
between query and document token embeddings.
For each query, the first-stage retriever returns the top-20 candidate pages within the same document. These retrieval results
are cached for efficient reranking evaluation.
Reranking Setup.Given the top-20 candidates from the first-stage retriever, the reranker processes all candidates
simultaneously in a single forward pass. Document page images are resized such that the largest dimension does not
exceed 1024 pixels. For ranking prediction, we extract logits at the position immediately after the prompt (before the first
output token) and compute scores for each candidate letter token (A, B, C, ...). The candidates are then reranked by their
corresponding logit scores.
Checkpoints.Table 7 summarizes the checkpoint names used in the experiments.
C. Additional Experimental Results
C.1. Ablation and Parameter Study Results on ColQwen
In addition to the main ablation and parameter studies in Secs. 5.3 and 5.4 based on DSE wiki−ss , we report corresponding
results with ColQwen in Table 8, Fig. 6a, and Fig. 6b. Overall, we observe a similar pattern with ColQwen as the first-stage
retriever.
C.2. Random Pruning vs. Text-to-Image Pruning
To verify that the benefit of token pruning comes from query-aware selection rather than simply reducing the number of
visual tokens, we compare our text-to-image (T2I) pruning strategy with random pruning under the same keep ratio. Random
pruning uniformly retains the same number of visual tokens per image, while T2I pruning keeps tokens with the highest
query-image similarity scores.
As shown in Table 9, T2I pruning consistently outperforms random pruning across almost all keep ratios and metrics. The
advantage is especially clear under aggressive compression. For example, at a keep ratio of 0.3, T2I pruning improves
20

Very Efficient Listwise Multimodal Reranking for Long Documents
0.1 0.3 0.5 0.7 0.9
Token Retention Ratio ()
30405060708090Recall (%)
Recall@1 Recall@3 Recall@5 LLM Time
100150200250300350400450
LLM Time (ms)
(a)Effect of image token keep ratio ρon reranking effectiveness
(Recall@1,3,5) and latency (LLM time in ms).
10 20 30 40
Number of Candidates (k)556065707580859095Recall (%)
Recall@1 Recall@3 Recall@5 LLM Time
100200300400500600700
LLM Time (ms)
(b)Effect of the number of input passages kon reranking effective-
ness (Recall@1,3,5) and latency (LLM time in ms).
Figure 6.Parameter studies on first-stage results from ColQwen.
Table 9.Comparison between random pruning and text-to-image (T2I) pruning under different visual-token keep ratios with DSE wikiss .
We report Recall@1/3/5on MMDocIR.
Random Pruning T2I PruningKeep RatioR@1 R@3 R@5 R@1 R@3 R@5
0.132.2 61.2 73.740.1 66.2 78.0
0.348.2 72.5 81.457.4 78.4 84.6
0.554.7 77.0 84.761.1 80.8 86.6
0.758.8 79.9 85.961.5 82.4 87.1
0.962.582.2 87.3 62.182.4 87.4
Recall@ 1/3/5 by9.2/5.9/3.2 points over random pruning. The gap becomes smaller as the keep ratio increases, since
most visual tokens are retained in both settings. These results confirm that query-aware pruning preserves more task-relevant
visual information than random token retention.
C.3. Correlation Between Pruning Scores and LLM Attention
The analysis above motivates text-to-image pruning as a cheap surrogate for attention-style pooling. We further provide
an empirical sanity check by measuring whether the pruning scores correlate with the LLM’s actual attention over visual
tokens.
For each visual tokenj, we use the pruning score
aj= max
1≤t≤N qst,j,
where st,jis the cosine similarity between query token tand visual token j. We then run the unpruned model and compute
the attention mass assigned to each visual token at the identifier-scoring position. Specifically, for layer ℓ, we average
attention over heads:
bℓ,j=1
HHX
h=1A(ℓ,h)
p,j,
where A(ℓ,h)
p,j denotes the attention from the scoring position pto visual token jin head h. We report the Spearman rank
correlation between{a j}and{b ℓ,j}.
We observe a moderate positive correlation, approximately 0.3, in mid-to-late LLM layers. This suggests that the proposed
pruning score captures meaningful query-relevant visual saliency, even though it is computed before the full multimodal
21

Very Efficient Listwise Multimodal Reranking for Long Documents
Table 10.Ranking quality and failure behavior on MMDocIR with the DSE wikiss first-stage retriever. Fail% denotes the percentage of
queries where the top-ranked page is incorrect. Near Miss denotes failures where the ground-truth page is ranked 2–3, and Catastrophic
Miss denotes failures where it is ranked lower than5.
Method P@1↑ nDCG@5↑ Mean Rank↓ Fail%↓ Near Miss Cat. Miss
DSE wiki−ss 50.6 65.6 3.60 49.4 52.0 36.3
MM-R5 73.177.9 2.8426.941.3 45.5
ZipRerank70.079.4 2.6230.057.8 30.7
ZipRerank −50% 68.9 78.2 2.71 31.1 52.9 32.6
Table 11.End-to-end efficiency on MMDocIR.
Method Vision ms Filter ms LLM ms Total ms TFLOPs/query Cached QPS Peak GPU (GB)
ZipRerank (Listwise)181.2 – 357.4 538.5 179.7 2.80 21.71
ZipRerank −50% (Listwise)180.2 4.5 269.4 454.1 84.9 3.65 20.05
MM-R5 (Listwise)873.2 – 3233.8 4107.0 263.2 0.31 23.04
LamRA (Listwise)352.7 – 529.3 881.9 368.2 1.89 28.31
DocReRank (Pointwise)401.1 – 737.8 1140.8 54.8 1.35 4.54
forward pass. The correlation is not expected to be perfect, since attention also reflects positional, formatting, and inter-
candidate interactions. Nevertheless, together with the random-pruning comparison in Appendix C.2, this result supports
that text-to-image pruning preserves more useful visual tokens than uninformed token retention.
C.4. Ranking Quality and Failure Behavior Beyond Recall@k
Recall@ konly measures whether the ground-truth page appears within the top- kresults, and does not fully capture the
quality of the produced ranking. We therefore provide additional ranking and failure analyses on MMDocIR with the DSE
first-stage retriever. As shown in Table 10, MM-R5 obtains the best P@ 1, while ZipRerank achieves better overall ranking
quality, with the highest nDCG@5and the lowest mean rank.
ZipRerank also exhibits fewer severe failures. Although its Fail% is slightly higher than MM-R5, its errors are more often
near misses at ranks 2–3, while MM-R5 has a higher fraction of catastrophic misses beyond rank 5. This suggests that
ZipRerank tends to place relevant evidence pages close to the top even when it misses rank1.
C.5. Efficiency Analysis
Table 11 reports end-to-end efficiency, including vision encoding, query-aware filtering, and LLM reranking. ZipRerank
takes 538.5 ms per query, compared with 4107.0 ms for MM-R5, achieving a 7.6× end-to-end speedup. With cached vision
embeddings, ZipRerank reaches2.80QPS, substantially higher than MM-R5’s0.31QPS.
The token-pruned variant further reduces total latency to 454.1 ms and lowers computation from 179.7 to84.9 TFLOPs/query,
with only 4.5ms filtering overhead. This shows that single-pass listwise scoring is the main source of latency reduction,
while query-aware pruning provides additional compute and memory savings.
22