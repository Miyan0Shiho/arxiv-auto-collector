# UNREAL: Unifying Retrieval and Long-Context with a Single Model

**Authors**: Edan Kinderman, Elad Hoffer, Yochai Blau, Brian Chmiel, Ron Banner, Daniel Soudry, Boris Ginsburg

**Published**: 2026-10-06 14:48:53

**PDF URL**: [https://arxiv.org/pdf/2610.08463v1](https://arxiv.org/pdf/2610.08463v1)

## Abstract
Long-context inference and Retrieval-Augmented Generation (RAG) handle evidence selection at vastly different scales, from a single long prompt to an entire corpus. We ask whether a single model-internal mechanism can select evidence across this range. We introduce UNifying REtrieval And Long-Context with a Single Model (UNREAL), a model-native evidence selection framework to span corpus retrieval and long-context inference. UNREAL encodes chunks and derives retrieval queries directly from the frozen LLM's internal representations. It adds fewer than 500K trainable parameters and leaves the backbone unchanged. On a 3B-token, 21M-chunk Wikipedia index, all four dense and hybrid UNREAL backbones outperform state-of-the-art retriever-reranker systems. The best model raises recall from 49.1% to 73.2% on HotpotQA, from 31.7% to 60.1% on 2WikiMultiHopQA, and from 8.8% to 14.4% on MuSiQue. Applied to long-context tasks, the same selection mechanism removes distractors before generation, raising NoLiMa accuracy from 1.0% to 24.83% at its maximum context length of 128K tokens, and LV-Eval's F1 score from 49.97% to 54.66% at 256K. UNREAL also reduces FLOPs and time-to-first-token relative to full-context inference from roughly 32K tokens onward, with larger gains as context grows. Together, these results establish model-internal evidence selection as a common foundation for corpus retrieval and evidence-sparse long-context inference.

## Full Text


<!-- PDF content starts -->

UNREAL: UNIFYINGRETRIEVAL ANDLONG-
CONTEXT WITH ASINGLEMODEL
Edan Kinderman1, Elad Hoffer1, Yochai Blau1, Brian Chmiel1, Ron Banner1,
Daniel Soudry1,2, Boris Ginsburg1
1NVIDIA2Technion
{ekinderman,ehoffer,yblau,bchmiel,rbanner,dsoudry,bginsburg}@nvidia.com
ABSTRACT
Long-context inference and Retrieval-Augmented Generation (RAG) handle evi-
dence selection at vastly different scales, from a single long prompt to an entire
corpus. We ask whether a single model-internal mechanism can select evidence
across this range. We introduceUNifyingREtrievalAndLong-Context with a
Single Model (UNREAL), a model-native evidence selection framework to span
corpus retrieval and long-context inference. UNREAL encodes chunks and de-
rives retrieval queries directly from the frozen LLM’s internal representations. It
adds fewer than 500K trainable parameters and leaves the backbone unchanged.
On a 3B-token, 21M-chunk Wikipedia index, all four dense and hybrid UNREAL
backbones outperform state-of-the-art retriever-reranker systems. The best model
raises recall from 49.1% to 73.2% on HotpotQA, from 31.7% to 60.1% on 2Wiki-
MultiHopQA, and from 8.8% to 14.4% on MuSiQue. Applied to long-context
tasks, the same selection mechanism removes distractors before generation, rais-
ing NoLiMa accuracy from 1.0% to 24.83% at its maximum context length of
128K tokens, and LV-Eval’s F1 score from 49.97% to 54.66% at 256K. UNREAL
also reduces FLOPs and time-to-first-token relative to full-context inference from
roughly 32K tokens onward, with larger gains as context grows. Together, these
results establish model-internal evidence selection as a common foundation for
corpus retrieval and evidence-sparse long-context inference.
8k32k128k 512k2M 8M32M100M
Context length405060Exact match (EM)
Nemotron-3.5-Lightning-30B-A3B
Long context RAG
8k32k128k 512k2M 8M32M100M
Context length354045505560
Qwen3.5-35B-A3B
Long context RAG
8k32k128k 512k2M 8M32M100M
Context length303540455055
Muse-Glimmer-30B
Long context RAG
Full context (all chunks)
Qwen3-Emb-0.6B + RerankerBM25
Qwen3-Emb-4B + RerankerBGE-large-en-v1.5 + Reranker
UNREAL (ours)
Figure 1: Exact match on question-answering as context length grows from 8K to 100M tokens.
Full context evaluation is shown only until computationally infeasible. For retrieval-based methods,
relevant context is first extracted from the input and then fed to the model to generate the answer.
Unlike baselines that rely on external models, UNREAL leverages the frozen LLM’s internal repre-
sentations to retrieve relevant context, maintaining the highest accuracy across long context lengths.
1
arXiv:2610.08463v1  [cs.CL]  6 Oct 2026

1 INTRODUCTION
Modern LLMs often process contexts where relevant information is sparsely distributed. To isolate
these key signals, evidence selection is done differently, depending on the scale. At long-context
scale, selection is implicit and internal: the model receives the full context and must suppress irrele-
vant content while generating. This avoids having a separate retriever, but is vulnerable to distractors
(Modarressi et al., 2025; Shi et al., 2023), and scaling it requires dedicated architectures, efficiency
mechanisms, and long-context training. At larger, corpus scale, this selection is explicit and ex-
ternal: a separate retriever selects passages before generation (Lewis et al., 2020). This creates a
two-model pipeline that must be trained, served, and kept aligned.
Despite being studied as alternative approaches (Xu et al., 2024; Li et al., 2024; 2025b; Lee et al.,
2024), long-context and RAG perform the same operation: query-conditioned selection of candidate
evidence. Only the scale differs, from hundreds of chunks in a prompt to millions in a corpus.
This suggests that the strengths of both approaches can be combined: make selection internal, so
evidence is ranked in the representation space of the model that will use it, and make selection
explicit, so distractors are removed during generation (Yu et al., 2024a). We therefore ask: can a
single pretrained LLM explicitly select evidence at both scales? An affirmative answer would make
retrieval and long-context inference two scales of one mechanism rather than competing systems.
1.1 MODEL-INTERNAL EVIDENCE SELECTION ACROSS SCALES
We answer this question withUNifyingREtrievalAndLong-Context with a Single Model (UN-
REAL), a model-native evidence selector that operates across both long-context and corpus scales.
We build on INTRA (Hoffer et al., 2026), which demonstrated intrinsic retrieval in encoder–decoder
models. However, INTRA relies on a separate encoder and cross-attention, which prevents its di-
rect application to today’s predominant decoder-only LLMs. UNREAL removes this restriction by
encoding candidate chunks and deriving retrieval queries directly from the frozen LLM’s internal
states. UNREAL adds fewer than 500K trainable parameters, trained with contrastive loss, on top
of the frozen LLM: a soft prompt that elicits a retrieval mode and layer weights that combine its
internal states for evidence ranking. Because it reads from the residual stream, the same mechanism
applies across dense-attention, linear-attention, and state-space architectures.
At the corpus scale, UNREAL retrieves from a 3B-token, 21M-chunk Wikipedia index. All four
tested backbones outperform the strongest dedicated retriever-reranker systems. The best raises
recall@10 from 49.1% to 73.2% on HotpotQA and from 31.7% to 60.1% on 2WikiMultiHopQA.
The retrieval-trained UNREAL module can also rank chunks within a long-context prompt and re-
turn only the selected text to the LLM. This explicit removal of distractors raises accuracy on differ-
ent long-context benchmarks: at the maximum evaluated length, NoLiMa accuracy rises from 1.0%
to 24.83% at 128K tokens, while LV-Eval’s F1 score rises from 49.97% to 54.66% at 256K. This
is achieved using the same LLM, without requiring an external retriever. The method also reduces
FLOPs and time-to-first-token versus full-context inference from roughly 32K tokens onward, with
larger gains as context grows. UNREAL thus turns corpus retrieval and long-context inference into
two scales of the same model-internal selection mechanism (Fig. 1).
1.2 CONTRIBUTIONS
1.Intrinsic corpus-scale retrieval with modern LLMs.UNREAL extends INTRA (Hoffer et al.,
2026) to decoder-only dense and hybrid LLMs, relying on a single frozen model to encode
candidates and extract retrieval queries. With fewer than 500K trainable parameters, all four
tested backbones outperform the strongest dedicated retriever–reranker systems on a 21M-chunk
Wikipedia index (Section 3).
2.Long-context selection with the same mechanism.UNREAL outperforms both full-context
inference and methods that rely on separate external retrievers across several long-context bench-
marks, including NoLiMa, HELMET, LV-Eval, and LOFT, while reducing FLOPs and time-to-
first-token at practical context lengths (Section 4).
3.One selector across six orders of magnitude.The same model-native module selects evidence
from an 8K-token context up to a 3B-token corpus, unifying long-context and retrieval reading
without architectural changes or backbone finetuning.
2

1 Encoding..
Layer 1Layer
Chunks
embeddings
LLM
Encoding
Retrieval 
tokens4Similarity
scoring
Learnable
Sum
....
..LayerRetrieve
chunks
Chunks
.. ..
Generation
Retrieve
chunksGenerationAnswer
Query Query
Retrieval Generation
25
7
6 3Figure 2: UNREAL trains retrieval tokensρand summation weightsα(fire indicates training), while
keeping the LLM frozen, enhancing its ability to identify relevant information in a context. Given
a long context, these trained variables enable UNREAL to select relevant chunks and pass them to
the same LLM for retrieval-augmented generation. Numbers indicate the order of operations.
2 METHOD
Standard retrieval-augmented generation (RAG) (Lewis et al., 2020) delegates evidence selection to
an external retriever, forcing production pipelines to maintain multiple models. In this work, we
unify this process within a single decoder-only LLM using its intrinsic retrieval capabilities. The
closest work to ours, INTRA (Hoffer et al., 2026), asks whether an encoder–decoder model can
instead retrieve directly from its own encoded representations. However, INTRA’s reliance on an
encoder–decoder architecture and cross-attention prevents its direct application to the more widely
adopted decoder-only LLMs, limiting the range of pretrained models it can use.
In this work, we seek the same intrinsic retrieval capability in decoder-only LLMs, including hybrid
architectures that interleave attention with other sequence mixers (Gu & Dao, 2024). Our method,
UNREAL, generalizes INTRA by replacing its encoder-derived chunk embeddingsk iand cross-
attention queriesq ℓwith representations available inside a decoder-only model. UNREAL consists
of three main components: chunk embeddings (Section 2.1), residual-state queries (Section 2.2),
and generation (Section 2.3). Fig. 2 provides an overview of the method.
2.1 CHUNK EMBEDDINGS
LetC={c i}M
i=1denote a corpus ofMchunks. Given a queryx, the goal is to retrieve the subset
S(x)⊆ {1, . . . , M}containing the evidence needed to answer it. INTRA encodes each chunk
once ask i= Enc(c i), whereEncdenotes the model encoder. Decoder-only LLMs lack a separate
encoder. UNREAL therefore uses the frozen LLM itself to encode each corpus chunk independently.
For a chunkc iof lengthT i, we select a single intermediate layerℓ cand extract its token-level
representations:
ki= LLM ℓc(ci)∈RTi×d,(1)
whereLLM ℓc(·)denotes the residual-stream states at layerℓ c, anddis the model’s hidden dimen-
sion. Thus, each chunk is represented by a sequence ofT itoken vectors.
Prior work has shown that intermediate layers often encode richer semantic representations than final
layers (Skean et al., 2025). Motivated by this observation, we evaluate representations from different
layers in our corpus-retrieval setting and selectℓ caccording to performance on a development set
(see the ablation in Table 4).
3

2.2 RESIDUAL STATE QUERIES
In order to perform query conditioned matching, UNREAL concatenatesRlearned retrieval tokens
{ρi∈Rd}R
i=1to the query token embeddings{x t∈Rd}Tq
t=1, whereT qis the query length. It also
augments the input with an initial contextC 0(x), which represents the chunks ranked highest for the
queryxby BM25 (Robertson & Zaragoza, 2009). Together, we get
xret=
C0(x), x 1, . . . , x Tq, ρ1, . . . , ρ R
,(2)
Then it reads the residual stream states at the retrieval token positions:
∀ℓ:q ℓ(xret) =
LLM ℓ(xret)
(|xret|−R+1):|x ret|∈RR×d.(3)
Placing the retrieval tokens last ensures that, under the causal mask, their states condition on both
the query and the initial context. These internal layer-wise query representationsq ℓare used to score
every chunkc iwith the late-interaction MaxSim operator (Khattab & Zaharia, 2020):
si(x;ρ, α) = MaxSimX
ℓαℓqℓ(xret), ki
,(4)
whereα ℓare learned layer-mixing coefficients andMaxSim(u, v)≜P
amax b⟨ua, vb⟩.
The retrieved chunk set is then selected by
SUNREAL (x) ={i∈ {1, . . . , M}:s i(x)is among the top-nscores}.(5)
Retrieval is therefore based on the internal representationsq ℓand internal chunk encodingsk iinstead
of an external index and model. Onlyρandαare added as learned parameters, the LLM remains
frozen.
Because our read-out operates on the residual stream, it can be applied uniformly across architectures
that use softmax attention, linear attention, or state-space layers, including hybrids that combine
these layer types. Aggregating read-outs across these layer types is further motivated by the causal
attention interpretation of selective SSMs (Ali et al., 2025; Jiang et al., 2026).
Retrieval training.The retrieval loss is a multi-positive InfoNCE objective (Oord et al., 2018; Chen
et al., 2020) that contrasts oracle chunks with sampled alternatives. For a queryx, letO(x)contain
the oracle indices, corresponding to chunks that contain the annotated evidence needed to answer
the query. LetN(x)contain the hard negatives chunks. OverB(x) =O(x)∪ N(x), we minimize
Lretrieval =−1
|O(x)|X
j∈O(x)logexp(s j(x)/τ)P
i∈B(x)exp(s i(x)/τ),(6)
whereτis the temperature. Training updates only the retrieval tokensρ iand layer-mixing coeffi-
cientsα ℓ, while the LLM parameters remain frozen. We then apply the resulting selector to retrieval
tasks in Section 3 and long-context benchmarks in Section 4.
Compression.To reduce corpus-scale storage, we partition the full token-level chunk representa-
tionsk i∈RTi×dintoL pcontiguous groups and mean-pool each group, compressing chunks from
Ti×dtoL p×d. Similarly, we mean-pool theRretrieval-token statesq ℓacrossGgroups, com-
pressing queries fromR×dtoG×d. The parametersL pandGtrade off retrieval accuracy against
index size and MaxSim compute cost.
2.3 SCORING AND GENERATION
Following Eq. (5),S UNREAL (x)contains the indices of thenhighest-scoring chunks. To provide
the model with this relevant context, the selected chunksC UNREAL (x) = [c i:i∈ S UNREAL (x)]
are concatenated with the query in the LLM prompt:
y= LLM([C UNREAL (x), x]).(7)
Unlike INTRA, which passes the selected encoder memories through cross-attention, UNREAL re-
encodes the selected chunks during generation. Inference thus uses one LLM forward pass to form
the retrieval queriesq ℓ, and a second ordinary generation pass over the selected chunks.
In the following sections, we present how UNREAL can be used for both corpus retrieval and long-
context evidence selection. We then discuss the connection between retrieval and long-context tasks.
4

020406080Complete-Evidence
recall@10
25.934.6
27.634.644.648.8 49.169.1 70.573.2 72.9HotpotQA
0204060
20.717.524.6 25.929.631.7 30.357.1 55.360.1 58.92WikiMultiHopQA
051015
3.04.34.96.07.18.6 8.812.6 12.714.413.8MuSiQue
02040Complete-Evidence
recall@10
17.7 16.3 14.718.325.4 24.528.140.142.447.4 46.3HoVer
010203040
15.9
12.8 13.215.819.222.2 22.228.3 27.432.2
28.5IIRC
0204060
36.0
28.934.440.943.147.4 47.551.956.153.4 54.4SQuAD v2
0255075100Complete-Evidence
recall@10
58.568.171.976.7 76.8 78.6 78.9 81.6 84.5 85.8 86.5FEVER
0102030
8.924.6
19.125.924.426.0 26.724.928.930.1 30.6Natural Questions
02040
23.325.9 26.330.533.836.0 36.545.747.249.6 49.0Average across 8 datasets
BM25
Hybrid RAG
UNREAL-Qwen3.5-4BBGE-large-en-v1.5
LateOn (LI) + reranker
UNREAL-Qwen3.5-35B-A3BQwen3-Emb-0.6B
Qwen3-Emb-4B + reranker
UNREAL-Muse-Glimmer-30BQwen3-Emb-4B
 
UNREAL-Nemo3.5-Lightning-30B-A3BSingle-hop dataset Multi-hop datasetFigure 3:UNREAL scales to full-corpus Wikipedia retrieval.Complete-evidence recall@10
on 21M Wiki-2018 chunks across eight datasets; an example is correct only when all annotated evi-
dence chunks are retrieved. Across four architectures, UNREAL consistently outperforms baselines,
especially on multi-hop datasets. Bars show means with 95% BCa confidence intervals.
3 UNREALAS ARETRIEVALMECHANISM
Our experiments evaluate whether UNREAL remains effective at a scale requiring genuine retrieval
over the 3B-token, 21M-chunk Wiki-2018 corpus (Karpukhin et al., 2020a), with a mean chunk
length of 140 tokens. We report complete-evidence recall@n: the fraction of examples withall
annotated oracle chunks retrieved in the top-nresults. This is particularly demanding for multi-hop
tasks, where partial evidence receives no credit. Average recall@nappears in Figs. 9 and 10.
Benchmarks and baselines.The evaluation spans HotpotQA, 2WikiMultiHopQA, MuSiQue,
HoVer, IIRC, SQuAD v2, FEVER, and Natural Questions (Yang et al., 2018; Ho et al., 2020;
Trivedi et al., 2022; Jiang et al., 2020; Ferguson et al., 2020; Rajpurkar et al., 2018; Thorne et al.,
2018; Kwiatkowski et al., 2019). We map oracle chunks to the shared Wiki-2018 corpus using a
KILT-style procedure (Petroni et al., 2021). We compare against BM25, Qwen3-Embedding and
BGE dense retrievers, the Jina cross-encoder reranker, LightOn multi-vector (late-interaction) re-
triever, and hybrid RAG based on RRF (Robertson & Zaragoza, 2009; Zhang et al., 2025; Xiao
et al., 2024b; Jina AI, 2024; Sourty et al., 2026; Cormack et al., 2009). See Appendix B for the full
details.
3.1 RESULTS
We demonstrate UNREAL’s architectural generality across four backbones: the dense models
Qwen3.5-4B and Muse-Glimmer-30B, the linear-attention hybrid Qwen3.5-35B-A3B, and the
Mamba–attention hybrid Nemotron-3.5-Lightning-30B-A3B (Meta Superintelligence Lab, 2026;
Qwen Team, 2026; NVIDIA, 2026).
Full corpus evaluation.Fig. 3 reports complete-evidence recall@10. Each UNREAL backbone
leads on most of the eight datasets. The largest gains occur on HotpotQA, 2WikiMultiHopQA,
MuSiQue, and HoVer, challenging multi-hop tasks requiring evidence from multiple articles. A
similar pattern appears for recall@20 in Fig. 8. For ablation studies please refer to Table 3, Table 4
and Table 5.
Generation evaluation.Table 1 reports end-to-end question-answering performance with
Nemotron-3.5-Lightning fixed as the generator and only the retriever providing its top-5chunk con-
text varied. We also report an oracle upper bound using only oracle chunks and a no-context lower
bound. UNREAL attains the highest EM and F1 on all three datasets while using the same LLM for
retrieval and generation. Experiments with Qwen3.5-35B-A3B as the generator appear in Table 2.
5

Table 1: Exact match (EM) and macro-averaged F1 on three multi-hop QA datasets, generated by
Nemotron-3.5-Lightning under each retrieval method’s top-5 retrieved context. UNREAL achieves
the highest EM and F1 scores across all three datasets while relying on a single LLM for both
retrieval and generation.
HotpotQA 2Wiki MuSiQue
Method EM F1 EM F1 EM F1
No context 28.5 37.7 29.9 34.6 6.8 16.6
Oracle 64.3 78.9 57.5 66.8 47.5 58.2
BM25 39.1 49.7 29.1 35.5 6.4 14.6
BGE-large-en-v1.5 39.6 51.2 28.5 35.6 8.6 17.6
Qwen3-Emb-0.6B 37.9 48.5 28.2 35.5 10.1 18.1
Qwen3-Emb-4B 39.2 51.4 31.6 38.8 10.5 18.1
Hybrid RAG 41.5 53.1 32.5 39.8 10.0 18.9
LateOn + reranker 43.3 55.2 30.8 38.7 11.4 20.1
Qwen3-Emb-4B + reranker 43.3 55.3 31.4 39.1 13.2 22.6
UNREAL-Nemo3.5-Lightning52.6 65.8 42.0 50.1 16.9 25.4
Overall, UNREAL turns a frozen LLM into a full-corpus retriever by training only the retrieval-
token embeddingsρand layer-mixing weightsα: fewer than 0.5M added parameters in total, under
0.005% of each tested backbone’s parameter count.
4 UNREALAS ALONGCONTEXTMECHANISM
In this section, we investigate whether UNREAL’s retrieval mechanism can tackle long-context tasks
by treating the full prompt as a collection of chunks to retrieve from. Many tasks framed as long-
context understanding are, in practice, sparse-evidence problems: only a small number of chunks
contain the information needed to answer the query, while the remaining context is irrelevant. This
structure closely resembles retrieval tasks such as multi-hop question answering (Ho et al., 2020).
In this setting, the answer quality depends strongly on retrieving the required evidence. Even when
evidence is retrieved, distractors may interfere with generation. Thus, expanding the context window
ensures evidence is available, but not that the model can locate and use it effectively. While prior
works (Xu et al., 2024; Li et al., 2024; 2025b) have identified the connection between retrieval and
long context data, these approaches rely on an external retriever to select the relevant context. In
contrast, UNREAL is the first to use the frozen LLM’s internal representations to select evidence
and generate the answer.
4.1 RESULTS
Given a long input as context, we treat it as a retrieval problem, i.e., we first split it intoMchunks
with a mean chunk length of 140 tokens and use the frozen LLM to encode each chunk separately
(Eq. (1)). Then, UNREAL ranks the chunks against the input query (Eqs. (3) and (4)), selects the
top-n(Eq. (5)), and gives their unchanged text to the same frozen LLM for answering (Eq. (7)).
Full experimental details are provided in Appendix B. Similarly, we include other retrieval base-
lines, such as BM25 and embedding models paired with rerankers. We wish to emphasize that the
UNREAL models used in this section are the same ones presented in Section 3, i.e., no special
long-context training data was involved.
To evaluate evidence selection at lengths far beyond those of existing long-context benchmarks, we
follow the methodology of LOFT (Lee et al., 2024) and construct an QA benchmark spanning 8K to
100M tokens. We draw queries and their corresponding oracle chunks from QA benchmarks and pad
them with random distractor passages from Wiki-2018. As shown in Fig. 1, full-context accuracy
drops sharply as context length grows, whereas retrieval-based approaches perform significantly
better. UNREAL outperforms all baselines at large scales and enables generation using evidence
selected from 100M tokens, far beyond the context limits of current LLMs. For results on standard
LOFT see Fig. 12.
6

4k 8k 16k 32k 64k 128k
Context length (tokens)020406080100Answer accuracy
NoLiMa
16k 32k 64k 128k 256k
Context length (words)50525456LV-Eval F1
LV-Eval
Oracle (gold chunks)
BGE-large-en-v1.5 + RerankerFull context (all chunks)
Qwen3-Emb-4B + RerankerBM25
UNREAL (ours)Figure 4:Left:NoLiMa accuracy across context lengths (4K–128K tokens) using Nemotron-3-
Nano.Right:LV-Eval F1 across context lengths (16K–256K words, averaged over LooGLE-SD and
MultiFieldQA-en) using Nemotron-3.5-Lightning. In both cases, full-context performance degrades
as context grows, whereas UNREAL achieves the highest performance.
020406080Substring EM55.5
45.456.0 55.160.2Nemotron-3.5-Lightning-30B-A3B
020406080
67.2
54.266.0 65.970.3Qwen3.5-35B-A3B
Full context (all chunks)
Qwen3-Emb-4B + RerankerBM25
UNREAL (ours)BGE-large-en-v1.5 + Reranker
Figure 5: Substring exact-match on the RAG subset of the HELMET long context benchmark, with
Nemotron-3.5-Lightning-30B-A3B and Qwen3.5-35B-A3B as generators. UNREAL outperforms
full-context inference and other retrieval methods for both generators.
Fig. 4 presents results on NoLiMa (Modarressi et al., 2025), a needle-in-a-haystack benchmark
where queries and target needles share no direct keyword overlap, alongside English QA subsets
from LV-Eval (Yuan et al., 2024). As context lengths scale up to 128K tokens in NoLiMa and 256K
words in LV-Eval, full-context performance degrades sharply. In contrast, retrieval baselines remain
far more stable, with UNREAL consistently achieving the highest accuracy across all scales.
Fig. 5 shows results on the RAG subset of the HELMET QA long-context benchmark (Yen et al.,
2024). The tasks use context lengths of 8K–128K tokens, and the reported scores are averaged
across these lengths. Full-context inference remains a strong baseline, but UNREAL achieves the
highest exact-match score for both generators.
Top-ntrade-off.In Fig. 6 (left) we examine how the number of selected chunks,n, affects perfor-
mance on the long-context NoLiMa benchmark. Accuracy follows an inverted-U pattern: increasing
ninitially improves performance by raising evidence recall, but eventually reduces performance as
additional distractors interfere with generation. Similar non-monotonic trade-offs have been re-
ported in prior work (Yu et al., 2024b; Jin et al., 2024). UNREAL demonstrates that the same
pattern arises when a single decoder-only LLM performs both retrieval and generation.
Intrinsic retrieval.To evaluate whether a frozen LLM can perform retrieval, we test Nemotron-3-
Nano on NoLiMa by segmenting the text into chunks. Within every attention layer, we mean-pool
the key embeddings to yield a single key vector per chunk. We similarly average the query embed-
dings over the question tokens and compute a dot product between the pooled key and query vectors.
This process yields a per-chunk similarity score, allowing us to compute recall. Fig. 6 (right) shows
that these “intrinsic” untrained layer representations retrieve evidence above the random baseline.
This finding motivates UNREAL, which trains retrieval tokensρto strengthen the retrieval signal
already present in the frozen model. For full details, see Appendix A.2.1.
7

1 2 510 20 50M
Number of retrieved chunks n2030405060Accuracy (%)
Accuracy (%)
Average recall@n1 2 5 10 20 50
Number of retrieved chunks n0.00.10.20.30.40.50.60.70.80.9Average recall@n
Intrinsic, per layer, not trained
Intrinsic, best layer (L5), not trainedUNREAL, trained
Random0.20.40.60.81.0
Average recall@n
Figure 6: Recall and accuracy on the NoLiMa benchmark for Nemotron-3-Nano.Left:The effect
of the number of selected chunksnon UNREAL’s accuracy and recall.Right:Embeddings from
the frozen LLM outperform random retrieval, motivating UNREAL, which is trained to enhance this
capability.
4.2 EFFICIENCY
Does UNREAL’s additional retrieval pass make it less efficient than full-context inference, or do
its savings dominate at sufficiently long contexts? Although UNREAL uses separate retrieval and
generation passes, both reduce context-dependent computation. During retrieval, it encodes chunks
independently, avoiding most inter-chunk attention in the prefill. During generation, it attends only
to the selected chunks, greatly reducing both context attention and the KV-cache size. For fixed
chunk size and selection budget, UNREAL therefore scales linearly with context length, whereas
with standard attention full-context inference scales quadratically.
We determine where these savings outweigh the extra pass in both FLOPs and wall-clock time.
Appendix C gives the full FLOP difference in equation 8. Applying this to the three tested backbones
in our experimental setting, Table 6 shows the UNREAL uses less FLOPs for context lengths over
18K tokens. We also measure stage-wise time-to-first-token with vLLM on a single H100. Figure 7
shows the resulting speedup: it grows with context length across all backbones, with dashed curves
projecting the full-context runtime beyond its last feasible measurement using the FLOP model.
As can be seen, UNREAL reduces time-to-first-token at commonly used context lengths, roughly
32K tokens onward, with substantial speedups at longer contexts. Full methodology is provided in
Appendix D.
5 RELATEDWORK
Retrieval models.ColBERT and ColBERTv2 score passages by MaxSim over token representa-
tions (Khattab & Zaharia, 2020; Santhanam et al., 2022). LLM retrievers contrastively train gener-
ators for retrieval (Wang et al., 2024; Ma et al., 2024; Zhang et al., 2025). UNREAL is also con-
trastively trained, but the chunk encoder and reader remain a frozen generative decoder. REALM,
RAG, and related methods jointly train distinct retriever and reader components (Guu et al., 2020;
Lewis et al., 2020; Izacard et al., 2023). We instead share the frozen backbone representation across
retrieval and generation, while retaining a separately trained query head. INTRA (Hoffer et al.,
2026) is the closest predecessor: it introduced retrieval from an encoder–decoder model’s inter-
nal representations. We adapt the method to standard decoder-only LLMs and use the selector for
long-context tasks.
Efficient attention.Landmark Attention, InfLLM, Quest, MoBA, NSA, and HiLS-Attention select
blocks of tokens within an LLM’s input and perform attention only on those blocks (Mohtashami
& Jaggi, 2023; Xiao et al., 2024a; Tang et al., 2024; Lu et al., 2025; Yuan et al., 2025; Hu et al.,
2026) to improve efficiency. Sparse attention fixes or learns a restricted set of positions (Beltagy
et al., 2020; Zaheer et al., 2021; Kitaev et al., 2020; Xiao et al., 2023; Xu et al., 2026); block-level
8

10K 100K 1M 10M 100M
Context length L (tokens)100101102103Speedup  =  baseline(L) / UNREAL(L)
T = 128, n = 10UNREAL speedup over full-context baseline
Muse-Glimmer-30B
Nemotron-3.5-Lightning-30B-A3B
Qwen3.5-35B-A3B
Dots + solid: directly measured baseline. Dashed: FLOPs-scaled projection beyond the last measured length.Figure 7: Time-to-first-token speedup of UNREAL over full-context inference. Solid curves use
directly measured full-context runtimes; dashed curves use FLOP-scaled projections beyond the last
measured context length.
methods summarize and rank KV blocks (Mohtashami & Jaggi, 2023; Xiao et al., 2024a; Tang et al.,
2024; Lu et al., 2025; Yuan et al., 2025; Xu et al., 2025); and hierarchical methods route attention
through chunk summaries (Hu et al., 2026). UNREAL instead uses a retrieval-trained mechanism
built on the frozen decoder’s representations to select chunks before generation, so the LLM receives
only the selected text for generation.
RAG and long context.Prior work compares retrieval pipelines with full-context models (Xu et al.,
2024; Li et al., 2024; 2025b) or asks whether long context subsumes retrieval (Lee et al., 2024).
UNREAL differs from both approaches. Unlike retrieval pipelines, it requires no external retrieval
model, avoiding the need to develop and maintain multiple models in production. Unlike full-context
baselines, UNREAL is trained on retrieval tasks and removes unselected chunks before generation,
preventing them from distracting the generator (Shi et al., 2023; Yu et al., 2024a).
6 LIMITATIONS
Our training data originated from Wikipedia QA, so transfer to heterogeneous domains and lan-
guages remains for future work. For now, our long-context experiments focus on sparse-evidence
tasks. UNREAL storesL pvectors per chunk, costing more to build and store than a single-vector
dense index, and relies on a cheap BM25 initial context. We generate answers with general-purpose
LLMs rather than task-specific span extractors such as SpanBERT (Joshi et al., 2020), which are bet-
ter suited to extractive QA and achieve higher exact match scores. While applying retrieval to long-
context tasks substantially reduces context length and improves generation efficiency, it introduces
additional steps for chunk encoding and search; these extra steps are inherent to any retrieval-based
approach, not unique to UNREAL.
Multi-pass agentic RAG systems interleave reasoning with repeated retrieval (Li et al., 2025a; Asai
et al., 2024). UNREAL instead focuses on a single-pass retrieval mechanism that could serve as
a component within future agentic pipelines. Because these systems evaluate an entire iterative
reasoning-and-retrieval pipeline, whereas UNREAL isolates a single retrieval stage, they are not
directly comparable and are therefore excluded from our baselines.
7 DISCUSSION ANDFUTUREDIRECTIONS
This work asks whether corpus retrieval and long-context inference can share a single evidence-
selection mechanism inside a decoder-only LLM. UNREAL answers this question by using the
LLM’s representations to encode and rank candidate chunks, with fewer than 500K trainable pa-
rameters while keeping the backbone frozen. As shown, UNREAL outperformed SOTA retrieval
9

methods on both retrieval and long-context benchmarks. These findings support a unified view of
corpus retrieval and long-context inference as the same evidence-selection problem operating at dif-
ferent scales, and provide a path toward LLMs that retrieve and use relevant information without
relying on a separate retrieval model.
7.1 FUTURE DIRECTIONS
We close by discussing what our main finding, that retrieval supervision improves a model’s evi-
dence selection, implies for training and evaluating context-scalable models.
Train the model to recall.UNREAL makes the model’s internal retrieval ability trainable. Retrieval
supervision teaches the model’s representations to rank relevant chunks, improving recall rather than
relying on this ability to emerge from pretraining alone. As shown, the learned retriever improves
both corpus-scale retrieval and in-prompt long-context selection.
An alternative path to context scalability.These results motivate training in-model retrieval to-
gether with generation, rather than treating a larger context window as sufficient. Under this view,
sparse-evidence long-context tasks should be formulated primarily as retrieval problems during both
training and evaluation. Models should learn to identify the evidence they need before generating,
and evaluations should report evidence recall alongside answer accuracy.
Benchmarks should separate evidence regimes.This perspective does not reduce all long-context
understanding to retrieval. New benchmarks should annotate the evidence required for each an-
swer and distinguish sparse-evidence tasks from evidence-dense problems that require integrating
information across a large fraction of the input and cannot be solved by selecting a few chunks.
10

REFERENCES
Ameen Ali, Itamar Zimerman, and Lior Wolf. The hidden attention of Mamba models. InPro-
ceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), pp. 1516–1534, 2025. doi: 10.18653/v1/2025.acl-long.76. URLhttps:
//aclanthology.org/2025.acl-long.76/.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection. InInternational Conference on Learning
Representations (ICLR), 2024.
Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The Long-Document Transformer,
2020. URLhttps://arxiv.org/abs/2004.05150.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. InInternational conference on machine learning,
pp. 1597–1607. PmLR, 2020.
Gordon V . Cormack, Charles L. A. Clarke, and Stefan B ¨uttcher. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. InProceedings of the 32nd International ACM
SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759. ACM,
2009. doi: 10.1145/1571941.1572114.
James Ferguson, Matt Gardner, Hannaneh Hajishirzi, Tushar Khot, and Pradeep Dasigi. IIRC: A
dataset of incomplete information reading comprehension questions. InProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing, pp. 1137–1147, 2020. doi:
10.18653/v1/2020.emnlp-main.86.
Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In
First Conference on Language Modeling (COLM), 2024. URLhttps://arxiv.org/abs/
2312.00752.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-
augmented language model pre-training. InProceedings of the 37th International Conference on
Machine Learning, ICML’20, 2020.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-
hop QA dataset for comprehensive evaluation of reasoning steps. InProceedings of the 28th
International Conference on Computational Linguistics, pp. 6609–6625, 2020.
Elad Hoffer, Yochai Blau, Edan Kinderman, Ron Banner, Daniel Soudry, and Boris Ginsburg. Re-
trieval from within: An intrinsic capability of attention-based models, 2026. URLhttps:
//arxiv.org/abs/2605.05806.
Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui
Han, Yushi Bai, Kewei Tu, Haitao Mi, and Leo Liang. Hierarchical sparse attention done right:
Toward infinite context modeling, 2026. URLhttps://arxiv.org/abs/2607.02980.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning
with retrieval augmented language models.Journal of Machine Learning Research, 24(251):
1–43, 2023. URLhttp://jmlr.org/papers/v24/23-0037.html.
Jindong Jiang, Amala Sanjay Deshmukh, Kateryna Chumachenko, Karan Sapra, Zhiding Yu, Guilin
Liu, Andrew Tao, Pavlo Molchanov, Jan Kautz, and Wonmin Byeon. Stateful token reduction for
long-video hybrid VLMs, 2026. URLhttps://arxiv.org/abs/2603.00198.
Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal.
HoVer: A dataset for many-hop fact extraction and claim verification. InFindings of the Asso-
ciation for Computational Linguistics: EMNLP 2020, pp. 3441–3460, 2020. doi: 10.18653/v1/
2020.findings-emnlp.309.
11

Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan ¨O. Arik. Long-context llms meet rag: Over-
coming challenges for long inputs in rag.ArXiv, abs/2410.05983, 2024. URLhttps:
//api.semanticscholar.org/CorpusID:273229050.
Jina AI. Jina Reranker v2 Base Multilingual, 2024. URLhttps://jina.ai/models/
jina-reranker-v2-base-multilingual/. Released June 25, 2024.
Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zettlemoyer, and Omer Levy. Span-
bert: Improving pre-training by representing and predicting spans.Transactions of the association
for computational linguistics, 8:64–77, 2020.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In
Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pp. 6769–6781. Association for Computational Linguistics, 2020a. doi: 10.18653/
v1/2020.emnlp-main.550.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. InPro-
ceedings of the 2020 conference on empirical methods in natural language processing (EMNLP),
pp. 6769–6781, 2020b.
Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search via contex-
tualized late interaction over BERT. InProceedings of the 43rd International ACM SIGIR
Conference on Research and Development in Information Retrieval, pp. 39–48, 2020. doi:
10.1145/3397271.3401075.
Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In
International Conference on Learning Representations, 2020. URLhttps://openreview.
net/forum?id=rkgNKkHtvB.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion
Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: A benchmark for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl a00276.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. InProceedings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles, 2023.
Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua, Devendra Singh Sachan, Michael Boratko,
Yi Luan, S ´ebastien MR Arnold, Vincent Perot, Siddharth Dalmia, et al. Can long-context lan-
guage models subsume retrieval, rag, sql, and more?arXiv preprint arXiv:2406.13121, 2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems, volume 33, pp. 9459–9474. Curran Associates, Inc., 2020.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models, 2025a. URL
https://arxiv.org/abs/2501.05366.
Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. Long context vs. RAG for LLMs: An evaluation
and revisits, 2025b. URLhttps://arxiv.org/abs/2501.01880.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval aug-
mented generation or long-context LLMs? A comprehensive study and hybrid approach. In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing:
Industry Track, 2024. URLhttps://arxiv.org/abs/2407.16833.
12

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He,
Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun Lai, Yanru
Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Neo Y . Zhang, Zhilin Yang, Xinyu Zhou,
Mingxing Zhang, and Jiezhong Qiu. MoBA: Mixture of block attention for long-context LLMs,
2025. URLhttps://arxiv.org/abs/2502.13189.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. Fine-tuning LLaMA for multi-
stage text retrieval. InProceedings of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval, pp. 2421–2425, 2024. doi: 10.1145/3626772.
3657951.
Meta Superintelligence Lab. Muse Glimmer Model Card.https://huggingface.co/
meta-models/Muse-Glimmer-30B, 2026.
Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A. Rossi, Seunghyun
Yoon, and Hinrich Sch ¨utze. NoLiMa: Long-context evaluation beyond literal matching, 2025.
URLhttps://arxiv.org/abs/2502.05167.
Amirkeivan Mohtashami and Martin Jaggi. Random-access infinite context length for transformers.
InAdvances in Neural Information Processing Systems (NeurIPS), 2023. Landmark Attention.
NVIDIA. NVIDIA Nemotron 3.5 Lightning 30B-A3B model card.https://huggingface.
co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, 2026.
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predic-
tive coding.arXiv preprint arXiv:1807.03748, 2018.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James
Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, et al. Kilt: a benchmark for knowl-
edge intensive language tasks. InProceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies, pp.
2523–2544, 2021.
Qwen Team. Qwen3.5: Towards native multimodal agents.https://qwen.ai/blog?id=
qwen3.5, 2026.
Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable questions
for SQuAD. InProceedings of the 56th Annual Meeting of the Association for Computational
Linguistics, pp. 784–789, 2018. doi: 10.18653/v1/P18-2124.
Stephen Robertson and Hugo Zaragoza. The Probabilistic Relevance Framework: BM25 and Be-
yond.Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/
1500000019.
Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Col-
BERTv2: Effective and efficient retrieval via lightweight late interaction. InProceedings of the
2022 Conference of the North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies, pp. 3715–3734, 2022. doi: 10.18653/v1/2022.naacl-main.
272.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Sch ¨arli,
and Denny Zhou. Large language models can be easily distracted by irrelevant context, 2023.
URLhttps://arxiv.org/abs/2302.00093.
Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Patel, Jalal Naghiyev, Yann LeCun, and
Ravid Shwartz-Ziv. Layer by layer: Uncovering hidden representations in language mod-
els.ArXiv, abs/2502.02013, 2025. URLhttps://api.semanticscholar.org/
CorpusID:276107264.
Rapha ¨el Sourty, Antoine Chaffin, Paulo Roberto Moura Junior, and Am ´elie Chatelain. Denseon
with the lateon: Fully open dense and late-interaction models for multilingual, long-context, and
code search.arXiv preprint arXiv:2607.27178, 2026.
13

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest:
Query-aware sparsity for efficient long-context LLM inference. InProceedings of the 41st Inter-
national Conference on Machine Learning (ICML), 2024. URLhttps://arxiv.org/abs/
2406.10774.
James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. FEVER: A large-
scale dataset for fact extraction and verification. InProceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics, pp. 809–819, 2018.
doi: 10.18653/v1/N18-1074.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multi-
hop questions via single-hop question composition.Transactions of the Association for Compu-
tational Linguistics, 10:539–554, 2022. doi: 10.1162/tacl a00475.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improv-
ing text embeddings with large language models. InProceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11897–11916, 2024.
doi: 10.18653/v1/2024.acl-long.642.
Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan
Liu, and Maosong Sun. InfLLM: Training-free long-context extrapolation for LLMs with an
efficient context memory. InAdvances in Neural Information Processing Systems (NeurIPS),
2024a.
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming lan-
guage models with attention sinks, 2023. URLhttps://arxiv.org/abs/2309.17453.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. C-pack:
Packed resources for general chinese embeddings. InProceedings of the 47th international ACM
SIGIR conference on research and development in information retrieval, pp. 641–649, 2024b.
Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang,
Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-
token context intelligence.arXiv preprint arXiv:2606.19348, 2026.
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian,
Evelina Bakhturina, Mohammad Shoeybi, and Bryan Catanzaro. Retrieval meets long context
large language models. InThe Twelfth International Conference on Learning Representations
(ICLR), 2024. URLhttps://arxiv.org/abs/2310.03025.
Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. XAttention: Block sparse
attention with antidiagonal scoring, 2025. URLhttps://arxiv.org/abs/2503.16428.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural Language
Processing, pp. 2369–2380, 2018. doi: 10.18653/v1/D18-1259.
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat,
and Danqi Chen. HELMET: How to evaluate long-context language models effectively and thor-
oughly, 2024. URLhttps://arxiv.org/abs/2410.02694.
Tan Yu, Anbang Xu, and Rama Akkiraju. In defense of RAG in the era of long-context language
models, 2024a. URLhttps://arxiv.org/abs/2409.01666.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and
Bryan Catanzaro. RankRAG: Unifying context ranking with retrieval-augmented generation in
LLMs. InAdvances in Neural Information Processing Systems, 2024b. URLhttps://arxiv.
org/abs/2407.02485.
Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie,
Y . X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang,
and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse
attention. InProceedings of the 63rd Annual Meeting of the Association for Computational Lin-
guistics (ACL), 2025. URLhttps://arxiv.org/abs/2502.11089.
14

Tao Yuan, Xuefei Ning, Dong Zhou, Zhijie Yang, Shiyao Li, Minghui Zhuang, Zheyue Tan, Zhuyu
Yao, Dahua Lin, Boxun Li, et al. Lv-eval: A balanced long-context benchmark with 5 length
levels up to 256k.arXiv preprint arXiv:2402.05136, 2024.
Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon,
Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers
for longer sequences, 2021. URLhttps://arxiv.org/abs/2007.14062.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding:
Advancing text embedding and reranking through foundation models, 2025. URLhttps:
//arxiv.org/abs/2506.05176.
15

A ADDITIONAL EXPERIMENTS
A.1 CORPUS-SCALE RETRIEVAL RESULTS
Figs. 8 to 10 report Complete-evidence recall@20, average recall@10, and average recall@20, re-
spectively, across 8 datasets using the full Wiki-2018 corpus, demonstrating UNREAL’s superiority
over SOTA embedding models, rerankers, and late-interaction methods.
020406080Complete-Evidence
recall@20
33.644.0
34.041.853.8 52.6 53.074.3 75.8 77.5 77.5HotpotQA
0204060
25.621.528.9 31.034.9 35.3 33.961.0 60.364.3 63.62WikiMultiHopQA
05101520
4.05.66.68.59.710.7 10.915.716.618.517.8MuSiQue
0204060Complete-Evidence
recall@20
23.2 22.719.323.632.8
27.232.447.550.154.3 53.6HoVer
010203040
19.517.616.120.324.8 25.5 25.733.5 33.337.835.1IIRC
0204060
41.9
36.140.848.150.9 52.0 52.557.662.259.4 60.6SQuAD v2
0255075100Complete-Evidence
recall@20
64.073.677.781.5 81.6 83.0 83.4 85.1 87.6 88.6 89.3FEVER
010203040
11.930.1
23.931.5 31.4 31.533.131.535.8 37.1 37.4Natural Questions
0204060
28.031.4 30.935.840.0 39.7 40.650.852.754.7 54.4Average across 8 datasets
BM25
Hybrid RAG
UNREAL-Qwen3.5-4BBGE-large-en-v1.5
LateOn (LI) + reranker
UNREAL-Qwen3.5-35B-A3BQwen3-Emb-0.6B
Qwen3-Emb-4B + reranker
UNREAL-Muse-Glimmer-30BQwen3-Emb-4B
 
UNREAL-Nemo3.5-Lightning-30B-A3BSingle-hop dataset Multi-hop dataset
Figure 8:Complete-evidence recall@20on 21M Wiki-2018 chunks across 8 datasets; an example
is correct only when all annotated evidence chunks are retrieved. Bars show means with 95% BCa
confidence intervals.
020406080Average recall@10
50.254.9 52.557.865.5 68.0 68.378.3 79.8 81.2 81.3HotpotQA
020406080
43.639.550.5 51.555.6 57.2 56.071.0 69.072.3 71.52WikiMultiHopQA
010203040
18.617.322.425.727.431.2 30.4 31.1 31.934.432.7MuSiQue
020406080Average recall@10
47.945.1 45.449.557.6 57.561.868.7 68.973.0 71.7HoVer
02040
 32.5
26.129.432.737.841.1 40.5 41.5 41.645.443.2IIRC
0204060
36.0
28.934.440.943.147.4 47.551.956.153.4 54.4SQuAD v2
0255075100Average recall@10
62.673.277.182.0 82.1 84.0 84.1 86.5 89.1 90.1 90.6FEVER
02040
22.444.1
36.946.4 45.9 47.0 47.944.549.0 50.0 50.6Natural Questions
0204060
39.2 41.143.648.351.954.2 54.659.2 60.7 62.5 62.0Average across 8 datasets
BM25
Hybrid RAG
UNREAL-Qwen3.5-4BBGE-large-en-v1.5
LateOn (LI) + reranker
UNREAL-Qwen3.5-35B-A3BQwen3-Emb-0.6B
Qwen3-Emb-4B + reranker
UNREAL-Muse-Glimmer-30BQwen3-Emb-4B
 
UNREAL-Nemo3.5-Lightning-30B-A3BSingle-hop dataset Multi-hop dataset
Figure 9:Average recall@10on 21M Wiki-2018 chunks across 8 datasets. Bars show means with
95% BCa confidence intervals.
16

020406080Average recall@20
56.562.557.563.071.6 70.5 70.982.1 83.5 84.4 84.5HotpotQA
020406080
48.444.854.0 55.659.4 59.6 58.574.2 73.276.0 75.22WikiMultiHopQA
010203040
21.6 20.926.030.432.734.4 33.635.937.439.4 38.0MuSiQue
020406080Average recall@20
54.0 52.1 50.455.564.460.064.773.6 73.977.6 76.7HoVer
0204060
37.4
32.2 34.138.244.4 45.1 45.046.6 47.050.6 48.9IIRC
0204060
41.9
36.140.848.150.9 52.0 52.557.662.259.4 60.6SQuAD v2
0255075100Average recall@20
68.578.682.786.5 86.5 88.1 88.2 89.7 91.7 92.4 92.8FEVER
0204060
30.152.4
44.855.2 56.0 55.6 57.553.558.3 58.9 59.7Natural Questions
0204060
44.847.5 48.854.158.2 58.2 58.964.265.9 67.3 67.1Average across 8 datasets
BM25
Hybrid RAG
UNREAL-Qwen3.5-4BBGE-large-en-v1.5
LateOn (LI) + reranker
UNREAL-Qwen3.5-35B-A3BQwen3-Emb-0.6B
Qwen3-Emb-4B + reranker
UNREAL-Muse-Glimmer-30BQwen3-Emb-4B
 
UNREAL-Nemo3.5-Lightning-30B-A3BSingle-hop dataset Multi-hop datasetFigure 10:Average recall@20on 21M Wiki-2018 chunks across 8 datasets. Bars show means with
95% BCa confidence intervals.
Table 2 reports end-to-end question-answering performance with Qwen3.5-35B-A3B fixed as the
generator and only the retriever providing its top-5chunk context varied.
Table 2: Exact match (EM) and macro-averaged F1 on three QA datasets, generated by Qwen3.5-
35B-A3B under each retrieval method’s top-5 retrieved context. “Oracle” denotes the upper-bound
result, obtained by answering the question based only on the oracle chunks.
HotpotQA 2Wiki MuSiQue
Method EM F1 EM F1 EM F1
No context 25.0 34.6 27.7 33.3 6.9 14.8
Oracle 60.8 74.9 53.6 61.7 42.6 54.6
BM25 36.8 47.3 28.1 34.4 7.0 15.2
BGE-large-en-v1.5 36.9 47.3 27.6 33.7 9.7 18.1
Qwen3-Emb-0.6B 34.9 46.0 28.5 35.0 7.7 15.4
Qwen3-Emb-4B 36.5 48.7 31.7 38.4 10.4 18.4
Hybrid RAG 39.3 51.3 30.7 37.5 8.9 18.1
LateOn + reranker 41.3 53.2 31.5 38.5 10.8 19.6
Qwen3-Emb-4B + reranker 40.8 52.6 31.2 37.9 11.8 20.8
UNREAL-Qwen3.5-35B-A3B48.2 61.6 40.2 47.2 17.8 27.1
A.2 LONG-CONTEXT RESULTS
In Fig. 11 we present the effect of retrieved distractors vs random distractors on the NoLiMa dataset.
As can be seen, retrieved distractors reduce the NoLiMa score more than random distractors.
17

2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
chunks shown (gold always present)5060708090100NoLiMa score
oracle
random distractors
retrieved distractorsFigure 11:Distractor identity matters.With the gold chunk always present, retrieved distractors
cause more interference than the same number of random distractors, and the gap grows with the
budget.
1k 3k 10k 30k 100k
Mean prompt tokens01020304050LOFT score
k=5k=10k=20
NQ
HotpotQA MuSiQue T opiOCQAQUESTQAMPARI020406080LOFT score
0
No context Oracle (gold chunks) Full context (all chunks) UNREAL top-k (ours)
Figure 12:LOFT generation.(a) All selective budgets outperform full-context reading. (b) Per-
formance by dataset atn= 10against gold-only context. We use each dataset’s designated LOFT
metric.
A.2.1 INTRINSIC RETRIEVAL
Fig. 6 (right) evaluates whether the frozen Nemotron-3-Nano can retrieve evidence on NoLiMa
without retrieval training. We split each context into chunks and score them against the question
using the model’s attention keys and queries.
For attention layerℓand headh, letkℓ,h
i,tdenote the key vector at tokentof chunkc i, andqℓ,h
tthe
query vector at tokentof the questionx. We mean-pool these vectors over the chunk and question
tokens, respectively:
¯kℓ,h
i=1
|ci||ci|X
t=1kℓ,h
i,t,¯qℓ,h=1
|x||x|X
t=1qℓ,h
t.
We compute their dot product within each head and average over theH ℓheads to obtain a chunk
score:
sℓ
i(x) =1
HℓHℓX
h=1D
¯qℓ,h,¯kℓ,h
iE
.
For each layer separately, we rank chunks bysℓ
i(x)and compute recall@nover the top-nchunks. No
retrieval tokens or learned layer weights are used. These untrained representations retrieve evidence
above the random baseline.
18

A.3 ABLATION STUDIES
Table 3 presents a recipe ablation for Nemotron-3.5-Lightning-30B-A3B, in which we vary, one at
a time, the design choices used in the main method. Table 4 presents an ablation study of the layer
from which corpus chunk embeddings are produced. We find that middle-to-late layers yield the
strongest retrieval performance, a result that is reproduced across other architectures as well.
Table 3: Recipe ablation for UNREAL-Nemotron-3.5-Lightning-30B-A3B, single-knob departures
from the main method.Ris the number of retrieval tokens;Gis the number of groups they are
mean-pooled into on the query side; ML (multi-layer) vs. SL (single-layer) is whether the read-
out combines all 6 global-attention blocks or only the block used for the corpus readout; HN is
the number of hard negatives per query;|C 0(x)|is the number of BM25-retrieved context chunks
prepended to the query;Pis the number of learnable soft prompt tokens used at the beginning of
the sequence. Avg. R@10 is recall averaged over each query’s oracle chunks; All R@10 requires
every oracle chunk to be retrieved.
HotpotQA MuSiQue SQuAD v2
Design choice Avg R@10 All R@10 Avg R@10 All R@10 Avg R@10
Baseline 81.28 72.93 32.70 13.79 54.43
R=64→1677.93(-3.35)67.83(-5.10)31.91(-0.79)13.04(-0.75)53.82(-0.61)
G=4→177.91(-3.37)68.37(-4.56)29.69(-3.01)11.49(-2.30)52.94(-1.49)
ML→SL 73.02(-8.26)61.01(-11.92)26.20(-6.50)9.86(-3.93)46.93(-7.50)
HN=500→10079.40(-1.88)70.59(-2.34)28.77(-3.93)11.58(-2.21)52.86(-1.57)
|C0(x)|=5→178.50(-2.78)68.83(-4.10)31.36(-1.34)11.95(-1.84)52.68(-1.75)
|C0(x)|=5→071.08(-10.20)56.73(-16.20)28.83(-3.87)8.94(-4.85)50.16(-4.27)
P=1→079.61(-1.67)70.86(-2.07)32.17(-0.53)13.21(-0.58)52.92(-1.51)
Lp=7→178.20(-3.08)69.11(-3.82)30.28(-2.42)12.45(-1.34)48.52(-5.91)
Table 4: Ablation on the layer used to create the corpus chunk embeddings, for UNREAL-
Nemotron-3.5-Lightning-30B-A3B.
HotpotQA MuSiQue SQuAD v2
Corpus layer Avg R@10 All R@10 Avg R@10 All R@10 Avg R@10
Layer 8 79.68 70.25 31.74 12.33 53.66
Layer 18 78.98 69.42 31.65 12.45 54.05
Layer 30 80.19 71.39 32.40 13.33 54.85
Layer 38 80.62 71.99 31.94 12.29 54.01
Layer 4281.28 72.9332.7013.79 54.43
Layer 48 80.19 71.6432.7812.75 54.13
Table 5 examines how the number of pooled vectors per corpus chunk,L p, affects retrieval quality
for Nemotron-3.5-Lightning-30B-A3B. We encode the Wiki-2018 corpus atL p∈1,3,7,11and
report Average recall@10 and Complete-evidence recall@10. Recall increases monotonically with
Lpacross both metrics, confirming that a finer-grained chunk representation yields more accurate
retrieval, though this comes at the cost of higher memory and compute during evaluation.
B EXPERIMENTAL SETTING
B.1 RETRIEVAL DATA
We construct our retrieval data setting by mapping gold evidence from existing datasets to a shared
DPR Wikipedia 2018 corpus (Karpukhin et al., 2020b), using a KILT-like method (Petroni et al.,
2021). The corpus consists of around 21M chunks, with a median of around 141 Nemotron-3.5-
Lightning tokens per chunk. The downstream tasks span three categories: single-hop QA (Natural
Questions (Kwiatkowski et al., 2019), SQuAD v2 (Rajpurkar et al., 2018)), multi-hop QA (Hot-
potQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), MuSiQue (Trivedi et al., 2022),
19

Table 5: Effect of the number of pooled vectors per corpus chunkL pon retrieval recall@10, for
UNREAL-Nemotron-3.5-Lightning-30B-A3B. Avg. R@10 is the recall averaged over each query’s
oracle chunks; All R@10 requires every oracle chunk of a query to be retrieved. Recall increases
monotonically withL p, but at the cost of higher memory and compute during evaluation.
HotpotQA MuSiQue 2Wiki SQuAD v2
LpAvg R@10 All R@10 Avg R@10 All R@10 Avg R@10 All R@10 Avg R@10
1 78.20 69.11 30.28 12.45 67.43 55.21 48.52
3 79.34 70.58 31.15 12.70 69.51 57.11 51.61
7 81.28 72.93 32.70 13.79 71.52 58.90 54.43
11 81.54 73.40 34.34 14.67 72.38 59.68 55.42
IIRC (Ferguson et al., 2020)), and fact verification (FEVER (Thorne et al., 2018), HoVer (Jiang
et al., 2020)).
Based on KILT (Petroni et al., 2021), for each dataset, we map its oracle chunks to Wikipedia
2018 chunks via title matching. We then compute a text-alignment quality score based on length-
robust unigram, bigram, and trigram containment overlap between the oracle evidence text and each
candidate chunk. The score ranges from 0 to 1, where 1.0 means that the oracle text is essentially
contained verbatim in the chunk. For NQ, we directly reuse existing DPR bi-encoder gold mappings,
bypassing this KILT mapping step. For evaluation, similar to KILT, we report results on a filtered
set that retains an oracle only if it comes from an exact title match or has a fallback alignment score
above a quality threshold (0.6 by default).
We additionally retrieve lexical signals for each query. For each query, we use its input text as a
BM25 query and retain the top chunks as BM25 context, which is provided as additional input to
UNREAL (C 0(x)). We then mine oracle-seeded hard negatives: the query’s oracle chunks were
used as seed queries by running their text through BM25. The top lexically similar chunks across
all seeds are pooled and deduplicated, and any chunk overlapping with the query’s oracle set is
discarded. This yields up to 512 hard negatives per query that are lexically close to the gold evidence
but guaranteed not to belong to it, providing a clean hard-negative pool for contrastive training.
B.2 LONG-CONTEXT BENCHMARKS
We wish to note that the UNREAL models evaluated in the long-context setting were not trained
on long-context data. Instead, the UNREAL modules were trained exclusively on standard retrieval
data (as detailed in Section B.1) and subsequently evaluated on both retrieval and long-context tasks.
Following the methodology of LOFT (Lee et al., 2024), we construct a new open-domain QA bench-
mark scaling context lengths from 8K to 100M tokens (Fig. 1). Our goal is to evaluate long-context
generation on context scales significantly larger than those present in current benchmarks.
We sample 500 queries at random from the validation splits of four QA datasets: Natural Ques-
tions (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al.,
2020), and MuSiQue (Trivedi et al., 2022), totaling 2,000 questions. Each query is paired with ora-
cle passages from the DPR Wiki-2018 corpus (approximately 21 million passages). We retain only
questions whose gold passages match corpus entries by exact article title.
To construct the extended contexts, all queries share a single distractor pool. We remove all gold
passages associated with any validation query across the four datasets to ensure distractors do not
overlap with target evidence. Contexts are constructed across ten lengths: 8K, 16K, 32K, 128K,
256K, 512K, 1M, 10M, 30M, and 100M tokens, measured using the Nemotron tokenizer over con-
text passages. Distractors are drawn sequentially from the head of the shuffled corpus to fill the
token budget alongside the gold passages. Each gold passage is placed at a relative context depth
sampled uniformly at random for that query; these relative depths remain fixed across all context
lengths. Consequently, as context length scales, a question maintains its gold evidence at identical
relative positions, altering only the volume of surrounding distractor text. All stochastic choices,
including question sampling, corpus shuffling, and passage positioning, use fixed random seeds to
guarantee deterministic inputs across evaluated models and runs.
20

To evaluate retrieval methods on long-context benchmarks, contexts are split into sentence-aligned
chunks averaging 141 tokens (mostly ranging from 123 to 165 tokens) to match the Wiki-2018
length distribution used during retriever training. The chunks form an exact cover of the source text
and are created on a per-document basis to prevent them from crossing document boundaries. For
generation, retrieved chunks are reassembled into native prompt templates in their original document
order and scored using native metrics.
B.3 METHOD AND TRAINING CONFIGURATION
On the corpus side, we use the full Wikipedia-2018 corpus, containing around 21M chunks. Each
chunk is encoded only once using the frozen LLM, by extracting the outputs of a single layer and
compressing them intoL p= 7pooled vectors.
On the query side, we useR= 64retrieval tokens pooled intoG= 4groups, together with an
initial contextC 0(x)consisting of the top-5 BM25 chunks. The query embeddings are obtained by
a learnable sum of the full-attention outputs of the model. Moreover, only for query processing,
a single learnedsoft-promptvector is concatenated to the beginning of the retrieval input. Unlike
ρ, this vector is never read out or scored; it only steers the frozen LLM toward a retrieval-oriented
computation and is trained jointly withρandα. These components are the only learnable parameters
in UNREAL, totaling fewer than 500K parameters, or less than2×10−5of the frozen model in all
cases. The exact number of learnable parameters is(R+ 1)D+|α|, whereDis the embedding
dimension and|α|is the number of layer-mixing weights.
We trained the UNREAL models using the training sets of Natural Questions (Kwiatkowski et al.,
2019), SQuAD v2 (Rajpurkar et al., 2018), HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho
et al., 2020), MuSiQue (Trivedi et al., 2022), IIRC (Ferguson et al., 2020), FEVER (Thorne et al.,
2018), and HoVer (Jiang et al., 2020), all mapped into a single shared Wikipedia-2018 corpus (see
Appendix B.1). We note that all retrieval baselines used in this work, such as Qwen3-Embedding
(Zhang et al., 2025), were also fine-tuned on high-quality retrieval datasets like HotpotQA and
Natural Questions.
For the contrastive objective, we draw500hard negatives per query from a BM25 ranking seeded
on that query’s oracle chunks. We discard the top10ranks, as they are typically near-duplicates of
the oracle chunks themselves.B(x)is then the deduplicated union of the oracle and hard-negative
chunks from all queries in a microbatch, ensuring that a chunk serving as an oracle for one query
is never treated as a negative for that query. We train for15Ksteps with a global batch of256
queries, using AdamW withβ= (0.9,0.95), weight decay0.1, gradient clipping at norm1.0,
and100warmup steps followed by linear decay to zero. At evaluation time, each query is scored
exhaustively against allMchunks, without an approximate index or candidate pre-filter.
The per-backbone settings are as follows. For Qwen3.5-35B-A3B (40blocks,d= 2048), we read
the index atℓ c= 27, using a learning rate of5×10−3. For Muse-Glimmer-30B (52blocks,
d= 6656), we useℓ c= 47, and a learning rate of5×10−3. For Nemotron-3.5-Lightning-30B-A3B
(52blocks,d= 2688), we useℓ c= 42, and a learning rate of10−2.
C FLOP ANALYSIS
UNREAL replaces full-context inference with chunk encoding, retrieval, and generation over se-
lected evidence. This avoids processing irrelevant context but adds retrieval and re-encoding work.
We compare these costs and derive the context length above which UNREAL requires fewer FLOPs,
counting only the work on which the two pipelines differ.
Full-context inference vs. UNREAL.Full-context inference prefills[context, x]. UNREAL in-
stead:
(i) Encodes each chunk independently up to layerℓ c(equation 1);
(ii) Runs one forward pass through allLlayers onx ret(equation 2), of length|x ret|=
|C0(x)|T+T q+R;
(iii) Prefills[C UNREAL (x), x], of lengthnT+T q, for generation (equation 7).
21

After prefill, both pipelines perform the same autoregressive decoding except for the cached context
length: full context attends toN+T qprompt tokens, whereas UNREAL attends tonT+T q.
Thus UNREAL uses fewer attention FLOPs and a smaller KV cache, which benefits memory-bound
decoding.
C.1 SETTING
All chunks have equal length,T i≡T, so the context given to the model containsN≡MTtokens.
Both pipelines use a KV cache, andT gdenotes the number of decode forward passes. A multiply-
add counts as two FLOPs.
Weights.For identical layers, passing one token through the firstℓlayers costs2Θℓ/LFLOPs,
whereΘcounts backbone matrix parameters active per token. It includes all sequence-mixer and
dense-FFN projections; in MoE layers, it counts the router, shared experts, and selected routed ex-
perts. It excludes token embeddings, which are lookups, and the LM output head, whose generation
cost cancels in equation 8.
Attention.One causal query-key pair costs4L adaFLOPs, summed over theL aglobal-attention
layers:2d afor the score and2d afor the value accumulation. For a standard decoder,L a=Land
da=d.
Linear-attention and state-space layers have no position-dependent pairwise term, so we include
their projections inΘand neglect their small recurrent-update cost. We also omit sliding-window
attention: full context attends to up toWkeys per token, so this conservatively understates UN-
REAL’s savings.
Omitted operations.Softmax, normalization, and activations are ignored. The selection stage is
also neglected: BM25 forC 0(x)involves no dense matrix products, and the remaining selection
steps are tiny. These are mean pooling (aboutdFLOPs per token), theα-weighted layer sum, and
MaxSim scoring (equation 4,2dGL pFLOPs per chunk). Chunk encoding alone costs2Θℓ c/Lper
context token, and the ratio(1 + 2GL p/T)dL/(2Θℓ c)is about2×10−7for the models below (this
ratio covers pooling and MaxSim).
C.2 FLOPDIFFERENCE: UNRAELVS.FULL-CONTEXT INFERENCE
The two pipelines perform identical work on two things:
• the context tokens in layers1, . . . , ℓ c, including all intra-chunk attention there;
• the shared generation-side work: the query tokens’ weight products and query-to-query
attention in the prefill; all decode-step weight products and attention to the query and pre-
viously generated tokens; and all required generation logits.
Letλdenote the fraction ofΘin layersℓ c+ 1, . . . , L, for identical layers,λ= 1−ℓ c/L. LetL a,≤c
count global-attention layers throughℓ c. The difference∆≡F full−F UNREAL is
∆ = 2ΘλN|{z}
(a)+ 2L ada
N(N−T) + 2(T q+Tg)(N−nT)
| {z }
(b)
−
2ΘnT+ 2L adanT(nT+ 1)
| {z }
(c)
−
2Θ|x ret|+ 2L ada|xret|(|x ret|+ 1)
| {z }
(d)
+ 2d a(La−La,≤c)N(T+ 1)| {z }
(e).(8)
(a)Deeper layers on the context (UNREAL saving).Full-context inference passes every con-
text token through allLlayers. UNREAL reads the chunk embeddings at layerℓ c(equa-
22

tion 1) and discards the chunk states, since the selected chunks are re-encoded during gen-
eration. It therefore never runs layersℓ c+ 1, . . . , Lon the context, saving2Θλper context
token.
(b)Attention to unselected chunks (UNREAL saving).Each part counts causal query-key pairs,
at4L adaFLOPs per pair.
–Prefill.The full-context prefill computes allN(N+ 1)/2causal pairs among the
context tokens. UNREAL’s independent chunk encoding computes only theM T(T+
1)/2 =N(T+1)/2intra-chunk pairs. The remainingN(N−T)/2cross-chunk pairs
give the first term.
–Query.In full context, each of theT qquery tokens attends to allNcontext tokens. In
UNREAL it attends only to thenTselected tokens, a difference ofT q(N−nT)pairs.
–Decode.Likewise, each of theT gdecoded tokens attends toN−nTfewer keys, a
difference ofT g(N−nT)pairs.
(c)Re-encoding the selected chunks (UNREAL extra cost).UNREAL’s generation pre-
fill passes thenTselected tokens through all layers again (2ΘnT) and computes their
nT(nT+ 1)/2causal self-attention pairs. The query’s attention to these tokens is already
counted in (b).
(d)Retrieval pass (UNREAL extra cost).The forward pass onx rethas no counterpart in full-
context inference. Its tokens cost2Θ|x ret|in weights plus|x ret|(|x ret|+ 1)/2attention
pairs.
(e)Intra-chunk attention aboveℓ c(UNREAL saving).This restores the full-context-only intra-
chunk work in theL a−La,≤c global-attention layers above the readout.
The+1parts of (c)-(d) are diagonal-pair corrections, only1/κof the corresponding weight
costs, whereκ≡Θ/(L ada). Term (e) is a second correction, across the three backbones and
n∈ {5,10,20}, omitting it raises the roots by at most0.21%, so we omit it below.
C.2.1 SIMPLIFYING ASSUMPTIONS
We now reduce equation 8 under two assumptions:
(S1)Short queries and generations:T q=Tg=T.
(S2)The context is much longer than the selected evidence:N≫nT. Sincen≥1, this also
impliesN≫T. For fixed, moderaten,|C 0(x)|, andR/T, it further givesN≫ |x ret|,
since|x ret|= (|C 0(x)|+ 1)T+R=O(nT).
∆≈2ΘλN|{z}
deeper layers+ 2L adaN2
|{z}
cross-chunk attention−2Θ 
nT+|x ret|
|{z }
re-encoding and retrieval pass.(9)
The neglected terms have relative size of orderT/Nand(nT/N)2, which vanish in the long-context
limit. The diagonal corrections are also smaller than their weight terms byL ada/Θ = 1/κ; ad-
ditionally,κ≫1for all evaluated backbones. Because these dropped terms have mixed signs,
Eq. equation 9 is an approximation, not a bound.
C.2.2 BREAK-EVEN CONTEXT LENGTH
Dividing Eq. equation 9 by2L ada, UNREAL requires fewer FLOPs (∆>0) if and only if
N2+λκN−κ 
nT+|x ret|
>0,
that is, forN > N∗with
N∗=1
2
−λκ+q
λ2κ2+ 4κ 
nT+|x ret|
.(10)
23

Last-layer readout.Forℓ c=L(i.e.λ= 0), Eq. equation 10 reduces to
N∗=q
κ 
nT+|x ret|
.(11)
This is the length at which the avoided cross-chunk attention, about2L adaN2, equals the weight
cost of the tokens UNREAL processes additionally. Anyλ >0adds the linear saving (a) and lowers
N∗below this value.
Self-consistency.Forλ= 0, letQ=nT+|x ret|. The conditionQ≪κgivesN∗/Q=p
κ/Q≫
1, so the predicted root satisfies (S2). This generally holds for large LLMs with modest selected
evidence.
Longer generations.WhenT g≫T, term (b) adds4L adaTg(N−nT)FLOPs of savings, low-
ering the break-even length towardnTfrom above.
C.3N∗FOR THE EVALUATED BACKBONES
We evaluate equation 10 for the three 30B-scale backbones, using their released configurations and
the per-backboneℓ cof Appendix B.3.
For all three backbones we useT= 141(the median chunk length in Nemotron tokens),|C 0(x)|=
5,R= 64, andT q=Tg=T, so|x ret|= 910. Table 6 reports the results.
Table 6: Approximate break-even context lengthN∗(tokens; equation 10), withT= 141,T q=
Tg=T, and|x ret|= 910.Θcounts active non-embedding matrix parameters.
N∗(tokens)
BackboneΘL a κ ℓ c/L λ n= 5n= 10
Muse-Glimmer-30B 25.2B 134.73×10547/52 0.096 13,049 17,437
Nemotron-3.5-Lightning 2.87B 61.17×10542/52 0.201 6,321 8,471
Qwen3.5-35B-A3B 2.44B 105.95×10427/40 0.323 4,117 5,569
Across these settings, equation 10 is within0.55%of the numerical root of the full equation 8.
D EFFICIENCY BENCHMARK:METHODOLOGY AND FULL RESULTS
We report measured time-to-first-token (TTFT) and throughput for UNREAL’s chunk-then-select
pipeline against full-context inference. This section gives the full methodology, the FLOPs model,
and results for all four backbones from Section 3.
Setup.We benchmark performance only, not accuracy: all models are randomly initialized
(load format="dummy") and all inputs are random token ids (skip tokenizer init), so
no checkpoints or tokenizers are required. We serve each backbone with vLLM (Kwon et al., 2023)
rather than a plaintransformersforward pass, so results reflect PagedAttention, FlashAttention-
3, and continuous batching. We use the chunking regime evaluated throughout Section 4 (T=128,
n=10) and sweepLover the same ten values as Fig. 1 (8K–100M tokens).
All measurements use bf16 weights and activations on a single NVIDIA H100 80GB GPU, except
Qwen3.5-35B-A3B’s KV cache, stored in fp8 since its≈64GB of resident weights (256 routed
experts) leave too little bf16 headroom for a 256K-token cache. With this, all four baselines are
measured directly up to 256K tokens. vLLM dispatches each backbone’s optimized fused kernels
(FlashInfer’s Gated-DeltaNet for linear attention, native Mamba2/MoE mixers for Nemotron-3.5-
Lightning)
FLOPs model.We count prefill multiply-add FLOPs as 2 FLOPs/MAC, walking each backbone’s
realper-layer sequence rather than a single network-wide ratio. Every layer contributes a linear
(O(L)) projection cost – QKVO for full-/sliding-attention, or the real linear-attention / Mamba2
projection sizes otherwise plus its feed-forward cost (dense or MoE, charging only the active and
24

10K 100K 1M 10M 100M
Context length L (tokens)100102104106Time (s)
2×
9×
80×
793×
T = 128, n = 10Qwen3.5-4B
10K 100K 1M 10M 100M
Context length L (tokens)100102104106Time (s)
1×
3×
23×
225×Muse-Glimmer-30B
10K 100K 1M 10M 100M
Context length L (tokens)100102104106Time (s)
2×
15×
146×
1455×Qwen3.5-35B-A3B
10K 100K 1M 10M 100M
Context length L (tokens)100102104106Time (s)
2×
7×
64×
638×Nemotron-3.5-Lightning-30B-A3B
Baseline, measured Baseline, FLOPs-projected UNREALFigure 13: Measured time-to-first-token, baseline vs. UNREAL, for all four backbones (random
weights/inputs, served with vLLM on an NVIDIA H100, bf16). Solid markers are directly mea-
sured; the dashed baseline segment is a FLOPs-scaled projection past the point where a single full-
context forward pass exceeds our measurement budget or the available KV cache. Dotted vertical
connectors mark the speedup (baseline time / UNREAL time) at a few representative context lengths,
read directly off the time gap between the two curves.
any always-on shared experts, with 2 or 3 weight matrices matching each backbone’s real gated/non-
gated activation). Full-attention layers additionally costC attn(L)=2L2dattn(dattn=nheadsdhead);
sliding-attention layers cost2Lmin(L, W)d attnfor windowW– linear inLonceL>W, the same
logic that makes UNREAL’s ownT-chunked encoding linear; linear-attention and Mamba2 layers
add no quadratic term. The LM head is counted only on passes that emit logits. Summed over the
real layer sequence, full-context inference costsFLOPs base(L) =O(L) +O(L2), while UNREAL
costsFLOPs UNREAL (L) =O(L) +O(L T)|{z }
encode, throughℓ c+O(M)|{z}
score+O(1)|{z}
query+O(1)|{z}
generate(M=L/T): the quadratic
term shrinks toL·L C, and generation always reads exactlynT=1,280tokens regardless ofL.
Measured wall-clock results.Figs. 13 and 14 report measured TTFT and throughput for all four
backbones. The baseline curve is shown only up to the context length at which a single for-
ward pass exceeds our 75-second measurement budget (solid markers), matching the convention
of Fig. 1; we extend it with a dashed, FLOPs-scaled projection (t proj(L) =t measured (Lcutoff)·
FLOPs base(L)/FLOPs base(Lcutoff)) for visual continuity across the same 8K–100M range. UN-
REAL’s chunk-encoding time is computed from its measured, steady-state per-chunk throughput.
Chunk encoding is parallel across theM=L/Tchunks, so total encode time scales linearly withM
plus the directly measured, constant-size query and generation passes. This measuredT UNREAL (L)
omits two components, both excluded because they are negligible or out of scope rather than favor-
able to UNREAL: (1) the MaxSim scoring step (O(M)in the FLOPs model above) is not separately
benchmarked, as it is a lightweight vector similarity computation external to the LLM forward
pass that accounts for under0.01%of UNREAL’s total FLOPs at every backbone and length we
tested; and (2) any orchestration overhead between the encode and generate stages (selecting the
top-kchunks, assembling the generation prompt), which a production serving stack would incur but
which our per-stage vLLM benchmarking (each stage measured on its own, isolated engine) does not
capture. The measured curves corroborate the FLOPs estimate: UNREAL’s TTFT grows roughly
linearly withLwhile the baseline’s grows quadratically, so the baseline becomes both slower and,
eventually, infeasible to run at all, in the same context-length regime where Fig. 1 shows its accuracy
collapsing.
25

10K 100K 1M 10M 100M
Context length (tokens)102103104Prefill throughput (tok/s)
2×
9×
80×
793×Qwen3.5-4B
10K 100K 1M 10M 100M
Context length (tokens)102103104Prefill throughput (tok/s)
3×
23×
225×Muse-Glimmer-30B
10K 100K 1M 10M 100M
Context length (tokens)102103104105Prefill throughput (tok/s)
2×
15×
146×
1455×Qwen3.5-35B-A3B
10K 100K 1M 10M 100M
Context length (tokens)102103104Prefill throughput (tok/s)
2×
7×
64×
638×Nemotron-3.5-Lightning-30B-A3B
Baseline, measured Baseline, FLOPs-projected UNREALFigure 14: Measured prefill throughput (tokens/second), baseline vs. UNREAL, for all four back-
bones. Same data and conventions as Fig. 13.
26