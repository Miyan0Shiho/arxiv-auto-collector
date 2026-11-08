# CacheGraft: Enhancing Retrieval-Augmented Generation via Latent-Space Document Enrichment

**Authors**: Vihan Lakshman, Peter Baile Chen, Zhuohan Gu, Michael Cafarella

**Published**: 2026-11-05 03:43:22

**PDF URL**: [https://arxiv.org/pdf/2611.07393v1](https://arxiv.org/pdf/2611.07393v1)

## Abstract
Recent work in machine learning has demonstrated that transformer-based language model agents can communicate state not only through text, but also directly in latent space via key-value (KV) caches. In this paper, we argue that such cache-to-cache translation can provide utility beyond the original motivating application of real-time agent communication and can instead be utilized for retrieval-augmented generation (RAG). Building on the Cache-to-Cache (C2C) fusion framework, our proposed system CacheGraft encodes each corpus passage offline into a fused KV cache representation between a larger sharer model (4B parameters in our experiments) and a smaller receiver model (0.6B parameters) such that this cache can be injected into the receiver at query time with no need for the larger model. Across five QA benchmarks and two independent locally-run judges, CacheGraft improves answer quality over the text-only receiver baseline in 29 out of 30 comparisons, of which 17 of the 30 are statistically significant. Notably, these gains are zero-shot with respect to the fuser, which was never trained on retrieval data. These initial results suggest that latent-space document enrichment may be a promising avenue for enhancing the performance of RAG systems, particularly in compute-constrained settings where it is only feasible to deploy a small model at inference time.

## Full Text


<!-- PDF content starts -->

CacheGraft: Enhancing Retrieval-Augmented Generation
via Latent-Space Document Enrichment
Vihan Lakshman
MIT CSAIL
Cambridge, USA
vihan@mit.eduPeter Baile Chen
MIT CSAIL
Cambridge, USA
peterbc@mit.edu
Zhuohan Gu
MIT CSAIL
Cambridge, USAMichael Cafarella
MIT CSAIL
Cambridge, USA
michjc@csail.mit.edu
Abstract
Recent work in machine learning has demonstrated that transformer-
based language model agents can communicate state not only
through text, but also directly in latent space via key-value (KV)
caches. In this paper, we argue that such cache-to-cache transla-
tion can provide utility beyond the original motivating application
of real-time agent communication and can instead be utilized for
retrieval-augmented generation (RAG). Building on the Cache-to-
Cache (C2C) fusion framework, our proposed system CacheGraft
encodes each corpus passage offline into a fused KV cache rep-
resentation between a larger sharer model (4B parameters in our
experiments) and a smaller receiver model (0.6B parameters) such
that this cache can be injected into the receiver at query time
with no need for the larger model. Across five QA benchmarks
and two independent locally-run judges, CacheGraft improves an-
swer quality over the text-only receiver baseline in 29 out of 30
comparisons, of which 17 of the 30 are statistically significant. No-
tably, these gains are zero-shot with respect to the fuser, which
was never trained on retrieval data. These initial results suggest
that latent-space document enrichment may be a promising avenue
for enhancing the performance of RAG systems, particularly in
compute-constrained settings where it is only feasible to deploy a
small model at inference time.
CCS Concepts
•Information systems→Information retrieval.
Keywords
retrieval-augmented generation, KV caching, model collaboration,
cache translation
1 Introduction
Recent breakthroughs in the machine learning community [ 7,8,16]
have demonstrated that transformer-based large language model
(LLM) agents can communicate information between one another
not only through the standard medium of text, but also inlatent
spaceby translating the key-value (KV) cache from asendermodel
to areceivermodel’s internal vector space. Notably, this translation,
typically accomplished by learning a separate adapter model, is
feasible even if the two models have a differing number of weights
and layers [7, 8, 16].sharer (4B)
receiver (0.6B)passage𝑑 fuser
fused KV
cachepersist
receiver (0.6B) question𝑞 answer[𝑞]as textload
[𝑝 𝑑]as KVPhase 1 — offline, once per passage
Phase 2 — online, per query
Figure 1: The sharer and fuser run only offline, once per
corpus passage, with no knowledge of any query. The fused
KV cache is the only artifact that crosses the offline/online
boundary: at query time the receiver loads it and decodes
alone, reading the passage [𝑝 𝑑] in latent space and the ques-
tion[𝑞]as text.
The overarching motivation driving this effort in KV cache trans-
lation is the vision of heterogeneous LLMs collaborating in a real-
time multi-agent system by passing state via KV cache vectors as
opposed to raw text. By operating in latent space, KV cache transla-
tion offers the promise of improving both thequalityof multi-agent
computation, since KV caches constitute richer conceptual repre-
sentations of information in an LLM’s native language, and the
performanceof such systems by eliminating the model’s initial step
of converting a textual input into KV vectors.
The central observation behind this paper is that KV cache trans-
lation can be useful beyond the original motivating goal of real-time
multi-agent LLM communication. To that end, we introduce the
CacheGraft framework applying the KV cache translation paradigm
for retrieval-augmented generation (RAG) where an LLM marshals
knowledge from an external corpus to enhance its context before
generating a response. Applying KV cache translation to RAG is
not immediately obvious because typical RAG deployments often
consist of a single LLM; it is unclear where cross-model cache trans-
lation even comes into the picture. Our key insight is that we can
arXiv:2611.07393v1  [cs.IR]  5 Nov 2026

KEIR @ CIKM 2026, November 2026, Rome, Italy Vihan Lakshman, Peter Baile Chen, Zhuohan Gu, and Michael Cafarella
leverage KV cache translation during theoffline indexingphase
of RAG pipelines by enriching the representation of a document
through fusing the KV cache of a larger model (that cannot be
deployed at query time) with the cache of a smaller model that does
run online. Thus, the larger model serves as a de facto teacher and
guides the smaller receiver model to better “read" the contents of
the documents.
Unlike multi-agent communication, the offline phase of a RAG
pipeline is not latency-critical, making the overhead of KV cache
fusion more palatable than in the real-time communication setting.
Moreover, this offline phase may enable using more powerful, but
computationally expensive LLM models for indexing. These insights
drive the development of our CacheGraft framework. In summary,
we make the following contributions in this paper:
•Conceptual Insight:To our knowledge, this work is the
first to make the observation that KV cache translation can
be leveraged in applications outside of the canonical moti-
vating example of real-time agent-to-agent communication.
In particular, we demonstrate that the Cache-to-Cache (C2C)
fusion technique [ 8] can lead to improved generation quality
in retrieval-augmented generation (RAG) settings.
•Proof-of-Concept Evaluation:We instantiate CacheGraft
using an off-the-shelf pretrained KV cache fusion model pub-
lished by the C2C team1that fuses the KV cache of Qwen3-
4B-Base model with the cache from a Qwen3-0.6B model
into the latter’s KV space. We evaluate our proposed sys-
tem2on five knowledge-intensive RAG question-answering
benchmarks which differ from the original C2C evaluation
benchmarks which largely focus on reasoning tasks. Interest-
ingly, CacheGraft achieves these gains using a cache fusion
model that was not explicitly trained for these retrieval ap-
plications.
•Long-Term Research Vision:In addition to our proof-of-
concept experiment results, we argue that KV cache fusion
for RAG applications is a rich and understudied area and we
highlight several open research problems for the community.
2 Related Work
Our work in this paper sits at the confluence of three research
threads in machine learning and information retrieval that were
previously disconnected. First is the aforementioned line of work
on KV cache translation [ 7,8,16] including the C2C framework [ 8]
we directly use in our experiments. We focus on using C2C in
our evaluation of CacheGraft because, to our knowledge, it has
demonstrated efficacy with the largest gap between the sharer and
receiver models (4B to 0.6B parameters) and thus constitutes the
most scalable translation method to date.
Secondly, on the RAG front, a number of works have proposed
precomputing and storing a model’s KV cache representation of
documents and using the cache, not the raw text, at query time
to reduce latency by eliminating the step of converting text into
KV values [ 5,11,12,22]. However, these methods all use the KV
cache of a single model and do not consider enriching the cache
1https://huggingface.co/nics-efc/C2C_Fuser
2Our code is available at https://github.com/vihan-lakshman/CacheGraftrepresentation with signal from an additional, more powerful LLM,
which differentiates our approach in this paper.
Finally, multiple research efforts have also examined the problem
ofdocument enrichment, namely preprocessing the text in a corpus
to be more amenable to RAG applications [ 4,10,14,19]. CacheGraft
can also be viewed as a document enrichment strategy, but in latent
space as opposed to at the textual level.
3 Background: Cache-to-Cache (C2C)
In this section, we briefly review the details of the C2C frame-
work [ 8] as CacheGraft repurposes this same fusion model for
offline document enrichment. In short, C2C enables asharerLLM
to pass contextual understanding to areceiverLLM directly in la-
tent space, by fusing the sharer’s key-value (KV) cache into the
receiver’s. Both underlying models remain frozen; the only trained
component is a lightweight per-layerfusernetwork. More formally,
let𝑥=[𝑥 0, . . . , 𝑥 𝑛−1]denote the input token sequence (in our set-
ting, the RAG prompt; see Section 4). The sequence is tokenized
independently by each model. Both frozen models then prefill on
the same input: the receiver produces its per-layer KV cache C𝑛(𝑥)
for layers 𝑛= 1, . . . , 𝑁 , and the sharer produces its cache CS
𝑚(𝑥)
for its own layers 𝑚= 1, . . . , 𝑁′. The text input is consumed twice,
once by each model’s prefill.
Because the two models may differ in depth, a layer-mapping
strategyGassigns to each receiver layer 𝑛a sharer layerG(𝑛) ; C2C
adoptsterminal alignment, pairing final layers first and proceeding
backward. A per-layer fuserF 𝑛then produces the fused cache:
CF(𝑥)=
F𝑛 C𝑛(𝑥),CS
G(𝑛)(𝑥)	𝑁
𝑛=1.(1)
The fused cacheCF(𝑥)lives entirely in the receiver’s KV space, has
the same sequence length 𝑛as the input (no extra token positions
are introduced), andreplacesthe receiver’s original prompt cache.
At inference time, the receiver performs text generation alone.
With generated prefix 𝑦[0:𝑖]=[𝑦 0, . . . ,𝑦 𝑖−1]and the receiver’s ordi-
nary self-produced cache C(𝑦[0:𝑖])over that prefix, the next token
is predicted as
𝑦𝑖+1=P 𝑦𝑖CF(𝑥)⊕C(𝑦[0:𝑖]),(2)
where⊕denotes sequence-wise concatenation. After fusion, the
sharer plays no further role: newly generated tokens append receiver-
native cache entries behind the fused prompt cache exactly as in
standard autoregressive decoding.
Two properties of this construction are essential for the RAG set-
ting in CacheGraft. First, the fused cache is a self-contained artifact:
Equation 2 requires only CF(𝑥)and the receiver, not the sharer or
the fuser. Second, with causal attention, the fused entries at a given
position depend only on tokens at earlier positions. Together these
properties allow the fusion step of Equation 1 to be moved offline
entirely, computed once per document without knowledge of any
query.
One limitation of the C2C framework in the interactive multi-
agent setting is that it requires running a forward pass of the text
input 𝑥throughboththe sharer and receiver model, which limits the
total possible amount of latency savings. A major goal of subsequent
work in KV cache translation, such as Latent Cache Flow [ 16], is to
eliminate this shortcoming. However, in the RAG setting, this fusion

CacheGraft: Enhancing Retrieval-Augmented Generation
via Latent-Space Document Enrichment KEIR @ CIKM 2026, November 2026, Rome, Italy
approach is not a notable hindrance since the cache enrichment
step happens offline and not at query time.
4 CacheGraft: Adapting C2C for RAG
We now describe the details of our CacheGraft system. As a matter
of notation, we will denote a RAG prompt by 𝑥=[𝑝, 𝑑, 𝑞] which
is a concatenation of a fixed preamble 𝑝(also known as a system
prompt), a retrieved document 𝑑, and the question 𝑞. We focus our
attention on knowledge-intensive question-answering applications
that require a factual answer typically not found in the model’s
own weights.
With causal attention, both models’ pre-fusion cache entries at
positions 𝑖≤|𝑝 𝑑| are independent of any token at position 𝑗>𝑖 ;
the projector acts position-wise and layer-wise on those entries; and
the fusion of[𝑝 𝑑] does not depend on 𝑞. Hence computing [𝑝, 𝑑, 𝑞]
in one pass and computing [𝑝, 𝑑] alone yield identical KV over
[𝑝, 𝑑] . As a consequence, for top-1 retrieval with a fixed preamble,
the document always occupies the same position range, so no RoPE
re-rotation [ 5] is needed. We leave extending our framework to
support top- 𝑘retrieval for 𝑘> 1using techniques such as TurboRag
finetuning [11] or CacheBlend [22] for future work.
Figure 1 depicts the full CacheGraft pipeline. Offline, for each pas-
sage, we prefill[𝑝, 𝑑] with fusion enabled and persist the receiver-
space KV, keyed by a content hash. We also maintain an additional
vector database index of document embeddings. Online, we iden-
tify the most relevant document to the query based on embedding
similarity and load that document’s fused KV cache into thebase
receiver and decode the question. Crucially, we do not need any-
thing from the sharer or the cache projection fuser model on the
query path.
5 Experiments
5.1 Experimental Setup
We conduct all experiments on a single NVIDIA A100-40GB GPU
and use standard greedy decoding at inference time.
Models.In all reported experiments, we use Qwen3-0.6B [ 21]
as our receiver and Qwen3-4B-Base [ 21] as the sharer, with the
released C2C projector for that pair. The projector is trained on
general chat distillation and never sees retrieval data, so every result
is zero-shot with respect to fuser training.
Data.We utilize five standard knowledge-intensive question
answering benchmarks for our RAG evaluation: SQuAD v1.1 [15],
AdversarialQA [ 2], CovidQA [ 13], QASPER [ 6] and TriviaQA [ 9].
For search, we use bge-small [ 20] embeddings to perform dense top-
1 retrieval. For evaluation, we retain one question per gold passage.
We also run our experiments under retrieved andoracle (perfect-
retrieval) conditions to understand the impact of search quality on
the overall RAG performance.
Metrics.We utilize the LLM-as-judge mechanism [ 23] as our
primary evaluation metric. We evaluate all RAG outputs using
two judges, both locally-run open-weight models scored on raw
unmodified generations: phi-4 [1] and Gemma-2-9B [18]. We also
report contains which measures whether the generated answer
contains the gold ground truth answer included in the test dataset.We also compute statistical significance measures using the exact
two-sided paired McNemar test over the set of evaluation questions.
We do not rely on traditional RAG metrics such as Exact Match (EM)
and F1 score because we observed that the generated answers from
fused KV caches tend to be more verbose (illustrated in Table 2)
which hurts performance on these token-centric metrics.
5.2 Results
As shown in Table 1, we find that delivering the passage as a pro-
jected KV cache improves over using the receiver with raw text on
nearly every metric across every benchmark by up to 8 percentage
points, with the exception of QASPER, which shows similarly clear
gains with an oracle retriever but a disagreement between the two
judges when using top-1 embedding-based search.
In Table 3, we observe that QASPER has the smallest gap between
the sharer and receiver RAG quality which explains the more mixed
results from cache fusion: there is limited headroom for the sharer
to lift the receiver. Overall, Table 3 suggests two preconditions for
CacheGraft to provide value on knowledge-intensive RAG tasks: the
sharer must have headroom over the receiver, and the documents
must supply factual signal the sharer can exploit.
We also include select examples of CacheGraft’s successes and
failures on the SQuAD dataset in Table 2. We observe that cache
fusion can indeed steer the receiver model towards relevant portions
of the passage in several cases, but also makes mistakes. We also
note that the outputs from CacheGraft tend to be more verbose than
both the receiver and sharer models which also limits CacheGraft’s
total latency gains as shown in Table 4. Nevertheless, Table 4 still
illustrates an advantage for CacheGraft in terms of GPU memory
consumption. We posit that the increased verbosity of CacheGraft’s
fused KV cache can be mitigated by explicitly training for retrieval
tasks, which we highlight as another direction for future work.
6 Limitations and Open Problems
6.1 Limitations
The primary practical limitation of our proposal is storage since we
must store the KV cache for every document which can consume
orders of magnitude more space than raw text (on the order of
a terabyte for 100K documents, absent any optimizations). This
problem is inherited from prior works that propose to precom-
pute the KV caches for retrieval such as TurboRAG [ 11], Block
Attention [ 12], and CAG [ 3] and can be mitigated with KV cache
compression techniques [ 17] that are complementary to our work.
Another limitation of our present study is that we focus on retriev-
ing a single document and only consider a single C2C fuser model
in our evaluations. Extending our system to support top- 𝑘retrieval
and additional models is another promising direction for future
work.
6.2 Open Problems
The results in this paper present a proof of concept demonstration
of KV cache fusion to enhance RAG applications and open many
directions for future research in this domain. Below, we highlight
two particularly interesting directions:

KEIR @ CIKM 2026, November 2026, Rome, Italy Vihan Lakshman, Peter Baile Chen, Zhuohan Gu, and Michael Cafarella
Table 1: Performance of CacheGraft in comparison to using only the 0.6B receiver model and only the 4B sharer model across
two LLM judges and the contains metric, which measures if the correct answer is included in the output text. CacheGraft
improvements over the receiver-only baseline are marked in bold. Δreports CacheGraft minus Receiver-Only for each metric.
phi-4 Gemma contains
Dataset Setting𝑛Top-1 Recall Receiver-Only CacheGraft Sharer-OnlyΔReceiver-Only CacheGraft Sharer-OnlyΔReceiver-Only CacheGraft Sharer-OnlyΔ
AdversarialQAretrieved41659.4 30.836.841.8+6.0∗34.938.948.3+4.124.530.333.7+5.8∗
oracle — 42.149.359.1+7.2∗44.251.463.9+7.2∗33.942.548.3+8.7∗
SQuADretrieved80068.1 51.859.666.4+7.9∗59.262.570.8+3.2∗48.655.561.1+6.9∗
oracle — 73.079.486.5+6.4∗78.681.889.4+3.169.976.981.2+7.0∗
CovidQAretrieved80053.2 33.136.941.9+3.8∗50.853.859.8+3.021.022.624.4+1.6
oracle — 53.759.161.1+5.4∗71.776.980.0+5.2∗32.635.437.2+2.8
TriviaQAretrieved80054.3 38.141.657.8+3.5∗49.252.474.1+3.1∗45.948.965.5+3.0∗
oracle — 50.854.070.1+3.262.963.485.0+0.561.062.881.5+1.8
QASPERretrieved58336.2 15.315.717.2+0.435.9 33.9 35.5−2.017.718.818.2+1.1
oracle — 34.737.537.9+2.957.460.859.2+3.432.739.034.8+6.3∗
∗𝑝<0.05, exact two-sided paired McNemar over questions.
Table 2: Example outputs of CacheGraft and the receiver-
only Text-RAG baseline. The last two rows are failure cases:
in each, CacheGraft answers with a different span of the
same passage.
Over how many species of trees can be found in the Saxon Garden?(gold: 100)
text-RAG: 15.5×
CacheGraft: 100✓
Where was the Continental Edison Company located?(gold: France)
text-RAG: New York City.×
CacheGraft: The Continental Edison Company was located in France.✓
Who did Rollo sign the treaty of Saint-Clair-sur-Epte with?(gold: King Charles III)
text-RAG: Rollo.×
CacheGraft: Rollo signed the treaty of Saint-Clair-sur-Epte with King Charles III of West
Francia.✓
Where were missing patents later found?(gold: patent archives)
text-RAG: The missing patents were later found in patent archives.✓
CacheGraft: The missing patents were found in the United States, Britain, and Canada.×
Whose writings were widely quoted by the Third Reich?(gold: Luther)
text-RAG: Luther’s writings were widely quoted by the Third Reich.✓
CacheGraft: The writings of Heinrich Himmler were widely quoted by the Third Reich.×
Table 3: Relationship between CacheGraft’s gains and the
opportunity for improvement, all computed in the oracle-
retrieval setting. Headroom is Sharer-Only minus Receiver-
Only under contains. Answer-in-doc is the sharer’s contains
improvement when given the gold passage versus answering
closed-book (no passage). Gain is CacheGraft’s improvement
over Receiver-Only under the phi-4 judge.
Dataset Headroom Answer-in-doc Gain
AdversarialQA+14.4+34.6+7.2
SQuAD+11.3+69.0+6.4
CovidQA+4.4+34.1+5.4
TriviaQA+20.5+40.7+3.2
QASPER+2.0+30.2+2.9Table 4: Query-time cost, batch size 1, greedy decoding with
24 maximum new tokens, warm, over 100 SQuAD questions
on one A100-40GB.
System Latency (ms) GPU Memory (GB) # Tokens
Receiver text-RAG (0.6B)334 1.22 8.2
CacheGraft (0.6B + Cache)5741.2714.3
Sharer text-RAG (4B)652 7.62 12.6
(1)Training custom fuser models tailored for RAG:As
mentioned previously, we consider a single off-the-shelf C2C
fuser model that was not targeted for retrieval tasks. Could
we improve the performance of CacheGraft by training a
fuser model with retrieval-specific training objectives and
datasets? How far can we scale the gap between the sharer
and receiver model before cache fusion fails? Could we even
train the fuser to be more compressible to augment standard
KV cache compression techniques?
(2)Combining with textual document enrichment:Prior
work such as EnrichIndex [ 4] and AutoIndex [ 14] propose to
modify document text to improve RAG performance. How
does text-level document enrichment interplay with the la-
tent space strategy of our method? Is there a text enrichment
method that enhances KV cache translation beyond simply
using the raw documents?
7 Conclusion
Thus far, the idea of translating the KV cache from one LLM to
another possibly very different model has been applied exclusively
to communicating state in real-time multi-agent systems. The key
insight at the heart of this paper is that the same idea can be re-
purposed for RAG by applying KV cache translation on documents
offline between a larger sharer model and a smaller receiver to
enhance the receiver’s online answer quality. In fact, we argue that
RAG might be an evenbetterfit for the Cache-to-Cache fusion
paradigm because running forward passes through both the sharer
and receiver model (as C2C requires) is a negligible cost for offline

CacheGraft: Enhancing Retrieval-Augmented Generation
via Latent-Space Document Enrichment KEIR @ CIKM 2026, November 2026, Rome, Italy
document indexing, but may be limiting in real-time inference set-
tings. We operationalize this insight in the form of CacheGraft, a
prototype RAG system that utilizes C2C fusion to enrich a small
receiver model’s KV cache representation with signal from a larger
model that does not run at query time. Our initial evaluation ex-
tends C2C to zero-shot knowledge-intensive question answering
tasks not present in fuser training where we find that CacheGraft
consistently improves generation quality over the receiver-only text
baseline on datasets where the larger sharer model has headroom
to transfer. Perhaps surprisingly, these positive results on question-
answering come from using an off-the-shelf pre-trained C2C fuser
model that was not tuned for retrieval in any respect. Our findings
also raise a number of interesting questions for future work, which
we outline in Section 6.2, that we hope will drive further work on
latent-space document enrichment for RAG applications.
References
[1]Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya
Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauff-
mann, et al. Phi-4 technical report.arXiv preprint arXiv:2412.08905, 2024.
[2]Max Bartolo, Alastair Roberts, Johannes Welbl, Sebastian Riedel, and Pontus
Stenetorp. Beat the ai: Investigating adversarial human annotation for reading
comprehension.Transactions of the Association for Computational Linguistics,
8:662–678, 2020.
[3]Brian J Chan, Chao-Ting Chen, Jui-Hung Cheng, and Hen-Hsen Huang. Don’t do
rag: When cache-augmented generation is all you need for knowledge tasks. In
Companion proceedings of the ACM on web conference 2025, pages 893–897, 2025.
[4]Peter Baile Chen, Tomer Wolfson, Michael Cafarella, and Dan Roth. Enrichindex:
Using llms to enrich retrieval indices offline.arXiv preprint arXiv:2504.03598,
2025.
[5]Peter Baile Chen, Yi Zhang, Dan Roth, Samuel Madden, Jacob Andreas, and Mike
Cafarella. Log-augmented generation: Scaling test-time reasoning with reusable
computation. InInternational Conference on Learning Representations, volume
2026, pages 3339–3358, 2026.
[6]Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt
Gardner. A dataset of information-seeking questions and answers anchored
in research papers. InProceedings of the 2021 Conference of the North Ameri-
can Chapter of the Association for Computational Linguistics: Human Language
Technologies, pages 4599–4610, 2021.
[7]Lucio M Dery, Zohar Yahav, Henry Prior, Qixuan Feng, Jiajun Shen, and Arthur
Szlam. Latent space communication via kv cache alignment.arXiv preprint
arXiv:2601.06123, 2026.
[8]Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang,
and Yu Wang. Cache-to-cache: Direct semantic communication between large
language models. InInternational Conference on Learning Representations, volume
2026, pages 43130–43158, 2026.
[9]Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A
large scale distantly supervised challenge dataset for reading comprehension.
InProceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.
[10] Yongjia Lei, Nedim Lipka, Zhisheng Qi, Utkarsh Sahu, Koustava Goswami, Franck
Dernoncourt, Ryan A Rossi, and Yu Wang. Rl-index: Reinforcement learning for
retrieval index reasoning.arXiv preprint arXiv:2606.16316, 2026.
[11] Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. Turborag:
Accelerating retrieval-augmented generation with precomputed kv caches for
chunked text. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 6599–6612, 2025.
[12] Dongyang Ma, Yan Wang, and Tian Lan. Block-attention for efficient prefilling. In
International Conference on Learning Representations, volume 2025, pages 63774–
63788, 2025.
[13] Timo Möller, Anthony Reina, Raghavan Jayakumar, and Malte Pietsch. Covid-qa:
A question answering dataset for covid-19. InProceedings of the 1st Workshop on
NLP for COVID-19 at ACL 2020, 2020.
[14] Sam O’Nuallain, Nithya Rajkumar, Ramya Narayanasamy, Hanna Jiang, Shreyas
Chaudhari, and Andrew Drozdov. Autoindex: Learning representation programs
for retrieval.arXiv preprint arXiv:2607.18603, 2026.
[15] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad:
100,000+ questions for machine comprehension of text. InProceedings of the 2016
conference on empirical methods in natural language processing, pages 2383–2392,
2016.[16] Maximillian Rossi, Prajwal Raghunath, and Eugene Wu. Latent cache flow:
Model-to-model communication without text.arXiv preprint arXiv:2605.22863,
2026.
[17] Luohe Shi, Hongyi Zhang, Yao Yao, Zuchao Li, and Hai Zhao. Keep the cost down:
A review on methods to optimize llm’s kv-cache consumption.arXiv preprint
arXiv:2407.18003, 2024.
[18] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy
Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari,
Alexandre Ramé, et al. Gemma 2: Improving open language models at a practical
size.arXiv preprint arXiv:2408.00118, 2024.
[19] Omri Uzan, Ron Polonsky, Douwe Kiela, and Christopher Potts. Document
optimization for black-box retrieval via reinforcement learning.arXiv preprint
arXiv:2604.05087, 2026.
[20] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Pack-
aged resources to advance general chinese embedding, 2023.
[21] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical
report.arXiv preprint arXiv:2505.09388, 2025.
[22] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. Cacheblend: Fast large language model
serving for rag with cached knowledge fusion. InProceedings of the twentieth
European conference on computer systems, pages 94–109, 2025.
[23] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu,
Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging
llm-as-a-judge with mt-bench and chatbot arena.Advances in neural information
processing systems, 36:46595–46623, 2023.