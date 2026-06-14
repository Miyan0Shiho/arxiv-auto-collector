# Uncertainty-Aware Hybrid Retrieval for Long-Document RAG

**Authors**: Hoin Jung, Xiaoqian Wang

**Published**: 2026-06-11 16:30:45

**PDF URL**: [https://arxiv.org/pdf/2606.13550v1](https://arxiv.org/pdf/2606.13550v1)

## Abstract
Retrieval augmented generation (RAG) depends critically on the quality and granularity of retrieved evidence. Large retrieval units preserve context but often introduce irrelevant content, which can dilute answer bearing evidence and worsen long context utilization. Fine-grained units are more compact, but they may be difficult to retrieve reliably because short chunks can lack semantic, lexical, or bridging cues needed to match the query. We propose Uncertainty-aware Multi-Granularity RAG (UMG-RAG), a training-free hybrid retrieval framework that treats chunk granularity as query-specific reliability estimation. Instead of training a new retriever or modifying the generator, UMG-RAG uses existing dense and sparse retrievers as complementary experts across multiple chunk granularities. For each query, it converts each expert-granularity score list into an evidence distribution, estimates reliability from distribution entropy, and fuses candidates according to query-specific semantic, lexical, and granularity confidence. We further introduce UMGP-RAG, a parent promotion variant that uses fine-grained hits to locate relevant evidence while returning broader non-redundant parent chunks for local coherence. Experiments on question answering benchmarks show that uncertainty-aware fusion and parent promotion improve generation quality while maintaining a lightweight, plug-and-play retrieval pipeline.

## Full Text


<!-- PDF content starts -->

Uncertainty-Aware Hybrid Retrieval for Long-Document RAG
Hoin Jung, Xiaoqian Wang
Elmore Family School of Electrical and Computer Engineering
Purdue University
West Lafayette, IN 47907
{jung414, joywang}@purdue.edu
Abstract
Retrieval augmented generation (RAG) de-
pends critically on the quality and granularity
of retrieved evidence. Large retrieval units pre-
serve context but often introduce irrelevant con-
tent, which can dilute answer bearing evidence
and worsen long context utilization. Fine-
grained units are more compact, but they may
be difficult to retrieve reliably because short
chunks can lack semantic, lexical, or bridg-
ing cues needed to match the query. We pro-
poseUncertainty-aware Multi-Granularity
RAG (UMG-RAG), a training-free hybrid
retrieval framework that treats chunk granu-
larity as query-specific reliability estimation.
Instead of training a new retriever or modi-
fying the generator, UMG-RAG uses existing
dense and sparse retrievers as complementary
experts across multiple chunk granularities. For
each query, it converts each expert-granularity
score list into an evidence distribution, esti-
mates reliability from distribution entropy, and
fuses candidates according to query-specific
semantic, lexical, and granularity confidence.
We further introduceUMGP-RAG, a par-
ent promotion variant that uses fine-grained
hits to locate relevant evidence while returning
broader non-redundant parent chunks for local
coherence. Experiments on question answer-
ing benchmarks show that uncertainty-aware
fusion and parent promotion improve genera-
tion quality while maintaining a lightweight,
plug-and-play retrieval pipeline.
1 Introduction
Retrieval augmented generation (RAG) (Lewis
et al., 2020) has become a standard framework for
grounding large language models in external knowl-
edge. In a typical pipeline, a retriever first selects
relevant text units from a database, and a generator
then produces an answer conditioned on the re-
trieved context. Although this framework reduces
the need for fully parametric knowledge storage,its effectiveness depends heavily on the granular-
ity and precision of retrieval (Zhang et al., 2025b).
When retrieved units are too large, the generator re-
ceives a mixture of relevant and irrelevant content.
This can push the model into long context regimes
where useful evidence may be diluted or placed in
positions that are poorly utilized by the reader, of-
ten called the lost-in-the-middle phenomenon (Liu
et al., 2023). When retrieved units are too small,
such as isolated sentences, the retrieved evidence
may be compact but difficult to recover reliably,
because short units may lack semantic cues, entity
aliases, or bridging context needed for matching
the query. We frame this tension as a retrieval-stage
granularity tradeoff between coarse units that pre-
serve context but introduce noise and fine-grained
units that provide compact evidence but risk being
missed or fragmented.
Figure 1 illustrates this tradeoff. Coarse retrieval
can return a long context that contains the answer
but also includes substantial irrelevant information,
making the generator less likely to use the answer-
bearing evidence and potentially reintroducing po-
sitional bias in long-context reading. Fine-grained
retrieval can avoid this noise, but the exact local
evidence may be missed when the query requires
surrounding information that is not present in the
small unit. This suggests that neither coarse nor
fine retrieval units are universally preferable. In-
stead, an effective retriever should be able to exploit
both: fine-grained units are useful when they pro-
vide high-confidence evidence, while coarser units
are useful when local context is needed to make the
evidence retrievable or coherent.
Existing solutions often address only one side of
this tradeoff. Long-context interventions calibrate
positional attention (Hsieh et al., 2024; Jung and
Wang, 2026), adjust positional encodings (Zhang
et al., 2024), reorder documents (Wang et al., 2025),
or compress prompts (Jiang et al., 2024a) to reduce
the effect of lost-in-the-middle. These methods
1arXiv:2606.13550v1  [cs.AI]  11 Jun 2026

Query: Which city will host the first match of the 2026 FIFA World Cup?
RetrieverCoarse RetrievalThe2026 FIFA World Cup will be the 23rdFIFA World Cup, thequadrennial international men'ssoccerchampionship ..... First match(es) will be played: June 11, 2026. June 11, 20261:00 p.m. UTC−6 Mexico Match 1 South Africa. Estadio Azteca, Mexico City.......Retrieved Long Context
RetrieverChunk 1:The2026 FIFA World Cup will be ...Fine-grained  RetrievalRetrieved Short UnitsChunk 2:The United 2026 bid ...Chunk 3:First match(es) will be played ...Chunk 4: A player or team official is ...Reader (Generator)
lost-in-the-middleirrelevant context  dilutes evidence
ReaderStruggles
Insufficient Matching CuesReader (Generator)Chunk 1Chunk 2Chunk 4Exact Evidence Missedfails to retrieve the exact local context
RetrieverStrugglesFigure 1:Motivation of retrieval granularity tradeoff in RAG.Coarse retrieval units preserve document-level
context but may introduce irrelevant content that dilutes answer-bearing evidence. Fine-grained units are more
compact, but they may lack sufficient semantic or lexical cues, causing the retriever to miss the exact local evidence.
are valuable, but they generally assume that the
relevant evidence has already been retrieved and
placed somewhere in the prompt. As a result, they
do not directly resolve the retrieval-stage decision
of whether the system should retrieve a compact
local chunk, a broader parent context, or a combina-
tion of both. Advanced RAG systems address frag-
mentation by constructing knowledge graphs (Guo
et al., 2024), retrieving at multiple abstraction lev-
els (Zheng et al., 2025), or inferring sentence-level
relations (Tao et al., 2025). However, these meth-
ods often require graph construction, abstraction-
level indexing, or post-retrieval sentence linking,
and they do not explicitly estimate query-specific
reliability across retrieval sources and granularities.
In this work, we propose Uncertainty-aware
Multi-Granularity RAG (UMG-RAG), a training-
free hybrid retrieval framework for compact long-
document RAG. Instead of training a new retriever
or modifying the generator, UMG-RAG uses exist-
ing dense and sparse retrievers as complementary
experts. Dense retrieval captures semantic simi-
larity, while sparse retrieval captures lexical and
entity-level matching. Each expert retrieves candi-
dates across multiple chunk granularities, and the
method estimates the confidence of each expert–granularity pair from the sharpness of its candidate
score distribution. A peaked distribution indicates
that the expert has a clear retrieval preference for
the query, while a flat distribution indicates higher
retrieval uncertainty. UMG-RAG uses this con-
fidence to fuse dense and sparse evidence into a
single ranked list, as shown in Figure 2.
We further introduce UMGP-RAG, a parent-
promotion extension that improves local coherence
without giving up fine-grained retrieval precision.
When a highly ranked chunk comes from a fine-
grained unit, UMGP-RAG promotes it to a broader
parent chunk and removes redundant overlaps. This
design allows fine-grained retrieval to act as a pre-
cise locator of relevant regions, while the final con-
text returned to the generator remains locally co-
herent. Importantly, the framework is training-free
and can be applied on top of existing dense and
sparse retrieval backbones.
Our contributions are as follows:
•We formalize a retrieval granularity tradeoff in
long-document RAG, where coarse units im-
prove contextual coverage but add distracting
content, while fine-grained units are compact
but harder to retrieve reliably.
•We propose UMG-RAG, a training-free
2

uncertainty-aware hybrid retrieval method that
estimates query-specific reliability for each ex-
pert and granularity pair, and further introduce
UMGP-RAG, a parent-promotion extension
that converts fine-grained hits into broader
non-redundant contexts for generation.
•We evaluate the proposed methods on long-
document RAG benchmarks with multiple
dense retrievers and generators, showing that
uncertainty-aware fusion and parent promo-
tion improve generation quality while keeping
the retrieval pipeline lightweight.
2 Related Work
2.1 Long Context Interventions for Lost in the
Middle
Prior work has shown that language models often
fail to use relevant evidence when it appears in the
middle of a long prompt (Liu et al., 2024). Recent
methods address this issue mainly at the reader
or prompt level. FITM (Hsieh et al., 2024) cali-
brates positional attention bias using dummy docu-
ments. MS-PoE (Zhang et al., 2024) modifies ro-
tary positional encodings to improve long-context
utilization. PINE (Wang et al., 2025) reassigns
document positions based on content importance.
LongLLMLingua (Jiang et al., 2024a) compresses
long prompts through question-aware pruning and
reordering. These approaches are complementary
to ours, but they generally assume that relevant evi-
dence has already been retrieved. In contrast, we
intervene earlier in the RAG pipeline by construct-
ing a compact, high-signal retrieval context before
generation.
2.2 Retrieval Granularity and Hybrid
Retrieval in RAG
Another line of work modifies the retrieval unit
or retrieval structure. LongRAG (Jiang et al.,
2024b) increases the retrieval unit size to preserve
document-level context, while LightRAG (Guo
et al., 2024) constructs a knowledge graph to re-
trieve connected entities and relations. MAL-RAG
(Zheng et al., 2025) retrieves from multiple ab-
straction levels, including summaries, paragraphs,
and sentences. SAKI-RAG (Tao et al., 2025) uses
sentence-level attention knowledge to recover logi-
cally connected evidence. These methods highlight
the importance of retrieval granularity, but they of-
ten rely on predefined retrieval structures, graphconstruction, or additional evidence-connection
mechanisms.
Hybrid dense-sparse retrieval is also widely used
in RAG because dense retrievers capture seman-
tic similarity while sparse retrievers are effective
for lexical and entity-level matching. Prior RAG
systems such as HyPA-RAG (Kalra et al., 2024)
combine dense and sparse retrieval, often using
Reciprocal Rank Fusion as a simple fusion rule
(Cormack et al., 2009). However, fixed fusion
does not account for query-specific reliability. A
lexical query may favor sparse evidence, while a
paraphrastic query may favor dense evidence, and
both may require different chunk granularities. Our
method differs by estimating confidence for each
expert-granularity pair from its score distribution,
allowing dense-sparse fusion and retrieval granu-
larity to adapt per query without training a new
retriever or modifying the generator.
3 Proposed Method
3.1 Overview and Problem Setup
We propose an uncertainty-aware hybrid retrieval
framework for retrieval-augmented generation. Un-
like training-based methods, our framework does
not introduce a new retriever, train an adapter, or
modify the generator. Instead, it uses existing pre-
trained dense and sparse retrievers as complemen-
tary retrieval experts and adaptively selects evi-
dence across multiple chunk granularities without
any training.
As motivated in Figure 1, RAG faces a retrieval
granularity tradeoff. Coarse units preserve local
context but may introduce distractors, while fine-
grained units are compact but can be harder to re-
trieve reliably. We address this tradeoff by retriev-
ing from multiple granularities and estimating, for
each query, which retrieval source and granularity
are reliable.
LetD={d 1, . . . , d N}denote a document cor-
pus and let qdenote an input query. Each docu-
ment is segmented into overlapping sentence-level
chunks with multiple granularities. We define the
granularity set as G={2,4,8,16,32}, where
g∈ G denotes the number of sentences in a re-
trieval unit. Let Ugbe the set of chunks generated
at granularity g, and let U=S
g∈GUgbe the com-
plete multi-granularity retrieval space.
We use two retrieval experts,
E={dense_retriever,sparse_retriever}.
3

The2026 FIFA World Cup will be hosted ...S1Mexico, the United States, and Canada ...S2The opening match ... Estadio Azteca...S3Estadio Azteca is located inMexico City ...S4The tournament willbeginonJune 11 ...S5Mexico was placed in Group A ...S6Several venues across North America ...S7The event will feature 48 teams ...S8Source DocumentWhich city will host the first match of the 2026 FIFA World Cup?
Query
Chunk GranularityS1S2S3Dense Expert(semantic matching)CandidateChunkScoreChunk 2 (S3 –S4)0.86Chunk 4 (S2 –S5)0.63Chunk 8 (S1 –S8)0.41...Sparse Expert(lexical/entity matching)CandidateChunkScoreS4S5S6...S7S8Chunk 2S1S2S3S4S5S6S7S8Chunk 4S1S2S3S4S5S6S7S8Chunk 8Chunk 2 (S3 –S4)0.80Chunk 4 (S2 –S5)0.39Chunk 8 (S1 –S8)0.15Uncertainty-awareExpert x Granularity Pair WeightingExpert x Granularity PairWeightChunk 2 (S3 –S4)DenseChunk 4 (S2 –S5)DenseChunk 8 (S1 –S8)Dense...Chunk 2 (S3 –S4)SparseChunk 4 (S2 –S5)SparseChunk 8 (S1 –S8)Sparse...Fused Candidates by UMG-RAG scoreChunk 2 (S3 –S4)...0.95Chunk 4 (S2 –S5)0.51Chunk 8 (S1 –S8)0.22
sharper score distribution → higher confidenceFigure 2:Overview of uncertainty-aware multi-granularity RAG.A source document is segmented into
overlapping chunks of different granularities. Existing dense and sparse experts retrieve candidates from these
granularities. UMG-RAG estimates query-specific confidence for each expert–granularity pair from the sharpness
of its score distribution, then fuses the candidates into a single ranked list.
The dense expert captures semantic similarity,
while the sparse expert captures lexical and entity-
level matching. The framework is model-agnostic
and only requires each expert to score a query-
chunk pair.
For each expert e∈ E and granularity g∈ G , the
retriever assigns a raw relevance score
se,g(q, u)∈R, u∈ U g.
A central challenge is that these scores are not di-
rectly comparable. Dense scores, sparse scores,
and scores from different chunk granularities may
have different scales and distributions. Therefore,
using a fixed interpolation weight between dense
and sparse retrieval is suboptimal. A fixed weight
assumes that the same retriever and granularity are
equally reliable for every query, although lexical,
entity-centric, and paraphrastic queries may require
different retrieval behavior.
Therefore, instead of combining raw scores with
a fixed dense–sparse weight, we estimate the query-
specific confidence of each expert–granularity pair.
This confidence is derived from the uncertainty of
its candidate score distribution and determines how
much that retrieval source contributes to the final
context.
3.2 Uncertainty-Aware Multi-Granularity
RAG
We now describe UMG-RAG. As shown in Fig-
ure 2, the method retrieves candidates from each
expert-granularity pair, converts their scores intonormalized evidence distributions, estimates uncer-
tainty using entropy, and fuses candidates into a
single ranked list.
For a query q, each expert–granularity pair (e, g)
first retrieves an intermediate candidate set
Ce,g(q) = TopMu∈Ugse,g(q, u),
where Mis the number of candidates retained from
each expert–granularity pair before fusion. In our
implementation, we set M= 100 . This value
is intentionally larger than the final retrieval size
Kso that the fusion stage can compare a suffi-
ciently diverse set of candidates from both dense
and sparse experts across granularities. At the same
time, M= 100 keeps the intermediate pool com-
putationally manageable, since the final context is
selected only after uncertainty-aware fusion.
To compare score distributions across experts
and granularities, we normalize the raw scores
within each candidate set:
˜se,g(q, u) =se,g(q, u)−µ e,g(q)
σe,g(q) +ϵ,
where µe,g(q)andσe,g(q)are the mean and stan-
dard deviation of the scores in Ce,g(q), and ϵis a
small constant for numerical stability.
We then convert the normalized scores into a
score-induced evidence distribution:
pe,g(u|q,C e,g) =exp(˜s e,g(q, u))P
v∈Ce,g(q)exp(˜s e,g(q, v)).
This distribution is defined only over the retrieved
candidate set. It should not be interpreted as a cal-
ibrated probability over the full corpus. Instead,
4

it represents how expert eat granularity gconcen-
trates its evidence on candidate urelative to its
other candidates.
This distribution provides a query-specific relia-
bility signal. If an expert–granularity pair assigns
most probability mass to a small number of candi-
dates, it has a clear retrieval preference for the cur-
rent query. If the distribution is nearly uniform, the
expert–granularity pair cannot clearly distinguish
among its candidates. We quantify this ambiguity
using normalized entropy:
He,g(q) =−P
u∈Ce,g(q)pe,g(u|q,C e,g) logp e,g(u|q,C e,g)
log|C e,g(q)|.
The entropy value lies in [0,1] . Low entropy indi-
cates concentrated evidence and therefore low re-
trieval uncertainty. High entropy indicates diffuse
evidence and therefore high retrieval uncertainty.
We convert uncertainty into confidence:
ce,g(q) = 1−H e,g(q).
The confidence values are then normalized across
all experts and granularities:
we,g(q) =ce,g(q)P
e′∈EP
g′∈Gce′,g′(q) +ϵ.
This weighting makes the retrieval process
uncertainty-aware by assigning larger weights to
expert–granularity pairs with lower entropy and
smaller weights to pairs with higher entropy.
The final evidence probability of a chunk is de-
fined as the confidence-weighted mixture of candi-
date distributions:
P(u|q) =P
e∈EP
g∈Gwe,g(q)p e,g(u|q,C e,g)I[u∈ C e,g(q)].
Here, P(u|q) measures the evidence mass as-
signed to chunk uafter accounting for both re-
triever confidence and granularity confidence.
Because the generator should receive compact
context, we rank chunks by evidence utility:
R(u|q) =P(u|q)p
ℓ(u).
Here, ℓ(u) is the approximate token length of u.
The square-root penalty mildly favors compact
chunks while still allowing longer chunks when
they receive strong evidence support.
The final retrieval context is constructed by se-
lecting the topKchunks according toR(u|q):
SK(q) = TopKu∈U candR(u|q),Ucand=S
e∈ES
g∈GCe,g(q)is the fused candidate
pool. The selected chunks in SK(q)are concate-
nated and used as the retrieval context for the gen-
erator. Following common top- KRAG evaluation
practice, we set K= 5 in the main experiments
and use the same value across all retrieval methods
for a fair comparison.
3.3 Parent Promotion with Overlap-Aware
Deduplication
We further extend UMG-RAG with parent promo-
tion, denoted UMGP-RAG. The goal is to use fine-
grained chunks as precise retrieval signals while
returning a broader local unit to the generator.
Letπ8(u)denote the parent chunk of uat granu-
larity 8. In our implementation, retrieved chunks
from granularity 2or4are promoted to their corre-
sponding parent chunk:
ρ(u) =(
π8(u), u∈ U 2∪ U4,
u,otherwise,
where ρ(u) is the unit returned after parent promo-
tion. Thus, fine-grained chunks serve as retrieval
locators, while the returned parent chunks provide
the neighboring sentences needed for local coher-
ence.
After applying ρ(·), multiple fine-grained can-
didates may map to the same parent chunk. We
merge such duplicates using the same bounded evi-
dence aggregation used in the implementation. Let
M(v) ={u∈ U cand :ρ(u) =v} be the set
of candidates mapped to promoted chunk v. We
define
A(v|q) = 1−Y
u∈M(v)(1−P(u|q)).
This aggregation increases the evidence score of a
parent chunk when multiple retrieved candidates
support it, while keeping the score bounded in [0,1] .
Promoted candidates are then ranked by
RP(v|q) =A(v|q)p
ℓ(v).
To avoid returning redundant contexts, we remove
promoted candidates that substantially overlap with
already selected chunks. Let span(v) denote the
sentence span covered by candidate chunk v. For a
candidate vand a previously selected chunk s, we
define span overlap as
overlap(v, s) =|span(v)∩span(s)|
min(|span(v)|,|span(s)|).
5

A candidate is skipped if at least 75% of the shorter
span overlaps with any previously selected chunk.
This rule removes near-duplicate or nested con-
texts while allowing partially overlapping chunks
to remain when they may provide complementary
evidence.
The final context is constructed by selecting the
topKnon-redundant promoted chunks:
SP
K(q) = TopKDedupv∈{ρ(u):u∈U cand}RP(v|q),
where Ucand is the fused candidate pool produced
by UMG-RAG. The selected promoted chunks are
concatenated and used as the final retrieval con-
text for the generator. The detailed algorithm is
introduced in Algorithm 1 in Appendix A.
4 Implementation Details
4.1 Models and Datasets
We evaluate our RAG pipeline on Natural Ques-
tions (NQ) (Kwiatkowski et al., 2019) and Hot-
PotQA (Yang et al., 2018), following prior long-
document RAG evaluation settings (Li et al., 2024;
Petroni et al., 2020). NQ evaluates factual ques-
tion answering over external knowledge, while Hot-
PotQA requires more compositional evidence re-
trieval and provides a complementary testbed for
multi-hop reasoning.
For each dataset, we construct a task-specific re-
trieval corpus from the KILT (Petroni et al., 2020)
validation split. The corpus contains all Wikipedia
documents appearing in the gold provenance anno-
tations, plus 10,000 randomly sampled distractor
Wikipedia documents that do not overlap with the
positive set. This results in 14,274 documents for
NQ, consisting of 4,274 positive documents and
10,000 distractors, and 20,519 documents for Hot-
PotQA, consisting of 10,519 positive documents
and 10,000 distractors. We use the same corpus
for all retrieval methods to ensure that performance
differences come from the retrieval strategy rather
than the candidate document pool.
For generators, we useQwen2.5-3B-Instruct
(Yang et al., 2024) andLlama-3.2-3B-Instruct
(Grattafiori et al., 2024). For dense retrievers, we
evaluate BERT (Devlin et al., 2019), BGE-M3
(Chen et al., 2024), andQwen3-Embedding-4B
(Zhang et al., 2025a). For sparse retrieval, we use
SPLADEv3 (Lassance et al., 2024). All methods
are evaluated without task specific training or fine-
tuning, so performance differences reflect retrieval
and context construction rather than updates to the
retriever or generator parameters.4.2 Evaluation Metrics
We evaluate both retrieval quality and generation
quality. For retrieval, we report answer recall at five
retrieved chunks (AR@5). A retrieval is counted as
successful if the normalized gold answer appears
in any of the top five retrieved contexts.
For generation, we report token-level F1 and
answer recall (AR). F1 measures token overlap
between the generated answer and the gold an-
swer after normalization. Generation AR measures
whether the normalized gold answer appears in the
generated response. We follow standard question
answering normalization by lowercasing text, re-
moving punctuation, and removing articles. Higher
values indicate better performance for all metrics.
For readability, the main tables report mean per-
formance. We additionally report standard devia-
tions estimated from 1,000 bootstrap resamples in
Appendix C.
4.3 Baselines
We compare our methods against standard and long
context RAG baselines. The standard RAG base-
line retrieves from a fixed retrieval unit and passes
the retrieved text directly to the generator. Summa-
rized RAG compresses retrieved documents using
the Falcon summarization model (Falconsai, 2023).
REPLUG (Shi et al., 2024) represents a retrieval-
augmented black-box language modeling baseline
that combines retrieved documents without updat-
ing the generator. LongLLMLingua (Jiang et al.,
2024a) represents prompt compression based con-
text selection, while MS-PoE (Zhang et al., 2024)
and PINE (Wang et al., 2025) represent long con-
text reader side interventions for mitigating posi-
tion bias. LongRAG (Jiang et al., 2024b) uses
larger retrieval units to preserve broader context.
We also include a simple hybrid retrieval base-
line. Following common hybrid retrieval practice
in RAG (Kalra et al., 2024), we combine dense
and sparse retrieval results using Reciprocal Rank
Fusion (RRF) (Cormack et al., 2009), denoted Hy-
brid. This baseline tests whether fixed dense-sparse
fusion is sufficient. In contrast, our UMG-RAG
estimates query specific confidence for each expert
and granularity before fusion. UMGP-RAG fur-
ther promotes fine grained hits to parent chunks
and removes redundant overlaps. All comparison
methods are used in a plug-and-play setting without
additional training, to isolate the effect of retrieval
strategy.
6

Retriever MethodRetrieval
AR@5↑Qwen2.5-3B-Instruct Llama-3.2-3B-Instruct
F1↑AR↑ F1↑AR↑
SPLADEv3 RAG 0.7994 0.4745 0.4529 0.4923 0.4741
BERTRAG
0.44770.2270 0.2013 0.2414 0.2178
Summarized RAG 0.1704 0.1491 0.1972 0.1703
LongLLMLingua 0.2304 0.2080 0.2573 0.2429
MS-PoE 0.2239 0.1974 0.2435 0.2192
REPLUG 0.1373 0.1174 0.2254 0.1896
PINE 0.1915 0.1843 0.2084 0.2097
LongRAG 0.4441 0.2093 0.1851 0.2318 0.2094
Hybrid 0.7011 0.4265 0.4018 0.4504 0.4293
UMG-RAG (Ours) 0.6852 0.4350 0.4110 0.4514 0.4304
UMGP-RAG (Ours) 0.7797 0.4572 0.4307 0.4811 0.4533
BGE-M3RAG
0.85620.4407 0.3994 0.4941 0.4593
Summarized RAG 0.3243 0.2880 0.3520 0.3127
LongLLMLingua 0.4581 0.4184 0.4631 0.4462
MS-PoE 0.4432 0.4015 0.5005 0.4709
REPLUG 0.4701 0.4216 0.4962 0.4459
PINE 0.4547 0.4420 0.4452 0.4603
LongRAG 0.9101 0.4598 0.4219 0.5113 0.4910
Hybrid 0.8241 0.4927 0.4727 0.4986 0.4766
UMG-RAG (Ours) 0.8023 0.4809 0.4593 0.4873 0.4716
UMGP-RAG (Ours) 0.8759 0.5052 0.4794 0.5128 0.4878
Qwen3-
Embedding-4BRAG
0.86010.4427 0.3990 0.4939 0.4621
Summarized RAG 0.3333 0.2915 0.3537 0.3162
LongLLMLingua 0.4459 0.4057 0.4561 0.4346
MS-PoE 0.4481 0.4064 0.5046 0.4720
REPLUG 0.4796 0.4367 0.5120 0.4642
PINE 0.4628 0.4466 0.4291 0.4438
LongRAG 0.9253 0.4744 0.4329 0.5145 0.4949
Hybrid 0.8114 0.4803 0.4621 0.4916 0.4741
UMG-RAG (Ours) 0.7949 0.4807 0.4582 0.4974 0.4790
UMGP-RAG (Ours) 0.8745 0.4961 0.4748 0.5220 0.4984
Table 1:Results on the Natural Questions dataset.The best retrieval and generation scores within each retriever
block are highlighted inbold, and the second best scores are underlined .
5 Results and Analysis
Tables 1 and 2 report retrieval and generation
performance on NQ and HotPotQA. We report
retrieval answer recall at five retrieved chunks
(AR@5), and generation performance using F1 and
answer recall (AR) with two instruction tuned gen-
erators. Across most retriever and generator com-
binations, UMGP-RAG achieves the strongest or
highly competitive generation performance. This
result supports the central hypothesis of this work:
the best RAG context is not necessarily the longest
context or the context with the highest raw retrieval
hit rate, but the context that is compact, locally co-
herent, and aligned with the query. In several cases,
standard dense retrieval or LongRAG can achieve
competitive retrieval AR@5 because the answer
string appears somewhere in a long retrieved pas-sage. However, such passages may contain sub-
stantial irrelevant content and can reintroduce posi-
tional bias when the answer-bearing span appears
in a less favorable location, making it harder for the
generator to identify and use the correct evidence.
UMGP-RAG instead uses fine-grained retrieval
to locate precise evidence and parent promotion to
return a coherent local context, reducing distractor
content and improving downstream generation.
Although the BERT-based variants do not out-
perform SPLADE-only, they still improve over
other BERT-based RAG baselines. This result
is expected because BERT is a weaker dense
retriever than recent embedding models, while
SPLADE provides strong lexical matching for
answer-bearing spans. The goal of UMG-RAG is
not to replace a strong sparse retriever, but to adap-
tively combine dense and sparse evidence when
7

Retriever MethodRetrieval
AR@5↑Qwen2.5-3B-Instruct Llama-3.2-3B-Instruct
F1↑AR↑ F1↑AR↑
SPLADEv3 RAG 0.6043 0.4613 0.3850 0.4798 0.3966
BERTRAG
0.36380.2599 0.2087 0.2803 0.2279
Summarized RAG 0.2275 0.1864 0.2469 0.2025
LongLLMLingua 0.2833 0.2271 0.3021 0.2362
MS-PoE 0.2612 0.2112 0.2860 0.2339
REPLUG 0.1798 0.1466 0.2560 0.1984
PINE 0.2424 0.2039 0.2557 0.2180
LongRAG 0.3230 0.2299 0.1861 0.2625 0.2104
Hybrid 0.5207 0.4081 0.3311 0.4245 0.3479
UMG-RAG (Ours) 0.5134 0.4163 0.3416 0.4223 0.3470
UMGP-RAG (Ours) 0.5830 0.4453 0.3645 0.4460 0.3655
BGE-M3RAG
0.78110.4613 0.3798 0.4913 0.4048
Summarized RAG 0.3856 0.3161 0.4109 0.3366
LongLLMLingua 0.4706 0.3807 0.4668 0.3771
MS-PoE 0.4622 0.3805 0.5001 0.4123
REPLUG 0.3941 0.3150 0.4287 0.3362
PINE 0.4442 0.3787 0.4409 0.3941
LongRAG 0.7457 0.4658 0.3845 0.4799 0.3984
Hybrid 0.6471 0.4775 0.3975 0.4986 0.4175
UMG-RAG (Ours) 0.6273 0.4749 0.3957 0.4827 0.4018
UMGP-RAG (Ours) 0.7002 0.5034 0.4216 0.5133 0.4275
Qwen3-
Embedding-4BRAG
0.79710.4883 0.4041 0.4991 0.4132
Summarized RAG 0.3976 0.3261 0.4315 0.3546
LongLLMLingua 0.4603 0.3693 0.4644 0.3738
MS-PoE 0.4948 0.4073 0.5172 0.4295
REPLUG 0.4075 0.3252 0.4343 0.3411
PINE 0.4621 0.3980 0.4463 0.4027
LongRAG 0.7280 0.4524 0.3748 0.4765 0.3961
Hybrid 0.6091 0.4626 0.3837 0.4709 0.3904
UMG-RAG (Ours) 0.5845 0.4622 0.3820 0.4752 0.3929
UMGP-RAG (Ours) 0.6761 0.4919 0.4071 0.5068 0.4200
Table 2:Results on the HotPotQA dataset.The best retrieval and generation scores within each retriever block are
highlighted inbold, and the second best scores are underlined .
both sources are available. The stronger perfor-
mance with BGE-M3 and Qwen3-Embedding-4B
further suggests that the proposed framework bene-
fits from stronger dense experts.
Ablation study.The comparison between UMG-
RAG and UMGP-RAG further shows the value
of parent promotion. UMG-RAG evaluates the
effect of uncertainty aware dense-sparse fusion
across granularities, while UMGP-RAG adds par-
ent promotion and overlap aware deduplication.
The consistent improvement of UMGP-RAG in-
dicates that adaptive fusion is most effective when
fine-grained hits are converted into coherent par-
ent contexts. This supports our design choice of
using small chunks for retrieval signals and parent
chunks for final context construction.6 Conclusion
We introduce UMG-RAG, a training-free
uncertainty-aware hybrid retrieval framework for
compact long document RAG. The method uses
existing dense and sparse retrievers across multiple
chunk granularities and estimates query specific
confidence from candidate score distributions. We
further propose UMGP-RAG, which promotes
fine grained hits to parent chunks and removes
redundant overlaps to improve local coherence.
Experiments on NQ and HotPotQA show that
uncertainty-aware fusion and parent promotion
improve generation quality while preserving
a lightweight pipeline. These results suggest
that effective RAG requires retrieving not only
answer-containing passages, but compact, coherent
contexts that the generator can use reliably.
8

Limitations
Retrieval preprocessing cost.UMG-RAG and
UMGP-RAG retrieve candidates across multiple
granularities using both dense and sparse retriev-
ers. As shown in the computational cost analysis
in Appendix B, this increases retrieval preprocess-
ing cost compared with single-granularity RAG
baselines. This overhead is the main cost of the
proposed framework. However, the additional com-
putation occurs before generation and is used to
construct a more compact context, which can re-
duce generation time and memory usage.
Dependence on existing retrieval experts.The
proposed framework is training-free and does not
update the dense retriever, sparse retriever, or gen-
erator. This makes the method lightweight and
easy to apply, but it also means that its perfor-
mance depends on the quality of the underlying
retrieval experts. If both dense and sparse retriev-
ers fail to assign useful scores to answer-bearing
chunks, uncertainty-aware fusion cannot fully re-
cover the missing evidence. Thus, UMG-RAG is
best understood as an adaptive fusion and context
construction framework rather than a replacement
for strong retrieval models.
Ethical Considerations
This work aims to improve retrieval and context
construction for RAG systems. The proposed
method does not introduce new training data, does
not finetune the retriever or generator, and does not
require collecting user information. However, it
relies on external corpora and pretrained retrieval
and generation models, which may contain fac-
tual errors, outdated information, or social biases.
Improving retrieval quality may increase the like-
lihood that a model uses retrieved evidence, but
it does not guarantee that the final generated an-
swer is correct, unbiased, or safe. In real-world
deployments, corpus curation, privacy protection,
and output monitoring remain necessary.
References
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Gordon V Cormack, Charles LA Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperformscondorcet and individual rank learning methods. In
Proceedings of the 32nd international ACM SIGIR
conference on Research and development in informa-
tion retrieval, pages 758–759.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. InProceedings of the 2019 conference of the
North American chapter of the association for com-
putational linguistics: human language technologies,
volume 1 (long and short papers), pages 4171–4186.
Falconsai. 2023. Text summarization model.
https://huggingface.co/Falconsai/text_
summarization.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2(3).
Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li,
Zifeng Wang, Long Le, Abhishek Kumar, James
Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Kr-
ishna, and 1 others. 2024. Found in the middle: Cali-
brating positional attention bias improves long con-
text utilization. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 14982–
14995.
Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng
Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024a.
Longllmlingua: Accelerating and enhancing llms in
long context scenarios via prompt compression. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 1658–1677.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024b.
Longrag: Enhancing retrieval-augmented gener-
ation with long-context llms.arXiv preprint
arXiv:2406.15319.
Hoin Jung and Xiaoqian Wang. 2026. The cost
of context: Mitigating textual bias in multimodal
retrieval-augmented generation.arXiv preprint
arXiv:2605.05594.
Rishi Kalra, Zekun Wu, Ayesha Gulley, Airlie Hilliard,
Xin Guan, Adriano Koshiyama, and Philip Colin Tre-
leaven. 2024. Hypa-rag: A hybrid parameter adap-
tive retrieval-augmented generation system for ai le-
gal and policy applications. InProceedings of the 1st
workshop on customizable nlp: Progress and chal-
lenges in customizing nlp for a domain, application,
group, or individual (customnlp4u), pages 237–256.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
9

Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Carlos Lassance, Hervé Déjean, Thibault Formal, and
Stéphane Clinchant. 2024. Splade-v3: New baselines
for splade.arXiv preprint arXiv:2403.06789.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei,
and Michael Bendersky. 2024. Retrieval augmented
generation or long-context llms? a comprehensive
study and hybrid approach. InProceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing: Industry Track, pages 881–
893.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2023. Lost in the middle: How lan-
guage models use long contexts.arXiv preprint
arXiv:2307.03172.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick
Lewis, Majid Yazdani, Nicola De Cao, James Thorne,
Yacine Jernite, Vladimir Karpukhin, Jean Mail-
lard, and 1 others. 2020. Kilt: a benchmark for
knowledge intensive language tasks.arXiv preprint
arXiv:2009.02252.
Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-
joon Seo, Richard James, Mike Lewis, Luke Zettle-
moyer, and Wen-tau Yih. 2024. Replug: Retrieval-
augmented black-box language models. InProceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers), pages 8371–8384.
Wenyu Tao, Xiaofen Xing, Zeliang Li, and Xiangmin
Xu. 2025. SAKI-RAG: Mitigating context fragmen-
tation in long-document RAG via sentence-level at-
tention knowledge integration. InProceedings of
the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 1195–1213, Suzhou,
China. Association for Computational Linguistics.
Ziqi Wang, Hanlin Zhang, Xiner Li, Kuan-Hao Huang,
Chi Han, Shuiwang Ji, Sham M. Kakade, Hao Peng,and Heng Ji. 2025. Eliminating position bias of lan-
guage models: A mechanistic approach. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and
40 others. 2024. Qwen2 technical report.arXiv
preprint arXiv:2407.10671.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025a. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Zhengxuan Zhang, Yin Wu, Yuyu Luo, and Nan Tang.
2025b. Fine-grained retrieval-augmented genera-
tion for visual question answering.arXiv preprint
arXiv:2502.20964.
Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao,
Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, and
Zhangyang Wang. 2024. Found in the middle: How
language models use long contexts better via plug-
and-play positional encoding.Advances in Neural
Information Processing Systems, 37:60755–60775.
Zheng Zheng, Xinyi Ni, and Pengyu Hong. 2025. Mul-
tiple abstraction level retrieve augment generation.
arXiv preprint arXiv:2501.16952.
10

A Detailed Algorithm of UMGP-RAG
Algorithm 1UMGP-RAG
Require: Query q, experts E, granularities G,M= 100 ,
K= 5, overlap cutoff0.75
Ensure:Selected context chunksS
1:U cand← ∅
2:for all(e, g)∈ E × Gdo
3:C e,g(q)←TopMu∈U gse,g(q, u)
4: Computep e,g(u|q,C e,g)from normalized scores
5: ComputeH e,g(q)andc e,g(q) = 1−H e,g(q)
6:U cand← U cand∪ Ce,g(q)
7:end for
8: Normalize{c e,g(q)}to obtainw e,g(q)
9:Compute P(u|q) andR(u|q) =P(u|q)/p
ℓ(u) for
allu∈ U cand
10: SortU cand byR(u|q)in descending order
11:U promoted ← ∅,A(v)←0
12:for alluin rankedU canddo
13:v←π 8(u)ifu∈ U 2∪ U4, otherwisev←u
14:U promoted ← U promoted ∪ {v}
15:A(v)←1−(1−A(v))(1−P(u|q))
16:end for
17: SortU promoted byA(v)in descending order
18:S ← ∅
19:for allvin rankedU promoted do
20:if|S|=Kthen
21:break
22:end if
23:ifS=∅ormax s∈Soverlap(v, s)<0.75then
24:S ← S ∪ {v}
25:end if
26:end for
27:returnS
B Computational Cost Analysis
Table 3 presents a comprehensive computational
cost analysis of both the retrieval preprocessing
and generation phases across the evaluated meth-
ods. This investigation is conducted on the NQ
dataset, utilizing the Qwen3-Embedding-4B model
for dense retrieval and the Llama-3.2-3B-Instruct
generator as detailed in the main paper. The base-
line models exhibit a fast retrieval preprocess-
ing time of 0.15 seconds per question. In con-
trast, our proposed multi-granularity approaches
require higher retrieval preprocessing times, peak-
ing at 5.36 seconds for UMGP-RAG due to multi-
granualirty. This increased overhead is the main
computational cost of the framework, caused by
multi-granularity dense and sparse retrieval. How-
ever, the resulting compact context reduces genera-
tion time and memory usage during the computa-
tionally demanding generation phase. By ensuring
that only a highly relevant and compact context
is fed into the language model, our approaches
achieve the lowest generation times (between 0.30
and 0.33 seconds per question) and the smallestpeak memory footprints (under 7000 MiB). This
demonstrates that the initial investment in thorough
database preparation directly translates to substan-
tial speedups and memory reductions when gener-
ating the final response.
C Statistical Robustness with Bootstrap
Resampling
The main results report mean retrieval and genera-
tion performance for readability. To assess statisti-
cal robustness, we additionally estimate standard
deviations using 1,000 bootstrap resamples over
the evaluation examples. Each bootstrap sample
is drawn with replacement from the original eval-
uation set, and all metrics are recomputed on the
resampled set. Tables 4 and 5 report the bootstrap
results on Natural Questions with Qwen2.5-3B-
Instruct and Llama-3.2-3B-Instruct, respectively.
Tables 6 and 7 report the corresponding results on
HotPotQA.
Overall, the bootstrap results show that the main
trends are stable. UMGP-RAG consistently im-
proves over UMG-RAG, supporting the benefit
of parent promotion and overlap-aware dedupli-
cation. The results also show that the strongest
retrieval AR@5 does not always correspond to the
strongest generation performance. For example,
long-context retrieval methods can achieve high
answer coverage, but the resulting context may
still include distractors or place answer-bearing
evidence in less favorable positions for the gener-
ator. In contrast, UMGP-RAG tends to produce
stronger generation performance by constructing
compact and locally coherent contexts.
The standard deviations are small relative to the
observed performance differences in most settings,
indicating that the improvements are not driven by
a small number of examples. These results further
support the conclusion that uncertainty-aware fu-
sion and parent promotion provide a robust benefit
for long-document RAG.
D AI Assistant Acknowledgement
AI assistance was used to support code debugging
and improve the clarity of the manuscript. The
authors carefully reviewed, edited, and verified all
AI-assisted content and take full responsibility for
the final version of this publication.
11

MethodRetrieval
Preprocessing Time
(s/question)Retrieval
Preprocessing Mem.
(MiB)Generation Time
(s/question)Generation Mem.
(MiB)
Standard RAG
0.15 79540.71 7558
Summ. RAG 0.28 7594
LongLLMLingua 4.70 24398
MS-PoE 0.71 7554
REPLUG 1.01 9452
PINE 1.03 7558
LongRAG 0.71 7618
Hybrid 1.59 8396 0.27 6908
UMG-RAG5.16 8396 0.30 6762
UMGP-RAG5.36 8396 0.33 6716
Table 3: Computational cost analysis of the retrieval preprocessing and generation phases on the NQ dataset using
Qwen3-Embedding-4B and Llama-3.2-3B-Instruct. Time is reported in seconds per question and peak memory in
MiB. While the proposed multi-granularity methods require additional time during the retrieval preprocessing step,
they achieve the most efficient generation phase with the lowest time and memory requirements.
E Computational Infrastructure and
Budget
All experiments were conducted on a single
NVIDIA RTX A5000 GPU with 24GB memory.
Because the proposed methods are training-free
and do not update retriever or generator parame-
ters, the computational budget is reported as re-
trieval preprocessing time, generation time, and
peak memory usage per question rather than train-
ing GPU hours. The detailed time and memory
measurements are reported in Appendix B.
12

Retriever MethodRetrieval
AR@5↑Qwen2.5-3B-Instruct
F1↑AR↑
SPLADEv3 RAG 0.7994±0.0074 0.4745±0.0083 0.4529±0.0094
BERTRAG
0.4477±0.00900.2270±0.0069 0.2013±0.0072
Summarized RAG 0.1704±0.0064 0.1491±0.0068
LongLLMLingua 0.2304±0.0071 0.2080±0.0078
MS-PoE 0.2239±0.0070 0.1974±0.0074
REPLUG 0.1373±0.0059 0.1174±0.0059
PINE 0.1915±0.0066 0.1843±0.0072
LongRAG 0.4441±0.0093 0.2093±0.0070 0.1851±0.0073
Hybrid 0.7011±0.0086 0.4265±0.0090 0.4018±0.0093
UMG-RAG (Ours) 0.6852±0.0088 0.4350±0.0084 0.4110±0.0092
UMGP-RAG (Ours) 0.7797±0.0078 0.4572±0.0087 0.4307±0.0091
BGE-M3RAG
0.8562±0.00680.4407±0.0085 0.3994±0.0089
Summarized RAG 0.3243±0.0080 0.2880±0.0085
LongLLMLingua 0.4581±0.0083 0.4184±0.0089
MS-PoE 0.4432±0.0083 0.4015±0.0092
REPLUG 0.4701±0.0082 0.4216±0.0093
PINE 0.4547±0.0084 0.4420±0.0092
LongRAG 0.9101±0.0055 0.4598±0.0083 0.4219±0.0093
Hybrid 0.8241±0.0071 0.4927±0.0085 0.4727±0.0092
UMG-RAG (Ours) 0.8023±0.0074 0.4809±0.0081 0.4593±0.0093
UMGP-RAG (Ours) 0.8759±0.0063 0.5052±0.0086 0.4794±0.0092
Qwen3-
Embedding-4BRAG
0.8601±0.00670.4427±0.0084 0.3990±0.0092
Summarized RAG 0.3333±0.0084 0.2915±0.0086
LongLLMLingua 0.4459±0.0085 0.4057±0.0089
MS-PoE 0.4481±0.0083 0.4064±0.0090
REPLUG 0.4796±0.0084 0.4367±0.0094
PINE 0.4628±0.0084 0.4466±0.0096
LongRAG 0.9253±0.0050 0.4744±0.0084 0.4329±0.0093
Hybrid 0.8114±0.0071 0.4803±0.0088 0.4621±0.0092
UMG-RAG (Ours) 0.7949±0.0077 0.4807±0.0082 0.4582±0.0093
UMGP-RAG (Ours) 0.8745±0.0064 0.4961±0.0085 0.4748±0.0091
Table 4:Results on the Natural Questions dataset (Qwen2.5-3B-Instruct).The best retrieval and generation
scores within each retriever block are highlighted inbold, and the second best scores are underlined .
13

Retriever MethodRetrieval
AR@5↑Llama-3.2-3B-Instruct
F1↑AR↑
SPLADEv3 RAG 0.7994±0.0074 0.4923±0.0081 0.4741±0.0097
BERTRAG
0.4477±0.00900.2414±0.0074 0.2178±0.0076
Summarized RAG 0.1972±0.0068 0.1703±0.0071
LongLLMLingua 0.2573±0.0079 0.2429±0.0081
MS-PoE 0.2435±0.0073 0.2192±0.0076
REPLUG 0.2254±0.0071 0.1896±0.0071
PINE 0.2084±0.0067 0.2097±0.0074
LongRAG 0.4441±0.0093 0.2318±0.0073 0.2094±0.0077
Hybrid 0.7011±0.0086 0.4504±0.0083 0.4293±0.0092
UMG-RAG (Ours) 0.6852±0.0088 0.4514±0.0080 0.4304±0.0097
UMGP-RAG (Ours) 0.7797±0.0078 0.4811±0.0086 0.4533±0.0093
BGE-M3RAG
0.8562±0.00680.4941±0.0082 0.4593±0.0094
Summarized RAG 0.3520±0.0081 0.3127±0.0086
LongLLMLingua 0.4631±0.0083 0.4462±0.0096
MS-PoE 0.5005±0.0086 0.4709±0.0091
REPLUG 0.4962±0.0085 0.4459±0.0089
PINE 0.4452±0.0080 0.4603±0.0093
LongRAG 0.9101±0.0055 0.5113±0.0081 0.4910±0.0095
Hybrid 0.8241±0.0071 0.4986±0.0083 0.4766±0.0097
UMG-RAG (Ours) 0.8023±0.0074 0.4873±0.0078 0.4716±0.0095
UMGP-RAG (Ours) 0.8759±0.0063 0.5128±0.0084 0.4878±0.0094
Qwen3-
Embedding-4BRAG
0.8601±0.00670.4939±0.0084 0.4621±0.0093
Summarized RAG 0.3537±0.0083 0.3162±0.0087
LongLLMLingua 0.4561±0.0083 0.4346±0.0098
MS-PoE 0.5046±0.0083 0.4720±0.0094
REPLUG 0.5120±0.0084 0.4642±0.0088
PINE 0.4291±0.0082 0.4438±0.0095
LongRAG 0.9253±0.0050 0.5145±0.0082 0.4949±0.0096
Hybrid 0.8114±0.0071 0.4916±0.0083 0.4741±0.0096
UMG-RAG (Ours) 0.7949±0.0077 0.4974±0.0080 0.4790±0.0093
UMGP-RAG (Ours) 0.8745±0.0064 0.5220±0.0084 0.4984±0.0094
Table 5:Results on the Natural Questions dataset (Llama-3.2-3B-Instruct).The best retrieval and generation
scores within each retriever block are highlighted inbold, and the second best scores are underlined .
14

Retriever MethodRetrieval
AR@5↑Qwen2.5-3B-Instruct
F1↑AR↑
SPLADEv3 RAG 0.6043±0.0065 0.4613±0.0061 0.3850±0.0063
BERTRAG
0.3638±0.00670.2599±0.0055 0.2087±0.0053
Summarized RAG 0.2275±0.0053 0.1864±0.0054
LongLLMLingua 0.2833±0.0055 0.2271±0.0056
MS-PoE 0.2612±0.0054 0.2112±0.0054
REPLUG 0.1798±0.0049 0.1466±0.0047
PINE 0.2424±0.0052 0.2039±0.0053
LongRAG 0.3230±0.0062 0.2299±0.0051 0.1861±0.0053
Hybrid 0.5207±0.0066 0.4081±0.0062 0.3311±0.0060
UMG-RAG (Ours) 0.5134±0.0068 0.4163±0.0062 0.3416±0.0064
UMGP-RAG (Ours) 0.5830±0.0067 0.4453±0.0061 0.3645±0.0065
BGE-M3RAG
0.7811±0.00570.4613±0.0064 0.3798±0.0065
Summarized RAG 0.3856±0.0058 0.3161±0.0064
LongLLMLingua 0.4706±0.0061 0.3807±0.0065
MS-PoE 0.4622±0.0059 0.3805±0.0066
REPLUG 0.3941±0.0060 0.3150±0.0061
PINE 0.4442±0.0062 0.3787±0.0065
LongRAG 0.7457±0.0059 0.4658±0.0061 0.3845±0.0065
Hybrid 0.6471±0.0063 0.4775±0.0063 0.3975±0.0063
UMG-RAG (Ours) 0.6273±0.0064 0.4749±0.0062 0.3957±0.0066
UMGP-RAG (Ours) 0.7002±0.0061 0.5034±0.0062 0.4216±0.0068
Qwen3-
Embedding-4BRAG
0.7971±0.00530.4883±0.0060 0.4041±0.0065
Summarized RAG 0.3976±0.0058 0.3261±0.0063
LongLLMLingua 0.4603±0.0061 0.3693±0.0063
MS-PoE 0.4948±0.0060 0.4073±0.0067
REPLUG 0.4075±0.0060 0.3252±0.0063
PINE 0.4621±0.0061 0.3980±0.0065
LongRAG 0.7280±0.0062 0.4524±0.0061 0.3748±0.0066
Hybrid 0.6091±0.0063 0.4626±0.0060 0.3837±0.0062
UMG-RAG (Ours) 0.5845±0.0068 0.4622±0.0060 0.3820±0.0064
UMGP-RAG (Ours) 0.6761±0.0064 0.4919±0.0060 0.4071±0.0066
Table 6:Results on the HotPotQA dataset (Qwen2.5-3B-Instruct).The best retrieval and generation scores
within each retriever block are highlighted inbold, and the second best scores are underlined .
15

Retriever MethodRetrieval
AR@5↑Llama-3.2-3B-Instruct
F1↑AR↑
SPLADEv3 RAG 0.6043±0.0065 0.4798±0.0062 0.3966±0.0069
BERTRAG
0.3638±0.00670.2803±0.0055 0.2279±0.0057
Summarized RAG 0.2469±0.0055 0.2025±0.0053
LongLLMLingua 0.3021±0.0057 0.2362±0.0057
MS-PoE 0.2860±0.0057 0.2339±0.0056
REPLUG 0.2560±0.0053 0.1984±0.0052
PINE 0.2557±0.0052 0.2180±0.0055
LongRAG 0.3230±0.0062 0.2625±0.0056 0.2104±0.0056
Hybrid 0.5207±0.0066 0.4245±0.0059 0.3479±0.0064
UMG-RAG (Ours) 0.5134±0.0068 0.4223±0.0060 0.3470±0.0067
UMGP-RAG (Ours) 0.5830±0.0067 0.4460±0.0059 0.3655±0.0063
BGE-M3RAG
0.7811±0.00570.4913±0.0061 0.4048±0.0064
Summarized RAG 0.4109±0.0060 0.3366±0.0061
LongLLMLingua 0.4668±0.0062 0.3771±0.0062
MS-PoE 0.5001±0.0062 0.4123±0.0068
REPLUG 0.4287±0.0060 0.3362±0.0065
PINE 0.4409±0.0060 0.3941±0.0065
LongRAG 0.7457±0.0059 0.4799±0.0059 0.3984±0.0066
Hybrid 0.6471±0.0063 0.4986±0.0060 0.4175±0.0069
UMG-RAG (Ours) 0.6273±0.0064 0.4827±0.0059 0.4018±0.0068
UMGP-RAG (Ours) 0.7002±0.0061 0.5133±0.0060 0.4275±0.0066
Qwen3-
Embedding-4BRAG
0.7971±0.00530.4991±0.0060 0.4132±0.0065
Summarized RAG 0.4315±0.0060 0.3546±0.0061
LongLLMLingua 0.4644±0.0060 0.3738±0.0063
MS-PoE 0.5172±0.0059 0.4295±0.0065
REPLUG 0.4343±0.0061 0.3411±0.0064
PINE 0.4463±0.0059 0.4027±0.0067
LongRAG 0.7280±0.0062 0.4765±0.0061 0.3961±0.0063
Hybrid 0.6091±0.0063 0.4709±0.0060 0.3904±0.0066
UMG-RAG (Ours) 0.5845±0.0068 0.4752±0.0060 0.3929±0.0068
UMGP-RAG (Ours) 0.6761±0.0064 0.5068±0.0059 0.4200±0.0064
Table 7:Results on the HotPotQA dataset (Llama-3.2-3B-Instruct).The best retrieval and generation scores
within each retriever block are highlighted inbold, and the second best scores are underlined .
16