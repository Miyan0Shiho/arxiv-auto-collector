# Cross-Attention Calibrated Deduplication for Retrieval-Augmented Generation System

**Authors**: Phuong Le Huy, Nam H. Nguyen, Quan V. Dang

**Published**: 2026-07-27 12:09:03

**PDF URL**: [https://arxiv.org/pdf/2607.24332v1](https://arxiv.org/pdf/2607.24332v1)

## Abstract
Common chunking strategies in Retrieval-Augmented Generation (RAG) systems often create redundant chunks. These redundant chunks make the vector database bigger and slow down retrieval. A common fix is cosine-similarity thresholding. This method reduces each chunk to a single vector, then compares vectors using a similarity score. But a single vector can lose the fine-grained, token-level detail needed to tell a true duplicate apart from a chunk that just shares the same topic. We propose Cross-Attention Calibrated Deduplication (CACD). CACD checks each new chunk against an in-memory pool of chunks already kept, using a cross-encoder instead of a single pooled vector. This keeps token-level detail all the way to the final comparison. CACD combines three parts: the cross-encoder comparison itself, a New Information Score (NIS) that measures how much of a chunk is not explained by a candidate already kept, and a majority vote across several candidates rather than a single best match. NIS is calculated from the attention entropy of the cross-encoder. We tested CACD against five existing filtering methods, nine chunking strategies, and 18 configurations, all on the full SQuAD 1.1 validation set. In our experiments, CACD removes 9.75% of chunks on average. This drop rate is close to other semantic-level methods, and much higher than exact-match filters, which barely remove anything. In these experiments, CACD also processes each configuration in 51.0 seconds on average, about 27% faster than the strongest baseline, NERExact (69.6s), and about 7x faster than cosine-similarity filtering (356.7s). These results come from a single dataset, so we present them as an early comparison, not a general claim. Code for the baseline evaluation and for CACD is available at https://github.com/lehuyphuong/rag_bench and https://github.com/lehuyphuong/cacd_dedup.

## Full Text


<!-- PDF content starts -->

Cross-Attention Calibrated Deduplication for
Retrieval-Augmented Generation System
Phuong Le Huy∗, Nam H. Nguyen∗, and Quan V . Dang∗,†
∗Full Stack Data Science
†Department of Computer Science, University College London
Emails: phuonglehuy172k@gmail.com, namnqs12@gmail.com, dangvanquan.nd@gmail.com
Abstract—Common chunking strategies in Retrieval-
Augmented Generation (RAG) systems often create redundant
chunks. These redundant chunks make the vector database
bigger and slow down retrieval. A common fix is cosine-similarity
thresholding. This method reduces each chunk to a single vector,
then compares vectors using a similarity score. But a single
vector can lose the fine-grained, token-level detail needed to tell
a true duplicate apart from a chunk that just shares the same
topic. We propose Cross-Attention Calibrated Deduplication
(CACD). CACD checks each new chunk against an in-memory
pool of chunks already kept, using a cross-encoder instead
of a single pooled vector. This keeps token-level detail all the
way to the final comparison. CACD combines three parts:
the cross-encoder comparison itself, a New Information Score
(NIS) that measures how much of a chunk is not explained by
a candidate already kept, and a majority vote across several
candidates rather than a single best match. NIS is calculated
from the attention entropy of the cross-encoder. We tested CACD
against five existing filtering methods, nine chunking strategies,
and 18 configurations, all on the full SQuAD 1.1 validation
set. In our experiments, CACD removes 9.75% of chunks on
average. This drop rate is close to other semantic-level methods,
and much higher than exact-match filters, which barely remove
anything. In these experiments, CACD also processes each
configuration in 51.0 seconds on average, about 27% faster than
the strongest baseline, NERExact (69.6s), and about 7×faster
than cosine-similarity filtering (356.7s). These results come from
a single dataset, so we present them as an early comparison,
not a general claim. Code for the baseline evaluation and for
CACD is available at https://github.com/lehuyphuong/rag bench
and https://github.com/lehuyphuong/cacd dedup.
Index Terms—retrieval-augmented generation, chunk dedupli-
cation, cross-encoder, attention entropy, vector database, infor-
mation retrieval, new information score
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) is a way to make
a language model answer questions using information it was
not trained on [1]. Instead of relying only on what the model
memorized during training, a RAG system keeps an external
store of documents, retrieves the passages most relevant to a
question, and gives those passages to the model as context
before it generates an answer. This reduces hallucination and
lets the system stay up to date without retraining the model
itself.
A RAG system is built from two connected flows. As shown
in Figure 1, the first flow prepares the document collection
for retrieval: raw data is split into smaller pieces, called
chunks, by a chunking step, and each chunk is turned into
Fig. 1. A typical RAG pipeline. Documents are chunked, embedded, and
stored in a vector store; at query time, the query is embedded, relevant chunks
are retrieved, and the LLM generates a response from the combined prompt
and context.
a vector representation by an embedding model; these vectors
are stored in a vector store. The second flow handles each
incoming question: the query is embedded and sent to the
vector store, which returns the most relevant chunks as context;
this context is combined with the original query into a prompt,
and a large language model (LLM) reads the prompt and
produces the final response [2].
As new documents keep being added to the vector store over
time, chunking strategies often produce chunks that overlap
or repeat the same information, sometimes from the same
document and sometimes across different documents [3] [4].
This redundancy grows the vector store without adding new
information, slows down retrieval, and can even hurt answer
quality if the retrieved context is full of repeated content
instead of diverse, relevant information. Deciding whether a
new chunk is a genuine duplicate of something already in the
vector store, without discarding a chunk that only shares a
topic, is the central problem this paper addresses.
The main contributions of this paper are:
•Cross-Attention Calibrated Deduplication (CACD), a fil-
tering method with three parts working together: a cross-
encoder that compares each new chunk against the full,
persistently growing index instead of pooled similarity
vectors; a New Information Score (NIS), derived from
the entropy of the cross-encoder’s attention matrix, that
scores how much of a chunk is not explained by a given
candidate; and a majority vote across several retrieved
candidates, so that one misleading nearest neighbor can-
not flip the outcome on its own.
•An evaluation of five existing filtering methods that had
arXiv:2607.24332v1  [cs.CL]  27 Jul 2026

FixedSize
Equal-length windowsRecursive
paragraph
sentence sentence
wordSemantic
Split at high-distance
points
Overlapping
Sliding window,
shared overlapAdaptiveEntropy
dense text repetitive
Size follows local entropyAdaptiveSentenceLen
Groups sentences by length
HierarchicalParentChild
Parent
Child Child
Both levels indexed,
child keeps parent_idContextual
[Context] [Context] [Context] [Context]
Each chunk gets a
context headerTopicBased
Boundaries at topic changeFig. 2. Schematic overview of the nine chunking strategies compared in
this work. Each panel shows how one document (horizontal bar) is split into
chunks under that strategy’s mechanism.
not previously been benchmarked against one another,
alongside CACD, across nine chunking strategies and
eighteen configurations on the SQuAD 1.1 validation set.
The rest of this paper is organized as follows. Section II re-
views related work on chunking strategies and existing filtering
methods. Section III describes CACD in detail: the retrieval,
scoring, and decision stages, the guards that protect against
known failure cases, and the decision thresholds used. Section
IV presents the experimental setup and results, comparing
CACD against the five baseline methods and breaking results
down by chunking strategy. Section V concludes and discusses
current limitations.
II. RELATEDWORK
A. Chunking Strategies
RAG systems can split documents in many ways before
indexing. Figure 2 shows the nine strategies compared here.
•FixedSizesplits text into equal-length windows, ignoring
sentence or paragraph breaks [5].
•Recursivesplits at paragraph breaks first, then sentence
breaks, then word breaks, using a smaller split only when
a larger one does not fit [6].
•Semanticplaces a boundary between two nearby sen-
tences when the distance between their embeddings is
high, keeping related sentences together [7].
•Overlappinguses a sliding window with overlap between
chunks, so text near an edge can appear in more than one
chunk [8].
•AdaptiveEntropyuses smaller chunks where local text
entropy is high (information-dense) and larger chunks
where it is low (repetitive) [9].
•AdaptiveSentenceLengroups more short sentences or
fewer long ones per chunk, based on the average sentence
length nearby [9].
•HierarchicalParentChildindexes a larger parent chunk
together with its smaller child chunks, each child linked
back to its parent [10].
ExactNorm
chunk i chunk j
"Paris is the Capital." "paris is the capital"
normalize(): NFKC · lowercase · trim
sha256: 9f3a2e... (same)
Identical after normalization
→drop later chunkMinHashLSH
J(i,j) = |A B| / |A B|
(estimated via MinHash)
shingles(i) shingles(j)
A∩B
MinHash signature agreement (6/8 slots match)
J≥0.7→LSH bucket match
→drop chunk j
Similarity
embedding space
keptcos sim = 0.80
new chunk
Max similarity to any kept chunk
≥0.8→dropNERExact
chunk i
The [Paris]LOC Olympics start in [2024]DATE .
chunk j
In[2024]DATE , the Olympics are in [Paris]LOC .
E(i) = {(paris,LOC), (2024,DATE)} = E(j)
chunk i chunk j
Same entity set despite different wording
→drop chunk jFig. 3. Schematic overview of the four baseline deduplication filters
compared against CACD. ExactNorm and MinHashLSH compare lexical
or character-level similarity; Similarity compares dense embedding cosine
distance; NERExact compares named-entity set equality.
•Contextualadds a short header, such as the document
title, to each chunk before embedding, so the embedding
also carries some context from the document [11].
•TopicBasedgroups sentence embeddings with k-means
and joins nearby sentences from the same group into one
chunk, placing a boundary where the topic changes [12].
B. Chunk Deduplication Techniques
Alongside chunking strategies, several filtering methods
have been used to remove redundant chunks before index-
ing [13]. Figure 3 summarizes the four baseline methods
compared against CACD in this work; each appears to have a
different blind spot.
•ExactNormremoves a chunk if its text, after normal-
ization (case-folding and whitespace collapsing), exactly
matches an already-kept chunk [13]. Because it requires
an exact match, it may not catch chunks that convey the
same information with even minor wording differences.
•MinHashLSHestimates Jaccard similarity between
chunks from character-level shingle sets and removes a
chunk once its estimated similarity to an already-kept
chunk crosses a threshold [13]. Since it operates on
lexical overlap, it may miss paraphrased duplicates and,
depending on the shingle size and threshold chosen, may
also flag lexically similar but topically distinct chunks.
•Similarityembeds each chunk into a single pooled vector
and removes it if its cosine similarity to any already-kept
chunk exceeds a threshold [13]. Collapsing a chunk to
one vector can make it difficult to tell a genuine duplicate
apart from a chunk that only shares a topic or vocabulary,
which may lead to removing chunks that are not actually
redundant.
•NERExactextracts the named entities in each chunk and
removes a later chunk if its entity set exactly matches an

New chunk A Stage 1: Retrieval 
Find K nearest kept chunks 
Stage 2: Cross-Encoder Scoring 
Duplicate prob + New Information Score 
Stage 3: Decision 
Majority vote Keep 
Drop 
Fig. 4. CACD’s three-stage pipeline. A new chunk is scored against the K
nearest chunks already kept, and a majority vote across those K candidates
decides whether to keep or drop it.
New chunk A Embedding V ector v_A 
Chunks kept so far Pool I (in-memory) 
T opK(v_A, I, K) 
Exact search Candidate 1 
Candidate 2 
Candidate 3 K candidates 
Fig. 5. Stage 1: the new chunk is embedded, then compared against every
chunk already kept in the in-memory pool to retrieve the K nearest candidates.
already-kept chunk’s [13]. This signal depends entirely
on named entities, so it may leave redundancy in entity-
sparse text unaddressed, and two chunks that mention the
same entities can still be treated as duplicates even if they
describe different information about those entities.
These four methods and CACD are compared under the
same experimental setup in Section IV.
III. METHODS
A. Overview
Cross-Attention Calibrated Deduplication (CACD) pro-
cesses chunks one at a time as a document collection is
ingested. For each new chunkA, CACD decides whether to
insert it into a persistent indexIor discard it as redundant,
checking it against every chunk kept so far in the same
ingestion run, not only the chunks in the current batch.
The pipeline has three stages, shown in Figure 4. Stage
1 retrieves theKchunks inIclosest toA. Stage 2 scores
Aagainst each of theseKcandidates with a cross-encoder,
producing a duplicate probability and a New Information
Score (NIS). Stage 3 turns these per-candidate scores into one
KEEP/DROPdecision forAthrough a majority vote, subject to
the guards in Section III-D. A chunk voted DROPis discarded.
B. Stage 1: Retrieval
Ais embedded into a vectorv A, and theKnearest chunks
[15] [16] already inIare retrieved by an exact search over
Iheld in memory (Figure 5):Iis a single growing matrix
of embeddings, and retrieval is one matrix–vector product
followed by picking theKlargest scores.
Candidate 1 
[CLS] A [SEP] B [SEP] 
Cross-encoder 
(last layer attention) 
Duplicate probability 
p_dup New Information Score 
NIS(B|A) New chunk A Fig. 6. Stage 2: chunk A and a candidate are encoded together; an example
attention matrix is shown, from which both a duplicate probability and the
New Information Score are computed.
This search is exact, not approximate, so it always finds
the trueKnearest chunks under cosine similarity, unlike an
approximate index such as HNSW [14], at the cost of scanning
the whole pool per query rather than a faster approximate
lookup. At the data sizes used in this paper (a few thousand
to roughly ten thousand chunks per configuration), this was
still fast in wall-clock terms in our experiments: an earlier
version of Stage 1 built on an external vector store was in
practice slowed down mostly by per-query storage overhead,
which the in-memory version removes. IfIis still empty,A
is kept automatically, since there is nothing yet to compare it
against.
C. Stage 2: Cross-Encoder Scoring
Each candidate pair(A, B)is jointly encoded by
a pretrained cross-encoder [17] [18] as one sequence,
[CLS]A[SEP]B[SEP], in a single forward pass (Figure 6).
This produces a duplicate probabilityp dup∈[0,1]and the
model’s final-layer attention matrix, averaged across attention
heads.
The New Information Score (NIS) uses the part of that
attention matrix showing how tokens ofBattend back to
tokens ofA. For each tokenjofB, its attention overAis
turned into a probability distributionp(· |j)over the tokens
ofA, and its Shannon entropy [19] is computed:
H(j) =−|A|X
i=1p(i|j) logp(i|j),(1)
where|A|is the number of tokens inA. A lowH(j)means
tokenj’s attention concentrates on a few tokens ofA, so
Aexplainsj; a highH(j)means attention spreads evenly
acrossA, sojcarries informationAdoes not have.H(j)
is normalized bylog|A|, the value it takes when attention is

token j of B 
A1 A2 A3 A4 A5 token j of B 
A1 A2 A3 A4 A5 
0.02 0.02 0.90 0.03 0.03 0.21 0.20 0.19 0.19 0.21 
One strong connection, the rest are weak 
H( j) is small => A explained token j No strong connection, A has no match for j 
H( j) is large => j carries new information Low NIS: attention focuses on one token of A High NIS: attention spreads evenly across A Fig. 7. How a single tokenjofBattends to the tokens ofAin the two
extreme cases behind NIS. When attention concentrates on one token ofA
(left),H(j)is small andAis treated as explainingj. When attention spreads
evenly acrossA(right),H(j)is large andjis treated as carrying information
Adoes not have.
exactly uniform overA, and averaged over all|B|tokensjof
Bto give
NIS(B|A) =1
|B||B|X
j=1H(j)
log|A|∈[0,1].(2)
NIS(B|A)→0meansBis fully explained byA(redun-
dant);NIS(B|A)→1meansBis largely unaccounted for
byA.
Figure 7 illustrates both cases for a single tokenjofB.
When one token ofAreceives most of the attention,H(j)
stays small, sojis treated as explained byA. When attention
spreads evenly across all tokens ofAinstead,H(j)grows
toward its maximum, sojis treated as carrying information
Adoes not have.
To check whetherp dup, the cross-encoder’s own duplicate
probability, gives a reasonable signal on its own, chunk pairs
were built with a known, exact overlap level, from 100% down
to 0% in steps of 10%. One chunk stayed fixed. The other
was built from paraphrases (not copies) of a set fraction of
the first chunk’s sentences, with the rest replaced by unrelated
sentences. This was done for two different topic pairs, and the
results were averaged.
Table I shows cosine similarity andp dupat each overlap
level.p dupfollows roughly the same downward trend as cosine
similarity as overlap decreases, though the pattern is not fully
clean: it flattens at a couple of points (40% and 30% overlap
give nearly the same value), and at 10% overlap it sits slightly
above its value at 20%. Overall,p duplands close to cosine
similarity rather than clearly ahead of or behind it, which is a
reasonable result for a signal used entirely on its own, without
the guards or the majority vote described in Sections III-D
and III-E.
To do better thanp dupalone, CACD does not stop here:
it combinesp dupwith NIS and a majority vote across several
candidates, as described in Section III-E. Table II and Table III
report how the full CACD pipeline performs against the
baseline filters on the SQuAD 1.1 validation set.TABLE I
COSINE SIMILARITY VS.p DUP ACROSS A SEMANTIC-OVERLAP RANGE,
AVERAGED OVER TWO TOPIC PAIRS.
Overlap % Cosinep dup
100 0.888 0.817
90 0.864 0.693
80 0.796 0.673
70 0.780 0.631
60 0.738 0.581
50 0.719 0.519
40 0.676 0.478
30 0.601 0.478
20 0.553 0.421
10 0.193 0.293
0 0.036 0.133
D. Guards
Three conditions adjust the decision around scoring.
The parent-child guard excludes a candidateBfrom scoring
againstAwhen hierarchical chunking links them as parent and
child, since their overlap is intentional, not duplication; sibling
children of the same parent are still scored normally, since they
may genuinely duplicate each other.
The header guard strips any prepended[Context: ...]
header from bothAandBbefore scoring only, so a header
shared by every chunk of a document does not dominate the
comparison; the stored text keeps the header.
The length-aware guard protects a chunk longer than 300
characters from being dropped by a highp dupor a low NIS
alone, on the assumption that a longer chunk is more likely to
carry unique information, unless its NIS against the deciding
candidate falls below a floor of 0.3, indicating the candidate
explains it too thoroughly for length alone to justify keeping
it.
E. Stage 3: Decision
The two probability thresholds used below,τ highandτ low,
and the NIS thresholdτ NIS, are not hand-picked:τ highand
τlowcome from a cost-sensitive cutoff [24] that balances the
cost of wrongly dropping a non-duplicate chunk against the
cost of wrongly keeping a genuine duplicate, which under
the symmetric setting used in this paper’s experiments gives
τhigh= 0.8andτ low= 0.2;τ NIS= 0.8is the midpoint of the
normalized entropy scale from Eq. 2, following the retention
target used in SemDeDup [25].
An earlier version of CACD based its decision on a single
candidate, the one with the highestp dupamong theKretrieved.
This made the outcome sensitive to a single retrieval error:
if Stage 1 returned one misleadingly high-scoring but unrep-
resentative neighbor,Acould be dropped even though every
other candidate indicated it was not a duplicate. CACD instead
treats every valid candidate (one that passes the guards) as an
independent vote [20] [21], and requires a majority before
committing to a decision [22], as shown in Figure 8.
For each valid candidateB, a highp dup(≥τ high) votes
Drop, unless the length-aware guard protectsA; a lowp dup
(≤τ low) votes Keep; and in between, NIS decides, voting Keep
whenBleaves enough ofAunexplained (≥τ NIS) and Drop

Candidate 1 Decision Threshold 
Keep 
Candidate 2 Decision Threshold 
Drop 
Candidate 3 Decision Threshold 
Keep 
Keep 
V ote Fig. 8. Stage 3: each candidate is compared against the decision thresholds
and casts a Keep or Drop vote; the majority across all votes decides the
outcome for chunk A.
otherwise. V otes are tallied with early exit, stopping as soon
as either side reaches a majority. If no majority is reached
before every candidate has voted, CACD defaults to Keep, so
an inconclusive vote never silently discards content.
The timing results in Section IV-A use a batched imple-
mentation that scores allKcandidates for a whole micro-
batch of chunks in one cross-encoder pass before voting on
any of them, which is faster on a GPU than voting candidate-
by-candidate as described above. Both versions always reach
the same decision.
IV. RESULTS
Setup.Experiments were conducted on the full SQuAD 1.1
validation set [23] (2,067 passages, 10,570 question-answer
pairs), across 18 chunking configurations built from nine
chunking strategies at two size settings each. CACD was
compared against five baselines (NOFILTER, EXACTNORM,
MINHASHLSH, SIMILARITY, NEREXACT), all run on the
same chunking outputs, embedding model, and evaluation
steps, so differences in the results come from the filtering
method itself.
Configuration and metrics.CACD usedτ NIS= 0.8,
K= 5, and symmetric costs (c FP=c FN= 1), giving
τhigh= 0.8andτ low= 0.2, the same values described in
Section III-E. Precision, Recall, and IoU measure retrieval
quality after filtering [13]; storage and ingestion time measure
cost. GPU-accelerated embedding and FP16 cross-encoder
inference were used, with Stage 1 retrieval run as an exact
in-memory CPU search; these choices affect speed only, not
which chunks any method keeps or drops.
A. Main Results
Table II reports results averaged per chunking configuration
across all 18 configurations. In this evaluation, CACD reaches
the highest drop rate of any method (9.75%, next closest is
SIMILARITYat 8.40%) and the smallest index (27.13 MB,
versus 27.38–30.64 MB for the rest). Ingestion time (51.01s)
is close to MINHASHLSH (50.74s) and far faster than the
other semantic-level baselines, SIMILARITY(356.70s) and
NEREXACT(69.59s); only the two baselines that do little
comparison work, NOFILTERand EXACTNORM, are faster.
Precision (0.3818) and IoU (0.3263) sit above SIMILARITY
and NEREXACTbut below NOFILTER, EXACTNORM, and
MINHASHLSH, which is expected since those three removeTABLE II
RESULTS AVERAGED PER CHUNKING CONFIGURATION(18
CONFIGURATIONS),FULLSQUAD 1.1VALIDATION SET(2,067
DOCUMENTS, 10,570QUESTIONS). STORAGE ANDTIME ARE
PER-CONFIGURATION MEANS,NOT TOTALS. BEST VALUE PER COLUMN IS
INBOLD,INDEPENDENT OF METHOD.
Method Prec. Rec. IoU MB Time (s) Drop %
NoFilter0.39240.71430.336430.64 32.16 0.00
ExactNorm 0.3908 0.7152 0.3355 30.3631.740.63
MinHashLSH (0.8) 0.39030.71530.3351 30.27 50.74 0.86
Similarity (0.8) 0.3745 0.7122 0.3216 27.38 356.70 8.40
NERExact 0.3798 0.7057 0.3256 28.29 69.59 6.37
CACD 0.3818 0.7001 0.326327.1351.019.75
TABLE III
CACDRESULTS BY CHUNKING STRATEGY,AVERAGED OVER EACH
STRATEGY’S TWO SIZE CONFIGURATIONS. BEST VALUE PER COLUMN IS
INBOLD.
Strategy Prec. Rec. IoU MB Time (s) Drop %
FixedSize 0.3841 0.6112 0.3086 27.09 51.50 8.38
Recursive 0.4185 0.5684 0.3141 30.17 57.95 12.96
Semantic 0.3599 0.6783 0.3098 22.70 48.45 11.68
Overlapping 0.4153 0.7850 0.3793 31.91 53.58 4.82
AdaptiveEntropy 0.3394 0.7614 0.3110 18.87 31.70 6.89
AdaptiveSentenceLen 0.25860.89900.254510.28 18.251.90
HierarchicalParentChild0.50410.72040.430449.94 91.86 14.66
Contextual 0.3687 0.6622 0.3117 26.37 45.50 10.88
TopicBased 0.3880 0.6147 0.3169 26.84 60.3015.55
very little content and so stay close to the NOFILTERupper
bound. Recall is the one area where CACD trails every base-
line in this run (0.7001, at most 0.015 below NEREXACT),
read as the cost of removing more content than any other
method tested rather than a sign of weaker retrieval.
Table III breaks these results down by chunking strat-
egy. HIERARCHICALPARENTCHILDhas the highest Precision
(0.5041) and IoU (0.4304), along with the highest drop rate
(14.66%) and the largest, slowest index, since its parent and
child spans overlap on purpose, exactly the kind of overlap
CACD is designed to catch. ADAPTIVESENTENCELENshows
close to the opposite pattern: the highest Recall (0.8990), the
smallest index, and the lowest drop rate (1.90%), since its
chunks are already short and largely non-redundant. TOP-
ICBASEDhas the highest drop rate overall (15.55%), suggest-
ing its topic-grouped chunks carry more repeated content than
simpler strategies.
Overall, in these experiments CACD’s effect depends on
how much real overlap a chunking strategy introduces: strate-
gies that create structured or repeated content see the largest
benefit, while strategies that already produce short, separate
chunks see very little change.
V. CONCLUSION
This paper presented CACD, a chunk deduplication method
that scores each new chunk against a persistently growing
index using a cross-encoder, combining a calibrated duplicate
probability with an attention-derived New Information Score
(NIS) and a majority vote across several retrieved candidates
rather than a single best match.

In this evaluation, across five filtering baselines, nine chunk-
ing strategies, and eighteen configurations on the full SQuAD
1.1 validation set, CACD reaches the highest drop rate of any
method tested (9.75%), while using less index storage than
every baseline except Similarity and ingesting faster than both
other semantic-level baselines (Similarity, NERExact). Preci-
sion and IoU land above Similarity and NERExact in this run,
though below the three baselines that filter close to nothing and
stay near the NoFilter upper bound by construction; Recall
trails every other method by at most 0.0056 relative to the
closest baseline (NERExact).
By chunking strategy, the benefit in these experiments is
largest for HierarchicalParentChild, where deliberate paren-
t/child overlap is exactly the case a pooled-vector score is most
likely to mistake for redundancy, termed here false-redundancy
collapse, and smallest for strategies like AdaptiveSentenceLen
that already produce short, largely disjoint chunks.
However, three limitations are worth considering:
•The cost ratio and NIS thresholds were chosen by com-
paring a handful of settings on this one dataset rather than
through a principled calibration procedure, and whether
a less aggressive threshold could narrow the Recall gap
while keeping most of the storage and Precision/IoU
benefit remains untested.
•A chunk voted DROP is discarded outright, including
any small amount of content it might carry that is not
explained by other kept chunks; this loss of partial infor-
mation may help explain why CACD’s Precision, Recall,
and IoU trail NoFilter’s upper bound in this evaluation.
•The probability of duplication depends on the quality of
the cross-encoder model. By observing several experi-
ments, we conclude that many cross-encoder models such
asnli-deberta-v3-small, stsb-roberta-large, msmarco-
MiniLM-L6-en-de-v1, qnli-distilroberta-base and qnli-
electra-base[27] provide better outcomes. And, in order
to increase recalls, precision and IoU metrics more, a
fine-tune processing is highly recommended.
REFERENCES
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H.
K¨uttler, M. Lewis, W. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,”
inAdvances in Neural Information Processing Systems (NeurIPS), vol.
33, 2020, pp. 9459–9474.
[2] Trang, “RAG Pipeline Diagram: A Beginner’s Guide to How It
Works,”Designveloper Blog, Aug. 18, 2025. [Online]. Available: https://
www.designveloper.com/blog/rag-pipeline-diagram/. [Accessed: Jul. 14,
2026].
[3] K. Lee, D. Ippolito, A. Nystrom, C. Zhang, D. Eck, C. Callison-
Burch, and N. Carlini, “Deduplicating Training Data Makes Language
Models Better,” inProc. 60th Annual Meeting of the Association for
Computational Linguistics (ACL), 2022, pp. 8424–8445.
[4] R. Shah, K. Mukherjee, A. Tyagi, S. K. Karnam, D. Joshi, S. Bhosale,
and S. Mitra, “R2D2: Reducing Redundancy and Duplication in Data
Lakes,”Proc. ACM Manag. Data, vol. 1, no. 4, Art. no. 268, Dec. 2023.
[5] S. R. Bhat, M. Rudat, J. Spiekermann, and N. Flores-Herr, “Rethinking
Chunk Size For Long-Document Retrieval: A Multi-Dataset Analysis,”
arXiv preprint arXiv:2505.21700, 2025.
[6] V . J. J. Kreileder, J. Reisinger, and A. Fischer, “Evaluating Chunking
Strategies for Retrieval-Augmented Generation on Academic Texts,”
arXiv preprint arXiv:2607.01852, 2026.[7] R. Qu, R. Tu, and F. S. Bao, “Is Semantic Chunking Worth the
Computational Cost?” inFindings of the Association for Computational
Linguistics: NAACL 2025, Albuquerque, NM, USA, Apr. 2025, pp.
2155–2177.
[8] M. ´Smigielski, M. Rajkowski, M. Zbrocki, M. Bernacki-Janson, K.
Kunicki, J. Godziszewska, M. Piasecki, and K. Wojtasik, “Chunking
Methods on Retrieval-Augmented Generation – Effectiveness Evalu-
ation Against Computational Cost and Limitations,”arXiv preprint
arXiv:2606.00881, 2026.
[9] P. Kondapalli, M. Z. R. Saimon, S. Das Polok, G. S. Chakraborty,
M. A. Abrar, and S. Batra, “Query-Aware Adaptive Chunking for
RAG in Mental Health and Well-Being Policy Question Answering,”
in2026 5th International Conference on Innovative Practices in Tech-
nology and Management (ICIPTM), Noida, India, 2026, pp. 1–6, doi:
10.1109/ICIPTM69057.2026.11465692.
[10] P. Elchafei, H. Emam, M. Alansary, M. Swain, and M. Schedl, “H-RAG
at SemEval-2026 Task 8: Hierarchical Parent–Child Retrieval for Multi-
Turn RAG Conversations,”arXiv preprint arXiv:2605.00631, 2026.
[11] J. Singh and C. Merola, “Reconstructing Context: Evaluating Ad-
vanced Chunking Strategies for Retrieval-Augmented Generation,”arXiv
preprint arXiv:2504.19754, 2025.
[12] S. Hadawle, “Different Chunking Strategies to Improve Retrieval-
Augmented Generation (RAG),”GoPenAI Blog, Oct. 17,
2025. [Online]. Available: https://developer.nvidia.com/blog/
finding-the-best-chunking-strategy-for-accurate-ai-responses/.
[Accessed: Jul. 14, 2026].
[13] D. Berdyugina, A. Cohen, and Y . Rioual, “Reducing Redundancy
in Retrieval-Augmented Generation through Chunk Filtering,”arXiv
preprint arXiv:2604.24334, 2026.
[14] Yu. A. Malkov and D. A. Yashunin, “Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World
Graphs,”IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, vol. 42, no. 4, pp. 824–836, 2020.
[15] P. Cunningham and S. J. Delany, “k-Nearest Neighbour Classifiers:
2nd Edition (with Python Examples),”arXiv preprint arXiv:2004.04523,
2020.
[16] L. Mahon and M. Lapata, “K*-Means: A Parameter-free Clustering
Algorithm,”arXiv preprint arXiv:2505.11904, 2025.
[17] M. Vast, B. Van Cooten, L. Soulier, and B. Piwowarski, “Understand-
ing Matching Mechanisms in Cross-Encoders,” inProc. Workshop on
Explainability in Information Retrieval (WExIR25), SIGIR 2025, New
York, NY , USA, 2025.
[18] H. Ananthakrishnan, J. Dolby, H. Kokel, H. Samulowitz, and K. Srinivas,
“Can Cross Encoders Produce Useful Sentence Embeddings?”arXiv
preprint arXiv:2502.03552, 2025.
[19] S. Vajapeyam, “Understanding Shannon’s Entropy metric for Informa-
tion,”arXiv preprint arXiv:1405.2061, 2014.
[20] G. Louppe, “Understanding Random Forests: From Theory to Practice,”
Ph.D. dissertation, Univ. of Li `ege, Li `ege, Belgium, 2014,arXiv preprint
arXiv:1407.7502.
[21] M. Abdoli, M. Akbari, and J. Shahrabi, “Bagging Supervised Autoen-
coder Classifier for Credit Scoring,”arXiv preprint arXiv:2108.07800,
2021.
[22] L. Yang, Q. Chen, Y . Zhang, Y . Shi, R. Pan, A. M. Lipani, and W.
Y . Wang, “Contextual Retrieval Augmented Generation,”arXiv preprint
arXiv:2309.09564, 2023.
[23] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+
Questions for Machine Comprehension of Text,” inProc. 2016 Conf.
Empirical Methods in Natural Language Processing (EMNLP), 2016,
pp. 2383–2392.
[24] C. Elkan, “The Foundations of Cost-Sensitive Learning,” inProc. 17th
Int. Joint Conf. Artificial Intelligence (IJCAI), 2001, pp. 973–978.
[25] A. Abbas, K. Tirumala, D. Simig, S. Ganguli, and A. S. Morcos,
“SemDeDup: Data-efficient learning at web-scale through semantic
deduplication,”arXiv preprint arXiv:2303.09540, 2023.
[26] T. M. Cover and J. A. Thomas,Elements of Information Theory, 2nd
ed. Hoboken, NJ, USA: Wiley-Interscience, 2006.
[27] Sentence Transformers, “cross-encoder (Sentence Transformers – Cross-
Encoders),”Hugging Face, n.d. [Online]. Available: https://huggingface.
co/cross-encoder/models. [Accessed: Jul. 26, 2026].