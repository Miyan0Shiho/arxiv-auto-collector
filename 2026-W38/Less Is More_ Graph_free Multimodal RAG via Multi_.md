# Less Is More: Graph-free Multimodal RAG via Multi-signal Late Fusion

**Authors**: Tithi Rakshit, Hongkuan Zhou, Lavdim Halilaj, Yuqicheng Zhu

**Published**: 2026-09-16 20:54:08

**PDF URL**: [https://arxiv.org/pdf/2609.19417v1](https://arxiv.org/pdf/2609.19417v1)

## Abstract
Graph-based retrieval-augmented generation (RAG) is widely used for multimodal, cross-document question answering. However, building corpus-level graphs is expensive, slow to query, and difficult to maintain. We present TrioRAG, a graph-free multimodal framework that integrates evidence from three complementary signals: the question, the anchor image, and a VLM-enhanced query generated from both. Each signal retrieves independently over a shared multi-vector index of page text and page images, and the results are combined through late fusion. Further, we introduce AutoQA, a multimodal automotive benchmark whose questions are grounded in noisy, web-sourced images rather than clean document-sourced figures. Its questions require reasoning across manuals. We position it as a model-curated testbed rather than a human-validated gold standard. Across three benchmarks, TrioRAG matches or outperforms graph-based systems while reducing total cost and accelerating per-query inference by 1.6-2.3 times. By construction, AutoQA grounds its questions in out-of-corpus web images. In this setting image retrieval reaches only 19.3% document-level recall, while text-derived signals, especially the VLM-enhanced query, keep retrieval robust.

## Full Text


<!-- PDF content starts -->

Less Is More: Graph-free Multimodal RAG
via Multi-signal Late Fusion
Tithi Rakshit1*Hongkuan Zhou2,3Lavdim Halilaj2Yuqicheng Zhu2,3
1University of Tübingen, Tübingen, Germany
2Corporate Research, Robert Bosch GmbH, Renningen, Germany
3University of Stuttgart, Stuttgart, Germany
tithi.rakshit@student.uni-tuebingen.de
{hongkuan.zhou, lavdim.halilaj, yuqicheng.zhu}@de.bosch.com
Abstract
Graph-based retrieval-augmented generation
(RAG) is widely used for multimodal, cross-
document question answering. However, build-
ing corpus-level graphs is expensive, slow to
query, and difficult to maintain. We present
TrioRAG, a graph-free multimodal framework
that integrates evidence from three complemen-
tary signals: the question, the anchor image,
and a VLM-enhanced query generated from
both. Each signal retrieves independently over
a shared multi-vector index of page text and
page images, and the results are combined
through late fusion. Further, we introduce
AutoQA, a multimodal automotive benchmark
whose questions are grounded in noisy, web-
sourced images rather than clean document-
sourced figures. Its questions require reasoning
across manuals. We position it as a model-
curated testbed rather than a human-validated
gold standard. Across three benchmarks, Tri-
oRAG matches or outperforms graph-based
systems while reducing total cost and accel-
erating per-query inference by 1.6-2.3 ×. By
construction, AutoQA grounds its questions in
out-of-corpus web images. In this setting image
retrieval reaches only 19.3% document-level re-
call, while text-derived signals, especially the
VLM-enhanced query, keep retrieval robust.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) has become the standard paradigm for
grounding large language models in external knowl-
edge. For multimodal and cross-document rea-
soning, recent systems increasingly turn to graph-
based designs (Edge et al., 2024; Guo et al., 2025b;
Wan and Yu, 2025; Guo et al., 2025a) that build
graphs over documents, pages, entities, text, and
visual elements. Such a structure is a natural fit
for multimodal corpora, where evidence is spread
across text, figures, tables, and images.
*Work done during an internship at Robert Bosch GmbH.
a)  Graph-based RAG vs. TrioRAG.
b)  Existing benchmarks vs. AutoQA.Graph-based RAGComplex pipeline=> High Indexing & Inference timeExisting benchmarksQuery images takenfrom sourcedocuments.Simple Multimodal RAG=> Low Indexing & Inference time
in-the-wild queryimagesCross document QATrioRAG (Ours) Graph build is slow.No costly Graph creation.AutoQA (Ours)Queries don't reflectreal-world conditions.Realistic automotivetroubleshooting.
Why is my car leakingoil under the engine?Figure 1:Vehicle Diagnostics: RAG Pipelines and
Benchmarks: a)Overhead-free: TrioRAG achieves
strong performance without graph construction or com-
plex retrieval pipelines. b)Out-of-corpus images: Au-
toQA uses messy, web-sourced diagnostic images rather
than images drawn from the source corpus.
However, this expressivity is costly (Figure 1).
Graph-based systems require expensive extraction,
alignment, and construction, and are difficult to
update, as corpus changes can force re-extraction,
or partial rebuilds (Zhang et al., 2026; Peng et al.,
2026). These challenges are amplified in multi-
modal settings, where text, images, and entities
must be linked across large collections. This raises
a practical question for the graph systems and
benchmarks we study: is a corpus-level graph nec-
essary for competitive multimodal cross-document
retrieval, or can simpler retrieval methods extract
comparable evidence at lower cost?
To address this challenge, we propose TrioRAG,
a graph-free multimodal RAG framework.1Tri-
oRAG indexes page text and images in a shared
multi-vector space and retrieves three complemen-
tary signals: the text query, the anchor image, and
a VLM-enhanced query grounded on the anchor
image. The resulting rankings are fused via late fu-
1Code:https://github.com/hk-zh/TrioRAG
arXiv:2609.19417v1  [cs.CL]  16 Sep 2026

sion. By avoiding graph construction and traversal,
TrioRAG achieves performance comparable to or
better than the graph-based systems we evaluate.
Existing multimodal RAG benchmarks usually
draw query images from source documents, creat-
ing near-duplicate visual matches that can overesti-
mate the effectiveness of image retrieval compared
to out-of-corpus query images, which are common
in real-world settings. To evaluate retrieval un-
der realistic conditions, we introduce AutoQA, a
multimodal, cross-document question-answering
benchmark in the automotive domain. Each ques-
tion is grounded in a noisy, web-sourced image and
requires synthesizing evidence across two technical
manuals. AutoQA deliberately creates a domain
gap between query images and document pages,
allowing us to study retrieval when no direct visual
match exists in the corpus. Our experiments reveal
that, in this setting, image retrieval becomes sub-
stantially less reliable, while text-derived signals,
especially the VLM-enhanced query, recover much
of the missing evidence and keep retrieval robust.
Our contributions are as follows:A graph-free
frameworkthat fuses three query signals: the
text query, the anchor image query, and a VLM-
enhanced query, over a shared multi-vector index,
matching or exceeding the graph-based systems
we evaluate.AutoQA, a model-curated bench-
mark of multimodal, cross-document automotive
QA pairs grounded in noisy, web-sourced images,
with validation status discussed in the Limitations
and Ethics statements.A retrieval-reliability anal-
ysisshowing that, on AutoQA’s deliberately out-
of-corpus query images, image retrieval collapses
while text-derived signals, especially the VLM-
enhanced query, keep retrieval robust. On the
benchmarks we study, this reliability gap, rather
than the choice of fusion method, is what most
closely tracks downstream accuracy.
2 Related Work
Retrieval-augmented generation (RAG)(Lewis
et al., 2020) grounds language models in exter-
nal knowledge. It does so by retrieving relevant
passages with a dense retriever (Karpukhin et al.,
2020) and conditioning a reader on the retrieved
evidence. Fusion-in-Decoder (Izacard and Grave,
2021) and Atlas (Izacard et al., 2023) show that
jointly attending to multiple passages improves per-
formance, motivating multi-evidence readers rather
than single-match retrieval. However, these meth-ods assume textual knowledge, limiting their use
for visual documents. OCR or captioning pipelines
(Xu et al., 2020; Riedler and Langer, 2024) are
often used instead, but they flatten layouts and dis-
card visual structure.
Visual document retrieval.Recent work re-
trieves directly from visual inputs. Building on late
interaction models (Khattab and Zaharia, 2020),
ColPali (Faysse et al., 2025) uses multi-vector page
embeddings with MaxSim, avoiding OCR. Vis-
RAG (Yu et al., 2025) and M3DocRAG (Cho et al.,
2025) show strong performance in document QA
at scale, and ViDoRAG (Wang et al., 2025) adds
iterative reasoning over retrieved pages. We use
multi-vector retrieval but differ in query construc-
tion. We combine multiple complementary query
signals instead of a single representation.
Query expansion for retrieval.Query expan-
sion methods generate intermediate text to improve
retrieval. HyDE (Gao et al., 2023) prompts a lan-
guage model to generate a hypothetical document,
whose embedding is used for retrieval. We extend
this to multimodal inputs by conditioning gener-
ation on both the question and an anchor image,
grounding abstract terms in visual content. This
serves as one of several retrieval signals rather than
a standalone method.
Graph-based RAG.RAG systems increasingly
use graph structures to support retrieval and reason-
ing (Guo et al., 2025b; Zhou et al., 2025a,b, 2026;
Chaturvedi et al., 2026). In multimodal settings,
MMGraphRAG (Wan and Yu, 2025) aligns image
and text graphs via spectral clustering for entity
linking, while RAG-Anything (Guo et al., 2025a)
extends graph-based retrieval to text and vision.
V AT-KG (Park et al., 2025) constructs concept-
level graphs across modalities. Graph structure can
also be introduced after retrieval for structured rea-
soning, e.g., argumentation-based RAG (Zhu et al.,
2025). However, methods that construct corpus-
level graphs can incur substantial costs from extrac-
tion, alignment, and traversal.
Rank aggregation and fusion.Rank aggre-
gation has a long history in information retrieval
(Li et al., 2025; Santra et al., 2025; Rackauckas,
2024). Classical methods like Borda count (Aslam
and Montague, 2001) and Reciprocal Rank Fusion
(Cormack et al., 2009) combine sparse and dense
signals. Social-choice aggregation has also been
used to robustly combine LLM outputs (Potyka
et al., 2024). Recent work extends them to mul-
timodal settings (Samuel et al., 2025). We use

unsupervised fusion methods and study how they
behave under heterogeneous signal reliability, es-
pecially with noisy query images.
Benchmarks.Document QA benchmarks
evolve from single-page (Mathew et al., 2021) to
multi-page (Cho et al., 2025) and long-document
reasoning (Ma et al., 2024; Li et al., 2024). How-
ever, most assume clean inputs and aligned visual
evidence, limiting realism. To address this gap, Au-
toQA introduces cross-document automotive rea-
soning with web-sourced question images.
3 Methodology
Given a question q, an anchor image v, and a corpus
ofMpagesP={p j}M
j=1, our pipeline retrieves
a set of pages RK⊂ P and passes them, together
with (q, v) , to a vision-language reader that pro-
duces the answer a. The pipeline has three stages:
hybrid indexing, multi-signal retrieval, and late fu-
sion, followed by a generation step, with no graph
construction at any point (Figure 2). We describe
the default multi-vector configuration here; a dense
variant is evaluated as an ablation (Section 6.2.1).
Hybrid indexing.Each page pjis stored under
two views: its structured text xj, extracted with
MinerU (Wang et al., 2024), and its rendered page
image yj. Both are encoded with Jina-v4 (Gün-
ther et al., 2025), a unified multimodal encoder
that maps text and images into a shared embed-
ding space, and they are added to a single index.
Keeping a text and an image view of every page
preserves both textual content and visual layout
without any structural parsing, enabling the query
match of either view.
Multi-signal queries.We derive three retrieval
signals from the multimodal question (q, v) , each
surfacing different evidence. A text query ( q), an
image query ( v), and an enhanced query ei.e., a
short textual description produced by prompting a
VLM with both qandv(prompt in Appendix A.4).
The enhanced question grounds abstract question
terms in what is visible in the anchor image and
is embedded as text. Combining the three signals
improves recall over any single one (Table 6).
Late-interaction retrieval.Each question is a
set of token embeddings. For a question Qand
an index entry cwith embedding set Vc, we score
relevance by late interaction (MaxSim) (Khattab
and Zaharia, 2020; Günther et al., 2025):
s(Q, c) =X
u∈Qmax
w∈V c⟨u, w⟩.(1)Because text and image entries share one embed-
ding space, every question scores against both
views, so a single ranked list can contain text-view
and image-view entries for different pages. Each of
q,v, anderetrieves its top- 10entries under s(·,·) ,
yielding three ranked listsT,I, andE.
Late fusion.The three lists are merged at page
level into a single fused ranking. Our default aggre-
gator is Reciprocal Rank Fusion (RRF) (Cormack
et al., 2009): for a page pwith rank rL(p)in list L,
RRF(p) =X
L∈{T,I,E}1[p∈L]
k+r L(p), k= 60,
(2)
and the top K=10 pages by RRF(p) form the re-
trieved set RK. RRF needs no score normalisation
and prevents any single list from dominating. Be-
cause the best way to combine lists depends on how
reliable each one is, we also evaluate three alterna-
tives (Borda count, mean of z-normalised scores,
and mean of normalised ranks) in Section 6.2.4.
Answer generation.The VLM reader is given
the question and anchor image (q, v) together with
the fused context, and produces the answer a. Be-
cause fusion merges the lists at the page level, each
page in RKcontributes a single view, i.e., the page
text or the page image, whichever entry ranked
highest for that page. Hence, the context is a mix
of page-text passages and page images.
4 AutoQA Benchmark
AutoQA is a multimodal cross-document question-
answering benchmark for the automotive domain.
It is built by an automated multi-stage pipeline,
with a model-based filtering step after each stage;
Appendix A.1 describes the pipeline in more detail.
Document-pair construction.We start with
200 automotive workshop manuals rendered into
page images. A vision-language model (Qwen3-
VL-32B) produces a structured summary of each
manual; the summaries are embedded with Jina-
v4 and clustered with UMAP (McInnes et al.,
2018) and HDBSCAN (McInnes et al., 2017).
Within each cluster, candidate document pairs are
formed from the top nearest neighbours of every
manual by summary-embedding cosine similarity,
weighted by cluster-membership confidence. A
VLM (Qwen3-VL-32B) then removes pairs that
lack shared technical grounding or cross-document
complementarity, leaving 556 pairs.
Image grounding.Rather than reuse figures
from the manuals, AutoQA grounds each question

Vector StoreEmbedding Model
IndexExtraction
Documents
PDF to Images Page as ImageOCR Extraction      Query
      (Text)T
PDF       Enhanced
    QueryT      Query
       (Image)
Top KItem B
....
Item KItem C
Item B
....
Item KItem D
Item B
....
Item KItem AItem C
Item AItem B
Fusion ModuleText itemImage itemAgent
Agent User
Hybrid
IndexingFigure 2: Our TrioRAG pipeline. The multimodal question is turned into three signals: text, image, and VLM-
enhanced text, retrieved independently over a hybrid multi-vector index and merged into a single fused ranking. The
top-ranked pages are passed to the VLM reader alongside the anchor image to generate the answer.
in a separately sourced image, creating a deliberate
domain gap between question images and docu-
ment pages. From the document summaries we
generate mechanic-oriented image-search queries,
post them to DuckDuckGo Image Search, and filter
the returned web images with a VLM to discard
noise and mismatches.
Question generation.For each validated pair
and its grounding image, GPT-4o generates cross-
document questions from the page images, ex-
tracted text, and the image, producing 1,449 candi-
date QA pairs. Each question is labelled as factual,
procedural, or diagnostic and is constructed to re-
quire evidence from both manuals.
Filtering.GPT-4o scored every candidate on
factual grounding, answer accuracy, answer com-
pleteness, cross-document dependency, absence of
source leakage, and image grounding. Retaining
items meeting all criteria at maximum quality re-
duced the 1,449 candidates to 885 final QA pairs
across 405 distinct images. A 50-item human vali-
dation of the final set (Appendix A.2) confirms high
answerability (80%) and reference-answer com-
pleteness (86%), with lower strict cross-document
necessity (46%) and image grounding (52%).
5 Evaluation
5.1 Datasets
We evaluate on three benchmarks spanning differ-
ent domains and question types.MMLongBench
(Ma et al., 2024) has 1,091 questions over long,
multi-page documents; with no anchor figures, it is
a text-centric setting.M3SciQA(Li et al., 2024)
has 452 cross-document scientific QA pairs, eachgrounded in an anchor figure and hinging on tables,
charts, and figures.AutoQA(ours) has 885 cross-
document automotive QA pairs; each is grounded
in a web-sourced image and needs evidence from
two manuals, with question images that are noisy
and out-of-domain relative to the document pages.
5.2 Implementation details
We usejina-embeddings-v4in multi-vector mode
(128-dim/token) by default, with dense mode
(2048-dim) as an ablation. Page text is extracted
with MinerU (Wang et al., 2024) and indexed with
full-page images in one Qdrant collection (top-10
per signal, RRF k=60 ; Section 3). The enhanced
question is generated by the VLM reader,Qwen2.5-
VL-7B-Instruct, run on a single NVIDIA H200
GPU. For a controlled comparison, TrioRAG, and
all the baselines use the same Qwen2.5-VL-7B
reader and the same number of retrieved pages,
so accuracy differences reflect retrieval and fusion
rather than the reader or context size.
5.3 Evaluation metric
We report mean accuracy under an LLM-as-judge
protocol (Zheng et al., 2023) with GPT-4o-mini.
Given the question, ground-truth answer, and pre-
diction, the judge returns a binary score, counting
factually consistent predictions correct, including
partial answers with the key information and cor-
rect refusals on unanswerable questions. Since
AutoQA is built with GPT-family models, we use
the same judge throughout for comparability and
note this dependence in the Limitations.

Method Index MMLB M3Sci AutoQA
MMGraphRAG D 26.95 – –
RAG-Anything D 27.22 28.76 58.19
Naive RAG + Concat D 33.73 31.42 55.63
Naive RAG + Concat M 35.56 20.69 45.31
TrioRAG (Ours) D 33.73 51.99 58.92
TrioRAG (Ours) M35.56 54.86 60.11
Table 1: Accuracy (%) across benchmarks. Index: D
= dense, M = multivec. Graph baselines use dense
indexing; MMGraphRAG is run on MMLongBench.
6 Results
6.1 Main Results
Table 1 compares TrioRAG against two graph-
based systems, RAG-Anything (Guo et al., 2025a)
and MMGraphRAG (Wan and Yu, 2025), that sup-
port only dense retrieval; for fairness, we report
TrioRAG in both dense and multi-vector modes.
TrioRAG is best across all three benchmarks. Fur-
thermore, even Naive RAG + Concat (combining
signals before retrieval without RRF) outperforms
both graph systems on MMLongBench, suggesting
that graph construction alone does not inherently
improve retrieval quality on text-centric tasks.
The gap is largest on M3SciQA, where TrioRAG
nearly doubles RAG-Anything. As M3SciQA is
grounded in figures, tables, and charts, direct im-
age retrieval and VLM-enhanced queries exploit
visual evidence that graph-based text extraction
discards. Feeding full-page images to the reader
amplifies this advantage. On AutoQA, the gap nar-
rows to 1.92pp (McNemar p=0.34 , within noise
under GPT-4o-mini; under a different-family judge
it widens to +8.14 ,p<10−5, Table 3), as answers
are mostly stated in text and both systems rely on
text retrieval; we still edge out the graph baseline
while being far cheaper (Table 2), the trade-off we
target. Separate retrieval followed by fusion also
outperforms Naive RAG + Concat. Concatenating
text and image tokens weakens MaxSim for both
modalities, causing a sharp drop for M3SciQA un-
der multi-vector indexing. On MMLongBench, all
variants perform identically as the absence of an
anchor image collapses all methods to text-only
retrieval.
Efficiency.Table 2 reports total wall time (index-
ing + inference) on a single H200, with identical
settings across methods. Our pipeline completes
in under 6 hours in all settings, achieving a 12 ×
speedup on AutoQA, 20 ×on M3SciQA, and 65 ×
on MMLongBench. The gains come from remov-
ing graph construction: both baselines run repeatedLLM passes for entity/relation extraction and cross-
modal alignment to index the documents, a heavy
per-document cost. TrioRAG does a single embed-
ding pass at indexing and plain retrieve-and-read at
query time, without graph traversal. It is also faster
at inference than RAG-Anything. MMGraphRAG
has lower query-time latency on MMLongBench
but pays for costly graph construction, resulting in
higher total runtime while still underperforming in
accuracy (Table 1).
Benchmark Method Infer (h) Total (h)
MMLongBench MMGraphRAG1.2466.21
RAG-Anything 3.97 244.77
TrioRAG (Ours) 1.743.75
M3SciQA RAG-Anything 2.40 119.53
TrioRAG (Ours)1.48 5.93
AutoQA RAG-Anything 4.32 62.36
TrioRAG (Ours)2.39 5.31
Table 2: Total time (indexing + inference) on a single
H200. Inference time refers to the query-time cost,
excluding one-time indexing.
Judge TrioRAG RAG-Anything McNemarp
GPT-4o-mini 60.11 58.19 0.34
Opus 4.8 38.42 30.28<10−5
Table 3: Cross-family judge accuracy (%) on 885 Au-
toQA items. The system ranking is stable and the gap
becomes significant under the different-family judge.
6.2 Ablations
6.2.1 Indexing
Multi-vector indexing outperforms dense on all
three benchmarks, most on M3SciQA (Table 1),
where answers are grounded in visually distinctive
figures. MaxSim’s patch-level matching surfaces
visually relevant pages that dense pooled embed-
dings miss, giving the reader richer visual context.
We use multi-vector throughout.
6.2.2 Input representation
We compare three reader contexts (Table 4): full-
page images; extracted text with cropped figures;
and extracted text with full-page images. Full-page
images preserve layout, charts, and table structure
that text extraction and cropping discard, and win
on M3SciQA. Adding extracted text gives a further
gain where answers are often stated in text (MM-
LongBench, AutoQA) but slightly hurts on purely
figure-based M3SciQA. As AutoQA is our target
setting, we adopt it as the default.

Input Representation MMLB M3Sci AutoQA
Doc Page Image 34.7455.7559.48
Text + Cropped Image 31.53 51.42 57.95
Text + Doc Page Image35.5654.8660.11
Table 4: Accuracy (%) by reader input representation
(MMLB = MMLongBench, M3Sci = M3SciQA). We
compare three reader contexts: full document-page im-
ages; extracted text with cropped figures; and extracted
text with full-page images.
6.2.3 Question strategy
Table 5 compares the three signals under RRF. On
M3SciQA, where the anchor figures originate from
source documents and retrieves reliably, adding the
enhanced question helps (E+I over T+I by 1.99pp,
and the full triple a further 0.44pp). On AutoQA,
T+I beats E+I since web-sourced question images
mislead the VLM, making enoisy, while text stays
reliable. T+I+E is best on both and is our default.
Question Strategy M3SciQA AutoQA
T + I 52.43 59.77
E + I 54.42 58.98
T + I + E54.86 60.11
Table 5: Accuracy (%) by Query strategy. T = text, I =
image, E = enhanced query . T+I+E signal is best.
6.2.4 Retrieval Reliability and Signal
Complementarity
We measure retrieval at the document level. A
signal succeeds on a question if its top-10 contains
a page from a reference document (for M3SciQA,
the reference paper must be found; the anchor paper
is given as the query). We report Recall@10 for
each signal and their unions in Table 6.
Image retrieval works only when the query
image is in the corpus.On M3SciQA, the query
image is a figure taken directly from a corpus doc-
ument. Because of this, the image signal retrieves
that source document almost perfectly at 99.6%.
However, this near-duplicate self-matching is de-
ceptive because the actual reference paper is re-
covered only 46.7% of the time. On AutoQA, this
apparent advantage disappears. AutoQA uses web-
sourced images with no counterpart in the corpus,
causing image recall to collapse to 19.3%. This
sharp contrast highlights the core problem AutoQA
is designed to expose: realistic, out-of-corpus query
images defeat visual retrieval. Text and enhanced
text signals, by contrast, remain strong and reliable,
achieving 81.6% and 78.5% recall on AutoQA.Text-derived signals dominate but remain
complementary.Because different query signals
fail on different questions, their union outperforms
any single signal (87.1% recall on AutoQA and
97.8% on M3SciQA). However, these gains come
almost entirely from text and enhanced signals.
T∪E alone reaches 97.3% on M3SciQA and 86.6%
on AutoQA, and adding the image signal con-
tributes little beyond that. This pattern provides an
empirical basis for retrieving multiple text-derived
query signals rather than relying on a single repre-
sentation. Ultimately, these results foreshadow the
limited standalone value of image retrieval when
handling realistic queries.
Recall@10 M3SciQA AutoQA
Per signal
T 95.4 81.6
I 46.719.3
E 91.2 78.5
Combined
T∪E 97.3 86.6
T∪I∪E97.8 87.1
Image-Miss subset only
T – 78.9
T∪E –84.0
Table 6: Document-level Recall@10 (%). Image re-
trieval collapses on AutoQA’s out-of-corpus query im-
ages; the multi-signal union recovers most of the loss.
The enhanced query rescues retrieval when
the image signal fails.The clearest evidence for
the enhanced query is on the questions where im-
age retrieval fails entirely. On the 714 AutoQA
questions in this regime (80.7% of the benchmark),
adding the enhanced query to text raises reference
recall from 78.9% to 84.0%. The enhanced query
recovers 37 questions that text alone misses, con-
tributing genuinely complementary evidence rather
than redundantly re-ranking the same documents
text already finds.
Detailed retrieval metrics.Table 7 and Table 8
report per-signal and fused document-level metrics
(R@1, R@5, R@10, MRR@10, nDCG@10) with
95% bootstrap confidence intervals (10k resam-
ples). We omit MMLongBench, which has no an-
chor image: IandEare undefined and all aggrega-
tors collapse to the text signal (R@10 83.0%). No-
tably, M3SciQA reference retrieval is largely a text
task: text alone already reaches 95.4 R@10, leav-
ing little headroom for the other signals, whereas
on AutoQA the text signal is lower (81.6) and the
enhanced query adds complementary coverage.

Run R@1 R@5 R@10 MRR@10 nDCG@10
T 69.0 94.5 95.4 80.9 84.6
I 2.2 46.7 46.7 20.6 27.2
E 60.6 91.2 91.2 74.9 79.1
RRF 48.0 96.9 96.9 70.5 77.3
Borda 49.6 97.1 97.8 71.8 78.5
Score-mean 33.6 96.7 97.6 63.3 72.1
Rank-mean 38.7 96.0 97.6 64.9 73.3
Table 7: M3SciQA ( n=452 ) document-level retrieval
metrics (%).
Run R@1 R@5 R@10 MRR@10 nDCG@10
T 31.4 74.2 81.6 48.8 45.5
I 4.4 14.7 19.3 8.6 8.6
E 31.0 71.9 78.5 47.3 44.0
RRF 33.1 72.3 79.0 49.2 44.3
Borda 34.0 71.4 82.1 49.9 46.1
Score-mean 11.4 61.4 81.7 31.8 35.9
Rank-mean 11.3 59.8 80.8 31.1 35.1
Table 8: AutoQA ( n=885 ) document-level retrieval
metrics (%).
Fusion method has little effect.Given reliable
signals, the choice among rank aggregators mat-
ters far less than the signals themselves. Across
four unsupervised aggregators (Borda count, score
mean, rank mean, RRF), accuracy differences are
not statistically significant on either benchmark (Ta-
ble 10; paired tests, Borda vs. RRF: p≈0.31 on
M3SciQA, p≈0.45 on AutoQA). RRF is also in-
sensitive to the rank constant on AutoQA (accuracy
60.11,60.0,59.77 atk=60,15,20 ). We therefore
adopt RRF, the simplest option requiring no score
normalisation, as the default.
Fusion shifts early ranks, not coverage.The
detailed metrics (Tables 7–8) refine this picture.
Across the four aggregators, coverage barely moves
(AutoQA R@10 spans only 3.1pp), whereas early
ranks vary widely (MRR@10 spans 18.8pp). On
M3SciQA, fusing the three signals lowers text-only
R@1 from 69.0 to 48.0 while raising R@10 from
95.4 to 96.9: fusion trades top-rank precision for
broader coverage. Because the reader consumes the
full top-10, coverage rather than early-rank order
is what drives downstream accuracy, which is why
the aggregators are statistically indistinguishable
in accuracy (Table 10).
Retrieval success drives accuracy.To connect
retrieval quality directly to answers, we split Tri-
oRAG’s accuracy (with the fixed Qwen2.5-VL-7B
reader) by whether the gold document was retrieved
in the top-10 (Table 9). On every benchmark and
under both judges, retrieving the gold document
raises accuracy substantially: by +15.5 pp (GPT-Bench Judge Acc ret Acc miss
AutoQA GPT-4o-mini 63.4 47.8
AutoQA Opus 4.8 43.2 20.4
M3SciQA GPT-4o-mini 53.2 14.3
MMLB GPT-4o-mini 38.0 13.0
Table 9: TrioRAG accuracy (%) with the fixed Qwen2.5-
VL-7B reader, split by whether the gold document was
in the retrieved top-10 (Acc ret) or missed (Acc miss); the
Judgecolumn is the LLM-as-judge that scored correct-
ness (AutoQA shown under both model families).
Fusion Method M3SciQA AutoQA
Borda57.08(52.7–61.7)59.66(56.4–62.9)
Score Mean 56.86(52.2–61.5)58.87(55.6–62.2)
Rank Mean 55.31(50.7–60.0)59.55(56.3–62.9)
RRF 54.86(50.4–59.5)60.11(56.8–63.3)
Table 10: Accuracy (%) by fusion method with 95%
bootstrap CIs (10k resamples).
4o-mini) and +22.8 pp (Opus 4.8) on AutoQA,
+38.9 pp on M3SciQA, and +25.0 pp on MMLong-
Bench. Retrieval outcome thus moves accuracy
far more than the choice of aggregator (a 1.24pp
spread on AutoQA, Table 10). The analysis is
observational (accuracy tracks rather than causes
retrieval success), but it reinforces that retrieval
reliability governs downstream accuracy.
7 Conclusion
In this paper, we propose TrioRAG, a graph-free,
multi-signal pipeline that matches or exceeds the
graph-based systems we evaluate. We evaluate
it on three benchmarks, with per-query inference
1.6 to 2.3 ×faster (Table 2). Our central finding
is that retrieval reliability matters more for down-
stream accuracy than the choice of fusion method:
text-derived signals keep retrieval robust where
image retrieval collapses on out-of-corpus query
images. Because indexing is a single embedding
pass, the graph-free design also keeps the corpus
cheap to update: adding a document requires only
re-embedding the changed pages, with no entity
re-extraction, cross-modal re-alignment, or graph
rebuilding. These results suggest that, for multi-
modal cross-document QA, retrieval-signal quality
matters more than pipeline complexity. Promis-
ing directions include stronger visual signals for
out-of-corpus queries and adaptive, per-question
weighting of the text, image, and enhanced signals.

Limitations
Model-curated benchmark.AutoQA is gener-
ated and filtered by vision-language models, so
reported accuracies reflect a model-curated refer-
ence rather than a human gold standard. To gauge
quality, we ran a 50-item human validation (Ap-
pendix A.2): answerability (80%) and reference-
answer completeness (86%) are high, while strict
cross-document necessity (46%) and image ground-
ing (52%) are lower. These measured rates should
be kept in mind when reading AutoQA results; a
larger study with inter-annotator agreement is left
to future work.
Judge and construction share a model fam-
ily.Answers are produced by Qwen2.5-VL-7B and
judged by GPT-4o-mini, so there is no answerer–
judge circularity; however, AutoQA is generated
by GPT-4o and judged by GPT-4o-mini, both in the
GPT family, which may inflate its absolute scores.
To bound this, we re-judged all AutoQA items with
a different-family judge (Claude Opus 4.8, Table 3):
absolute scores fall for all systems, but the rank-
ing is unchanged and the TrioRAG–RAG-Anything
gap grows and becomes significant, consistent with
GPT-family inflation rather than a reversal. We
keep a single judge across datasets for comparabil-
ity.
Fusion choice.The four aggregators we com-
pare do not differ significantly in accuracy (Ta-
ble 10). We default to RRF because it needs no
score normalization and keeps any single signal
from dominating the fused ranking, a useful prop-
erty given the heterogeneous reliability of our three
signals; we do not claim it is optimal.
Single component choices.All methods (Tri-
oRAG and the graph-based baselines) share the
same reader (Qwen2.5-VL-7B) and encoder (Jina-
v4), so these fixed components do not explain the
accuracy differences, which stem from retrieval and
fusion design. Absolute accuracies would change
with a stronger reader; whether the relative gains
hold across reader and encoder families is left to
future work.
Ethics Statement
Source images and licensing.AutoQA grounds
each question in a web image (via DuckDuckGo
Image Search) that we did not author and that may
carry copyright. Rather than redistribute the files,
we release each image’s source URL and a SHA-
256 content hash, so users fetch images from theiroriginal sources and can verify they match ours
even if some links later break.
Source documents and release.The workshop
manuals are proprietary service documents, so we
do not redistribute them. Image URLs and content
hashes carry no proprietary content and will be
released; release of the QA items and document
references is subject to institutional review, and
we will release the maximally reproducible subset
permitted.
Model-curated construction.AutoQA is built
with Qwen3-VL-32B (summaries, pairing, image
search and verification) and GPT-4o (QA genera-
tion and verification); as noted in the Limitations,
it is a model-curated testbed rather than a human-
validated gold standard and its absolute accuracies
should be read accordingly.
Intended use and risks.AutoQA targets re-
trieval and reasoning over automotive technical
documentation. It is a research benchmark and is
not intended to provide safety-critical repair guid-
ance; answers produced by systems evaluated on it
should not be relied upon for real vehicle servicing
without expert verification.
AI assistance.The authors used AI-based tools
to assist with drafting and editing the manuscript.
All scientific claims, experimental design, and anal-
ysis are the authors’ own, and the authors take full
responsibility for the content.
Acknowledgments
The authors thank the International Max Planck
Research School for Intelligent Systems (IMPRS-
IS) for supporting Yuqicheng Zhu and Hongkuan
Zhou.
References
Javed A Aslam and Mark Montague. 2001. Models
for metasearch. InProceedings of the 24th annual
international ACM SIGIR conference on Research
and development in information retrieval, pages 276–
284.
Prateek Chaturvedi, Yuqicheng Zhu, Hongkuan Zhou,
Dongzhuoran Zhou, Yunjie He, Steffen Staab, Fei
Du, Jie Tang, and Evgeny Kharlamov. 2026. SCAIR:
Schema-conditioned agentic iterative reasoning for
enterprise knowledge graphs. InProceedings of the
64th Annual Meeting of the Association for Computa-
tional Linguistics (Volume 6: Industry Track), pages
1089–1104, San Diego, California, USA. Association
for Computational Linguistics.

Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He,
and Mohit Bansal. 2025. M3DocVQA: Multi-modal
multi-page multi-document understanding. InPro-
ceedings of the IEEE/CVF International Conference
on Computer Vision Workshops (ICCVW): Findings
of ICCV, pages 6237–6247.
Gordon V Cormack, Charles LA Clarke, and Stefan
Buettcher. 2009. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In
Proceedings of the 32nd international ACM SIGIR
conference on Research and development in informa-
tion retrieval, pages 758–759.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2025. Colpali: Efficient document retrieval with
vision language models. InInternational Conference
on Learning Representations (ICLR).
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Michael Günther, Saba Sturua, Mohammad Kalim
Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang,
Sedigheh Eslami, Scott Martens, Maximilian Werk,
Nan Wang, and 1 others. 2025. jina-embeddings-v4:
Universal embeddings for multimodal multilingual
retrieval. InProceedings of the 5th Workshop on
Multilingual Representation Learning (MRL 2025),
pages 531–550.
Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and
Chao Huang. 2025a. Rag-anything: All-in-one rag
framework.arXiv preprint arXiv:2510.12323.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2025b. Lightrag: Simple and fast retrieval-
augmented generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025, pages
10746–10761.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information
processing systems, 33:9459–9474.
Chuhan Li, Ziyao Shangguan, Yilun Zhao, Deyuan Li,
Yixin Liu, and Arman Cohan. 2024. M3sciqa: A
multi-modal multi-document scientific qa benchmark
for evaluating foundation models. InFindings of the
Association for Computational Linguistics: EMNLP
2024, pages 15419–15446.
Yuxin Li, Chen Xu, Jun Xu, and Ji-Rong Wen. 2025.
Bordarag: Resolving knowledge conflict in retrieval-
augmented generation via borda voting process. In
Proceedings of the 34th ACM International Confer-
ence on Information and Knowledge Management,
pages 1737–1747.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, and 1 others. 2024. Mmlongbench-doc:
Benchmarking long-context document understanding
with visualizations.Advances in Neural Information
Processing Systems, 37:95963–96010.
Minesh Mathew, Dimosthenis Karatzas, and C.V . Jawa-
har. 2021. Docvqa: A dataset for vqa on document
images. InProceedings of the IEEE/CVF winter con-
ference on applications of computer vision, pages
2200–2209.
Leland McInnes, John Healy, and Steve Astels. 2017.
hdbscan: Hierarchical density based clustering.J.
Open Source Softw., 2(11):205.
Leland McInnes, John Healy, and James Melville. 2018.
Umap: Uniform manifold approximation and pro-
jection for dimension reduction.arXiv preprint
arXiv:1802.03426.
Hyeongcheol Park, Jiyoung Seo, MinHyuk Jang, Hogun
Park, Ha Dam Baek, Gyusam Chang, Hyeonsoo
Im, and Sangpil Kim. 2025. Vat-kg: Knowledge-
intensive multimodal knowledge graph dataset for
retrieval-augmented generation.arXiv preprint
arXiv:2506.21556.

Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Sil-
iang Tang. 2026. Graph retrieval-augmented gener-
ation: A survey.ACM Transactions on Information
Systems, 44(2):35:1–35:52.
Nico Potyka, Yuqicheng Zhu, Yunjie He, Evgeny Khar-
lamov, and Steffen Staab. 2024. Robust knowledge
extraction from large language models using social
choice theory. InProceedings of the 23rd Interna-
tional Conference on Autonomous Agents and Multi-
agent Systems, AAMAS ’24, page 1593–1601, Rich-
land, SC. International Foundation for Autonomous
Agents and Multiagent Systems.
Zackary Rackauckas. 2024. Rag-fusion: a new take
on retrieval-augmented generation.arXiv preprint
arXiv:2402.03367.
Monica Riedler and Stefan Langer. 2024. Beyond text:
Optimizing rag with multimodal inputs for industrial
applications.arXiv preprint arXiv:2410.21943.
Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco,
Kate Sanders, Seun Eisape, Arun Reddy, Alexander
Martin, Andrew Yates, Eugene Yang, Cameron Car-
penter, and 1 others. 2025. Mmmorrf: Multimodal
multilingual modularized reciprocal rank fusion. In
Proceedings of the 48th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, pages 4004–4009.
Payel Santra, Madhusudan Ghosh, Debasis Ganguly,
Partha Basuchowdhuri, and Sudip Kumar Naskar.
2025. Hf-rag: Hierarchical fusion-based rag with
multiple sources and rankers. InProceedings of the
34th ACM International Conference on Information
and Knowledge Management, pages 5202–5207.
Xueyao Wan and Hang Yu. 2025. Mmgraphrag:
Bridging vision and language with interpretable
multimodal knowledge graphs.arXiv preprint
arXiv:2507.20804.
Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan
Qu, Fukai Shang, and 1 others. 2024. Mineru: An
open-source solution for precise document content
extraction.arXiv preprint arXiv:2409.18839.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu,
Shihang Wang, Pengjun Xie, and Feng Zhao. 2025.
Vidorag: Visual document retrieval-augmented gener-
ation via dynamic iterative reasoning agents. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 9113–
9134.
Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu
Wei, and Ming Zhou. 2020. Layoutlm: Pre-training
of text and layout for document image understanding.
InProceedings of the 26th ACM SIGKDD interna-
tional conference on knowledge discovery & data
mining, pages 1192–1200.Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. Vis-
rag: Vision-based retrieval-augmented generation on
multi-modality documents. InInternational Confer-
ence on Learning Representations (ICLR).
Ze Yu Zhang, Zitao Li, Yaliang Li, Bolin Ding, and
Bryan Kian Hsiang Low. 2026. Respecting temporal-
causal consistency: Entity-event knowledge graph for
retrieval-augmented generation. InProceedings of
the 19th Conference of the European Chapter of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 2017–2054.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, and 1 oth-
ers. 2023. Judging llm-as-a-judge with mt-bench
and chatbot arena.Advances in neural information
processing systems, 36:46595–46623.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Yuan He, Jiaoyan Chen, Steffen Staab, and Evgeny
Kharlamov. 2025a. Evaluating knowledge graph
based retrieval augmented generation methods un-
der knowledge incompleteness.arXiv preprint
arXiv:2504.05163.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Hongkuan Zhou, Jiaoyan Chen, Steffen Staab, Yuan
He, and Evgeny Kharlamov. 2025b. GR-agent:
Adaptive graph reasoning agent under incomplete
knowledge. InWorkshop on Scaling Environments
for Agents.
Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang,
Hongkuan Zhou, Yuan He, Jiaoyan Chen, Steffen
Staab, and Evgeny Kharlamov. 2026. What breaks
knowledge graph based rag? benchmarking and
empirical insights into reasoning under incomplete
knowledge. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
2522–2538.
Yuqicheng Zhu, Nico Potyka, Daniel Hernández, Yuan
He, Zifeng Ding, Bo Xiong, Dongzhuoran Zhou,
Evgeny Kharlamov, and Steffen Staab. 2025. Ar-
grag: Explainable retrieval augmented generation
using quantitative bipolar argumentation. InPro-
ceedings of The 19th International Conference on
Neurosymbolic Learning and Reasoning, volume 284
ofProceedings of Machine Learning Research, pages
697–718. PMLR.
A Appendix
A.1 AutoQA construction details
AutoQA is created through an automated multi-
stage pipeline, with a model-based filtering step
applied after each stage (Figure 3).
Document selection.We begin with 200 work-
shop manuals selected from a larger HTML corpus.

Embed
Summaries
Q & AV erified
Q & A
PDF
PDF to
Images
VLM 
Summary
GenerationDocument 
SummariesClusteringDocument Pair
VLM Pair 
Validation
Validated PairsQuery
GenerationWeb Search
VLM Image
Search Query
GenerationWeb Images Document Pair +
Web Images+
VLM 
Web Image
Validation
+
Verified Web
ImagesVLM Q&A
GenerationVLM Q&A
VerificationFigure 3: The AutoQA construction pipeline: manuals are summarised and clustered to form document pairs, each
pair is grounded in a web-sourced image, and GPT-4o generates and filters cross-document QA pairs.
Documents are ranked using a composite score
that considers length, vocabulary richness, figure
and table density, heading depth, cross-references,
and automotive-term density. Near-duplicate docu-
ments are removed before selection.
Clustering and pairing.Each manual is
first summarized using Qwen3-VL-32B. The sum-
maries are embedded with Jina-v4, reduced using
UMAP ( 10components, 15neighbours, minimum
distance 0.1, cosine metric), and clustered with
HDBSCAN (minimum cluster size 3, minimum
samples 2, Excess of Mass (EOM) selection). This
process produces 17clusters, while low-confidence
assignments are treated as noise. Within each clus-
ter, candidate document pairs are formed from the
top five cosine neighbours of every document. Pairs
are scored as 0.7·cos(i, j) + 0.3·(probi·probj),
resulting in 591candidate pairs. Qwen3-VL-32B
then performs a lightweight screening step to verify
the presence of a shared technical topic and a mean-
ingful cross-document dependency. This removes
35pairs, leaving556for the next stage.
Image grounding.For each document pair,
Qwen3-VL-32B generates three mechanic-style im-
age search queries. These queries are submitted to
DuckDuckGo Image Search, and the returned pho-
tographs are filtered by Qwen3-VL-32B to ensure
they depict relevant automotive components, are
real-world images, and meet a minimum quality
threshold (confidence>0.8).Question generation and filtering.GPT-4o
generates up to three cross-document question-
answer pairs for each document pair using a tem-
perature of 0.3. Each question is grounded in a
verified image and in 20sampled pages from each
document, producing 1,449 candidate QA pairs.
GPT-4o then evaluates each candidate on factual
grounding, answer accuracy, answer completeness,
cross-document dependency, absence of source
leakage, and image grounding. Only candidates
that satisfy every criterion and receive the highest
overall quality rating are retained. Cross-document
dependency and overall quality are the most restric-
tive criteria, with pass rates of 71.8% and72.0% ,
respectively; the remaining criteria rarely eliminate
candidates.
The final dataset contains 885 QA pairs
grounded in 405distinct web-sourced images, with
multiple questions sharing the same image when
applicable. Questions are categorized as procedural
(340,38.4% ), diagnostic ( 292,33.0% ), or factual
(253,28.6%).
A.2 Human validation of AutoQA
To provide an independent, non-model check of
benchmark quality, we drew a uniform random
sample of 50final AutoQA items and had a
human annotator label each for four properties
(Table 11). Answerability ( 80%) and reference-
answer completeness ( 86%) are high, whereas
strict cross-document necessity ( 46%) and correct

image grounding ( 52%) are lower. The human
cross-document-necessity rate ( 46%) is below the
71.8% reported by our GPT-4o construction filter
(Appendix A.1); within the final 885items this field
is always true by construction, so the filter cannot it-
self validate it. Part of the gap also reflects a stricter
human criterion (both manuals must benecessary,
rather than merely combining information from
both). We therefore report these measured rates
rather than assuming uniform benchmark quality;
a larger study with inter-annotator agreement is a
natural next step.
Property Human pass rate
Answerable 40/50 (80%)
Requires both manuals 23/50 (46%)
Correctly image-grounded 26/50 (52%)
Reference answer complete 43/50 (86%)
Table 11: Human validation on a uniform random sam-
ple of 50 AutoQA items.
A.3 Qualitative Examples
We present three AutoQA items, each with the
TrioRAG prediction (multi-vector index, RRF fu-
sion, Qwen2.5-VL-7B reader) and the GPT-4o-
mini judge verdict. Every question is grounded
in a web-sourced anchor photo and needs evidence
from two manuals. The examples span the three
question types and include one failure that illus-
trates a characteristic cross-document error. Ques-
tions, reference answers, and judge rationales are
verbatim; predictions are abridged, with [. . .] mark-
ing omitted text. The anchor figures are synthetic il-
lustrations of each image type, not the actual bench-
mark images, which we release only as source
URLs and content hashes (see Ethics Statement).
Example 1: Procedural (Judged correct).The
anchor photo (Fig. 4) shows a counter-hold toolkit.
Question.In the photo, I see a counter-hold
tool similar to T10554. How do I use it to remove
the camshaft adjuster and ensure proper alignment
during installation?
Reference answer.To remove the camshaft
adjuster, fit the counter-hold tool T10554 with
T10172A to the camshaft pulley, ensuring the pins
are properly inserted into the holes. Hold the inlet
camshaft in position using these tools while loos-
ening the securing bolt of the camshaft adjuster.
During installation, ensure the guide sleeve is in-
serted in the camshaft and renew the bolt for the
camshaft adjuster, screwing it in by hand. The
Figure 4: Synthetic illustration of the anchor image used
in Example 1: a camshaft counter-hold toolkit. This is
not the actual benchmark image.
Figure 5: Synthetic illustration of the anchor image
used in Example 2: a transverse four-cylinder engine
showing the fuel rail, injector connectors, and engine-
control-unit connector.
camshaft pulleys should be able to turn without
rocking.
TrioRAG prediction (abridged).Position
counter-hold tool T10554 so its pins align with
the holes in the camshaft adjuster and secure it
with T10172A. Loosen the securing bolt one turn
without fully removing it, then withdraw the ad-
juster from the camshaft. [. . .] To reinstall, refit
T10554/T10172A to hold the camshaft, realign the
adjuster, and tighten the securing bolt without over-
tightening. [. . .] Finally, confirm the adjuster is
correctly aligned with the camshaft.
Judge (correct).“The generated answer pro-
vides a correct and detailed procedure for using the
counter-hold tool T10554 to remove and reinstall
the camshaft adjuster, ensuring proper alignment,
which matches the expected answer.”
Example 2: Factual (judged incorrect).The
anchor photo (Fig. 5) shows an engine-control-unit
connector with several fuel-injector leads.

Figure 6: Synthetic illustration of the anchor image used
in Example 3: an engine-bay fuse box beside the battery.
This is not the actual benchmark image.
Question.In the photo, I see a 91-pin connector
and several fuel injector connections. How do these
components interact in the 2.0L diesel engine and
the 1.6L petrol engine?
Reference answer.In the 2.0L diesel engine,
the 91-pin connector is linked to the engine/motor
control unit (J623), which manages the injectors
for cylinders 1 to 4 (N30, N31, N32, N33). For the
1.6L petrol engine, the 94-pin connector connects
to the engine/motor control unit (J623), controlling
the injectors similarly but with different wiring con-
figurations (T2ln, T2lo, T2lp, T2lq). Both engines
use their respective connectors to facilitate precise
fuel injection timing and control.
TrioRAG prediction (abridged).The 91-pin
connector is the interface between the ECU and the
engine’s sensors and actuators, and it controls the
fuel injectors in both the 2.0L diesel and the 1.6L
petrol engine. [. . .] In both engines the ECU sends
signals through the 91-pin connector to open and
close the injectors, with only minor differences due
to engine design.
Judge (incorrect).“The generated answer incor-
rectly refers to a 91-pin connector for both engines
instead of specifying the correct connectors for
each engine; it also omits exact details about how
the fuel injectors are linked to the different engine
configurations as specified in the expected answer.”
The model substitutes generic ECU boilerplate
for the petrol engine’s 94-pin connector and the
cylinder-specific injector labels, a failure that re-
quires jointly grounding both manuals rather than
either alone.
Example 3: Diagnostic (judged correct).The
anchor photo (Fig. 6) shows an engine-bay fuse
box beside the battery.Question.In the photo, I see the fuse holder
near the battery. Which fuses should I check to
troubleshoot the gearbox communication issue?
Reference answer.Check Fuse 12 on Fuse
Holder B (SB12) and Fuse 6 on Fuse Holder C
(SC6) as they are related to the automatic gearbox
control unit. These fuses are crucial for ensuring
proper communication between the gearbox and
other systems. Inspect them for continuity and
replace if necessary.
TrioRAG prediction (abridged).The fuse
holder near the battery is Fuse holder B (SB). The
fuse related to the automatic gearbox control unit
(J217) is Fuse 12 (SB12). With the ignition off,
locate holder B and check whether SB12 is blown,
replacing it if so. [. . .] If SB12 is intact, check the
gearbox relays, the wiring harnesses to the con-
trol unit, and the control unit itself, consulting the
“Relays on fuse holder BSB” section for relay loca-
tions.
Judge (correct).“The generated answer cor-
rectly identifies Fuse 12 (SB12) related to the gear-
box control unit and provides troubleshooting steps
that align with the expected answer.”(Top-10 fused
pages spanning 8 documents.)The reader recovers
the holder-B gearbox fuse SB12 but not the second
relevant fuse SC6 on holder C; the judge accepts
the partially complete answer, consistent with the
partial-credit protocol in Section 5.3.
A.4 Prompts
AutoQA QA generation.GPT-4o receives sam-
pled pages from both documents and the anchor
photo, and is instructed to produce QA pairs. Core
instructions below (abridged):
You are generating a cross-document
multimodal QA dataset for automotive
workshop manuals. You are given sampled
pages from Document A and Document B and
one real-world mechanic photo.
Requirements:
1. Each question is from a mechanic
referencing the photo.
2. Each question must mention at least
one visible detail from the photo.
3. Each pair must require both documents,
combining distinct information from
each.
4. The answer must fully resolve
the question with document-specific
details.
5. Rely on document-specific details
only; do not add generic advice.
6. Use a mix of factual, procedural, and
diagnostic questions.
7. Do not mention document names, page
numbers, or “Document A/B”.

8. State facts and procedures directly,
as an automotive expert would.
Output a JSON array of pairs, each with
question, answer, question_type, and
reference_docs.
Enhanced query (AutoQA).The VLM reader
is prompted with the question and anchor image to
produce the enhanced query; prompts for the other
benchmarks differ only in domain framing.
You are given an automotive figure
and a question about it. Your task
is to generate a grounded natural
language question that integrates both
the textual question and the visual
information from the figure.
Instructions:
- Carefully analyze the question and the
figure simultaneously.
- Identify what parts of the figure are
relevant to the question.
- Produce a single, well-formed natural
language question that is explicitly
grounded in the visual content.
- Preserve the original intent of the
question.
- The output question should be more
specific and informative than the
original question alone.
- Do NOT answer the question. Do NOT
explain. Do NOT invent details.
Question: {question}
Grounded question:
Answer judging.At evaluation, GPT-4o-mini
receives the question, the ground-truth answer, and
the prediction, and returns a binary score.
You are an expert evaluator assessing
answers from a multimodal document QA
system. Evaluate whether the generated
answer correctly responds to the
question based on the expected answer.
Question: {question}
Expected Answer: {answer}
Generated Answer: {prediction}
Accuracy (0 or 1): Does the generated
answer match the factual content of the
expected answer?
- 1: Factually correct and aligns with
the expected answer.
- 0: Factually incorrect or contradicts
the expected answer.
Instructions:
- Focus on factual correctness, not
writing style or format.
- Partial matches are accurate if the
correct information is present.
- For numerical answers, check if values
match or are equivalent.
- For list answers, check if all key
elements are present.
- If the expected answer is “Not
answerable” and the generated answer
indicates inability to answer, consider
it accurate.A.5 Index Footprint
For AutoQA, the unified text-and-page-image mul-
tivector collection stores 7,682 page-level points
and occupies 4.2 GB on disk (3.9 GiB), includ-
ing token-level vectors and payload. Each page is
indexed as a single point whose multivector repre-
sentation is computed in one embedding pass with
jina-embeddings-v4.