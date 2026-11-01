# Reading Beyond OCR Noise: Robust Cross-Lingual RAG for Legislative Archives

**Authors**: Radha Krishna Deshpande, Ajoy Mondal, Akash Manna, C V Jawahar

**Published**: 2026-10-26 11:24:16

**PDF URL**: [https://arxiv.org/pdf/2610.35240v1](https://arxiv.org/pdf/2610.35240v1)

## Abstract
India's historical public archives present severe computer vision and document image analysis challenges: decades of parliamentary records survive only as degraded, multi-column scanned pages featuring mixed scripts, complex layouts, and dense visual noise where OCR failure is the primary obstacle to information retrieval. We present an end-to-end framework that couples vision-based layout segmentation and multi-script OCR with an agentic retrieval-augmented generation (RAG) pipeline. Our approach specifically addresses how visual degradation and layout fragmentation propagate into text extraction errors, using an iterative agentic loop to navigate around OCR noise and layout ambiguities before generating answers. Evaluated on the 200,000-page Punjab Vidhan Sabha archive, our system demonstrates robust handling of historical document degradation, outperforming standard text-only baselines.

## Full Text


<!-- PDF content starts -->

Reading Beyond OCR Noise: Robust
Cross-Lingual RAG for Legislative Archives
Radha Krishna Deshpande, Ajoy Mondal, Akash Manna, and C V Jawahar
CVIT, International Institute of Information Technology, Hyderabad, India
(radhakrishna.d,akash.manna)@research.iiit.ac.in ,
(ajoy.mondal,jawahar)@iiit.ac.in
Abstract. India’s historical public archives present severe computer vi-
sion and document image analysis challenges: decades of parliamentary
records survive only as degraded, multi-column scanned pages featur-
ing mixed scripts, complex layouts, and dense visual noise where OCR
failure is the primary obstacle to information retrieval. We present an
end-to-end framework that couples vision-based layout segmentation and
multi-script OCR with an agentic retrieval-augmented generation (RAG)
pipeline. Our approach speciﬁcally addresses how visual degradation and
layout fragmentation propagate into text extraction errors, using an iter-
ative agentic loop to navigate around OCR noise and layout ambiguities
before generating answers. Evaluated on the 200,000-page Punjab Vid-
han Sabha archive, our system demonstrates robust handling of historical
document degradation, outperforming standard text-only baselines. The
code is publicly available at GitHub Repo .
Keywords: Document Retrieval, Cross-lingual, Retrieval Augmented
Generation, Large Language Models
1Introduction
Historical legislative archives are diﬃcult to process because the main challenge
is often the document image itself. Faded text, irregular layouts, marginal notes,
and mixed scripts can introduce errors during layout analysis and OCR, which
then propagate to retrieval and question answering [ 21,22,29]. This is particularly
challenging for large archives containing Hindi, Punjabi, and English documents
with varying print quality and complex layouts.
Rather than treating OCR as a separate preprocessing step, we study the
complete pipeline from document images to retrieval and answer generation.
Our framework combines layout-aware OCR with an agentic retrieval loop that
can check the retrieved evidence, reformulate the query, and search again when
the available context is insuﬃcient. This helps keep generated answers grounded
in the original document evidence.
Retrieval-augmented generation (RAG) was introduced to combine language
models with external knowledge sources [ 13] and has since been extended to
iterative and agent-based retrieval [ 8,20,24]. Recent document retrieval meth-
ods have also moved toward vision-based representations, reducing reliance on
arXiv:2610.35240v1  [cs.CV]  26 Oct 2026

2 R K Deshpande et al.
OCR for document retrieval [ 7,18]. Multi-page document understanding fur-
ther shows the importance of preserving visual information during retrieval and
reasoning [ 6]. At the same time, OCR benchmarks have shown that multimodal
language models still face diﬃculties with text recognition, particularly for multi-
lingual and visually challenging documents [ 17], while recent work demonstrates
that OCR errors can signiﬁcantly aﬀect downstream question answering [ 21].
Language-model-based OCR correction has therefore become an active research
direction, especially for historical documents [ 3,25].
These developments motivate an end-to-end approach for historical legisla-
tive archives in which visual processing, retrieval, and answer generation are
evaluated together. We focus on the Punjab Vidhan Sabha proceedings, a large
multilingual archive spanning 1950–2024. Our system uses open-weight models
throughout the pipeline so that the complete workﬂow can be deployed locally
and the retrieved evidence can be traced back to the source pages.
In this paper, we contribute:
–an agentic RAG framework that combines layout-aware OCR, hybrid re-
trieval, and iterative evidence checking for a multilingual legislative archive;
–an evaluation of cross-lingual question answering on a large-scale historical
document collection;
–an analysis of how OCR and layout errors aﬀect retrieval and answer quality,
together with an ablation of the agentic retrieval loop.
2Related Work
Work on multilingual archive navigation sits at the intersection of several distinct
lines of research: retrieval-augmented generation, agentic search, multilingual
text embeddings, and multimodal document retrieval. None of these paradigms
were originally optimized for a corpus precisely like ours, making it necessary to
examine their contributions and limitations regarding noisy, multi-script legisla-
tive archives.
2.1 Agentic and Multilingual Retrieval-Augmented Generation
Retrieval-augmented generation originated from the premise that language mod-
els perform better when granted access to external knowledge rather than de-
pending solely on parametric memory [ 13]. Gao et al. categorize RAG frame-
works into naive, advanced, and modular designs, emphasizing that iterative
query reformulation yields greater accuracy gains than merely scaling retriever
capacity [ 8].
Agentic RAG advances this by treating retrieval as an adaptive, multi-step
agent loop rather than a static query-response pipeline [ 20,24]. Architectures
that dynamically navigate between keyword indexing, dense semantic search,
and document-level browsing oﬀer crucial ﬂexibility for decades-old legislative
records [ 1]. While agentic orchestration has been successfully demonstrated on
historical collections like the CIA’s declassiﬁed archives, those implementations
typically remained monolingual without addressing script-switching constraints
[26].

Robust Cross-Lingual RAG for Legislative Archives 3
Cross-lingual and multilingual RAG present distinct structural hurdles. Chirkova
et al. demonstrated that maintaining output consistency requires robust prompt
design rather than auxiliary translation steps [ 5], while Liu et al. observed
performance degradation in cross-lingual retrieval tasks [ 15]. Conversely, Li et
al. highlighted that incorporating multilingual evidence can stabilize reasoning
in ambiguous contexts [ 14]. Inﬂuenced by these ﬁndings, our system indexes
Hindi, Punjabi, and English sources concurrently. Foundation models such as
Multilingual-E5-Large provide strong cross-lingual mappings, serving as an im-
portant precedent for our multi-script workﬂow [ 23,28].
2.2 OCR and Historical Multilingual Document Processing
Historical document retrieval requires careful accounting for OCR degradation.
Zhang et al. quantiﬁed how noise introduced during text extraction degrades
downstream RAG performance [ 29], and Piryani et al. showed that these errors
compound signiﬁcantly in non-Latin and multi-script environments like Punjabi
Gurmukhi and Hindi Devanagari [ 21].
Recent post-correction methodologies utilize LLMs to clean historical text
and generalize better than traditional sequence models [ 25]. Bourne demon-
strated that injecting domain-speciﬁc historical context further improves entity
recognition for specialized legislative terminology [ 3]. Romein et al. conﬁrm that
structurally complex, multi-script documents remain a primary challenge in AI-
assisted archive digitization [ 22].
2.3 Multimodal and Vision-Language Document Retrieval
Traditional text-only pipelines often discard critical page layout structures such
as tables, headers, and column boundaries, motivating a shift toward multimodal
document retrieval. Ma et al. review this paradigm shift toward vision-language
document encodings [ 19]. Frameworks such as ColPali [ 7], DSE [ 18], and ad-
vanced visual embedding architectures like NetraEmbed [ 12] encode document
images directly into dense vector spaces, preserving layout semantics and layout
structures across multiple languages. Cho et al. extended this to multi-page re-
trieval frameworks [ 6], while Liu et al. emphasize the integration of sparse lexical
search with dense visual embeddings as an optimal trajectory for complex docu-
ment repositories [ 16]. Our optimized pipeline leverages these insights by pairing
robust multilingual textual dense representations with an adaptive agentic loop
tailored for historical legislative document search.
3Methodology
3.1 Query Management and Multi-Lingual Processing
The system initiates by receiving a user query, denoted as q, in its native script
and language. Rather than relying on intermediate machine translation, our ar-
chitecture processes the query natively against a multilingual semantic vector
space, preserving domain-speciﬁc legislative terminology and regional context
across Hindi, Punjabi, and English. Fig. 1shows the pipeline: the query is em-
bedded natively, the agent loop checks context suﬃciency and rewrites the query
if needed, and the generator produces the grounded answer.

4 R K Deshpande et al.
Fig. 1. Architecture of the proposed Agentic Retrieval-Augmented Generation (RAG)
system. The framework features multi-lingual query processing, an iterative LLM-
driven evaluation loop to ensure context suﬃciency, and a grounded response gen-
eration pipeline.
3.2 Oﬄine Processing and Real-Time Retrieval
The foundation of the retrieval system relies on a corpus of documents, D=
{d1, d2, . . . , d n}. During the oﬄine phase, an embedding model maps these doc-
uments into a dense vector space alongside their associated metadata. Let E(·)
represent the multilingual embedding function. The resulting document embed-
dings are stored systematically within a Vector Database.
For real-time retrieval, the native query qis embedded directly to produce
a query vector vq=E(q). The system queries the Vector Database to retrieve
the top- kmost relevant context chunks, C={c1, c2, . . . , c k}, by calculating the
similarity score between vqand the document vectors vdi.
sim(q, di) =v⊤
qvdi
∥vq∥2∥vdi∥2. (1)
3.3 Iterative Agent Loop for Context Evaluation
A deﬁning feature of this architecture is the autonomous Agent Loop, designed to
mitigate hallucinations and ensure the retrieved context is adequately informa-
tive despite historical OCR degradation. Once the initial context Cis retrieved,

Robust Cross-Lingual RAG for Legislative Archives 5
a dedicated evaluation LLM analyzes its suﬃciency to answer qbased on an
information gain threshold function:
G(C, q) =|q|∑
j=1I(wj∈Tokens (C)) (2)
–If the context score satisﬁes the suﬃciency threshold, the workﬂow proceeds
directly to the generation phase.
–If the context is deemed insuﬃcient, the agent triggers an adaptive query
rewriting mechanism to generate an optimized query variant q′. The opti-
mization objective minimizes semantic ambiguity:
q′= arg max
˜qSim(E(˜q), E(q))−λLOCR_noise (˜q) (3)
Here, λbalances semantic similarity to qagainst LOCR_noise (˜q), a conceptual
penalty on query terms likely to be missed due to OCR errors in the indexed
pages, which in practice is approximated by prompting the rewriting LLM
to preserve the intent of qwhile rephrasing terms that failed to retrieve
evidence. The system then loops back to the retrieval step using this reﬁned
query to fetch a more robust contextual foundation.
3.4 Grounded Response Generation
Upon validating context suﬃciency, the generation LLM synthesizes a grounded
response directly in the user’s native language. This generation is strictly condi-
tioned on both the native query and the validated context, maximizing factual
grounding without translation intermediaries. The probability of generating the
response sequence of tokens wcan be modeled as:
P(R|q, C) =T∏
t=1P(wt|w<t, q, C) (4)
3.5 Algorithm: Native-Language Agentic RAG Loop
Algorithm 1 Native-Language Multi-Step Agentic Retrieval
Require: User query q, Vector DB collection V, Maximum loops L
Ensure: Grounded response Rin native language
1:current _query←q
2:forstep = 1toLdo
3: C←V .query (current _query, k = 5)
4: is_sufficient, eval _log←LLM_Evaluate (current _query, C [0])
5: ifis_sufficient orstep == Lthen
6: break
7: end if
8: current _query←LLM_Rewrite (current _query, eval _log)
9:end for
10:R←LLM_Generate (q, C )
11:return R

6 R K Deshpande et al.
4Experiments and Results
4.1 Dataset
The dataset used in this research consists of Punjab Vidhan Sabha (legislative
assembly of Punjab state) documents collected from archival records spanning
from the 1950s to 2024. The collection includes assembly proceedings, debates,
reports, notices, and other oﬃcial legislative documents published across diﬀer-
ent time periods.
The records were digitized from bound volumes held by [ https://pvs.neva.
gov.in/ ], a process that was carried out incrementally over several decades
rather than in a single digitization eﬀort. This matters for the corpus’s charac-
teristics: pages from the 1950s and 1960s were typically typeset on older printing
equipment and later scanned at lower resolution, often with visible ink bleed-
through, faded typeface, and skewed page boundaries, while more recent decades
(post-2020) are comparatively cleaner but still exhibit dense multi-column lay-
outs and inconsistent formatting across sessions.
Documents in the corpus are predominantly in Hindi (Devanagari) and Pun-
jabi (Gurmukhi), with English appearing throughout, particularly in earlier-era
records and administrative headers; it is common for a single page to mix two
or all three languages, for instance an English-language question header fol-
lowed by a Punjabi response, or bilingual tabular data. The collection totals ap-
proximately 200,000 pages a covering assembly sessions, starred and unstarred
questions, budget and ﬁnancial statements, committee reports, and procedural
notices. Fig. 2shows representative pages illustrating the multi-column layouts,
mixed scripts, and visual degradation typical of the corpus.
Given the scale of the archive, we did not attempt to hand-annotate the full
collection. Instead, we constructed a stratiﬁed evaluation benchmark of 1,000
pages and 200 gold-annotated question-answer pairs, sampled to balance repre-
sentation across decade, script, and document type, so that evaluation results
reﬂect the corpus’s actual heterogeneity rather than only its cleaner, more re-
cent portion. To isolate the contribution of the vision pipeline speciﬁcally, we
additionally hand-transcribed a smaller subset of 20 pages, selected to span a
range of degradation levels, as ground truth for OCR quality evaluation.
4.2 Experimental Setup
We ran our experiments on a single Nvidia RTX-4060 (16 GB VRAM). Our eval-
uation framework implemented a robust local storage backend using ChromaDB
alongside an adaptive agentic retrieval architecture driven by BAAI/bge-m3 for
dense and hybrid token-level multilingual embeddings (supporting English, Pun-
jabi Gurmukhi, and Hindi Devanagari).
As described in Section 3.5, the retrieval index spans the full 200,000-page
Punjab Vidhan Sabha archive. All quantitative results in this section are com-
puted against our stratiﬁed evaluation benchmark of 1,000 pages and 200 gold-
annotated queries, drawn from this corpus and balanced across decade, script,
and document type. Additionally, to isolate vision-side performance, we hand-

Robust Cross-Lingual RAG for Legislative Archives 7
Fig. 2. Overview of dataset pages featuring tabular data, multi-scripts, and faded text
characteristics.
transcribed a subset of 20 pages and evaluated multiple layout segmentation and
OCR model combinations against this ground truth.
4.3 Metrics
We evaluate the proposed system at three levels: OCR recognition, document
retrieval, and answer quality.
OCR Recognition: We use Character Recognition Rate (CRR) and Word
Recognition Rate (WRR) to evaluate OCR performance. Both measures are
derived from the Levenshtein edit distance, which counts substitutions, deletions,
and insertions required to transform the predicted sequence into the reference
sequence [ 9,10].
The Character Recognition Rate is deﬁned as
CRR = 1 −Sc+Dc+Ic
Nc(5)
=Nc−Sc−Dc−Ic
Nc, (6)
where Sc,Dc, and Icdenote the number of character substitutions, character
deletions, and character insertions, respectively, and Ncdenotes the number of
characters in the reference text.
Similarly, the Word Recognition Rate is deﬁned as
WRR = 1 −Sw+Dw+Iw
Nw(7)
=Nw−Sw−Dw−Iw
Nw, (8)

8 R K Deshpande et al.
where Sw,Dw, and Iwdenote the number of word substitutions, word deletions,
and word insertions, respectively, and Nwdenotes the number of words in the
reference text [ 9,10].
Retrieval: We evaluate retrieval eﬀectiveness using Mean Reciprocal Rank
(MRR) and Hit Rate at k(Hit@k). MRR measures the reciprocal rank of the
highest-ranked relevant document for each query and averages this value over
all queries [ 27]:
MRR@1 =1
|Q||Q|∑
i=11
rank i. (9)
Hit@k measures the proportion of queries for which at least one relevant
document is retrieved within the top- kranked results:
Hit@k =1
|Q||Q|∑
i=1I(rank i≤k). (10)
Here, Qdenotes the set of evaluation queries, |Q|denotes the number of
queries, rank idenotes the rank of the ﬁrst relevant document for query i, and
I(·)is the indicator function that returns 1when the condition is satisﬁed and
0otherwise.
Answer Quality: We use the F1 score, which combines Precision and Recall
through their harmonic mean:
F1 =2PR
P+R, (11)
where Pdenotes Precision and Rdenotes Recall. Precision measures the pro-
portion of predicted answer tokens that are correct, while Recall measures the
proportion of reference answer tokens that are correctly identiﬁed. The F1 score
therefore provides a balanced measure of answer quality [ 4].
4.4 Retrieval Performance Comparison
Our primary goal was to evaluate and compare the retrieval performance of var-
ious multilingual embedding models and our proposed optimization framework
on the domain-speciﬁc Punjab Vidhan Sabha (PVS) benchmark.
As shown in Table 1, token-level and smaller cross-lingual models like MuRIL
and LaBSE struggle heavily with the complex, multi-script historical docu-
ments, yielding near-zero or low MRR@1 scores. Larger dense retrievers such
as Multilingual-E5-Large and NetraEmbed scale better, achieving Hit@3 scores
above 0.81. However, BGE-M3 and our optimized pipeline establish the strongest
baseline and performance standard, successfully navigating historical OCR degra-
dation and mixed-script text to secure top-tier retrieval accuracy.
Fig.3shows the full pipeline from query to grounded answer , and Fig. 4
compares the pages retrieved by each model for one Punjabi query.

Robust Cross-Lingual RAG for Legislative Archives 9
Table 1. PVS Retrieval Evaluation on the 1,000-page / 200-query stratiﬁed benchmark
(indexed against the full 200,000-page archive).
Model & Pipeline Setup Eval. Benchmark MRR@1 Hit@3 Hit@5 Prec@3
MuRIL 1,000 Pages / 200 Queries 0.000 0.005 0.015 0.002
LaBSE 1,000 Pages / 200 Queries 0.320 0.510 0.590 0.170
Embedding-gemma (300M) 1,000 Pages / 200 Queries 0.525 0.735 0.775 0.245
NetraEmbed (dim=768) 1,000 Pages / 200 Queries 0.580 0.816 0.890 0.272
NetraEmbed (dim=1536) 1,000 Pages / 200 Queries 0.598 0.835 0.905 0.278
NetraEmbed (dim=2560) 1,000 Pages / 200 Queries 0.605 0.843 0.912 0.281
Multilingual-E5-Large 1,000 Pages / 200 Queries 0.620 0.855 0.915 0.285
Standard BGE-M3 1,000 Pages / 200 Queries 0.775 0.900 0.940 0.300
Ours (Optimized Pipeline) 1,000 Pages / 200 Queries 0.810 0.925 0.955 0.325
Fig. 3. End-to-end cross-lingual RAG pipeline for legislative documents, from the user
query and document retrieval to grounded answer generation in Punjabi and translated
to english for the reference.

10 R K Deshpande et al.
Fig. 4. Qualitative comparison of cross-lingual embedding models for historical leg-
islative document retrieval. The same Punjabi query is given to multilingual-e5-large,
BGE-M3, LaBSE, Netra embedding (2560 dimensions), and EmbeddingGemma-300M.
The retrieved document from each model is compared with the ground-truth relevant
document shown in the rightmost column. The retrieved results demonstrate diﬀer-
ences in cross-lingual semantic matching, with several models returning visually and
semantically unrelated legislative records. In contrast, our proposed system retrieves
the relevant document containing the required information, demonstrating its eﬀec-
tiveness for retrieval from multilingual historical archives.

Robust Cross-Lingual RAG for Legislative Archives 11
5Ablation
5.1 OCR and Layout Pipeline Evaluation
Our production pipeline combines a ﬁne-tuned YOLO [11] layout model with
our CRNN-based OCR recognizer. We compare it with a CRAFT -based layout
model [ 2] using the same CRNN recognizer, along with Google OCR and Surya
OCR.
Table 2. OCR pipeline comparison on the 20-page ground-truth subset.
Layout Model OCR Model CRR (%) WRR (%)
Gemini-3.1 Gemini-3.1 99.7 89.6
Surya2 Surya2 99.6 88.9
Sarvam Sarvam 99.5 89.4
YOLO (Fine-tuned) CRNN (Ours) 95.67 80.03
CRAFT CRNN (Ours) 89.00 60.35
The ﬁne-tuned YOLO + CRNN pipeline achieves 95.67% CRR and 80.03%
WRR, compared with 89.00% CRR and 60.35% WRR for CRAFT + CRNN.
Since the same CRNN recognizer is used in both pipelines, the improvement
shows the eﬀect of the layout model on the quality of the extracted text. Google
OCR and Surya OCR achieve higher recognition scores, while our pipeline re-
mains fully local and provides greater control for processing the large archive.
Table 3. PVS Retrieval Evaluation using diﬀerent OCR models on the 200-query
benchmark.
OCR Model MRR@1 Hit@3 Hit@5 Prec@3
Gemini-3.1 0.825 0.920 0.960 0.3533
Sarvam 0.820 0.935 0.955 0.325
Surya2 0.815 0.925 0.950 0.305
Ourocr 0.810 0.925 0.955 0.325
5.2 Ablation on Agent Loop Iterations
To quantify the contribution of the iterative agentic loop, we vary the maximum
number of retrieval–rewrite iterations using a ﬁxed 200-query evaluation dataset,
L∈ {1,2,3}, and measure retrieval quality (MRR@1, Precision@3) alongside
end-to-end answer correctness (Answer F1-Score) on the same 200-query bench-
mark. L= 1corresponds to a single-pass, non-agentic retrieval baseline with no
query rewriting.Fig. 5illustrates how iterative query rewriting corrects retrieval
errors. In the ﬁrst two iterations, the system retrieves irrelevant documents,

12 R K Deshpande et al.
but after further query reﬁnement, the third iteration successfully retrieves the
correct document containing the required information.
As shown in Fig. 6, increasing Lfrom 1 to 3 improves MRR@1 from ap-
proximately 0.89 to 0.97 and Precision@3 from approximately 0.33 to 0.37, but
the eﬀect is most pronounced on Answer F1-Score, which rises from approxi-
mately 0.18 to 0.35 — nearly a 2×relative gain. This asymmetry indicates that
the agentic loop’s primary beneﬁt is not marginal re-ranking of already-retrieved
passages, but recovering answer-relevant context that single-pass retrieval misses
entirely, consistent with our claim that iterative query reformulation is essential
for navigating OCR noise and layout ambiguity in this corpus.
To assess how well diﬀerent instruction-tuned language models comprehend
and adapt to the domain-speciﬁc historical text and OCR-induced artifacts
within our legislative archive corpus, we evaluate the average perplexity scores
across three prominent backbones: Mistral-7B, Phi-3, and Qwen2.5 as shown in
Fig. 7.
Fig. 5. Iterative retrieval correction through query rewriting. The ﬁrst two iterations
retrieve irrelevant same documents, while the third iteration retrieves the correct doc-
ument containing the relevant information. The example demonstrates a cross-lingual
setting, where the query is in Punjabi and the retrieved document is in English.

Robust Cross-Lingual RAG for Legislative Archives 13
Fig. 6. Performance metrics vs Iteration
loops of the proposed system across diﬀer-
ent agent loop iterations. The results show
how the system performance changes as the
number of iterations increases.
Fig. 7. Average Perplexity scores of
the language models on multi-script
documents. Qwen2.5 achieves the low-
est perplexity (4.46), followed by Phi-3
(4.99) and Mistral-7B (7.81).
5.3 Limitations
Despite the strong performance of our multilingual agentic RAG framework,
several limitations remain. Severe OCR errors in degraded pages can still aﬀect
dense retrieval and entity matching. The iterative agentic loop and LLM-based
suﬃciency checks also increase inference latency compared with standard re-
trieval. In addition, complex ﬁnancial tables and unstructured tabular data may
lose hierarchical structure during text-to-chunk conversion. Future work will fo-
cus on improving OCR robustness, reducing inference latency, and preserving
complex table structures.
6Conclusion
In this work, we introduced an Agentic-RAG framework designed to bridge the
gap between complex historical archives and user information needs. By blending
iterative query rewriting, adaptive document retrieval, and vision-language gen-
eration, our approach eﬀectively tackles the nuances of multilingual document
search.
Evaluated on the Punjab Vidhan Sabha debates dataseta challenging col-
lection spanning multiple decades in Punjabi, Hindi, and Englishthe system
demonstrated robust resilience against real-world hurdles such as historical vo-
cabulary shifts, degraded text, and severe OCR noise. Unlike traditional, rigid
RAG pipelines, the iterative loop implemented here acts as a self-correcting
mechanism, dynamically verifying context suﬃciency to minimize hallucinations
and ground ﬁnal answers directly in source material.
Ultimately, this study highlights that agentic workﬂows can signiﬁcantly en-
hance information discovery in multilingual public records. Beyond historical

14 R K Deshpande et al.
text exploration, the principles demonstrated here pave the way for more depend-
able, transparent, and accessible digital archives for researchers, policymakers,
and the public.
Future work will focus on multi modal retrieval, expanding the system to
more Indian languages and government archives, and supporting conversational
search with long-context reasoning. User studies with researchers and other
archive users will further evaluate its usability and reliability.
Acknowledgment : This work is supported by MeitY, Government of India,
through the NLTM-Bhashini project.We also sincerely thank Prof. Gurpreet
Singh Lehal( gslehal@gmail.com ) for his valuable support and assistance during
the project evaluation.
References
1.Ayanami, et al.: A-RAG: Scaling agentic retrieval-augmented generation via hier-
archical retrieval interfaces. arXiv preprint arXiv:2602.03442 (2025)
2.Baek, Y., Lee, B., Han, D., Yun, S., Lee, H.: Character region awareness for text
detection. In: Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition. pp. 9365–9374 (2019)
3.Bourne, J.: Clocr-c: Context leveraging ocr correction with pre-trained language
models (2025), https://arxiv.org/abs/2408.17428
4.Chinchor, N.: Muc-4 evaluation metrics. In: Proceedings of the Fourth Message
Understanding Conference (1992)
5.Chirkova, N., et al.: Retrieval-augmented generation in multilingual settings. arXiv
preprint arXiv:2407.01463 (2024)
6.Cho, H., et al.: M3DocRAG: Multi-modal retrieval is what you need for multi-page
multi-document understanding. arXiv preprint arXiv:2411.04952 (2024)
7.Faysse, M., et al.: ColPali: Eﬃcient document retrieval with vision language mod-
els. arXiv preprint arXiv:2407.01449 (2024)
8.Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M.,
Wang, H.: Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 (2024)
9.IIIT Hyderabad: Ihtr 2023: Indian handwritten text recognition. Indian Language
OCR Competition (2023), https://ilocr.iiit.ac.in/ihtr/task.html
10.IIIT Hyderabad: Icdar 2024 rdtag: Evaluation metrics. ICDAR 2024 Robust Docu-
ment Text and Graphics (2024), https://ilocr.iiit.ac.in/icdar_2024_rdtag/
task.html
11.Jocher, G., Chaurasia, A., Qiu, J.: Ultralytics yolo (2023), https://github.com/
ultralytics/ultralytics
12.Kolavi, A.S., Jain, V.: M3dr: Towards universal multilingual multimodal document
retrieval (2025), https://arxiv.org/abs/2512.03514
13.Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks. In: Advances in Neural Information
Processing Systems (NeurIPS). vol. 33, pp. 9459–9474 (2020)
14.Li, B., et al.: BordIRLines: A dataset for geopolitical border disputes in multilin-
gual rag. arXiv preprint arXiv:2410.01171 (2024)
15.Liu, W., Trenous, S., Ribeiro, L.F.R., Byrne, B., Hieber, F.: XRAG: Cross-lingual
retrieval-augmented generation. In: Findings of the Association for Computational
Linguistics: EMNLP 2025 (2025)

Robust Cross-Lingual RAG for Legislative Archives 15
16.Liu, Y., et al.: Roles of mllms in visually rich document retrieval for rag: A survey.
In: Proceedings of IJCNLP 2025 (Long Papers) (2025)
17.Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.C., Liu, C.L., Jin,
L., Bai, X.: Ocrbench: On the hidden mystery of ocr in large multimodal models.
Science China Information Sciences 67(12) (2024)
18.Ma, X., et al.: DSE: Document screenshot embeddings. arXiv preprint
arXiv:2406.11251 (2024)
19.Ma, X., et al.: Scaling beyond context: A survey of multimodal retrieval-augmented
generation for document understanding. arXiv preprint arXiv:2510.15253 (2025)
20.Mishra, S., Niroula, S., Yadav, U., Thakur, D., Gyawali, S., Gaire, S.: SoK: Agentic
retrieval-augmented generation (RAG): Taxonomy, architectures, evaluation, and
research directions. arXiv preprint arXiv:2603.07379 (2025)
21.Piryani, B., Mozafari, J., Jatowt, A.: MultiOCR-QA: A multilingual dataset for
qa on ocr-processed texts. In: Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval. pp. 2038–2048
(2025)
22.Romein, C.A., et al.: Transforming historical newspaper research and preservation
through AI: A global perspective. Publications 7(1), 10 (2025)
23.Sener, S.: Advanced retrieval augmented generation: Multilingual semantic re-
trieval across document types by ﬁne-tuning transformer based language models
and OCR integration. Engineering and Technology Journal 9(7), 4404–4411 (2024)
24.Singh, A., et al.: Agentic retrieval-augmented generation: A survey on agentic
RAG. arXiv preprint arXiv:2501.09136 (2025)
25.Thomas, A., Gaizauskas, R., Lu, H.: Leveraging llms for post-ocr correction of his-
torical newspapers. In: Proceedings of the Third Workshop on Language Technolo-
gies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024.
pp. 116–121. ELRA and ICCL (2024)
26.Černý, J., Avramov, K., Pendse, L.: A multi-stage agentic AI system for extracting
information from large digital archives: Case study on the czechoslovak year 1968
in CIA’s FOIA collection. The Electronic Library (2026). https://doi.org/10.
1108/EL-06-2025-0272
27.Voorhees, E.M.: The trec-8 question answering track report. In: Proceedings of the
Eighth Text REtrieval Conference (TREC-8) (1999)
28.Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., Wei, F.: Multilingual E5
text embeddings: A technical report. arXiv preprint arXiv:2402.05672 (2024)
29.Zhang, R., et al.: OHRBench: A benchmark for ocr-induced noise eﬀects on rag
pipelines. arXiv preprint (2024)