# Invoice Haystack: Benchmarking Document Retrieval and Visual Question Answering Under Strong Visual Homogeneity

**Authors**: Heethanjan Kanagalingam, Thenukan Pathmanathan, Mokeeshan Vathanakumar, Basim Azam, Sarah Monazam Erfani

**Published**: 2026-06-24 03:17:30

**PDF URL**: [https://arxiv.org/pdf/2606.25343v1](https://arxiv.org/pdf/2606.25343v1)

## Abstract
Vision Language Models have achieved near-human performance on single-document Visual Question Answering, yet their effectiveness degrades significantly when retrieving information from large collections of visually homogeneous documents. Existing multi-document benchmarks aggregate diverse document types, creating artificial separation in embedding space that does not reflect enterprise document repositories where thousands of records share identical visual templates. We identify this as embedding collapse and introduce Invoice Haystack, a benchmark with 1,500 anonymized invoice images paired with 200 discriminative question-answer pairs, specifically designed to stress-test retrieval under strong visual homogeneity. Invoice Haystack exhibits a mean pairwise cosine similarity of 0.73, compared to 0.38 (DocHaystack) and 0.31 (InfoHaystack) in existing benchmarks, posing a fundamentally more challenging retrieval problem. Addressing the identified challenge, we propose VL-RAG, a hybrid retrieval-augmented generation framework that jointly leverages text and visual embeddings to harness the complementary strengths of both modalities, followed by a VLM-based verification filter for precise document identification. VL-RAG achieves 60.0\% Recall@1 on Invoice Haystack-500, outperforming existing state-of-the-art method by up to an absolute 13.5 percentage points. It further improves retrieval considerably on DocHaystack-1000 (77.1\% vs.\ 75.2\%) and InfoHaystack-1000 (84.5\% vs.\ 80.0\%), establishing the proposed dual-stream fusion as a consistently superior retrieval strategy across both homogeneous and heterogeneous document collections.

## Full Text


<!-- PDF content starts -->

Invoice Haystack: Benchmarking Document
Retrieval and Visual Question Answering Under
Strong Visual Homogeneity
Heethanjan Kanagalingam1, Thenukan Pathmanathan2, Mokeeshan
Vathanakumar1, Basim Azam1, Sarah Monazam Erfani1, and Naveed
Akhtar1
1The University of Melbourne, Melbourne, Australia
2Lakehead University, Thunder Bay, Canada
heethanjanheetha@gmail.com
Abstract.Vision Language Models have achieved near-human perfor-
mance on single-document Visual Question Answering, yet their effec-
tiveness degrades significantly when retrieving information from large
collectionsofvisuallyhomogeneousdocuments.Existingmulti-document
benchmarks aggregate diverse document types, creating artificial sepa-
ration in embedding space that does not reflect enterprise document
repositories where thousands of records share identical visual templates.
We identify this as embedding collapse and introduce Invoice Haystack,
a benchmark with 1,500 anonymized invoice images paired with 200
discriminative question-answer pairs, specifically designed to stress-test
retrieval under strong visual homogeneity. Invoice Haystack exhibits a
mean pairwise cosine similarity of 0.73, compared to 0.38 (DocHaystack)
and 0.31 (InfoHaystack) in existing benchmarks, posing a fundamentally
more challenging retrieval problem. Addressing the identified challenge,
we propose VL-RAG, a hybrid retrieval-augmented generation frame-
work that jointly leverages text and visual embeddings to harness the
complementary strengths of both modalities, followed by a VLM-based
verification filter for precise document identification. VL-RAG achieves
60.0% Recall@1 on Invoice Haystack-500, outperforming existing state-
of-the-art method by up to an absolute 13.5 percentage points. It fur-
ther improves retrieval considerably on DocHaystack-1000 (77.1% vs.
75.2%) and InfoHaystack-1000 (84.5% vs. 80.0%), establishing the pro-
posed dual-stream fusion as a consistently superior retrieval strategy
across both homogeneous and heterogeneous document collections.
The Invoice Haystack benchmark dataset, code, and project details are
available at https://heethanjan.github.io/invoice-haystack/.
Keywords:Document Retrieval·RAG·Vision-Language Models·Fi-
nancial Document Understanding·Invoice HaystackarXiv:2606.25343v1  [cs.CV]  24 Jun 2026

2 Heethanjan et al.
Fig. 1:Comparison of data domains across visual retrieval benchmarks. Unlike gen-
eral visual question answering datasets like RetVQA [56], and WebVQA [11], which
rely on natural imagery, or Document Haystack [12], which aggregates diverse text
documents, our proposed Invoice Haystack (right) targets the specialized domain of
financial documentation. This focused scope challenges models to distinguish between
visually similar transactional forms, requiring a deeper understanding of dense tabular
structures and spatial layouts compared to general image or document collections.
1 Introduction
The rapid advancement of Vision Language Models (VLMs) has fundamentally
transformed the landscape of Document AI, catalyzing a paradigm shift from
traditional modular pipelines comprising Optical Character Recognition (OCR),
layout analysis, and rule-based information extraction [28,71,72] toward uni-
fied end-to-end architectures capable of simultaneous visual perception and se-
mantic understanding [36,39]. Contemporary state-of-the-art VLMs, including
ChatGPT-5.2 [54], Gemini 3.0 Pro [23], Claude 3.7 Sonnet [7], and Qwen3-
VL [9], have demonstrated unprecedented proficiency in single-document Visual
Question Answering (VQA), achieving near-human performance on established
benchmarks such as DocVQA [50], InfographicVQA [49], ChartQA [48], and
VisualMRC [64]. These models leverage transformer-based architectures with
cross-modal attention mechanisms [47,57,66] to jointly encode visual features
andtextualcontent,effectivelysolvingthesingle-documentcomprehensionprob-
lem for many practical applications [43,44]. However, these models still face
significant challenges when tasked with reasoning over extensive collections of
images or documents [69], limiting their deployment in real-world applications
such as financial auditing, legal discovery, and supply chain management due to
the lack of dedicated benchmarks for evaluating multi-document retrieval and
reasoning performance [32].
The existing multi-image retrieving and reasoning benchmarks are primar-
ily constructed on a small scale, as highlighted in works such as [20,34]. Each
question in these benchmarks is paired with only up to 30 images, a limited
scope that does not align with real-world scenarios requiring retrieval across
thousands of files. Recent efforts, such as the DocHaystack and InfoHaystack
benchmarks [12], have attempted to address this by scaling contexts up to 1,000
documents. However, while these benchmarks successfully address the challenges
of scale and ambiguity, they fail to account forvisual homogeneity. Datasets like
DocHaystack rely on visually diverse collections, ranging from bar charts to
handwritten letters, that possess distinct visual features. Consequently, in the

Invoice Haystack 3
high-dimensional embedding space, these data points map to distinct clusters,
artificially inflating retrieval performance because the negatives are easily dis-
tinguishable. This setup does not reflect operational enterprise realities, where
a repository of thousands of invoices may utilise only a dozen templates.
Torigorouslyaddressthisgap,weintroducetheInvoiceHaystackbenchmark.
We identify invoices as the ideal stress test, as they combine complex spatial
structures with sparse, high-value information that is visually repetitive but
semantically distinct. The benchmark is constructed through a rigorous four-
stage pipeline comprising anonymization of personally identifiable information
(PII), VLM-based question generation, automated LLM filtering, and expert
human validation and correction.
To enable VLMs to navigate this challenging environment, we propose VL-
RAG (Vision-Language RAG). We posit that in homogeneous haystacks, neither
visual features nor textual features alone are sufficient for differentiation. Vision
encoders collapse visually identical templates into indistinguishable representa-
tions, a phenomenon we term embedding collapse, while text-only approaches
discard valuable layout and structural signals [60]. VL-RAG fundamentally re-
structures the retrieval hierarchy by incorporating both vision and text encoders
to capture dense embeddings. By integrating these semantic signals alongside
visual features, VL-RAG enables precise and discriminative document retrieval
across visually homogeneous collections.
Our key contributions are:(1) Invoice Haystack Benchmark.We intro-
duce the Invoice Haystack benchmark of 1,500 anonymized invoice images paired
with 200 discriminative question-answer pairs, specifically engineered to stress-
test retrieval under extreme visual homogeneity. The benchmark is evaluated
across three corpus scales, 500, 1,000, and 1,500 documents, enabling controlled
measurement of how retrieval performance degrades as the size of the homoge-
neous pool increases. Invoice Haystack exhibits a mean pairwise cosine similarity
of 0.73, nearly double the 0.31–0.38 range of prior benchmarks, posing a qualita-
tively harder retrieval challenge than any existing multi-document dataset.(2)
VL-RAG Method.We propose a hybrid dual-stream approach that integrates
densetextembeddings(BGE-Large)alongsidestructuralvisualfeatures(SigLIP,
OpenCLIP) to address the embedding collapse that cripples vision-only retrieval
in homogeneous collections. Proposed VL-RAG achieves up to 13.5 percentage
pointsgainonInvoiceHaystackforRecall@1,whilemaintainingacross-the-board
superiority over existing state-of-the-art methods for Document Haystack and
InfoHaystackforRecall@1,3,5.Ourresultsconclusivelyestablishourdual-stream
fusion technique as the priority method.
2 Related Work
Document Understanding with Vision-Language Models.Document AI
has evolved from modular pipelines combining OCR [3,62], layout analysis [59,
77], and NLP models [18,45] toward unified end-to-end architectures. LayoutLM

4 Heethanjan et al.
[28,71,72] pioneered spatial-aware pre-training by integrating 2D position em-
beddings with text, while OCR-free models including Doughnut [36], Pix2Struct
[39], and Nougat [10] eliminated external OCR dependency through vision trans-
formers [19,46]. Recent multimodal architectures like mPLUG-DocOwl [75],
UniDoc [25], and UDOP [65] unified vision, text, and layout representations.
Contemporary VLMs including ChatGPT-5.2 [54], Gemini 3.1 Pro [24], Qwen3-
VL [9], and InternVL-3.5 [67] achieve over 90% accuracy on single-document
VQA benchmarks [48–50,64]. However, performance degrades substantially in
multi-document settings. Notably, reasoning performance has been reported to
degrade by up to 85% as input length scales, even when models successfully re-
trieve the correct evidence [21]. This stark decline, despite the advent of million-
token windows, indicates fundamental ongoing challenges in document-scale rea-
soning.
Multi-DocumentBenchmarks&HomogeneityGap.Earlymulti-document
benchmarks including MultimodalQA [63] and WebQA [11] limited contexts
to 10-30 images per question [2,56], insufficient for enterprise-scale evaluation.
DocHaystack and InfoHaystack [12] recently advanced the field by scaling to
1,000 documents through multi-stage filtering with LLM-based question genera-
tion [6,54] and human validation, ensuring answer uniqueness and visual ground-
ing. Nevertheless, this design still falls short on reflecting real-world enterprise
repositories, which are dominated by highly structured, heavily templated docu-
ments such as invoices, receipts, and standardised forms [33,71]. In these largely
homogenouscollections, documents from different vendors or transactions often
share near-identical visual layouts, leading to a collapse in visual embedding
space. Yet, existing benchmarks do not evaluate this critical setting, leaving a
fundamental gap in assessing retrieval performance reliably.
RetrievalMethods&Vision-onlyLimitations.Contemporaryvision-based
retrieval leverages contrastive models pre-trained on natural images, including
CLIP [57], SigLIP [76], and DINOv2 [55], combined with vision transform-
ers [19,46] and vision-language alignment modules [15,42]. ColPali [22] intro-
duced late interaction mechanisms for page-level retrieval, achieving state-of-
the-art performance on visually rich document benchmarks. Text-based dense
retrieval methods, including DPR [31] and ColBERT [35], excel on text-based
documents but remain limited when documents share strong visual and seman-
tic similarity at scale. Hybrid approaches like Flamingo [4] and RA-CM3 [74]
combine modalities but focus on few-shot learning over natural image-text pairs
rather than large-scale document retrieval.
Retrieval-Augmented Generation for Documents.Text-based RAG sys-
tems combine retrieval with generation: RAG [41] integrates DPR [31] and
BART [40] for open-domain QA, REALM [26] incorporates retrieval into pre-
training, Atlas [30] demonstrates strong few-shot performance through Fusion-
in-Decoder [29], achieving 42.4% exact match on NaturalQuestions [38] with
only 64 training examples (45.1% with a Wikipedia-only index), and recent ad-
vances including Self-RAG [8], and CRAG [73] introduce retrieval-on-demand
and corrective mechanisms. Multimodal RAG extensions remain limited: RA-

Invoice Haystack 5
Fig. 2:Invoice Haystack Generation Pipeline. Our dataset is curated through a com-
prehensive four-stage workflow. Step 1 (Invoice Image Anonymization) employs Gemini
3 Pro to sanitize raw invoice images, ensuring sensitive information is removed. Step
2 (Complex Q&A Generation) utilizes Gemini 2.5 Flash to analyze the anonymized
documents and synthesize a raw dataset of question-answer pairs. Step 3 (AI Fil-
tering) refines this output using GPT-5.2 to filter for quality and relevance. Finally,
Step 4 (Human Validation) subjects the filtered dataset to a three-pronged manual
review—verifying anonymization, question quality, and answer accuracy—to produce
the final Invoice Haystack benchmark.
CM3 [74] evaluates on-web images, REVEAL [27] uses Wikipedia images, and
MuRAG [13] requires manual modality selection [51,74]. V-RAG [12] addresses
visual document retrieval at scale through an ensemble of vision encoders (CLIP,
SigLIP, and OpenCLIP) combined with a VLM-based relevance filtering mod-
ule, achieving 9% and 11% improvement in Recall@1 over prior best baselines
on the DocHaystack-1000 and InfoHaystack-1000 benchmarks, respectively, and
enabling GPT-4o to improve by over 55% on DocHaystack-200. However, purely
vision-centric approaches remain limited when documents share strong visual
similarity, as evidenced by the substantially lower retrieval performance of in-
dividual CLIP-based models on document-heavy collections compared to text-
based BM25 retrieval using OCR [58], motivating the need for hybrid architec-
tures that jointly encode visual layout and textual content.
3 Proposed Invoice Haystack Benchmark
Our Invoice Haystack benchmark addresses a critical gap in document AI eval-
uation: assessing retrieval performance under strong visual homogeneity. Unlike
existing benchmarks that aggregate diverse document types [12,49], our bench-
marktargetstemplate-heavycollectionssuchasinvoices,wheredocumentsfollow
a similar layout structure while differing only in fine-grained semantic content.
We construct this benchmark through a four-stage pipeline combining multi-
modal AI models with rigorous human oversight, as illustrated in Figure 2.
Motivation.Invoices provide an ideal domain for evaluating homogeneous doc-
ument retrieval. They combine complex spatial layouts with multi-column ta-
bles [71,72], critical information distributed across distinct regions (headers,
line entries, footers) [16,50], and extreme template standardization. Enterprise
repositories exhibit 70-85% template reuse, with fewer than 15 vendor templates
accountingforthemajorityofdocuments[5,61].Twoinvoicesfromthesametem-
plate may exhibit cosine similarity exceeding 0.95 in vision-only embeddings [12]
yet contain completely different semantic content distinguishable only through

6 Heethanjan et al.
text analysis. This structure directly mirrors real-world scenarios where systems
must locate specific transactions among thousands of visually similar documents
for auditing, compliance, and dispute resolution [1,68].
3.1 Data Curation Pipeline
The benchmark is constructed through a four-stage workflow - see Figure 2. The
pipeline is designed to maximize data quality, preserve privacy, and ensure that
everyretainedquestion-answerpairisbothdiscriminativeandvisuallygrounded.
Stage 1: Invoice Image Anonymization.We employ Gemini 3 Pro [23]
to sanitize 2,000 raw invoice images via a structured multi-modal prompt tar-
geting seven sensitive information categories: company and customer identifiers,
physical addresses, contact details, corporate branding and logos, temporal iden-
tifiers such as invoice numbers and dates, personal identifiers such as account
managers, and financial account details. Critically, all numerical values are pre-
served in their correct relational form — substituted figures are synthetically
generated such that line totals, subtotals, tax computations, and final amounts
remainmathematicallyconsistentthroughout,ensuringthatanonymizationdoes
not corrupt the semantic integrity of the financial content. Layout geometry,
including table structures, column alignments, fonts, and visual separators, is
preserved exactly to prevent models from exploiting anonymization artifacts as
discriminative signals in their predictions.
Stage 2: Complex Q&A Generation.Gemini 2.5 Flash [17] analyzes each
anonymized image to generate one question-answer pair per document, yielding
2,000 raw candidate pairs. Question design enforces unique document identifi-
cation through specific visual entities, such as company names, customer identi-
fiers, product descriptions, or transaction dates, with multi-field questions (e.g.,
"What is the order no associated with the contract invoice transaction dated
15-OCT-2023 and building no 506890?") preventing non-unique answers across
template instances and ensuring retrieval requires precise semantic matching.
Generic questions (e.g.,"What is the invoice total?") are explicitly excluded.
Answers are constrained to short extractive spans (1–2 words preferred, 5-word
maximum) appearing verbatim in the image, with generation temperature set
to 0.7 to balance output diversity and format adherence.
Stage 3: AI Filtering.GPT-5.2 [54] performs automated quality filtering over
the 2,000 candidate pairs, removing questions that are generic, ambiguous, non-
discriminative, or answerable without image access. This reduces the dataset
to 200 high-quality pairs that satisfy uniqueness and visual dependency con-
straints. The filtering step acts as a scalable quality gate, efficiently eliminating
the majority of low-quality outputs before human review.
Stage 4: Human Validation.Expert annotators apply a three-pronged man-
ual review to all 200 filtered question-answer pairs and their associated images.
Reviewers verify anonymization completeness by scanning for residual PII, in-
cluding real company names, addresses, contact information, invoice numbers

Invoice Haystack 7
Fig. 3: (a)Distribution of question types in Invoice Haystack across eight semantic
categories.(b)Spatial positioning of answers within the documents.
enabling external correlation, and recognizable branding. Question quality is as-
sessed for discriminative power, clarity, and non-trivial reasoning requirements.
Answer accuracy is confirmed by direct visual inspection of source images, en-
suring that answers appear exactly as specified and no ambiguous alternatives
exist. Entries failing any criterion are corrected or changed. Additionally, the
full corpus of 2,000 anonymized images underwent manual inspection to remove
documents with anonymization failures, rendering artifacts, or quality issues,
reducing the final corpus to 1,500 high-quality invoice images. This produced
the final validated benchmark of 200 question-answer pairs evaluated against
1,500 documents. Details of the prompts utilized across these stages, along with
sample anonymized images and questions, are provided in the supplementary
material.
3.2 Benchmark Statistics and Analysis
The final Invoice Haystack benchmark comprises 200 validated question-answer
pairs evaluated against a corpus of 1,500 visually homogeneous invoice images.
Figure 3(a) presents the distribution of question types across the dataset. Ques-
tions span eight semantic categories targeting different invoice fields. This dis-
tribution reflects the diversity of information types present in real invoice docu-
ments and ensures comprehensive evaluation across different retrieval scenarios.
The deliberate exclusion of yes/no questions and generic quantity queries en-
forces that every question demands specific document-grounded reasoning. Fig-
ure 3(b) shows the spatial distribution of ground-truth answer locations across
all invoice documents. Each point denotes the centre of the OCR bounding box
matched to a ground-truth answer string, with all positions normalised to an
A4 coordinate system (2,480 x 3,508 pixels). The clustering patterns reveal con-
sistent yet varied spatial placement across diverse invoice layouts. This spatial
structure confirms that answers are not trivially co-located and that retrieval
models must reason over the full document rather than attending to a single
fixed region. Table 1 compares Invoice Haystack with the most closely related

8 Heethanjan et al.
Fig. 4:Proposed VL-RAG architecture. The approach operates via two parallel
streams. The Vision Stream (top) employs SigLIP and OpenCLIP to capture layout
and structural features, while the Text Stream (bottom) uses OCR followed by BGE-
Large to encode semantic content. Similarity scores for all three encoders are averaged
(Sim Avg) to rank the corpus, and the top-mcandidates undergo VLM-based binary
verification to produce the final top-koutput.
Table 1:Comparison of Invoice Haystack with existing document retrieval bench-
marks. Visual homogeneity is measured by mean within-category SigLIP cosine simi-
larity. The proposed benchmark is high-potential and difficult and stronger in homo-
geneity.
Benchmark Corpus Size Doc Type Homogeneity
DocHaystack [12] 1,000 Mixed 0.38
InfoHaystack [12] 1,000 Mixed 0.31
Invoice Haystack (Ours) 1500 Invoices 0.73
existing document retrieval benchmarks. Invoice Haystack exhibits a mean pair-
wise cosine similarity of 0.73, more than double the 0.31–0.38 range of prior
benchmarks, reflecting a fundamentally harder retrieval setting in which docu-
ments cannot be easily distinguished by visual features alone.
4 Methodology: Proposed VL-RAG Approach
The central hypothesis of this work is that in visually homogeneous document
collections, vision-only and text-only embeddings are fundamentally insufficient
for precise retrieval. Standard vision encoders trained on natural images tend to
collapse template-based documents into indistinguishable representations, fail-
ing to resolve fine-grained semantic differences that exist only in textual con-
tent [12,78]. To address this, we propose VL-RAG (Vision-Language Retrieval-
Augmented Generation), a hybrid dual-stream framework that combines dense
text embeddings with structural visual features. As illustrated in Figure 4,
the proposed method operates through three core components: a text encod-
ing stream that captures semantic content via OCR and dense embeddings, a
vision stream that encodes layout and structural features, and a VLM-based ver-

Invoice Haystack 9
ification filter that refines the final retrieval output. Together, these components
enable precise document identification in homogeneous collections where visual
similarity alone is insufficient.
Text Stream.The text stream is designed to capture the high-precision se-
mantic information that vision models systematically fail to capture. An OCR
engine first extracts raw text from each invoice image, converting the visual
document into a structured textual representation. This extracted text is then
encoded using a dense text encoder, and the resulting embedding captures the
precise semantic content of each document. Documents encoded in this manner
are well-separated in text embedding space, provided the discriminative signal is
absent in vision-only retrieval. The text stream exploits this separability, com-
puting a similarity score Sim Tfor each query-document pair:
SimT(q, di) =ϕT(ˆq)·ϕ T(ti)
∥ϕT(ˆq)∥∥ϕ T(ti)∥, (1)
whereϕ T(·)denotes the BGE-Large encoder,t iis the OCR-extracted text
of documentd i, andqrepresents the input query (withˆqbeing its textual rep-
resentation).
Vision Stream.The vision stream captures structural and layout-level features
that complement the text signal. Following V-RAG [12], we adopt SigLIP [76]
and OpenCLIP [14] as the vision encoders, ensembling their outputs to form the
visual component of VL-RAG. This choice is further validated by our ablation
study (Table 4), which shows that the SigLIP + OpenCLIP combination consis-
tently outperforms any single vision encoder. Together, these encoders compute
similarity scores Sim Sand Sim Orespectively, as follows:
SimS(q, di) =ϕS(Iq)·ϕS(Ii)
∥ϕS(Iq)∥∥ϕ S(Ii)∥,Sim O(q, di) =ϕO(Iq)·ϕO(Ii)
∥ϕO(Iq)∥∥ϕ O(Ii)∥(2)
whereϕ S(·)andϕ O(·)denote the SigLIP and OpenCLIP encoders respectively,
andI iis the image of documentd i.
Score Fusion.As depicted in Figure 4, the three similarity scores from the text
stream (Sim T) and vision stream (Sim S, Sim O) are combined through average
fusion to produce a unified ranking score:
SimAvg=1
3(Sim T+Sim S+Sim O). (3)
The corpus is then sorted by Sim Avgin descending order. The top-mcandidate
set is formally defined as:
Cm= argtop
S⊆D,|S|=mSimAvg(q, di). (4)
This fusion ensures that a document is promoted only when it scores well across
both modalities, reducing false positives caused by either visual layout matches

10 Heethanjan et al.
that lack textual alignment or textual matches in documents with divergent
layouts. The top-mcandidate setC mis further filtered as below.
VLM Filter.Vector similarity scoring provides a high-recall shortlist but does
not guarantee that any retrieved document actually answers the query. To con-
vert this high-recall set into a high-precision result, VL-RAG applies a VLM
Filter stage over the top-mcandidates. Each candidate imageI iis paired with
the original queryqand passed to Qwen3-VL, which performs binary verification
via the prompt:“[question] Can this image provide the answer to this question?
Answer only yes or no.”. The model’s response is parsed, and only candidates
that return ayesresponse are retained:
Rk={d i∈ Cm|VLM(I i, q) =yes} (5)
If no candidate passes verification, the top-mshortlist is returned unfiltered as
a fallback to ensure that a result is always produced. This stage is essential in
homogeneous settings where multiple visually similar documents score highly
under vector similarity. The VLM provides the semantic reasoning capacity to
resolve such ambiguity that vector similarity alone cannot.
5 Experiments
Implementation Details.All of our experiments are conducted under a stan-
dardized retrieval-augmented generation pipeline to ensure a fair comparison.
For the vision stream, we use frozen pre-trained checkpoints of SigLIP [76]
ViT-SO400M/14@384 variant and OpenCLIP [14] ConvNeXt-XXL@1024 vari-
ant, evaluated in zero-shot mode without fine-tuning. For the text stream, we
employ BGE-Large-En-v1.5 [70] as the dense text encoder, applied over text
extracted via a DeepSeek OCR engine. The VLM filter and final answer gener-
ation both use Qwen3-VL-8B Instruct [9], configured with greedy decoding and
a maximum of 128 new tokens. For the VLM filter, we set the candidate pool
size tom= 10atk∈ {1,3,5}for all retrieval experiments (ablation for the
selection ofmis provided in the supplementary material) and report results.
All inference is executed on NVIDIA A100 (80GB) GPUs. For each query, the
haystack is constructed by combining the ground-truth document withN−1
distractors drawn from the respective benchmark split.
Baselines.We compare VL-RAG against a range of retrieval approaches span-
ning sparse text, dense vision, and vision-language methods. BM25 [58] serves
asthesparsetextbaselineappliedoverOCR-extractedcontent.Vision-onlybase-
linesincludeCLIP[57]ViT-L/14@336variant,SigLIP[76]ViT-SO400M/14@384
variant, OpenCLIP [14] ConvNeXt-XXL@1024 variant, Jina-CLIP-v2 [37], and
Nomic-Embed-Vision-v1.5[52].TheprimarymultimodalbaselineisV-RAG[12],
a recent vision-centric retrieval-augmented generation system. For a fair compar-
ison, we adapt V-RAG to use Qwen3-VL-8B-Instruct [9] as its VLM component,
matching the VLM configuration in VL-RAG. All models are evaluated under
identical conditions across three metrics: Recall@1, Recall@3, and Recall@5.

Invoice Haystack 11
Table 2:Retrieval performance (Recall@1 / Recall@3 / Recall@5) across Document
Haystack, InfoHaystack, and proposed Invoice Haystack benchmarks. Best results per
column are bold. Proposed VL-RAG achieves across-the-board superior performance.
Document Haystack InfoHaystack Invoice Haystack
100 200 1000 100 200 1000 500 1000 1500
Model R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5
BM25 (OCR) [58] 69.7 79.8 81.7 65.1 75.2 78.0 62.4 70.6 73.4 69.7 79.8 81.7 47.7 63.2 69.7 32.9 43.2 51.0 43.0 55.5 58.0 41.5 48.0 52.0 38.5 45.5 49.5
Jina-CLIP-v2 [37] 62.4 77.1 84.4 58.7 69.7 78.0 35.8 54.1 61.5 62.4 77.1 84.4 62.6 84.5 88.4 38.7 59.4 67.7 4.5 9.5 12.5 4.0 5.5 8.0 2.5 5.0 5.0
Nomic-Embed [52] 78.9 84.4 87.2 78.0 83.5 86.2 61.5 69.7 74.3 78.9 84.4 87.2 54.2 61.9 63.2 49.7 59.4 60.0 34.5 44.5 49.0 28.0 38.0 44.0 26.5 36.0 40.0
CLIP [57] 46.8 66.1 68.8 44.0 64.2 66.1 27.5 41.3 47.7 46.8 66.1 68.8 66.5 78.7 85.8 53.6 68.4 72.3 10.0 16.0 22.0 8.0 13.0 15.0 7.0 11.0 14.0
SigLIP [76] 57.8 66.1 72.5 55.1 60.6 64.2 41.3 51.4 54.1 57.8 66.1 72.5 65.8 83.2 89.0 45.8 62.6 69.7 28.0 40.5 46.0 24.0 33.0 39.5 20.0 30.0 37.0
OpenCLIP [14] 66.1 75.2 81.7 56.9 72.5 77.1 41.3 59.6 66.1 66.1 75.2 81.7 77.4 87.7 89.7 57.4 72.3 79.4 22.0 31.5 37.0 20.0 28.0 31.5 19.5 24.0 28.0
V-RAG [12] 88.1 89.9 89.9 83.5 87.2 87.2 75.2 80.7 80.7 92.3 95.5 95.5 88.4 92.9 92.9 80.0 83.2 85.2 46.5 49.0 49.0 43.0 45.5 46.0 40.0 43.0 43.0
VL-RAG (Ours) 89.9 91.7 91.7 86.2 88.1 88.1 77.1 81.7 81.7 94.2 96.8 96.8 94.2 96.8 96.8 84.5 87.7 89.0 60.0 63.5 63.5 53.0 57.0 57.5 50.0 53.5 54.0
5.1 Main Results
Table 2 reports Recall@1, Recall@3, and Recall@5 retrieval performance across
all three benchmarks. We organize findings by benchmark domain to highlight
the distinct behavior of each model class under visual homogeneity.
Invoice Haystack:This benchmark represents the hardest retrieval setting due
to extreme visual homogeneity. Vision-only models exhibit severe performance
collapse across all corpus scales. CLIP achieves only 7.0% Recall@1 on the 1500-
document split, and Jina-CLIP-v2 degrades further to 2.5%, performing near
chance. SigLIP and OpenCLIP fare marginally better at 20.0% and 19.5%, re-
spectively, yet remain far below practical utility. Even V-RAG, which aggregates
multiple vision encoders, reaches only 40.0% Recall@1 at the largest scale. This
degradation reflects embedding collapse: as visually identical templates prolifer-
ate, vision encoders cannot resolve instance-level semantic differences. Notably,
BM25 achieves 38.5% Recall@1 on the 1500-split, outperforming all vision-only
baselines and only 1.5% less than V-RAG itself, which directly demonstrates
that textual content is a primary discriminative signal in this domain. VL-RAG
achieves 50.0% Recall@1 on the hardest split, improving over V-RAG by 10 per-
centage points and confirming the necessity of explicit text encoding in visually
homogeneous retrieval.
Document Haystack.On this visually heterogeneous benchmark, all mod-
els perform substantially better, and the performance ordering shifts. V-RAG
achieves75.2%Recall@1atthe1000-documentscale,whileNomic-Embed-Vision
performs competitively at 61.5%. VL-RAG reaches77.1%Recall@1 at 1000
documents, improving upon V-RAG across all scales. The smaller absolute gain
relative to Invoice Haystack confirms that visual features are already sufficiently
discriminative in heterogeneous corpora, and the text stream provides a comple-
mentary rather than compensatory signal.
InfoHaystack.Results follow a similar trend. V-RAG achieves 80.0% Recall@1
atthe1000-documentscale,whileVL-RAGreaches84.5%,againof4.5percent-
agepoints.Themostpronouncedimprovementoccursatthe200-documentscale,
where VL-RAG surpasses V-RAG by 5.8 percentage points (94.2% vs 88.4%).
CLIP performs more competitively in this domain than on Invoice Haystack,
consistent with its natural image pre-training aligning better with visually di-
verse infographic content.

12 Heethanjan et al.
Visual Question Answering (VQA) results.VQA accuracy had been eval-
uated across three frameworks, which are zero-shot, V-RAG, and our proposed
VL-RAG. With each V-RAG and VL-RAG entry reflecting the best accuracy
achieved across Top-1/3/5 retrieval depths(Table 3). To evaluate these bench-
marks, we employ a model-based assessment (LLM as a Judge) by leveraging
GPT-4o-mini [53] following [12]. Zero-shot results are included only for smaller
corpus scales (100 and 200 documents for DocHaystack and InfoHaystack; 500
for Proposed Invoice Haystack for capable models), as the larger splits, and even
the 500-document Invoice split for InternVL3-8B and Qwen3-VL-8B, exceed the
practical context-window capacities. These zero-shot numbers serve as a mean-
ingful lower bound: even frontier models score only 18.0% (Gemini) and 33.0%
(GPT-5.2) on InvoiceVQA-500 without retrieval support, confirming that the
proposed Invoice Haystack poses a challenge that cannot be bypassed by simply
expanding context. Across all four VLMs and corpus scales, VL-RAG consis-
tently outperforms V-RAG with gains ranging from 2 to 16 percentage points.
The improvements are most pronounced on our Invoice Haystack benchmark,
where visual homogeneity is most severe: Gemini-3-Flash improves from 51.0%
to60.0%on InvoiceVQA-500 and from 44.0% to53.0%on InvoiceVQA-1500,
while GPT-5.2 rises from 49.5% to61.0%at the 500-document scale. Among
all models, Qwen3-VL-8B achieves the highest single accuracy of63.5%on
InvoiceVQA-500.
On the heterogeneous benchmarks, VL-RAG continues to improve over V-
RAG, though with smaller margins consistent with visual features already pro-
vidingreasonablediscriminabilityinthosesettings.Gemini-3-Flashreaches76.1%
on DocHaystack-1000 and73.5%on InfoHaystack-1000 under VL-RAG. As ex-
pected, accuracy declines monotonically as corpus scale increases from 500 to
1500 documents across all models; however, VL-RAG maintains a clear and
consistent advantage at every scale, demonstrating that incorporating textual
signals alongside visual retrieval is a robust and universally beneficial strategy
regardless of document homogeneity.
5.2 Ablation Study
To isolate the contribution of each architectural component, we conduct a sys-
tematic ablation on Document Haystack-1000, InfoHaystack-1000, and Invoice
Haystack-1500, reporting averages across all three with different encoder combi-
nations (Table 4).
Individual encoder performance.Vision-only encoders in isolation perform
poorly on Invoice Haystack. CLIP alone yields 7.0% Recall@1, SigLIP 20.0%,
and OpenCLIP 19.5%, confirming that no single vision encoder provides suffi-
cientdiscriminativepowerundervisualhomogeneity.Thetextembedding(BGE)
alone achieves 31.5% Recall@1 on Invoice Haystack-1500, with an average Re-
call@1of52.4%acrossallbenchmarks,substantiallyoutperformingallindividual
vision encoders in the average.
The No-CLIP finding.A surprising result emerges when examining CLIP’s
contribution to the ensemble. Adding CLIP to the vision stream consistently

Invoice Haystack 13
Table 3:VQA accuracy (%) comparison across DocHaystack, InfoHaystack, and Pro-
posed Invoice Haystack benchmarks. Zero-shot results use full document context; en-
tries marked “–” reflect context-window limitations. For V-RAG and VL-RAG, the best
accuracy across Top-1/3/5 is reported. Best result per column is in bold.
Doc Info Invoice
Approach Model100 200 1000 100 200 1000 500 1000 1500
Zero-ShotGemini-3-Flash 74.396.3–83.9 87.7– 18.0 – –
GPT-5.2 45.0 44.0 – 40.6 36.1 – 33.0 – –
InternVL3-8B 78.9 75.2 – 55.5 56.1 – – – –
Qwen3-VL-8B94.593.6 – 72.9 74.2 – – – –
Model+V-RAGGemini-3-Flash 84.4 85.3 75.2 83.2 80.074.251.0 48.5 44.0
GPT-5.2 78.9 77.8 72.5 73.5 72.9 67.1 49.5 45.5 41.5
InternVL3-8B 71.6 66.9 60.6 55.5 54.8 51.6 32.0 30.0 26.5
Qwen3-VL-8B 86.2 80.7 61.9 68.4 71.6 61.9 48.0 45.0 41.0
Model+VL-RAGGemini-3-Flash 86.2 84.476.182.6 83.9 73.5 60.056.5 53.0
GPT-5.2 83.5 77.1 73.4 75.5 72.9 64.5 61.0 54.0 52.0
InternVL3-8B 73.4 66.9 65.1 56.1 56.1 51.6 42.5 36.0 33.5
Qwen3-VL-8B 87.2 83.5 73.4 72.9 73.574.2 63.556.0 52.5
degrades performance. The configuration CLIP + SigLIP + OpenCLIP achieves
an average Recall@1 of 44.0%, while SigLIP + OpenCLIP alone (combined with
the text stream) performs better. When CLIP is included in the full VL-RAG
ensemble (CLIP + SigLIP + OpenCLIP + Text), average Recall@1 drops to
54.5% compared to the No-CLIP equivalent. On Invoice Haystack-1500 specifi-
cally, adding CLIP reduces Recall@1 from 50.0% to 33.5%. This counterintuitive
result suggests that CLIP’s natural-image pre-training introduces retrieval noise,
prioritizing global aesthetic similarity over document-level semantic content.
VLMfiltercontribution.AddingtheVLMfilterconsistentlyimprovesperfor-
manceacrossallconfigurations.ThefullVL-RAGpipeline(SigLIP+OpenCLIP
+ Text + VLM Filter) achieves the best average Recall@1 of 70.5%, compared
Table 4:Ablation study for VL-RAG on Document Haystack-1000, InfoHaystack-
1000, and Invoice Haystack-1500. Results reported as Recall@1 / Recall@3 / Recall@5.
Avg. denotes the mean across all three benchmarks.
Components Doc-1000 Info-1000 Invoice-1500 Average
CLIP SigLIP OpenCLIP Text VLM R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5 R@1 R@3 R@5
✓✗ ✗ ✗ ✗ 27.5 41.3 47.7 53.6 68.4 72.3 7.0 11.0 14.0 29.4 40.2 44.7
✗✓✗ ✗ ✗ 41.3 51.4 54.1 45.8 62.6 69.7 20.0 30.0 37.0 35.7 48.0 53.6
✗ ✗✓✗ ✗ 41.3 59.6 66.1 57.4 72.3 79.4 19.5 24.0 28.0 39.4 52.0 57.8
✗ ✗ ✗✓✗ 58.7 76.1 82.6 67.1 80.7 85.2 31.5 42.5 49.5 52.4 66.4 72.4
✓ ✓✗ ✗ ✗ 42.2 58.7 63.3 58.7 74.8 79.4 14.5 23.0 29.0 38.5 52.2 57.2
✓ ✓ ✓✗ ✗ 45.9 66.1 74.3 65.2 76.1 81.9 21.0 30.5 35.5 44.0 57.6 63.9
✓ ✓ ✓ ✓✗ 57.8 77.1 78.0 72.3 81.3 84.5 33.5 42.0 47.0 54.5 66.8 69.8
✗✓ ✓ ✓✗ 62.4 78.9 78.9 70.3 81.3 85.8 35.0 45.0 48.5 55.9 68.4 71.1
✓ ✓ ✓ ✓ ✓ 77.1 82.6 82.6 83.2 85.8 86.5 48.0 52.0 52.5 69.4 73.5 73.8
✗✓ ✓ ✓ ✓ 77.1 81.7 81.7 84.5 87.7 89.0 50.0 53.5 54.0 70.5 74.3 74.9

14 Heethanjan et al.
Fig. 5:Top-kselection ablation analysis for VQA on our Invoice Haystack benchmark.
We evaluate the impact of different retrieval depths (k= 1,3,5) on VQA accuracy
across all three corpus scales (Invoice Haystack-500, Invoice Haystack-1000, Invoice
Haystack-1500) for four VLMs: Gemini-3-Flash-Preview, GPT-5.2, InternVL3-8B, and
Qwen3-VL-8B-Instruct, all integrated with our VL-RAG framework.
to 55.9% without the filter for the same encoder set. On Invoice Haystack-1500,
the VLM filter raises Recall@1 from 35.0% to 50.0%, with the most notable gains
in Recall@3 and Recall@5 (54.0% and 54.5% respectively). The filter is particu-
larly effective in homogeneous settings where vector similarity fails to separate
the correct document from visually near-identical candidates.
Top-kRetrieval Analysis.Figure 5 analyses the effect of retrieval depth on
VQA accuracy across all four VLMs on our Invoice Haystack benchmark. Gen-
erally, accuracy increases askgrows from 1 to 5, since providing more can-
didate documents raises the likelihood that the correct invoice appears in the
VLM context. However, this trend is not monotonic for all models: Gemini and
Qwen3-VL both show a slight accuracy decline at top-5 compared to Top-3 on
Invoice Haystack-500 (Gemini: 60.0%→58.0%; Qwen3-VL: 64.0%→63.0%),
suggesting that beyonda certain retrieval depth, additional candidates introduce
distractors that degrade answer precision for stronger models. GPT-5.2 and In-
ternVL3 benefit more consistently from increasingkacross all corpus scales. In-
voice Haystack-500 yields the highest accuracy across all models andksettings,
while Invoice Haystack-1500 remains the hardest split, reflecting the greater re-
trieval difficulty at larger corpus scales. The persistent performance gap across
corpus scales, regardless ofkconfirms that the primary bottleneck is retrieval
precision under visual homogeneity rather than the number of candidates passed
to the VLM.
6 Conclusion
This work demonstrated that embedding collapse is a fundamental and previ-
ously unaddressed flaw in multi-document retrieval benchmarks, and that visu-
ally homogeneous enterprise collections represent a qualitatively distinct chal-
lenge from the diverse documents on which existing methods were developed. We
introduced Invoice Haystack to stress-test retrieval under strong visual homo-
geneity, where even the strongest vision ensemble reaches only 40.0% Recall@1
at scale.VL-RAGaddresses this through dual-stream fusion of text and visual
signals, achieving 50.0% Recall@1 on Invoice Haystack-1500 while maintaining

Invoice Haystack 15
consistent gains on heterogeneous benchmarks, confirming hybrid encoding as a
consistently beneficial strategy regardless of corpus composition. Current limita-
tions include fixed equal-weight fusion and zero-shot encoders without domain-
specific fine-tuning; future work should investigate learnable fusion weights con-
ditioned on corpus statistics, encoder fine-tuning on financial documents, and
benchmark extension to contracts and purchase orders.
Acknowledgements
Naveed Akhtar is a recipient of the Australian Research Council Discovery Early
Career Researcher Award (project # DE230101058) funded by the Australian
Government. This research was also supported by The University of Melbourne’s
Research Computing Services and the Petascale Campus Initiative. This work
is also partially supported by the Google Research Scholar Program Award.
References
1. van der Aalst, J., Weske, M., Grünbauer, D.: Workflow patterns in invoice process-
ing automation. ACM Computing Surveys54(6), 1–35 (2021)
2. Abaskohi, A., Gella, S., Carenini, G., Laradji, I.H.: Fm2ds: Few-shot multimodal
multihop data synthesis with knowledge distillation for question answering (2025),
https://arxiv.org/abs/2412.07030
3. ABBYY: ABBYY FineReader Engine: Advanced OCR SDK.https://www.abbyy.
com/ocr-sdk/(2024), accessed: 2024-02-23
4. Alayrac, J.B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K.,
Mensch, A., Millican, K., Reynolds, M., et al.: Flamingo: A visual language model
for few-shot learning. In: NeurIPS (2022)
5. Anderson, M., Davis, P., Wilson, K.: Survey of enterprise document management
practices. Information Management Journal47(3), 28–41 (2023)
6. Anthropic: The claude 3 model family: Opus, sonnet, haiku. Tech. rep., Anthropic
(2024)
7. Anthropic: Claude 3.7 sonnet model card. Tech. rep., Anthropic (2025),https:
//www.anthropic.com/news/claude-3-7-sonnet
8. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-rag: Learning to retrieve,
generate, and critique through self-reflection. In: The Twelfth International Con-
ference on Learning Representations (2023)
9. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao,
C.,Ge,C.,Ge,W.,Guo,Z.,Huang,Q.,Huang,J.,Huang,F.,Hui,B.,Jiang,S.,Li,
Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D.,
Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun,
Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T.,
Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F.,
Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y.,
Zhu, K.: Qwen3-vl technical report (2025),https://arxiv.org/abs/2511.21631
10. Blecher, L., Cucurull, G., Scialom, T., Stojnic, R.: Nougat: Neural optical under-
standing for academic documents. arXiv preprint arXiv:2308.13418 (2023)

16 Heethanjan et al.
11. Chang, Y., Narang, M., Suzuki, H., Cao, G., Gao, J., Bisk, Y.: Webqa: Multihop
and multimodal qa. In: Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition. pp. 16495–16504 (2022)
12. Chen, J., Xu, D., Fei, J., Feng, C.M., Elhoseiny, M.: Document haystacks: Vision-
languagereasoningoverpilesof1000+documents.In:ProceedingsoftheComputer
Vision and Pattern Recognition Conference. pp. 24817–24826 (2025)
13. Chen, W., Hu, H., Chen, X., Verga, P., Cohen, W.: Murag: Multimodal retrieval-
augmented generator for open question answering over images and text. In: Pro-
ceedings of the 2022 Conference on Empirical Methods in Natural Language Pro-
cessing. pp. 5558–5570 (2022)
14. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C.,
Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive
language-image learning. In: 2023 IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR). p. 2818–2829. IEEE (Jun 2023).https://doi.
org/10.1109/cvpr52729.2023.00276,http://dx.doi.org/10.1109/CVPR52729.
2023.00276
15. Dai, W., Li, J., Li, D., Tiong, A., Zhao, J., Wang, W., Li, B., Fung, P.N., Hoi,
S.: Instructblip: Towards general-purpose vision-language models with instruction
tuning. Advances in neural information processing systems36, 49250–49267 (2023)
16. Davis,L.,Chen,S.,Soricut,V.,Bansal,M.:Visually-richdocumentunderstanding:
Challenges and opportunities. arXiv preprint arXiv:2203.06482 (2022)
17. DeepMind, G.: Gemini 2.5 flash technical report. Tech. rep., Google DeepMind
(2025),https://deepmind.google/models/gemini/flash
18. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidi-
rectional transformers for language understanding. In: Proceedings of the 2019
conference of the North American chapter of the association for computational
linguistics: human language technologies, volume 1 (long and short papers). pp.
4171–4186 (2019)
19. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner,
T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is
worth 16x16 words: Transformers for image recognition at scale. arXiv preprint
arXiv:2010.11929 (2020)
20. Du, H., Zhang, J., Nan, G., Deng, W., Chen, Z., Zhang, C., Xiao, W., Huang, S.,
Pan, Y., Qi, T., Leng, S.: From easy to hard: The mir benchmark for progressive
interleaved multi-image reasoning (2025),https://arxiv.org/abs/2509.17040
21. Du, Y., Tian, M., Ronanki, S., Rongali, S., Bodapati, S., Galstyan, A., Wells, A.,
Schwartz, R., Huerta, E.A., Peng, H.: Context length alone hurts llm performance
despite perfect retrieval. arXiv preprint arXiv:2510.05381 (2025)
22. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P.:
Colpali: Efficient document retrieval with vision language models. arXiv preprint
arXiv:2407.01449 (2024)
23. Google DeepMind: Gemini (2026),https://deepmind.google/models/gemini/,
accessed: 2026-03-04
24. Google DeepMind: Gemini 3.1 pro model card (February 2026),https://
deepmind.google/models/model-cards/gemini-3-1-pro/, accessed: 2026-03-04
25. Gu, J., Kuen, J., Morariu, V.I., Zhao, H., Jain, R., Barmpalios, N., Nenkova,
A., Sun, T.: Unidoc: Unified pretraining framework for document understanding.
Advances in Neural Information Processing Systems34, 39–50 (2021)
26. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.: Retrieval augmented language
model pre-training. In: International conference on machine learning. pp. 3929–
3938. PMLR (2020)

Invoice Haystack 17
27. Hu, Z., Iscen, A., Sun, C., Wang, Z., Chang, K.W., Sun, Y., Schmid, C., Ross,
D.A., Fathi, A.: Reveal: Retrieval-augmented visual-language pre-training with
multi-source multimodal knowledge memory. In: Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition. pp. 23369–23379 (2023)
28. Huang, Y., Lv, T., Cui, L., Lu, Y., Wei, F.: Layoutlmv3: Pre-training for document
aiwithunifiedtextandimagemasking.In:ProceedingsofACMMultimedia(2022)
29. Izacard, G., Grave, E.: Leveraging passage retrieval with generative models for
open domain question answering. In: Proceedings of the 16th conference of the
european chapter of the association for computational linguistics: main volume.
pp. 874–880 (2021)
30. Izacard, G., Lewis, P., Lomeli, M., Hosseini, L., Petroni, F., Schick, T., Dwivedi-
Yu, J., Joulin, A., Riedel, S., Grave, E.: Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine Learning Research24(251), 1–43
(2023)
31. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih,
W.t.: Dense passage retrieval for open-domain question answering. In: EMNLP
(2020)
32. Kashyap, S., Shirai, S., Mihindukulasooriya, N., Samulowitz, H.: Structtext: A
synthetic table-to-text approach for benchmark generation with multi-dimensional
evaluation (2025),https://arxiv.org/abs/2507.21340
33. Katti, A.R., Reisswig, C., Guder, C., Brarda, S., Bickel, S., Höhne, J., Faddoul,
J.B.: Chargrid: Towards understanding 2d documents. In: Proceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. pp. 4459–4469
(2018)
34. Kazemi, M., Dikkala, N., Anand, A., Devic, P., Dasgupta, I., Liu, F., Fatemi, B.,
Awasthi, P., Guo, D., Gollapudi, S., Qureshi, A.: Remi: A dataset for reasoning
with multiple images (2024),https://arxiv.org/abs/2406.09175
35. Khattab, O., Zaharia, M.: Colbert: Efficient and effective passage search via con-
textualized late interaction over bert. In: SIGIR (2020)
36. Kim, G., Hong, T., Yim, M., Nam, J., Park, J., Yim, J., Hwang, W., Yun, S.,
Han, D., Park, S.: Ocr-free document understanding transformer. In: European
Conference on Computer Vision. pp. 498–517. Springer (2022)
37. Koukounas, A., Mastrapas, G., Eslami, S., Wang, B., Akram, M.K., Günther, M.,
Mohr, I., Sturua, S., Wang, N., Xiao, H.: jina-clip-v2: Multilingual multimodal
embeddings for text and images. arXiv preprint arXiv:2412.08802 (2024)
38. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C.,
Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al.: Natural questions: a bench-
mark for question answering research. Transactions of the Association for Compu-
tational Linguistics7, 453–466 (2019)
39. Lee, K., Joshi, M., Turc, I.R., Hu, H., Liu, F., Eisenschlos, J.M., Khandelwal, U.,
Shaw,P.,Chang,M.W.,Toutanova,K.:Pix2struct:Screenshotparsingaspretrain-
ing for visual language understanding. In: International Conference on Machine
Learning. pp. 18893–18912. PMLR (2023)
40. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoy-
anov, V., Zettlemoyer, L.: Bart: Denoising sequence-to-sequence pre-training for
natural language generation, translation, and comprehension. In: Proceedings of
the 58th annual meeting of the association for computational linguistics. pp. 7871–
7880 (2020)
41. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for

18 Heethanjan et al.
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
42. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pre-
training with frozen image encoders and large language models. In: International
conference on machine learning. pp. 19730–19742. PMLR (2023)
43. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tun-
ing. In: Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition. pp. 26296–26306 (2024)
44. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural
information processing systems36, 34892–34916 (2023)
45. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M.,
Zettlemoyer, L., Stoyanov, V.: Roberta: A robustly optimized bert pretraining
approach. arXiv preprint arXiv:1907.11692 (2019)
46. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin
transformer:Hierarchicalvisiontransformerusingshiftedwindows.In:Proceedings
of the IEEE/CVF international conference on computer vision. pp. 10012–10022
(2021)
47. Lu,J.,Batra,D.,Parikh,D.,Lee,S.:Vilbert:Pretrainingtask-agnosticvisiolinguis-
tic representations for vision-and-language tasks. Advances in neural information
processing systems32(2019)
48. Masry, A., Do, X.L., Tan, J.Q., Joty, S., Hoque, E.: Chartqa: A benchmark for
question answering about charts with visual and logical reasoning. In: Findings of
the association for computational linguistics: ACL 2022. pp. 2263–2279 (2022)
49. Mathew, M., Bagal, V., Tito, R., Karatzas, D., Valveny, E., Jawahar, C.: Info-
graphicvqa. In: Proceedings of the IEEE/CVF Winter Conference on Applications
of Computer Vision. pp. 1697–1706 (2022)
50. Mathew, M., Karatzas, D., Jawahar, C.: Docvqa: A dataset for vqa on document
images. In: Proceedings of the IEEE/CVF winter conference on applications of
computer vision. pp. 2200–2209 (2021)
51. Mei, L., Mo, S., Yang, Z., Chen, C.: A survey of multimodal retrieval-augmented
generation. arXiv preprint arXiv:2504.08748 (2025)
52. Nussbaum, Z., Morris, J.X., Duderstadt, B., Mulyar, A.: Nomic embed: Training a
reproducible long context text embedder. arXiv preprint arXiv:2402.01613 (2024)
53. OpenAI: Gpt-4o: Multimodal capabilities with enhanced reasoning. Tech. rep.,
OpenAI (2024)
54. OpenAI: Models. OpenAI API Documentation (2026),https://developers.
openai.com/api/docs/guides/latest-model, accessed: 2026-03-04
55. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V.,
Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust
visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
56. Penamakuri, A.S., Gupta, M., Gupta, M.D., Mishra, A.: Answer mining from a
pool of images: Towards retrieval-based visual question answering. arXiv preprint
arXiv:2306.16713 (2023)
57. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. In: International conference on machine learning. pp.
8748–8763. PmLR (2021)
58. Robertson, S., Zaragoza, H.: The probabilistic relevance framework: BM25 and
beyond, vol. 4. Now Publishers Inc (2009)

Invoice Haystack 19
59. Schreiber, S., Agne, S., Wolf, I., Dengel, A., Ahmed, S.: Deepdesrt: Deep learning
for detection and structure recognition of tables in document images. In: 2017 14th
IAPR international conference on document analysis and recognition (ICDAR).
vol. 1, pp. 1162–1167. IEEE (2017)
60. Shen, Z., Lo, K., Wang, L.L., Kuehl, B., Weld, D.S., Downey, D.: VILA: Improv-
ing structured content extraction from scientific PDFs using visual layout groups.
Transactions of the Association for Computational Linguistics10, 376–392 (2022).
https://doi.org/10.1162/tacl_a_00466,https://aclanthology.org/2022.
tacl-1.22/
61. Smith, B., Johnson, R.: Template diversity in enterprise invoice repositories: An
empirical study. In: Document Engineering (2023)
62. Smith, R.: An overview of the tesseract ocr engine. In: Ninth international con-
ference on document analysis and recognition (ICDAR 2007). vol. 2, pp. 629–633.
IEEE (2007)
63. Talmor, A., Yoran, O., Catav, A., Lahav, D., Wang, Y., Asai, A., Ilharco, G.,
Hajishirzi, H., Berant, J.: Multimodalqa: Complex question answering over text,
tables and images. In: ICLR (2021)
64. Tanaka, R., Nishida, K., Yoshida, S.: Visualmrc: Machine reading comprehension
on document images. In: Proceedings of the AAAI Conference on Artificial Intel-
ligence. vol. 35, pp. 13878–13888 (2021)
65. Tang, Z., Yang, Z., Wang, G., Fang, Y., Liu, Y., Zhu, C., Zeng, M., Zhang, C.,
Bansal, M.: Unifying vision, text, and layout for universal document processing. In:
Proceedings of the IEEE/CVF conference on computer vision and pattern recog-
nition. pp. 19254–19264 (2023)
66. Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,L.,Gomez,A.N.,Kaiser,
Ł., Polosukhin, I.: Attention is all you need. Advances in neural information pro-
cessing systems30(2017)
67. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S.,
Shao, J., Wang, Z., Chen, Z., Zhang, H., Yang, G., Wang, H., Wei, Q., Yin, J.,
Li, W., Cui, E., Chen, G., Ding, Z., Tian, C., Wu, Z., Xie, J., Li, Z., Yang, B.,
Duan, Y., Wang, X., Hou, Z., Hao, H., Zhang, T., Li, S., Zhao, X., Duan, H.,
Deng, N., Fu, B., He, Y., Wang, Y., He, C., Shi, B., He, J., Xiong, Y., Lv, H., Wu,
L., Shao, W., Zhang, K., Deng, H., Qi, B., Ge, J., Guo, Q., Zhang, W., Zhang,
S., Cao, M., Lin, J., Tang, K., Gao, J., Huang, H., Gu, Y., Lyu, C., Tang, H.,
Wang, R., Lv, H., Ouyang, W., Wang, L., Dou, M., Zhu, X., Lu, T., Lin, D.,
Dai, J., Su, W., Zhou, B., Chen, K., Qiao, Y., Wang, W., Luo, G.: Internvl3.5:
Advancing open-source multimodal models in versatility, reasoning, and efficiency
(2025),https://arxiv.org/abs/2508.18265
68. Wang, Z., Huang, M., Xu, Y., Chen, H.: Automated financial document processing:
Challenges and opportunities. In: FinNLP Workshop (2023)
69. Wu, T.H., Biamby, G., Quenum, J., Gupta, R., Gonzalez, J.E., Darrell, T., Chan,
D.M.: Visual haystacks: A vision-centric needle-in-a-haystack benchmark (2025),
https://arxiv.org/abs/2407.13766
70. Xiao, F., Luan, H., Zhao, M., Li, J., Wu, L., Xu, J.: Bge: A family of general text
embeddings. arXiv preprint arXiv:2309.07597 (2023)
71. Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., Zhou, M.: Layoutlm: Pre-training of
text and layout for document image understanding. In: Proceedings of the 26th
ACM SIGKDD international conference on knowledge discovery & data mining.
pp. 1192–1200 (2020)

20 Heethanjan et al.
72. Xu, Y., Xu, Y., Lv, T., Cui, L., Wei, F., Wang, G., Lu, Y., Florencio, D., Zhang, C.,
Che, W., Zhang, M., Zhou, L.: Layoutlmv2: Multi-modal pre-training for visually-
rich document understanding. In: Proceedings of the Annual Meeting of the Asso-
ciation for Computational Linguistics (ACL) (2021)
73. Yan, S.Q., Gu, J.C., Zhu, Y., Ling, Z.H.: Corrective retrieval augmented generation
(2024)
74. Yasunaga, M., Aghajanyan, A., Shi, W., James, R., Leskovec, J., Liang, P., Lewis,
M., Zettlemoyer, L., Yih, W.t.: Retrieval-augmented multimodal language model-
ing. arXiv preprint arXiv:2211.12561 (2022)
75. Ye, J., Hu, A., Xu, H., Ye, Q., Yan, M., Dan, Y., Zhao, C., Xu, G., Li, C., Tian,
J., Qi, Q., Zhang, J., Huang, F.: mplug-docowl: Modularized multimodal large
language model for document understanding (2023),https://arxiv.org/abs/
2307.02499
76. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language im-
age pre-training. In: Proceedings of the IEEE/CVF international conference on
computer vision. pp. 11975–11986 (2023)
77. Zhong, X., Tang, J., Yepes, A.J.: Publaynet: largest dataset ever for document
layout analysis. In: 2019 International conference on document analysis and recog-
nition (ICDAR). pp. 1015–1022. IEEE (2019)
78. Zhou, Y., Wang, X., Kan, M.: Understanding embedding collapse in vision-
language models. In: ICML (2024)