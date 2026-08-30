# A Comparative Evaluation of Digitization Pipelines for Historiographical Sources

**Authors**: Marina Gómez Rey, Patricia Callejo, Mario Muñoz-Organero, Carlos Alario-Hoyos

**Published**: 2026-08-25 16:03:01

**PDF URL**: [https://arxiv.org/pdf/2608.24976v1](https://arxiv.org/pdf/2608.24976v1)

## Abstract
Purpose: The digitization of historical documents presents fundamental challenges for modern information retrieval and Artificial Intelligence (AI) systems. Optical character recognition (OCR) errors in source corpora propagate through retrieval-augmented generation (RAG) pipelines, compromising the factual accuracy of generated outputs. Methods: This study presents a systematic evaluation of PDF-to-text extraction pipelines applied to historiographical secondary sources on the Visigothic period. We assess thirteen distinct approaches spanning three methodological families: direct extraction, Large Language Model (LLM) post-correction, and chunk-and-extract. Documents are stratified into five categories based on production method and visual complexity. Performance is measured using character error rate (CER) and word error rate (WER) against manually corrected ground truth. Results: Results demonstrate that direct extraction with Marker achieves superior performance (98.70% CER accuracy; 97.71% WER accuracy overall), while conventional OCR pipelines exhibit substantial degradation on scanned documents and complex layouts. Embedded-text extraction performs well on digital PDFs but fails on scanned documents. LLM post-correction does not provide systematic improvements and frequently degrades accurate extractions. Conclusion: End-to-end document parsing is the most reliable approach for heterogeneous historical collections. Document characteristics such as scan quality, layout complexity, and the presence of embedded text layers have a significant impact on extraction accuracy. LLM-based post-correction should not be assumed beneficial by default and requires validation before large-scale application.

## Full Text


<!-- PDF content starts -->

A Comparative Evaluation of Digitization Pipelines for
Historiographical Sources
Marina G´ omez Rey1*, Patricia Callejo1*, Mario Mu˜ noz-Organero1and
Carlos Alario-Hoyos1
1*Telematics Engineering Department, Universidad Carlos III de Madrid, Legan´ es,
Madrid, 28911, Spain.
*Corresponding author(s). E-mail(s): marinago@pa.uc3m.es; pcallejo@it.uc3m.es;
Contributing authors: munozm@it.uc3m.es; calario@it.uc3m.es;
Abstract
Purpose:The digitization of historical documents presents fundamental challenges for modern infor-
mation retrieval and Artificial Intelligence (AI) systems. Optical character recognition (OCR) errors
in source corpora propagate through retrieval-augmented generation (RAG) pipelines, compromising
the factual accuracy of generated outputs.
Methods:This study presents a systematic evaluation of PDF-to-text extraction pipelines applied to
historiographical secondary sources on the Visigothic period. We assess thirteen distinct approaches
spanning three methodological families: direct extraction, Large Language Model (LLM) post-
correction, and chunk-and-extract. Documents are stratified into five categories based on production
method and visual complexity. Performance is measured using character error rate (CER) and word
error rate (WER) against manually corrected ground truth.
Results:Results demonstrate that direct extraction with Marker achieves superior performance
(98.70% CER accuracy; 97.71% WER accuracy overall), while conventional OCR pipelines exhibit
substantial degradation on scanned documents and complex layouts. Embedded-text extraction per-
forms well on digital PDFs but fails on scanned documents. LLM post-correction does not provide
systematic improvements and frequently degrades accurate extractions.
Conclusion:End-to-end document parsing is the most reliable approach for heterogeneous historical
collections. Document characteristics such as scan quality, layout complexity, and the presence of
embedded text layers have a significant impact on extraction accuracy. LLM-based post-correction
should not be assumed beneficial by default and requires validation before large-scale application.
Keywords:Digitization, OCR, historical documents, RAG
1 Introduction
Information quality fundamentally determines the
reliability of knowledge systems [5]. In text-
based applications, degradation introduced at the
digitization stage can cascade through subse-
quent processing layers, affecting search, retrieval,and knowledge generation tasks. This depen-
dency becomes critical when historical documents
serve as knowledge sources for modern Artificial
Intelligence (AI) systems, especially Retrieval-
Augmented Generation (RAG) architectures that
enhance Large Language Model (LLM) outputs
with retrieved contextual information [22, 24].
1
arXiv:2608.24976v1  [cs.DL]  25 Aug 2026

RAG-based systems operate through a multi-
stage process: documents are first segmented
into smaller chunks, which are then transformed
into vector representations using embedding mod-
els, and subsequently stored in vector databases.
When a user submits a query, the system retrieves
semantically relevant chunks and supplies them
as context to an LLM, which generates responses
conditioned on both the query and retrieved
information. This architecture has become foun-
dational for conversational AI systems, enabling
question answering over specific document col-
lections, supporting domain-specific applications,
and grounding model outputs in external knowl-
edge sources rather than relying solely on pre-
trained parameters.
However, the effectiveness of RAG-based sys-
tems depends critically on the quality of the
underlying document corpus. When source docu-
ments contain OCR character substitutions, word
segmentation failures, or layout corruption, these
errors propagate through the retrieval and gener-
ation stages [46]. RAG systems may fail to match
queries to relevant chunks due to corrupted entity
names or key terms. Even when retrieval succeeds,
corrupted content is passed directly to the genera-
tion model, which may reproduce or amplify errors
in its outputs. Because modern LLMs typically
generate fluent responses independent of source
quality, users may be unaware that responses are
grounded in corrupted content.
Historical texts present distinct challenges for
OCR systems integrated into RAG-based sys-
tems. Unlike contemporary digital documents, his-
torical materials frequently exhibit multi-column
layouts, non-standard typographical conventions,
archaic orthography, physical degradation, and
mixed content types [17]. These characteristics
introduce systematic errors including character
substitutions, word segmentation failures, reading
order corruption, and layout misinterpretation.
When such errors persist in digitized corpora, they
propagate through information retrieval pipelines
and may be reproduced by LLMs, creating factu-
ally incorrect outputs that appear authoritative to
end users.
Consider a representative failure case: when
queried “Who was deposed by Wit´ erico?”, a RAG-
based AI system responds “Liuva 11” rather than
the correct “Liuva II”. This error originates from
OCR misrecognition of the Roman numeral IIas the Arabic number 11. While superficially
minor, such errors systematically corrupt histor-
ical knowledge in systems deployed to millions
of users. The error becomes particularly rele-
vant because modern conversational AI systems
present outputs with high confidence regardless
of source quality, rendering digitization errors
invisible to non-expert users.
Despite advances in neural OCR architectures
and vision-language models [11], no existing solu-
tion reliably digitizes complex historiographical
documents. Recent work has explored LLM-based
post-correction [16, 20, 41], hybrid layout analysis
approaches [11], and end-to-end document pars-
ing [24]. However, systematic comparative evalu-
ations on heterogeneous historical corpora remain
limited. This study addresses three research ques-
tions:
RQ1:How do contemporary PDF-to-text
extraction pipelines perform across different
document types in historiographical secondary
sources, and what error patterns characterize each
approach?
RQ2:To what extent does document hetero-
geneity (production method, visual complexity,
scan quality) affect extraction accuracy across
pipeline architectures?
RQ3:To what extent does LLM-based post-
correction improve OCR quality for historical
documents?
We present a systematic evaluation of thirteen
extraction pipelines on a stratified corpus of his-
toriographical secondary sources on the Visigothic
period. This corpus is selected because it exhibits
the heterogeneity typical of digital humanities
collections, including poor quality scans, mod-
ern digitizations, native digital PDFs, and com-
plex layouts, with sufficient linguistic complexity
(Latin terms, proper nouns, Roman numerals) to
stress-test OCR systems beyond standard bench-
marks. We measure character and word level accu-
racy against manually corrected ground truth,
with evaluation focused on the quality require-
ments of modern AI systems, particularly RAG
architectures where OCR errors in source docu-
ments propagate directly to generated outputs.
This work makes three key contributions. First,
we provide empirical performance comparison of
thirteen extraction pipelines across five document
categories, revealing substantial variation in accu-
racy and characteristic failure modes. Second, we
2

demonstrate how document characteristics (pro-
duction method, scan quality, and layout complex-
ity) systematically affect extraction quality, with
direct implications for workflow design and tool
selection. Third, we present evidence that LLM-
based post-correction, despite theoretical promise,
can degrade performance under standard edit
distance metrics.
2 Related Work
OCR technology has evolved from rule-based pat-
tern matching systems to neural architectures
trained on large document corpora. Contemporary
approaches leverage convolutional neural networks
for character recognition combined with recur-
rent architectures for sequence modeling, substan-
tially improving accuracy over traditional meth-
ods. However, performance varies significantly
across document types and quality levels. Tesser-
act, a widely deployed open-source engine, uses
LSTM-based recognition following image prepro-
cessing and layout analysis, with a Connectionist
Temporal Classification (CTC) loss function [43].
EasyOCR represents an alternative approach,
employing end-to-end convolutional architectures
that bypass explicit segmentation stages [15]. The
effectiveness of these systems on historical docu-
ments with complex layouts and degraded quality
remains an open question, particularly relevant
in recent RAG-based systems where extraction
errors can corrupt retrieved context and generated
outputs.
Historical documents pose distinct challenges
for OCR systems. Khan et al. [17] survey AI
approaches for historical document transcription,
identifying three challenges: non-standard typog-
raphy, physical degradation, and archaic linguistic
forms. Martinek et al. [26] address data scarcity by
combining fully convolutional segmentation net-
works with recurrent recognition models, achiev-
ing competitive accuracy on 19th-century Ger-
man Fraktur texts with minimal training data.
Fleischhacker et al. [11] show that explicit lay-
out analysis prior to OCR substantially improves
extraction quality for multi-column 19th-century
documents, demonstrating that structure detec-
tion before text recognition outperforms direct
OCR approaches on complex layouts. Accurate
extraction of reading order requires thorough anal-
ysis of document layout. Breuel [6] introducedwhitespace rectangle-based approaches for docu-
ment structure detection. Contemporary systems
have evolved toward unified architectures that
jointly perform layout detection, table recognition,
and reading order determination [47].
Large language models have been proposed for
OCR error correction. Kanerva et al. [16] evaluate
LLM-based post-correction on historical Finnish
and Swedish documents, finding that improve-
ments depend on base OCR quality and language
characteristics. Sastre et al. [41] show that con-
straining LLM outputs to valid lexical items
reduces hallucination while preserving correction
capability. Levchenko [20] proposes frameworks
for evaluating LLM performance on historical
OCR, emphasizing domain-appropriate metrics.
These studies reveal tensions between linguistic
plausibility and preservation of original text.
Document quality fundamentally determines
RAG system reliability, as retrieval quality
directly determines generation quality. Zhang et
al. [46] demonstrate that OCR errors cascade
through RAG pipelines, corrupting both retrieval
and generation stages. Liu et al. [23] demonstrate
that document positioning within prompts sig-
nificantly affects LLM attention distribution and
accuracy, while Cuconasu et al. [8] show that
both document type and retrieval position influ-
ence RAG effectiveness. As AI systems scale to
incorporate digitized archives and scholarly repos-
itories, OCR quality emerges as a critical challenge
for factual accuracy and reliability. Our work pro-
vides systematic evaluation of PDF extraction
pipelines on historiographical documents, measur-
ing character and word level accuracy to inform
tool selection for digitization workflows that feed
RAG systems.
3 Methodology
This section describes the experimental method-
ology for evaluating PDF-to-text extraction
pipelines on historiographical sources. We first
detail corpus construction and document cat-
egorization, explain ground truth preparation,
describe each evaluated pipeline with its configu-
ration, provide tool selection rationale, and finally
present the evaluation metrics.
3

3.1 Corpus Construction and
Selection
We curated a corpus of historiographical sec-
ondary sources on the Visigothic kingdom (415–
721 CE) from Spanish academic publications.
This corpus was selected for three reasons. First,
Spanish academic publications from the late 20th
and early 21st centuries exhibit the heterogene-
ity typical of digital humanities collections: poor
quality scans, modern digitizations, native digital
PDFs, and complex multi-column layouts. Sec-
ond, the corpus contains sufficient linguistic and
typographical complexity to stress-test OCR sys-
tems: Latin terminology, proper nouns, Roman
numerals, specialized historical vocabulary, and
non-standard notation such as dagger symbols
and superscript footnotes. Third, historiographi-
cal texts present a critical use case for digitiza-
tion quality, as minor OCR errors (misrecognized
dates, corrupted proper nouns, confused Roman
numerals) can introduce significant factual inac-
curacies when these documents feed information
retrieval or RAG systems.
The selected corpus comprises 14 docu-
ments [1, 7, 10, 13, 25, 27, 29, 30, 32–34, 36, 39,
40], yielding 15 document units since one source
is split into two fragments. These are distributed
across five categories of three documents each.
This selection ensures that the evaluation cap-
tures the range of challenges that practitioners
encounter when digitizing historical scholarship.
Type 1: Low-Quality Scans with Anno-
tations:These documents are degraded scans
from older print publications. They contain man-
ual marginalia, underlining, physical stains, and
other artifacts introduced during the scanning
process or present in the original printed copies.
The image quality is generally poor, with uneven
lighting, skewed pages, and low contrast between
text and background, which are common errors
in historical texts [44]. These represent worst-
case scenarios for automated digitization and are
included to test the robustness of each pipeline
under adverse conditions. An example is shown in
Figure 1.
Type 2: Multi-Column Complex Lay-
outs:These documents come from magazines,
journals, or newspapers and feature two-column or
multi-column layouts, embedded figures, captions,
and non-linear reading order. They are included
Fig. 1: Example of Type 1 document.
Fig. 2: Example of Type 2 document.
specifically to test the layout analysis capabili-
ties of each pipeline, as incorrect column detection
leads to reading order corruption where text from
adjacent columns is interleaved. An example is
shown in Figure 2.
Type 3: Clean Scans:These documents
are high-quality scans from modern publications
with minimal visual artifacts and standard single-
column layouts. Although they originate from
scanned print sources, the scanning quality is
good, with clear typography, high contrast, and
minimal noise. These documents establish base-
line performance on favorable historical materials,
representing the best-case scenario for scanned
documents. An example is shown in Figure 3.
4

Fig. 3: Example of Type 3 document.
Fig. 4: Example of Type 4 document.
Type 4: Digital PDFs:These documents
were originally created digitally (e.g., in Microsoft
Word) and exported to PDF with embedded text
layers. Because the text was never printed and
scanned, these documents contain no visual degra-
dation and their embedded text layers should
be a faithful representation of the original con-
tent. They are included to test the performance
of embedded-text extraction approaches under
ideal conditions and to provide an upper bound
for extraction quality. An example is shown in
Figure 4.
Type 5: No Embedded Text:These doc-
uments are scanned documents that, by their
nature, do not contain embedded text layers.
Unlike previous document types, where some
scanned PDFs may include an OCR-generated
text layer added during the scanning process,
these documents provide only image data. Any
pipeline that relies solely on extracting embed-
ded text will produce empty output for these
documents, making them a critical test for distin-
guishing between true OCR capability and mere
text layer extraction. An example is shown in
Figure 5.
Fig. 5: Example of Type 5 document.
3.2 Ground Truth Construction
The ground truth was constructed manually by
transcribing all text contained in the 15 selected
documents. To manage workload while maintain-
ing representativeness, we selected approximately
15 pages from each document. These manual tran-
scriptions ensure maximum accuracy and avoid
errors from automated processing. The resulting
manually edited transcriptions were saved as plain
text files for subsequent use in the evaluation of
the pipeline outputs.
3.3 Evaluated Pipelines
We evaluated thirteen PDF-to-text extraction
pipelines organized into three methodological fam-
ilies. The evaluation follows a comparative design
where the thirteen extraction pipelines process
the same corpus and their outputs are compared
against the manually corrected ground truth. We
organize the pipelines into three methodological
families based on their processing approach:direct
extractionapplies text extraction or OCR with-
out post-processing;LLM post-correctionadds
language model correction to direct extraction
outputs;chunk-and-extractsegments pages into
visual blocks before applying OCR or vision mod-
els to each block. This organization isolates the
contribution of each processing component (text
extraction, layout analysis, OCR engine, post-
correction) and reveals how these interact with
document characteristics. We focus exclusively
on open-source tools for reproducibility and to
reflect the practical constraints of digital human-
ities projects, where commercial API costs may
be prohibitive for large-scale digitization. Below,
we describe each pipeline and its configuration in
detail.
3.3.1 Direct Extraction
The first approach relied on using extraction tools
directly, without performing any subsequent post-
processing actions. Three main tools were used:
5

PyMuPDF [2], Docling [14], and Marker [9], with
Docling evaluated under three different configura-
tions.
PyMuPDFis a Python library for extracting,
converting, and manipulating PDF documents. It
provides a set of functionalities including pars-
ing PDF and other document formats, extracting
text and images from pages, modifying docu-
ment content by inserting, deleting, or rearranging
pages and annotations, and retrieving text in
different output formats. PyMuPDF also sup-
ports advanced document analysis features such
as handling annotations and form fields, con-
verting documents between formats, and optional
OCR support. Additionally, it includes integra-
tion capabilities for pipelines such as RAG/LLM
workflows.
For this study, the text extraction was per-
formed using theto markdownfunction from the
pymupdf4llmlibrary, which converts PDF con-
tent to Markdown format. This approach is fast
and straightforward but completely dependent on
embedded text existence and quality, failing on
scanned documents without text layers.
Doclingis an open-source document process-
ing and parsing toolkit originally developed by
IBM, designed to integrate with the generative AI
ecosystem [24]. Its main objective is to convert a
wide variety of document formats into structured,
machine-readable representations that can be con-
sumed by Generative AI systems. It supports
end-to-end document workflows including RAG,
question answering, and LLM-based analysis.
Docling provides parsing of a wide range of for-
mats with advanced capabilities for layout under-
standing, table structure recognition, and OCR
for scanned content. It offers integration with AI
frameworks and provides utilities for document
chunking and serialization into different output
formats including Markdown, HTML, and JSON.
We evaluated three configurations:
1.Docling (No OCR):Configured with
doocr=False, extracts text from document
structure while preserving formatting through
Markdown symbols. Behaves similarly to
PyMuPDF but maintains structural informa-
tion.
2.Docling + Tesseract:Configured with
doocr=Trueusing Tesseract [43], whichapplies LSTM-based recognition with prepro-
cessing, layout analysis, and region segmenta-
tion. Performance typically degrades on noisy
images, complex layouts, or unusual fonts.
3.Docling + EasyOCR:Configured with
doocr=Trueusing EasyOCR, a convolutional
neural network architecture that bypasses
explicit segmentation. Expected to handle
noisy images better but weaker at layout anal-
ysis for complex page structures.
Markeris a document parser developed
by Datalab that converts PDFs, images, and
office documents into structured formats such
as Markdown, JSON, or HTML. Unlike tradi-
tional pipelines that chain separate extraction
and processing tools, Marker takes an end-to-
end approach built on the Surya model suite, a
set of vision transformers trained specifically for
document understanding.
For layout detection, Marker uses a modi-
fied version of EfficientViT, a lightweight vision
transformer architecture; for text recognition, it
adapts Donut (Document Understanding Trans-
former) [18], a model that reads text directly from
document images without a separate OCR step.
Both models were trained from scratch to handle
complex elements such as tables, equations, and
multi-column layouts in over 90 languages.
Internally, Marker follows a three-stage
pipeline. First, specialized models extract raw
content, detecting layout regions, recognizing
text, parsing table structures, and identifying
equations. Second, a set of processors refine
this output by merging fragmented text spans,
correcting hyphenation artifacts, and establish-
ing reading order. Third, renderers convert the
processed content into the desired output format.
A key configuration option for our purposes is
forcing OCR on every page, even when the PDF
already contains embedded text. This ensures con-
sistent extraction quality across both digital and
scanned documents.
3.3.2 LLM Post-Correction
LLM post-correction pipelines add language
model error correction to direct extraction out-
puts. The motivation is that LLMs can identify
6

and reduce common OCR errors such as mis-
spellings, incorrect word segmentation, and mis-
placed punctuation through their understanding
of language patterns.
In this workflow, the text obtained from the
previous extraction stage is used as input to
an LLM, which is prompted explicitly to cor-
rect the provided text and return a cleaned
version. The model selected for this task was
Qwen3 (8B parameters) [38], chosen because it
provides a good balance between performance and
computational resource requirements, delivering
acceptable results without the need for large-scale
infrastructure.
The pipeline operates as follows: the raw
OCR text is first segmented into paragraphs.
Each paragraph is then processed independently
by the LLM, which receives a carefully engi-
neered prompt instructing it to correct errors
and return cleaned text. Processing paragraphs
independently helps manage context lengths and
allows the model to focus on localized corrections,
reducing the risk of hallucinations and improving
overall results.
Creating an effective prompt is essential in
Generative AI tasks, and following prompt engi-
neering guidelines can significantly improve task
outcomes. Common best practices include provid-
ing a clear set of instructions, properly separating
sections within the prompt, using techniques such
as few-shot prompting, and specifying explicit
constraints on the output format. The prompt
used for this task is the following:
You are an expert linguist in Spanish
and your task is to **correct text
coming from OCR**.
**RULES**:
- Keep the original content exactly as
it is, without inventing any
information.
- Correct incorrect spacing, misplaced
hyphens, and broken or joined words.
- Fix accents and typographical errors.
- Correct words with spelling mistakes.
- Do NOT modify proper names except to
correct accents.
- Do not rewrite or summarize: only
correct.**Text to correct**:
"""
{paragraph}
"""
Return **only the corrected text**,
with no comments, explanations,
additional text, or extra words.
Do not include "Corrected:" at the
beginning or add any extra text at
the end.
Five post-correction pipelines were evalu-
ated, each combining a different direct extraction
method with the same LLM correction step.
1.PyMuPDF + LLM: PyMuPDF extraction
followed by paragraph-level LLM correction.
2.Docling (No OCR) + LLM: Docling
embedded-text extraction followed by LLM
post-processing.
3.Docling + Tesseract + LLM: Docling with
Tesseract OCR extraction followed by LLM
correction.
4.Docling + EasyOCR + LLM: Docling
with EasyOCR extraction followed by LLM
correction.
5.Marker + LLM: Marker extraction followed
by LLM correction.
3.3.3 Chunk and Extract
Inspired by Fleischhacker et al. [11], who demon-
strate that detecting document layout before text
extraction significantly improves results, a two-
step process was implemented: first, the document
was segmented into chunks, and second, the text
was extracted from each chunk.
The chunking procedure converts each PDF
page into an image using PyMuPDF and
OpenCV, an open-source computer vision library
that provides the image processing primitives used
throughout the pipeline. The DPI (dots per inch)
used for this conversion is a critical parameter:
values that are too low reduce image quality and
produce fewer chunks, while values that are too
high cause excessive fragmentation. We used val-
ues between 350 and 450 DPI depending on docu-
ment complexity. The page image then undergoes
contrast enhancement and morphological opera-
tions that merge nearby characters into coherent
text regions. Contours are detected around these
7

regions and extracted as individual image files,
each representing a text block.
In practice, some blocks still contained too
many elements, for instance, an entire page
extracted as a single block. To address this, the
chunking function was made recursive. Each block
is evaluated against three criteria: (1) aspect
ratio exceeding 2.5, suggesting multiple columns;
(2) ink density below 40%, measured through
the Otsu thresholding method [35], an auto-
matic binarisation technique that separates fore-
ground text from background, indicating whites-
pace between distinct regions; and (3) more than
three disconnected contours, indicating multiple
independent text sections. If any criterion is met,
the block is re-segmented.
The recursive process introduced duplicate
blocks, which are removed. Some visual noise
present in the original scanned pages is also
extracted as blocks, which must be handled during
text extraction.
With the blocks prepared, the next step is text
extraction. Three approaches were evaluated.
1.Chunk + MiniCPM-V:Each extracted
block is submitted to the MiniCPM-V [31]
vision model (8B parameters). Vision models
are designed to process images as input accom-
panied by a text prompt specifying the desired
output. The prompt used instructs the model
to: (1) first verify whether the image contains
text, if not, return “NO TEXT”, and (2) if text
is present, return only the original text, keep-
ing paragraphs separated, without inventing
words, completing cut-off sentences, or adding
any text not present in the image. The “NO
TEXT” instruction accounts for the fact that
some blocks consist solely of noise from the
original documents.
2.Chunk + Qwen3-VL:The same chunking
and prompting strategy is applied using the
Qwen3-VL [38] vision model (4B parameters).
This smaller model was included to evaluate
whether a more lightweight vision model could
achieve comparable extraction quality.
3.Chunk + EasyOCR:Instead of a vision
model, direct OCR is applied to each block
using EasyOCR. Since the main limitation of
EasyOCR is layout detection, while its strength
lies in accurate text extraction at the wordlevel, and the layout has already been deter-
mined through the chunking process, Easy-
OCR should produce more reliable word-level
extraction when applied to individual, well-
segmented blocks.
3.4 Tool Selection Rationale
The tools evaluated in this study were selected
based on their importance in the open-source doc-
ument parsing ecosystem, as evidenced by commu-
nity adoption and their established role as base-
lines in the Document AI and Vision-Language
Model (VLM) literature.
For OCR engines, we include two comple-
mentary approaches. Tesseract [12, 42], with over
72,000 GitHub stars, is the most widely deployed
open-source OCR engine and serves as the funda-
mental baseline for document digitization. It relies
on a Long Short-Term Memory (LSTM) architec-
ture and represents the traditional approach to
text recognition. EasyOCR [4, 15], with approxi-
mately 29,000 stars, represents the deep learning
approach to scene text recognition. It combines
a CRAFT detector with a ResNet-LSTM-CTC
classifier, offering superior performance on non-
standard fonts compared to traditional engines.
For PDF parsing, we evaluate three tools span-
ning different design philosophies. PyMuPDF [2,
28] dominates Python PDF text extraction with
over 41 million monthly PyPI downloads [37].
Rather than performing OCR, it accesses the
underlying PDF object tree directly, enabling
high-fidelity extraction of text and vector graph-
ics from digitally born documents. Docling [3, 14],
which has accumulated over 52,000 GitHub stars
since its release in 2024, was selected for its
sophisticated layout analysis and table recognition
capabilities, as well as its direct integration with
multiple OCR engines. Marker [9], with approx-
imately 32,000 stars, employs the Surya engine
for unified layout analysis and text recognition,
including specialized handling of mathematical
formulas through LaTeX reconstruction.
For vision-language models, we selected
MiniCPM-V [45], which has demonstrated per-
formance equivalent to commercial models such
as GPT-4V. For LLM post-correction, we chose
Qwen3-8B [38] because it achieves performance
comparable to models with nearly twice its
8

parameter count, specifically matching Qwen2.5-
14B across several benchmarks, while remaining
deployable on consumer hardware through Ollama
and released under the Apache 2.0 license, ensur-
ing full reproducibility.
3.5 Evaluation Metrics
We employ edit distance-based metrics standard
in OCR evaluation literature:
Character Error Rate (CER):Levenshtein
distance at the character level, computed as
CER =Sc+Dc+Ic
Nc×100, whereS c,Dc,Icdenote
character-level substitutions, deletions, and inser-
tions, andN cis the total number of characters in
the ground truth.
Word Error Rate (WER):Levenshtein dis-
tance at the word level, computed as WER =
Sw+Dw+Iw
Nw×100, whereS w,Dw,Iwdenote word-
level substitutions, deletions, and insertions, and
Nwis the total number of words in the ground
truth.
CER Accuracy:Reported as 1−CER, repre-
senting the fraction of correctly recognized charac-
ters. Negative values occur when insertions exceed
ground truth length.
WER Accuracy:Reported as 1−WER, repre-
senting the fraction of correctly recognized words.
Following Levenshtein [21] and Berger et al.
[5], we use standard dynamic programming imple-
mentations for edit distance computation. Macro-
averages are computed across documents within
each type and overall across the full corpus. This
approach weights each document equally regard-
less of length, providing balanced performance
assessment across heterogeneous materials.
4 Results
4.1 Overall Performance
Table 1 presents macro-averaged accuracy across
all document types. Marker achieves the high-
est overall performance (98.70% CER accuracy;
97.71% WER accuracy), substantially outper-
forming all alternatives. Among conventional
OCR approaches, Docling + Tesseract yields the
strongest results (89.06% CER accuracy; 82.28%
WER accuracy), followed by Docling + EasyOCR
(85.39% CER; 78.07% WER).Table 1: CER and WER accuracy (%) across all
document types.
Pipeline CER WER
PyMuPDF* 84.05 79.43
Docling (No OCR)* 83.42 77.85
Docling + EasyOCR 85.39 78.07
Docling + Tesseract 89.06 82.28
Marker 98.70 97.71
PyMuPDF + LLM 55.95 53.82
Docling (No OCR) + LLM 77.08 73.29
Docling + EasyOCR + LLM 74.58 67.40
Docling + Tesseract + LLM 78.77 73.91
Marker + LLM 82.82 79.30
Chunk + MiniCPM-V 9.36 2.02
Chunk + Qwen3-VL 47.76 43.72
Chunk + EasyOCR 52.08 42.72
*These pipelines rely on embedded text and yield
0% accuracy on Type 5 documents; these values are
excluded from the average.
Embedded-text extraction approaches
(PyMuPDF, Docling No OCR) show moderate
overall performance but completely fail on doc-
uments without embedded text layers (Type 5),
yielding 0% accuracy. Their overall scores are
therefore artifacts of corpus composition rather
than true capability measures. When considering
only documents with embedded text (Types 1–4),
PyMuPDF achieves 84.05% CER accuracy and
79.43% WER accuracy, while Docling (No OCR)
achieves 83.42% CER and 77.85% WER.
LLM post-correction pipelines exhibit mixed
results. All LLM-augmented pipelines show
degraded overall performance relative to their base
OCR systems. For example, Docling + Tesseract
achieves 89.06% CER accuracy, but adding LLM
correction reduces this to 78.77%.
Chunk-based approaches with vision-language
models perform poorly. Chunk + MiniCPM-V
achieves only 9.36% CER accuracy with nega-
tive accuracy values on several document types,
indicating systematic hallucination and over-
insertion. Chunk + Qwen3-VL (47.76% CER) per-
forms better but remains far below conventional
OCR baselines.
4.2 Performance by Document Type
Table 2 presents the results for all thirteen
pipelines across document types, revealing how
production method and visual complexity affect
extraction quality.
9

Table 2: Average CER and WER accuracy (%) by document type for all evaluated pipelines.
Type 1 Type 2 Type 3 Type 4 Type 5
Low Quality Multi-Column Clean Scan Digital No Embed
PipelineCER WER CER WER CER WER CER WER CER WER
PyMuPDF 91.40 80.12 57.27 54.49 89.98 85.51 97.52 97.60 0.00 0.00
Docling (No OCR) 90.04 84.01 55.15 45.33 91.93 88.04 96.54 94.00 0.00 0.00
Docling + EasyOCR 92.90 83.91 82.39 69.51 91.97 88.07 96.54 94.00 63.13 54.85
Docling + Tesseract 91.54 82.13 84.66 72.84 95.41 87.78 96.54 94.00 77.17 74.64
Marker 97.79 96.26 98.13 96.62 99.56 98.00 98.32 98.05 99.70 99.59
PyMuPDF + LLM 77.39 66.51 21.36 18.34 61.36 62.74 63.70 67.67 0.00 0.00
Docling (No OCR) + LLM 86.74 81.38 50.33 46.88 83.94 81.80 87.32 83.10 0.00 0.00
Docling + EasyOCR + LLM 86.18 79.97 78.89 72.32 72.56 66.19 85.53 81.64 49.74 36.86
Docling + Tesseract + LLM 78.57 72.49 73.42 66.72 84.82 82.29 89.05 84.23 67.97 63.80
Marker + LLM 90.56 87.44 90.31 87.64 51.66 45.35 89.49 84.67 92.10 91.40
Chunk + MiniCPM-V−153.60−155.88 32.91 23.77 36.18 26.07 70.19 63.74 61.13 52.37
Chunk + Qwen3-VL 50.41 43.35 17.37 14.51 39.22 36.53 77.67 75.88 54.10 48.33
Chunk + EasyOCR 41.06 25.57 34.18 26.68 42.70 36.63 76.81 70.19 65.66 54.55
Type 1 (Low-Quality Scans with Anno-
tations):Marker achieves 97.79% CER and
96.26% WER accuracy despite severe degradation
and annotations. Conventional OCR approaches
perform reasonably (Docling + EasyOCR: 92.90%
CER accuracy; Docling + Tesseract: 91.54%
CER accuracy), and direct extraction methods
achieve similar character accuracy (PyMuPDF:
91.40% CER accuracy) due to the presence of
embedded OCR layers in these legacy scans.
LLM post-correction degrades results across all
base pipelines, and chunk-based approaches per-
form poorly, with Chunk + MiniCPM-V yield-
ing strongly negative accuracy (−153.60% CER
accuracy), indicating massive hallucination and
over-insertion of text.
Type 2 (Multi-Column Complex Lay-
outs):Layout complexity substantially degrades
performance. Direct approaches fail significantly
(PyMuPDF: 57.27% CER accuracy; Docling No
OCR: 55.15% CER accuracy) due to incorrect
reading order. Forcing OCR through Docling
improves results considerably (Docling + Tesser-
act: 84.66% CER accuracy; Docling + Easy-
OCR: 82.39% CER accuracy). Marker main-
tains robust performance (98.13% CER accuracy;
96.62% WER accuracy), demonstrating superior
layout understanding. LLM post-correction pro-
duces mixed effects in this category: it severely
degrades PyMuPDF + LLM (21.36% CER accu-
racy), while Docling + EasyOCR + LLM (78.89%
CER accuracy) performs comparably to its base.Type 3 (Clean Scans):On high-
quality single-column scans, most conventional
approaches perform well. Marker achieves near-
perfect accuracy (99.56% CER accuracy; 98%
WER accuracy). Conventional OCR reaches
strong levels (Docling + Tesseract: 95.41% CER
accuracy; Docling + EasyOCR: 91.97% CER
accuracy), and direct extraction performs ade-
quately when OCR layers are present (Docling No
OCR: 91.93% CER accuracy; PyMuPDF: 89.98%
CER accuracy). However, LLM post-correction
again degrades performance substantially in
most cases, with PyMuPDF + LLM dropping to
61.36% CER accuracy.
Type 4 (Digital PDFs):All methods per-
form well on digital documents with embedded
text. Marker achieves 98.32% CER accuracy and
98.05% WER accuracy. The other direct extrac-
tion approaches reach their best performance
(PyMuPDF: 97.52% CER accuracy; Docling No
OCR: 96.54% CER accuracy), while forced OCR
pipelines perform identically to the no-OCR base-
line (Docling + Tesseract: 96.54% CER accuracy;
Docling + EasyOCR: 96.54% CER accuracy), sug-
gesting that the embedded text is of sufficient
quality and forcing OCR neither helps nor hurts.
LLM post-correction degrades all results substan-
tially, with PyMuPDF + LLM falling to 63.70%
CER accuracy.
Type 5 (No Embedded Text):Scans with-
out embedded text expose fundamental limita-
tions of extraction-based approaches. PyMuPDF
10

(a) Original text
(b) Extracted text
Fig. 6: Word segmentation failures.
and Docling No OCR achieve 0% accuracy,
producing empty outputs. Their LLM-enhanced
counterparts also yield 0%, as there is no text
to correct. OCR-based methods show degraded
but functional performance (Docling + Tesser-
act: 77.17% CER accuracy; Docling + EasyOCR:
63.13% CER accuracy). Chunk-based approaches
achieve moderate results in this category (Chunk
+ EasyOCR: 65.66% CER accuracy; Chunk +
MiniCPM-V: 61.13% CER accuracy), suggest-
ing that for documents where no embedded text
exists, even these lower-performing approaches
provide some value. Marker maintains exceptional
accuracy (99.70% CER accuracy; 99.59% WER
accuracy).
4.3 Error Pattern Analysis
Qualitative analysis reveals characteristic error
patterns for each pipeline family.
Word Segmentation Errors:Embedded-text
extraction and basic OCR frequently introduce
spurious spaces within words or merge distinct
words. Figure 6 illustrates “aclamaci´ on” incor-
rectly segmented as “aclama ci´ on”, while words
such as “Concilio” and “abri´ o” are also erro-
neously split, introducing noise. These errors
severely degrade word-level accuracy despite mod-
erate character-level accuracy.
Character Substitutions:Visually similar
characters are frequently confused:t/l,I/1,O/0,
a/u. Roman numerals prove particularly prob-
lematic, with “II” consistently misrecognized as
“11”. Additionally, highly distorted character
sequences appear, such as “destruy´ endolo” becom-
ing “dcslruy´ eiiidolo”, and symbols likesbeing
(a) Original text
(b) Extracted text
Fig. 7: Roman numeral misrecognition.
misrecognized as§. Figure 7 shows the Liuva II/11
error motivating this study.
Layout Order Corruption:Multi-column doc-
uments exhibit reading order errors where text
from adjacent columns is interleaved. Embedded-
text approaches particularly struggle, often read-
ing left-to-right across column boundaries rather
than top-to-bottom within columns. When the
system does not correctly detect the column struc-
ture, it may attempt to read the page in a single,
linear sequence, ignoring the intended reading
order.
Symbol Misinterpretation:Specialized sym-
bols (†,‡,§) are frequently misrecognized as let-
ters. The dagger symbol (†) indicating death dates
consistently becomes “t”, eliminating semantic
information. Punctuation errors also occur, with
commas becoming dots or hyphens.
Visual Noise:Images, stains, and artifacts gen-
erate spurious character sequences. Poor-quality
regions can produce strings of random charac-
ters as OCR systems attempt to interpret visual
noise as text. In image regions, the recognizer may
attempt to interpret visual patterns as characters,
resulting in the insertion of unexpected symbols
into the output.
Footnote Errors:Footnotes are particularly dif-
ficult to extract accurately and often contain the
highest concentration of errors. Smaller font sizes
make characters harder to distinguish at the pixel
level. Superscript numbers preceding footnotes are
often mistaken for other numbers, substituted for
other symbols, or omitted entirely.
LLM Post-Correction Errors:Post-correction
with LLMs introduces distinct error modes. The
model sometimes hallucinates errors absent in
the source text, for example, identifying a sup-
posed repetition of a preposition where the actual
error is a typographical artifact. Instead of return-
ing corrected text, the model sometimes produces
11

explanations despite explicit instructions. Some
paragraphs with heavy OCR corruption prove
too complex for the model to correct accurately.
Words not well represented in the vocabulary of
the model are left uncorrected (e.g., “v´ andulos”
remaining instead of “v´ andalos”). A significant
issue arises with Latin words closely resembling
Spanish ones: the model incorrectly normalizes
correct Latin forms into Spanish (e.g., “posses-
sores” becoming “poseedores”), introducing errors
rather than correcting them.
Vision Model Errors:Vision models used in
chunk-based extraction exhibit additional failure
modes. They sometimes add extra text not present
in the original block (e.g., prefixing output with
labels such as “TEXTO DE LA IMAGEN:”).
Blocks consisting solely of noise are sometimes
transcribed as characters rather than identified as
non-text. Spelling errors from conventional OCR
persist, and the chunking process itself introduces
ordering difficulties, particularly for multi-column
layouts.
5 Discussion
5.1 RQ1: Pipeline Performance and
Error Patterns
Our results demonstrate clear performance dif-
ferentiation across the pipeline families analyzed.
End-to-end document parsing (Marker) substan-
tially outperforms all alternatives, achieving near-
optimal accuracy even on degraded and com-
plex documents. This suggests that joint train-
ing of layout detection and text recognition
on document-specific corpora yields qualitative
improvements over modular pipelines combining
generic computer vision and OCR components.
Conventional OCR systems (Tesseract, Easy-
OCR) show moderate performance with charac-
teristic failure modes: word segmentation errors,
character substitutions, and layout corruption on
multi-column documents. Performance degrades
predictably with scan quality and layout com-
plexity. Embedded-text extraction (PyMuPDF,
Docling No OCR) succeeds only when high-quality
text layers exist, completely failing on true scans.
Vision-language models show disappointing
performance despite theoretical advantages. Their
tendency toward hallucination and verbose out-
puts (including explanatory text rather than puretranscription) makes them unsuitable for pro-
duction digitization workflows. The Chunk +
MiniCPM-V pipeline produced negative accuracy
values on Type 1 documents (−153.60% CER),
meaning it introduced more erroneous characters
than the total length of the reference text. Even
the best vision model approach, Chunk + Qwen3-
VL (47.76% CER overall), remains far below
conventional OCR baselines. The poor results may
reflect misalignment between vision model train-
ing objectives (general scene understanding, visual
question answering) and the precise character-
level accuracy required for OCR.
5.2 RQ2: Document Heterogeneity
Document stratification reveals systematic perfor-
mance variation across production methods and
visual complexity. Clean digital PDFs (Type 4)
and high-quality scans (Type 3) present mini-
mal challenges for most pipelines. Multi-column
layouts (Type 2) and degraded scans with anno-
tations (Type 1) expose fundamental limitations
of conventional approaches.
The Type 5 results (no embedded text) are
particularly instructive. The 0% accuracy of
embedded-text approaches underscores a critical
limitation: tools like PyMuPDF and basic Docling
configurations silently fail on true scans, pro-
ducing empty outputs or fragmentary text. This
presents serious risks for automated workflows
where silent failures may go undetected.
The consistent performance of Marker across
all document types (>97% CER accuracy across
all categories) suggests that its architectural
choices specifically address document heterogene-
ity. The combination of EfficientViT-based lay-
out detection and Donut-based text recognition,
trained end-to-end on diverse documents, appears
to generalize effectively across production meth-
ods and quality levels.
An interesting observation in the Type 4 (Digi-
tal PDF) results is that Docling produces identical
results regardless of whether OCR is enabled or
not (96.54% CER for all three Docling configura-
tions). This confirms that for digitally-born doc-
uments with clean embedded text, the OCR step
is redundant. By contrast, forcing OCR makes a
substantial difference on Type 2 (multi-column)
and Type 5 (no embedded text) documents, where
it can improve or enable extraction entirely.
12

5.3 RQ3: LLM Post-Correction
Our results provide evidence that LLM-based
post-correction does not systematically improve
OCR quality when evaluated using edit-distance
metrics. All LLM-augmented pipelines show
degraded overall performance relative to their base
extraction systems. This finding aligns with Kan-
erva et al. [16], who caution against assumptions of
universal LLM benefit for OCR correction. Several
factors explain this degradation:
•Metric Alignment:Edit-distance metrics
penalize any deviation from ground truth,
including LLM “corrections” of already-
accurate text. LLMs trained on modern corpora
may normalize archaic spellings, adjust punc-
tuation to contemporary conventions, or “fix”
correct but unusual phrasings—all counted as
errors under strict edit distance.
•Hallucination:Despite carefully engineered
prompts, LLMs exhibit tendency toward hallu-
cination, inventing errors that do not exist in
input text and adding explanatory commentary
rather than pure corrections.
•Non-Determinism:LLM outputs vary across
runs due to sampling, introducing inconsistency
unsuitable for reliable digitization pipelines.
•Vocabulary Coverage:Historical proper
nouns, archaic terms, and Latin phrases are
poorly represented in LLM training corpora,
leading to incorrect “normalizations.” The
observed case where correct Latin “possessores”
was changed to Spanish “poseedores” exempli-
fies this problem.
•Complex Corruption:Some OCR-corrupted
paragraphs are so severely damaged that the
model fails to reconstruct the original meaning,
producing only partial corrections while leaving
fundamental errors intact.
These findings suggest that LLM post-
correction may benefit from alternative evaluation
frameworks emphasising semantic preservation
and readability over strict character-level fidelity.
For applications prioritising human readability
(e.g., publicly accessible digital files), LLM nor-
malisation might improve user experience despite
degrading edit-distance scores. Conversely, for
scholarly applications requiring faithful reproduc-
tion of original text, LLM correction appears
counterproductive.Our findings align with recent literature on
LLM-based OCR correction. Koynov [19] docu-
mented similar challenges including hallucination
and vocabulary coverage gaps, while Kanerva
et al. [16] demonstrated that LLM effectiveness
varies significantly across languages and base
OCR quality levels.
5.4 Limitations
This study acknowledges some limitations. First,
the corpus is limited to Spanish historiographical
texts on a specific topic. Generalisation to other
languages, domains, and time periods requires
validation. Second, manual ground truth con-
struction is labour-intensive, limiting corpus size.
Larger-scale evaluation would strengthen find-
ings. Third, we employ edit-distance metrics stan-
dard in OCR research but potentially misaligned
with downstream application requirements (e.g.,
semantic search, fact extraction). Fourth, LLM
experiments used a single model (Qwen3 8B) with
specific prompting strategies; alternative mod-
els and prompts might yield different results.
Fifth, we do not evaluate commercial OCR ser-
vices (Google Cloud Vision, AWS Textract, Azure
Computer Vision) due to cost and reproducibility
constraints, though these may outperform open-
source alternatives.
6 Conclusion
This study presents a systematic evaluation of
PDF-to-text extraction pipelines for historio-
graphical documents, revealing substantial perfor-
mance variation across approaches and document
types. End-to-end document parsing achieves
superior and stable performance (Marker: 98.70%
CER accuracy overall), while conventional OCR
degrades on complex layouts and degraded scans.
Embedded-text extraction succeeds only on dig-
ital documents and silently fails on true scans.
LLM-based post-correction does not systemati-
cally improve quality under edit-distance evalua-
tion and frequently degrades accurate extractions.
These findings have direct implications for
digital humanities practice and AI knowledge sys-
tems. As conversational AI and RAG architectures
scale to incorporate digitised historical collections,
OCR quality emerges as a critical bottleneck. Sys-
tematic errors at the digitisation stage propagate
13

through retrieval and generation pipelines, pro-
ducing outputs that appear confident but contain
errors invisible to end users.
Our stratified analysis provides actionable
guidance for practitioners. End-to-end parsing
proved to be the most reliable approach across
all document types and should be the preferred
choice when working with heterogeneous historical
collections. Since document characteristics such
as scan quality, layout complexity, and the pres-
ence of embedded text layers have a significant
impact on extraction accuracy, workflows should
be adapted accordingly rather than applying a sin-
gle pipeline uniformly. Additionally, LLM-based
post-correction should not be assumed to be ben-
eficial by default—our results show that it can
introduce new errors, so it should be validated
on representative samples before being applied at
scale. Improving digitisation quality is not only
a technical challenge but also a necessary step
to ensure that historical knowledge remains accu-
rate and trustworthy as it becomes increasingly
accessed through AI systems.
Statements and Declarations
Funding:This work has been supported
by the Madrid Government (Comunidad de
Madrid-Spain) under the Multiannual Agree-
ment with UC3M (BADDO-CM-UC3M), by
the “Generation of Reliable Synthetic Health
Data for Federated Learning in Secure Data
Spaces” Research Project (Agencia Estatal
de Investigaci´ on (AEI)/European Regional
Development Fund (ERDF), EU) funded by
Ministerio de Ciencia, Innovaci´ on y Univer-
sidades (MCIN)/AEI/10.13039/501100011033
under Grant PID2022-141045OB-C43, as
well as by the GENIE Learn project under
Grant PID2023-146692OB-C31 funded by
MICIU/AEI/10.13039/501100011033 and by
ERDF/UE.
References
[1] Arias PP (2020) Hacia la unidad de Hispania.
Explicaciones sociales a las ofensivas militares
visigodas en la Pen´ ınsula Ib´ erica (siglos VI-
VIII). Gladius 40:73–92[2] Artifex Software (2024) PyMuPDF: Python
binding for MuPDF. https://github.com/
pymupdf/PyMuPDF, accessed: 2026-02-16
[3] Auer C, Lysak M, Nassar A, et al (2024)
Docling technical report. arXiv preprint
arXiv:240809869
[4] Baek Y, Lee B, Han D, et al (2019) Char-
acter region awareness for text detection. In:
Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pp
9365–9374
[5] Berger B, Waterman MS, Yu YW (2020) Lev-
enshtein distance, sequence comparison and
biological database search. IEEE transactions
on information theory 67(6):3287–3294
[6] Breuel TM (2003) High performance docu-
ment layout analysis. In: Proceedings of the
Symposium on Document Image Understand-
ing Technology
[7] Cabrerizo OD (2019) El di´ alogo entre el dis-
curso y la realidad de la Gens Gothorvm
en los siglos V-VII. Universitat de Barcelona
(Spain)
[8] Cuconasu F, Trappolini G, Siciliano F,
et al (2024) The power of noise: Redefin-
ing retrieval for RAG systems. In: Proceed-
ings of the 47th International ACM SIGIR
Conference on Research and Development in
Information Retrieval, pp 719–729
[9] Datalab (2024) Marker: Convert PDF to
markdown + JSON quickly with high accu-
racy. https://github.com/datalab-to/marker,
accessed: 2026-02-16
[10] D´ avila AMGC (1991) Las clases sociales en la
sociedad visig´ otica y el III Concilio de Toledo.
In: Concilio III de Toledo: XIV Centenario.
589-1989, Arzobispado de Toledo, pp 411–426
[11] Fleischhacker D, Kern R, G¨ oderle W (2025)
Enhancing ocr in historical documents with
complex layouts through machine learning.
International Journal on Digital Libraries
26(1):3
14

[12] Google (2024) Tesseract OCR. https:
//github.com/tesseract-ocr/tesseract,
accessed: 2026-02-16
[13] Guzm´ an AA (1989) De Recaredo a Don
Rodrigo. Historia 16 (163):55–62
[14] IBM Research (2024) Docling: Document
processing framework. https://github.com/
docling-project, accessed: 2026-02-16
[15] JaidedAI (2022) EasyOCR. https://github.
com/JaidedAI/EasyOCR, accessed: 2026-02-
16
[16] Kanerva J, Ledins C, K¨ apyaho S, et al (2025)
Ocr error post-correction with llms in his-
torical documents: No free lunches. arXiv
preprint arXiv:250201205
[17] Khan A, Rai U, Singh SS, et al (2024)
Ocr approaches for humanities: Applications
of artificial intelligence/machine learning on
transcription and transliteration of historical
documents. Digital Studies in Language and
Literature 1(1-2):85–112
[18] Kim G, Hong T, Yim M, et al (2022) Ocr-
free document understanding transformer. In:
European Conference on Computer Vision,
Springer, pp 498–517
[19] Koynov R, Doan THA (2025) Opportunities
and challenges of llms as post-ocr correctors.
pp 111–118
[20] Levchenko MA (2025) Evaluating llms for
historical document ocr: A methodological
framework for digital humanities. In: Pro-
ceedings of the First on Natural Language
Processing and Language Models for Digital
Humanities, pp 75–85
[21] Levenshtein VI (1966) Binary codes capable
of correcting deletions, insertions and rever-
sals. Soviet Physics Doklady 10(8):707–710
[22] Lewis P, Perez E, Piktus A, et al
(2020) Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances
in neural information processing systems
33:9459–9474[23] Liu NF, Lin K, Hewitt J, et al (2024) Lost
in the middle: How language models use long
contexts. Transactions of the Association for
Computational Linguistics 12:157–173
[24] Livathinos N, Auer C, Lysak M, et al (2025)
Docling: An efficient open-source toolkit
for ai-driven document conversion. arXiv
preprint arXiv:250117887
[25] L´ opez GR (1987) La Hispania de Recaredo:
hacia la unidad peninsular. Historia 16
(131):37–42
[26] Mart´ ınek J, Lenc L, Kr´ al P (2020) Building
an efficient ocr system for historical docu-
ments with little training data. Neural Com-
puting and Applications 32(23):17209–17227
[27] Masana JV (1991) Hispania durante la ´ epoca
del III Concilio de Toledo seg´ un Gregorio
Magno. In: Concilio III de Toledo: XIV Cen-
tenario. 589-1989, Arzobispado de Toledo, pp
485–496
[28] McKie R, Liu J (2024) PyMuPDF:
High-performance rendering and data
extraction from PDF and other docu-
ment formats. Artifex Software, Inc., URL
https://pymupdf.readthedocs.io/en/latest/
app1.html, technical Documentation
[29] Moreno LAG (1993) Dos cap´ ıtulos sobre
administraci´ on y fiscalidad del Reino de
Toledo. In: De la antig¨ uedad al medievo: sig-
los IV-VIII, Fundaci´ on S´ anchez-Albornoz, pp
291–314
[30] Moreno LAG (2014) La organizaci´ on territo-
rial de la Iglesia hispanogoda. In: La Iglesia
en la Historia de Espa˜ na, Fundaci´ on Rafael
del Pino, pp 169–184
[31] OpenBMB (2024) MiniCPM-o: Multimodal
large language model. https://github.com/
OpenBMB/MiniCPM-o, accessed: 2026-02-
16
[32] Orlandis J (1962) El poder real y la sucesi´ on
al trono en la monarqu´ ıa visigoda. Consejo
Superior de Investigaciones Cient´ ıficas, Dele-
gaci´ on de Roma
15

[33] Orlandis J (1989) Cr´ onica del III Concilio de
Toledo
[34] Orlandis J (2000) La doble conversi´ on reli-
giosa de los pueblos germ´ anicos (siglos IV
al VIII). Anuario de Historia de la Iglesia
9:69–84
[35] Otsu N, et al (1975) A threshold selection
method from gray-level histograms. Auto-
matica 11(285-296):23–27
[36] Pem´ an JM (1950) Historia de Espa˜ na con-
tada con sencillez
[37] PyPI Stats (2025) PyMuPDF download
statistics. URL https://pypistats.org/
packages/pymupdf, accessed: 2026-02-16
[38] Qwen Team (2025) Qwen3 technical report.
https://qwen.ai/blog?id=qwen3, accessed:
2026-02-16
[39] Rovira JO (1992) Semblanzas visigodas. Edi-
ciones Rialp
[40] Salinero RG (2023) Recaredo y el Concilio
III de Toledo (589). In: Hispania visigoda. El
tiempo de los b´ arbaros: de su asentamiento
en la Pen´ ınsula Ib´ erica a la conquista isl´ amica
de los Omeyas, Pinolia, pp 73–85
[41] Sastre I, Etcheverry L, Rey G, et al (2025)
Post-ocr correction using large language mod-
els with constrained decoding
[42] Smith R (2007) An overview of the tesseract
ocr engine. In: Ninth International Confer-
ence on Document Analysis and Recognition
(ICDAR 2007), pp 629–633, https://doi.org/
10.1109/ICDAR.2007.4376991
[43] Smith RW (2013) History of the tesseract
ocr engine: what worked and what didn’t.
In: Document Recognition and Retrieval XX,
SPIE, p 865802
[44] Sulaiman A, Omar K, Nasrudin MF (2019)
Degraded historical document binarization:
A review on issues, challenges, techniques,
and future directions. Journal of imaging
5(4):48[45] Yao Y, Yu T, Zhang A, et al (2024) Minicpm-
v: A gpt-4v level mllm on your phone. arXiv
preprint arXiv:240801800
[46] Zhang J, Zhang Q, Wang B, et al (2025) Ocr
hinders rag: Evaluating the cascading impact
of ocr on retrieval-augmented generation. In:
Proceedings of the IEEE/CVF International
Conference on Computer Vision, pp 17443–
17453
[47] Zhang Q, Wang B, Huang VSJ, et al
(2024) Document parsing unveiled: Tech-
niques, challenges, and prospects for struc-
tured information extraction. arXiv preprint
arXiv:241021169
16