# SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation

**Authors**: Yuyang Dong, Nobuhiro Ueda, Krisztián Boros, Daiki Ito, Takuya Sera, Masafumi Oyamada

**Published**: 2025-05-20 14:03:24

**PDF URL**: [http://arxiv.org/pdf/2505.14381v1](http://arxiv.org/pdf/2505.14381v1)

## Abstract
With the increasing adoption of Large Language Models (LLMs) and
Vision-Language Models (VLMs), rich document analysis technologies for
applications like Retrieval-Augmented Generation (RAG) and visual RAG are
gaining significant attention. Recent research indicates that using VLMs can
achieve better RAG performance, but processing rich documents still remains a
challenge since a single page contains large amounts of information. In this
paper, we present SCAN (\textbf{S}emanti\textbf{C} Document Layout
\textbf{AN}alysis), a novel approach enhancing both textual and visual
Retrieval-Augmented Generation (RAG) systems working with visually rich
documents. It is a VLM-friendly approach that identifies document components
with appropriate semantic granularity, balancing context preservation with
processing efficiency. SCAN uses a coarse-grained semantic approach that
divides documents into coherent regions covering continuous components. We
trained the SCAN model by fine-tuning object detection models with
sophisticated annotation datasets. Our experimental results across English and
Japanese datasets demonstrate that applying SCAN improves end-to-end textual
RAG performance by up to 9.0\% and visual RAG performance by up to 6.4\%,
outperforming conventional approaches and even commercial document processing
solutions.

## Full Text


<!-- PDF content starts -->

arXiv:2505.14381v1  [cs.AI]  20 May 2025SCAN: Semantic Document Layout Analysis for Textual and Visual
Retrieval-Augmented Generation
Yuyang Dong*, Nobuhiro Ueda*, Krisztián Boros, Daiki Ito, Takuya Sera, Masafumi Oyamada
NEC Corporation
{dongyuyang, nobuhiro-ueda, krisztian-boros, ito-daiki, takuya-sera, oyamada}@nec.com
Abstract
With the increasing adoption of Large Lan-
guage Models (LLMs) and Vision-Language
Models (VLMs), rich document analysis
technologies for applications like Retrieval-
Augmented Generation (RAG) and visual RAG
are gaining significant attention. Recent re-
search indicates that using VLMs can achieve
better RAG performance, but processing rich
documents still remains a challenge since a sin-
gle page contains large amounts of information.
In this paper, we present SCAN ( Semanti C
Document Layout ANalysis), a novel approach
enhancing both textual and visual Retrieval-
Augmented Generation (RAG) systems work-
ing with visually rich documents. It is a VLM-
friendly approach that identifies document com-
ponents with appropriate semantic granularity,
balancing context preservation with processing
efficiency. SCAN uses a coarse-grained seman-
tic approach that divides documents into coher-
ent regions covering continuous components.
We trained the SCAN model by fine-tuning ob-
ject detection models with sophisticated anno-
tation datasets. Our experimental results across
English and Japanese datasets demonstrate that
applying SCAN improves end-to-end textual
RAG performance by up to 9.0% and visual
RAG performance by up to 6.4%, outperform-
ing conventional approaches and even commer-
cial document processing solutions.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2021; Gao et al., 2024; Fan et al., 2024) tech-
nology enables Large Language Models (LLMs)
to provide more accurate responses to user queries
by retrieving and leveraging relevant knowledge
and documents. These knowledge sources such as
company financial reports, web pages, insurance
manuals, and academic papers often contain com-
plex charts, tables, diagrams, and other non-textual
*Equal contributionelements, collectively referred to as rich documents.
Enabling RAG to accurately support these rich doc-
uments is an important research problem.
RAG frameworks are generally categorized into
textual RAG and visual RAG. For textual RAG
with rich documents, we need to convert documents
(PDFs, images, etc.) into text, use a text retrieval,
and then generate responses by LLMs. In contrast,
visual RAG can directly process rich documents as
images using multimodal embedding and retrieval,
and then uses Vision Language Models (VLMs) to
generate responses.
1.1 Challenges
The key part for textual RAG with rich documents
is to convert documents into textual formats such
as markdown. Recent research (Zhang et al., 2025;
Fu et al., 2024) indicates that using VLMs for doc-
ument conversion yields better results than tradi-
tional OCR methods. However, it is still a chal-
lenge to have a VLM perform text conversion on
rich document pages , since a whole page can con-
tain a large amount of complicated content, which
can lead to incomplete results and conversion er-
rors.
On the other hand, visual RAG directly retrieves
the images of rich document pages, but faces a
similar challenge when using a VLM to process
visual question answering (VQA) on those rich
document pages . This is because user queries
often target only specific portions of a document
page, meaning that other parts of this page are
noise, leading to low accuracy.
Therefore, the common challenge is to use a
VLM to process (text conversion or VQA) an
entire rich document page containing abundant
information at once. One potential solution is
to further divide a document page into small re-
gions. Traditional document layout analysis tech-
nologies such as DocLayout-YOLO (Zhao et al.,
2024b) can achieve this objective, but they focus

Figure 1: Conventional fine-grained layout analysis result (Left, DocLayout-YOLO) vs. Our coarse-grained
semantic layout analysis result (Right, SCAN).
Figure 2: Overview of applying our SCAN model to current textual and visual RAG systems.
on fine-grained analysis, breaking down content
into small components such as titles, paragraphs,
tables, figures, and captions. This approach could
lose important context when processing isolated
components and potentially lead to reduced RAG
accuracy. In our experiments, conventional layout
analysis methods with VLM text conversion and
VQA reduce accuracy by 5.8%–8.7% for textual
RAG and 28.7%–40.5% for visual RAG, respec-
tively.
1.2 Contributions
To address these challenges, we propose SCAN, a
novel approach that performs VLM-friendly seman-
tic document layout analysis with “coarse granular-
ity.” Figure 1 compares the layout analysis results
between SCAN and a conventional fine-grained
method. SCAN can semantically divide regions
into boxes that cover continuous components re-
lated to the same topic. For example, each of the
semantic_box [3], [4], and [5] corresponds to inde-
pendent topics of IT Services ,Social Infrastructure ,
andOthers .To train a powerful SCAN model, we annotated
more than 24k document pages with semantic lay-
out labels. SCAN model is fine-tuned from pre-
trained object detection models with this training
data. We also designed a specific box selection
strategy and post-processing techniques for RAG
applications. Figure 2 gives an overview of apply-
ing our SCAN model into both textual and visual
RAG pipelines. Concretely, each page of an input
document is treated as an image and decomposed
into semantic chunks by our SCAN model. For
textual RAG, the images of semantic chunks are
passed to a VLM that performs text conversion, and
then input into existing textual RAG systems. For
visual RAG, the resulting image chunks can be the
retrieval targets that are directly input to existing
visual RAG systems.
We test SCAN’s performance using three
datasets featuring both English and Japanese docu-
ments. Although SCAN is trained on Japanese data,
the experiments show that it can achieve good per-
formance on English benchmarks. Experimental re-
sults show that in textual RAG, applying SCAN can

improve end-to-end performance by 1.7%–9.0%,
while in visual RAG, SCAN can enhance end-to-
end performance by 3.5%–6.4%. Moreover, al-
though our SCAN requires multiple processes on
VLMs rather than a single page-level processing,
the total computational time is roughly the same
while the total token usage is reduced.
2 Related Work
2.1 Document Layout Analysis
CNN backbones such as DocLayout-YOLO (Zhao
et al., 2024b) and transformer backbones such
as DiT (Li et al., 2022), LayoutLMv3 (Huang
et al., 2022), LayoutDETR (Yu et al., 2023), Bee-
hive (Auer et al., 2024) have been proposed for
high-performance document layout analysis which
are trained with synthetic and human-annotated
datasets Zhong et al. (2019); Li et al. (2020); Pfitz-
mann et al. (2022). End-to-end document conver-
sion systems have been developed using the above
models. Docling (Auer et al., 2024), Marker (Data-
lab, 2024), and MinerU (Wang et al., 2024) provide
a comprehensive pipeline for document layout anal-
ysis and text conversions. There are also production
systems such as Azure Document Intelligence (mi-
crosoft, 2025) and LlamaParse Premium (Llama-
Parse.AI, 2025).
2.2 VLMs for Document Conversion
Vision-Language Models (VLMs) have emerged
as powerful tools for multimodal understanding
tasks. Open models such as the Qwen-VL se-
ries (Bai et al., 2025), or InternVL (Chen et al.,
2024b) have also demonstrated impressive capa-
bilities in visual document understanding. There
are also small-sized OCR-specialized VLMs such
as GOT (Wei et al., 2024), Nougat (Blecher
et al., 2023), DocVLM (Nacson et al., 2024), and
olmOCR (Poznanski et al., 2025).
2.3 Textual and visual RAG
The rapid progress of LLMs has further strength-
ened RAG, enabling models to inject external
knowledge into responses with higher precision
and coverage (Lewis et al., 2021; Gao et al., 2024;
Fan et al., 2024). Typical RAG pipelines in pre-
vious works first converted document images or
PDFs into plain text, and only then performed in-
dexing and retrieval over the extracted passages
(Zhang et al., 2025). Recent results show that using
VLMs for document text conversion yields betterresults than traditional OCR tools (Zhang et al.,
2025; Fu et al., 2024). On the other hand, with the
increasing availability of multimodal embedding
and VLMs, there is a growing interest in multi-
modal RAG systems that can directly index and re-
trieve images, and can leverage VLMs for answer
generation (Yu et al., 2025; Tanaka et al., 2025;
Faysse et al., 2024).
3 Method
3.1 SCAN Model Training
Given a single page image from a rich document,
the semantic document layout analysis problem is
to identify and delineate different regions of the
image using bounding boxes, each associated with
a specific semantic type. This formulation follows
classical layout analysis approaches and can be
viewed as a specialized object detection task. Ac-
cordingly, we fine-tune pre-trained object detection
models on a dataset with semantic layout labels,
leveraging their robust feature extraction capabil-
ities while adapting them to the requirements of
document layout analysis.
3.1.1 Training Data
We define the semantic boxes are regions that con-
tain content unified by a single topic or meaning.
Unlike structural divisions (e.g., paragraphs or sec-
tions), semantic boxes are defined by their coher-
ent meaning. In other words, all content within
a box relates to the same subject and can be un-
derstood independently, without needing context
from surrounding boxes. A semantic box might
contain various structural elements, multiple para-
graphs, sections, tables, or charts, as long as they
all contribute to the same semantic topic. Besides
semantic boxes, we also define global boxes, which
correspond to the title, header, footer, date, and au-
thor of the document. The global boxes are seman-
tically related to all the elements in the page, while
the semantic boxes are more local and can be inde-
pendent of each other. Figure 1 (right) shows four
semantic boxes and three global boxes. Introducing
global boxes as well as semantic boxes allows us to
represent the page’s semantic dependency structure
more accurately.
As we are the first to define semantic document
layout analysis in this context, we developed our
training and evaluation dataset through a rigorous
annotation process. This labor-intensive but crucial
step was fundamental to achieving a high-quality

model. We collaborated with a specialized data
annotation company in Tokyo, engaging six expert
Japanese annotators, in a reasonable payment and
contract. We asked them to locate the semantic
boxes and global boxes in the document images and
label them with the corresponding types. The box
types we defined are: semantic_box ,title,header ,
footer ,date, and author .
Source documents for annotation were collected
from Common Crawl PDFs (Turski et al., 2023).
To ensure diversity in our training data, we first
performed clustering on the CC-PDF corpus us-
ing image embeddings with MiniCPM-Visual-
Embedding (Rhapsody Group, 2024) model. We
then selected a balanced number of candidates from
each cluster for annotation, creating a representa-
tive dataset that encompasses a wide range of doc-
ument styles, formats, and domain-specific layouts.
We finally labeled 24,577 images and used 22,577,
1,000, and 1,000 samples for training, validation,
and test sets. Appendix A provides examples of
our dataset.
3.1.2 Object Detection Model Finetuning
There are two popular families of object detection
architectures: the CNN-based (YOLO series (Red-
mon et al., 2016; Khanam and Hussain, 2024)) and
the transformer-based (DETR series (Carion et al.,
2020; Zhao et al., 2024a)). We tried to fine-tune
YOLO11-X (56.9M parameters)1with 5 and RT-
DETR-X (67.4M parameters)2models with our
semantic layout annotation data supported on the
Ultralytics framework (under New AGPL-3.0 Li-
cense) to develop our SCAN models.
To evaluate the fine-tuned models in terms of
bounding box granularity and precision, we de-
veloped a matching-based IoU (Intersection over
Union) metric. This metric first uses the Hungarian
algorithm to perform bipartite matching between
predicted and ground-truth bounding boxes. And
then, it calculates an average IoU over the matched
bounding box pairs and unmatched bounding boxes.
For unmatched bounding boxes, we assigned an
IoU of 0 to penalize excessive or insufficient predic-
tions. Table 1 shows the two models’ IoU scores on
the validation set of our dataset. Because each pre-
dicted bounding box has its confidence score, we
varied the confidence threshold to select the optimal
predicted bounding box set. With the optimal confi-
dence threshold, the performance of the YOLO11-
1https://docs.ultralytics.com/models/yolo11/
2https://docs.ultralytics.com/models/rtdetr/Model Conf. threshold IoU
YOLO11-X0.2 53.7
0.3 58.0
0.4 55.8
0.5 49.7
RT-DETR-X0.2 48.9
0.3 54.8
0.4 58.4
0.5 59.6
Table 1: The Intersection over Union (IoU) scores of
fine-tuned object detection models. We varied the confi-
dence threshold for predicted bounding boxes from 0.2
to 0.5.
X and RT-DETR-X finetuned models is similar,
while the RT-DETR-X-based model is slightly bet-
ter. We primarily use RT-DETR-X hereafter.
3.2 Predicted Box Selection
Object detection models typically output many
bounding boxes along with their corresponding
confidence scores. We then select some of the
bounding boxes for the final prediction. The sim-
plest selection approach would be to establish a
fixed confidence threshold. However, this strategy
proves suboptimal for document layout analysis
due to document-specific strict constraints:
•Predicted bounding boxes should cover all
textual and visual elements to minimize infor-
mation loss
•Predicted bounding boxes should not overlap
with each other to minimize information re-
dundancy
This challenge distinguishes document layout anal-
ysis from traditional object detection tasks. While
conventional object detection aims to identify and
localize specific objects with sufficient accuracy,
document layout analysis must ensure mutually
exclusive and collectively exhaustive predictions.
To address this, we developed an adaptive se-
lection strategy based on two metrics: coverage
ratio and non-overlap ratio. The coverage ratio Rc
measures the proportion of the page area contained
within any of the selected boxes:
Rc=A(Sn
i=1Bi)
A(P), (1)
where Birepresents the region of selected box i,
A(B)represents the area of region B,A(P)de-
notes the whole page area, and nis the number of

selected boxes. The non-overlap ratio Roquantifies
the degree of redundancy between selected boxes:
Ro= 1−AS
1≤i<j≤n(Bi∩Bj)
A(P).(2)
We then define a weighted sum score Sto balance
these competing objectives:
S=α·Rc+β·Ro, (3)
where αandβare weighting parameters control-
ling the trade-off between coverage and overlap.
Our implementation follows a greedy progres-
sive approach: we first establish a minimum confi-
dence threshold to create a candidate pool of boxes.
Starting with an empty selection, we iteratively add
boxes in descending order of confidence, calculat-
ing the score Sat each step. A box is added to
the final selection if it increases the overall score
S; otherwise, the box is skipped. This approach
ensures we maximize information coverage while
minimizing redundancy. We tuned the parameters
α,β, and confidence threshold in terms of the IoU
score, which resulted in α= 0.9,β= 0.1, and a
confidence threshold of 0.4. We used fine-tuned RT-
DETR-X and the validation split of DocSemNet
for the parameter tuning.
In addition, to further refine the selection result,
we remove boxes completely covered by another
selected box, contributing to less redundancy while
keeping coverage. After employing these strategies,
we achieved an IoU score of 60.2. Appendix C
shows the examples before and after the predicted
box selection.
3.3 Post-processing for RAG
We first extract the sub-images of these elements by
the coordinates of SCAN’s predicted bounding box
result. For textual RAG, we use VLMs to convert
these sub-images into text, then combine them ac-
cording to their original reading order relationships
to obtain the page-level result for retrieval. We
can use general rule-based reading order methods;
in our experiments, we simply sorted the reading
order based on the coordinates of the upper-left
corners of various elements.
For visual RAG, since we aim to use semantic
boxes as new retrieval object units to replace origi-
nal pages with larger information content, we have
three options and ablation in Section 4.4 test the
performance of them.1. all-box : Directly using the blocks segmented
after semantic layout analysis as new objects for
image retrieval. Note that titles, headers, etc., are
treated as independent images.
2. semantic_box-only : Using only the content
of semantic_boxes as new objects for image re-
trieval.
3. concatenation : Using a concatenation
method to combine global information with seman-
tic box content. Specifically, we append the sub-
images of (title, header, footer, date, and author) to
each semantic_box sub-image.
3.4 Cost Discussion for VLM text conversion
When applying SCAN for textual RAG, we replace
a single high-resolution document page with small
sub-images, but the VLM inference cost is almost
unchanged. This is because current mainstream
VLMs consider number of tokens based on image
pixels. For example, the Qwen-2.5-VL model cor-
responds 28 ×28 pixels to a single visual token.
SCAN divides images into sub-images, but the to-
tal number of pixels remains essentially unchanged,
or decreased since SCAN skips some white areas.
Therefore, the number of input visual tokens re-
mains roughly the same. Although multiple VLM
calls will increase the total text tokens due to re-
peated text conversion prompts, this increase is
negligible since visual tokens dominate the compu-
tational cost.
During inference, because of the KV cache
mechanism, time efficiency generally increases
linearly with the length of output tokens. There-
fore, for an expect text conversion result, the cost
of SCAN’s multiple inferences with sub-images
should be comparable to a single inference with the
entire page image. Moreover, SCAN’s segmenta-
tion embodies a divide-and-conquer approach that
can be parallelized, fully utilizing GPU capabilities
when resources permit. Our experiments in Section
4.5 empirically validate this analysis.
4 Experiments
4.1 Datasets and Setting
We use three datasets to evaluate the RAG perfor-
mance. One English dataset OHR-Bench (Zhang
et al., 2025) and two Japanese datasets, Allga-
nize (Allganize.ai, 2024) and our in-house BizMM-
RAG. Every dataset is for both textual and visual
RAG. Table 2 gives a detailed summary of these
datasets.

Dataset Domain Lang. #Docs #Pages #QA QA Type
OHR-Bench (Zhang et al., 2025) Academic, Administration, Finance, En 1,261 8,561 8,498 Text, Table, Formula,
Law, Manual, Newspaper, Textbook Chart, Reading Order
Allganize (Allganize.ai, 2024) Finance, Information Technology, Manufacturing, Ja 64 2,167 300 Text, Table, Chart
Public Sector, Retail/Distribution
BizMMRAG Finance, Government, Medical, Ja 42 3,924 146 Text, Table, Chart
Consulting sectors, Wikipedia
Table 2: Statistics of our evaluation datasets.
Retrieval Generation Overall
TXT TAB FOR CHA RO ALL TXT TAB FOR CHA RO ALL TXT TAB FOR CHA RO ALL
Ground Truth 81.2 69.6 74.8 70.3 9.8 70.0 49.4 46.0 34.0 47.0 28.2 43.9 45.0 34.6 28.0 32.9 18.7 36.1
Pipeline-based OCR
MinerU 67.7 48.5 51.1 16.5 5.9 50.1 45.9 39.3 28.6 9.7 29.5 36.7 41.4 28.5 23.0 9.3 17.8 30.0
Marker 75.2 57.8 55.4 19.7 5.9 56.6 44.5 37.8 27.8 10.9 26.2 35.9 40.1 28.1 22.3 10.0 16.2 29.5
OCR-specialized small VLM
GOT-580M 62.1 41.0 48.7 17.4 3.7 45.4 37.5 28.5 24.1 8.5 7.1 27.8 35.3 22.9 20.1 8.2 5.3 24.6
Nougat-350M 59.1 32.7 44.2 11.3 4.4 40.9 36.7 22.9 22.9 6.4 6.9 25.5 33.5 18.4 19.4 5.8 3.6 14.5
SCAN-GOT-580M (ours) 60.4 44.1 42.6 28.5 4.5 45.8 (+0.4) 45.4 47.4 27.3 20.3 23.2 36.3 (+8.5) 38.4 25.2 20.3 15.2 17.4 28.1 (+3.5)
SCAN-Nougat-350M (ours) 61.7 27.9 37.1 10.8 4.3 39.5 (−1.4) 41.9 22.7 21.9 6.4 4.1 27.3 ( +1.8) 37.3 17.9 17.5 6.4 5.4 23.5 ( +9.0)
General VLM
InternVL2.5-78B 68.6 57.9 55.6 45.1 2.7 56.2 41.7 41.8 29.0 33.6 3.3 35.8 38.2 31.0 23.3 22.9 3.1 29.6
olmOCR-7B 72.5 58.4 55.4 24.8 5.0 56.6 44.8 40.5 30.4 19.0 8.4 36.0 40.6 30.3 23.7 12.8 7.1 29.6
Qwen2.5-VL-72B 75.1 60.0 60.0 38.2 5.3 59.6 44.3 42.1 31.8 27.0 11.6 37.5 40.6 31.1 26.1 19.0 8.8 31.1
DocLayout-YOLO-Qwen2.5-VL-72B 63.5 12.4 36.0 11.7 5.4 36.3 (−23.3) 41.2 10.2 19.0 7.2 18.2 24.4 (−13.1) 38.4 10.0 16.3 7.7 12.5 22.4 (−8.7)
Beehive-Qwen2.5-VL-72B 73.3 16.0 42.3 16.3 6.1 42.6 (−17.0) 46.6 11.3 21.3 10.2 27.2 28.2 (−9.3) 43 10.9 19.5 8.6 16.6 25.3 (−5.8)
SCAN-Qwen2.5-VL-72B (ours) 75.2 56.8 56.6 36.8 6.2 58.3 (−1.3) 47.4 41.7 30.2 23.9 28.3 39.5 (+2.0) 43.3 31.3 25.2 17.9 17.5 32.8 (+1.7)
Table 3: Textual RAG results on OHR-Bench. Comparison of various OCR methods across different evaluation
metrics. TXT, TAB, FOR, CHA, RO, and ALL represent text, table, formula, chart, reading order, and their average,
respectively. The RO category includes questions that require identifying the correct reading order to associate
information from separate paragraphs.
When evaluating OHR-Bench for textual RAG,
we follow the same setting as in the pa-
per (Zhang et al., 2025). We use BGE-m3 (Chen
et al., 2024a) and BM25 as retrieval models,
and meta-llama/Llama-3.1-8B-Instruct and
Qwen/Qwen2-7B-Instruct as answer models to
generate answers according to the retrieved top-2
results. We evaluate three metrics: (a) retrieval,
which calculates LCS (Longest Common Subse-
quence) to measure evidence inclusion in retrieved
content; (b) generation, which measures the F1-
score of QA when provided with the ground truth
page; and (c) overall, which calculates the F1-score
of QA for the end-to-end RAG pipeline. The F1-
score is calculated using precision and recall of
common tokens between the generated result and
the ground truth. The final scores will be the aver-
age score across the combination of two retrieval
models and two answer models. On the other hand,
we also apply OHR-Bench to visual RAG, which
uses ColQwen2-v1.0 (Faysse et al., 2024) as an
image retrieval model with top-5 retrieval, and use
Qwen/Qwen2.5-VL-7B as an answer model.
For the Japanese datasets BizMMRAG and
Allganize, we employ intfloat/multilingual-
e5-large as a textual retrieval model, andColQwen2-v1.0 as an image retrieval model.
For answer generation, we employ GPT-4o
(gpt-4o-2024-08-06 ) in textual RAG and
Qwen2.5-VL-7B in visual RAG. For each query,
the top-5 retrieved results are used. For evaluation,
we adopt the LLM-as-a-judge framework (Gu
et al., 2025), using GPT-4o with a temperature
of 0.3 to assign an integer score from 1 to 5 to
each generated answer. Answers receiving a score
greater than 4 are considered correct (assigned a
value of 1), while others are considered incorrect
(assigned a value of 0). Final accuracy is computed
based on these binary scores.
We test all three post-processing options in vi-
sual RAG and report the best result.
4.2 Textual RAG Evaluation Results
OHR-Bench. Table 3 presents the comprehensive
evaluation results of our SCAN method applied
to various VLMs for text conversion in textual
RAG. Among conventional approaches, general
VLMs demonstrate superior performance, followed
by pipeline-based OCR methods, and the last are
OCR-specialized small VLMs.
Our SCAN model can improve the performance
of VLM-based text conversions. Even the strong
baseline of Qwen2.5-VL-72B model achieves an

BizMMRAG Allganize
TXT TAB CHA ALL TXT TAB CHA ALL
Textual RAG
LlamaParse-Premium 83.3 59.1 69.1 70.5 80.1 59.2 72.0 70.7
Qwen2.5-VL-72B 85.0 52.3 69.5 68.7 84.5 68.4 62.2 71.7
DocLayout-YOLO-Qwen2.5-VL-72B 61.7 25.0 28.6 38.4 (−30.3) 49.3 21.1 23.2 31.2 (−40.5)
Beehive-Qwen2.5-VL-72B 70.0 29.6 21.4 40.3 (−28.4) 61.3 40.8 26.8 43.0 (−28.7)
SCAN-Qwen2.5-VL-72B (ours) 81.0 70.5 73.8 75.3 (+6.6) 86.6 79.0 68.3 78.0 (+6.3)
Visual RAG
Qwen2.5-VL-7B 71.7 56.8 57.1 58.9 75.9 71.1 62.5 69.9
SCAN-Qwen2.5-VL-7B (ours) 75.0 63.6 57.1 65.3 (+6.4) 79.4 64.5 76.3 73.4 (+3.5)
Table 4: Textual and visual RAG results on Japanese datasets: BizMMRAG and Allganize.
TXT TAB FOR CHA RO ALL
Qwen2.5-VL-7B 84.0 68.6 71.5 58.7 67.9 70.2
SCAN-Qwen2.5-VL-7B (ours) 86.1 68.6 73.2 61.7 87.8 75.5 (+5.3)
Table 5: Visual RAG results on OHR-Bench.
impressive overall score of 31.1% (the ground truth
is 36.1%), applying SCAN to Qwen2.5-VL-72B
further improves the performance to 32.8%. The
performance gains are larger when applying SCAN
to OCR-specialized small VLMs. With GOT and
Nougat, our SCAN improvements are 3.5% and
9.0%, respectively, enabling these smaller models
to achieve competitive performance comparable
to much larger-sized VLMs. This finding has im-
portant implications for deployment scenarios with
computational constraints, suggesting that our se-
mantic layout analysis approach can help bridge the
efficiency-performance gap. The generation results
exhibit similar improvement patterns. The results
also indicate that SCAN’s enhancements for struc-
tured content elements such as reading order (RO),
tables (TAB) and formulas (FOR) become increas-
ingly significant. This suggests that the semantic
segmentation approach is particularly valuable for
preserving the relationships between elements that
have spatial dependencies.
On the other hand, applying SCAN slightly de-
grades the retrieval performance for some models.
This is because retrieval represents a relatively sim-
pler task within the RAG pipeline, primarily requir-
ing the correct identification of keywords rather
than comprehensive document understanding. In
contrast, the subsequent question-answering stage
demands precise and complete conversion of doc-
ument content into text, where SCAN’s semantic
layout analysis proves particularly advantageous.
We can also see that conventional fine-grained
document analysis methods substantially de-
grade overall performance, with DocLayout-
YOLO (Zhao et al., 2024b) reducing retrieval per-formance by 23.3% and overall performance by
8.3%. Beehive (Auer et al., 2024) has the same
trends that reduce 5.8% overall. These degrada-
tions are particularly severe for structured content
types such as tables and charts. This proves that
conventional layout analysis methods typically seg-
ment documents into small atomic regions, which
also break the structure of documents. In contrast,
our semantic box approach preserves the integrity
of semantically coherent regions, maintains their
holistic structure while still providing the organiza-
tional benefits of layout analysis. This preservation
of semantic unity enables VLMs to process each
region with full contextual awareness, leading to
more accurate text conversion and, ultimately, su-
perior RAG performance.
BizMMRAG and Allganize. The upper section
of Table 4 presents the textual RAG performance
for Japanese document datasets. The results have
the same trends as the English OHR-Bench eval-
uation, demonstrating that our SCAN methodol-
ogy yields substantial improvements over general
VLMs. Specifically, on the BizMMRAG dataset,
our SCAN-enhanced approach demonstrates no-
table 6.6% improvements compared to the baseline
Qwen2.5-VL-72B model: text accuracy decreased
by 4.0%, while table increased by 18.2% and chart
increased by 4.3%. We observe similar trends for
the Allganize dataset. It is particularly notewor-
thy that our SCAN-enhanced method outperforms
LlamaParse-Premium (LlamaParse.AI, 2025), a
commercial product offered by LlamaIndex, specif-
ically designed for document conversion. This com-
parison against a commercial solution validates the
practical utility and competitive advantage of our

OHR-Bench BizMMRAG Allganize
semantic_box-only 75.1 65.3 72.1
concatenation 73.4 65.3 73.4
all-box 75.5 58.9 71.7
Table 6: Ablation study to compare the different post-
processing strategies in visual RAG.
Setting # Input tokens # Output tokens Time (s)
Single-page 11,295 (prompt=255) 1,005 71.3
Multi-chunks 8,079 (prompt=1,071) 1,092 72.7
Table 7: Token numbers, processing time comparison
of VLM text conversion.
research contribution. In addition, the result shows
the same trends: using fine-grained document anal-
ysis, DocLayout-YOLO, and Beehive heavily de-
grades overall performance.
4.3 Visual RAG Evaluation Results
OHR-Bench. Table 5 presents the result of OHR-
Bench in visual RAG. When applying our SCAN
approach to divide original pages into seman-
tic chunks and performing visual RAG on these
chunks, we observed an overall improvement of
5.3% compared to processing entire pages. While
the Table task has the same performance as the
baseline, we can see modest improvements in other
categories, along with a significant enhancement in
the Reading Order task. Recall that Reading Order
problems require examining different paragraphs
and articles to summarize answers. This demon-
strates that dividing a page image into independent
semantic chunks can reduce noise in reading order-
related problems, enabling the model to provide
more accurate responses.
BizMMRAG and Allganize. The lower sec-
tion of Table 4 presents the visual RAG results for
Japanese document datasets. The findings demon-
strate that on both the BizMMRAG and Allga-
nize benchmarks, our SCAN methodology exhibits
substantial accuracy improvements. Specifically,
SCAN improves 6.4% on BizMMRAG and 3.5%
on Allganize. This result also shows that our SCAN
approach enables the VLM to achieve significantly
enhanced performance in Japanese VQA.
4.4 Ablation study of post-processing
Table 6 shows visual RAG results with different
post-processing options. all-box performed the on
OHR-Bench, while for BizMMRAG and Allga-
nize, concatenation yielded better scores. Uponexamining the OHR-Bench data, we found that the
questions are generally unrelated to titles, head-
ers, and other global boxes. This could explain
why the all-box approach (which indexes different
elements separately) and the semantic_box-only
approach (which directly filters out title elements)
performed relatively well. Conversely, we observed
that queries in BizMMRAG and Allganize typically
required consideration of both global information
and main content, most likely causing the superior
performance of the concatenation method for these
datasets. For practical applications, we recommend
experimenting with different approaches to select
the best post-processing option.
4.5 Cost Comparison of VLM Text
Conversion
Table 7 compares the processing time of VLM-
based text conversion between single-page process-
ing and multiple semantic chunks processing.3We
can see that the average processing time for se-
mantic chunks (our SCAN approach) is similar to
the single-page processing time. The majority of
input tokens are vision tokens, compared to the
text tokens from conversion prompts. While the
total input tokens for multiple chunks are fewer,
the output tokens are more. This shows that our
SCAN approach can reduce token-based costs (es-
pecially favorable for pay-as-token-usage API mod-
els) while extracting more textual information, all
within approximately the same processing time.
5 Conclusion
We presented SCAN, a semantic document layout
analysis approach for modern textual and visual
RAG systems. By introducing coarse-grained se-
mantic segmentation that preserves topical coher-
ence, SCAN effectively reduces the information
processing burden on VLMs while maintaining se-
mantic integrity across document components. To
develop SCAN, we labeled more than 24k docu-
ment images with semantic layouts and trained a
robust semantic layout analysis model. Our com-
prehensive evaluation across multiple datasets, lan-
guages, and document types demonstrates SCAN’s
ability to enhance 1.7%–9.0% for textual RAG and
3.5%–6.4% for visual RAG. In addition, SCAN
achieves these improvements while reducing token
usage costs and maintaining the same processing
time.
3We used Qwen-2.5-VL-72B locally for this experiment.

Ethical Statement
In this work, we study semantic document layout
analysis for RAG. To the best of our knowledge,
there is no negative societal impact in this research.
All our training data consists of publicly available
PDFs from the internet, which similarly presents
no ethical concerns. Our SCAN model aims to
improve information extraction without introduc-
ing biases in the underlying content. We believe
that improved document analysis can enhance ac-
cessibility of information for users across different
languages and document formats. While our sys-
tem improves RAG capabilities, users should still
be mindful of the general limitations of AI systems
when relying on generated answers.
We used Claude-3.7-Sonnet to polish the writ-
ing of the paper. We are responsible for all the
materials presented in this work.
Limitations
While our SCAN approach offers significant ad-
vantages, we acknowledge several limitations that
present opportunities for future research:
1. Our SCAN model operates based on spatial
image layout. In certain documents where contents
that should logically form a single semantic chunk
is physically separated in space and do not fit in a
rectangular, our current model cannot yet establish
these connections. This limitation could poten-
tially be addressed through an additional trainable
reading order model coupled with a semantic_box
merging mechanism.
2. Our current model was trained primarily on
Japanese data. While experiments demonstrate im-
provements on English benchmarks as well, this
may not represent the optimal model for all lan-
guages. Japanese documents have unique layout
characteristics, such as vertical writing and right-to-
left orientation, which differ from English conven-
tions. Further analysis and exploration are needed,
and future work could involve annotating purely
English data to investigate whether higher perfor-
mance could be achieved for English RAG applica-
tions.
3. SCAN’s semantic layout was designed for
dense, content-rich document RAG. For simpler
pages, designing an adaptive approach that intel-
ligently decides whether to apply semantic layout
analysis or process the page as a single unit could
provide better generalizability in future iterations.References
Allganize.ai. 2024. Allganize rag leader-
board. https://huggingface.co/datasets/
allganize/RAG-Evaluation-Dataset-JA .
Christoph Auer, Maksym Lysak, Ahmed Nassar,
Michele Dolfi, Nikolaos Livathinos, Panos Vage-
nas, Cesar Berrospi Ramis, Matteo Omenetti, Fabian
Lindlbauer, Kasper Dinkla, Lokesh Mishra, Yusik
Kim, Shubham Gupta, Rafael Teixeira de Lima,
Valery Weber, Lucas Morin, Ingmar Meijer, Viktor
Kuropiatnyk, and Peter W. J. Staar. 2024. Docling
technical report. Preprint , arXiv:2408.09869.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shi-
jie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu,
Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei
Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo
Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang
Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin.
2025. Qwen2.5-vl technical report. arXiv preprint
arXiv:2502.13923 .
Lukas Blecher, Guillem Cucurull, Thomas Scialom,
and Robert Stojnic. 2023. Nougat: Neural optical
understanding for academic documents. Preprint ,
arXiv:2308.13418.
Nicolas Carion, Francisco Massa, Gabriel Synnaeve,
Nicolas Usunier, Alex-ander Kirillov, and Sergey
Zagoruyko. 2020. End-to-end object detection with
transformers. In Proceedings of the European Con-
ference on Computer Vision (ECCV) , pages 213–229.
Springer.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024a. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint , arXiv:2402.03216.
Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo
Chen, Sen Xing, Muyan Zhong, Qinglong Zhang,
Xizhou Zhu, Lewei Lu, et al. 2024b. Internvl: Scal-
ing up vision foundation models and aligning for
generic visual-linguistic tasks. In Proceedings of
the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 24185–24198.
Datalab. 2024. Marker. https://github.com/
VikParuchuri/marker .
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. Preprint ,
arXiv:2405.06211.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo.
2024. Colpali: Efficient document retrieval with vi-
sion language models. Preprint , arXiv:2407.01449.

Ling Fu, Biao Yang, Zhebin Kuang, Jiajun Song, Yuzhe
Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu,
Mingxin Huang, Zhang Li, Guozhi Tang, Bin Shan,
Chunhui Lin, Qi Liu, Binghong Wu, Hao Feng, Hao
Liu, Can Huang, Jingqun Tang, Wei Chen, Lianwen
Jin, Yuliang Liu, and Xiang Bai. 2024. Ocrbench
v2: An improved benchmark for evaluating large
multimodal models on visual text localization and
reasoning. Preprint , arXiv:2501.00321.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey. Preprint ,
arXiv:2312.10997.
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun
Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni,
and Jian Guo. 2025. A survey on llm-as-a-judge.
Preprint , arXiv:2411.15594.
Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and
Furu Wei. 2022. Layoutlmv3: Pre-training for doc-
ument ai with unified text and image masking. In
Proceedings of the 30th ACM International Confer-
ence on Multimedia .
Rahima Khanam and Muhammad Hussain. 2024.
Yolov11: An overview of the key architectural en-
hancements. Preprint , arXiv:2410.17725.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2021.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. Preprint , arXiv:2005.11401.
Junlong Li, Yiheng Xu, Tengchao Lv, Lei Cui, Cha
Zhang, and Furu Wei. 2022. Dit: Self-supervised pre-
training for document image transformer. Preprint ,
arXiv:2203.02378.
Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu
Wei, Zhoujun Li, and Ming Zhou. 2020. Docbank:
A benchmark dataset for document layout analysis.
Preprint , arXiv:2006.01038.
LlamaParse.AI. 2025. Llamaparse: Transform un-
structured data into llm optimized formats. https:
//www.llamaindex.ai/llamaparse .
microsoft. 2025. Azure ai document in-
telligence. https://azure.microsoft.
com/en-us/products/ai-services/
ai-document-intelligence .
Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz,
Elad Ben Avraham, Alona Golts, Yair Kittenplon,
Shai Mazor, and Ron Litman. 2024. Docvlm:
Make your vlm an efficient reader. Preprint ,
arXiv:2412.08746.Birgit Pfitzmann, Christoph Auer, Michele Dolfi,
Ahmed S Nassar, and Peter W J Staar. 2022.
Doclaynet: A large human-annotated dataset for
document-layout analysis.
Jake Poznanski, Jon Borchardt, Jason Dunkelberger,
Regan Huff, Daniel Lin, Aman Rangapur, Christo-
pher Wilhelm, Kyle Lo, and Luca Soldaini.
2025. olmOCR: Unlocking Trillions of Tokens
in PDFs with Vision Language Models. Preprint ,
arXiv:2502.18443.
Joseph Redmon, Santosh Divvala, Ross Girshick, and
Ali Farhadi. 2016. You only look once: Unified,
real-time object detection. In Proceedings of the
IEEE conference on computer vision and pattern
recognition (CVPR) , pages 779–788.
OpenBMB Rhapsody Group. 2024. Memex: Ocr-
free visual document embedding model as your
personal librarian. https://huggingface.co/
RhapsodyAI/minicpm-visual-embedding-v0 .
Accessed: 2024-06-28.
Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke
Nishida, Kuniko Saito, and Jun Suzuki. 2025.
Vdocrag: Retrieval-augmented generation over
visually-rich documents. In CVPR .
Michał Turski, Tomasz Stanisławek, Karol Kaczmarek,
Paweł Dyda, and Filip Grali ´nski. 2023. Ccpdf: Build-
ing a high quality corpus for visually rich documents
from web crawl data. In International Conference on
Document Analysis and Recognition , pages 348–365.
Springer.
Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan
Qu, Fukai Shang, et al. 2024. Mineru: An open-
source solution for precise document content extrac-
tion. arXiv preprint arXiv:2409.18839 .
Haoran Wei, Chenglong Liu, Jinyue Chen, Jia Wang,
Lingyu Kong, Yanming Xu, Zheng Ge, Liang Zhao,
Jianjian Sun, Yuang Peng, Chunrui Han, and Xi-
angyu Zhang. 2024. General ocr theory: Towards
ocr-2.0 via a unified end-to-end model. Preprint ,
arXiv:2409.01704.
Ning Yu, Chia-Chih Chen, Zeyuan Chen, Rui Meng,
Gang Wu, Paul Josel, Juan Carlos Niebles, Caiming
Xiong, and Ran Xu. 2023. Layoutdetr: Detection
transformer is a good multimodal layout designer.
arXiv preprint arXiv:2212.09877 .
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025.
Visrag: Vision-based retrieval-augmented gener-
ation on multi-modality documents. Preprint ,
arXiv:2410.10594.
Junyuan Zhang, Qintong Zhang, Bin Wang, Linke
Ouyang, Zichen Wen, Ying Li, Ka-Ho Chow, Con-
ghui He, and Wentao Zhang. 2025. Ocr hinders rag:
Evaluating the cascading impact of ocr on retrieval-
augmented generation. Preprint , arXiv:2412.02592.

Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei,
Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie
Chen. 2024a. Detrs beat yolos on real-time object
detection. Preprint , arXiv:2304.08069.
Zhiyuan Zhao, Hengrui Kang, Bin Wang, and Con-
ghui He. 2024b. Doclayout-yolo: Enhancing doc-
ument layout analysis through diverse synthetic data
and global-to-local adaptive perception. Preprint ,
arXiv:2410.12628.
Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes.
2019. Publaynet: largest dataset ever for document
layout analysis. In 2019 International Conference on
Document Analysis and Recognition (ICDAR) , pages
1015–1022. IEEE.

A Examples from Our Semantic Layout
Dataset
Figure 3 shows some examples of our semantic
layout dataset. It contains diverse document pages,
including research papers, administrative reports,
user manuals, slides, flyers, and more. The dataset
is annotated with the semantic_box class and the
five global box classes: title,header ,footer ,date,
andauthor .
B Training Details of Our Object
Detection Models
We fine-tuned two off-the-shelf object detection
models: YOLO11-X and RT-DETR-X. We fol-
lowed the default settings provided by the Ultr-
alytics framework (version 8.3.28)4mostly, but
we explicitly set or tuned some important hyper-
parameters as shown in Table 8. For the YOLO11-
X fine-tuning, we used 4 NVIDIA L40 GPUs
(48GB), which took about 4 hours to finish 30
epochs. For the RT-DETR-X fine-tuning, we used
8 NVIDIA A100 GPUs (80GB), which took about
16 hours to finish 120 epochs.
C Examples of Before and After the
Predicted Box Selection
Figure 4–6 exemplify our predicted box selection
strategy for the semantic document layout analysis.
D Experimental Details
D.1 Environment
For the text conversion with general VLMs, we
used 8 NVIDIA L40S (48GB) GPUs with IN-
TEL(R) XEON(R) GOLD 6548N CPU, and the
details are as follows.
python 3.12
vllm ==0.7.3 (V0 version )
torch ==2.5.1
torchaudio ==2.5.1
torchvision ==0.20.1
transformers ==4.49.0
ultralytics ==8.3.28
vLLM settings
- temperature : 0.3
- top_p : 0.95
- max_tokens : 8192
- repetition_penalty : 1.1
- tensor_parallel ==4
4https://github.com/ultralytics/ultralytics/
releases/tag/v8.3.28D.2 Prompts for the VLM Text Conversion
You are a powerful OCR assistant tasked
with converting PDF images to the
Markdown format . You MUST obey the
following criteria :
1. Plain text processing :
- Accurately recognize all text content
in the PDF image without guessing or
inferring .
- Precisely recognize all text in the
PDF image without making assumptions
in the Markdown format .
- Maintain the original document
structure , including headings ,
paragraphs , lists , etc .
2. Formula Processing :
- Convert all formulas to LaTeX .
- Enclose inline formulas with $ $. For
example : This is an inline formula $
E = mc ^2 $.
- Enclose block formulas with $$ $$. For
example : $$ \ frac {-b \pm \ sqrt {b^2
- 4ac }}{2 a} $$.
3. Table Processing :
- Convert all tables to LaTeX format .
- Enclose the tabular data with \ begin {
table } \ end { table }.
4. Chart Processing :
- Convert all Charts to LaTeX format .
- Enclose the chart data in tabular with
\ begin { table } \ end { table }.
5. Figure Handling :
- Ignore figures from the PDF image ; do
not describe or convert images .
6. Output Format :
- Ensure the Markdown output has a clear
structure with appropriate line
breaks .
- Maintain the original layout and
format as closely as possible .
Please strictly follow these guidelines
to ensure accuracy and consistency
in the conversion . Your task is to
accurately convert the content of
the PDF image using these format
requirements without adding any
extra explanations or comments .
D.3 Prompts for LLM-as-a-judge
System :
You are an expert evaluation system for
a question answering chatbot .
You are given the following information :
- a user query and reference answer
- a generated answer
You may also be given a reference answer
to use for reference in your
evaluation .
Your job is to judge the relevance and
correctness of the generated answer .
Output a single score that represents a
holistic evaluation .
You must return your response in a line
with only the score .

Do not return answers in any other
format .
On a separate line provide your
reasoning for the score as well .
Follow these guidelines for scoring :
- Your score has to be between 1 and 5,
where 1 is the worst and 5 is the
best .
- Your output format should be in JSON
with fields " reason " and " score "
shown below .
- If the generated answer is not
relevant to the user query , you
should give a score of 1.
- If the generated answer is relevant
but contains mistakes , you should
give a score between 2 and 3.
- If the generated answer is relevant
and fully correct , you should give a
score between 4 and 5.
Example Response in JSON format :
{{
" reason ": " The generated answer has
the exact same metrics as the
reference answer , but it is not
as concise .",
" score ": "4.0"
}}
User :
## User Query
{ query }
## Reference Answer
{ reference_answer }
## Generated Answer
{ generated_answer }

Figure 3: Examples of our semantic layout dataset.

Hyper-parameter YOLO11-X RT-DETR-X
Batch size {8, 16, 32, 64} {8, 16, 32, 64}
Learning rate {1e-4, 5e-4, 1e-3, 5e-3} {5e-5, 1e-4, 5e-4, 1e-3, 5e-3}
Max training epochs {30, 40, 80, 120} {80, 120, 160}
Weight decay {5e-5, 1e-4, 5e-4} {1e-5, 1e-4, 1e-3}
Warmup epochs {5, 10} {5, 10}
Image size 1024
Dropout 0.0
Optimizer AdamW
Learning rate scheduler cos_lr
Table 8: Hyper-parameters used for fine-tuning object detection models. We tuned the hyper-parameters in the
brackets in terms of the mean average precision (mAP) on the validation set. The bold values are the best hyper-
parameters for each model.
Figure 4: Semantic document layout analysis result before (left) and after (right) our predicted box selection strategy
(1/3).

Figure 5: Semantic document layout analysis result before (left) and after (right) our predicted box selection strategy
(2/3).
Figure 6: Semantic document layout analysis result before (left) and after (right) our predicted box selection strategy
(3/3).