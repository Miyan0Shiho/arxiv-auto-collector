# XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism on a Novel Benchmark

**Authors**: Shuai Liu, Youmeng Li, Jizeng Wei

**Published**: 2025-04-14 14:19:57

**PDF URL**: [http://arxiv.org/pdf/2504.10258v1](http://arxiv.org/pdf/2504.10258v1)

## Abstract
Document Reading Order Recovery is a fundamental task in document image
understanding, playing a pivotal role in enhancing Retrieval-Augmented
Generation (RAG) and serving as a critical preprocessing step for large
language models (LLMs). Existing methods often struggle with complex
layouts(e.g., multi-column newspapers), high-overhead interactions between
cross-modal elements (visual regions and textual semantics), and a lack of
robust evaluation benchmarks. We introduce XY-Cut++, an advanced layout
ordering method that integrates pre-mask processing, multi-granularity
segmentation, and cross-modal matching to address these challenges. Our method
significantly enhances layout ordering accuracy compared to traditional XY-Cut
techniques. Specifically, XY-Cut++ achieves state-of-the-art performance (98.8
BLEU overall) while maintaining simplicity and efficiency. It outperforms
existing baselines by up to 24\% and demonstrates consistent accuracy across
simple and complex layouts on the newly introduced DocBench-100 dataset. This
advancement establishes a reliable foundation for document structure recovery,
setting a new standard for layout ordering tasks and facilitating more
effective RAG and LLM preprocessing.

## Full Text


<!-- PDF content starts -->

XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism on a
Novel Benchmark
Shuai Liu
Tianjin University
Tianjin, China
shuai liu@tju.edu.cnYoumeng Li*
Tianjin University
Tianjin, China
liyoumeng@tju.edu.cnJizeng Wei
Tianjin University
Tianjin, China
weijizeng@tju.edu.cn
Abstract
Document Reading Order Recovery is a fundamental task
in document image understanding, playing a pivotal role
in enhancing Retrieval-Augmented Generation (RAG) and
serving as a critical preprocessing step for large lan-
guage models (LLMs). Existing methods often struggle
with complex layouts(e.g., multi-column newspapers), high-
overhead interactions between cross-modal elements (vi-
sual regions and textual semantics), and a lack of robust
evaluation benchmarks. We introduce XY-Cut++ , an ad-
vanced layout ordering method that integrates pre-mask
processing, multi-granularity segmentation, and cross-
modal matching to address these challenges. Our method
significantly enhances layout ordering accuracy compared
to traditional XY-Cut techniques. Specifically, XY-Cut++
achieves state-of-the-art performance (98.8 BLEU overall)
while maintaining simplicity and efficiency. It outperforms
existing baselines by up to 24% and demonstrates consistent
accuracy across simple and complex layouts on the newly
introduced DocBench-100 dataset. This advancement es-
tablishes a reliable foundation for document structure re-
covery, setting a new standard for layout ordering tasks and
facilitating more effective RAG and LLM preprocessing.
1. Introduction
Document Reading Order Recovery is a fundamental task
in document image understanding, serving as a cornerstone
for enhancing Retrieval-Augmented Generation (RAG) sys-
tems and enabling high-quality data preprocessing for large
language models (LLMs). Accurate layout recovery is
essential for applications such as digital publishing and
knowledge base construction. However, this task faces
several significant challenges: (1) complex layout struc-
ture (e.g., multi-column layout, nested text boxes, non-
rectangular text regions, cross-page content), (2) inefficient
cross-modal alignment due to the high computational costs
*Corresponding author.of integrating visual and textual features, and (3) the lack
of standardized evaluation protocols for block-level read-
ing order. Traditional approaches like XY-Cut [2] fail to
model semantic dependencies in complex designs, while
deep learning methods such as LayoutReader [13] suffer
from prohibitive latency, limiting real-world deployment.
Compounding these issues, existing datasets like Reading-
Bank [13] focus on word-level sequence annotations, which
inadequately evaluate block-level structural reasoning—a
necessity for real-world layout recovery. Although Om-
niDocBench [8] recently introduced block-level analysis
support, its coverage of diverse complex layouts (e.g., news-
papers, technical reports) remains sparse, further hindering
systematic progress.
To address these challenges, we propose XY-Cut++,
an advanced framework for layout ordering that incor-
porates three core innovations: (a) pre-mask processing
to mitigate interference from high-dynamic-range elements
(e.g., title, figure, table), (b) multi-granularity region split-
ting for the adaptive decomposition of diverse layouts,
and (c) lightweight cross-modal matching leveraging min-
imal semantic cues. Our approach significantly outper-
forms traditional methods in layout ordering accuracy. Ad-
ditionally, we introduce DocBench-100, a novel bench-
mark dataset designed for evaluating layout ordering tech-
niques. It includes 100 pages (30 complex and 70 reg-
ular layouts) with block-level reading order annotations.
Extensive experiments on DocBench-100 demonstrate that
our method achieves state-of-the-art performance. Specifi-
cally, our method achieves BLEU-4 scores of 98.6(complex)
and 98.9(regular), surpassing baselines by 24% on aver-
age while maintaining 1.06× FPS of geometric-only ap-
proaches.
The contributions of this paper are summarized as fol-
lows:
1. We present a simple yet high-performance enhanced XY-
Cut framework that fuses shallow semantics and geome-
try awareness to achieve accurate layout ordering.
2. We establish a new benchmark dataset for layout order-arXiv:2504.10258v1  [cs.CV]  14 Apr 2025

DocBench-100
papernotebookbook……
Visual Information：
Shallow Semantics:Text,Title,Figure…XY-Cut++
End2end Eval(Images)Order Eval(Jsons)newspaper
123456789101112131415161718192021222324252627282930313233“Index”Extraction
Layout DetectionFigure 1. XY-Cut++ Architecture with Hierarchical Mask Mechanism and DocBench-100 Benchmark. (left) DocBench-100 dataset
composition (9 document categories, Multi-type layout) and dual evaluation protocols: end-to-end image-based assessment (requiring
layout parsing models) and direct JSON-based metric computation. (right) Algorithm workflow integrating adaptive pre-mask processing,
multi-granularity segmentation, and cross-modal matching. Unlike neural network models, our method is simple, effective, and strongly
interpretable.
3 41 2
6 75②④
③⑤①
⑥1 2 3 45
6 71U2
3U46U71U2U3U4U5U6U7
Figure 2. XY-Cut Recursive Partitioning Workflow and Failure
Analysis in Complex Layouts: (1) Initial document segmentation
steps, (2) Connectivity assumption violations caused by cross-
layout cell structures (cell 5), and (3) Error propagation through
subsequent layout ordering.
ing by curating complex layouts from document detec-
tion datasets, enriching the diversity of challenging lay-
outs and offering a more comprehensive resource than
existing datasets.
3. We achieved state-of-the-art results on both existing and
new datasets, and provided extensive ablations to each
component in our methods.
2. Related Work
Document Reading Order Recovery has seen significant ad-
vancements, evolving from early heuristic rule-based ap-
proaches to sophisticated deep learning approaches.2.1. Traditional Approaches
The XY-Cut algorithm [2] is a foundational technique in
Document Reading Order Recovery that recursively divides
documents into smaller regions based on horizontal and
vertical projections. As illustrated in Fig. 2, while it is effec-
tive for simple layouts, it struggles with complex structures,
leading to inaccuracies in reading order recovery. Specifi-
cally, the rigid threshold mechanisms of XY-Cut can intro-
duce hierarchical errors when handling intricate layouts,
resulting in suboptimal performance. To address these limi-
tations, various enhancements have been proposed. For in-
stance, dynamic programming optimizations [7] have been
introduced to improve segmentation stability. Additionally,
mask-based normalization techniques [11] have been de-
veloped to mitigate some of the challenges associated with
complex layouts. However, these improvements are still in-
sufficient for handling intricate layouts and cross-page con-
tent. In our analysis of the XY-Cut algorithm, as discussed
in [7, 11] and illustrated in Fig. 3, we identified that many
challenging cases arise from L-shaped inputs. A straight-
forward solution involves initially masking these L-shaped
regions and subsequently restoring them. Upon implement-
ing this approach, we found it to be remarkably effective
while maintaining both simplicity and efficiency.
2.2. Deep Learning Approaches
Recent advances in Document Reading Order Recovery
have been driven by deep learning models that effectively
leverage multimodal cues. The LayoutLM series [1, 3,

NoCutError Cut
①②
③xFigure 3. Challenges posed by L-shaped inputs: (1) segmenta-
tion failure due to the inability to process L-shaped structures, and
(2) missegmentation caused by improper handling of L-shaped re-
gions. ( The correct order is ➁+➂ ➀)
14, 16, 17] pioneered the integration of textual semantics
with 2D positional encoding, where LayoutLMv2 [17] in-
troduced spatially aware pre-training objectives for cross-
modal alignment and LayoutLMv3 [3] further unified
text/image embeddings through self-supervised masked lan-
guage modeling and image-text matching. LayoutXLM [16]
extended this framework to multilingual document under-
standing. XYLayoutLM [1] advanced the field with its Aug-
mented XY-Cut algorithm and Dilated Conditional Posi-
tion Encoding, addressing variable-length layout model-
ing and generating reading order-aware representations for
enhanced document understanding. Building upon these
foundations, LayoutReader [13] demonstrated the effec-
tiveness of deep learning in explicit reading order predic-
tion through the sequential modeling of paragraph rela-
tionships. Critical to these advancements are large-scale
datasets like DocBank [5], which provides weakly super-
vised layout annotations for fine-grained spatial modeling,
and PubLayNet [20], which contains over 360,000 scien-
tific document pages with hierarchical labels that encode
implicit reading order priors through structural patterns.
Specialized benchmarks like TableBank [4] further address
domain-specific challenges by preserving the ordering of
tabular data for table structure recognition.
Architectural innovations have significantly enhanced
spatial reasoning capabilities. DocRes [19] introduces
dynamic task-specific prompts (DTSPrompt), enabling the
model to perform various document restoration tasks, such
as dewarping, deshadowing, and appearance enhancement,
meanwhile improving the overall readability and structure
of documents. [18] iteratively aggregates features across
different layers and resolutions, resulting in more refined
feature representations that are beneficial for complex lay-
out analysis tasks. By leveraging Hierarchical Document
Analysis (HDA), models can more effectively capture intri-
cate structural relationships within documents, facilitating
more accurate predictions of reading order. Furthermore,
advancements in unsupervised learning methods, such as
those employed in DocUNet [6], have enabled more ef-
fective handling of document distortions, thereby enhanc-
ing OCR performance and layout analysis accuracy. Doc-
Former [15] unified text, visual, and layout embeddings
through transformer fusion, improving contextual under-standing for logical flow prediction. Complementary to
these approaches, EAST [21] established robust text detec-
tion through geometry-aware frameworks, serving as a crit-
ical preprocessing step for element-level sequence deriva-
tion.
2.3. Benchmarks for Document Reading Order Re-
covery
Document Reading Order Recovery has seen significant ad-
vancements with deep learning models like LayoutLM [14]
and LayoutReader [13], which integrate visual and textual
features for tasks such as reading order prediction. How-
ever, existing methods face critical limitations in handling
complex layouts (e.g., multi-column structures, cross-page
content). A key challenge is the lack of benchmarks that
directly evaluate block-level reading order, which is es-
sential for applications like document digitization. While
datasets such as ReadingBank [13] provide word-level an-
notations for predicting reading sequences, their design
complicates the evaluation of methods focused on block-
level performance. Specifically, word-level sequence an-
notations cannot simplify the assessment of dependencies
between text blocks (e.g., the order of paragraphs or figures
spanning multiple columns), which are essential for model-
ing complex layouts. Recently, OmniDocBench [8] has sup-
ported block-level analysis but suffers from limited coverage
of complex layouts and sparse representation of domain-
specific layouts (e.g., newspapers). To address these gaps,
we introduce DocBench-100, which offers a broader range
of layout structures and explicit metrics for assessing read-
ing order accuracy, thereby enabling robust benchmarking
of layout recovery systems.
3. Methodology
Our geometry-semantic fusion pipeline, as shown in Fig. 4,
consists of four coordinated stages: (1) PP-DocLayout [10]
extracts visual features and shallow semantic labels from
document images; (2) highly dynamic elements (e.g., titles,
tables, figures) are pre-masked to alleviate the ”L-shape”
problem; (3) cross-layout elements are identified through
global layout analysis, followed by mask processing, real-
time density estimation, and heuristic sorting to ensure log-
ical content ordering; and (4) masked elements are re-
mapped using nearest IoU edge-weighted margins. By inte-
grating geometry-aware processing with shallow semantic
information, our method achieves state-of-the-art perfor-
mance on DocBench-100 and OmniDocBench [8], demon-
strating its effectiveness in addressing complex document
layout challenges.
3.1. Multi-Granularity Segmentation
As shown in Fig. 5, our hybrid segmentation framework
combines masking, preliminary splitting, and layout-aware

doc
titletext text
text
image texttext
text
paragraph titletext
footnotetit
le
te
xtheader imageheader
a
s
i
d
e
t
e
x
t1
2
3
4
tabletext 5
c. Multi -Granularity Segmentation d.Cross -Modal Matchingsingle -layout text cross -layout text
single -layout text
single -layout 
textcross -layout text
cross -layout textsingle -layout 
text
single -layout 
text
a. Layout Detection(PP -DocLayout)XY-Cut++:  Hierarchical Mask -Based Layout Ordering Framework
13 7
4
image 98
5
6102
table11
b. Pre -Mask ProcessingFigure 4. End-to-End Layout Ordering for Diverse Document Layouts Framework Overview: (a) Layout Detection(We use PP-
Doclayout[10]), (b) Pre-Mask Processing, (c) Multi-Granularity Segmentation, and (d) Cross-Model Matching.
2.CountIou>α>=2Cross -Layout Text1.len>β*median_len
PreCut2.interval<w/5
3.no
text3.no
text1 32
4 5①②
③
④1.Pre Cut
2.Td<0.9 =>YXCut
median line
1.len>β1*median_len
Figure 5. Multi-Granularity Segmentation: (1) cross-layout ele-
ment masking, (2) preliminary segmentation via pre-cut, and (3)
ecursive density-driven partitioning. The enhanced XY-Cut algo-
rithm adapts its splitting axis selection through real-time density
evaluation ( τd), prioritizing horizontal splits for content-dense re-
gions and vertical splits otherwise.
2 41 3
6 751 32
4 5Reorder by (index, label priority, y1, x1)
(DV1,DH1) Dedge3
Diou1Diou2
Dedge1(DV3,DH3)
Diou3B1' (x11',y11',x21',y21') B3' (x13',y13',x23',y23')1D1<D4<D3<D5<D2
B2' (x12',y12',x22',y22')
… …3B1(x1,y1,x2,y2)
B2(x1,y1,x2,y2)D3<D4<D1<D5<D2
Figure 6. Cross-Modal Matching: (1) semantic hierarchy-aware
stage decomposition and (2) adaptive distance metric matching
with dynamic policies and semantic-specific tuning. Subsequently,
cells are reordered based on Index, Label Priority, Y1, and X1.refinement through three key phases:
Phase 1: Cross-Layout Detection Compute document-
level median bounding box length and establish adaptive
threshold with scaling factor β= 1.3:
Tl=β·median( {li}N
i=1) (1)
Detect cross-layout elements using dual criteria:
Ccross(Bi) =(
T l i>Tl∧P
j̸=iIoverlap (Bi, Bj)≥2
Fotherwise
(2)
Masked cross-layout elements are preserved for subsequent
layout-aware splitting, while standard components undergo
immediate processing.
Phase 2: Geometric Pre-Segmentation Central ele-
ments and isolated graphical components are identified
through:
P(Bi) = I∥ci−cpage∥2
dpage≤0.2
∧(ϕtext(Bi) =∞)(3)
where cidenotes bounding box center coordinates,
and classified visual elements include {Figure, Table }.
Here, ϕtext(Bi)represents the distance from the bounding
boxBito the nearest text box,and ∞indicates that Biis
not adjacent to any text box.
Phase 3: Density-Driven Refinement The adaptive XY-
Cut algorithm dynamically selects splitting axes based on
regional content density:
τd=PKc
k=1w(Cc)
kh(Cc)
kPKs
k=1w(Cs)
kh(Cs)
k(4)
Direction-aware splitting strategy:
S(R) =(
XY-Cut τd> θv(θv= 0.9)
YX-Cut otherwise(5)

Atomic region representation:
Ri=⟨x(i)
1, y(i)
1, x(i)
2, y(i)
2, Ci⟩, C i∈ Ctype (6)
with (x(i)
1, y(i)
1)and(x(i)
2, y(i)
2)defining the bounding box
coordinates. The content type Cidistinguishes between
Cross-layout (spanning multiple grid units) and Single-
layout (contained within one grid unit) components.
3.2. Cross-Modal Matching
To establish cross-modal reading coherence, we propose a
geometry-aware alignment framework with two core com-
ponents in Fig. 6.
Multi-Stage Semantic Filtering : Orders candidates
through label priority:
Lorder:Lcross-layout ≻ L title≻ L vision≻ L others
M(l)
sem=(
(Bp, B′
o)Lp=l
∧L′
o⪰l)
, l∈ L order
Msem=[
l∈L orderM(l)
sem(7)
whereLvision denotes visual element detection objectives in-
cluding tables, images, seals, and their corresponding ti-
tles, with Bprepresenting pending bounding boxes and B′
o
indicating ordered candidate regions.
Adaptive Distance Metric : Computes joint geometric
distance with early termination. Given layout element Bp=
(x1, y1, x2, y2)and candidate B′
o= (x′
1, y′
1, x′
2, y′
2), we de-
fine:
D(Bp, B′
o, l) =4X
k=1wk·ϕk(Bp, B′
o) (8)
where ϕkencodes four geometric constraints:
•Intersection Constraints:
ϕ1=(
1,ifdir(Bp)̸= dir( B′
o)∨IoU partial < τ overlap
0,otherwise
•Boundary Proximity:
ϕ2=wedge×(
dx+dy, (diagonal adjacency)
min(dx, dy),(axis-aligned)
•Vertical Continuity:
ϕ3=(
−y′
2, l∈ L cross-layout ∧y1> y′
2
y′
1, (baseline alignment)
•Horizontal Ordering:
ϕ4=x′
1We further introduce two parameterization strategies:
Dynamic Weight Adaptation . Scale-sensitive weights
via dimensional analysis:
wk= [max( h, w)2,max( h, w),1,max( h, w)−1](9)
where h, w denote page dimensions, it establishes a distance
metric with clear priorities.
Semantic-Specific Tuning . Optimal weights from grid
search on 2.8k documents:
wedge=

[1,0.1,0.1,1]l∈ L title∩ O horiz
[0.2,0.1,1,1]l∈ L title∩ O vert
[1,1,0.1,1] l∈ L cross-layout
[1,1,1,0.1] l∈ L otherwise(10)
Statistical validation demonstrates significant improve-
ments: +2.3 BLEU-4 over uniform baselines. The overall
process can be simply expressed in algorithm 1.
Algorithm 1 The overall flow of cross-model matching
1:Initialize Lorder:Lcross-layout ≻ L title≻ L vision≻ L others
2:Initialize Label Priority : doc title ¿ title ¿ otherwise
3:Initialize :Maligned← B′
ordered
4:foreach masked label L∈ L orderdo
5: foreach pending box B∈ B pending-L do
6: Initialize Dmin← ∞
7: foreach ordered box B′∈ B′
ordered do
8: Dcurr←0
9: fork←1to 4do
10: Dcurr←Dcurr+wk·ϕk(B, B′)
11: ifDcurr> D minthen
12: break ▷filter out
13: end if
14: end for
15: ifDcurr< D minthen
16: Dmin←Dcurr,B′
best←B′
17: end if
18: end for
19: Maligned← M aligned∪ {(B, B′
best)}
20: end for
21: Reorder Maligned by (Index, Label Priority, Y1, X1)
22:end for
4. Experiments
In this section, we describe the experimental setup for eval-
uating our proposed method on the DocBench-100 bench-
mark. We compare our approach with several state-of-the-
art baselines and demonstrate its effectiveness through both
quantitative and qualitative analyses.
4.1. Dataset
We introduce DocBench-100, a novel benchmark dataset
constructed from the document detection dataset [10], to

1
23
4
5
6
7
8
9
10
11
12
1314
151617
18
19
20
21
22
23
8
91
2
3
4
5
6
7
10
11
12
1314
151617
18
19
20
21
22
2324
25
26
27
28
2930
31
32
337
91
2
3
4
5
6
8
11
10
13
1214
151617
18
19
20
21
22
2324
25
26
27
28
2930
31
32
3324
25
26
27
28
2930
31
32
33
 (a)Input         (b)XY-Cut   (c)MinerU   (d)XY-Cut++(Ours)Figure 7. Visualization of Complex Page Dcfrom DocBench-100 Dataset Using Different Layout Analysis Methods. (a) Input Image, (b)
XY-Cut[2], classic projection-based segmentation, (c) MinerU[12], an end-to-end Document Content Extraction tool, (d) XY-Cut++, our
proposed method.
1
23
4
5
63
41
2
5
63
41
2
5
6
(a)Input         (b)XY-Cut   (c)MinerU   (d)XY-Cut++(Ours) 
Figure 8. Visualization of Regular Page Drfrom DocBench-100 Dataset Using Different Layout Analysis Methods. (a) Input Image, (b)
XY-Cut[2], a classic projection-based segmentation, (c) MinerU[12], an end-to-end Document Content Extraction tool, (d) XY-Cut++, our
proposed method.
address the lack of a standardized dataset for evaluating
Document Reading Order Recovery tasks. Existing bench-
marks fail to adequately cover the diverse layout complex-
ities encountered in real-world documents. DocBench-100
consists of two subsets: the complex subset ( Dc) and the
regular subset ( Dr), reflecting varying levels of layout com-
plexity found in everyday documents.
Complex Subset ( Dc):This subset contains 30 pages
sourced from newspapers, books, and textbooks. The
column distribution is 3.3% single-column, 6.7% double-
column, 90% multi-column layouts (with ≥3columns and
irregular document titles). These features make Dcpar-
ticularly challenging and representative of complex, multi-
column layouts typical of newspapers and scientific jour-
nals.
Regular Subset ( Dr):This subset contains 70 pages
sourced from academic papers, books, textbooks, exam pa-
pers, slides, financial reports, notebooks, and magazines.The column distribution is 38.6% single-column, 54.4%
double-column, and 7% three-column layouts. This sub-
set represents more common and simpler layouts typically
found in academic papers and books.
Dataset Composition: To reflect the prevalence of sim-
pler layouts in daily use, we set the ratio of DrtoDcat
7:3. The entire dataset comprises 100 original page im-
ages, each accompanied by a ground truth (GT) JSON file
and an input JSON (No Index) file. The GT files contain
essential information, including:
•page id: A unique identifier for each page.
•page size : The dimensions of the page.
•bbox : Bounding boxes for each element on the page.
•label : Class labels for different types of content (e.g.,
text, figure).
•index : The reading order index for each element.
All reading orders have been manually verified to ensure
their accuracy.

Table 1. Progressive Component Analysis on DocBench-100. Metric Key: BLEU-4 ↑/ ARD ↓/ Tau ↑.
Method Dc Dr µ
BLEU-4 ↑ARD↓Tau↑BLEU-4 ↑ARD↓Tau↑BLEU-4 ↑ARD↓Tau↑
XY-Cut 0.749 0.233 0.878 0.819 0.098 0.912 0.797 0.139 0.902
+Pre-Mask 0.818 0.196 0.887 0.823 0.087 0.920 0.822 0.120 0.910
+MGS 0.946 0.164 0.969 0.969 0.036 0.985 0.962 0.074 0.980
+CMM 0.986 0.023 0.995 0.989 0.003 0.997 0.988 0.009 0.996
Table 2. Ablation Study of Pre-Mask, Multi-Granularity Segmentation (MGS), and Cross-Modal Matching (CMM) on DocBench-
100.Metric Key: BLEU-4 ↑.
Method Mask Mask Cross-Layout Pre-Cut Adaptive Scheme ϕ1ϕ2ϕ3ϕ4Dynamic Weights Multi-Stage BLEU-4 ↑
Baseline 0.797
+Pre-Mask ✓ 0.822
+MGS✓ ✓ 0.905
✓ ✓ 0.914
✓ ✓ 0.923
✓ ✓ ✓ ✓ 0.962
+CMM✓ ✓ ✓ ✓ ✓ 0.963
✓ ✓ ✓ ✓ ✓ ✓ 0.765
✓ ✓ ✓ ✓ ✓ ✓ 0.858
✓ ✓ ✓ ✓ ✓ ✓ ✓ 0.985
✓ ✓ ✓ ✓ ✓ ✓ 0.881
✓ ✓ ✓ ✓ ✓ ✓ 0.694
✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ 0.988
Table 3. Reading Order Recovery Performance: BLEU-4 ↑Re-
sults on DocBench-100.The best results are in bold.
Method Dc Dr µ
XY-Cut [2] 0.749 0.818 0.797
LayoutReader [3, 13] 0.656 0.844 0.788
MinerU [12] 0.701 0.946 0.873
XY-Cut++(ours) 0.986 0.989 0.988
Necessity of DocBench-100: The introduction of
DocBench-100 fills a critical gap in the field by providing a
comprehensive dataset that covers both simple and complex
layouts. This diversity allows for robust evaluation of read-
ing order recovery models across various scenarios. The in-
clusion of detailed ground truth (GT) annotations facilitates
direct testing and comparison with other models, thereby
promoting reproducibility and advancing further research
in this domain.
4.2. Setup
Baselines We compare our method with the following
state-of-the-art approaches:
•XY-Cut [2]: A classic method for projection-based seg-
mentation.•LayoutReader [3, 13]: A LayoutLMv3-based model fine-
tuned on 500k samples.
•MinerU [12]: An end-to-end document content extrac-
tion tool.
Evaluation Metrics We evaluate the performance using
the following metrics:
•BLEU-4 [9] (↑): Measures the similarity between candi-
date and reference texts using up to 4-gram overlap.
•ARD (↓): Absolute Relative Difference, quantifies predic-
tion accuracy by comparing predicted and actual values.
•Tau (↑): Kendall’s Tau, measures the rank correlation
between two sets of data.
•FPS (↑): Frames Per Second, a measure of how many
frames a system can process per second.
4.3. Main Results
We evaluate our method on DocBench-100, analyzing com-
ponent contributions and comparing against state-of-the-
art baselines. All metrics are computed over the union of
DcandDr, subsets unless specified.
4.3.1. Progressive Component Ablation
Table 1 demonstrates the cumulative impact of each techni-
cal component:

Table 4. Reading Order Recovery Performance on Textual Content of OmniDocBench (Excluding Figures/Tables with Insignificant Layout
Impact). Metric Key: BLEU-4 ↑/ ARD ↓/ Tau ↑. The red color indicates the best results and the blue color indicates the second-best
results.
Method Single Double Three Complex Mean
BLEU-4 ARD Tau BLEU-4 ARD Tau BLEU-4 ARD Tau BLEU-4 ARD Tau BLEU-4 ARD Tau
XY-Cut [2] 0.895 0.042 0.931 0.695 0.230 0.794 0.702 0.090 0.923 0.717 0.120 0.866 0.753 0.118 0.878
LayoutReader [3, 13] 0.988 0.004 0.995 0.831 0.084 0.918 0.595 0.208 0.805 0.716 0.116 0.864 0.783 0.099 0.906
MinerU [12] 0.961 0.025 0.969 0.933 0.037 0.971 0.923 0.042 0.965 0.887 0.050 0.932 0.926 0.039 0.959
XY-Cut++(ours) 0.993 0.004 0.996 0.951 0.027 0.974 0.967 0.033 0.984 0.901 0.064 0.942 0.953 0.037 0.972
Table 5. Model Efficiency and Semantic Information Usage on DocBench-100 and OmniDocBench. Key Metrics: FPS (Total Pages/Total
Times) ↑. FPS values are averaged over 10 runs on Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz with 256GB memory. The red color
indicates the best results and the blue color indicates the second-best results.
Method Semantic Info FPS
DocBench-100 OmniDocBench Mean
XY-Cut [2] ✗ 685 289 487
LayoutReader [3, 13] ✗ 17 27 22
MinerU [12] ✗ 10 12 11
XY-Cut++(ours) ✓ 781 248 514
•XY-Cut Baseline : Achieves 0.749 BLEU-4 on complex
layouts ( Dc), showing limitations in handling complex-
layout(e. g. L-shape) elements.
•+Mask : Improves BLEU-4 by 6.9 points (from 0.749 to
0.818) on Dcvia adaptive thresholding (Equation (1),
β= 1.3), reducing false splits by 15.9%.
•+MGS : Delivers a 19.7 absolute BLEU-4 gain on Dc
through three-phase segmentation, with density-aware
splitting (Equation (5)) reducing ARD by 29.7%.
•+CMM : Achieves near-perfect alignment (0.995 τ) onDc
through geometric constraints (Equations (7)-(10)), final-
izing a 90.1% ARD reduction from baseline.
The complete model reduces ARD by 93.5% compared
to baseline (0.139 →0.009), demonstrating superior rank-
ing consistency. Notably, our method maintains balanced
performance across both subsets (0.988 µ-BLEU), proving
effective for diverse layout types.
4.3.2. Architectural Analysis
We perform a systematic dissection of core components
through controlled ablations to evaluate their contribu-
tions:
Pre-Mask Processing (Pre-Mask): To alleviate the
”L-shaped” problem, we applied preliminary masking on
highly dynamic elements such as titles, tables, and figures.
This approach reduced visual noise and improved reading
order recovery, resulting in a 2.5-point BLEU-4 score in-
crease, as shown in Table 2.
Multi-Granularity Segmentation (MGS): Mask cross-
layout is a very direct method to solve the problem thatXY-Cut cannot segment L-shaped input. Pre-Cut real-
izes preliminary sub-page division through page analysis,
thereby avoiding page content mixing affecting sorting. The
adaptive splitting strategy enables reasonable segmenta-
tion through real-time density estimation. Table 2 shows
additive benefits of the mask cross-layout (+8.3), Pre-Cut
(+9.2), and the adaptive splitting scheme (+10.1) over the
baseline (82.2 BLEU-4).
Cross-Modal Matching (CMM): As shown in Table 2,
the single-stage strategy performs comparably to the base-
line approach. This suggests that employing a detection
model equipped with text, title, and annotation labels is
sufficient to achieve nearly consistent performance across
various scenarios. Notably, on the OmniDocBench dataset,
which has few label categories, our method still achieves
state-of-the-art results. Furthermore, we observe that the
edge-weighted margin distance plays a crucial role among
the four distance metrics examined. This finding highlights
the significance of dynamic weights based on shallow se-
mantics. In contrast, implementing cross-modal matching
results in an overall improvement of 2.7 points in BLEU-4
scores.
4.3.3. Benchmark Comparison
As shown in Table 3, our approach establishes new bench-
marks across all evaluation dimensions. Specifically, it out-
performs XY-Cut by a significant margin, achieving a +23.7
absolute improvement on Dc(from 74.9 to 98.6). Addition-
ally, it surpasses LayoutReader by +5.3 on Dr(from 94.6
to 98.9), despite not using any learning-based components.

Furthermore, our method achieves a Kendall’s τof 0.996
overall, indicating near-perfect ordinal consistency (with
p <0.001in the Wilcoxon signed-rank test). Visual results
presented in Fig. 7 and 8 further demonstrate the robustness
in handling multi-column layouts and cross-page elements,
where previous methods frequently fail.
To further validate the versatility and robustness of our
proposed method, we conducted extensive evaluations on
the OmniDocBench dataset [8], which features a diverse
and challenging set of document images. As shown in Ta-
ble 4, our proposed method (XY-Cut++) achieves state-of-
the-art (SOTA) performance across almost all layout types,
despite challenging subpage nesting patterns (see Limita-
tions). Notably, as shown in Table 5, XY-Cut++ achieves
a superior balance between performance and speed, at-
taining an average FPS of 514 across DocBench-100 and
OmniDocBench. This performance surpasses even the di-
rect projection-based XY-Cut algorithm, which achieves an
average FPS of 487. The significant speed improvement
of XY-Cut++ is primarily attributed to semantic filtering,
which minimizes redundant processing by handling each
block only once. In contrast, XY-Cut requires repeatedly
partitioning blocks into different subsets, resulting in in-
creased recursive depth and computational overhead. This
optimization enhances computational efficiency without los-
ing performance, making XY-Cut++ more robust and versa-
tile for diverse document layouts.
5. Conclusion
In this work, we address the critical challenges in docu-
ment layout recovery and semantic conversion, particularly
focusing on the needs of Retrieval-Augmented Generation
(RAG) and Large Language Models (LLMs) for accurate in-
formation extraction. We introduce a hierarchical document
parsing framework that incorporates a Mask mechanism-
based layout recovery algorithm. This framework en-
ables precise reading order recovery through pre-mask pro-
cessing, multi-granularity segmentation, and cross-modal
matching. Additionally, it achieves superior performance
on the DocBench-100 and OmniDocbench datasets, out-
performing baseline methods and providing robust support
for downstream RAG and LLM tasks. It is noteworthy that
while traditional methods are known for their computa-
tional efficiency, they often fall short in performance. Our
approach not only maintains simplicity and efficiency but
also matches or surpasses the performance of deep learn-
ing methods. This advantage is primarily attributed to two
key innovations: (1) the strategic use of shallow semantic
labels as structural priors during layout analysis, and (2) a
hierarchical mask mechanism that effectively captures the
document’s topological structure. These innovations not
only enhance performance but also maintain model effi-
ciency, suggesting promising directions for improving deeplearning models in layout recovery tasks. We believe our
work can inspire new approaches to tackle the challeng-
ing tasks in document understanding. Our code will be
released at: https://github.com/liushuai35/
PaddleXrc .
6. Limitations and Future Work
Our method currently struggles to handle pages with nested
semantic structures, like sub-pages within a single page, as
it cannot effectively sort elements within these complex lay-
outs. To tackle this, we plan to enhance our framework by
adding a sub-page detection module. This module will iden-
tify sub-page regions, enabling element sorting within them
while maintaining the natural reading order (top-to-bottom,
left-to-right) for the sub-pages themselves. By balancing
global and local sorting, this approach aims to enhance the
robustness of our method for complex page structures.
References
[1] Zhangxuan Gu, Changhua Meng, Ke Wang, Jun Lan,
Weiqiang Wang, Ming Gu, and Liqing Zhang. Xylayoutlm:
Towards layout-aware multimodal networks for visually-rich
document understanding. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition ,
pages 4583–4592, 2022. 2, 3
[2] Jaekyu Ha, Robert M Haralick, and Ihsin T Phillips. Re-
cursive xy cut using bounding boxes of connected compo-
nents. In Proceedings of 3rd International Conference on
Document Analysis and Recognition , pages 952–955. IEEE,
1995. 1, 2, 6, 7, 8
[3] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu
Wei. Layoutlmv3: Pre-training for document ai with unified
text and image masking. In Proceedings of the 30th ACM
International Conference on Multimedia , pages 4083–4091,
2022. 2, 3, 7, 8
[4] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming
Zhou, and Zhoujun Li. Tablebank: Table benchmark for
image-based table detection and recognition. In Proceedings
of the Twelfth Language Resources and Evaluation Confer-
ence, pages 1918–1925, 2020. 3
[5] Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei,
Zhoujun Li, and Ming Zhou. Docbank: A benchmark dataset
for document layout analysis. In Proceedings of the 28th
International Conference on Computational Linguistics . In-
ternational Committee on Computational Linguistics, 2020.
3
[6] Ke Ma, Zhixin Shu, Xue Bai, Jue Wang, and Dimitris Sama-
ras. Docunet: Document image unwarping via a stacked u-
net. In Proceedings of the IEEE conference on computer
vision and pattern recognition , pages 4700–4709, 2018. 3
[7] J-L Meunier. Optimized xy-cut for determining a page read-
ing order. In Eighth International Conference on Docu-
ment Analysis and Recognition (ICDAR’05) , pages 347–351.
IEEE, 2005. 2

[8] Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui
Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang,
Xiaomeng Zhao, et al. Omnidocbench: Benchmarking di-
verse pdf document parsing with comprehensive annotations.
arXiv preprint arXiv:2412.07626 , 2024. 1, 3, 9
[9] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing
Zhu. Bleu: a method for automatic evaluation of machine
translation. In Proceedings of the 40th annual meeting of the
Association for Computational Linguistics , pages 311–318,
2002. 7
[10] Ting Sun, Cheng Cui, Yuning Du, and Yi Liu. Pp-
doclayout: A unified document layout detection model to
accelerate large-scale data construction. arXiv preprint
arXiv:2503.17213 , 2025. 3, 4, 5
[11] Phaisarn Sutheebanjard and Wichian Premchaiswadi. A
modified recursive xy cut algorithm for solving block or-
dering problems. In 2010 2nd International Conference
on Computer Engineering and Technology , pages V3–307.
IEEE, 2010. 2
[12] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang,
Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu,
Fukai Shang, et al. Mineru: An open-source solution
for precise document content extraction. arXiv preprint
arXiv:2409.18839 , 2024. 6, 7, 8
[13] Zilong Wang, Yiheng Xu, Lei Cui, Jingbo Shang, and Furu
Wei. Layoutreader: Pre-training of text and layout for read-
ing order detection. In Proceedings of the 2021 Confer-
ence on Empirical Methods in Natural Language Processing ,
pages 4735–4744, 2021. 1, 3, 7, 8
[14] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei,
and Ming Zhou. Layoutlm: Pre-training of text and layout
for document image understanding. In Proceedings of the
26th ACM SIGKDD international conference on knowledge
discovery & data mining , pages 1192–1200, 2020. 3
[15] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu
Zhang, and Furu Wei. Layoutlmv2: Multi-modal pre-
training for visually-rich document understanding. In Pro-
ceedings of the 59th Annual Meeting of the Association for
Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1:
Long Papers) , pages 3452–3464, 2021. 3
[16] Yiheng Xu, Tengchao Lv, Lei Cui, Guoxin Wang, Yijuan
Lu, Dinei Florencio, Cha Zhang, and Furu Wei. Layoutxlm:
Multimodal pre-training for multilingual visually-rich doc-
ument understanding. arXiv preprint arXiv:2104.08836 ,
2021. 3
[17] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei,
Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang,
Wanxiang Che, et al. Layoutlmv2: Multi-modal pre-training
for visually-rich document understanding. In Proceedings
of the 59th Annual Meeting of the Association for Compu-
tational Linguistics and the 11th International Joint Confer-
ence on Natural Language Processing (Volume 1: Long Pa-
pers) . Association for Computational Linguistics, 2021. 3
[18] Fisher Yu, Dequan Wang, Evan Shelhamer, and Trevor
Darrell. Deep layer aggregation. In Proceedings of the
IEEE conference on computer vision and pattern recogni-
tion, pages 2403–2412, 2018. 3[19] Jiaxin Zhang, Dezhi Peng, Chongyu Liu, Peirong Zhang,
and Lianwen Jin. Docres: a generalist model toward uni-
fying document image restoration tasks. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 15654–15664, 2024. 3
[20] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Pub-
laynet: largest dataset ever for document layout analysis.
In2019 International conference on document analysis and
recognition (ICDAR) , pages 1015–1022. IEEE, 2019. 3
[21] Xinyu Zhou, Cong Yao, He Wen, and Yuzhi Wang. East:
An efficient and accurate scene text detector. In Proceed-
ings of the IEEE Conference on Computer Vision and Pattern
Recognition , pages 2642–2651, 2017. 3