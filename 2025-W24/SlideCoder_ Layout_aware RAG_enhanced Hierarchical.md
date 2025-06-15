# SlideCoder: Layout-aware RAG-enhanced Hierarchical Slide Generation from Design

**Authors**: Wenxin Tang, Jingyu Xiao, Wenxuan Jiang, Xi Xiao, Yuhang Wang, Xuxin Tang, Qing Li, Yuehe Ma, Junliang Liu, Shisong Tang, Michael R. Lyu

**Published**: 2025-06-09 17:39:48

**PDF URL**: [http://arxiv.org/pdf/2506.07964v1](http://arxiv.org/pdf/2506.07964v1)

## Abstract
Manual slide creation is labor-intensive and requires expert prior knowledge.
Existing natural language-based LLM generation methods struggle to capture the
visual and structural nuances of slide designs. To address this, we formalize
the Reference Image to Slide Generation task and propose Slide2Code, the first
benchmark with difficulty-tiered samples based on a novel Slide Complexity
Metric. We introduce SlideCoder, a layout-aware, retrieval-augmented framework
for generating editable slides from reference images. SlideCoder integrates a
Color Gradient-based Segmentation algorithm and a Hierarchical
Retrieval-Augmented Generation method to decompose complex tasks and enhance
code generation. We also release SlideMaster, a 7B open-source model fine-tuned
with improved reverse-engineered data. Experiments show that SlideCoder
outperforms state-of-the-art baselines by up to 40.5 points, demonstrating
strong performance across layout fidelity, execution accuracy, and visual
consistency. Our code is available at
https://github.com/vinsontang1/SlideCoder.

## Full Text


<!-- PDF content starts -->

arXiv:2506.07964v1  [cs.CV]  9 Jun 2025
SlideCoder: Layout-aware RAG-enhanced Hierarchical Slide
Generation from Design
Wenxin Tang1*, Jingyu Xiao2*, Wenxuan Jiang3, Xi Xiao1, Yuhang Wang4,
Xuxin Tang5, Qing Li6, Yuehe Ma7, Junliang Liu8, Shisong Tang5, Michael R. Lyu2
1Tsinghua University,2The Chinese University of Hong Kong,3Northeastern University
4Southwest University,5Kuaishou Technology,6Peng Cheng Laboratory
7BNU-HKBU United International College,8Dalian Maritime University
twx24@mails.tsinghua.edu.cn, jyxiao@link.cuhk.edu.hk, xiaox@sz.tsinghua.edu.cn
Abstract
Manual slide creation is labor-intensive and re-
quires expert prior knowledge. Existing nat-
ural language-based LLM generation meth-
ods struggle to capture the visual and struc-
tural nuances of slide designs. To address
this, we formalize the Reference Image to
Slide Generation task and propose Slide2Code,
the first benchmark with difficulty-tiered sam-
ples based on a novel Slide Complexity Met-
ric. We introduce SlideCoder, a layout-aware,
retrieval-augmented framework for generat-
ing editable slides from reference images.
SlideCoder integrates a Color Gradient-based
Segmentation algorithm and a Hierarchical
Retrieval-Augmented Generation method to
decompose complex tasks and enhance code
generation. We also release SlideMaster, a 7B
open-source model fine-tuned with improved
reverse-engineered data. Experiments show
that SlideCoder outperforms state-of-the-art
baselines by up to 40.5 points, demonstrating
strong performance across layout fidelity, ex-
ecution accuracy, and visual consistency. Our
code is available at https://github.com/
vinsontang1/SlideCoder .
1 Introduction
Slide creation is essential in academic and pro-
fessional communication for visually conveying
complex ideas. However, manual design is labor-
intensive and time-consuming (Al Masum et al.,
2005). While templates offer some relief, they en-
force fixed layouts and styles, limiting flexibility.
Recent progress in Large Language Models
(LLMs) (Nam et al., 2024; Ge et al., 2023) has
sparked interest in automatic slide creation. Au-
toPresent (Ge et al., 2025), an early study on
the Natural Language (NL) to slide generation
task, fine-tunes a LLAMA-based model (Grattafiori
et al., 2024) on the diversified SLIDESBENCH
*These authors contributed equally.
The design of this 
image is great.  Help 
me convert it into a 
editable  slide.
Here is the python 
code for generating 
the slide. 
Figure 1: Illustration of slide generation scenarios from
design and mistakes made by MLLMs.
dataset. It translates NL instructions into Python
code, which invokes SLIDESLIB, a high-level API
built on python-pptx (Canny, 2023), to construct
each slide. This pipeline reduces manual effort and
streamlines design workflows.
Despite Autopresent’s capability to generate
slides from natural language input, several signifi-
cant challenges remain unaddressed.
First, natural language inherently lacks an
accurate description of slide visual design (e.g.,
color, layout, and style) and users sometimes
directly input the design image for slide gener-
ation. For example, as shown in Figure 1, a user
sees a nice design from non-editable slides (png
and pdf format) or other source like webpage de-
sign, and hopes to convert it into an editable slide
(pptx format). Or the user lacks the skills to make
slides, they can generate the slide by input their
design image. In these scenarios, the Multimodal
Large Language Models (MLLMs) are needed to
understand the design and generate slides.
Second, MLLMs face limitations when han-
dling complex slides, particularly those incorpo-
rating diverse element types and high element
density. As illustrated in Figure 1, these discrep-
ancies can be divided into three categories: miss,
which stands for the complete omission of certain

visual or textual elements (e.g., the top left corner
of the shape is missing); incorrect , referring to de-
viations in visual styles or attributes from those
specified or expected in the reference slides (e.g.,
title is not bold); and disorder , which describes
significant differences in spatial arrangements and
alignment of elements compared to the original lay-
out (e.g., the three subheadings are not properly
positioned and aligned.).
Third, MLLMs’ insufficient comprehension
of the python-pptx library leads to the genera-
tion of syntactically invalid or non-executable
code. Autopresent (Ge et al., 2025) attempts to
address this issue by constructing SLIDESLIB, a
simplified library built upon python-pptx, encap-
sulating commonly used operations into a set of
high-level APIs. However, this operation inher-
ently restricts the flexibility and comprehensive-
ness of slide generation. Specifically, SLIDESLIB
currently supports only five basic operation types,
which neglects more intricate layouts and design
requirements commonly encountered in realistic
scenarios. Consequently, presentations produced
by this approach tend to be overly simplistic, inad-
equately capturing complex human intentions and
detailed visual expectations.
To address the aforementioned limitations, we in-
troduce SlideCoder, a layout-aware RAG-enhanced
hierarchical slide generation framework, which can
understand the complex slides and python-pptx li-
brary accurately. First, we formulate a novel task,
Reference Image (RI) to slide generation , i.e.,
automatically generating the code for replicating
the slide, which is visually consistent with RI. To
evaluate the performance of SlideCoder under com-
plex slide scenarios, we propose a novel Slide
Complexity Metric (SCM), and construct a new
benchmark Slide2Code with different difficulty lev-
els based on SCM. Second, we develop a novel
Color Gradients-based Segmentation algorithm
(CGSeg ) that effectively decomposes slide images
into semantically meaningful regions. Besides,
we propose the Layout-aware Prompt , which in-
tegrates the position information of elements to
enhance MLLM’s understanding of slide layout.
Third, we propose a novel Hierarchical Retrieval-
Augmented Generation (H-RAG)-based Code
Generation method, which employs a dual-level
retrieval-augmented knowledge base (Cuconasu
et al., 2024; Fan et al., 2024) to explicitly enhance
MLLMs’ understanding of the python-pptx library.
At the higher level, a Shape Type Knowledge Base(TS-KB) systematically classifies slide elements
and standardizes their descriptions using python-
pptx API terminologies. At the lower level, a Oper-
ation Function Knowledge Base (OF-KB) captures
precise syntactic patterns and invocation paradigms
of python-pptx library functions.
To further enhance the MLLM’s ability to gen-
erate high-quality slides, we build a PPTX reverse-
engineering tool to construct high quality training
data for fine-tuning a 7B model SlideMaster based
on Qwen-VL-7B (Bai et al., 2025), which can ap-
proaches the performance of the closed-sourced
model GPT-4o (Achiam et al., 2023). Our contri-
butions are summarized as follows:
•We define reference image (RI) to slide gener-
ation task and propose a novel Slide Complex-
ity Metric (SCM), based on which we con-
struct Slide2Code, the first difficulty-leveled
benchmark with 300 samples.
•We propose SlideCoder, which consists of a
novel Color Gradients-based Segmentation al-
gorithm (CGSeg), a Layout-aware Prompt and
a Hierarchical Retrieval-Augmented Genera-
tion (H-RAG)-based Code Generation method
for enhancing the MLLM’s understanding on
the complex slides and python-pptx library.
•We train SlideMaster, a 7B open-source model
approaching the performance of GPT-4o. To
enable effective fine-tuning, we also build
a comprehensive PPTX reverse-engineering
tool for precise code generation.
2 Related Work
2.1 Multimodal Large Language Models for
Code Generation
The multimodal large model demonstrates excel-
lent capabilities in visually rich code generation
scenarios, such as UI code generation (Xiao et al.,
2024, 2025; Yun et al., 2024; Wan et al., 2024),
SVG code generation (Rodriguez et al., 2025;
Nishina and Matsui, 2024; Wu et al., 2024; Xing
et al., 2024), and visually rich programming ques-
tions (Li et al., 2024; Zhang et al., 2024a; Ma et al.,
2025). However, MLLMs are not yet capable of
plug-and-play use across tasks and still produce
subtle errors, therefore, some studies explore their
code repair abilities (Yang et al., 2024; Yuan et al.,
2024; Zhang et al., 2024b).

2.2 Slide Generation and Understanding
Previous work on slide generation has predomi-
nantly focused on basic content extraction from
input documents. With the recent advancements
in large language models (Fu et al., 2022; Hu and
Wan, 2014; Kan, 2007; Sefid and Wu, 2019), sev-
eral studies have begun to explore LLM-based slide
generation. For example, (Zheng et al., 2025) uti-
lizes LLMs to generate slides based on pre-defined
slide templates and user-provided text. (Ge et al.,
2025) introduces the task of natural language (NL)
to slide code generation, aiming to organize visual
slide content through textual input. However, its
use of coarse-grained natural language descriptions
and a native agent design significantly limits the
quality of the generated slides.
3 Slide2Code Benchmark
We construct the Slide2Code benchmark to evalu-
ate the performance of multimodal large language
models (MLLMs) on the Reference Image (RI) to
slide generation task. Each instance includes a ref-
erence slide image and its corresponding PPTX
slide. Slide2Code enables comparison of MLLM
backbones under varying complexity. §3.1 formally
defines the task, §3.2 describes our unified com-
plexity scoring system based on element quantity,
diversity, and visual density, and §3.3 details data
collection and sampling.
3.1 Task Description
This work addresses the task of Reference Image
(RI) to slide generation, where the input is a slide’s
reference image I0and the goal is to generate
Python code using the python-pptx library. Let
F0denote the original slide file corresponding to
I0. Given a generation framework Gand Multi-
modal Large Language Models (MLLMs) M, the
generated code Cg=GM(I0)can be executed to
obtain a new slide file Fg, whose rendered image
is denoted as Ig. As the original code C0forF0is
unavailable, we assess the performance of Gand
Mby comparing (I0, F0)and(Ig, Fg).
3.2 Slide Complexity Metric
To evaluate slide complexity, we propose a Tri-
Metric Slide Complexity Metric (SCM) that inte-
grates production difficulty and visual complexity.
Due to the mismatch between visual appearance
and construction effort, for example, inserting a
visually complex image may require minimal op-erations. To adress this, we assess slides using:
(1) element count, (2) element type count (e.g.,
textbox, placeholder), and (3) Element Coverage
Ratio. The first two reflect operational cost, the
third captures visual richness. Since reference com-
plexity labels are not available, we evaluate the
relative complexity of sample iwithin a collection
Y={1,2,3, ..., N}.
Letcibe the number of elements and eithe num-
ber of distinct element types in sample i. The El-
ement Coverage Ratio viis the proportion of ac-
tivated color grids to total grids in the image of
sample i, computed via the gradient-based segmen-
tation algorithm CGSeg (see §4.1 for details).
Each raw dimension score xi∈ {ci, ei, vi}is
normalized as ˜xi=σ
xi−µ√
σ2+ϵ
, where µandσ2
denote the mean and variance over all samples in
setY, respectively. Here, σ(·)is the sigmoid func-
tion (Han and Moraga, 1995), and ϵis a small
constant for numerical stability. The final complex-
ity score for slide iis computed via a weighted
aggregation: zi=α·˜ci+β·˜ei+γ·˜vi, where
α+β+γ= 1and the weights α, β, γ reflect the im-
portance of production effort and visual complexity.
This metric shows a strong correlation with human
judgment, as detailed in Section §5.4.
3.3 Data Collection
To construct a comprehensive benchmark that cap-
tures diverse slide characteristics, we randomly
sample approximately 32,000 Zenodo10k (Zheng
et al., 2025) slide instances, the largest publicly
available slide dataset, to construct the slide set Y
as described in §3.2. To enhance diversity and al-
low comparative analysis, we additionally incorpo-
rate SLIDEBENCH samples in Y. This unified set
is then used to calculate the normalized complexity
scores zfor all slides. KMeans algorithm is used
to obtain three clusters, whose cluster centers are
sorted in order of zto define the simple, medium,
and complex levels. From each cluster, we ran-
domly select 100 representative samples from Yto
form the final Slide2Code benchmark.
Figure 2 shows that both Zenodo10k and
SLIDEBENCH contain a significantly larger pro-
portion of simple and medium slides. In contrast,
Slide2Code exhibits a more balanced composition
across all three levels, allowing a more equitable
evaluation of slide generation models under vary-
ing structural and visual complexities.

0 20 40 60 80 100
Proportion (%)SLIDESBENCHZenodo10kSlide2Code
41.1% 42.0% 16.9%25.7% 43.1% 31.1%33.1% 33.4% 33.4%Simple Medium ComplexFigure 2: Proportion of samples across three levels in the
Slide2Code, Zenodo10k, and SLIDEBENCH datasets.
4 Methodology
In this section, we introduce SlideCoder, a uni-
fied end-to-end framework for generating Python-
executable slide code from reference images (RIs).
We assume a scenario where a user provides a de-
sign layout (" Design ") and embedded visual ele-
ments such as pictures or background images (" Pic-
tures "). SlideCoder comprises three core mod-
ules. First, a Color Gradients-based Segmenta-
tion (CGSeg) algorithm segments the input Design
into semantically meaningful regions. Second, a
Hierarchical Retrieval-Augmented Code Gen-
eration module, consisting of three collaborative
agents Describer ,Coder , and Assembler , gener-
ates the slide code. Third, a Layout-aware Prompt
mechanism enhances the Assembler agent to en-
sure spatial consistency and syntactic correctness.
Finally, based on this framework, we fine-tune a
7B open-source model, named SlideMaster.
4.1 Color Gradient-based Segmentation
To reduce the difficulty of MLLM in understand-
ing complex slide design, we proposed CGSeg, a
recursive color gradient-based segmentation algo-
rithm to divide slide design into blocks. As shown
in Algorithm 1, CGSeg starts by dividing the input
image (Figure 4a) into a grid and computing the So-
bel magnitude for each block to measure the inten-
sity of the color gradient (lines 4–5). Blocks with
gradient magnitudes significantly higher than the
median are marked as activated block (lines 6–14),
as visualized in Figure 4b. To group visually co-
herent regions, CGSeg applies a flood-fill (Burtsev
and Kuzmin, 1993) operation to the binary activa-
tion mask (line 15), identifying connected regions
corresponding to sub-images (line 16), as shown in
Figure 4c. These sub-images are further segmented
recursively to ensure a hierarchical decomposition
of the image Im, along with the corresponding
positional information pm(lines 1–3 and 17–23),Algorithm 1 Color Gradient-based Segmentation
(CGSeg)
Require: Image I, Grid size g, Depth D, Max depth Dmax,
Threshold T
Ensure: List of segmented sub-images
1:ifD=Dmaxthen
2: return ∅
3:end if
4:G←SPLIT (I, g) //g×ggrid blocks
5:C←GRADMAG(G) // gradient magnitudes
6:Cmid←MEDIAN (C)
7:M←0g×g// binary mask
8:foreachcijinCdo
9: ifcij> T·Cmidthen
10: Mij←1 // activate the block
11: else
12: Mij←0
13: end if
14:end for
15:M←FILL(M) // flood-fill
16:Ms←REGIONS (M)// split connected regions
17:R← ∅
18:foreachminMsdo
19: Im, pm←CROP(I, m ) // get sub-image
20: add ImandpmtoR
21: R′←CGS EG(Im, g, D +1, Dmax, T)
22: add all in R′toR
23:end for
24:return R
with the final segmentation result shown in Fig-
ure 4d. This recursive structure allows CGSeg to
adaptively refine segment granularity based on lo-
cal visual complexity, which is crucial for handling
slides with heterogeneous layout densities.
4.2 Hierarchical Retrieval-Augmented Code
Generation Module
4.2.1 Generation Process
We design three collaborative MLLM agents whose
code generation processes are augmented by H-
RAG. Describer is responsible for generating a
global Design description (Overall Description) as
well as block descriptions (Block Description) for
each segmented blocks. Based on block and their
associated block description, Coder produces cor-
responding code snippets. Subsequently, Assem-
bler generates the complete slide code by layout-
aware prompt, which will be elaborated in §4.3,
along with the Pictures provided. Executing this
code produces a slide that structurally and visually
aligns with the Reference Image(RI). If the gener-
ated code is not executable Assembler applies a
self-refinement mechanism to correct syntax errors,
where errors serves as the feedback to prompt the
MLLM to re-generate the code.
Beyond the above inputs, each agent draws
knowledge from distinct bases according to its role.

Color Gradient-based SegmentationHierarchical RAG-based Code GenerationThis textbox’s content is “understanding ...Overall Description
AssemblerLayout-awarePromptYouareapython-pptxexpert.BasedontheinformationandcodesnippetsIprovide,pleaseassembleacompletepython-pptxscript:<Design>referstothereferenceimageforthisslide.Itsglobaldescriptionis<OverallDescription>.Thecodesnippetsandtheirlayoutpositionsaregivenas<CodeSnippets>,<Position*>.Herearesomesyntaxrulesthatmightbeuseful:<Grammar>.Thebackgroundandimagespathis...Pictures and PromptKnowledge BaseOperation Function
<Grammar>Coder
Describer
Blocks and PositionDesign
…<𝑥!,𝑦!,𝑤!,ℎ!><𝑥",𝑦",𝑤",ℎ"><𝑥#,𝑦#,𝑤#,ℎ#>
There is a paragraphrunsin textbox...This autoshapeincludes a textbox...This slide is titled "Everyday Objects ...Block Descriptionfrom pptx import Presentation…font.color.rgb = RGBColor()…from pptx import Presentation…textbox = slide.shapes.add_textbox…from pptx import Presentation…textbox = slide.shapes.add_textbox…Code Snippets
from pptx import Presentation…textbox = slide.shapes.add_textboxfont.color.rgb= RGBColor()slide.shapes.add_shape…Final Code
Slide
Shape Type
Figure 3: The framework of SlideCoder.
(a) Input Image
 (b) Activated Grid Blocks
 (c) Flood-filled Regions
 (d) Final result
Figure 4: An example of CGSeg applied to a slide reference image. The algorithm begins by computing color
gradients (a-b), fills them (c), and recursively segments sub-regions (d).
The form and origin of the knowledge used in each
agent’s prompt are detailed in §4.2.2.
4.2.2 Hierarchical Retrieval-Augmented
Generation
Hierarchical Retrieval-Augmented Generation(H-
RAG) comprises a Shape Type Knowledge Base
and an Operation Function Knowledge Base. The
former contains descriptions of objects from the
python-pptx documentation, used in Describer to
guide standardized description generation. For ex-
ample, in “This autoshape includes a textbox ...”,
both terms are object names from the documenta-
tion. The latter includes full syntax specifications
(e.g., parameters, return values, etc.). Appendix F
details their structure.
We employ BGE M3-Embedding (Chen et al.,
2024) to embed entries and build a vector-based
retrieval database. For a prompt p, its vector qpis
computed, and cosine similarity cos(qp, ki)is used
to match ki. The top- krelevant entries are inserted
intop. Given the size of the Shape Type Knowl-
edge Base, all entries are included in Describer to
ensure complete type coverage.
In the hierarchical pipeline, agents collaborate
progressively. Describer retrieves object types
from the Shape Type Knowledge Base to identify
elements in block images and output standardized
descriptions. Coder uses these to query the Opera-
tion Function Knowledge Base and generate codesnippets. Assembler uses these snippets to retrieve
full syntax patterns and generate executable code.
4.3 Layout-aware Prompt
After Coder completes the generation of code snip-
pets for blocks, Assembler is applied to assemble
these code snippets for generating the final slide in
an accurate manner. The assembly prompt needs
to meet the following requirements: (1) ensure that
each block appears in the correct position in the
final slide; (2) avoid syntax errors in the merged
code and ensure code context consistency.
To achieve above goals, layout-aware prompt in-
jects the layout position using python-pptx standard
positioning units (inches) to ensure the position
correctness and retrieve the grammar <Grammar>
from Knowledge Base to avoid syntax errors and
code conflicts. Since the resolution of the Design
differs from the actual slide layout size, we apply
proportional scaling to the Position (< x, y, w, h >)
extracted from Color Gradients-based Segmenta-
tion (CGSeg) algorithm to map it onto the slide co-
ordinates, denoted as <Position*> . Subsequently,
the reference image design <Design> , global body
description <Overall description.> , partial codes
<Code Snippets> from Coder , layout representa-
tion<Position*> , and syntactic patterns <Gram-
mar> retrieved from the Hierarchical Retrieval-
Augmented Generation(H-RAG) knowledge base
are integrated into a predefined prompt template

to construct the final layout-aware prompt (see Ap-
pendix E for details).
4.4 SlideMaster
Using the SLIDESBENCH training set, we con-
struct a dataset of (RI, instruction, program) triplets.
The reverse-engineering tool proposed by (Ge et al.,
2025) produces labels (Python code) for only a
limited set of slide styles, resulting in suboptimal
training data quality. To mitigate this, we develop
a new reverse-engineering tool capable of handling
a broader spectrum of slide styles, thereby enhanc-
ing label quality. The effectiveness of this tool is
analyzed in §5.3. We fine-tune our model, Slide-
Master, based on Qwen2.5-VL-7B-Instruct (Bai
et al., 2025), using LoRA (Hu et al., 2022). Full
configuration details are provided in Appendix C.
5 Experiments and Results
5.1 Experimental Setup
Model . To evaluate the performance of the Slide-
Coder, we employ state-of-the-art (SOTA) models,
including GPT-4o (Achiam et al., 2023), Gemini-
2.0-flash (Google, 2025), and SlideMaster, which
is a fine-tuned model based on the open-source
Qwen2.5-VL-7B-Instruct (Bai et al., 2025). The
SOTA models are accessed via their official APIs,
with GPT-4o using version 20241120 and Gemini-
2.0-flash accessed in May 2025. For both models,
the maximum token limit and temperature are set to
4096 and 0, respectively. Same as (Ge et al., 2025),
we allow both Coder andAssembler agents up to
three self-refinement attempt. The first successful
attempt is taken as the output. If Coder fails to gen-
erate executable code after the maximum number
of attempts, the corresponding block is discarded.
IfAssembler fails, the corresponding sample is
marked as execution failure.
Metric . To comprehensively assess generation
quality, we adopt four metrics, using the notations
defined in §3.1. (1) Global Visual Metrics , in-
cluding CLIP (Hessel et al., 2021) and SSIM (Nils-
son and Akenine-Möller, 2020) scores computed
between the original image I0and the generated
image Ig; (2) Local Structural Metrics , which
compare the original and generated slide files F0
andFgin terms of content similarity and position
similarity, following (Ge et al., 2025); (3) Execu-
tion, defined as the success rate of executing Cg
without errors; and (4) Overall Score , calculated as
the average of all metric values across all samples,with failed executions assigned a score of zero.
5.2 Quantitative Results and Analysis
The upper part of Table 1 presents the performance
of different frameworks on our proposed bench-
mark, evaluated using the metrics introduced in
Section 3.1. The results show that SlideCoder con-
sistently achieves the best performance across all
difficulty levels. Specifically, its overall score sur-
passes the best baseline by 40.5, 34.0, and 29.9
points on the simple, medium, and complex levels,
respectively, demonstrating the overall superior-
ity of our framework. For execution success rate,
SlideCoder outperforms the best baseline by 38%,
32%, and 27% across the three difficulty levels,
indicating that the proposed H-RAG and CGSeg
mechanisms significantly enhance model perfor-
mance and reduce task difficulty.
Moreover, SlideCoder outperforms all baselines
in both Local Structural Metrics and Global Visual
Metrics, confirming its strong fidelity in preserving
both the structural layout and visual appearance of
the original slides. The stepwise decline in Slide-
Coder’s overall score across increasing difficulty
levels further indicates its ability to leverage vi-
sual and structural cues from the input slides. In
contrast, baseline models relying solely on natu-
ral language descriptions exhibit weak sensitivity
to slide complexity, failing to reflect the difficulty
hierarchy in their overall scores.
On the SLIDESBENCH dataset (as shown in
the lower part of Table 1), SlideCoder also sur-
passes all baselines across all metrics, with an
overall score of 78.8 when using GPT-4o as the
backbone, representing a 11.9 improvement over
the best-performing baseline. Notably, the open-
source fine-tuned model SlideMaster also demon-
strates competitive performance, outperforming the
best GPT-4o-based baseline on both datasets.
5.3 Reverse Tool Analysis
Table 2 summarizes the supported object types and
corresponding styles in our proposed reverse engi-
neering tool. Our tool supports 10 commonly used
object types and 44 distinct object styles, whereas
Autopresent (Ge et al., 2025) only supports 5 object
types and 16 styles. Detailed comparisons can be
found in Appendix B. To quantitatively assess the
reverse engineering capabilities of both tools, we
adopt two evaluation metrics:
Reconstruction Ratio : This metric calculates
the ratio between the number of shapes in the slide

Table 1: Results on Slide2Code (top) and SLIDESBENCH (bottom) using SlideCoder and AutoPresent with
different MLLMs. Green ,yellow , and redindicate simple, medium, and complex levels in SlideCoder. Bolded
values mark the best result per level.
Framework Backbone Execution%Local Structural Metrics Global Visual MetricsOverallContent Position Clip SSIM
Slide2Code
AutoPresentAutoPresent61.0 92.7 78.9 70.8 80.3 48.6
53.0 89.6 77.3 69.2 79.1 41.4
67.0 87.2 71.4 65.9 73.4 48.5
Gemini2.0-flash57.0 91.4 78.3 69.7 79.0 44.8
68.0 88.7 79.9 66.3 71.6 51.5
66.0 89.3 72.2 63.1 64.7 45.2
GPT-4o58.0 92.7 80.9 68.8 75.6 45.4
50.0 92.3 74.6 67.6 72.6 36.8
69.0 90.3 73.3 62.3 63.3 47.1
SlideCoderSlideMaster86.0 92.4 87.4 77.6 91.1 76.7
75.0 84.7 79.8 75.4 86.4 61.7
73.0 76.1 70.5 72.4 82.8 54.2
Gemini2.0-flash97.0 94.5 88.6 81.3 90.7 87.0
90.0 90.9 84.6 82.3 85.5 76.6
88.0 92.7 80.9 81.7 81.2 71.6
GPT-4o99.0 96.3 88.1 79.8 91.8 89.1
100.0 92.5 84.7 81.5 86.2 85.5
96.0 94.3 80.0 80.7 82.6 78.4
SLIDESBENCH
AutoPresentAutoPresent 84.1 92.2 67.2 81.6 73.7 65.3
Gemini2.0-flash 56.4 91.7 62.9 77.1 66.0 40.4
GPT-4o 86.7 92.5 76.3 78.0 70.8 66.9
SlideCoderSlideMaster 87.2 91.5 76.9 73.4 80.0 68.4
Gemini2.0-flash 89.7 90.0 85.4 81.8 80.0 75.0
GPT-4o 94.9 94.8 83.9 82.1 80.9 78.8
Table 2: Object Types and Corresponding Style count
Type Name Ours AutoPresent’s
title 10 3
textbox 10 5
bullet points 8 5
background color 1 1
image 2 2
placeholder 4 –
freeform 2 –
connector 5 –
table 4 –
triangle 5 –
reconstructed from the reverse-engineered code
and the original slide. Our tool achieves a recon-
struction ratio of 90.38%, significantly outperform-
ing (Ge et al., 2025), which only reaches 65.67%.
This demonstrates the broader object type coverage
enabled by our tool.
CLIP Score : Our method achieves a CLIP
score (Hessel et al., 2021) of 88.66%, whereas
Autopresent (Ge et al., 2025) only achieves
69.87%. The higher score indicates that our reverse-
engineered slides more accurately preserve the vi-
sual and stylistic details of the original, owing to
the broader support for object types and styles.5.4 Slide Complexity Metric Analysis
To evaluate the effectiveness of the proposed Slide
Complexity Metric (SCM), we conducted a human
subject study. A total of 100 samples were ran-
domly selected from the Slide2Code benchmark for
evaluation. Four doctoral students were recruited
as annotators, each assigned 50 slides to assess.
The annotators were instructed to score each slide
from the perspective of three dimensions: the num-
ber of shapes, the diversity of shape types, and the
level of element coverage. The scoring range was
0–100, following the protocol in Appendix D. Each
slide was rated independently by two annotators,
and the final score was their average.
To assess the alignment between SCM and hu-
man perception, we first compute the Pearson corre-
lation coefficient (Cohen et al., 2009) between the
SCM complexity scores and the averaged human
scores. The result is r= 0.873with a p-value of
2.776×10−32, indicating a strong and statistically
significant correlation. Additionally, we calculated
the intraclass correlation coefficient (Koo and Li,
2016) between the SCM scores and each individual
annotator’s score to assess consistency. The ICC
result is 0.726with a p-value of 1.186×10−31,

ReferenceSlideCoderGPT-4oGemini2.0-flashSlideMaster(7B)AutoPresentGPT-4oGemini2.0-flashAutoPresent(8B)SimpleMedianComplex
ErrorError
Error
ErrorError
Figure 5: Examples of slides generated by different methods in three difficulty levels.
demonstrating substantial agreement between SCM
and human evaluations. These results confirm that
SCM is a reliable and objective metric aligned with
human judgment of slide complexity.
5.5 Ablation Study
Table 3: Overall performance of ablation study.
Setting Execution% Overall
SlideCoder100.0 89.9
100.0 85.8
100.0 82.2
w/o Layout100.0 81.2
93.9 73.6
93.9 71.8
w/o CGSeg75.8 55.4
51.5 39.6
69.7 48.4
w/o H-RAG90.9 80.4
81.8 69.3
84.8 70.7
Native Setting75.8 53.9
48.5 37.4
66.7 46.9
We design three ablation settings to validate the
effectiveness of different components in our frame-
work: (1) w/o Layout, removes the layout-aware
prompt; (2) w/o CGSeg, disables both the CGSeg
mechanism and the layout-aware prompt; (3) w/o
H-RAG, removes the <Grammar> content from
all prompts.(4) Native setting, which removes H-
RAG on top of the w/o CGSeg setting. Detailed
descriptions are provided in Appendix A.1. We
randomly sample 33 instances from each difficulty
level, resulting in a total of 99 samples, and per-
form inference using GPT-4o. The overall results
are reported in Table 3, with detailed metric result
provided in Appendix A.2. After removing each
component, both execution rate and overall scoreexhibit varying degrees of decline, which demon-
strates the contribution of each component to the
overall framework. Notably, the w/o CGSeg set-
ting shows significant performance drops across all
metrics. Although slightly better than the Native
setting due to the presence of H-RAG.
5.6 Case Study
Figure 5 presents slides generated by different mod-
els under three levels of difficulty. It can be ob-
served that models based on natural language often
fail to satisfy the detailed and layout-specific re-
quirements of reference images. These models fre-
quently produce slides with overlapping elements
or content that extends beyond canvas boundaries.
In medium and complex samples, the generated
code often fails to compile. In contrast, Slide-
Coder’s CGSeg mechanism enables the MLLM
to focus more effectively on fine-grained details.
Moreover, the layout-aware prompt helps ensure
that the spatial arrangement of elements aligns
more closely with reference image.
6 Conclusion
We introduce a new Reference Image to Slide Gen-
eration task and a novel Slide Complexity Met-
ric for evaluating slide complexity. Based on this
metric, we build the Slide2Code benchmark with
different levels of difficulty. We also propose Slide-
Coder enhanced by a Color Gradients-based Seg-
mentation algorithm, a Layout-aware Prompt and
a Hierarchical Retrieval-Augmented Code Genera-
tion for accurate slide generation. A high-quality
training set is curated to fine-tune a 7B open-source
model. Experimental results show that SlideCoder
outperforms the strongest baselines.

Limitations
In this work, we take the first step toward vision-
based slide generation. While our method achieves
substantial improvements across multiple evalu-
ation metrics, several limitations remain unad-
dressed. First, the current framework focuses on
generating a single slide from one reference image
and does not explore the multi-slide generation sce-
nario. Second, we assume that user input contains
separate design and image components, and do not
handle the case where a complete slide with em-
bedded pictures is provided as input. Third, due to
budget and time constraints, our segmentation al-
gorithm adopts a fixed-rule paradigm. Future work
may investigate more flexible model-based detec-
tion approaches to enable adaptive and accurate
block partitioning.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report. arXiv preprint arXiv:2303.08774 .
Shaikh Mostafa Al Masum, Mitsuru Ishizuka, and
Md Tawhidul Islam. 2005. ’auto-presentation’: a
multi-agent system for building automatic multi-
modal presentation of a topic from world wide web
information. In IEEE/WIC/ACM International Con-
ference on Intelligent Agent Technology , pages 246–
249. IEEE.
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl
technical report. arXiv preprint arXiv:2502.13923 .
SV Burtsev and Ye P Kuzmin. 1993. An efficient flood-
filling algorithm. Computers & graphics , 17(5):549–
561.
Steve Canny. 2023. Python-ptx documentation.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. In Findings of the Associa-
tion for Computational Linguistics ACL 2024 , pages
2318–2335.
Israel Cohen, Yiteng Huang, Jingdong Chen, Jacob Ben-
esty, Jacob Benesty, Jingdong Chen, Yiteng Huang,
and Israel Cohen. 2009. Pearson correlation coeffi-
cient. Noise reduction in speech processing , pages
1–4.Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. In Proceedings of the 47th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval , pages 719–729.
Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing
Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Pro-
ceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining , pages 6491–
6501.
Tsu-Jui Fu, William Yang Wang, Daniel McDuff, and
Yale Song. 2022. Doc2ppt: Automatic presentation
slides generation from scientific documents. In Pro-
ceedings of the AAAI Conference on Artificial Intelli-
gence , volume 36, pages 634–642.
Jiaxin Ge, Zora Zhiruo Wang, Xuhui Zhou, Yi-Hao
Peng, Sanjay Subramanian, Qinyue Tan, Maarten
Sap, Alane Suhr, Daniel Fried, Graham Neubig, and
1 others. 2025. Autopresent: Designing structured vi-
suals from scratch. arXiv preprint arXiv:2501.00912 .
Yingqiang Ge, Wenyue Hua, Kai Mei, Juntao Tan,
Shuyuan Xu, Zelong Li, Yongfeng Zhang, and 1 oth-
ers. 2023. Openagi: When llm meets domain experts.
Advances in Neural Information Processing Systems ,
36:5539–5568.
Google. 2025. Gemini API. https://ai.google.
dev/gemini-api . Accessed: 2025-05-19.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. arXiv preprint arXiv:2407.21783 .
Jun Han and Claudio Moraga. 1995. The influence of
the sigmoid function parameters on the speed of back-
propagation learning. In International workshop on
artificial neural networks , pages 195–201. Springer.
Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le
Bras, and Yejin Choi. 2021. Clipscore: A reference-
free evaluation metric for image captioning. arXiv
preprint arXiv:2104.08718 .
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, and 1 others. 2022. Lora: Low-rank
adaptation of large language models. ICLR , 1(2):3.
Yue Hu and Xiaojun Wan. 2014. Ppsgen: Learning-
based presentation slides generation for academic
papers. IEEE transactions on knowledge and data
engineering , 27(4):1085–1097.
Min-Yen Kan. 2007. Slideseer: A digital library of
aligned document and presentation pairs. In Proceed-
ings of the 7th ACM/IEEE-CS joint conference on
Digital libraries , pages 81–90.

Terry K Koo and Mae Y Li. 2016. A guideline of
selecting and reporting intraclass correlation coeffi-
cients for reliability research. Journal of chiropractic
medicine , 15(2):155–163.
Kaixin Li, Yuchen Tian, Qisheng Hu, Ziyang Luo, Zhiy-
ong Huang, and Jing Ma. 2024. Mmcode: Bench-
marking multimodal large language models for code
generation with visually rich programming problems.
arXiv preprint arXiv:2404.09486 .
Jenny GuangZhen Ma, Karthik Sreedhar, Vivian Liu,
Pedro A Perez, Sitong Wang, Riya Sahni, and Ly-
dia B Chilton. 2025. Dynex: Dynamic code synthesis
with structured design exploration for accelerated ex-
ploratory programming. In Proceedings of the 2025
CHI Conference on Human Factors in Computing
Systems , pages 1–27.
Daye Nam, Andrew Macvean, Vincent Hellendoorn,
Bogdan Vasilescu, and Brad Myers. 2024. Using an
llm to help with code understanding. In Proceedings
of the IEEE/ACM 46th International Conference on
Software Engineering , pages 1–13.
Jim Nilsson and Tomas Akenine-Möller. 2020. Under-
standing ssim. arXiv preprint arXiv:2006.13846 .
Kunato Nishina and Yusuke Matsui. 2024. Svgedit-
bench: A benchmark dataset for quantitative assess-
ment of llm’s svg editing capabilities. In Proceedings
of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 8142–8147.
Juan A Rodriguez, Abhay Puri, Shubham Agarwal, Is-
sam H Laradji, Sai Rajeswar, David Vazquez, Christo-
pher Pal, and Marco Pedersoli. 2025. Starvector:
Generating scalable vector graphics code from im-
ages and text. In Proceedings of the AAAI Conference
on Artificial Intelligence , volume 39, pages 29691–
29693.
Athar Sefid and Jian Wu. 2019. Automatic slide gener-
ation for scientific papers. In Third International
Workshop on Capturing Scientific Knowledge co-
located with the 10th International Conference on
Knowledge Capture (K-CAP 2019), SciKnow@ K-
CAP 2019 .
Yuxuan Wan, Yi Dong, Jingyu Xiao, Yintong Huo,
Wenxuan Wang, and Michael R Lyu. 2024. Mrweb:
An exploration of generating multi-page resource-
aware web code from ui designs. arXiv preprint
arXiv:2412.15310 .
Ronghuan Wu, Wanchao Su, and Jing Liao. 2024.
Chat2svg: Vector graphics generation with large lan-
guage models and image diffusion models. arXiv
preprint arXiv:2411.16602 .
Jingyu Xiao, Yuxuan Wan, Yintong Huo, Zhiyao Xu,
and Michael R Lyu. 2024. Interaction2code: How
far are we from automatic interactive webpage gener-
ation? arXiv preprint arXiv:2411.03292 .Jingyu Xiao, Ming Wang, Man Ho Lam, Yuxuan Wan,
Junliang Liu, Yintong Huo, and Michael R. Lyu.
2025. Designbench: A comprehensive benchmark
for mllm-based front-end code generation. Preprint ,
arXiv:2506.06251.
Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang,
Dong Xu, and Qian Yu. 2024. Empowering llms
to understand and generate complex vector graphics.
arXiv preprint arXiv:2412.11102 .
John Yang, Carlos E Jimenez, Alex L Zhang, Kil-
ian Lieret, Joyce Yang, Xindi Wu, Ori Press,
Niklas Muennighoff, Gabriel Synnaeve, Karthik R
Narasimhan, and 1 others. 2024. Swe-bench multi-
modal: Do ai systems generalize to visual software
domains? arXiv preprint arXiv:2410.03859 .
Mingyue Yuan, Jieshan Chen, Zhenchang Xing, Aaron
Quigley, Yuyu Luo, Tianqi Luo, Gelareh Moham-
madi, Qinghua Lu, and Liming Zhu. 2024. Design-
repair: Dual-stream design guideline-aware frontend
repair with large language models. arXiv preprint
arXiv:2411.01606 .
Sukmin Yun, Haokun Lin, Rusiru Thushara, Mo-
hammad Qazim Bhat, Yongxin Wang, Zutao
Jiang, Mingkai Deng, Jinhong Wang, Tianhua Tao,
Junbo Li, and 1 others. 2024. Web2code: A
large-scale webpage-to-code dataset and evaluation
framework for multimodal llms. arXiv preprint
arXiv:2406.20098 .
Fengji Zhang, Linquan Wu, Huiyu Bai, Guancheng Lin,
Xiao Li, Xiao Yu, Yue Wang, Bei Chen, and Jacky
Keung. 2024a. Humaneval-v: Evaluating visual un-
derstanding and reasoning abilities of large multi-
modal models through coding tasks. arXiv preprint
arXiv:2410.12381 .
Linhao Zhang, Daoguang Zan, Quanshun Yang, Zhi-
rong Huang, Dong Chen, Bo Shen, Tianyu Liu, Yong-
shun Gong, Pengjie Huang, Xudong Lu, and 1 oth-
ers. 2024b. Codev: Issue resolving with visual data.
arXiv preprint arXiv:2412.17315 .
Hao Zheng, Xinyan Guan, Hao Kong, Jia Zheng, Weixi-
ang Zhou, Hongyu Lin, Yaojie Lu, Ben He, Xianpei
Han, and Le Sun. 2025. Pptagent: Generating and
evaluating presentations beyond text-to-slides. arXiv
preprint arXiv:2501.03936 .
A Detail ablation analysis
A.1 Details of Ablation Settings
•w/o Layout: Removes only the layout-aware
prompt, meaning that the input to Assembler
does not contain the positional coordinates of
each block.
•w/o CGSeg: Disables the CGSeg mechanism.
Since the goal of Coder is to generate par-
tial code and Assembler is responsible for

code assembly, the removal of CGSeg renders
Assembler unnecessary. Consequently, both
Assembler and its layout-aware prompt are
removed in this setting, and the output code
generated by Coder is directly treated as the
final output of the framework.
•w/o H-RAG: Disables the retrieval of knowl-
edge base content for all agents.
•Native setting: Disables both H-RAG and
CSeg components. Specifically, we input ordi-
nary prompts that do not incorporate H-RAG,
allowing the MLLMs to generate complete
slide code directly from the reference image.
This setup is used to evaluate the baseline
capability of native MLLMs in handling the
reference image to slide code generation task.
A.2 Detailed Analysis of Ablation Results
Table 4 provides a detailed evaluation metrics under
different ablation settings.
In the w/o Layout setting , the Position score
under the complex level drops significantly from
81.35 to 72.16. This is primarily because, in com-
plex cases, the CGSeg algorithm typically divides
the Reference Image(RI) into more blocks, and
without layout information, the Agent struggles
to model spatial relationships among multiple el-
ements. This often leads to overlapping or out-
of-bound content, causing a sharp decline in the
Position metric and slightly affecting other metrics
as well.
In the w/o CGSeg setting , both the CGSeg
mechanism and the layout-aware prompt are re-
moved. As a result, a single Describer Agent is
required to handle the entire complex slide, which
exceeds its processing capacity, often leading to
code generation failures and a sharp drop in ex-
ecution success rate. Its performance is slightly
better than the Native setting due to the additional
knowledge provided by H-RAG.
In the w/o H-RAG setting , the <Grammar>
component is removed from each Agent. Ex-
cluding this component from Describer reduces
its ability to accurately identify the correspond-
ing python-pptx object. Similarly, removing it
from Coder andAssembler deprives the Agents
of essential syntactic guidance, often resulting in
version-related errors caused by inconsistencies
between the model’s training data and the current
version of the python-pptx library. These combined
factors lead to overall performance degradation.In the Native setting , both the CGSeg mech-
anism and H-RAG are removed, leaving a single
Coder Agent to handle the entire slide without any
auxiliary support. This reduces the framework to
a plain MLLM-based inference process, severely
limiting its ability to generate structured and exe-
cutable code, and resulting in the lowest execution
rate and overall performance.
B Detailed comparisons of Reverse Tool
Table 5 lists the object types and their styles sup-
ported by our reverse engineering tool.
Table 6 lists the object types and their styles
supported by AutoPresent’s reverse engineering
tool.
C LoRA fine-tuning parameters
The LoRA fine-tuning parameters are listed in Ta-
ble 7.
D Evaluation Dimensions and Scoring
Criteria
The evaluation guidelines for the four doctoral stu-
dent annotators are provided in Figure 6.
E Prompt Templates
The prompt templates for the Describer and Coder
are shown in Figure 7 and Figure 8, respectively.
Layout-aware prompt is shown in Figure 9.
F Details of the Knowledge Base
Construction
Figure 10 presents several examples from the Shape
Type Knowledge Base, which consists of object
types defined in the python-pptx library along with
their corresponding descriptions. Figure 11 shows
an example from the Operation Function Knowl-
edge Base, which includes the function name, pa-
rameters, return value, usage example, and a textual
explanation of the function.

Table 4: Detailed performance analysis under several ablation settings. Green ,yellow , and redindicate simple,
medium, and complex levels in SlideCoder. Bolded values mark the best result per level.
Setting Execution%Global Visual Metrics Local Structural MetricsOverallContent Position Clip SSIM
SlideCoder100.0 97.1 89.9 80.8 92.9 89.9
100.0 92.7 86.5 82.7 85.8 85.8
100.0 95.0 81.3 82.2 82.3 82.2
w/o Layout100.0 88.8 86.4 81.2 79.2 81.2
93.9 90.4 75.2 80.9 78.4 73.6
93.9 93.6 72.2 80.3 76.4 71.8
w/o CGSeg75.8 90.4 86.5 69.4 73.1 55.4
51.5 91.7 81.4 68.5 71.4 39.6
69.7 93.0 83.2 68.1 69.0 48.4
w/o H-RAG90.9 98.6 88.4 79.7 91.8 80.4
81.8 91.6 84.7 81.7 87.8 69.3
84.8 94.0 87.9 81.3 83.4 70.7
Native Setting75.8 90.0 87.9 71.1 71.2 53.9
48.5 92.9 83.3 66.7 69.5 37.4
66.7 92.6 85.7 66.5 70.4 46.9
Table 5: The object types and their styles supported by our reverse engineering tool.
Object Type Styles
textbox Position, Text frame margin, Alignment, Paragraph spacing, Font style, Fill
color, Font size, Bold, Italic, Underline
rectangle Position, Line color, Line width, Fill color
object_placeholder Position, Fill color, Object position
freeform Position, Fill color
bullet_points Position, Item content, Font size, Font color, Fill color, Bold, Italic, Underline
image Position, Image path
background_color Color
connector Start position, End position, Arrow color, Arrow width, Arrow style
table Position, Cell height, Cell fill color, Text inside cell
triangle Position, Type, Line color, Line width, Fill color
Table 6: The object types and their styles supported by AutoPresent’s reverse engineering tool.
Object Type Styles
title Font size, Font color, Fill color
textbox Position, Font size, Bold, Font color, Fill color
bullet_points Position, Item content, Font size, Font color, Fill color
image Position, Image path
background color Color

Slide Complexity Evaluation Guide Purpose of Evaluation This guideline is intended to assist you in subjectively evaluating the complexity of slide samples based on the following three dimensions: 1. Number of Shapes 2. Diversity of Shape Types 3. Visual Complexity Each dimension should be scored on a scale from 0 to 100. You are expected to assess each slide independently and provide a final overall score reflecting your holistic judgment of the slide’s complexity. Evaluation Procedure For each slide, please follow these steps: 1. Review the slide thoroughly to understand its structure and element layout. 2. Evaluate each of the three dimensions separately (see detailed criteria below). 3. Based on your judgment, assign a comprehensive overall score (0–100).Record your scores (three dimensions + overall) clearly in the scoring table. Scoring Dimensions and Criteria 1. Number of Shapes Refers to the total count of visual elements on the slide, including but not limited to: text boxes, diagrams, arrows, lines, images, geometric shapes, etc. • 0–20: Very few elements (e.g., only a title and 1–3 text boxes). • 21–50: Moderate amount of shapes (e.g., 4–10 elements, such as text + one chart). • 51–80: High density of shapes (e.g., 11–20 elements, visually filled slide). • 81–100: Extremely dense, cluttered with over 20 elements. 2. Diversity of Shape Types Measures how varied the types of visual components are. Common types include text boxes, images, tables, flowcharts, icons, arrows, geometric shapes (e.g., rectangles, circles, lines), and more. • 0–20: Only one type used (e.g., all text). • 21–50: Two or three different types, basic variety. • 51–80: Four to six types, indicating notable diversity. • 81–100: Rich variety with more than six distinct shape types. 3. Visual Complexity Refers to how complex the slide appears in terms of visual density, layout structure, information layering, and cognitive load. It captures the subjective perception of how “complicated” the slide looks. • 0–20: Very clean and minimalist, with generous whitespace. • 21–50: Well-structured, moderately filled, visually comfortable. • 51–80: Noticeably dense, some clutter, yet still readable. • 81–100: Overwhelming amount of information, chaotic layout, hard to scan quickly. Overall Score Guidelines After rating the three dimensions above, you are asked to provide a final overall score (0–100) that reflects your subjective judgment of the slide’s overall complexity. 
⚠ Note: This does not need to be a simple average of the three scores. Instead, consider how each factor influences the overall perception of complexity. Figure 6: Evaluation guidelines provided to the four doctoral student annotators.

Figure 7: Prompt of Describer.

Prompt of Coder Code generation process Please write Python code to create a PowerPoint slide that matches the following description: {block_description}  The following is an introduction in python-pptx API Documentation: {<Grammar> }  Please generate Python code using the python-pptx library to create a PowerPoint slide based on the provided codes. The code should:   1. Create a new PowerPoint presentation.   2. Add a slide using the slide layout with index 6 (typically a Blank Layout) to ensure a clean slate for custom content placement.   3. Include all text elements and shapes as specified in the slide, with properties such as font, size, color, and alignment accurately applied.   4. Use inches (in) units for all size and position measurements, directly converting them using python-pptx's Inches() function for shapes and positions, and Pt for font sizes.   5. Save the presentation in the output/generated_ppts directory with a descriptive filename (e.g., generated_slide.pptx).   6. Ensure the code is well-commented and handles any necessary imports. {block_image}  Fix code process You are a python-pptx expert. The previous code generated an error. Please fix the code. Error message:   {error_message} Previous code: {code}  Introduction in python-pptx API Documentation: {<Grammar> } Please provide the complete corrected code that will create the PowerPoint slide successfully.    Figure 8: Prompt of Coder.

Layout-aware prompt  You are a python-pptx expert. Based on the information and code snippets I provide, please assemble a complete python-pptx script:  <Design> refers to the reference image for this slide.   Its global description is <Overall Description>.   The code snippets and their layout positions are given as  <Code Snippets1>, <Position1*>.  <Code Snippets1>, <Position1*>.  …  Here are some syntax rules that might be useful: <Grammar>.   The background and images path is ...   Background path: {background_image_path} Image1 Path: {image_path_1} Image1 Coordinates: Left: {x1} inches Top: {y1} inches Width: {w1} inches Height: {h1} inches Please provide the complete corrected code that will create the PowerPoint slide successfully. Please generate Python code using the python-pptx library to create a PowerPoint slide based on the provided codes. The code should: 1. Create a new PowerPoint presentation. 2. Add a slide using the slide layout with index 6 (typically a Blank Layout) to ensure a clean slate for custom content placement. 3. Include all text elements and shapes as specified in the slide, with properties such as font, size, color, and alignment accurately applied. 4. Use inches (in) units for all size and position measurements, directly converting them using python-pptx's Inches() function for shapes and positions, and Pt for font sizes. 5. Save the presentation in the output/generated_ppts directory with a descriptive filename (e.g., generated_slide.pptx). 6. Ensure the code is well-commented and handles any necessary imports.  Figure 9: Layout-aware prompt.

Auto Shape An auto shape is a predefined, customizable shape in PowerPoint, such as a rectangle, ellipse, or block arrow, with approximately 180 variations. Auto shapes can have a fill, outline, and contain text. Some include adjustable features, indicated by yellow diamond handles (e.g., to modify the corner radius of a rounded rectangle). A text box is a specific type of auto shape, typically rectangular, with no default fill or outline.  ########  Picture A picture in PowerPoint refers to a raster image, such as a photograph or clip art, treated as a distinct shape type with unique behaviors compared to auto shapes. Note that an auto shape can have a picture fill, where an image serves as the shape’s background instead of a color or gradient, but this is a separate feature.  ########  Graphic Frame A graphic frame is a container that automatically appears in a PowerPoint file when adding graphical objects like tables, charts, SmartArt diagrams, or media clips. It cannot be inserted independently and typically requires no direct interaction from the user.  ########  Group Shape A group shape is created when multiple shapes in PowerPoint are grouped, enabling them to be selected, moved, resized, or filled as a single unit. The group shape is only visible through its bounding box when selected, containing the individual member shapes.  ########  Line/Connector Lines are linear shapes distinct from auto shapes. Some lines, known as connectors, can attach to other shapes and remain connected when those shapes are moved. Connectors are not yet fully supported in some contexts, but they are valuable for creating dynamic diagrams.  ########  Content Part A content part involves embedding external XML data, such as SVG, within a PowerPoint presentation. PowerPoint itself does not actively utilize content parts, and they can generally be ignored without impacting functionality.  …… Figure 10: Examples from the Shape Type knowledge base.

# Function: `pptx.Presentation`  ## Function Name  `pptx.Presentation`  ## Function Parameters  - **pptx** (`Union[str, IO[bytes], None]`, optional, default: `None`)   - Description: Specifies the source of the presentation.     - If a `str`, it represents the file path to a `.pptx` file.     - If an `IO[bytes]`, it represents a file-like object containing the `.pptx` file data.     - If `None`, loads the built-in default presentation template.   - Constraints: The file or stream must be a valid `.pptx` file if provided.  ## Function Return Value  - **Type**: `presentation.Presentation` - **Description**: A `Presentation` object representing the loaded or newly created PowerPoint presentation.  ## Function Python Example  ```python from pptx import Presentation  # Create a new presentation using the default template prs = Presentation()  # Load an existing presentation from a file prs = Presentation("existing_presentation.pptx")  # Load a presentation from a file-like object from io import BytesIO with open("existing_presentation.pptx", "rb") as f:     prs = Presentation(BytesIO(f.read())) ```  ## Function Purpose  The `pptx.Presentation` function is the primary entry point for creating or loading a PowerPoint presentation. It initializes a `Presentation` object, which provides access to slides, slide masters, layouts, and other presentation components, enabling programmatic manipulation of presentation content. Figure 11: An example from the Operation Function knowledge base.

Table 7: LoRA fine-tuning configuration used in our
experiments.
Parameter Value
Rank 8
Max Sequence Length 4096
Batch Size 4
Gradient Accumulation Steps 8
Learning rate 1e-4
Epochs 10
Warmup Ratio 0.1
Mixed Precision bf16