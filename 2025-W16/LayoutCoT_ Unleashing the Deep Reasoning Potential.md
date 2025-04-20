# LayoutCoT: Unleashing the Deep Reasoning Potential of Large Language Models for Layout Generation

**Authors**: Hengyu Shi, Junhao Su, Huansheng Ning, Xiaoming Wei, Jialin Gao

**Published**: 2025-04-15 03:12:01

**PDF URL**: [http://arxiv.org/pdf/2504.10829v1](http://arxiv.org/pdf/2504.10829v1)

## Abstract
Conditional layout generation aims to automatically generate visually
appealing and semantically coherent layouts from user-defined constraints.
While recent methods based on generative models have shown promising results,
they typically require substantial amounts of training data or extensive
fine-tuning, limiting their versatility and practical applicability.
Alternatively, some training-free approaches leveraging in-context learning
with Large Language Models (LLMs) have emerged, but they often suffer from
limited reasoning capabilities and overly simplistic ranking mechanisms, which
restrict their ability to generate consistently high-quality layouts. To this
end, we propose LayoutCoT, a novel approach that leverages the reasoning
capabilities of LLMs through a combination of Retrieval-Augmented Generation
(RAG) and Chain-of-Thought (CoT) techniques. Specifically, LayoutCoT transforms
layout representations into a standardized serialized format suitable for
processing by LLMs. A Layout-aware RAG is used to facilitate effective
retrieval and generate a coarse layout by LLMs. This preliminary layout,
together with the selected exemplars, is then fed into a specially designed CoT
reasoning module for iterative refinement, significantly enhancing both
semantic coherence and visual quality. We conduct extensive experiments on five
public datasets spanning three conditional layout generation tasks.
Experimental results demonstrate that LayoutCoT achieves state-of-the-art
performance without requiring training or fine-tuning. Notably, our CoT
reasoning module enables standard LLMs, even those without explicit deep
reasoning abilities, to outperform specialized deep-reasoning models such as
deepseek-R1, highlighting the potential of our approach in unleashing the deep
reasoning capabilities of LLMs for layout generation tasks.

## Full Text


<!-- PDF content starts -->

LayoutCoT: Unleashing the Deep Reasoning Potential of Large Language Models
for Layout Generation
Hengyu Shi1,2*,Junhao Su1*,Huansheng Ning2,Xiaoming Wei1,Jialin Gao1B
1Meituan
2University of Science and Technology Beijing
{shihengyu02, sujunhao02, weixiaoming, gaojialin04 }@meituan.com
ninghuansheng@ustb.edu.cn
Abstract
Conditional layout generation aims to automatically gen-
erate visually appealing and semantically coherent layouts
from user-defined constraints. While recent methods based
on generative models have shown promising results, they
typically require substantial amounts of training data or ex-
tensive fine-tuning, limiting their versatility and practical
applicability. Alternatively, some training-free approaches
leveraging in-context learning with Large Language Mod-
els (LLMs) have emerged, but they often suffer from limited
reasoning capabilities and overly simplistic ranking mech-
anisms, which restrict their ability to generate consistently
high-quality layouts. To this end, we propose LayoutCoT,
a novel approach that leverages the reasoning capabilities
of LLMs through a combination of Retrieval-Augmented
Generation (RAG) and Chain-of-Thought (CoT) techniques.
Specifically, LayoutCoT transforms layout representations
into a standardized serialized format suitable for processing
by LLMs. A Layout-aware RAG is used to facilitate effective
retrieval and generate a coarse layout by LLMs. This prelim-
inary layout, together with the selected exemplars, is then fed
into a specially designed CoT reasoning module for iterative
refinement, significantly enhancing both semantic coherence
and visual quality. We conduct extensive experiments on five
public datasets spanning three conditional layout genera-
tion tasks. Experimental results demonstrate that Layout-
CoT achieves state-of-the-art performance without requiring
training or fine-tuning. Notably, our CoT reasoning module
enables standard LLMs, even those without explicit deep rea-
soning abilities, to outperform specialized deep-reasoning
models such as deepseek-R1, highlighting the potential of
our approach in unleashing the deep reasoning capabilities
of LLMs for layout generation tasks.
∗Equal Contribution. Name is ordered by alphabet.
BCorresponding Author.1. Introduction
Layout is a fundamental element in graphic design, de-
termined by the deliberate and structured arrangement of vi-
sual components. Automated layout generation has recently
emerged as an important research topic, aiming to signif-
icantly alleviate the workload of designers while address-
ing diverse user requirements [ 11,19,21–23,26]. Current
automated layout generation tasks broadly fall into three
categories: i) content-aware layouts, which integrate visual
or textual content for contextually appropriate designs (e.g.,
aligning text boxes with salient image regions in posters);
ii) constraint-explicit layouts, which strictly adhere to prede-
fined constraints, commonly found in user interface (UI) de-
sign and document typesetting; iii) and text-to-layout tasks,
which translate abstract textual instructions into spatially
coherent layouts, crucial for cross-modal creative tools.
Early approaches [ 9,12] mainly employed generative
models, learning implicit design rules from extensive labeled
datasets. Methods such as LayoutGAN [ 25] and LayoutDif-
fusion [ 43] have shown promising capabilities in generating
plausible layouts, yet their reliance on substantial training
data and domain-specific fine-tuning limits scalability and
versatility. When faced with limited training data, purely gen-
erative methods [ 14,16,18,19,28] often fail to effectively
capture sparse data distributions, restricting generalization
and necessitating separate models for distinct tasks. Fur-
thermore, the demand for extensive labeled datasets makes
training generative layout models costly and inefficient, high-
lighting the necessity for a more flexible and versatile layout
generation paradigm.
With the advancement of large language models (LLMs)
[1,2,17,41,42], layout generation methods are shifting to-
wards a training-free paradigm. Recent research has shown
that LLMs inherently possess a certain understanding of
layout principles, such as alignment and spatial coherence
[3,17]. While leveraging this capability, approaches such as
LayoutPrompter [ 28] utilize in-context learning combined
1arXiv:2504.10829v1  [cs.CV]  15 Apr 2025

Figure 1. Qualitative evaluation of the multi-stage LayoutCoT framework. Each successive stage systematically refines the spatial arrangement
of layout elements, resulting in designs with enhanced coherence and rationality.
with post-processing ranking mechanisms to guide LLMs
in layout generation. However, this methodology demon-
strates excessive focus on quantitative evaluation metrics,
frequently resulting in structurally impractical bounding box
configurations containing either disproportionately small el-
ements or physically unfeasible arrangements. Moreover,
when confronted with complex layout scenarios, the sim-
plistic reasoning paradigms and elementary ranking systems
inherent in such approaches tend to produce suboptimal
compositions that violate both human visual perception prin-
ciples and real-world application constraints.
To unleash the deep reasoning abilities of LLMs in layout
generation, we must resolve two key issues: first, how to se-
lect suitable contexts as references for layout generation; and
second, how to design a chain of thought that enables LLMs
to engage in human-like deep reasoning during layout design.
Therefore, we propose LayoutCoT, a novel approach that in-
tegrates a layout-aware RAG with Chain-of-Thought (CoT)
in LLMs to address those above limitations. Specifically,
LayoutCoT employs a layout-aware exemplar retrieval mech-
anism, dynamically selecting suitable reference contexts us-
ing layout similarity [ 32], which better captures structural
and semantic similarities compared to previous methods like
LayoutPrompter [ 28]. We further introduce a meticulously
designed Chain-of-Thought (CoT) reasoning framework that
refines the layout through three sequential stages, each ad-
dressing a distinct aspect of the design process. In the firststage, the LLM is prompted to evaluate each element from
the specified categories and determine their initial positions,
emphasizing logical structure and visual appeal without ac-
counting for interactions among element sizes. In the second
stage, the LLM is guided to optimize the sizes and positions
of these elements, effectively resolving overlaps or poten-
tial crowding issues. Finally, in the third stage, fine-grained
mathematical adjustments are applied to specific layout di-
mensions to avoid unreasonable arrangements—such as the
occlusion of salient elements—and to ensure both aesthetic
quality and practical viability.
We conducted comprehensive experiments across five
public datasets (PKU [ 14], CGL [ 46], RICO [ 29], Publaynet
[45], and WebUI [ 40]), spanning three distinct layout gen-
eration tasks. Our results demonstrate that LayoutCoT con-
sistently outperforms existing state-of-the-art methods with-
out requiring any training or fine-tuning. Notably, our CoT
module enables standard LLMs, such as GPT-4, to surpass
the performance of specialized deep-reasoning models like
DeepSeek-R1 [ 10], underscoring the potential of our ap-
proach for practical layout generation tasks. The main con-
tributions of our work are summarized as follows:
•We introduce LayoutCoT, a method that effectively har-
nesses the deep reasoning capabilities of general-purpose
LLMs through Chain-of-Thought prompting, significantly
improving the performance and practicality of training-
free layout generation.
2

•LayoutCoT requires no task-specific training or fine-
tuning, efficiently producing high-quality, visually coher-
ent layouts by decomposing complex generation tasks into
clear reasoning steps.
•Extensive empirical validation demonstrates LayoutCoT’s
superior performance and versatility across various tasks
and datasets, highlighting its ability to elevate standard
LLMs above specialized deep-reasoning counterparts.
2. Related Work
Layout generation has emerged as an active area of re-
search in recent years. It can be broadly divided into three cat-
egories: content-aware layout generation [ 14,44], constraint-
explicit layout generation [ 21–23], and text-to-layout genera-
tion [ 15,27]. Content-aware layout generation aims to place
layout elements on a given canvas in a visually pleasing man-
ner while avoiding overlap with any background images on
the canvas. Constraint-Explicit layout generation typically
involves arranging predefined elements, or elements with
specified dimensions, or even specified relationships among
elements, on a blank interface in a reasonable configuration.
Text-to-layout generation, meanwhile, focuses on creating
layouts based on natural language descriptions, which is
generally more challenging.
2.1. Generative layout generation
Early work in this field commonly utilized generative
models for layout generation. Research based on Genera-
tive Adversarial Networks (GANs) [14, 26] and Variational
Autoencoders (V AEs) [ 20,23] pioneered the application of
generative models to layout generation, achieving noticeable
success in improving layout aesthetics through latent vari-
able optimization. With the advent of the Transformer [ 38]
architecture, numerous studies [ 11,15,22,27,33] emerged
that adopted a sequence-to-sequence approach, further en-
hancing the quality of generated layouts. More recently, the
proposal of diffusion [ 12] models has led to additional meth-
ods [16, 18] that bolster generative capabilities in scenarios
with explicit layout constraints, offering further advance-
ments in both quality and overall visual appeal. However,
the existing generative layout methods are not capable of
handling all types of layout generation tasks using a single
model. Moreover, these approaches typically rely heavily on
extensive datasets and require substantial training or fine-
tuning, making practical applications challenging.
2.2. LLM-based Layout Generation
Large language models have demonstrated exceptional
few-shot performance [ 1,2,6,37] across various natural lan-
guage processing (NLP) tasks. Recent studies have shown
that LLMs themselves can generate layouts in structured
descriptive languages such as HTML [ 28,36]. For instance,LayoutNUWA [ 36] leverages Code Llama’s [ 34] code com-
pletion for layout generation, while PosterLlama [ 35] fine-
tunes models such as MiniGPT-4 [ 48] to further enhance
code completion capabilities for layout generation tasks.
However, these methods rely heavily on sophisticated fine-
tuning strategies and are dependent on large amounts of data.
In contrast, LayoutPrompter [ 28] and LayoutGPT [ 7] utilize
GPT-3 [ 2] via in-context learning to achieve training-free
layout generation, revealing that LLMs inherently possess
the capacity for layout generation. Although these methods
do not require extensive, painstaking training and can gen-
eralize across various tasks, the quality of their generated
layouts remains a concern.
2.3. Retrieval-Augmented Generation and Chain of
Thought
Retrieval-Augmented Generation (RAG) [ 24] is a cutting-
edge technique that combines retrieval and generation to en-
hance the model’s generative capabilities. By retrieving rele-
vant examples or information [ 5,30], RAG introduces more
comprehensive contextual data during the generation process,
thereby improving the quality of the outputs. Content-aware
graphic layout generation aims to automatically arrange vi-
sual elements based on given content, such as e-commerce
product images. In this context, the Retrieval-Augmented
Layout Transformer (RALF) [ 13] leverages the retrieval of
layout examples most similar to the input image to assist
in layout design tasks. However, the RAG in RALF is con-
strained by its input paradigm, lacking plug-and-play char-
acteristics and performing suboptimally across other tasks.
Chain-of-Thought (CoT) [ 4,31,39] is a method that en-
hances model reasoning capabilities through the simulation
of human thought processes. CoT achieves this by progres-
sively solving problems through phased reasoning, offering
more interpretable and coherent solutions to complex tasks.
Although CoT has demonstrated powerful reasoning advan-
tages across numerous domains, it has not yet been applied to
layout design tasks. We attempt to integrate RAG with CoT
for layout generation tasks, utilizing RAG to provide rich
contextual information, while employing CoT for phased
reasoning and optimization. This synergistic approach aims
to enhance the quality and coherence of the generated results,
combining the strengths of both methodologies effectively.
3. Methods
Figure. 2 illustrates the overall structure of LayoutCoT,
which consists of three main components: the Layout-aware
RAG, the layout coarse generator, and the layout CoT
module. Specifically, given the labels and bounding boxes
(bboxes) of an input example layout, we use the Layout-
aware RAG to select the top-k relevant layout examples as
prompting examples. Next, the layout coarse generator takes
these prompting examples and the given constraints to pro-
3

Figure 2. Overview of the LayoutCoT framework. Our training-free approach initially employs a Layout-aware RAG to prompt the LLM
for a coarse layout prediction, establishing a logical and visually coherent arrangement of elements. This is subsequently refined via a
multi-stage Chain-of-Thought (CoT) module, which iteratively enhances the layout by resolving spatial conflicts and fine-tuning dimensions.
The proposed framework is versatile and applicable to a wide array of layout generation tasks.
duce an initial coarse layout. Finally, both the prompting
examples and this coarse layout are fed into the layout CoT
module for fine-grained layout adjustments. Because the
layout coarse generator lacks a detailed reasoning process,
relying solely on the prompting examples and constraints
may not fully satisfy the design requirements. Therefore,
the layout CoT module breaks down the layout task into a
multi-stage thought process; each stage refines the layout,
ultimately yielding a more precise and aesthetically pleasing
result.
3.1. Layout-aware RAG
We employ a Retrieval-Augmented Generation (RAG)
mechanism to dynamically select relevant layout examples
as prompts, thereby enhancing the flexibility and coher-
ence of generation. Specifically, let L=
(bi, ci)	m
i=1and
ˆL=
(ˆbj,ˆcj)	n
j=1denote two layouts, where each element
consists of a bounding box bi(orˆbj) and an associated label
ci(orˆcj). We use a dissimilarity function from LTSim [ 32]
D(L,ˆL)based on a soft matching scheme:
D(L,ˆL) = min
ΓX
(i,j)∈ΩΓi,jµ 
bi, ci,ˆbj,ˆcj
,s.t.Γi,j≥0,
(1)
where Γi,jrepresents a fractional assignment aligning ele-
ment (bi, ci)with(ˆbj,ˆcj), and µ(·)is a cost function that
captures positional and label discrepancies. By solving this
optimization, we obtain the minimal “transport cost” be-
tween the two layouts.Next, we transform this cost-based measure into a layout-
level similarity via an exponential mapping:
LTSim( L,ˆL) = exp
−D(L,ˆL)
. (2)
A lower cost D(L,ˆL)implies higher layout similarity; we
set the scale parameter to 1.0for simplicity [32].
To retrieve the top- Kexamples for a given layout L, we
consider a database DB={ˆL1,ˆL2, . . . , ˆLN}. We compute
LTSim 
L,ˆLi
for each candidate ˆLi∈DB, and select the
top-Klayouts that yield the highest scores:
RK(L) = TopK ˆL∈DBLTSim 
L,ˆL
. (3)
We then collect these ˆLk∈ R K(L)as prompting exam-
ples in subsequent generation stages. By leveraging the soft
matching mechanism in LTSim , the Layout-aware RAG
remains robust even when the input and candidate layouts
differ in the number or types of elements.
3.2. Coarse Layout Generation
After retrieving the top- Kexamples via RAG (Sec-
tion 3.1), we convert each selected layout ˆLkinto a pre-
defined HTML format. Let RK(L) ={ˆL1, . . . , ˆLK}be the
retrieved set; we encode each ˆLkas an HTML snippet and
combine all of them with the given constraint c. Formally,
we provide the large language model (LLM) with:
n
HTML( ˆL1), . . . , HTML( ˆLK), co
,
4

Figure 3. Details of the CoT Module. We illustrate the overall conversational logic of the multi-stage CoT. Depending on the type of task,
there are slight variations in the details to adapt to specific requirements.
where HTML( ˆLk)denotes the HTML representation of lay-
outˆLk, andcspecifies user-defined constraints.
We then prompt the LLM to generate ncandidate layouts
L1
HTML =LLM
{HTML( ˆL1), . . . , HTML( ˆLK), c}
.
(4)
By repeating the above generation process ntimes, we obtain
n
L1
HTML , . . . ,Ln
HTMLo
,
each in HTML format. To select the most promising one
from these ncandidates, we employ a ranker τ(·)following
LayoutPrompter [ 28]. Specifically, we compute a layout
quality score τ 
Lj
HTML
for each generated layout Lj
HTML
and pick the highest-scoring layout as our coarse result:
Lcoarse
HTML = arg max
Lj
HTML,1≤j≤nτ 
Lj
HTML
.(5)
This coarse layout, Lcoarse
HTML , preserves the key structural
constraints specified by the user while serving as the initial
configuration for subsequent refinements.
3.3. Chain of Thought for Layout Refinement
Once we have obtained the coarse layout Lcoarse
HTML un-
der constraint c, we employ a multi-stage Chain-of-Thought
(CoT) strategy to iteratively refine the layout. Concretely,
we break the entire layout generation task (still under the
user-defined constraint c) into isimpler refinement subprob-
lems, denoted by {Q1
L, Q2
L, . . . , Qi
L}. Each Qt
L(1≤t≤i)
focuses on a specific aspect of improving the layout (e.g.,
rearranging bounding boxes, reducing overlaps, or adjusting
spacing among elements).Iterative Refinement Procedure. At the first refinement
step (t= 1), we feed the following information to the large
language model (LLM):
n
HTML( ˆL1), . . . , HTML( ˆLK),Lcoarse
HTML , c, Q1
Lo
,
where {HTML( ˆL1), . . . , HTML( ˆLK)}are the top- K
prompting examples (cf. Section 3.1), Lcoarse
HTML is the ini-
tial coarse layout, cis the user-defined constraint, and Q1
Lis
the first refinement subproblem. The LLM then outputs:
Lrefine 1
HTML =LLM
HTML( ˆL1), . . . ,Lcoarse
HTML , c, Q1
L
.
(6)
At the t-th refinement step ( 1< t≤i), the input is
updated with the newly refined layout from the previous
step,Lrefine t−1
HTML , together with the same top- Kexamples,
constraint c, and the next subproblem Qt
L:
n
HTML( ˆL1), . . . , HTML( ˆLK),Lrefine t−1
HTML , c, Qt
Lo
,
yielding:
Lrefine t
HTML =LLM
HTML( ˆL1), . . . ,Lrefine t−1
HTML , c, Qt
L
.
(7)
After completing all isubproblems, the final refined lay-
out is:
Lrefine i
HTML =LLM
HTML( ˆL1), . . . ,Lrefine i−1
HTML , c, Qi
L
.
This multi-stage, question-and-answer refinement process
allows the LLM to tackle each localized challenge in turn,
resulting in a more coherent and aesthetically pleasing final
layout.
5

Table 1. Dataset statistics.
Dataset Tasks Training Set Test Set Element Types
PKU content-aware layout generation 9,974 905 3
CGL content-aware layout generation 38,510 1,647 5
RICO constraint-explicit layout generation 31,694 3,729 25
PubLayNet constraint-explicit layout generation 311,397 10,998 5
WebUI text-to-layout 3,835 487 10
4. Experiments
4.1. Experimental Setup
We conduct detailed experiments on three different types
of tasks: content-aware, constraint-explicit, and text-to-
layout, using five distinct datasets: PKU [ 14], CGL [ 46],
RICO [ 29], Publaynet [ 45], and WebUI [ 40]. Since no offi-
cial test set splits are provided for the PKU and CGL datasets,
we used 9,974 labeled images in the PKU dataset as the train-
ing set and the database for the Layout-aware RAG module,
with 905 unlabeled images serving as the test set. In the CGL
dataset, we took 38,510 images as the training set and the
database for the Layout-aware RAG module, and selected an
additional 1,647 images (with no overlap with the training
set) as the test set. For other datasets, we adopt the official
data splits, as detailed in Table .1
4.2. Implement Details
In our experiments, we provide details on the choice of
thek-value for the top- kretrieval in the Layout-aware RAG.
When generating the Coarse Layout, we follow the approach
of LayoutPrompter [ 28] and set k=10. For the CoT stage, we
setk=4. We also adopt LayoutPrompter’s setting of n=10
for the number of generation times by LLMs when selecting
the Coarse Layout. Due to the ’Text-Davinci-003’ model
used by LayoutPrompter is no longer available, we replace
the LLM with ’gpt-4’ for comparison. Furthermore, when
conducting the LayoutCoT experiments, we also use ’gpt-4’
as the LLM. In addition, we will provide the performance
results of ’DeepSeek-R1’ after removing the CoT module.
4.3. Evaluation Metrics
For the constraint-explicit layout generation and text-to-
layout tasks, we utilize four standard metrics for evaluation:
Alignment measures how well the elements within the lay-
out align with each other. Overlap calculates the area of
overlap between any two elements in the layout. Maximum
IoU, mIoU assesses the maximum intersection over union
between the generated layout and the actual layout, providing
a measure of layout accuracy. Frechet Inception Distance
(FID) quantifies the similarity between the distribution of
generated layouts and the distribution of real layouts, using
features extracted from a deep neural network trained on
images. For the content-aware layout generation task, we
reference the eight metrics used in [ 14] along with a newlydesigned metric Rethat we introduce. The metric Reis used
to assess the reasonableness of each element’s size within
the layout. Let
ri=Si
Si
train, d i=lnri, τ = ln(1 .1),
where Siis the predicted average area of label iandSi
train
is the corresponding training -set reference. The constant τ
encodes a ±10% tolerance band.
score i= exp 
−max(0 , di−τ)
. (8)
Eqa.(8)yields score i= 1 whenever 0.9≤ri≤1.1
and decays exponentially (symmetrically for oversized and
undersized elements) once the deviation exceeds 10%.
ForNlabels, the overall area-error metric is
Rcoarse
e =1
NNX
i=1score i. (9)
A more extreme-error-sensitive alternative is
Re= exp
−vuut1
NNX
i=1 
max(0 , di−τ)2
. (10)
BothRcoarse
e andRelie in (0,1].
4.4. Results on Content-Aware Layout Generation
The quantitative experimental results are shown in Ta-
ble 2. As seen from the table, our method surpasses all
the evaluation metrics used in content-aware layout gener-
ation tasks compared to both the training-based baseline
methods and another training-free method, LayoutPrompter
[28]. Notably, our proposed CoT module enables ”GPT-4”
to even outperform the deep reasoning large language model
”DeepSeek-R1” in this layout generation task. Moreover, our
performance on the Overlap metric in the PKU dataset was
suboptimal, prompting further visual analysis. As illustrated
in Figure. 4, the very low scores on the Overlap metric for
both LayoutPrompter and LayoutCoT†result from numer-
ous bad cases. In these cases, layout bounding boxes are
extremely small and densely clustered in the upper-left cor-
ner, leading to low Overlap values but highly unaesthetic
and impractical layout outcomes. Hence, we introduce the
Remetric in Sec. 4.3 to measure the reasonableness of lay-
out sizes, on which we have performed exceptionally well.
Qualitative comparative cases are further provided in Figure.
4.
4.5. Results on Constraint-Explicit Layout Genera-
tion
Table 3 presents the results of the quantitative evaluation,
where we compare LayoutCoT with several training-based
6

Table 2. Content-Aware layout generation task results, all obtained under unconstrained generation conditions. LayoutCoT†denotes the
results when using ’DeepSeek-R1’ as the model with the CoT module removed, while both LayoutPrompter and LayoutCoT use ’gpt-4’ as
the model.
CGL PKU
Method Training-free Content Graphic Content Graphic
Occ↓Rea↓Uti↑Align↓Und l↑Und s↑Ove↓ Val↑Re↑Occ↓Rea↓Uti↑Align↓Und l↑Und s↑Ove↓ Val↑Re↑
CGL-GAN [47] ✗ 0.489 0.268 0.147 0.0420 0.275 0.244 0.269 0.876 - 0.219 0.175 0.226 0.0062 0.575 0.259 0.0605 0.707 -
DS-GAN [14] ✗ 0.451 0.224 0.193 0.0485 0.370 0.301 0.075 0.893 - 0.209 0.217 0.229 0.0046 0.585 0.424 0.026 0.903 -
RALF [13] ✗ 0.336 0.197 0.247 0.0023 0.943 0.884 0.027 0.995 0.874 0.171 0.150 0.246 0.0007 0.893 0.840 0.018 0.998 0.981
LayoutPrompter [28] ✓ 0.415 0.184 0.227 0.0091 0.338 0.317 0.0049 0.996 0.812 0.251 0.171 0.237 0.0021 0.824 0.809 0.0005 0.997 0.865
LayoutCoT†✓ 0.267 0.182 0.225 0.0018 0.515 0.501 0.0023 0.994 0.803 0.214 0.180 0.264 0.0016 0.925 0.849 0.0004 1.000 0.806
LayoutCoT (ours) ✓ 0.170 0.101 0.260 0.0005 0.958 0.958 0.0018 0.998 0.939 0.207 0.074 0.289 0.0002 0.980 0.946 0.0013 1.000 1.000
Figure 4. Qualitative Results for the Content-aware Layout Generation Task. LayoutCoT†denotes the results when using ’DeepSeek-R1’ as
the model with the CoT module removed. As can be seen, both LayoutPrompter and LayoutCoT †tend to generate dense and very small
layout boxes in the upper-left corner, which is visually unreasonable. In contrast, LayoutCoT effectively corrects this error and achieves
satisfactory results.
baselines and the training-free method LayoutPrompter.
From the metrics, it is evident that, even without any train-
ing, LayoutCoT achieves optimal performance on most in-
dicators. The corresponding qualitative results are shown
in Figure. 5, which demonstrates that LayoutCoT exhibits
outstanding layout control capabilities and generation qual-
ity. LayoutCoT not only effectively meets various input con-
straints—including element type constraints, size constraints,
and element relationship constraints—but also produces visu-
ally pleasing layouts by avoiding overlap between elements
and enhancing overall alignment. These findings collectively
validate the effectiveness of LayoutCoT.
4.6. Results on Text-to-Layout
The quantitative metrics are presented in Table 4. Be-
cause text-to-layout is one of the most challenging tasks in
layout generation, LayoutCoT falls short in the mIoU metriccompared to the parse-then-place [ 27] method. However, it
performs exceptionally well on the remaining three metrics.
In many scenarios, LayoutCoT more effectively fulfills tex-
tual descriptions and excels in aesthetics, avoiding element
overlap and maintaining element alignment.
4.7. Ablation Studies
To demonstrate the effectiveness of our approach, we
conducted an overall ablation study, as well as ablations
focusing on the selection method for prompting examples in
the Layout-aware RAG and on the choice of the number of
RAG layouts kin the CoT module. All our ablation studies
are conducted on the PKU dataset within the representative
Content-Aware layout generation task.
Overall ablation study. In Table. 5, we demonstrate the
role of each module. It is evident that the Layout-aware
RAG and the CoT module complement each other, yielding
7

Table 3. Quantitative comparison with other methods on constraint-
explicit layout generation tasks. LayoutCoT†denotes the results
when using ’DeepSeek-R1’ as the model with the CoT module
removed.
RICO PubLayNet
Tasks Methods mIoU ↑ FID↓ Align↓Overlap ↓mIoU↑ FID↓ Align↓Overlap ↓
Gen-TBLT [22] 0.216 25.633 0.150 0.983 0.140 38.684 0.036 0.196
LayoutFormer++ [19] 0.432 1.096 0.230 0.530 0.348 8.411 0.020 0.008
LayoutPrompter [28] 0.604 2.134 0.0018 0.0285 0.619 1.648 0.0009 0.0037
LayoutCoT†0.718 4.970 0.0015 0.0208 0.795 2.608 0.0001 0.0027
LayoutCoT (Ours) 0.719 2.189 0.0008 0.0220 0.772 1.227 0.0001 0.0024
Gen-TSBLT [22] 0.604 0.951 0.181 0.660 0.428 7.914 0.021 0.419
LayoutFormer++ [19] 0.620 0.757 0.202 0.542 0.471 0.720 0.024 0.037
LayoutPrompter [28] 0.707 3.072 0.0020 0.0505 0.727 1.863 0.0007 0.076
LayoutCoT†0.782 1.252 0.0005 0.0331 0.844 1.701 0.0001 0.034
LayoutCoT (Ours) 0.795 3.506 0.0004 0.0305 0.847 1.430 0.0003 0.004
Gen-RCLG-LO [21] 0.286 8.898 0.311 0.615 0.277 19.738 0.123 0.200
LayoutFormer++ [19] 0.424 5.972 0.332 0.537 0.353 4.954 0.025 0.076
LayoutPrompter [28] 0.583 2.249 0.0007 0.0436 0.627 2.224 0.0008 0.011
LayoutCoT†0.724 2.399 0.0006 0.0301 0.779 1.853 0.0002 0.004
LayoutCoT (Ours) 0.725 3.569 0.0009 0.0145 0.769 1.408 0.0005 0.002
CompletionLayoutTransformer [11] 0.363 6.679 0.194 0.478 0.077 14.769 0.019 0.0013
LayoutFormer++ [19] 0.732 4.574 0.077 0.487 0.471 10.251 0.020 0.0022
LayoutPrompter [28] 0.541 2.656 0.0010 0.0405 0.754 1.282 0.0009 0.0008
LayoutCoT†0.682 2.236 0.008 0.0309 0.843 0.840 0.0007 0.0003
LayoutCoT (Ours) 0.716 2.112 0.0008 0.0117 0.770 0.531 4e-5 0.0006
RefinementRUITE [33] 0.811 0.107 0.133 0.483 0.781 0.061 0.029 0.020
LayoutFormer++ [19] 0.816 0.032 0.123 0.489 0.785 0.086 0.024 0.006
LayoutPrompter [28] 0.874 0.225 0.0011 0.1095 0.756 0.143 0.003 0.0006
LayoutCoT†0.869 0.876 0.0006 0.0714 0.801 0.265 0.001 0.0002
LayoutCoT (Ours) 0.890 0.353 0.0003 0.0520 0.839 0.141 0.0004 0.0003
Table 4. Quantitative comparison with baselines on text-to-layout.
LayoutCoT†denotes the results when using ’DeepSeek-R1’ as the
model with the CoT module removed.
Methods mIoU ↑ FID↓ Align↓ Overlap ↓
Mockup [15] 0.193 37.012 0.0059 0.4348
parse-then-place [27] 0.684 2.959 0.0008 0.1380
LayoutPrompter 0.174 4.773 0.0009 0.0107
LayoutCoT†0.164 1.018 0.0009 0.0068
LayoutCoT (Ours) 0.199 1.732 0.00006 0.0061
excellent results across all metrics. Regarding the higher
Overlap score when the CoT module is used, we provide
an explanation in Sec. 4.4. In reality, although the Overlap
metric increases, the CoT module effectively rectifies the
layout’s unreasonableness and lack of aesthetic appeal.
Table 5. Abalation study of each method.
RAG CoT Occ↓Rea↓Uti↑Align↓Und l↑Und s↑Ove↓ Val↑
✓ ✗ 0.251 0.171 0.237 0.0021 0.824 0.809 0.0005 0.997
✗✓ 0.209 0.161 0.231 0.0004 0.931 0.844 0.0014 0.999
✓ ✓ 0.207 0.074 0.289 0.0002 0.980 0.946 0.0013 1.000
The Method of the Layout-aware RAG. Table. 6 reports
the experimental results obtained with different RAG -based
strategies for selecting prompting examples. Specifically,
random denotes randomly selecting kexamples; the mIoU
strategy chooses examples according to the maximum IoU
between saliency maps; DreamSim [ 8] ranks candidates by
their DreamSim scores. The results show that random selec-
tion degrades layout -generation performance—sometimes
Figure 5. Qualitative Results for the Constraint-Explicit Layout
Generation Task. The RICO dataset features a wide variety of label
categories, making the completion task on RICO more challenging
compared to other task types, we conduct visualizations based on
the completion task of the RICO dataset. It is evident that Layout-
CoT designs more rational layouts, with better element overlap and
alignment compared to other methods.
performing even worse than omitting RAG entirely. By con-
trast, the mIoU strategy, which assesses relevance between
saliency maps, offers a modest improvement, while Dream-
Sim—considering both image and layout coherence—yields
further gains over mIoU. Finally, the ltsim method, which
measures overall layout similarity, proves the most effective
and achieves the best generation results.
Table 6. Abalation study of Layout-aware RAG.
RAG method Occ↓Rea↓Uti↑Align↓Und l↑Und s↑Ove↓ Val↑
- 0.209 0.161 0.231 0.0004 0.931 0.844 0.0014 0.999
random 0.299 0.184 0.241 0.0009 0.808 0.783 0.0023 0.997
mIoU 0.230 0.163 0.265 0.0003 0.951 0.894 0.0010 1.000
DreamSim [8] 0.252 0.086 0.273 0.0004 0.972 0.932 0.0008 0.999
ltsim 0.207 0.074 0.289 0.0002 0.980 0.946 0.0013 1.000
Selection of k.In Table. 7, we present the impact of differ-
8

ent numbers of prompting examples on the CoT module’s
performance. It can be observed that when k=2, the results
are relatively poor. This is because having fewer prompting
examples means the LLM lacks sufficient data references
for deep reasoning, thus impairing its inference capability.
On the other hand, when k=6, the outcome is essentially
the same as when k=4. We think that once the number of
prompting examples reaches a certain threshold, increasing
kfurther does not significantly stimulate deeper reasoning
in the LLM, and also consumes more resources. Therefore,
we choose k=4.
Table 7. Abalation study of the selection of k.
kOcc↓Rea↓Uti↑Align↓Und l↑Und s↑Ove↓ Val↑
2 0.232 0.119 0.282 0.0009 0.835 0.797 0.0019 0.999
40.207 0.074 0.289 0.0002 0.980 0.946 0.0013 1.000
6 0.216 0.075 0.301 0.0004 0.968 0.939 0.0010 1.000
Effectiveness of the Multi -Stage of CoT. To verify that
a multi -stage chain of thought is more effective than a
single -stage approach, we combined all sub -problems from
the multi -stage process into a single question and performed
a one -stage refinement. The quantitative metrics, presented
in Table. 8, show that the single -stage strategy performs con-
siderably worse than the multi -stage variant. This disparity
arises because layout generation follows a logical sequence:
resolving sub -problems step by step enhances layout ratio-
nality, whereas tackling multiple issues simultaneously is
impractical and yields limited improvements.
Table 8. Abalation study of the effectiveness of multi-stage.
Stage Occ↓Rea↓Uti↑Align↓Und l↑Und s↑Ove↓ Val↑
1 0.219 0.167 0.241 0.0005 0.933 0.851 0.0017 0.999
3 0.207 0.074 0.289 0.0002 0.980 0.946 0.0013 1.000
5. Conclusion
This paper focuses on unleashing the deep reasoning abili-
ties of LLMs in layout generation tasks without requiring any
training. To address the limitations of existing methods and
enhance models’ performance in layout generation, we pro-
pose LayoutCoT, which comprises three key components: a
Layout-aware RAG, a Coarse Layout Generator, and a Chain
of Thought for Layout Refinement. Specifically, we select
prompting examples through a carefully designed RAG ap-
proach to generate a coarse layout, then further decompose
the layout task into multiple simpler subproblems. By inte-
grating the RAG results, we refine the coarse layout. Our
method activates the deep reasoning capabilities of LLMs in
layout generation, achieving state-of-the-art results on five
public datasets spanning three different types of layout tasks.
These findings demonstrate that LayoutCoT is a highly versa-tile, data-efficient, and training-free approach that produces
high-quality layouts while satisfying given constraints.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo
Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al. Gpt-4 technical report. arXiv preprint
arXiv:2303.08774 , 2023. 1, 3
[2]Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. Language models are few-shot learners.
Advances in neural information processing systems , 33:
1877–1901, 2020. 1, 3
[3]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan,
Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri
Edwards, Yuri Burda, Nicholas Joseph, Greg Brock-
man, et al. Evaluating large language models trained
on code. arXiv preprint arXiv:2107.03374 , 2021. 1
[4]Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng,
Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang
Zhou, Te Gao, and Wangxiang Che. Towards rea-
soning era: A survey of long chain-of-thought for
reasoning large language models. arXiv preprint
arXiv:2503.09567 , 2025. 3
[5]Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu,
Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei
Cao, Jie Ma, et al. A survey on knowledge-oriented
retrieval-augmented generation. arXiv preprint
arXiv:2503.10677 , 2025. 3
[6]Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,
Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul
Barham, Hyung Won Chung, Charles Sutton, Sebastian
Gehrmann, et al. Palm: Scaling language modeling
with pathways. Journal of Machine Learning Research ,
24(240):1–113, 2023. 3
[7]Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani,
Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang,
and William Yang Wang. Layoutgpt: Compositional
visual planning and generation with large language
models. Advances in Neural Information Processing
Systems , 36:18225–18250, 2023. 3
[8]Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy
Chai, Richard Zhang, Tali Dekel, and Phillip Isola.
Dreamsim: Learning new dimensions of human vi-
sual similarity using synthetic data. arXiv preprint
arXiv:2306.09344 , 2023. 8
[9]Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza,
Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron
Courville, and Yoshua Bengio. Generative adversar-
ial nets. Advances in neural information processing
systems , 27, 2014. 1
9

[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing
reasoning capability in llms via reinforcement learning.
arXiv preprint arXiv:2501.12948 , 2025. 2
[11] Kamal Gupta, Justin Lazarow, Alessandro Achille,
Larry S Davis, Vijay Mahadevan, and Abhinav Shri-
vastava. Layouttransformer: Layout generation and
completion with self-attention. In Proceedings of the
IEEE/CVF International Conference on Computer Vi-
sion, pages 1004–1014, 2021. 1, 3, 8
[12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising
diffusion probabilistic models. Advances in neural
information processing systems , 33:6840–6851, 2020.
1, 3
[13] Daichi Horita, Naoto Inoue, Kotaro Kikuchi, Kota Ya-
maguchi, and Kiyoharu Aizawa. Retrieval-augmented
layout transformer for content-aware layout generation.
InProceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition , pages 67–76,
2024. 3, 7
[14] Hsiao Yuan Hsu, Xiangteng He, Yuxin Peng, Hao
Kong, and Qing Zhang. Posterlayout: A new bench-
mark and approach for content-aware visual-textual
presentation layout. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recogni-
tion, pages 6018–6026, 2023. 1, 2, 3, 6, 7
[15] Forrest Huang, Gang Li, Xin Zhou, John F Canny,
and Yang Li. Creating user interface mock-ups from
high-level text descriptions with deep-learning models.
arXiv preprint arXiv:2110.07775 , 2021. 3, 8
[16] Mude Hui, Zhizheng Zhang, Xiaoyi Zhang, Wenxuan
Xie, Yuwang Wang, and Yan Lu. Unifying layout gener-
ation with a decoupled diffusion model. In Proceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 1942–1951, 2023. 1, 3
[17] Shima Imani, Harsh Shrivastava, and Liang Du. Math-
ematical reasoning using large language models, 2024.
US Patent App. 18/144,802. 1
[18] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu
Otani, and Kota Yamaguchi. Layoutdm: Discrete diffu-
sion model for controllable layout generation. In Pro-
ceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition , pages 10167–10176,
2023. 1, 3
[19] Zhaoyun Jiang, Huayu Deng, Zhongkai Wu, Jiaqi Guo,
Shizhao Sun, Vuksan Mijovic, Zijiang Yang, Jian-
Guang Lou, and Dongmei Zhang. Unilayout: Taming
unified sequence-to-sequence transformers for graphic
layout generation. arXiv preprint arXiv:2208.08037 , 2
(3):4, 2022. 1, 8
[20] Akash Abdu Jyothi, Thibaut Durand, Jiawei He, Leonid
Sigal, and Greg Mori. Layoutvae: Stochastic scenelayout generation from a label set. In Proceedings of
the IEEE/CVF International Conference on Computer
Vision , pages 9895–9904, 2019. 3
[21] Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and
Kota Yamaguchi. Constrained graphic layout genera-
tion via latent optimization. In Proceedings of the 29th
ACM International Conference on Multimedia , pages
88–96, 2021. 1, 3, 8
[22] Xiang Kong, Lu Jiang, Huiwen Chang, Han Zhang,
Yuan Hao, Haifeng Gong, and Irfan Essa. Blt: bidirec-
tional layout transformer for controllable layout gen-
eration. In European Conference on Computer Vision ,
pages 474–490. Springer, 2022. 3, 8
[23] Hsin-Ying Lee, Lu Jiang, Irfan Essa, Phuong B Le,
Haifeng Gong, Ming-Hsuan Yang, and Weilong Yang.
Neural design network: Graphic layout generation with
constraints. In Computer Vision–ECCV 2020: 16th Eu-
ropean Conference, Glasgow, UK, August 23–28, 2020,
Proceedings, Part III 16 , pages 491–506. Springer,
2020. 1, 3
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel,
et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. Advances in neural information
processing systems , 33:9459–9474, 2020. 3
[25] Jianan Li, Jimei Yang, Aaron Hertzmann, Jianming
Zhang, and Tingfa Xu. Layoutgan: Generating graphic
layouts with wireframe discriminators. arXiv preprint
arXiv:1901.06767 , 2019. 1
[26] Jianan Li, Jimei Yang, Jianming Zhang, Chang Liu,
Christina Wang, and Tingfa Xu. Attribute-conditioned
layout gan for automatic graphic design. IEEE Trans-
actions on Visualization and Computer Graphics , 27
(10):4039–4048, 2020. 1, 3
[27] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Weijiang Xu,
Ting Liu, Jian-Guang Lou, and Dongmei Zhang. A
parse-then-place approach for generating graphic lay-
outs from textual descriptions. In Proceedings of the
IEEE/CVF International Conference on Computer Vi-
sion, pages 23622–23631, 2023. 3, 7, 8
[28] Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-
Guang Lou, and Dongmei Zhang. Layoutprompter:
awaken the design ability of large language models.
Advances in Neural Information Processing Systems ,
36:43852–43879, 2023. 1, 2, 3, 5, 6, 7, 8
[29] Thomas F Liu, Mark Craft, Jason Situ, Ersin Yumer,
Radomir Mech, and Ranjitha Kumar. Learning design
semantics for mobile apps. In Proceedings of the 31st
Annual ACM Symposium on User Interface Software
and Technology , pages 569–579, 2018. 2, 6
[30] Ziwei Liu, Liang Zhang, Qian Li, Jianghua Wu, and
Guangxu Zhu. Invar-rag: Invariant llm-aligned retrieval
10

for better generation. arXiv preprint arXiv:2411.07021 ,
2024. 3
[31] Yijia Luo, Yulin Song, Xingyao Zhang, Jiaheng Liu,
Weixun Wang, GengRu Chen, Wenbo Su, and Bo
Zheng. Deconstructing long chain-of-thought: A struc-
tured reasoning optimization framework for long cot
distillation. arXiv preprint arXiv:2503.16385 , 2025. 3
[32] Mayu Otani, Naoto Inoue, Kotaro Kikuchi, and Riku
Togashi. Ltsim: Layout transportation-based similar-
ity measure for evaluating layout generation. arXiv
preprint arXiv:2407.12356 , 2024. 2, 4
[33] Soliha Rahman, Vinoth Pandian Sermuga Pandian, and
Matthias Jarke. Ruite: Refining ui layout aesthetics us-
ing transformer encoder. In Companion Proceedings of
the 26th International Conference on Intelligent User
Interfaces , pages 81–83, 2021. 3, 8
[34] Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten
Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu
Liu, Romain Sauvestre, Tal Remez, et al. Code llama:
Open foundation models for code. arXiv preprint
arXiv:2308.12950 , 2023. 3
[35] Jaejung Seol, Seojun Kim, and Jaejun Yoo. Poster-
llama: Bridging design ability of langauge model
to contents-aware layout generation. arXiv preprint
arXiv:2404.00995 , 2024. 3
[36] Zecheng Tang, Chenfei Wu, Juntao Li, and Nan
Duan. Layoutnuwa: Revealing the hidden layout
expertise of large language models. arXiv preprint
arXiv:2309.09506 , 2023. 3
[37] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier
Martinet, Marie-Anne Lachaux, Timoth ´ee Lacroix,
Baptiste Rozi `ere, Naman Goyal, Eric Hambro, Faisal
Azhar, et al. Llama: Open and efficient foundation
language models. arXiv preprint arXiv:2302.13971 ,
2023. 3
[38] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need.
Advances in neural information processing systems , 30,
2017. 3
[39] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. Chain-of-thought prompting elicits reasoning in
large language models. Advances in neural information
processing systems , 35:24824–24837, 2022. 3
[40] Jason Wu, Siyan Wang, Siman Shen, Yi-Hao Peng, Jef-
frey Nichols, and Jeffrey P Bigham. Webui: A dataset
for enhancing visual ui understanding with web seman-
tics. In Proceedings of the 2023 CHI Conference on
Human Factors in Computing Systems , pages 1–14,
2023. 2, 6
[41] Kevin Yang, Dan Klein, Nanyun Peng, and Yuandong
Tian. Doc: Improving long story coherence with de-tailed outline control. arXiv preprint arXiv:2212.10077 ,
2022. 1
[42] Kevin Yang, Yuandong Tian, Nanyun Peng, and
Dan Klein. Re3: Generating longer stories with re-
cursive reprompting and revision. arXiv preprint
arXiv:2210.06774 , 2022. 1
[43] Guangcong Zheng, Xianpan Zhou, Xuewei Li, Zhon-
gang Qi, Ying Shan, and Xi Li. Layoutdiffusion: Con-
trollable diffusion model for layout-to-image gener-
ation. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages
22490–22499, 2023. 1
[44] Xinru Zheng, Xiaotian Qiao, Ying Cao, and Ryn-
son WH Lau. Content-aware generative modeling of
graphic design layouts. ACM Transactions on Graphics
(TOG) , 38(4):1–15, 2019. 3
[45] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes.
Publaynet: largest dataset ever for document layout
analysis. In 2019 International conference on docu-
ment analysis and recognition (ICDAR) , pages 1015–
1022. IEEE, 2019. 2, 6
[46] Min Zhou, Chenchen Xu, Ye Ma, Tiezheng Ge, Yuning
Jiang, and Weiwei Xu. Composition-aware graphic lay-
out gan for visual-textual presentation designs. arXiv
preprint arXiv:2205.00303 , 2022. 2, 6
[47] Min Zhou, Chenchen Xu, Ye Ma, Tiezheng Ge, Yuning
Jiang, and Weiwei Xu. Composition-aware graphic lay-
out gan for visual-textual presentation designs. arXiv
preprint arXiv:2205.00303 , 2022. 7
[48] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and
Mohamed Elhoseiny. Minigpt-4: Enhancing vision-
language understanding with advanced large language
models. arXiv preprint arXiv:2304.10592 , 2023. 3
11