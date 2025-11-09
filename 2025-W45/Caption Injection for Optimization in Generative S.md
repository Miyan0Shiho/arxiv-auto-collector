# Caption Injection for Optimization in Generative Search Engine

**Authors**: Xiaolu Chen, Yong Liao

**Published**: 2025-11-06 05:37:27

**PDF URL**: [http://arxiv.org/pdf/2511.04080v1](http://arxiv.org/pdf/2511.04080v1)

## Abstract
Generative Search Engines (GSEs) leverage Retrieval-Augmented Generation
(RAG) techniques and Large Language Models (LLMs) to integrate multi-source
information and provide users with accurate and comprehensive responses. Unlike
traditional search engines that present results in ranked lists, GSEs shift
users' attention from sequential browsing to content-driven subjective
perception, driving a paradigm shift in information retrieval. In this context,
enhancing the subjective visibility of content through Generative Search Engine
Optimization (G-SEO) methods has emerged as a new research focus. With the
rapid advancement of Multimodal Retrieval-Augmented Generation (MRAG)
techniques, GSEs can now efficiently integrate text, images, audio, and video,
producing richer responses that better satisfy complex information needs.
Existing G-SEO methods, however, remain limited to text-based optimization and
fail to fully exploit multimodal data. To address this gap, we propose Caption
Injection, the first multimodal G-SEO approach, which extracts captions from
images and injects them into textual content, integrating visual semantics to
enhance the subjective visibility of content in generative search scenarios. We
systematically evaluate Caption Injection on MRAMG, a benchmark for MRAG, under
both unimodal and multimodal settings. Experimental results show that Caption
Injection significantly outperforms text-only G-SEO baselines under the G-Eval
metric, demonstrating the necessity and effectiveness of multimodal integration
in G-SEO to improve user-perceived content visibility.

## Full Text


<!-- PDF content starts -->

Caption Injection for Optimization in Generative
Search Engine
1stXiaolu Chen
School of Cyber Science and Technology
University of Science and Technology of China
Hefei, China
xiaoluchen@mail.ustc.edu.cn2ndYong Liao*
School of Cyber Science and Technology
University of Science and Technology of China
Hefei, China
yliao@ustc.edu.cn
Abstract—Generative Search Engines (GSEs) leverage
Retrieval-Augmented Generation (RAG) techniques and Large
Language Models (LLMs) to integrate multi-source information
and provide users with accurate and comprehensive responses.
Unlike traditional search engines that present results in ranked
lists, GSEs shift users’ attention from sequential browsing
to content-driven subjective perception, driving a paradigm
shift in information retrieval. In this context, enhancing the
subjective visibility of content through Generative Search Engine
Optimization (G-SEO) methods has emerged as a new research
focus. With the rapid advancement of Multimodal Retrieval-
Augmented Generation (MRAG) techniques, GSEs can now
efficiently integrate text, images, audio, and video, producing
richer responses that better satisfy complex information needs.
Existing G-SEO methods, however, remain limited to text-based
optimization and fail to fully exploit multimodal data. To address
this gap, we propose Caption Injection, the first multimodal
G-SEO approach, which extracts captions from images and
injects them into textual content, integrating visual semantics to
enhance the subjective visibility of content in generative search
scenarios. We systematically evaluate Caption Injection on
MRAMG, a benchmark for MRAG, under both unimodal and
multimodal settings. Experimental results show that Caption
Injection significantly outperforms text-only G-SEO baselines
under the G-Eval metric, demonstrating the necessity and
effectiveness of multimodal integration in G-SEO to improve
user-perceived content visibility.
Index Terms—Generative Search Engine Optimization, Prompt
Engineering, Large Language Model, Image Caption
I. INTRODUCTION
Generative Search Engines (GSEs) integrate the advan-
tages of Retrieval-Augmented Generation (RAG) and Large
Language Models (LLMs), enabling a deeper understanding
of user queries and selective integration of retrieved results
to provide accurate and comprehensive responses. Compared
with traditional search engines, GSEs retain powerful re-
trieval capabilities while leveraging semantic understanding
and generative abilities to produce a fundamentally differ-
ent mechanism for presenting search results. A schematic
illustration is shown in Fig.1. Traditional search engines
typically display retrieved web content sources in ranked
lists based on relevance, where the most relevant sources
appear at the top. In contrast, GSEs further parse the retrieved
sources, semantically extract and synthesize query-relevant
Corresponding authorsegments, and provide holistic responses enriched with cita-
tion references. With the growing complexity and diversity
of user information needs, Multimodal Retrieval-Augmented
Generation (MRAG) techniques have emerged and rapidly
evolved, endowing GSEs with enhanced capabilities for cross-
modal understanding and information integration. By jointly
processing text, images, audio, and video, GSEs can deliver
richer, more informative responses that better satisfy complex
user intents. Consequently, the rise of GSEs is transforming
how users interact with information, shifting their focus from
sequential browsing of multiple sources to directly holistic per-
ception of generated content that aligns more closely with their
information needs. This need-oriented presentation mechanism
enables more focused and efficient information acquisition,
significantly reducing distractions from irrelevant content. As a
result, GSEs are increasingly becoming a mainstream channel
for information access in the generative era.
Similar to the pivotal role of Search Engine Optimization
(SEO) in traditional retrieval systems, Generative Search En-
gine Optimization (G-SEO) has emerged as a key research
direction in generative search environments, whose objective
is to enhance the subjective visibility of content source within
GSE responses, ensuring that certain content receives greater
user attention in the generated results. However, the black-
box nature of LLMs makes the visibility mechanism within
GSEs highly complex and difficult to control, introducing
new challenges in optimizing content exposure. Although
traditional SEO techniques are well-established [1], [2], they
primarily target ranking-based retrieval scenarios and are ill-
suited for improving subjective visibility under generative
retrieval scenarios. Recent studies have begun to explore GSE-
oriented optimization approaches [3]–[5]. Nevertheless, most
of these methods focus solely on text-level optimization, such
as injecting specific textual patterns [6], [7] or reconstructing
textual semantics [8], [9]. As these approaches overlook the
rich potential of multimodal data, their ability to improve
content visibility is thereby fundamentally limited.
To bridge this gap, we propose Caption Injection, the
first multimodal G-SEO method. Caption Injection leverages
Prompt Engineering to project visual semantics from images
into the natural language space and seamlessly integrate them
into textual content, achieving cross-modal fusion that en-arXiv:2511.04080v1  [cs.IR]  6 Nov 2025

hances the subjective visibility of content. We design experi-
ments under both unimodal and multimodal GSE settings and
systematically evaluate the proposed method on the MRAMG,
a benchmark of MRAG, using the G-Eval evaluation frame-
work. Experimental results demonstrate that Caption Injection
consistently outperforms text-only G-SEO baselines across
both scenarios, exhibiting superior cross-modal adaptability
and confirming the critical role of multimodal integration in
enhancing content visibility within GSEs. In summary, the key
contributions of our work are threefold:
•We extend the G-SEO task from a unimodal to a mul-
timodal setting, systematically defining the multimodal
G-SEO problem and constructing a corresponding evalu-
ation environment.
•We propose Caption Injection, the first multimodal G-
SEO approach, which integrates visual semantics through
cross-modal semantic injection to enhance content visi-
bility in generative retrieval.
•Extensive experiments on the MRAG benchmark
MRAMG show that Caption Injection achieves signifi-
cant improvements over text-based baselines under both
unimodal and multimodal scenarios, underscoring the
essential role of multimodal information in advancing G-
SEO.
II. RELATEDWORK
A. Text-based Generative Search Engine Optimization
SEO techniques have matured over years of development
[1], [2], [10]. Traditional SEO methods typically influence
the ranking mechanisms of retrieval systems by leveraging
multiple webpage-level features, such as keywords [11], [12],
webpage structure [13], [14], and ranking strategies [15],
[16]. However, when LLMs are incorporated into GSEs as
response generation components, they exhibit implicit content
preferences due to their probabilistic generation mechanisms.
In this setting, retrieval results are influenced more by la-
tent semantic priors than by explicit ranking functions. As
a result, SEO methods based on explicit signals, such as
keywords, are less effective at influencing the visibility of con-
tent sources, thereby limiting their applicability in generative
retrieval environments. In essence, optimization in generative
search has shifted from explicit signal control to implicit
semantic alignment. To address this challenge, researchers
have proposed the research direction of G-SEO [3], which
aims to improve content source visibility in generative retrieval
scenarios. Existing G-SEO methods primarily focus on seman-
tic interventions at the textual level and can be broadly grouped
into two categories. The first category comprises content
rewriting-based optimization methods, exemplified by GEO
[3]. These approaches reconstruct the textual semantics of
content source to better align with the generative mechanisms
of LLMs. Techniques include fine-tuning pre-trained models
[9], designing multi-agent collaboration frameworks [4], and
constructing role-based intent analysis prompts [8]. The sec-
ond category focuses on prompt injection-based optimizationmethods, which guide LLM generation by inserting carefully
designed semantic or sequential textual prompts [5], [7], [17].
These studies demonstrate that even minor prompt variations
can substantially affect the exposure of content sources in
generative responses. Despite these advances, most G-SEO
methods remain largely confined to the textual modality. They
rely exclusively on textual features for optimization while
largely overlooking the rich semantic information present
in other modalities, such as visual or auditory data. This
limitation significantly constrains optimization effectiveness
in MRAG scenarios. To address this gap, we propose Cap-
tion Injection, an innovative multimodal G-SEO approach
that injects image semantics into textual content to facilitate
cross-modal optimization, thereby effectively enhancing the
subjective visibility of content sources within generative search
environments.
B. Multimodal Content Understanding
The widespread multimodal data has driven rapid develop-
ment in multimodal learning, and the associated challenges
and progress have been extensively explored [18]–[20]. In
practical applications of GSEs, retrieval outputs are typically
text-centric, with images serving as supplementary contextual
cues, as shown in Fig.1. As a core technology supporting
multimodal GSEs, MRAG often maps images into a textual
semantic space to assist response generation [21]–[23], [37].
Therefore, multimodal G-SEO tasks must leverage existing
multimodal research approaches, with a core challenge being
how to model and generate semantics from images to text. This
requires the model to deeply understand content across modali-
ties and integrate multimodal information in a unified semantic
space, thereby highlighting key information while suppressing
noise. To this end, we further examine a series of repre-
sentative studies in recent multimodal content understanding
research. These works typically rely on visual–language pre-
training models with semantic alignment mechanisms, such
as the CLIP [24] and BLIP series [25], [26], which achieve
semantic fusion between vision and language through cross-
modal representation learning and provide knowledge sup-
port for downstream tasks [27], [28]. Among these studies,
image captioning is one of the classic multimodal content
understanding tasks [27], [29], aiming to generate more fine-
grained descriptions of images. Recent studies have improved
generation performance by constructing positive and negative
sample data augmentation strategies [30]–[32]. In particular,
Capsfusion [33], [34] further explore explicitly injecting visual
knowledge into structured text descriptions to generate more
semantically precise and contextually aligned captions. These
studies provide important insights into our proposed cross-
modal optimization method. Unlike image captioning, multi-
modal retrieval and MRAG tasks not only focus on modeling
visual and language features but also emphasize semantic
matching and joint reasoning across modalities. A common
approach is to map images into the textual space and interact-
ing with the query text. Some works integrate multiple image
captions and contextual information to generate more compre-

Fig. 1. Comparison of result presentation across different types of search engines. Traditional search engines (blue section on the left) display retrieved web
content sources in a ranked list, where higher-ranked results are typically more relevant to the query. GSEs retrieve relevant content sources and leverage
LLMs to generate comprehensive responses with cited references. Compared with unimodal GSEs (green section in the middle) that process only textual
information, multimodal GSEs (yellow section on the right) jointly interpret textual and visual information, producing responses with richer semantics and
higher information density.
hensive text, thereby improving retrieval accuracy [21]–[23],
[35]. Other studies encode images and jointly decode them
with text to generate responses [21], [22], [36], [37]. These
methods underscore the importance of cross-modal alignment
as a foundational principle for designing effective multimodal
GSEs. Beyond basic semantic alignment, researchers have
explored cross-modal intent understanding to enhance higher-
level semantic fusion. For example, CPAD [38] strengthens
the alignment of visual and textual features via hierarchical
modeling, BEAR framework [39] jointly models emotions
and intentions through multimodal mapping and completion,
and CAGC [40] mitigates inconsistencies caused by modality
gaps using global-context-guided contrastive learning. These
high-level cross-modal modeling studies further emphasize
the essential role of semantic fusion in complex multimodal
understanding. Overall, multimodal content understanding has
made significant progress. However, systematic research on
explicitly incorporating multimodal information into G-SEO
tasks remains limited. Motivated by this, we aim to inject
image semantics into the generative optimization process,
constructing a cross-modal G-SEO mechanism that offers a
more comprehensive and fine-grained perspective for advanc-
ing multimodal optimization within GSE.III. METHODOLOGY
A. Generative Search Engine Optimization Assumption
To clarify the research context and methodological assump-
tions, we briefly introduce the task formulation of G-SEO,
which is defined under the GSE setting. A GSE system
generates responses to user queries with RAG and LLM,
whose overall workflow is illustrated in Fig.1. Specifically,
given a user queryq, the GSE retrieves a set of web content
sourcesS=Retrieval(q) ={s 1, s2, . . . , s N}, and then
invokes its LLM module to synthesize an integrated coherent
natural-language responser=generate(S, q), which consists
of a series of sentences with supporting citations, represented
asr={l 1, l2, . . . , l m}, where each sentencel k(1≤k≤m)is
associated with one or more cited sourcesS t⊆S(1≤ |S t| ≤
N). In multimodal GSE scenarios, certain sentences may
be accompanied by semantically relevant images to enhance
both content presentation and perceptual understanding. GSE
responses are presented as coherent paragraphs, rather than
being ordered purely by the relevance between each source
snand queryq. Consequently, users’ attention is no longer
solely guided by positional order but is instead influenced
by their subjective reading preferences and perceptual focus.
Based on this observation, we define subjective visibility as a
key metric in GSE scenarios, which measures the distribution
of user attention and perceptual bias across content sources
during information consumption.We view G-SEO under mul-
timodal GSE as a new optimization challenge, since the LLM

must jointly model the complex semantic interactions between
visual and textual modalities in an inherently different process
from purely text-to-text relevance modeling. Building upon
prior studies [3], [8], we follow their evaluation protocols
and GSE setup, and extend these formulations to the mul-
timodal context. To address this challenge, we propose Cap-
tion Injection, the first multimodal G-SEO method designed
specifically for multimodal GSE scenarios. By mapping image
features into the natural language space and injecting them
into textual content, Caption Injection enables cross-modal
semantic fusion, thereby enhancing optimization effectiveness
and improving the subjective visibility of content sources in
both unimodal and multimodal GSE settings.
Fig. 2. Illustration of the Caption Injection pipeline. The image of the web
content source is first captioned by visual-language models (VLMs), mapping
visual representations into the natural language space. The textual content is
then injected with the rewritten caption leveraging LLMs, enabling G-SEO
with integrated multimodal information.
B. Caption Injection
Similar to previous G-SEO studies, we treat textual con-
tent as the primary optimization target, since text plays a
dominant role in the GSEs’ responses. To extend G-SEO
into multimodal scenarios by effectively leveraging image
semantics for textual content, we propose a visual-semantic
injection method for G-SEO, termed Caption Injection. The
core idea is to extract semantically relevant visual descriptions
and naturally integrate them into the textual content to achieve
cross-modal optimization. Unlike prior rewriting-based G-SEO
approaches, we argue that visual information is often partial
and sparse in semantic expression, making it less suitable as
the main optimization driver, yet highly valuable as a seman-
tic complement. Therefore, we utilize only the text-relevant
portion of visual semantics to enhance the source content
selectively, rather than rewriting the entire text based on image
information. Inspired by CapsFusion [33] and VeCLIP [34],
we design a structured three-stage semantic injection pipeline:(1) Structural Generation, (2) Alignment Refinement, and (3)
Semantic Injection, as illustrated in Fig.2. The entire process is
implemented through carefully designed prompt engineering,
with detailed prompt templates provided in Table I–III. For
clarity, we present only the core prompt logic in this paper,
while the unified system and role configurations used as LLM
parameters remain consistent and are not elaborated. Complete
prompt templates will be released with our open-source code.
(1) Structural Generation. Although image captions are of-
ten used as the basic semantic context, we observe that existing
captions in real-world data are frequently incomplete, superfi-
cial, or even inconsistent with the image content, making them
suboptimal for direct integration into the textual stream. To
address this, we first generate a structural caption for each im-
age, which captures the underlying semantic elements around
three key components: object, action, and scene. This triplet
design is inspired by recent advances in Cognitive Science
and Computer Vision [41]–[44], which have demonstrated
that these three dimensions effectively characterize essential
visual semantics. The resulting structural caption contains
only abstract semantic elements, without specific contextual
instantiations, serving as a reliable foundation for subsequent
refinement.
TABLE I
PROMPT FOR STRUCTURAL CAPTION GENERATION
Prompt for structural caption generation
Generate a concise and objective caption for this image, describing the
main objects, actions, and scene present.
(2) Alignment Refinement. After obtaining the structural
caption, we perform semantic alignment and refinement to
strengthen the correspondence between visual semantics and
textual content. Specifically, we extract semantically relevant
fragments from the text based on the three elements of
the structural caption, enabling alignment between textual
semantics and visual features. We then rewrite and expand the
caption accordingly to achieve a precise mapping of image
features into the natural language space. During this process,
we preserve the syntactic structure of the original structural
caption to maintain the LLM’s learned attention distribution
over semantic entities when interpreting image semantics,
thereby ensuring semantic stability under the GSE’s black-
box generation mechanism. The optimized output, termed the
refined caption, exhibits broader semantic coverage and more
natural linguistic expression, and serves as the high-quality
input for the semantic injection stage. To ensure alignment
quality, we perform random sampling of generated results,
which are independently reviewed by multiple experts based
on semantic similarity, where the final validation is determined
by consensus among reviewers.
(3) Semantic Injection. In the final stage, we control the
insertion position of the refined caption through prompt design,
allowing it to merge naturally into the textual segment exhibit-
ing the highest relevant. This process is fully automated by the

TABLE II
PROMPT FOR CAPTION REFINEMENT
Prompt for caption refinement
### Your task
Carefully read the source, and then rewrite the caption to make it more
expressive and attention-grabbing. Requirements are as follows:
- Retain the core subject, action, and scene of the original caption
(who/what/when/where/how).
- Enrich the caption only with the most relevant information from the
source, ignoring unrelated details.
### Input
Source:{source}
Original Caption:{caption original}
### Output
Rewritten Caption: [caption rewritten]
LLM, which determines the optimal insertion point according
to contextual dependencies. As a result, visual information is
explicitly transferred into the textual content, while maintain-
ing fluency and coherence. This approach enables controllable
enhancement of visual semantics without disrupting the orig-
inal structure, thereby improving the cross-modal consistency
and information richness of G-SEO generation.
TABLE III
PROMPT FOR CAPTION INJECTION
Prompt for caption refinement
### Your task
1. Insert the text only at these positions, ensuring smooth and coherent
context. 2. Do not delete or modify any other part of the given source or
the text, and do not add anything other than the given text.
### Input
Source: source
Text: text
### Output
Source: [Optimized Source]
By designing a three-stage generation–refinement–injection
pipeline, our proposed Caption Injection for G-SEO achieves
controllable visual-semantic augmentation while preserving
textual dominance. The method effectively balances cross-
modal alignment with generation quality, offering a practical
yet extensible approach for advancing future multimodal G-
SEO research.
IV. EXPERIMENTS ANDRESULTS
A. Experimental Setting
1) Generative Search Engine Simulation:Our work aims to
explore a general G-SEO optimization approach that can be
applied to both unimodal and multimodal GSEs. To eliminate
the influence of retrieval differences on evaluation results, we
focus solely on the content generation process. Experiments
are conducted on existing datasets, and the GSE workflow isconfigured as a single-turn response generation process with-
out a retrieval stage. In the unimodal GSE scenario, we gen-
erate responses using only the textual part of content sources
and an LLM, simulating a typical text-driven generative search
process. For the multimodal GSE scenario, we refer to and
compare several representative MRAG benchmark methods
[21]–[23], [37]. By comparatively analyzing generation-level
precision, recall, relevance, and fluency across representative
MRAG benchmarks, we observe that the configuration using
“text content + image captions + LLM” generally yields
higher response quality than that using “text content + original
images + multimodal LLM”. Based on this analysis, we adopt
the former configuration in our multimodal GSE simulation
to ensure stable and representative performance of G-SEO
methods under the simulated environment. To minimize the
interference of model hallucination and ensure fair comparison
across different methods, we employ the open-source GLM-
4-9B model, which is recognized for its low hallucination
rate1. This configuration provides stable responses and a more
objective foundation for evaluating the optimization effective-
ness of different G-SEO strategies, thereby reinforcing both
methodological fairness and experimental reliability.
2) Datasets:Multimodal G-SEO represents an emerging
research direction, and currently, no publicly available dataset
is specifically designed for this task. Since RAG serves as
the upstream process of GSEs and shares a highly similar
query–content structure with the generative response setting of
G-SEO, we base our experiments on the MRAG benchmark
framework. After comprehensive investigation and compara-
tive analysis, we identify the MRAMG benchmark [22] as
the most representative choice due to its balanced modality
composition and broad cross-domain coverage. MRAMG pro-
vides multimodal content sources covering both text and image
modalities and domain-diverse queries, thereby making it well-
suited to support the input diversity and content complexity
required by multimodal G-SEO tasks. MRAMG consists of
4,800 query-content pairs drawn from six distinct domains,
categorized into three difficulty levels according to task com-
plexity:
•Easy-level Web data: Includes the Wit, Wiki, and Web
datasets derived from Wikipedia articles, totaling 1,850
samples. These datasets focus on fundamental text–image
understanding within relatively simple and homogeneous
contexts.
•Medium-level academic data: includes the Arxiv dataset
containing 200 samples collected from papers published
on arXiv between 2023 and 2024. Each sample often in-
cludes multiple figures, designed to evaluate cross-modal
semantic fusion and academic knowledge generation.
•Hard-level lifestyle data: includes Recipe and Manual
datasets with 2,750 samples. These datasets feature high
image density and complex instruction-following tasks,
enabling assessment of model reasoning under challeng-
ing multimodal conditions.
1https://github.com/vectara/hallucination-leaderboard/

Except for the Web dataset, where each query corresponds
to two content sources, all other datasets contain a one-to-
one query–content pairing structure. During preprocessing,
we strictly follow the original MRAMG data structure and
splits, preserving all annotations, modality ratios, and image
resolutions without any additional cleaning, relabeling, or
augmentation, ensuring full reproducibility of our experiments.
3) Baselines:To comprehensively evaluate the effective-
ness of Caption Injection in enhancing the subjective visibility
of content sources under both unimodal and multimodal G-
SEO scenarios, we select and faithfully reproduce several rep-
resentative G-SEO methods as comparison baselines. Accord-
ing to their optimization strategies, existing G-SEO approaches
can be categorized into different types, from which we choose
representative implementations for experimental comparison:
•Traditional SEO [3]: Follows classical SEO principles
by guiding the generative model to explicitly incorporate
highly relevant keywords during the generation process,
simulating keyword layout optimization in traditional
SEO.
•Fluency Expression Optimization (Expression-
enhancement class) [3]: Refines and rewrites the
textual content to improve fluency and naturalness
without altering the original semantic meaning, thereby
enhancing readability and stylistic quality.
•Statistics-based and Quotation-based Addition Optimiza-
tion (Content-enrichment class) [3]: Enriches the content
by selectively inserting factual statistics or quotations
from authoritative sources, aiming to increase the infor-
mativeness and persuasive strength of the text.
All text-only baselines are rigorously reproduced from their
original implementations to ensure consistency with prior
work. To guarantee experimental fairness, both the baseline
methods and our proposed Caption Injection are implemented
under the same generative backbone model, GLM-4-9B. For
samples in the MRAMG benchmark where image descriptions
are missing, we employ Qwen-2.5-VL-7B2to automatically
generate the corresponding visual captions, ensuring the com-
pleteness and semantic consistency of multimodal inputs.
4) Evaluation Metrics:In the G-SEO task, users primar-
ily interpret the information conveyed by content sources
through overall comprehension and perception. Therefore, the
subjective visibility of content sources serves as the most
essential metric for evaluating the optimization effectiveness
of G-SEO methods. To objectively assess how different G-
SEO strategies enhance the subjective visibility of content
sources, we adopt G-Eval 2.0 [8] as the subjective evaluation
framework, which is specifically adapted for the G-SEO
scenario. G-Eval [45] is a prompt-based evaluation metric
that aligns well with human judgment. It instructs LLMs to
conduct assessments through structured prompts with step-
by-step guidance and has been widely applied to evaluate
visibility improvement in G-SEO tasks [3]. G-Eval 2.0 [8], an
2https://github.com/BradyFU/Awesome-Multimodal-Large-Language-
Models/tree/Evaluation?tab=readme-ov-fileenhanced version tailored for G-SEO evaluation, refines this
framework by assessing content sources along seven subjective
dimensions: relevance, fluency, diversity, uniqueness, click-
follow likelihood, positional salience, and content volume.
Each dimension is rated on a six-level scale from 0 to 5,
following the standard evaluation protocol in prior work. The
final subjective visibility score is computed as the average
across all seven dimensions. To measure the optimization
effectiveness of each G-SEO method, we follow prior studies
and calculate the relative improvement ratio of subjective
visibility, formulated as follows, wheresandrdenote the
content source and its corresponding GSE response before
optimization, ands′andr′represent the optimized source and
regenerated response, respectively.impression s(r)indicates
either the sub-dimensional score or the averaged subjective
visibility score. To mitigate randomness in evaluation results,
each sample’s optimization effect is averaged over three in-
dependent runs, and the mean score is reported as the final
performance of the G-SEO method on that data instance.
improvement (s,s′)=impressions′(r′)−impression s(r)
impr s(r)+1×100 %
B. Results and Analysis
1) Main Result:Table IV presents the comparative results
between our proposed Caption Injection and other text-only
G-SEO baselines on the MRAMG benchmark under both
unimodal and multimodal GSE settings. Several key findings
can be observed. Across both settings, Caption Injection
consistently achieves the highest improvement in subjective
visibility, with relative gains of +1.85% and +1.09% in uni-
modal and multimodal scenarios, respectively. These gains
exceed the second-best method by 14% and 18%, demon-
strating the substantial benefit of incorporating multimodal
information for G-SEO optimization. Meanwhile, all G-SEO
methods show a clear performance degradation when moving
from unimodal to multimodal settings, empirically confirming
our hypothesis that multimodal G-SEO is a significantly
more challenging subtask due to the complexity of cross-
modal information grounding. Among the four multimodal QA
datasets, MRAMG-Wit, MRAMG-Wiki, MRAMG-Web, and
MRAMG-Arxiv, Caption Injection achieves highly competi-
tive results, indicating its strong ability to leverage visual-
grounded captions for reasoning and content optimization.
In contrast, all methods exhibit the weakest improvements
on MRAMG-Manual, whereas even in the similarly difficult
MRAMG-Recipe, G-SEO models maintain moderate effec-
tiveness. This discrepancy can be attributed to the extreme
text length of MRAMG-Manual whose average is 6,365.4
characters [22], which makes it challenging for generative
models to extract key optimization cues. This observation
highlights long-text optimization as an open research problem
in future G-SEO studies. The Traditional SEO performs no-
ticeably worse than all generative optimization methods, with
its maximum visibility improvement limited to 0.64% across
all datasets. This finding aligns with previous GEO and RAID
G-SEO studies [3], [8], reaffirming that conventional keyword-

based SEO techniques are no longer effective in generative
search environments.
2) Ablation Study:To verify the effectiveness of the refined
captions produced by the Caption Injection in enhancing the
subjective visibility of content sources, we conducted an ab-
lation study by comparing three variants of injected captions:
the original image caption, the caption generated by VLM,
and the refined caption. The caption injection mechanism
was kept identical across all variants to ensure comparability.
The results on the MRAMG-Arxiv dataset are reported in
Table V. As shown in Table V, the refined captions consis-
tently yield higher subjective visibility scores than both the
original and VLM-generated captions, in both unimodal and
multimodal GSE settings. This result indicates that refinement
enriches captions with more semantically informative and
LLM-usable content, thus facilitating more effective generative
optimization. Interestingly, in certain subjective dimensions,
particularly under the multimodal setting, the VLM-generated
captions occasionally outperform the refined ones. We attribute
this to the semantic alignment preference of the LLMs used
in the GSE response generation process. Current multimodal
understanding techniques often capture only shallow cross-
modal semantics, leading the models to favor representations
that align better with their internal semantic space rather than
those aligned with human-perceived meaning. These findings
highlight the importance of semantic-level caption refinement
in improving G-SEO performance, while also underscoring a
key limitation of current multimodal alignment capabilities,
which we will take as an open research direction for future
G-SEO.
V. LIMITATION
Our experiments validate the effectiveness of multimodal in-
formation fusion in the G-SEO task. However, we also observe
that the optimization improvement of existing G-SEO methods
under multimodal GSE scenarios remains notably lower than
that in unimodal settings. Although Caption Injection outper-
forms other baseline methods, its overall optimization effect
still has room for improvement. We therefore consider mul-
timodal G-SEO to be a highly challenging research problem,
which further highlights the exploratory value of our work
in the early stage of multimodal G-SEO research. Currently,
while Caption Injection achieves a fusion of visual and textual
semantics, this integration remains at a relatively shallow
mapping level. The absence of a deep interaction mechanism
between visual and textual features results in insufficient
utilization of visual information, leading to limited enhance-
ment of textual content by the generated captions. Future
work should focus on developing deeper cross-modal feature
fusion mechanisms under the G-SEO framework to construct
a more unified multimodal semantic space. Moreover, another
possible reason for the limited optimization performance lies
in the unmodeled GSE preference toward input data. Without
considering such inherent model biases, the optimized outputs
may fail to align with the internal semantic representations of
the GSE, thereby constraining their influence on GSE semanticunderstanding. This observation suggests that future research
could explore cross-model optimization strategies to better
interpret and adapt to the black-box nature of GSEs.
VI. CONCLUSION
The emergence of GSEs has shifted users’ information
acquisition channels from query-based linear ranking toward
subjective perception of paragraph-level content, motivating
researchers to explore G-SEO to enhance the subjective visi-
bility of content. With the advancement of MRAG, GSEs are
now capable of processing both textual and visual inputs. How-
ever, this also introduces new challenges for G-SEO: how to
effectively improve the subjective visibility of content sources
within multimodal semantic spaces. To address this, we extend
the G-SEO task to multimodal settings for the first time and
propose Caption Injection, the first G-SEO method designed
for multimodal GSE scenarios to integrate visual and textual
semantics, offering a transferable semantic alignment strategy
for multimodal G-SEO. Experimental results demonstrate that
visual information can significantly enhance the subjective
visibility of content sources in both unimodal and multimodal
GSE settings, confirming the potential of multimodal semantic
fusion in G-SEO tasks. Although we have not yet explored
deep multimodal feature fusion mechanisms in G-SEO, our
work offers a practical approach and supporting evidence for
incorporating visual information into G-SEO research. In the
future, we plan to further investigate fine-grained modeling
of visual semantic interactions, taking model preference into
account for interpretable and robust multimodal optimization
strategies to further advance multimodal G-SEO research.
VII. ETHICALSTATEMENT
We strictly adhere to the ethical standards of academic
research in artificial intelligence, including the principles
of transparency in data usage, model utilization, and result
presentation. All LLMs used in this study are sourced from
open communities, and the experimental datasets are publicly
available open datasets. We fully comply with the relevant
copyright and licensing terms during their use. This research is
conducted solely to explore and validate methods for optimiz-
ing the visibility of web content sources in GSE, rather than
manipulating search results, and it has not been applied to any
real-world search engine operations or other potential misuse.
All outputs generated by the LLMs are used exclusively for
academic research and methodological validation, without any
commercial use or value judgment.
VIII. AI-GENERATEDCONTENTACKNOWLEDGEMENT
In this work, we utilized Dreamina AI to create the il-
lustrations in Fig. 1 and 2, specifically depicting the neural
architecture of the LLM and VLM. Additionally, ChatGPT
was employed to refine the text and prompts within the
manuscript. No other content in this paper was generated or
modified using AI tools.

TABLE IV
COMPARISON OF THE RELATIVE IMPROVEMENT IN SUBJECTIVE VISIBILITY ACHIEVED BY DIFFERENTG-SEOMETHODS UNDER UNIMODAL AND
MULTIMODALGSESETTINGS ON THEMRAMGBENCHMARK.
Method Arxiv Manual Recipe Web Wiki WIT
tran seo -0.74 -8.93 0.64 -3.05 -0.14 -1.58
flue expr 0.98-5.88 2.210.04 0.36 0.09
quat addi 0.55 -11.38 1.970.460.970.70
stat addi -1.68 -11.26 -1.75 -4.50 -1.93 -3.63
capt addi1.09-9.12 1.85 0.181.100.68
TABLE V
EVALUATION OFG-SEOMETHODS ON DIFFERENT METRICS(VALUES×100)
Method Rele. Infl. Dive. Uniq. Clic. Sub.Posi. Sub.Volu. Average
capt addi(original)0.79 1.120.303.670.600.92 1.87 1.18
capt addi(generated) 0.70 0.91 0.24 3.440.700.64 1.65 1.06
capt addi(rewriten) 0.67 1.000.413.44 0.59 0.71 1.69 1.11
REFERENCES
[1] N. Yalc ¸ın and U. K ¨ose, “What is search engine optimization: Seo?”
Procedia - Social and Behavioral Sciences, vol. 9, pp. 487–493,
2010, world Conference on Learning, Teaching and Administration Pa-
pers. [Online]. Available: https://www.sciencedirect.com/science/article/
pii/S1877042810022901
[2] F. ALMUKHTAR, N. MAHMOODD, and S. KAREEM, “Search engine
optimization: A review,”Applied Computer Science, vol. 17, no. 1, p.
70–80, Mar. 2021. [Online]. Available: https://ph.pollub.pl/index.php/
acs/article/view/3098
[3] P. Aggarwal, V . Murahari, T. Rajpurohit, A. Kalyan, K. Narasimhan, and
A. Deshpande, “Geo: Generative engine optimization,” inProceedings
of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining, ser. KDD ’24. New York, NY , USA: Association for
Computing Machinery, 2024, p. 5–16. [Online]. Available: https://doi.
org/10.1145/3637528.3671900
[4] Q. Chen, J. Chen, H. Huang, Q. Shao, J. Chen, R. Hua, H. Xu,
R. Wu, R. Chuan, and J. Wu, “Beyond keywords: Driving generative
search engine optimization with content-centric agents,” 2025. [Online].
Available: https://arxiv.org/abs/2509.05607
[5] S. Pfrommer, Y . Bai, T. Gautam, and S. Sojoudi, “Ranking manipulation
for conversational search engines,” inProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Processing, Y . Al-
Onaizan, M. Bansal, and Y .-N. Chen, Eds. Miami, Florida, USA:
Association for Computational Linguistics, Nov. 2024, pp. 9523–9552.
[Online]. Available: https://aclanthology.org/2024.emnlp-main.534/
[6] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and
M. Fritz, “Not what you’ve signed up for: Compromising real-world llm-
integrated applications with indirect prompt injection,” inProceedings
of the 16th ACM Workshop on Artificial Intelligence and Security,
ser. AISec ’23. New York, NY , USA: Association for Computing
Machinery, 2023, p. 79–90. [Online]. Available: https://doi.org/10.1145/
3605764.3623985
[7] A. Kumar and H. Lakkaraju, “Manipulating large language models to
increase product visibility,” 2024. [Online]. Available: https://arxiv.org/
abs/2404.07981
[8] X. Chen, H. Wu, J. Bao, Z. Chen, Y . Liao, and H. Huang, “Role-
augmented intent-driven generative search engine optimization,” 2025.
[Online]. Available: https://arxiv.org/abs/2508.11158
[9] F. L ¨uttgenau, I. Colic, and G. Ramirez, “Beyond seo: A transformer-
based approach for reinventing web content optimisation,” 2025. [On-
line]. Available: https://arxiv.org/abs/2507.03169
[10] K. I. Roumeliotis and N. D. Tselikas, “An effective seo techniques and
technologies guide-map,”Journal of Web Engineering, vol. 21, no. 5,
pp. 1603–1649, July 2022.[11] A. P. Kanara, P. Kumari, and B. R. Prathap, “Python driven keyword
analysis for seo optimization,” in2024 10th International Conference
on Advanced Computing and Communication Systems (ICACCS), vol. 1,
March 2024, pp. 1170–1176.
[12] P. Vadlapati, “Autotrendykeywords: Real-time ai-driven trend-based seo
using llms,” 2024.
[13] G. Chodak and K. Bła ˙zyczek, “Large language models for search engine
optimization in e-commerce,” inAdvanced Computing, D. Garg, J. J.
P. C. Rodrigues, S. K. Gupta, X. Cheng, P. Sarao, and G. S. Patel, Eds.
Cham: Springer Nature Switzerland, 2024, pp. 333–344.
[14] S. S. Shaffi and I. Muthulakshmi, “Search engine optimization by
using machine learning for web page classification,” in2022 Interna-
tional Conference on Augmented Intelligence and Sustainable Systems
(ICAISS), Nov 2022, pp. 342–349.
[15] V . Srinivas and P. Gowda, “A page rank-based analytical design of
effective search engine optimization,”Iaes International Journal of
Artificial Intelligence (Ij-Ai), vol. 14, no. 1, pp. 73–82, 2025.
[16] N. Bardas, T. Mordo, O. Kurland, and M. Tennenholtz, “Automatic
document editing for improved ranking,” inProceedings of the 48th
International ACM SIGIR Conference on Research and Development
in Information Retrieval, ser. SIGIR ’25. New York, NY , USA:
Association for Computing Machinery, 2025, p. 2779–2783. [Online].
Available: https://doi.org/10.1145/3726302.3730168
[17] F. Nestaas, E. Debenedetti, and F. Tram `er, “Adversarial search engine
optimization for large language models,” 2024. [Online]. Available:
https://arxiv.org/abs/2406.18382
[18] T. Baltru ˇsaitis, C. Ahuja, and L.-P. Morency, “Multimodal machine
learning: A survey and taxonomy,”IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 41, no. 2, pp. 423–443, Feb
2019.
[19] Y . Zhu, Y . Wu, N. Sebe, and Y . Yan, “Vision + x: A survey on
multimodal learning in the light of data,”IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9102–9122, Dec
2024.
[20] P. Xu, X. Zhu, and D. A. Clifton, “Multimodal learning with transform-
ers: A survey,”IEEE Transactions on Pattern Analysis and Machine
Intelligence, vol. 45, no. 10, pp. 12 113–12 132, Oct 2023.
[21] Z.-A. Ma, T. Lan, R.-C. Tu, Y . Hu, Y .-S. Zhu, T. Zhang, H. Huang,
Z. Wu, and X.-L. Mao, “Multi-modal retrieval augmented multi-modal
generation: Datasets, evaluation metrics and strong baselines,” 2025.
[Online]. Available: https://arxiv.org/abs/2411.16365
[22] Q. Yu, Z. Xiao, B. Li, Z. Wang, C. Chen, and W. Zhang, “Mramg-
bench: A comprehensive benchmark for advancing multimodal retrieval-
augmented multimodal generation,” inProceedings of the 48th In-
ternational ACM SIGIR Conference on Research and Development
in Information Retrieval, ser. SIGIR ’25. New York, NY , USA:

Association for Computing Machinery, 2025, p. 3616–3626. [Online].
Available: https://doi.org/10.1145/3726302.3730288
[23] Z. Zhu, D. Lee, H. Zhang, S. Sree Harsha, L. Feujio, A. Maharaj, and
Y . Li, “MuRAR: A simple and effective multimodal retrieval and answer
refinement framework for multimodal question answering,” inProceed-
ings of the 31st International Conference on Computational Linguistics:
System Demonstrations, O. Rambow, L. Wanner, M. Apidianaki, H. Al-
Khalifa, B. D. Eugenio, S. Schockaert, B. Mather, and M. Dras,
Eds. Abu Dhabi, UAE: Association for Computational Linguistics,
Jan. 2025, pp. 126–135. [Online]. Available: https://aclanthology.org/
2025.coling-demos.13/
[24] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal,
G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever,
“Learning transferable visual models from natural language supervi-
sion,” inProceedings of the 38th International Conference on Machine
Learning, ser. Proceedings of Machine Learning Research, M. Meila
and T. Zhang, Eds., vol. 139. PMLR, 18–24 Jul 2021, pp. 8748–8763.
[Online]. Available: https://proceedings.mlr.press/v139/radford21a.html
[25] J. Li, D. Li, C. Xiong, and S. Hoi, “BLIP: Bootstrapping language-image
pre-training for unified vision-language understanding and generation,”
inProceedings of the 39th International Conference on Machine Learn-
ing, ser. Proceedings of Machine Learning Research, K. Chaudhuri,
S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol.
162. PMLR, 17–23 Jul 2022, pp. 12 888–12 900. [Online]. Available:
https://proceedings.mlr.press/v162/li22n.html
[26] J. Li, D. Li, S. Savarese, and S. Hoi, “BLIP-2: Bootstrapping language-
image pre-training with frozen image encoders and large language mod-
els,” inProceedings of the 40th International Conference on Machine
Learning, ser. Proceedings of Machine Learning Research, A. Krause,
E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol.
202. PMLR, 23–29 Jul 2023, pp. 19 730–19 742. [Online]. Available:
https://proceedings.mlr.press/v202/li23q.html
[27] D. Caffagni, F. Cocchi, L. Barsellotti, N. Moratelli, S. Sarto, L. Baraldi,
L. Baraldi, M. Cornia, and R. Cucchiara, “The revolution of multimodal
large language models: A survey,” inFindings of the Association for
Computational Linguistics: ACL 2024, L.-W. Ku, A. Martins, and
V . Srikumar, Eds. Bangkok, Thailand: Association for Computational
Linguistics, Aug. 2024, pp. 13 590–13 618. [Online]. Available: https:
//aclanthology.org/2024.findings-acl.807/
[28] X. Zhang, J. Guo, S. Zhao, M. Fu, L. Duan, J. Hu, Y . X. Chng, G.-H.
Wang, Q.-G. Chen, Z. Xu, W. Luo, and K. Zhang, “Unified multimodal
understanding and generation models: Advances, challenges, and oppor-
tunities,” 2025. [Online]. Available: https://arxiv.org/abs/2505.02567
[29] J. Li, T. Tang, W. X. Zhao, J.-Y . Nie, and J.-R. Wen, “Pre-trained
language models for text generation: A survey,”ACM Comput. Surv.,
vol. 56, no. 9, Apr. 2024. [Online]. Available: https://doi.org/10.1145/
3649449
[30] H. Xu, P.-Y . Huang, X. Tan, C.-F. Yeh, J. Kahn, C. Jou, G. Ghosh,
O. Levy, L. Zettlemoyer, W.-t. Yih, S.-W. Li, S. Xie, and C. Fe-
ichtenhofer, “Altogether: Image captioning via re-aligning alt-text,” in
Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, Y . Al-Onaizan, M. Bansal, and Y .-N. Chen, Eds.
Miami, Florida, USA: Association for Computational Linguistics, Nov.
2024, pp. 19 302–19 318. [Online]. Available: https://aclanthology.org/
2024.emnlp-main.1075/
[31] N. Rotstein, D. Bensaid, S. Brody, R. Ganz, and R. Kimmel, “ Fuse-
Cap: Leveraging Large Language Models for Enriched Fused Image
Captions ,” in2024 IEEE/CVF Winter Conference on Applications of
Computer Vision (WACV). Los Alamitos, CA, USA: IEEE Computer
Society, Jan. 2024, pp. 5677–5688. [Online]. Available: https://doi.
ieeecomputersociety.org/10.1109/WACV57701.2024.00559
[32] L. Fan, D. Krishnan, P. Isola, D. Katabi, and Y . Tian, “Im-
proving clip training with language rewrites,” inAdvances in
Neural Information Processing Systems, A. Oh, T. Naumann,
A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds.,
vol. 36. Curran Associates, Inc., 2023, pp. 35 544–35 575. [On-
line]. Available: https://proceedings.neurips.cc/paper files/paper/2023/
file/6fa4d985e7c434002fb6289ab9b2d654-Paper-Conference.pdf
[33] Q. Yu, Q. Sun, X. Zhang, Y . Cui, F. Zhang, Y . Cao, X. Wang, and
J. Liu, “ CapsFusion: Rethinking Image-Text Data at Scale ,” in2024
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2024,
pp. 14 022–14 032. [Online]. Available: https://doi.ieeecomputersociety.
org/10.1109/CVPR52733.2024.01330[34] Z. Lai, H. Zhang, B. Zhang, W. Wu, H. Bai, A. Timofeev, X. Du,
Z. Gan, J. Shan, C.-N. Chuah, Y . Yang, and M. Cao, “Veclip: Improving
clip training via visual-enriched captions,” inComputer Vision – ECCV
2024: 18th European Conference, Milan, Italy, September 29–Octo-
ber 4, 2024, Proceedings, Part XLII. Berlin, Heidelberg: Springer-
Verlag, 2024, p. 111–127. [Online]. Available: https://doi.org/10.1007/
978-3-031-72946-1 7
[35] H. Zhu, J.-H. Huang, S. Rudinac, and E. Kanoulas, “Enhancing inter-
active image retrieval with query rewriting using large language models
and vision language models,” inProceedings of the 2024 International
Conference on Multimedia Retrieval, ser. ICMR ’24. New York,
NY , USA: Association for Computing Machinery, 2024, p. 978–987.
[Online]. Available: https://doi.org/10.1145/3652583.3658032
[36] O. Barbany, M. Huang, X. Zhu, and A. Dhua, “ Leveraging Large Lan-
guage Models for Multimodal Search ,” in2024 IEEE/CVF Conference
on Computer Vision and Pattern Recognition Workshops (CVPRW). Los
Alamitos, CA, USA: IEEE Computer Society, Jun. 2024, pp. 1201–
1210. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/
CVPRW63382.2024.00127
[37] W. Chen, H. Hu, X. Chen, P. Verga, and W. Cohen, “MuRAG: Multi-
modal retrieval-augmented generator for open question answering over
images and text,” inProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, Y . Goldberg, Z. Kozareva,
and Y . Zhang, Eds. Abu Dhabi, United Arab Emirates: Association
for Computational Linguistics, Dec. 2022, pp. 5558–5570. [Online].
Available: https://aclanthology.org/2022.emnlp-main.375/
[38] M. Ye, Q. Shi, K. Su, and B. Du, “Cross-modality pyramid alignment
for visual intention understanding,”IEEE Transactions on Image Pro-
cessing, vol. 32, pp. 2190–2201, 2023.
[39] Q. Yang, Q. Shi, T. Wang, and M. Ye, “Uncertain multimodal intention
and emotion understanding in the wild,” inProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition (CVPR), June
2025, pp. 24 700–24 709.
[40] K. Sun, Z. Xie, M. Ye, and H. Zhang, “Contextual augmented global
contrast for multimodal intent recognition,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), June 2024, pp. 26 963–26 973.
[41] M. Reger, O. Vrabie, G. V olberg, and A. Lingnau, “Actions at a glance:
The time course of action, object, and scene recognition in a free recall
paradigm,”Cognitive, Affective, & Behavioral Neuroscience, pp. 1–15,
2025.
[42] C. Bretti and P. Mettes, “Zero-shot action recognition from diverse
object-scene compositions,”arXiv preprint arXiv:2110.13479, 2021.
[43] D. Driess, J.-S. Ha, and M. Toussaint, “Deep visual reasoning: Learning
to predict action sequences for task and motion planning from an initial
scene image,”arXiv preprint arXiv:2006.05398, 2020.
[44] X. Wang and Z. Zhu, “Context understanding in computer vision: A
survey,”Comput. Vis. Image Underst., vol. 229, no. C, Mar. 2023.
[Online]. Available: https://doi.org/10.1016/j.cviu.2023.103646
[45] Y . Liu, D. Iter, Y . Xu, S. Wang, R. Xu, and C. Zhu, “G-eval: NLG
evaluation using gpt-4 with better human alignment,” inProceedings
of the 2023 Conference on Empirical Methods in Natural Language
Processing, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore:
Association for Computational Linguistics, Dec. 2023, pp. 2511–2522.
[Online]. Available: https://aclanthology.org/2023.emnlp-main.153/