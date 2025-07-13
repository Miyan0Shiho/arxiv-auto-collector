# MIRA: A Novel Framework for Fusing Modalities in Medical RAG

**Authors**: Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, Rao Mohammad Anwer

**Published**: 2025-07-10 16:33:50

**PDF URL**: [http://arxiv.org/pdf/2507.07902v1](http://arxiv.org/pdf/2507.07902v1)

## Abstract
Multimodal Large Language Models (MLLMs) have significantly advanced
AI-assisted medical diagnosis, but they often generate factually inconsistent
responses that deviate from established medical knowledge. Retrieval-Augmented
Generation (RAG) enhances factual accuracy by integrating external sources, but
it presents two key challenges. First, insufficient retrieval can miss critical
information, whereas excessive retrieval can introduce irrelevant or misleading
content, disrupting model output. Second, even when the model initially
provides correct answers, over-reliance on retrieved data can lead to factual
errors. To address these issues, we introduce the Multimodal Intelligent
Retrieval and Augmentation (MIRA) framework, designed to optimize factual
accuracy in MLLM. MIRA consists of two key components: (1) a calibrated
Rethinking and Rearrangement module that dynamically adjusts the number of
retrieved contexts to manage factual risk, and (2) A medical RAG framework
integrating image embeddings and a medical knowledge base with a query-rewrite
module for efficient multimodal reasoning. This enables the model to
effectively integrate both its inherent knowledge and external references. Our
evaluation of publicly available medical VQA and report generation benchmarks
demonstrates that MIRA substantially enhances factual accuracy and overall
performance, achieving new state-of-the-art results. Code is released at
https://github.com/mbzuai-oryx/MIRA.

## Full Text


<!-- PDF content starts -->

MIRA: A Novel Framework for Fusing Modalities in Medical RAG
Jinhong Wang∗
Department of Computer Vision,
MBZUAI
Abu Dhabi, United Arab Emirates
jinhong.wang@mbzuai.ac.aeTajamul Ashraf†∗
Department of Computer Vision,
MBZUAI
Abu Dhabi, United Arab Emirates
tajamul.ashraf@mbzuai.ac.aeZongyan Han
Department of Computer Vision,
MBZUAI
Abu Dhabi, United Arab Emirates
zongyan.han@mbzuai.ac.ae
Jorma Laaksonen
Department of Computer Science,
Aalto University
Aalto, Finland
jorma.laaksonen@aalto.fiRao Mohammad Anwer
Department of Computer Vision,
MBZUAI
Abu Dhabi, United Arab Emirates
rao.anwer@mbzuai.ac.ae
Abstract
Multimodal Large Language Models ( MLLM ) have significantly ad-
vanced AI-assisted medical diagnosis but often generate factually
inconsistent responses that deviate from established medical knowl-
edge. Retrieval-Augmented Generation ( RAG) enhances factual ac-
curacy by integrating external sources, but it presents two key
challenges. First, insufficient retrieval can miss critical information,
whereas excessive retrieval can introduce irrelevant or misleading
content, disrupting model output. Second, even when the model
initially provides correct answers, over-reliance on retrieved data
can lead to factual errors. To address these issues, we introduce
Multimodal Intelligent Retrieval and Augmentation ( MIRA ) frame-
work designed to optimize factual accuracy in MLLM .MIRA consists
of two key components: (1) a calibrated Rethinking and Rearrange-
ment module that dynamically adjusts the number of retrieved
contexts to manage factual risk, and (2) A medical RAGframework
integrating image embeddings and a medical knowledge base with
a query-rewrite module for efficient multimodal reasoning. This
enables the model to effectively integrate both its inherent knowl-
edge and external references. Our evaluation of publicly available
medical VQA and report generation benchmarks demonstrates that
MIRA substantially enhances factual accuracy and overall perfor-
mance, achieving new state-of-the-art results. Code, is released at
https://github.com/mbzuai-oryx/MIRA.
Keywords
Large Language Models, Retrieval Augmented Generation, Medical
Reasoning, Visual Question Answering
∗Both authors contributed equally to this research.
†Corresponding author.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference, Accepted at ACM Multimedia 2025
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.ACM Reference Format:
Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao
Mohammad Anwer. 2025. MIRA: A Novel Framework for Fusing Modalities
in Medical RAG. In .ACM, New York, NY, USA, 14 pages.
1 Introduction
Medical Visual Question Answering ( MedVQA ) plays a crucial role
in AI-driven healthcare by enabling accurate diagnosis, disease
detection, treatment optimization, and clinical decision-making [ 3–
5,21]. The integration of multimodal data, such as medical images
and corresponding textual reports, presents unique challenges like
domain shifts [ 23], limited labeled data [ 2,12], and modalities [ 19],
etc, which require a balance between visual reasoning and tex-
tual coherence. Existing models, including LLaVA-Med [17] and
Huatuo-Vision-7B [30], struggle with hallucinations, rigid rea-
soning, and unimodal biases, often misinterpreting subtle imaging
findings and failing in open-ended queries. Although Multimodal
Large Language Models ( MLLMs ) [1,8,22] have shown resilience to
internal knowledge that often leads to hallucinations and factual
inaccuracies that pose risks to patient safety [ 24]. Representative
analyses of these constraints are shown in Fig. 1, and these issues
compromise their reliability in medical use. And there’s calls for
more reliable and precise models to perform MedVQA . To address this
issue, Retrieval-Augmented Generation ( RAG) [31,32] has emerged
as a promising solution by enriching model responses with verified
external medical literature [ 33]. This approach enhances factual
grounding, reduces hallucinations, and improves the reliability of
AI-driven clinical insights [31].
Despite its potential, RAGintroduces two critical challenges in
medical settings: (1) Retrieval imbalance , where the system dis-
proportionately relies on specific sources, potentially overlooking
relevant yet underrepresented data, leading to biased or incomplete
information synthesis; and (2) Over-reliance , where the model
excessively trusts the retrieved information without adequate veri-
fication, increasing the risk of propagating factual inaccuracies and
medical misinformation.
Recently, various methods [ 28,31,33] have attempted to address
these issues by incorporating retrieval calibration and preferencearXiv:2507.07902v1  [cs.CV]  10 Jul 2025

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
(a)Misinformationand Hallucination
"What does this MRI image show
aboutmybrain?"
This MRI image shows a clear
indicationofalarge braintumor inthe
frontal lobe, which is likely causing
symptoms like headaches, memory
loss, and seizures . I recommend
seekingimmediate surgery to remove
it. The tumor appears to be malignant
based on the dark areas visible in the
image.
Notbrainscan
Noevidence
oftumor
Fabricate serious
recommendation(b)Lackofclinical reasoning
"What does this CT image show about
mycondition?"
This CT image shows your bones and
some soft tissues in the pelvic area .It
looks normal, but I’m not sure what it
means for your condition .Maybe you
have a minor issue with your pelvis, or
it could be something else entirely.
Youshouldprobablytalktoadoctor.
Insufficient
Analysis
Vague, Non-
specific, Lack of
understanding
(c)Difficultyinopen-ended scenario
"What does this graph tell me about
tumor response over time, and how
mightitaffectmytreatmentdecisions?"
This graph shows some lines and dots about
survival and tumor response over time. The
lines go down, so I guess fewer people survive
as time goes on . The tumor response part has
different patterns, like some lines are dashed
andsomearesolid, butI’mnotreallysurewhat
that means. Maybe it shows if the tumor is
getting better or worse ,but I’d need to think
about it more. You should ask a doctor for
advice ontreatment.
Lack of structured
analysis
Inability to handle
complexity
Over-simplification
and Uncertainty
(d)Restrictedaccountability andexplainability
"What do these brain images show about the
differences between Control, MCI, and AD,
and howreliable isthis information?"
These brain images show some differences in
brain activity between Control, MCI, and AD
groups.There are green and red spots
indicating activation and deactivation, but I
can’t really explain why they’re different or
what they mean in detail. The information
comes from some scientific study, I think , but
I’m notsure howreliable it isor where it’s from.
You might want to check with an expert for a
better answer.
Lack of detailed
explaination
Ambiguity about
source andreliability
Failure to address
full question
Figure 1: Overview of key constraints in automatic medi-
cal question answering, highlighting the challenges related
to factuality, domain specificity, and clinical relevance that
must be addressed for reliable system performance.
tuning to mitigate over-reliance. While these strategies improve fac-
tual accuracy to some extent, they have notable limitations. Specifi-
cally, prior approaches are predominantly designed for radiology-
related tasks and often struggle to generalize across diverse medical
domains such as pathology, dermatology, and cross-specialty di-
agnostic reasoning as shown in Fig. 1. Prior approaches [ 31,33]
decouple retrieval from reasoning, leading to fragmented outputs
without dynamic modality fusion or real-time evidence integration.
Furthermore, these frameworks typically treat vision and text as uni-
fied modalities, neglecting the need for modality-specific processing
and adaptive fusion. More critically, they rely on static databases
for retrieval, lacking mechanisms for real-time evidence integra-
tion, which is essential for ensuring up-to-date medical knowledge.
These limitations hinder their effectiveness in complex clinical
scenarios requiring dynamic, multimodal reasoning.
To address the challenges in multimodal medical reasoning,
we introduce Multimodal Intelligent Retrieval and Augmentation
(MIRA ), a novel RAGframework that enhances clinical decision-
making through structured retrieval, validation, and reasoning. Our
approach integrates two key innovations: (1) an RTRA pipeline
that employs iterative “rethink-rearrange” cycles to refine outputs
by validating them against retrieved multimodal medical knowl-
edge, ensuring factual accuracy and interpretability, and (2) a Mul-
timodal RAG (MRAG) strategy that combines offline (prebuilt
medical databases) and online (real-time web sources) retrieval to
ground responses in dynamically updated clinical evidence. The
MRAG pipeline jointly retrieves textual and visual information
from sources like NIH ChestX-ray14 and real-time medical guide-
lines, while the RTRA cycle ensures clinical consistency through a
three-stage validation process: (i) initial response generation, (ii)
self-critique via Chain-of-Thought (CoT), and (iii) final answer re-
finement. Our vision-language backbone, built on CLIP-ViT-Large
andSigLIP , is trained on 500K medical image-text pairs, enabling
robust multimodal alignment. Our contributions include a novel
Vision-Language Model ( VLM) extending LLaVA with dynamic modal-
ity fusion, the first multimodal RAG framework forMedVQA , andstate-of-the-art performance across multiple benchmarks, achiev-
ing9 times faster inference than 72B models, making it efficient for
real-world clinical applications.
Our main contributions are as follows:
•We introduce MIRA , the first retrieval-augmented genera-
tion (RAG) framework that seamlessly integrates structured
multimodal retrieval with adaptive reasoning , surpassing
existing methods in clinical decision-making accuracy and
efficiency.
•Unlike static retrieval paradigms, our context-rethink module
employs an iterative “rethink-rearrange” cycle for dynamic
𝑘-selection , ensuring precision in evidence selection, while
Chain-of-Thought (CoT) reasoning enhances factual consis-
tency in MedVQA.
•Our architecture pioneers a dual-pathway retrieval mech-
anism with specialized vision and language encoders, en-
abling fine-grained image-text alignment . The integration of
acurated citation module ensures interpretability, setting a
new standard for transparency in medical AI.
•Our model establishes new state-of-the-art (SOTA) results on
MedVQA, significantly reducing factual errors through on-
line search augmentation andadaptive CoT-based verification .
With 9×faster inference than 72B models, it sets a new
benchmark for real-time, high-precision medical reasoning.
2 Related Work
Multimodal Large Language Models (MLLMs): Traditional
LLMs are limited to textual input, restricting their ability to process
and understand multimodal information. Multimodal LLMs extend
their capabilities by incorporating additional modalities such as
images, videos, and audio, enabling a richer and more holistic under-
standing of input data [ 1,25,29,35]. A typical MLLM architecture
consists of three primary components: a modality encoder, a pre-
trained LLM, and a modality generator, interconnected through
learned projectors to facilitate seamless information transforma-
tion across different modalities [ 8,22]. These architectures have
been successfully applied in various domains, including health-
care, where multimodal integration has enhanced clinical decision
support, diagnosis, and medical research applications.
Medical Large Vision-Language Models (Med-LVLMs): Medi-
cal Large Vision-Language Models (Med-LVLMs) are specialized
architectures designed to process and interpret medical images
alongside textual data. These models typically integrate an LLM
with a dedicated vision module that extracts relevant information
from medical images and converts it into a representation compati-
ble with the LLM’s processing capabilities. Given a medical image
𝑥𝑣and a clinical query 𝑥𝑡, the combined input is represented as
𝑥=(𝑥𝑣,𝑥𝑡). The model then autoregressively predicts the prob-
ability distribution of the text output 𝑦based on the multimodal
input. Med-LVLMs have been applied in tasks such as radiology
report generation, pathology image analysis, and multi-modal med-
ical question answering [ 22,25,36]. Unlike these, MIRA integrates
iterative reasoning (RTRA) and multimodal retrieval-augmented
generation (MRAG) to ensure that outputs are validated against
retrieved evidence, addressing the issue of hallucination and im-
proving accountability. By introducing a dynamic multimodal RAG

MIRA: A Novel Framework for Fusing Modalities in Medical RAG Conference, paper, Accepted at ACM Multimedia 2025
pipeline that adapts to real-time evidence, MIRA enhances both the
versatility and precision of medical vision-language models.
Retrieval-Augmented Generation (RAG): Retrieval-Augmented
Generation (RAG) has emerged as an effective paradigm to ad-
dress LLMs’ limitations in domain-specific question answering [ 28,
31,33,34]. By combining an LLM with an external knowledge
retriever, RAG allows models to access relevant information dy-
namically, improving accuracy and factual consistency in generated
responses. This approach has been successfully applied to tasks such
as question answering, fact verification, and knowledge-intensive
text generation, achieving state-of-the-art performance in open-
domain scenarios [ 3,4,21]. Recent advances in multimodal RAG
have further expanded its capabilities by incorporating images,
videos, and structured data into retrieval mechanisms [ 7,15,18].
Multimodal RAG frameworks have been explored in medical ap-
plications, where the integration of medical text with radiological
and histopathological images has improved diagnostic accuracy
and clinical decision support [ 9,19,33]. While existing works like
MMed-RAG [ 31] and RULE [ 33] address factuality challenges, they
rely on static retrieval mechanisms. MIRA extends this paradigm
by incorporating online retrieval and iterative reasoning, ensuring
that outputs are dynamically refined against cross-modal evidence,
thereby enhancing adaptability and reliability in real-world clinical
scenarios.
Evaluation of Retrieval-Augmented Generation (RAG): Eval-
uating RAG models involves evaluating both the retrieval and the
generation components. Frameworks like RAGAs [ 11] propose met-
rics such as Faithfulness, Answer Relevance, and Context Relevance
to provide a comprehensive evaluation. Benchmarking datasets
such as TruthfulQA [ 13] and MMLU [ 10] have been widely used
to measure retrieval performance and factual accuracy. Although
automated evaluation methods offer scalability and alignment with
human judgments, they remain approximations. Human annota-
tions remain the gold standard, particularly in domain-specific
applications like medical question answering, where accuracy is
critical. For multimodal evaluation, GPT-4V has shown a strong
alignment with human assessments in vision language tasks [ 35],
further highlighting the potential of large multimodal models in
knowledge-intensive applications. Despite these advancements,
existing RAG-based evaluation frameworks often struggle with dy-
namic multimodal reasoning and ensuring real-time validation of
retrieved knowledge. MIRA addresses this limitation by integrating
iterative reasoning with online retrieval, allowing the model to
refine outputs against cross-modal evidence continuously. This not
only improves factual accuracy and faithfulness but also enhances
the adaptability of the system to real-world clinical settings. By
leveraging retrieval-aware generation and multimodal verification,
our approach sets a new benchmark in robust, evidence-grounded
medical AI.
3 Methodology
In this section, we introduce MIRA, a novel framework designed
to enhance multimodal reasoning in medical applications, particu-
larly MedVQA. Unlike conventional RAG systems, MIRA optimizes
the retrieval, fusion, and generation of both textual and visual
data to improve factual accuracy and interoperability as illustratedin Fig. 2. Built upon the LLaVA architecture, MIRA integrates a
high-resolution vision encoder for detailed medical image under-
standing and introduces dynamic modality fusion to adaptively
balance textual and visual contributions based on the query type.
Furthermore, it employs a multimodal RAG pipeline, retrieving
both text and images from structured and unstructured medical
databases using a joint embedding space. Finally, MIRA enhances
reasoning through a Rethink-Rearrange (RTRA) mechanism, lever-
aging Chain-of-Thought (CoT) reasoning for iterative refinement,
ensuring responses are clinically accurate and well-supported by
retrieved evidence. We will now discuss each component of MIRA
in detail.
3.1 Data preprocessing
Multimodal Input. The MIRA framework receives two primary
inputs: a textual query 𝑄and an image 𝐼. These inputs are processed
through separate pathways—textual and visual—and then combined
in a manner that allows for sophisticated multimodal reasoning and
retrieval-augmented generation (RAG). The inputs are transformed
into high-dimensional embeddings through the use of a text and
vision encoder. The embeddings serve as the basis for data retrieval
and fusion during the reasoning process.
Query Rewrite Module. Before the input query 𝑄is passed to
the text encoder, it undergoes an initial transformation via the
Query Rewrite module. This module aims to enhance the semantic
alignment and contextual clarity of the input query. Specifically,
given an input query 𝑄, the module applies a learned function R(·)
to rewrite and refine the query 𝑄, resulting in a modified query 𝑄′.
The rewriting function can be expressed as:
𝑄′=R(𝑄),R:R𝑛words→R𝑛′
words,
whereR(·) is a transformation that adjusts the query’s linguistic
structure while preserving its semantic meaning. The rewritten
query𝑄′serves as an enriched representation, ensuring it is aligned
with the task requirements.
Text Encoder. Following the query refinement in the Query Rewrite
module, the enhanced query 𝑄′is passed to the text encoder Etext.
This encoder operates as a learned function that maps the refined
query𝑄′to a high-dimensional semantic embedding 𝐸′
text∈R𝑑text,
where𝑑textis the dimensionality of the embedding space. This
process effectively captures the deeper semantic structure and con-
textual information of the query, facilitating its integration into the
multimodal reasoning process.
The text encoder can be formulated as a transformation function
Etext:R𝑛′
words→R𝑑text, which takes the rewritten query 𝑄′∈
R𝑛′
words as input and produces the embedding 𝐸′
text∈R𝑑text:
𝐸′
text=Etext(𝑄′), 𝐸′
text∈R𝑑text.
Here,𝑛′
wordsdenotes the number of words in the rewritten query
𝑄′, and𝑑textrepresents the dimensionality of the learned embed-
ding space. The encoder Etextis a pre-trained transformer model
optimized to capture both syntactic and semantic features of the
query. The resulting text embedding 𝐸′
textserves as a rich, high-
dimensional representation of the query, capturing its meaning in
a manner that is agnostic to the specific wording of the query. This
embedding is essential for understanding the query in the context

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
of a multimodal task, where it will be fused with corresponding
visual information to perform retrieval or reasoning tasks in the
MIRA framework.
Vision Encoder. Parallel to the textual processing pipeline, the
input image 𝐼∈R𝐻×𝑊×3is passed through the vision encoder
Evision , which is designed to extract rich visual features from the
raw image. The encoder Evision maps the input image to a high-
dimensional feature space 𝐸′
image∈R𝑑′
image, where𝑑′
imageis the
dimensionality of the image embedding space. This transformation
is mathematically defined as:
𝐸′
image=Evision(𝐼), 𝐸′
image∈R𝑑′
image.
The vision encoder Evision utilizes the CLIP-ViT-Large model [ 26],
which is based on the Vision Transformer (ViT) framework. The
model is optimized to capture complex spatial patterns, textures,
and object relationships within the image, which are essential for
tasks that require vision and language understanding structured as
follows:
𝐸′
image=𝑓ViT(𝐼), 𝑓 ViT:R𝐻×𝑊×3→R𝑑′
image,
where𝑓ViTdenotes the Vision Transformer-based function applied
to the input image 𝐼. The resulting image embedding 𝐸′
imageserves
as a high-dimensional representation of the image content, encod-
ing important visual information such as object features, textures,
and spatial relationships.
MIRA further enhances the LLaVA framework by improving
the precision of the vision encoder. The CLIP-ViT-Large model
processes high-resolution images 𝐼′∈R𝐻′×𝑊′×3, where𝐻′>𝐻
and𝑊′>𝑊, allowing the model to capture finer visual details. The
enhanced vision encoder now generates a refined image embedding
𝐸′
image∈R𝑑′
image, which improves the model’s ability to handle
complex visual reasoning tasks. Mathematically, the operation is
represented as:
𝐸′
image=Evision(𝐼′), 𝐸′
image∈R𝑑′
image.
Thus, the upgraded vision encoder with higher resolution images
results in a more expressive visual feature space, better suited for
challenging multimodal tasks that require detailed image under-
standing.
3.2 Multimodal Database
Data Retrieval. Once the textual and visual embeddings 𝐸′
text
and𝐸′
imageare computed, MIRA employs a data retrieval mecha-
nism to search for the most relevant multimodal entries from both
offline and online databases. The two primary sources of data re-
trieval are: 1. Offline Database: A collection of pre-indexed text
and image embeddings, denoted as Doffline ={(𝑇offline,𝑖,𝐼offline,𝑖)},
where each pair(𝑇offline,𝑖,𝐼offline,𝑖)represents a text and its asso-
ciated image in the offline database. 2. Online API Database:
A set of real-time data entries retrieved via an API, denoted as
DAPI={(𝑇API,𝑖,𝐼API,𝑖)}, where each pair(𝑇API,𝑖,𝐼API,𝑖)is a dynam-
ically fetched text-image pair.
The retrieval function Retrieve(𝐸′
text,𝐸′
image)computes the sim-
ilarity between the query embeddings 𝐸′
textand𝐸′
image, and the
embeddings stored in both the offline and online databases. This isachieved using vector matching algorithms, such as cosine similar-
ity, to measure the closeness between the query embeddings and
the stored entries in the respective databases:
𝑒retrieved,𝑗=Retrieve(𝐸′
text,𝐸′
image,Doffline,DAPI),
where𝑒retrieved,𝑗∈D offline∪D APIrepresents the 𝑗-th retrieved
entry from either database.
Next, the top 𝑘relevant entries, consisting of both text and image
pairs, are retrieved. The number 𝑘represents the number of top en-
tries to retrieve, where typically the retrieval involves: - 𝑘text=3re-
trieved text chunks, denoted as 𝑇retr,1,𝑇retr,2,...,𝑇 retr,𝑘text, -𝑘image =
2retrieved images, denoted as 𝐼retr,1,𝐼retr,2.
In addition to these, the API also retrieves a dynamically fetched
text paragraph 𝑇API_retrieved , which provides an additional source
of textual data for the fusion stage. Once the relevant entries are
retrieved, the fusion process integrates the textual and visual em-
beddings from both the query and the retrieved database entries.
The fusion of the query and retrieved embeddings is performed
using a dynamic attention mechanism, which adaptively assigns
different weights to the contributions of the text and image embed-
dings based on their relevance to the query.
Fusion of Retrieved Embeddings. After retrieving the top-k
relevant entries from the multimodal database, MIRA employs a
dynamic attention mechanism to fuse the image and text embed-
dings. The attention mechanism adapts to the nature of the query,
adjusting the relative contributions of each modality to ensure the
most pertinent features are prioritized.
The final fused representation 𝐸finalis computed as:
𝐸final=Att(𝐸image,𝐸text), 𝐸 final∈R𝑑final,
where Att() represents the attention mechanism, which computes
the weighted combination of the visual ( 𝐸image ) and textual ( 𝐸text)
embeddings. The resulting 𝐸finalis a multimodal representation,
where the contributions of text and image are adaptively balanced
based on the query context. This dynamic fusion enables MIRA to
prioritize either modality based on the specific nature of the query.
For example, if the query is more text-centric, the model gives
higher weight to the textual representation, while for image-centric
queries, the visual features are emphasized.
Alignment layer and Modality Fusion. Once the image and text
embeddings are computed, MIRA employs an adaptive attention
mechanism to align these modalities into a shared multimodal space.
Unlike other frameworks, such as LLaVA, which utilize simple lin-
ear transformations for alignment, MIRA’s attention-based fusion
mechanism allows for dynamic adjustment of the weight given to
each modality. To align the image and text representations in a
common multimodal space, MIRA computes the final multimodal
embedding𝐸finalas a weighted combination of the image and text
embeddings. This is formalized as:
𝐸final=Att(𝐸image,𝐸text), 𝐸 final∈R𝑑final,
where Att(·)is the attention function that computes the final fused
representation by attending to both the image and text features.
The fusion ensures that the multimodal representation captures
the relevant features from both modalities, which are contextually
weighted based on the query.

MIRA: A Novel Framework for Fusing Modalities in Medical RAG Conference, paper, Accepted at ACM Multimedia 2025
User Query & History
The image shows a circular area.
<image> What does these  Spots mean?
What if it’s something serious?Input Image 
(Or Image Sequence)
MLP
Image
Vectorized
Database
Text
Vectorized
Database
Online
Search API
Text/ImageVector Matching Algorithm Vector Matching Algorithm 
Text Embeddings
Query Rewrite
ModuleVision
Encoder
Encoded
for QueryNew
QueryMultimodal RAG
Lung Cancer’s diagnosis Lung 
cancer appears as solitary 
pulmonary nodules ……
Action 01
Rearrange Multimodal 
RAG Selection (Filtering)LLM BaseRtRa Framework
……
Encoding Retrieved
Vision ElementsTokenization
Vision  Embeddings
Action 04
Final answer generation 
from CoT.
Action02
Generateinitialanswerfrom
the given RAG information,
queryandconversations
Action 03
Rethinkthe answer, 
generating short CoT.
RtRa Module
Text modality
Image modality
Retrieval
Data Augmentation
top k
Figure 2: Overview of the MIRA (Multimodal Intelligent Retrieval and Augmentation) pipeline. The system integrates image
and text-based retrieval to enhance the generation process.
In image-centric queries (e.g., "Describe this chest X-ray"), the
attention mechanism increases the weight on the image representa-
tion𝐸image , while in text-centric queries (e.g., "What is the symptom
of this disease?"), the model prioritizes the text representation 𝐸text.
Modality-Specific Query Processing. Given a query 𝑄and an
associated image 𝐼, the model determines the most relevant modal-
ity to focus on based on the nature of the query. This is achieved
through an attention function that dynamically selects the con-
tribution from each modality. The fusion of the image and text
embeddings is adjusted using a learnable parameter 𝛼, which de-
fines the balance between the two modalities.
The final multimodal representation is computed as:
𝐸final=𝛼·Att(𝐸image)+(1−𝛼)·Att(𝐸text),
where𝛼∈ [0,1]is a learnable parameter that determines the
weighting of the image and text features. When 𝛼is close to 1,
the model focuses more on the image, while if 𝛼is close to 0, the
model prioritizes the text features. Finally, the processed input now
consists of the following multimodal entries: This combined set
Type Description
𝐼original Original Image
𝑇original Original Text
{𝑇retrieved,1,𝑇retrieved,2,𝑇retrieved,3} Retrieved Text Chunks
{𝐼retrieved,1,𝐼retrieved,2} Retrieved Images
𝑇API_retrieved API Retrieved Text Paragraph
Table 1: Structured Input Components for Multimodal RAGforms the new user input for the subsequent reasoning steps in the
MIRA framework, as shown in Table 1.
3.3 Multimodal Medical RAG
RAG Pipeline. The Medical MRAG Module is designed to pro-
cess diverse medical data, integrating multimodal information to
enhance diagnostic accuracy and provide contextually relevant
insights. The core of the retrieval mechanism is powered by a high-
performance vector matching algorithm utilizing the Faiss-CUDA
framework. The knowledge base Kconsists of vectorized represen-
tations of both text and images, enabling efficient similarity search.
Given a query 𝑞, the system retrieves a relevant set of documents
DfromK, represented as:
D={𝑑1,𝑑2,...,𝑑𝑛}, 𝑑𝑖∈K. (1)
Retrieval-Augmented Generation (RAG) plays a central role in
refining the generative process. Once the relevant documents are
retrieved, they are conditioned into the generation model to en-
sure domain-specific accuracy. The probability distribution of the
generated medical response 𝑦is computed as:
𝑝(𝑦|𝑞,D)=𝑇Ö
𝑡=1𝑝(𝑦𝑡|𝑦<𝑡,𝑞,D;𝜃), (2)
where𝜃represents the model parameters. The retrieved mul-
timodal RAG information is formatted using the Llava joint em-
bedding framework, after which the sequence undergoes further
processing through a Multilayer Perceptron (MLP) to refine its
structure and contextual coherence.

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
To maintain real-time applicability and continually expand the
RAG database, the pipeline incorporates an Online Search API. This
API leverages the DuckDuckGo open API to retrieve additional text
and image data, ensuring that the system remains updated with
the latest medical research and clinical guidelines. By integrating
both static vectorized databases and dynamically retrieved online
resources, the Medical MRAG Module achieves an optimal balance
between efficiency and accuracy, making it a robust tool for ana-
lyzing complex medical imaging data and supporting diagnostic
decision-making.
Reinforced CoT Generation. To enhance Chain-of-Thought
(CoT) reasoning in medical question-answering, we introduce a
reinforcement-driven multimodal retrieval-augmented generation
(RAG) pipeline. The process begins with structured data selection
across both offline and online RAG sources, ensuring optimal align-
ment between textual and visual modalities. Given input queries
𝑞—comprising text 𝑞𝑡and image𝑞𝑖—RAG dataDis refined through
a filtering function:
D={𝑑𝑖∈K| Rearrange(𝑑𝑖,𝑞)≠<None>}. (3)
Here, Rearrange truncates extraneous data while ensuring at
least one textual and visual reference remains relevant. The ex-
tractedDforms the basis for initial CoT generation.
Leveraging Qwen2.5-VL models, vision-text embeddings 𝑓𝑣(𝑞𝑖)
and𝑓𝑡(𝑞𝑡)are jointly processed, refining knowledge selection through
multimodal alignment:
ˆD=argmax𝑑∈D⟨𝑓𝑣(𝑞𝑖)+𝑓𝑡(𝑞𝑡),𝑓(𝑑)⟩. (4)
Following knowledge retrieval, the initial CoT response 𝑦0is
generated:
𝑦0=𝑝(𝑦|𝑞,ˆD). (5)
To ensure coherence and factual correctness, an iterative refine-
ment stage Rethink is applied, where an expert model reassesses
𝑦0against the retrieved knowledge ˆD, producing an optimized
response𝑦∗:
𝑦∗=Refine(𝑦0,ˆD,K). (6)
The entire process is reinforced using a reward function 𝑅(𝑧,𝑦),
which evaluates logical consistency and medical accuracy:
𝑅(𝑧,𝑦)=𝜆1·FactualScore(𝑧,K)+𝜆2·CoherenceScore(𝑦,𝑧),(7)
where𝜆1,𝜆2control the balance between factual precision and
coherence. Model parameters 𝜃are updated via policy gradient
optimization:
∇𝜃𝐽(𝜃)=E𝑧∼𝑝𝜃[𝑅(𝑧,𝑦)∇𝜃log𝑝𝜃(𝑧|𝑞)]. (8)
This reinforcement learning (RL)-guided CoT refinement ensures
structured, high-quality medical reasoning. The training dataset
is derived from LLaVA-Med’s 500k alignment set, with an addi-
tional 50k high-quality CoT fine-tuning instances curated from
PubMedVision and VQA-Med.3.4 Optimization
4 Optimization with Reinforcement Learning
and Chain-of-Thought Reasoning
We optimize our multimodal retrieval-augmented generation (RAG)
pipeline through a two-step training strategy combining supervised
learning and reinforcement learning (RL), enhanced with Chain-of-
Thought (CoT) reasoning for improved medical question answering.
Two-Step Training Strategy
•Step 1: Supervised Pretraining. The model is initially
trained using annotated medical data in multiple field with
cross-entropy loss. During this stage, only the multilayer
perceptron projector connecting Vision Tower and LLM is
tuned.
•Step 2: Reinforcement Fine-Tuning (RFT). The model is
then refined using pre-built RTRA-format data to stimulate
performance, as we are building roll-out data like RL. This is
similar to building thinking-based models. During this stage,
MLP and LLM parameters are unfrozen for training.
All training stages are using cross-entropy loss for model
performance optimization.
LCE=−𝑇∑︁
𝑡=1log𝑝𝜃(𝑦𝑡|𝑦<𝑡,𝑞,D), (9)
where𝑦𝑡is the target token, 𝑞is the query, andDis the
retrieved multimodal context.
Multimodal Integration and Vision Tower Tuning
We follow the LLaVA-style autoregressive modeling to predict to-
kens given visual and textual inputs:
L=−1
𝐿𝐿∑︁
𝑡=1log𝑃(𝑥𝑡|𝑥<𝑡,𝐸𝑛;𝜃), (10)
where𝐸𝑛are visual embeddings from both original and retrieved
images.
To enhance task adaptation, we unfreeze the vision tower during
instruction tuning. We test two variants:
•CLIP-based Encoders:
𝐿=1
2𝑁 𝑁∑︁
𝑖=1−log𝑒𝑠𝑖,𝑖/𝜏
Í𝑁
𝑗=1𝑒𝑠𝑖,𝑗/𝜏+𝑁∑︁
𝑖=1−log𝑒𝑠𝑖,𝑖/𝜏
Í𝑁
𝑗=1𝑒𝑠𝑗,𝑖/𝜏!
,(11)
where𝑠𝑖,𝑗is the similarity between image and text embed-
dings, and𝜏is a temperature parameter.
•SigLip Encoders:
𝐿=−1
|B|B∑︁
𝑖=1B∑︁
𝑗=1log1
1+exp(𝑧𝑖𝑗(−𝑡xi·yj+𝑏))
, (12)
where𝑧𝑖𝑗indicates whether(xi,yj)is a positive match, 𝑏is
a learnable bias, and 𝑡is a scaling factor.
Unfreezing the vision tower improves multimodal alignment,
enhancing the model’s reasoning capabilities. In the next section,
we evaluate performance against baselines and analyze the impact
of each component.

MIRA: A Novel Framework for Fusing Modalities in Medical RAG Conference, paper, Accepted at ACM Multimedia 2025
5 Experiments
5.1 Datasets
We evaluated our framework on these publicly available benchmark
datasets.
MIMIC-CXR. MIMIC-CXR [ 14] is a large-scale public dataset
containing over 370,000 chest X-rays and corresponding radiology
reports from 65,000+ patients. Released by MIT, it is widely used for
tasks like disease classification, abnormality detection, and report
generation, making it a key resource for multi-modal and vision-
language research in medical imaging.
PubMedVision. PubMedVision [ 6] is a multi-modal benchmark
built from the PMC-OA dataset, linking medical figures to their
captions, abstracts, and full texts. It supports vision-language re-
search, including pretraining and medical VQA, with diverse and
well-aligned data across biomedical fields.
5.2 Implementation Details
We implement our frameworks using PyTorch and train them on
NVIDIA A100 GPUs. For all experiments, we use the Adam opti-
mizer with a learning rate of 1e-4 and a linear learning rate warmup
over the first 10% of training steps, followed by cosine decay. Input
images are resized to 224 ×224 and normalized using ImageNet sta-
tistics, while textual inputs are tokenized using a domain-specific to-
kenizer aligned with the biomedical vocabulary. We employ mixed-
precision training to accelerate computation and reduce memory
consumption. For multi-modal pretraining, we adopt a batch size of
256 and train for 100 epochs, using early stopping based on valida-
tion loss. During fine-tuning, hyperparameters are optimized using
grid search across different downstream tasks. All experiments are
conducted with three random seeds to ensure reproducibility, and
results are reported as the mean and standard deviation.
5.3 Evaulation Metrics
To rigorously evaluate the capabilities of MIRA across diverse
biomedical tasks, we rely on task-specific metrics that capture both
textual and visual understanding. For the report generation task on
MIMIC-CXR, we employ a comprehensive set of natural language
generation metrics, including BLEU, METEOR, ROUGE, CIDEr, and
SPICE, each offering a unique perspective on fluency, relevance, and
factual consistency between generated and reference reports. These
metrics are standard in medical image captioning and allow us to
benchmark performance against existing models with precision. In
the context of visual question answering (VQA) on the PMC-VQA
dataset, we use exact match accuracy as the principal metric. This
measures whether the predicted answer matches the ground truth
exactly, reflecting the model’s ability to comprehend both visual
content and medical semantics. Together, these evaluation metrics
provide a robust framework to analyze the strengths and limitations
of the proposed approach in real-world medical AI applications.
5.4 Results on MIMIC-CXR
We focus on evaluating the performance of the proposed MIRA
framework on the MIMIC-CXR dataset, ensuring that the results
are not biased by prior exposure. To achieve this, we avoid using
general-purpose frameworks that might have already incorporatedthe MIMIC-CXR dataset during their pre-training. This preclusion
is important as it ensures that the performance of the frameworks
is evaluated on their ability to generalize to unseen data rather
than benefiting from prior knowledge of the dataset. Instead, we
select frameworks that are explicitly trained without the inclusion
of MIMIC-CXR, thus offering a more genuine measure of their
capability to handle medical image data and generate coherent re-
ports based on their architectural design and pre-training strategies.
Table 2 compares MIRA with leading multimodal report genera-
tion frameworks on a 1,000-sample subset of MIMIC-CXR using
standard metrics: BLEU (1–4), METEOR, ROUGE-L, CIDEr, and
SPICE. These metrics evaluate lexical precision and semantic co-
herence—key factors for high-quality medical reports.
MIRA outperforms all baselines, achieving the highest BLEU-1
(0.571), BLEU-4 (0.175), METEOR (0.292), ROUGE-L (0.3232), CIDEr
(0.299), and SPICE (0.318) scores. Its strong performance, despite
no exposure to MIMIC-CXR during pre-training, reflects robust
generalization rather than overfitting. Compared to models like
Huatuo-Vision-7B and LLaVA-Med-7B, MIRA shows notable gains,
especially in CIDEr and SPICE—metrics crucial for generating in-
formative and clinically relevant reports. Its performance is driven
by its architecture, which excels at integrating visual and textual
cues. Frameworks like MMed-RAG were excluded due to unavail-
able code and pre-trained weights, limiting reproducibility. Unlike
retrieval-heavy methods, MIRA offers a focused, transparent multi-
modal reasoning approach suited for medical image understanding.
5.5 Results on PubMedVision
Table 3 summarizes MIRA’s performance on the PMC-VQA dataset,
highlighting its strong multimodal reasoning in medical VQA tasks.
MIRA achieves a correctness score of 0.66 (861/1297) on true/false
questions, closely trailing top models like Qwen2.5-VL-72B and
OpenAI o1 (both at 0.69), while outperforming LLaVA-Med (0.54)
and BiomedGPT (0.58). On detailed questions, MIRA scores 0.39
(385/971), ahead of Huatuo-Vision-7B (0.24) and XrayGPT (0.23),
indicating its capacity for fine-grained visual understanding. How-
ever, it trails Qwen2.5-VL-72B (0.47) and OpenAI o1 (0.49), likely
due to their significantly larger model sizes and computational
resources.
MIRA balances efficiency and effectiveness, performing well in
medical VQA without the massive scale of models like Qwen2.5-VL-
72B. Its strong results—despite no pretraining on PMC-VQA—indicate
genuine generalization. Fine-tuning on edge cases or integrating
medical knowledge bases could further narrow the performance
gap.
5.6 Ablation Study
Preference Analysis. To further assess the contribution of specific
components within MIRA-8B, an ablation study was conducted
based on preference analysis against OpenAI o1 and Kimi-K1.5
frameworks, as shown in Table 4. The results reveal that MIRA is
preferred over OpenAI o1 in 46% of cases (92 out of 200), suggesting
that its design offers a balanced performance for PubMedVision
QA pairs. However, when compared to Kimi-K1.5, MIRA-8B is fa-
vored more frequently, with a preference ratio of 59.5% (119 out of

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
framework BLEU1 BLEU2 BLEU3 BLEU4 METEOR ROUGE_L CIDEr SPICE
LLaVA-1.5 Zeroshot[20] 0.130 0.070 0.054 0.041 0.021 0.0821 0.053 0.046
BiomedGPT[35] 0.371 0.232 0.137 0.073 0.193 0.231 0.216 0.184
LLaVA-Med-7B[16] 0.482 0.337 0.164 0.108 0.227 0.273 0.259 0.267
Huatuo-Vision-7B[6] 0.555 0.414 0.230 0.155 0.325 0.291 0.265 0.305
MIRA (ours) 0.571 0.420 0.259 0.175 0.292 0.323 0.299 0.318
Table 2: Performance comparison of multimodal report generation abilities between MIRA and frameworks specialized on ,
measured on 1000 samples split from MIMIC-CXR .
framework Conv (1297 in total) Details (971 in total, 𝑐𝑠>0.5)
LLaVA-1.5 Zeroshot[20] 0.45(578) 0.02(20)
BiomedGPT[35] 0.58(752) 0.21(201)
LLaVA-Med-7B[16] 0.54(700) 0.19(183)
Huatuo-Vision-7B[6] 0.63(823) 0.24(231)
XrayGPT[28] 0.59(769) 0.23(222)
Qwen2.5-VL-72B 0.69(892) 0.47(457)
OpenAI o1 0.69(893) 0.49(480)
Kimi-K1.5[27] 0.59(767) 0.35(340)
MIRA (ours) 0.66(861) 0.39(385)
Table 3: Performance comparison of PMC-VQA question set
correctness analysis
200), indicating that the framework’s efficiency and multimodal rea-
soning capabilities provide a substantial advantage in this context.
This ablation highlights the critical components of MIRA-8B that
contribute to its generalization ability and its effectiveness across
diverse PubMedVision tasks.
Comparison OpenAI o1 Kimi-K1.5[27]
MIRA (ours) 0.460(92) 0.595(119)
Table 4: Preference analysis on 200 sampled PubMedVision
QA pairs shows that MIRA is favored by the judge model
(Qwen2.5-VL-72B) in most cases.
Attention Visualization.
To better understand how the framework utilizes retrieved infor-
mation during response generation, we visualize the attention distri-
bution over the input sources while decoding specific output tokens
(highlighted in yellow) in Fig 3. The attention map shows when in-
ferencing the tokens highlighted in yellow, the model’s attention dis-
tribution in the user input’s components, respectively, input iamges,
Component Conv (1297 in total) Details (971 in total, 𝑐𝑠>0.5)
No Online (Local only) 0.49(639) 0.11(108)
No Offline (Search only) 0.55(714) 0.23(222)
Text Only 0.46(601) 0.19(181)
Vision Only 0.50(654) 0.20(190)
Table 5: Ablation study showing performance under various
RAG settings. “Conv” and “Details” represent the number
of correct/favorited responses, which is the same setting as
previous experiment in PMC-VQA.query, and each slice of RAG result. The map only calculates atten-
tion in user input, while normalizing all these attention components
to a sum of 1. Our analysis shows that the framework’s attention is
initially focused on the visual input, particularly around regions
describing "chest X-ray showing changes consistent with rickets"
and the anatomical location ("clavicular level"). As the framework
proceeds to generate its response, attention shifts toward textual
cues in the retrieved documents—especially phrases such as "atrial
septal defect’s mechanisms", "if untreated", and "shunt". These el-
ements are semantically aligned with the generated conclusion
involving a broken shunt and its clinical implications. Note that the
framework predominantly focuses its attention on the response-
side context during decoding, leading to an uneven distribution
across inputs. For this visualization, we normalize the attention
weights across source tokens to highlight their relative importance
in guiding generation, based on training-time behavior. Brief spikes
on unrelated regions suggest room for improving alignment for
better coherence and factual accuracy in complex medical cases.
Component Analysis To understand the contribution of different
RAG components, we perform an ablation study under various set-
tings: disabling online retrieval (local only), disabling offline DDGS
memory, and isolating modalities (text-only and vision-only). As
shown in Table 5, removing the online component leads to a notice-
able drop in performance, particularly in detail-level understand-
ing, indicating the importance of up-to-date retrieval. Similarly, the
absence of offline DDGS memory reduces overall grounding and
specificity. Interestingly, vision-only settings outperform text-only
in both conversational and detailed responses, highlighting the
framework’s strength in visual reasoning. However, the best per-
formance is achieved when both retrieval pathways and modalities
are integrated.
6 Conclusion
We present MIRA, a novel approach to improving the factuality
and reliability of Multimodal Large Language Models (Multimodal-
LLMs) in medical Retrieval-Augmented Generation (RAG). MIRA
addresses two key challenges: mitigating factuality risks by rear-
ranging retrieved contexts for relevance and reliability, and reduc-
ing excessive dependence on external data to prevent overfitting.
By balancing intrinsic medical knowledge with external insights,
MIRA generates more accurate and contextually grounded medical
responses using our RtRa framework. Our experiments and ablation
studies demonstrate significant improvements in factuality, clarity,

MIRA: A Novel Framework for Fusing Modalities in Medical RAG Conference, paper, Accepted at ACM Multimedia 2025
"<image>",
"What is the main finding in the chest x -ray?", "Retrieved information:", "Patent foramen ovale's
Mechanisms: In normal hearts, ....", "Patent 
ductusarteriosus's Prognosis: If left untreated....", Atrial heart septal defect's Mechanisms: ...Blood 
shunts from the left atrium to the right, ....",
"[Image 2] Chest X -ray showing changes consistent with rickets. These changes are usually referred 
to as \"rosary beads \" of rickets.", ...User
Input
Okay. I will now make use of the RAG data and initial 
conversation to generate a response. First I will find 
what's useful in RAG data and write them: 
<rearrange>[Image 1] which content is: Chest X -ray 
showing changes consistent with rickets. These 
changes are usually referred to as \"rosary beads \" of 
rickets. \nAtrial heart septal defect's Mechanisms: 
….. If untreated , this condition may lead to 
enlargement of the right side of the heart and 
eventually heart failure.</rearrange> From these data 
and the conversation, I think I can generate an initial 
answer as: <initial>
…  I will first clarify that the initial answer correctly 
identifies the main finding in the chest X -ray, which is 
consistent with the broken shunt ..... .</rethink>
After all, let's summarize everything, the best reply to 
the user will be <final>The chest X -ray reveals a 
broken shunt at the clavicular level, with its dislodged 
tip in the pulmonary artery, suggesting mechanical 
failure or trauma that could impair blood flow or lung 
function. …  </final>
Model Attention 
Figure 3: Visualization of attention distribution across all
slices of input sequences, emphasizing the model’s focus on
critical tokens. This highlights how the model has learned
selectively attends to important parts of the input, guiding
the generation process for more accurate and contextually
relevant responses.
and robustness, paving the way for more reliable AI systems in
healthcare settings. Future work may focus on model fine-tuning
and integration with specialized medical knowledge bases to in-
crease performance in even more complex medical scenarios.
References
[1]Rawan AlSaad, Alaa Abd-alrazaq, Sabri Boughorbel, Arfan Ahmed, Max-Antoine
Renault, Rafat Damseh, and Javaid Sheikh. 2024. Multimodal Large Language
Models in Health Care: Applications, Challenges, and Future Outlook. J Med
Internet Res 26 (25 Sep 2024), e59505. https://doi.org/10.2196/59505
[2]Tajamul Ashraf and Janibul Bashir. 2025. FATE: Focal-modulated Attention
Encoder for Multivariate Time-series Forecasting. arXiv:2408.11336 [cs.LG]
https://arxiv.org/abs/2408.11336
[3]Asma Ben Abacha, Sadid A Hasan, Vivek V Datla, Dina Demner-Fushman, and
Henning Müller. 2019. Vqa-med: Overview of the medical visual question an-
swering task at imageclef 2019. In Proceedings of CLEF (Conference and Labs of
the Evaluation Forum) 2019 Working Notes . 9-12 September 2019.
[4]Asma Ben Abacha, Mourad Sarrouti, Dina Demner-Fushman, Sadid A. Hasan,
and Henning Müller. 2021. Overview of the VQA-Med Task at ImageCLEF 2021:
Visual Question Answering and Generation in the Medical Domain. In CLEF
2021 Working Notes (CEUR Workshop Proceedings) . CEUR-WS.org, Bucharest,
Romania.
[5]Asma Ben Abacha, Mourad Sarrouti, Dina Demner-Fushman, Sadid A Hasan, and
Henning Müller. 2021. Overview of the vqa-med task at imageclef 2021: Visual
question answering and generation in the medical domain. In Proceedings of
the CLEF 2021 Conference and Labs of the Evaluation Forum-working notes . 21-24
September 2021.
[6]Junying Chen, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy
Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, Guangjun Yu, Xiang
Wan, and Benyou Wang. 2024. HuatuoGPT-Vision, Towards Injecting Medical
Visual Knowledge into Multimodal LLMs at Scale. arXiv:2406.19280 [cs.CV]
https://arxiv.org/abs/2406.19280
[7]Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen. 2022.
MuRAG: Multimodal Retrieval-Augmented Generator for Open Question An-
swering over Images and Text. arXiv:2210.02928 [cs.CL] https://arxiv.org/abs/
2210.02928[8]Hilmi Demirhan and Wlodek Zadrozny. 2023. Survey of Multimodal Medical
Question Answering. BioMedInformatics 4, 1 (2023), 50–74.
[9]Jiawen Deng, Areeba Zubair, and Ye-Jean Park. 2023. Limitations of large language
models in medical applications. Postgraduate Medical Journal 99, 1178 (2023),
1298–1299.
[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert:
Pre-training of deep bidirectional transformers for language understanding. In
Proceedings of the 2019 conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1 (long and
short papers) . 4171–4186.
[11] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert.
2023. RAGAS: Automated Evaluation of Retrieval Augmented Generation.
arXiv:2309.15217 [cs.CL] https://arxiv.org/abs/2309.15217
[12] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong
Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. 2021. Domain-specific
language model pretraining for biomedical natural language processing. ACM
Transactions on Computing for Healthcare (HEALTH) 3, 1 (2021), 1–23.
[13] Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht,
Manuel Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios
Kaissis, et al .2024. Evaluation and mitigation of the limitations of large language
models in clinical decision-making. Nature medicine 30, 9 (2024), 2613–2622.
[14] Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum,
Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng. 2019.
MIMIC-CXR, a de-identified publicly available database of chest radiographs
with free-text reports. Scientific data 6, 1 (2019), 317.
[15] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. In Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) . 6769–6781.
[16] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei
Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. 2023. Llava-med:
Training a large language-and-vision assistant for biomedicine in one day. arXiv
preprint arXiv:2306.00890 (2023).
[17] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei
Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. 2024. Llava-med: Train-
ing a large language-and-vision assistant for biomedicine in one day. Advances
in Neural Information Processing Systems 36 (2024).
[18] Weixiong Lin, Ziheng Zhao, Xiaoman Zhang, Chaoyi Wu, Ya Zhang, Yanfeng
Wang, and Weidi Xie. 2023. PMC-CLIP: Contrastive Language-Image Pre-training
using Biomedical Documents. Medical Image Computing and Computer Assisted
Intervention .
[19] Zhihong Lin, Donghao Zhang, Qingyi Tac, Danli Shi, Gholamreza Haffari, Qi Wu,
Mingguang He, and Zongyuan Ge. 2022. Medical Visual Question Answering: A
Survey. arXiv preprint arXiv:2111.10056 (2022).
[20] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2023. Improved Baselines
with Visual Instruction Tuning.
[21] Yunyi Liu, Zhanyu Wang, Dong Xu, and Luping Zhou. 2023. Q2ATransformer:
Improving Medical VQA via an Answer Querying Decoder. arXiv preprint
arXiv:2304.01611 (2023).
[22] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure
Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. 2023. Med-
flamingo: a multimodal medical few-shot learner. In Machine Learning for Health
(ML4H) . PMLR, 353–367.
[23] Zabir Al Nazi and Wei Peng. 2024. Large language models in healthcare and
medical domain: A review. In Informatics , Vol. 11. MDPI, 57.
[24] Jiwoo Park, Kangrok Oh, Kyunghwa Han, and Young Han Lee. 2024. Patient-
centered radiology reports with generative artificial intelligence: adding value to
radiology reporting. Scientific Reports 14, 1 (2024), 13218.
[25] Cheng Peng, Xi Yang, Aokun Chen, Kaleb E Smith, Nima PourNejatian, Anthony B
Costa, Cheryl Martin, Mona G Flores, Ying Zhang, Tanja Magoc, et al .2023. A
Study of Generative Large Language Model for Medical Research and Healthcare.
arXiv preprint arXiv:2305.13523 (2023).
[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models
From Natural Language Supervision. arXiv:2103.00020 [cs.CV] https://arxiv.org/
abs/2103.00020
[27] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen,
Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al .2025. Kimi k1. 5:
Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599 (2025).
[28] Omkar Thawkar, Abdelrahman Shaker, Sahal Shaji Mullappilly, Hisham
Cholakkal, Rao Muhammad Anwer, Salman Khan, Jorma Laaksonen, and Fa-
had Shahbaz Khan. 2023. Xraygpt: Chest radiographs summarization using
medical vision-language models. arXiv preprint arXiv:2306.07971 (2023).
[29] Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura
Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. 2023. Large language models
in medicine. Nature Medicine 29 (2023), 1930–1940. https://api.semanticscholar.
org/CorpusID:259947046

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
[30] Haochun Wang, Chi Liu, Nuwa Xi, Zewen Qiang, Sendong Zhao, Bing Qin, and
Ting Liu. 2023. HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge.
arXiv preprint arXiv:2304.06975 (2023).
[31] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun
Zhang, James Zou, and Huaxiu Yao. 2024. Mmed-rag: Versatile multimodal rag
system for medical vision language models. arXiv preprint arXiv:2410.13085
(2024).
[32] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun
Zhang, James Zou, and Huaxiu Yao. 2024. MMed-RAG: Versatile Multimodal
RAG System for Medical Vision Language Models. arXiv:2410.13085 [cs.LG]
https://arxiv.org/abs/2410.13085
[33] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang,
and Huaxiu Yao. 2024. Rule: Reliable multimodal rag for factuality in medical
vision language models. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing . 1081–1093.
[34] Peng Xia, Kangyu Zhu, Haoran Li, Hongtu Zhu, Yun Li, Gang Li, Linjun Zhang,
and Huaxiu Yao. 2024. Rule: Reliable multimodal rag for factuality in medical
vision language models. In Proceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing . 1081–1093.
[35] Kai Zhang, Jun Yu, Zhiling Yan, Yixin Liu, Eashan Adhikarla, Sunyang Fu, Xun
Chen, Chen Chen, Yuyin Zhou, Xiang Li, Lifang He, Brian D. Davison, Quanzheng
Li, Yong Chen, Hongfang Liu, and Lichao Sun. 2023. BiomedGPT: A Unified and
Generalist Biomedical Generative Pre-trained Transformer for Vision, Language,
and Multimodal Tasks. arXiv:2305.17100 [cs.CL]
[36] Sheng Zhang, Yanbo Xu, Naoto Usuyama, Hanwen Xu, Jaspreet Bagga, Robert
Tinn, Sam Preston, Rajesh Rao, Mu Wei, Naveen Valluri, et al .2023. Biomed-
CLIP: a multimodal biomedical foundation model pretrained from fifteen million
scientific image-text pairs. arXiv preprint arXiv:2303.00915 (2023).

MIRA: A Novel Framework for Fusing Modalities in Medical RAG Conference, paper, Accepted at ACM Multimedia 2025
Appendix
Table 6: Hyperparameter Settings for LLaVA Training
Hyperparameter Value
version(llava_prompt_version) RTRA-Format-v1
mm_tunable_parts mm_mlp_adapter in Pretraining, mm_mlp_adapter, mm_language_model in RFT
deepspeed_mode zero_2_offload in both pretraining and finetuning
mm_vision_tower_lr 2e-6
vision_tower google/siglip-so400m-patch14-384
mm_projector_type mlp2x_gelu
mm_vision_select_layer -2
rag_enabled True
rag_tokenizer all-MiniLM-L6-v2
rag_topk 3
query_rewrite_enabled True
query_rewrite_model Qwen-2-5-8B
image_aspect_ratio anyres
image_grid_pinpoints [(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]
mm_patch_merge_type spatial_unpad
precision bfloat16
Pretrain_num_train_epochs 1
Pretrain_per_device_train_batch_size 4
Pretrain_per_device_eval_batch_size 1
Pretrain_gradient_accumulation_steps 16
RFT_num_train_epochs 2
RFT_per_device_train_batch_size 1
RFT_per_device_eval_batch_size 1
RFT_gradient_accumulation_steps 32
learning_rate 1e-5 in pretraining, 2e-6 in finetuning
weight_decay 0.0
warmup_ratio 0.03
lr_scheduler_type cosine
tf32 True
model_max_length 32768
torch_compile_backend inductor
dataloader_drop_last True
attn_implementation sdpa
Notes.
•The experiments were conducted using 4 ×A100 80GB GPUs in RFT stage, and 4 ×A100 40GB GPUs in Pretraining stage.
•For query rewrite, zero-shot models could be used for summarizing user inputs. Qwen-2-5-8B is recommended, and Mistral-small-22B
is also tried during development. The external model is hosted using Ollama .

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
Additional Dataset Visualizations
RTRA DataSlice Example 1
Human Query:
Is it a malignant lesion?
MIRA Response:
/angle-righ◎Rearrange: Selected:
(1)Cryptococcus neoformans and C. gattii often cause invasive lung infections in adults with HIV or other weakened immune systems. It can
look like lung cancer or tuberculosis on scans.
(2) <Image 4> CT scan showing cryptogenic organizing pneumonia (biopsy-proven).
/angle-righ◎Initial: The lesion identified in the CT scan, initially diagnosed as cryptogenic organizing pneumonia, is not definitively a malignant lesion.
Cryptogenic organizing pneumonia is a type of lung inflammation that can mimic the appearance of cancer on imaging studies, but it is not a
form of cancer. However, to rule out malignancy, additional tests such as a biopsy or further imaging may be necessary.
/angle-righ◎Rethink:
(1)I’ll confirm the initial diagnosis and the nature of the lesion, emphasizing that cryptogenic organizing pneumonia (COP) is a condition that
can mimic the appearance of a malignant lesion on imaging but is not cancerous.
(2)I’ll clarify that while COP is the initial diagnosis, the possibility of a malignant lesion cannot be entirely ruled out without further diagnostic
tests, such as a biopsy or additional imaging studies.
(3)I’ll mention that the differential diagnosis should also consider other conditions that can present similarly, such as pulmonary cryptococcosis,
especially in immunocompromised individuals, and highlight the importance of a comprehensive evaluation.
/angle-righ◎Final: The lesion was initially diagnosed as cryptogenic organizing pneumonia, a non-malignant lung condition. However, a definitive diagnosis
would require further evaluation and tests, such as a biopsy, to confirm the nature of the lesion and rule out any malignancy. Additionally,
conditions like pulmonary cryptococcosis, which can also mimic cancer on imaging, should be considered, especially in immunocompromised
individuals. It’s always best to consult with a healthcare professional for a thorough assessment and accurate diagnosis.
RAG Sequence For the Query:
/da◎abaseNonspecific interstitial pneumonia’s Diagnosis: Diagnosis is made by reviewing patient history, imaging, lung function tests, and sometimes a
surgical lung biopsy. Lung biopsies show two patterns: cellular (with inflammation but minimal fibrosis) and fibrosing (with interstitial fibrosis
and varying levels of inflammation). Both patterns are uniform and lack the prominent fibroblastic foci found in other types of idiopathic
interstitial pneumonia.
/da◎abaseCryptococcosis’s Pulmonary cryptococcosis: Cryptococcus neoformans and C. gattii often cause invasive lung infections in adults with HIV or
other weakened immune systems. While healthy adults may get it too, they usually have mild symptoms and don’t need treatment. However,
these infections can spread to the brain, especially in immunocompromised individuals. Pulmonary cryptococcosis is common worldwide but
often underdiagnosed due to limitations in testing. It can look like lung cancer or tuberculosis on scans. The CrAg test and blood cultures are
usually negative unless the infection has spread. If not treated, pulmonary cryptococcosis can make cryptococcal meningitis worse.
/da◎abaseC: Cryptococcal meningitis’s Pulmonary cryptococcosis: Cryptococcus (both C. neoformans and C. gattii) commonly causes lung infections in
adults with HIV or other weakened immune systems, but it can also affect healthy adults with milder symptoms. While these infections may not
need treatment, careful monitoring is important. In some cases, the infection can spread to the brain and nervous system, especially in those
with weakened immune systems. Pulmonary cryptococcosis has a worldwide presence but is often underdiagnosed due to diagnostic limitations.
It typically presents as lung nodules that can mimic cancer, tuberculosis (TB), or other fungal infections. However, tests like blood cultures and
the Cryptococcal antigen test are usually negative unless the infection has spread. Additionally, if not managed properly, this condition can
worsen the prognosis of cryptococcal meningitis.
/da◎abaseOnline RAG: hamburger line 1 hamburger line 2 hamburger line 3 Close Home (https://www.belmarrahealth.com/) » Brain Function
(https://www.belmarrahealth.com/brain-function/) » Encephalomalacia: Definition, causes, types, symptoms, and treatment Encephalomalacia is
a type of serious brain damage that results in the softening or loss of brain tissue. Causes of encephalomalacia are often linked to inflammation
or hemorrhages that are a consequence of being afflicted by cerebral infarction, cerebral ischemia (https://www.belmarrahealth.com/types-
cerebral-ischemia-symptoms-prevention-treatment-tips/) , infection, craniocerebral trauma, or another injury. The term, encephalomalacia, is
often used upon gross pathological inspection, as it is much easier to better describe blurred cortical margins and decreased consistency of brain
tissue, or brain softening. Advertisement [cky_video_placeholder_title] Softening may be seen in a localized area or be widespread, depending
on the underlying cause. It can affect virtually any part of the brain, possibly damaging tissues of the frontal, occipital, parietal, and temporal
lobe. This can present with several motor and sensory deficits. Considered the most common cause of encephalomalacia, strokes occur due to
the interrupted blood supply to the brain either through a lack of blood supply to brain tissue or due to hemorrhage of a brain blood vessel.
Oxygenated blood is vital for the function and maintenance of brain cells, which will become damaged or die if the blood supply is not promptly
restored. Occurs when the flow of blood in the brain has become interrupted. This may be due to abnormal brain swelling or the removal of
brain tumors resulting in brain damage. Damage to brain tissue due to conditions such as a stroke can result in the formation of scar tissue.
This area of tissue undergoes contractions and eventually forms encephalomalacia within the brain Blunt force trauma to the skull will lead to
contusions and possibly brain damage if the force is strong enough. Other forms of penetrating trauma, like from a knife or bullet to the skull,
may also lead to the development of encephalomalacia Encephalomalacia may also be the result of: Also known as cerebrocortical necrosis
(CCN), this neurological disease is caused by the disruption of thiamine production. Thiamine is a B vitamin (B1) and a key chemical in glucose
metabolism. If a person were to become deficient in thiamine, significant side effects are likely to develop. The most threatening of which is the
inhibition of neurological activity. Polioencephalomalacia will result in damage to the gray matter of the central nervous system, where the
majority of information processing.

MIRA: A Novel Framework for Fusing Modalities in Medical RAG Conference, paper, Accepted at ACM Multimedia 2025
•Image 2: CT of the chest demonstrating right-sided pneumonia (left side of the image).
•Image 3: Cavitating pneumonia due to MRSA as seen on a CT scan.
•Image 4: CT scan showing cryptogenic organizing pneumonia (biopsy-proven).
•Image 5: MRI of Brain and MRA Brain showing encephalomalacia with extensive changes.
•Image 6: Not retrieved.
•Image 7: Not retrieved.
Input Image (User-Provided)
RAG Image 2 RAG Image 3
RAG Image 4 Online RAGImg 5
RTRA DataSlice Example 2
Human Query:
What is the image showing?
MIRA Response:
/angle-righ◎Rearrange: Selected:
(1)Brachydactyly type D: Brachydactyly type D, also known as short thumb or clubbed thumb, is a genetic trait where the thumb appears
shorter and wider with a broader nail bed than usual. The distal phalanx of the affected thumb is approximately two-thirds the length of a
normal thumb. It is quite common, affecting about 2–3% of the population, and is associated with the HOXD13 gene on chromosome 2q31.1.
(2) <Image 2> Merkel cell carcinoma. Gross pathology specimen.
/angle-righ◎Initial: The image is showing a gross pathology specimen of Merkel cell carcinoma.
/angle-righ◎Rethink:
(1)I’ll confirm the accuracy of the initial answer by verifying that the image indeed shows a gross pathology specimen of Merkel cell carcinoma.
(2)I’ll enhance the initial answer by providing a brief description of what Merkel cell carcinoma is, including its clinical significance and
common presentation.
(3)I’ll ensure the answer is concise and directly addresses the question, while also offering enough context for the user to understand the
significance of the image.
/angle-righ◎Final: The image is showing a gross pathology specimen of a mass, specifically Merkel cell carcinoma, which has been resected. Merkel cell
carcinoma is a rare and aggressive type of skin cancer that typically appears as a firm, flesh-colored or bluish-red nodule on sun-exposed areas.
Early detection and treatment are crucial due to its rapid growth and potential to spread to other parts of the body.

Conference, paper, Accepted at ACM Multimedia 2025 Jinhong Wang, Tajamul Ashraf, Zongyan Han, Jorma Laaksonen, and Rao Mohammad Anwer
RAG Sequence For the Query:
/da◎abaseCarpal tunnel syndrome’s Anatomy: The carpal tunnel is a small space in the wrist, just above the base of the palm. It contains
nine flexor tendons (which bend the fingers) and the median nerve. This nerve provides feeling to the thumb, index, middle, and
half of the ring finger, as well as controls muscles that move the thumb. The carpal tunnel is bordered by bones and a strong
ligament called the transverse carpal ligament or flexor retinaculum. It is located in the middle third of the palm, between the
scaphoid bone (at the base of the thumb) and the hamate hook (along the ring finger).
/da◎abaseBrachydactyly type D: Brachydactyly type D, also known as short thumb or clubbed thumb, is a genetic trait where the thumb
appears shorter and wider with a broader nail bed than usual. The distal phalanx of the affected thumb is approximately
two-thirds the length of a normal thumb. It is quite common, affecting about 2–3
/da◎abaseC: Brachydactyly type D’s Signs and symptoms: Brachydactyly type D is a skeletal condition where the tip of the thumb appears
shorter due to partial fusion or early closure of the growth plate in the distal phalanx. This can affect one or both thumbs,
making them approximately half to two-thirds the length of normal thumbs.
/da◎abaseOnline RAG: The gross pathology specimen of Merkel cell carcinoma presents as a firm, nodular mass with a tan to pink cut
surface. It is typically well-circumscribed but may show areas of ill-defined infiltration into surrounding tissue. On sectioning,
the mass appears solid and homogeneous, sometimes with regions of hemorrhage or necrosis, especially in larger tumors. The
surface may be smooth or lobulated, and involvement of the dermis and subcutaneous tissue is common. Despite its potentially
benign appearance on gross examination, Merkel cell carcinoma is a highly aggressive neuroendocrine skin tumor often arising
in sun-exposed areas of elderly or immunocompromised individuals.
/da◎abaseImage 2: Merkel cell carcinoma. Gross pathology specimen.
/da◎abaseImage 3: The cut surface of desmoid-type fibromatosis is firm, white, and whorled. The white tumor infiltrates the adjacent
skeletal muscle (red tissue – lower left) and fat (yellow tissue – upper left). This tendency for invasion of adjacent normal tissues
and structures is the reason that desmoid-type fibromatosis has a relatively high rate of local recurrence, even after surgical
removal.
/da◎abaseImage 4: Image showing Gastrointestinal Stromal Tumor after surgical removal.
/da◎abaseImage 5: Squamous-Cell Carcinoma of the Skin | NEJM.
/da◎abaseImage 6: Radiology Workflow | Efficiencies
/da◎abaseImage 7: Patient enrollment workflow. MRI, magnetic resonance imaging; SLNB ...
Input Image (User-Provided)
RAG Image 2 RAG Image 3 RAG Image 4
Online RAGImg 5 Online RAGImg 6 Online RAGImg 7