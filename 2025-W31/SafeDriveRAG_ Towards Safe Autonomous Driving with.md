# SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation

**Authors**: Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, Huadong Ma

**Published**: 2025-07-29 08:40:17

**PDF URL**: [http://arxiv.org/pdf/2507.21585v1](http://arxiv.org/pdf/2507.21585v1)

## Abstract
In this work, we study how vision-language models (VLMs) can be utilized to
enhance the safety for the autonomous driving system, including perception,
situational understanding, and path planning. However, existing research has
largely overlooked the evaluation of these models in traffic safety-critical
driving scenarios. To bridge this gap, we create the benchmark (SafeDrive228K)
and propose a new baseline based on VLM with knowledge graph-based
retrieval-augmented generation (SafeDriveRAG) for visual question answering
(VQA). Specifically, we introduce SafeDrive228K, the first large-scale
multimodal question-answering benchmark comprising 228K examples across 18
sub-tasks. This benchmark encompasses a diverse range of traffic safety
queries, from traffic accidents and corner cases to common safety knowledge,
enabling a thorough assessment of the comprehension and reasoning abilities of
the models. Furthermore, we propose a plug-and-play multimodal knowledge
graph-based retrieval-augmented generation approach that employs a novel
multi-scale subgraph retrieval algorithm for efficient information retrieval.
By incorporating traffic safety guidelines collected from the Internet, this
framework further enhances the model's capacity to handle safety-critical
situations. Finally, we conduct comprehensive evaluations on five mainstream
VLMs to assess their reliability in safety-sensitive driving tasks.
Experimental results demonstrate that integrating RAG significantly improves
performance, achieving a +4.73% gain in Traffic Accidents tasks, +8.79% in
Corner Cases tasks and +14.57% in Traffic Safety Commonsense across five
mainstream VLMs, underscoring the potential of our proposed benchmark and
methodology for advancing research in traffic safety. Our source code and data
are available at https://github.com/Lumos0507/SafeDriveRAG.

## Full Text


<!-- PDF content starts -->

SafeDriveRAG: Towards Safe Autonomous Driving with
Knowledge Graph-based Retrieval-Augmented Generation
Hao Ye
State Key Laboratory of Networking
and Switching Technology, Beijing
University of Posts and
Telecommunications
Beijing, China
haoye@bupt.edu.cnMengshi Qi∗
State Key Laboratory of Networking
and Switching Technology, Beijing
University of Posts and
Telecommunications
Beijing, China
qms@bupt.edu.cnZhaohong Liu
State Key Laboratory of Networking
and Switching Technology, Beijing
University of Posts and
Telecommunications
Beijing, China
liuzhaoh@bupt.edu.cn
Liang Liu
State Key Laboratory of Networking
and Switching Technology, Beijing
University of Posts and
Telecommunications
Beijing, China
liangliu@bupt.edu.cnHuadong Ma
State Key Laboratory of Networking
and Switching Technology, Beijing
University of Posts and
Telecommunications
Beijing, China
mhd@bupt.edu.cn
Abstract
In this work, we study how vision-language models (VLMs) can be
utilized to enhance the safety for the autonomous driving system,
including perception, situational understanding, and path plan-
ning. However, existing research has largely overlooked the eval-
uation of these models in traffic safety-critical driving scenarios.
To bridge this gap, we create the benchmark (SafeDrive228K) and
propose a new baseline based on VLM with knowledge graph-based
retrieval-augmented generation (SafeDriveRAG) for visual ques-
tion answering (VQA). Specifically, we introduce SafeDrive228K,
the first large-scale multimodal question-answering benchmark
comprising 228K examples across 18 sub-tasks. This benchmark
encompasses a diverse range of traffic safety queries, from traffic
accidents and corner cases to common safety knowledge, enabling a
thorough assessment of the comprehension and reasoning abilities
of the models. Furthermore, we propose a plug-and-play multimodal
knowledge graph-based retrieval-augmented generation approach
that employs a novel multi-scale subgraph retrieval algorithm for
efficient information retrieval. By incorporating traffic safety guide-
lines collected from the Internet, this framework further enhances
the model’s capacity to handle safety-critical situations. Finally, we
conduct comprehensive evaluations on five mainstream VLMs to
assess their reliability in safety-sensitive driving tasks. Experimen-
tal results demonstrate that integrating RAG significantly improves
performance, achieving a +4.73% gain in Traffic Accidents tasks,
∗Corresponding author
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
MM ’25, Dublin, Ireland
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2035-2/2025/10
https://doi.org/10.1145/3746027.3755868+8.79% in Corner Cases tasks and +14.57% in Traffic Safety Com-
monsense across five mainstream VLMs, underscoring the potential
of our proposed benchmark and methodology for advancing re-
search in traffic safety. Our source code and data are available at
https://github.com/Lumos0507/SafeDriveRAG.
CCS Concepts
•Information systems →Multimedia information systems ;
Information retrieval .
Keywords
Vision Language Models; Traffic Safety; Retrieval-Augmented Gen-
eration; Visual Question Answer; Autonomous Driving
ACM Reference Format:
Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, and Huadong Ma. 2025.
SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-
based Retrieval-Augmented Generation. In Proceedings of the 33rd ACM
International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin,
Ireland. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3746027.
3755868
1 Introduction
In recent years, Vision-Language Models [21, 42, 49] (VLMs) have
attracted considerable attention in the field of autonomous driving.
Numerous studies [ 26,27,40] have explored to integrate VLMs into
autonomous driving systems to enhance perception and decision-
making capabilities in conventional traffic environments, demon-
strating impressive performance across a range of general tasks.
However, these existing efforts primarily focus on standard driv-
ing scenarios, while systematically overlooking the safety perfor-
mance of VLMs under non-standard conditions, such as traffic
accidents [ 50] and corner cases [ 3]. Given that safety is one of the
most critical metrics [ 45] for end-to-end autonomous driving, an in-
adequate focus on safety evaluation could result in system failures
at key moments, leading to severe consequences. Consequently,arXiv:2507.21585v1  [cs.AI]  29 Jul 2025

MM ’25, October 27–31, 2025, Dublin, Ireland Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, & Huadong Ma
whether VLMs can adequately understand and respond to com-
plex traffic incidents and simultaneously make effective decisions
to ensure the safety of driving decisions remains an urgent yet
challenging issue.
To address this issue, numerous efforts have been made to build
visual-language datasets [ 4,15,22,23,26,38,40,41] tailored for
complex driving scenarios. However, these existing datasets gener-
ally concentrate on a specific subdomain such as perception and
recognition [ 38], extreme weather conditions [ 29], corner-case sce-
narios [ 19,44], or particular driving skills [ 22]. Despite the recent
growth of multimodal datasets related to driving safety, a unified
benchmark capable of systematically and comprehensively evaluat-
ing models’ performance in safe driving remains lacking.
To this end, we propose a new benchmark SafeDrive228K de-
signed to comprehensively evaluate VLMs ability to understand
and respond in diverse safety-critical driving situations. This bench-
mark consists of three sets of sub-tasks: (1) Traffic Accident Tasks ,
containing 10K real-world traffic accident videos along with 102K
structured question-answer pairs for assessing a model’s capabili-
ties in recognizing, analyzing, and managing actual accident sce-
narios; (2) Corner Case Tasks , comprising 10K images of real-world
corner cases and 69K structured question-answer pairs to evaluate
the model’s robustness in handling rare or intricate traffic contexts;
and (3) Traffic Safety Commonsense Tasks , featuring 26K images
and 57K structured question-answer pairs that span common safety
issues in everyday driving, thereby testing the model’s fundamental
knowledge of driving safety.
Then, we conduct the systematic evaluation of several represen-
tative open-source VLMs, including Qwen2.5-vl [ 2], LLAVAA [ 21],
and Phi-4 [ 30], under the resource constraints typically encoun-
tered in in-vehicle systems. In particular, the parameter sizes of the
selected models do not exceed 7B, ensuring that they are feasible
for real-world deployment. Our experimental results indicate that
the mainstream models still perform suboptimally in all three sub-
tasks. Across every task, the average scores remained below 60%,
underscoring the fact that current VLMs lack the specialized driv-
ing knowledge and situational understanding necessary to reliably
handle complex traffic scenarios. Consequently, they fall short of
meeting the stringent requirements demanded by high-reliability
autonomous driving applications.
Furthermore, we introduce a new method named SafeDriveRAG,
a plug-and-play retrieval-augmented generation (RAG) approach
based on a multimodal graph structure to enhance the safety of
autonomous driving system. This method transforms a large cor-
pus of traffic safety documentation into a structured multimodal
knowledge graph, utilizing textual, visual, and semantic entities. To-
gether with our novel multi-scale subgraph retrieval algorithm, the
approach enables efficient information extraction and enhanced in-
ference, substantially improving the quality of generated responses
and inference accuracy in safety-critical driving scenarios. Experi-
mental results further confirm the effectiveness of our approach:
across the three sub-tasks, the model’s performance improved by
4.73%, 8.79%, and 14.57%, demonstrating substantial potential for
real-world driving scenarios.
Our main contributions can be summarized as follows:
(1)We propose the first large-scale autonomous driving safety
QA benchmark, named SafeDrive228K , which is a multitask andmulti-scenario dataset for systematically evaluating the reason-
ing capabilities of widely-utilized VLMs in safety-critical driving
contexts.
(2)We design a new baseline SafeDriveRAG , a multimodal graph-
based plug-and-play RAG method that significantly enhances VLMs
information utilization, reasoning and generation capabilities in
traffic safety tasks, achieving substantial improvements over the
original models across all three sub-tasks.
(3)We conduct a systematic evaluation of multiple mainstream
open-source VLMs, highlighting their performance and limitations
in handling traffic accidents, corner cases, and traffic safety com-
monsense.
2 Related Work
VLMs in Autonomous Driving. In recent years, advances in re-
search on Vision-Language Models [ 12,24,27,31,36,42,49] (VLMs)
have led to significant progress in the joint comprehension of im-
ages and text, demonstrating outstanding performance on tasks
such as image captioning, cross-modal retrieval, and visual ques-
tion answering. Leveraging these versatile capabilities, researchers
have begun integrating VLMs into autonomous driving systems
to enhance safety and interpretability. For instance, LMDrive [ 39]
combines a visual encoder with a large language model, enabling
natural language command execution within autonomous driving
systems; and DriveVLM [ 41] introduces a Chain-of-Thought [ 43]
(CoT) inference mechanism that allows the VLM to generate a
complete driving trajectory plan. However, most existing VLMs
rely primarily on generic image-text data for pre-training and lack
real-world driving experience or specialized domain knowledge,
constraining their effectiveness in safety-critical scenarios. Con-
sequently, in this work, we develop a newly multimodal question-
answering benchmark focused on traffic safety encompassing tasks
like traffic accidents, corner cases, and common safety knowledge,
to promote the practical applicability of VLMs in real-world road
environments.
Multimodal Driving Datasets. As shown in Table 1, alongside the
development of autonomous driving technologies, researchers have
constructed various datasets endowed with vision-language capabil-
ities [ 6,32], targeting diverse perception and decision-making tasks.
For instance, NuScenes-QA [ 38] and CODA-LM [ 4] primarily ad-
dress scene description, environmental perception, and driving ad-
vice generation; DriveLM [ 40] employs a graph-structured question-
answering paradigm to integrate perception, prediction, planning,
action generation, and motion control; and IDKB [ 22] draws on
extensive driving manuals and simulation data to establish a large-
scale knowledge base for the traffic domain. Despite the substantial
progress made by these multimodal datasets for perception and pre-
diction tasks, they often neglect the systematic assessment of traffic
safety considerations. Existing work [ 25,26,34,35,41,46,47] tends
to focus on granular technical evaluations, lacking a thorough ex-
amination of VLMs’ performance in acquiring safety-related knowl-
edge, responding to risks, and performing complex reasoning. To
fill this gap, in this work, we present the first large-scale multimodal
question-answering benchmark centered on traffic safety compre-
hension and reasoning skills. In contrast to traditional datasets
that focus on perception and planning, our benchmark emphasizes

SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation MM ’25, October 27–31, 2025, Dublin, Ireland
Table 1: Comparison of existing multimodal driving datasets in terms of QA types (Single or Multiple choice), question categories
(accident, corner case, safety commonsense, laws and regulation) and data size.
Dataset Venue TasksQA Types Question Categories
Size
Single Multiple Accident Corner CaseSafety
CommonsenseLaws & Reg.
BDD-X[16] ECCV18 Video QA ✗ ✗ ✗ ✗ ✓ ✗ 26K
DRAMA[26] WACV23 Video QA ✗ ✗ ✓ ✗ ✓ ✗ 102K
DriveLM[40] ECCV24 Image QA ✓ ✗ ✗ ✗ ✓ ✗ 2M
nuScenes-QA[38] AAAI24 Image QA ✓ ✗ ✗ ✗ ✗ ✗ 460K
CODA-LM[4] WACV25 Image QA ✓ ✗ ✓ ✗ ✗ ✗ 60K
IDKB[22] AAAI24 Image QA ✓ ✓ ✗ ✗ ✓ ✓ 1M
SafeDrive228K – Video QA, Image QA ✓ ✓ ✓ ✓ ✓ ✓ 228K
VLMs’ ability to understand and reason about a wide range of
hazardous and complex driving situations.
Retrieval-Augmented Generation (RAG). RAG can enhance
large model outputs by retrieving relevant information from exter-
nal knowledge sources [ 7,11]. Existing RAG methods generally fall
into two categories: text-block-based approaches [ 28,37] that seg-
ment text into minimal semantic units, parenthesis or sentences, to
facilitate rapid matching, and graph-based approaches [ 8,9,13,33]
which extract entities and utilize structured modeling to build
knowledge graphs that improve the semantic relevance of retrieval.
As large language models expand to multimodal inputs, multimodal
RAG [ 5,14] has emerged, supporting joint retrieval and interpreta-
tion. Against this backdrop, we propose a graph-based multimodal
RAG framework. Compared to previous graph-based methods, our
approach employs a multi-scale subgraph retrieval algorithm to
mitigate interference from redundant information and reduce com-
putational overhead, thus improving entity and text-block matching
efficiency. We also extend the types of retrievable information by
introducing image nodes into the knowledge graph, further boost-
ing the model’s capacity for knowledge integration and reasoning
in complex traffic scenarios.
3 SafeDrive228K Benchmark
We propose the first large-scale benchmark SafeDrive228K , which
encompasses three domains, i.e., traffic accidents, corner cases, and
traffic safety commonsense. The benchmark includes totally 228K
multimodal question-answer pairs that aims to comprehensively
evaluate models’ cognitive and reasoning capacities in diverse traf-
fic safety scenarios. Below, we outline the benchmark’s construction
process in detail.
3.1 Data Source
Traffic Accidents. We incorporate a traffic accident subset into
the SafeDrive228K benchmark to address the stringent safety re-
quirements in accident scenarios, which offer a rigorous test of the
perception and understanding in hazardous contexts. For this pur-
pose, we adopt CAP-DATA [ 10] as the primary data source, which
contains 11K real-world accident videos with detailed annotations.
Corner Cases. In real-world driving environments, corner cases
generally refer to rare but high-risk scenarios, such as sudden ani-
mal crossings or visibility issues caused by extreme weather. Com-
pared to traffic accidents, corner cases place greater emphasis onthe model’s reasoning and response to ‘unknown hazards’. To build
this subset, we adopt CODA-LM [ 4], which includes 9,768 images
capturing from real-world edge-case road scenarios.
Traffic Safety Commonsense. Finally, the traffic safety common-
sense subset focuses on fundamental knowledge and rules of every-
day driving, such as recognizing road signs and adhering to traffic
legal regulations. In contrast to the two aforementioned subsets,
this portion aims to assess models’ ‘basic safety knowledge’ and
‘daily compliance’ competencies. Specifically, we collected approxi-
mately 1,100 relevant documents, amounting to over 2,600 pages
from Chinese internet sources. We also incorporated driving manu-
als and test data from the IDKB dataset [ 22] to maximize coverage
of various traffic safety knowledge points. Since some content in
IDKB is available in multiple languages, and most VLMs currently
offer limited multilingual support, we converted the relevant data
into English and merged it with the collected documents.
3.2 Dataset Construction
As depicted in Fig. 1, we design a semi-automated data generation
process, which comprises three main stages: source data processing,
Q&A pair generation, and data validation.
Source Data Processing. For the traffic accident subset, we metic-
ulously selected 9K videos from the original collection to form the
basis of our benchmark, removing clips with poor image quality.
To ensure that critical information in each video was fully captured
during Q &A generation, we leveraged a large language model (LLM)
to produce detailed video descriptions for each clip, building on
CAP-DATA’s [ 10] existing annotations. For corner cases subset,
we selected 9K representative images from CODA-LM [ 4] to fo-
cus on rare but high-risk road scenarios. With respect to traffic
safety commonsense subset, we employed layout detection and
OCR techniques [ 18] to extract text and data blocks from the col-
lected documents, categorizing them into ‘Traffic Safety Common-
sense Documents’ and ‘Traffic Safety Commonsense Driving Test
Documents’. We then reorganized the extracted information from
the driving test documents into a standardized question-answer
format. Lastly, we integrated the driving manuals and test data
from IDKB [ 22] into these two document sets. In this framework,
the Traffic Safety Commonsense Documents serve as references
for subsequent RAG tasks, whereas the Driving Test Documents
are utilized for model evaluation.

MM ’25, October 27–31, 2025, Dublin, Ireland Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, & Huadong Ma
Data Data Processing Source DataSource Data Processing
CAP -DATA
InternetCODA
IDKBGeneration Details
Automatic Selection
Detector andOCR
QAGeneration
Few-shot prompt strategy
Visual Language Model
GenerationUser PromptCorner Case Input
Image Annotation data"perception": {}
"vehicles" : [] &Traffic Accident Input
Video Metadataweather
scenes
…
DetailsThe video 
depicts a rural …System Prompt
You are a helpful AI assistant 
specializing in traffic safety (Corner 
Cases ) question generation. You will 
receive a…
Generated Q&A Final QAPool
 Script Check
 LLM Check
 Expert Review
Adjust
Reasonable answers
Quality Control
Figure 1: Our proposed SafeDrive228K benchmark is constructed through a semi-automated workflow that consolidates and
process multiple source datasets, and then employs various tools with VLMs, with expert quality checks ensuring the reliability
and accuracy of the final Q &A. Then, the QA generation takes traffic accident and corner cases as input to synthesis prompts.
(a) Distribution of question word
across different sub-tasks.
(b) Distribution of MCQ word
across different sub-tasks.
(c) Distribution of QA answers
word across different sub-tasks.
(d) Word cloud distribution of all
questions.
Figure 2: Statistical distributions of our proposed benchmark,
i.e., distributions of word counts in questions and answers,
and a top-k word cloud visualization of questions. MCQ
means Multiple Choice Question.
Q&A Pair Generation. To ensure diversity in the generated ques-
tions, we adopted a few-shot prompt strategy. For each question
type, we designed a corresponding prompt and provided at least
one example for each distinct question category. After generat-
ing the Q &A pairs for each category, we performed a preliminary
inspection to verify the correctness of the output format. Any ques-
tions or answers that did not adhere to the required format were
regenerated.
Data Quality Control. To guarantee high-quality outputs, we
conducted a three-step validation procedure on the generated Q &A
pairs. First, we employed scripts to filter out obviously incorrect
entries, such as empty answer options or overly short responses.Next, we used GPT-4o-mini [ 31] to evaluate the remaining answers
for logical consistency. Finally, for questions flagged as containing
unreasonable answers, we enlisted 10 experts with driving experi-
ence to perform an in-depth review, providing professional answers
for contentious items. These expert-reviewed answers served as
our gold-standard data to ensure reliability and accuracy.
Following the above process, we obtained totally 228K Q &A
pairs of high quality that span multiple driving scenarios, providing
a solid foundation for subsequent evaluations of VLM performance
in traffic safety–oriented autonomous driving tasks.
3.3 Dataset Statistics
As shown in Table 1, our benchmark offers three primary innova-
tions compared with existing autonomous driving datasets.
Traffic Accidents. A unique aspect of this benchmark is our in-
clusion of real-world traffic accident videos as one of its core data
sources for question generation. For accident scenarios, we de-
signed 7 targeted question types, covering accident type detection ,
accident prevention ,emergency handling ,legal regulation recognition ,
right-of-way determination ,complex road condition decision-making ,
andgeneral questions . General question here refers to allowing the
model to independently generate questions, without imposing any
specific question bias or constraint. This approach ensures broader
coverage, thereby enhancing the assessment of the model’s over-
all understanding. This design aims to comprehensively evaluate
a model’s capabilities in managing high-risk scenarios, spanning
the full timeline from pre-accident risk prediction to post-accident
response. Based on 9,331 traffic accident videos, we automatically
generated a total of 102K Q &As.
Corner Cases. Our benchmark also encompasses numerous object-
level corner cases to evaluate a model’s safety performance when
handling rare and complex scenarios. In the corner case, we devised
5 categories of questions: object recognition ,object localization ,dan-
ger prevention ,emergency handling , and general questions . These
questions focus on a model’s perception, reasoning, and response

SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation MM ’25, October 27–31, 2025, Dublin, Ireland
Multimodal Graph Indexing
Chunk Node
 Image Entity NodeEntity Node
Entity -to-Entity RelationshipEntity -to-Chunk Relationship
Multi -Scale Subgraph RetrievalQuery
Q: Which safety procedure(s) might have been overlooked during 
the three -point turn that could lead to a collision or unsafe maneuver?
A) Failing to check for traffic before turning left across the road
B) Not looking behind while backing up toward the right curb
C) Using the left turn signal instead of the right signal to pull over
D) Turning the steering wheel left instead of right when reversingKeywords
Three-point turn
Pull over
Left curb
Integration & GenerationQuery &Keywords
GetKey Entity
 GetRelated Entity
GetTop-kEntity
left turn signal Chunk 1
GetTop-kChunkEntity Score
left turn signal Chunk 2
right curb Chunk k
···Keyword –Entity similarity score
Query –Chunk similarity score
left turn signal: A flashing light …
three -point turn : diagram showing …
···
right curb :  right -side boundary of …
𝛼1
𝛼2
𝛼𝑘···
Chunk Score𝛼1
𝛼1
𝛼2𝛽1
𝛽2
𝛽𝑘······
VLM
Entity
ChunkEntity Nodes
Entity 1:  three -point turn 
Entity 2:  right curb 
Entity 3:  check for traffic
···
…Signal with your right turn 
signal, then pull over to the right 
and stop. Signal with your left 
turn  signal, then check carefully 
for traffic from  all directions. 2. 
Turn left, go across the road so 
you come to  a stop while you 
face the left curb or edge of  the 
road. <image> 3. Look again for 
traffic. Turn your steering wheel  
as far to the right as possible, 
then look behind  you as you 
back up. Stop before you. …
Chunk Nodes
Chunk 1: A three -point turn can be  used to 
turn  …
Chunk 2: Look again for traffic. Turn your 
steering wheel  as far to the right as…
···
-----Entity ----- 
-----Chunk ----- 
-----Query ----- 
Base on   
And with 
The answer is
A), B), D) 
Figure 3: SafeDriveRAG leverages a multimodal graph indexing method in conjunction with a multi-scale subgraph retrieval
approach, effectively overcoming the challenge of slow subgraph retrieval and enhancing both operational efficiency and
overall performance.
under ’unknown hazard’ conditions. Drawing on 9,768 corner case
scenarios, we ultimately generated 69K questions, providing a rich
resource to test VLM performance in atypical driving contexts.
Traffic Safety Commonsense. To facilitate a comprehensive as-
sessment of a model’s understanding of everyday driving safety
knowledge, we collected an extensive set of traffic safety common-
sense data from both domestic and international sources. After
applying layout detection and OCR for structured data extraction
and standardized processing, we compiled and generated 57K Q &A
items. These cover a wide range of topics including driving regula-
tions, traffic sign recognition, and behavioral guidelines.
Ultimately, our benchmark in total comprises 9,331 traffic acci-
dent videos and 35K images, for a total of 228K multimodal Q &A
pairs. As illustrated in Fig. 2, we show several statistical distribu-
tions, such as question complexity, answer length, and a keyword
cloud, to provide a more intuitive overview of the dataset’s content
characteristics and coverage.
4 SafeDriveRAG Method
Most existing Vision-Language Models (VLMs) are pre-trained on
general-purpose datasets and lack specialized domain knowledge
in traffic safety [ 41]. In real-world autonomous driving scenar-
ios,particularly those that are complex or pose high risks—this
deficiency can lead to significant safety concerns, as the model
may struggle to make accurate judgments. To address this issue,
we propose a new Graph-RAG framework, namely SafeDriveRAG ,
which not only extends the types of information retrievable within
the graph structure but also introduces an efficient multi-scale
subgraph retrieval algorithm. By substantially reducing retrieval
latency, it enhances inference efficiency for autonomous driving,
where rapid response times are critical.The overall architecture of our proposed RAG framework is
depicted in Fig. 3 and comprises two core modules: Multimodal
Indexing Module that constructs a semantically aware representa-
tion of multimodal knowledge, and Multi-Scale Subgraph Retrieval
Module that enables fast and accurate information retrieval.
4.1 Problem Setting
We formally define the SafeDriveRAG task as a multimodal question-
answering problem. Given an input scene 𝐼, which may be a video 𝑉
or an image 𝑆, together with an associated question 𝑄, the objective
is to predict the most suitable answer or generate relevant text in
response. For multiple-choice questions, each question 𝑄is accom-
panied by a finite set of candidate answers 𝐴={𝑎1,𝑎2,···,𝑎𝑛}.
The goal is to select one or more options from 𝐴that best address
𝑄. The formulation can be expressed as follows:
𝑎∗=arg max
𝑎∈𝐴𝑣𝑙𝑚(𝑎|𝐼,𝑄), (1)
where𝑣𝑙𝑚refers to the VLM model, and 𝑎∗denotes the chosen
optimal subset of answers. For open-ended question-answering,
the objective is to generate a natural language answer 𝐴with the
highest semantic relevance, based on the input modalities (video or
image) and the question 𝑄. This can be written as:
𝐴=𝑣𝑙𝑚(𝑄,𝐼). (2)
On the other hand, we define the RAG framework, denoted by
𝑀. In RAG, the indexing module 𝑅is responsible for extracting
a knowledge graph 𝐺from the raw documents and constructing
an index𝜙. The retrieval module 𝜓(·)subsequently leverages the
query and indexed data to locate the relevant documents. Formally:
𝑀= 𝐺,𝑅=(𝜑,𝜓). (3)

MM ’25, October 27–31, 2025, Dublin, Ireland Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, & Huadong Ma
4.2 Multimodal Indexing Module
In real-world traffic scenarios, most safety documents contain rich
textual data alongside images, charts, and other visual elements. Ex-
isting RAG systems typically support only text-level knowledge re-
trieval, which limits their ability to effectively utilize visual content.
To enhance safe decision-making in realistic driving environments,
we designed a multimodal indexing module tailored to traffic safety
contexts.
Within this module, we build a heterogeneous graph structure
𝐺={𝑉,𝐸}containing three types of nodes and two types of edges,
providing a unified representation of both text and images, as the
following:
Entity Node 𝑉𝑒: To capture structured knowledge, represent key
semantic units in the documents.
Image Entity Node 𝑉𝑖: Correspond to important image content ref-
erenced in the documents. We insert a placeholder symbol <image>
in the text to indicate the location of the image.
Text Chunk Node 𝑉𝑐: Represent original contextual paragraphs,
preserving the complete semantic information needed for contex-
tual modeling.
Entity–Entity Edge 𝐸𝑒𝑒: Capture semantic relationships or spa-
tial/temporal associations between entities.
Entity–Chunk Edge 𝐸𝑒𝑐: Link entities to their corresponding
contextual fragments, ensuring semantic coherence and traceability.
Ultimately, the output of the indexing module is a heterogeneous
multimodal knowledge graph 𝐺, which serves as a unified and
enriched representation of traffic safety knowledge, as follows:
𝐺=({𝑉𝑒,𝑉𝑖,𝑉𝑐},{𝐸𝑒𝑒,𝐸𝑒𝑐}). (4)
4.3 Multi-Scale Subgraph Retrieval Module
During the retrieval stage, our goal is to rapidly identify entities
or chunks most pertinent to the input query, thereby supplying
support for subsequent answer generation. The overall retrieval
process proceeds as follows:
1. Key Information Extraction: A Vision Language Model (VLM)
is used to extract a set of critical keywords 𝐾𝑄={𝑘1,𝑘2,...,𝑘𝑚}
from the input query 𝑄, capturing its semantic focus.
2. Multi-Scale Subgraph Retrieval: Using the extracted keywords
𝐾𝑄to obtain a semantic subgraph 𝐺𝑄⊆𝐺.
3. Answer Generation: The retrieved entities 𝑉𝑒, images𝑉𝑖, and
chunks𝑉𝑐are then passed to the Vision Language Model for the
final answer production.
Most existing subgraph retrieval modules feature complex de-
signs, often relying on extensive graph traversal or intricate path-
matching algorithms. Although these approaches can perform well
in offline knowledge reasoning, they struggle to fulfill the demand-
ing real-time requirements of autonomous driving. To address this
gap, we propose a multi-scale subgraph retrieval module that evalu-
ates both entity-level and text-level semantics across multiple scales,
thereby maintaining high retrieval quality while significantly im-
proving efficiency and real-time inference. The module’s key design
elements for achieving efficient retrieval are as follows:
1. Keyword-Driven Entity Initialization: Based on the keywords
extracted from the query, we locate corresponding entity nodes inthe graph by similarity scores, the formula is as follows:
𝑉anchor ={𝑣𝑗|𝑠(𝑘𝑖,𝑣𝑗)≥𝛿1}, (5)
where𝛿1is a similarity threshold, 𝑘𝑖∈𝐾𝑄,𝑠(𝑘𝑖,𝑣𝑗)represents the
semantic similarity between the keyword 𝑘𝑖and entity𝑣𝑗.
2. Multi-Hop Entity Expansion: For each anchor entity 𝑣∈
𝑉anchor , we perform ℎ-hop expansion to obtain candidate entities
𝑉𝑐𝑎𝑛𝑑:
𝑉𝑐𝑎𝑛𝑑=Ø
𝑣∈𝑉𝑎𝑛𝑐ℎ𝑜𝑟𝐸𝑥𝑝𝑎𝑛𝑑(𝑣,ℎ). (6)
Starting from the anchor points, we perform multi-hop semantic ex-
pansion within the graph, bounded by a preset hop limit to discover
potentially relevant entity nodes.
3. Entity Semantic Relevance Matching: For each expanded entity
node, we calculate its semantic similarity with the query’s keywords
and select the top-k entities.
4. Chunk Semantic Matching: Drawing on the textual blocks
associated with the selected entity nodes, we compute the similarity
𝑆(𝑐)between the query 𝑞and each chunk 𝑐:
𝑆(𝑐)=𝛼·©­
«∑︁
𝑣∈𝑉𝑐𝑎𝑛𝑑𝑠(𝑞,𝑣)·𝜆𝑘𝑣ª®
¬+(1−𝛼)·𝑠(𝑞,𝑐), (7)
where𝜆is the path decay factor, 𝑘𝑣is the number of hops, and 𝛼is
the chunk semantic score weight.
5. Multimodal Information Output: Finally, the chosen entities
𝑉={𝑣1,...,𝑣𝑘}, along with their associated images and chunks
𝐶={𝑐1,...,𝑐𝑘}, are combined into a high-relevance subgraph. This
subgraph is then fed into the VLM to provide coherent and context-
rich knowledge for the answer-generation phase.
Through this design, our Graph-RAG achieves an optimal bal-
ance between efficiency and accuracy, offering a stable, rapid, and
scalable retrieval-augmented capability ideally suited for real-world
applications such as autonomous driving.
5 Experiments
In this section, we present a systematic evaluation of five main-
stream open-source VLMs on our newly constructed multimodal
traffic safety question-answering benchmark. These experiments
aim to assess each model’s understanding and reasoning capabilities
under safety-critical driving scenarios.
5.1 Experimental Settings
Model Select : We selected five mainstream open-source VLMs,
all with parameter sizes under 7B, to accommodate the stringent
computational constraints of in-vehicle deployment. Compared to
larger-scale models, lightweight VLMs exhibit higher adaptabil-
ity in real-world scenarios and more accurately reflect the actual
performance achievable in an on-board environment. Specifically,
we include the following models: Qwen2.5-vl-7B [ 42], Qwen2.5-vl-
3B [42], LLAVAA-OneVison-7B [ 17], LLAVAA-OneVison-0.5B [ 17],
Phi-4-multimodal-instruct-4.5B [ 30]. Each model was evaluated in
both its original setting and a RAG-enhanced variant. To ensure
fairness, we used a uniform prompt across all models under the
same input conditions. It should be noted that RAG-enhanced ver-
sions only incorporate external knowledge, such as entities and

SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation MM ’25, October 27–31, 2025, Dublin, Ireland
Table 2: Quantitative results of VLMs across all tasks (i.e., traffic accidents, corner cases and traffic safety commonsense). Green
indicates the methods incorporating our proposed RAG method. Bold highlights the better results.
Multi-choice Question Question & AnswerSafeDrive Score
Models Category Single Multiple Overall R-1 R-L SemScore Overall
Qwen-2.5-vl [2]3B w/o rag 54.91% 46.20% 54.23% 15.31% 14.01% 51.33% 26.88% 44.89%
3B w/ rag 65.43 % 48.93 % 62.50 % 19.25 %16.99 % 55.33 % 30.52 % 57.38 %
7B w/o rag 61.42% 58.70 % 61.90% 16.57% 14.91% 56.79% 29.42% 53.14%
7B w/ rag 67.81 % 58.17% 66.07 % 20.05 %18.32 % 58.04 % 32.14 % 60.18 %
LLAVAA-OneVision [17]0.5B w/o rag 22.90% 13.18 % 20.27% 8.14% 7.25% 44.85% 20.08% 20.57%
0.5B w/ rag 36.51 % 10.89% 29.71 % 15.88 %14.30 % 46.79 % 25.65 % 29.52 %
7B w/o rag 57.51% 38.25% 54.15% 14.08% 13.04% 51.90% 26.34% 48.32%
7B w/ rag 65.50 % 42.49 % 59.53 % 20.78 %19.34 % 56.67 % 32.26 % 57.24 %
Phi-4-multimodal-instruct [30]4.5B w/o rag 37.88% 48.23 % 41.02% 15.45% 14.18% 53.90% 27.84% 43.16%
4.5B w/ rag 51.42 % 45.19% 51.41 % 18.20 %16.78 % 56.63 % 30.53 % 51.55 %
Figure 4: Results of the SafeDrive228K benchmark w.r.t SafeDrive Score across the sub-tasks of Traffic Accidents, Corner Cases,
and Traffic Safety Commonsense.
chunks, to assist models in understanding and generating appro-
priate responses.
RAG Configuration : To enhance the knowledge support of VLMs
in traffic safety scenarios, we introduced a retrieval-augmented gen-
eration (RAG) mechanism and constructed an extensive external
document repository. This repository primarily consists of traffic
safety legislation, driving guides, and accident response manuals
collected from the Internet, totaling over 1.32 million tokens. Dur-
ing text pre-processing, and following standard practices [ 13], we
divided the documents into chunks with a maximum length of 1200
tokens, applying an overlapping sliding window of 100 tokens to
preserve semantic integrity and contextual coherence. We then
employed NanoVectorBase [ 48] to build an efficient vector-retrieval
index, facilitating low-latency, semantically aware subgraph infor-
mation retrieval. In the RAG module, the top-k for entity-node
retrieval is set to 5, while the top-k for chunk retrieval is set to 3,
balancing both entity- and context-level information for optimal
performance.5.2 Evaluation Metrics
Throughout the evaluation, we divided the questions into two main
types: multiple-choice andopen-ended . Each type was assessed us-
ing different strategies: For multiple-choice tasks, we used regular
expressions to extract the selected answer options from each VLM’s
output and compared these options to the ground truth. To maintain
rigorous standards, a model was deemed correct only if it provided
all valid options for a given question. For open-ended tasks, we em-
ployed both ROUGE [20] and SEMScore [1] to measure the similarity
between the model-generated answer and the reference answer.
ROUGE evaluates on N-gram overlap, while SEMScore assesses
sentence-level vector similarity to assess the semantic coherence of
the generated answer, placing more emphasis on meaning rather
than exact wording. We separately computed the scores for multiple-
choice and open-ended questions, weighting each by the number
of corresponding items. The results were then summed to yield an
overall performance metric, referred to as the SafeDrive Score . This
composite score serves as a key indicator of how effectively a VLM
performs in the safety-critical domain of autonomous driving.

MM ’25, October 27–31, 2025, Dublin, Ireland Hao Ye, Mengshi Qi, Zhaohong Liu, Liang Liu, & Huadong Ma
SafeDriveRAG  RespondTraffic Accident QA
Legal Compliance 
Q:Which traffic law(s) do you think were violated in the video that led to the accident?
A: Violated Article 22 of the Road Traffic Safety Law of the PRC.
B: Not a violation.
C:Violated Article 43 of the Road Traffic Safety Law of the PRC.
D:Violated Article 72 of the Implementation Regulations of the Road Traffic Safety Law of the PRC.
Output: OptionC
Corner Case
Object Recognition
Q: "Describe how the ego car should handle encountering a sudden obstacle  appearing in the lane ahead while navigating through the traffic cones 
and barriers  described in the scenario.", 
Answer : The ego car should first assess the distance and speed relative to the obstacle. If there is enough space and time to safely change lanes 
without crossing into restricted areas (marked by barriers or cones), the car should signal and move into an adjacent lane. I f not, it must brake 
smoothly to a stop, ensuring it maintains distance from the vehicle ahead (the BMW X3) to avoid a rear -end collision. The car sh ould also 
communicate its intentions via turn signals and adjust speed according to the 40 mph  limit while navigating the upcoming curve and exit, all while 
avoiding any areas demarcated by traffic cones or barriers.
Output: While  navigating through the traffic cones and approaching the exit ramp shown in the scenario, the ego car must remain highly aler t for 
any sudden obstacles appearing in the lane ahead —such as a fallen cone, debris, or a vehicle making an abrupt stop or lane chang e. If a sudden 
obstacle —such as a fallen traffic cone, stalled vehicle, or unexpected pedestrian …Entity：
<TRAFFIC CONE SIGNAGE >:The Traffic Light Signage is a triangular sign with three colored circles, …
< OBSTACLE/>: An unexpected object or hazard that appears in the car’s lane. 
 Chunk：
… A sudden obstacle refers to an unexpected object, such as a pedestrian, animal, fallen object, or vehicle, that 
abruptly appears in the driving lane. It typically requires the ego car to respond immediately by braking, 
steering, or re -planning its route to avoid a collision . …
…"The Traffic Light Signage is a triangular sign with three colored circles, which is depicted in Image 0, 
while the Edge Indicator Reflective Elements (One -Way Road) are reflective elements placed at the edge of …
<image 0> ：Ori Image ：
SafeDriveRAG  Respond
Entity：
<ROAD TRAFFIC SAFETY LAW>:sets rules for drivers, vehicles, and pedestrians to promote safe…
<ARTICLE 43 OF THE ROAD TRAFFIC SAFETY LAW>: This means the driver failed to keep a safe ..
…
Chunk：
… are easy to occur, or in case of such weather conditions as dust, hailstorm, rain, snow, fog, freezing, etc., it shall run at a lower 
speed.​ARTICLE 43 For any two motor vehicles running in the same driveway, the vehicle at the back shall keep a safe distance  
from the vehicle ahead enough for taking emergency braking measures …
…Stay attentive to the signage in place. \n <image_0>On the road, leave at least 2 seconds between you and the vehicle in front o f you, 
as the stopping distance increases with speed. \nSpeed is responsible for at least 26% of fatal accidents. \nOn the highway, 1 lin e = 
danger, 2 lines = safety. \n <image_1> <image_2> Rules and penalties …𝑡=0 𝑡=1 𝑡=2 𝑡=3
Figure 5: Visualized results of traffic accidents and corner cases, showing the retrieved entities, images, and chunks after
subgraph retrieval.
5.3 Experimental Results
Overall Model Performance. We presents the performance of
various models in three sub-tasks within the SafeDrive228K, as sum-
marized in Table 2 and Fig. 4. We can clearly observe that the overall
performance of the original models on all tasks is generally poor.
Across the three sub-tasks, most models fail to exceed 50% on the
SafeDrive Score, with only the Qwen-2.5-vl-7B model slightly sur-
passing 50. This finding underscores the limited modeling capacity
and reasoning ability of current VLMs for traffic safety scenarios. A
closer examination of each sub-task reveals that the traffic accident
task yields the weakest results, with an average score of just 40.31%.
Performance on the corner case task averages 45.02%, whereas the
traffic safety commonsense task sees an average of 40.71%. These
findings highlight the substantial room for improvement in main-
stream VLMs when dealing with traffic safety applications. Further
enhancements involving structural optimization and knowledge
augmentation appear necessary to overcome these challenges.
Effectiveness of RAG Enhancement. A comparison of pre- and
post-RAG results in Fig. 4 shows that incorporating RAG signif-
icantly boosts the accuracy and completeness of model outputs.
In particular, the average score increase in the traffic safety com-
monsense sub-task was 14.57, while corner cases improved by 8.79
points and traffic accident tasks by 4.73. Although there was a
notable performance gain in the accident sub-task, the relatively
small increase can be attributed to input truncation issues arising
from large token counts (due to video data combined with entity
and chunk information) in smaller models ( ≤7B). Nonetheless, the
example of Qwen2.5-vl-3B demonstrates that smaller models can
approach or match 7B-level performance once equipped with RAG,
indicating significant potential for lighter models under enhanced
retrieval settings.
Discussion. Overall, the Qwen series stood out, with Qwen2.5-
vl-7B achieving an average score of 60.2, exhibiting greater con-
sistency and stability across all tasks compared to other models.
LLAVA-OneVision-7B showed the largest performance jump (aver-
age +14.39) after RAG integration in both corner cases and traffic
safety commonsense tasks, suggesting a strong synergy with theretrieval mechanism. The Phi model, while smaller in size, still
delivered robust results. In general, 7B models consistently outper-
formed their 3B counterparts, affirming the importance of model
capacity in handling complex tasks.
5.4 Ablation Study of RAG
We conducted the ablation study on the traffic safety common-
sense sub-task, and we adopted Qwen2.5-vl-7B with a 10% test
set to compare three widely-used RAG methods: Naïve RAG [ 28],
MiniRAG [ 9], and our proposed SafeDriveRAG. We report their mea-
sured subgraph retrieval times (Speed Time) were 46.22 s, 9519.98 s,
and 884.10 s, respectively, while the SafeDrive Scores were 60.18%,
61.26%, and 62.07%. Although Naïve RAG is the fastest in raw re-
trieval time, its lack of structured modeling leads to weaker overall
performance; while MiniRAG achieves a modest accuracy boost but
at the cost of significantly longer retrieval times. In contrast, our
proposed SafeDriveRAG strikes a more favorable trade-off balance
between efficiency and accuracy, demonstrating strong potential
for real-world safety-critical deployments.
6 Conclusion
In this paper, we presented SafeDrive228K, the first comprehensive
multimodal benchmark for evaluating VLMs in complex safety-
critical driving scenarios. Furthermore, we designed SafeDriveRAG,
a plug-and-play RAG method underpinned by a multimodal graph
structure and an efficient multi-scale subgraph retrieval algorithm.
Experiments showed that SafeDriveRAG consistently enhances
performance across multiple safety-related driving tasks, demon-
strating strong real-world deployment potential and promising
contributions to related fields.
Acknowledgments
This work is partly supported by the Funds for the National Natural
Science Foundation of China under Grant 62202063 and U24B20176,
Beijing Natural Science Foundation (L243027), Beijing Major Sci-
ence and Technology Project under Contract No. Z231100007423014.

SafeDriveRAG: Towards Safe Autonomous Driving with Knowledge Graph-based Retrieval-Augmented Generation MM ’25, October 27–31, 2025, Dublin, Ireland
References
[1]Ansar Aynetdinov and Alan Akbik. 2024. Semscore: Automated evaluation
of instruction-tuned llms based on semantic textual similarity. arXiv preprint
arXiv:2401.17072 (2024).
[2]Shuai Bai, Keqin Chen, and Xuejing Liu et al. 2025. Qwen2.5-VL Technical Report.
arXiv preprint arXiv:2502.13923 (2025).
[3]Jasmin Breitenstein, Jan-Aike Termöhlen, and Daniel Lipinski et al. 2021. Corner
cases for visual perception in automated driving: Some guidance on detection
approaches. arXiv preprint arXiv:2102.05897 (2021).
[4]Kai Chen, Yanze Li, and Wenhua Zhang et al. 2024. Automated Evaluation of
Large Vision-Language Models on Self-driving Corner Cases. arXiv preprint
arXiv:2404.10595 (2024).
[5]Wenhu Chen, Hexiang Hu, and Xi Chen et al. 2022. MuRAG: Multimodal Retrieval-
Augmented Generator for Open Question Answering over Images and Text. In
Proc. Empir. Methods Nat. Lang. Process. Association for Computational Linguistics,
Abu Dhabi, United Arab Emirates, 5558–5570.
[6]Thierry Deruyttere, Simon Vandenhende, and Dusan Grujicic et al. 2019.
Talk2Car: Taking Control of Your Self-Driving Car. In Proc. Empir. Methods Nat.
Lang. Process. (EMNLP-IJCNLP) . Association for Computational Linguistics, Hong
Kong, China, 2088–2098.
[7]Hanxing Ding, Liang Pang, and Zihao Wei et al. 2024. Retrieve Only When It
Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large
Language Models. arXiv preprint arXiv:2402.10612 (2024).
[8]Darren Edge, Ha Trinh, and Newman Cheng et al. 2025. From Lo-
cal to Global: A Graph RAG Approach to Query-Focused Summarization.
arXiv:2404.16130 [cs.CL] https://arxiv.org/abs/2404.16130
[9]Tianyu Fan, Jingyuan Wang, and Xubin Ren et al. 2025. MiniRAG: Towards Ex-
tremely Simple Retrieval-Augmented Generation. arXiv preprint arXiv:2501.06713
(2025).
[10] Jianwu Fang, Lei-Lei Li, and Kuan Yang et al. 2022. Cognitive Accident Prediction
in Driving Scenes: A Multimodality Benchmark. CoRR abs/2212.09381 (2022).
[11] Yunfan Gao, Yun Xiong, and Xinyu Gao et al. 2024. Retrieval-Augmented Gen-
eration for Large Language Models: A Survey. arXiv preprint arXiv:2312.10997
(2024).
[12] Akshay Gopalkrishnan, Ross Greer, and Mohan Trivedi. 2024. Multi-Frame,
Lightweight & Efficient Vision-Language Models for Question Answering in
Autonomous Driving. arXiv preprint arXiv:2403.19838 (2024).
[13] Zirui Guo, Lianghao Xia, and Yanhua Yu et al. 2024. LightRAG: Simple and Fast
Retrieval-Augmented Generation. arXiv preprint arXiv:2410.05779 (2024).
[14] Ziniu Hu, Ahmet Iscen, and Chen Sun et al. 2023. REVEAL: Retrieval-Augmented
Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Mem-
ory. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. 23369–23379. doi:10.1109/
CVPR52729.2023.02238
[15] Muhammad Monjurul Karim, Zhaozheng Yin, and Ruwen Qin. 2023. An attention-
guided multistream feature fusion network for early localization of risky traffic
agents in driving videos. IEEE Trans. Intell. Veh. 9, 1 (2023), 1792–1803.
[16] Jinkyu Kim, Anna Rohrbach, and Trevor Darrell et al. 2018. Textual Explanations
for Self-driving Vehicles. In Proc. Eur. Conf. Comput. Vis. 563–578.
[17] Bo Li, Yuanhan Zhang, and Dong Guo et al. 2024. LLaVA-OneVision: Easy Visual
Task Transfer. arXiv preprint arXiv:2408.03326 (2024).
[18] Chenxia Li, Weiwei Liu, and Ruoyu Guo et al. 2022. PP-OCRv3: More At-
tempts for the Improvement of Ultra Lightweight OCR System. arXiv preprint
arXiv:2206.03001 (2022).
[19] Kaican Li, Kai Chen, and Haoyu Wang et al. 2022. Coda: A real-world road
corner case dataset for object detection in autonomous driving. In Proc. Eur. Conf.
Comput. Vis. Springer, 406–423.
[20] Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries.
InProc. Text Summarization Branches Out . 74–81.
[21] Haotian Liu, Chunyuan Li, and Qingyang Wu et al. 2023. Visual Instruction
Tuning. In Proc. Adv. Neural Inf. Process. Syst. , Vol. 36. 34892–34916.
[22] Yuhang Lu, Yichen Yao, and Jiadong Tu et al. 2024. Can LVLMs Obtain a Driver’s
License? A Benchmark Towards Reliable AGI for Autonomous Driving. arXiv
preprint arXiv:2409.02914 (2024).
[23] Changsheng Lv, Mengshi Qi, and Liang Liu et al. 2025. T2sg: Traffic topology
scene graph for topology reasoning in autonomous driving. In Proc. IEEE Conf.
Comput. Vis. Pattern Recognit. 17197–17206.
[24] Changsheng Lv, Mengshi Qi, and Xia Li et al. 2024. SGFormer: Semantic Graph
Transformer for Point Cloud-Based 3D Scene Graph Generation. In Proc. AAAI
Conf. Artif. Intell. , Vol. 38. 4035–4043.
[25] Changsheng Lv, Shuai Zhang, and Yapeng Tian et al. 2023. Disentangled coun-
terfactual learning for physical audiovisual commonsense reasoning. Proc. Adv.
Neural Inf. Process. Syst. 36 (2023), 12476–12488.
[26] Srikanth Malla, Chiho Choi, and Isht Dwivedi et al. 2023. DRAMA: Joint Risk
Localization and Captioning in Driving. In Proc. IEEE Winter Conf. Appl. Comput.
Vis.1043–1052.
[27] Jiageng Mao, Junjie Ye, and Yuxi Qian et al. 2024. A Language Agent for Au-
tonomous Driving. arXiv preprint arXiv:2311.10813 (2024).[28] Yuning Mao, Pengcheng He, and Xiaodong Liu et al. 2020. Generation-augmented
retrieval for open-domain question answering. arXiv preprint arXiv:2009.08553
(2020).
[29] Aboli Marathe, Deva Ramanan, and Rahee Walambe et al. 2023. WEDGE: A Multi-
Weather Autonomous Driving Dataset Built From Generative Vision-Language
Models. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. Workshops . 3317–3326.
[30] Microsoft, Abdelrahman Abouelenin, and Atabak Ashfaq et al. 2025. Phi-4-
Mini Technical Report: Compact yet Powerful Multimodal Language Models via
Mixture-of-LoRAs. arXiv preprint arXiv:2503.01743 (2025).
[31] OpenAI, Josh Achiam, and Steven Adler et al. 2024. GPT-4 Technical Report.
arXiv preprint arXiv:2303.08774 (2024).
[32] SungYeon Park, MinJae Lee, and JiHyuk Kang et al. 2024. VLAAD: Vision and
Language Assistant for Autonomous Driving. In Proc. IEEE Winter Conf. Appl.
Comput. Vis. Workshops . 980–987. doi:10.1109/WACVW60836.2024.00107
[33] Mengshi Qi, Weijian Li, and Zhengyuan Yang et al. 2019. Attentive Relational
Networks for Mapping Images to Scene Graphs. In Proc. IEEE Conf. Comput. Vis.
Pattern Recognit. 3952–3961. doi:10.1109/CVPR.2019.00408
[34] Mengshi Qi, Jie Qin, and Yi Yang et al. 2021. Semantics-Aware Spatial-Temporal
Binaries for Cross-Modal Video Retrieval. IEEE Trans. Image Process. 30 (2021),
2989–3004. doi:10.1109/TIP.2020.3048680
[35] Mengshi Qi, Yunhong Wang, and Annan Li et al. 2020. STC-GAN: Spatio-
Temporally Coupled Generative Adversarial Networks for Predictive Scene Pars-
ing.IEEE Trans. Image Process. 29 (2020), 5420–5430. doi:10.1109/TIP.2020.2983567
[36] Mengshi Qi, Yunhong Wang, and Jie Qin et al. 2019. KE-GAN: Knowledge
Embedded Generative Adversarial Networks for Semi-Supervised Scene Parsing.
InProc. IEEE Conf. Comput. Vis. Pattern Recognit. 5232–5241. doi:10.1109/CVPR.
2019.00538
[37] Hongjin Qian, Peitian Zhang, and Zheng Liu et al. 2024. Memorag: Moving
towards next-gen rag via memory-inspired knowledge discovery. arXiv preprint
arXiv:2409.05591 (2024).
[38] Tianwen Qian, Jingjing Chen, and Linhai Zhuo et al. 2024. nuScenes-QA: A
Multi-modal Visual Question Answering Benchmark for Autonomous Driving
Scenario. In Proc. AAAI Conf. Artif. Intell. , Vol. 38. 4542–4550.
[39] Hao Shao, Yuxuan Hu, and Letian Wang et al. 2024. LMDrive: Closed-Loop
End-to-End Driving with Large Language Models. In Proc. IEEE Conf. Comput.
Vis. Pattern Recognit. 15120–15130.
[40] Chonghao Sima, Katrin Renz, and Kashyap Chitta et al. 2025. DriveLM: Driving
with Graph Visual Question Answering. In Proc. Eur. Conf. Comput. Vis.
[41] Xiaoyu Tian, Junru Gu, and Bailin Li et al. 2024. DriveVLM: The Convergence
of Autonomous Driving and Large Vision-Language Models. arXiv preprint
arXiv:2402.12289 (2024).
[42] Peng Wang, Shuai Bai, and Sinan Tan et al. 2024. Qwen2-VL: Enhancing Vision-
Language Model’s Perception of the World at Any Resolution. arXiv preprint
arXiv:2409.12191 (2024).
[43] Jason Wei, Xuezhi Wang, and Dale Schuurmans et al. 2022. Chain-of-Thought
Prompting Elicits Reasoning in Large Language Models. Proc. Adv. Neural Inf.
Process. Syst. 35 (2022), 24824–24837.
[44] Licheng Wen, Xuemeng Yang, and Daocheng Fu et al. 2023. On the Road with
GPT-4V(ision): Early Explorations of Visual-Language Model on Autonomous
Driving. arXiv preprint arXiv:2311.05332 (2023).
[45] World Health Organization. 2023. Global Status Report on Road Safety
2023. https://www.who.int/teams/social-determinants-of-health/safety-and-
mobility/global-status-report-on-road-safety-2023. Accessed: 2025-07-21.
[46] Yiran Xu, Xiaoyin Yang, and Lihang Gong et al. 2020. Explainable Object-Induced
Action Decision for Autonomous Vehicles. In IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) .
[47] Zhenhua Xu, Yujia Zhang, and Enze Xie et al. 2024. DriveGPT4: Interpretable
End-to-End Autonomous Driving via Large Language Model. IEEE Robot. Autom.
Lett. (2024).
[48] Gus Ye. 2024. nano-vectordb. https://github.com/gusye1234/nano-vectordb.
[49] Tianyu Yu, Haoye Zhang, and Qiming Li et al. 2024. RLAIF-V: Open-Source AI
Feedback Leads to Super GPT-4V Trustworthiness. arXiv preprint arXiv:2405.17220
(2024).
[50] Ou Zheng, Mohamed Abdel-Aty, and Zijin Wang et al. 2023. Avoid: Au-
tonomous vehicle operation incident dataset across the globe. arXiv preprint
arXiv:2303.12889 (2023).