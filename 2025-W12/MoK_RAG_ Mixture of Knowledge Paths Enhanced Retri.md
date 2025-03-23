# MoK-RAG: Mixture of Knowledge Paths Enhanced Retrieval-Augmented Generation for Embodied AI Environments

**Authors**: Zhengsheng Guo, Linwei Zheng, Xinyang Chen, Xuefeng Bai, Kehai Chen, Min Zhang

**Published**: 2025-03-18 04:27:02

**PDF URL**: [http://arxiv.org/pdf/2503.13882v1](http://arxiv.org/pdf/2503.13882v1)

## Abstract
While human cognition inherently retrieves information from diverse and
specialized knowledge sources during decision-making processes, current
Retrieval-Augmented Generation (RAG) systems typically operate through
single-source knowledge retrieval, leading to a cognitive-algorithmic
discrepancy. To bridge this gap, we introduce MoK-RAG, a novel multi-source RAG
framework that implements a mixture of knowledge paths enhanced retrieval
mechanism through functional partitioning of a large language model (LLM)
corpus into distinct sections, enabling retrieval from multiple specialized
knowledge paths. Applied to the generation of 3D simulated environments, our
proposed MoK-RAG3D enhances this paradigm by partitioning 3D assets into
distinct sections and organizing them based on a hierarchical knowledge tree
structure. Different from previous methods that only use manual evaluation, we
pioneered the introduction of automated evaluation methods for 3D scenes. Both
automatic and human evaluations in our experiments demonstrate that MoK-RAG3D
can assist Embodied AI agents in generating diverse scenes.

## Full Text


<!-- PDF content starts -->

MoK-RAG: Mixture of Knowledge Paths Enhanced
Retrieval-Augmented Generation for Embodied AI Environments
Zhengsheng Guo, Linwei Zheng, Xinyang Chen, Xuefeng Bai, Kehai Chen∗, Min Zhang
Institute of Computing and Intelligence, Harbin Institute of Technology, Shenzhen, China
zhengshguo@gmail.com, 220110604@stu.hit.edu.cn
Abstract
While human cognition inherently retrieves
information from diverse and specialized
knowledge sources during decision-making
processes, current Retrieval-Augmented Gener-
ation (RAG) systems typically operate through
single-source knowledge retrieval, leading
to a cognitive-algorithmic discrepancy. To
bridge this gap, we introduce MoK-RAG,
a novel multi-source RAG framework that
implements a mixture of knowledge paths en-
hanced retrieval mechanism through functional
partitioning of a large language model (LLM)
corpus into distinct sections, enabling retrieval
from multiple specialized knowledge paths.
Applied to the generation of 3D simulated
environments, our proposed MoK-RAG3D
enhances this paradigm by partitioning 3D
assets into distinct sections and organizing
them based on a hierarchical knowledge tree
structure. Different from previous methods that
only use manual evaluation, we pioneered the
introduction of automated evaluation methods
for 3D scenes. Both automatic and human
evaluations in our experiments demonstrate that
MoK-RAG3D can assist Embodied AI agents
in generating diverse scenes.
1 Introduction
The rapid advancements in language models have
led to the emergence of Retrieval-Augmented
Generation (RAG) (Ji et al., 2024), which
synergize the generative capacity of large language
models (LLMs) (Wu et al., 2024b) with external
knowledge retrieval. These systems substantially
enhance the ability of LLMs to produce more
accurate and contextually relevant responses by
retrieving pertinent information from a predefined
knowledge corpus. This approach has proven
effective in various applications, including question
answering and document generation, where
grounding in external data improves output quality
(Wang et al., 2024b).
∗Corresponding Author
a. Human Beings Retrieve Different Knowledge from Multiple  Knowledge Paths
b. LLM Retrieve Knowledge from Single  SourceReply
Somatosensory
Knowledge
Logical
KnowledgeVisual
KnowledgeAuditory
Knowledge
Single
SourceQuery
retrieveReply
retrieveQuery
Prefrontal 
LobeOccipital 
LobeParietal 
Lobe
Temporal 
Lobe(Auditory Knowledge ）（Visual Knowledge ）（Somatosensory Knowledge ）
（Logical Knowledge ）Multiple Knowledge PathsFigure 1: A figure showing the difference between
human beings and LLM agents. In human cognition,
decisions are often made by retrieving information from
diverse knowledge sources. However, current Retrieval-
Augmented Generation (RAG) systems typically rely
on a single knowledge corpus.
While existing RAG systems demonstrate
notable effectiveness, they remain fundamentally
constrained by their reliance on a singular, generic
knowledge corpus (Jiang et al., 2024; Sergent,
1987). This limitation prevents them from
fully emulating the multifaceted and modular
nature of human cognition, where decision-
making inherently involves dynamic retrieval
from multiple specialized knowledge sources.
Neuroscientific studies by Roland and Zilles
(1998); Gazzaniga (1995); Barrett et al. (2003)
reveal that human knowledge organization follows
a specialized neural architecture: the left cerebral
hemisphere predominantly processes analytical and
logical information, while the right hemisphere
specializes in creative synthesis and holistic pattern
recognition. This neurocognitive division enables
humans to perform context-sensitive information
retrieval from distinct neural repositories whenarXiv:2503.13882v1  [cs.LG]  18 Mar 2025

formulating responses to complex queries. Current
RAG implementations, as illustrated in Figure 1,
contrast sharply with this biological paradigm
by operating through a monolithic knowledge
base. Such architectural simplicity inherently
restricts their ability to perform domain-specific
information retrieval and contextual adaptation,
always leading to responses that are incomplete
or lacking key details. We refer to this problem as
Reply Missing .
To address this problem, we propose MoK-
RAG (Mixture of Knowledge Paths Enhanced
Retrieval-Augmented Generation), a novel RAG
framework that segments the LLM corpus into
distinct sections, enabling simultaneous retrieval
from multiple specialized knowledge paths. Our
approach models human cognitive specialization
to enhance contextual relevance, adaptability, and
mitigate the Reply Missing .
Preliminary experiments indicate that the
generation of 3D simulated environments, also
exhibits a high occurrence rate of the Reply Missing
problem. Thus We extend MoK-RAG to propose
MoK-RAG3D, a specialized adaptation designed to
enhance the generation of 3D environments. MoK-
RAG3D adheres to the MoK-RAG framework
while introducing two domain-specific techniques:
First, it splits 3D assets into distinct retrieval
sections, categorizing them based on their types
and contextual relevance. Second, Structural
Knowledge Organization is utilized to organize
these sections using a hierarchical knowledge tree
structure, which facilitates efficient retrieval and
assembly of assets, ensuring that the generated
environments are both cohesive and contextually
appropriate. The contributions of our work can be
summarized as follows:
1.To address the Reply Missing problem,
we introduce MoK-RAG, the first multi-
source RAG framework enabling multi-path
knowledge retrieval.
2.To mitigate the high occurrence of Reply
Missing in 3D environment generation, we
extend MoK-RAG to propose MoK-RAG3D.
3.MoK-RAG3D pioneers automated evaluation
for 3D scene generation, with both automatic
and human assessments confirming its effec-
tiveness in enhancing Embodied AI agents’
ability to generate diverse scenes.2 Related Work
Retrieval-Augmented Generation Retrieval-
Augmented Generation (RAG) enhances LLMs
by retrieving relevant document chunks from
external knowledge bases using semantic similarity.
Existing RAG methods mainly focus on improving
retrieval algorithms (Wang et al., 2024a; Jiang et al.,
2024; Ji et al., 2024; Qian et al., 2024) or refining
generation quality (Qi et al., 2024; Wu et al., 2024a;
Fang et al., 2024; Adak et al., 2025; Gou et al.,
2023). Additionally, RAG-based LLM agents have
gained attention (Zhu et al., 2024; Wang et al.,
2024b). However, these approaches treat all items
as a single corpus, ignoring inherent object features
for multi-source retrieval. To address this, we
introduce MoK-RAG, the first multi-source RAG
agent system.
Embodied AI Environments Generation Pre-
vious works rely on 3D artists for environment
design (Deitke et al., 2020; Gan et al., 2020;
Khanna et al., 2024; Kolve et al., 2017; Li
et al., 2023; Puig et al., 2018; Xia et al.,
2018), which limits scalability. Some methods
construct scenes from 3D scans (Ramakrishnan
et al., 2021; Savva et al., 2019; Szot et al.,
2021), but these lack interactivity. Procedural
frameworks like PROCTHOR (Deitke et al., 2022)
and Phone2Proc (Deitke et al., 2023a) generate
scalable environments. HOLODECK (Yang et al.,
2024) is a system that generates 3D environments
to match a user-supplied prompt. However, these
approaches retrieve 3D objects from a single corpus
without leveraging object relationships. MoK-
RAG3D addresses this by utilizing multi-source
retrieval to enhance contextual coherence.
3 Methodology
Problem Formulation In this paper, we study
the problem of utilizing RAG for Embodied AI
Environments. Existing RAG systems mainly
focus on retrieving knowledge from a monolithic
knowledge base. To advance this technique,
we first explore how this design hinders the
effectiveness of RAG systems.
In many cases, an ideal reply of RAG
systems is structured and composed of multiple
interdependent sections. However, traditional
RAG systems, which rely on a single monolithic
knowledge corpus, often fail to retrieve all
necessary components, leading to responses that

§2.1.1＆§2.2.2 Splitting Module
Splitting ModuleConstraint ModuleQA
Agent
Classify the knowledge 
into categories based on 
the following specified 
rules:
[Category 1] :……
[Category 2] : ……
[Category 3] :……
……Splitting 
PromptComplete
CorpusInput 
Query
ReplyOrganized 
Multi -source KnowledgeConstraint Policy
Category 1
Category 2
Category 3
……Organize accepted knowledge:
-Arrange in [specific format]
-Center around knowledge 
from [specific category]Accept
Reject
-Knowledge from 
[specific category]
classifier§2.1.2＆§2.2.3 Constraint Module §2.2.4 Generation Module
AgentKG
 Agent
……… (Adapative by tasks)
Splitting
Agent
Single Corpusmain_object
paired_object
other_object
Constraint
Agent
Organized Assets
Layout
Agent
Constraint Policy
Constrain objects in 
a tree format, with 
main objects serving 
as roots. In the target 
scene,  a child node’s 
layout position 
depends on its parent 
node.Splitting 
Prompt 
Classify the objects 
into categories 
according to the 
following defined 
rules:
[main_object ]: ……
[paired_object ]: ……
[other_object ]: ……
……
MoK -
Rag
MoK -
Rag
3D➢Left/Right
➢Rotation and 
OrientationLayout Policy
Accept
Reject
-Knowledge from 
[specific category]
➢Distance
➢Support
Figure 2: Overview of MoK-RAG and MoK-RAG3D. MoK-RAG consists of Splitting and Constraint modules
for multi-source knowledge retrieval and Generation Module for response generation. MoK-RAG3D refines the
Generation Module of the MoK-RAG framework into a dedicated Layout Module to facilitate scene generation.
are incomplete or lacking key details. We refer to
this problem as Reply Missing . For example, in a
multimodal query answering task, an LLM may be
asked to generate an image of a dragon along with
a detailed caption. A traditional RAG framework,
constrained to a single retrieval source, may only
retrieve partial information—either the image or
the caption—resulting in an incomplete response.
Inspired by the decision-making processes of
human cognition, retrieving information from
diverse and specialized knowledge sources could
be a promising direction. Hence, our aim is to
address this issue by leveraging multiple retrieval
sources, ensuring each section of the reply is
constructed from the most relevant knowledge
base. For example, the new RAG design could
retrieve the image from an image database while
simultaneously retrieving the text description from
a textual knowledge source, thereby retrieving
knowledge from diverse knowledge paths and
completing the structured response.
To this end, we propose MoK-RAG to achieve
retrieving knowledge through diverse knowledge
paths and propose MoK-RAG3D to adapt MoK-
RAG to the problem of generation of 3D simulated
environments. As illustrated in Figure 2, the MoK-
RAG framework consists of three key components:
a Splitting Module , which partitions a knowledgebase into multiple knowledge paths, a Constraint
Knowledge Module , which organizes the retrieved
knowledge, and a Generation Module to generate
reply. MoK-RAG3D refines the Generation
Module of the MoK-RAG framework into a
dedicated Layout Module to facilitate scene
generation. Next, we will illustrate the design of
each module.
3.1 MoK-RAG
Splitting Module The core insight of MoK-
RAG lies in retrieving knowledge from diverse
knowledge paths. Hence the key is to segment
the knowledge base Kinto multiple specialized
knowledge bases K1, K2, ..., K m, each base
aligned with a specific domain or contextual theme.
To achieve this goal, a dedicated splitting module
is employed to partition the knowledge base. This
module can be implemented using a classifier or
an LLM-based agent, depending on the specific
task requirements. The category set can be
predefined or dynamically determined based on
the characteristics of the task. Notably, when
the number of categories is set to one, MoK-
RAG degenerates into a conventional RAG system,
making it a more generalized framework.
Formally, given a knowledge base K=
{k1, k2, ..., k n}containing nknowledge pieces and

a category set C={c1, c2, ..., c m}, the objective
of the splitting module is to assign each knowledge
piece kito an appropriate category cj. This process
can be represented as:
fsplit:ki→cj,
where cj= arg max
c∈CAlignmentScore (ki, c)(1)
where fsplitdenotes the splitting function, and
AlignmentScore (ki, c)is a relevance function that
evaluates the alignment between kiand category c.
Constraint Module After segmenting the knowl-
edge base into multiple sections, it becomes
essential to not only retrieve relevant knowledge
but also to effectively organize the retrieved
information into a structured form that enhances the
generative capabilities of the LLM. While retrieval
algorithms have been extensively studied in prior
works (¸ Sakar and Emekci, 2025), we focus here
on the organization of retrieved knowledge from
multiple knowledge bases. This process can be
categorized into two key aspects:
Access Policy. This policy determines whether
a retrieved knowledge piece from a specific
knowledge base should be accepted or rejected for
inclusion in the final output. Formally, given a
retrieved knowledge set K′={k′
1, k′
2, ..., k′
l}, an
access function faccess is defined as:
faccess(k′
i) =(
1,if SelectionScore (k′
i)≥τ
0,otherwise
(2)
where SelectionScore (k′
i)represents the relevance
score of k′
i, and τis a predefined threshold
controlling knowledge selection.
Knowledge Organization Policy. This policy
defines the structural arrangement of the final
knowledge representation. For instance, if the
final output is a hierarchical knowledge tree,
the organization algorithm must determine the
placement of different nodes and their interrela-
tions. Formally, given a hierarchical knowledge
representation T= (N, E ), where Ndenotes the
set of nodes (knowledge units) and Erepresents the
edges (relationships), the organization function forg
assigns retrieved knowledge pieces to appropriate
nodes:
forg(k′
i)→nj,
where nj∈N, Relation (nj, nk)∈E.(3)
By enforcing these constraints, MoK-RAG
can effectively refines the retrieved information,ensuring both the quality and structured coherence
of the knowledge used for generation.
3.2 MoK-RAG3D
Due to the complex requirements of 3D environ-
ment generation, the Reply Missing Problem occurs
more frequently in this domain. In this section, we
first analyze its occurrence rate and then introduce
the key modules of MoK-RAG3D.
The Occurance Rate of Problem Reply Missing
The task of 3D environment creation involves
generating realistic virtual spaces based on textual
descriptions. Within this task, we define two
critical categories of objects: main objects and
paired objects. Main objects are the central
elements of an environment, without which the
scene cannot be functionally or semantically
complete. For instance, in a living room, a
sofa serves as the main object—without it, the
room loses its defining purpose. Similarly, in
a bedroom, the bed is indispensable. On the
other hand, paired objects refer to elements that
frequently appear together within the same context,
reinforcing semantic and functional coherence.
Examples include a pot and a stove in a kitchen
or a monitor and a keyboard in an office setting.
We observed that the Reply Missing problem
frequently occurs in the 3D environment generation
task. This problem manifests as missing essential
objects, either main objects or paired objects,
leading to incomplete or inconsistent environments.
A missing main object results in a failure
to establish the fundamental identity of the
environment, while a missing paired object disrupts
the expected co-occurrence patterns, reducing
realism and usability.
To quantify this issue, we conducted an
empirical study where we generated 100 3D
environment samples using a standard procedural
generation approach. Each generated environment
was then annotated by human experts to identify
missing main objects and paired objects. The
results, presented in Figure 3, indicate that 31%
of environments lack their main objects, while
a significantly higher 59% of paired objects are
missing.
These findings highlight the severity of the
Reply Missing problem in 3D environment creation.
The high occurrence rate of missing elements
underscores the urgent need for an improved
retrieval and generation mechanism. Addressing

this issue is essential for ensuring the completeness,
realism, and functional integrity of automatically
generated 3D environments.
Figure 3: The Occurance Rate of Problem Reply
Missing from main objects and paired objetcs two
aspects.
Splitting Module of MoK-RAG3D Following
the structural design of MoK-RAG, the splitting
module in MoK-RAG3D is implemented using
an LLM-based agent. To address the Reply
Missing problem in 3D environment generation,
the 3D object base is partitioned into three distinct
sections: the main objects base , the paired objects
base, and the other objects base .
Formally, given a 3D object knowledge base
O={o1, o2, ..., o n}consisting of nobjects, and
a category set C={cmain, cpaired, cother}, the
splitting module assigns each object oito the most
appropriate category cj. This can be formulated as:
fsplit:oi→cj,
where cj= arg max
c∈CAlignmentScore (oi, c)(4)
where fsplitrepresents the splitting function, and
AlignmentScore (oi, c)is a scoring function that
quantifies the relevance of object oito category c.
Constraint Module of MoK-RAG3D As illus-
trated in Figure 2, the splitting agent partitions
the knowledge base into three sections: the main
objects base, the paired objects base, and the other
objects base, denoted as C={cmain, cpaired, cother}.
The constraint module in MoK-RAG3D follows the
fundamental structure of MoK-RAG and consists
of two crucial aspects:
Access Policy: After retrieving the most relevant
objects from different knowledge bases using LLM
Agent, an access policy is applied to filter and
refine the retrieved objects. Formally, given a
set of retrieved objects O={o1, o2, ..., o k}from
multiple knowledge bases, the access functionfaccess operates as follows:
Ofiltered =faccess(O), (5)
where Ofiltered represents the final selection of
objects after filtering redundant or irrelevant
elements.
Knowledge Organization Policy: The retrieved
knowledge is structured into a hierarchical tree
using a multi-round querying strategy of LLM.
Specifically, given the retrieved main objects set
M={m1, m2, ..., m p}, the paired objects set
P={p1, p2, ..., p q}, and the other objects set
O={o1, o2, ..., o r}, the hierarchical organization
follows these steps:
•Root Node Decision: The main objects from
Mare selected as the root nodes of the
hierarchical tree.
•Node Hierarchy Determination: The LLM
is iteratively queried to determine the child
nodes for each parent node. This process
continues recursively until all objects from
PandOare assigned appropriate positions
within the tree. Ultimately, this results in
the construction of multiple hierarchical trees,
where the number of trees corresponds to the
number of main objects in M.
This hierarchical organization ensures that the
retrieved knowledge is systematically structured,
allowing for a coherent and contextually rich 3D
environment generation.
Layout Module of MoK-RAG3D After con-
structing the 3D layout tree, it is crucial to establish
the spatial relationships between different objects
to ensure a coherent scene structure. We define
four key relationship categories as follows:
•Left/Right: This relationship specifies the
relative horizontal positioning of objects,
determining whether an object is placed to
the left or right of another object.
•Rotation and Orientation: This aspect defines
the angular alignment of objects, ensuring that
they are correctly rotated to fit the intended
scene context.
•Distance: This relationship governs the spatial
separation between objects, maintaining a
realistic distribution of objects within the
environment.

•Support: This category captures structural
dependencies, ensuring that objects requiring
support (e.g., a book on a table) are correctly
positioned with respect to their supporting
surfaces.
To determine these relationships, we iteratively
query the LLM for each of the four defined
relationships along every edge in the layout tree.
During the layout process, the position of the root
node (i.e., the main object) is first determined.
Subsequently, child nodes are placed iteratively by
considering their respective relationships with their
parent nodes, ensuring a structurally consistent and
semantically meaningful 3D environment.
4 Experiments
4.1 Experimental Setup
Datasets. We utilize Objaverse 1.0 (Deitke
et al., 2023b), a large-scale dataset containing
over 800,000 3D models, as the source for
object selection in 3D environment construction.
Following Yang et al. (2024), we evaluate our
LLM agents across two categories: residential and
diverse scenes. The residential scenes include
bathroom, bedroom, kitchen, and living room.
For diverse scenes, we use the MIT Scenes
Dataset (Quattoni and Torralba, 2009), which
provides the largest available collection of indoor
scene categories across various domains.
Metrics. Following Yang et al. (2024), we
conduct large-scale human evaluations to assess the
quality of generated 3D environments. Annotators
rate the scenes on a scale of 1 to 5 based on asset
selection, layout coherence, and overall alignment
with the intended scene type. Additionally, inspired
by Wu et al. (2024b), we introduce an automated
evaluation method for 3D environment generation.
Specifically, we leverage different LLMs as
evaluators to provide an objective assessment of
the generated scenes.
Models. MoK-RAG3D comprises three core
components: the splitting agent, the constraint
agent, and the QA agent, all implemented using
GPT-4-1106-preview (Achiam et al., 2023). In
our current implementation, MoK-RAG3D can
generate a single room in approximately three
minutes, including the time required for API calls
and layout optimization. All experiments are
conducted on a MacBook equipped with an M1
chip.4.2 Human Evaluation
To assess the quality of MoK-RAG3D-generated
scenes, we conduct comprehensive human evalu-
ations involving 120 participants across two user
studies: (1) a comparative analysis of residential
scene generation; (2) an evaluation of MoK-
RAG3D’s capability in generating diverse scenes.
Residential Scenes Evaluation. We conducted
a human evaluation with 120 generated scenes,
evenly distributed across four residential scene
types (30 scenes per type) for both MoK-RAG3D
and the HOLODECK baseline. Both systems
utilized the same Objaverse asset set to ensure a
fair comparison.
For MoK-RAG3D, we provided the scene type
(e.g., “bedroom”) as the input prompt for scene
generation. Scenes of the same type from both
systems were paired, resulting in 120 matched
scene pairs. Each pair was presented to annotators
as two shuffled top-down view images, ensuring
that the generating system remained anonymous.
Annotators were asked to evaluate each scene
based on three key criteria: (1) Asset Selection:
Which system selects 3D assets that are more
accurate and faithful to the scene type? (2) Layout
Coherence: Which system arranges 3D assets in
a more realistic and logically consistent manner
(considering position and orientation)? (3) Overall
Preference: Given the scene type, which scene is
preferred overall?
Figure 4 shows a clear preference for MoK-
RAG3D in human evaluations compared to
HOLODECK. Annotators favored MoK-RAG3D
in Asset Selection (42%), Layout Coherence
(42%), and demonstrated a significant preference
in Overall Preference (48%). These results indicate
that MoK-RAG3D produces more realistic and
semantically appropriate 3D environments.
MoK -RAG3D 
507
48% Holodeck
385
36%Equal
168
16%Overall Preference
MoK -RAG3D 
428
42%
Holodeck
301
29%Equal
291
29%Asset Selection
MoK -RAG3D 
430
42%
Holodeck
359
35%Equal
231
23%Layout Coherence
Figure 4: Comparative human evaluation of MoK-
RAG3D and HOLODECK across three criteria. The pie
charts show the distribution of annotator preferences,
showing both the percentage and the actual number of
annotations favoring each system.

2.32.83.33.84.3 Human PreferenceHolodeck Residential AverageMoK -RAG3D Residential Average
Better than Holodeck Average
wine cellar
 casino
 laundromat locker room
garage
 children room
 dental office
 dining roomFigure 5: Human evaluation on 52 scene types from MIT Scenes Dataset (Quattoni and Torralba, 2009) with
qualitative examples. The two horizontal lines represent the average score of MoK-RAG3D and HOLODECK on
four types of residential scenes (bedroom, living room, bathroom and kitchen.)
Scenes Diversity Analysis. To assess MoK-
RAG3D’s performance beyond residential scenes,
we conducted a human evaluation on 52 scene
types from the MIT Scenes Dataset, covering five
categories: Stores (deli, bakery), Home (bedroom,
dining room), Public Spaces (museum, locker
room), Leisure (gym, casino) and Working Space
(office, meeting room). We prompt MoK-RAG3D
to produce five outputs for each type using only
the scene name as the input, accumulating 260
examples across the 52 scene types. Annotators
are presented with a top-down view image and a
360-degree video for each scene and asked to rate
them from 1 to 5 (with higher scores indicating
better quality), considering asset selection, layout
coherence, and overall match with the scene type.
Figure 5 demonstrates the human preference
scores for diverse scenes with qualitative examples.
Compared to SpiltRagFor3D’s performance in
residential scenes, SpiltRagFor3D achieves higher
human preference scores over half of (29 out of 52)
the diverse scenes.
4.3 Automatic Evaluation
Automatic Evaluation on generated 3D scenes.
To assess the quality of the generated environments,
we employ two evaluation models: (1) GPT-4o, aclosed-source model that shares the same origin as
the LLM agents used in our system; and (2) LLaV A,
an open-source model known for its strong multi-
modal comprehension capabilities.
To facilitate evaluation, we transform each
generated 3D environment into a sequence of four
images by rotating the scene every 90 degrees.
These images are then fed into the evaluation
models. The evaluation models are asked to rate
them from 1 to 5 (with higher scores indicating
better quality), considering asset selection, layout
coherence, and overall match with the scene type.
Shown in Figure 6, the evaluation results from
both models in all residential scenarios consistently
indicate that MoK-RAG3D outperforms Holodeck
in most environments. These results highlight
MoK-RAG3D’s overall advantage in generating
high-quality 3D environments.
Effectiveness evaluation( Reply Missing ).MoK-
RAG3D’s multi-source retrieval enables precise
control over feature-specific content, offering a
distinct advantage in mitigating Reply Missing .
Figure 7 illustrates that incorporating the MoK-
RAG3D method leads to a substantial reduction in
the missing rate, with a decrease of 9.52% for main
objects and 27.22% for paired objects.

3.603.83
3.383.85
3.723.83
3.453.88
3.003.303.603.904.20
Bathroom Bedroom Kitchen Living RoomAssessment ScoreHolodeck MoK-RAG3D
3.453.523.50 3.503.523.533.523.58
3.303.403.503.603.70
Bathroom Bedroom Kitchen Living RoomAssessment ScoreHolodeck MoK-RAG3D
(a) Automatic evaluation results by gpt4o (b) Automatic evaluation results by llavaFigure 6: Automatic evaluation results comparing Holodeck and MoK-RAG3D across residential scenarios. The
results from both GPT-4o and Llava consistently indicate MoK-RAG3D outperforms Holodeck in all evaluated
environments.
30.77%58.97%
21.25%31.25%
0%20%40%60%80%
Main Objects Paired ObjectsMissing RateHolodeck MoK-RAG3D
(-9.52% )(-27.72% )
Figure 7: Comparison of missing rates between
Holodeck and MoK-RAG3D for main objects and
paired objects. MoK-RAG3D significantly reduces the
missing rate.
Effectiveness evaluation(scene quality). We use
CLIP Score to assess the visual coherence between
the top-down view of a generated scene and
its designated scene type, following the prompt
template: “a top-down view of [scene type].”
Additionally, human-designed scenes from iTHOR
serve as an upper bound for reference. As
shown in Figure 8, MoK-RAG3D outperforms
HOLODECK in most scenarios and closely
approaches iTHOR, demonstrating its ability to
generate scenes comparable to human-beings.
4.4 Visual Results
Figure 9 is one residential scenes example
exhibiting intuitive main objects-centric zoning,
which can prove the effectiveness of Mok-RAG3D.
More visual results can be seen in Appendix.
5 Conclusion
In this paper, we introduce MoK-RAG, the first
RAG framework enabling multi-path knowledge
31.8034.38
30.6234.94
32.1534.20
30.6033.53
31.7234.51
30.7133.96
2830323436
Bathroom Bedroom Kitchen Living RoomClip ScoreMoK-RAG3D Holodeck iTHORFigure 8: CLIP Score comparison over four residential
scene types. * denotes iTHOR scenes are designed by
human experts.
Bathroom
 Bedroom
 Kitchen
 Living Room
Figure 9: one example of residential scene results.
retrieval through functional partitioning of LLM
knowledge bases, facilitating concurrent multi-
source information retrieval. We further extend
this framework to 3D environment generation with
MoK-RAG3D, which improves scene realism and
diversity. Additionally, MoK-RAG3D pioneers
automated evaluation for 3D scene generation, with
both automatic and human assessments validating
its effectiveness in enhancing Embodied AI agents’
ability to generate diverse scenes.

6 Limitations
In this paper, MoK-RAG and MoK-RAG3D
demonstrates excellent performance in enhancing
Embodied AI agents’ ability to generate diverse
scenes. However, due to the lack of domain-
specific hardware resource, it struggles with testing
real robot in the generated scenes. This highlights
the need for further enhancement to evaluation of
the generated 3D scenes.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 .
Sayantan Adak, Pauras Mangesh Meher, Paramita Das,
and Animesh Mukherjee. 2025. REVerSum: A multi-
staged retrieval-augmented generation method to
enhance Wikipedia tail biographies through personal
narratives. In Proceedings of the 31st International
Conference on Computational Linguistics: Industry
Track , pages 732–750, Abu Dhabi, UAE. Association
for Computational Linguistics.
NA Barrett, MM Large, GL Smith, F Karayanidis,
PT Michie, DJ Kavanagh, R Fawdry, D Henderson,
and BT O’Sullivan. 2003. Human brain regions
required for the dividing and switching of attention
between two features of a single object. Cognitive
brain research , 17(1):1–13.
Matt Deitke, Winson Han, Alvaro Herrasti, Aniruddha
Kembhavi, Eric Kolve, Roozbeh Mottaghi, Jordi
Salvador, Dustin Schwenk, Eli VanderBilt, Matthew
Wallingford, Luca Weihs, Mark Yatskar, and Ali
Farhadi. 2020. Robothor: An open simulation-to-real
embodied ai platform. In 2020 IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition
(CVPR) , pages 3161–3171.
Matt Deitke, Rose Hendrix, Ali Farhadi, Kiana Ehsani,
and Aniruddha Kembhavi. 2023a. Phone2proc:
Bringing robust robots into our chaotic world.
InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages
9665–9675.
Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca
Weihs, Oscar Michel, Eli VanderBilt, Ludwig
Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and
Ali Farhadi. 2023b. Objaverse: A universe of
annotated 3d objects. In Proceedings of the
IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 13142–13153.
Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca
Weihs, Kiana Ehsani, Jordi Salvador, Winson Han,
Eric Kolve, Aniruddha Kembhavi, and RoozbehMottaghi. 2022. Procthor: Large-scale embodied
ai using procedural generation. Advances in Neural
Information Processing Systems , 35:5982–5994.
Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang,
Xiaojun Chen, and Ruifeng Xu. 2024. Enhancing
noise robustness of retrieval-augmented language
models with adaptive adversarial training. In
Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers) , pages 10028–10039, Bangkok,
Thailand. Association for Computational Linguistics.
Chuang Gan, Jeremy Schwartz, Seth Alter, Martin
Schrimpf, James Traer, Julian De Freitas, Jonas
Kubilius, Abhishek Bhandwaldar, Nick Haber,
Megumi Sano, Kuno Kim, Elias Wang, Damian
Mrowca, Michael Lingelbach, Aidan Curtis, Kevin T.
Feigelis, Daniel Bear, Dan Gutfreund, David Cox,
James J. DiCarlo, Josh H. McDermott, Joshua B.
Tenenbaum, and Daniel L. K. Yamins. 2020.
Threedworld: A platform for interactive multi-modal
physical simulation. ArXiv , abs/2007.04954.
Michael S Gazzaniga. 1995. Principles of human
brain organization derived from split-brain studies.
Neuron , 14(2):217–228.
Qi Gou, Zehua Xia, Bowen Yu, Haiyang Yu,
Fei Huang, Yongbin Li, and Nguyen Cam-Tu.
2023. Diversify question generation with retrieval-
augmented style transfer. In Proceedings of the
2023 Conference on Empirical Methods in Natural
Language Processing , pages 1677–1690, Singapore.
Association for Computational Linguistics.
Yuelyu Ji, Zhuochun Li, Rui Meng, Sonish Sivarajku-
mar, Yanshan Wang, Zeshui Yu, Hui Ji, Yushui Han,
Hanyu Zeng, and Daqing He. 2024. RAG-RLRC-
LaySum at BioLaySumm: Integrating retrieval-
augmented generation and readability control for
layman summarization of biomedical texts. In
Proceedings of the 23rd Workshop on Biomedical
Natural Language Processing , pages 810–817,
Bangkok, Thailand. Association for Computational
Linguistics.
Ziyan Jiang, Xueguang Ma, and Wenhu Chen.
2024. Longrag: Enhancing retrieval-augmented
generation with long-context llms. arXiv preprint
arXiv:2406.15319 .
Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay
Haresh, Brennan Shacklett, Dhruv Batra, Alexander
Clegg, Eric Undersander, Angel X Chang, and
Manolis Savva. 2024. Habitat synthetic scenes
dataset (hssd-200): An analysis of 3d scene scale
and realism tradeoffs for objectgoal navigation.
InProceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition , pages
16384–16393.
Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli
VanderBilt, Luca Weihs, Alvaro Herrasti, Matt
Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu,

et al. 2017. Ai2-thor: An interactive 3d environment
for visual ai. arXiv preprint arXiv:1712.05474 .
Chengshu Li, Ruohan Zhang, Josiah Wong, Cem
Gokmen, Sanjana Srivastava, Roberto Martín-Martín,
Chen Wang, Gabrael Levine, Michael Lingelbach,
Jiankai Sun, et al. 2023. Behavior-1k: A benchmark
for embodied ai with 1,000 everyday activities
and realistic simulation. In Conference on Robot
Learning , pages 80–93. PMLR.
Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li,
Tingwu Wang, Sanja Fidler, and Antonio Torralba.
2018. Virtualhome: Simulating household activities
via programs. In Proceedings of the IEEE conference
on computer vision and pattern recognition , pages
8494–8502.
Jirui Qi, Gabriele Sarti, Raquel Fernández, and Arianna
Bisazza. 2024. Model internals-based answer attribu-
tion for trustworthy retrieval-augmented generation.
InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pages
6037–6053, Miami, Florida, USA. Association for
Computational Linguistics.
Hongjin Qian, Zheng Liu, Kelong Mao, Yujia Zhou,
and Zhicheng Dou. 2024. Grounding language
model with chunking-free in-context retrieval. In
Proceedings of the 62nd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers) , pages 1298–1311, Bangkok, Thailand.
Association for Computational Linguistics.
Ariadna Quattoni and Antonio Torralba. 2009. Rec-
ognizing indoor scenes. In 2009 IEEE conference
on computer vision and pattern recognition , pages
413–420. IEEE.
Santhosh K Ramakrishnan, Aaron Gokaslan, Erik
Wijmans, Oleksandr Maksymets, Alex Clegg, John
Turner, Eric Undersander, Wojciech Galuba, Andrew
Westbury, Angel X Chang, et al. 2021. Habitat-
matterport 3d dataset (hm3d): 1000 large-scale
3d environments for embodied ai. Advances in
Neural Information Processing Systems Datasets and
Benchmarks Track .
Per E Roland and Karl Zilles. 1998. Structural divisions
and functional fields in the human cerebral cortex.
Brain research reviews , 26(2-3):87–105.
Tolga ¸ Sakar and Hakan Emekci. 2025. Maximizing rag
efficiency: A comparative analysis of rag methods.
Natural Language Processing , 31(1):1–25.
Manolis Savva, Abhishek Kadian, Oleksandr
Maksymets, Yili Zhao, Erik Wijmans, Bhavana
Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra
Malik, et al. 2019. Habitat: A platform for embodied
ai research. In Proceedings of the IEEE/CVF
international conference on computer vision , pages
9339–9347.
Justine Sergent. 1987. A new look at the human split
brain. Brain , 110(5):1375–1392.Andrew Szot, Alexander Clegg, Eric Undersander,
Erik Wijmans, Yili Zhao, John Turner, Noah
Maestre, Mustafa Mukadam, Devendra Singh
Chaplot, Oleksandr Maksymets, et al. 2021. Habitat
2.0: Training home assistants to rearrange their
habitat. Advances in neural information processing
systems , 34:251–266.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, et al.
2024a. Searching for best practices in retrieval-
augmented generation. In Proceedings of the
2024 Conference on Empirical Methods in Natural
Language Processing , pages 17716–17736.
Zheng Wang, Shu Teo, Jieer Ouyang, Yongjun Xu, and
Wei Shi. 2024b. M-RAG: Reinforcing large language
model performance through retrieval-augmented
generation with multiple partitions. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 1966–1978, Bangkok, Thailand. Association
for Computational Linguistics.
Di Wu, Jia-Chen Gu, Fan Yin, Nanyun Peng, and
Kai-Wei Chang. 2024a. Synchronous faithfulness
monitoring for trustworthy retrieval-augmented gen-
eration. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing ,
pages 9390–9406, Miami, Florida, USA. Association
for Computational Linguistics.
Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang,
Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon
Wetzstein. 2024b. Gpt-4v (ision) is a human-aligned
evaluator for text-to-3d generation. In Proceedings
of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 22227–22238.
Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax,
Jitendra Malik, and Silvio Savarese. 2018. Gibson
env: Real-world perception for embodied agents. In
Proceedings of the IEEE conference on computer
vision and pattern recognition , pages 9068–9079.
Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt,
Alvaro Herrasti, Winson Han, Jiajun Wu, Nick
Haber, Ranjay Krishna, Lingjie Liu, et al. 2024.
Holodeck: Language guided generation of 3d
embodied ai environments. In Proceedings of the
IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 16227–16237.
Junda Zhu, Lingyong Yan, Haibo Shi, Dawei Yin, and
Lei Sha. 2024. ATM: Adversarial tuning multi-agent
system makes a robust retrieval-augmented generator.
InProceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing , pages
10902–10919, Miami, Florida, USA. Association for
Computational Linguistics.
A APPENDIX

a resort lobby with marble floors and plants
 a rooftop bar with city views and modern seating
 a vintage diner with checkered floors
a family -friendly apartment with colorful decor, a playroom, and cozy bedrooms
a small bakery with fresh bread
 a stylish New York -style apartment with large windows and vintage furniture
a mid -century dining room with a wooden table and retro chairs
a Mediterranean courtyard with tiled floorsFigure 10: Some qualitative example of query-based scene results.

Warehouse Music StudioWine Cellar PantryFigure 11: Some qualitative examples of results of scenes from MIT Indoor Scenes dataset.

Prison Cell Tv Studio
GarageFigure 12: Some qualitative examples of results of scenes from MIT Indoor Scenes dataset.