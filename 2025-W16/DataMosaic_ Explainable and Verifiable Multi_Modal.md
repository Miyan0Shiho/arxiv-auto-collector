# DataMosaic: Explainable and Verifiable Multi-Modal Data Analytics through Extract-Reason-Verify

**Authors**: Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang

**Published**: 2025-04-14 09:38:23

**PDF URL**: [http://arxiv.org/pdf/2504.10036v1](http://arxiv.org/pdf/2504.10036v1)

## Abstract
Large Language Models (LLMs) are transforming data analytics, but their
widespread adoption is hindered by two critical limitations: they are not
explainable (opaque reasoning processes) and not verifiable (prone to
hallucinations and unchecked errors). While retrieval-augmented generation
(RAG) improves accuracy by grounding LLMs in external data, it fails to address
the core challenges of trustworthy analytics - especially when processing
noisy, inconsistent, or multi-modal data (for example, text, tables, images).
We propose DataMosaic, a framework designed to make LLM-powered analytics both
explainable and verifiable. By dynamically extracting task-specific structures
(for example, tables, graphs, trees) from raw data, DataMosaic provides
transparent, step-by-step reasoning traces and enables validation of
intermediate results. Built on a multi-agent framework, DataMosaic orchestrates
self-adaptive agents that align with downstream task requirements, enhancing
consistency, completeness, and privacy. Through this approach, DataMosaic not
only tackles the limitations of current LLM-powered analytics systems but also
lays the groundwork for a new paradigm of grounded, accurate, and explainable
multi-modal data analytics.

## Full Text


<!-- PDF content starts -->

DataMosaic: Explainable and Verifiable Multi-Modal Data
Analytics through Extract-Reason-Verify
Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang
Hong Kong University of Science and Technology (Guangzhou)
Guangzhou, China
ABSTRACT
Large Language Models (LLMs) are transforming data analytics, but
their widespread adoption is hindered by two critical limitations:
they are not explainable (opaque reasoning processes) and not verifi-
able(prone to hallucinations and unchecked errors). While retrieval-
augmented generation (RAG) improves accuracy by grounding
LLMs in external data, it fails to address the core challenges of
trustworthy analytics—especially when processing noisy, inconsis-
tent, or multi-modal data ( e.g.,text, tables, images). We propose
DataMosaic , a framework designed to make LLM-powered ana-
lytics both explainable and verifiable. By dynamically extracting
task-specific structures ( e.g.,tables, graphs, trees) from raw data,
DataMosaic provides transparent, step-by-step reasoning traces
and enables validation of intermediate results. Built on a multi-agent
framework, DataMosaic orchestrates self-adaptive agents that align
with downstream task requirements, enhancing consistency, com-
pleteness, and privacy. Through this approach, DataMosaic not
only tackles the limitations of current LLM-powered analytics sys-
tems but also lays the groundwork for a new paradigm of grounded,
accurate, and explainable multi-modal data analytics.
1 INTRODUCTION
In today’s data-driven world, the majority of information exists in
complex sources such as text documents, PDFs, images, and social
media posts, holding immense untapped potential for actionable
insights [ 8,12,21,22]. However, analyzing these data remains a sig-
nificant challenge due to its diversity, lack of predefined structure,
and the sheer volume of information generated every day [ 1,24].
From healthcare records to legal contracts and financial reports,
these complex sources are critical for decision-making across in-
dustries [4, 9, 18].
Breaking Barriers: LLMs for Multi-Modal Data Analytics. Re-
cently, multi-modal large language models (LLMs) have emerged
as powerful tools to analyze and derive insights from any type of
data, offering unprecedented capabilities to analyze text, visual, and
structured information in a unified manner [ 2,11,26]. For instance,
by feeding a collection of data with varying formats—such as text
documents, images, tables, graphs, or audio—into a multi-modal
LLM ( e.g.,GPT or DeepSeek), users can pose a data analytical task
in natural language, abstracting away the complexities of handling
specific file types or formats. The LLM acts as a versatile analyst,
interpreting the input data in context, applying appropriate reason-
ing strategies, and generating actionable insights tailored to the
specified task [5, 10].
Challenges Amidst Opportunities. Applying LLMs on multi-
modal data analytics faces several key challenges.
ExtensibilityEfficiency
Consistency
Completeness
AccuracyPrivacyDataMosaicFigure 1: Six core dimensions of DataMosaic and potential
research topics under each dimension.
(1)Not Explainable : Multi-modal LLMs often face challenges
in terms of explainability [ 3,15]. When handling various
data types such as text, images, and tables, these models
do not provide a clear and understandable way of showing
how they arrive at their results. Although they can process
diverse data modalities, the internal processes and decision
- making mechanisms are often opaque.
(2)Low Accuracy : Directly using LLMs to analyze a collec-
tion of data can lead to low accuracy [ 17]. LLMs are not
specifically engineered for accurate numerical or statistical
analysis [ 13,28]. Their architecture and training are focused
more on general language understanding and generation
rather than precise quantitative computations. When con-
fronted with complex tasks that require exact numerical
calculations or in - depth domain - specific knowledge,
their reasoning capabilities may fall short [ 19]. This can
result in incorrect predictions, inaccurate classifications, or
unreliable summaries, undermining the reliability of the
analysis [20].
(3)Incomplete Data : LLMs may not inherently identify or
process all necessary information within or beyond a given
dataset [ 7]. They might miss key data points or fail to rec-
ognize missing information. In some cases, they may even
1arXiv:2504.10036v1  [cs.CL]  14 Apr 2025

Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang
A Table B Graph C Tree
Ascent Industries Co. report various asset 
categories in thousands of dollars, which included 
cash and cash equivalents of $1,299,000... Blue 
Dolphin Energy Company had inventory... GSE 
System. Inc had deposits of $106,077...
Company Cash and Cash Equivalents
Ascent Industry $1,299,000
Blue Dolphin Energy $18,713,000
GSE System, Inc $21,411,563
Broad Street Reality, Inc $1,360,000
Cross Timbers Royalty $1,111,159
Q1: Which company has the highest Cash 
and Cash Equivalents?
The company is GSE System, Inc .Broad Street Realty, Inc. have the highest Cash and 
Cash Equivalents, with $ 1,360 thousand.D1
A1
S1
A1’Professor Lee's Profile: Dean, School of 
Computer Science, Research Interests: 
Computer Vision. Working on the AI for Robotics 
initiative in partnership with Professor Wang. 
Professor Harris’s Profile: Associate Professor... 
Q2: How many steps are needed for Dr. 
Harris to reach Dr. Lee? 
3 hops from Dr. Harris to Dr. Lee.We need to know how Dr. Harris and Dr. Lee are 
connected to others. Their relationships are unclear .D2
A2
S2
A2’Dr. Lee Dr. Harris
Dr. Wang Dr. ChenDepartment of
AutomationDepartment of
Computer Science
Cooperate
Subordinate
A Comprehensive Study of Knowledge Editing for 
Large Language Models: Reference: [1]...[101] 
What does the knowledge neuron thesis have to 
do with knowledge? Exploiting Novel GPT -4 APIs: 
Reference...
Q3: Find the reference and citation of the 
paper A.
'Reference': [' R1', 'Rn'], 'Citation': [' C1']D3
A3
S3
A3’"Citation": [ ],
"Reference": ["Exploiting Novel GPT -4 APIs"] 
UnKE : Unstructured Knowledge Editing in 
Large Language Models
Exploiting Novel GPT -4 APIs What does the knowledge neuron 
thesis have to do with knowledge?A Comprehensive Study of Knowledge 
Editing for Large Language Models 
C1
R1 Rn... 
Figure 2: Comparison of LLM Responses Before and After Data Structure Transformation. Here, 𝐴𝑖is the answer obtained
directly from the question 𝑄𝑖and the original document 𝐷𝑖, while𝐴′
𝑖is the answer derived from the question 𝑄𝑖and the
structured knowledge 𝑆𝑖extracted from the original document 𝐷𝑖.
attempt to “hallucinate” by generating plausible but incor-
rect information, which can introduce inaccuracies [6].
(4)Inconsistent Data : When data from multiple sources is
jointly used, inconsistencies may arise due to differences in
formats, units, or even conflicting information [ 21]. LLMs
may not inherently resolve these conflicts and could pro-
duce misleading results by synthesizing inconsistent data
without flagging the discrepancies.
(5)Data Leakage : One of the most significant concerns with
using LLMs for data analytics is the potential exposure of
private or sensitive data [ 14,16,25]. Sending private or
confidential information ( e.g.,personally identifiable infor-
mation, financial data) to external LLMs poses risks of data
breaches or misuse, especially if the model is hosted by
third-party providers.
(6)Low Efficiency : Using large-scale LLMs ( e.g.,models with
billions of parameters like DeepSeek-671B) for data an-
alytics can result in high computational costs and la-
tency [ 23,27]. These models require significant compu-
tational resources for inference, making them inefficient
for real-time or large-scale data processing tasks.
These limitations collectively highlight the challenges of apply-
ing LLMs to complex, real-world data analytics tasks. Addressing
these issues often requires complementary tools, fine-tuning, hy-
brid systems, or additional preprocessing steps.
Desiderata: Explainable and Verifiable Data Analytics. One of
the principal gaps in grounding multi-modal data analytics lies in
the absence of explicit, meticulously curated data structures that are
specifically tailored to particular data analysis tasks. Such structures
are essential as they not only ensure the process is explainable
(addressing Challenge 1) but also enhance its accuracy (tacklingChallenge 2). Furthermore, it is crucial to guarantee that the data
integrated from multiple sources and modalities is both complete
(overcoming Challenge 3) and consistent (resolving Challenge 4).
This is vital for reliable multi-modal data analytics. Moreover, in
numerous applications, data security is of utmost importance, and
thus, data should be safeguarded within the enterprise premises
(meeting Challenge 5). Finally, yet importantly, the data analytics
process needs to be efficient in terms of speed (addressing Challenge
6) to meet the near-real-time demands of modern applications.
Next, we will discuss the benefits of using LLMs for multi-modal
data analytics, specifically when involving well-structured data.
Example 1. Consider the three cases presented in Figure 2, where
each case is provided with a collection of documents and a data ana-
lytical task articulated in natural language
[A] Table. Given the document 𝐷1and query𝑄1, if we could extract
a table𝐷′
1from𝐷1, the LLM can directly query the table to find the
company with the highest value in the “Cash and Cash Equivalents”
column, and gives the result GSE System, Inc.
[B] Graph. Given the document 𝐷2and query𝑄2, if we could extract
a Graph𝐷′
2from𝐷2, showing the relationships between different
individuals (e.g., Dr. Lee, Dr. Harris, Dr. Wang, Dr. Chen), the LLM
can can traverse the graph to determine the shortest path between Dr.
Harris and Dr. Lee.
[C] Tree. Given the document 𝐷3and query𝑄3, if we could extract
and organize the data in a tree structure 𝐷′
3, showing the hierarchical
relationships and references in a document, the LLM can navigate the
tree to locate the relevant reference and citation information. 2
As shown by Example 1, structure extraction offers clear benefits.
It not only boosts the accuracy of LLMs in reasoning but also makes
them more interpretable.
2

DataMosaic: Explainable and Verifiable Multi-Modal Data Analytics through Extract-Reason-Verify
Our Vision. In the realm of machine learning powered data ana-
lytics, bridging the gap in analyzing complex data sources has long
been the elusive ‘holy grail’. The diversity, scale, and unstructured
nature of such data have historically posed insurmountable chal-
lenges, hindering accurate and efficient analysis. Despite extensive
research efforts over decades, this issue remains unresolved, con-
straining organizations from fully capitalizing on their data assets.
However, recent breakthroughs in LLMs inspire us to envision a fu-
ture where data analytics can be democratized, enabling individuals
to extract actionable insights from complex and unstructured data
with unprecedented ease and precision. Our vision encompasses
six key pillars: task-specific data structure extraction and transfor-
mation, step-by-step thinking and action, data completeness, data
consistence, data privacy, and high efficiency.
Our Proposal. We propose DataMosaic , an agentic workflow de-
signed to address the key challenges in transforming complex,
multi-modal data into actionable insights. It bridges the gap be-
tween unstructured or semi-structured data and the requirements
of precise, scalable data analysis by leveraging advanced techniques
for structured extraction, reasoning, and adaptation. An overview
ofDataMosaic ’s main aspects is shown in Figure 1. DataMosaic
ensures explainability through interpretable workflows, improves
accuracy and verifiability with step-by-step data extraction and
reasoning, handles incomplete data through iterative search and
verification, resolves inconsistencies across data sources using in-
telligent reconciliation, mitigates data leakage risks with locally
deployable models, and enhances efficiency with fine-tuned small
models. By integrating these capabilities, DataMosaic empowers
users to extract meaningful insights from diverse sources—such
as text, PDFs, and multi-modal data—while democratizing data
analysis and making it accessible across domains.
Contributions. We make the following notable contributions.
•We discuss the desiderata of explainable and verifiable
multi-modal data analytics. (Section 2)
•We adopt an agentic framework with a now iterative think-
extract-verfiy workflow, towards achieving explainable and
verifiable multi-modal data analytics. (Section 3)
•We further identify key open problems to guide future re-
search in multi-modal data analytics. (Section 4)
2 PROBLEMS AND DESIDERATA
2.1 Problem of Multi-Modal Data Analytics
Multi-modal data analytics aims to extract meaningful insights by
integrating and analyzing data from diverse sources, such as text,
images, audio, video, and sensor data.
Theinput consists of diverse data sources in different modalities,
such as text, images, audio, video, and sensor data, along with a
natural language question that seeks to extract insights or answer
a specific problem. These inputs are often heterogeneous in format,
structure, and semantics, requiring integration and alignment for
meaningful analysis.
Theoutput is a unified response in the form of text ( e.g.,sum-
maries or explanations), tables ( e.g.,structured data), charts ( e.g.,visualizations), or a combination of these formats, designed to pro-
vide actionable insights that address the query while integrating
information from multiple modalities.
2.2 Desiderata
To effectively ground multi-modal data analytics, a system must
satisfy several key desiderata that address the inherent challenges
of integrating and analyzing diverse data sources. These desiderata
ensure that the system is not only capable of handling complex data
but also provides actionable insights that are reliable, interpretable,
and secure. The following points outline the essential requirements:
2.2.1 Explainability. The system must provide clear and under-
standable explanations, particularly through structured data repre-
sentation. This involves using explicit, meticulously curated data
structures that are tailored to specific tasks. These structures allow
for interpretability by clearly showing how data points and features
influence the analysis, enhancing transparency and traceability.
2.2.2 High Accuracy. The system should achieve high precision,
minimizing errors and ensuring reliability. It must be capable of
accurate numerical and statistical computations, especially for tasks
requiring precise quantitative insights, and incorporate domain-
specific knowledge to enhance accuracy.
2.2.3 Completeness. The system must ensure that all relevant data
points are identified and included in the analysis. This involves han-
dling incomplete datasets through iterative search and verification
to avoid missing critical information.
2.2.4 Consistency. The system should resolve inconsistencies
across different data sources by aligning formats, units, and re-
solving conflicting information. This ensures that synthesized data
is coherent and reliable.
2.2.5 Data Privacy. The system must protect sensitive data by
minimizing exposure to external models. It should use locally de-
ployable models and secure data handling practices to prevent data
breaches or misuse.
2.2.6 High Efficiency. The system should operate efficiently, mini-
mizing computational costs and latency. This includes using fine-
tuned small models and optimizing processes to handle large-scale
data processing tasks effectively.
3 THE DATAMOSAIC FRAMEWORK
To achieve the question-based data extraction and transformation
for reasoning mentioned earlier, we propose the DataMosaic frame-
work, as shown in Figure 3. It is a general workflow divided into
the following core modules:
(1)Question Decomposition: It decomposes a complex user-
provided questions into sub-question. Note that, which
sub-questions to generate might depend on the answers for
previous sub-questions to be further optimized.
(2)Structure Selection: Given a sub-question, it decides which
data structured data is preferred, either unstructured, semi-
structure, or structured.
(3)Seek: Given input multimodal input data and a sub-question,
it identifies the fragments of input whose information needs
3

Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang
Question
Questio n Decomposition
Sub-questions
Extraction
Reasoning
AnswerSub-answersNo extra
data needed
Some sub -questions
remain unprocessedSub-question s
can be optimized
Structure
unreasonable
Agent
Main Flow
Optimized FlowInformation 
insufficient
Structure
Selection
Seek
Thinker
Figure 3: A Multi-Agent Workflow
to be extracted. Compared with a full scan using very large
language models, the main purpose of this module is to
improve efficiency.
(4)Extraction: Given a targeted data structure and data frag-
ments, it extracts required data structure.
(5)Reasoning: Given the sub-question and the extracted data,
it will perform the reasoning, providing an answer to the
sub-question.
(6)Thinker: It keeps evaluating the quality and sufficiency
of the outputs of sub-questions. This allows for dynamic
adaptation where some sub-questions may proceed directly
through the pipeline while others might require additional
processing or alternative approaches.
In the data analysis process, the user’s input question serves as
the system’s starting point. The question 𝑄can integrate natural lan-
guage with other modalities like images or tables. This architecture
enables DataMosaic to tackle complex, multi-step reasoning tasks
by decomposing questions into manageable sub-questions, selecting
suitable data structures, retrieving relevant information, extract-
ing structured knowledge, and conducting targeted reasoning—all
while dynamically adapting to the unique challenges of each sub-
task. DataMosaic operates as a modular framework, where each
component functions independently, allowing flexible adaptation
to different domains, problem types, and scenarios.
3.1 Question Decomposition
For the user-provided question 𝑄, we first perform Question De-
composition. This step breaks the complex, holistic question 𝑄
into multiple smaller, more manageable sub-questions. Through
this process, we can more accurately identify the different aspectsrelated to the question and transform them into a set of indepen-
dent, queryable sub-questions. The decomposition is represented
as follows:
𝑄=Decompose(𝑄1,𝑄2,...,𝑄𝑖)
where𝑄is the user’s original question, and 𝑄𝑖are the sub-questions
derived from it. Each sub-question 𝑄𝑖contains a specific, manage-
able query target. These sub-questions are then used in the subse-
quent steps to extract relevant data from the data lake. In this step,
we use an LLM to perform the decomposition, with the following
prompt:
You are a reasoning expert. The following is a complex question: {question}.
Please decompose this question into multiple smaller sub −questions, each
of which should be solvable with single −step reasoning. Each sub −question
should focus on one small aspect and should be clear and easy to
understand. Separate each sub −question with the symbol '||'
3.2 Structure Selection
The structure selection phase selects an appropriate data struc-
ture based on the specific needs of each sub-question to effectively
handle data and support reasoning. Different sub-questions may
require different structured forms to represent the data, ensuring
the efficiency and accuracy of the analysis. In our framework, we
consider structures such as tables, graphs, trees, and natural lan-
guage descriptions. In this phase, for each sub-question 𝑄𝑖, we
select an appropriate structure 𝑆𝑖from a given set of possible struc-
tures𝑆={𝑠1,𝑠2,...,𝑠𝑛}that best expresses the information and
relationships in the question. This can be represented as:
𝑆𝑖=Select(𝑄𝑖,𝑆)
where𝑆𝑖is the selected data structure for sub-question 𝑄𝑖, and𝑆
is the set of candidate data structures. The selection is based on
factors such as the nature of the question, the type of data, and the
expected method of analysis. Here, we also use an LLM to assist in
selecting the data structure, with the following prompt:
This is a data structure selection task. Based on the given {question}, choose
the most suitable data structure to answer the question. You can choose from
the following options: {structure 1}, {structure 2}, ...{structure n}. Return your
answer in the following format: {answer: data structure}, and tell me the
reason for your selection.
3.3 Seek
In the Seek module, we locate and match relevant data fragments
𝐹from the multimodal data lake 𝐿to each sub-question 𝑄𝑖. We
employ a vector-based matching approach for this purpose.
For text data, we segment long texts into smaller chunks 𝑇=
{𝑡1,𝑡2,...,𝑡𝑛}, then convert each chunk into a vector representa-
tion:
𝑣𝑡𝑖=Encoder(𝑡𝑖), 𝑣𝑡𝑖∈R𝑑
For image data, we first generate textual descriptions 𝐶=
{𝑐1,𝑐2,...,𝑐𝑚}of the images, then convert each description into a
vector:
𝑣𝑐𝑖=Encoder(𝑐𝑖), 𝑣𝑐𝑖∈R𝑑
4

DataMosaic: Explainable and Verifiable Multi-Modal Data Analytics through Extract-Reason-Verify
1. Questio n Decomposition
Accuracy EfficiencyStructure Selection
Accuracy Efficiency Efficiency AccuracyExtraction
Accuracy PrivacyReasoningHow many steps are needed for 
Dr. Harris to reach Dr. Lee? 
3 hops from Dr. Harris to Dr. Lee.AnswerQuestion
SeekQuestion...Q1
Q2
Qi
...
ThinkerIs the structure reasonable?
Is the information sufficient? Is the structured data reasonable?Can the sub -question be optimized?S1
S2
S3Tree
Graph
Table
Qi
Accuracy Efficiency Extensibility
Completeness Consistency Consistency
Data Fragments Qi
Structured Data Di’S2
+Qi Di’ Ai
A1
 A
Example:
1) Find the coloration relationship of Dr. Harris. 
2) Find the coloration relationship of Dr. Lee.  
                                         ......
i) Count the steps from Dr. Harris to Dr. Lee. Example:
Sub-question 1: Graph
Sub-question 2: Graph
                      ...
Sub-question i: None (Description)Action:
Decompose the question 
input by the user and plan 
the steps to obtain 
multiple sub -questions.Action:
Select an appropriate data 
structure according to the 
sub-question to support 
subsequent information 
extraction and better 
reasoning. 
Action:
According to the sub -question, 
search for the relevant 
information in the document 
to obtain the data fragments.
Action:
Based on the Structured Data obtained in the 
previous step, obtain the sub -answer for the 
current sub -question. Finally, based on all 
the sub -questions, integrate and output the 
final answer.Action:
Extract information from 
the data structure selected 
in Step 2 and the data 
fragments obtained in Step 
3 to get the structured data.
Example:
Document 1: Paper A (Dr. Harris, Dr. Chen, Dr. John, Dr. Zhang...)
Document 2: Paper B (Dr. Zhang, Dr. Harris, Dr. Mike, Dr. Li...)
Document 3: Paper C (Dr. Mon, Dr. Harris, Dr. John, Dr. Li...)
                                            ......Sub-question 1: Example:
Extraction Schema: (Person1, Relationship, Person2)Sub-question 1:
Structure Data: (Dr. Harris, Dr. Chen, Cooperate), (Dr. Harris, 
Dr. Zhang, Cooperate), (Dr. Harris, Dr. John, Cooperate), (Dr. 
John, Dr. Chen, Cooperate)...1 2
3 4 5
Thinker: 
If the sub -question can be optimized → Step 1
If the chosen data structure is inappropriate → Step 2
If the retrieved information is insufficient → Step 3Thinker: 
If the retrieved information is insufficient → Step 3
Figure 4: A General Framework of DataMosaic
We calculate cosine similarity between each data chunk’s vector
and the sub-question vector:
sim(𝑄𝑖,𝐹)=𝑣𝑄𝑖·𝑣𝐹
∥𝑣𝑄𝑖∥∥𝑣𝐹∥
The most relevant data fragments are then selected for further
processing in the Extraction module.
3.4 Extraction
The Extraction module processes the data fragments from the Seek
module to generate structured data according to the previously
selected structure. This module transforms raw information into
precisely organized formats optimized for reasoning.
For each sub-question 𝑄𝑖with its data fragments 𝐹𝑖and selected
structure𝑆𝑖, we extract the structured data:
𝐷′
𝑖=Extract(𝑄𝑖,𝐹𝑖,𝑆𝑖)
Depending on the selected data structure, the extraction process
follows different methods:
Table. First, we extract the schema based on the question, defining
the columns and relationships needed. Then, we extract tuples from
the data fragments matching this schema and merge them into a
coherent table structure.
Graph. We begin by generating a graph schema that defines the
types of nodes and edges ( e.g.,(person, cooperate, person)). Next,
we extract specific triplets from the data fragments based on thisschema. Finally, these triplets are merged to construct a complete
graph where entities form nodes connected by relationship edges.
Tree. Starting with tree schema generation, we define the hier-
archical relationships ( e.g., citation networks). Then we extract
parent-child tuples from the data fragments, followed by merg-
ing these tuples into a cohesive tree structure that preserves the
hierarchical relationships.
Description. For natural language descriptions, we transform the
data fragments into a concise, relevant summary focused on an-
swering the specific sub-question.
Each data structure implements specific operations (as shown
in Table 1) that facilitate construction and manipulation of the
extracted information, ensuring it’s optimally structured for the
reasoning phase.
3.5 Thinker
The Thinker serves as the critical decision-making component
that evaluates outputs at key junctures and determines the flow
of information processing. It performs several specific evaluation
functions:
Sub-question Optimization: After Question Decomposition, the
Thinker evaluates whether the generated sub-questions are opti-
mally formulated. It assesses if they are properly scoped, clear, and
5

Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, Nan Tang
Table 1: Data Structures and Their Supported Operations
Data Structure Supported Operations
Table create_table ,add_row ,get_row ,delete_row ,
add_column ,add_subtable ,visualize , . . .
Graph add_node ,get_nodes ,delete_node ,add_edge ,
get_neighbor ,build_from_triplets ,add_subgraph ,
merge_semantic_nodes ,visualize , . . .
Tree add_node ,get_nodes ,delete_node ,add_child ,
build_from_triplets ,merge_semantic_nodes ,prune ,
visualize , . . .
Description transform
sufficiently atomic to enable effective information retrieval and rea-
soning. If sub-questions are too complex or ambiguous, the Thinker
redirects them back for refinement.
Structure Verification: Following Structure Selection, the Thinker
examines whether the chosen data structure (table, graph, tree, or
description) is appropriate for representing the information needed
to answer each sub-question. It considers factors such as relation-
ship complexity, hierarchical nature of the data, and reasoning
requirements.
Information Sufficiency: After the Seek phase, the Thinker de-
termines if the information retrieved is sufficient to answer the
sub-question. If crucial information is missing, it may trigger ad-
ditional search iterations with modified parameters or different
approaches.
Structured Data Verification: Following Extraction, the Thinker
evaluates whether the structured data is properly organized and
contains all necessary elements for reasoning. It checks for com-
pleteness, consistency, and suitability for the specific reasoning
tasks ahead.
The Thinker implements these evaluations through specific deci-
sion criteria and thresholds. For instance, when assessing informa-
tion sufficiency, it may examine whether the key entities mentioned
in the sub-question are adequately covered, whether the relation-
ships between these entities are clearly presented, whether the
retrieved information is consistent with known facts, and whether
the necessary numerical or quantitative data are complete. These
decision points are represented by dashed lines in Figure 3 and
the annotations under the Reasoning and Extraction modules in
Figure 4, showing how the Thinker monitors and directs the flow
between modules.
3.6 Reasoning
In the final stage of the workflow, the Reasoning module inte-
grates all extracted structured data to generate answers to the sub-
questions and ultimately compose the final answer to the user’s
original question.
For each sub-question 𝑄𝑖and its corresponding structured data
𝐷′
𝑖, reasoning is performed:
𝐴𝑖=Reasoning(𝑄𝑖,𝐷′
𝑖)
As controlled by the Thinker, some sub-questions may remain
unprocessed if sufficient information is not available or if theirprocessing is deemed unnecessary for the final answer. The final
answer𝐴is derived by integrating all the processed sub-answers:
𝐴=Integrate(𝐴1,𝐴2,...,𝐴𝑛)
This integration process considers the relationships between
sub-questions (whether parallel or sequential) and combines their
answers in a coherent manner to provide a comprehensive response
to the original user question.
4 OPEN PROBLEMS
4.1 Information Extraction from New Data
Modalities
Extracting actionable insights from emerging modalities ( e.g.,3D
point clouds, real-time sensor streams) remains unsolved due to
modality-specific representation gaps. For instance, aligning spa-
tiotemporal sensor data with textual logs requires novel embed-
ding spaces that preserve causal-temporal dependencies. Current
methods ( e.g., contrastive learning) struggle with latent cross-
modal invariants, such as mapping LiDAR geometry to natural
language without losing structural fidelity, necessitating hybrid
neuro-symbolic extractors.
4.2 Information Extraction from New Domains
Domain shifts ( e.g.,from finance to biomedicine) expose brittle-
ness in schema induction and entity linking. A core challenge is
domain-agnostic schema learning: automating the discovery of
domain-specific ontologies ( e.g.,legal clauses, genomic variants)
without labeled data. This demands zero-shot transfer of extraction
rules while avoiding semantic drift ( e.g.,conflating “risk” in finance
vs. healthcare), which current LLM-based adapters fail to address
rigorously.
4.3 Information Extraction for New Data
Structures
Novel structures like hypergraphs ( e.g., multi-way scientific re-
lationships) or topological maps lack extraction grammars. Key
issues include defining minimal schema constraints for understud-
ied structures ( e.g.,cellular complexes in spatial data) and ensuring
composability during merging. For example, how to extract a hy-
pergraph from a mix of text and equations while preserving 𝑛-ary
relations, without resorting to heuristics or manual curation.
4.4 Checking Completeness: Unknown
Unknowns
Current completeness checks ( e.g.,null counting, coverage metrics)
fail to detect contextual omissions in multi-modal data. For instance,
verifying whether a medical report’s table omits critical image find-
ings requires reasoning about cross-modal entailments. Open prob-
lems include formalizing completeness certificates (proofs that all
task-relevant data is extracted) via probabilistic logic, and detecting
“unknown unknowns” through adversarial schema perturbations
or information-theoretic coverage bounds.
6

DataMosaic: Explainable and Verifiable Multi-Modal Data Analytics through Extract-Reason-Verify
REFERENCES
[1]Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora,
Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma
Brunskill, et al .2021. On the opportunities and risks of foundation models. arXiv
preprint arXiv:2108.07258 (2021).
[2]Zui Chen, Zihui Gu, Lei Cao, Ju Fan, Samuel Madden, and Nan Tang. 2023.
Symphony: Towards Natural Language Query Answering over Multi-modal
Data Lakes. In 13th Conference on Innovative Data Systems Research, CIDR 2023,
Amsterdam, The Netherlands, January 8-11, 2023 . www.cidrdb.org. https://www.
cidrdb.org/cidr2023/papers/p51-chen.pdf
[3] Yunkai Dang, Kaichen Huang, Jiahao Huo, Yibo Yan, Sirui Huang, Dongrui Liu,
Mengxi Gao, Jie Zhang, Chen Qian, Kun Wang, et al .2024. Explainable and
interpretable multimodal large language models: A comprehensive survey. arXiv
preprint arXiv:2412.02104 (2024).
[4]Evert de Haan, Manjunath Padigar, Siham El Kihal, Raoul Kübler, and Jaap E
Wieringa. 2024. Unstructured data research in business: Toward a structured
approach. Journal of Business Research 177 (2024), 114655.
[5] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery,
Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang,
et al. 2023. Palm-e: An embodied multimodal language model. (2023).
[6] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2025. A
survey on hallucination in large language models: Principles, taxonomy, chal-
lenges, and open questions. ACM Transactions on Information Systems 43, 2
(2025), 1–55.
[7]Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in
natural language generation. ACM computing surveys 55, 12 (2023), 1–38.
[8] Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang,
Jianfeng Gao, et al .2024. Multimodal foundation models: From specialists to
general-purpose assistants. Foundations and Trends ®in Computer Graphics and
Vision 16, 1-2 (2024), 1–214.
[9] Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang,
Fei Huang, Xianpei Han, Le Sun, and Yongbin Li. 2024. Structrag: Boosting
knowledge intensive reasoning of llms via inference-time hybrid information
structurization. arXiv preprint arXiv:2410.08815 (2024).
[10] Xiao Liu and Yanan Zheng. 2023. Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin
Yang, and Jie Tang. Gpt understands, too. AI Open 2 (2023).
[11] Chenyang Lyu, Minghao Wu, Longyue Wang, Xinting Huang, Bingshuai Liu,
Zefeng Du, Shuming Shi, and Zhaopeng Tu. 2023. Macaw-llm: Multi-modal
language modeling with image, audio, video, and text integration. arXiv preprint
arXiv:2306.09093 (2023).
[12] Supriya V Mahadevkar, Shruti Patil, Ketan Kotecha, Lim Way Soong, and
Tanupriya Choudhury. 2024. Exploring AI-driven approaches for unstructured
document analysis and future horizons. Journal of Big Data 11, 1 (2024), 92.
[13] Seyed Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy
Bengio, and Mehrdad Farajtabar. [n.d.]. GSM-Symbolic: Understanding the Limi-
tations of Mathematical Reasoning in Large Language Models. In The Thirteenth
International Conference on Learning Representations .
[14] Luis Bernardo Pulido-Gaytan, Andrei Tchernykh, Jorge M Cortés-Mendoza,
Mikhail Babenko, and Gleb Radchenko. 2020. A survey on privacy-preserving
machine learning with fully homomorphic encryption. In Latin American High
Performance Computing Conference . Springer, 115–129.
[15] Peng Qi, Zehong Yan, Wynne Hsu, and Mong Li Lee. 2024. Sniffer: Multimodal
large language model for explainable out-of-context misinformation detection. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition .
13052–13062.
[16] Xuedi Qin, Chengliang Chai, Nan Tang, Jian Li, Yuyu Luo, Guoliang Li, and
Yaoyu Zhu. 2022. Synthesizing Privacy Preserving Entity Resolution Datasets. In
38th IEEE International Conference on Data Engineering, ICDE 2022, Kuala Lumpur,
Malaysia, May 9-12, 2022 . IEEE, 2359–2371. https://doi.org/10.1109/ICDE53745.
2022.00222
[17] Safal Shrestha, Minwu Kim, and Keith Ross. 2025. Mathematical Reasoning in
Large Language Models: Assessing Logical and Arithmetic Errors across Wide
Numerical Ranges. arXiv preprint arXiv:2502.08680 (2025).
[18] Vivek Sriram, Ashley Mae Conard, Ilyana Rosenberg, Dokyoon Kim, T Scott
Saponas, and Amanda K Hall. 2025. Addressing biomedical data challenges and
opportunities to inform a large-scale data lifecycle for enhanced data sharing,
interoperability, analysis, and collaboration across stakeholders. Scientific Reports
15, 1 (2025), 6291.
[19] Nan Tang, Chenyu Yang, Ju Fan, Lei Cao, Yuyu Luo, and Alon Y. Halevy. 2024.
VerifAI: Verified Generative AI. In 14th Conference on Innovative Data Systems
Research, CIDR 2024, Chaminade, HI, USA, January 14-17, 2024 . www.cidrdb.org.
https://www.cidrdb.org/cidr2024/papers/p5-tang.pdf
[20] Nan Tang, Chenyu Yang, Zhengxuan Zhang, Yuyu Luo, Ju Fan, Lei Cao, Sam
Madden, and Alon Y. Halevy. 2024. Symphony: Towards Trustworthy Question
Answering and Verification using RAG over Multimodal Data Lakes. IEEE DataEng. Bull. 48, 4 (2024), 135–146. http://sites.computer.org/debull/A24dec/p135.pdf
[21] Jiaqi Wang, Hanqi Jiang, Yiheng Liu, Chong Ma, Xu Zhang, Yi Pan, Mengyuan
Liu, Peiran Gu, Sichen Xia, Wenjun Li, et al .2024. A comprehensive review of
multimodal large language models: Performance and challenges across different
tasks. arXiv preprint arXiv:2408.01319 (2024).
[22] Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli
Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al .2024. Leave no document
behind: Benchmarking long-context llms with extended multi-doc qa. arXiv
preprint arXiv:2406.17419 (2024).
[23] Wenxiao Wang, Wei Chen, Yicong Luo, Yongliu Long, Zhengkai Lin, Liye Zhang,
Binbin Lin, Deng Cai, and Xiaofei He. 2024. Model compression and efficient
inference for large language models: A survey. arXiv preprint arXiv:2402.09748
(2024).
[24] Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Philip S Yu.
2023. Multimodal large language models: A survey. In 2023 IEEE International
Conference on Big Data (BigData) . IEEE, 2247–2256.
[25] Runhua Xu, Nathalie Baracaldo, and James Joshi. 2021. Privacy-preserving
machine learning: Methods, challenges and directions. arXiv preprint
arXiv:2108.04417 (2021).
[26] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and
Dong Yu. 2024. MM-LLMs: Recent Advances in MultiModal Large Language
Models. In Findings of the Association for Computational Linguistics ACL 2024 .
12401–12430.
[27] Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming
Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, et al .2024. A survey on efficient
inference for large language models. arXiv preprint arXiv:2404.14294 (2024).
[28] Yizhang Zhu, Shiyin Du, Boyan Li, Yuyu Luo, and Nan Tang. [n.d.]. Are Large
Language Models Good Statisticians?. In The Thirty-eight Conference on Neural
Information Processing Systems Datasets and Benchmarks Track .
7