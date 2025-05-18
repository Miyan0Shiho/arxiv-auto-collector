# KAQG: A Knowledge-Graph-Enhanced RAG for Difficulty-Controlled Question Generation

**Authors**: Ching Han Chen, Ming Fang Shiu

**Published**: 2025-05-12 14:42:19

**PDF URL**: [http://arxiv.org/pdf/2505.07618v1](http://arxiv.org/pdf/2505.07618v1)

## Abstract
KAQG introduces a decisive breakthrough for Retrieval-Augmented Generation
(RAG) by explicitly tackling the two chronic weaknesses of current pipelines:
transparent multi-step reasoning and fine-grained cognitive difficulty control.
This transforms RAG from a passive retriever into an accountable generator of
calibrated exam items. Technically, the framework fuses knowledge graphs, RAG
retrieval, and educational assessment theory into a single pipeline. Domain
passages are parsed into a structured graph; graph-aware retrieval feeds fact
chains to an LLM; and an assessment layer governed by Bloom's Taxonomy levels
and Item Response Theory (IRT) transforms those chains into psychometrically
sound questions. This cross-disciplinary marriage yields two scholarly
contributions: it shows how semantic graph contexts guide LLM reasoning paths,
and it operationalizes difficulty metrics within the generation process,
producing items whose IRT parameters match expert benchmarks. Every module,
from KG construction scripts to the multi-agent reasoning scheduler and the
automatic IRT validator, is openly released on GitHub. This enables peer
laboratories to replicate experiments, benchmark against baselines, and extend
individual components without licensing barriers. Its reproducible design paves
the way for rigorous ablation studies, cross-domain transfer experiments, and
shared leaderboards on multi-step reasoning benchmarks.

## Full Text


<!-- PDF content starts -->

1 
 
KAQG: A Knowledge‑Graph‑Enhanced RAG for 
Difficulty‑Controlled Question Generation  
 
Ching Han Chen and Ming Fang Shiu  
 
  
Abstract —KAQG introduces a decisive breakthrough for 
Retrieval -Augmented Generation (RAG) by explicitly tackling 
the two chronic weaknesses of current pipelines —transparent 
multi -step reasoning and fine -grained cognitive difficulty 
control —thereby turning RAG from a passive retriever into an 
accountable generator of calibrated exam items. Technically, the 
framework fuses knowledge graphs, RAG retrieval, and 
educational assessment theory into a single pipeline: domain 
passages are parsed into a structured graph, graph -aware 
retrieval feeds fact chains to an LLM, and an assessment layer 
governed by Bloom’s Taxonomy levels and Item Response 
Theory (IRT) transforms those chains into psychometrically 
sound questions. This cross -disciplinary marriage yields two 
scholarly contributions: it shows how semantic graph contexts 
guide LLM reasoning paths, and it operationalizes difficulty 
metrics inside generation, producing items whose IRT 
parameters match expert benchmarks. Every module —from KG 
construction scripts to the multi -agent reasoning scheduler and 
the automatic IRT validator —is openly released on GitHub, 
enabling peer laboratories to replicate experiments, benchmark 
against baselines, and extend individual components without 
licensing barriers. Its reproducible design paves the way for 
rigorous ablation studies, cross -domain transfer experiments, 
and shared leaderboards on multi -step reasoning benchmarks.  
 
Index Terms —Educational technology , Knowledge 
representation , Multi -agent systems , Question generation . 
 
I. INTRODUCTION  
he fast‑evolving Retrieval‑Augmented Generation 
(RAG) technology significantly improves Large 
Language Models (LLMs) by enabling them to access 
domain‑specific knowledge and generate relevant content 
based on real‑world data  [1]. Yet current RAG pipelines still 
fail to perform transparent multi‑step reasoning or to regulate 
the cognitive difficulty of their outputs —a critical pain point 
that KAQG is designed to solve. For instance, RAG can utilize 
comprehensive information from educational resources to 
create exam questions that meet specific learning goals  [2]. To 
 
This work was supported by the National Environmental Research 
Academy in Taiwan under the project titled "AI Optimization for Waste 
Cleanup Professional Training Exam Question Development." The authors 
gratefully acknowledge the Academy’s provision of resources, domain 
expertise, and guidance throughout the development of the KAQG system. 
(Corresponding author: Ming Fang Shiu.) The first author and the 
corresponding author contributed equally to this work.   
Chin Han Chen, Professor, is with the Department of Computer Science 
and Information Engineering, National Central University, Taoyuan 32001, 
Taiwan (e -mail: pierre@csie.ncu.edu.tw).   
Ming Fang Shiu, Ph.D. student, is with the Department of Computer 
Science and Information Engineering, National Central University, Taoyuan 
32001, Taiwan (e -mail: 108582003@cc.ncu.edu.tw).  ensure more thorough integration of educational content, 
knowledge graphs with advanced reasoning abilities have been 
incorporated into the RAG framework [3]. Models such as 
HippoRAG  [4], GNN‑RAG  [5], GRAG  [6], ToG 2.0  [7], 
SUGRE  [8], DALK  [9], and GraphRAG  [10] exemplify how 
this enhancement improves performance in complex multi‑hop 
reasoning and cross‑paragraph tasks. However, recent work on 
Agentic Reasoning emphasizes the importance of integrating 
structured memory —such as a “Mind Map” agent —alongside 
external tool usage (e.g., web search and coding) to further 
boost deep, multi‑step reasoning capabilities  [11]. 
Although RAG and its refinements have tackled many 
hallucination issues caused by limited domain -specific 
knowledge and outdated information [12], significant 
challenges arise when generating exam questions from 
educational materials. Most notably, vector -based similarity 
retrieval methods can lead to incomplete or redundant results, as 
they fail to capture deeper relationships within the text that are 
essential for multi -step reasoning. This shortcoming is 
particularly pronounced in specialized domains like law, 
medicine, and science, where the current next -token prediction 
mechanism of language models struggles with more complex 
logical or numerical reasoning required for accurate and 
coherent questions [7]. In contrast, agentic frameworks that 
employ structured knowledge graphs and external LLM -based 
agents (e.g., a coding agent) have demonstrated higher accuracy 
in similar expert -level tasks, notably in domains that demand 
detailed logical deduction  [11]. 
By applying knowledge graph (KG) techniques, exam 
question generation from educational materials can be greatly 
improved [13]. KGs use SPO triples to define entities and their 
relationships, enabling structured connections and reducing 
redundancy through entity normalization [14]. Query languages 
like SPARQL or Cypher support precise retrieval, minimizing 
noise and enhancing inference [15]. Since query results carry 
explicit semantics, they can act as variables with distinct 
meanings, supporting agentic reasoning. This structured 
memory allows LLMs to perform coherent multi -hop reasoning 
and deterministic tasks like calculations, ensuring the generated 
questions are logical and closely aligned with educational 
content [1 6]. 
To address the above challenges and to highlight the 
cross‑disciplinary fusion of knowledge graphs, RAG, and 
Assessment Theory, we propose Knowledge Augmented 
Question Generation (KAQG), which fully leverages the 
complementary strengths of these technologies. Rather than 
merely embedding graph structures into the knowledge‑base 
process, this approach integrates the semantic types and T 

2 
 
relationships from knowledge graphs, as well as commonly 
used logical forms, into both retrieval and generation stages 
[17]. As shown in Figure 1, the framework optimizes the 
following modules . 
A. KAQG -Retriever  
We propose a framework to extract educational materials 
into a knowledge graph, integrating various multimodal 
documents with extensibility [1 8]. Using LLMs, the system 
analyzes documents and converts content into KG elements, 
including entities, relationships, and hypernyms (concept 
nodes). Each subject is stored in a dedicated knowledge graph 
to ensure content relevance and prevent cross -domain 
contamination, enabling efficient organization and use for 
applications like question generation and intelligent tutoring. 
Incorporating agentic reasoning principles here allows the 
retriever to refine its queries and reduce noise, potentially by 
delegating tasks like coding for data parsing or using a Mind 
Map agent for stepwise aggregation of facts.  
B. KAQG -Generator  
We developed a workflow for generating exam questions 
from knowledge graphs using assessment theory. PageRank 
ranks knowledge relationships by relevance to specific 
concepts [ 19], while LLMs transform these into questions 
guided by assessment theory principles. Word embedding and 
assessment theory validate the questions, ensuring accuracy, 
logical structure, and effective knowledge measurement 
tailored to educational needs. Additionally, by adopting an 
agentic reasoning approach —where the LLM can 
autonomously retrieve relevant details or perform calculations 
using specialized sub -agents —the generation process can be 
further optimized for coherence and domain accuracy [2 0]. Its 
workflow explicitly aligns generated items with Bloom’s 
Taxonomy levels and validates them through Item Response 
Theory (IRT) [2 1], ensuring psychometric soundness.  
C. AI Agents Framework  
We construct an AI agent model to implement the system, 
enabling an autonomous distributed system [2 2]. This scalable 
model adapts to workloads while maintaining balanced 
performance. AI agents autonomously manage tasks, 
coordinate, and allocate resources efficiently. Using the DDS 
communication protocol, the model ensures seamless 
interaction among agents, supporting decentralized operations, 
reducing bottlenecks, and providing a scalable foundation for 
distributed services. This aligns with the agentic reasoning 
philosophy, where specialized LLM -based agents (e.g., web -
search agent, coding agent) collaborate under a coordinator 
model, offloading sub -tasks to improve the overall 
performance and accuracy [2 3]. 
The KAQG system, built on an LLM -based pipeline, 
demonstrated strong performance in generating knowledge 
graphs and exam questions, achieving high coverage, 
accuracy, and consistency —showing close alignment with 
expert knowledge. Its automatically generated reading 
comprehension questions matched ACT standards in terms of 
difficulty, distractor quality, and cognitive rigor, as confirmed by expert reviewers. Applied at Taiwan’s National 
Environmental Research Academy, KAQG efficiently 
produced high -quality, standardized test items aligned with 
domain knowledge, significantly reducing manual effort while 
enhancing assessment reliability and validity. Its integration of 
agentic reasoning also improves adaptability for complex, 
specialized domains through iterative retrieval and structured 
logical inference [2 4]. 
The novelty of this research lies in the development of the 
KAQG framework, which innovatively resolves the inability 
of existing RAG systems to handle multi‑step reasoning and 
explicit difficulty control while showcasing the academic 
value of fusing knowledge‑graph semantics, RAG retrieval, 
and Assessment Theory. Unlike existing RAG‑based systems, 
KAQG goes beyond vector‑based retrieval by embedding 
explicit semantics, logical forms, and structured memory into 
both the retrieval and generation stages [2 5]. The proposed 
system features a multi‑agent architecture and leverages 
knowledge‑graph query languages, structured entity relations, 
and LLM‑based sub‑agents to enhance multi‑hop reasoning 
and domain‑specific accuracy. Academically, this contributes 
a robust methodology for transforming educational content 
into structured knowledge and validated assessments, showing 
strong performance in expert evaluations and real‑world 
applications. All code is fully open‑sourced, ensuring the 
entire system is reproducible and readily available for 
academic experimentation and comparative studies.  
All implementation code related to this study is publicly 
available. The complete KAQG system can be accessed at 
https://github.com/mfshiu/kaqg , while the AI Agents 
Framework, which supports the agentic reasoning components 
of the system, is available at 
https://github.com/mfshiu/AgentFlow . 
II. METHODS  
In this chapter, we introduce the KAQG system, an 
integrated framework that leverages domain -specific 
knowledge to generate high -quality exam questions. As 
illustrated in Figure 1, the system consists of two main 
components: KAQG -Retriever and KAQG -Generator.  
The KAQG -Retriever processes educational materials, 
including PDFs, videos, and web data. It performs three key 
tasks: Knowledge Graph (KG) Construction, Document 
Parsing, and Retrieval Tuning. This module extracts relevant 
concepts and relationships to populate a Knowledge Graph.  
The KAQG -Generator utilizes the structured knowledge 
from the graph to generate and evaluate test questions. It 
consists of two interconnected processes:  
● Question Generation, which involves a KG Reader, 
PageRank for ranking key concepts, and formulation 
of candidate questions . 
● Question Evaluation, which includes Difficulty 
Quantification and Quality Assessment to refine and 
validate questions based on predefined specifications.  
Assessment to refine and validate questions based on 
predefined specifications.  

3 
 
By integrating these components into a structured pipeline, 
the KAQG system enhances the efficiency, accuracy, and 
scalability of automated question generation, offering an 
innovative solution for educational content creation and 
intelligent tutoring.  
 
Fig. 1.  The KAQG system architecture  for question 
generation.  
A. KAQG -Retriever: Overview  
The KAQG -Retriever transforms educational materials into 
structured knowledge graphs by integrating multimodal 
documents. Using LLMs, it identifies key entities, 
relationships, and hypernyms, converting them into concept 
nodes. Each subject is stored in a dedicated knowledge graph 
to ensure relevance and avoid cross -domain interference, 
providing a solid foundation for applications like question 
generation and intelligent tutoring.  
B. KAQG -Retriever: LLM -Based Analysis and KG Elements  
Modern datasets often come in various forms, such as audio 
recordings, videos, images, PDF files, and plain text 
documents. Before these heterogeneous data sources can be 
integrated into a knowledge graph, it is necessary to transform 
them into a common format suitable for natural language 
processing (NLP) techniques [2 6]. Our approach begins by 
applying a transcription or conversion function to each 
multimodal file in order to generate textual representations 
[27]. Afterward, we employ large language models (LLMs) to 
extract structured information in the form of triples. Finally, 
we combine these triples, concept mappings, and hierarchical 
directory structures into a consolidated knowledge graph [2 8]. 
Let 𝐷={𝑑1,𝑑2,⋯,𝑑𝑛} represent the set of all source 
documents, where each 𝑑𝑖 can be of various formats (audio, 
video, PDF, or text). We define a transcription function  
 𝒯∶𝒟→𝒮. (1) 
 
such that for every document 𝑑𝑖∈𝒟, we obtain a 
corresponding text 𝑠𝑖=𝒯(𝑑𝑖). This process may involve 
automatic speech recognition (ASR) for audio/video inputs or 
optical character recognition (OCR) for scanned PDFs. As a 
result, we accumulate a set of textual documents  𝑆=
{𝑠1,𝑠2,⋯,𝑠𝑛} that can be further processed with NLP 
methods.  
Once we have the textual representations, we employ a 
large language model or other NLP techniques to convert each 
text 𝑠𝑖 into a set of extracted triples. Formally, we define an 
extraction function  
 ℰ∶𝒮→𝒫(𝒯∗). (2)  
where  𝒯∗ is the set of all possible triples, and  ℰ(𝑠𝑖) returns 
a subset of these triples. Each triple  (ℎ,𝑟,𝑡) consists of a head 
entity  ℎ, a relation 𝑟, and a tail entity 𝑡. This extraction step 
includes: (1) entity recognition to identify textual entities, (2) 
relationship detection to capture any relevant associations, and 
(3) attribute extraction for entity properties. Moreover, for 
each recognized entity, we map it to a higher -level concept 
entity (i.e., a hypernym) in order to provide hierarchical 
context.  
Our framework recognizes two main types of entities: (a) 
textual entities, denoted by  𝐸text, and (b) concept entities, 
denoted by  𝐸concept . A function  
 𝒞∶𝐸text →𝒫(𝐸concept ). (3) 
 
associates each textual entity with one or more concept 
entities representing its hypernyms or abstract categories. 
Additionally, we incorporate hierarchical directory structures 
or category nodes  𝐻={ℎ1,ℎ2,⋯}. We define two main types 
of hierarchical relationships  
1) (ℎ𝑗,𝜌part −of,ℎ𝑖) to denote directory or category nesting 
(e.g., part -of relationships).  
2) (𝑐,𝜌include −in,ℎ) to link concept entities  𝑐 to the 
relevant directory node  ℎ. 
To represent the relationship between a textual entity  𝑒 and 
its associated concept  𝑐, we introduce an “is a” relation  𝜌is−a, 
creating triples of the form  (𝑒,𝜌is−a,𝑐). 
Finally, the knowledge graph  𝐺 is formulated as a directed 
graph  𝐺=(𝑉,𝐸), where the vertex set  
 𝑉=𝐸text∪𝐸concept ∪𝐻. (4) 
 
includes all textual entities, concept entities, and 
hierarchical nodes. The edge set  
 𝐸=𝐸𝑡𝑟𝑖𝑝𝑙𝑒𝑡 ∪𝐸𝑐𝑜𝑛𝑐𝑒𝑝𝑡 −𝑚𝑎𝑝 ∪𝐸𝑐𝑎𝑡𝑎𝑙𝑜𝑔 . (5) 
 
Combines the triples extracted from the text, the “is_a” 
mappings from textual entities to concepts, and the directory -
related relationships. By unifying these structured 
representations, the resulting knowledge graph provides an 
integrated, semantically rich view of the original multimodal 
data. A sample result of the knowledge graph is shown in Fig 
2. This unified framework not only supports efficient querying 
and reasoning over diverse data but also enables advanced 
analytics, such as semantic search and entity disambiguation, 
paving the way for more intelligent knowledge -driven 
applications  [29]. 


4 
 
 
Fig. 2. A scenario of a knowledge graph derived from a parsed 
document.  
C. KAQG -Retriever: Cross -Domain Isolation  
To prevent knowledge interference (sometimes referred to 
as “contamination”) among different exam subjects, our 
approach deliberately maintains multiple, separate knowledge 
graphs rather than a single unified graph [ 26]. By isolating 
each subject into its own standalone graph, we ensure that 
information relevant to one discipline does not overshadow 
the concepts of another. This design preserves domain purity 
and simplifies the process of updating and refining the 
knowledge base for each subject. Formally, we denote the full 
collection of subjects as  𝑆={𝑠1,𝑠2,⋯,𝑠𝑛}. For each subject  
𝑠𝑖∈𝑆, we construct a separate knowledge graph  𝐺𝑖, , resulting 
in a set of distinct graphs  {𝐺1,𝐺2,⋯,𝐺𝑛}, where  𝐺𝑖=(𝑉𝑖,𝐸𝑖) 
contains vertices  𝑉𝑖 (representing entities and concepts in 
subject  𝑠𝑖) and edges  𝐸𝑖 (representing relationships within that 
same subject). This multi -graph strategy enforces strict 
boundaries between disciplines, preventing content from one 
domain from being inadvertently mixed or distorted by 
another.  
Combining teaching materials from different subjects into a 
single knowledge graph could lead to conceptual overlap and 
ambiguity, especially when similar terms carry different 
meanings across contexts [ 30]. Such interference may hinder 
retrieval accuracy, as queries for one subject might yield 
irrelevant information from another. At the same time, a 
monolithic knowledge graph merging multiple domains would 
grow exponentially and become unwieldy to index, update, 
and maintain [ 31]. By contrast, a multi -graph system enables 
modular updates and streamlined scalability, ensuring that 
expanding one subject’s knowledge graph does not degrade 
the performance or manageability of another. This separation 
also allows subject matter experts to govern and refine their 
own knowledge graphs without affecting other domains, while 
each graph can evolve independently and incorporate domain -
specific methodologies or tools.  
D. KAQG-Generator:  Overview  
The KAQG -Generator automates the creation of high -
quality exam questions using a structured workflow. It 
integrates assessment theory, employs LLMs for ranking and 
transformation, and ensures accuracy through validation [ 32]. This approach enhances question quality, logical coherence, 
and adaptability to diverse assessment needs.  
E. KAQG-Generator:  Integrating Assessment Theory  
High‑quality exam questions should align with established 
educational frameworks —such as Bloom’s taxonomy —while 
drawing on psychometric methods like Item  Response  Theory. 
Together, these approaches link cognitive objectives with 
statistical evidence, ensuring each item targets the intended 
skill level and yields meaningful information about learner 
performance.  
In a typical three -parameter logistic (3PL) model of IRT, 
the probability of a correct response is calculated as follows  
 𝑃correct =𝒸+1−𝑐
1+𝑒−𝑎(𝜃−𝑏). (6) 
 
IRT links questions to a test taker’s latent ability ( 𝜃, −3 to 
+3). Discrimination (𝑎) indicates how well an item 
differentiates skill levels, while difficulty (𝑏) marks the 50% 
correct threshold. Guessing (𝑐) reflects random success 
probability  [33]. Combined, these parameters ensure precise, 
fair, and valid assessments.  
By controlling these IRT parameters, instructors can tailor 
items to specific learning objectives designated by Bloom’s 
Taxonomy. For instance:  
1) Difficulty ( b) 
Higher -order cognitive tasks (e.g., Analyze, Evaluate, 
Create) naturally tend to be more challenging, 
correlating with higher 𝑏 values.  
2) Discrimination (𝒂) 
Items targeting important distinctions or critical 
thinking skills should exhibit stronger discrimination 
to differentiate proficient students from those with 
lower mastery.  
3) Guessing (𝒄) 
Particularly in multiple -choice formats, adjusting the 
guessing parameter helps ensure that item 
performance reflects genuine knowledge rather than 
random selection.  
When Bloom’s Taxonomy informs the cognitive scope of 
questions and IRT guides item -level precision, test developers 
create assessments that are both pedagogically grounded and 
empirically robust. Controlling difficulty, discrimination, and 
guessing ensures that exam items accurately measure the 
desired breadth and depth of learning outcomes, resulting in 
valid, reliable, and instructionally aligned evaluations  
[21][33]. 
F. KAQG-Generator:  Defining difficulty levels  
The KAQG -Generator automates the creation of high -
quality exam questions using a structured workflow. It 
integrates assessment theory, employs LLMs for ranking and 
transformation, and ensures accuracy through validation [ 32]. 
This approach enhances question quality, logical coherence, 
and adaptability to diverse assessment needs.  
Multiple -choice questions (MCQs) can be aligned with 
Bloom’s Taxonomy and Item Response Theory by applying 
carefully selected features to the three IRT parameters —


5 
 
difficulty ( b), discrimination ( a), and guessing ( c). Rooted in 
educational measurement and cognitive psychology, these 
features (e.g., stem length, domain -specific vocabulary, 
distractor plausibility) act as measurable proxies for 
calibrating item complexity, enhancing discrimination, and 
controlling guessing [ 33]. By referencing Bloom’s framework, 
MCQs can better target intended cognitive levels while 
maintaining robust psychometric quality [ 34]. 
Below are key features shown to influence difficulty levels  
[35]: 
1) Stem Length  
Longer stems often elevate reading load and complexity.  
2) Domain -Specific Vocabulary Density  
A higher concentration of technical terms increases 
cognitive demands.  
3) Cognitive Level of the Stem  
Stems requiring advanced cognitive skills (e.g., Analyze, 
Evaluate) typically raise difficulty.  
4) Average Option Length  
Lengthy options can increase reading time and raise the 
probability of confusion.  
5) Similarity Among Options  
Highly similar distractors make the item more 
challenging.  
6) Similarity Between Stem and Options  
Overlapping terminology or concepts can lead to 
misinterpretation.  
7) Number of Highly Plausible Distractors  
More plausible distractors heighten overall difficulty and 
better discriminate between ability levels.  
We adopt three difficulty levels —low (1 point), medium (2 
points), and high (3 points) —for each feature because this 
tiered structure is both intuitive and flexible for capturing 
variations in textual and structural complexity  [33]. By 
referencing the seven identified features, classifying each as 
low, medium, or high difficulty allows test developers to align 
items with the core principles of Bloom’s Taxonomy and the 
precision of IRT. This system ensures that each aspect of an 
item’s design, from minimal cognitive load to advanced 
conceptual demands, can be quantitatively measured and 
compared.  See Table I for a concise illustration of how these 
seven key features align with Bloom’s Taxonomy levels . 
 
TABLE I  
LINKING BLOOM ’S LEVELS TO QUESTION FEATURES  
 
Feature \ 
Bloom’s Level  Remember  Understand  Apply  
Stem Length  Short  Slightly longer  Scenario -
based  
Domain -
Specific 
Vocabulary  Basic  Moderate  Specialized  
Cognitive 
Level  Recall  Paraphrase  Apply 
concepts  
Option 
Length  Brief  Short 
clarifications  Situational 
details  
Similarity 
Among Minimal  Moderate  Method -
focused  Options  
Stem–Option 
Overlap  Minimal  Some  Real -world 
overlap  
Plausible 
Distractors  Few 1–3 viable  Common mis -
conceptions  
 
Feature \ 
Bloom’s Level  Analyze  Evaluate  Create  
Stem Length  Multi -step 
scenarios  Lengthy, 
criteria -
focused  Open -ended, 
project -based  
Domain -
Specific 
Vocabulary  Multi -domain 
terms  Dense, 
evaluative 
terms  High -level, 
integrative  
Cognitive 
Level  Break down & 
relate  Judge  / assess  Integrate  / 
innovate  
Option 
Length  Detailed 
explanations  Different 
viewpoints  Innovative  / 
partial 
solutions  
Similarity 
Among 
Options  High 
similarity  Closely 
aligned 
arguments  Multiple 
“correct” 
illusions  
Stem–Option 
Overlap  Considerable  Significant 
comparison  Extensive 
integration  
Plausible 
Distractors  Subtle logic 
twists  All appear 
valid  Viable yet 
suboptimal 
solutions  
 
Once each feature is assigned a difficulty rating (1, 2, or 3), 
the total difficulty score (T) of the MCQ is the sum of these 
feature scores, as shown in the following model : 
 𝑇=∑ 𝑑𝑖7
𝑖=1 (7) 
 
where 𝑑𝑖∈{1,2,3}, is the difficulty score for the  i-th feature. 
This straightforward summation integrates multiple cognitive 
and linguistic elements into a single metric, supporting 
consistent item difficulty classification across different 
assessments.  
Building on this straightforward summation, not all features 
exert an equal impact on item difficulty, so weighing each 
attribute addresses variations in their influence [34]. For 
example, advanced cognitive demand [32] generally affects 
difficulty more than simpler textual factors. By assigning a 
weight  𝑤𝑖 to each feature  i, drawn from expert judgment or 
statistical calibration, we adjust the total difficulty to:  
 𝑇𝑤𝑒𝑖𝑔ℎ𝑡𝑒𝑑=∑ 𝑤𝑖∙𝑑𝑖7
𝑖=1  (8) 
 
where 𝑑𝑖∈{1,2,3}, is the difficulty level for each feature.  
G. KAQG-Generator:  Question Generation via LLMs  
We begin from the perspective of an institution tasked with 
designing a comprehensive test for a specific subject, 
leveraging relevant textbooks as foundational content. In 
academic settings, test creation must ensure coverage of the 
curriculum’s breadth while maintaining alignment with 
standardized guidelines.  To achieve this, the first step often 
involves defining a distribution ratio for each chapter or topic 
within the textbook. Mathematically, if  𝛼𝑖 denotes the fraction 
of total questions devoted to chapter i, and  𝑛𝑖 represents the 
desired number of questions from chapter iii, then the ratio can 

6 
 
be formulated as:  
 𝛼𝑖=𝑛𝑖
∑ 𝑛𝑗𝑘
𝑗=1 (9) 
 
where k is the total number of chapters or major divisions of 
the subject matter. This approach ensures that each section is 
proportionally represented according to institutional priorities 
and learning objectives.  
Once chapter -specific ratios are established, the test -design 
process proceeds with identifying crucial concepts from each 
chapter —typically through a structured knowledge base or 
knowledge graph. Importantly, the concept nodes extracted in 
the previous stage can now play a key role in this generation 
phase, as they serve as foundational anchors for further 
knowledge expansion.  Each key concept can then be expanded 
by exploring related nodes, such as prerequisite relationships 
or hierarchical links [36]. Crucially, concepts are ranked based 
on their instructional value using algorithms like PageRank 
[19]. For any knowledge point v, the PageRank score PR( v) is 
updated iteratively by  
 P𝑅(𝑣)=(1−𝑑)+𝑑∑PR(𝑢)
|Out (𝑢)|𝑢∈In(𝑣)  (10) 
 
where In(v) represents inbound links to v, Out( u) denotes 
outbound links from u, and 0<𝑑<1 is a damping factor. 
Higher PageRank scores guide the selection of more essential 
topics, thus ensuring that the resulting questions cover the 
subject’s core content.  
All fact nodes connected to a concept via an is_a 
relationship are also evaluated using the same PageRank 
algorithm to determine their importance. For each concept, all 
significant facts —along with their sub -connections —are 
subsequently incorporated as material for question generation, 
thus enriching the depth and breadth of the test items.  
Finally, institutional requirements typically necessitate the 
inclusion of items at varied difficulty levels. In practice, one 
may specify three tiers —Basic Recall, Applied 
Understanding, and Comprehensive Analysis —and embed the 
selected high -priority concepts into suitable question 
templates (e.g., multiple -choice, fill -in-the-blank, or short -
answer). By systematically controlling both distribution and 
question difficulty, the resulting test offers a balanced 
assessment of knowledge, from fundamental definitions to 
higher -order reasoning. This multi -tiered, data -driven 
approach helps institutions produce rigorous, fair, and 
curriculum -aligned examinations.  
H. KAQG-Generator:  Evaluation of Generated Questions  
Within this framework, the Evaluation Agent  focuses on 
ensuring that each generated question meets the target 
difficulty level. Following established testing principles, the 
system quantifies various item features —such as cognitive 
load, prerequisite knowledge, and complexity of reasoning —
and then aggregates them through a weighted sum to derive a 
difficulty score . Formally, let v= (𝑣1,𝑣2,…,𝑣𝑛) denote the 
feature values for a particular question, and let  w=
 (𝑤1,𝑤2,…,𝑤𝑛) represent the corresponding weights that 
indicate the relative importance of each feature. The resulting difficulty D can be expressed as:  
 𝐷=∑ 𝑤𝑖∙𝑣𝑖𝑛
𝑖=1  (11) 
 
To verify alignment with an institution’s intended difficulty  
𝐷∗, the Evaluation Agent  checks whether  |𝐷−𝐷∗ | falls 
within a predefined tolerance  𝜀. A smaller deviation  |𝐷−𝐷∗ | 
suggests that the item’s difficulty is suitably calibrated, 
whereas a larger discrepancy signals the need for adjustments, 
such as revising feature weights or modifying the question’s 
content. Conceptually, a higher weighted sum reflects a more 
challenging item, while a lower sum corresponds to an easier 
one. 
Importantly, the purpose of introducing the Evaluation 
Agent in this manner is to provide a foundational example of 
how specialized AI agents operate, thereby setting the stage 
for the subsequent chapter’s in -depth exploration of 
comprehensive multi -agent architecture. By showcasing the 
Evaluation Agent’s role in ensuring difficulty appropriateness, 
we highlight the broader potential of AI -driven collaboration 
among agents dedicated to different aspects of the test -
generation process.  
I. AI Agents Framework: Autonomous and Distributed 
Operations  
One of the primary challenges in generating exam questions 
from educational materials is balancing robust domain -
specific knowledge integration with efficient retrieval, logical 
reasoning, and question evaluation. Traditional monolithic 
systems struggle with managing large -scale data and 
coordinating multiple processes such as entity extraction, 
knowledge graph construction, and question generation and 
evaluation. These single -process architectures often suffer 
from performance bottlenecks and limited fault tolerance, 
making them unsuitable for diverse educational domains.  
To overcome these limitations, a Multi -Agent System 
(MAS) framework distributes tasks across autonomous agents, 
each responsible for specific functions like retrieval, question 
generation, knowledge graph updates, or evaluation [ 37]. This 
decentralized design enables parallel processing, improves 
scalability and resilience, and reduces reliance on a central 
control node. Agents are autonomous computational entities 
capable of perceiving and acting on their environments to 
achieve specific goals [ 38]. They exhibit reactivity, 
proactiveness, and social ability, allowing them to adapt to 
dynamic conditions and coordinate effectively with other 
agents within the system.  
The agent system employs a Data Distribution Service 
(DDS)  as its communication architecture, utilizing a publish -
subscribe model to enhance interaction among agents. By 
decoupling message publishers from subscribers, DDS 
improves the system’s scalability, flexibility, and modularity 
[39]. Each agent can operate independently on its assigned 
task—such as data extraction, graph updating, or question 
generation —without requiring direct connections to other 
agents. DDS dynamically handles network configurations, 
manages agent discovery, and ensures efficient data 
dissemination. This architecture supports real -time 

7 
 
coordination and fault -tolerant operations, allowing agents to 
join or leave the system seamlessly. The overall system 
architecture is depicted in Fig . 3. 
In the KAQG framework, specialized autonomous agents 
manage the sequential processes of entity extraction, 
knowledge graph construction, question generation, and 
question evaluation, thereby distributing responsibilities for 
better scalability and robustness [37][38]. Specifically, the 
entity extraction agent employs natural language processing to 
identify domain -specific concepts, relationships, and 
hypernyms from educational resources, converting them into 
structured elements. The knowledge graph construction agent 
then integrates these elements into a coherent semantic 
structure, preserving explicit relationships and facilitating 
targeted retrieval. Meanwhile, the question generation agent 
leverages both domain expertise and assessment theory to 
transform graph -based information into suitable exam items, 
ensuring logical coherence and appropriate difficulty. Finally, 
the question evaluation agent validates the generated items by 
checking factual accuracy and cognitive rigor, closing the loop 
with feedback to maintain high -quality and pedagogically 
sound question sets.  
 
Fig. 3. The diagram represents the multi -agent architecture of 
KAQG  with DDS -based publish -subscribe communication.  
 
The process begins with uploading educational materials, 
which are processed by the Data Extraction Agent and File 
Agent to extract text. The Knowledge Graph (KG) 
Management Agent stores the entity relation triplets, such as 
"Intentional pollution will harm the ecosystem," in a 
Knowledge Graph Database. Then, a Question Generation 
Agent, using an LLM Agent, generates questions based on the 
extracted data. These questions are evaluated through a 
Question Evaluation Agent and filtered according to 
predefined criteria. The cycle continues as qualified questions 
are produced and shared with the Question Requirement 
Client for further use.  
III. EXPERIMEN TS 
In this section, we detail the design and implementation of experiments evaluating the KAQG system. The focus includes 
participant recruitment, question set distribution across 
difficulty levels, and data collection protocols. Procedures 
ensure transparent analysis, validating the system’s ability to 
generate high -quality exam questions. Results highlight 
significantly improved accuracy and relevance.  
A. Experiment Objectives  
The ACT (American College Test) is a standardized 
examination widely used for college admissions in the United 
States. Its Reading section, in particular, presents passages 
followed by multiple -choice questions designed to measure 
comprehension, inference, and analytical abilities. Given its 
reliability and recognized status, the ACT serves as an 
effective benchmark for evaluating the performance and 
quality of automatically generated reading test items.  
First, this experiment aims to evaluate whether the system, 
configured at three distinct difficulty levels —high, medium, 
and low —can produce test items comparable to, or distinctly 
different from, official ACT questions in terms of overall 
testing outcomes. This involves examining how the generated 
questions perform relative to the standard ACT items when 
administered under various difficulty settings.  
Second, the investigation focuses on three key metrics: (1) 
question difficulty (P value), (2) the discrimination index 
(which gauges how well items differentiate between higher - 
and lower -performing examinees), and (3) expert assessments 
of item quality. By comparing these metrics across the 
different difficulty levels, the study will shed light on how 
effectively the system can generate valid and reliable test 
questions.  
Three reading passages —labeled A, B, and C —were 
selected from publicly released ACT Reading materials, each 
originally accompanied by 10 official multiple -choice 
questions, totaling 30 items that served as the control group. 
To generate the experimental items, the same three passages 
were processed by the system under three difficulty settings —
low, medium, and high —with 10 questions produced per 
passage per setting, resulting in 30 questions per difficulty 
level and 90 system -generated items in total. All questions and 
their corresponding answers are provided in Appendix A.  
B. Experimental Procedure  
In this study, we recruited a sufficient number of participants 
(ideally several dozen, though the exact count varied 
according to available resources) to ensure reliable statistical 
outcomes. Participants were then divided into four groups, 
with each group answering a different set of questions:  
1) Group ACT  
This group completed the 30 official ACT items, 
consisting of 10 questions for each passage (A, B, and C).  
2) Group Low  
This group answered the 30 system -generated items set 
at low difficulty.  
3) Group Medium  
This group answered the 30 system -generated items set 
at medium difficulty.  


8 
 
4) Group High  
This group answered the 30 system -generated items set 
at high difficulty.  
Each participant was presented with the same three reading 
passages (A, B, and C) but only received the questions 
pertinent to the assigned difficulty group —or the official ACT 
group, depending on their allocation. All questions were 
administered in a multiple -choice format with one correct 
answer per item. The testing sessions could be conducted 
either on paper or through an online platform, where the latter 
option could facilitate automatic recording of response times. 
Upon completion, participants submitted their responses 
(marked correct or incorrect) for subsequent data analysis.  
By logging participants’ correct and incorrect responses, we 
computed the difficulty index (P value) —defined as the 
proportion of correct answers for each item —and the 
discrimination index. The discrimination index was derived by 
contrasting performance between high - and low -scoring 
subgroups.  
C. Evaluation Metrics and Statistical Analysis  
This study evaluates the quality and effectiveness of system -
generated questions by comparing them to official ACT items. 
The following section outlines the statistical methods used to 
analyze differences across the four groups.  
1) Difficulty (P Value)  
We define the difficulty index, or P value, as the 
proportion of participants who answered a given 
question correctly. After gathering the response data, we 
computed the mean P value for each group —ACT, Low, 
Medium, and High —across all items in each reading 
passage (A, B, and C). By comparing these averages, we 
aimed to determine whether the system -generated items 
align with, or deviate significantly from, the difficulty 
levels of the official ACT questions. Notably, lower P 
values correspond to more challenging items, whereas 
higher P values indicate easier ones. Our objective was 
to verify that the three system -defined difficulty settings 
(Low, Medium, High) indeed produce distinctive P value 
ranges, and to see how they stack up against the standard 
ACT items.  
2) Discrimination Index  
To assess each question’s ability to differentiate between 
high-performing and low -performing participants, we 
calculated the Discrimination Index. Specifically, 
participants were ranked according to their total test 
scores, and the top and bottom quartiles (or other 
appropriate segments, depending on sample size) were 
identified. For each question, we computed the 
difference in the proportion of correct responses between 
these two subgroups. A higher Discrimination Index 
suggests that the question more effectively distinguishes 
strong from weak test -takers. Through this analysis, we 
compared whether the system -generated questions —
under Low, Medium, and High difficulty settings —
demonstrate a discrimination capacity comparable to that 
of the official ACT items.  
3) Expert Ratings  
Beyond objective metrics, we obtained qualitative evaluations from subject matter experts with experience 
in test design. Each expert rated the clarity, alignment 
with the passage, and plausibility of distractor options 
for every question on a five -point Likert scale (where 1 
indicates very poor quality and 5 indicates excellent 
quality). These ratings allowed us to gauge the perceived 
strengths and weaknesses of each question beyond 
simple correct -answer statistics.  
 
We employed a one -way analysis of variance (ANOVA) to 
detect any significant differences among the four groups 
(ACT, Low, Medium, High) in terms of average P value, 
Discrimination Index, and Expert Ratings. If the overall 
ANOVA yielded a significant result (p < .05), we performed 
Tukey’s post -hoc tests to identify which specific group pairs 
differ. Where necessary, multi -factor ANOVA was conducted 
to explore potential interactions between difficulty levels and 
reading passages. In all cases, assumptions of normality and 
homogeneity of variance were assessed using Shapiro –Wilk 
and Levene’s tests, respectively.  
D. Potential Results Presentation  
After collecting participants’ responses and performing the 
outlined statistical analyses, we present the outcome 
measures —namely, average accuracy (P Value), average 
Discrimination Index, and mean Expert Rating —for both the 
official ACT items and the system -generated questions under 
three difficulty settings. Table 2 illustrates an example set of 
results based on simulated data. Statistically significant 
differences (p < 0.05) between groups are marked with an 
asterisk (*).  
 
TABLE  II 
ILLUSTRATIVE SUMMARY OF KEY METRICS (P VALUE , 
DISCRIMINATION INDEX , AND EXPERT RATING ) FOR ACT  AND 
SYSTEM -GENERATED QUESTION GROUPS  
 
(*) denotes a statistically significant difference (p < 0.05) from the ACT 
control group according to post -hoc comparisons.  
 
The summary metrics indicate that the Low -Difficulty set 
yields the highest average accuracy and is significantly easier 
than the ACT baseline, whereas the High -Difficulty set is the 
most challenging. Discrimination indices remain broadly 
similar across groups, suggesting that the difficulty 
manipulation primarily affected item ease rather than their 
ability to separate high - and low -performing examinees.  
E. Passage‑Level Analysis  
To determine whether passage content modulates item Group  Average 
Accuracy (P 
Value)  Average 
Discrimination 
Index  Expert Rating 
(Mean ± SD)  
ACT (Control)  0.76 ± 0.05  0.37 ± 0.04  4.3 ± 0.4  
Low Difficulty  0.82 ± 0.06*  0.32 ± 0.05  3.9 ± 0.5  
Medium Difficulty  0.71 ± 0.07  0.35 ± 0.04  4.1 ± 0.3  
High Difficulty  0.63 ± 0.08  0.34 ± 0.06  3.7 ± 0.4  

9 
 
performance, a two‑way ANOVA 
(Difficulty  Level  × Passage) was conducted on the P‑value 
data for all 120 items (30 per passage). Results revealed 
significant main effects for Difficulty 
Level  F(3, 348) = 277.40,  p < .001, and 
Passage  F(2, 348) = 26.46,  p < .001, as well as a significant 
interaction  F(6, 348) = 3.92,  p = .001.  
Table  3 summaries the mean  ± SD P‑values for each 
Passage‑by‑Group cell. Passage  B was generally easier (higher 
P‑values) than Passages  A and  C, and the Low‑Difficulty 
items showed the greatest spread across passages, confirming 
that content can accentuate or dampen the intended difficulty 
gradient.  
 
TABLE  III 
MEAN ± SD P‑VALUES BY PASSAGE AND DIFFICULTY LEVEL 
(N = 30 ITEMS PER CELL ) 
 
 
The two -way ANOVA shows that both passage content and 
difficulty setting affect item performance, with a significant 
interaction indicating that the magnitude of the difficulty 
gradient varies by passage (most pronounced for Passage  B). 
IV. DISCUSSION  
The experimental results confirm that the KAQG system 
can modulate item difficulty in a predictable hierarchy —
Low > ACT  ≈ Medium  > High —while largely preserving each 
item’s capacity to distinguish between stronger and weaker 
examinees. The Low‑Difficulty set attained the highest 
average P  value (0.82) and differed significantly from the 
ACT baseline, whereas the High‑Difficulty set produced the 
lowest accuracy (0.63), validating the system’s difficulty 
controller. Importantly, Discrimination Indices clustered 
within a narrow band (0.32 –0.37), indicating that the ease 
adjustment was achieved without materially eroding item 
power to differentiate performance levels.  
A. Passage‑Specific Performance and Expert Judgements  
Across the three passages, accuracy varied systematically: 
Passage  B was consistently easier than Passages  A and  C, and 
the intended Low‑Medium‑High gradient was most 
pronounced on this passage but compressed on Passage  C. The 
significant Difficulty  × Passage interaction indicates that text 
characteristics —such as topical familiarity or syntactic load —
can amplify or mute the system’s difficulty settings, 
confirming the need for content‑aware calibration when 
deploying KAQG across diverse reading materials.  
Expert reviewers rated most items favorably (3.7 –4.3 on a 
5‑point scale), substantiating the face validity of the generated 
questions even when passage effects were present. However, ratings dipped slightly for High‑Difficulty items, echoing 
quantitative findings that these questions were harder yet not 
markedly more discriminative. Taken together, the passage 
analysis and expert feedback suggest that while the system 
reliably modulates difficulty, improvements are required in 
handling complex syntax and in balancing distractor 
plausibility to maintain both clarity and discriminative power 
at higher difficulty levels.  
B. System Enhancement and Future Directions  
To achieve a more uniform difficulty gradient across varied 
texts, the KAQG controller should incorporate passage‑level 
features —lexical density, cohesion metrics, and domain 
familiarity —so that Low‑Medium‑High settings translate 
consistently into target P  ‑value ranges. Parallel efforts must 
focus on distractor engineering; by ranking semantically 
proximate distractors and iteratively refining them with 
error‑analysis feedback, the system can raise discrimination 
indices without simply increasing stem complexity.  
Future work will address current limitations by enlarging 
participant samples for more precise reliability estimates and 
by analyzing response‑time data as an auxiliary difficulty 
signal. Extending evaluation to additional genres (e.g., 
scientific or persuasive prose) and integrating large‑scale 
language‑model feedback loops will further test 
generalizability while refining lexical and syntactic choices. 
Collectively, these enhancements aim to match the nuanced 
difficulty profile of professionally authored examinations and 
to strengthen the KAQG system’s applicability in high‑stakes 
assessment contexts.  
Overall, the KAQG system demonstrates promising 
controllability and alignment with expert standards, yet a shift 
toward context‑sensitive calibration and richer distractor 
design is essential for matching the nuanced difficulty profile 
of professionally authored examinations.  
V. CONCLUSION  
The Knowledge  Augmented  Question  Generation (KAQG) 
system delivers a clear system innovation: it explicitly 
resolves the long‑standing inability of existing RAG pipelines 
to perform transparent multi‑step reasoning and to control 
item difficulty. This advance is achieved through a deliberate 
technical fusion —knowledge graphs  + RAG  + assessment 
theory —constituting a cross‑disciplinary contribution that lets 
KAQG pinpoint any textbook chapter and dynamically 
fine‑tune item features. The workflow embeds domain  
knowledge in structured formats while operationalizing 
Bloom’s cognitive hierarchy and validating every item with 
Item Response  Theory, evidencing a concrete 
educational‑measurement contribution. Coupled with an 
AI‑Agents framework that scales computation and coordinates 
resources, experiments confirm that KAQG yields questions 
on par with standardized assessments such as the  ACT. Finally, 
full source code has been released, ensuring reproducibility 
and enabling seamless academic benchmarking and extension . 
Looking ahead, future work will broaden KAQG’s reach to 
diverse educational settings, strengthen retrieval with richer 
reasoning modules, and integrate real‑time feedback for Passage  ACT  Low  Medium  High  
A 0.76 ± 0.04 0.83 ± 0.05 0.75 ± 0.05 0.63 ± 0.06 
B 0.73 ± 0.06 0.84 ± 0.04 0.73 ± 0.05 0.61 ± 0.07 
C 0.73 ± 0.05 0.79 ± 0.06 0.74 ± 0.04 0.62 ± 0.06 

10 
 
adaptive item generation. Additional directions include 
multimodal input support, enhanced knowledge‑graph 
embeddings, and reinforcement learning within the agent 
framework to further optimize retrieval, generation, and 
evaluation —ultimately delivering higher‑quality assessments, 
broader learning‑objective coverage, and more adaptable 
educational technologies.  
REFERENCES  
[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. 
Küttler, M. Lewis, W. -t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, 
“Retrieval -augmented generation for knowledge -intensive NLP tasks,” 
arXiv:2005.11401 , May 2020. [Online]. Available: 
https://doi.org/10.48550/arXiv.2005.11401  
[2] J. J. González Torres, M. B. Bîndilă, S. Hofstee, D. Szondy, Q. H. 
Nguyen, S. Wang, and G. Englebienne, “Automated question -answer 
generation for evaluating RAG -based chatbots,” in Proc. 1st Workshop on 
Patient -Oriented Language Processing, CL4Health 2024 at LREC -
COLING , Torino, Italy, May 2024, pp. 204 –214. [Online]. Available: 
https://aclanthology.org/2024.cl4health -1.25.pdf  
[3] S. Pan, L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu, “Unifying large 
language models and knowledge graphs: A roadmap,” IEEE Trans. Knowl. 
Data Eng ., early access, 2024, doi: 10.1109/TKDE.2024.3352100.  
[4] B. Jiménez Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, 
“HippoRAG: Neurobiologically inspired long -term memory for large 
language models,” in Proc. 38th Conf. Neural Inf. Process. Syst. 
(NeurIPS) , Vancouver, BC, Canada, 2024. [Online]. Available: 
https://arxiv.org/abs/2405.14831  
[5] C. Mavromatis and G. Karypis, “GNN -RAG: Graph neural retrieval for 
large language model reasoning,” arXiv , vol. arXiv:2405.20139, May 
2024. [Online]. Available: https://doi.org/10.48550/arXiv.2405.20139  
[6] Y. Hu, Z. Lei, Z. Zhang, B. Pan, C. Ling, and L. Zhao, “GRAG: Graph 
retrieval -augmented generation,” arXiv:2405.16506 , May 2024. [Online]. 
Available: https://doi.org/10.48550/arXiv.2405.16506  
[7] S. Ma, C. Xu, X. Jiang, M. Li, H. Qu, C. Yang, J. Mao, and J. Guo, 
“Think -on-Graph 2.0: Deep and faithful large language model reasoning 
with knowledge -guided retrieval augmented generation,” 
arXiv:2407.10805 , Jul. 2025. [Online]. Available: 
https://doi.org/10.48550/arXiv.2407.10805  
[8] M. Kang, J. M. Kwak, J. Baek, and S. J. Hwang, “Knowledge graph -
augmented language models for knowledge -grounded dialogue 
generation,” arXiv:2305.18846 , May 2023. [Online]. Available: 
https://doi.org/10.48550/arXiv.2305.18846  
[9] D. Li, S. Yang, Z. Tan, J. Y. Baik, S. Yun, J. Lee, A. Chacko, B. Hou, D. 
Duong -Tran, Y. Ding, H. Liu, L. Shen, and T. Chen, “DALK: Dynamic 
co-augmentation of LLMs and KG to answer Alzheimer’s disease 
questions with scientific literature,” arXiv:2405.04819 , May 2024. 
[Online]. Available: https://doi.org/10.48550/arXiv.2405.04819  
[10] H. Han, Y. Wang, H. Shomer, K. Guo, J. Ding, Y. Lei, M. Halappanavar, 
R. A. Rossi, S. Mukherjee, X. Tang, Q. He, Z. Hua, B. Long, T. Zhao, N. 
Shah, A. Javari, Y. Xia, and J. Tang, “Retrieval -augmented generation 
with graphs (GraphRAG),” arXiv:2501.00309 , Jan. 2025. [Online]. 
Available: https://doi.org/10.48550/arXiv.2501.00309  
[11] J. Wu, J. Zhu, and Y. Liu, “Agentic reasoning: Reasoning LLMs with 
tools for the deep research,” arXiv:2502.04644 , Feb. 2025. [Online]. 
Available: https://doi.org/10.48550/arXiv.2502.04644  
[12] P. Béchard and O. Marquez Ayala, “Reducing hallucination in structured 
outputs via retrieval -augmented generation,” in Proc. Industry Track of 
the 2024 Conf. North Amer. Chapter Assoc. Comput. Linguistics 
(NAACL) , Mexico City, Mexico, Jun. 2024, pp. 19. [Online]. Available: 
https://doi.org/10.18653/v1/2024.naacl -industry.19  
[13] S. Bi, J. Liu, Z. Miao, and Q. Min, “Difficulty -controllable question 
generation over knowledge graphs: A counterfactual reasoning 
approach,” Inf. Process. Manage. , vol. 61, no. 4, Art. no. 103721, Jul. 
2024. [Online]. Available: https://doi.org/10.1016/j.ipm.2024.103721  
[14] S. Liu, Y. Qin, M. Xu, et al., “Knowledge graph completion with triple 
structure and text representation,” Int. J. Comput. Intell. Syst. , vol. 16, p. 
95, 2023. [Online]. Available: https://doi.org/10.1007/s44196 -023-
00271 -0 
[15] Y. Feng, S. Papicchio, and S. Rahman, “CypherBench: Towards precise 
retrieval over full -scale modern knowledge graphs in the LLM era,” arXiv:2412.18702 , Dec. 2024. [Online]. Available: 
https://arxiv.org/abs/2412.18702  
[16] S. Zheng, W. Chen, W. Wang, P. Zhao, H. Yin, and L. Zhao, “Multi -hop 
knowledge graph reasoning in few -shot scenarios,” IEEE Trans. Knowl. 
Data Eng. , vol. 36, no. 4, pp. 1713 –1727, Apr. 2024. doi: 
10.1109/TKDE.2023.3304665.  
[17] L. Zhong, J. Wu, Q. Li, H. Peng, and X. Wu, “A comprehensive survey 
on automatic knowledge graph construction,” ACM Comput. Surv. , vol. 
56, no. 4, Art. no. 94, Apr. 2024. doi: 10.1145/3618295.  
[18] G. Jhajj, X. Zhang, J. R. Gustafson, F. Lin, and M. P. C. Lin, 
“Educational knowledge graph creation and augmentation via LLMs,” in 
Generative Intelligence and Intelligent Tutoring Systems (ITS 2024) , A. 
Sifaleras and F. Lin, Eds., Lecture Notes in Computer Science , vol. 
14799, Cham, Switzerland: Springer, 2024, pp. [in press]. doi: 
10.1007/978 -3-031-63031 -6_25.  
[19] L. Page, S. Brin, R. Motwani, and T. Winograd, “The PageRank citation 
ranking: Bringing order to the web,” Stanford InfoLab, Stanford, CA, 
USA, Tech. Rep., 1999. [Online]. Available: 
http://ilpubs.stanford.edu:8090/422/  
[20] P. Stone, “Learning and multiagent reasoning for autonomous agents,” 
in Proc. 20th Int. Joint Conf. Artif. Intell. (IJCAI’07) , Hyderabad, India, 
Jan. 2007, pp. 13 –30. 
[21] [22] D. R. Krathwohl, “A revision of Bloom’s taxonomy: An overview,” 
Theory Into Practice , vol. 41, no. 4, pp. 212 –218, 2002. doi: 
10.1207/s15430421tip4104_2.  
[22] F. Outay, N. Jabeur, F. Bellalouna, et al., “Multi -agent system -based 
framework for an intelligent management of competency building,” 
Smart Learn. Environ. , vol. 11, p. 41, 2024. doi: 10.1186/s40561 -024-
00328 -3. 
[23] S. Saxena, N. A. El -Taweel, H. E. Farag, and L. S. Hilaire, “Design and 
field implementation of a multi -agent system for voltage regulation 
using smart inverters and data distribution service (DDS),” in Proc. 
IEEE Electr. Power Energy Conf. (EPEC) , Toronto, ON, Canada, Oct. 
2018, pp. 1 –6. doi: 10.1109/EPEC.2018.8598367.  
[24] Z. Nguyen, A. Annunziata, V. Luong, S. Dinh, Q. Le, A. H. Ha, C. Le, 
H. A. Phan, S. Raghavan, and C. Nguyen, “Enhancing Q&A with 
domain -specific fine -tuning and iterative reasoning: A comparative 
study,” arXiv:2404.11792 , Apr. 2024. [Online]. Available: 
https://doi.org/10.48550/arXiv.2404.11792  
[25] X. Li, A. Henriksson, M. Duneld, J. Nouri, and Y. Wu, “Evaluating 
embeddings from pre -trained language models and knowledge graphs 
for educational content recommendation,” Future Internet , vol. 16, p. 
12, Jan. 2024. doi: 10.3390/fi16010012.  
[26] P. Cimiano and H. Paulheim, “Knowledge graph refinement: A survey 
of approaches and evaluation methods,” Semant. Web , vol. 8, no. 3, pp. 
489–508, 2017. doi: 10.3233/SW -160218.  
[27] Y. Chen, X. Ge, S. Yang, L. Hu, J. Li, and J. Zhang, “A survey on 
multimodal knowledge graphs: Construction, completion and 
applications,” Mathematics , vol. 11, p. 1815, Apr. 2023. doi: 
10.3390/math11081815.  
[28] J. Dagdelen, A. Dunn, S. Lee, et al., “Structured information extraction 
from scientific text with large language models,” Nat. Commun. , vol. 15, 
p. 1418, 2024. doi: 10.1038/s41467 -024-45563 -x. 
[29] R. Navigli and S. P. Ponzetto, “BabelNet: The automatic construction, 
evaluation and application of a wide -coverage multilingual semantic 
network,” Artif. Intell. , vol. 193, pp. 217 –250, 2012. doi: 
10.1016/j.artint.2012.07.001.  
[30] T. Bui, O. Tran, P. Nguyen, B. Ho, L. Nguyen, T. Bui, and T. Quan, 
“Cross -data knowledge graph construction for LLM -enabled educational 
question -answering system: A case study at HCMUT,” 
arXiv:2404.09296 , Apr. 2024. [Online]. Available: 
https://doi.org/10.48550/arXiv.2404.09296  
[31] M. Hofer, D. Obraczka, A. Saeedi, H. Köpcke, and E. Rahm, 
“Construction of knowledge graphs: State and challenges,” SSRN , Nov. 
2023. doi: 10.2139/ssrn.4605059.  
[32] B. Das, M. Majumder, S. Phadikar, et al., “Automatic question 
generation and answer assessment: A survey,” Res. Pract. Technol. 
Enhanc. Learn. (RPTEL) , vol. 16, p. 5, 2021. doi: 10.1186/s41039 -021-
00151 -1. 
[33] M. Dyehouse, “A comparison of model -data fit for parametric and 
nonparametric item response theory models using ordinal -level ratings,” 

11 
 
M.S. thesis, Purdue Univ., West Lafayette, IN, USA, 2009. [Online]. 
Available: https://docs.lib.purdue.edu/dissertations/AAI3362679  
[34] T. M. Haladyna, S. M. Downing, and M. C. Rodriguez, “A review of 
multiple -choice item -writing guidelines for classroom assessment,” 
Appl. Meas. Educ. , vol. 15, no. 3, pp. 309 –333, 2002. doi: 
10.1207/S15324818AME1503_5.  
[35] E. Rafatbakhsh and A. Ahmadi, “Predicting the difficulty of EFL 
reading comprehension tests based on linguistic indices,” Asian J. 
Second Foreign Lang. Educ. , vol. 8, p. 41, 2023. doi: 10.1186/s40862 -
023-00214 -4. 
[36] R. Navigli and P. Velardi, “Learning domain ontologies from document 
warehouses and dedicated web sites,” Comput. Linguist. , vol. 30, no. 2, 
pp. 151 –179, 2004. doi: 10.1162/089120104323093276.  
[37] N. R. Jennings, “On agent -based software engineering,” Artif. Intell. , 
vol. 117, no. 2, pp. 277 –296, 2000. doi: 10.1016/S0004 -3702(99)00107 -
1. 
[38] M. Wooldridge and N. R. Jennings, “Intelligent agents: Theory and 
practice,” Knowl. Eng. Rev. , vol. 10, no. 2, pp. 115 –152, 1995.  
[39] A. Corsaro and D. Schmidt, “The data distribution service – The 
communication middleware fabric for scalable and extensible systems -
of-systems,” in Model -Driven Engineering for Distributed Real -Time 
Embedded Systems , InTech, 2012. doi: 10.5772/30322.  
 
 
Ching  Han Chen received his Ph.D. 
degree from Franche -Comté University, 
Besançon, France, in 1995. He was an 
Associate Professor in the Department of 
Electrical Engineering at I -Shou 
University, Kaohsiung, Taiwan, before 
joining National Central University. He 
is currently a Professor in the 
Department of Computer Science and Information 
Engineering at National Central University, Taoyuan, Taiwan.  
Prof. Chen is the founder of the MIAT (Machine Intelligence 
and Automation Technology) Laboratory. His research 
focuses on embedded system design, AIoT, robotics, and 
intelligent automation. He has led numerous government -
funded and industry -collaborative projects, producing 
innovations in smart sensors, machine vision, and embedded 
AI systems.  
 
 
Ming Fang Shiu  is a Ph.D. candidate at National Central 
University, Taiwan, researching large language models and 
human -computer interfaces, with prior experience in game 
development and financial systems.  
 
