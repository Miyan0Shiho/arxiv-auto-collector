# MuaLLM: A Multimodal Large Language Model Agent for Circuit Design Assistance with Hybrid Contextual Retrieval-Augmented Generation

**Authors**: Pravallika Abbineni, Saoud Aldowaish, Colin Liechty, Soroosh Noorzad, Ali Ghazizadeh, Morteza Fayazi

**Published**: 2025-08-11 16:11:09

**PDF URL**: [http://arxiv.org/pdf/2508.08137v1](http://arxiv.org/pdf/2508.08137v1)

## Abstract
Conducting a comprehensive literature review is crucial for advancing circuit
design methodologies. However, the rapid influx of state-of-the-art research,
inconsistent data representation, and the complexity of optimizing circuit
design objectives make this task significantly challenging. In this paper, we
propose MuaLLM, an open-source multimodal Large Language Model (LLM) agent for
circuit design assistance that integrates a hybrid Retrieval-Augmented
Generation (RAG) framework with an adaptive vector database of circuit design
research papers. Unlike conventional LLMs, the MuaLLM agent employs a Reason +
Act (ReAct) workflow for iterative reasoning, goal-setting, and multi-step
information retrieval. It functions as a question-answering design assistant,
capable of interpreting complex queries and providing reasoned responses
grounded in circuit literature. Its multimodal capabilities enable processing
of both textual and visual data, facilitating more efficient and comprehensive
analysis. The system dynamically adapts using intelligent search tools,
automated document retrieval from the internet, and real-time database updates.
Unlike conventional approaches constrained by model context limits, MuaLLM
decouples retrieval from inference, enabling scalable reasoning over
arbitrarily large corpora. At the maximum context length supported by standard
LLMs, MuaLLM remains up to 10x less costly and 1.6x faster while maintaining
the same accuracy. This allows rapid, no-human-in-the-loop database generation,
overcoming the bottleneck of simulation-based dataset creation for circuits. To
evaluate MuaLLM, we introduce two custom datasets: RAG-250, targeting retrieval
and citation performance, and Reasoning-100 (Reas-100), focused on multistep
reasoning in circuit design. MuaLLM achieves 90.1% recall on RAG-250, and 86.8%
accuracy on Reas-100.

## Full Text


<!-- PDF content starts -->

MuaLLM: A Mu ltimoda l Large L anguage M odel
Agent for Circuit Design Assistance with Hybrid
Contextual Retrieval-Augmented Generation
Pravallika Abbineni∗, Saoud Aldowaish∗, Colin Liechty∗, Soroosh Noorzad∗, Ali Ghazizadeh†, Morteza Fayazi∗
Abstract —Conducting a comprehensive literature review is
crucial for advancing circuit design methodologies. However,
the rapid influx of state-of-the-art research, inconsistent data
representation, and the complexity of optimizing circuit design
objectives ( e.g.power consumption) make this task significantly
challenging. Traditional manual search methods are inefficient,
time-consuming, and lack the reasoning capabilities required
for synthesizing complex circuits. In this paper, we propose
MuaLLM, an open-source multimodal Large Language Model
(LLM) agent for circuit design assistance that integrates a
hybrid Retrieval-Augmented Generation (RAG) framework with
an adaptive vector database of circuit design research papers.
Unlike conventional LLMs, the MuaLLM agent employs a
Reason + Act (ReAct) workflow for iterative reasoning, goal-
setting, and multi-step information retrieval. It functions as
a question-answering design assistant, capable of interpreting
complex queries and providing reasoned responses grounded in
circuit literature. Its multimodal capabilities enable processing
of both textual and visual data, facilitating more efficient and
comprehensive analysis. The system dynamically adapts using
intelligent search tools, automated document retrieval from the
internet, and real-time database updates. Unlike conventional
approaches constrained by model context limits, MuaLLM de-
couples retrieval from inference, enabling scalable reasoning
over arbitrarily large corpora. At the maximum context length
supported by standard LLMs, MuaLLM remains up to 10x less
costly and 1.6x faster while maintaining the same accuracy.
This allows rapid, no-human-in-the-loop database generation,
overcoming the bottleneck of simulation-based dataset creation
for circuits. To evaluate MuaLLM, we introduce two custom
datasets: RAG-250, targeting retrieval and citation performance,
and Reasoning-100 (Reas-100), focused on multistep reasoning
in circuit design. MuaLLM achieves 90.1% recall on RAG-250,
highlighting strong multimodal retrieval and citation accuracy.
On Reas-100, it reaches 86.8% accuracy, demonstrating robust
reasoning capabilities on complex design queries.
Index Terms —Circuit design automation, LLM, multimodal
design assistant, RAG, agentic workflow, Reasoning and Act
(ReAct) framework, database updating, scalable, open-source.
I. I NTRODUCTION
The rapid evolution of circuit design introduces numerous
challenges for researchers and engineers, particularly when it
comes to effectively retrieving, and utilizing vast volumes of
technical information spread across various research papers.
Developing innovative circuit design methodologies requires
∗Department of Electrical and Computer Engineering, University of Utah,
Salt Lake City, UT, 84112 USA.†Department of Electrical Engineering and
Computer Science, University of Michigan, Ann Arbor, MI, 48109 USA. (e-
mail: u1475870@umail.utah.edu, u1275778@utah.edu, u1485690@utah.edu,
soroosh.noorzad@utah.edu, alighazi@umich.edu, m.fayazi@utah.edu).an extensive literature review. However, the traditional manual
search process is both inefficient and time-consuming. It is
further complicated by inconsistent data representation formats
and the complexity of optimizing design objectives such as
area and power [1]. These challenges slow down circuit design
innovations.
Large Language Models (LLMs) [2] have emerged as
powerful tools for addressing the aforementioned challenges.
As intelligent and fast conversational agents, LLMs provide
context-aware insights and real-time assistance, enhancing
productivity and decision-making. However, deploying LLMs
in technical domains such as circuit design presents unique
issues. Conventional LLMs often generate responses that
lack domain-specific relevance. This matter gets exacerbated
because of the hallucinating [3] nature of LLMs generative
process. This leads to outputting incorrect information, partic-
ularly when dealing with technical abbreviations or nuanced
design requirements. A common mitigation strategy is to
provide the model with the entire set of relevant documents as
context during inference, which can help reduce hallucinations.
However, this brute-force method leads to significant draw-
backs. 1) It is constrained by fixed context window sizes [4],
[5]. 2) It incurs high computational and memory costs, and be-
comes increasingly impractical as the number of papers grows,
which limits scalability and efficiency. Moreover, while LLMs
excel at generating coherent text, they often struggle with
reasoning through iterative tasks or complex processes [6].
This shortfall necessitates a systematic approach to harness
the potential of LLMs effectively in circuit design workflows.
To overcome these challenges, this paper presents
MuaLLM, an open-source1intelligent design assistant capable
of answering complex, multi-step circuit design questions
using a reasoning-driven approach. The proposed system ad-
dresses the aforementioned limitations through the following:
1)Reasoning and Act (ReAct)-based framework: De-
signing circuits involves multiple steps requiring logical
reasoning, iterative actions, and decision-making [7]. As
discussed, traditional LLMs, which operate in a passive
question-answer mode, are inadequate for these require-
ments. The ReAct-based framework [8] introduces a more
dynamic and interactive workflow, allowing the system to
reason about tasks and execute them through an iterative
loop of action and feedback. This capability is critical
1https://anonymous.4open.science/r/MuaLLM-1472/README.mdarXiv:2508.08137v1  [cs.LG]  11 Aug 2025

for automating circuit design optimization and managing
workflows with complex requirements.
2)Context-aware hybrid RAG: To address domain-specific
queries, traditional RAG [9] frameworks enhance LLMs
by retrieving relevant documents from both structured and
unstructured data sources. These frameworks rely on either
sparse retrieval (keyword-based matching) or dense re-
trieval (semantic matching) techniques to identify relevant
content [10]. This content is then passed as context to the
LLM for generating answers to the query which mitigates
hallucinations. However, these approaches often fall short
in highly technical fields, including circuit design, due to
their limitations in handling the complexity and specificity
of such domains because: a) Sparse retrieval systems [11]
lack semantic depth and often fail to align with the nu-
anced terminology in technical queries; b) Dense retrieval
systems [12] may overlook rare but critical domain-specific
terms that are pivotal for precise technical solutions; c)
Modal limitations prevent effective integration of both
textual and visual data, such as technical graphs, tables,
and circuit schematics, which are crucial for solving circuit
design challenges. To address these limitations, the pro-
posed hybrid RAG framework combines the strengths of
both sparse and dense retrieval techniques. Sparse retrieval
excels in capturing domain-specific terminology, while
dense retrieval ensures semantic relevance. This hybrid
approach is further augmented by multimodal capabilities,
enabling the system to effectively retrieve and generate
responses based on both textual and visual data. By
leveraging continuous updates, MuaLLM provides precise,
contextually relevant, and up-to-date information tailored
to the circuit design workflow.
The main contributions of this paper are summarized as
follows:
•Agentic workflow: Developing a ReAct-based workflow
with reasoning capabilities to automate and optimize the cir-
cuit design procedures, leveraging its ability to decompose
complex problems into manageable steps. This approach
provides circuit designers with real-time support, and im-
proved decision-making, ensuring efficiency in addressing
intricate design challenges.
•Multimodal search capabilities: Enabling efficient retrieval
of both textual and visual information (circuit schematics,
tables, and graphs), enhancing the accessibility and usability
of technical data.
•Enhanced context-aware hybrid RAG: Designing a hybrid
system combining sparse and dense retrieval techniques
to provide precise, context-aware responses for text and
image-based circuit design-related queries. Since MuaLLM
retrieves only the most relevant chunks via a hybrid RAG
pipeline, decoupling inference from corpus size making
MuaLLM a scalable approach. At the maximum context
length supported by standard LLMs, MuaLLm remains up
to10x less costly and 1.6x faster while maintaining the
same accuracy.•Customized tools: Incorporating custom-implemented tools
into the ReAct agentic workflow such as an intelligent
search engine, dynamic database updater, and netlist gen-
erator to support real-time decision-making. These tools
enable rapid, no-human-in-the-loop database generation,
which addresses one of the main bottlenecks in circuit
design: the time-consuming process of providing labeled
data through simulations.
•Evaluation: Validating the performance of MuaLLM using
two custom datasets: a) RAG-250 for assessing retrieval and
citation capabilities; b) Reas-100 for evaluating multi-step
reasoning in circuit design. The results show that MuaLLM
achieves 90.1% recall on RAG-250 and 86.8% accuracy
on Reas-100. Both datasets, along with codes, are open-
sourced to encourage reproducibility and future research.
II. B ACKGROUND
The manual design of circuits is highly time-consuming
and prone to human error, creating a need to accelerate
the design process for faster time-to-market Integrated Cir-
cuits (ICs) [13]. Artificial Intelligence (AI) algorithms have
demonstrated promising results in automating circuit design, as
they can learn patterns without direct human intervention [7].
Among the prominent AI-based techniques used for this pur-
pose are Bayesian Optimization (BO), Deep Neural Networks
(DNNs), Graph Neural Networks (GNNs), and Reinforcement
Learning (RL) [14]–[17].
Despite the effectiveness of these methods, they either have
a slow runtime or require a large amount of labeled training
data, limiting their adaptability in high-dimensional, rapidly
evolving design spaces. AnGeL [18] has significantly reduced
both the runtime and the amount of required training data;
however, it still demands extensive manual effort to provide
its static labeled training set. Our work shifts focus to an earlier
but critical part of the design process: enabling faster iterations
through literature-driven question answering and reasoning
with LLMs.
LLMs offer new opportunities for circuit design automation
by introducing dynamic knowledge retrieval and reasoning.
Masala-CHAI [19], Auto-SPICE [20], and AMSNet [21] au-
tomate circuit netlist generation from schematics, reducing
reliance on static datasets and enhancing design efficiency.
Moreover, there are multiple works that have focused on circuit
design generation using LLMs. ADO-LLM [22] improves BO
by infusing domain expertise, while C IRCUIT SYNTH [23]
leverages LLMs for automated circuit topology synthesis.
The GLayout framework [24] enables LLM-driven analog
layout generation using RAG, while Artisan [25] employs
tree-of-thoughts and chain-of-thoughts approaches to improve
operational amplifier design [26], [27]. Also, LaMAGIC [28],
AnalogXpert [29], Atelier [30], and LADAC [31], further
reflect the rapid expansion of LLM-based approaches in circuit
design.
Hybrid approaches that integrates RAG with LLMs improve
knowledge retrieval and reduce hallucinations. Ask-EDA [32]
combines RAG with abbreviation de-hallucination, improving

accuracy and recall in circuit design queries. While effective
for downstream design tasks, these approaches do not fully
support generalizable, literature based exploration with multi-
modal reasoning.
General-purpose deep research frameworks [33], [34] exist
but are not well-suited for literature survey and question-
answering based tasks in the circuit design domain [35]. Fur-
thermore, many LLM approaches lack robust multimodal rea-
soning and domain-specific optimization needed for handling
complex design queries with multiple constraints. MuaLLM
bridges these gaps by integrating a hybrid RAG framework
with a multimodal LLM agent that processes the data in all
the modalities (text, images, tables, and graphs). This enables
efficient knowledge retrieval, iterative reasoning through a
ReAct workflow, and adaptive decision-making, significantly
improving circuit design automation, and scalability.
III. M ETHODOLOGY
A. Database
The database serves as the foundation for RAG systems.
Research papers in the field of circuit design are challenging
to integrate into databases due to their unstructured and
multimodal nature. To overcome these challenges, MuaLLM
develops a robust preprocessing pipeline that structures and
embeds text and images into a unified vector database, en-
abling precise and contextually enriched retrieval tailored to
the domain as shown in Fig. 1.
1) Preprocessing text and images: To preprocess the re-
search papers, MuaLLM extracts textual data and images from
PDF documents using the unstructured library [36]. The
extracted text is cleaned to remove artifacts and segmented into
overlapping chunks. MuaLLM uses paragraph-level chunk-
ing as it gives us better search results in comparison with
character-level chunking [37]. Overlapping chunking ensures
the semantic continuity is preserved across related sections,
maintaining the flow of information. This step is crucial, as
processing entire documents is computationally expensive and
constrained by the token limits of LLMs. Simultaneously,
images are extracted and stored for further processing. These
images, ranging from circuit diagrams to plots, are critical for
multimodal analysis.
2) Embeddings: Embeddings enable efficient similarity-
based retrieval by mapping data into a vector space. While
initially OpenAI’s CLIP (Contrastive Language-Image Pre-
training) model [38] was used for multimodal embedding,
its general-purpose training limited its ability to handle the
domain-specific details found in circuit design.
To overcome this limitation, MuaLLM adopts a descriptive
embedding approach for images [39]. Each image is processed
using LLM to generate detailed descriptions, capturing: a)
The type of visual content ( e.g. “circuit diagram”, “plot”,
or “specification table”); b) Key elements, such as labeled
electronic component types ( e.g. resistor or capacitor); c)
Domain-specific technical details ( e.g.the circuit functionality
of the circuit schematic image).The enriched descriptions are embedded using the
voyage-2 [40] model and stored alongside textual embed-
dings in the vector database. Each embedding is paired with
metadata, including the path to the original image. This setup
ensures that when an embedding is retrieved, the correspond-
ing image provides the necessary context for the generative
model.
While it is true that large language models can hallucinate,
our use of LLM-generated text interpretations is not for factual
reasoning, but strictly to enhance search recall and indexing of
visual content. These interpretations are embedded alongside
original image captions using the V oyage-2 model to improve
multimodal retrieval accuracy. Importantly, we evaluate the
downstream impact of these embeddings on search perfor-
mance using our custom benchmark datasets (RAG-250 and
Reas-100). Thus, even if occasional hallucinations occur in
the LLM-generated image descriptions used for embedding,
they do not impact the agent’s reasoning or final outputs, as
the embeddings are solely intended to enhance image-aware
search relevance.
3) Contextualization: Traditional RAG systems [9] divide
documents into smaller chunks for efficient retrieval. While
computationally effective, this approach often isolates chunks
from their broader context, leading to ambiguous results. This
lack of context can diminish the utility and reliability of the
retrieved information [41].
To generate context and improve response relevance,
MuaLLM implements a contextual caching mechanism [41].
This caching system stores reusable context data generated
during previous queries, minimizing redundant Application
Programming Interface (API) calls and reducing computational
costs.
4) Unified vector database: All preprocessed text and im-
ages, along with their enriched embeddings and metadata, are
stored in a unified vector database. This structure facilitates
seamless multimodal search, allowing effective querying of
both text and images. By preserving context, the system
ensures that isolated chunks or images are retrieved with
their full meaning which significantly enhances relevance and
usability for technical queries.
B. Contextual hybrid RAG
RAG [9] improves model responses by integrating exter-
nal knowledge through a combination of data retrieval and
response generation. In this process, the model first retrieves
relevant information from an external knowledge base using
either semantic search or keyword search. This ensures that the
most pertinent and useful data is selected based on the query.
After retrieval, this information is passed to a generative model
as context, which then processes it to generate accurate and
contextually relevant responses. The detailed implementation
of the hybrid RAG is as follows.
1) Dual search: The hybrid contextual RAG [41] system
utilizes two distinct retrieval mechanisms: semantic search and
keyword search.

CorpusChunk 1
Chunk 2
Chunk NRun prompt for every 
chunk & image 
summary to situate it 
within the documentChunk 2 + Context 2 Chunk 1 + Context 1Preprocessing (with Contextual Retrieval ) Runtime (with Reranking ) 
Image 1
Image 2
Image MEmbedding 
Model
TF-IDF Query
Top N 
ChunksRank 
Fusion Top K 
Chunks
Final response 
with both text 
& images
RerankerGenerative 
ModelImage 1 Summary + 
Image Context 1
Image 2 Summary + 
Image Context 2Chunk N + Context N 
Image M Summary + 
Image Context MVector 
Database
TF-IDF 
Index TF-IDF
EncodingsEmbeddings
Run prompt for every 
Image to summarize 
the contents of the 
ImagesFig. 1. The proposed contextualized multimodal RAG workflow.
•Semantic search: This method is designed to retrieve doc-
uments based on the meaning or underlying concept of the
query. It excels in situations where the user’s query may
involve synonyms or abstract terms. This allows the model
to fetch documents that are contextually related but may not
have an exact keyword match. The embeddings facilitate
this semantic search, ensuring that the retrieved documents
are conceptually aligned with the user’s intent.
•Keyword search: In contrast to semantic search, BM25 [12],
a keyword-based search algorithm, focuses on exact
matches of query terms within the document corpus.
By utilizing both search methods in parallel, the hybrid
RAG system effectively captures both conceptually relevant
and keyword-specific results, thereby expanding the breadth
of retrieved data. This approach is particularly advantageous
for circuit design, which is rich in both technical and semantic
terminology.
2) Ranking: Once the semantic and BM25-based results are
retrieved, the system combines the outputs from both methods
to form a unified set of relevant documents [42]. To prioritize
the most relevant documents, each retrieval result is assigned a
weighted score based on the search method used to retrieve it.
The semantic weight typically holds more significance when
context and conceptual relevance are more important, whereas
the BM25 weight is employed for keyword-matching results.
3) Re-ranking: The re-ranking step further refines the re-
trieval results [41]. After the initial retrieval phase, the results
are re-ranked using the Cohere [43] model to reorder the
documents based on their true relevance to the query. The
re-ranking process leverages Cohere ’s [43] understanding
of the relationship between the query and the documents,
ensuring that the final output is textually and semantically
aligned.
4) Response generation: Once the relevant documents have
been retrieved and refined, they are passed to the generative
model ( e.g.GPT-4o [5]), which synthesizes the information
to generate a response. The generative model uses the external
knowledge extracted during the retrieval process to fill in any
gaps within its internal knowledge. This creates a response
that is both accurate and contextually appropriate.While GPT-4o is used in our primary implementation,
the system is modular and model-agnostic: Claude 3.5
Sonnet [44] or other frontier LLMs can be easily swapped
in for generation. This flexibility allows MuaLLM to adapt to
evolving model capabilities, cost structures, or domain-specific
strengths depending on deployment requirements.
5) Multimodality: An important extension of the hybrid
RAG system is its ability to handle multimodal content. This
means that it can process not only textual documents but also
images, graphs, and tables. By distinguishing between text
and image content during the retrieval process, the system
ensures that all types of data are handled appropriately. This
feature is especially important in fields such as circuit design,
where visual content plays a significant role alongside text.
The hybrid contextual RAG system integrates the best features
of the retrieval-based and generation-based approaches. By
combining semantic, keyword-based search, and contextual
understanding, it generates highly relevant and accurate re-
sponses.
C. Agentic Workflow
The ReAct-based agentic workflow [8] enables iterative
problem-solving by cycling through thought ,action ,pause ,
andobservation steps. Unlike standard LLMs that passively
generate answers, the agent actively reasons through the query,
determines the necessary information, and selects the appro-
priate tools. During the action step, the model retrieves data
using the hybrid RAG system, queries databases or employs
multimodal tools as required. After pausing for results, it
observes the retrieved data, reassesses gaps, and, if necessary,
selects additional tools, such as fetching external documents
or updating the database, to refine its understanding. This
cycle repeats until the agent has gathered all the necessary
information to answer the query. This ensures a more context-
aware, accurate, and adaptive response compared to static
LLM-based retrieval.
D. Tools
These custom-implemented tools are integrated into the
ReAct-based agentic framework of MuaLLM to enable intel-
ligent, multimodal task execution in real-time. Each tool is

invoked dynamically during the reasoning process, allowing
the agent to adaptively retrieve, enrich, and utilize information
for circuit design queries.
1) Database Searcher (search db): This tool retrieves rel-
evant information from large-scale structured and unstructured
circuit design literature. It supports both sparse keyword-
based retrieval using BM25 and dense semantic search using
vector embeddings. Results are re-ranked using the Cohere
reranker, enabling high-precision multimodal retrieval of both
textual content (e.g., parameter tables) and visual content (e.g.,
circuit schematics). The tool operates with low latency and is
optimized for recall and ranking quality across hybrid search
workflows.
2) Automatic Paper Fetcher (paper fetcher): The pa-
perfetcher tool enables the agent to autonomously fetch
academic literature from public repositories such as Google
Scholar and arXiv. When queried about papers not present in
the local database, the agent invokes this tool to download the
PDF files, which are then passed to the preprocessing pipeline.
This allows MuaLLM to dynamically expand its knowledge
base and access the latest research with no human intervention.
3) Dynamic Database Updater (search db –load data):
After new papers are fetched, the search db –load data tool
preprocesses the documents and updates the vector database
in real-time. It extracts text and images from PDFs, generates
enriched descriptions for visual content using LLM-based
captioning, and embeds the content using the V oyage-2 model.
This tool ensures the database remains up-to-date, facilitating
continuous learning and accurate response generation for fu-
ture queries.
4) Netlist Generator: This tool automates the conversion
of schematic circuit diagrams into SPICE-compatible netlists.
It employs a YOLO-based object detection model to iden-
tify circuit components and OpenCV’s connected components
analysis to extract nodes. The detected elements are parsed
to form structured netlists, enabling simulation-ready outputs
and eliminating the need for manual circuit annotation. The
netlist generator is especially valuable for accelerating dataset
creation and validating extracted designs.
5) Netlist Generator: Converts circuit schematic images
into accurate circuit netlists using a pipeline that combines
object detection with image processing techniques. The first
step is component detection and node identification. In this
step, we use a YOLO model [45], trained on circuit symbols,
to detect components in the schematic and identify their
labels and bounding boxes. To identify the nodes, we use
OpenCV’s Connected Components with Stats function [46],
which uniquely recognizes all connected wires. To facilitate
this, detected components are first removed from the image,
leaving only isolated clusters of connected wires that form
the nodes. It should be noted that due to the presence of text
annotations and other noise in the image, the identified nodes
are validated to include only those that intersect the edges of
two or more circuit components.
In the next step, for each detected component, we identify
which validated nodes are connected to. Each node is assigneda unique numerical identifier, with ground-connected nodes
clustered together and assigned the lowest identifier to reflect
their shared electrical potential. A visual overview example
of this process is shown in Fig. 2. Using this tool, MuaLLM
rapidly creates a large database of circuits, addressing the long-
standing challenge of time-consuming labeled data generation
through simulations.
IV. E VALUATION
A. Datasets
Since no publicly available benchmark datasets exist for the
evaluation of our task, we curated two custom datasets—RAG-
250 and Reas-100—in collaboration with domain experts.
These datasets are designed to evaluate the performance of
MuaLLM, we curated two datasets: RAG-250 and Reas-100.
These datasets are designed to assess the system’s perfor-
mance in literature-based question-answering and multi-step
reasoning tasks related to circuit design. Specifically, we
selected research papers on Bandgap Reference (BGR) [47]
and oscillator circuits [48] as representative domains to bench-
mark MuaLLM’s capabilities. While these focus areas provide
structured evaluation scenarios, it is important to note that
MuaLLM is generalizable and can be applied to any class of
circuit literature, enabling broad utility across circuit design
workflows.
1) RAG-250 Dataset: This dataset is a collection of
question-and-answer (Q&A) pairs derived from literature on
BGR circuits. Since no standard Q&A dataset exists for the
circuit design domain, we manually extracted 250 questions
from technical papers. Each question is mapped to expert-
verified answers, serving as ground truth labels for evaluating
response accuracy. By using this dataset, we assess how
well our multimodal RAG retrieves information, synthesizes
insights, and generates accurate, literature-backed responses.
2) Reas-100 Dataset: This dataset is designed to evaluate
MuaLLM’s multi-step reasoning and analytical capabilities in
circuit design. This dataset, which contains 100 reasoning
questions, unlike standard Q&A datasets [49], focuses on
complex circuit design queries that require deeper reasoning.
Answering these queries involves breaking down the prob-
lem into logical steps, using relevant tools ( e.g. search db,
paper fetcher ), and synthesizing a structured response. To
evaluate MuaLLM’s performance, we manually compare its
responses against expert-generated solutions.
By testing both RAG-250 and Reas-100, a comprehen-
sive evaluation of factual accuracy, retrieval efficiency, and
reasoning depth of MuaLLM is obtained. Table I shows a
sample query from each dataset, demonstrating the types of
factual and reasoning-based problems the system is tested on.
As it is illustrated in Table I, the RAG-250 example query
asks for a direct explanation of a known paper, relying on
factual recall. In contrast, the Reas-100 example query requires
analyzing design parameters, referencing multiple sources, and
synthesizing a solution, demonstrating multi-step reasoning
and decision-making.

(e)V1 0 5 Volt_src 1.2V
Qp 2 3 7 BJT
Qn 1 4 6 BJT
R4 0 1 Res 1k Ω
R3 6 5 Res 2k Ω
R2 0 2 Res 1k Ω
R1 7 5 Res 2k Ω
R5 3 4 Res 0.1k ΩV1 0 5 Volt_src 1.2V
Qp 2 3 7 BJT
Qn 1 4 6 BJT
R4 0 1 Res 1k Ω
R3 6 5 Res 2k Ω
R2 0 2 Res 1k Ω
R1 7 5 Res 2k Ω
R5 3 4 Res 0.1k Ω
(a)R2 R4Qn QpR5R1 R3
V1+
1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
(a)R2 R4Qn QpR5R1 R3
V1+
1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
R2 R4Qn QpR5R1 R3
V1+Res Res
ResBJT Res
ResBJTVolt_src
GND GND GND
(b)1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
R2 R4Qn QpR5R1 R3
V1+Res Res
ResBJT Res
ResBJTVolt_src
GND GND GND
(b)1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
(c)R2 R4Qn QpR5R1 R3
V1
1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
(c)R2 R4Qn QpR5R1 R3
V1
1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
(d)R2 R4Qn QpR5R1 R3
V1+
1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩ
(d)R2 R4Qn QpR5R1 R3
V1+
1.2V
1kΩ2kΩ2kΩ 1kΩ0.1kΩFig. 2. An example of automatic circuit schematic image to netlist generation steps using MuaLLM. (a) The original circuit schematic image. (b) The detection
model output with identified components and their bounding boxes. (c) Removing components and clustering nodes. (d) Nodes and their connections to the
components. (e) The final generated netlist.
TABLE I
SAMPLE QUERIES FROM RAG-250 AND REAS-100 DATASETS .
Dataset Example
RAG-250Question: Explain the proposed quadrature voltage-controlled
oscillator in the paper ”CMOS Quadrature VCOs Using the
Varactor Coupling Technique”.
Answer: The proposed quadrature voltage-controlled oscillator
(QVCO) comprises two cross-coupled LC VCOs coupled via
varactors, implemented in TSMC 0.18 µm CMOS. Each includes
an LC tank (inductors and varactors) for tuning and cross-coupled
transistors for negative resistance. Varactors enable both tuning
and 90° quadrature coupling. Resistors bias the varactors and
PMOS bodies. The design operates from 3.86–4.22 GHz with
3.7 mW power consumption. The circuit is shown in Fig. 1.
Reas-100Question: Design a sub-BGR with a power consumption in the
range of 50 nW.
Answer: To design a sub-BGR with power consumption in
the range of 50 nW, I would implement Osaki’s sub-BGR
architecture as presented in their paper “1.2-V Supply, 100-nW,
1.09-V Bandgap and 0.7-V Supply, 52.5-nW, 0.55-V Subbandgap
Reference Circuits for Nanowatt CMOS LSIs”. Their design
achieves 52.5 nW power consumption using a nano-ampere
current reference circuit, a bipolar transistor, and PTAT voltage
generators. The key component is the voltage divider circuit
shown in the extracted figure that divides the base-emitter voltage
of the bipolar transistor without resistors, enabling sub-1V opera-
tion. By following this architecture with the transistor sizing and
current reference values specified in their implementation, a sub-
BGR with approximately 50 nW power consumption is realized.
B. MuaLLM Results
1) RAG-250 Dataset: To evaluate the retrieval quality and
response accuracy, the system’s answers (using both GPT
andClaude as the generative models) are compared against
expert-verified ground truth responses. The multimodal RAG
achieves an overall recall of 88.1% withClaude and90.1%
with GPT, demonstrating its ability to retrieve and synthe-
size relevant information effectively. Table II summarizes the
performance evaluation of the multimodal RAG system. The
assessment covers overall response accuracy, image citation
precision, equation citation recall, and text citation F1 score,
showcasing the system’s robust performance across multi-
modal content. The query distribution (the number of each
modal queries) is also listed in Table II.
The chosen metrics in Table II, i.e.precision, recall, and
F1 score, evaluate key aspects of the multimodal RAG sys-
tem’s performance. Recall assesses completeness for overallTABLE II
MUALLM’ S MULTIMODAL RAG PERFORMANCE EVALUATION TESTED
WITH BOTH CL A U D E ANDGPT.
Evaluation MetricQuery
DistributionMuaLLM Multimodal RAG
Claude GPT
Overall Response (Recall) 250 88.1% 90.1%
Image Citation (Precision) 130 89.8% 93.02%
Equation Citation (Recall) 45 79.0% 73.67%
Text Citation (F1 Score) 75 93.0% 96.0%
0 20 40 60 80 100
Number of Pages00.10.20.30.40.50.6Total Cost ($) Per Query
(a)0 20 40 60 80 100
Number of Pages00.10.20.30.40.50.6Total Cost ($) Per Query
(a) (b)0 50 100 150
Number of Pages81012141618Latency (seconds) Per Query GPT-4o
Claude Sonnet 3.5
MuaLLM GPT -4o
MuaLLM Claude Sonnet 3.5GPT-4o
Claude Sonnet 3.5
MuaLLM GPT -4o
MuaLLM Claude Sonnet 3.5
GPT-4o
Claude Sonnet 3.5
MuaLLM GPT -4o
MuaLLM Claude Sonnet 3.5GPT-4o
Claude Sonnet 3.5
MuaLLM GPT -4o
MuaLLM Claude Sonnet 3.5Context 
length limit
Context 
length limit
Fig. 3. (a) Cost and (b) latency vs pages per query comparison between
MuaLLM and the conventional LLMs. The hatched area shows the context
length limitation that the conventional LLMs face when processing around
more than 100 pages.
responses and equation citations, ensuring relevant information
is captured. Precision measures accuracy in image citations,
which is crucial for visual data such as circuit diagrams.
The F1 score balances precision and recall for text citations,
ensuring both accuracy and coverage.
Since there are no widely available benchmarks to evaluate
in terms of scalability and efficiency, we compare MuaLLM
against a baseline where full research papers are passed
directly as context to standard LLMs ( e.g. GPT-4o [5],
Claude Sonnet 3.5 [50]) for question-answering tasks. While
this approach mimics the core objective, retrieving answers
from literature, it quickly hits context length limits-fifteen pa-
pers (1,400 tokens/page) saturate most LLM windows [4] [5],
making this approach impractical for comprehensive literature
reviews. As shown in Fig. 3, cost and latency grow with input
page number under the direct full context method. In contrast,
MuaLLM retrieves only the most relevant chunks via a hybrid
RAG pipeline, decoupling inference from corpus size for cost

GNDVDD
VBEI
Current 
reference 
circuitVoltage 
divider
QPTAT 
voltage 
generator1/M 
VBE1/M 
VREF
GNDVDD
VBEI
Current 
reference 
circuitVoltage 
divider
QPTAT 
voltage 
generator1/M 
VBE1/M 
VREFCompare the power dissipation of the sub -
BGR circuits described in “1.2-V Supply, 
100-nW, 1.09 -V Bandgap and 0.7 -V Supply, 
52.5-nW, 0.55 -V Sub Bandgap Reference 
Circuits for Nanowatt CMOS LSIs ” and “A 
CMOS Bandgap and Sub -Bandgap Voltage 
Reference Circuits for Nanowatt Power 
LSIs.” Identify which circuit has the 
lowest power dissipation and provide its 
architecture.
Thought 1: I need to compare the sub -BGR 
circuit architectures from both papers and 
identify which has lower power 
dissipation.
Act 1: search_db  [Architecture and Power 
dissipation of sub -BGR circuit in "1.2 -V 
Supply, 100 -nW..." paper]
Obs 1: The sub -BGR circuit has a power 
dissipation of 52.5 nW (0.052 μW) and it ’s 
architecture as shown in fig 4.
Thought 2: Now I need to find the 
architecture and power dissipation of the 
sub-BGR circuit in the second paper.
Act 2: search_db  [Architecture and power 
dissipation of sub -BGR in "A CMOS Bandgap 
and Sub-Bandgap..." paper]]
Obs 2: No relevant results found in the 
database.
Thought 3: This suggests the paper is not 
there in the database, I need to download 
and process the second paper to extract 
the required information.
Act 3: paper_fetcher  [“ A CMOS Bandgap and 
Sub-Bandgap Voltage Reference 
Circuits..."]
Obs 3: Paper downloaded successfully and 
stored at {path_to_PDF}.
Thought 4: I need to process the PDF to 
chunk the text and extract images, then 
store in the database for searching.
Act 4: pdf2json  [{path_to_PDF}]
Obs 4: PDF processed successfully. Text 
and images are extracted.
Thought 5: Now I need to store these in 
db.
Act 5: search_db  –load_data
Obs 5: 63 embeddings were stored in db
Thought 6: Now I can search for the 
architecture and power dissipation of the 
sub-BGR circuit in the second paper again.
Act 6: search_db  [Architecture and power 
dissipation of sub -BGR in "A CMOS Bandgap 
and Sub-Bandgap..." paper ]After comparing the sub -
BGR circuit 
architectures from the 
two papers on nanowatt 
CMOS LSIs, I can 
determine that: 
- The sub -BGR circuit 
from "1.2 -V Supply, 100 -
nW, 1.09 -V Bandgap and 
0.7-V Supply, 52.5 -nW, 
0.55-V Subbandgap 
Reference Circuits for 
Nanowatt CMOS LSIs" has 
a power dissipation of 
52.5 nW (0.052 μW). 
- The sub -BGR circuit 
from "A CMOS Bandgap and 
Sub-Bandgap Voltage 
Reference Circuits for 
Nanowatt Power LSIs" has 
a power dissipation of 
110 nW (0.11 μW). 
Therefore, the sub -BGR 
circuit with the lower 
power dissipation is 
from the first paper, 
with a power dissipation 
of 52.5 nW.
Architecture of the low -
power sub -BGR circuit:
The sub -BGR circuit 
consists of four main 
blocks connected between 
power (VDD) and ground. 
It starts with a current 
reference circuit that 
feeds a transistor pair 
(PMOS and BJT), creating 
a temperature -dependent 
voltage VBE. This 
voltage passes through a 
voltage divider, 
reducing it to 1/M VBE, 
before entering a PTAT 
voltage generator that 
produces the final 
reference output (1/M 
VREF).QueryQuery
ReAct (Reason + Act)ReAct (Reason + Act)AnswerAnswerFig. 4. Example 1: ReAct (Reason + Act) framework for a sample query.
and computation. At maximum context length, it is at least
10x more efficient in cost and 1.6x faster while preserving
accuracy. One-time preprocessing costs $0.21/paper, enabling
low-latency, cost-stable queries suitable for literature-intensive
workflows.
While we benchmark our system using direct API-based
approaches with GPT-4o andClaude Sonnet 3.5 , it
is important to note that direct comparisons with existing
domain-specific agents such as Ask-EDA [32] are not fea-
sible, as it is not open-sourced. Its implementation details
are unavailable for evaluation. Meanwhile, Masala-CHAI [19],
MATLAB Circuit Simulator [51], and similar systems focus on
synthesis tasks such as SPICE netlist generation from circuit
schematics and circuit optimization rather than literature-
grounded question answering. Hence, they address different
goals and are not directly aligned with the problem MuaLLM
Thought 1: I should first 
search for the BGR circuit 
specifications and architecture 
from all the papers.Which BGR circuit has the 
lowest power?
Act 1: load_titles
Obs 1: Retrieved 3 paper 
titles.
Thought 2: I need to know the 
power consumption in "1.2 -V 
Supply, 100 -nW, 1.09 -V 
Bandgap..." paper.
Act 2: Search_db  [power 
consumption in “1.2-V Supply … 
”]
Obs 2: BGR circuit has power 
dissipation of 100 nW, sub -BGR 
circuit has 52.5 nW.
Thought 3: I need to check "A 
CMOS Bandgap and Sub -
Bandgap..." paper.
Act 3: search_db  power 
consumption in “A CMOS …”]
Obs 3: BGR circuit has power 
dissipation of 108 nW, sub -BGR 
circuit has 110 nW.
Thought 4: I need to check "A 
CMOS bandgap reference without 
resistors" paper.
Act 4: search_db  [power 
consumption in “A CMOS bandgap 
….”]
Obs 4: BGR circuit has power 
consumption of 1.4 mW.Based on the power consumption 
data analysis across three 
different papers on bandgap 
reference circuits, I can 
determine that:
1. "1.2 -V Supply, 100 -nW, 1.09 -V 
Bandgap and 0.7 -V Supply, 52.5 -nW, 
0.55-V Subbandgap Reference 
Circuits for Nanowatt CMOS LSIs" 
has a power dissipation of 100 nW.
2. "A CMOS Bandgap and Sub -Bandgap 
Voltage Reference Circuits for 
Nanowatt Power LSIs" has a power 
dissipation of 108 nW.
3. "A CMOS bandgap reference 
without resistors" has a power 
consumption of 1.4 mW.
Therefore, the BGR circuit with 
the lowest power consumption is 
from the first paper, with a power 
dissipation of 100 nW.
Power Consumption Comparison
Paper 1: 100 nW
Paper 2: 108 nW
Paper 3: 1.4 mW (14,000 nW)
Below is the BGR architecture from 
the paper "1.2 -V Supply, 100 -nW, 
1.09-V Bandgap and 0.7 -V Supply, 
52.5-nW, 0.55 -V Subbandgap 
Reference Circuits for Nanowatt 
CMOS LSIs".
GNDVDD
VBEI
Current 
reference 
circuit
QPTAT 
voltage 
generator VREF
GNDVDD
VBEI
Current 
reference 
circuit
QPTAT 
voltage 
generator VREFQueryQuery
ReAct (Reason + Act)ReAct (Reason + Act)AnswerAnswerFig. 5. Example 2: ReAct (Reason + Act) framework for a sample query.
solves.
2) Reas-100 Dataset: The reasoning performance of
MuaLLM is evaluated by comparing its responses against
expert-generated answers. MuaLLM achieves an accuracy of
84.0% with Claude and 86.8% with GPT, demonstrating
its ability to provide structured and logical responses. Two
examples of the reasoning agent’s capabilities are shown in
Fig. 4 and Fig. 5. For demonstration purposes, we illustrate
how the workflow would appear if the database contains six
papers.
The ReAct (Reason + Act) framework for a sample query is
shown in Fig. 4. This demonstrates an iterative reasoning pro-
cess where the agent combines logical thinking with strategic
actions. It begins by identifying the need to compare two sub-
BGR circuits and retrieves available data from the database.
Upon discovering incomplete information, the agent reasons
that the missing document must be fetched, processed, and
stored in the database. Without human intervening, the agent
dynamically adapts by iterating through steps of reasoning ,
action , and observation until all relevant information is gath-
ered. This adaptive approach enables the agent to efficiently
solve complex, multi-step queries with minimal intervention.
In the example shown in Fig. 5, to compare the power
consumption of multiple BGR circuits, the agent first reasons
that it must retrieve the titles of all relevant papers in the
database, rather than directly searching for power values. This
strategic decision ensures the agent has a comprehensive view

Query
Consider a BGR circuit with 
the following 
specifications, Technology: 
0.18um, supply voltage: 
1.8V, current consumption: 
17uA, average vref: 
1141.9mV.
Answer these questions: A) 
what is the active area in 
mm2 of such a BGR? B) output 
the high -level block diagram 
of a BGR with the 
aforementioned 
specifications.
Based on the data presented 
in Table 2 of reference [3] 
column, the active area for 
a BGR circuit with the 
specified characteristics — 
0.18 μm technology, 1.8V 
supply voltage, 17 μA 
current consumption, and an 
average Vref of 1141.9 mV — 
is 0.38 mm ². 
While the search did not 
provide a specific BGR 
architecture diagram  for 
these specifications, a 
typical design generally 
includes key blocks such as 
a startup circuit, PTAT 
generator, CTAT generator, 
summing node, and output 
buffer to ensure stable and 
accurate voltage reference 
generation. Based on the data presented in 
[3] column of Table 2 in the 
paper titled *"Sub -μW Auto -
Calibration Bandgap Voltage 
Reference With 1σ Inaccuracy of 
±0.12% Within −40 ◦C to 120 ◦C"*, 
the active area for a BGR circuit 
with the specified 
characteristics — 0.18 μm 
technology, 1.8V supply voltage, 
17 μA current consumption, and an 
average Vref of 1141.9 mV — is 
**0.38 mm ²**. For the BGR 
architecture, reference [3] 
titled *"Single -Trim Switched 
Capacitor CMOS Bandgap Reference 
With a 3σ Inaccuracy of +0.02%, 
−0.12% for Battery -Monitoring 
Applications"* presents a BGR 
design that features a switched -
capacitor summing amplifier. This 
architecture combines multiple 
PTAT and CTAT voltage nodes with 
weighted coefficients to generate 
a stable reference voltage. The 
corresponding block diagram is 
shown in Fig. 2, demonstrating 
key components such as PTAT and 
CTAT generators, transistors Q1, 
Q2, and Q3 for voltage 
generation, and a switched -
capacitor summing amplifier for 
precise voltage reference output. 
Specification
Supply Voltage (V)
Current 
Consumption (µA)
Average V REF (mV)
Trimming MethodThis 
work
Technology 65nm
1.2 - 2.5
0.81
1001.11001.1
Auto 
1-pointAuto 
1-point
Trimmed TC 
(ppm/ºC)Trimmed TC 
(ppm/ºC)
PSRR
Active Area (mm2)18.6-29 
(µ:22.3)18.6-29 
(µ:22.3)
58 dB 
@10Hz 
0.04[7]
0.13µm
3.3
120
11601160
UniformUniform
5.8-13.5 
(µ:8.75)5.8-13.5 
(µ:8.75)
82 dB 
@10Hz 82 dB 
@10Hz 
0.080.08[15]
0.18µm
1.3-1.8
0.15
11701170
UniformUniform
12.2-41 
(µ:26.3)12.2-41 
(µ:26.3)
52 dB 
@100Hz 52 dB 
@100Hz 
0.00820.0082[16]
90nm
1.15-1.3
0.50
723723
UniformUniform
43-61.7 
(µ:53.1)43-61.7 
(µ:53.1)
50 dB 
@DC 50 dB 
@DC 
0.00820.0082[6]
0.5µm
2.1-5
38
11961196
Manual
1-pointManual
1-point
3.98-8.3 
(µ:5.87)3.98-8.3 
(µ:5.87)
84 dB 
@100Hz 84 dB 
@100Hz 
0.0530.053[5]
0.13µm
2.6-5
94
24712471
Manual
1-pointManual
1-point
0.9-7.52 
(µ:3.0)0.9-7.52 
(µ:3.0)
83 dB 
@DC 83 dB 
@DC 
0.06160.0616[4]
0.16µm
1.8
116
1087.51087.5
Manual
1-point
5-12 
N.A.5-12 
N.A.
74 dB 
@DC 74 dB 
@DC 
0.120.120.18µm
1.8
3.2-5.5 
(µ:4.3)
76 dB 
@DC [17]
0.18µm
1.2-1.8
0.08
10901090
UniformUniform
N.A. 
(µ:147)N.A. 
(µ:147)
62 dB 
@100Hz 
0.02940.0294[14]
0.18µm
1.3-1.8
28
547547
UniformUniform
1.7-10.6 
N.A.1.7-10.6 
N.A.
62 dB 
@DC 62 dB 
@DC 
0.00940.0094[3]
17
1141.9
Manual
1-point
0.38Multimodal RAG OutputMultimodal RAG Output
VDDVDD
VEB,24×IPTAT
Q2VDDVDD
VEB,3ITI
Q3VDDVDD
VEB,1IPTAT
Q1-A-A
A+B+ 1 A+B+ 1
-B-B-A
A+B+ 1
-BVEB,1
+++
SC summing amplifierVREFVEB,2
VEB,3VDD
VEB,24×IPTAT
Q2VDD
VEB,3ITI
Q3VDD
VEB,1IPTAT
Q1-A
A+B+ 1
-BVEB,1
+++
SC summing amplifierVREFVEB,2
VEB,3
Comparison TableComparison TableReAct OutputReAct OutputFig. 6. A comparison example between outputs of multimodal RAG and
ReAct agent to a sample query.
of all available sources. Once the corresponding paper titles are
obtained, the agent iteratively searches for power consumption
details in each paper. By methodically examining one source
at a time, the agent efficiently handles incomplete information,
identifies gaps, and continues searching until all necessary data
is gathered. This structured reasoning approach enables the
agent to comprehensively analyze the available information
and accurately determine which BGR circuit has the lowest
power consumption in this example.
Fig. 6 shows a comparison example between the outputs of
the Multimodal RAG (MRAG) and ReAct frameworks for the
same query. This is to assess their performance in retrieving
both numerical values and relevant circuit architectural dia-
gram details. In the case of the MRAG, the system successfully
retrieves the required values and the relevant comparison table
(Fig. 6 at the bottom [52]) via a direct search. However, since
Resistor
Capacitor
Inductor
BJT
MOSFET
Voltage_src
Current_src
GND
OPAMP
Background1.0
0.8
0.6
0.4
0.2
0.0Resistor
Capacitor
Inductor
BJT
MOSFET
Voltage_src
Current_src
GND
OPAMP
Background
0.79
0.81
0.87
0.86
0.99
0.93
1.00
0.88
1.000.08 0.35
0.18
0.03
0.07
0.10
0.10
0.140.05
0.04
0.12 0.19 0.04 0.14 0.01 0.07 0.120.01
0.01Predicted
TrueFig. 7. Netlist generator component detection confusion matrix.
TABLE III
NETLIST GENERATOR PERFORMANCE EVALUATION .
Precision Recall F1 Score Weighted Mean Average Precision
0.94 0.91 0.92 0.99
MRAG lacks reasoning capabilities, it fails to recognize that it
should further search for the BGR architecture by referring to
the paper “titled [3]” (the corresponding column is highlighted
in the comparison table in Fig. 6). Consequently, MRAG fails
to provide the relevant circuit architecture diagram.
In contrast, the ReAct agent demonstrates superior reason-
ing abilities. After retrieving the comparison table, the agent
reasons that it should specifically search the database for the
reference “titled [3]” to obtain the required circuit architecture
diagram details. By following this reasoning step, the ReAct
agent successfully identifies and retrieves the appropriate BGR
architecture diagram. This example highlights the superiority
of the ReAct framework in handling complex, multi-step
queries. While MRAG retrieves known values, the ReAct
agent’s ability to plan, adapt, and explore additional references
enables it to provide a more complete and accurate response.
C. Netlist Generator Results
The netlist generator is trained and tested on more than
700 and 150 component instances, respectively. The netlist
generator performance metrics are summarized in Table III.
Also, the normalized confusion matrix for the component
detection is depicted in Fig. 7. It should be noted that Masala-
CHAI [19] has not reported any results regarding its schematic
image to netlist generation accuracy to compare with.
V. C ONCLUSION
In this work, we introduce MuaLLM, an open-source
ReAct-based LLM agent with multimodal RAG, to enhance

the efficiency of literature review in circuit design. By in-
tegrating iterative reasoning with a hybrid retrieval frame-
work, MuaLLM effectively processes complex multi-objective
queries. The agent dynamically adapts to new literature us-
ing integrated custom tools such as the Automatic Paper
Fetcher, Database Updater, and Netlist Generator. MuaLLM
is validated on two custom datasets (RAG-250 and Reas-
100). The agent achieves an overall 90.1% recall on RAG-
250. On Reas-100, it reaches 86.8% accuracy, demonstrating
its multimodal citation and multistep reasoning capabilities at
scale. At the maximum context length supported by standard
LLMs, MuaLLm remains up to 10x less costly and 1.6x faster
while maintaining the same accuracy.REFERENCES
[1] M. Fayazi, Z. Colter, Z. Benameur-El Youbi, J. Bagherzadeh, T. Ajayi,
and R. Dreslinski, “FASCINET: A Fully Automated Single-Board
Computer Generator Using Neural Networks,” IEEE Transactions on
Computer-Aided Design of Integrated Circuits and Systems , vol. 41,
no. 12, pp. 5435–5448, 2022.
[2] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever,
“Language Models are Unsupervised Multitask Learners,” OpenAI ,
2019.
[3] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qin, and T. Liu, “A Survey on Hallucination in
Large Language Models: Principles, Taxonomy, Challenges, and Open
Questions,” ACM Transactions on Information Systems , vol. 1, no. 1,
p. 58, January 2024.
[4] Anthropic, “What Is The Maximum Prompt
Length?” https://support.anthropic.com/en/articles/
7996856-what-is-the-maximum-prompt-length, 2023, accessed:
2025-05-30.
[5] OpenAI, “GPT-4o,” https://platform.openai.com/docs/models/gpt-4o,
2024, accessed: 2025-05-30.
[6] G. Gendron, Q. Bao, M. Witbrock, and G. Dobbie, “Large
Language Models Are Not Strong Abstract Reasoners,” arXiv preprint
arXiv:2305.19555 , 2023.
[7] M. Fayazi, Z. Colter, E. Afshari, and R. Dreslinski, “Applications of
Artificial Intelligence on the Modeling and Optimization for Analog
and Mixed-Signal Circuits: A Review,” IEEE Transactions on Circuits
and Systems I: Regular Papers , vol. 67, no. 8, pp. 2405–2418, 2020.
[8] S. Yao et al. , “ReAct: Synergizing Reasoning and Acting in Language
Models,” arXiv preprint arXiv:2210.03629 , 2022.
[9] P. Lewis et al. , “Retrieval-Augmented Generation for Knowledge-
Intensive NLP Tasks,” arXiv preprint arXiv:2005.11401 , 2020.
[10] P. Mandikal and R. Mooney, “Sparse Meets Dense: A Hybrid
Approach to Enhance Scientific Document Retrieval,” arXiv preprint
arXiv:2401.04055 , 2024.
[11] T. Formal, C. Lassance, B. Piwowarski, and S. Clinchant, “SPLADE v2:
Sparse Lexical and Expansion Model for Information Retrieval,” arXiv
preprint arXiv:2109.10086 , 2021.
[12] S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework:
BM25 and Beyond,” Foundations and Trends in Information Retrieval ,
vol. 3, no. 4, pp. 333–389, 2009. [Online]. Available: https:
//doi.org/10.1561/1500000019
[13] T. Ajayi, Y . Cherivirala, K. Kwon, S. Kamineni, M. Saligane, M. Fayazi,
S. Gupta, C.-H. Chen, D. Sylvester, D. Blaauw et al. , “Fully Au-
tonomous Mixed Signal SoC Design & Layout Generation Platform,”
2020.
[14] W. Lyu, P. Xue, F. Yang, C. Yan, Z. Hong, X. Zeng, and D. Zhou, “An
Efficient Bayesian Optimization Approach for Automated Optimization
of Analog Circuits,” IEEE Transactions on Circuits and Systems I:
Regular Papers , vol. 65, no. 6, pp. 1954–1967, 2017.
[15] M. Fayazi, M. T. Taba, A. Tabatabavakili, E. Afshari, and R. Dreslinski,
“FuNToM: Functional Modeling of RF Circuits Using a Neural Network
Assisted Two-Port Analysis Method,” in 2023 IEEE/ACM International
Conference on Computer Aided Design (ICCAD) . IEEE, 2023, pp. 1–8.
[16] Z. Dong, W. Cao, M. Zhang, D. Tao, Y . Chen, and X. Zhang, “CktGNN:
Circuit Graph Neural Network for Electronic Design Automation,” arXiv
preprint arXiv:2308.16406 , 2023.
[17] K. Settaluri, A. Haj-Ali, Q. Huang, K. Hakhamaneshi, and B. Nikolic,
“AutoCkt: Deep Reinforcement Learning of Analog Circuit Designs,”
in2020 Design, Automation & Test in Europe Conference & Exhibition
(DATE) . IEEE, 2020, pp. 490–495.
[18] M. Fayazi, M. T. Taba, E. Afshari, and R. Dreslinski, “AnGeL: Fully-
Automated Analog Circuit Generator Using a Neural Network Assisted
Semi-Supervised Learning Approach,” IEEE Transactions on Circuits
and Systems I: Regular Papers , vol. 70, no. 11, pp. 4516–4529, 2023.
[19] J. Bhandari et al. , “Masala-CHAI: A Large-Scale SPICE Netlist
Dataset for Analog Circuits by Harnessing AI,” arXiv preprint
arXiv:2411.14299 , 2024, version 3, submitted on 4 February 2025.
[20] J. Bhandari, V . Bhat, Y . He, S. Garg, H. Rahmani, and R. Karri,
“Auto-SPICE: Leveraging LLMs for Dataset Creation via Automated
SPICE Netlist Extraction from Analog Circuit Diagrams,” arXiv
preprint arXiv:2411.14299v1 , 2024, preprint, available online. [Online].
Available: https://arxiv.org/abs/2411.14299v1

[21] Z. Tao, Y . Shi, Y . Huo, R. Ye, Z. Li, L. Huang, C. Wu, N. Bai, Z. Yu,
T.-J. Lin et al. , “AMSNet: Netlist Dataset for AMS Circuits,” in 2024
IEEE LLM Aided Design Workshop (LAD) . IEEE, 2024, pp. 1–5.
[22] Y . Yin, Y . Wang, B. Xu, and P. Li, “ADO-LLM: Analog Design
Bayesian Optimization with In-Context Learning of Large Language
Models,” in Proceedings of the International Conference on Computer-
Aided Design (ICCAD) , 2024, p. 8, arXiv preprint arXiv:2406.18770.
[Online]. Available: https://doi.org/10.48550/arXiv.2406.18770
[23] P. Vijayaraghavan, L. Shi, E. Degan, and X. Zhang, “ CIRCUIT sYNTH :
Leveraging Large Language Models for Circuit Topology Synthesis,” in
2024 IEEE LLM Aided Design Workshop (LAD) . IEEE, 2024, pp. 1–6.
[24] A. Hammoud, C. Goyal, S. Pathen, A. Dai, A. Li, G. Kielian, and
M. Saligane, “Human Language to Analog Layout Using GLayout
Layout Automation Framework,” in Proceedings of the 2024 ACM/IEEE
International Symposium on Machine Learning for CAD , 2024, pp. 1–7.
[25] Z. Chen, J. Huang, Y . Liu, F. Yang, L. Shang, D. Zhou, and X. Zeng,
“Artisan: Automated Operational Amplifier Design via Domain-specific
Large Language Model,” in Proceedings of the 61st ACM/IEEE Design
Automation Conference , 2024, pp. 1–6.
[26] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V . Le,
D. Zhou et al. , “Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models,” Advances in neural information processing systems ,
vol. 35, pp. 24 824–24 837, 2022.
[27] S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y . Cao, and
K. Narasimhan, “Tree of thoughts: Deliberate Problem Solving with
Large Language Models,” Advances in neural information processing
systems , vol. 36, pp. 11 809–11 822, 2023.
[28] C.-C. Chang, Y . Shen, S. Fan, J. Li, S. Zhang, N. Cao, Y . Chen, and
X. Zhang, “LaMAGIC: Language-Model-based Topology Generation for
Analog Integrated Circuits,” arXiv preprint arXiv:2407.18269 , 2024.
[29] H. Zhang, S. Sun, Y . Lin, R. Wang, and J. Bian, “AnalogXpert: Automat-
ing Analog Topology Synthesis by Incorporating Circuit Design Exper-
tise into Large Language Models,” arXiv preprint arXiv:2412.19824 ,
2024.
[30] J. Shen, Z. Chen, J. Zhuang, J. Huang, F. Yang, L. Shang, Z. Bi,
C. Yan, D. Zhou, and X. Zeng, “Atelier: An Automated Analog Circuit
Design Framework via Multiple Large Language Model-Based Agents,”
IEEE Transactions on Computer-Aided Design of Integrated Circuits
and Systems , 2025.
[31] C. Liu, Y . Liu, Y . Du, and L. Du, “LADAC: Large Language Model-
driven Auto-Designer for Analog Circuits,” Authorea Preprints , 2024.
[32] L. Shi, M. Kazda, B. Sears, N. Shropshire, and R. Puri, “Ask-EDA: A
Design Assistant Empowered by LLM, Hybrid Rag and Abbreviation
De-hallucination,” arXiv preprint arXiv:2406.06575 , 2024, version 1,
submitted on 3 June 2024.
[33] B. Jin et al. , “Search-R1: Training LLMs to Reason and Lever-
age Search Engines with Reinforcement Learning,” arXiv preprint
arXiv:2503.09516 , 2025.
[34] P. Jiang, Y . Ren, X. Li, Z. Zhang, Q. Liu, and M. Huang, “DeepRetrieval:
Hacking Real Search Engines and Retrievers with Large Language
Models via Reinforcement Learning,” arXiv preprint arXiv:2503.00223 ,
2025.
[35] S. Kamp, M. Fayazi, Z. Benameur-El, S. Yu, and R. Dreslinski, “Open
Information Extraction: A Review of Baseline Techniques, Approaches,
and Applications,” arXiv preprint arXiv:2310.11644 , 2023.
[36] U. IO, “Unstructured,” 2025, accessed: 2025-03-14. [Online]. Available:
https://github.com/Unstructured-IO/unstructured
[37] Pinecone, “Chunking Strategies,” 2025, accessed: 2025-03-14. [Online].
Available: https://www.pinecone.io/learn/chunking-strategies/
[38] A. Radford et al. , “Learning Transferable Visual Models From Natural
Language Supervision,” arXiv preprint arXiv:2103.00020 , 2021, version
1, submitted on 26 February 2021.
[39] M. Riedler and S. Langer, “Beyond Text: Optimizing RAG with
Multimodal Inputs for Industrial Applications,” arXiv:2410.21943
[cs.CL] , 2024.
[40] V . AI, “V oyage 2,” 2024, accessed: 2025-03-14. [Online]. Available:
https://zilliz.com/ai-models/voyage-2
[41] Anthropic, “Contextual Retrieval,” 2024, accessed: 2025-02-11. [On-
line]. Available: https://www.anthropic.com/news/contextual-retrieval
[42] V . Jindal, S. Bawa, and S. Batra, “A review of ranking approaches
for semantic search on Web,” Information Processing & Management ,
vol. 50, no. 2, pp. 416–425, 2014.
[43] Cohere, “Cohere Rerank,” 2024, accessed: 2025-03-14. [Online].
Available: https://cohere.com/rerank[44] Anthropic, “Claude 3 haiku,” 2024, accessed: 2025-03-14. [Online].
Available: https://www.anthropic.com
[45] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look
Once: Unified, Real-Time Object Detection,” in Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition , 2016,
pp. 779–788.
[46] G. Bradski, A. Kaehler et al. , “OpenCV ,” Dr. Dobb’s Journal of
Software Tools , vol. 3, no. 2, 2000.
[47] Y . Osaki, T. Hirose, N. Kuroki, and M. Numa, “1.2-v Supply, 100-nw,
1.09-v Bandgap and 0.7-v Supply, 52.5-nw, 0.55-v Subbandgap
Reference Circuits for Nanowatt CMOS LSIs,” IEEE Journal of
Solid-State Circuits , vol. 48, no. 6, pp. 1530–1538, 2013.
[48] H. Cao, T. Huang, X. Liu, H. Wang, J. Jin, and W. Wu, “A 5.2 GHz
Trifilar Transformer-Based Class-F23 Noise Circulating VCO with FoM
of 192.6 dBc/Hz,” in 2023 IEEE Asian Solid-State Circuits Conference
(A-SSCC) . IEEE, 2023, pp. 1–3.
[49] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov,
and C. Manning, “HotpotQA: A Dataset for Diverse, Explainable
Multi-hop Question Answering,” arXiv preprint arXiv::1809.09600v1 ,
2018, submitted on 25 September 2018.
[50] Anthropic, “Claude API Pricing,” https://www.anthropic.com/pricing,
2024, accessed: 2025-05-30.
[51] MATLAB, “MATLAB Circuit Simulator,” 2025, accessed: 2025-05-30.
[Online]. Available: https://www.mathworks.com/help/sps/powersys/ug/
building-and-simulating-a-simple-circuit.html
[52] U. Chi-Wa, M.-K. Law, R. P. Martins, and C.-S. Lam, “Sub- µW Auto-
Calibration Bandgap V oltage Reference With 1 σInaccuracy of ±0.12%
Within -40 °c to 120 °c,” IEEE Journal of Solid-State Circuits , vol. 59,
no. 2, pp. 540–550, 2023.