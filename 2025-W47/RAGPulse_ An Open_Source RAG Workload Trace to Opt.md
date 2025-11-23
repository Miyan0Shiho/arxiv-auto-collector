# RAGPulse: An Open-Source RAG Workload Trace to Optimize RAG Serving Systems

**Authors**: Zhengchao Wang, Yitao Hu, Jianing Ye, Zhuxuan Chang, Jiazheng Yu, Youpeng Deng, Keqiu Li

**Published**: 2025-11-17 05:06:47

**PDF URL**: [https://arxiv.org/pdf/2511.12979v1](https://arxiv.org/pdf/2511.12979v1)

## Abstract
Retrieval-Augmented Generation (RAG) is a critical paradigm for building reliable, knowledge-intensive Large Language Model (LLM) applications. However, the multi-stage pipeline (retrieve, generate) and unique workload characteristics (e.g., knowledge dependency) of RAG systems pose significant challenges for serving performance optimization. Existing generic LLM inference traces fail to capture these RAG-specific dynamics, creating a significant performance gap between academic research and real-world deployment. To bridge this gap, this paper introduces RAGPulse, an open-source RAG workload trace dataset. This dataset was collected from an university-wide Q&A system serving that has served more than 40,000 students and faculties since April 2024. We detail RAGPulse's system architecture, its privacy-preserving hash-based data format, and provide an in-depth statistical analysis. Our analysis reveals that real-world RAG workloads exhibit significant temporal locality and a highly skewed hot document access pattern. RAGPulse provides a high-fidelity foundation for researchers to develop and validate novel optimization strategies for RAG systems, such as content-aware batching and retrieval caching, ultimately enhancing the efficiency and reliability of RAG services. The code is available at https://github.com/flashserve/RAGPulse.

## Full Text


<!-- PDF content starts -->

RAGPulse: An Open-Source RAG Workload Trace to Optimize
RAG Serving Systems
Zhengchao Wang, Yitao Hu, Jianing Ye, Zhuxuan Chang, Jiazheng Yu, Youpeng Deng, Keqiu Li
Tianjin University, China
Abstract
Retrieval-Augmented Generation (RAG) is a critical paradigm for
building reliable, knowledge-intensive Large Language Model (LLM)
applications. However, the multi-stage pipeline (retrieve, generate)
and unique workload characteristics (e.g.,knowledge dependency)
of RAG systems pose significant challenges for serving performance
optimization. Existing generic LLM inference traces fail to capture
these RAG-specific dynamics, creating a significant performance
gap between academic research and real-world deployment.
To bridge this gap, this paper introduces RAGPulse, an open-
source RAG workload trace dataset. This dataset was collected from
an university-wide Q&A system serving that has served more than
40,000 students and faculties since April 2024. We detail RAGPulse’s
system architecture, its privacy-preserving hash-based data format,
and provide an in-depth statistical analysis. Our analysis reveals
that real-world RAG workloads exhibit significant temporal local-
ity and a highly skewed hot document access pattern. RAGPulse
provides a high-fidelity foundation for researchers to develop and
validate novel optimization strategies for RAG systems, such as
content-aware batching and retrieval caching, ultimately enhancing
the efficiency and reliability of RAG services. The code is available
at https://github.com/flashserve/RAGPulse.
1 Introduction
Large Language Models (LLMs) [ 1–3] have demonstrated power-
ful capabilities across a spectrum of natural language processing
tasks. However, their practical application is often hindered by two
inherent limitations:knowledge cutoff, where LLM models are un-
aware of information post-dating their training, andhallucination,
where LLM models confidently generate incorrect, nonsensical, or
fabricated information.
Retrieval-Augmented Generation (RAG) [ 4–6] is an advanced
paradigm designed to address these challenges. The core idea of
RAG is to synergize the powerful reasoning and generative capa-
bilities of LLMs with the real-time, factual accuracy of external
knowledge bases. As shown in Figure 1, instead of directly tasking
LLM with an answer, the RAG workflow employs a multi-stage
pipeline:
•Retrieve: Once the RAG service system receives a request,
an embedding model vectorizes the user query and retrieves
the most semantically relevant knowledge chunks from a
large-scale vector database (e.g.,FAISS [7]).
•Augment: These retrieved chunks are then concatenated as
"context" with the original user query and a system prompt.
•Generate: Finally, this "augmented prompt" is submitted to
the LLM, guiding it to produce a factually-grounded and
accurate output based on the provided real-time context.
∗Corresponding author: Yitao Hu (email: yitao@tju.edu.cn).
User Query Vector DatabaseRetrieve
LLM Output1 2
34
50.20.60.40.80.20.70.30.40.70.10.20.9
0.10.80.50.40.70.2Embedding
D1D2D3...Context(Chunks)System PromptAugment
GenerationAugmented PromptMost RelevantConcatenate
Submitted
6Figure 1: Workflow of RAG.
By decoupling knowledge storage from the model’s reasoning
capabilities, the RAG paradigm significantly enhances the reliabil-
ity, timeliness, and traceability of LLM applications. It has rapidly
become the standard for building enterprise-grade Q&A systems,
intelligent document analysis tools, and trustworthy AI assistants
(such as the university-wide policy Q&A service in this study).
However, despite its significance, optimizing the performance
and cost of RAG serving systems presents a new set of challenges.
Unlike pure LLM inference, the performance bottlenecks in RAG
systems are composite. Performance is not only dependent on the
generation stage (e.g.,KV cache efficiency) but is also heavily influ-
enced by the retrieval stage (e.g.,vector database latency) and the
complex interactions between these stages (e.g.,adaptive batching).
We find that existing generic LLM inference traces fail to capture
the unique workload dynamics specific to RAG [ 8,9]. As our sub-
sequent analysis reveals (§3.2), real-world RAG workloads exhibit
highly skewed knowledge chunk access frequencies and significant
inter-request temporal locality (i.e.,ahot documentphenomenon).
These characteristics are critical for designing efficient retrieval
caching and content-aware batching strategies, yet they are entirely
absent from generic LLM traces.
This performance gap between research tools (generic traces) and
practical deployments (RAG systems) severely hinders optimization
research for RAG serving. Academia and industry urgently require
a public benchmark dataset that accurately reflects real-world RAG
workload characteristics, which is the primary focus of this paper.
2 Preliminary and Motivation
2.1 Why We Need a RAG-Specific Trace?
Existing generic LLM inference traces exhibits significant limita-
tions when applied to RAG serving system. This inadequacy stems
from the unique characteristics of RAG workloads, which differ
fundamentally from pure LLM inference and necessitate the devel-
opment of RAG-specific traces.
First, RAG possesses inherent multi-stage pipeline complexity.
A RAG request must sequentially flow through distinct phases, in-
cluding retrieval, re-ranking, generation, and so on [ 5,6,10]. This
architecture dictates that system latency, resource consumption,
1arXiv:2511.12979v1  [cs.LG]  17 Nov 2025

Zhengchao Wang, Yitao Hu, Jianing Ye, Zhuxuan Chang, Jiazheng Yu, Youpeng Deng, Keqiu Li
1000 2000 3000 4000 5000 6000 7000 8000
Input Length (tokens)0.00.20.40.60.81.0Cumulative Probability
0 350 700 1050 1400 1750
Output Length (tokens)0.00.20.40.60.81.0
Figure 2: CDF of Input and Output Token Lengths in RAG-
Pulse.
and caching behaviors are governed by complex inter-stage inter-
actions. Generic LLM traces, which typically focus only on the
generation phase, fail to characterize these composite effects.
Second, RAG performance demonstrates strong dependencies
on the knowledge base and query patterns. Its workload dynamics,
such as query similarity, embedding cache hit rates, and retrieval
latency, fluctuate significantly with real-world user behavior. Fur-
thermore, in many practical applications, conseutive queries often
exhibit significant inter-request contextual dependencies. Captur-
ing these dynamics and inter-request correlations is critical for opti-
mizing KV cache reuse, batching, and scheduling strategies [ 11,12].
Moreover, RAG-specific traces are fundamental to enabling the
co-optimization of retrieval and generation components. Tradi-
tional approaches that treat retrieval and generation as indepen-
dent black boxes restrict systemic design potential. In contrast, RAG
trace data can inform more sophisticated system designs, such as
dynamic retrieval caching or query-adaptive batching [13].
Finally, existing RAG trace data, often derived from synthetic
or genetic chat workloads, creates a performance gap between
RAG optimization research and practical deployment. Evaluating
optimization strategies using non-representative traces leads to a
disconnect between research findings and real-world system perfor-
mance. This discrepancy poses severe challenges to the efficiency,
stability, and reliability of RAG serving systems.
Therefore, a real-world RAG trace is needed to advance the
practical application of RAG research.
2.2 What We Could Do with a RAG Trace?
A RAG-specific trace dataset that reflects realistic workload charac-
teristics would be a critical tool for research and optimization. It
would unlock several key pathways for investigation and applica-
tion in both academia and industry:
Precise Performance Bottleneck Analysis.The trace would
allow researchers to conduct fine-grained breakdowns of end-to-
end latency in RAG systems, precisely quantifying the respective
latency contributions and bottlenecks of the retrieval, reranking,
and generation stages.
Informed Optimization of Scheduling and Caching Poli-
cies.It would provide a realistic foundation for evaluating and
designing advanced system strategies, including KV cache reuse
efficiency, adaptive batching algorithms, and retrieval caching, all
under practical inter-request correlation patterns.
High-Fidelity Workload Modeling and Benchmarking.The
data would provide the foundation for building high-fidelity RAGworkload models, simulators, and standardized benchmarks, thereby
ensuring the validity and reproducibility of system evaluations.
Study of Emerging Application Behaviors.It would support
the exploration of advanced RAG applications, such as tracking
and analyzing how real-world workloads evolve from simple Q&A
patterns into more complex, agent-like reasoning and multi-turn
task-solving behaviors.
These applications show how a high-fidelity RAG trace would
serve as a key enabler for bridging the gap between theoretical RAG
systems research and practical deployment, driving advancement
in the field.
3 Introduction to RAGPulse
To bridge the gap between RAG systems research and practical
deployment, we introduce RAGPulse, a RAG workload trace dataset
collected from a real-world deployment. The dataset originates
from an university-wide policy Q&A system, which has been in
continuous operation since April 2024, serving over 40,000 students
and faculties. We are now open-sourcing the core of RAGPulse as a
real-world RAG trace. We are committed to continuously updating
this dataset as our service evolves, with plans to release further
RAG and future Agent traces to reflect evolving workload patterns.
3.1 Data Format and Content
Each record in the RAGPulse trace captures key system-level run-
time information for a RAG request. The dataset we are releasing
at this time covers 7,106 request records sampled from one week of
the system’s operation. As shown in Figure 2, the input and output
token lengths of the RAGPulse workload demonstrate a clear con-
centration trend. Specifically, input token lengths primarily cluster
around 3500 tokens, while output token are concentrated around
500 tokens. Furthermore, as detailed in List 1, each record in the
main trace file contains the following key fields:
1{
2" timestamp ": "27" ,
3" input_length ": 3861 ,
4" output_length ": 127 ,
5" hash_ids ": {
6" sys_prompt ": [8325 , 8326 , 11575] ,
7" passages_ids ": [6123 , 7239 , 6124 , 1167 , 7250 ,
5448] ,
8" history ": [15215] ,
9" web_search ": [20319 , 20320] ,
10" user_input ": [23648]
11},
12" session_id ": "1758081660427 - xa8rbsd2uco1 "
13}
Listing 1: Request Component Sample.
•timestamp : It denotes the request submission time, which
is calculated in seconds starting from the trace’s initial
moment (12:00:00).
•input_length : The total input token length of the request.
This is the sum of the token lengths of all components
included in the request (e.g.,system prompt, retrieved doc-
ument, chat history).
•output_length : The total token length of the model-generated
output.
2

RAGPulse : An Open-Source RAG Workload Trace to Optimize RAG Serving Systems
0.0000.0250.0500.0750.100QPS
00:00 00:00 00:00 00:00 00:00 00:00 00:00 00:00
Time (utc)05101520T okens/s
Figure 3: Throughput over time in RAGPulse.
•hash_ids : A comprehensive collection of hash identifiers
representing every component of the request’s input. This
includes unique IDs for the the system prompt, all retrieved
documents, user chat history, external web search results,
and user’s question.
•session_id : The conversation identifier to which the re-
quest belongs.
To rigorously protect user privacy, all original textual content has
been removed and replaced with remapped hash_ids. This approach
ensures complete anonymization while preserving all the structural
and temporal characteristics required for research, making it highly
suitable for system performance and scheduling policy analysis.
3.2 Dataset Features and Statistical Analysis
A preliminary statistical analysis of the RAGPulse trace data reveals
four key characteristics that are highly instructive for RAG serving
system design.
3.2.1 Periodic Load Fluctuation.
RAGPulse records the arrival time of each request. As illustrated
in Figure 3, the workload exhibits clear periodicity. The request
volume peaks during daytime (working hours) and decreases signifi-
cantly at night. This pattern is highly consistent with the real-world
user behavior of serving systems [2, 8].
3.2.2 Dynamic Input Composition.
As shown in Figure 4, we further analyzed the composition of
the request input tokens. The analysis reveals that the proportional
contribution of different components is not static, but varies dy-
namically with the total input length.
Specifically, for shorter requests (with a total input length of
less than 2000 tokens), the system_prompt is the dominant com-
ponent, accounting for over 70% of the tokens. In these cases, the
contribution from retrieved passages and history is minimal. How-
ever, as the total input length increases (exceeding 3000 tokens),
the load composition shifts significantly. The token contribution
from retrieved passages and history (dialogue history) increases
substantially, collectively approaching 40% of the total.
This dynamic input composition provides a critical insight for
RAG system optimization: the system’s processing overhead (such
2000 3000 4000 5000 6000
Input Length (Binned)0.00.20.40.60.81.0Average Component ProportionSys Prompt Passage History Web Search User Input Other
1060 1080 1100 1120 11400.960.981.00Figure 4: Proportion of Input Components Across Different
Input Lengths in RAGPulse.
as the memory and computation for context processing) is not uni-
form, but is highly heterogeneous and dependent on the request’s
length and type.
3.2.3 Text Usage Frequency.
When analyzing the usage of the hash_ids field within the trace,
we observed a significant phenomenon: the reuse rate of hash_ids
corresponding to retrieved document from the vector database is
far higher than that of other components.
Specifically, as shown in Figure 5(a), a small subset of core knowl-
edge chunks ("hot" documents) is frequently retrieved and refer-
enced by a large volume of different user requests, while the vast
majority of chunks are accessed much less often. This highly skewed
usage distribution exhibits a typical "long-tail effect." This finding
provides strong evidence that Retrieval Caching, which targets
these retrieved content chunks, is a highly valuable optimization
direction for RAG systems.
3.2.4 Document Overlap between Requests.
Further analysis of the request time-series reveals significant
temporal locality. As shown in Figure 5(b), within any given time
window, a large proportion of concurrently arriving requests tend
to access the same or highly similar hash_ids for knowledge chunks.
This phenomenon results in high inter-request document over-
lap. This confirms our hypothesis regarding strong inter-request
contextual dependencies in real-world RAG workloads. This char-
acteristic indicates that the system could gain significant benefits
from two strategies: (1) Content-aware batching of requests that
access the same documents; and (2) the corresponding KV Cache for
these "hot" documents will have exceptionally high reuse potential.
4 Dataset Usage and Applications
To validate the effectiveness of our proposed RAGPulse dataset for
evaluating system performance in real-world environments, we
have constructed a dedicated benchmarking platform. This chapter
provides a detailed description of the platform’s specific implemen-
tation, the composition and usage methodology of the dataset, and
the performance evaluation metrics employed.
3

Zhengchao Wang, Yitao Hu, Jianing Ye, Zhuxuan Chang, Jiazheng Yu, Youpeng Deng, Keqiu Li
0 50 100 150 200 250 300 350
Passage Frequency0.60.70.80.91.0Cumulative Probability
(a) CDF of Passage Frequency12:00 12:00 12:00 12:00 12:00 12:00 12:00 12:00
Time (Hourly Bins)0.00.20.40.60.8Overlap Ratio
(b)Document Overlap Across Requests
Figure 5: Partially Unique Characteristics of RAG Traces in
RAGPulse.
4.1 Implementation
We constructed a prototype system for an online RAG service to
simulate real-world applications. The system’s core is built upon
vLLM [ 14], a widely adopted open-source inference framework. To
emulate a typical RAG workflow, the system integrates an end-to-
end pipeline, encompassing user request reception, text processing,
and final result generation. All experiments were conducted on a
server equipped with a single NVIDIA A800 GPU (80GB VRAM).
For the Generator component of the RAG pipeline, we deployed
Qwen2.5-14B as the LLM model.
4.2 Scaling RAGPulse to Any Scale
RAGPulse’s trace data is characterized by a significant temporal
span1. This property provides researchers with high flexibility, al-
lowing them to slice or scale the trace data according to specific
experimental goals. For example, a high-traffic period (e.g.,one
hour) can be isolated to benchmark the system’s peak concurrent
processing capabilities. Conversely, the entire long-term trace can
be used to assess the long-term efficacy of caching policies, such as
the KV cache management. Moreover, an analysis of throughput
fluctuations in the trace enables a deeper comprehension of the
system’s behavioral patterns under varying loads.
4.3 Dataset Text Decryption
To protect user privacy, all original text within the dataset has been
replaced by hash IDs. Nonetheless, for system-level simulations,
especially for the evaluation of components reliant on specific text
content like KV cache, the characteristics of the original text (such
as length) are essential. Therefore, researchers must employ a text
reconstruction strategy when using this dataset. The central princi-
ple of this strategy is to generate a unique, random-content place-
holder text for every unique hash ID, ensuring the token length is
strictly identical to that of the original text. It is imperative that the
same hash ID maps consistently to the exact same placeholder text
in all requests, thereby guaranteeing the simulation’s consistency
and validity.
4.4 Metrics and Setups
To comprehensively evaluate the performance of the RAG serving
system, we adopted two key metrics widely recognized in the field
of LLM serving:
1Currently, we are releasing one week’s data, with plans for longer-period data in the
future.
User Query
Nginx Python APIsWeb Service
TXT PDFDocuments
... JPG ...
Python Extract OCR Extract
Python Program Prompt & LLM
Text Post-processingRaw TextD1D2...
Processed Text0.20.60.40.80.20.7
0.30.40.70.10.20.9
0.10.80.50.40.70.2Vector EmbeddingAgent LLM
Multimodal
Image
Audio
...
Final PromptWeb Search
Web Page1
Web Page2
Web Page3
...RAG
D1
D2
D3
...Extra Tool
API Call 
Code
Interaction
...Multi
Hop
Global Database
Partitioned Database
User’s private Database
...Vector Database SystemOnline
OfflineAnswer
Inference & OutputFigure 6: Twen’s System Architecture.
•Time To First Token (TTFT): This measures the total time
elapsed from the moment the system receives a request
until the first output token is generated and returned. It is
the core benchmark for system responsiveness and user-
perceived latency.
•Time Per Output Token (TPOT): This measures the average
time required to generate each subsequent token after the
first token has been produced. This metric primarily reflects
the system’s throughput and processing efficiency during
the generation phase.
4.5 Demo Use
The RAGPulse benchmark utilizes a client-server architecture to
implement its workload replay mechanism.
The server-side component, built on the vLLM framework, man-
ages the core inference tasks. It receives client requests via an
OpenAI-compatible API and supports both streaming and non-
streaming generation modes. This capability is essential for cap-
turing key performance metrics, including TTFT and TPOT. The
server employs an asynchronous, concurrent design to maintain
stability and high throughput under heavy load.
The client-side component acts as the load generator, driving
the entire benchmark execution. Its three-stage workflow begins
with Data Preprocessing, where the client reads the trace file and
reconstructs text inputs by generating random token sequences
that strictly match the original token lengths for each hash ID.
Next, the Request Scheduling module dispatches the processed
requests to the server, strictly adhering to the original timestamps
from the trace file. To enhance flexibility, a time-scaling factor is
introduced, allowing researchers to accelerate or decelerate the
workload replay. Finally, the Metric Collection module records the
per-request performance data, such as TTFT and TPOT, and ensures
results are persistently stored for subsequent analysis.
Please refer to https://github.com/flashserve/RAGPulse/blob/
main/example/single_online_instance/README.md for details.
4

RAGPulse : An Open-Source RAG Workload Trace to Optimize RAG Serving Systems
TXT ... PDFJPG ...Various Documents
LLMPython ProgramOCR
Raw TextD1D2 ... D3D4
Normalized TextNormalization
D’1D’2...Split
Doc Chunk
Chunk1Chunk2 ...
0.20.60.40.80.20.7
0.30.40.70.10.20.9
0.10.80.50.40.70.2Vector Embedding
Vector Database System
Figure 7: Offline Architecture in Twen.
5 Data Source: Twen System Architecture
The RAGPulse trace dataset, which is sampled from Twen.ai, an
university-wide Q&A RAG system that has served over 40,000
students and faculties since April 2024. This section details Twen’s
system architecture to provide context for how the trace data was
generated.
5.1 System Overview
The core design philosophy of Twen is the separation of retrieval
and generation. To achieve this, Twen adopts a Python-based mi-
croservice architecture (FastAPI, Docker) to decouple the CPU-
intensive retrieval tasks from the GPU-intensive generation tasks.
The overall system architecture is illustrated in Figure 6:
•Generator Model: We use Qwen3-235B-A22B-Instruct-2507
model for main Q&A and agent interaction functions [ 15].
•Embedding Model: We use the locally deployed open-source
model infgrad/stella-large-zh-v3-1792d for vectorizing all
knowledge base documents.
•Orchestration: LangChain is used to build and orchestrate
prompt templates, model call chains, and retrieval flows [ 16].
•Vector Database: Qdrant [ 17] is used to store and index
knowledge base vectors with a tag-based structure, orga-
nized into global, tagged, and user-personal databases.
•Document Maintenance: The system uses Python programs
and LLM for content extraction, format standardization, and
automatic segmentation of files.
5.2 Offline Stage
As shown in Figure 7, the core task of the offline stage is to construct
the high-quality vector knowledge base.
The knowledge base corpus (university policies and information)
is acquired via three channels: (1) Manual submission by faculty
and department leaders; (2) An upload system for authorized ad-
ministrators; and (3) Targeted web crawlers that periodically pull
data from public department websites.
Based on their storage formats, the raw documents are catego-
rized into two types: the first consists of files like TXT, from which
content can be easily extracted by Python programs; the second
includes files like JPG, where content is not explicitly stored and
LLMMultimodal
Image
Audio
Text
...
RAG
D1
D2
D3
...Extra Tool
API Call 
Code
Interaction
...Web Search
Web Page1
Web Page2
Web Page3
...Multi-Hop
Iterative
Final PromptFigure 8: Online Architecture in Twen.
requires extraction using LLM-based OCR. The extracted raw con-
tent often has messy formatting. To address this, we developed
an specialized agent for text normalization. The normalized text
is then intelligently segmented using another dedicated LLM2to
ensure semantic integrity, resulting in high-quality corpus chunks
suitable for building the vector database.
5.3 Online Stage
As shown in Figure 8, the online stage handles user requests and
generates the RAGPulse trace records in the process.
We have developed our agent system, a comprehensive frame-
work where a central large language model, which we refer to as
"the Agent LLM, " acts as an autonomous controller. When a request
arrives, this Agent LLM autonomously plans the optimal processing
strategy based on user information and prompts. It can interpret
queries from multimodal perspectives (images, audio, text,etc.) and
dynamically orchestrates a suite of tools—including web search,
RAG, and a series of predefined external tools (API calls, program-
ming,etc.) to comprehensively solve problems. Furthermore, the
system architecture supports proactive self-reflection: the Agent
LLM first assesses its own response, and if it deems the current
answer insufficient, it will iteratively employ various tools in a
multi-hop manner until reaching the final response.
The system calls the Qwen3-235B model asynchronously in
streaming mode using LangChain’s AsyncIteratorCallbackHandler .
During this call, the Langfuse CallbackHandler is activated, cap-
turing the complete call metadata (including latency, token lengths,
user_id, session_id, and retrieved hash_ids). This captured data is
sent to the Langfuse database, forming the RAGPulse trace dataset
analyzed in this study.
6 Conclusion
RAG has emerged as the standard paradigm for building trust-
worthy, high-performance LLM applications. However, its unique,
composite architecture (retrieval, reranking, generation) introduces
2We use a variety of LLM models ranging from 7B to 70B for different sub-tasks for
cost efficiency.
5

Zhengchao Wang, Yitao Hu, Jianing Ye, Zhuxuan Chang, Jiazheng Yu, Youpeng Deng, Keqiu Li
unprecedented challenges for system optimization. The central the-
sis of this paper is that existing generic LLM traces fail to capture
the RAG-specific workload dynamics, creating a significant perfor-
mance gap between academic research and practical deployment.
To bridge this gap, we design and open-source RAGPulse, a
publicly available RAG workload trace dataset derived from an
university-wide Q&A service that has served over 40,000 students
and faculties since April 2024. Through an in-depth analysis of this
dataset, we have quantified critical characteristics of real-world
RAG workloads: a highly skewed hot document access pattern and
significant inter-request temporal locality. These findings provide
clear empirical support and research direction for RAG system op-
timizations, such as content-aware batching and retrieval caching.References
[1]Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li,
and Wei Lin. Llumnix: Dynamic scheduling for large language model serving. In
18th USENIX symposium on operating systems design and implementation (OSDI
24), pages 173–191, 2024.
[2]Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang,
Yongwei Wu, Weimin Zheng, and Xinran Xu. Mooncake: Trading more storage
for less computation — a KVCache-centric architecture for serving LLM chatbot.
In23rd USENIX Conference on File and Storage Technologies (FAST 25), pages
155–170, Santa Clara, CA, February 2025. USENIX Association.
[3]Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang,
Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi,
Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao
Jia. Specinfer: Accelerating large language model serving with tree-based specu-
lative inference and verification. In Rajiv Gupta, Nael B. Abu-Ghazaleh, Madan
Musuvathi, and Dan Tsafrir, editors,Proceedings of the 29th ACM International
Conference on Architectural Support for Programming Languages and Operating
Systems, Volume 3, ASPLOS 2024, La Jolla, CA, USA, 27 April 2024- 1 May 2024,
pages 932–949. ACM, 2024.
[4]Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Ganesh Ananthanarayanan,
Ravi Netravali, and Junchen Jiang. Ragserve: Fast quality-aware rag systems
with configuration adaptation.arXiv preprint arXiv:2412.10543, 2024.
[5] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad
Shoeybi, and Bryan Catanzaro. Rankrag: unifying context ranking with retrieval-
augmented generation in llms. InProceedings of the 38th International Conference
on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2025.
Curran Associates Inc.
[6] Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. Turborag:
Accelerating retrieval-augmented generation with precomputed kv caches for
chunked text.arXiv preprint arXiv:2410.07590, 2024.
[7] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. The
faiss library. 2024.
[8] Yuxin Wang, Yuhan Chen, Zeyu Li, Xueze Kang, Yuchu Fang, Yeju Zhou, Yang
Zheng, Zhenheng Tang, Xin He, Rui Guo, Xin Wang, Qiang Wang, Amelie Chi
Zhou, and Xiaowen Chu. BurstGPT: A real-world workload dataset to optimize
llm serving systems. InProceedings of the 31st ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V.2 (KDD ’25), Toronto, ON, Canada, 2025.
ACM.
[9]shareAI. Sharegpt-chinese-english-90k bilingual human-machine qa dataset.
https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k, 2023.
[10] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-
rag: Learning to retrieve, generate, and critique through self-reflection. InThe
Twelfth International Conference on Learning Representations, ICLR 2024, Vienna,
Austria, May 7-11, 2024. OpenReview.net, 2024.
[11] Shubham Agarwal, Sai Narayan Sundaresan, Subrata Mitra, Debabrata Mahapa-
tra, Archit Gupta, Rounak Sharma, Nirmal Joshua Kapu, Tong Yu, and Shiv Kumar
Saini. Cache-craft: Managing chunk-caches for efficient retrieval-augmented
generation.Proceedings of the ACM on Management of Data, 3:1 – 28, 2025.
[12] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin
Jin. Ragcache: Efficient knowledge caching for retrieval-augmented generation.
arXiv preprint arXiv:2404.12457, 2024.
[13] Zheng Wang, Shu Teo, Jieer Ouyang, Yongjun Xu, and Wei Shi. M-RAG: Reinforc-
ing large language model performance through retrieval-augmented generation
with multiple partitions. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar,
editors,Proceedings of the 62nd Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pages 1966–1978, Bangkok, Thailand,
August 2024. Association for Computational Linguistics.
[14] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory
management for large language model serving with pagedattention. InPro-
ceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles,
2023.
[15] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji
Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men,
Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng
Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang,
Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng
Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang
Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report.
arXiv preprint arXiv:2309.16609, 2023.
[16] Langchain. https://python.langchain.com/docs/introduction/, 2024.
[17] Qdrant. https://qdrant.org.cn/, 2025.
6