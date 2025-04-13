# Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents

**Authors**: Despina Tomkou, George Fatouros, Andreas Andreou, Georgios Makridis, Fotis Liarokapis, Dimitrios Dardanis, Athanasios Kiourtis, John Soldatos, Dimosthenis Kyriazis

**Published**: 2025-04-07 22:02:19

**PDF URL**: [http://arxiv.org/pdf/2504.05527v1](http://arxiv.org/pdf/2504.05527v1)

## Abstract
This paper introduces a novel integration of Retrieval-Augmented Generation
(RAG) enhanced Large Language Models (LLMs) with Extended Reality (XR)
technologies to address knowledge transfer challenges in industrial
environments. The proposed system embeds domain-specific industrial knowledge
into XR environments through a natural language interface, enabling hands-free,
context-aware expert guidance for workers. We present the architecture of the
proposed system consisting of an LLM Chat Engine with dynamic tool
orchestration and an XR application featuring voice-driven interaction.
Performance evaluation of various chunking strategies, embedding models, and
vector databases reveals that semantic chunking, balanced embedding models, and
efficient vector stores deliver optimal performance for industrial knowledge
retrieval. The system's potential is demonstrated through early implementation
in multiple industrial use cases, including robotic assembly, smart
infrastructure maintenance, and aerospace component servicing. Results indicate
potential for enhancing training efficiency, remote assistance capabilities,
and operational guidance in alignment with Industry 5.0's human-centric and
resilient approach to industrial development.

## Full Text


<!-- PDF content starts -->

Bridging Industrial Expertise and XR with
LLM-Powered Conversational Agents
Despina Tomkou∗, George Fatouros∗‡, Andreas Andreou†, Georgios Makridis‡
Fotis Liarokapis†, Dimitrios Dardanis‡, Athanasios Kiourtis‡, John Soldatos∗, Dimosthenis Kyriazis‡
∗Innov-Acts Ltd. , Nicosia, Cyprus
{dtomkou, gfatouros, jsoldat }@innov-acts.com
†CYENS Centre of Excellence , Nicosia, Cyprus
{a.andreou, f.liarokapis }@cyens.org.cy
‡University of Piraeus , Piraeus, Greece
{gfatouros, gmakridis, ddardanis, kiourtis, dimos }@unipi.gr
Abstract —This paper introduces a novel integration of
Retrieval-Augmented Generation (RAG) enhanced Large Lan-
guage Models (LLMs) with Extended Reality (XR) technologies
to address knowledge transfer challenges in industrial environ-
ments. The proposed system embeds domain-specific industrial
knowledge into XR environments through a natural language
interface, enabling hands-free, context-aware expert guidance for
workers. We present the architecture of the proposed system con-
sisting of an LLM Chat Engine with dynamic tool orchestration
and an XR application featuring voice-driven interaction. Per-
formance evaluation of various chunking strategies, embedding
models, and vector databases reveals that semantic chunking,
balanced embedding models, and efficient vector stores deliver
optimal performance for industrial knowledge retrieval. The
system’s potential is demonstrated through early implementation
in multiple industrial use cases, including robotic assembly,
smart infrastructure maintenance, and aerospace component
servicing. Results indicate potential for enhancing training ef-
ficiency, remote assistance capabilities, and operational guidance
in alignment with Industry 5.0’s human-centric and resilient
approach to industrial development.
Index Terms —eXtended Reality, Large Language Models,
Retrieval-Augmented Generation, Conversational AI, Remote
Assistance, Knowledge Management, Smart Manufacturing
I. I NTRODUCTION
Industry 5.0 is redefining industrial operations by empha-
sizing sustainable, resilient and human-centric systems [1].
It recognizes the role of industry not only in economic
growth but also in achieving broader societal objectives by
prioritizing worker well-being. At the same time, it advocates
for the development of agile business processes capable of
withstanding geopolitical disruptions and natural crises [2].
Despite rapid technological progress, industries still face
critical challenges related to knowledge transfer and the avail-
ability of expert support [3]. A shortage of skilled profes-
sionals, combined with geographical constraints that often
necessitate on-site presence, results in significant production
bottlenecks across various sectors [4]. For example, when
Samsung relocated one of its major production facilities, the
company encountered a substantial decline in skilled labor due
to difficulties in retaining and attracting specialized personnel.
As a result, experts had to be flown in from other regionsfor essential maintenance and operational support, leading to
costly downtime and logistical complications [5]. This case
illustrates the pressing need for efficient, remotely accessible
knowledge-sharing solutions—particularly in scenarios involv-
ing maintenance, assembly and troubleshooting, where timely
professional intervention is crucial.
Extended Reality (XR) technologies are increasingly used
in industrial contexts to support training, problem-solving and
remote assistance. By offering immersive simulations of real-
world environments and equipment, XR enables safe, hands-
on instruction without requiring access to physical machinery
and allows experts to guide workers remotely. However, many
existing XR applications operate in isolation from domain-
specific data, which limits their effectiveness—especially in
the absence of on-demand expert availability.
These limitations are evident across several core use cases:
•Training: XR provides safe, naturalistic environments
for practicing complex tasks without risking damage
to real equipment or disrupting operations. However,
engagement is typically restricted to predefined teaching
modules that cannot easily adapt to the specific needs or
experience of individual trainees.
•Remote Assistance: XR enables off-site experts to pro-
vide visual guidance during on-site operations but lacks
deep integration with structured industrial documentation.
•Real-Time Assistance: Augmented Reality (AR) overlays
can guide workers step-by-step in real-time, yet often
miss the dynamic reasoning and flexibility required in
difficult, unexpected situations.
The convergence of Large Language Models (LLMs)
and XR technologies presents a promising avenue to over-
come these limitations. Recent advancements in generative
AI—exemplified by models such as OpenAI’s GPT-4—have
demonstrated strong capabilities in natural language under-
standing, generation and question answering. With appropriate
prompt engineering, these models can be tailored to user-
specific needs [6]. Furthermore, by incorporating Retrieval-
Augmented Generation (RAG) techniques [7], LLMs can
access and utilize external data sources, thereby improvingarXiv:2504.05527v1  [cs.CL]  7 Apr 2025

factual accuracy and enabling domain-specific expertise [8].
When combined, the immersive capabilities of XR and the
contextual intelligence of RAG-enhanced LLMs can yield ad-
vanced support systems for training and operational guidance
that are both interactive and adaptable to individual users’
proficiency and learning pace.
This paper introduces a novel system that embeds domain-
specific industrial knowledge into XR environments via a
RAG-enhanced LLM-powered conversational engine. At its
core, the system incorporates technical documentation into
XR settings through a natural language interface. To en-
sure high-quality retrieval performance, we evaluate various
chunking strategies, embedding models and vector databases
to optimize the relevance and speed of information access.
An implementation framework is also developed, enabling
voice-based interaction with this embedded knowledge in XR
environments.
The system is applied in multiple industrial scenarios,
including robotic assembly, smart water pipe maintenance,
aircraft component servicing and edge device installation [9].
These applications demonstrate how the proposed approach
enhances XR-based support with real-time, contextual and
personalized knowledge delivery, supporting a more efficient
and human-centric collaboration.
The remainder of this paper is organized as follows: Sec-
tion II reviews related work on LLM-powered conversational
assistants and XR applications in industry. Section III presents
the system architecture along with preliminary evaluation
results. Section IV discusses real-world use cases and imple-
mentations. Section V concludes the paper with key insights
and directions for future research.
II. B ACKGROUND AND RELATED WORK
A. LLMs as Conversational Assistants
The integration of LLMs with XR technologies can reshape
industrial and manufacturing applications by providing intel-
ligent, context-aware conversational interfaces. Transformer-
based models such as GPT-4, Claude, Gemini, and LLaMA
have proven highly effective across diverse AI applications,
including code generation, search enhancement and decision-
making support due to their large-scale pretraining and zero-
shot generalization capabilities [10], [11].
However, despite these strengths, LLMs exhibit critical
limitations when applied to domain-specific contexts like
industrial training and maintenance. Issues such as hallucina-
tions, outdated or imprecise responses, and poor adaptability
can pose significant risks in precision-critical industrial envi-
ronments [12]. To address these challenges, domain-specific
knowledge injection strategies have been proposed, catego-
rized into static embeddings, modular adapters, prompt en-
gineering, and dynamic retrieval [13]. Among these, dynamic
retrieval—implemented through RAG is recognized as particu-
larly effective in evolving knowledge domains, allowing LLMs
to dynamically access external documentation and manuals
through vector databases.Recent advancements, such as Agentic RAG, further en-
hance dynamic retrieval by employing autonomous AI agents
capable of advanced reasoning, contextual memory manage-
ment, and adaptive retrieval mechanisms [14]. These agents
break down complex user queries into actionable steps, thus
providing consistent, personalized and accurate guidance [15].
Real-world deployments, including an LLM-based assistant
for training supply chain employees, have demonstrated sig-
nificant cost reductions and improved user satisfaction through
the provision of immediate and clear instructions. [16].
In manufacturing, LLMs can address labor shortages and
skill gaps by providing interactive mentoring, real-time as-
sistance, predictive maintenance, and regulatory support [17].
These capabilities enable less experienced workers to perform
complicated tasks proficiently and safely, highlighting the
strategic role of LLMs in industrial settings.
B. AR and VR in Industrial Applications
AR and Virtual Reality (VR) significantly enhance industrial
applications by improving worker interaction with equipment
and engagement during operations. Despite their potential, the
adoption of these technologies remains limited because of high
development costs and the absence of standardized platforms.
[18], [19].
Nevertheless, AR/VR technologies substantially enhance
user participation and intuitive interaction with sophisticated
industrial systems. Applications such as digital twins, re-
mote collaboration, custom manufacturing, and virtual training
scenarios demonstrate considerable value in improving the
learning experience and operational throughput. [20], [21].
Specifically, AR and VR simplify advanced tasks such
as assembly, error detection, and quality assurance through
visual guidance and real-time feedback. Designers and en-
gineers utilize these technologies to rapidly prototype prod-
ucts in three-dimensional virtual environments, accelerating
product development cycles [22]. Training simulations offered
by AR/VR platforms enable workers to perform intricate
procedures safely in controlled virtual settings. This allows
for repeated practice, leading to deeper comprehension and
improved task proficiency. [23], [24].
Furthermore, XR effectively supports remote assistance
by enabling geographically dispersed professionals and field
workers to collaborate and interact with equipment in virtual
real-time settings. [25]. As a result, travel needs are reduced
and response times for expert interventions are improved.
Building upon this existing research, our work integrates
RAG-enhanced LLM conversational agents directly into XR
environments. Unlike traditional text-based assistants, our sys-
tem leverages XR interfaces, delivering hands-free, visual, and
procedural guidance tailored to user expertise and needs.
III. A RCHITECTURE
The proposed system integrates RAG-enhanced LLMs with
XR to deliver domain-specific knowledge within industrial
environments as illustrated in Fig. 1.

Fig. 1. Architectural Overview of RAG-Enhanced LLM with XR Integra-
tion for Industrial Environments. The diagram illustrates the bi-directional
communication flow between the XR Application and LLM Chat Engine
through middleware services, highlighting the document processing pipeline
that populates the vector database for knowledge retrieval.
Our architecture delivers expert guidance accessible from
voice commands through two primary components: the LLM
Chat Engine for natural language processing and knowledge
retrieval and the XR Application for providing immersive vi-
sual interfaces. The system operates via a structured communi-
cation protocol that connects natural speech recognition in XR
to the back-end knowledge services. Through intelligent query
routing and RAG techniques, domain-specific information is
retrieved from a vector database populated with embedded
documentation. The following subsections detail the technical
specifications and implementation of each architectural com-
ponent.
A. LLM Chat Engine
The LLM Chat Engine forms the core intelligence of the
system, orchestrating communication between agents, man-
aging prompts, and maintaining conversation context during
industrial XR-based interactions. As illustrated in Fig. 2, the
engine integrates a modular architecture centered on the Query
Router Agent, which controls the end-to-end query processing
workflow. This agent evaluates user inputs, leverages meta-
data from registered tools, and dynamically routes queries
to specialized components based on factors such as query
content, historical interactions, and predefined system prompts
[26]. These prompts enforce domain-specific constraints, en-
suring responses remain aligned with industrial objectives. The
following paragraphs elaborate on the LLM Chat Engine’s
internal components and processes.
1) Dynamic Tool Orchestration: Each RAG tool in the
engine is associated with a document stored in the vector
database. During document ingestion, metadata (e.g., title,
version, summary) is registered with the Query Router Agent,
enabling targeted retrieval by narrowing the search space. This
approach improves efficiency and accuracy compared to brute-
force similarity searches across all documents. Additionally,
the router invokes auxiliary agents to enhance context-aware
responses:
Fig. 2. High-level Architectural View of LLM Chat Engine
•PdM Agent: Fetches predictive maintenance (PdM) out-
puts from external services, providing insights on ma-
chinery condition.
•XAI Agent: Retrieves model explanations from an eX-
plainable AI service, helping the user interpret algorith-
mic predictions [27].
•IoT Agent: Acquires current and historical sensor read-
ings from IoT devices, enabling real-time monitoring and
setup guidance.
2) Response Generation: The engine synthesizes answers
by combining retrieved document excerpts, system prompts,
user history, and outputs from auxiliary agents. This enables
automated decomposition of complex industrial queries into
actionable steps, functioning as a reasoning engine for real-
time decision support.
3) Implementation: The engine is deployed as a REST
API via FastAPI, offering endpoints for query processing,
document injection, and session management. Security is en-
forced through API key authentication, while LlamaIndex and
LangChain frameworks establish agent and tool construction.
Session persistence ensures continuity during extended XR
training scenarios.
4) Knowledge Integration: A key sub-system of the LLM
Chat Engine is the Data Injection Component , which processes
and indexes source documents to enable efficient RAG oper-
ations. As illustrated in Fig. 3, the pipeline consists of the
following stages:
•Document Parsing & Chunking: Three docu-
ments—ranging from 74 to 554 pages, with a mean of
331 pages and standard deviation of 197—were converted
to text and then segmented into coherent chunks. Instead
of fixed-length splits—which risk incomplete or arbitrary
divisions—the system applies a semantic approach
that respects headings and subheadings, preserving the
logical structure of technical materials.
•Embeddings Creation: Each chunk is transformed into
a high-dimensional vector representation, capturing se-
mantic content. This facilitates robust similarity matching
even when queries do not use the exact wording of the
original text.

•Vector Database: The resulting vectors are stored in a
database that supports efficient similarity search. Meta-
data are crucial here, allowing the system to filter out
irrelevant content before performing vector retrieval.
•Query Tools: One specialized query tool per use case
leverages both embeddings and metadata for targeted
retrieval. Each tool includes specific fields such as author,
document type, and version to refine the scope of a
search.
Fig. 3. Data and Knowledge Injection Pipeline
5) Performance Evaluation: We benchmarked the re-
trieval system using RAGChecker [28], testing three recog-
nized strategies for chunking, embedding models, and vector
databases on industrial documents of varying lengths and
domains. Each approach was evaluated against four metrics:
Claim Recall (CR) ,Context Precision (CP) ,Hallucination
(Hallu.) , and Faithfulness (Faith.) . Tables I–III summarize the
top-performing configurations.
Among the chunking strategies, Semantic Chunking out-
performed fixed-length methods, reaching a faithfulness of
99.07% while minimizing hallucination (Table I). For embed-
dings, the Mpnet model yielded the highest recall (95.28%) but
had greater hallucination, whereas OpenAI’s model balanced
both accuracy and reliability (Table II) [29], [30]. Finally,
Pinecone ranked highest in the vector database comparison,
achieving 93.37% recall and 96.5% faithfulness (Table III).
These results validate the engine’s ability to provide precise,
context-aware technical guidance for industrial XR applica-
tions.
TABLE I
CHUNKING STRATEGY PERFORMANCE METRICS
Chunking CR CP Hallu. Faith. Rank
Semantic Context 93.05 99.13 0.21 99.07 1
Fixed length=2028 97.12 99.05 3.59 91.14 2
Fixed length=1024 85.41 95.37 4.51 95.49 3TABLE II
EMBEDDING MODEL PERFORMANCE METRICS
Embedding CR CP Hallu. Faith. Rank
Mpnet 95.28 99.4 5.37 90.28 1
OpenAI - small 86.61 96.47 1.06 98.97 2
OpenAI - ada 93.7 98.6 3.56 96.49 3
TABLE III
VECTOR STORE PERFORMANCE METRICS
Vector DB CR CP Hallu. Faith. Rank
Pinecone 93.37 97.22 2.54 96.5 1
Chroma 92.11 98.05 2.01 95.18 2
Faiss 90.11 98.05 3.21 94.48 3
Note : Bold values indicate the best performance for each metric.
B. XR Application
Recent advancements in XR have revolutionized human-
computer interaction, particularly through the integration of
sophisticated speech processing capabilities. In our research,
we propose a novel speech interaction framework for aug-
mented reality applications that leverages AI to create intuitive,
hands-free user experiences.
Fig. 4. XR Architecture
Fig. 4 illustrates the speech interaction architecture in an
AR application, integrating Speech-to-Text (STT) and Text-
to-Speech (TTS) systems to enable seamless communication
with the LLM Chat Engine. The process begins when the user
speaks into the AR app, which captures the audio and forwards
it to the speech recognition system. The STT module, powered
by engines such as Azure Speech, Whisper AI, or Siemens AI
V oice Assistant, transcribes the speech into text and sends it
to the Engine’s API.
Once the LLM Chat Engine processes the user’s query,
it generates a text response. This response is sent back to
the speech synthesizer, which converts the text into natural-
sounding speech using TTS technologies like Azure Speech,
OpenAI TTS, or TFLite models. Additionally, the system

includes a mechanism for mapping speech to lip and body
animations, ensuring a more interactive AR experience.
Real-time processing capabilities of STT and TTS technolo-
gies reduce latency, enabling fluid and natural conversations.
This bidirectional speech interaction eliminates the need for
manual input, making hands-free AI chatbot interactions pos-
sible. Furthermore, the integration of deep-learning models
enhances noise reduction and supports multiple languages,
improving accessibility and usability across different environ-
ments.
This architecture allows users to communicate effortlessly
with the LLM Chat Engine, making applications more in-
teractive and engaging. Whether for remote AI assistance,
interactive learning, or accessibility support, the combination
of STT and TTS in XR enhances user experience by providing
audio feedback and visual interactions in real-time.
IV. A PPLICATION AND USECASES
The proposed LLM-XR architecture is being evaluated
across different industrial environments, each presenting
unique challenges in training, remote assistance and oper-
ational efficiency. These environments are used to explore
the feasibility of the proposed solution through early-stage
evaluation and testing
A. Support for Industrial Assembly and Maintenance
In manufacturing environments that involve robotic assem-
bly and system commissioning, our system is being inte-
grated into XR workflows in experimental setups to assess
its potential to support and improve remote troubleshooting.
In virtual training modules, technicians get an engaging way
to learn, thus reducing the need for physical testing. The
proposed system supports hands-free access to commissioning
checklists, troubleshooting guides and equipment manuals,
enabling novice and expert users to reduce downtime, errors
and enhance productivity. Fig. 5 depicts an example response
from LLM Chat Engine to a user query requesting assistance
in troubleshooting citing the source document.
Fig. 5. Example response from the LLM Chat Engine to a user query.B. Predictive Maintenance in Smart Infrastructure Operations
Our solution supports predictive maintenance for infrastruc-
ture scenarios involving complex systems, such as smart water
networks. The system is being designed to provide simulations
driven by anomaly detection models, with content dynamically
adjusted from LLM Chat engine’s agents utilizing external
services and tools. Visual overlays can show malfunction
indicators, sensor data and AI-generated recommendations
such as step-by-step instructions offering granular support
during inspections and repair workflows. Fig. 6 and 7 illustrate
an example AR application of a smart water pipe assisting
technicians to find the location of a leakage.
Fig. 6. AR application of a smart water pipe rendering IoT data available
from LLM agent.
Fig. 7. AR application of a smart water pipe rendering predictive maintenance
data available from LLM agent.
C. Stress-aware Maintenance in Safety-Critical Environments
Our XR-LLM integration focuses on safety, precision and
stress mitigation in aerospace contexts where component main-
tenance involves high cognitive and physical demand. The as-
sistant can provide step-by-step safety procedures and answer
real-time maintenance questions, aims to provide feedback and
reference while considering biometric from wearables such as
smartwatches.

V. C ONCLUSION AND FUTURE WORK
Our paper has presented a novel architecture integrating
LLM agents with XR environments to address critical knowl-
edge transfer challenges in industrial settings. By combining
an intelligent LLM Chat Engine with XR applications, our
system enables hands-free, context-aware interactions that
adapt to users’ expertise levels across diverse industrial sce-
narios. Performance evaluation of various chunking strategies,
embedding models, and vector databases has validated our
approach to knowledge retrieval, with semantic chunking,
balanced embedding models, and efficient vector stores like
Pinecone delivering optimal results for industrial documen-
tation. Future work will focus on full-scale integration and
evaluation of our system in the presented industrial use cases,
implementing required optimizations based on deployment
feedback. We plan to test the LLM Chat Engine with smaller,
locally-hosted open-source models to address privacy concerns
when handling confidential documents. Additionally, we aim
to enhance multimodal capabilities through computer vision
integration and implement advanced personalization based
on user performance and biometric feedback. This continued
development promises to further bridge the gap between im-
mersive technologies and contextual intelligence for industrial
applications.
ACKNOWLEDGMENT
Part of the research leading to the results presented in this
paper has received funding from the European Union’s funded
Project XR5.0 [GA 101135209].
REFERENCES
[1] J. M. Ro ˇzanec, I. Novalija, P. Zajec, K. Kenda, H. Tavakoli Ghinani,
S. Suh, E. Veliou, D. Papamartzivanos, T. Giannetsos, S. A. Menesidou
et al. , “Human-centric artificial intelligence architecture for industry
5.0 applications,” International journal of production research , vol. 61,
no. 20, pp. 6847–6872, 2023.
[2] European Commission, “Industry 5.0: Towards a sustainable,
human-centric and resilient european industry,” 2021, publications
Office of the European Union, Luxembourg. Available
at: https://op.europa.eu/en/publication-detail/-/publication/
468a66b1-6db4-11eb-aeb5-01aa75ed71a1.
[3] P. Wurster, P. Finkel, and R. Radler, “Large language models (llm) in
production: An analysis of the potential for transforming production
processes in modern factories,” Industry 4.0 Science , vol. 40, no. Edition
6, pp. 50–54, 2024.
[4] S. Ongbali, S. Afolalu, S. Oyedepo, A. Aworinde, and M. Fajobi, “A
study on the factors causing bottleneck problems in the manufacturing
industry using principal component analysis,” Heliyon , vol. 7, no. 5,
2021.
[5] P. Sheldon and S.-H. Kwon, “Samsung in vietnam: Fdi, busi-
ness–government relations, industrial parks, and skills shortages,” The
Economic and Labour Relations Review , vol. 34, pp. 1–20, feb 2023.
[6] G. Fatouros, J. Soldatos, K. Kouroumali, G. Makridis, and D. Kyriazis,
“Transforming sentiment analysis in the financial domain with Chat-
GPT,” Machine Learning with Applications , vol. 14, p. 100508, 2023,
publisher: Elsevier.
[7] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel et al. , “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” Advances in
neural information processing systems , vol. 33, pp. 9459–9474, 2020.
[8] G. Fatouros, K. Metaxas, J. Soldatos, and M. Karathanassis, “Market-
senseai 2.0: Enhancing stock analysis through llm agents,” arXiv preprint
arXiv:2502.00415 , 2025.[9] A. Kiourtis, A. Mavrogiorgou, G. Makridis, D. Kyriazis, J. Soldatos,
G. Fatouros, D. Ntalaperas, X. Papageorgiou, B. Almeida, J. Guedes
et al. , “Xr5. 0: Human-centric ai-enabled extended reality applications
for industry 5.0,” in 2024 36th Conference of Open Innovations Associ-
ation (FRUCT) . IEEE, 2024, pp. 314–323.
[10] M. Shao, A. Basit, R. Karri, and M. Shafique, “Survey of
Different Large Language Model Architectures: Trends, Benchmarks,
and Challenges,” IEEE Access , vol. 12, pp. 188 664–188 706, 2024,
publisher: Institute of Electrical and Electronics Engineers (IEEE).
[Online]. Available: http://dx.doi.org/10.1109/ACCESS.2024.3482107
[11] G. Fatouros, K. Metaxas, J. Soldatos, and D. Kyriazis, “Can large
language models beat wall street? evaluating gpt-4’s impact on financial
decision-making with marketsenseai,” Neural Computing and Applica-
tions , pp. 1–26, 2024.
[12] J. My ¨oh¨anen, “Improving industrial performance with language models:
a review of predictive maintenance and process optimization,” 2023.
[13] Z. Song, B. Yan, Y . Liu, M. Fang, M. Li, R. Yan, and X. Chen,
“Injecting Domain-Specific Knowledge into Large Language Models:
A Comprehensive Survey,” 2025, eprint: 2502.10708. [Online].
Available: https://arxiv.org/abs/2502.10708
[14] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei, “Agentic retrieval-
augmented generation: A survey on agentic rag,” 2025. [Online].
Available: https://arxiv.org/abs/2501.09136
[15] F. Bousetouane, “Agentic systems: A guide to transforming industries
with vertical ai agents,” 2025. [Online]. Available: https://arxiv.org/abs/
2501.00881
[16] A. Gezdur and J. Bhattacharjya, “Innovators and transformers:
enhancing supply chain employee training with an innovative
application of a large language model,” International Journal of
Physical Distribution & Logistics Management , 2025, publisher:
Emerald Publishing Limited. [Online]. Available: https://www.emerald.
com/insight/content/doi/10.1108/ijpdlm-12-2023-0492/full/html
[17] M. Baptista, N. Yue, M. M. M. Islam, and H. Prendinger, Large
Language Models (LLMs) for Smart Manufacturing and Industry X.0 ,
03 2025, pp. 97–119.
[18] X. Wang, A. Yew, S.-K. Ong, and A. Y . Nee, “Enhancing smart shop
floor management with ubiquitous augmented reality,” International
Journal of Production Research , vol. 58, no. 8, pp. 2352–2367, 2020.
[19] C. Moro, J. Birt, Z. Stromberga, C. Phelps, J. Clark, P. Glasziou,
and A. M. Scott, “Virtual and augmented reality enhancements to
medical and science student physiology and anatomy test performance:
A systematic review and meta-analysis,” Anatomical sciences education ,
vol. 14, no. 3, pp. 368–376, 2021.
[20] M. Breque, L. De Nul, A. Petridis et al. , “Industry 5.0: towards a
sustainable, human-centric and resilient european industry,” Luxem-
bourg, LU: European Commission, Directorate-General for Research
and Innovation , vol. 46, 2021.
[21] J. Carmigniani, B. Furht, M. Anisetti, P. Ceravolo, E. Damiani, and
M. Ivkovic, “Augmented reality technologies, systems and applications,”
Multimedia tools and applications , vol. 51, pp. 341–377, 2011.
[22] A. Berni and Y . Borgianni, “Applications of virtual reality in engineering
and product design: Why, what, how, when and where,” Electronics ,
vol. 9, no. 7, p. 1064, 2020.
[23] G. Makransky and L. Lilleholt, “A structural equation modeling investi-
gation of the emotional value of immersive virtual reality in education,”
Educational Technology Research and Development , vol. 66, no. 5, pp.
1141–1164, 2018.
[24] M. R. Marner, A. Irlitti, and B. H. Thomas, “Improving procedural
task performance with augmented reality annotations,” in 2013 IEEE
International Symposium on Mixed and Augmented Reality (ISMAR) .
IEEE, 2013, pp. 39–48.
[25] X. Wang, S. K. Ong, and A. Y . Nee, “A comprehensive survey of
augmented reality assembly research,” Advances in Manufacturing ,
vol. 4, pp. 1–22, 2016.
[26] S. G. Ayyamperumal and L. Ge, “Current state of llm risks and ai
guardrails,” arXiv preprint arXiv:2406.12934 , 2024.
[27] G. Makridis, V . Koukos, G. Fatouros, and D. Kyriazis, “Virtualxai:
A user-centric framework for explainability assessment leveraging gpt-
generated personas,” arXiv preprint arXiv:2503.04261 , 2025.
[28] D. Ru, L. Qiu, X. Hu, T. Zhang, P. Shi, S. Chang, C. Jiayang,
C. Wang, S. Sun, H. Li et al. , “Ragchecker: A fine-grained framework
for diagnosing retrieval-augmented generation,” Advances in Neural
Information Processing Systems , vol. 37, pp. 21 999–22 027, 2024.

[29] K. Song, X. Tan, T. Qin, J. Lu, and T.-Y . Liu, “Mpnet: Masked and
permuted pre-training for language understanding,” Advances in Neural
Information Processing Systems , vol. 33, pp. 16 857–16 867, 2020.
[30] N. B. Korade, M. B. Salunke, A. A. Bhosle, P. B. Kumbharkar, G. G.
Asalkar, and R. G. Khedkar, “Strengthening sentence similarity iden-
tification through openai embeddings and deep learning.” International
Journal of Advanced Computer Science & Applications , vol. 15, no. 4,
2024.