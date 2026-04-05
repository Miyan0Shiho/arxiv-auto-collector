# The Energy Footprint of LLM-Based Environmental Analysis: LLMs and Domain Products

**Authors**: Alicia Bao, Jiamian He, Angel Hsu, Diego Manya, Ji, Zhang

**Published**: 2026-03-31 00:19:07

**PDF URL**: [https://arxiv.org/pdf/2604.00053v1](https://arxiv.org/pdf/2604.00053v1)

## Abstract
As large language models (LLMs) are increasingly used in domain-specific applications, including climate change and environmental research, understanding their energy footprint has become an important concern. The growing adoption of retrieval-augmented (RAG) systems for climate-domain specific analysis raises a key question: how does the energy consumption of domain-specific RAG workflows compare with that of direct generic LLM usage? Prior research has focused on standalone model calls or coarse token-based estimates, while leaving the energy implications of deployed application workflows insufficiently understood. In this paper, we assess the inference-time energy consumption of two LLM-based climate analysis chatbots (ChatNetZero and ChatNDC) compared to the generic GPT-4o-mini model. We estimate energy use under actual user queries by decomposing each workflow into retrieval, generation, and hallucination-checking components. We also test across different times of day and geographic access locations. Our results show that the energy consumption of domain-specific RAG systems depends strongly on their design. More agentic pipelines substantially increase inference-time energy use, particularly when used for additional accuracy or verification checks, although they may not yield proportional gains in response quality. While more research is needed to further test these initial findings more robustly across models, environments and prompting structures, this study provides a new understanding on how the design of domain-specific LLM products affects both the energy footprint and quality of output.

## Full Text


<!-- PDF content starts -->

The Energy Footprint of LLM-Based Environmental
Analysis: LLMs and Domain Products
A Preprint
Alicia Bao∗
Department of Computer
Science
University of North Carolina at
Chapel Hill
Chapel Hill, NC 27516
alicia.bao@unc.edu
Jiamian He
Arboretica
Rotterdam, The Netherlands
jiamian@arboretica.com
Angel Hsu
Department of Public Policy
University of North Carolina at
Chapel Hill
Chapel Hill, NC 27516
angel.hsu@unc.edu
Diego Manya
Data-Driven EnviroLab
UNC Institute for Environment
Chapel Hill, NC 27516
diego.manya@unc.edu
Ji (James) Zhang
Arboretica
Rotterdam, The Netherlands
james@arboretica.com
April 2, 2026
Abstract
As large language models (LLMs) are increasingly used in domain-specific applications,
includingclimatechangeandenvironmentalresearch, understandingtheirenergyfootprinthas
become an important concern. The growing adoption of retrieval-augmented (RAG) systems
for climate-domain specific analysis raises a key question: how does the energy consumption
of domain-specific RAG workflows compare with that of direct generic LLM usage? Prior
research has focused on standalone model calls or coarse token-based estimates, while leaving
the energy implications of deployed application workflows insufficiently understood. In
this paper, we assess the inference-time energy consumption of two LLM-based climate
analysis chatbots (ChatNetZero and ChatNDC) compared to the generic GPT-4o-mini
model. We estimate energy use under actual user queries by decomposing each workflow
into retrieval, generation, and hallucination-checking components. We also test across
different times of day and geographic access locations. Our results show that the energy
consumption of domain-specific RAG systems depends strongly on their design. More
agentic pipelines substantially increase inference-time energy use, particularly when used for
additional accuracy or verification checks, although they may not yield proportional gains in
response quality. While more research is needed to further test these initial findings more
robustly across models, environments and prompting structures, this study provides a new
understanding on how the design of domain-specific LLM products affects both the energy
footprint and quality of output.
KeywordsLLMs·energy consumption·domain products
∗Authors contributed equally to this work and are listed alphabetically.arXiv:2604.00053v1  [cs.SE]  31 Mar 2026

Energy consumption of LLMsA Preprint
1 Introduction
While the rapid proliferation of Generative Artificial Intelligence (GenAI) and Large Language Model (LLM)-
powered chatbots represents a transformative shift in how information is generated and accessed, major
questions remain regarding the energy-intensity and environmental footprint of utilizing these tools compared
to other means of information acquisition, such as traditional Internet search tools. One approach that has
been touted as a potential solution to very large, generic LLM-based chatbots that have been trained with
massive amounts of data [1] is the development of Domain-Specific Retrieval-Augmented generation (RAG)
chatbots that represent a “smaller” (small-LLM) [ 2] alternative. Particularly in the environment, climate and
sustainability domains, domain-specific RAG chatbots have emerged, promising more accurate information
related to climate science and policy compared to generic chatbots [ 3,4,5]. While these tools claim higher
accuracy and “anti-hallucination” filters, what is less transparent is whether the proposed climate benefits of
these tools may be paradoxically offset from their energy consumption and environmental impacts.
The question of GenAI’s energy and environmental footprints has risen in scrutiny and public concern, given
the increasing number of data centers planned or in construction across the U.S. as a result of AI’s growing
needs for training and running models [ 6,7]. Although data centers are estimated to only account for 1% of
total global electricity demand [ 8], this percentage is projected to grow to as much as 12% in 2030 [ 9]. Prior
research has demonstrated that while model training is technically much more energy-intensive than inference
[10], the inference phase in aggregate is responsible for the vast majority of AI’s energy consumption today
given the scale of AI usage [ 11]. In light of rising public concern, hyperscaler AI companies like Microsoft and
Google have released estimates of the water, energy and environmental impacts of AI usage [ 12], although
critics have pointed out the lack of transparency and potential inaccuracies in these estimates [13].
Here, wepresentamethodologicalrefinementofapriormeasurementframework[ 14], enablinganempiricalend-
to-end carbon emission assessment of real-world, domain-specific LLM-RAG products used for environmental
analysis. We introduce a query-level methodology to decompose and quantify the energy consumption of
RAG sub-components, rigorously benchmarking the relative costs of the Retrieval and the Generation stages.
Ultimately, our findings yield the first quantitative framework for evaluating this complex environmental
trade-off (energy consumption vs quality), leading to actionable, data-driven architectural guidelines for
designing demonstrably more sustainable generative AI systems.
2 Background
2.1 ChatNetZero and ChatNDC - Climate-domain specific chatbots
ChatNetZero.ai (CNZ) is a climate-specific chatbot designed to reduce the well-known limitations of generic
LLMs, especially hallucination, by pairing a RAG pipeline with purpose-built validation and referencing
modules [ 4]. In the model, heterogeneous data sources (e.g., PDFs and spreadsheets) are converted to plain
text, segmented into paragraph-preserving chunks, and embedded alongside user queries to enable semantic
retrieval. Each chunk is tagged with document and page metadata to enable semantic retrieval to preserve
provenance. CNZ then applies query processing that distinguishes actor-specific questions (e.g., named
governments or companies) from generic questions, prioritizing Net Zero Tracker [ 15] entities when detecting
actor-specific contexts. CNZ then retrieves a bounded set of highly relevant chunks per actor while using
top-ranked report chunks for non-actor queries. Retrieved evidence is combined with the user prompt and
passed to an LLM under tightly constrained prompting (e.g., temperature set to 0 and explicit instructions
to answer only from provided text and acknowledge uncertainty). To further mitigate hallucination, an
anti-hallucination module embeds the draft output sentence-by-sentence and verifies each claim against
retrieved source chunks, excluding sentences that cannot be traced back to the underlying corpus. A reference
module then appends per-sentence citations, including document name, page number, and matched location,
linking users directly to original source pages for manual verification. Finally, CNZ extends beyond narrative
Q&A by transforming structured Net Zero Tracker variables into natural-language statements, enabling
quantitative and counting-style queries to be handled through the same retrieval-and-grounding workflow.
ChatNDC (ChatNDC.org, beta) is a climate policy AI chatbot launched in January 2025 by Data-Driven
EnviroLab and Arboretica to streamline credible analysis of countries’ Nationally Determined Contributions
(NDCs) under the Paris Agreement. The system is grounded in a curated corpus of verified climate documents
spanning officially submitted national policy texts reported to the The United Nations Framework Convention
on Climate Change (UNFCCC), country-specific scientific and policy assessments, and global syntheses on
climate science, finance, mitigation, and adaptation (including the latest IPCC assessments), enabling users
2

Energy consumption of LLMsA Preprint
Figure 1: Design of ChatNetZero [4].
to interrogate emissions targets, timelines, sectoral priorities, and changes across successive NDC iterations.
To minimize hallucinations, ChatNDC pairs generation with an “anti-hallucination” logic that cross-verifies
outputs against trusted sources in real time and returns granular, auditable references (both embedded in the
narrative response and linked back to precise locations in the underlying documents) for users to directly
validate claims. Beyond question and answering capabilities, the platform supports document-level exploration
via a data catalog, allowing users to query one or multiple source documents and receive evidence-forward
responses that surface and highlight the relevant text passages, effectively turning complex climate reports
into interactive analytical objects. In practice, this grounding-and-referencing approach improves reliability
over generic chatbots. For example, on questions such as whether Egypt has declared a net-zero target, where
unconstrained LLMs may produce false positives, ChatNDC was designed to provide a defensible answer
with traceable citations to Egypt’s submitted materials.
2.2 LLM energy consumption components: training, inference
For the purposes of evaluating LLM-based chatbots, operational energy consumption is comprised primarily
of two main phases: training and inference [ 16]. Training includes pre-training and any subsequent fine-tuning
and is typically concentrated in short but highly compute-intensive runs [ 9]. The energy demand during this
phase is determined by a range of factors, including model parameter count, the size of the training dataset,
processor, hardware Thermal Design Power (TDP), and the Power Usage Effectiveness (PUE) of the data
center’s cooling infrastructure [ 17]. For instance, it is estimated that the training of GPT-4 consumed around
50 GWh of electricity [ 18]. Similarly, while Google’s Gemini series employs more energy-efficient architectures
and specialized Tensor Processing Units (TPUs), the training cost for LLMs remains at a comparable volume,
equivalent to the annual energy consumption of thousands of average households [ 19]. Although these figures
represent an initial surge in electricity demand, the total energy consumption becomes fixed once the model
is deployed.
In contrast, the inference phase begins after deployment and covers the electricity required to process prompts
and generate responses in real time. This distinction is analytically important because the two phases differ
not only in temporal profile but also in their energy consumption, since training usually happens over a
short, discrete period of time, whereas the inference (e.g., user phase) is continuous and scales directly with
usage volume. Evidence indicates that for a model of GPT-3’s scale, once daily active users reach the tens
of millions, it takes a mere 43 days of continuous operation for cumulative inference energy to equal the
3

Energy consumption of LLMsA Preprint
initial training energy [ 20]. For popular models, this ‘energy parity point’ is reached even more rapidly, as
the computational complexity of inference increases non-linearly with the length of the input context [16].
Early work on the environmental cost of LLMs and AI emphasized the substantial resource demands of the
training phase, while more recent work has argued that inference deserves equal or greater attention once
models are deployed across user-facing applications like chatbots [21, 10, 16].
2.3 Previous research on measuring LLM inference methods
To quantify carbon emissions during the LLM inference phase, current research primarily adopts three
methods: local hardware-based monitoring, architecture-driven parameterized estimation, and end-to-end
API-based estimation.
Hardware-based Monitoring (Local Deployment): The most prevalent approach involves direct energy
measurement via hardware-level sensors (GPU, RAM, CPU) or Cloud Service Provider (CSP) embedded
calculators. This method is favored for its high precision and controllability within laboratory environments.
Consequently, such studies [22, 23, 24] typically deploy open-source models for measurement.
Architecture-drivenParameterizedEstimation: Analyticaltools, suchasLLMCarbon[ 25]and LLMCO 2
[26], utilize parameterized models to estimate energy consumption and carbon emissions. These techniques rely
heavily on specific model architecture details to compute Floating Point Operations (FLOPs). Furthermore, to
bridge the gap between theoretical FLOPs and actual consumption, the R-ICE [ 27] research trains regression
models within the framework, creating benchmarks that more accurately map inference carbon and energy
metrics across diverse hardware configurations.
End-to-End API-based Estimation: For commercial ‘black-box’ models, where architectural details
and local deployment are limited, end-to-end API estimation has emerged as an efficient evaluation path.
Related studies include the MELODI framework [ 28], and the subsequent work by Jegham et al. [ 14] both
of which systematically evaluate the environmental impacts of LLM inference. Jegham et al. [ 14] propose
a framework that estimates energy usage based on three core components: inference latency, estimated
model/GPU scale, and PUE. This method has the generalizability that it represents a new trend, a viable
approach for conducting end-to-end assessments of commercial black-box models.
2.4 Limitations of previous research
Previous approaches to estimating LLM energy consumption remain limited in ways that reduce their
usefulness for evaluating deployed chatbot products. Local measurement methods generally require direct
access to the model runtime and hardware, which makes them best suited to locally hosted, open-weight
systems rather than proprietary API-based models. As a result, much of the existing empirical literature
has focused on smaller (e.g., less than 40 billion parameters) or open-source models, while widely used
closed models such as GPT-4o remain difficult to assess directly because their architectures, deployment
configurations, and hardware allocations are not publicly disclosed. In these settings, FLOPs-based analytical
methods can be informative, but they depend on assumptions about model structure and infrastructure that
are often unverifiable in commercial deployments. Prior reviews similar note that local prompt-level methods
have largely concentrated on open-source or small-scale settings and do not fully capture the infrastructure
complexity of production inference [14].
Recent API-based estimation methods help address part of this gap. In particular, Jegham et al. [ 14]
propose an infrastructure-aware framework that combines public API latency and throughout data with
published hardware specifications and environmental multipliers to estimate prompt-level inference impacts
for proprietary as well as open models. Their study represents an important advance because it moves
beyond purely local measurement and makes closed commercial systems more tractable for sustainability
analysis. This approach is also designed primarily to estimate the environmental footprint of individual
model calls across standardized query-size categories, rather than the full execution logic of application-layer
systems. Their “How Hungry is AI” dashboard relies on daily-scraped API performance data from Artificial
Analysis, which allows for cross-model benchmarking, but is less suited to reconstructing the exact sequence
of operations performed by a specialized RAG product for a particular user query [14].
This distinction matters because in practice, real-world RAG systems, especially those used with agentic
workflows, could trigger multiple LLM API calls within a single user request, with the number and structure
of calls determined by the system design rather than token length alone. Agentic or multi-stage pipelines
therefore introduce energy costs that cannot be inferred reliably from prompt size categories or from a single
4

Energy consumption of LLMsA Preprint
end-to-end API call. Additionally, different pipeline components rely on different computational resources:
retrieval may be primarily CPU-bound, whereas generation and LLM-based hallucination checks depend more
heavily on GPU-backed inference. For this reason, accurately estimating inference-time energy consumption
for deploying RAG system requires a pipeline decomposition that reflects actual execution patterns under
real queries.
Another limitation of prior work is that energy use is often quantified in isolation from output quality
or accuracy. Existing studies have substantially improved the understanding of LLM inference’s resource
demands, but they generally do not examine whether additional computational cost translates into better
answers, stronger retrieval grounding, or reduced hallucination in applied settings. These gaps leave an
important practical question unresolved: whether more energy-intensive workflows deliver meaningful gains
in response quality, especially when systems add extra inference steps for routing, validation or reference
checking.
This present study addresses these gaps by shifting the unit of analysis from the standalone model call
to the deployed application workflow. Rather than estimating energy use from coarse token-length bands
alone, we empirically quantify the energy consumption of domain-specific LLM and RAG products under
actual user queries in a specialized environmental context. We further decompose each RAG workflow
into its constituent inference-time steps, including query classification, retrieval, response generation, and
hallucination checking, and estimate the energy associated with each component separately. This step-level
approach makes it possible to identify where energy is actually consumed within LLM-based chatbot systems,
including components that would be obscured in model-level averages. In addition, to improve the robustness
of our estimates, we test energy consumption across different times of day and geographic locations by varying
IP addresses, thereby capturing temporal and spatial variation in API response behavior and producing more
reliable average energy estimates across systems and user contexts. Finally, by evaluating both energy use
and response quality, our framework makes visible the trade-offs between computational cost and performance
that are central to the real-world design of LLM-based applications. While foundational studies, particularly
Jegham et al. [ 14], provide important advances in benchmarking the electricity consumption of LLM APIs
through an infrastructure-aware framework that integrates public API performance data, region-specific
environmental multipliers, and inferred hardware configurations, our study extends this literature by moving
from model-level, token-range-based estimation to the direct empirical evaluation of deployed LLM and RAG
applications under real queries.
3 Methods
3.1 Dataset
To evaluate energy consumption and response quality under realistic usage conditions, we constructed a
dataset of 102 domain-specific test questions reflecting the kinds of queries users are likely to ask in deployed
climate-LLM applications. The dataset was manually curated to cover both direct factual retrieval and higher-
order analytical tasks within the climate policy domain. Substantively, the questions span corporate climate
commitments, net-zero targets, offset conditions, and national climate policy topics, including country’s Paris
Agreement pledges or NDCs, for which the ChatNDC.org platform was designed to analyze. We classified the
questions following Bloom's Taxonomy [ 29]: Knowledge, Comprehension, Application, Analysis, Evaluation
and Creation with 48 questions of the knowledge class and approximately 10 of each of the other classes. We
particularly focused on the Knowledge class due to their particular alignment with the use-case of our custom
models (CNZ and ChatNDC) as well as for the considerable breadth of questions that could be included in
this category such as questions that just retrieve data as well as those that summarize it. Figure 4 shows the
median and variability of the token size for each answer by each model, highlighting the significant variability
in output token from unconstrained models such as GPT 4o-Mini, and the very limited variance in answer
length from constrained model such as GPT 4o-Mini (200) and CNZ. A subset of these questions, between
19 to 25 for each model, was selected for manual evaluation on their factual accuracy and embellishment
prevalence.
5

Energy consumption of LLMsA Preprint
Figure 2: Architectures of (a) ChatNetZero and (b) ChatNDC.
Table 1: Representative questions from the evaluation dataset.
Question type Example questions
Knowledge “Has Marathon Petroleum made any climate pledges?”; “Does Foxconn have any climate
strategy?”; “What is the total emissions reduction target of Malaysia?”; “Does Taiwan
have a carbon trading scheme or policy under its latest NDC?”
Comprehension “Interpret how ESG regulations influence business strategy and investor decisions”;
“Explain the relationship between local air pollution and public health outcomes.”
Application “What are Scope 3 emissions and what should be included in a net-zero target?”; “If
an entity doesn’t control out-of-boundary emissions, how can it credibly set a net-zero
target?”
Analysis “Are Costco’s or Walmart’s net zero target 1.5 degree aligned?”; “How does Walmart’s
climate goals compare with other large retail stores?”; “How do England and Scotland
NDCs compare to the UK’s?”
Evaluation “Is Apple’s net zero target credible?”; “How is supermarket chain Jeronimo Martins
doing on their commitment to reduce food waste by 50% by 2025?”
Creation “Propose an ESG audit model that integrates social, environmental, and ethical per-
formance indicators into one scorecard.”; “Develop a set of innovative KPIs to measure
corporate progress toward net-zero goals beyond traditional carbon metrics.”
3.2 Energy Consumption Estimation
3.2.1 ChatNetZero & ChatNDC structure
Unlike a direct API call for LLMs, different domain RAG systems have different structures. CNZ uses a
simple RAG as shown in Figure 1. It mainly contains one retrieval, one LLM call with a self-fine-tuned GPT
4o-mini model, and uses the pairwise cosine similarity method to check the LLM’s raw response relevance
by sentence with the retrieval documents. If the cosine score is smaller than a threshold, the sentence in
response will be deleted from the final output. We use cosine similarity to determine how semantically similar
two texts are by measuring the cosine of the angle between their vector representations. In our pipeline, a
low cosine score indicates that a generated sentence is semantically inconsistent (potentially indicating a
hallucination) compared to existing content, triggering its removal to ensure a concise output.
ChatNDC is a more complex RAG, with two more LLM inference calls than CNZ. As shown in Figure 2, the
system first uses an LLM inference (GPT-4o) to categorize the user query by theme, which determines the
relevant databases for retrieval. The retrieved content is then passed to the second LLM inference call (GPT
6

Energy consumption of LLMsA Preprint
Figure 3: Median answer token size by each model.
4o-Mini) with a database-aware prompt template to generate the response. Finally, a third LLM inference
call (GPT 4o-Mini) performs a sentence-level hallucination check to validate the generated content before
outputting the final answer. Compared to the CNZ cosine similarity method, which relies on a static threshold
value, ChatNDC employs an LLM-based approach for hallucination checks. This choice is driven by the
LLM’s flexible reasoning capability and its automated citation verification (cross-referencing) ability. Unlike
fixed mathematical metrics, the LLM can intelligently assess whether a generated statement is accurately
supported by the retrieved references, ensuring both factual integrity and logical consistency.
Thus, for the RAG systems, the total energy consumption ( Etotal) is modeled by aggregating the energy costs
across three main inference parts of the RAG systems:
•Retrieval: retrieve the documents in the vector database by query (Spreadsheets or PDFs).
•LLM Inference: API request to completion of the generated answer.
•Hallucination Check: pairwise algorithm or LLM for the sentence-level consistency verification.
The total energy consumption is represented by:E total=E retrieval+E inference +E hallucination .
3.2.1 Experiment Design
Our objective is to quantify the inference-time and energy consumption of different LLM-based chatbot
pipelines by measuring execution time and decomposing the computational steps within each workflow. To
do so, we conduct controlled experiments that isolate the major components of each pipeline and evaluate
their performance under variation in time of day. This design allows us to generate a more representative
estimate of average energy consumption by accounting for fluctuations in latency associated with network
conditions and server load. Detailed energy-estimation procedures are described in Sections 3.2.3–3.2.5.
All experiments were conducted in a Google Colab environment using Python 3 and a single-core CPU
runtime. All LLMs were accessed through API calls with the temperature parameter fixed at 0. Setting
the ‘temperature’ to 0 reduces stochastic variation by making generation deterministic, thereby improving
reproducibility and minimizing noise in repeated measurements of latency, token use, and estimated energy
consumption.
Because both CNZ and ChatNDC use GPT 4o-Mini as their primary generation model, we restrict comparison
to four pipelines: CNZ, ChatNDC, GPT 4o-Mini, and GPT 4o-Mini (200). The GPT 4o-Mini (200) condition
imposes a maximum output length of 200 words in order to provide a controlled baseline for comparison
with CNZ, whose responses are inherently constrained to this approximate length. This shared base-model
design provides a consistent benchmark and allows us to isolate differences attributable to workflow structure
rather than differences in model family, parameter scale, or provider infrastructure. Although absolute energy
consumption can vary across other model architectures and hosting environments, holding the underlying
model constant improves comparability across our experimental conditions.
7

Energy consumption of LLMsA Preprint
The experiment consists of two complementary parts, both based on the same evaluation dataset. The first
part focuses on energy estimation robustness. To account for variation in network latency and API server
load, each pipeline was executed repeatedly across four runs: one run at a randomly selected time and
three runs during fixed time windows in the morning (08:00–10:30), afternoon (14:00–16:30), and evening
(20:00–22:30). All runs originated from the Netherlands. For each run, we recorded complete execution time
and token usage and converted these measurements into estimated energy consumption. Query and response
tokens were counted using the tiktoken library [30] configured for GPT 4o-Mini.
Finally, we combine estimated cumulative energy consumption with response quality evaluation to examine
the trade-offs between computational cost and performance across systems. By jointly examining energy use,
token consumption and answer quality, we assess the relative efficiency of each pipeline and identify whether
additional workflow and RAG design complexity yield meaningful improvements in output quality.
3.2.2 LLM consumption estimation method
Building upon Jegham et al.’s methodology [ 14], we refine their approach to measure LLM energy consumption.
As shown in Formula 1, Equeryrepresents the total energy consumption. The total inference time, Ttotal
(the sum of terms in the first bracket), corresponds to the elapsed time from calling the API until the final
full answer is obtained. The total workload, Wtotal(the sum of terms in the second bracket), is calculated
according to Formula 2. The PUE is the 1.12 as mentioned in the Microsoft datacenter report [ 31] where
OpenAI holds their LLM models.
Equery (kWh) =/parenleftiggOutput Length
TPS+Latency
3600/parenrightigg
(PGPU ×UGPU+Pnon-GPU ×Unon-GPU )·PUE (1)
UGPU =G×D GPU
N×B, U non-GPU =G×D non-GPU
N×B.(2)
Here, Gdenotes the number of GPUs assigned per model, Nthe number of GPUs per node, and Bis the
batch size. DGPUrepresents the assigned power draw of GPUs, expressed as a fraction of their maximum
rated power, while Dnon−GPU corresponds to a conservatively estimated fixed utilization fraction for non-GPU
components (e.g., CPU, memory, storage, cooling). The LLM parameters, as reported in the reference paper,
are summarized below:
Model Size GPUG D gpuDnongpu N B
GPT-4o Large H100 8 0.6 0.5 8 8
GPT 4o-Mini Medium A100 4 1.2 0.5 8 8
Therefore we estimate W·PUE according to the Equations 1 and 2, and the Ttotalis obtained from the
previously outlined and then we estimate the parameters of the formula, presented in the Table below:
ModelU GPUtotal UGPU PGPU Pnon−GPU WtotalPUEW·PUE
GPT-4o 0.42 0.2875 5.6 4.6 0.7075 1.12 0.7924
GPT 4o-Mini 0.24 0.103125 3.2 3.3 0.343125 1.12 0.3843
EGPT-4o (kWh) =T llm(h)·0.7075 (kW)·1.12 =T llm(h)·0.7924 (kW)(3)
EGPT 4o mini (kWh) =T llm(h)·0.343125 (kW)·1.12 =T llm(h)·0.3843 (kW)(4)
3.2.3 Retrieval consumption estimation method
In the retrieval part, since all processes are executed within a Google Colab environment using a single
CPU core (typically an Intel(R)-Xeon(R)-CPU@2.20GHz), the retrieval energy consumption is approximated
as the energy draw of a single CPU core. The official specifications for this Intel(R)-Xeon(R)-CPU series
[32] list several variants with TDP values ranging from 150W to 330W, with a number of cores ranging
8

Energy consumption of LLMsA Preprint
from 12 to 44, but the Colab platform does not specify the exact sub-model utilized during the session. For
cross-verification, we adopt a power consumption standard of about 8.5W per core based on LLMCarbon
data to serve as our primary calculation benchmark. Furthermore, to accurately reflect the total energy
footprint, including infrastructure overhead, we incorporate a PUE value of 1.09, as disclosed in the Google
Data Center Efficiency Report [ 33]. The final energy consumption for this phase is thus determined by the
product of the measured execution time, the power per core, and the PUE factor as follows:
Eretrieval =Tretrieval ·PCPU·PUE(5)
Eretrieval (kWh) =T retrieval (h)·0.0085 (kW)·1.09 =T retrieval (h)·0.009265 (kW)(6)
3.2.4 Hallucination Check consumption estimation method
As described above in section 3.2.1, CNZ and NDC employ different hallucination-checking approaches,
and these differences are reflected in how their energy consumption is estimated. In CNZ, hallucination
checking is implemented through sentence-level cosine similarity between each generated sentence and the
retrieved source documents. Because this process does not require an additional LLM call, its energy use is
attributed to the retrieval/post-processing stage and accounted for within Eretrieval. In contrast, ChatNDC
performs hallucination checking through an additional GPT 4o-Mini call; therefore, the energy consumption
is estimated using the same approach as forE GPT 4o mini .
ECNZ, hc (kWh) =T hc(h)·0.0085 (kW)·1.09(7)
=Thc(h)·0.009265
(8)
ENDC, hc (kWh) =T hc(h)·0.343125 (kW)·1.12
=Thc(h)·0.3843
3.3 Evaluation
To assess the answers, we employ a Human-Expert Evaluation method [ 34]. Evaluation is conducted at
the statement level, treating each statement as the fundamental unit of assessment. When we identified
that the RAG-generated response wasn’t relevant or pertinent to the question asked, we excluded from the
response-quality evaluation for both the RAG system and the pure GPT baseline. Each statement within a
generated response is evaluated in two aspects: Factual Accuracy and Embellishment Prevalence.
Factual Evaluation: For statements containing verifiable claims or ground-truth-dependent information,
a binary scoring system is applied. A statement receives a score of 1 (Correct) if and only if it is entirely
consistent with the ground truth; otherwise, it receives a score of 0. The aggregated Factual Index for each
answer is then calculated as the ratio between the number of correct factual statements and the total number
of factual statements.
Embellishment Evaluation:For each answer we identified the number of Non-Factual statements, defined
as statements that are not associated to a validated source. The embellishments index for each answer is
calculated as the ratio between the number of non-factual statements to the total number of statements in
the answer.
4 Results
4.1 Energy Consumption
Our results indicate that models vary in terms of their median energy consumption, with the climate-domain
specific chatbots CNZ and ChatNDC not necessarily consuming less energy compared to GPT 4o-Mini.
Figure 4 shows the median energy consumption per successful query across the four model configurations
evaluated. Among all systems, ChatNDC had the highest energy use, consuming4 .53×10−3kWh per query.
This consumption was substantially higher than GPT 4o-Mini, which consumed1 .13×10−3kWh, and less
energy when constrained to a response of 200 words or less (4o-200:5 .0×10−4kWh). CNZ had the lowest
9

Energy consumption of LLMsA Preprint
Figure 4: Median energy use of models per query.
Figure 5: Energy use across query steps: (a) Mean energy use of models by each step of the query; (b)
Proportion of energy use by step of the query.
mean energy consumption per query (4 .08×10−4kWh) and the smallest variation, with a mean absolute
deviation of1 .01×10−4. In contrast, ChatNDC showed both the highest average consumption and the largest
variability, consistent with a workflow that includes multiple model calls and a more elaborate post-retrieval
verification process compared to CNZ.
Figure 5 decomposes total query energy into retrieval, inference, and hallucination-checking components.
Across all models, inference accounts for the largest share of total energy use. For the pure GPT baselines,
energy consumption was almost entirely attributable to inference, since we were unable to assess estimates
of energy consumption for the retrieval stages. In the case of CNZ, inference also remained a dominant
component, while retrieval and its cosine-similarity-based hallucination check contributed only a small share
of the total energy use. This pattern is consistent with CNZ’s lightweight verification design, which does not
require an additional LLM call like ChatNDC.
10

Energy consumption of LLMsA Preprint
ChatNDC exhibited a markedly different energy profile from CNZ. Although inference remained the largest
single component of total energy use in both systems, hallucination checking accounted for a substantially
larger share in ChatNDC. Specifically, ChatNDC’s hallucination-checking stage represented 30.9% of total
energy use, compared with only 8.4% in CNZ. This disparity arises because ChatNDC performs statement-
level verification through an additional GPT 4o-Mini call, whereas CNZ relies on a simpler, non-agentic
cosine-similarity method. As a result, ChatNDC’s hallucination consumed approximately 45 times more
energy than CNZ’s lighter-weight algorithm. Retrieval, by contrast, remained the smallest energy component
in both RAG systems, and was similar in proportional terms across CNZ and ChatNDC. These results show
that the main difference in energy use between the two RAG pipelines lies not in retrieval, but in the much
more energy-intensive validation stage introduced by LLM-based hallucination checking.
The retrieval and hallucination steps observed in CNZ and NDC queries are part of the additional RAG
processes in their architectures. Although GPT models can search the internet when a topic is outside their
training data, the API does not expose explicit retrieval or hallucination steps that can be recorded.
Exploring the relation between output token size and Energy consumption (Figure 6), we observe a particularly
strong relation between both variables for the answers generated by CNZ, GPT 4o-Mini and 4o-Mini (200).
To the contrary the results of ChatNDC shows a weaker association between output token size and energy
consumption. These results highlight not only the direct relation between these two variables, but also the
potential effect of truncating the answers for increased energy efficiency. Finally, it also highlights that
depending on the system architecture that relation can be modified such as the case of ChatNDC where
output length is not strongly corrected to energy, due to the additional steps required by the model and its
configuration.
Figure 7 shows that mean factual scores are broadly similar across models, with only modest differences
relative to the variation across questions. By contrast, embellishment differs more clearly by system. CNZ
has the lowest mean embellishment score of CNZ (M = 0.15; SD = 0.16), which is notably significantly lower
than the three other models (GPT 4o-Mini (M = 0.55, SD = 0.27); GPT 4o-Mini (200) (M = 0.59, SD =
0.22), and ChatNDC (M = 0.62, SD = 0.14)). This result suggests that CNZ produces more concise and
tightly grounded responses, while the other systems tend to generate more elaborate outputs without clear
gains in factual performance.
Figure 8 reinforces this interpretation. There is no clear evidence that higher energy use is associated with
higher factual scores. Instead, systems with substantially greater energy consumption, especially ChatNDC,
achieve factual performance that is broadly similar to lower-energy alternatives whereas CNZ combines low
energy usage with low embellishment.
Figure 6: Output token vs Energy Usage by model.
11

Energy consumption of LLMsA Preprint
Figure 7: a) Mean factual score by model. b) Mean embellishment score by model.
Figure 8: a) Energy use vs. factual score of questions, by model. b) Energy use vs. embellishment score of
questions, by model.
Figure 8 explores the association between embellishment, factual accuracy and output length, as measured
by token size. Shorter outputs do not necessarily guarantee reduced embellishment, as observed in GPT
4o-Mini (200)’s relatively high embellishment score (score = 0.59). CNZ, however, which is prompted to
keep responses under 150 words [ 4], had a much lower embellishment score (0.15). Longer responses, such
as those generated by GPT 4o-Mini and ChatNDC, were more often associated with higher embellishment,
particularly ChatNDC. In ChatNDC’s case, this result appears to stem partly from its system design, which
requires responses between 400-600 words in a research-paper style. For instance, questions such as “How
many NDCs has Brazil submitted” that could be answered directly in a single sentence were often followed by
additional and sometimes embellished content to satisfy the target length. This result suggests that although
12

Energy consumption of LLMsA Preprint
Figure 9: a) Output token size vs. factual score of questions, by model. b) Output token size vs. embellishment
score of questions, by model.
RAG-based architectures can help reduce non-factual statements, those gains may be offset when models are
required to produce unnecessarily long answers.
5 Discussion
As interest in the environmental sustainability of LLMs has grown, increasing attention has been paid to the
energy costs associated with their real-world deployment. Simultaneously, RAG systems have been promoted
as a promising approach for improving response accuracy, grounding model outputs in external sources, and
reducing hallucinations in domain-specific applications [ 4]. Yet an important question this study attempts
to address is how the energy consumption of domain-specific RAG systems compare with the direct use of
generic LLMs.
Our findings show that while RAG can improve factual grounding, it introduces additional computational steps,
such as retrieval, routing, and post-generation verification that may increase inference-time energy use. While
domain-specific grounding may allow smaller or more constrained models to perform competitively without
relying on more expensive generic model outputs, understanding this trade-off is essential for evaluating
not only the quality benefits of RAG, but also its implications for the sustainable design of LLM-based
applications.
A central implication of our findings is that inference-time energy use is shaped not only by the underlying
model, but also by the amount of text a system is designed to generate. Prior work shows that inference
energy rises with both input and output length, with output length often exerting a particularly strong
influence because each newly generated token incurs additional decoding steps. Controlled experiments by
Poddar et al. [ 35] show near-linear energy growth with output length and also find that, when input and
output lengths are held constant, task complexity itself has only a small effect on energy use. Luccioni et
al. [16] likewise emphasize that generative tasks are substantially more energy-intensive than discriminative
ones, in part because generation requires repeated token production rather than a single prediction. In this
context, our results suggest that output-length decisions are not only an energy issue but also a quality
issue: systems that encourage longer or more elaborated answers may create more embellishment, reflected in
the higher embellishment scores. We do not interpret this as evidence that long answers are inherently less
factual, but it does suggest that expanding output length can increase both energy demand and the space for
non-essential or weakly grounded content.
Our comparison between CNZ and ChatNDC further shows that system design choices can have large
consequences for energy consumption even when systems rely on the same base generation model. Both
products are RAG systems, yet their energy profiles diverge sharply because of differences in workflow
structure. In our experiments, the main distinction was not retrieval itself, which remained a relatively small
share of total energy in both systems, but the additional inference steps introduced by ChatNDC’s more
13

Energy consumption of LLMsA Preprint
agentic verification process. This result is consistent with the broader agentic RAG literature, which describes
agentic pipelines as adding planning, reflection, iterative refinement, and tool use on top of standard retrieval
and generation [ 36]. Such steps may improve flexibility and robustness, but they also add inference-time
compute through additional model calls. Our results reinforce a key design point: from an energy perspective,
the major cost of agentic RAG may lie less in retrieval than in the accumulation of extra post-retrieval
reasoning and validation steps.
These findings point toward a more selective design strategy for future RAG systems. Rather than applying
the same high-cost workflow to every query, developers may achieve better energy efficiency by routing
questions according to their complexity and information requirements. Recent work on LLM routing [ 37]
and cascades [ 38] shows that simpler queries can often be handled by smaller or cheaper models, while
more complex questions are reserved for stronger models, thereby reducing cost without large quality losses.
Relatedly, sparse mixture-of-experts architectures offer another design path by activating only a subset
of model parameters for each token, which can improve capability while limiting active compute relative
to dense models [ 39,40]. For domain-specific RAG systems, these suggest a practical next step: simple
factual retrieval questions could be handled by a lightweight pipeline, whereas open-ended, comparative, or
reasoning-heavy questions could trigger a larger model or a more agentic verification stage. When developers
design domain-specific RAG systems, energy should become an explicit consideration alongside factual
accuracy, latency, and usability.
6 Limitations
This study is not without its limitations. First, our energy estimates rely on end-to-end response time
rather than direct server-side response time from the model provider. This challenge is a common constraint
in API-based LLM research, since proprietary services provide token counts and some response lifecycle
metadata but generally do not provide detailed server-side information needed for more transparent physical
measurement. Recent work on API-based energy estimation highlights this problem, noting that API-served
LLMs largely remain “black boxes” whose energy costs depend heavily on what providers choose to disclose
[41]. Therefore, our estimates should be interpreted as application-level approximations rather than direct
measurements of energy consumption.
Second, our experiments are limited to single question-response interactions. While this design improves
experimental control, it doesn’t fully reflect how LLM applications are used in practice, where users may input
follow-up prompts in multiple rounds of inference. In addition, LLM applications tend to incur live search
during user conversations, triggering additional energy usage that is not reflected in this research when using
generic LLM API calls. Future work should extend analysis from per-query energy estimation to per-session
energy to measure how repeated prompting, conversational memory, and iterative refinement alter both total
consumption and the energy-quality trade-off in generated responses. This extension is important because
emerging research on reasoning and test-time compute suggests that additional inference-time computation
can substantially increase energy demand as more tokens or steps are introduced.
Our study also prioritizes methodological refinement over broad benchmarking across multiple models. We
intentionally held the core model family relatively constant (GPT) in order to isolate the effects of workflow
design, but this choice limits the generalizability of our absolute energy estimates across providers, model
architectures, and hardware stacks. Prior benchmarking studies have shown substantial variation in inference
impacts across commercial APIs, and public benchmarking resources such as the ML.ENERGY Leaderboard
now cover dozens of model-task-hardware configurations, illustrating how strongly efficiency can differ across
models and deployment setups [ 14]. Future work could therefore expand the empirical scope of this framework
by testing a larger set of LLMs, broader question datasets, and more access locations in order to improve
external validity and support cross-model comparisons.
Last, both CNZ and ChatNDC rely on static, linear RAG workflows that lack a mechanism to calibrate
computational budgets relative to query complexity, resulting in energy redundancy for trivial tasks and
insufficiency for sophisticated reasoning. To address these limitations, future research should pivot toward
“Agentic Workflow” [ 42,43,44] frameworks that utilize autonomous decision chains for elastic reasoning. Such
frameworks can dynamically modulate search breadth and reflection depth, transitioning energy consumption
from rigid, fixed loads to adaptive, flexible loads. Also, integrating a Sparse Mixture-of-Experts (MoE)
architecture offers a synergistic pathway. These system designs aim to enhance the flexibility of previous
RAGs, balancing energy awareness with high-performance computing.
14

Energy consumption of LLMsA Preprint
7 Conclusion
This study examined the inference-time energy consumption of climate domain-specific LLM and RAG-based
chatbot systems by moving beyond model-level estimates to workflow-level evaluation under real user queries.
Our results show that energy use is shaped not only by the underlying model, but also by system design
choices, including retrieval structure, output length, and hallucination-checking method. In particular,
more agentic RAG workflows can substantially increase energy consumption without necessarily producing
proportional gains in response quality in the evaluation setting examined here. Our findings also suggest that
domain-specific RAG systems are not inherently more or less energy-intensive than generic LLM use, but their
energy profile depends on how the workflow is constructed. By jointly evaluating energy consumption and
answer quality, this study contributes a more application-relevant framework for assessing the sustainability
trade-offs of LLM-based chatbot design. Future research should extend this approach to multi-prompt and
follow-up interactions, a wider range of models and locations, and more dynamic agentic workflows.
Acknowledgements
This work was supported by grants from the IKEA Foundation and William and Flora Hewlett Foundation
to A.Hsu. The authors acknowledge the contributions of Mariam Maher, Arboretica, in the algorithmic
design of ChatNetZero. The development of ChatNDC was supported by an award from the Climateworks
Foundation, and the University of North Carolina at Chapel Hill supported the development of ChatNetZero
through a seed grant to A. Hsu.
References
[1]Veronika Samborska. Scaling up: how increasing inputs has made artificial intelligence more capable.
Our World in Data, January 2025.
[2]Konstantinos Vrettos and Michail E Klontzas. Accurate and Energy Efficient: Local Retrieval- Augmented
Generation Models Outperform Commercial Large Language Models in Medical Tasks.
[3]Saeid Vaghefi, Qian Wang, Veruska Muccione, Jingwei Ni, Mathias Kraus, Julia Bingler, Tobias
Schimanski, Chiara Colesanti Senni, Nicolas Webersinke, Christian Huggel, and Markus Leippold.
ChatIPCC: Grounding Conversational AI in Climate Science.SSRN Electronic Journal, 2023.
[4]Angel Hsu, Mason Laney, Ji Zhang, Diego Manya, and Linda Farczadi. Evaluating ChatNetZero, an
LLM-Chatbot to Demystify Climate Pledges. InProceedings of the 1st Workshop on Natural Language
Processing Meets Climate Change (ClimateNLP 2024), pages82–92, Bangkok, Thailand, 2024.Association
for Computational Linguistics.
[5]Zeinab Shahbazi and Sara Behnamian. Using Large Language Models to Detect and Debunk Cli-
mate Change Misinformation.Big Data and Cognitive Computing, 10(1), January 2026. Publisher:
Multidisciplinary Digital Publishing Institute.
[6]Sam Meredith. A ‘thirsty’ generative AI boom poses a growing problem for Big Tech, December 2023.
Section: COP28.
[7]Michael Copley. Data centers are booming. But there are big energy and environmental risks.NPR,
October 2025.
[8] IEA. Energy demand from AI – Energy and AI – Analysis, 2025.
[9]Sophia Chen. How much energy will AI really consume? The good, the bad and the unknown.Nature,
639(8053):22–24, March 2025. Bandiera_abtest: a Cg_type: News Feature Publisher: Nature Publishing
Group Subject_term: Machine learning, Computer science, Sustainability, Energy.
[10]Lynn H. Kaack, Priya L. Donti, Emma Strubell, George Kamiya, Felix Creutzig, and David Rolnick.
Aligning artificial intelligence with climate change mitigation.Nature Climate Change, 12(6):518–527,
June 2022.
[11]James O’Donnell and Casey Crownhart. We did the math on AI’s energy footprint. Here’s the story you
haven’t heard.MIT Technology Review, May 2025.
[12]Cooper Elsworth, Keguo Huang, David Patterson, Ian Schneider, Robert Sedivy, Savannah Goodman,
Ben Townsend, Parthasarathy Ranganathan, Jeff Dean, Amin Vahdat, Ben Gomes, and James Manyika.
Measuring the environmental impact of delivering AI at Google Scale, August 2025. arXiv:2508.15734
[cs].
15

Energy consumption of LLMsA Preprint
[13]Pengfei Li, Jianyi Yang, Mohammad A. Islam, and Shaolei Ren. Making AI Less "Thirsty": Uncovering
and Addressing the Secret Water Footprint of AI Models, March 2025. arXiv:2304.03271 [cs].
[14]Nidhal Jegham, Marwan Abdelatti, Chan Young Koh, Lassad Elmoubarki, and Abdeltawab Hendawi.
How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference, November
2025. arXiv:2505.09598 [cs].
[15] Net Zero Tracker. Net Zero Stocktake 2025. Technical report, NewClimate Institute, Oxford Net Zero,
Energy and Climate Intelligence Unit, and Data-Driven EnviroLab., 2025.
[16]Alexandra Sasha Luccioni, Yacine Jernite, and Emma Strubell. Power Hungry Processing: Watts Driving
the Cost of AI Deployment? InThe 2024 ACM Conference on Fairness Accountability and Transparency,
pages 85–99, June 2024. arXiv:2311.16863 [cs].
[17]David Patterson, Joseph Gonzalez, Quoc Le, Chen Liang, Lluis-Miquel Munguia, Daniel Rothchild,
David So, Maud Texier, and Jeff Dean. Carbon Emissions and Large Neural Network Training, April
2021. arXiv:2104.10350 [cs].
[18]Ariyanti Sri and Suryanegara Muhammad. Trade-Off Between Energy Consumption and Three Configu-
ration Parameters in Artificial Intelligence (AI) Training: Lessons for Environmental Policy.
[19]Ben Cottier, Robi Rahman, Loredana Fattorini, Nestor Maslej, Tamay Besiroglu, and David Owen. The
rising costs of training frontier AI models, February 2025. arXiv:2405.21015 [cs].
[20]Giovanni Cocca Guardia, Gabriel Hermosilla Vigneau, Danny Espín-Sarzosa, and Bárbara Dumas Feris.
Assessing the Energy Impact and Carbon Footprint of AI Model Training: A Case Study Using GPU
Servers.Resources, Conservation and Recycling, 2024.
[21]Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the Carbon
Emissions of Machine Learning, November 2019. arXiv:1910.09700 [cs].
[22]Brad Everman, Trevor Villwock, Dayuan Chen, Noe Soto, Oliver Zhang, and Ziliang Zong. Evaluating the
carbon impact of large language models at the inference stage. In2023 IEEE international performance,
computing, and communications conference (IPCCC), pages 150–157. IEEE, 2023.
[23]Maximilian Dauner and Gudrun Socher. Energy costs of communicating with AI.Frontiers in Commu-
nication, 10:1572947, 2025. Publisher: Frontiers Media SA.
[24]Sophia Nguyen, Beihao Zhou, Yi Ding, and Sihang Liu. Towards Sustainable Large Language Model
Serving.ACM SIGEnergy Energy Informatics Review, 4(5):134–140, December 2024.
[25]Ahmad Faiz, Sotaro Kaneda, Ruhan Wang, Rita Osi, Prateek Sharma, Fan Chen, and Lei Jiang.
LLMCarbon: Modeling the end-to-end Carbon Footprint of Large Language Models, January 2024.
arXiv:2309.14393 [cs].
[26]Zhenxiao Fu, Fan Chen, Shan Zhou, Haitong Li, and Lei Jiang. LLMCO 2: Advancing Accurate Carbon
Footprint Prediction for LLM Inferences.ACM SIGEnergy Energy Informatics Review, 5(2):63–68, July
2025.
[27]Samarth Sikand, Rohit Mehra, Priyavanshi Pathania, Nikhil Bamby, Vibhu Saujanya Sharma, Vikrant
Kaulgud, Sanjay Podder, and Adam P. Burden. Breaking the ICE: Exploring promises and challenges of
benchmarks for Inference Carbon & Energy estimation for LLMs. In2025 IEEE/ACM 9th International
Workshop on Green and Sustainable Software (GREENS), pages 55–59, April 2025. arXiv:2506.08727
[cs].
[28]Erik Johannes Husom, Arda Goknil, Lwin Khin Shar, and Sagar Sen. The Price of Prompting: Profiling
Energy Use in Large Language Models Inference, March 2026. arXiv:2407.16893 [cs].
[29]C. P. Ormell. Bloom’s Taxonomy and the Objectives of Education.Educational Research, 17(1):3–18,
November 1974.
[30] Tiktoken, March 2026. original-date: 2022-12-01T23:22:11Z.
[31] Microsoft. Measuring energy and water efficiency for Microsoft Datacenters.
[32] Intel. Intel Xeon Processor.
[33] Google. Power usage effectiveness – Google Data Centers.
[34]Anja Belz and Ehud Reiter. Comparing Automatic and Human Evaluation of NLG Systems. In Diana
McCarthy and Shuly Wintner, editors,11th Conference of the European Chapter of the Association
for Computational Linguistics, pages 313–320, Trento, Italy, April 2006. Association for Computational
Linguistics.
16

Energy consumption of LLMsA Preprint
[35]Soham Poddar, Paramita Koley, Janardan Misra, Niloy Ganguly, and Saptarshi Ghosh. Towards
SustainableNLP:InsightsfromBenchmarkingInferenceEnergyinLargeLanguageModels. InProceedings
of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 12688–12704, Albuquerque,
New Mexico, 2025. Association for Computational Linguistics.
[36]Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic Retrieval-Augmented
Generation: A Survey on Agentic RAG, February 2025. arXiv:2501.09136 [cs].
[37]Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed
Kadous, and Ion Stoica. RouteLLM: Learning to Route LLMs with Preference Data, February 2025.
arXiv:2406.18665 [cs].
[38]Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to Use Large Language Models While
Reducing Cost and Improving Performance, May 2023. arXiv:2305.05176 [cs].
[39]Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive Mixtures of
Local Experts.Neural Computation, 3(1):79–87, March 1991.
[40]Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff
Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, January
2017. arXiv:1701.06538 [cs].
[41]Lars Krupp, Daniel Geißler, Francisco M. Calatrava-Nicolas, Vishal Banwari, Paul Lukowicz, and Jakob
Karolus. This Is Taking Too Long – Investigating Time as a Proxy for Energy Consumption of LLMs,
March 2026. arXiv:2603.15699 [cs] version: 1.
[42]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct:
Synergizing Reasoning and Acting in Language Models, March 2023. arXiv:2210.03629 [cs].
[43]Yining Ye, Xin Cong, Shizuo Tian, Jiannan Cao, Hao Wang, Yujia Qin, Yaxi Lu, Heyang Yu, Huadong
Wang, Yankai Lin, Zhiyuan Liu, and Maosong Sun. ProAgent: From Robotic Process Automation to
Agentic Process Automation, November 2023. arXiv:2311.10751 [cs].
[44]Yuntao Wang, Shaolong Guo, Yanghe Pan, Zhou Su, Fahao Chen, Tom H. Luan, Peng Li, Jiawen Kang,
and Dusit Niyato. Internet of Agents: Fundamentals, Applications, and Challenges.IEEE Transactions
on Cognitive Communications and Networking, 12:4476–4501, 2026. arXiv:2505.07176 [cs].
[45]Diego Manya-Gutierrez, Alicia Bao, Jiamian He, Angel Hsu, and Ji Zhang. Replication Data for:
Assessing the Energy Consumption of LLMs and Domain Products, 2026.
Supplementary material
The experimental data are publicly available on the Data-Driven EnviroLab’s Dataverse page [45].
17