# Mathematical Reasoning for Unmanned Aerial Vehicles: A RAG-Based Approach for Complex Arithmetic Reasoning

**Authors**: Mehdi Azarafza, Mojtaba Nayyeri, Faezeh Pasandideh, Steffen Staab, Achim Rettberg

**Published**: 2025-06-05 13:09:24

**PDF URL**: [http://arxiv.org/pdf/2506.04998v1](http://arxiv.org/pdf/2506.04998v1)

## Abstract
Autonomous UAV operation necessitates reliable mathematical reasoning for
tasks such as trajectory planning and power management. While traditional
flight control relies on hardcoded equations, recent Large Language Models
(LLMs) offer potential for more flexible problem-solving but struggle with
reliably selecting and applying correct mathematical formulations and executing
precise multi-step arithmetic. We propose RAG-UAV, a retrieval-augmented
generation framework designed to improve the mathematical reasoning of several
LLMs (including GPT o1/Turbo, Llama-3.2/3.3, Mistral, and DeepSeek R1) in
UAV-specific contexts by providing access to relevant domain literature. To
conduct an initial assessment, we introduce the UAV-Math-Bench, a small problem
set comprising 20 UAV-centric mathematical problems across four difficulty
levels. Our experiments demonstrate that incorporating retrieval substantially
increases exact answer accuracy (achieving up to 75% with o1), reduces
instances of incorrect formulation selection (from 25% without RAG to 5% with
RAG), decreases numerical errors, reducing Mean Squared Error (MSE) by orders
of magnitude for the best-performing models. This pilot study indicates that
RAG can enable general-purpose LLMs to function as more reliable tools for
engineering analysis, although direct real-time flight control requires further
investigation and validation on a larger scale. All benchmark data, question
and answer are publicly available.

## Full Text


<!-- PDF content starts -->

arXiv:2506.04998v1  [cs.AI]  5 Jun 2025MATHEMATICAL REASONING FOR UNMANNED AERIAL
VEHICLES : A RAG-B ASED APPROACH FOR COMPLEX
ARITHMETIC REASONING
A P REPRINT
Mehdi Azarafza
Department of Computer Science
Hamm-Lippstadt University of Applied Sciences
Lippstadt, Germany
mehdi.azarafza@hshl.de
Mojtaba Nayyeri
Institute For Artificial Intelligence
University of Stuttgart
Stuttgart, Germany
mojtaba.nayyeri@ipvs.uni-stuttgart.de
Faezeh Pasandideh
Department of Computer Science
Hamm-Lippstadt University of Applied Sciences
Lippstadt, Germany
faezeh.pasandideh@hshl.de
Steffen Staab
Institute for Artificial Intelligence
University of Stuttgart, Germany
Stuttgart, Germany
Steffen.Staab@ipvs.uni-stuttgart.de
Achim Rettberg
Department of Computer Science
Hamm-Lippstadt University of Applied Sciences
Lippstadt, Germany
achim.rettberg@hshl.de
June 6, 2025
ABSTRACT
Autonomous UA V operation necessitates reliable mathematical reasoning for tasks such as trajectory
planning and power management. While traditional flight control relies on hardcoded equations,
recent Large Language Models (LLMs) offer potential for more flexible problem-solving but struggle
with reliably selecting and applying correct mathematical formulations and executing precise multi-
step arithmetic. We propose RAG-UA V , a retrieval-augmented generation framework designed to
improve the mathematical reasoning of several LLMs (including GPT o1/Turbo, Llama-3.2/3.3,
Mistral, and DeepSeek R1) in UA V-specific contexts by providing access to relevant domain literature.
To conduct an initial assessment, we introduce the UA V-Math-Bench, a small problem set comprising
20 UA V-centric mathematical problems across four difficulty levels. Our experiments demonstrate
that incorporating retrieval substantially increases exact answer accuracy (achieving up to 75% with
o1), reduces instances of incorrect formulation selection (from 25% without RAG to 5% with RAG),
decreases numerical errors, reducing Mean Squared Error (MSE) by orders of magnitude for the
best-performing models. This pilot study indicates that RAG can enable general-purpose LLMs
to function as more reliable tools for engineering analysis, although direct real-time flight control
requires further investigation and validation on a larger scale. All benchmark data, question and
answer are publicly1available.
1https://github.com/Mehdiazarafza/UA V-RAG

RAG-UA V (Azarafza et al.) A P REPRINT
o1 without RAG:
Pprofile≈1
8ρσC D0πR2(ΩR)3×h
1 +k v
ΩR2i
Model Output
A: The answer is Insufficient input data. ✗o1 + RAG
Pblade(V) =P0
1 +3V2
Ω2R2
.
Model Output
A: The answer is approximately 1.79. ✓.
Figure 1: Comparison of Standard o1 and RAG with Prompting for Arithmetic Reasoning in UA V Applications
Keywords Large Language Model ·Retrieval-Augmented Generation ·Unmanned Aerial Vehicle ·UA V-Math-Bench
1 Introduction
Unmanned Aerial Vehicles (UA Vs) are aircraft that operate without an onboard pilot, enabling a wide range of
applications from surveillance and mapping to delivery and emergency response. Autonomous UA V missions routinely
demand multi-step mathematical reasoning—for instance, balancing energy budgets while re-planning swarms in
variable wind. Existing flight-control code handles fixed equations, but engineers still spend time locating the right
aerodynamic, communication-channel, or power-consumption formulas and checking the arithmetic. This process
demands a solid understanding of physics, systems engineering, and domain-specific formulas, and solid mathematical
reasoning.
Large Language Models (LLMs), pretrained on vast corpora including physics texts and UA V design literature, offer a
powerful new tool for assisting engineers. By leveraging their embedded knowledge, LLMs can help identify relevant
equations, suggest performance evaluation strategies, validate reasoning steps, and even automate portions of UA V
planning and analysis. This positions LLMs as promising co-pilots in the engineering design loop, augmenting human
expertise and accelerating development cycles. While LLMs show promise for interpreting engineering queries and
suggesting relevant design principles, they also exhibit important limitations. Their responses are constrained by the
static nature of their pretraining data and may lack access to up-to-date, domain-specific documentation, standards, or
experimental datasets. Moreover, LLMs may "hallucinate"—generating plausible but incorrect equations or assumptions,
especially when faced with technical edge cases or incomplete prompts. Retrieval-Augmented Generation (RAG) offers
a practical solution by grounding LLM outputs in authoritative external sources. In this hybrid approach, relevant
documents, retrieved at query time and used to condition the model’s response. This improves factual accuracy, enhances
trustworthiness, and enables more context-aware assistance for engineers working on UA V design and operations.
1.1 Motivation Example
Consider the task of computing the blade power of a UA V in forward flight, given specific parameters. As illustrated in
Figure 1, a prominent LLM without access to external knowledge fails to identify the correct physical formulation,
deeming the provided data "insufficient" (Appendix A.2). In contrast, when augmented with relevant domain literature
via Retrieval-Augmented Generation (RAG), the same model successfully retrieves the appropriate equation, follows
detailed reasoning steps, and computes an accurate result (Appendix A.3). This example highlights a key challenge:
LLMs may possess implicit knowledge but struggle with reliable access and application of domain-specific mathematical
models, motivating our investigation into RAG as a solution. Figure 1 illustrates the results generated by the o1 model
for the query:
" Consider the following information:
• The constant value P0: 1.5
• The forward speed v: 110 km/h
• The blade angular velocity Ω: 300 radians/second
• The rotor radius R: 0.4 m
Based on this information, compute the Blade power of UA V in forward flight."
2

RAG-UA V (Azarafza et al.) A P REPRINT
1.2 Contribution
In this work, we investigate how RAG can enhance the mathematical reasoning capabilities of LLMs for domain-specific
UA V applications. We equip several LLMs with a vector-based retrieval mechanism over a corpus of UA V literature.
Our contributions include:
•RAG-UA V Framework: We propose and implement a RAG framework tailored for improving mathematical
and physical reasoning in UA V scenarios.
•UA V-Math-Bench: We construct a categorized problem set of 20 UA V-centric mathematical reasoning problems
across four difficulty levels to facilitate initial evaluation.
•Empirical Analysis: We conduct experiments comparing RAG-augmented and non-RAG LLM performance
(ChatGPT-Turbo), quantifying improvements in answer accuracy, formulation selection, and numerical error
across different models and problem complexities.
• Analysis of RAG Benefits: We demonstrate that RAG leads to substantial performance gains, particularly on
more complex, multi-step problems, validating its potential for enhancing LLM-based engineering analysis in
this domain.
2 Related Work
We review the relevant literature in two key areas related to this work: (a) retrieval-augmented generation for reasoning,
and (b) LLM in UA Vs.
2.1 Retrieval-Augmented Generation for Reasoning
Retrieval-augmented generation (RAG) is a method that combines language model generation with conventional
information retrieval to extend the knowledge and reasoning limitations of LLMs. RAG was shown to significantly
boost performance on knowledge-intensive NLP tasks by allowing a seq2seq model to consult a non-parametric memory
(a Wikipedia index) during generation [ 1]. By conditioning on retrieved text, the model produced more accurate
and specific outputs than a purely parametric baseline and even surpassed specialized retrieve-and-read pipelines on
open-domain question answering. RAG’s general recipe is to embed the query, use vector similarity search to find
semantically relevant documents, and then inject those documents (or snippets) into the model context.
While early applications of RAG focused on factual QA and knowledge retrieval, recent studies have adapted the
concept to improve logical and mathematical reasoning. Other works have proposed related ideas, such as retrieving
analogous solved problems to guide the model (a form of case-based reasoning) or using knowledge graphs to provide
constraints during reasoning. All of these efforts point to the value of external knowledge in complementing the
reasoning process of LLM. Retrieval can offer helpful anchors in the context of arithmetic, such as reminding the model
of a relevant formula, a well-known arithmetic rule, or an example solution that it can adapt. Our approach leverages
this insight by providing each arithmetic query with related reference content drawn from a prepared corpus (papers),
effectively bridging pure linguistic reasoning with algorithmic assistance from examples.
2.2 LLM in UA Vs
The intersection of LLMs and UA Vs has opened new avenues for high-level autonomous decision-making in aerial
robotics. Researchers have begun to integrate large language models into UA V control loops for tasks that require
understanding complex instructions and making context-aware decisions. For example, TypeFly demonstrated an
end-to-end system where an LLM translates natural language commands into drone flight actions, effectively piloting a
quadcopter via language [ 2]. Similarly, the REAL framework incorporated an LLM as part of a UA V’s planning and
control system, using the model to reason about goals and constraints during navigation [ 14]. These approaches treat
the LLM as a high-level “brain” that can interpret mission objectives, perform reasoning (e.g., break down a task into
subtasks), and interface with low-level APIs or controllers. In multi-agent scenarios or human-UA V interaction, LLMs
can serve as cognitive agents that understand dialogues or coordinate strategies among UA V teammates. Initial case
studies report that LLM-driven UA Vs can handle novel, complex directives more flexibly than pre-programmed systems.
However, a recurring observation is that the reliability of such systems is fundamentally limited by the LLM’s reasoning
fidelity. UA V control demands precise, logically consistent decisions, and LLMs, if not properly constrained, may
produce outputs that violate physical constraints or Our work aligns with this goal by tackling one particular weakness –
mathematical reasoning – which is vital for UA V decision-making (e.g., calculating distances, timing sequences, or
resource allocations). By improving mathematical accuracy through RAG, we aim to reduce one source of error in
LLM-driven UA V systems, thereby contributing to safer and more trustworthy autonomous operations.
3

RAG-UA V (Azarafza et al.) A P REPRINT
3 Methodology
3.1 RAG-UA V Framework
We propose a RAG framework designed to augment LLMs with domain-specific mathematical knowledge. As illustrated
in Figure 2, the framework operates in several stages: (1) A user query is input, typically a mathematical problem related
to UA Vs. (2) The query is embedded into a vector representation using a chosen embedding model. (3) This query
embedding is used to perform a similarity search against a vector database containing embedded text segments from a
corpus of UA V literature. (4) The most relevant text segments are retrieved. (5) The retrieved chunks are combined
with the original user query into a structured prompt. (6) This augmented prompt is fed to the LLM, which generates a
response that includes the solution to the mathematical problem.
3.2 Constructing UA Vs’ Vector Database
Our knowledge base was constructed from a corpus of five research papers relevant to UA V mathematical modeling
[6,11,10,9,8]. Initially, our investigation involved processing PDFs directly fed to the models. However, we
identified inaccuracies in the models’ output that were subsequently attributed to errors during PDF parsing rather than
inherent limitations in their core reasoning abilities. To accurately capture mathematical formulations, we processed
the PDF documents using Mathpix2to convert them into LaTeX format. This process aimed to preserve equations
and symbols more accurately than direct PDF parsing. The resulting LaTeX output was segmented into page-level
chunks, yielding a total of 162 segments for the five papers. These segments were then embedded using two different
models: all-mpnet-base-v2 (768 dimensions)[ 5] for open-source LLMs and OpenAI’s text-embedding-ada-002 (1536
dimensions)[3] for OpenAI models. The resulting vector representations were stored in a Chroma vector database.
where [6]
For running open-source models, we used Ollama [4] and for OpenAI models, we used the API.
Figure 2: Workflow for RAG-UA V
3.3 LLM Prompting Strategy
The prompt template used to query the LLMs incorporates both the original problem statement and the retrieved text
segments. The template instructs the model to utilize the provided documents for identifying relevant formulations but
allows the model to use its internal capabilities for performing the numerical calculations. A simplified structure of the
prompt is: "Use only the provided documents: [Retrieved Documents]. Question: [Original Question]. This strategy
was chosen to isolate the effect of retrieval on formulation selection while allowing the LLM to perform the numerical
computation, leveraging its inherent arithmetic abilities. It also reflects a potential real-world use case where an LLM
might be provided with domain knowledge but still rely on its internal calculation engine. The full prompt template is
provided in Appendix A.1.
2https://mathpix.com/
4

RAG-UA V (Azarafza et al.) A P REPRINT
3.4 UA V-Math-Bench: Evaluation Problem Set
To evaluate the complexity of computational tasks and analytical reasoning required in solving the provided problems,
the questions are categorized into four distinct levels: Basic, Intermediate, Advanced, and Expert. The evaluation set
serves to demonstrate the potential of the RAG-UA V framework and highlight performance differences between models
and task complexity. This classification considers the number of parameters involved, the mathematical complexity,
and the depth of domain-specific knowledge required to obtain a solution. The development of UA Vs is intrinsically
linked to this increasing complexity. As UA V technology advances, the challenges associated with their deployment
grow more intricate, requiring sophisticated problem-solving techniques, advanced mathematical models, and a deep
understanding of engineering principles. The categorization of difficulty levels thus reflects the rising complexity of
real-world UA V-related problems encountered by professionals today.
3.4.1 Basic Level Questions
Basic level questions require direct computation based on a given set of parameters, typically involving straightforward
mathematical formulas. These problems involve fundamental arithmetic or algebraic manipulations without requiring
extensive domain knowledge.
•Q1-1: Computation of elevation angle speed based on coordinate positions.
•Q1-2: Calculation of blade power using given constants and basic aerodynamics equations.
•Q1-3: Computational latency determination based on CPU allocation and task parameters.
•Q1-4: Atmospheric attenuation factor computation using given wavelength and scattering distribution.
•Q1-5: Received power computation using path loss and channel gain formulas.
At this level, UA Vs are typically used for simple tasks such as altitude measurement and power consumption estimation,
often relying on basic physics and aerodynamics.
3.4.2 Intermediate Level Questions
Intermediate questions extend beyond direct computations and require additional steps, such as probability computations
or logarithmic transformations. These problems involve a moderate level of engineering knowledge and mathematical
formulation.
•Q2-1: Probability computation of a line-of-sight (LoS) connection based on given probability constants.
•Q2-2: Induced power computation in forward UA V flight using additional aerodynamics principles.
•Q2-3: Data transmission rate calculation using Shannon’s theorem.
•Q2-4: Achievable data rate computation requiring logarithmic calculations.
•Q2-5: Power consumption due to communication, incorporating multiple power-related parameters.
At this intermediate level, UA Vs become more complex in their operations, handling tasks such as communication
protocols, energy management, and data transmission in various environmental conditions.
3.4.3 Advanced Level Questions
Advanced questions involve multi-step problem-solving approaches that integrate multiple mathematical principles.
These computations require a solid grasp of signal processing, aerodynamics, and wireless communication concepts.
•Q3-1: Path loss calculation for a LoS connection incorporating frequency and speed of light.
•Q3-2: Parasite power computation based on aerodynamic drag and air density factors.
•Q3-3: Downlink transmission rate computation using transmission bandwidth and noise power.
•Q3-4: Bandwidth allocation to meet a specific data rate requirement.
•Q3-5: Power consumption due to hovering, integrating multiple aerodynamic and energy parameters.
In this phase, UA Vs are employed for complex tasks such as networked communication between UA Vs, long-duration
flights, and advanced environmental monitoring, where power efficiency and reliable data transmission are critical.
5

RAG-UA V (Azarafza et al.) A P REPRINT
Table 1: Categorization of questions by difficulty level, number of variables, and nested equations.
Difficulty Question # Vars # Nest. Eqns
BasicQ1-1 2 1
Q1-2 3 1
Q1-3 2 1
Q1-4 2 1
Q1-5 3 1
IntermediateQ2-1 2 2
Q2-2 4 2
Q2-3 3 1
Q2-4 3 1
Q2-5 4 2
AdvancedQ3-1 4 2
Q3-2 5 2
Q3-3 4 2
Q3-4 3 2
Q3-5 5 3
ExpertQ4-1 6 3
Q4-2 6 3
Q4-3 5 3
Q4-4 6 4
Q4-5 7 4
3.4.4 Expert Level Questions
Expert-level questions demand an extensive understanding of multiple concepts and their integration into a single
computational framework. These problems often require optimization, iterative solutions, or advanced probabilistic and
statistical analyses.
•Q4-1: Average path loss computation involving LoS and NLoS path loss models.
•Q4-2: Total UA V power consumption in forward flight, integrating multiple power components.
•Q4-3: Transmission delay computation based on bandwidth, power, and noise parameters.
•Q4-4: Utility ratio determination requiring multi-variable calculations and resource allocation.
•Q4-5: Comprehensive UA V power consumption computation incorporating aerodynamic, communication,
and electrical parameters.
At the expert level, UA Vs are operating in highly sophisticated environments, with tasks requiring optimal energy
management, high-level coordination, and complex decision-making algorithms, often for missions such as autonomous
fleet management or large-scale environmental surveying [15].
Table 1 categorizes the questions by difficulty, with additional details on the number of variables and nested equations.
4 Evaluation and Results
This section presents the experimental setup and results demonstrating the impact of RAG on the mathematical
reasoning performance of various LLMs on the UA V-Math-Bench. We compare RAG-augmented models against
ChatGPT non-RAG baselines and other models with RAG using three primary metrics: Exact Answer Accuracy,
Formulation Selection Error Rate, and Mean Squared Error (MSE).
4.1 Evaluation Metrics
Exact Answer Accuracy: This metric measures the percentage of problems for which the model produces a final
numerical answer that precisely matches the ground truth answer provided by human experts. Answers must match
within a predefined tolerance for floating-point comparisons ( 10−4), where applicable.
6

RAG-UA V (Azarafza et al.) A P REPRINT
Table 2: Comparison of performance metrics across evaluated LLMs.
Metric Chat o1 o1-p Turbo Mist L-3B R1 L-70B
Correct (%) 30.00 75.00 70.00 60.00 5.00 10.00 50.00 35.00
Correct Form. (%) 75.00 95.00 85.00 75.00 70.00 60.00 90.00 65.00
Miscalc. (%) 45.00 15.00 10.00 20.00 55.00 50.00 35.00 30.00
MisC†(%) 60.00 21.05 11.76 26.67 78.57 83.33 38.89 46.15
†MisC = Miscalculation among correctly formulated answers.
Chat = ChatGPT Turbo, L-3B = LLaMA3.2 3B, L-70B = LLaMA3.3 70B, R1 = DeepSeek R1, Mist = Mistral 7B.
Miscalc. = Miscalculation (out of total attempts).
Formulation Selection Error Rate: This metric quantifies the percentage of problems where the model either selects
an incorrect mathematical formula or fails to identify any relevant formula from the provided context. This is assessed
through the expert manual inspection of the model’s step-by-step reasoning process.
Mean Squared Error (MSE): We calculate the MSE between the model’s final numerical answer and the ground truth
answer across the benchmark. Due to potentially large differences between model outputs and ground truths, we report
results using log10(MSE)for clearer visualization. A lower MSE indicates closer agreement with the reference answer.
4.2 Comparison of models performance
To ensure a comprehensive comparison, we employed the correct formulations and verified the answers in all LLM.
The complete set of answers, the corresponding questions and the details of the model are available on the GitHub
repository of the paper.
Table 2 compares the performance of several LLMs based on key metrics. We evaluated both correct and wrong
answers, and if a model did not answer or asked for more information, we counted that as zero. ChatGPT is the only
model without RAG. Among the models, ol performed the best with correct answers 75% and the lowest error in the
calculations. OpenAI’s o1 model consistently identified the correct formulation and computed values close to the
reference answers, resulting in a considerably lower MSE compared to others. This model also provided optimized
and well-structured answers, demonstrating high reasoning capabilities in complex questions. Notably, OpenAI’s o1
model demonstrated a degree of semantic robustness, capable of interpreting physical constants such as the speed of
light despite variations in user input, including misspellings and non-standard variable assignments. For instance, when
presented with the statement defining ’G’ as the "speed light", the model inferred the intended concept, correlating ’c’
with the symbol for the speed of light.
Models like Mistral 7B and LLaMA 3.2 3B gave mostly wrong answers and made many mistakes. Mistral 7B
demonstrated lower performance overall. DeepSeek 70B[ 13] and LLaMA 3.3 70B gave more accurate answers than
smaller models, and DeepSeek made fewer mistakes overall. DeepSeek provides very detailed and precise answers
(AppendixA.4) and is considered the best open-source LLM model in our evaluation. ChatGPT-4-Turbo without rang
and Turbo with RAG were in the middle - they could answer some questions but often miscalculated. This shows that
larger models and better training are important in answering technical questions such as those about RAG UA Vs.
The o1-preview model demonstrated the most robust performance, exhibiting a low miscalculation rate of 11.76% when
correct formulations were achieved. In contrast, Mistral 7B and Llama 3.2 3b showed markedly higher miscalculation
rates (78.57% and 83.33%, respectively), suggesting a significant challenge in maintaining computational accuracy for
these models.
Figure 7 compares the performance of several LLMs based on MSE. Since MSE values are extremely large due to the
significant differences between human responses and model predictions, we used a log 10 scale to make the evaluation
clearer. The results show that o1 outperforms all other models dramatically with an MSE of just 214.74.
In particular, significant deviations can occur depending on the model’s ability to select and apply the correct formulation.
For example, in one benchmark question, the expert-provided ground truth is 64,207,051.66, while ChatGPT returned a
significantly lower value of 70.2. Similar differences are observed in several models. However, the o1 model consistently
identified the correct formulation and computed values close to the reference answers, resulting in a considerably lower
MSE compared to others.
4.3 Performance by Difficulty Level
In this section we provide a comparative evaluation of eight large language models across four difficulty tiers, summa-
rized in Table 3.
7

RAG-UA V (Azarafza et al.) A P REPRINT
Figure 3: Compare state-of-the-art LLMs
Table 3: Accuracy comparison across difficulty levels for evaluated models.
Level Chat o1 o1-p Turbo M7B L3-3B R1 L3-70B
Basic 60% 80% 100% 100% 20% 0% 60% 60%
Intermediate 0% 60% 80% 60% 0% 20% 40% 60%
Advanced 40% 80% 60% 60% 0% 0% 80% 20%
Expert 20% 80% 40% 20% 0% 20% 20% 0%
Chat = ChatGPT, o1-p = o1-preview, Turbo = GPT-4 Turbo, M7B = Mistral 7B,
L3-3B = LLaMA 3.2 3B, R1 = DeepSeek R1, L3-70B = LLaMA 3.3 70B.
The o1 model outperforms all others, achieving 80% accuracy even on Expert-level questions. While most models
handle Basic questions well, performance drops notably for harder levels. Models like Mistral 7B and LLaMA 3.2
3B perform poorly overall, especially on Advanced and Expert tasks. ChatGPT and DeepSeek R1 show moderate
accuracy, but only o1 maintains consistently high performance across all levels. Interestingly, the O1 model achieved
80% accuracy on Expert-level problems, a performance not mirrored at the Intermediate level, where two errors
occurred. One specific instance involved a slight numerical deviation in the model’s output (0.5535 vs. the correct 0.5),
classified as incorrect under rigorous human expert assessment, even though the model itself indicated the result was
’approximately 0.50.’ For another mistake from the model, there was a unit conversion error: we assumed θis in radians
and applied a conversion to degrees, but o1 already considered the angle in degrees. This led to an incorrect input to the
exponential term and significantly altered the result.
While Expert problems involve more complex chains of reasoning and integration of multiple concepts, the model might
excel at retrieving and structuring these high-level solutions. However, it may be less robust to handling refined details
like precise numerical tolerances or implicit unit conventions that were critical for correctness in some Intermediate
problems.
4.4 Examination of Attention Patterns
Understanding the internal processes of LLMs, particularly in complex tasks like mathematical reasoning, remains an
active research area [ 12]. To gain preliminary insights into how the Deep Seek R1 model processes input and potentially
leverages information, we visualized the attention patterns of its first layer for a selected question and its generated
answer.
Figure 4 shows attention patterns for two different heads for first layer, revealing similar diagonal and localized attention
structures within the generated sequence.
8

RAG-UA V (Azarafza et al.) A P REPRINT
(a) Attention Matrix: Head 1 of DeepSeek R1
 (b) Attention Matrix: Head 15 of DeepSeek R1
Figure 4: Attention matrix visualization for different heads in DeepSeek R1 (Layer 1).
Figure 5 highlights attention weights during the calculation of the horizontal distance, showing the model attending to
relevant numerical tokens from the input.
Horizontal distance =p
(65−47)2+ (73−23)2
Figure 6 illustrates the tokens involved in calculating theta, illustrating that the model also considers the previously
computed token "horizontal distance" in its calculations.
Figure 5: Distance Calculation Attention Weights
9

RAG-UA V (Azarafza et al.) A P REPRINT
Figure 6: Equation Finding Attention Weights
Figure 7: MSE Comparison (Log10(MSE))
10

RAG-UA V (Azarafza et al.) A P REPRINT
5 Conclusion
In this work, we demonstrated that retrieval-augmented generation (RAG) can substantially improve the mathematical
reasoning capabilities of Large Language Models (LLMs), particularly within the context of domain-specific UA V
applications requiring access to physical models and accurate multi-step computations. By providing LLMs with
relevant external knowledge via RAG, we observed a significant increase in their ability to correctly select formulations
and compute accurate numerical results on a set of UA V-centric mathematical problems.
Our experiments showed consistent gains in accuracy and substantial reductions in numerical error across several
LLMs when augmented with RAG. These improvements were most pronounced on problems involving multiple steps
and nested operations, suggesting that RAG effectively mitigates common failure modes of standalone LLMs on
complex mathematical tasks. While this study utilizes a limited problem set, the findings indicate that RAG holds
significant potential for enabling LLMs to function as more reliable tools for engineering analysis in technical domains
like UA V development, by providing them with the necessary domain knowledge to perform complex calculations more
accurately.
Limitations
This work demonstrates that Retrieval-Augmented Generation (RAG) substantially enhances Large Language Models’
(LLMs) mathematical reasoning, improving accuracy and reducing errors in UA V-specific multi-step computations by
providing relevant external knowledge. However, this study’s findings are limited by its small benchmark size and
further practical deployment requires rigorous evaluation of the RAG framework’s latency, computational cost, and
real-time performance.
While LLMs show promise for engineering support, their direct integration into autonomous decision-making processes
for UA Vs remains a critical area requiring further research and validation due to safety-critical implications.
References
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp
tasks. In Advances in Neural Information Processing Systems , volume 33, pages 9459–9474, 2020.
[2]Wenhao Wang, Yanyan Li, Long Jiao, Jiawei Yuan. GSCE: A prompt framework with enhanced reasoning for
reliable LLM-driven drone control. arXiv preprint arXiv:2502.12531 , 2025.
[3]Zilliz. text-embedding-ada-002: AI Model for Text Embeddings. https://zilliz.com/ai-models/
text-embedding-ada-002 , 2024. Accessed: March 25, 2025.
[4]Ollama. Ollama Open-Source Model Platform. https://ollama.com/ , 2025. Accessed: March 10, 2025.
License MIT.
[5]Hugging Face. all-mpnet-base-v2: Pretrained Sentence Transformer Model. https://huggingface.co/
sentence-transformers/all-mpnet-base-v2 , 2025. Accessed: March 25, 2025. Licence apache-2.0.
[6]Leiyu Wang, Haixia Zhang, Shuaishuai Guo, Dongfeng Yuan. Deployment and association of multiple UA Vs in
UA V-assisted cellular networks with the knowledge of statistical user position. IEEE Transactions on Wireless
Communications , 21(8):6553–6567, 2022.
[7]Yonglin Tian, Fei Lin, Yiduo Li, Tengchao Zhang, Qiyao Zhang, Xuan Fu, Jun Huang, Xingyuan Dai, Yutong
Wang, Chunwei Tian, et al. UA Vs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility.
arXiv preprint arXiv:2501.02341 , 2025.
[8]Attai Ibrahim Abubakar, Michael S. Mollel, Oluwakayode Onireti, Metin Ozturk, Iftikhar Ahmad, Syed Muhammad
Asad, Yusuf Sambo, Ahmed Zoha, Sajjad Hussain, Muhammad Ali Imran. Coverage and throughput analysis of an
energy efficient UA V base station positioning scheme. Computer Networks , 232:109854, 2023. Elsevier.
[9]Liangkun Yu, Xiang Sun, Sihua Shao, Yougan Chen, Rana Albelaihi. Backhaul-aware drone base station placement
and resource management for FSO-based drone-assisted mobile networks. IEEE Transactions on Network Science
and Engineering , 10(3):1659–1668, 2023. IEEE.
11

RAG-UA V (Azarafza et al.) A P REPRINT
[10] Qian Liu, Hairong Liang, Rui Luo, Qilie Liu. Energy-efficiency computation offloading strategy in UA V aided
V2X network with integrated sensing and communication. IEEE Open Journal of the Communications Society ,
3:1337–1346, 2022. IEEE.
[11] Kyungho Ryu, Wooseong Kim. Energy efficient deployment of aerial base stations for mobile users in multi-hop
UA V networks. Ad Hoc Networks , 157:103463, 2024. Elsevier.
[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang.
Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997 , 2023.
[13] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, et al. DeepSeek-R1:
Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
[14] Andrea Tagliabue, Kota Kondo, Tong Zhao, Mason Peterson, Claudius T. Tewari, Jonathan P. How. Real:
Resilience and adaptation using large language models on autonomous aerial robots. In 2024 IEEE 63rd Conference
on Decision and Control (CDC) , pages 1539–1546. IEEE, 2024.
[15] Faezeh Pasandideh, Alireza Najafzadeh, João Paulo Javidi da Costa, Giovanni Almeida Santos, Daniel Valle de
Lima, Edison Pignaton de Freitas. Providing an energy efficient UA V BS positioning mechanism to improve wireless
connectivity. Ad Hoc Networks , 170:103767, 2025. https://www.sciencedirect.com/science/
article/pii/S1570870525000150 .
A Appendix
This section presents the appendix, which includes the prompt templates, questions, and corresponding answers related
to the models.
A.1 Prompt template
We used the following prompt in the RAG process:
To answer the questions, use only the documents provided. Formulations are provided for all the questions in the
provided documents; however, to calculate the result, you may use your calculation skills.
context Question: question
A.2 Question 1-2: GPT o1 Answer without RAG
Prompt
Consider this information:
• The constant value p0:1.5
• The forward speed v:110 km h−1
• The blade angular velocity Ω:300 rad s−1
• The rotor radius R:0.4 m
Based on this information, compute the Blade power of UA V in forward flight?
12

RAG-UA V (Azarafza et al.) A P REPRINT
o1 Response without RAG
A key difficulty with this problem is that, as stated, there is not enough information to arrive at a unique
numerical value for the rotor (blade) power in forward flight. In helicopter-type rotor analyses, one typically
needs at least:
• The vehicle’s weight (or total thrust the rotor must produce).
• The air density ( ρ).
• Basic blade-geometry parameters (number of blades, chord, airfoil drag coefficient, etc.).
• A suitable rotor-theory formula (e.g., blade-element or momentum theory).
Simply giving
•p0= 1.5(an unspecified “constant”),
• forward speed v= 110 km/h (≈30.6m/s),
• angular velocity Ω = 300 rad/s,
• rotor radius R= 0.4m
does not pin down the aerodynamic power.
Why More Information is Needed
1. In forward flight, total rotor power typically has at least two major components:
• Induced (or “lift-induced”) power, often estimated from momentum theory.
•Profile (or “parasite”) power due to drag on the spinning blades themselves, often estimated from
blade-element theory.
2.Both components depend on air density ( ρ) and on how much thrust the rotor must generate (which in
turn depends on the UA V’s weight and flight condition).
3.Even if one assumes the thrust requirement is known, there are empirical or semi-empirical “constants”
in the formulas (such as blade drag coefficients, tip-loss factors, etc.) that must be specified or lumped
into a known expression before a numeric answer can come out.
Typical Example Formulas
By way of illustration (not a complete solution), in many helicopter texts one finds a “profile-power” formula of
the form
Pprofile≈1
8ρσC D0πR2(ΩR)3×
1 +kv
ΩR2
,
where
•σis the rotor solidity (depends on blade chord and number of blades),
•CD0is the airfoil’s drag coefficient,
•kis an empirical factor ( ≈4to6),
•vis forward speed, and
•ΩRis the blade tip speed.
Even here, you can see one must know ρ,σ,CD0, and so on.
Conclusion
Because the problem as posed does not provide the rotor thrust (or vehicle weight), the air density, or enough
blade-geometry/aerodynamic details, there is no way to turn the given numbers into a single “watts” (or kW)
value for rotor power. One can only proceed once those other parameters and the specific power formula are
specified.
13

RAG-UA V (Azarafza et al.) A P REPRINT
A.3 Question 1-2: GPT o1 Answer with RAG
Prompt
Consider this information:
• The constant value p0:1.5
• The forward speed v:110 km h−1
• The blade angular velocity Ω:300 rad s−1
• The rotor radius R:0.4 m
Based on this information, compute the Blade power of UA V in forward flight?
o1 Response with RAG
Below is a concise way to obtain the “blade power” term for forward flight from the documents. From (66) in
the provided text, the blade-profile (or “blade”) power in forward flight is the first bracketed term, namely
Pblade(V) =P0
1 +3V2
Ω2R2
.
Given Data
•P0= 1.5(constant), Forward speed V= 110 km h−1
• Blade angular velocity Ω = 300 rad s−1, Rotor radius R= 0.4 m
Calculation Steps Step 1: Convert V into meters per second
V(m s−1) = 110 ×1000
3600≈30.56 m s−1.
Step 2: Compute the ratio V2/(Ω2R2)
•Ω2R2= (300 rad s−1)2×(0.4 m)2= 90 000 ×0.16 = 14 400 (m s−1)2.
•V2≈(30.56 m s−1)2≈933.9136 (m /s)2.
• Hence,V2
Ω2R2≈933.9136
14400≈0.064855 .
(Using the input’s rounded value for V2/(Ω2R2)≈0.0649 for the next step as per the provided text.)
Step 3: Substitute into the blade-power expression: Using the approximation V2/(Ω2R2)≈0.0649 :
1 + 3×0.0649≈1 + 0 .1947 = 1 .1947.
Pblade(V) = 1 .5×1.1947≈1.79205 .
(Rounding to 1.79as per the provided text.)
Result: Thus, the blade power at V= 110 km h−1is approximately 1.79(in the same power units as P0).
A.4 Question 1-1: Deep Seek Answer with RAG
Prompt
Consider this information:
1. Ground user’s position: (47,23)m
2. UA V-BS’s position: (65,73,15)m
Based on this information, what speed is elevation angle?
14

RAG-UA V (Azarafza et al.) A P REPRINT
DeepSeek R1 Response
Okay, so I need to figure out the elevation angle based on the given positions of the ground user and the
UA V-BS... Okay, so I need to figure out the elevation angle based on the given positions of the ground user and
the UA V-BS. Let me start by understanding what an elevation angle is. From what I remember, the elevation
angle in the context of wireless communications or between two points is the angle between the line connecting
the two points (in this case, the ground user and the UA V) and the horizontal plane. So it’s like looking up from
the ground to the UA V , and the angle your line of sight makes with the horizontal.
First, I need to find the distance between these two points in 3D space because the elevation angle depends on
their positions. The ground user is at (47,23) meters, which I assume is in a 2D plane, so their z-coordinate is 0.
The UA V-BS is at (65,73,15)m, so its x and y coordinates are 65 and 73 respectively, and it’s 15 meters above
the ground.
To find the elevation angle, I think I can model this as a right triangle where the vertical distance from the
ground user to the UA V is one leg (which is 15 meters), and the horizontal distance between them is the other
leg. The hypotenuse would then be the straight-line distance between the two points in 3D space.
So, first step: calculate the horizontal distance between the two points. Since both are given in x and y
coordinates, I can use the Euclidean distance formula for 2D:
Horizontal distance =p
(65−47)2+ (73−23)2 (1)
Let me compute that:
65−47 = 18
73−23 = 50
So, horizontal distance =
Horizontal distance =p
182+ 502=√
324 + 2500 =√
2824 (2)
Calculating√
2824 . Let’s see: 532is 2809, so√
2824 is a bit more than 53. Maybe approximately 53.11 meters.
Now, the vertical distance is given as 15 meters.
So now I have a right triangle with opposite side (vertical) = 15m and adjacent side (horizontal) ≈53.11m.
The elevation angle θcan be found using the tangent function:
tan(θ) =opposite
adjacent=15
53.11(3)
Let me compute that: 15 divided by approximately 53.11 is roughly 0.2823.
Now, to find θ, I take the arctangent of 0.2823. Using a calculator, arctan(0 .2823) is about 15.7 degrees.
Wait, let me double-check my calculations because sometimes it’s easy to mix up the sides. The elevation angle
is indeed calculated using tan(θ) =opposite /adjacent , which in this case is vertical over horizontal distance.
So yes, that should be correct.
Alternatively, I could use the sine function since sin(θ) =opposite
hypotenuse. Let me compute the hypotenuse first:
Hypotenuse =p
horizontal2+vertical2= (4)
p
53.112+ 152 (5)
Calculating 53.11squared: approx 2824, and 15squared is 225, so total under the root is 2824 + 225 = 3049 .
Hypotenuse ≈√
3049≈55.22meters (6)
Then
sin(θ) =15
55.22≈0.2717 (7)
Taking arcsin of that gives θ≈15.75degrees, which aligns with my previous calculation using tangent. So
both methods give me around 15.7 to 15.8 degrees, so I can be confident it’s approximately 15.7 degrees.
I think that’s the elevation angle between the ground user and the UA V-BS based on their positions.
The elevation angle between the ground user at (47,23)m and the UA V-BS at (65,73,15)m is calculated as
follows:
1. Horizontal Distance Calculation:
•∆x= 65−47 = 18 m
•∆y= 73−23 = 50 m
• Horizontal distance =√
182+ 502≈53.11m
2. Vertical Distance:
• Vertical distance = 15 m (z-coordinate of UA V-BS)
3. Elevation Angle Calculation using Tangent:
tan(θ) =opposite
adjacent=15
53.11≈0.2823 (8)
θ≈arctan(0 .2823) ≈15.7◦(9)
4. Verification using Sine:
Hypotenuse =p
53.112+ 152≈55.22m (10)
sin(θ) =15
55.22≈0.2717 (11)
θ≈arcsin(0 .2717) ≈15.75◦(12)15