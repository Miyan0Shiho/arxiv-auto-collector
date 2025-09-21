# Causal-Counterfactual RAG: The Integration of Causal-Counterfactual Reasoning into RAG

**Authors**: Harshad Khadilkar, Abhay Gupta

**Published**: 2025-09-17 21:18:47

**PDF URL**: [http://arxiv.org/pdf/2509.14435v1](http://arxiv.org/pdf/2509.14435v1)

## Abstract
Large language models (LLMs) have transformed natural language processing
(NLP), enabling diverse applications by integrating large-scale pre-trained
knowledge. However, their static knowledge limits dynamic reasoning over
external information, especially in knowledge-intensive domains.
Retrieval-Augmented Generation (RAG) addresses this challenge by combining
retrieval mechanisms with generative modeling to improve contextual
understanding. Traditional RAG systems suffer from disrupted contextual
integrity due to text chunking and over-reliance on semantic similarity for
retrieval, often resulting in shallow and less accurate responses. We propose
Causal-Counterfactual RAG, a novel framework that integrates explicit causal
graphs representing cause-effect relationships into the retrieval process and
incorporates counterfactual reasoning grounded on the causal structure. Unlike
conventional methods, our framework evaluates not only direct causal evidence
but also the counterfactuality of associated causes, combining results from
both to generate more robust, accurate, and interpretable answers. By
leveraging causal pathways and associated hypothetical scenarios,
Causal-Counterfactual RAG preserves contextual coherence, reduces
hallucination, and enhances reasoning fidelity.

## Full Text


<!-- PDF content starts -->

Causal-Counterfactual RAG: The Integration of Causal-Counterfactual
Reasoning into RAG
1Harshad Khadilkar,2Abhay Gupta
1Indian Institute of Technology Bombay
2Indian Institute of Technology Patna
1harshadkhadilkar,2abhaygupta
Abstract
Large language models (LLMs) have trans-
formed natural language processing (NLP), en-
abling diverse applications by integrating large-
scale pre-trained knowledge. However, their
static knowledge limits dynamic reasoning over
external information, especially in knowledge-
intensive domains. Retrieval-Augmented Gen-
eration (RAG) addresses this challenge by com-
bining retrieval mechanisms with generative
modeling to improve contextual understand-
ing. Traditional RAG systems suffer from dis-
rupted contextual integrity due to text chunk-
ing and over-reliance on semantic similarity
for retrieval, often resulting in shallow and
less accurate responses. We propose Causal-
Counterfactual RAG , a novel framework that
integrates explicit causal graphs representing
cause-effect relationships into the retrieval pro-
cess and incorporates counterfactual reasoning
grounded on the causal structure. Unlike con-
ventional methods, our framework evaluates
not only direct causal evidence but also the
counterfactuality of associated causes, combin-
ing results from both to generate more robust,
accurate, and interpretable answers. By lever-
aging causal pathways and associated hypo-
thetical scenarios, Causal-Counterfactual RAG
preserves contextual coherence, reduces hallu-
cination, and enhances reasoning fidelity.
1 Introduction
Large language models (LLMs) have revolution-
ized the field of natural language processing (NLP),
enabling numerous applications across various do-
mains. However, their dependence on static, pre-
trained knowledge constrains their capacity to in-
corporate and reason over dynamically updated
external information, especially in knowledge-
intensive areas. Retrieval-Augmented Generation
(RAG) addresses this by integrating external knowl-
edge retrieval with generative modeling, improv-
ing contextual understanding and enhancing outputquality (Lewis et al., 2020). Recent research ad-
vances in RAG have focused on two main areas:
first, boosting retrieval efficiency through adaptive
and modular designs; and second, better organizing
external knowledge, with graph-based RAG meth-
ods emerging as a leading approach (Gao et al.,
2023).
Despite these improvements, current RAG sys-
tems still face significant challenges that hinder re-
trieval fidelity and answer accuracy (Barnett et al.,
2024). Such challenges include disrupted contex-
tual coherence due to fragmented text chunking,
an over-reliance on semantic similarity instead of
causal relevance in retrieval, and insufficient preci-
sion in selecting truly pertinent documents. More-
over, the lack of counterfactual information within
retrieval databases limits robust reasoning and re-
duces confidence in generated answers (Pearl and
Mackenzie, 2018). Our theoretical and empirical
analyses reveal that many existing RAG systems
frequently fail to retrieve causally grounded content
appropriately aligned with user queries, often re-
sulting in answers that appear relevant superficially
but lack deep grounding.
To address these limitations, we propose Causal-
Counterfactual RAG , a novel framework that inte-
grates explicit causal graphs encoding cause-effect
relationships directly into the retrieval mechanism,
further enriched by counterfactual reasoning based
on these causal structures (Pearl, 2009). Unlike
previous models, our approach evaluates not only
direct causal evidence but also its counterfactual
alternatives, allowing the system to consider hy-
pothetical scenarios alongside actual causes. This
dual reasoning process enables the generation of
responses that are more causally coherent, robust,
and interpretable.
We validate our framework through extensive
experimentation on diverse datasets with varying
context lengths and complexities. Benchmarking
against standard RAG approaches, our model con-
1arXiv:2509.14435v1  [cs.CL]  17 Sep 2025

sistently outperforms in retrieval and generation
metrics (Chen et al., 2023).
The main contributions of this work are:
•Identification and analysis of critical short-
comings in existing RAG systems concern-
ing causal grounding, content relevance, and
counterfactual reasoning.
•Introduction of the Causal-Counterfactual
RAG framework that effectively combines
causal graph-guided retrieval and counterfac-
tual inference to improve answer quality and
reliability.
•Demonstration of our framework’s ability to
mitigate hallucinations and enhance answer in-
terpretability, offering valuable insights for de-
signing robust retrieval-augmented language
systems.
2 Related Work
2.1 Retrieval Augmented Generation
Retrieval-Augmented Generation (RAG) enhances
large language models (LLMs) by integrating ex-
ternal knowledge retrieval to tackle knowledge-
intensive tasks. Early RAG frameworks primar-
ily focused on semantic similarity-based retrieval
methods, combining retrieved documents with gen-
erative models to improve answer relevance and
contextual understanding. Subsequent research has
introduced modular retrieval architectures and it-
erative retrieval-generation processes to optimize
efficiency and adaptability (Asai et al., 2023).
Recent advancements in RAG have explored
structuring external knowledge as graphs to sup-
port multi-hop and hierarchical retrieval strategies
(Feng et al., 2023). These models utilize graph-
based representations, such as dynamically updated
knowledge graphs and node/edge relations to im-
prove retrieval quality, answer precision, and con-
textual reasoning (Pan et al., 2023). Multi-stage
retrieval frameworks like PolyRAG introduce hier-
archical retrieval via knowledge pyramids, while
others incorporate graph-based ranking algorithms
and temporal graph reasoning to handle complex
queries.
2.2 Causality and Counterfactuality
Recent progress in Retrieval-Augmented Gener-
ation (RAG) systems has focused on embedding
causal reasoning to enhance the precision of re-
trieved information and improve the coherence ofgenerated responses (Kiciman et al., 2023). Causal
discovery techniques increasingly harness large
language models (LLMs) to efficiently construct
causal graphs, employing strategies like LLM-
assisted breadth-first search for mapping causal
structures and approaches such as Corr2Cause for
inferring causality from correlations (Jin et al.,
2023). Despite these promising developments, the
majority of existing methods concentrate on build-
ing causal graphs or estimating causal effects but
fall short of integrating causal reasoning directly
into the combined retrieval and generation process
of RAG systems.
Importantly, counterfactual reasoning-the eval-
uation of hypothetical “what-if” scenarios to rig-
orously test causal claims remains markedly un-
derexplored within the RAG domain. Although
well-established in causal inference and machine
learning, counterfactual reasoning is rarely embed-
ded as a fundamental mechanism in RAG frame-
works. While some studies incorporate partial
causal signals into transformer architectures or
leverage causal cues for pre-retrieval filtering, com-
prehensive counterfactual inference that dynami-
cally reconciles retrieved knowledge with the user’s
query intent is seldom realized.
This lack of explicit counterfactual validation
renders RAG models susceptible to generating re-
sponses that, despite appearing plausible, are vul-
nerable to alternative causal explanations or hypo-
thetical interventions. Addressing this deficiency
is crucial to developing the next generation of
retrieval-augmented language models capable of
producing responses that are not only contextually
relevant but also causally consistent and resilient
under counterfactual scrutiny.
3 Limitations of Regular RAG
Traditional Retrieval-Augmented Generation
(RAG) systems primarily rely on semantic
similarity to fetch relevant documents by matching
the meanings of words or phrases (Figure 1).
However, this method often results in retrieving
text chunks that, while similar in wording, may be
contextually irrelevant or only tangentially related,
leading to lower precision and recall (Ma et al.,
2023). A significant limitation of regular RAG
systems is the absence of causal inferencing and
counterfactual reasoning (Zhao et al., 2024). They
lack mechanisms to verify whether the retrieved
information truly holds under alternative “what-if”
2

Source Documents
 Chunking & Embedding
Embed Query
 Semantic Search
 LLM Generation
Vector StoreLimitation 1:
Disrupted Coherence
(Chunking can break content)User QueryLimitation 2:
Retrieval Failure
(Semantic search can be biased or 
retrieve irrelevant context)
Final Answer
Limitation 3:
Generating Issues
(Context ignorance, bias amplification, 
hallucination)Limitation 4:
Lack of Counterfactual Reasoning
(The answer is not tested against
hypothetical scenarios)Figure 1: The standard RAG pipeline, highlighting four failure points: broken context from chunking, biased
semantic search, unreliable LLM generation, and a lack of trustworthiness checks.
scenarios, which is essential for building trust
and robustness in reasoning. For instance, if a
query asks about the impact of a new regulation
on a company’s financial report, a conventional
RAG might retrieve documents discussing related
topics broadly but fail to validate whether these
cause-effect relationships are applicable in the
specific context, resulting in answers that include
hallucinations or misleading causal assertions (Liu
et al., 2023). This shortfall underscores the need to
incorporate causal and counterfactual reasoning
into RAG to improve the accuracy and reliability
of retrieved knowledge and generated responses.
4 Methodology
Our framework (Figure 2), the Causal-
Counterfactual RAG , operates on a Causal
Knowledge Graph (CKG) that is constructed prior
to receiving any queries. This graph is populated
by a powerful LLM that analyzes a source
corpus to extract cause-and-effect relationships,
representing key events as nodes with vector
embeddings and their causal links as directed
edges.
Our methodology unfolds in a three-step process.
First, a user’s causal query is parsed into its logi-
cal components (evidence, intervention, outcome).
Second, a precise two-stage retrieval process maps
these components to the CKG, using a combination
of fast vector search and LLM-based verification
to ensure semantic and polarity alignment.
The core of our approach is the final step: a coun-terfactual validation loop. For each potential cause
retrieved from the graph, the system programmat-
ically generates its logical opposite (e.g., its ab-
sence) and simulates the downstream effects of this
hypothetical intervention. A synthesis LLM then
analyzes both the factual evidence and the results
of these simulations to confirm which causes were
truly necessary for the outcome. This ensures the
final answer is a robust, validated explanation that
distinguishes true causation from simple correla-
tion.
4.1 Document Indexing
Document indexing transforms unstructured text
into a structured Causal Knowledge Graph (CKG)
(Hogan et al., 2021). A large language model
(Google’s Gemini 1.5) first analyzes document
chunks to extract explicit (cause, effect) pairs (Li
et al., 2022). These pairs are then intelligently
ingested into the graph using a crucial node consol-
idation strategy to prevent redundancy with their
embeddings (384-dimensional vectors from Sen-
tenceTransformer all-MiniLM-L6-v2 ) and source
document (as metadata) (Su et al., 2022). Before
creating a new event node, a two-stage verification
process is employed: a rapid semantic search iden-
tifies conceptually similar candidate nodes, after
which a qualitative reasoning Gemini LLM model
performs a strict check to confirm they are truly in-
terchangeable and share the same polarity. Causal
relationships are then formed with either the exist-
ing or a newly created node, and each relationship
stores a reference back to the original source text,
3

ensuring the final CKG is both semantically coher-
ent and fully traceable.
4.2 Causal Query Parsing
At query time, the pipeline begins by deconstruct-
ing the user’s natural language query into its fun-
damental causal components. This is done using a
Gemini 1.5 LLM, prompted for Natural Language
Understanding (NLU) into a structured schema
based on Structural Causal Models (SCMs). This
extracts the central components like evidences, hy-
pothetical interventions, query variable, and main
event. This initial step transforms an ambiguous
query into a formal, machine-readable problem,
ensuring the system correctly understands which
events are facts to be conditioned on and which are
causal links to be investigated.
4.3 Two-Stage Context Retrieval
Once the key events are extracted, the system re-
trieves the relevant factual context from our Causal
Knowledge Graph (CKG). To ensure high preci-
sion, we utilize a two-stage retrieval process.
•Vector Similarity Search : Each identified
event is encoded into a 384-dimensional
embedding vector by SentenceTransformer
all-MiniLM-L6-v2 . A fast approximate near-
est neighbor search is performed on the
graph’s Neo4j vector index to retrieve a set of
top-k candidate event nodes that are semanti-
cally similar.
•LLM-based Verification : The top-k candi-
dates from the vector search are then sub-
jected to rigorous semantic and polarity ver-
ification using the Gemini LLM. The model
is prompted to determine if a candidate node
describes the same core event and, critically,
has the same polarity (e.g., "increase" vs. "de-
crease") as the event from the query.
Only the nodes that pass this verification are used
as entry points to traverse the CKG and retrieve
the causes and established cause-effect relationship
rules associated with the observed outcome. This
two-stage process effectively filters out nodes that
are topically related but contextually incorrect.
4.4 Counterfactual Intervention and
Simulation
This step forms the core of our validation engine.
For each potential cause identified in the factual re-
trieval stage, the system performs a counterfactualtest to determine its necessity. This involves two
sub-processes:
•Counterfactual Cause Generation : An
LLM (Gemini 1.5) is prompted with detailed
instructions and retry logic to generate the
single best logical opposite for the cause in
question, ensuring plausible alternative state
generation beyond simple negations.
•Simulated Graph Traversal : The newly gen-
erated counterfactual cause+, is encoded (384-
dimensional embedding) and used to initiate
a new query. The system finds semantically
similar and contextually equivalent nodes in
the graph (as in section 4.3) and traces their
downstream effects to simulate the alternate
reality where the original cause was absent.
4.5 Synthesized Causal and Counterfactual
Reasoning
In the final stage, all collected information-the fac-
tual causal chains and the outcomes of the counter-
factual simulations-is compiled into comprehensive
evidence packages. These packages are passed to a
final Gemini 1.5 LLM with specialized prompting
that instructs it to synthesize and prioritize coun-
terfactual simulation results when making causal
necessity judgments.
The model reasons step-by-step and generates
a robust, grounded explanation. If the absence of
a potential cause leads to the absence of the final
effect in simulation, it concludes that cause was
necessary. Thus, the final user answer distinguishes
validated causes from mere correlations.
5 Experiment
The Causal-Counterfactual RAG is evaluated
through a comparative study against Regular RAG
to assess its performance across various metrics,
providing a comprehensive understanding of its
strengths and effectiveness relative to these estab-
lished models.
5.1 Experimental Setup
Baseline : We evaluate two RAG variants: Regu-
lar RAG, and our proposed Causal-Counterfactual
RAG . Regular RAG serves as a standard baseline,
relying primarily on semantic similarity for re-
trieval. Our method extends these approaches by in-
tegrating counterfactual reasoning within the causal
framework. In our comparison pipeline, we system-
atically evaluate Regular RAG as baselines against
4

Upload Document
Ask Query
Final Response
Cause-Effect Pair
Extraction
Query Parsing
Synthesized Causal and
Counterfactual
Reasoning + User Query
Causal Knowledge
Graph
For each cause,
generate the
hypothetical
counterfactual
scenario  
Event 2
Event 1
Event 3
Embedding Search
Relevant Nodes
Relevant
Counterfactual
Nodes
Retrieve Whole
Component
Extract All Possible
Causes
Retrieving whole
component
Factual Causal Rules (Cause-Effect Relationships)
Counterfactual
Causal Rules
LLM
LLMFigure 2: Architecture of Causal-Counterfactual RAG : For each query, cause–effect pairs are extracted, relevant
causal and counterfactual graph components are retrieved, and LLM reasoning synthesizes robust answers by
validating causes through both factual and counterfactual evidence.
theCausal-Counterfactual RAG to highlight the
improvements achieved through combining causal
and counterfactual insights (Es et al., 2023).
Datasets : Most QA benchmark datasets focus on
fact-based retrieval and classic NLP tasks, which
do not adequately assess discourse-level under-
standing or causal reasoning (Yang et al., 2018).
To better evaluate retrieval-augmented generation
systems on counterfactual and causally rich queries,
we use the some custom dataset and OpenAlex cor-
pus of academic papers, enabling rich grounding of
queries (Priem et al., 2022). Large language mod-
els generate multiple grounded questions per docu-
ment, ensuring explicit answerability and effective
evaluation of discourse-level and counterfactual
reasoning.
This evaluation setup specifically tests the ability of
our RAG pipeline to handle robust causal queries,
going beyond surface-level fact retrieval to inter-
pret underlying causal dynamics and alternatives.
Metrics & Implementation details : We adopted
multiple rigorous evaluation metrics to quantita-
tively compare our Causal-Counterfactual RAG
against baseline regular RAG system. Following
metrics evaluate the quality of retrieval/generation,
causal reasoning and counterfactual reasoning
robustness (Yu et al., 2023).
A. Precision & Recall : Precision measures how
many of the retrieved documents or answers are
actually relevant. High precision means fewer ir-relevant or incorrect documents are included in the
results. Recall measures how many of the total rele-
vant documents or answers in the entire dataset are
successfully retrieved by the system. High recall
means the system misses fewer relevant pieces of
information.
Precision =|Rq|
|Sq|(1)
Recall =|Rq|
|R|(2)
where,
Rq=no. of relevant retrieved documents,
Sq=total number of documents retrieved.
R=total no. of relevant documents in the corpus.
B. Causal Chain Integrity Score : This met-
ric gauges the robustness and dependability of the
causal chain produced by the pipeline. It quantifies
how well the identified chain of causes supports the
observed outcome by integrating two complemen-
tary assessment modalities: semantic similarity and
advanced LLM-driven evaluation.
The semantic similarity component is computed
by encoding both the generated answer and the
ideal (ground truth) answer into dense vector
representations using the SentenceTransformer
all-MiniLM-L6-v2 model, and then calculating
the cosine similarity between these embeddings.
This measures the lexical and semantic overlap be-
tween the answers.
5

Precision Recall CIS CRS
Evaluation Metrics020406080100Performance Score (%)60.1374.58
53.62
49.1280.57
78.18
75.58
69.90RAG Model Performance Comparison Across Metrics
Regular RAG Causa-Counterfactual RAG
CIS: Causal Chain Integrity Score  |  CRS: Counterfactual Robustness ScoreFigure 3: Performance comparison between a Regular RAG and our proposed Causal-Counterfactual RAG . The
evaluation is conducted across four metrics: traditional Precision and Recall, and two causal reasoning metrics, the
Causal Chain Integrity Score (CIS) and the Counterfactual Robustness Score (CRS).
Concurrently, an advanced large language model
judge is employed to provide a holistic evaluation
of the generated answer’s correctness, faithfulness,
and coherence within the context of the original
query and ground truth. In our updated pipeline,
we use the LLaMA-3.1-8B-Instant model, a highly
capable and efficient foundational model deployed
via Groq’s platform as the judge LLM. This model
receives a prompt containing the question, ideal
answer, generated answer, and relevant contextual
information, and evaluates according to a detailed
rubric focusing on two criteria:
1.Correctness: How accurately does the gen-
erated answer address the question based on
the ideal answer and context? Scored from 1
(completely wrong) to 5 (perfectly correct).
2.Faithfulness & Reasoning: How well is the
generated answer supported by the context?
Does it demonstrate clear, accurate reasoning
without hallucinated facts? Also scored 1 (not
supported or wrong reasoning) to 5 (fully sup-
ported and clear reasoning).
The LLM judge produces an output with these two
integer scores, which are then averaged and nor-malized to a 0.0 to 1.0 scale to form the LLM judge
score component ( LJ) of the metric.
To combine these complementary assessments,
we apply a weighted sum, producing the Causal
Chain Integrity Score (CCIS), which quantifies
both surface-level similarity and deeper causal rea-
soning quality:
CCIS =w1×Sim +w2×LJ (3)
LJ=C_Score +FR_Score
10(4)
where,
CCIS :Causal Chain Integrity Score,
w1,w2:weighting coefficients balancing the
contributions of each score,
Sim :cosine similarity between embeddings,
LJ:normalized LLM judge score,
C_Score :Correctness Score,
FR_Score :Faithfulness & Reasoning Score
This hybrid evaluation enables the pipeline to
robustly assess causal chain quality, capturing both
textual closeness and nuanced reasoning fidelity,
yielding a reliable measure of causal explanation.
6

C. Counterfactual Robustness Score : This
metric evaluates the pipeline’s capability to per-
form comprehensive causal analysis through the
generation and examination of hypothetical coun-
terfactual scenarios. It measures how well the sys-
tem reasons about “what-if” alternatives by sys-
tematically altering causes and observing resultant
outcomes within these scenarios. The evaluation
employs a hybrid approach analogous to the one de-
fined for Causal Chain Integrity (Equation 3), com-
bining semantic similarity between the generated
and ideal answers with an LLM-based judgment of
answer validity relative to the query and context.
This composite score thus captures the depth and
reliability of counterfactual reasoning embedded in
the model’s responses.
5.2 Performance Comparison
Figure 3 compares Causal-Counterfactual RAG
with a baseline Regular RAG across four met-
rics: precision, recall, causal chain integrity,
and counterfactual robustness. The results show
clear gains from combining causal reasoning
with counterfactual validation, positioning Causal-
Counterfactual RAG as a more reliable framework
for knowledge-intensive tasks.
Causal grounding with counterfactual vali-
dation ensures robust reasoning .Causal-
Counterfactual RAG delivers both high precision
(80.57) and recall (78.18), while achieving a
strong causal chain integrity score (75.58). Un-
like standard approaches, it tests counterfactual
variations of causes to distinguish genuine causal
links from correlations, reflected in its robustness
score (69.90). This dual design reduces superficial
matches and promotes logically consistent answers.
Regular retrieval lacks causal and counterfac-
tual alignment. Regular RAG reaches decent re-
call (74.58) but suffers from low precision (60.13)
and weak causal integrity (53.62). Without coun-
terfactual checks, it fails to filter spurious corre-
lations, which is evident in its lower robustness
score (49.12). This leads to factually broad but less
reliable reasoning.
Towards a causal -counterfactual paradigm. The
comparison shows that while Regular RAG re-
trieves broadly, it lacks causal depth and robustness.
By explicitly modeling causal chains and validating
them with counterfactuals, Causal-Counterfactual
RAG combines wide coverage with strong logicalalignment, setting a new benchmark for trustworthy
retrieval-augmented generation.
5.3 Conclusion and Future Work
This research introduced the Causal-
Counterfactual RAG , a novel framework
designed to address a critical limitation in existing
retrieval-augmented systems: their inability to
distinguish necessary causes from mere correla-
tions. By operating on a Causal Knowledge Graph,
our pipeline moves beyond simple information
retrieval. Its core contribution is a counterfactual
validation loop that programmatically tests the
necessity of each potential cause by simulating its
absence. This methodology enables the system to
construct answers that are not just plausible but
are robustly verified, significantly enhancing the
faithfulness, depth, and trustworthiness of causal
explanations. Our work represents a crucial step
from passive knowledge retrieval toward active
and verifiable machine reasoning.
While our framework demonstrates strong perfor-
mance on complex causal queries, it is a special-
ized engine. Our primary goal for future work is
to develop a hybrid, multi-pipeline framework that
provides comprehensive reasoning for all types of
user queries. The central component of this future
system will be an intelligent query routing mecha-
nism. This initial step will classify a user’s query
to determine the most appropriate reasoning path:
•Factual Queries : Will be routed to a fast and
efficient standard RAG pipeline.
•Relational Queries : Will be handled by a
standard knowledge graph RAG optimized for
non-causal entity relationships.
•Causal & Counterfactual Queries : Will be
directed to our current, powerful counterfac-
tual validation pipeline.
By integrating these specialized pipelines, we aim
to create a single, unified system that dynamically
selects the optimal engine for any given task. This
will result in a fully robust and adaptive framework
that offers the perfect balance of speed and reason-
ing depth, capable of handling the full spectrum of
user intent from simple fact-finding to deep causal
analysis.
7

Limitations
The pipeline’s primary architectural limitations are
rooted in the LLM-driven process of constructing
the causal knowledge graph, which is susceptible
to error propagation. The reliability of the entire
system hinges on the LLM’s ability to accurately
extract cause-effect pairs from unstructured text.
This extraction process is vulnerable to several fail-
ure modes: the LLM may misinterpret mere cor-
relation as causation, fail to capture the nuanced
difference between direct and contributing factors,
or fabricate relationships not explicitly stated in the
source text. Any such inaccuracies are enshrined
as ground truth within the causal graph, fundamen-
tally undermining the trustworthiness of the subse-
quent counterfactual reasoning, regardless of how
precise that reasoning process is. Concurrently,
the counterfactual reasoning stage, while power-
ful, introduces a layer of computational complexity
that increases query latency compared to a standard
single-pass RAG, posing a potential constraint for
real-time applications.References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
arXiv preprint arXiv:2310.11511 .
Scott Barnett, Pranshu Nema, Vageesh Mishra, and
Željko Agi ´c. 2024. Seven failure points when en-
gineering a retrieval augmented generation system.
arXiv preprint arXiv:2401.05856 .
Jiacheng Chen, Alon Lin, Xinyue Chen, Qian Zha, an-
shuman anantha, Xiaozhi Lee, anshuman Liu, jin-
jie Li, Hubo Wang, anshuman anantha, et al. 2023.
Benchmarking large language models in retrieval-
augmented generation. In Proceedings of the 2nd
Workshop on Retrieval-Enhanced Machine Learning ,
pages 25–39.
Shir Es, João G Ramos, Javier de la Fuente, and Raúl
de la Fuente. 2023. Rag vs fine-tuning: Pipelines,
tradeoffs, and a case study on agriculture. arXiv
preprint arXiv:2310.01996 .
Wenhao Feng, Yuhan He, Hong-Han Chen, and Yong
Zhang. 2023. Knowledge graphs for rag: A survey.
arXiv preprint arXiv:2310.19830 .
Yunfan Gao, Yuncheng Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Han. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997 .
Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia
d’Amato, Gerard de Melo, Claudio Gutierrez, José
Emilio L Gayo, Sabrina Kirrane, Sebastian Neumaier,
Axel Polleres, et al. 2021. Knowledge graphs. ACM
Computing Surveys (CSUR) , 54(4):1–37.
Zhijing Jin, Yuen An, Bang An Chen, Zhaoning Liu,
Michael I Jordan, and Bernhard Schölkopf. 2023.
Can large language models infer causality from cor-
relation? arXiv preprint arXiv:2306.05836 .
Emre Kiciman, Robert Ness, Amit Sharma, and Chen-
hao Wang. 2023. Causal reasoning and large lan-
guage models: A survey. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 13401–13423.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Timo Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive NLP tasks. In Advances in
Neural Information Processing Systems , volume 33,
pages 9459–9474.
Zhaoning Li, He Huang, and Heng Ji. 2022. Causal
event extraction from text. In Proceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies , pages 5687–5697.
8

Jia-Yu Liu, Mariann Zahera, and Zhe Wang. 2023.
Evaluating and improving the factuality of retrieval-
augmented language models. arXiv preprint
arXiv:2311.08316 .
Xinbei Ma, Yeyang Gong, Pengcheng Li, Yang Lu,
and Nan Duan. 2023. Query rewriting for retrieval-
augmented large language models. arXiv preprint
arXiv:2305.14283 .
Shirui Pan, Linhao Chen, Yuxiang Wang, Chen Zhang,
Xiao Wang, Jiayin Li, Yufan Chen, Hao Chen, and Ri-
chong Wang. 2023. Unifying large language models
and knowledge graphs: A roadmap. arXiv preprint
arXiv:2306.08302 .
Judea Pearl. 2009. Causality: Models, reasoning, and
inference.
Judea Pearl and Dana Mackenzie. 2018. The book of
why: the new science of cause and effect.
Jason Priem, Heather Piwowar, and Richard Orr. 2022.
Openalex: A fully open bibliographic database.
arXiv preprint arXiv:2205.01833 .
Yuhui Su, Chen Chen, Weijia Chen, and Ke Zhao. 2022.
Rethinking the role of language models in knowledge
graph completion. In Proceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing , pages 6164–6177.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600 .
Wenhao Yu, Hongming Bai, Yilin Zhang, Zhi-Hong Li,
Jialu Zhu, and Meng Zhao. 2023. Chain-of-note: En-
hancing robustness in retrieval-augmented language
models. arXiv preprint arXiv:2311.09210 .
Zhaoning Zhao, Yiming Yang, Shujian Shi, and Qian
Wang. 2024. CausalRAG: Causal-based retrieval-
augmented generation for problem-solving. In Pro-
ceedings of the 2024 Conference on Empirical Meth-
ods in Natural Language Processing .
9