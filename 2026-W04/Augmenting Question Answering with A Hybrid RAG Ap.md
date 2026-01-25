# Augmenting Question Answering with A Hybrid RAG Approach

**Authors**: Tianyi Yang, Nashrah Haque, Vaishnave Jonnalagadda, Yuya Jeremy Ong, Zhehui Chen, Yanzhao Wu, Lei Yu, Divyesh Jadav, Wenqi Wei

**Published**: 2026-01-19 02:08:47

**PDF URL**: [https://arxiv.org/pdf/2601.12658v1](https://arxiv.org/pdf/2601.12658v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful technique for enhancing the quality of responses in Question-Answering (QA) tasks. However, existing approaches often struggle with retrieving contextually relevant information, leading to incomplete or suboptimal answers. In this paper, we introduce Structured-Semantic RAG (SSRAG), a hybrid architecture that enhances QA quality by integrating query augmentation, agentic routing, and a structured retrieval mechanism combining vector and graph based techniques with context unification. By refining retrieval processes and improving contextual grounding, our approach improves both answer accuracy and informativeness. We conduct extensive evaluations on three popular QA datasets, TruthfulQA, SQuAD and WikiQA, across five Large Language Models (LLMs), demonstrating that our proposed approach consistently improves response quality over standard RAG implementations.

## Full Text


<!-- PDF content starts -->

Augmenting Question Answering with A
Hybrid RAG Approach
Tianyi Yang∗, Nashrah Haque∗, Vaishnave Jonnalagadda∗, Yuya Jeremy Ong†1
Zhehui Chen‡, Yanzhao Wu§, Lei Yu¶, Divyesh Jadav∥1, Wenqi Wei∗2
∗Fordham University, New York, NY , USA,{ty10, nhaque14, vj1, wwei23}@fordham.edu
†Plastic Lab, New York, NY , USA, yuyajeremyong@gmail.com
‡Google, Mountain View, California, USA, zchen451@gatech.edu
§Florida International University, Miami, FL, USA, yawu@fiu.edu
¶Rensselaer Polytechnic Institute, Troy, NY , USA, yul9@rpi.edu
∥Independent Consultant
Abstract—Retrieval-Augmented Generation (RAG) has
emerged as a powerful technique for enhancing the quality
of responses in Question-Answering (QA) tasks. However, ex-
isting approaches often struggle with retrieving contextually
relevant information, leading to incomplete or suboptimal
answers. In this paper, we introduce Structured-Semantic
RAG (SSRAG), a hybrid architecture that enhances QA
quality by integrating query augmentation, agentic routing,
and a structured retrieval mechanism combining vector and
graph based techniques with context unification. By refining
retrieval processes and improving contextual grounding, our
approach improves both answer accuracy and informative-
ness. We conduct extensive evaluations on three popular
QA datasets, TruthfulQA, SQuAD and WikiQA, across five
Large Language Models (LLMs), demonstrating that our
proposed approach consistently improves response quality
over standard RAG implementations.
Index Terms—Question-answering, RAG, query process-
ing
I. INTRODUCTION
Large Language Models (LLMs), such as OpenAI’s
GPT series [1], Google’s PaLM [2] and Gemini [3],
Meta’s Llama series [4], and Anthropic’s Claude [5] have
significantly contributed towards the advancements in
natural language processing (NLP), achieving remarkable
performance in applications like question answering, text
summarization, machine translation, and sentiment analy-
sis. These advancements have made LLMs indispensable
across industries such as finance, healthcare, legal and
consulting, where automated decision-making and data
retrieval are critical [6]–[8].
Despite their capabilities, LLMs often struggle with
retrieval and response generation quality, particularly in
open-domain question answering. A key limitation is
their reliance on parametric memory, which restricts their
ability to recall and retrieve factual data dynamically. This
issue is further exacerbated by hallucinations [9], [10],
1Work done when the authors were with IBM Research.
2The corresponding author thanks the partial support from Fordham-
IBM Research Fellowship and Fordham Faculty Research Grant.where LLMs generate responses that appear confident but
contain inaccurate or misleading information. Traditional
mitigation strategies have sought to enhance factual
consistency and reasoning reliability through several
complementary methods. For instance, RAG [11] aug-
ments model prompts with external data sources, thereby
grounding responses in verifiable data. Logical reasoning
frameworks like Chain of Thought (CoT) [12], [13]
decompose multi-step tasks into sequential steps, while
Chain of Verification (CoVe) [14] introduces iterative
checks against sub-questions to reduce factual errors.
Additionally, self-learning or self-refinement mechanisms
[15] allow the model to iteratively evaluate and correct its
own responses. Despite these promising techniques, each
brings distinct trade-offs. CoT and CoVe often incur high
computational overhead during inference [14], [16], and
may introduce logical inconsistencies such as self-looping
or convergence toward incoherence [17]. RAG, for its
part, mitigates hallucinations by anchoring responses in
external databases, yet it can falter when retrieval modules
fail to capture nuanced semantic or structural context [18],
[19], particularly when retrieval relies solely on vector-
based methods or simple keyword matching.
To improve the accuracy and efficiency of LLM-driven
question answering, we introduce Structured-Semantic
RAG (SSRAG), a hybrid RAG framework that optimizes
data retrieval and integration. Unlike standard RAG imple-
mentations that rely exclusively on vector-based retrieval,
our approach combines vector and graph based retrieval,
by unifying them into single context representation, to
enhance both semantic precision and structural grounding
in retrieved data. Additionally, our framework introduces
query understanding and augmentation and agentic query
routing modules, which are dynamically refining retrieval
queries and directing them to the most relevant data
sources. Our technical contributions are threefold:
•Query Understanding and Augmentation: We
introduce a lightweight prompting-based dynamic
1arXiv:2601.12658v1  [cs.CL]  19 Jan 2026

Table I: SSRAG vs. other RAG systems.✓= strong native support;◦= partial/indirect; – = absent.
CriterionSSRAGRAPTOR Learned-RAG Agentic RAG
Training-free✓ ✓–✓
Needs labeled/synthetic supervision – –✓–
Real-time freshness (web routing)✓– –/◦ ◦
Explicit factual vs temporal routing✓– –◦
Graph & vector merged in one rerank✓– – –
Global dedup to reduce contradictions✓– – –
Precompute-heavy (summary trees / retraining) –✓ ✓–
query refinement mechanism that expands and clarify
user queries.
•Agentic Query Routing: We propose an rule-
guided intelligent query routing mechanism that
directs augmented queries to the most relevant data
sources from existing fact database or live web
search, optimizing efficiency and mitigating retrieval
failures.
•Hybrid Retrieval and Context Unification: We
develop a hybrid retrieval strategy that integrates
vector and graph-based retrieval techniques to pro-
vide richer and more reliable semantic and structural
grounding. These hybrid contexts are unified into
single context representations that are utilized by an
LLM to generate a factually coherent response.
Through extensive evaluations of the TruthfulQA,
SQuAD, and WikiQA datasets across five advanced
LLMs, our proposed framework, SSRAG demonstrates
substantial improvements in response accuracy and fac-
tual consistency, significantly reducing hallucinations
in question-answering tasks. Our findings highlight the
potential of hybrid retrieval strategies in augmenting
LLM-driven question answering, offering a scalable and
adaptable approach to enhancing the trustworthiness of
AI-generated responses.
II. RELATEDWORK
The mitigation of hallucinations in LLMs has been
extensively studied, leading to various strategies to en-
hance factual accuracy and response reliability. One foun-
dational approach, Chain-of-Thought (CoT) prompting,
improves reasoning by decomposing complex problems
into sequential steps [12]. CoT has demonstrated strong
performance in multi-step reasoning tasks, particularly in
mathematics and logic [13]. However, CoT is susceptible
to error propagation, where incorrect reasoning at earlier
stages can cascade through later steps [16]. This weakness
is magnified in knowledge-intensive tasks, as CoT relies
solely on the model’s parametric memory, which may con-
tain outdated or inaccurate information [11]. Furthermore,
CoT often struggles with abstract reasoning, particularly
when dealing with tasks outside its training data [13].
To address these issues, Chain-of-Verification (CoVe)
introduces an explicit verification phase, where responsesare iteratively checked against generated sub-questions
and independent verification models [14]. CoVe reduces
factual inconsistencies by breaking complex queries into
verifiable units. However, this approach significantly
increases computational overhead by requiring multiple
inference passes. Other mitigation techniques include self-
consistency decoding [20], fact-checking pipelines [21],
and confidence-based filtering [22]. Despite these ad-
vancement, hallucinations persist, particularly in ambigu-
ous or low-resource data domains [9], [10].
A complementary approach, RAG, enhances reliability
by incorporating external data during inference [11].
Unlike CoT and CoVe, RAG accesses external data
sources during inference, allowing retrieval of more up-
to-date and verifiable information [11]. Traditional RAG
relies on static document retrieval and appends retrieved
documents directly to the model’s input context, while
adaptive retrieval mechanisms, such as the Layered Query
Retrieval framework, dynamically refine sources based on
evolving query understanding, improving retrieval accu-
racy and minimizing exposure to irrelevant content [23].
Recent RAG models have also explored cross-document
reasoning, leveraging multiple retrieved sources to verify
consistency before response generation [24]. Despite
merit, existing RAG systems suffer from semantic drift,
where retrieved documents may be loosely relevant but
structurally inconsistent, leading to unreliable output [18].
In summary, prior works attempt to optimize in terms
of (i) query augmentation, which improves recall but
leaves source selection and cross-source consistency
unaddressed, (ii) multi-score RAG, which aggregates
information from multiple sources but often lacks ex-
plicit temporal or factual routing, and (iii) graph-centric
RAG, which emphasizes entity-relationship structures but
heavily depends on the quality of ingested knowledge
graphs or often loses fine-grained details similar to
RAPTOR [25]. Other RAG systems, such as and Self-
RAG require additional model training, which introduces
additional overhead [26], [27]. An independent yet
concurrent work [28] closely related to ours is limited
to where the structured knowledge graphs and domain-
specific retrieval sources are available, and could struggle
to adapt to rapidly evolving data.
In contrast, SSRAG improves traditional RAG models

Figure 1: SSRAG Architecture. The user query Qis first augmented and then routed to the most suitable data source.
The retrieved context is fused through a hybrid pipeline (graph + vector) before generating the final answerR.
by enhancing retrieval accuracy, factual consistency,
and response reliability without the need for model
retraining. It bridges the gaps of prior approaches by (1)
augmenting queries for improved clarity, (2) employing
an agentic router that makes explicit, training-free data
sourcing decisions, and (3) unifying retrieved content
through robust hybrid retrieval and consistency-checking
mechanisms. Moreover, its rule-guided routing and single
reranking step make it easier to audit and reproduce than
fine-tuned agent behaviors, which can drift with changing
data distributions. As summarized inTable I, these
design choices result in more deterministic, interpretable
outputs and establish SSRAG as a flexible and transparent
framework for both knowledge-intensive reasoning and
general-purpose language modeling.
III. METHODOLOGY
This section presents SSRAG, a hybrid RAG frame-
work designed to enhance factual accuracy and reliability
in LLM outputs. Unlike conventional RAG implemen-
tations, which rely solely on vector-based retrieval, our
approach introduces a modular architecture containing
three interdependent components:Query Understanding
and Augmentation,Agentic Query Routing,Hybrid Re-
trieval Mechanism & Content Unification, as depicted in
Figure 1. Specifically,Query Understanding and Aug-
mentationiteratively refines, decomposes, and expands
user queries with contextual cues, improving retrieval pre-
cision.Agentic Query Routingdynamically directs queries
to the most appropriate data sources based on semantic
and structural context.Hybrid Retrieval Mechanism &
Content Unificationintegrates vector and graph-based
retrieval strategies to capture complementary structure and
relevance information, and unifies them into single con-Algorithm 1Query Understanding & Augmentation
1:functionAUGMENTQUERY(query)
2:decompquery←DecomposeQuery(query)
3:entities←ExtractKeyEntities(decompquery)
4:intent←DetectIntent(decompquery, entities)
5:augQuery←EnhanceQuery(query, intent, entities)
returnaugQuery, entities
6:end function
text representation. By optimizing these components in an
end-to-end framework, SSRAG systematically improves
retrieval effectiveness with more contextually relevant,
semantically grounded, and structurally coherent retrieval
that mitigates factual inconsistencies in the generated
responses. In the following sections, we provide a detailed
formulation of each component and its contribution to
retrieval efficiency and generation accuracy.
A. Query Understanding and Augmentation
The first stage of our framework, as outlined in
Algorithm 1, focuses on refining user queries to enhance
retrieval quality through an LLM. The LLM performs
query decomposition,key entity extraction,intent detec-
tion, andquery augmentation, generating a semantically
enriched query representation, denoted as Qaug. The
process begins withquery decomposition, where the LLM
analyzes the syntactic and semantic structures of the
input to identify meaningful components. To extractkey
entities, the LLM is prompted to recognize salient terms
that define the query’s core intent. Next, the LLM infers
user intentby interpreting contextual cues, uncovering
implicit aspects such as subtopics and retrieval goals.
Finally, the LLM reconstructs and enhances the query by
integrating extracted entities and inferred intent, the query

Algorithm 2Agentic Query Routing
1:functionROUTEQUERY(augmentedQuery, entities)
2:ifIsHistoryorFactual(augmentedQuery)then
3:result←QueryWikipediaDB (augmentedQuery)
4:ifresult=emptythen
5:route←GoogleAPI(augmentedQuery)
6:else
7:route←WikipediaAPI(entities)
8:end if
9:else
10:route←GoogleAPI(augmentedQuery)
11:end if
returnaugmentedQuery, route
12:end function
is augmented, ensuring greater specificity and contextual
clarity.
Implementation specifics:
•Entity extraction:We prompt an LLM to return
a list of named entities (people, organizations,
geographic locations, dates, etc.) using a single-turn
instruction.
•Intent cues:We detect query intent
(”who/what/when/where/how”) or generate
one if one is missing from the original query, as
well as if a temporal cue (”today”, ”currently”,
”lately”, etc) is present or missing, as well as factual
cues (proper nouns, historical periods).
•Augmentation:Using an LLM, theEnhanceQuery
step expands and replaces acronyms (for example,
”RL”→”reinforcement learning”. It also canonical-
izes different aliases by mapping through an alias
store, and that if multiple surfaces then map to the
same canonical but keep the longest surface form
(for example, ”Churchill/Sir Winston/Sir Winston
Churchill” would all be replaced by ”Sir Winston
Churchill”). Given the extracted entities and intent
cues, the LLM would take prompt input: ”Given
entities=[...], time hints=[...], intent=[...]. Rewrite a
single augmented query that: (1) Expands acronyms
on first mention in parentheses, (2) Preserves named
entities verbatim, (3) Uses≤40 tokens.”
Illustrative example.Given the query:
“Albert Einstein’s birth place?”,
the system generates an enhanced augmented query by
processing the user query as
“Where was Albert Einstein born?”
B. Agentic Query Routing
Once the user query is augmented, the next crucial
step is to determine the most appropriate data source for
retrieval. This is accomplished through ouragentic query
routing(Algorithm 2) mechanism, which employs a LLM
as an intelligent agent to dynamically classify queries androute them to the most relevant external retrieval system.
This dynamic classification ensures context-aware routing,
reducing the likelihood of retrieving irrelevant or outdated
information. Given an augmented query Qaug, the routing
process considers either of the two key factors:
•Historical/Factual Queries: Queries containing
well-defined entities (e.g., scientific terms, historical
figures) are routed to Wikipedia. The system either
fetches relevant content from the database or, if data
is not available, extracts using the Wikipedia API.
•Temporal sensitivity: Time-sensitive queries (e.g.,
current events, market trends) are directed to real-
time sources like Google. Google API is utilized to
retrieve the latest indexed web pages relevant to the
query.
Implementation specifics, givenQ aug:
•If: temporal cues are present (for example, ”latest”,
”in 2023”, ”today”, ”this week”, ”trending”), route
to Google Web.
•Else: there doesn’t exist a temporal cue, then more
likelier than not factual cues dominate (for example,
named historical figures, scientific concepts, defini-
tions), route to Wikipedia DB corpus.
•All routing is prompt-guided and deterministic given
Qaug. The LLM would take inputQ augand is
asked ”Decide if this user query is time-sensitive
(temporal) or not (factual). Respond with exactly
one token: TEMPORAL or FACTUAL.” The router
output would either point the vector, graph, and
hybrid retriever to either web snippets returned by
Google API or pre-curated Wikipedia corpus.
Illustrative examples.
•“What are the latest breakthroughs in AI?” →
Google, as user query is related to the current events.
•“Where did Albert Einstein was born?”
→Wikipedia, as user query is related to Histori-
cal/Factual events.
By integrating Wikipedia Database or API for factual
queries and Google API for real-time information re-
trieval, our approach ensures that queries are matched
with the most appropriate data source. This agentic
routing process minimizes irrelevant lookups, enhances
retrieval accuracy, and improves factual reliability in
downstream processing.
C. Hybrid Retrieval & Context Unification
To enhance factual consistency and retrieval effec-
tiveness, we propose aHybrid Retrieval Mechanism
that unifiesvectorandgraph-basedretrieval. Graph-
based retrieval captures entity relationships and structural
dependencies, enabling precise reasoning, while vector-
based retrieval ensures semantic matching through high-
dimensional embeddings. Given a query Q, the system

Figure 2: SSRAG: A Comprehensive View of Query Processing, Context Retrieval, and Unification
retrieves data from heterogeneous sources using both
methods, transforming documents into structured graphs
and vector embeddings to preserve both semantic and
structural dependencies.
Algorithm 3outlines the Hybrid Retrieval & Context
Unification process. Specifically, graph-based context
data is first converted into textual representations using
an LLM such as GPT-4. These textual representations
are then transformed into vector embeddings through an
embedding model, effectively vectorizing the nodes and
relationships extracted from the graph. Once graph-based
vectors are generated, they are combined with existing
vector-based embeddings, and the resulting vectors are re-
ranked based on their cosine similarity to the user’s query.
To ensure the diversity and relevance of the contexts, the
top ’2k’ relevant documents are selected, followed by
a deduplication filter where an LLM is employed to
eliminate any redundant contexts by evaluating text and
cosine similarity scores. Finally, the top k relevant texts
are retrieved and used as a context for generating the
final response.
Illustrative example.Given a query“How does reinforce-
ment learning apply to robotics?”, the retrieval operates
in parallel:
•Graph-based retrieval: Extracts structured rela-
tionships between entities such asReinforcement
Learning,Robotics, andPolicy Optimization.
•Vector-based retrieval: Computes semantic embed-
dings to identify relevant documents.Algorithm 3Hybrid Retrieval & Context Unification
1:functionCONTEXTUNIFICATION(topkEmbeds, relevantN-
odes, relations, embedModel, QueryVector (Qv))
2:CombVectors (Cv)←topkEmbeds
3:foreach node in relevantNodesdo
4:nodeRel←GetRelation(node, relations)
5:textRep←LLM Convert(node, nodeRel)
6:vector←embedModel.vectorize(textRep)
7:Cv.add(vector)
8:end for
9:rel vectors, SimScores←GetTop2k(Qv, Cv)
10:retrievedtexts←GetTexts(rel vectors)
11:filteredtexts←TextDedup(retrievedtexts, SimScores)
12:returnfilteredtexts
13:end function
•Joint rerank: Concatenate vector candidates and
graph-to-text candidates, score each by cosine simi-
larity. SelectTop-2kfor redundancy control. Then
doTextDedupto remove near-duplicates, specifically
exact-match removal and high cosine tie-break.
Finally, takeTop-kas the unified evidence set.
An Illustrative Example of SSRAG Framework.
Figure 2Illustrates an example of our proposed frame-
work. The user query, “Modi visit to US,” is inherently
ambiguous as it does not explicitly indicate a specific
timeline—whether referring to a past, present, or future
event. Unlike ChatGPT, which exhibits recency bias
by assuming queries are related to current events and
responding accordingly, our approach makes no such
assumptions. Instead, it systematically considers all

possible timelines to generate a more comprehensive and
contextually enriched response. Furthermore, the context
unification phase leverages a diverse set of structured and
unstructured data sources, ensuring a high-quality and
well-informed response to the user.
D. Implementation Details
The implementation of SSRAG follows a modular
design that integrates vector and graph-based retrieval
into a scalable architecture. This section details the
implementation of each component.
Vector-Based Retrievalenables efficient similarity
search by encoding documents into high-dimensional
embeddings. Given a query, the system performs a
nearest-neighbor search over a precomputed Facebook AI
Similarity Search (FAISS) [29] vector index to retrieve
semantically relevant documents [30]. Specifically, the
query will be embedded using OpenAI’stext-embedding-
3-smallmodel and matched against stored document
embeddings. The process consists of the following steps:
•Data Processing: Text is extracted and preprocessed
using LangChain’s text-splitting.
•Embedding Generation: Document chunks are
transformed into dense vectors using an OpenAI
embedding model.
•Vector Indexing: FAISS stores and indexes embed-
dings for optimized retrieval.
•Similarity Search: Queries are encoded and
matched against stored vectors.
•LLM Integration: Retrieved documents provide
external context for response generation.
Graph-Based Retrievalextracts structured knowledge
by representing information as an entity-relationship
graph, enabling relational reasoning for complex queries.
Instead of retrieving semantically similar text, the system
performs:
•Entity Extraction: Identifies key concepts (e.g.,
Reinforcement Learning, Robotics, Policy Optimiza-
tion).
•Graph Construction: Constructs an entity-
relationship graph in Neo4j.
•Query Execution: Uses Cypher-based queries to
retrieve relevant subgraphs.
•Context Augmentation: Passes structured graph
output to the LLM.
SSRAG-A hybrid approach.The final stage combines
both retrieval strategies into a unified framework, and
performs,
•Parallel Processing: Queries are processed through
both retrieval pipelines.
•Context Unification: Retrieved data is merged into
a unified representation.
•LLM-Based Filtering: Removes redundant context
before response generation.•Final Response: The hybrid contexts are passed to
the LLM for generation.
Our approach leverages both structured and unstructured
data, improving retrieval precision and response quality.
IV. EXPERIMENTS& RESULTS
To evaluate the effectiveness of the proposed SSRAG
framework, we conduct experiments on three benchmark
Q&A datasets across five large language models. This
section describes the experimental setup, presents the
main results, and provides qualitative insights through
case studies.
A. Data, Models & Evaluation Metrics
Datasets.Our experiments leverage three benchmark
datasets that collectively assess truthfulness, fact verifica-
tion, and contextual reasoning in open-domain settings:
TruthfulQA as released in 2022 [31], SQuAD (Stanford
Question Answering Dataset) v2.0 [32] and WikiQA
(2015) [33]. For each dataset we sample 900 questions
uniformly at random (seed = 42) to evaluate the LLM’s
generated responses. Context from Wikipedia snapshot
as of February 14, 2025 was extracted and stored in both
Vector and Graph databases.
LLMs and Reasoning Frameworks.We assess
the performance of SSRAG using five LLMs along-
side two reasoning-based frameworks Chain-of-Thought
(CoT) [12] and Chain-of-Verification (CoVe) [14]. Specif-
ically, we evaluate OpenAI’s GPT-4 [1], Meta’s Llama
2 [34], TinyLlama [35], IBM Granite Model [36], and
Google Gemini [37].
Evaluation Metrics.We consider a set of metrics cover-
ing linguistic quality, factual consistency, and retrieval
effectiveness of LLM generation.BLEU-1[38] measures
the precision of matching n-grams (typically 1-4) between
system outputs and reference texts, where higher scores
[0-1] indicate better alignment with reference words and
more relevant content in the generated output.ROUGE-
1[39] evaluates n-gram recall, with higher scores [0-1]
reflecting better capture of contextually relevant words
in the output. These two metrics assess theretrieval
segmentof the RAG architecture. By comparison,%True
GPT Judgemeasures factual accuracy, with higher scores
[0-100%] showing better alignment with ground truth.
SelfCheckGPT[40] detects hallucinations [0-1], where
a score closer to 1 indicates more inconsistencies in the
model’s output. These metrics evaluate thegeneration seg-
mentof RAG. In addition, we considerRAGAS (Retrieval-
Augmented Generation Assessment Score)[41], a multi-
faceted metric designed for RAG systems. RAGAS
combines retrieval precision, contextual alignment, and
factual correctness.

Table II: Baseline RAG vs. Graph RAG vs. SSRAG on Truthful QA Dataset
Model MetricBaseline
RAGGraph
RAGLearning
to RAGCoTCoT with
Graph RAGCoT with
SSRAGCoVeCoVe
with GraphCoVe with
SSRAG
GPT-4% true (GPT-judge) 57% 67%87%57% 63%81%58% 64%83%
ROUGE-1 43.2% 58%82.3%47% 54.7%83.4%48% 57.7%77%
BLEU-1 47% 65%86.4%45.4% 62.7%82%41.1% 57.9%83%
Self CheckGPT 0.51 0.40.180.47 0.370.160.47 0.420.15
LLaMA 2% true (GPT-judge) 52% 65%83%52% 58%88%53% 59%87%
ROUGE-1 52.1% 59.8%75.2%49.2% 53.3%81.2%51% 58%79.2%
BLEU-1 52% 68%82.4%47.3% 63.1%87.1%45.1% 61.1%85%
Self CheckGPT 0.52 0.380.230.51 0.410.180.54 0.370.2
TinyLLaMA% true (GPT-judge) 47% 57%75%46% 52%72%47% 53%72%
ROUGE-1 32.1% 47%81.1%51.2% 58.7%79%56% 59%73%
BLEU-1 41% 51%91%53.5% 67.8%79%52.2% 55%81%
Self CheckGPT 0.54 0.470.270.56 0.440.290.49 0.410.29
IBM Granite% true (GPT-judge) 52% 61%79%47% 53%69%48% 54%72%
ROUGE-1 51.4% 56%79.8%56.7% 57.2%71%55% 57%76%
BLEU-1 53% 61%89.1%51.1% 64.8%81%51.1% 57.2%83.2%
Self CheckGPT 0.45 0.420.190.46 0.330.230.57 0.380.21
Gemini 1.5% true (GPT-judge) 56% 67%82%48% 54%78%49% 55%79%
ROUGE-1 50% 59%85.2%50% 62.1%76%49.8% 52%77%
BLEU-1 55% 58%91.6%52.67% 65.7%87%52.67% 57.4%87.2%
Self CheckGPT 0.49 0.3330.290.480.310.32 0.48 0.320.3
Table III: Baseline RAG vs. Graph RAG vs. SSRAG on SQuAD Dataset
Model MetricBaseline
RAGGraph
RAGLearning
to RAGCoTCoT with
Graph RAGCoT with
SSRAGCoVeCoVe
with GraphCoVe with
SSRAG
GPT-4% true (GPT-judge) 54% 66%84%56% 62%83%58% 63%84%
ROUGE-1 43.2% 58%82.3%47% 54.7%83.4%48% 57.7%77%
BLEU-1 47% 65%86%45% 63%82%41% 58%83%
Self CheckGPT 0.52 0.370.150.43 0.330.130.47 0.390.13
LLaMA 2% true (GPT-judge) 53% 62%82%51% 57%89%52% 58%89%
ROUGE-1 49.9% 54%82.3%47.8% 58.2%79%52.3% 56%81.2%
BLEU-1 51% 63%90%51% 66%80%50% 60%86%
Self CheckGPT 0.53 0.320.210.49 0.390.20.52 0.380.198
TinyLLaMA% true (GPT-judge) 49% 52%74%43% 49%74%44% 50%69%
ROUGE-1 31.5% 49%85.2%49.1% 59.9%77%51% 57%78%
BLEU-1 47% 58%96%54% 66%84%54% 58%87%
Self CheckGPT 0.47 0.410.290.52 0.410.290.59 0.390.3
IBM Granite% true (GPT-judge) 51% 63%81%50% 56%76%51% 57%77%
ROUGE-1 52% 57%81.2%54.3% 59.9%76.5%53% 56%74%
BLEU-1 54% 59%94%50% 69%85%50% 59%87%
Self CheckGPT 0.42 0.390.210.43 0.320.230.511 0.370.22
Gemini 1.5% true (GPT-judge) 53% 65%83%49% 55%81%50% 56%78%
ROUGE-1 49.7% 56%78.9%49.8% 59.8%71%47.7% 54%72%
BLEU-1 56% 57%93%51% 69%87%51% 60%86%
Self CheckGPT 0.41 0.430.270.460.3 0.280.54 0.310.28
B. Main Results
Our proposed SSRAG framework integratesgraph-
basedandvector-basedretrieval mechanisms to enhance
contextual relevance, reasoning clarity, and factual con-
sistency in LLMs. Our results demonstrate that SSRAG
consistently outperforms conventional RAG models and
standalone retrieval systems on three popular QA datasets,
TruthfulQA, SQuAD and WikiQA, across five LLMs. By
leveraging structured relationships and semantic represen-
tations, our approach achieves significant improvements
in faithfulness, answer relevancy, and context precision,
reinforcing its potential as a robust solution for mitigating
hallucinations and enhancing factual grounding.
Improvements Over Existing RAG Frameworks.We
first compare our SSRAG approach against the Baseline
RAG and Graph RAG methods, focusing on GPT-4’s
performance across TruthfulQA (Table II) and SQuAD
(Table III). On TruthfulQA, SSRAG improves % True(GPT-judge) from 57% (Baseline) and 67% (Graph) to
87%, marking a 30% boost over the Baseline and 20%
over Graph. We observe similar gains in ROUGE-1
(0.432 →0.58→0.823) and BLEU-1 (47% →65%
→86.4%). Additionally, SelfCheckGPT decreases to
0.18—significantly lower than 0.51 (Baseline) and 0.4
(Graph), indicating fewer hallucinations.
On the SQuAD dataset, SSRAG likewise outperforms
both Baseline vector RAG and Graph RAG. Specifically,
% True (GPT-judge) improves from 54% to 66% to 84%,
while ROUGE-1 jumps from 0.432 and 0.58 to 0.823.
BLEU-1 follows a similar trend (47% →65%→86%),
and SelfCheckGPT is minimized at 0.15, substantially
below the 0.52 and 0.37 observed with baseline and graph,
respectively. These consistent improvements highlight
that SSRAG yields both stronger factuality and fewer
hallucinations.
Further breakdowns across different models reinforce

these trends. For instance, TinyLLaMA and IBM Granite
show substantial factuality gains when using SSRAG,
with IBM Granite achieving an 83.2% BLEU-1 score on
TruthfulQA, a significant leap from 53% (Baseline) and
61% (Graph). Similarly, TinyLLaMA sees its BLEU-1
score rise from 41.5% (Baseline) and 51.9% (Graph) to
91.1%, demonstrating that SSRAG enhances both small
and large models alike.
Enhancing Reasoning with Chain of Thought (CoT)
and Chain of Verification (CoVe).Next, we evaluate
how SSRAG interacts with well-known reasoning strate-
gies CoT and CoVe, again focusing on GPT-4 for the
TruthfulQA and SQuAD tasks. On TruthfulQA, CoT with
Baseline RAG achieves 57%, %true (GPT-judge) while
CoT with Graph RAG improves to 63%. In contrast,
CoT with SSRAG reaches 81%, surpassing Baseline and
Graph by 24 and 18 points, respectively. This pattern
extends across LLaMA 2, TinyLLaMA, and IBM Granite,
all of which see at least a 15-20% boost when CoT is
applied alongside SSRAG. A similar trend appears under
CoVe (58% →64%→83%), demonstrating that SSRAG
meaningfully boosts factual accuracy.
We see parallel gains on SQuAD. CoT with SSRAG
attains 83% % True (GPT-judge), notably higher than the
56% and 62% for CoT with Baseline and Graph RAG.
Likewise, CoVe jumps from 58% and 63% to 84%, a
strong indication that combining SSRAG with reasoning
frameworks reduces inaccuracies across multiple data
sets.
Moreover, the consistent reduction in SelfCheckGPT
scores across CoT and CoVe setups further validates that
SSRAG not only improves retrieval accuracy but also
decreases hallucinated completions, resulting in more
factually sound and contextually relevant outputs.
Consistent Enhancement Across Different LLMs and
Multiple Datasets.To thoroughly investigate whether
SSRAG can benefit diverse model architectures and use
cases, we evaluate it on five LLMs that span a range
of parameter sizes, training objectives, and deployment
settings: GPT-4, LLaMA 2, TinyLLaMA, IBM Granite,
and Gemini 1.5. Despite the substantial differences among
these models—GPT-4 and LLaMA 2 serve as high-
parameter research benchmarks, TinyLLaMA is crafted
for lower-resource deployments, and IBM Granite and
Gemini 1.5 emphasize enterprise-oriented or domain-
specific functionalities—SSRAG consistently boosts key
performance measures such as factual accuracy, ROUGE-
1, and BLEU-1. Notably, TinyLLaMA registers one of the
most striking improvements, increasing its factual accu-
racy from 47% under Baseline RAG to 75% with SSRAG,
indicating that even relatively compact models benefit
markedly from the targeted retrieval and generation
enhancements introduced by our approach. Meanwhile,
IBM Granite and Gemini 1.5 each realize gains of roughly15–20% in factual accuracy, coupled with a 10% rise in
ROUGE-1 scores, showing that SSRAG aids both mid-
range and enterprise-driven language models in reducing
errors and producing more reliable outputs. We further
validate SSRAG’s efficacy across two distinct datasets
that each pose different challenges. TruthfulQA sees an
average gain of 20% in factual accuracy across the five
LLMs. In particular, GPT-4 improves from 57% to 87%
on the GPT-judge metric, while LLaMA 2 advances
from 52% to 83%. On the SQuAD benchmark, which
emphasizes reading comprehension and context-sensitive
question answering, GPT-4’s factual accuracy increases
from 54% to 84%, and LLaMA 2’s from 53% to 82%.
These findings demonstrate consistent improvements in
factual correctness, evidence alignment, and generation
quality, as reflected in higher ROUGE-1 and BLEU-1
scores. Across two datasets and five models, our results
showcase SSRAG’s model-agnostic capability to enhance
factual grounding and retrieval relevance.
C. Case Studies
To further evaluate SSRAG, we conduct two case
studies on the WikiQA dataset with 900 questions, aiming
at gaining a deeper understanding of the framework’s
functionality and performance. Both case studies utilize
RAGAS, as traditional metrics like ROUGE-1 or BLEU-
1 often fall short in assessing RAG frameworks due to
their inability to capture the intricacies of retrieval and
reasoning.
SSRAG evaluation with RAGAS. Table IVshows the
RAGAS results across the five LLM models with 900
questions from the WikiQA dataset. Specifically, LLaMA
2 demonstrates the best performance across all RAGAS
metrics, outperforming all other models in factual ac-
curacy and context precision. By comparison, GPT-4,
IBM Granite, and Gemini 1.5 Pro exhibit comparable
performance, ranking just below LLaMA 2, with strong
but slightly lower scores across all metrics. TinyLLaMA,
being a smaller-scale model, naturally exhibited lower
performance but still achieved reasonable results given
its architectural constraints.
Sensitivity on the number of retrieved documents.
Table Vinvestigates the impact of varying the number
of retrieved documents, denoted as N, on the overall
performance of the framework. Documents were retrieved
from both a vector database and a graph database, and
then processed through the unified context module. We
test different values of N, specifically 5, 10, 15, 20, to gen-
erate responses for WikiQA questions and evaluate them
with RAGAS. Table V shows the results and we have
the following observations. First, optimal performance is
achieved at N = 20, where models generated the most con-
textually rich and factually accurate responses. Second,
increasing N beyond 20 resulted in diminishing returns, as

Table IV: RAGAS Evaluation of SSRAG Framework on WikiQA Dataset
Model Faithfulness Answer Relevancy Context Relevancy Context Precision
LLAMA2 92% 89% 91% 88%
GPT-4 88% 87% 86% 85%
IBM Granite 3.0 88% 87% 86% 85%
TinyLLAMA 80% 78% 79% 77%
Gemini 1.5 pro 88% 87% 86% 85%
Table V: Sensitivity on the number of retrieved documents on WikiQA Dataset
Hyperparameter Faithfulness Answer Relevancy Context Relevancy Context Precision
N = 5 75% 77% 79% 75%
N = 10 79% 81% 80% 83%
N = 15 82% 85% 90% 91%
N = 20 91% 89% 92% 92%
excessive retrieval introduced redundant information and
increased computational overhead. Third, Lower N values
(5 or 10) lead to a loss of contextual precision, reducing
the overall quality of generated responses. These results
suggest a trade-off in retrieval settings: while increasing
N improves response quality up to a threshold, excessive
retrieval can cause context overflow within the LLM’s
processing limits.
V. DISCUSSION
In this section, we discuss the broader implications
of, highlighting its strengths, limitations, and potential
directions for future research. While SSRAG offers
substantial advancements, it presents several challenges
that warrant further investigation. One primary concern
is computational overhead, as graph-based retrieval
enhances structured reasoning but incurs additional
preprocessing and query-time costs, making it less
feasible for real-time applications. Another limitation
is the dependence on proprietary LLM APIs, such as
GPT-4, which increases operational costs and restricts
accessibility. While open-source alternatives exist, they
require further optimization to match the performance
of proprietary models. Additionally, although effective
in open-domain QA, adapting SSRAG to specialized do-
mains (e.g., healthcare, finance) requires tailored retrieval
mechanisms and domain-specific graph construction.
Lastly, scalability in large-scale deployments remains an
open question. Integrating both vector and graph-based
retrieval efficiently at scale requires further research into
retrieval fusion techniques that balance speed, accuracy,
and computational efficiency.
To further refine and enhance SSRAG, future research
can explore several promising directions. First, Optimized
graph-based retrieval could be developed through ad-
vanced indexing methods such as compressed adjacency
lists, incremental graph updates, and hierarchical graph
structures. These optimizations would reduce querylatency while preserving retrieval effectiveness. Addition-
ally, advanced fusion strategies could be implemented
by exploring hybrid techniques such as hypergraph
embeddings and retrieval weighting mechanisms to
better integrate structured and unstructured knowledge
representations. Another potential avenue is adaptive
learning mechanisms, where reinforcement learning and
user-feedback-driven retrieval updates dynamically refine
retrieval processes, improving accuracy and adaptability
to evolving knowledge bases. By addressing these chal-
lenges and pursuing these research directions, SSRAG
has the potential to become a highly scalable, efficient,
and domain-adaptive retrieval system.
VI. CONCLUSION
We introduced SSRAG, a hybrid RAG framework that
enhances factual accuracy and structured reasoning in
large language models. By integrating both graph-based
and vector-based retrieval mechanisms with dynamic
query augmentation and agentic query routing, our ap-
proach effectively reduces hallucinations while improving
response relevance. Our methodology provides a scalable
and reliable advanced RAG framework that balances
structured reasoning with adaptable context retrieval.
By leveraging multi-source retrieval strategies, SSRAG
establishes a foundation for more factually grounded and
context-aware LLMs.
REFERENCES
[1]J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L.
Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat
etal., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774 ,
2023.
[2]A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra,
A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann
etal., “Palm: Scaling language modeling with pathways,” Journal
ofMachine Learning Research , vol. 24, no. 240, pp. 1–113, 2023.
[3]G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju,
S. Pathak, L. Sifre, M. Rivi `ere, M. S. Kale, J. Love etal., “Gemma:
Open models based on gemini research and technology,” arXiv
preprint arXiv:2403.08295, 2024.

[4]H. Touvron, T. Lavril, G. Izacard etal., “LLaMA: Open
and efficient foundation language models,” arXiv preprint
arXiv:2302.13971, 2023.
[5]“The claude 3 model family: Opus, sonnet, haiku.” [Online].
Available: https://api.semanticscholar.org/CorpusID:268232499
[6]T. Eloundou, S. Manning, P. Mishkin, and D. Rock, “Gpts are
gpts: An early look at the labor market impact potential of large
language models,” arXiv preprint arXiv:2303.10130, 2023.
[7]T. H. Kung, M. Cheatham, A. Medenilla, etal., “Performance of
ChatGPT on USMLE: Potential for AI-Assisted Medical Educa-
tion Using Large Language Models,” JMIR Medical Education ,
2023, in Press.
[8]J. Cui, Z. Li, Y . Yan, B. Chen, and L. Yuan, “Chatlaw: Open-source
legal large language model with integrated external knowledge
bases,” arXiv preprint arXiv:2306.16092v1, 2023.
[9]K. Benkirane, L. Gongas, S. Pelles, N. Fuchs, J. Darmon,
P. Stenetorp, D. Adelani, and E. S ´anchez, “Machine translation
hallucination detection for low and high resource languages
using large language models,” in Findings oftheAssociation
forComputational Linguistics: EMNLP, 2024.
[10] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora,
S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill
etal., “On the opportunities and risks of foundation models,”
arXiv preprint arXiv:2108.07258, 2021.
[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel
etal., “Retrieval-augmented generation for knowledge-intensive
nlp tasks,” in International Conference onNeural Information
Processing Systems, 2020.
[12] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia,
E. H. Chi, Q. V . Le, and D. Zhou, “Chain-of-thought prompting
elicits reasoning in large language models,” in International
Conference onNeural Information Processing Systems , 2022, pp.
24 824–24 837.
[13] T. Kojima, S. S. Gu, M. Reid, Y . Matsuo, and Y . Iwasawa,
“Large language models are zero-shot reasoners,” in International
Conference onNeural Information Processing Systems , 2022, pp.
22 199–22 213.
[14] S. Dhuliawala, M. Komeili, J. Xu, R. Raileanu, X. Li, A. Celiky-
ilmaz, and J. Weston, “Chain-of-verification reduces hallucination
in large language models,” in Findings oftheAssociation for
Computational Linguistics ACL, 2024, pp. 3563–3578.
[15] T. Ferdinan, J. Koco ´n, and P. Kazienko, “Into the unknown:
Self-learning large language models,” in IEEE International
Conference onData Mining Workshops (ICDMW), 2024.
[16] C.-Y . Tai, Z. Chen, T. Zhang, X. Deng, and H. Sun, “Exploring
chain of thought style prompting for text-to-sql,” in Proceedings
oftheConference onEmpirical Methods inNatural Language
Processing, 2023.
[17] K. Stechly, K. Valmeekam, and S. Kambhampati, “Chain of
thoughtlessness? an analysis of cot in planning,” in International
Conference onNeural Information Processing Systems, 2024.
[18] A. Martin, H. F. Witschel, M. Mandl, and M. Stockhecke,
“Semantic verification in large language model-based retrieval
augmented generation,” in Proceedings oftheAAAI Symposium
Series, vol. 3, no. 1, 2024, pp. 188–192.
[19] T. E. Kim and F. Diaz, “Towards fair rag: On the impact of
fair ranking in retrieval-augmented generation,” in ACM SIGIR
Conference onInnovative Concepts andTheories inInformation
Retrieval (ICTIR), 2025.
[20] X. Wang, J. Wei, D. Schuurmans, Q. V . Le, E. H. Chi, S. Narang,
A. Chowdhery, and D. Zhou, “Self-consistency improves chain
of thought reasoning in language models,” in International
Conference onLearning Representations, 2024.
[21] E. Fadeeva, A. Rubashevskii, A. Shelmanov, S. Petrakov, H. Li,
H. Mubarak, E. Tsymbalov, G. Kuzmin, A. Panchenko, T. Baldwin
etal., “Fact-checking the output of large language models
via token-level uncertainty quantification,” in Findings ofthe
Association forComputational Linguistics ACL, 2024.[22] S. Farquhar etal., “Detecting hallucinations in large language
models using semantic entropy-based uncertainty estimators,”
Nature, vol. 615, pp. 82–89, 2024.
[23] J. Huang, M. Wang, Y . Cui, J. Liu, L. Chen, T. Wang, H. Li,
and J. Wu, “Layered query retrieval: An adaptive framework for
retrieval-augmented generation in complex question answering
for large language models,” Applied Sciences , vol. 14, no. 23, p.
11014, 2024.
[24] L. Forer and T. Hope, “Inferring scientific cross-document
coreference and hierarchy with definition-augmented relational
reasoning,” arXiv preprint arXiv:2409.15113, 2024.
[25] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and
C. D. Manning, “Raptor: Recursive abstractive processing for
tree-organized retrieval,” arXiv preprint arXiv:2401.18059 , 2024.
[26] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag:
Learning to retrieve, generate, and critique through self-reflection,”
arXiv preprint arXiv:2310.11511, 2023.
[27] T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica,
and J. E. Gonzalez, “Raft: Adapting language model to domain
specific rag,” arXiv preprint arXiv:2403.10131, 2024.
[28] B. Sarmah, B. Hall, R. Rao, S. Patel, S. Pasquali, and D. Mehta,
“Hybridrag: Knowledge graph-augmented retrieval-augmented
generation for domain-specific applications,” 2024. [Online].
Available: https://arxiv.org/abs/2408.04948
[29] M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E.
Mazar ´e, M. Lomeli, L. Hosseini, and H. J ´egou, “The faiss library,”
arXiv preprint arXiv:2401.08281, 2024.
[30] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity
search with gpus,” IEEE Transactions onBigData , vol. 7, no. 3,
pp. 535–547, 2019.
[31] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring
how models mimic human falsehoods,” in Proceedings of
the Annual Meeting ofthe Association forComputational
Linguistics , S. Muresan, P. Nakov, and A. Villavicencio, Eds.,
2022.
[32] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “Squad: 100,000+
questions for machine comprehension of text,” in Proceedings
oftheConference onEmpirical Methods inNatural Language
Processing, 2016.
[33] Y . Yang, W.-t. Yih, and C. Meek, “Wikiqa: A challenge
dataset for open-domain question answering,” in Proceedings
oftheConference onEmpirical Methods inNatural Language
Processing, 2015.
[34] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi,
Y . Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale etal.,
“Llama 2: Open foundation and fine-tuned chat models,” arXiv
preprint arXiv:2307.09288, 2023.
[35] J. Min and Others, “Tinyllama: Low-resource inference of llama
for edge devices,” 2023, accessed: September 15, 2024.
[36] IBM, “IBM Granite Models for Watsonx.ai,” 2023, accessed:
September 15, 2024.
[37] Google DeepMind, “Google Gemini: Next-generation foundation
model,” 2023, accessed: September 15, 2024.
[38] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method
for automatic evaluation of machine translation,” in Proceedings
oftheannual meeting oftheAssociation forComputational
Linguistics, 2002.
[39] C.-Y . Lin, “ROUGE: A package for automatic evaluation of
summaries,” in ACL Workshop onText Summarization Branches
Out, 2004.
[40] P. Manakul, A. Liusie, and M. Gales, “Selfcheckgpt: Zero-resource
black-box hallucination detection for generative large language
models,” in The Conference onEmpirical Methods inNatural
Language Processing, 2023.
[41] S. Es, J. James, L. E. Anke, and S. Schockaert, “Ragas: Automated
evaluation of retrieval augmented generation,” in Proceedings of
theConference oftheEuropean Chapter oftheAssociation for
Computational Linguistics: System Demonstrations, 2024.