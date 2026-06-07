# Agent-Orchestrated Adaptive RAG: A Comparative Study on Structured and Multi-Hop Retrieval

**Authors**: Anuj Maharjan, Devinder Kaur, Richard Molyet

**Published**: 2026-06-04 03:38:46

**PDF URL**: [https://arxiv.org/pdf/2606.05658v1](https://arxiv.org/pdf/2606.05658v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding their responses in external knowledge, but conventional pipelines rely on static, single-step retrieval that limits performance on complex queries. This paper presents an Agent-Orchestrated Adaptive RAG framework that introduces dynamic query decomposition, iterative retrieval, and a bounded self-reflective evaluation loop. We evaluate the system across two complementary datasets: a domain-specific DevOps knowledge base and the multi-hop reasoning benchmark MuSiQue. Using metrics that include overall score, citation accuracy, mean reciprocal rank, and topic coverage, we find that query decomposition yields consistent gains in the structured domain (overall score $+0.04$, MRR $+0.17$ on DevOps) but degrades ranking precision on the multi-hop benchmark, while the reflection mechanism improves citation accuracy at a substantial latency cost. These contrasting results show that agentic enhancements are not universally beneficial and must be applied selectively according to query and domain characteristics. Our findings argue for adaptive, cost-aware orchestration rather than uniformly aggressive reasoning pipelines.

## Full Text


<!-- PDF content starts -->

Agent-Orchestrated Adaptive RAG: A Comparative
Study on Structured and Multi-Hop Retrieval
Anuj Maharjan
Dept. of Electrical Engineering
and Computer Science
University of Toledo
Toledo, OH, USA
anuj.maharjan@utoledo.eduDevinder Kaur
Dept. of Electrical Engineering
and Computer Science
University of Toledo
Toledo, OH, USA
devinder.kaur@utoledo.eduRichard G. Molyet
Dept. of Electrical Engineering
and Computer Science
University of Toledo
Toledo, OH, USA
richard.molyet@utoledo.edu
Abstract—Retrieval-Augmented Generation (RAG) enhances
Large Language Models (LLMs) by grounding their responses
in external knowledge, but conventional pipelines rely on static,
single-step retrieval that limits performance on complex queries.
This paper presents an Agent-Orchestrated Adaptive RAG
framework that introduces dynamic query decomposition, it-
erative retrieval, and a bounded self-reflective evaluation loop.
We evaluate the system across two complementary datasets:
a domain-specific DevOps knowledge base and the multi-hop
reasoning benchmark MuSiQue. Using metrics that include
overall score, citation accuracy, mean reciprocal rank, and topic
coverage, we find that query decomposition yields consistent gains
in the structured domain (overall score+0.04, MRR+0.17on
DevOps) but degrades ranking precision on the multi-hop bench-
mark, while the reflection mechanism improves citation accuracy
at a substantial latency cost. These contrasting results show that
agentic enhancements are not universally beneficial and must be
applied selectively according to query and domain characteristics.
Our findings argue for adaptive, cost-aware orchestration rather
than uniformly aggressive reasoning pipelines.
Index Terms—Retrieval-Augmented Generation, Agentic AI,
Query Decomposition, Multi-Hop Reasoning, Large Language
Models
I. INTRODUCTION
Large Language Models (LLMs) have shown strong ca-
pabilities in natural language understanding and generation,
yet they remain constrained by two persistent limitations:
factual hallucination and an inability to access knowledge
beyond their training data [1], [2]. These limitations are
especially consequential in high-stakes operational settings
such as DevOps and Site Reliability Engineering (SRE), where
the correct interpretation of technical documentation can be
the difference between rapid incident resolution and prolonged
downtime [3].
Retrieval-Augmented Generation (RAG) addresses these
limitations by allowing a model to consult an external knowl-
edge base before generating a response [4], [5]. Conventional
NaiveRAG follows a fixed, single-pass procedure: the query
is embedded, the top-kmost similar chunks are retrieved, and
these are prepended to the prompt as context [6]. This works
well for isolated factual lookups, but it struggles with queries
that require reasoning across several documents or chaining
intermediate results [7], [8].Agentic RAG extends this paradigm by delegating retrieval
to autonomous agents that can plan, retrieve iteratively, and
critique their own output [9], [10]. Instead of a single retrieve-
then-generate step, an agentic system can decompose a com-
plex query into sub-questions, gather evidence for each, and
refine its answer through self-reflection [11], [12]. Interest in
this direction has grown quickly, but most studies evaluate
agentic RAG on a single domain or benchmark. As a result,
it remains unclear how these enhancements behave across
domains that differ in structure and reasoning demand.
This paper addresses that gap through a deliberately dual-
dataset evaluation. We make the following contributions:
1) We implement a complete Agent-Orchestrated Adaptive
RAG system composed of a Query Classifier, a Query
Decomposer, an Answer Evaluator, and a central Or-
chestrator, built on a fully local, privacy-first inference
stack [13], [14].
2) We evaluate the system on two complementary datasets:
a custom DevOps knowledge base that represents struc-
tured, domain-specific retrieval, and MuSiQue [15], a
compositional multi-hop benchmark. This pairing lets us
observe agentic behavior under contrasting conditions.
3) We characterize the tradeoffs introduced by decompo-
sition and reflection, showing empirically that decom-
position is domain-dependent and that reflection adds
significant latency for inconsistent quality gains.
II. BACKGROUND ANDRELATEDWORK
A. Generative Models and LLMs
Generative models learn an underlying data distribution
and sample novel outputs from it, a lineage that runs from
generative adversarial networks [16], [17] to the autoregressive
language models that dominate today [18]. A language model
factorizes the probability of a token sequence and predicts each
token conditioned on its predecessors,
P(x n|x1, x2, . . . , x n−1),(1)
which lets it generate coherent, contextually appropriate text
across domains.
The success of modern LLMs is largely attributable to the
Transformer architecture [19], which replaced recurrence witharXiv:2606.05658v1  [cs.IR]  4 Jun 2026

attention. The self-attention operation weighs the relevance of
every token to every other token,
Attention(Q, K, V) = softmaxQK⊤
√dk
V,(2)
whereQ,K, andVare the query, key, and value projections
andd kis the key dimensionality. Multi-head attention runs
several such operations in parallel to capture diverse linguis-
tic relationships, while feedforward sublayers and positional
encodings [20], [21] supply non-linearity and sequence order.
Input text is first segmented by subword tokenization such as
byte-pair encoding [22] and embedded into dense vectors be-
fore passing through the stacked Transformer layers [23], [24].
At generation time, decoding strategies—greedy selection,
beam search [25], and stochastic methods such as temperature
scaling and nucleus sampling [26]—balance coherence against
diversity.
Despite their fluency, these models exhibit well-documented
limitations: they hallucinate and lack grounding in verifiable
sources [1], can encode social and cultural bias from their
training data [27], and are costly to train and serve [28]. These
limitations are precisely what retrieval-based augmentation is
designed to mitigate.
B. Naive and Advanced RAG
The foundational RAG framework [4] couples the para-
metric knowledge stored in an LLM’s weights with non-
parametric knowledge retrieved from an external corpus, and
demonstrated that grounding generation in retrieved evidence
reduces hallucination and improves factual accuracy. A stan-
dard pipeline has three components. Theretrieverfetches
relevant documents, typically via dense retrieval in which
queries and documents are encoded by neural encoders [29]–
[31] and compared in vector space, as in Dense Passage
Retrieval [32] with similaritysim(q, d) =q⊤d. Theaugmen-
tationstep injects the retrieved passages into the prompt, and
thegenerator—the LLM—produces a response conditioned
on both the query and the retrieved context.
Naive RAG retrieves a fixed top-kset and appends it
directly, which is simple but admits irrelevant or redundant
passages [6]. Advanced RAG counters this with query rewrit-
ing and expansion [33], re-ranking of candidates, and chunking
strategies that improve context selection [8]. Recent empir-
ical work on chunking, retrieval, and re-ranking for policy-
document question answering confirms that these pipeline
design choices substantially shape downstream answer quality
[34]. Because exhaustive nearest-neighbor search becomes
prohibitive as a corpus grows, efficient retrieval relies on ap-
proximate nearest-neighbor indexes such as FAISS [35]–[37],
which trade a small recall loss for large gains in latency. Multi-
hop RAG variants extend this further by retrieving iteratively
rather than once, refining the context across reasoning steps
[7].
C. Agentic RAG
Surveys have charted the shift from static pipelines to
agentic systems capable of dynamic tool use, iterative retrieval,and self-correction [9], [10], [38]. Whereas a Naive pipeline
executes a predetermined sequence, an agentic system deter-
mines its control flow at runtime, using intermediate outputs
to guide subsequent actions and invoking external tools only
as needed [39]. Agentic RAG systems are typically organized
around four interacting capabilities:query understanding and
planning, which interprets the request and selects a solving
strategy;tool and retrieval orchestration, which decides when
and how to retrieve;multi-hop reasoning and decomposition,
which breaks complex problems into sub-tasks whose results
are later combined; andself-reflection, which critiques gener-
ated output and triggers refinement when issues are detected
[40], [41].
These capabilities have concrete instantiations in the liter-
ature. The ReAct framework [12] formalized the interleaving
of reasoning traces and actions, Toolformer [42] showed that
models can learn to call external tools, and Self-RAG [11]
introduced learned reflection signals for selective retrieval and
critique. Corrective RAG [43] added an explicit retrieval-
correction step, and Adaptive-RAG [44] routed queries to
retrieval strategies of differing complexity based on predicted
question difficulty [45], [46]. Our system shares this lineage
but is distinguished by an explicit multi-agent decomposition
architecture that we evaluate across two contrasting domains
rather than a single benchmark.
D. Reasoning and Multi-Hop Retrieval
Several prompting frameworks elicit explicit reasoning
in LLMs. Chain-of-Thought prompting [47] encourages the
model to generate intermediate reasoning steps before commit-
ting to an answer, improving performance on tasks that require
composition. Least-to-Most prompting [48] decomposes a hard
problem into a sequence of simpler sub-problems solved in
order, an approach that aligns closely with the query decompo-
sition strategy adopted in this work. Self-refinement methods
iterate on a model’s own draft using its self-generated feedback
[49], and inference-time self-correction has been surveyed as
a general mechanism for improving reliability [46].
Multi-hop question answering benchmarks provide the
testbed for these methods. HotpotQA [50] popularized explain-
able multi-hop evaluation, and MuSiQue [15] was deliberately
constructed by composing single-hop questions so that answer-
ing requires genuine evidence integration across independent
documents and resists single-hop shortcuts. Tree- and graph-
structured retrieval has been proposed to better capture the
dependencies among reasoning steps [51]. Despite this activ-
ity, comparatively few studies place a multi-hop benchmark
alongside a structured, domain-specific corpus to isolate how
the same agentic mechanisms behave under different reasoning
demands—the contrast this work is designed to expose.
III. SYSTEMARCHITECTURE
The proposed system extends the conventional RAG
pipeline with an agent-orchestrated control layer capable of
adaptive routing, query decomposition, and bounded self-
correction. Rather than committing to a single retrieval strat-

User Query
Query Embedding
Model
Vector Database
(FAISS)
Top-kRetrieved
Chunks
Prompt Construction
LLM (Llama-
3.1-8B)
Generated Answer
Fig. 1. Standard Naive RAG pipeline: a single-pass retrieval process with no
adaptive control or iterative reasoning.
User Query
Query
Classifier Agent
Orchestrator
Query
Decomposer AgentDirect Retrieval Path
Retriever +
LLM (FAISS)
Answer
Evaluator Agent
Final ResponseReflection
Fig. 2. Proposed Agent-Orchestrated Adaptive RAG architecture, showing
dynamic routing, decomposition, and a bounded self-reflective evaluation loop
(dashed feedback).
egy, the system selects a strategy per query through coordi-
nated interaction among specialized agents.
A. Overview
Figure 1 shows the Naive RAG baseline for reference, and
Fig. 2 shows the proposed architecture. A central Orchestrator
routes each query, invoking only the agents a given query
requires. Two adaptive behaviors sit on top of the baseline
retrieval path: decomposition-based reasoning for multi-hop
queries and reflection-based correction for quality assurance.B. Orchestrator and Control Logic
The Orchestrator is the central controller. It routes queries
based on the classifier output, triggers decomposition when
required, and manages the evaluation–reflection cycle. Its
decision logic is rule-based:
1) If the query is simple→direct retrieval.
2) If the query is complex→decomposition pipeline.
3) If evaluation fails→trigger reflection (at most two
retries).
This design ensures that the system incurs additional com-
putation only when a query warrants it, preserving efficiency
for the common simple case.
C. Query Classification and Metadata Filtering
The Query Classifier Agent categorizes each incoming
request into document types, enabling metadata-aware filtering
before similarity ranking [45]. Because the vector store keeps
document metadata alongside dense embeddings, filtering by
attributes such as document type (for example, restricting
retrieval to runbooks or incident reports) narrows the candidate
space and improves semantic precision. This separates Agentic
RAG from purely similarity-driven Naive pipelines, which
cannot reason over structured attributes.
D. Query Decomposition
The Query Decomposer Agent enables multi-hop reasoning
by splitting a complex query into ordered sub-queries [48],
[51]. The workflow is: (i) detect multi-hop structure, flagging
queries that involve multiple entities or chained dependencies;
(ii) generate ordered sub-queries such that earlier results can
inform later retrieval; (iii) retrieve independently for each sub-
query; and (iv) aggregate the intermediate answers into a single
coherent response that aligns with the original intent.
E. Answer Evaluation and Reflection
The Answer Evaluator Agent adds a self-reflective layer
that critiques generated responses along four axes: relevance
to the query, citation accuracy, support (grounding in retrieved
context), and hallucination detection [11], [49]. When the
evaluator flags an issue, the Orchestrator reprocesses the
query—refining the prompt or adjusting the retrieved context
by selecting different chunks or re-ranking results. To bound
cost and latency, the reflection loop is capped at two iterations,
balancing potential quality gains against responsiveness [46].
F . Data Preprocessing and Indexing
The quality of a RAG system depends as much on how
its corpus is prepared as on the model that consumes it;
retrieval failures are frequently attributable to preprocessing
and segmentation rather than to the generator [4], [8]. Hetero-
geneous corpora are particularly challenging, since standard
text extraction often discards structural cues—table headers,
hierarchical lists, contextual groupings—that are essential for
accurate retrieval. We therefore convert source files (PDF,
DOCX, YAML) to structured Markdown using IBM Docling
[52], [53], which uses layout-aware models to preserve visual

Raw Documents
(PDF, DOCX,
YAML)
Docling Structural
Conversion
Structured
Markdown Output
YAML Frontmatter
Injection
Semantic Chunking
(600 / 100 overlap)
Embedding
Generation (BGE-
base-en-v1.5)
FAISS Index
Construction
Fig. 3. Document ingestion and preprocessing pipeline for the proposed
DevOps knowledge base.
hierarchy and decompose documents into semantic blocks
rather than emitting an undifferentiated stream of text.
A custom preprocessing layer then enriches each document
with YAML frontmatter encoding attributes such as document
type and service, which the vector store later uses for attribute-
based filtering [45]. This metadata is what allows the agentic
pipeline to reason over structure, not just surface similar-
ity. Documents are segmented with a token-based semantic
chunking strategy: 600-token chunks with a 100-token overlap.
The overlap preserves contextual continuity across boundaries,
and the chunk size was selected empirically to balance three
competing constraints—the LLM context window, the retrieval
discrimination capacity of the index, and the preservation of
technical coherence. Segmentation involves a recall–precision
tradeoff: larger chunks improve semantic completeness while
smaller chunks sharpen retrieval specificity, and improper
segmentation can isolate evidence needed to reconstruct a
multi-hop reasoning chain.
Each chunk is embedded with the BGE encoder and indexed
in FAISS together with its metadata, enabling hybrid filtering
in which attribute constraints (for example, restricting to
runbooks or incident reports) are applied before similarity
ranking to narrow the candidate space. In the agentic setting
the index is not a passive store but an interactive component:
agents may issue multiple retrieval calls, adjust filters, and re-
rank results across reasoning steps, turning retrieval from a
single lookup into a sequential decision process. The full in-
gestion pipeline—structural conversion, frontmatter injection,
chunking, embedding, and index construction—is summarized
in Fig. 3.TABLE I
DEVOPSKNOWLEDGEBASECOMPOSITION
Document Type Count Avg. Words Key Metadata
Standard 5 150 Doc ID, Doc Type
Architecture 10 200 Doc ID, Doc Type
Runbooks 20 180 Doc ID, Doc Type
Incidents 15 100 Doc ID, Doc Type
Postmortem 15 120 Doc ID, Doc Type
Onboarding 15 120 Doc ID, Doc Type
G. Implementation and Knowledge Base
The generative core is Meta’s Llama-3.1-8B-Instruct, quan-
tized to 4-bit GGUF for local inference viallama.cpp[13],
[14]. Retrieval uses the BAAIBGE-base-en-v1.5embed-
ding model [31], [54], chosen for its strong performance on
semantic-similarity benchmarks, and vector search is handled
by FAISS [35]. The entire stack runs locally, a deliberate
privacy-first choice for the DevOps setting, where operational
documentation is often sensitive. Similarity between a query
embeddingqand a document embeddingd iis computed by
cosine distance,
sim(q, d i) =q·d i
∥q∥∥d i∥,(3)
which is scale-invariant and well-suited to nearest-neighbor
search.
The DevOps knowledge base comprises 80 documents
(approximately 10,000 words) spanning six documentation
types—Standard, Architecture, Runbooks, Incidents, Post-
mortem, and Onboarding—generated with LLM assistance and
validated manually to reflect realistic organizational knowl-
edge. Although domain-specific, the corpus is constructed to
emulate broader properties of real knowledge bases, including
structural heterogeneity, implicit cross-document relationships,
and multi-step reasoning requirements.
IV. EXPERIMENTALSETUP
A. Datasets
We evaluate on two contrasting datasets. TheDevOps
knowledge base represents structured, domain-specific re-
trieval with terminology alignment and contextual grounding
in a constrained space. Its composition is summarized in
Table I, spanning six documentation types with distinct lengths
and roles—from short Standard references to longer Archi-
tecture and Runbook documents.MuSiQue[15] represents
open-domain compositional reasoning, where evidence must
be integrated across multiple independent documents. Together
they probe system behavior under both structured and multi-
hop conditions.
B. Evaluation Metrics
Table II summarizes the metrics. Overall Score is an aggre-
gate quality measure; Critical Source Recall captures whether
key supporting documents are retrieved; MRR and Success@5
measure ranking quality; Citation Accuracy measures whether

TABLE II
EVALUATIONMETRICSUSED IN THESTUDY
Metric Description
Overall Score Aggregate performance measure
Critical Source Recall Retrieval of key supporting documents
MRR Ranking quality of retrieved results
Success@5 Retrieval success within top 5 results
Citation Accuracy Correctness of cited sources
Topic Coverage Breadth of information covered
Latency End-to-end response time
TABLE III
IMPACT OFQUERYDECOMPOSITION(BASELINE VS. DECOMPOSED)
DevOps MuSiQue
MetricBase Decomp. Base Decomp.
Overall Score 0.814 0.855 0.786 0.809
Critical Source Recall 0.750 0.806 0.833 0.604
MRR 0.556 0.722 0.469 0.102
Success@5 0.833 1.000 1.000 0.063
Citation Accuracy 0.750 0.917 0.906 1.000
Topic Coverage 0.625 0.558 0.609 0.859
Avg. Latency (s) 21.00 47.66 21.84 74.87
cited sources support the claims; Topic Coverage measures
breadth; and Latency measures end-to-end response time.
V. RESULTS ANDDISCUSSION
A. Impact of Query Decomposition
Decomposition behaves very differently across the two
datasets. On the DevOps corpus (Table III), it improves nearly
every retrieval metric: overall score rises from 0.814 to 0.855,
MRR from 0.556 to 0.722, Success@5 reaches a perfect 1.0,
and citation accuracy climbs to 0.917. The cost is latency,
which more than doubles from 21 s to 48 s.
On MuSiQue (Table III), the picture inverts. Although
citation accuracy reaches 1.0 and topic coverage rises sharply
(0.609 to 0.859), ranking-based metrics collapse: MRR falls
from 0.469 to 0.102 and Success@5 from 1.0 to 0.063.
Decomposition therefore broadens coverage but fragments the
contextual signal that ranking depends on, harming precision
in genuinely multi-hop settings.
B. Impact of Reflection
The reflection mechanism yields, at best, mixed results
(Table IV). On DevOps, overall score actuallydecreasesfrom
0.870 to 0.781 with no gain in recall or ranking, while latency
roughly doubles. On MuSiQue, overall score also falls (0.721
to 0.666) and recall and MRR both decline; citation accuracy
improves slightly, but latency increases roughly sixfold, from
17 s to 104 s. These results indicate that, as currently designed,
reflection imposes a large computational cost without reliable
quality gains.
Fig. 4. Decomposition effectiveness across datasets.
TABLE IV
REFLECTIONIMPACT(BASELINE VS. FULLAGENTIC)
DevOps MuSiQue
MetricBase Full Base Full
Overall Score 0.870 0.781 0.721 0.666
Critical Source Recall 1.000 1.000 0.706 0.618
MRR 0.778 0.778 0.757 0.659
Citation Accuracy 1.000 1.000 0.882 0.941
Avg. Latency (s) 10.57 21.78 17.21 103.53
C. Strategy Distribution
The Orchestrator selects among three strategies: Standard
RAG (direct retrieval), Complex RAG (multi-step retrieval
without explicit decomposition), and Decomposed RAG. On
DevOps, the system routes most queries through standard
retrieval (69.2%), with decomposition adopted for only about
5% of queries—consistent with the structured, largely single-
hop nature of the domain. On MuSiQue, complex queries
dominate, and decomposition is adopted for roughly 16%
of queries, reducing reliance on undifferentiated complex
retrieval. Notably, the distribution is identical between the
decomposition-enabled and full-agentic configurations, con-
firming that reflection operates as a post-hoc corrective step
rather than a routing mechanism.
Several patterns emerge across the two datasets. First,
strategy selection is strongly dependent on dataset complex-
ity: the structured DevOps corpus favors standard retrieval,
whereas the multi-hop MuSiQue benchmark draws on ad-
vanced strategies far more often. Second, decomposition in-
troduces a new processing pathway but never dominates the
pipeline, indicating that the orchestrator invokes it selectively
rather than reflexively. Third, the comparatively small share
of decomposed queries—even on MuSiQue, where 49% of
queries remain in the complex-RAG category—suggests that
the decomposition trigger is conservative and that not all
complex queries that could benefit are actually routed for
decomposition. This points directly to triggering policy as a
high-value target for future optimization.
D. Tradeoff Analysis
Three tradeoffs emerge consistently.Accuracy vs. latency:
every agentic addition increases response time, with reflection
on MuSiQue approaching a sixfold slowdown—likely unac-
ceptable in real-time DevOps use.Precision vs. coverage:

Fig. 5. Reflection effectiveness: quality-component breakdown.
Fig. 6. Routing strategy distribution across DevOps Dataset.
decomposition broadens retrieval coverage but can dilute the
contextual signal that ranking metrics depend on, lowering
MRR and Success@5 in multi-hop settings.Control vs.
efficiency: modular agents improve interpretability and enable
adaptive behavior, but at the cost of additional LLM calls and
retrieval operations that limit scalability.
E. Summary of Findings
Decomposition helps in structured domains where chained
reasoning maps cleanly onto well-defined relationships, but it
is not universally beneficial and degrades ranking precision on
harder multi-hop data. Reflection offers limited, inconsistent
gains relative to its cost. And because a large share of queries
are handled well by standard RAG, adaptive orchestration—
applying expensive strategies only when warranted—is essen-
tial rather than optional.
F . Limitations and Threats to Validity
Several limitations qualify these findings and frame the
scope of our claims. First, the DevOps knowledge base is
modest in scale (80 documents, roughly 10,000 words) and
was generated with LLM assistance before manual validation;
while it is designed to emulate realistic structural heterogene-
ity, results on a larger production corpus may differ, and the
absolute metric values should be read as indicative rather than
definitive. Second, the evaluation query sets are correspond-
ingly small, so per-metric differences—particularly the sharp
ranking drops on MuSiQue—carry wide confidence intervals
and would benefit from larger samples and significance testing.
Third, the entire study uses a single generator (Llama-3.1-
8B-Instruct) and a single embedding model; the interaction
between agentic strategies and model capacity is unexplored,
and a stronger generator might absorb some errors that decom-
position currently compensates for. Fourth, the orchestrator’s
Fig. 7. Routing strategy distribution across MusiQue Dataset.
routing relies on rule-based heuristics, so the observed strategy
distributions reflect those specific rules rather than an optimal
policy. Finally, latency figures are environment-dependent and
were measured under local quantized inference; they capture
relative trends across configurations rather than portable ab-
solute timings. We view these constraints not as undermining
the central result—that agentic benefits are domain-dependent
and cost-laden—but as delimiting the conditions under which
it was established.
VI. CONCLUSION ANDFUTUREWORK
We presented an agent-orchestrated adaptive RAG system
that combines query classification, decomposition, answer
evaluation, and a bounded reflection loop to select retrieval
strategies dynamically. Across a structured DevOps corpus
and the multi-hop MuSiQue benchmark, query decomposition
improved retrieval quality in the structured domain but reduced
ranking precision in the multi-hop setting, and reflection
delivered marginal quality gains at substantial latency cost.
The clear lesson is that agentic enhancements must be ap-
plied selectively and with explicit cost awareness rather than
uniformly.
A. Adaptive Strategy Selection
A central limitation we observed is that decomposition and
reflection are triggered by static, heuristic rules. Future work
can replace these with learning-based strategy selection, in
which the system predicts the optimal reasoning strategy from
query features. Reinforcement learning or meta-learning could
be used to balance quality against efficiency directly, learn-
ing when decomposition genuinely helps—as in the DevOps
domain—and when it harms ranking, as on MuSiQue.
B. Cost-Aware Optimization
Given the large latency penalties measured for agentic com-
ponents, orchestration should become explicitly cost-aware.
Incorporating a cost model that accounts for retrieval time, the
number of reasoning steps, and per-call inference cost would
let the orchestrator weigh expected quality gains against their
computational price, which is essential for latency-sensitive
DevOps deployments where the sixfold reflection slowdown
we observed would be unacceptable.
C. Improved Query Decomposition
The current decomposition strategy can be refined to address
the precision loss seen on multi-hop data. Promising directions

include generating more semantically coherent sub-queries,
explicitly preserving contextual dependencies between sub-
tasks so that ranking signal is not fragmented, and integrating
structured reasoning representations such as knowledge or
reasoning graphs to capture inter-step relationships that flat
sub-query lists miss.
D. Enhanced Reflection Mechanisms
Reflection in this work is a bounded retry loop. More
sophisticated designs could introduce fine-grained error de-
tection and classification, selective re-execution of only the
pipeline components implicated in a failure rather than the
whole query, and confidence-based stopping criteria that halt
refinement once marginal gains fall below a threshold. Such
targeted reflection would aim to recover its quality benefits
while shedding much of its current cost.
E. Integration with External Tools
The framework can be extended beyond retrieval and gener-
ation to incorporate additional tools—knowledge graphs, live
APIs, or symbolic reasoning engines—following the tool-use
paradigm of ReAct [12] and Toolformer [42]. In a DevOps
context this could mean querying live monitoring systems or
ticketing APIs, enabling responses grounded in the current
state of a system rather than static documentation alone.
F . Scalability and Real-World Deployment
Finally, the system should be evaluated at larger scale and
under realistic conditions, including noisy queries, much larger
corpora, and concurrent user load. Establishing robustness,
throughput, and operational cost in production settings is
necessary before agentic RAG can move from a research
prototype to a deployed assistant. Recent advances in adap-
tive retrieval and cost-aware agentic reasoning reinforce the
importance of these directions [34], [38].
REFERENCES
[1] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xu, E. Ishii, Y . J. Bang,
A. Madotto, and P. Fung, “Survey of hallucination in natural language
generation,”ACM Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023.
[2] G. Mialon, R. Dess `ı, M. Lomeli, C. Nalmpantis, R. Pasunuru,
R. Raileanu, B. Rozi `ere, N. Goyal, A. Joulin, and T. Scialom, “Aug-
mented language models: A survey,”arXiv preprint arXiv:2302.07842,
2023.
[3] Amazon Bedrock, “Building a genai-rag runbooks-based chatops assis-
tant,” DevOps.com, 2024.
[4] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” inAdvances
in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp.
9459–9474.
[5] K. Guu, K. Leeet al., “Realm: Retrieval-augmented language model
pre-training,” inICML, 2020.
[6] L. Amorimet al., “Empirical evaluation of rag patterns: Naive, fusion,
and self-rag,” inSBC, 2024.
[7] S. Feldmanet al., “Multi-hop question answering: A survey,”arXiv
preprint, 2023.
[8] Y . Gaoet al., “Retrieval-augmented generation for large language
models: A survey,”arXiv preprint arXiv:2312.10997, 2023.
[9] P. Ghadekaret al., “Agentic retrieval-augmented generation: A survey,”
arXiv preprint arXiv:2501.09136, 2025.
[10] B. Wanget al., “The evolution of retrieval-augmented generation,”
Journal of AI Research, 2024.[11] A. Asaiet al., “Self-rag: Learning to retrieve, generate, and critique
through self-reflection,” inProceedings of ICLR, 2024.
[12] S. Yaoet al., “React: Synergizing reasoning and acting in language
models,”arXiv preprint arXiv:2210.03629, 2022.
[13] Meta AI, “The llama 3 herd of models,” Meta AI, Tech. Rep., 2024.
[14] G. Gerganov, “Gguf: A new format for local llm inference,” llama.cpp
documentation, 2023.
[15] H. Trivediet al., “Musique: Multihop questions via single-hop ques-
tion composition,”Transactions of the Association for Computational
Linguistics, 2022.
[16] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
S. Ozair, A. Courville, and Y . Bengio, “Generative adversarial nets,” in
Advances in Neural Information Processing Systems, 2014, pp. 2672–
2680.
[17] J. Gui, Z. Sun, Y . Wen, D. Tao, and J. Ye, “A survey on generative
adversarial networks: Algorithms, theory, and applications,”IEEE Trans-
actions on Knowledge and Data Engineering, 2021.
[18] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askellet al., “Language models
are few-shot learners,”Advances in Neural Information Processing
Systems, vol. 33, pp. 1877–1901, 2020.
[19] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
L. Kaiser, and I. Polosukhin, “Attention is all you need,” inAdvances
in Neural Information Processing Systems, 2017, pp. 5998–6008.
[20] P. Shaw, J. Uszkoreit, and A. Vaswani, “Self-attention with relative
position representations,” inProceedings of NAACL, 2018.
[21] P. Dufter, M. Schmitt, and H. Sch ¨utze, “Position information in trans-
formers: An overview,”Computational Linguistics, vol. 48, no. 3, pp.
733–763, 2022.
[22] R. Sennrich, B. Haddow, and A. Birch, “Neural machine translation of
rare words with subword units,”arXiv preprint arXiv:1508.07909, 2015.
[23] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,”arXiv
preprint arXiv:1810.04805, 2018.
[24] A. Radford and K. Narasimhan, “Improving language understanding by
generative pre-training,” 2018.
[25] A. Graves, “Sequence transduction with recurrent neural networks,”
arXiv preprint arXiv:1211.3711, 2012.
[26] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y . Choi, “The curious case
of neural text degeneration,”arXiv preprint arXiv:1904.09751, 2019.
[27] E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell, “On
the dangers of stochastic parrots: Can language models be too big?” in
Proceedings of the 2021 ACM Conference on Fairness, Accountability,
and Transparency, 2021, pp. 610–623.
[28] E. Strubell, A. Ganesh, and A. McCallum, “Energy and policy consid-
erations for deep learning in nlp,” inProceedings of the 57th Annual
Meeting of the Association for Computational Linguistics, 2019.
[29] T. Mikolovet al., “Distributed representations of words and phrases,”
inNeurIPS, 2013.
[30] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of
word representations in vector space,”arXiv preprint arXiv:1301.3781,
2013.
[31] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using
siamese bert-networks,” inProceedings of EMNLP, 2019.
[32] V . Karpukhin, B. Oguzet al., “Dense passage retrieval for open-domain
question answering,” inEMNLP, 2020.
[33] L. Wanget al., “Query2doc: Query expansion with large language
models,” inProceedings of EMNLP, 2023.
[34] A. Maharjan and U. Yadav, “Chunking, retrieval, and re-ranking: An
empirical evaluation of rag architectures for policy document question
answering,”arXiv preprint arXiv:2601.15457, 2026.
[35] J. Johnsonet al., “Billion-scale similarity search with gpus,”IEEE
Transactions on Big Data, 2019.
[36] M. Douzeet al., “The faiss library for efficient similarity search,”arXiv
preprint arXiv:2401.08281, 2024.
[37] P. Ghadekar, “Sentence meaning similarity detector using faiss,” in
ICCUBEA, 2023.
[38] S. Mishra, S. Niroula, U. Yadav, D. Thakur, S. Gyawali, and
S. Gaire, “Sok: Agentic retrieval-augmented generation (rag): Taxon-
omy, architectures, evaluation, and research directions,”arXiv preprint
arXiv:2603.07379, 2026.
[39] L. Wanget al., “A survey of large language model based autonomous
agents,”arXiv preprint arXiv:2308.11432, 2023.
[40] A. Ng, “The agentic workflow landscape,” DeepLearning.AI, 2024.

[41] L. Yan, “Multi-agent collaboration in generative ai,” Medium, 2024.
[42] T. Schicket al., “Toolformer: Language models can teach themselves
to use tools,”arXiv preprint arXiv:2302.04761, 2023.
[43] S. Yanet al., “Corrective retrieval augmented generation,” inProceed-
ings of ICLR, 2024.
[44] S. Jeonget al., “Adaptive-rag: Learning to adapt retrieval-augmented
large language models through question complexity,” inProceedings of
NAACL, 2024.
[45] Weights and Biases, “Agentic rag: Enhancing retrieval with ai agents,”
WandB Reports, 2024.
[46] J. Huet al., “Self-correction in llms during inference: A survey,”
Transactions of the ACL (TACL), 2024.
[47] J. Weiet al., “Chain-of-thought prompting elicits reasoning in llms,” in
NeurIPS, 2022.
[48] D. Zhouet al., “Least-to-most prompting enables complex reasoning,”
arXiv preprint arXiv:2205.10625, 2023.
[49] A. Madaanet al., “Self-refine: Iterative refinement with self-feedback,”
arXiv preprint arXiv:2303.17651, 2023.
[50] Z. Yanget al., “Hotpotqa: A dataset for diverse, explainable multi-hop
question answering,” inProceedings of EMNLP, 2018.
[51] Y . Shiet al., “Reasoning in trees: Multi-hop question answering in rag,”
inProceedings of the WWW Conference, 2026.
[52] IBM Research, “Docling: An efficient open-source toolkit for ai-driven
document conversion,” inAAAI, 2025.
[53] ——, “Docling technical report v2.0,”arXiv preprint arXiv:2501.17887,
2025.
[54] BAAI, “Bge-base-en-v1.5 technical specifications,” HuggingFace, 2023.