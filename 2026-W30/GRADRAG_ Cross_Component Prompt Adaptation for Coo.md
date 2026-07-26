# GRADRAG: Cross-Component Prompt Adaptation for Coordinated Multi-Agent RAG

**Authors**: Paolo Pedinotti, Enrico Santus

**Published**: 2026-07-23 13:54:38

**PDF URL**: [https://arxiv.org/pdf/2607.21324v1](https://arxiv.org/pdf/2607.21324v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems increasingly employ multiple LLM agents. Yet, most prior work optimizes components in isolation rather than coordinating improvements across the pipeline. We introduce GRADRAG, a framework for cross-component prompt adaptation that models the RAG pipeline as a computational graph and propagates structured evaluation feedback to update upstream agents. An Evaluator critiques downstream answers and supporting evidence, producing actionable feedback that a Prompt Optimizer uses to iteratively update adaptive agents, such as retrievers, graph constructors, and answerers. The Evaluator also triggers early stopping when the output is deemed satisfactory. We evaluate GRADRAG on the SQUALITY and QMSUM benchmarks under two retrieval paradigms: flat chunk-based retrieval using IRCoT-style query refinement (Trivedi et al., 2023), and graph-based retrieval that constructs and iteratively enriches an entity-relation graph from the document. Across both settings, GRADRAG consistently outperforms one-step refinement baselines that update only the final generator, achieving a 12-15 percentage point net preference margin in LLM-judged pairwise comparisons, with most gains realized within two refinement iterations.

## Full Text


<!-- PDF content starts -->

GRADRAG: Cross-Component Prompt Adaptation
for Coordinated Multi-Agent RAG
Paolo Pedinotti*
Bloomberg
pedinotti.paolo@gmail.comEnrico Santus†
Bloomberg
esantus@bloomberg.net
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems increasingly employ multiple LLM agents.
Yet, most prior work optimizes components
in isolation rather than coordinating im-
provements across the pipeline. We intro-
duceGRADRAG, a framework forcross-
component prompt adaptationthat models
the RAG pipeline as a computational graph
and propagates structured evaluation feedback
to update upstream agents. AnEvaluatorcri-
tiques downstream answers and supporting evi-
dence, producing actionable feedback that a
Prompt Optimizeruses to iteratively update
adaptive agents (e.g., retrievers, graph con-
structors, answerers). The Evaluator also trig-
gers early stopping when the output is deemed
satisfactory. We evaluate GRADRAG on the
SQUALITY and QMSUMbenchmarks under
two retrieval paradigms: (i) flat (chunk-based)
retrieval using IRCoT-style query refinement
(Trivedi et al., 2023), and (ii) graph-based re-
trieval that constructs and iteratively enriches
an entity–relation graph from the document.
Across both settings, GRADRAG consistently
outperforms one-step refinement baselines that
update only the final generator, achieving a12–
15 percentage point net preference marginin
LLM-judged pairwise comparisons, with most
gains realized within two refinement iterations.
1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a central paradigm for enabling large lan-
guage models (LLMs) to reason over information
beyond their parametric knowledge. Early RAG
systems followed a simpleretrieve-then-generate
workflow in which a retriever selects relevant text
and a generator produces an answer conditioned on
it. More recently, RAG architectures have evolved
*Paolo Pedinotti contributed to this work during his intern-
ship at Bloomberg.
†Equal contribution.into increasinglyagenticsystems, where multiple
LLM-based components cooperate to formulate
retrieval queries, organize evidence, perform multi-
step reasoning, and evaluate or revise intermediate
outputs.
Despite these advances, most RAG pipelines
still optimize components in isolation. Refine-
ments are typically applied locally—for exam-
ple, through query rewriting, evidence filtering,
or post-hoc answer editing viaself-reflectionor
self-correction. Consequently, errors introduced
early in the pipeline, such as incomplete retrieval or
poorly structured evidence, often propagate down-
stream and limit overall performance. This re-
veals a coordination problem in multi-agent RAG
pipelines: evaluation signals produced at the fi-
nal generation stage are rarely used to improve
upstream components such as retrievers or graph
construction modules.
In this work, we study document-grounded
query answering and query-focused summariza-
tion, where retrieval operates within a given docu-
ment (or document set) by selecting relevant spans
for each query. This setting corresponds to bench-
marks such as SQUALITY and QMSUM, where
each query is paired with long documents. Al-
though these documents fit within modern LLM
context windows, our goal is not to compare RAG
with full-context inference but to use these bench-
marks as a controlled testbed for analyzing how
different RAG pipelines behave and how coordina-
tion between their components can be improved.
We introduceGRADRAG, a framework for
cross-component prompt adaptationin agentic
RAG systems (Figure 1). GRADRAG models the
RAG pipeline as a computational graph whose
nodes correspond to agents such as the retriever,
graph constructor, and answer generator. Instead of
refining only the final answer, the framework prop-
agates evaluation feedback to multiple upstream
agents responsible for retrieval and evidence con-
arXiv:2607.21324v1  [cs.CL]  23 Jul 2026

Figure 1: GRADRAG: cross-component prompt adaptation in an agentic RAG pipeline.Grey (forward pass):A
Retrieval Agent gathers evidence from either document chunks (Vector RAG) or a structured entity–relation graph
(GraphRAG, constructed by a separate agent that is not shown in the mock up), which is assembled into a context
and passed to an Answer Generation Agent to produce a candidate answer.Orange (evaluation and control):An
Evaluator reviews the answer and its supporting evidence and returns feedback with a binaryStop?decision. IfYes,
the answer is accepted. IfNo, a Prompt Optimizer updates the prompts of one or more agents (as indicated by the
evaluation), and the next forward pass is executed.
struction. At test time, anEvaluatoragent analyzes
the generated answer together with its supporting
evidence and produces structured feedback identi-
fying missing information, weak reasoning links,
or irrelevant context. APrompt Optimizercon-
verts this feedback into prompt updates for adaptive
agents across refinement iterations. The process
operates entirely through prompt updates and does
not modify model parameters.
By allowing downstream critiques to influence
upstream decisions, GRADRAG enables coor-
dinated improvements beyond one-step answer
rewriting. Rather than optimizing retrieval, struc-
turing, and generation independently, the frame-
work aligns their behavior through feedback-driven
prompt refinement within a bounded test-time bud-
get.
We evaluate GRADRAG on SQUALITY
and QMSUMusing two representative retrieval
paradigms. The first is aflat (chunk-based)
retrievalpipeline, where the system iteratively
rewrites search queries and retrieves document
chunks using a hybrid dense–lexical retriever fol-
lowing an IRCoT-style strategy (Trivedi et al.,
2023). The second is agraph-based retrieval
pipeline that extracts and iteratively enriches an
entity–relation graph to assemble evidence for an-
swering the query. Within each paradigm we com-pare GradRAG against controlled one-step refine-
ment baselines in which only the final answer
generator is updated, isolating the effect of cross-
component prompt adaptation. Across both set-
tings, GRADRAG consistently outperforms these
baselines, with most gains realized within the first
two refinement iterations. Section 5 shows that
cross-component prompt adaptation yields a12–15
percentage point net preference marginin LLM-
guided pairwise evaluations. GRADRAG therefore
provides a general framework for coordinating im-
provements across heterogeneous components in
modular RAG pipelines. This work makes the fol-
lowing contributions:
•We introduce a framework forcross-
component prompt adaptationin agentic
RAG, enabling evaluator feedback to update
the prompts of multiple upstream agents at
test time.
•We show thatcoordinated agent refinement
via shared evaluative feedback improves align-
ment between retrieval, structuring, and gen-
eration.
•We propose amodular graph-based abstrac-
tionthat applies the same feedback-driven
optimization across both flat (chunk-based)

and graph-based RAG pipelines.
2 Related Work
2.1 Agentic Approaches and RAG
Recent work in Retrieval-Augmented Generation
(RAG) increasingly incorporates agentic compo-
nents. Asai et al. (2023) introduce areflection-
based mechanismin which the model produces
intermediate tokens representing retrieval deci-
sions and evidence assessments, enabling itera-
tive retrieval and answer refinement guided by a
learned critic. This work established that RAG
performance can benefit from multi-step, agentic
decision-making.
Subsequent approaches distribute retrieval and
reasoning across specialized components. For ex-
ample, Jeong et al. (2024) and Lee et al. (2025)
propose routing agents that decide whether and
how extensively to retrieve and which backend to
query. Trivedi et al. (2023) frame retrieval as a
sequential decision process that iteratively refines
search queries, while Jiang et al. (2025) use Monte
Carlo Tree Search guided by learned reward mod-
els. In this literature,multi-agenttypically refers to
pipelines composed of multiple LLM calls with dis-
tinct prompts, each specializing in a subtask such
as retrieval routing, evidence assessment, or answer
synthesis.
Multi-agent retrieval optimization has become a
common strategy. Yan et al. (2024) train a rel-
evance judge that decides whether to continue
querying or switch to generation, while Chang
et al. (2025) generate partial answers from retrieved
documents and use a judge to select the most in-
formative ones. Despite their success, these ap-
proaches apply feedback locally, focusing on in-
dividual stages such as retrieval control or answer
selection. Consequently, evaluation signals gen-
erated at the final answer stage rarely influence
upstream components such as retrieval or evidence
construction. In contrast, our work addresses this
coordination problem directly: how feedback on
downstream answer quality can be used to adapt
multiple upstream components of a RAG pipeline
in a unified manner.
2.2 LLMs as Critic Agents and Prompt
Optimizers
Thereflectionpattern can be viewed as a three-
stage loop of generation, evaluation, and adaptation.
Early systems (Madaan et al., 2023; Shinn et al.,2023) demonstrated inference-time refinement in
which models iteratively revise outputs using self-
or evaluator-generated critiques (later described
astextual gradientsby Yuksekgonul et al. (2024),
orverbal reinforcement). These approaches treat
natural-language critiques as optimization signals
guiding subsequent reasoning steps.
This idea was later generalized intoautomatic
prompt optimization, where an LLM searches
for prompts that improve downstream performance
(Zhou et al., 2023; Opsahl-Ong et al., 2024).
Frameworks such as DSPy (Khattab et al., 2023)
and TextGrad (Yuksekgonul et al., 2024) pro-
vide programming abstractions for this process:
DSPy focuses on optimizing prompting structure
and demonstrations, while TextGrad formalizes
critique-driven prompt updates over computational
graphs.
Our work builds on these foundations but fo-
cuses specifically on coordinating prompt adapta-
tion across multiple components of a RAG pipeline.
GRADRAG instantiates critique-driven prompt op-
timization within agentic RAG systems and empir-
ically evaluates how coordinating updates across
retrieval, structuring, and generation components
improves retrieval-augmented reasoning over long
documents.
3 Method: GRADRAG
GRADRAGis a test-time framework for coordinat-
ing multiple agents in Retrieval-Augmented Gener-
ation (RAG) pipelines throughcross-component
prompt adaptation. GRADRAG builds on
critique-driven prompt optimization and reflection-
based approaches, instantiating them in agentic
RAG systems where feedback on downstream an-
swer quality is used to adapt multiple upstream
components in a unified manner.
Overview.As illustrated in Figure 1, GRADRAG
executes a RAG pipeline in iterative refinement cy-
cles. Each cycle consists of a forward pass that
produces intermediate artifacts (e.g., retrieved text
chunks or extracted graph structures) and a final
answer, followed by an evaluation step. AnEvalu-
atoragent reviews the answer together with its sup-
porting evidence and produces structured natural-
language feedback. If refinement is required, a
Prompt Optimizerconverts this feedback into up-
dated prompts for one or more adaptive agents. The
next cycle then runs with the updated prompts. This
process operates entirely through prompt updates

and does not modify model parameters. The full
refinement loop is summarized in Algorithm 1.
Computational graph abstraction.GRADRAG
models the RAG workflow as a computational
graph whose nodes represent agents, programs, or
data artifacts, and whose edges represent compu-
tational transformations. Nodes corresponding to
LLM-based agents are associated with prompts,
which may be either fixed or adaptive. In our frame-
work,adaptive agentsare those whose prompts
can be updated across refinement cycles based on
evaluator feedback. This abstraction allows hetero-
geneous components — such as classical retrievers
and LLM-based agents — to coexist within the
same pipeline.
Cross-component prompt adaptation.Unlike
one-step reflection approaches that refine only the
final answer generator, GRADRAG allows evalu-
ator feedback to update the prompts of multiple
upstream agents. For example, feedback indicating
missing causal or explanatory information may trig-
ger changes to both the retrieval strategy (to target
more relevant evidence) and the answer generation
style (to emphasize causal structure). Prompt up-
dates are applied iteratively within a bounded test-
time budget, enabling coordinated improvements
across components.
Early stopping.Following prior reflection-based
architectures (Madaan et al., 2023; Shinn et al.,
2023), GRADRAG employs an early-stopping
mechanism. After each evaluation step, the Evalu-
ator returns a discrete decision indicating whether
the current output is satisfactory. Refinement cy-
cles stop when the Evaluator signals satisfaction or
when a maximum number of iterations is reached
(Algorithm 1, lines 6–8).
3.1 Vector RAG Technical Details
For flat retrieval, we implement a document-
grounded Vector RAG pipeline in which the in-
put document is segmented into overlapping text
chunks. A Retrieval Agent follows an IRCoT-style
strategy (Trivedi et al., 2023), iteratively generat-
ing sub-queries based on the context retrieved so
far. Each sub-query triggers a hybrid retrieval step
that combines dense similarity search and lexical
matching to select relevant chunks.
Within GRADRAG, the Retrieval Agent is adap-
tive: its prompt may be updated across refinement
cycles based on evaluator feedback. For instance,Algorithm 1GRADRAG: Cross-Component
Prompt Adaptation
Require: Query q, document(s) D, initial prompts {pa}for
agentsa∈ A, max iterationsT
1:fort= 1toTdo
2:Forward pass:
3: Execute RAG pipeline with current prompts{p a}:
4: – generate intermediate artifacts (e.g., retrieved
chunks or extracted entity–relation graphs)
5: – assemble supporting evidence (e.g., via chunk
aggregation or graph community summaries)
6: – produce answery t
7:Evaluation:
8: Evaluator reviews (yt,supporting evidence) and re-
turns feedbackf tand decisiond t
9:ifd t=SATISFACTORYthen
10:returny t
11:end if
12:Prompt adaptation:
13:foreach adaptive agenta∈ A adaptdo
14: Update promptp a←Optimize(p a, ft)
15:end for
16:end for
17:returnfinal answery T
critiques indicating missing explanations or tem-
poral relationships can lead the agent to generate
more targeted sub-queries in subsequent iterations.
Retrieved chunks are aggregated to form the con-
text passed to the Answer Generation Agent, as
specified in the forward pass of Algorithm 1.
3.2 GraphRAG Technical Details
Our graph-based RAG pipeline builds on prior
graph-oriented approaches (Edge et al., 2025;
Pedinotti and Santus, 2026). The document is seg-
mented into larger text spans, from which a Graph
Extraction Agent identifies entities and relations.
Extracted entities are merged into a consolidated
graph, and community detection is performed using
the Leiden algorithm to identify groups of related
entities. For each community, the system generates
a natural-language summary and a set of candidate
answers, each annotated with a helpfulness score
indicating relevance to the query. Only answers
with positive scores are retained and concatenated
to form the evidence context supplied to the An-
swer Generation Agent.
As in the vector-based setting, both the Graph
Extraction Agent and the Answer Generation
Agent are adaptive. Evaluator feedback highlight-
ing missing entities, relations, or explanatory struc-
ture is used to refine the extraction prompt, guiding
subsequent graph enrichment toward information
that is most useful for answering the query. Over
successive refinement cycles, this feedback-driven

adaptation effectively induces a task-oriented ontol-
ogy, shaping which types of entities and relations
are prioritized during graph construction (Pedinotti
and Santus, 2026; Pedinotti et al., 2026). The re-
sulting enriched graph is then used to assemble
evidence for the final answer in the next forward
pass (Algorithm 1, lines 2–5).
Implementation.All pipelines are implemented
using the Microsoft AutoGen framework (Wu et al.,
2023). The prompts are provided in Appendices A
and B to support reproducibility.
4 Experimental Setup
Research Questions.We study whether coordi-
nating prompt adaptation across multiple agents
improves RAG pipelines. Specifically, we ask:
(1) Doescross-component prompt adaptation—
where evaluator feedback updates upstream agents
such as retrievers or graph extractors — improve
performance compared toone-step refinement
that updates only the final answer generator? (2)
Are these improvements consistent across different
RAG architectures, including flat (chunk-based)
and graph-based retrieval pipelines?
GRADRAG Systems and Baselines.To answer
these questions, we evaluate GRADRAG under two
retrieval paradigms:flat (chunk-based) retrieval
andgraph-based retrieval. For each paradigm,
we compare aone-step refinement baseline, in
which feedback updates only the answer generator,
against afull GRADRAGvariant that propagates
feedback to multiple agents. The systems are:
•IRCoT (Trivedi et al., 2023) + One-Step
Refinement:A vector-based RAG pipeline
in which only the Answer Generation Agent
is adaptive. The Retrieval Agent uses a fixed
prompt, so improvements arise solely from
iterative answer rewriting.
•Full GRADRAG Vector:The same pipeline,
but evaluator feedback updates both the An-
swer Generation Agent and the Retrieval
Agent, enabling refinement of sub-query gen-
eration and evidence selection.
•GraphRAG (Edge et al., 2025) + One-Step
Refinement:A graph-based RAG pipeline
where only the Answer Generation Agent is
adaptive. The Graph Extraction Agent uses
a fixed prompt, so improvements arise only
from refining the final answer.Dataset Docs Questions Avg Doc Length Avg Resp Length
SQUALITY 52 260 4,995 words 241 words
QMSUM35 281 10,668 words 65 words
Table 1: Statistics about the test sets.
•Full GRADRAG Graph:The same pipeline,
but evaluator feedback updates both the An-
swer Generation Agent and the Graph Extrac-
tion Agent, enabling adaptive graph enrich-
ment.
This setup forms a controlled ablation: within
each retrieval paradigm, the baseline and GradRAG
variants share the same architecture, model, and re-
trieval process, differing only in whether evaluator
feedback updates upstream agents. This isolates
the effect of cross-component prompt adaptation.
Task.We evaluate all systems on theQuery-
Focused Summarization(QFS) task. In QFS, the
model receives a document (or document set) d
and a query q, and must produce a summary that
answers the query using information drawn from
the document. The task lies at the intersection of
abstractive summarization and question answering,
requiring both global document understanding and
targeted evidence selection.
Datasets.We conduct experiments on two stan-
dard QFS datasets:SQUALITY(Wang et al.,
2022) andQMSUM(Zhong et al., 2021). These
datasets represent complementary domains: nar-
rative texts (SQUALITY) and long meeting tran-
scripts (QMSUM). Both datasets are also included
in the ZeroSCROLLS (Shaham et al., 2023) and
LongBench (Bai et al., 2024) benchmarks. Al-
though these documents fit within the context win-
dows of modern LLMs, we use these datasets as
controlled benchmarks for studying the behavior
of RAG pipelines and evaluating coordination be-
tween their components. Table 1 summarizes their
main characteristics.
Experimental Protocol.GRADRAG operates
in atest-time refinementsetting. For each test
instance (query–document pair), the system exe-
cutes an iterative loop consisting of a forward pass
followed by evaluation and, if necessary, prompt
adaptation. During refinement, anEvaluator Agent
analyzes the generated answer together with its sup-
porting evidence and produces structured feedback
identifying missing information or reasoning gaps.
Importantly, the Evaluator doesnothave access

to the reference answer, preventing data leakage
during optimization.
If the Evaluator determines that the answer is
unsatisfactory, its critique is used to update the
prompts of the adaptive agents, and another refine-
ment cycle is executed. Otherwise, the process ter-
minates through anearly-stoppingdecision. Each
test instance is processed independently and opti-
mized prompts are reset after each example, en-
suring that no information is carried across test
instances.
This protocol enables a controlled comparison
between one-step refinement and cross-component
adaptation by isolating the effect of prompt updates
within a single query. While this per-instance re-
finement increases computational cost, it avoids
cross-example contamination. In deployment sce-
narios, efficiency could be improved by batching
multiple instances before updating prompts, which
we leave for future work.
Evaluation Metrics.Traditional evaluation met-
rics for query-focused summarization, such as
ROUGE and BERTScore, correlate poorly with
human judgments because they primarily measure
lexical or embedding overlap rather than higher-
level properties such as coherence, factuality, and
reasoning (Kryscinski et al., 2019; Liu et al., 2023;
Nguyen et al., 2024).
During refinement, GRADRAG uses an inter-
nalEvaluator Agentthat analyzes generated an-
swers together with their supporting evidence and
produces critiques guiding prompt updates. This
evaluator operates without access to gold reference
answers to prevent data leakage. For final system
comparison, we instead adopt anLLM-as-a-Judge
evaluation paradigm in which an external LLM
compares system outputs using a structured rubric.
Given two candidate answers for the same input,
the judge selects the better response based on over-
all answer quality and use of supporting evidence.
We reportwin rates, defined as the proportion of
pairwise comparisons in which one system is pre-
ferred over the other (Liu et al., 2025).
To improve evaluation robustness, we follow rec-
ommendations from recent work on LLM-based
evaluation (Baumann et al., 2025). Specifically,
we reportposition consistency(Shi et al., 2025),
which measures how often the judge’s preference
remains unchanged when the order of candidate
answers is reversed. We also apply a two-sided
binomial sign testto assess statistical significance.We also conducthuman validationon a subset
of examples. Annotators are shown the same pair-
wise outputs and asked to select the better response
using the same rubric. Agreement between LLM
and human preferences is measured using accuracy
and Cohen’sκcomputed on majority-vote labels.
Finally, to analyze refinement dynamics (Sec-
tion 5), we track several per-iteration diagnostics:
average response length,lexical density(content-
word ratio),ROUGE-1between the system re-
sponse and the gold reference answer, andtopic
entropy, defined as the Shannon entropy of topic
distributions obtained by fitting LDA to the set of
system responses at each iteration.
Implementation Details and Hyperparameters.
Forflat (chunk-based) retrieval, documents are
segmented into overlapping text chunks using fixed
character lengths (default 400 characters with a
40-character overlap). Retrieval follows an IRCoT-
style strategy (Trivedi et al., 2023), in which the
Retrieval Agent iteratively generates sub-queries
based on the context retrieved so far. Each sub-
query triggers a retrieval step, where the Retrieval
Agent selects the retrieval strategy to apply: dense
similarity search via FAISS (Douze et al., 2025) or
lexical matching via BM25 (Robertson et al., 1995).
We perform three retrieval iterations and select the
top retrieved chunks at each step, aggregating them
to form the final context passed to the Answer Gen-
eration Agent. The choice of retrieval strategy is
controlled by the agent’s prompt and may change
across refinement iterations as a result of evaluator
feedback.
Forgraph-based retrieval, documents are seg-
mented into larger overlapping spans (default 1000
characters with a 200-character overlap) to sup-
port more reliable entity and relation extraction.
A Graph Extraction Agent identifies entities and
relations from these spans, which are aggregated
into a graph structure that is incrementally enriched
across refinement iterations. The resulting graph is
then used to assemble evidence for answer genera-
tion. Across all systems, the maximum number of
refinement iterations is capped at three.
Models.All generation agents (Retriever, Graph
Extractor, and Answer Generator) useGemini-2.5-
Flash(Comanici et al., 2025) (June 2025 version)
via the Google Generative Language API.1Repet-
itive subtasks—such as sub-query generation in
1https://ai.google.dev/gemini-api/docs/models

flat retrieval, entity–relation extraction in graph re-
trieval, and community-level summarization and
candidate-answer generation—use the more cost-
efficientGemini-2.5-Flash-Lite(July 2025 ver-
sion) through the same API. For vector indexing,
we use thetext-embedding-3-smallembedding
model.2For evaluation, we use the openDeepSeek-
V3.1model (DeepSeek-AI, 2025) as an external
LLM judge. The prompts used by the system and
the evaluator are provided in Appendices A and B
to support reproducibility.
Computational Cost.We analyze the additional
overhead introduced by cross-component prompt
adaptation using the GraphRAG pipeline on a sam-
ple of 13 query–document pairs. A full refinement
cycle requires on average 35s per query compared
to 30.5s for the one-step baseline, corresponding
to roughly a 10% increase. Most runtime (about
70%) is spent in the forward pipeline—graph ex-
traction and answer generation—which is identical
across systems. Token usage increases from 6,040
tokens per query in the baseline to 8,140 tokens in
GRADRAG, primarily due to critique processing
during prompt updates.
Retrieval Paradigm SQUALITY QMSUM
Flat (Vector RAG)
Full GRADRAG win rate (%) 56.5 56.0
Position consistency (%) 79.3 55.1
p-value 0.028 0.090
Graph-based RAG
Full GRADRAG win rate (%) 57.5 55.8
Position consistency (%) 82.5 87.9
p-value 0.005 0.020
Table 2: Pairwise evaluation results. Values report the
percentagewin rateof GRADRAG against the corre-
sponding one-step baseline.
2https://platform.openai.com/docs/guides/
embeddingsMetric SQUALITY QMSUM
Full GRADRAG win rate (%) 58.5 61.5
Position consistency (%) 79.3 59.7
p-value 0.0038 0.0007
Table 3: Effect of an additional refinement iteration for
flat retrieval.
Method SQUALITY QMSUM
Flat Retrieval
IRCoT + One-step refinement 1.82 1.60
Full GRADRAG Vector 1.84 1.63
Graph Retrieval
GraphRAG + One-step refine-
ment1.83 1.63
Full GRADRAG Graph 1.72 1.68
Table 4: Average number of refinement iterations before
early stopping.
5 Results
Human Validation.Human evaluation results
show substantial agreement between LLM-based
judgments and human preferences. Across the an-
notated subset, the LLM achieves a 75.86% agree-
ment rate with human evaluators, a Cohen’s κof
0.59, and an F1 score of 0.8. Human preferences
exhibit moderate variability, with an average anno-
tation entropy of 0.642 , indicating partial disagree-
ment on more challenging examples, particularly
those involving longer responses.
Main Results.Table 2 reports pairwise compar-
ison results between one-step baselines and full
GRADRAG variants. Comparisons are performed
within each retrieval paradigm (flat vs. graph-
based), isolating the effect of cross-component
prompt adaptation from differences in retrieval ar-
chitecture.
Across both datasets and retrieval paradigms,
full GRADRAG variants consistently outperform
their one-step counterparts. Win rates range be-
(a) Length
 (b) Lexical density
 (c) ROUGE-1
 (d) Topic entropy
Figure 2: Evolution of system responses across iterations.

Retrieval Paradigm SQUALITY QMSUM
Iteration 2
Flat (Vector) GradRAG win
rate (%)54.0 54.6
Graph-based GradRAG win
rate (%)53.6 55.6
Iteration 3
Flat (Vector) GradRAG win
rate (%)56.5 56.0
Graph-based GradRAG win
rate (%)57.5 55.8
Table 5: Pairwise preference rates for GRADRAG over
one-step baselines when the number of refinement itera-
tions is fixed to a given value (without early stopping).
tween 55% and57%, corresponding to anet pref-
erence marginof approximately 12–15percent-
age points when compared with the corresponding
baseline preferences (e.g., 56% vs.44%). These
results indicate that allowing evaluator feedback to
update multiple upstream agents yields consistent
improvements over refining only the final answer
generator.
Evaluation robustness diagnostics are reported
alongside preferences (Section 4). Three of the
four comparisons yield statistically significant dif-
ferences at the p <0.05 level. The remaining
case—flat retrieval on QMSUM—shows marginal
significance, reflecting greater variability in evalua-
tor preferences for this setting.
To further investigate this case, we run an addi-
tional refinement iteration for flat retrieval on QM-
SUM. As shown in Table 3, the additional iteration
increases the performance gap and yields statisti-
cally significant results, suggesting that the earlier
marginal outcome was likely due to insufficient
refinement rather than instability in the evaluation
procedure.
Effect of Refinement Budget.Table 4 reports
the average number of refinement iterations before
early stopping. Across systems, refinement typi-
cally terminates after fewer than two iterations, in-
dicating that satisfactory outputs are often reached
well before the maximum iteration budget.
Table 5 reports pairwise preference rates as
a function of a fixed number of refinement it-
erations, evaluated without early stopping. For
flat (chunk-based) retrieval, performance gener-
ally improves as additional refinement iterations
are permitted, with win rates increasing from it-eration 2 to iteration 3 on both datasets. Graph-
based GRADRAG shows a similar pattern, achiev-
ing preference rates above 50% at both iteration
limits across datasets. While gains are not strictly
monotonic in all cases—most notably on QMSUM,
where the win rate slightly decreases from itera-
tion 2 to iteration 3—the graph-based variant re-
mains consistently preferred over the one-step base-
line.
Overall, these results suggest that both flat and
graph-based pipelines benefit from additional re-
finement under fixed budgets, while early stopping
helps balance refinement gains against potential
over-specialization in structured retrieval settings.
Response Evolution Across Iterations.Figure 2
illustrates how system outputs evolve across re-
finement iterations. Response length increases
sharply after the first refinement and then stabilizes,
while lexical density increases steadily across iter-
ations. ROUGE-1 scores improve monotonically,
and topic entropy decreases over time, indicating
progressively more focused responses.
Taken together, these trends suggest that refine-
ment primarily improves responses by increasing
informational density and topical focus rather than
by simply adding more content. This pattern is con-
sistent across retrieval paradigms and supports the
interpretation that cross-component prompt adapta-
tion produces more targeted and coherent outputs.
6 Conclusion
We introducedGRADRAG, a framework for coor-
dinating multiple agents in Retrieval-Augmented
Generation (RAG) pipelines through cross-
component prompt adaptation. By modeling RAG
systems as computational graphs and propagating
evaluation feedback across agents, GRADRAG en-
ables coordinated refinement of retrieval, evidence
construction, and answer generation at test time.
Across both vector- and graph-based retrieval
settings, GRADRAG achieves a12–15 percent-
age point net preference marginin pairwise
comparisons. These results indicate that allowing
evaluation feedback to influence upstream com-
ponents—rather than refining only the final an-
swer—consistently improves system outputs. More
broadly, our findings highlight the importance of
coordinating adaptation across heterogeneous com-
ponents in modular RAG pipelines.

7 Limitations
While GRADRAG introduces a general framework
for agentic optimization in RAG systems, several
aspects offer opportunities for further refinement:
•Computational cost.The approach is compu-
tationally intensive, as both the forward exe-
cution and refinement phases rely on LLM
generation. However, this cost scales pre-
dictably with the number of adaptive agents
and remains manageable for moderate config-
urations. Moreover, efficiency gains can be
achieved through parallelization.
•Evaluation vs. deployment gap.Our evalu-
ation protocol — where each refinement cy-
cle is triggered after a single forward execu-
tion — was designed to ensure data isolation
and interpretability. While this setup is less
reflective of deployment conditions, it pro-
vides a clean benchmark for studying agentic
feedback dynamics. In contrast, recent ap-
proaches to automatic prompt optimization,
such as DSPy (Khattab et al., 2023), derive
optimized prompts from multiple examples.
Future research may extend GRADRAG to
batch- or continual-feedback regimes to better
approximate real-world usage.
•Evaluator configuration.The Evaluator
Agent operates using general summarization
principles (e.g., coherence, fluency) without
external reward signals. This design iso-
lates the intrinsic capability of LLMs to self-
evaluate while avoiding data leakage. Future
implementations could integrate additional su-
pervision signals, such as reward models or
heuristic metrics, to strengthen evaluation fi-
delity.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
Preprint, arXiv:2310.11511.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. LongBench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), pages 3119–3137, Bangkok, Thailand.
Association for Computational Linguistics.Joachim Baumann, Paul Röttger, Aleksandra Urman,
Albert Wendsjö, Flor Miriam Plaza del Arco, Jo-
hannes B. Gruber, and Dirk Hovy. 2025. Large
Language Model Hacking: Quantifying the Hidden
Risks of Using LLMs for Text Annotation.Preprint,
arXiv:2509.08825.
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Ma-
hashweta Das, and Na Zou. 2025. MAIN-RAG:
Multi-agent filtering retrieval-augmented generation.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 2607–2622, Vienna, Austria.
Association for Computational Linguistics.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
and 1 others. 2025. Gemini 2.5: Pushing the Fron-
tier with Advanced Reasoning, Multimodality, Long
Context, and Next Generation Agentic Capabilities.
Preprint, arXiv:2507.06261.
DeepSeek-AI. 2025. DeepSeek-V3 Technical Report.
Preprint, arXiv:2412.19437.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré,
Maria Lomeli, Lucas Hosseini, and Hervé Jégou.
2025. The faiss library.Preprint, arXiv:2401.08281.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2025. From local to global: A
graph rag approach to query-focused summarization.
Preprint, arXiv:2404.16130.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-RAG: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), pages 7036–7050, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shi-
jie Wang, Xin Zhao, Yang Song, and Tao Zhang.
2025. RAG-star: Enhancing deliberative reasoning
with retrieval augmented verification and refinement.
InProceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers), pages 7064–7074,
Albuquerque, New Mexico. Association for Compu-
tational Linguistics.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari,
Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T.
Joshi, Hanna Moazam, Heather Miller, Matei Za-
haria, and Christopher Potts. 2023. Dspy: Compiling
declarative language model calls into self-improving
pipelines.Preprint, arXiv:2310.03714.

Wojciech Kryscinski, Nitish Shirish Keskar, Bryan Mc-
Cann, Caiming Xiong, and Richard Socher. 2019.
Neural text summarization: A critical evaluation. In
Proceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pages 540–551, Hong
Kong, China. Association for Computational Linguis-
tics.
Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen
Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa
Rangwala, and Christos Faloutsos. 2025. HybGRAG:
Hybrid retrieval-augmented generation on textual and
relational knowledge bases. InProceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
879–893, Vienna, Austria. Association for Computa-
tional Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
pages 2511–2522, Singapore. Association for Com-
putational Linguistics.
Yinhong Liu, Han Zhou, Zhijiang Guo, Ehsan Shareghi,
Ivan Vuli ´c, Anna Korhonen, and Nigel Collier. 2025.
Aligning with human judgement: The role of pair-
wise preference in large language model evaluators.
Preprint, arXiv:2403.16950.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdan-
bakhsh, and Peter Clark. 2023. Self-refine: iterative
refinement with self-feedback. InProceedings of the
37th International Conference on Neural Information
Processing Systems, NIPS ’23, Red Hook, NY , USA.
Curran Associates Inc.
Huyen Nguyen, Haihua Chen, Lavanya Pobbathi, and
Junhua Ding. 2024. A comparative study of quality
evaluation methods for text summarization.Preprint,
arXiv:2407.00747.
Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David
Broman, Christopher Potts, Matei Zaharia, and Omar
Khattab. 2024. Optimizing instructions and demon-
strations for multi-stage language model programs.
InProceedings of the 2024 Conference on Empiri-
cal Methods in Natural Language Processing, pages
9340–9366, Miami, Florida, USA. Association for
Computational Linguistics.
Paolo Pedinotti, Peter Baumann, Nathan Jessurun,
Leslie Barrett, and Enrico Santus. 2026. MetaGraph:
A large-scale meta-analysis of GenAI in financial
NLP (2022–2025). InProceedings of the Fifth Work-
shop on Generation, Evaluation and Metrics (GEM),
pages 848–861, San Diego, California, USA. Associ-
ation for Computational Linguistics.Paolo Pedinotti and Enrico Santus. 2026. StructSurvey:
Structured agentic retrieval for automated survey pa-
per generation. InProceedings of the First Workshop
on Structured Understanding, Retrieval, and Gen-
eration in the LLM Era (SURGeLLM 2026), pages
162–181, San Diego, California, United States. Asso-
ciation for Computational Linguistics.
Stephen Robertson, S. Walker, S. Jones, M. M. Hancock-
Beaulieu, and M. Gatford. 1995. Okapi at trec-3.
InOverview of the Third Text REtrieval Conference
(TREC-3), pages 109–126. Gaithersburg, MD: NIST.
Uri Shaham, Maor Ivgi, Avia Efrat, Jonathan Berant,
and Omer Levy. 2023. ZeroSCROLLS: A zero-shot
benchmark for long text understanding. InFind-
ings of the Association for Computational Linguis-
tics: EMNLP 2023, pages 7977–7989, Singapore.
Association for Computational Linguistics.
Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, We-
icheng Ma, and Soroush V osoughi. 2025. Judging
the Judges: A Systematic Study of Position Bias in
LLM-as-a-Judge.Preprint, arXiv:2406.07791.
Noah Shinn, Federico Cassano, Ashwin Gopinath,
Karthik Narasimhan, and Shunyu Yao. 2023. Re-
flexion: language agents with verbal reinforcement
learning. InProceedings of the 37th International
Conference on Neural Information Processing Sys-
tems, NIPS ’23, Red Hook, NY , USA. Curran Asso-
ciates Inc.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Alex Wang, Richard Yuanzhe Pang, Angelica Chen, Ja-
son Phang, and Samuel R. Bowman. 2022. SQuAL-
ITY: Building a long-document summarization
dataset the hard way. InProceedings of the 2022 Con-
ference on Empirical Methods in Natural Language
Processing, pages 1139–1156, Abu Dhabi, United
Arab Emirates. Association for Computational Lin-
guistics.
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu,
Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang,
Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadal-
lah, Ryen W White, Doug Burger, and Chi Wang.
2023. AutoGen: Enabling Next-Gen LLM Appli-
cations via Multi-Agent Conversation.Preprint,
arXiv:2308.08155.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
Preprint, arXiv:2401.15884.
Mert Yuksekgonul, Federico Bianchi, Joseph Boen,
Sheng Liu, Zhi Huang, Carlos Guestrin, and James
Zou. 2024. Textgrad: Automatic "differentiation" via
text.Preprint, arXiv:2406.07496.

Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia
Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli
Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir
Radev. 2021. QMSum: A new benchmark for query-
based multi-domain meeting summarization. InPro-
ceedings of the 2021 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, pages
5905–5921, Online. Association for Computational
Linguistics.
Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han,
Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy
Ba. 2023. Large language models are human-level
prompt engineers.Preprint, arXiv:2211.01910.
A Prompts for Evaluator Agents
A.1 Flat Retrieval
Response evaluator agent
You are an expert judge evaluating both the
retrieved context and the generated answer
for completeness, coherence, and overall
quality.
Your task:
1. Context Evaluation:
- First, imagine an ideal, fully satisfying
answer to the question - one that
captures all key facts, causal and
temporal connections, character
motivations, outcomes, and thematic
meaning.
- Then compare this ideal answer with the
retrieved context. Identify what
information, links, or nuances are
missing, underdeveloped, ambiguous, or
irrelevant.
- Note any missing elements that would
prevent a reader from forming a complete
understanding or that weaken the logical
or emotional flow of the narrative.
2. Answer Evaluation:
- Assess whether the answer uses the
available context effectively.
- Evaluate clarity, coherence, factual
accuracy, completeness, and narrative
flow.
- Check if it connects events and entities
logically, avoids contradictions, and
reads fluently in tone and length.
Decision:
- SATISFACTORY: The context includes all major
information and connections needed for an
ideal answer, and the answer expresses them
clearly and coherently.
- NEEDS_REFINEMENT: Any important fact,
relation, or narrative link is missing,
unclear, or the answer lacks fluency,
structure, or accuracy.
Critique must include:
1. What an ideal answer would contain that is
not fully supported by the context.
2. Which parts of the context are irrelevant or
insufficient.3. How the answer could improve in factual
precision, logical or causal connections,
completeness, and stylistic quality.
Output format:
DECISION: [SATISFACTORY or NEEDS_REFINEMENT]
CRITIQUE: [detailed explanation]
The question was the following:
{original_query}
The answer of the system was:
{generated_answer}
When generating the answer, the system had
access to the following context:
{retrieved_context}
Answer Generation Agent critique prompt
You are evaluating the prompt used for answer
generation in a RAG system.
Your goal is to evaluate the prompt and propose
actionable improvements based on feedback
obtained from the system's output.
The feedback contains strategies to improve
both the retrieved content and the style of
the answer; focus only on the style.
Current answer generation prompt:
{current_prompt}
Based on this information, determine if there
is a problem with the answer generation
prompt that needs to be fixed.
Provide a detailed critique to the prompt
reporting what you learned in the previous
critique. The critique must be actionable:
it must include concrete actions that would
likely improve the output.
If false, leave the critique empty.
Answer generation prompt optimizer prompt
You are optimizing a prompt for answer
generation in a RAG system. The prompt must
be general, but it also has to be as
adherent as possible to the following
critique.
The current prompt is:
{current_prompt}
The current critique of the answer generation
process is:
{answer_generation_critique}
Based on this critique, generate a new prompt
that will be used to instruct the LLM how
to better generate answers from retrieved
vector information. The prompt should
incorporate the feedback to improve answer
quality, relevance, and coherence.
Provide only the optimized prompt without
additional commentary.
"""

Retrieval agent critique prompt
You are evaluating the queries made by an
iterative content retriever in a RAG system.
Your goal is to provide an accurate evaluation
of the queries based on the following
feedback.
The original question of the user was:
{question}
QUERIES MADE:
{queries_formatted}
RESPONSE EVALUATOR FEEDBACK:
{response_evaluator_critique}
The previous critique for the answer generator
was:
{answer_generator_critique}
Based on the response evaluator's feedback,
provide:
1. TYPES OF QUERIES TO FOCUS ON:
- What types of queries would address the
issues mentioned in the feedback?
- What aspects should future queries target
to improve the answer?
- Remember that each query must target a
specific part of the text. A good query
targets specific content that can be
found in a single chunk.
Your task is not to rate the queries, but to
identify what types of new queries would
best address the issues raised in the
feedback.
Relate your suggestions directly to the user
original question and the retrieved
content, specifying what missing details or
narrative connections new queries should
aim to uncover.
Provide a clear, actionable analysis.
Retrieval agent prompt optimizer
You are optimizing a prompt for a query agent
in a vector RAG system.
The current prompt is:
{current_prompt}
IMPORTANT CLARIFICATION:
- The planner you are optimizing asks the model
to generate a sub-query to answer the
question, based on the context retrieved so
far and the previous sub-queries.
- The planner does NOT control the
retrieval/embedding mechanism itself
- Your instructions should focus on which
aspects a sub-query must target
You will receive:
1. The original question to answer
2. The sub-queries that were made
3. A critique of the retrieval plan identifying
what worked and what needs to be improved.
{original_question}{previous_queries}
{retrieval_plan_critique}
Based on this information, generate an
optimized prompt that instructs the query
agent to create a good sub-query.
Provide only the optimized prompt as
instructions for the query planner, without
additional commentary.
Generate a full optimized prompt (not partial
additions) that will replace the current
one used by the query agent.
A.2 Graph Retrieval
Response evaluator agent
You are an expert judge evaluating both the
retrieved context and the generated answer
for completeness, coherence, and overall
quality.
Your task:
1. Context Evaluation:
- First, imagine an ideal, fully satisfying
answer to the question - one that
captures all key facts, causal and
temporal connections, character
motivations, outcomes, and thematic
meaning.
- Then compare this ideal answer with the
retrieved context. Identify what
information, links, or nuances are
missing, underdeveloped, ambiguous, or
irrelevant.
- Note any missing elements that would
prevent a reader from forming a complete
understanding or that weaken the logical
or emotional flow of the narrative.
2. Answer Evaluation:
- Assess whether the answer uses the
available context effectively.
- Evaluate clarity, coherence, factual
accuracy, completeness, and narrative
flow.
- Check if it connects events and entities
logically, avoids contradictions, and
reads fluently in tone and length.
Decision:
- SATISFACTORY: The context includes all major
information and connections needed for an
ideal answer, and the answer expresses them
clearly and coherently.
- NEEDS_REFINEMENT: Any important fact,
relation, or narrative link is missing,
unclear, or the answer lacks fluency,
structure, or accuracy.
Critique must include:
1. What an ideal answer would contain that is
not fully supported by the context.
2. Which parts of the context are irrelevant or
insufficient.

3. How the answer could improve in factual
precision, logical or causal connections,
completeness, and stylistic quality.
Output format:
DECISION: [SATISFACTORY or NEEDS_REFINEMENT]
CRITIQUE: [detailed explanation]
The question was the following:
{original_query}
The answer of the system was:
{generated_answer}
When generating the answer, the system had
access to the following context:
{retrieved_context}
Answer generation critique prompt
You are evaluating the prompt used for answer
generation in a GraphRAG system.
Your goal is to evaluate the prompt and propose
actionable improvements based on feedback
obtained from the system's output.
The feedback contains strategies to improve
both the retrieved content and the style of
the answer; FOCUS ONLY ON THE STYLE.
Current answer generation prompt:
{current_prompt}
Feedback from the response evaluation:
{response_evaluator_output}
Based on this information, determine if there
is a problem with the answer generation
prompt that needs to be fixed.
First, provide your reasoning explaining why
there is or isn't a problem.
Then, provide a critique focusing on the
specific issue.
Answer generation prompt optimizer
You are optimizing a prompt for answer
generation in a GraphRAG system.
The current critique of the answer generation
process is:
{answer_generation_critique}
Based on this critique, generate a new prompt
that will be used to instruct the LLM how
to better generate answers from retrieved
graph information. The prompt should
incorporate the feedback to improve answer
quality, relevance, and coherence.
Provide only the optimized prompt without
additional commentary.
Graph Extraction agent critique prompt
You are evaluating the prompt given to an LLM
to enrich a given graph with the
information extracted from the text.
The prompt tells the system how to extract
entities and relationships.You will be provided with feedback explaining
which information is missing in the graph.
You have to think: which entity/relationship
types would make this information available
in the graph?
Clearly specify the entities and relationship
types in your answer.
Include only a few entities and relationship
types (not more than 6-7).
Based on this, you have to identify entity and
relationship types that should be included
in the graph and are not specified in the
current prompt.
Focus only on the most crucial
entity/relationship types to meet the
evaluation requirement. Specify only a few
entity/relationship types, the ones that
are most important.
For each entity/relationship, you have to
include examples (each example is a phrase
or a sentence).
The feedback from the system is:
{response_evaluator_output}
The previous critique for the answer generator
was:
{answer_generator_critique}
The current prompt is:
{current_prompt}
Please, determine which entities/relationships
the current prompt is missing based on the
feedback.
Graph Extraction agent prompt optimizer
Your goal is to generate an instruction for an
LLM that enriches a graph using information
from the text.
The instruction must have the following format:
"Focus on:" + LIST OF ENTITY/RELATIONSHIP
TYPES
The instruction tells the model which
entity/relationship types to expand the
graph with.
Include only A FEW entities and relationship
types (NOT MORE THAN 6-7 IN TOTAL).
You have to accompany each entity/relationship
with a full explanation and some examples.
Each example is a phrase or a sentence.
Specify only the relationships and the
entities, don't use formatted examples
since they can mislead the model output
format.
Your suggestions must be based on this feedback:
{graph_extraction_feedback}
Provide only the instruction without additional
commentary.

B LLM-as-a-Judge Evaluation Details
Model
•Model: deepseek-ai/DeepSeek-V3.1 with
thinking mode enabled
• Size: 685B parameters
Prompt Template
You are evaluating two answers to a question.
Your task is to determine which answer
better adheres to the gold-standard
reference
answer.
**Question:**
{question}
**Gold Standard Reference Answer:**
{gold_standard}
**Answer A (from {system_a}):**
{answer_a}
**Answer B (from {system_b}):**
{answer_b}
**Instructions:**
1. First, provide your reasoning about which
answer is better and why.
2. After your reasoning, you MUST end your
response with EXACTLY one of these options
on the last line, with nothing else:
A
B
Do not add any punctuation, explanation, or
other text on the final line. Just the
single letter or word.