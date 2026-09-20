# TRACE: Accountable Agentic Retrieval for Source Discovery in Digital Archives

**Authors**: Donghan Bian, Marie Puren, Florian Cafiero

**Published**: 2026-09-17 08:44:36

**PDF URL**: [https://arxiv.org/pdf/2609.19897v1](https://arxiv.org/pdf/2609.19897v1)

## Abstract
Historical archives pose a difficult retrieval problem for retrievalaugmented generation systems: documents are OCR-degraded, heterogeneous across genres and sources, and require strong source traceability for scholarly and institutional use. We introduce TRACE, a training-free agentic retrieval framework designed for accountable source discovery over historical corpora. The system was developed in the context of DECIDON, an interdisciplinary project on the circulation of political discourse between parliamentary debates and the press during the French Third Republic, involving digitised historical collections and institutional use cases. The prototype is currently deployed internally within the project and accessible to 24 researchers across six partner institutions. We evaluate TRACE on HistoriQA-ThirdRepublic, a benchmark of 1,752 French historical questions over parliamentary debates and newspapers from 1887, with documents derived from Biblioth{è}que nationale de France digitised collections. TRACE achieves R@10 = 0.856 and MRR = 0.653, outperforming sparse, dense, graph-based, and agentic RAG baselines, with the largest gains on multi-hop and cross-corpus questions. At approximately $0.02 per question under the default hosted inference configuration, TRACE also remains economically feasible for heritage institutions, laboratories or companies that cannot rely on costly local GPU infrastructure. These results suggest that, for large digital libraries and archives, retrieval accountability and corpus-aware agent design can provide a practical alternative to heavier training-based or graph-construction approaches.

## Full Text


<!-- PDF content starts -->

TRACE: Accountable Agentic Retrieval for Source Discovery in
Digital Archives
Donghan Bian
École nationale des chartes – PSL
Centre Jean-Mabillon
Paris, France
EPITA
EPITA Research Laboratory
Le Kremlin-Bicêtre, FranceMarie Puren
EPITA
EPITA Research Laboratory
Le Kremlin-Bicêtre, France
École nationale des chartes – PSL
Centre Jean-Mabillon
Paris, FranceFlorian Cafiero
EPITA
EPITA Research Laboratory
Le Kremlin-Bicêtre, France
Geneva Graduate Institute
Centre for Digital Humanities and
Multilateralism
Geneva, Switzerland
Abstract
Historical archives pose a difficult retrieval problem for retrieval-
augmented generation systems: documents are OCR-degraded,
heterogeneous across genres and sources, and require strong
source traceability for scholarly and institutional use. We introduce
TRACE, a training-free agentic retrieval framework designed for
accountable source discovery over historical corpora. The system
was developed in the context of DECIDON, an interdisciplinary
project on the circulation of political discourse between parlia-
mentary debates and the press during the French Third Republic,
involving digitised historical collections and institutional use cases.
The prototype is currently deployed internally within the project
and accessible to 24 researchers across six partner institutions.
We evaluate TRACE on HistoriQA-ThirdRepublic, a benchmark of
1,752 French historical questions over parliamentary debates and
newspapers from 1887, with documents derived from Bibliothèque
nationale de France digitised collections. TRACE achieves R@10 =
0.856 and MRR = 0.653, outperforming sparse, dense, graph-based,
and agentic RAG baselines, with the largest gains on multi-hop and
cross-corpus questions. At approximately $0.02 per question under
the default hosted inference configuration, TRACE also remains
economically feasible for heritage institutions, laboratories or com-
panies that cannot rely on costly local GPU infrastructure. These
results suggest that, for large digital libraries and archives, retrieval
accountability and corpus-aware agent design can provide a prac-
tical alternative to heavier training-based or graph-construction
approaches.
CCS Concepts
•Information systems →Question answering;Information
retrieval;•Computing methodologies →Natural language pro-
cessing.
Keywords
agentic retrieval, retrieval-augmented generation, historical cor-
pora, digital archives, source discovery, OCR-degraded text, multi-
hop retrieval1 Introduction
Archival documents have long served as a cornerstone of historical
research [ 6]. With the convergence of computer science and digital
humanities, large volumes of paper-based materials have been digi-
tised, making large-scale search increasingly feasible. Yet deploying
retrieval-augmented generation (RAG) [ 19] in historical research
settings remains hindered by three compounding challenges. At the
technicallevel, high OCR error rates introduce semantic drift dur-
ing embedding and degrade retrieval precision [ 7,34]. At thequery
level, historical questions routinely span multiple heterogeneous
sources, extended temporal scopes, or require cross-document syn-
thesis rather than single-fact lookup. At themethodologicallevel,
most RAG systems optimise for generation quality [ 24,31], whereas
historical scholarship imposes strict source traceability, interpretive
transparency, and the possibility of human intervention at every
retrieval step [ 27], requirements with no counterpart in general-
purpose RAG settings. These constraints are also shared by libraries,
archives, and institutional repositories that hold large digitised col-
lections but lack corpus-specific supervision or local GPU infras-
tructure.
In response, we propose TRACE (Training-free Retrieval with
Agentic Corpus Exploration), a lightweight, training-free agentic
RAG framework designed for accountable source discovery over
historical corpora. This work is grounded in DECIDON,1a French
National Research Agency project on political discourse circulation
between parliamentary debates and the press during the French
Third Republic, involving computer scientists, historians, and the
Bibliothèque nationale de France (BnF). In historical and archival
research, retrieval is not a hidden preprocessing step but a scholarly
operation in its own right: a retrieved source must be discoverable,
inspectable, and citable. TRACE is also intended to complement
Corpusense, the IIIF-based tool developed within the Mezanno
project for transforming digitised serial sources into structured
data [ 1,5,22]. An open-source version of TRACE is also released2.
Our contributions are threefold:
This is the authors’ version of the work, posted for personal and non-commercial
use. The definitive version was accepted for publication in theProceedings of the 35th
ACM International Conference on Information and Knowledge Management (CIKM ’26),
November 7–11, 2026, Rome, Italy, and is available at https://doi.org/10.1145/3799682.
3840134.
1https://anr.fr/Projet-ANR-25-CE38-4063
2https://github.com/Kepler1908/TRACE
1
arXiv:2609.19897v1  [cs.AI]  17 Sep 2026

Donghan Bian, Marie Puren, and Florian Cafiero
•We reformulate historical archival retrieval assource discov-
ery: an applied, corpus-aware document retrieval problem
under OCR degradation, heterogeneous source structure,
and source-traceability constraints, evaluated on retrieval
quality rather than on generation metrics.
•We introduce TRACE, a training-free agentic retrieval
framework combining corpus-targeted decomposition,
fused warm start, explicit retrieval verdicts with deferred
re-evaluation, and count-grouped reranking over agent-
authored rationales.
•We show empirically that this orchestration of mature, off-
the-shelf retrieval components outperforms purpose-built
graph-based and agentic RAG systems on a 1,752-question
French historical benchmark, while remaining training-free
and costing approximately $0.02 per question.
2 Related Work
2.1 From Retrieval Effectiveness to Retrieval
Accountability
Agentic RAG has emerged in two broad forms: training-based meth-
ods using reinforcement learning [ 3,17,20,31] and training-free
methods relying on framework design [ 4,11,24]. Both demon-
strate meaningful retrieval gains, but share a critical limitation:
they are evaluated end-to-end on generation metrics (Exact Match,
F1, LLM-judged accuracy), leaving retrieval quality unexamined
and conflating parametric knowledge with grounded evidence re-
trieval.3Graph-based RAG [ 12,14,15,37] offers complementary
structural indexing with consistent multi-hop gains, but introduces
two constraints relevant here: graph construction is a form of lossy
compression, and entity extraction degrades severely under OCR
noise, causing structural errors that propagate into retrieval [ 37].
Historical archival corpora, which combine substantial OCR noise,
heterogeneous sub-corpora, and strict source-traceability require-
ments, strain the assumptions of both paradigms simultaneously,
motivating a retrieval-first framework that operates directly on
noisy text without intermediate structural extraction.
2.2 RAG for Historical Corpora in Digital
Humanities
RAG has been increasingly explored in digital humanities for query-
ing large, heterogeneous, and OCR-noisy archival collections span-
ning newspaper archives [ 23,29], parliamentary debates [ 26], clas-
sical texts [ 13], administrative records [ 18], and personal diary
archives [ 21,36]. A recurring finding is that corpus-specific prop-
erties, such as OCR artifacts, archaic vocabulary, and temporal
structure, consistently dominate retrieval performance: named-
entity injection at reranking partially mitigates OCR errors [ 29];
date-based metadata filtering substantially improves precision on
chronologically dense collections [ 18]; and dense retrieval system-
atically fails on historical vocabulary, with sparse retrieval and
reranking outperforming embedding-based approaches [33].
Alongside retrieval performance, a parallel concern has emerged
around source traceability and human oversight [ 18,26,36]: in
3Self-RAG [3] and RAGentA [4] do report recall, but as one metric among several.historical scholarship, identifying relevant sources is a scholarly ob-
jective in its own right, and retrieval must preserve an inspectable,
citable path from query to evidence. These studies motivate a
retrieval-first evaluation perspective in which source discovery,
not only answer generation, becomes the object of measurement.
3 Methodology
3.1 Problem Setting and Architecture Overview
LetD=D 1∪···∪D 𝐾be a heterogeneous archival corpus par-
titioned into 𝐾sub-corpora that differ in genre, vocabulary, and
document quality. Given a question 𝑞in the language of the corpus,
the system must return an ordered list of document identifiers. The
gold set𝐺𝑞⊆D may span multiple sub-corpora and is not recov-
erable from the surface form of 𝑞. We evaluate with recall@ 𝑘as
the primary metric.
TRACE addresses the challenges identified above through a six-
stage pipeline, as shown in Figure 1.
3.2 Planner: Corpus-Targeted Decomposition
The planner is a single LLM call that decomposes 𝑞into a set of
𝑀sub-questions{𝑠1,...,𝑠𝑀}, where𝑀≤𝑀 max. Each𝑠𝑖is a tuple
(text𝑖,𝑐𝑖,entities𝑖)where𝑐𝑖∈C identifies the target sub-corpus.
The design follows the broader intuition of plan-and-execute and
decomposition-based agentic retrieval [ 2,3,30]. The prompt en-
forces three invariants: (i) the sub-questions jointly cover the mean-
ing of𝑞without loss or addition; (ii) each 𝑠𝑖targets exactly one sub-
corpus, so cross-corpus questions are fanned out into per-corpus
sub-questions; (iii) entity names retain their original-language sur-
face form to avoid OCR mismatch from translation.
Corpus targeting guides Phase 1 candidate allocation but does
not restrict the agent loop: the agent may search across corpus
boundaries when evidence suggests cross-corpus connections
(§3.4).
3.3 Phase 1: Fused Warm Start
For each sub-question 𝑠𝑖, Phase 1 produces an initial candidate
pool by fusing complementary retrieval channels via reciprocal-
rank fusion (RRF, 𝑘=60) [ 8]. Each channel independently scores
chunks andmax-poolsper parent document: a document with 𝑁
chunks contributes exactly once per channel—its best chunk score—
preventing any single long document from dominating the pool
through sheer volume of passages. This hybrid design follows re-
cent findings that sparse, dense, metadata-aware, and reranking
strategies often compensate for different failure modes in historical
corpora [18, 33].
Sparse lexical (BM25) [28] retrieves over tokenised chunks
and is effective for proper nouns and domain terms that
survive OCR.
Dense semantic uses cosine similarity over pre-computed
chunk embeddings, capturing paraphrases and conceptual
matches that lexical search misses.
Temporal uses date-based retrieval with exponential decay
scoring by temporal distance from the target date, activating
when a sub-question carries an explicitly resolvable date
[18].
2

TRACE: Accountable Agentic Retrieval for Source Discovery in Digital Archives
Figure 1: Overview of the TRACE pipeline.
The two principal channels are strongly complementary rather
than redundant. Replaying the executed sub-questions through each
channel independently, the mean Jaccard overlap between the BM25
and dense top-20 pools is only0 .19, and28.8%of the gold documents
recovered by the fusion are reachable through exactly one of the
two. Fusing them retrieves78 .9%of gold documents against70 .1%
for the better single channel. This provides implicit robustness to
OCR noise without explicit preprocessing: BM25 matches tokens
that survive digitisation, while dense retrieval captures semantic
paraphrases of corrupted text. In an imbalanced corpus, however,
applying a global result cap systematically truncates minority sub-
corpus documents. We address this by pre-computing the set of
document IDs belonging to the target sub-corpus and passing these
ascandidate constraintsto each channel, so the cap applieswithin
the target sub-corpus rather than across the full collection.
3.4 Phase 2: Agent Loop
The warm-start pool seeds a bounded agent loop with aneffective
step budget. At each step, the agent either issues a new search
query—through sparse, dense, temporal, or exact-substring search—
or reviews a pending batch of retrieved chunks. Search and review
strictly alternate: after each search returns candidates, the agent
must review them before issuing another query, ensuring that each
retrieval is immediately exploited and subsequent queries can build
on newly discovered evidence. This design is related to ReAct-
style alternation between reasoning and action [ 32], but adapts
it to source discovery by making retrieval verdicts explicit. Every
reviewed chunk receives one of three verdicts:
accept The chunk provides evidence toward answering 𝑞.
The agent records a short free-text rationale ( ≤120charac-
ters, reused at reranking time), and the parent document is
deduplicated from subsequent retrieval.reject The chunk is off-topic. Only the chunk—not the par-
ent document—is excluded from future retrieval, preserving
access to sibling passages of the same long document.
hold The chunk is possibly relevant but uncertain given
current context. It enters a boundedheld poolfor deferred
re-evaluation (§3.5).
The agent is always presented with both 𝑞and𝑠𝑖: the sub-
question focuses the search direction, but relevance is defined with
respect to the original question, ensuring that cross-cutting evi-
dence is not discarded by an overly narrow interpretation of the
current sub-question.
Temporal search receives special treatment: the agent’s specified
date window is silently expanded by one day on each boundary,
compensating for indeterminacy in date-boundary decisions with-
out exposing the expansion to the agent. Exposing it would risk
anchoring subsequent temporal reasoning on artificially widened
boundaries. Results from date expansion are conservatively auto-
held rather than accepted, entering the deferred pool for later re-
evaluation.
To prevent circular exploration, the agent maintains a five-slot
structured memory: prior findings, question analysis, evidence gath-
ered, gaps remaining, and next steps. This memory is updated by
replacementat every step. Replacement semantics force the agent
to synthesise rather than accumulate, keeping the memory com-
pact and current. Alongside this, the system tracks aneffective
step counterthat increments only on productive actions. Dupli-
cate queries, parse failures, and empty results are skipped with
informative feedback, preserving the step budget for genuinely
new exploration; two consecutive identical search attempts trigger
forced termination to prevent degenerate loops.
Each agent step is a single-shot LLM call, receiving a system
prompt and a constructed user prompt in JSON mode. The agent
retains no memory of prior API calls; all state is reconstructed
3

Donghan Bian, Marie Puren, and Florian Cafiero
from the structured memory and a windowed action log contain-
ing the last eight entries. The output is a JSON object ( thought ,
action ,args ,memory_update ) parsed without tool-calling APIs.
This design eliminates context pollution from accumulated con-
versation history, makes calls stateless and parallelisable across
sub-questions, and renders the system provider-agnostic: any LLM
supporting JSON-mode output is sufficient, with no dependency
on proprietary function-calling interfaces.
3.5 Finish, Briefing, and Replan
When the agent terminates, aheld re-evaluationcall presents the
full accepted set, the stage memory, and summaries of all held
chunks to the LLM, which promotes any held chunk now supported
by the accumulated evidence. This is the payoff of the three-way
verdict: early uncertainty is resolved against late context rather
than collapsed into a premature binary decision at first encounter.
Abriefingis then synthesised as a structured object containing
a narrative summary, confirmed facts, confirmed dates, confirmed
entities, remaining gaps, and search hints. It serves as the sole
communication channel between sub-questions. This enables later
sub-questions to start from the evidence frontier rather than search-
ing blind. Before each non-initial sub-question 𝑠𝑖(𝑖>1), a replanner
receives the previous briefing and the remaining plan, and may
modify remaining sub-questions or spawn new ones when the brief-
ing reveals gaps not covered by the original decomposition. The
total number of sub-questions is capped at 𝑀maxto bound cost. This
transforms the retrieval plan from a static script into an adaptive
investigation while keeping sub-question processing auditable.
3.6 Merge and Count-Grouped Reranking
After all sub-questions complete, their accepted sets{𝐴 1,...,𝐴𝑀}
are merged. Because sub-questions are structurally decoupled and
share no accepted sets during their respective agent loops, a doc-
ument accepted by multiple sub-questions represents convergent
evidence from separate investigative paths rather than confirmation
bias from shared state. For each document 𝑑∈Ð
𝑖𝐴𝑖we compute
aconfirmation count cnt(𝑑)=|{𝑖 :𝑑∈𝐴𝑖}|, and the final ranking
proceeds in two stages. First, documents are partitioned by cnt(𝑑)
and groups are placed in decreasing count order. Within each group
of size≥2, an LLM ranks the documents using the concatenation of
the agent’s own acceptance rationales—the ≤120-character justifi-
cations recorded during the agent loop—rather than re-reading full
document text. Singleton groups pass through without an LLM call,
so the reranker’s cost is proportional to the number of ties rather
than the total candidate count.
4 Experimental Setup
4.1 Corpus and Questions
We use HistoriQA-ThirdRepublic [ 25] as our evaluation dataset.
It consists of a text corpus and a question-answer dataset built
around the same type of historical materials that motivate the
DECIDON use case: parliamentary debates and newspapers from
the French Third Republic. The benchmark therefore serves both
as an evaluation resource and as a controlled proxy for the broader
applied problem of accountable source discovery over BnF-derived
digitised collections.The corpus consists of 3,386 French-language documents from
the French Third Republic period (1887), drawn from three hetero-
geneous sub-corpora: parliamentary debate transcripts from the
Journal Officiel de la République françaiseand two Parisian daily
newspapers (Le Gaulois,L’Intransigeant) representing distinct edi-
torial positions on the political events of the period. The documents
are derived from OCR-digitised historical collections held by the
BnF, reflecting the type of large-scale digitised material encoun-
tered in national library and archival infrastructures. They contain
substantial OCR errors, irregular layouts, and genre-specific vocab-
ulary, making the corpus a realistic testbed for accountable retrieval
in digital heritage settings. Documents are split into 5,664 sentence-
boundary chunks with a maximum of 1,000 whitespace tokens and
no overlap.
The evaluation set comprises 1,752 questions with gold docu-
ment sets, constructed following the methodology of Pellet et al .
[25] in collaboration with a domain historian, who validated the
factual and contextual coherence of the generated questions. Ques-
tions span four types reflecting patterns of historical inquiry:
single-hop (SH, 𝑛=889), where each question targets a single
source document; multi-hop generic (MH-Generic, 𝑛=529), re-
quiring cross-source synthesis across parliamentary and press
sources; bridge-entity (MH-BridgeEnt, 𝑛=142), where a shared
named entity connects documents from different sub-corpora;
and comparative (MH-Comp, 𝑛=192), contrasting viewpoints
across heterogeneous sources. Gold sets may span multiple sub-
corpora, making corpus-aware retrieval essential. Prior evaluation
on this corpus demonstrates the severity of the retrieval challenge:
standard dense retrieval achieves a Recall@3 of only 14.3% on
cross-newspaper queries and 47.8% on cross-domain queries [25].
4.2 Baselines and Metrics
We conduct three sets of experiments.(1) Main resultsevaluate
TRACE under two backbone configurations: DeepSeek-V3.2 [ 9], the
earlier model used during system development, and DeepSeek-V4-
Flash [ 10], the configuration adopted for the remaining experiments.
(2) Baseline comparisoncompares TRACE against six systems
across three retrieval paradigms: basic retrieval (BM25 and dense
retrieval), graph-based RAG (HippoRAG 2 [ 15] and LinearRAG [ 37]),
and agentic RAG (A-RAG [ 11] and MA-RAG [ 24]). These baselines
are selected for the distinctiveness of their design or strong reported
performance on multi-hop retrieval benchmarks.(3) Cumulative
ablationprogressively adds components to a baseline agent loop
to quantify the contribution of each mechanism. Experiments (2)
and (3) use DeepSeek-V4-Flash and Qwen3-Embedding-8B [ 35]
throughout.
All baseline systems were adapted to operate on the French
corpus. Adaptations were limited to what correctness required:
English prompts and extraction templates were translated, and
the SpaCy [ 16] pipeline used for entity extraction in the graph-
based systems was replaced with its French equivalent. All systems
consume an identical corpus conversion—the same documents, the
same sentence-boundary chunking at 1,000 whitespace tokens, and
the same document identifiers—and share the embedding model
and LLM backbone wherever applicable, with retrieval depth fixed
across systems. Differences therefore reflect retrieval architecture
4

TRACE: Accountable Agentic Retrieval for Source Discovery in Digital Archives
rather than corpus preprocessing or model choice. The conversion
and adaptation scripts are released alongside the system.
For a question with gold set 𝐺𝑞and ranked result list 𝜋, we
report:
R@𝑘=|𝐺𝑞∩𝜋 1:𝑘|
|𝐺𝑞|,MRR(𝑞)=1
|𝐺𝑞|∑︁
𝑔∈𝐺𝑞1
rank𝜋(𝑔),
where rank𝜋(𝑔)=∞ if𝑔∉𝜋 . This multi-gold MRR averages the
reciprocal rank overallgold documents, penalising systems that
surface only the easiest gold document while burying the rest. For
agentic systems, we additionally report precision 𝑃acc=|𝐴𝑞∩
𝐺𝑞|/|𝐴𝑞|and recall𝑅acc=|𝐴𝑞∩𝐺𝑞|/|𝐺𝑞|over the accepted set
𝐴𝑞, measuring the quality of accept/reject decisions independently
of the final ranking. In the main experiments, we also report the
average number of accepted documents for each question type.
5 Results
5.1 Main Results
Table 1 reports per-type retrieval performance on the full 1,752-
question evaluation set. The two backbone models exhibit a consis-
tent precision-recall trade-off: DeepSeek-V3.2 achieves higher over-
all recall by accepting more documents on average, while DeepSeek-
V4-Flash yields substantially higher 𝑃accwith a more selective ac-
ceptance strategy. Both models show the same difficulty gradient
across question types: single-hop questions are near-ceiling, multi-
hop generic questions maintain high recall but lower MRR, and
bridge-entity and comparative questions remain hardest as they
require cross-corpus evidence chains. Across all types, 𝑅accclosely
tracks R@10, indicating that the performance bottleneck lies in the
agent loop rather than in the final reranking step. Attributing every
missed gold document in the V4-Flash run to a cause confirms this:
82.6%of misses occur inside the loop and only17 .4%are documents
that were accepted but ranked below rank ten. Within the loop,
documents that were surfaced and then judged irrelevant (36 .8%of
misses) slightly outnumber those never retrieved by any channel
(36.5%), with a further9 .3%held but never promoted. Relevance
judgment, not retrieval coverage, is thus the single largest source
of error.
The more conservative acceptance behaviour of DeepSeek-V4-
Flash should be interpreted as a different operating point rather
than a simple degradation: the agent declines to accept marginally
relevant documents, improving accepted-set precision while mod-
estly reducing accepted-set recall. This matters in digital-archive
workflows, where every accepted document may become part of a
human reading queue. A configuration that returns a smaller, more
precise accepted set can therefore be preferable even when raw
recall is slightly lower.
In our hosted inference setup, DeepSeek-V4-Flash consumed
a total of 177.6 M input tokens and 39.7 M output tokens, cost-
ing approximately $36 at the API prices used for the experiment.
DeepSeek-V3.2 cost approximately $93, or 158% more, driven by
higher per-token pricing and 1.8 ×greater input consumption from
deeper agent exploration. At approximately $0.02 per question, the
V4-Flash configuration makes large-scale corpus exploration eco-
nomically feasible for humanities laboratories operating under tight
resource constraints, while remaining well below the practical costof maintaining equivalent local GPU deployment for occasional or
project-based use. Given these deployment advantages, we adopt
DeepSeek-V4-Flash as the backbone for practical use.
We also quickly tested our system’s latency. On a 200-question
timed sample of target dataset, TRACE issues on average33 .8LLM
calls per question, totalling169s of model time. Latency is almost
entirely provider-bound:97 .8%of per-question wall-clock is spent
awaiting the hosted model, leaving roughly four seconds of frame-
work computation, so responsiveness tracks the serving backend.
Because sub-questions and questions are processed independently,
serial timing is a worst case; running eight questions concurrently,
the same workload completes in22s per question. Source discovery
is therefore an asynchronous operation at archival scale, which
matches how the prototype is used in practice, where researchers
issue queries continuously or submit a prepared question set.
5.2 Comparison with Baselines
Table 2 compares TRACE against representative baselines across
three retrieval paradigms.
On basic retrieval, dense retrieval performs well on single-hop
questions but degrades substantially on multi-hop types, a pattern
consistent with what Pellet et al . [25] observed on this corpus.
Sparse retrieval (BM25) is competitive on single-hop questions
but performs poorly on comparative questions, where relevant
evidence is dispersed across heterogeneous sources and lexical
overlap between the query and the evidence may be limited.
Graph-based systems yield a more surprising result: both Hip-
poRAG 2 and LinearRAG perform at levels comparable to dense
retrieval across most question types, without recovering their re-
ported multi-hop advantage. Both systems return largely what
dense retrieval already returns:7 .1and6.7of their top-ten docu-
ments respectively, against3 .0for BM25. The documents they add
are a wash: they gain98and113gold documents over dense while
losing86and132, a net of +12and−19out of roughly1 ,900. The
graph layer therefore perturbs the dense ranking without contribut-
ing retrieval signal.
TRACE achieves consistent and substantial gains across all ques-
tion types, with the margin widening on multi-hop questions that
require cross-corpus evidence chains.
The two agentic baselines, A-RAG and MA-RAG, do not produce
explicit ranked document lists, instead generating answers from
their full retrieval trajectory. To place them on the same footing as
the other systems, we rank each trajectory byorder of discoveryand
evaluate the resulting list. This is a generous protocol: it credits the
baselines with any relevant document encountered at any point in
their trajectory, regardless of whether it was used in the generated
answer, and their trajectories are long enough for R@10 to be
meaningful (median16and14documents respectively). It is also
asymmetric in their disfavour in one respect, since discovery order
is not a relevance ranking, whereas TRACE returns a deliberately
ranked list. Panel (b) reports the same accepted sets under precision
and recall, independently of ordering. Under both views TRACE
achieves substantially higher precision and recall simultaneously,
indicating that the advantage stems from the quality of individual
retrieval decisions rather than from a larger candidate pool.
5

Donghan Bian, Marie Puren, and Florian Cafiero
Table 1: Main retrieval results on the full evaluation set (N=1,752). #RET is the mean number of accepted documents.
Type R@3 R@5 R@10 MRR P acc Racc #RET
DeepSeek-V3.2 (0324)
SH 0.872 0.904 0.922 0.797 0.246 0.933 8.1
MH – Generic 0.684 0.808 0.892 0.532 0.250 0.932 11.0
MH – BridgeEnt 0.644 0.761 0.820 0.508 0.276 0.838 8.5
MH – Comparative 0.484 0.596 0.727 0.383 0.140 0.846 21.1
All 0.7540.830 0.8840.648 0.2380.91510.4
DeepSeek-V4-Flash
SH 0.852 0.881 0.893 0.793 0.452 0.902 4.6
MH – Generic 0.739 0.832 0.889 0.560 0.349 0.906 8.5
MH – BridgeEnt 0.648 0.725 0.750 0.472 0.348 0.754 5.7
MH – Comparative 0.500 0.578 0.674 0.396 0.275 0.789 14.9
All0.7630.820 0.8560.653 0.3930.8797.0
Table 2: Baseline comparison on HistoriQA-ThirdRepublic. Panel (a) reports R@10 and MRR for all systems. Panel (b) reports
accepted-set precision and recall for agentic systems.
(a) R@10 and MRR per question type (all systems)
System SH MH – Generic MH – BridgeEnt MH – Comp All
R@10 MRR R@10 MRR R@10 MRR R@10 MRR R@10 MRR
Basic retrieval
BM25 0.774 0.606 0.596 0.381 0.623 0.408 0.180 0.098 0.643 0.467
Dense retrieval 0.879 0.722 0.665 0.400 0.665 0.413 0.302 0.129 0.734 0.535
Graph-based RAG
HippoRAG 2 0.879 0.581 0.669 0.371 0.665 0.392 0.323 0.130 0.737 0.453
LinearRAG 0.875 0.696 0.656 0.377 0.673 0.413 0.281 0.119 0.727 0.513
Agentic RAG
A-RAG 0.439 0.248 0.385 0.209 0.430 0.225 0.107 0.037 0.385 0.211
MA-RAG 0.636 0.422 0.603 0.325 0.630 0.377 0.273 0.121 0.586 0.356
TRACE (ours)0.893 0.793 0.889 0.560 0.750 0.472 0.674 0.396 0.856 0.653
(b) Agent decision quality (𝑃 acc/𝑅acc) per question type (agentic systems only)
System SH MH – Generic MH – BridgeEnt MH – Comp All
𝑃acc𝑅acc𝑃acc𝑅acc𝑃acc𝑅acc𝑃acc𝑅acc𝑃acc𝑅acc
A-RAG 0.049 0.606 0.074 0.489 0.084 0.514 0.016 0.185 0.056 0.517
MA-RAG 0.056 0.786 0.127 0.682 0.150 0.701 0.061 0.323 0.085 0.697
TRACE (ours)0.452 0.902 0.349 0.906 0.348 0.754 0.275 0.789 0.393 0.879
The purpose of this comparison is not simply to establish that
TRACE outperforms existing systems. More importantly, it surfaces
a structural blind spot in how RAG systems are typically evaluated:
generation-centric metrics obscure the quality of the underlying
retrieval process, and systems optimised for open-domain QA do
not naturally transfer to domain-specific deployment. Effective
retrieval in historical research requires targeted design choices.
5.3 Cumulative Ablation Study
Table 3 reports a cumulative ablation in which components are
added one at a time to a baseline agent loop. Because the mecha-
nisms in TRACE are designed to interact—decomposition producesthe sub-questions that memory and replanning operate on, while
the warm start seeds the pool that the agent loop refines—we adopt
a cumulative rather than leave-one-out design, which better reflects
the system’s intended operating mode.
Sub-question decomposition yields the largest gain (+8.9%), dra-
matically improving multi-hop types while leaving single-hop per-
formance largely unchanged. Memory and replanning (+2.4%) pri-
marily benefits bridge-entity questions by enabling evidence to
accumulate across sub-question boundaries. The Phase 1 warm
start (+5.0%) stabilises all question types by seeding the agent with
a high-quality initial candidate pool. The hold-and-re-evaluation
mechanism contributes modestly in aggregate (+0.5%), but acts as a
6

TRACE: Accountable Agentic Retrieval for Source Discovery in Digital Archives
Table 3: Cumulative ablation on the full evaluation set (N=1,752). Each level adds one component to the previous level. Δis the
change in overall R@10 from the previous level.
SH MH – Generic MH – BridgeEnt MH – Comp All
Lv Added componentR@10 MRR R@10 MRR R@10 MRR R@10 MRR R@10Δ
0 Baseline (agent only) 0.813 0.713 0.647 0.435 0.352 0.247 0.466 0.284 0.688 —
1 + Decomposition 0.790 0.699 0.849 0.539 0.637 0.413 0.625 0.368 0.777 +8.9%
2 + Memory & replan 0.807 0.707 0.846 0.536 0.775 0.455 0.669 0.371 0.801 +2.4%
3 + Phase 1 warm start 0.885 0.785 0.882 0.559 0.736 0.471 0.695 0.411 0.851 +5.0%
4 + Hold & re-eval (full) 0.893 0.793 0.889 0.560 0.750 0.472 0.674 0.396 0.856 +0.5%
conservative safety layer: its value is concentrated in marginal cases
where a premature rejection would permanently foreclose a docu-
ment whose relevance only becomes clear under late-accumulated
context.
6 Discussion, Deployment, and Future Work
The results show that accountable source discovery over noisy
historical archives benefits less from heavier indexing or model
training than from corpus-aware orchestration. TRACE com-
bines four design choices that proved particularly important in
this setting: decomposition into corpus-targeted sub-questions,
multi-channel fused retrieval, explicit accept/reject/hold decisions,
and final ranking based on cross-sub-question confirmation. On
HistoriQA-ThirdRepublic, this design achieves R@10 = 0.856 and
MRR = 0.653, outperforming sparse, dense, graph-based, and agentic
RAG baselines while remaining training-free.
The DECIDON use case also highlights why retrieval should be
treated as an applied scholarly task rather than as a hidden pre-
processing step for answer generation. In digital-heritage settings,
users need to identify relevant sources, understand why they were
retrieved, and preserve a traceable path from query to evidence.
TRACE is currently deployed as an internal prototype within the DE-
CIDON project and is accessible to 24 researchers from six partner
institutions: EPITA, EHESS-CRH, Inria, the Bibliothèque nationale
de France, the École nationale des chartes, and LARHRA. The pro-
totype has been accessible to project researchers since March 2026
and has supported several hundred exploratory research queries
over parliamentary debates and the press. It remains an early-stage
internal research service rather than a production deployment, and
we report it as such: a systematic evaluation of adoption, workflow
impact, and expert reading burden is deferred to future work.
The prototype supports four historical case studies: the politi-
cisation of nature, the Colonial Party, social and tax legislation,
and anti-parliamentarianism during the French Third Republic. In
these cases, historians use TRACE to locate relevant passages, com-
pare documentary contexts, and follow the circulation of actors,
arguments, vocabularies, and framings over time. The system re-
mains a research service rather than a public-facing BnF service, but
the planned integration with Corpusense would support a broader
BnF-facing workflow in which TRACE helps identify and docu-
ment relevant sources, while Corpusense supports the downstream
transformation of selected IIIF collections or serial documents into
structured data [1, 5, 22].Several limitations remain. First, the planner occasionally col-
lapses an inherently multi-hop query into a single sub-question,
preventing the replan mechanism from firing. This is rare—multi-
hop questions receive two or more sub-questions in roughly98%of
cases—but consequential when it happens: of the eighteen affected
questions, thirteen fail, a failure rate of72%against a20%base
rate. Second, the benchmark covers a single historical year, 1887;
generalisation to other periods, languages, and archival genres re-
mains to be tested, and a second corpus is under construction for
this purpose. Third, the remaining pipeline constants—step and
sub-question budgets, memory size, and the temporal expansion
window—were fixed during development on this corpus rather than
tuned per question type, and we would expect their optimal val-
ues to differ on a collection with different structure. Forth, some
prompt-level assumptions, such as the relation between parliamen-
tary debates and newspapers, are corpus-specific and would require
adjustment for other documentary settings. Finally, TRACE is de-
signed so that every returned document carries a traceable path
from query to evidence—the sub-question that sought it, the search
that surfaced it, and the rationale recorded when it was accepted—
but we evaluate retrieval effectiveness rather than the downstream
value of that trace. Whether it measurably improves expert trust or
reduces verification effort remains open, and is the natural subject
of the user study we plan alongside further deployment. Preliminary
local-inference tests on an RTX PRO 6000 workstation also showed
that smaller models still struggle with multi-step planning and
structured-output stability, suggesting that local deployment will
require either larger institution-grade infrastructure or a protocol-
optimised version of TRACE.
Future work will focus on making TRACE a mixed-initiative
source discovery system. Rather than placing historians only at the
end of the pipeline, we plan to test lightweight checkpoints where
expert users can validate or revise the planner’s decomposition,
inspect held documents before final rejection, and adjust the final
accepted set. Additional deployment work will measure latency,
token consumption, source reuse, and expert reading burden across
larger samples, enabling a more precise evaluation of the trade-
off between retrieval accuracy, computational cost, and human
workload.
GenAI Usage Disclosure
Generative AI tools were used to assist with language editing and
code completion during the preparation of this work. All scientific
claims, experimental design choices, analyses, reported results, and
7

Donghan Bian, Marie Puren, and Florian Cafiero
final text were produced, reviewed, and validated by the authors,
who remain fully accountable for the content of the paper.
References
[1] Nathalie Abadie, Marie Carlin, Edwin Carlinet, Joseph Chazalon, Pascal Cristofoli,
et al.2026. Mezanno : des sources sérielles aux données structurées pour les
humanités numériques. Démonstrations de la conférence EGC 2026. https:
//hal.science/hal-05444727
[2] Paul J. L. Ammann, Jonas Golde, and Alan Akbik. 2025. Question Decomposition
for Retrieval-Augmented Generation. arXiv:2507.00355. https://arxiv.org/abs/
2507.00355
[3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
arXiv:2310.11511. doi:10.48550/arXiv.2310.11511
[4] Ines Besrour, Jingbo He, Tobias Schreieder, and Michael Färber. 2025. RAGentA:
Multi-Agent Retrieval-Augmented Generation for Attributed Question Answer-
ing. arXiv:2506.16988. doi:10.48550/arXiv.2506.16988
[5]Bibliothèque nationale de France. 2026. Mezanno : libérer les don-
nées des archives numérisées. Chroniques de la BnF, no. 105, avril–
juillet 2026. https://www.bnf.fr/fr/mezanno-liberer-les-donnees-des-archives-
numerisees-echos-de-recherche
[6] Francis X. Blouin Jr. and William G. Rosenberg. 2011. Authoritative History and
Authoritative Archives. InProcessing the Past: Contesting Authorities in History
and the Archives, Francis X. Blouin Jr. and William G. Rosenberg (Eds.). Oxford
University Press, Oxford, UK. doi:10.1093/acprof:oso/9780199740543.003.0002
[7] Jonathan Bourne. 2026. Reading the unreadable: creating a dataset of 19th century
English newspapers using image-to-text language models.Digital Scholarship in
the Humanities41, 1 (2026), 22–40. doi:10.1093/llc/fqaf151
[8] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
InProceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval. Association for Computing Machinery,
New York, NY, USA, 758–759. doi:10.1145/1571941.1572114
[9] DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language
Models. arXiv:2512.02556. doi:10.48550/arXiv.2512.02556
[10] DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Efficient Million-Token
Context Intelligence. Technical report. https://huggingface.co/deepseek-
ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
[11] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan Wang, Pengyu Wang, Xiaorui
Wang, and Zhendong Mao. 2026. A-RAG: Scaling Agentic Retrieval-Augmented
Generation via Hierarchical Retrieval Interfaces. arXiv:2602.03442. doi:10.48550/
arXiv.2602.03442
[12] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2025. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization. arXiv:2404.16130. doi:10.48550/arXiv.2404.16130
[13] Yang Fan, Zhang Qi, Xing Wenqian, Liu Chang, and Liu Liu. 2025. Research on
Graph-Retrieval Augmented Generation Based on Historical Text Knowledge
Graphs. arXiv:2506.15241. doi:10.48550/arXiv.2506.15241
[14] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779. doi:10.
48550/arXiv.2410.05779
[15] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. arXiv:2502.14802. doi:10.48550/arXiv.2502.14802
[16] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020.
spaCy: Industrial-strength Natural Language Processing in Python. Software.
doi:10.5281/zenodo.1212303
[17] Yulong Hui, Chao Chen, Zhihang Fu, Yihao Liu, Jieping Ye, and Huanchen Zhang.
2026. Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box
Retrieval. arXiv:2510.27566. doi:10.48550/arXiv.2510.27566
[18] Jeong Ha Lee, Ghazanfar Ali, and Jae-In Hwang. 2025. A Retrieval-Augmented
Generation System for Accurate and Contextual Historical Analysis: AI-Agent
for the Annals of the Joseon Dynasty.Computer Animation and Virtual Worlds
36, 4 (2025), e70048. doi:10.1002/cav.70048
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InAdvances in Neural Informa-
tion Processing Systems, Vol. 33. 9459–9474. https://arxiv.org/abs/2005.11401
[20] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models. arXiv:2501.05366. doi:10.48550/arXiv.2501.05366
[21] Claire Lin, Bo-Han Feng, Xuanjun Chen, Te-Lun Yang, Hung-yi Lee, and Jyh-
Shing Roger Jang. 2025. A Preliminary Study of RAG for Taiwanese Historical
Archives. arXiv:2511.07445. doi:10.48550/arXiv.2511.07445[22] Mezanno Project. 2026. Corpusense. Project website. https://mezanno.xyz/
corpusense/ Accessed 2026-05-13.
[23] Anthony Mudet and Souhail Bakkali. 2025. Hybrid Retrieval-Augmented Gener-
ation for Robust Multilingual Document Question Answering. arXiv:2512.12694.
https://arxiv.org/abs/2512.12694
[24] Thang Nguyen, Peter Chin, and Yu-Wing Tai. 2025. MA-RAG: Multi-Agent
Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning.
arXiv:2505.20096. doi:10.48550/arXiv.2505.20096
[25] Aurélien Pellet, Marie Puren, and Julien Perez. 2026. HistoriQA-ThirdRepublic:
Multi-Hop Question Answering Corpus for Historical Research, Parliamentary
Debates from the French Third Republic (1870–1940). InProceedings of the Lan-
guage Resources and Evaluation Conference. ELRA Language Resources Associa-
tion, Palma de Mallorca, Spain. https://hal.science/hal-05438255
[26] Julien Perez, Aurélien Pellet, and Marie Puren. 2025. Évaluation automatique
du retour à la source dans un contexte historique long et bruité. Application
aux débats parlementaires de la Troisième République française. InActes de
l’atelier Évaluation des modèles génératifs (LLM) et challenge 2025 (EvalLLM),
Frédéric Bechet, Adrian-Gabriel Chifu, Karen Pinel-sauvagnat, Benoit Favre,
Eliot Maes, and Diana Nurbakova (Eds.). ATALA and ARIA, Marseille, France,
138–150. https://aclanthology.org/2025.jeptalnrecital-evalllm.12/
[27] Marie Puren, Donghan Bian, Aurélien Pellet, Julien Perez, and Florian Cafiero.
2026. Aligner méthode historique et RAG : transformer un assistant conver-
sationnel en chaîne de preuve auditable et discutable. HAL preprint. https:
//hal.science/hal-05460105
[28] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389. doi:10.1561/1500000019
[29] The Trung Tran, Carlos-Emiliano González-Gallardo, and Antoine Doucet. 2025.
Retrieval Augmented Generation for Historical Newspapers. InProceedings of the
24th ACM/IEEE Joint Conference on Digital Libraries. Association for Computing
Machinery, New York, NY, USA. doi:10.1145/3677389.3702542
[30] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee,
and Ee-Peng Lim. 2023. Plan-and-Solve Prompting: Improving Zero-Shot Chain-
of-Thought Reasoning by Large Language Models. arXiv:2305.04091. doi:10.
48550/arXiv.2305.04091
[31] Tianle Xia, Ming Xu, Lingxiang Hu, Yiding Sun, Wenwei Li, Linfang Shang,
Liqun Liu, Peng Shu, Huan Yu, and Jie Jiang. 2026. Search-P1: Path-Centric
Reward Shaping for Stable and Efficient Agentic RAG Training. arXiv:2602.22576.
doi:10.48550/arXiv.2602.22576
[32] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629. doi:10.48550/arXiv.2210.03629
[33] Yongan Yu, Xianda Du, Qingchen Hu, Jiahao Liang, Jingwei Ni, Dan Qiang, Kaiyu
Huang, Grant McKenzie, Renee Sieber, and Fengran Mo. 2025. WeatherArchive-
Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather
Archives. arXiv:2510.05336. doi:10.48550/arXiv.2510.05336
[34] Junyuan Zhang, Qintong Zhang, Bin Wang, Linke Ouyang, Zichen Wen, Ying
Li, Ka-Ho Chow, Conghui He, and Wentao Zhang. 2025. OCR Hinders RAG:
Evaluating the Cascading Impact of OCR on Retrieval-Augmented Generation.
arXiv:2412.02592. doi:10.48550/arXiv.2412.02592
[35] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through
Foundation Models. arXiv:2506.05176. doi:10.48550/arXiv.2506.05176
[36] Jing Zhou, Li Si, and Wenjun Hou. 2025. Humanities-in-the-Loop: Using Close
Reading as a Method for Retrieval-Augmented Generation (RAG).Proceedings of
the Association for Information Science and Technology62, 1 (2025), 1185–1189.
doi:10.1002/pra2.1355
[37] Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao
Chen, Qinggang Zhang, and Xiao Huang. 2025. LinearRAG: Linear Graph
Retrieval Augmented Generation on Large-scale Corpora. arXiv:2510.10114.
doi:10.48550/arXiv.2510.10114
8