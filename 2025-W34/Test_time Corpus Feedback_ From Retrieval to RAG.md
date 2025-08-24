# Test-time Corpus Feedback: From Retrieval to RAG

**Authors**: Mandeep Rathee, Venktesh V, Sean MacAvaney, Avishek Anand

**Published**: 2025-08-21 10:57:38

**PDF URL**: [http://arxiv.org/pdf/2508.15437v1](http://arxiv.org/pdf/2508.15437v1)

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a standard framework for
knowledge-intensive NLP tasks, combining large language models (LLMs) with
document retrieval from external corpora. Despite its widespread use, most RAG
pipelines continue to treat retrieval and reasoning as isolated components,
retrieving documents once and then generating answers without further
interaction. This static design often limits performance on complex tasks that
require iterative evidence gathering or high-precision retrieval. Recent work
in both the information retrieval (IR) and NLP communities has begun to close
this gap by introducing adaptive retrieval and ranking methods that incorporate
feedback. In this survey, we present a structured overview of advanced
retrieval and ranking mechanisms that integrate such feedback. We categorize
feedback signals based on their source and role in improving the query,
retrieved context, or document pool. By consolidating these developments, we
aim to bridge IR and NLP perspectives and highlight retrieval as a dynamic,
learnable component of end-to-end RAG systems.

## Full Text


<!-- PDF content starts -->

Test-time Corpus Feedback: From Retrieval to RAG
Mandeep Rathee
L3S Research Center
rathee
@l3s.deVenktesh V
TU Delft
venkyviswa12
@gmail.comSean MacAvaney
University of Glasgow
sean.macavaney
@glasgow.ac.ukAvishek Anand
TU Delft
avishek.anand
@tudelft.nl
Abstract
Retrieval-Augmented Generation (RAG)
has emerged as a standard framework for
knowledge-intensive NLP tasks, combining
large language models (LLMs) with document
retrieval from external corpora. Despite its
widespread use, most RAG pipelines continue
to treat retrieval and reasoning as isolated
components—retrieving documents once
and then generating answers without further
interaction. This static design often limits
performance on complex tasks that require
iterative evidence gathering or high-precision
retrieval. Recent work in both the information
retrieval (IR) and NLP communities has begun
to close this gap by introducing adaptive
retrieval and ranking methods that incorporate
feedback. In this survey, we present a
structured overview of advanced retrieval
and ranking mechanisms that integrate such
feedback. We categorize feedback signals
based on their source and role in improving the
query, retrieved context, or document pool. By
consolidating these developments, we aim to
bridge IR and NLP perspectives and highlight
retrieval as a dynamic, learnable component of
end-to-end RAG systems.
1 Introduction
Large language models (LLMs) augmented with
retrieval have become a dominant paradigm for
knowledge-intensive NLP tasks. In a typical
retrieval-augmented generation (RAG) setup, an
LLM retrieves documents from an external cor-
pus and conditions generation on the retrieved ev-
idence (Lewis et al., 2020b; Izacard and Grave,
2021). This setup mitigates a key weakness of
LLMs—hallucination—by grounding generation in
externally sourced knowledge. RAG systems now
power open-domain QA (Karpukhin et al., 2020),
fact verification (V et al., 2024; Schlichtkrull et al.,
2023), knowledge-grounded dialogue, and explana-
tory QA.Despite their widespread use, many RAG sys-
tems rely on static, off-the-shelf retrieval mod-
ules—e.g., BM25 (Robertson et al., 1995) or dense
dual encoders (Karpukhin et al., 2020)—that are
minimally adapted to the downstream task or do-
main. While re-rankers (Nogueira et al., 2020;
Pradeep et al., 2023b) can improve ranking preci-
sion, the underlying retrieval often remains brittle
in scenarios that demand complex reasoning: multi-
hop QA, claim verification, procedural queries, or
dialogue-based question answering. These tasks
frequently require iterative lookups, query decom-
position, or high-precision evidence—capabilities
that static retrieval pipelines lack.
In contrast to the prevailing view of retrieval as
a fixed first step, a growing body of work in the
IR community treats retrieval as a feedback-driven,
adaptive process —where signals from the output
stage is used to guide when to retrieve, how to
reformulate queries, and which evidence to include.
Definition
In this survey, we define feedback in RAGs
as any signal—derived from the corpus at dif-
ferent levels – retrieval, ranking, or generation
components. This feedback is used to improve
the query, the context used for generation, or
the set of retrieved documents.
We note that such feedback may be applied in
one or multiple rounds and can originate from inter-
nal model signals (e.g., uncertainty or confidence),
external modules (e.g., rankers or verifiers), or user
behavior (e.g., clicks or clarifications). Our notion
of corpus feedback or simply feedback arises at
three key stages:
1.Query-level feedback , where the input query
is rewritten, expanded, or decomposed using
model introspection or relevance signals (refer
Section 3);arXiv:2508.15437v1  [cs.IR]  21 Aug 2025

2.Retrieval-level feedback , where rankers or
corpus structure are used to revise or expand
the document pool across rounds (refer Sec-
tion 4);
3.Generation-time feedback , where confi-
dence, or verifier critiques trigger new re-
trievals or corrections (refer Section 5).
Figure 1 shows an overview of these feedback
stages. This survey synthesizes recent work that
operationalizes these feedback signals across RAG
pipelines. We organize methods by where and how
feedback is applied – not by architecture or dataset –
emphasizing how feedback improves retrieval adap-
tively rather than statically. Our scope is deliber-
ately focused on retrieval-centric innovations in
RAG. We do not cover standalone prompting or
answer-generation strategies unless they directly
influence the retrieval component. Our goal is to
help NLP researchers treat retrieval as a dynamic,
learnable component—just as vital as the gener-
ator—especially in tasks that require reasoning
over incomplete, multi-part, or contextual knowl-
edge. We also review the experimental landscape
for retrieval-centric RAG in Section 6: common
benchmarks, evaluation metrics, and emerging stan-
dards for assessing retriever quality in knowledge-
intensive tasks. By consolidating these develop-
ments, this survey attempts to bridge the gap be-
tween information retrieval and NLP communities,
highlighting how feedback can drive the next gen-
eration of retrieval-aware, reasoning-capable RAG
systems.
2 Preliminaries
2.1 Retrieval System
The core objective of a retrieval system is to iden-
tify and rank a subset of documents (d1, d2, ..., d k)
from a large corpus Cbased on their estimated rel-
evance to a query q. Classical retrieval approaches,
such as BM25 (Robertson et al., 1995), rely on
exact term matching and produce sparse relevance
scores. In contrast, dense retrieval methods employ
neural encoders to project queries and documents
into a shared embedding space, enabling semantic
similarity matching (Karpukhin et al., 2020). Since
first-stage retrievers often produce noisy candi-
dates, modern pipelines incorporate a second-stage
re-ranking step using more expressive models. This
includes LLM-based rankers (Pradeep et al., 2023b;Ma et al., 2024; Sun et al., 2023) and reasoning-
augmented models such as ReasonIR (Shao et al.,
2025), Rank-1 (Weller et al., 2025), and Rank-
R1 (Zhuang et al., 2025), which refine the initial
rankings by modeling deeper interactions between
the query and candidate documents.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020b) is a hybrid paradigm that enhances
the generative capabilities of large language mod-
els (LLMs) by incorporating non-parametric exter-
nal knowledge during inference. This design mit-
igates well-documented limitations of standalone
LLMs, including hallucinations, confident but in-
correct outputs, and inability to reflect up-to-date
or domain-specific information due to static pre-
training (Hurst et al., 2024; Grattafiori et al., 2024;
Yang et al., 2025a).
RAG introduces an explicit retrieval step: for
a query q, a retriever selects a set of top- kdocu-
ments {d1, . . . , d k}from an external corpus. A
generator Gthen conditions on both qand the
retrieved context to produce the output a=
G(q, d1, . . . , d k)where Gis typically an encoder-
decoder or decoder-only LLM fine-tuned to inte-
grate retrieved evidence into its generation process.
2.3 Challenges in RAG
A central challenge in RAG is that generation qual-
ity is tightly coupled with retrieval quality. If rel-
evant (i.e., answer-containing) documents are not
retrieved, or if irrelevant context is included, the
generator is prone to producing incorrect or mis-
leading outputs (Cuconasu et al., 2024, 2025; Liu
et al., 2024).
Consequently, improving the top- kretrieval re-
sults is crucial. This can be viewed both as a se-
lection problem (how to retrieve the most relevant
documents) and a filtering problem (how to sup-
press distracting or noisy context). To this end,
several methods have been proposed that incorpo-
rate various forms of feedback , ranging from sim-
ple lexical overlap to more sophisticated agentic
or reasoning-based signals, to guide and refine the
retrieval process.
In this survey, we systematically categorize
these feedback mechanisms and analyze their effec-
tiveness across different components of the RAG
pipeline. We focus on how feedback is acquired,
represented, and integrated into retrieval, with the

Retrieval Reranker 
results Generation Query 
Encoder 
query 
    representation 
 new results 
Ranker 
Relevance 
Dynamic 
Retrieval Query 
Understanding 
Generated 
Output 
Feedback levels Query Level Retrieval Level Generation Time query 
Query Understanding Ranker Relevance Dynamic Retrieval 
- Query Expansion 
- Query Rewriting 
- Query Vector Adaption 
- Query Decomposition  - Result refinement   - When to retrieve 
- What (new query) to retrieve Figure 1: Illustration of feedback signals across the RAG pipeline. Feedback can modify the query (e.g., rewriting),
the retrieved pool (e.g., ranker-based expansion), or the generation loop (e.g., retrieval triggers based on uncertainty).
aim of providing a comprehensive taxonomy and
highlighting open research challenges.
3 Query-level feedback
We first focus on the first aspect which feedback in
RAG systems impact – the query . A fundamental
factor influencing the performance of RAG systems
is indeed the formulation of the input query. Poorly
phrased, underspecified, or ambiguous queries can
lead to irrelevant retrieval, ultimately degrading the
quality of the generated output.
To address this, a variety of feedback-driven
query reformulation methods have been proposed.
Feedback may be applied in one or multiple rounds
to iteratively enhance retrieval effectiveness and
overall answer quality. In this section, we focus
on the feedback improving query representations
and categorize them into two broad families based
on the source and representation of feedback: (i)
pseudo-relevance feedback from retrieved docu-
ments, and (ii) generative relevance feedback from
large language models.
3.1 Pseudo-Relevance Feedback (PRF)
Pseudo-relevance feedback (PRF) techniques mod-
ify queries based on the content of top- kretrieved
documents, assumed to be relevant. These meth-
ods operate either in the lexical space or in dense
embedding spaces.Lexical PRF. Classical PRF methods such as
RM3 (Jaleel et al., 2004), Rocchio (Rocchio, 1971),
and KL-divergence-based models (Zhai and Laf-
ferty, 2001) extract high-frequency terms from
pseudo-relevant documents to expand the origi-
nal query. These approaches rely on exact term
matching and term frequency statistics. Enhance-
ments like Latent Concept Expansion (LCE) (Met-
zler and Croft, 2007) and Local Context Anal-
ysis (LCA) (Xu and Croft, 1996) leverage co-
occurrence patterns or latent topic structures but
still operate in the discrete term space. While ef-
fective for certain domains, these methods are lim-
ited by the vocabulary mismatch problem : relevant
documents may not share terms with the query,
especially in low-resource or noisy scenarios.
Semantic PRF. To address lexical mismatch (Za-
mani and Croft, 2016, 2017) use word embeddings
to expand queries with semantically related terms.
More recent techniques adopt dense retrieval set-
tings – (Wang et al., 2023a) performs feedback-
based expansion in contextualized token embed-
ding space, while ANCE-PRF (Yu et al., 2021)
averages document embeddings to interpolate with
the query vector. These methods enable richer se-
mantic matching but remain sensitive to the ambi-
guity or sparsity of the original query (Jagerman
et al., 2023).

Key Insights
Expands queries using information from top-
ranked documents, improving recall but still
limited by vocabulary mismatch, query ambi-
guity, and noise in initial retrievals.
3.2 Generative Relevance Feedback (GRF)
Generative relevance feedback (GRF) methods em-
ploy large language models (LLMs) to generate
query expansions, reformulations, or conceptually
enriched representations. Unlike PRF, where feed-
back is extracted from retrieved documents, GRF
generates feedback via prompting, generation, or
learning signals.
LLM-only Feedback. Several methods prompt
pre-trained LLMs to produce reformulated or ex-
panded queries. QueryExpansion (QE) (Jagerman
et al., 2023) employs different prompting styles, in-
cluding Chain-of-Thought (CoT) prompting (Wei
et al., 2022), to elicit stepwise explanations and
derive new query terms. While these methods can
function without initial retrieval, many still use re-
trieved documents as input, which can introduce
noise. Hybrid systems such as MILL (Jia et al.,
2024), GRF+PRF (Mackie et al., 2023), and Blend-
Filter (Wang et al., 2024) combine lexical PRF
and GRF by verifying consistency between gener-
ated expansions and retrieved evidence. Word-level
filtering methods like Word2Passage (Choi et al.,
2025) and ReAL (Chen et al., 2025a) further refine
queries using token importance estimates.
Feedback from Generated Answers. Be-
yond generating expansions, some methods use
LLM-generated answers as implicit feedback.
Generation-Augmented Retrieval (GAR) (Mao
et al., 2021) generates answer-like contexts
(titles, passages, summaries) using a model like
BART (Lewis et al., 2020a), which are then con-
catenated to the query. However, this introduces
risks of hallucination and irrelevant additions.
To refine this idea, RRR (Arora et al., 2023)
iteratively updates the query based on retrieval
performance, using a feedback loop constrained
by a document budget. LameR (Shen et al., 2024)
first generates multiple answers, augments them
with the query, and performs a second retrieval
pass—effectively building a feedback loop from
generation to retrieval. InteR (Feng et al., 2024)
and Iter-RetGen (Shao et al., 2023) perform tighterintegration between RAG and GAR by alternating
between generation and retrieval for iterative
refinement.
Optimization-based Feedback. Recent work
aims to move beyond prompting heuristics by di-
rectly optimizing queries for retrieval objectives.
DeepRetrieval (Jiang et al., 2025) introduces a re-
inforcement learning framework where the query
generation process is trained end-to-end to maxi-
mize retrieval metrics (e.g., recall, nDCG), using
document-level reward signals. This eliminates
reliance on manual prompting or ground truth su-
pervision.
We refer readers to comprehensive surveys such
as (Song and Zheng, 2024) for broader coverage
of query rewriting and optimization techniques be-
yond the RAG context.
Key Insights
GRF methods use LLMs to generate or opti-
mize query reformulations, offering richer se-
mantics and adaptability, but are prone to hal-
lucination (irrelevant but plausible-sounding
terms) and require strategies to control noise.
4 Retrieval-level feedback
Retrieval in RAG pipelines is often bottlenecked by
the bounded recall of the first-stage retriever. Once
the top- kdocuments are selected, re-ranking can
improve their ordering, but cannot recover relevant
documents missed in the initial retrieval. This lim-
itation motivates adaptive retrieval methods that
incorporate feedback, often from neural rankers
or structural knowledge of the corpus, to refine
or expand the retrieved document set across one
or more rounds. In this section, we examine two
prominent classes of adaptive retrieval strategies,
neighborhood-based corpus expansion andquery
vector adaptation .
Neighborhood-based Corpus Expansion relies
on the clustering hypothesis that posits that co-
relevant documents tend to be similar to one an-
other. GraphAR (MacAvaney et al., 2022) formal-
izes this intuition by constructing a corpus graph
using lexical similarity between documents. Af-
ter reranking an initial retrieved set, the method
expands the document pool by including neigh-
bors of top-ranked documents in the graph. Vari-
ants such as LADR (Kulkarni et al., 2023) and

Cat.1 Approach Approach Description
Lexical Pseudo Relevance FeedbackQuery LevelLexical PRF (Jaleel et al., 2004) Expand queries using top-k document terms
Rocchio (Rocchio, 1971) Adjust vector using relevant feedback
KL Expansion (Zhai and Lafferty, 2001) Optimize query based on feedback documents
Adaptive Relevance Feedback (Lv and Zhai, 2009) Adaptive weights per query and feedback set
Relevance Modeling (Metzler and Croft, 2005) Interpolate query with new expansion terms
LCE (Metzler and Croft, 2007) Discover latent concepts for expansion
LCA (Xu and Croft, 1996) Use co-occurrence statistics for expansion
Semantic Pseudo Relevance Feedback
EQE (Zamani and Croft, 2016) Words with similar embeddings are used in query expansion
RLM/RPE (Zamani and Croft, 2017) Train a models to output words relevance
ANCE PRF (Yu et al., 2021) Expand using contrastive dense embeddings
Colbert PRF (Wang et al., 2023a) Contextual embedding expansion with late interaction
Generative Relevance Feedback
GRF (Mackie et al., 2023) Generate contexts with LLMs for queries
GAR (Mao et al., 2021) Expand using answer and passage metadata
QueryExpansion (Jagerman et al., 2023) Prompt-based query rewriting techniques
LameR (Shen et al., 2024) Append generated answers to original query
InteR (Feng et al., 2024) Alternate between generation and retrieval
Iter-Retgen (Shao et al., 2023) Interplay between GAR (or GRF) and RAG to improve answer generation
BlendFilter (Wang et al., 2024) Use both LLM-generated query and original query for retrieval
RRR (Arora et al., 2023) Interplay between GAR (or GRF) and RAG to improve retrieval
MILL (Jia et al., 2024) Use both PRF and GRF for query expansion
ReAL (Chen et al., 2025a) Learn original and expanding query terms weights
Word2Passage (Choi et al., 2025) Use granular word-level importance for query expansion
DeepRetrieval (Jiang et al., 2025) RL training to optimize the rewritten queryRetrieval LevelGraphAR (MacAvaney et al., 2022) Adaptive retrieval using a corpus graph
LADR (Kulkarni et al., 2023) Use lexical results for dense retrieval
QUAM (Rathee et al., 2025b) Adaptive retrieval using doc-doc similarities as feedback
LexBoost (Kulkarni et al., 2024) Improve lexical retrieval using semantic corpus graph
SUNAR (V et al., 2025) Use answer uncertainty as feedback
ORE (Rathee et al., 2025c) Dynamic documents selection for ranking
SlideGAR (Rathee et al., 2025a) Use LLM-based listwise ranker’s feedback for adaptive retrieval
ReFIT (Gangi Reddy et al., 2025) Update query vector using Ranker feedback
TOUR (Sung et al., 2023) Update query representation using ranker feedbackGeneration-TimeRule-Based Retrieval
SKR (Wang et al., 2023b) Ask LLM if information needed
IRCoT (Trivedi et al., 2023) Retrieve if CoT has not provided the final answer
Adaptive RAG (Jeong et al., 2024) Classifier’s feedback for retrieval
Retrieval-on-Demand via Feedback Signals
FLARE (Jiang et al., 2023) Token probability as feedback
DRAD (Su et al., 2024a) Check hallucination in answer and trigger retrieval to mitigate
DRAGIN (Su et al., 2024b) Token probability as feedback
Rowen (Ding et al., 2024) Answer consistency as feedback
SeaKR (Yao et al., 2025) Internal states of the LLM as feedback
CRAG (Yan et al., 2024) Use retrieval evaluator to judge if context is relevant
CoV-RAG (He et al., 2024) Chain-of-Verification using a trained model
SIM-RAG (Yang et al., 2025b) External critic model to judge if context is sufficient
Prompt-Based Methods
Self-Ask (Press et al., 2023) Decompose the complex query into sub-queries
DeComP (Khot et al., 2023) Decompose complex query into sub-queries
ReAct (Yao et al., 2022) Use each reasoning step to trigger retrieval
Searchain (Xu et al., 2024) Generate chain-of-questions and trigger retrieval if needed
MCTS-RAG (Hu et al., 2025) Dynamically integrates reasoning and retrieval in MCTS
SMR (Lee et al., 2025) Mitigates overthinking in retrieval by guiding LLMs through discrete actions
Learned or Agentic Methods
Self-RAG (Asai et al., 2024) Train LLM to predict reflection tokens that trigger retrieval and judge context
Search-R1 (Jin et al., 2025) Train LLM to decompose query and generate tokens that trigger retrieval
Search-O1 (Li et al., 2025a) Decide autonomously when to retrieve by detecting the presence of uncertain words
R1-Searcher (Song et al., 2025) Reward for triggering search tokens
ReZero (Dao and Le, 2025) Introduces an RL framework that rewards the act of retrying search queries
DeepResearcher (Zheng et al., 2025) Use F1 score-based reward for answer accuracy
WebThinker (Li et al., 2025b) Adapt model to use commercial search engines during training
ZeroSearch (Sun et al., 2025) Approximate the real search engine behavior during training
Table 1: Summary of feedback-based retrieval and RAG methods.

LexBoost (Kulkarni et al., 2024) improve effi-
ciency by using dense bi-encoders and incorpo-
rating query-document and document-document
edges. QUAM (Rathee et al., 2025b) general-
izes these approaches by introducing query affin-
ity modeling, taking into account the degree of
similarity between neighbors and their relevance.
The ORE framework (Rathee et al., 2025c) fur-
ther refines this strategy by prioritizing expanded
documents based on their expected utility toward
the ranker’s final relevance. SUNAR (V et al.,
2025) incorporates uncertainty over multiple LLM-
generated answers to adjust retrieval weights, offer-
ing a feedback loop grounded in generation uncer-
tainty, though it may amplify hallucinated answers.
SlideGAR (Rathee et al., 2025a) uses LLM-based
listwise rankers (Pradeep et al., 2023b,a) to itera-
tively expand and refine the document pool over a
document graph, closing the loop between ranking,
selection, and feedback-driven retrieval.
Query Vector Adaptation updates the query rep-
resentation based on feedback from ranked doc-
uments. ReFIT (Gangi Reddy et al., 2025) and
TOUR (Sung et al., 2023) both adjust the query vec-
tor in dense retrieval space using intermediate rele-
vance scores from neural rankers. These adapted
queries are used to perform second-stage retrieval,
improving coverage of relevant documents.
Key Insights
Relevance feedback improves recall via effi-
cient corpus expansion or query adaptation,
but risks adding noise when similarity links or
feedback are unreliable.
5 Generation-time feedback
RAG systems face two fundamental challenges:
determining when to retrieve external knowledge,
since not all queries benefit from it, and how to re-
trieve relevant content effectively (Su et al., 2024b).
Classical RAG pipelines rigidly follow a fixed se-
quence of retrieval, optionally ranking, followed
by generation, limiting their ability to adapt to the
context or task. To address these limitations, recent
work has introduced adaptive RAG , where the re-
trieval strategy is dynamically adjusted according
to the query, the model feedback, or the complexity
of the task. We categorize this emerging line of
work into three main classes.5.1 Rule-Based and Discriminative
Approaches
In-Context RALM (Retrieval-Augmented Lan-
guage Model) (Ram et al., 2023) proposes retriev-
ing relevant context documents during inference
at fixed intervals (every stokens, known as the re-
trieval stride), using the last ltokens of the input as
the retrieval query. In a similar spirit, IRCoT (In-
terleaving Retrieval in a CoT) (Trivedi et al., 2023)
dynamically retrieves documents if the CoT (Wei
et al., 2022) step has not provided the answer. At
first, it uses the original question to retrieve the
context and then uses the last generated CoT sen-
tence as a query for subsequent retrieval. How-
ever, both of these methods retrieve the context
regardless of whether the LLM needs external con-
text or not. Hence, the unnecessary retrieval steps
add additional latency cost during answer genera-
tion. Also, the noisy retrieved context can lead to
a wrong answer. CtRLA (Huanshuo et al., 2025)
devises a latent space probing-based approach for
making decisions regarding retrieval timings for
adaptive retrieval augmented generation. The au-
thors extract latent vectors that represent abstract
concepts like honesty andconfidence and use these
dimensions to steer retrieval and control LLM be-
havior, leading to better performance and robust
answers. To overcome the over-retrieval limitation
of rule-based dynamic RAG methods, retrieval-on-
demand approaches have been proposed. These
methods trigger retrieval only when the LLM needs
it, based on either external feedback (Section 5.2)or
the LLM’s own assessment (Section 5.3).
Key Insights
The rule-based methods help in answer gen-
eration based on the retrieved context. How-
ever, these rules-based retrieval results in over-
retrieval and add latency costs, and may pro-
vide noisy context, which can result in a wrong
answer.
5.2 Retrieval-on-Demand via Feedback
Signals
The feedback signals can come from different
sources, including the answer uncertainty, the
model’s internal states, or context faithfulness and
sufficiency. SKR (Wang et al., 2023b) asks LLM
itself if additional information is needed to answer
the query. If yes, then the retrieval round is trig-
gered; otherwise, the answer is generated from the

LLM’s internal knowledge. However, the judg-
ment is solely based on LLM, and without context,
they try to be overconfident (Xiong et al., 2024).
FLARE (Jiang et al., 2023) retrieves the documents
only if the token probability is below a predefined
threshold and uses the last generated sentence as a
query for retrieval (excluding the uncertain tokens)
and generates the response until the next uncer-
tain token or completion is done. However, these
uncertain tokens are not equally important to trig-
ger a retrieval round. Based on this, DRAD (Su
et al., 2024a) uses an external module for hallucina-
tion detection on entities in the generated answer;
if the answer contains hallucination, the retrieval
is triggered. The last generated sentence (with-
out a hallucinated entity) is used as a query for
retrieval. However, the choice of the new query
for retrieval relies on heuristic strategies. Since
the model’s information needs may extend beyond
the last sentence or CoT, it could require context
from a broader span of the generation to effectively
build confidence. Based on this motivation, DRA-
GIN (Su et al., 2024b), similar to FLARE, also
considers the token probabilities as a criterion of
the retrieval round but does not consider the un-
certain tokens as a part of the new query. Further,
it also reformulates the query using the keywords
based on the model’s internal attention weights and
reasoning. SeaKR (Yao et al., 2025) computes the
self-aware uncertainty using internal states of the
LLM. If the uncertainty is above a threshold, then
a retrieval round is triggered.
Other types of works, like Rowen (Ding et al.,
2024), consider the LLM’s answer consistency as
feedback. Rowen considers answer consistency
across languages of the same question with seman-
tically similar variations, and the consistency over
answers generated by different LLMs. If the total
consistency is below a predefined threshold, then
the retrieval round is triggered. However, similar
to SUNAR (V et al., 2025), the consistency can be
toward wrong answers.
However, these approaches consider all queries
equally complex and might end up with noisy con-
text retrieval and hence a wrong answer. Adap-
tive RAG (Jeong et al., 2024) uses a query routing
mechanism that predicts whether the query needs
retrieval or not. Further, it also decides on the num-
ber of retrieval rounds based on query complex-
ity. However, it assumes that the retrieved context
is relevant to the query without assessing its rele-
vancy or sufficiency. Towards the relevancy, CRAG(Corrective RAG) (Yan et al., 2024) evaluates the
relevance scores using a fine-tuned model, and clas-
sifies the retrieved document into correct, incorrect,
and ambiguous. If the context is not correct, then a
rewritten query is issued to the web search engine.
Similar fashion, SIM-RAG (Yang et al., 2025b) fo-
cuses on the context sufficiency angle, and trains
a lightweight critic model that provides feedback
if the retrieved context is sufficient to generate the
answer. If the information is not sufficient, then a
new query is formulated using the original query
and the already retrieved context, and a retrieval
round is triggered. Further CoV-RAG (He et al.,
2024) identifies errors, including reference and an-
swer correctness, and truthfulness, and then scores
them using a trained verifier. Based on the scores,
either provide a final or rewrite the query and do a
further retrieval round.
Key Insights
The external feedback signals help in reduc-
ing retrieval rounds. These signals can come
from different sources, at the LLM level (e.g.,
token generation confidence), at the answer
level ( e.g., uncertainty or hallucination), and
at the context level (e.g., relevancy or suffi-
ciency). However, these methods may still
retrieve noisy or irrelevant context, and com-
plexity assessment remains a challenge.
5.3 Self-Triggered Retrieval via Reasoning
In this section, we discuss works where LLM au-
tonomously makes the decision on when to retrieve
and how to retrieve through query decomposition
or planning-based approaches without external trig-
gers. These approaches are also termed Reasoning
RAG orAgentic RAG . These approaches can be
divided into mainly two categories: first, where
the instructions for query decomposition, when
to retrieve, and what to retrieve are provided in
the prompt along with few-shot examples; second,
where the language models are trained to decide by
themselves whether to decompose the query, when
to retrieve, and what to retrieve.
Prompt-Based Methods. DeComP (Khot et al.,
2023) divides a task into granular sub-tasks and
delegates them to different components through ac-
tions. However, DeComP only acts as a trigger for
when to retrieve and employs a BM25 retriever for
getting relevant documents in a single shot. It does

not subsequently generate reasoning steps to im-
prove retrieval, thus not providing much indication
as to how to retrieve. ReAct (Yao et al., 2022) in-
terleaves the generation of verbal reasoning traces
with actions that interact with the external environ-
ment. The verbal reasoning traces act as indica-
tors of how to retrieve, and the actions themselves
serve as triggers (when to retrieve). Similarly, Self-
Ask (Press et al., 2023) proposes to decompose the
original complex query into simpler sub-questions
iteratively interleaved by a retrieval step and inter-
mediate answer generation. At each step, the LLM
makes a decision to generate a follow-up question if
more information is needed, or it may generate the
final answer. Authors observed that this approach
helped cover diverse aspects of complex queries
and improved search and downstream answering
performance.
However, these approaches do not have provi-
sion for correction of the entire reasoning trajec-
tory, and an intermediate error may cause cascading
failures. Searchain (Xu et al., 2024) proposes to
mitigate this by constructing a global reasoning
chain first, where each node comprises a retrieval-
oriented query, an answer from LLM to the query,
and a flag indicating if additional knowledge is
needed to arrive at a better answer. SMR (State
Machine Reasoning) (Lee et al., 2025) identifies
the issues of the CoT-based query decomposition
and retrieval methods like ReAct (Yao et al., 2022),
where the CoT might result in redundant reasoning
(new queries that result in the retrieval of the same
documents) and misguided reasoning (new query
diverges from the user’s intent). To address these
limitations, SMR proposes three actions: Refine,
Rerank, and Stop. Action Refine updates the query
using the feedback from the already retrieved docu-
ments, and a retrieval round is triggered. Then the
retrieved documents are ranked according to the
old query to make sure only the relevant informa-
tion is used to answer. Finally, the Stop action is
called to stop the reasoning if a sufficient retrieval
quality is achieved, which helps in token efficiency
and prevents overthinking.
MCTS-RAG (Hu et al., 2025) combines
Monte Carlo Tree Search (MCTS) with Retrieval-
Augmented Generation (RAG) to improve reason-
ing and retrieval in language models. It guides the
search for relevant information using MCTS to ex-
plore promising retrieval paths, enhancing answer
accuracy. However, it is computationally expensive
due to the iterative tree search process and maystruggle with highly noisy or irrelevant documents.
Search-O1 (Li et al., 2025a) proposes an agentic
search workflow for reasoning augmented retrieval
by letting the Large Reasoning Models (LRMs)
like O1 decide autonomously when to retrieve by
detecting the presence of salient uncertain words in
their output. Additionally, they augment the work-
flow with a reason-in-documents step, where LRMs
analyze the documents in depth to remove noise
and reduce redundancy before employing them to
generate the final answer.
Key Insights
Query decomposition and interleaving reason-
ing with retrieval improve coverage for com-
plex questions by deciding when and what to
retrieve, but are prone to cascading errors and
redundant steps.
Learned or Agentic Methods. The Agentic
models go beyond prompt instructions and use
search/retrieval as a tool. These models are trained
to trigger this tool during answer generation. The
training process mainly focuses on giving rewards
for correct tool calls and context usage. In addition,
similar to RAG methods, the retrieved documents
are used as context to generate intermediate an-
swers or the final answer. The search tool might
have access to a local database or a web search
engine to retrieve up-to-date knowledge.
Self-RAG (Asai et al., 2024) trains to predict
reflection tokens for deciding when to retrieve and
for estimating the relevance of retrieved documents.
In addition, it judges the retrieved documents based
on the generated answers and their factuality. How-
ever, it can fail when its self-reflection misjudges
retrieval needs or relevance, leading to missed in-
formation or reliance on irrelevant context.
Search-R1 (Jin et al., 2025) is an extension of
the DeepSeek-R1 (Guo et al., 2025) model, where
the retrieval is a component training process. It
autonomously generates search queries and per-
forms real-time retrieval during step-by-step rea-
soning processes through reinforcement learning,
including GRPO and PPO. The retrieval is trig-
gered by <search> and</search> tokens, and
the retrieved context is enclosed in <information>
and </information> tokens. Similarly, R1-
Searcher (Song et al., 2025) also uses an RL frame-
work and uses two-stage rewards. The first stage
has a retrieval reward that helps the model to use

the correct format to trigger the retrieval, and the
second stage has an answer reward that encourages
the model to learn to utilize external retrieval effec-
tively. While both these methods encourage better
integration of external knowledge, they still inherit
the limitations of retrieval latency and potential
noise from the search source.
ReZero (Retry-Zero) (Dao and Le, 2025) intro-
duces an RL framework that rewards the act of
retrying search queries following an unsuccessful
initial attempt, and it encourages LLM to explore
alternative queries rather than prematurely stop-
ping. The training process provides positive sig-
nals/rewards (feedback) if the model executes a
retry action after failed searches, teaching the phi-
losophy of "try one more time". However, these
local database-based searches might miss the up-
to-date knowledge and could generate answers for
queries that require such knowledge.
DeepResearcher (Zheng et al., 2025) and Web-
Thinker (Li et al., 2025b) interact in real time with
commercial search engines during training, which
leads to noisy context from the web (since the qual-
ity of these documents is unpredictable) and a high
number of API calls. To address these limitations,
ZeroSearch (Sun et al., 2025) argues that since
LLM has acquired enough world knowledge dur-
ing heavy pre-training, it does not need to use a
search engine during training. Since the LLM it-
self can generate a good-quality document from its
parametric memory that answers the query, as well
as noisy documents. Hence, it can approximate
the real search engine behavior during training and
reduce the training costs and noise, but its effective-
ness depends on the LLM’s pre-trained knowledge.
Verifier-Based Feedback. Re2Search++ (Xiong
et al., 2025) proposes a fine-tuned critic model that
verifies intermediate answers and provides feed-
back based on its correctness to improve the quality
of intermediate queries and retrieval.
Key Insights
Learned or Agentic methods train models to
decide when and how to retrieve, boosting au-
tonomy and integration, but introduce retrieval
latency, noise, and dependence on external
web search. In addition, the use of an external
web search engine makes it difficult to evaluate
the retrieval performance.6 Datasets and Evaluation Benchmarks
The evaluation of IR and RAG systems relies on di-
verse datasets that test different aspects of retrieval
and answer generation capabilities, including the
retrieval, ranking, and answer quality.
6.1 IR Specific
Information retrieval has a long history of eval-
uation campaigns, including those from TREC,
CLEF, NTCIR, and FIRE. The queries used in
these collections are often developed to strike a
balance between having too many relevant docu-
ments in the target collection (which can be too
easy to retrieve and too difficult to properly anno-
tate (V oorhees et al., 2022)) and too few relevant
documents. Sometimes challenging topics are also
developed deliberately (V oorhees, 2005). The topic
development process often involves manual refor-
mulation of queries to ensure good coverage of
relevance assessments. Mirroring the annotation
process itself, it can be beneficial for automated
retrieval systems to also perform various forms of
query understanding (expansion or rewriting) to
help ensure high recall. Hence, the approaches
described in Section 3 show performance gains.
Comprehensive benchmarks like BEIR (Thakur
et al., 2021) offer heterogeneous evaluation across
17 diverse datasets spanning multiple domains
and retrieval tasks, enabling zero-shot general-
ization assessment. In addition, evaluation sets
such as TREC DL (Deep Learning) and its vari-
ants (Craswell et al., 2020, 2021, 2022, 2023) are
used to evaluate IR systems.
6.2 Question Answering
Question Answering datasets for retrieval eval-
uation fall into two primary categories based
on complexity. Single-hop QA benchmarks in-
clude Natural Questions (NQ) (Kwiatkowski et al.,
2019), TriviaQA (Joshi et al., 2017), SQuAD (Ra-
jpurkar et al., 2016), and PopQA (Mallen et al.,
2023) test the RAG system’s ability to retrieve
and utilize information to generate the final an-
swer. Multi-hop QA benchmarks comprise 2Wiki-
MultiHopQA (Ho et al., 2020) (questions requir-
ing evidence from exactly two Wikipedia articles
via bridging/comparison), HotpotQA (Yang et al.,
2018), and MuSiQue (Trivedi et al., 2022), which
features compositionally complex questions con-
structed from interconnected sub-questions. Due
to the dependency on intermediate answers, such

questions cannot be answered through isolated
single-step retrieval. Therefore, this design nat-
urally motivates the RAG systems to decompose
the original query and new queries based on the
intermediate answers help in the retrieval of better
context.
6.3 Fact Checking/Verification
These datasets assess models’ ability to verify
claims against retrieved evidence. FEVER (Fact
Extraction and VERification) (Thorne et al., 2018)
is an open-domain fact-checking benchmark that re-
quires retrieval over a large collection of Wikipedia
articles and training NLI models to classify claims
as supported, refuted, or not having enough in-
formation based on Wikipedia evidence. While
FEVER deals with simple claims, HoVeR (Jiang
et al., 2020) proposes a fact-checking bench-
mark with claims that require multi-hop reason-
ing and multi-turn retrieval. QuanTemp (V et al.,
2024) is the first to propose a large open-domain
benchmark for fact-checking numerical claims. It
comprises claims that require interleaving claim
decomposition and retrieval based on the veri-
fication output of sub-claims. This relates to
query-level feedback and generation-time feed-
back. AveriTeC (Schlichtkrull et al., 2023) is a
real-world claim verification dataset that includes
diverse claim types with supporting or refuting evi-
dence gathered through web search, making it par-
ticularly valuable for evaluating RAG systems on
authentic fact-checking scenarios that require veri-
fication against potentially noisy or contradictory
web sources.
6.4 Complex Reasoning
Emerging benchmarks introduce novel relevance
criteria requiring awareness of reasoning struc-
ture. BRIGHT (SU et al., 2025) defines docu-
ment relevance not by topical alignment but by
whether passages contain logical constructs (de-
ductive steps, analogies, constraints) necessary to
derive answers. This challenges lexical retrievers
that lack inference-awareness. However, reasoning-
augmented retrieval methods, such as CoT (Chain-
of-Thought), have shown performance gains. Fur-
ther, there exist reasoning/agentic tasks like Deep
Research (Wu et al., 2025), GPQA (Rein et al.,
2023), MATH500 (Cobbe et al., 2021), which in-
volve access to the web search engine, have been
used in the RAG setting (Li et al., 2025a; Wei et al.,
2025). However, due to the absence of the cor-pus, it is hard to evaluate the retrieval performance.
Recent benchmarks like BrowseComp-Plus (Chen
et al., 2025b) provide a curated corpus for Deep Re-
search tasks and enable the evaluation of retrieval
performance.
6.5 Evaluation Metrics
Evaluating retrieval systems and Retrieval-
Augmented Generation (RAG) pipelines is critical
for ensuring the accuracy, relevance, and reliability
of generated responses. Retrieval evaluation
typically focuses on metrics such as recall@k,
precision, mean reciprocal rank (MRR), and hit
rate, which assess how effectively the system
retrieves pertinent documents or passages given a
query. In contrast, RAG evaluation is more holistic,
combining retrieval quality with generation fidelity
and coherence. Common approaches include mea-
suring answer correctness using exact match (EM)
or F1 score, assessing faithfulness to retrieved
evidence to detect hallucinations, and evaluating
relevance and fluency through human or automated
scoring (e.g., BLEU, ROUGE, or BERTScore).
Recent frameworks like RAGAS (ExplodingGradi-
ents, 2024), ARES (Saad-Falcon et al., 2024), and
CRUX (Ju et al., 2025) also emphasize end-to-end
evaluation, where the interplay between retrieval
accuracy and generation quality is analyzed to
identify bottlenecks—such as irrelevant documents
leading to incorrect answers—making compre-
hensive evaluation essential for diagnosing and
improving RAG system performance.
Challenges . The current RAG evaluation methods
mainly focus on the retrieval and final answer per-
formance. However, Reasoning RAG systems are
highly dependent on intermediate reasoning steps
and retrieval rounds. Therefore, it is also impor-
tant to consider additional evaluation dimensions
such as computational cost, efficiency, or number
of retrieval rounds.
7 Challenges and Future Directions
Despite recent advances, test-time corpus-level
feedback in RAG systems faces several key limita-
tions related to computational cost, feedback qual-
ity, decision-making, and evaluation.
Computational Cost of Adaptive Retrieval.
Many feedback-driven approaches involve costly
operations such as multiple retrieval rounds, re-
ranking with large models, or traversing corpus
graphs (Rathee et al., 2025b; Kulkarni et al., 2023;

Hu et al., 2025). These methods often apply uni-
formly across queries, regardless of complexity. Ef-
ficient strategies—e.g., lightweight rankers, selec-
tive triggering, or confidence-aware stopping (Jiang
et al., 2023; Yao et al., 2025)—are crucial to make
such systems viable at scale. Notably, recent work
shows that smaller models can achieve competitive
performance if given high-quality context (V et al.,
2025), highlighting the importance of retrieval effi-
ciency.
Noisy and Unstructured Corpus Feedback. Re-
trieved documents often contain redundant or irrele-
vant content, and most systems lack mechanisms to
assess document utility beyond relevance ranking.
Few methods exploit inter-document structure such
as semantic similarity or topical diversity (MacA-
vaney et al., 2022; Rathee et al., 2025b). Struc-
tured representations (e.g., retrieval graphs, clus-
ters) could improve feedback signals by enabling
more targeted document selection and filtering.
Lack of Feedback-Aware Decision Policies.
Many RAG systems perform fixed sequences of
retrieval and reformulation without explicit deci-
sion criteria for when feedback is sufficient or
which action (re-ranking, rewriting, reretrieval) to
take (Su et al., 2024b; Li et al., 2025a). Learning
retrieval control policies based on document-level
or generation-time signals is a promising but un-
derexplored direction.
Inadequate Evaluation of Feedback Behavior.
Existing benchmarks emphasize answer correct-
ness or static retrieval recall, but rarely measure
feedback effectiveness across rounds. Datasets of-
ten lack annotations for document utility, retrieval
iteration, or evidence sufficiency. Metrics that
credit systems for improving retrieval through feed-
back—e.g., via answer change, document set re-
finement, or reduced over-retrieval—are needed to
advance the field (Zheng et al., 2025).
We believe that tackling these challenges is es-
sential to make corpus-level feedback a robust and
efficient component of real-world RAG pipelines,
closing the loop between retrieval and reasoning in
complex language tasks.
Limitations
This survey focuses exclusively on test-time feed-
back mechanisms that involve interaction with
the corpus in Retrieval-Augmented Generation(RAG) systems. We refer to this as corpus-level
feedback —signals derived from retrieved docu-
ments, re-rankers, document-document relation-
ships, or other corpus-grounded structures. Several
related forms of feedback fall outside our scope.
First, we do not cover feedback mechanisms that
operate independently of the corpus—such as LLM
self-refinement, planning, or reasoning without
retrieval. For example, techniques that rewrite
queries based solely on model introspection (e.g.,
self-refine (Madaan et al., 2023)) without consult-
ing retrieved content are not considered corpus
feedback and are excluded.
Second, our focus is restricted to retrieval-centric
adaptation. We do not survey approaches that mod-
ify the generation module unless they directly in-
form or adapt retrieval via corpus-level signals.
Third, we do not cover training-time feedback or
methods that rely on offline supervised signals to
fine-tune retrievers. Our interest is in test-time
feedback mechanisms that dynamically update the
query, document pool, or ranking without modify-
ing model parameters.
References
Daman Arora, Anush Kini, Sayak Ray Chowdhury, Na-
garajan Natarajan, Gaurav Sinha, and Amit Sharma.
2023. Gar-meets-rag paradigm for zero-shot infor-
mation retrieval. CoRR , abs/2310.20158.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR 2024, Vienna, Austria, May
7-11, 2024 . OpenReview.net.
Xinran Chen, Ben He, Xuanang Chen, and Le Sun.
2025a. Not all terms matter: Recall-oriented adap-
tive learning for PLM-aided query expansion in open-
domain question answering. In Proceedings of the
63rd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , pages
22139–22151, Vienna, Austria. Association for Com-
putational Linguistics.
Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping
Nie, Kai Zou, Andrew Liu, Joshua Green, Kshama
Patel, Ruoxi Meng, Mingyi Su, Sahel Shari-
fymoghaddam, Yanxi Li, Haoran Hong, Xinyu
Shi, Xuye Liu, Nandan Thakur, Crystina Zhang,
Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025b.
Browsecomp-plus: A more fair and transparent eval-
uation benchmark of deep-research agent. arXiv
preprint arXiv:2508.06600 .
Jeonghwan Choi, Minjeong Ban, Minseok Kim, and
Hwanjun Song. 2025. Word2Passage: Word-level

importance re-weighting for query expansion. In
Findings of the Association for Computational Lin-
guistics: ACL 2025 , pages 8276–8296, Vienna, Aus-
tria. Association for Computational Linguistics.
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian,
Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias
Plappert, Jerry Tworek, Jacob Hilton, Reiichiro
Nakano, Christopher Hesse, and John Schulman.
2021. Training verifiers to solve math word prob-
lems. Preprint , arXiv:2110.14168.
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and
Daniel Campos. 2021. Overview of the trec 2020
deep learning track. Preprint , arXiv:2102.07662.
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel
Campos, and Jimmy Lin. 2022. Overview of the trec
2021 deep learning track. In Text REtrieval Confer-
ence (TREC) . NIST, TREC.
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel
Campos, Jimmy Lin, Ellen M. V oorhees, and Ian
Soboroff. 2023. Overview of the trec 2022 deep
learning track. In Text REtrieval Conference (TREC) .
NIST, TREC.
Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel
Campos, and Ellen M. V oorhees. 2020. Overview
of the trec 2019 deep learning track. Preprint ,
arXiv:2003.07820.
Florin Cuconasu, Simone Filice, Guy Horowitz, Yoelle
Maarek, and Fabrizio Silvestri. 2025. Do RAG
systems suffer from positional bias? CoRR ,
abs/2505.15561.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. In Proceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval , SIGIR ’24, page
719–729, New York, NY , USA. Association for Com-
puting Machinery.
Alan Dao and Thinh Le. 2025. Rezero: Enhancing
LLM search ability by trying one-more-time. CoRR ,
abs/2504.11001.
Hanxing Ding, Liang Pang, Zihao Wei, Huawei Shen,
and Xueqi Cheng. 2024. Retrieve only when it
needs: Adaptive retrieval augmentation for halluci-
nation mitigation in large language models. CoRR ,
abs/2402.10612.
ExplodingGradients. 2024. Ragas: Supercharge your
llm application evaluations. https://github.com/
explodinggradients/ragas .
Jiazhan Feng, Chongyang Tao, Xiubo Geng, Tao Shen,
Can Xu, Guodong Long, Dongyan Zhao, and Daxin
Jiang. 2024. Synergistic interplay between search
and large language models for information retrieval.
InProceedings of the 62nd Annual Meeting of theAssociation for Computational Linguistics (Volume 1:
Long Papers) , pages 9571–9583, Bangkok, Thailand.
Association for Computational Linguistics.
Revanth Gangi Reddy, Pradeep Dasigi, Md Arafat Sul-
tan, Arman Cohan, Avirup Sil, Heng Ji, and Han-
naneh Hajishirzi. 2025. A large-scale study of
reranker relevance feedback at inference. In Pro-
ceedings of the 48th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval , SIGIR ’25, page 3010–3014, New York,
NY , USA. Association for Computing Machinery.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models. CoRR , abs/2407.21783.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning. arXiv preprint
arXiv:2501.12948 .
Bolei He, Nuo Chen, Xinran He, Lingyong Yan,
Zhenkai Wei, Jinchang Luo, and Zhen-Hua Ling.
2024. Retrieving, rethinking and revising: The chain-
of-verification can improve retrieval augmented gen-
eration. In Findings of the Association for Compu-
tational Linguistics: EMNLP 2024 , pages 10371–
10393, Miami, Florida, USA. Association for Com-
putational Linguistics.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. In Proceedings of the 28th Inter-
national Conference on Computational Linguistics ,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Yunhai Hu, Yilun Zhao, Chen Zhao, and Arman Cohan.
2025. MCTS-RAG: enhancing retrieval-augmented
generation with monte carlo tree search. CoRR ,
abs/2503.20757.
Liu Huanshuo, Hao Zhang, Zhijiang Guo, Jing Wang,
Kuicai Dong, Xiangyang Li, Yi Quan Lee, Cong
Zhang, and Yong Liu. 2025. CtrlA: Adaptive
retrieval-augmented generation via inherent control.
InFindings of the Association for Computational
Linguistics: ACL 2025 , pages 12592–12618, Vienna,
Austria. Association for Computational Linguistics.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Os-
trow, Akila Welihinda, Alan Hayes, Alec Radford,
and 1 others. 2024. Gpt-4o system card. CoRR ,
abs/2410.21276.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. In Proceedings of the 16th

Conference of the European Chapter of the Associ-
ation for Computational Linguistics: Main Volume ,
pages 874–880, Online. Association for Computa-
tional Linguistics.
Rolf Jagerman, Honglei Zhuang, Zhen Qin, Xuanhui
Wang, and Michael Bendersky. 2023. Query expan-
sion by prompting large language models. CoRR ,
abs/2305.03653.
Nasreen Abdul Jaleel, James Allan, W. Bruce Croft,
Fernando Diaz, Leah S. Larkey, Xiaoyan Li, Mark D.
Smucker, and Courtney Wade. 2004. Umass at
TREC 2004: Novelty and HARD. In Proceedings
of the Thirteenth Text REtrieval Conference, TREC
2004, Gaithersburg, Maryland, USA, November 16-
19, 2004 , volume 500-261 of NIST Special Publica-
tion. National Institute of Standards and Technology
(NIST).
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong Park. 2024. Adaptive-RAG: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. In Proceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers) , pages 7036–7050, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Pengyue Jia, Yiding Liu, Xiangyu Zhao, Xiaopeng Li,
Changying Hao, Shuaiqiang Wang, and Dawei Yin.
2024. MILL: Mutual verification with large language
models for zero-shot query expansion. In Proceed-
ings of the 2024 Conference of the North American
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume
1: Long Papers) , pages 2498–2518, Mexico City,
Mexico. Association for Computational Linguistics.
Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu Tian,
SeongKu Kang, Zifeng Wang, Jimeng Sun, and Ji-
awei Han. 2025. Deepretrieval: Hacking real search
engines and retrievers with large language models
via reinforcement learning. CoRR , abs/2503.00223.
Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles
Dognin, Maneesh Singh, and Mohit Bansal. 2020.
HoVer: A dataset for many-hop fact extraction and
claim verification. In Findings of the Association
for Computational Linguistics: EMNLP 2020 , pages
3441–3460, Online. Association for Computational
Linguistics.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-r1:
Training llms to reason and leverage search engines
with reinforcement learning. CoRR , abs/2503.09516.Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. In Proceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Jia-Huei Ju, Suzan Verberne, Maarten de Rijke, and
Andrew Yates. 2025. Controlled retrieval-augmented
context evaluation for long-form RAG. CoRR ,
abs/2506.20051.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 6769–6781,
Online. Association for Computational Linguistics.
Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao
Fu, Kyle Richardson, Peter Clark, and Ashish Sab-
harwal. 2023. Decomposed prompting: A modular
approach for solving complex tasks. In The Eleventh
International Conference on Learning Representa-
tions .
Hrishikesh Kulkarni, Nazli Goharian, Ophir Frieder,
and Sean MacAvaney. 2024. Lexboost: Improving
lexical document retrieval with nearest neighbors. In
Proceedings of the ACM Symposium on Document
Engineering 2024 , DocEng ’24, New York, NY , USA.
Association for Computing Machinery.
Hrishikesh Kulkarni, Sean MacAvaney, Nazli Goharian,
and Ophir Frieder. 2023. Lexically-accelerated dense
retrieval. In Proceedings of the 46th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, SIGIR 2023, Taipei,
Taiwan, July 23-27, 2023 , pages 152–162. ACM.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research. Transactions of the Association for Compu-
tational Linguistics , 7:452–466.
Dohyeon Lee, Yeonseok Jeong, and Seung-won Hwang.
2025. From token to action: State machine reason-
ing to mitigate overthinking in information retrieval.
CoRR , abs/2505.23059.
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
Veselin Stoyanov, and Luke Zettlemoyer. 2020a.
BART: Denoising sequence-to-sequence pre-training
for natural language generation, translation, and com-
prehension. In Proceedings of the 58th Annual Meet-
ing of the Association for Computational Linguistics ,
pages 7871–7880, Online. Association for Computa-
tional Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020b.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Proceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems , NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025a. Search-o1: Agentic search-enhanced
large reasoning models. Preprint , arXiv:2501.05366.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yu-
tao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng
Dou. 2025b. Webthinker: Empowering large rea-
soning models with deep research capability. CoRR ,
abs/2504.21776.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts. Transactions of the Association
for Computational Linguistics , 12:157–173.
Yuanhua Lv and ChengXiang Zhai. 2009. Adaptive rel-
evance feedback in information retrieval. In Proceed-
ings of the 18th ACM Conference on Information and
Knowledge Management , CIKM ’09, page 255–264,
New York, NY , USA. Association for Computing
Machinery.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. In Proceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval , SIGIR ’24, page
2421–2425, New York, NY , USA. Association for
Computing Machinery.
Sean MacAvaney, Nicola Tonellotto, and Craig Mac-
donald. 2022. Adaptive re-ranking with a corpus
graph. In Proceedings of the 31st ACM International
Conference on Information & Knowledge Manage-
ment, Atlanta, GA, USA, October 17-21, 2022 , pages
1491–1500. ACM.
Iain Mackie, Shubham Chatterjee, and Jeffrey Dalton.
2023. Generative and pseudo-relevant feedback for
sparse, dense and learned sparse retrieval. CoRR ,
abs/2305.07477.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdan-
bakhsh, and Peter Clark. 2023. Self-refine: Itera-
tive refinement with self-feedback. In Advances in
Neural Information Processing Systems , volume 36,
pages 46534–46594. Curran Associates, Inc.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822, Toronto,
Canada. Association for Computational Linguistics.
Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong
Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen.
2021. Generation-augmented retrieval for open-
domain question answering. In Proceedings of the
59th Annual Meeting of the Association for Compu-
tational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Vol-
ume 1: Long Papers) , pages 4089–4100, Online. As-
sociation for Computational Linguistics.
Donald Metzler and W. Bruce Croft. 2005. A markov
random field model for term dependencies. In Pro-
ceedings of the 28th Annual International ACM SI-
GIR Conference on Research and Development in
Information Retrieval , SIGIR ’05, page 472–479,
New York, NY , USA. Association for Computing
Machinery.
Donald Metzler and W. Bruce Croft. 2007. Latent con-
cept expansion using markov random fields. In Pro-
ceedings of the 30th Annual International ACM SI-
GIR Conference on Research and Development in
Information Retrieval , SIGIR ’07, page 311–318,
New York, NY , USA. Association for Computing
Machinery.
Rodrigo Frassetto Nogueira, Zhiying Jiang, Ronak
Pradeep, and Jimmy Lin. 2020. Document ranking
with a pretrained sequence-to-sequence model. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2020, Online Event, 16-20 Novem-
ber 2020 , volume EMNLP 2020 of Findings of ACL ,
pages 708–718. Association for Computational Lin-
guistics.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023a. Rankvicuna: Zero-shot listwise doc-
ument reranking with open-source large language
models. CoRR , abs/2309.15088.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023b. Rankzephyr: Effective and robust
zero-shot listwise reranking is a breeze! CoRR ,
abs/2312.02724.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah Smith, and Mike Lewis. 2023. Measuring and
narrowing the compositionality gap in language mod-
els. In Findings of the Association for Computational
Linguistics: EMNLP 2023 , pages 5687–5711, Singa-
pore. Association for Computational Linguistics.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. In Proceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing , pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Mandeep Rathee, Sean MacAvaney, and Avishek Anand.
2025a. Guiding retrieval using llm-based listwise
rankers. In Advances in Information Retrieval -
47th European Conference on Information Retrieval,
ECIR 2025, Lucca, Italy, April 6-10, 2025, Proceed-
ings, Part I , volume 15572 of Lecture Notes in Com-
puter Science , pages 230–246. Springer.
Mandeep Rathee, Sean MacAvaney, and Avishek Anand.
2025b. Quam: Adaptive retrieval through query affin-
ity modelling. In Proceedings of the Eighteenth ACM
International Conference on Web Search and Data
Mining , WSDM ’25, page 954–962, New York, NY ,
USA. Association for Computing Machinery.
Mandeep Rathee, Venktesh V , Sean MacAvaney, and
Avishek Anand. 2025c. Breaking the lens of the
telescope: Online relevance estimation over large
retrieval sets. In Proceedings of the 48th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval , SIGIR ’25, page
2287–2297, New York, NY , USA. Association for
Computing Machinery.
David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien Di-
rani, Julian Michael, and Samuel R. Bowman. 2023.
GPQA: A graduate-level google-proof q&a bench-
mark. CoRR , abs/2311.12022.
S.E. Robertson, S. Walker, and M.M. Hancock-Beaulieu.
1995. Large test collection experiments on an opera-
tional, interactive system: Okapi at trec. Information
Processing & Management , 31(3):345–360. The Sec-
ond Text Retrieval Conference (TREC-2).
JJ Rocchio. 1971. Relevance feedback in information
retrieval. The SMART Retrieval System-Experiments
in Automatic Document Processing/Prentice Hall .
Jon Saad-Falcon, Omar Khattab, Christopher Potts, and
Matei Zaharia. 2024. ARES: An automated evalua-
tion framework for retrieval-augmented generation
systems. In Proceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 1: Long Papers) , pages 338–354,
Mexico City, Mexico. Association for Computational
Linguistics.
Michael Schlichtkrull, Zhijiang Guo, and Andreas Vla-
chos. 2023. Averitec: A dataset for real-world claim
verification with evidence from the web. In Ad-
vances in Neural Information Processing Systems ,
volume 36, pages 65128–65167. Curran Associates,
Inc.
Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muen-
nighoff, Xi Victoria Lin, Daniela Rus, Bryan
Kian Hsiang Low, Sewon Min, Wen-tau Yih,Pang Wei Koh, and Luke Zettlemoyer. 2025. Rea-
sonir: Training retrievers for reasoning tasks. CoRR ,
abs/2504.20595.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language models
with iterative retrieval-generation synergy. In Find-
ings of the Association for Computational Linguis-
tics: EMNLP 2023 , pages 9248–9274, Singapore.
Association for Computational Linguistics.
Tao Shen, Guodong Long, Xiubo Geng, Chongyang Tao,
Yibin Lei, Tianyi Zhou, Michael Blumenstein, and
Daxin Jiang. 2024. Retrieval-augmented retrieval:
Large language models are strong zero-shot retriever.
InFindings of the Association for Computational Lin-
guistics: ACL 2024 , pages 15933–15946, Bangkok,
Thailand. Association for Computational Linguistics.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
CoRR , abs/2503.05592.
Mingyang Song and Mao Zheng. 2024. A survey of
query optimization in large language models. CoRR ,
abs/2412.17558.
Hongjin SU, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Liu Haisu, Quan
Shi, Zachary Siegel, Michael Tang, Ruoxi Sun, Jin-
sung Yoon, Sercan Arik, Danqi Chen, and Tao Yu.
2025. Bright: A realistic and challenging bench-
mark for reasoning-intensive retrieval. In Interna-
tional Conference on Representation Learning , vol-
ume 2025, pages 48941–48991.
Weihang Su, Yichen Tang, Qingyao Ai, Changyue
Wang, Zhijing Wu, and Yiqun Liu. 2024a. Miti-
gating entity-level hallucination in large language
models. In Proceedings of the 2024 Annual Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval in the Asia Pa-
cific Region , SIGIR-AP 2024, page 23–31, New York,
NY , USA. Association for Computing Machinery.
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024b. DRAGIN: Dynamic retrieval
augmented generation based on the real-time informa-
tion needs of large language models. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 12991–13013, Bangkok, Thailand. Association
for Computational Linguistics.
Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan
Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei
Huang, and Jingren Zhou. 2025. Zerosearch: Incen-
tivize the search capability of llms without searching.
CoRR , abs/2505.04588.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is ChatGPT good at search?

investigating large language models as re-ranking
agents. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 14918–14937, Singapore. Association for
Computational Linguistics.
Mujeen Sung, Jungsoo Park, Jaewoo Kang, Danqi Chen,
and Jinhyuk Lee. 2023. Optimizing test-time query
representations for dense retrieval. In Findings of
the Association for Computational Linguistics: ACL
2023 , pages 5731–5746, Toronto, Canada. Associa-
tion for Computational Linguistics.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. In Thirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2) .
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. In Proceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers) , pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics , 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of
the 61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers) ,
pages 10014–10037, Toronto, Canada. Association
for Computational Linguistics.
Venktesh V , Abhijit Anand, Avishek Anand, and Vinay
Setty. 2024. Quantemp: A real-world open-domain
benchmark for fact-checking numerical claims. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , SIGIR ’24, page 650–660, New
York, NY , USA. Association for Computing Machin-
ery.
Venktesh V , Mandeep Rathee, and Avishek Anand. 2025.
SUNAR: Semantic uncertainty based neighborhood
aware retrieval for complex QA. In Proceedings of
the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Lin-
guistics: Human Language Technologies (Volume 1:
Long Papers) , pages 5818–5835, Albuquerque, New
Mexico. Association for Computational Linguistics.
Ellen M. V oorhees. 2005. The TREC robust retrieval
track. SIGIR Forum , 39(1):11–20.Ellen M. V oorhees, Nick Craswell, and Jimmy Lin.
2022. Too many relevants: Whither cranfield test
collections? In SIGIR ’22: The 45th International
ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval, Madrid, Spain, July
11 - 15, 2022 , pages 2970–2980. ACM.
Haoyu Wang, Ruirui Li, Haoming Jiang, Jinjin Tian,
Zhengyang Wang, Chen Luo, Xianfeng Tang, Mon-
ica Xiao Cheng, Tuo Zhao, and Jing Gao. 2024.
BlendFilter: Advancing retrieval-augmented large
language models via query generation blending and
knowledge filtering. In Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language
Processing , pages 1009–1025, Miami, Florida, USA.
Association for Computational Linguistics.
Xiao Wang, Craig MacDonald, Nicola Tonellotto, and
Iadh Ounis. 2023a. Colbert-prf: Semantic pseudo-
relevance feedback for dense passage and document
retrieval. ACM Trans. Web , 17(1).
Yile Wang, Peng Li, Maosong Sun, and Yang Liu.
2023b. Self-knowledge guided retrieval augmenta-
tion for large language models. In Findings of the
Association for Computational Linguistics: EMNLP
2023 , pages 10303–10315, Singapore. Association
for Computational Linguistics.
Jason Wei, Zhiqing Sun, Spencer Papay, Scott McK-
inney, Jeffrey Han, Isa Fulford, Hyung Won Chung,
Alex Tachard Passos, William Fedus, and Amelia
Glaese. 2025. Browsecomp: A simple yet chal-
lenging benchmark for browsing agents. CoRR ,
abs/2504.12516.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten
Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le,
and Denny Zhou. 2022. Chain-of-thought prompt-
ing elicits reasoning in large language models. In
Advances in Neural Information Processing Systems ,
volume 35, pages 24824–24837. Curran Associates,
Inc.
Orion Weller, Kathryn Ricci, Eugene Yang, Andrew
Yates, Dawn J. Lawrie, and Benjamin Van Durme.
2025. Rank1: Test-time compute for reranking in
information retrieval. CoRR , abs/2502.18418.
Junde Wu, Jiayuan Zhu, and Yuyuan Liu. 2025. Agentic
reasoning: Reasoning llms with tools for the deep
research. CoRR , abs/2502.04644.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang,
Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing
Song, Dengyu Wang, Minjia Zhang, Zhiyong Lu,
and Aidong Zhang. 2025. Rag-gym: Systematic opti-
mization of language agents for retrieval-augmented
generation. Preprint , arXiv:2502.13957.
Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu,
Junxian He, and Bryan Hooi. 2024. Can llms express
their uncertainty? an empirical evaluation of confi-
dence elicitation in llms. In The Twelfth International
Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net.

Jinxi Xu and W. Bruce Croft. 1996. Query expansion us-
ing local and global document analysis. In Proceed-
ings of the 19th Annual International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , SIGIR ’96, page 4–11, New York,
NY , USA. Association for Computing Machinery.
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024. Search-in-the-chain: In-
teractively enhancing large language models with
search for knowledge-intensive tasks. In The Web
Conference 2024 .
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective retrieval augmented generation.
CoRR , abs/2401.15884.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025a. Qwen3 technical report. arXiv preprint
arXiv:2505.09388 .
Diji Yang, Linda Zeng, Jinmeng Rao, and Yi Zhang.
2025b. Knowing you don’t know: Learning when
to continue search in multi-round rag through self-
practicing. In Proceedings of the 48th International
ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval , SIGIR ’25, page
1305–1315, New York, NY , USA. Association for
Computing Machinery.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Izhak Shafran,
Karthik R Narasimhan, and Yuan Cao. 2022. React:
Synergizing reasoning and acting in language models.
InNeurIPS 2022 Foundation Models for Decision
Making Workshop .
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao,
Linmei Hu, Liu Weichuan, Lei Hou, and Juanzi Li.
2025. SeaKR: Self-aware knowledge retrieval for
adaptive retrieval augmented generation. In Proceed-
ings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers) , pages 27022–27043, Vienna, Austria. Associa-
tion for Computational Linguistics.
HongChien Yu, Chenyan Xiong, and Jamie Callan. 2021.
Improving query representations for dense retrieval
with pseudo relevance feedback. In Proceedings of
the 30th ACM International Conference on Informa-
tion & Knowledge Management , CIKM ’21, page
3592–3596, New York, NY , USA. Association for
Computing Machinery.
Hamed Zamani and W. Bruce Croft. 2016. Embedding-
based query language models. In Proceedings of the2016 ACM International Conference on the Theory
of Information Retrieval , ICTIR ’16, page 147–156,
New York, NY , USA. Association for Computing
Machinery.
Hamed Zamani and W. Bruce Croft. 2017. Relevance-
based word embedding. In Proceedings of the 40th
International ACM SIGIR Conference on Research
and Development in Information Retrieval , SIGIR
’17, page 505–514, New York, NY , USA. Association
for Computing Machinery.
Chengxiang Zhai and John Lafferty. 2001. Model-based
feedback in the language modeling approach to infor-
mation retrieval. In Proceedings of the Tenth Inter-
national Conference on Information and Knowledge
Management , CIKM ’01, page 403–410, New York,
NY , USA. Association for Computing Machinery.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments. CoRR ,
abs/2504.03160.
Shengyao Zhuang, Xueguang Ma, Bevan Koopman,
Jimmy Lin, and Guido Zuccon. 2025. Rank-r1: En-
hancing reasoning in llm-based document rerankers
via reinforcement learning. CoRR , abs/2503.06034.

A Literature Compilation
A.1 Search Strategy
We conducted a comprehensive search on Google
Scholar. We first focused on highly relevant Nat-
ural language Processing (NLP) venues such as
ACL, EMNLP, NAACL, COLM and journals like
TACL to collect RAG related literature. We also
extensively curated papers from IR venues like
SIGIR, ECIR, CIKM, WSDM to cover informa-
tion retrieval literature and recent advancements in
RAG systems.
A.2 Compilation Strategy
After careful review of Abstract, Introduction, Con-
clusion and Limitations we only retained papers
that employ feedback mechanisms for improving
retrieval and other components of RAG system
which also helped synthesize our definition of feed-
back described in Section 3