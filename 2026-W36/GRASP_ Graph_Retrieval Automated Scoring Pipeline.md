# GRASP: Graph-Retrieval Automated Scoring Pipeline for Label-Free Multi-Topic Essay Grading

**Authors**: Aafreen Husain, Samar Shailendra, Saad Sajid Hashmi

**Published**: 2026-09-03 13:48:54

**PDF URL**: [https://arxiv.org/pdf/2609.03857v1](https://arxiv.org/pdf/2609.03857v1)

## Abstract
Automated short-answer grading research has historically focused on exams consisting solely of questions pertaining to a single topic. Automatic grading of exams containing questions about more than one topic remains less explored. In this work, a Graph-Retrieval Automated Scoring Pipeline (GRASP) is introduced for grading label-free multi-topic science exams. Label-free exams are short-answer exams in which a student's responses to several distinct topics are merged into a single paragraph, with no markup labels or segmentation indicating which span answers which question. Reference answers for each question are encoded into a FAISS vector index via Sentence-BERT, and a semantic similarity graph is constructed over this set of reference answers. At grading time, sentence count heuristics, with a large language model used to resolve ambiguous cases, are first applied to predict how many distinct topics were answered in the student essay. This process is performed without training data or domain-specific example essays. Candidate reference nodes, each storing one (question, reference answer, concatenation of both) from the reference index, are then retrieved through cosine similarity based Retrieval-Augmented Generation (RAG) and Graph Retrieval-Augmented Generation (GRAG). GRAG operates by taking the top cosine matches as seed nodes and then performing a graph traversal over strong edges to find additional reference nodes that may have been missed by RAG. The Hungarian algorithm is then used to optimally assign one reference node per question segment such that no reference is duplicated. Each segment is then graded against its assigned reference independently using GPT-4.1-mini. This experiment is performed to show the effect of retrieval quality on grading accuracy and the benefit of graph-augmented retrieval versus strict cosine similarity methods at various levels of essay complexity.

## Full Text


<!-- PDF content starts -->

This is a pre-print version of the paper accepted in ICONIP 2026.
GRASP: Graph-Retrieval Automated Scoring
Pipeline for Label-Free Multi-Topic Essay
Grading
Aafreen Husain, Samar Shailendra, and Saad Sajid Hashmi
Melbourne Institute of Technology, Australia
Abstract.Automatedshort-answergradingresearchhashistoricallyfo-
cused on exams consisting solely of questions pertaining to a single topic.
Automatic grading of exams containing questions about more than one
topic remains less explored. In this work, a Graph-Retrieval Automated
Scoring Pipeline (GRASP) is introduced for grading label-free multi-
topic science exams. Label-free exams are short-answer exams in which
a student’s responses to several distinct topics are merged into a single
paragraph, with no markup labels or segmentation indicating which span
answers which question. Reference answers for each question are encoded
into a FAISS vector index via Sentence-BERT, and a semantic similarity
graph is constructed over this set of reference answers. At grading time,
sentence count heuristics, with a large language model used to resolve
ambiguous cases, are first applied to predict how many distinct topics
were answered in the student essay. This process is performed without
training data or domain-specific example essays. Candidate reference
nodes, each storing one (question, reference answer, concatenation of
both) from the reference index, are then retrieved through cosine similar-
ity based Retrieval-Augmented Generation (RAG) and Graph Retrieval-
Augmented Generation (GRAG). GRAG operates by taking the top co-
sine matches as seed nodes and then performing a graph traversal over
strongedgestofindadditionalreferencenodesthatmayhavebeenmissed
by RAG. The Hungarian algorithm is then used to optimally assign one
reference node per question segment such that no reference is duplicated.
Each segment is then graded against its assigned reference independently
using GPT-4.1-mini. This experiment is performed to show the effect of
retrieval quality on grading accuracy and the benefit of graph-augmented
retrieval versus strict cosine similarity methods at various levels of essay
complexity.
Keywords:Automated essay scoring·Graph-augmented retrieval gen-
eration·Multi-topic grading·SBERT·Hungarian algorithm·Few-shot
learning.
1 Introduction
Consider an automated examination system that generates science exams. Each
question it can ask comes from a list of reference nodes it stores. Each node has
1
arXiv:2609.03857v1  [cs.IR]  3 Sep 2026

This is a pre-print version of the paper accepted in ICONIP 2026.
an exam question as well as the answer to that question. Questions will range
from Physics to Chemistry to Biology to Earth Science. To assess students’
understanding at the sub-topic level, the system builds a larger question of re-
latednsub-topics randomly selected from its database. For example, it might
pull three questions about mineral hardness and then paraphrase them into one
exam question. The student is given this question and must answer thentopics
in one running paragraph. There are no numbered parts, no section headings,
and no markers that can be used to tell where the response to one topic ends
and another begins.
This task is challenging because thentopics were all sampled from the same
section of the database and are thus grounded in the same scientific concept.
Their reference answers share wording and logic, and the embeddings of the
gold reference nodes lie close together in vector space. An untrained retrieval
model, when queried fornmatching references, may returnncopies of the same
reference rather thanndistinct ones. Retrieval thus faces a three-fold test:
(i) There is never explicit mention of how many topicsnthere are. This must
be parsed from the student’s submission and exam question.
(ii) Answers can contain multiple sentences, yet there are no annotations con-
necting a sentence to the topic it addresses. The system must determine
which sentence answers which topic based solely on semantics.
(iii) Sentences must be matched to the appropriate reference node from the pool
of reference nodes with no annotated guidance.
Existing automated essay scoring (AES) systems [1, 30] assume one topic
per response and score against a single known reference. They cannot segment
student’s answer tontopics asked in the question. RAG-based grading sys-
tems [2, 21] retrieve evidence before scoring but pull from a flat index with
no awareness of how references relate to each other. This causes retrieval accu-
racy to collapse when there are multiple related nodes which must all be found
simultaneously.
ThispaperintroducestheGRASPpipeline.Thispipelineaddressestheselim-
itations. First, GRASP pipeline detects how many topics question and student
answercontains.Itsplitstheanswerandthequestionintotopic-focusedsegments
so that each part can be graded against the reference it actually addresses. Sec-
ondly, it organises reference nodes into a graph that captures how they relate
to one another. This allows related references to be retrieved together rather
than from a flat, unaware index. Finally, it assigns each segment to its most
appropriate reference node through an optimal global matching step. This en-
sures that distinct but semantically similar references are kept apart rather than
collapsed into near-duplicates. Our main contribution is this end-to-end prob-
lem formulation for label-free multi-topic grading, together with the evaluation
methodology that separates retrieval quality from grading quality. The graph
expansion mechanism itself adapts existing GraphRAG-style retrieval [9, 14] to
this new setting rather than introducing a new retrieval algorithm.
The remainder of this paper is structured as follows. Section 2 reviews re-
lated work. Section 3 formally defines the multi-topic grading problem. Section 4
2

This is a pre-print version of the paper accepted in ICONIP 2026.
describes the GRASP pipeline. Section 5 presents results and analysis. Section 6
concludes with directions for future work.
2 Literature Survey
Researchonautomatedshort-answerandessaygradingspansfiveinterconnected
thematic areas. This survey covers foundational and recent work across AES
models, semantic similarity, RAG-based systems, graph-enhanced retrieval, and
LLM-based feedback generation.
2.1 Automated Essay Scoring (AES) Models
Transfer learning using pre-trained transformer models has dominated AES
research over the past decade for holistic scoring and multi-dimensional scor-
ing tasks. Xue et al. [30] introduce hierarchical BERT-based transfer learning
approach for multidimensional essay scoring which attained improvements of
4.5% Quadratic Weighted Kappa (QWK) on ASAP dataset and 8.1% on CELA
dataset by segmenting essays into hierarchical representations and attending
over segments using attention pooling. Amin et al. [1] show few-shot transformer
based AES can attain QWK=0.97 and QWK=0.94 for holistic scoring and con-
tent scoring respectively on ASAP with little supervision, but the model was
not interpretable and raised concerns regarding scoring bias. Wang et al. [28] in-
troduce multi-scale BERT which jointly learns document-level, token-level, and
segment-level representations via joint training to improve holistic scoring ac-
curacy on ASAP. Sun and Wang [26] use regression heads trained on top of
fine-tuned BERT models for scoring across multiple traits (e.g., content, organ-
isation, style) and show competitive results across different scoring dimensions.
Do et al. [5] introduce ArTS, an AES method that frames multi-trait AES as
an autoregressive sequence generation problem via T5. By generating scores one
trait at a time, they achieve state-of-the-art results on multiple benchmarks.
Cummins et al. [3] introduce constrained multi-task learning to AES, jointly
optimising for several scoring objectives under constraint and showing improved
generalisation to unseen prompts. Rodriguez et al. [24] explore the utility of
pre-trained language models for AES and conclude that contextual embeddings
significantly outperform feature engineered baselines, Elmassry et al. [10] pro-
vide a systematic review finding this trend to hold true across publications from
2019 to 2023.
2.2 Semantic Similarity Approaches
Semantic similarity is the representational paradigm employed by most state-
of-the-art short-answer grading systems. Reimers and Gurevych [23] created
Sentence-BERT (SBERT), a framework which fine-tunes BERT via a siamese
networkarchitecturetoobtaindensefixed-lengthsentenceembeddingsthatallow
for fast cosine similarity searches. The embedding model used for indexing and
3

This is a pre-print version of the paper accepted in ICONIP 2026.
retrieval in this work is SBERT. Patil and Agrawal [20] used siamese Bi-LSTM
networkswithattentionmechanismsappliedtoshort-answergrading.Theirwork
shows that siamese semantic similarity scoring can correctly pair reference an-
swers with student submissions when applied to narrowly defined domains. Gao
et al. [12] introduce SimCSE, a simple contrastive learning framework for sen-
tence embeddings that trains representation models with both supervised and
unsupervisedcontrastivelearningobjectives.KhattabandZaharia[17]introduce
ColBERT,anefficientpassageretrievalframeworkbuiltuponcontextualizedlate
interaction that allows dense retrieval to scale to large corpora. A disadvantage
of calculating sentence similarity via a scalar function such as SBERT is that we
are given no causal inference to understand how two passages may be related,
graph-augmented approaches seek to solve this shortcoming. Li et al. [19] pro-
pose a hybrid neural architecture for AES which extracts linguistic, semantic,
and structural attributes of an essay document which are fused over a BERT
encoder. Conversely, Janda et al. [16] showed syntactic, semantic, and sentiment
features contributed jointly to overall automated essay score. Vaswani et al. [27]
introduced the self-attention mechanism.
2.3 RAG-Based Grading Systems
Retrieval-augmented generation provides an elegant solution to grounding LLM
outputs to factual evidence. Qiu et al. [21] introduce SteLLA, a structured
grading framework that leverages RAG with LLMs to accomplish reference-
grounded short-answer grading. SteLLA transforms reference grading criteria
intoquestion-answerpairsandgroundsLLMscoringinretrievedevidence,reach-
ing Cohen’sκ= 0.672on undergrad biology exams. Arslan et al. [2] review
RAG systems utilizing LLMs and empirically show that predicting with relevant
evidence passages retrieved beforehand drastically improves factual grounding
and decreases hallucination in generated feedback. However, flat retrieval, the
method implemented by base RAG models, introduces noisy duplicate passages,
lacks the ability to model relationships between reference passages, and doesn’t
scale to the multi-reference problem of retrieving allnnodes. Dai and Le [4] and
Radford et al. [22] laid the pre-training groundwork that retrieval-augmented
systems utilize for language model knowledge.
2.4 Graph-Enhanced Retrieval (GRAG)
Graph-augmented retrieval directly addresses the limitations of flat RAG by
organising retrieved knowledge into structured entity-relation graphs. Edge et
al. [9] proposed GraphRAG, which constructs entity-relation graphs from re-
trieved text to enable multi-hop reasoning over both local subgraphs and global
community summaries. This approach is particularly effective for queries requir-
ing synthesis across multiple related documents, such as identifying all reference
nodes relevant to a multi-topic essay. Hu et al. [14] introduced GRAG: Graph
Retrieval-Augmented Generation, which organises retrieved evidence into entity-
relation structures to enable structured multi-hop reasoning and reduce retrieval
4

This is a pre-print version of the paper accepted in ICONIP 2026.
noise from irrelevant passages. Both methods are directly instantiated in the
GRASP pipeline: reference nodes are connected by cosine similarity edges, and
GRAG retrieval expands from FAISS [7] seed results along strong edges (≥0.70
weight) to recover reference nodes that are semantically adjacent but not the
top-scoring singleton retrievals.
2.5 LLM-Based Feedback Generation
Recent research has investigated applications of LLMs towards generating in-
structionally valuable feedback along with scores. To better adapt LLMs to au-
tomatic scoring tasks, Latif and Zhai [18] fine-tuned ChatGPT with reference-
aligned annotations and showed that LLMs could be steered towards conforming
to marking schemes through instruction tuning. Stahl et al. [25] examined chain-
of-thought prompting and trait-aware prompting techniques for the joint task
of scoring and feedback generation, showing that chained prompts significantly
improve trait-level scoring coherence. Hussein et al. [15] introduced a trait-based
deep learning AES system that generates adaptive feedback conditioned on in-
dividual student performance profiles to vary the specificity of feedback. Hou
et al. [13] showed that supplementing LLM prompts with engineered linguis-
tic features narrows performance gaps between small- and large-scale models,
indicating that feature-aware prompting may serve as a more economical alter-
native to fine-tuning. Xiao et al. [29] designed a human-AI collaborative scoring
framework for essays informed by adual-process theory, which modularizes rapid
holisticscoringfromdeliberatereference-groundedreview.Inacross-disciplinary
survey of auto-grading methods with LLMs, Eneye et al. [11] found fairness, in-
terpretability, and domain adaptation to remain open issues, which aligns with
our design of GRASP to be training-data free.
3 Problem Definition
Considering the automated examination system which was introduced in Sec-
tion 1, the grading problem is described as following constituent tasks:
Amulti-topic essayis a continuous student textEwhich is written in re-
sponse to a question which can includenrelated topics, wheren∈ {1,2,3,4}.
The text contains no structural markers such as “Part 1:”, numbered sections, or
line breaks between answers. The answers appear in an arbitrary order that may
not correspond to the order of topics on the exam question. Critically, forn≥2
the topics addressed in any single essay are not chosen at random from the full
reference pool. They are required to form a connected subgraph in the reference
node similarity graph (see Section 4). This means that their reference answers
are semantically related with pairwise cosine similarity≥0.42. This constraint
reflects realistic exam design. Topics in the same question tend to share a do-
main or theme. This makes the retrieval problem significantly harder, because
the correct reference nodes for a given essay are closer together in embedding
space than arbitrary pairs.
5

This is a pre-print version of the paper accepted in ICONIP 2026.
Areference nodeis a tripleR i= (q i, ai, ti), whereq iis the original exam
question,a iis the reference correct answer, andt i=qi⊕aiis the concatenation
of both. The full reference index contains|R|= 79nodes drawn from the Sci-
EntsBank dataset, one node per unique (question, reference answer) pair. This
includes multiple science topics including physics, chemistry, biology, and earth
science.
Thetopic detection taskrequires the system to predictnfrom the student
essayEtogether with the paraphrased exam question, with no access to Part N:
labels or any other structural information. The system must infernfrom lin-
guistic content alone.
Thesegmentation taskrequires partitioningEintonnon-overlapping, con-
tiguous segments{s 1, . . . , s n}such that each segments icontains the student’s
complete answer to exactly one topic in the question. Segments must cover the
full text with no gaps.
Theretrieval taskoperates on a paraphrased version of the exam question
as a query and retrieves candidate reference nodes from indexI, whereIis
built using SBERT [23] embeddings of reference answers only. This is a cross-
modal retrieval design: the query is a paraphrased exam question while the
index contains reference answers. Both retrieval configurations return the same
number of candidatesk=n+ 2, so that any difference in downstream grading
is attributable to retrieval quality rather than to differing candidate-set sizes.
Retrieval must identify allncorrect reference nodes within thesekcandidates.
Theassignment taskmaps thenparaphrased question segments to the re-
trieved candidate reference nodes through a one-to-one matching, so that each
segment is paired with a distinct reference node and no candidate is reused. This
is solved with the Hungarian algorithm. Full details are given in Section 4.
Thegrading taskassigns a scoreg i∈ {1,2}to each (student answer segment,
reference node) pair, where score 2 denotes a correct answer and score 1 denotes
a partially correct or incomplete answer. The final essay score is:
G(E) =nX
i=1gi, G(E)∈[n,2n](1)
To isolate retrieval quality from grading quality, we additionally define a
TRUE Oracleupper bound that bypasses detection, segmentation, and retrieval
entirely: it grades thengold reference answers directly against their gold ref-
erence nodes, achieving full retrieval (N-Hit= 1) by construction. The Oracle
measures the ceiling imposed by the LLM grader alone, against which RAG and
GRAG are compared.
Looking at the examination system: for the student who has written three
sentences about mineral hardness, the system must identifyn= 3, split the three
sentences into three segments, retrievek=n+ 2 = 5candidate reference nodes,
match each of the three segments to a distinct reference node, and award a score
of 1 or 2 per segment. The final score ranges from 3 to 6. The challenge is that
all three correct reference nodes were drawn from the same topical cluster in
6

This is a pre-print version of the paper accepted in ICONIP 2026.
the database , they relate to scratch testing and hardness , so graph-augmented
retrieval is necessary to reliably recover all three simultaneously.
4 Methodology
GRASP grades a label-free multi-topic essay in two phases, as shown in Fig. 1.
Phase A covers the offline steps that are run once to build the dataset, the FAISS
index,andthesimilaritygraph.PhaseBcoverstheper-essaystepsrunatgrading
time: detection, segmentation, retrieval, assignment, and grading. Splitting the
work this way lets the expensive corpus setup be built once and reused, so each
essay only triggers the lighter per-essay steps.
Phase A — built once (offline)
Load SciEntsBank
keep rows which has correct + partial answers
Build 79 reference nodes
one per unique exam question
reference node = question q + reference answer a + t (q+a)
Build F AISS index + similarity graph
SBER T encodes reference answers (384-d)
edges where cosine ≥ 0.42 
Paraphrase question + synthetic essays
 essays and paraphrased question created from connected
reference nodesPhase B — per essay
Step 1 — Hybrid topic detection
sentence counts agree → return n · else LLM decides
Step 2 — Segment essay into n parts
split on sentences · LLM only if count ≠ n
Step 3a — RAG retrieval
query: paraphrased question
retrieves: (reference_id,
faiss_score), k = n+2
cosine similarity onlyStep 3b — GRAG retrieval
query: paraphrased question
seeds + strong edge ≥ 0.70  
neighbours, k = n+2
same budget as RAG
Step 4 — Hungarian assignment
match each segment to one distinct reference node
Step 5 — Grade each part (GPT -4.1-mini)
score 1 (partial) or 2 (correct) · essay score = sum
same grader used for Oracle, RAG, GRAG
TRUE Oracle (upper bound)
gold reference nodes +  gold
segmented student answer taken
directly
no detection, no segmentation, no
retrieval
N-Hit = 1 by constructionEvaluate
N-Hit · QWK ·
(11 reps, 95% CI)essay ,
paraphrased question,
reference nodes,
FAISS, graph
Fig. 1.GRASP Pipeline
4.1 Phase A: Offline steps for building the dataset
This phase generates the static resources against which each essay will be scored.
It runs once, separately from any student submission, and outputs the reference
nodes, FAISS index, and similarity graph that will be used in Phase B.
Dataset.We use the SciEntsBank dataset1from HuggingFace [8]. Labels
other thancorrect(scored 2) andpartially_correct_incomplete(scored 1) are
1https://huggingface.co/datasets/nkazi/SciEntsBank
7

This is a pre-print version of the paper accepted in ICONIP 2026.
discarded. Rows for which the diagram filter detected a visual stimulus are fil-
tered out. Rows are dropped where the text component is empty.
Build reference nodes.Reference nodesR i= (q i, ai, ti)are built, one node
per remaining unique (question, reference answer) pair in dataset. Nodes store
Q,Aand concatenated QA pairst i.
Build FAISS index.Every reference answera iis encoded with SBERT
(defaultall-MiniLM-L6-v2). Answer embeddings are stored in a FAISS inner-
product index. Note: index only stores reference answer-form text. Query inputs
are always exam questions. Retrieval must bridge the question-answer semantic
gap using frozen SBERT embeddings alone. The essay-construction heuristic
compounds the difficulty: by construction, allnreference nodes connected inG
for a given essay (every essay is guaranteed to have one) have reference answers
which are semantically similar. Their embeddings will tend to cluster together
in space. This makes it more difficult for naive flat top-kretrieval to recover all
nwithout graph expansion.
Build node similarity graphG= (V, E).NodesVare simply the set of
79 reference nodes constructed previously. Add an edge(R i, Rj)∈Eweighted
wijiffcos(e ai, eaj)≥0.42. Edges with weightw ij≥0.70are deemed strong and
are used later during graph expansion.
Build paraphrase cache.Each reference questionq iis paraphrased once
using GPT-4.1-mini, output is stored in cache keyed on reference node, reusing
for every essay. This amounts to one LLM call per unique reference node.
Construct similarity-constrained topic groups.Before constructing es-
says, groups ofnreference nodes are preselected fromG. These groups exhibit
strong semantic similarity between every reference node, the topics within each
essay should relate to each other, as would be in a real realistic exam domain
where the questions on the same test paper are generally drawn from the same
domain or theme. Forn= 1essays, each node in its own is a single-topic group.
For groups of sizen≥2, nodes must be strongly connected inG, formally
speaking, each group forms a clique. Cliques are chosen as follows. Forn= 2,
every edge(R i, Rj)with weight≥0.42is valid so long as questions are distinct.
Forn≥3, recursively enumerate all cliques of size≥ninG, then take the size-n
subsets. Each group is labelled by its minimum edge weightw min. Groups with
highw min, that correspond to sets of topics that are more semantically coherent
are preferred.
Synthetic essay assembly.For eachn∈ {1,2,3,4}160 label-free syn-
thetic essays are generated. Essays are assembled as follows. LetHbe the set
of reference nodes forming ann-topic group. Sample student answers∀R i∈H
from pool of available sentences forR i, then they are concatenated in node or-
der. This results in student essayEwith no structural tags, separators, or line
breaks. Parallel paraphrased question essay is constructed from per-topic para-
phrase question from cache. Numeric gold score for essay is computed as sum of
gold scores of sourcenanswers.
8

This is a pre-print version of the paper accepted in ICONIP 2026.
4.2 Phase B: Per-Essay Processing Steps
This phase runs once per student essay. Using the resources built in Phase A,
it applies five steps in sequence: topic detection, segmentation, retrieval, assign-
ment, and grading.
Hybrid topic detection.Lets 1ands 2respectively denote the sentence
count ofEand of its paraphrased question. Ifs 1=s 2, the agreed value is
returned immediately asnwithout calling the LLM. Otherwise, the LLM is
called once at temperature 0, with both texts provided as context along with
the sentence counts as explicit hints. Prompt is carefully designed to be robust
to common failure mode of over-detection, where student writes elaborates the
same topic for several consecutive sentences. The hybrid approach skips the
LLM call entirely for the majority of essays, where the student sentence count
and question sentence count already agree, incurring only a single call on the
remaining essays where they disagree.
Sentence-first segmentation heuristic.Eis segmented by attempting to
split on sentence boundaries first. LLM is called if the resulting segment count
̸=detectedn. Answer segments and, in parallel, question segments obtained. by
splitting the per-topic paraphrases andE.
Equal-kretrieval.Both retrieval methods are configured to have budget
k=n detected + 2computed from detected value ofn. Extra two candidates gives
room for retrieval candidates to “move around” during assignment step. Hold-
ing budget equal between RAG and GRAG configurations isolates any grading
differences and bias.
Retrieval by RAG.Paraphrased question essay is encoded with SBERT
into fixed-dimensional query vector. Topk=n+ 2reference nodes are obtained
cosine distance from FAISS index.
Retrieval by GRAG.Let seed setScontain the top 3 reference nodes
returned by FAISS under exact same settings as RAG. Candidate pool is se-
lected by performing graph expansion out of seeds, following outgoing strong
edges (w≥0.70, one hop) from each seed node to retrieve candidate pool. Each
candidateR jis scored with combined score of score(R j) =α·cos(e Q, eaj) +β·
max s∈Swsjwhereα= 0.80andβ= 0.20, where second term corresponds to
max over seedssof strong edge weight betweenR jands. As opposed to RAG
above, candidates are retrieved using GRAG.
Assignment step.With candidate pool of reference input,each paraphrased
question segment is embedded independently with SBERT and a cost matrix is
constructed. Hungarian algorithm is used to map each paraphrased question seg-
ment to a reference node. Both RAG and GRAG assignments use this procedure.
TRUE Oracle choice.Each essay is also graded via a TRUE Oracle path
that grades the gold reference nodes and answer segments directly, bypassing
detection, segmentation, retrieval, and assignment. This gives N-Hit= 1by
construction and serves as the upper bound against which RAG and GRAG are
compared.
LLM grading with score variance.Each answer segment, reference node
pairfromassignmentprocedureispassedtoGPT-4.1-mini.Modelindependently
9

This is a pre-print version of the paper accepted in ICONIP 2026.
gradessegmentas2(correct)or1(partiallycorrect),totalingafterwardsforessay
score. Grading procedure is repeated 11 times over at temperature 0.7 for Ora-
cle, RAG, and GRAG, while holding the deterministic retrieval and assignment
stages constant. Repeating the stochastic grading enables empirical computation
of score variance and 95% confidence intervals for each essay, in addition to the
point estimate of mean score.
5 Results
Oracle,RAG,andGRAGareevaluatedonessayswithtopiccountsn∈ {1,2,3,4}
from SciEntsBank. For each essay, the paragraph question is used as the query
to retrieve candidate reference nodes from the reference-answer index. The re-
trieved candidates are then assigned to the paraphrased question segments via
the Hungarian algorithm, and each (student-answer segment, assigned reference,
question) triple is graded by the LLM as 2 (correct) or 1 (partially correct). Each
essay is graded 11 times at temperature 0.7. The summed essay score ranges
fromnto2n, so the full grading scale spans 1 to 8. We report two metrics:
N-Hit (fraction of essays for which retrieval recovered all gold reference nodes)
and Quadratic Weighted Kappa (QWK). The Oracle is given the gold reference
nodes directly (N-Hit= 1by construction), which isolates grader ability from
retrieval. Hybrid topic detection recovered the correctnfor 90.8% of essays.
5.1 Evaluation Metrics
We assess the pipeline using two metrics, defined below: the N-Hit rate for re-
trieval quality and Quadratic Weighted Kappa for grading agreement.
N-Hit rate.For each essay withntopics, the system assigns one reference
node per topic segment. N-Hit is1if allngold reference nodes are retrieved in
the assigned set and0otherwise. The N-Hit rate is the mean over all essays.
Quadratic Weighted Kappa (QWK).QWK is a statistical measure of
thelevelofagreementbetweentworaterswhoassignitemstoanordinalscale[6].
It is well suited to ordinal classification problems, such as integer student scores,
and penalises larger disagreements more heavily than smaller ones. The coeffi-
cient ranges from−1to1, where1indicates perfect agreement, with0indicating
no agreement better than a random process.
5.2 Retrieval.
GRAG retrieves significantly more gold reference nodes than flat RAG (Fig. 2).
Atn= 1, 88.8% vs 80.0%, atn= 2, 71.2% vs 55.0%, atn= 4, 59.4% vs 38.1%.
The biggest difference is atn= 3, where GRAG attains 62.5% while RAG
drops to 25.6%. GRAG recovers the complete gold set over twice as frequently.
Since an essay’s topics arise from a connected subgraph of the topic graph, gold
reference nodes are tightly clustered in embedding space. Flat top-kretrieval
returns near-duplicates of the seed node, GRAG traces strong graph edges out
10

This is a pre-print version of the paper accepted in ICONIP 2026.
n=1 n=2 n=3 n=4
n (number of topics per essay)0%20%40%60%80%100%N-Hit rate100.0% 100.0% 100.0% 100.0%
80.0%
55.0%
25.6%38.1%88.8%
71.2%
62.5%59.4%N-Hit (all gold reference nodes recovered)  (α=0.80, β=0.20, T=0.7, reps=11)
Oracle
RAG
GRAG
Fig. 2.N-Hit by topic countn: fraction of essays for which retrieval recovered all gold
reference nodes. Oracle is perfect by construction.
to the cluster’s neighbours. These results imply graph structure benefits the
harder examples more.
5.3 Grading and false positives.
A false positive (FP) is a retrieved reference node that is not in the gold set:
the grader is shown the wrong reference and asked to mark the student’s answer
segment against it. The damage is proportional to the semantic closeness of the
wrong reference to the correct one. The cleanest case arises from the pitch clus-
ter. The gold reference node asks what happens to pitch when the string is pulled
tighter, RAG retrieves the reference node about ashorterstring instead. Both
reference answers state only the effect, “the pitch will be higher”, because the
cause (tightness vs shortness) is encoded in the question stem, not in the refer-
enceanswer.Moststudentscopyonlytheeffect,e.g.“thepitchwouldgethigher”,
so the grader, comparing the student’s effect-claim against the reference’s effect-
claim, awards full credit 2 even on the wrong reference. The ambiguity is built
into the dataset: four pitch reference nodes share a common effect and vary only
on cause. The FP rate quantifies this exposure (Fig. 3): RAG runs 17.7–30.1%
acrossn= 1–4, GRAG 10.5–15.8%. QWK reflects where the wrong references
actually hurt: GRAG beats RAG atn= 1(0.52 vs 0.46),n= 2(0.29 vs 0.26),
andn= 3(0.23 vs 0.15). Atn= 4both RAG and GRAG collapse (Oracle 0.50,
GRAG 0.02, RAG 0.05). The grader’s own ceiling falls here. Oracle drops from
0.64 atn= 3to 0.50, but the large gap between Oracle and GRAG shows that
retrieval and assignment errors pull performance well below even that lowered
ceiling. We viewn= 4grading as an open problem rather than a solved case:
with only four topics compounding retrieval, assignment, and grading error at
once, this setting is the clearest evidence that GRASP, and graph-augmented
retrieval more broadly, has not yet closed the gap to reliable grading at the high-
11

This is a pre-print version of the paper accepted in ICONIP 2026.
n=1 n=2 n=3 n=4
n (number of topics per essay)0.00.20.40.60.8Quadratic Weighted KappaQWK and retrieval false-positive rate (α=0.80, β=0.20)
Oracle (QWK)
RAG (QWK)GRAG (QWK)
RAG (FP rate)GRAG (FP rate)0.0%5.0%10.0%15.0%20.0%25.0%30.0%
Retrieval false-positive rate
(assigned reference ∉ gold set)
20.0%
11.2%28.5%
15.4%30.1%
15.8%17.7%
10.5%0.560.500.64
0.50
0.46
0.26
0.15
0.050.52
0.29
0.23
0.02
Fig. 3.Left axis: QWK by topic countnfor Oracle, RAG, and GRAG. Right axis:
retrieval false-positive rate (assigned reference/∈gold set) for RAG and GRAG.
est topic counts we tested. All QWK values are reported as the mean over 11
grading repetitions, with 95% confidence intervals shown as error bars in Fig. 3.
5.4 Why we report per-n.
Pooled across alln, QWK is essentially flat (Oracle 0.978, RAG 0.892, GRAG
0.901 atα= 0.8,β= 0.2). This is an instance of Simpson’s paradox:nis corre-
lated with both predictor and outcome, since higher-nessays carry higher gold
totals, and any grader that approximately tracksninherits that correlation,
inflating pooled agreement on the 1–8 scale. We report both pooled and per-n
results because each answers a different question, and both are valid. At test
time the system is not toldn, it must detect it from the essay and question
alone (recovered correctly for 90.8% of essays). The pooled figure therefore re-
flects genuine end-to-end performance, where correctly inferring how many parts
an essay has is itself part of the task and legitimately contributes to agreement.
The per-nfigures holdnfixed and so isolate the retrieval contribution, exposing
the differences between methods that the pooled number masks. Neither is an
artifact to be discarded: the pooled result measures the full deployed pipeline,
while the per-nresult attributes credit to retrieval, and these are reported to-
gether throughout. These results imply that GRAG’s higher per-nQWK reflects
a genuine retrieval improvement rather than an effect of the wider score range,
sinceholdingnfixedremovesthescore-rangeadvantagethepooledfigurecarries.
5.5 Limitations
Our evaluation essays are synthetic: they are assembled by concatenating sen-
tences from separately authored, single-topic SciEntsBank reference answers, in
12

This is a pre-print version of the paper accepted in ICONIP 2026.
the fixed order of their source reference nodes, and the paraphrased exam ques-
tions are generated from the same underlying reference questions. This construc-
tionguaranteescleangroundtruthfortopiccount,segmentationboundaries,and
gold reference nodes, which real multi-topic student essays do not provide, but
it does not capture ways genuine students blend topics within a single sentence,
use cross-referential language between topics, or answer topics out of the order in
whichtheywereasked.Asaresult,thedetection,segmentation,andretrievalfig-
ures reported here should be read as an upper bound relative to real deployment,
and validating the pipeline on genuine multi-topic student responses, including
answers presented out of order, is an important direction for future work.
6 Conclusion and Future Work
We presented GRASP, a label-free pipeline for grading synthetically constructed
multi-topicscienceessaysdesignedtoapproximaterealisticmulti-topicexamset-
tings. In it, a graph-augmented retrieval module called GRAG recovers reference
nodes left behind by flat similarity retrieval. Between one and four topics per
essay, GRAG retrieved the entire gold set of references significantly more fre-
quently than flat RAG at everyn, and by wider margins as the gold set became
more tightly clustered. The effect was starkest atn= 3, where GRAG retrieved
the full gold set more than twice as often as its RAG baseline. This translated to
higher grading agreement at low and moderate topic counts. By supplementing
our experiments with a TRUE Oracle that retrieved all references perfectly, we
isolated grading errors at higher topic counts from retrieval errors: even under
perfect retrieval, QWK falls to 0.50 at n = 4, indicating that grading quality
and retrieval quality are at least partially separable. This pattern is consistent
with a ceiling imposed by the LLM grader itself, though we tested only GPT-4.1-
mini as the grader and cannot rule out that a stronger or differently-prompted
modelwouldnarrowthisgap;weleaveamulti-gradercomparisontofuturework.
Taken together, our experiments point to a narrow but reliable conclusion: given
a set of gold references clustered together semantically, access to graph structure
recovers them more reliably than does embedding similarity.
GRAG’s largest gains came where flat embedding retrieval failed most no-
ticeably. When the relevant reference nodes sat close together in embedding
space and gold-set similarity scores alone were insufficient to pick out the cor-
rect references from their near-duplicates. This suggests a larger application for
our method. While we used document similarity to build a graph connecting
reference nodes, that graph was only one way of connecting those nodes, and
our retrieval mechanism can operate over any connections an examiner chooses
to draw. The best connections aren’t necessarily those implied by SBERT simi-
larity scores. Consider a co-occurrence graph that connects reference nodes that
appear together on real exams/course curriculum rather than ones that simply
have similar reference answers. Exploring such alternative graph constructions
is a promising direction for future work.
13

This is a pre-print version of the paper accepted in ICONIP 2026.
Bibliography
[1] Amin, T., Tanoli, Z.U.R., Aadil, F., Awan, K.M., Lim, S.: Enhancing Essay
Scoring: An Analytical and Holistic Approach With Few-Shot Transformer-
Based Models. IEEE Access13, 12483–12501 (2025)
[2] Arslan, M., Ghanem, H., Munawar, S., Cruz, C.: A Survey on RAG with
LLMs. Procedia Computer Science246, 3781–3790 (2024)
[3] Cummins, R., Zhang, M., Briscoe, T.: Constrained Multi-Task Learning for
Automated Essay Scoring. In: Erk, K., Smith, N.A. (eds.) Proceedings of
the 54th Annual Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers). pp. 789–799. Association for Computational
Linguistics, Berlin, Germany (2016)
[4] Dai, A.M., Le, Q.V.: Semi-supervised Sequence Learning (2015)
[5] Do, H., Kim, Y., Lee, G.G.: Autoregressive Score Generation for Multi-trait
Essay Scoring (2024)
[6] Doewes,A.,Kurdhi,N.A.,Saxena,A.:Evaluatingquadraticweightedkappa
as the standard performance metric for automated essay scoring. In: Pro-
ceedings of the 16th International Conference on Educational Data Mining
(EDM). pp. 103–113 (2023).https://doi.org/10.5281/zenodo.8115784
[7] Douze, M., Guzhva, A., Deng, C., Johnson, J., Szilvasy, G., Mazaré, P.E.,
Lomeli, M., Hosseini, L., Jégou, H.: The Faiss library. IEEE Transactions
on Big Data (2025).https://doi.org/10.1109/TBDATA.2025.3618474
[8] Dzikovska, M., Nielsen, R., Brew, C., Leacock, C., Giampiccolo, D., Ben-
tivogli, L., Clark, P., Dagan, I., Dang, H.T.: SemEval-2013 Task 7: The
Joint Student Response Analysis and 8th Recognizing Textual Entailment
Challenge. In: Volume 2: Proceedings of the Seventh International Work-
shop on Semantic Evaluation (SemEval 2013). pp. 263–274. Association for
Computational Linguistics (2013)
[9] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt,
S., Metropolitansky, D., Ness, R.O., Larson, J.: From Local to Global: A
Graph RAG Approach to Query-Focused Summarization (2025)
[10] Elmassry, A.M., Zaki, N., Alsheikh, N., Mediani, M.: A Systematic Re-
view of Pretrained Models in Automated Essay Scoring. IEEE Access13,
121902–121917 (2025)
[11] Frederick Eneye, T.A.N., Ijezue, C.F., Imam Amjad, A., Amjad, M., Butt,
S., Castañeda-Garza, G.: Advances in Auto-Grading with Large Language
Models: A Cross-Disciplinary Survey. In: Proceedings of the 20th Workshop
on Innovative Use of NLP for Building Educational Applications (BEA
2025). pp. 477–498. Association for Computational Linguistics, Vienna,
Austria (2025)
[12] Gao, T., Yao, X., Chen, D.: SimCSE: Simple Contrastive Learning of Sen-
tence Embeddings (2022)
14

This is a pre-print version of the paper accepted in ICONIP 2026.
[13] Hou, Z.J., Ciuba, A., Li, X.L.: Improve LLM-based Automatic Essay Scor-
ing with Linguistic Features (2025)
[14] Hu, Y., Lei, Z., Zhang, Z., Pan, B., Ling, C., Zhao, L.: GRAG: Graph
Retrieval-Augmented Generation (2025)
[15] Hussein, M.A., Hassan, H.A., Nassef, M.: A Trait-based Deep Learning
Automated Essay Scoring System with Adaptive Feedback. International
Journal of Advanced Computer Science and Applications11(5) (2020)
[16] Janda, H.K., Pawar, A., Du, S., Mago, V.: Syntactic, Semantic and Sen-
timent Analysis: The Joint Effect on Automated Essay Evaluation. IEEE
Access7, 108486–108503 (2019)
[17] Khattab, O., Zaharia, M.: ColBERT: Efficient and Effective Passage Search
via Contextualized Late Interaction over BERT (2020)
[18] Latif, E., Zhai, X.: Fine-tuning ChatGPT for Automatic Scoring (2023)
[19] Li, X., Yang, H., Hu, S., Geng, J., Lin, K., Li, Y.: Enhanced hybrid neural
network for automated essay scoring. Expert Systems39(10), 1–22 (2022)
[20] Patil, P., Agrawal, A.: Auto Grader for Short Answer Questions. Stanford
CS224N course project report (2018)
[21] Qiu, H., White, B., Ding, A., Costa, R., Hachem, A., Ding, W., Chen, P.:
SteLLA: A Structured Grading System Using LLMs with RAG (2025)
[22] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.: Lan-
guage Models are Unsupervised Multitask Learners. Tech. rep., OpenAI
(2019)
[23] Reimers, N., Gurevych, I.: Sentence-BERT: Sentence Embeddings using
Siamese BERT-Networks (2019)
[24] Rodriguez, P.U., Jafari, A., Ormerod, C.M.: Language models and Auto-
mated Essay Scoring (2019)
[25] Stahl, M., Biermann, L., Nehring, A., Wachsmuth, H.: Exploring LLM
Prompting Strategies for Joint Essay Scoring and Feedback Generation
(2024)
[26] Sun, K., Wang, R.: Automatic Essay Multi-dimensional Scoring with Fine-
tuning and Multiple Regression (2024)
[27] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N.,
Kaiser, Ł., Polosukhin, I.: Attention is All you Need. In: Advances in Neural
Information Processing Systems 30 (NeurIPS 2017). pp. 5998–6008 (2017)
[28] Wang, Y., Wang, C., Li, R., Lin, H.: On the Use of BERT for Automated
Essay Scoring: Joint Learning of Multi-Scale Essay Representation (2022)
[29] Xiao, C., Ma, W., Song, Q., Xu, S.X., Zhang, K., Wang, Y., Fu, Q.: Human-
AI Collaborative Essay Scoring: A Dual-Process Framework with LLMs. In:
15th International Learning Analytics and Knowledge Conference. pp. 293–
305. Association for Computing Machinery (2025)
[30] Xue, J., Tang, X., Zheng, L.: A Hierarchical BERT-Based Transfer Learning
Approach for Multi-Dimensional Essay Scoring. IEEE Access9, 125403–
125415 (2021)
15