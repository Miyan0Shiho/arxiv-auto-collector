# Question Decomposition for Retrieval-Augmented Generation

**Authors**: Paul J. L. Ammann, Jonas Golde, Alan Akbik

**Published**: 2025-07-01 01:01:54

**PDF URL**: [http://arxiv.org/pdf/2507.00355v1](http://arxiv.org/pdf/2507.00355v1)

## Abstract
Grounding large language models (LLMs) in verifiable external sources is a
well-established strategy for generating reliable answers. Retrieval-augmented
generation (RAG) is one such approach, particularly effective for tasks like
question answering: it retrieves passages that are semantically related to the
question and then conditions the model on this evidence. However, multi-hop
questions, such as "Which company among NVIDIA, Apple, and Google made the
biggest profit in 2023?," challenge RAG because relevant facts are often
distributed across multiple documents rather than co-occurring in one source,
making it difficult for standard RAG to retrieve sufficient information. To
address this, we propose a RAG pipeline that incorporates question
decomposition: (i) an LLM decomposes the original query into sub-questions,
(ii) passages are retrieved for each sub-question, and (iii) the merged
candidate pool is reranked to improve the coverage and precision of the
retrieved evidence. We show that question decomposition effectively assembles
complementary documents, while reranking reduces noise and promotes the most
relevant passages before answer generation. Although reranking itself is
standard, we show that pairing an off-the-shelf cross-encoder reranker with
LLM-driven question decomposition bridges the retrieval gap on multi-hop
questions and provides a practical, drop-in enhancement, without any extra
training or specialized indexing. We evaluate our approach on the MultiHop-RAG
and HotpotQA, showing gains in retrieval (MRR@10: +36.7%) and answer accuracy
(F1: +11.6%) over standard RAG baselines.

## Full Text


<!-- PDF content starts -->

arXiv:2507.00355v1  [cs.CL]  1 Jul 2025Question Decomposition for Retrieval-Augmented Generation
Paul J. L. Ammann Jonas Golde Alan Akbik
Humboldt-Universität zu Berlin
{paul.ammann, jonas.max.golde.1, alan.akbik}@hu-berlin.de
Abstract
Grounding large language models (LLMs)
in verifiable external sources is a well-
established strategy for generating reliable an-
swers. Retrieval-augmented generation (RAG)
is one such approach, particularly effective for
tasks like question answering: it retrieves pas-
sages that are semantically related to the ques-
tion and then conditions the model on this evi-
dence. However, multi-hop questions, such as
“Which company among NVIDIA, Apple, and
Google made the biggest profit in 2023?, ” chal-
lenge RAG because relevant facts are often dis-
tributed across multiple documents rather than
co-occurring in one source, making it difficult
for standard RAG to retrieve sufficient infor-
mation. To address this, we propose a RAG
pipeline that incorporates question decomposi-
tion: (i) an LLM decomposes the original query
into sub-questions, (ii) passages are retrieved
for each sub-question, and (iii) the merged can-
didate pool is reranked to improve the cover-
age and precision of the retrieved evidence.
We show that question decomposition effec-
tively assembles complementary documents,
while reranking reduces noise and promotes the
most relevant passages before answer genera-
tion. Although reranking itself is standard, we
show that pairing an off-the-shelf cross-encoder
reranker with LLM-driven question decompo-
sition bridges the retrieval gap on multi-hop
questions and provides a practical, drop-in en-
hancement, without any extra training or spe-
cialized indexing. We evaluate our approach on
the MultiHop-RAG and HotpotQA, showing
gains in retrieval ( MRR @10 : +36 .7%) and
answer accuracy ( F1 : +11 .6%) over standard
RAG baselines.
1 Introduction
Retrieval-augmented generation (RAG) addresses
knowledge gaps in large language models (LLMs)
by retrieving external information at inference time
Figure 1: (a) Standard retrieval in RAG versus (b) our
approach using question decomposition and reranking.
(Lewis et al., 2020). While effective, RAG’s perfor-
mance depends heavily on retrieval quality; irrele-
vant documents can mislead the model and degrade
the quality of its output (Cho et al., 2023; Shi et al.,
2023). For example, when asked “Who painted
Starry Night ?” a naive retriever may surface a gen-
eral Wikipedia article on Post-Impressionism rather
than the specific page on Vincent van Gogh , of-
fering little direct evidence for the correct answer.
This issue becomes more pronounced in multi-hop
QA tasks, where supporting facts are spread across
multiple documents. For instance, a single, un-

differentiated search for the query “Which com-
pany among NVIDIA, Apple, and Google made
the biggest profit in 2023?” might return a broad
market overview article mentioning all three com-
panies together, but omit their individual 2023 earn-
ings reports—forcing the model to respond without
access to the necessary disaggregated information.
Challenges of Multi-hop Retrieval. Complex
questions often require reasoning over multiple
entities, events, or steps, which are rarely ad-
dressed within a single document. While the indi-
vidual facts needed to answer such questions may
be simple, the required evidence is typically dis-
tributed across multiple sources. To improve re-
trieval coverage in multi-hop QA settings, our ap-
proach decomposes the original question into sim-
pler subqueries—a process we refer to as question
decomposition (Perez et al., 2020). By breaking
down a complex query into focused subqueries,
question decomposition increases the likelihood of
retrieving documents that address distinct aspects
of the information need, especially when informa-
tion sources are self-contained.
Consider the question: “Which planet has more
moons, Mars or Venus?” In a standard RAG
pipeline, the entire question is embedded as a sin-
gle unit, and the retriever attempts to find a single
passage that answers it directly (cf. Figure 1a). In
practice, this often results in retrieving a general
article about planetary science or solar system for-
mation. We assume that relevant facts are located
in two self-contained documents—one about Mars
and the other about Venus. With QD, we exploit the
fact of increasingly capable LLMs to generate fact-
seeking subquestions such as “How many moons
does Mars have?” and “How many moons does
Venus have?”, each of which is more likely to re-
trieve a precise, relevant answer from its respective
source (cf. Figure 1b).
Contributions. In this paper, we present a retrieval-
augmented generation pipeline that integrates ques-
tion decomposition with reranking to improve
multi-hop question answering. Our QD compo-
nent uses a LLM to decompose complex questions
into simpler subqueries, each addressing a specific
part of the information need, and thus requires no
fine-tuning or task-specific training. Retrieved re-
sults from all subqueries are aggregated to form a
broader and more semantically relevant candidate
pool.
To mitigate the noise introduced by retrieving
documents for each subquery, we apply a pre-trained reranker that scores each candidate passage
based on its relevance to the original complex query.
This substantially improves precision by filtering
out irrelevant results. In combination, question
decomposition ensures broad evidence coverage,
while reranking distills this expanded set into a
concise collection of highly relevant passages.
We evaluate our approach on the MultiHop-RAG
and HotpotQA benchmarks and demonstrate sub-
stantial gains in recall and ranking metrics over
standard RAG and single-component variants. We
further analyze the inference overhead, showing
that the added cost of QD remains manageable.
Our main contributions are as follows:
•We propose a question decomposition
(QD)–based RAG pipeline for multi-hop ques-
tion answering, where a LLM decomposes
complex questions into simpler subqueries
without any task-specific training.
•To improve precision, we incorporate a cross-
encoder reranker that scores retrieved pas-
sages based on their relevance to the origi-
nal complex query, effectively filtering noise
from the expanded candidate pool introduced
by QD.
•We empirically validate our approach on the
MultiHop-RAG and HotpotQA benchmarks,
demonstrating substantial improvements in re-
trieval recall, ranking quality, and final an-
swer accuracy—achieved without any domain-
specific fine-tuning.
We release our code1on GitHub for reproducibility.
2 Methodology
Our pipeline follows the retrieval-augmented gener-
ation framework of Lewis et al. (2020), which com-
bines a retriever with a generative language model.
The goal is to answer a natural language query q
by grounding the language model’s response in
documents retrieved from a large corpus D.
Retrieval. In the first step, a query encoder fq
and a document encoder fdproject queries and
documents into a shared vector space (Karpukhin
et al., 2020). During retrieval, the query representa-
tionfq(q)is compared to all document embeddings
fd(d)using inner product similarity. Subsequently,
we select the top- kmost relevant documents:
1https://github.com/Wecoreator/qd_rag

R(q) = Top -kd∈D(⟨fq(q), fd(d)⟩)
Here,⟨·,·⟩denotes the similarity score between
the query and document embeddings, computed as
inner product similarity in the shared embedding
space. This dense retrieval stage identifies doc-
uments that are semantically similar to the query
and provides candidates for grounding the language
model’s response.
Reranking. To refine the initial retrieval set
R(q), we apply a pre-trained reranker that com-
putes fine-grained relevance scores between the
query qand each candidate document d∈R(q).
Cross-encoder rerankers are a staple of modern in-
formation retrieval and already feature in recent
RAG systems (Glass et al., 2022; Wang et al.,
2024b). We therefore deliberately employ an off-
the-shelf model. Each query-document pair is
jointly encoded by a transformer model, producing
a single relevance score gϕ(q, d)∈R, where ϕde-
notes the model parameters. The top- kdocuments
(ranked in descending order of gϕ(q, d)) form the
final reranked set R′(q). Only these top- kranked
passages are passed to the generator, while the rest
are discarded. Unlike the retrieval stage, where
queries and documents are encoded independently
for efficiency, reranking involves joint encoding
of each pair, which increases computational cost
but enables more accurate relevance estimation by
modeling interactions between query and document
tokens.
Generation. A pretrained autoregressive LLM
receives the concatenation of qand the top-ranked
passages and then generates the answer. Specifi-
cally, we concatenate the query with the top-ranked
passages R′(q) ={d1, . . . , d r}into a single input
sequence:
x= [q;d1;d2;. . .;dr]
The model then generates the answer token-by-
token, modeling the conditional probability:
p(y|x) =TY
t=1p(yt|y<t, x).
This way, we enable the language model to at-
tend over the complete retrieved context and gen-
erate a response grounded in multiple evidences
simultaneously.3 RAG with Question Decomposition
Anaive RAG system encodes the user query q
once and retrieves the top- kmost relevant passages.
These retrieved documents are then concatenated
with the query and used as input to the language
model, which generates an answer (Lewis et al.,
2020; Karpukhin et al., 2020). Notably, this base-
line assumes that the top-ranked passages contain
all necessary evidence, treating each question as
single-hop and ignoring multi-step reasoning or
dependencies across documents.
Our proposed pipeline augments the standard
RAG framework with two additional components:
aquestion decomposition module and a reranking
module. A comparison between our approach and
a naive RAG baseline is illustrated in Figure 1. To
address the challenges posed by multi-hop ques-
tions, which can degrade retrieval performance in
standard RAG, we (i) decompose the original query
into a set of simpler sub-queries, (ii) retrieve docu-
ments for each sub-query, (iii) merge and dedupli-
cate the retrieved results, and (iv) apply a reranker
to filter out noisy or weakly relevant candidates.
From this filtered set, only the top- kpassages R′(q)
are passed to the language model. The full pipeline
is described in Section 3.
3.1 QD Module
Given a complex question q, we define a prompt-
ing function DECOMPOSE (q, p)that produces a
set of sub-queries {˜q1, . . . , ˜qn}, where pis a
fixed natural language prompt provided to an
instruction-tuned language model. The number
of sub-queries nis not fixed but typically small,
depending on how many distinct aspects or rea-
soning steps are involved in answering q. The fi-
nal set of queries used for retrieval is defined as
Q={q} ∪DECOMPOSE (q, p), where the origi-
nal query qis always retained to preserve baseline
retrieval performance.
3.2 Reranker Module
Decomposing a complex question qinto multi-
ple sub-queries {˜q1, . . . , ˜qn}naturally increases
retrieval coverage but also introduces the risk of
noise. Since documents are retrieved independently
for each sub-query, some may be overly specific,
only partially relevant, or even unrelated to the orig-
inal question. To address this, we apply a reranking
module that scores each retrieved document based
on its relevance to the original complex query q.

Algorithm 1 Retrieval with question decomposition: Given a complex query q, the algorithm first
generates sub-queries using an LLM, retrieves documents for each, and aggregates the results. A reranker
then filters the merged candidate set, and the top- kpassages are selected for downstream generation.
Require: Query q, documents D, cutoff k
Ensure: R′(q): top-kpassages relevant to q
1:Q← {q} ∪DECOMPOSE (q0) ▷original and decomposed queries
2:C← ∅ ▷global candidate set
3:for all q∈Qdo
4: C←C∪TOP-K(q,D) ▷Add top-k candidates for each query
5:end for
6:C←RERANK (C) ▷using a pre-trained reranker
7:R′(q)←HEAD(C, k) ▷retain highest-scoring k
8:return R′(q)
This step helps to realign the expanded candidate
pool with the user’s initial intent by filtering out
documents that, while relevant to a sub-question,
do not meaningfully contribute to answering qas
a whole. The goal is to retain only passages that
clearly address distinct aspects of the original ques-
tion, improving precision in the final evidence set.
4 Experiments
We evaluate our proposed question decomposition
pipeline on established multi-hop question answer-
ing benchmarks, focusing specifically on the re-
trieval stage. This allows us to isolate and directly
measure improvements in evidence selection, in-
dependent of downstream generation. Following
prior work, we report results on the evaluation split,
as gold test labels are not publicly available.
4.1 Datasets
We use the following datasets in our experiments:
MultiHop-RAG. MultiHop-RAG (Tang and
Yang, 2024) is specifically designed for RAG
pipelines and requires aggregating evidence from
multiple sources to answer each query. In addition
to question-answer pairs, it provides gold evidence
annotations, enabling fine-grained evaluation of
both retrieval accuracy and multi-hop reasoning.
Importantly, the retrieval and generation compo-
nents are evaluated separately, allowing for focused
analysis of each component. This separation allows
fair comparison across systems.
HotpotQA. HotpotQA (Yang et al., 2018) is a
widely used multi-hop question answering bench-
mark constructed over Wikipedia. It features ques-
tions that explicitly require reasoning over two ormore supporting passages. Gold answers and an-
notated supporting facts are provided, making it
suitable for evaluating both retrieval and end-to-
end QA performance. In this work, we focus on re-
trieval accuracy to assess how well different strate-
gies recover the necessary evidence.
4.2 Baselines
To assess the individual and combined contribu-
tions of question QD and reranking within multi-
hop RAG, we evaluate four system configurations:
1.Naive RAG is the base setup in which a single
query qis embedded, and the top- kmost rele-
vant passages are retrieved from the corpus D
using dense retrieval.
2.RAG + QD modifies the retrieval stage by
introducing question decomposition. The
original query qis transformed into a set of
sub-queries {˜q1, . . . , ˜qn}, and retrieval is per-
formed independently for each element of
Q={q} ∪ { ˜qi}. The retrieved results are
merged, and the top- kpassages are selected
based on similarity scores. This setup in-
creases retrieval coverage by capturing infor-
mation across multiple query aspects.
3.RAG + Reranker retains the single-query re-
trieval approach but adds a reranking step.
To support more diverse initial candidates,
we retrieve the top- 2kpassages for the origi-
nal query ( 2×kcandidates), which are then
scored by a reranker. The top- kpassages ac-
cording to this score are selected as final input.
4.RAG + QD + Reranker combines both com-
ponents. It first decomposes the query into

sub-queries, retrieves documents for each,
merges the results, and applies reranking to
select the final top- kpassages. This configura-
tion aims to improve both evidence coverage
and ranking precision in multi-hop QA sce-
narios.
4.3 Evaluation Metrics
We report dataset-specific evaluation metrics in ac-
cordance with the protocols defined for each bench-
mark.
MultiHop-RAG. Following Tang and Yang
(2024), we report the following three retrieval-
oriented metrics:
•Hits@ kfork∈{4,10}which represents the
percentage of questions for which at least one
gold evidence appears in the top- kretrieved
passages.
•MAP@10 (mean average precision ) com-
putes the average precision at each rank po-
sition where a gold passage is retrieved, and
then averages this over all queries. We trun-
cate at rank 10.
•MRR@10 (mean reciprocal rank ) computes
the mean of the reciprocal rank of the firstcor-
rect passage, rewarding systems that surface a
gold document as early as possible. We also
truncate at rank 10.
HotpotQA. For HotpotQA, we adopt the official
QA-centric evaluation metrics introduced in the
original benchmark (Yang et al., 2018; Rajpurkar
et al., 2016). Results are reported separately for
(i) answer accuracy, (ii) supporting fact prediction,
and (iii) their joint correctness. The joint metric
constitutes a stricter criterion, requiring both the
predicted answer and the corresponding support-
ing evidence to be correct. This provides a more
comprehensive assessment of system performance
by jointly evaluating generation quality and the
relevance of retrieved evidence.
•EM(exact match ) measures whether the pre-
dicted answer exactly matches the reference
answer string.
•F1, Precision, Recall measure token-level
overlap between the predicted and reference
answers, thus allowing for partially correct
answers.•Supporting-Fact EM, F1, Precision, Re-
callare the same metrics applied to the gold-
labeled supporting facts.
•Joint EM, F1, Precision, Recall considers a
prediction correct only if both the answer and
allsupporting facts are correct. This metric
captures the system’s ability to jointly gener-
ate correct answers and identify the correct
supporting evidence.
4.4 Implementation Details
Retrieval We embed each passage chunk using
bge-large-en-v1.5 (d=1024 ) (Xiao et al., 2023).
The resulting embeddings are stored in a FAISS
IndexFlatIP index to enable exact maximum in-
ner product search. This setup ensures that any
observed gains are attributable to question decom-
position and reranking, rather than approximations
introduced by approximate nearest neighbor search
(Douze et al., 2024; Facebookresearch, 2024).
Reranker We rescore the retrieved passages us-
ing the bge-reranker-large cross-encoder (Xiao
et al., 2023). The model outputs a relevance logit
for each query–passage pair. We then sort the pas-
sages by their scores and retain the top- kpassages,
which are appended to the prompt for answer gen-
eration.
Generation Model We generate answers us-
ingQwen2.5-32B-Instruct (Qwen Team, 2024;
Yang et al., 2024), operating in bfloat16 precision.
We use maximum sequence length of 512 tokens.
Software In our implementations, we use
LangChain (LangChain, 2025), Huggingface
Transformers (Wolf et al., 2020), and faiss-cpu
(Yamaguchi, 2025). All our experiments are exe-
cuted on NVIDIA A100 GPUs with 80GB of mem-
ory.
4.5 Hyperparameters
We use the following hyperparameters across all
experiments: the number of retrieved passages is
fixed at k= 10 for all datasets, consistent with the
official evaluation settings of both HotpotQA and
MultiHop-RAG. Both sub-query generation and
answer synthesis are performed with a sampling
temperature of 0.8; and we apply nucleus sampling
with Top- p= 0.8.

5 Results
5.1 MultiHop-RAG
We present retrieval results on the MultiHop-
RAG dataset in Table 1. Question decompo-
sition ( QD) and reranking ( RR) individually im-
prove recall-oriented metrics: QDyields +4.4 per-
centage points on Hits@4 and +2.9 on Hits@10,
while RRachieves a +7.6 point gain on Hits@4.
Reranking also substantially improves MAP@10
and MRR@10. Our proposed pipeline, which
combines both modules ( QD+RR), achieves the
strongest results overall, reaching 87.2% Hits@10
and 0.635 MRR@10.
For comparison, the strongest configurations in
the original MultiHop-RAG paper (Tang and Yang,
2024), which use text-ada-002 (OpenAI, 2022)
andvoyage-02 (V oyage AI Innovations Inc., 2024)
embeddings with bge-reranker-large reranker.
Despite using a smaller embedding model, we
demonstrate strong improvements over the reported
74.7% Hits@10 and 0.586 MRR@10. Our QD+RR
thus improves Hits@10 by 16.5% and MRR@10
by 8.4%. However, we also notice that our ap-
proach falls short on MAP@10.
Interestingly, despite the larger retrieval pool
from decomposition, MAP@10 also increases
(0.322 vs. 0.274 in RR), suggesting that rerank-
ing not only filters noise but leverages the broader
context to prioritize relevant passages. These find-
ings reinforce the complementary strengths of QD
and reranking: decomposition expands coverage,
and reranking restores precision.
5.2 HotpotQA
Table 2 presents answer-level, supporting-fact, and
joint metrics on the devsplit of HotpotQA.2Apply-
ing question decomposition ( QD) alone yields only
marginal improvements over the naive RAG base-
line, with answer F 1increasing from 31.3 to 32.3
and EM from 25.4 to 26.1. Reranking ( RR) leads to
stronger gains (F 1: 32.9, EM: 26.4), demonstrating
its effectiveness in improving retrieval relevance.
The combined system ( QD+RR) achieves the best
overall results, with the highest answer EM (28.1),
F1(35.0), precision (37.1), and recall (34.8), indi-
cating that improved coverage and ranking together
lead to better evidence-grounded answers.
1Results taken from Tang and Yang (2024).
2The official test set is hidden; as we do not train new
models, we follow standard practice and evaluate on the dev
set.For supporting-fact metrics, QD+RRachieves
the highest precision (46.8), despite having lower
EM (17.9) and F 1(11.2) compared to RR, which
achieves the highest supporting-fact EM (19.6) and
F1(12.9). Interestingly, QD+RRachieves the high-
est supporting-fact and joint precision (46.8 and
23.1, respectively), even though decomposition
typically expands the retrieval pool and might be
expected to reduce precision. This suggests that
reranking effectively filters out less relevant candi-
dates, even when starting from a broader and poten-
tially noisier set. Moreover, the results indicate that
decomposed sub-queries may surface complemen-
tary evidence that, after reranking, leads to more
complete and better-aligned evidence sets. In some
cases, a single document may contain answers to
multiple sub-parts of a complex query, allowing
the system to retrieve multi-hop evidence more
efficiently than anticipated. These findings high-
light the strength of combining decomposition with
reranking: the former improves coverage, while
the latter restores precision.
5.3 Ablation: subqueries generated vs. gold
evidences
Table 3 compares the number of gold evidence sen-
tences per query with the number of subqueries
produced by the question decomposition module.
We instruct the LLM to generate at most 5 sub-
queries per query in order to keep our experiments
strictly zero-shot. Most questions require only two
or three supporting facts (e.g., 67.4% of HotpotQA
have two), yet the LLM almost always generates
exactly five subqueries (93.3% on MultiHop-RAG,
98.6% on HotpotQA), matching the prompt limit.
However, we note that allowing variable-size de-
composition could better align with actual evidence
needs.
Correlation analysis. Both Pearson and Spear-
man coefficients are near zero (Table 5), indicating
no correlation relationship between the number of
sub-queries. This suggests that the LLM does not
aim to predict the number of reasoning steps (or
“hops”), but instead produces a diverse set of fo-
cused subqueries. Importantly, our goal was not
to mirror the gold evidence count, but to ensure
broad coverage through over-complete decomposi-
tion, increasing the chance of retrieving all relevant
evidence. The near-zero correlation scores sug-
gest the model applies a fixed subquery “budget”
defined by the prompt, rather than adapting to ques-
tion complexity.

System Hits@4 Hits@10 MAP@10 MRR@10
text-ada-002 (+RR)†0.616 0.706 0.463 0.548
voyage-02 (+RR)†0.663 0.747 0.480 0.586
Naive RAG 0.611 0.781 0.217 0.464
+QD 0.655 0.810 0.238 0.498
+RR 0.687 0.781 0.274 0.574
+QD+RR(ours) 0.763 0.872 0.322 0.635
Table 1: Retrieval performance on the MultiHop-RAG eval split.†: We report the best baselines from Tang and
Yang (2024), including text-ada-002 andvoyage-002 models with reranking.
System EM F 1 P R
Naive RAG 25.4 31.3 33.1 31.2
QD 26.1 32.3 34.3 32.0
RR 26.4 32.9 35.0 32.7
QD+RR 28.1 35.0 37.1 34.8
supporting-fact metrics
Naive RAG 18.4 12.0 42.8
QD 17.0 10.6 44.1
RR 19.6 12.9 44.9
QD+RR 17.9 11.2 46.8
joint metrics
Naive RAG 8.7 5.9 20.2
QD 8.0 5.2 20.7
RR 9.5 6.4 21.4
QD+RR 8.9 5.8 23.1
Table 2: HotpotQA devresults. Upper block: answer
metrics; middle: supporting-fact metrics; lower: joint
metrics.
5.4 Efficiency
Table 4 reports end-to-end retrieval latency (exclud-
ing generation) for 250 MultiHop-RAG queries.
While Naive RAG is extremely fast (0.03s/query),
adding reranking ( RR) increases latency substan-
tially to 0.88s/query due to the cost of scoring and
sorting candidate passages with a cross-encoder.
The overhead of question decomposition ( QD) is
16.7s/query. This is primarily due to the additional
LLM inference required to generate subqueries.
When combined, the full QD+RRsystem reaches
18.9s/query, thus slower than the simple naive RAG
baseline. However, once decomposed, subqueries
can be reused (e.g., through caching) so that the
latency remains identical to the baseline. A prac-
tical implementation is trivial: keep a small key-Gold evidences Subqueries
Dataset 2 3 ≥4 3 4 5
MultiHop-RAG 42.2 30.4 15.6 0.2 5.4 93.3
HotpotQA 67.4 24.0 8.6 0.0 0.5 98.6
Table 3: Distribution of required gold evidences vs. sub-
queries generated by QD. Rows sum to 100 %; buckets
<1%are omitted.
System Total (s) Per-query (s)
Naive RAG 7.9 0.03
RR 219.8 0.88
QD 4183.9 16.7
QD+RR 4734.9 18.9
Table 4: Retrieval wall-clock times on 250 MultiHop-
RAG queries.
value store whose key is the raw user query and
whose value is the list of generated sub-queries; on
a cache hit the expensive QD LLM call is skipped
entirely. These results highlight a key tradeoff:
while QD+RRachieves the best retrieval quality
(Section 5.1), it does so at the cost of increased
latency.
6 Related Work
Retrieval-Augmented Generation and Multi-
Hop QA. RAG augments LLMs with access to
external information at inference time, address-
ing their inherent limitations in handling up-to-
date or specialized knowledge (Lewis et al., 2020).
RAG has shown promise in knowledge-intensive
tasks such as open-domain and multi-hop ques-
tion answering (QA), where single-document re-
trieval is often insufficient (Yang et al., 2018; Joshi
et al., 2017). However, RAG performance heav-
ily depends on the quality of retrieved content—

Figure 2: Absolute counts of gold evidences (blue) vs.
subqueries generated (orange). Left: MultiHop-RAG;
right: HotpotQA.
irrelevant or misleading passages can significantly
impair answer quality (Cho et al., 2023; Shi et al.,
2023; Yan et al., 2024).
Question Decomposition for Multi-Hop Re-
trieval. To better address multi-hop queries that
span multiple evidence sources, recent work has
explored decomposing complex questions into sim-
pler subqueries (Feldman and El-Yaniv, 2019; Yao
et al., 2023; Fazili et al., 2024; Xu et al., 2024;
Shao et al., 2023) using large language models as
synthetic data generator (Golde et al., 2023; Li
and Zhang, 2024). This decomposition strategy
allows models to target different aspects of a query
independently, thereby facilitating more complete
evidence aggregation (Press et al., 2023). However,
this approach is not without limitations. One no-
table issue is the "lost-in-retrieval" problem (Zhu
et al., 2025), where LLMs fail to match the re-
call performance of specialized models such as
those trained for named entity recognition (Golde
et al., 2024). Further, many of these approaches
rely on sequential subquestion resolution, which
introduces latency and increases the risk of cas-
cading errors (Mavi et al., 2024). Alternative tech-
niques involve decomposing queries using special-
ized models or fine-tuning decomposition modules
(Min et al., 2019; Srinivasan et al., 2022; Zhouet al., 2022; Wang et al., 2024a; Wu et al., 2024),
limiting their generality. Our work instead adopts a
single-step decomposition approach using general-
purpose LLMs without task-specific training, en-
suring modularity and ease of integration.
Reranking for Precision Retrieval. Reranking
methods further refine the retrieval stage by scoring
initially retrieved candidates using more expres-
sive models, typically cross-encoders (Nogueira
and Cho, 2020). These models evaluate query-
document pairs jointly, capturing fine-grained inter-
actions and significantly improving relevance over
dual-encoder architectures (Reimers and Gurevych,
2019). Reranking has proven effective in boosting
precision for multi-hop and complex QA pipelines
(Tang and Yang, 2024). Our approach leverages
cross-encoder reranking in conjunction with ques-
tion decomposition, which together enhance both
document coverage and ranking quality.
Complementary Approaches. A range of com-
plementary strategies has been proposed to opti-
mize retrieval for complex queries, including adap-
tive retrieval (Jeong et al., 2024), corrective rerank-
ing (Yan et al., 2024), and self-reflective generation
(Asai et al., 2023). Techniques such as hypothetical
document embeddings (HyDE) (Gao et al., 2022)
and query rewriting (Chan et al., 2024; Ma et al.,
2023) focus on improving the retrieval query itself.
While promising, many of these methods involve
non-trivial training or model customization. In con-
trast, our method is lightweight, model-agnostic,
and easily deployable within existing RAG archi-
tectures.
7 Conclusion
This study examined how LLM-based question de-
composition (QD) and cross-encoder reranking in-
fluence retrieval-augmented generation for com-
plex and multi-hop question answering. Across
four system variants and two datasets, the combi-
nation of QD and reranking provided the largest
gains, increasing retrieval and answer correctness,
without requiring extra training or domain-specific
tuning. Splitting a query into focused sub-queries
broadened evidence coverage, while the reranker
promoted the most relevant passages, yielding im-
provements on benchmark datasets.
But the approach is not without downsides. If a
query is already precise, decomposition can intro-
duce noise, and reranking cannot remove every

irrelevant passage. Both modules also add compu-
tation, which may be prohibitive in low-latency sce-
narios. Performance further depends on the quality
of the LLM used for sub-query generation and on
an appropriate choice of reranker.
Future work. Employing QD only when a query
is predicted to need multi-hop reasoning could pre-
serve most benefits while cutting overhead. The
incorporation of both QD and reranking inevitably
increases computational overhead, which can be
a limitation in low-latency, real-time deployments.
Future work could therefore focus on efficiency-
oriented variants, e.g. swapping in smaller instruc-
tion models for QD or using lightweight rerankers,
to keep response times low without sacrificing ac-
curacy. Additional gains may come from testing
alternative LLMs, rerankers and prompts, and from
tuning the number of sub-queries and retrieved pas-
sages. Additionally, human studies and domain-
specific evaluations can deepen our understanding
of real-world impact and clarify how generated
sub-queries relate to required evidence.
Limitations
While our approach improves multi-hop retrieval
quality, it has several limitations that warrant fur-
ther attention.
Single-hop and adverse cases. Question decompo-
sition can be counterproductive when the original
query is already specific. In such cases, subqueries
may introduce noise or distract from the original
intent. In rare instances, none of the generated sub-
queries retrieve stronger evidence than the original
query alone.
Prompt and model sensitivity. The quality of
subqueries is sensitive to both the prompt design
and the underlying LLM. This dependence may
require prompt tuning or model selection when
adapting the method to new domains or languages,
potentially limiting generalization.
Computational overhead. As discussed in §5.4,
generating Msubqueries and reranking M×k
candidate passages substantially increases latency
and GPU requirements. This motivates future work
on more efficient decomposition strategies, such as
lightweight LLMs, retrieval-aware early stopping,
or subquery caching.
Pipeline complexity. Our design adds two separate
modules to the standard RAG stack. Although both
are plug-and-play, and rerankers are already com-
monly used in RAG pipelines (Saxena et al., 2025),every extra component increases engineering over-
head, latency, and potential points of failure.
Reranker and domain dependence. The ob-
served gains rely on a strong, domain-aligned cross-
encoder reranker. When the reranker is mismatched
with the retrieval or task domain, the benefits of
decomposition may diminish or vanish entirely.
Lack of iterative retrieval. Our pipeline operates
in a single-shot manner: subqueries are generated
once and not updated based on retrieved evidence.
This limits its ability to support adaptive multi-
step reasoning, which might be necessary for more
complex tasks.
Acknowledgments
We thank all reviewers for their valuable com-
ments. Jonas Golde is supported by the Bundesmin-
isterium für Bildung und Forschung (BMBF) as
part of the project “FewTuRe” (project num-
ber 01IS24020). Alan Akbik is supported by
the Deutsche Forschungsgemeinschaft (DFG, Ger-
man Research Foundation) under Emmy Noether
grant “Eidetic Representations of Natural Lan-
guage” (project number 448414230) and under
Germany’s Excellence Strategy “Science of Intelli-
gence” (EXC 2002/1, project number 390523135).
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-RAG: Learning
to Retrieve, Generate, and Critique through Self-
Reflection. Preprint , arXiv:2310.11511.
Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo,
Wei Xue, Yike Guo, and Jie Fu. 2024. RQ-RAG:
Learning to Refine Queries for Retrieval Augmented
Generation. Preprint , arXiv:2404.00610.
Sukmin Cho, Jeongyeon Seo, Soyeong Jeong, and
Jong C. Park. 2023. Improving Zero-shot Reader
by Reducing Distractions from Irrelevant Documents
in Open-Domain Question Answering. Preprint ,
arXiv:2310.17490.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The Faiss library. Preprint ,
arXiv:2401.08281.
Facebookresearch. 2024. Faiss indexes.
https://github.com/facebookresearch/faiss/wiki/
Faiss-indexes.
Barah Fazili, Koustava Goswami, Natwar Modani, and
Inderjeet Nair. 2024. GenSco: Can Question Decom-

position based Passage Alignment improve Question
Answering? Preprint , arXiv:2407.10245.
Yair Feldman and Ran El-Yaniv. 2019. Multi-Hop Para-
graph Retrieval for Open-Domain Question Answer-
ing. Preprint , arXiv:1906.06606.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2022. Precise Zero-Shot Dense Retrieval without
Relevance Labels. Preprint , arXiv:2212.10496.
Michael Glass, Gaetano Rossiello, Md Faisal Mahbub
Chowdhury, Ankita Naik, Pengshan Cai, and Alfio
Gliozzo. 2022. Re2G: Retrieve, Rerank, Generate.
InProceedings of the 2022 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies ,
pages 2701–2715, Seattle, United States. Association
for Computational Linguistics.
Jonas Golde, Patrick Haller, Felix Hamborg, Julian
Risch, and Alan Akbik. 2023. Fabricator: An open
source toolkit for generating labeled training data
with teacher LLMs. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing: System Demonstrations , pages 1–11, Sin-
gapore. Association for Computational Linguistics.
Jonas Golde, Felix Hamborg, and Alan Akbik. 2024.
Large-scale label interpretation learning for few-shot
named entity recognition. In Proceedings of the 18th
Conference of the European Chapter of the Associa-
tion for Computational Linguistics (Volume 1: Long
Papers) , pages 2915–2930, St. Julian’s, Malta. Asso-
ciation for Computational Linguistics.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C. Park. 2024. Adaptive-
RAG: Learning to Adapt Retrieval-Augmented Large
Language Models through Question Complexity.
Preprint , arXiv:2403.14403.
Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A Large Scale Dis-
tantly Supervised Challenge Dataset for Reading
Comprehension. Preprint , arXiv:1705.03551.
Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen-tau Yih. 2020. Dense Passage Retrieval
for Open-Domain Question Answering. Preprint ,
arXiv:2004.04906.
LangChain. 2025. LangChain.
https://www.langchain.com/.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih,
Tim Rocktäschel, Sebastian Riedel, and Douwe
Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. Technical Report
arXiv:2005.11401, arXiv.Kunze Li and Yu Zhang. 2024. Planning first, ques-
tion second: An LLM-guided method for control-
lable question generation. In Findings of the Asso-
ciation for Computational Linguistics: ACL 2024 ,
pages 4715–4729, Bangkok, Thailand. Association
for Computational Linguistics.
Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and
Nan Duan. 2023. Query Rewriting for Retrieval-
Augmented Large Language Models. Preprint ,
arXiv:2305.14283.
Vaibhav Mavi, Anubhav Jangra, and Adam Jatowt.
2024. Multi-hop Question Answering. Preprint ,
arXiv:2204.09140.
Sewon Min, Victor Zhong, Luke Zettlemoyer, and Han-
naneh Hajishirzi. 2019. Multi-hop Reading Com-
prehension through Question Decomposition and
Rescoring. Preprint , arXiv:1906.02916.
Rodrigo Nogueira and Kyunghyun Cho. 2020. Pas-
sage Re-ranking with BERT. Technical Report
arXiv:1901.04085, arXiv.
OpenAI. 2022. New and improved embed-
ding model. https://openai.com/index/new-and-
improved-embedding-model/.
Ethan Perez, Patrick Lewis, Wen-tau Yih, Kyunghyun
Cho, and Douwe Kiela. 2020. Unsupervised
Question Decomposition for Question Answering.
Preprint , arXiv:2002.09758.
Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt,
Noah A. Smith, and Mike Lewis. 2023. Measuring
and Narrowing the Compositionality Gap in Lan-
guage Models. Preprint , arXiv:2210.03350.
Qwen Team. 2024. Qwen2.5: A Party of Foundation
Models! https://qwenlm.github.io/blog/qwen2.5/.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev,
and Percy Liang. 2016. SQuAD: 100,000+ Ques-
tions for Machine Comprehension of Text. Preprint ,
arXiv:1606.05250.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence Embeddings using Siamese BERT-
Networks. Technical Report arXiv:1908.10084,
arXiv.
Yash Saxena, Ankur Padia, Mandar S. Chaudhary,
Kalpa Gunaratna, Srinivasan Parthasarathy, and
Manas Gaur. 2025. Ranking Free RAG: Replacing
Re-ranking with Selection in RAG for Sensitive Do-
mains. Preprint , arXiv:2505.16014.
Zhihong Shao, Yeyun Gong, Yelong Shen, Min-
lie Huang, Nan Duan, and Weizhu Chen. 2023.
Enhancing Retrieval-Augmented Large Language
Models with Iterative Retrieval-Generation Synergy.
Preprint , arXiv:2305.15294.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan
Scales, David Dohan, Ed Chi, Nathanael Schärli, and
Denny Zhou. 2023. Large Language Models Can
Be Easily Distracted by Irrelevant Context. Preprint ,
arXiv:2302.00093.
Krishna Srinivasan, Karthik Raman, Anupam Samanta,
Lingrui Liao, Luca Bertelli, and Mike Bendersky.
2022. QUILL: Query Intent with Large Language
Models using Retrieval Augmentation and Multi-
stage Distillation. Preprint , arXiv:2210.15718.
Yixuan Tang and Yi Yang. 2024. MultiHop-RAG:
Benchmarking Retrieval-Augmented Generation for
Multi-Hop Queries. Preprint , arXiv:2401.15391.
V oyage AI Innovations Inc. 2024. V oyage AI | Home.
https://www.voyageai.com/.
Shuting Wang, Xin Yu, Mang Wang, Weipeng Chen,
Yutao Zhu, and Zhicheng Dou. 2024a. RichRAG:
Crafting Rich Responses for Multi-faceted Queries
in Retrieval-Augmented Generation. Preprint ,
arXiv:2406.12566.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing
Huang. 2024b. Searching for Best Practices in
Retrieval-Augmented Generation. In Proceedings
of the 2024 Conference on Empirical Methods in
Natural Language Processing , pages 17716–17736,
Miami, Florida, USA. Association for Computational
Linguistics.
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le
Scao, Sylvain Gugger, and 3 others. 2020. Hugging-
Face’s Transformers: State-of-the-art Natural Lan-
guage Processing. Preprint , arXiv:1910.03771.
Jian Wu, Linyi Yang, Yuliang Ji, Wenhao Huang,
Börje F. Karlsson, and Manabu Okumura. 2024. Gen-
Dec: A robust generative Question-decomposition
method for Multi-hop reasoning. Preprint ,
arXiv:2402.11166.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2023. C-Pack:
Packed Resources For General Chinese Embeddings.
Preprint , arXiv:2309.07597.
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024. Search-in-the-Chain: In-
teractively Enhancing Large Language Models with
Search for Knowledge-intensive Tasks. Preprint ,
arXiv:2304.14732.
Kota Yamaguchi. 2025. Faiss-cpu: A library for effi-
cient similarity search and clustering of dense vec-
tors.Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling.
2024. Corrective Retrieval Augmented Generation.
Technical Report arXiv:2401.15884, arXiv.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan
Li, Dayiheng Liu, Fei Huang, Guanting Dong, Hao-
ran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, and
43 others. 2024. Qwen2 Technical Report. Preprint ,
arXiv:2407.10671.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. HotpotQA: A
Dataset for Diverse, Explainable Multi-hop Question
Answering. Preprint , arXiv:1809.09600.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
ReAct: Synergizing Reasoning and Acting in Lan-
guage Models. Preprint , arXiv:2210.03629.
Ben Zhou, Kyle Richardson, Xiaodong Yu, and Dan
Roth. 2022. Learning to Decompose: Hypothetical
Question Decomposition Based on Comparable Texts.
Preprint , arXiv:2210.16865.
Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang,
and Wei Hu. 2025. Mitigating lost-in-retrieval prob-
lems in retrieval augmented multi-hop question an-
swering. Preprint , arXiv:2502.14245.
A Additional Ablation Results
Dataset Pearson (p) Spearman (p)
MultiHop-RAG −0.022 (0 .27)−0.007 (0 .71)
HotpotQA 0.017 (0 .15) 0 .012 (0 .32)
Table 5: Correlation between the number of sub-queries
and the number of gold evidences per query.