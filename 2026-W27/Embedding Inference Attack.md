# Embedding Inference Attack

**Authors**: Cedric Fitiavana Raelijohn, Sébastien Gambs, Jean-Francois Rajotte

**Published**: 2026-07-01 02:56:39

**PDF URL**: [https://arxiv.org/pdf/2607.01276v1](https://arxiv.org/pdf/2607.01276v1)

## Abstract
Embedding models are essential components of modern Information Retrieval (IR) systems, yet they are typically hidden behind APIs. Recent works have shown that dense IR system can lead to security vulnerabilities such as embedding inversion attacks. However, such attacks usually require that the attacker knows the embedding model for the attack to be applicable. In this paper, we study IR systems under a black-box setting in which the adversary observes only the unordered set of retrieved documents, without ranking or similarity scores. We demonstrate that in such contexts, tailored queries allow an adversary to identify which embedding model is in use from a set of known model candidate, which we coin as an embedding inference attack (EIA). We also show that certain queries remain discriminative even when the system includes a reranker as a potential defense mechanism. We further validate our method on a real Retrieval-Augmented Generation (RAG) system, in which the tailored queries bypass the LLM's tendency to reject inputs it does not recognize as well-formed questions. Finally, we propose and evaluate other mitigation strategies such as similarity thresholds.

## Full Text


<!-- PDF content starts -->

Embedding Inference Attack
Cedric Fitiavana Raelijohn and Sébastien Gambs and Jean-Francois Rajotte
Université du Québec à Montréal, Canada
raelijohn.cedric_fitiavana@courrier.uqam.ca,gambs.sebastien.@uqam.ca,
rajotte.jean-francois@uqam.ca
Abstract
Embedding models are essential components
of modern Information Retrieval (IR) systems,
yet they are typically hidden behind APIs. Re-
cent works have shown that dense IR system
can lead to security vulnerabilities such as em-
bedding inversion attacks. However, such at-
tacks usually require that the attacker knows
the embedding model for the attack to be appli-
cable. In this paper, we study IR systems under
a black-box setting in which the adversary ob-
serves only the unordered set of retrieved doc-
uments, without ranking or similarity scores.
We demonstrate that in such contexts, tailored
queries allow an adversary to identify which
embedding model is in use from a set of known
model candidate, which we coin as an em-
bedding inference attack (EIA). We also show
that certain queries remain discriminative even
when the system includes a reranker as a po-
tential defense mechanism. We further vali-
date our method on a real Retrieval-Augmented
Generation (RAG) system, in which the tai-
lored queries bypass the LLM’s tendency to re-
ject inputs it does not recognize as well-formed
questions. Finally, we propose and evaluate
other mitigation strategies such as similarity
thresholds.
1 Introduction
With the growth of data and the rise of artificial
intelligence (AI), IR systems have evolved into
modern applications, such as dense (or semantic)
IR, in which documents and queries are encoded
into vector representations (embeddings vectors)
using a neural network, denoted as embedding
models (Karpukhin et al., 2020a). This approach
enables semantic matching rather than exact key-
word matching, significantly improving retrieval
performance, especially in large scale datasets.
More recently, Retrieval-Augmented Generation
(RAG) has emerged as a popular paradigm combin-
ing dense retrieval with general-purpose large lan-
guage models (LLMs) to generate context-awareresponses. This is especially relevant for domain-
specific fields such as finance, healthcare and edu-
cation (Chang et al., 2024; Arslan et al., 2024).
With respect to privacy, embedding vectors are
often assumed to be safe to expose because they are
only numerical values that do not directly convey
an interpretable information for a human. However,
recent works have demonstrated the feasibility of
conducting embedding inversion attacks, which
allow to reconstruct the text represented by em-
beddings vectors (Morris et al., 2023; Zhang et al.,
2025a). For instance, this could happen in the com-
mon situation in which a RAG system is based on
publicly accessible data and is implemented to help
users navigate a large set of documents, such as
documentation, policies, etc. In this situation, the
queries could contain sensitive information such
as personal medical conditions. Embedding inver-
sion attacks often rely on strong assumptions about
the adversary’s capabilities such as access to large
sets of leaked text–embedding pairs or knowledge
of the victim’s embedding model and architecture
(Morris et al., 2023; Huang et al., 2024; Zhang
et al., 2025a). We show here how IR systems based
on publicly available datasets relaxes these assump-
tions, making them more vulnerable to inversion
attacks.
In this work, we introduce the concept of Embed-
ding Inference Attack (EIA), which aims to infer
the embedding model used by the target system in
a black-box setting in which the adversary has only
access to the API of the system. This setting is
realistic because embedding models are commonly
accessible only through API, for instance in Seman-
tic Scholar AI-powered research tool and OpenEvi-
dence searching PubMed and medical journals. As
a specific use case, we are particularly interested in
IR for education in which high school students can
ask questions about their homework but might in-
advertently leak personal information (e.g., name,
age or medical conditions of some family mem-arXiv:2607.01276v1  [cs.CR]  1 Jul 2026

bers). Knowledge of the embedding model in an
IR system such as a RAG can also unlock security
vulnerabilities other than inversion attacks, such
as corpus poisoning (Zhong et al., 2023), mem-
bership inference in a vector database (Anderson
et al., 2025) and search-engine optimization (SEO)
attacks (Ben-Tov and Sharif, 2024).To the best of
our knowledge, our work is the first black-box at-
tack aiming to infer the embedding model on IR
systems.
Contributions.Our contributions can be summa-
rized as follows.
•We propose EIA, a novel approach to infer the
embedding model of a black-box IR system.
We also show how to instantiate this attack
through a set of tailored queries allowing an
adversary to identify which embedding model
is used from a set of known candidates.
•We experimentally validate our method on
multiple open-weight embedding models and
a Retrieval-Augmented Generation (RAG)
system, in which the tailored queries bypass
the LLM’s tendency to reject inputs that do
not correspond to well-formed questions.
•We demonstrate that certain queries remain
discriminative even when the system includes
a reranker or uses similarity thresholds as pos-
sible mitigation strategies against an EIA.
•We showcase the performance of our attack on
real-world question-answering (QA) datasets,
including MS MARCO and an educational
QA dataset collected from Alloprof.
Outline.The outline of the paper is as follows.
First in Section 2, we provide the background no-
tions on information retrieval and present the sys-
tem and adversary models that we consider before
describing the related work on attacks involving
the embedding model in Section 3. Afterwards, we
introduce our EIA approach and how to implement
it by generating discriminative queries in Section 4.
Then, we describe the experimental setting in Sec-
tion 5 before reporting on the results in Section 6.
Finally, we evaluate possible countermeasures in
Section 7 before concluding in Section 8 and then
discussing the limitations.2 Background
In this section, we first provide an overview of
information retrieval setting considered before de-
scribing the systems and adversary models.
2.1 Overview of information retrieval
Embedding model and information retrieval.
In a dense IR system (Karpukhin et al., 2020b),
queries and documents are converted by an em-
bedding model into dense embedding vectors. A
desired characteristic for IR is that relevant docu-
ments should to be close to a given query in the
corresponding embedding space. Conversely, ir-
relevant documents to the query should be located
farther away. To retrieve the corresponding doc-
uments for a given query, similarity measures be-
tween their corresponding vectors and the query
are computed. Classical similarity measures in-
clude the Euclidean distance or cosine similarity,
the latter being the most commonly used (Xie et al.,
2023).
Reranker.A reranker is generally a model that is
used to refine and re-order the initial set of retrieved
documents to achieve better relevance for a given
query (Hu et al., 2019). The retrieval pipeline with
a reranker generally operates in two phases. First,
an initial retrieval step performed typically through
embedding models returns a relatively large set
of documents (e.g., top-50). Second, these doc-
uments, along with the query, are then passed to
the reranker, which selects the top- kmost relevant
documents (e.g., top-3 or top-5) (Calam, 2023).
Rerankers primarily use cross-encoder archi-
tectures, in which the query and each docu-
ment are concatenated and processed together
through a transformer model to produce a relevance
score (Calam, 2023; Lu et al., 2025). This contrasts
with the embedding models used in the initial re-
trieval, which encode queries and documents in-
dependently and compute similarity via a distance
metric (Reimers and Gurevych, 2019). The cross-
encoder approach enables deeper semantic under-
standing of the query-document relationship ob-
tained from the first retrieval stage, as it processes
each individual query-document pair (i.e., the query
paired with each of the top- kretrieved documents)
resulting in a more expensive computation, thus ex-
plaining the latency tradeoff (Calam, 2023; Pandit
et al., 2025). Recently, researchers have demon-
strated that large language models (LLMs), such as
ChatGPT and Mistral variants fine-tuned for rank-

ing tasks, can also serve as rerankers (Pandit et al.,
2025). Rather than directly generating answers,
these models evaluate the relevance of candidate
documents with respect to a query and can lever-
age their stronger reasoning capabilities to improve
ranking quality (Sharifymoghaddam et al., 2025).
2.2 System and adversary models
We consider a black-box information retrieval sys-
tem that is accessible only through an API. In
this system, a query, given as a sequence of to-
kensq∈Vn, is encoded by an embedding model
ϕ:Vn→Rdinto a fixed-length query embedding
vector e∈Rd. Similarly, the same embedding
model ϕconverts a document D∈Vn, also repre-
sented as a sequence of tokens, into a fixed-length
document embedding vectorf=ϕ(D)∈Rd.
The system ranks documents by computing the
similarity between the query embedding eiand doc-
ument embeddings fj, and returns the top- kmost
relevant documents (Zeng et al., 2024). Several
similarity metrics are commonly used to compare
vector representations in information retrieval sys-
tem. Cosine similarity is independent of vector
magnitude and calculates the cosine of the angle
between two vectors. Its value ranges from -1 to 1,
with smaller angles corresponding to higher similar-
ity. Euclidean (L2) distance, in contrast, calculates
the straight-line distance between two points in
vector space, in which smaller distances indicate
greater similarity. Dot product (inner product) simi-
larity is calculated as the product of the two vectors’
magnitudes and the cosine of the angle between
them.(Manning et al., 2008b) In this work, we use
cosine similarity exclusively, as its insensitivity to
vector magnitude is critical when comparing high-
dimensional text embeddings of varying length.
The objective of the attacker is to infer the em-
bedding model ϕvused by the IR system. In this
context, the attacker is following the black-box ad-
versary setting in which he can only query the vic-
tim system with any questions and only has access
to the (unranked) retrieved documents. Moreover,
similarly to most of the work in the literature we
make the following assumptions:
1.The retrieval system uses a publicly available
source of documents (e.g., Wikipedia) and
for a given query qi, it returns a set Siofk
unranked documents, which are outputs that
can differ only by the order of documents.
For instance, the set d1, d2, d3 is consideredto be equivalent to d3, d1, d2(Manning et al.,
2008a).
2.One of the candidate model is the victim
model. This assumption is realistic because
several retrieval-system vendors publicly dis-
close the embedding models available through
their services. For example, OpenAI and Co-
here both document their supported embed-
ding models and associated API identifiers
(OpenAI, 2024; Cohere, 2026). Thus, in these
situations the possible set of candidate embed-
ding models can be considered to be public
information.
However in contrast to other previous works,we
consider the challenging setting in which the at-
tacker does not have access to the embedding vec-
tors.
3 Related work
Model stealing.Dziedzic et al. (2023) have
shown that it is easy to steal sentence embedding
encoders via a black-box API that returns only the
vector of the input text and might also expose meta-
data including the architecture family (e.g., BERT
or RoBERTa) and possibly pre-training datasets.
Afterwards, the adversary can instantiate a pub-
licly available pre-trained transformer checkpoint
with the same architecture or at least the same out-
put dimensionality and train it with a dataset of
sentence pairs drawn from any open-source text
distribution. Another work by Tamber et al. (2025)
demonstrates that API-only access can be enough
to efficiently replicate the effectiveness of com-
mercial embedding models by distilling them into
local BERT-based models while preserving most
retrieval performance. In their work, the attacker
only sees input text and output embedding vectors
and has unrestricted access to the embedding API.
Model inference and similarity.Pasquini et al.
(2024) have shown how to send very few crafted
queries to identify specific LLMs based on their
answers. In this context, the attacker can send
queries to a victim RAG application and observe
the outputs. These outputs are then fed to a sepa-
rate ML-based inference model predicting which
of 42 candidates LLM versions is generating the
answers, achieving over 95% accuracy. Caspari
et al. (2024) compares embedding models with
various similarity measures in order to help select-
ing the most suitable for a given IR or RAG sys-

tem. The authors determine pair-wise model simi-
larities based on embedding vectors with Central
Kernel Alignments (CKA) or resulting document
retrieval. The models’ document retrievals are com-
pared with and without ranking. Their experiments
reveal clear intra -and inter -family clusters when
comparing retrieval over large top- kwhich is es-
sentially the full dataset corpus (3 600 document
chunks), but show that at realistic RAG settings
e.g., top -10, retrieval similarity between models
is low and highly variable, implying that models
with similar representations can still surface largely
distinct document sets.
Inversion attacks.Morris et al. (2023) introduce
Vec2Text, an embedding inversion attack to recon-
struct text produced by embedding models such as
OpenAI Text-embedding-ada-002 and gtr-t5-base.
In this work, the attacker is assumed to have a large
dataset of leaked text–embedding pairs from the
target encoder or from another leaked dataset, in
which case the attacker needs to train the decoder
so that it can generate candidate texts whose em-
beddings are close to a given target embedding.
Vec2Text frames the inversion attack as a con-
trolled generation in which starting from an initial
hypothesis, the model iteratively corrects the text
by repeatedly re -embedding the current guess with
black -box access to the encoder and conditioning
on the target embedding, the current embedding
and their difference to move the hypothesis in em-
bedding space towards the target.
Li et al. (2023) have propose GEIA for Gener-
ative Embedding Inversion Attack, which uses a
conditional language model training to inverse sen-
tence embeddings directly. During the training, the
decoder is trained by doing teacher forcing, which
means that at each step the decoder is fed with the
ground -truth of the previous token rather than its
own predicted token to learn to reconstruct the orig-
inal sentence from its embedding. Here also, the
attacker is assumed to have an auxiliary dataset sim-
ilar to the distribution of the dataset of the victim
and a black-box access in which he can only query
the model victim to obtain an text-embedding pairs
from his corpus.
Huang et al. (2024) have introduced TEIA for
Transferable Embedding Inversion Attack, which
is an inversion attack that learns a surrogate model
of the embedding model victim. The setting is
more restricted than other works in the sense that
the attacker cannot query or access the embeddingmodel victim but needs to have access to a small
leaked set of text-embeddings pairs from it. Huang
et al. (2024) re-implemented the GEIA attack as
a “Direct Attack” baseline and evaluate both meth-
ods on the same datasets and the same embeddings.
Across all reported dataset–encoder combinations,
TEIA consistently achieves higher reconstruction
quality than GEIA in terms of Rouge-L1and em-
bedding cosine similarity. For example, on QNLI
dataset with OpenAI embedding, RougeL improves
from 0.1433 to 0.2226 and cosine similarity from
0.2797 to 0.4772. Similarly on AGNEWS dataset,
TEIA increases RougeL from 0.0612 to 0.1271 and
cosine similarity from 0.1162 to 0.4301.
Zhang et al. (2025a) have introduced ZSinvert,
a Universal Zero-shot Embedding Inversion attack
that does not require prior decoder training but the
attacker is assumed to have access to the exact
embedding vector and black -box query access to
the encoder that produced it. ZSinvert inverts an
embedding vector by using an LLM with a mod-
ified beam search that directly optimizes cosine
similarity between the encoder’s output on candi-
date texts and the target embedding. More pre-
cisely, it first generates a rough candidate with an
open -ended prompt, then repeatedly paraphrases
and corrects this candidate while always being
guided by the target embedding without ever train-
ing an encoder-specific inversion model.
Chen et al. (2025) have proposed a few-shot in-
version attacks called ALGEN for Alignment and
Generation, which aligns the victim embeddings
to the attack space and using a generative model
to reconstruct the original text. The attacker is
assumed to have a small number of leaked embed-
ding vectors (without the corresponding original
documents) and has black -box access to the vic-
tim encoder. The attacker independently trains a
separate “local” embedding-to-text generator on
public data using their own model. Afterwards, a
linear transformation matrix is learned that maps
embeddings from the victim’s embedding space to
the attacker’s embedding space.
Finally, Kim et al. (2026) have introduced
Zero2Text, a training -free cross -domain embed-
ding inversion. In this setting, the attacker does not
need any prior leaked embedding-text pairs from
the victim and is assumed to have only a black-box
access in which he can query the victim embedding
1Rouge-L is a metric used to evaluate how well the gen-
erated summary preserves the ordering and content of the
reference (Lin, 2004).

model via its API and see its output. Zero2Text
uses a pretrained LLM to generate candidate sen-
tences and a dynamically updated linear projection
(learned via online ridge regression) to map the
attacker’s local embedding space into the victim’s
embedding space for scoring and selection.
Table 1 summarizes embedding reconstruction
attacks according to the attacker capabilities and
the underlying attack approach. All attacks require
either access to the embedding model or a set of
leaked text-embedding pairs.
4Embedding inference attack description
For a given query qi, the retrieved documents from
the victim Sv
iand candidate Sc
iare compared using
a Jaccard index, denoted by the following equation:
Ji=|Sv
i∩Sc
i|
|Sv
i∪Sc
i|.(1)
To perform the EIA, the attacker iteratively queries
the retrieval system and compares the returned doc-
ument sets. Two embedding models, denoted as ϕc
(candidate) and ϕv(victim), are considered likely
to be the same if the Jaccard index is maximized
for any question, Ji= 1,∀i . However, even if
ϕc̸=ϕ v, it is possible for the Jaccard index to
reach 1 for some queries. This approach trivially
extends to a set of model candidates {ϕi
c}by query-
ing until all but one candidate are eliminated.
Brute-force generation of discriminative queries.
The attacker can use an LLM to generate a set of
questions, one for each document of the document
corpus used by the victim system. The generated
questions are then used to query the victim model
to perform the EIA. To limit the number of API
calls on the IR system, the attacker aims to use a
question that eliminates all the candidate models.
To identify queries that effectively distinguish be-
tween candidate models, the attacker compares the
retrieved document sets produced by each model.
For a given query qi, the similarity between two
models is measured using the Jaccard index Jifrom
equation (1). Queries for which Ji̸= 1 indicate
discrepancies in retrieval are therefore considered
potentially discriminative.
LetMdenote the set of candidate models. For
each queryq i, we define an elimination score as:
E(qi) =|{m∈ M |J(v,m)
i̸= 1}|,(2)
in which J(v,m)
i represents the Jaccard index be-
tween the victim model vand a candidate modelm. The elimination score E(qi)corresponds to the
number of candidate models whose retrieval differs
from the victim model. Intuitively, higher values of
E(qi)indicate stronger discriminative power. The
attacker then selects the subset of queries that max-
imizes discrimination across all candidate models.
In particular, queries with E(qi) =|M| are ideal,
as they allow the attacker to eliminate all incorrect
candidates while queries with lower elimination
scores are less effective and are discarded directly.
A limitation of this brute-force attack is the high
number of queries required to generate and test to
find a single fully discriminative query. To address
this, hereafter we show how to create queries that
are discriminative in a more informed manner.
Generation of adversarial discriminative
queries.We have explored three types of
adversarial discriminative queries:
1.Random Strings, denoted asRnd. Str., which
consist of sequences of arbitrary characters,
such as for example “fsaduifhsdaifnjklsdan”.
These queries allow to explore parts of the
embedding space that are not covered by the
training of the LLM.
2.Non-Queries, denoted asNon-Q., are instruc-
tions rather than actual questions. An example
of Non-Queries is “Short answer only” and
the rationale behind them is that they allow to
reach parts of the embedding space not related
to documents.
3.Probing Queries, denoted asProb. Q.,
are questions designed to obtain information
about the victim system itself. Examples of
probing queries include “Who is your devel-
oper or creator?” or “Are you an open-source
model or a proprietary model?” and their main
objective is to investigate areas of the embed-
ding that are related to the model.
5 Experimental setting
5.1 Datasets
We have used two datasets for our experiments.
The first one is the MS MARCO passage develop-
ment dataset that was developed by Microsoft to
support research in machine reading comprehen-
sion and information retrieval (Bajaj et al., 2016).
This dataset is built from anonymized real user
queries issued to the Bing search engine, paired
with passages extracted from web documents. This

Attack Encoder access Training Data knowledge Core Idea
Vec2Text (Morris et al., 2023) Black-box✓ Large corpus leak of
text–embedding pairsTrain an inversion model and iteratively re-
fine candidate texts by minimizing embed-
ding differences.
GEIA (Li et al., 2023) Black-box✓ A similar distribution
corpus and queried text–
embedding pairsTrain a conditional language model to di-
rectly reconstruct text from embeddings.
TEIA (Huang et al., 2024) None✓ Small leaked set of text–
embedding pairsLearn a surrogate encoder and perform in-
version through transferability.
ZSInvert (Zhang et al., 2025a) Black-box×Target embedding only Use LLM-guided beam search with adver-
sarial decoding to optimize embedding simi-
larity without training.
ALGEN (Chen et al., 2025) Black-box✓ Small set of leaked vic-
tim embeddings onlyAlign embedding spaces and reconstruct text
using a locally trained generator.
Zero2Text (Kim et al., 2026) Black-box×No leakage Use LLM and online embedding-space pro-
jection without training a specific decoder.
Table 1: Attacker assumptions in existing embedding inversion attacks.
subset contains 7000 queries, 8.8 million passages
and 7400 relevance judgments (qrels) that was
human-annotated, indicating which documents are
considered relevant to each query. We further se-
lected only the documents having associated qrels,
which leads to a subset of 7400 documents.
The second one is an educational ques-
tion–answering dataset, created to support the de-
velopment and evaluation of question-answering
and information retrieval systems in a real-world
educational context (Lefebvre-Brossard et al.,
2023). It contains questions from K-12 students
across different subjects including mathematics,
science and geography along with explanatory an-
swers provided by educators (which we denote as
documents). This dataset is primarily in French,
as it is designed for French-speaking students in
Québec but a subset includes English queries and
documents. We then restrict our experiments to this
subset and selected 585 out of 30000 documents.
For both datasets, the documents are assumed to
be publicly accessible, aligning with the assump-
tions of our adversary model in which no private
data leakage is required.
5.2 Information retrieval pipeline
We evaluated our attack on a controlled IR pipeline
hosted on a High Performance Computing cluster,
using a single NVIDIA H100 GPU partitioned into
two GPU of 40GB instances via NVIDIA Multi-
Instance GPU (MIG) technology, along with 64
CPU cores and 64GB of RAM. The full evalua-
tion required approximately 8 GPU hours without
reranker and 12 – 15 hours when using a reranker.The IR system uses an embedding model to encode
both queries and documents before retrieving the
top-kdocuments based on cosine similarity.
Embedding models.We have used 13 open-
source embedding models that we grouped into
five families. The E5 models are BERT-based dual
encoders trained with large-scale weakly super-
vised contrastive learning on query–document pairs
for general-purpose text embeddings (Wang et al.,
2024). The English SBERT models (all-MiniLM-
L6-v2, all-mpnet-base-v2) are Siamese BERT-style
dual encoders from the Sentence-Transformers
framework, optimized for sentence-level seman-
tic similarity and retrieval (Reimers and Gurevych,
2019). The multilingual SBERT models, includ-
ing LaBSE, employ multilingual Transformer dual
encoders trained with translation ranking and con-
trastive objectives to yield language-agnostic sen-
tence embeddings (Reimers and Gurevych, 2019;
Feng et al., 2022). gtr-t5-base is a T5 encoder-
based dual encoder from the Generalizable T5
Retriever family (Ni et al., 2022), while Qwen3-
Embedding-4B from the Qwen3 family is specif-
ically designed for text embedding and ranking
tasks (Zhang et al., 2025b).
All models are lightweight dual encoders or
encoder-style architectures commonly used for
dense retrieval and semantic similarity, with em-
bedding dimensionalities ranging from 384 to 2560.
These models were selected based on their wide
use in prior work and their suitability for retrieval-
oriented evaluation, notably Brown et al. (2025).
Table 2 summarizes the models and their embed-

ding dimensionalities.
Family Model Dim.
E5e5-small, e5-small-v2 384
e5-base, e5-base-v2 768
e5-large, e5-large-v2 1024
English SBERTall-MiniLM-L6-v2 384
all-mpnet-base-v2 768
Multilingual SBERTparaphrase-multilingual-MiniLM-L12-v2 384
LaBSE 768
paraphrase-multilingual-mpnet-base-v2 768
T5 encoder gtr-t5-base 768
Qwen3 Qwen3-Embedding-4B 2560
Table 2: Open-source embedding models evaluated in
our experiments.
Retrieval parameters and query types.For
each model, we retrieve the top- kdocuments with:
k= 3 andk= 5 . We evaluate four cate-
gories of query: generated questions obtained from
Qwen2.5-1.5B-Instruct, Random Strings (Rnd.
Str.), Non-Queries (Non-Q.) and Probing Queries
(Prob. Q.) manually crafted by the attacker and
described previously in Section 4.
Reranking.We evaluate our EIA performance
with (+ reranker) and without reranker. We used the
cross-encoder/ms-marco-MiniLM-L6-v2reranker
model and use 25 documents as an initial retrieval.
5.3 RAG evaluation
To validate our approach in a more realistic deploy-
ment scenario, we conduct additional experiments
using an open-source RAG system provided by
AnythingLLM2. AnythingLLM is a free and open-
source, all-in-one AI application in which user can
host a LLM locally or build their RAG system with-
out requiring technical coding background or ex-
pensive infrastructure (Mintplex Labs, 2024). The
RAG was setup by using three embedding models
: all-MiniLM-L6-V2, nomic-embed-text-v1, e5-
small as well as one large language model for an-
swer generation : Qwen3 Vision 4B instruct. We
have chosen k= 3 for document retrieval as well
as an optional reranker.
For each of the embedding models, documents
are stored on LanceDB, which is the default lo-
cal vector database of AnythingLLM with a de-
fault tokenization of 1000 with an overlap of 20.
AnythingLLM have different ways of interact-
ing with the documents and the underlying LLM.
2https://github.com/Mintplex-Labs/
anything-llmMore precisely, we evaluated our attack using
two interaction modes. In theChat mode, the
LLM combines both the embedded documents
and its own general -knowledge base to produce
more conversational flexible responses while in
theQuery modethe system restricts its answers
strictly to information retrieved from the embedded
documents, treating each interaction as a focused
question-and-answer session.
5.4 Overall evaluation protocol
We perform a round-robin evaluation over the em-
bedding models, in which each model is treated
in turn as the victim while the remaining models
act as candidates. For each query qi, we compare
the retrieved document sets between the victim
system and candidate models using the Jaccard
index as detailed previously. This allows us to
quantify retrieval similarity and identify discrimi-
native queries for model inference. We report re-
sults across all configurations to analyze (1) the
impact of kon model distinguishability, (2) the
impact of different query types as well as (3) the
effect of reranking on the attack. Table 3 summa-
rizes of the different configurations used for the
experimentation.
Setup # Models k
IR Pipeline 13 3, 5
AnythingLLM 3 3
Table 3: Summary of experimental configurations.
6 Results
In this section, we first review the results of the
EIA in the information retrieval pipeline before
reporting the success of the attack on the RAG
framework.
6.1 Results on IR Pipeline
Discriminative queries.As illustrated in Table 4,
1249 generated queries are sufficient to successfully
discriminate among all evaluated models when re-
trieving the top three documents ( k= 3 ). Figure 1
shows the distributions of the number of randomly
selected questions needed to successfully discrimi-
nate. The histograms are produced by repeating the
experiment of 1) selecting a document at random,
2) generating a question, 3) filtering out discrimi-
nated candidate embedding models and repeating
until the victim model is the only candidate left. On

average, it usually takes few queries to discriminate
the victim model from the candidate ones.
(a) MSMarco
 (b) Alloprof
Figure 1: Distribution of the number of randomly gen-
erated questions needed to infer the embedding model.
Effect of top- kretrieved documents.The num-
ber of discriminative queries is directly influenced
by the number of retrieved documents k. As shown
in Figure 2, decreasing kdegrades the attack’s per-
formance as a restricted search space limits the op-
portunity to retrieve divergent documents across
different models. Conversely, increasing kim-
proves the inference performance as an expanded
retrieval pool naturally amplifies the probability
that different models will retrieve distinct docu-
ment sets. These results are consistent with that
Caspari et al. (2024), who demonstrated that re-
trieval similarity decreases with increasing kfor
low values ofk.
Figure 2: Effect of top- kon the performance of the
attack.
Effect of Reranker.From an attacker’s perspec-
tive, the presence and architecture of a reranker in
the target system are typically unknown. Neverthe-
less, EIA remains effective even in such situation.
More precisely as shown in Table 4, when the tar-
get system uses a reranker but the candidate models
do not, the EIA performance’s improve for all can-
didate models when k= 3. Furthermore, as previ-
ously explained, increasing the number of retrieveddocuments enhances the attack’s performance by
increasing the probability of models retrieving di-
vergent document sets.
Adversarial discriminative queries.We evalu-
ate three types of adversarial discriminative queries
as described in Section 4 : (1) Random String de-
noted as Rnd. Str., (2) Non-queries denoted as
Non-Q and (3) Probing Queries denoted as Prob.Q.
As shown in Table 4, all types of adversarial dis-
criminative queries successfully discriminate the
victim model from the candidate models. Even
when the target system employs a reranker, the at-
tack remains effective but the number of successful
adversarial discriminative queries decreases.
6.2 Results on RAG
To evaluate our attack in a realistic RAG envi-
ronment, we conducted experiments with Any-
thingLLM using the MSMarco dataset. Unlike the
controlled IR pipeline experiments, the attacker
can only interact with the system through user
queries and observe the retrieved document chunks
returned by the framework. During the experimen-
tation, we first observed that the query mode in
AnythingLLM does not apply any retrieval thresh-
old filtering unless we add a user-specified similar-
ity threshold for the retrieval. As a result, the re-
trieval behaviour is the same across different query
categories both for query mode or chat mode.
Random strings and non-queries.For random
string queries, the LLM used for the generation
generally refused to produce answers as it often
interprets the input as random characters or typo-
graphical noise. This behaviour was consistently
observed across all evaluated LLMs ready-to-use
in AnythingLLM, which includes (1) Qwen3 Vi-
sion 4B Instruct, (2) Gemma3 1B, (3) Llama3.2 3b,
(4) pHI-3.5 3.8B and (5) Mistral 7b. Nonetheless,
despite the refusal at the generation level, the re-
trieval system still returned top- kdocument chunks
indicating that retrieval was still performed even
for inputs that are semantically meaningful. This
demonstrates that the retrieval component could
remain accessible to the attacker even when the
generation model rejects the query.
For the non-queries type, we observed differ-
ent behaviors depending on whether the input con-
tained a question mark. More precisely, when the
non-query did not contain a question mark, the gen-
eration model typically responds with clarification
or refusal messages such as “there is no text content

in any of the provided context”. In contrast, when
a question mark was added to the same non-query,
the generation model often produced generic con-
firmation responses such as “yes” or “certainly”.
Despite these differences in generated responses,
the retrieval system still returned document chunks
for all non-query inputs.
Probing queries.Probing queries successfully
triggered both generation and retrieval behaviours,
allowing the attacker to observe differences in
the retrieved documents across embedding mod-
els. The experiments also evaluated the impact
of reranking with the AnythingLLM setup. We
observed that adversarial discriminative queries re-
mained effective when comparing an embedding
model with reranking against different embedding
models without reranking. The retrieved docu-
ment chunks differed sufficiently to discriminate
between candidate embedding models.
Setup Gen. Q. Rnd. Str. Non-Q. Prob. Q.
(max: 7433) (max: 11) (max: 11) (max: 11)
IR 1249 11 11 11
IR + reranker 1860 9 7 11
RAG — 11 11 11
RAG + reranker — 11 11 11
Table 4: Discriminative query results fork= 3.
Setup Gen. Q. Rnd. Str. Non-Q. Prob. Q.
(max: 7433) (max: 11) (max: 11) (max: 11)
IR 4547 11 11 11
IR + reranker 4877 11 11 11
RAG — — — —
RAG + reranker — — — —
Table 5: Discriminative query results fork= 5.
7 Potential defense mechanisms
Applying a similarity threshold reduces the effec-
tiveness of the attack by filtering out retrieved doc-
uments that do not exceed the pre-defined thresh-
old. Our proposed defence is directly applied to
the IR pipeline in which cosine similarity is de-
termined between the query embedding and each
retrieved document. Documents whose similarity
score falls below the predefined threshold are then
discarded. As shown in Table 6 for the generated
queries, adding a minimum similarity threshold of
0.25 to the IR system had almost no impact as the
system still produced 1247 discriminative queries
fork= 3 . However, increasing the threshold to 0.6
significantly reduced the number of discriminative
queries.Thr. 0.25 Thr. 0.50 Thr. 0.60 Thr. 0.75
Queryk=3k=5k=3k=5k=3k=5k=3k=5
Gen. Q.1247 4545 580 2244 194 721 8 35
Rnd. Str.9 9 0 0 0 0 0 0
Non-Q.10 10 0 0 0 0 0 0
Prob. Q.11 11 0 0 0 0 0 0
Table 6: Effect of similarity retrieval threshold on the
IR system.
While a similarity threshold can reduce the ef-
fectiveness of EIA, it may also negatively affect
retrieval quality by filtering potentially relevant
documents. To evaluate this trade-off, we mea-
sured retrieval performance on the MS MARCO
passage retrieval benchmark using mean recipro-
cal rank at 10 and recall at 1000 (MRR@10and
Recall@1000) with different similarity thresholds.
Docs/Retrievalis simply how many documents are
returned on average for each query, after applying
the similarity threshold.
Table 7 shows that the impact of similarity
threshold is model-dependent. For larger model
such ase5-large-v2, retrieval effectiveness re-
mains the same even at a threshold of 0.75, with
MRR@10 and Recall@1000 maintaining their
original values. Similarly,gtr-t5-baseshow small
degradation up to a threshold of 0.60. However,
LaBSEis particularly sensitive to the similarity
threshold as shown by the low score on MRR@10
and Recall@1000 dropping from 0.5419 to near
zero. For lighter models, such asall-MiniLM-L6-
v2andparaphrase-multilingual-MiniLM-L12-v2,
both MRR@10 and Recall@1000 scores decrease
as the threshold increase. These results suggest
that even if similarity threshold can significantly
reduce the attack, the choice of threshold should
be adapted to the retrieval model on the system in
order to preserve retrieval performance.
Another possible defence mechanism, for RAG
system only, consists in hiding the retrieved doc-
ument chunks from the user. This aligns with the
notion of “retrieved chunk leakage” and “answer
leakage” described in Bodea et al. (2026) in which
the focus is on preventing sensitive content from
being surfaced to end users. While this limits the
attacker’s ability to observe retrieved chunks and
behaviour, it also reduces system transparency, as
users can no longer verify the origin of the gener-
ated information. Zhou et al. (2024) survey paper
discusses transparency as one of six key dimen-

Metric Models 0.00 0.25 0.50 0.60 0.75
MRR@10e5-large-v2 0.3587 0.3587 0.3587 0.3587 0.3587
gtr-t5-base 0.3494 0.3494 0.3494 0.3494 0.3405
LaBSE 0.0603 0.0603 0.0462 0.0120 0.0005
all-MiniLM-L6-v2 0.3046 0.3046 0.3045 0.2997 0.2092
paraphrase-multilingual-MiniLM-L12-v2 0.1957 0.1957 0.1957 0.1945 0.1460
Recall@1000e5-large-v2 0.9777 0.9777 0.9777 0.9777 0.9777
gtr-t5-base 0.9793 0.9793 0.9793 0.9793 0.7690
LaBSE 0.5419 0.5419 0.1456 0.0173 0.0007
all-MiniLM-L6-v2 0.9653 0.9653 0.9316 0.8178 0.3937
paraphrase-multilingual-MiniLM-L12-v2 0.8586 0.8586 0.8421 0.7389 0.3397
Docs/Retrievale5-large-v2 1000.0 1000.0 1000.0 1000.0 998.6
gtr-t5-base 1000.0 1000.0 1000.0 999.5 36.8
LaBSE 1000.0 999.7 47.3 1.0 0.0
all-MiniLM-L6-v2 1000.0 1000.0 435.8 132.7 11.9
paraphrase-multilingual-MiniLM-L12-v2 1000.0 1000.0 835.2 357.3 29.7
Table 7: Effect of similarity threshold on retrieval performance.
sions of trustworthiness in RAG systems, defined
as:“Making RAG system processes and decisions
clear and understandable to users, fostering trust
and accountability”.
Finally, one could introduce noise into the re-
trieved set by adding random or synthetic docu-
ments. Although this may obscure the retrieval
patterns exploited by the attacker, it might also neg-
atively impact the quality and relevance of a RAG
system’s responses. Cuconasu et al. (2024) shows
that random documents and semantically related
but non -relevant document that do not contain the
answer (called distractors) can possibly alter model
behaviour risking reduced answer quality and ac-
curacy.
8 Conclusion
In this work, we have proposed a novel attack
called an embedding inference attack by which
an adversary can infer the embedding model used
by an information retrieval system. This knowl-
edge is often necessary to conduct embedding in-
version attacks and can even unlock other privacy
vulnerabilities. Our primary motivation is the in-
version of user queries, which may reveal private
or sensitive information about the user. The at-
tack assumes access to the document collection
used by the retrieval system, which is realistic as
many IR applications rely on publicly available
documents, as well as access to a set of candidate
embedding models, which is also realistic because
many systems use open-source embedding models
or publicly available APIs rather than embedding
models developed in a private manner.9 Limitations
This work presents some limitations that could be
addressed in future research. First, our evaluation
does not investigate the impact of document chunk
size and chunk overlap on the effectiveness of the
attack. Since these parameters directly influence
the retrieval process in RAG systems, studying
their effect could provide a better understanding
of the robustness of different configurations. In
addition, our experiments mainly focus on open-
source embedding models. Future work will ex-
tend the evaluation of the EIA to closed-source and
proprietary systems, such as models provided by
companies such as OpenAI or Google in order to
assess whether the attack remains effective in such
settings.
References
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2025. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmented
generation.International Conference on Information
Systems Security and Privacy, 2:474–485. Publisher
Copyright: © 2025 by SCITEPRESS – Science and
Technology Publications, Lda.; 11th International
Conference on Information Systems Security and Pri-
vacy, ICISSP 2025 ; Conference date: 20-02-2025
Through 22-02-2025.
Muhammad Arslan, Hussam Ghanem, Saba Munawar,
and Christophe Cruz. 2024. A survey on rag with
llms.Procedia Computer Science, 246:3781–3790.
28th International Conference on Knowledge Based
and Intelligent information and Engineering Systems
(KES 2024).
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-

drew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary,
and Tong Wang. 2016. Ms marco: A human gen-
erated machine reading comprehension dataset. In
InCoCo@NIPS.
Matan Ben-Tov and Mahmood Sharif. 2024. Gaslite-
ing the retrieval: Exploring vulnerabilities in dense
embedding-based search.Proceedings of the 2025
ACM SIGSAC Conference on Computer and Commu-
nications Security.
Andreea-Elena Bodea, Stephen Meisenbacher, Alexan-
dra Klymenko, and Florian Matthes. 2026. Sok: Pri-
vacy risks and mitigations in retrieval-augmented gen-
eration systems.Preprint, arXiv:2601.03979.
Andrew Brown, Muhammad Roman, and Barry De-
vereux. 2025. A systematic literature review of
retrieval-augmented generation: Techniques, metrics,
and challenges.Big Data and Cognitive Computing,
9(12).
James Calam. 2023. Rerankers and two-stage retrieval.
Pinecone Learn.
Laura Caspari, Kanishka Ghosh Dastidar, Saber Zer-
houdi, Jelena Mitrovic, and Michael Granitzer. 2024.
Beyond benchmarks: Evaluating embedding model
similarity for retrieval augmented generation sys-
tems. InProceedings of the Workshop Information
Retrieval’s Role in RAG Systems (IR-RAG 2024) co-
located with the 47th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval (SIGIR 2024), Washington DC, USA, 07 18,
2024, CEUR Workshop Proceedings, pages 62–70.
CEUR-WS.org.
Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu,
Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi,
Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang,
Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie.
2024. A survey on evaluation of large language mod-
els.ACM Trans. Intell. Syst. Technol., 15(3).
Yiyi Chen, Qiongkai Xu, and Johannes Bjerva. 2025.
ALGEN: Few-shot inversion attacks on textual em-
beddings via cross-model alignment and generation.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pages 24330–24348, Vienna, Austria.
Association for Computational Linguistics.
Cohere. 2026. Cohere’s embed models (details and
application). Accessed: 2026-06-03.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. Rethinking relevance: How noise and dis-
tractors impact retrieval-augmented generation. In
Proceedings of the 14th Italian Information Retrieval
Workshop (IIR 2024). CEUR-WS.
Adam Dziedzic, Franziska Boenisch, Mingjian Jiang,
Haonan Duan, and Nicolas Papernot. 2023. Sentenceembedding encoders are easy to steal but hard to
defend. InICLR 2023 Workshop on Pitfalls of limited
data and computation for Trustworthy ML.
Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Ari-
vazhagan, and Wei Wang. 2022. Language-agnostic
BERT sentence embedding. InProceedings of the
60th Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
878–891, Dublin, Ireland. Association for Computa-
tional Linguistics.
Minghao Hu, Yuxing Peng, Zhen Huang, and Dong-
sheng Li. 2019. Retrieve, read, rerank: Towards
end-to-end multi-document reading comprehension.
InProceedings of the 57th Annual Meeting of the As-
sociation for Computational Linguistics, pages 2285–
2295, Florence, Italy. Association for Computational
Linguistics.
Yu-Hsiang Huang, Yuche Tsai, Hsiang Hsiao, Hong-Yi
Lin, and Shou-De Lin. 2024. Transferable embed-
ding inversion attack: Uncovering privacy risks in
text embeddings without model queries. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 4193–4205, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020a. Dense passage retrieval for
open-domain question answering. InProceedings
of the 2020 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 6769–
6781, Online. Association for Computational Lin-
guistics.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020b. Dense passage
retrieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Doohyun Kim, Donghwa Kang, Kyungjae Lee,
Hyeongboo Baek, and Brent Byunghoon Kang.
2026. Zero2text: Zero-training cross-domain in-
version attacks on textual embeddings.Preprint,
arXiv:2602.01757.
Antoine Lefebvre-Brossard, Stephane Gazaille, and
Michel C. Desmarais. 2023. Alloprof: a new
french question-answer education dataset and its
use in an information retrieval case study.CoRR,
abs/2302.07738.
Haoran Li, Mingshi Xu, and Yangqiu Song. 2023. Sen-
tence embedding leaks more information than you
expect: Generative embedding inversion attack to
recover the whole sentence. InFindings of the As-
sociation for Computational Linguistics: ACL 2023,
pages 14022–14040, Toronto, Canada. Association
for Computational Linguistics.

Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Yang Lu, Yue Chen, and Carsten Eickhoff. 2025. Cross-
encoder rediscovers a semantic variant of bm25.
arXiv preprint.
Christopher D. Manning, Prabhakar Raghavan, and Hin-
rich Schütze. 2008a.Evaluation of unranked re-
trieval sets, chapter 8. Cambridge University Press.
Section 8.2.
Christopher D. Manning, Prabhakar Raghavan, and Hin-
rich Schütze. 2008b.Scoring, term weighting, and
the vector space model, chapter 6. Cambridge Uni-
versity Press.
Mintplex Labs. 2024. Anythingllm: A full-stack appli-
cation that turns any document, resource, or content
into context that any llm can use as references during
chatting. https://github.com/mintplex-labs/
anything-llm. Accessed: 2026-04-29.
John Morris, V olodymyr Kuleshov, Vitaly Shmatikov,
and Alexander Rush. 2023. Text embeddings reveal
(almost) as much as text. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 12448–12460, Singapore.
Association for Computational Linguistics.
Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo
Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan,
Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022.
Large dual encoders are generalizable retrievers. In
Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, pages
9844–9855, Abu Dhabi, United Arab Emirates. As-
sociation for Computational Linguistics.
OpenAI. 2024. New embedding models and api updates.
Accessed: 2026-06-03.
Tejul Pandit, Sakshi Mahendru, Meet Raval, and Dhvani
Upadhyay. 2025. The evolution of reranking models
in information retrieval: From heuristic methods to
large language models.CLNLP’25, page 15.
Dario Pasquini, Evgenios M. Kornaropoulos, and
Giuseppe Ateniese. 2024. Llmmap: Fingerprinting
for large language models.ArXiv, abs/2407.15847.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
Sahel Sharifymoghaddam, Ronak Pradeep, Andre
Slavescu, Ryan Nguyen, Andrew Xu, Zijian Chen,
Yilin Zhang, Yidi Chen, Jasper Xian, and Jimmy Lin.
2025. Rankllm: A python package for reranking withllms. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in
Information Retrieval, SIGIR ’25, page 3681–3690,
New York, NY , USA. Association for Computing
Machinery.
Manveer Singh Tamber, Jasper Xian, and Jimmy Lin.
2025. Can’t hide behind the API: Stealing black-box
commercial embedding models. InFindings of the
Association for Computational Linguistics: NAACL
2025, pages 1958–1969, Albuquerque, New Mexico.
Association for Computational Linguistics.
Liang Wang, Nan Yang, Xiaolong Huang, Binx-
ing Jiao, Linjun Yang, Daxin Jiang, Rangan Ma-
jumder, and Furu Wei. 2024. Text embeddings by
weakly-supervised contrastive pre-training.Preprint,
arXiv:2212.03533.
Xingrui Xie, Han Liu, Wenzhe Hou, and Hongbin
Huang. 2023. A brief survey of vector databases.
In2023 9th International Conference on Big Data
and Information Analytics (BigDIA), pages 364–371.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, and Jiliang Tang. 2024. The good
and the bad: Exploring privacy issues in retrieval-
augmented generation (RAG). InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 4505–4524, Bangkok, Thailand. Association
for Computational Linguistics.
Collin Zhang, John X. Morris, and Vitaly Shmatikov.
2025a. Universal zero-shot embedding inversion.
ArXiv, abs/2504.00147.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025b. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Zexuan Zhong, Ziqing Huang, Alexander Wettig, and
Danqi Chen. 2023. Poisoning retrieval corpora by in-
jecting adversarial passages.ArXiv, abs/2310.19156.
Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian,
Zheng Liu, Chaozhuo Li, Zhicheng Dou, Tsung-
Yi Ho, and Philip S. Yu. 2024. Trustworthiness in
retrieval-augmented generation systems: A survey.
ArXiv, abs/2409.10102.