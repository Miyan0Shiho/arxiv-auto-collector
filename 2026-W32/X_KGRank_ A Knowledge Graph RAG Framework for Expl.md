# X-KGRank: A Knowledge Graph RAG Framework for Explainable Recommendations via Pattern Mining and LLM Re-Ranking

**Authors**: Meenakshi Rajpurohit, Jainish Patel

**Published**: 2026-08-03 05:56:40

**PDF URL**: [https://arxiv.org/pdf/2608.01732v1](https://arxiv.org/pdf/2608.01732v1)

## Abstract
Modern recommender systems produce predictions that users cannot interrogate. The two dominant improvements, collaborative filtering and LLM-based reasoning, each fall short: collaborative filtering captures behavioural signals but offers no reasoning, while large language models (LLMs) generate fluent explanations but hallucinate and are poorly grounded in a user's history. We present X-KGRank, a knowledge graph retrieval augmented framework that unifies structural collaborative filtering with LLM-based explanation. From the MovieLens-1M dataset (6,040 users, 3,704 items, 988,129 interactions) we construct a heterogeneous knowledge graph of 9,762 nodes and 999,264 edges spanning three relation types (RATED, HAS_GENRE, and CO_RATED) persisted in Neo4j. We train a LightGCN ranker with content-aware SBERT initialization and a rating weighted BPR objective, and apply a popularity selective routing strategy that grounds long-tail items (1,855 of 3,704) in knowledge-graph paths while serving popular items from pre-trained knowledge, reducing KG-augmented generations by roughly 50%. On the MovieLens-1M test set under a 99-sample protocol, X-KGRank achieves NDCG@10 = 0.2956 and Recall@10 = 0.5371, improving over a strong popularity baseline by 17.1% on both metrics, by 15.6% on NDCG@20 (0.3449 vs. 0.2983), and by 14.6% on MRR (0.2435 vs. 0.2124). Across three LLM backbones evaluated on 16 cases, a 1.5-billion-parameter model (Qwen2.5-1.5B) matches a 7-billion-parameter model (Mistral-7B) on heuristic explanation quality (0.97 vs. 0.94), yet qualitative analysis shows the smaller model is more prone to factual fabrication.

## Full Text


<!-- PDF content starts -->

X-KGRank: A Knowledge Graph RAG Framework
for Explainable Recommendations via Pattern
Mining and LLM Re-Ranking
Meenakshi Rajpurohit
Dept. of Computer Engineering
San Jose State University
San Jose, CA, USA
meenakshi.rajpurohit@sjsu.eduJainish Patel
Dept. of Computer Engineering
San Jose State University
San Jose, CA, USA
jainish.patel@sjsu.edu
Abstract—Modern recommender systems produce predictions
that users cannot interrogate. The two dominant improvements,
collaborative filtering and LLM-based reasoning, each fall short:
collaborative filtering captures behavioural signals but offers
no reasoning, while large language models (LLMs) generate
fluent explanations but hallucinate and are poorly grounded
in a user’s history. We present X-KGRank, a knowledge-
graph retrieval-augmented framework that unifies structural
collaborative filtering with LLM-based explanation. From the
MovieLens-1M dataset (6,040 users, 3,704 items, 988,129 inter-
actions) we construct a heterogeneous knowledge graph of 9,762
nodes and 999,264 edges spanning three relation types—RATED,
HAS GENRE, and CO RATED—persisted in Neo4j. We train
a LightGCN ranker with content-aware SBERT initialisation
and a rating-weighted BPR objective, and apply a popularity-
selective routing strategy that grounds long-tail items (1,855 of
3,704) in knowledge-graph paths while serving popular items
from pretrained knowledge, reducing KG-augmented generations
by roughly 50%. On the MovieLens-1M test set under a 99-
sample protocol, X-KGRank achieves NDCG@10= 0.2956and
Recall@10= 0.5371, improving over a strong popularity baseline
by 17.1% on both metrics, by 15.6% on NDCG@20 (0.3449vs.
0.2983), and by 14.6% on MRR (0.2435vs.0.2124). Across
three LLM backbones evaluated on 16 cases, a 1.5-billion-
parameter model (Qwen2.5-1.5B) matches a 7-billion-parameter
model (Mistral-7B) on heuristic explanation quality (0.97vs.
0.94), yet qualitative analysis shows the smaller model is more
prone to factual fabrication.
Index Terms—recommender systems, knowledge graph,
retrieval-augmented generation, graph neural networks, large
language models, explainability
I. INTRODUCTION
When a user asks a recommender for “a mind-bending
thriller with unexpected plot twists,” what happens? A content-
based system that relies only on genre tags labelled “Thriller”
may retrieve both a genuine psychological thriller such as
What Lies Beneathand an unrelated disaster film such as
Twister, because both carry the same tag. A collaborative-
filtering model ranks items by behavioural similarity but offers
no user-specific explanation of why a movie was recom-
mended. An LLM-based recommender can generate fluent
explanations, but it hallucinates and is factually unreliable: it
may miss release years, invent cast members, or attribute filmsto the wrong directors. The results have limited interpretability
and give no clear signal of why they should be trusted.
These failure modes arise directly from our own experimental
analysis, illustrated in Fig. 1.
Existing frameworks each address part of this problem
but each leaves a critical gap. Graph-based collaborative-
filtering methods such as LightGCN [1], KGAT [3], and
GraphSAGE [2] provide strong behavioural ranking signals,
yet their effectiveness degrades as interaction sparsity in-
creases. LLM-based recommenders such as P5 [4] offer good
natural-language reasoning, but they hallucinate and often do
not ground recommendations in a user’s specific interaction
history. Knowledge-graph-based recommenders incorporate
structural information into ranking, but they typically use the
graph as model input rather than as evidence for explanation
generation.
A key observation behind our approach is that not every
item requires the same level of expensive grounding to produce
a trustworthy explanation. For popular items, an LLM often
already possesses the relevant pretrained knowledge; for long-
tail (less popular) items, that pretrained knowledge is less reli-
able, and knowledge-graph evidence becomes more valuable.
A popularity-selective routing strategy—open LLM prompts
for popular candidates and KG-grounded prompts for long-
tail candidates—focuses retrieval where it is most useful. We
adopt and extend this design, inspired by K-RagRec [8], into
a complete recommendation framework.
We present X-KGRank, a six-stage knowledge-graph
retrieval-augmented framework for explainable LLM recom-
mendation. X-KGRank constructs a heterogeneous knowl-
edge graph in Neo4j, trains LightGCN embeddings ini-
tialised from SBERT-derived item representations, mines
structural and community-level graph features using node2vec
and modularity-based clustering, retrieves candidate-specific
knowledge-graph paths at inference time, and routes each can-
didate through one of two LLM explanation prompts accord-
ing to its popularity bucket. On MovieLens-1M, X-KGRank
improves over a strong baseline by 17% on NDCG@10
while also producing user-specific natural-language explana-
arXiv:2608.01732v1  [cs.IR]  3 Aug 2026

Fig. 1. Two failure modes of LLM-based recommendation and X-KGRank’s knowledge-graph-grounded solution. (a) The LLM fabricates a film’s director,
cast, and year. (b) Genre-tag retrieval returnsTwister, a disaster film tagged “Thriller,” for a mind-bending-thriller query. (c) X-KGRank retrieves an explicit
rating-and-co-rating path and conditions its explanation on that behavioural evidence. All examples are drawn from our experiments.
tions grounded in graph evidence. Our work makes three main
contributions: (i) an end-to-end framework for knowledge-
graph retrieval-augmented recommendation; (ii) a popularity-
selective routing strategy; and (iii) an explanation-quality
analysis across the Flan-T5-large, Qwen2.5-1.5B, and Mistral-
7B backbones.
II. RELATEDWORK
The literature relevant to X-KGRank spans three inter-
secting lines of research: graph neural networks for struc-
tural collaborative filtering, LLM-based recommendation, and
retrieval-augmented generation over knowledge graphs.
A. Graph Neural Networks for Recommendation
Graph neural networks have become a dominant frame-
work for collaborative filtering, treating user–item interac-
tions as a graph and propagating embeddings through multi-
hop neighbourhoods. He et al. [1] introduced LightGCN,
which simplifies standard graph convolution by removing
non-linear activations and feature transformations, retaining
only neighbourhood aggregation; this achieves state-of-the-
art performance while reducing model complexity. We adopt
LightGCN as our ranker and add LLM-based explanations that
operate on retrieved KG paths.B. LLMs for Recommendation
Recent work explores LLMs as recommenders. Geng et
al. [4] proposed P5, unifying five recommendation tasks—
sequential recommendation, rating prediction, explanation
generation, review-related tasks, and direct recommendation—
under a single text-to-text framework. While flexible, the pure-
LLM approach has two weaknesses: it hallucinates, and it
lacks grounding in user-specific behavioural signals. Our work
addresses these weaknesses by retaining a structural model for
ranking and confining the LLM to an explanatory role over
retrieved KG evidence.
C. Retrieval-Augmented Generation over Knowledge Graphs
Retrieval-augmented generation (RAG) has emerged as a
remedy for both the sparsity limits of pure structural methods
and the hallucination problems of pure-LLM methods. He
et al. [7] proposed G-Retriever, which retrieves task-relevant
subgraphs and conditions an LLM on them for textual graph
question answering, showing an advantage over dense vector
retrieval. Most directly related, Wang et al. [8] introduced K-
RagRec, a knowledge-graph retrieval-augmented framework
for LLM-based recommendation that applies expensive sub-
graph retrieval selectively to cold-start items while serving
popular items through cheaper structural ranking. X-KGRank

builds on the K-RagRec design along two axes. First, we
provide an open implementation on MovieLens-1M, includ-
ing the Neo4j graph database, the LightGCN ranker, the
popularity-selective routing strategy, and the MLP projection
layer. Second, we evaluate the explanation layer across three
LLMs (Flan-T5-large, Qwen2.5-1.5B, Mistral-7B), surfacing
a previously unreported trade-off between explanation quality
and factual reliability in small models.
III. PRELIMINARIES ANDPROBLEMFORMULATION
A. Knowledge Graph and Notation
LetU={u 1, . . . , u n}denote the set of users,I=
{i1, . . . , i m}the set of movies, andCthe set of content
categories (genres). We model their relationships as a hetero-
geneous knowledge graph
G= (V,E, τ),V=U ∪ I ∪ C,(1)
where each edge is assigned a relation type by the function
τ:E → Rwith relation set
R={RATED,HAS GENRE,CO RATED}.(2)
A RATED edge(u, i)carries an integer rating from1to5;
a HAS GENRE edge(i, c)links an item to a genre; and
a CO RATED edge(i, j)carries a weightw ijequal to the
number of users who rated both items.
IV. METHODOLOGY
In this section we introduce the key concepts used to
build the system and then describe each component of the
framework in detail. Fig. 2 gives a high-level overview of
the complete pipeline, and Fig. 3 expands each stage with
implementation detail.
A. Knowledge Graph Construction
We construct a knowledge graphG= (V,E, τ)whose
vertex setV=U ∪ I ∪ Ccomprises users, items (movies),
and content nodes (genres). Edges are of three types: RATED,
HAS GENRE, and CO RATED. Fig. 4 shows the resulting
schema and the three relation types.
The RATED relation carries ratings in the range1–5. The
HAS GENRE edge connects an item to its genre nodes. The
CO RATED edge encodes behavioural co-occurrence and lets
the LLM generate the crucial “users who ratedAalso rated
B” explanation; without CO RATED edges, the only available
path between two movies would pass through user nodes,
which carry no semantic context for detailed explanations.
For example, in MovieLens-1M, User 1007 ratedRaiders of
the Lost Ark(1981) five stars, and the knowledge graph links
this behaviour to related items, helping surface user-relevant
relationships.
By converting the MovieLens-1M tables into a knowledge
graph, Neo4j provides meaning and context that plain lists
and tables do not, allowing us to uncover connections use-
ful for recommendation. The full knowledge graph contains
9,762 nodes (6,040 users, 3,704 movies, and 18 genres) and
999,264 edges: 988,129 RATED, 6,190 HAS GENRE, and4,945 CO RATED. This structured context provides the LLM
with more relevant grounding than semantic search alone.
B. Structural Representation Learning
LightGCN propagates information over a graph of users and
items connected by rating edges. It removes feature transfor-
mation and non-linear activation, keeping only neighbourhood
aggregation, which is well suited to user–item interactions.
Our LightGCN uses3propagation layers and an embedding
dimension of128, yielding a model with1,296,384parame-
ters.
Sentence-BERT (SBERT) turns text into vector embeddings
that capture semantic meaning; instead of starting from ran-
dom weights, the model begins with sentence-level semantic
similarity. Each movie is represented using its title and genre,
encoded with theall-MiniLM-L6-v2SBERT model into
a384-dimensional semantic vector, so movies with similar
titles and genres receive nearby representations. Because the
recommender uses a128-dimensional embedding space, a lin-
ear projection layer maps each384-dimensional SBERT vector
into the LightGCN dimension. Rather than initialising item
embeddings randomly, we therefore start from embeddings
that already encode semantic information about each movie;
during training these embeddings are updated by user–item
interactions, so the model learns both user-preference patterns
and movie content. This design directly targets the cold-start
problem: items with few ratings would otherwise behave as
noise after training, and SBERT initialisation gives such items
a meaningful starting representation. The item score for a user
is the inner product of their learned embeddings,
s(u, i) =e⊤
uei,(3)
and the ranker is trained with a rating-weighted BPR objec-
tive using a 50/50 mix of hard and uniform negatives. The
validation NDCG@10 of the trained ranker is0.3122.
C. Graph Pattern Mining
LightGCN learns from direct user–movie interactions but
does not capture long-range graph patterns. We use two
graph-mining techniques: node2vec/DeepWalk embeddings
and greedy modularity-based community detection.
node2vec performs random walks through the graph,
traversing edges in sequence (e.g., user→movie→user).
From48,810random walks of length20, a skip-gram model
with negative sampling produces64-dimensional vectors for
all9,762nodes. Movies with similar graph structure receive
similar embeddings even when no single user rated both,
giving the system structural knowledge beyond direct ratings.
We additionally apply greedy modularity maximisation
(Clauset–Newman–Moore) and find11communities with
modularityQ= 0.082. The size distribution is bimodal: two
large communities—one dominated by movies and one by
users—hold95.1%of all nodes, while the remaining nine
range from446nodes down to single digits. This reflects
the underlying bipartite topology, in which users and movies
form the two principal components. We do not interpret these

Fig. 2. High-level overview of the X-KGRank pipeline. A user and a candidate item set enter a LightGCN ranker trained over a knowledge graph stored in
Neo4j. Each candidate is routed by popularity; the LLM generates explanations while the final ranking is determined by LightGCN scores.
Fig. 3. Detailed X-KGRank architecture. Offline (left): (1) a heterogeneous knowledge graph is stored in Neo4j; (2) a LightGCN ranker is trained with
SBERT-initialised embeddings; (3) node2vec and community features are mined. At inference (right): (4) candidates are routed by popularity; (5) cold items
trigger a four-tier KG path retrieval over an in-memory NetworkX projection; (6) the retrieved path is serialised into a text prompt for explanation, while
ranking stays fixed by LightGCN scores.
communities as definitive categories; instead, they serve as coarse neighbourhood context for downstream retrieval and

Fig. 4. Schema of the X-KGRank knowledge graph. Three node
types (User, Movie, Genre) are connected by three relation edges:
RATED (User→Movie), HAS GENRE (Movie→Genre), and CO RATED
(Movie→Movie).
Fig. 5. A 24-node subgraph of the X-KGRank knowledge graph showing
User, Movie, and Genre nodes connected by RATED, HAS GENRE, and
CO RATED edges.re-ranking (Fig. 6).
D. Popularity-Selective Routing
For very popular items, the LLM has typically already
learned the relevant facts from pretraining; for less popu-
lar long-tail items, the knowledge graph provides concrete
behavioural evidence that the LLM lacks. Applying KG re-
trieval only where it is useful makes the system efficient.
We define item popularityπ(i)as the number of training
interactions involving itemi, and the median popularityπ∗=
median i∈Iπ(i). Items are split into two groups,
Icold={i:π(i)≤π∗},I warm={i:π(i)> π∗},(4)
where cold (long-tail) items benefit from KG retrieval and
warm (head) items can be served from pretrained knowledge
without graph augmentation. The median split (p= 0.50)
yields1,855cold and1,849warm items; note that interactions
are not balanced, because popular items receive many more
ratings despite the two groups containing the same number of
items.
Before invoking the LLM, the system retrieves20candi-
date items, either from an SBERT–FAISS index for natural-
language queries or from the top-scoring LightGCN candidates
for user-ID queries. It then checks each candidate’s popularity.
If the item is cold, the system performs KG retrieval and
extracts a2-hop path connecting the user to the item from
the in-memory NetworkX projection of the graph, and inserts
this path into the LLM prompt as grounding context. If the
item is warm, no KG retrieval is performed and the LLM
receives an open prompt, relying on its pretrained knowledge
of the item and the user’s preferences. This reduces expensive
path-retrieval queries by roughly50%.
E. MLP Projector to the LLM Embedding Space
The knowledge graph provides each movie with a64-
dimensional node2vec vector, whereas the Flan-T5-base en-
coder operates in a768-dimensional language-embedding
space. To bridge this gap we train a small neural network
that maps KG vectors into the LLM space. The KG em-
beddingzKG
i∈R64captures a movie’s neighbourhood, its
CO RATED relationships, and its position within graph com-
munities, but is only meaningful inside the graph embedding
space; the Flan-T5-base text embeddingt i∈R768captures
language meaning from the movie’s title and genre. We
therefore train a projectorf θ:R64→R768, a multi-layer
perceptron
fθ(z) =W 3LN 
GELU(W 2LN(GELU(W 1z)))
,(5)
whose output is normalised to unit length. GELU activations
let the network learn complex relationships between graph
structure and language meaning, and layer normalisation sta-
bilises training (preventing the degenerate solution of one
identical output for every movie). The projector has281,088
trainable parameters, far fewer than the220M parameters of
Flan-T5-base, and aligns the two representations. For each of
the3,704movies present in both the knowledge graph and the

Fig. 6. Graph pattern mining. Left: PCA of node2vec embeddings by node type. Right: 11 communities from greedy modularity maximisation (Q= 0.082);
the two largest hold95.1%of nodes.
text metadata, we build a target text embedding by passing the
title and genre through the Flan-T5-base encoder and mean-
pooling the token embeddings into a single768-dimensional
vector.
We use an InfoNCE objective rather than a simple re-
construction loss, because reconstruction drives the projected
vectors toward the average text vector (mean regression),
whereas InfoNCE keeps each movie individually identifiable,
which is what retrieval and re-ranking require:
LInfoNCE =−logexp 
sim(f θ(zi),ti)/τ
P
jexp 
sim(f θ(zi),tj)/τ, τ= 0.07.
(6)
A smaller temperature sharpens the distinction between similar
and dissimilar items. The training loss decreased from4.01at
epoch10to3.11at epoch40; a uniform random projector over
a batch of256would yield an expected loss oflog 256 = 5.55,
so the achieved value represents a44%reduction, indicating
that the projector has learned a non-trivial alignment. After
training, the projector is frozen so that the KG-to-LLM align-
ment stays consistent. The projector establishes an alignment
between the KG and LLM embedding spaces; in the runtime
pipeline reported here, however, the retrieved KG path is
supplied to the LLM as text rather than as a soft-prompt
embedding, and deploying the projected embedding as a soft
prompt is left to future work.
F . LLM Re-Ranking with Grounded Explanations
A good explanation should cite real evidence: that the user
liked a particular movie, that the movie shares a genre with
the recommendation, that two movies are connected by a
CO RATED edge, or that the candidate is linked through a
graph path. Such evidence reduces hallucination.
Every recommendation should expose a path the user can
understand, which makes the system more trustworthy. The
system tries to find a path between the user and the candidate,but the graph is sometimes sparse and a clear path does not
always exist. We therefore use a four-tier fallback strategy, be-
cause the LLM always needs something to ground its prompt.
(1)Direct shortest path: the system finds the shortest path
between the user and the movie—for example, User 2652 rated
Network(1976) five stars—which is high-quality evidence.
(2)Shared-genre bridge: if no direct path is found, the system
connects the user and candidate through a shared genre; this is
weaker but still understandable. (3)CO RATED behavioural
bridge: if the genre path fails, the system uses a behavioural-
similarity path based on co-rating patterns rather than content.
(4)Embedding-similarity fallback: if all graph paths fail, the
system returns a placeholder edge—the weakest grounding,
but enough to prevent complete failure.
The LLM receives a prompt containing the user profile (their
five most highly rated training items), the candidate metadata
(the recommended movie and its genre), and, for cold items,
the retrieved KG path; warm items receive an open prompt.
The prompt forces a fixed two-sentence output, which keeps
inference cost low and forces the model to state only the main
reasons. We compare three instruction-tuned LLMs under
the same prompt: Flan-T5-large (780M parameters, encoder–
decoder), Qwen2.5-1.5B-Instruct (1.5B parameters, decoder),
and Mistral-7B-Instruct-v0.2 (7B parameters, decoder). For
Flan-T5 we use beam search with4beams; for Qwen and
Mistral we use greedy decoding, which makes evaluation more
consistent. Although the LLM produces explanations in natural
language, the final ranking is determined by LightGCN scores,
so the LLM’s role is explanation over the retrieved KG path.
V. EXPERIMENTALSETUP
A. Dataset
We evaluate on MovieLens-1M, which contains roughly one
million1–5star ratings from6,040users on3,704movies.
The dataset is partitioned into988,129training,6,040valida-

TABLE I
DATASET ANDKNOWLEDGE-GRAPHSTATISTICS(MOVIELENS-1M)
Statistic Value
Users 6,040
Movies 3,704
Genres 18
Train interactions 988,129
Validation interactions 6,040
Test interactions 6,040
Avg. ratings per user 165.4
KG nodes 9,762
KG edges 999,264
RATED edges 988,129
HAS GENRE edges 6,190
CO RATED edges 4,945
Bipartite graph density 0.088
tion, and6,040test interactions under leave-one-out splitting,
with each user contributing at least20ratings (5-core filter-
ing). The constructed knowledge graph contains9,762nodes
and999,264edges, with RATED (988,129), HAS GENRE
(6,190), and CO RATED (4,945). Table I summarises the
dataset statistics.
B. Evaluation Protocol
We adopt a 99-sample protocol [9]: for each user in the
test set, the held-out positive item is paired with99sampled
negatives, yielding a100-item ranking pool. All metrics are
computed over this pool and averaged across test users.
C. Metrics
We report four standard ranking metrics: (a) NDCG@K,
which captures both relevance and position; (b) Recall@K,
the fraction of held-out positives appearing in the top-K;
(c) HR@K, the fraction of users with at least one positive in
the top-K; and (d) MRR, the mean reciprocal rank of the first
relevant item. We report NDCG and Recall atK∈ {5,10,20}
and HR atK= 10. For explanation quality (Section VI-B) we
report a quality score that combines length adequacy, reference
rate, sentence structure, and the presence of specific reasoning
keywords, scaled to[0,1]. This is a proxy metric; rigorous
evaluation would require human annotation.
D. Baselines
We compare against three baselines.Randomassigns each
candidate in the100-item pool a uniform random score.
Popularityscores each candidate by its training interaction
countπ(i); because MovieLens-1M is long-tailed, this is a
strong baseline.LightGCN+SBERTis the structural ranker
trained without KG path retrieval or LLM re-ranking, and
measures the contribution of the structural model alone.
E. LLM Backbones
The ranking and explanation stages are evaluated with
three instruction-tuned LLMs: Flan-T5-large, Qwen2.5-1.5B-
Instruct, and Mistral-7B-Instruct-v0.2. The MLP projector that
maps node2vec embeddings into the LLM embedding space
targets the smaller Flan-T5-base for alignment efficiency.TABLE II
RANKINGPERFORMANCE ONMOVIELENS-1M (99-SAMPLEPROTOCOL)
Metric Random Popularity X-KGRank
NDCG@5 0.0267 0.20460.2400
NDCG@10 0.0434 0.25250.2956
NDCG@20 0.0692 0.29830.3449
Recall@5 0.0457 0.30990.3645
Recall@10 0.0980 0.45860.5371
Recall@20 0.2014 0.64010.7327
HR@10 0.0980 0.45860.5371
MRR 0.0502 0.21240.2435
F . Implementation Details
All stages are implemented in PyTorch and PyTorch
Geometric, with HuggingFace Transformers for the LLM
backbones,sentence-transformersfor SBERT encod-
ing, NetworkX for graph construction and traversal, and
FAISS for nearest-neighbour retrieval. LightGCN training,
node2vec/DeepWalk training, and the MLP projector are
trained on a single NVIDIA A100 GPU via Google Co-
lab Pro in fp32, and LLM inference is performed in fp16.
End-to-end training across all stages takes approximately
three hours. Code, trained checkpoints, and configuration
files are released at https://github.com/MeenakshiRajpurohit/
graph-rag-recommend.
VI. RESULTS
A. Main Results
Table II reports ranking performance on the MovieLens-1M
test set under the 99-sample protocol.
Random ranking on a100-item pool with a single positive
is a degenerate baseline included only as a sanity check, so
the large gap between Random and the other methods is not
informative. The meaningful comparison is X-KGRank versus
Popularity, which is strong on MovieLens-1M because most
interactions concentrate on popular films; beating it requires
recovering user-specific preference signals. X-KGRank im-
proves over Popularity by+17.1%on NDCG@10 (0.2956vs.
0.2525),+17.1%on Recall@10 (0.5371vs.0.4586),+15.6%
on NDCG@20 (0.3449vs.0.2983), and+14.6%on MRR
(0.2435vs.0.2124). The relative improvement narrows at
K= 20(Recall@20 lift is+14.5%), indicating that X-
KGRank surfaces additional relevant items beyond popularity-
based ranking.
B. LLM Backbone Comparison
To assess how the choice of language model affects ex-
planation quality and inference cost while holding the up-
stream pipeline (LightGCN, KG, KG paths, prompt) fixed, we
evaluate three instruction-tuned LLMs on16test cases drawn
from four randomly sampled active users. Each LLM produces
two-sentence explanations per recommendation, scored by the
quality metric defined in Section V-C. Table III reports the
mean quality and latency.
The per-case quality scores are reported in Fig. 7, with
the summary in Fig. 8. Qwen2.5-1.5B and Mistral-7B are

TABLE III
LLM BACKBONECOMPARISON(16 CASES)
LLM Params Quality (Mean) Speed (s)
Flan-T5-large 780M 0.46 1.1
Qwen2.5-1.5B-Instruct 1.5B 0.97 4.2
Mistral-7B-Instruct-v0.2 7B 0.94 4.5
roughly tied, and both clearly outperform Flan-T5-large. Qwen
achieves the highest mean quality (0.97), edging Mistral (0.94)
by a small margin. The gap between the two is dominated
by a handful of cases in which Mistral produces tokeniser
artefacts (e.g., concatenated tokens such as “amust watch” or
“BraveHeart(19five)”) that lower its score; these are surface-
level glitches rather than substantive content failures. Flan-T5-
large lags far behind at0.46, producing truncated, repetitive,
and partial explanations across many cases; its only advantage
is speed (1.1s vs.4.2and4.5s).
That a1.5B-parameter model matches a7B-parameter
model on this task is the most substantive efficiency finding
from these comparisons: Qwen2.5-1.5B achieves marginally
higher explanation quality than Mistral-7B at less than a
quarter of the parameter count and identical latency. Where
memory and cost are constraints, Qwen offers a clear ben-
efit. We interpret this as evidence that grounded explanation
generation—where the KG path is retrieved—reduces depen-
dence on model scale, since much of the factual content is
supplied externally by the KG path rather than recalled from
the model’s parameters.
The quality metric, however, is a proxy. It measures proper-
ties such as sentence length, title reference, sentence structure,
and keyword presence, but does not directly assess factual
accuracy. Manual inspection of high-scoring outputs reveals
factual errors, including misattributed directors, fabricated co-
stars, and incorrect release years (documented in Section VII).
The16-case comparison is sufficient to rank the three models
(Qwen≈Mistral≫Flan-T5) but does not support the precise
Qwen-versus-Mistral margin.
VII. QUALITATIVEANALYSIS
To illustrate X-KGRank’s behaviour concretely, we walk
through cases drawn from a natural-language RAG query
demonstration. The first shows the system working as in-
tended; the second shows genuine limitations. Both use the
SBERT–FAISS configuration for candidate retrieval and KG
path extraction.
A. Success Case: “Moving drama about family and loss”
For this query, SBERT–FAISS retrieved six candidates; the
top three wereMy Family(1995, score0.60),Two Family
House(2000, score0.56), andThe Funeral(1996, score0.54).
All three are dramas aligned with the query, indicating that the
SBERT semantic match operates correctly. For each candidate,
the system extracted a two-hop KG path connecting the user
to the candidate through a shared dramatic film:#1 My Family (1995)
User 1 --[RATED 4.0]--> Sixth Sense, The (1999)
Sixth Sense, The (1999) --[RATED 5.0]--> User 62
#2 Two Family House (2000)
User 1 --[RATED 4.0]--> E.T. the Extra-Terrestrial
(1982)
E.T. the Extra-Terrestrial (1982) --[RATED 4.0]-->
User 173
#3 The Funeral (1996)
User 1 --[RATED 5.0]--> Schindler’s List (1993)
Schindler’s List (1993) --[RATED 3.0]--> User 225
These paths surface meaningful behaviour: viewers who rated
films likeSixth Sense,E.T., andSchindler’s Listalso engage
with highly dramatic family films. The intermediate users (62,
173, 225) are not semantically meaningful in themselves; they
are the structure through which the recommendations acquire
behavioural grounding. The Flan-T5-large explanations are
weak in this case, restating the query rather than reasoning
from the retrieved path.
B. Failure Case: “Mind-bending thriller with unexpected plot
twist”
For this query, SBERT–FAISS retrievedWhat Lies Beneath
(2000, similarity0.55),A Simple Plan(1998,0.54), and
Twister(1996,0.54). The first two are appropriate thrillers that
match the query intent.Twister, however, is a tornado-disaster
film whose genre includes “Thriller” but whose narrative—
storm-chasing scientists pursuing tornadoes—has no relation-
ship to a mind-bending or twist-driven plot. This is a clear
retrieval error. The retrieved KG paths also reveal a limitation:
#1 What Lies Beneath (2000)
User 1 --[RATED 4.0]--> Girl, Interrupted (1999)
Girl, Interrupted (1999) --[RATED 3.0]--> User 90
#3 Twister (1996)
User 1 --[RATED 4.0]--> Girl, Interrupted (1999)
Girl, Interrupted (1999) --[RATED 3.0]--> User 90
The paths forWhat Lies BeneathandTwisterare identi-
cal: both route through the same intermediate film (Girl,
Interrupted) and the same intermediate user (90). Because
the demo user has a limited rating history, the shortest-path
search repeatedly returns the same high-degree bridge node
regardless of the candidate, producing paths that carry no
candidate-specific information; the KG path therefore does
not discriminate between the appropriateWhat Lies Beneath
and the inappropriateTwister. The LLM compounds both
errors: forTwister, Flan-T5-large generates “Twisteris a mind-
bending thriller with unexpected plot twists,” a factually false
assertion that simply projects the query words onto the candi-
date rather than reasoning from evidence. This case exposes
two limitations: retrieval operates on genre tags rather than
narrative content, so any film tagged “Thriller” is eligible for
thriller-intent queries; and KG paths degrade when users with
sparse history fail to differentiate candidates.

Fig. 7. Per-case explanation quality across 16 (user, movie) pairs for the three LLM backbones.
Fig. 8. LLM explanation quality. Left: mean±standard deviation over 16 cases. Right: score distributions.
C. LLM Case Study: Factual Accuracy versus Fluency
To illustrate the qualitative differences identified in Sec-
tion VI-B, we examine the explanations generated by
Qwen2.5-1.5B and Mistral-7B for an identical set of
recommendations. User 2909’s profile spans classic and
mid-century cinema—Casablanca(1942),Double Indemnity
(1944),Lawrence of Arabia(1962),Back to the Future(1985),
andMurder in the First(1995)—and LightGCN ranked four
candidates reached by distinct KG paths:
52 Pick-Up (1986):
User 2909 --[RATED 4.0]--> My Fair Lady --[RATED
4.0]--> User 183
Bram Stoker’s Dracula (1992):
User 2909 --[RATED 4.0]--> Awakenings --[CO_RATED
w=198]--> Dracula
Batman Forever (1995):
User 2909 --[RATED 5.0]--> Payback --[CO_RATED
w=263]--> Batman Forever
Lifeboat (1944):
User 2909 --[RATED 4.0]--> North by Northwest
--[RATED 3.0]--> User 23HereBram Stoker’s DraculaandBatman Foreverare reached
through CO RATED edges, while52 Pick-UpandLifeboat
are reached through shared-user RATED bridges.
Mistral-7B produces factually accurate explanations. For
Batman Foreverit correctly identifies the cast—“an iconic
performance by Val Kilmer as Batman and Jim Carrey as The
Riddler”—and forLifeboatit gives an accurate plot summary:
“a group of strangers are stranded at sea on a lifeboat after their
ship is sunk by a German U-boat.” These are correct, verified,
and specific. Mistral’s weakness is tokenisation defects such
as the concatenated “amust watch.”
Qwen2.5-1.5B produces fluent but fabricated explanations.
For52 Pick-Up, Qwen confidently states that the film was
“directed and co-written by John Landis, with an ensemble
cast including Dan Aykroyd as Frank McHale Jr.”—a fabri-
cation. The 1986 film was directed by John Frankenheimer
from an Elmore Leonard novel and features neither Landis nor
Aykroyd, and Qwen additionally mistakes the year as 1973.
The output is grammatically polished and rich in specific-
sounding detail, which is precisely what makes the fabrication

dangerous: the heuristic quality metric ranks this fluent-but-
false output above Mistral’s accurate one.
VIII. LIMITATIONS
We analyse the limitations of X-KGRank to contextualise
its contributions and guide future work.(a) Single domain.All
results are obtained on MovieLens-1M; we have not validated
the framework on another domain such as e-commerce or
news.(b) Genre-tag retrieval.As the failure case in Sec-
tion VII-B shows, SBERT–FAISS retrieves over movie titles
and genre tags rather than plot content; richer content represen-
tations (plot summaries, reviews) would mitigate this but are
left to future work.(c) Path degeneracy for sparse histories.
When a user has few highly rated films, the shortest-path
search returns the same high-degree bridge node, producing
candidate-independent paths.(d) Heuristic explanation metric.
Our quality metric captures sentence length, title reference,
structure, and keyword presence, but does not assess factual
accuracy; manual inspection confirms that even high-scoring
explanations contain factual errors, and rigorous evaluation
would require human annotation.(e) Low community modular-
ity.The node2vec communities have modularityQ= 0.082,
which is low for well-separated networks; we use them only as
weak contextual features and do not rely on hard partitioning,
but more structured graphs or alternative community methods
are left to future exploration.
IX. CONCLUSION
We presented X-KGRank, a knowledge-graph retrieval-
augmented framework for explainable recommendation that
integrates structural collaborative filtering, graph pattern min-
ing, and LLM-based explanation generation. We constructed
a knowledge graph from MovieLens-1M and trained a Light-
GCN model with SBERT-based content-aware initialisation
and a rating-weighted BPR objective. A popularity-selective
routing strategy applies expensive KG retrieval only to long-
tail items. On MovieLens-1M, the full pipeline improved
over the popularity baseline by17%on NDCG@10 and
Recall@10. A comparison across three LLMs shows that
Qwen2.5-1.5B matches the7B-parameter Mistral-7B on ex-
planation quality, yet qualitative analysis shows that smaller
models are more prone to factual errors—suggesting that exter-
nal KG grounding reduces the dependence of recommendation
quality on model scale, while factual explanation reliability
still benefits from larger LLMs. Future directions include
human evaluation of explanation quality, validation across
additional domains, and content-attribute knowledge graphs
built from plot summaries and reviews.
X. PROJECTARTIFACTS ANDIMPLEMENTATION
RESOURCES
Because this project involved a complete implementation
of the X-KGRank framework, all source code, notebooks,
datasets, and visualisations have been organised and madepublicly accessible. These resources cover the full pipeline—
from knowledge-graph construction and structural represen-
tation learning to popularity-selective routing, KG path re-
trieval, and LLM-based explanation—ensuring transparency,
reproducibility, and ease of verification.
A. Source Code Repository
The complete Python implementation—including the
knowledge-graph construction scripts, the LightGCN ranker,
graph pattern mining, the MLP projector, and the LLM expla-
nation pipeline—is hosted on GitHub:
https:
//github.com/MeenakshiRajpurohit/graph-rag-recommend
REFERENCES
[1] X. He, K. Deng, X. Wang, Y . Li, Y . Zhang, and M. Wang, “LightGCN:
Simplifying and Powering Graph Convolution Network for Recommen-
dation,” inProc. 43rd Int. ACM SIGIR Conf. Research and Development
in Information Retrieval, 2020, pp. 639–648.
[2] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive Representation
Learning on Large Graphs,” inAdvances in Neural Information Pro-
cessing Systems (NeurIPS), 2017, pp. 1024–1034.
[3] X. Wang, X. He, Y . Cao, M. Liu, and T.-S. Chua, “KGAT: Knowl-
edge Graph Attention Network for Recommendation,” inProc. 25th
ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining, 2019,
pp. 950–958.
[4] S. Geng, S. Liu, Z. Fu, Y . Ge, and Y . Zhang, “Recommendation as
Language Processing (RLP): A Unified Pretrain, Personalized Prompt &
Predict Paradigm (P5),” inProc. 16th ACM Conf. Recommender Systems
(RecSys), 2022, pp. 299–315.
[5] Y . Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao,
“Large Language Models are Zero-Shot Rankers for Recommender
Systems,” inProc. European Conf. Information Retrieval (ECIR), 2024.
[6] K. Bao, J. Zhang, Y . Zhang, W. Wang, F. Feng, and X. He, “TALLRec:
An Effective and Efficient Tuning Framework to Align Large Language
Model with Recommendation,” inProc. 17th ACM Conf. Recommender
Systems (RecSys), 2023, pp. 1007–1014.
[7] X. He, Y . Tian, Y . Sun, N. V . Chawla, T. Laurent, Y . LeCun, X. Bresson,
and B. Hooi, “G-Retriever: Retrieval-Augmented Generation for Textual
Graph Understanding and Question Answering,” inAdvances in Neural
Information Processing Systems (NeurIPS), 2024.
[8] S. Wang et al., “K-RagRec: Knowledge Graph Retrieval-Augmented
Generation for LLM-based Recommendation,”arXiv preprint
arXiv:2501.02226, 2025.
[9] W. Krichene and S. Rendle, “On Sampled Metrics for Item Recommen-
dation,” inProc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery
& Data Mining, 2020, pp. 1748–1757.