# Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)

**Authors**: Junki Mori, Kazuya Kakizaki, Taiki Miyagawa, Jun Sakuma

**Published**: 2025-10-08 07:15:50

**PDF URL**: [http://arxiv.org/pdf/2510.06719v1](http://arxiv.org/pdf/2510.06719v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
grounding them in external knowledge. However, its application in sensitive
domains is limited by privacy risks. Existing private RAG methods typically
rely on query-time differential privacy (DP), which requires repeated noise
injection and leads to accumulated privacy loss. To address this issue, we
propose DP-SynRAG, a framework that uses LLMs to generate differentially
private synthetic RAG databases. Unlike prior methods, the synthetic text can
be reused once created, thereby avoiding repeated noise injection and
additional privacy costs. To preserve essential information for downstream RAG
tasks, DP-SynRAG extends private prediction, which instructs LLMs to generate
text that mimics subsampled database records in a DP manner. Experiments show
that DP-SynRAG achieves superior performanec to the state-of-the-art private
RAG systems while maintaining a fixed privacy budget, offering a scalable
solution for privacy-preserving RAG.

## Full Text


<!-- PDF content starts -->

Differentially Private Synthetic Text Generation
for Retrieval-Augmented Generation (RAG)
Junki Mori1, Kazuya Kakizaki1, Taiki Miyagawa1, Jun Sakuma2,3,
1NEC Corporation,2Institute of Science Tokyo,
3RIKEN Center for Advanced Intelligence Project
{junki.mori,kazuya1210,miyagawataik}@nec.com
sakuma@c.titech.ac.jp
Abstract
Retrieval-Augmented Generation (RAG) en-
hances large language models (LLMs) by
grounding them in external knowledge. How-
ever, its application in sensitive domains is lim-
ited by privacy risks. Existing private RAG
methods typically rely on query-time differen-
tial privacy (DP), which requires repeated noise
injection and leads to accumulated privacy loss.
To address this issue, we propose DP-SynRAG,
a framework that uses LLMs to generate dif-
ferentially private synthetic RAG databases.
Unlike prior methods, the synthetic text can
be reused once created, thereby avoiding re-
peated noise injection and additional privacy
costs. To preserve essential information for
downstream RAG tasks, DP-SynRAG extends
private prediction, which instructs LLMs to
generate text that mimics subsampled database
records in a DP manner. Experiments show that
DP-SynRAG achieves superior performanec to
the state-of-the-art private RAG systems while
maintaining a fixed privacy budget, offering a
scalable solution for privacy-preserving RAG.
1 Introduction
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020) has been widely used to enhance the
performance of large language models (LLMs)
(Gao et al., 2024) by leveraging external knowledge
databases. However, recent studies have identified
significant privacy risks in RAG systems when their
databases contain sensitive information (Zeng et al.,
2024; Qi et al., 2025; Jiang et al., 2025; Maio et al.,
2024; Wang et al., 2025b). For instance, extrac-
tion attacks against medical chatbots for patients
or recommender systems for customers can expose
private data to attackers. Moreover, sensitive in-
formation in the retrieved documents may also be
inadvertently revealed to benign users during nor-
mal interactions through LLM outputs (Figure 1).
To develop secure RAG systems, differential pri-
vacy (DP), a formal framework for protecting in-
User Side Provider Side
User
LLMKnowledge
Database
+User Query:                  …
Retrieved Documents:
1. John Smith, 32 years old. Reported 
symptoms: frequent wheezing and 
diﬃculty breathing. Diagnosed 
with asthma on July 5, 2022.
2. …User Query: 
I o�en have wheezing and shortness of 
breath. What illness could this be?
Retrieve
LLM’s Response:
Like John Smith, 32 , who had the same 
symptoms and was diagnosed with 
asthma, you may also have asthma.Leakage!Figure 1: A demonstration of privacy risks in RAG
databases: sensitive information contained in retrieved
documents (e.g., patient names) may be revealed to
benign users through LLM’s responses.
dividual records, has begun to be used (Grislain,
2025; Koga et al., 2025; Wang et al., 2025a). Ex-
isting approaches ensures DP by injecting noise
into the LLM’s responses to user queries, thereby
reducing the influence of any single record.
However, these methods must add noise to each
response; thus, they consume the privacy budget
proportionally to the number of queries. Conse-
quently, in typical RAG scenarios involving many
queries under a fixed privacy budget, the utility of
responses degrades substantially.
To avoid repeated noise injection, we propose
a text generation method using LLMs, called
DifferentiallyPrivateSynthetic text generation for
RAGdatabases (DP-SynRAG). Once synthetic
texts with DP guarantees are generated, they can be
reused indefinitely as the RAG database without in-
curring additional privacy budget, regardless of the
number of queries. To achieve high-quality private
text generation, we adoptprivate prediction(Hong
et al., 2024; Tang et al., 2024; Amin et al., 2024;arXiv:2510.06719v1  [cs.CR]  8 Oct 2025

Gao et al., 2025), which prompts the LLM with sub-
sampled database records and rephrasing instruc-
tions, while perturbing the aggregated output token
distributions to limit per-record information leak-
age. One limitation of these approaches is that they
capture only global properties of the entire dataset,
discarding not only sensitive details but also im-
portant knowledge needed for RAG. DP-SynRAG
mitigates this issue by clustering documents in a
DP manner based on keywords and document-level
embeddings, thereby grouping semantically similar
documents and separating distinct topics. Apply-
ing private prediction to these clusters in parallel
enables large-scale synthetic text generation that
preserves cluster-specific knowledge at low privacy
cost. Finally, to reduce the effect of low-quality
synthetic texts, we apply LLM-based self-filtering
to improve downstream RAG performance.
We validate our approach on three datasets tai-
lored to our setting. Results show that our method
outperforms existing private RAG approaches in
most cases while maintaining a fixed privacy bud-
get, demonstrating its effectiveness and scalability
for privacy-preserving RAG applications.
2 Related Works
2.1 Privacy Risks of RAG
When sensitive information is stored in the external
databases used by RAG, various privacy risks arise.
The first major threat is that adversarial prompts
can trigger extraction attacks, causing personally
identifiable information (PII) or raw database text
to be leaked through the LLM’s output to malicious
users (Zeng et al., 2024; Qi et al., 2025; Jiang et al.,
2025; Maio et al., 2024; Peng et al., 2025). Wang
et al. (2025b) has also shown that extraction may
occur even from benign queries. The second major
threat is membership inference attacks (MIAs), in
which an adversary infers whether specific target
data exist in the database. Several studies (Li et al.,
2025; Anderson et al., 2025; Naseh et al., 2025;
Liu et al., 2025) have shown that such attacks are
also effective against RAG systems.
Our paper proposes a general defense method
that counters all the above attacks by ensuring DP.
2.2 Privacy-preserving RAG
Various methods have been proposed to protect the
privacy of RAG. One early approach paraphrases
documents in the database using LLMs to remove
sensitive information (Zeng et al., 2025). However,this method ignores privacy risks during retrieval
and lacks DP guarantees. As non-DP defenses, re-
searchers have proposed training embedding mod-
els robust to adversarial queries (He et al., 2025),
detecting MIA queries and excluding the corre-
sponding documents (Choi et al., 2025), and per-
turbing data embeddings (Yao and Li, 2025).
Recent studies have also proposed directly en-
forcing DP on LLM outputs (Grislain, 2025; Koga
et al., 2025; Wang et al., 2025a). However, because
each query output is perturbed, these methods con-
sume the privacy budget per query, which increases
the risk of leakage as queries accumulate.
2.3 Differentially Private Text Generation
Our method relies on DP-guaranteed synthetic text
generation. Generated data can be used for down-
stream tasks such as training, in-context learning,
and RAG without incurring extra privacy costs.
Early approaches apply local differential privacy
(LDP) sanitization to raw text (Yue et al., 2021;
Chen et al., 2023), but this severely reduces utility
due to the strong noise to each word.
Recent work instead leverages the generative ca-
pabilities of LLMs to produce DP synthetic text.
One direction isprivate fine-tuning, where an LLM
fine-tuned on private data using DP-SGD gener-
ates synthetic text under DP guarantees (Yue et al.,
2023; Kurakin et al., 2024; Yu et al., 2024). These
methods, however, are computationally expensive
due to DP-SGD and impractical for RAG, where
the underlying knowledge database is periodically
updated, making repeated retraining infeasible.
Another direction isprivate prediction, which
applies a DP mechanism to the output token distri-
bution when an LLM paraphrases the original text.
This is typically implemented viasubsample-and-
aggregation(Nissim et al., 2007), where the private
dataset is divided into disjoint subsets, and non-
private predictions from each subset are privately
aggregated. Private prediction has been applied to
generate a small number of texts for prompt tun-
ing or in-context learning (Hong et al., 2024; Tang
et al., 2024; Gao et al., 2025), while (Amin et al.,
2024) proposes generating large-scale data without
assuming a specific downstream task. However,
these methods capture only average dataset proper-
ties, which is insufficient for RAG applications.
Accurate query answering requires synthetic
data that preserves locality, i.e., the distinctive fea-
tures of the original dataset. Current private pre-
diction methods lack mechanisms for generating

locality-aware outputs, leaving a significant gap in
their applicability to RAG. Our method fills this
gap. Although a recent study (Amin et al., 2025)
reports that clustering improves synthetic data qual-
ity, their approach assumes public cluster centers,
which is unrealistic in a private RAG setting as
considered in this work.
3 Preliminaries
3.1 Differential Privacy
LetDbe a set of possible datasets. Two datasets
DandD′are calledneighborsif one is obtained
from the other by dropping a single record.
Definition 1( (ε, δ) -DP).A randomized mechanism
Mis(ε, δ) -differentially private if any neghboring
dataset D, D′∈ D and any set Sof possible out-
puts, it holds that
Pr[M(D)∈ S]≤eεPr[M(D′)∈ S] +δ.
In this paper, we employ two representative DP
algorithms: theGaussian mechanismand theex-
ponential mechanism. The Gaussian mechanism
adds Gaussian noise N(0, σ2Id)to a real-valued
function f(D)∈Rd. The exponential mechanism
selects an output ywith probability proportional to
exp
ε·u(D,y)
2∆∞u
, where u(D, y) is the utility func-
tion and ∆∞u= supy,D,D′|u(D, y)−u(D′, y)|
denotes its sensitivity. Both mechanisms satisfy
(ε, δ)-DP (see Appendix B).
3.2 Retrieval-Augmented Generation
RAG combines information retrieval with LLMs
to improve the factual accuracy of responses. It
guides generation by conditioning the LLM on rele-
vant information retrieved from an external corpus.
Formally, let D={d i}N
i=1denote a document
corpus, and let Vbe a vocabulary space. Given
a user query q∈ V∗, RAG first uses an embed-
ding model E:V∗→Rdto encode both the
query and each document di∈D into a shared vec-
tor space. A similarity function sim(E(q),E(d i))
ranks the documents, enabling the selection of
the top- kmost relevant contexts. The retrieved
documents (di1, . . . , d ik)are concatenated with
the query to form an augmented prompt: paug=
(q, d i1, . . . , d ik). The LLM Lthen conditions its
generation on paugand repeatedly draws tokens
from the token spaceTaccording to
yn∼softmax(L(p aug, y<n)/τ),(1)where y<nis the sequence of previously generated
tokens and τ≥0 is the temperature. The function
Lmaps input tokens to a logit vector inR|T |.
3.3 Private Prediction
To generate DP synthetic data, our method relies
on the private prediction framework, which enables
LLMs to produce DP outputs. It has been applied
to inference tasks that protect training data (Flem-
ings et al., 2024) and RAG databases (Koga et al.,
2025; Grislain, 2025; Wang et al., 2025a), and DP
synthetic data generation that preserves the privacy
of original datasets (Amin et al., 2024; Hong et al.,
2024; Tang et al., 2024; Gao et al., 2025).
This framework follows a subsample-and-
aggregate paradigm: a randomly selected subset of
the private dataset is divided into disjoint partitions,
from which non-private predictions are obtained
and then privately aggregated. We leverage the in-
herent uncertainty of token sampling in LLMs to
perform private aggregation without adding explicit
noise, thereby reducing output distortion (Amin
et al., 2024). The key insight is that sampling to-
kens from the aggregation of clipped token logits
via softmax can be viewed as an exponential mech-
anism. Let Ds⊂D be a subsampled subset. For
eachdi∈D s, a prompt pi= (p, d i)is constructed
using a task-specific prompt template p. The logit
of the n-th token is computed for each pi, clipped
to the range[−c, c], and summed:
zn(Ds) =X
di∈Dsclipc(L(p i, y<n)).
Sampling a token from zn(Ds)via softmax, as
described in Eq. (1), constitutes an exponential
mechanism with sensitivity ∆∞zn=c. Thus, the
sequence y≤Tgenerated by repeatedly applying
this process satisfies DP (Amin et al., 2024).
4 Proposed Method
Existing approaches to our problem (Grislain,
2025; Koga et al., 2025) respond to user queries in
RAG systems via private prediction mechanisms.
However, when directly applied to multiple queries,
their total privacy budget grows linearly with the
number of queries. To address this, we propose
DP-SynRAG, which generates synthetic data re-
sembling the private data in advance through pri-
vate prediction while ensuring DP. Since using this
synthetic data as a knowledge corpus for later RAG
inference is post-processing, the privacy budget
remains fixed regardless of the number of queries.

Current methods for generating synthetic data
via private prediction often produce tokens that cap-
ture only the average characteristics of randomly
subsampled subsets of the original data (see Section
3.3). While this supports domain-specific data gen-
eration, it often loses the fine-grained factual details
useful as RAG context. In contrast, DP-SynRAG
first clusters the dataset based on keywords and
document-level embeddings under DP, grouping
semantically similar documents and separating dis-
tinct topics. Applying private prediction to these
subsets in parallel can generate large volumes of
high-quality synthetic texts covering diverse topics
in the original dataset at a low privacy budget.
4.1 Problem Formulation
We consider the problem of returning a privacy-
preserving response to a user query qunder the
RAG framework, where retrieval dataset D=
{di}N
i=1is private. Formally, our goal is to de-
sign a mechanism that outputs differentially private
sequence of Ttokens y≤Twith respect to D, while
satisfying the two objectives simultaneously:
1. Privacy-budget efficiency:The mechanism
should satisfy DP with a fixed privacy budget inde-
pendent of the number of queries, ensuring scala-
bility in multi-query RAG scenarios.
2. RAG-specific utility:The generated responses
should exhibit high utility in terms of fact-based
effectiveness in downstream RAG tasks rather than
generic language-model metrics (e.g., perplexity).
4.2 Overview of DP-SynRAG
Figure 2 illustrates an overview of DP-SynRAG.
The core idea is to partition the dataset into co-
herent topical subsets and apply rephrasing within
each subset with DP guarantees. Specifically it
comprises two differentially private stages: (1) soft
clustering based on keywords and document em-
beddings and (2) synthetic text generation. The full
algorithm is shown in Appndix A (Algorithm 1).
Stage 1 identifies representative keywords under
DP (Figure 2 (a)) and uses them to softly cluster
documents into multiple topic-specific subsets (Fig-
ure 2 (b)). To ensure each subset contains seman-
tically similar documents, we further refine them
with embedding-based similarity (Figure 2 (c)).
Stage 2 leverages LLMs to generate synthetic
texts for each subset in parallel (Figure 2 (d)). By
rephrasing documents in a DP manner, we create
synthetic data that retains the semantic richness
of the original corpus while protecting sensitiveinformation. A post-processing self-filtering step
further improves the utility of the synthetic dataset
for downstream RAG inference (Figure 2 (e)).
4.3 Stage 1: Soft Clustering Based on
Keywords and Document Embeddings
(a) Keywords Histogram Generation.This step
privately constructs a histogram to extract represen-
tative keywords for forming clusters. From each
document, we first extract a set of Kdistinct key-
words that best represent the document instead of
counting all words. This extraction from the vo-
cabulary space Vis performed by prompting the
LLMLto select Krepresentative keywords, and
formulated as the function ExtK
L:V∗→ {0,1}|V|,
whereP
v∈VExtK
L(di)v=K , indicating the K
keywords extracted from document di. Summing
across all documents yields a histogram h(D) =P
di∈DExtK
L(di). To release h(D) under DP, we
add Gaussian noise toh(D):
h′(D) =h(D) +N(0, σ2
hI|V|).
(b) Keywords-based Soft Clustering.From the
histogram h′(D), we select the top- Rmost fre-
quent keywords, denoted by W={w 1, . . . , w R},
where w1is the most frequent and wRthe least
frequent among W. Each keyword wrdefines a
cluster Cr, and a document is assigned to Crif and
only if it contains wr, subject to the constraint that
each document may belong to at most Lclusters.
To reduce the dominance of uninformative high-
frequency words, cluster assignment proceeds in
reverse order of frequency, from wRup to w1. For-
mally, for keywordw r(r=R, . . . ,1), we define
Cr=n
di∈Dwr∈di,X
r′>r1[di∈Cr′]< Lo
,
where 1[·]is the indicator function. In this way,
a document may belong to at most Lclusters an-
chored by relatively infrequent but representative
keywords. This controlled overlap prevents docu-
ments from being absorbed into clusters dominated
by uninformative words, thereby improving utility.
(c) Embedding-based Retrieval.By keywords-
based clustering, documents with similar topics
are grouped together. However, some documents
remain as outliers at the document-level, thereby
introducing noise in subsequent synthetic data gen-
eration. To mitigate this, we remove outliers from
each cluster in parallel based on document-level

LLM…Histogram 
Private
DatabaseKeywords 
samples(b)Keyword-based
So� Clustering document embedding
noisy mean embedding(a)Keywords Histogram Genera�on (c)Embedding-based Retrieval
Subset Stage 1: So� Clustering Based on Keywords and Document Embeddings 
tokens
…Stage 2: Synthe�c Text Genera�on
Rephrase the text:
Rephrase the text:
Rephrase the text:
template documents… … …
Subset(d) Private Predic�on (e) Self-ﬁltering
Synthe�c 
DatabaseUseful for task {task}?
YESsynthe�c text
top-
Vocab. Space noisecount
word...
LLM…
assignment order
sampleaggrega�onFigure 2: A two stage pipeline ofDP-SynRAG. Stage 1 first constructs a noisy histogram from the Kkeywords
extracted from each document (a). Each document is assigned to up to Lclusters formed by the top- Rkeywords
from the histogram (b). From these clusters, relevant subsets are retrieved using embeddings (c). Stage 2 generates
DP synthetic text by rephrasing the documents in each subset and privately aggregating the clipped output token
logits (d). Finally, the LLM filters the synthetic texts based on their usefulness for the downstream task (e).
similarity. First, we compute the mean embedding
µ(C r)of each cluster via the Gaussian mechanism:
µ(C r) =X
di∈CrE(di) +N(0, σ2
µId),
where E(di)is the normalized embedding of di.
This mean embedding1reflects the dominant char-
acteristics of the cluster. We then privately retrieve
the top- kdocuments most similar to this embed-
ding. The similarity threshold θs∈[0,1] is se-
lected using the exponential mechanism with pri-
vacy parameter εθs, as in (Grislain, 2025). The
utility function with sensitivity1is defined as
u(Cr, θs) =−X
i1[θs∈[0, s ir]]−k,(2)
where sir= sim(E(d i), µ(C r))is the similarity be-
tweenE(di)andµ(C r). Using the selected thresh-
old, we retrieve the relevant subsetS r⊆C ras
Sr={d i∈Cr|sim(E(d i), µ(C r))> θ s}.
4.4 Stage 2: Synthetic Text Generation.
(d) Private Prediction.By applying the private
prediction method from Section 3.3 in parallel
to each subset using the LLM Land a rephras-
ing prompt template p(e.g., “Rephrase the fol-
lowing text:”), we generate a total of Rsynthetic
1We omit division by cluster size to ensure µ(Cr)is de-
fined even when the cluster is empty, and to keep the noise
level consistent across clusters. A constant multiplicative
factor does not affect similarity computations.texts. Specifically, for each subsetS r, clipped and
summed logits for the n-th token are computed
aszn(Sr) =P
di∈Srclipc(L(p i, yr,<n)), where
pi= (p, d i). These logits are used to sequentially
generate a token sequence yr,≤T of length Tvia
the exponential mechanism.
For the clipping method, we follow the approach
of Grislain (2025), which emphasizes tokens with
larger logit values and thereby reduces the impact
of noise. The details are given in Appendix A.2.
(e) Self-filtering.Synthetic text generated from
small subsets is often low-quality, as tokens are
sampled from a nearly random distribution, which
increases the likelihood that useful information
for downstream RAG tasks is lost. Because such
text introduces noise, we apply self-filtering us-
ing LLMs. In methods like Self-RAG (Asai et al.,
2024), unrelated documents are removed after re-
trieval based on their query relevance; however,
this approach increases inference-time computa-
tional cost. Therefore, we instead prompt the LLM
with non-private downstream task information and
the synthetic text, then filter the text according to
whether it contains information essential for solv-
ing the downstream task prior to inference. The
filtered outputs are then used to construct the syn-
thetic RAG database. This filtering serves as a
post-processing step in synthetic text generation.

5 Privacy Analysis
Our algorithm generates Rsynthetic texts with min-
imal total privacy budget, using the overlapping par-
allel composition introduced by Smith et al. (2022)
and converted to zCDP (Appendix B). The formal
privacy guarantee is stated below. We sketch the
proof here and defer the full version to Appendix B.
Theorem 1.DP-SynRAG (Algorithm 1 in Ap-
pendix A) satisfies (ε, δ) -DP for any δ >0 and
ε=ρ+p
4ρlog(1/δ), where
ρ=K
2σ2
h+L1
8ε2
θs+1
2σ2µ+T
2c
τ2
.
Proof Overview. We adopt zero-concentrated dif-
ferential privacy (zCDP) (Bun and Steinke, 2016),
a variant of DP, as it provides tighter composition
bounds and more precise privacy accounting. Our
algorithm comprises two sequentially composed
mechanisms: (a) histogram generation ( Mhist) and
(b-d) keyword-based clustering followed by paral-
lel operations on each cluster ( Mclus). Within Mclus,
each cluster undergoes a sequence of operations:
(c) retrieval ( Mretr), (d) private prediction ( Mpred).
We exclude self-filtering from the privacy analy-
sis since it a post-processing. Our proof proceeds
in three steps. First, we show that Mhist,Mretr,
andMpredeach satisfy ρhist,ρretr, and ρpred-zCDP,
respectively, since they rely on Gaussian or expo-
nential mechanisms (or their compositions). Next,
each algorithm executed in parallel within a cluster
satisfies (ρretr+ρ pred)-zCDP by sequential com-
position, and Mclussatisfies L(ρ retr+ρ pred)-zCDP
due to overlapping parallel composition with L
overlaps. Finally, by sequentially composing Mhist
andMclus, the entire algorithm satisfies ρ-zCDP
withρ=ρ hist+L(ρ retr+ρ pred). We then convert
ρ-zCDP to (ε, δ) -DP using the conversion lemma
(Bun and Steinke, 2016).
6 Experiments
6.1 Settings
Datasets.We evaluate our method on three datasets
(details in Appendix C.2).Medical Synth(Gris-
lain, 2025) is a synthetic medical records dataset
containing 100 fictitious diseases. It includes pa-
tient queries describing symptoms and correspond-
ing doctor responses. Doctor responses to other pa-
tients serve as the private knowledge base. Perfor-
mance is measured by accuracy, defined as whetherthe LLM’s output includes the correct fictitious dis-
ease name based on retrieved diagnoses from prior
patients.Movielens-1M(Harper and Konstan,
2015) is used in a natural language form to study
privacy in RAG, as it includes user profiles as well
as movie ratings. Using GPT-5, we generate textual
descriptions of each user’s preferences from their
profiles and favorite movies, defined as their top-
10 rated movies. Private RAG documents include
each user’s profile, generated preferences, and liked
movies. The task is to recommend movies for a
query user by referring to favorites of similar users.
For simplicity, we restrict the dataset to the 30
most frequently rated movies. Accuracy is mea-
sured by whether the LLM’s output includes any
of the user’s top-10 favorites.SearchQA(Dunn
et al., 2017), a standard RAG benchmark, consists
of Jeopardy!-derived question–answer pairs with
associated search snippets. We use training ques-
tions with at least 40 supporting snippets containing
the correct answer, grouped into six bins by snippet
count (40–50 to 90–100). we randomly sample 17
questions from each bin, yielding 102 in total.
Compared Methods.We compare ourDP-
SynRAGwith four approaches, each illustrating a
different privacy–utility trade-off: (1)Non-RAG
excludes any RAG database ( ε= 0 ) and relies
solely on the LLM’s general knowledge, represent-
ing the lower bound of utility. (2) non-privateRAG
uses RAG database without privacy constraints
(ε=∞ ), representing the upper bound of utility.
(3)DP-RAG(Grislain, 2025) is a representative
private RAG approach that operates under a fixed
privacy budget, which accumulates over multiple
queries. (4)DP-Synth(Amin et al., 2024) is a
representative DP-based synthetic data generation
method that operates under a fixed privacy budget
independent of the number of queries and belongs
to the same category as DP-SynRAG.
Models.As an embedding model, we use multi-
qa-mpnet-base-dot-v1 model (109M parameters)
from the Sentence-Transformers library (Reimers
and Gurevych, 2019), designed for semantic search.
We compare three LLMs for text generation: Phi-
4-mini-instruct (3.8B) (Microsoft et al., 2025),
Gemma-2 (2B) (Team et al., 2024), Llama-3.1 (8B)
(Grattafiori et al., 2024).
Implementation Details.Unless otherwise speci-
fied, the overall privacy budget is fixed at εtotal=
10. For DP-RAG, we consider a per-query bud-
getεquery ; if the number of queries is m, then
εtotal≈mε query . We set δ= 10−3for all datasets.

Dataset Method Privacy BudgetModel
Phi-4-mini Gemma-2-2B Llama-3.1-8B
Medical SynthNon-RAGε total= 00.00 0.00 0.00 0.00 0.00 0.00
RAGε total=∞87.00 0.00 85.20 0.00 86.20 0.00
DP-Synth (Amin et al., 2024)ε total= 100.00 0.00 0.00 0.00 0.00 0.00
DP-SynRAG (Ours)ε total= 1067.26 2.22 67.06 1.68 61.26 2.33
DP-RAG (Grislain, 2025)εquery= 1059.92 0.44 67.06 0.44 48.94 0.38(εtotal≈10000)
MovielensNon-RAGε total= 022.60 0.00 34.00 0.00 43.60 0.00
RAGε total=∞67.80 0.00 54.60 0.00 70.80 0.00
DP-Synth (Amin et al., 2024)ε total= 1037.60 2.60 16.64 2.29 46.12 2.54
DP-SynRAG (Ours)ε total= 1042.56 1.97 41.08 2.19 54.12 2.51
DP-RAG (Grislain, 2025)εquery= 1034.72 0.54 40.48 0.48 56.80 0.62(εtotal≈5000)
SearchQANon-RAGε total= 065.69 0.00 70.59 0.00 88.24 0.00
RAGε total=∞92.16 0.00 94.12 0.00 95.10 0.00
DP-Synth (Amin et al., 2024)ε total= 1060.20 2.91 20.20 3.15 40.00 3.88
DP-SynRAG (Ours)ε total= 1089.61 3.22 85.10 2.13 91.18 3.25
DP-RAG (Grislain, 2025)εquery= 1085.10 1.75 83.14 1.75 84.90 1.78(εtotal≈1000)
Table 1: Performance comparison across datasets, methods, and models under fixed total privacy budgets εtotal,
except for DP-RAG, which uses per-query budget εquery . The number of queries is 1,000 for Medical Synth, 500
for Movielens, and 102 for SearchQA. We report mean and standard deviation of the accuracy (%) over 5 runs.
For RAG, the number of retrieved documents is
k= 10 for Medical Synth and SearchQA, and
k= 15 for Movielens. The inference temperature
is fixed at 0. The hyperparameters of the proposed
and baseline methods are tuned on the validation
set (see Appendix C.3 for details). Each method
is executed five times, and the average result is re-
ported. We build the public vocabulary space using
the NLTK (v3.9.1) English word corpus (Bird et al.,
2009), excluding common stopwords.
6.2 Results
Main Results.Table 1 presents the average accu-
racy across three datasets and three models. DP-
SynRAG substantially outperforms Non-RAG by
exploiting the RAG database while ensuring DP.
In particular, Medical Synth requires answers con-
taining fictitious disease names, which standalone
LLMs lacking domain knowledge cannot handle,
while DP-SynRAG achieves over 60% accuracy
across all models. As expected, compared with
DP-Synth, whose synthetic text reflects only av-
erage characteristics and is therefore unsuitable
for RAG, DP-SynRAG effectively retains critical
information (e.g., disease names) in the database.
Furthermore, even when the total privacy budget
of DP-SynRAG equals the per-query budget of DP-
RAG, DP-SynRAG demonstrates superior perfor-
mance in most cases. Note that the total budget
of DP-RAG far exceeds εtotal= 10 , as its total
budget scales with the number of queries.Accuracy vs. Number of Queries.To highlight
a key advantage of DP-SynRAG, namely its con-
stant privacy budget regardless of query count, we
compare its accuracy with DP-RAG across dif-
ferent numbers of queries under fixed total pri-
vacy budgets ( εtotal). Figure 3 presents results
for three datasets using Llama-3.1. Each line de-
picts the accuracy of each method under different
fixed εtotal values. When inference involves only
a single query, both methods achieve comparable
accuracy. However, as the number of queries in-
creases, DP-RAG’s performance steadily declines;
even at εtotal= 20 , it fails completely once the
query count reaches 20. These results demonstrate
that generating synthetic text with DP guarantees
is an effective strategy for RAG.
Privacy of RAG.We evaluate the privacy risks of
RAG and the effectiveness of DP using Medical
Synth to quantify instances in which sensitive infor-
mation is leaked from the LLM outputs. Sensitive
information is defined as the full names of patients
contained in the database records. Specifically, we
measure how often a patient’s full name appears in
the model’s responses. The evaluation uses 1,000
benign queries employed in the performance eval-
uation and 100 targeted attack queries explicitly
crafted to extract patient names. The targeted at-
tack uses the prompt template:“Who is the patient
diagnosed with {disease}?”for 100 diseases.
Table 2 reports the average number of leaks
across five experimental runs for each method. The

1 5 10 15 20
Number of Queries020406080Accuracy (%)
Llama-3.1-8B on Medical Synth
1 5 10 15 20
Number of Queries3040506070Accuracy (%)
Llama-3.1-8B on Movielens
1 5 10 15 20
Number of Queries5060708090100Accuracy (%)
Llama-3.1-8B on SearchQA
RAG (εtotal=∞)
Non-RAG (εtotal=0)DP-RAG (εtotal=20)
DP-RAG (εtotal=10)
DP-RAG (εtotal=5)
DP-RAG (εtotal=3)DP-SynRAG (εtotal=20)
DP-SynRAG (εtotal=10)
DP-SynRAG (εtotal=5)
DP-SynRAG (εtotal=3)Figure 3: Accuracy versus number of queries under various fixed total privacy budgets. Since DP-SynRAG can
reuse generated synthetic data as a RAG database without incurring additional privacy costs, its accuracy remains
constant regardless of the number of queries. In contrast, DP-RAG needs to allocate a smaller privacy budget per
query as the number of queries increases, causing its accuracy to decrease significantly.
results indicate that even benign queries can cause
RAG to inadvertently reveal patient names, demon-
strating a clear privacy risk. Moreover, targeted
attacks substantially increase the number of leaks.
While DP-RAG significantly reduces leakage, a
small number of leaks still occur under benign
queries. In contrast, our proposed method achieves
a sufficiently low probability of leakage. Even with
εtotal= 10 , the per-token privacy budget for syn-
thetic text is sufficiently small to effectively prevent
the disclosure of sensitive information.
MethodPhi-4-mini Gemma-2-2B Llama-3.1-8B
Benign Attack Benign Attack Benign Attack
RAG 4 85 12 90 22 81
DP-RAG 2.8 0 1.8 2 6 0
DP-SynRAG 0.5 1.25 0 0.25 2.4 1
Table 2: Average occurrences of patient full-name leak-
age in Medical Synth under 1,000 benign queries and
100 attack queries. We use εtotal= 10 for DP-SynRAG
andε query = 10for DP-RAG.
Ablation Study.The proposed method centers on
keyword-based clustering and private prediction,
operating independently of other components. To
assess the impact of additional features, we perform
an ablation study on three elements: document-
based retrieval, soft versus hard clustering, and
self-filtering. Table 3 reports the accuracy when
each element is disabled. The results indicate that
these components enhance performance on most
datasets. In particular, using hard clustering causes
many documents to be grouped under irrelevant
keywords, substantially degrading the quality of
the generated synthetic text. Note that self-filteringMethod Phi-4-mini Gemma-2-2B Llama-3.1-8B
Medical Synth
DP-SynRAG 67.26 67.06 61.26
w/o Retrieval 65.92 ↓1.34 61.46 ↓5.60 57.74 ↓3.52
w/o Self-filtering 66.78 ↓0.48 66.74 ↓0.32 52.20 ↓9.06
Hard clustering (L= 1) 42.52 ↓24.74 51.40 ↓15.66 29.38 ↓31.88
Movielens
DP-SynRAG 42.56 41.08 54.12
w/o Retrieval 42.28 ↓0.28 42.84 ↑1.76 53.76 ↓0.36
w/o Self-filtering 40.68 ↓1.88 38.40 ↓2.68 45.12 ↓9.00
Hard clustering (L= 1) 38.36 ↓4.20 38.32 ↓2.76 46.56 ↓7.56
SearchQA
DP-SynRAG 89.61 85.10 91.18
w/o Retrieval 89.22 ↓0.39 83.73 ↓1.37 90.98 ↓0.20
w/o Self-filtering 89.61 85.10 91.18
Hard clustering (L= 1) 76.67 ↓12.94 67.06 ↓18.04 82.94 ↓8.24
Table 3: The average accuracy when each component
of DP-SynRAG is removed: embedding-based retrieval,
self-filtering, and soft clustering. The subscript indicates
the accuracy difference between the full DP-SynRAG
and the DP-SynRAG without each component.
is not applied to SearchQA because this dataset
includes diverse question types rather than fixed
tasks.
7 Conclusion
This study introduces DP-SynRAG, a novel frame-
work for generating privacy-preserving synthetic
texts for RAG that preserves both data utility and
formal DP guarantees. By creating synthetic RAG
databases, DP-SynRAG eliminates the need for re-
peated noise injection and enables unlimited query
access within a fixed privacy budget. Experiments
on multiple datasets show that DP-SynRAG con-
sistently achieves performance better than existing
private RAG methods in most cases, demonstrating
its scalability and practical effectiveness.

Limitations
While DP-SynRAG shows strong performance and
scalability for privacy-preserving RAG, several lim-
itations remain. First, the method assumes that the
RAG database contains multiple related documents
to generate informative synthetic texts. When
such redundancy is limited, the generated texts be-
come less informative, reducing downstream per-
formance. This limitation reflects the DP constraint
that bounds each record’s influence. Moreover, be-
cause clustering relies on keyword overlap, it per-
forms poorly when documents share few common
terms. In practice, however, this issue is often mit-
igated, as documents on similar topics typically
exhibit sufficient overlap.
Second, like other DP-based text generation
methods, DP-SynRAG experiences significant util-
ity loss under an extremely tight privacy budget
(e.g., εtotal≈1) due to per-token privacy enforce-
ment. Nonetheless, as shown in the main text, the
per-token budget remains sufficiently small enough
to keep information leakage negligible, as con-
firmed empirically.
Third, our approach includes several hyperpa-
rameters that control clustering and noise cali-
bration. However, as demonstrated in the Ap-
pendix C.3, fixed default values perform consis-
tently well across different models and datasets,
minimizing the need for extensive tuning.
Finally, when the RAG database is updated, the
synthetic database must be regenerated to maintain
privacy guarantees. This regeneration, however,
does not require additional model retraining, keep-
ing overall maintenance costs modest.
References
Kareem Amin, Salman Avestimehr, Sara Babakniya,
Alex Bie, Weiwei Kong, Natalia Ponomareva, and
Umar Syed. 2025. Clustering and median ag-
gregation improve differentially private inference.
Preprint, arXiv:2506.04566.
Kareem Amin, Alex Bie, Weiwei Kong, Alexey Ku-
rakin, Natalia Ponomareva, Umar Syed, Andreas
Terzis, and Sergei Vassilvitskii. 2024. Private pre-
diction for large-scale synthetic text generation. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2024, pages 7244–7262, Miami,
Florida, USA. Association for Computational Lin-
guistics.
Maya Anderson, Guy Amit, and Abigail Goldsteen.
2025. Is my data in your retrieval database? mem-
bership inference attacks against retrieval augmentedgeneration. InProceedings of the 11th International
Conference on Information Systems Security and Pri-
vacy, page 474–485. SCITEPRESS - Science and
Technology Publications.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Steven Bird, Ewan Klein, and Edward Loper. 2009.Nat-
ural language processing with Python: analyzing text
with the natural language toolkit. " O’Reilly Media,
Inc.".
Mark Bun and Thomas Steinke. 2016. Concentrated
differential privacy: Simplifications, extensions, and
lower bounds. InTheory of Cryptography, pages 635–
658, Berlin, Heidelberg. Springer Berlin Heidelberg.
Mark Cesar and Ryan Rogers. 2021. Bounding, concen-
trating, and truncating: Unifying privacy loss com-
position for data analytics. InProceedings of the
32nd International Conference on Algorithmic Learn-
ing Theory, volume 132 ofProceedings of Machine
Learning Research, pages 421–457. PMLR.
Sai Chen, Fengran Mo, Yanhao Wang, Cen Chen, Jian-
Yun Nie, Chengyu Wang, and Jamie Cui. 2023. A
customized text sanitization mechanism with differ-
ential privacy. InFindings of the Association for
Computational Linguistics: ACL 2023, pages 5747–
5758, Toronto, Canada. Association for Computa-
tional Linguistics.
Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook
Lee, and Jinseong Park. 2025. Safeguarding privacy
of retrieval data against membership inference at-
tacks: Is this query too close to home?Preprint,
arXiv:2505.22061.
Matthew Dunn, Levent Sagun, Mike Higgins, V . Ugur
Guney, V olkan Cirik, and Kyunghyun Cho.
2017. Searchqa: A new q&a dataset augmented
with context from a search engine.Preprint,
arXiv:1704.05179.
James Flemings, Meisam Razaviyayn, and Murali An-
navaram. 2024. Differentially private next-token pre-
diction of large language models. InProceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long
Papers), pages 4390–4404, Mexico City, Mexico. As-
sociation for Computational Linguistics.
Fengyu Gao, Ruida Zhou, Tianhao Wang, Cong Shen,
and Jing Yang. 2025. Data-adaptive differentially
private prompt synthesis for in-context learning. In
International Conference on Representation Learn-
ing, volume 2025, pages 60152–60180.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,

and Haofen Wang. 2024. Retrieval-augmented gener-
ation for large language models: A survey.Preprint,
arXiv:2312.10997.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh
Goyal, Anthony Hartshorn, Aobo Yang, Archi Mi-
tra, Archie Sravankumar, Artem Korenev, Arthur
Hinsvark, and 542 others. 2024. The llama 3 herd of
models.Preprint, arXiv:2407.21783.
Nicolas Grislain. 2025. Rag with differential privacy.
In2025 IEEE Conference on Artificial Intelligence
(CAI), pages 847–852.
F. Maxwell Harper and Joseph A. Konstan. 2015. The
movielens datasets: History and context.ACM Trans.
Interact. Intell. Syst., 5(4).
Jiaming He, Cheng Liu, Guanyu Hou, Wenbo Jiang,
and Jiachen Li. 2025. Press: Defending privacy in
retrieval-augmented generation via embedding space
shifting. InICASSP 2025 - 2025 IEEE International
Conference on Acoustics, Speech and Signal Process-
ing (ICASSP), pages 1–5.
Junyuan Hong, Jiachen T. Wang, Chenhui Zhang,
Zhangheng LI, Bo Li, and Zhangyang Wang. 2024.
DP-OPT: Make large language model your privacy-
preserving prompt engineer. InThe Twelfth Interna-
tional Conference on Learning Representations.
Changyue Jiang, Xudong Pan, Geng Hong, Chenfu
Bao, Yang Chen, and Min Yang. 2025. Feedback-
guided extraction of knowledge base from
retrieval-augmented llm applications.Preprint,
arXiv:2411.14110.
Tatsuki Koga, Ruihan Wu, and Kamalika Chaud-
huri. 2025. Privacy-preserving retrieval-augmented
generation with differential privacy.Preprint,
arXiv:2412.04697.
Alexey Kurakin, Natalia Ponomareva, Umar Syed, Liam
MacDermed, and Andreas Terzis. 2024. Harnessing
large-language models to generate private synthetic
text.Preprint, arXiv:2306.01684.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang.
2025. Generating is believing: Membership infer-
ence attacks against retrieval-augmented generation.
InICASSP 2025 - 2025 IEEE International Confer-
ence on Acoustics, Speech and Signal Processing
(ICASSP), pages 1–5.Mingrui Liu, Sixiao Zhang, and Cheng Long. 2025.
Mask-based membership inference attacks for
retrieval-augmented generation. InProceedings of
the ACM on Web Conference 2025, WWW ’25, page
2894–2907, New York, NY , USA. Association for
Computing Machinery.
Christian Di Maio, Cristian Cosci, Marco Maggini,
Valentina Poggioni, and Stefano Melacci. 2024. Pi-
rates of the rag: Adaptively attacking llms to leak
knowledge bases.Preprint, arXiv:2412.18295.
Microsoft, :, Abdelrahman Abouelenin, Atabak Ash-
faq, Adam Atkinson, Hany Awadalla, Nguyen Bach,
Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav
Chaudhary, Congcong Chen, Dong Chen, Dong-
dong Chen, Junkun Chen, Weizhu Chen, Yen-Chun
Chen, Yi ling Chen, Qi Dai, and 57 others. 2025.
Phi-4-mini technical report: Compact yet powerful
multimodal language models via mixture-of-loras.
Preprint, arXiv:2503.01743.
Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh
Chaudhari, Alina Oprea, and Amir Houmansadr.
2025. Riddle me this! stealthy membership infer-
ence for retrieval-augmented generation.Preprint,
arXiv:2502.00306.
Kobbi Nissim, Sofya Raskhodnikova, and Adam Smith.
2007. Smooth sensitivity and sampling in private
data analysis. InProceedings of the Thirty-Ninth
Annual ACM Symposium on Theory of Computing,
STOC ’07, page 75–84, New York, NY , USA. Asso-
ciation for Computing Machinery.
Yuefeng Peng, Junda Wang, Hong Yu, and Amir
Houmansadr. 2025. Data extraction attacks
in retrieval-augmented generation via backdoors.
Preprint, arXiv:2411.01705.
Zhenting Qi, Hanlin Zhang, Eric P. Xing, Sham M.
Kakade, and Himabindu Lakkaraju. 2025. Follow
my instruction and spill the beans: Scalable data
extraction from retrieval-augmented generation sys-
tems. InThe Thirteenth International Conference on
Learning Representations.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing. Associa-
tion for Computational Linguistics.
Josh Smith, Hassan Jameel Asghar, Gianpaolo Gioiosa,
Sirine Mrabet, Serge Gaspers, and Paul Tyler. 2022.
Making the most of parallel composition in differen-
tial privacy. InProceedings on Privacy Enhancing
Technologies Symposium, page 253–273.
Xinyu Tang, Richard Shin, Huseyin A Inan, Andre
Manoel, Fatemehsadat Mireshghallah, Zinan Lin,
Sivakanth Gopi, Janardhan Kulkarni, and Robert Sim.
2024. Privacy-preserving in-context learning with
differentially private few-shot generation. InThe
Twelfth International Conference on Learning Repre-
sentations.

Gemma Team, Morgane Riviere, Shreya Pathak,
Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati-
raju, Léonard Hussenot, Thomas Mesnard, Bobak
Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu,
Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela
Ramos, Ravin Kumar, Charline Le Lan, Sammy
Jerome, and 179 others. 2024. Gemma 2: Improving
open language models at a practical size.Preprint,
arXiv:2408.00118.
Haoran Wang, Xiongxiao Xu, Baixiang Huang, and
Kai Shu. 2025a. Privacy-aware decoding: Mitigating
privacy leakage of large language models in retrieval-
augmented generation.Preprint, arXiv:2508.03098.
Yuhao Wang, Wenjie Qu, Yanze Jiang, Zichen Liu, Yue
Liu, Shengfang Zhai, Yinpeng Dong, and Jiaheng
Zhang. 2025b. Silent leaks: Implicit knowledge ex-
traction attack on rag systems through benign queries.
Preprint, arXiv:2505.15420.
Dixi Yao and Tian Li. 2025. Differentially private re-
trieval augmented generation with random projection.
InICLR 2025 Workshop on Building Trust in Lan-
guage Models and Applications.
Da Yu, Peter Kairouz, Sewoong Oh, and Zheng Xu.
2024. Privacy-preserving instructions for aligning
large language models. InProceedings of the 41st
International Conference on Machine Learning, vol-
ume 235 ofProceedings of Machine Learning Re-
search, pages 57480–57506. PMLR.
Xiang Yue, Minxin Du, Tianhao Wang, Yaliang Li,
Huan Sun, and Sherman S. M. Chow. 2021. Dif-
ferential privacy for text analytics via natural text
sanitization. InFindings of the Association for Com-
putational Linguistics: ACL-IJCNLP 2021, pages
3853–3866, Online. Association for Computational
Linguistics.
Xiang Yue, Huseyin Inan, Xuechen Li, Girish Kumar,
Julia McAnallen, Hoda Shajari, Huan Sun, David
Levitan, and Robert Sim. 2023. Synthetic text gener-
ation with differential privacy: A simple and practical
recipe. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1321–1342, Toronto,
Canada. Association for Computational Linguistics.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Yiding Liu,
Yue Xing, Han Xu, Jie Ren, Yi Chang, Shuaiqiang
Wang, Dawei Yin, and Jiliang Tang. 2024. The good
and the bad: Exploring privacy issues in retrieval-
augmented generation (RAG). InFindings of the As-
sociation for Computational Linguistics: ACL 2024,
pages 4505–4524, Bangkok, Thailand. Association
for Computational Linguistics.
Shenglai Zeng, Jiankun Zhang, Pengfei He, Jie Ren,
Tianqi Zheng, Hanqing Lu, Han Xu, Hui Liu, Yue
Xing, and Jiliang Tang. 2025. Mitigating the privacy
issues in retrieval-augmented generation (rag) via
pure synthetic data.Preprint, arXiv:2406.14773.A Algorithm Details
This section describes the details of our DP-
SynRAG. Algorithm 1 shows the complete pro-
cedure. In the following, we explain in detail the
algorithm components that require further clarifica-
tion.
A.1 Keywords Extraction from Documents
The keyword extraction step (line 4), which extracts
Kkeywords from each document di, is designed to
reduce the sensitivity of the Gaussian mechanism
applied during histogram generation. To achieve
this, both the extraction prompt and the document
are provided as input to the LLM. We employ the
same prompt template across all datasets for this
process. as shown below.
Keywords Extraction Prompt
Extract { K} single words from the follow-
ing document that represent key information
specific to the content.
Document: {d i}
A.2 Private Prediction
In private prediction, we prompt the LLM to
rephrase each document diin a subset (line 14).
The resulting token logits are then privately ag-
gregated within the subset (line 15) to generate
synthetic text (line 16).
Rephrasing Prompt.We use the rephrasing
prompt explicitly instructs the LLM to preserve
the important information useful for downstream
RAG tasks, as shown below. This prompt template
is applied consistently across the entire dataset.
Rephrasing Prompt
Rephrase the following document with-
out altering the important information con-
tained within it.
Document: {d i}
Clipping Method.For the clipping method, we
follow the approach of Grislain (2025), which em-
phasizes tokens with larger logit values and thereby
reduces the influence of noise. The t-th element
of the logit l(t)is clipped as follows. First, we
exponentiate l(t)with a normalization factor to
highlight large values:
lexp(t) =el(t)
max sel(s).

Algorithm 1Differentially Private Synthetic Text Generation for RAG (DP-SynRAG)
1:Input:Private databaseD={d i}N
i=1, V ocabularyV, Embedding modelE, LLML,task
2:Parameters:K, J, L, σ h, σµ, εθs, c, τ, T, θ p
3:Output:Synthetic databaseD synth ={y j}for RAG
# Stage 1: (a) Keywords Histogram Generation
4:ExtK
L(di)∈R|V|, i= 1, . . . , N ▷Extract distinctKkeywords fromd iinstructingL
5:h(D)←P
di∈DExtK
L(di)▷Histogram of keywords
6:h′(D)←h(D) +N(0, σ2
hI|V|)
# Stage 1: (b) Keywords-based Soft Clustering
7:W={w 1, . . . , w R} ←top-Rmost frequent keywords fromh′(D)▷Descending order
8:C r←
di∈Dwr∈di,P
r′>r 1[di∈Cr′]< L	
, r=R, R−1, . . . ,1▷ Define Rsoft clusters
9:for allclusterC rdo
# Stage 1: (c) Embedding-based Retrieval
10:µ(C r)←P
di∈CrE(di) +N(0, σ2
µI)▷Compute mean embeddings
11:Selectθ svia exponential mechanism with utility function defined as:
u(Cr, θs) =−P
i 1h
θs∈[0,sim(E(d i), µ(C r))]i
−k
12:S r← {d i∈Cr|sim(E(d i), µ(C r))> θ s}▷Retrieve relevant documents
# Stage 2: (d) Private Prediction
13:forn= 1toTdo
14:p i←concat(p, d i), di∈Sr ▷Prompt:p=“Rephrase the following document:”
15:z n(Sr)←P
di∈Crclipc(L(p i, yr,<n))▷Compute clipped logits and sum them
16:y r,n∼softmax(z n(Sr)/τ)▷Sample tokens
17:end for
18:Sety r= (y r,1, . . . , y r,T)
19:end for
# Stage 2: (e) Self-filtering
20:D synth←[]
21:for ally rin{y r}R
r=1do
22:p r←concat(p filter(task), y r)▷ p filter(task):task-specific filtering prompt
23:response∼ L(p r)
24:ifresponse=YESthen
25:D synth←D synth∪ {y r}
26:end if
27:end for
28:Return:D synth

Next, to reduce information loss from clipping
within the range [−c, c] , we shift the center so that
the maximum and minimum have equal magnitude:
lcent(t) =lexp(t)−max slexp(s) + min slexp(s)
2.
Finally, we rescalelcent(t)to lie within[−c, c]:
clipc(l) (t) =lcent(t) min
1,c
||lcent||∞
.
A.3 Self-filtering
Self-filtering filters synthetic texts by instructing
the LLM to determine, based on task information,
whether a synthetic text yrcontains information
relevant to the task (line 20-27). Because the filter-
ing prompt varies across tasks, the prompts used
for each dataset are listed below. Note that self-
filtering is not applied to SearchQA, as it includes
diverse questions and lacks a fixed task.
Self-filtering Prompt on Medicla Synth
Does the following document contain any
specific diagnosis names, even if they are
fictional? Answer only YES or NO.
Document: {y r}
Answer:
Self-filtering Prompt on Movielens
Does the following document contain spe-
cific movie titles released in the 20th cen-
tury? Answer only YES or NO.
Document: {y r}
Answer:
B Privacy Analysis
In this section, we present the complete proof of
Theorem 1. We first introduce zero-concentrated
differential privacy (zCDP) (Bun and Steinke,
2016), a variant of DP that offers tighter composi-
tion bounds and more accurate privacy accounting,
which we use in our analysis.
Definition 2( ρ-zCDP).A randomized mechanism
Msatisfies ρ-concentrated differential privacy ( ρ-
zCDP) if for allα >1
Dα(M(D)||M(D′))≤ρα,
where Dα(P||Q) is the R ´enyi divergence of order
αbetween distributionsPandQ.The widely used two DP algorithms defined in
Section 3.1: Gaussian mechanism and exponential
mechanism both guarantee zCDP.
Lemma 2((Bun and Steinke, 2016)).Gaussian
mechanismGM:D →Rdof the form
GM(D) =f(D) +N(0, σ2Id)
satisfiesρ-zCDP forρ=(∆2f)2
2σ2.
Lemma 3((Cesar and Rogers, 2021)).Exponential
mechanismEM:D → Yof the form
Pr[EM(D) =y]∝expε·u(D, y)
2∆∞u
satisfiesρ-zCDP forρ=1
8ε2.
Here, ∆pfdenotes the Lp-sensitivity of a function
f:D →Rd, defined as
∆pf= sup
D,D′∥f(D)−f(D′)∥p.
We note that (ε, δ) -DP and ρ-zCDP can be con-
verted into each other by the following lemma.
Lemma 4(Relationship between DP and zCDP
(Bun and Steinke, 2016)).Let M:D → Y satisfy
ρ-zCDP . Then Msatisfies (ε, δ) -DP for all δ >0
and
ε=ρ+p
4ρlog(1/δ).
Thus, to achieve a given (ε, δ) -DP guarantee, it
suffices to satisfyρ-zCDP with
ρ=p
ε+ log(1/δ)−p
log(1/δ)2
.
As a final step, we introduce two composition
theorems: the sequential composition theorem and
the overlapping parallel composition theorem. The
overlapping parallel composition is originally pro-
posed by Smith et al. (2022); in this paper, we adapt
it to the zCDP framework and provide a proof.
Lemma 5(Sequential Composition (Bun and
Steinke, 2016)).Let M:D → Y andM′:D ×
Y → Z . Suppose Msatisfies ρ-zCDP and
M′satisfies ρ′-zCDP as a function of its first ar-
gument. Define M′′:D → Z byM′′(D) =
M′(D, M(D)). Then, it holds that
Dα(M′′(D)||M′′(D′))≤D α(M(D)||M(D′))
+ sup
y∈YDα(M′(D, y)||M′(D′, y)),(3)
and therefore,M′′satisfies(ρ+ρ′)-zCDP .

Lemma 6(Overlapping Parallel Composition).Let
RandLbe positive integers. Let P(di, r)be a
proposition that depends only on di∈D andr∈
[R], and is independent of other samples in D. For
eachr∈[R], define a subsetC r⊂Das
Cr={d i∈D|P(d i, r) = True},where
RX
r=11[P(d i, r) = True]≤Lfor anyd i∈D.
LetM:D → Y be a mechanism that satisfies
ρ-zCDP . IfM′is the mechanism defined by
M′(D) = (M(C 1), . . . ,M(C R)),
thenM′satisfiesLρ-zCDP .
Proof. LetD, D′be neighboring datasets. Without
loss of generality, assume D=D′∪ {d i}. Since
the assignment of dito each subset Crdepends on
onlydiandr, it holds that Cr=C′
rforrsuch that
P(di, r) = False andCr=C′
r∪ {d i}forrsuch
thatP(d i, r) = True. We have for allα >1
Dα(M′(D)∥M′(D′))
=RX
r=1Dα(M(C r)∥M(C′
r))
=X
r:P(d i,r)=TrueDα(M(C r)∥M(C′
r))≤Lρ.
We now prove Theorem 1, establishing the pri-
vacy analysis of Algorithm 1.
Proof. Our algorithm comprises two sequentially
composed mechanisms: (a) histogram genera-
tion (Mhist) and (b-d) keyword-based clustering
followed by parallel operations on each cluster
(Mclus). Within Mclus, each cluster undergoes a
sequence of operations: (c) retrieval ( Mretr), (d)
private prediction ( Mpred). Formally, the full algo-
rithmMis defined as
M(D) =M clus(D,M hist(D)),
and, given a histogramh′,M clusis defined as
Mclus(D,h′) =
(M pred(M retr(C1)), . . . ,M pred(M retr(CR))).
We first prove Mhist,Mretr, andMpredsatisfy
ρhist,ρretr,ρpred-zCDP, respectively. Since Mhistgenerates the histogram using the Gaussian mech-
anism with sensitivity√
K, it satisfies ρhist-zCDP
withρhist=K
2σ2
hby Lemma 2. Mretris a composi-
tion of the Gaussian mechanism and the exponen-
tial mechanism, both with sensitivity 1. Therefore,
from Lemma 2 and Lemma 3, it satisfies ρretr-zCDP
withρretr=1
8ε2
θs+1
2σ2µ.Mpredis a composition of
Tapplications of the exponential mechanism, so by
Lemma 3, it satisfies ρpred=T
2 c
τ2-zCDP. Hence,
by Lemma 5,M clussatisfies(ρ ret+ρ gen)-zCDP.
For the overall algorithmM,Lemma 5 gives
Dα(M(D)||M(D′))≤D α(M hist(D)||M hist(D′))
+ sup
h′∈R|V|Dα(M clus(D, h′)||M clus(D′, h′)).
For any histogram h′, the assignment of di∈D to
a cluster Crdepends only on diandr. Thus, by
Lemma 6,
Dα(M(D)||M(D′))≤ρ hist+L(ρ retr+ρ pred)
=K
2σ2
h+L1
8ε2
θs+1
2σ2µ+T
2c
τ2
=ρ.
Therefore, the algorithm Msatisfies ρ-zCDP,
which can be converted to(ε, δ)-DP by Lemma 4.
C Details of Experiments
C.1 Computational Resources
All experiments in this study use 4 NVIDIA A100
GPUs with 40 GB memory each, running on a
Linux-based server cluster equipped with Intel
Xeon Silver 4216 CPUs and 755 GB RAM. Re-
producing the main results in Table 1 (5 runs of
5 methods across 3 datasets and 3 models) takes
approximately 48 hours.
C.2 Datasets
In this paper, We use publicly available three
datasets under their respective usage licenses: Med-
ical Synth2, Movielens3, and SearchQA4The de-
tails of them are summarized in Table 4. Each
dataset’s queries are divided into validation and
test sets: 1,000 each for Medical Synth, 500 each
for Movielens, and 102 each for SearchQA. For
Medical Synth and Movielens, patients or users
2Apache-2.0 license, https://huggingface.co/
datasets/sarus-tech/medical_dirichlet_phi3
3See the README for license details, https://
grouplens.org/datasets/movielens/
4BSD 3-Clause license, https://github.com/nyu-dl/
dl4ir-searchQA/tree/master

not included in the query sets are used as the re-
trieval database for RAG. Table 5 shows examples
of queries from each dataset and the corresponding
top-1 retrieved document from the database. The
following describes the preprocessing details for
Movielens.
Preprocessing of Movielens.As the first step in
converting MovieLens data into text, we use GPT-
5 to generate textual descriptions of each user’s
movie preferences from the user profile, the user’s
10 highest-rated films (restricted to the dataset’s 30
most frequently rated titles), and the genres of those
films available in MovieLens. Because MovieLens
does not include user names, we assign each user a
GPT-5–generated pseudonym. We use the template
below to generate these preferences.
Movie Preference Generation Prompt
The following user is one of the users in
the MovieLens dataset collected in 2000.
Describe this user’s movie preferences ac-
cording to the following conditions:
1. Describe in one sentence that fully re-
flects their profile and the characteristics of
the movies they like.
2. Do not include specific movie titles or
the user’s profile information.
3. Begin with either "He" or "She".
4. Provide only the user’s preferences.
User: {name} is a {age}-year-old {gender}
{occupation}. He/She likes {movie_1},
{movie_2}, ...
{movie_1}
Genres: {genre_1}, {genre_2}, ...
...
Using these generated preferences, we then cre-
ate database documents with the template below.
We use these documents as a RAG database to an-
swer queries consisting of each user’s profile and
generated preferences.
Database Template for Movielens
{name} is a {age}-year-old {gender} {occu-
pation}.
{generated preference}
In particular, he/she likes {movie_1},
{movie_2}, ...C.3 Hyperparameter Search
Hyperparameters of each method are tuned using
the validation queries.
Compared Methods.For DP-RAG, we mainly
adopt the parameter values reported in the original
paper (Grislain, 2025). The output token length is
set to 70 for Medical Synth and MovieLens, and to
30 for SearchQA, as its answers are shorter. The
top-p value, which controls the number of retrieved
documents, is set to 0.02 for MedicalSynth (follow-
ing the original paper) and 0.05 for MovieLens and
SearchQA. For DP-Synth, the batch size is fixed at
100 for all datasets.
Proposed Method.The hyperparameters used for
the main results of DP-SynRAG (Table 1) are sum-
marized in Table 6. The other hyperparameters
can be computed from those listed in the table.
The parameters εθs,ρhist, and ρretrare set based
on the total privacy budget. The parameters K
andTare chosen according to the average token
length per record in the dataset and kept constant
across all experiments. The parameter Rdeter-
mines the number of words extracted from the
noisy histogram. To minimize the probability of
extracting words with original zero counts, Ris
set at the position where the word frequency ap-
proximately corresponds to 3σhwhen the words
are sorted by frequency. For Medical Synth and
MovieLens, this corresponds to R= 500 , and for
SearchQA, R= 1000 . The parameters Landkare
tuned using the validation queries. To assess the
sensitivity of DP-SynRAG to its hyperparameters
K,R,L, and k, Tables 7-10 report the average ac-
curacy on the test queries as each hyperparameter
varies. The results indicate that accuracy remains
stable except when the hyperparameters are ex-
tremely small, suggesting that extensive fine-tuning
is unnecessary.
D Examples of Synthetic Texts
Table 11 presents synthetic data samples generated
by our proposed method. We include both good ex-
amples, which preserve essential information, and
bad examples, which lose key information and are
of low quality as text due to being generated from a
small subset. In the good examples, sensitive infor-
mation such as names is removed or replaced with
LLM-generated pseudonyms, thereby protecting
privacy.

Dataset # Database # Query
(Val/Test)Answer Set Task Description
Medical
Synth8,000 1,000 / 1,000 100 fictional
disease namesGiven a patient’s symptom description, re-
trieve similar doctor responses and output
the correct fictitious disease name. Accu-
racy is evaluated by whether the correct dis-
ease name is included in the output.
Movielens 4,083 500 / 500 Top-30 frequent
movie titlesRecommend movies for a querying user
based on their profile and generated prefer-
ences by retrieving similar users’ favorites.
Accuracy is evaluated by whether the out-
put includes any of the user’s top-10 favorite
movies within the top-30 frequent titles.
SearchQA 7,054 102 / 102 Factual answers Answer Jeopardy! questions by retriev-
ing search snippets. Accuracy is evaluated
based on whether the model output includes
the gold answer.
Table 4: Summary of datasets used for evaluation.
Dataset Query Sample Top-1 Retrieved Document
Medical
SynthI am Katarina Nordberg, I am dealing with
severe itching specifically around my
waistline, I also notice redness on my ears,
and I find that my skin reacts unusually to
cotton fabrics, exhibiting heightened
sensitivity. What is my disease?Patient Fernando Lund is experiencing
severe itching specifically around the
waistline, redness in his ears, and
heightened sensitivity to cotton fabrics.
Based on these symptoms, the medical
condition diagnosed is Flumplenoxis. The
recommended treatment for this condition
is the administration of Doozy Drops.
MovielensThis user is a 35-year-old male college
student. He gravitates toward timeless,
character-driven epics that blend action
with adventure, sci-fi/fantasy, war, and
crime, favoring heroic quests and moral
complexity in richly realized worlds while
also appreciating enduring family-friendly
fantasy musicals. What movie is
recommended for this user? Answer with
movies released in the 20th century.Logan Butler is a 35-year-old male
executive. He favors timeless,
character-driven epics that blend action
and adventure with rich world-building,
moral complexity, and touches of wit and
romance across sci-fi, crime drama, and
fantastical adventure. In particular, he
likes Star Wars: Episode IV - A New
Hope (1977), The Godfather (1972), and
The Princess Bride (1987).
SearchQAQuestion: The discovery of the Comstock
Lode in 1859 attracted miners &
prospectors to this stateIn 1859, two young prospectors struck
gold in the Sierra Nevada lands. Henry
Comstock discovered a vein of gold called
a lode. The Comstock Lode attracted
thousands of prospectors. Miners came
across the United States, as well as from
France, Germany, Ireland, Mexico, and
China. One of every three miners was...
Table 5: Examples of queries and their top-1 retrieved documents in RAG. The correct answers contained in the
documents are highlighted in blue.

DatasetK R L k T ε θsρhist ρretr
Medical Synth 10 500 5 80 70 0.4 0.1 0.009
Movielens 10 500 5 100 70 0.4 0.1 0.009
SearchQA 10 1000 5 100 70 0.4 0.1 0.009
Table 6: Hyperparameters of DP-SynRAG for the main results presented in Table 1. K: number of keywords
extracted from each document; R: number of clusters; L: number of overlapping documents across clusters; k:
number of retrieved documents; T: token length of synthetic texts; εθs: privacy parameter of threshold selection;
ρhist: zCDP parameter of the histogram generation step;ρ retr: zCDP parameter of the retrieval step.
DatasetK= 5K= 10K= 20
Medical Synth 67.18 2.29 67.26 2.22 67.84 1.68
Movielens 44.48 1.71 42.56 1.97 44.16 1.79
SearchQA 91.57 2.26 89.61 3.22 92.16 1.83
Table 7: Sensitivity analysis of DP-SynRAG with respect to the number of keywords extracted from each document
K. We setε total= 10. All other hyperparameters exceptKare set to the values listed in Table 6.
DatasetR= 100R= 300R= 500R= 700R= 1000
Medical Synth 52.84 3.47 67.56 1.22 67.26 2.22 66.68 2.91 66.36 2.00
Movielens 44.16 2.18 44.60 1.74 42.56 1.97 40.56 1.31 44.32 2.42
SearchQA 72.94 1.12 88.24 4.27 90.78 1.78 90.00 1.89 89.61 3.22
Table 8: Sensitivity analysis of DP-SynRAG with respect to the number of clusters R. We set εtotal= 10 . All other
hyperparameters exceptRare set to the values listed in Table 6.
DatasetL= 1L= 3L= 5L= 7L= 10
Medical Synth 42.52 4.84 62.82 2.97 67.26 2.22 67.86 1.88 66.14 2.25
Movielens 38.36 2.54 40.44 3.10 42.56 1.97 44.96 2.10 43.12 3.42
SearchQA 76.67 2.54 85.62 1.50 89.61 3.22 88.56 2.04 85.29 1.70
Table 9: Sensitivity analysis of DP-SynRAG with respect to the number of overlapping documents across clusters L.
We setε total= 10. All other hyperparameters exceptLare set to the values listed in Table 6.
Datasetk= 40k= 60k= 80k= 100k= 120
Medical Synth 57.84 3.00 65.68 1.58 67.26 2.22 68.16 2.38 67.42 1.88
Movielens 40.32 1.30 41.84 1.86 44.20 1.17 42.56 1.97 43.48 2.59
SearchQA 87.25 3.47 90.78 0.88 89.61 2.03 89.61 3.22 90.59 1.91
Table 10: Sensitivity analysis of DP-SynRAG with respect to the number of retrieved documents k. We set
εtotal= 10. All other hyperparameters exceptkare set to the values listed in Table 6.

Dataset Good Synthetic Text Bad Synthetic Text
Medical
SynthPatient K, displaying symptoms of sudden
limb weakness, uncontrolled gas release,
and a peculiar tingling sensation in the left
nasal passage, has been diagnosed with
Flibberflamia Frigibulitis. To effectively
manage this condition, a treatment plan
tailored to address the specific needs of
Flibberflamia FrigibulitisA case file lists certain anomalies L’Andre
Duche whose assumed french inspired
nickname appears incorrect was mistaken
it has another " name given in it it
indicates he suffering in issues of, haying
filds for distance. as result confusion
problems his way see think, problems his
of breath have an haze,. Following a set
list by certain of symptoms
Movielens Zachary ¨Scarlett ¨Lee is a 25-year-old male.
He has a strong affinity for dark, complex,
and thought-provoking films that often
blend elements of drama, crime, and the
supernatural. His favorite movies include
American Beauty (1999), and The Usual
Suspects (1995).18-month college female participant Val
Addabelle isn’ covered ( but information
does show) sales Associate like to paint as
it reminds me and the rest with in, äction
thrill rides she most finds appealing The
genre for suspense ¨movie that was shown
at school by,The classic in year she in was
able a bit for
SearchQAOn this initial mention of renowned
developer Mikhail Kalash transformer of
the iconic device, the first AK-47 assault
rifle was created in him Russia, The
AK-47 was first introduced to Russian
forces in 1949.Following specific guidelines a toy
sweetening alternative, referred to as
conjunctions - artificially and naturally
used to reduce the meaning without an
interruption of the overall message -
compounds are used in sugarcoaster and
they produce the same result when two or
no items - (a) are compared to (x)=
another item; 2. Two substances with the
combination
Table 11: Examples of synthetic texts generated by DP-SynRAG for three datasets. Good examples (green) preserve
essential information for downstream RAG tasks. Bad examples (red) lose key information and are of low quality as
text.