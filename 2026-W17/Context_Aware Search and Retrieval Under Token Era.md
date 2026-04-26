# Context-Aware Search and Retrieval Under Token Erasure

**Authors**: Sara Ghasvarianjahromi, Joshua Barr, Yauhen Yakimenka, Jörg Kliewer

**Published**: 2026-04-20 15:42:12

**PDF URL**: [https://arxiv.org/pdf/2604.18424v1](https://arxiv.org/pdf/2604.18424v1)

## Abstract
This paper introduces and analyzes a search and retrieval model for RAG-like systems under {token} erasures. We provide an information-theoretic analysis of remote document retrieval when query representations are only partially preserved. The query is represented using term-frequency-based features, and semantically adaptive redundancy is assigned according to feature importance. Retrieval is performed using TF-IDF-weighted similarity. We characterize the retrieval error probability by showing that the vector of similarity margins converges to a multivariate Gaussian distribution, yielding an explicit approximation and computable upper bounds. Numerical results support the analysis, while a separate data-driven evaluation using embedding-based retrieval on real-world data shows that the same importance-aware redundancy principles extend to modern retrieval pipelines. Overall, the results show that assigning higher redundancy to semantically important query features improves retrieval reliability.

## Full Text


<!-- PDF content starts -->

1
Context-Aware Search and Retrieval Under Token Erasure
Sara Ghasvarianjahromi, Joshua Barr, Yauhen Yakimenka, and J ¨org Kliewer
Helen and John C. Hartmann Department of Electrical and Computer Engineering
New Jersey Institute of Technology, Newark, New Jersey, 07102, USA
Email:{sg273, jb794, yauhen.yakimenka, jkliewer}@njit.edu
Abstract—This paper introduces and analyzes a search and
retrieval model for RAG-like systems under token erasures. We
provide an information-theoretic analysis of remote document
retrieval when query representations are only partially preserved.
The query is represented using term-frequency-based features,
and semantically adaptive redundancy is assigned according
to feature importance. Retrieval is performed using TF-IDF-
weighted similarity. We characterize the retrieval error proba-
bility by showing that the vector of similarity margins converges
to a multivariate Gaussian distribution, yielding an explicit
approximation and computable upper bounds. Numerical results
support the analysis, while a separate data-driven evaluation us-
ing embedding-based retrieval on real-world data shows that the
same importance-aware redundancy principles extend to modern
retrieval pipelines. Overall, the results show that assigning higher
redundancy to semantically important query features improves
retrieval reliability.
Index Terms—Information retrieval, retrieval-augmented gen-
eration, semantic communication, importance-aware encoding.
I. INTRODUCTION
Search and retrieval systems play an important role in mod-
ern information processing, with applications ranging from
web search engines to question answering and recommenda-
tion systems [1], [2]. In particular, retrieval-augmented gen-
eration (RAG) architectures use feature-based retrieval com-
ponents to select documents that provide relevant context for
downstream processing [3]–[7]. These components typically
represent queries and documents using sparse or dense feature
vectors, such as term frequency–inverse document frequency
(TF-IDF), Best Matching 25 (BM25), or learned embeddings,
and determine relevance through similarity measures including
cosine similarity orℓ 2distance [8], [9].
In this paper, we provide an information-theoretic charac-
terization of retrieval error in RAG-like systems under token
erasures. The results establish a theoretical foundation for
importance-aware redundancy allocation in retrieval systems
and clarify how semantic importance should be quantified
when retrieval reliability is the primary objective. In particular,
we analyze how random token erasures affect similarity-based
retrieval decisions and the probability of selecting an incorrect
document.
This problem is closely related to semantic communication,
which emphasizes preserving task-relevant meaning rather
than exact data fidelity [10]–[12]. Early works in this area
This work was in part supported by US NSF grants 2107370 and 2201824.
This paper was presented in part at IEEE Information Theory Workshop
(ITW), Sydney, Australia, October 2025 [21].focused on representing information in a more semantically
efficient manner [5], while more recent approaches have
leveraged deep learning and generative models to develop
end-to-end semantic processing frameworks [13], [14]. A no-
table direction in this literature is importance-aware semantic
representation guided by pre-trained language models, where
semantic relevance is used to prioritize the most informative
content [15]. A growing body of work has also investigated
token-level semantic representations, in which tokens serve as
fundamental semantic units and contextual modeling is used
to improve efficiency and robustness [16], [17].
These ideas are particularly relevant to retrieval systems
used in RAG pipelines, where query and document features
must be conveyed reliably enough to preserve retrieval ac-
curacy. However, while recent work has primarily focused
on evaluating retrieval performance in RAG-style systems
[7], [18]–[20], the retrieval component itself has rarely been
analyzed from an information-theoretic perspective. In partic-
ular, the effect of random token erasures on similarity-based
retrieval decisions and the resulting retrieval error probability
has not been systematically characterized.
In earlier work [21], we took a first step toward address-
ing this gap by introducing a simplified analytical model
for retrieval under token erasures. That work focused on
a minimal setting with two candidate documents, TF-IDF-
based query features, repetition-based redundancy, and squared
ℓ2similarity. By modeling the distortion induced by token
erasures, we showed that the original and observed similarity
margins converge jointly to a Gaussian distribution, enabling
a closed-form approximation of the retrieval error probability.
The analysis demonstrated that allocating higher redundancy
to more important query features significantly reduces the
probability of retrieving the wrong document.
This paper substantially extends that preliminary study.
Most importantly, the analysis here applies to an arbitrary
number of documents. This generalization is nontrivial, as it
requires characterizing the joint distribution of multiple simi-
larity margins and controlling the resulting high-dimensional
error events. To this end, we develop a multivariate Gaussian
approximation of the similarity-margin vector and derive com-
putable upper bounds on the retrieval error probability using
Bonferroni-type expansions and the ˇSid´ak inequality. These
results provide a principled understanding of how feature
importance, redundancy allocation, and query support size
interact in large-scale retrieval settings.
In addition to the theoretical analysis, we complement
the TF-IDF-based model with a data-driven evaluation us-
ing embedding-based retrieval on real-world data. While thearXiv:2604.18424v1  [cs.IR]  20 Apr 2026

2
analytical results are derived under a TF-IDF abstraction,
the experiments demonstrate that the same importance-aware
transmission principles extend to modern embedding-based
retrieval pipelines. This separation clarifies the scope of the
theory while illustrating the broader relevance of the proposed
design approach.
The rest of this paper is organized as follows. In Section II
we introduce the notation and terminology, the basic defini-
tions and describe the overall system model. In Sections III-A
and III-B we provide a detailed modeling and analysis of
the individual system components specified for TF-IDF-based
and embedding-based settings, respectively. Section IV de-
rives the retrieval error probability using a multivariate Gaus-
sian approximation of the score-margin vector, establishes
computable upper bounds using Bonferroni-type expansions
and the ˇSid´ak inequality, and includes a complexity com-
parison of the resulting approximations in Subsection IV-D.
Section V presents numerical results for the TF-IDF-based
model together with a separate data-driven evaluation using
an embedding-based retrieval pipeline. Finally, the paper is
concluded in Section VI.
II. PROBLEMSTATEMENT
A. Notation & Terminology
Bold lowercase letters denote vectors, and(·)⊤denotes
transpose. For a vectora, we writesupp(a)for the set of
indices of its nonzero entries, and|supp(a)|for the cardinality
of this support set. We denote by∥a∥ 2:=√
a⊤athe Euclidean
norm. Finally,◦denotes the Hadamard product.
To maintain consistent terminology across the two retrieval
settings studied in this paper, we use the termtokento
denote a basic transmitted query unit. In the TF-IDF-based
setting, this unit corresponds to an indexed feature associated
with a query term, whereas in the embedding-based setting
it corresponds to a standard tokenizer-defined text token.
Likewise, we userepresentation computationas a common
name for the block that produces the query-side and document-
side representations used for retrieval. In the TF-IDF-based
setting, this block corresponds to weighting of the recovered
query features, whereas in the embedding-based setting it
corresponds to encoder-based embedding generation from the
surviving query tokens.
B. System Model
We consider a remote document retrieval system designed
to identify the document most relevant to a given queryqcon-
sisting ofL qterms. Here, “terms” refers to basic textual units
such as words or tokens, depending on the retrieval model. The
system consists of a transmitter, a receiver, and a token erasure
process, as illustrated in Fig. 1. LetV={t 1, t2, . . . , t N}
denote the vocabulary ofNterms from which queries and
documents are formed. A queryqand each documentd iin
the corpus are formed from elements ofV.
At the transmitter, the input queryqis passed through
an importance scoring block, which determines the relative
importance of its terms. This representation may take different
forms depending on the retrieval model. For example, it maycorrespond to a sparse term-based representation in the TF-
IDF setting or to a semantic representation in the embedding-
based setting. In either case, the resulting representation is
intended to capture the retrieval-relevant information contained
in the query and to guide the subsequent allocation of trans-
mission redundancy.
The query information is then augmented with importance-
aware redundancy to improve robustness to random erasures.
Specifically, repetitions are assigned according to the impor-
tance of individual query tokens. We assume a token erasure
model with probabilityϵ, under which some transmitted query
tokens may be unavailable for retrieval. Such erasures may
arise, for example, from packet drops caused by link outages,
wireless impairments, or network congestion.
The query information available for retrieval is determined
by the subset of unerased query tokens. The quality of the
resulting query representation depends on both the redundancy
assigned to individual query tokens and the erasure pattern
affecting them. The reconstructed information is then passed
through an representation computation block, interpreted ac-
cording to the retrieval model under consideration. In the TF-
IDF case, this step combines the recovered term information
with the corresponding IDF weights to form the representation
used for retrieval, whereas in the embedding-based case it
computes the embedding associated with the successfully
received tokens. The resulting query-side representation is then
used for comparison with the document representations.
In parallel, each documentd i∈ D={d 1, d2, . . . , d n}
is converted into the representation used for retrieval by the
relevant representation computation method, either TF-IDF or
embeddings. The receiver then performs a similarity check
between the reconstructed query-side representation and each
document representations. Based on the resulting similarity or
distance scores, a decision rule is applied to select the retrieved
document, denoted byd ˆk.
Thus, the overall system consists of the following com-
mon stages: importance scoring, importance-aware redun-
dancy assignment, random token erasures affecting the query
representation, representation computation for queries and
documents, and retrieval through similarity comparison and
decision making. This formulation is intentionally model-
agnostic and serves as a common framework for the two
retrieval settings studied in this paper. In the TF-IDF case,
the importance scoring, redundancy allocation, and similarity
measure are specified using a term-based analytical model over
V. In the data-driven case, these components are instantiated
using learned embedding representations and the correspond-
ing retrieval metric.
III. ANALYSIS
In this section, we specialize the general system model in
Fig. 1 to two retrieval settings: the TF-IDF-based setting and
the embedding-based setting, and analyze the corresponding
system components in each case. In both settings, the trans-
mitter mitigates the effect of erasures by assigning semantic
repetition budget according to the importance of the query in-
formation. The key difference lies in the object being protected
and the representation used for retrieval. In the TF-IDF-based

3
qImportance
scoreAdding semantic
redundancy×e
Representation
computationSimilarity
checkDecision
ruledˆk
Representation
computationd1, d2, ..., d n
Fig. 1. Remote document retrieval system. The tokenized query is first converted into a feature representation and augmented with importance-dependent
redundancy. Random token erasures then affect this representation. Retrieval is performed by comparing the resulting query representation with document
feature representations and selecting a document according to a predefined decision rule.
setting, erasures act on sparse TF-based feature coordinates,
and retrieval is performed using a TF-IDF-weighted distance
measure. In the embedding-based setting, erasures act on
tokenizer-defined text tokens, and retrieval is performed using
encoder-generated dense embeddings together with cosine
similarity.
A. TF-IDF-Based Retrieval
We begin by characterizing the statistical structure of the
vocabulary and the query generation process, which motivates
the term-frequency representation adopted in this section.
We then describe the TF-based importance scoring and the
corresponding repetition-based redundancy assignment, where
the number of repetitions is determined by term importance.
Next, we model token erasures and their effect on the query
representation available for retrieval. We then introduce the
TF-IDF-specific representation computation used for retrieval.
Finally, we formalize the TF-IDF-weighted similarity com-
putation and the minimum-distance decision rule used for
document selection, which together form the basis for the
retrieval error analysis developed in the following sections.
1)Vocabulary and Query:As described in the system
model, we assume a predefined vocabularyVwithNdistinct
terms. For example, in a natural language processing (NLP)
setting, the vocabulary may consist of all words in the English
language. Empirical studies show that term frequencies in such
corpora follow Zipf’s law [22], a heavy-tailed distribution.
According to this law, each termt i∈ Vis assigned a
unique ranki∈ {1,2, . . . , N}based on its frequency in
the vocabulary, witht 1being the most frequent term and
tNbeing the least frequent. The frequency of a term is
inversely proportional to its rankiand satisfiesf i∝1
iα, where
α≥0is the Zipfian exponent controlling the decay rate.
This relationship characterizes the heavy-tailed distribution
observed in natural language, where a small number of terms
occur very frequently while the majority appear rarely. To
convert these frequencies into a probability distribution over
V, we introduce the normalizing constantPN
j=11/jα. The
probability of selecting a term at ranki, in accordance with
Zipf’s law, is then given by
πi=1/iα
PN
j=11/jα.(1)To model a queryq= (w 1, w2, . . . , w Lq), we assume that its
terms are drawn independently from the vocabularyVaccord-
ing to the Zipf distribution. The real queries exhibit term de-
pendencies, but the i.i.d. model provides analytical tractability
and acts as a first-order approximation. This modeling choice
is consistent with empirical observations that natural-language
queries inherit the same heavy-tailed term-frequency structure
as the underlying corpus; consequently, sampling from Zipf’s
law provides a realistic model for query term occurrences and
preserves the dominance of high-frequency words. Accord-
ingly, the vectorc q∈RNrepresents the count of each term
in the query, and the frequency with which each term appears
is modeled asc q∼Multinomial(L q,{π1, . . . , π N}), where
Lqis query length and{π 1, . . . , π N}denotes the probabilities
of the vocabulary terms determined by their ranks according
to Zipf’s law in (1).
2)Importance Score:In the TF-IDF setting, the impor-
tance of each query term is quantified through its term
frequency (TF). To obtain the relative frequencies of terms
in the query, we normalize the count vectorc qby dividing
the count of each term by the query lengthL q. This yields the
normalized vector ecq=cq
Lq, which satisfiesPN
i=1ecq,i= 1.
This normalization step produces a probability-like represen-
tation of the query content and is consistent with the standard
TF method commonly used in natural language processing and
information retrieval [23], [24].
According to this definition, the TF value associated with
termiin queryqis given by
TF(i) =cq,i
Lq,(2)
wherec q,idenotes the number of occurrences of termiin the
queryq. Thus, in the TF-IDF case, the term-frequency values
serve as the importance scores that determine the relative
contribution of the query terms and guide the subsequent
allocation of transmission resources.
3)Adaptive Repetition Under Token Erasures:To im-
prove robustness to token erasures while prioritizing the most
informative parts of the query, we first derive a modified
query representationv qfrom the normalized count vector ecq.
Specifically, we suppress a set ofl shigh-frequency but low-
informative terms, commonly referred to as stop words in

4
NLP [25], by forcing their corresponding coordinates to zero.
LetL s⊆ {1,2, . . . , N}denote the stop-word index set. To
formalize this operation, we define a coordinate-wise masking
operatorG:RN→RNacting componentwise as
G(x) i=(
0,ifi∈ L s,
xi,otherwise.
The resulting stop-word-filtered query representation is
vq=G(ecq).
This step removes terms that contribute little to retrieval dis-
crimination while still consuming resources, thereby reducing
overhead with limited impact on retrieval accuracy.
Although the original vector ecqsatisfiesPN
i=1ecq,i= 1, the
modified vectorv qgenerally satisfiesPN
i=1vq,i<1because
the stop-word coordinates are set to zero. Let
Sq≜supp(v q) ={i:v q,i̸= 0}
and letK q=|S q|denote the number of active coordinates
after stop-word removal andMis the number of remaining
token.
Rather than working with the fullN-dimensional vector,
we represent the query sparsely through the index–value pairs
(i, vq,i)for alli∈ S q. This keeps only the TF-weighted
coordinates that are relevant for retrieval. To make this sparse
representation more robust to token erasures, we assignr i
repetitions to each pair(i, v q,i), withr i= 0whenever
vq,i= 0. To control the overall repetition budget, we define
the design rate
R=MPN
i=1ri,
Since coordinates with larger TF values typically play a more
important role in retrieval, we assign repetitions proportionally
to their TF values:
ri=&
M
RPN
j=1vq,jvq,i'
.(3)
We restrict attention to queries with at least one non-stopword
term, that is,M≥1. Without integer rounding, this allocation
gives exactlyP
iri=M/R. Since⌈x⌉ ≥x, the rounded
allocation satisfiesP
iri≥M/R, so the effective rate is upper
bounded byR.
Next, we model the effect of token erasures on this repeated
sparse representation. We assume that erasures occur indepen-
dently across repetitions and across coordinates, for the sake of
analytical traceability. If the pair(i, v q,i)is repeatedr itimes,
then the probability that at least one copy remains available is
1−ϵri, whereas the probability that all copies are erased isϵri.
To formalize this, we define an indicator vectore= [e i]N
i=1
with independent components distributed as
ei=(
1,with probability1−ϵri,
0,with probabilityϵri.(4)
Here,e i= 1indicates that at least one repetition of(i, v q,i)
is retained, wherease i= 0indicates that all repetitions
associated with coordinateiare erased.Accordingly, the query representation available for retrieval
is obtained by preserving coordinateiwhene i= 1and setting
it to zero otherwise:
ˆvq,i=(
vq,i,ife i= 1,
0,ife i= 0.
Equivalently, we can write it as ˆvq=vq◦e. This expression
makes explicit that token erasures independently preserve
or remove the active coordinates ofv qaccording to the
repetition-dependent probabilities in (4).
4)Representation Computation:We next specify the rep-
resentation computation block in Fig. 1 for the TF-IDF re-
trieval setting. In this case, the block incorporates the inverse
document frequency (IDF) information associated with the
vocabulary terms. Specifically, let
IDF(i) = logn+ 1
ni+ 1
≜ζi,(5)
wheren idenotes the number of documents in the corpusD
in which termt iappears. Letζ∈RNdenote the vector of
IDF weights.
The TF-IDF representation associated with the query vector
available for retrieval is obtained by weighting each coordinate
ofˆvqby its corresponding IDF value. Equivalently, the query-
side representation used for retrieval is ˆvq◦ζ. Thus, in the
TF-IDF case, the representation computation block transforms
the TF-based query representation into a TF-IDF-weighted
representation that emphasizes rare and informative terms in
the subsequent similarity computation.
5)Similarity Check and Decision Rule:At the receiver,
the objective is to identify the documentd ˆkfrom the corpus
D={d 1, d2, . . . , d n}that is most relevant to the transmitted
queryq. We assume that all unique terms appearing in the
corpus are contained in the predefined vocabularyV, i.e.,Sn
j=1Tdj⊆ V, whereT dj={t|t∈d j}denotes the set
of terms in documentd j.
To compare the reconstructed query representation with the
corpus, the receiver computes the term-frequency (TF) vectors
for all documents, denotedv d1,vd2, . . . ,v dn. Using the IDF
weights defined in (5), the receiver then evaluates the TF-IDF-
weighted squaredℓ 2distance between the reconstructed query
vector ˆvqand each document vectorv dj:
ˆsj=X
i∈Sqζ2
i 
ˆvq,i−vdj,i2, j= 1,2, . . . , n.(6)
Only coordinates inS qare evaluated, reflecting that trans-
mission and reconstruction occur solely over these indices.
We define the retrieval score overS qto focus the analysis on
the interaction between query features and document represen-
tations, excluding coordinates where the reconstructed query
carries no information. In practice, this requires knowledge of
Sqat the receiver, which can be ensured by communicating
the query support as low-rate side information.
The document achieving the smallest value ofˆs jis selected
as the most relevant:
ˆk= arg min
j∈{1,2,...,n}ˆsj.(7)

5
Thus,d ˆkis the document whose TF-IDF representation is
closest to the reconstructed query vector under the minimum-
distance decision rule. Ties, if any, are broken using a fixed
tie-breaking rule, for example uniformly at random among tied
candidates, so that the selected index is well-defined.
B. Embedding-Based Retrieval
In modern NLP, pretrained encoder models such as sen-
tence transformers [26] map variable-length text sequences
into fixed-dimensional dense vectors, known as embeddings.
These models are trained so that semantically related texts are
mapped to nearby points in the embedding space, with cosine
similarity serving as a standard measure of semantic proximity
[27]. Unlike sparse and interpretable TF-IDF representations,
embeddings capture contextual and distributional meaning but
are generally not directly interpretable at the coordinate level.
This embedding-based view is particularly relevant for modern
retrieval pipelines, where queries and documents are repre-
sented in a dense semantic space rather than through sparse
term-based features. In this section, we therefore specialize
the general system model in Fig. 1 to the embedding-based
retrieval setting and describe the components that differ from
the TF-IDF case.
1)Importance Score:Letq= (w 1, w2, . . . , w Lq)denote
the input query, where the terms are now processed as tokens
under the tokenizer associated with the pretrained embedding
model. For each query, we first compute the full-query embed-
dingz full. To quantify the semantic importance of tokeni, we
remove that token from the query, recompute the embedding
of the shortened query, and denote the result byz \i. The
semantic loss induced by removing tokeniis then measured
using cosine similarity as [15]
score i= 1−z⊤
fullz\i
∥zfull∥2∥z\i∥2.(8)
A large value ofscore iindicates that removing tokenicauses
the query embedding to shift substantially, meaning that the
token carries significant semantic content relative to the full
query. Repeating this procedure for all query tokens yields a
semantic-importance score for each token, where larger values
ofscore iindicate greater semantic contribution to the full-
query embedding.
2)Adaptive Repetition Under Token Erasures:As in the
TF-IDF case, redundancy is introduced to improve robustness
to token erasures, but the repeated units are now query tokens
rather than TF-weighted coordinates. Given the semantic-
importance scores{score i}Lq
i=1and a total repetition budget
B=L q/R, we assign an integer repetition countr ito each
token so as to maximize the expected preserved semantic
content after erasures. Assuming independent erasures with
probabilityϵfor each repeated copy, the expected retained
contribution of tokeniis(1−ϵri) score i.
Since erasures are independent across tokens, the expected
retained semantic content decomposes as a sum over tokens,which leads to the following separable optimization problem:
max
r1,...,rLqLqX
i=1 
1−ϵri
score i
s.t.LqX
i=1ri=B,
ri∈Z≥0, i= 1, . . . , L q.(9)
This discrete optimization problem is solved using dynamic
programming because the repetition counts are integer-valued
and coupled through a total redundancy-budget constraint
[28]. The procedure computes the optimal value recursively
over tokens and remaining budget, and then backtracks to
obtain the repetition assignment. The resulting solution is
importance-aware, assigning more redundancy to tokens with
larger semantic contribution. In some cases, a token may be
assignedr i= 0, meaning that it is omitted from the repeated
representation. This typically occurs for tokens with negligible
semantic contribution, analogous to low-informative terms in
the TF-IDF case.
We next model the effect of token erasures on this repeated
token representation. We assume independent erasures across
repeated copies, with erasure probabilityϵper copy. A token
is retained if at least one of itsr icopies remains available,
and is otherwise removed. Thus, the probability that token
iis preserved is1−ϵri, whereas the probability that it is
completely erased isϵri.
In the embedding-based setting, the query representation
used for retrieval is obtained by recomputing the query em-
bedding from the surviving token sequence. We denote the
resulting post-erasure query embedding by bz. Therefore, unlike
the TF-IDF case, where erasures act directly on sparse feature
coordinates, the embedding-based pipeline forms the query
representation used for retrieval by re-embedding the surviving
query tokens.
3)Similarity Check and Decision Rule:Letz djdenote
the embedding of documentd j∈ Dobtained from the same
pretrained encoder. Retrieval is performed by comparing the
reconstructed query embedding bzwith the document embed-
dings using cosine similarity. Specifically, the score assigned
to documentd jis
ˆsj= 1−bz⊤zdj
∥bz∥2∥zdj∥2, j= 1,2, . . . , n.(10)
The document with the smallestˆs jis retrieved:
ˆk= arg min
j∈{1,2,...,n}ˆsj.(11)
To define the reference relevant document, we use the ranking
induced by the full-query embeddingz full. In other words, the
ground-truth document is taken to be the one that would be
retrieved in the absence of channel erasures.
IV. THEORETICALRESULTS
In this section, we develop theoretical results for theTF-
IDF-based retrieval frameworkintroduced in Section III-A.
Specifically, we characterize the probability of retrieval error

6
for remote document retrieval over an erasure channel, defined
as the probability that the receiver selects a document different
from the ground-truth choice obtained under perfect (no-
erasure) conditions. Our analysis proceeds in three steps. First,
conditioned on the query representationv q, we establish a
multivariate Gaussian approximation for the TF-IDF score-
margin vector induced by the erasure channel. Second, we
use this approximation to characterize the resulting error
probability through a Gaussian orthant probability. Third,
we derive computable analytic approximations and bounds
based on classical tools for Gaussian orthant probabilities,
in particular Bonferroni-type expansions [29] and the ˇSid´ak
inequality [30].
A. Preliminaries
In this subsection, we introduce the score-margin represen-
tation, its coordinatewise decomposition, and the associated
first- and second-order moments that will be used in the
probability-of-error analysis.
For each competing documentj̸=k, wherekdenotes the
ground-truth document, define the score margin
ˆ∆j≜ˆsj−ˆsk,(12)
whereˆs kdenotes the reconstructed score of the ground-
truth document, andˆs jdenotes the reconstructed score of
competitorj. Expanding the score margin using (6) and
substitutingˆv q,i=eivq,i, we obtain
ˆ∆j=X
i∈Sqh
ζ2
i
v2
dj,i−v2
dk,i
+ei2ζ2
ivq,i 
vdk,i−vdj,ii
,
(13)
where{e i}i∈Sqare independent Bernoulli random variables.
Thus, for eachj̸=k, we may rewrite ˆ∆jas distinct weighted
sums of the same underlying erasure indicators{e i}i∈Sq,
ˆ∆j=X
i∈Sq 
Cj,i+eiδj,i
,(14)
where
Cj,i=ζ2
i 
v2
dj,i−v2
dk,i
, δ j,i= 2ζ2
ivq,i 
vdk,i−vdj,i
.
Stacking them=n−1competing margins into a vector
yields
b∆= ˆ∆j
j̸=k∈Rm,(15)
which can be written as a sum of independent, though not
identically distributed, random vectors indexed by the query
coordinates
b∆=X
i∈Sq 
Ci+eiδi
,(16)
where
Ci:= (C j,i)j̸=k,δ i:= (δ j,i)j̸=k,(17)
withC idenoting the deterministic baseline contributions of
coordinateito the competing score margins, andδ ithe
corresponding transmission-dependent contributuin. Defining
Ui:=C i+eiδi, we obtain
b∆=X
i∈SqUi.(18)From (16) and the facts thatE[e i] =p iandVar(e i) =
pi(1−p i), it is easy to see that the conditional mean and
covariance matrix of b∆givenv qare
µ(vq) =E[b∆|v q] =X
i∈Sq 
Ci+piδi
,(19)
and
Σ(v q) = Cov( b∆|v q) =X
i∈Sqpi(1−p i)δiδ⊤
i.(20)
Also, define the centered random vectors
Yi:=U i−E[U i|vq] = (e i−pi)δi.(21)
Then
E[Y i|vq] =0,Cov(Y i|vq) =p i(1−p i)δiδ⊤
i.(22)
Assumption 1(Asymptotic framework).All identities above
hold for fixed vocabulary sizeN. The asymptotic statements
below are interpreted along a sequence of problem instances
indexed byt, with vocabulary sizesN t→ ∞and query
representationsv(t)
qhaving supportsS(t)
q⊆[N t]such that
K(t)
q≜|S(t)
q| → ∞. For eacht, all probabilistic statements
are understood conditionally on the realized query repre-
sentationv(t)
q, and hence on the induced redundancy levels
{r(t)
i}and corresponding survival probabilities{p(t)
i}. For
notational simplicity, we suppress the indextand writeN,S q,
andK q=|S q|, with all limits understood along this sequence.
Since b∆is a sum of independent but generally non-
identically distributed random vectors, a standard i.i.d. central
limit theorem does not apply directly. To establish asymptotic
Gaussianity, we therefore impose the following directional
Lindeberg condition on the projected summands, which en-
sures that in every fixed projectionu⊤b∆, no single coordinate
contributionu⊤Yicarries a non-negligible fraction of the total
variance [31].
Lemma 1(Directional Lindeberg condition).Under Assump-
tion 1, conditioned onv q, for every fixed unit vectoru∈Rm
and everyη >0,
1
u⊤Σ(v q)uX
i∈SqE"
 
u⊤Yi2
×1n
|u⊤Yi|> ηq
u⊤Σ(v q)uovq#
−→0,
asK q→ ∞.
(23)
Proof:The proof is presented in Appendix A
Lemma 2(Asymptotic Gaussianity of the margin vector).
Under Assumption 1 and Lemma 1, conditioned onv q,
b∆d− → N 
µ(vq),Σ(v q)
,asK q→ ∞.(24)
Proof:Condition onv q. Then the vectors{U i}i∈Sqare
independent, and
b∆−µ(v q) =X
i∈SqYi,

7
where{Y i}i∈Sqare independent, centered random vectors.
Fixu∈Rm. Consider the scalar variables
Zi:=u⊤Yi, i∈ S q.
Conditioned onv q, the variables{Z i}i∈Sqare independent
and centered, and their total variance isX
i∈SqVar(Z i|vq) =X
i∈Squ⊤Cov(Y i|vq)u=u⊤Σ(v q)u.
Lemma 1 is precisely the Lindeberg condition for the trian-
gular array{Z i}i∈Sq. Moreover, by (20),u⊤Σ(v q)u≥0.
Therefore, ifu⊤Σ(v q)u>0, the Lindeberg–Feller central
limit theorem for triangular arrays implies thatP
i∈SqZip
u⊤Σ(v q)ud− → N(0,1),asK q→ ∞.
Equivalently,
u⊤ b∆−µ(v q)d− → N 
0,u⊤Σ(v q)u
.
Ifu⊤Σ(v q)u= 0, thenP
i∈SqVar(Z i|vq) = 0, soZ i=
0almost surely for everyi∈ S q. Hence
u⊤ b∆−µ(v q)
= 0almost surely.
Thus, for every fixedu∈Rm,
u⊤ b∆−µ(v q)d− → N 
0,u⊤Σ(v q)u
.
Hence, by the Cram ´er–Wold theorem [32],
b∆d− → N 
µ(vq),Σ(v q)
.
Theorem 1(Asymptotically exact Gaussian approximation of
the conditional error probability).Let
Pe(vq) = Pr( bk̸=k|v q)
denote the conditional probability of error. Under the asymp-
totic normality of b∆established in Lemma 2, suppose that
the limiting Gaussian lawN 
µ(vq),Σ(v q)
assigns zero
probability to the boundary of the positive orthant. Then
Pe(vq)−
1−Φ m 
0;−µ(v q),Σ(v q)
→0,asK q→ ∞.
(25)
Proof:By definition of the score margins,
bk=k⇐⇒ ˆ∆j>0for allj̸=k,
which is equivalent to
b∆∈Rm
+,Rm
+:={x∈Rm:xj>0,∀j}.
Hence
Pe(vq) = 1−Pr( b∆∈Rm
+).
LetZ∼ N 
µ(vq),Σ(v q)
. By Lemma 2, b∆d− →Z.
Moreover, by assumption,Pr(Z∈∂Rm
+) = 0, soRm
+is a
continuity set forZ[32]. Therefore,
Pr(b∆∈Rm
+)→Pr(Z∈Rm
+).
SincePr(Z∈Rm
+) = Φ m 
0;−µ(v q),Σ(v q)
, it follows that
Pe(vq) = 1−Pr( b∆∈Rm
+)→1−Φ m 
0;−µ(v q),Σ(v q)
,
which is exactly (25).B. Bonferroni-Type Bounds
To obtain computable bounds on the conditional probability
of error, stated in Theorem 1, define the competitor error
events
Ej:={ˆ∆j≤0}, j= 1, . . . , m.
Since retrieval is incorrect if and only if at least one competitor
has nonpositive margin, we have
Pe(vq) = Pr
m[
j=1Ejvq
.
For any integerb≥1, define theb-fold intersection sums
Bb=X
1≤j 1<···<j b≤mPr 
Ej1∩ ··· ∩E jb|vq
.(26)
By the inclusion–exclusion principle,
Pe(vq) =mX
b=1(−1)b+1Bb.(27)
The Bonferroni inequalities [29] are obtained by truncating the
inclusion–exclusion expansion in (27). In particular, for every
positive integertsuch that2t−1≤m,
2tX
b=1(−1)b+1Bb≤P e(vq)≤2t−1X
b=1(−1)b+1Bb.(28)
Hence, truncation after an odd number of terms yields a
rigorous upper bound, while truncation after an even number
of terms yields a lower bound. In particular, choosingt= 1
andt= 2gives
Pe(vq)≤B 1, P e(vq)≤B 1−B 2+B 3.
First-order Bonferroni approximation:Under the Gaus-
sian approximation in Lemma 2, each marginal event proba-
bility is approximated by the corresponding one-dimensional
Gaussian tail. Therefore, the resulting first-order Bonferroni
approximation is the union bound given as
B(1)(vq) :=mX
j=1Φ 
−µj(vq)p
Σjj(vq)!
.(29)
Third-order Bonferroni approximation:Sharper approx-
imations are obtained by incorporating pairwise and triple
intersections. Under the Gaussian approximation of Lemma 2,
Pr(E j∩Eℓ|vq)≈Φ 2 
0;µjℓ(vq),Σjℓ(vq)
,
and
Pr(E j∩Eℓ∩Eu|vq)≈Φ 3 
0;µjℓu(vq),Σjℓu(vq)
,
whereµjℓ(vq)andΣ jℓ(vq)denote the subvector and sub-
matrix ofµ(v q)andΣ(v q)corresponding to indices(j, ℓ),
and similarlyµjℓu(vq)andΣ jℓu(vq)correspond to indices
(j, ℓ, u).

8
Accordingly, the Gaussian approximations of the second-
and third-order intersection sums are
eB2(vq) =X
1≤j<ℓ≤mΦ2 
0;µjℓ(vq),Σjℓ(vq)
,(30)
eB3(vq) =X
1≤j<ℓ<u≤mΦ3 
0;µjℓu(vq),Σjℓu(vq)
.(31)
Combining these terms withB(1)(vq)from (29) yields the
Gaussian third-order Bonferroni approximation
B(3)(vq) :=B(1)(vq)−eB2(vq) +eB3(vq).(32)
Accordingly,min{B(1)(vq),eB(3)(vq)}may be used as a
tighter computable approximation.
C.ˇSid´ak Bound
Bonferroni-type bounds provide rigorous and systematically
improvable upper bounds on the union probability associated
with competitor error events, and hence on the conditional
probability of error. However, their computational complexity
grows rapidly with the order of the expansion, since higher-
order approximations require evaluating an increasing number
of low-dimensional Gaussian orthant probabilities. When the
number of competitorsmis large, this combinatorial growth
limits their practical usefulness. As an alternative, we invoke
theˇSid´ak inequality [30], which yields a tractable lower bound
on multivariate Gaussian orthant probabilities in terms of
one-dimensional marginals. When the Gaussian margin vector
has nonnegative pairwise correlations, the ˇSid´ak inequality
applies directly. In the present model, this condition is shown
below to hold asymptotically as the support size increases,
making the resulting ˇSid´ak approximation increasingly tight
and computationally efficient. For anm-dimensional Gaussian
random vectorZ= (Z 1, . . . , Z m)⊤∼ N(µ,Σ)whose
pairwise correlation coefficients satisfy
ρjℓ=Σjℓp
ΣjjΣℓℓ≥0, j̸=ℓ,
theˇSid´ak inequality states that [30]
Pr(Z>0)≥mY
j=1Pr(Z j>0),(33)
with equality if and only if the coordinates are independent.
SincePr(Z j>0) = Φ 
µj/p
Σjj
for a univariate Gaussian
random variable, the inequality in (33) yields the explicit
product lower bound
Pr(Z>0)≥mY
j=1Φ 
µjp
Σjj!
.(34)
Applying the ˇSid´ak inequality to the Gaussian approximation
of the margin vector in Theorem 1 yields the following approx-
imation. If all pairwise correlations inΣ(v q)are nonnegative,
then (34) gives
Pr(b∆>0|v q)≳mY
j=1Φ 
µj(vq)p
Σjj(vq)!
.(35)The nonnegativity condition is essential: for Gaussian vectors,
(34) does not hold in general when some pairwise correlations
are negative. Thus, whenever the correlation condition is
satisfied, the right-hand side of (35) gives a rigorous lower
bound on the corresponding Gaussian orthant probability.
Equivalently, this leads to the following approximation for the
conditional probability of error:
Pe(vq)≲1−mY
j=1Φ 
µj(vq)p
Σjj(vq)!
.(36)
Here,≳and≲indicate approximations induced by the Gaus-
sian approximation, rather than rigorous inequalities for the
exact conditional error probability.
In the present setting, empirical evaluation ofΣ(v q)in-
dicates that the vast majority of pairwise correlations are
nonnegative, with any negative correlations observed for small
query lengths typically being weak. In such cases, the product
expression in (35) may still serve as a heuristic approximation
in practice, although it is no longer guaranteed to be a rigorous
bound when the nonnegativity condition is violated. This
expression is fully analytic, inexpensive to evaluate, and scales
linearly in the number of competitors. As shown next, the
nonnegativity condition holds asymptotically in the present
model as the number of active query coordinates grows.
To justify the nonnegativity condition in a tractable asymp-
totic setting, we introduce in the next lemma an auxiliary
random-ensemble model for the document coordinates. This
i.i.d. assumption is used only to analyze the limiting sign of
the Gaussian margin correlations inΣ(v q); it is not part of
the main fixed-corpus retrieval model, but rather provides a
probabilistic justification for the ˇSid´ak approximation in an
asymptotic regime.
Lemma 3(Asymptotic positivity of competitor correlations
under an i.i.d. model).Fix two distinct competitorsj̸=ℓ.
Assume that, for eachi∈ S q, the random variables{v dr,i:
r∈ {k, j, ℓ}}are i.i.d. copies of a real random variableF
withE[F2]<∞andVar(F)>0, and that these triples are
independent acrossi∈ S q. Assume also that the weights
ai:= 4p i(1−p i)ζ4
iv2
q,i, i∈ S q,
satisfy
c1Kq≤X
i∈Sqai≤c2Kq
for some constants0< c 1≤c2<∞and all sufficiently large
Kq, and thatsupi∈Sqai<∞. Then
ρjℓ(vq) =Σjℓ(vq)p
Σjj(vq)Σℓℓ(vq)a.s.− − − − − →
Kq→∞1
2.
In particular,ρ jℓ(vq)>0for all sufficiently largeK q, almost
surely.
Proof:The proof is presented in Appendix B.
D. Complexity
In this subsection, we compare the computational com-
plexity of the multivariate normal (MVN) approximation and
the computable approximations developed above. All methods

9
share the same preliminary moment computation based on
µ(vq)andΣ(v q)in (19) and (20). Therefore, the main
complexity differences mainly arise in the evaluation of the
resulting approximation or bound. The exact MVN approxi-
mation in Theorem 1 requires evaluation of anm-dimensional
Gaussian orthant probability, wherem=n−1is the number
of competitors. In practice, this typically involves anO(m3)
covariance factorization, together with additional numerical
cost for MVN integration or Monte Carlo evaluation. Whenm
is large, this cost becomes prohibitive, which motivates the use
of simpler analytic approximations. Among the computable
alternatives, the first-order Bonferroni approximationB(1)(vq)
and the ˇSid´ak approximation both require onlymunivariate
Gaussian CDF evaluations once the relevant moments are
available, and therefore scale linearly inm. By contrast, the
third-order Bonferroni approximationB(3)(vq)requires sum-
mation over m
3
triplets and therefore has complexityO(m3).
The ˇSid´ak expression is the least expensive overall, since
computingµ(v q)anddiag(Σ(v q))from the coordinatewise
decomposition over the active supportS qcostsO(K qm),
yielding total complexityO(K qm)while avoiding covariance
factorization and higher-dimensional Gaussian integration.
V. NUMERICALRESULTS
In this section, we evaluate the retrieval error probability of
the proposed system and assess the accuracy of the theoretical
approximations and bounds. Using synthetic data generated to
match the assumptions of the theoretical model, we compare
Monte Carlo simulations with the Gaussian approximation,
the Bonferroni bounds, and the ˇSid’ak bound. We also report
simulation results on real-world query–document corpora to
illustrate system behavior beyond the idealized setting. The
results show how query length, erasure probability, and coding
rate affect retrieval performance. They demonstrate the accu-
racy and tightness of the derived approximations and bounds
in finite-dimensional regimes under the theoretical model.
A. Datasets
This section describes the datasets used in the numerical
evaluation. We consider both a synthetic dataset designed to
exactly match the statistical assumptions of the theoretical
model and a real-world corpus, preprocessed and augmented
to enable controlled, simulation-based experimentation.
1)Synthetic dataset:We generate a synthetic dataset to
evaluate the retrieval error probability under the stochastic era-
sure model. The purpose is to construct a document corpus and
queries whose statistics are consistent with the assumptions
of the theoretical model. We consider a document collection
of sizen, where each document is represented by a sparse
bag-of-words vector over a vocabularyVof sizeN. The
document terms are sampled i.i.d. from a Zipf distribution
with exponentα= 1, capturing the heavy-tailed term statistics
commonly observed in natural language. Each document has
a fixed lengthL doc= 20000, and its term-frequency vector
is formed accordingly. In each Monte Carlo trial, a query of
lengthL qis generated independently by sampling its terms
i.i.d. from the same Zipf distribution.
Fig. 2. UMAP of generated question embeddings
2)Real dataset:To construct a realistic evaluation corpus,
we use the Natural Questions (NQ) dataset released by Google
[33], which contains real user queries paired with long-answer
passages from Wikipedia. We retain only the question–answer
pairs and discard the full articles. Each answer passage is
cleaned by HTML stripping and additional preprocessing,
including the removal of tables, special symbols, and non-
text artifacts, resulting in a text-only corpus ofnqueries and
their corresponding documents.
To expand the query set while preserving semantic consis-
tency, we apply an LLM-based augmentation procedure. For
each original question–document pair, a Llama3 (8B) model
[34] generates 29 additional questions associated with the
same answer passage. The prompt is designed to preserve
semantic consistency while avoiding near-duplicate outputs,
producing a naturally long-tailed distribution of query variants.
This yields a corpus of30queries associated with the samen
documents.
A qualitative embedding analysis using Uniform Manifold
Approximation and Projection (UMAP) [35], shown in Fig. 2,
confirms that the augmented queries remain clustered around
the original queries. This indicates minimal semantic drift and
supports the use of the augmented corpus for simulation-based
evaluation.
B. TF-IDF-Based Retrieval Results (Synthetic Data)
Figures 3 and 4 show the retrieval error probabilities ob-
tained on synthetic data forn= 10andn= 100, respectively.
As expected, the error probability increases with the erasure
probabilityϵand decreases as the query lengthL qincreases
and the rateRdecreases, corresponding to higher redundancy.
Atϵ= 1, the error probability becomes one for all query
lengths because, when retrieval is performed over the support
ofv q, the receiver selects the document closest to the zero
vector on that support, which is necessarily different from the
ground-truth document. Larger values ofL qtypically increase
the number of active query coordinatesK q, so that the score
margins aggregate a larger number of independent coordinate-
wise contributions. This leads to stronger concentration of the
margins and improved separation between the ground-truth
document and its competitors, which is consistent with the
Gaussian approximation developed in section IV.
The multivariate normal (MVN) approximation closely
tracks the simulation results across most operating regimes.

10
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=10, Lq=50, R=1
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=10, Lq=150, R=1
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=10, Lq=50, R=0.5
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=10, Lq=150, R=0.5
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
Fig. 3. Retrieval error probability versus erasure probability forn= 10documents, shown for different query lengthsL qand data ratesR= 1and1
2.
The largest discrepancies appear in the small-ϵregime, where
retrieval errors are rare, and the performance is governed by
extreme tail events together with finite-K qeffects. Among the
computable analytic bounds shown in the figures, the ˇSid´ak
bound is generally the tightest, improving on both the first-
and third-order Bonferroni bounds over the plotted parameter
ranges1.
C. Embedding-Based Retrieval Results (Data-Driven)
In addition to the TF-IDF–based simulations, we evaluate a
data-driven retrieval pipeline based on pretrained semantic em-
beddings. This experiment departs from the analytical model
in Sec. IV and is intended to assess whether semantic-aware
repetition remains effective in a realistic embedding-based
setting, rather than to validate the theoretical error bounds.
The empirical error probability is estimated over Monte Carlo
trials, each consisting of semantic-importance estimation, opti-
mized rate allocation, erasure realization, query reconstruction,
and embedding-based retrieval. This data-driven evaluation
complements the TF-IDF analysis by demonstrating the benefit
of semantic-aware repetition under token-level erasures in
modern embedding-based retrieval systems.
Fig. 5 shows the retrieval error on the Google NQ corpus as
a function of the erasure probabilityϵunder LLM-based re-
trieval. Performance is evaluated using Top-Kaccuracy, where
retrieval is considered correct if the ground-truth document
appears among the topKranked results (K= 1,3,5). As
expected, the error increases monotonically withϵand ap-
proaches one under severe erasures. Across all Top-Kmetrics,
the proposed importance-aware repetition scheme consistently
outperforms the no-encoding baseline. Notably, even under
compression (R= 2), the semantic-aware scheme achieves
1The computable analytic bounds appear empirically tightest in the small-
erasure regime, which is also the most practically relevant operating region.lower error than the uncoded baseline (R= 1), because re-
dundancy is selectively assigned to the most informative query
terms, whereas the baseline transmits each term only once with
no protection against erasures. Increasing redundancy (smaller
R) further reduces the error by increasing the probability that
critical features survive the erasure channel. Allowing larger
Klowers the absolute error because the correct document is
more likely to appear among multiple high-ranked candidates,
while the relative performance ordering remains unchanged,
indicating that semantic-aware protection improves the overall
ranking quality rather than only the top result.
VI. CONCLUSION
In conclusion, this paper establishes an information-
theoretic framework for understanding remote document re-
trieval under token erasures and shows that importance-aware
redundancy can significantly improve retrieval reliability. By
deriving a multivariate Gaussian approximation for TF-IDF
score margins and developing computable error bounds, the
paper provides both theoretical insight and practical tools
for analyzing retrieval performance. The numerical and data-
driven results further show that protecting semantically im-
portant query components leads to more robust retrieval, not
only in the analytical TF-IDF setting but also in modern
embedding-based pipelines.
APPENDIXA
PROOF OFLEMMA1
Condition onv q, fix a unit vectoru∈Rmandη >0, and
write
ai:=u⊤δi, Z i:=u⊤Yi= (e i−pi)ai, i∈ S q.
whereδ iis defined in (17). Then
s2
u:=X
i∈SqVar(Z i|vq) =u⊤Σ(v q)u.

11
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=100, Lq=50, R=1
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=100, Lq=150, R=1
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=100, Lq=50, R=0.5
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, ε0.00.20.40.60.81.0Retrieval Error
n=100, Lq=150, R=0.5
Theoretical
Simulation
Sidak bound
3-rd order Bonferroni bound
1-st order Bonferroni bound
Fig. 4. Retrieval error probability versus erasure probability forn= 100documents, shown for different query lengthsL qand data ratesR= 1and1
2.
0.0 0.2 0.4 0.6 0.8 1.0
Erasure Probability, 
0.00.20.40.60.81.0Retrieval Error
Top-1
Top-3
Top-5No Encoding (R=1)
Our Encoding (R=1)
Our Encoding (R=1/2)
Our Encoding (R=2)
Fig. 5. Google NQ ( intfloat/e5-large-v2) retrieval robustness under erasures:
error probability as a function ofϵfor baseline and proposed rate-adaptive
repetition schemes, evaluated for Top-kretrieval.
Ifs2
u= 0, thenZ i= 0almost surely for everyi∈ S q, and
the directional Lindeberg ratio is0by convention. Hence it
suffices to consider the cases2
u>0.
Letr 0≥1be a fixed finite integer, independent ofK q, and
define
Gr0:={i∈ S q:ri≤r0}, B r0:=S q\Gr0,
wherep i= 1−ϵriandϵ∈(0,1)is fixed. Assume that the
following technical regularity conditions, which are used to
verify the directional Lindeberg condition, hold asK q→ ∞.
(A1)Singleton-dominated regime on the good set:there exist
constants0< c 1≤c2<∞, independent ofK q, such
that for all sufficiently largeK q,
c1
Lq≤vq,i≤c2
Lqfor alli∈G r0,
and|G r0| → ∞.
(A1) captures a singleton-dominated sparse-query
regime: in the TF-based model, normalized term-frequency weights for single occurrences are naturally
of order1/L q, and (A1) requires a growing subsetG r0
of active coordinates to remain uniformly comparable to
this scale.
(A2)Uniform boundedness of document differences and IDF
weights:there exist constantsC d, Cζ<∞such that for
alli∈ S qand allj̸=k,
|vdk,i−vdj,i| ≤C d, ζ i≤C ζ.
This assumption is a mild boundedness condition en-
suring uniform control of the coordinates enteringδ i.
In standard TF-IDF settings, normalized document TF
coordinates are bounded and IDF weights are finite for
any finite corpus.
(A3)Directional non-degeneracy on the good set:for every
fixed unit vectoru∈Rm, there exists a constantc u>0,
independent ofK q, such that for all sufficiently large
Kq,X
i∈Gr0 
u⊤δi2≥cuX
i∈Gr0v2
q,i.
(A3) ensures that, in every fixed projection direction
u, the good-set vectorsδ iretain nontrivial projected
energy. It rules out degenerate cases in whichu⊤δi
becomes collectively too small, and is used to lower-
bound the projected variance in the Lindeberg argument.
(A4)Tailℓ 2-negligibility:
P
i∈Br0v2
q,iP
i∈Gr0v2
q,i→0.
(A4) requires theℓ 2-mass of the large-repetition tailB r0
to be negligible relative to that of the bounded-repetition

12
setG r0. Thus, the query energy is asymptotically domi-
nated by the good set, and the bad-set contribution does
not affect the Lindeberg verification.
We show that
1
s2uX
i∈SqE
Z2
i1{|Z i|> ηs u}vq
→0,asK q→ ∞,
which is exactly (23).
Sincee i∈ {0,1}andp i∈[0,1], we have|e i−pi| ≤1
almost surely, and therefore
|Zi| ≤ |a i|a.s. (37)
Moreover, for each competitorj̸=k,δ j,i= 2ζ2
ivq,i(vdk,i−
vdj,i), so by (A2),|δ j,i| ≤2C2
ζCdvq,i. Hence
∥δi∥2
2≤m(2C2
ζCd)2v2
q,i.
Since∥u∥ 2= 1, it follows that
a2
i= (u⊤δi)2≤ ∥δ i∥2
2≤K2v2
q,i, K:= 2C2
ζCd√m.(38)
We now split the Lindeberg sum over the good and bad sets:
1
s2uX
i∈SqE
Z2
i1{|Z i|> ηs u}vq
=TG+TB,
where
TG:=1
s2uX
i∈Gr0E
Z2
i1{|Z i|> ηs u}vq
,
and
TB:=1
s2uX
i∈Br0E
Z2
i1{|Z i|> ηs u}vq
.
Contribution of the good set:Fori∈G r0,
pi(1−p i)=ϵri(1−ϵri)≥c ϵ,r0, c ϵ,r0:= min
1≤r≤r 0ϵr(1−ϵr)>0.
Therefore,
s2
u=X
i∈Sqpi(1−p i)a2
i≥X
i∈Gr0pi(1−p i)a2
i≥cϵ,r0X
i∈Gr0a2
i.
Hence
max
i∈Gr0a2
i
s2u≤1
cϵ,r0max i∈Gr0a2
iP
j∈Gr0a2
j.
Using (38) and (A3),
max i∈Gr0a2
iP
j∈Gr0a2
j≤K2max i∈Gr0v2
q,i
cuP
j∈Gr0v2
q,j.
By (A1), for all sufficiently largeK q,
max
i∈Gr0v2
q,i≤c2
2
L2q,X
j∈Gr0v2
q,j≥ |G r0|c2
1
L2q.
Thus
max i∈Gr0a2
iP
j∈Gr0a2
j≤K2c2
2
cuc2
1·1
|Gr0|→0,
since|G r0| → ∞. Consequently,
max
i∈Gr0a2
i
s2u→0.Therefore, for all sufficiently largeK q,
|ai| ≤ηs ufor alli∈G r0.
Combining this with (37), we get
1{|Z i|> ηs u}= 0for alli∈G r0,
for all sufficiently largeK q. HenceT G= 0eventually.
Contribution of the bad set:UsingZ2
i1{|Z i|> ηs u} ≤
Z2
i, we obtain
TB≤1
s2uX
i∈Br0E[Z2
i|vq] =P
i∈Br0pi(1−p i)a2
i
s2u.
Sincep i(1−p i)≤1/4, (38) yields
X
i∈Br0pi(1−p i)a2
i≤1
4X
i∈Br0a2
i≤K2
4X
i∈Br0v2
q,i.
For the denominator, restricting toG r0and using (A3) gives
s2
u≥X
i∈Gr0pi(1−p i)a2
i≥cϵ,r0X
i∈Gr0a2
i≥cϵ,r0cuX
i∈Gr0v2
q,i.
Therefore,
TB≤K2
4cϵ,r0cu·P
i∈Br0v2
q,iP
i∈Gr0v2
q,i→0
by (A4).
Combining the good- and bad-set bounds, we conclude that
1
s2uX
i∈SqE
Z2
i1{|Z i|> ηs u}vq
→0,asK q→ ∞.
This verifies the directional Lindeberg condition (23).
APPENDIXB
PROOF OFLEMMA3
Fix two distinct competitorsj̸=ℓ. Recall that
Σjℓ(vq) =X
i∈Sqpi(1−p i)δj,iδℓ,i,
whereδ j,i= 2ζ2
ivq,i 
vdk,i−vdj,i
. Definea i:= 4p i(1−
pi)ζ4
iv2
q,i,Wi:= (v dk,i−vdj,i)(vdk,i−vdℓ,i), andU i:=
(vdk,i−vdj,i)2, then
Σjℓ(vq) =X
i∈SqaiWi,Σ jj(vq) =X
i∈SqaiUi.
For eachi∈ S q, the random variablesv dk,i,vdj,i, andv dℓ,i
are i.i.d. copies of a real random variableFwithE[F2]<
∞andVar(F)>0, and the triples(v dk,i, vdj,i, vdℓ,i)are
independent acrossi. LetF 1, F2, F3be i.i.d. copies ofF. Then
Wid= (F 1−F 2)(F1−F 3), U id= (F 1−F 2)2,
so
E[W i] =E[(F 1−F 2)(F1−F 3)] = Var(F),
and
E[Ui] =E[(F 1−F 2)2] = 2 Var(F).
In particular,W iandU ihave finite first moments.

13
Now letA Kq:=P
i∈Sqai. By assumption,
0< c 1Kq≤A Kq≤c2Kq,sup
i∈Sqai<∞,
for all sufficiently largeK q. Hence
max i∈Sqai
AKq≤supi∈Sqai
c1Kq− − − − − →
Kq→∞0.
Therefore, by a weighted strong law of large numbers,
P
i∈SqaiWi
AKqa.s.− − − − − →
Kq→∞E[W i] = Var(F).
Likewise,
P
i∈SqaiUi
AKqa.s.− − − − − →
Kq→∞E[Ui] = 2 Var(F).
Hence
ρjℓ(vq) =Σjℓ(vq)p
Σjj(vq)Σℓℓ(vq)a.s.− − − − − →
Kq→∞
Var(F)p
(2 Var(F))(2 Var(F))=1
2.
In particular,ρ jℓ(vq)>0for all sufficiently largeK q, almost
surely.
REFERENCES
[1] S. Buttcher, C. L. Clarke, and G. V . Cormack,Information retrieval:
Implementing and evaluating search engines. MIT Press, 2016.
[2] S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi, R. Rana,
and S. Nanayakkara, “Improving the domain adaptation of retrieval aug-
mented generation (RAG) models for open domain question answering,”
Transactions of the Association for Computational Linguistics, vol. 11,
pp. 1–17, 2023.
[3] E. Dimitrakis, K. Sgontzos, and Y . Tzitzikas, “A survey on question an-
swering systems over linked data and documents,”Journal of Intelligent
Information Systems, vol. 55, no. 2, pp. 233–259, 2020.
[4] F. Zhu, W. Lei, C. Wang, J. Zheng, S. Poria, and T.-S. Chua, “Re-
trieving and reading: A comprehensive survey on open-domain question
answering,”arXiv preprint arXiv:2101.00774, 2021.
[5] Q. Lan, D. Wen, Z. Zhang, Q. Zeng, X. Chen, P. Popovski, and
K. Huang, “What is semantic communication? A view on conveying
meaning in the era of machine intelligence,”Journal of Communications
and Information Networks, vol. 6, no. 4, pp. 336–371, 2021.
[6] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[7] A. Salemi and H. Zamani, “Evaluating retrieval quality in retrieval-
augmented generation,” inProceedings of the 47th International ACM
SIGIR Conference on Research and Development in Information Re-
trieval, 2024, pp. 2395–2400.
[8] J. Li, Y . Yuan, and Z. Zhang, “Enhancing llm factual accuracy with rag
to counter hallucinations: A case study on domain-specific queries in
private knowledge-bases,”arXiv preprint arXiv:2403.10446, 2024.
[9] S. Knollmeyer, S. Pfaff, M. U. Akmal, L. Koval, S. Asif, S. G. Mathias,
and D. Großmann, “Hybrid retrieval for retrieval augmented generation
in the german language production domain,”Journal of Advances in
Information Technology, vol. 16, no. 6, 2025.
[10] G. Shi, Y . Xiao, Y . Li, and X. Xie, “From semantic communication to
semantic-aware networking: Model, architecture, and open problems,”
IEEE Communications Magazine, vol. 59, no. 8, pp. 44–50, 2021.
[11] X. Luo, H.-H. Chen, and Q. Guo, “Semantic communications: Overview,
open issues, and future research directions,”IEEE Wireless Communi-
cations, vol. 29, no. 1, pp. 210–219, 2022.
[12] Z. Lu, R. Li, K. Lu, X. Chen, E. Hossain, Z. Zhao, and H. Zhang,
“Semantics-empowered communications: A tutorial-cum-survey,”IEEE
Communications Surveys & Tutorials, 2023.[13] W. Yang, H. Du, Z. Q. Liew, W. Y . B. Lim, Z. Xiong, D. Niyato, X. Chi,
X. Shen, and C. Miao, “Semantic communications for future internet:
Fundamentals, applications, and challenges,”IEEE Communications
Surveys & Tutorials, vol. 25, no. 1, pp. 213–250, 2022.
[14] C. Chaccour, W. Saad, M. Debbah, Z. Han, and H. V . Poor, “Less data,
more knowledge: Building next generation semantic communication
networks,”IEEE Communications Surveys & Tutorials, 2024.
[15] S. Guo, Y . Wang, S. Li, and N. Saeed, “Semantic importance-aware
communications using pre-trained language models,”IEEE Communi-
cations Letters, vol. 27, no. 9, pp. 2328–2332, 2023.
[16] L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and
D. Niyato, “Token communications: A large model-driven framework for
cross-modal context-aware semantic communications,”IEEE Wireless
Communications, vol. 32, no. 5, pp. 80–88, 2025.
[17] B. Liu, L. Qiao, Y . Wang, Z. Gao, Y . Ma, K. Ying, and T. Qin,
“Text-guided token communication for wireless image transmission,” in
2025 IEEE/CIC International Conference on Communications in China
(ICCC), 2025, pp. 1–6.
[18] J. Xian, T. Teofili, R. Pradeep, and J. Lin, “Vector search with openai
embeddings: Lucene is all you need,” inProceedings of the 17th ACM
International Conference on Web Search and Data Mining, 2024, pp.
1090–1093.
[19] J. Lin, A. H. Chen, C. Lassance, X. Ma, R. Pradeep, T. Teofili, J. Xian,
J.-H. Yang, B. Zhong, and V . Zhong, “Gosling grows up: Retrieval with
learned dense and sparse representations using anserini,” inProceedings
of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval, 2025, pp. 3223–3233.
[20] M. Wang, B. Tan, Y . Gao, H. Jin, Y . Zhang, X. Ke, X. Xu, and Y . Zhu,
“Balancing the blend: An experimental analysis of trade-offs in hybrid
search,”arXiv preprint arXiv:2508.01405, 2025.
[21] S. Ghasvarianjahromi, Y . Yakimenka, and J. Kliewer, “Context-aware
search and retrieval over erasure channels,” in2025 IEEE Information
Theory Workshop (ITW), 2025, pp. 821–826.
[22] S. T. Piantadosi, “Zipf’s word frequency law in natural language: A
critical review and future directions,”Psychonomic Bulletin & Review,
vol. 21, pp. 1112–1130, 2014.
[23] A. Mishra and S. Vishwakarma, “Analysis of TF-IDF model and
its variant for document retrieval,” in2015 international Conference
on Computational Intelligence and Communication Networks (CICN),
2015, pp. 772–776.
[24] S. Ibrihich, A. Oussous, O. Ibrihich, and M. Esghir, “A review on recent
research in information retrieval,”Procedia Computer Science, vol. 201,
pp. 777–782, 2022.
[25] C. Fox, “A stop list for general text,” inAcm Sigir Forum, vol. 24, no.
1-2. ACM New York, NY , USA, 1989, pp. 19–21.
[26] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using
siamese bert-networks,” inProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing and the 9th Inter-
national Joint Conference on Natural Language Processing (EMNLP-
IJCNLP), 2019, pp. 3982–3992.
[27] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of
word representations in vector space,”arXiv preprint arXiv:1301.3781,
2013.
[28] R. Bellman,Dynamic Programming. Princeton University Press, 1957.
[29] W. Feller,An Introduction to Probability Theory and its Applications,
Volume 2. John Wiley & Sons, 1991, vol. 2.
[30] Z. ˇSid´ak, “Rectangular confidence regions for the means of multivariate
normal distributions,”Journal of the American Statistical Association,
vol. 62, no. 318, pp. 626–633, 1967.
[31] A. W. Van der Vaart,Asymptotic statistics. Cambridge University Press,
2000, vol. 3.
[32] P. Billingsley,Convergence of Probability Measures. John Wiley &
Sons, 2013.
[33] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh,
C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Leeet al., “Natural
questions: a benchmark for question answering research,”Transactions
of the Association for Computational Linguistics, vol. 7, pp. 453–466,
2019.
[34] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman,
A. Mathur, A. Schelten, A. Yang, A. Fanet al., “The llama 3 herd of
models,”arXiv e-prints, pp. arXiv–2407, 2024.
[35] L. McInnes, J. Healy, and J. Melville, “UMAP: Uniform manifold
approximation and projection for dimension reduction,”arXiv preprint
arXiv:1802.03426, 2018.