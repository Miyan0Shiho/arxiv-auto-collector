# BiRD: A Bidirectional Ranking Defense Mechanism for Retrieval Augmented Generation

**Authors**: Chengcai Gao, Zhihong Sun, Xiaochuan Shi, Qiufeng Wang, Chao Liang

**Published**: 2026-05-19 17:11:00

**PDF URL**: [https://arxiv.org/pdf/2605.20123v1](https://arxiv.org/pdf/2605.20123v1)

## Abstract
The growing adoption of Retrieval-Augmented Generation (RAG) has led to a rise in adversarial attacks. Existing defenses, relying on semantic analysis or voting, face a trade-off between high computational cost and limited robustness under strong poisoning attacks. Their fundamental limitation is the exclusive focus on semantic content relevance, while neglecting the retrieval context that is critically defined by ranking structures. To this end, we investigate the bidirectional ranking behavior of poisoned and benign documents, and discover a key discriminative pattern: poisoned documents exhibit significantly stronger alignment between their backward rankings and the query's forward ranking. Capitalizing on this, we propose BiRD, a bidirectional ranking defense mechanism built upon a dual-signal framework that leverages forward ranking to assess semantic content relevance and backward ranking to quantify ranking context consistency. This design directly addresses the fundamental limitation of prior approaches, enabling simultaneous efficiency and robustness. Extensive evaluation across 3 datasets with 3 retrievers and 3 LLMs under 2 attack scenarios validates BiRD's effectiveness. Notably, BiRD reduces the attack success rate of PoisonedRAG by up to 54% while simultaneously improving task accuracy by up to 56%, with average additional latency under 1 second.

## Full Text


<!-- PDF content starts -->

BiRD: A Bidirectional Ranking Defense Mechanism for
Retrieval Augmented Generation
Chengcai Gao1, Zhihong Sun2, Xiaochuan Shi1, Qiufeng Wang3, Chao Liang1
1 Wuhan University, 2 Naval University of Engineering, 3 Xi’an Jiaotong-Liverpool University
1{2023302051075, shixiaochuan, cliang}@whu.edu.cn, 2 zhihong.sun@whu.edu.cn, 3 Qiufeng.Wang@xjtlu.edu.cn
Abstract
The growing adoption of Retrieval-Augmented Generation
(RAG) has led to a rise in adversarial attacks. Existing de-
fenses, relying on semantic analysis or voting, face a trade-off
between high computational cost and limited robustness under
strong poisoning attacks. Their fundamental limitation is the
exclusive focus on semantic content relevance, while neglect-
ing the retrieval context that is critically defined by ranking
structures. To this end, we investigate the bidirectional rank-
ing behavior of poisoned and benign documents, and discover
a key discriminative pattern: poisoned documents exhibit
significantly stronger alignment between their backward rank-
ings and the query’s forward ranking. Capitalizing on this, we
propose BiRD, a bidirectional ranking defense mechanism
built upon a dual-signal framework that leverages forward
ranking to assess semantic content relevance and backward
ranking to quantify ranking context consistency. This de-
sign directly addresses the fundamental limitation of prior
approaches, enabling simultaneous efficiency and robustness.
Extensive evaluation across 3 datasets with 3 retrievers and
3 LLMs under 2 attack scenarios validates BiRD’s effective-
ness. Notably, BiRD reduces the attack success rate of Poi-
sonedRAG by up to 54% while simultaneously improving
task accuracy by up to 56%, with average additional latency
under 1 second.
1 Introduction
While Large Language Models (LLMs) have significantly
advanced natural language processing, they struggle with gen-
erating hallucinations [1] and accessing real-time knowledge
[2]. To address these limitations, Retrieval-Augmented Gener-
ation (RAG) emerges as a pivotal preliminary stage,enhancing
the accuracy and reliability of generated content by retriev-
ing relevant information from external knowledge bases as
context [3]. However, RAG systems introduce new security
vulnerabilities, most notably corpus poisoning attacks [4 –7].
Attackers inject a few meticulously crafted malicious docu-
ments into the corpus to manipulate retrieval results, thereby
User query ( q)
𝑑3𝑞
𝑑4𝑞
𝑑5𝑞
𝑑31
𝑑41
𝑑51
𝑑42
𝑑52𝑑22
𝑑33
𝑑43
𝑑53𝑑23
𝑑34
𝑑44
𝑑54𝑑14
𝑑35
𝑑45
𝑑55𝑑15
𝑑25
Forward ranking Backward rankings
𝑑1𝑞
𝑑2𝑞𝑑11 𝑑21
𝑑12𝑑32
𝑑13
𝑑24𝑑𝑖𝑞: the i-th doc. in the forward ranking of query q 
𝑑𝑗𝑖: the j-th doc. in the backward ranking of query 𝑑𝑖𝑞 
Figure 1: Bidirectional Ranking Defense (BiRD) framework.
inducing downstream LLMs to generate misleading [8], bi-
ased [9], or even harmful content [4]. Therefore, designing
robust defense mechanisms to secure the retrieval stage in
RAG systems has become a crucial and urgent priority.
Existing defenses against corpus poisoning in RAG sys-
tems typically adopt one of three strategies. First, clustering-
filtering methods [10, 11] identify anomalies by analyzing
semantic content, such as entity-intent relationships, but this
detailed parsing introduces substantial computational over-
head. Building on semantic analysis yet aiming for simpler de-
tection, voting-consistency approaches [12, 13] rely on cross-
verifying outputs across multiple paths. However, their depen-
dence on consensus leaves them vulnerable to high-intensity
poisoning attacks that can manipulate the voting outcome.
Alternatively, adversarial defense techniques [14 –16] lever-
1arXiv:2605.20123v1  [cs.CR]  19 May 2026

age the internal knowledge of LLMs to filter semantically
inconsistent documents. This strategy, however, is inherently
limited by the LLM’s own knowledge boundaries and sus-
ceptibility to deception. Despite these different designs, a
fundamental and common shortcoming persists: these meth-
ods focus predominantly on analyzing the semantic content
within documents, while critically overlooking the retrieval
context, a dimension defined by the ranking structures that
govern document order during retrieval (Figure 1).
To demonstrate its importance, we statistically analyzed
the distribution of poisoned documents, specifically their oc-
currence probability at each ranking position, in bidirectional
retrieval experiments conducted across 3 datasets, 3 retriev-
ers, and 2 attack methods (see Section 3.1 for details). The
results, shown in Figure 2, reveal two systematic patterns.
First, the left column illustrates how frequently poisoned doc-
uments occupy each position in forward retrieval. Second,
the right matrix depicts, for each document position treated
as a new query in backward retrieval, how likely poisoned
documents are to appear at various ranks in the resulting
backward list. This observed strong alignment, where poi-
soned documents not only achieve high forward ranks but also
maintain consistent backward-forward ranking relationships,
provides a novel, ranking-based signal for defense. This find-
ing motivates the design of a method that jointly evaluates
intra-document content relevance and inter-document context
consistency.
Building on this insight, we propose BiRD, a bidirectional
ranking defense mechanism for RAG. BiRD operates through
three stages: (1) the forward retrieval stage assesses semantic
content relevance; (2) the backward retrieval stage quantifies
ranking context consistency; and (3) the threat filtering stage
synthesizes both signals to filter poisoned documents from
the top ranked results. By directly leveraging the ranking
structures overlooked by prior defenses, BiRD avoids both
their high computational costs and vulnerability to strong
poisoning attacks, offering an efficient and robust defense for
RAG.
The main contributions of this paper can be summarized as
follows:
•We discover, for the first time, a key discriminative pat-
tern in bidirectional retrieval:poisoned documents ex-
hibit significantly stronger alignment between their back-
ward rankings and the query’s forward ranking com-
pared to benign documents.
•We propose BiRD, a novel defense mechanism that in-
troduces and utilizes context consistency, which is mea-
sured through bidirectional ranking alignment, in con-
junction with content relevance to protect RAG systems
against diverse poisoning attacks.
•We conduct extensive experiments across 3 diverse
datasets, utilizing 3 different retrievers and 3 LLMs,
Figure 2: Rank Position Poison Frequency Statistical
Heatmaps.
and under 2 distinct RAG attack scenarios, to compre-
hensively validate both the effectiveness and robustness
of BiRD.
2 Related Work
This section surveys the related work in RAG security re-
search. We begin by introducing the foundational framework
of RAG in Section 2.1, followed by a review of the relevant
work on RAG attacks in Section 2.2 and RAG defenses in
Section 2.3, respectively. Finally, we discuss the broad appli-
cations of bidirectional search in Section 2.4.
2.1 RAG System
The standard RAG workflow primarily consists of two stages:
retrieval and generation. During the retrieval stage, the system
filters a set of top-ranked documents from an external corpus
based on semantic similarity (e.g., cosine or dot product)
between the user’s query and documents. In the generation
stage, the large language model synthesizes a final answer by
using both the original query and the retrieved document set
as context [2, 17].
By leveraging external knowledge, RAG enhances the ac-
curacy and breadth of generated content [18, 19]. Its key
advantage is the direct use of up-to-date or specialized infor-
mation absent from training data, which helps reduce LLM
hallucinations. The system also offers scalability and flexi-
bility [20], as knowledge can be updated by refreshing the
retrieval corpus without retraining the generative model.
However, beyond its performance benefits, RAG also intro-
duces new attack surfaces. The vulnerability of RAG mainly
manifests in the fragility of the retriever during the retrieval
phase. When a query is maliciously tampered with or the
2

corpus is poisoned, the document ranking produced by the
retriever may include irrelevant or malicious documents [14].
2.2 RAG Attack
The core objective of RAG attacks is to manipulate the sys-
tem’s input or knowledge base in order to mislead its output.
At the input level, attackers may attempt to embed impercep-
tible instructions within user queries or retrieved content to
hijack LLM behavior [21, 22]. In practice, however, directly
accessing or manipulating user queries is often difficult.
Considering the static nature of the knowledge base and
the dynamic nature of queries in RAG systems, it is easier for
attackers to perform corpus poisoning attacks. Among these,
corpus poisoning is the most prevalent and poses a particu-
larly realistic threat to RAG security. Attackers represented
by PoisonedRAG [4] directly inject maliciously crafted doc-
uments into the knowledge base, either through adversarial
optimization to boost the retrieval ranking of irrelevant or
harmful documents, or by embedding specific biases [9] or
stances [8] within documents to systematically distort model
outputs. GASLITE [6] proposes white-box attacks on RAG
to evaluate its vulnerabilities under worst-case scenarios. Re-
GENT [7] is a reinforcement learning-based framework that
monitors the interaction dynamics between the attacker and
the target RAG system, continuously refining poisoned docu-
ments through relevance, generation, and naturalness rewards.
Given the prevalence of corpus poisoning attacks, this paper
will primarily evaluate the effectiveness of existing defense
measures against corpus poisoning attacks.
2.3 RAG Defense
Current defense research primarily develops along three tra-
jectories: some approaches isolate malicious documents
through clustering and filtering techniques. For instance, Se-
ConRAG [10] relies on an external LLMs to construct and
compare semantic graphs, while TrustRAG [11] utilizes an
internal LLM for conflict resolution. Both methods signifi-
cantly increase system overhead—for example, SeConRAG’s
runtime is three to four times that of the original RAG.
Another category of methods is based on the voting-
consistency assumption, which posits a semantic consen-
sus among benign documents. ReliabilityRAG [12] defends
against attacks by constructing a contradiction graph and
transforming it into a maximum independent set problem.
However, as this problem is NP-hard [23], processing large
scale retrieval sets requires discarding a substantial number
of documents to reduce computational costs, leading to sig-
nificant information loss. Similarly, RobustRAG [13] extracts
document keywords as context, which also results in infor-
mation loss. Such methods tend to fail when the density of
malicious documents increases.The third category leverages adversarial knowledge, uti-
lizing the internal knowledge and inherent security mech-
anisms of LLMs. For example, AstuteRAG [15] dynami-
cally integrates internal knowledge with retrieved content
through heuristic selection, while InstructRAG [16] guides
the retrieval process with self-generated rationales. Both ap-
proaches depend on the intrinsic knowledge and safety mech-
anisms of LLMs. Other studies attempt to obtain more robust
representations through metric learning [14]; however, the
scarcity of high-quality annotated data in dynamic settings
limits their practical applicability.
Overall, existing defense methods focus primarily on se-
mantic content while overlooking the importance of ranking
information. Since ranking directly quantifies document rel-
evance and context [24], it inherently contains rich security
signals. Therefore, delving into the patterns within ranking
behavior is crucial for constructing a robust RAG defense
framework.
2.4 Bidirectional Search
Bidirectional retrieval is primarily manifested in the interac-
tive matching between the initial input and output. In the
forward interaction, the focus is on understanding the content
of the input to capture its semantic proximity to the query. In
the backward interaction, the output interacts inversely with
the input, interpreting the context of the input based on their
mutual proximity. This bidirectional concept has been widely
applied across various domains. In language models, bidirec-
tional attention captures both textual content and contextual
information [25]. In text retrieval, performance is enhanced
by generating pseudo-documents corresponding to queries
and pseudo-queries corresponding to documents [26]. In im-
age retrieval, different strategies are applied to images within
a query’s close set (ranked by original similarity) and its far
set (re-ranked using contextual similarity from k-reciprocal
nearest neighbors) [24, 27 –29]. In recommendation systems,
head ranking and tail ranking are utilized to select appropriate
items for users.
Inspired by this concept, we investigates the behaviors of
poisoned and benign documents under bidirectional retrieval
in Section 3.1. We discovered distinct distribution patterns
of poisoned and benign documents in bidirectional ranking.
Based on this finding, we propose the BiRD defense method,
which, for the first time, successfully applies the aforemen-
tioned concept to the RAG defense problem.
3 Motivation
This section aims to provide a clear understanding of the be-
havioral differences between poisoned and benign documents
in the bidirectional ranking. We first present the pilot experi-
ments in Section 3.1 that reveal these differences, followed by
3

(a) HotpotQA-ANCE
 (b) HotpotQA-Contriever
 (c) HotpotQA-DPR
(d) HotpotQA-Contriever
 (e) MSMARCO-Contriever
 (f) NQ-Contriever
(g) PoisonedRAG
 (h) Topic-Flip-PRO
 (i) Topic-Flip-CON
Figure 3: Rank position poison frequency comparison: (Top) different retrievers on HotpotQA; (Middle) Contriever across
datasets; (Bottom) comparison of attack algorithms.
a reason analysis in Section 3.2 that explains the underlying
mechanisms.
3.1 Pilot Experiments
To gain an in-depth understanding of the behavioral differ-
ences between poisoned and benign documents in the re-trieval system, we conducted statistical analyses of bidirec-
tional ranking patterns using the HotpotQA, MS-MARCO,
NQ, and PROCON datasets, along with the Contriever, DPR,
and ANCE retrievers, under PoisonedRAG [4] and Topic-
FlipRAG [8] attack. Given the massive scale of the full
HotpotQA, MS-MARCO, and NQ datasets (each contain-
ing millions of documents), we first constructed a subset.
4

Figure 4: t-SNE visualization of textual embeddings for be-
nign documents, poisoned documents, and queries generated
by Contriever on the subset of HotpotQA.
For each target query, we selected the 20 most relevant be-
nign documents from the corpus and additionally poisoned 5
documents to form the subset. We then performed forward
retrieval on the three retrievers to obtain the top 20 most
relevant documents. The frequency of poisoned documents
appearing at each rank position constitutes the first column
of the heatmap. For each document retrieved in the forward
ranking, we conducted a backward retrieval to obtain its back-
ward ranking. We then calculated the frequency of poisoned
documents at each rank position in these backward rankings.
Each backward ranking corresponds to one row in the right
side of heatmap, with the k-th row representing the backward
ranking result of the document that ranked k-th in the forward
ranking.
Figure 3 shows the distribution of poisoned documents
in the bidirectional ranking under PoisonedRAG attack on
the HotpotQA dataset with different retrievers. Figure 3
shows the distribution of poisoned documents in the bidirec-
tional ranking under Topic-FlipRAG attack on the PROCON
datasets. More details can be found in the Appendix A.2.
Through systematic experimental analysis as shown in Fig-
ures 3, we have discovered and revealed for the first time
systematic differences in backward ranking behavior between
poisoned and benign documents:
•Forward Ranking Distribution Asymmetry:In the
forward ranking process, poisoned documents exhibit a
"top-heavy" distribution, predominantly concentrating at
the peak of the retrieval list to maximize their influence.
In contrast, benign documents are largely dispersed to-
ward the bottom, failing to achieve a comparable level
of retrieval priority.
•Backward Ranking Stability:When generating a back-
ward ranking using a poisoned document as the pivot,
other poisoned documents remain highly concentrated atthe top positions, reflecting the strong contextual correla-
tion among them. Conversely, the backward ranking of
a benign document shows a significantly more uniform
and consistent distribution across the corpus, reflecting
the inherent diversity of benign content.
•Forward-Backward Consistency:Poisoned documents
exhibit a high degree of structural similarity between the
forward ranking of queries and their own backward rank-
ing. In contrast, benign documents manifest a significant
divergence between the two, suggesting a more complex
and diverse set of contextual relationships.
3.2 Reason Analysis
To further elucidate the underlying mechanisms governing
our statistical observations, we visualize, using t-SNE, the
textual embeddings generated by Contriever for the subset of
the HotpotQA dataset that we have constructed. As illustrated
in Figure 4, the visualization demonstrates a stark dichotomy
in the distribution of the two document types. The poisoned
documents (red crosses) tend to form highly dense clusters.
In stark contrast, the embeddings of benign documents (green
circles) are widely and uniformly dispersed, reflecting the
inherent semantic diversity of a genuine knowledge base.
While prior research has observed the tendency of poisoned
documents to cluster [11], a reasonable explanation has been
lacking. As shown in the projection, to maximize the attack
success rate and ensure that the LLM generates the target ad-
versarial output with high confidence, attackers must elevate
the ranking position (determined by semantic similarity) of
otherwise irrelevant poisoned documents. To achieve this,
attackers employ various techniques to optimize the embed-
dings of poisoned documents, bringing them closer to the tar-
get query. Simultaneously, they must constrain the magnitude
of these modifications to avoid easy detection. Consequently,
these documents approach the query embedding along similar
trajectories, causing the generated malicious documents to
become highly consistent in the embedding space and con-
verge into tight clusters. In contrast, benign documents, due to
their diverse origins and heterogeneous subject matter, remain
naturally scattered in a diffuse distribution.
Based on this observation, our core argument is as follows:
poisoned documents can be effectively distinguished and fil-
tered by analyzing content relevance and context consistency
to the query. content relevance, determined by semantic simi-
larity, is informed by the finding of forward ranking distribu-
tion asymmetry. context consistency, reflecting a document’s
context consistency to the query through ranking similarity
between forward and backward rankings, is informed by the
findings of backward ranking stability and forward-backward
consistency. The primary advantage of BiRD is that it does
not rely on LLMs to extract deep semantic information; in-
stead, it fully leverages the ranking signals inherent in the
5

𝑑31
𝑑41
𝑑51
𝑑42
𝑑52𝑑22
𝑑33
𝑑43
𝑑53𝑑23
𝑑34
𝑑44
𝑑54𝑑14
𝑑35
𝑑45
𝑑55𝑑15
𝑑25
Stage 1: Forward Retrieval Stage 2: Backward Retrieval
𝑑11
 𝑑21
𝑑12
𝑑32
𝑑13
𝑑24
Context 
consistencyContent 
relevance
Content 
relevance
Content 
relevance
Content 
relevance
Content 
relevanceContext 
consistencyContext 
consistencyContext 
consistencyContext 
consistencyq
q
q
q
q𝑑1𝑞
𝑑2𝑞
𝑑3𝑞
𝑑4𝑞
𝑑5𝑞𝑑1𝑞
𝑑2𝑞
𝑑3𝑞
𝑑4𝑞
𝑑5𝑞𝑅fw𝑞
𝑅fw𝑞
𝑅fw𝑞
𝑅fw𝑞
𝑅fw𝑞𝑅bw1
𝑅bw2
𝑅bw3
𝑅bw4
𝑅bw5𝑟cc1
𝑟cc2
𝑟cc3
𝑟cc4
𝑟cc5𝑟cr1
𝑟cr2
𝑟cr3
𝑟cr4
𝑟cr5
Stage 3: Threat FilteringRisk
evaluation{𝑟cr𝑖}
{𝑟c𝑐𝑖}Threshold
classification
𝑑3𝑞
𝑑4𝑞
𝑑5𝑞
User query (q) : Who won the 2024 U.S. presidential election?
LLMAnswer: Donald Trump
Poisoned
document
Benign
document
Figure 5: Overview of BiRD method. The process is mainly divided into three stages: forward retrieval, backward retrieval, and
threat filtering. It integrates the query-document content relevance from the forward ranking and context consistency from the
bidirectional ranking to detect and filter poisoned documents.
retrieval process and achieves lower computation cost, distill-
ing robust security features from the system’s fundamental
ranking behavior.
4 Methodology
This section details the BiRD method as illustrated in Figure 5.
First, Section 4.1 formally defines the problem to be addressed
in RAG security. Subsequently, we introduces BiRD method,
including forward retrieval (Section 4.2), backward retrieval
(Section 4.3), and threat filtering (Section 4.4). Finally, we
analyze the time and space complexity of BiRD in Section 4.5.
4.1 Threat Model
We consider a poisoning attack scenario where an adversary
aims to mislead a RAG system by injecting carefully crafted
malicious documents into the retrieval corpus. The attacker
selects a set of target queries Qtarget and specifies an intendedfalse answer r(q) for each q∈Q target. For instance, for a query
such as “Who won the 2024 U.S. presidential election?”, the
attacker may intend the RAG system to output the wrong
answerr(q) =“Kamala Harris”.
To achieve this goal, the attacker injects a set of poisoned
documents Dpoison into the corpus. We denote the original
corpus asD benign , and the poisoned corpus as
D′=D benign∪D poison .(1)
The objective of the corpus poisoning attack is that for a
target query q∈Q target, at least one poisoned document is
retrieved in the top-kresults fromD′. Formally, let
top-k 
R(q,D′)
←d 1<d 2<···<d k,(2)
where R(·,·) is the retriever and d1<···<d kdenotes the
ranking order.
Subsequently, these kdocuments serve as context, along
with the query, as input to a LLM. The attack is considered
6

successful if the LLM produce the target false answer:
LLM 
q,top-k(R(q,D′))
=r(q),∀q∈Q target.(3)
From the defender’s perspective, we operates without any
assumptions regarding the attack strategy, making it applica-
ble to both white-box and black-box scenarios. The defender
only has access to the initial ranking list returned by the re-
triever Rfor a given user query q, which may contains a
mixture of benign and poisoned documents. The defender’s
objective is to filter out the poisoned documents from this
initial ranking without knowing which documents are mali-
cious, thereby obtaining a clean subset of documents Dclean,
where |Dclean|=N . Ideally, the filtered context provided to
the LLM should contain no poisoned documents.
4.2 Forward Retrieval
Content relevance:To quantitatively measure the semantic
content relevance, we utilize the semantic similarity function
provided by the retriever. Given a user query qand a corpus
D′, the retriever Rreturns the top- kdocuments based on
their relevance to the query, forming the forward ranking (an
ordered list):
Rq
fw←top-k 
R(q,D′)
←dq
1<dq
2<···<dq
k.(4)
Here, dq
idenotes the i-th document in the forward ranking of
query q. For each dq
iin this ranking, its content relevance is
computed as:
ri
cr=sim(q,dq
i),(5)
where sim(·,·) denotes the similarity function used by the
retriever, e.g., cosine similarity.
4.3 Backward Retrieval
For each document dq
iin the forward ranking Rq
fw, we treat dq
i
itself as a new query. We then perform retrieval on the corpus
to obtain its backward ranking (also an ordered list):
Ri
bw←top-k 
R(dq
i,D′)
←di
1<di
2<···<di
k.(6)
Here, di
jdenotes the j-th document in the backward ranking
of query dq
i.Context consistency:To measure the ranking
context consistency, we compute the ranking similarity be-
tween the backward ranking and the forward ranking:
ri
cc=ρ 
Rq
fw,Ri
bw
=

1−6∑
d∈C(rf(d)−r b(d))2
|C| 
|C|2−1 ,if|C| ≥2,
0,if|C|<2.(7)
where
C={d|d∈Rq
fw∧d∈Ri
bw}(8)Algorithm 1Bidirectional Ranking Defense (BiRD)
Require: User query q, document corpus D, retriever R, pa-
rametersk,ε
Ensure:Benign document setD clean
1:Rq
fw←R(q,D′)▷Forward ranking
2:foreach documentdq
i∈Rq
fwdo
3:Ri
bw←R(dq
i,D′\{dq
i})▷Backward ranking
4:ri
cr←sim(q,dq
i)
5:ri
cc←ρ(Rq
fw,Ri
bw)
6:S(dq
i)←ri
cr/(1−ri
cc)
7:end for
8:D clean← {dq
i|dq
i∈Rq
fwandS(dq
i)≤ε}▷Filter to
obtain clean set
9:returnD clean
represents the set of common documents in both rankings,
|C|denotes the number of common documents, and rf(d)
andrb(d)denote the rank of document dinRq
fwandRi
bw,
respectively. ri
cc∈[−1,1] . According to the ranking inequal-
ity, when the forward ranking Rq
fwand the backward ranking
Ri
bwcompletely coincide, ri
cc=1; when they are completely
opposite, ri
cc=−1 . Here, ρrepresents the Spearman’s rank
correlation coefficient. We will discuss the impact of ranking
similarity method in detail in Section 5.4.
4.4 Threat Filtering
Risk evaluation:Based on the key observation that poisoned
documents tend to exhibit high backward/forward ranking
similarity (i.e., ri
ccis close to 1), whereas the backward rank-
ing of a benign document significantly diverges from the
forward ranking (i.e., for each d∈D benign ,rccis close to 0),
we design the following filtering criterion.
For each document dq
i∈Rq
fw, to amplify the distinction
between poisoned and benign documents, we compute a final
composite score using division:
S(dq
i) =ri
cr
1−ricc.(9)
Threshold classification:Given a predefined threshold ε, the
final set of benign documentsD cleanis then:
Dclean={dq
i|dq
i∈Rq
fwandS(dq
i)≤ε}.(10)
In our method, the threshold εis currently selected em-
pirically. In practical deployment, εcan be determined by
sampling poisoned documents generated by existing attack
methods to form a validation set (see details in Section 5.6).
The impact of the threshold εwill be discussed in detail in
Section 5.4. The complete algorithm is detaild in Algorithm 1.
7

4.5 Complexity Analysis
Time complexity: Assume that the time complexity of a
single retrieval operation by the retriever RisO(N) , where
N=|D′|is the total number of documents in the corpus. The
overall time complexity of the BiRD algorithm consists of
the following components: one forward retrieval O(N) ,k
backward retrievals O(kN) , and the computation of ranking
correlations O(k2)(e.g., using the Spearman’s rank corre-
lation coefficient). Therefore, the total time complexity is
O((k+1)N+k2). Since typically k≪N (e.g., k=20 , while
Ncan be on the order of millions), and if matrix computa-
tion is employed, the kbackward retrievals can be performed
simultaneously, then the dominant overhead stems from ap-
proximately 2 times the retrieval operation time (i.e., one
forward and one effectively parallelized backward retrieval).
In practice, the retrieval time in RAG often constitutes only a
small fraction of the overall RAG pipeline. Compared to other
methods that require invoking external LLMs for generation,
our approach essentially does not affect the real-world user
experience of RAG.
Space complexity:The space complexity of BiRD is O( N),
consistent with the original retriever. If one opts to precom-
pute and store the dense pairwise similarity matrix, the space
overhead increases to O(N2).
Furthermore, in practical applications, the document cor-
pus is usually static and known in advance. Thus, pairwise
similarities between all documents can be precomputed and
cached. This allows the reuse of existing similarity results
during the backward retrieval phase, avoiding redundant cal-
culations. This precomputation strategy significantly reduces
the runtime overhead of the algorithm.
5 Experiments
This chapter provides a detailed description of the extensive
and systematic experiments conducted to evaluate our defense
method. We first introduce the fundamental experimental
setup (Section 5.1), followed by a comprehensive discussion
of the baseline experiments (Section 5.2), dynamic perfor-
mance experiments (Section 5.4) , and runtime experiments
(Section 5.5).
5.1 Experimental Setup
Datasets:Our evaluation encompasses three prominent and
widely-adopted open-domain question answering datasets,
chosen for their diversity in complexity and task nature: Nat-
ural Questions (NQ) [30], which consists of real-world ques-
tions derived from Google search queries; MSMARCO [31], a
large-scale machine reading comprehension dataset from Mi-
crosoft; and HotpotQA [32], a challenging dataset designed
for multi-hop reasoning. These datasets collectively cover a
spectrum from straightforward factoid retrieval to complex,interconnected reasoning scenarios, thereby enabling a robust
and holistic assessment of our defense method’s performance
and resilience across varied contexts.
Retrievers:To ensure the general applicability of our ap-
proach across different retrieval paradigms, we tested our
defense method using three prevalent and widely-used dense
retrieval models: Contriever [33], a general-purpose retrieval
model trained with contrastive learning; ANCE [34] (Approx-
imate Nearest Neighbor Negative Contrastive Estimation),
which employs hard negative sampling via an approximate
nearest neighbor search; and DPR [35] (Dense Passage Re-
triever), the seminal and commonly adopted model for dense
passage retrieval.
Large Language Models:To validate the generalization
capability of our defense method and ensure its robustness
is not model-specific, we selected several mainstream open-
source large language models for evaluation, including Qwen-
7B [36], Mistral-7B [37], and Llama-3.1-8B.
Attack Method:We adopt the PoisonedRAG [4] attack
and prompt injection attack [21] as our primary threat model
to simulate adversarial conditions. This attack strategy aims
to manipulate the retrieval system’s output by injecting metic-
ulously crafted poisoned documents into the knowledge base.
Baseline Methods:To benchmark the performance of
BiRD, we compare it against several representative and con-
ceptually diverse defense methods from prior work:
•Vanilla RAG:The original, undefended RAG system
serving as the vulnerable baseline.
•InstructRAG [16]:InstructRAG enhances the Retrieval-
Augmented Generation process by employing self-
synthesized rationales. It specifically instructs the lan-
guage model to explain the derivation of the ground-truth
answer from retrieved documents to improve internal
reasoning.
•RobustRAG [13]:This defense method is based on key-
word extraction. Instead of providing the full set of
retrieved documents as context, the defender extracts
key phrases and provides only these keywords to the
large language model.
•ReliabilityRAG [12]:This is a defense method based on
solving the Maximum Independent Set (MIS) problem.
The defender constructs a contradiction graph using the
answers generated by the large language model, which
are produced by pairing each document individually with
the query. The maximum independent set is then solved
within this graph, as this set corresponds to the subset of
benign documents.
Implementation Details:For the purpose of this experi-
ment, we set the attacker’s corruption size mto 5. For our
BiRD method, we set the filtering threshold εto 2.5 and the
number of initially retrieved documentskto 20 in default.
8

Table 1: Experimental Results on NQ. Best results (↓for ASR,↑for ACC) are in bold; second-best results are underlined.
Model Baseline VenueANCE Contriever DPR
Clean PIA PoisonedRAG Clean PIA PoisonedRAG Clean PIA PoisonedRAG
Qwen2.5-7BVanillaRAG -0.76 0.20/ 0.59 0.87 / 0.370.770.26 / 0.69 0.89 / 0.340.730.29 / 0.58 0.87 / 0.44
RobustRAG arXiv ’24 0.60 0.74 / 0.48 0.61 / 0.40 0.62 0.79 / 0.49 0.62 / 0.42 0.60 0.70 / 0.50 0.62 / 0.43
InstructRAG ICLR ’25 0.68 0.55 / 0.49 0.77 / 0.29 0.63 0.51 / 0.57 0.79 / 0.24 0.63 0.54 / 0.48 0.79 / 0.32
ReliabilityRAG NeurIPS ’25 0.48 0.41 / 0.44 0.61 / 0.43 0.54 0.40 / 0.44 0.70 / 0.53 0.55 0.32 / 0.50 0.70 / 0.41
BiRD Ours 0.61 0.22 /0.64 0.08/0.750.620.25/0.71 0.16/0.620.64 0.25/0.60 0.30/0.63
Mistral-7BVanillaRAG -0.77 0.16/ 0.60 0.93 / 0.160.760.20 / 0.59 0.91 / 0.200.800.29 /0.560.91 / 0.18
RobustRAG arXiv ’24 0.21 0.33 / 0.15 0.40 / 0.17 0.17 0.33 / 0.14 0.49 / 0.18 0.16 0.34 / 0.18 0.45 / 0.14
InstructRAG ICLR ’25 0.38 0.48 / 0.39 0.63 / 0.27 0.25 0.39 / 0.51 0.54 / 0.33 0.30 0.34 / 0.48 0.55 / 0.31
ReliabilityRAG NeurIPS ’25 0.12 0.29 / 0.17 0.40 / 0.18 0.090.18/ 0.15 0.35 / 0.14 0.120.28/ 0.11 0.41 / 0.15
BiRD Ours 0.62 0.19 /0.61 0.06/0.710.65 0.20 /0.64 0.10/0.720.66 0.29 / 0.55 0.24/0.65
Llama-3.1-8BVanillaRAG -0.880.21 / 0.72 0.83 / 0.78 0.870.30 / 0.77 0.85 / 0.76 0.89 0.31/ 0.65 0.84 / 0.68
RobustRAG arXiv ’24 0.36 0.75 / 0.52 0.50 / 0.45 0.30 0.79 / 0.57 0.45 / 0.33 0.25 0.72 / 0.59 0.51 / 0.44
InstructRAG ICLR ’25 0.83 0.34 /0.810.59 / 0.67 0.83 0.31 /0.810.57 / 0.68 0.78 0.34 /0.750.69 / 0.62
ReliabilityRAG NeurIPS ’25 0.22 0.68 / 0.44 0.50 / 0.39 0.31 0.64 / 0.44 0.46 / 0.33 0.24 0.64 / 0.39 0.52 / 0.42
BiRD Ours 0.720.18/ 0.72 0.15/0.790.790.29/ 0.80 0.19/0.820.77 0.38 / 0.66 0.28/0.75
Table 2: Experimental Results on Msmarco. Best results ( ↓for ASR, ↑for ACC) are in bold; second-best results are underlined.
Model Baseline VenueANCE Contriever DPR
Clean PIA PoisonedRAG Clean PIA PoisonedRAG Clean PIA PoisonedRAG
Qwen2.5-7BVanillaRAG -0.810.26 /0.690.86 / 0.470.81 0.26/0.630.85 / 0.490.780.26 /0.650.88 / 0.47
RobustRAG arXiv ’24 0.71 0.69 / 0.52 0.69 / 0.56 0.66 0.71 / 0.61 0.69 / 0.48 0.69 0.65 / 0.58 0.67 / 0.51
InstructRAG ICLR ’25 0.79 0.63 / 0.56 0.80 / 0.35 0.76 0.53 / 0.55 0.84 / 0.35 0.75 0.50 / 0.55 0.79 / 0.32
ReliabilityRAG NeurIPS ’25 0.64 0.27 / 0.59 0.58 / 0.60 0.64 0.30 / 0.62 0.57 / 0.55 0.70 0.31 / 0.59 0.58 / 0.57
BiRD Ours 0.590.22/0.69 0.17/0.760.60 0.27 /0.63 0.24/0.710.640.25/ 0.63 0.27/0.68
Mistral-7BVanillaRAG -0.810.13 /0.720.88 / 0.24 0.860.20 / 0.59 0.75 / 0.32 0.770.23 / 0.55 0.76 / 0.32
RobustRAG arXiv ’24 0.20 0.04 / 0.04 0.22 / 0.08 0.24 0.04 / 0.02 0.28 / 0.11 0.230.03/ 0.08 0.18 / 0.13
InstructRAG ICLR ’25 0.24 0.14 / 0.57 0.30 / 0.21 0.25 0.15 / 0.51 0.28 / 0.32 0.25 0.15 / 0.49 0.37 / 0.28
ReliabilityRAG NeurIPS ’25 0.180.03/ 0.06 0.20 / 0.05 0.180.03/ 0.06 0.19 / 0.16 0.120.03/ 0.11 0.17 / 0.12
BiRD Ours 0.69 0.12 / 0.69 0.12/0.750.71 0.23 /0.62 0.14/0.760.67 0.20 /0.61 0.14/0.71
Llama-3.1-8BVanillaRAG -0.900.29 /0.830.82 / 0.77 0.90 0.35/ 0.82 0.89 / 0.74 0.89 0.31/ 0.73 0.81 / 0.79
RobustRAG arXiv ’24 0.34 0.81 / 0.61 0.39 / 0.50 0.40 0.72 / 0.54 0.39 / 0.49 0.35 0.59 / 0.59 0.43 / 0.51
InstructRAG ICLR ’25 0.76 0.39 / 0.72 0.60 / 0.70 0.74 0.37 / 0.74 0.58 / 0.74 0.74 0.49 / 0.75 0.61 / 0.69
ReliabilityRAG NeurIPS ’25 0.40 0.67 / 0.48 0.46 / 0.44 0.39 0.65 / 0.540.33/ 0.34 0.30 0.56 / 0.53 0.37 / 0.39
BiRD Ours 0.82 0.28/ 0.82 0.30/0.870.81 0.36 /0.830.35 /0.870.83 0.35 /0.76 0.36/0.83
Metrics:We employ Attack Success Rate (ASR) and Ac-
curacy (ACC) as our primary evaluation metrics. For each
response generated by the RAG system, the attack is consid-
ered successful if the output contains an exact string match
to the adversary’s target incorrect answer. Conversely, the
response is considered correct if it matches the ground-truth
answer. These metrics allow us to quantitatively assess both
the vulnerability of the system under attack and the effective-
ness of the defense methods in preserving answer quality.
5.2 Baseline Experiments
To systematically evaluate our approach, we conducted large-
scale and comprehensive experiments. We selected Vanil-
laRAG, InstructRAG, RobustRAG, and ReliabilityRAG asbaselines and employed two attack methods, PoisonedRAG
and PIA. For each configuration, we performed on three
datasets, three retrievers, and three LLMs. The responses gen-
erated by the RAG system were evaluated via string matching
against both the ground-truth answers and the target incorrect
answers.
As shown in Tables 1, 2, and 3, BiRD achieves state-of-
the-art performance across numerous baselines. Under the
PoisonedRAG attack, BiRD reduces the ASR by up to 87%
and improves Acc by 55% compared to VanillaRAG on the
Mistral-NQ-ANCE configuration. In the face of PIA, al-
though BiRD does not reach the absolute lowest ASR on
Mistral-7B, it ensures a more practical response accuracy
compared to ReliabilityRAG. Consequently, our approach
achieves the optimal trade-off between defense robustness
9

Table 3: Experimental Results on HotpotQA. Best results ( ↓for ASR, ↑for ACC) are in bold; second-best results are underlined.
Model Baseline VenueANCE Contriever DPR
Clean PIA PoisonedRAG Clean PIA PoisonedRAG Clean PIA PoisonedRAG
Qwen2.5-7BVanillaRAG -0.61 0.34/ 0.49 0.93 / 0.24 0.66 0.32/ 0.54 0.93 / 0.290.66 0.42/0.550.95 / 0.24
RobustRAG arXiv ’24 0.54 0.74 /0.500.88 / 0.33 0.58 0.73 / 0.52 0.79 / 0.41 0.59 0.65 / 0.41 0.78 / 0.30
InstructRAG ICLR ’25 0.54 0.60 / 0.49 0.81 / 0.200.670.57 / 0.59 0.82 / 0.16 0.59 0.63 / 0.53 0.84 / 0.19
ReliabilityRAG NeurIPS ’25 0.48 0.42 / 0.33 0.65 / 0.36 0.50 0.36 / 0.39 0.70 / 0.39 0.420.42/ 0.39 0.60 / 0.31
BiRD Ours 0.490.34/ 0.49 0.22/0.590.57 0.36 /0.60 0.24/0.690.61 0.48 / 0.54 0.41/0.62
Mistral-7BVanillaRAG -0.740.27 /0.490.93 / 0.150.790.27 /0.670.93 / 0.160.720.39 / 0.46 0.95 / 0.13
RobustRAG arXiv ’24 0.080.19/ 0.04 0.41 / 0.11 0.02 0.22 / 0.06 0.29 / 0.10 0.04 0.17 / 0.05 0.35 / 0.12
InstructRAG ICLR ’25 0.65 0.67 / 0.29 0.78 / 0.24 0.74 0.57 / 0.40 0.80 / 0.28 0.60 0.65 / 0.29 0.89 / 0.19
ReliabilityRAG NeurIPS ’25 0.060.19/ 0.04 0.33 / 0.12 0.040.18/ 0.04 0.36 / 0.13 0.020.13/ 0.04 0.35 / 0.11
BiRD Ours 0.61 0.27 / 0.46 0.18/0.670.69 0.27 / 0.59 0.17/0.680.69 0.38 /0.48 0.28/0.59
Llama-3.1-8BVanillaRAG -0.79 0.35/ 0.60 0.88 / 0.65 0.87 0.33/ 0.69 0.79 / 0.71 0.750.43 / 0.56 0.90 / 0.61
RobustRAG arXiv ’24 0.21 0.73 / 0.42 0.44 / 0.37 0.22 0.78 / 0.52 0.45 / 0.37 0.26 0.70 / 0.42 0.51 / 0.38
InstructRAG ICLR ’25 0.78 0.53 / 0.50 0.79 / 0.42 0.82 0.44 / 0.42 0.61 / 0.51 0.69 0.38/ 0.35 0.68 / 0.41
ReliabilityRAG NeurIPS ’25 0.21 0.49 / 0.31 0.36 / 0.30 0.26 0.51 / 0.40 0.38 / 0.29 0.22 0.60 / 0.330.44/ 0.26
BiRD Ours 0.70 0.39 /0.64 0.33/0.820.78 0.37 /0.72 0.33/0.85 0.750.50 /0.640.51 /0.78
Table 4: Experimental results for bidirectional score abla- tion.
Best results ( ↓for ASR, ↑for ACC) are in bold; second-best
results are underlined.
Setting ri
crri
ccHotpotQA MSMARCO NQ
Forward Retrieval✓- 0.34 /0.700.41 /0.750.53 / 0.68
Backward Retrieval -✓0.52 / 0.62 0.42 / 0.71 0.44 /0.70
Bidirectional Retrieval✓ ✓0.24/ 0.69 0.24/ 0.71 0.16/ 0.62
and system utility.
5.3 Abalation Study
To verify the necessity of the bidirectional retrieval score, we
conducted ablation studies across multiple datasets. The ex-
perimental design evaluates performance metrics under three
configurations: content-relevance-only filtering ( ri
cr, thresh-
old 0.85), context-consistency-only filtering ( ri
cc, threshold
0.85), and the proposed composite score S(dq
i)(threshold
2.5).
Table 4 shows a significant performance complementarity
between the composite score and the single-signal variants.
While content relevance ( ri
cr) maintains a marginal lead in
Acc on HotpotQA and MSMARCO, reflecting its efficiency
in capturing direct semantic relevance, its security remains
highly vulnerable. In contrast, incorporating context con-
sistency ( ri
cc) as a structural signal acts as a security filter,
reducing the ASR on the NQ dataset by 0.37 compared to the
ri
cr-only strategy. The results indicate that the bidirectional
framework achieves a substantial improvement in defensi-
bility at the cost of minimal utility loss (<5%), striking an
optimal balance between security and practicality.Table 5: Experimental results for different rank similarity
metrics. Best results ( ↓for ASR, ↑for ACC) are in bold;
second-best results are underlined.
Method Jaccard RBO Spearman(Ours)
VanillaRAG 0.93 / 0.29 0.93 / 0.29 0.93 / 0.29
BiRD 0.37 / 0.54 0.88 / 0.400.24/0.69
|∆|(BiRD−Vanilla) 0.56 / 0.25 0.05 / 0.110.69/0.40
5.4 Dynamic Performance Experiments
Impact of context-consistency metric ρ(·,·) .To evaluate
the influence of different measurement methods of the sim-
ilarity between two rankings (used to compute ri
cc) on the
performance of our BiRD defense mechanism, we conducted
a series of experiments using three widely-adopted metrics:
Jaccard distance [38], Spearman’s rank correlation coeffi-
cient [39], and Rank-biased Overlap (RBO) [40].
As shown in Table 5, the BiRD defense mechanism signifi-
cantly reduces both the ASR and ACC when Spearman’s rank
correlation coefficient is employed as the ranking similarity
metric. In contrast, when using Jaccard distance or RBO, the
defense fails to effectively decrease the ASR and ACC. Fur-
thermore, the ASR actually increases when the RBO method
is applied.
Impact of filtering threshold ε.To evaluate the sensitivity
and operational characteristics of the BiRD defense mecha-
nism regarding the filtering threshold, we conducted a series
of experiments by varying εwithin a reasonable range of
[1,3.5].
As illustrated in Figure 7, the results reveal a clear and
compelling trade-off curve. Through a holistic evaluation of
both metrics, we determined that the interval between ε=2
andε=3 offers the most favorable balance. Within this
10

Figure 6: The variation of ASR and ACC across different datasets in relation to the parameter kin BiRD, where the difference
between the bars at the same position in the clustered bar chart represents the magnitude of our reduction or improvement.
Figure 7: ASR and ACC versus the filtering threshold εin
BiRD.
range, the system maintains a high accuracy (ACC >0.6)
while keeping the attack success rate at a relatively controlled
level (ASR<0.25).
In practical applications, the selection of an appropriate
threshold must be based on the specific trade-off between
security and utility. A lower εsignifies a more stringent
defense that may discard a greater number of documents,
including clean ones. Conversely, a higher εrepresents a
more permissive defense strategy that may include poisoned
documents but offers higher system utility.
Impact of retrieval size k.To evaluate the impact of the re-
trieval size kon the robustness of our defense method, we
conducted a series of experiments on the HotpotQA, MS-
MARCO, and NQ datasets. We employed Qwen and Con-
triever as the LLM and retriever respectively while varying
the parameterkwithin a reasonable range from 10 to 50.
As illustrated in Figure 6, BiRD consistently reduces the
ASR and improves the ACC to varying degrees across all
tested values of k. Our approach remains effective in lowering
the ASR even when kis large. This performance demonstrates
that BiRD can successfully filter out poisoned documents even
Figure 8: The variation of ASR and ACC across different
datasets in relation to the parametermfor attack.
in scenarios where they occupy a very small proportion of the
total results, such as when only m=5 poisoned documents
are present within an initial retrieval set ofk=50.
Impact of corruption size m.To evaluate the impact of the
corruption size m, which represents the number of poisoned
documents targeting a single query, on the robustness of our
defense method, we conducted a series of experiments us-
ing three large language models: Qwen, Mistral, and Llama.
These experiments utilized HotpotQA as the dataset and Con-
triever as the retriever while varying the parameter mwithin
a reasonable range from 0 to 5.
As illustrated in Figure 8, BiRD maintains high robust-
ness against variations in the corruption size. Even as m
increases, our defense mechanism keeps the performance of
the RAG system within a secure and reliable range. In con-
trast, VanillaRAG exhibits a significant increase in ASR and
a notable decline in ACC even when faced with only a single
poisoned document. These results further highlight the inher-
ent vulnerability of standard RAG systems and demonstrate
the effectiveness of our proposed defense.
11

Table 6: Efficiency Analysis: Average Runtime (s/query) on
three datasets. Best results ( ↓for ASR, ↑for ACC) are in
bold; second-best results are underlined.
MethodNQ HotpotQA MSMARCO
Cont. ANCE DPR Cont. ANCE DPR Cont. ANCE DPR
Model: Qwen2.5-7B-Instruct
InstructRAG 5.57 6.07 5.69 5.65 6.18 5.80 5.36 5.67 5.71
ReliabilityRAG 8.92 28.30 27.98 25.90 26.53 26.61 25.72 25.53 25.48
RobustRAG 25.73 23.32 23.24 19.73 23.43 21.81 31.72 22.09 21.68
BiRD (Ours) 3.00 3.61 3.00 3.00 6.05 5.44 3.00 4.83 4.22
Model: Mistral-7B
InstructRAG 4.93 4.55 4.82 4.67 3.98 3.86 4.794.92 4.34
ReliabilityRAG 13.37 14.31 13.80 14.68 16.62 14.61 11.18 11.89 11.23
RobustRAG 10.21 12.51 11.02 12.56 12.87 12.24 8.50 7.83 8.46
BiRD (Ours) 3.616.05 4.83 4.93 7.37 6.15 7.27 3.718.49
Model: Llama-3.1-8B
InstructRAG 4.91 4.965.54 5.54 5.23 5.21 6.226.13 5.73
ReliabilityRAG 39.49 59.88 40.79 38.01 40.01 38.69 38.15 38.05 37.65
RobustRAG 46.59 46.55 44.98 23.56 48.45 23.15 21.30 22.48 21.91
BiRD (Ours) 3.005.44 4.22 4.326.76 5.54 6.66 6.057.88
5.5 Runtime Experiments
In addition to evaluating defense effectiveness and utility
preservation, we conducted a comprehensive runtime analysis
to assess the practical efficiency overhead introduced by dif-
ferent defense mechanisms. In the real-world deployment of
secure RAG systems, computational cost is a critical factor.
Using the same hardware configuration of four A6000
48GB GPUs, we measured the average end-to-end inference
time per query for each defense method across three evaluated
LLMs, three datasets, and three retrievers. The reported time
encompasses document retrieval, defense processing, and the
LLM generation process.
The runtime comparison is presented in Table 6. The results
reveal significant efficiency differences among the various de-
fense strategies. Compared to previous work, our proposed
BiRD method introduces only a minimal overhead. On aver-
age, it adds a mere 0.9s of additional latency, which stands in
stark contrast to the extra overhead of tens of seconds required
by other methods. This demonstrates that the bidirectional
ranking and filtering process of BiRD is highly efficient.
The runtime experiments confirm that BiRD achieves an
ideal balance between security and efficiency. While provid-
ing robust defense, it maintains a latency profile very close
to the vulnerable Vanilla RAG baseline and is significantly
more efficient than alternative defense methods.
5.6 Validation-Set Threshold Selection
Goal and setup.To validate the practicality of the threshold
selection strategy described in Section 4.2.3, we construct a
validation set on MSMARCO using poisoned documents from
PoisonedRAG. Specifically, we treat PoisonedRAG-poisoned
documents as the validation set, compute the score S(dq
i)for
poisoned documents, and estimate their score distribution.Table 7: PIA defense on MSMARCO with ε=2.5 selected
from PoisonedRAG validation-set. Best results ( ↓for ASR, ↑
for ACC) are in bold; second-best results are underlined.
Baseline Contriever ANCE DPR
RobustRAG 0.71 / 0.61 0.69 / 0.52 0.65 / 0.58
InstructRAG 0.53 / 0.55 0.63 / 0.56 0.50 / 0.55
ReliabilityRAG 0.30 / 0.62 0.27 / 0.59 0.31 / 0.59
BiRD (Ours)0.27/0.63 0.22/0.69 0.25/0.63
We then select a fixed threshold ε=2.5 according to this
distribution. Importantly, we donotuse any PIA-poisoned
documents in this calibration process. After selecting ε, we
directly apply it to defend against PIA on MSMARCO under
the Qwen2.5-7B model and three retrievers. The defense
results are summarized in Table 7.
Analysis.As shown in Table 7, the threshold obtained from
PoisonedRAG validation generalizes well to the unseen PIA
setting. Across all three retrievers, BiRD achieves the lowest
ASR and the highest ACC among the compared defenses,
indicating that calibrating εusing PoisonedRAG-poisoned
documents is a feasible and effective deployment strategy.
This also suggests that the score distribution of poisoned
documents provides a stable signal for selecting εwithout
requiring access to the specific attack type encountered at test
time.
6 Conclusions
In conclusion, our proposed BiRD defense method introduce
and utilizes context consistency. The proposed defense strat-
egy is highly efficient, with minimal overhead compared to
other defense methods.
For future work, we plan to explore adaptive thresholding
techniques to further enhance the balance between security
and utility. Additionally, we aim to investigate the applica-
bility of our defense mechanism in other retrieval-augmented
tasks beyond question answering, such as summarization and
dialogue systems.
7 Acknowledgments
We thank the anonymous reviewers for their constructive feed-
back. We also thank the maintainers of the open-source re-
trievers and LLMs used in our evaluation, as well as the
authors who released the datasets and baselines that enabled
reproducible comparisons.
Ethical Considerations
We have reviewed the USENIX Security Ethics Guidelines.
This work aligns with its principles as it presents a defensive
12

security technique and aims to improve the robustness of
RAG systems against corpus poisoning.
Human subjects and sensitive data.Our study does not
involve human subjects experiments and does not collect
personal or sensitive user data. All experiments are performed
on public datasets and pre-trained models.
Dual-use and misuse risk.While our analysis discusses at-
tack behaviors for evaluation purposes, we focus on mitiga-
tion and do not provide operational guidance for deploying
real-world attacks. We encourage responsible disclosure and
careful deployment of RAG systems in safety-critical settings.
Bias and fairness.Corpus poisoning may amplify biases
present in data sources. Our defense is designed to reduce the
impact of maliciously injected content, but it is not a complete
solution to data bias; we recommend complementary bias
auditing when deploying RAG.
Open Science
We commit to releasing the artifacts necessary to reproduce
our results, including (1) the implementation of BiRD, (2)
scripts for running attacks and defenses, and (3) evaluation
code for ASR/ACC and runtime measurement.
The datasets are available at: https://public.
ukp.informatik.tu-darmstadt.de/thakur/
BEIR/datasets/ , The models are available at:
https://huggingface.co/.
References
[1]L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang,
Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu, “A
Survey on Hallucination in Large Language Models:
Principles, Taxonomy, Challenges, and Open Questions,”
ACM Trans. Inf. Syst., vol. 43, no. 2, pp. 42:1–42:55,
2025.
[2]Y . Yao, J. Duan, K. Xu, Y . Cai, Z. Sun, and Y . Zhang,
“A survey on large language model (LLM) security and
privacy: The Good, The Bad, and The Ugly,”High-
Confidence Computing, vol. 4, no. 2, p. 100211, 2024.
[3]Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai,
J. Sun, M. Wang, and H. Wang, “Retrieval-Augmented
Generation for Large Language Models: A Survey,”
2024.
[4]W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG:
Knowledge Corruption Attacks to Retrieval-Augmented
Generation of Large Language Models,” 2024.
[5]Z. Zhong, Z. Huang, A. Wettig, and D. Chen, “Poisoning
Retrieval Corpora by Injecting Adversarial Passages,”
inProceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, H. Bouamor,J. Pino, and K. Bali, Eds. Singapore: Association for
Computational Linguistics, 2023, pp. 13 764–13 775.
[6]M. Ben-Tov and M. Sharif, “GASLITEing the Retrieval:
Exploring Vulnerabilities in Dense Embedding-based
Search,” 2025.
[7]H. Song, Y .-a. Liu, R. Zhang, J. Guo, J. Lv, M. de Ri-
jke, and X. Cheng, “The Silent Saboteur: Impercepti-
ble Adversarial Attacks against Black-Box Retrieval-
Augmented Generation Systems,” 2025.
[8]Y . Gong, Z. Chen, M. Chen, F. Yu, W. Lu, X. Wang,
X. Liu, and J. Liu, “Topic-FlipRAG: Topic-Orientated
Adversarial Opinion Manipulation Attacks to Retrieval-
Augmented Generation Models,” 2025.
[9]G. Bagwe, S. S. Chaturvedi, X. Ma, X. Yuan, K.-C.
Wang, and L. Zhang, “Your RAG is Unfair: Exposing
Fairness Vulnerabilities in Retrieval-Augmented Gener-
ation via Backdoor Attacks,” 2025.
[10] X. Si, M. Zhu, S. Qin, L. Yu, L. Zhang, S. Liu, X. Li,
R. Duan, Y . Liu, and X. Jia, “SeCon-RAG: A Two-
Stage Semantic Filtering and Conflict-Free Framework
for Trustworthy RAG,” 2025.
[11] H. Zhou, K.-H. Lee, Z. Zhan, Y . Chen, Z. Li, Z. Wang,
H. Haddadi, and E. Yilmaz, “TrustRAG: Enhancing Ro-
bustness and Trustworthiness in Retrieval-Augmented
Generation,” 2025.
[12] Z. Shen, B. Imana, T. Wu, C. Xiang, P. Mittal, and
A. Korolova, “ReliabilityRAG: Effective and Provably
Robust Defense for RAG-based Web-Search,” 2025.
[13] C. Xiang, T. Wu, Z. Zhong, D. Wagner, D. Chen, and
P. Mittal, “Certifiably Robust RAG against Retrieval
Corruption,” 2024.
[14] X. Xian, G. Wang, X. Bi, J. Srinivasa, A. Kundu,
C. Fleming, M. Hong, and J. Ding, “On the Vulnerabil-
ity of Applying Retrieval-Augmented Generation within
Knowledge-Intensive Application Domains,” 2025.
[15] F. Wang, X. Wan, R. Sun, J. Chen, and S. O. Arik,
“Astute RAG: Overcoming Imperfect Retrieval Augmen-
tation and Knowledge Conflicts for Large Language
Models,” inProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), W. Che, J. Nabende, E. Shutova, and
M. T. Pilehvar, Eds. Vienna, Austria: Association for
Computational Linguistics, 2025, pp. 30 553–30 571.
[16] Z. Wei, W.-L. Chen, and Y . Meng, “InstructRAG:
Instructing Retrieval-Augmented Generation via Self-
Synthesized Rationales,” inThe Thirteenth International
Conference on Learning Representations, 2024.
13

[17] X. Ma, Y . Gong, P. He, H. Zhao, and N. Duan, “Query
Rewriting in Retrieval-Augmented Large Language
Models,” inProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing,
H. Bouamor, J. Pino, and K. Bali, Eds. Singapore:
Association for Computational Linguistics, 2023, pp.
5303–5315.
[18] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rock-
täschel, S. Riedel, and D. Kiela, “Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks,” in
Advances in Neural Information Processing Systems,
vol. 33. Curran Associates, Inc., 2020, pp. 9459–9474.
[19] Y . Zhang, Y . Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang,
E. Zhao, Y . Zhang, C. Xu, Y . Chen, L. Wang, A. T. Luu,
W. Bi, F. Shi, and S. Shi, “Siren’s Song in the AI Ocean:
A Survey on Hallucination in Large Language Models,”
2025.
[20] R. Xu, H. Liu, S. Nag, Z. Dai, Y . Xie, X. Tang, C. Luo,
Y . Li, J. C. Ho, C. Yang, and Q. He, “SimRAG: Self-
Improving Retrieval-Augmented Generation for Adapt-
ing Large Language Models to Specialized Domains,” in
Proceedings of the 2025 Conference of the Nations of the
Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1:
Long Papers), L. Chiruzzo, A. Ritter, and L. Wang, Eds.
Albuquerque, New Mexico: Association for Computa-
tional Linguistics, 2025, pp. 11 534–11 550.
[21] Y . Zhang, Q. Li, T. Du, X. Zhang, X. Zhao, Z. Feng,
and J. Yin, “HijackRAG: Hijacking Attacks against
Retrieval-Augmented Large Language Models,” 2024.
[22] Y . Jiao, X. Wang, and K. Yang, “PR-Attack: Coordi-
nated Prompt-RAG Attacks on Retrieval-Augmented
Generation in Large Language Models via Bilevel Opti-
mization,” 2025.
[23] J. Liu, “Maximal independent sets in bipartite graphs,”
Journal of Graph Theory, vol. 17, no. 4, pp. 495–507,
1993. [Online]. Available: https://onlinelibrary.wiley.
com/doi/abs/10.1002/jgt.3190170407
[24] Q. Leng, R. Hu, C. Liang, Y . Wang, and J. Chen, “Bidi-
rectional ranking for person re-identification,” in2013
IEEE International Conference on Multimedia and Expo
(ICME), 2013, pp. 1–6.
[25] M. Seo, A. Kembhavi, A. Farhadi, and H. Hajishirzi,
“Bidirectional Attention Flow for Machine Comprehen-
sion,” 2018.
[26] L. Wang, N. Yang, and F. Wei, “Query2doc: Query
Expansion with Large Language Models,” 2023.[27] D. Qin, S. Gammeter, L. Bossard, T. Quack, and L. van
Gool, “Hello neighbor: Accurate object retrieval with
k-reciprocal nearest neighbors,” inCVPR 2011, 2011,
pp. 777–784.
[28] H. Jegou, H. Harzallah, and C. Schmid, “A contextual
dissimilarity measure for accurate and efficient image
search,” in2007 IEEE Conference on Computer Vision
and Pattern Recognition, 2007, pp. 1–8.
[29] Z. Zhong, L. Zheng, D. Cao, and S. Li, “Re-ranking
Person Re-identification with k-reciprocal Encoding,”
2017.
[30] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins,
A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. De-
vlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M.-W.
Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov,
“Natural Questions: A Benchmark for Question Answer-
ing Research,”Transactions of the Association for Com-
putational Linguistics, vol. 7, pp. 452–466, 2019.
[31] P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao,
X. Liu, R. Majumder, A. McNamara, B. Mitra,
T. Nguyen, M. Rosenberg, X. Song, A. Stoica, S. Ti-
wary, and T. Wang, “MS MARCO: A Human Generated
MAchine Reading COmprehension Dataset,” 2018.
[32] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen,
R. Salakhutdinov, and C. D. Manning, “HotpotQA:
A Dataset for Diverse, Explainable Multi-hop Question
Answering,” inProceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing,
E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, Eds.
Brussels, Belgium: Association for Computational Lin-
guistics, 2018, pp. 2369–2380.
[33] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bo-
janowski, A. Joulin, and E. Grave, “Unsupervised Dense
Information Retrieval with Contrastive Learning,” 2022.
[34] L. Xiong, C. Xiong, Y . Li, K.-F. Tang, J. Liu, P. N.
Bennett, J. Ahmed, and A. Overwijk, “Approximate
Nearest Neighbor Negative Contrastive Learning for
Dense Text Retrieval,” inInternational Conference on
Learning Representations, 2020.
[35] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu,
S. Edunov, D. Chen, and W.-t. Yih, “Dense Passage Re-
trieval for Open-Domain Question Answering,” inPro-
ceedings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP), B. Webber,
T. Cohn, Y . He, and Y . Liu, Eds. Online: Association
for Computational Linguistics, 2020, pp. 6769–6781.
[36] Qwen, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng,
B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang,
14

J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin,
K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue,
P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia,
X. Ren, X. Ren, Y . Fan, Y . Su, Y . Zhang, Y . Wan, Y . Liu,
Z. Cui, Z. Zhang, and Z. Qiu, “Qwen2.5 Technical Re-
port,” 2025.
[37] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford,
D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel,
G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux,
P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and
W. E. Sayed, “Mistral 7B,” 2023.
[38] P. Jaccard, “The distribution of the flora in the alpine
zone.”New Phytologist, vol. 11, no. 2, pp. 37–50, 1912.
[Online]. Available: https://nph.onlinelibrary.wiley.
com/doi/abs/10.1111/j.1469-8137.1912.tb05611.x
[39] E. C. FIELLER, H. O. HARTLEY , and E. S. PEAR-
SON, “TESTS FOR RANK CORRELATION COEFFI-
CIENTS. I,”Biometrika, vol. 44, no. 3-4, pp. 470–481,
1957.
[40] W. Webber, A. Moffat, and J. Zobel, “A similarity mea-
sure for indefinite rankings,”ACM Transactions on In-
formation Systems, vol. 28, no. 4, pp. 1–38, 2010.
A Appendix
A.1 Formulation of Bidirectional Ranking De-
fense
To facilitate a clear understanding of the proposed framework,
we provide a comprehensive summary of the mathematical
notations and variables used throughout this paper. Table 8
categorizes the symbols related to the poisoning attack sce-
nario, the RAG process, and the specific parameters of BiRD
method. This serves as a quick reference for the formal defi-
nitions of the attack surface and the defensive scoring mecha-
nisms.
A.2 Preliminary Experiments
We perform a series of preliminary experiments to evaluate
the effectiveness of our approach. Figure 9 is a 3x3 matrix that
shows the discovery under PoisonedRAG attack. Figure 10 is
a 2x3 matrix that shows the discovery under Topic-FlipRAG
attack.Table 8: Summary of Variables and Mathematical Notations
Symbol Description
Dbenign Original benign document corpus
Dpoison Injected poisoned document set
D′Poisoned corpusD benign∪D poison
NTotal number of documents inD′
k Number of top documents retrieved by the sys-
tem
R(·,·) Retriever function mapping (query, corpus) to
a ranked list
top-k(·) Function that returns the top- kranked docu-
ments
dq
i Thei-th doc. in the forward ranking of query
q
di
j The j-th doc. in the backward ranking of query
dq
i
LLM(·,·)Large Language Model function
Dclean Filtered document set provided to the LLM as
context
Rq
fwForward ranking list for queryq
Ri
bwBackward ranking list usingdq
ias a query
ri
cr Content relevance ofdq
i(forward similarity)
ri
cc Context consistency of dq
i(ranking similarity)
Sim(·,·)Similarity function used by the retriever
ρ(·,·)Spearman’s rank correlation coefficient
CCommon documents betweenRq
fwandRi
bw
rf(d),r b(d)Rank ofdin forward and backward rankings
S(dq
i)Composite score fordq
i
εDefense threshold for filtering
15

(a) HotpotQA-ANCE
 (b) HotpotQA-Contriever
 (c) HotpotQA-DPR
(d) MSMARCO-ANCE
 (e) MSMARCO-Contriever
 (f) MSMARCO-DPR
(g) NQ-ANCE
 (h) NQ-Contriever
 (i) NQ-DPR
Figure 9: Comprehensive comparison of rank position poisoned frequency heatmaps across three datasets (rows) and three
retrievers (columns).
16

(a) Topic-Flip-PRO (ANCE)
 (b) Topic-Flip-PRO (Contriever)
 (c) Topic-Flip-PRO (DPR)
(d) Topic-Flip-CON (ANCE)
 (e) Topic-Flip-CON (Contriever)
 (f) Topic-Flip-CON (DPR)
Figure 10: Rank position poison frequency for Topic-Flip attacks: PRO strategy (top row) and CON strategy (bottom row) across
ANCE, Contriever, and DPR retrievers.
17