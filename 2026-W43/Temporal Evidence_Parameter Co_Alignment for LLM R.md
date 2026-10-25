# Temporal Evidence-Parameter Co-Alignment for LLM Reasoning over Temporal Knowledge Graphs

**Authors**: Jiapu Wang

**Published**: 2026-10-20 13:16:17

**PDF URL**: [https://arxiv.org/pdf/2610.28176v1](https://arxiv.org/pdf/2610.28176v1)

## Abstract
Large Language Models (LLMs) often struggle with temporally grounded reasoning because their static parametric memory is misaligned with the dynamic evolution of real-world knowledge. Although RAG and Parameter-Efficient Fine-Tuning (PEFT) provide practical ways to enhance LLMs, they remain insufficient for temporal reasoning. Existing retrieval methods typically prioritize semantic relevance while overlooking temporal validity, event order, and evolutionary dependencies, resulting in evidence that is relevant in isolation but temporally fragmented. Meanwhile, conventional PEFT methods usually employ fixed and input-agnostic low-rank updates, making it difficult to capture time-dependent entity semantics and relation drift. To address these limitations, we propose Temporal Evidence-Parameter Co-Alignment (TEPCA), a framework for LLM reasoning over TKGs. TEPCA aligns LLM reasoning with temporal knowledge at both the evidence and parameter levels. At the evidence level, it introduces time-constrained Dynamic Relevance-Guided Path (DRGP) pruning to preserve globally coherent temporal paths, together with Mutual-Information (MI)-based pruning to retain query-informative local facts under temporal constraints. At the parameter level, it integrates Dynamic-Gated LoRA (DG-LoRA), a timestamp-conditioned low-rank adaptation mechanism that dynamically modulates the effective adaptation subspace according to temporal context. This design enables LLMs to better capture fine-grained entity evolution and time-dependent relation semantics while preserving parameter efficiency. Experimental results on two TKG benchmarks demonstrate that TEPCA achieves state-of-the-art performance. Additional evaluations of DG-LoRA on eight commonsense QA datasets further verify its general input-adaptive capacity, scalability across LLM backbones, and interpretability.

## Full Text


<!-- PDF content starts -->

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 1
Temporal Evidence-Parameter Co-Alignment for
LLM Reasoning over Temporal Knowledge Graphs
Jiapu Wang
Abstract—Large Language Models (LLMs) often struggle with
temporally grounded reasoning because their static parametric
memory is misaligned with the dynamic evolution of real-world
knowledge. Although Retrieval-Augmented Generation (RAG)
and Parameter-Efficient Fine-Tuning (PEFT) provide practical
ways to enhance LLMs, they remain insufficient for temporal
reasoning. Existing retrieval methods typically prioritize semantic
relevance while overlooking temporal validity, event order, and
evolutionary dependencies, resulting in evidence that is relevant
in isolation but temporally fragmented. Meanwhile, conventional
PEFT methods usually employ fixed and input-agnostic low-rank
updates, making it difficult to capture time-dependent entity
semantics and relation drift. To address these limitations, we
propose Temporal Evidence- Parameter Co-Alignment (TEPCA), a
framework for LLM reasoning over Temporal Knowledge Graphs
(TKGs). TEPCA aligns LLM reasoning with temporal knowledge
at both the evidence and parameter levels. At the evidence level,
it introduces time-constrainedDynamic Relevance-Guided Path
(DRGP) pruningto preserve globally coherent temporal paths,
together withMutual-Information (MI)-based pruningto retain
query-informative local facts under temporal constraints. At
the parameter level, it integratesDynamic-Gated LoRA (DG-
LoRA), a timestamp-conditioned low-rank adaptation mechanism
that dynamically modulates the effective adaptation subspace
according to temporal context. This design enables LLMs to better
capture fine-grained entity evolution and time-dependent relation
semantics while preserving parameter efficiency. Experimental
results on two TKG benchmarks demonstrate that TEPCA
achieves state-of-the-art performance. Additional evaluations of
DG-LoRA on eight commonsense QA datasets further verify its
general input-adaptive capacity, scalability across LLM backbones,
and interpretability.
Index Terms—Large Language Model, Temporal Knowledge
Graph, Temporal Reasoning, Time-aware Retrieval, Dyanmic-
Gated LoRA.
I. INTRODUCTION
LARGE Language Models (LLMs) [1], [2] exhibit emergent
capabilities in logical reasoning [3] and generalization [4].
However, a fundamental architectural paradox persists: while
the world is continuously evolving, the parametric memory of
LLMs remains frozen at the pre-training stage [5]. This “static-
dynamic mismatch” renders state-of-the-art models brittle
against temporal drift, often leading to hallucinations where
outdated facts are presented as current truths [6], [7]. Since
retraining LLMs is computationally prohibitive, researchers
often turn to standard Retrieval-Augmented Generation (RAG)
[8]. Yet, standard RAG retrieves facts based on semantic simi-
larity rather than temporal relevance, resulting in a confusing
Jiapu Wang is with the School of Computer Science and Engineering,
Nanjing University of Science and Technology, Nanjing, Jiangsu, 210094,
China (e-mail: jiapu.wang@njust.edu.cn).
Fig. 1: Effect of TKGs constraints on LLM reasoning.
“contextual collage” that fails to capture the evolutionary lineage
of events.
Integrating Temporal Knowledge Graphs (TKGs) [9], [10]
provides a structured foundation for modeling fact provenance,
temporal validity, and event evolution, thereby offering a
promising way to mitigate the static–dynamic mismatch of
LLMs (Fig. 1). However, the key challenge lies not merely in
the availability of external temporal knowledge, but in whether
such knowledge can be retrieved, ordered, and utilized under
temporal constraints. Existing pipelines, including standard
RAG and GraphRAG [11], [12], typically prioritize semantic
relevance or static graph proximity while overlooking valid-
ity intervals, temporal order, and evolutionary dependencies.
Consequently, they tend to flatten dynamic histories into static
snapshots and retrieve facts that are individually relevant but
temporally incompatible with the query, forming a fragmented
evidence context rather than a coherent temporal evidence
chain [13]. Such evidence-level temporal misalignment ob-
scures the evolutionary lineage of events and prevents LLMs
from reliably distinguishing outdated states from current truths,
allowing hallucinations to persist even when external knowledge
is provided.
This limitation becomes more pronounced in time-sensitive
reasoning scenarios. Most retrieval-augmented or graph-
augmented methods [8], [14], [15] still operate on near-
static snapshots and remain insufficiently sensitive to temporal
arXiv:2610.28176v1  [cs.CL]  20 Oct 2026

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 2
dynamics [5]. As illustrated by Linet al.[16], snapshot-
based methods such as Plan-on-Graph (PoG) [17] can perform
reasonably well on slow-changing facts, but degrade markedly
on fast-changing ones due to stale parametric knowledge
and the lack of explicit temporal constraints [18]. Similarly,
GraphRAG-style pipelines often retrieve short and local chains
from near-static KGs, which may be semantically plausible yet
chronologically inconsistent [19]. Without explicitly modeling
validity windows, temporal precedence, and cross-evidence
consistency, the retrieved paths may conflict with the query
timestamp or with one another, thereby weakening factual
grounding and exacerbating hallucinations caused by temporal
inconsistency [20], [21].
Nevertheless, evidence-level alignment alone is insufficient
for temporally grounded reasoning, because the LLM must also
internalize and utilize time-dependent semantics through its
adaptation mechanism. Mainstream Parameter-Efficient Fine-
Tuning (PEFT) methods, such as LoRA [22] and DoRA [23],
are computationally efficient and effective in many downstream
tasks, but they usually employ fixed and input-agnostic low-
rank updates. This design implicitly assumes that entity and
relation semantics remain stable across contexts, which is often
violated in evolving knowledge environments. For instance, the
geopolitical role and relational context of Germany may vary
substantially across historical periods, and such period-specific
semantics are difficult to capture with a shared time-invariant
adaptation subspace [24]. By forcing heterogeneous temporal
states into the same static low-rank space, conventional
PEFT may induce a form of “temporal specificity collapse”,
where time-specific signals are over-smoothed or incorrectly
transferred across timestamps. Consequently, the model may
propagate outdated temporal semantics and generate predictions
that are inconsistent with the query-specific temporal context.
To address these challenges, we propose TEPCA (Temporal
Evidence-ParameterCo-Alignment), a framework that aligns
LLM reasoning with temporal knowledge at both the evidence
and parameter levels. At the evidence level, TEPCA retrieves
temporally grounded subgraphs from TKGs by jointly con-
sidering semantic relevance, temporal validity, and evolution-
ary structure. Specifically, it introducesDynamic Relevance-
Guided Path (DRGP) pruningto preserve globally coherent
temporal paths that reflect long-range event evolution, and
Mutual Information–based (MI-based) pruningto retain query-
informative local facts under temporal constraints. These two
complementary pruning strategies allow the model to access
evidence that is not only relevant to the query but also consistent
with its temporal context.
At the parameter level, TEPCA introduces Dynamic-Gated
LoRA (DG-LoRA), a timestamp-conditioned adaptation mech-
anism designed to overcome the static nature of conventional
low-rank fine-tuning. Instead of applying the same low-rank
update across all temporal contexts, DG-LoRA dynamically
modulates the effective adaptation subspace according to
temporal signals such as query time and validity windows.
This enables the model to allocate adaptation capacity in
an input- and time-dependent manner, thereby capturing fine-
grained semantic drift in evolving entities and relations. By
jointly aligning what evidence should be retrieved, when theevidence is valid, and how the LLM should adapt to time-
dependent semantics, TEPCA provides a principled framework
for accurate, interpretable, and temporally grounded LLM
reasoning over dynamic knowledge.
The main contributions of this paper are summarized as
follows:
•This paper proposes aTemporalEvidence-ParameterCo-
Alignment (TEPCA) framework, which effectively leverages
temporal dynamics within Temporal Knowledge Graphs
(TKGs) for enhancing Large Language Models (LLMs)
reasoning, while significantly improving interpretability;
•This paper proposes a novel temporal graph retrieval strategy
that combines Dynamic Relevance-Guided Path (DRGP)
pruning for global subgraph optimization with Mutual
Information (MI)-based filtering to extract query-relevant
details;
•This paper designs Dynamic-Gated LoRA, a time-aware
adaptation mechanism that dynamically adjusts the low-rank
capacity of LLMs based on temporal context, enabling fine-
grained reasoning over temporally evolving knowledge.
II. RELATEDWORK
A. Temporal Knowledge Graph Reasoning
Temporal Knowledge Graph Reasoning (TKGR) aims to
reason over temporal data within Temporal Knowledge Graphs
(TKGs) to predict unobserved relations or infer future events.
Existing TKGR methods can be broadly categorized into rule-
based and embedding–based approaches.
Rule-based TKGR methodsutilize logical rules and
temporal constraints to infer dynamic relations between entities,
thereby providing highly interpretable reasoning processes.
TLogic [25] extracts logical rules from TKGs using temporal
random walk methods for predicting future events. Building on
this, TILP [26] proposes a constrained random walk technique
and models different types of temporal features to optimize the
learning process of temporal logical rules. Further enhancing
rule discovery, TR-Rules [27] employs a lightweight algorithm
to reduce temporal redundancy and mine acyclic rules from
TKGs.
Other approaches focus on refining logical reasoning or
integrating broader patterns. For instance, LGevo [28] jointly
models local and global evolutionary patterns via dynamic
subgraph sampling and a temporal fusion encoder, which can
inform the formulation of temporal rules. Meanwhile, TFLEX
[29] achieves efficient reasoning on TKGs by combining fuzzy
logic with temporal logic operators. Conversely, TPRG [30]
infers new facts by analyzing the logical relationships of multi-
hop paths in TKGs and applying fourteen manually defined
temporal logical rules. In a different direction, LogiQ [31]
encodes temporal logic rules and applies rule-guided reward
shaping to steer reinforcement learning agents toward rule-
consistent actions, bridging logical reasoning with reinforce-
ment learning. TLmod [32] introduces a pruning strategy that
calculates confidence scores for rules, selecting only those
with high confidence for the reasoning on TKGs. While rule-
based TKGR methods are highly interpretable, they may lack
flexibility when dealing with complex or sparse data.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 3
Fig. 2: The framework of TEPCA. TEPCA fine-tunes LLMs via Dynamic-Gated LoRA (DG-LoRA), which injects time-adaptive
low-rank updates modulated by query-conditioned gating. Query Expansion then identifies temporally relevant event chains to
enhance contextual understanding. The Graph Retrieval module applies Dynamic Relevance-Guided Path (DRGP) Pruning and
Mutual Information (MI)–based Pruning to distill globally consistent paths and locally informative facts from TKGs. Global
paths are selected based on reverse chronological constraints (e.g., ti< t3andt3> t2> t1), semantic relevance, and temporal
decay, ensuring temporal consistency and causal validity. Finally, LLM Reasoning integrates the retrieved temporal subgraph
into the fine-tuned LLM to produce time-sensitive and context-aware predictions.
Embedding-based TKGR methods[33], [34] utilize em-
bedding techniques to learn high-dimensional representations
of entities and relations, capturing the semantic and temporal
features of TKGs, thereby enabling the prediction of future
events and the inference of unknown relations. RE-NET
[35] employs Recurrent Neural Networks (RNNs) and Graph
Neural Networks (GNNs) to model historical facts, thereby
enhancing the prediction of future events in TKGs. Building
on this, RE-GCN [36] predicts future events by leveraging a
combination of Relational Graph Convolutional Networks (R-
GCN) and Gated Recurrent Units (GRU) to encode historical
facts. TiRGN [37] simultaneously integrates local and global
encoders to comprehensively explore historical development
patterns, while HGLS [38] captures semantic and temporal
dependencies at both sub-graph and global graph levels.
Addressing the interpretability challenge, xERTE [39] emerges
as a pioneering explainable TKGR method, integrating causal-
aware relational attention, interpretable inference graphs, and
dynamic pruning techniques. CENET [40] further enhances
reasoning capabilities by integrating both historical and non-
historical event data to model robust entity distributions and
employing a contrastive learning algorithm to identify key
entities. While deep learning-based TKGRs are typically better
suited for handling large-scale and complex data, they often
lack straightforward interpretability.
B. Large Language Model Reasoning
Large Language Models (LLMs) reasoning on TKGs [9],
[41] typically involves leveraging TKGs to enhance the gener-ative capabilities of LLMs for reasoning tasks. Existing LLM
reasoning methods can be broadly classified into two categories:
prompt learning-based and fine-tuning-based reasoning.
Prompt Learning-based LLM Reasoningfocuses on
improving reasoning abilities using contextual information, cus-
tomized prompts, or zero-shot learning, without requiring fine-
tuning of the LLMs. PPT [42] introduces specialized prompts
tailored for different temporal interval types, enabling effective
extraction of timestamp semantics. HFL [43] adopts a token-
only input paradigm to model time intervals, integrating both
short-term and long-term historical contexts for robust temporal
reasoning. In-context learning (ICL) strategies are explored by
GPT-NeoX [44], which enhances temporal predictions without
fine-tuning by leveraging structured examples. Mixtral-8x7B-
CoH [45] applies a Chain-of-History (CoH) mechanism to
model higher-order historical dependencies in a progressive
manner. Similarly, zrLLM [46] investigates zero-shot relational
reasoning with LLMs for temporal fact completion.
To further improve temporal constraint modeling,
TimeR4[47] introduces a two-stage retrieve-then-reason
framework. It first extracts temporal constraints via a retrieve-
rewrite module and then identifies relevant evidence using a
retrieve-rerank module. Most recently, LLM-DA [5] proposes
a dynamic adaptation strategy that incrementally updates the
LLM based on recent events, enhancing generalization for
future event prediction.
Fine-tuning-based LLM Reasoningenhances the perfor-
mance on temporal reasoning tasks by fine-tuning LLMs.
Luoet al.[48] utilizes historical chains to fine-tune LLMs,

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 4
improving their ability to predict future events. GenTKG [49]
incorporates few-shot parameter-efficient instruction-tuning and
a temporal logical rule-based retrieval strategy to enhance the
reasoning ability and interpretability of LLMs. ChapTER [50] is
a prefix-tuning-based contrastive historical modeling approach
for TKGR that balances textual and temporal information
through virtual time prefixes and pseudo-Siamese encoders.
TKGIG [51] introduces an explainable temporal reasoning
task to predict future events, using the ExplainTemp dataset
generated via a KG-guided strategy.
Nonetheless, existing approaches remain largely opaque,
offering limited interpretability and making the reasoning
process difficult to trace. Additionally, most adopt uniform low-
rank adaptation schemes that fail to account for the temporal
dynamics of evolving entities. These shortcomings substantially
undermine the robustness and real-world applicability of LLMs
in producing temporally grounded and explainable outputs.
III. PRELIMINARY
a) Temporal Knowledge Graph (TKG):A TKG is defined
as a tuple G={E,R,T,Q} , where Erepresents the set
of entities, Rdenotes the set of relations, Tis the set of
timestamps, and Qcomprises the set of quadruplets. Each
quadruplet (es, r, eo, t)∈ Q consists of a subject entity es, an
object entitye o, a relationr, and a timestampt.
b) Graph Retrieval-Augmented Generation (GraphRAG):
GraphRAG [14] improves the reasoning capability of LLMs
by incorporating a graph retriever Mθ1(G∗|q,G) , which
retrieves temporally and contextually relevant KGs in response
to a query q. The retrieved TKGs are then processed by LLMs
Mθ2(A|q, G∗)to generate the answer:
M(A|q,G) =X
G∗⊆GMθ2(A|q, G∗)· M θ1(G∗|q,G),(1)
where Ais the generated answer, GandG∗denote the original
and retrieved TKGs, respectively.
c) Low-Rank Adaptation (LoRA):LoRA [22] is a
parameter-efficient fine-tuning method designed to adapt LLMs
without updating their original weights. Rather than updating
the full-rank weight matrices, LoRA introduces trainable low-
rank components into selected layers, substantially reducing
the number of trainable parameters while maintaining task
performance. LoRA can be formulated as:
y=xW+xW AWB, (2)
where x∈RS×d 1andy∈RS×d 2are the input and output
of the LLM, Sdenotes the token sequence length, and d1,d2
are the input and output channel dimensions. The pre-trained
weight W∈Rd1×d2is fixed (non-trainable), WA∈Rd1×ris
initialized via uniform Kaiming distribution [52], while WB∈
Rr×d2is zero-initialized. The rank rsatisfies r≪min(d 1, d2)
and is shared across samples.
IV. DYNAMIC-GATEDLORA FINE-TUNING
TKGs demand models that adapt to entities whose roles
and relations evolve over time. Conventional LoRA, which
applies the same low -rank updates uniformly, presumes staticsemantics and cannot capture this temporal–semantic variability.
To overcome these limitations, we propose DG -LoRA, a novel
fine-tuning framework that conditions low -rank adaptations on
TKGs. DG -LoRA incorporates a lightweight gating network
that dynamically adjusts both the magnitude and the rank
of LoRA updates according to the timestamp of each event.
This mechanism empowers the model to specialize its internal
representations for different time steps, markedly enhancing
its ability to reason over evolving knowledge.
DG-LoRA enhances standard LoRA by incorporating a
time-aware gating mechanism into the low-rank adaptation.
Specifically, we extend the input to include timestamps: for
each token representation xtokassociated with a timestamp
t, the combined input is x= [x tok;t]. The formulation of
DG-LoRA is given as follows:
y=xW+ 
xWA⊙
(xW C≥0)⊙tanh(xW C)
| {z }
gatec
WB,
(3)
where Wis the frozen pretrained weight matrix, and WA∈
Rd1×randWB∈Rr×d2parameterize the static low-rank
update. The learnable matrix WC∈Rd1×rgenerates the gate
features:
gatec= (xW C≥0)⊙tanh(xW C),(4)
where the hard threshold (xW C≥0) induces token-wise
sparsity and tanh(xW C)provides continuous scaling. By
conditioning the gate on both the token and timestamp, DG-
LoRA dynamically adjusts both the magnitude and the effective
rank of the adaptation, allowing the model to adapt its low-rank
subspace to different temporal contexts. This equips the model
with temporally grounded reasoning capabilities necessary for
handling evolving knowledge in TKGs. A detailed gradient
analysis is presented in Subsection VI-A , with the theoretical
justification for DG-LoRA provided in Subsection VI-B.
V. METHODOLOGY
In this section, we propose aKnowledgeGraph–Grounded
Retrieval andTemporalAdaptation (TEPCA) framework. As
displayed in Fig. 2, the core of our proposed method TEPCA
mainly contains three components:
•Query Expansionincorporates relevant historical event chains
to enrich the temporal context of the query;
•Graph Retrievalleverages Dynamic Relevance-Guided Path
Pruning and Mutual Information-based pruning algorithms
to effectively extract the most relevant global and local
information;
•LLM Reasoningintegrates the retrieved temporal knowledge
into the LLM to generate context-aware and time-sensitive
responses.
A. Query Expansion
Temporal queries are often underspecified, lacking the his-
torical context necessary for reliable inference. This contextual
sparsity can result in ambiguous or incomplete reasoning,
especially in dynamic environments where entities and relations
evolve over time. To address this issue, we introduce a query

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 5
expansion mechanism that enriches the input query with
temporally grounded events from the past. This augmentation
enables LLMs to access temporally relevant context, thereby
facilitating more accurate and context-aware reasoning.
Given the edge r(ex, ey, tl), query expansion explores the
temporal random walks [53] to retrieve multi-hop relational
paths that involve the entity exand occurred prior to timestamp
tl. To ensure both temporal relevance and causal consistency,
TEPCA constrains the random walk process by integrating
both temporal order and interval-based decay into a unified
transition probabilityP xy(t):
Pxy(t) =I[t < t l]·exp(−λ(t l−t))P
(ex,e′y,t′)∈GI[t′< tl]·exp(−λ(t l−t′)),(5)
where I[·]ensures only past events are considered, and
the exponential decay term exp(−λ(t l−t)) favors more
recent interactions relative to the query timestamp. The decay
coefficient λcontrols how strongly temporal distance influences
the transition likelihood.
Through query expansion, we obtain a set of historical events
that capture the temporal context and semantics relevant to the
query. Each retrieved historical fact in Pexpis first linearized
into a natural language sentence and subsequently encoded
by a Pretrained Language Model (PLM). The enhanced query
representations qis then derived via mean pooling:
sq=1
|Pexp||Pexp|X
i=1PLM(P exp[i]),(6)
where PLM(P exp[i])denotes the contextual embedding of the
i-th linearized event. This aggregation mechanism synthesizes
the semantic information from historical evidence, yielding
a query representation that is both temporally grounded and
contextually enriched.
B. Graph Retrieval
Graph retrieval extracts both global and local temporal
contexts from TKGs to support accurate reasoning. Due to the
evolving nature of entities and relations in TKGs, effective
subgraph extraction must account for both structural relevance
and temporal specificity. To this end, the graph retrieval pipeline
consists of two complementary pruning strategies:Dynamic
Relevance-Guided Path Pruningefficiently identifies multi-
hop paths that encode global temporal-relational patterns;
andMutual Information–Based Pruningleverages mutual
information to select the subset of quadruplets that best align
semantically and temporally with the query.
a) Dynamic Relevance-Guided Path Pruning:Dynamic
Relevance-Guided Path (DRGP) pruning aims to extract glob-
ally coherent temporal paths from the query-induced candidate
graph Gq. Given a temporal query q= (e s, r,?, t q), we first
restrict the candidate space to historical facts:
G−
q={g= (e u, rg, ev, tg)∈ G q|tg< tq}.(7)
This query-time constraint prevents post-query evidence from
being used during path construction. Starting from the head
entity es, DRGP expands candidate paths within G−
qby jointlyconsidering query relevance, path-context compatibility, and
temporal order.
Unlike fact-level pruning, DRGP treats a reasoning path
as an ordered temporal structure. For a candidate path R=
[g1, g2, . . . , g i], where gj= (e j, rj, ej+1, tj), we enforce the
reverse chronological constraint:
tq> t1> t2>···> t i,(8)
where this constraint encourages the retrieved path to trace
supporting evidence backward from the query timestamp to
earlier events.
At the j-th expansion step, let R<j= [g 1, . . . , g j−1]denote
the existing partial path. We encode the path context as:
hR<j= Pool 
eg1,eg2, . . . ,e gj−1
,(9)
where eglis the representation of the linearized quadruplet
gl. For the first hop, we set hR<1=sq, where sqdenotes the
expanded query representation.
To guide path expansion, we define a transition-aware
relevance score:
Ψ(gj|q, R <j) =β qcos(s q,egj)
+βpcos(h R<j,egj) +β rcos(r,r gj),(10)
where randrgjdenote the representations of the query relation
and the candidate relation, respectively. The coefficients βq,βp,
andβrbalance query relevance, path-context coherence, and
relation-level compatibility. Different from ordinary semantic
retrieval, this score evaluates whether a candidate fact is suitable
as the next hop of the current reasoning path, rather than only
measuring its isolated relevance to the query.
To model temporal proximity, we assign each candidate fact
a decay weight:
αj= exp (−λ 1(tq−tj)),(11)
where λ1>0controls the preference for recent historical facts.
The score of a partial path R= [g 1, g2, . . . , g i]is then defined
as:
SDRGP (R) =Pi
j=1αjΨ(gj|q, R <j)
Pi
j=1αj,(12)
where the normalization mitigates length bias and allows paths
with different lengths to be compared fairly.
After at most Lexpansion steps, DRGP selects the top- k
paths satisfying Eq. (8):
RTop-k = TopKR∈R allSDRGP (R),(13)
where Ralldenotes all candidate paths constructed from G−
q.
The optimized global subgraph is obtained by aggregating the
selected paths:
Gopt
q=[
R∈R Top-kR.(14)
DRGP therefore performs path-level pruning to preserve
query-relevant, path-coherent, and chronologically valid global
evidence. This global evidence complements the subsequent
MI-based pruning, which performs fact-level selection to retain
locally informative evidence from the same temporal candidate
space.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 6
b) Mutual Information-based Pruning:While DRGP
pruning preserves globally coherent temporal paths, path-level
selection may overlook local facts that are individually critical
to the query. To capture such fine-grained evidence, we further
prune the query-induced subgraph Gqby selecting quadruplets
with high query-specific information gain. Since the joint
distribution between natural-language queries and temporal
quadruplets is unavailable and difficult to estimate from sparse
TKG observations, directly computing the mutual information
I(q;g) is intractable. We therefore approximate it using
a temporal-aware contrastive pointwise mutual-information
estimator defined over the finite candidate setG q.
Given a query q= (e s, r,?, t q)and a candidate quadruplet
g= (e′
s, r′, e′
o, tg)∈ G q, we utilize a temperature-scaled cosine
similarity to compute the semantic compatibility between q
andg:
ϕ(q, g) =cos(s q,eg)
τ,(15)
where τ >0 controls the sharpness of the compatibility
distribution.
To incorporate temporal proximity, we introduce a time-
decay weight:
ω(q, g) = exp (−λ 2(tq−tg)),(16)
where λ2controls the strength of temporal decay. This design
favors more recent historical facts, while earlier facts can still
be retained if they exhibit strong semantic compatibility.
Combining semantic compatibility with temporal proximity,
we define a query-conditioned evidence distribution overG−
q:
pθ(g|q) =ω(q, g) exp(ϕ(q, g))P
g′∈G−
qω(q, g′) exp(ϕ(q, g′)),(17)
where it assigns larger probabilities to facts that are both
semantically compatible with the query and temporally close
to the query timestamp.
However, structurally central facts or high-frequency re-
lations can be over-selected even when they carry limited
query-specific information. To correct this background bias,
we introduce a query-independent background distribution:
p0(g) =ρ(g)P
g′∈G−
qρ(g′),(18)
where ρ(g) denotes the background salience of g, i.e., its
likelihood of being selected without conditioning on query
semantics. Forg= (e′
s, r′, e′
o, tg), we instantiateρ(g)as:
ρ(g) = gdeg(e′
s) +gdeg(e′
o) +ηgfreq(r′) +ϵ,(19)
wheregdeg(·) denotes the normalized entity degree in the can-
didate temporal subgraph, gfreq(r′)is the normalized frequency
of relation r′,ηbalances entity-level and relation-level salience,
andϵis a smoothing constant for numerical stability.
The local information score is then defined as the log-ratio
between the query-conditioned evidence distribution and the
background distribution:
SMI(q, g) = logpθ(g|q)
p0(g).(20)where SMI(q, g) provides a temporal-aware contrastive ap-
proximation of pointwise mutual information, quantifying the
information gain of a candidate fact after conditioning on the
query and its timestamp relative to its background salience.
For a query q, the ranking induced by SMI(q, g) is equivalent
to:
SMI(q, g)∝ϕ(q, g) + logω(q, g)−logρ(g).(21)
Thus, the proposed score selects facts that are semantically
aligned with the query, temporally close to the query timestamp,
and informative beyond what is expected from their background
salience.
Finally, we construct the local evidence set by selecting the
top-kquadruplets with the highest information scores:
Vlocal
q= Topk({SMI(q, g)|g∈ G q}).(22)
Through the dual-pruning strategy, MI-based pruning comple-
ments DRGP pruning: DRGP captures long-range temporal
structures through globally coherent paths, whereas MI-based
pruning retains fine-grained local evidence that directly supports
precise answer localization.
C. LLM Reasoning
LLM Reasoning empowers LLMs to perform temporally
grounded and semantically coherent reasoning by integrating
localized evidence and globally structured context extracted
from TKGs. It is composed of two key components:Graph
Encoding, which encodes the optimized global subgraph into
dense representations that preserve both temporal order and
relational dependencies; andAnswer Generation, where the
LLM leverages both the global subgraph embedding and the
pruned local facts to infer accurate answers aligned with the
query’s intent.
a) Graph Encoding:To incorporate global contextual
knowledge into LLMs, we first encode the optimized subgraph
Gopt
qusing a graph encoder [54]. The encoder captures both
relational and temporal dependencies among entities, generating
node-level embeddings that retain the subgraph’s structural and
temporal characteristics. The resulting node embeddings are
aggregated using mean pooling (M-POOL) to form a holistic
representation of the subgraph. This representation is then
projected into the LLM’s input space through a multilayer
perceptron (MLP), ensuring compatibility with the LLM’s
embedding space. Formally, this encoding process is defined
as:
hglobal
q=MLP ϕ2 
M-POOL 
GNN ϕ1(Gopt
q)
,(23)
where GNN ϕ1(·)denotes the graph encoder with parameters ϕ1,
andMLP ϕ2(·)maps the pooled vector to the LLM-compatible
space. The resulting embedding hglobal
q captures rich temporal
and relational semantics from the global subgraph.
b) Answer Generation:In the final step, the LLM
generates an answer Aqby jointly conditioning on the original
query q, the encoded global representation hglobal
q, and the set
of local quadrupletsVlocal
q.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 7
These components are embedded and formatted as part of
the input prompt, allowing the LLM to perform context-aware
reasoning. The whole process is defined as:
Mθ(Aq|q,G) =X
G∗∈SqMθ3(Aq|q,hglobal
q,Vlocal
q)
| {z }
LLM Reasoning
· Mθ2(hglobal
q,Vlocal
q|q, G∗)· M θ1(G∗|q,G)
| {z }
Graph Retrieval,
where Mθ1(G∗|q,G) models the distribution over subgraphs
G∗, given the query and full graph. Mθ2encodes local and
global structure from the subgraph G∗, producing represen-
tations suitable for downstream reasoning. Mθ3is an LLM-
based module that conditions on the original query, the local
quadruplets Vlocal
q, and the graph-encoded global representation
hglobal
q to generate the final answer.
c) Finetuning Objective.:To capture temporal dynamics
during fine-tuning, we integrate DG-LoRA into the LLM,
enhancing standard LoRA with a timestamp-conditioned gating
mechanism (see Eq. (3) in Section IV). For each token
representation xtokassociated with timestamp t, the input to
DG-LoRA is formed as x= [x tok;t], enabling the model to
adjust adaptation strength based on temporal context. The low-
rank update ∆W DGis computed via Eq. (3), with the gating
vector defined in Eq. (4). This mechanism allows the model
to adapt flexibly to varying temporal contexts.
The fine-tuning objective maximizes the log-likelihood of
the ground-truth answer Aq, conditioned on the query q, the
DRGP-derived global summary hglobal
q, and the MI-filtered local
subgraphVlocal
q:
L=E(q,Vlocalq,hglobal
q,Aq)∼D
logP θ+∆W DG 
Aqq,hglobal
q,Vlocal
q
,
where θdenotes the frozen pretrained parameters and
∆W DG(x, t) is the timestamp-conditioned low-rank adap-
tation. During training, only the DG-LoRA parameters
(WA, WB, WC)are updated, enabling temporally grounded
adaptation with minimal parameter overhead.
VI. THEORETICALANALYSIS OFDG-LORA
A. Optimization Dynamics of DG-LoRA
We first analyze DG-LoRA from the perspective of gradient
propagation, highlighting how the input-conditioned gate
changes the optimization dynamics of low-rank adaptation
parameters.
For standard LoRA in Eq. (2), the gradients of WAandWB
with respect to the loss functionLare
∂L
∂WA=x⊤∂L
∂yW⊤
B∈Rd1×r,(24)
∂L
∂WB= (xW A)⊤∂L
∂y∈Rr×d2.(25)For DG-LoRA in Eq. (3), where y=xW+ (xW A⊙
gatec)WBand gatec=G(xW C), the gradients are
∂L
∂WA=x⊤ ∂L
∂yW⊤
B
⊙gatec
,(26)
∂L
∂WB= 
xWA⊙gatec⊤
∂L
∂y
,(27)
∂L
∂WC=x⊤ ∂L
∂yW⊤
B
⊙xW A⊙1{xW C≥0}
⊙ 
1−gatec⊙gatec
.(28)
Here,∂L
∂y∈RS×d 2is the output gradient,∂L
∂WA,∂L
∂WC∈Rd1×r
are the gradients of the compression and gating matrices, and
∂L
∂WB∈Rr×d2is the gradient of the reconstruction matrix.
Compared with standard LoRA, DG-LoRA introduces two
optimization effects. First, gatecselectively scales the gradients
ofWAandWB, so only activated low-rank components receive
effective updates. This makes parameter optimization input-
dependent and sparse. Second, WCreceives gradients jointly
determined by the reconstruction path, compressed features, and
gate derivative, enabling it to learn which low-rank components
should be activated under different inputs.
B. Theoretical Justification for DG-LoRA
We further interpret DG-LoRA as input-conditional low-rank
adaptation. Given an input representation xcontaining temporal
information, DG-LoRA produces
y=xW+ ∆y,(29)
∆y= (xW A⊙G(xW C))WB,(30)
where Wis frozen, WAandWBare low-rank trainable
matrices, and WCis the gating matrix. The gating function is
G(z) =I(z≥0)⊙tanh(z).(31)
Input-conditional adaptation.Standard LoRA applies a
fixed update:
∆yLoRA =xW AWB.(32)
In contrast, DG-LoRA can be rewritten as
∆y=xW AD(x)W B, D(x) = Diag(G(xW C)),(33)
which induces an input-specific effective update:
∆W(x) =W AD(x)W B.(34)
Thus, DG-LoRA defines a family of input-conditional low-rank
updates rather than a single static adapter. When xincludes
temporal signals, the effective update becomes conditioned on
timestamps or validity intervals.
Adaptive effective rank.Let
A(x) ={i|G i(xW C)̸= 0}(35)
be the active set of low-rank components. Since D(x) is
diagonal and sparse,
rank(∆W(x)) = rank(W AD(x)W B)≤ |A(x)| ≤r.(36)
Therefore, DG-LoRA dynamically adjusts the effective rank
according to input complexity, allocating fewer components to
simple contexts and more capacity to temporally specific ones.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 8
Bounded sparse update.The indicator term I(z≥0)
suppresses inactive components, while tanh(z) bounds active
gates within (−1,1) . Thus, ∆W(x) is constrained by sparse
and bounded diagonal modulation, reducing uncontrolled low-
rank perturbations and improving robustness under uneven
temporal data density.
Overall, DG-LoRA improves standard LoRA by replacing
a globally shared low-rank update with an input-conditional
effective update and by controlling its effective rank through
sparse gating. For TKG reasoning, this enables different
temporal contexts to activate different adaptation subspaces,
thereby better modeling time-dependent entity semantics and
relation drift.
VII. EXPERIMENTS
In our experiments, we aim to answer the following re-
search questions:RQ1:Does TEPCA achieve state-of-the-art
performance on temporal knowledge graph reasoning tasks?
RQ2:How much does each component of TEPCA contribute to
temporal reasoning performance?RQ3:How does the rank of
DG-LoRA affect the balance between adaptation capacity and
generalization?RQ4:How do fine-tuning iterations affect the
convergence and stability of TEPCA compared with standard
LoRA?RQ5:Does DG-LoRA exhibit general input-adaptive
adaptation capacity across different LLM backbones and
commonsense reasoning datasets?RQ6:Can the dynamic
low-rank subspaces learned by DG-LoRA reveal interpretable
input-dependent adaptation patterns?
A. Datasets
Table I reports dataset statistics, including the number of
entities, relations, and temporal facts, as well as the granularity
of each dataset. These statistics characterize scale and temporal
resolution and support the evaluation of TEPCA across settings.
TABLE I: Statistical information of datasets.
Dataset Entities Relations #Train #Val #Test Granularity
ICEWS14 6,869 230 74,845 8,514 7,371 24 hours
GDELT 5,850 238 79,319 9,957 9,715 15 mins
Integrated Crisis Early Warning System (ICEWS) is a TKG
for international crisis early warning and analysis, which
collects and integrates data on political events, social dynamics,
and international relations worldwide. ICEWS14 [55] contains
events that occurred from 1/1/2014 to12/31/2014 . GDELT
[56], short for the Global Database of Events, Language, and
Tone, is a large-scale TKG that records political and social
events between geopolitical entities over time. Each record
contains a timestamp, interacting entities, event types, and
other contextual metadata. Due to its real-world coverage and
long temporal span, GDELT serves as a challenging benchmark
for temporal reasoning under high noise and data sparsity.
To further demonstrate DG-LoRA’s effectiveness, we evalu-
ate it on eight NLP question-answering benchmarks curated by
LLaMA-Adapter [57]: BoolQ [58], PIQA [59], SIQA [60], Hel-
laSwag [61], WinoGrande [62], ARC-e [63], ARC-c [63], andOBQA [64]. These benchmarks span reading comprehension
and physical, social, and scientific commonsense, providing a
diverse testbed for evaluating DG-LoRA’s dynamic low-rank
subspace gating.
B. Baselines
We compare TEPCA with representative baselines from
three categories: traditional TKGR methods, PLM-based TKGR
methods, and LLM-based TKGR methods. These baselines
cover embedding-based temporal modeling, rule-based tem-
poral reasoning, prompt-based temporal prediction, and fine-
tuning-based LLM reasoning, providing a comprehensive
comparison with both conventional and recent approaches.
Traditional TKGR methods.We include representative
embedding-based and rule-based TKGR methods, including
RE-GCN [36], xERTE [39], TLogic [25], TR-Rules [27],
LGevo [28], and LogiQ [31]. These methods model tem-
poral relational patterns through recurrent graph encoding,
explainable temporal subgraph reasoning, temporal rule mining,
evolutionary pattern modeling, or rule-guided reinforcement
learning.
PLM-based TKGR methods.We further compare with PLM-
based temporal reasoning methods, including ChapTER [50],
PPT [42], and HFL [43]. These methods formulate TKGR
with pre-trained language models by incorporating timestamp
semantics, temporal prompts, or historical fact modeling.
LLM-based TKGR methods.We also evaluate against
recent LLM-based TKGR methods, including InternLM2-
ICL, Llama-2-7B-ICL, and GPT-NeoX-ICL [44]; Llama-2-
7B-CoH and Vicuna-7B-CoH [48]; Mixtral-8x7B-CoH [45];
GPT-NeoX-20B-GenTKG and Llama-2-7B-GenTKG [49]; and
GPT3.5-turbo-LLM-DA [5]. These baselines represent in-
context learning, chain-of-history reasoning, temporal logical
rule-enhanced generation, and dynamic LLM adaptation for
temporal prediction.
C. Evaluation and Implementation
Evaluation Metrics.We follow the standard filtered tempo-
ral link prediction protocol and report Hit@N ( N= 1,3,10 )
as the main evaluation metrics. For each query, all candidate
entities are ranked according to the model prediction, and other
known valid facts appearing in the train, validation, or test
split are filtered out before computing the final rank. Hit@N
measures whether the ground-truth entity appears among the
top-Nranked candidates, thereby reflecting the model’s ability
to identify temporally valid missing entities under a given
query timestamp.
Entity Matching Strategy.Since LLM-based TKGR meth-
ods generate textual answers rather than entity IDs, we map
generated outputs to candidate entities before computing
link prediction metrics. For the main comparison, we adopt
Longest Match, which deterministically selects the longest
candidate entity string appearing in the generated output. This
protocol ensures reproducible entity-level evaluation and avoids
introducing an additional semantic encoder into the primary
metric. Cosine Similarity Matching and Open Generation
Matching are used only for auxiliary analysis: the former

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 9
TABLE II: Link prediction results on ICEWS14 and GDELT. Bold values indicate the best results; “–” denotes unavailable data.
Method Type MethodsICEWS14 GDELT
Hit@1 Hit@3 Hit@10 Hit@1 Hit@3 Hit@10
Embed/Rule-basedxERTE [39] 0.330 0.454 0.570 0.085 0.159 0.265
RE-GCN [36] 0.313 0.473 0.626 0.084 0.171 0.299
TLogic [25] 0.332 0.476 0.602 0.113 0.212 0.351
TR-Rules [27] 0.339 0.485 0.611 – – –
RE-H2AN [65] 0.312 0.458 0.603 0.114 0.199 0.331
LGevo [28] 0.367 0.557 0.728 0.129 0.219 0.361
LogiQ [31] 0.357 0.510 0.642 – – –
LLMs-basedPPT (BERT-cased) [39] 0.273 0.408 0.563 – – –
ChapTER (BERT-uncased) [50] 0.242 0.380 0.527 – – –
HFL (BERT-uncased) [43] 0.277 0.427 0.573 – – –
ICL (InternLM2) [44] 0.301 0.432 0.560 0.084 0.152 0.254
ICL (Llama-2-7B) [44] 0.252 0.427 0.504 0.060 0.164 0.246
ICL (GPT-NeoX-20B) [44] 0.334 0.460 0.565 – – –
CoH (Llama-2-7B) [48] 0.349 0.470 0.591 – – –
CoH (Vicuna-7B) [48] 0.328 0.457 0.656 – – –
CoH (Mixtral-8x7B) [45] 0.331 0.496 0.649 – – –
LLM-DA (GPT-3.5-turbo) [5] 0.356 0.515 0.662 0.121 0.221 0.331
GenTKG (GPT-NeoX-20B) [49] 0.350 0.474 0.575 0.102 0.167 0.273
GenTKG (Llama-2-7B) [49] 0.361 0.472 0.535 0.139 0.225 0.304
TEPCA (Llama-2-7B) 0.371 0.559 0.768 0.143 0.239 0.363
evaluates robustness to lexical variants and paraphrases but
may over-credit semantically similar incorrect entities, whereas
the latter provides a stricter surface-form matching protocol.
Backbone LLM.Unless otherwise specified, TEPCA uses
Llama-2-7B as the backbone LLM for the main TKGR
experiments. We choose Llama-2-7B for two reasons. First,
it is an open-source backbone that has been widely adopted
in prior LLM-based TKGR studies, enabling controlled com-
parison under comparable model capacity. Second, our goal
is not to rely on increasingly larger LLMs, but to examine
whether temporally aligned retrieval and timestamp-conditioned
adaptation can compensate for the static-memory limitation of
LLMs. Additional experiments on multiple LLM backbones
are conducted to evaluate the generality of DG-LoRA beyond
a single model architecture.
Retrieval and Pruning Settings.For Query Expansion and
DRGP pruning, the temporal decay coefficient is set to λ= 0.1
by default. The maximum retrieval threshold is set tok= 20
on both ICEWS14 and GDELT, which provides a balance
between evidence coverage and prompt length. For MI-based
pruning, the temperature parameter in semantic compatibility
is set to τ= 0.1 , and the temporal decay coefficient is set
toλ2= 0.1 unless otherwise specified. In the background
salience term ρ(g), we set η= 0.5 to balance entity-level
structural salience and relation-level frequency, and use ϵ=
10−6for numerical stability. The normalized entity degree and
normalized relation frequency are computed within the query-
induced temporal candidate subgraph G−
q. All hyperparameters
are selected according to validation performance.
DG-LoRA Configuration.DG-LoRA is applied to the
attention projection layers of the backbone LLM. The rank of
DG-LoRA is selected from {4,8,16,32} on the validation set.
Accordingly, we set the DG-LoRA rank to 8 on ICEWS14
and 32 on GDELT, balancing temporal adaptability and model
complexity. During fine-tuning, the original LLM parametersare frozen, and only the DG-LoRA parameters are updated.
The maximum number of fine-tuning epochs is set to 4, and the
final checkpoint is selected according to validation performance
to avoid overfitting to temporally sparse intervals.
All experiments are implemented in PyTorch and conducted
on four NVIDIA A100 GPUs with an Intel Xeon Gold 6326
CPU.
D. RQ1: TKGR Performance Comparison
The link prediction results for ICEWS14 and GDELT are
presented in Table II. The experimental analyses are listed as
follows:
(1) Experimental results indicate that the proposed TEPCA
achieves state-of-the-art performance across all metrics on both
ICEWS14 and GDELT, demonstrating its robustness in TKGR.
This improvement is attributed to two key design choices: (i) the
dynamic relevance-guided path pruning combined with mutual
information–based pruning filters out irrelevant facts to retain
only time-aligned, semantically rich paths from the TKGs;
and (ii) the Dynamic-Gated LoRA (DG-LoRA) fine-tuning
adapts the low-rank parameters of LLMs to query-specific
temporal and semantic contexts. Together, these components
enable TEPCA to better contextualize queries with temporally
precise knowledge, especially in time-sensitive or multi-hop
reasoning scenarios.
(2) TEPCA demonstrates clear advantages over existing
paradigms. Embedding-based TKGRs, constrained by static
representations, struggle to capture the evolving temporal
dynamics of entities and relations. Rule-based approaches
provide interpretability but suffer from limited flexibility and
generalizability, especially in sparse or temporally ambiguous
contexts. While LLM-based methods leverage powerful pre-
trained knowledge, they often rely on temporally misaligned
evidence and uniform fine-tuning strategies that overlook
the dynamic temporal roles of entities. In contrast, TEPCA

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 10
TABLE III: Ablation study on ICEWS14. The best results are in bold. “(-)” denotes replacing DG-LoRA with original LoRA.
“w/o” indicates “without”.
MethodsICEWS14 GDELT
Hit@1 Hit@3 Hit@10 Hit@1 Hit@3 Hit@10
TEPCA(-)LoRA 0.347 0.524 0.748 0.132 0.220 0.342
TEPCAw/oQE 0.359 0.550 0.759 0.135 0.228 0.347
TEPCAw/oDRGP 0.339 0.519 0.727 0.124 0.213 0.332
TEPCAw/oMI 0.318 0.541 0.751 0.118 0.219 0.344
TEPCA0.371 0.559 0.768 0.143 0.239 0.363
Fig. 3: Impact of DG-LoRA rank on both datasets.
integrates time-aligned retrieval with query-adaptive fine-tuning,
enabling more precise and context-aware reasoning over TKGs.
(3) GDELT is a more challenging dataset for temporal
link prediction than ICEWS14, with performance metrics for
all methods being substantially lower. This disparity stems
from its larger scale, higher sparsity, and complex event types
and temporal dynamics, requiring robust and generalizable
reasoning capabilities. TEPCA’s state-of-the-art performance on
GDELT, supported by comprehensive evaluation, demonstrates
its robust and generalizable effectiveness in handling complex
TKGs, thus addressing a gap in the literature.
(4) LLM-based methods generally outperform traditional
TKGR methods. For example, while LGeV o attains a Hit@1
of 0.367 on ICEWS14, several LLM-based methods (e.g.,
LLM-DA and TEPCA) achieve comparable or higher scores.
This indicates LLMs have an inherent advantage in lever-
aging their rich pre-trained knowledge and reasoning for
TKG tasks. Nevertheless, LGeV o remains highly competitive,
even surpassing some LLM-based methods (e.g., ICL and
GenTKG on ICEWS14 Hit@1), suggesting well-engineered
traditional methods are strong baselines and that substantial
gains result from effectively integrating LLMs with TKG-
specific mechanisms, as exemplified by TEPCA.
E. RQ2: Component Effectiveness
The ablation study on ICEWS14 and GDELT aims to
evaluate the impact of different modules on temporal reasoning
performance. We compare four variants of TEPCA by individu-
ally removing key components: the Dynamic Relevance-Guided
Path (DRGP) pruning, the Mutual Information (MI)-based
pruning, the Query Expansion (QE) module, and the Dynamic-Gated LoRA (DG-LoRA). The experimental results are shown
in Table III.
Impact of DRGP Pruning.Removing the DRGP module
(TEPCA w/o DRGP) leads to a noticeable decline in Hit@10 on
both datasets, confirming its importance in capturing temporally
consistent multi-hop reasoning paths that support LLM-based
inference. While the drop in Hit@1 is modest, the larger
decrease in Hit@3 and Hit@10 suggests that DRGP enhances
the ranking quality of candidate answers by modeling global
temporal structures rather than merely refining local evidence.
Impact of MI-based Pruning.The absence of MI-based
pruning (TEPCA w/o MI) leads to the lowest Hit@1 across all
variants, despite outperforming the DRGP ablation on Hit@3
and Hit@10. This observation indicates that the MI module
plays a pivotal role in selecting highly informative and query-
specific evidence, which is particularly crucial for accurate
top-1 predictions. Without this mechanism, the retained local
subgraph is likely to include semantically weaker or noisy
facts, thereby undermining precise answer localization even
when the global context remains intact.
Impact of Query Expansion.Removing the query expansion
module (TEPCA w/o QE) results in consistent performance
drops across all metrics, especially on GDELT. This underscores
the importance of temporal context enrichment in disambiguat-
ing entity roles and enhancing both global path construction
and local evidence selection.
Impact of DG-LoRA.Replacing DG-LoRA with the standard
LoRA configuration (TEPCA (-) LoRA) results in consistent
performance deterioration, validating the necessity of temporal-
adaptive fine-tuning. This confirms that entity semantics evolve
over time, and fixed-rank adaptations are inadequate for
capturing such temporal variability. DG-LoRA dynamically
adjusts the low-rank update capacity based on timestamp-aware
signals, enabling the model to better specialize in temporally
diverse entity roles and relation patterns.
These ablation results collectively demonstrate the comple-
mentary contributions of the proposed components. DRGP
provides global temporal structure, MI-based pruning ensures
fine-grained relevance, query expansion enriches the contextual
scope, and DG-LoRA enhances temporal adaptability.
F . RQ3: Rank Sensitivity
To examine the impact of the DG-LoRA rank on reasoning
performance, we vary the rank parameter r∈ {4,8,16,32}
and evaluate Hit@10 on ICEWS14 and GDELT. As shown in
Figure 3, increasing the rank from 4 to 8 leads to substantial
gains on both datasets. Specifically, Hit@10 on ICEWS14 rises

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 11
Fig. 4: Comparison of varying numbers of fine-tuning on ICEWS14 dataset within Hit@10 under different matching strategies.
The left subplot is “TEPCA w/ DG-LoRA” model, and the right subplot is “TEPCA w/ LoRA” model.
TABLE IV: Experimental results on eight commonsense reasoning datasets using various LoRA methods for Llama3.2-1B/3B,
Llama2-7B, Qwen2.5-7B, and Llama3-8B. Notably, the rank value is 32 ( r= 32 ) andbestindicates the highest average metric
value among all methods.
LLMs Methods BoolQ PIQA SIQA HellaSwag WinoGrande ARC-e ARC-c OBQAAvg.
ChatGPT - 73.1 85.4 68.5 78.5 66.1 89.8 79.9 74.8 77.0
Llama3.2-1BLoRA 63.9 75.7 72.0 64.7 67.4 71.0 53.6 67.2 66.9
DG-LoRA (Ours) 63.7 75.9 72.9 74.1 68.4 71.1 54.9 68.2 68.7
Llama3.2-3BLoRA 71.5 83.2 75.4 86.3 79.2 82.5 70.1 79.6 78.5
DG-LoRA (Ours) 71.5 84.5 78.8 92.3 81.2 83.8 72.4 80.0 80.5
Llama2-7BLoRA 69.8 79.9 79.5 83.6 82.6 79.8 64.7 81.0 77.6
DG-LoRA (Ours) 71.9 80.4 80.7 90.6 84.4 79.6 68.9 80.8 79.7
Qwen2.5-7BLoRA 68.4 88.8 80.3 94.6 84.8 95.3 88.6 89.6 86.3
DG-LoRA (Ours) 71.3 88.2 79.9 93.7 83.0 96.2 90.2 89.6 86.5
Llama3-8BLoRA 72.2 87.4 79.8 94.6 84.9 88.6 77.6 86.0 83.9
DG-LoRA (Ours) 75.7 89.1 81.7 95.8 85.6 91.5 81.1 86.4 85.8
from 0.733 to 0.768, while GDELT improves from 0.353 to
0.359. This demonstrates that increasing the rank enhances
the adaptation capacity, allowing it to better capture temporal
variations in entity behavior.
However, further increasing the rank to 16 results in a
notable performance drop on ICEWS14, while GDELT exhibits
only marginal improvement. This suggests that although
higher ranks provide greater expressiveness, they may also
introduce overfitting, particularly in temporally dense datasets
like ICEWS14. Interestingly, setting the rank to 32 partially
restores performance on ICEWS14 and yields continued but
minor improvement on GDELT. These results indicate that
excessively high ranks offer diminishing returns and may even
degrade performance due to overparameterization.
These observations suggest that moderate DG-LoRA ranks
strike the best balance between expressiveness and generaliza-
tion.
G. RQ4: Fine-tuning Dynamics
Figure 4 illustrates the temporal reasoning performance
(Hit@10) on the ICEWS14 dataset across varying fine-tuningiterations. We compare our proposed “TEPCA w/ DG-LoRA”
(Figure 4, left) against the standard “TEPCA w/ LoRA” (Fig-
ure 4, right) using three evaluation protocols:Cosine Similarity
Matching,Longest Match, andOpen Generation Matching.
Effectiveness of Matching Strategies. Cosine Similarity
Matchingconsistently outperformsLongest MatchandOpen
Generation Matchingacross all experimental settings. This
superiority suggests that embedding-based evaluation better
captures the semantic proximity between predicted entities and
ground truths, avoiding the limitations of rigid exact string
matching. Conversely, the consistently lower scores forOpen
Generation Matchinghighlight the strictness of standard TKGR
metrics, where minor lexical variations often result in false
negatives.
Convergence and Stability of DG-LoRA.Analyzing the
training dynamics ofDG-LoRA(Figure 4, left) reveals rapid
convergence. The model achieves near-optimal performance at
the first iteration and maintains stability through the third. This
indicates that the time-aware gating mechanism facilitates swift
alignment with temporal patterns without requiring extensive
weight updates. However, a sharp performance degradation

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 12
4 8 12 16 20 24 28
Layers4
8
12
16Token S15 21217111917
181010 471216
11 610 871114
11 81011111217Query (BoolQ)
4 8 12 16 20 24 28
Layers4
8
12
16Token S24 58411 910
15 68310 715
52928410
20 2611 91420Query (ARC)
4 8 12 16 20 24 28
Layers4
8
12
16Token S23 678111120
72797213
22 71113 7814
14 491310 813Query (OBQA)
4 8 12 16 20 24 28
Layers4
8
12
16Token S73814121718
15 384814 8
4810 610 816
612 710111126Value (BoolQ)
4 8 12 16 20 24 28
Layers4
8
12
16Token S15 351710 6
19786810
1101210 81110
15 636191610Value (ARC)
4 8 12 16 20 24 28
Layers4
8
12
16Token S1613 2591118
31112 76413
1615131411 49
514 41110 813Value (OBQA)
Fig. 5: The low-rank subspace heatmap (the number of gatec̸= 0) of the Query ( qproj) and Value ( vproj) mappings for ARC,
BoolQ, and OBQA question texts, based on the LLaMA-3.2-3B models (r= 32).
occurs at the 4-th iteration. This suggests that prolonged fine-
tuning induces overfitting, where dynamic low-rank adapters
memorize training noise at the expense of generalizability to
unseen timestamps.
Superiority over Standard LoRA.A comparison of the
two subplots underscores the structural advantage of our
approach. The baselineLoRA(Figure 4, right) exhibits a
slower, more volatile convergence trajectory, peaking only
at the 2-nd iteration with a lower maximum score. Unlike
the significant oscillations observed in standard LoRA,DG-
LoRAnot only converges faster but also sustains a higher
performance plateau. This confirms that the dynamic gating
mechanism effectively regulates adaptation, preventing the
“temporal specificity collapse” common in static adapters and
ensuring more reliable temporal reasoning.
H. RQ5: Generality across LLM Backbones
To further illustrate the effectiveness of DG-LoRA, we
conduct further rich experiments on the 8 question-answering
task of NLP field. As shown in Table IV, DG-LoRA consistently
outperforms standard LoRA across all evaluated LLMs and
most commonsense reasoning datasets. Notable gains are ob-
served on LLaMA2-7B, where DG-LoRA achieves an average
accuracy of 79.7%, surpassing the 77.6% obtained by standard
LoRA. Significant improvements are also observed on specific
tasks, such as BoolQ (+2.1), HellaSwag (+7.0), and ARC-c
(+4.2). In addition, DG-LoRA yields consistent improvementsover standard LoRA on LLaMA3.2-1B (+1.8), LLaMA3.2-
3B (+2.0), and LLaMA3-8B (+1.9). These improvements can
be attributed to DG-LoRA’s dynamic adaptation mechanism,
which employs a learnable gating function gatecconditioned
on temporal and contextual signals.
Unlike standard LoRA that uses static low-rank updates
shared across all inputs, DG-LoRA introduces a token-wise
sparse gating controller (Eq. 3–4) to modulate the contribution
of low-rank components dynamically. This allows the model
to specialize its reasoning for different contexts and semantic
patterns, thereby enhancing generalization and representation
flexibility. By adaptively discovering input-dependent low-rank
subspaces, DG-LoRA mitigates the over-sharing limitation of
traditional LoRA and demonstrates superior performance across
diverse and challenging commonsense reasoning tasks.
I. RQ6: Interpretability of Dynamic Subspaces
The visualization of Dynamic-Gated Low Rank (DG-LoRA)
effective low-rank subspaces reveals dynamic adjustments
across both layers and tokens, providing empirical validation
for the adaptive mapping mechanisms. As illustrated in Figure
5, the dimensions of token-specific subspaces demonstrate
substantial variation across different layers, confirming the filter
controller’s capacity to identify optimal low-rank subspaces
during the compression process. Furthermore, the low-rank
subspace dimension distribution across the three datasets
(BoolQ, ARC, and OBQA) exhibits significant diversity at both

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 13
token and layer levels. This observed diversity substantiates our
core argument that DG-LoRA can adaptively learn the dynamic
low rank subspace based on the different input features in
diverse dataset fields.
VIII. CONCLUSION
In this paper, we propose a novelTemporalEvidence-
ParameterCo-Alignment (TEPCA) framework that integrates
Temporal Knowledge Graphs (TKGs) with Large Language
Models (LLMs) to enhance both temporal reasoning and
interpretability. We introduce two complementary pruning
strategies: a time-constraineddynamic relevance-guided path
pruningthat captures global semantic and temporal struc-
ture, andmutual information-based pruningthat extracts
query-specific, temporally aligned subgraphs. Additionally,
we propose Dynamic-Gated LoRA (DG-LoRA), a time-aware
parameter-efficient tuning mechanism that adaptively modulates
low-rank adaptation based on temporal context, enabling LLMs
to better capture entity evolution and time-sensitive semantics.
One limitation is that the timestamp-specific gating within
DG-LoRA may overfit during periods with sparse data, leading
to the memorization of noise from low-sample intervals. A
promising avenue for future work could involve exploring
temporal data augmentation, multi-scale regularization, and
mixture-of-experts architectures to improve robustness across
both sparse and dense time periods.
REFERENCES
[1]Y . Chang, X. Wang, J. Wang, Y . Wu, L. Yang, K. Zhu, H. Chen, X. Yi,
C. Wang, Y . Wanget al., “A survey on evaluation of large language
models,”ACM Transactions on Intelligent Systems and Technology,
vol. 15, no. 3, pp. 1–45, 2024.
[2]W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y . Hou, Y . Min, B. Zhang,
J. Zhang, Z. Donget al., “A survey of large language models,”arXiv
preprint arXiv:2303.18223, 2023.
[3]C. ChenghaoZhu, N. Chen, Y . Gao, Y . Zhang, P. Tiwari, and B. Wang,
“Is your llm outdated? a deep look at temporal generalization,” in
Proceedings of the 2025 Conference of the Nations of the Americas
Chapter of the Association for Computational Linguistics: Human
Language Technologies, 2025, pp. 7433–7457.
[4]C. Wu, J. Wang, M. Gao, X. Zhuo, J. Guo, R. Lei, H. Luo, T. Chen,
H. Zhou, S. Panet al., “Cdtp: A large-scale chinese data-text pair
dataset for comprehensive evaluation of chinese llms,”arXiv preprint
arXiv:2510.06039, 2025.
[5]J. Wang, S. Kai, L. Luo, W. Wei, Y . Hu, A. W.-C. Liew, S. Pan, and
B. Yin, “Large language models-guided dynamic adaptation for temporal
knowledge graph reasoning,”Advances in Neural Information Processing
Systems, vol. 37, pp. 8384–8410, 2024.
[6]J. Ouyang, T. Pan, M. Cheng, R. Yan, Y . Luo, J. Lin, and Q. Liu, “Hoh:
A dynamic benchmark for evaluating the impact of outdated information
on retrieval-augmented generation,”arXiv preprint arXiv:2503.04800,
2025.
[7]A. Bajpai, A. Goyal, A. Anwer, and T. Chakraborty, “Temporally
consistent factuality probing for large language models,”arXiv preprint
arXiv:2409.14065, 2024.
[8]Y . Hu, Z. Lei, Z. Zhang, B. Pan, C. Ling, and L. Zhao, “Grag: Graph
retrieval-augmented generation,”arXiv preprint arXiv:2405.16506, 2024.
[9]J. Wang, B. Wang, M. Qiu, S. Pan, B. Xiong, H. Liu, L. Luo, T. Liu,
Y . Hu, B. Yinet al., “A survey on temporal knowledge graph completion:
Taxonomy, progress, and prospects,”arXiv preprint arXiv:2308.02457,
2023.
[10] K. Liang, L. Meng, M. Liu, Y . Liu, W. Tu, S. Wang, S. Zhou, X. Liu,
F. Sun, and K. He, “A survey of knowledge graph reasoning on graph
types: Static, dynamic, and multi-modal,”IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 46, no. 12, p. 9456–9478, 2024.[11] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2023.
[12] Q. Zhang, S. Chen, Y . Bei, Z. Yuan, H. Zhou, Z. Hong, J. Dong,
H. Chen, Y . Chang, and X. Huang, “A survey of graph retrieval-
augmented generation for customized large language models,”arXiv
preprint arXiv:2501.13958, 2025.
[13] A. Gade, J. G. Jetcheva, and H. Trivedi, “It’s about time: Incorporating
temporality in retrieval augmented language models,” in2025 IEEE
Conference on Artificial Intelligence, 2025, pp. 75–82.
[14] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and
S. Tang, “Graph retrieval-augmented generation: A survey,”arXiv preprint
arXiv:2408.08921, 2024.
[15] X. He, Y . Tian, Y . Sun, N. V . Chawla, T. Laurent, Y . LeCun, X. Bres-
son, and B. Hooi, “G-retriever: Retrieval-augmented generation for
textual graph understanding and question answering,”arXiv preprint
arXiv:2402.07630, 2024.
[16] J. Lin, S. Wang, X. Guo, J. Shun, and Y . Zhu, “Temporal reasoning with
large language models augmented by evolving knowledge graphs,”arXiv
preprint arXiv:2509.15464, 2025.
[17] L. Chen, P. Tong, Z. Jin, Y . Sun, J. Ye, and H. Xiong, “Plan-on-graph:
Self-correcting adaptive planning of large language model on knowledge
graphs,”Advances in Neural Information Processing Systems, vol. 37,
pp. 37 665–37 691, 2024.
[18] M. Jin, G. Shi, Y .-F. Li, B. Xiong, T. Zhou, F. D. Salim, L. Zhao, L. Wu,
Q. Wen, and S. Pan, “Towards expressive spectral-temporal graph neural
networks for time series forecasting,”IEEE Transactions on Pattern
Analysis and Machine Intelligence, vol. 47, no. 6, p. 4926–4939, 2025.
[19] K. Guo, H. Shomer, S. Zeng, H. Han, Y . Wang, and J. Tang, “Empowering
graphrag with knowledge filtering and integration,”arXiv preprint
arXiv:2503.13804, 2025.
[20] B. Dhingra, J. R. Cole, J. M. Eisenschlos, D. Gillick, J. Eisenstein, and
W. W. Cohen, “Time-aware language models as temporal knowledge
bases,”Transactions of the Association for Computational Linguistics,
vol. 10, pp. 257–273, 2022.
[21] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen,
W. Peng, X. Feng, B. Qinet al., “A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions,”
ACM Transactions on Information Systems, vol. 43, no. 2, pp. 1–55,
2025.
[22] E. J. Hu, Y . Shen, P. Wallis, Z. Allen-Zhu, Y . Li, S. Wang, L. Wang,
W. Chenet al., “Lora: Low-rank adaptation of large language models.”
ICLR, vol. 1, no. 2, p. 3, 2022.
[23] S.-y. Liu, C.-Y . Wang, H. Yin, P. Molchanov, Y .-C. F. Wang, K.-T. Cheng,
and M.-H. Chen, “Dora: Weight-decomposed low-rank adaptation,” in
Forty-first International Conference on Machine Learning, 2024.
[24] A. Taheri, K. Gimpel, and T. Berger-Wolf, “Learning to represent the
evolution of dynamic graphs with recurrent models,” inCompanion
proceedings of the 2019 world wide web conference, 2019, pp. 301–307.
[25] Y . Liu, Y . Ma, M. Hildebrandt, M. Joblin, and V . Tresp, “Tlogic: Temporal
logical rules for explainable link forecasting on temporal knowledge
graphs,” inProceedings of the AAAI conference on artificial intelligence,
2022, pp. 4120–4127.
[26] S. Xiong, Y . Yang, F. Fekri, and J. C. Kerce, “TILP: Differentiable
learning of temporal logical rules on knowledge graphs,” inInternational
Conference on Learning Representations, 2023, pp. 1–11.
[27] N. Li, E. Haihong, S. Li, M. Sun, T. Yao, M. Song, Y . Wang, and H. Luo,
“Tr-rules: Rule-based model for link forecasting on temporal knowledge
graph considering temporal redundancy,” inFindings of the Association
for Computational Linguistics: EMNLP, 2023, pp. 7885–7894.
[28] X. Luo, X. Wang, X. Wang, and R. Liu, “Temporal knowledge graph
reasoning with local-global evolutionary patterns,”Pattern Recognition,
p. 112643, 2025.
[29] X. Lin, C. Xu, F. Su, G. Zhou, T. Hu, N. Li, M. Sun, H. Luoet al.,
“TFLEX: Temporal feature-logic embedding framework for complex
reasoning over temporal knowledge graph,” inAdvances in Neural
Information Processing Systems, 2023, pp. 73 039–73 081.
[30] L. Bai, M. Chen, L. Zhu, and X. Meng, “Multi-hop temporal knowledge
graph reasoning with temporal path rules guidance,”Expert Systems with
Applications, vol. 223, p. 119804, 2023.
[31] T. Chen, L. Yang, Z. Wang, S. Luo, and J. Long, “Enhancing extrapolation
reasoning on temporal knowledge graphs with logic rules and queries,”
inIEEE International Conference on Acoustics, Speech and Signal
Processing, 2025, pp. 1–5.
[32] L. Bai, W. Yu, D. Chai, W. Zhao, and M. Chen, “Temporal knowledge
graphs reasoning with iterative guidance by temporal logical rules,”
Information Sciences, vol. 621, pp. 22–35, 2023.

JOURNAL OF L ATEX CLASS FILES, VOL. 14, NO. 8, AUGUST 2021 14
[33] S. Deng, H. Rangwala, and Y . Ning, “Dynamic knowledge graph based
multi-event forecasting,” inthe ACM SIGKDD Conference on Knowledge
Discovery and Data Mining, 2020, pp. 1585–1595.
[34] J. Wang, B. Wang, J. Gao, X. Li, Y . Hu, and B. Yin, “QDN: A quadruplet
distributor network for temporal knowledge graph completion,”IEEE
Transactions on Neural Networks and Learning Systems, 2023.
[35] W. Jin, M. Qu, X. Jin, and X. Ren, “Recurrent Event Network:
Autoregressive structure inference over temporal knowledge graphs,” in
the Conference on Empirical Methods in Natural Language Processing,
2020, pp. 6669–6683.
[36] Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y . Wang, and X. Cheng,
“Temporal knowledge graph reasoning based on evolutional representation
learning,” inInternational ACM SIGIR Conference on Research and
Development in Information Retrieval, 2021, pp. 408–417.
[37] Y . Li, S. Sun, and J. Zhao, “TiRGN: Time-guided recurrent graph
network with local-global historical patterns for temporal knowledge
graph reasoning.” inthe International Joint Conferences on Artificial
Intelligence, 2022, pp. 2152–2158.
[38] M. Zhang, Y . Xia, Q. Liu, S. Wu, and L. Wang, “Learning long- and
short-term representations for temporal knowledge graph reasoning,” in
Proceedings of the ACM on Web Conference, 2023, pp. 2412–2422.
[39] Z. Han, P. Chen, Y . Ma, and V . Tresp, “Explainable subgraph reasoning for
forecasting on temporal knowledge graphs,” inInternational Conference
on Learning Representations, 2021, pp. 1–12.
[40] Y . Xu, J. Ou, H. Xu, and L. Fu, “Temporal knowledge graph reasoning
with historical contrastive learning,” inProceedings of the AAAI
Conference on Artificial Intelligence, vol. 37, no. 4, 2023, pp. 4765–4773.
[41] Z. Chu, J. Chen, Q. Chen, W. Yu, H. Wang, M. Liu, and B. Qin,
“Timebench: A comprehensive evaluation of temporal reasoning abilities
in large language models,”arXiv preprint arXiv:2311.17667, 2023.
[42] W. Xu, B. Liu, M. Peng, X. Jia, and M. Peng, “Pre-trained language
model with prompts for temporal knowledge graph completion,” in
Findings of the Association for Computational Linguistics, 2023, pp.
7790–7803.
[43] W. Xu, B. Liu, M. Peng, Z. Jiang, X. Jia, K. Liu, L. Liu, and M. Peng,
“Historical facts learning from long-short terms with language model
for temporal knowledge graph reasoning,”Information Processing &
Management, vol. 62, no. 3, p. 104047, 2025.
[44] D.-H. Lee, K. Ahrabian, W. Jin, F. Morstatter, and J. Pujara, “Tempo-
ral knowledge graph forecasting without knowledge using in-context
learning,” 2023, p. 544–557.
[45] Y . Xia, D. Wang, Q. Liu, L. Wang, S. Wu, and X. Zhang, “Enhancing
temporal knowledge graph forecasting with large language models via
chain-of-history reasoning,”arXiv preprint arXiv:2402.14382, 2024.
[46] Z. Ding, H. Cai, J. Wu, Y . Ma, R. Liao, B. Xiong, and V . Tresp,
“zrllm: Zero-shot relational learning on temporal knowledge graphs with
large language models,” inProceedings of the Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies, 2024, pp. 1877–1895.
[47] X. Qian, Y . Zhang, Y . Zhao, B. Zhou, X. Sui, L. Zhang, and K. Song,
“Timer4: Time-aware retrieval-augmented large language models for
temporal knowledge graph question answering,” inProceedings of the
2024 Conference on Empirical Methods in Natural Language Processing,
2024, pp. 6942–6952.
[48] R. Luo, T. Gu, H. Li, J. Li, Z. Lin, J. Li, and Y . Yang, “Chain of
history: Learning and forecasting with llms for temporal knowledge
graph completion,”arXiv preprint arXiv:2401.06072, 2024.
[49] R. Liao, X. Jia, Y . Li, Y . Ma, and V . Tresp, “GenTKG: Generative
forecasting on temporal knowledge graph with large language models,”
inFindings of the Association for Computational Linguistics: NAACL
2024, 2024, pp. 4303–4317.
[50] M. Peng, B. Liu, W. Xu, Z. Jiang, J. Zhu, and M. Peng, “Deja vu:
Contrastive historical modeling with prefix-tuning for temporal knowledge
graph reasoning,” inFindings of the Association for Computational
Linguistics: NAACL 2024, 2024, pp. 1178–1191.
[51] C. Yuan, Q. Xie, J. Huang, and S. Ananiadou, “Back to the future:
Towards explainable temporal reasoning with large language models,” in
Proceedings of the ACM on Web Conference, 2024, pp. 1963–1974.
[52] K. He, X. Zhang, S. Ren, and J. Sun, “Delving deep into rectifiers:
Surpassing human-level performance on imagenet classification,” in
Proceedings of the IEEE international conference on computer vision,
2015, pp. 1026–1034.
[53] G. H. Nguyen, J. B. Lee, R. A. Rossi, N. K. Ahmed, E. Koh, and S. Kim,
“Dynamic network embeddings: From random walks to temporal random
walks,” in2018 IEEE International Conference on Big Data, 2018, pp.
1085–1092.[54] P. Veli ˇckovi ´c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and
Y . Bengio, “Graph attention networks,”arXiv preprint arXiv:1710.10903,
2017.
[55] A. García-Durán, S. Duman ˇci´c, and M. Niepert, “Learning sequence
encoders for temporal knowledge graph completion,” inthe Conference
on Empirical Methods in Natural Language Processing, 2018, pp. 4816–
4821.
[56] K. Leetaru and P. A. Schrodt, “GDELT: Global data on events, location,
and tone,” inISA Annual Convention, 2013, pp. 1–49.
[57] Z. Hu, Y . Lan, L. Wang, W. Xu, E.-P. Lim, R. K.-W. Lee, L. Bing,
and S. Poria, “Llm-adapters: An adapter family for parameter-efficient
fine-tuning of large language models,”arXiv preprint arXiv:2304.01933,
2023.
[58] C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and
K. Toutanova, “BoolQ: Exploring the surprising difficulty of natural
yes/no questions,” inProceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long and Short Papers),
2019, pp. 2924–2936.
[59] Y . Bisk, R. Zellers, J. Gao, Y . Choiet al., “PIQA: Reasoning about
physical commonsense in natural language,” inProceedings of the AAAI
conference on artificial intelligence, vol. 34, no. 05, 2020, pp. 7432–7439.
[60] M. Sap, H. Rashkin, D. Chen, R. Le Bras, and Y . Choi, “Social
iqa: Commonsense reasoning about social interactions,” inProceedings
of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural
Language Processing, 2019, pp. 4463–4473.
[61] R. Zellers, A. Holtzman, Y . Bisk, A. Farhadi, and Y . Choi, “Hellaswag:
Can a machine really finish your sentence?” inProceedings of the 57th
Annual Meeting of the Association for Computational Linguistics, 2019,
pp. 4791–4800.
[62] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y . Choi, “Winogrande:
An adversarial winograd schema challenge at scale,”Communications of
the ACM, vol. 64, no. 9, pp. 99–106, 2021.
[63] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick,
and O. Tafjord, “Think you have solved question answering? try arc, the
ai2 reasoning challenge,”arXiv preprint arXiv:1803.05457, 2018.
[64] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal, “Can a suit of armor
conduct electricity? a new dataset for open book question answering,” in
Proceedings of the 2018 Conference on Empirical Methods in Natural
Language Processing, 2018, pp. 2381–2391.
[65] J. Guo, M. Chen, Y . Zhang, J. Huang, and Z. Liu, “Hierarchical
hypergraph recurrent attention network for temporal knowledge graph
reasoning,” inIEEE International Conference on Acoustics, Speech and
Signal Processing, 2023, pp. 1–5.