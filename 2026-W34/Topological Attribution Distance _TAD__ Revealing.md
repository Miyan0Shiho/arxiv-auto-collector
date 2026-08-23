# Topological Attribution Distance (TAD): Revealing Segment-Level RAG Influence on LLM Output Geometry for Incident Log Analysis

**Authors**: Reza Fayyazi, Michael Zuzak, Shanchieh Jay Yang

**Published**: 2026-08-17 16:21:18

**PDF URL**: [https://arxiv.org/pdf/2608.16775v1](https://arxiv.org/pdf/2608.16775v1)

## Abstract
Large Language Models (LLMs) are increasingly being deployed in cybersecurity operations to assist cybersecurity analysts with rapid decision-making against emerging threats. However, there is a main criteria that must be met when using LLMs in cybersecurity, that is, trust in the generated outputs. As Agentic AI is integrated into operational systems, a robust evidence attribution and provenance tracking technique is essential to trace the origins of model generations. When autonomous agents make a decision (right or wrong), the ability to trace back through the decision chain is critical, as without it, teams cannot identify which segment of the data caused the model generation. Existing methods often struggle to distinguish among complex and highly similar evidence sources, such as cyber incident logs. This reveals a key gap: current approaches do not adequately capture the holistic geometric relationship between the retrieved evidence and the generated response for reliable evidence verification. To bridge this gap, we propose Topological Attribution Distance (TAD), inspired by Topology, to characterize and capture the global geometric shape of an output and its changes against its retrieved logs. In other words, if the embeddings of a specific source log drastically changes the geometry of the model's response in the embedding space, this suggests that such log is a critical source for the model's generated response. Therefore, TAD is powered by segment-level ablation attribution to investigate incident logs of an actual cyberattack. We demonstrate how TAD finds the most attributed logs on LLM outputs in an adaptive manner. This can provide an explainable and trustworthy tracing based on each LLM's hidden state to understand how geometrically different retrieved logs influence the model generation, and provide evidence verification in cybersecurity and Agentic-AI workflows.

## Full Text


<!-- PDF content starts -->

Topological Attribution Distance (TAD):
Revealing Segment-Level RAG Influence on LLM
Output Geometry for Incident Log Analysis
1stReza Fayyazi
Dept. of Computer Engineering
Rochester Institute of Technology
Rochester, NY , USA
rf1679@rit.edu2ndMichael Zuzak
Dept. of Computer Engineering
Rochester Institute of Technology
Rochester, NY , USA
mjzeec@rit.edu3rdShanchieh Jay Yang
Dept. of Engineering
Gonzaga University
Spokane, WA, USA
yangj@gonzaga.edu
Abstract—Large Language Models (LLMs) are increasingly
being deployed in cybersecurity operations to assist cybersecurity
analysts with rapid decision-making against emerging threats.
However, there is a main criteria that must be met when using
LLMs in cybersecurity, that is, trust in the generated outputs. As
Agentic AI is integrated into operational systems, a robust evidence
attribution and provenance tracking technique is essential to trace
the origins of model generations. When autonomous agents make
a decision (right or wrong), the ability to trace back through
the decision chain is critical, as without it, teams cannot identify
which segment of the data caused the model generation. Existing
methods often struggle to distinguish among complex and highly
similar evidence sources, such as cyber incident logs. This reveals a
key gap: current approaches do not adequately capture the holistic
geometric relationship between the retrieved evidence and the
generated response for reliable evidence verification. To bridge
this gap, we propose Topological Attribution Distance (TAD),
inspired by Topology, to characterize and capture the global
geometric shape of an output and its changes against its retrieved
logs. In other words, if the embeddings of a specific source log
drastically changes the geometry of the model’s response in the
embedding space, this suggests that such log is a critical source
for the model’s generated response. Therefore, TAD is powered by
segment-level ablation attribution to investigate incident logs of
an actual cyberattack. We demonstrate how TAD finds the most
attributed logs on LLM outputs in an adaptive manner. This can
provide an explainable and trustworthy tracing based on each
LLM’s hidden state to understand how geometrically different
retrieved logs influence the model generation, and provide evidence
verification in cybersecurity and Agentic-AI workflows.
Index Terms—LLM, Topology, TDA, RAG, Wasserstein Dis-
tance, Attribution, Incident Log Analysis, TAD
I. INTRODUCTION
Retrieval-Augmented Generation (RAG) has become a
widely used approach in Agentic-AI workflows by grounding
Large Language Models (LLMs) with relevant external knowl-
edge. In cybersecurity, RAG allows LLMs to access up-to-date
information, such as vulnerability reports and incident logs,
to help support more informed security analysis and decision-
making. However, retrieval alone does not guarantee correct
interpretation, particularly when models are exposed to large
pools of highly structured and semantically similar information.
As noted by Zhang et al. [1], RAG systems remain susceptibleto ungrounded outputs when queries are ambiguous, retrieved
context is excessive, or source quality is poor. This raises a
critical concern for AI adoption in cybersecurity applications,
and that is:trustin the generated outputs. As Agentic AI
becomes integrated into operational systems, robust evidence
attribution and provenance tracking are essential for tracing
model decisions back to their originating sources. Without such
traceability, decisions made by AI agents (whether right or
wrong) are difficult to diagnose, verify, and audit. Therefore,
there remains the challenge to reliably attributing model
decisions to their originating sources for evidence verification.
Understanding why a model generated a particular response
and which retrieved context influenced the models’ decision
is critical for providing trust, transparency, and auditability in
the age of Agentic AI.
To tackle these challenges, there have been several lines of
work that investigated providing explainability and attribution
for LLMs’ generated responses [2]–[7], yet each carries notable
limitations. One prominent approach employs LLM-based
judges to assess provenance [2]–[4]. Because these methods
rely on the same class or another set of models being evaluated,
their accuracy is fundamentally constrained by the biases,
hallucination tendencies, and self-assessment patterns inherent
to LLMs themselves. A second body of work focuses on
token-level attribution by producing importance scores for
individual input tokens to identify which tokens influence
a given generation the most [5]–[7], or similarity measures
(e.g., cosine similarity and Rouge-L). These methods are either
computationally expensive or ineffective on highly structured
and similar data, such as incident logs. Therefore, these methods
do not captureholistic segment-level grounding, which is the
degree to which amodel’s response as a wholeis driven by
retrieved sources. In other words, a security log’s influence on
model behavior is better understood holistically, as a complete
unit of evidence, rather than as a collection of individually
scored tokens. For example, in structurally similar logs, the
key distinction may arise from the connectivity between a
few tokens, such as psexec.exe appearing at an unusual
timestamp. Token-level attribution can dilute such signals across
arXiv:2608.16775v1  [cs.CR]  17 Aug 2026

many shared, non-discriminative tokens. This highlights a key
gap in current attribution methods: the lack of a mechanism
that aggregates token-level signals into coherent segment-level
attribution that identifies when a small but critical feature makes
an entire retrieved segment influential.
To account for these challenges, we turn into Topological
Data Analysis (TDA), a mathematical framework for studying
theshapeof data [8]. TDA draws on concepts from Topology,
where algebraic invariants, such as homology and homotopy
groups, are assigned to topological spaces to characterize their
structure [9]. Homotopy and homology are two lenses on
the same underlying question: “When are two topological
spaces the same”? Homotopy is a continuous deformation
between two maps ( f, g:X→Y ), formalized as a continuous
function ( H:X×[0,1]→Y ) satisfying ( H(x,0) =f(x) ) and
(H(x,1) =g(x) ). Intuitively, homotopy captures the idea that
two spaces are equivalent if one can be continuously stretched
or compressed into the other. While homotopy is the more intu-
itive notion, it is generally intractable to compute directly [10].
Homology addresses this by extracting computable algebraic
invariants (e.g., H0 for connected components, H1 for finding
holes, etc.) that are preserved under homotopy equivalence.
In other words, homology acts as a computationally tractable
proxy for homotopy, meaning that if two spaces have different
homology, they cannot be homotopy-equivalent.
Existing research has shown that feed-forward networks
progressively transforms the topology of the data as they
flow through the layers, which means that Topology has the
capability to map the non-linear space [11]. Prior work has
also shown that deeper layer does not always mean better
representation, demonstrating the importance of effect of all
the layers into the final conclusion [12]. Therefore, in the
context of LLMs, the geometric and topological properties of
the embedding spaces over the layers (i.e., non-linear hidden-
state spaces) can produce meaningful explanations about model
decisions, attribution, and uncertainty [13], [14]. Interestingly,
the auto-regressive nature of LLMs (i.e., prior context shaping
the response) can provide a powerful lens for reasoning
about model generations. Instead of comparing isolated token
representations from the response, we can characterize thetoken
connectivity of a response as a wholethrough the embedding
space and trace the influence of each prior context (each
segment) individually. In other words, if the inclusion and
omission of a retrieved context embeddings can strongly deform
the geometry of the model’s response, this can suggest as
evidence that the context served as a grounding source. This
topological perspective directly addresses the segment-level
attribution gap identified above.
Building on this intuition, we proposeTopological Attri-
bution Distance (TAD), a segment-level attribution metric
to identify the segments (e.g., logs) that cause the most
geometric change in an LLM’s generation. TAD operates by
tracking the topological signatures of the generated response’s
embedding geometry over the transformer layers, both with
and without each retrieved segment present. More specifically,
we compute the generated response’s persistence diagram fromtheir embedding geometries across each transformer layer, with
and without each segment in the context, and quantify the
changes on the response embeddings with Wasserstein distance.
By measuring the topological cost of significant changes on a
generated response’s geometry, TAD produces an attribution
score at the segment-level. This score captures the degree to
which each retrieved segment geometrically influenced the
model’s output.
We evaluate TAD on a real-world cyberattack logs, and
on three different scenarios, namelyDirect,Regular, and
Indirect. These scenarios point to the level of difficulty
of a response with respect to the ground-truth attack logs.
More specifically, theDirectcase contains many tokens from
the log in the response, theRegularcase contains some
specific keywords, and theIndirectcase contains little-to-none
token overlaps between the logs and the response. Moreover,
we compare TAD against multiple baselines, namely LLM-
as-a-judge, similarity measures, and token-level attribution
techniques. We demonstrate the effectiveness of the proposed
TAD metric across four different LLMs (with different sizes and
architectures). Our results demonstrate that TAD significantly
outperforms existing baselines with over 97% accuracy on
average in tracing the top-attributed logs, and achieving over
39% gap in Precision compared to the best-performing baseline
in all three scenarios. In Figure 1, we provide a case study to
demonstrate how different attribution techniques fail in tracing
security logs to the response, and how TAD works well in these
cases. Together, these results position TAD as an auditable,
provenance-aware measure capable of tracing model decisions
back to their retrieved sources, and therefore, building trust in
cybersecurity operations and critical decision-making.
The key contributions of this work are as follows:
•We formalize the problem of segment-level attribution in
RAG systems as a topological comparison problem, draw-
ing an explicit connection between homotopy-theoretic
reasoning and embedding-space geometry.
•We introduceTopological Attribution Distance (TAD),
a novel attribution metric that captures segment-level
influence of changes in LLM response geometry over
the hidden states, and compute their Wasserstein distance
to find the top-attributed segments.
•We demonstrate that TAD is well-suited for incident log
analysis, as logs share high token overlap within each
other, and show how TAD is capable of identifying the
logs that influence the LLM response’s geometry the most.
•We demonstrate that the proposed TAD metric outperforms
other baselines with an average of 97% accuracy on four
different LLMs, and with over 39% gap in Precision for
the three different difficulty-level cases.
II. RELATEDWORKS
A. Challenges of Existing Attribution Metrics for LLMs
The emergence of RAG systems has led to a line of research
on source attribution, which is verifying whether the generated
text is supported by identified source documents. For LLM-
based judges, there have been different works proposed to

Case Study: From an LLM Generation to Tracing Back its Supporting Evidence
1. Analyst query.
Examine logs around 00:01–00:05 on 2025-01-01. Exactly one of the following five logs is evidence of malicious activity. Identify its log_id and
explain why it is malicious in one sentence.
2. Candidate incident logs
Log Log-ID Event State
1 2e9a4...5a094jdoestartsteams.exe. Benign
2 8b1d6...fa5bd SYSTEMmodifiesHKLM\...\WindowsUpdate. Benign
3 d0c4a...73f06bjdoelogs on viaexplorer.exe. Benign
4 3a7e1...bd5a02 jdoestartscode.exe. Benign
5 7c3f0...b0a48 jdoewritesHKCU\...\Run→%TEMP%\svc.exe. Malicious
3. LLM Response (Gemma3-4b).
“Log ID 7c3f0a9e2b514d68 is the indicator of malicious activity, as the system initiated the launch of an executable from a temporary location, suggesting an attempt
to establish covert operation.”
Observed problem. The model identifies the correct event but partially hallucinates its exact log ID, requiring evidence localization.
⇓Can the Supporting Evidence be Verified?
4. Evidence localization
Family Method Result Operational consequence
SimilarityCosine✗Log 4 Retrieves a semantically similar but benign process event.
ROUGE-L✗Log 2 Lexical overlap favors a benign registry event.
Self-eval.Citation✗Logs 1,3,4,5 Includes the true log but also incorrectly cites multiple benign events.
LLM judge ▲Logs 4,5 Includes the true log but leaves a benign candidate for manual review.
Token-level
attributionLEA [5]✗Log 3 Ranks a benign logon event first as logs share so many overlapping tokens.
Segment-level
attributionTAD (ours) ✓Log 5 Uniquely identifies the log supporting the response; verifying evidence localization.
Fig. 1: Evidence-localization case study. The Gemma model identifies the malicious registry-persistence event but partially
hallucinates its exact log identifier, making the generated answer insufficient for direct verification. Existing methods retrieve
benign logs, return multiple candidates, or cite almost the full context. TAD uniquely localizes Log-5 by measuring the effect
on the response geometry, allowing an analyst to trace the response to its supporting evidence.
evaluate RAG systems. Self-RAG [2], RAGAS [3] and ARES
[4] employ LLM-based judges to self-assess RAG systems,
either through the generation model itself, or through another
set of LLMs. However, these methods are constrained by the
evaluation models’ self-interpretation and reasoning biases,
and they do not fully decouple attribution assessment from
parametric model knowledge (i.e., hidden states).
Token-attribution methods aim to quantify the contribution
of individual input features, typically token embeddings, on a
model’s output. Numerous works have addressed the problem
of attributing model predictions to individual input features [15].
In LLMs, LIME [16] generates variations of the input text (e.g.,
by masking or removing tokens) and observes changes in output
probabilities [7], and SHAP [17] scores are derived by masking
or perturbing subsets of input tokens and observing output
variation [7]. However, these methods are computationally
expensive, with SHAP being exponential, and cannot scale
with a large pool of retrieved context (e.g., security logs). LRP[18] is a backpropagation-based method that redistributes the
model’s prediction score backward through the network layers,
assigning relevance scores to input features, and there have
been works used LRP for attribution [6], [19]. Furthermore, in
our previous work [5], we proposed the LLM Embedding-based
Attribution (LEA) metric, a computationally efficient token-
level attribution metric based on linear dependency analysis
of input embeddings. Overall, despite significant progress,
existing feature attribution techniques are limited to thetoken-
leveland can struggle when retrieved information contains
highly overlapping tokens (e.g., logs). These methods While
token-level attribution is useful for identifying which tokens
most influence a given prediction, it does not capture holistic
segment-level grounding, which is the degree to which the
model’s response as a wholeis driven by the retrieved context.
This points to a broader open challenge in the field: bridging
the gap between token-level attribution into segment-level
attribution tracing in RAG systems.

B. Topological Data Analysis (TDA) in LLMs
The application of Topological Data Analysis to language
models spans several interrelated directions [20]. Draganov and
Skiena [21] demonstrate that persistent homology applied to
word embedding clouds encodes meaningful linguistic structure.
Rottach et al. [22] introduced Unified Topological Signatures,
which is a framework that aggregates topological and geometric
descriptors to characterize embedding spaces. The authors
revealed that models within the same family show similar
topological properties, which suggests that architecture and
training data strongly shape the embedding space geometry.
Within LLM internal representations, there are some works
that apply topology directly to hidden-state activations. Fitz
et al. [23] compare the layer-wise topological complexity of
Transformer and LSTM hidden-states using persistent homol-
ogy and perforation, and they found that LSTMs exhibit richer
topological structure specific to natural language. Gardinazzi
et al. [13] employ zigzag persistence to track how topological
features evolve across layers, introducing a Persistence Similar-
ity metric that identifies redundant layers and allows principled
pruning with minimal performance loss. Fay et al. [14] propose
persistent homology as an architecture-agnostic adversarial
detection framework, identifying a consistent “topological
compression” signature under prompt injection and backdoor
attacks, where adversarial inputs simplify latent-space structure,
and demonstrating clean separation of normal and adversarial
activations via barcode summary statistics.
Despite their promise, existing topological approaches to
LLM analysis have notable limitations. Fitz et al. [23] demon-
strated transformer models does not show topological com-
plexity as going deeper in the layers (compared with LSTM).
However, their perforation metric excludes H0homology to
capture connected components. We will show in this paper,
that H0 plays a key role in finding shape differences in the
embedding space. Gardinazzi et al. [13] showed the importance
of each layer in different LLMs. While certain layers contribute
more significantly than others to model outputs, pruning even
the least critical layers resulted in substantial degradation
in benchmark performance. This indicates the importance of
representation across all layers, and therefore, motivating us
to incorporate this insight into our proposed methodology.
Furthermore, Fay et al. [14] identified topological compression
as an adversarial signature across six LLMs. However, the
framework focuses exclusively on a last-token hidden state
and relies on batch-level statistics, which requires them to
depend on batches of data on similar categories (clean or
poisoned) to obtain barcode summaries. This dependence on
homogeneous batches to define relational geometry prevents the
attribution of a single generated response in isolation, and lacks
real-time applicability. Finally, none of these works account
for segment-level attribution and multi-source influence on
response generation.
III. BACKGROUND ONTOPOLOGICALDATAANALYSIS
In this section, we explain the mathematical foundations of
Topological Data Analysis (TDA). We discuss the constructionof simplicial complexes from point cloud data, persistent
homology to track topological features across scales, and the
Wasserstein distance for comparing persistence diagrams.
A. Point Clouds and Metric Spaces
The input to TDA is typically a point cloud, i.e., a finite set
represented as:
V={v 1, v2, . . . , v n} ⊆M,(1)
where each point viis embedded in a feature space Mequipped
with a metric d:M×M→R , so that (M, d) forms ametric
space. The metric d(vi, vj)quantifies the dissimilarity between
any two points. A common choice whenM=Rnis theL ∞
distance,
d(v, w) =∥v−w∥ ∞= max
k|vk−wk|.(2)
The central insight of TDA is that a discrete set of points
carries latent geometric and topological information that can
be systematically extracted from their pairwise distances [24].
Thus, doing this requires lifting the point cloud to a higher-level
combinatorial structure called a simplicial complex.
B. Simplicial Complexes
A simplicial complex provides a combinatorial representation
of a point cloud using basic building blocks called simplices
[25]. A k-simplex is defined as the convex hull of k+1 affinely
independent points. In other words, k= 0 gives vertices, k= 1
gives edges, k= 2 gives filled triangles, etc. One standard
construction for building a simplicial complex from a metric
point cloud is theVietoris–Rips complex[26]. Given a scale
parameter ε >0 , the Vietoris–Rips complex over a point set
Vis defined as:
Rε=
σ⊆Vd(vi, vj)≤ε∀v i, vj∈σ	
.(3)
That is, a k-simplex is included wheneverallpairwise distances
among its k+1 vertices are at most ε. The choice of εdecides
which connections are formed: too small avalue causes an
almost empty complex capturing only trivial local structure,
while too large avalue collapses all points into a single fully
connected component, which motivates studying the complex
across all scales simultaneously.
C. Filtrations and Multi-Scale Analysis
There is no value of εthat can be globally optimal. TDA
resolves this ambiguity by consideringall scales simultaneously
through afiltration[25], which is a nested sequence of
simplicial complexes:
∅=K 0⊆K 1⊆K 2⊆ ··· ⊆K m=K,(4)
These are indexed by an increasing sequence of parameter
values ε0< ε1<···< ε m. Asεgrows, new simplices are
progressively added. Initially, points appear as isolated vertices;
edges then form between nearby points; eventually, higher-
dimensional simplices emerge, which represents clustered
regions of the space. Given a scalar function f:V →R

defined on the point cloud V ⊂Rn, the associatedsublevel-set
filtration[27] is the nested family of subsets
Vt=f−1 
(−∞, t]
=
v∈ Vf(v)≤t	
,(5)
which is obtained by progressively admitting points in order
of increasing the f-value as the threshold tgrows. At each
step, a simplicial complex (e.g., a Vietoris–Rips complex) is
built on Vt, capturing how the topology of the space evolves
as points are included according to their functional values.
At each stage of the filtration, the shape of the complex is
quantified usinghomology[9]. For each non-negative integer
k, thek-th homology group Hk(K), typically computed over
a field F(most often F2={0,1} ), captures independent
topological features of dimension k. More specifically, H0(K)
counts connected components, H1(K) captures independent
loops or cycles, H2(K) detects enclosed voids, and higher
Hk(K)encode their higher-dimensional analogues.
D. Persistent Homology: Birth, Death, and Persistence
Persistent Homology tracks how homological features evolve
across a filtration [24], [25], [28], [29]. As the filtration
parameter εincreases, topological featuresare born(first
appear in Hk(Kε)) and eventuallydie(merge with older
components or are filled in by higher-dimensional simplices).
Thepersistenceof the feature is:
pers(α) =ε death−εbirth.(6)
Features with high persistence are considered topologically
significant, while features with low persistence are typically
attributed to noise or sampling variability [27]. Note that
at least one connected component in H0never dies, since
some component persists for the entire filtration. The complete
lifecycle of topological features can be summarized in two
equivalent representations:
Persistence Barcode[30], [31]: A persistence barcode is a
multiset of intervals [bi, di), drawn as horizontal bars, where
each bar corresponds to one topological feature and its length
equals the feature’s persistence. Long bars indicate stable
features and short bars (near zero) indicate a noisy structure.
Persistence Diagram[25], [27], [32]: A persistence diagram
is a multiset of points (bi, di)∈R2with di≥bi, plotted in
the half-plane above the diagonal ∆ ={(x, x) :x∈R} . The
diagonal itself is included with infinite multiplicity, representing
features that die immediately upon birth. Points far from ∆
correspond to significant features, while points near ∆indicate
noise. The persistence diagram arising from a filtration function
fin dimension kis denoted as Dgmk(f). Therefore, having
established a means of representing topological features as
persistence diagrams, a natural question arises: “how does one
quantify the similarity (or distance) between two such diagrams
in a geometrically meaningful way?” We will discuss this next.
E. Wasserstein Distance
To compare two persistence diagrams PandQ, which may
have different cardinalities, one seeks apartial matching γthat minimizes a total cost [27], [32], [33]. To accommodate
unmatched points, let bP=P∪∆ andbQ=Q∪∆ denote
the diagrams augmented with the diagonal ∆ ={(x, x) :
x∈R} , treated as trivial features with zero persistence. A
matchingis then a bijection π:bP→bQ, where any point
paired with a diagonal point is consideredunmatched. The cost
of transporting a point u= (b u, du)to a point v= (b v, dv)
under theL∞norm is:
∥u−v∥ ∞= max 
|bu−bv|,|du−dv|
,(7)
while matching u= (b, d) to its orthogonal projection onto ∆
incurs the persistence-proportional cost
∥u−∆∥ ∞=d−b
2,(8)
reflecting the minimal L∞displacement required to remove a
topological feature of persistence d−b . Therefore, the (∞,1) -
Wasserstein distance( W(∞)
1) between persistence diagrams
PandQis defined as:
W(∞)
1(P, Q) = inf
γ"X
(u,v)∈γ∥u−v∥ ∞+
X
u∈P\γd(u,∆) +X
v∈Q\γd(v,∆)# (9)
where γranges over all partial matchings between Pand
Q, and d(u,∆) = (d u−bu)/2denotes the ℓ∞-distance from
u= (b u, du)to its orthogonal projection onto ∆. Intuitively,
the first sum penalizes geometric discrepancy between matched
features, while the second and third sums penalize unmatched
features in PandQ, respectively. Overall, Wasserstein distance
provides a geometrically meaningful and computationally
tractable measure of dissimilarity between persistence diagrams
[33]–[35]. This quantifies how much topological structure must
be changed to transform one diagram into another.
IV. METHODOLOGY
In this section we instantiate the TDA pipeline discussed in
Sec. III within the context of LLMs, and discuss the proposed
methodology on using Wasserstein distance to track attribution
of generations over the transformer layers.
A. Mapping TDA concepts to LLMs
Token Embeddings as Metric Spaces: In the LLM setting,
each point viin the point cloud corresponds to avector
representation (embedding)of a token. The space is M=Rn,
where nis the model’s hidden dimension, and the metric d
captures geometric dissimilarity between embeddings. We adopt
theL∞distance, as can show salient directional differences
in high-dimensional embedding spaces.
Simplicial Complexes to Encode Semantic Relationships:
We apply the Vietoris–Rips complex to the token embedding
point cloud. At threshold ε, two token embeddings are
connected by an edge whenever their L∞distance is at most ε.
Higher-order simplices form when groups of embeddings are
mutually within εof one another. Since no single εcaptures

Fig. 2: TAD’s procedure of tracing attribution with corresponding retrieved data. First, for a generated response, we get the
LLM Output shape using Vietoris Rips algorithm. Next, to make TAD scalable across a vast amount of logs, we start by
partitioning data into√
Nand do Wasserstein distance of the original output vs. the ablated batch output. We do this over all
the transformer layers and aggregate the overall distance. For batches with largest gap and above, we proceed to Step-3 to
analyze the remaining logs. Finally, we do one-by-one ablation with Wasserstein distance to trace the top attributed log(s).
the full hierarchical organization of the embedding space,
we construct the full Vietoris-Rips filtration over the token
embeddings. Each Kεirepresents the simplicial complex at
scale εi, and the nested sequence tracks how semantic groupings
form and merge as the threshold grows.
Filtrations can be driven by scalar functions defined on the
embeddings. In our setting, a natural choice is thesublevel-
set filtrationparameterized byLayer-wise Hidden States.
The filtration captures which tokens are “attended to” at each
stage. This construction connects topological analysis directly
to model-internal signals, making the filtration sensitive to
saliency and anomalous patterns in LLM representations.
Homology Groups for Topological Invariants of the Embed-
ding Space:At each scale εi, we compute the homology groups
of the Vietoris-Rips complex built over the token embeddings:
H0reflects how embeddings group into coherentsemantic
regions (clusters)at a given scale. H1capturescyclesin the
embedding space, which may correspond to gaps or circularly-
related semantic concepts in representation. Higher-dimensional
groups Hk(k≥2 ) encode more abstract multi-way relational
structures among embeddings. These homological invariants
provide a compact summary of the global geometry of the
embedding space at each scale.
Persistent Homology to Track Semantic Structure Across
Scales:Applying persistent homology to the token embedding
filtration allows for a complete record of which semantic
structures appear and disappear as εgrows. Abirthevent in H0
corresponds to a new isolated connected component emerging
at scale εb. Adeathevent in H0corresponds to two connected
components merging into a broader group at scale εd. Births and
deaths in H1correspond to the formation and filling of cycles
of embeddings, which may indicate conceptual relationships.
In this work, our primary focus is on H0homology, but
we will provide H1results in Appendix A. Thepersistence
pers(α) =ε d−εbof a semantic feature αquantifies its
stability. Features with high persistence represent robust and
meaningful patterns in the embedding space, such as well-defined topics, while features with low persistence could arise
from noise or incidental embedding proximity. The multiset
of all such persistence values forms apersistence diagram
Dgm(X)⊂R2, where each feature αcontributes a point
(εb, εd). The structural similarity between two such diagrams
can then be quantified via theWasserstein distance(refer to
Equ. 9). Therefore, one could ask: “How can these topological
descriptors be leveraged to systematically track the attribution
and mapping of semantic domains across embedding spaces?”
We discuss this in the following section.
B. Topological Attribution Distance (TAD)
We now turn to the problem of attributing model generations
to individual incident log segments. Our approach uses per-
sistent homology to quantify how each retrieved log segment
influences the topological structure of the model’s response. To
quantify how different RAG segments influence model output,
we must first ground our approach in the fundamental properties
of decoder-only LLMs: autoregressive token generation. Since
each generated token depends exclusively on all preceding
tokens in the context window, a critical implication follows:
every element in the prompt contributes to shaping the
response geometry with varying degrees of causal influence.
Therefore, the positional placement, sequential ordering, and
compositional structure of RAG segments within the context
window can produce measurable effects on model output.
To comprehensively measure how a given set of retrieved
logs contributes to the final response, the methodology requires
full context concatenation: all segments are appended sequen-
tially into a single context along with the generated response.
This design ensures that the causal relationship between
each segment and the model’s output can be traced without
ambiguity. By doing so, segment-level contribution can be
isolated through controlled ablation: systematically excluding
individual segments across experimental runs while keeping
all other variables constant. Together, these two mechanisms
(i.e., full concatenation and targeted ablation) form the basis
for attributing response content back to its originating context.

To get the topological mapping of a model’s output with
each log segment contributing to the output, we compute the
(∞,1) -Wasserstein distance (refer to Equ. 9) of H0 homology
between their corresponding persistence diagrams on each layer
of an LLM. The reason for H0 homology is that it is more
computationally efficient (see Sec. V), and prior work has
shown H0’s effectiveness on geometric reorganization tasks
[36]. To capture an overall representation, we aggregate the
Wasserstein values over the layers to obtain the attribution of
each log segment. This is due to the nature of transformers that
every layer correspond to the final decision, as also discussed
by prior work [13]. The following is the proposed TAD metric:
T AD(P, Q) =LX
ℓ=1W(∞)
1(Dgm0(Pℓ),Dgm0(Qℓ))(10)
where Dgm 0represents the H0 homology, and PlandQl
are persistent diagrams of the ablated context and the full
context on each layer, respectively. This proposed extension
can provide attribution beyond token-level impact, as it can
provide explainability at segment-level. The key insight is that
each log segment that most strongly influences the output,
bears representational similarity to the output itself in the
feature space. Therefore, ablating these influential segments
accordingly should produce substantially larger Wasserstein
distances, providing a direct measure of their contribution.
Figure 2 demonstrates our proposed methodology. First, we
characterize the topological structure of the generated output
by constructing a Vietoris-Rips complex over its representation
and extracting H0 persistent homology with the full context
+ the generated response. Next, since per-log ablation is
computationally prohibitive at scale, we adopt ascreen-then-
confirmattribution strategy. In thescreenstage, the input
log sequence is partitioned into approximately√
Nbatches.
Each batch is ablated in turn, and the resulting perturbation
to output shape is quantified via the Wasserstein distance
between the persistence diagrams of the original and ablated
LLM response, computed independently at each transformer
layer and aggregated across layers into a single distance
measure per batch. The batch containing the largest aggregate
distance is then selected for finer-grained analysis. In the
confirmstage, we perform single-log ablation restricted to this
high-attribution batch, again computing layer-wise Wasserstein
distances between resulting persistence diagrams to identify
and rank the individual log entries most responsible for shaping
the model’s output geometry. InAlgorithm 1, we demonstrate
the full detail process for the proposed TAD methodology.
V. DISCUSSION ONCOMPLEXITY
The cost of TAD decomposes into two stages: transformer
inference and persistent homology. Let Ndenote the number of
candidate log entries in the context, Tthe number of response
tokens in the point cloud, dthe hidden dimension of the model,
andLthe number of transformer layers.
Inference cost.TAD is attribution-by-ablation, therefore, the
complexity is the number of forward passes Missued toAlgorithm 1TAD: Topological Attribution Distance
Require: Context Cwith logs L, LLM Mwith Llayers,
homology dimensiond, min grouping sizeτ(default9)
Ensure:Set of spike logsS ⊆ L
1:R← M(C)▷Response, generated once and held fixed
2:D(1:L)
base←DIAGRAMS(C, R, d)
3:if|L|< τthen▷Too small to screen; confirm every log
4:returnCONFIRM(L, C, R,D(1:L)
base )
5:end if
6:H ←SCREEN(L, C, R,D(1:L)
base)
7:S ←CONFIRM(H, C, R,D(1:L)
base)
8:returnS
9:procedureDIAGRAMS(C′, R, d)▷Only response tokens
10:X(1:L)←HIDDENSTATES(M, C′⊕R)
11:return 
VIETORISRIPS(X(ℓ)↾R, d)L
ℓ=1
12:end procedure
13:procedureSCREEN(L pool, C, R,D(1:L)
base )
14:k← ⌈p
|Lpool|⌉
15:G ←PARTITION(L pool, k)
16:foreach groupG i∈ Gdo
17:C′←ABLATE(C, G i)▷Only the context changes
18:D′(1:L)←DIAGRAMS(C′, R, d)
19:∆ i←TAD(D′,Dbase)
20:end for
21:L hot←S
G∈G hotG
22:if|L hot| ≥τthen▷re-partition
23:returnSCREEN(L hot, C, R,D(1:L)
base )
24:end if
25:returnL hot
26:end procedure
27:procedureCONFIRM(H, C, R,D(1:L)
base )
28:foreach logl∈ Hdo
29:C′←ABLATE(C,{l})
30:D′(1:L)←DIAGRAMS(C′, R, d)
31:∆ l←TAD(D′,Dbase)
32:end for
33:returnFLAGSPIKES(H,∆)
34:end procedure
35:procedureFLAGSPIKES(X,∆)
36:Sort∆descending with orderσ
37:gaps[i]←∆ σ[i]−∆ σ[i+1]
38:c←arg max igaps[i]
39:return{X σ[i]:i≤c∧gaps[c]>0}
40:end procedure
the model. A purely linear case, in which each log is ablated
individually, requires M=N+1 passes (one baseline plus one
per log), which scales linearly in context length and becomes
prohibitive for long Agentic traces. Our proposed screen-then-
confirm procedure instead partitions the pool into ⌈√
N⌉groups

and recurses only on the groups flagged as spikes, which
reduces the expected cost toM=O(√
N)passes.
Topological cost.For each forward pass and each layer ℓ,
TAD builds a point cloud Xℓ∈RT×dfrom the response-token
hidden states and computes the persistence diagram of its
Vietoris–Rips filtration. Forming the pairwise distance matrix
costs O(T2d). When restricted to H0, only the 1-skeleton is
required: the O(T2)edges are sorted in O(T2logT) time. The
per-pass cost is therefore O 
L T2(d+ logT)
, i.e. quadratic
in the number of response tokens and linear in network depth.
Extending to H1is substantially more expensive, as it needs
2-skeleton, which may contain O(T3)triangles, and can have
cubic complexity in the number of simplices. Overall, the end-
to-end cost of TAD is thus O 
M·L·T2(d+ logT)
in the
H0homology, with M=O(√
N)under adaptive screening,
and it is the configuration we adopt throughout our experiments.
VI. RESULTS
In this section, we will introduce our dataset, experimental
design, and TAD results for incident log analysis.
A. Dataset Curation & Attack Scenario
We curated the dataset from logs collected during a real
cyberattack executed in the Game of Active Directory (GOAD)
environment [37] and monitored using the Wazuh security
framework [38]. GOAD is an intentionally vulnerable Windows
Active Directory environment designed for penetration testing
and security training. The attack was conducted in October
2025 and during this timeframe, we identified 20 log entries
as the ground-truth attack evidence. These entries correspond
to events directly associated with the adversarial activities
performed during the attack, while the remaining logs represent
benign activity, or events not directly attributable to the attack.
Cybersecurity logs often contain long sequences of nearly
identical events, such as brute-force attempts that differ
only in timestamps, ports, or event identifiers. To reduce
this redundancy, we applied Gestalt Pattern Matching to
group consecutive logs with ≥90% sequence similarity
and represented each group with a single entry. We added
ahow-many-consecutive metadata field to record the
number of original events represented by each compressed
log. This preserves event frequency while preventing repetitive
activity from dominating the analysis. After compression, the
dataset contains 587 log entries. Therefore, the compression
reduced the log volume while preserving the semantics and
frequency of the log events. In Table I and Table II, we
demonstrate the details of our dataset.
Attack Progression Summary:The attack consists of a
multi-stage intrusion involving pass-the-hash authentication,
lateral movement, remote execution, tool deployment, and
defense evasion. On the GOAD environment [37], the attacker
operated from NPC-PETYERBAELI host and compromised
therobb.starkaccount to gain access to several systems.
1)Credential Validation and Initial Access:On October
10, at approximately 18:51, a logon attempt for the
robb.stark account failed on NPC-PETYERBAELI .TABLE I: Dataset overview. Each window is a slice of Windows
event logs surrounding the attack logs. #GT refers to the number
of ground-truth attack-relevant logs.
W Date Start (UTC) End (UTC) #Logs #GT
1 2025-10-10 18:51:21 18:52:16 71 3
2 2025-10-10 18:56:59 18:57:45 48 1
3 2025-10-10 19:02:21 19:03:40 48 2
4 2025-10-10 19:09:46 19:11:17 50 4
5 2025-10-10 19:54:49 19:55:47 46 3
6 2025-10-10 20:00:00 20:01:41 59 3
7 2025-10-11 03:35:51 03:37:51 81 1
8 2025-10-11 06:56:27 06:57:05 47 1
9 2025-10-11 18:56:41 18:57:26 50 1
10 2025-10-11 20:48:45 20:48:46 87 1
Total(10 windows) 587 20
TABLE II: Ground-truth attack events categories of the dataset.
Technique Count
Pass-the-Hash (NTLM remote logon) 9
Registry modification (BAM, tool execution) 4
Malicious service creation (PowerShell payload) 2
PSEXESVC service installation 2
Credential brute force (logon failure) 1
Interactive logon, compromised credentials 1
Registry value integrity change 1
Total 20
This was later followed by a successful elevated interactive
logon on the same host. Within seconds, the attacker used
the same account in a successful pass-the-hash authenti-
cation against winterfell . This sequence shows that
the attacker had obtained the account’s NTLM credential.
2)Lateral Movement and Service Deployment:Be-
tween approximately 18:57-19:10, the attacker used
pass-the-hash authentication to access winterfell ,
kingslanding , and meereen . Several authentication
attempts used consecutive source ports, showing that the
activity was carried out through an automated sequence.
After successful NTLM authentication, the attacker de-
ployed PowerShell-based services with AMSI-bypass
functionality on winterfell . These services allowed
remote command execution.
3)Remote Execution:Between approximately 19:55
and 20:01, the attacker deployed the PSEXESVC ser-
vice and continued pass-the-hash authentication from
NPC-PETYERBAELI.
4)Post-Exploitation Tooling and Defense Evasion:
On October 11, the attacker deployed and executed
additional post-exploitation tools across several
systems. COLIncrease.exe was executed on
castelblack under the robb.stark account.
A separate copy of the executable was later
executed again on NPC-PETYERBAELI , as shown
by a change in the BAM entry’s checksum.
The attacker also executed PsExec64.exe on

NPC-PETYERBAELI to support further remote
execution. Finally, DefenderRemover.exe was
executed on vdi-samwell-tar to disable Windows
Defender and reduce endpoint security protections.
Overall, the sequence shows that the attacker systematically
expanded access across the environment and weakened security
controls to support continued malicious activity.
B. Experimental Design
In real-world scenarios, where thousands or millions of
log entries may be generated, providing all available logs
directly to an LLM is often impractical due to context-
window limitations, increased inference cost, and the presence
of irrelevant information that can degrade response quality.
A targeted investigation allows the model to focus on the
subset of evidence most relevant to the analyst’s hypothesis
while reducing noise and computational overhead. Therefore,
our experimental scenario is motivated by a targeted threat-
investigation workflow. We assume that a security analyst has
already developed a suspicion that a particular user account
may have been compromised. The analyst subsequently queries
the system using a prompt of the following form:
Examine the logs recorded between [start time] and [end
time] on [date]. Is there evidence of an attack in these logs?
This formulation bounds the investigation by date and time,
reflecting a realistic post-alert workflow. Rather than performing
unrestricted threat hunting, the model assesses whether the
supplied logs support the analyst’s hypothesis.
Furthermore, directly comparing natural-language responses
from multiple LLMs introduces confounding factors such as
differences in wording, detail, structure, confidence, and event
interpretation. To improve comparability, we assume that a
canonical response derived from the ground-truth attack logs
is already available and hold this response constant across
all evaluated models. To curate these canonical responses, we
usedClaude-Opus-4.5model to generate high-quality outputs
that closely match with the ground-truth attack logs. This
design constitutes a controlled experimental abstraction. In a
real-world deployment, each model would generate its own
investigation report, and the resulting conclusions could vary
according to the model’s training data, reasoning capabilities,
and contextual understanding. Therefore, the use of a canonical
response improves experimental comparability, but does not
evaluate the quality of the model-specific response generation.
To evaluate TAD on different difficulty levels, we considered
three scenarios: “Direct”, “Regular”, and “Indirect”. “Direct”
is for when the LLM significantly copies the words from the
source logs. The “Regular” case is for when an LLM copies
some keywords but gives a more general interpretation. The
“Indirect” case is for when the LLM almost do not use any
critical word that can map the response to the specific log(s)
via a keyword-search. The reason for this is that we will
show TAD is not based on finding exact keywords and puttinghigher attention to those. Instead, it is based on the overall
interpretation of each log (as a segment).
Furthermore, we compare TAD against multiple baselines:
LLM-as-a-judge, in-line citation, similarity-based measures
(Cosine Similarity and ROUGE-L), and a token-level attribution
technique (LEA). These baselines represent commonly used
approaches to assess the provenance of LLM-generated re-
sponses. For the LLM-as-a-judge and in-line citation baselines,
we use greedy-decoding to get deterministic and reproducible
responses. Appendix C shows the engineered prompts for these
methods. For token-level attribution, we adopt our previously
proposed LEA metric [5], as it is based on linear dependency
of input embeddings and is computationally efficient. Note
that we do not consider other token-level attribution methods,
such as SHAP, due to their high computational complex-
ity. Finally, for the cosine embedding model, we used the
Alibaba-NLP/gte-modernbert-base model, as it is
well-suited for long-context semantic search (8,192 tokens).
The LLMs used for analysis are: Qwen3-4B,
Gemma3-4B, Qwen2.5-7B, and Granite4.1-8B .
The choice of these models are due to their long context
window (a minimum of 128K context window) to handle large
security logs and the ability to follow-instructions well. It is
worth noting that we conducted our experiments on a system
equipped with two Intel Xeon E5-2650 CPUs, 256 GB of
RAM, and a NVIDIA Tesla L40S GPU. The code & data is
available at: https://github.com/RezzFayyazi/TAD.
C. TAD for Incident Log Analysis
As discussed in the previous section, we compare TAD with
different baselines under three settings,direct,regular, and
indirect, which differ in how openly the response refers to
the logs. All methods use the same selection rule: we sort the
candidates by score, find the largest gap between consecutive
scores, and return everything above it. We do this because an
analyst does not know in advance how many logs influenced
a response. A fixed top- kcutoff would either cut off real
contributors or add noise, whereas the gap rule lets the scores
set their own threshold.
Table III demonstrates the results. As can be seen, TAD
obtains the best accuracy and F1-score in almost all three
settings, although the margin differs. On average, for the
“Direct”, “Regular”, and “Indirect” cases, TAD outperforms the
closest baseline by1.60%,3.11%, and6.63%in accuracy, and
by7.35% 15.77%, and17.63%margin in F1-score, respectively.
Note that the margin ranges from1.60%to6.63%in accuracy,
and from7.35%to17.63%in F1-score, with the “Indirect” case
exhibiting the widest margin. This is because other baselines
perform better when there are high-token overlaps from the
logs and the response, and perform poorly when the response
contains little-to-none shared tokens with the logs. It is worth
noting that in Appendix A, we provide TAD results based on
H1 homology, and in Appendix B, we provide a qualitative
example along with some specific examples on how TAD traces
a log that contributed the most to the LLM response.

TABLE III: Performance comparison of TAD (H0 homology) and other attribution tracing methods across four LLMs. The
Direct, Regular, and Indirect cases point to the difficulty-level of the response with respect to the ground-truth logs.
Case MethodQwen3-4B Gemma-3-4B Qwen2.5-7B Granite-4.1-8B
Accuracy F1-score Accuracy F1-score Accuracy F1-score Accuracy F1-score
DirectLLM-as-judge 0.9693 0.6897 0.9046 0.2821 0.97960.7500 0.9625 0.6333
In-line citation 0.9710 0.5143 0.9693 0.4000 0.9455 0.4074 0.97790.6829
Cosine similarity 0.8058 0.2192 0.8058 0.2192 0.8058 0.2192 0.8058 0.2192
ROUGE-L 0.7990 0.1690 0.7990 0.1690 0.7990 0.1690 0.7990 0.1690
LEA 0.9199 0.4051 0.9199 0.4051 0.9199 0.4051 0.8535 0.2712
TAD (ours) 0.9847 0.7097 0.9779 0.6061 0.9830 0.6667 0.9830 0.6667
RegularLLM-as-judge 0.8790 0.3604 0.7274 0.1304 0.9267 0.4416 0.9659 0.6296
In-line citation 0.9727 0.6000 0.9063 0.2466 0.9489 0.3478 0.9659 0.6296
Cosine similarity 0.9131 0.3544 0.9131 0.3544 0.9131 0.3544 0.9131 0.3544
ROUGE-L 0.6968 0.1010 0.6968 0.1010 0.6968 0.1010 0.6968 0.1010
LEA 0.8007 0.1583 0.8262 0.1774 0.8007 0.1583 0.7104 0.1414
TAD (ours) 0.9847 0.7097 0.9727 0.5000 0.9796 0.6000 0.9813 0.6452
IndirectLLM-as-judge 0.7819 0.2099 0.7172 0.1170 0.8756 0.2626 0.6968 0.1275
In-line citation 0.9131 0.3377 0.88760.1951 0.9216 0.2333 0.8790 0.2198
Cosine similarity 0.6048 0.1008 0.6048 0.1008 0.6048 0.1008 0.6048 0.1008
ROUGE-L 0.6542 0.1057 0.6542 0.1057 0.6542 0.1057 0.6542 0.1057
LEA 0.8041 0.0171 0.7836 0.0155 0.8041 0.0171 0.8041 0.0171
TAD (ours) 0.9796 0.6000 0.9438 0.1538 0.9761 0.5625 0.9659 0.3750
Moreover, based on our observation, the baselines indicated a
clear pattern: Every baseline reaches high recall but much lower
precision: it finds the influential logs, but only by returning a
large set in which they sit among many false positives. TAD
does this differently by reaching high Precision for the “Direct”,
“Regular”, and “Indirect” cases with 94.23%, 86.90%, and
57.70% respectively. On average, for the “Direct”, “Regular”,
and “Indirect” cases, TAD outperforms the closest baseline by
39.47%,47.46%, and40.88%margin in Precision, respectively.
This matters in practice, as in the real-world, an analyst has
limited time and a pool of thousands of logs. Therefore, a
method that returns so many candidates (i.e., high recall) to
reveal a few real ones, restores the manual work that automatic
attribution is meant to remove. In addition, by looking at the
results, one might ask: “Why does the quality of the F1 score
differ for each LLM?”. The reason is thelearned representation
of each LLM individually. The learned representation matters on
how each LLM connect tokens as a segment in the embedding
space, as discussed by Rottach et al. [22]. This means that if
an LLM learned about cybersecurity more than the other, it
naturally connects tokens in a more complex way. Therefore, an
LLM likeGemma3-4Bperforming worse than other models in
our experiments, demonstrate that this model has not properly
learned a mapping between security logs and the response.
Furthermore, the selected baselines perform worse for
different reasons. First, LLM-based self-assessment is sensitive
to prompt design, meaning that different engineered prompts
can give different attribution judgments. This variability arises
because the decision comes from the model’s own reasoning
over a follow-up prompt, rather than TAD’s mathematicallytraceable internal property of the original context (that caused
the LLM generation). Second, similarity and token-level
methods fail because incident logs share highly overlapping
tokens, causing the log that shaped the response almost identical
to its near-duplicates under cosine similarity. TAD avoids both
problems by measuring what the model does with the context
(for a generated response) rather than how the context is
similar to the response. Removing an influential log causes a
large, clearly separated spike in the distance between layer-
wise persistence diagrams. Because the response is fixed and
the computation is deterministic, the result is reproducible,
traceable, and reflects how the model maps context to response.
Future work could investigate the importance of individual
layers on the generation and explore scalable variants of
the Vietoris-Rips construction. We see this as a step toward
transparent and verifiable attribution that allows analysts to
inspect and validate the evidence, rather than simply trust the
model’s output.
VII. CONCLUSION
We introduced Topological Attribution Distance (TAD) to
capture the global geometric shape of an LLM output and
its changes against its retrieved context. In other words, we
show that when the embedding of a retrieved context (e.g.,
security logs) drastically changes the geometry of the model’s
response, this serves as robust verification of evidence, and can
be traced. Therefore, we designed TAD powered by segment-
level ablation attribution to investigate incident logs of an
actual cyberattack, and we demonstrated how TAD finds the
most attributed logs to the LLM output adaptively. TAD can
provide an explainable and trustworthy tracing based on each

LLM’s hidden state to understand how different RAG segments
geometrically influence the model generation, and provide
evidence verification in cybersecurity operations and Agentic-
AI workflows.
ACKNOWLEDGMENTS
This material is based upon work supported by the National
Science Foundation under Grant No. 2344237 and No. 2502341.
The authors gratefully acknowledge Dr. Justin Pelletier and Mr.
Forrest Fuqua for their valuable contribution to the curation of
the dataset.
AI DISCLOSURE
We used ChatGPT to assist with sentence-level editing and
grammar checking. The tool also contributed to the aesthetic
refinement of Fig. 1 and Fig. 3, and prompt-boxes. The authors
independently reviewed and verified the accuracy, originality,
and integrity of all content, including the cited references.
REFERENCES
[1]W. Zhang and J. Zhang, “Hallucination Mitigation for Retrieval-
Augmented Large Language Models: A Review,”Mathematics, vol. 13,
no. 5, p. 856, 2025.
[2]A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-RAG: Learning
to Retrieve, Generate, and Critique through Self-Reflection,”The Twelfth
International Conference on Learning Representations, 2023.
[3]S. Es, J. James, L. E. Anke, and S. Schockaert, “RAGAs: Automated
evaluation of retrieval augmented generation,” inProceedings of the 18th
Conference of the European Chapter of the Association for Computational
Linguistics: System Demonstrations, 2024, pp. 150–158.
[4]J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “ARES: An
Automated Evaluation Framework for Retrieval-Augmented Generation
Systems,” inProceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), 2024, pp. 338–354.
[5]R. Fayyazi, M. Zuzak, and S. J. Yang, “LLM Embedding-based Attri-
bution (LEA): Quantifying Source Contributions to Generative Model’s
Response for Vulnerability Analysis,”arXiv preprint arXiv:2506.12100,
2025.
[6]H. Hu, C. He, X. Xie, and Q. Zhang, “LRP4RAG: Detecting Halluci-
nations in Retrieval-Augmented Generation Via Layer-Wise Relevance
Propagation,”Available at SSRN 5199254.
[7]L. M. Paes, D. Wei, H. J. Do, H. Strobelt, R. Luss, A. Dhurandhar,
M. Nagireddy, K. N. Ramamurthy, P. Sattigeri, W. Geyeret al., “Multi-
level explanations for generative language models,” inProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), 2025, pp. 32 291–32 317.
[8]F. Chazal and B. Michel, “An Introduction to Topological Data Analysis:
Fundamental and Practical Aspects for Data Scientists,”Frontiers in
Artificial Intelligence, vol. 4, p. 667963, 2021.
[9]J. R. Munkres, S. G. Krantz, and H. R. Parks,Elements of Algebraic
Topology. CRC press, 2025.
[10] D. Anick, “The Computation of Rational Homotopy Groups is #P-Hard,”
Lecture Notes in Pure and Applied Mathematics, vol. 114, pp. 1–56, 01
1989.
[11] E. Paluzo-Hidalgo, “Latent Space Topology Evolution in Multilayer
Perceptrons,”arXiv preprint arXiv:2506.01569, 2025.
[12] Y . Harun, K. Lee, J. Gallardo, G. Krishnan, and C. Kanan, “What
Variables Affect Out-of-Distribution Generalization in Pretrained Models?”
Advances in Neural Information Processing Systems, vol. 37, pp. 56 479–
56 525, 2024.
[13] Y . Gardinazzi, K. Viswanathan, G. Panerai, A. Ansuini, A. Cazzaniga,
and M. Biagetti, “Persistent Topological Features in Large Language
Models,” inInternational Conference on Machine Learning. PMLR,
2025, pp. 18 811–18 830.
[14] A. Fay, I. Garc ´ıa-Redondo, Q. Wang, H. Dubossarsky, and A. Monod,
“The Shape of Adversarial Influence: Characterizing LLM Latent Spaces
with Persistent Homology,” inInternational Conference on Learning
Representations, vol. 2026, 2026, pp. 146 129–146 175.[15] D. Li, Z. Sun, X. Hu, Z. Liu, Z. Chen, B. Hu, A. Wu, and M. Zhang,
“A Survey of Large Language Models Attribution,”arXiv preprint
arXiv:2311.03731, 2023.
[16] M. T. Ribeiro, S. Singh, and C. Guestrin, ““Why Should I Trust You?”
Explaining the Predictions of Any Classifier,” inProceedings of the 22nd
ACM SIGKDD International Conference on Knowledge Discovery and
Data Mining, 2016, pp. 1135–1144.
[17] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting
Model Predictions,”Advances in Neural Information Processing Systems,
vol. 30, 2017.
[18] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. M ¨uller, and
W. Samek, “On Pixel-Wise Explanations for Non-Linear Classifier
Decisions by Layer-Wise Relevance Propagation,”PloS one, vol. 10,
no. 7, p. e0130140, 2015.
[19] R. Achtibat, S. M. V . Hatefi, M. Dreyer, A. Jain, T. Wiegand,
S. Lapuschkin, and W. Samek, “AttnLRP: Attention-Aware Layer-Wise
Relevance Propagation for Transformers,” inProceedings of the 41st
International Conference on Machine Learning, 2024, pp. 135–168.
[20] P. Sekuloski, D. Kitanovski, I. Goshev, K. Mishev, M. S. Misheva, and
V . D. Ristovska, “Exploring the Potential of Topological Data Analysis for
Explainable Large Language Models: A Scoping Review,”Mathematics,
vol. 14, no. 2, p. 378, 2026.
[21] O. Draganov and S. Skiena, “The Shape of Word Embeddings: Quan-
tifying Non-Isometry with Topological Data Analysis,” inFindings of
the Association for Computational Linguistics: EMNLP 2024, 2024, pp.
12 080–12 099.
[22] F. Rottach, W. Rudman, B. Rieck, H. Scells, and C. Eickhoff, “From
Topology to Retrieval: Decoding Embedding Spaces with Unified
Signatures,”arXiv preprint arXiv:2511.22150, 2025.
[23] S. Fitz, P. Romero, and J. J. Schneider, “Hidden Holes: Topological
Aspects of Language Models,”arXiv preprint arXiv:2406.05798, 2024.
[24] G. Carlsson, “Topology and Data,”Bulletin of the American Mathematical
Society, vol. 46, no. 2, pp. 255–308, 2009.
[25] Edelsbrunner, Letscher, and Zomorodian, “Topological Persistence and
Simplification,”Discrete & Computational Geometry, vol. 28, no. 4, pp.
511–533, 2002.
[26] J.-C. Hausmann, “On the Vietoris–Rips Complexes and a Cohomology
Theory,” inProspects in Topology: Proceedings of a Conference in Honor
of William Browder, no. 138. Princeton University Press, 1995, p. 175.
[27] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer, “Stability of Persistence
Diagrams,” inProceedings of the Twenty-First Annual Symposium on
Computational Geometry, 2005, pp. 263–271.
[28] A. Zomorodian and G. Carlsson, “Computing Persistent Homology,”
inProceedings of the Twentieth Annual Symposium on Computational
Geometry, 2004, pp. 347–356.
[29] E. Munch, “Applications of Persistent Homology to Time Varying
Systems,” Ph.D. dissertation, Duke University, 2013.
[30] R. Ghrist, “Barcodes: the Persistent Topology of Data,”Bulletin of the
American Mathematical Society, vol. 45, no. 1, pp. 61–75, 2008.
[31] G. Carlsson, A. Zomorodian, A. Collins, and L. Guibas, “Persistence
Barcodes for Shapes,” inProceedings of the 2004 Eurographics/ACM
SIGGRAPH Symposium on Geometry Processing, 2004, pp. 124–135.
[32] F. Chazal, D. Cohen-Steiner, M. Glisse, L. J. Guibas, and S. Y . Oudot,
“Proximity of Persistence Modules and Their Diagrams,” inProceedings
of the twenty-fifth annual symposium on Computational geometry, 2009,
pp. 237–246.
[33] Y . Mileyko, S. Mukherjee, and J. Harer, “Probability Measures on the
Space of Persistence Diagrams,”Inverse Problems, vol. 27, no. 12, p.
124007, 2011.
[34] D. Cohen-Steiner, H. Edelsbrunner, J. Harer, and Y . Mileyko, “Lipschitz
Functions Have L p-Stable Persistence,”Foundations of Computational
Mathematics, vol. 10, no. 2, pp. 127–139, 2010.
[35] A. Arulandu, D. Gottschalk, T. Payne, A. Richardson, and T. Weighill,
“Through the Grapevine: Vineyard Distance as a Measure of Topological
Dissimilarity,”arXiv preprint arXiv:2510.24472, 2025.
[36] L. Yang, “Persistent Homology for Distribution Drift Detection in LLM
Embedding Streams,” inThe ICML 2026 Workshop on Hypothesis Testing.
[37] Orange-Cyberdefense, “Game of Active Directory (GOAD),” 2025.
[Online]. Available: https://github.com/Orange-Cyberdefense/GOAD
[38] Wazuh, “Wazuh: The Open Source Security Platform),” 2025. [Online].
Available: https://wazuh.com/

APPENDIX
A. TAD results based on H1 homology
In Table IV, we show the TAD values based on H1 homology.
As can be seen, the results are generally competitive in both
settings, with H1 being higher in F1-score for the “Indirect”
case for all four models. In addition, H1 results are also slightly
higher for larger models, which could indicate that H1 can
capture more abstract or higher-order representations as model
capacity increases. However, due to H1 being computationally
more expensive (cubic) than H0 (quadratic), we used H0 as
the default setting for TAD.
B. TAD Specific Examples
To demonstrate in more detail on how TAD actually attributes
the logs to the response, we provided a detailed qualitative
example in Figure 3. As can be seen, TAD correctly attributes
log-35 with the highest contribution into the LLM’s response,
with a total Wasserstein of 1951.13 and containing the largest
gap (1162.72) compared to the other logs.
In addition, we provided three specific examples for the
“Direct”, “Regular”, and “Indirect” cases. The results show
how TAD correctly attributes all cases, and interestingly, for
the indirect case, where there is no keyword similarity, TAD
still managed to attribute to the correct log.
Direct Case – Gemma3-4B – TAD
LLM Output
At 03:36:16.781 , agent castelblack
(192.168.56.22 ) logged “Registry Value
Entry Added to the System. ” A registry value
associated with COLIncrease.exe , located on
user robb.stark ’s Desktop, was added under the
Windows bam\State\UserSettings registry path
inscheduledmode.
TAD’s Traced Log
log_id=iLZX0ZkBDpTsYhzSRWvf
timestamp=2025-10-11T03:36:16.781Z
agent_name=castelblack
agent_ip=192.168.56.22
rule_description="Registry Value Entry Added
to the System"
target_user=null
target_domain=null
full_log="Registry Value ’[x32]
HKEY_LOCAL_MACHINE\System\
CurrentControlSet\Services\bam\State\
UserSettings\S-1-5-21-3733740424-\\
1936739333-3027215852-1115\\Device\
HarddiskVolume1\Users\robb.stark\Desktop\
COLIncrease.exe’ added Mode: scheduled"Indirect Case – Qwen3-4B – TAD
LLM Output
An execution ledger on this box shows that a well-known
utility for launching programs on other boxes was run
from a fetch folder in a staff profile. Storing that kit in a
personal folder rather than an approved software area,
then running it, points to an outsider preparing to reach
further boxes.
TAD’s Traced Log
log_id=B7YP0pkBDpTsYhzSDo05
timestamp=2025-10-11T06:57:04.447Z
agent_name=npc-petyerbaeli
agent_ip=192.168.56.154
rule_description="Registry Value Entry Added
to the System"
target_user=null
target_domain=null
full_log="Registry Value ’[x32]
HKEY_LOCAL_MACHINE\System\
CurrentControlSet\Services\bam\State\
UserSettings\S
-1-5-21-3733740424-1936739333-3027215852
-1115\\Device\HarddiskVolume1\Users\robb.
stark\Downloads\PsExec64.exe’ ..."
Regular Case – Granite4.1-8B – TAD
LLM Output
WINTERFELL accepts a successful remote logon for
robb.stark at 18:57, authenticated with NTLM from
NPC-PETYERBAELI . Presenting a hash rather than
a typed password, and doing so against a domain
controller from an ordinary workstation, is the signature
of a replayed credential. This is unauthorized lateral
movement rather than legitimate remote administration.
TAD’s Traced Log
log_id=FbV8z5kBDpTsYhzSINoU
timestamp=2025-10-10T18:57:20.490Z
agent_name=winterfell
agent_ip=192.168.56.11
rule_description="Successful Remote Logon
Detected - User:\robb.stark - NTLM
authentication, possible pass-the-hash
attack... Verify that NPC-PETYERBAELI is
allowed to perform RDP connections"
target_user=robb.stark
target_domain=NORTH
system_message="\"An account was successfully
logged on. ... Security ID: S
-1-5-21-37337...1115 Account Name: robb.
stark Account Domain: NORTH ...
Workstation Name: NPC-PETYERBAELI Source
Network Address: 192.168.56.154 ...
Package Name (NTLM only): NTLM V2 Key
Length....\""

TABLE IV: TAD comparison based on H0 & H1 homology.
Case MethodQwen3-4B Gemma-3-4B Qwen2.5-7B Granite-4.1-8B
Accuracy F1-score Accuracy F1-score Accuracy F1-score Accuracy F1-score
Direct TAD (H0) 0.98470.7097 0.9779 0.6061 0.98300.6667 0.98300.6667
TAD (H1) 0.9847 0.7273 0.9710 0.5405 0.9830 0.7059 0.9830 0.7222
Regular TAD (H0) 0.9847 0.7097 0.9727 0.5000 0.9796 0.6000 0.9813 0.6452
TAD (H1)0.9830 0.6667 0.9438 0.2667 0.9830 0.6667 0.9847 0.7097
Indirect TAD (H0)0.9796 0.6000 0.94380.1538 0.97610.5625 0.96590.3750
TAD (H1) 0.9813 0.6452 0.9438 0.2326 0.9761 0.5882 0.96080.4103
TAD’s Qualitative Example
1. Analyst query
Examine logs around 06:56–06:57 on 2025-10-11. Is there evidence of an attack in these logs?
2. Candidate incident logs after the Screen process (refer to Algorithm 1)
Log Event State Total-Wasserstein Gap
31 BAM registry checksum changed forpowershell.exe. Benign682.54 0.0
32 BAM registry checksum changed forcmd.exe. Benign 708.30 25.75
29 BAM registry checksum changed forApplicationFrameHost.exe. Benign710.32 2.03
34 BAM registry entry added forrundll32.exe. Benign 749.56 39.24
30 BAM registry checksum changed forwindows.immersivecontrolpanel. Benign782.09 32.53
33 BAM registry entry added for temporaryUn.exe. Benign 788.61 6.52
35 BAM registry entry added for PsExec64.exe inrobb.stark ’s Downloads. Malicious 1951.33 1162.72
3. LLM response (Granite-4.1-8B) for the regular case
“NPC-PETYERBAELI records a Background Activity Moderator entry at 06:57 showing PsExec64.exe ran from robb.stark ’s Downloads folder. PsExec spawns
processes on remote machines, and a copy staged in a download folder rather than an approved install path indicates an operator preparing to execute code on other hosts.”
4. Evidence localization
Family Method Result Operational consequence
Segment-level attribution TAD (ours) ✓Log 35 Uniquely identifies the log mostly influencing the response geometry
Fig. 3: TAD qualitative example to demonstrate how Wasserstein distance measures each log’s effect on the response geometry.
C. LLM Self-Eval Baseline Prompts
Below we provide the prompts used for the LLM-based
self-assessment baselines (LLM-as-Judge and in-line Citation).
These prompts are designed to evaluate the extent to which the
LLM can map the generated responses to the source logs and
attribute their outputs to the specific retrieved segments that
support them. We used greedy decoding for these cases to get
deterministic responses. However, a key limitation of LLM-
based judges is their sensitivity to engineering of prompts, as
variations in prompt wording or structure can lead to different
evaluation outcomes. TAD, on the other hand, does not rely
on a secondary set of evaluation prompts to trace the logs and
will only look at the response geometry.In-line Citation Prompt
{formatted logs}
Here is your response based on these logs:
"""{response}"""
For every sentence in the response, add inline citations
to ALL logs that support that sentence.
- If multiple logs support a sentence, cite all of them.
- Do not leave factual claims uncited.
Output format: Add citations with <<logN>> at the
end of each sentence.

LLM-as-Judge Prompt
You are an expert at analyzing cybersecurity incident
responses.
Below are security logs, each tagged with
<<logN>>...<<logN/>>:
{formatted logs}
You already examined these logs and produced this
response:
"""
{response}
"""
TASK: Identify which specific logs you relied upon to
produce this response.
RULES:
1. Only list logs that directly contributed to the response’s
conclusions
2. Output ONLY the log tags that were used (e.g.,
<<log5>>,<<log12>>)
3. Do not include logs that were present but not
referenced in the response
Available log tags:{available tags}
OUTPUT FORMAT:
List the relevant log tags, one per line:
<<logX>>
<<logY>>
RELEVANT LOGS: