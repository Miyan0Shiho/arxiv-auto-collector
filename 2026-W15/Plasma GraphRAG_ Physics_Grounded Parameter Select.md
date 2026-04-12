# Plasma GraphRAG: Physics-Grounded Parameter Selection for Gyrokinetic Simulations

**Authors**: Ruichen Zhang, Feda AlMuhisen, Chenguang Wan, Zhisong Qu, Kunpeng Li, Youngwoo Cho, Kyungtak Lim, Virginie Grandgirard, Xavier Garbet

**Published**: 2026-04-07 10:04:36

**PDF URL**: [https://arxiv.org/pdf/2604.06279v1](https://arxiv.org/pdf/2604.06279v1)

## Abstract
Accurate parameter selection is fundamental to gyrokinetic plasma simulations, yet current practices rely heavily on manual literature reviews, leading to inefficiencies and inconsistencies. We introduce Plasma GraphRAG, a novel framework that integrates Graph Retrieval-Augmented Generation (GraphRAG) with large language models (LLMs) for automated, physics-grounded parameter range identification. By constructing a domain-specific knowledge graph from curated plasma literature and enabling structured retrieval over graph-anchored entities and relations, Plasma GraphRAG enables LLMs to generate accurate, context-aware recommendations. Extensive evaluations across five metrics, comprehensiveness, diversity, grounding, hallucination, and empowerment, demonstrate that Plasma GraphRAG outperforms vanilla RAG by over $10\%$ in overall quality and reduces hallucination rates by up to $25\%$. {Beyond enhancing simulation reliability, Plasma GraphRAG offers a methodology for accelerating scientific discovery across complex, data-rich domains.

## Full Text


<!-- PDF content starts -->

1
Plasma GraphRAG: Physics-Grounded Parameter
Selection for Gyrokinetic Simulations
Ruichen Zhang, Feda AlMuhisen, Chenguang Wan, Zhisong Qu, K unpeng Li, Youngwoo Cho, Kyungtak Lim,
Virginie Grandgirard, and Xavier Garbet
Abstract —Accurate parameter selection is fundamental to
gyrokinetic plasma simulations, yet current practices rel y heav-
ily on manual literature reviews, leading to inefﬁciencies and
inconsistencies. We introduce Plasma GraphRAG, a novel fra me-
work that integrates Graph Retrieval-Augmented Generatio n
(GraphRAG) with large language models (LLMs) for automated ,
physics-grounded parameter range identiﬁcation. By const ruct-
ing a domain-speciﬁc knowledge graph from curated plasma
literature and enabling structured retrieval over graph-a nchored
entities and relations, Plasma GraphRAG enables LLMs to
generate accurate, context-aware recommendations. Exten sive
evaluations across ﬁve metrics, comprehensiveness, diver sity,
grounding, hallucination, and empowerment, demonstrate t hat
Plasma GraphRAG outperforms vanilla RAG by over 10% in
overall quality and reduces hallucination rates by up to 25%.
Beyond enhancing simulation reliability, Plasma GraphRAG
offers a methodology for accelerating scientiﬁc discovery across
complex, data-rich domains.
Index Terms —Gyrokinetic, large language model, graph
retrieval-augmented generation, agent, graphRAG
I. I NTRODUCTION
Understanding turbulence and transport in magnetically
conﬁned plasmas remains one of the central challenges in
fusion energy research [1]. Gyrokinetic (GK) simulations a re
indispensable for modeling these multiscale phenomena, as
they capture the kinetic behavior of charged particles whil e re-
ducing the dimensionality of the full six-dimensional Vlas ov–
Maxwell system [2]. Modern GK codes are widely used to
investigate plasma stability, turbulence-driven transpo rt, and
This research is supported by the National Research Foundat ion, Singapore.
This research is also supported by Seatrium New Energy Labor atory, Singa-
pore Ministry of Education (MOE) Tier 1 (RT5/23 and RG24/24) , the Nanyang
Technological University (NTU) Centre for Computational T echnologies in
Finance (NTU-CCTF), and the Research Innovation and Enterp rise (RIE) 2025
Industry Alignment Fund - Industry Collaboration Projects (IAF-ICP) (Award
I2301E0026), administered by Agency for Science, Technolo gy and Research
(A*STAR). (Corresponding author: Xaiver Garbet)
Ruichen Zhang and Kunpeng Li are with the School of Physical a nd Mathe-
matical Sciences, Nanyang Technological University, Sing apore, and also with
the College of Computing and Data Science, Nanyang Technolo gical Univer-
sity, Singapore (e-mail: ruichen.zhang@ntu.edu.sg; kunp eng.li@ntu.edu.sg).
Feda AlMuhisen and Virginie Grandgirard are with CEA, IRFM, F-
13108 Saint Paul-lez-Durance, France (e-mail: virginie.g randgirard@cea.fr;
feda.almuhisen@cea.fr).
Chenguang Wan, Zhisong Qu, Youngwoo Cho, and Kyungtak
Lim are with the School of Physical and Mathematical Science s,
Nanyang Technological University, Singapore. (e-mail: ch en-
guang.wan@ntu.edu.sg; zhisong.qu@ntu.edu.sg; youngwoo .cho@ntu.edu.sg;
kyungtak.lim@ntu.edu.sg).
Xaiver Garbet is with the School of Physical and Mathematica l Sciences,
Nanyang Technological University, Singapore, and also wit h CEA, IRFM,
F-13108 Saint Paul-lez-Durance, France (e-mail: xavier.g arbet@ntu.edu.sg).performance limits in devices ranging from tokamaks to stel -
larators [3]. These tools form the computational backbone o f
fusion research, providing insight into conﬁnement optimi za-
tion, transport barrier formation, and scenario developme nt for
future reactors.
A critical prerequisite for accurate GK simulations is the
selection of appropriate input parameter ranges, includin g
normalized temperature and density gradients, safety fact ors,
magnetic shear, and collisionality. These parameters have a
decisive impact on turbulence characteristics and transpo rt
predictions, and their accurate speciﬁcation is essential not
only for predictive modeling but also for the construction
of surrogate models and databases. Traditionally, identif ying
suitable parameter ranges has relied on expert judgment and
manual reviews of experimental and theoretical literature . This
manual process is time-consuming, error-prone, and difﬁcu lt to
reproduce, which results in inconsistencies across benchm ark-
ing studies and undermine conﬁdence in simulation outcomes .
Recent progress in artiﬁcial intelligence (AI), particula rly
large language models (LLMs), has opened new possibilities
for automating knowledge extraction from unstructured sci en-
tiﬁc literature [4]. An LLM is a type of deep learning model
trained on massive corpora to capture statistical patterns of
language, enabling it to perform tasks such as summarizatio n,
reasoning, and question answering. To enhance domain speci -
ﬁcity, LLMs are often combined with Retrieval-Augmented
Generation (RAG), a paradigm where relevant documents are
ﬁrst retrieved from a knowledge base and then supplied as
context to the LLM during generation [5]. While RAG has
demonstrated effectiveness in many ﬁelds, standard implem en-
tations treat the literature as a ﬂat corpus, failing to capt ure
thestructured interdependencies among physical variables that
are central to plasma physics [6]. This limitation often lea ds
to hallucinations or incomplete recommendations, particu larly
in highly technical applications such as gyrokinetic model ing.
To address these challenges, we introduce Plasma
GraphRAG , a novel framework that integrates Graph
Retrieval-Augmented Generation (GraphRAG) with LLMs
to automate the identiﬁcation of physics-grounded param-
eter ranges for gyrokinetic simulations. Plasma GraphRAG
builds a domain-speciﬁc knowledge graph, where nodes rep-
resent plasma parameters, device metadata, and bibliograp hic
sources, while edges encode physical couplings and co-
occurrence relations. Structured retrieval over this grap h pro-
vides the LLM with explicit relational context, enabling ac -
curate, transparent, and reproducible parameter recommen da-arXiv:2604.06279v1  [physics.plasm-ph]  7 Apr 2026

2
tions while signiﬁcantly mitigating hallucinations1. The key
contributions of this study are summarized as follows:
•We introduce Plasma GraphRAG, which constructs a
domain-speciﬁc graph for GK modeling by harmoniz-
ing diverse literature sources into a uniﬁed, code-facing
parameter space. By adopting standardized gyrokinetic
notation and explicitly linking parameters with biblio-
graphic evidence, the graph resolves long-standing in-
consistencies across simulation codes and provides a
reproducible foundation for parameter integration.
•Plasma GraphRAG employs a retrieval mechanism that
encodes physical couplings and co-occurrence relations
among plasma parameters. Compared to standard RAG,
this structured retrieval provides the LLM with richer
context, enabling more accurate parameter extraction and
improved interpretability through transparent evidence
paths, while substantially reducing hallucinations.
•We evaluate Plasma GraphRAG on controlled GK pa-
rameter identiﬁcation benchmarks, showing improved re-
sponse quality, diversity, and grounding relative to base-
line methods. The framework streamlines the preparation
of surrogate model databases, alleviates expert workload,
and offers a scalable pathway toward reproducible, data-
driven discovery in plasma turbulence and transport stud-
ies. However, the current benchmark remains limited in
scope, and the evaluation metrics are primarily heuristic,
suggesting that larger-scale and quantitatively validate d
studies will be essential in future work.
II. R ELATED WORK
This section reviews prior work in two relevant areas: (i) GK
plasma physics and turbulence modeling, and (ii) LLMs with
retrieval-based reasoning. The former outlines the evolut ion
of GK simulation tools and surrogate modeling techniques,
while the latter focuses on RAG and GraphRAG approaches
for scientiﬁc question answering.
A. Gyrokinetic Plasma Physics
GK theory has been the cornerstone of turbulence and
transport modeling in magnetically conﬁned plasmas for sev -
eral decades. Foundational work has established the nonlin ear
GK formalism for low-frequency waves, which underpins
modern turbulence simulations [7]. Since then, dedicated local
solvers such as GS2,GENE ,CGYRO , andGKW have been
developed and widely adopted to model ion- and electron-sca le
turbulence, trapped electron modes, and zonal ﬂow dynamics
[8], [9]. In parallel, global gyrokinetic codes such as GYSELA ,
ORB5 , andGT5D have been introduced to simulate turbulence
and transport across the entire tokamak volume, capturing
ﬁnite-radius effects and global proﬁle variations [10]. Th ese
codes have enabled benchmarked studies of core transport
phenomena, including the Dimits shift, and are now integral to
1In this study, hallucination refers to cases where the model generates
information that is inconsistent with the retrieved eviden ce or established
physical principles. A lower hallucination rate therefore indicates higher
factual consistency and physical reliability in the model’ s responses.scenario development for fusion devices. In parallel, redu ced-
order models such as Trapped Gyro-Landau Fluid (TGLF) [11]
and quasilinear models like QuaLiKiz [3] have been intro-
duced to bridge detailed GK physics and integrated transpor t
modeling. These models have achieved substantial speedups
at the cost of some ﬁdelity, enabling broader exploration of
design spaces.
More recently, the ﬁeld has embraced hybrid and data-
driven modeling to address computational bottlenecks. Neu ral
network surrogates trained on large GK datasets [9] have
been developed to provide ﬂux predictions several orders of
magnitude faster than ﬁrst-principles simulations. Gener ative
models [12] and transformer-based 5D surrogates [13] have
further enabled direct emulation of nonlinear turbulence w ith
preserved spatiotemporal dynamics. Multi-ﬁdelity method s
have been proposed to combine reduced models with sparse
high-ﬁdelity GK data for improved predictive accuracy [14] .
Ongoing benchmarking efforts have continued to validate
solvers such as GENE andCGYRO under emerging physics
regimes [15]. This evolution reﬂects a growing trend toward
surrogate-augmented, data-driven workﬂows for turbulenc e-
informed fusion scenario design.
B. LLMs with Retrieval
LLMs have demonstrated impressive capabilities across a
range of NLP tasks, but they continue to struggle with factua l
accuracy and domain speciﬁcity. To address these limita-
tions, retrieval-augmented generation (RAG) has emerged a s
a promising approach. Early systems such as DrQA [16]
and REALM [17] have shown that integrating external re-
trieval with neural models improves factual grounding in
open-domain question answering. The formal RAG framework
introduced by [18] has combined dense vector retrievers
with encoder-decoder models like BART and T5 to produce
contextualized answers using retrieved documents. Howeve r,
standard RAG approaches have treated the underlying corpus
as a ﬂat collection of documents, ignoring structured rela-
tionships such as scientiﬁc couplings, units, or hierarchi cal
parameter dependencies. This design choice has limited the ir
effectiveness in technical domains, where relational reas oning
and symbolic consistency are critical.
To address these shortcomings, recent work has explored
GraphRAG, a class of models that incorporate structured
knowledge representations [19]. These methods have embed-
ded queries into graph spaces, retrieved subgraphs compose d
of entities and their relations, and linearized the result i nto
evidence paths for LLMs [20]. Applications have spanned
enterprise QA, manufacturing documents, and e-commerce
workﬂows [21], demonstrating improved interpretability a nd
accuracy. In the scientiﬁc domain, systems such as PaperQA
[22] have adapted RAG to scholarly literature, yielding en-
hanced factual coverage and traceable citations. Benchmar king
studies [23] have reported that GraphRAG excels at multi-ho p
reasoning and relationship synthesis. Nonetheless, chall enges
remain. Graph construction can introduce noise, especiall y in
heterogeneous corpora, and task-speciﬁc graph schemas mus t
often be hand-designed. Survey work [24] has highlighted

3
hybrid architectures that combine both ﬂat and graph-based
retrieval to balance coverage, precision, and system compl ex-
ity. Taken together, the above work suggests that GraphRAG
can provide a promising foundation for LLM-based reason-
ing in structured scientiﬁc domains such as plasma physics.
In this context, we propose Plasma GraphRAG, a domain-
speciﬁc GraphRAG framework tailored to GK simulations.
Our method leverages structured graphs to support repro-
ducible, well-grounded parameter recommendations derive d
from literature evidence, addressing both domain complexi ty
and LLM hallucination risk.
III. P LASMA GRAPH RAG F RAMEWORK
This section presents the architecture of Plasma GraphRAG,
our proposed framework for automated, physics-grounded
parameter range identiﬁcation in GK simulations.
A. Data Collection and Physics-Grounded Preprocessing
The ﬁrst stage of Plasma GraphRAG is the construction of
a physics-grounded corpus tailored to GK parameter identiﬁ -
cation. This step deﬁnes the scope and quality of downstream
knowledge graph construction and retrieval. We curate a
structured dataset focused on descriptors that are standar d in
core transport and GK code-comparison studies, following t he
normalized notation of Bourdelle et al. [25]. At this stage, no
numerical thresholds or scan protocols are ﬁxed, and those a re
instantiated later during benchmark evaluation.
The variable families span geometry and magnetic
equilibrium (q, s, R 0/a, r/a ),thermodynamics and
composition (Ti/Te, Zeﬀ),transport-driving gradients
(R/Ln, R/LTe, R/LTi), and kinetic or stability
proxies (γE×B(a/cs), νei(a/cs)) used in core turbulence
analyses [26]. These variables are mapped into a standardiz ed
feature space, as deﬁned in Eq. (1), ensuring a consistent
input–output interface with GK codes and modeling pipeline s.
[26]. These variables are mapped into a standardized featur e
spaceXGK, i.e.,
XGK=

q, s, R 0/a, r/a,
Ti/Te, Zeﬀ,
R/Ln, R/LTe, R/LTi,
γE×B(a/cs), νei(a/cs),
[βe, ρ∗] (Optional)

, (1)
where additional dimensionless quantities such as βeand
ρ∗are included when available. This harmonized feature set
ensures consistent input/output interface with GK codes an d
modeling pipelines.
To ensure physical integrity of the dataset, a harmoniza-
tion process is applied that isolates quasi-steady-state c ore
intervals, reconciles units and normalization schemes acr oss
sources, and removes tuples that are either incomplete or
inconsistent with standard GK usage. This process yields a
clean subset, i.e.,
Dclean=/braceleftbig
x∈D/vextendsingle/vextendsingleVGK(x;{C,N,Q})/bracerightbig
, (2)whereCensures completeness, Nenforces unit-normalization
coherence, andQrestricts to valid quasi-steady-state pro-
ﬁles. The predicate VGKoperationalizes these quality criteria
without binding to any operating point. Each predicate in
VGKis veriﬁed as follows. Completeness (C) ensures that
all core variables in Eq. (1) are present; records missing
geometry, gradient, or thermodynamic terms are discarded.
Normalization coherence (N) checks unit consistency and
normalization to machine parameters (e.g., R0,cs), reconcil-
ing mixed units or gradient deﬁnitions. Quasi–steady-state
validity (Q) requires temporal stability, retaining data where
relative variations |˙X/X|<5%over several conﬁnement
times. Together, these ﬁlters ensure physically consisten t and
reproducible parameter tuples for graph-based analysis.
Device-speciﬁc parameters are then transformed into the
uniﬁed feature space (1) via a deterministic formatting op-
eratorF(·). Here,F(·)denotes a deterministic formatting
operator that converts heterogeneous, device-speciﬁc qua n-
tities into the uniﬁed feature space deﬁned in Eq. (1). It
standardizes variable names, applies normalization rules (e.g.,
toR0,a, andcs), and reformats derived quantities such
asR/Ly=−(R/y)(∂y/∂r)to ensure consistency across
all sources. For instance, normalized gradient lengths are
computed from experimental proﬁle data using


Dfmt=F(Dclean),
R
Ly=−R
y∂y
∂r,(3)
wherey∈ {n, Te, Ti}. All notation and normalization
choices conform to the conventions adopted in cross-code
GK/transport benchmarks [25].
To facilitate reproducibility and transparency in retriev al
and generation, we summarize internal statistics over XGK
coordinates, which is given by Eq. (4)


µ=1
nn/summationdisplay
i=1xi,
σ=/radicaltp/radicalvertex/radicalvertex/radicalbt1
n−1n/summationdisplay
i=1(xi−µ)2,(4)
wherexidenotes the i-th sample in a given coordinate. These
corpus-level statistics support interpretability, norma lization,
and coverage estimation in the later GraphRAG retrieval
process.
B. LLM with GraphRAG
The second stage of Plasma GraphRAG integrates large
language models with structured retrieval over a typed pa-
rameter graph tailored for GK simulations. As illustrated i n
Figure 1, the system processes raw plasma literature into
semantic chunks and extracts entities and relations to buil d
a physics-grounded knowledge graph. This procedure is semi -
automated: initial entity and relation extraction is perfo rmed
using a domain-adapted NLP pipeline based on named-entity
recognition (NER) and dependency parsing, while manual
validation and correction are applied to ensure physical co n-
sistency and symbol standardization. In principle, the pip eline

4
Fig. 1. LLM-guided parameter range recommendation grounde d in structured retrieval.
can operate fully automatically for large-scale ingestion , but
expert-in-the-loop curation remains essential to maintai n ac-
curacy in highly technical contexts. Given a natural-langu age
query, relevant parameter nodes are retrieved and expanded
into ad-hop evidence subgraph, which is then linearized
and provided to the LLM [27]. The model generates multi-
ple answer candidates, guided by a reranking objective that
promotes evidence coverage and factual consistency. When
evidence is insufﬁcient, the system abstains or returns a lo w-
conﬁdence response. This section details each component of
the pipeline, including graph indexing, retrieval, linear ization,
and generation.
1) Graph-Based Indexing: Plasma GraphRAG constructs
a typed, text-attributed graph G= (V,E,T)that encodes
domain knowledge for gyrokinetic modeling. Nodes v∈V
are typed via a map τ:V→{param,dev,src,...}, where
parameter nodes are deﬁned as
P={v∈V|τ(v) =param}. (5)
Each parameter node v∈Pcarries an attribute xv=T(v)that
aggregates deﬁnitional text, ﬁgure captions, and bibliogr aphic
snippets. Prior to attribution, a normalization layer Nsym(NER
and regex-based) standardizes symbols and aliases.
Text Embeddings: Descriptions xvare embedded using a
sentence-level encoder, such as SentenceBERT [28], yieldi ng
zv= LM(xv)∈Rd, v∈P. (6)
Stacked embeddings form the parameter matrix, i.e.,
Z=
z⊤
p1
z⊤
p2...
z⊤
p|P|
∈R|P|×d, P={p1,...,p |P|}. (7)Typed Edges and Weights: The overall edge set is the
union of relation-speciﬁc subsets, i.e.,
E=/uniondisplay
r∈RE(r),R={r1,r2,r3,r4,...}, (8)
withr1=co-mention ,r2=deﬁnition -link,r3=
physical -coupling , andr4=table -row. For parame-
ter–parameter pairs (pi,pj)∈P×P, the edge weight is
computed as
w(pi,pj) =αsim(zpi,zpj)+βcooc(pi,pj)+γphys(pi,pj),
(9)
whereα+β+γ= 1and each term captures semantic similar-
ity, co-mention frequency, or documented physical couplin g,
respectively. The resulting adjacency matrix is given by
Aij=w(pi,pj), A∈R|P|×|P|. (10)
Thresholding and top- ksparsiﬁcation may be applied to derive
a binary graph when needed.
Normalization and Storage: For downstream diffusion and
aggregation, a symmetric normalization is applied as
/braceleftBigg
/tildewideA=D−1
2(A+I)D−1
2,
D= diag((A+I)1).(11)
The index stores both the raw and normalized graphs: (Z,A)
and(Z,/tildewideA), which expose semantic structure and domain-
speciﬁc relations for GraphRAG-based retrieval. This com-
pletes the graph-based indexing layer.
2) Graph-Guided Retrieval: This module selects a query-
speciﬁc, topology-aware evidence subgraph that captures b oth
semantic relevance and domain-speciﬁc relations, serving as
the structured context for generation. Given a natural-lan guage
queryxq, we ﬁrst embed it using the same sentence encoder
employed at indexing time, i.e.,
zq= LM(xq)∈Rd, (12)

5
To reﬂect the compositional nature of scientiﬁc queries, we
extract salient entities (e.g., parameter names, device fe atures)
fromxqand embed each individually, i.e.,
zei= LM(ei)∈Rd, e i∈Eq, (13)
yielding a query representation {ze1,...,z e|Eq|}aligned with
graph node embeddings.
Node-query relevance is computed via cosine similarity, i. e.,
cos(u,v) =u⊤v
/bardblu/bardbl2/bardblv/bardbl2, (14)
with the ﬁnal score per node p∈Pgiven by the best-aligned
entity embedding, i.e.,
score(p) = max
ei∈Eqcos/parenleftbig
zei, zp/parenrightbig
. (15)
Top-kparameter seeds are then selected as
Pk= TopKp∈Pscore(p). (16)
Here, the top- koperation selects the kparameter nodes with
the highest semantic similarity scores to the query, formin g
the initial seed set for graph expansion.
To incorporate inductive biases and neighborhood seman-
tics, a composite score is introduced, i.e.,
sθ(p|q) =λ1cos/parenleftbig
zq,zp/parenrightbig
+λ2τ(p/bardblq) +λ3max
u∈N(p)cos/parenleftbig
zq,zu/parenrightbig
,
(17)
whereλi≥0and/summationtext
iλi= 1 . The weights control the
contribution of different relevance components: λ1emphasizes
direct semantic similarity between the query and parameter
node,λ2adjusts the inﬂuence of the type prior τ(p/bardblq), and
λ3accounts for neighborhood similarity by aggregating con-
textual information from adjacent nodes N(p). The function
τ(p/bardblq)encodes a type prior (e.g., boosting parameters for
range-related queries), and N(p)denotes neighbors of node p.
Either the raw semantic score in Eq. (15) or the hybrid score
sθ(·)in Eq. (17) may be used to derive Pk.
To recover broader relational context, the selected seed se t
is expanded via a d-hop neighborhood, which is given by
/braceleftBigg
P∗=Pk∪N(d)(Pk),
N(d)(S) =∪p∈SN(d)(p),(18)
wheredis a tunable hyperparameter.
The ﬁnal evidence subgraph is the induced subgraph on this
expanded node set, i.e.,
G∗= induce/parenleftbig
P∗, E/parenrightbig
, (19)
which is then passed to the linearization and generation
modules. Formally, the retrieval objective can be cast as
G∗= arg max
G′⊆R(G)Sim/parenleftbig
xq, G′/parenrightbig
, (20)
whereR(G)is the feasible subgraph region (e.g., those
rooted at Pk) andSim(·,·)aggregates node-level afﬁnities and
optional coverage/diversity criteria. The choices of k,d, and
weightsλiare speciﬁed in the Experiments section.3) Answer Generation: Conditioned on the retrieved sub-
graphG∗= (P∗,E∗), the generator synthesizes responses
explicitly grounded in the linearized evidence. The query
and subgraph context are concatenated into a single input
sequence under a traversal policy π, ensuring stable ordering
and provenance retention, i.e.,
F(xq,G∗) =/bracketleftbig
lin(G∗;π) ;xq/bracketrightbig
. (21)
The generation objective is posed as conditional maximum
likelihood over the answer space A, which is given by
a∗= argmax
a∈Apφ/parenleftbig
a|F(xq,G∗)/parenrightbig
, (22)
with autoregressive factorization over tokens. To encoura ge
explicit grounding, decoding is guided by a reranking objec tive
that rewards coverage of retrieved evidence and penalizes
hallucinations or verbosity. Among an N-hypothesis setH, the
ﬁnal answer a†is selected by maximizing this reranking score
[29] When retrieval is insufﬁcient, the system abstains fro m
answering or marks the output as low-conﬁdence. Conﬁdence
is estimated from the aggregate relevance of the retrieved
nodes and the coverage of evidence cited in the response. Pos t-
processing further enforces citation consistency and outp ut
formatting to match plasma-physics conventions.
Algorithm 1 summarizes the end-to-end workﬂow of Plasma
GraphRAG. Starting from a natural-language query, the syst em
encodes entities, scores parameter nodes, and expands them
into a query-speciﬁc evidence subgraph. This subgraph is
linearized and combined with the query to form the input for
the generator, which produces multiple candidate answers. A
reranking mechanism then selects the ﬁnal output by jointly
maximizing likelihood and grounding while penalizing hal-
lucinations and verbosity. If retrieval signals are weak, t he
system abstains or assigns low conﬁdence, thereby maintain ing
reliability. Through this pipeline, Plasma GraphRAG ensur es
that parameter recommendations are both interpretable and
aligned with plasma-physics conventions.
IV. N UMERICAL RESULTS
A. Parameter Setting
To assess the effectiveness of Plasma GraphRAG in
physics-grounded parameter identiﬁcation, we constructe d a
controlled question-answering benchmark targeting key as -
pects of gyrokinetic modeling. The evaluation set comprise s 10
representative questions drawn from canonical literature and
simulation benchmarks, spanning four categories: (i) equi lib-
rium and geometry descriptors, (ii) thermodynamic ratios a nd
species composition, (iii) transport-driving gradients, and (iv)
stability and collisionality proxies. All questions are po sed
in natural language and require extraction or inference of
parameter ranges grounded in the source corpus.
We compare three models under identical retrieval con-
ﬁgurations: (1) GraphRAG with GPT-3.5-turbo , (2)
GraphRAG with LLaMA-3.1-8B , and (3) a vanilla RAG
baseline. The corpus consists of peer-reviewed gyrokineti c
studies normalized according to established conventions a nd
encoded into a text-attributed, typed parameter graph for s truc-
tured retrieval. We set the retrieval chunk size to 1200 toke ns

6
Fig. 2. Visualization of sample user interactions with the P lasma GraphRAG and sample response generations. The answer is generated using GPT-4o as the
generator.
with 100 tokens overlapping to ensure the construction of th e
knowledge graph captures sufﬁcient entities and relations hips
while retaining enough context information for interpreti ng
them properly. Responses are evaluated using ﬁve metrics:
Diversity ,Comprehensiveness ,Hallucination ,Directness , and
Empowerment . These metrics respectively assess coverage
breadth, factual grounding, linguistic clarity, and pract ical
usefulness, with hallucination deﬁned as any statement inc on-
sistent with the retrieved subgraph [30]. All evaluations u se
a temperature of 0.0 for deterministic outputs. We validate
reproducibility by running each evaluation three times.
B. Simulation Results
As illustrated in Figure 2, we present a case study highlight -
ing how Plasma GraphRAG facilitates grounded interactions
for gyrokinetic parameter exploration. In Scenario A, the u ser
seeks general knowledge about discharge modeling; the syst em
retrieves related entities, documents, and researchers, o ffer-
ing a comprehensive and citation-backed summary. Scenario
B and C delve into parameter-speciﬁc queries—identifying
key variables for turbulence and quantifying the range of
a speciﬁc parameter across simulations. In both cases, the
agent retrieves and linearizes a relevant evidence subgrap h
from the knowledge graph, enabling responses that are both
precise and interpretable. This showcases the agent’s abil ity to
support nuanced scientiﬁc inquiry through structured retr ieval,
minimizing hallucination while promoting traceability an d
domain consistency.
Figure 3 compares GraphRAG with GPT-4o against
the vanilla RAG baseline across ﬁve evaluation metrics.
GraphRAG achieves consistently higher scores in all cate-
gories, demonstrating its ability to capture a broader rang eof plasma parameters and deliver responses that are both
informative and actionable for simulations.
Diversity measures the variety of unique parameters cor-
rectly mentioned in each answer, while comprehensiveness
quantiﬁes the proportion of relevant parameters covered re l-
ative to ground-truth literature references. Hallucinati on cap-
tures the percentage of statements unsupported or contradi cted
by the retrieved subgraph, serving as an indicator of factua l
reliability. Directness reﬂects linguistic clarity and co ncise-
ness, computed as the inverse of average response length
normalized by informativeness, and empowerment evaluates
how actionable or simulation-ready the suggested paramete r
ranges are, as judged by expert annotators. All scores are
normalized to a 0–100 scale and averaged over ten benchmark
queries.
It also shows a 35.25% reduction in hallucinations, conﬁrm-
ing that graph-structured retrieval helps ground response s more
faithfully in the literature. Overall, the results highlig ht that
GraphRAG provides a more balanced and reliable framework,
better suited for accuracy, reproducibility, and interpre tability
in plasma parameter determination.
Figure 4 compares the performance of GraphRAG when
paired with Llama3.1-8B and GPT-4o across ﬁve evaluation
metrics, where higher scores indicate better performance. The
results show that GPT outperforms Llama across all metrics,
demonstrating its superior ability to generate broad, accu rate,
and well-grounded parameter recommendations. This improv e-
ment highlights GPT’s stronger reasoning capacity and rich er
contextual understanding, which enable it to capture com-
plex relationships among gyrokinetic parameters and produ ce
more informative and actionable responses. In contrast, Ll ama
delivers comparatively narrower and less detailed outputs ,
reﬂecting its limited contextual modeling capability. Ove rall,

7
Algorithm 1: Plasma GraphRAG: GraphRAG-based
Parameter Identiﬁcation
1:Input: Queryxq; graphG= (V,E,T)with parameter
nodesP⊆Vand embeddings{zp}p∈P;
hyperparameters k,d, traversal policy π; rerank weights
(η1,η2,η3); thresholds (τ,κ); hypothesis count N.
2:Output: Answera†or A BSTAIN
3:// Step 1: Encode query and entities
4:zq←LM(xq)
5:Extract query entities Eq={ei}and encode each
zei←LM(ei)
6:// Step 2: Score candidate parameters
7:foreachp∈Pdo
8:score(p)←maxei∈Eqcos(zei,zp)
9:end for
10:// Step 3: Retrieve relevant subgraph
11:Pk←Top-kscored parameters
12:P∗←Pk∪N(d)(Pk)
13:G∗←induced subgraph from P∗
14:// Step 4: Compose input for generation
15:xin←[lin(G∗;π);xq]
16:// Step 5: Generate and rerank answers
17:Generate NhypothesesH={a(1),...,a(N)}
18:foreacha∈H do
19: Compute coverage cov(a), hallucination penalty
hall(a), lengthlen(a)
20:J(a)←logpφ(a|
xin)+η1cov(a)−η2hall(a)−η3len(a)
21:end for
22:a†←argmax a∈HJ(a)
23:// Step 6: Conﬁdence check
24:conf←1
|P∗|/summationtext
p∈P∗score(p)
25:ifconf< τ orcov(a†)< κ then
26: return ABSTAIN
27:else
28: returna†
29:end if
Diversity
ComprehensivenessHallucinationsDirectnessEmpowerment405060708090ScoreGraphRAG
RAG
Fig. 3. Experiment results for comparing performance betwe en GraphRAG
and Vanilla RAG with GPT-4o.Diversity
ComprehensivenessHallucinationsDirectnessEmpowerment405060708090ScoreLlama
GPT
Fig. 4. Experiment results for comparing performance betwe en GraphRAG
with Llama3.1-8b and GraphRAG with GPT-4o.
GPT provides the most balanced and high-quality performanc e
across all evaluation dimensions.
Figure 5 compares the structural components of the knowl-
edge graphs constructed with Llama and GPT-3.5-turbo, fo-
cusing on the number of entities, relationships, and detect ed
communities. While both models extract a large set of entiti es
from the plasma literature, GPT identiﬁes more entities (91 8
vs. 787) and, more importantly, captures a much higher
number of relationships (414 vs. 148). This richer connecti vity
translates into a graph with stronger inter-parameter link s,
which provides the retrieval model with better context for
answering parameter-related queries. Moreover, GPT detec ts
45 distinct communities within the knowledge graph, wherea s
Llama yields only a single loosely connected cluster. Each
community corresponds to a cohesive subgraph in which
entities co-occur frequently across the literature and sha re
strong semantic or physical associations. In practice, the se
clusters align closely with meaningful physics concepts. F or
example, one community centers on magnetic geometry de-
scriptors, another on turbulence-driving gradients, and o thers
on collisionality or shearing rates. This structure indica tes that
GPT not only captures a denser web of parameter relationship s
but also organizes them into interpretable, physics-consi stent
domains. Such emergent clustering enhances both the trans-
parency and the interpretability of downstream GraphRAG
reasoning, enabling the agent to retrieve evidence that mir rors
the way plasma physicists naturally group related quantiti es.
Figure 6 compares the performance of GraphRAG when
combined with DeepSeek-R1 and Claude 3.7 Sonnet across the
ﬁve evaluation metrics, where higher scores indicate bette r per-
formance. Overall, DeepSeek-R1 achieves consistently hig her
or comparable scores in all metrics, showing clear advantag es
incomprehensiveness ,hallucination control , and empower-
ment . This suggests that DeepSeek-R1 produces broader, more
reliable, and practically useful parameter recommendatio ns.
Claude 3.7, on the other hand, performs slightly better in
directness and maintains competitive diversity, indicating that
its responses are concise and well-structured but somewhat
less extensive in contextual coverage. Taken together, the

8
Llama GPT02004006008001000ScoreEntity Number
Relationship Number
Community Number
Fig. 5. Components in the Knowledge Graph constructed with L lama3.1-8b
and GPT-3.5-turbo.
Diversity
ComprehensivenessHallucinationsDirectnessEmpowerment7075808590ScoreDeepSeek-R1
Claude 3.7
Fig. 6. Experiment results for comparing performance betwe en GraphRAG
with DeepSeek-R1 and GraphRAG with Claude 3.7 Sonnet.
results demonstrate that DeepSeek-R1 offers more balanced
and overall stronger performance, while Claude 3.7 priorit izes
brevity and clarity in its outputs.
Figure 7 presents the evaluation of three Llama models
with increasing parameter sizes, 3B, 8B, and 70B, across the
ﬁve performance metrics. The results show a clear positive
correlation between model scale and overall performance.
Llama-70B achieves the highest scores in almost all metrics ,
particularly in directness and hallucination control, whe re it
approaches 80 points, indicating that larger models not onl y
provide clearer and more precise answers but also remain
more faithful to the retrieved evidence. Improvements are
also evident in diversity and comprehensiveness, suggesti ng
that the expanded capacity of the 70B model enables it to
capture a wider range of plasma descriptors and generate
richer, more context-aware parameter recommendations. By
contrast, the smaller Llama-3B and Llama-8B models perform
considerably lower, especially in empowerment, reﬂecting
their limited ability to produce guidance that is practical lyDiversity
ComprehensivenessHallucinationDirectnessEmpowerment30405060708090ScoreLlama 3b
Llama 8b
Llama 70b
Fig. 7. Scores of Llama models (3B, 8B, 70B) across ﬁve evalua tion metrics,
showing clear gains with larger model sizes.
Diversity
ComprehensivenessHallucinationDirectnessEmpowerment405060708090100ScoreLlama3.2-3b
Llama3-8b
Llama3.3-70b
Deepseek-R1-685b
GPT-4o-1.8T
Claude 3.7 Sonnet-230B
Fig. 8. Comparison of multiple LLM families, where ultra-la rge models (GPT-
4o, Claude) outperform smaller Llamas and DeepSeek-R1 show s strength in
diversity.
useful for parameter setting.
Figure 8 extends the evaluation to a broader set of LLM
families, including the Llama series (3B, 8B, 70B), DeepSee k-
R1 (685B), GPT-4o (1.8T), and Claude 3.7 Sonnet (230B).
The results reveal a clear scaling trend: while smaller Llam a
models achieve only modest performance across all metrics,
larger-scale systems demonstrate dramatic improvements, par-
ticularly in comprehensiveness, directness, and empowerm ent.
Among the ultra-large models, GPT-4o and Claude 3.7 Son-
net dominate the evaluation, reaching near-perfect scores in
comprehensiveness and maintaining strong performance in
diversity and empowerment, which underscores their superi or
reasoning and grounding capacity. DeepSeek-R1 also perfor ms
competitively, especially in diversity and hallucination control,
reﬂecting its architectural emphasis on deep reasoning. In
contrast, even the largest Llama-70B lags behind these fron tier
models, showing that while scaling within a family improves
results, architecture and training quality remain decisiv e fac-

9
tors. Overall, the ﬁgure highlights a hierarchy in capabili ty:
smaller Llamas are lightweight but limited, while ultra-la rge
models like GPT-4o and Claude set the benchmark for high-
quality, well-grounded responses, albeit at much higher co m-
putational cost.
V. C ONCLUSION
In this work, we have introduced Plasma GraphRAG, a
framework that integrates GraphRAG with large language
models to automate the identiﬁcation of parameter ranges in
gyrokinetic simulations. Unlike traditional manual revie ws,
Plasma GraphRAG constructs a physics-informed knowledge
graph and applies structured retrieval to explicitly captu re
parameter relationships. This design enables accurate, co m-
prehensive, and reproducible recommendations while reduc ing
hallucinations. Experimental results show that GraphRAG
consistently outperforms vanilla RAG across key metrics su ch
as diversity, grounding, and interpretability. Neverthel ess, the
current evaluation is limited by the relatively small bench mark
dataset and the use of heuristic metrics for assessing outpu t
quality. Future work will expand the benchmark to cover
a broader range of plasma regimes and simulation codes,
incorporate quantitative validation against experimenta l data,
and explore reinforcement learning–based optimization fo r
adaptive retrieval and evidence weighting. Overall, Plasm a
GraphRAG accelerates surrogate model development and pro-
vides a scalable foundation for reliable, interpretable pa rameter
selection in plasma physics and other scientiﬁc domains.
REFERENCES
[1] J. Candy, R. Waltz, and W. Dorland, “The local limit of glo bal gyroki-
netic simulations,” Physics of Plasmas , vol. 11, no. 5, pp. L25–L28,
2004.
[2] W. W. Lee, “Gyrokinetic particle simulation model,” Journal of Com-
putational Physics , vol. 72, no. 1, pp. 243–269, 1987.
[3] X. Garbet, Y . Idomura, L. Villard, et al. , “Gyrokinetic simulations of
turbulent transport,” Nuclear Fusion , vol. 50, no. 4, p. 043002, 2010.
[4] H. Du, R. Zhang, D. Niyato, et al. , “Exploring collaborative distributed
diffusion-based ai-generated content (aigc) in wireless n etworks,” IEEE
Network , vol. 38, no. 3, pp. 178–186, 2024.
[5] Y . Gao, Y . Xiong, X. Gao, et al. , “Retrieval-augmented generation for
large language models: A survey,” arXiv preprint arXiv:2312.10997 ,
vol. 2, no. 1, 2023.
[6] R. Zhang, H. Du, Y . Liu, et al. , “Interactive ai with retrieval-augmented
generation for next generation networking,” IEEE Network , vol. 38,
no. 6, pp. 414–424, 2024.
[7] E. Frieman and L. Chen, “Nonlinear gyrokinetic equation s for low-
frequency electromagnetic waves in general plasma equilib ria,” The
Physics of Fluids , vol. 25, no. 3, pp. 502–508, 1982.
[8] F. Jenko, W. Dorland, M. Kotschenreuther, et al. , “Electron temperature
gradient driven turbulence,” Physics of plasmas , vol. 7, no. 5, pp. 1904–
1910, 2000.
[9] J. Candy, E. A. Belli, and R. Bravenec, “A high-accuracy e ulerian
gyrokinetic solver for collisional plasmas,” Journal of Computational
Physics , vol. 324, pp. 73–93, 2016.
[10] P. Donnel, X. Garbet, Y . Sarazin, et al. , “A multi-species collisional
operator for full-f global gyrokinetics codes: Numerical a spects and
veriﬁcation with the gysela code,” Computer Physics Communications ,
vol. 234, pp. 1–13, 2019.
[11] G. Staebler, J. Kinsey, and R. Waltz, “A theory-based tr ansport model
with comprehensive physics,” Physics of Plasmas , vol. 14, no. 5, 2007.
[12] B. Clavier, D. Zarzoso, D. del Castillo-Negrete, et al. , “Generative-
machine-learning surrogate model of plasma turbulence,” Physical Re-
view E , vol. 111, no. 1, p. L013202, 2025.
[13] G. Galletti, F. Paischer, P. Setinek, et al. , “5d neural surrogates for
nonlinear gyrokinetic simulations of plasma turbulence,” arXiv preprint
arXiv:2502.07469 , 2025.[14] S. Maeyama, M. Honda, E. Narita, et al. , “Multi-ﬁdelity information
fusion for turbulent transport modeling in magnetic fusion plasma,”
Scientiﬁc Reports , vol. 14, no. 1, p. 28242, 2024.
[15] D. Kim, T. Moon, C. Sung, et al. , “Veriﬁcation of fast ion effects on
turbulence through comparison of gene and cgyro with l-mode plasmas
in kstar,” arXiv preprint arXiv:2408.13731 , 2024.
[16] D. Chen, A. Fisch, J. Weston, et al. , “Reading wikipedia to answer
open-domain questions,” arXiv preprint arXiv:1704.00051 , 2017.
[17] K. Guu, K. Lee, Z. Tung, et al. , “Retrieval augmented language model
pre-training,” in International conference on machine learning . PMLR,
2020, pp. 3929–3938.
[18] P. Lewis, E. Perez, A. Piktus, et al. , “Retrieval-augmented generation
for knowledge-intensive nlp tasks,” Advances in neural information
processing systems , vol. 33, pp. 9459–9474, 2020.
[19] T. T. Procko and O. Ochoa, “Graph retrieval-augmented g eneration for
large language models: A survey,” in 2024 Conference on AI, Science,
Engineering, and Technology (AIxSET) , 2024, pp. 166–169.
[20] B. Peng, Y . Zhu, Y . Liu, et al. , “Graph retrieval-augmented generation:
A survey,” arXiv preprint arXiv:2408.08921 , 2024.
[21] S. Knollmeyer, O. Caymazer, and D. Grossmann, “Documen t graphrag:
Knowledge graph enhanced retrieval augmented generation f or docu-
ment question answering within the manufacturing domain,” Electronics ,
vol. 14, no. 11, p. 2102, 2025.
[22] J. Lála, O. O’Donoghue, A. Shtedritski, et al. , “Paperqa: Retrieval-
augmented generative agent for scientiﬁc research,” arXiv preprint
arXiv:2312.07559 , 2023.
[23] H. Han, H. Shomer, Y . Wang, et al. , “Rag vs. graphrag: A systematic
evaluation and key insights,” arXiv preprint arXiv:2502.11371 , 2025.
[24] Z. Ji, N. Lee, R. Frieske, et al. , “Survey of hallucination in natural
language generation,” ACM computing surveys , vol. 55, no. 12, pp. 1–
38, 2023.
[25] C. Bourdelle, J. Citrin, B. Baiocchi, et al. , “Core turbulent transport
in tokamak plasmas: bridging theory and experiment with qua likiz,”
Plasma Physics and Controlled Fusion , vol. 58, no. 1, p. 014036, 2015.
[26] P. Rodriguez-Fernandez, N. T. Howard, and J. Candy, “No nlinear
gyrokinetic predictions of sparc burning plasma proﬁles en abled by
surrogate modeling,” Nuclear Fusion , vol. 62, no. 7, p. 076036, 2022.
[27] S. Wu, Y . Xiong, Y . Cui, et al. , “Retrieval-augmented generation for nat-
ural language processing: A survey,” arXiv preprint arXiv:2407.13193 ,
2024.
[28] N. Reimers and I. Gurevych, “Sentence-bert: Sentence e mbeddings using
siamese bert-networks,” arXiv preprint arXiv:1908.10084 , 2019.
[29] P. Zhao, H. Zhang, Q. Yu, et al. , “Retrieval-augmented generation for
ai-generated content: A survey,” arXiv preprint arXiv:2402.19473 , 2024.
[30] H. Yu, A. Gan, K. Zhang, et al. , “Evaluation of retrieval-augmented
generation: A survey,” in CCF Conference on Big Data . Springer,
2024, pp. 102–120.