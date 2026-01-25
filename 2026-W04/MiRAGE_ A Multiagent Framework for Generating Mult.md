# MiRAGE: A Multiagent Framework for Generating Multimodal Multihop Question-Answer Dataset for RAG Evaluation

**Authors**: Chandan Kumar Sahu, Premith Kumar Chilukuri, Matthew Hetrich

**Published**: 2026-01-21 21:39:09

**PDF URL**: [https://arxiv.org/pdf/2601.15487v1](https://arxiv.org/pdf/2601.15487v1)

## Abstract
The rapid evolution of Retrieval-Augmented Generation (RAG) toward multimodal, high-stakes enterprise applications has outpaced the development of domain specific evaluation benchmarks. Existing datasets often rely on general-domain corpora or purely textual retrieval, failing to capture the complexity of specialized technical documents where information is inextricably multimodal and reasoning requires synthesizing disjoint evidence. We address this gap by introducing MiRAGE, a Multiagent framework for RAG systems Evaluation, that leverages a collaborative swarm of specialized agents to generate verified, domain-specific, multimodal, and multi-hop Question-Answer datasets. MiRAGE orchestrates a swarm of specialized agents: a recursive context optimization loop to aggregate scattered evidence, an adversarial verifier agent to guarantee factual grounding, and an agent to recognize the expert persona and the relevant domain to mimic expert cognitive workflows. Extensive empirical evaluation across four distinct domains (regulations, finance, quantitative biology, and journalism) demonstrates that MiRAGE generates datasets with significantly higher reasoning complexity (>2.3 average hops) and factual faithfulness. Our ablation studies point that MiRAGE can be powered by LLMs if textual descriptions of the images are available. Visual grounding still remains a frontier. By automating the creation of gold standard evaluation datasets that reflect the latent thematic structure of proprietary corpora, MiRAGE provides the necessary infrastructure to rigorously benchmark the next generation information retrieval systems.

## Full Text


<!-- PDF content starts -->

MiRAGE: A Multiagent Framework for Generating Multimodal Multihop
Question-Answer Dataset for RAG Evaluation
Chandan Kumar Sahu1*, Premith Kumar Chilukuri1*, Matthew Hetrich1*†
1ABB Inc
Abstract
The rapid evolution of Retrieval-Augmented
Generation (RAG) toward multimodal, high-
stakes enterprise applications has outpaced
the development of domain specific evalu-
ation benchmarks. Existing datasets often
rely on general-domain corpora or purely tex-
tual retrieval, failing to capture the complex-
ity of specialized technical documents where
information is inextricably multimodal and
reasoning requires synthesizing disjoint evi-
dence. We address this gap by introduingMi-
RAGE1, aMultiagent framework forRAGsys-
temsEvaluation, that leverages a collaborative
swarm of specialized agents to generate veri-
fied, domain-specific, multimodal, and multi-
hop Question-Answer datasets. MiRAGE or-
chestrates a swarm of specialized agents: a re-
cursive context optimization loop to aggregate
scattered evidence, an adversarial verifier agent
to guarantee factual grounding, and an agent to
recognize the expert persona and the relevant
domain to mimic expert cognitive workflows.
Extensive empirical evaluation across four dis-
tinct domains (regulations, finance, quantita-
tive biology, and journalism) demonstrates that
MiRAGE generates datasets with significantly
higher reasoning complexity ( >2.3 average
hops) and factual faithfulness. Our ablation
studies point that MiRAGE can be powered
by LLMs if textual descriptions of the images
are available. Visual grounding still remains
a frontier. By automating the creation of gold
standard evaluation datasets that reflect the la-
tent thematic structure of proprietary corpora,
MiRAGE provides the necessary infrastructure
to rigorously benchmark the next generation
information retrieval systems.
1 Introduction
Large Language Models (LLMs) have demon-
strated remarkable capabilities in encoding world
*Equal contribution.
†Corresponding author: matthew.hetrich1@us.abb.com
1https://github.com/ChandanKSahu/MiRAGEknowledge within their parameters. However, they
face fundamental limitations regarding rare enti-
ties, cutoff dates, and the high computational cost
of retraining to update information (Kandpal et al.,
2023; Mallen et al., 2023). Retrieval-Augmented
Generation (RAG) has emerged as the de facto
solution to these challenges, mitigating halluci-
nations by grounding generation in external, non-
parametric knowledge bases (Lewis et al., 2020).
As RAG systems are increasingly deployed in high-
stakes enterprise domains ranging from medical di-
agnosis (Xiong et al., 2024) to wind energy (Meyur
et al., 2025), the imperative for rigorous, scenario-
specific evaluation has intensified.
Despite the rapid adoption of RAG, evalu-
ating these systems remains a non-trivial chal-
lenge. Standard benchmarks, such as Natural Ques-
tions (Kwiatkowski et al., 2019) or MS MARCO
(Nguyen et al., 2016), typically rely on open-
domain data that fails to reflect the complexity of
specialized corpora. In real-world environments,
knowledge is rarely confined to text. It is inextrica-
bly multimodal, locked within charts, technical dia-
grams, and complex layouts. Recent work on Mul-
timodal RAG, such as MuRAG (Chen et al., 2022),
highlights that models restricted to textual retrieval
neglect the massive amount of knowledge present
in visual modalities. Furthermore, while single-hop
retrieval is relatively mature, existing systems strug-
gle significantly with multi-hop queries that require
synthesizing disjoint pieces of evidence scattered
within a document (Tang and Yang, 2024).
To bridge this gap, research has pivoted toward
synthetic dataset generation, as obtaining human-
annotated data for multi-hop reasoning is both time-
consuming and resource-intensive (Wu et al., 2024).
However, current synthetic frameworks often suffer
from critical deficiencies. Pipelines like DataMor-
gana (Filice et al., 2025) or SMMQG (Wu et al.,
2024) generally employ linear generation strategies
that lack robust feedback mechanisms. This oftenarXiv:2601.15487v1  [cs.AI]  21 Jan 2026

results in hallucinated evaluation datasets. There is
a paucity of high-quality benchmarks that simulta-
neously address domain specificity, multimodality,
and complex reasoning steps.
To address these limitations, we introduceMi-
RAGE, a Multi-Agentic framework designed to
generate robust multihop multimodal expert-level
evaluation datasets. Unlike linear prompting strate-
gies, MiRAGE orchestrates a swarm of specialized
agents to mimic the cognitive workflow of a do-
main expert. We propose a dynamiccontext opti-
mization loop, where a retrieval agent recursively
builds a semantic context window, gathering scat-
tered evidence to support complex inquiries before
a question-answer pair is formulated. Crucially, we
address the reliability of synthetic data through an
adversarial verification phase, employing a dedi-
cated agent to fact-check generated answers against
source context ensuring that the generated insight
is consistent with the source.
Our contributions are summarized as follows:
1.We propose a model-agnostic, multi-agent
framework that automates the ingestion and
semantic segmentation of complex multi-
modal documents, preserving the semantic de-
pendencies between text and visual elements.
2.We introduce a novel generative methodology
that utilizes recursive context expansion and
persona injection to produce multi-hop QA
pairs. This allows for the creation of questions
that require logical deduction across disjoint
chunks, surpassing the complexity of extrac-
tive QA.
3.We provide an extensive empirical evalua-
tion across four distinct domains: regula-
tions, finance, science, and journalism. We
demonstrate that MiRAGE generates datasets
with significantly higher reasoning complexity
while strictly adhering to the latent thematic
distribution of the source domain.
4.Our ablation study revealed that the do-
main/persona injection, multihop context and
QA verification play a crucial role in the qual-
ity of the generated QA dataset. MiRAGE can
be powered by LLMs if the textual descrip-
tions of the images contained in the document
are available.2 Literature Review
2.1 Evaluation of RAG Systems
Evaluating RAG systems remains a non-trivial chal-
lenge due to the dual dependencies of retrieval
precision and generation faithfulness. Traditional
evaluation often relies on static, general-domain
datasets such as Natural Questions (Kwiatkowski
et al., 2019), MS MARCO (Nguyen et al., 2016)
and MIRAGE (Park et al., 2025). However, these
benchmarks fail to capture the nuances of domain-
specific corpora. Consequently, the generation
of synthetic datasets became inevitable. Frame-
works likeDataMorgana(Filice et al., 2025) and
DQABench(Zheng et al., 2025) leverage LLMs to
generate QA pairs for metrics-based evaluation (Es
et al., 2024).
The proprietary nature ofDataMorganaand the
schema-based nature ofDQABench, limit their us-
age and generalizability. Frameworks likeRAG-
Probe(Sivasothy et al., 2025) highlight that system
failures occur most frequently when prompts com-
bine multiple constraints. Thus, most synthetic gen-
erators produce simple, single-constraint queries.
MiRAGE addresses the reliability of synthetic data
through a dedicated QA generation agent with a
distinct verifier agent that fact-checks generated
answers to mitigate hallucination.
2.2 Multimodal Context
Real-world documents (E.g., technical standards,
manuals, and research papers) are inherently mul-
timodal, interleaving text with tables, charts, and
diagrams. Recent literature on Visual QA (Kim
et al., 2025) and Multimodal RAG (Mei et al.,
2025) emphasizes that critical information is often
locked in non-textual formats. Recent works like
SPIQA(Pramanick et al., 2024) andWikiMixQA
(Foroutan et al., 2025) have attempted to bridge the
multimodal gap by interpreting scientific figures.
MuRAG(Chen et al., 2022) requires joint reason-
ing over images and text. However, the critical
bottleneck of a unified perception stage still per-
sists (Zhao et al., 2023). Modalities are often pro-
cessed in isolation, breaking the document’s seman-
tic flow. The standard OCR frequently degrades
tabular structure. MiRAGE addresses this by im-
plementing a multimodal ingestion phase where
a vision agent generates descriptions and extracts
table structures into an enriched markdown. This
preserves the semantic proximity between textual
and visual elements in a shared vector space, mir-

Semantic Chunking
 Description Agent
 
Domain Analysis
Agent
 Context V erification
 
Expert Persona 
and Domain Chunks Vector Index QA Dataset
 Technical Corpus
 
Retriever and RerankerCheck Relevance of
Retrieved Chunk  
 QA Generation
 
QA Verification
 
Deduplication &
Refinement
 Yes
No
YesNoMultihop Context BuildingQA Generation
& VerificationMultimodal Data
Processing
Data Multimodal
EmbeddingFigure 1: The multiagent framework of MiRAGE to evaluate RAG systems
roring the source layout.
2.3 Multi-hop Reasoning
Effective RAG systems shall go beyond simple in-
formation retrieval to synthesize information across
scattered but semantically related content. Re-
search intoMultiHop-RAG(Tang and Yang, 2024)
andHotpotQA(Yang et al., 2018) has demonstrated
that systems performing well on single-hop queries
often fail significantly when required to connect
multiple pieces of evidence. While datasets like
MuSiQue(Trivedi et al., 2022) andGraphhop-
per(Koner et al., 2021) introduce compositional
queries, they largely rely on open-domain sources
(e.g., Wikipedia) or extractive answers. Crucially,
they lack the specific expert persona required for
domain-specific tasks, such as engineering or fi-
nance (Schnitzler et al., 2024). MiRAGE automates
the creation of complex reasoning chains through a
context optimization loop. The context agent iter-
atively expands search queries to gather scattered
evidence, allowing the QA generator agent to create
complex, multi-hop Q&A pairs that mimic expert
analysis rather than simple keyword matching.
2.4 Agentic Frameworks for Dataset
Generation
The complexity of generating verified, multimodal,
multi-hop questions has led to the emergence of
agentic frameworks.SMMQG(Wu et al., 2024) em-
ploys an interplay between a retriever, LLM, and
a multimodal model to synthesize questions. Simi-
larly,WeQA(Meyur et al., 2025) utilize Human-AI
teaming to ensure domain relevance. However,the linear pipeline ofSMMQG(Wu et al., 2024)
lacks a robust feedback mechanism for deduplica-
tion, often resulting in semantic redundancy and
inflated datasets. MiRAGE advances this paradigm
by organizing agents into a generation swarm. By
incorporating a selector agent to filter for difficulty
and a curator agent that performs hierarchical dedu-
plication and clustering, MiRAGE ensures the final
output is not only accurate but diverse, minimizing
redundancy while maximizing semantic coverage.
3 Methodology
We propose a multimodal multi-agentic framework
designed to generate a question-answer dataset
(qi, ai)∈ D from a technical corpora C.Dis a
collection of validated pairs (qi, ai). Mathemati-
cally, the dataset generation process is defined by
D= Φ
{(qi, ai)∼P(q, a|V= 1,C,Θ)}N
i=1
(1)
The joint probability of generating a valid pair from
its context is
P(q, a,V= 1|C,Θ) =
X
Cs⊆Ch
P(V= 1|q, a, Cs, θV)
·P(q, a|Cs, θQA)
·P(Cs|C, θ S)i(2)
where Csis a semantic context (a subset of
chunks) derived from C. The parameter set Θ =
{θV, θQA, θS}represents the configurations for
the verification, generation, and semantic search

agents, respectively. The framework operates in
five phases: (1) multimodal data ingestion and se-
mantic chunking, (2) identification of expert per-
sona and domain, (3) semantic multihop context
building, (4) agentic QA generation and verifica-
tion, and lastly (5) refinement and deduplication.
The architecture is model-agnostic, allowing for the
interchange of underlying language models. The
framework is illustrated in Fig. 1.
3.1 Multimodal Data Ingestion and Semantic
Chunking
Raw technical documents contain complex layouts
where textual information is inextricably linked
with visual artifacts. To address this, we imple-
ment a hybrid parsing pipeline to structure C. We
utilize a document layout analysis engine for struc-
tural segmentation and a description agent powered
by a Vision Language Model (VLM) for visual
interpretation. The description agent generates a
dense textual description difor every visual ele-
mentv i∈ C.
di=A(v i|πdesc)(3)
where πdescis the prompt todescribethe technical
details.
We employ a semantic chunking strategy to seg-
ment the source document into semantically coher-
ent chunks. The document text Tis processed via
a sliding window of length Lwith an overlap of
l. For each window Wt, the semantic chunking
agent identifies the optimal partition C∗from all
the possible partitions ofW t,
C∗= arg min
C∈Partitions(W t)|C|−1X
j=1[1−Sim(c j, cj+1)]
+λ|C|
(4)
where cj, cj+1are adjacent chunks in candidate
partition C, and Sim(·) is a semantic similarity
function. λis a regularization hyperparameter that
prevents over-fragmentation.
3.2 Identification of Expert Persona and
Domain
We incorporate an agent to determine the core do-
main DCand an expert persona PCby performing
a global analysis of the corpus. These parameterscondition the generation agents to reflect the do-
main knowledge and the style and complexity of
a subject matter expert. Each chunk ci∈ C is
represented by its multimodal embedding ei. Let
E={e 1, . . . , e M}be the collection of embeddings.
The agent employs a topic modeling pipeline to
discover the latent thematic structure within the
corpus using three steps. First, Eis projected into
a lower-dimensional space using manifold learn-
ing. Second, density-based clustering partitions
the chunks into Kthematic clusters {τ1, . . . , τ K}
where each cluster correspond to a distinct topic.
Third, we utilize class-based TF-IDF to generate a
representative keyword list Rkfor each cluster τk,
refined using Maximal Marginal Relevance (MMR)
to promote keyword diversity. Finally, the agent
synthesizes the domain and persona from the dom-
inant topic representations{R k},
(DC, PC)∼ A(D, P| {R k}K
k=1, πDP)(5)
where πDPis the prompt to extract the domain of
the corpus and a relevant expert persona.
3.3 Semantic Multihop Context Building
We implement a recursive, agentic retrieval pro-
cess that expands an initial seed chunk cseedinto a
complete semantic context Cs. Let Stdenote the
set of chunks constituting the context at iteration
t, where S0={c seed}. At step t, a multimodal
agent analyzes Stto detect missing information.
We model this as a conditional generation task:
(zt, Qsearch
t ) =A(S t|πcomp)(6)
where πcomp is the completeness prompt. The out-
put consists of a boolean status zt(complete/incom-
plete) and, if incomplete, a set of retrieval queries
Qsearch
t ={qs
1, . . . , qs
k}. The process terminates
ifzt= 1 ort≥δ max, yielding Cs=S t. For
each generated query qs∈Qsearch
t , we employ a
hybrid retrieval strategy. We first retrieve the top-
Ncandidates from Efollowed by reranking by a
multimodal reranker.
To prevent context drift, we verify if a candidate
chunk ccand retrieved by the query qsspecifically
offers the information missing in Stor is related to
St. The context expands by
St+1=St∪{ccand| A(S t, qs, ccand|πadd)}(7)
where πaddis the prompt that assesses the suitabil-
ity of ccand. This ensures the construction of Csis
strictly monotonic regarding information utility.

Table 1: Summary of selected corpora for MiRAGE evaluation.
Dataset Name Domain # Pages # Images # Tables # Tokens
S&P Global Annual Reports Finance 1302 1,120 2,800 0.9M
UNECE GTRs Regulation 7594 150 3,450 3.8M
Quantitative Biology Science 8336 9,400 850 5.2M
NYTimes Opinions Journalism>3000 3,050 25 2.1M
3.4 QA Generation and Verification
We generate candidate question-answer pairs (q, a)
from the context Csand verify them. This cor-
responds to the probability terms P(q, a|Cs, θQA)
andP(V= 1|q, a, Cs, θV)defined in Eq. 2. To
ensure technical depth, we condition the QA gen-
eration on the context Cs, the domain DC, and
the persona PC(encapsulated in θQA). The set of
candidatesD cand is generated as:
Dcand={(q i, ai)}M
i=1∼ A(Cs
i, PC, DC|πQA)
(8)
whereπ QAis the QA generation prompt.
To ensure that the agent is relying only on the
provided context Cs, a verifier agent assesses each
candidate(q, a)∈ D cand againstCs. The verifica-
tion function evaluates: (1) Correctness: ais factu-
ally supported by Cs. (2) Necessity: qrequires in-
formation within Csto get a. The validated dataset
contribution from contextCsis:
DCs
i={(q, a)∈ D cand| A(q, a, Cs
i|πver)}
(9)
whereπ veris the verification prompt.
3.5 Refinement and Deduplication
The final stage aggregates the DCs
iinto the final
cohesive corpus Dvia hierarchical clustering and
deduplication. We employ a two-stage clustering
process. First, we apply community detection on
the question embeddings to form a set of high-
level clusters KQthat share semantic themes. Sec-
ond, within each cluster k∈ K Q, we sub-cluster
the answers to identify redundancy. The union of
these sub-clusters constitutes the fine-grained an-
swer cluster setK A.
The similarity metric between pairs ui= (q i, ai)
anduj= (q j, aj)is a weighted sum of semantic
similarity and source lineage overlap:
Sim(u i, uj) =αcos(e ai, eaj)+(1−α)·J(Cs
i, Cs
j)
(10)
where eais the answer embedding, and J(·) is the
Jaccard similarity of the source contexts Cs
iand
Cs
jassociated with uianduj, respectively. αis a
weighting factor.Lastly, we employ a stratified policy to refine
the QA pairs within each answer sub-cluster into
representative units. We define a refinement func-
tionΨthat operates on the QA pairs in each cluster
K∈ K A. If the internal similarity of the cluster
exceeds the threshold τt, they are merged by the
refinement agent; otherwise, the original units are
retained. Mathematically,
Ψ(K) =

A(K|π ref)ifmin
ui,uj∈KSim(u i, uj)> τ t
Kotherwise
(11)
where πrefis the prompt instructing the agent to
synthesize a unified refined set of QA pairs that
encompasses the unique details from all QA pairs
inK. The final optimized dataset Dis constructed
by the union of all the refined QA pairs:
D=[
K∈K AΨ(K)(12)
4 Experiments
4.1 Corpora Selection
To validate the efficacy ofMiRAGEacross the full
spectrum of enterprise and information retrieval
challenges, we curated a diverse suite of corpora
spanning four distinct domains: finance, regula-
tions, science, and journalism. TheS&P Global
Annual Reports(S&P Global, 2023) represent the
financial domain, where strategic narratives are
deeply intertwined with dense tabular data and fi-
nancial visualizations. TheUNECE Global Techni-
cal Regulations(United Nations Economic Com-
mission for Europe, 2023) are characterized by
rigid hierarchical structures, precise definitions,
and conditional logic. We selected scientific publi-
cations from the quantitative biology domain sub-
mitted to the Arxiv2in the month of January of
2025. This corpus introduces extreme lexical speci-
ficity regarding protein structures and complex vi-
sual artifacts like 3D molecular renderings. The
last dataset consists of the Opinions from The New
York Times collected from the Visual News dataset
2https://arxiv.org/list/q-bio/2025-01?skip=0&
show=2000

—
© 2026 ABB. All rights reserved. Slide 2
The table tracks the 
components and 
changes in 
Accumulated Other 
Comprehensive Loss 
(AOCL) for a fiscal 
year…
The table defines the 
annual financial 
components impacting 
an entity's retirement 
and postretirement 
benefit plans ...Based on the summary of changes in 
Accumulated Other Comprehensive Loss 
for 2021, what amount was reclassified to 
net earnings for Pension and Postretirement 
Benefit Plans , and what specific item 
detailed in the changes to benefit 
obligations corresponds to this 
reclassification for retirement plans ?
For the year ended December 31, 2021, 
$15 million was reclassified from 
accumulated other comprehensive income 
to net earnings for the Pension and 
Postretirement Benefit Plans category. 
This figure corresponds to the $(15) 
million in \"Recognized actuarial (gain) 
loss\" for Retirement Plans as detailed in 
the table of other changes in plan assets 
and benefit obligations.
Chunk #275
Chunk #243Table chunks as markdown Generated DescriptionsQuestion
AnswerFigure 2: A sample question-answer pair generated from two related chunks with relevant keywords highlighted
(Liu et al., 2021). The datasets are listed in Table 1.
These domains provide a comprehensive validation
ground, moving from highly structured text-heavy
documents to unstructured visually complex media.
4.2 Model Selection
The MiRAGE framework is architected to be
model-agnostic, supporting a modular interchange
of LLMs and VLMs. Our architecture primar-
ily leverages state-of-the-art proprietary models
to ensure maximum reasoning fidelity. We utilize
Gemini-2.5-Flash and GPT-5-Mini as the core en-
gines for the reasoning agents. Their high-context
windows, advanced vision encoding, and superior
instruction-following capabilities are critical for
the multi-hop QA generation and verification tasks.
The multimodal chunks are embedded with the
Nomic model (Nussbaum et al., 2024) for seman-
tic retrieval based on textual queries3. The re-
trieval precision during the multihop context build-
ing is refined with multimodal rerankers. We em-
ploy the LLM-as-a-reranker (Abdallah et al., 2025)
paradigm utilizing the proprietary models to rank
the retrieved chunks. This ensures that the final con-
text window contains only the most semantically
pertinent chunks.
4.3 Metrics
To rigorously evaluate the quality of the generated
dataset D, we employ a comprehensive suite of
metrics that assesses retrieval accuracy, reasoning
complexity, multimodal integration, and domain
coverage. We utilize both standard automated met-
3The token limits of the embedding and reranker mod-
els based on the CLIP and SigLIP architectures render them
ineffective for our multimodal tasks.rics (Es et al., 2024) and novel agentic evaluation
protocols based on the LLM-as-a-Judge paradigm
(Zheng et al., 2023).
For each QA pair (q, a) and retrieved context Cs,
we measure the faithfulness and relevance. To vali-
date that Dnecessitates multi-hop context building
rather than simple extraction, we implement a rea-
soning trace evaluator T(q, Cs). It consists of hop
count ( H), number of distinct retrieval or reasoning
steps required to derive afrom(q, Cs). We employ
a VLM agent as a verifier to ensure the generated
answer ais visually grounded. The agent verifies
that specific visual features referenced in a(e.g.,
trend lines, molecular structures) are present in the
image set V⊂Cs. We quantify the alignment be-
tween the latent topics of the source corpus and the
generated QA dataset using the Jensen-Shannon
(JS) divergence. Let PCandPDbe the discrete
probability distributions over latent topic clusters
for the corpus and dataset, respectively. The cover-
age metric is defined as:
DJS(PC||PD) =1
2DKL(PC||M)+1
2DKL(PD||M)
(13)
where M=1
2(PC+PD). Lower divergence val-
ues imply that the synthetic dataset faithfully re-
produces the thematic distribution of the original
domain.
5 Results
We present the empirical evaluation of the QA
datasets generated by MiRAGE. We create 1000
QA pairs from each corpus. The prompts excluding
the domain specific few shot examples are included
in Appendix A. A sample QA pair is illustrated in

Table 2: Performance comparison of MiRAGE across four domains.
Dataset Model Faith.↑Rel.↑Avg Hops (H)↑Vis. Gr.↑
S&P Global
(Finance)Gemini 2.5 Flash0.96 0.86 2.840.21
GPT 5 Mini 0.91 0.81 2.420.28
UNECE GTRs
(Regulation)Gemini 2.5 Flash 0.940.902.45 0.38
GPT 5 Mini0.960.822.60 0.45
Q-Bio Arxiv
(Science)Gemini 2.5 Flash0.830.92 2.350.42
GPT 5 Mini 0.810.94 2.550.42
NYTimes
(Journalism)Gemini 2.5 Flash 0.91 0.93 1.100.32
GPT 5 Mini0.93 0.95 1.250.24
Figure 2. Our analysis focuses on the complexity
of reasoning, the faithfulness of the generated an-
swers, and the semantic alignment with the source
domains.
5.1 Overall Performance
Table 2 summarizes the performance of the Mi-
RAGE framework powered by two distinct state-
of-the-art models: Gemini 2.5 Flash and GPT 5
Mini. Our analysis highlights the framework’s abil-
ity to generate complex, domain-aligned question-
answer pairs while revealing nuanced differences
in model capabilities across various tasks. The
JS divergence from Table 3 reflects that the QA
dataset Dgenerated by MiRAGE covers the topics
inCeffectively.
Reasoning Complexity:A key achievement of
MiRAGE is its consistent generation of multi-hop
questions across technical domains. For the Fi-
nance, Regulation, and Science corpora, the aver-
age hop count ( H) consistently exceeds 2.3, peak-
ing at 2.84 with Gemini 2.5 Flash on the S&P
Global dataset. This demonstrates the efficacy
of the semantic multihop context-building phase,
which successfully forces models to synthesize in-
formation from disjoint sources. GPT 5 Mini shows
a slight advantage in generating more complex rea-
soning chains in the highly structured regulatory
and scientific domains. In contrast, the lower hop
count for the NYTimes corpus (avg. H≈1.2 )
reflects the open and less connected nature of jour-
nalistic content. The NYTimes corpus did not con-
tain the chunks relevant to the queries to make the
context complete.
Faithfulness and Relevance:MiRAGE demon-
strates exceptional performance in maintaining fac-
tual grounding and contextual relevance, with faith-
fulness scores consistently above 0.91 for three of
the four domains and relevance scores exceeding
0.81 across all experiments. This validates the ef-
fectiveness of the adversarial verifier agent, which
successfully filters out hallucinations and ensuresthat generated answers are strictly supported by
the retrieved context. Both Gemini 2.5 Flash and
GPT 5 Mini perform competitively with marginal
differences.
Visual Grounding:While excelling in textual
reasoning, the evaluation exposes visual grounding
as a persistent challenge for current VLMs. The
visual grounding scores remain moderate across
all domains, with a maximum of 0.45 achieved by
GPT 5 Mini on the UNECE GTRs dataset. This
suggests that while models can describe images,
generating complex questions that require precise
reasoning about specific visual elements (E.g., cor-
relating specific data points in a financial chart)
remains a frontier. Out of the 1093 QA pairs gen-
erated for the finance domain, only 84 QA pairs
are multimodal. It indicates that the VLMs prefer
textual content to generate the QA pairs. From
a closer look at the multimodal QA pairs, we hy-
pothesize that the generated descriptions made the
visual elements partially redundant leading to a
consistently lower score visual grounding score
across all evaluations.
5.2 Ablation Study
We conducted a systematic component-wise abla-
tion using a subset (2 out of 14 documents) of the
S&P Globalannual reports. This corpus was se-
lected for its high density of visual artifacts and
complex tabular structures. We isolate the impact
of the agentic architecture and the data representa-
tion strategies. The results are summarized in Table
3.
5.2.1 Impact of Agentic Architecture
Multihop Context:Removing the recursive re-
trieval step forces the model to generate questions
based solely on the initial seed chunk. This results
in a collapse of the difficulty score4(0.85→0.61 ).
4The domain specific expert agent rates the generated QA
pair on a discrete scale of [0,10] , later normalized to [0,1] for
easier interpretation.

Table 3: Ablation study on the S&P Global Annual Reports (Finance) dataset
Category Configuration Faith.↑Rel.↑Diff↑Avg Hops↑Vis. Gr.↑JSD↓
Baseline MiRAGE 0.97 0.95 0.851.92 0.41 0.08
Agentic Architecture(-) Multihop Context 0.93 0.82 0.61 - 0.21 4.35
(-) QA Verifier Agent 0.74 0.76 0.62 1.67 0.320.06
(-) Domain & Persona 0.91 0.88 0.52 1.81 0.11 0.11
Data RepresentationFixed Chunk Size (2048) 0.84 0.79 0.702.010.26 0.15
Image Only (No Description) 0.71 0.79 0.73 1.34 0.621.73
Description Only (No Image) 0.93 0.89 0.78 1.72 - 2.12
Without the multihop context, the generation re-
verts to simple extractive QA yielding a degrada-
tion in domain alignment (JSD≈4.35) as well.
Verifier Agent:The removal of the adversarial
verifier causes the most significant drop in faithful-
ness ( 0.97→0.74 ). Qualitative analysis reveals
that the generation agent frequently hallucinates
relationships between unconnected data points to
satisfy the prompt’s complexity requirements. The
drop in Relevance ( 0.95→0.76 ) further indicates
that unverified questions often drift from the core
semantic content of the context.
Domain/Persona Injection:Ablating the do-
main analysis prevents the model from adopting the
specific expert persona (e.g., ‘Financial Reporting
Analyst’ for the S&P Global annual reports) and
relevant domain (‘Corporate Financial Reporting
and Analysis’). The Difficulty of the generated
questions drops significantly ( 0.85→0.52 ). This
indicates that without domain/persona condition-
ing, the model defaults to generic QAs rather than
the deep, deductive reasoning characteristic of do-
main experts.
5.2.2 Impact of Data Representation
Chunking Strategy:We replaced the agentic se-
mantic chunking (Eq. 4) with a standard fixed-
window approach (2048 tokens). As shown in Ta-
ble 3, fixed chunk size marginally increased the av-
erage hops ( 1.92→2.01 ). The marginal change in
the average hops makes it inconclusive to highlight
the importance of semantic chunking to avoid frag-
mentation of tables and semantically continuous
text blocks. However, the drop in the faithfulness
and relevance allude to the importance of semantic
chunking.
Multimodal Configurations:We compared the
hybrid MiRAGE approach againstimage Only(no
descriptions) anddescription Only(no raw images)
configurations. TheImage Onlysetting results in
higher visual grounding ( 0.62) indicating the in-
creased importance of images. However, it is ac-companied by the lowest faithfulness ( 0.71) and
lower relevance indicating the struggle of VLMs
to bridge the semantic gap between the images
and the text. Conversely, thedescription Onlyset-
ting performs comparably with the full MiRAGE
indicating that LLMs can power MiRAGE if de-
scriptions of the visual artifacts are available.
6 Conclusion
We presentMiRAGE, a comprehensive multi-
agent framework that automates the generation
of high-fidelity, multimodal, multi-hop evaluation
datasets, addressing the critical need of domain-
specific benchmarks for high-stakes RAG applica-
tions. By orchestrating a swarm of specialized
agents through a recursivecontext optimization
loop,adversarial verification, anddomain-expert
recognition, MiRAGE successfully transcends the
limitations of linear synthetic pipelines, producing
datasets that exhibit deep deductive reasoning and
strict factual adherence across diverse corpora. Our
empirical evaluation of the finance, science, regula-
tions and journalism domains confirm the efficacy
of MiRAGE in preserving semantic dependencies
within complex technical documents. Our ablation
studies reveal that current state-of-the-art VLMs
still rely significantly on dense textual descriptions
to bridge the visual reasoning gap. Ultimately, Mi-
RAGE establishes a robust standard for automated
dataset generation, empowering organizations to
rigorously stress-test RAG systems against the la-
tent complexity and specific thematic distributions
of their proprietary data.
Limitations and Future Work
While MiRAGE advances the quality of synthetic
benchmarks, it is not without limitations. The mul-
tiagentics architecture is computationally intensive.
The multihop context building and QA verifica-
tion loops result in higher token costs and latency.
Future work will focus on optimizing the agen-
tic workflow for token efficiency. Exploring the
performance of open-source models, would help
democratize the framework.

References
Abdelrahman Abdallah, Bhawna Piryani, Jamshid
Mozafari, Mohammed Ali, and Adam Jatowt. 2025.
How good are LLM-based rerankers? an empiri-
cal analysis of state-of-the-art reranking models. In
Findings of the Association for Computational Lin-
guistics: EMNLP 2025, pages 5693–5709, Suzhou,
China. Association for Computational Linguistics.
Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and
William Cohen. 2022. Murag: Multimodal retrieval-
augmented generator for open question answering
over images and text. InProceedings of the 2022
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 5558–5570.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. Ragas: Automated evalua-
tion of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158.
Simone Filice, Guy Horowitz, David Carmel, Zohar
Karnin, Liane Lewin-Eytan, and Yoelle Maarek.
2025. Generating q&a benchmarks for rag evalu-
ation in enterprise settings. InProceedings of the
63rd Annual Meeting of the Association for Com-
putational Linguistics (Volume 6: Industry Track),
pages 469–484.
Negar Foroutan, Angelika Romanou, Matin Ansaripour,
Julian Martin Eisenschlos, Karl Aberer, and Rémi
Lebret. 2025. Wikimixqa: A multimodal benchmark
for question answering over tables and charts.arXiv
preprint arXiv:2506.15594.
Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric
Wallace, and Colin Raffel. 2023. Large language
models struggle to learn long-tail knowledge. In
International conference on machine learning, pages
15696–15707. PMLR.
Byeong Su Kim, Jieun Kim, Deokwoo Lee, and
Beakcheol Jang. 2025. Visual question answering:
A survey of methods, datasets, evaluation, and chal-
lenges.ACM Computing Surveys, 57(10):1–35.
Rajat Koner, Hang Li, Marcel Hildebrandt, Deepan Das,
V olker Tresp, and Stephan Günnemann. 2021. Graph-
hopper: Multi-hop scene graph reasoning for visual
question answering. InInternational Semantic Web
Conference, pages 111–127. Springer.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, and 1 others. 2019. Natural questions: a
benchmark for question answering research.Trans-
actions of the Association for Computational Linguis-
tics, 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Fuxiao Liu, Yinghan Wang, Tianlu Wang, and Vicente
Ordonez. 2021. Visual news: Benchmark and chal-
lenges in news image captioning. InProceedings of
the 2021 conference on empirical methods in natural
language processing, pages 6761–6771.
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigating
effectiveness of parametric and non-parametric mem-
ories. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 9802–9822.
Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen.
2025. A survey of multimodal retrieval-augmented
generation.arXiv preprint arXiv:2504.08748.
Rounak Meyur, Hung Phan, Sridevi Wagle, Jan Strube,
Mahantesh Halappanavar, Sameera Horawalavithana,
Anurag Acharya, and Sai Munikoti. 2025. Weqa:
A benchmark for retrieval augmented generation in
wind energy domain. InProceedings of the Fourth
Workshop on NLP for Positive Impact (NLP4PI),
pages 239–251.
Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao,
Saurabh Tiwary, Rangan Majumder, and Li Deng.
2016. Ms marco: A human-generated machine read-
ing comprehension dataset.
Zach Nussbaum, Brandon Duderstadt, and Andriy Mul-
yar. 2024. Nomic embed vision: Expanding the latent
space.arXiv preprint arXiv:2406.18587.
Chanhee Park, Hyeonseok Moon, Chanjun Park, and
Heui-Seok Lim. 2025. Mirage: A metric-intensive
benchmark for retrieval-augmented generation eval-
uation. InFindings of the Association for Computa-
tional Linguistics: NAACL 2025, pages 2883–2900.
Shraman Pramanick, Rama Chellappa, and Subhashini
Venugopalan. 2024. Spiqa: A dataset for multimodal
question answering on scientific papers.Advances in
Neural Information Processing Systems, 37:118807–
118833.
Julian Schnitzler, Xanh Ho, Jiahao Huang, Florian
Boudin, Saku Sugawara, and Akiko Aizawa. 2024.
Morehopqa: More than multi-hop reasoning.arXiv
preprint arXiv:2406.13397.
Shangeetha Sivasothy, Scott Barnett, Stefanus Kurni-
awan, Zafaryab Rasool, and Rajesh Vasa. 2025. Rag-
probe: Breaking rag pipelines with evaluation sce-
narios. In2025 IEEE/ACM 4th International Confer-
ence on AI Engineering–Software Engineering for AI
(CAIN), pages 60–71. IEEE.

S&P Global. 2023. Annual reports.
https://investor.spglobal.com/
sec-filings-reports/annual-reports/ . Ac-
cessed: 2023-10-27.
Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.arXiv preprint arXiv:2401.15391.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
United Nations Economic Commission for Europe.
2023. Global technical regulations (GTRs).
https://unece.org/transport/standards/
transport/vehicle-regulations-wp29/
global-technical-regulations-gtrs . Ac-
cessed: 2023-10-27.
Ian Wu, Sravan Jayanthi, Vijay Viswanathan, Simon
Rosenberg, Sina Khoshfetrat Pakazad, Tongshuang
Wu, and Graham Neubig. 2024. Synthetic multi-
modal question generation. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2024,
pages 12960–12993.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Associa-
tion for Computational Linguistics ACL 2024, pages
6233–6251.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Ruochen Zhao, Hailin Chen, Weishi Wang, Fangkai
Jiao, Xuan Long Do, Chengwei Qin, Bosheng Ding,
Xiaobao Guo, Minzhi Li, Xingxuan Li, and 1 oth-
ers. 2023. Retrieving multimodal information for
augmented generation: A survey.arXiv preprint
arXiv:2303.10868.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, and 1 others.
2023. Judging llm-as-a-judge with mt-bench and
chatbot arena.Advances in neural information pro-
cessing systems, 36:46595–46623.
Yihang Zheng, Bo Li, Zhenghao Lin, Yi Luo, Xuanhe
Zhou, Chen Lin, Guoliang Li, and Jinsong Su. 2025.
Revolutionizing database q&a with large language
models: Comprehensive benchmark and evaluation.
InProceedings of the 31st ACM SIGKDD Confer-
ence on Knowledge Discovery and Data Mining V . 2,
pages 5960–5971.A Prompts
PROMPTS_DESC["description"]
Provide a technical summary of the image/table
for documentation.
Format: Single continuous paragraph, under 250
words. No bullets.
Content Requirements:
1. Identify the image type (table/figure) and
its primary objective.
2. Technical Analysis:
- For Plots/Charts: Define axes/units,
variables, and key trends/regions.
- For Diagrams: Identify components,
connection flows, and system boundaries.
- Note: Describe visual attributes only if
they encode data; ignore decorative elements
and metadata.
3. Conclusion: Summarize critical insights and
practical design implications.
PROMPTS_CHUNK["semantic_chunking"]
You are a Semantic Chunking Engine. Segment
markdown into coherent, verbatim chunks.
Processing Rules:
1. **Exclusions:** Ignore Table of Contents and
Lists of Figures/Tables.
2. **Cohesion:** Merge orphan titles and short
subsections into adjacent text. Ensure
chunks are semantically self-contained.
3. **Status:** Mark as INCOMPLETE only if the
final chunk ends abruptly (cut-off).
Chunk Classifications:
- **figure**: Image (‘![...]‘) with "Figure X"
caption, description, and key. Set <artifact
> to image path.
- **standalone image**: Image *without* "Figure
X" caption. Set <artifact> to image path.
- **table**: "Table X" caption + markdown table
+ footnotes. Set <artifact> to ’None’.
- **table with images**: Table containing
‘![...]‘. Set <artifact> to image path(s).
- **text**: Paragraphs, lists, or definitions.
Set <artifact> to ’None’.
Output Format:
<chunk_id><|#|><chunk_type><|#|><content><|#|><
artifact><|#|><status><|#|><chunk_end>
Field Definitions:
- chunk_id: Sequential integer starting at 1.
- chunk_type: text | table | table with images |
figure | standalone image
- content: Exact unmodified markdown.
- artifact: Extracted image path(s) or ‘None’.
- status: COMPLETE | INCOMPLETE
PROMPTS["domain_and_expert_from_topics"]
I have analyzed a technical document collection
and extracted the following key topics:
{topic_list_str}
Based on these topics, please determine:
1. The specific technical or professional domain
these topics belong to.
2. A specific expert role title for a
professional in this domain.
Format your response exactly as follows (do not
add any other text):
<|#|>START<|#|>
<|#|>Domain: <The Domain>

<|#|>Expert Role: <The Expert Role>
<|#|>END<|#|>
PROMPTS_CHUNK["completion_verification"]
You are a Chunk Completion Verification Agent.
Evaluate if the provided text is
semantically self-contained given that you
are a(n) {expert_persona} working in the {
domain} domain .
Criteria for INCOMPLETE status:
1. Missing Artifacts: References to Figures,
Tables, or Sections not present in the chunk
(e.g., "see Figure 1").
2. Undefined Context: Acronyms, technical terms,
or variables used without definition or
prior explanation.
3. Broken Continuity: Implicit references (e.g.,
"as mentioned above," "this method") or
text describing a missing visual.
4. Rule: Do not assume expert inference. If a
definition or artifact is missing, it is
INCOMPLETE. Universal units are allowed.
Instructions:
- If COMPLETE: Confirm self-containment.
- If INCOMPLETE: Generate specific search
queries to retrieve the missing definitions
or artifacts.
Required Output Format:
Status: COMPLETE, Query: None, Explanation: <
brief reasoning>
OR
Status: INCOMPLETE, Query: <
specific_search_query_1> | <
specific_search_query_2>, Explanation: <list
missing refs/definitions>
PROMPTS_CHUNK["chunk_addition_verification"]
You are a Chunk Addition Verification Agent ({
expert_persona}, {domain}). Determine how a
CANDIDATE CHUNK relates to an INCOMPLETE
ORIGINAL CHUNK based on a specific SEARCH
QUERY.
Classify as:
1. EXPLANATORY: Directly resolves the missing
element. It provides the specific figure/
table, defines the unknown term, or supplies
the explicitly referenced prior context.
2. RELATED: Contextually relevant but does not
solve the specific gap. Includes general
theory, complementary data, or content
useful for multi-hop QA (even if it
references the same missing artifact).
3. UNRELATED: No semantic overlap or domain
relevance.
Output Format:
Status: <EXPLANATORY | RELATED | UNRELATED>
Explanation: <Brief justification>
PROMPTS["multi_hop_qa_generation"]
You are a(n) {expert_persona} in {domain_context
}. Construct a high-quality Question-Answer
pair by synthesizing information across the
provided text chunks.
Content:
{content}
**Execution Protocol (Strict Order):**
1. **Chunk Count:** Identify how many distinctchunks are present.
2. **Keyword Extraction:** List critical
technical keywords for *each* chunk.
3. **Relationship Mapping:** Identify "Bridge
Keywords"---concepts that relate or
intersect across the chunks.
4. **QA Synthesis:** Frame question-answer pairs
that requires synthesizing these related
keywords. The question must be unsolvable
without combining info from multiple points.
5. **Decomposition:** Map specific parts of the
Question and Answer back to their source
chunks.
**Critical Constraints:**
- **No Hallucination:** Answer ONLY using
provided content.
- **Self-Sufficiency:** The question must be
standalone. NEVER use phrases like "the
provided figure," "the text above," or "
Section 2.1" without context. Explicitly
name the object (e.g., "In Figure XX from
document YY...").
- **Complexity:** The question must be multi-hop
(requires connecting A to B).
**Output Format:**
<|#|>ANALYSIS<|#|>
Chunk Count: <Integer>
Keywords per Chunk: <Chunk 1: [A, B], Chunk 2: [
C, D]>
Related Keywords: <[A] relates to [C] via...>
<|#|>QA_GENERATION<|#|>
Question: <Your specific, self-contained
question>
Answer: <Concise technical answer>
Relevance: <0-10>
Difficulty: <0-10>
<|#|>DECOMPOSITION<|#|>
Question Source: <"Part of Q" -> derived from
Chunk X>
Answer Source: <"Part of A" -> derived from
Chunk Y>
<|#|>END<|#|>
PROMPTS["question_answer_verification"]
You are a QA Verification Agent ({expert_persona
}, {domain_context}). Evaluate the validity
of the following QA pair.
Context: Users see the Question WITHOUT the
Content. The Question must be completely
self-contained.
Evaluation Rules:
1. **Standalone Principle (QUESTION_CORRECT |
INCORRECT):**
- The Question is INCORRECT if it relies on
implicit context or vague references (e.g.,
"the provided figure", "this table", "the
described method", "Section 2").
- The Question is CORRECT only if it
explicitly names the subject, standard, or
artifact (e.g., "In Figure XX of the
document YY", "The inflation-time chart...")
.
2. **Factuality (ANSWER_CORRECT | INCORRECT):**
- The Answer must be factually supported by
specific data/principles in the Content.
3. **Necessity (REQUIRES_CONTENT |
CAN_ANSWER_WITHOUT_CONTENT):**
- Determine if the specific provided Content
is required to answer, or if general domain

knowledge suffices.
Inputs:
Content: {content}
Question: {question}
Answer: {answer}
Required Output Format:
QUESTION_[CORRECT|INCORRECT]
ANSWER_[CORRECT|INCORRECT]
[REQUIRES_CONTENT|CAN_ANSWER_WITHOUT_CONTENT]
Justification: <Brief reasoning>
PROMPTS["rerank_vlm"]
You are a VLM Reranking Agent. Rank the provided
chunks (text and images) by relevance to
the Query.
Input Markers: ‘<CHUNK_START id=N>‘ and ‘<
IMAGE_START>‘.
Instructions:
1. Analyze both textual context and visual data
to determine relevance.
2. List ALL chunk IDs in descending order (Rank
1 = Highest).
3. Output strictly in this format (no
conversational text):
<Rank 1>Chunk <id>
<Rank 2>Chunk <id>
<Rank N>Chunk <id>
PROMPTS["deduplication_rank"]
You are a Data Curator ({expert_persona}, {
domain}). Reorder the provided cluster of QA
pairs based on their relationship to the
core topic.
Task: Sort the list from **Most Distinct/Unique
** (least similar) to **Most Representative/
Redundant** (most similar).
Constraint: Preserve all text verbatim. Do not
omit sub-questions or modify content.
Candidates:
{candidates_text}
Required Output Format:
<|#|>START<|#|>
Question<|#|><Question_Text><|#|>Answer<|#|><
Answer_Text>
<|#|>NEXT<|#|>
Question<|#|><Question_Text><|#|>Answer<|#|><
Answer_Text>
<|#|>END<|#|>
PROMPTS["deduplication_merge"]
You are a Data Curator ({expert_persona}, {
domain}). Synthesize the provided QA cluster
into the MINIMAL set of high-quality pairs.
Processing Logic:
1. **Merge:** Combine complementary pairs into
comprehensive questions (integrating sub-
questions) and unified answers.
2. **Deduplicate:** Select the single best
version for exact or near-duplicates.
3. **Goal:** Zero redundancy while retaining
full information coverage.
Input Candidates:
{candidates_text}
Required Output Format:
<|#|>START<|#|>
Question<|#|><Merged/Refined Question><|#|>
Answer<|#|><Merged/Refined Answer><|#|>NEXT<|#|>
Question<|#|><Second Pair (if needed)><|#|>
Answer<|#|><Second Answer>
<|#|>END<|#|>