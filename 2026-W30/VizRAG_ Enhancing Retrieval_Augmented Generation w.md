# VizRAG: Enhancing Retrieval-Augmented Generation with Hypergraph Visualization

**Authors**: Yanbin Wei, Yang Chen, Renling Gan, Ziru Liu, Xinyu Fu, Chun Kang, Ning Lu, Rui Liu, Yu Zhang, James Kwok

**Published**: 2026-07-22 07:09:46

**PDF URL**: [https://arxiv.org/pdf/2607.19830v1](https://arxiv.org/pdf/2607.19830v1)

## Abstract
Hypergraph-based RAG systems surpass traditional graph-based approaches by organizing complex n-ary atomic facts among entities, rather than relying solely on binary relationships. Despite the advancements in multimodal large language models (MLLMs) with enhanced visual capabilities, current hypergraph-based RAG frameworks predominantly restrict knowledge retrieval and reconstruction to a unimodal, text-centric paradigm. This limitation prevents them from fully leveraging the powerful visual perception capabilities of modern MLLMs. To address this gap, we systematically explore the integration of hypergraph awareness in RAG systems through visual cues. By incorporating visual representations of hypergraphs into the RAG pipeline, we introduce VizRAG, the first RAG system to support visual hypergraph structure awareness. Experimental results demonstrate that VizRAG significantly outperforms strong baselines, validating the promising potential of hypergraph visualization as a novel approach for RAG systems.

## Full Text


<!-- PDF content starts -->

VizRAG: Enhancing Retrieval-Augmented Generation with Hypergraph
Visualization
Yanbin Wei1,2*Yang Chen1*Renling Gan*Ziru Liu3Xinyu Fu3
Chun Kang4Ning Lu1,2Rui Liu3†Yu Zhang1†James Kwok2
1Southern University of Science and Technology
2Hong Kong University of Science and Technology
3Huawei Research4Beihang University
Abstract
Hypergraph-based Retrieval-Augmented Gen-
eration (RAG) systems surpass traditional
graph-based approaches by organizing com-
plexn-ary atomic facts among entities, rather
than relying solely on binary relationships. De-
spite the advancements in multimodal large lan-
guage models (MLLMs) with enhanced visual
capabilities, current hypergraph-based RAG
frameworks predominantly restrict knowledge
retrieval and reconstruction to a unimodal, text-
centric paradigm. This limitation prevents them
from fully leveraging the powerful visual per-
ception capabilities of modern MLLMs. To
address this gap, we systematically explore the
integration of hypergraph awareness in RAG
systems through visual cues. By incorporating
visual representations of hypergraphs into the
RAG pipeline, we introduce VizRAG, the first
RAG system to support visual hypergraph struc-
ture awareness. Experimental results demon-
strate that VizRAG significantly outperforms
strong baselines, validating the promising po-
tential of hypergraph visualization as a novel
approach for RAG systems.
1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a key framework for improving LLM perfor-
mance in knowledge-intensive tasks like question
answering, document analysis, and intelligent assis-
tance (Lewis et al., 2020; Gao et al., 2023; Li et al.,
2024). By leveraging external knowledge sources,
RAG reduces hallucinations in LLMs, enhancing
both the reliability and quality of generated outputs.
Moreover, RAG enables seamless integration of pri-
vate or domain-specific knowledge bases, allowing
LLMs to adapt effectively to specialized contexts.
However, conventional RAG systems predomi-
nantly utilize chunk-level retrieval based on vector
similarity (Asai et al., 2023; Yang et al., 2024).
*Equal contribution.While this approach achieves reasonable effective-
ness, it often overlooks inter-chunk relationships
and hierarchical semantic structures. As a result,
retrieved content may lack coherence and exhibit
weak connectivity, limiting the model’s ability to
structurally interpret knowledge, a critical factor
for producing context-rich and coherent responses.
To address these shortcomings, recent studies
have explored graph-based structures within RAG
frameworks. For instance, GraphRAG (Edge et al.,
2024) and LightRAG (Guo et al., 2024) utilize
knowledge graphs to enhance entity-level index-
ing and retrieval by explicitly modeling seman-
tic relationships. Hyper-RAG (Feng et al., 2025a)
and Cog-RAG (Hu et al., 2026) further extend this
paradigm by using hypergraphs to capture higher-
order relationships among multiple entities, offer-
ing greater expressiveness than traditional graphs.
However, the approaches remain predominantly
text-centric, restricting knowledge retrieval and rea-
soning to unimodal paradigms and failing to lever-
age advanced visual perception capabilities of mod-
ern multimodal large language models (MLLMs).
In contrast, humans often rely on visual intu-
ition to perceive and reason about complex struc-
tures, such as loops, clusters, or sets, within graphi-
cal representations. This cognitive advantage sug-
gests that integrating visualized hypergraphs into
the RAG process could complement text-based re-
trieval with multimodal benefits, such as intuitive
reasoning shortcuts and enriched structural under-
standing. Inspired by this insight, we systemati-
cally analyze theadvantages (Sec. 2.1)andfeasi-
bility (Sec. 2.2)of using hypergraph visualization
to enhance RAGs, to reveal the untapped potential
of hypergraph visualization as anovel direction.
Nevertheless, incorporating hypergraph visual-
ization into RAG systems poses unique challenges
(Sec. 2.3), includingvisual congestionandrender-
ing bias. Addressing these challenges is essential
to fully realize the potential of visual hypergraphs
arXiv:2607.19830v1  [cs.CL]  22 Jul 2026

in RAG systems. To this end, we proposeHyper-
Viz(Sec. 3), a toolkit that provides scientifically
grounded solutions to these challenges, making hy-
pergraph visualization an effective and accessible
plugin for RAG systems.
Building onHyperViz, we introduceVizRAG
(Sec. 4), a novel RAG framework that integrates
hypergraph visualization to enhance the retrieval
and reasoning capabilities of MLLMs. By combin-
ing visual hypergraph awareness deeply into the
retrieval strategies,VizRAGdistinguishes itself
from the prior works and significantly improves
the coherence and quality of generated responses
over strong baselines in extensive experiments.
Our contributions are summarized as follows:
•We introducehypergraph visualizationas
a novel and promising direction for enhanc-
ing RAG systems, systematically analyzing
its potential, feasibility, and challenges.
•We developHyperViz, a plug-and-play, well-
encapsulated toolkit that addresses key chal-
lenges in hypergraph visualization.
•We proposeVizRAG, a visualization-driven
RAG framework that incorporates hypergraph
visual awareness into the entire RAG pipeline,
enabling multimodal retrieval and reasoning.
•Extensive experiments show thatVizRAGsig-
nificantly outperforms strong RAG baselines
on multiple knowledge-intensive tasks, high-
lighting the promising enhancement of hyper-
graph visualization for RAG.
2 Hypergraph Visualization in RAG
In this section, we clarify the situation of hyper-
graph visualization in RAG, including its 1) advan-
tages, 2) feasibility, and 3) specific challenges.
2.1 Advantages
We first highlight why hypergraph visualization is a
meaningful extension for hypergraph-based RAG.
1) Structural Discrimination.Text-only retrieval
often over-relies on lexical overlap and shallow se-
mantics. In hypergraph-based RAG, many failure
cases arise fromstructurally mismatched but lex-
ically similarevidence: candidate passages may
share keywords with the query but encode different
high-order relations among entities. Hypergraph
visualization provides explicit spatial and group-
ing cues (e.g., overlap patterns, bridge entities, andcross-group links), helping MLLMs separate true
relational matches from superficial textual matches.
This enables finer-grained structural disambigua-
tion before final answer generation.
Note 2.1.Hypergraph visualization enables RAG
systems to achieve finer-grained structural discrim-
ination in hypergraphs.
2) Multi-hop Reasoning Composition.As the rea-
soning hop increases, relevant evidence becomes
more scattered across entities, hyperedges, and
chunks, making path composition fragile in text-
only settings. Hypergraph visualization offers an
externalized structural workspace: it exposes candi-
date evidence chains as visible substructures, mak-
ing it easier for MLLMs to identify plausible multi-
hop paths and reject shortcuts that break relational
constraints. This is particularly important for deep
reasoning, where correctness depends on compos-
ing multiple structurally consistent steps rather than
selecting a single highly similar passage.
Note 2.2.Visual structural cues bring increasingly
prominent benefits for complex deep multi-hop rea-
soning tasks.
To empirically validate these advantages, we
present experimental results in Sec. 5.3.
2.2 Feasibility
Apart from the benefits, we argue that hypergraph
visualization is feasible in current RAG systems:
(1) Usable MLLM Hypergraph Perception:Re-
cent off-the-shelf MLLMs already demonstrate
strong visual parsing ability for graph-like struc-
tures, including patterns like region overlap and
bridge-like connectors (as evidenced in Sec. 5.3).
These capabilities are tightly aligned with the
key signals required to understand the hypergraph
knowledge structure in RAGs, such as identify-
ing overlapping entities, analyzing multi-hop con-
nections, and interpreting relational constraints en-
coded in hyperedges. Because current MLLMs can
effectively parse hypergraph visualizations with-
out fine-tuning, integrating visualization into RAG
systems can be seamless.
(2) Lightweight System Cost:Although visualiza-
tion introduces extra rendering and transfer over-
head, this overhead is lightweight relative to end-
to-end RAG latency, especially when retrieval and
generation already dominate the runtime. A de-
tailed efficiency comparison is provided in Sec. 5.3,
where the extra cost of visualization is very small
compared to retrieval time. Therefore, hypergraph

Figure 1: As the chunk size increases to even encom-
pass the full document, more nodes and hyperedges are
included in the visualization, leading to visual conges-
tion that reduces structure clarity.
visualization does not introduce a significant com-
putational burden, making it suitable for deploy-
ment in existing RAG frameworks.
2.3 Specific Challenges
Though beneficial and feasible for integrating hy-
pergraph visualization into RAG, it still faces two
non-trivial challenges that hinder its effectiveness.
Visual Congestion.Knowledge-intensive cor-
pora often contain a high density of entities and
multi-entity hyperedges, leading to visual clutter in
hypergraph visualization. As shown in Fig. 1, the
bigger chunk size results in more node aggregation,
edge crossing, and structural overlap. Although
these features reveal critical structural cues, exces-
sive clutter can obscure reasoning paths and de-
grade performance. Dense node clusters may cause
localized visual overload, whereas sparse low-hop
structures risk leaving large blank spaces. Balanc-
ing information density and clarity requires em-
ploying reasonable sampling strategies and adap-
tive element size adjustments. To address these
challenges, effective structure filtering mechanisms
and spatial layout optimization techniques are es-
sential for preserving key information and ensuring
readability across varying graph densities.
Rendering Bias.Another major challenge in hy-
pergraph visualization is the uncontrollable render-
ing bias, where a single visualization emphasizes
specific hypergraph properties, leading to partial
or skewed structural understanding. Specifically,
existing hypergraph layouts can be broadly cate-
gorized into two paradigms: i)Set-based layouts,
which represent high-order relationships (i.e., hy-
peredges) as set elements, such as through region
containment, to highlight entity grouping relations.
Figure 2: Rendering bias illustration. (a) Layout vari-
ants for the same hypergraph; (b) Structure perception
tasks; (c) Performance gap caused by rendering bias.
ii)Induced layouts, which focus on direct belong-
ingship between entities and their belonging groups
by representing hyperedges as special nodes. In
this approach, the nodes within each hyperedge
are explicitly linked to the corresponding special
node via pairwise edges. While both approaches
provide valuable perspectives, they inherently em-
phasize different aspects of the hypergraph struc-
ture, which can lead to biased or incomplete inter-
pretations depending on the chosen visualization
method. To quantify this, we evaluate the Gemini-
3 Flash model’s structural perception onoverlap-
core identificationandbridge hyperedge detection,
where tasks are illustrated in Fig. 2(b), with 2,000
hypergraphs derived from the UltraDomain Le-
gal doc/chunks (Qian et al., 2025). As shown in
Fig. 2(c), significant discrepancies emerge between
set-based and Induced visualizations. However,
combining both layouts effectively mitigates uni-
lateral perception deviations, with the strongest
complementary effect observed when pairing set-
based and induced views. This highlights the need
for a unified paradigm in RAG systems to harness
layout-induced biases for improved performance.
3 HyperViz Toolkit
This section introducesHyperViz, a toolkit for
hypergraph visualization in RAG systems. Tak-
ing the document hypergraph and text descriptions
of entities/hyperedges, HyperViz generates visu-
alization views that tackle visual congestion and
rendering bias. By integrating structural and se-
mantic considerations, it ensures document-based
hypergraph visualization aligns with RAG require-
ments. To keep focus, we highlights its solutions to
the aforementioned challenges in this section, with
additional features are leaved in App. B.1.

3.1 Solution to Visual Congestion
To address visual congestion, HyperViz employs a
multi-step optimization strategy:
(1) Weighted Space Allocation:Given a hyper-
graph G= (V, E) , where Vrepresents entity
nodes and Erepresents hyperedges, we assign dis-
tinct space weight coefficients wvfor nodes and we
for hyperedges1. The total visual occupation cost is
expressed as C=P
v∈VAv·wv+P
e∈ELe·we,
where AvandLedenote the actual visual areas and
lengths occupied by entity nodes and hyperedges,
respectively. To ensure clarity, a maximum allow-
able cost Cmaxis defined as Cmax=γA , where A
is the area of the predefined canvas and γcontrols
the congestion threshold. Together, these formu-
lations quantify the visual occupation by area and
define an upper bound for maintaining clarity.
(2) Pruning Low-Importance Elements:When C
>Cmax, we rank all entities in Gby an importance
score Importance(e) , defined as Importance(e) =
α·Centrality(e) +β·Relevance(e) , where
Centrality(e) is the degree centrality of the entity
einG, and Relevance(e) reflects a query-specific
semantic alignment by the BM25 score between
the text descriptions of eand Query Q, and α,β
denote weighting coefficients. Based on this rank-
ing, pruning is applied to the low-importance nodes,
and they are eliminated from their associated hyper-
edges until C≤C max. This ensures that the result-
ing pruned hypergraph retains critical high-order
relationships while significantly reducing visual
clutter, improving interpretability and usability.
(3) Global Uniform Scaling: The filtered hy-
pergraph is then scaled to fit the canvas dimen-
sions, maximizing canvas utilization and eliminat-
ing marginal blank spaces to avoid space wasting.
In summary, the multi-step pipeline effectively
mitigates visual congestion. As an illustration
shown in Fig. 1, the pruned visualization gener-
ated by HyperVizpreserves claritywhileretaining
key structural and semantic informationin the doc-
ument knowledge.
3.2 Solution to Rendering Bias
To fit diverse scenarios and increase the usability
of Hyperviz, we categorize the addressing of ren-
dering bias into two scenarios:
1wv, we∈[1,10] are grid-searched on large knowledge
hypergraphs without data leakage under Gemini-3 Flash su-
pervision via tasks in Fig.2. More hyperparameter details are
in App. B.2(1) User-driven Customization:To support tasks
requiring specific rendering biases, HyperViz em-
ploys a layout-centric design encompassing both
set-based and Induced paradigms, each offering
three layout variants. The set-based layouts in-
clude i) Set-Hier, ii) Set-Radial, and iii) Set-Force,
while the Induced layouts consist of i) Bi-Inc, ii)
Ra-Cen, and iii) Cli-Exp. They are illustrated in
Fig. 2(a), with detailed descriptions provided in
App. B.3. Users can flexibly select a single layout
or combine multiple layouts as needed.
(2) Automated Bias Mitigation:To address un-
controlled rendering biases, HyperViz incorporates
aprobe-driven dual-viewmechanism as the de-
fault approach. Specifically, a small batch of k
queries ( k= 15 in our experiments, random from
training sets) is used to evaluate all nine combina-
tions of set-based and induced-graph layouts, in-
spired by their complementary strengths. The lay-
out combination that best enhances hypergraph per-
ception onoverlap-core identificationandbridge
hyperedges detection(illustrated in Fig. 2(b))2is
selected, and the two selected layouts are ren-
dered side-by-side in a dual-view format within
a single image, as demonstrated in Fig. 2(c) left-
top. Results in Fig. 2(c) show that the dynamic
probe-driven dual-view significantly improves hy-
pergraph perception on both tasks compared to ei-
ther single layouts or their random combinations.
This finally benefits the RAG quality (Sec. 5.3).
3.3 Plug-and-Play Integration
HyperViz serves as a middleware that bridges
knowledge hypergraphs and visualization work-
flows tailored to RAG scenarios. It seamlessly inte-
grates with existing RAG pipelines in a plug-and-
play manner, offering well-encapsulated interfaces
to ensure ease of use for users.
4 VizRAG Framework
In this section, we introduce the VizRAG frame-
work (illustrated in Fig. 3), where visualization
plays a dual role in both indexing and retrieving.
4.1 Vision-awared Bottom-up Indexing
Given a corpus D(e.g., books, reports, or manuals),
we segment it into a set of overlapping chunks
using a fixed-length sliding window to preserve
2Both tasks are chosen by their similarity to essential RAG
requirements, such as identifying key entities within groups
and uncovering critical relationships among entity clusters.

Bottom-Up Indexing Top-Down Retrieval
Query :
What are the main features of
Apache Spark for distributed
computing?
Query Keywords :
1. Apache Spark
2. Distributed computing
...
Top Hypergraph
Top-k Relevant
Relevant
Hyper edges
Neighbor
EntitiesSubgraph
HyperV iz
Anchor
 Entities
Top-k Relevant
Bottom HypergraphRelevant
EntitiesNeighbor
Hyper edges
HyperV iz
Answer : Apache Spark provides
in-memory computation and fault
tolerance for efficient distributed
computing.Subgraph
Document
Book
Report...
Chunk 1
Chunk 2
...
Chunk N
Chunk 1 ThemeTheme Relations Key Entity
Chunk 2 Theme
...
Chunk N Theme......
...
...MergeTop HypergraphTop Inter -Chunk Indexing
Bottom Intra-Chunk IndexingChunk 1 
Entity & Relations
Entity RelationsChunk 2 
Entity & Relations
...
Chunk N  
Entity & Relations
Chunk-W ise HypergraphMerge
Bottom Hypergraph
MLLM
Hypergraph Visualizations
HDatabase
1. Semantic Embeddings
2. Hypergraphs
HyperV izS
Figure 3: The overall framework of VizRAG.
semantic continuity:
D={D 1, D2, . . . , D N},
where Direpresents the i-th chunk, serving as the
fundamental unit for subsequent processing.
We employ a two-layer hierarchical hypergraph
indexing approach: (i) bottom-layer intra-chunk in-
dexing, capturing fine-grained intra-chunk seman-
tics. (ii) top-layer inter-chunk indexing, capturing
global relationships across chunks.
4.1.1 Bottom Intra-Chunk Indexing
For each chunk Di, we extract entities (e.g., per-
sons, events, organizations) and their descriptions
using LLMs. These entities form the vertex set VDi
of the fine-grained entity hypergraph. Semantic re-
lationships among the entities are used to construct
hyperedges EDi, capturing high-order associations
such as co-occurrence in events or causal links:
(
VDi= LLM(P ext_entity (Di)),
EDi= LLM(P ext_relation (Di,VDi)),
where Pext_entity andPext_relation are prompts de-
signed for entity and hyperedge extraction (detailed
in Appendix G).
Once the hypergraph GDi={V Di,EDi}is con-
structed, we generate its hypergraph visualization
using HyperViz:
IDi=HyperViz(G Di).
4.1.2 Top Inter-Chunk Indexing
Using predefined prompts Pext_theme andPext_key
(in App. G), MLLMs are guided to perform seman-
tic parsing on each chunk Di, extracting its theme
Etheme
Diand a subset of key entities Vkey
Direlevant to
the theme. Importantly, the intra-chunk knowledgestructure visualized in IDiis integrated into the
MLLM to enhance semantic understanding:
(
Etheme
Di= MLLM(P ext_theme (Di,IDi)),
Vkey
Di= MLLM(P ext_key (Etheme
Di, Di,IDi,VDi)).
4.1.3 Hierarchical Hypergraph Merging
Merging all extracted entities, their relations, and
chunk themes from all chunks, we get a fine-
grained bottom hypergraph GBotand a theme-aware
top hypergraph Gtop, which are connected via
shared vertices (i.e., theme-related key entities
{Vkey}). Both hypergraphs are stored in a hyper-
graph database.
(GBot={S
Di∈DVDi,S
Di∈DEDi},
GTop={S
Di∈DVkey
Di,S
Di∈DEtheme
Di}.
4.2 Vision-enhanced Top-down Retrieval
VizRAG employs a visualization-guided top-down
retrieval strategy, leveraging rendered hypergraph
images as structural bridges from the global theme
layerGTopto the fine-grained entity layer GBot.
Given a queryq, the keywords are first extracted:
Xquery= LLM(P keyword (q)),
where X∗={x 1, x2, . . .} andPkeyword is in
App. G.
4.2.1 Top Retrieval and Locating
These keywords Xquery are used to retrieve the top-
krelevant hyperedges ErelfromGTopbased on vec-
tor similarity R(·,·) , where ETopare edges in GTop:
Erel=S
xi∈X queryR(x i,ETop).
We then construct a h-hop induced subgraph Gsub
Top
centered around the retrieved hyperedges Ereland
visualize it using HyperViz:
ITop=HyperViz(Subgraph(G Top,Erel,hops=h)).

VizRAG then exploits the MLLM’s visual struc-
tural perception to locate key entities from ITop,
union with the direct neighbors of hyperedges Erel:
ˆV=MLLM(P locate(q,I Top,Ce_rel))∪ V dif,
Vdif=S
ei∈ErelN(e i,GTop),
where Ndenotes retrieving neighbors from the
hypergraph, Ce_relis the relevant context of e_rel∈
Erel,Plocate prompts the MLLM to locate query-
relevant entities in the rendered subgraph structure.
4.2.2 Bottom Retrieval and Answer
Generation
With ˆVas entity anchors, we retrieve the top- k
matching vertices from GBotand expand context via
their neighbor hyperedges, whereV Bot={V Di}:
Vrel=S
vi∈ˆVR(vi,VBot),
Edif=S
vi∈VrelN(v i,GBot).
The retrieved entity subgraph is rendered as a visual
cueI Botto expose fine-grained evidence paths:
IBot=HyperViz(Subgraph(G Bot,Vrel,hops=h)).
Finally, Vrel,Edif, their corresponding contexts
Cv_rel,Ce_dif, and the visualization IBotare jointly
fed to the MLLM to generate the final answer:
A= MLLM(q,I Bot,Vrel,Edif,Cv_rel,Ce_dif).
5 Experiments
In this section, we perform extensive experiments
to evaluate the proposed methods. The experimen-
tal setup is aligned with prior works (Hu et al.,
2026; Feng et al., 2025b).
5.1 Experimental Setup
To systematically evaluate our method across di-
verse application scenarios, we adopt five datasets
from two benchmarks: Mix, CS, and Agriculture
from the UltraDomain benchmark (Qian et al.,
2025), and Neurology and Pathology from the MI-
RAGE benchmark (Xiong et al., 2024). UltraDo-
main covers typical RAG applications across dif-
ferent domains, while MIRAGE focuses on medi-
cal question answering and domain-specific knowl-
edge coverage. Data statistics are in App. C.
Based on domain consistency and semantic cor-
relation within the texts, we categorize the datasets
into three types to enable a comprehensive anal-
ysis of the model’s adaptability:Cross-domainSparse(Mix): Fragmented passages from unrelated
domains with weak semantic coherence.Intra-
domain Sparse(CS, Agriculture): Domain-specific
documents with weak inter-passage context.Intra-
domain Dense(Neurology, Pathology): Highly
structured medical textbooks with strong semantic
continuity from MIRAGE. Additionally, we fol-
low the data processing and query procedure of
LightRAG, utilizing Gemini-3 Flash to generate
complex, document-related queries. Besides, to en-
hance the robustness and reduce stochasticity, we
evaluate 10 times for each result. All experiments
are conducted on 4 A100-80G GPUs.
Baselines.We compared our approach with the
state-of-the-art and popular Graph-enhanced RAG
methods: GraphRAG (Edge et al., 2024), Ligh-
tRAG (Guo et al., 2024), HiRAG (Huang et al.,
2025), and hypergraph RAGs including Hyper-
RAG (Feng et al., 2025a) and Cog-RAG (Hu et al.,
2026). The baseline details are in App. D.
Implementation Details.To ensure fairness and
consistency for both the baselines and proposed
methods, we validate on five different LLMs for
information extraction and question answering, in-
cluding Gemini-3 Flash (Pichai et al., 2025), GPT-5
Mini (Singh et al., 2025), GLM-5V Turbo (Hong
et al., 2026), Claude Haiku 4.5 (Anthropic, 2025),
and Qwen3VL-8B (Bai et al., 2025). The result
evaluation is default on Gemini-3 Flash, as well as
the text-embedding-3-small embedding model for
vector encoding and retrieval tasks.
Evaluation Metrics.Following the recent works,
we adopt two evaluation strategies: Selection-based
(Guo et al., 2024; Huang et al., 2025) and Score-
based (Wang et al., 2024; Feng et al., 2025a), pro-
viding both relative and absolute perspectives on
model performance.The Selection-basedeval-
uation uses LLMs to report win rates of answer
quality between two methods.The Score-based
evaluation employs LLMs to score responses of dif-
ferent methods. Both strategies assess models from
six dimensions: Comprehensiveness, Empower-
ment, Relevance, Consistency, Clarity, and Logical.
We report both per-dimension and overall average
scores. More Evaluation Details are in App. E.
5.2 Main Results
Our primary results are presented in Tab. 1 and
Fig. 4, and more results are provided in App. F.
VizRAG consistently outperforms all baselines
across multiple dimensions. Additionally, we have
several key insights:

Mix CS Agriculture Neurology Pathology
GraphRAGVizRAGGraphRAGVizRAGGraphRAGVizRAGGraphRAGVizRAGGraphRAGVizRAG
Comp. 28.7%71.3%24.5%75.5%20.8%79.2%22.7%77.3%20.5%79.5%
Empo. 24.6%75.4%23.2%76.8%14.8%85.2%15.6%84.4%11.7%88.3%
Rele. 33.9%66.1%27.5%72.5%23.1%76.9%25.2%74.8%19.6%80.4%
Cons. 28.9%71.1%23.7%76.3%17.9%82.1%19.5%80.5%18.6%81.4%
Clar. 34.7%65.3%24.8%75.2%26.5%73.5%24.2%75.8%18.5%81.5%
Logi. 27.8%72.2%25.1%74.9%15.9%84.1%21.6%78.4%17.8%82.2%
Overall 29.8%70.2%24.8%75.2%19.8%80.2%21.5%78.5%17.8%82.2%
LightRAGVizRAGLightRAGVizRAGLightRAGVizRAGLightRAGVizRAGLightRAGVizRAG
Comp. 30.5%69.5%22.8%77.2%16.6%83.4%20.9%79.1%22.9%77.1%
Empo. 22.2%77.8%19.5%80.5%12.6%87.4%15.7%84.3%17.6%82.4%
Rele. 28.9%71.1%19.8%80.2%17.7%82.3%20.7%79.3%24.8%75.2%
Cons. 26.8%73.2%21.5%78.5%13.7%86.3%17.8%82.2%19.7%80.3%
Clar. 30.6%69.4%16.7%83.3%14.8%85.2%18.5%81.5%18.8%81.2%
Logi. 27.7%72.3%21.6%78.4%15.6%84.4%18.8%81.2%18.9%81.1%
Overall 27.8%72.2%20.3%79.7%15.2%84.8%18.7%81.3%20.4%79.6%
HiRAGVizRAGHiRAGVizRAGHiRAGVizRAGHiRAGVizRAGHiRAGVizRAG
Comp. 36.6%63.4%32.8%67.2%33.7%66.3%27.5%72.5%32.9%67.1%
Empo. 31.5%68.5%28.7%71.3%28.5%71.5%23.6%76.4%29.6%70.4%
Rele. 37.7%62.3%39.8%60.2%36.7%63.3%27.6%72.4%33.8%66.2%
Cons. 31.6%68.4%32.6%67.4%29.6%70.4%24.7%75.3%29.9%70.1%
Clar. 38.6%61.4%42.7%57.3%36.8%63.2%23.5%76.5%32.7%67.3%
Logi. 32.6%67.4%32.7%67.3%30.8%69.2%23.8%76.2%28.6%71.4%
Overall 34.8%65.2%34.9%65.1%32.7%67.3%25.1%74.9%31.2%68.8%
Hyper-RAGVizRAGHyper-RAGVizRAGHyper-RAGVizRAGHyper-RAGVizRAGHyper-RAGVizRAG
Comp. 37.7%62.3%38.6%61.4%40.8%59.2%31.6%68.4%33.8%66.2%
Empo. 34.9%65.1%35.7%64.3%31.7%68.3%28.6%71.4%28.8%71.2%
Rele. 47.2%52.8%40.7%59.3%38.7%61.3%39.2%60.8%28.7%71.3%
Cons. 34.6%65.4%36.2%63.8%36.8%63.2%29.8%70.2%26.6%73.4%
Clar. 46.6%53.4%39.7%60.3%33.6%66.4%32.7%67.3%24.7%75.3%
Logi. 35.6%64.4%37.6%62.4%34.7%65.3%26.8%73.2%28.7%71.3%
Overall 39.4%60.6%38.1%61.9%36.0%64.0%31.4%68.6%28.5%71.5%
Cog-RAGVizRAGCog-RAGVizRAGCog-RAGVizRAGCog-RAGVizRAGCog-RAGVizRAG
Comp. 40.7%59.3%42.6%57.4%46.6%53.4%37.7%62.3%39.7%60.3%
Empo. 39.5%60.5%40.5%59.5%37.8%62.2%34.6%65.4%34.7%65.3%
Rele. 44.7%55.3%44.6%55.4%42.5%57.5%43.7%56.3%34.7%65.3%
Cons. 40.6%59.4%41.6%58.4%41.7%58.3%35.8%64.2%33.5%66.5%
Clar. 47.6%52.4%45.7%54.3%39.8%60.2%38.7%61.3%29.8%70.2%
Logi. 41.6%58.4%43.7%56.3%40.7%59.3%32.6%67.4%34.7%65.3%
Overall 42.4%57.6%43.1%56.9%41.5%58.5%37.2%62.8%34.5%65.5%
Table 1: Average win rates of six evaluation metrics across five datasets. The comparison is made between baselines
and VizRAG. Among them, we refer to the six metrics as Comp. (Comprehensiveness), Empo. (Empowerment),
Rele. (Relevance), Cons. (Consistency), Clar. (Clarity), and Logi. (Logical).
1)VizRAG consistently wins against all base-
lines on selection-based evaluation.As shown
in Tab. 1, the overall win rates are 70.2%–82.2%
vs. GraphRAG, 72.2%–84.8% vs. LightRAG,
and 65.1%–74.9% vs. HiRAG across five datasets.
VizRAG also outperforms strong hypergraph base-
lines, with 60.6%–71.5% vs. Hyper-RAG and
56.9%–65.5% vs. Cog-RAG. This confirms the
effectiveness of VizRAG on pairwise comparison.
2)VizRAG brings robust and comprehensive
improvement on response quality.Figs. 4(a) and
4(b) show that VizRAG achieves higher scores
acrossall datasetsandall evaluation dimensions,
and Fig. 4(c) shows the same trend acrossall tested
LLM backbones. These results verify that addingvisual hypergraph cues delivers persistent and ro-
bust gains for hypergraph-based RAG systems.
5.3 Ablation Study
Effects of Method Components.We report score-
based ablations on three representative datasets in
Tab. 2 to validate the necessity of key components
in our proposed methods: 1) visualization in index-
ing, 2) visualization in retrieval, 3) HyperViz for
addressing visual congestion, and 4) HyperViz for
addressing rendering bias. As the results shown in
Tab. 2, removing any component leads to obvious
performance degradation on all datasets, confirm-
ing their empirical effectiveness.
Advantages of Structural Discrimination.To

Figure 4: Test results by scoring. (a) is the comparison results on five datasets; (b) is the comparison results on six
dimensions; (c) shows the evaluation results on different LLMs.
Models Mix CS Neurology
VIZRAG 88.0 89.7 89.9
w./o. Indexing Visualization87.3 88.5 89.4
w./o. Retrieval Visualization86.8 87.3 88.2
w./o. HyperViz (Address Visual Congestion)87.5 88.1 88.9
w./o. HyperViz (Probe-driven Dual-View)87.9 88.4 89.2
Table 2: Ablation results by overall scores. All ablated
variants underperform full VizRAG on all datasets.
evaluate whether hypergraph visualization im-
proves structural disambiguation, we define a bi-
nary classification task: distinguishing between
structurally aligned positive candidates and lexi-
cally similar but structurally mismatched negative
candidates. Using 200 query groups from the Ul-
traDomain dataset (Qian et al., 2025), each query
is paired with one correct match (positive) and one
distractor (negative). The goal is to classify each
candidate as “positive” or “negative”. As shown in
Tab. 3, integrating hypergraph visualization into the
Hypergraph RAG pipeline (Feng et al., 2025b) im-
proves accuracy from 71.5% to 79.0%, highlighting
its effectiveness in resolving structural ambiguities.
Method Accuracy (%)
Hypergraph RAG w/o Visualization 71.5
Hypergraph RAG w/ Visualization 79.0
Table 3: Structural disambiguation results on lexically
matched hard negative samples.
Enhancing Multi-hop Reasoning Composition.
To assess whether hypergraph visualization im-
proves multi-hop reasoning, we evaluate 900 ques-
tions (evenly split across 2-hop, 3-hop, and 4-hop)
from MusiQue (Trivedi et al., 2022) and HotpotQA
(Yang et al., 2018). Using Exact Match (EM) as
the metric, we compare the Hyper-RAG pipeline
(Feng et al., 2025b) with and without visual hy-
pergraph input. Results in Tab. 4 show consistent
improvements across all reasoning depths, with
the most significant gains in 3-hop (+6.5%) and
4-hop (+10.6%) tasks, highlighting the utility of
hypergraph visualization in complex reasoning.
MLLM Hypergraph Perception Capability.We
also evaluate the hypergraph perception capabili-Reasoning EM w/o visualization w/ visualization
2-hop Reasoning 72.3 74.0
3-hop Reasoning 60.5 67.0
4-hop Reasoning 48.7 59.3
Table 4: Visualization effects on multi-hop reasoning.
Frozen MLLM Overlap-core Identification Bridge Hyperedge Detection
Qwen3VL-8B 86.8% 83.2%
Gemini-3 Flash 90.5% 88.9%
Table 5: Accuracy (%) of off-the-shelf MLLMs on key
structural detection tasks.
ties of representative frozen off-the-shelf MLLMs
through two key tasks:overlap-core identification
andbridge-hyperedge detection. Both tasks (illus-
trated in Fig. 2(b)) are conducted on a fixed 1,000-
sample subset from the UltraDomain dataset. As
shown in Tab. 5, all MLLMs demonstrate notable
accuracy in identifying these structural cues from
hypergraphs, which serves as a strong foundation
for integrating hypergraph visualization into their
reasoning processes within RAG pipelines.
Efficiency Analysis.We analyze the retrieval
Figure 5: Trade-off between
retrieval time and score.efficiency for the
VizRAG and base-
lines. As shown
in Fig. 5, VizRAG
achieves the high-
est overall score
with low retrieval
time, striking a bal-
ance between efficiency and performance. The
optimized visualization adds only 0.21 seconds per
query, a lightweight overhead compared to the base-
line, while yielding significant score gains (Tab. 2).
This makes VizRAG ideal for scenarios demanding
both accuracy and speed.
6 Conclusion
We introduce hypergraph visualization as a novel
enhancement for RAG systems, along with the
plug-and-play HyperViz toolkit and VizRAG
framework, enabling multimodal hypergraph rea-
soning and achieving significant gains in real-world

knowledge-intensive tasks.
Limitations
While VizRAG demonstrates promising results, it
faces several limitations. Its performance heavily
depends on the chosen MLLM’s ability to perceive
visual structures, which may limit its effectiveness
in scenarios involving complex visual reasoning
tasks. Rendering combined with dual-view prob-
ing introduces additional system overhead, particu-
larly in latency-sensitive applications. Furthermore,
the reliance on LLM-as-a-judge evaluations may
lead to evaluator bias. Future work will focus on
developing adaptive visualization strategies that dy-
namically adjust to task-specific requirements and
incorporating broader, human-centered evaluation
methods to reduce bias and improve robustness.
References
AI Anthropic. 2025. System card: Claude opus 4 &
claude sonnet 4.Claude-4 Model Card.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2023. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations.
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhi-
fang Guo, Qidong Huang, Jie Huang, Fei Huang,
Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng
Li, and 45 others. 2025. Qwen3-vl technical report.
arXiv preprint arXiv:2511.21631.
Alain Bretto. 2013. Hypergraph theory.An introduction.
Mathematical Engineering. Cham: Springer, 1:209–
216.
Haeyong Chung, Santhosh Nandhakumar, and Seung-
won Yang. 2020. Gridset: Visualizing individual
elements and attributes for analysis of set-typed data.
IEEE Transactions on Visualization and Computer
Graphics, 28(8):2983–2998.
Christopher Collins, Gerald Penn, and Sheelagh Carpen-
dale. 2009. Bubble sets: Revealing set relations
with isocontours over existing visualizations.IEEE
transactions on visualization and computer graphics,
15(6):1009–1016.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
Preprint, arXiv:2404.16130.Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu,
Shihui Ying, Shaoyi Du, Han Hu, and Yue Gao.
2025a. Hyper-rag: Combating llm hallucinations
using hypergraph-driven retrieval-augmented genera-
tion.Preprint, arXiv:2504.08758.
Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu,
Shihui Ying, Shaoyi Du, Han Hu, and Yue Gao.
2025b. Hyper-rag: Combating llm hallucinations
using hypergraph-driven retrieval-augmented genera-
tion.arXiv preprint arXiv:2504.08758.
Emden R Gansner and Stephen C North. 2000. An open
graph visualization system and its applications to soft-
ware engineering.Software: practice and experience,
30(11):1203–1233.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
Preprint, arXiv:2312.10997.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. 2024. Lightrag: Simple and fast retrieval-
augmented generation.Preprint, arXiv:2410.05779.
Aric Hagberg, Pieter Swart, and Daniel S Chult. 2008.
Exploring network structure, dynamics, and func-
tion using networkx. Technical report, Los Alamos
National Lab.(LANL), Los Alamos, NM (United
States).
Wenyi Hong, Xiaotao Gu, Ziyang Pan, Zhen Yang, Yut-
ing Wang, Yue Wang, Yuanchang Yue, Yu Wang,
Yanling Wang, Yan Wang, and 1 others. 2026. Glm-
5v-turbo: Toward a native foundation model for mul-
timodal agents.arXiv preprint arXiv:2604.26752.
Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue,
Xingliang Hou, Zhiqiang Tian, Yue Gao, and
Shaoyi Du. 2026. Cog-rag: Cognitive-inspired
dual-hypergraph with theme alignment retrieval-
augmented generation. InProceedings of the AAAI
Conference on Artificial Intelligence, volume 40,
pages 31032–31040.
Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu
Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen,
and James Cheng. 2025. Retrieval-augmented gen-
eration with hierarchical knowledge.Preprint,
arXiv:2503.10150.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Ro-
main Vuillemot, and Hanspeter Pfister. 2014. Up-
set: visualization of intersecting sets.IEEE trans-
actions on visualization and computer graphics,
20(12):1983–1992.

Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin,
Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei Han,
Le Sun, and Yongbin Li. 2024. Structrag: Boosting
knowledge intensive reasoning of llms via inference-
time hybrid information structurization.Preprint,
arXiv:2410.08815.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2024. Graph retrieval-augmented generation:
A survey.Preprint, arXiv:2408.08921.
Sundar Pichai, Demis Hassabis, and Koray
Kavukcuoglu. 2025. A new era of intelligence with
gemini 3.Mountain View, CA: Google.
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao,
Defu Lian, Zhicheng Dou, and Tiejun Huang. 2025.
Memorag: Boosting long context processing with
global memory-enhanced retrieval augmentation. In
Proceedings of the ACM on Web Conference 2025,
pages 2366–2377.
Bhaskarjit Sarmah, Dhagash Mehta, Benika Hall, Ro-
han Rao, Sunil Patel, and Stefano Pasquali. 2024.
Hybridrag: Integrating knowledge graphs and vec-
tor retrieval augmented generation for efficient infor-
mation extraction. InProceedings of the 5th ACM
International Conference on AI in Finance, pages
608–616, New York, NY , USA.
Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart,
Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin,
Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 oth-
ers. 2025. Openai gpt-5 system card.arXiv preprint
arXiv:2601.03267.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Minzheng Wang, Longze Chen, Cheng Fu, Shengyi
Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan
Xu, Lei Zhang, Run Luo, and 1 others. 2024. Leave
no document behind: Benchmarking long-context
llms with extended multi-doc qa. InProceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pages 5627–5646, Miami,
Florida, USA.
Yuan Xia, Jingbo Zhou, Zhenhui Shi, Jun Chen, and
Haifeng Huang. 2025. Improving retrieval aug-
mented language model with self-reasoning. InPro-
ceedings of the AAAI conference on artificial intelli-
gence, volume 39, pages 25534–25542.
Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong
Zhang. 2024. Benchmarking retrieval-augmented
generation for medicine. InFindings of the Associa-
tion for Computational Linguistics ACL 2024, pages
6233–6251, Bangkok, Thailand.
Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita
Bhalla, Xiangsen Chen, Sajal Choudhary, RongzeGui, Ziran Jiang, Ziyu Jiang, and 1 others. 2024.
Crag-comprehensive rag benchmark. InAdvances in
Neural Information Processing Systems, volume 37,
pages 10470–10490, Red Hook, NY , USA.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Lingxi Zhang, Yue Yu, Kuan Wang, and Chao Zhang.
2024. ARL2: Aligning retrievers with black-box
large language models via self-guided adaptive rele-
vance labeling. InProceedings of the 62nd Annual
Meeting of the Association for Computational Lin-
guistics, pages 3708–3719, Bangkok, Thailand.

A Related Works
RAG Systems.Existing RAG systems can be
categorized into three types based on their knowl-
edge structures: (i) Document-based RAG meth-
ods (Asai et al., 2023; Zhang et al., 2024; Xia
et al., 2025; Yang et al., 2024) primarily rely on
flat document structures for retrieval and genera-
tion. However, they struggle with modeling se-
mantic relationships and contextual dependencies,
limiting the accuracy and completeness of gener-
ated responses. (ii) GraphRAG methods (Sarmah
et al., 2024; Peng et al., 2024) address the limita-
tions by incorporating knowledge graphs, which
enhance the organization and semantic expression
of content retrieval by structurally representing en-
tities and their relationships. Despite these advance-
ments, GraphRAG often constructs sparse graph
structures, failing to fully capture higher-order se-
mantic associations and complex contextual rela-
tions. (iii) HypergraphRAG (Feng et al., 2025a;
Hu et al., 2026) extends this approach by utilizing
hypergraphs, where hyperedges connect multiple
nodes to represent higher-order semantic connec-
tions and group semantics, significantly enhancing
the expressiveness of knowledge structures and re-
trieval quality. However, these HypergraphRAG
methods solely rely on text input, which fail to
adapt modern MLLMs’ super vision capability, mo-
tivating our exploration of hypergraph visualization
to enhance them.
Hypergraph Visualization.Existing hypergraph
visualization methods can be broadly grouped into
two categories. (i)Set-based visualization meth-
odstreat each hyperedge as a set and focus on
overlap and intersection patterns. Representative
designs include contour-based approaches such as
Bubble Sets (Collins et al., 2009), matrix-based in-
tersection analysis such as UpSet (Lex et al., 2014),
and element-attribute analysis for dense set sys-
tems such as GridSet (Chung et al., 2020). (ii)
Induced visualization methodsconvert a hyper-
graph into an incidence (bipartite) graph by intro-
ducing virtual nodes for hyperedges (Bretto, 2013),
and then apply mature graph visualization tech-
niques (Gansner and North, 2000; Hagberg et al.,
2008) to the induced structure. The first preserves
set semantics directly, while the second offers bet-
ter compatibility with scalable graph layouts and
interaction pipelines.B HyperViz Toolkit Details
This appendix supplements the HyperViz toolkit’s
extra characteristics not covered in the main texts.
B.1 Additional HyperViz Characteristics
Customizable Visual Styles.HyperViz offers
extensive customization options during render-
ing, including adjustable parameters for node
shapes, edge thickness, background colors, node
outline styles, and graph margins. For exam-
ple, node shapes can be set to options such as
ellipse ,circle ,box, orpolygon , with defaults
likeellipse androunded box commonly used to
differentiate entities and hyperedges. Edge thick-
ness can be customized within a range of 0.1 to
10 pixels, while background colors support RGB,
HEX, or standard color names. Node outline
styles include options such as solid ,dashed , and
dotted . The spacing between the graph and can-
vas edges can be adjusted within a range of 0 to
100 pixels. Additionally, in the set-based paradigm,
hyperedge region colors can be selected from pre-
defined palettes (e.g., pastel orvivid ) or defined
by the user. HyperViz also provides toggle options
such as show_legend ,show_title ,show_footer ,
andshow_edge_labels , allowing users to control
the display of legends, titles, footers, and edge la-
bels. These features collectively enable HyperViz
to adapt to a wide range of visualization tasks and
requirements.
Interactive Visualization.Beyond static PNG
outputs, HyperViz preserves stable visual IDs (en-
tity IDs as E1, E2,... , hyperedge IDs as H1,
H2,... ) and exports synchronized graph-text map-
pings, which support interactive front-end behav-
iors without changing backend retrieval logic. In
practical deployment, we bind these IDs to meta-
data tables and enable interaction primitives, in-
cluding zoom, pan, hover-tooltip, and click-to-
expand neighborhood. Clicking an entity can
trigger one-hop or multi-hop expansion through
N(v,G) , while clicking a hyperedge opens its
member set, relation description, source chunk
ID, and evidence text. This ID-consistent design
guarantees that interactive exploration and MLLM
prompting refer to the same structural objects.
Scalability Optimization.HyperViz uses a
staged compression-render pipeline for dense sub-
graphs. First, it applies budgeted subgraph con-
struction with explicit caps (e.g., max_edges and

max_entities ) and preserves high-utility struc-
tures through query-aware expansion. Second, for
panel rendering, it uses mode-specific simplifica-
tion: induced layouts keeps explicit E-H partic-
ipation edges, and set-based layouts emphasizes
group semantics by contour aggregation. Third, for
multi-view usage, render_hypergraph_collage
arranges complementary views in a bounded panel
grid to avoid single-canvas overplotting. Together
with the main-text pruning strategy (importance-
based filtering under a congestion budget), these
controls preserve readability while retaining key
high-order evidence.
More detailed characteristics and features of
HyperViz will be compiled into a comprehensive
manual, which will be made available on a dedi-
cated website following the acceptance of the paper,
along with the code.
B.2 Hyperparameter Optimization Details
We adopt a unified grid search strategy to optimize
the full set of hyperparameters in HyperViz, in-
cluding wv,we,γ,α, and β. Due to the inherent
mutual dependencies across these hyperparameters,
all parameters are jointly optimized on real-world
knowledge hypergraphs, rather than adopting sepa-
rate stage-wise tuning. All spatial weight parame-
terswv, weand structural regularization parameters
γ, α, β are searched simultaneously within their re-
spective ranges of[1,10]and[0,1].
The optimization is performed entirely on real
knowledge hypergraphs constructed from the Ul-
traDomain Legal dataset (Qian et al., 2025). To
strictly prevent data leakage, this tuning subset is
excluded from all evaluation test sets throughout
our experiments. The adopted real legal hyper-
graphs naturally cover diverse density distributions,
with node degrees ranging from 1to9and hyper-
edge sizes spanning [2,11] , fully encompassing
sparse, moderate, and dense structural patterns in
real document scenarios.
All grid search trials are supervised by the struc-
tural perception capability of Gemini-3 Flash, uti-
lizing the overlap-core identification and bridge hy-
peredge detection tasks illustrated in Fig. 2. The fi-
nal parameter combination is selected to maximize
structural perception accuracy, which we equate to
visual readability for MLLM-centric visualization
evaluation.B.3 Description of Layouts in Hyperviz
HyperViz provides six layout variants under two
complementary paradigms: set-based views for
group semantics and Induced views for explicit in-
cidence relations. In implementation, these lay-
outs are achieved by customizing layout algo-
rithms in Graphviz (Gansner and North, 2000) with
paradigm-specific post-processing (e.g., contour
generation for set regions and bipartite incidence
expansion for induced views).
Set-based layouts represent high-order relation-
ships (i.e., hyperedges) as set elements through
region containment to highlight entity grouping re-
lations. In Hyperviz, it is implemented in three
diverse layout variants:
•Set-Hier layoutis achieved by the Graphviz
dotlayout algorithm. It emphasizes top-down
hierarchy and dependency chains, making
parent-child style organization and progres-
sive evidence flow easier to inspect.
•Set-Radial layoutis achieved by the
Graphviz twopi layout algorithm. It empha-
sizes center-periphery organization around
high-degree entities, which is useful for iden-
tifying hubs, overlap cores, and radius-like
influence patterns.
•Set-Force layoutis achieved by the Graphviz
fdp/sfdp layout algorithms. It emphasizes
community structure and local cluster sepa-
ration, helping reveal dense semantic groups
and weak inter-group bridges.
Induced layouts emphasize direct element-group
relationships by representing hyperedges as spe-
cial nodes, with pairwise edges explicitly connect-
ing these nodes to their corresponding hyperedge
nodes.
•Bi-Inc:uses NetworkX bipartite_layout
to separate entity nodes and virtual hyperedge
nodes into two partitions, making membership
links directly readable. Incidence edges are
drawn with light dashed strokes, entity nodes
are styled as blue-outlined circles, and virtual
hyperedge nodes are styled as orange-outlined
squares.
•Ra-Cen:first selects the center as the highest-
degree entity in the induced graph, then com-
putes shortest-path distance shells and ren-
ders them using NetworkX (Hagberg et al.,

2008) shell_layout . This design empha-
sizes center-periphery organization for neigh-
borhood expansion and multi-hop anchor trac-
ing; if no valid center is available, it falls back
tospring_layout.
•Cli-Exp:keeps the induced incidence back-
bone and additionally expands each hyper-
edgee={v 1, v2, . . . , v k}into pairwise entity
links (i.e., k
2
clique edges). The panel is laid
out by NetworkXspring_layout(seed=42,
k= 0.95 ), where dashed incidence edges pre-
serve E–H semantics and solid clique edges
improve local connectivity perception.
To maximize the interpretability of visualized
hypergraphs, explicit semantic annotations are em-
bedded:(1) Node-Level Annotation.Intuitive text
labels are attached to all entity nodes, presenting
clear semantic information.(2) Hyperedge-Level
Annotation.Internal inclusion relations of hyper-
edges are summarized and displayed via unified
legend labels, clarifying hierarchical associations.
C Dataset Statistics
Statistics Mix CS Agriculture Neurology Pathology
Total Documents 61 10 12 1 1
Total Chunks 560 1992 1813 1790 824
Total Tokens 615,355 2,190,803 1,993,515 1,968,716 905,760
Table 6: Statistical information of the reported datasets.
Table 6 reports the dataset statistics used in this
paper.
D Baseline Details
We compare VizRAG with representative methods
spanning graph and hypergraph RAG families.
GraphRAG(Edge et al., 2024): entity-relation
graph construction with graph-guided retrieval and
community-level summaries.
LightRAG(Guo et al., 2024): lightweight graph-
enhanced retrieval with dual-level graph context
integration.
HiRAG(Huang et al., 2025): hierarchical graph
retrieval with multi-level structural traversal.
Hyper-RAG(Feng et al., 2025a): hypergraph re-
trieval modeling high-order relations through hy-
peredges.
Cog-RAG(Hu et al., 2026): cognitive-inspired
dual-hypergraph retrieval with theme-aligned top-
down evidence activation.For fairness, we keep the chunking strategy, em-
bedding model, and evaluation protocol aligned
with Cog-RAG (Hu et al., 2026) and keep consis-
tent across methods.
E Evaluation Protocol Details
We evaluated the proposed method and baseline
across six key dimensions: Comprehensiveness,
Empowerment, Relevance, Consistency, Clarity,
and Logical Coherence.
E.1 Selection-based Evaluation
For qualitative comparison, we report win rates
between methods. To ensure fairness, the order
of answers in each method pair was alternated in
the prompts, and the average result was calculated.
The evaluation dimensions are described below:
•Comprehensiveness (Comp.): Measures
how thoroughly the answer addresses all as-
pects and details of the question.
•Empowerment (Empo.): Assesses the extent
to which the answer helps the reader under-
stand the topic and make informed decisions.
•Relevance (Rele.): Evaluates how accurately
the response focuses on the key aspects of the
question without including irrelevant informa-
tion.
•Consistency (Cons.): Examines the ability of
the system to synthesize and integrate infor-
mation from multiple sources into a coherent
and logical response.
•Clarity (Clar.): Gauges whether the response
avoids unnecessary verbosity and redundancy
while providing complete and easily under-
standable information.
•Logical Coherence (Logi.): Checks whether
the response maintains consistent and rational
arguments without contradictions throughout.
E.2 Score-based Evaluation
For quantitative assessment, we employed LLMs
to score the responses based on the following indi-
cators:
•Comprehensiveness (0-100): Evaluates
whether the response comprehensively covers
all critical aspects of the question and identi-
fies any omissions.

Figure 6: Score-based metric comparison of full five datasets.
•Empowerment (0-100): Measures the cred-
ibility and persuasiveness of the response.
High scores are given to answers that provide
strong evidence or cite authoritative sources.
•Relevance (0-100): Determines the degree
to which the response stays focused on the
question and avoids unnecessary digressions.
•Consistency (0-100): Assesses whether
the response is logically organized, flows
smoothly, and the components are well-
connected and mutually supportive.
•Clarity (0-100): Examines whether the re-
sponse is expressed clearly and concisely, us-
ing precise language and unambiguous defini-
tions.
•Logical Coherence (0-100): Evaluates
whether the response maintains logical consis-
tency and is free from contradictions.
E.3 Rating Levels
Each evaluation dimension is assessed using five
discrete levels, with clear scoring criteria to ensure
consistency and transparency. For example, the
Comprehensivenessdimension is rated as follows:
•Level 1 (0-20): The response is highly in-
complete and omits significant portions or key
aspects of the question.•Level 2 (20-40): The response includes some
relevant content but misses many important
elements, making it insufficiently comprehen-
sive.
•Level 3 (40-60): The response addresses the
main aspects of the question but still has no-
table omissions.
•Level 4 (60-80): The response is mostly com-
prehensive, covering most aspects of the ques-
tion with minimal omissions.
•Level 5 (80-100): The response is exception-
ally thorough, addressing all aspects of the
question without any omissions, providing a
complete understanding to the reader.
F Additional Results
F.1 Detailed Scores on Individual Datasets
Fig.6 shows the results of scores under different
evaluation dimensions on the individual Mix, CS,
Agriculture, Neurology, and Pathology datasets.

G Prompt Templates
G.1 Core Prompt Set Aligned with VizRAG
Extracting Entities
Formulation:P ext_entity (Di)
Prompt:You are given one chunk
Di. Extract all important entities for
retrieval and reasoning. Return JSON only
with the schema:
{ "entities": [ { "entity_name": str, "en-
tity_type": one of [organization, person,
geo, event, role, concept, method, ar-
tifact, disease, treatment, mechanism],
"entity_description": str, "aliases": [str],
"source_spans": [str] } ] }
Requirements:
1) Use the same language as the input
chunk.
2) Merge duplicates and aliases.
3) Keep entities that are structurally useful
for downstream hyperedge construction.
Extracting Hyperedges
Formulation:P ext_relation (Di,VDi)
Prompt:You are given a chunk Di
and the extracted entity set VDi. Construct
hyperedges that represent semantically
coherent multi-entity relations (including
pairwise entity relationships as a special
case). Return JSON only with the schema:
{ "hyperedges": [ { "entities_set": [en-
tity_name], "relation_description": str, "re-
lation_keywords": [str], "relation_strength":
number in [0,100], "evidence_spans": [str]
} ] }
Requirements:
1) Each hyperedge must contain at least 2
(including 2) valid entities fromV Di.
2) Prefer compact but meaningful sets;
avoid trivial or redundant hyperedges.
3) If two hyperedges are near-duplicates,
keep the one with clearer evidence.Extracting Theme with Visual Cue
Formulation:P ext_theme (Di,IDi)
Prompt:You are given chunk text
Diand its hypergraph image IDi. This
visual input IDiencodes the intra-chunk
knowledge structure over entity objects and
their grouping membership. Infer a concise
theme that is consistent with both text and
structure. Return JSON only:
{ "theme_description": str, "sup-
port_entity_ids": ["E*"], "sup-
port_hyperedge_ids": ["H*"] }
Requirements:
1) Theme must reflect the central semantic
focus of the chunk.
2) Prioritize evidence supported by both text
and visual structure.
Extracting Theme-related Key Entities with
Visual Cue
Formulation:
Pext_key (Etheme
Di, Di,IDi,VDi)
Prompt:You are given: (a) theme
description, (b) chunk text Di, (c) hyper-
graph image IDiwith stable IDs, and (d)
candidate entities VDi. This visual input
IDiencodes the intra-chunk knowledge
structure over entity objects and their
grouping membership. Select theme-critical
key entities for top-layer indexing. Return
JSON only:
{ "key_entities": [ { "entity_name":
str, "entity_id": "E*" (if avail-
able), "key_entity_type": str,
"key_entity_description": str, "key_score":
number in [0,100], "selection_reason": str }
] }
Requirements:
1) Prioritize entities that are both theme-
relevant and structurally central.
2) Avoid selecting many near-duplicate
entities.

Extracting Query Keywords for Top Re-
trieval
Formulation:P keyword (q)
Prompt:Given user query q, extract
retrieval keywords for top-layer hyperedge
search. Return JSON only:
{ "theme_keywords": [str], "en-
tity_keywords": [str] }
Requirements:
1) Keep keywords short and retrieval-
oriented.
2) Cover both abstract intent (theme-level)
and concrete anchors (entity-level).
3) Remove redundant synonyms when
possible.
Locating Query-relevant Entities from Top-
view Visualization
Formulation:P locate(q,I Top,Ce_rel)
Prompt:You are given query q, top-
layer subgraph visualization ITop, and
context summaries Ce_rel of retrieved theme
hyperedges. The visual input ITopencodes
top-layer knowledge structure, including
entities, theme hyperedges, and their
structural connectivity. Identify entities in
the visual structure that are most relevant
for downstream bottom retrieval. Return
JSON only:
{ "selected_entities": [ { "entity_id": "E*",
"entity_name": str, "relevance_score":
number in [0,100], "reason": str } ] }
Requirements:
1) Use both visual structure (bridges,
overlaps, hubs) and textual context.
2) Prefer entities that can connect to
answer-critical evidence paths.Final Answer Generation with Bottom-view
Evidence
Formulation:
Panswer(q,I Bot,Vrel,Edif,Cv_rel,Ce_dif)
Prompt:You are given query q, bottom-
layer visualization IBot, retrieved entities
Vrel, expanded hyperedges Edif, and their
textual contexts Cv_rel,Ce_dif. The visual
inputIBotencodes bottom-layer evidence
structure over entities, hyperedges, and their
local reasoning paths. Generate the final
answer grounded in the provided evidence.
Return JSON only:
{ "answer": str, "support_entity_ids":
["E*"], "support_hyperedge_ids": ["H*"],
"confidence": number in [0,100] }
Requirements:
1) Prefer evidence-consistent reasoning over
unsupported speculation.
2) If evidence is insufficient, explicitly state
uncertainty.
3) Keep the answer concise but complete for
the question.

G.2 Evaluation Metrics
Selection-based Evaluation
Formulation:P eval_scoring (q,A 1,A2)
qdenotes user query,A 1andA 2denotes the response from two approaches.
Prompt:You will evaluate two answers to the same question based on six criteria: Com-
prehensiveness, Empowerment, Relevance, Consistency, Clarity, and Logical.
—Goal—
You will evaluate two answers to the same question by using the relevant documents based on six
criteria: Comprehensiveness, Empowerment, Relevance, Consistency, Clarity, and Logical.
-Comprehensiveness: How much detail does the answer provide to cover all aspects and details of
the question?
-Empowerment: How well does the answer help the reader understand and make informed
judgments about the topic?
...,
-Logical: How well does the system maintain consistent logical arguments without contradicting
itself across the response?
For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain
why. Then, select an overall winner based on these six categories.
Here is the question:q
Here are the two answers:
Answer 1:A 1;
Answer 2:A 2
Evaluate both answers using the six criteria listed above and provide detailed explanations for
each criterion.

Scoring-based Evaluation
Formulation:P eval_scoring (q,A,C)
qdenotes user query, Adenotes LLM response, Cdenotes the original text chunk that generated
the question.
Prompt:You are an expert tasked with evaluating answers to the questions by using the
relevant documents based on five criteria: Comprehensiveness, Diversity, Empowerment, Logical,
and Readability.
—Goal—
You will evaluate the answers to the question by using the relevant documents based on six criteria:
Comprehensiveness, Empowerment, Relevance, Consistency, Clarity, and Logical.
-Comprehensiveness-
Measure whether the answer comprehensively covers all key aspects of the question and whether
there are omissions.
Level|score range|description
Level 1 |0-20|The answer is extremely one-sided, leaving out key parts or important aspects of the
question.
Level 2 |20-40|The answer has some content, but it misses many important aspects of the question
and is not comprehensive enough.
Level 3 |40-60|The answer is more comprehensive, covering the main aspects of the question, but
there are still some omissions.
Level 4 |60-80|The answer is comprehensive, covering most aspects of the question, with few
omissions.
Level 5 |80-100 |The answer is extremely comprehensive, covering all aspects of the question with
no omissions, enabling the reader to gain a complete understanding.
...,
For each indicator, please give the problem a corresponding Level based on the descrip-
tion of the indicator, and then give a score according to the score range of the level.
Here is the question:q
Here are the relevant document:C
Here are the answer:A
Evaluate all the answers using the six criteria listed above. For each criterion, provide a summary
description, give a Level based on the description of the indicator, and then give a score based on
the score range of the level.