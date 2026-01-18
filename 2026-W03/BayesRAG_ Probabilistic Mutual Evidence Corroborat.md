# BayesRAG: Probabilistic Mutual Evidence Corroboration for Multimodal Retrieval-Augmented Generation

**Authors**: Xuan Li, Yining Wang, Haocai Luo, Shengping Liu, Jerry Liang, Ying Fu, Weihuang, Jun Yu, Junnan Zhu

**Published**: 2026-01-12 08:53:14

**PDF URL**: [https://arxiv.org/pdf/2601.07329v1](https://arxiv.org/pdf/2601.07329v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a pivotal paradigm for Large Language Models (LLMs), yet current approaches struggle with visually rich documents by treating text and images as isolated retrieval targets. Existing methods relying solely on cosine similarity often fail to capture the semantic reinforcement provided by cross-modal alignment and layout-induced coherence. To address these limitations, we propose BayesRAG, a novel multimodal retrieval framework grounded in Bayesian inference and Dempster-Shafer evidence theory. Unlike traditional approaches that rank candidates strictly by similarity, BayesRAG models the intrinsic consistency of retrieved candidates across modalities as probabilistic evidence to refine retrieval confidence. Specifically, our method computes the posterior association probability for combinations of multimodal retrieval results, prioritizing text-image pairs that mutually corroborate each other in terms of both semantics and layout. Extensive experiments demonstrate that BayesRAG significantly outperforms state-of-the-art (SOTA) methods on challenging multimodal benchmarks. This study establishes a new paradigm for multimodal retrieval fusion that effectively resolves the isolation of heterogeneous modalities through an evidence fusion mechanism and enhances the robustness of retrieval outcomes. Our code is available at https://github.com/TioeAre/BayesRAG.

## Full Text


<!-- PDF content starts -->

BayesRAG: Probabilistic Mutual Evidence Corroboration for
Multimodal Retrieval-Augmented Generation
Xuan Li1,*Yining Wang2,*Haocai Luo1Shengping Liu2
Jerry Liang2Ying Fu2Weihuang2Jun Yu2,†Junnan Zhu3,†
1University of Science and Technology of China,
2Unisound AI Technology Co.Ltd,
3MAIS, Institute of Automation, Chinese Academy of Sciences
harryjun@ustc.edu.cn, junnan.zhu@nlpr.ia.ac.cn
Abstract
Retrieval-Augmented Generation (RAG) has
become a pivotal paradigm for Large Language
Models (LLMs), yet current approaches strug-
gle with visually rich documents by treating
text and images as isolated retrieval targets. Ex-
isting methods relying solely on cosine similar-
ity often fail to capture the semantic reinforce-
ment provided by cross-modal alignment and
layout-induced coherence. To address these
limitations, we proposeBayesRAG, a novel
multimodal retrieval framework grounded in
Bayesian inference and Dempster-Shafer evi-
dence theory. Unlike traditional approaches
that rank candidates strictly by similarity,
BayesRAG models the intrinsic consistency
of retrieved candidates across modalities as
probabilistic evidence to refine retrieval con-
fidence. Specifically, our method computes the
posterior association probability for combina-
tions of multimodal retrieval results, prioritiz-
ing text-image pairs that mutually corroborate
each other in terms of both semantics and lay-
out. Extensive experiments demonstrate that
BayesRAG significantly outperforms state-of-
the-art (SOTA) methods on challenging mul-
timodal benchmarks. This study establishes
a new paradigm for multimodal retrieval fu-
sion that effectively resolves the isolation of
heterogeneous modalities through an evidence
fusion mechanism and enhances the robustness
of retrieval outcomes. Our code is available at
https://github.com/TioeAre/BayesRAG.
1 Introduction
RAG has fundamentally enhanced the generaliza-
tion capabilities of LLMs by grounding them in
external domain knowledge. However, real-world
knowledge, ranging from academic papers to indus-
trial reports, is predominantly presented in visually
* Equal contribution.
† Corresponding author.rich documents. In these documents, visual ele-
ments such as charts and diagrams are not merely
decorative add-ons, but are intrinsically intertwined
with textual explanations to convey complex se-
mantics. Consequently, relying solely on unimodal
text retrieval is insufficient. Verifying the semantic
consistency between visual and textual evidence is
paramount for building robust RAG systems.
In practice, RAG systems navigate vast repos-
itories containing thousands of pages. While the
advent of vision-language models has enabled mul-
timodal retrieval, current SOTA approaches primar-
ily focus on improving recall by expanding the Top-
kwindow and performing a naive union of text and
image retrieval results. We argue that this “bag-of-
evidence” approach is fundamentally flawed. By
treating modalities as independent channels and
simply concatenating their representations, exist-
ing methods fail to capture the criticalsemantic
interactionbetween modalities. Specifically, they
cannot distinguish whether a retrieved image and a
text chunk are logically corroborated or merely sta-
tistically similar (e.g., sharing common keywords
but divergent contexts). This inability leads to a
collection of high-similarity yet semantically dis-
connected fragments, which introduces noise and
confounds the downstream answer generator.
To address modal disparity, some approaches
employ Image-to-Text transduction or Knowledge
Graphs (KGs) to unify information into a textual
modality. However, these methods suffer from se-
vere information and structural loss. Critical in-
sights in knowledge-intensive domains are often
encoded in non-textual formats that are “ineffable”,
which is difficult to translate losslessly into de-
scriptions (Lumer et al., 2025). Furthermore, these
linearization processes remove document layout
information (Xiang et al., 2025). We consider that
document layout is not merely a stylistic choice
but serves as an implicit logical navigation map
designed by the author. Standard chunking strate-
1arXiv:2601.07329v1  [cs.CL]  12 Jan 2026

gies disrupt this topological structure, severing the
logical ties between spatially adjacent evidence.
In summary, developing a trustworthy multi-
modal RAG system necessitates addressing three
pivotal challenges: (1)Preserving Multimodal
Fidelity:The system should move beyond lossy
translation to perform native representation learn-
ing (Mei et al., 2025). (2)Cross-Modal Seman-
tic Corroboration:Achieving high confidence
requires a mechanism for evidence verification.
The system needs to facilitate interaction between
retrieved modalities, identifying instances where
visual and textual candidates semantically align.
Such alignment should be treated as reinforcing
evidence, thereby assigning higher posterior con-
fidence to mutually corroborating retrieval results.
(3)Layout-Induced Structural Understanding:
Beyond content semantics, the system should be
able to recover the contextual relationships of re-
trieved elements within the original document,
which allows for decoding the implicit semantic
topology within the document layout (Tong et al.,
2025), thereby restoring connections that are fre-
quently severed by traditional chunking strategies.
To address these challenges, we propose
BayesRAG. Diverging from traditional approaches
that rely solely on unimodal similarity matching,
we reconceptualize multimodal retrieval as a pro-
cess ofprobabilistic evidence fusion. Our core
intuition addresses the prevalent issue of semantic
inconsistency in heterogeneous retrieval; this is ex-
emplified by scenarios where text retrieval suggests
the fruit “apple” while visual retrieval identifies the
corporate logo of “Apple Inc.” Under such circum-
stances, independent similarity scores remain high
for both yet fail to capture the fatal discordance
between them. BayesRAG leverages Bayesian in-
ference to elegantly resolve this dilemma by utiliz-
ing similarity scores from embedding models as
thelikelihoodand modeling cross-modal semantic
consistency as thePrior Belief. This formulation
mathematically embodies the intuition that“only
text-image pairs that are semantically consistent
deserve higher trust, ”and it thereby dynamically
penalizes conflicting evidence during the fusion.
By computing the posterior probability,
BayesRAG automatically down-weights retrieval
candidates that exhibit high unimodal scores yet
contradict each other. This mechanism effectively
isolates high-fidelity evidence combinations
that are mutually corroborating, simultaneously
preserving native multimodal information andensuring the logical rigor of cross-modal reasoning.
Our main contributions are summarized as follows:
•We establish a unified probabilistic framework
that reconceptualizes multimodal retrieval as an
evidence fusion process; this paradigm effec-
tively preserves the native layout structure often
lost in traditional strategies.
•We introduce a Bayesian inference mechanism
grounded in Dempster-Shafer theory to model se-
mantic consistency; this approach resolves cross-
modal conflicts by prioritizing mutually corrobo-
rating evidence.
•Extensive experiments demonstrate that
BayesRAG achieves SOTA performance on
challenging benchmarks; our analysis confirms
that modeling evidence consistency effectively
eliminates modal conflicts and semantic
inconsistencies prevalent in existing baselines.
2 Method
In this section, we presentBayesRAG, a proba-
bilistic framework for multimodal retrieval.
2.1 Retrieval as Evidence Fusion
Traditional multimodal RAG systems typically
treat retrieval as independent feature-matching
tasks, where text and images are ranked solely by
their semantic similarity to the query. This often
leads to modal conflict, where a high-scoring text
and a high-scoring image may contradict each other
or originate from unrelated contexts.
BayesRAG redefines retrieval as aProbabilis-
tic Evidence Fusionprocess. We aim to identify
an evidence tuple E= (e txt, evision, escreenshot ),
consisting of evidence from text, image, and struc-
tural information, that is not only relevant to the
query but alsointernally consistent.Our intuition
is modeled via Bayesian inference:Prior(Inter-
nal Consistency): Before observing the query, how
likely is it that this text and image belong together?
This is determined by the inherent consistency of
the tuple E, exemplified by factors such as seman-
tic alignment or narrative proximity in the source
document.Likelihood(Semantic Match): Given
this evidence tuple, how likely is it to generate the
user’s query? We model this using vector similar-
ity enhanced by belief functions.Posterior(Final
Confidence): The updated probability that the tuple
is the optimal evidence given the query.
2

Relevant = ( 𝛼⋅𝑆)
Irrelevant = ( 𝛽⋅(1−𝑆))
Uncertainty =
1−𝑅𝑒𝑙𝑒𝑣𝑎𝑛𝑡 −𝐼𝑟𝑟𝑒𝑙𝑒𝑣𝑎𝑛𝑡
…Online Search (According to the 2015 National Survey of Latinos details, which …?)
Retrieval Results Score: 𝑆𝑡𝑒𝑥𝑡
Score: 𝑆𝑖𝑚𝑎𝑔𝑒
Score : 𝑆𝑠𝑐𝑟𝑒𝑒𝑛𝑠ℎ𝑜𝑡
Semantic consistency Prior 𝑃𝐸Likelihood 𝑃(𝑄|𝐸)Posterior
𝑷𝑬𝑸∝𝑷𝑸𝑬⋅𝑷𝑬
Rank by 𝑷𝑬𝑸
Top-k results
Final Answer: Foreign born (excl. PR)DocumentsParseText VDB
Image VDB
Screenshot VDB
Knowledge Graph
Text chunks Images ScreenshotsOffline Index
Vector Retrieval 
𝑷(𝑸|𝑬)=
𝐷𝑒𝑚𝑝𝑠𝑡𝑒𝑟 −Sℎ𝑎𝑓𝑒𝑟 (𝑆𝑡,𝑆𝑖,𝑆𝑠)
𝑤1
𝑤2
𝑤3
Weight of the edge 
in KG𝑝𝑟𝑜𝑏𝑖=1−𝑒−𝑘∗𝑤𝑖
𝑷𝑬=෍
𝑖𝑛
𝑝𝑟𝑜𝑏𝑖/𝑛QueryRetrieval ConsensusThe remaining… a high 
incidence of Latinos , … 
….
Retrieve image:
Retrieve screenshot:
Retrieve text:
The remaining… 
a high incidence 
of Latinos, ….𝐻𝑖𝑔ℎ  𝑟𝑒𝑡𝑟𝑖𝑒𝑣𝑎𝑙  𝑠𝑐𝑜𝑟𝑒𝑠  
⋅𝑆𝑒𝑚𝑎𝑛𝑡𝑖𝑐  𝑐𝑜𝑛𝑣𝑒𝑟𝑔𝑒𝑛𝑐𝑒  
→𝐸𝑛ℎ𝑎𝑛𝑐𝑒𝑑  𝐶𝑜𝑛𝑓𝑖𝑑𝑒𝑛𝑐𝑒Figure 1: The architecture of BayesRAG. We reconceptualize multimodal retrieval as a Bayesian inference process.
Our method computes the semantic likelihood P(Q|E) and derives the consistency prior P(E) via knowledge
graph topology. The final evidence tuples are re-ranked based on posterior probability P(E|Q) , ensuring semantic
relevance, mutual corroboration, and logical self-consistency.
2.2 Problem Formulation
LetDbe a multimodal document corpus. We de-
fine the sample space Θas the set of all possible
multimodal evidence tuples. A single evidence can-
didate is denoted as E= (t, v, s)∈Θ , where t
represents a textual segment, vrepresents a visual
component, and sdenotes the structural informa-
tion from screenshots.
LetQdenote the observed user query. Our goal
is to compute the posterior probability P(E|Q) ,
which represents the confidence that the tuple Eis
the optimal evidence given the query Q. According
to Bayes’ Theorem:
P(E|Q) =P(Q|E)·P(E)
P(Q)(1)
Since P(Q) (the marginal likelihood of the
query) is constant for all candidates during the rank-
ing phase, we simplify the objective function to:
P(E|Q)∝P(Q|E)|{z}
Embedding Likelihood·P(E)|{z}
Consistency Prior(2)
2.3 Bayesian Evidence Modeling
Modeling Likelihood with Belief Functions.The
likelihood P(Q|E) estimates the probability that
the query Qis derived from the evidence E. To
robustly handle the uncertainty in embedding co-
sine similarity scores (e.g., distinguishing between
“irrelevant” and “uncertain”), we incorporate the
Dempster-Shafer Theory of Evidence.
For each modality i∈ {t, v, s} , letSi∈[0,1]
be the normalized cosine similarity score with the
query. We construct a mass function miover theframe of discernment Θ ={Y,N} (Relevant, Irrel-
evant), as shown in Table 1.
Hypothesis Mass Function Formulation
Evidence (e)m(Y) =α·S i
Not Evidence (¯e)m(N) =β·(1−S i)
Uncertainty (Ω)m(Ω) = 1−m(Y)−m(N)
Table 1: Mass function definitions based on cosine sim-
ilarity scores. α, β∈[0,1] are hyperparameters repre-
senting trust in the embedding model.
We aggregate the mass functions from text, vi-
sion, and screenshot using Dempster’s Rule of
Combination ( mjoint =m i⊕m j). The likeli-
hood is then projected from the combined belief
function:
P(Q|E)≈BetP joint(Y)(3)
This formulation ensures that the likelihood is high
only when multiple modalities provide conflict-free
evidence supporting the query (see Appendix A.1).
Modeling the Consistency Prior P(E) .The
priorP(E) represents the probability that the tuple
E= (t, v, s) constitutes a meaningful, coherent
unit of informationindependentof the user’s query.
This captures the “topological” or “physical” re-
ality of the document structure. If the text tand
image vare semantically unrelated or spatially dis-
tant in the source document, P(E) should be low.
We model this as:
P(E) =f consistency (t, v, s)(4)
where fconsistency is modeled by the semantic con-
sistency or spatial distance between the text, image,
and screenshot.
3

To quantify the intrinsic coherence P(E) of an
evidence tuple E= (t, v, s) , we propose two mod-
eling strategies:(1)Graph-Topology Prior, which
constructs a multimodal knowledge graph from
document chunks to modelsemantic consistency.
This approach calculates connection strengths be-
tween retrieved nodes to prioritize tuples that form
strongly connected paths. Specifically, we check
the topological connectivity of the retrieved triplet
(t, v, s) within the graph and model their mutual
semantic consistency based on the weights of the
edges connecting these three nodes. (2)Layout
Prior, which leverages bounding box coordinates
to modelgeometric proximity, based on the assump-
tion that spatially adjacent elements on the same or
nearby pages have a higher probability of associa-
tion. Both strategies map physical or topological
signals into a probability space P(E)∈[0,1] (de-
tails in Appendix A.2). As shown in the “Spatial
vs. Semantic Priors” analysis in Section 3.3, incor-
porating either prior significantly improves QA per-
formance compared to standard embedding-only
retrieval. Notably, the semantic constraints offered
by the Graph-Topology Prior outperform the purely
geometric constraints of the Layout Prior.
3 Experiments
3.1 Experimental Settings
Evaluation Benchmarks. We evaluate BayesRAG
on two challenging multimodal Document QA
benchmarks: DocBench (Zou et al., 2025) and
MMLongBench-Doc (Ma et al., 2024). DocBench
comprises 229 documents across five domains (e.g.,
Finance, Law) with 1,102 QA pairs, featuring sub-
stantial length (avg. 66 pages, ∼46k tokens) to
test long-context capabilities. MMLongBench-Doc
complements this with 135 documents spanning 7
diverse types and 1,082 questions. Together, these
datasets cover over 2,000 questions in varied real-
world scenarios, providing a robust testbed for mul-
timodal long-document understanding.
Baselines.To evaluate the effectiveness of
BayesRAG, we compare it against a diverse
set of SOTA multimodal RAG systems, cover-
ing modular, visual-centric, agentic, and graph-
based paradigms:RAGFlow1: A widely adopted
open-source RAG engine that integrates retrieval
pipelines with agentic capabilities. It serves as
a representative baseline for standard, modular-
1https://github.com/infiniflow/ragflowMethodDomainsOverall
Aca. Fin. Gov. Law News
GPT-4o-mini 42.5 36.8 53.1 51.4 51.1 44.1
VisRAG 21.9 28.2 20.3 29.6 11.6 25.3
ViDoRAG 29.069.135.8 39.7 37.2 43.5
RAGFlow 45.2 34.0 41.8 48.1 23.8 39.0
RAGAnything 51.4 42.3 43.2 50.2 57.5 48.7
BayesRAG (ours)52.437.556.0 52.3 66.8 51.2
Table 2: Accuracy (%) on DocBench. Best performance
is highlighted inboldand second-best is underlined .
Domain categories include Academia (Aca.), Finance
(Fin.), Government (Gov.), Legal Documents (Law),
and News Articles (News).
ized RAG implementations used in industry.Vis-
RAG(Yu et al., 2025a): A visual-centric frame-
work that bypasses traditional OCR parsing. In-
stead of extracting text, VisRAG treats document
pages directly as images, utilizing VLMs for both
visual embedding and generation.ViDoRAG
(Wang et al., 2025b): An agent-based approach
that employs a Gaussian Mixture Model (GMM)
for multimodal fusion. It introduces an iterative
agentic workflow (consisting of Explore, Summa-
rize, and Reflect stages) to verify the reliability of
retrieved evidence dynamically.RAGAnything
(Guo et al., 2025): A knowledge-graph-based sys-
tem that reconceptualizes multimodal content as
interconnected entities. It employs a dual-graph
construction mechanism and hybrid retrieval, com-
bining structural navigation within the KG with
semantic vector matching to locate cross-modal
evidence. We also evaluate the performance of di-
rectly feeding PDFs intoGPT-4o-mini. For docu-
ments with fewer than 50 pages, we use page-level
screenshots as input. For longer documents, we
randomly sample 50 screenshots as input.
Evaluation Metrics.ForDocBench, we strictly
follow the official evaluation protocol, utilizing
the standard LLM-based scoring mechanism to en-
sure fair comparison with existing baselines. For
MMLongBench-Doc, in addition to reporting the
official rule-based accuracy (Acc), we introduce a
supplementary LLM-based evaluation. This is de-
signed to mitigate false negatives inherent in rigid
string matching, specifically addressing cases of
semantic equivalence (e.g., abbreviations or para-
phrasing) that rule-based metrics fail to capture.
Experimental Setup.To ensure a rigorous and
standardized evaluation, we employ GPT-4o-mini
as the final response generator across all experi-
ments. We utilize MinerU (Niu et al., 2025) for
4

MethodReports Tutorials Academic Guidebooks Brochures Industry Financial Overall
Acc Score Acc Score Acc Score Acc Score Acc Score Acc Score Acc Score Acc Score
GPT-4o-mini 29.8 36.7 29.0 38.2 16.8 24.0 24.0 32.4 29.5 34.641.3 50.638.3 41.0 28.2 35.2
VisRAG 27.9 30.0 19.4 20.8 21.5 24.0 19.8 23.7 19.8 23.7 19.7 24.6 11.1 11.1 21.3 23.8
ViDoRAG 29.8 36.1 33.3 39.5 16.5 21.0 27.6 32.0 19.6 24.7 21.8 25.9 35.3 36.7 26.5 31.4
RAGFlow 36.8 38.9 31.5 35.2 27.8 29.9 30.6 35.2 33.4 36.6 28.7 30.8 31.2 34.1 32.0 34.9
RAGAnything 40.6 45.3 39.5 46.0 32.0 34.6 37.3 43.5 36.439.630.5 33.3 39.2 42.7 37.1 41.5
BayesRAG (ours)42.6 46.7 41.8 53.6 32.7 34.8 41.6 46.4 37.0 39.4 31.1 38.641.2 48.6 38.8 44.1
Table 3: Accuracy (%) and GPT-Score (%) on MMLongBench-Doc across different domains and overall perfor-
mance. Best performance is highlighted inboldand second-best is underlined . Domain categories include Research
Reports/Introductions (Reports), Tutorials/Workshops (Tutorials), Academic Papers (Academic), Guidebooks
(Guidebooks), Brochures (Brochures), Administration/Industry Files (Industry), and Financial Reports (Financial).
Acc is calculated by the verification method in the MMLongBench-Doc paper. Score is the score verified by LLM.
Method Recall@1 Recall@3 Recall@5 Recall@10 Recall@20
Vector Retrieval 24.4 41.7 46.4 51.6 56.6
BayesRAG 27.5 44.3 54.7 61.7 76.6
Table 4: Retrieval performance comparison on the
MMLongBench-Doc dataset.Vector Retrievalde-
notes the baseline using raw similarity scores from em-
bedding models and text reranker without probabilis-
tic re-ranking.BayesRAGincorporates our proposed
Bayesian evidence fusion.
document parsing. We employ Qwen3-Embedding-
4B (Zhang et al., 2025) to encode textual chunks.
For image elements, we utilize PE-Core-G14-
448 (Bolya et al., 2025). Additionally, we use
colnomic-embed-multimodal-3b2to generate page-
level embeddings for global layout understanding.
All retrieved text candidates are refined using bge-
reranker-v2-m3 (Chen et al., 2024). For the con-
struction of the Knowledge Graph, we leverage the
Qwen3-VL-32B (Bai et al., 2025) to extract entities
and relations from both text and images. We build
all documents in the benchmark into one vector
database/knowledge graph for retrieval and evalu-
ation. Detailed hyperparameters and environment
configurations are provided in Appendix B to facil-
itate reproducibility. Crucially, due to the absence
of mature visual reranking models and the compu-
tational cost of VLM-as-rerankers, the baseline is
constrained to a shallow pool of visual candidates
(Top-20 or the adaptive top-k in its method). In
contrast, BayesRAG’s probabilistic fusion is com-
putationally lightweight, allowing us to expand the
initial visual pool to Top-512 without incurring
significant latency (see Appendix F).
3.2 Main Results
Performance comparison.Table 2 further vali-
dates the versatility of our approach. BayesRAG
2https://huggingface.co/nomic-ai/
colnomic-embed-multimodal-3bachieves an SOTA overall score of51.2%, surpass-
ing the strongest baseline RAGAnything by 2.5%.
It demonstrates dominance in text-intensive and
high-noise domains, such asGovernment(+12.8%)
andNews(+9.3%). This confirms that our Bayesian
evidence fusion not only handles multimodal align-
ment but also enhances textual retrieval by penal-
izing irrelevant evidence through consistency pri-
ors. As shown in Table 3, BayesRAG secures the
leading position in composite metrics, achieving
38.8%accuracy and an LLM score of44.1%. Our
framework exhibits exceptional robustness in het-
erogeneous domains, notablyGuidebooks(+4.3%
Acc over the runner-up) andTutorials/Workshops
(+7.6% Score). These documents typically fea-
ture intricate interplays between text, charts, and
layout. By explicitly modeling evidence consis-
tency, BayesRAG effectively filters out modal mis-
matches, whereas visual-centric methods like Vis-
RAG suffer from noise introduced by full-page
embedding (21.3% overall Acc).
Retrieval performance.We compare BayesRAG
against the Vector Retrieval baseline. The baseline
merely aggregates raw cosine scores from text, im-
age, and screenshot embedding models with a text
reranker, lacking mechanisms for semantic consis-
tency checks or structural priors. Utilizing the page-
level evidence annotations in MMLongBench-Doc,
we approximate retrieval recall to evaluate per-
formance. Table 4 demonstrates that BayesRAG
achieves substantial gains on all Recall@K met-
rics: Precision Enhancement (R@1): Recall@1
improves significantly by 3.1%. This result con-
firms that our Bayesian formulation effectively mit-
igates high-similarity hallucinations. By penalizing
conflicting evidence, where P(Q|E) drops due to
modal inconsistencies, the framework successfully
prioritizes truly corroborating evidence. Recall
Robustness (R@20): Crucially, the advantage be-
5

MethodBayesian
InferenceLikelihood Prior Reports Tutorials Academic Guidebooks Brochures Industry Financial Overall
Embedding-RAG%- - 39.2 27.9 26.4 31.3 32.6 25.3 19.8 32.7
BayesRAG (Linear Fusion)!Linear Weighted Graph 45.0 41.3 33.3 39.3 35.0 26.2 35.7 40.6
BayesRAG (Layout Prior)!Dempster-Shafer Layout 38.4 33.5 30.5 40.6 36.5 21.2 27.2 36.8
BayesRAG (Full)!Dempster-Shafer Graph 46.7 53.6 34.8 46.4 39.4 38.6 48.6 44.1
Table 5: Ablation study on Bayesian components.Embedding-RAG: Baseline with independent retrieval.BayesRAG
(Layout Prior): Incorporates spatial consistency constraints ( P(E) ) based on bounding box proximity (assigning
1.0 for adjacent elements, 0.1 otherwise).BayesRAG (Linear Fusion): Estimates likelihood ( P(Q|E) ) via linear
weighted averaging of normalized scores.BayesRAG (Full): Synergizes Graph-Topology priors with Dempster-
Shafer theory to robustly handle evidence conflict and uncertainty.
comes more pronounced as Kincreases, with Re-
call@20 surging by 20.0% (56.6% →76.6%). This
reveals that standard vector retrieval often buries
ground-truth evidence in the long tail due to low
semantic scores caused by modality misalignment
or OCR noise. BayesRAG leverages consistency
priors P(E) , to “rescue” these relevant but low-
scoring candidates.
Attribution Analysis.While BayesRAG demon-
strates strong retrieval capabilities, achieving a
Recall@20 of 76.6%, the final generation perfor-
mance (Score) stands at 44.1%. This discrepancy
reveals a substantialRetrieval-Generation Gap.
However, rather than a limitation, we interpret this
as BayesRAG successfully unlocking the retrieval
bottleneck that constrains previous systems. The
high recall indicates that our framework provides a
rich, high-fidelity context that current compact gen-
erators (like GPT-4o-mini) may struggle to fully
digest, but which lays a solid foundation for more
capable Multimodal LLMs. To investigate the cur-
rent generation bottleneck, we conduct a manual
analysis on a sample of 100 failure cases. We find
that 51% of the errors stem from query ambigu-
ity inherent in MMLongBench-Doc. Specifically,
questions often appear in generic forms such as
“What year is the report for?” In the RAG setting
that involves retrieving information across multiple
distinct documents, such queries lack the necessary
context to disambiguate the target document. This
observation elucidates the phenomenon reported in
Table 2 and Table 3, where directly feeding 50 page-
level screenshots of the specific target document
into GPT-4o-mini yields superior performance.
3.3 Ablation Experiments
To assess the individual contributions of our struc-
tural priors and evidential fusion mechanism, we
compare BayesRAG against three variants on
MMLongBench-Doc: Embedding-RAG, which re-
lies solely on embedding-based retrieval and a text
reranker; a variant utilizing only Layout Priors; anda variant employing Linear Fusion.
Bridging the Visual Reranking Gap.To isolate
the contribution of our Bayesian inference mecha-
nism, we compare BayesRAG against Embedding-
RAG. As shown in Table 5, the baseline employs
a hybrid retrieval strategy: text candidates are re-
fined using bge-reranker-v2-m3 (retrieving Top-
1024 and selecting Top-20), while visual candidates
(images and page screenshots) rely solely on raw
embedding scores (Top-20) due to the absence of
mature visual reranking models or the computa-
tional cost of VLM-as-rerankers. The results reveal
a significant performance disparity. BayesRAG
(Full) outperforms the baseline by a substantial
margin of11.4%in overall score. The performance
leap can be attributed to how BayesRAG handles
visual noise. In the baseline, visual retrieval is
limited to the Top-20 raw candidates. However,
embedding models may assign high scores to vi-
sually similar but semantically irrelevant images.
BayesRAG addresses this by expanding the initial
visual pool to Top-512 and utilizing the posterior
P(E|Q) as asoft reranker. By enforcing consis-
tency constraints, BayesRAG effectively filters out
high-scoring visual hallucinations that dominate
the baseline’s Top-20 list, ensuring that the final
retrieved context is semantically coherent.
Comparison of fusion strategy.We further in-
vestigate the fusion mechanism by comparing our
Dempster-Shafer-based approach against a stan-
dardLinear Fusion, which calculates P(Q|E)
as a weighted average of normalized embedding
scores. While Linear Fusion improves upon the
baseline (40.6% score), it still underperforms the
BayesRAG (Full) (44.1%) by a distinct margin.
The deficiency of linear weighting lies in its inabil-
ity to handlemodal conflicts. In scenarios where
text and image retrievers contradict each other, a
linear sum blindly averages the scores, potentially
retaining the noise. In contrast, Dempster-Shafer
formulation explicitly calculates aConflict Coef-
6

Question: How many samples in 
MMMU belong to sociology subject?VisRAG: 
Not answerable.ViDoRAG: 
Not answerable.
RAGFlow: 
Not answerable.RAG-Anything: 
Not answerable.
BayesRAG: 
The most direct and explicit information 
is in Figure 3 on page 3 (image 2), which 
states: "Sociology (287, 2.48%)". This 
indicates that there are 287 samples in 
the Sociology subject.
Therefore, the number of samples in 
MMMU that belong to the Sociology 
subject is 287.Ground Truth: 287provenance page
Figure 2: A qualitative comparison illustrating the efficacy of BayesRAG in broadening the scope of visual retrieval
and filtering inconsistent evidence. Our statistical analysis reveals that approximately 34% of failure cases in
baseline methods are attributed to this retrieval deficiency.
ficient K. When divergence between modalities
is detected, the belief mass is redistributed to the
uncertainty state or penalized, ensuring that only
mutually corroborating evidence survives. This is
particularly evident in theTutorialsdomain (41.3%
→53.6%), where precise alignment between com-
plex charts and descriptions is critical.
Spatial vs. Semantic Priors.TheLayout Priorin-
troduces a geometric constraint P(E) derived from
document layout analysis. It functions as a spatial
filter, assigning a high prior probability ( 1.0) to
evidence tuples where the visual and textual com-
ponents are spatially adjacent (within a specific
bounding box distance), and a penalty ( 0.1) oth-
erwise. As shown in Table 5, incorporating this
physical constraint alone boosts the overall score
from 32.7% to 36.8%. This result validates our hy-
pothesis thatspatial consistencyserves as a reliable
proxy forsemantic consistency. By strictly enforc-
ing layout coherence, the model effectively reduces
uncertainty, filtering out visually similar but struc-
turally unrelated hallucinations. While bounding
boxes provide a strong baseline, this motivates our
further exploration into Knowledge Graphs (in the
full model) to capture deeper semantic links be-
yond mere physical proximity.
3.4 Case Study
To explicitly demonstrate how BayesRAG handles
complex retrieval challenges, we visualize two rep-
resentative cases in Figure 2 and Figure 3, compar-
ing our method against leading baselines.
Case 1:Figure 2 depicts target text (“Sociology”)
buried in a complex layout. Baselines fail to cap-
ture this low-salience signal, returning “Not an-swerable”. BayesRAG addresses this by expanding
the retrieval candidate pool (Top- k) and leveraging
the semantic consistency between the query and the
image text to filter the resulting noise effectively.
This capability is critical, as our analysis reveals
that about 34% of baseline errors stem from similar
failures to retrieve ground-truth evidence, directly
resulting in incorrect “Not answerable” predictions.
Case 2:Figure 3 presents a challenge where base-
lines are misled by the visual similarity of a “Mo-
bile phone” distractor chart located on a different
slide. BayesRAG successfully retrieves the correct
evidence through its probabilistic scoring mecha-
nism. Specifically, the explicit match between the
query “Smartphone” and the textual component of
the target evidence tuple yields a higherlikelihood
score. Simultaneously, the semantic consistency be-
tween the tuple’s text and image results in a robust
priorscore. These factors synergize during the
Bayesian update to rank the correct “Smartphone”
tuple significantly higher than the visually similar
distractor, enabling the accurate calculation.
In summary, BayesRAG not only expands re-
call for obscure details in images but also robustly
filters confusing distractors through rigorous prob-
abilistic verification.
4 Related Works
Multimodal Vector Retrieval Paradigms.RAG
significantly enhances LLMs by integrating exter-
nal knowledge bases (Lewis et al., 2020), with
embedding models (Chen et al., 2024) playing
a pivotal role in this process. Recently, vision-
language models such as CLIP and ColPali (Faysse
et al., 2025) have extended RAG capabilities to
7

Question: what percent did "Smartphone Penetration" grow 
between 2013 and 2015?
Ground Truth: 26%
VisRAG: 
Not answerable.
RAGFlow: 
108.33%
BayesRAG: ViDoRAG: 
110%
RAG-Anything: 
108.33%
provenance pageImage 3 contains a bar chart titled "Smartphone 
Penetration" with data for the years 2012, 2013, 
2014, and 2015. For 2013, the bar chart shows a 
value of 24%. For 2015, the bar chart shows a 
value of 50%.  The growth in percentage points is 
the difference between the 2015 and 2013 values.
Growth = 50% - 24% = 26 percentage points
Figure 3: A sample is for semantic consistency between text and charts to find the correct evidence
retrieve images or document pages directly. Build-
ing on these advancements, RegionRAG (Li et al.,
2025) introduces a region-level contrastive learn-
ing objective, thereby refining retrieval granularity
from coarse-grained whole pages to specific se-
mantic regions. Broadly, current approaches fall
into two categories: (1) Visual-Centric Unifica-
tion: Methods like VisRAG (Yu et al., 2025b) and
M3DocRAG (Cho et al., 2024) unify modalities
by treating entire document pages as images for re-
trieval, effectively preserving layout but potentially
sacrificing fine-grained textual details. (2) Dual-
Stream Retrieval: Systems like VisDoM (Suri et al.,
2025) store and retrieve text and images separately.
They typically employ a late fusion strategy, where
separate visual and textual models generate inter-
mediate answers that are subsequently merged by
a fusion model. However, these separate streams
often lack interaction during the retrieval phase,
leading to potential semantic misalignment.
Knowledge-Structured Multimodal RAG.To
capture complex relationships within documents,
several approaches integrate Knowledge Graphs.
Methods such as HM-RAG, MMGraphRAG (Wan
and Yu, 2025), and mKG-RAG (Yuan et al., 2025)
utilise Multimodal LLMs (MLLMs) to extract en-
tities and relationships from both text and images.
RAG-Anything (Guo et al., 2025) employs a hy-
brid search mechanism that combines explicit KG
links with vector semantic similarity. It utilizes a
multi-signal scoring mechanism that weighs both
the topological importance of the graph structure
and the semantic similarity scores. While effec-
tive, these graph-based methods rely heavily on the
quality of upstream information extraction.
Evidence Verification and Reranking.Ensuring
the reliability of retrieved content is critical. For
text modality, cross-encoder rerankers are standardpractice. MIRAGE (Wei et al., 2025) ensures trust-
worthy evidence by cross-validation of inference
chains in enhanced retrieval-augmented graphs.
For images, frameworks like ViDoRAG (Wang
et al., 2025b), MDocAgent (Han et al., 2025),
HKRAG (Tong et al., 2025), mRAG (Hu et al.,
2025), and MADAM-RAG (Wang et al., 2025a)
adopt an agent-based approach, utilizing agents
such as the Inspector or Critical Agent to perform
iterative verification loops to filter evidence. How-
ever, these agent-based methods are often com-
putationally intensive due to repeated LLM calls,
resulting in substantial token consumption and high
latency. Furthermore, they lack a unified probabilis-
tic framework to explicitly model the mutual cor-
roboration between heterogeneous modalities—a
gap we aim to fill with BayesRAG.
5 Conclusion
In this work, we introduce BayesRAG to advance
multimodal RAG for visually rich documents by
transforming retrieval into a dynamic evidence fu-
sion process. By synergizing Bayesian inference
with Dempster-Shafer theory, our framework offers
a theoretically rigorous solution to the semantic in-
consistencies prevalent in heterogeneous data. We
demonstrate that explicitly modeling modal con-
flicts allows the system to filter out high-scoring
but semantically misaligned noise. This capability
effectively closes the gap between visual retrieval
and semantic verification. Our extensive evalua-
tions confirm that this probabilistic approach not
only achieves state-of-the-art performance but also
establishes a new paradigm for trustworthy multi-
modal RAG. We hope this shift towards evidence-
based corroboration paves the way for future sys-
tems that demand high-fidelity reasoning.
8

Limitations
Despite the substantial advancements achieved,
BayesRAG is currently subject to certain limi-
tations that point towards future research direc-
tions. First, we observe aRetrieval-Generation
Gap. While our Bayesian retriever achieves high
effective recall (61.7% @ Top-10 and 76.6% @
Top-20 in Table 4), the downstream generation ac-
curacy is bounded by the current reasoning capa-
bilities of MLLMs when interpreting information-
dense visual evidence. We anticipate that as gen-
erator models evolve, they will better utilize the
high-fidelity evidence provided by our framework.
Second, as our methodology is specialized formul-
timodalretrieval, its advantages are naturally less
pronounced in domains dominated by pure text or
massive continuous tables, such as certain financial
reports. In such contexts, the standard chunking
strategy often fragments tabular continuity. This re-
mains a common challenge inherent to most RAG
pipelines. BayesRAG prioritizes cross-modal con-
sistency, which is most effective when documents
contain rich interplay between texts and images.
References
Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen,
Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei
Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhi-
fang Guo, Qidong Huang, Jie Huang, Fei Huang,
Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng
Li, and 45 others. 2025. Qwen3-vl technical report.
Preprint, arXiv:2511.21631.
Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun
Cho, Andrea Madotto, Chen Wei, Tengyu Ma,
Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed,
Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong,
Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph
Feichtenhofer. 2025. Perception encoder: The best
visual embeddings are not at the output of the net-
work.Preprint, arXiv:2504.13181.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yu-
jie He, and Mohit Bansal. 2024. M3docrag:
Multi-modal retrieval is what you need for multi-
page multi-document understanding.Preprint,
arXiv:2411.04952.Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Om-
rani, Gautier Viaud, CELINE HUDELOT, and Pierre
Colombo. 2025. Colpali: Efficient document re-
trieval with vision language models. InThe Thir-
teenth International Conference on Learning Repre-
sentations.
Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and
Chao Huang. 2025. Rag-anything: All-in-one rag
framework.Preprint, arXiv:2510.12323.
Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li,
Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: A
multi-modal multi-agent framework for document
understanding.Preprint, arXiv:2503.13964.
Chan-Wei Hu, Yueqi Wang, Shuo Xing, Chia-Ju Chen,
Suofei Feng, Ryan Rossi, and Zhengzhong Tu.
2025. mrag: Elucidating the design space of multi-
modal retrieval-augmented generation.Preprint,
arXiv:2505.24073.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems (NeurIPS), pages 9459–
9474.
Yinglu Li, Zhiying Lu, Zhihang Liu, Yiwei Sun, Chuan-
bin Liu, and Hongtao Xie. 2025. Regionrag: Region-
level retrieval-augmented generation for visual docu-
ment understanding.Preprint, arXiv:2510.27261.
Elias Lumer, Alex Cardenas, Matt Melich, Myles
Mason, Sara Dieter, Vamse Kumar Subbiah,
Pradeep Honaganahalli Basavaraju, and Roberto Her-
nandez. 2025. Comparison of text-based and image-
based retrieval in multimodal retrieval augmented
generation large language model systems.Preprint,
arXiv:2511.16654.
Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen,
Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma,
Xiaoyi Dong, Pan Zhang, Liangming Pan, Yu-Gang
Jiang, Jiaqi Wang, Yixin Cao, and Aixin Sun. 2024.
Mmlongbench-doc: Benchmarking long-context doc-
ument understanding with visualizations. InAd-
vances in Neural Information Processing Systems,
volume 37, pages 95963–96010. Curran Associates,
Inc.
Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen.
2025. A survey of multimodal retrieval-augmented
generation.Preprint, arXiv:2504.08748.
Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang,
Linke Ouyang, Zhiyuan Zhao, Tao Chu, Tianyao
He, Fan Wu, Qintong Zhang, Zhenjiang Jin, Guang
Liang, Rui Zhang, Wenzheng Zhang, Yuan Qu, Zhifei
Ren, Yuefeng Sun, Yuanhong Zheng, Dongsheng
Ma, and 42 others. 2025. Mineru2.5: A decoupled
vision-language model for efficient high-resolution
document parsing.Preprint, arXiv:2509.22186.
9

Manan Suri, Puneet Mathur, Franck Dernoncourt,
Kanika Goswami, Ryan A. Rossi, and Dinesh
Manocha. 2025. VisDoM: Multi-document QA with
visually rich elements using multimodal retrieval-
augmented generation. InProceedings of the 2025
Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 6088–6109, Albuquerque, New Mexico.
Association for Computational Linguistics.
Anyang Tong, Xiang Niu, ZhiPing Liu, Chang Tian,
Yanyan Wei, Zenglin Shi, and Meng Wang. 2025.
Hkrag: Holistic knowledge retrieval-augmented gen-
eration over visually-rich documents.Preprint,
arXiv:2511.20227.
Xueyao Wan and Hang Yu. 2025. Mmgraphrag:
Bridging vision and language with inter-
pretable multimodal knowledge graphs.Preprint,
arXiv:2507.20804.
Han Wang, Archiki Prasad, Elias Stengel-Eskin, and
Mohit Bansal. 2025a. Retrieval-augmented genera-
tion with conflicting evidence. InSecond Conference
on Language Modeling.
Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu,
Shihang Wang, Pengjun Xie, and Feng Zhao. 2025b.
ViDoRAG: Visual document retrieval-augmented
generation via dynamic iterative reasoning agents.
InProceedings of the 2025 Conference on Empiri-
cal Methods in Natural Language Processing, pages
9124–9145, Suzhou, China. Association for Compu-
tational Linguistics.
Kaiwen Wei, Rui Shan, Dongsheng Zou, Jianzhong
Yang, Bi Zhao, Junnan Zhu, and Jiang Zhong.
2025. Mirage: Scaling test-time inference with
parallel graph-retrieval-augmented reasoning chains.
Preprint, arXiv:2508.18260.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang,
Shengyuan Chen, Zijin Hong, Xiao Huang, and Jin-
song Su. 2025. When to use graphs in rag: A com-
prehensive analysis for graph retrieval-augmented
generation.Preprint, arXiv:2506.05690.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025a. Vis-
RAG: Vision-based retrieval-augmented generation
on multi-modality documents. InThe Thirteenth In-
ternational Conference on Learning Representations.
Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Jun-
hao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang,
Xu Han, Zhiyuan Liu, and Maosong Sun. 2025b.
VisRAG: Vision-based retrieval-augmented genera-
tion on multi-modality documents. InThe Thirteenth
International Conference on Learning Representa-
tions.
Xu Yuan, Liangbo Ning, Wenqi Fan, and Qing Li.
2025. mkg-rag: Multimodal knowledge graph-
enhanced rag for visual question answering.Preprint,
arXiv:2508.05318.Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
Preprint, arXiv:2506.05176.
Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma,
Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong
Yu. 2025. DocBench: A benchmark for evaluating
LLM-based document reading systems. InProceed-
ings of the 4th International Workshop on Knowledge-
Augmented Methods for Natural Language Process-
ing, pages 359–373, Albuquerque, New Mexico,
USA. Association for Computational Linguistics.
10

A Implementation of Bayesian inference
A.1 Likelihood Estimation via
Dempster-Shafer Theory
In this appendix, we detail the calculation of
the likelihood term P(Q|E) , which represents
the semantic support of the evidence tuple E=
(etxt, eimg, escreenshot )for the query Q. Our im-
plementation utilizes Dempster-Shafer Theory to
robustly fuse heterogeneous retrieval scores and
explicitly model uncertainty.
A.1.1 Score Normalization and Sharpening
Since retrieval scores from different embedding
models (e.g., dense text embeddings vs. visual em-
beddings) often inhabit different numerical ranges,
direct combination is infeasible. We first apply a
normalization function to map raw scores into a
probabilistic interval[0,1].
For a raw score sfrom modality i, given the
empirically observed maximum ( smax) and mini-
mum ( smin) scores for that modality, the normal-
ized score˜s iis computed as:
˜si=s−s min
smax−smin(5)
A.1.2 Mass Function Construction
We define theFrame of Discernmentas Θ =Y, N ,
representing whether the evidence is “Relevant”
(Y) or “Irrelevant” ( N). For each normalized score
˜si, we construct a Basic Probability Assignment,
denoted as mass function mi(·), based on hyper-
parameters αandβ(representing our trust in the
retriever’s positive and negative capabilities):
mi(Y) =α·˜s i (6)
mi(N) =β·(1−˜s i)(7)
mi(Ω) = 1−m i(Y)−m i(N)(8)
Here, mi(Ω)represents theuncertainty(or igno-
rance) remaining after observing the score. We
enforce mi(Ω)≥0 by clipping. Based on exper-
imental experience, we fix α= 0.7 andβ= 0.6 ,
respectively, for all experiments.
A.1.3 Recursive Evidence Combination
To fuse evidence from multiple sources (Text, Im-
age, Screenshot), we apply Dempster’s Rule of
Combination recursively. Let mcurr be the ac-
cumulated mass and mnew be the mass of the
next incoming modality. The combined mass
mcomb=m curr⊕m newis calculated as follows:First, we compute the conflict coefficient K,
which measures the degree of contradiction be-
tween the two evidence sources:
K=m curr(Y)·m new(N)+
mcurr(N)·m new(Y)(9)
IfK→1 (e.g., K≥0.9 ), indicating extreme
conflict (one source is certain it’s relevant, the other
certain it’s not), we forcefully set the likelihood to
0. Otherwise, we proceed with the update rules:
mcomb(H) =P
X∩Z=H mcurr(X)·m new(Z)
1−K,
∀H∈Y, N,Ω
(10)
Specifically for our binary frame, the update logic
for the relevant hypothesisYis:
mcomb(Y) =1
1−K
mcurr(Y)·m new(Y)
+m curr(Y)·m new(Ω)
+m curr(Ω)·m new(Y)
(11)
A.1.4 Final Score
After fusing all evidence sources, we obtain the
final mass distribution mfinal. To convert this be-
lief function into a probability measure suitable
for ranking (i.e., P(Q|E) ), we apply thePignistic
Probability Transformation, which distributes the
uncertainty mass Ωequally among the singleton
hypotheses:
P(Q|E)≈BetP(Y) =m final(Y)+
mfinal(Ω)
2(12)
This final score reflects the aggregated semantic
likelihood, accounting for both the strength of indi-
vidual matches and the consistency across modali-
ties.
We provide the pseudocode for the likelihood
estimation processP(Q|E)in Algorithm 1.
A.2 Implementation of Priors
We detail the exact algorithms used to compute
the structural prior P(E) for both document layout
and knowledge graph scenarios. The prior P(E)
measures the intrinsic coherence of an evidence
tuple E= (e txt, eimg, escreenshot )derived from
geometric or semantic constraints.
11

Algorithm 1Likelihood Estimation via Dempster-
Shafer Fusion
Require: Raw retrieval scores S=
{stxt, simg, sscreenshot }
Require: Hyperparameters α, β∈[0,1] (Trust
factors)
Require:Normalization functionΦ(·)
Ensure:Likelihood scoreP(Q|E)
1:Initialize:
2:m curr← {Y: 0, N: 0,Ω : 1} {Initial state:
total ignorance}
3:Preprocessing:
4:S′←[Φ(s)forsinS] {Normalize scores to
[0,1]}
5:foreach scores∈ S′do
6:Construct Mass Function (BPA):
7:m new(Y)←α·s
8:m new(N)←β·(1−s)
9:m new(Ω)←max(0,1−m new(Y)−
mnew(N))
10:Compute Conflict CoefficientK:
11:K←m curr(Y)·m new(N) +m curr(N)·
mnew(Y)
12:ifK≥0.999then
13:return0.0{Extreme conflict detected}
14:end if
15:Apply Dempster’s Rule:
16:D←1−K
17:m next(Y)←(m curr(Y)·m new(Y) +
mcurr(Y)·m new(Ω) +m curr(Ω)·
mnew(Y))/D
18:m next(N)←(m curr(N)·m new(N) +
mcurr(N)·m new(Ω) +m curr(Ω)·
mnew(N))/D
19:m next(Ω)←(m curr(Ω)·m new(Ω))/D
20:m curr←m next
21:end for
22:Final Score:
23:Score←m curr(Y) +m curr(Ω)/2
24:returnScoreA.2.1 Layout-Aware Prior Calculation
For document-native formats (e.g., PDFs), we uti-
lize bounding box (BBox) coordinates to determine
spatial proximity. The calculation of P(E) layout
is implemented as a strict geometric consistency
check.
Given a text node t, an image node v, and a lay-
out shortcut node s, we first verify the consistency
of the metadata. The tuple is considered valid only
if all three components originate from the same
source document D. Let pg(·) denote the page
index and d(·,·) denote the Euclidean distance be-
tween the centers of two bounding boxes.
The prior is computed as a binary indicator func-
tion with a relaxation for adjacent pages:
P(E) bbox=(
1.0ifC dist∧ C page
ϵotherwise(13)
where ϵis a small constant (e.g., ϵ= 0.1 ). The
conditions are defined as:
1.Spatial Condition( Cdist): The spatial dis-
tance must be within a fraction of the page
diagonal lengthL diag:
d(t, v)< τ·L diag (14)
where τis a distance threshold parameter (e.g.,
τ= 2).
2.Pagination Condition( Cpage): The elements
must appear on the same or immediately adja-
cent pages relative to the screenshot reference:
max(|pg(t)−pg(s)|,|pg(v)−pg(s)|)< τ page
(15)
This implementation ensures that a high prior prob-
ability is assigned solely to figure-caption pairs or
spatially clustered information.
A.2.2 Knowledge Graph Topology Prior
For the knowledge graph, we construct a weighted
connectivity map and compute the coherence of
the retrieved triplet. This process consists of three
steps: Relation Aggregation, Edge Normalization,
and Triplet Coherence Scoring.
Step 1: Relation Aggregation. First, we map
retrieved entities to their source document chunks.
For any pair of chunks (u, v) , we aggregate the
weights of all relations rconnecting them in the
Knowledge Graph:
Suv=X
r∈Ru,vwr (16)
12

where wris the weight of a specific relation (de-
faulting to 1.0 or weighted by relation type).
Step 2: Probabilistic Edge Normalization. To
convert the unbounded raw score Suvinto a proba-
bility Pedge(u, v)∈[0,1) , we apply an exponential
saturation function. This models the intuition that
diminishing returns apply as evidence accumulates:
Pedge(u, v) = 1−exp(−κ·S uv)(17)
where κis a scaling factor (set to 0.1 in our ex-
periments). This defines the pairwise connectivity
map.
Step 3: Triplet Coherence Scoring. For a can-
didate tuple E= (e id1, eid2, eid3), we retrieve the
three pairwise edge probabilities: p12, p23, p13. We
calculate their average as the final probability
P(E) graph=p12+p 23+p 13
3(18)
B Reproducibility
In this section, we provide detailed configurations
for the baseline models to ensure reproducibility.
For most parameters, we use the default settings in
the original repository. Across all RAG systems,
we employ identical embedding and reranker mod-
els.
B.1 VisRAG
We follow to the reproducible baseline methodol-
ogy provided by UltraRAG3. We adapt the data
structures of MMLongBench-Doc and DocBench
to align with the benchmark schema required by
UltraRAG. Question answering is conducted using
the official configuration template4. Finally, an-
swer extraction and scoring are performed using
the evaluation scripts provided by the respective
benchmarks.
B.2 ViDoRAG
We follow the official implementation of Vi-
DoRAG5with specific modifications to the em-
bedding module. Specifically, we utilize MinerU
for document parsing. For vector database con-
struction, we employ Qwen3-Embedding-4B for
text segments and nomic-embed-multimodal-3b
for page-level PDF screenshots. We apply the
3https://github.com/OpenBMB/UltraRAG
4https://github.com/OpenBMB/UltraRAG/blob/
main/examples/visrag.yaml
5https://github.com/Alibaba-NLP/ViDoRAGproposed Gaussian Mixture Model (GMM)-based
hybrid strategy to effectively handle multi-modal
retrieval. GPT-4o-mini server as the backbone
model for the Seeker, Inspector, and Answer
agents.
B.3 RAGFlow
We deploy RAGFlow using the official Docker
image6. We employ Qwen3-Embedding-4B and
bge-reranker-v2-m3 (host via VLLM) for text
embedding and reranking, respectively. For mul-
timodal content, Qwen3-VL-32B is utilized to gen-
erate descriptions. All other parsing parameters
retain their default settings.
During the question-answering phase, we use
GPT-4o-mini and enable the following features:
Keyword Analysis,TOC Enhance,Use Knowledge
Graph, andReasoning. The specific chat configu-
ration parameters are detailed in Listing 1.
Listing 1: Configuration parameters for RAGFlow.
{
" language ": " English ",
" llm ": {
" max_tokens ": 4096,
" model_name ": "azure -gpt -4o- mini___OpenAI -
,→API@OpenAI -API - Compatible ",
" temperature ": 0.1
},
" meta_data_filter ": {
" method ": " automatic "
},
" prompt ": {
" empty_response ": "",
" keyword ":true,
" keywords_similarity_weight ": 0.7,
" opener ": "Hi! I'm your assistant . What can I do
,→for you ?",
" prompt ": " You are an intelligent assistant .
,→Please summarize the content of the
,→knowledge base to answer the question .
,→Please list the data in the knowledge
,→base and answer in detail . When all
,→knowledge base content is irrelevant to
,→the question , your answer must include
,→the sentence \'The answer you are
,→looking for is not found in the
,→knowledge base !\'Answers need to
,→consider chat history .\n Here is
,→the knowledge base :\n { knowledge }\n
,→The above is the knowledge base .",
" reasoning ":true,
" refine_multiturn ":false,
" rerank_model ": "bge - reranker -v2 - m3___OpenAI -
,→API@OpenAI -API - Compatible ",
" show_quote ":true,
" similarity_threshold ": 0.2,
" toc_enhance ":true,
" top_n ": 8,
" tts ":false,
" use_kg ":true,
" variables ": [
{
" key ": " knowledge ",
" optional ":true
}
]
},
" prompt_type ": " simple ",
" top_k ": 1024
}
6https://hub.docker.com/r/infiniflow/ragflow
13

B.4 RAGAnything
Consistent with our other setups, we employ
MinerU as the document parser. We utilize
Qwen3-VL-32B to generate document descriptions
and construct the knowledge graph. The vector re-
trieval component relies on Qwen3-Embedding-4B
andbge-reranker-v2-m3 . Finally, GPT-4o-mini
is employed as the downstream question-answering
model.
C Prompts in BayesRAG
The specific prompts employed in BayesRAG are
detailed below:
Prompt
You are an intelligent assistant. Please
summarize the content of the knowledge
base (both in text, images, and screenshots
of pdf pages) to answer questions, think
step by step.
First, analyze the keywords in the question,
and then check whether all texts and images
of the given knowledge base contain
content related to the keywords in the
question. Finally summarize and answer
questions.
When all knowledge base content is not
related to the question, your answer should
include the sentence “The answer you want
is not found in the knowledge base!”
Here is the question:
Question: {question}
The above is the question.
Here is the knowledge base:
{knowledge}
The above is the knowledge base.
Figure 4: Prompts used by the model to predict answers.Prompt
You are required to determine if a predicted
answer is correct or can reasonably answer
the question compared to the ground
truth. The question will be placed within
<question></question> tags, answer type
will be placed within <type></type>
tags, predicted answer will be placed
within <predict></predict> tags, and the
ground truth answer will be placed within
<gt></gt> tags.
You must output the final score as a
floating-point number (0.0 to 1.0) enclosed
within ‘<answer>’ tags.
Output Format: Outputonlythe final
numerical score inside the tags. Do not
provide reasoning.
Example:
<answer> 1.0 </answer>
Question: <question> {question} </ques-
tion>
Predict Answer: <predict> {predict answer}
</predict>
Ground Truth: <gt> {reference answer}
</gt>
Figure 5: Prompts used by evaluating the answer in
MMLongBench-Doc.
D Additional Case Studies
In the main text, we demonstrate cases where
BayesRAG improves retrieval recall for low-
salience details and disambiguates visually similar
charts. In this appendix, we provide additional
qualitative examples highlighting a critical advan-
tage of our framework: the ability to solve complex,
multi-hop reasoning tasks where baseline methods
fail to retrieve the correct evidence entirely.
A common failure mode in existing multimodal
RAG systems is the inability to locate evidence
that is intricately embedded within structured doc-
uments, such as complex tables, stacked charts, or
dense infographics. Single-stream retrievers (re-
14

Question:  What is the average performance gap 
between the ProgramFC  performance and our 
proposed system in the HOVER dataset? Please write 
down the answer in float format with 2 decimals.VisRAG:  
Not answerable.ViDoRAG:  
5.27
RAGFlow:  
Not answerable.RAG -Anything:  
Not answerable.
BayesRAG:  
Our QACHECK system achieves a macro -F1 score of 
55.67, 54.67, and 52.35 on HOVER two -hop, three -hop, 
and four -hop claims, respectively. To find the average 
performance gap, the scores for ProgramFC  are 
provided in Table 1 on page 5 (and repeated on page 6) 
are 54.27, 54.18, 52.88. Now, we calculate the 
performance gap for each hop: 55.67 -54.27=1.40, 
54.67 -54.18=0.49, 52.35 -52.88= -0.53. Finally, we 
calculate the average of these gaps: (1.40 + 0.49 + ( -
0.53)) / 3 = 0.4533...Rounded to two decimal places, 
the average performance gap is 0.45 .Ground Truth:  0.45 provenance page
Figure 6: Complex tabular reasoning requiring multi-hop numerical extraction.
Question:  How many Republicans are not confident at all that 
Trump keeps his own business interests separate from the 
decisions he makes as president?VisRAG:  
Not answerable.ViDoRAG:  
32%
RAGFlow:  
22%RAG -Anything:  
Not answerable.
BayesRAG:  
The relevant information is found on page 7 of the 
document, which includes a chart titled "Fewer than 
half are confident Trump keeps business interests 
separate.” This chart breaks down the confidence 
levels by political affiliation. For the "Rep/Lean Rep" 
(Republicans and Republican -leaning 
independents) category, the data shows: 10% **Not 
at all confident:** …, the answer is 10% . Ground Truth:  10%provenance page
Figure 7: Fine-grained visual-textual alignment in stacked bar charts.
Question: How many ways to connect with pwc
people?VisRAG:
Not answerable.ViDoRAG:
2
RAGFlow:
4RAG -Anything:
8
BayesRAG:
The answer can be found in the document on 
page 11 (image 11). The document explicitly lists 
the following ways to connect: 1.  Facebook: PwC 
Middle East 2.  Instagram: @pwcmiddleeast 3.  
LinkedIn: PwC Middle East 4.  Twitter: 
@PwC_Middle_East 5.  YouTube: PwC Middle 
East 6.  Website: pwc.com/ mecareers , there are 
6primary ways to connect with PwC people as 
listed in the document. Ground Truth: 6provenance page
Figure 8: Information enumeration from unstructured infographics.
15

lying solely on text chunks or naive visual em-
beddings) often miss these cues due to a lack of
fine-grained semantic alignment between the query
and the visual structure.
BayesRAG addresses this through its parallel re-
trieval architecture, simultaneously querying text
chunks, cropped image regions, and full-page PDF
screenshots. Crucially, the subsequent Bayesian fu-
sion mechanism calculates a posterior probability
representing the semantic consistency between the
query and these heterogeneous evidence sources.
This allows BayesRAG to identify and utilize evi-
dence that requires establishing strong cross-modal
links (e.g., mapping legend colors to textual cate-
gories, or aligning table headers with specific nu-
merical cells), tasks where baselines frequently hal-
lucinate or fail to answer.
We present three such challenging scenarios be-
low: complex tabular reasoning (Figure 6), fine-
grained chart interpretation (Figure 7), and info-
graphic enumeration (Figure 8).
E Reliability of Automated Evaluation
To address concerns regarding the reliabil-
ity of Qwen3-32B as an automated judge in
MMLongBench-Doc, we conduct a human verifi-
cation study on a stratified sample of 200 instances.
Our analysis yields two critical observations that
validate our evaluation protocol:High Human-
Model Agreement.We observe an exceptionally
high consistency between the Qwen3-32B judge
and expert human annotators, achieving an agree-
ment rate of98%. Disagreements are rare and
mostly occur in highly ambiguous cases where
even humans might diverge. This statistical evi-
dence strongly supports the use of Qwen3-32B as
a surrogate for human evaluation in this specific
context.Suitability for Factoid QA.The high
reliability stems from the intrinsic nature of the
MMLongBench-Doc dataset. Unlike open-ended
generation tasks, this benchmark primarily consists
of factoid questions requiring specific numerical
values or fixed entities as answers. In such closed-
ended scenarios, the evaluation task simplifies to
verification rather than subjective critique.
Crucially, our qualitative analysis highlights that
the LLM judge effectively resolves the limitations
of the original rule-based scoring scripts. Tradi-
tional regex-based methods often produce false
negatives due to valid lexical variations. By recog-
nizing semantic equivalence, Qwen3-32B providesa more accurate assessment of model performance
than rigid string matching, ensuring that correct
answers are not unfairly penalized due to minor
formatting deviations.
F Efficiency and Computational Analysis
To evaluate the practical overhead of BayesRAG,
we break down the time latency across the entire
inference pipeline. Table 6 details the average time
costs recorded on the MMLongBench-Doc. All ex-
periments are conducted on a server with NVIDIA
Tesla A800 80GB GPUs and a 12-core CPU.
Module Time (s)
Vector Retrieval (Text) 5.81
Vector Retrieval (Image) 26.18
Vector Retrieval (Screenshot) 52.09
Text Reranking (bge-reranker-v2-m3) 9.36
Bayes Fusion & Reranking 10.21
LLM Generation (QA) 29.17
Table 6: Average time cost (seconds) per query break-
down. Note thatVector Retrieval (Image/Screenshot)
is using an expanded search space (Top- k=512).Bayes
Fusionrepresents the re-ranking overhead introduced
by our method.
Retrieval Latency vs. Recall.As observed, the
visual retrieval components (Image and Screenshot)
dominate the latency, consuming approximately
52s. This is an intentional design choice rather
than an algorithmic limitation. To validate the
“Recall Enhancement” capabilities of BayesRAG
(as discussed in Section 3.4), we expand the im-
age retrieval window size to Top- 512(compared
to the standard Top-10 or Top-20). This ensures
that low-salience evidence is captured in the ini-
tial candidate pool. In production environments,
this latency can be significantly reduced via paral-
lelized asynchronous retrieval, etc, independent of
our proposed fusion algorithm.
Efficiency of Bayesian Inference.A critical find-
ing is the comparison between standard text rerank-
ing and our Bayesian fusion process. TheBayes
Fusionstep takes10.21s, which introduces only
a marginal overhead ( ∼0.85s) compared to the
standardText Reranking(9.36s). Theoretically,
combining heterogeneous candidates (Text, Image,
Screenshot) from a Top-512 pool could lead to a
combinatorial explosion ( 5123≈134 million tu-
ples). However, BayesRAG circumvents this issue
16

Method Reports Tutorials Academic Guidebooks Brochures Industry Financial Overall Generator Model
RAGAnything 45.3 46.0 34.6 43.5 39.6 33.3 42.7 41.5GPT-4o-miniBayesRAG (ours) 46.7 53.6 34.8 46.4 39.4 38.6 48.6 44.1
RAGAnything 53.4 47.4 40.2 51.9 39.6 37.0 50.4 47.1Qwen3-VL-32BBayesRAG (ours) 61.8 57.8 58.2 58.9 54.9 54.4 65.4 59.4
Table 7: Performance comparison with different generator models on MMLongBench-Doc. The switch to the
more capable Qwen3-VL-32B results in a significantly larger performance boost for BayesRAG compared to
RAGAnything, highlighting our method’s ability to retrieve high-quality evidence that stronger models can better
exploit.
MethodDomains TypesOverall Generator Model
Aca. Fin. Gov. Law. News Text Meta MM. Unans.
RAGAnything 51.4 42.3 43.2 50.2 57.5 77.6 10.0 57.4 11.2 48.7GPT-4o-miniBayesRAG (ours) 52.4 37.5 56.0 52.3 66.8 84.2 19.7 43.8 25.8 51.2
RAGAnything 57.4 44.7 49.3 50.7 61.0 82.0 11.2 60.7 19.3 52.4Qwen3-VL-32BBayesRAG (ours) 65.6 53.8 61.4 59.6 70.3 87.3 30.2 62.6 39.5 61.7
Table 8: Performance comparison on DocBench with different generators. BayesRAG consistently outperforms
baselines, with the performance gap widening significantly when using the stronger Qwen3-VL-32B generator,
verifying superior evidence recall.
by imposingmetadata constraintsbefore proba-
bility calculation. Specifically, we restrict the con-
struction of evidence tuples E= (t, v, s) to those
derived from the same source document. Further-
more, candidates yielding low probabilities—based
on retrieval scores and knowledge graph distribu-
tions—are directly pruned. These constraints ef-
fectively transform the dense combinatorial space
into a highly sparse matrix, significantly reduc-
ing computational complexity. This confirms that
BayesRAG provides a computationally efficient
solution for sophisticated multimodal reasoning,
effectively shifting the heavy lifting from expen-
sive LLM reasoning to a lightweight probabilistic
verification layer.
G Impact of Generator Models
To disentangle the contribution of the retrieval mod-
ule from the generation capabilities, we conduct a
comparative analysis using different backbones:
GPT-4o-miniand the more powerfulQwen3-
VL-32B. The results on MMLongBench-Doc and
DocBench are detailed in Table 7 and Table 8.
Performance Sensitivity to Generator Capacity.
As expected, replacing the generator with the SOTA
Qwen3-VL-32B leads to universal performance
improvements. However, a critical observation is
thedisproportionate gainachieved by BayesRAG
compared to the baseline RAGAnything. OnMMLongBench-Doc, while RAGAnything im-
proves by 5.6% (41.5% →47.1%), BayesRAG
achieves a dramatic surge of15.3%(44.1% →
59.4%). A similar trend is observed on DocBench,
where the performance gap between BayesRAG
and RAGAnything widens significantly from 2.5%
under GPT-4o-mini to 9.3% under Qwen3-VL-
32B.
Evidence for Superior Effective Recall.This
phenomenon provides compelling evidence for the
superior retrieval quality of BayesRAG. In RAG
systems, the performance is bounded by the qual-
ity of the retrieved context. A stronger genera-
tor cannot hallucinate correct answers if the sup-
porting evidence is missing (low recall). The fact
that BayesRAG benefits substantially more from a
stronger generator indicates that our Bayesian fu-
sion mechanism successfully retrieves high-fidelity,
complex multimodal evidence that is present in the
context but under-utilized by the weaker GPT-4o-
mini. In contrast, the baseline’s smaller improve-
ment suggests its retrieval bottleneck lies in miss-
ing evidence (low recall), which even a superior
generator cannot mitigate. Thus, these results con-
firm that BayesRAG delivers significantly higher
Effective Recall, providing a richer and more ac-
curate evidence set that unlocks the full potential
of advanced MLLMs.
17