# DeCoRAG: Cognitive Decoupling and Semantic-Aware Cropping for Complex Document Understanding

**Authors**: Shuo Wang, Kai Zhang, Wenyuan Huang, Yizheng Yu, Xia Liao, Junming Su, Qing Wang, Fang Xi

**Published**: 2026-07-27 15:28:02

**PDF URL**: [https://arxiv.org/pdf/2607.24554v1](https://arxiv.org/pdf/2607.24554v1)

## Abstract
Advancing multimodal retrieval-augmented generation (RAG) for complex document understanding presents a formidable dual dilemma of accuracy and efficiency, particularly in graph RAG. Processing structurally sparse yet visually dense layouts, such as extracting a tiny data marker from a financial chart, often incurs computationally prohibitive token overhead while still triggering catastrophic hallucination. However, multimodal Graph RAG pipelines rely on graph-construction stages that assume Vision-Language Models (VLMs) can resolve sparse semantics within high-density layouts. We challenge this assumption, revealing that forcing VLMs to localize visual evidence, interpret semantics, and extract relations triggers a "Visual Attention Sink," a mechanism driving catastrophic semantic loss, while full-page processing incurs massive computational overhead. Controlled interventions verify that this failure is boundary-driven rather than content-specific and that semantic anchoring mitigates it. To fundamentally correct this flawed paradigm, we introduce DeCoRAG, a multimodal Graph RAG pipeline that shifts knowledge processing from coupled visual-semantic reasoning to "Cognitive Decoupling." Rather than passively processing raw pixels, its graph-construction stage establishes a macroscopic Semantic Anchor to neutralize the attention sink. This anchor subsequently drives our Region-Aware Pruning and Cropping (RAP-Crop) mechanism, shifting the reasoning space from dense, noisy backgrounds to purified, intent-driven semantic clusters. The resulting graph supports hybrid retrieval and answer generation. Across complex document benchmarks, DeCoRAG improves the semantic pass rate by up to 12.5 percentage points over the strongest baseline and generalizes to DocVQA. RAP-Crop reduces offline graph-construction prompt tokens by 40.8% without sacrificing end-to-end accuracy.

## Full Text


<!-- PDF content starts -->

1
DeCoRAG: Cognitive Decoupling and
Semantic-Aware Cropping for Complex
Document Understanding
Shuo Wang, Kai Zhang, Wenyuan Huang, Yizheng Yu, Xia Liao, Junming Su, Qing Wang, and Fang Xi
✦
Abstract—Advancing multimodal retrieval-augmented generation
(RAG) for complex document understanding presents a formidable
dual dilemma of accuracy and efficiency, particularly in graph RAG.
Processing structurally sparse yet visually dense layouts—such as
extracting a tiny data marker from a financial chart—often incurs
computationally prohibitive token overhead while still triggering
catastrophic hallucination. However, multimodal Graph RAG pipelines
rely on graph-construction stages that assume Vision-Language Models
(VLMs) can resolve sparse semantics within high-density layouts. We
challenge this assumption, revealing that forcing VLMs to localize visual
evidence, interpret semantics, and extract relations triggers a “Visual
Attention Sink”—a mechanism driving catastrophic semantic loss—
while full-page processing incurs massive computational overhead.
Controlled interventions verify that this failure is boundary-driven rather
than content-specific and that semantic anchoring mitigates it. To
fundamentally correct this flawed paradigm, we introduce DeCoRAG, a
multimodal Graph RAG pipeline that shifts knowledge processing from
coupled visual-semantic reasoning to “Cognitive Decoupling.” Rather
than passively processing raw pixels, its graph-construction stage
establishes a macroscopic Semantic Anchor to neutralize the attention
sink. This anchor subsequently drives our Region-Aware Pruning and
Cropping (RAP-Crop) mechanism, shifting the reasoning space from
dense, noisy backgrounds to purified, intent-driven semantic clusters.
The resulting graph supports hybrid retrieval and answer generation.
Evaluated on complex document benchmarks, DeCoRAG outperforms
dense visual retrievers and multimodal Graph RAG baselines, yielding
up to a 12.5% absolute semantic pass rate improvement over the
strongest baseline. DocVQA results demonstrate generalization beyond
sparse charts and tables. Crucially, RAP-Crop reduces offline graph-
construction prompt tokens by 40.8% without sacrificing end-to-end
accuracy.
Index Terms—Multimodal RAG, Graph RAG, complex document under-
standing, visual attention sink, cognitive decoupling.
1 INTRODUCTION
The rapid evolution of Large Language Models (LLMs) [1],
[2] has established Retrieval-Augmented Generation (RAG)
Corresponding author: F. Xi.
S. Wang, K. Zhang, W. Huang, Y. Yu, J. Su, Q. Wang, and F. Xi are with
QiYuanLab, Beijing, China.
E-mail: wangshuo@qiyuanlab.com; zhangkai@qiyuanlab.com;
huangwenyuan@qiyuanlab.com; yuyizheng@qiyuanlab.com;
sujunming@qiyuanlab.com; d202510468@xs.ustb.edu.cn;
xifang@qiyuanlab.com.
X. Liao is with Beijing University of Posts and Telecommunications, Beijing,
China.
E-mail: liaoxia@bupt.edu.cn.[3]–[8] as the standard paradigm for open-domain QA. To
further support complex reasoning, this paradigm is evolv-
ing intoGraph RAG[9], [10], which incorporates struc-
tured relational representations (e.g., Knowledge Graphs).
However, extending Graph RAG tohigh-density industrial
documents—such as financial reports, engineering schemat-
ics, and academic papers [11]–[15]—presents a formidable
challenge. Unlike plain web texts, these documents are char-
acterized by a dense interplay of visual charts, tabular data,
and unstructured layouts. Extracting high-fidelity struc-
tured knowledge from such heterogeneous sources requires
not only recognizing optical characters but also precisely
comprehending the topological relationships between visual
entities.
Current MultimodalGraph RAGsystems typically rely
on a two-stage pipeline: graph construction (extracting en-
tities and relations) followed by graph-grounded reason-
ing. However, when applied to complex documents, this
pipeline faces a “Dual Dilemma” of accuracy and efficiency.
First, on the accuracy front (Graph Construction), we
observe the phenomenon ofInformation Collapse: when ex-
isting VLMs [16]–[19] are tasked with extractingstructured
knowledge triples(e.g.,⟨Revenue,Growth,12%⟩) from com-
plex figures and tables [20], they often hallucinate [21]–[23]
or output sparse relations [24], yielding incomplete graphs
that severely degrade downstream reasoning.Second, on
the efficiency front (Visual Encoding), addressing the res-
olution bottleneck creates a prohibitive computational cost.
To discern fine-grained details (e.g., axis labels or footnotes),
models require high-resolution inputs, yet the visual atten-
tion mechanism scales quadratically with image size [25]–
[27]. Standard solutions are ill-suited here: sliding windows
sever the global semantic topology required for cross-region
relations, while downsampling leads to unreadable content.
We argue that this dual dilemma originates from
a system-level limitation in current multimodal graph-
construction pipelines. Existing systems commonly couple
visual localization, semantic interpretation, and relational
graph extraction within a single VLM inference process.
While operationally simple, this monolithic stage must
discover sparse evidence regions, preserve global layout
semantics, and induce structured knowledge relations si-
multaneously. These responsibilities compete for the same
visual-token and attention budget, making graph construc-
arXiv:2607.24554v1  [cs.IR]  27 Jul 2026

2
tion both unreliable and inefficient as document resolution
and layout complexity increase.First, the accuracy failure
manifests as the Visual Attention Sink.By visualizing
the attention distribution, we discover that without explicit
semantic guidance, VLMs fail to ground their attention on
information-dense visual regions. Instead, the probability
mass “escapes” to the visual sink tokens (e.g., the top-left
corner [28]), mirroring the“Attention Sink”phenomenon
observed in streaming LLMs [29]–[31]. This model-level
failure mode is systematically exposed by the overloaded
graph-construction stage, leading to a “cognitive abandon-
ment” where the model generates hallucinated graph rela-
tions based on language priors rather than visual evidence.
Second, the efficiency bottleneck arises becausefull-page
processing contradicts the principle ofVisual Sparsity.
Semantically valid information (e.g., discrete data points
and textual labels) is often spatially localized, yet full-
page high-resolution parsing compels the model to expend
a disproportionate computational budget on encoding re-
dundant background regions. These observations suggest
that robust complex-document Graph RAG requires reor-
ganizing the knowledge-construction pipeline rather than
merely increasing VLM capacity. They naturally motivate
our decoupled design: utilizing global semantic grounding
to prevent attention drift, and localized high-resolution ex-
traction to exploit visual sparsity efficiently.
To tackle these challenges, we proposeDeCoRAG, a
mechanism-guided multimodal Graph RAG system that re-
organizes graph construction through a“Cognitive Decou-
pling”paradigm. Departing from the conventional “end-to-
end” extraction approach, we reformulate the pipeline into
three hierarchical phases:Phase I: Global Semantic Grounding,
Phase II: Region-Aware Pruning and Cropping (RAP-Crop), and
Phase III: Local Structural Extraction. In Phase I, we generate
a global description that serves as aSemantic Anchor.
This anchor performs two critical functions: (1)Attention
Redirection:It acts as a navigational query, forcing the
model’s attention heads to shift from the sink tokens back
to semantically corresponding visual entities, effectively
mitigating information collapse. (2)Region-Aware Pruning
and Cropping:Guided by the anchor, we introduce the
RAP-Cropalgorithm. Adopting a coarse-to-fine strategy, it
dynamically identifies and crops target clusters (Clusters-of-
Interest), discarding background noise to overcome visual
sparsity and maximize effective information density. Subse-
quently, these refined regions are fed into Phase III forLocal
Structural Extraction, ensuring high-fidelity reasoning.
Our contributions are summarized as follows:
•System-Level Mechanistic Insight:We identify the
“Visual Attention Sink” as a model-level failure
mode systematically exposed by tightly coupled
multimodal graph construction. We substantiate the
diagnosis through attention visualization, relation-
quality analysis, and controlled paired interventions
that separate boundary-driven attention collapse
from content-specific artifacts.
•Mechanism-Guided System Design & Efficiency:
We proposeDeCoRAG, an end-to-end multimodal
Graph RAG system built on theCognitive Decou-
plingparadigm, which adopts a “Describe-then-Extract” approach to bridge macroscopic semantic
anchoring with fine-grained evidence grounding.
Under this paradigm, ourRAP-Cropalgorithm dy-
namically isolates information-dense regions, achiev-
ing a40.8% prompt-token reductionand a29.6%
total-token reductionduring offline FetaTab graph
construction without degrading end-to-end answer
quality.
•Comprehensive System Evaluation:Extensive ex-
periments on SPIQA, SlideVQA, PaperTab, FetaTab,
and DocVQA evaluate end-to-end reliability, com-
ponent effectiveness, cross-layout generalization,
graph-construction efficiency, and robustness. DeC-
oRAG outperforms state-of-the-art visual retrievers
(e.g.,ColPali[32]), multimodal graph frameworks
(e.g.,MMGraphRAG[33]), and standard text-only
pipelines, while evaluator calibration confirms that
the end-to-end Semantic Pass Rate aligns closely
with human judgment.
2 RELATEDWORK
Multimodal and Graph-Based RAG.Retrieval-Augmented
Generation (RAG) [3] has greatly advanced the knowledge-
intensive reasoning capabilities of LLMs. To handle complex
relational reasoning, Graph RAG incorporates structured
Knowledge Graphs into the retrieval pipeline. Concurrently,
to process visually rich documents, multimodal retrievers
like ColPali [32] and VisRAG [34] leverage VLMs for di-
rect patch-level matching. Recent pioneering works, such
as MegaRAG [35] and MMGraphRAG [33], attempt to
bridge these two paradigms by constructing multimodal
knowledge graphs that integrate visual and textual enti-
ties. However, these methods primarily focus on macro-
level retrieval and fusion, often overlooking the microscopic
feature collapse that occurs when VLMs directly encode
high-density, structurally sparse documents. DeCoRAG ad-
dresses this gap through a mechanism-guided “Describe-
then-Extract” system that decouples semantic planning
from high-resolution graph extraction.
Attention Sink and VLM Hallucination.The phe-
nomenon of the “Attention Sink” was initially identified
in streaming LLMs [29], where attention mass dispropor-
tionately accumulates on initial sink tokens rather than
semantically meaningful features. A similar vulnerability
has been observed in Vision Transformers (ViTs), leading to
the introduction of Vision Registers [28] to absorb redundant
attention. In multimodal understanding, this lack of focused
visual grounding is a primary catalyst for object hallu-
cination [24]. In DeCoRAG, we empirically demonstrate
that explicitly anchoring the VLM with a global semantic
description successfully redirects attention away from these
visual traps, thereby recovering the model’s complex struc-
tural extraction capabilities.
Image Cropping for Visual Efficiency.The processing
of high-resolution documents poses a severe computational
bottleneck. Traditional image cropping methods typically
rely on candidate box generation and ranking, often incor-
porating spatial-aware features and rank consistency [36],
[37]. Recent advances explore end-to-end regression archi-
tectures or utilize VLMs via in-context learning to achieve

3
(a) Baseline (Standard Inference)
Attention Drift (Top-left Bias)
Z1
Z2Z3
Z4
Shallow Spatial Graph(b) Ours (Visual Grounding)
Semantic Anchoring (Z 2Pivot)
Z1
Z2Z3
Z4Bounding
BoxRegion
ConstraintsEfficiency
CakeFigure GeoProblem GeoGebra 5
Rich Semantic Graph
Fig. 1.Comparison of Attention Patterns and Graph Generation. (a) Baseline:The attention isdisproportionately attracted tothe top-left
corner (“Attention Sink”), despite scattered focus on other regions. This lack of coherent focus results in aShallow Spatial Graph(bottom),
capturing only fragmented positional relations while missing structural logic.(b) Ours:Our method effectively anchors attention to semantic regions
(Z1-Z4), with a strong focus onZ 2as apivotal spatial anchorthat connects surrounding entities. This precise grounding enables the recovery of
aRich Semantic Graph(bottom) containing complex logical dependencies (e.g.,part of, non-overlap).
free-form and subject-aware cropping (e.g., Cropper [38]).
While these methods excel in photographic composition and
aesthetic enhancement, they are suboptimal for document
parsing [39], where the goal is to preserve discrete, struc-
turally sparse semantic elements (e.g., data markers and
headers) rather than achieving visual harmony. Our RAP-
Crop algorithm departs from aesthetic cropping by dynam-
ically fusing visual density priors with VLM-guided seman-
tic anchors, acting as a targeted token pruning mechanism
that maximizes effective information density for complex
document reasoning.
3 ANALYSIS OFCOUPLEDMULTIMODALGRAPH
CONSTRUCTION
Before detailing our system design, we analyze why conven-
tional end-to-end multimodal graph-construction pipelines
fail on complex documents. These pipelines require a VLM
to localize visual evidence, interpret document semantics,
and induce graph relations within a single coupled inference
process. We empirically show that this coupled formulation
is inherently fragile: when sparse semantic elements are
embedded in high-resolution pages with large redundant
backgrounds, the model often loses reliable visual ground-
ing and produces incomplete or hallucinated graph rela-
tions. We identify theVisual Attention Sinkas a measur-
able manifestation of this coupled-pipeline failure, therebymotivating explicit semantic anchoring before fine-grained
graph extraction.
3.1 Experimental Setup: Visualizing Attention Flow
Unless otherwise specified, all analyses in this section use
Qwen-VL [40] under the same inference configuration. To
reveal the underlying mechanism, we extract the atten-
tion weights from the final transformer layer during the
autoregressive generation process. For each generated text
tokent, we compute a spatial attention mapM t∈RH×W
by averaging the attention weights across all heads and
reshaping the sequence of visual tokens back to the 2D
spatial grid of resolutionH×W. To capture the global visual
grounding throughout the reasoning phase, we perform
temporal aggregation to distill thespatial distributionof at-
tention: ¯M=1
LPL
t=1Mt, whereLdenotes the total length
of the generated text sequence. The resulting heatmap ¯M
quantifies the model’s cumulative focus on specific visual
regions, intuitively serving as a “where the model looked”
summary over the entire reasoning process.
3.2 The Failure Mode: Visual Attention Sink
Task Definition: Open-Schema Visual Extraction.Mul-
timodal Knowledge Graphs have emerged as a promis-
ing paradigm for bridging the semantic gap between un-
structured pixel inputs and interpretable reasoning [33],

4
[35]. A central challenge in this paradigm is to extract
structured knowledge from information-intensive visual
documents without assuming a pre-defined schema. In-
spired by dynamic schema induction [41], we formally
define open-schema visual extraction as follows: given a
figure or table imageI, the model must induce a set of
triplesG={(h, r, t)}capturing bothsemantic data(e.g.,
⟨Revenue,increase by,12%⟩) andvisual topology(e.g.,
⟨Legend,below,X-Axis⟩).
Observation: The Failure Mode of Unanchored Infer-
ence.To investigate the model’sinherent behavioron this
complex visual reasoning task, we evaluate the standard
inference pipeline, as visualized in Figure 1a.
Attention Drift & Spatial Shallowing.Under this base-
line setting, a distinct failure mode emerges. Acting as a
physical manifestation of high uncertainty[29], the at-
tention mass is disproportionately attracted to the top-left
corner (“Attention Sink”), failing to focus on specific objects.
This visual detachment leads to aShallow Spatial Graph.
Quantitatively, the extraction yieldis severely stunted,
capturing only 8 triples (missing over 70% of the 28 ground-
truth relations present in the image). Qualitatively, the
model suffers a severe “Semantic Loss”: itdegeneratescom-
plex topological constraints (e.g.,does notoverlap,part of)
into trivial positional prepositions (e.g.,above,left of). Fur-
thermore, it fails to classifyZ 1as arectangular region
(missingis typeedges), treating it merely as an abstract
symbol.
3.3 The Systemic Trigger: Structural Visual Sparsity
Unlike natural images characterized by spatially contin-
uous visual semantics, information-intensive visual docu-
ments (e.g., charts, tables, and presentation slides) exhibit
Structural Visual Sparsity. As revealed by our empirical
investigation, the valid“Semantic Elements”(e.g., textual
tokens, geometric primitives, and data markers) aredis-
cretely distributedacross the image canvas. In contrast,
the vast majority of the pixel space is dominated by non-
informative background regions (e.g., whitespace, padding,
or decorative borders).
This inherent sparsity poses a fundamental computa-
tional challenge. Since standard Vision Transformers process
images as uniform patches, the model is forced to expend
a disproportionate amount of its computational budget on
encoding redundant background features. Physically, this
results in an extremely lowEffective Information Den-
sity, where the attention mechanism struggles to filter
the sparse, localized signals from the overwhelming back-
ground, thereby increasing the susceptibility to attention
drift.
Motivation for Cognitive Decoupling:Consequently,
a graph-construction stage that relies on unguided self-
attention to localize sparse evidence and extract complex
relations simultaneously is inherently fragile. Structural Vi-
sual Sparsity amplifies the model’s tendency to lose visual
grounding amidst redundant background pixels. We there-
fore formulate a systematic countermeasure: the pipeline
first establishes an explicit cognitive prior—a macroscopic
Semantic Anchor—before performing fine-grained visual ex-
traction.3.4 The Solution: Semantic Anchoring & Graph Recov-
ery
The Verification (Ours): Reversing the Collapse.To verify
the efficacy of our proposed method in addressing this
limitation, we introduce aVisual Groundingintervention
(Figure 1b). By explicitly anchoring attention to semantic
regions (Z 1-Z4), the extracted graph successfully establishes
Z2as apivotal spatial anchor, as extracted triples consis-
tently employ it as thespatial datumto localize surrounding
entities. This intervention reverses the collapse, recovering
aRich Semantic Graphthat retains the full density of 28
triples. Crucially, it restores the missing high-level logical
dependencies (e.g.,non-overlapconstraints among siblings),
showing that stronger visual grounding supports complex
logical reasoning.
Quantitative Evidence: Prevalence and Correlation.To
verify that the “Attention Sink” is a systemic issue rather
than an isolated case, we conduct a quantitative analysis
on 500 complex figure and table images randomly sampled
from the SPIQA dataset [42], a comprehensive benchmark
featuring highly complex and information-dense visual doc-
uments. First, we aggregate the attention maps across all
samples to compute theGlobal Average Attention Maps1.
Under the baseline setting (Figure 2a), the visualization
reveals a pronounced and persistent hotspot strictly lo-
calized at the top-left patch, providing definitive visual
proof that the model universally defaults to this region. In
stark contrast, applying our Visual Grounding intervention
(Figure 2b) effectively mitigates this sink, redistributing the
attention mass towards the bottom-center regions where
core structural elements typically reside.
Furthermore, to establish the causal impact of this phe-
nomenon on extraction performance, we plot the relation-
ship between the severity of the attention sink (Top-Left
Density) and the extraction yield (Extracted Knowledge
Triplets) in Figure 2c. An observablenegative trendis
evident in the Baseline setting (red crosses): as the model al-
locates more attention to the non-semantic sink region (shift-
ing rightwards on the log scale), the extraction capability
severely deteriorates, with the majority of high-sink samples
trapped below 15 triples. Conversely, our Visual Grounding
intervention (green circles) systematically suppresses this
spurious activation, shifting the sink density leftwards into a
benign, low-activation regime. By decoupling the attention
mechanism from this visual trap, our method successfully
restores the model’s inherent extraction capacity.
3.5 Causal Diagnosis Through Controlled Interven-
tions
The prevalence and negative correlation above establish that
the top-left sink is systematic, but correlation alone does
not distinguish a boundary-driven attention attractor from
a specific visual artifact in the original top-left pixels. We
therefore conduct a paired diagnostic study on 100 SPIQA
1. Visualizing raw attention is often hindered by the extreme satu-
ration at the image boundaries (particularly the bottom-right corner),
which corresponds to sequence separators or boundary tokens. To
prevent this artifactual attention from masking the authentic intra-
image spatial attention distribution, we zero out the bottommost row
during visualization.

5
104
103
102
101
Visual Attention Sink (T op-Left Density) [Log Scale]5101520253035Extracted Knowledge Triplets (Count)Method
Baseline
Ours
(a) Avg. Map (Baseline) (b) Avg. Map (Ours) (c) Sink Density vs. Extraction Yield
Fig. 2.Quantitative Validation of the Attention Sink Phenomenon. (a)The aggregated attention map over 500 sampled SPIQA images reveals
a persistent hotspot at the top-left patch during baseline inference.(b)Our Visual Grounding method suppresses this sink, redistributing attention
towards the bottom-center regions.(c)The scatter plot illustrates an observable negative trend between sink severity (X-axis) and extraction yield
(Y -axis). Our method effectively shifts the sample distribution leftwards into a low-activation regime, recovering the inherent extraction capacity.
TABLE 1
Controlled Intervention Study for the Visual Attention Sink.Values are means over 100 paired SPIQA pages; parentheses denote mean
paired changes relative to Original. TL/BR denote top-left and bottom-right.
Input Condition Causal Test COCR↓TL Mass↓Paired Finding
Original baseline 3.49 0.0146 unmodified input
TL mask remove TL pixels 3.56 (+0.08) 0.0155 (+0.0008) no systematic change; not content-driven
Control mask mask non-TL corner 6.62 (+3.14) 0.0220 (+0.0074) generic masking does not mitigate the sink
TL blank margin add blank TL area 3.56 (+0.08) 0.0158 (+0.0011) attention leaves the original content’s TL
region
BR blank margin canvas-size control 2.70 (-0.79) 0.0113 (-0.0033) weak boundary-only reduction
TL shift alter image layout 12.57 (+9.09) 0.0221 (+0.0074) unstable; attention follows the new bound-
ary
Semantic anchor proposed mitigation0.80 (-2.68) 0.0045 (-0.0102) 88/100 improve; validity 75%→85%
pages using the same Qwen-VL backbone and inference
configuration. Each condition preserves the same image-
prompt pair as the original input and modifies only the
visual boundary, the top-left content, or the semantic prior.
We report two sink statistics: the raw top-left attention mass
and the Corner Over-Concentration Ratio (COCR), defined
over the temporally aggregated attention map ¯Mas
COCR( ¯M) =1
|ΩTL|P
(i,j)∈Ω TL¯Mij
1
HWPH
i=1PW
j=1¯Mij,(1)
whereΩ TLdenotes a fixed top-left diagnostic window. A
COCR greater than 1 indicates that the top-left window re-
ceives more attention density than an average image region.
Table 1 provides three mechanistic conclusions. First,
masking the original top-left pixels barely changes either
COCR or TL mass, which rules out the hypothesis that
the hotspot is caused by a specific visible object or OCR
fragment located in that region. Second, blank-margin and
layout-shift interventions show that unanchored attention is
sensitive to image boundaries: the attention mass can leave
the original content’s top-left area and reappear around the
newly induced boundary region. Third, these pixel- and
layout-level edits are diagnostic rather than reliable mitiga-
tions; generic masking can even amplify over-concentration.
In contrast, semantic anchoring directly changes the reason-
ing prior and robustly suppresses the sink, reducing COCR
by 77% and TL mass by 69%. The paired improvement isTABLE 2
Information Richness Assessment.Noise Ratio = (L0+L1)/Total.
DSSR measures core L3 fact retention.
Method Avg. Trip. Noise↓RDI↑DSSR↑
w/o Anchoring 13.5 42.1% 57.3% -
w/ Anchoring 15.3 29.5% 70.5% 84.0%
observed in 88/100 samples (sign testp≈1.9×10−15). The
same samples also improve structured-output validity from
75% to 85%, indicating that attention redirection is coupled
with graph quality rather than being a purely cosmetic
heatmap effect.
Beyond Quantity: Information Richness and Relational
Depth.While the restored extraction yield is clearly evi-
dent, a higher triplet count does not inherently guarantee
richer semantics, as unguided models often inflate outputs
with hallucinated entities or trivial spatial prepositions. To
rigorously quantify this on the same sampled dataset, we
classify the generated relations into a four-tier hierarchy:L0
(Hallucination/Invalid),L1 (Trivial Spatial)(e.g.,above),L2
(Basic Attribute)(e.g.,has), andL3 (High-Order Logical)
(e.g.,increase by).
As detailed in Table 2 and Figure 3, without semantic an-
choring (Baseline), the graph generation is severely inflated
by trivial spatial relations and visual hallucinations (Noise

6
Baseline
(w/o Semantic Anchoring)Ours
(w/ Semantic Anchoring)0246810121416Average Extracted Triplets per ImageOverall Avg: 13.5Overall Avg: 15.3Semantic Distribution of Generated Graph Triplets
L0: Hallucination/Invalid
L1: Trivial Spatial
L2: Basic Attribute
L3: High-Order Logical
Fig. 3.Semantic Distribution of Generated Triplets.Explicit Semantic
Anchoring effectively suppresses L0 and L1 noise while significantly
expanding the density of high-order logical triples.
Ratio: 42.1%). In stark contrast, explicit anchoring substan-
tially suppresses this noise and boosts theRelational Depth
Index (RDI)—the proportion of L2 and L3 triplets—from
57.3% to 70.5%. TheDense Semantic Subsumption Rate
(DSSR)of 84.0% further indicates that most core logical
facts retained by the dense baseline remain present after an-
choring. Together, these results show that attention redirec-
tion improves relational depth without inducing substantial
semantic loss.
Summary and Motivation.In summary, our analysis
identifies the Visual Attention Sink as the principal reliabil-
ity failure exposed when a monolithic graph-construction
stage must localize sparse evidence, preserve global con-
text, and extract relations simultaneously. Structural Visual
Sparsity further makes full-page processing inefficient and
increases susceptibility to attention drift. These findings
motivate a system-level decomposition in which global
grounding first establishes an explicit semantic prior and
subsequent stages perform selective high-resolution extrac-
tion. The following section presents DeCoRAG, which oper-
ationalizes this mechanism-guided design.
4 MECHANISM-GUIDEDSYSTEMDESIGN: DEC-
ORAG
4.1 System Overview and Execution Flow
As established in Section 3, the unguided attention mecha-
nism suffers from theVisual Attention Sink, whileStruc-
tural Visual Sparsityrenders full-page encoding computa-
tionally prohibitive. DeCoRAG is an end-to-end multimodal
Graph RAG pipeline spanning offline document inges-
tion and graph construction, online hybrid graph retrieval,
and answer generation. Within this complete pipeline,
DeCoRAG addresses the diagnosed failures by replacing
the monolithic graph-construction stage with a “Describe-
then-Extract” hierarchical process (Figure 4). Its three con-
struction phases communicate through explicit intermedi-
ate representations—the Semantic AnchorA, coordinate-
bearing crop setC, and Multimodal Knowledge GraphG—
beforeGis exposed to the downstream retrieval interface.ThisCognitive Decouplingparadigm operates in three
phases:(1) Phase I: Global Semantic Grounding.The
model generates a dense global description (Semantic An-
chor) from a downsampled view, forcing attention to
ground on valid semantic entities.(2) Phase II: RAP-
Crop.Guided by this anchor and visual morphology, this
algorithm dynamically crops Clusters-of-Interest, preserves
their original page coordinates, and discards background
noise, maximizing effective information density.(3) Phase
III: Local Structural Extraction.The refined high-resolution
patches, contextualized by the global anchor, are fed into the
reasoning module to construct the final Multimodal Knowl-
edge GraphGwithout resolution bottlenecks or attention
drift.
4.2 Phase I: Mitigating Attention Sink via Global Se-
mantic Grounding
As identified in Section 3, when a VLM is directly tasked
with complex structural extraction from a high-resolution
image, the overwhelming visual tokens and lack of explicit
spatial priors cause the attention to drift towards non-
semantic boundary patches. Phase I aims to resolve this
issue by explicitly establishing a strong contextual prior
before any fine-grained reasoning occurs.
Downsampled Global Input.Instead of feeding the
original high-resolution imageI HR, which exacerbates the
quadratic complexity of visual attention, we first generate
a downsampled global viewI global . This step captures the
macroscopic layout and topology of the document while
keeping the visual token count strictly bounded, effectively
preventing an early computational bottleneck.
Semantic Anchor Generation.We instantiate the VLM
with a domain-expert persona (e.g., “Scientific Data An-
alyst”) and prompt it to analyzeI global . The objective is
to extract a dense, information-rich summary, which we
term theSemantic Anchor(A). Specifically, the model is
constrained to bypass superficial visual aesthetics and di-
rectly output key structural components: (1) Figure/Table
Type and Semantics, (2) Structural Keypoints (e.g., Axes,
Headers), (3) Key Insights, and (4) Critical OCR Text.
Mechanistic Impact.From an architectural perspective,
the generation ofAserves as aforced attention redirection
mechanism. By requiring the model to articulate concrete
visual elements (e.g., “Revenue”, “12% Growth”) rather
than immediately solving the complex graph extraction
task, we inject strong semantic priors into the Transformer’s
context window. The resulting control signal increases vi-
sual grounding in subsequent reasoning and reduces the
probability mass assigned to sink regions, as quantified in
Section 3.
4.3 Phase II: RAP-Crop for Exploiting Visual Sparsity
As highlighted in our mechanistic analysis (Section 3), the
Structural Visual Sparsityof document images means that
semantically valid information is highly localized, while
the majority of the pixel space is redundant background.
Standard sliding-window cropping inevitably slices through
these semantic regions or wastes computational budget on
empty patches. To resolve this, we propose theRegion-
Aware Pruning and Cropping (RAP-Crop)algorithm, a

7
Downsample
111
VLM (Phase l)
Prompted as "Scientific Data        
Analyst" to mitigate attention sink.
Title:
Semantic Anchor
(Dense Global
Description)
Semantic AnchorVisual Density Stream
(Morphology)
Fusion & Dynamic Cropping
Col Identification & Background Discarding
Token Saving                High-Res  Local
Crops ({��})Anchor十
Guidance
High-Res Local
Crops({��})
VLM (Phase III)
Anchor-guided high-fidelity reasoning
Output: Multimodal      
Knowledge Graph(�)
8
ounodone cndgh bondbo ocrtobontothtecouoy 
ote
51.831.9
39.8Dus
m
3.2
Input:High-Resolution 
Chart Image (�퐻 )Semantic Grounding 
Stream(Regex/OCR)Financial reportPhase I: Global   
Semantic GroundingPhase Il: RAP-Crop AlgorithmPhase Ill: Anchor-Guided
Structural Extraction
•Chart Type:
     Multi-line plot
•Key Variables:
     Revenue, Growth
•OCR Text:
     Q4 peak, Legend A
Financial ReportToken Reduction
(Revenue)
(Q4) (12%)increase_by peaked_in
Fig. 4.The Cognitive-Decoupling Graph-Construction Subsystem within DeCoRAG.This subsystem separates semantic planning, evidence
routing, and local execution under a “Describe-then-Extract” paradigm.(Phase I)A VLM first generates a dense Global Description (Semantic
Anchor) from a downsampled view, mitigating the attention sink.(Phase II)Guided by this semantic anchor and visual morphology, dual-stream
RAP-Crop identifies Clusters-of-Interest (CoI), preserves their page coordinates, and prunes redundant background.(Phase III)The refined high-
resolution patches, contextualized by the semantic anchor, are processed by the VLM for high-fidelity structural extraction, yielding the Multimodal
Knowledge Graph used by downstream hybrid retrieval and answer generation.
dual-stream heuristic approach designed to dynamically
extract Clusters-of-Interest (CoI) while systematically dis-
carding redundant visual tokens to minimize computational
overhead (Algorithm 1).
Dual-Stream Region Detection.Instead of relying on
computationally heavy object detectors [43], RAP-Crop
fuses traditional computer vision priors with VLM-guided
semantic priors through a dual-stream architecture:
•Stream A: Visual Density.We hypothesize that chart-
ing elements (lines, axes, data markers) exhibit high
spatial frequency. We apply adaptive thresholding
and morphological dilation on the high-resolution
imageI HRto connect localized ink pixels into con-
tiguous blocks, yielding a set of coarse visual bound-
ing boxesB vis.
•Stream B: Semantic Grounding.To ensure critical tex-
tual elements (identified in Phase I) are not omitted,
we extract a set of semantic keywordsKfrom the
Semantic AnchorAvia regular expressions. We then
employ a lightweight OCR engine to map these
keywords back to their precise physical coordinates,
producing semantic bounding boxesB sem.
The two streams play complementary roles. Stream A pre-
serves high-recall coverage for dense visual structures even
when the downsampled semantic anchor misses small text,
while Stream B injects task-relevant OCR and entity cues
that prevent the cropper from spending budget on visually
dense but semantically irrelevant regions.
Greedy Merging and Token Pruning.The combined
proposalsB all=Bvis∪ Bsem often contain highly overlap-
ping or fragmented regions. We implement a greedy merg-ing strategy that processes candidate boxes in descending
order of their area and iteratively unifies adjacent boxes
whose spatial distance Dist(b, b′)falls below a dynamic
thresholdτ=ρmin(w, h), wherewandhare the page
width and height. Unless otherwise specified,ρ= 0.08and
the crop padding ratio is 0.1. This process yields the final
set of Clusters-of-Interest (B CoI). Finally, a padding ratio is
applied to each bounding box withinB CoIto preserve local
context. By physically cropping only these padded CoIs
into high-resolution sub-image patches (C) anddiscarding
the remaining background, RAP-Crop systematically inter-
rupts the influx of redundant tokens into the Transformer,
achieving substantial token reduction without sacrificing
resolution.
4.4 Phase III: Anchor-Guided Local Structural Extrac-
tion
With the redundant background physically eliminated by
RAP-Crop and the global context encapsulated in the Se-
mantic Anchor (A), the framework advances to the final
reasoning phase. In this stage, the VLM operates under a
multi-granularity multimodal context window, enabling it to
perceive fine-grained local details without losing macro-
scopic topological awareness.
Multi-Granularity Visual Prompting.The input to the
Phase III reasoning module is a carefully orchestrated com-
position. Visually, it receives the downsampled global view
Iglobal (to retain the overall layout) alongside the sequence
of high-resolution sub-image patchesC={c 1, c2, . . . , c k}
extracted by RAP-Crop. Textually, it is prompted with the
Semantic AnchorAgenerated in Phase I, serving as a
powerful cognitive prior.

8
Algorithm 1Region-Aware Pruning and Cropping (RAP-
Crop)
Input: High-resolution imageI HR, Semantic AnchorA,
threshold ratioρ, padding ratioα
Output: High-resolution patches and original-image boxes
C
1:τ←ρmin(WIDTH(I HR),HEIGHT(I HR))
2:Ibinary←ADAPTIVETHRESHOLD(I HR)
3:Idilated ←MORPHOLOGICALDILATION(I binary )
4:Bvis←FINDCONTOURS(I dilated )
5:K ←EXTRACTKEYWORDS(A)
6:Bsem←LIGHTWEIGHTOCR(I HR,K)
7:Ball← B vis∪ Bsem
8:BCoI← ∅
9:whileB all̸=∅do
10:Pop the largest boxbfromB all
11:if∃b′∈ Ballsuch that DIST(b, b′)< τthen
12:b←b∪b′
13:Removeb′fromB all
14:end if
15:B CoI← B CoI∪ {b}
16:end while
17:C ← ∅
18:foreachb∈ B CoIdo
19:b pad←APPLYPADDING(b, α)
20:C ← C ∪ {(CROP(I HR, bpad), bpad)}
21:end for
22:returnC
Crop-Global Coordinate Alignment.The downsampled
global view is used only for coarse semantic and layout
grounding; small-text OCR and numerical evidence are
read from original-resolution crops. To preserve cross-crop
topology, each crop is stored with its original page identifier
and padded bounding box[x 1, y1, x2, y2]. Triples extracted
from local crops are therefore projected back into the global
page coordinate system before graph merging. This design
prevents a crop from becoming an isolated evidence frag-
ment and allows relations such as containment, adjacency,
and non-overlap to be recovered across crop boundaries.
High-Fidelity Relational Reasoning.By feedingAinto
the context window prior to the visual tokens, we funda-
mentally alter the model’s attention dynamics. The VLM
no longer needs to blindly search the canvas or default to
the top-left visual sink tokens when faced with uncertainty.
Instead, its attention is explicitly guided by the anchor to the
corresponding high-resolution sub-image patchesC. This
focused visual grounding allows the model to accurately
read microscopic optical characters (e.g., exact axis values,
footnote texts) and parse complex topological structures.
Structured Graph Generation.Ultimately, the VLM acts
as a structured knowledge parser, tasked with inducing the
open-schema Multimodal Knowledge GraphG. Driven by
the explicit visual grounding, the model successfully over-
comes the “Information Collapse” failure mode identified
in Section 3. It outputs a comprehensive set of relational
triplesG={(h, r, t)}, where the relationsrtranscend
trivial positional prepositions (e.g.,above,left of) and cap-
ture rich, logically complex dependencies (e.g.,non-overlapconstraints,part ofhierarchies, and numerical comparisons
likeincrease by). This marks the successful recovery of
the model’s inherent extraction capacity, yielding a high-
density, high-fidelity knowledge graph.
4.5 System Integration: Hybrid Graph Retrieval
Within the complete DeCoRAG pipeline, Cognitive Decou-
pling redesigns the offline graph-construction subsystem
while the online stage performs hybrid graph retrieval and
answer generation. We integrate the constructed graphG
with a widely adopted hybrid retrieval mechanism to in-
stantiate this end-to-end workflow.
Specifically, given a user query, the system first employs
dense vector similarity to locate semantically relevant an-
chor nodes withinG. Subsequently, it executes a topological
traversal using Personalized PageRank (PPR) starting from
these anchors to fetch the interconnected sub-graphs. These
context-rich snippets are then fed into the LLM for final
answer generation. The end-to-end results in Section 5 show
that the higher-fidelity graphs constructed by DeCoRAG
improve the evidence available to this established retrieval
backend.
5 SYSTEMEVALUATION
5.1 Experimental Setup
Evaluation Scope, Datasets & Baselines.To evaluate DeC-
oRAG as an end-to-end multimodal Graph RAG pipeline,
we examine QA reliability, component effectiveness, cross-
layout generalization, offline graph-construction efficiency,
and parameter robustness. We select three benchmarks ex-
hibiting structural visual sparsity:SPIQA[42], [44]–[46] (sci-
entific figures/tables),SlideVQA[47] (presentation slides),
andPaperTab[48], [49] (academic tables). We further use
DocVQA[14] to evaluate general document VQA beyond
chart- and table-centric sparsity, andFetaTab[48] to mea-
sure high-resolution graph-construction efficiency. We com-
pareDeCoRAGagainst three leading technical paradigms:
(1)Standard Text-Only RAG; (2)End-to-End Visual Re-
trievers(ColPali-V1.2,ColQwen2-V1.0[50]); and (3)Mul-
timodal Graph RAG(MMGraphRAG[33]).
Evaluation Protocol.Evaluating open-schema knowl-
edge extraction is challenging as exact-match metrics pe-
nalize lexically diverse generations. Thus, we employ a 32B
LLM-as-a-judge [51], [52] outputting structured rationales
and scores based on Fact Consistency (40%), Information
Completeness (30%), Logical Structure (20%), and Expres-
sion Quality (10%). Our primary metric is theSemantic Pass
Rate (SPR), where a generation “Passes” if its weighted
score is≥0.5(i.e., factually consistent without severe
hallucinations).
5.2 Evaluator Calibration and Evidence Recall
SPR measures end-to-end QA success after graph retrieval
and answer generation; it is not an intermediate extraction
proxy. To calibrate the LLM-as-a-judge protocol, human
annotators independently judge 100 sampled answers with-
out observing the LLM score. Table 3 reports the resulting
agreement as integer match rates over these 100 samples,

9
TABLE 3
SPR Calibration and Evidence Recall.Recalls are
token-level/ROUGE-1 coverage of gold answer, rationale, and caption
evidence in retrieved passages. SlideVQA does not provide rationale or
caption fields.
DatasetHuman–LLM
AgreementAnswer
RecallRationale
RecallCaption
Recall
SPIQA 95% 68.8% 66.8% 76.8%
SlideVQA 86% 62.1% – –
TABLE 4
Main Results on Three High-Density Benchmarks.Performance is
measured by the Semantic Pass Rate (SPR). By employing cognitive
decoupling, DeCoRAG effectively overcomes information collapse,
significantly outperforming standard text-only pipelines, state-of-the-art
visual retrievers, and existing multimodal graph frameworks (evaluated
using the identical base VLM) across diverse document layouts.
Method SPIQA SlideVQA PaperTab
Text-Only RAG 78.7% 38.1% 62.1%
ColPali-V1.2 83.5% 39.7% 47.7%
ColQwen2-V1.0 86.8% 53.5% 66.8%
MMGraphRAG 77.8% 55.4% 63.9%
DeCoRAG (Ours) 89.2% 67.3% 79.3%
together with evidence recall, which measures whether re-
trieved graph context contains the gold answer, rationale,
and caption evidence. The high Human–LLM agreement
on SPIQA and SlideVQA indicates that SPR is aligned with
human judgment for the evaluated outputs, while evidence
recall verifies that the retrieved graph snippets preserve
source information used by downstream answer generation.
5.3 Main Results: Overcoming Information Collapse
As shown in Table 4,DeCoRAGconsistently establishes
a new state-of-the-art across all benchmarks, effectively
overcoming “Information Collapse”. OnSPIQA, it achieves
an 89.2% SPR, outperforming ColQwen2 (86.8%) andMM-
GraphRAG(77.8%), indicating that structural graphs alone
cannot compensate for the lack of visual decoupling. This
superiority is amplified on structurally extreme datasets.
OnSlideVQA, which features highly irregular layouts de-
manding cross-region multi-hop reasoning, ColPali drops
to 39.7%, consistent with the visual-grounding failure diag-
nosed in Section 3. In contrast,DeCoRAG, explicitly guided
by its Semantic Anchor, maintains a robust 67.3% (+27.6%
improvement). Similarly, on the extreme high-density grids
ofPaperTab, ColPali reaches 47.7%, while our RAP-Crop-
enabled system achieves 79.3%. These results demonstrate
the system-level benefit of combining explicit semantic
grounding with selective high-resolution extraction across
diverse layouts.
5.4 Generalization to Conventional Document VQA
The main benchmarks emphasize visually sparse structural
elements such as charts, tables, slides, and scientific figures.
To evaluate whether the proposed cognitive decoupling re-
mains effective in conventional document layouts, we addi-
tionally test DeCoRAG on 100 randomly sampled DocVQATABLE 5
Generalization Result on DocVQA.ANLS is the official Average
Normalized Levenshtein Similarity metric; LLM QA Pass follows the
same end-to-end SPR criterion.
Dataset ANLS EMMean LLM
ScoreLLM QA
Pass
DocVQA0.968 90.0% 0.926 97.0%
TABLE 6
Cross-Architecture Efficacy of Semantic Anchoring.Evaluated on
SlideVQA. The consistent gains (∆) show that Phase I benefits every
evaluated model scale and architecture.
Model w/o Phase I Full∆
Phi-3.5-Vis (4B) 47.7% 52.3%+4.6%
InternVL2.5 (8B) 49.2% 52.7%+3.5%
LLaVA-OV (7B) 45.4% 53.2%+7.8%
Pixtral-12B 47.9% 53.7%+5.8%
LLaVA-v1.6 (34B) 48.8% 56.4%+7.6%
Qwen3-VL (32B) 56.8%67.3% +10.5%
TABLE 7
Branch-Level RAP-Crop Ablation on 50 FetaTab Queries.Extraction
tokens are graph-extraction calls after the shared semantic-anchor
step; LLM QA Pass follows SPR.
VariantExtraction
Prompt TokensExtraction
Total TokensLLM QA
Pass
Visual-density branch only 660K 1.015M 86.0%
Semantic-grounding branch only 216K 0.322M 84.0%
Dual-stream RAP-Crop 411K 0.772M 86.0%
queries covering scanned documents, forms, letters, and
text-heavy pages. Table 5 reports both official string-based
metrics and judge-based QA metrics. DeCoRAG achieves
0.968 ANLS, 90.0% Exact Match, and 97.0% LLM QA Pass,
indicating that the semantic-anchor and crop-grounded
extraction strategy generalizes beyond sparse chart/table
structures to broad document visual question answering.
5.5 Ablation Study I: Efficacy of Semantic Anchoring
To isolate the contribution of Phase I, we evaluate the
system without the Semantic Anchor (w/o Phase I) on the
SlideVQAbenchmark, a formidable stress test for reasoning
across visually dense pages. TakingQwen3-VL (32B)as a
representative example, eliminating Phase I causes the SPR
to drop from 67.3% to 56.8%. This degradation indicates that
without a macroscopic semantic anchor, the model struggles
to maintain visual grounding. As shown in Table 6, Phase
I yields consistent gains across all evaluated architectures,
supporting its role as a reusable system component rather
than a model-specific prompt optimization.
5.6 Ablation Study II: Dual-Stream RAP-Crop and To-
ken Efficiency
Table 7 isolates the two branches of RAP-Crop on the
same FetaTab subset. The visual-density branch provides
high coverage and reaches 86.0% QA Pass, but it preserves
many dense regions and therefore consumes 660K extraction
prompt tokens. The semantic-grounding branch is much

10
TABLE 8
Full Offline FetaTab Graph-Construction Cost.Similar VLM request counts indicate that RAP-Crop reduces per-request visual input rather than
skipping documents.
Setting Offline Time VLM Req. Prompt Tokens Output Tokens Total Tokens
w/o RAP-Crop 41.85h 124,958 99.54M 36.75M 136.29M
w/ RAP-Crop 38.06h 125,112 58.91M 37.07M 95.98M
Reduction−9.1% +0.1%−40.8% +0.9%−29.6%
cheaper (216K prompt tokens), but its 84.0% QA Pass in-
dicates that anchor-derived keywords alone can miss fine-
grained visual structures. The dual-stream design keeps the
visual branch’s 86.0% QA Pass while reducing extraction
prompt and total tokens by 37.7% and 24.0%, respectively.
Graph scale shows the same trade-off: semantic-only extrac-
tion yields 2.2K nodes and 3.3K edges, whereas the dual
stream yields 6.2K nodes and 11.2K edges, close to the
visual-only graph scale of 6.2K nodes and 11.0K edges.
To quantify RAP-Crop’s ability to break the resolution-
efficiency bottleneck, we measure full offline graph-
construction cost on FetaTab [48], whose high-resolution
Wikipedia images emulate real-world industrial document
workloads. RAP-Crop is applied only during offline graph
construction; after indexing, online retrieval and answer
generation follow the same downstream QA pipeline. As
shown in Table 8, RAP-Crop reduces prompt tokens from
99.54M to 58.91M, a 40.8% reduction, and reduces total
tokens by 29.6%. Output tokens remain stable (+0.9%), and
VLM request counts are essentially unchanged (+0.1%),
which indicates that the gain comes from pruning redun-
dant visual input per request rather than reducing the
amount of processed evidence. The end-to-end SPR remains
stable (90.3% without RAP-Crop versus 90.6% with RAP-
Crop), showing that the observed token reduction preserves
answer quality during graph construction.
Threshold Robustness.RAP-Crop uses the dynamic
thresholdτ=ρmin(w, h)with defaultρ= 0.08and
padding ratio 0.1. Sweepingρover 0.06, 0.08, and 0.10
on 566 SPIQA images preserves dual-stream foreground,
OCR, and reference-OCR recalls at 1.00, while the average
crop count changes only from 1.18 to 1.08. This stability
suggests that crop merging is not brittle across realistic
layout variation.
6 LIMITATIONS ANDFUTUREWORK
While robust, DeCoRAG’s reasoning relies on the Phase I
Semantic Anchor; severe initial hallucinations may cascade,
motivating future Agentic Self-Correction mechanisms. Ad-
ditionally, Stream A relies on heuristic morphological op-
erations which may struggle with heavily degraded docu-
ments, paving the way for end-to-end differentiable routing.
Finally, we plan to extend this paradigm to Inter-Page Graph
Construction for multi-page industrial reasoning.
7 CONCLUSION
This paper diagnoses theVisual Attention Sinkas a model-
level reliability failure systematically exposed by tightly
coupled multimodal graph construction, whileStructuralVisual Sparsitymakes the same monolithic pipeline compu-
tationally inefficient. Controlled paired interventions show
that the sink is not a simple top-left content artifact and
that semantic anchoring substantially reduces both COCR
and top-left attention mass. Guided by this diagnosis,
DeCoRAGreorganizes graph construction into global se-
mantic grounding, dual-stream evidence routing, and lo-
cal high-resolution extraction. Across SPIQA, SlideVQA,
PaperTab, DocVQA, and FetaTab, this mechanism-guided
system improves graph-grounded QA accuracy, generalizes
to conventional document VQA, and reduces offline graph-
construction prompt tokens by 40.8% without degrading an-
swer quality. These results support pipeline-level cognitive
decoupling as an accurate and cost-efficient foundation for
document-scale multimodal Graph RAG.
REFERENCES
[1] H. Touvron, L. Martin, K. Stone, P . Albertet al., “Llama 2:
Open foundation and fine-tuned chat models,”arXiv preprint
arXiv:2307.09288, 2023.
[2] OpenAI, “Gpt-4 technical report,”arXiv preprint arXiv:2303.08774,
2023.
[3] P . Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel,
and D. Kiela, “Retrieval-augmented generation for knowledge-
intensive nlp tasks,” inAdvances in Neural Information Processing
Systems (NeurIPS), 2020.
[4] K. Guu, K. Lee, Z. Tung, P . Pasupat, and M.-W. Chang, “REALM:
Retrieval-augmented language model pre-training,” inInterna-
tional Conference on Machine Learning (ICML), 2020.
[5] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang,
and H. Wang, “Retrieval-augmented generation for large language
models: A survey,”arXiv preprint arXiv:2312.10997, 2023.
[6] V . Karpukhin, B. Oguz, S. Min, P . Lewis, L. Wuet al., “Dense
passage retrieval for open-domain question answering,” inPro-
ceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2020.
[7] G. Izacard and E. Grave, “Leveraging passage retrieval with gener-
ative models for open domain question answering,” inProceedings
of the 16th Conference of the European Chapter of the Association for
Computational Linguistics (EACL), 2021.
[8] Z. Guo, X. Ren, L. Xu, J. Zhang, and C. Huang, “Rag-anything:
All-in-one rag framework,”arXiv preprint arXiv:2510.12323, 2025.
[9] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Tru-
itt, D. Metropolitansky, R. O. Ness, and J. Larson, “From local to
global: A Graph RAG approach to query-focused summarization,”
arXiv preprint arXiv:2404.16130, 2024.
[10] X. He, Y. Tian, Y. Sun, N. V . Chawla, T. Laurent, Y. LeCun, X. Bres-
son, and B. Hooi, “G-Retriever: Retrieval-augmented generation
for textual graph understanding and question answering,” in
Advances in Neural Information Processing Systems (NeurIPS), 2024.
[11] Y. Huang, T. Lv, L. Cui, Y. Lu, and F. Wei, “LayoutLMv3: Pre-
training for document ai with unified text and image masking,” in
Proceedings of the 30th ACM International Conference on Multimedia,
2022.
[12] G. Kim, T. Hong, M. Yim, J. Nam, J. Park, J. Yim, W. Hwang,
S. Yun, D. Han, and S. Park, “Ocr-free document understanding
transformer,” inEuropean Conference on Computer Vision (ECCV),
2022.

11
[13] L. Blecher, G. Cucurull, T. Scialom, and R. Stojnic, “Nougat: Neural
optical understanding for academic documents,”arXiv preprint
arXiv:2308.13418, 2023.
[14] M. Mathew, D. Karatzas, and C. V . Jawahar, “Docvqa: A dataset for
vqa on document images,” inProceedings of the IEEE/CVF Winter
Conference on Applications of Computer Vision (WACV), 2021.
[15] A. Singh, V . Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra,
D. Parikh, and M. Rohrbach, “Towards vqa models that can read,”
inProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2019.
[16] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” in
Advances in Neural Information Processing Systems (NeurIPS), 2023.
[17] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping
language-image pre-training with frozen image encoders and
large language models,” inInternational Conference on Machine
Learning (ICML), 2023.
[18] J.-B. Alayrac, J. Donahue, P . Luc, A. Miechet al., “Flamingo:
a visual language model for few-shot learning,” inAdvances in
Neural Information Processing Systems (NeurIPS), 2022.
[19] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhaoet al., “Instructblip:
Towards general-purpose vision-language models with instruc-
tion tuning,” inAdvances in Neural Information Processing Systems
(NeurIPS), 2023.
[20] Y. Zhu, X. Wang, J. Chen, S. Qiao, Y. Ou, Y. Yao, S. Deng, H. Chen,
and N. Zhang, “Llms for knowledge graph construction and
reasoning: Recent capabilities and future opportunities,”World
Wide Web, 2024.
[21] S. Leng, H. Zhang, G. Chen, X. Li, S. Lu, C. Miao, and L. Bing,
“Mitigating object hallucinations in large vision-language mod-
els through visual contrastive decoding,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2024.
[22] S. Yin, C. Fu, S. Zhao, T. Xu, H. Wang, D. Sui, Y. Shen, K. Li,
X. Sun, and E. Chen, “Woodpecker: Hallucination correction for
multimodal large language models,”Science China Information
Sciences, 2024.
[23] T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen,
F. Huang, Y. Yacoob, D. Manocha, and T. Zhou, “Hallusionbench:
An advanced diagnostic suite for entangled language halluci-
nation and visual illusion in large vision-language models,” in
Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR), 2024.
[24] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen,
“Evaluating object hallucination in large vision-language models,”
inProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing (EMNLP), 2023.
[25] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly
et al., “An image is worth 16x16 words: Transformers for image
recognition at scale,” inInternational Conference on Learning Repre-
sentations (ICLR), 2021.
[26] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo,
“Swin transformer: Hierarchical vision transformer using shifted
windows,” inProceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), 2021.
[27] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. R ´e, “Flashattention:
Fast and memory-efficient exact attention with io-awareness,” in
Advances in Neural Information Processing Systems (NeurIPS), 2022.
[28] T. Darcet, M. Oquab, J. Mairal, and P . Bojanowski, “Vision trans-
formers need registers,” inInternational Conference on Learning
Representations (ICLR), 2024.
[29] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, “Efficient
streaming language models with attention sinks,” inInternational
Conference on Learning Representations (ICLR), 2024.
[30] C. Han, Q. Wang, H. Peng, W. Xiong, Y. Chen, H. Ji, and S. Wang,
“Lm-infinite: Zero-shot extreme length generalization for large
language models,”arXiv preprint arXiv:2308.16137, 2023.
[31] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song,
Y. Tian, C. R ´e, C. Barrett, Z. Wang, and B. Chen, “H 2O: Heavy-
hitter oracle for efficient generative inference of large language
models,” inAdvances in Neural Information Processing Systems
(NeurIPS), 2023.
[32] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot,
and P . Colombo, “Colpali: Efficient document retrieval with vision
language models,” inInternational Conference on Learning Represen-
tations (ICLR), 2025.[33] X. Wan and H. Yu, “Mmgraphrag: Bridging vision and language
with interpretable multimodal knowledge graphs,”arXiv preprint
arXiv:2507.20804, 2025.
[34] S. Yu, C. Tang, B. Xu, J. Cui, J. Ran, Y. Yan, Z. Liu, S. Wang, X. Han,
Z. Liu, and M. Sun, “VisRAG: Vision-based retrieval-augmented
generation on multi-modality documents,” inInternational Confer-
ence on Learning Representations (ICLR), 2025.
[35] C.-H. Hsiao, Y.-C. Wang, T.-S. Lin, Y.-R. Yeh, and C.-S. Chen,
“Megarag: Multimodal knowledge graph-based retrieval aug-
mented generation,”arXiv preprint arXiv:2512.20626, 2025.
[36] C. Wang, L. Niu, B. Zhang, and L. Zhang, “Image cropping with
spatial-aware feature and rank consistency,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2023.
[37] H. Zeng, L. Li, Z. Cao, and L. Zhang, “Reliable and efficient image
cropping: A grid anchor based approach,” inProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), 2019.
[38] S. H. Lee, J. Jiang, Y. Xu, Z. Li, J. Ke, Y. Li, J. He, S. Hickson,
K. Datsenko, S. Kim, M.-H. Yang, I. Essa, and F. Yang, “Cropper:
Vision-language model for image cropping through in-context
learning,” inProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), 2025.
[39] J. Ye, A. Hu, H. Xu, Q. Ye, M. Yan, Y. Dan, C. Zhao, G. Xu,
C. Li, J. Tian, Q. Qi, J. Zhang, and F. Huang, “mPLUG-DocOwl:
Modularized multimodal large language model for document
understanding,”arXiv preprint arXiv:2307.02499, 2023.
[40] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P . Wang, J. Lin, C. Zhou,
and J. Zhou, “Qwen-vl: A versatile vision-language model for un-
derstanding, localization, text reading, and beyond,”arXiv preprint
arXiv:2308.12966, 2023.
[41] J. Bai, W. Fan, Q. Hu, Q. Zong, C. Li, H. T. Tsang, H. Luo,
Y. Yim, H. Huang, X. Zhou, F. Qin, T. Zheng, X. Peng, X. Yao,
H. Yang, L. Wu, Y. Ji, G. Zhang, R. Chen, and Y. Song, “Au-
toschemakg: Autonomous knowledge graph construction through
dynamic schema induction from web-scale corpora,”arXiv preprint
arXiv:2505.23628, 2025.
[42] S. Pramanick, R. Chellappa, and S. Venugopalan, “Spiqa: A dataset
for multimodal question answering on scientific papers,” inAd-
vances in Neural Information Processing Systems (NeurIPS), 2024.
[43] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable
detr: Deformable transformers for end-to-end object detection,” in
International Conference on Learning Representations (ICLR), 2021.
[44] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque, “Chartqa: A
benchmark for question answering about charts with visual and
logical reasoning,” inFindings of the Association for Computational
Linguistics: ACL, 2022.
[45] F. Liu, J. M. Eisenschlos, F. Piccinno, S. Krichene, C. Pang, K. Lee,
M. Joshi, W. Chen, N. Collier, and Y. Altun, “Deplot: One-shot
visual language reasoning by plot-to-table translation,” inFindings
of the Association for Computational Linguistics: ACL, 2023.
[46] N. Methani, P . Ganguly, M. M. Khapra, and P . Kumar, “Plotqa:
Reasoning over scientific plots,” inProceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision (WACV), 2020.
[47] R. Tanaka, K. Nishida, K. Nishida, T. Hasegawa, I. Saito, and
K. Saito, “Slidevqa: A dataset for document visual question an-
swering on multiple images,” inProceedings of the AAAI Conference
on Artificial Intelligence, 2023.
[48] Y. Hui, Y. Lu, and H. Zhang, “UDA: A benchmark suite for
retrieval augmented generation in real-world document analysis,”
inAdvances in Neural Information Processing Systems (NeurIPS),
2024.
[49] W. Chen, H. Wang, J. Chen, Y. Zhang, H. Wang, S. Li, X. Zhou, and
W. Y. Wang, “TabFact: A large-scale dataset for table-based fact
verification,” inInternational Conference on Learning Representations
(ICLR), 2020.
[50] P . Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu,
J. Wang, W. Ge, Y. Fan, K. Dang, M. Du, X. Ren, R. Men, D. Liu,
C. Zhou, J. Zhou, and J. Yin, “Qwen2-vl: Enhancing vision-
language model’s perception of the world at any resolution,”arXiv
preprint arXiv:2409.12191, 2024.
[51] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang,
Z. Lin, Z. Li, D. Li, E. P . Xing, H. Zhang, J. E. Gonzalez, and
I. Stoica, “Judging LLM-as-a-Judge with MT-Bench and Chat-
bot Arena,” inAdvances in Neural Information Processing Systems
(NeurIPS), 2023.

12
[52] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu, “G-eval:
Nlg evaluation using gpt-4 with better human alignment,”arXiv
preprint arXiv:2303.16634, 2023.