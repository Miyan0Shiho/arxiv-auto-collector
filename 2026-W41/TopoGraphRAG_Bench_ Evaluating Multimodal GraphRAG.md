# TopoGraphRAG-Bench: Evaluating Multimodal GraphRAG on Layout-Grounded Evidence Reasoning

**Authors**: Ruochi Li, Jianzhe Lin, Haoxuan Zhang, Haihua Chen, Junhua Ding, Edward Gehringer, Yang Zhang

**Published**: 2026-10-07 03:15:31

**PDF URL**: [https://arxiv.org/pdf/2610.09360v1](https://arxiv.org/pdf/2610.09360v1)

## Abstract
Real-world documents distribute evidence across text, tables, figures, and captions within complex page layouts. Answering complex questions over such documents therefore requires more than retrieving relevant passages: systems must recover the evidence topology that connects heterogeneous evidence units. Existing GraphRAG evaluations remain largely text-centered, while multimodal document RAG benchmarks assess cross-modal retrieval and generation without directly evaluating recovery of the intended evidence topology. We introduce TOPOGRAPHRAG-BENCH, a layout-grounded benchmark for multimodal evidence reasoning in GraphRAG, comprising 2,024 questions over 201 long, visually rich documents. Questions are constructed bottom-up from text, figure, and table evidence units under three controlled topologies: single-hop retrieval, bridge-chain reasoning, and multi-source synthesis. To ensure that questions preserve their intended structure, we apply counterfactual validation for shortcut resistance, modality necessity, and evidence necessity. We evaluate text-only GraphRAG, page-level visual retrieval, and multimodal GraphRAG systems using retrieval, generation, and topology-aware reasoning metrics. Multimodal GraphRAG systems achieve the strongest overall performance, but still fail when visual-textual evidence alignment or multi-unit composition is incomplete. Text-only GraphRAG struggles when key dependencies are grounded in figures or tables, while page-level visual retrieval lacks the fine-grained structure needed for topology recovery. These findings motivate GraphRAG systems that move beyond text-derived entity relation graphs to explicitly model document layouts, cross-modal evidence alignment, and the reasoning roles of evidence units. Code and data are available at https://richardlrc.github.io/TopoGraphRAG-Bench/.

## Full Text


<!-- PDF content starts -->

TopoGraphRAG-Bench: Evaluating Multimodal
GraphRAG on Layout-Grounded Evidence Reasoning
Ruochi Li1, Jianzhe Lin5,†, Haoxuan Zhang2, Haihua Chen3
Junhua Ding4, Edward Gehringer1,*, Yang Zhang3,†,*
1Department of Computer Science, North Carolina State University, USA
2Department of Information Science, University of North Texas, USA
3The Anuradha and Vikas Sinha Department of Data Science, University of North Texas, USA
4School of Computing, University of Wyoming, USA
5Independent Researcher
1{rli14, efg}@ncsu.edu2{haoxuanzhang}@my.unt.edu3{yang.zhang, haihua.chen}@unt.edu
4{junhua.ding}@uwyo.edu5{jianzhelin}@meta.com
Abstract
Real-world documents distribute evidence across text, tables, figures, and cap-
tions within complex page layouts. Answering complex questions over such
documents therefore requires more than retrieving relevant passages: systems
must recover the evidence topology that connects heterogeneous evidence units.
Existing GraphRAG evaluations remain largely text-centered, while multimodal
document RAG benchmarks assess cross-modal retrieval and generation without
directly evaluating recovery of the intended evidence topology. We introduce
TOPOGRAPHRAG-BENCH, a layout-grounded benchmark for multimodal evi-
dence reasoning in GraphRAG, comprising 2,024 questions over 201 long, visually
rich documents. Questions are constructed bottom-up from text, figure, and table
evidence units under three controlled topologies: single-hop retrieval, bridge-
chain reasoning, and multi-source synthesis. To ensure that questions preserve
their intended structure, we apply counterfactual validation for shortcut resistance,
modality necessity, and evidence necessity. We evaluate text-only GraphRAG,
page-level visual retrieval, and multimodal GraphRAG systems using retrieval,
generation, and topology-aware reasoning metrics. Multimodal GraphRAG sys-
tems achieve the strongest overall performance, but still fail when visual-textual
evidence alignment or multi-unit composition is incomplete. Text-only GraphRAG
struggles when key dependencies are grounded in figures or tables, while page-
level visual retrieval lacks the fine-grained structure needed for topology recovery.
These findings motivate GraphRAG systems that move beyond text-derived entity
relation graphs to explicitly model document layouts, cross-modal evidence align-
ment, and the reasoning roles of evidence units. Code and data are available at
https://richardlrc.github.io/TopoGraphRAG-Bench/.
1 Introduction
Recent advances in multimodal large language models and retrieval-augmented generation have
reshaped how intelligent systems interact with complex documents, where evidence is often distributed
across prose, tables, figures, charts, captions, and layout structures. Prior work on document visual
*Corresponding Authors.
†Project Leader.
40th Conference on Neural Information Processing Systems (NeurIPS 2026). Track on Evaluations and Datasets.
arXiv:2610.09360v1  [cs.AI]  7 Oct 2026

question answering, multimodal QA, and long-document understanding has shown that answering
document questions often requires grounding in textual, tabular, and visual sources across pages
and layouts [ 14,20,13]. However, many information needs go beyond single-fact retrieval from
one passage, table cell, or figure caption. Complex questions often require connecting multiple
pieces of evidence [ 23,21]: one unit may reveal an intermediate entity or condition that determines
which subsequent unit is relevant, while several units may jointly support a comparison, summary,
or higher-level pattern. In these cases, the retrieval objective shifts from finding isolated relevant
evidence to recovering the dependency structure among evidence units. We therefore view complex
document retrieval as a structure-recovery problem: the system must identify relevant evidence units
as well as the entities, relations, and higher-order connections that bind them together.
GraphRAG has emerged as a promising paradigm for this structure-recovery view of retrieval by
organizing textual chunks, entities, relations, and summaries into graph structures that support long-
range dependency modeling, multi-hop reasoning, and synthesis beyond local similarity search.
Existing GraphRAG benchmarks evaluate graph construction, knowledge retrieval, and generation
across tasks such as fact retrieval, complex reasoning, contextual summarization, and creative
generation [ 26]. However, these evaluations remain largely text-centered: the evidence graph is
typically built from textual chunks, entities, or textual summaries. In parallel, document-centric
multimodal RAG benchmarks such as MMDocRAG, UniDoc-Bench, and REAL-MM-RAG have
advanced multimodal page retrieval, evidence selection, quote selection, and answer generation over
text, figures, and tables [ 2,17,24]. Yet these benchmarks are not designed to diagnose whether
systems recover the intended evidence topology across multimodal document layouts. This leaves
open the core question for multimodal GraphRAG: whether graph-structured retrieval can recover and
support reasoning over layout-grounded multimodal evidence structures, rather than being limited to
text-derived entities and chunks.
To study this question, we introduce a layout-grounded benchmark for multimodal evidence reason-
ing in GraphRAG. Rather than generating complex questions directly from whole documents, we
construct them bottom up: atomic questions are generated from layout-grounded evidence units and
then composed through explicit evidence links into three reasoning topologies, single-hop retrieval,
bridge-chain reasoning, and multi-source synthesis. We further apply counterfactual validation for
shortcut resistance, modality necessity, and evidence necessity, so that each instance preserves its
intended evidence structure. The resulting benchmark contains 2,024 questions across 201 documents,
including 302 single-hop, 1,014 bridge-chain, and 708 synthesis questions, with 773 cross-modal in-
stances. We evaluate text-only GraphRAG, VisRAG-style page retrieval, and multimodal GraphRAG
systems using retrieval and generation metrics. Results show that multimodal GraphRAG systems
perform best overall, but each paradigm faces distinct limitations: text-only GraphRAG misses visual
and tabular dependencies, page-level visual retrieval struggles with fine-grained evidence topology,
and multimodal GraphRAG remains limited by modality alignment failures and incomplete synthesis
across heterogeneous evidence.
In summary, our contributions are:
•Topology-aware multimodal GraphRAG benchmark.We introduce TOPOGRAPHRAG-
BENCH, a layout-grounded benchmark with 2,024 QA pairs over 201 long documents,
covering single-hop retrieval, bridge-chain reasoning, and multi-source synthesis across text,
figures, and tables.
•Bottom-up evidence-topology construction.We construct questions from verifiable layout-
grounded evidence units rather than whole-document prompts, and use counterfactual
validation to enforce shortcut resistance, modality necessity, and evidence necessity.
•Diagnostic evaluation of GraphRAG paradigms.We evaluate text-only GraphRAG, page-
level visual retrieval, and multimodal GraphRAG systems, showing that current systems
still struggle with cross-modal alignment and fine-grained evidence-topology recovery.
2 Related Work
GraphRAG methods and benchmarks.GraphRAG extends retrieval augmented generation by
organizing information into graph structures over chunks, entities, relations, communities, or sum-
maries, enabling retrieval beyond local similarity search. Representative systems include hierarchical
or community-based retrieval methods such as RAPTOR and Microsoft GraphRAG, graph-indexed
2

systems such as LightRAG, personalized PageRank-based methods such as HippoRAG, and KG-
or GNN-based systems such as G-Retriever, GFM-RAG, and ToG [ 18,3,5,7,8,12,19]. Recent
GraphRAG benchmarks evaluate whether graph-based retrieval improves fact retrieval, complex
reasoning, contextual summarization, and generation over conventional RAG [ 26,30]. However,
these evaluations remain largely text-centred, with graphs constructed from textual chunks, entities,
relations, or summaries, leaving multimodal document evidence structures underexplored.
Multimodal document RAG benchmarks.Document understanding benchmarks have evolved
from single-page DocVQA tasks to long-document and retrieval-augmented settings [ 14,15,31,22,
13,32]. Recent multimodal document RAG benchmarks further evaluate retrieval and generation
across pages, layouts, figures, tables, and screenshots. MMDocRAG introduces multi-page cross-
modal evidence chains, quote selection, and multimodal answer generation; UniDoc Bench compares
text-only, image-only, fusion, and joint retrieval paradigms; REAL MM RAG emphasizes realistic
multimodal retrieval queries and accurate labeling; and MMDocIR and ViDoRe focus on multimodal
document retrieval [ 2,17,24,1,11]. These works provide strong multimodal evaluation settings, but
they are not primarily designed to diagnose whether graph structured retrieval can recover evidence
topologies across multimodal document layouts.
Multi-evidence reasoning and question construction.Multi-hop QA benchmarks such as Hot-
potQA, 2WikiMultiHopQA, MuSiQue, and MultiHop RAG show that complex questions often
require connecting multiple pieces of evidence rather than retrieving a single fact [ 28,9,23,21].
Multimodal benchmarks extend this idea to text, tables, images, charts, and long documents, in-
cluding MultiModalQA, CHARGE, BRIDGE, and DocHop QA [ 20,27,25,16]. Building on these
foundations, we construct questions directly from layout-grounded evidence units in complete doc-
uments, retaining their page, layout, modality, and reasoning roles. Our benchmark covers both
entity-mediated bridge chains and synthesis over 3–6 complementary evidence points, with topology-
specific checks of evidence necessity. These annotations link the required reasoning structure to
retrievable document evidence, supporting separate evaluation of evidence retrieval and evidence
composition in multimodal GraphRAG.
3 Benchmark
3.1 Construction
Pipeline overview.Figure 1 summarizes our benchmark construction pipeline. Starting from layout-
grounded document units, we build evidence units, identify shared entities as cross-layout anchors,
generate questions under three controlled evidence topologies, and apply counterfactual topology
validation to remove shortcut questions, pseudo-multimodal questions, and synthesis questions that
do not require multiple evidence sources.
Layout-grounded evidence units.We build our benchmark on MMDocIR [ 1], a multimodal
document retrieval corpus of long, visually rich documents, and randomly sample 201 documents
from the original collection. Following MMDocIR, we reinterpret layout-level “quotes” as layout-
grounded evidence units. Textual units retain their original text, while visual units, including figures
and tables, preserve the original image and are augmented with OCR and VLM descriptions. Each
unit keeps its document ID, page ID, layout ID, modality label, and semantic representation for entity
extraction, evidence linking, and question construction.
The 201 documents cover nine of MMDocIR’s ten application domains. Figure 2 shows their
distribution using the original MMDocIR domain labels.
Anchor discovery.Inspired by MuSiQue’s bottom-up strategy for constructing multi-hop questions
from connected single-hop questions [ 23], we adapt this principle to multimodal document layouts.
Instead of prompting an LLM over an entire document, we construct questions over explicit evidence
topologies: isolated units support direct retrieval questions, entity-linked units form bridge-chain
questions, and complementary evidence sets support multi-source synthesis.
For each evidence unit, we extract salient entities and concepts using an LLM. Textual units are
processed from their original text, whereas visual units are represented by captions, topics, OCR
3

Figure 1: Overview of the TopoGraphRAG benchmark construction and validation pipeline. Starting
from layout-grounded document evidence units, the pipeline constructs single-hop, bridge-chain,
and multi-source synthesis QA instances, and then applies counterfactual topology validation to test
shortcut resistance, modality necessity, and evidence necessity.
outputs, and VLM descriptions. The entity schema covers people, organizations, demographic groups,
locations, dates, numerical values, metrics, and other domain-specific concepts. We normalize
entity mentions and apply fuzzy matching to identify recurring entities across evidence units. These
recurring entities serve as semantic anchors for evidence-grounded question composition, and their
page and modality distributions allow us to construct single-page, cross-page, and cross-modal
instances.
Figure 2: Domain distribution of the 201 documents used to construct TOPOGRAPHRAG-BENCH.
Slices show the percentage of documents in each domain, and the legend reports document counts.
Single-hop retrieval construction.Single-hop questions are constructed from individual text,
figure, or table units to evaluate local evidence grounding without compositional dependency. The
gold evidence set is restricted to the source unit, and the answer must be recoverable without
consulting any other layout in the document.
Bridge-chain construction.Bridge-chain questions are constructed by composing two atomic QA
pairs through a shared entity. The bridge QA asks for the shared entity indirectly, without mentioning
its surface form, while the target QA explicitly mentions that entity and asks about an attribute, value,
trend, or relation grounded in another evidence unit. An LLM then rewrites the target question by
4

Statistic Number
Documents201
- Pages/doc 45.5 / 27 / 318
- Evidence units/doc 335.1 / 207 / 2,423
Total Questions2,024
- Single-hop 302 (14.9%)
- Bridge-chain 1,014 (50.1%)
- Synthesis 708 (35.0%)
Evidence Distribution
- Cross-page 1,596 (78.9%)
- Cross-modal 773 (38.2%)
- Figure/table-required 1,353 (66.8%)
- Gold evidence units 2.3 / 2 / 6
- Gold pages 2.1 / 2 / 6
Evidence Modality
Text-only 671 (33.2%)
Figure-only 334 (16.5%)
Table-only 246 (12.2%)
Cross-modal 773 (38.2%)Single-hop Synthesis
Modality N Category N
Figure-only 122 Text-only 267
Table-only 113 Cross-modal 259
Text-only 67 Figure-only 113
Table-only 69
2-hop Bridge Synthesis Sources
Path N Sources N
text→text 337 3 evidence units 575
text→figure 204 4 evidence units 101
text→table 119 5 evidence units 23
figure→figure 99 6 evidence units 9
table→table 64
figure→text 62
figure→table 54
table→text 45
table→figure 30
Note:Slash-separated values denote average / median /
maximum. Bridge paths are ordered by hop sequence.
Table 1: Dataset statistics and fine-grained evidence topology.
replacing the explicit entity mention with the indirect description from the bridge QA, producing a
fluent multi-hop question with a controlled bridge dependency.
Multi-source synthesis construction.Synthesis questions are constructed to evaluate evidence
aggregation beyond a single bridge chain. Starting from shared entities, we select information-rich
entities that occur in at least five evidence units and extract concrete facts from each occurrence; for
figures and tables, the original image is additionally provided to a VLM to improve numerical and
structural fidelity. After removing near-duplicate facts, we retain entities with at least three distinct
evidence facts and ask an LLM to identify higher-level patterns such as trends, contradictions, causal
implications, group comparisons, and summary-level observations. Each resulting question is paired
with a structured answer that links evidence points to specific layouts and modalities, followed by a
concise synthesis conclusion.
3.2 Quality Assurance
The construction pipeline uses counterfactual validation to ensure that each instance preserves its
intended evidence topology rather than degenerating into an easier retrieval case. Question generation,
answer generation, and quality assessment are decoupled into independent LLM calls. For bridge-
chain candidates, we verify that the bridge entity is not revealed in the composed question, that the
final answer is not identical to the bridge entity, and that the two hops form a coherent dependency.
We then test single-source counterfactuals by providing only the first-hop or second-hop evidence;
instances that remain answerable from either source alone are removed as pseudo-multihop cases.
We also test modality dependence and evidence necessity. For cross-modal questions, we replace the
multimodal context with text-only context and remove cases that remain answerable without figures
or tables. For synthesis questions, we conduct leave-one-out ablations over supporting evidence
points and retain only instances with at least three required evidence points.
After automatic filtering, two PhD-level reviewers with experience in multimodal LLMs and retrieval-
augmented generation conducted a stratified manual audit of 303 questions (15.0% of the benchmark),
including targeted borderline cases. The audit was stratified by reasoning topology and evidence
modality. Reviewers independently assigned binary validity judgments based on answer correct-
ness, evidence grounding, attribution accuracy, modality necessity, and topology consistency, with
topology-specific checks for bridge-chain and synthesis questions. The reviewers achieved substantial
agreement (Cohen’s κ= 0.84 ). Disagreements were resolved through discussion, and audited
instances were revised or removed when their evidence or topology requirement was not valid. After
5

automatic validation and manual quality assurance, the strict benchmark contains 2,024 questions
across 201 documents, with approximately 15% single-hop retrieval, 50% bridge-chain reasoning,
and 35% multi-source synthesis.
4 Task Definition
Given a multimodal document collection, the task is to answer a question by retrieving and reasoning
over layout-grounded evidence units. Each document is represented as a set of evidence units
D={u i}N
i=1, where each unit is associated with a modality mi∈ {text,figure,table} , a page
index, a layout identifier, and a semantic representation derived from text, OCR, captions, or visual
descriptions. For each question q, the benchmark provides a gold answer aq, a set of supporting
evidence unitsE q, and a reasoning typeτ q∈ {single-hop,bridge-chain,synthesis}.
Unlike standard document QA settings, where target evidence is often treated as an unordered set
of relevant passages, this benchmark evaluates whether a system can recover and use the intended
evidence structure. In single-hop questions, the answer is grounded in one evidence unit. In
bridge-chain questions, evidence units form an ordered dependency path: an earlier hop resolves an
intermediate entity or condition that determines which later evidence unit is relevant. In synthesis
questions, multiple evidence units jointly support a higher-level conclusion, and no single source is
sufficient to recover the full answer.
4.1 Retrieval Task and Metrics
The retrieval stage takes a question qand the document’s evidence-unit set Das input, and returns
a ranked set of candidate contexts Rq={r 1, r2, . . . , r k}. The returned contexts may be textual
chunks, graph nodes, entity or relation descriptions, multimodal summaries, or page-level visual
contexts, depending on the retrieval system. The retrieval objective is not merely to find evidence
related to the final answer, but to retrieve the evidence units and intermediate links required by the
intended evidence structure.
For single-hop questions, successful retrieval requires covering the source evidence unit. For bridge-
chain questions, retrieval should cover both the bridge evidence and the target evidence, including the
intermediate bridge entity that links the two hops. For synthesis questions, retrieval should cover the
required evidence points that jointly support the synthesis conclusion. Thus, retrieval is evaluated as
evidence-structure recovery rather than flat relevance ranking.
Retrieval quality is evaluated using RAGAS context precision and context recall [ 4], instantiated with
topology-enriched references. Context precision measures whether relevant contexts are ranked ahead
of irrelevant ones, while context recall measures how much required reference information is covered
by the retrieved contexts. To make these metrics sensitive to compositional evidence requirements,
we expand the reference beyond the final answer: bridge-chain questions include reasoning traces and
bridge entities, while synthesis questions include required evidence points. This enriched reference
penalizes retrieval outputs that recover only final-hop evidence while missing bridge evidence or
partially covering synthesis support. Formal metric definitions are provided in Appendix B.
4.2 Generation Task and Metrics
The generation stage receives the question qand the retrieved contexts Rq, and produces an answer ˆaq.
A correct response must satisfy both answer-level and structure-level requirements: it should match
the gold answer or synthesis conclusion, and it should be grounded in the retrieved evidence while
covering the reasoning steps required by the question. This distinction is important for compositional
document reasoning. A model may produce the correct final answer while omitting the bridge
entity, skipping an intermediate hop, or relying on only one evidence source in a synthesis question.
Conversely, a model may retrieve or mention relevant evidence but fail to combine it into the
correct final answer. The generation task therefore evaluates answer correctness, evidence grounding,
response relevance, and reasoning-step completion.
Generation quality is evaluated using answer accuracy, faithfulness, response relevancy, and step
coverage. Answer accuracy measures agreement between the generated answer and the gold answer.
Faithfulness measures whether generated claims are supported by the retrieved contexts, and response
6

System Retrieval Generation
Context Precision Context Recall Answer Accuracy Faithfulness Response Relevancy Step Coverage
LightRAG 0.6952 0.6353 0.4045 0.7317 0.7269 0.6031
HippoRAG 0.3834 0.4353 0.3388 0.6517 0.5924 0.3952
Microsoft GraphRAG 0.6018 0.5839 0.3111 0.6802 0.6754 0.5402
VisRAG – – 0.3459 – 0.7688 0.4337
RAG-Anything0.8093 0.72150.5300 0.8346 0.7808 0.7010
MegaRAG 0.7125 0.66320.5350 0.9930 0.8172 0.7329
Table 2: Overall retrieval and generation performance on 2,024 questions. Retrieval metrics are RA-
GAS context precision and context recall. Step coverage is computed over bridge-chain and synthesis
questions only. VisRAG is omitted from the text-based retrieval evaluation, and its faithfulness is
not reported because its retrieved context is page-level visual input rather than textual or structured
evidence strings. Approximate 95% confidence intervals are reported in Appendix C.
relevancy measures whether the response addresses the question. Step coverage evaluates whether the
response covers the reasoning steps specified by the question structure. For bridge-chain questions,
this includes identifying the bridge entity and conveying the corresponding target-hop result. For
synthesis questions, this includes covering the required evidence points that support the intended
synthesis conclusion. Formal metric definitions and the judging protocol are provided in Appendix B.
5 Experiments
5.1 Experimental Setup
Evaluated systems.We evaluate six retrieval-augmented systems grouped by how they represent
document evidence. The text-only GraphRAG baselines, LightRAG, HippoRAG, and Microsoft
GraphRAG [ 5,7,3], construct graphs from textual chunks, entities, relations, or summaries, testing
how far text-derived graph retrieval can support layout-grounded multimodal reasoning. VisRAG
[29] represents page-level visual retrieval and tests whether direct access to document page images
is sufficient for layout-rich questions. The multimodal GraphRAG systems, RAG-Anything and
MegaRAG [ 6,10], incorporate visual, tabular, and textual evidence into graph-structured retrieval.
Together, these systems compare text-derived graphs, page-level visual retrieval, and multimodal
graph structures under the same benchmark topologies.
Model configuration.We fix the model backends within each system family. Text-only GraphRAG
systems use Qwen3-30B-A3B-Instruct with Qwen3-Embedding-8B. VisRAG uses Qwen3-VL-30B-
A3B-Instruct for generation and VisRAG-Ret for page-image retrieval. Multimodal GraphRAG
systems use Qwen3-VL-30B-A3B-Instruct with Qwen3-VL-Embedding-8B. All LLM-based eval-
uation metrics use Qwen3.5-35B-A3B as the judge. Detailed indexing, retrieval, and backend
configurations are provided in Appendix A. Cross-judge analyses and a human spot-check of step
coverage are reported in Appendix B.2.
Evaluation organization.We first report aggregate retrieval and generation results, then analyze
performance by reasoning topology, ordered bridge path, and synthesis composition. Additional
uncertainty estimates and complete fine-grained breakdowns are provided in Appendix C.
5.2 Main Results
Table 2 reports aggregate retrieval and generation performance. Multimodal GraphRAG systems
achieve the strongest overall generation results. MegaRAG obtains the highest point estimates for
answer accuracy, faithfulness, response relevancy, and step coverage, while RAG-Anything achieves
the best retrieval scores and comparable answer accuracy. Among text-only GraphRAG baselines,
LightRAG performs best overall, but remains below the multimodal GraphRAG systems on all
generation metrics.
Retrieval and generation do not produce identical system rankings. RAG-Anything has the highest
context precision and context recall, whereas MegaRAG achieves stronger faithfulness, response
relevancy, and step coverage. This gap suggests that layout-grounded multimodal reasoning depends
7

Figure 3: Capability profile across reasoning topology and evidence modality. Dashed lines denote
text-only GraphRAG systems, while solid lines denote visual or multimodal retrieval systems.
Bridge-MM aggregates mixed-modality and visual-only bridge-chain questions, and Synthesis-MM
aggregates figure-only, table-only, and cross-modal synthesis questions.
not only on retrieving relevant evidence, but also on aligning and composing the retrieved evidence
according to the intended topology.
VisRAG highlights a complementary limitation of page-level visual retrieval. Although it provides
direct access to page images, its answer accuracy and step coverage remain substantially below those
of multimodal GraphRAG systems. This suggests that page-level visual context alone is insufficient
for reliable reasoning over fine-grained layout units and cross-modal evidence dependencies. We next
analyze these trends by reasoning topology, bridge path, and synthesis composition.
5.3 Topology and Modality
Figure 3 shows answer accuracy across topology- and modality-conditioned subsets. Performance
generally declines as questions move from local grounding to structured multimodal reasoning.
Text-only systems can perform strongly when the required evidence is purely textual: HippoRAG
reaches 0.90 accuracy on single-hop text questions. However, the same systems degrade sharply
on multimodal bridge-chain questions. HippoRAG drops from 0.90 on single-hop text to 0.17 on
Bridge-MM, while LightRAG drops from 0.71 to 0.30. This suggests that strong text-centered graph
retrieval does not translate into robust recovery of cross-modal evidence dependencies.
VisRAG performs competitively on single-hop visual questions, achieving 0.65 on figure questions
and 0.56 on table questions. However, its accuracy falls to 0.26 on Bridge-MM and 0.25 on Synthesis-
MM, indicating that page-level visual retrieval alone does not reliably recover fine-grained evidence
topology. In contrast, multimodal GraphRAG systems show a more stable profile on compositional
multimodal subsets. RAG-Anything reaches 0.48 on Bridge-MM and 0.43 on Synthesis-MM, while
MegaRAG reaches 0.53 and 0.43, respectively. Additional retrieval-side results by reasoning topology
are reported in Appendix C. Overall, these results indicate that multimodal document reasoning
requires not only access to visual evidence, but also a mechanism for connecting heterogeneous
evidence units.
5.4 Bridge Paths
Figure 4 analyzes bridge-chain questions from two views: path-level answer accuracy and family-
level reasoning metrics. The radar plot shows that bridge-chain difficulty depends strongly on the
ordered modality sequence. Text →text paths are relatively easier: LightRAG, HippoRAG, RAG-
Anything, and MegaRAG achieve 57.6%, 49.5%, 63.3%, and 55.8% accuracy, respectively. In
contrast, paths requiring visual or tabular bridge grounding are much harder for text-only GraphRAG
systems. LightRAG drops to 12.0% on figure →table and 7.5% on table →figure, while HippoRAG
and Microsoft GraphRAG remain below 10% on both paths.
8

Answer Accuracy (%)
Family LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
Text-only 57.6 49.5 40.1 39.063.355.8
Mixed 36.6 23.5 23.1 26.9 44.250.4
Visual-only 18.4 7.4 11.9 24.3 53.857.8
Faithfulness (%)
Family LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
Text-only 77.9 75.2 75.2 – 92.498.6
Mixed 74.3 70.3 72.0 – 85.598.9
Visual-only 70.2 60.9 71.9 – 77.799.4
Step Coverage (%)
Family LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
Text-only 84.6 65.4 75.2 59.887.885.2
Mixed 65.1 44.5 57.2 49.8 72.679.9
Visual-only 44.9 23.9 40.1 49.4 72.183.0
Figure 4: Bridge-chain reasoning across ordered evidence paths and metric families. The radar plot
shows path-level answer accuracy. The tables summarize answer accuracy, faithfulness, and step
coverage by bridge evidence family. Faithfulness for VisRAG is omitted because its retrieved context
is represented as page-level visual input rather than textual evidence strings.
The summary tables show that this degradation is not limited to final-answer accuracy. On text →text
paths, LightRAG obtains 84.6% step coverage, but drops to 44.9% on visual-only paths. HippoRAG
declines more sharply, from 65.4% to 23.9%. This indicates that text-only GraphRAG systems often
fail to recover or express the required hop structure when the bridge dependency is grounded in
figures or tables.
VisRAG improves access to visual page content, but does not solve bridge dependency recovery. It
reaches 32.9% accuracy on figure →figure and 26.6% on table →table, but only 9.3% on figure →table
and 18.3% on table →figure. Its step coverage also remains below 50% on mixed and visual-only
bridge families. These results suggest that page-level visual retrieval can expose relevant visual
evidence, but does not reliably identify the specific layout unit linked by the bridge entity.
Multimodal GraphRAG systems are more robust. RAG-Anything and MegaRAG reach 51.8%
accuracy on figure →table, and MegaRAG reaches 65.0% on table →figure. At the family level,
MegaRAG achieves 83.0% step coverage on visual-only bridge paths, compared with 44.9% for
LightRAG and 49.4% for VisRAG. These gains indicate that multimodal graph structures help
preserve dependencies between heterogeneous evidence units. However, the remaining variation
across ordered paths shows that modality alignment remains a central bottleneck for bridge-chain
reasoning. Complete path-level accuracy and step-coverage results are reported in Appendix C.
5.5 Multi-Source Synthesis
Table 3 reports synthesis accuracy by evidence composition. Synthesis questions supported only by
textual evidence are consistently easier than those requiring visual, tabular, or cross-modal evidence.
For example, LightRAG reaches 46.6% accuracy on text-only synthesis but drops to 27.4% on
cross-modal synthesis, while HippoRAG drops from 44.4% to 18.4%. This shows that synthesis
difficulty is not only a function of aggregating multiple facts; it also depends on whether those facts
are grounded in heterogeneous document units.
Composition LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
Text-only 46.6 44.4 45.9 34.154.551.0
Figure-only 36.3 29.2 31.9 30.3 46.246.7
Table-only 30.1 22.8 27.5 22.1 34.438.4
Cross-modal 27.4 18.4 27.2 22.743.242.7
Table 3: Answer accuracy on synthesis questions by evidence composition.
9

VisRAG does not close this gap despite direct access to page-level visual inputs. It obtains 30.3%
accuracy on figure-only synthesis, 22.1% on table-only synthesis, and 22.7% on cross-modal synthesis,
suggesting that page-level visual retrieval is insufficient for aggregating multiple fine-grained evidence
points. Multimodal GraphRAG systems are more robust across evidence compositions: RAG-
Anything achieves 43.2% accuracy on cross-modal synthesis, while MegaRAG reaches 38.4%
on table-only and 42.7% on cross-modal synthesis. However, both systems still trail their text-
only synthesis performance, indicating that heterogeneous evidence aggregation remains a central
challenge. A complementary breakdown by synthesis pattern is provided in Appendix C.
6 Conclusion
In this paper, we presented a layout-grounded benchmark for evaluating multimodal evidence
reasoning in GraphRAG. The benchmark contains 2,024 QA pairs across 201 long and layout-
rich documents, with evidence grounded in text, figures, tables, and document layouts. It covers
three controlled evidence topologies: single-hop retrieval, bridge-chain reasoning, and multi-source
synthesis. Through a bottom-up construction pipeline and counterfactual validation, the benchmark is
designed to test whether systems can recover and use the intended evidence structure rather than rely
on shortcut retrieval or single-source evidence. Through evaluations of text-only GraphRAG, page-
level visual retrieval, and multimodal GraphRAG systems, we show that current systems still face
substantial challenges in layout-grounded multimodal reasoning. Multimodal GraphRAG systems
achieve the strongest overall performance, but their gains are conditional on correctly grounding
visual and tabular evidence, aligning evidence across modalities, and composing multiple evidence
units into complete reasoning steps. Text-only GraphRAG remains limited when key dependencies
are grounded outside text, while VisRAG-style page retrieval does not reliably recover fine-grained
evidence topology. These results suggest that a significant gap remains between current multimodal
retrieval systems and the needs of topology-aware document reasoning. We hope this benchmark will
support future work on GraphRAG systems that model not only textual entities and relations, but also
layout structure, modality alignment, and the reasoning roles of evidence units.
10

References
[1]Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu.
Mmdocir: Benchmarking multimodal retrieval for long documents. InProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, pages 30959–30993, 2025.
[2]Kuicai Dong, Yujing Chang, Shijie Huang, Yasheng Wang, Ruiming Tang, and Yong Liu.
Benchmarking retrieval-augmented multimodal generation for document question answering.
arXiv preprint arXiv:2505.16470, 2025.
[3]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
[4]Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. Ragas: Automated
evaluation of retrieval augmented generation. InProceedings of the 18th conference of the
european chapter of the association for computational linguistics: system demonstrations, pages
150–158, 2024.
[5]Zirui Guo, Lianghao Xia, Yanhua Yu, and Chao Huang. Lightrag: Simple and fast retrieval-
augmented generation.
[6]Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. Rag-anything: All-in-one
rag framework.arXiv preprint arXiv:2510.12323, 2025.
[7]Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobio-
logically inspired long-term memory for large language models.Advances in neural information
processing systems, 37:59532–59569, 2024.
[8]Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier
Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph
understanding and question answering.Advances in Neural Information Processing Systems,
37:132876–132907, 2024.
[9]Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a
multi-hop qa dataset for comprehensive evaluation of reasoning steps. InProceedings of the
28th International Conference on Computational Linguistics, pages 6609–6625, 2020.
[10] Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, and Chu-Song Chen.
Megarag: Multimodal knowledge graph-based retrieval augmented generation.arXiv preprint
arXiv:2512.20626, 2025.
[11] António Loison, Quentin Macé, Antoine Edy, Victor Xing, Tom Balough, Gabriel Moreira,
Bo Liu, Manuel Faysse, Céline Hudelot, and Gautier Viaud. Vidore v3: A comprehensive
evaluation of retrieval augmented generation in complex real-world scenarios.arXiv preprint
arXiv:2601.08620, 2026.
[12] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and Shirui Pan.
Gfm-rag: Graph foundation model for retrieval augmented generation. InThe Thirty-ninth
Annual Conference on Neural Information Processing Systems.
[13] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu
Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document
understanding with visualizations.Advances in Neural Information Processing Systems, 37:
95963–96010, 2024.
[14] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on
document images. InProceedings of the IEEE/CVF winter conference on applications of
computer vision, pages 2200–2209, 2021.
[15] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawa-
har. Infographicvqa. InProceedings of the IEEE/CVF Winter Conference on Applications of
Computer Vision, pages 1697–1706, 2022.
11

[16] Jiwon Park, Seohyun Pyeon, Jinwoo Kim, Rina Carines Cabal, Yihao Ding, and Soyeon Caren
Han. Dochop-qa: Towards multi-hop reasoning over multimodal document collections.arXiv
preprint arXiv:2508.15851, 2025.
[17] Xiangyu Peng, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, and Chien-Sheng Wu.
Unidoc-bench: A unified benchmark for document-centric multimodal rag.arXiv preprint
arXiv:2510.03663, 2025.
[18] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. InThe Twelfth
International Conference on Learning Representations, 2024.
[19] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning
of large language model on knowledge graph. InThe Twelfth International Conference on
Learning Representations.
[20] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco,
Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question answering over
text, tables and images. InInternational Conference on Learning Representations, 2021.
[21] Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for
multi-hop queries. InFirst Conference on Language Modeling.
[22] Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. Hierarchical multimodal transformers
for multipage docvqa.Pattern Recognition, 144:109834, 2023.
[23] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique:
Multihop questions via single-hop question composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022. doi: 10.1162/tacl_a_00475.
[24] Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz, Udi Barzelay,
and Leonid Karlinsky. Real-mm-rag: A real-world multi-modal retrieval benchmark. In
Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 31660–31683, 2025.
[25] Biao Xiang, Soyeon Caren Han, and Yihao Ding. Bridge: Benchmark for multi-hop reasoning
in long multimodal documents with grounded evidence.arXiv preprint arXiv:2603.07931,
2026.
[26] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang,
and Jinsong Su. When to use graphs in rag: A comprehensive analysis for graph retrieval-
augmented generation.arXiv preprint arXiv:2506.05690, 2025.
[27] Yuming Yang, Jiang Zhong, Li Jin, Jingwang Huang, Jingpeng Gao, Qing Liu, Yang Bai,
Jingyuan Zhang, Rui Jiang, and Kaiwen Wei. Benchmarking multimodal rag through a chart-
based document question-answering generation framework.arXiv preprint arXiv:2502.14864,
2025.
[28] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 conference on empirical methods in natural language
processing, pages 2369–2380, 2018.
[29] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo
Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on
multi-modality documents. InThe Thirteenth International Conference on Learning Represen-
tations.
[30] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang,
Sicong Liang, Xilin Liu, Yuchi Ma, et al. In-depth analysis of graph-based rag in a unified
framework.arXiv preprint arXiv:2503.04338, 2025.
12

[31] Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua.
Towards complex document understanding by discrete reasoning. InProceedings of the 30th
ACM International Conference on Multimedia, pages 4857–4866, 2022.
[32] Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao,
and Dong Yu. Docbench: A benchmark for evaluating llm-based document reading systems. In
Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural
Language Processing, pages 359–373, 2025.
Appendix Overview
The appendix includes the following sections:
•Appendix A:Reports baseline configuration and efficiency statistics, including indexing
parameters, retrieval settings, runtime, token usage, and graph/index size.
•Appendix B:Provides additional details about evaluation metrics and topology-aware step
coverage.
•Appendix C:Provides additional experimental results, including fine-grained synthesis-
pattern performance.
•Appendix D:Discusses limitations of the benchmark scope, topology coverage, evaluated
systems, and indexing cost.
•Appendix E:Describes license terms, data use, and the relationship between the upstream
MMDocIR assets and the new annotations introduced in this work.
A Baseline Configuration and Efficiency
This appendix reports the baseline configurations and efficiency statistics used in our experiments.
We use each system’s official or framework-default indexing pipeline unless a parameter is explicitly
specified in our evaluation wrapper. All systems are indexed independently at the document level.
Text-only GraphRAG systems operate on original layout-grounded text chunks, VisRAG indexes
page images, and multimodal GraphRAG systems index textual, tabular, and visual layout units
through their native multimodal pipelines.
Table 5 reports the efficiency and graph-size statistics of the evaluated systems. We separate indexing
time from retrieval and generation time because GraphRAG systems often move substantial computa-
tion to the offline indexing stage, where entities, relationships, communities, or multimodal graph
structures are constructed. Runtime, token usage, and graph statistics are reported per document
using the 201-document benchmark denominator. Missing entries indicate that the quantity is either
not applicable to the system design or not recorded in the available artifact.
The efficiency profile highlights different cost-performance tradeoffs across systems. Text-
only GraphRAG systems construct entity-relation graphs from textual chunks, while multimodal
GraphRAG systems incur additional indexing and token costs from processing visual and tabular evi-
dence. RAG-Anything produces the largest graph and highest token usage, reflecting its multimodal
extension over a LightRAG-style backbone. MegaRAG constructs a smaller multimodal graph but
has higher retrieval cost due to its multimodal graph and page-level retrieval process. VisRAG has
low indexing and retrieval cost because it indexes page images without constructing an entity-relation
graph, but this also means it does not expose fine-grained graph statistics.
B Evaluation Metrics and Topology-Aware Judging
B.1 Metric Definitions and Judging Protocol
For each question q, the benchmark provides a gold answer aq, a set of gold evidence units Eq, and
a reasoning topology τq∈ {single,bridge,synthesis} . A system returns an ordered list of retrieved
contextsR q= (r 1, . . . , r K)and a generated answerˆa q.
13

System Input unit Indexing / graph construction Retrieval configuration Model / retriever
LightRAG Text layout chunks Default LightRAG indexing pipeline for construct-
ing a text-derived graph over chunks, entities, and
relationsMixed retrieval mode;
top-k= 3Qwen3-30B-A3B-
Instruct; Qwen3-
Embedding-8B
HippoRAG Text layout chunks Default HippoRAG indexing pipeline over text
chunks and its native retrieval graphDefault retrieval; top-
k= 3Qwen3-30B-A3B-
Instruct; Qwen3-
Embedding-8B
MS-GraphRAG Concatenated text file Default Microsoft GraphRAG indexing workflow
for text-unit chunking, entity and relationship ex-
traction, and community constructionLocal search; top-3 en-
tities and top-3 relation-
ships; max context to-
kens 12000Qwen3-30B-A3B-
Instruct; Qwen3-
Embedding-8B
VisRAG Page images Default VisRAG page-image indexing using
VisRAG-Ret embeddings; no entity-relation graph
is constructed.Cosine page retrieval;
top-k= 3pages.Qwen3-VL-30B-
A3B-Instruct;
VisRAG-Ret
RAG-Anything Multimodal layout
chunksDefault RAG-Anything multimodal indexing
pipeline, with LightRAG as the internal graph
backbone and native processing for text, figures,
and tablesMixed retrieval mode;
top-k= 3Qwen3-VL-30B-
A3B-Instruct;
Qwen3-VL-
Embedding-8B
MegaRAG Page-content JSON Default MegaRAG multimodal knowledge-graph
construction over page-level text, figure, table, and
image representationsMixed retrieval mode;
chunk top- k= 6 ;
reranking disabledQwen3-VL-30B-
A3B-Instruct;
Qwen3-VL-
Embedding-8B
Table 4: Indexing, graph construction, and retrieval configurations for evaluated systems. We use each
system’s default or native indexing and graph-construction pipeline without algorithmic modification,
and standardize only the document input, model backend, and retrieval settings needed for fair
comparison.
Runtime (s/doc) LLM Usage (/doc) Graph / Index Size (/doc)
System Indexing Retrieval Generation Calls Prompt Tok. Completion Tok. Entities Relationships Chunks Communities
LightRAG 2347.73 3.81 8.94 206.43 848415.46 100313.98 847.55 865.26 116.55 –
HippoRAG 117.95 2.47 3.85 552.51 255395.94 56394.44 1686.16 2108.58 266.32 –
MS-GraphRAG 1942.94 50.82 11.90 531.37 909167.87 412309.60 667.77 1004.43 94.24 115.56
RAG-Anything 5844.99 18.74 21.70 668.80 2470006.99 286043.06 1542.71 2786.12 287.04 –
VisRAG 2.94 0.15 22.87 – – – – – – –
MegaRAG 3241.78 335.84 33.57 15.20 230093.20 5026.03 804.59 1142.22 45.55 –
Table 5: Efficiency and index statistics for evaluated RAG systems. Runtime, token usage, and
graph/index sizes are normalized by the 201-document benchmark denominator. “–” denotes not
applicable or not recorded in the available artifact. VisRAG is a visual page retrieval baseline and
does not construct an entity-relation graph.
Topology-enriched reference.Standard RAG evaluation usually compares retrieved contexts
against the final answer alone. In our benchmark, the reference also includes the reasoning structure
required by the question. We therefore define a topology-enriched referencey qas
yq=

(aq,Eq), τ q=single,
(aq,Eq, bq, πq), τ q=bridge,
(aq,Eq, Pq), τ q=synthesis,
where bqis the bridge entity, πqis the ordered bridge reasoning trace, and Pqis the set of required
synthesis evidence points. This reference is used for retrieval evaluation and for topology-aware
judging.
Context precision.Following the RAGAS context precision metric, we evaluate whether relevant
contexts are ranked before irrelevant ones. Let vk= Rel(r k, q, y q)∈ {0,1} indicate whether
retrieved contextr kis relevant to the topology-enriched reference. Precision at rankkis
Pk=1
kkX
i=1vi.
14

The context precision score is
CP(q) =(PK
k=1Pk·vkPK
k=1vk,PK
k=1vk>0,
0,otherwise.
Context recall.Following RAGAS context recall, we decompose the topology-enriched reference
yqinto atomic reference claims Cq. Let uc= Supp(c,R q)∈ {0,1} indicate whether claim cis
supported by the retrieved contexts. Context recall is
CR(q) =1
|Cq|X
c∈Cquc.
For bridge-chain questions, Cqincludes claims about both the bridge evidence and the target evidence.
For synthesis questions,C qincludes the required evidence points and the synthesis conclusion.
Answer accuracy.Answer accuracy measures agreement between ˆaqandaq. We use a normalized
LLM-as-judge score
Acc(q) =J acc(q,ˆa q, aq)∈[0,1],
where the judge assigns full credit to semantically equivalent answers, zero credit to incorrect or
irrelevant answers, and partial credit when a response captures only part of a multi-evidence answer.
Direct single-hop answers are judged primarily by semantic equivalence to the gold answer, while
bridge and synthesis answers are judged against the full gold conclusion.
Faithfulness.Following the RAGAS faithfulness metric, we decompose the generated answer ˆaq
into atomic response claims Dq. Letwd= Supp(d,R q)∈ {0,1} indicate whether response claim d
is supported by the retrieved context. Faithfulness is
Faith(q) =1
|Dq|X
d∈Dqwd.
This metric measures grounding in the retrieved evidence rather than agreement with the gold answer.
Response relevancy.Following the RAGAS response relevancy metric, the judge generates N
question variants {gi(ˆaq)}N
i=1from the response and compares them with the original question by
embedding similarity. Let Eobe the embedding of the original question and Egibe the embedding
of thei-th generated question. The response relevancy score is
RelAns(q) =1
NNX
i=1Egi·Eo
∥Egi∥∥E o∥.
Intuitively, a response is relevant when the original question can be reconstructed from the answer
content.
Topology-aware step coverage.RAGAS metrics evaluate retrieval quality, grounding, and answer
relevance, but they do not directly score whether the response covers the reasoning steps specified by
the question structure. We therefore define a required step setS q:
Sq={sbridge , starget}, τ q=bridge,
{sp:p∈P q}, τ q=synthesis.
For bridge-chain questions, the required steps correspond to the bridge and target hop results. For
synthesis questions, they correspond to the required evidence points; the synthesis conclusion is
evaluated by answer accuracy rather than counted as an additional step. Because single-hop questions
do not contain an intermediate or multi-source reasoning structure, step coverage is reported only for
bridge-chain and synthesis questions.
For each bridge-chain question, the judge receives four inputs: the question, the gold answer, the
annotated per-hop reasoning trace, and the system response. Each hop in the trace specifies a
sub-question, its gold result, and the source modality. The judge compares the response with the
15

annotated trace and marks a hop as covered when the response clearly conveys the corresponding
gold result, allowing for differences in wording. Judgments are based solely on these four inputs.
Synthesis questions follow the same protocol, with the annotated evidence points replacing the
per-hop reasoning trace and a covered/not-covered decision assigned to each required evidence point.
Letzs= 1if the judge marks step or evidence point sas covered, and zs= 0otherwise. Coverage
does not require the response to present an explicitly enumerated reasoning chain. Step coverage is
SC(q) =1
|Sq|X
s∈Sqzs.
Answer accuracy and step coverage jointly assess final-answer correctness and coverage of the re-
quired reasoning content. A response may provide the correct final answer while omitting intermediate
results, or cover some required steps without reaching the correct answer.
Metric applicability.Text-based context precision, context recall, and faithfulness require retrieved
contexts that can be represented as textual or structured evidence strings. VisRAG retrieves page-level
visual inputs rather than fine-grained textual evidence units, so its retrieval RAGAS scores and
text-based faithfulness are not directly comparable to the GraphRAG systems and are not reported.
B.2 Reliability of Step Coverage Evaluation
We assess the reliability of step coverage using three judges from distinct model families: Qwen3.5-
35B-A3B, GLM-5.1, and GPT-5.1. All six systems’ responses are held fixed, and each judge applies
the scoring protocol defined above at temperature 0. The analysis covers 1,722 multi-step questions,
comprising 1,014 bridge-chain and 708 synthesis questions. The remaining 302 single-hop questions
are excluded.
Sensitivity to judge choice.Table 6 reports mean step coverage under each judge. Although
absolute scores vary, MegaRAG and RAG-Anything consistently achieve the highest and second-
highest scores, respectively. Their advantage over the text-only GraphRAG and page-level visual
retrieval baselines therefore persists across evaluator families.
Table 6: Mean step coverage under three model judges, evaluated on fixed system responses. Qwen3.5
denotes Qwen3.5-35B-A3B.
System Qwen3.5 GLM-5.1 GPT-5.1
LightRAG 0.6031 0.6121 0.6743
HippoRAG 0.3952 0.4153 0.4858
Microsoft GraphRAG 0.5402 0.5721 0.6029
VisRAG 0.4337 0.5024 0.4783
RAG-Anything 0.7010 0.7349 0.7442
MegaRAG 0.7329 0.7723 0.7601
Hop-level cross-judge agreement.We additionally examine agreement on individual binary
decisions. Each evaluation unit consists of one annotated bridge-chain hop or synthesis evidence
point within a fixed system response. For each judge pair, we report the proportion of matching
covered/not-covered decisions and Cohen’sκ, both overall and separately by reasoning topology.
Table 7 shows overall agreement ranging from 0.8512 to 0.8958, with κranging from 0.7054 to
0.7923. Agreement remains high under both topologies: topology-specific agreement ranges from
0.8479 to 0.9028, and κranges from 0.7002 to 0.8039. These results indicate consistency at the level
of individual reasoning steps, complementing the system-level comparison above.
16

Table 7: Pairwise hop-level agreement and Cohen’s κ, macro-averaged with equal weight across the
six systems. Results are reported overall and by reasoning topology.
Overall Bridge-chain Synthesis
Judge pair AgreementκAgreementκAgreementκ
Qwen3.5 / GLM-5.1 0.8592 0.7208 0.8663 0.7328 0.8529 0.7105
Qwen3.5 / GPT-5.1 0.8512 0.7054 0.8479 0.7002 0.8541 0.7103
GLM-5.1 / GPT-5.1 0.8958 0.7923 0.9028 0.8039 0.8896 0.7814
Human spot-check.We manually annotated a stratified sample of 100 evaluation units, consisting
of 50 bridge-chain hops and 50 synthesis evidence points. The sample covered all six systems and
included cases where the model judges agreed and cases where they disagreed. Each unit received a
binary covered/not-covered label using the same criterion applied by the model judges.
Table 8 compares the human labels with each judge’s decisions on the same sample. Agreement
ranges from 0.83 to 0.91, with Cohen’s κranging from 0.6561 to 0.8190. These results provide
complementary evidence of alignment between model-based step-coverage judgments and human
annotation.
Table 8: Agreement between human annotations and model judges on 100 evaluation units: 50
bridge-chain hops and 50 synthesis evidence points.
Comparison Agreement Cohen’sκ
Human / Qwen3.5 0.8300 0.6561
Human / GLM-5.1 0.8800 0.7580
Human / GPT-5.1 0.9100 0.8190
C Additional Experimental Results
This appendix provides additional fine-grained experimental results that complement the main
analysis. These results are not required for the main comparison across system families, but they help
characterize where different reasoning patterns remain challenging.
C.1 Overall Metric Uncertainty
Table 9 reports approximate 95% confidence intervals for the overall metrics in Table 2. Intervals are
computed as mean ±1.96 standard errors over non-missing per-question scores. Step coverage is
defined for bridge-chain and synthesis questions; the single-hop subset is excluded from this metric,
so the overall step-coverage intervals use the 1,722 non-single-hop questions.
System Context Precision Context Recall Answer Accuracy Faithfulness Response Relevancy Step Coverage
LightRAG 0.695 [0.675, 0.715] 0.635 [0.619, 0.652] 0.405 [0.389, 0.420] 0.732 [0.720, 0.744] 0.727 [0.715, 0.739] 0.603 [0.585, 0.621]
HippoRAG 0.383 [0.362, 0.405] 0.435 [0.419, 0.452] 0.339 [0.322, 0.356] 0.652 [0.637, 0.666] 0.592 [0.576, 0.609] 0.395 [0.377, 0.413]
Microsoft GraphRAG 0.602 [0.580, 0.623] 0.584 [0.567, 0.601] 0.311 [0.298, 0.324] 0.680 [0.669, 0.691] 0.675 [0.662, 0.689] 0.540 [0.522, 0.558]
VisRAG – – 0.346 [0.330, 0.362] – 0.769 [0.759, 0.779] 0.434 [0.416, 0.451]
RAG-Anything0.809[0.792, 0.826]0.722[0.707, 0.736] 0.530 [0.515, 0.545] 0.835 [0.826, 0.843] 0.781 [0.772, 0.790] 0.701 [0.685, 0.717]
MegaRAG 0.713 [0.693, 0.732] 0.663 [0.647, 0.679]0.535[0.521, 0.549]0.993[0.991, 0.995]0.817[0.811, 0.823]0.733[0.718, 0.748]
Table 9: Approximate 95% confidence intervals for the overall metrics in Table 2. Step coverage is
computed over bridge-chain and synthesis questions only.
C.2 Retrieval Performance by Reasoning Topology
Table 10 reports retrieval performance by reasoning topology. RAG-Anything obtains the strongest
retrieval scores on bridge-chain and synthesis questions, while MegaRAG is strongest on single-hop
questions. The contrast with the generation results in Table 2 suggests that retrieval coverage alone
does not determine final answer quality; systems must also compose the retrieved evidence according
to the intended topology.
17

System Single-hop Bridge-chain Synthesis
Context Precision Context Recall Context Precision Context Recall Context Precision Context Recall
LightRAG 0.765 0.724 0.749 0.700 0.589 0.505
HippoRAG 0.702 0.711 0.449 0.500 0.154 0.226
Microsoft GraphRAG 0.729 0.707 0.618 0.617 0.524 0.484
RAG-Anything 0.848 0.8140.881 0.793 0.691 0.580
MegaRAG0.911 0.9060.740 0.768 0.589 0.410
Table 10: Retrieval performance by reasoning topology. VisRAG is omitted because its retrieved
context is page-level visual input rather than textual or structured evidence strings.
C.3 Complete Bridge-Path Results
Table 11 reports answer accuracy for all ordered bridge paths. The full breakdown shows that visual
and tabular paths are not uniformly difficult: paths such as figure →text and table →text are easier
than figure →table or table →figure, where both hops require non-textual grounding or cross-modal
alignment.
Bridge Path Number LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
text→text 337 57.6 49.5 40.1 39.063.355.8
text→figure 204 30.9 17.5 17.9 27.5 42.351.5
text→table 119 37.2 21.9 25.4 16.0 27.139.5
figure→text 62 49.6 39.9 30.2 38.371.060.1
figure→figure 99 25.5 10.1 16.4 32.9 58.363.4
figure→table 54 12.0 2.8 8.8 9.351.8 51.8
table→text 45 43.3 32.2 30.6 37.261.761.1
table→figure 30 7.5 2.5 1.7 18.3 50.065.0
table→table 64 18.0 9.4 12.5 26.6 50.450.8
Table 11: Answer accuracy (%) on bridge-chain questions by ordered evidence path.
Table 12 provides the corresponding step-coverage breakdown. Compared with final-answer accuracy,
step coverage more directly reflects whether the model recovers both bridge and target evidence roles.
Bridge Path Number LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
text→text 337 84.6 65.4 75.2 59.887.885.2
text→figure 204 62.5 40.0 53.4 50.5 68.480.6
text→table 119 62.6 44.5 58.4 40.8 64.771.4
figure→text 62 71.0 54.8 64.5 58.191.188.7
figure→figure 99 51.0 25.2 40.4 60.1 76.887.9
figure→table 54 35.2 10.2 35.2 29.6 68.580.6
table→text 45 75.6 51.1 61.1 58.986.7 86.7
table→figure 30 43.3 26.7 35.0 48.3 73.383.3
table→table 64 44.5 32.0 46.1 50.0 67.277.3
Table 12: Step coverage (%) on bridge-chain questions by ordered evidence path.
C.4 Synthesis Pattern Breakdown
Table 13 reports answer accuracy on synthesis questions by synthesis pattern. The results show that
synthesis difficulty varies not only by evidence modality, but also by the type of relation that must
be preserved across sources. Comparison and contradiction-style questions are often challenging
because they require the model to maintain distinctions among multiple evidence points, whereas
trend and causal questions can be easier when the relevant pattern is stated more explicitly in the
evidence.
18

Pattern Number LightRAG HippoRAG Microsoft GraphRAG VisRAG RAG-Anything MegaRAG
Panorama 354 37.6 29.1 36.7 29.7 48.549.6
Comparison 130 28.7 19.8 27.5 18.339.234.8
Contradiction 88 34.7 34.7 29.8 28.145.240.3
Trend 73 44.5 38.7 41.4 32.250.748.6
Causal 63 37.7 43.7 41.3 34.954.0 54.0
Table 13: Answer accuracy (%) on synthesis questions by synthesis pattern.
C.5 Cross-Corpus Generalization
To assess whether our construction methodology and main empirical findings extend beyond MMDo-
cIR, we construct an additional evaluation set from UniDoc-Bench [ 17]. This extension is evaluated
separately from the 2,024-question main benchmark.
Extension set construction.We apply the same construction pipeline, prompts, acceptance thresh-
olds, and validation stages to 60 financial and legal documents from UniDoc-Bench, without corpus-
specific modifications. Filename and content-hash checks identify no overlap with MMDocIR. The
pipeline produces 5,134 atomic questions and 2,137 bridge-chain candidates. After validation and
curation to match the main benchmark’s topology distribution, we retain 473 questions: 72 single-hop,
236 bridge-chain, and 165 synthesis questions. The retained instances preserve the same evidence-
topology annotations and satisfy the applicable topology-specific validation criteria used for the main
benchmark. Table 14 summarizes the extension, which includes both cross-modal bridge chains and
multi-evidence synthesis.
Table 14: Topology distribution of the 473-question extension constructed from 60 UniDoc-Bench
documents. Left: detailed counts. Right: inner sectors show topology proportions, and outer sectors
show the corresponding breakdowns and counts. T: text; F: figure; pts: evidence points.
Topology Total Breakdown Count
Single-hop 72Text 35
Figure 37
Bridge-chain 236Text→text 82
Text→figure 68
Other paths 86
Synthesis 1653 evidence points 92
4 evidence points 64
Other 9
Total 473
Evaluation setup.We evaluate one representative from each retrieval paradigm: LightRAG for
text-only GraphRAG, VisRAG for page-level visual retrieval, and RAG-Anything for multimodal
GraphRAG. All three systems retain their original configurations and use the same evaluation protocol
as the main benchmark, as detailed in Appendices A and B. The systems are evaluated on all 473
questions. Step coverage is computed only for the 401 bridge-chain and synthesis questions, consistent
with its applicability in the main experiments.
19

Table 15: Performance on the UniDoc-Bench extension. Step coverage is computed over bridge-chain
and synthesis questions only. Text-based context precision, context recall, and faithfulness are not
reported for VisRAG because its retrieved contexts are page images. Bold values indicate the highest
score among the three evaluated systems.
SystemContext
PrecisionContext
RecallAnswer
Accuracy FaithfulnessResponse
RelevancyStep
Coverage
LightRAG 0.7324 0.6518 0.5356 0.7871 0.6942 0.5716
VisRAG – – 0.3800 – 0.7133 0.5021
RAG-Anything0.8473 0.7259 0.5842 0.8847 0.8159 0.7632
Results.Table 15 shows that RAG-Anything achieves the highest answer accuracy and step cover-
age among the three systems. Its answer accuracy is 0.5842, compared with 0.5356 for LightRAG and
0.3800 for VisRAG. Its step coverage reaches 0.7632, compared with 0.5716 and 0.5021, respectively.
Both metrics therefore reproduce the ordering observed on MMDocIR: RAG-Anything outperforms
LightRAG, which in turn outperforms VisRAG.
With the construction procedure, system configurations, and evaluation protocol held fixed across
corpora, these results support the applicability of our methodology beyond MMDocIR. They also
show that the advantage of multimodal GraphRAG in final-answer correctness and coverage of
required reasoning content persists on the extension set for the three representative systems evaluated
here.
D Limitations
This work has several limitations. First, TOPOGRAPHRAG-BENCHis built on a sampled subset of
MMDocIR, so its document domains, layout styles, and evidence distributions are bounded by the
source corpus. Future work could extend the benchmark to broader document collections, including
scientific papers, reports, manuals, and real world enterprise documents.
Second, the benchmark focuses on three controlled evidence topologies: single hop retrieval, bridge
chain reasoning, and multi source synthesis. These categories enable targeted evaluation of evidence
structure recovery, but they do not cover all possible forms of document reasoning or longer depen-
dency chains. Future work could expand the topology space to support more diverse and realistic
reasoning patterns.
Third, the evaluated systems represent major retrieval paradigms, including text only GraphRAG,
page level visual retrieval, and multimodal GraphRAG, but they do not exhaust all possible parsers,
retrievers, graph construction strategies, or generation backends. Our experiments focus on comparing
retrieval and indexing paradigms under controlled generation and embedding configurations, rather
than systematically assessing sensitivity to backbone choice. In addition, GraphRAG systems can
incur substantial offline indexing costs because entities, relations, communities, or multimodal graph
structures must be constructed before retrieval. Future work could study additional system designs,
quality cost tradeoffs, and incremental indexing strategies for large scale multimodal GraphRAG
deployment.
E License and Data Use
Our benchmark is built on a sampled subset of the MMDocIR evaluation dataset [ 1]. We use 201
documents and their associated layout-grounded document units as the source corpus for constructing
new questions, evidence-topology annotations, and answers. The original documents, page images,
OCR outputs, layout units, and MMDocIR annotations remain subject to the license and usage terms
of the upstream MMDocIR dataset. We do not claim ownership over these source materials.
The new annotations introduced in this work, including the generated questions, gold answers,
evidence topology labels, bridge entities, synthesis evidence points, and validation metadata, are
released for academic and noncommercial research use. Users of the benchmark are responsible for
complying with the upstream MMDocIR license and any applicable terms for the original documents.
When redistributing or using this benchmark, users should cite both MMDocIR and this work.
20