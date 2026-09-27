# UniK: Universal Knowledge Perception for Digital and Physical AI

**Authors**: Nirmit Desai, Kunal Sawarkar, Aditya Mahakali, Dongkon Lee, Kevin Park, Eric Song

**Published**: 2026-09-21 00:56:36

**PDF URL**: [https://arxiv.org/pdf/2609.23971v1](https://arxiv.org/pdf/2609.23971v1)

## Abstract
Two transformative classes of AI systems are reshaping how organizations operate: \textit{digital AI}, which reasons over enterprise knowledge to power chatbots and agent workflows; and \textit{physical AI}, which learns to control robots and autonomous systems from video, gameplay, and sensor telemetry. Both face the same foundational bottleneck: raw knowledge at scale, spanning heterogeneous modalities, locked in private corpora that existing AI infrastructure cannot access reliably or efficiently. We propose \textit{Universal Knowledge Perception (UniK)} as a common platform for both classes, covering the full knowledge lifecycle (ingestion, enrichment, indexing, retrieval, and continuous evaluation) across modalities from rich text and video to molecular data and sensor telemetry. We present UniK, built on Polymath Retrieval (multi-index fusion over automatically enriched indices) with no task-specific fine-tuning. Across five digital AI domains (medical literature, open-domain QA, chemistry, legal video proceedings, and government open data) UniK combined with an open-source 70-billion-parameter model consistently matches or outperforms frontier proprietary LLMs that are orders of magnitude larger: 76\% RAG accuracy on government data versus 47\% for GPT-5; 77.9\% on medical QA without fine-tuning; topping all open-source chemistry pipelines. We show that the same infrastructure directly addresses the data curation, indexing, and retrieval challenges facing physical AI world model training, where the knowledge problem is harder but structurally identical.

## Full Text


<!-- PDF content starts -->

UniK: Universal Knowledge Perception
for Digital and Physical AI
Nirmit Desai*Kunal Sawarkar Aditya Mahakali Dongkon Lee Kevin Park Eric Song
AIntropy AI
Sep 2026
Abstract
Two transformative classes of AI systems are reshaping how organizations operate:digital AI,
which reasons over enterprise knowledge to power chatbots and agent workflows; andphysical AI,
which learns to control robots and autonomous systems from video, gameplay, and sensor telemetry.
Both face the same foundational bottleneck: raw knowledge at scale, spanning heterogeneous modal-
ities, locked in private corpora that existing AI infrastructure cannot access reliably or efficiently. We
proposeUniversal Knowledge Perception (UniK)as a common platform for both classes, covering
the full knowledge lifecycle (ingestion, enrichment, indexing, retrieval, and continuous evaluation)
across modalities from rich text and video to molecular data and sensor telemetry. We present UniK,
built on Polymath Retrieval (multi-index fusion over automatically enriched indices) with no task-
specific fine-tuning. Across five digital AI domains (medical literature, open-domain QA, chemistry,
legal video proceedings, and government open data) UniK combined with an open-source 70-billion-
parameter model consistently matches or outperforms frontier proprietary LLMs that are orders of
magnitude larger: 76% RAG accuracy on government data versus 47% for GPT-5; 77.9% on med-
ical QA without fine-tuning; topping all open-source chemistry pipelines. We show that the same
infrastructure directly addresses the data curation, indexing, and retrieval challenges facing physical
AI world model training, where the knowledge problem is harder but structurally identical.
1 Introduction
Two distinct classes of AI systems are converging toward a common knowledge perception problem.
Digital AI(large language models powering chatbots and agent workflows) must reason accurately over
private enterprise knowledge bases that were never part of their training data.Physical AI(robots,
autonomous vehicles, and embodied agents) must learn world models from video, gameplay, and sensor
telemetry collected at industrial scale. Both classes share the same core challenge: the knowledge they
need to function is heterogeneous, domain-specific, multi-modal, and far too large to fit in any model’s
context window.
In both cases, the bottleneck is not model intelligence butknowledge perception: the ability to
reliably access and surface the right knowledge at the right time. Perception is distinguished from the
understandingor reasoning layer, which is the province of the language models or world models that
consume retrieved knowledge. LeCun [13] draws a related distinction: perception extracts and structures
information from the world, while understanding requires a predictive world model capable of reasoning
and planning. This paper focuses on the perceptual substrate that any reasoning system depends on. A
stronger retrieval and enrichment platform makes any model more capable on knowledge-intensive tasks,
regardless of the model’s intrinsic reasoning ability. Through experimental results across domains, we
show that the model’s understanding and reasoning is no longer the limiting factor, the perception layer
is.
*Corresponding author:nirmit.desai@aintropy.ai
1
arXiv:2609.23971v1  [cs.AI]  21 Sep 2026

Figure 1 makes this concrete. A paralegal reviewing a zoning dispute must locate specific testimony
across hundreds of hours of city council and legislative video, with every word locked in the video
stream and invisible to any search tool. A robotics engineer curating training data for a manipulation
policy needs the subset of collected gameplay footage that demonstrates a particular skill sequence; no
semantic index exists to retrieve it. Both problems fail at the perception layer, not the model layer.
“What was the council’s deci-
sion on mobility and street ser-
vices at this session?”
“What are the top 3 fea-
tures Customer X needs in
our product to choose us over
competitors?”
“Find all drug interaction risks
across 100M clinical records
that current labeling doesn’t
cover.”
“Retrieve episodes where the
agent executes the objective
under time pressure.”
“Apply physics constraints and
curate these driving simulation
episodes for diversity across
road conditions, weather, and
time of day.”
“Which movement sequences
from this demo exhibit the
highest balance error?”
“Across 800 surgical videos,
find every case where a di-
abetic patient had anesthe-
sia complications and cross-
reference with EHR data.”
“Which patrol footage seg-
ments from the past 7 days
match the incident described
in report #4821?”
Figure 1: The knowledge perception problem across eight enterprise and AI domains. Row 1 (left to
right): city council proceedings, corporate sales meeting, pharmaceutical drug interaction records, and
gameplay training data. Row 2: autonomous driving simulation, humanoid robot demonstration, surgical
video with EHR cross-reference, and body-worn camera footage. In every case, relevant knowledge is
locked in unindexed video streams, unstructured documents, or heterogeneous sensor and telemetry logs.
The bottleneck is knowledge perception, not model capability.
For digital AI, this manifests as thegreat knowledge divide. More than 95% of enterprise AI pilots
fail to deliver return on investment [14], not because the underlying models are incapable, but because
the knowledge that drives enterprise value (regulations, clinical notes, legal proceedings, proprietary
research, government datasets) lives behind firewalls in formats and modalities that public pretraining
never touched. Two paradigms have been proposed to bridge this gap. Fine-tuning adapts an LLM
to domain knowledge directly, but causes catastrophic forgetting [22], is sensitive to training hyper-
parameters, and must be repeated as knowledge evolves. Retrieval-Augmented Generation (RAG) is
architecturally more sound: it retrieves context at inference time rather than baking knowledge into
weights, but it degrades at scale: dense retrieval accuracy falls as corpus size grows [18, 24], off-the-
shelf embeddings fail in specialized domains [5], and context window limits bound how much retrieved
knowledge an LLM can actually use [8, 6].
The mathematical foundations of this failure at scale are well understood and have been studied for
decades [23]. Dense retrieval maps both queries and documents to vectors in a shared embedding space
and relies on geometric proximity as a proxy for semantic relevance. This assumption degrades struc-
turally as corpus size grows through two mechanisms. First, in high-dimensional spaces thehubness
problememerges: a small fraction of document vectors become the nearest neighbors for a dispropor-
tionately large fraction of queries regardless of semantic content, biasing retrieval toward a handful of
“hub” documents and away from the long-tail items most relevant to specialized queries [16, 18]. Sec-
ond, general-purpose embedding models collapse domain-specific vocabulary: a SMILES string and its
systematic chemical name, a medical acronym and the condition it abbreviates, or a legislation refer-
2

ence and its colloquial title may land far apart in an embedding space trained on general web text, even
though they refer to the same entity [5, 24]. The practical consequence is striking: BM25, introduced in
the 1990s, remains competitive with neural retrieval on heterogeneous zero-shot benchmarks today [23],
while LLMs have progressed from near-random to near-human on language understanding tasks within
a single decade. Retrieval, not model intelligence, is the binding constraint for knowledge-intensive
applications, a constraint that manifests across agent memory, enterprise chatbots, scientific literature
mining, and physical AI data pipelines alike.
For physical AI, the same knowledge perception problem appears in a harder form, under physical
and real-time constraints that text retrieval never faces (Section 6.4). A world model must be trained on
episodes that are semantically relevant (similar tasks and object configurations), physically valid (suc-
cessful manipulations or informative failures), distribution-covering (diverse enough to prevent mode
collapse), and sequenced for curriculum learning. These properties cannot be specified by hand-labeled
metadata at the scale of millions of video episodes or billions of telemetry samples. They require the
same operations that digital AI demands of its knowledge infrastructure: automated enrichment, multi-
signal indexing, and efficient cross-index retrieval under task-specific queries—formalized in Section 3
as Polymath Enrichment, Polymath Indexing, and Polymath Retrieval. The difference is modality (ac-
tions, trajectories, and sensor readings instead of text), but the knowledge perception problem is struc-
turally identical.
We proposeUniversal Knowledge Perception (UniK)as a foundational platform that addresses both
classes of AI jointly, spanning the full knowledge lifecycle: discovery, ingestion, enrichment, indexing,
retrieval, and continuous evaluation. We argue that these six operations constitute a common platform
for digital and physical AI alike, and that investing in this platform rather than chasing larger models or
domain-specific adapters is the most tractable path to production-ready AI across the enterprise.
We present UniK and validate it across five digital AI domains spanning text, video, chemical,
and structured data modalities. Strong knowledge retrieval compensates for large differences in model
scale: UniK combined with a the opensource Llama3 70B instruct model (multiple versions tested
across 3.1 and 3.3) consistently matches or outperforms frontier proprietary models across domains
where grounded knowledge is essential. For physical AI, we characterize the knowledge infrastructure
challenge across three use cases (world model training from gameplay and teleoperation, visual encoder
pretraining from egocentric video, and operational intelligence from deployed robot fleets), and show
that the same UniK architecture applies directly, with modality-specific enrichment and indexing as the
primary points of differentiation.
2 Universal Knowledge Perception
2.1 UniK Platform Requirements
We define Universal Knowledge Perception as the capability of an AI system to reliably access and
utilize any relevant knowledge at the time it is needed, across the full diversity of knowledge types that
digital and physical AI systems encounter. Five requirements must hold simultaneously:
•Breadth: operates across modalities (text, video, structured data, molecular/chemical, and sen-
sor/telemetry) without requiring modality-specific pipelines. For digital AI this means a single
platform serving medical literature, legal video, and government datasets; for physical AI it means
the same platform ingesting robot telemetry, gameplay recordings, and egocentric video.
•Scale: maintains high retrieval accuracy as the knowledge corpus grows from thousands to billions
of items. Digital AI corpora span millions of documents; physical AI corpora span billions of
timestamped sensor samples and thousands of hours of video.
•Domain generality: achieves strong performance across domains (medicine, law, chemistry, gov-
ernment, robotics) without task-specific fine-tuning. No domain should require bespoke adapters;
the platform should generalize by construction.
3

•Currency: accommodates continuously evolving knowledge without retraining. Enterprise doc-
uments are updated daily; robot fleet data streams continuously.
•Efficiency: delivers knowledge at latencies compatible with the application: interactive response
times for digital AI agents, near-real-time episode retrieval for physical AI training pipelines.
Meeting all five requirements simultaneously is what makes UniK challenging to build. Any single
existing technique satisfies some but not all: dense retrieval degrades at scale; fine-tuned models go stale;
domain-specific adapters do not generalize; keyword search misses semantic meaning. The platform
must compose multiple techniques in a synergistic way.
2.2 The Knowledge Spectrum
Enterprise knowledge exists on a spectrum of modalities, each with distinct retrieval challenges:
Rich text(reports, papers, regulations, contracts) is the most studied, yet even here domain-specific
terminology creates retrieval failures when vocabulary gaps between query and document are large.
Video(legal proceedings, government meetings, training footage, product demonstrations) encodes
information in the temporal relationship between visual scenes and speech that text transcription alone
cannot capture; accurate retrieval requires multimodal enrichment.
Structured data(government databases, scientific datasets, enterprise records) demands that nat-
ural language queries be translated to structured queries while also leveraging the semantic content of
records.
Molecular and chemical datacombines structured identifiers (SMILES, IUPAC names) with un-
structured literature at scales exceeding 100 million documents, spanning tasks from property prediction
to reaction synthesis.
Sensor and telemetry datafrom deployed physical systems (robots, autonomous vehicles, indus-
trial equipment) is high-frequency, causally structured, and presents a fundamentally different retrieval
problem than text: relevant episodes must be located by temporal and causal relationship to a target be-
havior or failure mode, not by semantic similarity to a text query. This modality is the primary substrate
for physical AI world model training and operational intelligence, making it as central to UniK as text
is for digital AI.
4

Figure 2: The UniK platform: a common knowledge infrastructure for digital and physical AI. Six het-
erogeneous knowledge source modalities (left) feed into five core engine operations (center, numbered
1–5): Polymath Enrichment, Polymath Indexing, Polymath Retrieval, Query Routing, and Continuous
Evaluation, with an auto-didactic feedback loop from Continuous Evaluation back to Enrichment. The
same platform serves both digital AI applications (Enterprise Chat, Domain QA Agents, Multimodal
Agents, Video Intelligence) and physical AI applications (World Model Training, Robot Fleet Intelli-
gence, Visual Encoder Pretraining, Inverse Dynamics Estimation) without structural modification.
3 The UniK Platform
UniK is designed to serve both digital and physical AI with a single unified architecture. For digital AI, it
retrieves relevant documents from large enterprise corpora at inference time. For physical AI, it retrieves
semantically and physically relevant training episodes from large video and telemetry corpora at data-
pipeline time. The operations are identical; the modality adapters differ. Building one platform rather
than two separate systems is a deliberate architectural choice: improvements to enrichment, indexing,
and fusion compound across every downstream use case simultaneously.
The engine is governed by two design principles:no task-specific fine-tuning(the same pipeline
generalizes across domains and modalities without per-domain training) andpolymath by default[20]:
no single indexing or retrieval method reliably dominates at scale, so the engine combines lexical, dense,
and learned-sparse signals.
Multi-modality is widely studied, and yet falls short of addressing the challenges identified above.
We use the termPolymathto indicate a diversity of techniques, to address not only multi-modal inputs
but also distinct domain requirements. Thus,Polymath Retrievalis the engine’s query-time fusion op-
eration: a user query is decomposed, routed to one or more index types, and the resulting ranked lists
are fused via Reciprocal Rank Fusion (RRF). The companion operation isPolymath Indexing: con-
structing multi-type BM25, dense k-NN, and learned-sparse indices leveragingPolymath Enrichment
to augment the metadata fields: keyphrases, named entities, synonyms, topics, and captions extracted au-
tomatically [21]. Polymath Enrichment, Indexing, and Retrieval are the three operations that distinguish
UniK from pipelines that rely on a single retrieval modality or a commodity store.
A fourth distinguishing capability, named here for conceptual completeness and pursued as future
work, isAuto-didactic Retrieval: the automatic optimization of the retrieval engine’s hyperparameters—
query type, embedding dimension, fusion weights, BM25 field boosts, and RRFk—to maximize hit@k
for a given domain and corpus without manual configuration. Current deployments require domain-
specific parameter choices that must be set by practitioners; auto-didactic retrieval learns these from
5

labeled examples or weak supervision signals, analogous to recent work on online hyperparameter tun-
ing for RAG [19]. A parallel line of work automates the choice of pipeline components themselves
rather than their hyperparameters: AutoRAG [10] searches over chunking strategies, retrieval modules,
and reranking configurations to find the best-performing pipeline for a given corpus, which we view
as complementary to online hyperparameter tuning—one selects the pipeline shape, the other tunes it
continuously. We distinguishauto-configuration(one-time per-domain setup at ingestion time) from
auto-tuning(continuous online adaptation as the corpus and query distribution evolve); both are open
engineering challenges whose solution would complete the self-improving knowledge perception loop
shown in Figure 2.
Figure 3 situates these operations in the end-to-end pipeline. Theindexing pathconverts raw data
into Polymath Indices through enrichment, embedding, and multi-type indexing. Thequery pathde-
composes an incoming query, routes it through the Polymath Router to one or more index types, and
aggregates results via Fusion & Ranking. Commodity vector, graph, lexical, and relational stores are
the underlying infrastructure; UniK adds value at the enrichment step and at the query decomposition,
routing, and fusion steps. Neither path requires structural modification when the modality changes from
text to video to sensor telemetry; only the enrichment adapters differ.
This architecture explains why UniK extends naturally to physical AI. The indexing path ingests
robot telemetry episodes or egocentric video the same way it ingests documents: enrichment extracts
action types, object labels, and kinematic features rather than keyphrases and entities. The query path
routes a behavior specification or failure pattern through the same Polymath Router and returns the most
relevant training episodes by the same fusion and ranking mechanism. Only the enrichment vocabulary
differs.
Figure 3: End-to-end UniK pipeline.Index path(top, blue background): raw data flows through
Data Connectors intoPolymath Enrichment & Indexing(UniK, blue)—which applies KeyBERT, YAKE,
spaCy NER, VLMs, and multi-type index construction (BM25, dense, sparse)—and populates the cen-
tralPolymath Indices(vector, graph, lexical, relational).Query path(bottom, green background): a
user query is first enriched byPolymath Enrichment(query expansion, entity linking, intent classifica-
tion), then fed into the Polymath Indices from the left; results exit right intoFusion & Ranking(RRF,
cross-encoder reranking, context assembly), which produces the final grounded response. Commodity
components (gray) require no modification across domains; UniK value-add (blue) lies in enrichment,
multi-type indexing, and fusion.
6

4 Evaluation Across Domains
We evaluate the UniK platform across five domains spanning different modalities, corpus scales, and
task types. In all cases, the same underlying Polymath Retrieval architecture is used with no domain-
specific tuning. The generation model is Llama-3.3-70B-Instruct [7] throughout, allowing us to isolate
the contribution of retrieval quality.
Two distinct evaluation types are used in this section. The medical, open-domain, and chemistry
experiments (Sections 4.1–4.2) use established academic benchmarks: PubMedQA [9], NQ [11], Hot-
potQA [25], SQuAD [17], and the ChemRAG suite [30], all with standardized ground-truth labels and
public baselines. The legal video and government open data experiments (Sections 4.3–4.4) use leader-
boards contributed by AIntropy AI [2, 3]: curated corpora, question sets, and evaluation protocols de-
signed to reflect realistic production query distributions in those domains.
An important observation about this evaluation: all corpora used across all five domains are publicly
accessible. The LocalView and Seattle City Council video transcripts [2], the New Jersey government
datasets [3], the ChemRAG corpus [30], and all academic benchmark corpora are available online. Fron-
tier LLMs in our comparisons may have been pre-trained on data from these sources. The performance
gaps we report are therefore not an artifact of private versus public knowledge; they demonstrate thathow
a corpus is indexed, enriched, and retrieved at inference time dominates whether a model can retrieve
and use knowledge it may theoretically have encountered during training. This is a more fundamental
claim: retrieval quality determines accuracy for knowledge-intensive tasks even when the underlying
information is publicly available.
Table 1 characterizes the hit@10 and p95 latency of five UniK retrieval configurations across three
standard benchmarks. Cross-system fusion of lexical and dense signals (UniK-Realtime) achieves 91.2%
mean hit@10 at under 70 ms p95 latency, within 0.6 percentage points of the highest-accuracy configu-
ration (UniK-Precision) at 4–5×lower latency.
Table 1: Retrieval hit@10 and p95 latency across UniK configurations on HotpotQA (7,405 queries),
NQ (2,837 queries), and PubMedQA (1,000 queries). Configurations differ in fusion strategy; all use
the same enriched Polymath Index. Lexical-only and semantic-only baselines shown for reference.
Highlighted: Best accuracy coupled with low latency.Bold: best result per row.
Configuration HotpotQA NQ PubMedQA Mean p95 lat.
Lexical only (BM25) 91.5% 62.3% 93.0% 82.3% 61 ms
Semantic only (dense) 79.9% 79.3% 93.7% 84.3% 51 ms
UniK-Compact 92.4% 76.8% 96.5% 88.6% 59 ms
UniK-Realtime 95.1% 82.7% 95.7% 91.2% 68 ms
UniK-Precision†95.9% 83.4% 96.2% 91.8%748 ms
†Recommended for offline batch evaluation; p95 latency on NQ driven by large corpus + multi-field query expansion.
4.1 Rich Text and Medical Intelligence
We evaluate on three standard open-domain and medical QA benchmarks: PubMedQA [9] (62,249
biomedical documents), Natural Questions [11] (5M documents), and SQuAD [17] (2,067 documents,
10,570 questions). Table 3 shows retrieval accuracy with and without UniK enrichment. Gains are
largest where the semantic gap between query and document is widest: PubMedQA improves by 7.2
percentage points (hit@1) and NQ by over 10 points (hit@5) relative to the baseline without metadata.
Even on SQuAD, where the baseline is already high at 93.3%, enrichment adds 39 correct retrievals
across the development set.
For end-to-end question answering, enriched retrieval translates directly to downstream accuracy
gains. On PubMedQA, the UniK RAG pipeline achieves 77.9% without fine-tuning, second only to
RankRAG [26] (79.8%, fine-tuned) and ahead of all other non-fine-tuned methods including GPT-3.5 +
7

Table 2: UniK leaderboard: performance vs. frontier proprietary LLMs across all five evaluation do-
mains. UniK uses Llama-3.3-70B-Instruct (open-source, 70B parameters); frontier model sizes are
undisclosed but estimated at 10–100×larger.†Frontier LLMs answer without access to the domain
corpus (parametric memory only); UniK retrieves from the full indexed corpus at inference time.Bold:
best result per row.
Domain Modality Benchmark Metric UniK Best Frontier†∆
Medical Text PubMedQA RAG77.9%71.6% (GPT-3.5)+6.3 pp
Open Domain Text NQ Hit@560.5%50.0% (baseline)+10.5 pp
Chemistry Molecular MolInstruct EM64.5%53.2% (GPT-4o)+11.3 pp
Chemistry Molecular SciBench Score18.6%8.6% (GPT-4o)+10.0 pp
Legal Video Video LocalView RAG85.7%85.7% (GPT-4.1) tied
Legal Video Video Seattle CDP RAG79.3%76.0% (Claude)+3.3 pp
Gov. Data Structured NJ OD (Golden) RAG76.0%47.3% (GPT-5)+29.0 pp
Gov. Data Structured NJ OD (Bronze) RAG64.3%56.0% (Gemini)+8.3 pp
Table 3: Impact of UniK enrichment on retrieval accuracy. Baseline uses the same index structure
without enrichment metadata.
Configuration PubMedQA (hit@1) NQ (hit@5) SQuAD (hit@5)
Hybrid, no metadata 77.3% 49.99% 93.30%
+ existing metadata fields 78.8% 59.49% 93.58%
+ UniK enrichment 82.1% 60.48% 93.68%
Improvement over baseline+4.8 pp+10.5 pp+0.38 pp
RAG (71.6%) and all other fine-tuned baselines (Table 5). The mechanism is direct: medical domain
knowledge (synonyms such as “MI” for myocardial infarction, specialized acronyms, topical phrases)
encoded in UniK enrichment fields closes the vocabulary gap that hobbles off-the-shelf embedding mod-
els in specialized domains.
4.2 Chemistry and Pharmaceutical Intelligence
Chemistry is among the most demanding benchmarks for universal retrieval: the ChemRAG corpus
spans over 100 million documents across PubChem, PubMed, USPTO patents, Semantic Scholar, and
Wikipedia [30]. The benchmark tasks range from knowledge recall (MMLU-Chem) to structured pre-
diction (ChemBench4K) to molecular generation (Mol-Instructions) to quantitative scientific reasoning
(SciBench). No single retrieval strategy consistently dominates across all tasks [30], which is precisely
the setting where Polymath Retrieval adds the most value.
Table 6 shows results from the public ChemRAG leaderboard [1]. UniK with Llama-3.1-70B-
Instruct outperforms GPT-4o with the ChemRAG paper’s own retrieval baseline [30] on two of four
benchmarks, MolInstruct exact match (64.5% vs. 53.2%) and SciBench quantitative accuracy (18.6%
vs. 8.6%), the two most demanding generative and computational tasks. On ChemBench4K, GPT-4o
with domain-tuned retrieval leads (67.3% vs. 58.6%), while UniK with the same 70B model outperforms
the identical model without UniK retrieval by more than 2×on ChemBench4K (58.6% vs. 26.3%) and
MolInstruct (64.5% vs. 49.7%). The results establish UniK as the strongest open-source pipeline on
ChemRAG and demonstrate competitive performance against closed proprietary systems on the hardest
task categories.
8

Table 4: Answer accuracy on open-domain benchmarks: exact match vs. LLM-Judge. Exact match (EM)
significantly understates true accuracy for NQ and HotpotQA because it rejects semantically-equivalent
answers (e.g., “Bill Clinton” vs. “William Jefferson Clinton”). LLM-Judge (lenient, Grade≥2 on a 1–3
scale) recovers 20–26 percentage points by accepting correct paraphrases. PubMedQA is excluded here:
its yes/no/maybe constraint makes EM equal to LLM-Judge.
Dataset EM (strict) LLM-Judge (lenient)
Natural Questions 41.2%71.0%
HotpotQA 48.3%74.3%
LLM-Judge grades: 3 = Excellent, 2 = Acceptable, 1 = Poor. The lenient threshold (Grade≥2) matches human judgement
for open QA; the 20–26 pp EM undercount is attributable entirely to paraphrase mismatch, not to incorrect answers.
Table 5: RAG accuracy on PubMedQA. UniK [21, 20] is the only non-fine-tuned system above 74%.
System Accuracy Fine-tuned?
RankRAG [26] 79.8% Yes
UniK (this work) 77.9% No
AlzheimerRAG [12] 74.0% Yes
RAFT (LLaMA2-7B) [28] 73.3% Yes
GPT-3.5 + RAG [28] 71.6% No
MEDRAG + GPT-4 [29] 70.6% No
LLaMA2-7B + RAG [28] 58.8% No
4.3 Legal Video Intelligence
Legal proceedings, government hearings, and public testimony represent a large and largely untapped
enterprise knowledge modality: video archives where the information is encoded in speech, visual con-
text, and the structured flow of legal argument across thousands of hours of footage. We evaluate UniK
on two corpora: LocalView, spanning over 1,000 hours of local government meeting video from mul-
tiple U.S. states, and Seattle City Council proceedings (Seattle CDP), spanning about 1,200 hours of
council video [2].
The UniK video pipeline extends the core Polymath Retrieval architecture with multimodal en-
richment: Vision Language Model scene understanding and prosodic audio analysis augment transcript-
based indices, while an automatically constructed knowledge graph enables structured entity-relationship
queries across the corpus. Results on curated golden question sets (50 questions per corpus) and large
bronze sets (928–1,000 production queries) are shown in Tables 7 and 8. Figure 4 shows the UniK legal
video intelligence interface answering production queries against the Seattle CDP corpus; each panel
captures a different query type (factual lookup, speaker attribution, and multi-turn legislative context).
UniK leads on Seattle CDP across both evaluation sets (2.46 Golden, 2.38 Bronze), the corpus
where retrieval from∼1,200 hours of indexed video matters most. On LocalView, GPT-4.1 edges ahead
on the Golden set (2.44 vs. 2.33) and ties at the Bronze scale (2.57 vs. 2.57), while UniK uses a model
roughly 20×smaller. All frontier LLMs answer without access to the combined 2,200+ hours of indexed
video across both corpora; they rely on parametric memory or limited context, while UniK retrieves and
grounds answers in the actual proceedings. This comparison is not model against model but a retrieval-
augmented open-source model against a massive proprietary model operating from memory.
4.4 Government and Open Data Intelligence
We evaluate on the New Jersey State Open Data corpus [21], a real-world enterprise knowledge base cov-
ering state government datasets spanning public health, transportation, economics, and environmental
domains across text and tabular modalities. This benchmark isolates the challenge of retrieval-grounded
9

Table 6: ChemRAG benchmark results. UniK uses Llama-3.1-70B-Instruct; all other configurations
use the ChemRAG evaluation protocol. Bold: best within open-source models. ChemBench4K and
MolInstruct report average accuracy and exact match (EM) respectively; SciBench reports score with
tolerance.Highlighted: This paper.Bold: best result per row.
System ChemBench4K MolInstruct EM MMLU-Chem SciBench
GPT-4o + ChemRAG67.3%53.2%73.9%8.6%
o1 + ChemRAG 58.4% 45.0% 85.5% 43.6%
UniK + Llama-3.1-70B 58.6% 64.5% 66.0% 18.6%
Llama-3.1-70B + ChemRAG 26.3% 49.7% 61.1% 13.6%
Llama-3.1-8B + ChemRAG 25.9% 41.1% 52.2% 3.6%
Factual lookup: council vote retrieval from
Seattle CDP’s∼1,200 hours of indexed
video.
Speaker attribution: identifying who said
what across multi-speaker hearings.
Legislative context: multi-hop retrieval
across sessions and amendments.
Figure 4: UniK legal video intelligence in action on the Seattle City Council proceedings corpus. An-
swers are grounded in the indexed transcript, not recalled from model weights.
accuracy: answering questions about specific data points, legislation, or agency decisions requires find-
ing the relevant record in a large heterogeneous corpus, not inferring from general world knowledge.
Table 9 shows results from the public leaderboard [3]. UniK achieves 76% RAG accuracy on the
Golden set versus 47% for GPT-5, 46% for Gemini-3 Flash, and 42% for Claude Sonnet 4.6. The pattern
is consistent across the Bronze set (635 queries), where UniK maintains an 8-percentage-point lead over
the next best frontier model.
The NJ Open Data results make the retrieval argument concrete. Even if an LLM has encountered
snapshots of this public data during pretraining, it cannot reliably cite the specific regulatory table, the
exact agency decision by date, or the current numeric value of a government indicator; parametric knowl-
edge is stale, compressed, and unverifiable. Retrieval from an indexed, enriched corpus gives UniK a
29-percentage-point accuracy advantage on the Golden set precisely because the answer is grounded in
the source document rather than recalled from training weights. UniK’s 76% accuracy reflects current
pipeline coverage on a heterogeneous corpus with inconsistent semi-structured formatting; the accuracy
ceiling is retrieval coverage, not model capability.
10

Table 7: Legal video benchmark – Golden set (50 curated questions per corpus). Avg Grade on a 1–3
scale (3 = Excellent, 1 = Poor; higher is better). RAG Accuracy = avg grade / 3×100. Frontier LLMs
answer without access to the video corpus; UniK retrieves from LocalView (1,000+ hours) and Seattle
CDP (∼1,200 hours) using Llama-3.3-70B-Instruct.
SystemRAG Accuracy↑Avg Grade (3 = best)↑
LocalView Seattle CDP LocalView Seattle CDP
UniK + Llama-3.3-70B77.7%82.0%2.332.46
GPT-4.181.3%74.7%2.442.24
Claude Sonnet 4.6 64.3% 71.3% 1.93 2.14
Gemini 2.0 Flash 63.3% 62.7% 1.90 1.88
Table 8: Legal video benchmark – Bronze set (928–1,000 production queries per corpus). Same grading
scale and RAG Accuracy formula as Table 7.
SystemRAG Accuracy↑Avg Grade (3 = best)↑
LocalView Seattle CDP LocalView Seattle CDP
UniK + Llama-3.3-70B 85.7% 79.3% 2.57 2.38
GPT-4.185.7%73.7%2.572.21
Claude Sonnet 4.6 73.3% 76.0% 2.20 2.28
Gemini 2.0 Flash 64.7% 67.0% 1.94 2.01
Table 9: NJ Open Data benchmark. RAG Accuracy = avg grade / 3×100; Grade 3 = excellent, Grade
1 = poor (higher average grade = better). All frontier LLMs answer without corpus retrieval.
SystemRAG Accuracy↑Avg Grade (3 = best)↑
Golden Bronze Golden Bronze
UniK + Llama-3.3-70B 76.0% 64.3% 2.28 1.93
GPT-5 47.3% 52.3% 1.42 1.57
Gemini-3 Flash 46.0% 56.0% 1.38 1.68
Claude Sonnet 4.6 42.0% 39.7% 1.26 1.19
11

5 The Efficiency Argument
Across all five domains, the information bottleneck is retrieval quality rather than model scale. Table 2
summarizes performance of UniK + Llama-3.3-70B against leading frontier LLMs across all domains
and modalities. In every domain where grounded knowledge is essential (government data, specialized
medical QA, chemical generation, legal video at scale) a 70B open-source model with strong retrieval
matches or exceeds models that are an order of magnitude larger and accessed exclusively via proprietary
APIs.
Enterprises do not need the largest or most expensive proprietary models to achieve frontier-level
performance on knowledge-intensive tasks. Investing in the retrieval and enrichment layer makes a
smaller open-source model competitive, with the added benefits of cost efficiency, data privacy, and
on-premise deployment.
The efficiency gap is largest precisely where the knowledge is most private and domain-specific: NJ
government data, biomedical literature, chemical synthesis. In these settings, a larger model’s richer
parametric knowledge is irrelevant because the knowledge required was never in its training data. What
matters is whether the retrieval layer can surface the right document from the enterprise corpus. This
is the argument for investing in universal knowledge perception infrastructure rather than chasing larger
models.
6 Universal Knowledge Perception for Physical AI
Physical AI (robots, autonomous vehicles, and embodied agents that must act in the real world) requires
knowledge infrastructure that is fundamentally different in modality but structurally identical in opera-
tion to what digital AI requires. This section makes the case that UniK is the right platform for both, and
that the UniK platform provides a concrete path from the text-based results of the preceding sections to
the sensor- and video-based data challenges of physical AI. The benchmark results in Sections 4.1–4.4
demonstrate what UniK achieves for digital AI; the architecture and use cases below describe how the
same platform addresses the analogous problem for physical AI.
6.1 The Knowledge Problem in Physical AI
Physical AI systems (robots, autonomous vehicles, embodied agents) learn by training on large corpora
of interaction data: video, gameplay demonstrations, egocentric observations, and sensor telemetry.
The quality of these systems depends not only on the architecture of the world model or policy but
critically on the quality of the data pipeline that curates the training corpus. This pipeline faces the same
knowledge perception challenge that digital AI does, at larger scale and under harder constraints—
physical validity, coverage-aware sampling, and real-time ingestion among them, enumerated in full in
Section 6.4.
The unifying architecture for physical AI is theworld model: an internal simulator trained to predict
how the environment evolves given observations and actions. World models enable sample-efficient pol-
icy learning and planning without costly real-world rollouts [27, 31]. Leading examples include Meta’s
V-JEPA 2 [4], trained on over one million hours of internet video to achieve zero-shot robot manipula-
tion on unseen hardware, and NVIDIA’s GR00T N1 [15], a humanoid foundation model combining a
vision-language planner with a diffusion-based motor action model. LeCun’s Joint Embedding Predic-
tive Architecture (JEPA) [13] provides a theoretical grounding for this direction: rather than predicting
raw pixels, world models should predict in abstract latent space, learning representations of dynamics
and structure that generalize across embodiments.
The data bottleneck is underappreciated relative to the architectural progress receiving most of the
attention. A world model for robotics needs training episodes that are semantically relevant (similar
tasks and object configurations), physically valid (kinematically consistent trajectories), distribution-
covering (diverse enough to prevent mode collapse in learned dynamics), and appropriately sequenced
12

for curriculum learning. None of these properties can be specified by hand at the scale of millions
of episodes. They require automatic enrichment, Polymath Indexing over multiple signal types, and
efficient Polymath Retrieval under task-specific queries: exactly the operations UniK performs for digital
AI.
6.2 The Common Infrastructure
The UniK architecture applies directly to physical AI data pipelines. Table 10 shows how the six UniK
operations map across digital and physical AI use cases.
Table 10: UniK operations are common across digital and physical AI; modality-specific enrichment is
the principal point of differentiation.
UniK Opera-
tionDigital AI Physical AI
Ingestion Batch documents, streaming
articles, video transcriptsTelemetry streams, video
episodes, gameplay recordings
Enrichment Keyphrases, named entities,
synonyms, topicsObject labels, action types,
kinematic features, failure flags
Indexing BM25 lexical + dense seman-
tic, metadata-boostedSpatial + temporal + visual +
kinematic hybrid indices
Retrieval Polymath Retrieval (RRF fu-
sion over text, vector, and
graph indices)Episode retrieval by behavior
specification or failure pattern
Evaluation Retrieval accuracy on labeled
QA benchmarksCurriculum coverage, training
distribution metrics
Continuous
learningIncremental index updates as
documents changeOnline ingestion of new fleet
episodes and demonstrations
Both digital and physical AI face the same root problem: the knowledge an AI system needs to
function is distributed across a large, heterogeneous, evolving corpus that cannot be fully loaded into
any model’s context or training batch. The solution in both cases is a retrieval infrastructure that enriches
items with metadata, indexes them for efficient lookup, and retrieves the most relevant subset on demand.
UniK is that infrastructure.
6.3 Physical AI Use Cases
Three use cases show how UniK applies to the physical AI data pipeline:
UC1: Paired Video and Action Data (Gameplay / Teleoperation).Robot learning from human
demonstrations requires pairing video observations with action streams (joint angles, end-effector poses,
controller inputs). A semantic retrieval layer must align video and action streams temporally, validate
physical consistency of trajectories, and sample training batches that maximize coverage of the task
distribution. Automated curriculum generation (surfacing progressively harder episodes as the policy
improves) is a direct application of the retrieval and evaluation capabilities in UniK.
UC2: Egocentric Video Without Action Labels.Large quantities of egocentric video (first-person
human demonstrations, wearable cameras) exist without corresponding action labels. Pretraining visual
encoders and bootstrapping latent action spaces from this data requires solving aninverse dynamics
problem: inferring the implicit action or state transition that connects consecutive observations, without
ever observing the action signal directly. This differs from conventional video retrieval (finding a clip by
keyword): it requires identifying clips with meaningful, learnable state transitions—hand-object contact,
manipulation events, navigational transitions—and demands multimodal enrichment (object detection,
13

hand pose estimation, scene classification) combined with temporal index structures that track state
change across a sequence, not just per-frame semantic similarity.
UC3: Robot Fleet Telemetry.Deployed robot fleets generate continuous streams of high-frequency
sensor data: joint torques, force-torque readings, IMU signals, camera feeds, error logs. Operational in-
telligence (identifying failure modes, correlating failures with environmental conditions, retrieving simi-
lar past incidents) requires a retrieval system that can embed temporal signal episodes, detect anomalies,
cluster failure signatures, and surface relevant historical data in response to a new failure event.
6.4 Open Challenges
UniK for physical AI faces challenges beyond the digital case:
•Multi-modal temporal alignment:video, action streams, and sensor telemetry must be aligned
to sub-second precision before indexing; misalignment corrupts the semantic content of the episode.
•Physical validity:unlike text, where correctness is semantic, physical training data must satisfy
kinematic and dynamic constraints; invalid trajectories can harm policy learning even if semanti-
cally similar to valid ones.
•Coverage-aware sampling:world model training is highly sensitive to the distribution of training
data; retrieval must be diversity-aware, not just relevance-aware, to avoid mode collapse in the
learned dynamics.
•Scale and real-time ingestion:deployed robot fleets produce data at rates that require streaming
ingestion, online anomaly detection, and near-real-time indexing, which go beyond the batch
processing assumed by most RAG architectures.
•Evaluation:unlike text QA, there is no simple ground truth for whether a retrieved training
batch will improve a world model; evaluation requires a closed loop with model training, making
benchmark construction for physical AI retrieval significantly harder.
Despite these open problems, the structural case for UniK as the foundation of physical AI is strong.
The operations are the same as for digital AI; the engineering adapters differ. Organizations already
using UniK for enterprise document retrieval can extend the same infrastructure to physical AI data
curation with modality-specific enrichment as the adaptation layer rather than building a separate data
pipeline from scratch. The lesson from digital AI transfers directly: investing in the knowledge infras-
tructure layer compounds across every downstream use case, and physical AI is the next domain where
that investment will prove decisive.
7 Conclusion
Digital AI and physical AI are converging on the same foundational bottleneck: the knowledge they
need to function is heterogeneous, domain-specific, multi-modal, and distributed across corpora that are
too large, too private, and too dynamic for any model to internalize through training alone. Universal
Knowledge Perception is the paradigm we propose to address this bottleneck across both classes, not as
two separate efforts but as a single platform with shared operations and modality-specific adapters.
UniK validates this claim on the digital AI side. Across five domains (medical, open-domain, chem-
istry, legal video, and government data) domain-agnostic Polymath Retrieval combined with an open-
source 70B-parameter model consistently matches or outperforms frontier proprietary models that are
estimated to be orders of magnitude larger. The central finding is that retrieval quality matters more than
model scale for grounded, knowledge-intensive tasks: investing in the knowledge infrastructure layer
compounds across every domain and use case simultaneously.
14

The physical AI extension makes the platform argument concrete. World model training, visual
encoder pretraining from egocentric video, and robot fleet intelligence are all data curation and retrieval
problems. The same six UniK operations (ingestion, enrichment, indexing, retrieval, evaluation, and
continuous learning) apply directly, with modality-specific enrichment as the principal adaptation. The
architectural lesson from digital AI transfers: no single retrieval method dominates, Polymath Retrieval
over enriched indices is the right default, and domain generality requires investing in the platform rather
than in per-task adapters.
Open directions.Several challenges remain open for extending Universal Knowledge Perception
fully into the physical and virtual domains. Auto-didactic retrieval (Section 3)—learning per-domain
retrieval configuration and continuously adapting it online—remains unsolved outside of narrow hyper-
parameter tuning. On the physical AI side, multi-modal temporal alignment, physical validity checking,
and coverage-aware sampling (Section 6.4) require enrichment and indexing techniques with no direct
digital AI analogue, and evaluation itself is an open problem: unlike text QA, there is no simple ground
truth for whether a retrieved training batch improves a world model. Extending UniK to naturalistic,
non-verbal video (bodycam and surveillance streams, where the dominant signal is visual rather than
linguistic) and to action-rich virtual environments (game and simulation data reuse for physical AI train-
ing) are the two directions we view as most immediately tractable, and where we intend to focus future
work.
We view UniK as infrastructure in the same sense that databases were infrastructure for transactional
computing: a layer that every AI application will depend on, that rewards investment because the benefits
compound across use cases, and that will look obvious in retrospect. UniK is our implementation of that
infrastructure, and this paper is the first account of its performance across the breadth of domains it is
designed to serve.
References
[1] AIntropy AI. ChemRAG leaderboard v1.https://huggingface.co/spaces/
aintropy-ai/chemRAG-leaderboard-v1, 2025.
[2] AIntropy AI. Legal videos QA leaderboard v1.https://huggingface.co/spaces/
aintropy-ai/legal-videos-leaderboard-v1, 2025.
[3] AIntropy AI. NJ open data QA leaderboard v1.https://huggingface.co/spaces/
aintropy-ai/nj-open-data-leaderboard-v1, 2025.
[4] Mahmoud Assran, Quentin Duval, Randall Balestriero, Ishan Misra, Piotr Bojanowski, Pascal
Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. V-jepa 2: Self-supervised video models
enable understanding, prediction and planning, 2025.
[5] Scott Barnett, Stefanus Kurniawan, Srikanth Thudumu, Zach Brannelly, and Mohamed Abdel-
razek. Seven failure points when engineering a retrieval augmented generation system, 2024.
[6] Yufeng Du, Minyang Tian, Srikanth Ronanki, Subendhu Rongali, Sravan Bodapati, Aram Gal-
styan, Azton Wells, Roy Schwartz, Eliu A Huerta, and Hao Peng. Context length alone hurts llm
performance despite perfect retrieval, 2025.
[7] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models.arXiv preprint arXiv:2407.21783, 2024.
[8] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang
Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language
models?, 2024.
15

[9] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W. Cohen, and Xinghua Lu. Pubmedqa: A
dataset for biomedical research question answering, 2019.
[10] Dongkyu Kim, Byoungwook Kim, Donggeon Han, and Matou ˇs Eibich. AutoRAG: Automated
framework for optimization of retrieval augmented generation pipeline, 2024.
[11] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N.
Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav
Petrov. Natural questions: a benchmark for question answering research.Transactions of the
Association of Computational Linguistics, 2019.
[12] Aritra Kumar Lahiri and Qinmin Vivian Hu. Alzheimerrag: Multimodal retrieval augmented gen-
eration for pubmed articles.arXiv preprint arXiv:2412.16701, 2024.
[13] Yann LeCun. A path towards autonomous machine intelligence, 2022. Open Review.
[14] MIT Project NANDA. The genai divide: State of ai in business 2025. Tech-
nical report, Massachusetts Institute of Technology, July 2025. Found in:
https://www.legal.io/articles/5719519/MIT-Report-Finds-95-of-AI-Pilots-Fail-to-Deliver-ROI-
Exposing-GenAI-Divide.
[15] NVIDIA. Nvidia isaac gr00t n1: An open foundation model for generalist humanoid robots, 2025.
[16] Milo ˇs Radovanovi ´c, Alexandros Nanopoulos, and Miroslav Ivanovi ´c. Hubs in space: Popular
nearest neighbors in high-dimensional data.Journal of Machine Learning Research, 11:2487–
2531, 2010.
[17] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions
for machine comprehension of text.arXiv preprint arXiv:1606.05250, 2016.
[18] Nils Reimers and Iryna Gurevych. The curse of dense low-dimensional information retrieval for
large index sizes, 2021.
[19] Sherry Ruan et al. AutoRAG-HP: Automatic online hyper-parameter tuning for retrieval-
augmented generation. InFindings of the Association for Computational Linguistics: EMNLP
2024, 2024.
[20] Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki. Blended rag: Improving rag
(retriever-augmented generation) accuracy with semantic search and hybrid query-based retrievers.
InThe 7th IEEE International Conference on Multimedia Information Processing and Retrieval
(IEEE-MIPR 2024). IEEE, 2024.
[21] Kunal Sawarkar, Shivam R. Solanki, and Abhilasha Mangal. Metagen blended rag: Unlocking
zero-shot precision for specialized domain question-answering, 2025.
[22] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana,
and Suranga Nanayakkara. Improving the domain adaptation of retrieval augmented generation
(RAG) models for open domain question answering.Transactions of the Association for Compu-
tational Linguistics, 11:1–17, 2023.
[23] Nandan Thakur, Nils Reimers, Andreas R ¨uckl´e, Abhimanyu Srivastava, and Iryna Gurevych.
BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models.
InThirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks
Track, 2021.
16

[24] Orion Weller, Michael Boratko, Iftekhar Naim, and Jinhyuk Lee. On the theoretical limitations of
embedding-based retrieval, 2026.
[25] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering.arXiv preprint arXiv:1809.09600, 2018.
[26] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and
Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation in llms.
Advances in Neural Information Processing Systems, 37:121156–121184, 2024.
[27] Fanqi Zeng, Bencheng Liang, Jianhua Shi, and Wei Shen. A comprehensive survey on world
models for embodied AI, 2025.
[28] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E
Gonzalez. Raft: Adapting language model to domain specific rag. InFirst Conference on Language
Modeling, 2024.
[29] Xuejiao Zhao, Siyan Liu, Su-Yin Yang, and Chunyan Miao. Medrag: Enhancing retrieval-
augmented generation with knowledge graph-elicited reasoning for healthcare copilot, 2025.
[30] Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu,
and Jiawei Han. Benchmarking retrieval-augmented generation for chemistry.arXiv preprint
arXiv:2505.07671, 2025.
[31] Xiaoyuan Zhou, Haoyuan Xue, Yunbiao Gao, Jiale Huang, Chen Feng, Shangzhe Gao, Yingzi
Cheng, Lin Luo, Jiajun Pan, and Zhengwen Liao. A survey: Learning embodied intelligence from
physical simulators and world models, 2025.
17