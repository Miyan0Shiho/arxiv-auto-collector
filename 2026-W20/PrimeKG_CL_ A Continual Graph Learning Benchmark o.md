# PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs

**Authors**: Yousef A. Radwan, Yao Li, Qing Qing, Ziqi Xu, Xingtong Yu, Jiaxing Huang, Renqiang Luo, Xikun Zhang

**Published**: 2026-05-11 13:14:02

**PDF URL**: [https://arxiv.org/pdf/2605.10529v1](https://arxiv.org/pdf/2605.10529v1)

## Abstract
Biomedical knowledge graphs underwrite drug repurposing and clinical decision support, yet the upstream ontologies they depend on update on independent cycles that add millions of edges and deprecate hundreds of thousands more between releases. Yet existing continual graph learning has been studied almost exclusively on synthetic random splits of static, generic KGs, a regime that cannot reproduce the asynchronous, structured evolution real biomedical KGs undergo. To this end, we introduce PrimeKG-CL, a CGL benchmark built from nine authoritative biomedical databases (129K+ nodes, 8.1M+ edges, 10 node types, 30 relation types) with two genuine temporal snapshots (June 2021, July 2023; 5.83M edges added, 889K removed, 7.21M persistent), 10 entity-type-grouped tasks, multimodal node features, and a per-task persistent/added/removed test stratification. On three tasks (biomedical relationship prediction, entity classification, KGQA), we evaluate six CL strategies across four KGE decoders, plus LKGE, an LLM-RAG agent, and CMKL. We find that decoder choice and continual learning strategy interact strongly: no single strategy performs best across all decoders, and mismatched combinations can significantly degrade performance. Moreover, only DistMult exhibits a clear separation between persistent and deprecated knowledge, indicating that standard metrics conflate retention of still-valid facts with failure to forget outdated ones; this effect is absent under RotatE. In addition, multimodal features improve entity-level tasks by up to 60%, and a recent CKGE framework (IncDE) failed to scale to our 5.67M-triple base task across five attempts up to 350GB RAM. Data, pipeline, baselines, and the stratified split are released openly. Dataset:huggingface.co/datasets/yradwan147/PrimeKGCL|Code:github.com/yradwan147/primekg-cl-neurips2026

## Full Text


<!-- PDF content starts -->

PrimeKG-CL: A Continual Graph Learning
Benchmark
on Evolving Biomedical Knowledge Graphs
Yousef A. Radwan
Technology, Innovation, Entrepreneurship Department
King Abdullah University of Science and Technology
yousef.radwan@kaust.edu.sa
Yao Li
School of Computing and Information Systems
The University of Melbourne
yao.li5@student.unimelb.edu.au
Qing Qing
College of Computer Science and Technology
Jilin University
qingqing25@mails.jlu.edu.cnZiqi Xu
School of Computing Technologies
RMIT University
ziqi.xu@rmit.edu.au
Xingtong Yu
Department of Systems Engineering and Engineering Management
The Chinese University of Hong Kong
xtyu@se.cuhk.edu.hk
Jiaxing Huang
Department of Data Science and Artificial Intelligence
Hong Kong Polytechnic University
jiaxing.huang0508@outlook.com
Renqiang Luo∗
College of Computer Science and Technology
Jilin University
lrenqiang@jlu.edu.cnXikun Zhang∗
School of Computing Technologies
RMIT University
xikun.zhang@rmit.edu.au
Abstract
Biomedical knowledge graphs underwrite drug repurposing and clinical decision
support, yet the upstream ontologies they depend on (GO, HPO, MONDO, CTD)
update on independent cycles that add millions of edges and deprecate hundreds
of thousands more between releases. Yet existing continual graph learning (CGL)
has been studied almost exclusively on synthetic random splits of static, generic
KGs, a regime that cannot reproduce the asynchronous, structured evolution real
biomedical KGs undergo. To this end, we introducePrimeKG-CL, a CGL bench-
mark built from nine authoritative biomedical databases (129K+ nodes, 8.1M+
edges, 10 node types, 30 relation types) with two genuine temporal snapshots ( t0
∗Corresponding authors.
Preprint.arXiv:2605.10529v1  [cs.AI]  11 May 2026

June 2021, t1July 2023; 5.83M edges added, 889K removed, 7.21M persistent),
10 entity-type-grouped tasks, multimodal node features (BiomedBERT text, Mor-
gan fingerprints, R-GCN structural), and a per-task persistent/added/removed test
stratification. On three tasks (biomedical relationship prediction, entity classifi-
cation, KGQA), we evaluate six CL strategies across four KGE decoders, plus
LKGE, an LLM-RAG agent, and CMKL. We find that decoder choice and contin-
ual learning strategy interact strongly: no single strategy performs best across all
decoders, and mismatched combinations can significantly degrade performance
(e.g., EWC improves ComplEx by +167% but Distillation reduces RotatE by
57%). Moreover, only DistMult exhibits a clear separation between persistent and
deprecated knowledge ( ≈11× higher MRR on persistent vs. removed triples),
indicating that standard metrics conflate retention of still-valid facts with failure to
forget outdated ones; this effect is absent under RotatE. In addition, multimodal
features improve entity-level tasks by up to 60%, and a recent CKGE frame-
work (IncDE) failed to scale to our 5.67M-triple base task across five attempts
up to 350 GB RAM. Data, pipeline, baselines, and the stratified split are released
openly. Dataset: https://huggingface.co/datasets/yradwan147/PrimeKGCL | Code:
https://github.com/yradwan147/primekg-cl-neurips2026
1 Introduction
Biomedical knowledge graphs (KGs) underwrite drug discovery, disease understanding, and clinical
decision support [ 1,2], yet they are far from static: upstream ontologies (GO, HPO, MONDO, CTD)
update asynchronously, continuously introducing new entities and deprecating outdated associations.
A model trained on a 2021 snapshot will face substantial new knowledge by 2023, raising the
fundamental question:how can graph-based models continually learn from evolving biomedical KGs
without catastrophically forgetting?
Continual graph learning (CGL) has emerged to address this [ 3]. Methods such as Elastic Weight
Consolidation (EWC) [ 4], Experience Replay [ 5], and frameworks such as LKGE [ 6] mitigate
forgetting in knowledge graph embeddings (KGE). But existing CGL benchmarks construct temporal
tasks viasyntheticrandom splits of static generic KGs (LKGE on FB15k-237/WN18RR; IncDE [ 7]
on the same), missing the real dynamics: asynchronous additions from independent upstream
sources, removals as ontologies are refined, and domain-specific patterns of change (GO annotations
dominate additions; disease-phenotype links churn as HPO is refined) that random partitioning cannot
reproduce.
In this work, we introduce PrimeKG-CL (PrimeKG forContinualLearning), the first biomedical-
scale CGL benchmark with genuine temporal evolution. We reconstruct two snapshots of PrimeKG [ 1]
by re-querying upstream databases: t0(June 2021) and t1(July 2023, rebuilt from nine freely
accessible databases). The t0→t1partition exposes 5.8M added, 889K removed, and 7.2M persistent
edges; the pipeline supports extension to additional snapshots. The benchmark ships with multimodal
node features (BiomedBERT text, Morgan fingerprints [ 8], R-GCN structural), enabling modality-
specific forgetting analysis. All structural baselines (Naive, Joint, EWC, ER, SI [ 9], Distillation [ 10],
MIR [ 11], LKGE) operate on KGE backbones; CMKL integrates all three modalities through MoE
fusion.
Across six CL methods ×four KGE decoders on three tasks, we uncover three key findings that are
not observable in prior synthetic benchmarks.First, decoder and CL strategy interactqualitatively:
no single strategy performs best across all decoders, and mismatched combinations can substantially
degrade performance. For example, EWC improves ComplEx by +167% and slightly benefits RotatE
(+5%), but Distillation reduces RotatE by 57% and replay-based methods drop ComplEx by 38%.
Overall, RotatE [ 12] achieves the strongest performance (Naive AP = 0.084 filtered MRR, 45%
higher than DistMult-Naive’s 0.058 ), and EWC+RotatE is the best configuration ( 0.088 ).Second, the
t0→t1temporal partition reveals a clear separation between persistent and deprecated knowledge
that standard metrics (AP/AF) obscure. Under DistMult, naive drift produces an ≈11× gap between
persistent and removed triples, indicating that the model retains still-valid knowledge while down-
ranking deprecated facts (correctly unlearning deprecated triples). In contrast, RotatE shows little such
separation, suggesting that outdated knowledge is retained alongside valid facts. Finally, multimodal
2

Table 1: Comparison of PrimeKG-CL with existing continual graph learning benchmarks. PrimeKG-
CL is the first to offer real temporal evolution on a biomedical KG with multimodal features and
multiple evaluation tasks. “Mod.” = modality types provided (S: structural, T: textual, V: visual, M:
molecular).
Benchmark Domain Real Temp. Mod. #Entities #Edges #Rel. #Tasks #Methods Eval Tasks
LKGE [6] General✗S 15K 310K 237 5 6 LP
DiCGRL [3] Robotics✓S 1K 10K 9 4 4 LP
IncDE [7] General✗S 15K 310K 237 5 5 LP
ICEWS [13] Political✓S 13K 386K 256 – – LP
MRCKG [14] General✗S,T,V 15K 99K 279 5 18 LP
PrimeKG-CL Biomedical ✓ S,T,M 129K+ 8.1M+ 30 10 10 LP, KGQA, NC
features improve entity-level tasks by up to 60%, while a recent CKGE framework (IncDE) fails to
scale to our 5.67M-triple base task, despite five attempts with up to 350 GB RAM.
Contributions.(1) PrimeKG-CL, the first CGL benchmark on a real biomedical KG with genuine
t0→t 1evolution (129K+ nodes, 8.1M+ edges, 10 tasks), with multimodal features and a persis-
tent/removed/added test stratification. (2) A 6 ×4 CL×decoder matrix + CMKL + LLM-RAG, across
three tasks, 5 seeds, 5 CL metrics. (3) The decoder–CL contraindication and decoder-conditional
correct-forgetting findings. (4) Open release of data, pipeline, baselines, and stratified split.
2 Related Work
Continual Learning for Knowledge Graphs.Continual learning sequentially learns from a stream
of tasks without catastrophically forgetting prior knowledge [ 4]. LKGE [ 6] introduced a lifelong
KG embedding framework applying EWC, experience replay, and distillation to TransE, DistMult,
and RotatE [ 15,16,12] on synthetic splits of FB15k-237 and WN18RR; DiCGRL [ 3] evaluates on
a robot manipulation KG; IncDE [ 7] uses incremental distillation; Synaptic Intelligence [ 9] tracks
importance online. More recent methods include FastKGE [ 17] (LoRA adapters), ERPP [ 18]
(evolutionary relation path passing), SAGE [ 19] (scale-aware embedding), DebiasedKGE [ 20]
(disentangled learning), and workshop work on EWC for KG link prediction [ 21]. A recent study [ 22]
rethinks CKGE benchmarks and finds that synthetic partitioning obscures pattern shifts between
snapshots, directly motivating our entity-type-grouped task design. All existing continual KGE
methods are evaluated on synthetically partitioned generic KGs, leaving their effectiveness on real-
world biomedical KG evolution untested. Broader continual graph learning [ 23] targets citation
or social networks with node or graph classification, not KG-specific tasks. Concurrent work [ 14]
constructs multimodal continual KG benchmarks from DB15K, MKG-W, and MKG-Y using synthetic
temporal splits; these general-domain datasets have≤15K entities and no real temporal dynamics.
Biomedical Knowledge Graphs.PrimeKG [ 1] integrates 20+ databases into a unified schema
with 129K+ nodes across 10 biomedical entity types. TxGNN [ 2] leverages PrimeKG for therapeutic
use prediction. Hetionet [ 24] is an earlier integrative biomedical network. BioMedKG [ 25] enriches
PrimeKG with biological sequences and textual descriptions, and PrimeKGQA [ 26] provides question-
answer pairs for biomedical QA. None has been studied in a continual learning setting with real
temporal evolution. Temporal KGs such as ICEWS [ 13] model event-based knowledge with explicit
timestamps; in contrast, biomedical KGs undergostructuralevolution (entities and relation types are
added or removed as scientific understanding advances), which PrimeKG-CL captures.
Multimodal Knowledge Graph Embedding.Multimodal KGE combines structural, textual, and
visual features: MMKGR [ 27] (gated cross-modal attention), MCLEA [ 28] (contrastive multimodal
alignment), MoSE [ 29] (score-level ensemble), and OGM-GE [ 30] (gradient modulation). Wu et
al. [31] identify the “greedy modality” phenomenon where dominant modalities suppress others.
These works operate in the static setting; PrimeKG-CL enables studying multimodal KGE under
continual learning, where modality-specific forgetting patterns emerge.
Knowledge Graphs and Language Models.Recent work highlights the complementarity of KGs
and large language models (LLMs) [ 32]. Our benchmark includes a RAG baseline [ 33] combining
KG retrieval with Qwen2.5-7B for continual KGQA.
3

Phenotype
Drug
Disease
Gene/ProteinPathwayAnatomy
BioProc .
MolFunc .
CellComp
SideEffect𝒕𝟎
 June 2021
129,375 nodes   8.1M edges   30 relations         
      
   Phenotype
Drug
Disease
Gene/ProteinPathwayAnatomy
BioProc .
MolFunc .
CellComp
SideEffect𝒕𝟏
 July 2023
131,211 nodes   13.0M edges   25 relations         
      
   Knowledge Evolution
+5830,243
new edges
-899,012
deprecated edges
7.21M persistent
Nine databases
Updating asynchronously
 GO. HPO. CTD. MONDA
Bgee . Uberon . HGNC
Gene/Protein Disease Drug Phenotype Pathway Anatomy BioProc . MolFunc . CellComp SideEffectFigure 1: PrimeKG-CL knowledge evolution from t0(June 2021) to t1(July 2023): 5.83M new
edges, 889K deprecated, 7.21M persistent. Node size encodes per-entity-type frequency at each
snapshot.
In contrast to prior CGL benchmarks, PrimeKG-CL provides the first evaluation framework on
a real-world biomedical knowledge graph with genuine temporal evolution and multimodal node
features, encompassing three evaluation tracks and ten methods across two KGE decoder families.
3 Benchmark Construction
Two years of real biomedical knowledge, 5.83 M new edges from nine asynchronously updating
databases, 889 K deprecated edges as ontologies were corrected, and 7.21 M persistent edges, pro-
duces a temporal structure that random splits of FB15k-237 simply do not have. PrimeKG-CL
captures this structure and turns it into a continual-learning workload (Figure 1).
3.1 Temporal Snapshots
We construct two temporal snapshots of PrimeKG [ 1].t0(June 2021)is the original PrimeKG
release (129,375 nodes, 10 entity types, 8,100,498 edges, 30 relation types; integrating 20+ databases
including DrugBank, DisGeNET, GO, HPO, Reactome, MONDO).
t1(July 2023)is reconstructed by re-querying nine freely accessible databases (Bgee, CTD, GO,
Gene2GO, HPO, HPOA, MONDO, Uberon, HGNC), which update asynchronously at independent
cadences (GO monthly, HPO quarterly, MONDO continuously); the resulting snapshot has 134,211
nodes and 13,041,729 edges across 25 relation types. Seven databases with restrictive licensing
(DrugBank, UMLS, DrugCentral, SIDER) or API changes (DisGeNET) were not re-queried, so
relation types sourced from these carry-forward databases have zero turnover by construction; the 5
relation types that drop from t1are drug-protein and drug-drug interactions whose source vocabulary
was deprecated, and the temporal-evolution analysis (§3.2) and stratified evaluation (Table 5) operate
on the freshly re-queried subgraph. The pipeline supports extension to additional snapshots; per-
snapshot statistics are in Table 2.
3.2 Temporal Difference
The temporal difference betweent 0andt 1reveals substantial real-world knowledge evolution:
•Added edges:5,830,243 new triples appear in t1that were absent in t0, reflecting newly
discovered associations (e.g., new gene-GO annotations, updated disease-phenotype links).
•Removed edges:889,012 triples present in t0are absent from t1, representing depre-
cated or corrected knowledge (e.g., retracted gene-disease associations, updated ontology
hierarchies).
4

Table 2: Dataset statistics for the two temporal snapshots in PrimeKG-CL. The temporal difference
reveals substantial real-world knowledge evolution between June 2021 and July 2023.
Statistict 0(June 2021)t 1(July 2023) Difference
Nodes 129,375 134,211 +4,836
Edges 8,100,498 13,041,729 +4,941,231
Node types 10 10 0
Relation types 30 25−5
Added edges – – +5,830,243
Removed edges – –−889,012
Persistent edges – – 7,211,486
Continual tasks 10 (entity-type grouping)
Evaluation tracks LP, KGQA, Node Classification
•Persistent edges:7,211,486 triples remain unchanged across both snapshots, providing a
stable knowledge backbone.
This temporal difference is fundamentally different from synthetic CGL benchmarks that randomly
partition a static graph. The additions and removals follow domain-specific patterns: Gene Ontology
annotations account for the largest share of added edges (reflecting the rapid expansion of func-
tional annotations), while disease-phenotype associations show high turnover as HPO and MONDO
ontologies are refined.
Stratified evaluation on added, removed, and persistent edges.The t0→t 1partition directly
supports stratified continual-learning analysis. In particular, the test split of the base task ( 1,620,099
triples) stratifies cleanly into 1,443,243 persistenttriples (89.1%, present in both snapshots, on
which a correctly updating model shouldretainits predictive ability) and 176,856 removedtriples
(10.9%, present in t0but deprecated in t1, on which an ideal model shouldunlearnits initial learned
association). The later tasks provide theaddedstratum: t1-only triples grouped by entity type. We
release this three-way stratification ( test_stratification.json ) with the benchmark so that
future work can report per-stratum MRR, quantifying correct forgetting of deprecated knowledge
alongside the standard AP/AF/BWT/REM metrics.
3.3 Continual Task Sequence
We organize the continual learning task sequence using anentity-type groupingstrategy. Rather than
randomly partitioning triples into artificial time steps (as in LKGE [ 6]), we group edges by their head
and tail entity types to create semantically coherent tasks. Specifically, we define 10 continual tasks
based on the dominant entity-type pairs in PrimeKG:
1. Gene/Protein↔Gene/Protein (protein-protein interactions)
2. Gene/Protein↔Biological Process
3. Gene/Protein↔Molecular Function
4. Gene/Protein↔Cellular Component
5. Gene/Protein↔Disease
6. Gene/Protein↔Pathway
7. Disease↔Phenotype
8. Drug↔Disease (drug indications and contraindications)
9. Drug↔Side Effect
10. Anatomy↔Gene/Protein (tissue expression)
Each task is presented sequentially at both t0andt1(Figure 2). The cost of getting this wrong is
concrete and large: even the strongest multimodal model on this benchmark (CMKL with DistMult)
sees Disease-r1 peak at MRR ≈0.32 when first learned and decay to ≈0.09 by task 10, a 3.4×
collapse on exactly the relations a clinical decision-support system would query. The grouping
5

Training Progress  T2 T1 T3 T4 T5 T6 T7 T8 T9 T10
Base Dis-r1 Drug -r1 Dis-r2 GP-r2 GP-r3 Phen -r3 BP-r4 Phen -r4 AP-r5Continual Learning Mechanisms
EWC
Fisher information regulationMultimodal Replay
K-means diverse exemplar buffer (|M| = 1000 )
Catastrophic ForgettingFigure 2: The 10-task continual-learning sequence on PrimeKG-CL (top) and per-task memory
curves for CMKL+DistMult under EWC + multimodal replay ( |M|=1000 ) (bottom). Disease-r1
peaks at MRR ≈0.32 when first learned (task 2) and degrades to ≈0.09 by task 10, a 3.4× collapse
despite EWC and replay mitigations.
mirrors how upstream databases push updates (GO functional annotations, HPO phenotype links,
DrugBank interactions), creating a dual challenge oftask identity shiftsandtemporal adaptation
(t0→t1within each task), analogous to domain-incremental CL in vision [34].
3.4 Evaluation Tasks
PrimeKG-CL supports three evaluation tracks, chosen so that a continual-learning failure on each one
corresponds to a concrete downstream cost: a missed drug-repurposing candidate (link prediction), a
stale answer to a clinician’s natural-language query (KGQA), or a newly catalogued entity assigned
to the wrong type (node classification).
Biomedical Relationship Prediction (Link Prediction, Primary).Given a query (h, r,?) or
(?, r, t) , the model ranks all candidate entities; we use filtered MRR as the per-task metric aj,i[15],
with Hits@ K(K∈ {1,3,10} ) in the supplement. This task supports drug repurposing, target
identification, and disease gene prioritization — all of which require continually incorporating new
knowledge.
Letting aj,ibe the filtered MRR on task iafter training through task jover Ttasks, we re-
port the standard CL metrics: AP=1
TP
iaT,i,AF=1
T−1P
i<Tmax j(aj,i−aT,i),BWT=
1
T−1P
i<T(aT,i−ai,i), and REM= 1−AF.
Knowledge Graph Question Answering (KGQA).We evaluate continual KGQA using a RAG
pipeline [ 33] that retrieves subgraph context from the current snapshot for an LLM, reporting Exact
Match, token-level F1, and accuracy. This track stands in for clinical decision support, where
practitioners query evolving knowledge through natural language.
Biomedical Entity Classification (Node Classification).We classify each node into its PrimeKG
entity type (gene/protein, disease, drug, etc.) from KG-derived structural and optional multimodal
features, reporting Macro-F1 and the CL metrics. This is the graph-learning counterpart of biomedical
entity recognition: instead of locating spans in text, the model must assign the correct type to each
6

Table 3: Continual biomedical relationship prediction on PrimeKG-CL: AP = filtered MRR over
129K+ entities (mean ±std, 5 seeds).Bold= per-column best; underline = second;italic= oracle.
AF, BWT, REM in Section S2. Discussion in Section 4.2.
CL Strategy TransE DistMult ComplEx RotatE
Naive Sequential0.004±0.000 0.058±0.001 0.011±0.001 0.084±0.005
EWC [4]0.004±0.000 0.058±0.0010.029±0.0020.088±0.003
SI [9]0.005±0.000 0.037±0.001 0.019±0.001 0.053±0.003
Distillation [10]0.005±0.000 0.032±0.001 0.018±0.001 0.036±0.000
Experience Replay [5]0.004±0.000 0.051±0.001 0.007±0.000 0.087±0.004
MIR [11]0.003±0.000 0.051±0.001 0.007±0.000 0.087±0.005
LKGE [6]0.039±0.001— — —
Joint Training (oracle) —0.047±0.001 0.025±0.001 0.164±0.001
Multimodal (DistMult only)
CMKL (struct-only) —0.071±0.005— —
CMKL (MoE, multimodal) — 0.062±0.010 — —
RAG (Qwen2.5-7B) is decoder-free and is reported on the KGQA track only (§4.4); we omit it from this LP table.
entity in an evolving graph. Structural-only baselines reach AP ≈0.34 –0.37, showing the task is
non-trivial without multimodal signal.
3.5 Multimodal Features
PrimeKG-CL ships three node-feature modalities:textual(768-d BiomedBERT [ 35] [CLS] embed-
dings of entity descriptions),molecular(1024-bit radius-2 Morgan fingerprints [ 8] for drug nodes
with SMILES, projected through a learned MLP), andstructural(R-GCN [ 36] embeddings capturing
multi-relational neighborhood context). Together they enable study of modality-specific forgetting, a
direction impossible on existing structural-only CGL benchmarks.
4 Experiments
4.1 Experimental Setup
Baselines.Ten methods spanning the major CL families:Naive Sequential(lower bound),Joint
Training(oracle), regularization (EWC[ 4],SI[ 9]), replay (Experience Replay[ 5],MIR[ 11]),
Distillation[ 10], the architecture-plus-distillation frameworkLKGE[ 6], aRAGagent (Qwen2.5-
7B) [ 33], andCMKL(R-GCN + DistMult, MoE fusion [ 29], modality-aware EWC, multimodal
replay). Each applicable method is run with four KGE decoders: TransE [ 15], DistMult [ 16],
ComplEx [ 37], RotatE [ 12], via PyKEEN. Per-method details and methods we could not scale
(FastKGE [ 17], ERPP [ 18], SAGE [ 19], IncDE [ 7] which exceeded 350 GB RAM on our 5.67M-
triple base task) are in Section S3.
Setup. d= 256 ,η= 0.001 , batch 512, 100 epochs/task, Adam; 70/10/20 train/valid/test per
task fixed across methods and seeds. EWC λ= 10 ; replay buffer 1,000 triples, K-means diverse
selection; CMKL uses the same buffer size, MoE fusion, per-modality Fisher. λ-sweep over 50× and
buffer 500–5,000 produced <10% AP variation. Per-decoder hyperparameters are shared across CL
strategies (we flag decoder-specific CL re-tuning as future work). NVIDIA V100 (32 GB), 5 seeds
{42,123,456,789,1024}, mean±s.d., pairedt-tests atp <0.05.
4.2 Biomedical Relationship Prediction (Link Prediction) Results
Table 3 reports filtered MRR over all 129K+ candidate entities. Contrary to the assumption that CL
strategy dominates such a matrix, the column choice (the KGE decoder) moves AP by 20×, while the
six CL strategies within any given column move it by at most 4×and in opposite directions across
columns. The remainder of this section unpacks this interaction.
7

Figure 3: Decoder ×CL interaction (AP relative to Naive, per column): EWC transforms ComplEx
(+167%); SI/Distillation drop RotatE by 37%/57%; replay drops ComplEx by 38%.
Per-task dynamics and the difficulty of real biomedical KGs.Per-task peak MRR reaches ≥0.25
on several tasks (e.g., CMKL: Disease-r1 0.32, Gene/Protein-r2 0.31, Phenotype-r3 0.25), confirming
meaningful patterns are learned even when AP is low. Three factors keep absolute AP modest: 129K+
candidate entities (KG-FIT [ 38] reports TransE MRR = 0.048 on a 10K-entity PrimeKG subset),
10 entity ×30 relation types of mixed semantics, and PrimeKG sparsity ( <1% of possible edges)
leaving many true positives outside the filter set [12].
RotatE dominates the decoder spectrum.The single largest effect in Table 3 is the choice
of decoder: on Naive Sequential, the four decoders span 20× — TransE ( 0.004 )≪ComplEx
(0.011 )<DistMult ( 0.058 )<RotatE ( 0.084 , 45% above DistMult). We attribute this to PrimeKG-
CL’s compositional and approximately rotational regularities (gene-gene interactions, phenotype
hierarchies, protein-protein symmetries) that rotational scoring captures but translational/bilinear
cannot, unlike FB15k-237/WN18RR where DistMult and RotatE typically tie.Researchers must
control for decoder choice when comparing CL strategies on biomedical KGs.
The decoder–CL contraindication.The interaction between decoder family and CL strategy is
the central benchmark finding: a CL strategy that helps one decoder can actively harm another,
with effects large enough to erase the decoder’s advantage (visualized in Figure 3).EWCis
universal but decoder-sensitive in magnitude: invisible on TransE/DistMult, transformative on
ComplEx ( 0.011→0.029 , +167%, t= 34.2 ,p <10−4), and a small significant boost on RotatE
(0.084→0.088 , +5%, p= 0.013 ), with EWC + RotatE the best configuration overall.SI and
Distillationcatastrophically degrade rotational decoders, dropping RotatE AP to 0.053 (−37%)
and0.036 (−57%) respectively while achieving AF ≈0(see Table S1): they prevent the rotational
geometry from adapting.Replay(ER and MIR) matches EWC on RotatE ( ≈0.087 ) but drops
ComplEx to 0.007 , 38% below Naive, consistent with reported ComplEx fragility under noisy
negatives [ 39]. There is no universal best CL strategy: decoder-agnostic CL benchmarking can be
actively misleading.
Joint Training and CMKL.Joint Training mirrors the contraindication: DistMult Joint reaches
AP= 0.047 (<Naive’s 0.058 , gradient interference across heterogeneous relations [ 40,41]); RotatE
Joint reaches 0.164 , nearly double the best CL configuration. The same-decoder multimodal CMKL
(R-GCN+DistMult) achieves AP = 0.062±0.010 (+7% over DistMult-Naive/EWC at 0.058 ), with
AF= 0.043±0.008 trading higher AP for richer drift; it is the strongestmultimodalmethod but not
the overall best (EWC+RotatE: 0.088±0.003 , 42% above CMKL-DistMult), indicating decoder and
modality contribute independently.
4.3 Biomedical Entity Classification (Node Classification) Results
CMKL achieves the highest AP ( 0.591±0.006 ), outperforming structural-only baselines ( 0.344 –
0.370 ) by up to 60%, with near-zero forgetting (AF = 0.008±0.010 , BWT = +0.003 ). All methods
show positive BWT, indicating that learning later associations slightly improves classification of
earlier-seen entities. The structural-only baselines cluster in two groups (Naive, EWC, ER at
≈0.344 ; SI, Distillation, MIR at ≈0.362 ) and remain far below CMKL, confirming that the
entity-classification gain is driven by multimodal features, not by CL strategy.
8

Table 4: Continual biomedical entity classification on PrimeKG-CL (Macro-F1, mean ±std over 5
seeds). LKGE and RAG are excluded as not applicable to this track.
Method AP↑AF↓BWT↑
Naive Sequential0.344±0.004 0.011±0.003 0.003±0.005
Joint Training0.370±0.0020.003±0.0040.022±0.004
EWC [4]0.345±0.004 0.008±0.003 0.007±0.004
Experience Replay [5]0.344±0.006 0.010±0.003 0.006±0.005
SI [9]0.362±0.004 0.007±0.007 0.015±0.006
Distillation [10]0.362±0.004 0.007±0.007 0.015±0.006
MIR [11]0.362±0.004 0.007±0.007 0.015±0.006
CMKL (MoE) 0.591±0.006 0.008±0.010 +0.003±0.008
Figure 4: Per-task peak MRR for CMKL, Naive, and Joint; grey line =training-set size. CMKL
peaks at0.25–0.32on Disease-r1, Gene/Protein-r2, Phenotype-r3.
Figure 5: Per-task forgetting (peak −final MRR). CMKL concentrates forgetting on Disease-r1
(56%,0.227); Naive/EWC concentrate it on Base-t 0(70%).
4.4 Biomedical Knowledge Graph Question Answering (KGQA) Results
We position the KGQA track as a stress test rather than a finished evaluation. All three configurations
of Qwen2.5-7B-Instruct sit at or below token F1 = 0.015 , with full RAG ( ≤0.015 ) only marginally
above retrieval-only ( 0.003±0.001 ) and zero-shot ( 0.000 ). Forgetting is essentially absent (AF
≤0.001 ) because there is little performance to forget. KGQA on a real, 129K-entity biomedical
graph is a regime where current RAG pipelines are barely above retrieval; we therefore present
this track as unsolved rather than saturated, calibrated to register future LLM-RAG improvements.
Per-task breakdown and prompt templates are in Section S5.
4.5 Analysis
Per-task heterogeneity and entity-type-specific forgetting.Per-task peak MRR is concentrated on
three tasks (Figure 4): Disease-r1, Gene/Protein-r2, Phenotype-r3 each exceed 0.1; the others remain
near zero (heterogeneoussemanticdifficulty rather than scale). Forgetting is similarly concentrated
(Figure 5): CMKL puts 56% on Disease-r1 (0.227), while Naive and EWC put 70% on Base- t0
(0.152) with much less Disease forgetting (0.014). Gene/protein entities make up 73% of triples,
biasing performance toward gene/protein-centric tasks.
Correct forgetting is decoder-conditional.Standard AP and AF conflate two desiderata:retaining
persistentknowledge (still true at t1) andunlearning deprecatedknowledge (only true at t0). We
9

Table 5: Stratified filtered MRR on the final-task test set.Persistenttriples are in both t0andt1(a
correctly updating model should retain them);removedtriples are in t0but deprecated in t1(an ideal
model should forget them: lower is better);addedtriples are new in t1(mean MRR across tasks 1–9).
Values are mean±std over 5 seeds.
Method Decoder Persistent↑Removed↓Added↑
Naive Sequential DistMult0.096±0.005 0.009±0.000 0.055±0.001
Naive Sequential RotatE0.106±0.003 0.078±0.002 0.082±0.005
EWC DistMult0.095±0.007 0.012±0.001 0.055±0.001
EWC RotatE0.106±0.002 0.078±0.003 0.086±0.004
Joint Training DistMult0.213±0.006 0.097±0.004 0.081±0.002
Joint Training RotatE0.231±0.004 0.160±0.003 0.224±0.002
split the base-task test set into 1,443,243 persistent and 176,856 removed triples, with a separate
addedstratum of 1,156,876 task-1..9 test triples newly introduced at t1, and report per-stratum
filtered MRR over 5 seeds in Table 5. The pattern is again decoder-conditional. Under DistMult,
Naive’s drift yields a clean correct-forgetting signal: a persistent/removed ratio of 11× (0.096 vs
0.009 ); EWC mildly over-protects deprecated edges ( 8×), as its uniform Fisher penalty cannot
distinguish parameters that store still-true associations from those that store deprecated ones; Joint
Training, which optimizes on the union of both strata, keeps them together ( 2.2× ). Under RotatE,
all three methods collapse to ≈1.4× : rotational geometry retains patterns uniformly, dissolving
the implicit correct-forgetting signal that DistMult-Naive achieves through drift. We release the
split ( test_stratification.json ) so future methods can target a metric that current baselines
optimize only by accident.
5 Conclusion
We introduced PrimeKG-CL, the first CGL benchmark grounded in a real biomedical KG with
genuine temporal evolution (129K+ nodes, 8.1M+ edges, 10 entity-type tasks ×two snapshots,
multimodal features, three tasks). Across 6 CL methods ×4 decoders, decoder and CL strategy must
be co-designed: EWC+RotatE is the best practical configuration (0.088), SI/Distillation drop RotatE
by 37–57%, replay drops ComplEx by 38%, IncDE does not scale, and multimodal features add up to
+60% on entity classification. Stratified evaluation reveals a decoder-conditional correct-forgetting
signal that AP/AF average away. The consequences are concrete: any CL study on biomedical KGs
that ignores decoder family is reporting a confound, and any study reporting only AP/AF averages over
correct and incorrect forgetting. Data, pipeline, baselines, and stratification are released; Sections S7
and S8 cover limitations and availability.
References
[1]Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable precision
medicine.Scientific Data, 10(1):67, 2023.
[2]Kexin Huang, Payal Chandak, Qianwen Wang, Shreyas Haber, and Marinka Zitnik. A foundation model
for clinician-centered drug repurposing.Nature Medicine, 30(12):3601–3613, 2024.
[3]Angel Daruna, Mehul Gupta, Mohan Sridharan, and Sonia Chernova. Continual learning of knowledge
graph embeddings.IEEE Robotics and Automation Letters, 6(2):1128–1135, 2021.
[4]James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu,
Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic
forgetting in neural networks.Proceedings of the National Academy of Sciences, 114(13):3521–3526,
2017.
[5]David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P Lillicrap, and Gregory Wayne. Experience
replay for continual learning. InAdvances in Neural Information Processing Systems, volume 32, 2019.
[6]Yuanning Cui, Yuxin Wang, Zequn Sun, Wenqiang Liu, Yiqiao Jiang, Kexin Han, and Wei Hu. Lifelong
embedding learning and transfer for growing knowledge graphs. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 37, pages 4218–4226, 2023.
10

[7]Jiajun Liu, Wenjun Ke, Peng Wang, Ziyu Shang, Jinhua Gao, Guozheng Li, Ke Ji, and Yanhe Liu. Towards
continual knowledge graph embedding via incremental distillation. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 38, pages 8759–8768, 2024.
[8]David Rogers and Mathew Hahn. Extended-connectivity fingerprints.Journal of Chemical Information
and Modeling, 50(5):742–754, 2010.
[9]Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In
Proceedings of the 34th International Conference on Machine Learning, pages 3987–3995. PMLR, 2017.
[10] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network.arXiv
preprint arXiv:1503.02531, 2015.
[11] Rahaf Aljundi, Lucas Caccia, Eugene Belilovsky, Massimo Caccia, Min Lin, Laurent Charlin, and
Tinne Tuytelaars. Online continual learning with maximally interfered retrieval. InAdvances in Neural
Information Processing Systems, volume 32, 2019.
[12] Zhiqing Sun, Zhi-Hong Deng, Jian-Yun Nie, and Jian Tang. RotatE: Knowledge graph embedding by
relational rotation in complex space. InProceedings of the 7th International Conference on Learning
Representations, 2019.
[13] Alberto García-Durán, Sebastijan Duman ˇci´c, and Mathias Niepert. Learning sequence encoders for
temporal knowledge graph completion. InProceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 4816–4821, 2018.
[14] Linyu Li, Zhi Jin, Yichi Zhang, Dongming Jin, Yuanpeng He, Haoran Duan, Gadeng Luosang, and Nyima
Tashi. When modalities remember: Continual learning for multimodal knowledge graphs.arXiv preprint
arXiv:2604.02778, 2026.
[15] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translat-
ing embeddings for modeling multi-relational data. InAdvances in Neural Information Processing Systems,
volume 26, 2013.
[16] Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. Embedding entities and relations
for learning and inference in knowledge bases. InProceedings of the 3rd International Conference on
Learning Representations, 2015.
[17] Jiajun Liu, Wenjun Ke, Peng Wang, Jiahao Wang, Jinhua Gao, Ziyu Shang, Guozheng Li, Zijie Xu, Ke Ji,
and Yining Li. Fast and continual knowledge graph embedding via incremental LoRA. InProceedings of
the 33rd International Joint Conference on Artificial Intelligence, pages 2159–2167, 2024.
[18] Jing Yang, Xinfa Jiang, Xiaowen Jiang, Yuan Gao, Laurence T Yang, Shaojun Zou, and Shundong Yang.
From knowledge forgetting to accumulation: Evolutionary relation path passing for lifelong knowledge
graph embedding. InProceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval, pages 1197–1206, 2025.
[19] Yifei Li, Lingling Zhang, Hang Yan, Tianzhe Zhao, Zihan Ma, Muye Huang, and Jun Liu. SAGE: Scale-
aware gradual evolution for continual knowledge graph embedding. InProceedings of the 31st ACM
SIGKDD Conference on Knowledge Discovery and Data Mining, 2025.
[20] Junlin Zhu, Bo Fu, and Guiduo Duan. DebiasedKGE: Towards mitigating spurious forgetting in continual
knowledge graph embedding. InProceedings of the 34th ACM International Conference on Information
and Knowledge Management (CIKM), 2025.
[21] Gaganpreet Jhajj and Fuhua Lin. Elastic weight consolidation for knowledge graph continual learning: An
empirical evaluation.arXiv preprint arXiv:2512.01890, 2025.
[22] Tianzhe Zhao, Jiaoyan Chen, Yanchi Ru, Qika Lin, Yuxia Geng, Haiping Zhu, Yudai Pan, and Jun Liu.
Rethinking continual knowledge graph embedding: Benchmarks and analysis. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information Retrieval, 2025.
[23] Xikun Zhang, Dongjin Song, and Dacheng Tao. Cglb: Benchmark tasks for continual graph learning. In
Advances in Neural Information Processing Systems, volume 35, pages 13006–13021, 2022.
[24] Daniel Scott Himmelstein, Antoine Lizee, Christine Hessler, Leo Brueggeman, Sabrina L Chen, Dexter
Hadley, Ari Green, Pouya Khankhanian, and Sergio E Baranzini. Systematic integration of biomedical
knowledge prioritizes drugs for repurposing.eLife, 6:e26726, 2017.
11

[25] Tien Dang, Viet Thanh Duy Nguyen, Minh Tuan Le, and Truong-Son Hy. Multimodal contrastive
representation learning in augmented biomedical knowledge graphs.arXiv preprint arXiv:2501.01644,
2025.
[26] Xi Yan, Patrick Westphal, Jan Seliger, and Ricardo Usbeck. Bridging the gap: Generating a comprehensive
biomedical knowledge graph question answering dataset. InProceedings of the 27th European Conference
on Artificial Intelligence (ECAI), volume 392 ofFrontiers in Artificial Intelligence and Applications, pages
1198–1205, 2024.
[27] Shangfei Zheng, Weiqing Wang, Jianfeng Qu, Hongzhi Yin, Wei Chen, and Lei Zhao. MMKGR: Multi-hop
multi-modal knowledge graph reasoning. InProceedings of the 39th IEEE International Conference on
Data Engineering (ICDE), pages 96–109, 2023.
[28] Zhenxi Lin, Ziheng Zhang, Meng Wang, Yinghui Shi, Xian Wu, and Yefeng Zheng. Multi-modal contrastive
representation learning for entity alignment. InProceedings of the 29th International Conference on
Computational Linguistics, pages 2572–2584, 2022.
[29] Yu Zhao, Xiangrui Cai, Yike Wu, Haiwei Zhang, Ying Zhang, Guoqing Zhao, and Ning Jiang. Mose:
Modality split and ensemble for multimodal knowledge graph completion. InProceedings of the 2022
Conference on Empirical Methods in Natural Language Processing, pages 10527–10536, 2022.
[30] Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. Balanced multimodal learning via
on-the-fly gradient modulation. InProceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 8238–8247, 2022.
[31] Nan Wu, Stanisław Jastrzebski, Kyunghyun Cho, and Krzysztof J Geras. Characterizing and overcoming
the greedy nature of learning in multi-modal deep neural networks. InProceedings of the 39th International
Conference on Machine Learning, pages 24043–24055. PMLR, 2022.
[32] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language
models and knowledge graphs: A roadmap.IEEE Transactions on Knowledge and Data Engineering,
36(7):3580–3599, 2024.
[33] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Information Processing Systems, volume 33, pages 9459–9474,
2020.
[34] Gido M van de Ven and Andreas S Tolias. Three scenarios for continual learning.arXiv preprint
arXiv:1904.07734, 2019.
[35] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann,
Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural
language processing.ACM Transactions on Computing for Healthcare, 3(1):1–23, 2021.
[36] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, and Max Welling.
Modeling relational data with graph convolutional networks. InThe Semantic Web: 15th International
Conference (ESWC 2018), pages 593–607. Springer, 2018.
[37] Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, and Guillaume Bouchard. Complex
embeddings for simple link prediction. InProceedings of the 33rd International Conference on Machine
Learning, pages 2071–2080, 2016.
[38] Pengcheng Jiang, Lang Cao, Cao Xiao, Parminder Bhatia, Jimeng Sun, and Jiawei Han. KG-FIT:
Knowledge graph fine-tuning upon open-world knowledge. InAdvances in Neural Information Processing
Systems, volume 37, 2024.
[39] Daniel Ruffinelli, Samuel Broscheit, and Rainer Gemulla. You CAN teach an old dog new tricks! on
training knowledge graph embeddings. InProceedings of the 8th International Conference on Learning
Representations, 2020.
[40] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient
surgery for multi-task learning. InAdvances in Neural Information Processing Systems, volume 33, pages
5824–5836, 2020.
[41] Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro.
Learning to learn without forgetting by maximizing transfer and minimizing interference. InInternational
Conference on Learning Representations, 2019.
12

PrimeKG-CL: A Continual Graph Learning Benchmark on
Evolving Biomedical Knowledge Graphs
Supplementary Material
S1 Experimental Details
All experiments use NVIDIA V100 GPUs (32 GB) with PyTorch 2.0. Key hyperparameters: em-
bedding dimension 256, learning rate 0.001, batch size 512, 100 epochs per task, 50,000 sampled
triples per epoch. For structural-only baselines, a single EWC penalty λ= 10 is used. For CMKL,
modality-specific lambdas are used: λs= 10 (structural), λt= 5 (textual), λm= 1 (molecular),
λf= 5 (fusion), λr= 50 (main relation embeddings, set high because rotational decoders treat
relations as rotation angles where any drift warps the full embedding geometry). Replay buffer: 1,000
triples total via K-means selection on structural embeddings. ComplEx and RotatE decoders follow
the PyKEEN convention (entity dim = 2D ,Dcomplex pairs interleaved); RotatE relations are D
phases initialized uniformly in [−π, π] , ComplEx relations are full-complex 2Dreal values initialized
with Xavier. All results averaged over 5 seeds{42,123,456,789,1024}.
Scoring-function corrections (2026-04-12).During a mid-revision audit we identified and fixed
four bugs in the PyKEEN-based scoring pipeline that had affected earlier drafts. (i)RotatE complex-
tensor handling: raw PyKEEN entity embeddings are stored as [N,2D] real, which must be
converted to [N, D] complex via torch.view_as_complex ; our prior implementation treated the
raw tensor as already-complex, silently corrupting the head embedding. (ii)RotatE evaluation norm:
eval used L1instead of L2on the complex difference, inconsistent with training. (iii)ComplEx:
the same raw-to-complex conversion bug as RotatE. (iv)TransE training norm: training used L2
where the PyKEEN convention for TransE is L1; re-running confirmed the numerical effect is within
seed noise for TransE, so reported TransE numbers are unchanged. All four corrections were verified
against PyKEEN’s model.score_hrt reference. The new RotatE and ComplEx numbers reported
here post-date these fixes; we re-ran the full4-decoder matrix under the corrected pipeline.
IncDE scalability note.We attempted to evaluate IncDE [ 7], a representative incremental-
distillation CKGE method, following the official repository. Over five attempts (including the
bug fixes listed in the paper’s GitHub issue tracker) we observed out-of-memory failures at 64 GB,
200 GB, and 350 GB RAM, and deadlocks in the training loop on the 5.67 M-triple base task. In
our hands, the published implementation does not scale to the task-0 graph of PrimeKG-CL. We
cannot rule out implementation differences relative to the authors’ environment; we document our
configurations alongside the released code. This negative result is informative: it highlights a gap
between general-domain CKGE benchmarks ( ≤15 K entities) and real biomedical graphs where
methods that appear scalable on paper can become intractable in practice.
S2 Additional Results
Forward Transfer (FWT) equals zero for all continual methods in our sequential setting, as tasks are
non-overlapping and models are not evaluated on unseen tasks before training. Hits@1, Hits@3, and
Hits@10 are available in the released evaluation code.
S3 Baseline Details
We evaluate the following ten methods, each with the indicated hyperparameter setting. All KGE-
based baselines use embedding dimension d= 256 , learning rate η= 10−3, batch size 512, and 100
training epochs per task with the Adam optimizer.
•Naive Sequential.Fine-tunes on each task sequentially without any forgetting-mitigation
mechanism. Serves as a lower bound on knowledge retention.
•Joint Training.Trains on the union of all tasks simultaneously. Serves as a reference oracle
since no forgetting can occur by construction.
1

Table S1:Average Forgetting (AF) ↓for the full 6-method ×4-decoder matrix (mean ±std, 5
seeds). Negative values mean the task improved from its first-seen-performance to end-of-sequence
performance. SI and Distillation achieve near-zero AF but at a large AP cost on rotational decoders
(see Table 3); replay variants (ER, MIR) show moderate AF but devastating AP drops for ComplEx.
CL Strategy TransE DistMult ComplEx RotatE
Naive Sequential0.020±0.000 0.004±0.001 0.030±0.001 0.027±0.002
EWC0.017±0.000 0.006±0.001 0.010±0.001 0.022±0.003
SI≈0.000−0.002±0.000−0.002±0.000 0.004±0.002
Distillation≈0.000−0.002±0.000≈0.000−0.002±0.000
Experience Replay0.021±0.001 0.013±0.002 0.035±0.001 0.027±0.002
MIR0.022±0.000 0.013±0.001 0.034±0.001 0.028±0.002
Joint (oracle) —0.000 0.000 0.000
Table S2:Backward Transfer (BWT)↑for the full matrix.
CL Strategy TransE DistMult ComplEx RotatE
Naive Sequential−0.020±0.000−0.004±0.001−0.030±0.001−0.027±0.002
EWC−0.016±0.000−0.006±0.001−0.010±0.001−0.022±0.003
SI≈0.000 +0.006±0.000 +0.004±0.000 +0.007±0.002
Distillation≈0.000 +0.005±0.000 +0.002±0.001 +0.006±0.000
Experience Replay−0.021±0.001−0.012±0.002−0.035±0.001−0.027±0.002
MIR−0.021±0.000−0.011±0.001−0.034±0.001−0.028±0.002
Joint (oracle) —0.000 0.000 0.000
•EWC[ 4]. Regularization-based approach: penalizes parameter changes weighted by the
diagonal Fisher information from previous tasks. Defaultλ= 10.
•Experience Replay[ 5]. Memory-based approach: replays a 1,000-triple exemplar buffer
selected by K-means diverse sampling.
•SI (Synaptic Intelligence)[ 9]. Regularization-based: tracks parameter importance online
via accumulated gradients during training, without requiring a separate Fisher pass.
•Distillation[ 10]. Knowledge-distillation approach: minimizes divergence between current-
task predictions and snapshot model outputs.
•MIR (Maximally Interfered Retrieval)[ 11]. Replay variant: selects buffer exemplars
most affected by the current gradient update.
•LKGE[ 6]. Combined architecture and distillation framework, the state-of-the-art CGL
approach for KGs. Constrained to TransE backbone.
•RAG Agent (Qwen2.5-7B)[ 33]. Non-parametric retrieval-augmented generation pipeline.
Included for completeness; evaluated on the KGQA track.
•CMKL (R-GCN + DistMult).Multimodal CGL method with textual (BiomedBERT),
molecular (Morgan FP), and structural (R-GCN) features fused via mixture-of-experts [ 29];
uses modality-aware EWC ( λs=10, λ t=5, λ m=1, λ f=5, λ r=50) and a 1,000-triple multi-
modal replay buffer.
Methods we could not run at scale.FastKGE [ 17], ERPP [ 18], and SAGE [ 19] are not included
because their code was not publicly available at the time of submission. We additionally attempted
IncDE [ 7], following the official repository: across five attempts that scaled RAM up to 350 GB, the
published implementation did not complete training on our 5.67M-triple base task (out-of-memory
failures and training-loop deadlocks). We document our exact configurations alongside the released
code; the negative result itself documents the gap between general-domain CKGE benchmarks ( ≤15K
entities) and real biomedical-scale graphs.
S4 Per-task Learning Matrices
We complement the per-task and decoder-CL visualizations in the main paper (Figures 3 to 5) with full
per-task learning matrices in Figure S1. Each cell (i, j) reports filtered MRR on task jafter training
2

through task i(mean over 5 seeds, DistMult decoder), so the diagonal is each task’s peak performance
and below-diagonal columns trace catastrophic forgetting through the rest of the sequence. The
matrix makes two patterns visible that the aggregate AP/AF metrics flatten: (i) Disease-r1 and
Gene/Protein-r2 dominate the diagonal (peaks ≥0.30 ) but their column also shows the steepest
decay, so the highest-performing tasks contribute disproportionately to forgetting; (ii) the bottom
rows of the Naive matrix are nearly uniform horizontally, indicating that by the time the model has
trained through later tasks, its representation no longer differentiates the easier task-types it once
retained. The CMKL matrix shows the opposite pattern: peaks on Disease-r1, Gene/Protein-r2, and
Phenotype-r3 are preserved through the sequence, with the dominant decay isolated to Disease-r1
alone.
1 5 10
Evaluated on task1
5
10After training through task0.095
0.016 0.321
0.017 0.199 0.001
0.018 0.222 0.001 0.001
0.037 0.221 0.001 0.001 0.308
0.025 0.178 0.001 0.001 0.134 0.001
0.028 0.097 0.001 0.001 0.108 0.001 0.247
0.019 0.097 0.001 0.001 0.097 0.001 0.232 0.016
0.020 0.089 0.001 0.001 0.090 0.001 0.195 0.010 0.001
0.024 0.094 0.001 0.001 0.297 0.001 0.179 0.009 0.001 0.015CMKL
(MoE-DistMult)
1 5 10
Evaluated on task1
5
100.236
0.237 0.017
0.222 0.015 0.001
0.213 0.015 0.001 0.001
0.100 0.004 0.001 0.001 0.207
0.097 0.004 0 0.001 0.203 0.001
0.097 0.006 0 0.001 0.204 0.001 0.024
0.095 0.006 0.001 0.001 0.204 0.001 0.024 0.068
0.094 0.006 0 0.001 0.201 0.001 0.024 0.068 0.001
0.086 0.003 0 0.001 0.385 0.001 0.008 0.036 0.001 0.061Naive Seq.
(DistMult)
1 5 10
Evaluated on task1
5
100.238
0.237 0.017
0.223 0.015 0.001
0.215 0.015 0.001 0.001
0.102 0.004 0 0.001 0.217
0.099 0.004 0 0.001 0.212 0.001
0.098 0.006 0 0.001 0.211 0.001 0.029
0.098 0.006 0 0.001 0.212 0.001 0.028 0.065
0.095 0.006 0 0.001 0.212 0.001 0.029 0.065 0.001
0.086 0.003 0 0 0.378 0.001 0.010 0.034 0.001 0.064EWC
(DistMult)
1 5 10
Evaluated on task1
5
100.236
0.237 0.016
0.237 0.016 0.001
0.236 0.016 0.001 0.001
0.263 0.016 0.001 0.001 0.048
0.262 0.016 0.001 0.001 0.048 0.001
0.265 0.016 0.001 0.001 0.048 0.001 0.001
0.262 0.016 0.001 0.001 0.049 0.001 0.001 0.005
0.264 0.016 0.001 0.001 0.049 0.001 0.001 0.005 0.001
0.287 0.016 0.001 0.001 0.049 0.001 0.001 0.005 0.001 0.006SI
(DistMult)
0.000.050.100.150.200.250.30
Filtered MRR
Figure S1: Per-task learning matrices (DistMult, 5 seeds). Entry (i, j) = filtered MRR on task j
after training through task i; diagonal =peak per-task performance, off-diagonal decay =forgetting
trajectory. CMKL preserves the diagonal more uniformly than Naive/EWC.
S5 KGQA Details
We evaluate continual KGQA using three configurations of Qwen2.5-7B-Instruct to ablate the
contribution of retrieval:
•Full RAG:LLM with ChromaDB subgraph retrieval (200 questions/task).
•Retrieval-only:Extracts answers from retrieved triples via majority voting, without LLM
generation.
•Zero-shot LLM:LLM answers directly from parametric knowledge, without any retrieval.
The zero-shot LLM achieves token F1 = 0.000 across all tasks, failing to generate correct biomedical
entity names from parametric knowledge alone. Retrieval-only achieves token F1 = 0.003±0.001 ,
demonstrating that retrieved triples contain relevant information but simple entity extraction is
insufficient. Full RAG achieves the highest performance (token F1 ≤0.015), highlighting the
challenge of generating exact biomedical entity names from free-text LLM output. All configurations
exhibit near-zero forgetting (AF ≤0.001 ), as retrieval-augmented approaches store knowledge in the
index rather than model weights.
S6 Analysis Details
Per-task performance variation.A subset of tasks (Disease-r1, Gene/Protein-r2, Phenotype-r3)
achieve substantially higher peak MRR ( >0.1 ), driven by larger training sets and denser graph
neighborhoods. Other tasks, including Drug (r1), Disease (r2), BioProcess (r4), and Phenotype
(r4), are substantially harder, with peak MRR below 0.05 for all methods. This heterogeneity is a
realistic characteristic of biomedical KGs that synthetic benchmarks with uniform random splits do
not capture.
Gene/protein dominance effect.Gene/protein entities account for approximately 73% of triples
in PrimeKG, creating a substantial type imbalance. This affects all methods: performance on
gene/protein-centric tasks is consistently higher than on tasks involving rarer entity types.
3

Entity-type-specific forgetting patterns.For CMKL (MoE), Disease-r1 accounts for 56% of
total forgetting (0.227), followed by Base ( t0) at 18% and Phenotype-r3 at 17%. In contrast, Naive
Sequential and EWC concentrate forgetting on Base ( t0) at 70% (0.152), with substantially less
Disease forgetting (0.014). This divergence reveals that the dominant source of forgetting depends on
model architecture: multimodal representations shift the vulnerability from the largest task (Base,
which dominates structural embeddings) to disease-related tasks where textual and structural signals
interact. Drug, Gene/Protein-r3, and Anatomy/Pathway tasks show near-zero forgetting across all
methods, suggesting these entity types have more stable embedding neighborhoods.
Domain-specific insights.The biomedical domain introduces unique CGL challenges absent
from generic benchmarks. First, ontology updates (e.g., MONDO disease hierarchy changes) cause
systematic triple removals that look like “forgetting” but actually reflect corrected knowledge. Second,
highly connected hub nodes (e.g., TP53, BRCA1) participate in many relation types, creating inter-
task dependencies that complicate task-isolated continual learning. Third, the multimodal nature
of biomedical entities means that forgetting can be modality-specific: a model may retain a drug’s
molecular properties while forgetting its textual description-based associations.
S7 Limitations and Future Work
•Two snapshots.Only t0(2021) and t1(2023) are currently included; the construction
pipeline supports extension to additional snapshots as upstream databases update.
•Seven licensed databases excluded.DrugBank, UMLS, DrugCentral, SIDER, and others
with restrictive licensing could not be re-queried for t1; including them would enrich
drug-related dynamics.
•Gene/protein dominance.73% of triples involve gene/protein entities, biasing evaluation
toward this type. Type-balanced splits are a future direction.
•Evaluation paradigm gap.KGE methods use filtered MRR with all-entity ranking; the
RAG agent uses entity-name matching against retrieved subgraphs. Direct cross-paradigm
comparison requires caution.
Future work.Promising directions include integrating licensed databases for richer drug-disease
dynamics; co-designed decoder-CL pairs (e.g., replay variants for Hermitian bilinear decoders,
regularization schemes that preserve rotational geometry); multi-hop reasoning tasks; inductive
continual learning where new entity types appear across snapshots; and foundation-model approaches
for continual biomedical KG learning building on KG-LLM complementarity [32].
S8 Data Availability, License, and Maintenance
License.Code components are released under the MIT license. Data files are released under Creative
Commons Attribution 4.0 International (CC BY 4.0), inheriting PrimeKG’s license; users must
respect upstream-database licenses where applicable. The full LICENSE file ships in the dataset
archive root.Hosting.Long-term hosting via HuggingFace Datasets (DOI assigned upon accep-
tance) and a GitHub release tag at the publication SHA. Croissant 1.0 metadata ( croissant.json )
accompanies the release.Maintenance plan.We commit to (i) corrective releases for downstream-
database-driven errata for at least 24 months after publication, (ii) a tagged release for each future
tisnapshot we add, (iii) GitHub Issues triage with a 72h initial response target during the year fol-
lowing publication.Contents.Temporal snapshots ( kg_t0.csv ,kg_t1.csv ), task directories with
70/10/20 train/valid/test splits, multimodal features ( text_embeddings.pt ,mol_features.pt , R-
GCN tensors), persistent/added/removed test stratification ( test_stratification.json ), per-task
statistics, and reference baseline + CMKL implementations.Datasheet for Datasets.A datasheet
following Gebru et al. (2021) is included as DATASHEET.md in the release.Stratified evaluation
scope.The stratified-MRR table (Table 5) reports 6 cells (3 methods ×2 decoders, 5 seeds each), the
configurations our compute budget supported within the submission window. Extending the matrix to
the full 6 CL methods ×4 decoders is one of the first follow-on experiments the released benchmark
and stratification metadata enable.
4