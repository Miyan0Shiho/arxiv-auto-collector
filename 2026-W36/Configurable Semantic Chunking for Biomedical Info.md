# Configurable Semantic Chunking for Biomedical Information Extraction in Retrieval-Augmented Generation

**Authors**: Riya Ahuja, Tim Kacprowski, Roya Shiasi Sardoabi

**Published**: 2026-08-31 17:44:54

**PDF URL**: [https://arxiv.org/pdf/2608.31139v1](https://arxiv.org/pdf/2608.31139v1)

## Abstract
BioMedRAG introduced retrieval-augmented generation with a learned chunk scorer for biomedical information extraction. However, it relies on fixed-size chunking which can fragment semantic evidence. We propose a configurable semantic chunking framework that addresses this limitation by combining entity-preserving windows, trigger-centered chunking, proposition-first extraction, tiered trigger prioritization, and hierarchical relation resolution. The framework integrates with BioMedRAG by replacing only the chunk construction stage while preserving the embedding model, learned chunk scorer, generator, and evaluation protocol. We evaluate the framework on biomedical relation extraction benchmarks (GM-CIHT, DDI, ChemProt) and adverse event classification (ADE). On GM-CIHT, the full hybrid configuration achieves 82.6% F1, improving over the fixed-size baseline (74.2% F1) by 8.4 points under our experimental setup. Cross-dataset analysis shows that semantic chunking improves extraction datasets with explicit relation cues, such as GM-CIHT and DDI, while fixed chunking remains competitive or stronger for dense biochemical extraction and binary classification settings such as ChemProt and ADE. By externalizing chunking logic into configuration files, the framework provides an interpretable and adaptable alternative to rigid fixed-size chunking for biomedical RAG pipelines.

## Full Text


<!-- PDF content starts -->

Configurable Semantic Chunking for Biomedical Information Extraction in
Retrieval-Augmented Generation
Riya Ahuja1,2,∗,Tim Kacprowski1,2,Roya Shiasi Sardoabi1,2,∗
1Institute of Data Science in Biomedicine, Technische Universit ¨at Braunschweig, Germany
2Braunschweig Integrated Centre of Systems Biology (BRICS), Technische Universit ¨at Braunschweig,
Germany
∗Corresponding authors: Riya Ahuja and Roya Shiasi Sardoabi
r.ahuja@tu-braunschweig.de, t.kacprowski@tu-braunschweig.de,
roya.shiasi-sardoabi@tu-braunschweig.de
Abstract
BioMedRAG introduced retrieval-augmented gen-
eration with a learned chunk scorer for biomedi-
cal information extraction. However, it relies on
fixed-size chunking which can fragment seman-
tic evidence. We propose a configurable seman-
tic chunking framework that addresses this lim-
itation by combining entity-preserving windows,
trigger-centered chunking, proposition-first extrac-
tion, tiered trigger prioritization, and hierarchi-
cal relation resolution. The framework integrates
with BioMedRAG by replacing only the chunk
construction stage while preserving the embedding
model, learned chunk scorer, generator, and eval-
uation protocol. We evaluate the framework on
biomedical relation extraction benchmarks (GM-
CIHT, DDI, ChemProt) and adverse event classi-
fication (ADE). On GM-CIHT, the full hybrid con-
figuration achieves 82.6% F1, improving over the
fixed-size baseline (74.2% F1) by 8.4 points un-
der our experimental setup. Cross-dataset analy-
sis shows that semantic chunking improves extrac-
tion datasets with explicit relation cues, such as
GM-CIHT and DDI, while fixed chunking remains
competitive or stronger for dense biochemical ex-
traction and binary classification settings such as
ChemProt and ADE. By externalizing chunking
logic into configuration files, the framework pro-
vides an interpretable and adaptable alternative
to rigid fixed-size chunking for biomedical RAG
pipelines.
1 Introduction
Biomedical literature is expanding quickly, with PubMed
now indexing more than 39 million articles [National Li-
brary of Medicine, 2026 ]. Extracting structured knowledge,
such as drug-disease interactions, gene-protein relationships,
and adverse drug events, from this vast corpus is critical
for clinical decision support, drug discovery, and precision
medicine [Grouin and Grabar, 2024 ]. Retrieval-augmented
generation (RAG) has emerged as a powerful paradigm forthis task [Lewiset al., 2020 ], combining large language mod-
els (LLMs) with external evidence retrieval to improve factual
grounding and interpretability.
BioMedRAG [Liet al., 2025 ]introduced a special-
ized RAG framework for biomedical information extrac-
tion and reports strong results on relation extraction bench-
marks. However, it relies on fixed-size chunking, uni-
formly splitting sentences into 5-word windows regardless
of semantic structure. This strategy frequently fragments
relation-bearing expressions, resulting in incomplete or mis-
aligned evidence. For example, the sentence“Aspirin treats
headache by inhibiting prostaglandin synthesis”may be split
into chunks such as“Aspirin treats headache by inhibiting”
and“prostaglandin synthesis”, separating the drug mecha-
nism from its target. Such fragmentation reduces retrieval
precision and forces the LLM to infer missing relational con-
text.
To address this limitation, we propose aconfigurable
semantic chunking frameworkthat preserves meaningful
semantic boundaries through an incremental design. Our
framework builds incrementally on entity-aware segmenta-
tion that respects named entity boundaries. We progressively
incorporate: (1) relation-trigger-centered extraction with
tiered prioritization to distinguish explicit triggers (treats,in-
hibits) from generic terms (medication,drug); (2) hierarchi-
cal relation resolution to handle competing relation types; and
(3) proposition-first extraction that isolates minimal subject-
trigger-object spans. These components integrate through a
unified scoring mechanism that balances trigger confidence,
relation specificity, and contextual alignment, enabling seam-
less integration with BioMedRAG’s trained chunk scorer.
Accurate extraction of biomedical relations from litera-
ture has direct implications for clinical practice and transla-
tional medicine. Drug-drug interaction detection (DDI) in-
forms prescription safety and adverse event prevention, while
chemical-protein binding relationships (ChemProt) acceler-
ate drug target discovery and repurposing. Gene-disease as-
sociations extracted from biomedical texts support precision
medicine and clinical genomics. By improving retrieval qual-
ity in RAG systems, our semantic chunking framework en-
hances the reliability of automated biomedical knowledge ex-
traction, reducing manual curation burden and enabling real-
time literature-based clinical decision support.
arXiv:2608.31139v1  [cs.CL]  31 Aug 2026

We evaluate our framework on four biomedical bench-
marks from the BioMedRAG repository [Liet al., 2025 ]:
GM-CIHT, DDI, and ChemProt for triple extraction, and
ADE for adverse event classification. Experimental re-
sults reveal substantial improvements on complex relational
tasks requiring precise evidence localization, while detailed
component-wise analysis characterizes the contribution of
each framework element. Cross-dataset comparison pro-
vides practical insights into when semantic chunking benefits
retrieval-augmented generation systems.
Our key contributions are summarized as follows:
1. We propose aconfiguration-driven semantic chunk-
ing frameworkfor biomedical retrieval-augmented in-
formation extraction. The framework replaces fixed-
size windows with entity-preserving, trigger-aware, and
proposition-oriented evidence candidates.
2. We integrate the proposed chunking framework into
BioMedRAG while keeping the embedding model,
learned chunk scorer, generator, preprocessing pipeline,
and evaluation protocol unchanged. This controlled de-
sign isolates the effect of chunk construction from other
system components.
3. We conduct an empirical evaluation on four biomedical
benchmarks, including GM-CIHT, DDI, ChemProt, and
ADE, and show that semantic chunking is most effec-
tive for extraction tasks with explicit relation cues and
moderate entity density.
4. We provide component-wise ablation studies, including
the effects of entity-aware segmentation, proposition ex-
traction, trigger prioritization, relation hierarchy resolu-
tion, ranking strategy, and NER-based entity detection.
2 Related Work
2.1 Retrieval-Augmented Generation
Large language models encode knowledge parametrically
during training, but this knowledge is inherently limited: it
becomes outdated, lacks domain-specific coverage, and can
lead to hallucinated outputs [Lewiset al., 2020 ]. Retrieval-
augmented generation (RAG) mitigates these limitations by
conditioning generation on externally retrieved evidence.
Typical RAG pipelines operate in two stages: a retriever iden-
tifies relevant passages from a knowledge corpus, and a gen-
erator produces predictions conditioned on the retrieved evi-
dence, improving factual accuracy and interpretability [Lewis
et al., 2020 ].
Originally proposed for open-domain question answer-
ing[Lewiset al., 2020 ], RAG has since been applied to struc-
tured tasks such as knowledge base completion, dialogue sys-
tems, and code generation [Shusteret al., 2021; Zhouet al.,
2023 ]. In biomedical NLP, accessing current literature is crit-
ical for accurate information extraction. BioMedRAG [Liet
al., 2025 ]applies this paradigm to biomedical relation extrac-
tion and is the system we extend; we describe its pipeline in
Section 3.1.2.2 Text Chunking for Retrieval
Text chunking strategies strongly influence retrieval quality in
RAG systems [Gaoet al., 2023 ]. Fixed-size chunking splits
text into uniform windows, while sliding windows add over-
lap to preserve context. Recent work has investigated opti-
mal retrieval granularity, comparing document, passage, sen-
tence, and proposition-level units [Chenet al., 2024 ]. How-
ever, these methods are designed for multi-sentence docu-
ments, making them ill-suited to biomedical relation extrac-
tion tasks, where relations are typically expressed within sin-
gle sentences of 15-30 words.
Sub-sentence chunking methods aim to capture finer-
grained semantics than sentence or paragraph-level segmen-
tation. Proposition-based approaches decompose sentences
into atomic, self-contained units for retrieval [Chenet al.,
2024; Hosseiniet al., 2024 ]. Other work explores contex-
tual chunk embeddings using long-context models [G¨unther
et al., 2024 ]. These methods, however, are largely domain-
agnostic and do not enforce biomedical entity preservation or
explicit modeling of relation triggers.
2.3 Biomedical Information Extraction
Biomedical relation extraction aims to identify structured
relationships between entities in scientific literature [Zhou
et al., 2014 ]. Traditional approaches employ supervised
learning with manually designed linguistic and seman-
tic features [Fundelet al., 2007 ]or neural architectures
such as CNNs [Zenget al., 2014 ], LSTMs [Zhouet
al., 2016 ], and Graph Neural Networks [Zhu and others,
2019 ]. BioMedRAG [Liet al., 2025 ]demonstrates that ev-
idence quality critically influences extraction performance in
retrieval-augmented biomedical systems.
3 Method
3.1 Preliminaries: The BioMedRAG Pipeline
We briefly review BioMedRAG [Liet al., 2025 ], the retrieval-
augmented extraction framework our approach extends.
BioMedRAG performs biomedical information extraction
through retrieval-augmented generation. For each input sen-
tencex, it retrieves the top-krelevant evidence chunks from
a corpusDand generates triples conditioned on bothxand
the retrieved evidence [Liet al., 2025 ]. Candidate chunks are
stored in a relational key–value memory, retrieved by embed-
ding similarity, and re-ranked by a chunk scorer trained to
prefer evidence that improves downstream prediction. Re-
trieval quality therefore depends fundamentally on chunk
quality.
BioMedRAG constructs its chunk database by splitting
each sentence into fixed 5-word windows with no overlap:
Cfixed(x) ={(w i, . . . , w i+4)|i= 1,6,11, . . .},(1)
which ignores semantic structure. This strategy frequently
fragments entities, for instance, the hyphenated compound
alpha-ketoglutaratemay be split intoalpha-andketoglu-
tarateacross separate 5-word windows, and breaks relation-
bearing propositions across chunk boundaries, resulting in in-
complete or misaligned evidence for the language model. Our

framework replaces this chunk-construction stage; the em-
bedding model, learned chunk scorer, generator, and evalu-
ation protocol remain unchanged.
3.2 Problem Formulation
We address two biomedical information extraction tasks. The
primary task istriple extraction: given a sentencex=
(w1, . . . , w n), extract structured relations(h, r, t)wherehis
the subject entity (head),r∈ Ris a relation type, andtis
the object entity (tail). For example, from“aspirin inhibits
prostaglandin synthesis”, we extract⟨ASPIRIN,INHIBITS,
PROSTAGLANDIN⟩. The secondary task isrelation classifi-
cation: given a sentence with marked entity spans, predict
the relation type between them. Our chunking framework
addresses both tasks, as both require identifying relation-
bearing evidence within sentences.
3.3 Configurable Semantic Chunking Framework
Our framework mitigates the fragmentation introduced by
fixed-width chunking by constructing a hybrid candidate pool
and selecting evidence with an explicit bias toward struc-
turally complete relation spans.
Stage 1: Multi-Source Candidate Generation.For each
sentence, we generate candidate chunks from three sources.
(i)Entity-aware sliding windowspreserve biomedical en-
tity boundaries and capture relation-trigger context (Sec-
tions 3.4 and 3.5). (ii)Proposition-first extractionisolates
minimal subject–trigger–object spans (Section 3.8); overlap-
ping propositions are internally prioritized using tiered trig-
ger specificity and relation hierarchy (Sections 3.6 and 3.7).
(iii)Fixed-width fallback windows(the original 5-word splits)
are added to ensure non-empty coverage when entity/trigger
signals are weak.
Stage 2: Similarity-Based Selection with Proposition Bias.
Candidates are ranked by cosine similarity between chunk
embeddings and the target relation definition, consistent with
the embedding-based retrieval step in BioMedRAG [Liet al.,
2025 ]. We then apply aproposition bias: if at least one
proposition-based candidate appears among the top-Nranked
candidates, we promote the best such proposition into the fi-
nal top-kset. This guarantees that the downstream genera-
tor receives at least one chunk with verified subject–trigger–
object structure, while retaining high-similarity contextual
chunks for coverage. The selected chunks are finally passed
to BioMedRAG’s trained chunk scorer for re-ranking.
Configurability.All dataset-specific knowledge, including
trigger vocabularies, tier definitions, relation hierarchies,
negation rules, and context markers, is externalized in JSON
configuration files. This design makes the chunking decisions
explicit and inspectable, enabling adaptation to new biomed-
ical relation sets without code changes.
Shared Trigger Lexicon.Both entity-aware windows and
proposition extraction use a common tiered trigger vocabu-
lary for relation signal detection, ensuring consistent identifi-
cation of relation-bearing spans across candidate types.3.4 Entity-Aware Segmentation
This subsection describes the first candidate source in
Stage 1: entity-aware sliding windows.
Biomedical named entities (drugs, proteins, diseases) fre-
quently span multiple tokens and often include hyphenation
(e.g.,alpha-ketoglutarate,acetyl-CoA carboxylase). Fixed-
width chunking can cut through such entities, producing par-
tial strings that degrade retrieval similarity and downstream
relation evidence.
Entity Detection.We detect entity spans using lightweight
pattern rules: bracketed markup (e.g.,[aspirin]), capitalized
multi-token terms (e.g.,Tumor Necrosis Factor), and hyphen-
ated compounds (e.g.,alpha-ketoglutarate). Each entityeis
represented as a character span(e start, eend).
Scope of Entity Detection.This step is intended as a
boundary-preservation mechanism rather than as a complete
biomedical named entity recognition module. The detected
spans are used only to adjust chunk boundaries and construct
proposition candidates; final relation prediction is still per-
formed by the BioMedRAG scorer and generator. This design
favors low overhead and direct integration with the existing
pipeline, but it may miss lowercase or irregular biomedical
mentions. To examine this limitation, we compare pattern-
based entity spans with NER-based spans in the ablation
study.
Sliding Window Adjustment.Given a window sizewand
strides, we adjust window boundaries whenever a boundary
falls inside an entity span: (i) if the window end intersects an
entity, we extend it to include the full entity; (ii) if the window
start intersects an entity, we shift the start past the entity. This
guarantees that no entity is split across chunks.
Example.Consider“Tumor necrosis factor alpha receptor-
complex regulates immune responses. ”With fixed 5-token
windows, one chunk may end at“... alpha receptor-”while
the next begins with“complex ... ”, splitting the hyphenated
entityreceptor-complex. Our entity-aware approach moves
the boundary so thattumor necrosis factor alpha receptor-
complexremains intact within a single chunk.
3.5 Relation Trigger Detection
Entity-aware segmentation preserves entity boundaries but
remains agnostic to whether a chunk actually expresses a se-
mantic relation. We therefore augment the sliding window
strategy with trigger-based detection to emphasize relation-
bearing content.
Trigger Lexicon Construction.For each dataset, candi-
date triggers are collected from the training split by iden-
tifying words and short phrases that occur between or near
annotated entity pairs. These candidates are grouped by
their associated gold relation labels and inspected for rela-
tion specificity. Terms that frequently co-occur with a single
relation are retained as stronger triggers, whereas terms ap-
pearing across several relation types are assigned to lower
tiers or removed when they provide limited discriminative
value. Morphological variants expressing the same cue are
merged, for example, “inhibit”, “inhibits”, “inhibited”, and
“inhibition”. For instance, the INHIBITSrelation may include

triggers such as “inhibits”, “blocks”, “suppresses”, “antago-
nizes”, “downregulates”, and “attenuates”, depending on the
dataset-specific relation definition. If no trigger is detected,
the framework falls back to entity-aware sliding windows and
fixed-width chunks to preserve coverage.
Trigger-Centered Chunking.When a trigger is detected,
we generate a local context window of±4words around it to
capture nearby entities and modifiers. For example, in“As-
pirin effectively inhibits prostaglandin synthesis in tissues, ”
the triggerinhibitsproduces a chunk that captures both the
agent (Aspirin) and the target (prostaglandin synthesis). The
window size is intentionally small to emphasize local rela-
tional evidence.
Limitation.This approach treats all triggers equally, re-
gardless of semantic specificity. Consequently, a chunk con-
taining a generic term such asaffectsreceives the same pri-
ority as one containing a precise term such asinhibits. This
motivates tiered trigger prioritization, introduced next (Sec-
tion 3.6).
3.6 Tiered Trigger Prioritization
Not all relation triggers carry equal semantic weight. Generic
terms such asaffectsormodulatesare ambiguous, while spe-
cific terms likeinhibitsortherapyconvey precise mechanistic
or clinical meaning. We therefore organize triggers into three
tiers based on semantic specificity and directional clarity, fol-
lowing common distinctions in biomedical relation extraction
between explicit actions, contextualized actions, and generic
associations.
Trigger Tier Definitions.For each relation type, we define:
•Primary triggers(tier bonus 3): Highly specific terms
that unambiguously indicate the relation (e.g.,inhibits,
therapy,activates).
•Secondary triggers(tier bonus 2): Moderately specific
synonyms or related terms that express the relation with
reduced precision (e.g.,blocks,prescribed,enhances).
•Tertiary triggers(tier bonus 1): Generic or weaker sig-
nals that may indicate a relation but lack sufficient speci-
ficity when used in isolation (e.g.,reduces,medication,
associated).
Example.ForTREATS, primary triggers includetreats,
treatment, andtherapy; secondary triggers includepre-
scribed,administered, andtherapeutic; and tertiary triggers
includemedicationanddrug. The sentence“Aspirin ther-
apy alleviates headache”contains a primary trigger (ther-
apy), while“Aspirin medication alleviates headache”con-
tains only a tertiary trigger (medication).
Usage.Tier membership is used during proposition-first ex-
traction (Section 3.8) to rank overlapping candidate proposi-
tions. When multiple propositions cover the same text span,
the proposition containing a higher-tier trigger is retained, en-
suring that the most semantically precise relational evidence
is passed to the selection stage.Relation Priority weight
TREATS4
INHIBITS,STIMULATES,CAUSES,PREVENTS3
AFFECTS,INTERACTS WITH,REDUCES2
COEXISTS WITH1
Table 1: Relation hierarchy for GM-CIHT. Higher weights indicate
higher precedence during proposition ranking. Weights were tuned
on the development set; alternative orderings yielded lower F1.
Tier Assignment.Triggers were assigned to tiers based on
two criteria: (i) corpus frequency analysis of trigger–relation
co-occurrence in training data, where high co-occurrence
with a single relation indicates specificity, and (ii) linguis-
tic directness, where verb forms (e.g.,inhibits) are preferred
over nominalizations (e.g.,inhibition) and generic associa-
tions (e.g.,affects). Tier assignments were validated on the
development set (Table 2) by comparing proposition extrac-
tion precision across alternative classifications.
3.7 Relation Hierarchy Resolution
Building on tiered trigger prioritization, we address a sec-
ond source of ambiguity: sentences that express multiple
biomedical relations simultaneously. For example,“aspirin
treats inflammation by inhibiting COX-2”contains both a
TREATSrelation (aspirin–inflammation) and anINHIBITSre-
lation (aspirin–COX-2). Without explicit disambiguation, ex-
tracted propositions may emphasize relations that are less in-
formative for the target extraction objective.
We resolve this ambiguity through configurable relation hi-
erarchies that encode dataset- and task-specific priorities. For
GM-CIHT, we define the hierarchy shown in Table 1.
Priority Assignment.Relation weights reflect two explicit
criteria: (i)semantic specificity, indicating how directly a re-
lation encodes an actionable interaction between entities, and
(ii)task relevance, reflecting the importance of the relation
for the target extraction objective. In GM-CIHT,TREATS
receives the highest weight as the dataset emphasizes ther-
apeutic relationships. Directional mechanistic relations such
asINHIBITSandSTIMULATESare informative but secondary,
whileCOEXISTS WITHdenotes weak associative evidence.
Hierarchy Tuning.Relation weights were determined
through grid search on the development set (Table 2): we
evaluated all permutations of relation orderings and selected
the configuration maximizing extraction F1. The reported hi-
erarchy in Table 1 outperformed flat (equal-weight) and in-
verted orderings by 1.2–2.1% F1. For other datasets, relation
weights are specified through dataset-specific configuration
files using the same principle, while the core chunking algo-
rithm and selection strategy remain unchanged.
Example.Consider“Metformin treatment reduces glucose
levels by activating the AMPK pathway in diabetic patients. ”
This sentence supports multiple relations:
•TREATS:treatment(primary trigger), clinical context
(diabetic patients)
•STIMULATES:activating(primary trigger)

•AFFECTS:reduces(secondary trigger)
Although all relations are supported by valid triggers, the hi-
erarchy ensures that the proposition emphasizing the thera-
peutic relation (metformin–diabetes) is retained over mecha-
nistic or associative alternatives.
Combined Disambiguation.Relation hierarchy resolu-
tion operates jointly with tiered trigger prioritization dur-
ing proposition-first extraction (Section 3.8). When multi-
ple propositions are extracted from the same sentence, they
are ranked lexicographically: propositions associated with
higher-priority relations are preferred, and ties are resolved
by trigger tier strength. Only non-overlapping propositions
with the highest precedence are retained.
3.8 Proposition-First Extraction
Sliding windows may capture entity context or relation con-
text separately, but fail to preserve complete relational evi-
dence in a single chunk. For example,“Metformin, a widely
used antidiabetic, improves insulin sensitivity in diabetic
patients”may yield windows containing the entity (“Met-
formin...antidiabetic”) or the relation (“improves insulin sen-
sitivity”), but neither forms a complete triple.
We address this throughproposition-first extraction, in-
spired by [Hosseiniet al., 2024 ]. We extract candidate chunks
corresponding to atomic propositions: self-contained units
containing a subject, a relation trigger, and an object.
Proposition Patterns.We detect propositions using three
syntactic patterns:
•Infix: Entity 1–TRIGGER– Entity 2(e.g.,“Aspirin in-
hibits prostaglandins”);
•Prefix:TRIGGER– Entity 1. . . Entity 2(e.g.,“Treatment
of diabetes with metformin”);
•Postfix: Entity 1. . . Entity 2–TRIGGER(e.g.,“Aspirin–
COX-2 interaction”).
For each trigger, we identify nearest entities within a bounded
context (±25tokens) and extract the minimal span covering
subject, trigger, and object. We expand each span by±3to-
kens to capture adjacent negation markers (e.g.,does not,fails
to) and uncertainty hedges (e.g.,may,possibly) that can invert
or weaken the expressed relation, ensuring the downstream
model receives complete polarity information.
Pattern Coverage.These patterns cover the majority of
explicit relation expressions in biomedical corpora. Com-
plex constructions such as nested relations or coordinated ar-
guments are handled implicitly: sliding window candidates
provide fallback coverage when proposition patterns do not
match.
Example.Consider the sentence:“Aspirin therapy reduces
inflammation by inhibiting COX-2 expression. ”Two proposi-
tions are extracted:
1.“Aspirin therapy reduces inflammation”- Trigger:re-
duces(tertiary forAFFECTS), Priority:2 + 1 = 3
2.“Aspirin inhibiting COX-2”- Trigger:inhibiting(pri-
mary forINHIBITS), Priority:3 + 3 = 6The second proposition is ranked higher due to its stronger
trigger and higher-priority relation, ensuring precise mecha-
nistic evidence is retained.
Passive Handling.Passive constructions (e.g.,“treated
by”) reverse surface entity order. We detect passive triggers
followed by markers (by,with) and invert semantic roles ac-
cordingly.
Ranking and Integration.Overlapping propositions are
deduplicated by retaining those with the highest combined
priority:
P(p) =Weighthierarchy (r) +Bonus tier(t)(2)
whereris the detected relation type,tis the trigger
word, Weighthierarchy (r)∈ {1,2,3,4}follows Table 1, and
Bonus tier(t)∈ {3,2,1}for primary, secondary, and tertiary
triggers respectively. This formula integrates linguistic con-
fidence (trigger specificity) with clinical importance (relation
hierarchy) into a unified ranking score.
Proposition chunks complement sliding windows: proposi-
tions ensure structural completeness while windows preserve
broader context. Both candidate types are passed to the hy-
brid selection stage (Section 3.9).
3.9 Hybrid Selection Strategy
The previous stages produce a diverse pool of candidates:
entity-aware sliding windows, proposition spans (ranked us-
ing tiered triggers and relation hierarchy), and fixed-width
fallback windows. We employ a hybrid strategy to select the
final top-kchunks, integrating semantic scoring with struc-
tural prioritization.
Scoring.Following BioMedRAG [Liet al., 2025 ], all can-
didates are scored using cosine similarity between their aver-
aged token embeddings and the target relation definition em-
bedding.
Structural Prioritization via Proposition Bias.Embed-
ding similarity alone may favor verbose chunks over precise
propositions. To ensure that the contributions oftiered trig-
gersandrelation hierarchy(Sections 3.6–3.7) propagate to
the final selection, we apply aproposition bias:
1.Slot 1 (Semantic Best):The highest-similarity candi-
date is selected for broad context.
2.Slot 2 (Structural Best):If a proposition candidate, al-
ready ranked by combined priorityP(p)(Equation 2),
exists in the top-N(N=5), it is promoted to this slot.
This ensures that chunks containing high-tier triggers
and high-priority relations are retained even if their em-
bedding score is not the highest.
3.Fallback:Otherwise, the second-highest similarity
chunk is selected.
Design Rationale.Tiered triggers and relation hierarchy
are applied during proposition extraction rather than entity-
aware window generation. This reflects their distinct roles:
entity-aware windows providebroad contextual coverage,
capturing surrounding entities and modifiers that inform the
language model even without explicit relation signals. Propo-
sitions, in contrast, targetprecise relational evidencewhere

Dataset Task Rels Train Dev Test
GM-CIHT Extraction 22 3,734 492 465
DDI Extraction 4 1,027 258 1,094
ChemProt Extraction 5 4,111 2,411 3,438
ADE Classification 2 4,000 975 497
Table 2: Dataset statistics.Relsdenotes the number of rela-
tion types;Train,Dev, andTestdenote the number of instances
(sentence-label pairs) in each split. GM-CIHT covers 22 general
biomedical relations (therapeutic, mechanistic, associative). DDI fo-
cuses on drug-drug interactions. ChemProt targets chemical-protein
bindings with subtle semantic distinctions. ADE is binary adverse
event detection.
multiple competing triggers frequently co-occur within mini-
mal spans. Tiered prioritization and hierarchy resolution are
most effective when disambiguating such conflicts, which are
common in propositions but rare in broader windows. Fur-
thermore, applying internal scoring to entity-aware windows
risks over-filtering useful contextual chunks that lack explicit
triggers. The proposition bias ensures structurally complete
evidence (ranked by tiers and hierarchy) reaches final selec-
tion, while entity-aware windows contribute complementary
context ranked by semantic similarity alone.
Integration.Selected chunks are passed to BioMedRAG’s
trained chunk scorer for final re-ranking [Liet al., 2025 ],
maintaining full pipeline compatibility.
4 Experiments
4.1 Datasets
We evaluate on four biomedical datasets (Table 2): GM-
CIHT, DDI, ChemProt (triple extraction), and ADE (binary
relation classification). As in the BioMedRAG evaluation set-
ting, we do not consider link prediction, where inputs are too
short to admit meaningful sub-sentence chunking.
GM-CIHT serves as the primary development dataset for
designing the general semantic chunking strategy, includ-
ing entity-aware windows, trigger-tier usage, relation hier-
archy resolution, and proposition bias. For each bench-
mark, dataset-specific symbolic resources such as relation
labels, trigger vocabularies, tier assignments, and hierar-
chy definitions are specified through external configuration
files. We do not otherwise tune the chunk scorer or gener-
ator on DDI, ChemProt, or ADE. All corpora are converted
to a unified JSONL schema (SENTENCE,SUBJECT TEXT,
OBJECT TEXT,PREDICATE; ADE is mapped from its na-
tive format) so chunking, embedding, and generation share a
consistent interface.
4.2 Baselines and Experimental Setup
Baseline.We compare againstBioMedRAG (Fixed) [Liet
al., 2025 ], which uses fixed 5-word chunking. This baseline
is selected to isolate the effect of chunk construction while
keeping the remaining pipeline unchanged, including the em-
bedding model, learned chunk scorer, generator, preprocess-
ing, and evaluation protocol. Thus, the comparison evalu-
ates whether semantically structured evidence units improveDataset Method P R F1
GM-CIHTFixed 74.6 73.8 74.2
Ours 82.6 82.6 82.6
DDIFixed 78.2 78.2 78.2
Ours 79.2 79.2 79.2
ChemProtFixed 87.7 87.0 87.4
Ours 87.0 86.3 86.6
ADEFixed 87.0 90.7 88.8
Ours 86.1 88.6 87.3
Table 3: Main results (%).P,R, andF1denote micro-averaged Pre-
cision, Recall, and F1-score.Fixedis the BioMedRAG fixed 5-word
chunking baseline;Oursis the proposed semantic chunking frame-
work. Our semantic chunking achieves gains on GM-CIHT (+8.4
F1) and DDI (+1.0 F1), while fixed chunking remains competitive
on ChemProt and ADE.
BioMedRAG-style retrieval over fixed-size windows. Com-
parisons with sentence-level chunking, dependency-based
span extraction, and external proposition segmentation meth-
ods are left for future work.
Implementation.We use MedLLaMA-13B for computing
chunk and relation embeddings with 512-token maximum
length and mean-pooling. Following BioMedRAG [Liet al.,
2025 ], we train chunk scorers initialized from Llama-2-13B
with LoRA [Huet al., 2022 ](r=8,α=8, dropout0.1) for
1,000 steps using AdamW with bfloat16 precision. Separate
scorers are trained per dataset.
Chunking Parameters.Entity-aware windows use size
w=8, strides=3, bounds[5,12]tokens. Proposition extrac-
tion searches±25tokens around triggers with±3token con-
text margins. Proposition bias thresholdN=5. These pa-
rameters are kept fixed across datasets to support controlled
comparison; systematic sensitivity analysis of window size,
context range, and proposition bias threshold is left for future
work.
Configuration.The core chunking algorithm, window pa-
rameters, proposition bias threshold, embedding model,
learned scorer, generator, preprocessing pipeline, and evalua-
tion protocol are kept fixed across datasets. Dataset-specific
symbolic resources are provided through external JSON con-
figuration files.
Evaluation.Following BioMedRAG [Liet al., 2025 ], we
report micro-averaged Precision, Recall, and F1 for the ex-
traction tasks GM-CIHT, DDI, and ChemProt. For these
tasks, a prediction is considered correct only if the head span,
relation type, and tail span exactly match the ground truth.
For ADE, we report binary classification performance using
the provided entity spans. All experiments use NVIDIA A40
GPUs with 4-bit NF4 quantization and fixed random seeds.
4.3 Main Results
Table 3 presents results across all four datasets.
Our semantic chunking substantially improves extraction
performance on GM-CIHT (+8.4F1 points: 74.2→82.6) and
DDI (+1.0F1: 78.2→79.2). These datasets contain diverse

Configuration F1∆
Fixed (Baseline) 74.2 —
+ Entity-Aware 76.1 +1.9
+ Proposition 76.3 +0.2
+ Tiered Triggers 78.7 +2.4
+ Hierarchy (Full)82.6+3.9
Table 4: Incremental component ablation on GM-CIHT (%). Each
row adds one component to the configuration above it.F1is the
micro-averaged F1-score;∆is the change relative to the preceding
row.
Method 1 Ex. 3 Ex.∆
Fixed 74.2 75.1 +0.9
Ours 82.6 77.2−5.4
Table 5: Effect of the number of in-context examples on GM-CIHT
(%).1 Ex.and3 Ex.denote micro-averaged F1 with one and three
in-context demonstrations, respectively;∆is the change from one
to three.
relation types where tiered trigger prioritization and proposi-
tion extraction effectively disambiguate competing interpre-
tations. On GM-CIHT, gains in both precision and recall indi-
cate improved evidence quality without sacrificing coverage.
On ChemProt and ADE, fixed chunking remains com-
petitive, outperforming our method by0.8%and1.5%F1
respectively. ChemProt involves fine-grained mechanistic
distinctions (e.g.,AGONISTvs.ACTIVATOR) that benefit
from broader contextual cues preserved by fixed windows,
while ADE is a binary classification task where comprehen-
sive sentence-level context is more informative than targeted
propositions. These results highlight that semantic chunking
is most beneficial in settings with relational ambiguity rather
than uniformly dense supervision, a pattern we analyze fur-
ther in Section 4.5.
4.4 Ablation Studies
We analyze component contributions and in-context learning
sensitivity on GM-CIHT.
Component Ablation.Table 4 shows incremental gains
from each component.
Entity-aware chunking provides a strong foundation (+1.9
F1) by preventing fragmentation. Proposition extraction adds
a small gain (+0.2 F1), and tiered triggers yield a larger step
(+2.4 F1). The relation hierarchy yields the largest incremen-
tal gain (+3.9 F1) by aligning evidence selection with GM-
CIHT’s therapeutic focus, consistently prioritizingTREATS
relations over weaker associations such asCOEXISTS WITH
when multiple interpretations are present.
In-Context Learning Sensitivity.Table 5 examines
prompting requirements.
Fixed chunking improves with additional examples (+0.9
F1), while our method performs best with one example, de-
grading with three (−5.4 F1). This suggests semantically
coherent chunks saturate the model’s contextual needs with
fewer demonstrations, potentially reducing inference costsRanking F1∆
Cosine + proposition bias 82.6 —
70%cosine +30%tiered/hierarchy 79.4−3.2
Table 6: Ranking variant ablation on GM-CIHT (%).∆is the
change in micro-averaged F1 relative to the default ranking (cosine
similarity with proposition bias).
Entity detection F1∆
Regex (default) 82.6 —
NER 81.5−1.1
Table 7: Entity detection variant on GM-CIHT (%).∆is the change
in micro-averaged F1 relative to the default regex-based entity de-
tection.
compared to lower-quality evidence requiring more exam-
ples; the drop with three examples may also reflect context
limits or sensitivity to demonstration choice.
Ranking variant.Besides cosine similarity with proposi-
tion bias (Section 3.9), we evaluate a ranking that mixes70%
cosine similarity with30%of a structural score derived from
tiered triggers and relation hierarchy. Table 6 shows that this
mixed ranking reduces GM-CIHT F1 to79.4%(−3.2vs. co-
sine with proposition bias at82.6%). Proposition bias there-
fore appears sufficient to leverage structural signals: promot-
ing a proposition into the final top-kwhen it appears among
the top candidates by cosine avoids diluting or mis-scaling
the semantic similarity signal; the70/30weighting may also
be suboptimal or misaligned between score scales.
Entity detection: regex vs. NER.Entity spans in proposi-
tion extraction use the same lightweight pattern rules as Sec-
tion 3.4. We replace them with a pre-trained NER system
for boundaries, keeping the same ranking (cosine similarity
with proposition bias). Table 7 reports GM-CIHT F1: NER
reaches81.5%,−1.1below regex (82.6%). GM-CIHT of-
ten contains explicit markers, capitalization, and hyphenation
that regex handles reliably; NER can introduce false positives
or boundary errors that hurt minimal-span proposition extrac-
tion. NER may still help on corpora with less regular surface
forms; the accuracy-latency trade-off also matters for deploy-
ment.
4.5 Performance of Semantic Chunking in
Different Settings
We analyze dataset-dependent performance patterns to under-
stand when semantic chunking is most effective.
Success on GM-CIHT and DDI.Both datasets contain di-
verse relation types with relatively clear semantic boundaries
(e.g.,TREATSvs.INHIBITSvs.COEXISTS WITH). Tiered
trigger prioritization effectively disambiguates these cate-
gories, while entity-aware segmentation prevents fragmenta-
tion of complex biomedical terms. The observed gains on
GM-CIHT (+8.4 F1) and DDI (+1.0 F1) indicate that seman-
tic chunking is particularly beneficial for extraction tasks with
explicit relation signals and moderate entity spacing.

ChemProt: Entity Density Challenges.ChemProt ex-
hibits high entity density, with multiple chemicals and pro-
teins often appearing within short spans (5–10 tokens).
Entity-aware chunking can group multiple entity pairs into
a single chunk (e.g.,“compound X inhibits protein A and
protein B”), increasing ambiguity during relation assignment.
Moreover, ChemProt requires fine-grained biochemical dis-
tinctions (e.g.,ANTAGONISTvs.DOWNREGULATOR) that
rely on broader contextual cues beyond trigger words. In-
cremental ablations on ChemProt show an inverse pattern to
GM-CIHT: adding tiered triggers and the tuned hierarchy can
hurt F1, consistent with priorities designed for therapeutic re-
lations misaligning with biochemical relation types. In this
setting, fixed chunking may incidentally isolate simpler en-
tity pairs, yielding slightly stronger performance.
ADE: Classification vs. Extraction.ADE is formulated
as a binary classification task rather than structured relation
extraction. Performance therefore depends on broad contex-
tual signals such as symptom descriptions and temporal cues,
rather than precise subject–trigger–object spans. Proposition-
focused chunking may omit such supporting context, whereas
fixed chunking preserves heterogeneous evidence useful for
holistic classification.
Reproducibility.Our BioMedRAG Fixed GM-CIHT F1
(74.2%) is below the F1 reported in the original BioMedRAG
publication (81.42%), likely due to differences in hardware,
hyperparameters (e.g., shorter maximum sequence length
when training the chunk scorer under GPU memory limits),
and our own pipeline integration and dataset format unifica-
tion. The comparison between fixed and semantic chunking
remains meaningful because both conditions use the same
scorer, generator, and preprocessing.
Generalization.Overall, semantic chunking is most effec-
tive when (i) relation types are coarse-grained and trigger-
explicit, (ii) entity density is moderate, and (iii) the task em-
phasizes structured extraction over classification. Because
our evaluation intentionally follows the four BioMedRAG
benchmarks, these conclusions are best interpreted for
sentence-level biomedical relation extraction over published
literature rather than for clinical notes or document-level
retrieval. The configuration-driven design improves trans-
parency and reuse, but it still requires task-specific symbolic
resources, such as trigger lexicons and relation hierarchies,
which may limit scalability when transferring to many un-
seen biomedical domains. High-density corpora and longer-
context settings may benefit from future extensions incorpo-
rating adaptive entity-pair isolation, pair-specific proposition
selection, automatic trigger induction, or variable chunk gran-
ularity.
4.6 Error Analysis
We categorize the main failure modes of the proposed seman-
tic chunking framework according to their likely root causes.
First, errors occur when a relation is expressed without an
explicit lexical trigger. In such cases, proposition-first ex-
traction may not construct a complete subject–trigger–object
span, and the framework relies on entity-aware or fixed fall-
back chunks. Second, high entity density can increase am-biguity, particularly in ChemProt, where multiple chemicals
and proteins may appear within a short sentence. This makes
it difficult to assign the correct relation to the correct en-
tity pair. Third, fine-grained biochemical relations may re-
quire broader mechanistic context than a minimal proposi-
tion span provides. Relations such as AGONIST, ACTIVATOR,
and DOWNREGULATORcan share similar surface cues while
differing in biological meaning. Finally, ADE differs from
the extraction datasets because binary adverse-event clas-
sification often depends on sentence-level context, includ-
ing symptoms, temporal expressions, speculation, and nega-
tion. These failure modes explain why semantic chunking
improves datasets with explicit relation cues and moderate
entity density, while fixed-size chunking remains competitive
for dense extraction and classification settings.
5 Conclusion
Fixed-size chunking in retrieval-augmented biomedical re-
lation extraction frequently fragments entities and splits
relation-bearing propositions, degrading the quality of re-
trieved evidence for downstream generation.
We introduced a configurable semantic chunking frame-
work that combines entity-aware sliding windows with
proposition-first extraction. Tiered trigger prioritization or-
ganizes relation signals by semantic specificity, while relation
hierarchy resolution disambiguates competing interpretations
by encoding task-specific priorities. All components are ex-
ternalized in configuration files, enabling adaptation to new
biomedical domains without code modification.
Our approach yields substantial gains on extraction-
focused benchmarks, achieving +8.4 F1 points on GM-CIHT
(82.6%) and +1.0 F1 on DDI (79.2%). Ablation studies show
that relation hierarchy resolution contributes the largest in-
cremental gain in the GM-CIHT stack (+3.9 F1), and that
semantically coherent chunks peak with a single in-context
example (Table 5). The ranking ablation (Table 6) shows that
proposition bias outperforms mixing structural scores into co-
sine ranking; the NER ablation (Table 7) shows that pattern-
based entity spans match or beat NER on GM-CIHT under
our setup.
Absolute F1 for the fixed baseline is below published
BioMedRAG figures, but the fixed-versus-semantic compar-
ison is conducted under identical training and preprocessing
and remains the primary evidence for our claims.
Analysis reveals that semantic chunking is most effective
for datasets with coarse-grained relations and moderate entity
density. In contrast, for high-density corpora (ChemProt) or
classification-oriented tasks (ADE), fixed-size chunking re-
mains competitive, as entity preservation may group multiple
targets and proposition extraction can over-constrain contex-
tual evidence.
Future work includes adaptive chunking strategies that re-
spond to entity density, dataset-specific hierarchy design (or
disabling hierarchy) for fine-grained biochemistry tasks, ex-
tension to clinical notes and drug labels, and entity-pair-
specific isolation mechanisms to better handle multi-target
scenarios. Future releases of the integration code, configu-
ration files, and unified data conversion scripts will further

support reproducibility and facilitate comparison with future
BioMedRAG-based systems.
Code and Data Availability
All datasets used in this study are publicly available through
the BioMedRAG benchmark setting and the original dataset
sources. To support reproducibility, we will release the se-
mantic chunking implementation, dataset configuration files,
preprocessing scripts, unified JSONL conversion format, and
evaluation instructions upon publication. The configuration
files include trigger vocabularies, tier assignments, relation
hierarchies, negation rules, context markers, and chunking
parameters. These resources are intended to reproduce the
fixed-size and semantic chunking comparisons reported in
this work under the same BioMedRAG scorer and generator
pipeline.
Acknowledgements
Funded by the Deutsche Forschungsgemeinschaft (DFG,
German Research Foundation) – 527049502.
References
[Chenet al., 2024 ]Tong Chen, Hongwei Wang, Sihao Chen,
Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang,
and Dong Yu. Dense X retrieval: What retrieval
granularity should we use? InProceedings of the
2024 Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP), pages 15159–15177. As-
sociation for Computational Linguistics, 2024. doi:
10.18653/v1/2024.emnlp-main.845. Available at https://
aclanthology.org/2024.emnlp-main.845/.
[Fundelet al., 2007 ]Katrin Fundel, Robert K ¨uffner, and
Ralf Zimmer. RelEx—relation extraction using depen-
dency parse trees.Bioinformatics, 23(3):365–371, 2007.
[Gaoet al., 2023 ]Yunfan Gao, Yun Xiong, Xinyu Gao,
Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, and Haofen Wang. Retrieval-augmented generation
for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2023.
[Grouin and Grabar, 2024 ]Cyril Grouin and Natalia Grabar.
Year 2023 in biomedical natural language processing: A
tribute to large language models and generative AI.Year-
book of Medical Informatics, 33(1):241–248, 2024.
[G¨untheret al., 2024 ]Michael G ¨unther, Isabelle Mohr,
Daniel James Williams, Bo Wang, and Han Xiao.
Late chunking: Contextual chunk embeddings
using long-context embedding models.arXiv
preprint arXiv:2409.04701, 2024. Available at
https://arxiv.org/abs/2409.04701.
[Hosseiniet al., 2024 ]Mohammad Javad Hosseini, Yang
Gao, Tim Baumg ¨artner, Alex Fabrikant, and Reinald Kim
Amplayo. Scalable and domain-general abstractive propo-
sition segmentation.arXiv preprint arXiv:2406.19803,
2024.[Huet al., 2022 ]Edward J. Hu, Yelong Shen, Phillip Wallis,
Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large
language models. InInternational Conference on Learn-
ing Representations (ICLR), 2022.
[Lewiset al., 2020 ]Patrick Lewis, Ethan Perez, Aleksan-
dra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen tau Yih,
Tim Rockt ¨aschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive
NLP tasks. arXiv preprint arXiv:2005.11401, 2020. Also
published in NeurIPS 2020.
[Liet al., 2025 ]Mingchen Li, Halil Kilicoglu, Hua Xu,
and Rui Zhang. BiomedRAG: A retrieval aug-
mented large language model for biomedicine.Jour-
nal of Biomedical Informatics, 162:104769, 2025. doi:
10.1016/j.jbi.2024.104769. Available at https://pubmed.
ncbi.nlm.nih.gov/39814274/.
[National Library of Medicine, 2026 ]National Library of
Medicine. PubMed. https://pubmed.ncbi.nlm.nih.gov,
2026. PubMed comprises more than 39 million citations
for biomedical literature.
[Shusteret al., 2021 ]Kurt Shuster, Spencer Poff, Moya
Chen, Douwe Kiela, and Jason Weston. Retrieval aug-
mentation reduces hallucination in conversation. InFind-
ings of the Association for Computational Linguistics:
EMNLP 2021, pages 3784–3803, 2021. Available at https:
//aclanthology.org/2021.findings-emnlp.320/.
[Zenget al., 2014 ]Daojian Zeng, Kang Liu, Siwei Lai,
Guangyou Zhou, and Jun Zhao. Relation classification
via convolutional deep neural network. InProceedings of
COLING 2014, pages 2335–2344, 2014.
[Zhouet al., 2014 ]Deyu Zhou, Dingcheng Zhong, and Yu-
lan He. Biomedical relation extraction: From binary to
complex.Computational and Mathematical Methods in
Medicine, 2014:298473, 2014.
[Zhouet al., 2016 ]Peng Zhou, Wei Shi, Jun Tian, Zhenyu
Qi, Bingchen Li, Hongwei Hao, and Bo Xu. Attention-
based bidirectional long short-term memory networks for
relation classification. InProceedings of ACL 2016, pages
207–212, 2016.
[Zhouet al., 2023 ]Shuyan Zhou, Uri Alon, Frank F. Xu,
Zhiruo Wang, Zhengbao Jiang, and Graham Neubig.
DocPrompting: Generating code by retrieving the docs.
InInternational Conference on Learning Representations
(ICLR), 2023. Available at https://arxiv.org/abs/2207.
05987.
[Zhu and others, 2019 ]Yuhao Zhu et al. Graph neural net-
works with generated parameters for relation extraction.
InProceedings of ACL 2019, pages 1331–1339, 2019.