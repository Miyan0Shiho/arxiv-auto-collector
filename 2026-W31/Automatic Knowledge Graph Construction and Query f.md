# Automatic Knowledge Graph Construction and Query for Earthquake Catalogs

**Authors**: Yuxin Zhou, Huai Zhang, S. Mostafa Mousavi

**Published**: 2026-07-27 18:39:05

**PDF URL**: [https://arxiv.org/pdf/2607.24984v1](https://arxiv.org/pdf/2607.24984v1)

## Abstract
In recent years, the number of events in earthquake catalogs has significantly increased due to the utilization of more effective deep learning based detectors and phase pickers but answering open ended questions such as what characterizes this sequence? remains constrained by rigid spatiotemporal windowing and subjective expert interpretation. We present the first systematic application of graph based retrieval augmented generation GraphRAG directly to raw, tabular catalog records across three independently featured catalogs, a reservoir adjacent swarm, the 2019 Ridgecrest tectonic sequence, and the 2021 Maduo Mw7.4 aftershock sequence. Without the need for manual data structuring, the pipeline builds structurally complete, queryable knowledge graphs for all three. Rigorous evaluation individually verified against catalog derived ground truth and a rule based reference graph exposes failure modes, and four seismology informed prompt fixes eliminate all targeted fabrications while sharply improving mechanism reasoning. A vector RAG baseline demonstrates the graph layers distinctive value, catalog wide summarization and temporal stage comparison. In addition, we have identified two main pitfalls that need attention. GraphRAG thus offers a practical, transferable, near zero cost query interface for earthquake catalogs, where careful prompting ensures the results are consistently accurate and trustworthy.

## Full Text


<!-- PDF content starts -->

Automatic Knowledge Graph Construction and
Query for Earthquake Catalogs
Yuxin Zhou1,2
, Huai Zhang*1
,andS.Mostafa Mousavi2
Abstract
In recent years, the number of events in earthquake catalogs has significantly increased
due to the utilization of more effective deep-learning-based detectors and phase pickers but
answering open-ended questions such as “what characterizes this sequence?” remains con-
strained by rigid spatiotemporal windowing and subjective expert interpretation. We present
the first systematic application of graph-based retrieval-augmented generation (GraphRAG)
directly to raw, tabular catalog records across three independently featured catalogs: a
reservoir-adjacent swarm (Qiaojia-Dongchuan), the 2019 Ridgecrest tectonic sequence, and
the 2021 Maduo Mw 7.4 aftershock sequence. Without the need for manual data structuring,
the pipeline builds structurally complete, queryable knowledge graphs for all three. Rigorous
evaluation — 1,200 answers individually verified against catalog-derived ground truth and a
rule-based reference graph — exposes failure modes, and four seismology-informed prompt
fixes eliminate all targeted fabrications while sharply improving mechanism reasoning (up
to 2.90/3). A vector-RAG baseline demonstrates the graph layer’s distinctive value: catalog-
wide summarization and temporal-stage comparison. In addition, we have identified two main
pitfalls that need attention. GraphRAG thus offers a practical, transferable, near-zero-cost
query interface for earthquake catalogs, where careful prompting ensures the results are
consistently accurate and trustworthy.
Keywords:knowledge graph; retrieval-augmented generation; GraphRAG; earthquake cata-
log; large language model; benchmark evaluationCite this article asZhou, Y ., H.
Zhang, and S.M. Mousavi (2026).
Automatic Knowledge Graph
Construction and Query for
Earthquake Catalogs,The Seismic
Record0(0), 1–10,
doi: 00.0000/000000000.
Supplemental Material
Introduction
Dense-arraymonitoring(Rossetal.,2019;Shelly,2020)and
deep-learning phase detection/picking (Zhu and Beroza,
2019; Mousavi et al., 2020; Ross et al., 2018; Mousavi and
Beroza, 2022) have together driven an order-of-magnitude
expansion in earthquake-catalog size, for example, the
1. State Key Laboratory of Earth System Numerical Modeling and Application,
College of Earth and Planetary Sciences, University of Chinese Academy
of Sciences, Beijing 100049, China,
 https://orcid.org/0009-0000-2877-9173
(FA)
 https://orcid.org/0000-0003-0411-4841 (SA) 2. Department of Earth
and Planetary Sciences, Harvard University, Cambridge, MA 02138, USA,
https://orcid.org/0009-0000-2877-9173 (FA)
 https://orcid.org/0000-0001-
5091-5370 (TA)
*Corresponding author: H. Zhang, hzhang@ucas.ac.cn
© 2026. The Authors. This is an open access article distributed under the terms
of the CC-BY license, which permits unrestricted use, distribution, and
reproduction in any medium, provided the original work is properly cited.2021 Maduo𝑀𝑤7.4 aftershock sequence contains over
10,000 relocated events (Guan et al., 2024). Beyond scale,
a deeper challenge is characterizing what high-resolution
catalogs represent physically: mainshock-aftershock decay
(Gutenberg and Richter, 1944; Utsu et al., 1995), swarms,
foreshock sequences identifiable only in hindsight, and
induced seismicity (Gupta, 2002) are still distinguished
mainly by coarse qualitative criteria, or point-process mod-
elsfitafterthefact(Ogata,1988),thatdoesnotallowquanti-
tative,reproduciblecross-sequencecomparisoninrealtime.
However, this increase in the typical seizes of earthquake
catalogs has not helped yet to this major challenge. Our
analyses and interpretations of the high-resolution deep-
learningbasedcatalogsremainmoreorlesslimitedtoafew
classicalengineeredfeatures.
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record1
arXiv:2607.24984v1  [physics.geo-ph]  27 Jul 2026

Knowledge graphs organize such semantically linked
informationasreasoning-ready(entity,relationship,entity)
triples(Hoganetal.,2021),andhavebeenappliedtoearth-
quake emergency response (Qiu et al., 2024) and seismic
metadata organization (Davis and Hunt, 2024). Retrieval-
Augmented Generation (Lewis et al., 2020) is an artificial
intelligence architecture that enhances the accuracy and
physical reliability of large language models by grounding
their responses in dynamically retrieved, domain-specific
external data. Microsoft’s GraphRAG (Edge et al., 2024),
rely on RAG systems to automatically extracting enti-
ties/relationships,detectscommunitiesviatheLeidenalgo-
rithm (Traag et al., 2019), and generates natural-language
summaries and query access, with no predefined ontology
– part of a broader effort to unify LLMs and knowledge
graphs (Pan et al., 2024). This raises a natural question:
can GraphRAG be applied directly, end-to-end, to earth-
quake catalogs of varying scale, region, and character to
produce a working natural-language query interface, and
can its most damaging failure modes be suppressed with
modest, seismology-specific engineering effort rather than
afullcustomrebuild?
RAG has recently begun to be applied within seis-
mology and geohazard research specifically. Yao et al.
(2025) combine knowledge-graph construction with a
hybrid RAG strategy for earthquake emergency response,
extracting entities and relationships from thousands of
professional emergency-management documents; Orantes-
Jiménez (2025) use LLMs to build knowledge graphs from
earthquakenewsarticles.Recentreviews(Yuetal.,2025;Li
and Zhou, 2026) note that RAG in the geosciences remains
applied mainly to narrative, prose-style sources – news,
reports, technical literature – rather than to the raw, tab-
ular observational records a catalog itself consists of, and
call for RAG architectures tailored to structured geoscien-
tific data. Our study differs in three respects: it applies
RAGdirectlytostructured,numericalcatalogrecordsrather
than narrative text describing events after the fact; it tar-
getsfullyautomatic,schema-freeconstructionwithnoper-
catalogontologyorextraction-ruleengineering,asopposed
to the hand-designed ontologies underlying prior earth-
quake knowledge-graph work; and it evaluates GraphRAG
specifically, whose automatic community detection and
hierarchical summarization are built to transform thou-
sandsofbriefcatalogrecordsintocomprehensive,sequence-level summaries—a core capability that our benchmark
explicitlyevaluates.
Section 3 demonstrates that fully automatic indexing
and natural-language querying run end-to-end on all three
catalogs under default prompts, establishing mechani-
cal transferability while exposing low baseline answer
quality (catalog-only averages 0.76–1.22/3). Section 4
quantifies the catalog-dependent effect of narrative-text
enrichment. Section 5 describes an iterative, seismology-
oriented prompt-engineering scheme that eliminates the
targeted fabrication modes across all 600 post-fix answers.
Furthermore, this method introduced a baseline vector-
RAG comparison (embedding-similarity retrieval over the
same text chunks, with no graph layer) to quantify the
community-report layer’s impact on holistic questions.
Section 6 details the specific catalog errors identified
during verification—such as the systematic misdating of
the Ridgecrest mainshock and the hallucinated inclusion
of an unlisted Maduo event. We report these issues to
guide safe deployment, rather than as the study’s central
finding. Extended examples and full per-condition bench-
mark results as well as prompt details are provided in the
SupplementaryMaterial(SM).
Materials and Methods
Case Studies
Weusethreeindependentcatalogsdifferinginscale,region,
and origin, totaling 20,027 events (Figure 1; overview table
in SM Table S4): (1) the Qiaojia-Dongchuan relocation cat-
alog, 5,218 events recorded by a temporary dense array
adjacenttotheBaihetanReservoirbetween23Aug2022and
17Mar2023;(2)the2019Ridgecrestsequence,4,188events
(𝑀≥2.0) spanning July 2019, a standard catalog (Shelly,
2020) on a purely tectonic strike-slip system in the Mojave
Desert;and(3)the2021Maduo𝑀 𝑤7.4sequence,thelargest
ofthethreeat10,621events(𝑀≥0.5,1Jun2021–8Jun2023),
relocated by Guan et al. (2024). This aftershocks-only cat-
alog begins after the true 22 May 2021 mainshock and
containsnoMay-2021data.
GraphRAG pipeline and evaluation design
To effectively translate complex seismicity data into a
structured, queryable knowledge base, the system architec-
ture relies on two primary phases: Indexing and Retrieval
(Figure 2a).The indexing phase transforms raw, disparate
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record2

(a) Qiaojia-Dongchuan (n=5,218)
102°30'E 103°00'E 103°30'E26°00'N26°30'N27°00'N27°30'N
0 2 4 6
M(b) Ridgecrest (n=4,188)
117°45'W 117°30'W 117°15'W35°30'N35°45'N36°00'N
0 2 4 6
M(c) Maduo (n=10,621)
98°E 99°E34°N35°N
0 2 4 6
MFigure 1.Spatial distribution of seismic events. (a)
Qiaojia-Dongchuan; (b) Ridgecrest; (c) Maduo. Epicenters colored by
magnitude.
Alttext:Threeside-by-sidescattermapswithlatitude/longitudeaxes,eachtitledwithcatalognameandeventcount.Dotsmarkepicenters,coloredonadark-purple-to-yellow
magnitudescale(0–7)perasmallcolor-barlegendineachpanel’scorner.(a)Qiaojia-Dongchuan:aroughlycircular,denseblobofpointsnear103◦E,27◦N.(b)Ridgecrest:anarrow,
elongateddiagonalbandofpointstrendingNW-SE.(c)Maduo:along,thin,mostlyeast-westlineofpointsspanningabout98–99.5◦E,withtwosmallbluelakeoutlinesintheupper
left.
earthquake catalog entries into a deeply connected net-
work. It begins with LLM-based entity and relationship
extraction (usinggpt-4o-mini) to identify key seismic
features—suchasspecificearthquakes,faultstructures,and
their spatiotemporal links. Next, Leiden community detec-
tiongroupstheseinterconnectedeventsintohighlyrelated,
tectonicorsequence-basedclusters(e.g.,distinctswarmsor
aftershockzones).Thesystemthenprocessestheseclusters
through automated community-report generation, creating
high-level textual summaries that describe the defining
characteristicsofeachlocalizedsequence.
Duringtheretrievalphase,thesystemsupportstwocom-
plementary search modes for interacting with the indexed
seismic data.Global Searchperforms a holistic synthesis
across all community reports, making it ideal for answer-
ing broad, sequence-level questions regarding overarch-
ing migration patterns or aggregate statistics. In contrast,
LocalSearchexecutespreciseentityretrievaltoisolateexact
details concerning specific mainshocks, stations, or local-
ized catalog anomalies. Full parameter configurations for
boththeindexingandretrievalpipelinesaredetailedinthe
SupplementaryMaterial(SM).
Evaluationproceedsinthreestages:(i)capabilitydemon-
stration under default prompts (Section 3); (ii) narrative-
textenrichment,evaluatedwitha100-questionbenchmark
spanning five categories – (A) overall summarization, (B)
precise statistics, (C) local retrieval, (D) physical mecha-
nism, (E) temporal-stage comparison; 20 questions each(Section 4); and (iii) iterative, seismology-oriented prompt
engineeringtargetingtheexposedfailuremodes(Section5).
Two independent ground truths anchor scoring: statistics
computeddirectlyfromeachcatalog’srawrecords,and,for
Qiaojia-Dongchuan, a rule-based knowledge graph (10,766
nodes; 113,620 edges) answering the same questions deter-
ministically. To provide a point of comparison, we identi-
cally scored a standard vector-RAG baseline. This baseline
used the exact same text chunks, embeddings, and LLM as
ourGraphRAGpipeline,butreliedontop-10chunksimilar-
ityforretrievalratherthanagraphlayer
We employed an LLM assistant to score the 1,200
GraphRAGand300baselineresponses(from0to3)against
two ground truths. Following this automated scoring, the
authors manually validated all refusals and Category-B/D
answers against raw catalog statistics, and the first author
resolved all borderline evaluations. To validate the auto-
mated scoring, the first author conducted a blind audit
of 29 stratified-random answers. When adjudicated against
the ground truth, this audit showed 76% exact agreement
with the LLM workflow and 100% agreement within one
point,withzerounresolvedfabricationdisputes.Incontrast,
a purely unaided human review achieved only 52% exact
agreementanderroneouslyacceptedtwofabricatedanswers
as correct, highlighting the critical need for ground-truth
anchoring over subjective human judgment. Responses
weregradedusingthefollowingrubric:ascoreof3indicates
a correct, independently verifiable central claim; a score of
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record3

INDEXING (one-time, fully automatic) RETRIEVAL(interactive
queries)
Earthquake
catalog
(+ narrative
text)T ext
chunkingEntity &
relationship
extraction
(gpt-4o-mini)Leiden
community
detection
Knowledge graph (entities + relationships,
communities colored)Community
reports
(LLM summaries)Global Search
map-reduce over
community reports
Local Search
direct entity
retrievalGrounded
natural-language
answeruser query(a)
1 2 3
4
(b)Figure 2.(a) Schematic of the GraphRAG indexing and retrieval
pipeline. (b) Entity-relationship graph automatically constructed by
GraphRAG from the Qiaojia catalog (5,247 entities; 3,920relationships), with no hand-designed schema; GEO (coordinate/date
tags), EVENT (individual earthquake records), and a small residual of
Unclassified/Organization entities from generic extraction.
Alttext:Two-panelfigure.(a)Flowchart:abluedashedbox‘INDEXING’containsfourlinkedboxes–cataloginput,textchunking,LLMentity/relationshipextraction,Leiden
communitydetection–feedingasmallnode-clustericonlabeled‘Knowledgegraph’anda‘Communityreports’box;anorangedashedbox‘RETRIEVAL’shows‘GlobalSearch’and
‘LocalSearch’boxesbotharrowingintoa‘Groundednatural-languageanswer’box,withauser-queryarrowabove.(b)Adenseforce-directednode-linknetworkonagray
background:smalldotsjoinedbythingrayedges,coloredbylegend–blueGEO,tealEVENT,orangeUnclassified,greenOrganization–withlargetealhubnodesnearthecenter.
2 denotes a correct central claim grounded in the data, but
containingsecondaryerrors,omissions,orunverifiableele-
ments;ascoreof1applieswhenthecentralclaimiswrong
ormissing(e.g.,refusingananswerablequestionormisrep-
resenting data subsets) despite using genuine records; anda score of 0 is reserved for fabricated values, hallucinated
records,orfalseclaimspresentedasfactual.Refusalsearna
3onlyiftherequiredinformationiscompletelyabsentfrom
thecatalog.
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record4

To quantify uncertainty, 95% bootstrap confidence inter-
vals were calculated for the per-answer scores. For indi-
vidual category cells (𝑛=20), the interval half-widths
reached a maximum of 0.66, meaning score differences
below∼0.5 are indistinguishable from scoring noise.
For broader condition averages (𝑛=100), this resolu-
tion tightens to±0.22, establishing a∼0.3 threshold
for significance. Consequently, minor variations—such as
Maduo’s+0.04gain,thenarrative-enrichmentdeclines,and
GraphRAG’s overall edge over the baseline on Qiaojia and
Ridgecrest—fall within this noise margin. However, the
sharp improvements in Category D, the overall score gains
for Qiaojia (+0.73) and Ridgecrest (+0.45), and the base-
line’s outperformance of GraphRAG on Maduo remain sta-
tisticallyrobust(Table2)
Results: Capability Demonstration
Withnohand-designedschemaandnocatalog-specificcon-
figuration,GraphRAGautomaticallybuiltstructurallycom-
plete, queryable knowledge graphs for all three catalogs
from raw tabular records alone: Qiaojia-Dongchuan (5,247
entities, 3,920 relationships, 146 communities; Figure 2b),
Ridgecrest (1,985 entities, 2,449 relationships, 83 commu-
nities), and Maduo (15,402 entities, 11,784 relationships,
600 communities). Entity types are consistently dominated
by coordinate/date (GEO) tags (∼72–83%) across all three,
confirming that indexing capability is general and does
not depend on catalog region or scale. Per-catalog entity-
type and community-size distributions are given in the
SM (Figures S1–S2). The resulting graphs support natural-
language queries that a fixed-field database cannot: Local
Searchqueriesretrieveprecise,traceableindividualrecords
(e.g., correctly naming Ridgecrest’s𝑀7.1mainshock and
its 6 July date), and Global Search synthesizes holistic,
catalog-wide descriptions of sequence character. However,
thisout-of-the-boxfunctionalityhasreliabilityissues.When
reviewing 80 Ridgecrest benchmark answers (excluding
local retrieval), the system frequently misidentified smaller
earthquakes (𝑀3.5–𝑀5.5) as the sequence’s largest event.
Additionally, default prompts caused the model to hallu-
cinate fluent but baseless physical mechanisms, such as
falsely attributing Ridgecrest’s seismicity to reservoir or
fluid processes. Under the default, catalog-only condition,
the average scores were generally poor: 0.96 for Qiaojia,
1.22 for Ridgecrest, and 0.76 for Maduo out of a possi-ble 3. Qiaojia struggled most with mechanism questions
(Category D at 0.55/3), while Maduo suffered from uni-
formly low scores across all categories (0.45–0.95/3). These
baseline deficits, detailed in SM Table S5, motivated the
targeted prompt-engineering strategy in Section 5, which
successfullydoubledseveralofthesescores.Table1givessix
representative question/answer pairs – four correct, high-
quality responses spanning extremal retrieval, aftershock
association,narrative-groundedmechanismreasoning,and
a correct refusal, plus the two residual errors detailed in
Section 6 – to illustrate concretely what the model’s output
lookslike;extendedexamplesaregivenintheSM.
Results: Narrative-Text Enrichment
To assess the effect of narrative enrichment, each cata-
log was re-indexed with contextual literature—Baihetan
Reservoir background for Qiaojia, Shelly (2020) for
Ridgecrest, and Guan et al. (2024) for Maduo—and
re-evaluated. The effect is catalog-dependent and non-
monotonic (Category-D scores in SM Table S6). Qiaojia’s
Category-D score rises (0.55→1.35) but mainly reflects a
morecautioustonematchingthetext’s“contested”framing
rather than genuine use of its specific named entities.
For Ridgecrest, the score remained essentially unchanged
(1.45 to 1.40). When asked about industrial activity, the
model simply refused to answer instead of using the newly
providedtextbyShelly(2020)toconfidentlydenytheclaim.
While a blank refusal is the correct response when using
only the raw catalog—since the bare catalog lacks mech-
anism data—it represents a failure to utilize the enriched
text,whichcontainedtheevidenceneededtoexplicitlyrule
outthemechanism.WhileMaduo’sscoreincreased(0.75to
1.65),theenrichedtextintroducedamajornewmistake:the
model confused the natural deep-fluid processes described
by Guan et al.. with human-induced hydraulic fracturing.
This error did not occur when the model relied on the cat-
alog alone. Additionally, under default settings, the model
hallucinated the answer for the "largest-magnitude event".
It correctly named the real-world May 2021 mainshock,
but because the provided catalog data did not start until
June 1, the model pulled this fact from its outside memory
rather than the data—a "coincidentally correct" error that
persistedevenafterpromptengineering.
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record5

Table 1.
Representative example queries and GraphRAG answers, spanning correct, high-quality responses and the two
residual error classes of Section 6 (translated from the original Chinese-language queries; retrieval mode and
condition in parentheses; all answers from the fully re-verified post-fix benchmark).
Question Answer (abridged) Outcome
When, where, and at what mag-
nitude did the largest event in the
Qiaojia-Dongchuan catalog occur
(Global Search, catalog-only)?“The largest-magnitude event in the catalog occurred on 18
October 2022 at 22:12:35.080..., with magnitude 4.2.”Correct (true:𝑀4.18,
18 Oct 2022 22:12)
Which events are directly related
to the largest-magnitude event in
the 2019 Ridgecrest sequence
(Local Search, catalog-only)?“The largest-magnitude event ... is the M7.10 earthquake on
2019-07-06 at 03:19:52.860. This event is a major mainshock,
followed by a series of aftershocks directly related to it...”Correct
What is the approximate rup-
ture/triggering mechanism
of the 2021 Maduo𝑀 𝑤7.4
sequence (Global Search,
narrative-enriched)?“...generated by the KMPJ fault, clearly identified as the main
seismogenic fault, ... interacting with secondary faults (the
Maduo-Gande and Jiuzhi faults)... the aftershock sequence is
complex, with a bifurcating geometry...”Correct; grounded
in the enrichment
text rather than
pretrained priors
Is the 2019 Ridgecrest seismic-
ity associated with industrial activ-
ity such as hydraulic fracturing,
wastewater injection, or reser-
voir impoundment (Global Search,
catalog-only)?“I am sorry but I am unable to answer this question given the
provided data.”Correct refusal:
the purely tectonic
catalog contains
no such evidence;
default prompts
had answered
from background
knowledge
What is the trend in seismic-
ity rate over time in the 2019
Ridgecrest sequence (Global
Search, catalog-only)?“...on July 4, 36 events were recorded, with a maximum magni-
tude of 4.6. Activity then peaked on July 6, with multiple nearby
epicenters recorded, maximum magnitude 4.97...”Wrong: omits the
true𝑀7.1main-
shock on its own day
(Section 6)
What is the largest-
magnitude event in the Maduo
sequence (Global Search,
narrative-enriched)?“...the sequence’s main event is the𝑀 𝑤7.4 earthquake of 22
May 2021... recorded as EVENT 2021-05-22T00:00:00...”Wrong: catalog con-
tains no May-2021
data (Section 6)
Results: Iterative Prompt Engineering
Followingtheidentificationofmechanisticallydistinctfail-
ure modes in Sections 3 and 4, we deployed iteratively
verified prompt and configuration adjustments to address
four specific issues identically across all conditions. These
targeted errors consisted of magnitude fabrication via field
confusion, historical-earthquake conflation, cross-catalog
place-name contamination, and a scope-collapse bug in
community detection. Each intervention was tailored to a
diagnosed root cause; for instance, the scope-collapse issue
was traced to a default GraphRAG clustering parameter
thatinadvertentlydiscardedthemajorityofcatalogentities.
Comprehensive diagnostic evidence, exact prompt modifi-cations, and mechanistic details are provided in SM Text
S5.
Were-indexedeveryaffectedconditionandre-ranthefull
benchmark, individually re-verifying all answers (Figure 3;
Table 2). All three targeted fabrication modes were effec-
tively eliminated: no impossible magnitude value appeared
in any of the 600 re-verified post-fix answers (versus
repeated “𝑀10.4”/“𝑀12.8” before), no cross-catalog place-
name contamination was found, and historical-earthquake
conflation (previously∼1 in 5 relevant answers) disap-
peared.Despitethepromptfixes,twodistincttypesofhallu-
cinationsremained:presentingout-of-catalogeventsasreal
data,andinventingaggregatecounts(detailedinSection6).
While scores for physical mechanism questions (Category
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record6

D) improved dramatically—jumping from 0.55 to 2.90 for
Qiaojia and 1.45 to 2.85 for Ridgecrest—precise statistics
(CategoryB)remainedpersistentlyweak.CategoryBscored
between 0.50 and 1.04 out of 3, making it the lowest-
performing category in five out of six test conditions. The
mainissueisthatthemodelseverelyundercountstotalsand
rates.Thishappensbecausearetrieval-then-synthesizesys-
temtriestoanswermathquestionsusingafewtextsnippets
rather than scanning the entire database—a fundamental
architectural flaw rather than a prompt issue (detailed fur-
therinSMTextS5.4).
The vector-RAG baseline (Table 2) shows that
GraphRAG’s advantage concentrates precisely where
its community reports are designed to help: holistic sum-
marization (Category A: 1.85 vs. 1.30 on Qiaojia) and
temporal-stage comparison (Category E: 1.70 vs. 1.10 on
Ridgecrest), where top-𝑘chunk retrieval cannot synthe-
size catalog-wide structure and mostly returns hedged
partial descriptions or refusals. Category B is comparably
weak for both architectures (0.85–1.05 across the Qiaojia
and Ridgecrest catalog-only conditions; lower still for
GraphRAG on Maduo), confirming this limitation is
common to retrieval-then-synthesize systems rather than
specific to GraphRAG. The baseline’s perfect Category-D
scores (3.00) reflect uniform honest refusals scored as cor-
rect because the catalogs contain no mechanism evidence;
GraphRAG’s 2.85–2.90 comes from substantive grounded
answers,soCategoryDisuninformativehere–thearchitec-
tural comparison rests on Categories A and E. On Maduo,
however, the baseline’s uniformly honest refusals outscore
GraphRAG’s degraded index (average 1.40 vs. 0.80): the
graph layer’s value is conditional on a healthy index.
The ranking is sensitive to the refusal-scoring convention
(Table 2, note), so we report both. Notably, the baseline
reproduces the famous-mainshock intrusion of Section 6
– misdating Ridgecrest’s𝑀7.1from pretrained knowledge
while explicitly admitting the retrieved chunks lack it –
evidence that this hazard is intrinsic to LLM synthesis, not
toGraphRAG.
Residual Catalog-Specific Errors
Our per-answer verification of the fully fixed pipeline
(Section 5) also surfaced two narrower, catalog-specific
errors, concentrated on the two catalogs tied to a globally
famous mainshock – though not exclusively: a mechani-
0 1 2 3Mean veri fied score (0 –3)
+0.55
+0.50
+0.50A
Summarization
+0.29
−0.25
−0.25B
Statistics
+0.45
+0.15
−0.45C
Local retrieval
+2.35
+1.40
+0.45D
Mechanism
±0.00
+0.45
−0.05E
Stage
comparisonΔ
Qiaojia
RidgecrestMaduo
Before (default prompts)After (targeted fixes)Figure 3.Per-category verified benchmark scores before (open
circles; default prompts) and after (filled circles; targeted fixes) the
Section 5 prompt-engineering scheme, catalog-only condition. Arrows
show the direction of change for each catalog; the right column (∆)
gives the per-catalog score change. Physical-mechanism questions
(Category D) improve most; precise-statistics questions (Category B)
remain architecture-limited.
Alttext:Dumbbell(dot-and-arrow)plot.X-axis:meanverifiedscore,0to3.Y-axis:five
questioncategories(A–E),eachwiththreecoloredrowsforQiaojia(blue),Ridgecrest
(orange),andMaduo(green).Eachrowshowsanopencircle(defaultprompts)joined
byanarrowtoafilledcircle(afterfixes),pointingrightforgainsandleftforlosses;a
right-hand∆columnlistseachnumericchange.CategoryDshowsthelongestrightward
arrows,upto+2.35;CategoryBshowsshortarrows,includingtwonegativechanges.
cal scan of all 200 post-fix Qiaojia answers for dates or
coordinates outside the catalog’s range found six (3%, all
Category-Clocal-retrievalquestions,inbothconditions)list-
ingwhollyinventedeventrecords;thesamescanflagszero
Ridgecrest answers. We report these to inform responsible
deployment, not as a limitation of the pipeline’s core query
capabilitydemonstratedinSections3–5.
OnRidgecrest, the true𝑀7.1mainshock (6 Jul 2019,
the busiest day) is placed on 4 or 5 July instead in 50–
70% of relevant re-verified answers, often merged with the
𝑀6.4foreshock date; narrative enrichment did not reduce
this. The correct answers in Section 3 and Table 1 are
drawn from the complementary 30–50%: the same ques-
tiontypeyieldsthecorrectdateinasubstantialminorityof
answers,whichispreciselywhatmakesthiserrorhazardous
–spot-checkingafewcorrectanswerscannotexcludeit.On
Maduo,whosecatalogbegins1June2021withnoMaydata,
the real out-of-catalog 22-May-2021 mainshock is nonethe-
lesspresentedasthecatalog’sownlargesteventinupto60%
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record7

Table 2.
Verified category-average scores after the targeted fixes (Section 5), all catalogs/conditions, and the vanilla
vector-RAG baseline (catalog-only).
Condition A B C D E Avg
Qiaojia catalog-only 1.85 1.04 1.35 2.90 1.30 1.69
Qiaojia +narrative 1.85 0.90 1.30 1.60 1.50 1.43
Ridgecrest catalog-only 1.70 0.85 1.25 2.85 1.70 1.67
Ridgecrest +narrative 1.23 0.90 1.05 2.60 1.10 1.38
Maduo catalog-only 1.20 0.70 0.50 1.20 0.40 0.80
Maduo +narrative 0.65 0.50 0.90 2.15 1.15 1.07
Qiaojia vector-RAG baseline 1.30 1.00 1.15 3.00 1.00 1.49
Ridgecrest vector-RAG baseline 1.45 1.05 1.20 3.00 1.10 1.56
Maduo vector-RAG baseline 1.00 1.00 1.00 3.00 1.00 1.40
A:summarization;B:statistics;C:localretrieval;D:mechanism;E:stagecomparison.Baselinerowssharechunks,embed-
dings,LLM,andquestionswiththecorrespondingcatalog-onlycondition(Section2);theirCategory-Dscoresreflectuniform
honestrefusalsscoredascorrect.Underthestricterconventionscoringrefusalsonanswerablequestions0ratherthan1,the
baselineaveragesfallto0.98,1.20,and0.69.
of narrative-enriched answers (versus≤15% catalog-only),
alongsideanindependentlyfabricated“78,832aftershocks”
figure recurring in∼30% of enriched mechanism/stage-
comparison answers. We interpret the catalog-dependent
asymmetryasevidencethatanLLM’spretrainedknowledge
ofagloballyfamous,heavilyreportedeventcan,inaminor-
ityofanswers,overrideretrievedcatalog-groundedevidence
during synthesis – a narrower and more specific hazard
than the general “ungrounded generation” hallucination
described in the broader literature (Ji et al., 2023), and one
that our Section 5HISTORICAL_EVENTfix, designed for
a structurally similar problem (historical-earthquake con-
flation), did not fully generalize to suppress. These hazards
resist casual review: in our scoring audit (Section 2), two
fabricated-record answers were initially rated fully correct
byanunaidedhumanpassandwerecaughtonlywhentheir
dates and coordinates were checked against the catalog’s
actualrange.ExtendedexamplesaregivenintheSM.
Discussion and Conclusions
This study demonstrates that GraphRAG can be applied
directly,end-to-end,toearthquakecatalogsofmarkedlydif-
ferent scale, region, and origin – with no hand-designed
schema, no custom extraction rules, and no per-catalogreconfiguration – to produce structurally complete knowl-
edge graphs supporting flexible natural-language query,
with verified instances of precise record retrieval and
grounded mechanism reasoning (Table 1), though answer
reliabilityvariesbycatalogandquestioncategory(Table2).
Thisis,toourknowledge,thefirstsystematicdemonstration
of GraphRAG’s schema-free construction capability trans-
ferring across independent earthquake catalogs at essen-
tiallyzeroontology-engineeringcost,althoughdownstream
answerqualitydoesnottransferuniformly(post-fixcatalog-
onlyaveragesof1.69,1.67,and0.80/3).
Beyond capability, we show that reliability is not fixed
once GraphRAG is applied out of the box, but can be mea-
surablyimprovedwithmodest,seismology-orientedprompt
engineering: our four targeted fixes (Section 5) eliminated
three concrete error types from the re-verified answer set
entirely and raised mean scores from 0.76–1.22/3 (SM
Table S5) to 0.80–1.69/3 (Table 2) – catalog-only gains of
+0.73 for Qiaojia and +0.45 for Ridgecrest, but an essen-
tially flat +0.04 for Maduo, where gains in summarization
and mechanism reasoning were offset by declines in statis-
tics, retrieval, and stage comparison. These findings point
to a practical and affordable strategy for deploying LLM
query tools on earthquake catalogs: start with the default
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record8

GraphRAG pipeline, test it against a small ground-truth
benchmark, and refine the prompts instead of building a
new system from scratch. However, there is a catch: while
this approach successfully eliminated the targeted hallu-
cinations across all test cases, it only produced signifi-
cant overall score improvements for two of the three cata-
logs. Narrative-text enrichment, by contrast, reduced post-
fix overall scores for Qiaojia (1.69→1.43/3) and Ridgecrest
(1.67→1.38/3); it raised Maduo’s (0.80→1.07/3), but at the
cost of more frequent out-of-catalog mainshock intrusions
there (up to 60% of enriched answers versus≤15% catalog-
only),soitshouldbeadoptedonlywithper-catalogverifica-
tion(Section4).
Deploymentislimitedbytwomaincaveats,thefirstbeing
thatprecise-statisticsqueries(CategoryB)performedpoorly
across all conditions even after applying fixes, scoring just
0.50–1.04 out of 3. This category scored the lowest in five
out of six evaluation conditions, which points to a funda-
mentalstructuralflawinretrieval-then-synthesizearchitec-
tureswhenhandlingaggregateorrankingtasks.Therefore,
rather than relying on prompt engineering to fix this issue,
we recommend routing these specific mathematical ques-
tions directly to deterministic database or graph queries.
The second caveat involves the residual errors detailed in
Section6,whichprimarilyaffectedcatalogsassociatedwith
well-known mainshocks. These errors show that verifying
the output for each specific catalog remains a necessary
step before deployment, especially when analyzing well-
known events. A promising future fix would be explicitly
instructing the model to never use its pre-trained back-
groundknowledgeto"correct"theretrieveddata,thoughwe
did not have the chance to design and test that approach
in this study. Because all our results rely on a single model
(gpt-4o-mini), the rate of hallucinations and the effec-
tiveness of our fixes might vary if a different model is
used. However, since the vector-RAG baseline reproduced
the exact same hallucination (Section 6), we know this
specific flaw stems from the LLM itself, not GraphRAG’s
architecture. Ultimately, neither limitation invalidates our
mainconclusion:GraphRAGisahighlyusefultoolforauto-
matically building earthquake-catalog knowledge graphs,
performing strongly for Qiaojia and Ridgecrest, and more
modestly for Maduo. When the index is healthy (as with
Qiaojia and Ridgecrest), GraphRAG’s community-report
layer noticeably outperforms a standard vector-RAG base-
line on complex, holistic questions. Conversely, when theindex degrades (as with Maduo), the baseline matches or
evenbeatsGraphRAG.Overall,combiningthisarchitecture
withtargetedpromptengineeringyieldsgenuine,verifiable
improvementsinreliability.
Data and Resources
All data and codes are available athttps://doi.
org/10.5281/zenodo.21459373.TheSupplementary
Material accompanying this article provides full pipeline
and benchmark parameters, per-category and per-catalog
score tables, entity-type and community-size distributions,
exact prompt-engineering modifications, and extended
question/answerexamplessupportingSections3–6.
Declaration of Competing Interests
The authors acknowledge that there are no conflicts of
interestrecorded.
Acknowledgments
Y.Z. and S.M.M were supported by Harvard Milton Fund.
Theauthorsthank thedevelopersof theGraphRAGframe-
work and the seismological data providers whose catalogs
and publications made this study possible. We also thank
the members of Plantcore.AI, whose insights during our
discussionshelpedinspirethiswork.
References
Davis, W. and C. R. Hunt (2024). Knowledge graphs for seismic
dataandmetadata.Appl.Comput.Geosci.21,100151.
Edge,D.,H.Trinh,N.Cheng,J.Bradley,A.Chao,A.Mody,S.Truitt,
and J. Larson (2024). From local to global: A graph RAG
approachtoquery-focusedsummarization.
Guan, P. H., J. S. Lei, and D. P. Zhao (2024). Machine-learning
basedlocationofthe2021Mw7.4Maduoearthquakesequence:
Insight into intraplate seismogenesis.Tectonophysics888,
230458.
Gupta,H.K.(2002). Areviewofrecentstudiesoftriggeredearth-
quakes by artificial water reservoirs with special emphasis on
earthquakesinKoyna,India.Earth-Sci.Rev.58(3–4),279–310.
Gutenberg,B.andC.F.Richter(1944). Frequencyofearthquakes
inCalifornia.Bull.Seismol.Soc.Am.34(4),185–188.
Hogan, A., E. Blomqvist, M. Cochez, C. d’Amato, G. de Melo,
C. Gutierrez, S. Kirrane, J. E. L. Gayo, R. Navigli, S. Neumaier,
A. C. N. Ngomo, A. Polleres, S. M. Rashid, A. Rula,
L. Schmelzeisen, J. Sequeda, S. Staab, and A. Zimmermann
(2021). Knowledgegraphs.ACMComput.Surv.54(4),1–71.
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record9

Ji, Z., N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang,
A. Madotto, and P. Fung (2023). Survey of hallucination in
naturallanguagegeneration.ACMComput.Surv.55(12),248.
Lewis, P., E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and
D.Kiela(2020).Retrieval-augmentedgenerationforknowledge-
intensive NLP tasks. InAdvances in Neural Information
ProcessingSystems,Volume33,pp.9459–9474.
Li,W.andY.Zhou(2026).Towardknowledge-enhancedgeohazard
intelligence: A review of knowledge graphs and large language
models.GeoHazards7(2),40.
Mousavi,S.M.andG.C.Beroza(2022). Deep-learningseismology.
Science377(6607),eabm4470.
Mousavi, S. M., W. L. Ellsworth, W. Zhu, L. Y. Chuang, and
G. C. Beroza (2020). Earthquake transformer – an attentive
deep-learningmodelforsimultaneousearthquakedetectionand
phasepicking.Nat.Commun.11,3952.
Ogata,Y.(1988).Statisticalmodelsforearthquakeoccurrencesand
residualanalysisforpointprocesses.J.Am.Stat.Assoc.83(401),
9–27.
Orantes-Jiménez, D. (2025). Harnessing large language models
to build knowledge graphs from earthquake news.Int. J. Digit.
Earth18(2),2594950.
Pan, S., L. Luo, Y. Wang, C. Chen, J. Wang, and X. Wu (2024).
Unifying large language models and knowledge graphs: A
roadmap.IEEETrans.Knowl.DataEng.36(7),3580–3599.
Qiu, P. Y., L. K. Pang, Y. Luo, Y. H. Liu, H. Q. Xing, K. Liu, and
G. L. Zhuang (2024). Earthquake event knowledge graph con-
struction and reasoning.Geomat. Nat. Hazards Risk15(1),
2383768.
Ross, Z. E., M. A. Meier, E. Hauksson, and T. H. Heaton (2018).
Generalized seismic phase detection with deep learning.Bull.
Seismol.Soc.Am.108(5A),2894–2901.
Ross,Z.E.,D.T.Trugman,E.Hauksson,andP.M.Shearer(2019).
Searching for hidden earthquakes in Southern California.
Science364(6442),767–771.
Shelly,D.R.(2020).Ahigh-resolutionseismiccatalogfortheinitial
2019Ridgecrestearthquakesequence:Foreshocks,aftershocks,
andfaultingcomplexity.Seismol.Res.Lett.91(4),1971–1978.
Traag,V.A.,L.Waltman,andN.J.vanEck(2019).FromLouvainto
Leiden:Guaranteeingwell-connectedcommunities.Sci.Rep.9,
5233.
Utsu, T., Y. Ogata, and R. S. Matsu’ura (1995). The centenary of
theOmoriformulaforadecaylawofaftershockactivity.J.Phys.
Earth43(1),1–33.
Yao, L., F. Ren, and K. Du (2025). From knowledge graph con-
struction to retrieval-augmented generation: A framework for
comprehensive earthquake emergency support.Geo-Spat. Inf.
Sci.29(1).
Yu, R., S. Luo, R. Ghosh, L. Li, Y. Xie, and X. Jia (2025). RAG for
geoscience:Whatweexpect,gapsandopportunities.Zhu, W. Q. and G. C. Beroza (2019). PhaseNet: A deep-neural-
network-based seismic arrival-time picking method.Geophys.
J.Int.216(1),261–273.
Manuscript Received 00 Month 0000
https://www.seismosoc.org/publications/the-seismic-record/ • DOI: 00.0000/000000000The Seismic Record10