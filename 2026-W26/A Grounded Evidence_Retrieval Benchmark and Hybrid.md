# A Grounded Evidence-Retrieval Benchmark and Hybrid RAG Framework for Silicon Pixel Detector R&D

**Authors**: Tianqi Gao, Ruobing Jiang, Dawei Fu, Qiang Li, Matthew Kenzie

**Published**: 2026-06-23 15:49:58

**PDF URL**: [https://arxiv.org/pdf/2606.24725v1](https://arxiv.org/pdf/2606.24725v1)

## Abstract
The rapid growth of silicon pixel detector literature has made systematic evidence retrieval a practical bottleneck for detector R&D. Large language models alone are insufficient for this task, as specialised detector knowledge, long-tail technical details, and recent experimental results must be grounded in primary literature. We present the first evidence-grounded retrieval benchmark and a reproducible retrieval framework for silicon pixel detector studies, combining sparse lexical retrieval, dense semantic retrieval, hybrid retrieval, and graph-based literature exploration. The benchmark includes manually curated chunk-level evidence annotations, source-level diagnostics, semantic relevance checks, and negative-query abstention tests across two complementary detector-domain query sets. Systematic evaluation shows that hybrid sparse-dense retrieval provides the most reliable evidence recovery, while graph-based approaches are more effective for literature exploration than strict evidence ranking. These results highlight the importance of evidence-grounded retrieval for accessing long-tail detector knowledge and provide a practical foundation for retrieval-augmented tools supporting silicon detector research and high-energy physics instrumentation.

## Full Text


<!-- PDF content starts -->

A Grounded Evidence-Retrieval Benchmark and Hybrid RAG
Framework for Silicon Pixel Detector R&D
Tianqi Gaoa, Ruobing Jiang∗b, Dawei Fub, Qiang Lib, and Matthew Kenziea
aCavendish Laboratory, University of Cambridge, Cambridge, United Kingdom
bState Key Laboratory of Nuclear Physics and Technology, Peking University, Beijing,
China
June 24, 2026
Abstract
The rapid growth of silicon pixel detector literature has made systematic evidence retrieval
a practical bottleneck for detector R&D. Large language models alone are insufficient for this
task, as specialised detector knowledge, long-tail technical details, and recent experimental
results must be grounded in primary literature. We present the first evidence-grounded
retrieval benchmark and a reproducible retrieval framework for silicon pixel detector studies,
combining sparse lexical retrieval, dense semantic retrieval, hybrid retrieval, and graph-
based literature exploration. The benchmark includes manually curated chunk-level evidence
annotations, source-level diagnostics, semantic relevance checks, and negative-query abstention
tests across two complementary detector-domain query sets. Systematic evaluation shows that
hybrid sparse–dense retrieval provides the most reliable evidence recovery, while graph-based
approaches are more effective for literature exploration than strict evidence ranking. These
results highlight the importance of evidence-grounded retrieval for accessing long-tail detector
knowledge and provide a practical foundation for retrieval-augmented tools supporting silicon
detector research and high-energy physics instrumentation.
Keywords:silicon pixel detector; retrieval-augmented generation; hybrid retrieval; evidence-
grounded question answering; benchmark
∗Corresponding author: ruobing@stu.pku.edu.cn
The benchmark annotations, retrieval outputs, and evaluation code will be released upon publication.
1arXiv:2606.24725v1  [physics.ins-det]  23 Jun 2026

Contents
1 Introduction 3
2 Background and Motivation 4
2.1 Scientific retrieval and evidence-grounded literature access . . . . . . . . . . . . . 4
2.2 Retrieval challenges in silicon detector literature . . . . . . . . . . . . . . . . . . 5
2.3 Application scenarios for detector R&D . . . . . . . . . . . . . . . . . . . . . . . 6
3 Methodology 7
3.1 Design principles and corpus construction . . . . . . . . . . . . . . . . . . . . . . 7
3.2 Precision-oriented evidence retrieval . . . . . . . . . . . . . . . . . . . . . . . . . 7
3.3 Graph-guided literature exploration . . . . . . . . . . . . . . . . . . . . . . . . . 9
3.4 Evidence-grounded response generation . . . . . . . . . . . . . . . . . . . . . . . 9
4 Experiments 10
4.1 Experimental setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
4.2 Retrieval performance evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
4.3 Abstention behaviour on negative queries . . . . . . . . . . . . . . . . . . . . . . 17
5 Discussion and Conclusion 18
A Grounding and Abstention Prompt 22
B Retrieval Metrics 23
C Reproducibility Commands 23
2

1 Introduction
Silicon pixel detector research spans sensor design, radiation qualification, beam-test characteri-
sation, simulation, and electronics integration, with individual studies drawing on information
across detector technologies, experimental conditions, and software frameworks. This breadth is
scientifically valuable, but it also creates a practical problem for detector R&D: the evidence
needed to justify a detector-design choice or performance claim is often distributed across many
publications and subfields. Document-level search is therefore frequently insufficient. For example,
a detector study may require a specific passage on how substrate bias affects charge-collection
efficiency after irradiation, the timing performance achieved for a particular LGAD geometry [ 1],
or the threshold and reconstruction conditions used in a beam test. Similarly, questions involving
High-Voltage Monolithic Active Pixel Sensor (HV-MAPS), Depleted Monolithic Active Pixel
Sensor (DMAPS), or hybrid pixel assemblies [ 2,3] may require distinguishing sensor architectures,
irradiation fluences, operating voltages, and readout configurations that are superficially similar
but physically distinct.
Large language models (LLMs) are increasingly adopted for scientific question answering; however,
they remain limited in generating faithful and evidence-grounded responses for specialised
detector knowledge, long-tail technical details, and rapidly evolving detector developments [ 4–7].
Many detector-R&D questions depend on information that appears only in a small number of
publications and under specific operating conditions, making such knowledge difficult to recover
reliably from the parametric memory of a language model alone. Retrieval-augmented generation
(RAG) mitigates this limitation by grounding answer generation in evidence retrieved from
external corpora. In practice, the retrieval step is itself non-trivial for detector literature. The
challenge is not simply retrieving related publications, but identifying the exact supporting
evidence under the relevant detector configuration, irradiation condition, bias voltage, or readout
setting. Dense semantic retrieval enables matching of paraphrased queries [ 8–10], but may miss
exact acronyms, detector names, or process-specific terminology. Sparse lexical retrieval such
as BM25 remains a strong baseline for keyword-driven queries [ 11,12]. Hybrid approaches, for
example using reciprocal-rank fusion (RRF) [ 13,14], combine these signals and often provide
robust performance across heterogeneous query types. Recent graph-guided retrieval approaches
further introduce structured relationships between entities, concepts and documents to support
broader literature exploration [ 15–18]. While these methods can improve semantic exploration
and literature navigation, they are not automatically beneficial for detector-domain retrieval:
graph expansion and semantic broadening can introduce passages that are topically related but
less directly relevant to the required detector evidence. This tension between semantic coverage
and strict evidence precision is particularly important for detector R&D, where unsupported or
weakly grounded statements can mislead detector-design choices or physics-analysis decisions.
In this work, we develop and evaluate a grounded evidence-retrieval framework for silicon pixel
detector literature, framed as a software and computing tool for detector R&D rather than as a
general NLP benchmark. Rather than proposing a new retrieval algorithm, the aim of this study
is to understand how established retrieval strategies behave in a detector-instrumentation setting.
By using standard and well-understood retrieval components, we isolate the domain effect and
directly examine whether graph-enhanced retrieval improves strict evidence ranking for detector
literature, or whether its primary value lies instead in literature exploration and detector-concept
navigation. We therefore combine BM25 sparse retrieval, dense FAISS retrieval, reciprocal-
rank-fusion hybrid retrieval, detector-entity graph expansion, and graph-path retrieval within a
unified evaluation pipeline, summarised in Figure 1. Retrieval performance is evaluated using
manually curated chunk-level gold evidence, source-level diagnostics, semantic soft-gold checks,
and negative-query abstention tests across two complementary detector-domain benchmarks.
The results show that hybrid sparse–dense retrieval provides the strongest strict evidence-ranking
performance, while graph-based approaches remain valuable for exploratory literature navigation
3

Figure 1:Overview of the complete system pipeline. The six stages cover corpus acquisition, document
processing, index and knowledge-graph construction, hybrid and graph-guided retrieval, grounded answer
generation, and rigorous evaluation and benchmarking.
and detector-concept interpretation.
The main contributions of this work are:
•the first detector-domain benchmark for grounded evidence retrieval in silicon pixel detector
literature, including strict chunk-level annotations, source-level diagnostics, semantic soft-gold
evaluation, and negative-query abstention tests;
•a reproducible retrieval framework combining BM25 sparse retrieval, dense FAISS retrieval,
hybrid reciprocal-rank-fusion retrieval, detector-entity graph expansion, and graph-path
retrieval under a unified evaluation pipeline;
•an empirical study of retrieval behaviour in detector instrumentation literature, showing
that hybrid sparse–dense retrieval provides the strongest strict evidence-ranking performance,
while graph-based retrieval is more effective for literature exploration and detector-concept
navigation than for exact evidence retrieval, providing a practical domain-specific caution
against adopting GraphRAG-style expansion as a default evidence ranker;
•detector-facing benchmark queries and retrieval analyses covering irradiation studies, timing
performance, charge collection, detector simulation, readout configuration, and detector-
technology comparison, demonstrating how evidence-grounded retrieval can support practical
silicon detector R&D workflows.
2 Background and Motivation
2.1 Scientific retrieval and evidence-grounded literature access
Detector-physics literature differs from generic open-domain corpora in both language and
evidential structure. Terminology is dense with acronyms, device names, detector-specific
concepts, and configuration-dependent measurements that often carry highly specialised meanings.
4

Relevant evidence is frequently passage-local rather than document-level, and different claims
within the same paper may depend on distinct irradiation conditions, bias voltages, timing
configurations, beam-test assumptions, or reconstruction settings. These characteristics make
detector literature a long-tail scientific knowledge domain in which critical design and performance
information is distributed across specialised publications rather than concentrated in a small
number of widely cited sources. Consequently, detector-domain retrieval differs fundamentally
from generic document search and motivates the use of evidence-grounded retrieval pipelines.
RAG grounds downstream analysis and answer generation in retrieved evidence passages rather
than relying entirely on parametric language-model memory [ 5,19]. Recent work has demon-
strated the value of evidence-grounded retrieval in specialised high-energy-physics domains,
including muon-collider phenomenology and future-collider studies [ 20]. Hybrid sparse–dense
retrieval combines lexical precision with semantic matching and is widely used in scientific and
technical retrieval settings [ 9,21]. Graph-based retrieval and entity-linking approaches, including
GraphRAG-style methods, can additionally expose relationships between detector technologies,
irradiation studies, readout architectures, and performance measurements, providing useful
contextual expansion and interpretability [ 15]. However, broader semantic expansion is not
automatically beneficial for scientific retrieval. Retrieving loosely related passages can weaken
evidence precision if the highest-ranked citations become detached from the detector configu-
ration or measurement context relevant to the original query. This trade-off between semantic
coverage and evidence precision is particularly important for detector R&D, where unsupported
or weakly grounded statements may influence detector-design choices, performance projections,
or technology-selection decisions. This consideration becomes increasingly relevant for future
collider programmes, where detector concepts must be evaluated using evidence accumulated
across multiple sensor technologies, irradiation campaigns, beam tests, and simulation studies.
Several benchmark datasets exist for scientific retrieval and evidence-grounded information access,
including SciFact [ 22], SciDocs [ 23], and biomedical retrieval benchmarks such as BioASQ [ 24].
However, these benchmarks target generic scientific or biomedical literature and do not capture
the domain-specific characteristics of silicon pixel detector publications: configuration-dependent
measurements, irradiation conditions, exact detector terminology, and passage-local evidence
under stated operating conditions. A detector-domain benchmark therefore requires dedicated
corpus construction, query design, and chunk-level gold annotation that cannot be substituted
by existing general-purpose scientific retrieval resources.
2.2 Retrieval challenges in silicon detector literature
A central motivation for building an evidence-grounded retrieval system for detector literature is
that the information required to answer a detector-R&D question is rarely contained within a single
abstract or publication. Instead, relevant evidence is often distributed across multiple detector
studies, beam-test campaigns, irradiation measurements, simulation papers, and technology-
specific reports. Consider a query such as:“What is the time resolution of LGAD sensors at a
neutron fluence of1015neq/cm2?”Answering this query correctly requires identifying publications
reporting LGAD test results after neutron irradiation, matching the fluence level to the relevant
experimental conditions, distinguishing irradiated from unirradiated timing performance, and
verifying whether parameters such as bias voltage, temperature, sensor thickness, and gain-layer
design are stated. Similar challenges arise for questions involving HV-MAPS, DMAPS, hybrid
pixel sensors, detector simulations, or irradiation studies, where seemingly similar measurements
may correspond to substantially different detector configurations.
Detector-physics reasoning of this kind also requires understanding that many entities are
physically related. A query about depletion voltage may imply relevance to electric field formation,
charge collection, sensor capacitance, and bias optimisation. Likewise, a query concerning timing
performance may require connecting sensor geometry, gain mechanisms, front-end electronics,
5

and irradiation behaviour. This relational structure is one reason why graph-guided retrieval
and entity linking can provide useful contextual expansion. However, graph expansion can also
introduce adjacent but non-gold passages that pass semantic soft-gold checks while failing strict
evidence matching. In practice, graph-expanded context is valuable for navigating detector-
concept relationships and recovering surrounding technical context, but it should not displace
precision retrieval when an exact supporting passage exists in the corpus. This distinction
becomes increasingly important as detector studies grow in scale and complexity, requiring
evidence to be integrated across multiple sensor technologies and experimental campaigns.
2.3 Application scenarios for detector R&D
To characterise the structure of information needs in silicon pixel detector R&D, we define
representative application scenarios for the proposed framework in Table 1. These scenarios span
detector-technology comparison, irradiation and radiation-hardness studies, timing-performance
characterisation, charge-collection and signal-formation analysis, readout and front-end con-
figuration, detector-simulation and test-beam workflows, and answerability checking. Such
information-access requirements are expected to become increasingly important for future collider
detector programmes, where detector-design decisions must be informed by evidence accumu-
lated across multiple sensor technologies, irradiation campaigns, beam tests, and simulation
studies. Each scenario maps a recurring detector-R&D question to its underlying retrieval
objective, motivating a system capable of chunk-level evidence retrieval, source traceability, and
evidence-grounded literature access under stated operating conditions.
Table 1:Representative retrieval application scenarios and query intents for silicon pixel detector R&D.
Query category User intent Retrieval objectives
Technology compari-
sonHow do HV-MAPS, DMAPS, hybrid pix-
els, and LGAD-based detectors compare
for a given application?Sensor-architecture and
performance trade-off
studies.
Irradiation & radia-
tion hardnessHow does charge collection or leakage cur-
rent evolve with fluence and bias?Radiation-qualification
and annealing studies
under stated fluence.
Timing performance What time resolution is achieved for a
given LGAD or AC-LGAD geometry?Beam-test and laboratory
timing characterisation.
Charge collection &
signal formationHow do depletion, electric field, and
weighting field shape the collected signal?Sensor-physics and
TCAD/charge-transport
studies.
Readout & front-end What threshold, ToA/ToT, and calibra-
tion conditions are used?ASIC, Timepix, and front-
end configuration reports.
Simulation & test
beamWhich simulation framework and recon-
struction setup were used?Allpix Squared, Geant4,
and test-beam analysis lit-
erature.
Answerability check Is a requested detector claim supported
by the available corpus?Abstain rather than
produce unsupported
answers.
These characteristics create a practical challenge for literature navigation. Detector-R&D
questions often require connecting sensor architecture, operating conditions, irradiation behaviour,
and readout configuration across multiple sources, and correct conclusions frequently depend
on retrieving specific evidence fragments rather than entire documents. A query on the timing
resolution of an irradiated LGAD, for instance, requires different evidence from one on charge-
collection efficiency in a DMAPS sensor, even though both arise within the same corpus. This
motivates a retrieval framework capable of chunk-level evidence retrieval, source traceability,
6

and evidence-grounded literature access for silicon pixel detector studies. Such capabilities
may also support future detector-development programmes by enabling more efficient access to
the distributed technical knowledge accumulated across detector technologies and experimental
campaigns.
3 Methodology
3.1 Design principles and corpus construction
Figure 1 summarises the complete framework. The system is designed to support detector-oriented
workflows involving irradiation studies, timing measurements, charge-collection analysis, detector
simulation, sensor-technology comparison, and detector-performance interpretation. Unlike
generic scientific retrieval tasks, detector-domain retrieval presents three distinctive challenges.
First, supporting evidence is frequently passage-level rather than document-level, with critical
detector-performance information often appearing only within a small portion of a publication.
Second, detector claims depend strongly on operating conditions such as irradiation fluence,
bias voltage, temperature, particle type, and readout configuration. Third, many detector-
R&D questions involve long-tail technical knowledge distributed across specialised publications
and detector-specific studies. These characteristics motivate a retrieval-first architecture that
prioritises evidence precision, source traceability, and detector-configuration awareness.
The retrieval framework therefore combines a high-precision hybrid retrieval backbone with a
complementary graph-guided exploration layer. Hybrid retrieval is intended to maximise strict
evidence recovery for detector-R&D questions, while graph-guided expansion provides controlled
contextual exploration of related detector concepts and technologies without displacing the
strongest first-stage retrieval results. This separation allows the framework to support both
evidence-focused retrieval and broader literature navigation.
The corpus consists of silicon detector instrumentation literature and associated software resources,
including journal articles, conference proceedings, technical design reports, detector-simulation
documentation, and detector-development studies. Documents were selected to maximise detector-
relevant evidence coverage, with particular emphasis on publications reporting quantitative
measurements under stated operating conditions, including irradiation fluence, bias voltage,
temperature, timing performance, charge collection, detector geometry, and readout configuration.
Source documents are converted into cleaned text using standard PDF extraction followed
by lightweight preprocessing [ 25]. Headers, footers, watermarks, and duplicated captions are
removed where possible, while section headings, mathematical expressions, physical units, detector
terminology, and numerical values are preserved because they frequently carry the experimental
context required to interpret detector-performance measurements. Documents are segmented
into overlapping passage-level chunks of approximately 300–500 tokens, with section boundaries
preserved where possible so that measurement conditions remain associated with the corresponding
quantitative results.
Each chunk stores hierarchical identifiers and source metadata linking it to the original document
and publication context. The final indexed corpus comprises 378 source published documents
segmented into 8,442 passage-level chunks. These identifiers support both strict chunk-level
evaluation and source-level retrieval analysis, allowing exact evidence retrieval to be distinguished
from retrieval of the correct publication without the supporting passage.
3.2 Precision-oriented evidence retrieval
Because detector-performance claims often depend on exact operating conditions, first-stage
retrieval must maximise evidence precision while remaining robust to paraphrased detector
terminology. A retrieval system that retrieves the correct publication but fails to recover the
7

supporting passage is insufficient for evidence-grounded detector studies. We therefore employ a
hybrid retrieval strategy that combines sparse lexical matching with dense semantic retrieval.
Sparse retrieval preserves exact detector terminology and technical acronyms, while dense
retrieval improves robustness to paraphrased detector-physics descriptions. The two signals are
subsequently fused to provide a high-precision evidence-retrieval backbone for detector-domain
literature search.
Sparse retriever.The sparse component uses BM25 [ 11], which scores a tokenized query q
and document chunk daccording to term frequency, inverse document frequency, and length
normalization:
SBM25(q, d) =X
t∈qIDF(t)f(t, d)(k 1+ 1)
f(t, d) +k 1
1−b+b|d|/ |d|,(1)
where f(t, d) is the frequency of term tin chunk d,|d|is the chunk length, |d|the mean chunk
length, and k1andbare free parameters. The inverse document frequency IDF(t) =logN
d f(t)
assigns higher weight to rare terms, where Nis the total number of chunks and d f(t) the number
containing t. We use b= 0.75 as a standard length-normalization compromise. BM25 prioritises
exact term overlap, making it particularly effective in detector literature where key concepts are
expressed through stable acronyms and technical keywords such as LGAD, DMAPS, HV-MAPS,
ToA/ToT, and n eq/cm2. Its main limitation is that it cannot capture semantic equivalence when
relevant concepts are expressed using different terminology, motivating the complementary use
of dense retrieval.
Dense retriever.The dense component embeds queries and chunks into a shared vector
space using sentence-transformers/all-MiniLM-L6-v2 , indexed with FAISS under cosine
similarity [ 26]. Let f(·) denote the encoder mapping a text input to an embedding vector,
eq=f(q) ande d=f(d). Dense retrieval ranks chunks by cosine similarity:
Sdense(q, d) =eq·ed
∥eq∥∥e d∥, d∗= arg max
d∈DSdense(q, d).(2)
Dense retrieval captures conceptual similarity even when surface forms differ, for example
matching “charge-collection efficiency” to “collected-charge fraction”. However, embedding-based
similarity can over-generalise on fine-grained technical queries and may underperform on rare
acronyms or newly introduced detector terminology, which motivates combining it with BM25 in
a hybrid scheme.
Hybrid retriever.The two ranked lists are merged using weighted reciprocal-rank fusion
(RRF):
SRRF(c) =wd
K+r d(c)+wb
K+r b(c),(3)
where rd(c) and rb(c) denote the dense and BM25 ranks of chunk c, and wdandwbare the
corresponding fusion weights. We set K= 60 following the original RRF formulation [ 13], which
acts as a smoothing term that stabilises fusion across retrievers. The default values wd= 0.9 and
wb= 0.1 are selected based on a grid search over weight combinations, as described in Section 4.
Hybrid retrieval preserves the complementary strengths of lexical and semantic retrieval: BM25
is effective for acronym-specific detector queries, while dense retrieval improves robustness to
paraphrased detector-physics descriptions and semantically related terminology.
8

3.3 Graph-guided literature exploration
Graph-guided expansion is evaluated as a complementary literature-exploration strategy rather
than a replacement for high-precision evidence retrieval. Detector entities and concepts are
linked through a lightweight graph representation constructed from co-occurrence and metadata
relationships across the indexed corpus. The graph covers detector technologies, sensor-physics
concepts, readout electronics, performance metrics, irradiation studies, and simulation frameworks.
Starting from the initial hybrid-retrieval results, graph traversal is used to identify neighbouring
detector concepts and related passages that may provide additional contextual information.
In addition to static graph expansion, we evaluate an agentic graph configuration that introduces
a lightweight query-decomposition stage before retrieval. Complex detector-domain questions
are first decomposed into a small set of sub-queries targeting detector technologies, performance
metrics, operating conditions, or sensor-physics concepts. Retrieval is then performed over the
graph-expanded search space and the resulting evidence is merged before final ranking. The
purpose of this configuration is not to optimise strict evidence retrieval, but to investigate whether
structured decomposition can improve coverage for multi-concept detector questions involving
several interacting entities or operating conditions.
The primary purpose of graph-guided exploration is to support literature navigation and detector-
concept discovery. Detector-R&D questions frequently involve relationships between multiple
technologies, operating conditions, and performance metrics that are not always captured by
direct lexical matching. Graph-based exploration can therefore help expose connections between
detector technologies, irradiation behaviour, charge-collection mechanisms, timing-performance
studies, and simulation frameworks, providing broader contextual understanding of the retrieved
evidence.
However, semantic proximity is not necessarily equivalent to evidential relevance. Closely related
detector technologies may differ substantially in operating conditions, irradiation history, sensor
geometry, readout architecture, or measurement methodology. Consequently, unrestricted graph
expansion can introduce passages that are topically related but do not constitute the exact
supporting evidence required to answer a detector-R&D question. Figure 2 shows a representative
detector-entity subgraph extracted from the literature corpus. Nodes correspond to detector
technologies, sensor-physics concepts, readout electronics, performance metrics, and experimental
workflows, while edges represent co-occurrence and metadata relationships used during graph-
guided exploration. For this reason, graph-guided exploration is not treated as a replacement for
precision-oriented evidence retrieval. Instead, it functions as a controlled contextual-expansion
layer that complements the hybrid retrieval backbone while preserving the prioritisation of
high-confidence evidence passages.
3.4 Evidence-grounded response generation
Although the primary focus of this work is evidence retrieval, retrieved passages can be supplied
to a language model to support interactive literature exploration and evidence-grounded response
generation. The language model is not treated as the primary object of evaluation in this study;
rather, it serves as a downstream consumer of the retrieved evidence. This design reflects the
central objective of the framework, namely reliable evidence access and source traceability for
detector-domain literature.
Retrieved passages are provided together with source identifiers and passage metadata, allowing
generated responses to remain linked to the supporting detector literature. The language model
is instructed to answer only from the retrieved evidence, preserve citation traceability, and
abstain when the retrieved material is insufficient. Such behaviour is particularly important for
detector-domain questions involving irradiation studies, timing measurements, charge collection,
or detector-performance comparisons, where unsupported synthesis across incompatible detector
9

Figure 2:Representative detector-entity subgraph used for graph-guided literature exploration. Nodes
are grouped into detector technologies, sensor-physics concepts, readout electronics, performance metrics,
and experimental workflows. Edges represent literature-derived co-occurrence and metadata relationships
used for contextual expansion and detector-concept discovery.
configurations can misrepresent published results.
The prompt template includes the retrieved passages, source identifiers, and an explicit abstention
instruction. When no retrieved passage directly supports a requested detector claim under
the stated operating conditions, the model is instructed to return a designated abstention
response rather than generate unsupported content. Retrieved passages are packed in descending
retrieval-rank order up to a fixed context limit, ensuring that the highest-confidence evidence is
presented first. Citation traceability is preserved by requiring generated responses to reference
the supporting passage identifiers. The complete prompt template is provided in Appendix A.
4 Experiments
4.1 Experimental setup
The detector-literature corpus contains 378 source documents covering silicon detector instru-
mentation, detector simulation, and detector-performance studies. After document processing
and chunking, the corpus comprises 8,442 indexed passage-level chunks. The corpus spans a
broad range of silicon detector technologies, including HV-CMOS and DMAPS devices [ 2,3],
LGAD and AC-LGAD timing detectors [ 1], ALPIDE and MAPS sensors [ 27], Timepix3-based
detectors [28], and detector-simulation frameworks such as Allpix Squared and Geant4 [29, 30].
Evaluation is performed using two complementary benchmark sets designed to assess strict
chunk-level evidence retrieval, source-level retrieval, semantic evidence coverage, and abstention
behaviour on unsupported detector-domain queries. The benchmark composition is summarised
in Table 2. The core benchmark contains 60 detector-domain queries, including 48 answerable
queries with strict gold evidence and 12 negative queries. The curated extension benchmark
contains 56 queries, including 41 answerable queries with strict gold evidence and 15 negative
10

Table 2:Benchmark composition.
Benchmark Total Answerable with strict gold Negative
Core 60 48 12
Curated extension 56 41 15
Combined 116 89 27
Table 3:System artefacts used throughout the retrieval evaluation.
Artefact Value
Source documents 378
Text chunks in index 8442
Entity graph nodes 2914
Entity graph edges 18616
Retrieval configurations 6
Core strict-gold queries 48
Extension strict-gold queries 41
Negative/abstention queries total 27
queries. Together, the combined benchmark contains 116 detector-domain queries spanning
detector-technology comparison, irradiation studies, timing performance, charge collection,
detector simulation, readout architectures, and detector-performance analysis.
Queries are additionally grouped into four reasoning levels covering direct factual retrieval,
single-paper parameter retrieval, multi-evidence synthesis, and detector-physics reasoning. This
enables evaluation of retrieval performance as a function of reasoning complexity, rather than
solely by detector topic. The overall corpus, graph, and benchmark statistics used throughout
the retrieval study are summarised in Table 3.
Strict chunk-level gold labels are manually assigned to passages that directly support the
corresponding detector-domain claim under matching operating conditions. Source-level labels
test whether the correct publication is retrieved even when the exact supporting passage is
missed. Semantic soft-gold annotations provide a diagnostic for topically related but non-exact
evidence, while negative queries evaluate whether the system correctly abstains when the corpus
does not support a requested detector claim.
To assess the reliability of these strict labels, we performed an additional second-annotation
check on a stratified sample of 15 answerable core-benchmark queries, corresponding to 300
query–chunk pairs drawn from the pooled candidate evidence set. The second pass used the same
strict criterion: a chunk was labelled as gold only when it directly supported the detector-domain
claim under the relevant operating conditions, and not when it was merely topically related.
After calibration of this relevance criterion, agreement with the primary labels reached Cohen’s
κ= 0.619, with raw agreement of 0.897, positive-specific agreement of 0.680, and negative-specific
agreement of 0.938. Because gold passages are sparse within each candidate pool, the positive-
and negative-specific agreement values are reported alongsideκ.
Retrieval quality is evaluated using strict Hit@5 and Mean Reciprocal Rank (MRR) as the
primary metrics. Hit@5 measures whether the annotated gold evidence chunk appears within the
top five retrieved results, while MRR quantifies the ranking position of the first relevant passage.
To distinguish exact evidence retrieval from broader literature discovery, we additionally report
Paper@5 and Soft@5. Paper@5 measures whether the correct source publication is retrieved,
while Soft@5 measures retrieval of semantically relevant evidence that may not exactly match
the annotated gold passage. For negative queries, abstention accuracy is reported as the fraction
of unsupported queries on which the system correctly abstains.
11

Table 4:Main retrieval results on the 60-query core benchmark and 56-query curated extension benchmark.
Strict metrics use chunk-level gold evidence. Paper and soft metrics are reported as complementary
coverage diagnostics. The best score in each column is shown in bold.
Core 60 Extension 56
Method Hit@5 MRR Paper@5 Soft@5 Hit@5 MRR Paper@5 Soft@5
BM25 0.8960.789 0.983 0.9330.9270.679 0.946 0.857
Dense 0.625 0.527 0.917 0.833 0.756 0.501 0.839 0.804
Hybrid0.9170.773 0.967 0.9170.951 0.679 0.946 0.857
Graph 0.354 0.220 0.767 0.733 0.293 0.167 0.554 0.518
Graph-path 0.354 0.220 0.767 0.733 0.293 0.167 0.554 0.518
Agentic graph 0.312 0.220 0.767 0.733 0.268 0.142 0.589 0.536
Table 5:Bootstrap 95% confidence intervals for strict retrieval performance on the 48 answerable queries
in the core benchmark, estimated using 10,000 bootstrap resamples.
Method Hit@5 95% CI MRR 95% CI
BM25 0.896 [0.792, 0.979] 0.789 [0.686, 0.880]
Dense 0.625 [0.479, 0.750] 0.527 [0.399, 0.651]
Hybrid 0.917 [0.833, 0.979] 0.773 [0.672, 0.867]
Graph 0.354 [0.229, 0.500] 0.220 [0.130, 0.321]
Graph-path 0.354 [0.229, 0.500] 0.220 [0.129, 0.318]
Agentic graph 0.312 [0.188, 0.438] 0.220 [0.126, 0.322]
4.2 Retrieval performance evaluation
We first optimise the hybrid retrieval configuration and then compare sparse, dense, hybrid and
graph-guided retrieval strategies across the detector-domain benchmarks. The objective is to
determine whether graph-enhanced retrieval improves strict evidence ranking, or whether its
primary value lies in literature exploration and detector-concept navigation.
The default fusion weights wd= 0.9 and wb= 0.1 in Eq. (3)are selected through a grid search
over dense–sparse weight combinations on the answerable benchmark queries. Increasing the
dense contribution generally improves ranking quality, while a small lexical component remains
beneficial for preserving sensitivity to specialised detector terminology and exact detector names.
The configuration wd= 0.9 and wb= 0.1 achieves the best overall performance and is therefore
adopted throughout the remainder of the study.
We first compare all retrieval configurations under strict chunk-level evaluation and then examine
their broader source-level and semantic retrieval behaviour. Table 4 presents the main retrieval
results for all six configurations on the core and extension benchmarks. Figures 3 and 4 show
the strict Hit@5, MRR and Hit@k behaviour.
To assess statistical robustness, Table 5 reports 95% bootstrap confidence intervals for the
strict retrieval metrics using 10,000 bootstrap resamples. The resulting intervals confirm that
hybrid retrieval consistently achieves the strongest strict evidence-ranking performance, while
graph-based retrieval methods remain substantially below the lexical and hybrid baselines.
The statistical comparison in Table 6 shows that hybrid retrieval significantly outperforms dense
and graph-based retrieval methods, while the difference between hybrid retrieval and BM25 is
not statistically significant. This result reflects the strong performance of lexical retrieval in
detector-domain literature, where detector names, acronyms and operating-condition terminology
remain highly informative.
Hybrid retrieval provides the strongest strict chunk-level evidence ranking across both benchmarks,
reaching Hit@5 of 0.917 on the core benchmark and 0.951 on the extension benchmark. BM25
12

Table 6:Paired Wilcoxon signed-rank tests comparing hybrid retrieval against alternative retrieval
methods using query-level strict Hit@5 results on the answerable benchmark queries.
Comparisonp-value
Hybrid vs BM25 0.788
Hybrid vs Dense 3.24×10−3
Hybrid vs Graph 5.52×10−7
Hybrid vs Graph-path 5.52×10−7
Hybrid vs Agentic graph 1.62×10−7
Figure 3:Strict Hit@5 and MRR for the core and extension benchmarks. Hybrid retrieval provides the
strongest strict chunk-level evidence retrieval, while BM25 remains highly competitive because of exact
detector terminology.
remains highly competitive, reflecting the acronym-rich and terminology-stable nature of detector
literature. Dense retrieval performs worse under strict chunk-level evaluation, although its
Paper@5 scores show that it frequently retrieves the correct publication while missing the
exact supporting passage. Graph, graph-path and agentic graph configurations substantially
underperform the lexical and hybrid baselines under strict evaluation. The agentic graph
configuration performs slightly worse than the graph-path baseline, indicating that additional
query decomposition does not compensate for the loss of ranking precision introduced by graph
expansion. Their Paper@5 and Soft@5 scores remain considerably higher than their strict Hit@5
values, indicating that graph expansion often retrieves semantically related passages or correct
source papers without ranking the exact gold chunk highly enough. This distinction between
semantic exploration and exact evidence retrieval is a recurring feature of the detector-domain
corpus.
Figure 5 confirms that the sparse and dense retrieval components are complementary. BM25
provides the dominant strict-matching signal, while dense retrieval improves coverage for para-
phrased detector-physics queries and semantically related detector concepts. The full hybrid
configuration therefore provides the most robust retrieval performance across the benchmark.
Figure 6 shows retrieval performance as a function of reasoning complexity. All retrieval
configurations perform strongly on direct factual retrieval and single-paper parameter queries,
while performance decreases for multi-evidence synthesis and detector-physics reasoning tasks.
Hybrid retrieval remains the most stable configuration across all reasoning levels, indicating
that combining lexical and semantic retrieval signals improves robustness as query complexity
increases.
Figure 7 provides a complementary view by grouping queries according to detector topic rather
than reasoning complexity. Queries involving detector technologies, irradiation studies, and
timing measurements generally achieve the strongest retrieval performance, reflecting the stable
13

Figure 4:Strict Hit@k profiles for the core and extension benchmarks. The hybrid method consistently
reaches the highest or joint-highest recall at larger k, while graph-heavy configurations underperform on
strict chunk matching across allk.
Figure 5:Ablation of hybrid retrieval components. Removing BM25 or dense retrieval reduces strict
Hit@5. The full hybrid configuration provides the strongest performance, with BM25 contributing exact
terminology matching and dense retrieval contributing complementary semantic coverage.
terminology and well-defined evidence structure of these topics. Lower performance is observed
for broader detector-physics and cross-technology comparison queries, where supporting evidence
is often distributed across multiple publications.
Graph-based retrieval exhibits a markedly different behaviour from the lexical and hybrid
baselines. While strict chunk-level retrieval performance is substantially lower, graph-guided
methods achieve comparatively stronger source-level and semantic coverage. Figure 8 illustrates
this behaviour. As graph expansion becomes more aggressive, semantically related detector
concepts are increasingly retrieved, but exact supporting evidence is displaced by neighbouring
passages. In practice, graph expansion frequently identifies publications and passages that
are relevant to the broader detector concept, even when the exact supporting evidence is not
ranked highly enough to satisfy strict evaluation criteria. This behaviour suggests that graph
retrieval is better suited to literature exploration and detector-concept navigation than to exact
evidence ranking. Figure 9 further illustrates the distinction between strict evidence retrieval and
broader literature discovery. While graph-based methods perform poorly under strict chunk-level
14

Figure 6:Retrieval performance as a function of reasoning complexity. Performance generally decreases
as queries require increasingly complex evidence synthesis, while hybrid retrieval remains the most stable
configuration across all reasoning levels.
Figure 7:Retrieval performance across detector-domain query categories. Different detector topics
exhibit different retrieval characteristics, reflecting variations in terminology stability, evidence density,
and conceptual complexity.
evaluation, their source-level and semantic retrieval scores remain substantially higher. This
indicates that graph expansion frequently reaches the correct publication or a semantically
related passage, but does not reliably rank the exact supporting evidence highly enough for strict
retrieval tasks. The result reinforces the view that graph retrieval functions most effectively as
an exploratory literature-navigation tool rather than a primary evidence ranker.
Failure analysis.Table 7 summarises the dominant failure modes observed for the hybrid
retriever. Most remaining errors arise from conceptual synthesis and cross-technology comparison
queries rather than terminology matching. In these cases, the required evidence is distributed
across multiple publications and cannot always be recovered through retrieval of a single support-
ing passage. Terminology ambiguity contributes comparatively little to the residual error rate,
indicating that lexical and hybrid retrieval largely resolves detector-specific vocabulary matching.
The principal remaining limitation therefore lies in higher-level scientific reasoning rather than
first-stage evidence retrieval. These observations suggest that future improvements are more
likely to arise from multi-passage evidence synthesis and reasoning-aware retrieval strategies than
15

Figure 8:Effect of graph expansion on strict retrieval performance. While graph traversal increases
semantic coverage, excessive expansion dilutes exact evidence ranking and reduces strict chunk-level
retrieval accuracy.
Figure 9:Comparison of strict chunk-level retrieval, source-level retrieval, and semantic soft-gold retrieval.
Graph-based retrieval recovers relevant publications and semantically related passages more effectively
than exact supporting evidence.
from further optimisation of lexical matching.
Retrieval-performance trade-off.Figure 11 summarises the trade-off between retrieval
effectiveness and query latency. Dense retrieval provides the lowest latency but substantially
weaker strict evidence retrieval. BM25 achieves strong retrieval performance with moderate
latency, reflecting the stability of detector terminology and acronyms. Hybrid retrieval occupies
the most favourable operating point, achieving the highest strict Hit@5 while introducing only
a modest latency increase relative to BM25. In contrast, graph-based retrieval configurations
require additional graph-expansion and query-processing stages but do not improve strict retrieval
performance. These results further support hybrid retrieval as the preferred default configuration
for evidence-grounded detector-literature access.
Latency decomposition.Figure 10 decomposes per-query latency by pipeline stage. Sparse
and dense retrieval account for most of the cost in the BM25, dense, and hybrid configurations, so
hybrid retrieval inherits only a small fusion overhead on top of its two component retrievers. The
16

Table 7:Failure taxonomy for hybrid retrieval on the 48 answerable core benchmark queries.
Failure category Count Fraction
Conceptual synthesis queries 3 75%
Cross-technology comparison 1 25%
Detector-condition mismatch 0 0%
Acronym ambiguity 0 0%
Total failures 4 100%
Figure 10:Per-query latency decomposition by pipeline stage. Sparse and dense search dominate the
retrieval cost, while graph configurations add further graph-expansion and query-decomposition stages.
The breakdown shows where the additional latency of graph-based retrieval originates rather than its
total cost alone.
graph and agentic-graph configurations add distinct graph-expansion and query-decomposition
stages on top of the same retrieval backbone, which is the origin of their higher latency seen in
Figure 11. This decomposition clarifies that the additional cost of graph-based retrieval comes
from contextual expansion rather than from the core evidence-retrieval step.
4.3 Abstention behaviour on negative queries
Beyond strict evidence ranking, a trustworthy detector-literature assistant must recognise when
the corpus does not support a requested claim and abstain rather than generate unsupported
conclusions. The negative queries in each benchmark are therefore constructed so that no gold
evidence passage exists in the corpus. The system is expected to return a predefined abstention
marker when the retrieved evidence is insufficient to support the queried detector-domain claim,
and abstention accuracy is measured as the fraction of unsupported queries on which the system
correctly abstains.
On the 12 negative queries of the core benchmark, the hybrid configuration achieves an abstention
accuracy of 1.0, correctly declining to answer every unsupported query under the grounded-
generation prompt. This indicates that, when retrieval fails to identify supporting evidence,
the grounding constraint reliably triggers abstention rather than unsupported synthesis. Such
behaviour is particularly important for detector R&D applications, where fabricated detector-
17

Figure 11:Trade-off between strict retrieval performance and query latency. Hybrid retrieval achieves
the strongest strict Hit@5 while introducing only a modest latency increase relative to BM25. Graph-based
retrieval configurations incur additional latency without improving strict evidence-ranking performance.
performance claims or unsupported technology comparisons could lead to misleading scientific
conclusions.
Figure 12 illustrates the effect of graph expansion on abstention behaviour. Graph-guided
retrieval is more likely to surface semantically related passages even when the corpus does not
contain evidence supporting the queried claim. While such behaviour can be useful for literature
exploration, it weakens the abstention signal required for reliable evidence-grounded retrieval.
This observation is consistent with the strict-ranking results and further supports the use of
graph expansion as an optional exploratory layer rather than the default evidence ranker.
This study focuses on retrieval quality and abstention behaviour rather than end-to-end answer
scoring. In the detector-literature setting, retrieval accuracy is the primary determinant of
evidence-grounded behaviour because answer generation is constrained to operate only on
retrieved evidence. Evaluating retrieval quality therefore provides a direct assessment of the
reliability of the underlying literature-access framework for silicon detector R&D.
5 Discussion and Conclusion
This work presents a grounded evidence-retrieval benchmark and reproducible hybrid retrieval
framework for silicon pixel detector literature. The benchmark targets a practical detector-
R&D problem: locating source-attributed, passage-level evidence within a specialised literature
corpus characterised by stable detector terminology, configuration-dependent measurements, and
detector-physics reasoning. To support this task, we construct a detector-domain benchmark
with strict chunk-level annotations, source-level diagnostics, semantic soft-gold evaluation, and
negative-query abstention tests, together with a retrieval framework combining BM25 sparse
retrieval, dense semantic retrieval, hybrid reciprocal-rank fusion, and graph-guided retrieval
strategies.
18

Figure 12:Graph coverage and abstention behaviour on negative queries. Graph-based retrieval shows
higher false-positive rates than lexical and hybrid retrieval because entity expansion can retrieve plausible
but unsupported neighbouring evidence.
The principal finding is that hybrid sparse–dense retrieval provides the strongest strict evidence-
ranking performance. Hybrid retrieval achieves Hit@5 values of 0.917 on the core benchmark and
0.951 on the extension benchmark, consistently outperforming dense retrieval and all graph-based
retrieval configurations under strict chunk-level evaluation. BM25 remains highly competitive
because of the lexical stability of detector terminology and acronyms, while dense retrieval
improves robustness to paraphrased detector-physics descriptions. In contrast, graph-based
retrieval substantially improves semantic exploration and source discovery but does not improve
strict passage-level evidence ranking. Similarly, the agentic graph configuration increases system
complexity through query decomposition without providing a measurable gain in strict evidence
retrieval, suggesting that decomposition alone is insufficient when the primary limitation lies in
evidence ranking rather than query understanding. The results therefore reveal a clear distinction
between literature exploration and exact evidence retrieval within detector instrumentation
literature.
This distinction has practical implications for detector R&D. For detector instrumentation studies,
the most useful retrieval system is not necessarily the one that retrieves the broadest semantic
neighbourhood, but the one that reliably identifies the exact supporting evidence under the
relevant detector configuration and operating conditions. Detector-performance claims often
depend on specific irradiation fluences, bias voltages, temperatures, readout settings, and sensor
geometries, making precise evidence attribution essential. This requirement is expected to become
increasingly important for future detector-development programmes, where design decisions must
be informed by evidence accumulated across multiple sensor technologies, irradiation campaigns,
simulation studies, and beam-test measurements.
Several limitations remain. The benchmark labels were produced through a primary annotation
pass and do not yet include full multi-annotator adjudication across all query–chunk pairs.
However, the calibrated second-annotation check described in Section 4 gives Cohen’s κ= 0.619
on 300 sampled pairs, providing a quantitative reliability check for the strict relevance criterion.
The reported metrics should therefore still be interpreted under the current annotation scheme
and may shift under full reconciliation. The negative queries were constructed to be unsupported
by the corpus, but they were not independently exhaustively verified against every chunk. The
corpus is restricted to text-extracted evidence from published literature, and the dense retriever
uses a general-purpose sentence embedding model rather than a detector-domain encoder.
19

Overall, the results demonstrate that detector-domain retrieval should be evaluated at the level
of supporting evidence rather than only at the level of documents or generated answers. By
providing a benchmark, evaluation framework, and reproducible retrieval pipeline for silicon
detector literature, this work establishes a foundation for systematic evaluation of evidence-
grounded retrieval systems in detector R&D and future collider instrumentation studies.
Acknowledgements
This work is supported in part by the ... funding.
References
[1]G. Pellegrini, P. Fern´ andez-Mart´ ınez, M. Baselga, C. Fleta, D. Flores, V. Greco, S. Hidalgo,
A. Merlos, I. Mandi´ c, D. Quirion, et al. Technology developments and first measurements
of Low Gain Avalanche Detectors (LGAD) for high energy physics applications.Nuclear
Instruments and Methods in Physics Research Section A, 765:12–16, 2014. doi: 10.1016/j.
nima.2014.06.008.
[2]W. Snoeys. CMOS monolithic active pixel sensors for high energy physics.Nuclear
Instruments and Methods in Physics Research Section A, 765:167–171, 2014. doi: 10.1016/j.
nima.2014.05.070.
[3]H. Pernegger et al. First tests of a novel radiation hard CMOS sensor process for depleted
monolithic active pixel sensors.Journal of Instrumentation, 12:P06008, 2017. doi: 10.1088/
1748-0221/12/06/P06008.
[4]Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung.
Survey of hallucination in natural language generation.ACM computing surveys, 55(12):
1–38, 2023.
[5]P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. K¨ uttler, M. Lewis,
W.-t. Yih, T. Rockt¨ aschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InAdvances in Neural Information Processing Systems,
volume 33, pages 9459–9474, 2020.
[6]Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang.
Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2023.
[7] A. Mallen, A. Asai, V. Zhong, R. Das, D. Khashabi, and H. Hajishirzi. When not to trust
language models: Investigating effectiveness of parametric and non-parametric memories. In
Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics,
pages 9802–9822, 2023. doi: 10.18653/v1/2023.acl-long.546.
[8]N. Reimers and I. Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT-
networks. InProceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing,
pages 3982–3992, 2019. doi: 10.18653/v1/D19-1410.
[9]V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih.
Dense passage retrieval for open-domain question answering. InProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing, pages 6769–6781, 2020.
doi: 10.18653/v1/2020.emnlp-main.550.
20

[10]N. Thakur, N. Reimers, A. R¨ uckl´ e, A. Srivastava, and I. Gurevych. BEIR: A heterogeneous
benchmark for zero-shot evaluation of information retrieval models. InAdvances in Neural
Information Processing Systems: Datasets and Benchmarks, 2021.
[11]S. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and
beyond.Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi:
10.1561/1500000019.
[12]C. D. Manning, P. Raghavan, and H. Sch¨ utze.Introduction to Information Retrieval.
Cambridge University Press, Cambridge, 2008. doi: 10.1017/CBO9780511809071.
[13]G. V. Cormack, C. L. A. Clarke, and S. Buettcher. Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. InProceedings of the 32nd International
ACM SIGIR Conference on Research and Development in Information Retrieval, pages
758–759, 2009. doi: 10.1145/1571941.1572114.
[14]J. Lin and X. Ma. A few brief notes on DeepImpact, COIL, and a conceptual framework for
learned sparse retrieval.arXiv preprint arXiv:2106.14807, 2021.
[15]D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and J. Larson. From
local to global: A graph RAG approach to query-focused summarization.arXiv preprint
arXiv:2404.16130, 2024.
[16]Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang. LightRAG: Simple and fast retrieval-augmented
generation.arXiv preprint arXiv:2410.05779, 2024.
[17]M. Yasunaga, H. Ren, A. Bosselut, P. Liang, and J. Leskovec. QA-GNN: Reasoning with
language models and knowledge graphs for question answering. InProceedings of the 2021
Conference of the North American Chapter of the Association for Computational Linguistics,
pages 535–546, 2021. doi: 10.18653/v1/2021.naacl-main.45.
[18]W. Xiong, X. L. Li, S. Iyer, J. Du, P. Lewis, W. Y. Wang, Y. Mehdad, W.-t. Yih, S. Riedel,
D. Kiela, and B. Oguz. Answering complex open-domain questions with multi-hop dense
retrieval. InInternational Conference on Learning Representations, 2021.
[19]G. Izacard and E. Grave. Leveraging passage retrieval with generative models for open
domain question answering. InProceedings of the 16th Conference of the European Chapter
of the Association for Computational Linguistics, pages 874–880, 2021. doi: 10.18653/v1/
2021.eacl-main.74.
[20]R. Jiang, D. Fu, C. Jiang, T. Yang, Z. Wang, Y. Wu, Y. Ban, Y. Mao, and Q. Li. Agentic
Hybrid RAG for Evidence-Grounded Muon Collider Analysis. 6 2026.
[21]O. Khattab and M. Zaharia. ColBERT: Efficient and effective passage search via contextu-
alized late interaction over BERT. InProceedings of the 43rd International ACM SIGIR
Conference on Research and Development in Information Retrieval, pages 39–48, 2020. doi:
10.1145/3397271.3401075.
[22]D. Wadden, S. Lin, K. Lo, L. L. Wang, M. van Zuylen, A. Cohan, and H. Hajishirzi. Fact
or fiction: Verifying scientific claims. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing, pages 7534–7550, 2020. doi: 10.18653/v1/2020.
emnlp-main.609.
[23]A. Cohan, S. Feldman, I. Beltagy, D. Downey, and D. S. Weld. SPECTER: Document-level
representation learning using citation-informed transformers. InProceedings of the 58th
Annual Meeting of the Association for Computational Linguistics, pages 2270–2282, 2020.
doi: 10.18653/v1/2020.acl-main.207.
21

[24]G. Tsatsaronis et al. An overview of the BioASQ large-scale biomedical semantic indexing
and question answering competition.BMC Bioinformatics, 16:138, 2015. doi: 10.1186/
s12859-015-0564-6.
[25]P. Lopez. GROBID: Combining automatic bibliographic data recognition and term extraction
for scholarship publications. InResearch and Advanced Technology for Digital Libraries,
pages 473–474, 2009. doi: 10.1007/978-3-642-04346-8 62.
[26]J. Johnson, M. Douze, and H. J´ egou. Billion-scale similarity search with GPUs.IEEE
Transactions on Big Data, 7(3):535–547, 2021. doi: 10.1109/TBDATA.2019.2921572.
[27]M. Mager. ALPIDE, the monolithic active pixel sensor for the ALICE ITS upgrade.
Nuclear Instruments and Methods in Physics Research Section A, 824:434–438, 2016. doi:
10.1016/j.nima.2015.09.057.
[28]T. Poikela, J. Plosila, T. Westerlund, M. Campbell, M. De Gaspari, X. Llopart, V. Gromov,
R. Kluit, M. van Beuzekom, F. Zappon, et al. Timepix3: A 65k channel hybrid pixel readout
chip with simultaneous ToA/ToT and sparse readout.Journal of Instrumentation, 9:C05013,
2014. doi: 10.1088/1748-0221/9/05/C05013.
[29]S. Spannagel et al. Allpix squared: A modular simulation framework for silicon detectors.
Nuclear Instruments and Methods in Physics Research Section A, 901:164–172, 2018. doi:
10.1016/j.nima.2018.06.020.
[30]S. Agostinelli et al. GEANT4: A simulation toolkit.Nuclear Instruments and Methods in
Physics Research Section A, 506(3):250–303, 2003. doi: 10.1016/S0168-9002(03)01368-8.
Appendix
A Grounding and Abstention Prompt
The grounded generation component receives the user query together with the retrieved evidence
passages and associated source metadata. The model is instructed to answer only from the
supplied evidence, preserve source traceability, and abstain when the retrieved material does not
support the requested detector-domain claim.
The prompt follows the general structure:
You are a detector-literature assistant.
Use only the supplied evidence passages.
Requirements:
1. Do not use external knowledge.
2. Cite supporting passages when answering.
3. If the evidence is insufficient,
return ABSTAIN.
4. Do not infer detector performance
beyond what is explicitly stated.
Question:
{query}
Retrieved Evidence:
22

{retrieved_chunks}
Answer:
For negative-query evaluation, the model is expected to return the predefined token
ABSTAIN
when no retrieved passage directly supports the queried detector-domain claim.
B Retrieval Metrics
The primary evaluation metrics used throughout this work are strict chunk-level Hit@5 and
Mean Reciprocal Rank (MRR).
Hit@5.Hit@5 evaluates whether at least one gold evidence chunk appears among the top five
retrieved passages:
Hit@5 =1
NNX
i=11(rank i≤5),(4)
where Nis the number of benchmark queries and rank iis the rank of the first gold chunk for
queryi.
Mean Reciprocal Rank.MRR measures the average reciprocal rank of the first retrieved
gold passage:
MRR =1
NNX
i=11
rank i.(5)
Paper@5.Paper@5 evaluates whether the correct source publication is retrieved within the
top five results, regardless of whether the exact gold chunk is returned.
Soft@5.Soft@5 evaluates retrieval of semantically relevant supporting evidence that may not
exactly match the manually annotated gold chunk.
Abstention Accuracy.For negative queries, abstention accuracy is defined as
Abstention =Ncorrect abstain
Nnegative,(6)
whereN negative is the number of unsupported benchmark queries.
C Reproducibility Commands
The retrieval benchmarks are reproduced with the following commands. The retrieval and
grounded-literature-access entry point is chat.py ; the benchmark driver evaluate jinst.py
runs all six retrieval configurations against the chunk-level gold benchmark and reports strict,
paper-level, soft-gold, and abstention metrics.
23

Core 60-query benchmark.
python evaluate_jinst.py \
--benchmark data/eval/benchmark_jinst_gold_final.json \
--chat ./chat.py \
--configs bm25,dense,hybrid,graph,graph_path,agentic_graph \
--top_k 10 \
--candidate_k 80 \
--generate \
--out_dir data/eval/runs/jinst_eval_gold_final_rerank_v2
Extension 56-query benchmark.
python evaluate_jinst.py \
--benchmark data/eval/benchmark_pixel_extension56_final_v1.json \
--chat ./chat.py \
--configs bm25,dense,hybrid,graph,graph_path,agentic_graph \
--top_k 10 \
--candidate_k 80 \
--generate \
--out_dir data/eval/runs/pixel_extension56_final_full_v2
Single-query interactive retrieval and grounded response.
python chat.py \
--query "Time resolution of irradiated LGAD sensors" \
--mode hybrid \
--top_k 10 \
--json
Figure and table generation.
python make_paper_graphs.py
Each output directory contains the summary file, per-query metric file, and retrieved-evidence
payloads used to audit the reported numbers:
summary.csv
per_query_metrics.csv
payloads/
The reported core benchmark results correspond to
data/eval/runs/jinst_eval_gold_final_rerank_v2/
and the reported extension benchmark results correspond to
data/eval/runs/pixel_extension56_final_full_v2/
24