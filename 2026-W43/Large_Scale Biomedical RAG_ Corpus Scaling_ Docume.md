# Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs

**Authors**: Adrián Sánchez-Freire, Gonzalo Pérez-Gómez, Aitziber Atutxa, Ander Barrena

**Published**: 2026-10-19 13:08:38

**PDF URL**: [https://arxiv.org/pdf/2610.26339v1](https://arxiv.org/pdf/2610.26339v1)

## Abstract
We present a controlled, large-scale evaluation of how corpus scale, document representation, and reranking affect the effectiveness and efficiency of biomedical Retrieval-Augmented Generation (RAG) systems. Retrieval collections were progressively expanded while retaining the relevant source documents, enabling consistent comparisons across corpus sizes. We compared abstract, flat full-text, and hierarchical full-text representations across first-stage retrieval, reranking, and end-to-end multiple-choice question answering. Configurations selected on development data were evaluated on five external biomedical benchmarks and a held-out test set, while retrieval latency and index storage were measured to quantify operational cost. Performance patterns observed with small or controlled collections did not consistently persist as collections grew, and the usefulness of reranking depended on the corpus and retrieval setting. RAG improved three of four model configurations across the external benchmarks, with mean gains of up to 9.6 percentage points, and improved all four configurations on the held-out test set by 10.2-30.3 points. Abstract retrieval performed best when reference evidence was abstract-aligned, whereas full-text representations were generally more effective when richer evidence was needed. Hierarchical representation matched or outperformed flat full-text retrieval in most end-to-end comparisons while requiring 67% fewer indexed units and 44-56% less storage. These findings show that corpus scale should be treated as an explicit evaluation variable in biomedical RAG, and that the most appropriate design depends on evidence format, model behavior, and operational constraints.

## Full Text


<!-- PDF content starts -->

Large-Scale Biomedical RAG: Corpus Scaling, Document
Representation, and Efficiency Trade-offs
Adrián Sánchez-Freirea,∗, Gonzalo Pérez-Gómeza, Aitziber Atutxaaand Ander Barrenaa
aHiTZ Basque Center for Language Technology, University of the Basque Country (EHU), P.ºRafael Moreno "Pitxitxi", 2, Bilbao, 48013, Biscay, Spain
ARTICLE INFO
Keywords:
Retrieval-augmented generation
Corpus scaling
Document representation
Hierarchical retrieval
Retrieval efficiencyABSTRACT
We present a controlled, large-scale evaluation of how corpus scale, document representation, and
reranking affect the effectiveness and efficiency of biomedical Retrieval-Augmented Generation
(RAG) systems. Retrieval collections were progressively expanded while retaining the relevant source
documents, enabling consistent comparisons across corpus sizes. We compared abstract, flat full-
text,andhierarchicalfull-textrepresentationsacrossfirst-stageretrieval,reranking,andend-to-end
multiple-choice question answering. Configurations selected on development data were evaluated on
five external biomedical benchmarks and a held-out test set, while retrieval latency and index storage
weremeasuredtoquantifyoperationalcost.Performancepatternsobservedwithsmallorcontrolled
collectionsdidnotconsistentlypersistascollectionsgrew,andtheusefulnessofrerankingdependedon
the corpus and retrieval setting. RAG improved three of four model configurations across the external
benchmarks, with mean gains of up to 9.6 percentage points, and improved all four configurations on
theheld-outtestsetby10.2–30.3points.Abstractretrievalperformedbestwhenreferenceevidence
was abstract-aligned, whereas full-text representations were generally more effective when richer
evidencewasneeded.Hierarchicalrepresentationmatchedoroutperformedflatfull-textretrievalin
mostend-to-endcomparisonswhilerequiring67%fewerindexedunitsand44–56%lessstorage.These
findingsshowthatcorpusscaleshouldbetreatedasanexplicitevaluationvariableinbiomedicalRAG,
and that themost appropriatedesign dependson evidence format, modelbehavior, and operational
constraints.
1. Introduction
Large language models (LLMs) are increasingly used
for biomedical and medical question answering, but answers
based only on parametric knowledge may be incomplete,
outdated, or difficult to verify. Retrieval-Augmented Genera-
tion (RAG) addresses this limitation by providing external
documents at inference time, allowing the generator to
condition its answer on explicit evidence rather than only
on information encoded in its parameters [1].
TheeffectivenessofbiomedicalRAGdependsnotonly
on the generator, but also on the retrieval corpus and on
how its documents are represented. Expanding the corpus
may increase the availability of relevant evidence, but it
also introduces additional distractors and raises retrieval
and storage costs. Document representation creates a related
trade-off.Abstractsarecompactandefficient,butmayomit
answer-relevant details. Full-text retrieval preserves richer
evidence, but substantially increases the number of indexed
units and may fragment information across independent
chunks.Hierarchicalrepresentationsprovideanalternativeby
retainingrelationshipsbetweendifferentpartsofadocument.
These factors cannot be considered independently. As
the source corpus grows, each representation produces a
retrieval index of a different size and structure, which may
affectfirst-stageretrieval,reranking,theevidenceprovided
tothegenerator,andultimatelythequalityofthegenerated
answer. A representation that improves evidence coverage
maythereforebeimpracticalbecauseofitsstorageorlatency
∗Corresponding author
adrian.sanchez@ehu.eus(A. Sánchez-Freire)requirements,whilearerankerthatimprovesthefinalranking
may not necessarily improve downstream answer accuracy.
This study investigates these interactions through a
controlled evaluation of biomedical RAG systems under
increasingcorpusscale.Wedistinguishthenumberofsource
documents from the representation-dependent number of
retrievableunitsandcomparethreedocumentrepresentations:
abstracts, flat full-text chunks, and TreeRAG-style hierarchi-
cal representations [ 2]. The experiments use PubMed as a
largecollectionofrealbiomedicalliteratureandMIRIAD[ 3]
as a controlledsetting with question-linked source passages.
WeevaluatetheRAGpipelinebothmodularlyandend-to-
end.First-stageretrievalandrerankingareassessedseparately
todeterminehowcorpusgrowthaffectsevidenceselection.
Downstream performance is evaluated through multiple-
choice question answering (MCQA), using development
datasets, external biomedical benchmarks, and a held-out
MIRIAD test split. We compare LLM-only and RAG con-
figurations across compact general-purpose, biomedical, and
reasoning-oriented generators, and use a dataset-specific
Oracleconditiontocharacterizegenerationwhenreference
evidence is directly available. Retrieval latency and index
storage are also measured to quantify the practical cost of
each configuration.
Tothebestofourknowledge,existingbiomedicalRAG
studiescommonlyevaluaterelativelysmallortask-specific
retrievalcollections,whileasmallernumberuselargeindexes.
However, these systems are generally assessed at a single
fixed corpus scale, and corpus growth is rarely analyzed
jointly with document representation, retrieval, reranking,
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 1 of 15
arXiv:2610.26339v1  [cs.CL]  19 Oct 2026

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
downstreamgeneration,andefficiency.Consequently,itre-
mainsunclearwhetherconclusionsobtainedatonecollection
sizeremainvalidasbiomedicalknowledgeresourcesgrow.
Thisstudyaddressesthisgapthroughacontrolled,system-
levelempirical evaluationofbiomedical RAG undercorpus
growth.
This work makes the following contributions:
•It introduces a controlled corpus-scaling protocol that
retains gold sources while progressively adding non-
golddocumentsandseparatessource-documentscale
from representation-dependent retrieval-index scale.
•It conducts an end-to-end evaluation linking corpus
growth and document representation to first-stage
retrieval, reranking, downstream MCQA accuracy,
latency, and storage.
•It provides empirical evidence that advantages ob-
served at smaller corpus scales do not consistently
persistastheretrievalcollectiongrows,demonstrating
the limitations of evaluating biomedical RAG at a
single fixed scale.
•It derives empirically grounded guidance for selecting
biomedical RAG configurations according to evidence
characteristics, corpus scale, and operational con-
straints.
Together, these contributions characterize the robustness
andoperationaltrade-offsofbiomedicalRAGconfigurations
under controlled corpus growth rather than at an isolated
collection size. The resulting evidence is intended to support
researchers and developers designing scalable, literature-
grounded RAG systems for biomedical applications.
2. Related Work
2.1. Retrieval-Augmented Biomedical Question
Answering
Retrieval-Augmented Generation (RAG) conditionslan-
guagemodelsoninformationretrievedfromexternalsources
for knowledge-intensive tasks [ 1]. Subsequent work has
explored large retrieval memories, in-context and active
retrieval,andmodularretrieval–generationarchitectures[ 4–
7]. Surveys describe this broader design space and its role in
grounding outputs and incorporating external knowledge [ 8,
9].
Biomedical question answering has evolved from task-
specific and retrieval-based systems [ 10–12] to domain-
pretrained encoders [ 13–15] and general-purpose or med-
icallyspecialized LLMs[ 16–18].Medical QAbenchmarks,
including those consolidated in MIRAGE [ 19], provide
a standardized multiple-choice setting for evaluating how
retrieval choices affect final answer accuracy.
2.2. Document Representation in RAG
Passage-level retrieval makes search over large collec-
tionstractableandallowsgeneratorstousemultipleevidenceunits [20,21]. However, fixed-size segmentation can split
relevantevidence,removedocument-levelcontext,andcreate
redundantunits.Retrievalgranularitythereforeaffectsboth
retrieval and downstream generation [22, 23].
This issue is especially relevant for scientific literature.
Abstracts provide compact summaries, whereas full-text
articlesmaycontainevidenceabsentfromtheabstract[ 24].
Hierarchical and long-document methods instead seek to
preserve broader document structure [ 25,26]. TreeRAG
organizes documents into tree-structured retrieval units,
enabling comparison between abstract-level, flat full-text,
and hierarchical representations [2].
2.3. Corpus Scaling and Efficiency
Large RAG systems commonly use fixed retrieval col-
lections ranging from millions of passages to much larger
external memories [ 4,20,21,27]. General-domain scaling
studies show that larger collections can improve evidence
coverage, but with diminishing returns [28].
Biomedical retrieval-augmented systems have been eval-
uated across different QA tasks, corpora, retrieval strategies,
and evidence representations, including the studies of Otegi
et al. [29], Xiong et al. [ 19], Alonso et al. [ 30], Vladika and
Matthes [ 31], Li et al. [ 32], Lopez et al. [ 33], and Sohn et
al. [34]. Stuhlmann et al. [ 35] additionally examine retrieval
efficiency over a large PubMed-derived collection. Context-
selection studies further show that additional retrieved ma-
terial may introduce distractors or unfavorable retrieval–
generation trade-offs [36, 37].
Table 1 summarizes these lines of work. Existing studies
investigate individual aspects of corpus scale, document rep-
resentation,ranking,andefficiency,butdonotjointlyanalyze
their interaction under controlled biomedical corpus growth.
Tothebestofourknowledge,thisworkisthefirsttounify
these dimensions within a single experimental framework,
enablingasystematicstudyofthetrade-offsamongcorpus
scale,documentrepresentation,retrievalperformance,and
computational efficiency.
3. Methods
3.1. Datasets and Task Format
We formulate all tasks as multiple-choice question an-
swering (MCQA), enabling consistent evaluation through
accuracy across datasets and experimental configurations.
Developmentexperimentsusetwodatasets.MedAESQA
is a clinically grounded QA dataset containing 40 expert-
curated questions supported by biomedical evidence [ 38].
The complete dataset is used for development.MIRIAD
isalargesyntheticbiomedicalQAdatasetcontainingauto-
maticallygeneratedquestion–answerpairslinkedtosource
passages [ 3]. We sample 1,000 instances and divide them
into 200 development and 800 test examples.1
MedAESQAandMIRIADareoriginallyformulatedas
open-ended QA tasks. We convert them to MCQA while
1Thesamplingscriptisavailableinthemainexperimentalrepository;
see the Data Availability section.
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 2 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
Reference Max. index Biomed. Scaling Doc. repr. Efficiency
Borgeaud et al. [4] 28B units –∼– –
Izacard et al. [27] 387M units – – – –
Ning et al. [28] 264M units –✓–✓
Tao et al. [2] n.r.∼–✓–
Otegi et al. [29] 400K units✓–✓–
Xiong et al. [19] 54.2M units✓– – –
Alonso et al. [30] 54.2M units✓– – –
Vladika and Matthes [31] 20.6M units✓– – –
Stuhlmann et al. [35] 24M units✓– –✓
Li et al. [32] n.r.✓–∼–
Lopez et al. [33] 20K units✓–✓ ✓
Sohn et al. [34] 116.7M units✓– – –
This work 117.4M units✓ ✓ ✓ ✓
Table 1
Comparison with representative retrieval-augmented studies.Max. indexreports the largest retrieval collection evaluated by each
study in its reported indexing unit. The remaining columns indicate biomedical or medical evaluation, controlled corpus scaling,
document-representation comparison, and efficiency analysis. ✓denotes explicit coverage, ∼partial or indirect coverage, – not
addressed, and n.r. not reported.
preservingtheoriginalreferenceanswerasthecorrectoption
and generating three incorrect distractors. The conversion
pipelineconsistsofthreestages:generatingmedicallyplausi-
blecandidates,rewritingthemtoreducesuperficialcuessuch
asdifferencesinlengthorsyntax,andfilteringandranking
the candidates to select the final distractors. Correct-answer
positionsarebalancedacrossoptionlabels.Implementation
details, including model configuration, decoding parameters,
andfilteringheuristics,areprovidedinSupplementarySec-
tion B.
Final evaluation additionally uses MMLU-Med [ 39],
MedQA-US [ 40], MedMCQA [ 41], PubMedQA [ 42], and
BioASQ yes/no [ 43]. We use the preprocessed versions
introduced in MIRAGE [ 19], with minor modifications to
align them with our evaluation pipeline. MedAESQA and
the 200-instance MIRIAD development split are used for
model selection and configuration decisions. The external
benchmarks and the remaining 800 MIRIAD instances are
used exclusively for testing.
3.2. Corpora and Scaling Protocol
Retrieval experiments use two document collections.
PubMedprovidestheretrievalcorpusforMedAESQAand
the external biomedical benchmarks, whereas MIRIAD
source passages are used exclusively for MIRIAD experi-
ments.
Weconstructcorpusscalesbyprogressivelyincreasing
thenumberofnon-golddocumentsintheretrievalpoolwhile
retaining the available gold sources. Five shared scaling
factorsareappliedindependentlytoPubMedandMIRIAD:
control,tiny,small,medium, andlarge. The absolute corpus
sizes differ, but the shared factors provide corresponding
corpus-growth conditions across the two collections. The
largescale corresponds to the complete MIRIAD corpus.
Because PubMed can be expanded further, we include an
additionalfullsettingcontainingapproximately4.3million
source documents.Thescalingprotocolisdefinedatthesource-document
level. However, the number of units available to the retriever
depends on how each document is represented. Table 2
therefore reports both source documents and the resulting
representation-dependent index sizes.
3.3. Document Representations
We compare three document representations:
•Abstract-levelrepresentation.Eachsourcedocument
is represented by a single compact retrieval unit.
•Full-text chunking.Source documents are divided
into chunks of up to 1,600 characters with a 200-
character overlap using a recursive character-based
splitter.
•Tree-based representation.Source documents are
organized into hierarchical retrieval units using
TreeRAG [2].
For PubMed, all three representations are constructed
from the same set of publicly available full-text articles. The
abstract-level representation uses the original abstract associ-
atedwitheacharticle,whereasthefull-textcontentisused
to construct the flat chunks and TreeRAG representations.
ForMIRIAD,theoriginalsourcepassagesserveasthe
full-text documents. Because MIRIAD does not provide
separate abstracts, we generate LLM-based pseudo-abstracts
from these passages. Passages containing at most 300 tokens
are kept unchanged because they already match the intended
abstract-level granularity. Full-text chunks and TreeRAG
units are constructed from the original passages. Further
detailsonpseudo-abstractgenerationandTreeRAGconstruc-
tion are provided in Supplementary Sections C and D.
3.4. Models
We evaluate models with different architectures and
parameterscalesateachstageoftheRAGpipeline.First-stage
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 3 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
Retrieval units
Corpus Scale Factor (×) Source docsAbstract Full-text TreeRAG
PubMed Control 1.0 0.3K 0.3K 11.2K 2.6K
Tiny 6.6 1.8K 1.8K 53.4K 15.9K
Small 44.1 11.8K 11.8K 330.6K 105.7K
Medium 292.6 78.4K 78.4K 2.2M 700.2K
Large 1942.2 520.5K 520.5K 14.4M 5.2M
Full – 4.3M 4.3M 117.4M 38.3M
MIRIAD Control 1.0 1.0K 1.0K 4.2K 4.2K
Tiny 6.6 6.6K 6.6K 28.1K 27.4K
Small 44.1 44.1K 44.1K 186.9K 182.2K
Medium 292.6 292.6K 292.6K 1.2M 1.2M
Large 1942.2 1.9M 1.9M 8.2M 8.0M
Table 2
Corpus scales and representation-dependent retrieval-index sizes.Source docsdenotes the number of documents in each retrieval
pool, whereasRetrieval unitsreports the number of indexed abstracts, full-text chunks, or TreeRAG units. Factors are defined
relative to the corpus-specific control setting and applied independently to both corpora. Values are abbreviated for readability; K
denotes thousands and M denotes millions. Thefullsetting is available only for PubMed.
retrieval uses BM25 [ 44] as a sparse baseline, BGE-base-en-
v1.5[45]asalightweightdenseretrieverwithapproximately
0.1B parameters, and Qwen3-Embedding-8B [ 46] as a larger
dense retriever with 8B parameters.
For reranking, we compare Jina-Reranker-v3 [ 47], a
0.6Blistwisererankerwithadecoder-onlybackbone;BGE-
Reranker-v2-M3 [ 48,49], a 0.6B cross-encoder; and Qwen3-
Reranker-8B [ 46], an 8B LLM-based reranker. This selec-
tioncoversdifferentrerankingarchitecturesandprovidesa
comparison between compact and higher-capacity models.
Generation experiments use compact 4B-parameter mod-
els, reflecting growing interest in smaller models whose
computational requirements are more compatible with de-
ployment in biomedical and healthcare systems. We evaluate
threemodelsinfourconfigurations.Gemma-3-4B[ 50]serves
asageneral-purposebaseline,whileMedGemma-1.5-4B[ 51]
provides a biomedical counterpart from the same model
family. Qwen3.5-4B [ 52] is evaluated with thinking mode
disabled and enabled; we refer to these configurations as
Qwen3.5-4BandQwen3.5-4B-Think,respectively.Because
both configurations use the same underlying model, their
comparisonisolatestheeffectofthinkingmodewhilekeeping
the architecture and parameter count fixed.
3.5. Evaluation Protocol
3.5.1. Retrieval and Reranking
Retrieval quality is measured using Macro Recall@ 𝑘,
whichcomputesrecallforeachqueryandthenaveragesacross
queries. This gives equal weight to each question regardless
ofthenumberofassociatedgolddocuments.Goldevidenceis
matched at the source-document level so that retrieval units
derived from the same document are treated consistently
across representations.
ForMedAESQA,goldsourcescorrespondtothePubMed
articles supporting each question, although the original
evidence is provided at the abstract level. For MIRIAD, gold
sourcescorrespondtotheoriginalfull-textpassageslinked
to each question.We report Recall@100 for first-stage retrieval. Each
rerankerisappliedtothetop100retrievedcandidates,and
reranking quality is evaluated using Recall@5.
3.5.2. End-to-End Evaluation
End-to-end performance is measured using MCQA accu-
racy. We compare three configurations:LLM-only, in which
the generator receives only the question and answer options;
RAG,inwhichitadditionallyreceivesthetopfiveevidence
unitsproducedbytheretrievalpipeline;andOracle,inwhich
retrieval is bypassed and gold evidence is supplied directly.
Theretrievalconfigurationusedforend-to-endevaluation
is selected using the development results and fixed before
evaluation on the test sets.
The Oracle configuration is evaluated for MedAESQA
and MIRIAD, for which question-linked gold evidence is
available. It uses the gold abstracts for MedAESQA and
thegoldfull-textpassagesforMIRIAD.Thisconfiguration
measures generationperformance whenreference evidence
is available and helps distinguish retrieval failures from
generationlimitations.Becausethesuppliedevidencefollows
the annotation format of each dataset, Oracle results are
interpreted as dataset-specific reference points rather than
absolute upper bounds.
3.5.3. Retrieval Efficiency
Retrievalefficiencyisevaluatedusingquery-timelatency
and index storage. Latency is measured per question for
the same first-stage retrieval and reranking configurations
used in the retrieval-quality evaluation. We report first-stage
retrieval latency separately from retrieval-plus-reranking
latency to quantify the additional cost of reranking the top
100 candidates.
Index storage is measured after indexing as the total
Elasticsearchindexfootprint.Thiscapturesthedeployment
cost introduced by representation-dependent growth in the
number of retrieval units. Because absolute latency depends
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 4 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
on hardware, software, and implementation details, we inter-
pretlatencycomparativelyacrossconfigurationsmeasured
under the same experimental conditions. The computational
environmentandsoftware versionsaredescribedinSupple-
mentary Section A.
4. Results
The experimental results are organized into sequential
phasescorrespondingtothemaincomponentsoftheRAG
pipeline.Throughoutthestudy,corpusscaleanddocument
representation are treated as factors of analysis: Abstract,
Full-text,andTreeRAGareevaluatedacrosstheexperimental
phasesratherthanreducedtoasingleselectedrepresentation.
Developmentresultsareusedtoselectthefirst-stageretriever
andthereranking strategyappliedinthesubsequentend-to-
end experiments.
4.1. Phase I: Retriever Evaluation
Phase I compares the retrievers on the MedAESQA
and MIRIAD development sets using Macro Recall@100.
Retriever selection is based on performance at the largest
available corpus scale for each dataset:fullPubMed for
MedAESQA andlargeMIRIAD for the MIRIAD DEV split.
Figure 1 reports retrieval performance across corpus
scales and document representations.
On MedAESQA, differences between retrievers become
clearer as the PubMed corpus grows. The Abstract represen-
tationperformsbestacrossmostconfigurations,consistent
with the abstract-level evidence annotations in MedAESQA.
At thefullPubMed scale, Qwen3-Embedding-8B achieves
thehighestMacroRecall@100,reaching65.66%.BGE-base-
en-v1.5remainscompetitiveacrossmostscalesbutdecreases
more noticeably in thefullsetting, while BM25 shows the
strongest degradation as additional distractor documents are
added.
OnMIRIADDEV,retrievalisclosetosaturationatthe
smallest scales, with all retrievers obtaining high Macro
Recall@100. Differences between models and representa-
tionsbecomemorevisibleastheretrievalpoolgrows.The
Abstract representation is less effective than on MedAESQA.
Unlike the originalMIRIADpassages usedbythe Full-text
and TreeRAG representations, these abstracts are syntheti-
callygeneratedpseudo-abstractsandmayomitinformation
relevant to retrieval. At thelargeMIRIAD scale, Qwen3-
Embedding-8B achieves the highest Macro Recall@100,
reaching 88.50%, while BM25 remains close at 87.00%.
Qwen3-Embedding-8B obtains the highest Macro Re-
call@100atthelargestavailablescaleinbothdevelopment
datasetsandisthereforeselectedasthefirst-stageretriever
for the subsequent phases.
4.2. Phase II: Reranker Evaluation
Phase II fixes Qwen3-Embedding-8B as the first-stage
retriever and compares No-Reranker, BGE-Reranker-v2-
M3,Jina-Reranker-v3,andQwen3-Reranker-8B.IntheNo-
Reranker condition, the top five evidence units are taken
directly from the first-stage ranking. Configurations areevaluated using Macro Recall@5, with reranker selection
basedonfullPubMedforMedAESQAandlargeMIRIAD
for MIRIAD DEV.
Figure 2 reports the results across corpus scales and
document representations.
On MedAESQA, reranking does not provide a consis-
tentimprovementoverthefirst-stageranking.No-Reranker
remains comparable to the reranked configurations across
scales and obtains the highest Macro Recall@5 at thefull
PubMedscale,reaching25.28%withtheAbstractrepresenta-
tion. Performance generally decreasesas the PubMedcorpus
grows,whileAbstractremainsthestrongestrepresentation
across all configurations.
OnMIRIADDEV,theeffectofrerankingbecomesclearer
at larger corpus scales. At the control and tiny scales, Macro
Recall@5 is close to saturation for most configurations.
As the retrieval pool grows, the reranked configurations
moreconsistentlyoutperformNo-Reranker,particularlywith
the Full-text and TreeRAG representations. At thelarge
MIRIAD scale, Qwen3-Reranker-8B achieves the highest
Macro Recall@5, reaching 80.00% with TreeRAG.
Rerankerselectionthereforediffersbycorpus.Forsub-
sequent PubMed-based experiments, we retain the origi-
nal Qwen3-Embedding-8B ranking without reranking. For
MIRIAD-based experiments, we apply Qwen3-Reranker-8B
to the first-stage results.
4.3. Phase III: End-to-End Evaluation
PhaseIIIevaluateswhethertheretrievalconfigurations
selected in the previous phases improve downstream MCQA
accuracy. All corpus scales and document representations
remain under evaluation. PubMed-based experiments use
Qwen3-Embedding-8Bwithoutreranking,whereasMIRIAD
experiments use Qwen3-Embedding-8B followed by Qwen3-
Reranker-8B.WecomparethegeneratorsdescribedinSec-
tion 3.4 under the LLM-only, RAG, and Oracle conditions.
These experiments use MedAESQA and MIRIAD DEV,
which were also used to select the retrieval components. The
results therefore characterize development-stage behavior
rather than generalization to unseen test sets. Complete
accuracy values are reported in Supplementary Section E.
Figure 3 reports RAG accuracy gains, in percentage
points,relativetothecorrespondingLLM-onlybaselinefor
each generator.
RAG produces non-negative accuracy gains across the
development configurations shown in Figure 3, although
their magnitude varies substantially across generators, cor-
pus scales, and document representations. Some MIRIAD
configurationsapproachthedataset-specificOraclegain,and
asmallnumberexceedit.ThisispossiblebecausetheOracle
condition uses the evidence supplied by the dataset and is
treatedasareferencepointratherthanasanabsoluteupper
bound.
On MedAESQA, the effect of RAG varies with both the
generator and the document representation. Gemma-3-4B
obtainsthelargestgainsoverall,includinganimprovement
of 27.50 percentage points in its strongest configuration.
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 5 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen30.20.40.60.81.0Macro Recall@100
Control Tiny Small Medium Large Full(a) MedAESQA
BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen3 BM25 BGE Qwen3
Corpus scale0.20.40.60.81.0Macro Recall@100
Control Tiny Small Medium Large(b) MIRIAD DEVGranularity
Abstract Full-text TreeRAG Tied granularities
Figure 1:Retriever evaluation on DEV using Macro Recall@100. Panel (a) shows MedAESQA across PubMed corpus scales, and
panel (b) shows MIRIAD DEV across MIRIAD corpus scales. Colors denote the document representation: Abstract, Full-text, or
TreeRAG. Diagonal split fills indicate ties between representations. Retriever labels correspond to BM25, BGE-base-en-v1.5 (BGE),
and Qwen3-Embedding-8B (Qwen3). The bold label and black star indicate the highest-scoring retriever at the largest available
scale for each dataset.
Qwen3.5-4Balsobenefitsconsistently,whereasMedGemma-
1.5-4B shows greater variation across scales and represen-
tations. Qwen3.5-4B-Think obtains smaller gains overall,
alongside its higher LLM-only accuracy of 85.0%.
The Abstract representation is generally the strongest
and most stable MedAESQA setting, consistent with the
dataset’s abstract-level evidence annotations. TreeRAG is
oftencompetitivewithorstrongerthanflatFull-textchunking.
AtthefullPubMedscale,mostconfigurationsstillimprove
overtheircorrespondingLLM-onlybaselines,althoughthe
gains are generally smaller than at lower corpus scales,
particularly for TreeRAG. This pattern is consistent with the
decline in retrieval quality observed as additional PubMed
documents are introduced.
OnMIRIADDEV,RAGyieldslargegainsforGemma-3-
4B and MedGemma-1.5-4B, whose strongest configurations
improve accuracy by 32.00 and 30.50 percentage points,
respectively. The two Qwen3.5 configurations also benefit
from retrieval, although their higher LLM-only accuracies
leave less room for improvement.Full-textandTreeRAGgenerallyoutperformtheAbstract
representationonMIRIADDEV.Unliketheoriginalpassages
used to construct Full-text and TreeRAG units, the Abstract
representation consists of synthetically generated pseudo-
abstracts and may omit retrieval-relevant content. TreeRAG
remains competitive with flat Full-text chunking across
corpusscales,withonlysmalldifferencesbetweenthemin
many configurations. Both representations retain substantial
gains at thelargeMIRIAD scale.
Overall,thedevelopmentresultsshowthatRAGmatches
or improves upon the corresponding LLM-only baseline
across theevaluatedconfigurations, although themagnitude
of the gains depends on the generator, corpus scale, and
documentrepresentation.Thenext phaseevaluateswhether
thesepatternstransfertotestdatasetsthatwerenotusedfor
pipeline selection.
4.4. Generalization to Unseen Benchmarks
Thissectionevaluateswhethertheretrieval-augmented
pipeline selected during development transfers beyond the
datasets used for pipeline selection. We consider two test
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 6 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen30.20.40.60.81.0Macro Recall@5
Control Tiny Small Medium Large Full(a) MedAESQA
None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen3 None BGE Jina Qwen3
Corpus scale0.20.40.60.81.0Macro Recall@5
Control Tiny Small Medium Large(b) MIRIAD DEVGranularity
Abstract Full-text TreeRAG Tied granularities
Figure 2:Reranker evaluation on DEV using Macro Recall@5. Panel (a) shows MedAESQA across PubMed corpus scales, and
panel (b) shows MIRIAD DEV across MIRIAD corpus scales. All configurations use Qwen3-Embedding-8B as the first-stage
retriever. Colors denote the document representation: Abstract, Full-text, or TreeRAG. Diagonal split fills indicate ties between
representations. Labels correspond to No-Reranker, BGE-Reranker-v2-M3 (BGE), Jina-Reranker-v3 (Jina), and Qwen3-Reranker-8B
(Qwen3). The bold label and black star indicate the highest-scoring reranking configuration at the largest available scale for each
dataset.
settings. First, five external biomedical MCQA benchmarks
(MMLU-Med,MedQA-US,MedMCQA,PubMedQA,and
BioASQ-Y/N) are evaluated using thefullPubMed corpus
as the retrieval source. Second, the held-out MIRIAD TEST
split is evaluated using thelargeMIRIAD corpus. Although
MIRIAD TEST was not used for pipeline selection, it shares
thesameconstructionprocessandcorpusfamilyasMIRIAD
DEV and is therefore interpreted as an in-domain held-out
evaluation rather than as a fully external benchmark.
On the five external benchmarks, Gemma-3 and
MedGemma-1.5 improve over their LLM-only baselines
across every dataset and evaluated representation. Their
largestaveragegainsareobtainedwithTreeRAG,reaching
+7.7 and +9.6 percentage points, respectively. Qwen3.5
also benefits from RAG, although its average gains are
smaller:Full-textperformsbestat+2.5points,followedby
TreeRAGat+2.0points.ForQwen3.5-Think,RAGproduces
negative average gains across all three representations,
although TreeRAG performs best, with an average change of
−0.5points,comparedwith −1.2forFull-textand −2.0forAbstract. TreeRAG improves performance on MedMCQA
and BioASQ-Y/N, but these gains do not fully offset the
decreases observed on MMLU-Med, MedQA-US, and
PubMedQA.
The representation trends differ from those observed
on MedAESQA DEV. Abstract retrieval was generally
strongest during development, where the reference evidence
is abstract-based, whereas Full-text and TreeRAG obtain the
strongestexternalresultsformostgeneratorsandbenchmarks.
TreeRAGachievesthebestaveragegainforGemma-3and
MedGemma-1.5andremainscompetitivewithFull-textfor
Qwen3.5, despite using fewer indexed retrieval units than
flatfull-textchunking.BioASQ-Y/Nisanotableexception
tothebroaderpattern:Abstractremainscompetitive,while
TreeRAGmatchesorexceedsitforsomegenerators.Overall,
these results indicate that representation performance
dependsonboththebenchmarkandthegeneratorratherthan
on a single representation being consistently optimal.
Theheld-outMIRIADTESTresultspreservethemain
patterns observed on MIRIAD DEV. RAG improves all four
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 7 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
(a) MedAESQAAbstract
LLM-only baselineFull-text
Oracle gainTreeRAG
0102030
LLM-only 65.0%Gemma-3-4B
LLM-only 60.0%MedGemma-1.5-4B
ControlTiny SmallMediumLargeFull0102030
LLM-only 75.0%Qwen3.5-4B
ControlTiny SmallMediumLargeFull
LLM-only 85.0%Qwen3.5-4B-Think
Corpus size
(b) MIRIAD DEV
0102030
LLM-only 67.5%Gemma-3-4B
LLM-only 65.0%MedGemma-1.5-4B
ControlTiny SmallMediumLargeFull0102030
LLM-only 85.0%Qwen3.5-4B
ControlTiny SmallMediumLargeFull
LLM-only 85.0%Qwen3.5-4B-Think
Corpus sizeAccuracy gain (pp) Accuracy gain (pp)
Figure 3:End-to-end MCQA accuracy gains over each generator’s LLM-only baseline for (a) MedAESQA and (b) MIRIAD DEV,
reported in percentage points across corpus scales and document representations. Horizontal reference lines indicate the LLM-only
baseline at zero and the dataset-specific Oracle gain. MedAESQA uses Qwen3-Embedding-8B without reranking, whereas MIRIAD
DEV uses Qwen3-Embedding-8B followed by Qwen3-Reranker-8B.
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 8 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
RAGΔ
Model Corpus Dataset LLM-onlyAbstract Full-text TreeRAG
Gemma-3 PubMed MMLU-Med 57.4 +7.2 +7.8+8.5
MedQA-US 46.1 +11.0+13.4+10.8
MedMCQA 44.9 +1.8 +5.0+7.4
PubMedQA 51.4 +0.6+5.4+3.4
BioASQ-Y/N 70.2+8.3+4.7+8.3
Average 54.0 +5.8 +7.3 +7.7
MIRIAD MIRIAD TEST 65.8 +21.7+30.3+29.8
MedGemma-1.5 PubMed MMLU-Med 53.3 +6.9+7.8+6.7
MedQA-US 48.3 +5.2+6.0+4.6
MedMCQA 40.5 +1.7 +4.4+6.8
PubMedQA 41.2 +9.8 +10.0+15.0
BioASQ-Y/N 57.6 +13.0 +9.6+15.1
Average 48.2 +7.3 +7.6 +9.6
MIRIAD MIRIAD TEST 64.0 +15.9 +24.8+27.4
Qwen3.5 PubMed MMLU-Med 79.7 +0.1+2.2+1.8
MedQA-US 70.5 +1.4+4.0+1.3
MedMCQA 59.4−0.9+3.0 +3.0
PubMedQA 56.0−0.4+1.6+2.0
BioASQ-Y/N 82.5+2.1+1.8 +2.0
Average 69.6 +0.5 +2.5 +2.0
MIRIAD MIRIAD TEST 83.0 +10.2 +14.6+15.2
Qwen3.5-Think PubMed MMLU-Med89.8−1.7 −1.6 −1.3
MedQA-US90.2−4.0 −2.9 −3.6
MedMCQA 69.7−1.1+0.1+0.6
PubMedQA56.0−3.0 −0.2 −0.4
BioASQ-Y/N 84.3 +0.0−1.3+2.1
Average 78.0−2.0−1.2−0.5
MIRIAD MIRIAD TEST 84.1 +10.5 +14.4+15.0
Table 3
End-to-end MCQA performance on external benchmarks and MIRIAD TEST. LLM-only reports absolute accuracy (%), while RAG
Δreports the percentage-point change relative to the corresponding baseline for each document representation. PubMed denotes
retrieval over the full PubMed corpus, whereas MIRIAD denotes retrieval over the large MIRIAD corpus. Shaded cells show the
average across the five external benchmarks. Bold marks the largest improvement in each row, or the LLM-only baseline when no
RAG configuration improves upon it; ties are retained. Negative deltas are shown in red.
generators, with gains ranging from +10.2 to +30.3 percent-
age points. Abstract produces the smallest gain for every
generator,whereasFull-textandTreeRAGareconsistently
stronger. TreeRAG obtains the largest gain for three of the
four generators and remains within 0.5 points of Full-text for
Gemma-3. This pattern is consistent with the construction
of the MIRIAD representations: the Abstract condition uses
syntheticallygeneratedpseudo-abstracts,whereasFull-text
and TreeRAG preserve the original passage content more
directly.
Overall, the selected RAG pipeline transfers successfully
totheexternalbenchmarksforGemma-3,MedGemma-1.5,
and Qwen3.5, althoughthe magnitude ofthe gains andthe
strongest representation vary across generators and datasets.
TreeRAGisparticularlynotable,achievingthestrongestaver-
ageexternalgainsforGemma-3andMedGemma-1.5andthe
largest MIRIAD TESTgain for three ofthe four generators,
whileremainingcompetitivewithFull-textelsewhere.The
development-stagebehavioralsotransfersclearlytothein-
domainMIRIADTESTsplit,whereallgeneratorsimproveand the full-text-derived representations remain substantially
stronger than the pseudo-abstract representation.
4.5. Retrieval Efficiency and Trade-offs
Beyondretrievalqualityanddownstreamaccuracy,large-
scale biomedical RAG systems are constrained by query-
timelatencyandindexstorage.Weanalyzethesetrade-offs
on MedAESQA using thefullPubMed corpus, the largest
biomedical retrieval setting evaluated in this study.
Latency is evaluated on MedAESQA using thefull
PubMedAbstractindex,thestrongestrepresentationinthe
MedAESQA development retrieval experiments.
Figure4reportsMacroRecall@5againstmeanper-query
latency for the first-stage and reranked configurations.
The first-stage retrievers exhibit a clear latency–
effectiveness trade-off. BM25 has the lowest mean latency at
10.8 ms per query but obtains substantially lower Recall@5
than the dense retrievers. BGE-base-en-v1.5 increases
latencyto53.9mswhileprovidingamarkedimprovementin
retrieval effectiveness.Qwen3-Embedding-8Bachievesthe
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 9 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
10 100 1000
Latency (ms/query)0.050.100.150.200.25Macro Recall@5
BM25BGE-base-en-v1.5Qwen3-Embedding-8BReranker
BGE-Reranker-v2-M3 Jina-Reranker-v3 Qwen3-Reranker-8B
Figure 4:Efficiency–effectiveness trade-off on MedAESQA using thefullPubMed Abstract index. The x-axis reports mean
per-query latency on a logarithmic scale, and the y-axis reports Macro Recall@5. Open circles denote first-stage retrievers, while
filled markers denote the corresponding configurations after reranking the top 100 candidates. Colors identify the first-stage
retriever, marker shapes identify the reranker, and arrows connect each retriever to its reranked variants.
highest first-stage Macro Recall@5, reaching 25.28%, but
requires 1234.7 ms per query. Thus, BGE-base-en-v1.5 is
approximately23timesfasterthanQwen3-Embedding-8B,
although it does not reach the same retrieval effectiveness.
Rerankingintroducesasubstantiallatencyoverheadfor
every first-stage retriever. For BM25, reranking improves
Recall@5 butmoves the resultingpipelines intothe latency
rangeofsubstantiallymoreeffectivedenseretrievalconfig-
urations.Reranking BGE-base-en-v1.5canfurtherimprove
its Recall@5, but only at a query-time cost that exceeds that
of Qwen3-Embedding-8B without reranking. For Qwen3-
Embedding-8B,noneofthererankersimprovesuponthefirst-
stage Recall@5, and some reduce it while increasing latency.
Consequently, the three first-stage retrievers define the ob-
servedlatency–effectivenessfrontier:Qwen3-Embedding-8B
provides maximum effectiveness, BGE-base-en-v1.5 offers a
substantiallylower-latencyalternative,andBM25minimizes
latency at the cost of lower recall.
Storagerequirementsalsovarysubstantiallyacrossrepre-
sentations and retrievers. At thefullPubMed scale, Abstract
contains 4.3M indexed retrieval units and requires 5.9, 18.8,
and 73.3 GB for BM25, BGE-base-en-v1.5, and Qwen3-
Embedding-8B, respectively. Full-text chunking expands the
collection to 117.4M units, increasing the corresponding
indexsizesto199.5,481.0,and1985.1GB.TreeRAGindexes
38.3M units and requires 111.9, 228.1, and 872.0 GB. Rela-
tive to flat Full-text chunking, TreeRAG uses approximately
67% fewer retrieval units and reduces storage by 44–56%,
depending on the retriever.Overall, maximum retrieval effectiveness is obtained
with Qwen3-Embedding-8B without reranking, whereas
BGE-base-en-v1.5 provides a substantially lower-latency
alternative. TreeRAG offers a complementary storage trade-
off: it requires considerably fewer indexed retrieval units
thanFull-textwhileremainingcompetitivewith,andoften
outperforming, flat Full-text chunking in the downstream
evaluations.
5. Discussion
Our results show that biomedical RAG design choices
should be evaluated in relation to corpus scale. Larger
retrieval pools broaden the available evidence but also
introduceadditionaldistractors,makingevidenceselection
more difficult. The degradation observed for several retrieval
configurations at the largest corpus scales shows that con-
clusions obtained from small or controlled collections do
not necessarily transfer to large-scale biomedical retrieval.
Corpus scale should therefore be treated as an explicit
evaluation variable and considered jointly with document
representation, retrieval quality, generator behavior, and
deployment cost.
Documentrepresentationstronglyshapesthisinteraction.
AbstractretrievalperformsparticularlywellonMedAESQA,
whose reference evidence is provided at the abstract level. In
contrast, Full-text and TreeRAG generally produce stronger
gains on the external benchmarks evaluated using thefull
PubMed corpus. MIRIAD shows an even clearer preference
forrepresentationsthatpreservetheoriginalpassagecontent:
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 10 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
itssyntheticallygeneratedpseudo-abstractsareconsistently
weaker than Full-text and TreeRAG. These contrasting
patterns indicate that no document representation is uni-
versally optimal. Its effectiveness depends on the evidence
requirementsofthebenchmarkandonhowtheretrievedunits
interact with the generator.
TreeRAGprovidesaparticularlyrelevanteffectiveness–
efficiency trade-off. It achieves the strongest average ex-
ternal gains for Gemma-3 and MedGemma-1.5, remains
competitive with Full-text for Qwen3.5, and obtains the
largestMIRIADTESTgainforthreeofthefourgenerators.
At the same time, TreeRAG indexes 38.3M retrieval units,
comparedwith117.4Mfor flatFull-text chunking,reducing
storage requirements by 44–56% depending on the retriever.
These findings show that competitive retrieval over full-text
contentdoesnotnecessarilyrequireindexingthecomplete
setoffixed-lengthchunks.However,therelativeadvantage
of TreeRAG still varies across generators, benchmarks, and
corpus scales, and the present experiments do not isolate
hierarchicalorganizationasthesolecauseofitsperformance.
For Qwen3.5-Think, TreeRAG produces the strongest
externalaverageamongthethreerepresentations,although
itsmeanchangeremainsslightlynegativeat −0.5percentage
points. Its gains on MedMCQA and BioASQ-Y/N are offset
by decreases on MMLU-Med, MedQA-US, and PubMedQA.
Thismixedbehaviorfurtherindicatesthattheeffectivenessof
TreeRAGdependsontheinteractionbetweenthegenerator
and the benchmark.
The results also challenge the assumption that reranking
is an inherently beneficial stage. On MedAESQA, Qwen3-
Embedding-8B obtains the strongest Recall@5 without
reranking, and the evaluated rerankers either preserve or
reduce effectivenesswhile increasing latency. OnMIRIAD,
rerankingbecomesmoreusefulastheretrievalpoolgrows,
andQwen3-Reranker-8Bimprovesthefinalevidenceranking
atthelargestscale.Theutilityofrerankingthereforedepends
onthequalityofthefirst-stageranking,thecorpus,andthe
evidence units being ranked. Its value should be established
empiricallyratherthanassumedfromimprovementsobserved
in smaller or different retrieval settings.
Downstream performance further depends on the gen-
erator. Gemma-3 and MedGemma-1.5 obtain the largest
RAG gains, whereas Qwen3.5 shows smaller but mostly
positiveimprovementsontheexternalbenchmarks.Qwen3.5-
Think benefits strongly from RAG on MIRIAD TEST but
shows smaller, mixed, or negative changes on the external
benchmarksevaluatedwithPubMedretrieval.AstrongLLM-
only baseline reduces the available margin for improvement,
but baseline performance alone does not explain whether
retrievalwillhelp:thesamegeneratorcanresponddifferently
acrossbenchmarksanddocumentrepresentations.Retrieval
and generation should therefore be evaluated jointly rather
than treating retrieval metrics as sufficient predictors of end-
to-end performance.
Finally,theefficiencyanalysisillustratesthatthemostef-
fectiveconfigurationisnotnecessarilythemostpracticalone.Qwen3-Embedding-8B provides the highest first-stage re-
trieval effectiveness but has substantially higher latency than
BGE-base-en-v1.5, while reranking introduces additional
cost without consistently improving Recall@5. Similarly,
Full-text supports strong downstream results but produces
thelargestindexes,whereasTreeRAGretainscompetitiveper-
formance with a considerably smaller storage footprint. The
preferredbiomedicalRAGconfigurationthereforedepends
on the intended deployment setting: maximum-effectiveness
systems may justify larger models and indexes, whereas
latency-orstorage-constrainedapplicationsmayfavorlighter
retrievers or more compact full-text representations.
5.1. Limitations
Thisstudy hassomelimitations relatedtodatasetscope
andevaluationdesign.MedAESQAcontainsrelativelyfew
questions,whereasMIRIADprovidesalargerbutsyntheti-
callyconstructedsetting.Moreover,MIRIADTESTshares
thesameconstructionprocessandcorpusfamilyasMIRIAD
DEV and should therefore be interpreted as an in-domain
held-outevaluation.Wecomplementthesecontrolleddatasets
with five independently constructed biomedical benchmarks
to assess transfer beyond the development settings.
Theevaluationisrestrictedtomultiple-choicequestion
answering,enablingstandardizedcomparisonsacrosscorpus
scales, retrieval configurations, and generators. It does not
directly assess open-ended generation, explanation quality,
evidence citation, or clinical validity. In addition, retrieval
is evaluated through source-document matching because the
gold evidence formats differ across datasets and do not align
exactly with all retrieval units. Recall should therefore be
interpreted as source-level evidence coverage rather than
exact evidence-unit matching.
Finally,thereportedlatencyandstoragevaluesdepend
onthehardware,software,andindexingconfigurationused
inthisstudy.Theysupportcontrolledrelativecomparisons
between configurations but should not be interpreted as
universal deployment costs.
6. Conclusion
Thisworkaddressesanimportantlimitationofexisting
biomedicalRAGevaluation:systemsarecommonlyassessed
ata singlefixed corpusscale,evenwhenthe retrievalindex
itself islarge.Such evaluations cannot establishwhether an
observedadvantageremainsrobustasthecollectiongrows.
Under controlled scaling across PubMed and MIRIAD, from
the smallest conditions to the full PubMed corpus and
thecompleteMIRIADcollection,advantagesobservedfor
retrievers, rerankers, and document representations at the
control or tiny scales did not consistently persist at larger
scales,despiteretainingtheavailablegoldsources.Corpus
scale should therefore be treated as an explicit evaluation
variable rather than assuming that results obtained at one
collection size will generalize to another.
No document representation was consistently optimal.
Abstract retrieval was effective when the reference evidence
wasabstract-aligned,whereasFull-textandTreeRAGwere
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 11 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
generally stronger on the external benchmarks and MIRIAD.
TreeRAGprovidedafavorableeffectiveness–efficiencytrade-
off, remaining competitive with, and frequently outperform-
ing,flatFull-textchunkingwhileusingapproximately67%
fewerindexedretrievalunitsand44–56%lessstorage.Rerank-
ing was likewise dataset-dependent, improving evidence
selection at the largest MIRIAD scale but providing no
effectiveness advantage for MedAESQA retrieval over the
full PubMed corpus.
Overall, this study provides a controlled, large-scale
empirical characterization of biomedical RAG under corpus
growth and operational constraints. By jointly evaluating
retrieval, reranking, generation, latency, and storage, the
results provide evidence for selecting RAG configurations
according to effectiveness and computational cost.
CRediT Authorship Contribution Statement
AdriánSánchez-Freire:Conceptualization,Methodol-
ogy, Software, Investigation, Formal analysis, Validation,
Visualization, Writing – original draft, Writing – review
and editing.Gonzalo Pérez-Gómez: Conceptualization,
Methodology, Software, Investigation, Writing – review and
editing.AitziberAtutxa:Conceptualization,Methodology,
Supervision, Writing – review and editing.Ander Barrena:
Conceptualization, Methodology, Supervision, Writing –
review and editing.
Ethics statement
This study used biomedical literature and previously
released benchmark datasets. It did not involve the recruit-
mentofhumanparticipants,accesstoidentifiablepersonal
data,oranimalexperimentation.Therefore,ethicscommittee
approval and informed consent were not required.
Data Availability
The source code, experimental scripts, configurations,
and resources supporting this study are publicly available
in two GitHub repositories. The main experimental
repository is available at https://github.com/hitz-zentroa/
biomedical-rag-scaling . The pipeline used to convert
open-ended question-answering datasets into multiple-
choice question answering format is available at https:
//github.com/hitz-zentroa/qa-to-mcqa.
Acknowledgements
ThisworkissupportedbyMCIN/AEI/10.13039/5011000
11033andFEDER/UEviaEDHER-MED–EDHIA(PID2022-
136522OB-C22), TRUST-MED (PID2025-174880OB-I00),
and DeepKnowledge (PID2021-127777OB-C21). Secondary
support is provided by the HiTZ Center and the Basque
Government (IT1570-22), and EFA 104/01-LINGUATEC
IA (ERDF/INTERREG POCTEFA 2021–2027). Adrián
Sánchez-Freire is supported by a predoctoral research con-
tract from the University of the Basque Country (EHU),reference PIF25/58. The majority of the inference experi-
ments were carried out on the Leonardo supercomputer at
CINECA through resources provided by the EuroHPC Joint
Undertaking, project EHPC-AIF2026SC01-156.
Thefunders hadnorolein thestudydesign;datacollec-
tion,analysis,orinterpretation;preparationofthemanuscript;
or the decision to submit the article for publication.
Declaration of Generative AI and AI-Assisted
Technologies in the Manuscript Preparation
Process
During the preparation of this work, the authors used
ChatGPT (OpenAI) for language editing. After using this
tool,theauthorsreviewedandeditedthecontentasneeded
andtakefullresponsibilityforthecontentofthepublished
article.
References
[1]P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal,
H.Küttler,M.Lewis,W.-t.Yih,T.Rocktäschel,S.Riedel,D.Kiela,
Retrieval-augmented generation for knowledge-intensive nlp tasks,
in:H.Larochelle,M.Ranzato,R.Hadsell,M.Balcan,H.Lin(Eds.),
Advances in Neural Information Processing Systems, Vol. 33, Curran
Associates, Inc., 2020, pp. 9459–9474.
URL https://proceedings.neurips.cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.pdf
[2]W.Tao,X.Xing,Y.Chen,L.Huang,X.Xu,TreeRAG:Unleashingthe
powerofhierarchicalstorageforenhancedknowledgeretrievalinlong
documents, in: W. Che, J. Nabende, E. Shutova, M. T. Pilehvar (Eds.),
FindingsoftheAssociationforComputationalLinguistics:ACL2025,
Association for Computational Linguistics, Vienna, Austria, 2025, pp.
356–371.doi:10.18653/v1/2025.findings-acl.20.
URLhttps://aclanthology.org/2025.findings-acl.20/
[3]Q. Zheng, S. Abdullah, S. Rawal, C. Zakka, S. Ostmeier, M. Purk,
E. Reis, E. J. Topol, J. Leskovec, M. Moor, Miriad: Augmenting
llms with millions of medical query-response pairs (2025). arXiv:
2506.06091.
URLhttps://arxiv.org/abs/2506.06091
[4]S. Borgeaud,A. Mensch,J. Hoffmann,T. Cai,E. Rutherford, K.Mil-
lican, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark,
D. De Las Casas, A. Guy, J. Menick, R. Ring, T. Hennigan, S. Huang,
L. Maggiore, C. Jones, A. Cassirer, A. Brock, M. Paganini, G. Irving,
O. Vinyals, S. Osindero, K. Simonyan, J. Rae, E. Elsen, L. Sifre,
Improvinglanguagemodelsbyretrievingfromtrillionsoftokens,in:
K.Chaudhuri,S.Jegelka,L.Song,C.Szepesvari,G.Niu,S.Sabato
(Eds.),Proceedingsof the39thInternational Conference onMachine
Learning, Vol. 162 of Proceedings of Machine Learning Research,
PMLR, 2022, pp. 2206–2240.
URLhttps://proceedings.mlr.press/v162/borgeaud22a.html
[5]O.Ram,Y.Levine,I.Dalmedigos,D.Muhlgay,A.Shashua,K.Leyton-
Brown, Y. Shoham, In-context retrieval-augmented language models,
Transactions of the Association for Computational Linguistics 11
(2023) 1316–1331.doi:10.1162/tacl_a_00605.
URLhttps://aclanthology.org/2023.tacl-1.75/
[6]Z. Jiang, F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang,
J. Callan, G. Neubig, Active retrieval augmented generation, in:
H. Bouamor, J. Pino, K. Bali (Eds.), Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing,
AssociationforComputationalLinguistics,Singapore,2023,pp.7969–
7992.doi:10.18653/v1/2023.emnlp-main.495.
URLhttps://aclanthology.org/2023.emnlp-main.495/
[7]G. Mialon, R. Dessi, M. Lomeli, C. Nalmpantis, R. Pasunuru,
R. Raileanu, B. Roziere, T. Schick, J. Dwivedi-Yu, A. Celikyilmaz,
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 12 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
E.Grave,Y.LeCun,T.Scialom,Augmentedlanguagemodels:asurvey,
Transactions on Machine Learning ResearchSurvey Certification
(2023).
URLhttps://openreview.net/forum?id=jh7wH2AzKK
[8]Y.Gao,Y.Xiong,X.Gao,K.Jia,J.Pan,Y.Bi,Y.Dai,J.Sun,M.Wang,
H. Wang, Retrieval-augmented generation for large language models:
A survey (2024).arXiv:2312.10997.
URLhttps://arxiv.org/abs/2312.10997
[9]P.Zhao,H.Zhang,Q.Yu,Z.Wang,Y.Geng,F.Fu,L.Yang,W.Zhang,
J. Jiang, B. Cui, Retrieval-augmented generation for ai-generated
content: A survey, DataScience and Engineering 11 (1) (2026) 1–29.
doi:10.1007/s41019-025-00335-5.
URLhttps://doi.org/10.1007/s41019-025-00335-5
[10]P. Jacquemart, P. Zweigenbaum, Towards a medical question-
answeringsystem:afeasibilitystudy,Stud.HealthTechnol.Inform.
95 (2003) 463–468.
[11]S. J. Athenikos, H. Han, Biomedical question answering: A survey,
Computer Methods and Programs in Biomedicine 99 (1) (2010) 1–24.
doi:https://doi.org/10.1016/j.cmpb.2009.10.003.
URL https://www.sciencedirect.com/science/article/pii/
S0169260709002879
[12]Q.Jin,Z.Yuan,G.Xiong,Q.Yu,H.Ying,C.Tan,M.Chen,S.Huang,
X.Liu,S.Yu,Biomedicalquestionanswering:Asurveyofapproaches
and challenges, ACM Computing Surveys 55 (2) (2022) 1–36. doi:
10.1145/3490238.
URLhttp://dx.doi.org/10.1145/3490238
[13]J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, J. Kang,
Biobert:apre-trainedbiomedicallanguagerepresentationmodelfor
biomedical text mining, Bioinformatics 36 (4) (2019) 1234–1240.
doi:10.1093/bioinformatics/btz682.
URLhttp://dx.doi.org/10.1093/bioinformatics/btz682
[14]Y. Gu, R. Tinn, H. Cheng, M. Lucas, N. Usuyama, X. Liu, T. Nau-
mann, J.Gao, H. Poon, Domain-specificlanguage model pretraining
for biomedical natural language processing, ACM Transactions on
Computing for Healthcare 3 (1) (2021) 1–23.doi:10.1145/3458754.
URLhttp://dx.doi.org/10.1145/3458754
[15]M. Yasunaga, J. Leskovec, P. Liang, LinkBERT: Pretraining language
modelswithdocumentlinks,in:S.Muresan,P.Nakov,A.Villavicencio
(Eds.), Proceedings of the 60th Annual Meeting of the Association
forComputationalLinguistics (Volume1: Long Papers),Association
for Computational Linguistics, Dublin, Ireland, 2022, pp. 8003–8016.
doi:10.18653/v1/2022.acl-long.551.
URLhttps://aclanthology.org/2022.acl-long.551/
[16]K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung,
N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl, P. Payne, M. Senevi-
ratne, P. Gamble, C. Kelly, A. Babiker, N. Schärli, A. Chowdhery,
P. Mansfield, D. Demner-Fushman, B. Agüera y Arcas, D. Webster,
G.S.Corrado,Y.Matias,K.Chou,J.Gottweis,N.Tomasev,Y.Liu,
A.Rajkomar,J.Barral,C.Semturs,A.Karthikesalingam,V.Natarajan,
Large language models encode clinical knowledge, Nature 620 (7972)
(2023) 172–180.doi:10.1038/s41586-023-06291-2.
URLhttps://doi.org/10.1038/s41586-023-06291-2
[17]K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, M. Amin,
L. Hou, K. Clark, S. R. Pfohl, H. Cole-Lewis, D. Neal, Q. M. Rashid,
M. Schaekermann, A. Wang, D. Dash, J. H. Chen, N. H. Shah,
S. Lachgar, P. A. Mansfield, S. Prakash, B. Green, E. Dominowska,
B. Agüera y Arcas, N. Tomašev, Y. Liu, R. Wong, C. Semturs, S. S.
Mahdavi, J. K. Barral, D. R. Webster, G. S. Corrado, Y. Matias,
S. Azizi, A. Karthikesalingam, V. Natarajan, Toward expert-level
medical question answering with large language models, Nature
Medicine 31 (3) (2025) 943–950.doi:10.1038/s41591-024-03423-7.
URLhttps://doi.org/10.1038/s41591-024-03423-7
[18]H.Nori,N.King,S.M.McKinney,D.Carignan, E.Horvitz,Capabili-
tiesofgpt-4onmedicalchallengeproblems(2023). arXiv:2303.13375 .
URLhttps://arxiv.org/abs/2303.13375
[19]G.Xiong,Q.Jin,Z.Lu,A.Zhang,Benchmarkingretrieval-augmented
generation for medicine, in: L.-W. Ku, A. Martins, V. Srikumar (Eds.),
FindingsoftheAssociationforComputationalLinguistics:ACL2024,Association for Computational Linguistics, Bangkok, Thailand, 2024,
pp. 6233–6251.doi:10.18653/v1/2024.findings-acl.372.
URLhttps://aclanthology.org/2024.findings-acl.372/
[20]V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
W.-t.Yih,Densepassageretrievalforopen-domainquestionanswering,
in: B. Webber, T. Cohn, Y. He, Y. Liu (Eds.), Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing
(EMNLP),AssociationforComputationalLinguistics,Online,2020,
pp. 6769–6781.doi:10.18653/v1/2020.emnlp-main.550.
URLhttps://aclanthology.org/2020.emnlp-main.550/
[21]G. Izacard, E. Grave, Leveraging passage retrieval with generative
models for open domain question answering, in: P. Merlo, J. Tiede-
mann,R.Tsarfaty(Eds.),Proceedingsofthe16thConferenceofthe
EuropeanChapteroftheAssociationforComputationalLinguistics:
Main Volume, Association for Computational Linguistics, Online,
2021, pp. 874–880.doi:10.18653/v1/2021.eacl-main.74.
URLhttps://aclanthology.org/2021.eacl-main.74/
[22]Y. Liu, K. Hashimoto, Y. Zhou, S. Yavuz, C. Xiong, P. Yu, Dense
hierarchicalretrievalforopen-domainquestionanswering,in:M.-F.
Moens, X. Huang, L. Specia, S. W.-t. Yih (Eds.), Findings of the
AssociationforComputationalLinguistics:EMNLP2021,Association
forComputationalLinguistics,PuntaCana,DominicanRepublic,2021,
pp. 188–200.doi:10.18653/v1/2021.findings-emnlp.19.
URLhttps://aclanthology.org/2021.findings-emnlp.19/
[23]S. R. Bhat, M. Rudat, J. Spiekermann, N. Flores-Herr, Rethinking
chunksizeforlong-documentretrieval:Amulti-datasetanalysis(2025).
arXiv:2505.21700.
URLhttps://arxiv.org/abs/2505.21700
[24]P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, M. Gardner, A
dataset of information-seeking questions and answers anchored in
research papers, in: K. Toutanova, A. Rumshisky, L. Zettlemoyer,
D.Hakkani-Tur,I.Beltagy,S.Bethard,R.Cotterell,T.Chakraborty,
Y.Zhou(Eds.),Proceedingsofthe2021ConferenceoftheNorthAmer-
icanChapteroftheAssociationforComputationalLinguistics:Human
LanguageTechnologies,AssociationforComputationalLinguistics,
Online, 2021, pp. 4599–4610. doi:10.18653/v1/2021.naacl-main.365 .
URLhttps://aclanthology.org/2021.naacl-main.365/
[25]P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, C. D. Man-
ning,RAPTOR:Recursiveabstractiveprocessingfortree-organized
retrieval, in: The Twelfth International Conference on Learning
Representations, 2024.
URLhttps://openreview.net/forum?id=GN921JHCRw
[26]Z.Jiang,X.Ma,W.Chen,Longrag:Enhancingretrieval-augmented
generation with long-context llms (2024).arXiv:2406.15319.
URLhttps://arxiv.org/abs/2406.15319
[27]G. Izacard, P. Lewis, M. Lomeli, L. Hosseini, F. Petroni, T. Schick,
J.Dwivedi-Yu,A.Joulin,S.Riedel,E.Grave,Atlas:Few-shotlearning
with retrieval augmented language models, Journal of Machine
Learning Research 24 (251) (2023) 1–43.
URLhttp://jmlr.org/papers/v24/23-0037.html
[28]J.Ning,Y.Kong,Y.Long,J.Callan,Lessllm,moredocuments:Search-
ingforimprovedrag,in:R.Campos,A.Jatowt,Y.Lan,M.Aliannejadi,
C. Bauer, S. MacAvaney, A. Anand, Z. Ren, S. Verberne, N. Bai,
M. Mansoury (Eds.), Advances in Information Retrieval, Springer
Nature Switzerland, Cham, 2026, pp. 598–613.
[29]A. Otegi, I. San Vicente, X. Saralegi, A. Peñas, B. Lozano, E. Agirre,
Information retrieval and question answering: A case study on
covid-19scientificliterature,Knowledge-BasedSystems240(2022)
108072.doi:https://doi.org/10.1016/j.knosys.2021.108072.
URL https://www.sciencedirect.com/science/article/pii/
S0950705121011485
[30]I. Alonso, M. Oronoz, R. Agerri, Medexpqa: Multilingual
benchmarking of large language models for medical question
answering, Artificial Intelligence in Medicine 155 (2024) 102938.
doi:https://doi.org/10.1016/j.artmed.2024.102938.
URL https://www.sciencedirect.com/science/article/pii/
S0933365724001805
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 13 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
[31]J. Vladika, F. Matthes, Improving health question answering with
reliable and time-aware evidence retrieval, in: Findings of the Associ-
ationforComputationalLinguistics:NAACL2024,Associationfor
Computational Linguistics, Mexico City, Mexico, 2024, pp. 4752–
4763.doi:10.18653/v1/2024.findings-naacl.295.
URLhttps://aclanthology.org/2024.findings-naacl.295/
[32]M. Li, H. Kilicoglu, H. Xu, R. Zhang, Biomedrag: A
retrieval augmented large language model for biomedicine,
Journal of Biomedical Informatics 162 (2025) 104769.
doi:https://doi.org/10.1016/j.jbi.2024.104769.
URL https://www.sciencedirect.com/science/article/pii/
S1532046424001874
[33]I. Lopez, A. Swaminathan, K. Vedula, S. Narayanan,
F. Nateghi Haredasht, S. P. Ma, A. S. Liang, S. Tate, M. Maddali, R. J.
Gallo, N. H. Shah, J. H. Chen, Clinical entity augmented retrieval for
clinical information extraction, npj Digital Medicine 8 (1) (2025) 45.
doi:10.1038/s41746-024-01377-1.
URLhttps://doi.org/10.1038/s41746-024-01377-1
[34]J. Sohn, Y. Park, C. Yoon, S. Park, H. Hwang, M. Sung, H. Kim,
J. Kang, Rationale-guided retrieval augmented generation for medical
question answering, in: L. Chiruzzo, A. Ritter, L. Wang (Eds.),
Proceedingsofthe2025ConferenceoftheNationsoftheAmericas
Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), Association for
Computational Linguistics, Albuquerque, New Mexico, 2025, pp.
12739–12753.doi:10.18653/v1/2025.naacl-long.635.
URLhttps://aclanthology.org/2025.naacl-long.635/
[35]L. Stuhlmann, M. A. Saxer, J. Fürst, Efficient and reproducible
biomedicalquestionansweringusingretrievalaugmentedgeneration,
in: 2025 IEEE Swiss Conference on Data Science (SDS), IEEE, 2025,
p. 154–157.doi:10.1109/sds66131.2025.00029.
URLhttp://dx.doi.org/10.1109/SDS66131.2025.00029
[36]A. Leto, C. Aguerrebere, I. Bhati, T. Willke, M. Tepper, V. A. Vo,
Toward optimal search and retrieval for rag (2024). arXiv:2411.07396 .
URLhttps://arxiv.org/abs/2411.07396
[37]M. Iratni, M. Boughanem, T. Dkaki, Dynamic context selection for
retrieval-augmented generation: Mitigating distractors and positional
bias (2025).arXiv:2512.14313.
URLhttps://arxiv.org/abs/2512.14313
[38]D. Gupta, D. Bartels, D. Demner-Fushman, A dataset of medical
questions paired with automatically generated answers and evidence-
supported references, Scientific Data 12 (1) (Jun. 2025). doi:10.1038/
s41597-025-05233-z.
URLhttp://dx.doi.org/10.1038/s41597-025-05233-z
[39]D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song,
J.Steinhardt,Measuringmassivemultitasklanguageunderstanding,
in: International Conference on Learning Representations, 2021.
URLhttps://openreview.net/forum?id=d7KBjmI3GmQ
[40]D. Jin, E. Pan, N. Oufattole, W.-H. Weng, H. Fang, P. Szolovits, What
disease does this patient have? a large-scale open domain question
answering dataset from medical exams, Applied Sciences 11 (14)
(2021).doi:10.3390/app11146421.
URLhttps://www.mdpi.com/2076-3417/11/14/6421
[41]A. Pal, L. K. Umapathi, M. Sankarasubbu, Medmcqa: A large-
scale multi-subject multi-choice dataset for medical domain question
answering,in:G.Flores,G.H.Chen,T.Pollard,J.C.Ho,T.Naumann
(Eds.), Proceedings of the Conference on Health, Inference, and
Learning, Vol. 174 of Proceedings of Machine Learning Research,
PMLR, 2022, pp. 248–260.
URLhttps://proceedings.mlr.press/v174/pal22a.html
[42]Q. Jin, B. Dhingra, Z. Liu, W. Cohen, X. Lu, Pubmedqa: A dataset for
biomedical research question answering, in: Proceedings of the 2019
ConferenceonEmpiricalMethodsinNaturalLanguageProcessingand
the9thInternationalJointConferenceonNaturalLanguageProcessing
(EMNLP-IJCNLP), 2019, pp. 2567–2577.
[43]G.Tsatsaronis,G.Balikas,P.Malakasiotis,I.Partalas,M.Zschunke,
M.Alvers,D.Weißenborn,A.Krithara,S.Petridis,D.Polychronopou-
los,Y.Almirantis,J.Pavlopoulos,N.Baskiotis,P.Gallinari,T.Artieres,A.-C. Ngonga Ngomo, N. Heino, E. Gaussier, L. Barrio-Alvers,
G.Paliouras,Anoverviewofthebioasqlarge-scalebiomedicalseman-
ticindexingandquestionansweringcompetition,BMCBioinformatics
16 (2015) 138.doi:10.1186/s12859-015-0564-6.
[44]S. Robertson, H. Zaragoza, The probabilistic relevance framework:
Bm25andbeyond,FoundationsandTrendsinInformationRetrieval3
(2009) 333–389.doi:10.1561/1500000019.
[45]S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, J.-Y. Nie, C-pack:
Packed resources for general chinese embeddings, in: Proceedings
of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval, SIGIR ’24, Association for
Computing Machinery, New York, NY, USA, 2024, p. 641–649.
doi:10.1145/3626772.3657878.
URLhttps://doi.org/10.1145/3626772.3657878
[46]Y.Zhang,M.Li,D.Long,X.Zhang,H.Lin,B.Yang,P.Xie,A.Yang,
D. Liu, J. Lin, F. Huang, J. Zhou, Qwen3 embedding: Advancing text
embedding and reranking through foundation models, arXiv preprint
arXiv:2506.05176 (2025).
[47]F. Wang, Y. Li, H. Xiao, jina-reranker-v3: Last but not late interaction
for document reranking (2025).arXiv:2509.25085.
URLhttps://arxiv.org/abs/2509.25085
[48]C.Li,Z.Liu,S.Xiao,Y.Shao,Makinglargelanguagemodelsabetter
foundation for dense retrieval (2023).arXiv:2312.15503.
[49]J.Chen,S.Xiao,P.Zhang,K.Luo,D.Lian,Z.Liu,M3-embedding:
Multi-linguality, multi-functionality, multi-granularity text embed-
dings through self-knowledge distillation, in: L.-W. Ku, A. Martins,
V. Srikumar (Eds.), Findings of the Association for Computational
Linguistics: ACL 2024, Association for Computational Linguistics,
Bangkok, Thailand, 2024, pp. 2318–2335. doi:10.18653/v1/2024.
findings-acl.137.
URLhttps://aclanthology.org/2024.findings-acl.137/
[50]G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej,
S.Perrin,T.Matejovicova,A.Ramé,M.Rivière,L.Rouillard,T.Mes-
nard, G. Cideron, J. bastien Grill, S. Ramos, E. Yvinec, M. Casbon,
E. Pot, I. Penchev, G. Liu, F. Visin, K. Kenealy, L. Beyer, X. Zhai,
A. Tsitsulin, R. Busa-Fekete, A. Feng, N. Sachdeva, B. Coleman,
Y. Gao, B. Mustafa, I. Barr, E. Parisotto, D. Tian, M. Eyal, C. Cherry,
J.-T.Peter,D.Sinopalnikov,S.Bhupatiraju,R.Agarwal,M.Kazemi,
D. Malkin, R. Kumar, D. Vilar, I. Brusilovsky, J. Luo, A. Steiner,
A. Friesen, A. Sharma, A. Sharma, A. M. Gilady, A. Goedeckemeyer,
A.Saade,A.Feng,A.Kolesnikov,A.Bendebury,A.Abdagic,A.Vadi,
A. György, A. S. Pinto, A. Das, A. Bapna, A. Miech, A. Yang,
A. Paterson, A. Shenoy, A. Chakrabarti, B. Piot, B. Wu, B. Shahriari,
B. Petrini, C. Chen, C. L. Lan, C. A. Choquette-Choo, C. Carey,
C.Brick,D.Deutsch,D.Eisenbud,D.Cattle,D.Cheng,D.Paparas,
D.S.Sreepathihalli,D.Reid,D.Tran,D.Zelle,E.Noland,E.Huizenga,
E. Kharitonov, F. Liu, G. Amirkhanyan, G. Cameron, H. Hashemi,
H. Klimczak-Plucińska, H.Singh, H. Mehta, H. T.Lehri, H.Hazimeh,
I. Ballantyne, I. Szpektor, I. Nardini, J. Pouget-Abadie, J. Chan,
J. Stanton, J. Wieting, J. Lai, J. Orbay, J. Fernandez, J. Newlan,
J. yeong Ji, J. Singh, K. Black, K. Yu, K. Hui, K. Vodrahalli, K. Greff,
L. Qiu, M. Valentine, M. Coelho, M. Ritter, M. Hoffman, M. Watson,
M. Chaturvedi, M. Moynihan, M. Ma, N. Babar, N. Noy, N. Byrd,
N.Roy,N.Momchev,N.Chauhan,N.Sachdeva,O.Bunyan,P.Botarda,
P. Caron, P. K. Rubenstein, P. Culliton, P. Schmid, P. G. Sessa,
P. Xu, P. Stanczyk, P. Tafti, R. Shivanna, R. Wu, R. Pan, R. Rokni,
R.Willoughby,R.Vallu,R.Mullins,S.Jerome,S.Smoot,S.Girgin,
S. Iqbal, S. Reddy, S. Sheth, S. Põder, S. Bhatnagar, S. R. Panyam,
S. Eiger, S. Zhang, T. Liu, T. Yacovone, T. Liechty, U. Kalra, U. Evci,
V.Misra,V.Roseberry,V.Feinberg,V.Kolesnikov,W.Han,W.Kwon,
X.Chen,Y.Chow,Y. Zhu,Z.Wei,Z.Egyed,V.Cotruta,M. Giang,
P. Kirk, A. Rao, K. Black, N. Babar, J. Lo, E. Moreira, L. G. Martins,
O.Sanseviero,L.Gonzalez,Z.Gleicher,T.Warkentin,V.Mirrokni,
E. Senter, E. Collins, J. Barral, Z. Ghahramani, R. Hadsell, Y. Matias,
D. Sculley, S. Petrov, N. Fiedel, N. Shazeer, O. Vinyals, J. Dean,
D. Hassabis, K. Kavukcuoglu, C. Farabet, E. Buchatskaya, J.-B.
Alayrac,R.Anil,Dmitry,Lepikhin,S.Borgeaud,O.Bachem,A.Joulin,
A. Andreev, C. Hardin, R. Dadashi, L. Hussenot, Gemma 3 technical
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 14 of 15

Large-Scale Biomedical RAG: Corpus Scaling, Document Representation, and Efficiency Trade-offs
report (2025).arXiv:2503.19786.
URLhttps://arxiv.org/abs/2503.19786
[51]A.Sellergren,C.Gao,F.Mahvar,T.Kohlberger,F.Jamil,M.Traverse,
A. Tono, B. Sadjad, L. Yang, C. Lau, et al., Medgemma 1.5 technical
report, arXiv preprint arXiv:2604.05081 (2026).
[52]Qwen Team, Qwen3.5-4B, Hugging Face model card, retrieved on
September 3, 2026 (2026).
URLhttps://huggingface.co/Qwen/Qwen3.5-4B
Sánchez-Freire et al.:Preprint submitted to ElsevierPage 15 of 15

Supplementary Material
Large-Scale Biomedical RAG: Corpus Scaling,
Document Representation, and Efficiency Trade-offs
Adrián Sánchez-Freire, Gonzalo Pérez-Gómez, Aitziber Atutxa and Ander Barrena
HiTZ Basque Center for Language Technology,
University of the Basque Country (EHU), Spain
Supplementary file description. This document provides supplementary implementation and
reproducibility details, complete development-set results, and the prompt templates used in the
experimental pipeline. Main sections use appendix letters (A, B, C, ...), and tables, figures, and
equations are numbered within their corresponding section (e.g., Table A.1 and Figure F.1).
Contents
A Implementation Details and Reproducibility 2
A.1 Computational Environment . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
A.2 Software, Libraries, and Versions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
B MCQA Conversion Implementation Details 2
B.1 Raw Distractor Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
B.2 Distractor Rewriting and Length Calibration . . . . . . . . . . . . . . . . . . . . . 3
B.3 Distractor Filtering, Ranking, and MCQA Construction . . . . . . . . . . . . . . . 3
B.4 Answer Position Balancing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
C MIRIAD Pseudo-Abstract Generation Details 3
D TreeRAG Construction Details 4
E Phase III End-to-End Details 4
F Prompt Templates 6
1

A Implementation Details and Reproducibility
A.1 Computational Environment
Experiments were executed on a shared SLURM-managed compute environment. The CPU model
used for index construction and retrieval experiments was an AMD EPYC 7513 32-Core Processor .
Indexing jobs were submitted with 6 SLURM-allocated CPU cores. GPU-accelerated stages were
configured to use up to four NVIDIA A100-SXM4-80GB GPUs in parallel, with the effective number
of GPUs per job varying between one and four depending on job placement, resource availability,
and workload characteristics. The operating system was Linux kernel 4.18.0 with glibc 2.28.
A.2 Software, Libraries, and Versions
Table A.1 reports the main software libraries and versions used in the experiments.
Library Version
numpy 2.2.6
pandas 3.0.0
torch 2.10.0
transformers 5.8.0
sentence-transformers 5.4.1
vllm 0.19.1
datasets 4.5.0
elasticsearch 8.19.3
pyserini 1.5.0
faiss-cpu 1.13.2
lancedb 0.29.2Library Version
langchain-core 1.2.19
langchain-community 0.4.1
langchain-text-splitters 1.1.1
pyyaml 6.0.3
orjson 3.11.7
tqdm 4.67.3
psutil 7.2.2
codecarbon 3.2.3
matplotlib 3.10.8
openpyxl 3.1.5
Table A.1: Main software libraries and versions used in the experiments.
Model inference was performed with either vLLMor the Hugging Face transformers stack
depending on the experimental stage. vLLMwas used for high-throughput generation and embedding
extraction during dataset conversion, pseudo-abstract generation, and index construction. The
reported retriever, reranker, and end-to-end MCQA evaluations were performed with Hugging
Face-based implementations.
Elasticsearch 8.19.3 was used as the backend for the sparse and dense retrieval indexes. Sparse
retrieval used BM25 over the indexed text field, while dense retrieval used indexed dense_vector
fields with cosine similarity and bbq_hnsw approximate nearest-neighbor search. Indexes were built
once before evaluation and reused across the retriever and reranker experiments.
B MCQA Conversion Implementation Details
This section specifies the implementation details of the open-ended to MCQA conversion procedure
described in the main manuscript. Prompt templates are provided in Supplementary Section F.
The conversion pipeline is applied independently to each MedAESQA and MIRIAD question–
answer pair. Each instance preserves the original reference answer as the gold option and produces
three incorrect distractors, resulting in a four-option MCQA item with a single answer key. The
pipeline consists of raw distractor generation, distractor rewriting, distractor filtering and ranking,
MCQA construction, and answer-position balancing.
B.1 Raw Distractor Generation
For each question, we first generate a pool of candidate distractors from the original question
and reference answer. An instruction-tuned language model is prompted to produce exactly ten
medically plausible but incorrect distractors. The prompt requires candidates to be non-overlapping
2

with the correct answer, diverse, non-redundant, and similar in length to the reference answer. The
output is constrained to a JSON array to facilitate automatic parsing.
Generation is performed using meta-llama/Llama-3.3-70B-Instruct with sampling enabled,
temperature 0.7, top- p0.95, and a maximum generation length of 1000 new tokens. The prompt
template is shown in Figure F.1.
B.2 Distractor Rewriting and Length Calibration
Because raw distractors may introduce superficial cues, especially through differences in length
or syntactic structure, each candidate distractor is passed through a rewriting stage. The model
receives the original question, the reference answer, and the draft distractor, and is instructed to
preserve the distractor concept while matching the reference answer in syntactic structure, word
length, technical density, and visual footprint.
The generated text is post-processed to remove surrounding quotes and common conversational
prefixes. Empty outputs are discarded by falling back to the original distractor. Candidate
distractors that exactly match the reference answer are skipped before rewriting.
Rewriting is performed using meta-llama/Llama-3.3-70B-Instruct with sampling enabled,
temperature 0.7, top- p0.95, and a maximum generation length of 400 new tokens. The prompt
template is shown in Figure F.2.
B.3 Distractor Filtering, Ranking, and MCQA Construction
After rewriting, candidate distractors are cleaned by removing empty or malformed outputs, exact
duplicates, and candidates that are nearly identical to the correct answer. Near-identity is detected
using a two-step heuristic: a candidate is removed only when its character-level similarity to the
correct answer is above 0.85 and the two strings differ by at most one word. This avoids discarding
candidates that share surface form with the correct answer but differ substantially in content.
If fewer than three valid rewritten distractors remain, the pipeline falls back to the original raw
distractors and repeats the cleaning procedure. The remaining candidates are then ranked by an
instruction-tuned model, which selects the three best distractors according to medical plausibility,
non-redundancy, and lack of semantic overlap with the correct answer.
Ranking is performed using meta-llama/Llama-3.3-70B-Instruct with deterministic decoding
and a maximum generation length of 200 new tokens. The model outputs the indices of the selected
distractors, which are combined with the reference answer to form the final four-option MCQA
instance. The prompt template is shown in Figure F.3.
B.4 Answer Position Balancing
As a final post-processing step, we balance the position of the correct answer across labels A,
B, C, and D. The converted dataset is first constructed without enforcing a uniform answer-key
distribution. We then define an approximately balanced pool of target answer labels, shuffle it, and
assign one target label to each question. For each instance, the correct answer is moved to the
assigned position, and the three distractors are randomly placed in the remaining slots.
This preserves the content of all answer options while reducing positional bias in the final MCQA
dataset.
C MIRIAD Pseudo-Abstract Generation Details
This section specifies the implementation details of the pseudo-abstract generation procedure
described in the main manuscript. The prompt template is provided in Supplementary Section F.
Because MIRIAD does not provide separate abstracts, we generate pseudo-abstracts from the
original MIRIAD passages to define an abstract-level representation analogous to the PubMed
3

abstract setting. The original passages are still used directly for the full-text chunking representation
and as source documents for TreeRAG construction.
Pseudo-abstracts are generated independently for each passage using Qwen/Qwen3.5-35B-A3B
with vLLM. Passages containing 300 tokens or fewer are not summarized and are retained unchanged,
since they already match the target granularity of the abstract-level representation.
Generation is performed with thinking mode disabled, temperature 0.2, top- p0.95, a maximum
generation length of 450 tokens, and repetition penalty 1.1. The prompt template is shown in
Figure F.4.
For the abstract-level representation, the resulting pseudo-abstract is stored in the content
field. As for the other document representations, the indexed contents field is constructed by
concatenating the title, when available, with the corresponding content field.
D TreeRAG Construction Details
This section describes the implementation details of the TreeRAG-style hierarchical representation
described in the main manuscript. The prompt template used for hierarchical segmentation is
provided in Supplementary Section F.
For each source document, the text is first split into paragraphs, and each paragraph is assigned
a stable integer identifier. The generative model is not asked to rewrite or regenerate the document
text. Instead, it outputs a structured hierarchy containing node identifiers, short synthetic titles,
and lists of paragraph identifiers associated with each node. This design keeps the final indexed
content grounded in the original document text and reduces the risk of introducing generated
content into the retrieval units.
The hierarchy is generated dynamically for each document. Therefore, the number of nodes and
the depth of the tree are not fixed in advance, but depend on the thematic structure of the source
document. Each node groups semantically related paragraphs and receives a short synthetic title
summarizing its local content. For documents that exceed the model context window, the input is
processed in smaller consecutive blocks before reconstructing the final hierarchy.
TreeRAG construction was performed using Qwen/Qwen3.5-35B-A3B with vLLM. Generation was
run with thinking mode disabled, temperature 0.1, top- p0.8, and a maximum generation length
of 8000 tokens. The model output was parsed automatically to extract node identifiers, synthetic
titles, and paragraph indices. Malformed nodes and invalid paragraph references were removed
during post-processing.
After parsing, paragraph identifiers are replaced with the corresponding original paragraph text.
Each retrieval unit is then built by concatenating the synthetic titles inherited from its ancestor
nodes with the original text associated with the current node. The resulting hierarchy-enriched
snippets are indexed using the same retrieval backend as the other document representations.
Because TreeRAG requires an additional generative preprocessing step, its construction cost is
substantially higher than flat full-text chunking. This makes TreeRAG more expensive to build at
scale, although the resulting retrieval units can still be indexed and queried using the same backend
as the other representations.
E Phase III End-to-End Details
This section provides the detailed end-to-end MCQA accuracy results for the DEV datasets used in
Phase III, together with the decoding configuration used by the generator models. It complements
the aggregate trends shown in the Phase III figures of the main manuscript by reporting absolute
accuracy values for each generator, corpus scale, document representation, and evaluation setting.
All generator models were evaluated with deterministic decoding, using greedy generation and a
repetition penalty of 1.1. The maximum number of generated tokens was set to 100 for standard
MCQA generators and to 16,384 for reasoning-oriented generators. Because the decoding process is
deterministic, re-running the same generator–retrieval configuration with the same prompts does not
4

introduce sampling-induced variability. We therefore report one accuracy value per configuration
rather than averages over multiple stochastic runs.
This setting is appropriate for MCQA evaluation because the goal is to measure the accuracy
of each fixed generator–retrieval configuration under a controlled decoding policy, rather than to
estimate variability induced by sampling. Results for MedAESQA and MIRIAD DEV are reported
in Tables E.1 and E.2, respectively.
Generator Scale LLM-onlyRAGOracle
Abstract Full-text TreeRAG
Gemma-3Control 65.0 87.5 80.0 90.0
92.5Tiny 65.0 87.5 80.0 92.5
Small 65.0 87.5 80.0 90.0
Medium 65.0 85.0 77.5 90.0
Large 65.0 87.5 80.0 80.0
Full 65.0 85.0 75.0 77.5
MedGemma-1.5Control 60.0 80.0 60.0 75.0
80.0Tiny 60.0 77.5 62.5 75.0
Small 60.0 77.5 60.0 77.5
Medium 60.0 67.5 60.0 75.0
Large 60.0 72.5 60.0 80.0
Full 60.0 62.5 67.5 62.5
Qwen3.5Control 75.0 95.0 85.0 90.0
97.5Tiny 75.0 95.0 85.0 90.0
Small 75.0 95.0 85.0 90.0
Medium 75.0 95.0 85.0 87.5
Large 75.0 92.5 85.0 87.5
Full 75.0 92.5 80.0 85.0
Qwen3.5-ThinkControl 85.0 97.5 90.0 97.5
100.0Tiny 85.0 97.5 90.0 92.5
Small 85.0 95.0 87.5 95.0
Medium 85.0 95.0 90.0 90.0
Large 85.0 92.5 87.5 92.5
Full 85.0 90.0 85.0 85.0
Table E.1: End-to-end MCQA accuracy (%) on MedAESQA for each generator across corpus scales
and RAG document representations. All generators correspond to their 4B versions. Bold values
indicate the best RAG result for each generator and scale.
5

Generator Scale LLM-onlyRAGOracle
Abstract Full-text TreeRAG
Gemma-3Control 67.5 90.5 99.0 99.5
98.5Tiny 67.5 90.5 99.0 99.5
Small 67.5 91.0 98.5 98.5
Medium 67.5 90.5 95.0 97.0
Large 67.5 87.0 97.0 96.0
MedGemma-1.5Control 65.0 79.5 93.0 94.5
93.5Tiny 65.0 79.5 93.0 95.5
Small 65.0 83.0 91.5 93.5
Medium 65.0 81.0 91.0 93.0
Large 65.0 77.5 90.5 90.5
Qwen3.5Control 85.0 97.5 100.0 100.0
100.0Tiny 85.0 96.0 100.0 100.0
Small 85.0 97.5 100.0 100.0
Medium 85.0 95.0 98.5 99.5
Large 85.0 92.0 98.5 97.0
Qwen3.5-ThinkControl 85.0 97.0 100.0 100.0
100.0Tiny 85.0 97.0 100.0 100.0
Small 85.0 95.5 99.5 99.5
Medium 85.0 95.5 99.0 99.0
Large 85.0 93.5 97.5 98.0
Table E.2: End-to-end MCQA accuracy (%) on MIRIAD DEV for each generator across corpus
scales and RAG document representations. All generators correspond to their 4B versions. Bold
values indicate the best RAG result for each generator and scale.
F Prompt Templates
This section reports the complete prompt templates used throughout the experimental pipeline.
Figures F.1, F.2, and F.3 correspond to the three prompts used for MCQA conversion: raw distractor
generation, distractor rewriting, and final distractor selection. Figures F.4, F.5, and F.6 show the
prompts used for MIRIAD pseudo-abstract generation, hierarchical segmentation during TreeRAG
construction, and end-to-end MCQA answering, respectively.
In the end-to-end prompt, the allowed output letters are A, B, C, D, and E. Although the
datasets evaluated in this work contain at most four options, and some contain only two or three,
the broader answer format keeps the same prompt template transferable to MCQA datasets with
different numbers of answer choices.
6

Prompt template for raw distractor generation
System prompt
You are an expert medical educator who generates plausible but incorrect multiple-choice
distractors.
Rules:
- Distractors must be medically plausible but clearly incorrect.
- They must not overlap with or imply the correct answer.
- They must be SIMILAR IN LENGTH to the correct answer.
- They must be diverse and non-redundant.
- You MUST output ONLY a valid JSON array of exactly 10 strings.
- Do NOT output explanations or commentary.
- Do NOT output anything before or after the JSON.
User prompt
Question:
{question}
Correct answer:
{answer}
TASK:
Generate exactly 10 high-quality incorrect distractors that follow all rules.
OUTPUT FORMAT (strict):
[
"distractor one",
"distractor two",
"..."
]
Return ONLY the JSON array.
Figure F.1: Prompt template used to generate candidate distractors from each open-ended question–
answer pair.
7

Prompt template for distractor expansion
System prompt
You are a medical psychometrician specialized in eliminating ’length cue’ bias in board-style
exams. Your goal is to make the Distractor indistinguishable from the Reference Answer by
mirroring its syntactic structure, technical density, and visual footprint. You prioritize
structural symmetry over creative expansion.
User prompt
### CONTEXT
The current distractor creates length bias. You must recalibrate it to match the Reference
Answer’s profile.
### INPUT DATA
**Question:** "{question}"
**Reference Answer (The Pattern):** "{answer}"
**Draft Distractor (The Concept):** "{distractor}"
### INSTRUCTIONS
1. SYNTACTIC MIRRORING: Identify the grammatical structure of the Reference Answer (e.g.,
[Mechanism] leading to [Effect]). Rewrite the Distractor using this exact same template.
2. LENGTH CALIBRATION: The output must be within +/- 3 words of the Reference Answer. Do not
exceed this range.
3. TECHNICAL DENSITY: Match the number of specific medical entities (enzymes, receptors,
anatomical sites) used in the Reference Answer. Use high-level terminology to reach the
target length rather than simple adjectives.
4. FACTUALITY: Ensure the final text remains medically incorrect for the given question.
5. OUTPUT RULE: Provide ONLY the final rewritten text. No quotes, no preamble, no
explanations.
### REWRITTEN DISTRACTOR:
Figure F.2: Prompt template used to rewrite candidate distractors and reduce length-cue bias.
8

Prompt template for distractor ranking
System prompt
You are an expert evaluator of medical MCQ distractors.
You will be provided with a Question, the Correct Answer, and a numbered list of Candidate
Distractors.
Your task is to identify the INDICES (numbers) of the 3 best distractors.
Criteria for best distractors:
1. Medically plausible but incorrect.
2. Not redundant with each other.
3. Not a rephrasing of the correct answer.
OUTPUT FORMAT:
Return ONLY a JSON object with the list of indices (0-based integers).
Example: {"best_indices": [0, 4, 7]}
User prompt
Question:
{question}
Correct Answer:
{answer}
Candidate Distractors:
{distractors}
TASK:
Return the JSON object with the indices of the 3 best distractors.
Figure F.3: Prompt template used to select the final three distractors from the candidate pool.
9

Prompt template for MIRIAD pseudo-abstract generation
System prompt
You are a biomedical summarization assistant.
User prompt
Task: Write a concise abstract-style summary of the following passage.
Requirements:
- 4-6 sentences.
- Target length: approximately 230-300 words.
- Use formal, academic medical language.
- Structure: briefly introduce the context/objective, describe key methods or findings, and
state the main conclusion.
- Do NOT include greetings, explanations, or meta-commentary. Output only the summary.
Passage:
{content}
Summary:
Figure F.4: Prompt template used to generate pseudo-abstracts for MIRIAD.
Prompt template for hierarchical medical article chunking
System prompt
You are a Medical Information Specialist. Your goal is to structure medical articles by
grouping paragraph indices into semantically coherent hierarchical chunks.
User prompt
Input text is pre-processed with paragraph indices (<n>) at the beginning of each line.
### 1. Chunking logic
- Merge only consecutive paragraphs with the same medical concept.
- Split when topic changes (e.g., Methods -> Results).
### 2. Output format
- ID: hierarchical numbering (e.g., 1, 1.1, 1.1.1)
- TITLE: short descriptive title (3-6 words)
- INDICES: list of paragraph indices
- NO TEXT: output only structure
### Example
ID: 1
TITLE: Clinical Trial Design
INDICES: [<1>, <2>, <3>]
Input:
{document}
Figure F.5: Prompt template used for hierarchical chunking of medical articles. The variable is
document .
10

Prompt template for end-to-end MCQA generation
System prompt
You are a multiple-choice question answering assistant.
If context is provided, use the information from it to choose the best answer.
If no context is provided, answer using your own knowledge.
Answer format rules:
- Respond with ONLY one letter: A, B, C, D, or E.
- Do NOT include explanations or extra text.
User prompt
Context:
{context}
---
Question: {question}
choices:
{choices}
Answer:
Figure F.6: Prompt template used for end-to-end multiple-choice answer generation. The variables
arecontext,question , and choices.
11