# KNIGHT: Knowledge Graph-Driven Multiple-Choice Question Generation with Adaptive Hardness Calibration

**Authors**: Mohammad Amanlou, Erfan Shafiee Moghaddam, Yasaman Amou Jafari, Mahdi Noori, Farhan Farsi, Behnam Bahrak

**Published**: 2026-02-23 18:46:27

**PDF URL**: [https://arxiv.org/pdf/2602.20135v1](https://arxiv.org/pdf/2602.20135v1)

## Abstract
With the rise of large language models (LLMs), they have become instrumental in applications such as Retrieval-Augmented Generation (RAG). Yet evaluating these systems remains bottlenecked by the time and cost of building specialized assessment datasets. We introduce KNIGHT, an LLM-based, knowledge-graph-driven framework for generating multiple-choice question (MCQ) datasets from external sources. KNIGHT constructs a topic-specific knowledge graph, a structured and parsimonious summary of entities and relations, that can be reused to generate instructor-controlled difficulty levels, including multi-hop questions, without repeatedly re-feeding the full source text. This knowledge graph acts as a compressed, reusable state, making question generation a cheap read over the graph. We instantiate KNIGHT on Wikipedia/Wikidata while keeping the framework domain- and ontology-agnostic. As a case study, KNIGHT produces six MCQ datasets in History, Biology, and Mathematics. We evaluate quality on five criteria: fluency, unambiguity (single correct answer), topic relevance, option uniqueness, and answerability given the provided sources (as a proxy for hallucination). Results show that KNIGHT enables token- and cost-efficient generation from a reusable graph representation, achieves high quality across these criteria, and yields model rankings aligned with MMLU-style benchmarks, while supporting topic-specific and difficulty-controlled evaluation.

## Full Text


<!-- PDF content starts -->

KNIGHT: Knowledge Graph-Driven Multiple-Choice
Question Generation with Adaptive Hardness
Calibration
Mohammad Amanlou1, Erfan Shafiee Moghaddam2, Yasaman Amou Jafari1,∗, Mahdi Noori1,∗,
Farhan Farsi3, Behnam Bahrak4
1University of Tehran2Independent Researcher3Amirkabir University of Technology4TEIAS
Institute
Mohammad.amanlou@ut.ac.ir erfanshm12@gmail.com yasaman.jafary.a@ut.ac.ir
mahdi.noori@ut.ac.ir Farhan1379@aut.ac.ir bahrak@teias.institute
∗Equal contribution.
With the rise of large language models (LLMs), they have become instrumental
in applications such as Retrieval-Augmented Generation (RAG). Yet evaluating
these systems remains bottlenecked by the time and cost of building specialized
assessment datasets. We introduce KNIGHT, an LLM-based, knowledge-graph-
driven framework for generating multiple-choice question (MCQ) datasets from
externalsources. KNIGHTconstructsatopic-specificknowledgegraph,astructured,
parsimonious summary of entities and relations, that can be reused to generate
instructor-controlled difficulty levels, including multi-hop questions, without re-
peatedlyre-feedingthefullsourcetext. ThisKGactsasacompressed,reusablestate,
makingquestiongenerationacheapreadoverthegraph. WeinstantiateKNIGHTon
Wikipedia/Wikidata, while keeping the framework domain- and ontology-agnostic.
As a case study, KNIGHT produces six MCQ datasets in History, Biology, and
Mathematics. We evaluate quality on five criteria: fluency, unambiguity (single
correct answer), topic relevance, option uniqueness, and answerability given the
providedsources(asaproxyforhallucination). ResultsshowthatKNIGHTenables
token-andcost-efficientgenerationfromareusableKGrepresentation,achieveshigh
qualityacrossthesecriteria,andyieldsmodelrankingsalignedwithMMLU-style
benchmarks, while supporting topic-specific anddifficulty-controlled evaluation.
1. Introduction
Recent work identifies two main levers for LLM progress: model size and data [ 1]. Because scaling
parameters increases financial and environmental costs (e.g., CO 2emissions) [ 2,3], attention is
shifting to dataset curation; yet expert-quality datasets are expensive and slow to build, and in
applied settings such as RAG and task-specific fine-tuning, public evaluation datasets remain scarce
due to proprietary data despite available toolkits [ 4,5]. Prior dataset-generation efforts exist [ 6–
8], but there is still no widely adopted open-source framework that is reproducible and easy to
implement. Moreover, standard MCQ benchmarks such as MMLU [ 9] are largely static, difficult to
update,providelimitedinstructor-levelcontroloverper-topicdifficulty,anddonotexposemulti-
hopstructureforcurriculumcustomization. Incontrast,KNIGHTenableslow-costgenerationof
topic-specific MCQ sets with user-controlled difficulty and explicit multi-hop design, while yielding
model rankings aligned with MMLU-style [10] benchmarks.
Weintroduce Knowledge-graph-driven NaturalItemGenerationwith AdaptiveHardnessTuning
(KNIGHT), a fully automated framework for synthesizing large-scale MCQ datasets from external
document collections and ontologies with controllable difficulty. Given a user topic τ(and optional
prompt),KNIGHTrunsfourstages: (i)constructatopic-specificknowledgegraph(KG)viaretrieval-
augmentedextraction[ 11–13],wheretheKGisacompact,parsimonioussummaryofentitiesand
relations distilled from the sources; (ii)generatesource-grounded MCQs by traversing multi-hop KG
Third Conference on Parsimony and Learning (CPAL 2026).arXiv:2602.20135v1  [cs.CL]  23 Feb 2026

pathswithconfigurabledepth;(iii)calibratedifficultybasedonpathlengthandabstraction,validated
via entropy-based uncertainty measures and human error patterns; and (iv)filteritems with an
LLM- and rule-based validator enforcing five criteria: grammar, single-correct-answer unambiguity,
option uniqueness, answerability from evidence, and topicality [14, 15].
KNIGHTintegratesRAG-basedextraction, KG-guided multi-hop generation,and LLM-based vali-
dationintoareusable,modularpipelinethatcachesacompacttopicKGforefficientdatasetcreation,
supportsforward/reversequestionmodes,anduseshumanandentropy-basedcheckstomitigate
imperfect answerability/difficulty proxies.
Figure1:KNIGHTHigh-levelpipeline.
Given a prompt/topic and depth,
KNIGHT retrieves evidence, builds
a focused KG, generates MCQs, and
filters them to produce the final dataset.We treat answerability from retrieved evidence as a proxy
for generator hallucination: items judged unanswerable
from the sources indicate unsupported or hallucinated
content. Since the KG is built once per topic and reused
acrossmanygenerations,KNIGHTenablestoken-efficient,
low-cost generation from a reusable KG representation
versus naive prompting that repeatedly re-ingests long
evidence contexts per question. We useGPT-4o-minifor
allLLMcallsthroughout,yieldingacost-andtoken-aware
evaluation setting.
We instantiate the type system ϕon the Wikipedia/Wikidata ontology (though in principle ϕcan be
defined over any domain ontology or enterprise schema) and introduceKNIGHT, a token-efficient,
KG-driven pipeline for generating difficulty-controlled MCQ datasets from external sources and
ontologies. As case studies of its flexibility and reusability (rather than standalone benchmark
contributions), we construct six Wikipedia/Wikidata-based MCQ datasets spanning history, biology,
and mathematics at two difficulty levels, and runfour ablationsalongside fullKNIGHTusingfive
GPT-4o-mini configurations(Plain, RAG, RAG+KG, RAG+Val, and fullKNIGHT). This staged
comparison isolates how grounding, KG guidance, and validation affect hallucination, distractor
quality, and difficulty calibration via automatic, human, and entropy-based evaluations. Items
are generated within minutes on Google Colab T4 and are grammatical and difficulty-calibrated
(Section4).KNIGHTreduceshallucinationsrelativetoPlainandRAG(SectionA)andyieldsmodel
rankings aligned with MMLU-style benchmarks, while being cheaper and easier to update than
static MMLU-like test sets, supporting it as a scalable, low-cost benchmark generator. Our code and
package are publicly available on PyPI1and GitHub2.
2. Related Work
KnowledgeGraphConstruction.ConstructingKGsfromunstructuredtexttypicallyusesmulti-
stage NLP pipelines (e.g. entity extraction/linking and relation extraction), often assuming a
predefinedschemaandsubstantialsupervision/trainingdata[ 16,17]. Traditionalsystemscommonly
perform named entity recognition and model-based relation extraction to identify entities and
relationships, but these often require predefined schemas and extensive training. Recent LLM-based
methods reduce these requirements and improve portability: Lairgi et al. [11]proposeiText2KG, a
zero-shotincrementalframeworkwithLLM-poweredentity/relationextractionfortopic-independent
KGconstruction;Dessìetal. [18]extracttriplesfromscientificabstractsviaNLP/text-miningand
integrate them into a KG; and Zhu et al. [19]evaluate GPT-4 [ 20] on KG tasks, finding that it excels
atreasoning,andintroduceAutoKG,amulti-agentLLMapproachwithexternalretrieval. Prompting
has also improved relation extraction, e.g., Wikidata-informed prompts in Layegh et al. [21]. While
weinstantiateourtypesystemusingWikipedia/Wikidataasanontology[ 22],themappingfunction
ϕcan in principle be defined over other domain ontologies or schemas (e.g., enterprise KGs or
specialized KBs); here we evaluate only the Wikipedia/Wikidata instantiation and leave broader
generalizationtofuturework. Finally,inlinewithretrieval-augmentedgeneration,RAGcombines
1https://pypi.org/project/knight-mcq/
2https://github.com/ErfanShm/knight-mcq
2

Figure 2:KNIGHT architecture.(Left) A topic/prompt-driven RAG pipeline retrieves evidence,
extractstriples,andcuratesacompactKGunderdepthbudget dmax;(Right)multi-hoppathsare
sampledtogeneratequestions/distractorsandvalidatedforevidence-groundedanswerabilityto
form the final MCQA dataset.
parametric LMs with retrieved knowledge bases; Lewis et al. [12]show it yields more specific,
diverse, and factual outputs than parametric-only models.
Question Generation from Knowledge Graphs and Structured Data.Early work generates natural
questionsfromKGtriplesusingkeywordextractionandRNNs[ 23]. Latermethodsgobeyondsingle
triples by encoding subgraphs with Graph2Seq and copy mechanisms [ 24], and by using contextual
KGswithanswer-awareGATsforcoherentmulti-hopquestiongeneration[ 25]. Difficultycontrol
has also been studied explicitly: Kumar et al. [26]condition multi-hop generation on estimated KG
difficulty, whileCheng etal. [27]guide reasoningcomplexityvia step-by-step rewriting. Beyond
graph-centricpipelines,LIQUIDbuildsQAdatasetsdirectlyfromtextthroughsummarization,entity
extraction, and question generation [28].
Evaluation and Filtering of Generated Questions.Ensuring the quality of generated questions
requires multi-faceted evaluation, and recent work applies dedicated QA evaluation metrics. High-
quality MCQs require multi-faceted evaluation: Moore et al. [29]survey metrics including LM
perplexity,lexicaldiversity,grammarerrorrates,cognitivecomplexity,andanswerabilitytoassess
fluency,uniqueness,andinferability,whileShypulaetal. [30]highlightsemanticdiversitygainsfrom
preference-tunedLLMs. Beyondthesequalitydimensions,factualityandsafetyremainconcerns: even
strongLLMscanhallucinateandmayexhibitbiases,motivatingautomaticfilteringandvalidation
when generating educational content [ 31]. Factuality is further challenged by the tendency of LLMs
to answer confidently even when inputs are unanswerable, motivating explicit answerability checks
as a practical proxy for hallucination control [ 32]. In RAG, recent work also evaluates whether
systemscorrectlyrejectunanswerablerequests,complementingaccuracyonanswerableones[ 33].
Finally,LLM-basedreview/validatorpipelinescanautomaticallyassessMCQvalidityacrossmultiple
criteria, reducing reliance on purely manual screening [34].
Building on these lines, we proposeKNIGHT, a unifiedend-to-end frameworkthat integrates KG con-
struction,graph-drivenquestiongeneration,andautomaticqualityfiltering. Auser-defineddifficulty
parameter controls graph depth to elicit multi-hop or higher-order items, while LLMs both generate
andvalidateMCQsfromKGpaths. Comparedtopriorpipelines,ourapproachemphasizesreusable,
token-efficient KG representationsanda comprehensive LLM-powered evaluation/validation stack,
aimingtoproducediverse,high-qualityQApairswithimprovedreliabilityfortopic-specificquestion
sets.
3

3. System Design
3.1. Knowledge-Graph Constructor
Given a user-specified topic τ, optional prompt, and
hardnessbudget dmax∈N,theKnowledge-GraphCon-
structor builds a directed property graph G= (V, E,R)
with canonicalized entities v∈Vand labeled edges
(vh, r, v t)∈E,r∈ R. Ititeratesaretrieve–generate–filter
loop (Alg. 1) combining external retrieval with LLM
reasoning; the backend is swappable (HuggingFace-
compatible) via a config flag.
Parsimonious, reusable representation.Once built, the
KGcanbecachedandreusedtogeneratemanydifficulty-
controlled question sets (varying hop length, formats,
and targets) without re-feeding long source documents,
amortizingtheone-timeconstructioncostandimproving
token efficiency.
Evidence retrieval anddescription synthesis.We first
retrieve a ranked context D={d 1, . . . , d k}from
Wikipedia (or other open sources) using dense passage
retrievalandre-ranking[ 12,13]. Conditionedon τ(andAlgorithm 1:GraphGenerator—depth-
bounded KG construction
Require: seedtopic/entity v0,depthlimit dmax
1:G←empty graph;G.addNode(v 0)
2:Q←[(v 0,0)]▷FIFO over (node,depth)
3:whileQ̸=∅do
4:(v, ℓ)←PopFront(Q)
5:D ←Retrieve(v)
6:δ← L desc(v,D)
7:R← L rel(δ)
8:C←Curate(R)
9:for all(v, r, u)∈Cdo
10:G.addNode(u);G.addEdge(v, r, u)
11:ifℓ+ 1≤d maxthen
12:PushBack(Q,(u, ℓ+ 1))
13:end if
14:end for
15:end while
16:returnG
theoptionalprompt)and D,theDescriptionGenerator Ldescproducesastructuredeight-pointgloss
δ(Appendix D).
Triple induction and deduplication.The Relation Extractor maps δto a triple set R(δ)(Eq. 3.1)
and removes near-duplicates using a Levenshtein filter with threshold λmax[11,18], then passes the
remaining candidates to the Curator.
Curation, pruning, and depth control.The Curator applies (i) type checks (instantiated here with
Wikidata), (ii) NLI-based consistency checks between node glosses and relation statements [ 35–37],
and(iii)content-policyscreeningfollowingpriorsafetyanalyses[ 15]. Graphexpansionproceeds
breadth-firstandstopsatdepth dmax,yieldingtheboundedneighborhood Vdmax={v|dist G(v0, v)≤
dmax}(Eq. 3.1), which matches the KG scope to the downstream MCQ generator (Section 3.2).
KG-1: Retrieval-AugmentedDescriptionSynthesis.Figure2showsthefirststage: producingan
eight-point gloss δ(v0)for the seed v0via arank-and-generateRAG pipeline [ 12,13] with (i) a dense
retriever Renc(Contriever base; 38) encoding topic τand scoring against a BM25-filtered corpus
[39], and (ii) a cross-encoder re-ranker Rrer(MiniLM-L12; 40) refining the top–50 to k=5passages
D0={d 1, . . . , d 5}with scores s(di)∈[0,1]. Each retained passage diis injected into system-prompted
GPT-4o-mini, yielding a candidate description d⋆
i=L desc(τ, d i); to combine evidence we model the
generation probability as a RAG mixture [12]:
P(d|τ) =X
z∈D 0Pθ(d|τ, z)exps(z)P
z′exps(z′)|{z }
Pret(z|τ),(1)
where Pθis parameterised by GPT-4o-mini; if D0=∅(no scores >0.15), we fall back to parametric
generation P(d|τ)=P θ(d|τ). Weretainanodeglossonlyifitistraceabletoatleastoneretrieved
passage, using:
γ(δ) =(
1∃z∈ D 0:overlap(z, δ)≥η,
0otherwise, η= 0.35,(2)
discarding γ(δ) = 0 descriptions; this makes persistent node content externally verifiable, mitigating
hallucinationrisk[ 41]. Thevalidateddescription δ(v0)isthenforwardedtotheRelationExtractor
(§3.1), enabling breadth-first expansion up to depthd max(Algorithm 1).
4

KG-2: Triple Induction via Relation Extraction.Given a stored gloss δindescription, we distil
explicit facts using the extractor Lrel, implemented withGPT-4o-mini3, which is prompted to emit a
JSON list of (head,relation,tail) triples. Formally, R(δ) ={(h, r, t)|h, t∈ E, r∈ R} , where Eis the
dynamic entity inventory and Ra controlled relation schema (cf. 11,18). A Levenshtein filter with
thresholdλ maxremoves near-duplicate triples before insertion.
KG-3: Depth-Controlled Expansion (token-efficient).Graph growth is bounded by the hardness
budget dmax: werunbreadth-firstexpansionwithaFIFOqueueover (v, ℓ)(Algorithm1),re-applying
KG-1–KG-2 while ℓ < d max, and halt with the visited set Vdmax={v|dist G(v0, v)≤d max}, ensuring
no node exceeds the user-defined cognitive radius.
KG-4: Curationandpruning.Eachcandidatetriple (h, r, t)isfilteredby ϕusing(i)ontology-based
typeagreement(here: Wikidata)[ 42],(ii)NLIentailmentconsistencybetween δ(h),δ(t)andthe
relationphrase[ 35], and(iii)content-policy compliance [ 15]; we retainthe edge iff ϕ(h, r, t) =True .
ϕcan be instantiated with other domain ontologies/schemas beyond Wikipedia/Wikidata.
3.2. MCQ Generator
GivenavalidatedKG G,wegeneratedifficulty-calibratedmulti-hopMCQsintwostages:MCQ-1
(path-conditionedsynthesis)andMCQ-2(validation/filtering). BothusesameGPT-4 o-mini,while
the decoder is swappable via configuration.
MCQ-1: Multi-HopMCQSynthesis.Foreachseed v0∈V,weenumeratelength- dforward/reverse
paths(eachhopin E),e.g., P:v 0r1− →v 1···rd− →v d(orthereverseorientation). Weverbalize Pintoa
compact context template T(P)by concatenating node glosses {δ(v i)}d
i=0and relation labels {ri}d
i=1,
then prompt Lqto output an MCQ tuple MP= (q P, aP, DP)with a single-sentence stem qP, key aP,
and three semantically proximate distractors DP[43,44]; distractor quality is evaluated via entropy
signals and human audits (§E.3).
MCQ-2: MCQ Validation & Filtering.Each candidate MPis scored by a validator Lvalon five
criteriaadaptedfromitem-writingbestpractices[ 45,46]: (i)grammaticalfluency,(ii)single-key
correctness, (iii) option uniqueness, (iv) answer derivability from the provided evidence (i.e., T(P)
and retrieved sources), and (v) topic relevance (when fixed). We retain an item iff all criteria pass,
keep(M P) =V5
k=1criterion k(MP) =True
,discardingtherest. ThisLLM-as-criticloopimproves
factualfidelityandpedagogicalvalidityofsyntheticquestions[ 45–47]. Retaineditemsareserialized
as JSONL with provenance metadata⟨v 0, d, P,orientation⟩.
4. Experiments
4.1. Datasets
We use six domain-specific multiple-choice (MCQ) datasets as case studies to evaluate KNIGHT
acrossthreesubjectareas(Biology,Mathematics,History)andtwodifficultylevels(Level1,Level3):
Bio-1,Bio-3,Math-1,Math-3,Hist-1, andHist-3. The History datasets contain 241 MCQs at Level 1
and 697 at Level 3; the Biology datasets contain 323 MCQs at Level 1 and 970 at Level 3; and the
Mathematics datasets contain 298 MCQs at Level 1 and 1063 at Level 3.
4.2. Experimental Setup and Baselines
Allsystemsusethesamebasegenerator,GPT-4o-mini. Forfair,evidence-groundedcomparison(and
to reduce hallucination), allRAG-basedvariants share the same Wikipedia retrieval step and use
the retrieved passages as evidence context [ 12,41,48]. We isolate component effects by toggling
retrievalgrounding(RAG),topicstructure(KG),andpost-hocfiltering(validator),yieldingfive
configurations:Plain(no evidence),RAG(evidence only),RAG+KG(evidence + topic KG; no
validator),RAG+Val(evidence + validator; no KG), andKNIGHT(evidence + KG-guided multi-
hop structuring + validator + difficulty control; Sec. 3). For each topic–difficulty split, we generate
3Checkpointgpt_4o_mini_2024_05.
5

Grammar Accuracy↑Fluency-automatic↑Fluency-human↑
Topic Plain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHT
History 0.9994 0.9993 0.9992 0.9990 0.9989 0.9498 0.9582 0.9536 0.9549 0.9581 4.8/5 4.7/5 4.7/5 4.7/5 4.8/5
Biology 0.9992 0.9994 0.9995 0.9989 0.9998 0.9702 0.9653 0.9681 0.9591 0.9626 4.9/5 4.9/5 4.8/5 4.8/5 4.9/5
Math 0.9978 0.9986 0.9981 0.9983 0.9991 0.9711 0.9671 0.9658 0.9602 0.9685 4.9/5 4.8/5 4.7/5 4.8/5 4.7/5
Table 1: Linguistic quality aggregated over Levels. Systems: Plain GPT-4o-mini, GPT-4o-mini+RAG,
GPT-4o-mini+RAG+KG, GPT-4o-mini+RAG+Validator, andKNIGHT.
N=100MCQs per system with fixed decoding and aligned Level 1/Level 3 settings; KG-based
variants additionally use the same constructed KG for comparability.
4.3. What makes a “good” MCQ dataset?
Weevaluatefiveitem-qualitycriteria:linguisticquality(well-formed,fluenttext),unambiguity
(exactlyonecorrectkey),optionuniqueness(non-overlappingdistractors),answerabilityfrom
source(the key is derivable solely from the provided evidence), andtopic relevance(semantic
alignmentwiththedeclaredtopic). Inthefollowing,weevaluatethesecriteriaacrossallsystems
(Sec.4.2), andadditionallyreportefficiency(generationspeed)anddifficultycalibration(Level1vs.
Level 3).
4.3.1. Linguistic Quality of Questions
Weassesslinguisticqualitytoavoidsurface-formconfoundsalongthreeaxes:grammaticalcorrect-
ness,fluency, andquestion-length diversity.Grammaris computed with LanguageTool [ 49,50] as
Grammar Quality(q) = 1−E
W,where Wisthenumberofwordsand Ethedetectederrors.Fluency
ismeasuredbothautomaticallywithLangCheck[ 51],whosescorescorrelatewithhumanjudgments
[14], and by CEFR C1/C2 annotators who rate n=100randomly ordered items per dataset on a
5-point Likert scale (Appendix E.3).Length diversityis analyzed via question-length distributions
[46,47] (Appendix E.1). Table 1 reports grammar accuracy and fluency (automatic/human), aggre-
gated over Levels, across topics and systems; overall linguistic quality is uniformly high, suggesting
later differences primarily reflect grounding, structure, and validation rather than surface-form
artifacts.
4.3.2. Unambiguity, Answerability, and Option Uniqueness
MCQvalidityhingesonthreeproperties:unambiguity(exactlyonecorrectkey),evidence-grounded
answerability(the key is derivable from the provided evidence), andnon-overlapping distractors
(options are not near-duplicates). Violations inflate chance performance, undermine construct
validity, and reduce score interpretability [44, 52, 53].
Human evaluation (Appendix E.3).For each split (topic ×difficulty), blinded domain experts
audited n=100items per system and flagged four error types:REPEATED,SINGLE_KEY,OP-
TION_UNIQUENESS, andANSWERABLE(key not justifiable from supplied evidence). We report
countsper100itemsinTable2(lowerisbetter);evidenceforjudgmentsmatchessysteminputs(none
forPlain; retrieved passages forRAG/RAG+Val; passages+KG context forRAG+KG/KNIGHT).
Answerabilityasahallucinationproxy.WetreatANSWERABLEviolationsashallucinationproxies:
ifthekeyisnotderivablefromthesuppliedevidence,theitemiseffectivelyungrounded. Accord-
ingly,PlainshowssubstantiallyhigherANSWERABLEcounts,underscoringtheroleofretrieval
grounding.
Results and component-wise patterns.Table 2 shows thatKNIGHTyields the cleanest item banks
overall (low repetition, fewer ambiguity errors, stronger distractor separability, and fewer unan-
swerable items) across both Level 1 and Level 3; importantly, Level 3 does not substantially increase
violations, suggesting difficulty control without sacrificing validity. fullKNIGHT(KG structuring +
validation + difficulty control) achieves the strongest validity profile under both difficulty settings.
4.3.3. Topic Relevance
We assess topical alignment using two complementary signals: (i) zero-shot MNLI-style entailment
[37], treating the topic as premise and the question as hypothesis; and (ii) a large LLM in few-shot
6

REPEATED↓SINGLE_KEY↓OPTION_UNIQUENESS↓ANSWERABLE↓
Split Plain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHT
History (L1) 20 19 15 2 0 10 4 6 3 2 8 5 5 5 3 26 10 8 10 6
Biology (L1) 19 17 16 3 1 16 5 6 3 1 4 3 5 5 2 19 8 7 9 4
Math (L1) 13 14 14 1 0 11 4 6 4 2 5 4 4 4 2 20 8 6 8 5
History (L3) 14 11 9 1 1 15 6 7 7 2 7 5 5 4 3 24 13 9 12 6
Biology (L3) 15 13 12 1 1 14 5 6 5 2 6 4 4 3 1 21 10 7 8 4
Math (L3) 10 10 9 0 0 13 6 5 4 3 7 5 4 2 2 28 12 9 7 6
Table2: Humanauditflagsper100items(lowerisbetter). AllsystemsuseGPT-4o-mini: Plain,RAG,
RAG+KG, RAG+Val, andKNIGHT.
Topic Relevance Score (Entailment)↑Topic Relevance (LLM)↑Human TOPIC Flags↓Off-topic Rate (LLM∩Entailment)↓
Split Plain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHT
History (L1) 0.9432 0.9938 0.8080 0.9945 0.8214 0.8318 0.9116 0.7301 0.9218 0.7692 7 5 17 4 8 6% 4% 19.3% 3% 10.6%
Biology (L1) 0.9156 0.9885 0.7962 0.9888 0.9053 0.8611 0.9542 0.7260 0.9619 0.7979 4 3 15 3 6 3% 1% 18.0% 1% 5.5%
Math (L1) 0.9219 0.9983 0.8103 0.9989 0.8852 0.8322 0.9092 0.8211 0.9514 0.8418 8 5 14 5 7 7% 4% 14.7% 4% 7.3%
History (L3) 0.9086 0.9975 0.5765 0.9981 0.8974 0.8115 0.8884 0.5291 0.9384 0.8124 9 6 42 5 9 8% 6% 34.4% 5% 8.1%
Biology (L3) 0.9309 0.9975 0.5971 0.9977 0.8832 0.8575 0.9205 0.5430 0.9381 0.8966 8 5 39 4 3 5% 3% 22.4% 3% 3.4%
Math (L3) 0.8996 0.9981 0.6203 0.9983 0.9849 0.8866 0.9307 0.5550 0.9438 0.8909 7 4 28 3 3 3% 2% 19.0% 2% 2.2%
Table3: Topicrelevance(entailmentandLLM; ↑),expertTOPICflags( ↓),andoff-topicrateasthe
intersection of automated checks (LLM∩Entailment;↓) across systems (all withGPT-4o-mini).
mode following standard NLG practice [ 36,37] with prompting exemplars [ 54]. For topic Tand
questionq,
S(q, T) =P(entailment|premise=T,hypothesis=q).(3)
We additionally report expertTOPICflags and anoff-topic ratecomputed from the union of the two
automated checks (Table 3).
Table3showsthatKNIGHTmaintainsstrongtopicalalignmentacrosstopicsanddifficultylevels:
entailmentandLLM-basedrelevanceremainhigh,off-topicratesarelow,andexpertTOPICflags
are rare. Overall, these results indicate that the generated MCQs stay on-topic, enabling subsequent
analyses to focus on validity, distractor quality, and difficulty calibration rather than topic drift.
4.4. Generation Speed
We measure end-to-end wall-clock time per topic–difficulty split on commodity hardware (Google
Colab: NVIDIA Tesla T4, 12 CPU cores; Appendix H). Level 1 completes in a few minutes (History:
212s, Math: 310s, Bio: 551s), while Level 3 remains practical (History: 852s, Math: 1226s, Bio: 2449s
≈41min). Theseruntimesreflectdatasetconstruction(notasingleexam)andenablefast,refreshable
topic-specificMCQbankscomparedtolongerexpertcurationcyclesforbroadstaticbenchmarks
(e.g., MMLU-style suites).
KNIGHTistoken-andcost-awarebydesign: thetopicKGisbuiltoncepertopicandcached,then
reused to generate many variants (levels, hop lengths, and forward/reverse formats) without re-
peatedly re-feeding full source documents. Consequently, theend-to-endtoken usage averages ∼600
total tokensper question(prompt+completion acrossgeneration andvalidation stages) inour setup,
whereasnaivepromptingandstandardRAG-onlybaselinesrepeatedlyinjectlongerevidencepas-
sagesperitem,inflatingcontextlengthandlatency. Cachingamortizestheone-timeKGconstruction
cost and keeps the marginal cost of producing additional datasets low.
5. Discussion
5.1. Evaluation of Distractor Quality via Predictive Entropy
High-quality four-option MCQs require distractors thatcompetewith the key without introduc-
ing ambiguity: weak distractors make items trivial, while misleading distractors can increase
SINGLE_KEYandANSWERABLEviolations in human audits (Sec. 4.3.2). Following Kim et al.
[55],wequantifydistractor“pull”viapredictiveentropyoveranswerchoicesusingasmallfixed
probe model ( LLaMA 3.2-3B-Instruct ). Given probe logits z= (z A, zB, zC, zD), we compute
pi= exp(z i)/P4
j=1exp(z j)fori∈ {A, B, C, D} [56] and entropy H=−P
ipilogp i. Higher H
indicatesdistractorsreceivenon-trivialprobabilitymass(strongercompetition)andshouldcoincide
with lower probe accuracy when items are genuinely harder.
7

Mean EntropyH↑Std. Dev. ofH↑Probe Acc. (%)↓
Split Plain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHTPlain RAG RAG+KG RAG+ValKNIGHT
History (L1) 0.0 0.0106 0.0122 0.0098 0.0134 0.0 0.0058 0.0692 0.0046 0.0855 100.00 99.00 88.00 98.00 86.83
Biology (L1) 0.0 0.0038 0.0096 0.0046 0.0189 0.0 0.0007 0.0897 0.0004 0.0803 100.00 98.00 89.00 99.00 86.21
Math (L1) 0.0 0.0021 0.0256 0.0039 0.0231 0.0 0.0004 0.0991 0.0005 0.1084 100.00 100.00 84.00 98.00 84.29
History (L3) 0.0 0.0 0.0435 0.0011 0.0489 0.0 0.0 0.1846 0.0002 0.1703 100.00 99.00 69.00 99.00 66.29
Biology (L3) 0.0 0.0017 0.0191 0.0039 0.0278 0.0 0.0003 0.1156 0.0003 0.1144 100.00 99.00 71.00 99.00 66.48
Math (L3) 0.0 0.0 0.0699 0.0021 0.0826 0.0 0.0 0.2006 0.0001 0.2288 100.00 100.00 80.00 100.00 79.02
Table 4:Distractor competition via predictive entropy.Mean and standard deviation of predictive
entropy ( H) (higher ↑= more competitive distractors) and probe accuracy (lower ↓= harder items)
usingLLaMA 3.2-3B-Instructas a fixed probe.
History Biology MathKNIGHTAvg. MMLU ARC CSQA RACE MedMCQA OBQA Bench Avg.
Model L1 L3 L1 L3 L1 L3
GPT-4o[63]92.95 86.39 95.98 87.11 95.30 86.41 90.521st79.09 86.31 70.28 67.87 57.85 67.21 71.451st
Mistral Large[64]92.19 84.16 95.18 86.84 95.07 84.10 89.592nd68.76 72.32 55.35 70.17 43.44 58.66 61.452nd
Llama3-70B-Instruct[65]92.12 85.15 94.12 86.08 95.03 84.67 89.533rd59.67 67.09 55.49 58.21 41.67 40.94 53.853rd
Claude 3 Haiku[66]91.95 81.64 95.05 83.40 93.97 81.60 87.70 57.35 63.89 55.87 57.20 40.57 42.32 52.87
Qwen1.5 (1.8B)[67]76.76 72.45 79.88 72.89 75.84 71.43 74.88 9.99 15.84 31.13 34.91 4.70 20.37 19.49
Gemma (2B)[68]38.59 30.73 45.51 31.27 43.62 40.45 38.36 17.52 23.93 27.40 14.32 4.57 14.26 17.00
Human (n=200) 98.60 89.20 97.40 91.60 97.40 89.40 93.92 – – – – – – –
Table 5:Benchmark utility ofKNIGHT.Accuracy (%) of multiple models onKNIGHT, alongside
standard MCQ benchmarks.KNIGHT Avg.averages over domains and levels;Bench Avg.averages
over external suites. Superscripts denote rank by the corresponding average.
Findings(Table4).(1)Difficultysignal(Level1 →Level3).Acrossdomains,KNIGHTshowsthe
expected pattern: entropy increases from Level 1 to Level 3 while probe accuracy decreases, and Std.
Dev.risesatLevel3,indicatingabroader,morerealistichardnessspreadratherthantightlyclustered
difficulty. The same directionality is visible for KG-guided prompting (RAG+KG), reinforcing that
Htracks intended difficulty.
(2)Component-wisecontrast.Plainisdegenerate,withnear-zeroentropyand 100%probeaccuracy,
consistent with non-competitive distractors. Among grounded baselines,RAGandRAG+Val
remain near-zero in Hwith near-ceiling probe accuracy across splits, suggesting retrieval alone and
validator-only filtering (without KG guidance) does not reliably induce close distractors. In contrast,
RAG+KGsubstantially increases Hand lowers probe accuracy, indicating that KG-conditioned,
path-/fact-driven prompting is the main driver of semantically proximate distractors. FullKNIGHT
(RAG+KG+Validator)achievesthestrongestoverallcompetitionprofile(higher Hwithlowerprobe
accuracy) while remaining well-formed under the same validity constraints used in human audits
(Sec. 4.3.2).
(3)Domainpatternsandspread.MathatLevel3exhibitsthehighestentropy(andcorrespondingly
lowerprobeaccuracy),consistentwithespeciallyclosedistractors;HistoryandBiologyshowthe
same qualitative Level 1 →Level 3 shift. Across topics, the larger Level 3 Std. Dev. forRAG+KG
andKNIGHTsuggeststhatKG-drivenpromptingyieldsnotonlyharderitemsonaveragebutalso
greater within-split difficulty diversity, whereas non-KG baselines cluster near triviality.
5.2. AreKNIGHTdatasets reliable, hard, and usable as benchmarks?
Table5reportsacontrolledevaluationofmultipleLLMsonKNIGHT(threedomains ×twodiffi-
culty levels) alongside standard MCQ benchmarks (MMLU[ 9],ARC[57],CSQA[ 58],RACE[ 59],
MedMCQA[ 60],OpenBookQA[ 61]). We follow official (or de facto) evaluation scripts and adopt
theOpen-LLM-Leaderboard protocolof Myrzakhanet al. [62]tomitigate MCQselection biasand
ensurecross-modelcomparability;wereportbothaKNIGHTaverage(overdomains/levels)anda
benchmark average(over external suites).
Difficulty calibrationand reliability.Across models(from GPT-4o to2B-scale baselines),Level 3
accuracy is consistently lower than Level 1 within each domain, indicating a stable, model-agnostic
difficultyseparation. A200-itemhumanstudymirrorsthispattern(lowerL3vs.L1whileremaining
high),supportingthatitemsarewell-posed(unambiguouskeys,reasonabledistractors)andthat
Level 3 is genuinely more demanding rather than error-prone.
8

Convergentvaliditywithestablishedbenchmarks.Beyondwithin-domaincalibration,ranking
undertheKNIGHTaveragecloselymatchesthebenchmarkaverage(GPT-4ohighest,followedbyMistral
LargeandLlama3-70B-Instruct,withsmallermodelstrailing),suggestingthatKNIGHTcaptures
difficulty factors predictive of general QA competence rather than overfitting to a narrow topic
distribution or prompting style.
Parsimony and cost–quality trade-off.KNIGHTis token- and cost-aware: the topic KG is con-
structedonceandreusedasacompactrepresentationtogeneratemanyvariants(differentlevelsand
itempatterns)withoutrepeatedlyinjectinglongevidencepassages. Thisyieldslowmarginalcost
andenablesfrequentbenchmarkrefresh,whereasbroadstaticsuitesrequiresubstantialexperteffort
andlongerbuildcycles,andnaivepromptingpipelinesre-consumelongcontextseachgeneration
pass, increasing token and runtime overhead.
KNIGHTvs.broadstaticsuites.KNIGHTcomplementswide-coveragebenchmarkssuchasMMLU:
MMLUprovidesstandardizedbreadth,whereasKNIGHToffersrefreshable,topic-scopedevaluation
withfine-graineddifficultycontrolandexplicitmulti-hopstructure,whilepreservingrank-order
agreement with established suites.
5.3. Ablation Study: Component-wise Impact
We isolate the contribution of retrieval grounding, KG guidance, and validation using the staged
systems in Sec. 4.2 (Plain,RAG,RAG+KG,RAG+Val, and fullKNIGHT). We treatANSWERABLE
violations as a hallucination proxy (unsupported content under the system-provided evidence) and
use predictive entropy (Table 4) as an automatic signal of distractor competition and controlled
hardness.
Retrieval grounding (Plain →RAG).Table 2 shows that retrieval substantially reducesANSWER-
ABLEviolationsandstabilizestopicalalignment(Table3). However,RAGstillexhibitsnear-zero
entropywithnear-ceilingprobeaccuracy(Table4),suggestingthatgroundingalonedoesnotreliably
yield competitive distractors or meaningful difficulty separation.
ValidationwithoutKG(RAG →RAG+Val).Addingthevalidatormainlyimprovesitemvalidity: it
sharply reducesREPEATED,SINGLE_KEY, andOPTION_UNIQUENESSerrors and further lowers
ANSWERABLEviolations(Table2),whilepreservingtopicality(Table3). Yet,asinRAG,entropy
remainsnear zerowithnear-ceilingprobe accuracy(Table4), indicatingthatvalidationalone does
not strengthen distractor competition or enforce calibrated hardness.
KGguidancewithoutvalidation(RAG →RAG+KG).ConditioninggenerationonKGpaths/facts
is the main driver of stronger distractor competition: entropy rises markedly (especially at Level 3)
andprobeaccuracydrops(Table4),consistentwithsemanticallycloserdistractors. Withoutfiltering,
RAG+KGcanalsoincreasetopicaldriftandvalidityerrors(Table3;Table2),motivatingpost-hoc
constraints in the full pipeline.
Combined effect (RAG+KG →KNIGHT; RAG+Val →KNIGHT).Combining KG-guided struc-
turing with validation yields the strongest overall validity profile:KNIGHTachieves the lowest
violation rates, with consistently fewer unanswerable items across Level 1 and Level 3 (Table 2);
importantly,movingtoLevel3doesnotsubstantiallyincreaseviolations,suggestingdifficultycontrol
withoutdegradingvalidity.KNIGHTalsomaintainstopicality(Table3). Finally,itpreservesthe
KG-induced difficulty signal—higher entropy, lower probe accuracy, and increased dispersion at
Level 3—while avoiding the unfilteredRAG+KGerror patterns (Table 4). The contrast between
RAG+ValandKNIGHTisolates the added value of KG structuregiventhe same validator: KG
prompting provides an interpretable, reusable scaffold that induces competitive distractors and
controllable hardness, while the validator enforces item-writing constraints and further reduces
hallucination as measured by answerability.
9

6. Conclusion
We presentedKNIGHT, a knowledge-graph–driven framework fortoken-efficientandlow-costgener-
ation of topic-scoped four-option MCQ datasets withcontrollable difficulty. Across three domains
and two difficulty settings,KNIGHTproduces linguistically polished items while substantially
improvingvalidityoverretrieval-onlyprompting: expertauditsshowfewerduplicates,feweram-
biguous(multi-key)questions,strongeroptionuniqueness,andmarkedlylowerunanswerableitems,
treatingANSWERABLEviolations as a proxy for hallucination. Predictive entropy further provides
a practical, model-agnostic signal of distractor competitiveness and difficulty, aligning with both
human and model performance.
Beyond the specific case studies built from Wikipedia/Wikidata, the main contribution is the frame-
work itself: a reusable KG representation that can be constructed once per topic and then leveraged
to generate many question variants (levels, hop patterns, formats) at low marginal cost. This makes
KNIGHTcomplementarytobroadstaticbenchmarks: whilethosesuitesofferstandardizedcoverage,
KNIGHTenablesrapid,refreshable,syllabus-alignedevaluationwithexplicitmulti-hopstructure
and instructor-controlled difficulty.
Future work includes extendingKNIGHTbeyond single-answer MCQs, incorporating adaptive
difficulty tuning via model feedback, and strengthening robustness through adversarial evaluation
and explainability. We also plan to instantiate the framework on alternative ontologies and domains,
and to explore cross-lingual and multimodal settings.
References
[1]Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models.arXiv preprint arXiv:2001.08361, 2020.
[2]Imad Lakim, Ebtesam Almazrouei, Ibrahim Abualhaol, Merouane Debbah, and Julien Launay.
Aholisticassessmentofthecarbonfootprintofnoor,averylargeArabiclanguagemodel. In
Angela Fan, Suzana Ilic, Thomas Wolf, and Matthias Gallé, editors,Proceedings of BigScience
Episode#5–WorkshoponChallenges&PerspectivesinCreatingLargeLanguageModels,pages84–94,
virtual+Dublin,May2022.AssociationforComputationalLinguistics. doi: 10.18653/v1/2022.b
igscience-1.8. URLhttps://aclanthology.org/2022.bigscience-1.8/.
[3]YashGoel,AyanSengupta,andTanmoyChakraborty. Position: Enoughofscalingllms! lets
focus on downscaling.arXiv preprint arXiv:2505.00985, 2025.
[4]AndreiLopatenko. Compendiumofllm evaluationmethods. 2024. https://github.com/alo
patenko/LLMEvaluation.
[5]ExplodingGradients. Ragas: Supercharge your llm application evaluations. https://github.c
om/explodinggradients/ragas, 2024.
[6] Kristiyan Vachev, Momchil Hardalov, Georgi Karadzhov, Georgi Georgiev, Ivan Koychev, and
PreslavNakov. Leaf: Multiple-choicequestiongeneration. InEuropeanConferenceonInformation
Retrieval, pages 321–328. Springer, 2022.
[7]Vatsal Raina and Mark Gales. Multiple-choice question generation: Towards an automated
assessment framework.arXiv preprint arXiv:2209.11830, 2022.
[8]GiorgioBiancini,AlessioFerrato,andCarlaLimongelli. Multiple-choicequestiongeneration
using large language models: Methodology and educator insights. InAdjunct Proceedings of the
32nd ACM Conference on User Modeling, Adaptation and Personalization, pages 584–590, 2024.
[9]Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding.arXiv preprint
arXiv:2009.03300, 2020. URLhttps://arxiv.org/abs/2009.03300.
10

[10]Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and
Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL https:
//arxiv.org/abs/2009.03300.
[11]Yassir Lairgi, Ludovic Moncla, Rémy Cazabet, Khalid Benabdeslem, and Pierre Cléau. itext2kg:
Incremental knowledge graphs construction using large language models. InInternational
Conference on Web Information Systems Engineering, pages 214–229. Springer, 2024.
[12]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal,HeinrichKüttlerKulshreshtha,MikeLewis,Wen-tauYih,TimRocktäschel,Sebastian
Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks.
InAdvances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/
abs/2005.11401.
[13]KelvinGuu,KentonLee,ZoraTung,PanupongPasupat,andMing-WeiChang.Realm: Retrieval-
augmentedlanguagemodelpre-training. InProceedingsofthe37thInternationalConferenceon
Machine Learning (ICML 2020), pages 3929–3938, 2020. URL https://proceedings.mlr.press/
v119/guu20a.html.
[14]AlexanderFabbri,WojciechKryściński,etal.Sumeval: Re-evaluatingsummarizationevaluation.
InProceedings of EMNLP 2021, 2021.
[15]RickRejeleene,XiaoweiXu,andJohnTalburt. Towardstrustablelanguagemodels: Investigating
information quality of large language models.arXiv preprint arXiv:2401.13086, 2024.
[16]Lingfeng Zhong, JiaWu, QianLi, HaoPeng, andXindong Wu. Acomprehensivesurveyon
automatic knowledge graph construction.ACM Computing Surveys, 56(4):1–62, 2023. doi:
10.1145/3618295.
[17]Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard de Melo, Claudio
Gutierrez, Sabrina Kirrane, Jose Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier,
Axel-CyrilleNgongaNgomo,AxelPolleres,SabbirM.Rashid,AnisaRula,LukasSchmelzeisen,
Juan Sequeda, Steffen Staab, and Antoine Zimmermann. Knowledge graphs.ACM Computing
Surveys, 54(4):1–37, 2021. doi: 10.1145/3447772.
[18]Danilo Dessì, Francesco Osborne, Diego Reforgiato Recupero, Davide Buscaldi, and Enrico
Motta. Generating knowledge graphs by employing natural language processing and machine
learning techniques within the scholarly domain.Future Generation Computer Systems, 116:
253–264, 2021.
[19]YuqiZhu,XiaohanWang,JingChen,ShuofeiQiao,YixinOu,YunzhiYao,ShuminDeng,Huajun
Chen, and Ningyu Zhang. Llms for knowledge graph construction and reasoning: Recent
capabilities and future opportunities.World Wide Web, 27(5):58, 2024.
[20]OpenAI, Josh Achiam, Steven Adler, and .... Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
[21]Amirhossein Layegh, Amir H Payberah, Ahmet Soylu, Dumitru Roman, and Mihhail Matskin.
Wiki-based prompts for enhancing relation extraction using language models. InProceedings of
the 39th ACM/SIGAPP Symposium on Applied Computing, pages 731–740, 2024.
[22]DennyVrandečićandMarkusKrötzsch. Wikidata: Afreecollaborativeknowledgebase.Com-
munications of the ACM, 57(10):78–85, 2014. doi: 10.1145/2629489.
[23]Sathish Reddy, Dinesh Raghu, Mitesh M. Khapra, and Sachindra Joshi. Generating natural lan-
guage question-answer pairs from a knowledge graph using a RNN based question generation
model. In Mirella Lapata, Phil Blunsom, and Alexander Koller, editors,Proceedings of the 15th
Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long
Papers, pages 376–385, Valencia, Spain, April 2017. Association for Computational Linguistics.
URLhttps://aclanthology.org/E17-1036/.
11

[24]Yu Chen, Lingfei Wu, and Mohammed J Zaki. Toward subgraph-guided knowledge graph
question generation with graph neural networks.IEEE Transactions on Neural Networks and
Learning Systems, 2023.
[25]ZhenpingLi,ZhenCao,PengfeiLi,YongZhong,andShaoboLi. Multi-hopquestiongeneration
with knowledge graph-enhanced language model.Applied Sciences, 13(9):5765, 2023.
[26]VishwajeetKumar,YunchengHua,GaneshRamakrishnan,GuilinQi,LianliGao,andYuan-
Fang Li. Difficulty-controllable multi-hop question generation from knowledge graphs. In
Chiara Ghidini, Olaf Hartig, Maria Maleshkova, Vojtech Svátek, Isabel F. Cruz, Aidan Hogan,
Jie Song, Maxime Lefrançois, and Fabien Gandon, editors,The Semantic Web – ISWC 2019 – 18th
InternationalSemanticWebConference,Auckland,NewZealand,October26–30,2019,Proceedings,
Part I, volume 11778 ofLecture Notes in Computer Science, pages 382–398. Springer, 2019. doi:
10.1007/978-3-030-30793-6_22. URLhttps://doi.org/10.1007/978-3-030-30793-6_22.
[27]Yi Cheng, Siyao Li, Bang Liu, Ruihui Zhao, Sujian Li, Chenghua Lin, and Yefeng Zheng.
Guidingthegrowth: Difficulty-controllablequestiongenerationthroughstep-by-steprewriting.
In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors,Proceedings of the 59th
Annual Meeting of the Association for Computational Linguistics and the 11th International Joint
Conference on Natural Language Processing (Volume 1: Long Papers), pages 5968–5978, Online,
August2021.AssociationforComputationalLinguistics. doi: 10.18653/v1/2021.acl-long.465.
URLhttps://aclanthology.org/2021.acl-long.465/.
[28]SeongyunLee,HyunjaeKim,andJaewooKang.Liquid: aframeworkforlistquestionanswering
dataset generation. InProceedings of the AAAI Conference on Artificial Intelligence, volume 37,
pages 13014–13024, 2023.
[29]Steven Moore, Eamon Costello, Huy A Nguyen, and John Stamper. An automatic question
usabilityevaluationtoolkit. InInternationalConferenceonArtificialIntelligenceinEducation,pages
31–46. Springer, 2024.
[30]Alexander Shypula, Shuo Li, Botong Zhang, Vishakh Padmakumar, Kayo Yin, and Osbert
Bastani. Evaluating the diversity and quality of llm generated content.arXiv preprint
arXiv:2504.12522, 2025.
[31]RickRejeleene,XiaoweiXu,andJohnTalburt. Towardstrustablelanguagemodels: Investigating
information quality of large language models.arXiv preprint arXiv:2401.13086, 2024.
[32]Aviv Slobodkin, Omer Goldman, Avi Caciularu, Ido Dagan, and Shauli Ravfogel. The curious
case of hallucinatory (un)answerability: Finding truths in the hidden states of over-confident
large language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,Proceedings
of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3607–3625,
Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023
.emnlp-main.220. URLhttps://aclanthology.org/2023.emnlp-main.220/.
[33]XiangyuPeng,PrafullaKumarChoubey,CaimingXiong,andChien-ShengWu.Unanswerability
evaluationforretrievalaugmentedgeneration. InWanxiangChe,JoyceNabende,Ekaterina
Shutova, and Mohammad Taher Pilehvar, editors,Proceedings of the 63rd Annual Meeting of
theAssociationforComputationalLinguistics(Volume1: LongPapers),pages8452–8472,Vienna,
Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi:
10.18653/v1/2025.acl-long.415. URLhttps://aclanthology.org/2025.acl-long.415/.
[34]Sérgio Silva Mucciaccia, Thiago Meireles Paixão, Filipe Wall Mutz, Claudine Santos Badue,
AlbertoFerreiradeSouza,andThiagoOliveira-Santos. Automaticmultiple-choicequestion
generationandevaluationsystemsbasedonLLM:Astudycasewithuniversityresolutions.
InOwenRambow,LeoWanner,MariannaApidianaki,HendAl-Khalifa,BarbaraDiEugenio,
andStevenSchockaert,editors,Proceedingsofthe31stInternationalConferenceonComputational
Linguistics,pages2246–2260,AbuDhabi,UAE,January2025.AssociationforComputational
Linguistics. URLhttps://aclanthology.org/2025.coling-main.154/.
12

[35]Yixin Nie, Adina Williams, Emily Dinan, Mohit Bansal, Jason Weston, and Douwe Kiela.
Adversarial NLI: A new benchmark for natural language understanding. InProceedings of ACL,
pages 4885–4901, 2020.
[36]SamuelR.Bowman,GaborAngeli,ChristopherPotts,andChristopherD.Manning. Alarge
annotated corpus for learning natural language inference. InProceedings of the 2015 Conference
onEmpiricalMethodsinNaturalLanguageProcessing(EMNLP),pages632–642.Associationfor
Computational Linguistics, 2015. URLhttps://aclanthology.org/D15-1075/.
[37]AdinaWilliams,NikitaNangia,andSamuelR.Bowman. Broad-coveragechallengedatasets
for sentence understanding. InProceedings of the 2018 Conference of the North American Chapter of
the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), pages
1112–1122.AssociationforComputationalLinguistics,2018. URLhttps://aclanthology.org
/N18-1101/.
[38]Gautier Izacard, LucasHosseini, Emmanuel De Bézenac,and Vladimir Karpukhin. Unsuper-
vised dense information retrieval with contrastive learning. InEMNLP, 2022.
[39]StephenRobertsonandHugoZaragoza. TheProbabilisticRelevanceFramework: BM25and
beyond.Foundations and Trends in Information Retrieval, 2009.
[40]NilsReimersandIrynaGurevych. Makingmonolingualsentenceembeddingsmultilingual
using knowledge distillation. InEMNLP, 2020.
[41]Zihan Ji and et al. Survey of hallucination in natural language generation.ACM Computing
Surveys, 2023.
[42]DennyVrandečićandMarkusKrötzsch. Wikidata: Afreecollaborativeknowledgebase.Com-
munications of the ACM, 57(10):78–85, 2014.
[43]HaoYu, YimingCui,andWanxiangChe. Largelanguagemodelsasdistractorgeneratorsfor
multiple-choice qa. InProceedings of ACL, 2024.
[44]ThomasM. Haladyna,StevenM. Downing,and MichaelC.Rodriguez. Areview ofmultiple-
choice item-writing guidelines for classroom assessment.Applied Measurement in Education, 15
(3):309–334, 2002.
[45]MichaelAlfertshofer,SamuelKnoedler,CosimaCHoch,SebastianCotofana,AdrianaCPanayi,
MartinKauke-Navarro,StefanGTullius,DennisPOrgill,WilliamGAustenJr,BohdanPomahac,
etal. Analyzingquestioncharacteristicsinfluencingchatgpt’sperformancein3000usmle®-style
questions.Medical Science Educator, pages 1–11, 2024.
[46]XinXu,TongXiao,ZitongChao,ZhenyaHuang,CanYang,andYangWang. Canllmssolve
longer math word problems better?arXiv preprint arXiv:2405.14804, 2024.
[47]Andrew M Bean, Karolina Korgul, Felix Krones, Robert McCraith, and Adam Mahdi. Do
large language models have shared weaknesses in medical question answering?arXiv preprint
arXiv:2310.07225, 2023.
[48]AkariAsai,ZeqiuWu,YizhongWang,AvirupSil,andHannanehHajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection.arXiv preprint arXiv:2310.11511, 2023.
URLhttps://arxiv.org/abs/2310.11511.
[49]LanguageTool Developers. Languagetool: Open-source grammar, style, and spell checker.
https://languagetool.org/, 2025. Accessed: 2025-10-06.
[50]language-tool-python Contributors. language-tool-python: Python wrapper for languagetool.
https://pypi.org/project/language-tool-python/, 2025. Accessed: 2025-10-06.
13

[51]CitadelAI. Langcheck: Simple,pythonicbuildingblockstoevaluatellmapplications. https:
//github.com/citadel-ai/langcheck, 2023. Accessed: 2025-12-13.
[52]Steven M. Downing. The effects of violating standard item-writing principles on tests and
students: The consequences are serious.Medical Education, 39(3):291–296, 2005.
[53]Michael C. Rodriguez. Three options are optimal for multiple-choice items: A meta-analysis of
80 years of research.Educational Measurement: Issues and Practice, 24(2):3–13, 2005.
[54]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners.Advances in neural information processing systems, 33:1877–1901, 2020.
[55]Eunsu Kim, Juyoung Suk, Philhoon Oh, Haneul Yoo, James Thorne, and Alice Oh. Click:
A benchmark dataset of cultural and linguistic intelligence in korean, 2024. URL https:
//arxiv.org/abs/2403.06412.
[56]Hugo Touvron and et al. Llama: Open and efficient foundation language models. InNeurIPS,
2023.
[57]PeterClark,IsaacCowhey,OrenEtzioni,TusharKhot,AshishSabharwal,CarissaSchoenick,
and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning
challenge. InProceedingsofthe2018ConferenceonEmpiricalMethodsinNaturalLanguageProcessing
(EMNLP), 2018. URLhttps://arxiv.org/abs/1803.05457.
[58]Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A
question answering challenge targeting commonsense knowledge. InProceedings of the 2019
Conference of the North American Chapter of the Association for Computational Linguistics (NAACL),
2019. URLhttps://arxiv.org/abs/1811.00937.
[59]Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale
reading comprehension dataset from examinations. InProceedings of the 2017 Conference on
EmpiricalMethodsin NaturalLanguageProcessing(EMNLP),2017. URL https://arxiv.org/ab
s/1704.04683.
[60]Abhishek Pal, Logesh Kumar Umapathi, and Malaikannan Sankarasubbu. Medmcqa: A large-
scale multi-subject multi-choice dataset for medical domain question answering. InProceedings
of the Conference on Health, Inference, and Learning, volume 174 ofProceedings of Machine Learning
Research, pages 248–260, 2022. URLhttps://proceedings.mlr.press/v174/pal22a.html.
[61]Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct
electricity? anewdatasetforopenbookquestionanswering. InProceedingsofthe2018Conference
on Empirical Methods in Natural Language Processing (EMNLP), 2018. URL https://arxiv.org/
abs/1809.02789.
[62]Yerbolat Myrzakhan, Nelson F. Ho, Han Liu, et al. Open llm leaderboard: Heterogeneous,
dynamic, and robust evaluation of llms.arXiv preprint arXiv:2406.07545, 2024.
[63]OpenAI. GPT-4o: Systemcardandmodeloverview. https://openai.com/index/gpt-4o-sys
tem-card/, 2024. Accessed 2025-10-06.
[64]Mistral AI. Mistral large. https://mistral.ai/news/mistral-large/ , 2024. Accessed
2025-10-06.
[65]Meta AI. Llama 3 model card and evaluations. https://ai.meta.com/llama/ , 2024. Accessed
2025-10-06.
[66]Anthropic. Claude3modelfamily: Modelcardandsystemoverview. https://www.anthropic.
com/claude, 2024. Accessed 2025-10-06.
14

[67]Yuxiao Bai, Weizhe Dai, An Yang, et al. Qwen technical report: An open large language model
family.arXiv preprint arXiv:2309.16609, 2023.
[68]GoogleDeepMindandGoogleResearch. Gemma: Openmodelsbuiltfromtheresearchbehind
gemini.https://ai.google.dev/gemma, 2024. Accessed 2025-10-06.
[69]Thomas Petersen, Pouya Golchin, Jinwoo Im, and Felipe PJ de Barros. Electrokinetic ef-
fectsonflowandiontransportincharge-patternedcorrugatednanochannels.arXivpreprint
arXiv:2510.22182, 2025.
[70]Faezeh Dehghan Tarzjani and Bhaskar Krishnamachari. Computing the saturation throughput
for heterogeneous p-csma in a general wireless network. In2025 34thInternational Conference on
Computer Communications and Networks (ICCCN), pages 1–7. IEEE, 2025.
[71]LangChain AI. langchain: Build context-aware reasoning applications. https://github.com/l
angchain-ai/langchain, 2025.
[72] MatthewHonnibalandInesMontani. spaCy2: Naturallanguageunderstandingwithbloom
embeddings, convolutional neural networks and incremental parsing. https://spacy.io , 2017.
[73]ThomasWolf,LysandreDebut,VictorSanh,JulienChaumond,ClementDelangue,Anthony
Moi,PierricCistac,TimRault,RémiLouf,MorganFuntowicz,JoeDavison,SamShleifer,Patrick
vonPlaten,ClaraMa,YacineJernite,JulienPlu,CanwenXu,TevenLeScao,SylvainGugger,
Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art
natural language processing. InProceedings of the 2020 Conference on Empirical Methods in
NaturalLanguageProcessing: SystemDemonstrations,pages38–45.AssociationforComputational
Linguistics, 2020.
[74]Neo4j, Inc. Neo4j developer documentation. https://neo4j.com/docs/ . Accessed: May 20,
2025.
[75]Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdi-
nov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop
questionanswering. InProceedingsofthe2018ConferenceonEmpiricalMethodsinNaturalLanguage
Processing, pages 2369–2380. Association for Computational Linguistics, 2018.
A. Limitations
Modelchoice.KNIGHTismodularandcanusedifferentLLMsfordifferentstages. Inthispaper,
for cost and reproducibility, we use a single model (GPT-4o-mini) for all LLM calls; we do not
perform an exhaustive, task-wise model selection study.
Data domain.We selected History, Biology, and Mathematics to represent a broad spectrum of
relationaldiversity,rangingfromnarrative-heavyeventchainstoabstractlogicalstructures. However,
ourfindingsmaynotfullygeneralizetodomainswithlowrelationaldensity,suchasPhysics[ 69]
or Numerical computation [ 70], where knowledge is often encoded in numerical constants and
first-principle equations rather than explicit entity-relation triples. In such "calculation-heavy"
domains,thegraph-basedgroundingusedbyKNIGHTmayprovidelessutilitythaninthehighly
interconnected domains studied here.
Residual hallucination.Grounding and validation substantially reduce unsupported content, but
donoteliminateit. WeoperationalizehallucinationviatheANSWERABLEauditflag(Sec.4.3.2);
whileKNIGHTlowers this rate compared to baselines, it remains non-zero (Table 2).
Difficulty is multi-factorial.Our primary hardness control relies on KG-based signals (e.g., multi-
hopstructure/graphdistance),whichcorrelatewellwithobserveddifficulty,butlinguisticcom-
plexity and domain prerequisites can also affect hardness.
15

Evaluationscope.WeevaluateonlytheWikipedia/Wikidatainstantiationandthreedomainsas
case studies; extending the evaluation to other corpora and ontologies is left for future work.
B. Ethics Statement
KNIGHTis released to support research and educational use via transparent, reproducible, low-cost
generation of topic-scoped MCQ datasets with controllable difficulty. As with any open-source
content-generation tool, it may be misused (e.g., to produce misleading content); we therefore
encourage responsible use consistent with research integrity and institutional guidelines.
Our study uses only publicly available sources (Wikipedia/Wikidata) and does not involve personal
orsensitivedata. UsersapplyingKNIGHTtoprivatecorporashouldensurecompliancewithprivacy
and licensing requirements and avoid including personally identifiable information.
Expert annotation targeted item quality and contained no sensitive content; exemption from IRB
review was determined according to institutional guidelines. Any AI assistant usage was limited to
editorialandstylisticrevisionsanddidnotcontributetoresearchdesign,datacollection,oranalysis.
C. System Usage
In this section we first outline installation, then present the two user interfaces (API & CLI), and
finally detail KG construction, generation, and validation.
C.1. Installation and Configuration
The framework targetsPython≥3.11and installs in a single step:
$pip install knight-framework
Alltransitivelyrequiredlibrariesareversion-pinnedin uv.lock,notablyLangChain[ 71],spaCy3[ 72],
Transformers 4 [ 73], and the Neo4j [ 74] Python driver; this guarantees byte-identical reproduction.
Externalservices.ANeo4j5.xinstanceprovidespersistentKGstorage,accessedviatheBoltprotocol:
$export NEO4J_URI=bolt://localhost:7687
$export NEO4J_USER=neo4j
$export NEO4J_PASS=<pwd>
MemorycanbeusedinsteadofNeo4jinstancebypassing backend="memory" totheconstructorwhich
results in a non-persistent KG storage.
For item synthesis we default to GPT-4o-mini. The API key of LLM must be provided:
$export OPENAI_API_KEY=sk-••••
Any HuggingFace-compatible decoder (e.g., Llama-2 [ 56]) can be hot-swapped by setting
lm_backend="hf".
C.2. Unified Workflow (API & CLI)
Bothinterfacesexposeidenticalfunctionality;weillustrateeachworkflowwithaminimaldepth- 2
example that produces ten MCQs onBiology.
from knight import KnightFramework as kframe
kf = kframe(uri="bolt://localhost:7687",
user="neo4j", password="neo4j")
kf.build_kg(topic="Biology", depth=2)
16

ds = kf.generate(prompt="multiple-choice",
topic="Biology", depth=2, num_q=10)
report = kf.validate(ds)
ds.to_json("bio_d2.json")
Now for the CLI we have:
$knight –topic "Biology" –prompt "multiple-choice" –depth 2 –num-q 10 –output bio_d2.json
–validate
C.3. Advanced Settings
Allcomponentsareplug-and-play: alternativeKGs(e.g.,Wikidata),relationwhitelists,orcustom
prompttemplatescanbeswappedwithouttouchingcorelogic,promotingreproducibleablations.
Full configuration options for replication are available in ourREADME.md.
In sum,KNIGHTcombines structured knowledge retrieval with controllable LLM generation to
deliver fact-grounded, difficulty-calibrated MCQ datasets, suitable for both educational deployment
and rigorous LLM evaluation.
D. Prompts
Thissectiondocumentsthepromptsemployedinourstudy,withparticularemphasisonfew-shot
promptingtechniquesthatenablemodelstoperformnoveltaskswithoutparameterupdates[ 54].
Eachpromptconsistsofatitleshowingtowhichprocessitbelongsandwhetheritisasystemprompt
or a user prompt, a purpose explaining its usage, and content.
Structured Term Explanation System Prompt
Purpose:
Sets the LLM’s persona as a scientific subject-matter expert and defines a required 8-point structure for
generating comprehensive term explanations.
Content:
You are a subject-matter expert in a scientific field. Your task is to provide detailed, thorough, and
academically structured explanations about terms provided by the user. Each term should be explained
exhaustively using the following structure:
1.Definition and Scope – Provide a precise, scientific definition of the term. Outline its general scope,
including the boundaries and extent of its meaning and use.
2.Domains of Use – Identify all relevant scientific, technical, or professional domains where this term plays
a key role. Specify the fields in which this concept is critical and explain its importance in each.
3.Subfields and Disciplines – Break the term down into its major subfields, branches, or areas of study.
Provide a brief but comprehensive overview of each subfield, including key principles, practices, and
contributors.
4.Key Concepts and Mechanisms – Describe the most important ideas, mechanisms, or processes associated
with this term in various contexts. Explain how these ideas interconnect.
5.Real-WorldApplications–Discussthemajorpracticalapplicationsofthisconceptindifferentspheres,
such as industry, healthcare, environmental science, etc.
6.CaseStudiesandExamples–Providespecificcasestudies,examples,orpracticaldemonstrationsofthe
term in action. Show how it is applied in real-world scenarios.
7.Related and Overlapping Terms – Identify related or similar terms and concepts. Clarify how they are
connected, and explain any subtle distinctions.
8.Current Research and Trends – Briefly cover the current research directions, innovations, and debates
around this concept. Mention any ongoing advancements or challenges in the field.
Yourexplanationshouldbeclear,well-organized,scientificallyaccurate,andeducational. Assumethatthe
user is unfamiliar with the term, so explain each concept thoroughly. Use precise language and cite notable
17

research, when possible. Dive deeply into subtopics as needed to provide a full understanding of the term’s
scope and implications.
Structured Term Explanation User Prompt
Purpose:
Used when an unambiguous Wikipedia summary is found. It instructs the LLM (paired with the
System Prompt above) to generate the structured explanation for a specific “ [term]”, using the
“[wikipedia_summary]” as the primary source and optionally considering the “[parent_term]”.
Content:
Now,pleaseapplythestructuredexplanationapproachdefinedinthesystemprompttoexplaintheterm:
“[term]”.
Use the following Wikipedia context as the primary source for your explanation, structuring your response
according to the system prompt guidelines:
— Wikipedia Context —
“[wikipedia_summary]”
— End Wikipedia Context —
Also consider its relationship to the parent term “[parent_term]”.
(Note: The last line regarding “[parent_term]” is conditional)
Wikipedia Title Relevance Check System Prompt
Purpose:
SetsthecontextfortheLLM,tellingittoactasarelevanceclassifieror"domain-specificsemanticfilter". It
needs to decide if a given Wikipedia page title is a good source for defining a specific term, considering the
provided context. It explicitly asks for a "Yes" or "No" answer only.
Content:
YouareperformingarelevanceclassificationtasktoevaluatewhetheraWikipediapagetitleisanappropriate
definition source for a given term within a specific context.
You are expected to act as a domain-specific semantic filter.
Answer "Yes" only if the title refers directly to the term and aligns with the context.
If the title is ambiguous, only tangentially related, or contextually irrelevant, answer "No".
Respond with only one word: "Yes" or "No".
Wikipedia Title Relevance Check User Prompt
Purpose:
Provides the specific data for the LLM to evaluate: the “ [term]” needing definition, the “ [title_guess] ”
(candidate Wikipedia page title), and the “ [context_hint] ” (which could bea parent term, source text, or
"general knowledge"). It reiterates the request for a ’Yes’ or ’No’ answer.
Content:
Context: Information related to “[context_hint]”.
Term to define: “[term]”.
Candidate Wikipedia Page Title: “[title_guess]”.
Evaluate relevance and respond with only ’Yes’ or ’No’.
Forward MCQ Generation System Prompt
Purpose:
Instructs the LLM to act as a "structured question generation system". Its goal is to create an MCQ (question,
4 options, correct answer key) based on a multi-step path provided from a knowledge graph. The question
should require reasoning across the path, and the answer should be implied by the path details.
Content:
Youareastructuredquestiongenerationsystem. Yourtaskistogenerateaquestionandaconciseanswer
based on a multi-hop path in a knowledge graph and node descriptions.
18

The question must reflect reasoning over the multi-step relationships in the path.
The answer should be clearly implied by the path and descriptions, often referring to a specific node.
Forward MCQ Generation User Prompt
Purpose:
ProvidestheLLMwiththespecificdetailsneededtogeneratetheforwardMCQ:examplesofthetask,the
actualgraph“ [path_representation] ”,descriptionsofthe“ [start_node] ”and“[end_node] ”,anoptional
“[topic]” constraint, and strict formatting instructions for the output (Question, A, B, C, D, Correct Answer
key).
Content:
Follow the instructions in the system prompt to generate a multiple-choice question based on the provided
path and node descriptions.
— Few-Shot —
“[few-shot example]”
— End Few-Shot —
IMPORTANT: The generated Question and Options MUST be relevant to the overall topic: “[topic]”.
Now, generate for the following:
Path: “[path_representation]”
Start Node: “[start_node]”
Description: “[start_desc]”
End Node: “[end_node]”
Description: “[end_desc]”
IMPORTANT:YouMUSTgenerateexactlyfouroptions(A,B,C,D)andindicatethesinglecorrectanswer
key. Adhere strictly to the output format below.
Output:
Question: [Your generated question reflecting the multi-step path]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [A, B, C, or D]
Reverse MCQ Generation System Prompt
Purpose:
Sets the LLM’s role as a "reasoning assistant" focused on generatingreversequestions. The goal is to create an
MCQ where the “ [start_node] ” of the provided graph path is the correct answer. It suggests using the end
node’s perspective to guide the reasoning.
Content:
You are a reasoning assistant generating reverse questions from knowledge graph paths.
Your task is to generate a question that can be answered explicitly by the start node of a multi-hop path.
Use the end node’s perspective when possible to guide the reasoning backward.
Reverse MCQ Generation User Prompt
Purpose:
Provides the LLM with specific instructions and data to generate the reverse MCQ. It includes examples,
the graph, descriptions of “ [start_node] ” and “[end_node] ”, an optional “ [topic]” constraint, and strict
formatting instructions. Crucially, it emphasizes that the correct answer must be the “[start_node]”.
Content:
Followtheinstructionsinthesystemprompttogenerateamultiple-choicequestionwherethestartnode
(“[start_node]”) is the correct answer.
— Few-Shot —
19

“[few-shot example]”
— End Few-Shot —
IMPORTANT: The generated Question and Options MUST be relevant to the overall topic: “[topic]”.
Now, generate for the following:
Path: “[path_representation]”
Start Node: “[start_node]”
Description: “[start_desc]”
End Node: “[end_node]”
Description: “[end_desc]”
IMPORTANT:YouMUSTgenerateexactlyfouroptions(A,B,C,D)andindicatethesinglecorrectanswer
key (which MUST correspond to the option containing the Start Node name “ [start_node] ”). Adhere
strictly to the output format below.
Output:
Question: [Generated question targeting the start node]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: [Letter corresponding to the option containing the exact text “[start_node]”]
GPT Triplet Extraction System Prompt
Purpose:
This prompt instructs the LLM to extract significant subject-predicate-object triplets from the provided text.
It gives detailedguidelines on what to focus on (keyconcepts, important relationships) and what to ignore
(pronouns, generic terms). It specifies the required JSON output format and provides clear examples of
good and bad triplets.
Content:
You are an information-extraction specialist.
Extract only the most significant and meaningful “ [subject–predicate–object] ” triplets from any text you
receive.
Here are the guidelines you should follow :
•Focus on important entities: names, places, concepts, achievements.
•Include defining characteristics and significant relationships.
•Capture major influences, contributions, and key life events.
•Skip generic pronouns, articles, and common words.
•Write relations in clear lowercase and with underscores.
IMPORTANT: The generated output must accommodate with this format.
{
"triplets": [
{
"head": "specific_entity",
"relation": "significant_relation",
"tail": "important_concept"
},
{
"head": "major_figure",
"relation": "notable_achievement",
"tail": "specific_contribution"
}
]
}
Bellow are some of the good and bad examples:
— Few-Shot —
20

“[few-shot example]”
— End Few-Shot —
GPT Triplet Extraction User Prompt
Purpose:
Provides the LLM with the specific text to extract triplets based on the instructions given in system prompt.
Content:
Follow the instructions in the system prompt to extract subject-predicate-object triplets from the text below.
— Start of the text input —
“[text-content]”
— End of the text input —
MCQ Validation System Prompt
Purpose:
Defines the LLM’s role as an evaluator for MCQs generated from knowledge graph paths. It needs to
assess grammar/clarity, whether the correct answer key is supportedonlyby the provided path details, and
optionally, relevance to a given topic. It demands a specific output format.
Content:
YouareMCQ-validationassistant. Evaluateafour-optionmultiple-choicequestion(MCQ)usingonlythe
information supplied in the “Source Information” block. Answer with five “ [YES/NO] ” (or N/A) tags in the
exact order and casing shown below.
Checklist
1. GRAMMAR_FLUENCY
Is the Question spelled and phrased correctly and clearly?
2. SINGLE_CORRECT_KEY
Is exactly one option marked as correct?
3. OPTION_UNIQUENESS
Are all four options distinct (no duplicates or near-duplicates)?
4. ANSWERABLE_FROM_SOURCE
Does the indicated correct option follow solely from the Source (path, node excerpts) without outside
knowledge?
5. TOPIC_RELEVANCE
If a Topic is provided, is the MCQ clearly about that topic?
MCQ Validation User Prompt
Purpose:
Provides the LLM with the specific MCQ data (“ [question] ”, “[correct_answer_key] ”) and its source-
details (including the “ [path_representation] ”, “[start_node] ”, “[end_node] ” and etc) to evaluate. It lists
the evaluation criteria and specifies the required output format lines.
Content:Follow the instructions in the system prompt to evaluate the following MCQ basedonlyon the
Source Information.
— Few-Shot —
“[few-shot example]”
— End Few-Shot —
Now, evaluate the following question:
Question: “[question_text]”
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Correct Answer: “[correct_answer_key]”
21

Topic (optional): “[topic_or_blank]”
Source Information
Path: “[path_representation]”
Start Node: “[start_node]”
Description: “[start_desc]”
End Node “[end_node]”
Description: “[end_desc]”
IMPORTANT:YouMUSTgenerateexactly5responsesforeachcriterionbasedontheprovidedoutputbelow.
Output:
Grammar_Fluency: “[YES/NO]”
Single_Correct_Key: “[YES/NO]”
Option_Uniqueness: “[YES/NO]”
Answerable_From_Source: “[YES/NO]”
Topic_Relevant: “[YES/NO or N/A]”
Term Extraction System Prompt (Baseline pipeline)
Purpose:Defines the LLM’s role as an expert at identifying key encyclopedic terms from a text and specifies
strict JSON output requirements.
Content:Youareanexpertatidentifyingkeyencyclopedictermsfromatext.Extractonlythemostsignificant
and specific terms from the provided text. These terms should be ideal candidates for a Wikipedia or
encyclopedia lookup. Return your answer strictly in the JSON schema shown below.
GUIDELINES1. Focusonconcrete nouns, namedentities, andspecificscientific concepts. 2. Keepthe terms
concise and specific. 3. Extract the base form of a term (e.g., cell” instead of cells”). 4. Ensure the entire
output consists strictly of the JSON object, with no preceding or succeeding text.
OUTPUT FORMAT (MANDATORY)
"terms": [
"term1",
"term2",
"term3"
]
EXAMPLES (GOOD)
✓mitochondria”
✓Gregor Mendel”
✓photosynthesis”
✓natural selection”
AVOID (BAD)
×various aspects”
×complex functions”
×scientific study of life”
×living organisms”
E. Additional Evaluation Metrics
The main paper focuses on grammatical fluency, presence of a single correct answer option, answer-
ability, and topic-relevance. Here we documentadditionalmetrics that were computed for every
dataset but omitted from the core discussion for space and interpretability reasons.
E.1. Question Length Diversity
Recent studies indicate that question length can significantly influence LLM accuracy. Bean et al.
[47]foundthatlongermedicalexamquestionswereassociatedwithlowermodelaccuracy. Similarly,
Alfertshofer et al. [45]reported that ChatGPT was more likely to answer longer USMLE-style
questionsincorrectly. Xuetal. [46]likewiseobservedthatLLMsachievesignificantlyhigheraccuracy
inshortermathwordproblems. Motivatedbythesefindings,weensuredthatourgeneratedquestions
22

10 15 20 25 30
Words per question0.000.020.040.060.080.100.120.14DensityBio-3
N(µ= 17.6, σ= 3.8)
10 15 20 25 30
Words per question0.000.020.040.060.080.100.120.14DensityMath-3
N(µ= 17.1, σ= 3.9)
10 15 20 25 30 35
Words per question0.0000.0250.0500.0750.1000.1250.150DensityHist-3
N(µ= 17.2, σ= 3.9)
10 15 20 25 30
Words per question0.000.020.040.060.080.10DensityBio-1
N(µ= 18.1, σ= 4.2)
10 15 20 25 30
Words per question0.000.020.040.060.080.100.12DensityMath-1
N(µ= 17.2, σ= 4.2)
10 15 20 25
Words per question0.000.020.040.060.080.100.12DensityHist-1
N(µ= 16.4, σ= 4.0)Figure 3: Distribution of question lengths for each dataset (histograms), demonstrating an approxi-
mately normal shape.
spanabroadrangeoflengths. Thisdesignallowsus toevaluatemodelperformanceonboth short
andlongquestions. Figure3showstheresultingdistributionofquestionlengthsforeachdataset;
notably, these distributions closely approximate a normal shape, indicating a balanced mix of short
and long questions.
E.2. Formalizing High-Quality Four-Option MCQs
Akeycontributionofthisworkistheprecisespecificationandvalidationoffivecorecriteriathat
distinguish a high-quality multiple-choice question (MCQ) with exactly four options. Below we
restateeachcriterion,provideitsformaldefinition,andillustratecompliantversusnon-compliant
examples.
1. Grammatical FluencyEnsures the stem and options are free from spelling or grammatical errors
andreadnaturally. Formally,aquestion qsatisfiesthiscriterionifitpassesbothautomatedgrammar
checks and human inspection for clarity and style.
We quantitatively assess grammatical accuracy of each question qcomprising Wwords by detecting
the number of grammatical errors Eusing the LanguageTool4system [49,50]. TheGrammar Quality
Scoreis defined as
GrammarQuality(q) = 1−E
W(4)
which penalizes questions proportionally to the error frequency relative to length. For fluency
evaluation, we employ the LangCheck toolkit [ 51], which estimates naturalness via normalized
log-probabilities from a pretrained language model. Higher fluency scores correspond to more
coherent and natural text, a correlation empirically supported by Fabbri et al. [14].
Example of Non-compliance:
Which of the following is thecapitalcity of France?
This question exhibits a grammatical error due to the omission of the definite article “the” and
contains awkward phrasing.
4GitHub for LanguageTool
23

Compliant form:
Which of the following is the capital city of France?
This formulation demonstrates correct grammar and natural syntactic flow.
2. Single Correct KeyExactly one optiono k∈ {o 1, o2, o3, o4}is correct:
∃!ok: Correct(o k) = True.(5)
This avoids ambiguity in scoring and interpretation.
Example of Non-compliance:Which are prime numbers?
Options:{2,3,4,5}(with two correct answers: 2 and 3).
This violates the single-correct-key criterion due to multiple correct options.
Compliant form:Which number is the smallest prime?
Options:{2,4,6,8}(only one correct answer: 2).
Thisquestionsatisfiesthesingle-correct-keyrequirementbyprovidingexactlyoneunambiguous
correct option.
3. Option UniquenessAll distractors must differ sufficiently from each other. For optionso i, oj:
sim(o i, oj)< δ,(6)
wheresimis a lexical/semantic similarity metric andδa low threshold.
Example of Non-compliance:
Options: {“New York City”, “NYC”, “Los Angeles”, . . . }.
The options include near-duplicate distractors(“New York City” and“NYC”), violatingthe option
uniqueness criterion due to high lexical and semantic similarity.
Compliant form:
Options: {“New York City”, “Los Angeles”, “Chicago”, “Houston”}.
The distractors are lexically and semantically distinct, satisfying the option uniqueness requirement
by providing clearly differentiated answer choices.
4. Answerability from SourceThe correct answer must be derivable solely from the provided
external knowledgeGand questionq:
P(ok|G, q)≫P(o i|G, q),∀i̸=k.(7)
Example of Non-compliance:
IfGlacks “Eiffel Tower” data, asking “Where is the Eiffel Tower?” is invalid.
Compliant form:
IfGcontains “Paris is the capital of France,” asking “What is the capital of France?”
is valid.
5. Topic RelevanceEnsures semantic alignment with the specified domain topic T. We compute an
entailment score:
S(q, T) =P(entailment|
premise=T,hypothesis=q)(8)
Hereahighentailmentscoreindicatesthatthegeneratedquestionsarestronglyalignedwithand
highly pertinent to the specified topic.
Example of Non-compliance:
A photosynthesis question in “World History.”
Compliant form:
“What sparked the outbreak of World War I?” in “World History.”
24

E.3. Quantitative Analysis of Expert Annotations and Quality Flags
Dataset GRAMMAR SINGLE_KEY OPTION_UNIQUENESS ANSWERABLE TOPIC
Hist-1 1 2 3 6 9
Bio-1 0 1 2 4 7
Math-1 2 2 2 5 8
Hist-3 2 2 3 6 10
Bio-3 1 2 1 4 4
Math-3 2 3 2 6 4
Table6: Aggregatedexpert-raisedflagsindicatingpotentialqualityviolationsbydatasetandcriterion.
Fewer than 5% of items trigger any flag, underscoring overall question quality.
Our human evaluation protocol was carefully designed to maximize both reliability and validity,
following established best practices in NLP evaluation studies. We recruited a total of thirty domain
experts, organized into three groups of ten, each group specialized in one of the three dataset
domains, to answer the benchmark questions. All thirty respondents were Iranian (nine female,
twenty-onemale),ranginginagefrom29to54years. Noneoftheseexpertsreceivedanyformof
compensation; their participation was entirely voluntary, consistent with standard definitions of
volunteer engagement.
Eachexpertanswered40questionsfromthedatasetassignedtothem. Becausewehad10experts
per topic and two datasets per topic, this yields 5 experts per dataset; at 40 questions each, a total of
200questionswerecompletedforeverydataset. Thisdesignbalancesworkloadwhilepreserving
annotationconsistency. Annotatorsweregivenunlimitedtimeandunrestrictedaccesstorelevant
resources to ensure comprehensive, accurate responses.
In addition to their primary assignments, 100 further questions per dataset were randomly sampled
for quality auditing. All experts, beyond their 40 primary questions, reviewed and flagged 20
randomlysampledquestionsfromeachdatasetaccordingtoourfivecoreevaluationcriteria. The
consistency of flags and judgments across datasets indicates robust sampling and a well-distributed
evaluation workload. Beyond measuring response accuracy, experts were instructed to flag any
question exhibiting ambiguity or quality concerns across our five core evaluation criteria on the
100 random samples of each dataset, facilitating nuanced qualitative feedback alongside robust
inter-annotatoragreementanalyses. Fluencyannotationswereconductedbyadedicatedteamoffive
additional experts, all Iranian (four male, one female), aged 25 to 41, with CEFR C1/C2 proficiency
certifications.
Importantly, allthirty-fiveparticipants(the thirtyquestion-answerersplusthe fivefluencyannota-
tors)werefullybriefedonthestudy’sobjectives,providedinformedconsent,andwereawarethat
their responses and annotations would be published.
We computed the Pearson correlation coefficient rbetween human error rates Ehumanand model
entropy scoresHacross datasets and difficulty levels, obtaining:
r=cov(E human , H)
σEσH≈0.78,(9)
indicating a strong positive correlation between human-perceived difficulty and model uncertainty.
Third, inter-annotator agreement, measured by Fleiss’ Kappa κ, consistently exceeded 0.82 in all
domains:
κ=¯P− ¯Pe
1−¯Pe>0.82,(10)
where ¯Pand ¯Pedenoteobservedandchanceagreement,respectively. Thisconfirmshighannotation
reliability.
Thesefindingsaffirmthatourdifficultystratification,groundedinknowledgegraphdepth,mean-
ingfully aligns with human cognitive assessments of question complexity. Moreover, the greater
25

increaseinmodelentropyfromlevel1to3relativetotheriseinhumanerrorratessuggeststhatlarge
languagemodelspossessheightenedsensitivitytosubtlecomplexityvariations,pointingtoward
promising directions for interpretability research.
In addition to these quantitative measures, detailed quality control was conducted through expert-
flaggedqualityviolationsacrossfivecriteria: grammar,singlecorrectkey,optionuniqueness,an-
swerability from source, and topic relevance. Experts evaluated questions thoroughly without strict
time constraints, enabling rich qualitative feedback.
Table 6 presents the aggregated counts of expert-raised flags per criterion and dataset.
Therelativelylowincidenceofflaggedissuesatteststothehighlinguisticcorrectness,andseman-
tic validity of the generated datasets, even as difficulty increases. This strong human validation
corroborates the effectiveness of our combined human-algorithmic quality assurance approach.
Overall, the rigorous quantitative and qualitative quality validation presented here is critical for
establishing trust in our datasets for downstream NLP tasks and benchmarking. It sets a replicable
standard for future large-scale QA dataset construction, ensuring semantic rigor and interpretability.
E.4. Significance Analyses for Topic Relevance
GoalandSetting.Wecompareper-itemtopicalitybetweenoursystem(KNIGHT)andaGPT4o-
minibaseline across subjects (History, Biology, Math) and difficulty levels (L1/L3). We evaluate
twocontinuoustopicalitysignalsin [0,1]: (i)anMNLI-basedentailmentscorethattreatsthetopic
as premise and the question as hypothesis; and (ii) an LLM-based topicality score computed via
few-shot prompting. Higher is better in both. The baseline set includes about 100items per split,
andKNIGHTabout1,000.
Pre-processingandqualitycontrol.Foreachsplitandsignalwevalidatebounds( [0,1]),remove
exactitemduplicates,andretainallremainingobservations. Weanalyzeeachsplitseparately;an
“overall”roll-upisprovidedonlyfordescriptivecontextandnotasasubstituteforper-splitinference.
Whatwetestandwhy.Weaskwhethertopicalitydiffersmeaningfullybetweensystems. Tocover
complementary notions of difference we use:
•Welch’stfor mean differences under unequal variances and unbalanced sample sizes.
•Mann–WhitneyU(MWU)andBrunner–Munzel(BM)fordistributionaldifferencesrobust
to non-normality, ties, and unequal variances/shapes, critical under ceiling compression
and strongn, imbalance.
•A light1% winsorized Welchas a sensitivity check to stabilize variance when many scores
cluster near 1.0.
•Effectsizes,Hedges’ gandCliff’s δ,toquantifypracticaldifferences;byconvention, |g|≲0.2
and|δ|<0.147indicate small effects.
Multiple comparisons.Within each test family (e.g., all Welch tests across the six splits per signal),
we control family-wise error using Holm’s step-down procedure. Unless otherwise noted, “non-
significant” refers to Holm-adjustedpvalues.
Ceilingeffectsand n-imbalance: interpretivecaveat.Thebaselinedistributionsareheavilycom-
pressednear1.0withverysmallvariance,andthebaselinesamplesizeismuchsmaller( ∼100vs.
∼1,000 ); parametric standard errors can become unrealistically small even when mean gaps are tiny,
sometimesmakingraw |t|looklargerthanwarranted. Rank-basedtests(MWU,BM)andeffectsizes
are therefore more reliable arbiters; we prioritize them alongside Holm-adjusted decisions.
Results: MNLI-basedentailment.Table7reportsWelch,MWU,BM,effectsizes,andHolm-adjusted
pper split and for a pooled “overall” summary. Across all splits,all Holm-adjusted p >0.05 andall
26

effect sizes are small. Medians are nearly identical and both systems concentrate near the top of the
scale. The winsorized Welch check does not change conclusions.
Split (Entailment)Welcht
(p)MWU
(p)BM
(p)Hedges’g
Cliff’sδHolmp
History (L1)−0.98(0.33)p= 0.41p= 0.37−0.10| −0.06 0.44
Biology (L1)−1.12(0.26)p= 0.49p= 0.44−0.09| −0.05 0.52
Math (L1)−1.35(0.18)p= 0.38p= 0.31−0.12| −0.070.18
History (L3)−1.28(0.20)p= 0.46p= 0.40−0.11| −0.06 0.39
Biology (L3)−0.89(0.37)p= 0.52p= 0.48−0.08| −0.04 0.60
Math (L3)−1.41(0.16)p= 0.35p= 0.29−0.13| −0.07 0.21
Overall−1.47(0.14)p= 0.32p= 0.28−0.12| −0.07 0.30
Table 7: Per-item topicality comparison (MNLI-based entailment). All tests are non-significant after
Holm; effect sizes are uniformly small. The minimum Holm-adjustedpacross splits is≈0.18.
Results: LLM-basedtopicality.Table8showsthesamepattern:allHolm-adjusted p >0.05 and
smallHedges’ gand Cliff’s δacross splits. Nonparametric evidence again indicates no stochastic
dominance. Sensitivity checks are consistent.
Split (LLM)Welcht
(p)MWU
(p)BM
(p)Hedges’g
Cliff’sδHolmp
History (L1)−1.05(0.29)p= 0.43p= 0.39−0.11| −0.06 0.46
Biology (L1)−0.92(0.36)p= 0.47p= 0.41−0.10| −0.05 0.50
Math (L1)−1.22(0.22)p= 0.39p= 0.33−0.12| −0.07 0.22
History (L3)−1.18(0.24)p= 0.45p= 0.40−0.11| −0.06 0.41
Biology (L3)−0.71(0.48)p= 0.54p= 0.50−0.08| −0.04 0.65
Math (L3)−1.36(0.17)p= 0.36p= 0.30−0.13| −0.07 0.20
Overall−1.31(0.19)p= 0.34p= 0.30−0.12| −0.07 0.29
Table8: Per-itemtopicalitycomparison(LLM-basedtopicalityscore). Alltestsarenon-significant
after Holm; effect sizes are small.
Integrated interpretation and consistency with main text.Across signals and splits, any apparent
baseline edge in raw means is not supported once ceiling and n-imbalance are accounted for: (a) all
Holm-adjusted p >0.05 (the smallestadjusted pobserved is ≈0.18), (b) effectsizes are uniformly
small,and(c)rank-basedtestsdonotindicatedistributionalshifts. Thisalignswiththemain-text
statementthat“afterHolm,all p >0.05 (min≈0.18),”andexplainsanyresiduallargeraw tasan
artifactofceiling/ n-imbalanceratherthanapracticallymeaningfulgap. Wethereforetreattopicality
asmatchedand focus the discussion on difficulty control, diversity, and validity.
Reproducibility.For every split and signal we report sample sizes, means, standard deviations,
medians, IQRs, Welch (statistic, d.f., p), MWU ( U, tie-corrected p), BM (statistic, p), Hedges’ g,
Cliff’s δ,andHolm-adjusted p. Theanalysiscodefixesseeds,appliesidenticalpre-processingtoboth
systems, and exports a complete per-item CSV plus a YAML manifest of test settings (winsorization,
tie handling) to enable byte-identical re-analysis.
Note (ceiling/ n-imbalance).Due to the combination of sample-size imbalance (e.g., ∼100 baseline vs.
∼1,000inKNIGHT)andnear-zerobaselinevariance(ceilingeffects),Welch’s tcaninflatedespite
27

negligible practical gaps; we therefore ground conclusions in effect sizes, rank-based tests, and
Holm-adjusted decisions.
E.5. Visualizing Entropy Distributions
Weutilizeaboxenplotcombinedwithaswarmplotoverlay(Figure4)tovisualizeentropydistri-
butionsacrosstopicsanddifficultylevels. Thisvisualizationmethodeffectivelydisplaysnotonly
central tendencies and spread but also highlights data density and outliers in a granular manner.
Figure 4: Entropy distributions by topic and difficulty level visualized using boxen plots with
swarmoverlays. Difficultylevel3datasetsconsistentlyshowhigherentropyandwiderdistributions,
reflecting greater model uncertainty.
The plot reveals several important insights:
•Higher Median and Spread at Difficulty Level 3:Across all domains, the median entropy
and interquartile ranges for difficulty level 3 datasets are notably larger than those for level
1, consistent with increased uncertainty in model predictions on harder questions.
•Domain Variation in Entropy:Biology datasets exhibit relatively lower entropy values
overall, which aligns with the statistical tests showing insignificant entropy differences
betweendifficultylevelsinthisdomain. HistoryandMathdomainsdisplaysubstantially
higher entropy at difficulty level 3, indicating a clear gradation of complexity.
•Presence of Outliers and Data Density:The swarm plot overlay reveals the distribution
density and presence of multiple outliers with unusually highentropy values, particularly
in the higher difficulty datasets. This suggests that a subset of questions poses exceptional
uncertainty for the model, possibly due to ambiguous or complex content.
•Skewness in Distribution:Especially in the Math-3 dataset, entropy distributions show
positiveskewness,indicatingaheaviertailtowardhigheruncertaintyvalues,reinforcing
the notion of increased challenge in these questions.
Thisvisualizationthereforeprovidesanuancedandrichdepictionofhowdifficultymodulatesmodel
uncertainty, confirming and extending conclusions drawn from entropy statistics and statistical
hypothesis tests.
28

F. Qualitative MCQ Examples
In this appendix, we present representative multiple-choice questions generated by our system,
illustrating how questions are grounded in the constructed knowledge graph (see Figure 5 for a
question generation example). Each example consists of the question’s data and the underlying
knowledgeroute. Theseexamplesspandiversedomainsandarearrangedtodemonstrateincreasing
reasoning complexity as the length of the knowledge graph path (number of hops) grows from
onetothree. Theseexemplarscanclarifyonhowoursystemintegratefactualtriplestoformulate
coherent questions of varying difficulty, with the correct answer highlighted in bold.
Biology – Level 1
Route:
"medicine & biology"INCLUDES− − − − − − − − − →"Pharmacology"
Question:
Whichfieldofstudyisincludedinthecomprehensiveoverviewofmedicineandbiology,focusingonthe
effects of drugs on biological systems?
A. Toxicology
B. Microbiology
C. Pharmacology
D. Immunology
Biology – Level 2
Route:
"medicine & biology"UTILIZES− − − − − − − − →"biological insights"INFORMS− − − − − − − − →"medicine"
Question:
What comprehensive academic overview provides detailed insights into the definition, scope, domains, and
applications of both medicine and biology?
A. Brief medical summary
B. Textbook on pharmacology
C. Medicine and Biology Overview
D. Public–health article
Biology – Level 3
Route:
"Biomedicine"APPLIES− − − − − − − →"biological principles"ENCOMPASS− − − − − − − − − − − →"genetics..."IS_A_BRANCH_OF− − − − − − − − − − − − − − − →
"biology"
Question:
Which branch of medical science integrates biological and physiological principles into clinical practice?
A. Pharmacology
B. Genetics
C. Biomedicine
D. Immunology
29

History – Level 1
Route:
"Social History"EXAMINES− − − − − − − − − →"societal structures"
Question:
Which historical subfield focuses on the experiences and perspectives of ordinary people?
A. Economic
B. Political
C. Cultural
D. Social
History – Level 2
Route:
"context"IS_VITAL_IN− − − − − − − − − − →"archaeology"FOCUSES_ON− − − − − − − − − − − − →"human history"
Question:
What term refers to the surrounding circumstances that influence the interpretation of human history in
archaeology?
A. Background
B. Setting
C. Context
D. Environment
History – Level 3
Route:
"Ottoman Empire"ENTERED− − − − − − − − →"World War I"ON_THE_SIDE_OF− − − − − − − − − − − − − − − →"Central Powers"OPPOSED− − − − − − − − − →
"Allied Powers"
Question:
Which major empire entered World War I on the side of the Central Powers, opposing the Allied Powers?
A. Austro-Hungarian Empire
B. German Empire
C. Ottoman Empire
D. Russian Empire
Mathematics – Level 1
Route:
"fundamental branch"REFERS_TO− − − − − − − − − − →"arithmetic"
Question:
What fundamental branch of mathematics studies numbers and basic operations?
A. Algebra
B. Geometry
C. Calculus
D. Arithmetic
30

Mathematics – Level 2
Route:
"linear algebra"PROVIDES_TOOLS_FOR− − − − − − − − − − − − − − − − − − − − →"solving systems of linear equations"USED_IN− − − − − − − →
"optimization"
Question:
Whichbranchofmathematicsprovidesessentialtoolsforsolvingsystemsoflinearequations,atechnique
frequently used in optimization problems?
A. Calculus
B. Discrete Mathematics
C. Linear Algebra
D. Probability Theory
Mathematics – Level 3
Route:
"eigenvectors"ARE_KEY_CONCEPTS_IN− − − − − − − − − − − − − − − − − − − − − →"linear algebra"IS_A_BRANCH_OF− − − − − − − − − − − − − − − →"mathematics"
INVOLVES− − − − − − − − →"logical reasoning"
Question:
Which branch of mathematics involves the study of eigenvectors and is essential for logical reasoning in
proofs and theorems?
A) Geometry
B) Linear Algebra
C) Calculus
D) Statistics
G. Graph Update and Curator Mechanism
Inthisappendix,wedescribeKNIGHT’sGraphUpdateandCuratorMechanism,whichensuresthatthe
knowledge graph grows in a controlled, non-redundant manner. We reference both the illustrative
traversal in Figure 5 and the pseudocode of the Curator module in Algorithm 2.
G.1. Formal Definition of Node Curation
LetG= (V, E) be the current directed graph, and let t∈Vbe the node under expansion. The
relation extraction stage applied to node tproposes a raw candidate set R={t′
1, t′
2, . . .}of potential
new childnodes. The Curator filters Rto produceacuratedsubset C⊆Rofunique, relevantnew
topics that satisfy the following conditions:
C=
t′∈R| ∀v∈V,¬Equiv(t′, v)
∧ϕ(t′) =True	
,(11)
where Equiv(t′, v)holdsif t′isjudgedsemanticallyequivalentto v,eitherbyexactornormalized
string match orbyhigh semantic similaritybased on cosinesimilarity of embeddings. Curator is
also a content filter that validates candidate topics by enforcing:
1.Object type agreement:Ensures the candidate’s semantic type aligns with Wikidata taxon-
omy [42].
2.Entailment consistency:Validates logical consistency between the candidate’s description
and the relation via natural language inference (NLI) probes [35].
3.Content-policy compliance:Checks adherence to content guidelines and filters out halluci-
nated or inappropriate information [15].
31

Onlycandidatespassingallthesechecksareretained;othersarediscardedandflaggedforhuman
auditbytheCuratormodule. Empirically,thismulti-stagefilteringprunesapproximately7.6%of
candidateedgesacrossdomains(§4),substantiallyimprovingthequalityandanswerabilityofthe
generated knowledge graph.
G.2. Knowledge-Graph Construction
Given a seed topic v0,KNIGHTcrawls Wikipedia and populates a property graph G= (V, E)
in Neo4j through a recursive expansion process. For each term (starting with v0), the system
generates a comprehensive description. This description generation conditionally utilizes Wikipedia
ascontextualinformationifavailableanddeemedrelevantbyanLLM;otherwise,itreliesonthe
LLM with a structured prompt. From the generated description, subject-predicate-object triplets are
extracted. These triplets identify new potential entities (nodes) and their relationships (edges). The
newly identified entities are then treated as new terms, and the description generation and relation
extraction process is applied recursively to them. The depth parameter dcontrols the maximum
extent of this recursive expansion from the initial seed topic, effectively defining the scope of the
resulting KG:
Vd=
v∈Vdist 
v0, v
≤d	
,(12)
where dist(·,·)denotes the shortest-path distance within the constructed graph. Depth therefore
acts as anintrinsic hardness knob: increasing dintroduces longer reasoning chains, echoing multi-hop
QA observations [75].
G.3. Graph Update and Curator Mechanism
Inthisappendix,wedescribeKNIGHT’sGraphUpdateandCuratorMechanism,whichensuresthatthe
knowledge graph grows in a controlled, non-redundant manner. We reference both the illustrative
traversal in Figure 5 and the pseudocode of the Curator module in Algorithm 2.
G.4. Formal Definition of Node Curation
LetG= (V, E) be the current directed graph, and let t∈Vbe the node under expansion. The
relation extraction stage applied to node tproposes a raw candidate set R={t′
1, t′
2, . . .}of potential
new childnodes. The Curator filters Rto produceacuratedsubset C⊆Rofunique, relevantnew
topics that satisfy the following conditions:
C=
t′∈R| ∀v∈V,¬Equiv(t′, v)
∧ϕ(t′) =True	
,(13)
where Equiv(t′, v)holdsif t′isjudgedsemanticallyequivalentto v,eitherbyexactornormalized
string match or by high semantic similarity based on cosine similarity of embeddings. Algorithm 2
illustratestheprocessfordetectingduplicatesandsemanticaliases( ¬Equiv(t′, v)). Curatorisalsoa
content filter that validates candidate topics by enforcing:
1.Object type agreement:Ensures the candidate’s semantic type aligns with expected taxon-
omy.
2.Entailment consistency:Validates logical consistency between the candidate’s description
and the relation via natural language inference (NLI) probes.
3.Content-policy compliance:Checks adherence to content guidelines and filters out halluci-
nated or inappropriate information using our designed RAG system.
Onlycandidatespassingallthesechecksareretained;othersarediscardedandflaggedforhuman
audit by theCuratormodule. If a semantic alias is detected, the candidate node t′is merged with the
existingnode v,typicallybydiscarding t′andre-attributingrelationsto v. Empirically,thismulti-
stage filtering prunes approximately 7.6% of candidate edges across domains (§4), substantially
improving the quality and answerability of the generated knowledge graph.
32

Algorithm 2Curator module: Uniqueness and Alias Filtering
Input:current graphG= (V, E); topict; raw candidatesR.
Output:curated setC⊆R.
1:C← ∅
2:for eacht′∈Rdo
3:s←normalize(t′)
4:ifs∈ {normalize(v)|v∈V}then
continue▷duplicate name
5:end if
6:for eachv∈Vdo▷semantic alias check
7:ifcos(e(t′),e(v))≥τthen
8:merget′withv ▷alias
9:continue to nextt′
10:end if
11:end for
12:C←C∪ {t′}▷unique and valid
13:end for
14:returnC
Oncethecuratedset C⊆Risdetermined,theknowledgegraphisexpandedbyaddingeachnew
topict′∈Cas a vertex and linking it to its parentt. Formally, this update is expressed as:
V(G)←V(G)∪C,
E(G)←E(G)∪ {(t→t′)|t′∈C}.(14)
This procedure guarantees semantic uniqueness and prevents redundancy by merging semantically
equivalent nodes, for example, recognizing that “Second World War” and “World War II” represent
the same entity.
Beyondthedetectionandmergingofduplicatesandsemanticaliases,theCuratorfurtherapplies
criticalvalidationcheckstoensuretheintegrityandrelevanceofgraphexpansions. Thesechecks
include verifying that the candidate node’s semantic type conforms to expected classes based on the
Wikidatataxonomy[ 42],ensuringthattherelationsanddescriptionsarelogicallyconsistentthrough
naturallanguageinference(NLI)techniques[ 35],andscreeningforcompliancewithcontentpolicies
to exclude hallucinated or inappropriate information using our designed RAG system. [15].
G.5. Question Generation
Given a topic–seed nodev 0,KNIGHTsamples a length-dpath through the knowledge graph:
v0r1− →v 1r2− → ···rd− →v d,(15)
where edges (ri)and intermediate nodes (vi), along with their descriptions, are retrieved from the
graph. The facts along this chain are verbalized into declarative sentences and embedded into a
few-shot prompt template. An LLM (by default GPT-4o-mini) then generates a multiple-choice
question (MCQ) consisting of (i) a question stem, (ii) one correct answer option, and (iii) three
plausible distractors often related to knowledge graph concepts [43].
To increase item diversity without additional graph traversals,KNIGHTproducesbothforward and
reverse variants of the path in Equation (15):
1.Forward mode( →): The answer node is vd, and the question stem is framed from the
perspective of the seed node v0, “moving outward.” For example, for d= 2, the question
might be:“Which entity, founded byv 0, later merged withv 1?”
2.Reverse mode( ←): The path is traversed backwards, treating v0as the correct answer. The
stem is phrased around the end nodev d, e.g.,“v dtraces its origins to which founding entity?”
33

Empirically, reverse questions increase model entropy by 15–20% (Section 4), serving as an effective
difficulty augmentation while maintaining factual grounding. Increasing the path length dtypically
leads to progressively harder MCQ templates by requiring the integration of information across
more hops in the knowledge graph. All generated MCQs, regardless of direction, are subsequently
filtered by rigorous quality criteria.
Figure5illustratesasampleknowledge-graphtraversalstartingfromtheseednodeHafez. Atdepth
1 (purple path), the algorithm identifiesShirazas a connected node. Deeper traversals (not shown)
discover nodes such as “7th century,” “Iran,” and “>90 million,” which correspond to progressively
harderMCQtemplates. TheCuratormoduleensuressemanticuniquenessbyaddingentitieslike
“Shiraz” only once, even if discovered through multiple paths.
Figure 5: Knowledge-graph traversal fromHafeztoShiraz,7th century,Iran, and>90 million,
generating MCQ templates of increasing difficulty. The purple path denotes a 1-hop (Level 1)
question.
H. Generation Speed and Experimental Hardware Setup
OurentireQAdatasetgenerationpipelinewasexecutedonGoogleColab,utilizinganNVIDIATesla
T4 GPU with 12 CPU cores. Despite this relatively modest hardware setup compared to large-scale
compute clusters, the framework demonstrated efficient runtime performance.
As shown by the empirical measurements in Subsection 4.4, KNIGHT enables fast and scalable
generation without reliance on expensive or specialized hardware. This efficiency makes it practical
for widespread use in research or educational applications, effectively addressing the common time
and resource barriers associated with the construction of large-scale QA datasets.
I. System Parameters and Configuration
The behavior and output quality oftheKNIGHTframeworkare influenced by various parameters
controlling the underlying models and processes. These parameters are typically configured before
runningthegenerationpipelineandallowfortuningperformance,creativity,andfilteringthresholds.
Here,weprovideanoutlineofkeyparametersandtheirrolesinoursystem,alongwiththeempirical
considerations that led to our default choices.
I.1. Language Model Inference Parameters
The Large Language Models used for description synthesis ( Ldesc), relation extraction ( Lrel), MCQ
generation ( Lq) and validation ( Lval) are controlled by the standard generative parameters used
primarily. Theexactmodelnameisconfigurableviatheenvironmentandkeyparameterscanbe
configured within the system’s source files including:
•Temperature:Temperaturecontrolstherandomnessoftheoutputdistributionduringthe
phaseoftokengenerationwhichrangesfrom0.0to1.0. Withthebaseofempiricaltesting
across differenttasks we foundthat a moderatetemperature is beneficialfor creative tasks
34

like initial description generation, while a lower temperature is crucial for structured output
tasks like triplet extraction where precision is always the first priority.
ForinitialLLMresponseslikethedescriptiongenerationtask,amoderatetemperatureof
0.4isusedtobalancecreativeresponsegenerationwithfactualgrounding. Initialtestswith
higher temperatures led to less coherent text, while lower temperatures reduced desired
linguistic variation.
Fortripletextraction,alowertemperatureof 0.1isusedtoencouragemoredeterministic
and focused extraction of structured information. Testing higher values resulted in less
reliable triplet formats, failing to consistently adhere to the required structure.
•Max Tokens:This parameter is set to manage output size and prevent excessive generation
costs. In our implementation, the triplet extraction process explicitly sets this to 2000. This
valuewaschosenbasedontestingthatshowedthislimitisgenerallysufficienttogenerateall
theinformationneededintheoutput,whilepreventingextremelylong,potentiallyirrelevant
outputs that could occur without a limit and increase costs or context window issues.
Thechoiceofthespecific OPENAI_MODEL (defaultingto gpt-4o-mini-2024-05 )wasdrivenbyempirical
evaluation. Comparative testing against larger models like GPT-4 demonstrated that for our specific
tasks and prompt engineering,gpt-4o-mini-2024-05provided a strong balance of accuracy, speed,
and significantly lower cost, making it the most practical default for large-scale generation pipelines.
I.2. Retrieval-Augmented Generation (RAG) Parameters
The RAG components, primarily utilizing the Wikipedia API, fetch information used for description
synthesis. Key parameters influencing this process include:
•Text Splitting Parameters:When fetching full Wikipedia page content for processing, a
recursive text splitter is used to break the text into manageable pieces. The RAG system
is configured with a chunk_size of1000tokens and chunk_overlap of100tokens. These
values were found to effectively divide the raw text into segments large enough to retain
sufficient context and useful dataforthe LLM while the overlap helps maintain continuity
betweenchunks. Testingsmallersizessometimesbrokeuprelatedfacts,whilelargersizes
could exceed context windows or introduce noise without improving retrieval quality.
•Maximum Returned Chunk Length:The routine fetching Wikipedia summaries limits the
size of the returned text snippet used as context for node description generation. A default
limit of1000characters is applied in the code. This value was chosen because empirical
observationshowedthatsnippetssignificantlylongerthanthisrarelyaddedsignificantvalue
fordescriptionsynthesisandincreasedpromptlengthunnecessarily,potentiallydilutingthe
mostrelevantinformationfortheLLM.AlsoshortervaluesdeprivetheLLMfromuseful
and important context data.
•Number of Search Results Checked:When the system needs information about a concept,
it searches Wikipedia. This parameter limits the number of top search results from this
initial search that the system will look at more closely for relevance before start fetching
contentfromeachpagewhichhas 5asthedefaultvalue. Checkingfewerthan5oftenmissed
relevantpageswhichresultsinlimitingthescopeofthegeneratedgraph,whilechecking
significantlymoreaddedconsiderablelatencyandcostwithoutaproportionalincreasein
the retrieval of highly relevant content.
These parameters try to retrieve relevant, concise, and contextually appropriate information to
ground the generated descriptions and optimize the process both in quality and efficiency.
I.3. Graph Construction and Curator Parameters
Only one parameter involves in the KG construction and curation process:
35

•MaximumBranches:Thisparametercontrolshowmanynewtripletsareextractedfromeach
main idea’s (node’s) description during the recursive graph expansion with 2as the default
value. Through empiricaltesting,we exploredlimitingbranchesto1, 2,3or4. Limitingto
1oftenresultedinashallow,lessinterconnectedgraphstructurewithmissingimportant
related concepts. Allowing more than 2 branches significantly increased computational
costandgraphsizewithoutconsistentlyyieldingaproportionalincreaseinthequalityor
relevance of generated question paths and sometimes introduced noise.
I.4. Validation and Filtering Parameters
Parameters governing the validation and filtering of generated QA pairs include:
•Validation Sample Rate:This parameter allows specifying a sample rate ranging from
0.0 to 1.0 for LLM-based validation of generated QA pairs. 1.0 is the default used for
complete coverage in our experiments reported here to provide the best possible quality,
but the parameter was added to enable flexibility for situations where speed or cost is more
important than verifying each and every pair generated.
Bytuningtheseparameters,userscanadjustthetrade-offbetweengenerationspeed,datasetsize,
linguistic style, and the nuances of difficulty calibration for specific domains and use cases. The
default values in our experiments represent a balance to produce high-quality, difficulty-calibrated
datasets efficiently.
36