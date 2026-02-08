# RAGTurk: Best Practices for Retrieval Augmented Generation in Turkish

**Authors**: Süha Kağan Köse, Mehmet Can Baytekin, Burak Aktaş, Bilge Kaan Görür, Evren Ayberk Munis, Deniz Yılmaz, Muhammed Yusuf Kartal, Çağrı Toraman

**Published**: 2026-02-03 15:35:11

**PDF URL**: [https://arxiv.org/pdf/2602.03652v1](https://arxiv.org/pdf/2602.03652v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances LLM factuality, yet design guidance remains English-centric, limiting insights for morphologically rich languages like Turkish. We address this by constructing a comprehensive Turkish RAG dataset derived from Turkish Wikipedia and CulturaX, comprising question-answer pairs and relevant passage chunks. We benchmark seven stages of the RAG pipeline, from query transformation and reranking to answer refinement, without task-specific fine-tuning. Our results show that complex methods like HyDE maximize accuracy (85%) that is considerably higher than the baseline (78.70%). Also a Pareto-optimal configuration using Cross-encoder Reranking and Context Augmentation achieves comparable performance (84.60%) with much lower cost. We further demonstrate that over-stacking generative modules can degrade performance by distorting morphological cues, whereas simple query clarification with robust reranking offers an effective solution.

## Full Text


<!-- PDF content starts -->

RAGTurk: Best Practices for Retrieval Augmented Generation in Turkish
Süha Kağan Köse1, Mehmet Can Baytekin1, Burak Aktaş1, Bilge Kaan Görür1,
Evren Ayberk Munis2,Deniz Yılmaz3,Muhammed Yusuf Kartal4,Çağrı Toraman3
1Roketsan Inc., Artificial Intelligence Technologies Unit, Turkey
2Politecnico di Torino, Italy
3Middle East Technical University, Computer Engineering Department, Turkey
4TOBB University of Economics and Technology, AI Engineering Department, Turkey
kagan.kose@roketsan.com.tr,can.baytekin@roketsan.com.tr
burak.aktas@roketsan.com.tr,kaan.gorur@roketsan.com.tr
evrenayberk.munis@studenti.polito.it,deniz.yilmaz_12@metu.edu.tr
m.kartal@etu.edu.tr,ctoraman@metu.edu.tr
Abstract
Retrieval-Augmented Generation (RAG) en-
hances LLM factuality, yet design guidance
remains English-centric, limiting insights for
morphologically rich languages like Turkish.
We address this by constructing a compre-
hensive Turkish RAG dataset derived from
Turkish Wikipedia and CulturaX, comprising
question–answer pairs and relevant passage
chunks. We benchmark seven stages of the
RAG pipeline—from query transformation and
reranking to answer refinement—without task-
specific fine-tuning. Our results show that
complex methods like HyDE maximize accu-
racy (85%) that is considerably higher than
the baseline (78.70%). Also a Pareto-optimal
configurationusingCross-encoderReranking
andContextAugmentationachievescompara-
ble performance (84.60%) with much lower
cost. Wefurtherdemonstratethatover-stacking
generative modules can degrade performance
by distorting morphological cues, whereas sim-
ple query clarification with robust reranking
offers an effective solution.1
1 Introduction
Large LanguageModels (LLMs)perform strongly
across many NLP tasks, yet they struggle when
queriesrequirecurrent,domain-specific,orverifi-
ableinformation. Retrieval-AugmentedGeneration
(RAG) mitigates these limitations by incorporat-
ing external evidence during generation (Lewis
etal.,2020). Overtime,RAGhasdevelopedinto
a modular pipeline—spanning from query trans-
formation to answer refinement—whose compo-
nentscollectivelyshapesystemperformance(Gupta
et al., 2024; Zhao et al., 2024; Liu et al., 2025a).
Many studies have improved individual stages of
1Links to our datasets and source code are available at:
https://github.com/metunlp/ragturkthis pipeline: dense retrievers (Karpukhin et al.,
2020), late-interaction models (Khattab and Za-
haria, 2020), LLM-based query expansion (Gao
etal.,2022;Lietal.,2025),cross-encoderrerank-
ing (Nogueira and Cho, 2019), and hierarchical
retrieval (Tao et al., 2025). Recent work also em-
phasizes iterative reasoning in RAG (Asai et al.,
2024; Jiang et al., 2023b) and holistic evaluation
metrics beyond answer accuracy alone (Yu et al.,
2024; Es et al., 2024).
However,nearlyallpriorRAGresearchtargets
English. Formorphologically richandmoderately
resourcedlanguageslikeTurkish,thebehaviorof
RAG systems remains largely unknown. Turk-
ish morphology, flexible word order, and varia-
tion across sources introduce retrieval and ground-
ing challenges not reflected in English bench-
marks. Existing work examines isolated com-
ponents—retrievers (Bikmaz et al., 2025), cul-
tural QA (Simsek, 2025), embeddings (Ezerceli
et al., 2025), and hallucination detection (Taş et al.,
2025)—but not full pipelines evaluated end-to-end
on curated Turkish benchmarks. Relevant back-
ground on Turkish NLP challenges and benchmark
evaluation appears in (Hakkani-Tür et al., 2002;
Oflazer, 2014; Umutlu et al., 2025).
Meanwhile, the broader RAG ecosystem is shift-
ing toward pipeline-level optimization. Frame-
works such as AutoRAG (Kim et al., 2024), DSPy-
RAG(Khattabetal.,2024),GraphRAG(Edgeetal.,
2024) and RAGSmith (Kartal et al., 2025) show
thatperformancedependsoncoordinatedcompo-
nent interaction rather than any single module. Yet
thesesystemsarealsoEnglish-centric,leavingopen
questions about cross-linguistic transferability.
Toaddressthisgap,wepresentthefirstsystem-
atic,end-to-endevaluationofRAGpipelinecompo-
nentsforTurkishonadatasetconsistingoftwopartsarXiv:2602.03652v1  [cs.CL]  3 Feb 2026

with similar properties (well-formed Turkish text
andgroundedquestion–answerpairswithverifiable
evidence passages) but different sources. Using
4,891TurkishWikipediaarticlesand6,305Turkish
web articles derived from CulturaX (Nguyen et al.,
2024),weassesssevencorepipelinestagesundera
unified protocol.
Our contributions are:
•AcomprehensiveTurkishRAGbenchmark:
we construct a unified dataset sourced from
TurkishWikipediaandCulturaX,generating
groundedquestion–answerpairswithgoldevi-
dence passages across factual and interpretive
question types.
•A systematic, end-to-end pipeline study for
Turkish:we benchmarkseven core stages of
a modern RAG stack (query transformation,
candidate re-ranking, filtering & selection,
contextaugmentation,condensation,prompt
composition, and answer refinement) under a
unified protocol.
•Optimizedrecipesandreproduciblerelease:
we distill actionable strategies—identifying
a Pareto-optimal configuration that balances
accuracyandefficiencywhilehighlightingthe
risks of over-stacking LLM modules—and re-
leasethedatasets,prompts,configurationfiles,
and evaluation scripts to support reproducible
Turkish-RAG research.
2 Related Work
Retrieval-AugmentedGeneration(RAG)strength-
ensLargeLanguageModels(LLMs)withgrounded,
domain-specific information. Since Lewis et
al.(Lewiset al.,2020),RAGhas developedintoa
modularpipelinewherechoicesacrosscomponents
(from query transformation to retrieval, rerank-
ing/selection, context construction/condensation,
prompting, and answer refinement) jointly deter-
mineperformance(Guptaetal.,2024;Zhaoetal.,
2024;Liuetal.,2025a). Recentevaluationworkcor-
respondingly argues for holistic assessment beyond
generation quality, emphasizing retrieval effective-
ness, grounding, attribution, and factual reliabil-
ity (Yu et al., 2024; Es et al., 2024).
Retriever Architectures and Query Trans-
formation.Dense retrieval replaces sparse
lexical matching with neural representations;
DPR (Karpukhin et al., 2020) demonstrates stronggains over BM25 on knowledge-intensive tasks,
and hybrid dense+lexical approaches further im-
prove robustness across query types (Lin et al.,
2021). Complementary work improves recall via
query reformulation/expansion: HyDE (Gao et al.,
2022) generates hypothetical documentsto bridge
lexical gaps, while surveys of query expansion
and multi-query strategies show consistent benefits
forambiguousorunderspecifiedqueries(Lietal.,
2025).
Candidate Re-ranking, Context Selection,
and Condensation.Reranking provides finer-
grained relevance estimates after retrieval; cross-
encoders(oftenBERT-based)remainthestandard
for high-precision ranking (Nogueira and Cho,
2019). Hierarchical/structured approaches such
as TreeRAG (Tao et al., 2025) and related context-
selection/condensation methods (e.g., selective ex-
traction, compression, ordering) aim to pack the
most useful evidence into limited context windows.
Answer Refinement and Self-Reflective RAG.
Beyond “retrieve-then-generate,” self-reflective
methodssuchasSelf-RAG(Asaietal.,2024)and
FLARE (Jiang et al., 2023b) let models check sup-
port,triggeradditionalretrieval,andreviseanswers,
improving factuality and robustness via iterative
retrieval–generation.
System-Level Optimization: AutoRAG, DSPy-
RAG, and GraphRAG.Recent work optimizes
RAG end-to-end rather than tuning single com-
ponents: AutoRAG (Kim et al., 2024) automates
configuration over retrievers, chunking, query ex-
pansion, rerankers, prompts, and post-generation
modules;DSPy-RAG(Khattabetal.,2024)treats
RAG assembly as optimization over declarative
modules;andGraphRAG(Edgeetal.,2024)uses
graph-basedindexingandhierarchicalretrievalto
exploit document structure. These systems high-
light strong component interactions, but are eval-
uated largely on English, leaving open questions
for morphologically rich and underrepresented lan-
guages.
RAG andRetrieval inTurkish.Acomprehen-
sive survey of Turkish NLP resources (Çöltekin
et al., 2023) provides essential background on
corpora and lexical resources for the language.
RAG research for Turkish is emerging: Bıkmaz
etal.(Bikmazetal.,2025)analyzeretrieversand
rerankers for Turkish QA, and Şimşek (Simsek,

2025)comparesRAGagainstfine-tuningforcultur-
ally grounded QA. Turkish retrieval resources such
as TurkEmbed4Retrieval (Ezerceli et al., 2025)
emphasize language-tailored embeddings, while
Turk-LettuceDetect (Taş et al., 2025) highlights the
needforgrounded,verifiableoutputs. Inaddition,
recentTurkishLLMbenchmarkingeffortssuchas
TurkBench (Toraman et al., 2026) include evalu-
ations of retrieval-augmented generation (RAG);
however, they remain limited to general-purpose
benchmarking and do not target the task-specific
retrieval and reasoning challenges considered here.
However, most Turkish studies focus on isolated
components rather than full pipelines, and do not
systematically test how design choices transfer to
Turkishdatasets(e.g.,Wikipediaarticlesandbroad-
coverage web text such as CulturaX) under rich
morphology and source-dependent variation.
Positioningourwork.Toaddressthelackofnon-
English RAG benchmarks, we present the first sys-
tematic,end-to-endstudyofTurkishRAGpipelines.
We construct a two-part dataset sourced from Turk-
ish Wikipedia and CulturaX, and evaluate seven
corepipelinestages—fromquerytransformationto
answer refinement—under a unified protocol. Our
analysis identifies Pareto-optimal configurations
thatbalanceaccuracywithefficiency,offeringac-
tionable “recipes” for Turkish retrieval. We release
theseresourcesandfindingstosupportreproducible
research in morphologically rich languages.
3 Dataset Construction
To ensure broad coverage of real-world retrieval
scenarios, we construct a unified Turkish RAG
benchmarkcomprisingtwocomplementaryparts
thatsharehighstandardsfortextqualityandanswer-
ability but differ in source characteristics: theWeb
Part (CulturaX)is a diverse collection of Turkish
web pages derived from CulturaX (Nguyen et al.,
2024), filtered to retain contentful text across a
widerangeoftopics(e.g.,everydaylife,entertain-
ment,news),whiletheWikipediaPartconsistsof
Turkish Wikipedia articles providing encyclopedic
referencetextwithstrongeremphasisonbiography,
STEM, and history.
3.1 Corpus Acquisition and Filtering
Toensurevalidevaluation,wefilterrawcrawlsto
keeponlycontentful,answerabledocuments. For
theWebPart,westartfromCulturaXandsample
candidate Turkish pages. We then apply a two-stage filtering procedure guided by LLM-based
judgments to ensure high retrieval utility:
1.URL-only filtering (triage).Given only the base
websiteURL,anLLMestimateswhetherthesite
likelycontainsvaluable,informationalcontent—
operationally, pages with factual statements that
could answer user queries (valuable) and sub-
stantiveproseratherthannavigationmenus,link
lists,orboilerplate(informational). Thisstage
acts as a low-cost pre-filter to exclude low-value
pagetypessuchaspurelandingpages,naviga-
tion hubs, or aggregator farms.
2.Content-based filtering (page-level quality).For
pages passing stage (1), we fetch the page text
and apply a second LLM filter that checksco-
herence,content depth, andutility. We retain
pagesthatareunderstandableandcontent-rich
(e.g., blog posts, forum discussions, news arti-
cles) withgood quality. We reject pages that
arespam(keywordstuffing,bots),thin(insuffi-
cientcontent),orboilerplate(largelynon-textual
content).
Prompt design for filtering.Filtering prompts
aredesignedtoretainanswerablepassageswhileex-
cludingcontentthattriviallybreaksevaluation. We
usegemini2.5flash(Google,2025)forbothstages;
full prompts are in Appendix A (see Prompt A.1
and Prompt A.2).
Final Web corpus.After filtering, we retain
6,305 web pages. We convert each page to Mark-
downbypreservingthemaintextualsectionsand
mapping prominent HTML headings to Markdown
headers.
WebsiteFrequencyAnalysis.Wereportthebase-
domain frequency to ensure the Web Part is not
dominated by a single source. The top domains
includepopularsiteslikesikayetvar.comandhaber-
ler.com,butthetop10domainscoveronly ∼19.6%
of documents, ensuring diversity (see Appendix B,
Table 5 for full list).
WikipediaArticles.FortheWikipediaarticles,
acquisitionisstraightforwardgiventhestructured
nature of the source. We randomly sample Turk-
ish Wikipedia pages, exclude short articles (< 300
chars), and retrieve plain-text sections. After filter-
ing, we retain 4,891 articles and convert them to
Markdown, mapping sections to headers.

Table 1: Topic statistics per dataset part and overall
totals. Web/Wikipediapercentagesarecomputedw.r.t.
the full dataset (N=11,196).
TopicWebWikipedia Total
% % # %
Entertainment 10.7 8.12,104 18.8
Biography 1.4 13.91,712 15.3
Everyday Life 12.4 0.21,414 12.6
STEM 5.2 6.61,322 11.8
Politics 9.9 1.61,298 11.6
Professional 7.4 1.0 939 8.4
History 3.8 4.5 925 8.3
Organizations 3.4 2.2 627 5.6
Geography 1.0 3.7 525 4.7
Humanities 0.9 1.9 309 2.8
Uncategorized 0.1 0.0 21 0.2
Total 56.3 43.711,196 100.0
3.2 Header-Aware Chunking
Weapplyaheader-awarechunkingstrategy. Each
chunk inherits document and section context (e.g.,
pagetitleandsectionpath). Longsegmentsaresplit
iftheyexceed1,000characters. Wetrieddifferent
thresholdsandfoundthischaracterlimitprovides
the best balance between (i) preserving enough
local context for answerability and (ii) limiting
topic drift. This tokenizer-agnostic limit is particu-
larlystableforTurkish,whereagglutinationpacks
more information into fewer whitespace-delimited
tokens.
3.3 Topic Categorization
We annotate each document with a unified topic la-
belusinganLLM-basedclassifierusingagaingem-
ini 2.5 flash (Google, 2025) as the LLM provider.
Weadaptourcategorizationapproachandprompt
design considerations from prior work on large-
scale dataset construction (Gao et al., 2020; Sol-
daini et al., 2024; Weber et al., 2024; Penedo et al.,
2024;Wenzeketal.,2020;Elazaretal.,2023). Full
prompts are in Appendix A (see Prompt A.3).
Topic taxonomy.We define 10 broad topic cat-
egories (Table 1). The distribution highlights the
complementarynatureofthedataset: TheWebPart
leans towardsEveryday Life,Entertainment, and
Politics, reflecting the conversational web. The
WikipediaParthashighercoverageofBiography,
STEM, andHistory. Taken together, the dataset
spans the full range of user queries, from checking
facts on public figures to navigating forum advices.Table2: CorpusandQAstatisticsforthecomplementary
parts.
Statistic Web Part Wikipedia Part Total
Articles 6,305 4,891 11,196
Characters 9,933,523 22,821,895 32,755,418
Char./Article 1,575.50 4,666.10 2,925.46
Chunks 15,985 42,304 58,289
Chunks/Article 2.54 8.65 5.21
Char./Chunk 695.61 598.81 561.97
Questions 10,682 9,777 20,459
Questions/Article 1.69 2.00 1.83
Factual 6,522 5,196 11,718
Interpretation 4,160 4,581 8,741
3.4 Question–Answer Pair Generation
We use gpt-oss:120B (OpenAI, 2025) for gener-
ation and validate with gemini-2.5-flash (Google,
2025)asanauxiliaryconsistencycheck. Thiscross-
modelverificationreducesobvioushallucinations
andoff-topicgenerations,thoughitdoesnotguaran-
teeperfectgrounding. WeadaptourQAgeneration
approach and prompt design considerations from
prior work on large-scale dataset construction (Ra-
jpurkaretal.,2016;Yangetal.,2018;Bloometal.,
1956;AndersonandKrathwohl,2001;Zhengetal.,
2023b; Liu et al., 2023). Full prompts are in Ap-
pendix A (see Prompt A.4). Table 2 reports the
final statistics.
Interpreting corpus and QA statistics.The
statistics in Table 2 confirm that the two parts
offer complementary structural challenges. The
WebPartcontainsmoredocumentsbutwithshorter
average length, emphasizing precision in a broad
search space. TheWikipedia Partcontains fewer
but much longer documents, requiring effective
passageretrievalwithindense,sectionedtext. By
coveringboth, andmaintainingabalancedmixof
FactualandInterpretationquestions, the bench-
mark provides a robust testbed for Turkish RAG
systemsacrossthespectrumofqualityTurkishtext.
4 Methodology and Optimization
Optimizing RAG pipelines requires navigating a
vast combinatorial space of design choices. To
address this, we adopt a two-step approach: first,
wedefineacomprehensivedesignspaceofcandi-
date methods (Section 4.1); second, we employ a
budgeted genetic search (Section 4.2) to efficiently
identify high-performing configurations without
exhaustive enumeration.

4.1 RAG Design Space
WeevaluateamodularRAGpipeline(Figure1)and
varymethodswithinseventechniquefamilieswhile
holding constant the rest of the system (chunking
policy,indexconfiguration,andpromptstructure)
to isolate which design choices drive performance.
Figure 1: Overview of the RAG design space.
Selection rationale.Because the RAG literature
is rapidly expanding, we focus on a representa-
tive set of techniques that cover the most commoncontrolpointsinend-to-endRAG(query-side,re-
trieval/reranking, context selection/compression,
and generation-time refinement), are widely used
orfrequentlyreportedaseffective,andcanbeim-
plemented reproducibly as modular components
underaconsistentevaluationprotocol. Aconsistent
protocol is vital for emerging ecosystems like Turk-
ish (Bikmaz et al., 2025; Simsek, 2025; Ezerceli
et al., 2025; Taş et al., 2025; Hakkani-Tür et al.,
2002; Ezerceli et al., 2025; Umutlu et al., 2025),
whererigorousstandardsareoftenlacking(Umutlu
et al., 2025). For techniques that require prompts
(e.g., rewriting/decomposition, HyDE-style genera-
tion, reflection/revision, LLM reranking), we use
the default prompts and hyperparameters recom-
mended in the corresponding papers.
Querytransformation.Multi-QueryRetrieval:
Generate multiple semantically diverse rewrites of
the user query and retrieve for each, then merge re-
sults to improve recall (Rackauckas, 2024).Query
Decomposition:Break a complex question into
simpler sub-questions, retrieve for each part, and
aggregate evidence before answering (Zheng et al.,
2023a).Step-back Prompting:Ask a more gen-
eral “step-back” question to retrieve high-level
background context that helps answer the origi-
nal query (Zheng et al., 2023a).HyDE:Synthesize
ahypotheticalanswerdocumentfromthequeryand
useitasaretrievalquerytobettermatchrelevant
passages(Gaoetal.,2022).QueryRewriting/LLM-
based expansion:Rewrite and/or expand the query
withadditionalkeywordsandparaphrasestoreduce
lexicalmismatchandimproveretrieval(Wangetal.,
2023; Mao et al., 2021).
Candidatere-ranking.Cross-encoderreranking
(BERT):Score each (query, passage) pair jointly
with a cross-encoder and reorder retrieved pas-
sages by predicted relevance (Nogueira and Cho,
2019).LLM-basedreranking:Useaninstruction-
tuned LLM to judge relevance of candidates and
reorder/retain the most useful passages for answer-
ing (Lewis et al., 2020).
Candidate filtering & selection.Top- Ktrunca-
tion:Keep only the Khighest-ranked retrieved
passages to control context budget and reduce
noise(Karpukhinetal.,2020;Lewisetal.,2020).
Similarity thresholding:Discard candidates below
a similarity cutoff to avoid injecting weakly related
context into generation (Karpukhin et al., 2020;
Lewis et al., 2020).

Context augmentation.Prev/next chunk aug-
mentation (multi-granular context):Expand a re-
trieved chunk with its neighbors or multi-scale
spans to recover lost local coherence from seg-
mentation (Liu et al., 2025b).Relevant segment
extraction:Extract only the most relevant spans
within retrieved documents to maximize signal per
token in the context window (Liu et al., 2025b).
Context condensation.Prompt/context compres-
sion & redundancy pruning:Compress passages
and remove redundant content so more unique evi-
dence fits within the model’s context length (Jiang
etal.,2023a;Lietal.,2023).Tree-stylesummarize/
iterativerefine:Summarizeevidencehierarchically
or refine a running summary over multiple steps
topreservekeyfactsundertightbudgets(Madaan
et al., 2023).
Promptcomposition.Naiveconcatenation(RAG
baseline):Concatenatetheselectedpassagesintoa
singlecontextblockandpromptthemodeltoanswer
groundedinthatcontext(Lewisetal.,2020).Long-
context ordering (“lost in the middle”):Vary the
ordering/placement of evidence in long prompts to
studypositioneffectswheremid-contextfactsare
under-attended (Liu et al., 2024).
Answer refinement.Reflection/Revise:Gener-
ateaninitialanswer,critiqueitagainsttheretrieved
evidence, and revise to fix omissions or inconsis-
tencies (Shinn et al., 2023; Madaan et al., 2023).
4.2 Genetic Algorithm to Determine Effective
Combinations
The modular RAG design space is highly combi-
natorial, yielding ∼1,296 possible pipelines in our
setting—too many for exhaustive evaluation. To
efficiently navigate this space, we employ a con-
strained genetic algorithm (GA) inspired by Kartal
et al. (2025). GAs are well-suited for combinato-
rial optimization (Holland, 1975; Goldberg, 1989),
allowing us to identify near-optimal solutions by
evaluating only ∼200 pipelines (approx. 15% of
thesearchspace). Wesupplementthisautomated
search with manual evaluation of established base-
lines. The GA evolves a population of pipelines
by (i) evaluating them on a small query set, (ii)
selectingtopperformers,and(iii)generatingnew
candidates via crossover and mutation, enforcing
compatibility constraints.Genomeencoding.WeencodeaRAGpipeline
as a discrete genome:
g= (m 1, m2, . . . , m F)
where Fis the number of technique families. Each
genemf∈ M fselects exactly one method from
family f(optionally including a Nonechoice to
disable that slot).
Fitnessfunction.Weoptimizeacompositeobjec-
tivethatbalancesretrievaleffectivenessandanswer
qualityonanevaluationsubset Q. LetR(g)denote
a retrieval metric (e.g., nDCG@ kor Recall@ k)
andG(g)a generation metric (e.g., judged faith-
fulness/accuracy). We compute both metrics on Q
and normalize them to comparable scales (denoted
bye·). The fitness of genomegis:
Fit(g) =α· eR(g) + (1−α)· eG(g).(1)
Weset α= 0.5toweightretrievalandgeneration
equally, reflecting that strong RAG performance
requiresboth(i)retrievingrelevantevidenceand(ii)
producing a faithful and accurate response condi-
tioned on that evidence. This equal weighting also
avoidsover-specializingthesearchtowardpipelines
that optimize retrieval quality at the expense of an-
swer quality (or vice versa).
GAProcedureandevaluationbudget.Werun
theGAforasmallnumberofgenerationstoidentify
strong pipelines under a fixed evaluation budget.
Concretely, we use population size P= 20and
G= 10generations. Each candidate genome is
evaluated on a randomly sampled set of |Q|= 100
questionsperdomain. Thisyieldsatotalof P×G
candidate evaluations while keeping per-candidate
evaluationlightweight. ThisGA evaluation serves
as the primary empirical basis for the paper’s best-
practice conclusions (Section 5); the full algorithm
is detailed in Appendix B (Algorithm 1).
Reproducibility.All parameter values used in
thegeneticsearch(includingselectionandelitism
settings, crossover/mutation operators and rates,
constraintchecks,andrandomseeds)aswellasthe
fullPythonimplementationofthealgorithmusedin
our experiments are shared in the code repository.
4.3 Experimental Setup and Metrics
We select metrics to (i) capture complementary
failuremodes,(ii)alignwithcommonpracticeto
easecomparisonacrossRAGsystems,and(iii)re-
ducesensitivitytoanysinglenoisyautomaticsignal.

Concretely,forretrievalwereportamixofcoverage-
oriented metrics (Recall@5) andranking-quality
metrics(mAP,nDCG@5,MRR),sincedownstream
generation depends both on whether evidence is re-
trievedatallandonhowhighlyitisranked. Forgen-
eration, we combine an embedding-based semantic
similarity signal with an LLM-as-a-judge score
to balance paraphrase-tolerant matching with a
moreholisticassessmentofcorrectnessandanswer
quality; this choice follows the recommendation
to use multiple complementaryevaluation criteria,
includingLLM-judgestyleassessments,whenan-
alyzing benchmark and system outputs (Umutlu
et al., 2025).
4.3.1 Retrieval Metrics
Weevaluateretrievalagainsttheground-truthevi-
dencepassagesusedtogenerateeachQ&A.Fora
query qwithrelevantpassages D⋆andtheranked
listπk(q)oftop- kretrievedpassages: Wecompute
an overallretrieval scoreas an equally weighted
aggregateofRecall@5,mAP,nDCG@5,andMRR
(we report the component metrics alongside the
aggregate).
Retrieval(q) =α
Recall@5(q) + mAP(q)
+ nDCG@5(q) + MRR(q)(2)
where we set α= 0.25 to equally weight the
four complementary retrieval metrics (coverage
and ranking quality) without privileging any sin-
gle component, consistent with our equal-weight
aggregation choices elsewhere.
4.3.2 Generation Metrics
We measure end-to-end answer quality with two
complementary signals and compute an overall
generationscoreastheirequallyweightedaggregate
(we also report each component): (i)Semantic
similarity(embedding-basedsimilaritybetweenthe
model answer and the reference answer), and (ii)
LLM-Judge(an LLM-based judgment score for
answer quality/correctness). We aggregate them as
Generation(q) =αSim(q) + (1−α) Judge(q).
(3)
In our experiments we set α= 0.5to give equal
weight to semantic similarity and the judge sig-
nal, reflecting a conservative choice that avoids
over-optimizing to either an embedding proxy or
a single LLM-based evaluator (analogous to the
equal-weight aggregation used in our GA fitness
objective).Table 3: Compute and implementation details.
Component Setting
Embedding model embeddinggemma
Generator model gpt-oss:120B
Evaluator model gemini-2.5-flash
Reranker ms-marco-MiniLM-L-12-v2
Hardware M3 Ultra 80-core GPU
4.3.3 Latency and Practicality
Wereporttotaltokenusageforeachconfiguration
butdonotusethismetricwhenoptimizingwiththe
GA;wereportittotakeintoaccountthepracticality
oftheconfigurationwhensuggestingbestpractices.
Table3recordsmodelversionsandsystemsettings.
5 Best Practices
5.1 Evaluation Protocol
Our evaluation is performed within the GA pro-
cedure (Section 4.2). We sample a stratified ran-
dom subset of n= 100questions (balanced across
the Web and Wikipedia parts) from the unified
benchmark and use this set both to score candidate
genomes during the search and to report the final
performanceofthebestGA-selectedconfigurations.
Thefullbenchmarkcontains N≈20,459 grounded
QApairs;subsamplingkeepsevaluationtractable
whileenablingcontrolledcomparisonacrosscon-
figurations. Thisevaluationprocedureconstitutes
the empirical basis for our results.
Why n= 100 is sufficient.In pilot runs, we
variedthesubsetsizeandobservedthatperformance
estimates saturated around n= 100. Beyond this
point,variancereductionwasmarginalcompared
to the linear increase in evaluation cost. Thus,
n= 100serves as an efficient saturation point that
yieldsstablemeanestimatesandconsistentrelative
rankingsofcandidateconfigurations,whilekeeping
the per-candidate evaluation cost low enough for
the GA to explore many pipelines under a fixed
budget.
5.2 Overall Performance
Table 4 summarizes the performance of the top
configurations identified by the genetic search on
the unified Turkish RAG benchmark ( n= 100).
We compare the baseline against three distinct opti-
mal points found by the GA: a maximum-accuracy
configuration, a Pareto-optimal “best value” con-
figuration, and a production-friendly configuration.

Thecompleteperformanceresultsforallnotewor-
thy configurations are provided in Appendix B
(Table 6).
5.3 Component Analysis and Inferences
Our results highlight several key trade-offs in the
design space of Turkish RAG systems.
Maximizing Accuracy.If the goal is to maxi-
mize the end-to-end score, the winning recipe is
acombinationofstrongqueryexpansion(HyDE),
strong reranking (Cross-encoder Reranking), ag-
gressivecontextcompression(Tree-styleSumma-
rize), and Long-context Ordering. This configu-
ration achieves the top score of 85.00%, driven
byhighgenerationquality(0.823). However,this
comesatasignificantcost: approximately 3.7×the
tokenusageofthebaseline. HyDEandsummariza-
tion are computationally expensive, making this
approachsuitableonlywhenaccuracyisparamount
and resources are unconstrained.
The Pareto Winner.The “BestValue”configu-
ration(Cross-encoderReranking+Previous/Next
Chunk Augmentation + Long-context Ordering)
achieves 84.60% overall score, which is only 0.4
points behind the top performer, but with signifi-
cantlyloweroverhead. Comparedtothebaseline,
it yields a +5.9 point improvement for ∼2×to-
kens. This represents the cleanest recommendation
for practical applications: prioritize Cross-encoder
Reranking and local context enrichment (Previ-
ous/Next Chunk Augmentation) before adopting
expensive methods like HyDE.
Lightweight Best Option (Production-Friendly).
The “Production-Friendly” configuration (Query
Clarification + Cross-encoder Reranking + Previ-
ous/NextChunkAugmentation)achieves80.20%
overall score, with Retrieval 0.901 and Generation
0.704, consuming ∼1,738 tokens ( ≈1.74×base-
line tokens). A short clarification / rewrite step
plus cross-encoderreranking andadjacent-context
augmentation delivers a strong accuracy gain at
modest cost. This likely helps Turkish by reduc-
ingmorphology-drivenambiguityandimproving
matching for entity/surface-form variation, while
avoiding extra overhead from context reordering.
HyDEandLLMRerankingCosts.Wefindthat
HyDE is not a “free lunch” on Turkish datasets.
While it helps bridge semantic gaps, it often bal-
loons costs and can underperform if it introducesnoise. For instance, HyDE combined with Tree-
style Summarize led to high computational over-
head. Similarly, LLM-based Reranking was out-
performed by Cross-encoder Reranking in terms of
score-per-cost. Cross-encoder Reranking proved to
beastrong,reliabledefault, whereasLLM-based
Rerankingshouldbereservedfornichereasoning-
heavy cases.
RisksofOver-Stacking.StackingmultipleLLM-
based modules often degrades efficiency without
guaranteeing better performance. A maximalist
pipeline combining six complex modules achieved
only 79.60% accuracy, underperforming the sim-
pler“HighAccuracy”configuration(85.00%)while
consuming heavy token usage. Excessive LLM
post-processingin Turkish may distort morphologi-
calcuesoraccumulateerrors. Fromtheseresults,
wecanrecommendaddingatmostoneLLM-heavy
stage (e.g., summarization or reflection) only when
necessary, avoiding “stacking everything” without
proven gain.
Over-Filtering Problems.Strict filtering com-
bined with segment extraction can severely harm
retrievalinTurkish. Apipelineusingstrictthresh-
olds and segment extraction dropped to 78.30%
overallscore(Retrieval0.755),wellbelowbaseline.
Morphology and paraphrase variance in noisy web
data make hard thresholds risky, causing the sys-
tem to miss evidence. We recommend preferring
reranking over strict filtering unless thresholds are
carefully calibrated per domain.
5.4 Recommendations
Based on these findings, we propose the following
best practices for Turkish RAG:
•Recommended Default:UseCross-encoder
Reranking + Previous/Next Chunk Augmenta-
tion+Long-contextOrdering. Thispipeline
offers a strong balance of accuracy and effi-
ciency andshould serve asthe standardbase-
line for Turkish RAG experiments.
•High-Accuracy:For leaderboards or appli-
cations where score is critical, useHyDE +
Cross-encoderReranking+Tree-styleSumma-
rize+Long-contextOrdering. Thisrequiresa
higher budget for latency and tokens.
•Production-Friendly:For latency-constrained
applications,useQueryClarification+Cross-
encoder Reranking + Previous/Next Chunk

Table 4: Performance of GA-selected configurations on the unified benchmark ( n= 100). We report Overall Score,
Retrieval Score, Generation Score, and estimated Token usage per query. The “High Accuracy” model achieves the
best scores but at high cost, while the “Pareto Optimal” model offers the best balance.
ConfigurationOverall
ScoreRetrieval
ScoreGeneration
ScoreTokens
(est.)
High Accuracy(HyDE + Cross-encoder Reranking + Tree-style
Summarize + Long-context Ordering)85.00% 0.876 0.823∼3664
Pareto Optimal(Cross-encoder Reranking + Previous/Next Chunk
Augmentation + Long-context Ordering)84.60% 0.870 0.823∼1987
Production-Friendly(Query Clarification + Cross-encoder Reranking
+ Previous/Next Chunk Augmentation)80.20% 0.901 0.704∼1738
Baseline(Dense Retrieval + Similarity Thresholding + Naive
Concatenation)78.70% 0.872 0.702∼1000
Augmentation. Thisprovidesmeaningfulim-
provementsovernaiveRAGwhilemaintaining
fast response times and low token usage.
These recommendations are specific to Turkish
and were derived under our experimental setup.
Whileothermorphologicallyrichlanguagessuch
as Finnish, Hungarian, and Korean face similar
challenges(e.g.,morphologicalrichness,agglutina-
tion, surface-formvariation)(Tsarfatyet al.,2014;
Gerzetal.,2018),wedonotclaimthatourfindings
transferdirectly;validatinggeneralizabilitytoother
languages requires dedicated experiments.
6 Conclusion and Future Work
Wepresentedanend-to-end,domain-awarestudy
of Turkish RAG across informal web text and
Turkish Wikipedia. By benchmarking modular
choices across the RAG pipeline, we distill prac-
tical,domain-specificconfigurationguidanceand
provide resources to support reproducible Turkish-
RAG research.
Future Work.We plan to: (i) explore hybrid
RAGwithinafamilyinsteadofusingsinglemethod
from each, (ii) incorporate graph structure for re-
trievalandqueryexpansion(entitygraphs,hyper-
linkgraphs),(iii)scaletolargerandmorediverse
Turkish corpora (news, technical documentation,
legaltext),(iv)studyTurkishmorphology-awarere-
trieval features (e.g., lemma-aware sparse retrieval,
morphological analyzers), (v) incorporate more
noisy,real-world-likedata andunanswerableques-
tions to better assess system robustness (moving
beyond our current evaluation on relatively clean
questions), and (vi) examine document-side, index-
timemethods—includingpre-embeddingsandre-
latedpre-computation/cachingtechniques. Wealso
plancontrolledtechnique-familyablationsperdo-
main to quantify marginal gains and interactions
(retrieval vs. generation) under a fixed evaluationand budget. Additionally, we plan to conduct a
structured error analysis of end-to-end outputs, fo-
cusing on Turkish-specific failure modes such as
inflection-driven mismatch, over-normalization of
informal language, entity drift, and missing evi-
dence in multi-passage questions. A systematic
taxonomyandannotatederrorsetwillhelpseparate
retrievalversusgenerationerrorsandguidetargeted
improvements.
7Limitations and Ethical Considerations
Limitations.All recommendations in this pa-
per reflect our specific experimental setup (mod-
els, prompts, tokenization, corpus preprocessing,
hardware, and context limits). In practice, the
best-performing settings can shift across Turkish
corpora due to differences in domain, document
length distribution, content quality, noise/boiler-
plate, and latency constraints. We therefore po-
sition our findings asbest practices for generic
Turkishtextretrieval,andencouragepractitioners
tore-runasmallsweepontheirowndatatoiden-
tify the best point on the quality–latency tradeoff.
Whilethedatasetreflectsrealisticwebcontent,itis
cleaner than typical production RAG pipelines and
does not cover specialized domains such as legal
or technical documentation; extending the bench-
mark to such domains is left for future work. As
a defense-industry organization, we cannot release
or fully describe some proprietary data sources
used duringdevelopment (e.g., internal enterprise
documents); therefore, thispaper andthe released
resources focus on openly available data.
Reproducibility and releases.We release the
evaluationdatasets,thefullQAsetwithevidence
spans, all RAG configuration files, and scripts to
reproducemetricsinthecoderepository. Allrec-
ommendations in this paper reflect our specific
experimental setup (models, prompts, tokenization,

corpus preprocessing, hardware, and context lim-
its). In practice, the best-performing settings can
shift across Turkish corpora due to differences in
domain (formal vs. informal), document length dis-
tribution, noise/boilerplate, and latency constraints.
Wethereforepositionourfindingsasbestpractices
for generic Turkish text retrieval, and encourage
practitioners to re-run on their own data to identify
the best point on the quality–latency tradeoff.
Ethics.Informalwebdatacancontainsensitiveor
personalcontent;werecommendcarefulfiltering,
redaction,andlicense-awarerelease. Becausethe
data are collected from the public internet, they
may reflect societal biases and other problematic
content;anysuchcontentisincludedforresearch
purposes only and does not reflect the authors’
opinions or endorsements. LLM-based filtering
can itself introduce bias; we therefore document
filteringcriteriaandprovideauditsampleswhere
feasible.
DataProvenanceandCopyright.TheWebPart
of our dataset is derived from CulturaX (Nguyen
etal.,2024),apubliclyavailablemultilingualcor-
pus; we do not perform independent scraping of
websites. Source domains (e.g., haberler.com,
sikayetvar.com)wereincludedinCulturaXdueto
theirtopicaldiversity andpublicaccessibility. We
releaseonlyderivedannotations—question–answer
pairs,topiclabels,andchunkboundaries—rather
than redistributing full original articles. This ap-
proachalignswithstandardresearchpracticesfor
web-derived corpora and respects the original data
providers.
UseofGenerativeAI.GenerativeAIwasused
solelytoassistwithlanguageediting. Allscientific
contributions,dataconstruction,analysis,andinter-
pretations presented in this work are original and
were conducted entirely by the authors.
Acknowledgments
We gratefully acknowledge support from Roketsan
Inc. and the Google Gemini Academic Reward
Program,whichhelpedenabletheexperimentsand
computing resources used in this study.
References
AlfredV.AhoandJeffreyD.Ullman.1972.TheTheory
of Parsing, Translation and Compiling, volume 1.
Prentice-Hall, Englewood Cliffs, NJ.Lorin W. Anderson and David R. Krathwohl, editors.
2001.A Taxonomy for Learning, Teaching, and
Assessing: A Revision of Bloom’s Taxonomy of Edu-
cational Objectives. Longman, New York, NY.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
HannanehHajishirzi.2024. Self-RAG:Learningto
retrieve,generate,andcritiquethroughself-reflection.
InInternational Conference on Learning Representa-
tions (ICLR). Oral.
ErdoğanBikmaz,MohammedBriman,andSerdarAr-
slan.2025. BridgingthelanguagegapinRAG:Acase
studyonTurkishretrievalandgeneration.Researcher,
5(1):38–49.
BenjaminS.Bloom,MaxD.Engelhart,EdwardJ.Furst,
WalkerH.Hill,andDavidR.Krathwohl.1956.Taxon-
omyofEducationalObjectives: TheClassificationof
EducationalGoals.HandbookI:CognitiveDomain.
David McKay Company, Inc., New York, NY.
ÇağrıÇöltekin,A.SezaDoğruöz,andÖzlemÇetinoğlu.
2023. Resources for Turkish natural language pro-
cessing: Acritical survey.LanguageResourcesand
Evaluation, 57:449–488.
DarrenEdge,HaTrinh,NewmanCheng,JoshuaBradley,
Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky,RobertOsazuwaNess,andJonathan
Larson. 2024. From local to global: A graph RAG
approach to query-focused summarization. arXiv
preprint.Preprint, arXiv:2404.16130.
Yanai Elazar, Akshita Bhagia, Ian Magnusson, Abhi-
lasha Ravichander, Dustin Schwenk, Alane Suhr,
Pete Walsh, Dirk Groeneveld, Luca Soldaini, Sameer
Singh, Hanna Hajishirzi, Noah A. Smith, and
Jesse Dodge. 2023. What’s in my big data?
arXiv:2310.20707.
ShahulEs,JithinJames,LuisEspinosa-Anke,andSteven
Schockaert. 2024. RAGAs: Automated evaluation
of retrieval augmented generation. InProceedings
of the 18th Conference of the European Chapter of
theAssociationforComputationalLinguistics: Sys-
tem Demonstrations. Association for Computational
Linguistics.
Özay Ezerceli, Gizem Gümüşçekiçci, Tuğba Erkoç, and
Berke Özenç.2025. TurkEmbed4Retrieval: Turkish
embedding model for retrieval task. arXiv preprint.
Preprint, arXiv:2511.07595.
LeoGao, Stella Biderman, SidBlack, Laurence Gold-
ing, Travis Hoppe, Charles Foster, Jason Phang,
Horace He, Anish Thite, Noa Nabeshima, Shawn
Presser, and Connor Leahy. 2020. The Pile: An
800gbdatasetofdiversetextforlanguagemodeling.
arXiv:2101.00027.
LuyuGao,XueguangMa,JimmyLin,andJamieCallan.
2022. Precise zero-shot dense retrieval without rele-
vance labels.arXiv preprint arXiv:2212.10496.

Daniela Gerz, Ivan Vulić, Edoardo Maria Ponti, Roi
Reichart, and Anna Korhonen. 2018. On the rela-
tion between linguistic typology and (limitations of)
multilinguallanguagemodeling. InProceedingsof
the2018ConferenceonEmpiricalMethodsinNatu-
ralLanguageProcessing,pages316–327,Brussels,
Belgium. Association for Computational Linguistics.
DavidE.Goldberg.1989.GeneticAlgorithmsinSearch,
Optimization, and Machine Learning. Addison-
Wesley, Reading, MA.
Google. 2025. Gemini api: Model information. Ac-
cessed: 2025-12-24.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (RAG): Evolution, current
landscape and future directions. arXiv preprint.
Preprint, arXiv:2410.12837.
DilekZHakkani-Tür,KemalOflazer,andGökhanTür.
2002. Statisticalmorphologicaldisambiguationfor
agglutinative languages.Computers and the Humani-
ties, 36(4):381–410.
John H. Holland. 1975.Adaptation in Natural and
ArtificialSystems. UniversityofMichiganPress,Ann
Arbor, MI.
Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing
Yang,andLiliQiu.2023a. LLMLingua: Compress-
ing prompts for accelerated inference of large lan-
guage models.arXiv preprint arXiv:2310.05736.
ZhengbaoJiang,FrankF.Xu,LuyuGao,ZhiqingSun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023b. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guageProcessing,pages7969–7992,Singapore.As-
sociation for Computational Linguistics.
Vladimir Karpukhin, BarlasOguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen-tau Yih. 2020. Dense passage retrieval
foropen-domainquestionanswering. InProceedings
of EMNLP.
Muhammed Yusuf Kartal, Suha Kagan Köse, Korhan
Sevinç,andBurakAktas.2025. RAGSmith: Aframe-
work for finding the optimal composition of retrieval-
augmentedgenerationmethodsacrossdatasets.arXiv
preprint arXiv:2511.01386.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari,
Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan,SaifulHaq,AshutoshSharma,ThomasT.
Joshi, Hanna Moazam, Heather Miller, Matei Za-
haria,andChristopherPotts.2024. DSPy: Compiling
declarative language model calls into state-of-the-art
pipelines. InThe Twelfth International Conference
on Learning Representations (ICLR 2024).Omar Khattab and Matei Zaharia. 2020. ColBERT:
Efficient and effective passage search via contextu-
alized late interaction over BERT. InProceedings
of the 43rd International ACM SIGIR Conference on
Research and Development in Information Retrieval
(SIGIR ’20), pages 39–48. ACM.
Dongkyu Kim, Byoungwook Kim, Donggeon Han,
and Matouš Eibich. 2024. AutoRAG: Automated
framework for optimization of retrieval augmented
generation pipeline. arXiv preprint.Preprint,
arXiv:2410.20878.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni,VladimirKarpukhin,NamanGoyal,Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-
augmentedgenerationforknowledge-intensiveNLP
tasks.arXiv preprint arXiv:2005.11401.
Minghan Li, Xinxuan Lv, Junjie Zou, Tongna Chen,
Chao Zhang, SuchaoAn, Ercong Nie, and Guodong
Zhou.2025. Queryexpansionintheageofpre-trained
and large language models: A comprehensive survey.
arXiv preprint.Preprint, arXiv:2509.07794.
YuchengLi,BoDong,ChenghuaLin,andFrankGuerin.
2023. Compressing context to enhance inference
efficiency of large language models. InProceedings
of EMNLP.
Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-
HongYang,RonakPradeep,andRodrigoNogueira.
2021. Pyserini: A Python toolkit for reproducible
informationretrieval researchwithsparseand dense
representations. InProceedings of the 44th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’21),
pages 2356–2362, Virtual Event, Canada. ACM.
Charles Z. Liu, Imani Abayakoon, and Farookh
KhadeerHussain.2025a. Retrieval-augmentedgen-
eration: Asurveyofmethodologies,techniques,ap-
plications, and future directions. Preprint.
NelsonF.Liu,KevinLin,JohnHewitt,AshwinParan-
jape,MicheleBevilacqua,FabioPetroni,andPercy
Liang.2024. Lostinthemiddle: Howlanguagemod-
elsuselongcontexts.TransactionsoftheAssociation
for Computational Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using GPT-4 with better human
alignment. InProceedings of the 2023 Conference
on Empirical Methods in Natural Language Process-
ing, pages 2511–2522, Singapore. Association for
Computational Linguistics.
Zuhong Liu, Charles-Elie Simon, and Fabien Cas-
pani. 2025b. Passage segmentation of documents
for extractive question answering.arXiv preprint
arXiv:2501.09940.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,

Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdan-
bakhsh, and Peter Clark. 2023. Self-refine: Itera-
tive refinement with self-feedback.arXiv preprint
arXiv:2303.17651.
Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong
Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen.
2021. Generation-augmented retrieval for open-
domain question answering. InProceedings of ACL.
Thuat Nguyen, Chien Van Nguyen, Viet Dac Lai, Hieu
Man,NghiaTrungNgo,FranckDernoncourt,RyanA.
Rossi, and Thien Huu Nguyen. 2024. CulturaX: A
cleaned, enormous, and multilingual dataset for large
languagemodelsin167languages. InProceedingsof
the2024JointInternationalConferenceonCompu-
tationalLinguistics,LanguageResourcesandEval-
uation (LREC-COLING 2024), pages 4226–4237,
Torino, Italia. ELRA and ICCL.
Rodrigo Nogueira and Kyunghyun Cho. 2019. Pas-
sage re-ranking with BERT.arXiv preprint
arXiv:1901.04085.
Kemal Oflazer. 2014. Turkish and its challenges for
language processing.Language resources and evalu-
ation, 48(4):639–653.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.
Preprint, arXiv:2508.10925. https://openai.
com/research/gpt-oss-model-card/.
Guilherme Penedo, Hynek Kydlíček, Loubna Ben Allal,
Anton Lozhkov, Margaret Mitchell, Colin Raffel,
Leandro Von Werra, and Thomas Wolf. 2024. The
FineWeb datasets: Decanting the web for the finest
text data at scale. arXiv:2406.17557.
Zackary Rackauckas. 2024. RAG-fusion: a new take
on retrieval-augmented generation.arXiv preprint
arXiv:2402.03367.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. SQuAD: 100,000+ questions for
machine comprehension of text. InProceedings of
the2016ConferenceonEmpiricalMethodsinNatu-
ral Language Processing, pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.
NoahShinn,FedericoCassano,EdwardBerman,Ash-
win Gopinath, Karthik Narasimhan, and Shunyu Yao.
2023. Reflexion: Languageagentswithverbalrein-
forcementlearning.arXivpreprintarXiv:2303.11366.
NeurIPS 2023.
MuratSimsek.2025. Retrieval-augmentedgeneration
versus fine-tuning for Turkish cultural question an-
swering: A comprehensive evaluation and analysis.
Research Square preprint.
Luca Soldaini and 1 others. 2024. Dolma: an open
corpus of three trillion tokens for language model
pretraining research. InProceedings of the 62nd
AnnualMeetingoftheAssociationforComputational
Linguistics (ACL).WenyuTao,XiaofenXing,YirongChen,LinyiHuang,
andXiangminXu.2025. TreeRAG:Unleashingthe
power of hierarchical storage for enhanced knowl-
edge retrieval in long documents. InFindings of
the Association for Computational Linguistics: ACL
2025,pages356–371,Vienna,Austria.Association
for Computational Linguistics.
Selva Taş, Mahmut El Huseyni, Özay Ezerceli, Reyhan
Bayraktar, and Fatma Betül Terzioğlu. 2025. Turk-
lettucedetect: A hallucination detection models for
Turkish RAG applications. arXiv preprint.Preprint,
arXiv:2509.17671.
ÇağrıToraman,AhmetKaanSever, AyseAysuCengiz,
ElifEcemArslan,GörkemSevinç,MeteMertBirdal,
YusufFarukGüldemir,AliBuğraKanburoğlu,Sezen
Felekoğlu, Osman Gürlek, Sarp Kantar, Birsen Şahin
Kütük, Büşra Tufan, Elif Genç, Serkan Coşkun,
GupseEkinDemir,MuhammedEminArayıcı,Olgun
Dursun,OnurGungor,and3others.2026. Turkbench:
A benchmark for evaluating turkish large language
models.arXiv preprint arXiv:2601.07020.
ReutTsarfaty,DjaméSeddah,YoavGoldberg,Sandra
Kübler,MarieCandito,JenniferFoster,YannickVers-
ley, Ines Rehbein, and Lamia Tounsi. 2014. SPMRL-
SANCL2014sharedtaskonparsingmorphologically
richlanguages. InProceedingsoftheFirstJointWork-
shoponStatisticalParsingofMorphologicallyRich
Languages and Syntactic Analysis of Non-Canonical
Languages,pages103–109,Dublin,Ireland.Dublin
City University.
Elif Ecem Umutlu, Ayse Aysu Cengiz, Ahmet Kaan
Sever, Seyma Erdem, Burak Aytan, Busra Tufan,
AbdullahTopraksoy,EsraDarıcı,andCagriToraman.
2025. Evaluatingthequalityofbenchmarkdatasets
for low-resource languages: A case study on Turkish.
InProceedingsoftheFourthWorkshoponGeneration,
Evaluation and Metrics (GEM ²), pages 471–487,
Vienna, Austria and virtual meeting. Association for
Computational Linguistics.
LiangWang,NanYang,andFuruWei.2023. Query2doc:
Query expansion with large language models. In
Proceedings of EMNLP.
MauriceWeber, DanielFu, QuentinAnthony,Yonatan
Oren,ShaneAdams,AntonAlexandrov,Xiaozhong
Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams,
Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen,
Max Ryabinin, Tri Dao, Percy Liang, Christopher
Ré, Irina Rish, and Ce Zhang. 2024. RedPajama:
an open dataset for training large language models.
arXiv:2411.12372.
GuillaumeWenzek,Marie-AnneLachaux,AlexisCon-
neau, Vishrav Chaudhary, Francisco Guzmán, Ar-
mand Joulin, and Edouard Grave. 2020. CCNet:
Extracting high quality monolingual datasets from
web crawl data. InProceedings of the Twelfth Lan-
guageResourcesandEvaluationConference,pages
4003–4012.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
ChristopherD.Manning.2018. HotpotQA:Adataset
fordiverse,explainablemulti-hopquestionanswering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380,Brussels,Belgium.AssociationforCom-
putational Linguistics.
Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu,
and Zhaofeng Liu. 2024. Evaluation of retrieval-
augmented generation: A survey. arXiv preprint.
Preprint, arXiv:2405.07437.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024.
Retrieval-augmented generation for AI-generated
content: A survey. arXiv preprint.Preprint,
arXiv:2402.19473.
Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen,
Heng-Tze Cheng, Ed H. Chi, Quoc V. Le, and Denny
Zhou.2023a. Takeastepback: Evokingreasoningvia
abstraction in large language models.arXiv preprint
arXiv:2310.06117. ICLR 2024.
LianminZheng,Wei-LinChiang,YingSheng,Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang,
Joseph E. Gonzalez, and Ion Stoica. 2023b. Judging
LLM-as-a-judgewithMT-benchandchatbotarena.
arXiv preprint.Preprint, arXiv:2306.05685.

A Prompts and Validation Rubrics
A.1 URL-only Filtering Prompt
URL Filtering
Analyze the following website (base URL) and determine its eligibility:
Website: {url}
Evaluate:
1. Does this website likely contain valuable information (educational,
informative, useful content)?,→
2. Is the content on this website likely written in proper language (casual,
conversational)?,→
Based on your analysis of the website domain and typical content, provide:
- Status: "ELIGIBLE" if BOTH conditions are true, otherwise "NOT ELIGIBLE"
- Reason: Brief explanation (1-2 sentences)
Response format:
Status: [ELIGIBLE/NOT ELIGIBLE]
Reason: [Your explanation]
A.2 Content Filtering Prompt
Content Filtering
You are a data quality and style evaluator. You will be given TURKISH text taken
from a web page, along with the URL it came from. ,→
TASK 1 -- EVALUATION
Evaluate whether the text is:
- suitable for a RAG system,
- understandable,
- and "CLEAN" (everyday language; not nonsense/trash).
Definitions:
1) Informality level:
- "clean": Everyday language (blog/forum/social media), but:
*understandable
*slightly relaxed yet still structured
*similar to news-site tone
*sentences mostly well-formed
*no heavy slang, no spam
- "nonsense_or_spam": incoherent, random words, bot/spam, only links/hashtags,
etc.,→
2) Quality:
- "good": clear, coherent, not full of spelling errors, topic is followable,
usable for RAG,→
- "bad": too short, messy, major spelling/spam issues, topic not followable
3) ELIGIBLE criteria:
- mostly Turkish
- "Clean"
- Quality must be "good"
- definitely NOT "formal"
- definitely NOT "nonsense_or_spam"
- text length > 100 characters
- mostly about a single topic/theme
IMPORTANT: First do the evaluation and determine Status.
TASK 2 -- MARKDOWN CONVERSION (ONLY IF ELIGIBLE)
WARNING: Do this step ONLY if Status: ELIGIBLE. If NOT ELIGIBLE, do NOT convert
to markdown.,→

If Status: ELIGIBLE:
1) Detect headings and use markdown headings (#)
2) Split paragraphs
3) Remove unnecessary whitespace
4) Do not change content beyond that; do not add new content
OUTPUT FORMAT:
Status: [ELIGIBLE/NOT ELIGIBLE]
Reason: [Short explanation]
---MARKDOWN_START---
[ONLY if Status: ELIGIBLE, put the markdown-converted text here]
[If Status: NOT ELIGIBLE, leave this section COMPLETELY EMPTY]
---MARKDOWN_END---
URL: {url}
Text: {text}
A.3 Topic Classification Prompt
Topic Classification
You are labeling Turkish text for an LLM dataset. You must not use or infer any
"source type" (Wikipedia vs web) in your decision. Treat every document the
same.,→
,→
You will be given one document: title (optional), url (optional), and text (may
be truncated).,→
Task:
1) Assign exactly ONE topic category: topic_l1
2) Assign exactly ONE safety category: safety_label
Allowed values:
topic_l1 (choose exactly one):
- STEM
- Humanities
- Social_Sciences
- Professional_Applied
- Culture_Entertainment
- Everyday_Life
- Geography_Places
- Biography_People
- Organizations_Institutions
- Events_History
- Meta_Content
safety_label (choose exactly one):
- Safe
- Needs_Filtering
- Exclude
Safety guidelines:
- Safe: ordinary content with no clear policy risks.
- Needs_Filtering: contains potentially sensitive/age-restricted/controversial
material or advisory content (e.g., medical or financial advice, explicit
profanity/hate slurs contextually used, graphic descriptions) but not
clearly disallowed.,→
,→
,→
- Exclude: clearly disallowed or high-risk content, such as explicit
instructions for wrongdoing (e.g., making weapons, fraud), explicit sexual
content involving minors, actionable self-harm instructions, doxxing/PII,
extremist recruitment/praise, or pervasive hate/harassment.,→
,→
,→
Output rules:
- Output JSON only. No markdown. No extra keys.
- Keep rationale <= 200 characters, grounded only in the given text.
JSON format:

{
"topic_l1": "...",
"safety_label": "...",
"rationale": "..."
}
Now label this document:
TITLE: {{title}}
URL: {{url}}
TEXT: {{text}}
A.4 QA Generation Prompts
We employ two types of prompts for question generation depending on the context: single-chunk and
multi-chunk generation.
A.4.1 Single Chunk Generation
Single Chunk Generation
Generate exactly {num_questions} question-answer pair(s) that can be answered
from this text chunk:,→
Chunk ID: {chunk_id}{context_info}
Text:
{chunk_content}
Each question must be categorized into one of these two categories:
1.**FACTUAL **: Questions that test direct recall of specific details. The
answer is a specific name, date, number, or short verbatim phrase found
directly in the text.,→
,→
2.**INTERPRETATION **: Questions that test comprehension by asking for
explanations of causes, effects, or relationships between concepts in the
text. The answer requires synthesizing information rather than just quoting
it.,→
,→
,→
Requirements:
- Only ask about information explicitly stated in this text
- Make questions specific and factual
- Each question should be answerable from this chunk alone
- Provide complete, accurate answers based solely on the chunk content
- Categorize each question appropriately based on the type of cognitive task
required,→
- Return valid JSON with the specified structure
- Do NOT use markdown code blocks (like)
- Return ONLY the JSON object, no other text
A.4.2 Multi-Chunk Generation
Multi-Chunk Generation
Generate exactly {num_questions} question-answer pair(s) that require
information from multiple chunks below. ,→
These chunks are related. Generate questions that:
1. Require information from at least 2 of the provided chunks
2. Are about connections, relationships, comparisons, or broader concepts across
chunks,→
3. Cannot be answered from any single chunk alone{context_info}
Each question must be categorized into one of these two categories:

1.**FACTUAL **: Questions that test direct recall of specific details. The
answer is a specific name, date, number, or short verbatim phrase found
directly in the text.,→
,→
2.**INTERPRETATION **: Questions that test comprehension by asking for
explanations of causes, effects, or relationships between concepts in the
text. The answer requires synthesizing information rather than just quoting
it.,→
,→
,→
Chunks:
{chunks_text}
Requirements:
- Focus on relationships and connections between the chunks
- Make questions that require synthesis of information
- Provide complete answers that synthesize information from multiple chunks
- Categorize each question appropriately based on the type of cognitive task
required,→
- Return valid JSON with chunk IDs {chunk_ids} in related_chunk_ids
- Do NOT use markdown code blocks (like)
- Return ONLY the JSON object, no other text
A.5 QA Validation Rubric
QA Validation System Prompt
You evaluate question-answer pairs for accuracy.
Check if:
- Question is clear
- Answer is accurate based on provided text chunks
- Answer fully addresses the question
- Chunks contain all necessary information
Return JSON: {"is_correct": boolean, "reason": "brief explanation"}
Keep reason concise (max 50 words). Return ONLY valid JSON.
B Full Experimental Results and Algorithms
Algorithm 1Genetic search over modular RAG pipelines
Require:Families{M f}F
f=1, population sizeP, generationsG, mutation rateµ, evaluation setQ
1:Initialize populationP 0={g i}P
i=1by sampling valid genomes
2:fort= 1toGdo
3:EvaluateFit(g)for allg∈ P t−1onQ
4:Select elitesEand parentsS(e.g., tournament selection)
5:Create offspring via crossover over genomes inS
6:Mutate genes with probabilityµ
7:FormP t← E ∪ O
8:end for
9:returnbest genomes fromP G

Table 5: Top base domains by document frequency with cumulative coverage (Web Part).
Domain Docs Cumulative %
sikayetvar.com 227 3.6
haberler.com 158 6.1
posta.com.tr 134 8.2
mynet.com 132 10.3
donanimhaber.com 108 12.0
webtekno.com 100 13.6
onedio.com 98 15.2
sondakika.com 98 16.7
fanatik.com.tr 91 18.2
haberaktuel.com 89 19.6
Table 6: Complete performance results fornoteworthyevaluated RAG configurations.
RAG Methods Combination Overall Score Retrieval Generation Total Token Usage
hyde + ce_rerank + tree_summarize + long_context_reorder 85.00% 0.876 0.823 3,663.8
hyde + ce_rerank + llm_summarize + reflection_revising 84.90% 0.876 0.822 3,118.4
hyde + ce_rerank + tree_summarize + reflection_revising 84.80% 0.876 0.819 3,966.2
ce_rerank + adjacent_augmenter + long_context_reorder 84.60% 0.87 0.823 1,987.2
hyde + tree_summarize 84.50% 0.892 0.798 5,260.4
hyde + ce_rerank + tree_summarize + long_context_reorder + reflection_revising 84.50% 0.876 0.814 3,964.1
hyde + ce_rerank + adjacent_augmenter + tree_summarize + long_context_reorder 84.40% 0.876 0.812 4,906.3
ce_rerank + adjacent_augmenter + long_context_reorder 84.40% 0.865 0.822 2,036.8
hyde + tree_summarize + long_context_reorder 84.30% 0.892 0.794 5,276.3
ce_rerank + adjacent_augmenter + llm_summarize + long_context_reorder 83.40% 0.87 0.799 2,703.2
hyde + long_context_reorder + reflection_revising 83.10% 0.896 0.765 2,339.3
hyde + llm_rerank + tree_summarize 83.10% 0.868 0.795 4,295.2
hyde + adjacent_augmenter + long_context_reorder 82.90% 0.896 0.761 3,138.8
adjacent_augmenter + long_context_reorder 82.70% 0.896 0.758 2,147.0
llm_rerank + adjacent_augmenter + llm_summarize 82.70% 0.863 0.792 2,973.8
ce_rerank + llm_summarize + long_context_reorder 82.40% 0.87 0.778 2,167.4
ce_rerank + long_context_reorder + reflection_revising 82.40% 0.865 0.783 1,773.4
hyde + relevant_segment_extractor + llm_summarize + long_context_reorder 82.00% 0.891 0.75 2,685.5
ce_rerank + adjacent_augmenter + tree_summarize + long_context_reorder 81.90% 0.865 0.772 4,762.7
hyde + llm_summarize + long_context_reorder 81.50% 0.896 0.733 3,437.9
simple_query_refinement_clarification + ce_rerank + adjacent_augmenter + long_context_reorder 81.10% 0.904 0.719 1,928.6
adjacent_augmenter + llm_summarize + long_context_reorder 81.10% 0.896 0.726 3,073.2
hyde + llm_summarize 81.10% 0.896 0.726 3,409.4
ce_rerank + relevant_segment_extractor + llm_summarize + long_context_reorder + reflection_revising 81.10% 0.87 0.753 2,828.2
hyde + llm_summarize 80.90% 0.896 0.723 3,261.5
hyde + adjacent_augmenter + llm_summarize + long_context_reorder 80.70% 0.896 0.717 3,994.5
hyde + llm_summarize + long_context_reorder 80.60% 0.896 0.715 3,472.5
hyde + llm_summarize + long_context_reorder + reflection_revising 80.50% 0.896 0.714 3,806.4
ce_rerank + relevant_segment_extractor + tree_summarize + long_context_reorder 80.50% 0.87 0.74 3,886.9
simple_query_refinement_clarification + ce_rerank + adjacent_augmenter 80.20% 0.901 0.704 1,738.0
query_expansion_simple_multi_query_borda + ce_rerank + adjacent_augmenter 80.00% 0.887 0.712 1,431.4
llm_rerank + llm_summarize 80.00% 0.885 0.715 2,707.5
simple_query_refinement_clarification + llm_rerank + adjacent_augmenter 79.80% 0.903 0.693 2,291.5
simple_query_refinement_clarification + ce_rerank + adjacent_augmenter + reflection_revising 79.80% 0.9 0.697 1,991.6
simple_query_refinement_clarification + ce_rerank + adjacent_augmenter + tree_summarize + long_context_reorder +
reflection_revising79.60% 0.836 0.756 4,524.5
query_expansion_simple_multi_query_borda + ce_rerank + similarity_threshold + adjacent_augmenter + llm_summarize +
reflection_revising79.40% 0.882 0.706 2,155.3
hyde + ce_rerank + llm_summarize + long_context_reorder 79.40% 0.877 0.711 3,197.8
hyde + tree_summarize + long_context_reorder 79.30% 0.896 0.689 7,137.7
hyde + ce_rerank + adjacent_augmenter + llm_summarize 79.30% 0.877 0.708 3,343.6
vector_simple + simple_threshold + simple_listing (Baseline) 78.70% 0.872 0.702 1,000.4
adjacent_augmenter + tree_summarize + long_context_reorder 78.40% 0.896 0.672 7,216.6
ce_rerank + similarity_threshold + relevant_segment_extractor + tree_summarize + reflection_revising 78.30% 0.755 0.81 2,717.9