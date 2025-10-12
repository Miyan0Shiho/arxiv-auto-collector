# WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives

**Authors**: Yongan Yu, Xianda Du, Qingchen Hu, Jiahao Liang, Jingwei Ni, Dan Qiang, Kaiyu Huang, Grant McKenzie, Renee Sieber, Fengran Mo

**Published**: 2025-10-06 19:58:42

**PDF URL**: [http://arxiv.org/pdf/2510.05336v1](http://arxiv.org/pdf/2510.05336v1)

## Abstract
Historical archives on weather events are collections of enduring primary
source records that offer rich, untapped narratives of how societies have
experienced and responded to extreme weather events. These qualitative accounts
provide insights into societal vulnerability and resilience that are largely
absent from meteorological records, making them valuable for climate scientists
to understand societal responses. However, their vast scale, noisy digitized
quality, and archaic language make it difficult to transform them into
structured knowledge for climate research. To address this challenge, we
introduce WeatherArchive-Bench, the first benchmark for evaluating
retrieval-augmented generation (RAG) systems on historical weather archives.
WeatherArchive-Bench comprises two tasks: WeatherArchive-Retrieval, which
measures a system's ability to locate historically relevant passages from over
one million archival news segments, and WeatherArchive-Assessment, which
evaluates whether Large Language Models (LLMs) can classify societal
vulnerability and resilience indicators from extreme weather narratives.
Extensive experiments across sparse, dense, and re-ranking retrievers, as well
as a diverse set of LLMs, reveal that dense retrievers often fail on historical
terminology, while LLMs frequently misinterpret vulnerability and resilience
concepts. These findings highlight key limitations in reasoning about complex
societal indicators and provide insights for designing more robust
climate-focused RAG systems from archival contexts. The constructed dataset and
evaluation framework are publicly available at
https://anonymous.4open.science/r/WeatherArchive-Bench/.

## Full Text


<!-- PDF content starts -->

Work in progress.
WEATHERARCHIVE-BENCH: BENCHMARKING
RETRIEVAL-AUGMENTEDREASONING FORHISTORI-
CALWEATHERARCHIVES
Yongan Yu
∗, Xianda Du
∗, Qingchen Hu
∗, Jiahao Liang
∗, Jingwei Ni
 , Dan Qiang
Kaiyu Huang
 , Grant McKenzie
 , Ren ´ee Sieber
†, Fengran Mo
†
McGill University
 University of Waterloo
 Universit ´e de Montr ´eal
ETH Zurich
 Beijing Jiaotong University
yongan.yu@mail.mcgill.ca; renee.sieber@mcgill.ca; fengran.mo@umontreal.ca
ABSTRACT
Historical archives on weather events are collections of enduring primary source
records that offer rich, untapped narratives of how societies have experienced and
responded to extreme weather events. These qualitative accounts provide insights
into societal vulnerability and resilience that are largely absent from meteoro-
logical records, making them valuable for climate scientists to understand soci-
etal responses. However, their vast scale, noisy digitized quality, and archaic
language make it difficult to transform them into structured knowledge for cli-
mate research. To address this challenge, we introduce WEATHERARCHIVE-
BENCH, the first benchmark for evaluating retrieval-augmented generation (RAG)
systems on historical weather archives. WEATHERARCHIVE-BENCHcomprises
two tasks:WeatherArchive-Retrieval, which measures a system’s ability to lo-
cate historically relevant passages from over one million archival news seg-
ments, andWeatherArchive-Assessment, which evaluates whether Large Lan-
guage Models (LLMs) can classify societal vulnerability and resilience indi-
cators from extreme weather narratives. Extensive experiments across sparse,
dense, and re-ranking retrievers, as well as a diverse set of LLMs, reveal that
dense retrievers often fail on historical terminology, while LLMs frequently mis-
interpret vulnerability and resilience concepts. These findings highlight key
limitations in reasoning about complex societal indicators and provide insights
for designing more robust climate-focused RAG systems from archival con-
texts. The constructed dataset and evaluation framework are publicly available
at: https://github.com/Michaelyya/WeatherArchive-Bench.
1 INTRODUCTION
Extreme weather events are becoming increasingly frequent and severe as a result of climate change,
posing urgent challenges for climate adaptation and disaster preparedness (O’Brien et al., 2006).
Climate policymakers are expected to design targeted adaptation strategies that integrate disaster
response with long-term planning, including climate-resilient urban development (Xu et al., 2024)
and sustainable land use policies (Zuccaro et al., 2020). Achieving these goals requires not only
meteorological data, but also a deeper understanding of how communities, infrastructures, and eco-
nomic sectors have responded to climate hazards (Bollettino et al., 2020; Mallick et al., 2024; Yuan
et al., 2025). Historical archives provide such knowledge, documenting past extreme weather events
alongside their cascading economic impacts, community responses, and local adaptation practices
(Carey, 2012; Yu et al., 2025a). A systematic analysis of these records can reveal which factors were
most disruptive during a specified extreme weather event, thereby providing evidence-based insights
to inform future climate policy interventions.
∗Equal contribution.
†Corresponding author.
1arXiv:2510.05336v1  [cs.CL]  6 Oct 2025

Work in progress.
Table 1: Comparison of existing QA benchmarks with WEATHERARCHIVE-BENCH.
Dataset # Papers Paper Source Domain Historical Data Task
REPLIQA 17.9K Synthetic General✗Topic Retrieval + QA
CPIQA 4.55K Climate papers Climate Sci.✗Multimodal QA
ClimRetrieve 30 Reports Climate Sci.✗Document Retrieval
ClimaQA 23 Textbooks Climate Sci.✗Scientific QA
WeatherArchive 1.05M Hist. archives Climate Sci.✓Retrieval + QA + classification
Recent advances in generative AI provide the feasibility to process substantial collections and ex-
tract structured insights at scale. In particular, RAG is a methodology that combines information
retrieval systems with generative language models to enhance performance on knowledge-intensive
domain tasks (Lewis et al., 2020; Mo et al., 2025). In terms of the climate domain, RAG systems can
search through vast collections of weather records (Tan et al., 2024) and then interpret retrieved in-
formation into structured insights about climate impacts and societal responses (Vaghefi et al., 2023;
Xie et al., 2025). RAG thus serves as an essential approach for climate adaptation planning, helping
policymakers systematically learn from past weather impacts to enhance current risk assessment and
decision-making (Br ¨onnimann et al., 2019). However, applying RAG to historical climate archives
for extracting societal vulnerability and resilience insights is non-trivial. For instance, historical
documents contain outdated terminology, Optical Character Recognition (OCR) errors, and narra-
tive formats that mix weather accounts with unrelated text (Bingham, 2010; Verhoef et al., 2015).
Such noise and irregular structures pose challenges for both retrieval systems in locating relevant
passages and LLMs in interpreting societal insights. Similarly, LLMs face significant knowledge
gaps when processing historical archives, since these data are generally unavailable online and ex-
cluded from pre-training corpora (Liu et al., 2024). Thus, models might not be able to interpret
historical terminology for vulnerability and resilience analysis. Likewise, existing retriever mod-
els might also struggle to identify climate-relevant passages, especially when historical vocabulary
and narrative styles differ substantially from modern web content (Perełkiewicz & Po ´swiata, 2024).
More importantly, no existing benchmark systematically evaluates RAG performance on historical
climate archives, which constrains the possibility of optimizing the RAG systems in the climate
domain. As shown in Table 1, existing benchmarks focus on relatively small-scale and primarily
target scientific papers and reports rather than historically grounded archival data. None evaluates
the extraction of societal vulnerability and resilience indicators for climate adaptation planning. This
evaluation gap prevents the development of AI systems capable of translating irreplaceable historical
evidence of past climate responses into actionable intelligence for contemporary policy decisions.
To address this gap, we introduce WEATHERARCHIVE-BENCH, the first large-scale benchmark
for retrieval-augmented reasoning on historical weather archives. WEATHERARCHIVE-BENCHfo-
cuses on two complementary tasks:WeatherArchive-Retrieval, which evaluates retrieval models’
ability to identify evidence-based passages in response to specified extreme weather events, and
WeatherArchive-Assessment, which measures LLMs’ capacity to answer evidence-based climate
queries and classify indicators of societal vulnerability and resilience from archival narratives using
the retrieved passages. In this context, vulnerability refers to the susceptibility of communities, in-
frastructures, or economic systems to climate-related harm, while resilience denotes the capacity to
absorb and recover from climate shocks (Feldmeyer et al., 2019). Understanding these dimensions
from historical records is critical for identifying risk factors (Rathoure, 2025), designing interven-
tions, and learning from past adaptation strategies across contexts and time periods (Kelman et al.,
2016). To support rigorous evaluation, we curate over one million OCR-parsed archival documents
with dedicated preprocessing strategies, followed by expert validation and systematic quality con-
trol. We then evaluate a range of retrieval models and state-of-the-art LLMs on three core capabil-
ities required for climate applications: (1) processing archaic language and noisy OCR text typical
of historical documents, (2) understanding domain-specific terminology and concepts, and (3) per-
forming structured reasoning about socio-environmental relationships embedded in narratives. Our
results reveal significant limitations of current systems: dense retrieval models often fail to capture
historical terminology compared to sparse methods, while LLMs frequently misinterpret vulnerabil-
ity and resilience indicators. These findings highlight the need for methods that adapt to historical
archival data, integrate structured domain knowledge, and reason robustly under noisy conditions.
In summary, our contributions are threefold:
2

Work in progress.
1. We introduce WEATHERARCHIVE-BENCH, which provides two evaluation tasks:
WeatherArchive-Retrieval, assessing retrieval models’ ability to extract relevant historical
passages, andWeatherArchive-Assessment, evaluating LLMs’ capacity to classify societal
vulnerability and resilience indicators from archival weather narratives.
2. We release the first large-scale corpus of over one million historical archives, enriched
through preprocessing and human curation to ensure quality, enabling both climate scien-
tists and the broader community to leverage historical data.
3. We conduct comprehensive empirical analyses of state-of-the-art retrieval models and
LLMs on historical climate archives, uncovering key limitations in handling archaic lan-
guage and domain-specific terminology, and providing insights to guide the development
of more robust climate-focused RAG systems.
2 RELATEDWORK
2.1 CLIMATE-FOCUSEDNLP
The urgency of addressing environmental challenges has intensified in recent decades, driven by
mounting evidence of climate change, habitat degradation, and biodiversity loss (Rathoure, 2025).
Advancing disaster preparedness requires tools that can assess vulnerabilities and resilience using re-
alistic, context-rich cases, which urban planners and policymakers can directly act upon (Birkmann
et al., 2015; Jetz et al., 2012). Recent climate-focused NLP systems demonstrate impressive techni-
cal capabilities. ClimaX (Nguyen et al., 2023) achieves competitive performance with operational
forecasting systems by pre-training on diverse CMIP6 meteorological datasets, while ClimateGPT
(Thulke et al., 2024) integrates interdisciplinary climate research to outperform general-purpose
models on climate-specific benchmarks. Specialized models, such as WildfireGPT (Xie et al., 2024),
employ RAG for hazard-specific tasks, like wildfire risk assessment. Yet, these advances focus
largely on contemporary climate data and physical processes. They overlook real-world impact
records (e.g., detailed accounts of infrastructure failures, agricultural losses) embedded in archival
sources. Such records are essential for climate policy makers and urban planners to understand
long-term vulnerability patterns and inform policies aimed at future disaster preparedness.
2.2 BENCHMARKING INCLIMATEAI
Historically, progress in climate AI has been constrained by the scarcity of large-scale, practical
datasets that capture real-world climate impacts in sufficient temporal and geographic breadth (Zha
et al., 2025). Existing resources predominantly target either physical climate modelling tasks or
narrowly scoped contemporary text analysis, leaving historical, case-based impact records largely
untapped. For example, ClimateIE (Pan et al., 2025) offers 500 annotated climate science publi-
cations mapped to the GCMD+ taxonomy, yet focuses on technical entities such as observational
variables rather than societal consequences of extreme weather. Evaluation benchmarks face similar
limitations. ClimateBench (Watson-Parris et al., 2022) and those catalogued in ClimateEval (Kurfalı
et al., 2025) primarily assess meteorological prediction accuracy or general climate communication
tasks, with little insight into how well language models can identify and interpret vulnerability and
resilience patterns from complex, domain-specific archives. As Xie et al. (2025) shows that RAG of-
fers a promising avenue for climate experts to locate relevant cases within vast collections to support
analysis in the context of extreme events, yet its effectiveness in systematically capturing historical
vulnerability and resilience information remains untested. This gap in both data and evaluation mo-
tivates our introduction of WEATHERARCHIVE-BENCH, a large-scale benchmark for assessing AI
systems’ capacity to extract and synthesize real-world climate impact narratives.
3 WEATHERARCHIVE-BENCH
Our goal with WEATHERARCHIVE-BENCHis to provide a realistic benchmark for evaluating cur-
rent retrieval and reasoning capabilities in the context of climate- and weather-related archival texts.
In particular, we focus on the dual challenges of (i) constructing a high-quality corpus from histori-
cal archives and (ii) defining retrieval and generation tasks that capture the practical needs of climate
researchers. This section details our corpus collection pipeline and task formulation.
3

Work in progress.
3.1 CORPUSCOLLECTION
LLMs are generally pre-trained on large-scale internet corpora, which frequently include fake and
unreliable content (Roy & Chahar, 2021). In contrast, historical archives provide a unique and valu-
able information source, as copyright restrictions typically exclude them from LLM pretraining data.
Unlike standardized meteorological datasets, historical archives provide rich, narrative descriptions
of weather-related disruptions and community-level adaptation successes (Sieber et al., 2024). These
archives also capture public voices and societal perspectives that would be prohibitively expensive
to collect today, yet public perceptions remain documented in historical records (Thurstan et al.,
2016). Thus, our corpus offers contextualized insights that complement traditional climate data. For
climate scientists seeking to understand long-term patterns of societal vulnerability and resilience,
these narrative-rich sources provide invaluable evidence of how communities have historically ex-
perienced, interpreted, and responded to weather-related challenges.
Our corpus construction emphasizes both scale and reliability. Sourced from a proprietary archive
institution, we collected two 20-year tranches of archival articles from Southern Quebec, a region
representative of broader Northeastern American weather patterns: one covering a contemporary
period (1995–2014) and one covering a historical period (1880–1899). The articles were digitized
via OCR and subsequently cleaned to correct recognition errors via GPT-4o, ensuring textual quality
suitable for downstream applications. The detailed preprocessing and validation steps are provided
in Appendix A. We then segmented the articles into overlapping chunks using a sliding-window
approach, followed by the method proposed by (Sun et al., 2023), allowing each segment to pre-
serve sufficient semantic context while satisfying token-length constraints. The resulting dataset
comprises 1,035,862 passages, each standardized to approximately 256 tokens, which we used for
WeatherArchive-Retrievaltask creation.
3.2 TASK DEFINITION
WEATHERARCHIVE-BENCHincorporates two complementary tasks designed to mirror the work-
flow of climate scientists.WeatherArchive-Retrievaltests models’ ability to locate relevant historical
evidence. The other isWeatherArchive-Assessment, which evaluates their capacity to interpret com-
plex socio-environmental relationships within an archival report of an extreme weather event.
3.2.1 WEATHERARCHIVE-RETRIEVAL
Figure 1: The construction pipeline of the retrieval task in weather archive collections. The process
integrates newspaper collection, keyword frequency search, and human verification to construct a
high-quality corpus of weather-related articles with relevance judgments for each query.
335 Gold Weather-
related ArticlesGenerated QueryWhat political responsibility does the city attribute to
Westmount municipality regarding the flooding issues
associated with the intercepting sewer on St. Catherine street?
QA-PairsGold Retrieved article
 Newspaper 
Collection
Keyword 
Frequency SearchHuman 
VerificationVectorDBVector
EmbeddingRetriever
GPT-4oThe Coteau Harron main sewer was built in 1867 in a low
swampy valley ...... In regard to the intercepting sewer on St.
Catherine street, there is only one portion of it which is at present
needed during heavy rainstorms, that is between St. Lawrence
and St George streets. These floodings take place only when there
is an exceptionally heavy rainstorm, but these floodings would
not have occurred if Westmount municipality, who had the
permission of the city to drain their drainage into the intercepting
sewer, had abided by the plan which they submitted to the city.
In scientific domains such as climate analysis, scientists often rely on precedents embedded in long
historical archives (Herrera et al., 2003; Slonosky & Sieber, 2020; Sieber et al., 2022). A well-
designed retrieval task (Figure 1) is essential, as it evaluates a model’s ability to identify contex-
tually relevant and temporally appropriate information while providing a reliable foundation for
subsequent answer generation.
To construct the benchmark, we first ranked 1,035k passages by the frequency of keywords related
to disruptive weather events, as detailed in Appendix B.1. From this ranking, we selected the top
525 passages, which were then manually reviewed by domain experts to identify those providing
4

Work in progress.
complete evidential support for end-to-end question answering. After curation, 335 high-quality
validated passages were retained. To better characterize their topical coverage, we analyzed the
word frequencies of these passages and present the results as a word cloud in Appendix B.2. For
each passage, we generated domain-specific queries using GPT-4o, with prompts provided in Ap-
pendix C. These queries were designed to emulate real-world research intents, resulting in a realistic
retrieval benchmark composed of query–answer pairs.
The difficulty of this task stems from the nature of passages extracted from historical archives. Un-
like contemporary datasets, news archives use domain-specific terminology that has shifted over
time (e.g., outdated expressions for storms or floods; presented in Appendix D), which makes rel-
evance judgments nontrivial. Moreover, articles frequently embed descriptions of weather impacts
within broader narratives or unrelated sections such as advertisements, which introduces additional
noise into the retrieval process. By grounding evaluation in such historically situated and noisy data,
WeatherArchive-Retrievalestablishes a challenging yet realistic testbed for assessing the robustness
of retrieval models and systems.
3.2.2 WEATHERARCHIVE-ASSESSMENT
To effectively support climate scientists in disaster preparedness, language models must go beyond
retrieving relevant passages and demonstrate the ability to interpret societal vulnerability and re-
silience as documented in historical texts. To this end, we design an evaluation framework to assess
a model’s ability to reason about climate impacts across multiple levels, drawing on established ap-
proaches from vulnerability and adaptation research (Feldmeyer et al., 2019). An overview of the
societal vulnerability and resilience framework is provided in Appendix E. The framework com-
prises two complementary subtasks: (i) classification of societal vulnerability and resilience indi-
cators, and (ii) open-ended question answering to assess model generalization on climate impact
analysis. The construction pipeline is illustrated in Figure 2. A more detailed description of the
human validation process is provided in Appendix F, and ground-truth oracles are generated using
GPT-4.1 with structured prompting, as detailed in Appendix G.
Figure 2:WeatherArchive-Assessment- the construction pipeline of assessment task on societal
vulnerability and resilience. GPT-4.1 evaluates retrieved weather articles across multiple criteria,
with human verification ensuring quality before generating ground truth answers. This sample case
shows the assessment of rainstorm impacts.
Impacts and responses are described across the
Chippewa Valley, Wolf, Little Wolf, and Waupaca
rivers, and involve multiple towns...
"The heavy rainstorm caused a near two-foot rise in river levels, suspending logging operations and breaking up camps in
the Chippewa Valley. However, the increased water flow is expected to soon clear ice from reservoirs, enabling an earlier-
than-usual start to river navigation and log drives, with mills and transport systems preparing to resume operations."
335 Gold Weather-
related ArticlesGPT-4,1
Article
Metrics
Exposure
Sensitivity
Adaptability
Temporal scale
Rating
Functional
system
Spatial scaleSudden-Onset
Moderate
Robust
"short-term
absorptive capacity"
"transportation"
"regional"
Query :How has the recent heavy rainstorm affected the logging and river navigation
infrastructure in the Chippewa Valley, and what are the anticipated changes in
operations as a result?
-
Evidence
A heavy rainstorm.., followed by moderate weather,
has virtually suspended logging operations...
Logging operations... have virtually
suspended... The dams are frozen to such an
extent that logs cannot be run through.
A large number of men were sent up both rivers
this morning...The improvement at the mills are
about completed... 
Logging camps have broken up and operations
suspended, but men are being sent up the rivers
to prepare for an early drive...
The rains have caused a rise... leaving the
river free for a resumption of raft and boat
navigation...The Wisconsin Central Railway
is running extra log trains daily... 
Anwser :
Human Verification
Ground Truth for
Generation
THE LM CROP, Eau Claire, Wis, March 30 A heavy
rainstorm, which set in early Saturday morning, followed by
moderate weather, has virtually suspended logging
operations in the Chippewa Valley, and the camps, with few
exceptions, have broken up, after having accomplished a
good season's work. The rains have caused a rise of nearly
two feet in the Chippewa and Eau, which is perceptible at
this point, and there is a prospect of the ice moving out of
the Dells reservoir and other dams above before the close of
the week, leaving the river free for a resumption of raft and
boat navigation. The improvement at the mills are about
completed, and there is a possibility that several of the
establishments having a surplus of logs from last season will
commence operations this week. A large number of men
were sent up both rivers this morning, so as to be in readiness
for the drive, which is anticipated to take place earlier than
usual. ON THE WOLF AND WAUPACA RIVERS,
Waupaca, Wisconsin, March 30 The lumbermen, with their
teams, are rushing out of the woods and spending a few days
in the cities preparatory to going on to the river. The rivers
are all open now, but the dams are frozen to such an extent
that logs cannot be run through.
Societal Vulnerability.Vulnerability is widely conceptualized as a function of exposure, sensi-
tivity, and adaptive capacity (O’Brien et al., 2004). We operationalize this framework by prompt-
ing models to assign descriptive levels to each component. Prompt details are provided in Ap-
5

Work in progress.
pendix G.2. Specifically,exposurecharacterizes the type of climate or weather hazard, distin-
guishing between sudden-onset shocks (e.g., storms, floods), slow-onset stresses (e.g., prolonged
droughts, sea-level rise), and compound events involving multiple interacting hazards.Sensitivity
evaluates how strongly the system is affected by such hazards, ranging from critical dependence
on vulnerable resources to relative insulation from disruption.Adaptive capacitycaptures the abil-
ity of the system to respond and recover, spanning robust governance and infrastructure to fragile
conditions with little or no coping capacity.
This classification-wise evaluation examines whether models can move beyond surface-level text
interpretation toward structured reasoning about vulnerability, which is essential for anticipating
climate risks (Linnenluecke et al., 2012). In practice,exposureandadaptive capacityare often
signalled by explicit indicators (Brooks et al., 2005) such as infrastructure damage or recovery mea-
sures, which evaluate LLMs’ capacity to capture through climate factual extraction.Sensitivityis
more challenging, as it requires climate reasoning (Montoya-Rincon et al., 2023) about governance
quality, institutional strength, or social capital, factors that are seldom directly expressed in histor-
ical archives. By incorporating both explicit and implicit aspects of vulnerability, our framework
provides a rigorous test of whether models can integrate factual evidence with contextual inference.
Societal Resilience.Resilience is evaluated using indicators proposed by Feldmeyer et al. (2019),
which emphasize adaptation processes across three scales. On thetemporal scale, models must dis-
tinguish between short-term absorptive capacity (e.g., emergency response), medium-term adaptive
capacity (e.g., policy or infrastructure adjustments), and long-term transformative capacity (e.g.,
systemic redesign). On thefunctional system scale, models identify which systems are affected,
including health, energy, food, water, transportation, and information, highlighting their interde-
pendence in shaping preparedness. Lastly, on thespatial scale, models assess resilience across
levels (e.g., local, community, regional, national), capturing variation in adaptive capacity across
contexts. Through the experts’ annotation process, we are informed that temporal indicators are
often easier to identify since newspapers tend to report immediate damages and responses explicitly,
whereas functional and spatial dimensions are more challenging since they require models to infer
systemic interactions and contextual variation that are rarely stated explicitly in news archives. By
formulating these criteria into multiple-choice questions, we evaluate whether models can recognize
structured indicators of resilience within noisy archival narratives.
Question Answering (QA) in Climate AI.Each climate-related query is first processed by the
baseline retrieval model, which returns the top-3 passages as reference material. Gold-standard
answers are then required to complete the pipeline and ensure the usefulness of the RAG system
for climate scientists. While classifying vulnerability and resilience indicators evaluates whether
models can extract structured evidence, open-ended QA assesses their ability to generalize beyond
explicit textual signals. In particular, QA requires models to synthesize information scattered across
archival sources and to articulate climate impacts in ways that support scientific reasoning.
4 EXPERIMENTALSETUP
4.1 EVALUATIONMETRICS
WeatherArchive-Retrieval.We evaluate retrieval performance with the commonly used metrics,
including Recall@k, MRR@k, and nDCG@kfork∈ {3,10,50,100}. These metrics capture
complementary aspects of performance: coverage (Recall), top-rank relevance (MRR), and graded
relevance (nDCG).
WeatherArchive-Assessment.The downstream benchmark evaluates model performance on
climate-related reasoning tasks via expert-validated reference standards. Evaluation proceeds along
two dimensions: (i) Vulnerability and resilience indicator classification, models must identify and
categorize societal factors from historical weather narratives, with performance quantified through
F1, precision, and recall metrics to measure classification accuracy; (ii) Historical climate question
answering, models generate responses to evidence-based climate queries using retrieved archival
passages, with answer quality assessed via BLEU, ROUGE, and BERTScore for semantic similarity
to expert-authored responses, supplemented by token-level F1 between predictions and ground truth.
6

Work in progress.
Additionally, we employ LLM-based judgment using GPT-4.1 to evaluate climate reasoning qual-
ity beyond traditional similarity metrics, determining whether model-generated responses contain
factual errors. Outputs are compared against oracle answers and are judged correct if the model’s
response to the specified climate-related question matches or encompasses the oracle answer, and
incorrect otherwise. These metrics jointly test whether models can both recognize structured indi-
cators and reason accurately in free-form responses, aligning evaluation with the benchmark’s goal
of measuring climate-relevant retrieval-augmented reasoning.
4.2 EVALUATEDMODELS
Retrieval Models.We evaluate a set of retrieval models on the archival collections, including
three categories: (i) sparse lexical models: BM25 (BM25plus, BM25okapi) (Robertson et al., 2009)
and SPLADE (Formal et al., 2021) (ii) dense embedding models: ANCE (Xiong et al., 2020),
SBERT (Reimers & Gurevych, 2019), and large proprietary embeddings, including OpenAI’s text-
embedding-ada-002 (Neelakantan et al., 2022), Gemini’s text-embedding (Lee et al., 2025), IBM’s
Granite Embedding (Awasthy et al., 2025), and Snowflake’s Arctic-Embed (Yu et al., 2024) and (iii)
re-ranking models: cross-encoders applied on BM25 candidates (BM25plus+CE, BM25okapi+CE)
with a MiniLM-based reranker (Wang et al., 2020; Hofst ¨atter et al., 2020). Implementation details
for each model are provided in Appendix J.
Language Models.We consider a diverse suite of open-source and proprietary LLMs with var-
ious parameter scales. Open-source models include Qwen-2.5 (7B–72B), Qwen-3 (4B, 30B),
LLaMA-3 (8B, 70B), Mistral-8B and Ministral-8×7B. These families capture scaling effects, effi-
ciency–performance trade-offs, and robustness to long or noisy text. We also include DeepSeek-V3-
671B, which targets efficient scaling and adaptability. proprietary models include GPT (3.5-turbo,
4o), Claude (opus-4-1, sonnet-4) and gemini-2.5-pro, which are widely used in applied pipelines,
offering strong reasoning and summarization capabilities. All models are instruction-tuned versions,
denoted as “IT”. The computational costs of deploying these models are provided in Appendix H.
5 EXPERIMENTALRESULTS
In this section, we evaluate and analyze the performance of retrieval models and state-of-the-art
LLMs onWeatherArchive-RetrievalandWeatherArchive-Assessmenttasks in our benchmarks.
5.1 WEATHERARCHIVE-RETRIEVAL EVALUATION
Table 2: Retrieval performance onWeatherArchive-Retrievalacross sparse, dense, and re-ranking
models.Boldandunderline indicate the best and the second-best performance.
Recall nDCG
Category Model @3 @10 @50 @100 @3 @10 @50 @100
SparseBM25PLUS0.585 0.678 0.791 0.827 0.497 0.532 0.557 0.563
BM25OKAPI0.543 0.678 0.794 0.830 0.444 0.494 0.519 0.525
SPLADE0.075 0.182 0.478 0.645 0.060 0.097 0.160 0.188
DenseSBERT0.290 0.400 0.501 0.552 0.228 0.268 0.290 0.298
ANCE0.340 0.522 0.779 0.866 0.273 0.338 0.394 0.408
ARCTIC0.534 0.675 0.821 0.910 0.443 0.494 0.527 0.542
GRANITE0.546 0.719 0.887 0.946 0.448 0.512 0.550 0.559
OPENAI-3-SMALL0.516 0.678 0.854 0.919 0.436 0.494 0.533 0.544
OPENAI-3-LARGE0.481 0.651 0.854 0.922 0.400 0.461 0.507 0.518
OPENAI-ADA-002 0.510 0.702 0.881 0.955 0.421 0.492 0.531 0.543
GEMINI-EMBEDDING-001 0.573 0.749 0.916 0.9580.479 0.543 0.582 0.588
Re-RankingBM25PLUS+CE0.639 0.7610.812 0.827 0.532 0.579 0.590 0.593
BM25OKAPI+CE0.639 0.7610.818 0.8300.533 0.580 0.592 0.594
7

Work in progress.
Sparse Retrieval Models Achieve Strong Top-rank Relevance on Climate Archives.As shown
in Table 2, BM25 variants continue to perform strongly, often matching or surpassing dense alter-
natives in ranking quality at topk. The effectiveness of BM25 might be related to the nature of
climate-related queries, which usually contain technical terminology and domain-specific colloca-
tions (e.g., “flood damage,” “hurricane casualties,” “crop failure due to drought”). In such cases,
exact lexical matching is critical as sparse methods are able to capture these specialized terms di-
rectly, whereas dense representations may blur over distinctions or concepts. For instance, a query
about “storm surge fatalities” would benefit from precise overlap with passages containing the same
terminology, whereas a dense retriever might incorrectly emphasize semantically related but dis-
tinct expressions such as “storm warnings” or “storm intensity”, as an example case provided in
Appendix C. These findings highlight the importance of sparse methods in scientific and technical
domains where specialized vocabulary governs relevance.
Re-ranking Procedure Could Deliver Better Performance.With the effective sparse methods,
further deploying a re-ranker could achieve better performance. In this setup, BM25 provides high
lexical coverage at the candidate generation stage, and the re-ranker ranks the top candidates by
modelling fine-grained query–document interactions. Empirically, the results show that hybrid mod-
els such as BM25plus+CE and BM25okapi+CE consistently outperform both pure sparse and pure
dense baselines within the top-ranked results (e.g., top 3-10 passages), which are most critical for
downstream QA. This indicates that re-ranking models with baseline yields more robust perfor-
mance for climate-related retrieval.
5.2 WEATHERARCHIVE-ASSESSMENTEVALUATION
Table 3: Vulnerability and resilience indicator classification performance onWeatherArchive-
Assessmentacross diverse LLMs.Boldandunderline indicate the best and second-best results.
Vulnerability ResilienceAverage
Model Exposure Sensitivity Adaptability Temporal Functional Spatial
GPT-4O0.646 0.528 0.580 0.6230.6450.518 0.590
GPT-3.5-TURBO0.636 0.466 0.465 0.643 0.342 0.395 0.491
CLAUDE-OPUS-4-1 0.783 0.676 0.6750.8460.6250.614 0.703
CLAUDE-SONNET-4 0.7720.7380.597 0.652 0.635 0.603 0.666
GEMINI-2.5-PRO0.766 0.620 0.571 0.756 0.625 0.613 0.658
DEEPSEEK-V3-671B0.7980.4950.7090.760 0.613 0.608 0.664
MIXTRAL-8X7B-IT 0.273 0.214 0.241 0.322 0.214 0.326 0.265
MINISTRAL-8B-IT 0.437 0.188 0.246 0.458 0.419 0.370 0.353
QWEN3-30B-IT 0.658 0.444 0.300 0.730 0.342 0.364 0.478
QWEN3-4B-IT 0.320 0.275 0.184 0.4960.6450.285 0.368
QWEN2.5-72B-IT 0.744 0.434 0.676 0.735 0.498 0.515 0.600
QWEN2.5-32B-IT 0.533 0.312 0.449 0.609 0.469 0.365 0.456
QWEN2.5-14B-IT 0.405 0.392 0.295 0.357 0.234 0.303 0.331
QWEN2.5-7B-IT 0.338 0.091 0.225 0.330 0.308 0.329 0.270
LLAMA-3.3-70B-IT 0.367 0.429 0.244 0.481 0.531 0.355 0.401
LLAMA-3-8B-IT 0.243 0.198 0.184 0.194 0.290 0.286 0.233
Average 0.547 0.406 0.415 0.562 0.465 0.428 0.47
Factual Extraction vs. Climate Reasoning in Societal Vulnerability Assessment.Consistent
with prior work on scaling laws (Kaplan et al., 2020), larger models generally improve zero-shot
generation. This is likely because greater capacity increases the chance of encountering relevant pat-
terns during pretraining. Table 3 shows that Claude-Opus-4-1 achieves the best overall performance,
while among open-source models, DeepSeek-V3-671B ranks highest, followed by Qwen2.5-72B-
IT. Models perform well on explicit indicators of exposure and adaptability, such as infrastructure
damage or recovery measures, where factual extraction is sufficient. In contrast, sensitivity indi-
cator classification requires reasoning about the degree to which a system is affected by weather
stressors (Morss et al., 2011), including governance quality and social capital dimensions that are
8

Work in progress.
rarely explicit in historical archives. While some proprietary models maintain strong performance
on these tasks, open-source models show a sharper decline. Overall, larger models improve factual
extraction and can handle certain reasoning tasks effectively, but challenges remain for tasks that
require inferring implicit relationships with society and extreme weather from historical archives.
LLMs Struggle with Socio-environmental System Effects.Societal resilience indicator classifi-
cations require recognizing direct damages from disruptive weather events and reasoning about how
shocks propagate across geographic scales and interdependent systems. As shown in Table 3, mod-
els achieve relatively strong performance on temporal dimensions with a score of 0.562 on average,
with Claude-Opus-4-1 and DeepSeek-V3-671B reliably identifying immediate response capacities.
However, performance degrades on functional and spatial dimensions, where even sophisticated
models struggle to assess cross-system dependencies (e.g., over-predicting “transportation” or “in-
formation”) and multi-scale coordination (e.g., overlooking “local”). Samples are provided in Ap-
pendix L.2. Impacts are distributed unevenly across systems and exhibit inherently scale-dependent
propagation dynamics. This pattern reveals limitations as models perform well at identifying di-
rect impacts, yet are limited in reasoning over complex socio-environmental interdependencies that
mediate systemic resilience. This highlights that multi-scale vulnerability assessment still requires
human expertise.
Figure 3: Performance comparison of LLMs on free-form QA task across various metrics.
/uni00000013/uni00000011/uni00000013/uni00000015/uni00000018/uni00000013/uni00000011/uni00000013/uni00000018/uni00000013/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000018/uni00000013/uni00000011/uni00000014/uni00000013/uni00000013/uni00000013/uni00000011/uni00000014/uni00000015/uni00000018/uni00000013/uni00000011/uni00000014/uni00000018/uni00000013
/uni00000013/uni00000011/uni00000013/uni0000001b/uni00000017
/uni00000013/uni00000011/uni00000013/uni0000001b/uni00000016
/uni00000013/uni00000011/uni00000014/uni00000017/uni00000014
/uni00000013/uni00000011/uni00000014/uni00000013/uni0000001c
/uni00000013/uni00000011/uni00000013/uni00000019/uni0000001c
/uni00000013/uni00000011/uni00000013/uni0000001c/uni00000017
/uni00000013/uni00000011/uni00000013/uni00000016/uni00000018
/uni00000013/uni00000011/uni00000013/uni00000015/uni00000017
/uni00000013/uni00000011/uni00000014/uni00000015/uni00000017
/uni00000013/uni00000011/uni00000013/uni00000019/uni00000016
/uni00000013/uni00000011/uni00000014/uni00000014/uni00000015
/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000015
/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000016
/uni00000013/uni00000011/uni00000014/uni00000014/uni0000001b/uni00000025/uni0000002f/uni00000028/uni00000038
/uni00000013/uni00000011/uni00000014/uni00000013/uni00000011/uni00000015/uni00000013/uni00000011/uni00000016/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018
/uni00000013/uni00000011/uni00000017/uni00000014/uni00000016
/uni00000013/uni00000011/uni00000017/uni00000014/uni0000001b
/uni00000013/uni00000011/uni00000017/uni0000001c/uni0000001a
/uni00000013/uni00000011/uni00000017/uni00000018/uni00000016
/uni00000013/uni00000011/uni00000016/uni0000001c/uni0000001c
/uni00000013/uni00000011/uni00000017/uni00000014/uni0000001c
/uni00000013/uni00000011/uni00000014/uni00000019/uni00000015
/uni00000013/uni00000011/uni00000014/uni00000015/uni0000001a
/uni00000013/uni00000011/uni00000017/uni00000015/uni0000001c
/uni00000013/uni00000011/uni00000015/uni00000019/uni00000018
/uni00000013/uni00000011/uni00000016/uni0000001b/uni00000016
/uni00000013/uni00000011/uni00000016/uni00000013/uni00000013
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000014
/uni00000013/uni00000011/uni00000017/uni00000017/uni00000018/uni00000029/uni00000014/uni00000003/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000013/uni00000011/uni0000001b/uni00000018/uni00000013/uni00000011/uni0000001b/uni00000019/uni00000013/uni00000011/uni0000001b/uni0000001a/uni00000013/uni00000011/uni0000001b/uni0000001b/uni00000013/uni00000011/uni0000001b/uni0000001c/uni00000013/uni00000011/uni0000001c/uni00000013
/uni00000013/uni00000011/uni0000001b/uni0000001c/uni0000001a
/uni00000013/uni00000011/uni0000001b/uni0000001c/uni00000017
/uni00000013/uni00000011/uni0000001b/uni0000001c/uni00000016
/uni00000013/uni00000011/uni0000001b/uni0000001c/uni00000016
/uni00000013/uni00000011/uni0000001b/uni0000001c/uni00000015
/uni00000013/uni00000011/uni0000001b/uni0000001c/uni00000014
/uni00000013/uni00000011/uni0000001b/uni00000019/uni00000015
/uni00000013/uni00000011/uni0000001b/uni00000018/uni00000019
/uni00000013/uni00000011/uni0000001b/uni0000001b/uni00000013
/uni00000013/uni00000011/uni0000001b/uni00000019/uni00000013
/uni00000013/uni00000011/uni0000001b/uni0000001a/uni0000001b
/uni00000013/uni00000011/uni0000001b/uni0000001a/uni00000014
/uni00000013/uni00000011/uni0000001b/uni0000001b/uni0000001c
/uni00000013/uni00000011/uni0000001b/uni0000001b/uni00000019/uni00000025/uni00000028/uni00000035/uni00000037/uni00000036/uni00000046/uni00000052/uni00000055/uni00000048
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000013/uni00000011/uni0000001a
/uni00000013/uni00000011/uni00000018/uni0000001b/uni0000001b
/uni00000013/uni00000011/uni00000019/uni00000017/uni00000018
/uni00000013/uni00000011/uni00000019/uni00000018/uni0000001a
/uni00000013/uni00000011/uni00000019/uni0000001a/uni00000015
/uni00000013/uni00000011/uni00000018/uni00000014/uni00000016
/uni00000013/uni00000011/uni00000018/uni0000001a/uni0000001c
/uni00000013/uni00000011/uni00000019/uni00000016/uni00000013
/uni00000013/uni00000011/uni00000017/uni00000017/uni0000001b
/uni00000013/uni00000011/uni0000001a/uni00000016/uni0000001a
/uni00000013/uni00000011/uni00000019/uni00000013/uni00000019
/uni00000013/uni00000011/uni00000019/uni00000013/uni00000019
/uni00000013/uni00000011/uni00000018/uni00000017/uni00000016
/uni00000013/uni00000011/uni00000018/uni0000001b/uni00000018
/uni00000013/uni00000011/uni00000018/uni0000001b/uni00000018/uni00000035/uni00000032/uni00000038/uni0000002a/uni00000028/uni00000010/uni0000002f
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000016/uni00000011/uni00000018/uni00000010/uni00000057/uni00000058/uni00000055/uni00000045/uni00000052/uni00000026/uni0000004f/uni00000044/uni00000058/uni00000047/uni00000048/uni00000010/uni00000036/uni00000052/uni00000051/uni00000051/uni00000048/uni00000057/uni00000010/uni00000017
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016/uni00000030/uni0000004c/uni00000051/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001b/uni00000025
/uni00000030/uni0000004c/uni0000005b/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001b/uni0000005b/uni0000001a/uni00000025/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000016/uni00000013/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000017/uni00000025/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000015/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000016/uni00000015/uni00000025/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000016/uni00000010/uni0000001a/uni00000013/uni00000025
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000010/uni0000001b/uni00000025
From Retrieval to Reasoning: LLM Performance on Climate-specific QA.To evaluate how
retrieval-augmented LLMs translate historical climate records into actionable insights, we examine
their ability to synthesize retrieved passages into coherent, domain-specific answers. As shown in
Figure 3, performance varies across metrics. Claude-Sonnet-4 achieves the highest lexical fidelity
(BLEU: 0.141, F1: 0.497), while Qwen2.5-72B leads in semantic similarity (ROUGE-L: 0.737), re-
flecting stronger conceptual understanding despite lexical divergence. DeepSeek-V3 balances both
aspects, offering robust factual grounding and semantic coherence. Token-level metrics emphasize
the reliability of models with precise lexical alignment, whereas semantic evaluations reveal that
open-source frontier models are narrowing the gap with proprietary systems by capturing nuanced
reasoning needed for climate-specific queries. Overall, these results demonstrate that larger models
can effectively integrate retrieved passages, yet generating scientifically accurate answers remains
challenging, especially in the context of free-form climate-specific QA, with open-source systems
still scaling toward both lexical fidelity and contextual reasoning.
6 CONCLUSION
WEATHERARCHIVE-BENCHestablishes the first large-scale benchmark for evaluating the full RAG
pipeline on historical weather archives. By releasing a dataset of over one million archival news
segments, it enables climate scientists and the broader community to leverage historical data at
scale. With well-defined downstream tasks and evaluation protocols, the benchmark rigorously tests
both retrieval models and LLMs. In doing so, it transforms underutilized archival narratives into a
standardized resource for advancing climate-focused AI.
Our analyses reveal that hybrid retrieval approaches outperform dense methods on historical vo-
cabulary, while even proprietary LLMs remain limited in reasoning about vulnerabilities and socio-
environmental dynamics. Future research should address two identified challenges: (1) enhancing
retrieval methods to better handle historical vocabulary and narrative structures, and (2) improving
9

Work in progress.
models’ ability to reason about complex socio-environmental systems beyond surface-level factual
extraction. By offering a standardized evaluation resource, WEATHERARCHIVE-BENCHlays the
groundwork for future research toward AI systems that can translate historical climate experience
into actionable intelligence for adaptation and disaster preparedness.
ETHICSSTATEMENT
The WEATHERARCHIVE-BENCHis built from a collection of digitized historical newspapers pro-
vided through collaboration with an official organization, which remains anonymous at this stage.
This organization retains the copyright of the archival articles, but has granted permission to publish
the curated benchmark in support of the climate research community
Although the majority of extreme weather events in our dataset are recorded in North America, the
accounts capture how societies experienced and responded to climate hazards. These records provide
broadly relevant insights into resilience strategies and adaptation planning that extend beyond their
original geographical context. In addition, contributions from crowd-sourcing may be influenced
by geodemographic factors, which introduces variation but also enriches the dataset (Hendrycks
et al., 2020). As such, the benchmark reflects diverse societal perspectives on climate impacts and
responses, making it a valuable resource for studying adaptation strategies across societal contexts.
REPRODUCIBILITYSTATEMENT
The WEATHERARCHIVE-BENCHdataset is publicly available and fully reproducible. While we
cannot release the original newspaper print versions due to copyright restrictions, the post-OCR
documents used in our benchmark are included in the supplementary material. Our complete code-
base, including the data preprocessing pipeline and model evaluation scripts, is accessible through
an anonymous GitHub repository. Reproducing experiments that involve proprietary components
requires API keys for external services.
REFERENCES
Parul Awasthy, Aashka Trivedi, Yulong Li, Meet Doshi, Riyaz Bhat, Vishwajeet Kumar, Yushu
Yang, Bhavani Iyer, Abraham Daniels, Rudra Murthy, et al. Granite embedding r2 models.arXiv
preprint arXiv:2508.21085, 2025.
Yang Bai, Xiaoguang Li, Gang Wang, Chaoliang Zhang, Lifeng Shang, Jun Xu, Zhaowei Wang,
Fangshan Wang, and Qun Liu. Sparterm: Learning term-based sparse representation for fast text
retrieval.arXiv preprint arXiv:2010.00768, 2020.
Adrian Bingham. The digitization of newspaper archives: Opportunities and challenges for histori-
ans.Twentieth Century British History, 21(2):225–231, 2010.
Joern Birkmann, Susan L Cutter, Dale S Rothman, Torsten Welle, Matthias Garschagen, Bas
Van Ruijven, Brian O’neill, Benjamin L Preston, Stefan Kienberger, Omar D Cardona, et al.
Scenarios for vulnerability: opportunities and constraints in the context of climate change and
disaster risk.Climatic Change, 133(1):53–68, 2015.
Vincenzo Bollettino, Tilly Alcayna-Stevens, Manasi Sharma, Philip Dy, Phuong Pham, and Patrick
Vinck. Public perception of climate change and disaster preparedness: Evidence from the philip-
pines.Climate Risk Management, 30:100250, 2020.
Stefan Br ¨onnimann, Olivia Martius, Christian Rohr, David N Bresch, and Kuan-Hui Elaine Lin.
Historical weather data for climate risk assessment.Annals of the New York Academy of Sciences,
1436(1):121–137, 2019.
Nick Brooks, W Neil Adger, and P Mick Kelly. The determinants of vulnerability and adaptive
capacity at the national level and the implications for adaptation.Global environmental change,
15(2):151–163, 2005.
10

Work in progress.
Nitay Calderon, Roi Reichart, and Rotem Dror. The alternative annotator test for llm-as-a-
judge: How to statistically justify replacing human annotators with llms.arXiv preprint
arXiv:2501.10970, 2025.
Mark Carey. Climate and history: a critical review of historical climatology and climate change
historiography.Wiley Interdisciplinary Reviews: Climate Change, 3(3):233–249, 2012.
Daniel Feldmeyer, Daniela Wilden, Christian Kind, Theresa Kaiser, R ¨udiger Goldschmidt, Chris-
tian Diller, and J ¨orn Birkmann. Indicators for monitoring urban climate change resilience and
adaptation.Sustainability, 11(10):2931, 2019.
Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and St ´ephane Clinchant. Splade v2:
Sparse lexical and expansion model for information retrieval.arXiv preprint arXiv:2109.10086,
2021.
Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob
Steinhardt. Aligning ai with shared human values.arXiv preprint arXiv:2008.02275, 2020.
Ricardo Garc ´ıa Herrera, Rolando R Garc ´ıa, M Rosario Prieto, Emiliano Hern ´andez, Luis Gimeno,
and Henry F D ´ıaz. The use of spanish historical archives to reconstruct climate variability.Bull.
Am. Meteorol. Soc., 84(8):1025–1036, August 2003.
Sebastian Hofst ¨atter, Sophia Althammer, Michael Schr ¨oder, Mete Sertkan, and Allan Hanbury. Im-
proving efficient neural ranking models with cross-architecture knowledge distillation.arXiv
preprint arXiv:2010.02666, 2020.
Walter Jetz, Gavin H Thomas, Jeffery B Joy, Klaas Hartmann, and Arne O Mooers. The global
diversity of birds in space and time.Nature, 491(7424):444–448, 2012.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language
models.arXiv preprint arXiv:2001.08361, 2020.
Adriana Keating, Karen Campbell, Reinhard Mechler, Piotr Magnuszewski, Junko Mochizuki, Wei
Liu, Michael Szoenyi, and Colin McQuistan. Disaster resilience: what it is and how it can en-
gender a meaningful change in development policy.Development Policy Review, 35(1):65–91,
2017.
Ilan Kelman, Jean-Christophe Gaillard, James Lewis, and Jessica Mercer. Learning from the history
of disaster vulnerability and resilience research and practice for climate change.Natural Hazards,
82(Suppl 1):129–143, 2016.
Murathan Kurfalı, Shorouq Zahra, Joakim Nivre, and Gabriele Messori. Climate-eval: A compre-
hensive benchmark for nlp tasks related to climate change.arXiv preprint arXiv:2505.18653,
2025.
Finn Laurien, Juliette GC Martin, and Sara Mehryar. Climate and disaster resilience measurement:
Persistent gaps in multiple hazards, methods, and practicability.Climate Risk Management, 37:
100443, 2022.
Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gus-
tavo Hern ´andez ´Abrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, et al. Gemini embed-
ding: Generalizable embeddings from gemini.arXiv preprint arXiv:2503.07891, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
Jimmy Lin, Matt Crane, Andrew Trotman, Jamie Callan, Ishan Chattopadhyaya, John Foley, Grant
Ingersoll, Craig Macdonald, and Sebastiano Vigna. Toward reproducible baselines: The open-
source ir reproducibility challenge. InEuropean Conference on Information Retrieval, pp. 408–
420. Springer, 2016.
11

Work in progress.
Martina K Linnenluecke, Andrew Griffiths, and Monika Winn. Extreme weather events and the crit-
ical importance of anticipatory adaptation and organizational resilience in responding to impacts.
Business strategy and the Environment, 21(1):17–32, 2012.
Yang Liu, Jiahuan Cao, Chongyu Liu, Kai Ding, and Lianwen Jin. Datasets for large language
models: A comprehensive survey.arXiv preprint arXiv:2402.18041, 2024.
Tanwi Mallick, John Murphy, Joshua David Bergerson, Duane R Verner, John K Hutchison, and
Leslie-Anne Levy. Analyzing regional impacts of climate change using natural language process-
ing techniques.arXiv preprint arXiv:2401.06817, 2024.
James J McCarthy.Climate change 2001: impacts, adaptation, and vulnerability: contribution
of Working Group II to the third assessment report of the Intergovernmental Panel on Climate
Change, volume 2. Cambridge university press, 2001.
Fengran Mo, Kelong Mao, Yutao Zhu, Yihong Wu, Kaiyu Huang, and Jian-Yun Nie. Convgqr:
Generative query reformulation for conversational search. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4998–
5012, 2023.
Fengran Mo, Yifan Gao, Chuan Meng, Xin Liu, Zhuofeng Wu, Kelong Mao, Zhengyang Wang, Pei
Chen, Zheng Li, Xian Li, et al. Uniconv: Unifying retrieval and response generation for large
language models in conversations. InProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pp. 6936–6949, 2025.
Junko Mochizuki, Adriana Keating, Wei Liu, Stefan Hochrainer-Stigler, and Reinhard Mechler. An
overdue alignment of risk and resilience? a conceptual contribution to community resilience.
Disasters, 42(2):361–391, 2018.
Juan P Montoya-Rincon, Said A Mejia-Manrique, Shams Azad, Masoud Ghandehari, Eric W Harm-
sen, Reza Khanbilvardi, and Jorge E Gonzalez-Cruz. A socio-technical approach for the assess-
ment of critical infrastructure system vulnerability in extreme weather events.Nature Energy, 8
(9):1002–1012, 2023.
Rebecca E Morss, Olga V Wilhelmi, Gerald A Meehl, and Lisa Dilling. Improving societal out-
comes of extreme weather in a changing climate: an integrated perspective.Annual Review of
Environment and Resources, 36(1):1–25, 2011.
Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qim-
ing Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. Text and code embeddings by
contrastive pre-training.arXiv preprint arXiv:2201.10005, 2022.
Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. Climax:
A foundation model for weather and climate.arXiv preprint arXiv:2301.10343, 2023.
Geoff O’Brien, Phil O’keefe, Joanne Rose, and Ben Wisner. Climate change and disaster manage-
ment.Disasters, 30(1):64–80, 2006.
Karen O’Brien, Linda Sygna, and Jan Erik Haugen. Vulnerable or resilient? a multi-scale assessment
of climate impacts and vulnerability in norway.Climatic change, 64(1):193–225, 2004.
Huitong Pan, Mustapha Adamu, Qi Zhang, Eduard Dragut, and Longin Jan Latecki. Climateie:
A dataset for climate science information extraction. InProceedings of the 2nd Workshop on
Natural Language Processing Meets Climate Change (ClimateNLP 2025), pp. 76–98, 2025.
Michał Perełkiewicz and Rafał Po ´swiata. A review of the challenges with massive web-mined
corpora used in large language models pre-training. InInternational Conference on Artificial
Intelligence and Soft Computing, pp. 153–163. Springer, 2024.
Ashok K Rathoure. Vulnerability and risks.Intelligent Solutions to Evaluate Climate Change
Impacts, pp. 239, 2025.
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-
networks.arXiv preprint arXiv:1908.10084, 2019.
12

Work in progress.
Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and be-
yond.Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.
Pradeep Kumar Roy and Shivam Chahar. Fake profile detection on social networking websites: a
comprehensive review.IEEE Transactions on Artificial Intelligence, 1(3):271–285, 2021.
Ren´ee Sieber, Victoria Slonosky, Linden Ashcroft, and Christa Pudmenzky. Formalizing trust in
historical weather data.Weather Clim. Soc., 14(3):993–1007, July 2022.
Renee Sieber, Frederic Fabry, Victoria Slonosky, Muchen Wang, and Yumeng Zhang. Identifying
societal vulnerabilities and resilience related to weather using newspapers and artificial intelli-
gence. In104th Annual AMS Meeting 2024, volume 104, pp. 440060, 2024.
Victoria Slonosky and Ren ´ee Sieber. Building a traceable and sustainable historical climate
database: Interdisciplinarity and DRAW.Patterns (N. Y.), 1(1):100012, April 2020.
Barry Smit, Ian Burton, Richard JT Klein, and Johanna Wandel. An anatomy of adaptation to
climate change and variability.Climatic change, 45(1):223–251, 2000.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin,
and Zhaochun Ren. Is chatgpt good at search? investigating large language models as re-ranking
agents.arXiv preprint arXiv:2304.09542, 2023.
Zhen Tan, Dawei Li, Song Wang, Alimohammad Beigi, Bohan Jiang, Amrita Bhattacharjee, Man-
sooreh Karami, Jundong Li, Lu Cheng, and Huan Liu. Large language models for data annotation
and synthesis: A survey.arXiv preprint arXiv:2402.13446, 2024.
Nandan Thakur, Nils Reimers, Andreas R ¨uckl´e, Abhishek Srivastava, and Iryna Gurevych. Beir: A
heterogenous benchmark for zero-shot evaluation of information retrieval models.arXiv preprint
arXiv:2104.08663, 2021.
David Thulke, Yingbo Gao, Petrus Pelser, Rein Brune, Rricha Jalota, Floris Fok, Michael Ramos,
Ian Van Wyk, Abdallah Nasir, Hayden Goldstein, et al. Climategpt: Towards ai synthesizing
interdisciplinary research on climate change.arXiv preprint arXiv:2401.09646, 2024.
Ruth H Thurstan, Sarah M Buckley, and John M Pandolfi. Oral histories: informing natural resource
management using perceptions of the past. InPerspectives on Oceans Past, pp. 155–173. Springer,
2016.
Saeid Ashraf Vaghefi, Dominik Stammbach, Veruska Muccione, Julia Bingler, Jingwei Ni, Mathias
Kraus, Simon Allen, Chiara Colesanti-Senni, Tobias Wekhof, Tobias Schimanski, et al. Chatcli-
mate: Grounding conversational ai in climate science.Communications Earth & Environment, 4
(1):480, 2023.
Jesper Verhoef et al. The cultural-historical value of and problems with digitized advertisements:
Historical newspapers and the portable radio, 1950–1969.TS: Tijdschrift Voor Tijdschriftstudies,
38:51–60, 2015.
Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep self-
attention distillation for task-agnostic compression of pre-trained transformers.Advances in neu-
ral information processing systems, 33:5776–5788, 2020.
Duncan Watson-Parris, Yuhan Rao, Dirk Olivi ´e, Øyvind Seland, Peer Nowack, Gustau Camps-
Valls, Philip Stier, Shahine Bouabid, Maura Dewey, Emilie Fons, et al. Climatebench v1. 0: A
benchmark for data-driven climate projections.Journal of Advances in Modeling Earth Systems,
14(10):e2021MS002954, 2022.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua David Bergerson, John K Hutchison, Duane R
Verner, Jordan Branham, M Ross Alexander, Robert B Ross, Yan Feng, et al. Wildfiregpt: Tailored
large language model for wildfire analysis.arXiv preprint arXiv:2402.07877, 2024.
13

Work in progress.
Yangxinyu Xie, Bowen Jiang, Tanwi Mallick, Joshua David Bergerson, John K Hutchison, Duane R
Verner, Jordan Branham, M Ross Alexander, Robert B Ross, Yan Feng, et al. A rag-based multi-
agent llm system for natural hazard resilience and adaptation.arXiv preprint arXiv:2504.17200,
2025.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed,
and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text
retrieval.arXiv preprint arXiv:2007.00808, 2020.
Luo Xu, Kairui Feng, Ning Lin, ATD Perera, H Vincent Poor, Le Xie, Chuanyi Ji, X Andy Sun,
Qinglai Guo, and Mark O’Malley. Resilience of renewable power systems under climate risks.
Nature Reviews Electrical Engineering, 1(1):53–66, 2024.
Puxuan Yu, Luke Merrick, Gaurav Nuti, and Daniel Campos. Arctic-embed 2.0: Multilingual re-
trieval without compromise.arXiv preprint arXiv:2412.04506, 2024.
Yongan Yu, Qingchen Hu, Xianda Du, Jiayin Wang, Fengran Mo, and Ren ´ee Sieber. WXIm-
pactBench: A disruptive weather impact understanding benchmark for evaluating large language
models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar
(eds.),Findings of the Association for Computational Linguistics: ACL 2025, pp. 4016–4035, Vi-
enna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-256-
5. doi: 10.18653/v1/2025.findings-acl.207. URLhttps://aclanthology.org/2025.
findings-acl.207/.
Yongan Yu, Mengqian Wu, Yiran Lin, and Nikki G Lobczowski. Think: Can large language models
think-aloud?arXiv preprint arXiv:2505.20184, 2025b.
Long Yuan, Fengran Mo, Kaiyu Huang, Wenjie Wang, Wangyuxuan Zhai, Xiaoyu Zhu, You Li, Ji-
nan Xu, and Jian-Yun Nie. Omnigeo: Towards a multimodal large language models for geospatial
artificial intelligence.arXiv preprint arXiv:2503.16326, 2025.
Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong,
and Xia Hu. Data-centric artificial intelligence: A survey.ACM Computing Surveys, 57(5):1–42,
2025.
G Zuccaro, MF Leone, and C Martucci. Future research and innovation priorities in the field of
natural hazards, disaster risk reduction, disaster risk management and climate change adaptation:
A shared vision from the espresso project.International Journal of Disaster Risk Reduction, 51:
101783, 2020.
14

Work in progress.
APPENDIX
A OCR CORRECTIONQUALITYVALIDATION
Given the substantial noise inherent in OCR-digitized historical newspaper text, post-OCR correc-
tion was implemented as a critical preprocessing step to ensure corpus quality suitable for down-
stream applications. We employed GPT-4o with customized prompts to systematically correct OCR
errors that could significantly impact text comprehension and retrieval performance.
A.1 CORRECTIONPROCESS
The post-OCR correction targeted several categories of common OCR artifacts:
• Character recognition errors (e.g., ”COntInuInQ”→”continuing”).
• Removal of extraneous artifacts and formatting noise.
• Standardization of spacing and punctuation.
A.2 POST-OCR QUALITYASSESSMENT
To validate the effectiveness of our automated correction pipeline, we conducted a systematic eval-
uation comparing GPT-4o’s corrections against expert human annotations on a randomly selected
sample of 50 articles from the corpus.
Metric n-gram Score
BLEU 1-gram 0.911
2-gram 0.853
3-gram 0.817
ROUGE 1-gram 0.947
2-gram 0.919
L 0.943
Table 4: OCR correction quality comparing GPT-4o output against human annotations (n=50).
The consistently high scores across both BLEU and ROUGE metrics demonstrate strong alignment
between automated and human corrections, validating the reliability of our preprocessing approach.
These results confirm that the GPT-4o correction pipeline successfully preserves semantic content
while eliminating OCR artifacts that would otherwise compromise retrieval accuracy and down-
stream task performance.
B HISTORICALWEATHERARCHIVEINFORMATION
B.1KEYWORDS IN WEATHER ARCHIVE
Our keyword taxonomy was developed through consultation with climate historians and concluded
four primary categories that capture the multifaceted nature of weather-related archival content:
•Natural Disaster Terms:Direct references to weather-related hazardous events
•Climate Phenomena:Meteorological conditions and atmospheric processes
•Geographic Context:Spatial references that contextualize weather impacts
•Societal Response:Human and institutional responses to weather disruption
The expanded keyword list is presented in Table 5. These terms were finally counted based on their
frequency in our historical archives to be selected forWeatherArchive-Retrievaltask creation.
15

Work in progress.
Natural Disaster Climate Geographic Related Support
Natural disaster Extreme weather Mountain area Emergency response
Earthquake Heavy rain Coastal region Evacuation
Flood Snowstorm River basin Rescue operation
Hurricane Hail Urban flooding Government aid
Tornado Drought Rural area Disaster relief
Storm Heat wave Forest region Support troops
Tsunami Cold wave Civil protection
Landslide Weather damage Humanitarian assistance
Wildfire
V olcanic eruption
Table 5: Keywords used for frequency-based ranking of weather-related passages in historical
archives.
B.2 KEYWORDS WORDCLOUD
Figure 4: Word cloud for keywords in weather archives
Figure 4 presents a word cloud visualization of the 335 curated weather archive passages. We used
NLTK library to remove all punctuation and stop-words, and generated using TF-IDF weighting to
visualize the distinctive terms characterizing weather-related disruptions and societal responses.
This visualization reveals several prominent semantic clusters: direct meteorological phenomena
(”storm,” ”rain,” ”snow,” ”water”), infrastructure and impact terms (”damage,” ”city,” ”power,”
”road”), temporal markers (”yesterday,” ”time,” ”year”), and societal elements (”people,” ”police,”
”reports”). Notably, the prominence of terms like ”yesterday” and other temporal references re-
flects the immediacy of news reporting, while location-specific terms such as ”montreal,” ”quebec,”
and geographic descriptors (”river,” ”island”) emphasize the regional focus of the Southern Que-
bec archives. The prevalence of both historical (”railway,” ”telegraph”) and modern infrastructure
terminology validates our dataset’s temporal span.
C QUERYGENERATION-WeatherArchive-Retrieval
We employed GPT-4o to generate climate-specific queries for each of the 335 curated gold passages.
The prompt template ensures query specificity and uniqueness by requiring that each question can
only be answered using information from its corresponding passage, and the example demonstrates
16

Work in progress.
how this approach creates challenging retrieval scenarios that require precise matching of quantita-
tive details and geographic references within noisy archival text.
Query Generation - Prompt
Given the following passage about{Weather Type}, generate a single, focused question that meets
these criteria:
1. Can be answered using ONLY the information in this passage
2. Focuses on the weather impact based on the given location
3. Is detailed and specific to this exact situation
4. Requires understanding the passage’s unique context
5. Cannot be answered by other similar passages about{Weather Type}
Passage:
{Actual Passage}
Query Generation - Example Result
What was the economic impact of the July 1996 flooding in the Saguenay, Quebec, on the
insurance industry, and how did one company’s commercial-property claims contribute to this
situation?
The most expensive winter storm recorded in Canada from an insurance perspective was a
March 1991 tornado that tore through Sarnia, Ont., and caused $25 million worth of insurable
damage. None of Montreal’s previous ice storms made the insurance bureau’s list of the most
costly Canadian natural disasters. ”Two people who work with me had tree branches crash through
their roofs,” Medza said the heaviest storm damage occurs during the summer months, when high
winds can cause tornadoes and excessive rain brings sewer backups and floods homes. Of the 48
most costly Canadian storms, 31 of them occurred during the months of July and August. But it
was on Sept. 7, 1991, that a severe hail storm rained down on the city of Calgary, causing $342
million in insurable damage to homes and cars - the highest total recorded by any natural disaster in
Canada’s history. Quebec’s worst insurance bill was for $212 million after the July 1996 flooding
in the Saguenay - although about $108 million of that total came from one company reporting three
commercial-property claims. The cost of disaster Most expensive storms Cities Claims Amount paid
Calgary, Alta. (hail) Sept.
17

Work in progress.
D DIACHRONICSHIFTS INWORDUSAGE
Word Definition WeatherArchive samples current use Note
inundation“Great flood,
overflow.”
Collocations:
inundation
committee.The scheme was dropped as
of no use, and the chairman
of theinundationcommittee
got home again. NOW THE
RIVER. Saturday night’s ad-
vices from down river ports
were that at Three Rivers the
lake ice was on the move
since 2 p.m.flood,
floodingModern English
prefers flood;
inundationis now
rare outside
technical reports.
tempest“Violent
storm,” with
literary flavor.A temporary panic was
caused among the audience
at the Port St. Martin Theatre
by the sudden quenching
of the gas light. Cries were
raised of “Turn off the gas.”
The slamming of doors by
the wind and the roar of the
tempestdrowned the voices
of the actors.storm,severe
storm,
hurricane,
cycloneNeutral, precise
words dominate in
reporting;tempest
is archaic/poetic.
Table 6: Comparison of historical wordsinundation/tempestand their modern equivalents.
E FACTORSINFLUENCINGSOCIETALVULNERABILITY ANDRESILIENCE
Table 7: Factors influencing climate vulnerability. (Source: (Smit et al., 2000; McCarthy, 2001;
Feldmeyer et al., 2019))
ExposureThe degree of climate stress upon a particular unit of analysis. Climate stress
can refer to long-term changes in climate conditions or to changes in climate
variability and the magnitude and frequency of extreme events.
SensitivityThe degree to which a system will respond, either positively or negatively, to
a change in climate. Climate sensitivity can be considered a precondition for
vulnerability: the more sensitive an exposure unit is to climate change, the
greater are the potential impacts, and hence the more vulnerable.
AdaptabilityThe capacity of a system to adjust in response to actual or expected climate
stimuli, their effects, or impacts. The latest IPCC report (McCarthy et al.,
2001, p. 8) identifies adaptive capacity as “a function of wealth, technology,
education, information, skills, infrastructure, access to resources, and stability
and management capabilities”.
The vulnerability and resilience frameworks presented in Tables 7 and 8 provide theoretically
grounded and empirically validated dimensions for assessing climate impacts, as demonstrated in
prior studies (Kelman et al., 2016). These frameworks are particularly well-suited for evaluating
LLMs because they translate abstract concepts into concrete, observable indicators that can be sys-
tematically identified in historical texts. Moreover, their hierarchical structure enables fine-grained
assessment of model capabilities, ranging from surface-level factual extraction (e.g., identifying
damaged infrastructure) to more complex reasoning about systemic interactions (e.g., understanding
cross-scale governance coordination). The frameworks’ established validity in climate research en-
sures that WEATHERARCHIVE-BENCHevaluates genuine climate reasoning capabilities rather than
arbitrary categorization tasks. The entire structure is subsequently converted into structured prompts
18

Work in progress.
through chain-of-thought prompting, encouraging models to “think aloud.” Such instructions guide
the LLM to generate richer, more climate-relevant insights, which are then used to classify the in-
dicators. This approach aligns with findings from Chain-of-Thought and related studies (Wei et al.,
2022; Yu et al., 2025b).
Table 8: Factors influencing climate and disaster resilience. (Source: (Laurien et al., 2022;
Keating et al., 2017; Mochizuki et al., 2018))
Temporal ScaleResilience capacity across time horizons:short-term absorptive capacity
(immediate emergency response and coping mechanisms),medium-term
adaptive capacity(incremental adjustments and learning processes), and
long-term transformative capacity(fundamental system restructuring and
innovation for sustained resilience under changing conditions).
Functional ScaleResilience across critical infrastructure and service systems:health(med-
ical services and public health systems),energy(power generation and
distribution),food(agricultural systems and food security),water(supply
and sanitation systems),transportation(mobility and logistics networks),
andinformation(communication and data systems).
Spatial ScaleResilience across governance and geographic levels:local(community
and municipal capacity),regional(provincial and multi-municipal coordi-
nation), andnational(federal policies and cross-regional resource alloca-
tion and coordination mechanisms).
F WEATHERARCHIVE-ASSESSMENTORACLEGENERATION AND
VALIDATION
TheWeatherArchive-Assessmenttask requires reliable ground-truth oracles to evaluate model per-
formance on classifying societal vulnerability and resilience indicators. To construct these oracles,
we employ GPT-4.1 with carefully designed structured prompting, enabling consistent extraction of
categorical labels from archival narratives.
To validate the reliability of GPT-4.1–generated oracles, we recruited four independent domain ex-
perts. Each annotator was assigned a disjoint subset of 60 rows, ensuring diverse coverage of the
dataset. The resulting annotations yield an overall inter-annotator agreement ofκ Fleiss = 0.67, which
is generally interpreted as substantial agreement in the literature. Disagreements among annotators
were subsequently resolved through expert adjudication to establish high-quality reference labels.
Accuracy(f) =1
nnX
i=11{f(x i) =y i}= 0.82
ω=1
mmX
j=11{H 0jis rejected},whereω= 0.75>0.5
We further assess the alignment between GPT-4.1 and the expert annotations. GPT-4.1 achieves an
accuracy of0.82when compared against the adjudicated ground truth, highlighting its effective-
ness in streamlining the annotation process. To strengthen this validation, we adopt the statistical
framework proposed by Calderon et al. (2025), with parametersϵ= 0.05andα= 0.05, to rig-
orously compare GPT-4.1 outputs against human judgments. The evaluation shows a winning rate
ofω= 0.75, meaning GPT-4.1 outperforms the majority of human annotators in 75% of pairwise
comparisons. Sinceω >0.5, this result confirms that GPT-4.1 not only matches but surpasses aver-
age human performance in generating oracles, thereby validating their quality for use in benchmark
evaluation.
19

Work in progress.
G PROMPT DESIGN-WeatherAchive-Assessment
G.1 ORACLEANSWERGENERATION
G.1.1 PROMPTS
Oracle Answer - Prompt
You are a climate vulnerability and resilience expert. Implement a comprehensive assessment following the
IPCC vulnerability framework and multi-scale resilience analysis.
VULNERABILITY FRAMEWORK:
- **Exposure**: Characterize the type of climate or weather hazard.
• Sudden-Onset→Rapid shocks such as storms, floods, cyclones, or flash droughts
• Slow-Onset→Gradual stresses such as sea-level rise, prolonged droughts, or heatwaves
• Compound→Multiple interacting hazards (e.g., hurricane + flooding + infrastructure failure)
- **Sensitivity**: Evaluate how strongly the system is affected by the hazard.
• Critical→Highly dependent on vulnerable resources; likely severe disruption
• Moderate→Some dependence, but buffers exist; disruption noticeable but not catastrophic
• Low→Minimal dependence on hazard-affected resources; relatively insulated
- **Adaptability**: Determine the system’s capacity to respond and recover.
• Robust→Strong governance, infrastructure, technology, and social capital; effective recovery likely
• Constrained→Some coping mechanisms exist but are limited, uneven, or short-lived
• Fragile→Very limited or no capacity to cope; likely overwhelmed without external aid or systemic transfor-
mation
RESILIENCE FRAMEWORK:
- **Temporal Scale**: Choose the primary focus among [short-term absorptive capacity (emergency responses)
— medium-term adaptive capacity (policy/infrastructure adjustments) — long-term transformative capacity
(systemic redesign/migration)]
- **Functional System Scale**: Classify the single most affected system based on evidence. Options: [health,
energy, food, water, transportation, information]. Consider redundancy, robustness, recovery time, and interde-
pendence.
- **Spatial Scale**: Choose the primary level among [local — regional — national]. Highlight capacity
differences across scales.
INSTRUCTIONS:
- Always classify using the provided categories only, citing evidence from the document chunk.
- Ensure all classifications and selections are supported by evidence.
INPUT:
Query: query
Retrieved Document Chunk: context
OUTPUT FORMAT (follow this exact structure):
Region: [Extract/infer geographic region]
Exposure: [Sudden-Onset — Slow-Onset — Compound]
Sensitivity: [Critical — Moderate — Low]
Adaptability: [Robust — Constrained — Fragile]
Temporal: [short-term absorptive capacity — medium-term adaptive capacity — long-term transformative
capacity]
Functional: [health — energy — food — water — transportation — information]
Spatial: [local — regional — national]
EXAMPLE OUTPUT:
Region: Montreal
Exposure: Slow-Onset
Sensitivity: Moderate
Adaptability: Robust
Temporal: medium-term adaptive capacity
Functional: energy
Spatial: regional
Only output in the exact format above, using the exact categories as instructed. Do not include any addi-
tional text.
20

Work in progress.
G.1.2 EXAMPLE
Oracle Answer - Example
query: What specific infrastructure and agricultural impact did the British steamer Canopus experi-
ence due to the heavy gales in the United Kingdom?”
passage: ”STORMY WEATHER Heavy gales over the United Kingdom Bourne weather on
the Atlantic Disastrous loss of cattle shipments London, February 18 The weather continues very
unsettled over the whole of the United Kingdom, and gales are reported at several stations The
heavy gale which has raged at Penzance for the past two days has somewhat abated The wind is now
blowing strongly from the southwest and the barometer marks 28.70 inches The gale is still blowing
at Liverpool, but it has moderated a little London, February 18 The British steamer Canopus, Captain
Horsfall, which arrived at Liverpool yesterday from Boston, lost her boats and 247 head of cattle, and
sustained other damages in consequence of heavy weather Sports and Pastimes Curling Stuarton, X8,
February 18 The curling match between the Truro and Stuarton clubs, which took place here today,
resulted in a victory for Stuarton, which places the club in the van as good curlers Quebec, February
18 The Quebec Curling Club Challenge Cup was played for at the rink, St Charles street today, by the
Montreal Caledonia Curling Club and the Quebec Curling Club The play was excellent on both sides,
Quebec winning by 18 shots FEDERALIST London, February 19 At 2 a.m. the following was the
score in the six days’ walking match: Brown 328, Hazael 280, and ””Limping”” Day 258, and going
splendidly AQUATIC.”
region: United Kingdom
exposure: Sudden-Onset
sensitivity: Critical
adaptability: Constrained
temporal: short-term absorptive capacity
functional: food
spatial: national
G.2 LLM GENERATIONGENERATION
The table below presents the prompt designed to let LLMs generate the insights of a disaster record
with some detailed instructions on how it should incorporate context into the response.
G.2.1 PROMPTS
LLM Generation - Prompt
You are an expert in climate vulnerability and resilience. Use the retrieved document chunks to
provide a comprehensive answer to the query.
INSTRUCTIONS:
- Consider all retrieved document chunks together before answering.
- Synthesize the information; do not repeat text verbatim unless quoting is necessary for evidence.
- Ensure that your answer is **directly relevant to the query** and grounded in the provided context.
- Avoid adding information not supported by the provided chunks.
- If the retrieved chunks are not relevant to the query, should claim there is no mention in the provided
contexts.
INPUT:
Query: query
Retrieved Document Chunks:
context
OUTPUT:
[Provide a concise, evidence-based answer to the query, integrating information from all chunks]
21

Work in progress.
G.2.2 EXAMPLE
LLM Generation -GPT-4OExample Result
query:What specific infrastructure and agricultural impact did the British steamer Canopus experience
due to the heavy gales in the United Kingdom?”
”STORMY WEATHER Heavy gales over the United Kingdom Bourne weather on the Atlantic
Disastrous loss of cattle shipments London, February 18 The weather continues very unsettled over
the whole of the United Kingdom, and gales are reported at several stations The heavy gale which has
raged at Penzance for the past two days has somewhat abated The wind is now blowing strongly from
the southwest and the barometer marks 28.70 inches The gale is still blowing at Liverpool, but it has
moderated a little London, February 18 The British steamer Canopus, Captain Horsfall, which arrived
at Liverpool yesterday from Boston, lost her boats and 247 head of cattle, and sustained other damages
in consequence of heavy weather Sports and Pastimes Curling Stuarton, X8, February 18 The curling
match between the Truro and Stuarton clubs, which took place here today, resulted in a victory for
Stuarton, which places the club in the van as good curlers Quebec, February 18 The Quebec Curling
Club Challenge Cup was played for at the rink, St Charles street today, by the Montreal Caledonia
Curling Club and the Quebec Curling Club The play was excellent on both sides, Quebec winning by
18 shots FEDERALIST London, February 19 At 2 a.m. the following was the score in the six days’
walking match: Brown 328, Hazael 280, and ””Limping”” Day 258, and going splendidly AQUATIC.
region:United Kingdom
exposure:Sudden-Onset
sensitivity:Critical
adaptability:Constrained
temporal:short-term absorptive capacity
functional:transportation
spatial:regional
G.3 ANSWERCOVERAGEEVALUATION
The below table presents the prompt designed to compare standard answers with model-generated
answers to ensure that the model output covers the core information of the standard answer.
G.3.1 PROMPTS
Coverage Evaluation Instructions
You are an expert evaluator. Compare the oracle answer with the generated answer and determine if
the generated answer COVERS the key information stated in the oracle answer.
Oracle Answer: oracle answer
Generated Answer: generated answer
Task: Determine if the generated answer COVERS the key information from the oracle an-
swer.
Consider:
- Does the generated answer contain the main points from the oracle answer?
- Is the information accurate and relevant?
- Does it address the same question/topic?
Output ONLY: ”true” if it covers, ”false” if it doesn’t cover.
22

Work in progress.
H COMPUTATIONALCOSTS
For large proprietary models, a one-time evaluation on ourWeatherArchive-Assessmentbenchmark
costs approximately $1.2 for GPT-4o, $2.0 for Gemini-2-Pro, and $3.6 for Claude-Sonnet. For
open-source models, evaluations were conducted on a system equipped with four NVIDIA RTX
4090 GPUs (156GB for each GPU). The relatively modest computational requirements highlight
the accessibility of our benchmark to researchers with limited resources, while still supporting com-
prehensive evaluation of state-of-the-art models.
I ANNOTATIONPROCESS
We shuffled randomly selected 60 passages from this sample pool of 300 passages for manual an-
notation by our four annotators. After annotation, we calculated the Kappa coefficient between
annotation results to assess consistency. Simultaneously, we generated an overall annotation file by
comparing the annotations from all four annotators. For cases with high consensus among annota-
tors, the response from the annotator with the highest frequency was adopted as the final answer.
Where significant discrepancies existed, expert adjudication was used.
Finally, we compared this consolidated annotation with the responses generated by GPT-4.1 and
found an exceptionally high degree of alignment. Consequently, we conclude that the GPT-
generated results are highly credible and trustworthy.
J RETRIEVALMODELSSELECTED
1.Sparse retrievalWe include BM25 (BM25plus, BM25okapi) (Robertson et al., 2009) and
SPLADE (Formal et al., 2021). BM25 is a standard bag-of-words baseline that scores
documents using term frequency and inverse document frequency, implemented as a high-
dimensional sparse vector dot product. Following BEIR (Thakur et al., 2021), we use the
Anserini toolkit (Lin et al., 2016) with default Lucene parameters (k 1= 0.9,b= 0.4).
SPLADE extends sparse retrieval by predicting vocabulary-wide importance weights from
masked language model logits and applying contextualized expansion (Mo et al., 2023)
with sparse regularization (Bai et al., 2020), to capture semantically related terms beyond
exact token overlap.
2.Dense retrievalWe consider ANCE (Xiong et al., 2020), SBERT (Reimers & Gurevych,
2019), and large proprietary embeddings such as OpenAI’s text-embedding-ada-002 (Nee-
lakantan et al., 2022), Gemini’s text-embedding (Lee et al., 2025), IBM’s Granite Embed-
ding model (Awasthy et al., 2025) and Snowflake’s Arctic-Embed (Yu et al., 2024). These
models encode queries and passages into dense vectors and score relevance via inner prod-
uct. We adopt publicly available checkpoints for ANCE, SBERT, Arctic, while OpenAI
embeddings are queried via API. The inclusion of large proprietary models reflects the in-
creasing role of commercial LLM-derived embeddings in applied domain-specific retrieval
pipelines.
3.Re-ranking modelWe evaluate reranking models using cross-encoders (BM25plus+CE,
BM25okapi+CE) (Wang et al., 2020). Specifically, two BM25 models from Anserini first
retrieve the top 100 documents, after which a cross-attentional reranker jointly encodes the
query–document pairs to refine the ranking. Following Thakur et al. (2021), we employ
a 6-layer, 384-dimensional MiniLM (Wang et al., 2020) reranker model for our retrieval
task. The overall setup follows Hofst ¨atter et al. (2020).
23

Work in progress.
K ADDITIONALRESULTS
K.1 RETRIEVALEVALUATION
Table 9: Additional retrieval performance onWeatherArchive-Retrievalacross sparse, dense, and
re-ranking models. MRR@k results (k=1,3,5,10,50,100).
Category Model MRR@1 MRR@3 MRR@5 MRR@10 MRR@50 MRR@100
SparseBM25PLUS0.3701 0.4662 0.4796 0.4843 0.4898 0.4903
BM25OKAPI0.3015 0.4095 0.4251 0.4346 0.4395 0.4400
SPLADE0.0388 0.0542 0.0624 0.0715 0.0842 0.0866
DenseSBERT0.1522 0.2075 0.2187 0.2265 0.2307 0.2314
ANCE0.1791 0.2498 0.2674 0.2802 0.2919 0.2931
ARCTIC0.3164 0.4114 0.4255 0.4362 0.4439 0.4452
GRANITE0.3134 0.4144 0.4338 0.4458 0.4543 0.4551
OPENAI-3-SMALL0.3164 0.4075 0.4236 0.4356 0.4443 0.4453
OPENAI-3-LARGE0.2866 0.3726 0.3883 0.4016 0.4120 0.4130
OPENAI-ADA-002 0.2985 0.3905 0.4123 0.4255 0.4339 0.4350
GEMINI-EMBEDDING-001 0.3403 0.4463 0.4651 0.4776 0.4865 0.4871
HybridBM25PLUS CE0.3821 0.4945 0.5136 0.5191 0.5217 0.5219
BM25OKAPI CE0.3851 0.4965 0.5140 0.5202 0.5229 0.5231
Table 10: Additional retrieval performance onWeatherArchive-Retrievalacross sparse, dense, and
re-ranking models. Recall@k results (k=1,3,5,10,50,100).
Category Model Recall@1 Recall@3 Recall@5 Recall@10 Recall@50 Recall@100
SparseBM25PLUS0.3701 0.5851 0.6418 0.6776 0.7910 0.8269
BM25OKAPI0.3015 0.5433 0.6119 0.6776 0.7940 0.8299
SPLADE0.0388 0.0746 0.1134 0.1821 0.4776 0.6448
DenseSBERT0.1522 0.2896 0.3403 0.4000 0.5015 0.5522
ANCE0.1791 0.3403 0.4209 0.5224 0.7791 0.8657
ARCTIC0.3164 0.5343 0.5940 0.6746 0.8209 0.9104
GRANITE0.3134 0.5463 0.6299 0.7194 0.8866 0.9463
OPENAI-3-SMALL0.3164 0.5164 0.5851 0.6776 0.8537 0.9194
OPENAI-3-LARGE0.2866 0.4806 0.5493 0.6507 0.8537 0.9224
OPENAI-ADA-002 0.2985 0.5104 0.6030 0.7015 0.8806 0.9552
GEMINI-EMBEDDING-001 0.3403 0.5731 0.6537 0.7493 0.9164 0.9582
HybridBM25PLUS CE0.3821 0.6388 0.7224 0.76120.8119 0.8269
BM25OKAPI CE0.3851 0.63880.7164 0.76120.8179 0.8299
Table 11: Additional retrieval performance onWeatherArchive-Retrievalacross sparse, dense, and
re-ranking models. NDCG@k results (k=1,3,5,10,50,100).
Category Model NDCG@1 NDCG@3 NDCG@5 NDCG@10 NDCG@50 NDCG@100
SparseBM25PLUS0.3701 0.4968 0.5205 0.5320 0.5573 0.5631
BM25OKAPI0.3015 0.4439 0.4721 0.4941 0.5190 0.5247
SPLADE0.0388 0.0595 0.0749 0.0970 0.1604 0.1875
DenseSBERT0.1522 0.2283 0.2489 0.2680 0.2897 0.2978
ANCE0.1791 0.2730 0.3055 0.3376 0.3937 0.4077
ARCTIC0.3164 0.4430 0.4679 0.4939 0.5274 0.5418
GRANITE0.3134 0.4482 0.4829 0.5119 0.5496 0.5592
OPENAI-3-SMALL0.3164 0.4356 0.4642 0.4938 0.5334 0.5441
OPENAI-3-LARGE0.2866 0.4004 0.4286 0.4613 0.5072 0.5183
OPENAI-ADA-002 0.2985 0.4213 0.4600 0.4918 0.5312 0.5434
GEMINI-EMBEDDING-001 0.3403 0.4790 0.5125 0.5432 0.5816 0.5884
HybridBM25PLUS CE0.3821 0.5316 0.56600.5788 0.5904 0.5928
BM25OKAPI CE0.3851 0.53300.5648 0.5795 0.5921 0.5940
24

Work in progress.
K.2 WEATHERUNDERSTANDINGEVALUATION
Table 12: Recall evaluation results on Vulnerability and Resilience dimensions across models (av-
erage row in gray, Average column in gray).
Vulnerability ResilienceAverage
Model Exposure Sensitivity Adaptability Temporal Functional Spatial
GPT-4O0.654 0.679 0.557 0.6750.6750.524 0.627
GPT-3.5-TURBO0.629 0.460 0.504 0.593 0.354 0.482 0.504
CLAUDE-OPUS-4-10.841 0.859 0.766 0.8620.636 0.672 0.773
CLAUDE-SONNET-40.8410.652 0.719 0.786 0.631 0.663 0.715
GEMINI-2.5-PRO0.806 0.672 0.722 0.786 0.631 0.652 0.711
DEEPSEEK-V3-671B 0.764 0.460 0.745 0.848 0.598 0.605 0.670
MINISTRAL-8B-IT 0.444 0.217 0.268 0.431 0.406 0.372 0.356
MIXTRAL-8X7B-IT 0.266 0.226 0.290 0.300 0.190 0.305 0.263
QWEN2.5-72B-IT 0.792 0.551 0.643 0.767 0.531 0.508 0.632
QWEN2.5-32B-IT 0.532 0.396 0.465 0.653 0.446 0.404 0.483
QWEN2.5-14B-IT 0.430 0.466 0.300 0.414 0.240 0.331 0.363
QWEN2.5-7B-IT 0.384 0.168 0.313 0.381 0.281 0.355 0.314
LLAMA-3.3-70B-IT 0.389 0.503 0.287 0.503 0.537 0.351 0.428
LLAMA-3-8B-IT 0.241 0.175 0.153 0.238 0.253 0.281 0.224
QWEN-3-30B 0.719 0.428 0.394 0.814 0.354 0.416 0.521
QWEN-3-4B 0.370 0.261 0.201 0.527 0.675 0.266 0.383
Average 0.572 0.463 0.481 0.588 0.459 0.465 0.514
Figure 5: Comparison of LLM free-form QA performance across ROUGE-1 and LLM-Judge met-
rics
/uni00000013/uni00000011/uni00000016/uni00000019/uni00000013/uni00000011/uni00000016/uni0000001b/uni00000013/uni00000011/uni00000017/uni00000013/uni00000013/uni00000011/uni00000017/uni00000015/uni00000013/uni00000011/uni00000017/uni00000017/uni00000013/uni00000011/uni00000017/uni00000019
/uni00000013/uni00000011/uni00000017/uni00000015/uni0000001c
/uni00000013/uni00000011/uni00000017/uni00000014/uni0000001b
/uni00000013/uni00000011/uni00000017/uni00000017/uni0000001c
/uni00000013/uni00000011/uni00000017/uni00000016/uni00000018
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000019
/uni00000013/uni00000011/uni00000017/uni00000014/uni0000001b
/uni00000013/uni00000011/uni00000016/uni0000001c/uni0000001a
/uni00000013/uni00000011/uni00000016/uni00000019/uni00000017
/uni00000013/uni00000011/uni00000017/uni00000018/uni0000001a
/uni00000013/uni00000011/uni00000016/uni0000001a/uni00000013
/uni00000013/uni00000011/uni00000017/uni00000014/uni00000017
/uni00000013/uni00000011/uni00000017/uni00000014/uni00000015
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000018
/uni00000013/uni00000011/uni00000017/uni00000014/uni00000019/uni00000035/uni00000032/uni00000038/uni0000002a/uni00000028/uni00000010/uni00000014
/uni00000013/uni00000011/uni00000016/uni00000017/uni00000013/uni00000011/uni00000016/uni00000019/uni00000013/uni00000011/uni00000016/uni0000001b/uni00000013/uni00000011/uni00000017/uni00000013/uni00000013/uni00000011/uni00000017/uni00000015
/uni00000013/uni00000011/uni00000016/uni0000001b/uni0000001c
/uni00000013/uni00000011/uni00000016/uni0000001a/uni00000019
/uni00000013/uni00000011/uni00000017/uni00000013/uni00000018
/uni00000013/uni00000011/uni00000016/uni0000001c/uni00000016
/uni00000013/uni00000011/uni00000016/uni00000019/uni00000017
/uni00000013/uni00000011/uni00000016/uni0000001b/uni00000013
/uni00000013/uni00000011/uni00000016/uni00000019/uni00000019
/uni00000013/uni00000011/uni00000016/uni00000017/uni00000015
/uni00000013/uni00000011/uni00000017/uni00000014/uni00000017
/uni00000013/uni00000011/uni00000016/uni00000016/uni0000001c
/uni00000013/uni00000011/uni00000016/uni0000001a/uni0000001b
/uni00000013/uni00000011/uni00000016/uni0000001a/uni00000019
/uni00000013/uni00000011/uni00000016/uni00000019/uni00000015
/uni00000013/uni00000011/uni00000016/uni0000001a/uni0000001c/uni0000002f/uni0000002f/uni00000030/uni00000010/uni0000002d/uni00000038/uni00000027/uni0000002a/uni00000028
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000017/uni00000052
/uni0000002a/uni00000033/uni00000037/uni00000010/uni00000016/uni00000011/uni00000018/uni00000010/uni00000057/uni00000058/uni00000055/uni00000045/uni00000052/uni00000026/uni0000004f/uni00000044/uni00000058/uni00000047/uni00000048/uni00000010/uni00000036/uni00000052/uni00000051/uni00000051/uni00000048/uni00000057/uni00000010/uni00000017
/uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000039/uni00000016/uni00000030/uni0000004c/uni00000051/uni0000004c/uni00000056/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001b/uni00000025
/uni00000030/uni0000004c/uni0000005b/uni00000057/uni00000055/uni00000044/uni0000004f/uni00000010/uni0000001b/uni0000005b/uni0000001a/uni00000025/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000016/uni00000013/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000017/uni00000025/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000015/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000016/uni00000015/uni00000025/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni00000014/uni00000017/uni00000025
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000015/uni00000011/uni00000018/uni00000010/uni0000001a/uni00000025/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000011/uni00000016/uni00000010/uni0000001a/uni00000013/uni00000025
/uni0000002f/uni0000004f/uni00000044/uni00000050/uni00000044/uni00000010/uni00000016/uni00000010/uni0000001b/uni00000025
L CASESTUDY
L.1 SNOWSTORM
In this example, the ground truth reveals the insights into infrastructure and financial losses resulting
from a snowstorm. The output of most LLMs is the same as the ground truth. However,Qwen-72B
labeled the sensitivity asmoderateinstead ofcritical.
Question:What specific financial loss did the Toronto Telephone Company incur due to the
snowstorm on February 21, and how did the storm affect their infrastructure?
Reference Answer:Trains were late today in consequence of the heavy snowstorm last night.
Toronto, February 21. The snowstorm early this morning did great damage. The wires of the
Toronto Telephone Company were blown down from the JO ail building. Their losses alone
25

Work in progress.
will amount to $1,000. Thorold, Ont, February 21. A storm of rain and sleet set in at midnight
last night, covering the ground about three inches thick. Travel not impeded. Peterborough,
Ont, February 21. Snow fell last night and this morning to the depth of four inches. A
snowstorm set in at three o’clock this afternoon. Sleighing good; travel unimpeded. Brampton,
Ont, February 21. A wild storm set in last night. The roads in the country are said to never have
been worse. Chatham, Ont, February 21. Weather very stormy today. Heavy northwest wind,
with snow. Roads muddy and almost impassable. Grimsby, February 21. The most violent
storm of the season is now prevailing here.
Ground Truth
region: Toronto,exposure: Sudden-Onset,sensitivity: Critical,adaptability: Constrained
temporal: short-term absorptive capacity,functional: information,spatial: local
gpt-4o result
region: Toronto,exposure: Sudden-Onset,sensitivity: Critical,adaptability: Constrained
temporal: short-term absorptive capacity,functional: information,spatial: local
Qwen-72B result
region: Toronto,exposure: Sudden-Onset,sensitivity: Moderate ,adaptability: Constrained
temporal: short-term absorptive capacity,functional: information,spatial: local
L.2 RAINSTORM
In this example, the ground truth answer emphasizes the profound impact on real estate speculators
and specific merchants caused by a record-breaking rainstorm. Most LLMs are able to provide
similar answers to the ground truth, while they may underestimate the extent of the disaster and
label the functional astransportation.
Question:How did the record-breaking rainstorm in Toronto impact the financial situation of
real estate speculators and specific merchants, and what were the estimated monetary losses
they incurred according to the passage?
Reference Answer:GOSSIP FROM TORONTO: A Record-Breaking Rainstorm Blared Real
Estate Speculators Tided Over Their Trouble From our own correspondent Toronto, June
20 The rainstorm which deluged the city last evening was of the record-breaking character
Nothing like it has been seen for some time The damage resulting is roughly estimated in
the neighborhood of $100,000 Merchants in the central portion of the city were the greatest
sufferers and Guinane’s boot store and Armson & Stone’s millinery emporium lose between
them $2,000 on Yonge street The shopkeepers on the south side of King street, between Yonge
and Bay streets, are very heavy losers, Caldwell & Hodgins, wine merchants, at Queen and
John streets, will probably be minus $5,000 The affairs of the big speculators in real estate
here, reported some time ago in difficulty, have been settled so as to avoid an assignment
One prominent and wealthy lawyer, who is very largely interested in real estate in and about
Toronto, has obtained a five years’ extension from banks on $800,000, on which he will have
to pay an interest of $48.
Ground Truth
region: Toronto,exposure: Sudden-Onset,sensitivity: Critical,adaptability: Constrained
temporal: short-term absorptive capacity,functional: information,spatial: local
gpt-4o result
region: Toronto,exposure: Sudden-Onset,sensitivity: Moderate ,adaptability: Constrained
temporal: short-term absorptive capacity,functional: information,spatial: local
Qwen-72B result
region: Toronto,exposure: Sudden-Onset,sensitivity: Critical,adaptability: Constrained
26

Work in progress.
temporal: short-term absorptive capacity,functional: transportation ,spatial: local
M THEUSE OFLARGELANGUAGEMODELS
This research study employed LLMs in two ways:
Data processing and quality assurance.LLMs were used for post-OCR correction of the histor-
ical weather archive corpus. Given the multi-decade scale of digitized reports, this stage required
approximately 0.3 billion input and output tokens to ensure accuracy and readability. OCR correc-
tion was necessary, as historical sources often contained scanning artifacts, unclear typography, and
degraded text that could impair retrieval.
Benchmark creation and evaluation.LLMs generate oracle annotations for theWeatherArchive-
Assessmentbenchmark following the protocols described in Appendix G.1. We also include gener-
ative AI models as testbed models, including both proprietary and open-source LLMs, to evaluate
their performance on climate-related downstream tasks in specialized domains.
All research design, data collection, and analysis were conducted solely by the authors without
LLM’s assistance.
27