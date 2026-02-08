# A Human-in-the-Loop, LLM-Centered Architecture for Knowledge-Graph Question Answering

**Authors**: Larissa Pusch, Alexandre Courtiol, Tim Conrad

**Published**: 2026-02-05 10:10:19

**PDF URL**: [https://arxiv.org/pdf/2602.05512v1](https://arxiv.org/pdf/2602.05512v1)

## Abstract
Large Language Models (LLMs) excel at language understanding but remain limited in knowledge-intensive domains due to hallucinations, outdated information, and limited explainability. Text-based retrieval-augmented generation (RAG) helps ground model outputs in external sources but struggles with multi-hop reasoning. Knowledge Graphs (KGs), in contrast, support precise, explainable querying, yet require a knowledge of query languages. This work introduces an interactive framework in which LLMs generate and explain Cypher graph queries and users iteratively refine them through natural language. Applied to real-world KGs, the framework improves accessibility to complex datasets while preserving factual accuracy and semantic rigor and provides insight into how model performance varies across domains. Our core quantitative evaluation is a 90-query benchmark on a synthetic movie KG that measures query explanation quality and fault detection across multiple LLMs, complemented by two smaller real-life query-generation experiments on a Hyena KG and the MaRDI (Mathematical Research Data Initiative) KG.

## Full Text


<!-- PDF content starts -->

A Human-in-the-Loop, LLM-Centered Architecture for
Knowledge-Graph Question Answering
Larissa Pusch
Department of Visual and Data-Centric Computing
Zuse Institute Berlin
Berlin, Germany
pusch@zib.deAlexandre Courtiol
Department of Evolutionary Genetics
Leibniz Institute for Zoo and Wildlife Research
Berlin, Germany
courtiol@izw-berlin.de
Tim Conrad
Department of Visual and Data-Centric Computing
Zuse Institute Berlin
Berlin, Germany
conrad@zib.de
February 6, 2026
Abstract
LargeLanguageModels(LLMs)excelatlanguageunderstandingbutremainlimitedinknowledge-
intensive domains due to hallucinations, outdated information, and limited explainability. Text-based
retrieval-augmented generation (RAG) helps ground model outputs in external sources but struggles
with multi-hop reasoning. Knowledge Graphs (KGs), in contrast, support precise, explainable
querying,yetrequireaknowledgeofquerylanguages. Thisworkintroducesaninteractiveframework
in which LLMs generate and explain Cyphergraph queries and users iteratively refine them through
natural language. Applied to real-world KGs, the framework improves accessibility to complex
datasetswhilepreservingfactualaccuracyandsemanticrigorandprovidesinsightintohowmodel
performance varies across domains. Our core quantitative evaluation is a 90-query benchmark on
a synthetic movie KG that measures query explanation quality and fault detection across multiple
LLMs, complemented by two smaller real-life query-generation experiments on a Hyena KG and the
MaRDI (Mathematical Research Data Initiative) KG.arXiv:2602.05512v1  [cs.CL]  5 Feb 2026

A preprint - February 6, 2026
1 Introduction
LargeLanguageModels(LLMs)havebecomeintegraltoquestionanswering(QA)tasksduetotheiradvancedlanguage
understanding and generation capabilities. Yet their utility is often limited in knowledge-intensive domains where
precision and interpretability are critical. In such settings, LLMs frequently suffer from outdated or hallucinated
knowledgeandmaygenerateaplausible-soundingbutincompleteorincorrectanswer[11]withoutclearlyindicating
the sources or reasoning steps underlying its conclusion [1].
Toaddresstheselimitations,LLM-basedretrieval-augmentedgeneration(RAG)techniquesgroundmodeloutputsin
externalinformationsourceswithoutincludingthesesourcesintheLLMtrainingprocess. However,becausemostRAG
systemsrelyontext-basedretrieval[21],theystruggletosupportmulti-hopreasoning-thatis,answeringquestions
that require combining information from multiple, indirectly connected sources. For example, determining which
researcherscitedapaperthatimplementedaspecificalgorithminvolveschainingthroughauthors,publications,and
software entries. Traditional semantic similarity search often fails to capture such indirect relationships, especially
when relevant information spans documents or concepts with little lexical overlap [12].
Knowledge Graphs (KGs), structured representations of entities and their relationships, offer an alternative. Unlike
unstructured text, KGs encode domain knowledge in a way that supports logical inference, schema evolution, and
semantic querying [7]. This makes them well-suited for applications requiring contextualized, verifiable answers, such
as natural sciences, biomedicine or mathematics. Search engines have long recognized these advantages: Google’s
Knowledge Graph rollout in 2012 improved ranking and disambiguation by resolving queries over graph entities instead
of raw keywords [19, 7].
Despite recent progress, KG–based retrieval-augmented generation (KG-RAG) pipelines can still be suboptimal. Three
design choices are particularly limiting:
1.Linearised triples erase graph structure. Some pipelines flatten a KG or its subgraph into a list of
subject–predicate–object triples and pass them to the language model verbatim or in rigid templates [22]. The
resultingtextisverbose,inflatesthecontextwindow,andforcesthemodeltoreconstructrelationalstructure
that was explicit in the original schema.
2.Black-box retrieval hides the reasoning path. Other systems embed the graph using a graph neural network
(GNN) and retrieve nodes or paths by using similarity measures [22]. While the retrieved items and their
similarity scores are inspectable, the transformation that maps the query to that vector, and thus the full
reasoning path, remains implicit and can only be approximated with post-hoc explainability tools. An explicit
SPARQLorCypherquery would expose an auditable, deterministic path from data to answer.
3.No user-in-the-loop control. Many architectures treat the user as a passive consumer: once retrieval and
generation start, there is no hook to correct a mistaken entity match, broaden or narrow the scope, or override
themodel’sdraftingchoices. Lackofsuchfeedbackchannelsreducestransparencyandflexibilityandweakens
trust.
Inourpreviouswork[16],weevaluatedLLMsontranslatingbiomedicalquestionsinto Cypherqueriesoverabiomedical
KG. We released a benchmark question set and an open-source interface, but the lack of means to debug, refine or
explain queries meant that any failure required a complete regeneration. This highlights a disconnect between user
intent and graph-based query execution.
Toclosethisgap,wenowproposeaninteractiveframeworkinwhichLLMsnotonlygenerategraphqueriesbutalso
explain them innatural language. Users can provide feedback through natural-languageamendments, which are used
to iteratively update the query. This creates a feedback loop that supports interpretability and precise control. The
explanation capabilities of the framework are evaluated on a simple movie Knowledge Graph, coupled with a structured
evaluation query set. Additionally, we conduct two case studies to evaluate the pipeline on several LLMs and two
heterogeneousgraphs: aHyenaKGandamathematicalresearchKGfromtheMathematicalResearchDataInitiative
(MaRDI) [18].
The Hyena KG was derived from a subset of data collected for the long-term monitoring of a free-ranging population of
spottedhyenasfromtheNgorongoroCraterinTanzania(see https://hyena-project.com ). Biologistshavebeen
studyingthispopulationfornearly30years,collectingawiderangeofdata(ondemography,behaviour,morphology,
physiology,diseasesandgenetics)toaddressfundamentalecologicalandevolutionaryquestions(e.g. Whydosome
individuals disperse to other social groups while others do not? [2]), as well as specific questions related to this
particular population (e.g., How does pastoralism impact these hyenas? [3]). As the composition of social groups
(called clans) is dynamic, and interactions between individuals are strongly influenced by relatedness, much of the work
onthehyenadatarequirestrackingthemovementsofindividualsacrossclansandtheiraffiliations. Thismeansthat
2

A preprint - February 6, 2026
researchers require in-depth knowledge of the data structure and expertise in database manipulation to answer even the
simplest questions. In order to reduce the barriers to entry for new researchers, members of the Ngorongoro Hyena
Projecthavedevelopedasubstantial Rpackage( hyenaR,withover450exportedfunctionsatthetimeofwriting). While
thistoolisalreadyhelpful,usersstillneedtobeproficientinthe Rlanguage,whichnotallbiologistsare. Moreover,
new projects tend to require additional functions, thereby increasing the development and maintenance burden for the
developers. Finally,atoolsuchas hyenaRcannoteasilybeadaptedforthestudyofotherlong-termprojectsbyother
research groups since much of the coding is tailored to the idiosyncratic structure of the data. A reliable LLM/KG
combination could bea more appealingalternativebecause it represents ageneric solution that wouldlargely remove
the programming burden while providing an interface accessible to all.
MaRDI’sopenKnowledgeGraphalreadyconsistsofover700milliontriples,exposingthemthroughpublic SPARQL
and REST endpoints. This infrastructure is designed to support FAIR principles in mathematics, enabling users to, for
example,traceatheoremtoitssoftwareimplementationorlinkbenchmarkdatasetstoallrelatedpublications. However,
the technical barrier posed by SPARQLcontinues to limit accessibility. Our approach, applied to a subgraph of MaRDI,
could remove that barrier without sacrificing the precision and interpretability provided by curated KGs.
TocomparemanyLLMsunderidenticalconditions,wekeepthequestionsetsfortheMaRDIandHyenaKGssmall.
This trades depth per KG for breadth across models. As a result, the study is optimized to reveal failure modes and
inconsistencies across models, rather than to provide an exhaustive ranking of overall performance.
We evaluate our system on the three KGs, measuring the faithfulness of natural-language explanations on the synthetic
movie KG and the accuracy of generated queries on the MaRDI and Hyena KG. This work contributes toward making
semantic graph querying more transparent, interactive, and accessible to non-experts.
Insummary,wedesignedasystemthatcombinestheexpressivenessof CypherwiththeaccessibilityofLLMs,explicitly
targeting the limitations identified in existing KG-based RAG systems. The key contributions of this work are:
1.Aninteractivenatural-languagequeryframeworkwithtransparentreasoningthatletsusersaskquestionsabout
the contents of a Knowledge Graph, inspect and refine the generated queries through conversational feedback,
andobtainnatural-languageexplanationsandplausibilitychecksthatmakethereasoningpathfromdatato
answer explicit while grounding responses in curated graph data, reducing the risk of hallucinations.
2.Empirical evaluation on synthetic (movie-domain), and domain-expert (MaRDI KG and Hyena KG) graphs to
assess explanation quality, fault detection and query generation.
3.InsightsintocommonLLMfailuremodeswhengeneratingCypherqueries,performancevariabilityacross
models, and the importance of measuring both accuracy and amendment efficiency.
The novelty of our approach lies in combining transparent LLM-generated queries with an interactive amendment loop
that lets users shape and audit the reasoning process without needing to write Cypherthemselves. In the following, we
present the architecture and main components of this framework, demonstrating how they work together to enable query
generation,execution,transparentexplanation,anditerativerefinement. Tomaketheframework’scapabilitiesmore
concrete,webeginwithanexampleinteractionthatillustratesthesefunctionalities. Thisalsoservestohighlightthe
importance of transparency and control in Knowledge Graph interaction.
1.1 Example Interaction
Before describing the internal components, we present an example interaction to give a concrete impression of the
framework in action. In this two-turn exchange, a user asks a natural-language question, the LLM generates the
corresponding graph ( Cypher) query, it gets executed on the graph database and results are returned. The example also
shows how the framework handles clarification and refinement when the initial interpretation of a query does not align
with the user’s intent.
3

A preprint - February 6, 2026
T1 User:Which authors does graphclust have?
LLM→Cypher
MATCH (p:Publication {name:"graphclust"})-[:HAS_AUTHOR]->(a:Author)
RETURN a.name;
DB→Result Empty Result because no Publication node called *graphclust* exists.
LLM (short explanation).
1. Locate the single Publication node whose nameequals “graphclust”.
2. Traverse every outgoing :HAS_AUTHOR edge to neighbouring Authornodes.
3. Return the nameof each such Author.
Purpose (one line): List all authors associated with the publication graphclust .
Plausibilitycheck: Thepatternissensible: publications“have”authorsandthelabels/edgename
fit the domain. Some schemas flip the edge Author:AUTHORED−−−−−−−−→ Publication, but the chosen
Publication→Authordirection is equally valid.
T2 User:Actually, I meant the software package , not the publication.
LLM→Cypher
MATCH (s:SoftwarePackage {name: 'graphclust '})-[:HAS_AUTHOR]->(a:Author)
RETURN a.name;
DB→Result Tabea Rebafka
Thisexamplehighlightstheframework’scorecapabilities: translatingnaturallanguageintostructuredqueries,explaining
the resulting logic, and enabling users to refine them through conversational feedback. This allows the system to bridge
thegapbetweennaturallanguageandformalquerysyntax,withoutrequiring userstolearnaquery language,suchas
Cypher.
These capabilities are supported by a modular architecture which allows for switching of LLMs or KGs. The next
sectionintroducesthisarchitectureanddescribesitscorecomponents: querygeneration,explanation,execution,and
refinement.
2 Framework Architecture
Toovercomethelimitations ofcurrentKG-based systems,and makegraph-structuredknowledge moreaccessible, we
developedaninteractiveframeworkthatallowsuserstoqueryagivenKnowledgeGraphusingonlynaturallanguage
(NL).Atitscoreisamodularpipelinethattranslatesnatural-languageuserquestionsintoexecutablequeries(Generator),
runsthemagainsttheunderlyinggraphdatabase(Executor),andreturnstheresults. Thesamelargelanguagemodel
(LLM)isalsousedtoexplainthegeneratedqueries(Explainer)andtoprocessfollow-upinstructionsforrefinement
(Amender). Unlike conventional KG-based retrieval systems, which often treat users as passive recipients of generated
answers, our framework emphasizes transparency and user control, which enables a conversational interaction with the
underlying structured graph data, without requiring users to write Cypherqueries.
This paper builds on our earlier work introduced in [16], introducing and evaluating the missing explanation and
amendment capabilities. These enhancements shift the system from a static query generator to an interactive dialogue
agent for graph exploration.
4

A preprint - February 6, 2026
Figure1: ArchitectureDiagram;PurpleNodesarecontributedbytheuser,orangenodesaremodules,theblueoneis
the graph schema and the green nodes are pipeline outputs.
2.1 Framework Main Components
Query Generation
The query generation component translates a user’s natural-language question into an executable Cypherquery. Its
primary goal is to preserve semantic intent while conforming to the graph’s structure and constraints.
Inputs and Outputs : The component takes the user’s question and the Knowledge Graph schema as input. Its output is
a cleaned, executable Cypherquery string.
Implementation Details : The translation is performed using a schema-aware LangChain prompt (subsection A.2) that
constrains vocabulary and structure to valid node and edge types. The LLM (e.g., GPT-4, Claude, or DeepSeek) is
wrappedina LangChain -compatibleclass,ensuringconsistentbehaviorandeasymodelswitching. Post-processing
removes markdown wrappers, tags like thinking, and other extraneous content, resulting in a syntactically correct query.
Query Execution
Thiscomponentconnectstoa Neo4jinstanceandexecutesthe Cypherqueryreturnedbythegenerator. Itisresponsible
for retrieving structured answers from the Knowledge Graph.
Inputs and Outputs : Its input is the cleaned Cypherquery along with a database connector. The output is a list of
structured records representing the query result.
Implementation Details : Execution is handled using LangChain ’sNeo4jGraph object, which provides an abstraction
over query submission and result retrieval. Results are returned as a Pythondata-structure, ready for display or further
processing.
Query Explanation
Toimprovetransparencyandusertrust,thiscomponentgenerateshuman-readableexplanationsofthe Cypherquery.
These explanations help users understand how a query works and why a particular result was produced.
InputsandOutputs : Ittakesa Cypherqueryasinputandreturnsastructuredexplanation,includingastep-by-step
walkthrough, error flagging and a concise summary.
Implementation Details : The explanation is generated using the same LLM infrastructure as the query generation
componentbutwithadifferentprompt(subsectionA.1). Thepromptinstructsthemodeltodescribethequery’spurpose
and logic in plain language. Additionally, the model is asked to flag issues such as inverted relationships, invalid node
types, or implausible graph patterns. This supports interpretability without requiring users to read Cypherdirectly.
Query Amendment
Theamendmentcomponentenablesiterativerefinementbyallowinguserstoupdateanexistingqueryusingnatural
language. Rather than regenerating the query from scratch, the system applies targeted modifications based on user
feedback.
5

A preprint - February 6, 2026
InputsandOutputs : Inputsincludetheexisting Cypherquery,theuser’samendmentinstruction,andtheschema. The
output is a revised Cypherquery string.
Implementation Details : Amendments are handled via a dedicated prompt (subsection A.3) that asks the LLM to edit
thecurrentqueryin accordancewiththenewinstruction whileadheringtoschema constraints. Thisapproachaims
toavoidovercorrectionandretainrelevantquerystructure. TheLLM’soutputisparsedandreturnedforimmediate
re-executionorexplanation. Thiscomponentenablesconversationalquerycorrection,allowinguserstoadjustfilters,
entity types, or traversal logic without starting over.
2.2 Summary
We havenowdescribedthekey componentsofthe frameworkandhowtheyinteract toform anend-to-endpipeline.
This allows for the translation of natural-language questions into executable graph queries, which can be run against a
knowledge graph, and the results then be returned to the user. More technically, a large language model generates a
Cypherquery, which is executed via the Neo4jdriver, and the resulting records are displayed in the user interface.
The same model can also provide a step-by-step natural-language explanation of the query or apply an amendment
instruction expressed in natural language, without requiring full regeneration.
Thisprocesswasillustratedinthe“graphclust”exampleabove, whereauserrefinestheirquerytodistinguishbetweena
publication and a software package. The interaction demonstrates how the framework supports clarification and allows
non-experts to explore graph-structured data without needing to read or write Cypher.
Withthefullframeworkinplace,wefirstfocusonourmostdetailedquantitativeexperiment,theexplanationcomponent
in a controlled benchmark, and then turn to two case studies on real-world KGs.
3 Experiment: How well can LLMs explain CypherQueries?
Toensurethatuserscantrustandcontrolthequeryprocess,theyneedtounderstandwhatagenerated Cypherquery
actually does. In this section, we evaluate whether Large Language Models (LLMs) can support that understanding by
providing accurate, natural-language explanations and by identifying flawed queries when they occur.
We focus on two main questions:
1. Can LLMs generate clear and accurate natural language explanations of Cypherqueries?
2. Can they detect and flag syntactic or semantic problems in queries?
Toisolatetheseabilitiesfromdomaincomplexity,weuseadeliberatelysimplesetting: asyntheticKnowledgeGraph
about movies. This familiar domain makes it easier to spot errors and ensures that poor performance is not due to
obscure content or schema.
For instance, a sample query might be:
MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)
WHERE m.release_year > 2000
RETURN a.name, m.title
A correct explanation would state that the query retrieves the names of actors and the titles of movies they acted in,
limitedtomoviesreleasedaftertheyear2000. Aninaccurateexplanationmightomittheyearconstraint. Wedeliberately
introduced errors into these sample queries, such as replacing the relationship name with one that does not make sense
inthecontext,suchas (a:Actor)-[EATS]->(m:Movie) ,ormoresubtlemistakes,suchasflippingthedirectionofthe
relationship.
With this context in mind, we will next describe the benchmark setup, the models tested, and the evaluation results.
6

A preprint - February 6, 2026
3.1 Benchmark Dataset
City
ACTED_IN DIRECTEDPerson
Movie HAS_F AVORITEBIRTH_CITY
Critic
Figure2: SchemaofthesyntheticMovieKnowledgeGraphcreatedasbasisforthequeriesinthebenchmarkdataset.
The node types are Person, Movie, Critic and City, the relationships are DIRECTED ,ACTED_IN ,HAS_FAVORITE and
BIRTH_CITY .
Wedesignedabenchmarkof90 Cypherqueries(seeAppendixC)setinamovie-domainKnowledgeGraph(Fig:2),
systematically varying along three dimensions: path lengths, clause types, and error injections.
Unlike existing datasets ([5], [13], [20]) which lack fine-grained control over query structure and do not contain
systematicallyintroducederrors, ourbenchmarkwas createdfromscratchto allowperturbationvariationand targeted
evaluation.
Theresultingquerysetcapturesadiverserangeofstructuralandsemanticpatterns. Below,weillustrateeachdimension
with examples that show how these variations influence query construction and interpretation.
Path length
To vary structural complexity, queries included 1, 2, or 3 hops, that is, traversal paths of increasing length, where each
hop corresponds to following a relationship between connected nodes in the graph.
Example:
A 1-hop query might retrieve actors and the movies they acted in:
MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title
A 2-hop query could trace from actors to critics via the movie the actor worked in and the critic favored:
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p.name, c.name
Clause composition
Tovarylogicalandsyntacticcomplexity,weuseddifferent Cypherconstructs: abaseline MATCH;anOPTIONAL MATCH
that retrieves additional data when present; two consecutive MATCHclauses connected by WITHto pass intermediate
results; a WHEREfilter; and a CASEexpression that provides if–else logic.
Example use of CASE:
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Movie)
RETURN p, m,
CASE WHEN m.release_year < 2010 THEN 'old'ELSE 'new'END as movie_age
Perturbations
To test whether LLMs would recognize perturbations, we deliberately (i) reversed relationship direction, (ii) introduced
implausible relationship labels such as EATS (e.g. (a:Actor)-[EATS]->(m:Movie) ), (iii) replaced sensible node
7

A preprint - February 6, 2026
labels with implausible ones such as Food (e.g. (a:Actor)-[ACTED_IN]->(m:Food) ), (iv) omitted the node type
while giving the variable a misleading name, and (v) introduced clause-specific faults: contradictory WHEREconstraints
(e.g.afilmreleasedbothbeforeandafter2020),illogical WHEREconstraintssuchasareleaseyearof-1,orill-formed
value tests such as movie.title > 2020 .
Example:
Implausible relationship: The relationship EATSis implausible in this context, testing whether the model recognizes
semantic inconsistency.
MATCH (a:Actor)-[:EATS]->(m:Movie)
RETURN a.name, m.title
These variations allow us to assess not just surface-level understanding, but whether LLMs can handle structural,
semantic, and logical correctness in Cypherqueries.
3.2 LLMs
The LLMs tested were selected to be a good representation of the most commonly used models at the time of the
experiment that were also easily accessible via LangChain . This includes two GPT variants, a Claude model, and two
DeepSeek models. The 70b DeepSeek model was run locally, all other models were accessed via official APIs.
Table 1: LLMs used in explanation trials
llm version
deepseek-r1:70b 0c1615a8ca32
o1-preview-2024-09-12 2024-09-12
o3-mini-2025-01-31 2025-01-31
claude-3-7-sonnet-20250219 20250219
deepseek-reasoner-api date: 05.03.2025
3.3 Experimental Design
WeprovidedtheLLMwitheachqueryfromthebenchmarkandaninstructionprompt(subsectionA.1)toproducea
step-by-step explanation and give a brief summary of the query’s overall purpose. The LLM was also asked to flag
errors, with flipped relationships and ill-formed node or relationship names being specifically mentioned in the prompt.
Outputswerethenmanuallyevaluatedbythefirstauthor(L.P.)forexplanationcorrectnessofthesummary;whether
errors, when present, were explicitly identified and no errors were flagged if there were none; and whether the summary
omitted the year for queries that specified one. The following example illustrates how the step-by-step summary and the
one-sentence-summary were checked for completeness.
Example Query:
MATCH (p:Person name: "Alice")-[:ACTED_IN]->(m:Movie) RETURN p, m
Summary:
1. The query finds a node labeled Person with the property name set to "Alice".
2. It then follows outgoing ACTED_IN relationships from this Person node to find
connected nodes labeled Movie.
3. Finally, it returns both the Person and the associated Movie nodes.
Overall, this query retrieves the person "Alice" and the movies in which she has acted.
Everything in the query, including the relationship direction and names, is appropriate
and makes sense.
3.4 Statistical Analysis
Wecomputedthemeanaccuraciesandassociated95-percentWilsonconfidenceintervalsoftheone-sentencesummaries,
the detection of injected perturbations, and the avoidance of false positives. For one-sentence summaries, a prediction
wascountedascorrectwhenthesummaryprovidedanaccurateandcompletedescriptionofthequery. Forproblem
8

A preprint - February 6, 2026
detection, asuccesswasrecordedwhenthemodelexplicitlyidentifiedtheperturbationinthequeriesthatcontainedone.
For false positives, a success required that the model did not flag any error in queries without perturbations.
We compared the performances for each of these outcomes of each possible pair of LLMs, out of the five LLMs
considered,implementingtheMcNemar’stestforpairedbinaryoutcomesin Python. Foreachpair,weranthetestona
2x2 contingency table of query-level correctness (correct vs incorrect) and based the test exclusively on the discordant
pairs,thatis,queriesansweredcorrectlybyonemodelandincorrectlybytheother. Theexacttwo-sidedp-valuewas
obtained from a binomial test on the number of discordant pairs, under the null hypothesis that both models have equal
probability of being correct for any query (p = 0.5). To account for multiple pairwise comparisons among models, the
p-values were adjusted using the Holm step-down procedure to control the family-wise error rate at a nominal level
of 0.05. In addition, we recorded the direction of the difference for each pair of models by comparing the counts of
querieswhereonlytherowmodelwascorrectvs. onlythecolumnmodelwascorrect,sothatadjustedp-valuescould
be interpreted together with the sign of the performance difference.
Toexaminehowtheaccuracyofonesentencesummariesvariedacrossmultipledimensions(asdescribedinsection3.1,
section 3.1, section 3.1), we used the Rpackage spaMM[17] to fit a series of Generalized Linear Mixed-effects Models
(GLMMs) with a binomial link. We generated two alternative response variables representing accuracy: the first
considered a strict correctness criterion, and the second considered a relaxed criterion that did not treat a missing year
intheone-sentencesummaryasanerrorifthatyearappearedinthestep-by-stepexplanation. Inbothspecifications,
the model included LLM, hop count, clause type, and perturbation type as fixed-effect predictors, along with a
random intercept for the query. For each GLMM, we conducted Tukey-adjusted post-hoc pairwise comparisons for all
multi-levelpredictorsusingthefunction glht()fromthepackage multcomp [10]. Tovisualizetheresultingeffects,we
computedpartialdependenceestimatesandcorrespondingconfidenceintervalsusingthefunction pdep_effects(...,
intervals = "fixefVar") andplottedthosetogetherwithcompactletterdisplaysindicatinggroupsnotsignificantly
different from each others.
3.5 Results
3.5.1 Accuracy of one-sentence summaries
3.5.1.1 EffectoftheLMM TheaccuracyoftheLLMsoncorrectandcompleteone-sentenceexplanationsummaries
(Table 2) exceeded 70% for o1-preview, deepseek-reasoner-api, and o3-mini, with o1-preview performing the best.
Deepseek-r1:70breached66%,andclaude3.7sonnetwasthelowestat52%. Triviallywrongstatements(e.g.,“Xis
misspelled”followedbyasuggestiontochangethewordtooneofidenticalspelling)andstylisticcomplaintsaboutnode
andrelationshipnameswereignoredinthescoringbecausetheyareunlikelytomisleadreaders. AHolm-corrected
pairwise McNemar test (Table 3) showed that deepseek-reasoner-api, o1-preview, and o3-mini performed significantly
betterthanclaude3.7sonnetonone-sentencesummaries. ThisresultwasconfirmedbyaGLMMfitcontrollingfor
other sources of variation (Fig: 3a & Fig: 3b). The pattern held irrespective of whether or not year omissions were
counted as errors, but deepseek-reasoner-api significantly outperformed deepseek-r1:70b only in the latter case. A
detailed breakdown of disagreement counts is provided in Table 14.
3.5.1.2 Effectofqueryfeatures Ourquerysetvariedalongthreeaxes: clausetype,perturbation,andhopcount,
eachpotentiallyaffectingexplanationperformance(Fig:4). Forclausetype((Fig:4a)), CASEstatementsstandoutas
problematic; however, CASEqueries uniformly include a year and some LLMs struggled to include missing years in
theone-sentencesummaries,whichlikelyconfoundsthisresult. TheGLMMresultsshowthatclausetypenolonger
significantly influenced the accuracy of one-sentence summary once year mistakes were overlooked (Fig: 3h).
With respect to perturbations, only contradictory WHEREvaluesandflip relationship induce complete failures in
someLLMs(Fig:4b),resultinginasignificantlyworseaccuracyofone-sentencesummarywheneitherofthesetwo
perturbations are present, compared to no perturbation (Fig: 3e & Fig: 3f). More specifically, Claude failed on both for
allinstances,ando1-previewonlyfailedonall contradictory WHEREvalues. Notethatthelattercomprisesonlythree
queries because it applies solely to WHEREclauses. For other types of perturbation, no cross-model pattern emerged
consistently.
Regardinghop count, 1-hopqueries wereprocessed moreaccurately byLLMs than2- or3-hop queries(Fig: 4c). The
GLMMsshowsthat,asforclausetype,thehopcountnolongersignificantlyinfluencedtheaccuracyofone-sentence
summaryonceyearmistakeswereoverlooked(Fig:3d). When OPTIONAL MATCH orWITHwereusedtointroducean
extra hop ("(+1)" category) to the base hop count, the accuracy surprisingly improved, although such queries were too
few to test the resulting differences using a GLMM.
9

A preprint - February 6, 2026
3.5.2 Ability to flag faults in perturbed queries
3.5.2.1 EffectoftheLLM Amongperturbedqueries,o1-preview,claude3.7sonnet,anddeepseek-reasoner-api
exceeded 80%, o3-mini scored 77%, and deepseek-r1:70b scored 68% when flagging them as faulty (Table 2). A
Holm-corrected pairwise McNemar test (Table 4) showed that deepseek-reasoner-api and o1-preview performed
significantly better than deepseek-r1:70b. A detailed breakdown of disagreement counts is provided in Table 15.
3.5.2.2 Sensitivity to perturbation type in perturbation flagging Flipped relationships, nonsense node labels and
nonsenserelationshiplabelswerespecificallymentionedintheprompt. Allmodelsflaggedcontradictory WHEREclauses,
nonsense node- and relationship names, and type mismatches (Fig: 5). Many struggled with the combination of missing
nodetypeandmisleadingnodenames,whereonlyclaudeexceeded50%at79%. Forflippedrelationshipdirections,
o1-preview and deepseek-reasoner-api were flawless, whereas the others made occasional to frequent mistakes.
3.5.3 Avoidance of false positives
3.5.3.1 Effect of LLM Deepseek-r1:70b flagged a correct query as defective in 47% of cases, claude 3.7 sonnet in
33%,o1-previewin7%,ando3-minianddeepseek-reasoner-apineverdidso(Table2). AHolm-correctedpairwise
McNemar test (Table 5) showed no statistically significant differences between the models. Given the small sample
(n=15) and limited power, we interpret differences in this task primarily based on descriptive statistics. A detailed
breakdown of disagreement counts is provided in Table 16.
Table 2: Accuracy with 95% Wilson CI for correct one-sentence summary generation, problem detection, and false
positives.
One-sentence summary Problem detection False positives
LLM 𝑛correct acc. 95% CI 𝑛correct acc. 95% CI 𝑛correct acc. 95% CI
claude 3.7 sonnet 90 47 0.522 [0.420, 0.622] 75 64 0.853 [0.756, 0.916] 15 10 0.666 [0.417, 0.848]
deepseek-reasoner-api 90 66 0.733 [0.634, 0.814] 75 67 0.893 [0.803, 0.945] 15 15 1.000 [0.796, 1.000]
deepseek-r1:70b 90 60 0.666 [0.564, 0.755] 75 51 0.680 [0.568, 0.775] 15 8 0.533 [0.301, 0.752]
o1-preview 90 69 0.766 [0.669, 0.842] 75 66 0.880 [0.787, 0.936] 15 14 0.933 [0.702, 0.988]
o3-mini 90 64 0.711 [0.610, 0.795] 75 58 0.773 [0.667, 0.853] 15 15 1.000 [0.796, 1.000]
case
optional matchsimple
where clausewith clause
Clause Type0.00.20.40.60.81.0Fraction correctly explaineddeepseek-r1:70b
o1-preview
o3-miniclaude 3.7 sonnet
deepseek-reasoner-api
(a) By clause
contradictory where valuesflip relationship
illogical where value
no node label, misleading namenonsense node name
nonsense relation namewrong where type
Perturbation0.00.20.40.60.81.0Fraction correctly explaineddeepseek-r1:70b
o1-preview
o3-miniclaude 3.7 sonnet
deepseek-reasoner-api (b) By perturbation
1
1 (+1)2
2 (+1)3
3 (+1)
Number of Hops0.00.20.40.60.81.0Fraction correctly explaineddeepseek-r1:70b
o1-preview
o3-miniclaude 3.7 sonnet
deepseek-reasoner-api (c) By number of hops
Figure 4: How correctness of one-sentence summaries is influenced by query features.
10

A preprint - February 6, 2026
(a) LLM
ab
abb
b
507090
claude 3.7 sonnet deepseek−r1:70b
deepseek−reasoner−apio1
o3−mini
LLMAccuracy of one sentence summary (%) (b) LLM (year not error)
ab
acbcbc
406080100
claude 3.7 sonnet deepseek−r1:70b
deepseek−reasoner−apio1
o3−mini
LLMAccuracy of one sentence summary (%)
(c) Hop count
a
ab b
0255075100
1 2 3
n_hopsAccuracy of one sentence summary (%) (d) Hop count (year not error)
a aa
0255075100
1 2 3
n_hopsAccuracy of one sentence summary (%)
(e) Perturbation
a
bababbcaac
b
0255075100
None
flip relationship
nonsense relation namenonsense node name
no node label, misleading nameillogical where valuewrong where type
contradictory where values
perturbationAccuracy of one sentence summary (%) (f) Perturbation (year not error)
a
bababab
bab
ab
0255075100
None
flip relationship
nonsense relation namenonsense node name
no node label, misleading nameillogical where valuewrong where type
contradictory where values
perturbationAccuracy of one sentence summary (%)
(g) Clause Type
a
aaa
b
0255075100
simple
optional matchwith clausewhere clausecase
clause typeAccuracy of one sentence summary (%) (h) Clause Type (year not error)
a
aa a
a
0255075100
simple
optional matchwith clausewhere clausecase
clause typeAccuracy of one sentence summary (%)
Figure 3: Effects on explanation accuracy. Left column: year mismatches counted as errors (strict criterion). Right
column: yearmismatchesnotcountedaserrors(relaxedcriterion). Thesefigureswereconstructedusinganadditive
GLM. If items share a letter, they are not significantly different from each other.
11

A preprint - February 6, 2026
Table3: PairwiseMcNemartestscomparingLLMaccuracyofone-sentencesummarygenerated. P-valuesareHolm
corrected. Positive values indicate better performance of the model in the row over the model in the column.
claude 3.7 sonnet deepseek-reasoner-api deepseek-r1:70b o1-preview o3-mini
claude 3.7 sonnet - -0.0119 -0.2061 -0.0000 -0.0121
deepseek-reasoner-api 0.0119 - 1.0000 -1.0000 1.0000
deepseek-r1:70b 0.2061 -1.0000 - -0.5588 -1.0000
o1-preview 0.0000 1.0000 0.5588 - 1.0000
o3-mini 0.0121 -1.0000 1.0000 -1.0000 -
Table 4: Pairwise McNemar tests comparing LLMs on error detection capabilities. P-values are Holm corrected.
Positive values indicate better performance of the model in the row over the model in the column.
claude 3.7 sonnet deepseek-reasoner-api deepseek-r1:70b o1-preview o3-mini
claude 3.7 sonnet - -1.0000 0.0504 -1.0000 1.0000
deepseek-reasoner-api 1.0000 - 0.0003 1.0000 0.0312
deepseek-r1:70b -0.0504 -0.0003 --0.0005 -0.1953
o1-preview 1.0000 -1.0000 0.0005 -0.0504
o3-mini -1.0000 -0.0312 0.1953 -0.0504 -
contradictory where valuesflip relationship
illogical where value
no node label, misleading namenonsense node name
nonsense relation namewrong where type
Perturbation0.00.20.40.60.81.0Fraction correctly flaggeddeepseek-r1:70b
o1-preview
o3-miniclaude 3.7 sonnet
deepseek-reasoner-api
Figure 5: For each perturbation category (definitions in subsection 3.1), the bars show the fraction of queries in which
each evaluated LLM correctly signalled that something was wrong.
3.6 Discussion
Acrossthe90 Cypherqueriesinourbenchmark,allLLMsproducedcorrectandcompleteone-sentencesummaries
for more than half of the cases. Only three models, o1-preview, deepseek-reasoner-api, and o3-mini, exceeded the
70% accuracy mark, which is encouraging but still insufficient for robust real-world use. The gap between the best
performer (o1-preview) and the worst (claude) was nearly 20%, highlighting substantial variation in explanation quality
across models. These differences were confirmed by the generalized linear mixed model analysis, which showed
significant overall accuracy differencesacross LLMs, withclaude 3.7sonnet performing reliablyworse thandeepseek-
reasoner-api, o1-preview, and o3-mini under both strict and relaxed evaluation criteria, and deepseek-reasoner-api
further outperforming deepseek-r1:70b when year omissions are disregarded.
12

A preprint - February 6, 2026
Table5: PairwiseMcNemartestscomparingLLMsonfalsepositives. P-valuesareHolmcorrected. Positivevalues
indicate better performance (=less false positives) of the model in the row over the model in the column.
claude 3.7 sonnet deepseek-reasoner-api deepseek-r1:70b o1-preview o3-mini
claude 3.7 sonnet - -0.3750 1.0000 -0.5000 -0.3750
deepseek-reasoner-api 0.3750 - 0.1406 1.0000 –
deepseek-r1:70b -1.0000 -0.1406 - -0.2188 -0.1406
o1-preview 0.5000 -1.0000 0.2188 - -1.0000
o3-mini 0.3750 – 0.1406 1.0000 -
One weakness recurred throughout the evaluation: omission of years from the one-sentence summaries. For deepseek-
reasoner-api,morethanfouroutoffiveincorrectsummarieswerewrongforthisreason;foro1-previewando3-mini,
theomissionaccountedformorethanhalfoftheirerrors. Thissuggeststhatthesummary-generationobjectivemay
overweight brevity relative to faithfulness. Explicitly requiring the retention of temporal information during instruction
tuning could address this issue and improve performance. The mixed-effects analysis reinforces this interpretation:
one-hop queries only outperform three-hop queries when year omissions are counted as errors, indicating that the
apparent structural difficulty is largely driven by temporal information loss rather than query length alone.
Whendetectingflawedqueries,modelsvariedinhowtheybalancedsensitivityandspecificity. Claude,forexample,
exhibited high sensitivity (identified many injected defects) but low specificity (raised a relatively high number of false
alarms) suggesting a trade-off between the two. In contrast, o1-preview and deepseek-reasoner-api combined high
defect-detection rates (above 85%) with low false-positive rates (below 10%), showing fewer signs of this trade-off.
o3-miniachieveda slightlylowerdetectionrate(around77%)butdidnotproduceanyfalsepositives. Thesepatterns
provide a useful basis for choosing models according to the preferred balance between the two metrics.
Acrossmodels,thereisnosimpleone-to-onelinkbetweenstrongperturbationdetectionandlowfalsepositiverates.
Most systems that rank highly in detection accuracy also avoid many false positives, but claude 3.7 sonnet breaks
this pattern: despite being among the strongest at identifying actual perturbations, its false positive rate was only
midrange. Deepseek-reasoner-apiando1-previewillustratethecleanerrelationship,pairinghighdetectionaccuracy
with near-perfect avoidance of false positives, with o3-mini not far behind. In contrast, deepseek-r1:70b performed
worseonbothfronts. Overall,whilegooddetectionoftenalignedwithbetterfalsepositiveavoidance,thedifferences
among models - especially claude 3.7 sonnet’s mix of strong detection and moderate false positive control - show
thattheseabilitiesdonotalwaysimprovetogether. Consistentwiththeseobservations,theGLMMshowsthatunder
strict evaluation, flipped and contradictory relationships significantly reduce accuracy compared to no perturbation
or incorrect WHERE clauses, whereas under the relaxed criterion these differences largely disappear except for the
no-perturbation baseline.
Analysing how query features shape performance reveals a small set of consistent patterns. The apparent weakness
onCASEclauses is largely a temporality confound: all CASEqueries contained years, and missing-year omissions
account for a substantial share of errors. The mixed-effects results align with this explanation: CASE queries perform
significantly worse than all other clause complexities only when year omissions are treated as errors, further confirming
that temporality rather than clause structure drives this effect. The two perturbations that occasionally produced
complete failures, contradictory WHEREvalues and flipped relationship directions, point to specific blind spots (logical
consistency checks and small changes with large impact), although flipped relationship directions were specifically
mentionedinthepromptsubsectionA.1. Giventhesmallsizeofthecontradictory- WHEREset(n=3),wetreatclaimshere
asprovisional,especiallygiventhatotherquerieswithsimilarlogicalconsistencychecks,suchasillogical WHEREvalues,
did not show this pattern. Structurally, base 1-hop queries were easier than 2–3 hops, whereas “(+1)” variants created
byOPTIONAL MATCH orWITHwereunexpectedlyeasierthantheirbases,withthecausebeinguncertainandmeriting
follow-up investigation. These patterns were consistent with the effects estimated in our generalized linear mixed
models, which showed significant differences across LLMs, hop counts, perturbation types, and clause complexities.
Theseobservationsneedtobeinterpretedinlightofcertainlimitations. Thescopeofthisexperimentisintentionally
narrow, focusing on 90 handcrafted queries in a movie-style Knowledge Graph. While this controlled setting is
well-suited for isolating specific behaviours and testing fundamental capabilities, real-world graph queries, especially in
domains such as mathematics or biology, can be more complex and embedded in richer schema contexts. Furthermore,
the scoring process relied on human judgement, introducing some subjectivity in determining what constitutes a
"complete"andcorrectsummary,whichisunavoidablegiventhetool’shuman-in-the-loopdesign. Finally,because
LLMsevolverapidly,thesnapshotpresentedheredoesnotofferanyguaranteesaboutfuturecapabilities,andsubsequent
modelsmaysurpasstheperformanceobservedinthisstudy. Despitetheseconstraints,theevaluationhighlightsconcrete
13

A preprint - February 6, 2026
strengths and weaknesses that can directly inform the explanation framework’s future design goals. The following two
experiments examine different aspects of the framework, using examples from mathematical (MaRDI) and biological (a
populationofspottedhyenas)domainstoillustrateitscapabilitiesinhandlingquestionansweringondiverseKnowledge
Graphs.
4 Real-Life Examples: Question Answering on Biological and Mathematical Knowledge
Graphs
To determine how well Cypherquery generation from natural-language questions, explained in section 2, performs on
real-worldKnowledgeGraphs,weconductedtwoexperimentsindistinctdomains. Thefirstusesasoftware-centricslice
oftheMaRDIKnowledgeGraph(includingCRANpackages,Zenodosoftware,citedpapers,andauthorswithzbMATH
IDs). The second uses a newly constructed Knowledge Graph derived from long-term field data on a free-ranging
populationofspottedhyenasintheNgorongoroCrater,Tanzania(see https://hyena-project.com ),withquestions
formulatedbyprojectexperts. Acrossbothexperiments,wetested14 LangChain -compatibleLLMsunderidentical
constraintsandkeptthenumberofquestionsdeliberatelysmalltoenablesystematiccomparisonacrossmanymodels
ratherthanexhaustivecoverageofeachgraph. Accordingly,theresultsarenotintendedasafullbenchmarkranking;
instead, they should be interpreted as comparative performance under controlled conditions, highlighting failure modes
andmodel-specificdifficulties. OurcentralresearchquestioninbothcasesiswhetheranLLMcanproduceacorrect
query after an initial Cyphergeneration step and at most two user-submitted amendments, and whether particular
questions pose more difficulties than others.
4.1 LLMs
We aimed to include a broad selection of LangChain -compatible LLMs, including several open source variants, some
GPT models and a Claude model. This experiment evaluates different model versions than the explanation experiment
discussed earlier. Claude 3.7 sonnet, deepseek-reasoner-api, o1 and o3-mini were accessed via official APIs, the
remaining models were run locally.
Table 6: LLMs used in MaRDI experiment
LLM number of parameters version/hash
claude-3-7-sonnet-20250219 ? 20250219
deepseek-r1 70b 0c1615a8ca32
deepseek-reasoner-api ? date: 21.03.2025
exaone-deep 32b a9422f9a5071
gemma3 27b a418f5838eaf
gpt-5.2-2025-12-11 ? 2025-12-11
llama3.3 70b a6eb4748fd29
nemotron 70b 2262f047a28a
o1-2024-12-17 ? 2024-12-17
o3-2025-04-16 ? 2025-04-16
o3-mini-2025-01-31 ? 2025-01-31
o4-mini-2025-04-16 ? 2025-04-16
phi4 14b ac896e5b8b34
qwq 32b 009cb3f08d74
4.2 Statistical Analysis
For each LLM, we computed the mean accuracies and associated 95-percent Wilson confidence intervals of the
productionof correct Cypherquerieson thefirst attemptandwithin theallowed sequenceofone initialtryplusupto
two amendments. We then calculated the same accuracy measures on a per-question basis. This analysis was performed
across nine (for MaRDI) or five (Hyena KG) natural-language questions and fourteen LLMs.
14

A preprint - February 6, 2026
4.3 Experimental Design
Foreachquestioninthebenchmarkset,weprovidedthequestionansweringpipelinedescribedinsection2withthe
question,thegraphschemaandaninstructionprompt(subsectionA.2orsubsectionA.4). Ifthegeneratedquerywas
not judged to be correct by the researcher, up to two natural language amendments were supplied for the framework to
improve thequery, along withan amendmentprompt (subsection A.3). If thequery wasstill not deemedto be correct
after this, it was counted as a failure. This workflow was executed for each LLM mentioned in subsection 4.1.
4.4 Real-Life Example 1: Question Answering on the MaRDI Knowledge Graph
To assess real-life performance on a software-oriented Knowledge Graph, we tested 14 LLMs on nine natural-language
questions over a software-centric slice of the MaRDI Knowledge Graph (CRAN packages, Zenodo software, cited
papers, and authors with zbMATH IDs). The underlying subgraph connects software packages, publications, and
authors through citation and authorship relationships, requiring the models to reason over multi-step links between
research outputs and contributors. The question set is intentionally small to support consistent cross-model comparison
and to highlight typical error patterns rather than to provide exhaustive coverage. We evaluate whether models can
arrive at a correct Cypherquery after the initial generation and up to two user-submitted amendments, and we analyze
which question types lead to systematic difficulties.
For instance, a sample interaction might be:
Natural Language Question:
Which are the ten authors that created the most software packages?
Initially Generated Query:
MATCH (a:Author)-[:HAS_AUTHOR]->(s:SoftwarePackage)
RETURN a.name, COUNT(s) ASpackageCount ORDER BY packageCount DESC LIMIT 10
User Amendment Request:
The has_author relationship is the wrong way around.
Amended Query:
MATCH (s:SoftwarePackage)-[:HAS_AUTHOR]->(a:Author)
RETURN a.name, COUNT(s) ASpackageCount ORDER BY packageCount DESC LIMIT 10
4.4.1 Data
ThedataforthisanalysisconsistedofasubsetoftheMaRDIKnowledgeGraphandacustomsetofnatural-language
questions based on the graph.
4.4.1.1 Knowledge Graph
We created a subgraph of the MaRDI Knowledge Graph [18], restricted to software packages, authors, and publications
(Fig:6). WeextractedeveryCRANpackagethatcitesapaperwithazbMATHde-number[9,15],aswellastheauthors
ofthosepapers. WethenaddedalldeveloperswhobothmaintainCRANpackagesandholdazbMATHauthoridentifier.
Finally, the subgraph was supplemented with every Zenodo software whose authors likewise possess a zbMATH author
id [14]. The SPARQLqueries used for these selections can be found in Appendix B.
15

A preprint - February 6, 2026
Figure 6: Data Model for the MaRDI KG subgraph
4.4.1.2 Questions
Wedesignedninebenchmarkquestions(seeAppendixD)basedonthegraphschemaofourMaRDIsubgraph. Each
canbetranslatedinto Cypherusingvariousclausessuchas DISTINCT ,LIMIT,orWHERE.Theseclausesservedonly
as guidelines: any query that returned the expected result was considered correct, regardless of the constructs used.
Although this could in principle allow an incorrect query to match the expected output by coincidence, this is unlikely
becausetheexpectedresultsaretypicallyhighlyspecific. Anexamplequestion,whichcanbeansweredusingthe LIMIT
clause, is "Which are the ten authors that created the most software packages?" .
4.4.2 Results
We summarized the number of correct answers per model and tracked how many attempts the models needed for their
correct answers in Fig: 7. The LLMs o3-mini and GPT 5.2 were the only ones that solved all nine tasks; qwq, gemma3,
o4-mini, o1 and o3 followed with eight out of nine successfully solved tasks. Every other model reached 7/9 except
deepseek-r1:70b,whichmanagedonly5/9. Themajoritysucceededonthefirsttryforthemajorityoftheirquestions;
onlygemma3moreoftenrequiredtwoattemptsthanoneattempt,andathirdattemptwasseenonlyforGPT5.2,o1,
phi4:14bandnemotron:70b. Ifdeepseek-reasoner-api,o4-miniando3weresuccessful,itwasalwaysonthefirsttry.
ThenumberofcorrectanswersonthefirsttryandwithinfirsttriesaswellastheaccuracyandWilson95%CIarelisted
in Table 7.
Fig: 8 and Table 8 show the per-question accuracy (14 LLMs = maximum score 14), with the latter also containing total
counts and CIs. Four tasks were successfully solved by every LLM (the count-based query, the “simple” query, the
intersectionquery,andthe NOTquery),whilethe“sharedauthorsbetweenpackages”taskprovedhardest,withjustthree
correct solutions, and none of them on the first try.
Table 7: First-attempt and within-three-attempts accuracy with 95% Wilson CI on nine natural language questions.
LLM correct on first try accuracy first try 95% CI first try correct within 3 accuracy within 3 95% CI within 3
claude 3.7 sonnet 6 0.6667 [0.354, 0.879] 7 0.7778 [0.453, 0.937]
deepseek-reasoner-api 7 0.7778 [0.453, 0.937] 7 0.7778 [0.453, 0.937]
deepseek-r1:70b 4 0.4444 [0.189, 0.733] 5 0.5556 [0.267, 0.811]
exaone_deep:32b 6 0.6667 [0.354, 0.879] 7 0.7778 [0.453, 0.937]
gemma3:27b 3 0.3333 [0.121, 0.646] 8 0.8889 [0.565, 0.980]
gpt-5.2 8 0.8889 [0.565, 0.980] 9 1.0000 [0.701, 1.000]
llama3.3:70b 4 0.4444 [0.189, 0.733] 7 0.7778 [0.453, 0.937]
nemotron:70b 4 0.4444 [0.189, 0.733] 7 0.7778 [0.453, 0.937]
o1 7 0.7778 [0.453, 0.937] 8 0.8889 [0.565, 0.980]
o3-mini 8 0.8889 [0.565, 0.980] 9 1.0000 [0.701, 1.000]
o3 8 0.8889 [0.565, 0.980] 8 0.8889 [0.565, 0.980]
o4-mini 8 0.8889 [0.565, 0.980] 8 0.8889 [0.565, 0.980]
phi4:14b 6 0.6667 [0.354, 0.879] 7 0.7778 [0.453, 0.937]
qwq:32b 7 0.7778 [0.453, 0.937] 8 0.8889 [0.565, 0.980]
16

A preprint - February 6, 2026
Table8: First-attemptandwithin-three-attemptsaccuracyacross14modelsforeachnaturallanguagequestionwith
95% Wilson CI.
question first try correct first try accuracy 95% CI first try within 3 correct within 3 accuracy 95% CI within 3
Get me all authors and, if present, any datasets they are tied
to.9 0.642857 [0.388, 0.837] 12 0.857143 [0.601, 0.960]
Give me all people that created at least 5 datasets. 9 0.642857 [0.388, 0.837] 12 0.857143 [0.601, 0.960]
How many datasets did Rob Hyndman create? 13 0.928571 [0.685, 0.987] 14 1.000000 [0.785, 1.000]
Which are the ten authors that created the most software
packages?8 0.571429 [0.326, 0.786] 9 0.642857 [0.388, 0.837]
Which authors authored publications that contain the sub-
string “Pareto”?7 0.500000 [0.268, 0.732] 13 0.928571 [0.685, 0.987]
Which authors does the software package graphclust have? 14 1.000000 [0.785, 1.000] 14 1.000000 [0.785, 1.000]
Which authors worked on both the dataset “Bitcoin Dataset
with Missing Values” and “Rideshare Dataset without Miss-
ing Values”?14 1.000000 [0.785, 1.000] 14 1.000000 [0.785, 1.000]
Which software packages have no authors? 12 0.857143 [0.601, 0.960] 14 1.000000 [0.785, 1.000]
Which software packages share the same authors? 0 0.000000 [0.000, 0.215] 3 0.214286 [0.076, 0.476]
0 2 4 6 8
Count of Correct Answersphi4:14b
gemma3:27b
qwq:32b
exaone-deep:32b
llama3.3:70b
nemotron:70b
deepseek-r1:70b
deepseek-reasoner-api
claude 3.7 sonnet
o3-mini
o4-mini
o1
o3
gpt 5.2LLM6 1
3 5
7 1
6 1
4 3
4 1 2
4 1
7
6 1
8 1
8
7 1
8
8 11 Try
2 Tries
3 Tries
Figure7: ForeachLLM,thebarlengthindicatestheshareoftheninequestionseventuallyansweredcorrectly. The
colors within each bar partition that success rate by the number of attempts (first, second, or third) needed to reach the
correct answer (a maximum of three tries was allowed).
17

A preprint - February 6, 2026
1 Try 2 Tries 3 Tries
0 2 4 6 8 10 12 14
Count of Correct AnswersQ1
Q2
Q3
Q4
Q5
Q6
Q7
Q8
Q9Question Code14
7 6
8 1
13 1
9 1 2
9 2 1
1 2
14
12 2Which authors does the software package graphclust
have?
Which authors authored publications that contain the
substring "Pareto"?
Which are the ten authors that created the most
software packages?
How many datasets did Rob Hyndman create?
Give me all people that created at least 5 datasets.
Get me all authors and, if present, any datasets they
are tied to.
Which software packages share the same authors?
Which authors worked on both the dataset "Bitcoin
Dataset with Missing Values" and "Rideshare Dataset
without Missing Values"?
Which software packages have no authors?
Figure 8: For each question, the bar height denotes the number of the 14 evaluated LLMs that returned the correct
answer, illustrating the relative difficulty of the questions. Additionally, the bars are colored according to the number of
amendment tries to solve them.
4.5 Real-Life Example 2: Question Answering on a Hyena Knowledge Graph
To determine whether the performance obtained for the KG used in the previous section remains similar for a real-world
Knowledge Graph from a substantially different domain, we created a Knowledge Graph based on data from a
long-term study of a free-ranging population of spotted hyenas inhabiting the Ngorongoro Crater in Tanzania (see
https://hyena-project.com ). Weexaminedtheperformanceofourapproachforfivereal-worldquestionsabout
kinship, which were directly formulated by experts involved in the Ngorongoro Hyena Project. We executed the query
generation framework with 14 LangChain -compatible LLMs (see Table 6). As in subsection 4.4, we kept the number
ofquestionslowtoallowsystematiccomparisonacrossmanymodelsratherthanexhaustivecoverageofthegraph. This
setup does not identify the best-performing LLMs but highlights which ones fail or struggle under consistent conditions.
The main research question we focus on is, again, whether the LLMs are able to produce a correct query after an initial
Cypherquerygenerationstepandamaximumoftwouser-submittedamendmentstothequery. Wealsoanalyzethe
extent to which LLMs have more difficulties with certain questions.
A sample interaction might be:
Natural Language Question:
What is the proportion of cubs that were sired by philopatric males?
Initially Generated Query:
MATCH (cub:Hyena)-[:HAS_FATHER]->(dad:Hyena)
OPTIONAL MATCH (dad)-[:BIRTH_CLAN]->(bc:Clan)
OPTIONAL MATCH (dad)-[:CURRENT_CLAN]->(cc:Clan)
WITH COUNT(cub) AStotalCubs,
SUM( CASE WHEN dad.sex = 'male 'AND bc = cc THEN 1ELSE 0END)ASphilopatCubs
RETURN philopatCubs * 1.0 / totalCubs ASproportion
User Amendment Request:
With how this query is made, it is possible that the dad left, had the cub and came back.
Please check instead that he never changed his clan.
18

A preprint - February 6, 2026
Amended Query:
MATCH (cub:Hyena)-[:HAS_FATHER]->(dad:Hyena)
OPTIONAL MATCH (dad)-[:BIRTH_CLAN]->(clan:Clan)<-[:CURRENT_CLAN]-(dad)
WITH COUNT(cub) AStotalCubs,
COUNT
(CASE WHEN dad.sex = 'male 'AND clan IS NOT NULL
AND NOT (dad)-[:CHANGED_CLAN]->() THEN 1END)
ASphilopatricCubs
RETURN toFloat(philopatricCubs) / totalCubs ASproportion
4.5.1 Data
This experiment is based on a knowledge graph constructed from long-term ecological observation data and a
corresponding set of domain-expert natural-language questions.
4.5.1.1 Knowledge Graph
We created a KG based on real observational data collected and curated by members of the Ngorongoro Hyena Project.
Toobtainastructurallymeaningfulyetmanageablesubset(seeFig:9),wefocusedondifferentkinshiprelationships
betweenhyenas,suchasparents,socialmothers(sincefemalespottedhyenassometimesadoptacubthatisnottheir
genetic offspring [4]) and litter mates (mothers produce between one and three cubs per delivery [6]). Additionally, we
integrated information about the birth clan and clans the hyenas belonged to at different points in time (most males
change social group upon reaching sexual maturity [8]), as well as sightings (where and when a given hyena was
observed in the field).
Figure 9: Data Model for the Hyena KG
4.5.1.2 Questions
Wereceivedrealisticquestions(AppendixE)fromexpertstobeansweredonthehyenagraph. Thesefivequestions
focus on the reproductive behaviour of males. The first two questions focus on paternities involving males that did not
originatefromthecorepopulation—akeybiologicalvariableinfluencinggeneflowovertimeandspace,associated
with potential for local adaptation and risks of inbreeding depression. Those outsiders originate from clans outside the
NgorongoroCraterwhicharenotdirectlymonitoredbythebiologists,andarereferredtoasoriginatingfromclanX.
Thenexttwoquestionsfocusonhowthedispersalstatusofmalesconnectstotheirreproductivesuccess. Inspotted
hyenas, male dispersal is thought to be a consequence of their reproductive access to females, with males remaining or
leaving clans depending on mating opportunities [8, 2]. The last question focuses on the extent to which males can
manage to sire offspring in clans where they do not belong. Those events are thought to be rare since (i) intrusion into
otherclanscanbemetwithresistance,(ii)spottedhyenamalescanonlymatewithfemalesthatacceptthem,and(iii)
femalesarethoughttoprefermalestheyknowwell. Yet,itremainspossible,especiallyifmalesvisitaclanthatthey
were previously a member of.
The hyena experts wrote the questions before the graph existed and thus were not familiar with our LLM/KG approach,
thereby avoiding bias introduced by knowledge of the graph schema and mimicking a real-world trial of our framework.
They selected them because they are representative of the kind of questions they routinely have to answer as part of
largeranalyses. Theamendmentstothe CypherqueriesgeneratedbytheLLMs,ifnecessary,wereperformedbythe
first author (L.P.) in accordance with detailed explanations of how the questions were understood by the biologists.
19

A preprint - February 6, 2026
4.5.2 Results
Performancevariedwidely. Onlyo3anddeepseek-reasoner-apiansweredeveryquestioncorrectly(5/5). O4-mini,o1
and GPT 5.2 followed with four correct answers, and Claude and o3-mini solved three. Among the models run via
Ollama,onlyqwqproducedanycorrectresponses(2/5);theremainingOllamamodelsfailedtoanswerasinglequestion.
Even though the small sample introduces uncertainty, a 0/5 outcome still implies a low upper confidence limit of about
40 percent, so the poor performance of these models is unlikely to be an artifact of sample size. Efficiency, measured as
thenumberofattemptsrequiredpercorrectanswer,wasmixed. o4-miniando3-miniprovidedmostoftheircorrect
answers on the first try, whereas the other successful models needed at least as many second attempts as first ones, and
more than half of them required three attempts for some questions (see Fig: 10). The number of correct answers on the
first try and within first tries as well as the mean accuracy and Wilson 95% CI are listed in Table 9. No single question
was solved by every model, which is not surprising given that several models had zero correct answers overall (see
Fig: 11 and Table 10).
Table 9: First-attempt and within-three-attempts accuracy with 95% Wilson CI on five natural language questions.
LLM correct on first try accuracy first try 95% CI first try correct within 3 accuracy within 3 95% CI within 3
claude 3.7 sonnet 1 0.2 [0.036, 0.624] 3 0.6 [0.231, 0.882]
deepseek-reasoner-api 2 0.4 [0.118, 0.769] 5 1.0 [0.566, 1.000]
deepseek-r1:70b 0 0.0 [0.000, 0.434] 0 0.0 [0.000, 0.434]
exaone-deep:32b 0 0.0 [0.000, 0.434] 0 0.0 [0.000, 0.434]
gemma3:27b 0 0.0 [0.000, 0.434] 0 0.0 [0.000, 0.434]
gpt-5.2 1 0.2 [0.036, 0.624] 4 0.8 [0.376, 0.964]
llama3.3:70b 0 0.0 [0.000, 0.434] 0 0.0 [0.000, 0.434]
nemotron:70b 0 0.0 [0.000, 0.434] 0 0.0 [0.000, 0.434]
o1 1 0.2 [0.036, 0.624] 4 0.8 [0.376, 0.964]
o3 2 0.4 [0.118, 0.769] 5 1.0 [0.566, 1.000]
o3-mini 2 0.4 [0.118, 0.769] 3 0.6 [0.231, 0.882]
o4-mini 3 0.6 [0.231, 0.882] 4 0.8 [0.376, 0.964]
phi4:14b 0 0.0 [0.000, 0.434] 0 0.0 [0.000, 0.434]
qwq:32b 0 0.0 [0.000, 0.434] 2 0.4 [0.118, 0.769]
Table 10: First-attempt and within-three-attempts accuracy across 14 models for the hyena questions.
question first try correct first try accuracy 95% CI first try within 3 correct within 3 accuracy 95% CI within 3
Whatistheproportionofcubsthatweresiredbymalesborn
in clan X?7 0.500000 [0.268, 0.732] 8 0.571429 [0.326, 0.786]
Whatistheproportionofcubsthatweresiredbyphilopatric
males?1 0.071429 [0.013, 0.315] 7 0.500000 [0.268, 0.732]
What is the proportion of cubs that were sired by primary
dispersers?0 0.000000 [0.000, 0.215] 5 0.357143 [0.163, 0.612]
Whatistheproportionofcubswithknownfathersthatwere
sired by males that were members of clans other than the
clan of the female at conception?0 0.000000 [0.000, 0.215] 5 0.357143 [0.163, 0.612]
Whatistheproportionoflittersforwhichatleastonecub
has one father born in clan X?4 0.285714 [0.117, 0.546] 5 0.357143 [0.163, 0.612]
20

A preprint - February 6, 2026
0 1 2 3 4 5
Count of Correct Answersphi4:14b
gemma3:27b
qwq:32b
exaone-deep:32b
llama3.3:70b
nemotron:70b
deepseek-r1:70b
deepseek-reasoner-api
claude 3.7 sonnet
o3-mini
o4-mini
o1
o3
gpt 5.2LLM2
2 2 1
1 1 1
2 1
3 1
1 2 1
2 2 1
1 2 11 Try
2 Tries
3 Tries
Figure10: ForeachLLM,thebarheightindicatestheshareofthefivequestionseventuallyansweredcorrectly. The
colors within each bar partition that success rate by the number of attempts (first, second, or third) needed to reach the
correct answer (a maximum of three tries was allowed).
21

A preprint - February 6, 2026
0 1 2 3 4 5 6 7 8
Count of Correct AnswersQ1
Q2
Q3
Q4
Q5Question Code7 1
4 1
1 4 2
3 2
4 1What is the proportion of cubs that were sired by males
born in clan X?
What is the proportion of litters for which at least
one cub has one father born in clan X?
What is the proportion of cubs that were sired by
philopatric males?
What is the proportion of cubs that were sired by
primary dispersers?
What is the proportion of cubs with known fathers that
were sired by males that were members of clans other
than the clan of the female at conception?1 Try
2 Tries
3 Tries
Figure11: Foreachquestion,thebarlengthdenotesthenumberofthe14evaluatedLLMsthatreturnedthecorrect
answer, illustrating the relative difficulty of the questions. Additionally, the bars are colored according to the number of
amendment tries to solve them.
4.6 Discussion
Most large language models (LLMs) included in this experiment performed well on the questions in Appendix D, with
two LLMs (o3-mini and GPT 5.2) generating all nine queries correctly and eleven others achieving seven or eight
correctanswers. OnlyoneLLM,deepseek-r1:70b,generatedonlyfivecorrectqueries. Allowinguserstoiteratively
revisethequerywasstillbeneficial: whilemostcorrectqueriesweregeneratedonthefirstattempt,anon-negligible
sharerequired asecondor thirdamendment, especiallyforthe modelsgemma3:27b, llama3.3:70bandnemotron. Our
analysis further revealed which question structures were more difficult than others. For example, matching authors to a
specificsoftwarepackagewashandledreliablybyeverymodel,whereasidentifyingsoftwarepackageswhichsharedthe
same authors remained impossible for eleven of the fourteen LLMs.
However,strongperformanceoncarefullydesignedquestionsdidnottransfertothesecond,expert-drivencasestudy
in every case. When we applied the same pipeline to the Ngorongoro Hyena Knowledge Graph and five questions
formulatedbydomainexperts,performancegapswidenedvisibly. OpenAIo3anddeepseek-reasoner-apiperformed
best (5/5 questions answered correctly), o4-mini, o1 and GPT 5.2 reached 4/5, Claude 3.7 Sonnet and o3-mini 3/5;
among Ollama models, only qwq registered any correct answers (2/5), with the rest at 0. Efficiency also diverged:
o4-minimostoftensucceededonthefirstattempt,whereasothersuccessfulmodelsmostlyleanedonsecondorthird
tries. Had we relied on the MaRDI experiment alone, these differences would have remained hidden.
Two implications follow. First, expert-sourced questions are essential for discriminating among models that appear
comparableonsynthetictests,andseveralKnowledgeGraphsshouldbeusedtoevaluateamethod. Second,ifastrategy
with several amendment attempts is employed, efficiency, meaning how many tries a model needs, should be reported
alongsidethenumberofcorrectanswers,becausearisingnumberofretriesmakestheuseofthetoolmoreinconvenient.
In short, realistic questions and anadditional KnowledgeGraph revealedlimitations ofthe pipelineusing current LLMs,
and future evaluations should not ignore the realism of questions and potential differences between domains.
5 Conclusion
Weintroducedan interactiveframeworkthattranslates natural-languagequestionsinto auditable Cypher, explainsthe
resulting queries, and lets users iteratively amend them. This pipeline brings transparency and control to KG-based
22

A preprint - February 6, 2026
questionansweringwithoutaskinguserstolearnaquerylanguage. Acrossthreestudies,weshowedthatLLMscan
shoulderdistinctrolesinthispipelinewithvaryingskill: theyexplainandsanity-checkqueries(section3),generate
workable queries on a real mathematical KG (subsection 4.4), and remain useful, though uneven, on expert-written
questionsinabiologicalKG(subsection4.5). Onexplanationquality,severalmodelsproducedcorrectandcomplete
one-sentencesummariesformostmovie-graphqueries,yetarecurringfailurewasdroppingexplicityears-evidence
that brevity pressures can harm faithfulness. These results form the most detailed and statistically grounded part of our
evaluation. Concerning query generation, many models solved the MaRDI tasks on the first attempt, while the Hyena
studyexposedwidergaps: onlyafewmodelsconsistentlyreachedcorrectanswers,andefficiency(attemptspersuccess)
varied markedly,underscoringwhy realistic, expert-sourced questionsand cross-domain evaluation matter. These two
studies function as targeted case evaluations of the framework on real KGs, illustrating how model behavior shifts
withdomain,datarealism,andquestionstyle. Futureworkcouldemphasizehuman-centeredandevaluation-oriented
directions: refining explanation prompts to reliably retain temporal and numeric constraints; designing stronger
interactive tooling, such as diff views for query edits and side-by-side result previews to shorten the path from first draft
totrustedquery. Takentogether,ourresultssupportKG-centric,user-in-the-loopnaturallanguageQAsystemswith
transparent, auditable result provenance.
6 Acknowledgments
We want to thank Oliver Höner & Eve Davidian for providing us access to the hyena dataset and for formulating
expert-levelquestions. WethankallothermembersoftheNgorongoroHyenaProjectwhichalsocollecteddataused
here (Philemon Naman, Bettina Wachter & Arjun Dheer). The authors also thank Leonie F. Walter for answering
the hyena data questions independently using the Rpackage hyenaR(without relying on LLM/KG). This work was
supported by the Deutsche Forschungsgemeinschaft (DFG) (project grant 446 460135501).
23

A preprint - February 6, 2026
References
[1]Alejandro Barredo Arrieta et al. “Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities
and challenges toward responsible AI”. In: Information fusion 58 (2020), pp. 82–115.
[2]Eve Davidian et al. “Why do some males choose to breed at home when most other males disperse?” In: Science
Advances 2.3 (2016), e1501236. doi:10.1126/sciadv.1501236 .
[3]ArjunDheeretal.“Diurnalpastoralismdoesnotreducejuvenilerecruitmentnorelevateallostaticloadinspotted
hyenas”.In: JournalofAnimalEcology 91.11(2022),pp.2289–2300. doi:https://doi.org/10.1111/1365-
2656.13812 .
[4]Marion L East et al. “Maternal effects on offspring social status in spotted hyenas”. In: Behavioral Ecology 20.3
(2009), pp. 478–483. doi:https://doi.org/10.1093/beheco/arp020 .
[5]Yanlin Feng, Simone Papicchio, and Sajjadur Rahman. “CypherBench: Towards precise retrieval over full-scale
modern knowledge graphs in the LLM era”. In: arXiv preprint arXiv:2412.18702 (2024).
[6]LaurenceGFrank,StephenEGlickman,andPaulLicht.“Fatalsiblingaggression,precocialdevelopment,and
androgens in neonatal spotted hyenas”. In: Science252.5006 (1991), pp. 702–704. doi:https://doi.org/10.
1126/science.2024122 .
[7] Aidan Hogan et al. “Knowledge graphs”. In: ACM Computing Surveys (Csur) 54.4 (2021), pp. 1–37.
[8]Oliver P Höner et al. “Female mate-choice drives the evolution of male-biased dispersal in a social mammal”. In:
Nature448.7155 (2007), pp. 798–801. doi:https://doi.org/10.1038/nature06040 .
[9]Kurt Hornik. “The comprehensive R archive network”. In: Wiley interdisciplinary reviews: Computational
statistics4.4 (2012), pp. 394–398.
[10]TorstenHothorn,FrankBretz,andPeterWestfall.“SimultaneousInferenceinGeneralParametricModels”.In:
Biometrical Journal 50.3 (2008), pp. 346–363.
[11]LeiHuangetal.“Asurveyonhallucinationinlargelanguagemodels:Principles,taxonomy,challenges,andopen
questions”. In: ACM Transactions on Information Systems 43.2 (2025), pp. 1–55.
[12]Hao Liu et al. “Hoprag: Multi-hop reasoning for logic-aware retrieval-augmented generation”. In: arXiv preprint
arXiv:2502.12442 (2025).
[13]Makbule Gulcin Ozsoy et al. “Text2cypher: Bridging natural language and graph databases”. In: arXiv preprint
arXiv:2412.10064 (2024).
[14]IsabellaPetersetal.“Zenodointhespotlightoftraditionalandnewmetrics”.In: Frontiersinresearchmetrics
and analytics 2 (2017), p. 13.
[15]MatteoPetreraetal.“zbMATHOpen:APIsolutionsandresearchchallenges”.In: arXivpreprintarXiv:2106.04664
(2021).
[16]Larissa Pusch and Tim OF Conrad. “Combining LLMs and Knowledge Graphs to Reduce Hallucinations in
Question Answering”. In: arXiv preprint arXiv:2409.04181 (2024).
[17]François Rousset and Jean-Baptiste Ferdy. “Testing environmental and genetic effects in the presence of spatial
autocorrelation”. In: Ecography 37.8 (2014), pp. 781–790.
[18]MoritzSchubotzetal.“BravoMaRDI:awikibasepoweredknowledgegraphonmathematics”.In: arXivpreprint
arXiv:2309.11484 (2023).
[19]Amit Singhal. Introducing the Knowledge Graph: Things, Not Strings .https://blog.google/products/
search/introducing-knowledge-graph-things-not/ . Accessed 28 May 2025. 2012.
[20]Aman Tiwari et al. “Auto-Cypher: Improving LLMs on Cypher generation via LLM-supervised generation-
verification framework”. In: arXiv preprint arXiv:2412.12612 (2024).
[21]SiyunZhaoet al.“Retrievalaugmentedgeneration(rag) andbeyond:Acomprehensivesurveyonhowtomake
your llms use external data more wisely”. In: arXiv preprint arXiv:2409.14924 (2024).
[22]Zulun Zhu et al. “Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Compre-
hensive Survey”. In: arXiv preprint arXiv:2504.10499 (2025).
A Prompts
A.1 Explanation Prompt
You are a helpful assistant.
I have a Cypher query:
24

A preprint - February 6, 2026
{query}
Please do these things:
Provide a concise, step-by-step explanation of what this query does. Keep it as
short as necessary to be clear and thorough, do not include unnecessary details.←↪
Conclude with a one- or two-sentence summary describing the query’s overall purpose.
When trying to explain and summarize this query, please tell me if you noticed that
something was wrong with the query, including:←↪
- relationships that are the wrong way around, for example:
(b:Book)-[:READS]->(r:Scientist), where the correct one would be
(b:Book)<-[:READS]-(r:Scientist)←↪
←↪
- relationship names that don ´t make any sense in the context of the nodes they
connect, for example: (r:Scientist)-[:SOLVES]->(b:Book); this is wrong because a
book can 't solve a scientist←↪
←↪
- a node name that makes no sense in the context:
(r:Scientist)-[:READS]->(b:Airplane); this is wrong because you can 't read an
airplane←↪
←↪
- other things that you notice that don 't make sense
A.2 Cypher Generation Prompt
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
The cypher statement should only return nodes that are specifically asked for in the
question.←↪
Do not make the cypher query unnecessarily complex.
When the question asks for "What NODE_LABEL has X", the answer should be only the node
name, not other details.←↪
Cypher requires aggregate expressions, like COUNT(s), in the RETURN clause if you’re
using them in the ORDER BY clause.←↪
Example: MATCH (t:Tree)-[r:WAS_CUT]->(:Event) RETURN t.name, COUNT(*) AS cuttings ORDER
BY cuttings←↪
If it makes sense for the specific question and relationship, you can use bidirectional
matching to matche the relationship in both directions.←↪
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a
Cypher statement.←↪
Do not include any text except the generated Cypher statement.
The question is:
{question}
A.3 Amendment Prompt
You are given an existing Cypher query, a Knowledge Graph schema and an amendment
request. Modify the Cypher query according to the amendment request.←↪
The cypher query was generated on the basis of this question:
{question}
Existing Cypher query:
{current_query}
Graph schema:
{schema}
Amendment request:
25

A preprint - February 6, 2026
{amendment}
Provide the updated Cypher query only, without any explanations.
A.4 Hyena Generation prompt
A line about how the sexes are entered into the graph is added in the prompt so the query does not fail because of
different ways of writing this.
cypher_generation_template = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
The cypher statement should only return nodes that are specifically asked for in the
question.←↪
Do not make the cypher query unnecessarily complex.
When the question asks for "What NODE\_LABEL has X", the answer should be only the node
name, not other details.←↪
Cypher requires aggregate expressions, like COUNT(s), in the RETURN clause if you’re
using them in the ORDER BY clause.←↪
Example: MATCH (t:Tree)-[r:WAS_CUT]->(:Event) RETURN t.name, COUNT(*) AS cuttings ORDER
BY cuttings←↪
If it makes sense for the specific question and relationship, you can use bidirectional
matching to matche the relationship in both directions.←↪
The sexes in the graph are "male" and "female".
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a
Cypher statement.←↪
Do not include any text except the generated Cypher statement.
The question is:
{question}"""
B SPARQL
# get cran packages that cite papers that have a de nr (an identification number from zbMATH)
SELECT ?item ?itemLabel ?target ?targetLabel ?de_nr ?author ?authorLabel ?author_id
WHERE
{
?item wdt:P31 wd:Q57080.
?item wdt:P223 ?target.
?target wdt:P1451 ?de_nr.
?target wdt:P16 ?author.
?author wdt:P676 ?author_id
SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
#get cran packages that have authors that have a zbMATH author id
SELECT ?item ?itemLabel ?target ?targetLabel ?author_id WHERE
{
?item wdt:P31 wd:Q57080.
?item wdt:P16 ?target.
?target wdt:P676 ?author_id
SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
#get zenodo packages that have authors with a zbMATH author id
SELECT ?item ?itemLabel ?target ?targetLabel ?author_id WHERE
26

A preprint - February 6, 2026
{
?item wdt:P227 ?b.
?item wdt:P16 ?target.
?target wdt:P676 ?author_id
SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
C Explanation Queries
•Query: Cypherquery to be explained
•N Hops: Number of relationships between nodes in query
•Clause type: Types of clauses included in the query
•Perturbation: Type of error introduced into the query
Query N
HopsClause
TypePerturbation
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) RETURN p, m1simple None
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie) RETURN p, m1simple flip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie) RETURN p, m1simple nonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
RETURN p, m1simple nonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie) RETURN celebrity, m1simple no node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m,
c2simple None
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m,
c2simple flip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m, c2simple nonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Food)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m, c2simple nonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN
celebrity, m, c2simple no node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct3simple None
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct3simple flip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct3simple nonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->(ct:City)
RETURN p, m, c, ct3simple nonsense node
name
27

A preprint - February 6, 2026
Query N
HopsClause
TypePerturbation
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]- (c:Critic)
-[:BIRTH_CITY]-> (ct:City) RETURN celebrity, m, c,
ct3simple no node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p, m, m21
(+1)optional
matchNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]
-(m:Movie) OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p, m, m21
(+1)optional
matchflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]
->(m:Movie) OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p, m, m21
(+1)optional
matchnonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie) RETURN p,
m, m21
(+1)optional
matchnonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]
->(m:Movie) OPTIONAL MATCH (celebrity)-[:DIRECTED]
->(m2:Movie) RETURN celebrity, m, m21
(+1)optional
matchno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]
->(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) OPTIONAL MATCH
(p)-[:DIRECTED]->(m2:Movie) RETURN p, m, c, m22
(+1)optional
matchNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]
-(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) OPTIONAL MATCH
(p)-[:DIRECTED]->(m2:Movie) RETURN p, m, c, m22
(+1)optional
matchflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]
->(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) OPTIONAL MATCH
(p)-[:DIRECTED]->(m2:Movie) RETURN p, m, c, m22
(+1)optional
matchnonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic) OPTIONAL MATCH
(p)-[:DIRECTED]->(m2:Movie) RETURN p, m, c, m22
(+1)optional
matchnonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) OPTIONAL MATCH
(celebrity)-[:DIRECTED]->(m2:Movie) RETURN celebrity,
m, c, m22
(+1)optional
matchno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]
->(ct:City) OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p, m, c, ct, m23
(+1)optional
matchNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]
->(ct:City) OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p, m, c, ct, m23
(+1)optional
matchflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie)
RETURN p, m, c, ct, m23
(+1)optional
matchnonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY] ->(ct:City)
OPTIONAL MATCH (p)-[:DIRECTED]->(m2:Movie) RETURN p,
m, c, ct, m23
(+1)optional
matchnonsense node
name
28

A preprint - February 6, 2026
Query N
HopsClause
TypePerturbation
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) OPTIONAL MATCH (celebrity)-[:DIRECTED]->
(m2:Movie) RETURN celebrity, m, c, ct, m23
(+1)optional
matchno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WITH p, m MATCH (m)<-[:ACTED_IN]-
(coActor:Person) RETURN p, m, collect(DISTINCT
coActor) AS coactors1
(+1)with
clauseNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie) WITH p, m MATCH (m)<-[:ACTED_IN]-
(coActor:Person) RETURN p, m, collect(DISTINCT
coActor) AS coactors1
(+1)with
clauseflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie) WITH p, m MATCH (m)<-[:ACTED_IN]-
(coActor:Person) RETURN p, m, collect(DISTINCT
coActor) AS coactors1
(+1)with
clausenonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
WITH p, m MATCH (m)<-[:ACTED_IN]- (coActor:Person)
RETURN p, m, collect(DISTINCT coActor) AS coactors1
(+1)with
clausenonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WITH celebrity, m MATCH
(m)<-[:ACTED_IN]-(coActor:Person) RETURN celebrity,
m, collect(DISTINCT coActor) AS coactors1
(+1)with
clauseno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WITH p, m,c
MATCH (m)<-[:ACTED_IN]-(coActor:Person) RETURN p, m,
c, collect(DISTINCT coActor) AS coactors2
(+1)with
clauseNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WITH p, m, c
MATCH (m)<-[:ACTED_IN]-(coActor:Person) RETURN p, m,
c, collect(DISTINCT coActor) AS coactors2
(+1)with
clauseflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WITH p, m, c
MATCH (m)<-[:ACTED_IN]-(coActor:Person) RETURN p, m,
c, collect(DISTINCT coActor) AS coactors2
(+1)with
clausenonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic) WITH p, m, c MATCH
(m)<-[:ACTED_IN]-(coActor:Person) RETURN p, m, c,
collect(DISTINCT coActor) AS coactors2
(+1)with
clausenonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WITH celebrity,
m, c MATCH (m)<-[:ACTED_IN]-(coActor:Person) RETURN
celebrity, m, c, collect(DISTINCT coActor) AS coactors2
(+1)with
clauseno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WITH p, m, c, ct MATCH
(m)<-[:ACTED_IN]-(coActor:Person) RETURN p, m, c, ct,
collect(DISTINCT coActor) AS coactors3
(+1)with
clauseNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WITH p, m, c, ct MATCH (m)<-[:ACTED_IN]-
(coActor:Person) RETURN p, m, c, ct, collect(DISTINCT
coActor) AS coactors3
(+1)with
clauseflip relationship
29

A preprint - February 6, 2026
Query N
HopsClause
TypePerturbation
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WITH p, m, c, ct MATCH
(m)<-[:ACTED_IN]-(coActor:Person) RETURN p, m, c, ct,
collect(DISTINCT coActor) AS coactors3
(+1)with
clausenonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->(ct:City)
WITH p, m, c, ct MATCH (m)<-[:ACTED_IN]-
(coActor:Person) RETURN p, m, c, ct, collect(DISTINCT
coActor) AS coactors3
(+1)with
clausenonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WITH celebrity, m, c, ct MATCH
(m)<-[:ACTED_IN]-(coActor:Person) RETURN celebrity,
m, c, ct, collect(DISTINCT coActor) AS coactors3
(+1)with
clauseno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WHERE m.release_year > 2020 RETURN p, m1where
clauseNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie) WHERE m.release_year > 2020 RETURN p, m1where
clauseflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie) WHERE m.release_year > 2020 RETURN p, m1where
clausenonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
WHERE m.release_year > 2020 RETURN p, m1where
clausenonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WHERE m.release_year > 2020 RETURN
celebrity, m1where
clauseno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WHERE m.release_year = -1 RETURN p, m1where
clauseillogical where
value
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WHERE m.title > 2020 RETURN p, m1where
clausewrong where
type
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) WHERE m.release_year > 2020 AND
m.release_year < 2010 RETURN p, m1where
clausecontradictory
where values
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE
m.release_year > 2020 RETURN p, m, c2where
clauseNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE
m.release_year > 2020 RETURN p, m, c2where
clauseflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE
m.release_year > 2020 RETURN p, m, c2where
clausenonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic) WHERE m.release_year >
2020 RETURN p, m, c2where
clausenonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE
m.release_year > 2020 RETURN celebrity, m, c2where
clauseno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE
m.release_year = -1 RETURN p, m, c2where
clauseillogical where
value
30

A preprint - February 6, 2026
Query N
HopsClause
TypePerturbation
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE m.title
> 2020 RETURN p, m, c2where
clausewrong where
type
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) WHERE
m.release_year > 2020 AND m.release_year < 2010 RETURN
p, m, c2where
clausecontradictory
where values
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.release_year > 2020 RETURN p, m, c,
ct3where
clauseNone
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.release_year > 2020 RETURN p, m, c,
ct3where
clauseflip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.release_year > 2020 RETURN p, m, c,
ct3where
clausenonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->(ct:City)
WHERE m.release_year > 2020 RETURN p, m, c, ct3where
clausenonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.release_year > 2020 RETURN
celebrity, m, c, ct3where
clauseno node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.release_year = -1 RETURN p, m, c, ct3where
clauseillogical where
value
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.title > 2020 RETURN p, m, c, ct3where
clausewrong where
type
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) WHERE m.release_year > 2020 AND
m.release_year < 2010 RETURN p, m, c, ct3where
clausecontradictory
where values
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) RETURN p, m, CASE WHEN m.release_year <
2010 THEN ’old’ ELSE ’new’ END as movie_age1case None
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie) RETURN p, m, CASE WHEN m.release_year <
2010 THEN ’old’ ELSE ’new’ END as movie_age1case flip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie) RETURN p, m, CASE WHEN m.release_year < 2010
THEN ’old’ ELSE ’new’ END as movie_age1case nonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
RETURN p, m, CASE WHEN m.release_year < 2010 THEN
’old’ ELSE ’new’ END as movie_age1case nonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie) RETURN celebrity, m, CASE WHEN
m.release_year < 2010 THEN ’old’ ELSE ’new’ END as
movie_age1case no node label,
misleading name
31

A preprint - February 6, 2026
Query N
HopsClause
TypePerturbation
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) RETURN p, m, CASE WHEN m.release_year =
-1 THEN ’old’ ELSE ’new’ END as movie_age1case illogical where
value
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie) RETURN p, m, CASE WHEN m.title < 2010 THEN
’old’ ELSE ’new’ END as movie_age1case wrong where
type
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m,
c, CASE WHEN m.release_year < 2010 THEN ’old’ ELSE
’new’ END as movie_age2case None
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m,
c, CASE WHEN m.release_year < 2010 THEN ’old’ ELSE
’new’ END as movie_age2case flip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m, c,
CASE WHEN m.release_year < 2010 THEN ’old’ ELSE ’new’
END as movie_age2case nonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m, c, CASE WHEN
m.release_year < 2010 THEN ’old’ ELSE ’new’ END as
movie_age2case nonsense node
name
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN
celebrity, m, c, CASE WHEN m.release_year < 2010 THEN
’old’ ELSE ’new’ END as movie_age2case no node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m,
c, CASE WHEN m.release_year = -1 THEN ’old’ ELSE ’new’
END as movie_age2case illogical where
value
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic) RETURN p, m,
c, CASE WHEN m.title < 2010 THEN ’old’ ELSE ’new’ END
as movie_age2case wrong where
type
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct, CASE WHEN m.release_year
< 2010 THEN ’old’ ELSE ’new’ END as movie_age3case None
MATCH (p:Person {name: "Alice"})<-[:ACTED_IN]-
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct, CASE WHEN m.release_year
< 2010 THEN ’old’ ELSE ’new’ END as movie_age3case flip relationship
MATCH (p:Person {name: "Alice"})-[:LIKES_TO_EAT]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct, CASE WHEN m.release_year
< 2010 THEN ’old’ ELSE ’new’ END as movie_age3case nonsenserelation
name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->(m:Food)
<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->(ct:City)
RETURN p, m, c, ct, CASE WHEN m.release_year < 2010
THEN ’old’ ELSE ’new’ END as movie_age3case nonsense node
name
32

A preprint - February 6, 2026
Query N
HopsClause
TypePerturbation
MATCH (celebrity {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN celebrity, m, c, ct, CASE WHEN
m.release_year < 2010 THEN ’old’ ELSE ’new’ END as
movie_age3case no node label,
misleading name
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct, CASE WHEN m.release_year
= -1 THEN ’old’ ELSE ’new’ END as movie_age3case illogical where
value
MATCH (p:Person {name: "Alice"})-[:ACTED_IN]->
(m:Movie)<-[:HAS_FAVORITE]-(c:Critic)-[:BIRTH_CITY]->
(ct:City) RETURN p, m, c, ct, CASE WHEN m.title < 2010
THEN ’old’ ELSE ’new’ END as movie_age3case wrong where
type
Table 11: Explanation queries
D MaRDI Experiment Questions
•Question: Input Question
•Type:Query features that are included in the reference query
•Reference CypherQuery: Cypherquerythatoutputsthedesiredresults,whichtheoutputfromthegenerated
query can be compared to
33

A preprint - February 6, 2026
Table 12: MaRDI experiment questions, question types and example Cypher queries
Question Type Reference Cypher Query
Whichauthorsdoesthesoftware
package graphclust have?simple MATCH (sp:SoftwarePackage {name:"graphclust"})-[:HAS_AUTHOR]
->(a:Author) RETURN a;
Whichauthorsauthoredpublica-
tions that contain the substring
"Pareto"?where MATCH (p:Publication) WHERE p.title CONTAINS "Pareto" MATCH
(p)-[:HAS_AUTHOR]->(a:Author) RETURN DISTINCT a.name AS
authorName, a.authorId AS authorId
Which are the ten authors that
createdthemostsoftwarepack-
ages?top n MATCH (a:Author)<-[:HAS_AUTHOR]-(s:SoftwarePackage) RETURN
a.name AS authorName, a.authorId AS authorId, COUNT(s) AS
packageCount ORDER BY packageCount DESC LIMIT 10
HowmanydatasetsdidRobHyn-
dman create?counts MATCH (a:Author { name: "Rob Hyndman"
})<-[:HAS_AUTHOR]-(d:Dataset) RETURN count(d) AS
numberOfDatasets
Givemeallpeoplethatcreated
at least 5 datasets.distinct MATCH (p:Author)<-[:HAS_AUTHOR]-(d:Dataset) WITH p,
COUNT(d) AS numberOfDatasets WHERE numberOfDatasets >=
5 RETURN p.name AS authorName, numberOfDatasets ORDER BY
numberOfDatasets DESC
Getmeallauthorsand,ifpresent,
any datasets they are tied to.optional match MATCH (a:Author) OPTIONAL MATCH (a)<-[:HAS_AUTHOR]
-(d:Dataset) RETURN a.name AS authorName, a.authorId AS
authorId, COLLECT(d) AS datasets
Whichsoftwarepackagesshare
the same authors?shared MATCH (p:SoftwarePackage)-[:HAS_AUTHOR]->(a:Author)
WITH p, COLLECT(DISTINCT a.authorId) AS authorIds WITH
p, apoc.coll.sort(authorIds) AS sortedAuthorIds WITH
sortedAuthorIds, COLLECT(p) AS packages WHERE SIZE(packages)
> 1 RETURN sortedAuthorIds AS authorGroup, packages
Which authors worked on both
thedataset"BitcoinDatasetwith
MissingValues"and"Rideshare
Dataset without Missing Val-
ues"?intersection MATCH (a:Author)<-[:HAS_AUTHOR]-(d1:Dataset),
(a)<-[:HAS_AUTHOR]-(d2:Dataset) WHERE d1.name = "Bitcoin
Dataset with Missing Values" AND d2.name = "Rideshare
Dataset without Missing Values" RETURN a.name AS authorName,
a.authorId AS authorId
Which software packages have
no authors?not MATCH (p:SoftwarePackage) WHERE NOT (p)-[:HAS_AUTHOR]
->(:Author) RETURN p.packageId AS packageId, p.name AS
packageName
E Hyena Experiment Questions
•Question: Input Question
•N Hops: Number of relationships between nodes in the path based on the question
•Includes Calculation: If the question includes a calculation, such as a proportion
•IncludesComplicatedNicheConcept: Ifthequestionincludesaconceptwhichonlyanexpertinthefield
would understand
34

A preprint - February 6, 2026
Table 13: Hyena experiment questions
Question N Hops Includes Calculation IncludesComplicatedNiche
Concept
What is the proportion of cubs that were
sired by males born in clan X?2 True False
What is the proportion of litters for which
at leastone cubhas onefather bornin clan
X?3 True True
What is the proportion of cubs that were
sired by philopatric males?3 True True
What is the proportion of cubs that were
sired by primary dispersers?3 True True
What is the proportion of cubs with known
fathers that were sired by males that were
members of clans other than the clan of the
female at conception?6 True False
F Detailed Results
F.1 Explanation trials
The disagreements between models for the correct one-sentence summary are reported in Table 14. Disagreements
related to correct perturbation detection appear in Table 15, and disagreements for identifying false positives are
summarized in Table 16.
Table14: Totalnumberofdisagreementsbetweenthemodelsintheexplanationtrialwhengeneratingone-sentence-
summaries, further split up into the ones where only the row model or only the column model was correct
row_model col_model row_only col_only total_diff
claude 3.7 sonnet deepseek-reasoner-api 7 26 33
claude 3.7 sonnet deepseek-r1:70b 9 22 31
claude 3.7 sonnet o1-preview 1 23 24
claude 3.7 sonnet o3-mini 5 22 27
deepseek-reasoner-api deepseek-r1:70b 15 9 24
deepseek-reasoner-api o1-preview 7 10 17
deepseek-reasoner-api o3-mini 12 10 22
deepseek-r1:70b o1-preview 7 16 23
deepseek-r1:70b o3-mini 9 13 22
o1-preview o3-mini 13 8 21
35

A preprint - February 6, 2026
Table 15: Total number of disagreements between the models in the additional trial, further split up into the ones where
only the row model or only the column model was correct.
row_model col_model row_only col_only total_diff
claude 3.7 sonnet deepseek-reasoner-api 7 10 17
claude 3.7 sonnet deepseek-r1:70b 17 4 21
claude 3.7 sonnet o1-preview 6 8 14
claude 3.7 sonnet o3-mini 14 8 22
deepseek-reasoner-api deepseek-r1:70b 16 0 16
deepseek-reasoner-api o1-preview 3 2 5
deepseek-reasoner-api o3-mini 9 0 9
deepseek-r1:70b o1-preview 0 15 15
deepseek-r1:70b o3-mini 1 8 9
o1-preview o3-mini 8 0 8
Table16: Totalnumberofdisagreementsbetweenthemodelsinthefalse-positiveexperiment,splitintocaseswhere
only the row model or only the column model was correct.
row_model col_model row_only col_only total_diff
claude 3.7 sonnet deepseek-reasoner-api 0 5 5
claude 3.7 sonnet deepseek-r1:70b 3 1 4
claude 3.7 sonnet o1-preview 0 4 4
claude 3.7 sonnet o3-mini 0 5 5
deepseek-reasoner-api deepseek-r1:70b 7 0 7
deepseek-reasoner-api o1-preview 1 0 1
deepseek-reasoner-api o3-mini 0 0 0
deepseek-r1:70b o1-preview 0 6 6
deepseek-r1:70b o3-mini 0 7 7
o1-preview o3-mini 0 1 1
36