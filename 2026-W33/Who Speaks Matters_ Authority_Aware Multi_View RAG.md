# Who Speaks Matters: Authority-Aware Multi-View RAG over Italian Parliamentary Proceedings

**Authors**: Mirko Tritella, Riccardo Pozzi, Matteo Palmonari

**Published**: 2026-08-13 16:07:23

**PDF URL**: [https://arxiv.org/pdf/2608.13410v1](https://arxiv.org/pdf/2608.13410v1)

## Abstract
Parliamentary proceedings are a primary record of democratic deliberation, yet their volume and fragmentation make multi-perspective access difficult for citizens, journalists, and researchers. Applying Retrieval-Augmented Generation (RAG) to parliamentary transcripts introduces three specific risks: dominance of the most frequent speakers, inability to weight speakers according to topical expertise, and citation misattribution in politically sensitive text. We present ParliamentRAG, a RAG system for the Italian Chamber of Deputies that addresses these risks jointly. Its core contribution is a topic-dependent authority model that estimates each speaker's authority as a function of the current query, combining interpretable components such as profession, education, and previous interventions. Given a user query, the system retrieves relevant speech chunks, identifies topic-relevant experts across parliamentary groups, and generates a summary synthesizing their perspectives, accompanied by supporting quotations. ParliamentRAG is evaluated against Google NotebookLM on 15 policy topics via a two-level protocol combining automated metrics and blind A/B human evaluation by six domain experts. The system achieves higher coverage across political groups (0.97 vs. 0.95), perfect quotation faithfulness (1.00 vs. 0.95), and stronger expert preferences on source-related dimensions, while NotebookLM remains stronger on prose-oriented dimensions.

## Full Text


<!-- PDF content starts -->

Who Speaks Matters: Authority-Aware
Multi-View RAG over Italian Parliamentary
Proceedings⋆
Mirko Tritella1, Riccardo Pozzi1, and Matteo Palmonari1
University of Milano-Bicocca, Milan, Italy
mirkotritella1999@gmail.com
riccardo.pozzi@unimib.it
matteo.palmonari@unimib.it
Abstract. Parliamentary proceedings are a primary record of demo-
cratic deliberation, yet their volume and fragmentation make multi-
perspective access difficult for citizens, journalists, and researchers. Ap-
plying Retrieval-Augmented Generation (RAG) to parliamentary tran-
scripts introduces three specific risks: dominance of the most frequent
speakers, inability to weight speakers according to topical expertise, and
citation misattribution in politically sensitive text. We present Parlia-
mentRAG, a RAG system for the Italian Chamber of Deputies that
addresses these risks jointly. Its core contribution is a topic-dependent
authority model that estimates each speaker’s authority as a function
of the current query, combining interpretable components such as pro-
fession, education, and previous interventions. Given a user query, the
system retrieves relevant speech chunks, identifies topic-relevant experts
across parliamentary groups, and generates a summary synthesizing their
perspectives, accompanied by supporting quotations. ParliamentRAG is
evaluated against Google NotebookLM on 15 policy topics via a two-level
protocol combining automated metrics and blind A/B human evalua-
tion by six domain experts. The system achieves higher coverage across
political groups (0.97 vs. 0.95), perfect quotation faithfulness (1.00 vs.
0.95), and stronger expert preferences on source-related dimensions, while
NotebookLM remains stronger on prose-oriented dimensions.
Keywords:Retrieval-Augmented Generation·Parliamentary NLP·Expert
Finding·Knowledge Graphs·Quotation Faithfulness·Multi Perspective Sum-
marization.
1 Introduction
Parliamentary debates constitute a dense documentary record of democratic
deliberation: every floor speech, legislative act, and committee intervention
⋆Accepted at the In-Use Track of the 25th International Semantic Web Conference
(ISWC 2026). This preprint has not undergone peer review or any post-submission
improvements or corrections. Please cite the ISWC version.
arXiv:2608.13410v1  [cs.AI]  13 Aug 2026

2 M. Tritella et al.
expresses the position of an elected representative on matters of public concern.
At the time of our experiments (February 2026), the ongoing XIX Legislature of
the Italian Chamber of Deputies (October 2022–present) has already produced
608 plenary sessions, 6,010 debates, and 40,416 individual speeches
Manual navigation of such a corpus is excessively costly, and keyword-based
institutional search portals return isolated documents rather than synthesized
multi-perspective answers. Furthermore, a representative’s position on a given
topic typically emerges across multiple sessions and distinct debates rather than
within a single intervention, compounding the difficulty of reconstruction.
Journalists, researchers, civil society organizations, and citizens increasingly
require tools capable of exploring parliamentary activity in a structured and
interpretable way, enabling them to reconstruct political positions, monitor
legislative behavior, and compare the viewpoints of different parliamentary
groups and representatives. Facilitating access to these records is particularly
important for citizens, for whom parliamentary debates constitute a primary
source of information about how elected representatives discuss public issues,
justify political decisions, and position themselves on matters that directly affect
society. However, effective access to parliamentary debates requires more than
document retrieval alone: systems mediating access to institutional records should
provide representative and multi-perspective summaries, preserve attribution and
traceability to original speeches, and avoid over-representing isolated interventions
or highly active speakers.
Large Language Models (LLMs) are well suited for synthesis, but when
applied naively to political text they exhibit three risks in a civic-information
settings: (i) they over-represent the most vocal actors, reflecting the distributional
asymmetry of the corpus rather than the political landscape of the debate [ 24]; (ii)
they flatten all speakers to content-based similarity, ignoring the rich metadata
that distinguishes a topical expert from a tangential commenter; and (iii) they
hallucinate or misattribute citations at a rate that, although low, is incompatible
with journalistic or institutional use [15,18].
In this paper, we presentParliamentRAG, a novel RAG application to help
both politics professionals (e.g., journalists or analysts) and Italian citizens have
a faithful, multi-view, and balanced summary of positions expressed by the ten
parliamentary groups of the Italian Chamber of Deputies1during parliament
debates on a specific topic. Knowledge Graphs (KG) play a crucial role in the
development of the application at different levels: all data are stored as a KG
because of the rich relational information (e.g., Members of Parliament (MPs)
membership of parliamentary groups or MPs signatories for parliamentary acts);
graph-based retrieval is a component of the RAG system; the key authority-based
ranking method proposed in our paper depends explicitly on graph information.
Our main objective is to generate synthetic summaries of parliamentary
debates while respecting three fundamental principles: (i)multi-view repre-
sentation, ensuring that all parliamentary groups are represented rather than
over-relying on frequent speakers; (ii)authority awareness, modeling topical
1https://www.camera.it/leg19/46

Authority-Aware Multi-View Parliamentary RAG 3
authority dynamically so that the most relevant voices emerge per query; and
(iii)quotation traceability, requiring that every quotation corresponds to a
verbatim passage from an official parliamentary transcript. These principles aim
to support transparent and balanced access to parliamentary deliberation.
Figure 1 illustrates an authority-aware, multi-view summary of Italian parlia-
mentary positions on justice reform, including traceable verbatim quotations.
What is the position of the parliamentary groups on the
subject: reform of justice?Share
You•09:38
Chamber of Deputies — XIX Legislature
Introduction
The debate on justice reform was resumed in the Italian Chamber of Deputies, involving 51 members
of Chamber in 78 interventions between 25 October 2022 and 3 February 2026. Parliamentary
sittings, including sessions No. 4, 28, 39, 67 and 68, saw an intense debate on this crucial issue for
the Italian judicial system.
Government Position
The Government, with the Minister of Justice Carlo Nordio, places the reform of justice among its
priorities, despite the complexity of the issue. Nordio says that «I know perfectly well that these topics
are not particularly attractive either from the press or from the controversy, but - I repeat it once again -
they are our priority». The executive is committed to presenting reforms to improve the efficiency of
the judicial system, demonstrating an orientation towards structural intervention.
Positions of the Majority
For Brothers of Italy (Fratelli d’Italia), the party is in favour of reform, stressing the importance of
tackling crucial issues of Italian democracy. Daniela Dondi announces: «I already announce in
advance the favorable vote of the Brothers of Italy (Fratelli d’Italia) group». The party sees in the
reform an essential step to ensure the independence of the judicial system.
The League (Lega - Salvini Premier) opposes the proposed reform, highlighting the need to separate
careers between prosecutors and judges. Ingrid Bisa says that «the separation of careers is first and
foremost a constitutional guarantee reform. It serves to strengthen the impartiality of the judge». The
League (Lega - Salvini Premier) promotes a vision of justice that guarantees the independence of the
system.
Positions of the Opposition
The Democratic Party (Partito Democratico) criticizes the current approach of the majority, stressing
the need for consistent reform. Federico Gianassi points out that «I do not know if Minister Nordio will
win or if those pieces of the majority that think it should not be repealed will win, but I believe, we
believe, that even for these differences to date the majority and the Government, in addition to
spending many important words on the issues of justice, have not yet adopted consequent acts» . The
group considers it essential to ensure the effective implementation of the reforms already in place.Query Comm. Experts Interv. Stats Compass Text Check
Fig. 1.An example of ParliamentRAG’s authority-aware, multi-view summary of the
positions of the Italian parliamentary groups on the subjectreform of justice. Exact
citations are delimited by « and ». The image presents the first part of a longer answer
in Italian, automatically translated into English for this publication.
We operationalize these principles into measurable criteria and evaluate
ParliamentRAG through a two-level protocol. First, automated metrics assess
structural properties of the generated summaries, including coverage and balance

4 M. Tritella et al.
of parliamentary groups, citation fidelity, and alignment between cited speakers
and the topic-dependent authority model, providing a scalable and reproducible
validation of design compliance.
Second, we conduct a blinded expert survey with six domain experts (parlia-
mentary journalists, collaborators, and policy analysts), who evaluate outputs
along qualitative dimensions such as clarity, completeness, perceived balance,
and overall usefulness, as well as providing pairwise preferences.
We compare ParliamentRAG against NotebookLM, a strong commercial
baseline for document-grounded generation. ParliamentRAG achieves perfect
citation faithfulness (1.00 vs. 0.95), near-perfect group coverage (0.97 vs. 0.95),
and is consistently preferred on source-related dimensions, particularly Source
Coverage (5.7:1 preference ratio). While NotebookLM is preferred on prose-
quality aspects, overall satisfaction is comparable, indicating that ParliamentRAG
matches a strong baseline while excelling on the dimensions aligned with its
architectural design.
A live deployment is available at https://www.parliamentrag.it/ and
the source code at https://github.com/Emeierkeio/thesis-ParliamentRAG
under Apache-2.0 license.
2 Related Work
We identify five research threads closely related to the contributions of this work:
(i) Natural Language Processing (NLP) and (ii) Knowledge Graphs (KGs) applied
to the parliamentary domain, (iii) expert and authority modeling in information
retrieval, (iv) fairness- and diversity-aware ranking, and (v) faithful attribution
and grounded generation.
To the best of our knowledge, no prior work integrates structured parliamen-
tary graphs with query-dependent authority modeling and multi-view generation
within a unified RAG architecture.
Parliamentary NLP.Large parliamentary corpora, including ParlaMint [ 10]
and the Italian IPSA corpus [ 11], have supported tasks such as stance detection,
topic modelling, and speaker profiling. Ideological-scaling and embedding-based
approaches estimate corpus-level political positions [ 25,22], but do not support
per-query retrieval or multi-view synthesis. More recent LLM- and RAG-based
systemshaveimprovedaccesstoparliamentarymaterialthroughfacetedsearchfor
the Maltese Parliament [ 2] and hybrid vector–KG retrieval for German Bundestag
debates [ 19]. However, these systems do not jointly address the principles central
to our work: multi-view representation, authority-aware retrieval, and quotation-
level traceability (Section 1).
Knowledge Graphs in the Parliamentary Domain.Parliamentary
Knowledge Graphs and Linked Open Data initiatives have been developed to
improve transparency, interoperability, and large-scale analysis of legislative
proceedings. Examples include LinkedEP for European Parliament debates [ 1],
ParliamentSampo for Finnish parliamentary records [ 14], LinkedSaeima for the
Latvian parliament [ 4], and the DemocraSci Knowledge Graph for the Swiss

Authority-Aware Multi-View Parliamentary RAG 5
Parliament [ 23,5]. In Italy, the Chamber of Deputies exposes legislative data
through the OCD ontology and the official dati.camera.it portal [6]. However,
these resources are mainly designed for structured querying, exploration, and data
integration, rather than for query-dependent reasoning, multi-view generation,
or authority-aware interpretation. Our work addresses this gap by integrating
parliamentary KG data into a retrieval-augmented generation architecture for
multi-view and authority-aware inference (Section 3).
Expert Finding and Authority.Expertfindingformalisestheidentification
of authoritative individuals from text and metadata [ 3], distinguishing profile-
based from document-based approaches. Link-based ranking [ 20,16] yields query-
independent authority. Source-reliability estimation in RAG [ 13] assumes a single
factually correct answer, an assumption that does not hold in parliamentary
debate, where multiple legitimate viewpoints coexist. No prior system computes
interpretable, query-dependentauthority from observable parliamentary features
while modelling temporal decay and coalition-crossing invalidation.
Multi-View Retrieval and Fairness.Fairness-aware ranking [ 27] and
diversification methods such as MMR [ 7] optimise for content or demographic
diversity but are not integrated into RAG pipelines for political applications.
GraphRAG [ 9] exploits document structure but does not enforce coverage of
predefined groups.
Quotation Faithfulness.Attribution frameworks [ 21], post-hoc systems
such as RARR [ 12], and verification methods [ 17] improve quotation quality, but
the generator remains free to paraphrase the source. Our ParliamentaryRAG
system, instead, provides aby-constructionguarantee that every quotation is a
verbatim substring of a specific source document.
3 Parliamentary Knowledge Graph
ParliamentRAG operates over a knowledge graph (KG) that integrates parliamen-
taryproceedings,metadata,andlegislativeactivityintoaunified,Italian-language
representation. The underlying data is sourced from the open data infrastructure
of the Italian Chamber of Deputies [ 6], which publishes semantically structured
records in RDF format according to theOntology of the Chamber of Deputies.2
The dataset includes plenary and committee transcripts, legislative acts with
signatories, deputy biographical information, and committee memberships with
temporal validity.
The RDF data is transformed into a property graph stored in a single Neo4j3
instance, which serves as the sole data layer for the entire system. This architec-
tural choice is motivated by three factors: (i) the need to represent relationships
with properties (e.g., temporal validity such as start_date and end_date on
MEMBER_OF_GROUP andMEMBER_OF_COMMITTEE ; see Figure 2), which are naturally
handled in a property graph model; (ii) the integration of vector similarity search
within the same system, avoiding the overhead of a separate vector store; and
2https://dati.camera.it/ocd/classi.rdf
3https://neo4j.com/

6 M. Tritella et al.
Session
Debate
Phase
Speech
ChunkDeputy
Gov. MemberParl. Group
CommitteeParl. Act
HAS_DEBATE
HAS_PHASE
CONTAINS_SPEECH
HAS_CHUNK
NEXTSPOKEN_BY
SPOKEN_BYIS_PRESIDENT
IS_VICE_PRESIDENT
IS_SECRETARYCO_SIGNATORY
GOVERNMENT_REFERENCEPRIMARY_SIGNATORYMEMBER_OF_GROUP
(start_date, end_date)
MEMBER_OF_COMMITTEE
(start_date, end_date)IS_PRESIDENT
group presidency
Fig. 2.Knowledge graph schema stored in a single Neo4j instance.
(iii) the ability to combine graph traversal, keyword search, and dense retrieval
within a unified query infrastructure.
During transformation, all entities and relationships from the source ontology
are preserved. The graph includes 387 deputies (out of 400 elected members; 13
never intervened in plenary debate and thus have no associated speeches), 64
government members (only those who delivered at least one speech during a
Chamber session), 10 parliamentary groups, 80 committees, 608 sessions, 6,010
debates, 6,515 phases, 40,416 speeches, and 27,576 legislative acts. The graph is
further enriched with information not present in the RDF source: the educational
background and profession of MPs, extracted from their short biographical de-
scriptions on the Chamber’s website, are embedded and stored as node properties
(education_embedding ,profession_embedding ). Semi-structured parliamen-
tary data—debate and session transcripts published in XML format4—are parsed
to extract the hierarchical structure of proceedings and integrated into the graph
alongside institutional data. The resulting graph contains 232,755 nodes and
488,487 relationships across 13 node labels and 15 relationship types.
The overall schema is shown in Figure 2. Parliamentary proceedings are
organized as a hierarchical structureSession →Debate →Phase →Speech
→Chunk, mirroring the institutional workflow. Speeches are segmented into
151,073 chunks, which serve as the atomic retrieval units. Each chunk stores its
textual content, a 1,536-dimensional dense embedding (text-embedding-3-small),
its position within the speech, and character-level offsets ( start_char_raw ,
end_char_raw ) into the original transcript, allowing retrieved evidence to be
traced back to verbatim source text. Sequential chunks are linked by NEXTedges
(110,657 links), supporting context-window expansion during generation.
4https://documenti.camera.it/apps/commonServices/getDocumento.ashx?
sezione=assemblea&tipoDoc=formato_xml&tipologia=stenografico&idNumero=
0655&idLegislatura=19

Authority-Aware Multi-View Parliamentary RAG 7
Raw transcripts contain both substantive content and procedural elements
(e.g., voting announcements or formal statements), which are removed through
lightweight preprocessing. To preserve exact traceability, an alignment between
cleaned and raw text is maintained, enabling deterministic reconstruction of
original passages for quotation grounding.
Beyond the textual hierarchy, the graph captures the political and institutional
context of speakers. Deputies are linked to parliamentary groups and committees
through time-qualified relationships (505 group memberships, 1,515 committee
memberships), allowing affiliation to be resolved at any point in time—a capabil-
ity essential for coalition-crossing detection in the authority scoring (Section 4).
They are also connected to legislative acts via PRIMARY_SIGNATORY (27,373)
and CO_SIGNATORY (103,959) edges, providing a structural signal of topic en-
gagement complementary to speech content. Institutional roles ( IS\_PRESIDENT ,
IS_VICE_PRESIDENT ,IS_SECRETARY ) are recorded for committees and parlia-
mentary groups, and are used as signals in authority estimation.
The graph supports three types of access: dense semantic search over chunk
embeddings (via Neo4j’s native vector index), keyword search over act titles (via
a full-text index), and structural graph traversal. This unified representation lets
the system combine text and structure, supporting retrieval (e.g., traversing from
a parliamentary act to its signatories and their speeches), authority modelling
(e.g., aggregating a deputy’s legislative activity across signed acts), and quotation
grounding (i.e., linking each generated quote back to the exact source passage
via stored character offsets).
4 ParliamentRAG
ParliamentRAG is a web application built on top of Next.js5and FastAPI6that
provides an interactive interface for exploring Italian parliamentary proceedings
through retrieval-augmented generation. The system is publicly accessible at
https://www.parliamentrag.it.
The interface supports free-text queries over parliamentary debates and
presents structured, multi-perspective answers grounded in official transcripts. In
addition to the main search functionality, it offers auxiliary tools for targeted
exploration, including filtered search over legislative acts and speeches, authority
analysis of deputies on specific topics, and visualization of parliamentary positions.
We now describe the underlying pipeline that transforms a user query into
a multi-view, quotation-grounded response, jointly modelingwhat is saidand
who is saying it. The system follows a retrieval–ranking–generation pipeline in
which speaker authority, coverage of political groups, and quotation faithfulness
are enforced as first-class constraints.
Given a query q, the system (i) retrieves candidate evidence from parlia-
mentary proceedings via a dual-channel strategy, (ii) reranks evidence using a
composite score that incorporates query-dependent speaker authority and group
5https://nextjs.org/
6https://fastapi.tiangolo.com/

8 M. Tritella et al.
0. Query
Rewriting
1. Query
Embedding
2a. Dense
Channel2b. Graph
Channel
3. Merger &
Reranking5. Expert
Computation
6. Generation
Pipeline
Response4. Authority
Scoringa. Analyze
b. Write
c. Integrate
d. Cite
Fig. 3.Processing Pipeline. Steps 2a and 2b are executed in parallel.
coverage, (iii) selects representative experts for each parliamentary group, and
(iv) generates a structured multi-view answer with verbatim quotations extracted
from source documents. A schema of the pipeline is depicted in Figure 3.
4.1 Dual-channel Retrieval
The retrieval stage must identify speech chunks that are (i) semantically relevant
with respect to the user query, (ii) representative of all parliamentary groups,
and (iii) reflective of speakers whose expertise is evidenced by legislative activity
rather than solely by speech content.
To this end, we adopt a dual-channel approach. Adense channelperforms
vector similarity search over speech chunks, capturing thematically relevant
content independently of the speaker. Later, graph traversal is used to resolve
the associated speech, speaker, and parliamentary group, also evaluating if
the speaker’s current group differs from that at the time of the speech. The
primary limitation of the dense channel is that it cannot distinguish between
a deputy who has spoken tangentially about a topic and one who has actively
sponsored legislation in the same domain. Thegraph channelretrieval is based on
parliamentaryacts.First,actsrelevanttothequeryareselectedwithhybridlexical
and dense, semantic similarity, then the primary signatories and co-signatories
are obtained with graph traversal, as well as their speeches and chunks.
Additionally, when the query can be mapped to a parliamentary committee
via a curated keyword-based mapping7, the retrieval engine additionally privileges
chunks from speakers who are members of the relevant committee (e.g., justice or
foreign affairs; this information is found in the KG), since committee membership
is a strong institutional proxy for domain expertise.
7https://github.com/Emeierkeio/thesis-ParliamentRAG/blob/main/backend/
config/commissioni_topics.yaml

Authority-Aware Multi-View Parliamentary RAG 9
The graph channel explicitly considers expert-authored legislative activity.
A deputy who has co-signed a relevant act is, by this criterion, likely to have
engaged substantively with the topic. This signal is invisible to the dense channel,
which operates exclusively on chunk-level semantic similarity. The combination
of the two channels yields a more diverse result set, spanning a broader range of
political groups and evidence types, than either channel alone.
4.2 Authority-Aware Reranking and Expert Computation
Retrievedevidenceismergeddeduplicatingitemsbychunkidentifierandretaining
the higher score. Each evidence itemeis ranked using a weighted score:
score(e) =w rr(e) +w dd(e) +w vv(e) +w aauthority(s e, q) +w σσ(e),(1)
where sedenotes the speaker of e. The weights are set as follows: relevance
wr= 0.35, diversity wd= 0.15, coverage wv= 0.20, authority wa= 0.05, and
salience wσ= 0.25. Relevance is prioritized as the primary retrieval objective,
while salience promotes meaningful political content over procedural language.
Coverage and diversity ensure balanced representation across parliamentary
groups and speakers. Authority models speaker expertise with respect to the
query. Its weight is the smallest to inform ranking without overriding topical
relevance, since a higher authority weight would risk marginalizing speakers from
small parliamentary groups who may lack senior institutional roles but have
made substantive contributions to the debate.
The authority is computed after retrieval. This design ensures that authority
informs but does not determine evidence selection, preserving political balance.
Authority is also used for expert selection (one top-authority speaker per group),
which influences the generation pipeline’s but not the evidence pool itself.
Authority Score.We define a query-dependent authority score for each
speaker by aggregating multiple signals capturing biographical information, insti-
tutional roles, and parliamentary activity. The score combines cosine similarity
between query embeddings and the embeddings of speaker attributes (profes-
sion, education, committees, and roles) with activity-based signals derived from
legislative acts and speech interventions, both subject to temporal decay.
Formally, the raw authority score is computed as a weighted sum of heteroge-
neous components:
authority(s, q) =X
iwici(s, q),(2)
where components include semantic similarity terms, time-decayed counts of
parliamentary activity, and role-based priors.
The assigned weights are wprofession = 0.15,weducation = 0.10,wcommittee =
0.25,wlegislative acts = 0.20,wspeech interventions = 0.25, and winstitutional role = 0.05.
These weights are currently set empirically based on expert judgment and domain
knowledge, and are not learned from data. This design choice allows explicit
control over the contribution of each component in the authority model, making
the scoring function interpretable and adaptable. In particular, the formulation

10 M. Tritella et al.
enables future extensions towards personalization or context-specific reweighting
strategies.
Authority signals are time-decayed so that recent parliamentary activity
contributes more than older evidence, applying the same decay rate to both
legislative acts and speech interventions to prioritize current relevance and avoid
overemphasizing past prominence.
Coalition affiliation is time-dependent and resolved at query time, ensuring
that only activities within a speaker’s current political group are considered,
preventing historical shifts between coalitions from biasing present-day authority
estimates.
Expert Computation.At this point, for each parliamentary group g∈G,
the system selects the speaker with the highest query-dependent authority score
among those from gpresent in the evidence pool. This speaker serves as the
group’s expert representative: their interventions and cited passages become the
primary voice through which that group’s position is conveyed in the summary.
4.3 Multi-View Generation Pipeline
As visible in Figure 3, the generation process follows a four-stage pipeline—
Analyze, Generate, Integrate, Cite—that enforces coverage, grounding, and co-
herence by design. In the first stage (Analyze), the input query is split into
a set of simple, atomic claims, each linked to the evidence required for every
parliamentary group. This ensures that all relevant aspects of the query are
explicitly addressed in the following steps.
In the second stage (Generate), the system produces one section per parlia-
mentary group from a structuredposition briefsummarizing the top evidence
chunks for that group, ordered by authority score so that the most authoritative
speakers are cited first. All groups are treated uniformly, and if no supporting
evidence is available, this is stated explicitly instead of generating unsupported
content. The third stage (Integrate) combines these sections into a single coherent
narrative, while preserving alignment with the original content and ensuring that
all quotation placeholders are retained.
In the final stage (Cite), quotations are resolved deterministically into ver-
batim quotations from the parliamentary record. The language model never
generates quotation text: it only inserts placeholders with character offsets that
are later replaced with the text from the original speeches. As a result, every quote
is directly grounded in source text, preventing hallucinations by construction and
ensuring that all claims are supported by verifiable evidence.
5 Experimental Evaluation
The experimental evaluation focuses on the downstream quality of generated
responses and assesses whether ParliamentRAG effectively supports the explo-
ration of parliamentary debates and policy positions while satisfying the core
design principles introduced in Section 1 (balanced multi-view coverage across

Authority-Aware Multi-View Parliamentary RAG 11
parliamentary groups, authority-aware evidence selection, grounded and traceable
quotations). More specifically, we address the following research question:(i) what
is the quality of the analyses produced by ParliamentRAG, a RAG architecture
explicitly designed around the principles introduced above, in light of these design
principles; and (ii) how does this quality compare with that of analyses produced
by a top-tier general-purpose document-grounded RAG system explicitly instructed
to follow the same principles?
For our comparison, we selected a top-tier general-purpose source-grounded
system, namely Google NotebookLM8, used in a controlled settings (see Sec-
tion 5.1) for different reasons: we are not aware of other competing systems
evaluated and published in peer-reviewed publications; NotebookLM represents
a realistic and accessible alternative that end users could readily employ to
analyze parliamentary material without developing a custom pipeline, after they
autonomously collect the data; it is a highly engineered system backed by the
top-tier Gemini 3.0 LLM9used by hundred millions of users.
To answer our research questions, we consider a response-level automatic
analysis with a qualitative assessment of the perceived quality by domain experts
with a blind A/B test on a benchmark of questions. Automatic evaluation metrics
focus on the satisfaction of our design principles, such as coverage balance,
quotation faithfulness, and adherence to the intended analytical principles, while
experts’ judgments focus on different aspects of perceived quality and overall
satisfaction. We describe the benchmark construction and evaluation methodology
in the following sections and discuss experimental findings afterward.
5.1 Benchmark and Protocol
Selection of benchmark topic-based questions.A first step consists in
identifying topics of interest to guide the actual test. We collected 51 policy
topics through a survey involving 17 Italian citizens interested in politics, and
reduced them to 15 by selecting topics spanning different levels of perceived
polarization and public discussion intensity, as assessed directly in the survey.
Topics span eight thematic macro-areas: Justice & Civil Rights, Institutional
Reforms, Immigration, Labour & Welfare, Environment & Energy, Technology,
Defence & Foreign Policy, Economy & Finance, with varying debate intensity
and polarization (Likert 1–5). For each topic we create a query with the following
template: “Qual è la posizione dei gruppi parlamentari su {topic}?” (Italian for
“What is the position of the parliamentary groups on {topic}?”).
NotebookLM as a baseline.ParliamentRAG operates the retrieval step on
all151kchunksextractedfromtheparliamentproceedings.Feedingallproceedings
data to NotebookLM, a general-purpose system, is unfeasible and unfair because
the system is not engineered to work with this kind and size of data. We therefore
create a notebook for each topic by engineering its context to simulate the case
where an expert selects proceedings deemed of interest for a topic; observe that
8https://notebooklm.google.com/
9https://x.com/NotebookLM/status/2002115447425282449

12 M. Tritella et al.
this selection step is challenging, as a topic can be addressed across different
proceedings, while ParliamentRAG retrieves data from the whole proceeding data.
We select ≈150relevant+ 150distractor chunks per topic-specific notebook.
Chunks are enriched with metadata to provide additional context (speaker,
parliamentary group, date, session, debate). The relevant ones are obtained
starting from the query with ParliamentRAG retrieval pipeline, including the
reranking, which considers the authority score, while the distractors are obtained
from similar topics (same macro-area but different topic). Distractors provide
a fairer simulation of retrieval-based context and the presence of non-relevant
information in uploaded documents.
Comparison of ParliamentRAG and NotebookLM.A key distinction
between the two systems lies in how they are guided towards the design princi-
ples: ParliamentRAG incorporates them directly into its architecture through
explicit mechanisms, such as authority-aware ranking and multi-view retrieval
across parliamentary groups. By contrast, NotebookLM is guided through prompt
instructions, without architectural guarantees; all prompts are available in the
code base10. The comparison therefore evaluates whether structurally enforced
constraints lead to more reliable and informative outputs than prompt-level
control alone. Another important difference concerns the backbone LLMs: Parlia-
mentRAG uses GPT-4o, whereas NotebookLM used Gemini 3 at the time of the
experiments. Gemini 3 models are more recent (November 202511) and outrank
GPT-4o, released in May 202412, in public rankings13[8]. In summary, both
systems are instructed to produce per-group structured responses via equivalent
prompts, while NotebookLM is given an engineered context and backed by a
stronger LLM. This setup provides a conservative comparison for ParliamentRAG,
although it should be interpreted primarily as an evaluation of response quality
under curated retrieval conditions rather than as a fully independent end-to-end
retrieval comparison.
Automated Evaluation Protocol.To measure whether the generated
answers satisfy the core principles of ParliamentRAG, we first calculate automatic
metrics. Coverage is evaluated through two complementary measures.Groups
with Quotation (GQ)quantifies the fraction of the ten parliamentary groups for
which at least one speaker has been quoted in the response, whileCompleteness
measures the fraction of groups receiving a dedicated narrative section in the
generated analysis. To evaluate quotation reliability, we computeQuotation
Faithfulness, defined as the fraction of quotations that exactly match substrings
of the original parliamentary interventions. Finally, to analyze the behavior of
the authority-aware retrieval mechanism, we report the mean authority score
(MA) and the corresponding standard deviation (ASD) of the cited speakers
under the query-dependent authority model.
10https://github.com/Emeierkeio/thesis-ParliamentRAG/blob/main/Prompts.md
11https://blog.google/products-and-platforms/products/gemini/gemini-3/
12https://openai.com/index/hello-gpt-4o/
13https://arena.ai/leaderboard/text

Authority-Aware Multi-View Parliamentary RAG 13
Human Evaluation Protocol.Six domain experts, disjoint from the paper
authors, were selected among parliamentary journalists, parliamentary collabora-
tors, and policy analysts, with a mean experience of 7.2 years. They independently
rated both systems on nine Likert (1–5) dimensions: Answer Quality, Answer
Clarity, Answer Completeness, Quotation Relevance, Balance Perception, Balance
Fairness, Source Relevance, Source Authority, Source Coverage. Evaluators also
provided an overall A/B preference and a 1–5 satisfaction score. The protocol
follows the PARADISE framework [ 26] and the A/B preference format of Chatbot
Arena [8]. The evaluation was blind: system identities were hidden and left/right
assignment randomized per topic. Two evaluators completed all 15 topics, while
other evaluators partially completed the evaluation, yielding a total of N= 67
paired evaluations.
We assess differences between the two systems using non-parametric paired
tests (Wilcoxon signed-rank) with Holm–Bonferroni correction for multiple com-
parisons. In addition, we calculate Cohen’s das an effect size measure to quantify
the magnitude of observed differences.
5.2 Results
Table 1.Automated Evaluation: ParliamentRAG vs. NotebookLM. Mean values ( µ)
over the 15 topics and difference ( ∆). Mean Authority (MA) for NotebookLM is
computed from the query-related authority of the MPs mentioned by NotebookLM.
MetricParliamentRAG (µ) NotebookLM (µ)∆
Groups with Quotation (GQ)0.970.95+0.02
Completeness0.991.00−0.01
Quotation Faithfulness (QF)1.000.95+0.05
Mean Authority (MA)0.53 0.52 +0.01
Table 1 reports the resulting automated metrics. ParliamentRAG achieves
near-perfectquotation-levelgroupcoverage( GQ= 0.97vs.0 .95forNotebookLM),
which is architecturally guaranteed at generation-time by stratified per-group
generation. However, coverage remains bounded by the retrieval stage: if no
retrieved chunks are associated with a given political group, that group cannot
be represented during generation, explaining why GQ̸= 1.0. NotebookLM omits
one or more groups 5% of the times. On Quotation Faithfulness the gap is
structurally significant: ParliamentRAG reaches QF= 1.00by design, whereas
NotebookLM reaches0 .95, meaning approximately 5% of its quotations do not
correspond verbatim to the source—an error rate that, in a journalistic context,
is corrosive to credibility. Mean authority of cited speakers is modestly higher
for ParliamentRAG (0 .53vs.0 .52). We remind that Mean Authority (MA) for
NotebookLM is computed from the query-related authority of the MPs mentioned
by NotebookLM and that the relevant chunks are obtained with the same retrieval
strategy as ParliamentaryRAG and reranked considering the authority socre;
hence, this could cause the only slightly difference of0.01.

14 M. Tritella et al.
Table 2.Human evaluation comparing ParliamentRAG (PRAG) and NotebookLM
(NbLM).TheleftsectionreportsmeanLikertratings(1–5)foreachevaluationdimension,
while the right section reports the percentage of pairwise preferences in favor of PRAG,
ties, and preferences in favor of NbLM.
Dimension PRAG (µ) NbLM (µ) PRAG Pref. Ties NbLM Pref.
Answer Quality 4.04 4.30 30% 25% 45%
Answer Clarity 4.27 4.51 22% 48% 30%
Answer Complet. 4.12 4.09 24% 55% 21%
Quotation Relevance 4.37 4.36 24% 54% 22%
Balance Perception 4.66 4.48 21% 72% 7%
Balance Fairness 4.67 4.66 12% 78% 10%
Source Relevance 4.07 3.84 33% 48% 19%
Source Authority 4.21 4.00 30% 57% 13%
Source Coverage 4.64 4.39 25% 70% 5%
Overall Satisfaction 4.24 4.27 31% 42% 27%
Table 2 reports the results of the human evaluation. Overall satisfaction
is nearly identical across the two systems (4 .24vs.4 .27), suggesting compara-
ble perceived utility despite substantial architectural differences. However, the
dimension-level analysis reveals complementary strengths.
NotebookLM achieves higher ratings on prose-oriented dimensions, namely
Answer Quality (4 .30vs.4 .04) and Answer Clarity (4 .51vs.4 .27), and is more
frequently preferred on Answer Quality (45% vs. 30%). This likely reflects the
stronger fluency and stylistic coherence of a monolithic generation approach
powered by a frontier commercial LLM.
By contrast, ParliamentRAG consistently outperforms NotebookLM on dimen-
sions directly related to the objectives of the proposed architecture. In particular,
evaluators report higher ratings for Source Relevance (4 .07vs.3 .84), Source Au-
thority (4 .21vs.4 .00), and Source Coverage (4 .64vs.4 .39). Pairwise preferences
reinforce this pattern: ParliamentRAG is preferred more frequently on Source
Relevance (33% vs. 19%), Source Authority (30% vs. 13%), and especially Source
Coverage (25% vs. 5%). Similarly, ParliamentRAG obtains a clear advantage on
Balance Perception (4 .66vs.4 .48), although the large proportion of ties (72%)
indicates that both systems are often perceived as balanced.
Although none of the observed differences reaches statistical significance
after Holm–Bonferroni correction for multiple comparisons, the results exhibit
a coherent qualitative pattern. Small but consistent Cohen’s deffect sizes favor
ParliamentRAG on source- and balance-related dimensions, including Source
Relevance ( d= 0.35), Source Coverage ( d= 0.35), and Source Authority ( d=
0.28), while NotebookLM shows advantages on fluency-oriented dimensions such
as Answer Quality (d=−0.31) and Answer Clarity (d=−0.29).
Finally, after completing the per-topic evaluation, respondents were asked
whether they would recommend a similar system to their colleagues, with “no” set

Authority-Aware Multi-View Parliamentary RAG 15
as the predefined answer. They answered affirmatively in 72% of cases, suggesting
that such a system was generally perceived as useful.
6 Discussion
The downstream evaluation reveals complementary strengths between the two
systems. ParliamentRAG consistently performs better on source-oriented and
balance-related dimensions, including source authority, source coverage, source
relevance, and perceived political balance, while NotebookLM achieves higher
ratings on fluency-oriented aspects such as answer quality and clarity. This asym-
metry can be explained by ParliamentRAG prioritizing structural guarantees
through explicit multi-view retrieval and authority-aware evidence selection. Bet-
ter fluency and clarity in NotebookLM can be explained either as the outcome of
a monolithic generation pipeline or as the result of using a better backbone model.
However, the observed complementarity is not symmetric. Fluency limitations
can often be mitigated through stronger language models or lightweight rewriting
stages, whereas guarantees such as verbatim quotation faithfulness and systematic
per-group coverage require architectural support and cannot be reliably enforced
through prompting alone. It is also important to recall that NotebookLM was
evaluated with a curated, pre-retrieved context, rather than over the full par-
liamentary corpus as ParliamentRAG was. Thus, the comparison does not test
NotebookLM’s ability to retrieve evidence from the complete proceedings; in
practice, this task would remain substantially more challenging for end users
without a dedicated system such as ParliamentRAG.
Among the evaluated properties, quotation faithfulness is the strongest in-
variant enforced by the architecture. Since quotations are extracted through
offset-based retrieval directly from parliamentary transcripts, fabricated or al-
tered quotations are structurally prevented. Other properties, such as balanced
group coverage, should instead be interpreted as strong design objectives rather
than absolute guarantees, as they still depend on the availability of relevant
retrieved evidence.
Limitations.The evaluation is limited by the small benchmark size (15
topics) and moderate statistical power. In addition, the study compares the
complete system against a strong external baseline without internal ablation
experiments, leaving the contribution of architectural components to future work.
7 Conclusion
We have presented ParliamentRAG, an authority-aware, multi-view RAG system
for Italian parliamentary proceedings. On a 15-topic benchmark with expert A/B
evaluation, the system matches a strong commercial baseline (NotebookLM) on
overall satisfaction while excelling precisely on the dimensions it was designed
to target (group coverage, quotation faithfulness, source authority). Directions
for future work include the personalization of ranking considering users’ prefer-
ences, domain-specific embedding models, extension to the Senate and to earlier
legislatures, and cross-national generalization to European parliamentary families.

16 M. Tritella et al.
Supplemental Material Statement:Source code is released at https://github.
com/Emeierkeio/thesis-ParliamentRAG under Apache-2.0 license. The knowl-
edge graph can be reconstructed from public data (see Section 3).
Acknowledgments.We thank the six domain experts who contributed to the A/B
evaluation and the 17 respondents to the participatory topic-collection survey.
Use of Generative AI
Generative AI tools were used during the preparation of this work to assist with
language refinement. In particular, these tools were employed to paraphrase
selected passages and improve the clarity, fluency, and overall quality of the
English text.
References
1.van Aggelen, A., Hollink, L., Kemman, M., Kleppe, M., Beunders, H.: The debates of
the european parliament as linked open data. Semant. Web8(2), 271–281 (Jan 2017).
https://doi.org/10.3233/SW-160227,https://doi.org/10.3233/SW-160227
2.Azzopardi, J.: Llms and knowledge discovery in low-resource language parliamen-
tary corpora: The pq dashboard case study. In: Proceedings of the 17th Inter-
national Joint Conference on Knowledge Discovery, Knowledge Engineering and
Knowledge Management (IC3K 2025) - Volume 1: KDIR. pp. 159–170. SciTePress
(2025). https://doi.org/10.5220/0013835100004000 ,https://www.scitepress.
org/Papers/2025/138351/138351.pdf
3.Balog, K., Fang, Y., Rijke, M., Serdyukov, P., Si, L.: Expertise retrieval. Foundations
and Trends in Information Retrieval6, 127–256 (01 2012). https://doi.org/10.
1561/1500000024
4.Boj¯ ars, U., Dar‘gis, R., Lavrinovičs, U., Paikens, P.: Linkedsaeima: A linked open
dataset of latvia’s parliamentary debates. In: Acosta, M., Cudré-Mauroux, P.,
Maleshkova, M., Pellegrini, T., Sack, H., Sure-Vetter, Y. (eds.) Semantic Sys-
tems. The Power of AI and Knowledge Graphs. pp. 50–56. Springer International
Publishing, Cham (2019)
5.Brandenberger, L., Minder, J., Salamanca, L., Schlosser, S., Gasser, L., Jung, V.,
Shariat, K., Balode, M., Schmidt-Rohr, A., Babić, L., Perez-Cruz, F., Schweitzer,
F.: Democrasci - a parliamentary knowledge graph (4 legislative periods) (0.9.0)
[dataset] (2024). https://doi.org/10.5281/zenodo.13920293 ,https://doi.org/
10.5281/zenodo.13920293
6.Camera dei deputati: Portale dei dati aperti della camera dei deputati. https:
//dati.camera.it, accessed: 2026
7.Carbonell, J., Goldstein, J.: The use of mmr, diversity-based reranking for re-
ordering documents and producing summaries. In: Proceedings of the 21st An-
nual International ACM SIGIR Conference on Research and Development in
Information Retrieval. p. 335–336. SIGIR ’98, Association for Computing Ma-
chinery, New York, NY, USA (1998). https://doi.org/10.1145/290941.291025 ,
https://doi.org/10.1145/290941.291025

Authority-Aware Multi-View Parliamentary RAG 17
8.Chiang, W.L., Zheng, L., Sheng, Y., Angelopoulos, A.N., Li, T., Li, D., Zhu, B.,
Zhang, H., Jordan, M.I., Gonzalez, J.E., Stoica, I.: Chatbot arena: an open platform
for evaluating llms by human preference. In: Proceedings of the 41st International
Conference on Machine Learning. ICML’24, JMLR.org (2024)
9.Edge,D.,Trinh,H.,Cheng,N.,Bradley,J.,Chao,A.,Mody,A.,Truitt,S.,Metropoli-
tansky, D., Ness, R.O., Larson, J.: From local to global: A graph rag approach to
query-focused summarization (2025),https://arxiv.org/abs/2404.16130
10.Erjavec, T., Kopp, M., Ljubešić, N., Kuzman, T., Rayson, P., Osenova, P., Ogrod-
niczuk, M., Çöltekin, Ç., Koržinek, D., Meden, K., Skubic, J., Rupnik, P., Agnoloni,
T., Aires, J., Barkarson, S., Bartolini, R., Bel, N., Calzada Pérez, M., Dar‘gis, R.,
Diwersy, S., Gavriilidou, M., van Heusden, R., Iruskieta, M., Kahusk, N., Kryvenko,
A., Ligeti-Nagy, N., Magariños, C., Mölder, M., Navarretta, C., Simov, K., Tungland,
L.M., Tuominen, J., Vidler, J., Vladu, A.I., Wissik, T., Yrjänäinen, V., Fišer, D.:
Parlamint ii: Advancing comparable parliamentary corpora across europe. Language
Resources and Evaluation59(3), 2071–2102 (2025). https://doi.org/10.1007/
s10579-024-09798-w,https://doi.org/10.1007/s10579-024-09798-w
11.Frasnelli, V., Palmero Aprosio, A.: There’s something new about the Italian par-
liament: The IPSA corpus. In: Calzolari, N., Kan, M.Y., Hoste, V., Lenci, A.,
Sakti, S., Xue, N. (eds.) Proceedings of the 2024 Joint International Confer-
ence on Computational Linguistics, Language Resources and Evaluation (LREC-
COLING 2024). pp. 16037–16046. ELRA and ICCL, Torino, Italia (May 2024),
https://aclanthology.org/2024.lrec-main.1394/
12.Gao, L., Dai, Z., Pasupat, P., Chen, A., Chaganty, A.T., Fan, Y., Zhao, V., Lao,
N., Lee, H., Juan, D.C., Guu, K.: RARR: Researching and revising what language
models say, using language models. In: Rogers, A., Boyd-Graber, J., Okazaki, N.
(eds.) Proceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). pp. 16477–16508. Association for Computa-
tional Linguistics, Toronto, Canada (Jul 2023). https://doi.org/10.18653/v1/
2023.acl-long.910,https://aclanthology.org/2023.acl-long.910/
13.Hwang, J., Park, J., Park, H., Kim, D., Park, S., Ok, J.: Retrieval-augmented
generation with estimation of source reliability (2024)
14.Hyvönen, E., Sinikallio, L., Leskinen, P., Drobac, S., Leal, R., La Mela, M., Tuomi-
nen, J., Poikkimäki, H., Rantala, H.: Publishing and using parliamentary linked data
on the semantic web: Parliamentsampo system for parliament of finland. Semantic
Web16(1), SW–243683 (2025)
15.Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A.,
Fung, P.: Survey of hallucination in natural language generation. ACM Computing
Surveys55(12), 1–38 (Mar 2023). https://doi.org/10.1145/3571730 ,http://
dx.doi.org/10.1145/3571730
16.Kleinberg, J.M.: Authoritative sources in a hyperlinked environment. J. ACM
46(5), 604–632 (Sep 1999). https://doi.org/10.1145/324133.324140 ,https://
doi.org/10.1145/324133.324140
17.Liu, N.F., Zhang, T., Liang, P.: Evaluating verifiability in generative search engines
(2023),https://arxiv.org/abs/2304.09848
18.Maynez, J., Narayan, S., Bohnet, B., McDonald, R.: On faithfulness and factuality
in abstractive summarization (2020),https://arxiv.org/abs/2005.00661
19.Mosbach, S., Lai, J., Rustagi, K., Tran, D.N., Kraft, M., Bindereif, E., Az-
zam, M.: Parliamentary debates in the world avatar: A hybrid retrieval-
augmented generation system (2025), https://como.ceb.cam.ac.uk/media/
preprints/c4e-preprint-338.pdf, preprint

18 M. Tritella et al.
20.Page, L., Brin, S., Motwani, R., Winograd, T.: The PageRank Citation Ranking:
BringingOrdertotheWeb.Tech.rep.,StanfordDigitalLibraryTechnologiesProject
(1998),http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.31.1768
21.Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., Collins, M., Das, D., Petrov,
S., Tomar, G.S., Turc, I., Reitter, D.: Measuring attribution in natural language
generation models. Computational Linguistics49(4), 777–840 (Dec 2023). https:
//doi.org/10.1162/coli_a_00486,https://aclanthology.org/2023.cl-4.2/
22.Rheault,L.,Cochrane,C.:Wordembeddingsfortheanalysisofideologicalplacement
in parliamentary corpora. Political Analysis28(1), 112–133 (2020). https://doi.
org/10.1017/pan.2019.26
23.Salamanca, L., Brandenberger, L., Gasser, L., Schlosser, S., Balode, M., Jung,
V., Perez-Cruz, F., Schweitzer, F.: Processing large-scale archival records: The
case of the swiss parliamentary records. Swiss Political Science Review30(2),
140–153(2024). https://doi.org/https://doi.org/10.1111/spsr.12590 ,https:
//onlinelibrary.wiley.com/doi/abs/10.1111/spsr.12590
24.Santurkar, S., Durmus, E., Ladhak, F., Lee, C., Liang, P., Hashimoto, T.: Whose
opinions do language models reflect? In: Proceedings of the 40th International
Conference on Machine Learning. ICML’23, JMLR.org (2023)
25.Slapin,J.B.,Proksch,S.O.:Ascalingmodelforestimatingtime-seriespartypositions
from texts. American Journal of Political Science52(3), 705–722 (2008)
26.Walker, M.A., Litman, D.J., Kamm, C.A., Abella, A.: Paradise: A framework
for evaluating spoken dialogue agents (1997), https://arxiv.org/abs/cmp-lg/
9704004
27.Zehlike, M., Bonchi, F., Castillo, C., Hajian, S., Megahed, M., Baeza-Yates, R.:
Fa*ir: A fair top-k ranking algorithm. In: Proceedings of the 2017 ACM on Con-
ference on Information and Knowledge Management. p. 1569–1578. CIKM ’17,
ACM (Nov 2017). https://doi.org/10.1145/3132847.3132938 ,http://dx.doi.
org/10.1145/3132847.3132938