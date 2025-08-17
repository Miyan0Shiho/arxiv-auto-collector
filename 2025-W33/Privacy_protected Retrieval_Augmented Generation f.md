# Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering

**Authors**: Yunfeng Ning, Mayi Xu, Jintao Wen, Qiankun Pi, Yuanyuan Zhu, Ming Zhong, Jiawei Jiang, Tieyun Qian

**Published**: 2025-08-12 09:38:21

**PDF URL**: [http://arxiv.org/pdf/2508.08785v1](http://arxiv.org/pdf/2508.08785v1)

## Abstract
LLMs often suffer from hallucinations and outdated or incomplete knowledge.
RAG is proposed to address these issues by integrating external knowledge like
that in KGs into LLMs. However, leveraging private KGs in RAG systems poses
significant privacy risks due to the black-box nature of LLMs and potential
insecure data transmission, especially when using third-party LLM APIs lacking
transparency and control. In this paper, we investigate the privacy-protected
RAG scenario for the first time, where entities in KGs are anonymous for LLMs,
thus preventing them from accessing entity semantics. Due to the loss of
semantics of entities, previous RAG systems cannot retrieve question-relevant
knowledge from KGs by matching questions with the meaningless identifiers of
anonymous entities. To realize an effective RAG system in this scenario, two
key challenges must be addressed: (1) How can anonymous entities be converted
into retrievable information. (2) How to retrieve question-relevant anonymous
entities. Hence, we propose a novel ARoG framework including relation-centric
abstraction and structure-oriented abstraction strategies. For challenge (1),
the first strategy abstracts entities into high-level concepts by dynamically
capturing the semantics of their adjacent relations. It supplements meaningful
semantics which can further support the retrieval process. For challenge (2),
the second strategy transforms unstructured natural language questions into
structured abstract concept paths. These paths can be more effectively aligned
with the abstracted concepts in KGs, thereby improving retrieval performance.
To guide LLMs to effectively retrieve knowledge from KGs, the two strategies
strictly protect privacy from being exposed to LLMs. Experiments on three
datasets demonstrate that ARoG achieves strong performance and
privacy-robustness.

## Full Text


<!-- PDF content starts -->

Privacy-protected Retrieval-Augmented Generation for
Knowledge Graph Question Answering
Yunfeng Ning1,*, Mayi Xu1, *, Jintao Wen1, Qiankun Pi1, Yuanyuan Zhu1, Ming Zhong1, Jiawei
Jiang1, Tieyun Qian1,†
1School of Computer Science, Wuhan University, China
{ningyunfeng, xumayi, qty }@whu.edu.com
Abstract
Large Language Models (LLMs) often suffer from halluci-
nations and outdated or incomplete knowledge. Retrieval-
Augmented Generation (RAG) is proposed to address these
issues by integrating external knowledge like that in knowl-
edge graphs (KGs) into LLMs. However, leveraging private
KGs in RAG systems poses significant privacy risks due to
the black-box nature of LLMs and potential insecure data
transmission, especially when using third-party LLM APIs
lacking transparency and control. In this paper, we investi-
gate the privacy-protected RAG scenario for the first time,
where entities in KGs are anonymous for LLMs, thus pre-
venting them from accessing entity semantics. Due to the
loss of semantics of entities, previous RAG systems cannot
retrieve question-relevant knowledge from KGs by matching
questions with the meaningless identifiers of anonymous enti-
ties. To realize an effective RAG system in this scenario, two
key challenges must be addressed: (1) How can anonymous
entities be converted into retrievable information . (2) How to
retrieve question-relevant anonymous entities .
To address these challenges, we propose a novel Abstraction
Reasoning onGraph ( ARoG ) framework including relation-
centric abstraction and structure-oriented abstraction strate-
gies. For challenge (1), the first strategy abstracts entities into
high-level concepts by dynamically capturing the semantics
of their adjacent relations. Hence, it supplements meaning-
ful semantics which can further support the retrieval pro-
cess. For challenge (2), the second strategy transforms un-
structured natural language questions into structured abstract
concept paths. These paths can be more effectively aligned
with the abstracted concepts in KGs, thereby improving re-
trieval performance. In addition to guiding LLMs to effec-
tively retrieve knowledge from KGs, these two abstraction
strategies also strictly protect privacy from being exposed to
LLMs. Experiments on three datasets demonstrate that ARoG
achieves strong performance and privacy-robustness, estab-
lishing a new practical direction for privacy-protected RAG
systems.
1 Introduction
Large Language Models (LLMs) exhibit remarkable capa-
bilities across a wide range of natural language tasks (Wang
et al. 2024). However, their practical deployment remains
*Yunfeng Ning and Mayi Xu are co-first authors.
†Corresponding author: Tieyun Qian
LeBronSavannah
ReavesLakersBronny
L.A.husband ofmother offather ofpart oflives inlives inlocated inpart ofteammate oflives inID.1ID.2
ID.6ID.4ID.3
ID.5husband ofmother offather ofpart oflives inlives inlocated inpart ofteammate oflives inLLMsAnonymous entityEntityRelation
Step IStep IIStep AFigure 1: Comparison between previous RAG (Step A) with
privacy-protected RAG (Step I-II) system for KGQA.
constrained by critical limitations, including factual inaccu-
racies (hallucinations) and outdated or incomplete knowl-
edge (Ji et al. 2023). To address these issues, the Retrieval-
Augmented Generation (RAG) technique has been proposed
to first retrieve factual knowledge from external sources and
then incorporate them into LLMs during inference, thereby
enhancing the factual grounding and reliability of LLM out-
puts (Shi et al. 2024).
KGs can store a large amount of factual knowledge.
Hence, they are widely used in the RAG system as external
knowledge sources to provide supplementary knowledge to
LLMs (Sun et al. 2024). The effectiveness of KG-enhanced
RAG systems is commonly evaluated through Knowledge
Graph Question Answering (KGQA) task. Specifically,
given a question, the RAG system aims first to retrieve rele-
vant factual triplets from the KGs and then generate answers
based on them.
In practice, many private KGs contain sensitive infor-
mation like personal data and confidential company details
(Chen et al. 2023). When using such private KGs to answer
questions requiring privacy details, current RAG systems
must retrieve relevant privacy data and expose them to the
LLMs. As shown in the Step A of Figure 1, when retrieving
triplets to answer the question “Where does Bronny live?”,
the factual triplet (Bronny, lives in, L.A. )containing sen-
sitive information will be exposed to LLMs. The inherent
black-box nature of LLMs and potential insecure data trans-
mission between users and LLM servers highlight the ur-
gent need for robust privacy protections. This need is even
greater when using third-party LLM APIs, as users may lackarXiv:2508.08785v1  [cs.CL]  12 Aug 2025

transparency or direct control over how their private data are
collected, stored, or processed.
In this paper, we make the first attempt to address privacy
concerns when the RAG system retrieves external knowl-
edge from KGs. In this privacy-protected RAG scenario , as
depicted in the Step I of Figure 1, entities within the KG
are anonymous for LLMs and replaced with corresponding
unique Machine Identifiers (MIDs), which are encrypted and
contain no semantic content. As a result, as illustrated in the
Step II of Figure 1, the LLMs are unable to access the se-
mantic information (e.g., the specific types, names, and de-
scriptions) of these entities.
Due to the loss of semantics of entities, previous RAG
systems cannot retrieve question-relevant knowledge from
KGs by matching questions with the meaningless MIDs. To
realize an effective RAG system in this scenario, two key
challenges must be addressed: (1) How can anonymous en-
tities be converted into retrievable information . (2) How to
retrieve question-relevant anonymous entities .
To address these challenges, we propose a novel
Abstraction Reasoning onGraph ( ARoG ) framework in-
cluding relation-centric abstraction and structure-oriented
abstraction strategies.
For challenge (1), the relation-centric abstraction strat-
egy treats the anonymous entities as subject or object nouns
and their adjacent relations as predicate verbs, then abstracts
the entities into corresponding higher-level concepts. These
concepts are generated by LLMs based on the adjacent re-
lations of entities. This strategy captures their in-context se-
mantics and then appending to the MIDs, thereby overcom-
ing challenge (1). For instance, an entity that serves as the
subject of relations “time zones”, “contained by”, “popula-
tion” and the object of relation “citytown” can be abstracted
into “geographic location”.
For challenge (2), the structure-oriented abstraction strat-
egy transforms unstructured natural language questions into
structured abstract concept paths. These paths can be more
effectively aligned with the abstracted concepts in KGs than
naive questions, thereby improving retrieval performance.
For instance, the abstract concept path to the question “What
is the name of the daughter of the artist who had the The Mrs.
Carter Show World Tour?” is “Nicki Minaj (artist) →had
→The Mrs. Carter Show World Tour; Nicki Minaj (artist)
→has daughter named →Chiara Fattorini (person)”. No-
tably, the semantic similarity between this path and relevant
triplets persists even with inaccurate entities, as the path cap-
tures the relational structure of abstracted concepts.
To fulfill the complete privacy-protected RAG, our ARoG
framework also incorporates Abstraction-driven Retrieval
module and Generator module. These two modules ensure
robust performance in retrieving question-relevant triplets
from KGs and generating answers. To evaluate the effective-
ness of our ARoG framework, we conduct experiments in
theprivacy-protected RAG scenario on three popular yet di-
verse datasets including WebQSP, CWQ, and GrailQA. Our
ARoG framework achieves the state-of-the-art (SoTA) per-
formance on all three datasets.
Our contributions can be summarized as follows.
• We explore the privacy-protected RAG scenario for thefirst time, which aims to address privacy concerns when
the RAG system retrieves external knowledge from KGs.
• We propose a novel RAG-based framework ARoG, uti-
lizing two abstraction strategies to tackle challenges
in the privacy-protected RAG scenario while balancing
high performance and data privacy.
• We conduct extensive experiments across three public
datasets, providing empirical evidence of the effective-
ness and robustness of our proposed ARoG.
2 Related Work
In this section, we review related works for the KGQA task.
2.1 Semantic Parsing for KGQA
Semantic parsing (SP)-based methods treat LLMs as seman-
tic parsers that translate questions into formal queries us-
ing labeled exemplars. To address errors in the initial formal
queries, Li et al. (2023) utilize entity and relation binders
grounded in KGs, Nie et al. (2024) reformulate the genera-
tion process into code generation process; and Zhang et al.
(2025) adapt generation process to function calling using
Condition Graphs (CGs), a variant of KGs. However,these
methods do not fundamentally resolve the challenges posed
by formatting errors, and the performance is heavily limited
by the quality and quantity of labeled exemplars.
2.2 Retrieval-Agumented Generation for KGQA
Compared with SP-based methods, RAG-based methods
usually have superior performance. RAG-based methods
employ LLMs as retrievers to fetch question-relevant triplets
as evidence, and employ LLMs as generators to infer an-
swers with evidence. To improve the quality of evidence,
Jiang et al. (2023a) propose an iterative LLM-based “Read-
then-Reason” framework; Sun et al. (2024) utilize LLMs
to perform beam search on KGs; and Xu et al. (2024) uti-
lize LLMs to supplement additional knowledge. However,
previous RAG-based KGQA methods directly feed factual
triplets within KGs to LLMs, which may undermine data
privacy. The inherent black-box nature of LLMs and po-
tential insecure data transmission between users and LLM
servers highlight the urgent need for robust privacy protec-
tions. This need is even greater when using third-party LLM
APIs, as users may lack transparency or direct control over
how their privacy data are collected, stored, or processed.
In this paper, we make the first attempt to explore the
privacy-protected RAG scenario for KGQA, which aims to
address privacy concerns when the RAG system retrieves ex-
ternal knowledge from KGs.
3 Methodology
3.1 Preliminary
Knowledge Graphs (KGs) Knowledge Graphs (KGs)
contain abundant factual triplets ( eh, r, et), that is, KG =
{(eh, r, et)|eh, et∈ E, r∈ R} , where eh,r, and etrep-
resent the head entity, the relation, and the tail entity, re-
spectively. EandRdenote the set of entities and relations,
respectively.

(b) Structure-oriented Abstraction Thought: First, I need to identify the artist associated with "The Mrs. Carter Show World Tour." This tour is by Nicki Minaj. Second,.... The answer is Chiara Fattorini. Abstract Concept Path: (1) Nicki Minaj (artist) → had → The Mrs. Carter Show World Tour; (2) Nicki Minaj (artist) → has daughter named → Chiara Fattorini (person)
ID.2ID.3ID: m.ndgdx ID.1(a) Relation-centric Abstraction
ArtistLast_referenced_by Nominated_for Relevant Triplets: m.0qftnyr → artist → m.01mpq7s (musical artist) Candidate Triplets:(1) m.0qftnyr → artist → m.01mpq7s (musical artist); (2) m.0qftnyr → start_date → m.ndgdx (specific value); (3) m.0qftnyr → nominated_for→m.024bwn (award) (d) GeneratorAnswer: {No}. Additional information about the artist's family is required.Evidence:m.0qftnyr → artist → m.01mpq7s (musical artist) RelevantIrrelevant(c) Abstraction-driven RetrievalStart_dateAppendedAdjacent RelationsFilterNoun Concept: musical artist Related Relations:ArtistConcert_tour; Concerts; LLMs
ID:m.01mpq7sID: m.0dnb ID: m.0fzq ID: m.0dwvID: m.0hvsrmc Children Singers Concert_tours Parents AppendedAdjacent RelationsFilterNoun Concept:PersonRelated Relations:Children;Recordin-gsLLMsLLMs
Relevant Triplets: m. 01mpq7s (person) → children → m.0hvsrmc (person) Candidate Triplets:(1) m.01mpq7s (musical artist) → children → m.0hvsrmc (person); (2) m.01mpq7s (musical artist) → parents → m.0fzq (person) ; (3) m.01mpq7s  (person) → concert_tours → m.05200my (music concert tour)Output{m.0hvsrmc (person)}. Evidence:m.0qftnyr → artist → m.01mpq7s (musical artist); m. 01mpq7s (person) → children → m.0hvsrmc (person)LLMsInputQuestion: What is the name of the daughter of the artist who had the "The Mrs. Carter Show World Tour" ? Topic Entities: {"ID": "m.0qftnyr", "name": "The Mrs. Carter Show World Tour"} Relation RetrievalID: m.01mpq7sRecordings Appeared_on Concerts Concert_tours Relation FilteringEntity Abstraction
Relation RetrievalEntity AbstractionParents  Compositions Track RecordingsRelation FilteringBlue IvyFigure 2: An overview of ARoG framework, which includes four primarily modules: Relation-centric Abstraction, Structure-
oriented Abstraction, Abstraction-driven Retrieval and Generator modules. The Relation-centric Abstraction module consists
of three steps: relation retrieval, relation filtering and entity abstraction.
Knowledge Graph Question Answering (KGQA) Given
an nature language question q, the topic entities Eq=
e1
q, e2
q,· · ·, en
q	
(the main entities asked in the ques-
tion (Qiu et al. 2018)), and a knowledge graph KG, the
task of KGQA is to predict the answer entities Aq=
e1
a, e2
a,· · ·, en
a	
, which can answer the question. Follow-
ing previous works (Sun et al. 2024), we assume that the
topic entities and answer entities are labeled and linked to
the corresponding entities in KG, i.e.,Eq, Aq⊆ E.
Privacy-protected RAG scenario for KGQA To prevent
the LLM from accessing sensitive information within the
KG, we anonymize all entities by representing them as
MIDs. Specifically, we leverage the Freebase database sys-
tem (Bollacker et al. 2008) to map each entity to a unique,
encrypted MID. These MIDs are meaningless to the LLM.
Crucially, to maintain task feasibility, the names of the topic
entities explicitly mentioned in the question are available.
3.2 Overview
In order to solve the challenges in the privacy-protected
RAG scenario, we propose a novel Abstraction Reasoning
on Graph (ARoG) framework. The architecture of ARoG,
illustrated in Figure 2, follows a Retrieval-then-Generation
pipeline. It comprises four primary modules: the first three
constitute the retrieval phase, while the latter one belongs to
the generation phase, as detailed below.
(1) Relation-centric Abstraction (RA): To overcome the
challenge of How can anonymous entities be converted into
retrievable information , we develop the Relation-centric
Abstraction module. It utilizes LLMs to transform anony-mous entities into abstract concepts and then appends to the
anonymous identifies to supplement semantic.
(2) Structure-oriented Abstraction (SA): To overcome
the challenge of How to retrieve question-relevant anony-
mous entities , we develop the Structure-oriented Abstraction
module. It uses LLMs to convert unstructured questions into
structured abstract concept paths to guide the retrieval.
(3) Abstraction-driven Retrieval (AR): To retrieve the
question-relevant triplets from a large-scale KG, this mod-
ule sequentially performs searching and pruning operations
over the KG, driven by the two aforementioned abstraction
modules.
(4) Generator: This module employs an LLM to infer
answers based on the question and the question-relevant
triplets derived from Abstraction-driven Retrieval.
3.3 Relation-centric Abstraction Module
The Relation-centric Abstraction module treats relations ad-
jacent to entities as verbs and the entities themselves as sub-
ject or object nouns. Leveraging the generative capabilities
of LLMs, we infer abstract concepts for the entities from
these verbs, thereby enriching their semantics.
As illustrated in Part (a) of Figure 2, the Relation-centric
Abstraction module operates on a question q,ntopic entities
Eqand the KG, and comprises three steps as follows.
(1) Relation Retrieval First, starting from a set of ntopic
entities, we explore the KG to extract nsets of adjacent re-
lations. For each relation cluster Rt, we utilize an LLM to
identify the Wmost question-relevant relations, which are
denoted as Rt,opt. The process is expressed as follows.

Rt,opt= arg max
Rt,opt⊆Rt
|Rt,opt|≤WLLM (Rt, q, Inst rr, Err), (1)
where Wis a hyperparameter representing the width of the
retrieval, Inst rrdescribes the details of the relation retrieval
task,Errdenotes an exemplar, and LLM denotes LLM call.
After retrieval, we gather n×Wentity-relation pairs. Sub-
sequently, for each pair, we identify its adjacent entity clus-
ter, resulting in a total of n×Wcandidate entity clusters.
(2) Relation Filtering Second, while the entities in the
candidate clusters are meaningless, their adjacent relations
provide rich in-context semantic information. For a repre-
sentative entity ein the candidate entity clusters, we treat its
adjacent relations as verbs, denoted as Rv. To filter out irrel-
evant relations, we employ SentenceTransformer (Reimers
and Gurevych 2020) to select the top- Kmost relevant rela-
tions, denoted as Rv,opt, based on their cosine similarity to
the question q. The process is expressed as follows.
Rv,opt = arg max
Rv,opt⊆Rv
|Rv,opt|≤KX
r∈Rv,optcos(Emb(q),Emb(r)),(2)
where Emb(.) represents the embedding results calculated
by SentenceTransformer, and Kis set to 5 empirically.
(3) Entity Abstraction To abstract the representative
anonymous entity e, we first prompt an LLM to infer its cor-
responding concept based on the related relations Rv,opt. We
then append this abstract concept to the entity’s MID, yield-
ing the abstracted entity eabs. The procedure is expressed as
follows.
eabs=LLM (Rv,opt, e, Inst ea, Eea), (3)
where Inst eadescribes the details of the entity abstraction,
andEeadenotes a single exemplar for guidance.
Since the relations within the KG define the schema struc-
tures rather than sensitive information, sharing these rela-
tions with LLMs poses minimal privacy risk. Additionally,
to minimize LLM calls, we assume that entities within the
same entity cluster share a common concept, in line with
Jiang et al. (2023b). If an entity is associated with multiple
generated abstract concepts, these concepts are combined
into a comma-separated sequence.
This abstraction process is applied to each triple t=
(eh, r, et)in the KG, replacing its entities with their ab-
stracted counterparts to form an abstracted triple tabs=
(eabs
h, r, eabs
t). We then apply this transformation uniformly
to the entire set of triplets Tq={t1, t2,· · ·} , yielding its
abstracted counterpart, Tabs
q={tabs
1, tabs
2,· · ·}.
3.4 Structure-oriented Abstraction Module
The Structure-oriented Abstraction module converts an un-
structured question into a structured abstract concept path
Pq. During this process, unknown entities are also abstracted
into their corresponding concepts. Formally, Pqcomprises a
set of generated triplets, denoted as Pq={tp1, tp2,· · ·}. By
guiding the retrieval of relevant triplets and entities via se-
mantic matching, this path effectively isolates the LLM from
the raw entities within the KG.
A key advantage of this strategy is that the effectiveness
of abstract concept paths is not reliant on correct entities.For example, as shown in Part (b) of Figure 2, although the
generated entities “Nicki Minaj” and “Chiara Fattorini” are
incorrect, the two resulting triplets, which incorporate the
concepts “artist” and “person”, semantically align well with
relevant triplets incorporating abstracted entities in the KG.
Specifically, given a question q, we utilize an LLM to gen-
erate a structured abstract concept path Pq. The overall pro-
cess is expressed as follows.
Rat, P q=LLM (q, Inst sa, Esa), (4)
where Inst sadescribes the abstraction task in a chain-of-
thought (CoT) manner (Wei et al. 2023), Esadenotes multi-
ple exemplars. Rat denotes the rationales that represent the
thought process and underpin the generation of Pq, where
both are concurrently produced from a single process.
3.5 Abstraction-driven Retrieval Module
The Abstraction-driven Retrieval module retrieves question-
relevant triplets as the evidence to support the answer gener-
ation. It helps reduce noise data while decreasing the context
length input into the LLM.
Given a question q,ninitial topic entities Eqand the
knowledge graph KG, the retrieval phase over the KG is
organized into multiple iterative cycles (retrieval iterations).
We define two retrieval parameters: width Wand depth D.
Width Wdenotes the maximum number of topic entities and
retrieved relations at each iteration, and depth Drepresents
the maximum number of iterations. Each iteration explores
the 1-hop neighbor subgraphs of the topic entities and up-
dates them with the discovered neighbors.
After retrieving n×Wrelevant entity-relation pairs
through relation retrieval, previous RAG-based methods use
LLMs to retrieve additional entities but perform poorly in
the privacy-protected scenario. Instead, we first construct
candidate triplets in the Relation-centric Abstraction mod-
ule. Then, we retrieve question-relevant triplets and identify
newly discovered entities within these triplets based on their
semantic similarity to the triplets in the aforementioned ab-
stract concept paths.
Specifically, as shown in Part (c) of Figure 2, after the
Relation-centric Abstraction module, we gather a set of can-
didate triplets Tabs
q. Then, we retrieve Wmost relevant
triplets Tq,optfrom this set, based on their cosine similar-
ity with the triplets tpin the abstract concept path Pq. The
procedure is outlined as follows.
Tabs
q,opt= arg max
Tabs
q,opt⊆Tabs
q
|Tabs
q,opt|≤WX
tabs∈Tabs
q,opt
tp∈Pqcos(Emb(tabs),Emb(tp)).(5)
In each retrieval iteration, we retrieve question-relevant
triplets Tabs
q,opt and aggregate all triplets to date into a cumula-
tive set Tabs
q,all. This set is then fed into the Generator module.
Additionally, the topic entities are replaced with the newly
identified entities from Tabs
q,opt for the next retrieval iteration.
3.6 Generator Module
The Generator module infers the answers by feeding the
question and retrieved relevant triplets into the LLM-based
generator, which is similar to that of Sun et al. (2024).

Specifically, as shown in Part (d) of Figure 2, after the
Abstraction-driven Retrieval module, we gather a set of
question-relevant triplets Tabs
q,all as evidence. Next, we input
the evidence, along with the original question q, the instruc-
tionInst gand several exemplars Eginto an LLM-based
generator. The generator then produces a flag Flag and the
final answers Ans. The process is illustrated as follows.
Flag, Ans =LLM (Tabs
q,all, q, Inst g, Eg). (6)
We output the answers directly if Flag is positive; oth-
erwise, we proceed to the next retrieval iteration. These
answers may contain entities anonymized as MIDs. These
MIDs are replaced with their real names on the user side,
which saves a privacy map linking MIDs to their names.
4 Experiment
4.1 Experimental Setup
Datasets We conduct experiments on three KGQA datasets
including WebQSP, CWQ and GrailQA. The details of the
datasets’ splits and the number of selected samples are pro-
vided in Appendix A.1. WebQSP (Yih et al. 2016) contains
questions from WebQuestions (Berant et al. 2013) that are
answerable by Freebase (Bollacker et al. 2008). CWQ (Tal-
mor and Berant 2018) extends WebQSP, where most ques-
tions require at least 2-hop reasoning. GrailQA (Gu et al.
2021) is a diverse KGQA dataset, where most questions ne-
cessitate long-tail knowledge.
Experimental Settings We define two experimental set-
tings. The first is the #Total setting , where we conduct the
test on our complete sample set. The second is the #Filtered
setting , which simulates a strict privacy-protected scenario.
In this scenario, answering questions requires access to pri-
vate knowledge not included in the LLM’s pre-trained data.
To model this scenario, we remove all samples that can be
answered correctly via CoT prompting and conduct the test
on the remaining filtered subset.
Metrics We use exact match accuracy (Hits@1) as evalu-
ation metric, consistent with previous studies (Li et al. 2023;
Sun et al. 2024; Chen et al. 2024).
Implementation Details We employ GPT-4O-MINI -
2024-07-18 as the underlying LLM. For the Structure-
oriented Abstraction module and the Generator model, the
temperature parameter is set to 0 to ensure deterministic out-
puts. For the Relation-centric Abstraction module, the tem-
perature is set to 0.4 to introduce variability. Both the fre-
quency penalty and presence penalty are set to 0. The width
Wand depth Dof the retrieval are set to 3, striking a balance
between performance and efficiency. We repeat the experi-
ment 3 times and report the average scores. More implemen-
tation details are illustrated in the Appendix A.3.
4.2 Baselines
Methods for applying LLMs to the KGQA task are generally
divided into three categories by existing studies: Pure-LLM,
SP-based and RAG-based methods. To ensure a comprehen-
sive evaluation, we select SoTA methods from each category
as our baselines, which are listed as follows.Type MethodWebQSP CWQ GrailQA
#Tot #Fil #Tot #Fil #Tot #Fil
PureIO 68.6 13.8 50.9 10.9 31.5 5.8
CoT 67.2 0.0 55.1 0.0 35.5 0.0
CoT-SC 68.3 8.9 54.2 2.0 35.0 3.3
SPKB-BINDER 15.8 16.0 - - 45.0 45.3
KB-BINDER-R 37.0 11.7 - - 62.3 62.8
TrustUQA 47.4 48.4 - - - -
TrustUQA-R 60.9 53.1 - - - -
RAGToG 64.9 8.2 54.1 4.9 38.9 17.0
GoG 62.3 26.9 48.1 16.5 29.1 12.4
PoG 61.4 12.6 49.8 16.3 49.7 36.1
ARoG 74.7 58.9 60.0 36.3 78.7 71.8
Imp. ↑6.1↑5.8↑4.9↑19.8↑16.4↑9.0
Table 1: Comparison results (%) of Pure-LLM methods
(Pure), SP-based methods (SP), and RAG-based methods
(RAG), under the #Total setting (#Tot) and #Filtered setting
(#Fil). The “-” indicates that the SP-based method cannot be
run due to lacking special formal query annotations. “Imp.”:
the absolute improvement of ARoG relative to the second-
best performer. Bold : the best score. Underline : the second
best score.
Pure-LLM methods We select IO (Brown et al. 2020),
CoT (Wei et al. 2023) and CoT-SC (Wang et al. 2023) as
reference. Because they operate without external KGs and
do not tailor for KGQA, they serve as baselines for evaluat-
ing the intrinsic reasoning capabilities of LLMs.
SP-based methods We select KB-BINDER (Li et al.
2023) and TrustUQA (Zhang et al. 2025) as representative
SP-based methods. For their variants with dynamically ex-
emplars, we claim them as KB-BINDER-R and TrustUQA-
R, respectively.
RAG-based methods We select ToG (Sun et al. 2024),
PoG (Chen et al. 2024) and GoG (Xu et al. 2024) as repre-
sentative RAG-based methods. ToG (Sun et al. 2024) uses
LLMs to retrieve knowledge triplets via a beam search ap-
proach. PoG (Chen et al. 2024) enhances the retrieval pro-
cess through reflection, memory, and adaptive breadth. In a
different vein, GoG (Xu et al. 2024) leverages LLMs as a
flexible knowledge source, complementing the KG.
4.3 Performance Comparison
We compare ARoG with the baselines to demonstrate the
effectiveness for question answering, the results are repre-
sented in Table 1. Overall, ARoG achieves the best perfor-
mance across all three datasets under both settings.
First, ARoG significantly outperforms all RAG-based
baselines. Existing RAG methods struggle with KGs con-
taining anonymous entities, as they rely heavily on specific
entity information for effective retrieval and reasoning.
Second, ARoG also surpasses existing SP-based base-
lines. This is because ARoG leverages LLMs to directly par-
ticipate in the reasoning process through iterative retrieval-

and-generation mechanisms. In contrast, SP-based baselines
lack such iterative paradigms and are constrained by the cov-
erage and completeness of the KG.
Third, compared to Pure-LLM baselines, most RAG-
based baselines show no advantage, primarily due to their
inability to effectively retrieve entity information in the
privacy-protected scenario. In contrast, ARoG demonstrates
significant improvements by effectively retrieve question-
relevant triplets from the KG.
Finally, from the #Total setting to the #Filtered setting,
the internal knowledge of LLMs is not enough to answer
questions. Therefore, the performance of most RAG-based
methods drops significantly, since they fail to retrieve private
knowledge from the KG. However, ARoG compensates for
this limitation with two abstraction strategies, which helps
the LLMs perform reasoning over KGs without concrete en-
tity information. For SP-based methods, since they do not
require concrete entity information from the KG, their per-
formance remains relatively stable as well.
We also conduct experiments on Q WEN 3-32B-FP8. The
experiments yield findings that are consistent with the afore-
mentioned conclusions. The details are shown in Appendix
B.
4.4 Ablation Study
To assess the effectiveness of the two abstraction strategies
in ARoG, we perform ablation studies on three datasets.
Specifically, ablation studies are conducted separately on the
three steps of the Relation-centric Abstraction module and
on the Structure-oriented Abstraction module. Notably, re-
lation retrieval (Step 1) aims to retrieve relations and serves
as the foundation for all subsequent steps; relation filter-
ing (Step 2) is a prerequisite for entity abstraction (Step 3).
Therefore, in the ablation studies, we do not consider the
scenarios of removing Step 1 or removing Step 3 while re-
taining Step 2. The results are shown in Table 2.
ARoG KGQA
RA SA WebQSP CWQ GrailQA
Step1 Step2 Step3 SA#Tot #Fil #Tot #Fil #Tot #Fil
✓ × × ×63.9 53.6 36.4 20.7 72.6 66.9
✓ × × ✓71.6 52.7 54.5 30.1 75.2 68.0
✓ ×✓×68.3 57.4 47.9 26.9 76.9 70.7
✓ ×✓✓72.9 58.9 59.5 35.4 78.3 71.6
✓ ✓ ✓ ×68.1 56.8 50.3 30.5 76.3 70.7
✓ ✓ ✓ ✓74.7 58.9 60.0 36.3 78.7 71.8
Table 2: Results (%) of the ablation study. RA: Relation-
centric Abstraction module. SA: Structure-oriented Abstrac-
tion module. #Tot: the #Total setting. #Fil: #Filtered setting.
Bold : the best score.
Analysis on Relation-centric Abstraction Module
When omitting the Relation-centric Abstraction module, we
observe a performance decline of at least 2.5% under the
#Total setting and at least 3.8% under the #Filtered setting.The decline is most significant under the #Filtered setting,
where answering questions requires acquiring private
knowledge not found in the LLM’s pre-trained knowledge.
This suggests that the Relation-centric Abstraction module
is highly effective at extracting and abstracting in-context
semantic information from relations within KG, thereby
preserving the private knowledge.
Analysis on Structure-oriented Abstraction Module
When omitting the Structure-oriented Abstraction module,
we observe a performance decline of at least 2.4% under the
#Total setting and at least 1.1% under the #Filtered setting.
The decline is most significant on the CWQ dataset, whose
questions primarily require multi-hop reasoning. This sug-
gests that the Structure-oriented Abstraction module excels
at analyzing the structural semantics of questions, while the
abstract concept paths enhance multi-iteration retrieval.
4.5 Efficiency Study
We study the efficiency of RAG-based methods. Table 3 re-
ports the average LLM calls and token consumption by these
methods to answer a question across three datasets under the
#Total setting.
Dataset Method #Call #Input #Output #Total
WebQSPToG 15.4 6,356.4 1,357.0 7,713.4
GoG 10.6 12,031.0 343.3 12,374.3
PoG 10.1 5,872.5 336.0 6,208.6
ARoG 17.3 7,044.7 707.3 7,752.1
CWQToG 21.7 8,372.3 1,852.2 10,224.5
GoG 13.5 14,657.0 463.8 15,120.8
PoG 16.9 8,730.6 423.2 9,153.7
ARoG 25.4 1,0187.0 1,055.0 11,242.1
GrailQAToG 13.5 5,220.1 1,306.5 6,526.5
GoG 11.4 12,665.7 428.0 13,093.7
PoG 15.8 7,445.0 436.4 7,881.4
ARoG 12.5 5,044.7 560.7 5,605.5
Table 3: Efficiency comparison between RAG-based meth-
ods under the #Total setting. #Call denotes the number of
LLM calls, while #Input, #Output, #Total represent the num-
ber of input tokens, output tokens and total tokens, respec-
tively.
On the WebQSP and CWQ datasets, we observe that
ARoG is slightly less efficient than ToG and PoG. How-
ever, this disadvantage can be disregarded to some degree,
especially considering ARoG’s outstanding performance. In
contrast, the situation is different on the GrailQA dataset,
whose most questions rely on external knowledge to an-
swer. The reason is that ARoG can effectively utilize ex-
ternal knowledge while other RAG-based methods cannot,
leading to higher consumption during the retrieval phase.
Additionally, we observe that GoG needs fewer LLM calls
than most of others, but its input tokens consumption is too
high. This is primarily due to its complex prompt design,
which aims to decide whether to search knowledge from
KGs or use LLMs to generate knowledge.

4.6 Quantitative Study
To understand the impact of retrieval depth Dand width W
on ARoG’s performance, we conduct the quantitative study.
We vary these parameters from 1 to 4 (with a default of 3)
across three datasets, following the approach in ToG (Sun
et al. 2024). The results are presented in Figure 3.
1 2 3 4
width506070Hits@1
67.070.374.7 75.0
45.857.058.960.6WebQSP
1 2 3 4
width204060
51.655.360.0 60.3
22.033.036.3 37.2CWQ
1 2 3 4
width657075
72.177.178.7 78.6
63.370.471.870.4GrailQA
1 2 3 4
depth6070Hits@1
72.874.4 74.773.1
58.560.7
58.960.4WebQSP
1 2 3 4
depth204060
39.759.7 60.0 60.2
21.837.0 36.3 35.2CWQ
1 2 3 4
depth657075
69.778.2 78.7
77.1
64.572.1 71.870.3GrailQA#Total #Filtered
Figure 3: Performance (%) of ARoG with different retrieval
widths and depths.
We observe that, on the WebQSP dataset, the width Wis
particularly critical, while the depth Dhas little to no im-
pact, as most answer entities can be found within the 1-hop
subgraph of initial topic entities. Conversely, for the CWQ
and GrailQA datasets, both width and depth enhance perfor-
mance; however, the benefits diminish beyond a depth of 2,
since most answers are located within the 2-hop subgraph.
4.7 Deep Analysis
Comparison between Different Abstraction Strategies
To thoroughly validate the effectiveness of abstraction strat-
egy in the Structure-oriented Abstraction module, we com-
pare ARoG against two alternative abstraction strategies
as follows. (1) Chain-of-Thought (CoT) (Wei et al. 2023),
which replaces the abstract concept path with generated ra-
tionales, and (2) question decomposition (Dec) (Chen et al.
2024), which uses derived sub-questions. For a more com-
prehensive baseline, we also include the variant that omits
the Structure-oriented Abstraction module (“w/o SA”). The
performance of all these methods is illustrated in Figure 4.
/uni00000006/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f /uni00000006/uni00000029/uni0000004c/uni0000004f/uni00000057/uni00000048/uni00000055/uni00000048/uni00000047
/uni0000000b/uni00000044/uni0000000c/uni00000003/uni0000003a/uni00000048/uni00000045/uni00000034/uni00000036/uni00000033/uni00000018/uni00000013/uni00000018/uni00000018/uni00000019/uni00000013/uni00000019/uni00000018/uni0000001a/uni00000013/uni0000001a/uni00000018/uni0000002b/uni0000004c/uni00000057/uni00000056/uni00000023/uni00000014
/uni00000006/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f /uni00000006/uni00000029/uni0000004c/uni0000004f/uni00000057/uni00000048/uni00000055/uni00000048/uni00000047
/uni0000000b/uni00000045/uni0000000c/uni00000003/uni00000026/uni0000003a/uni00000034/uni00000015/uni00000018/uni00000016/uni00000013/uni00000016/uni00000018/uni00000017/uni00000013/uni00000017/uni00000018/uni00000018/uni00000013/uni00000018/uni00000018/uni00000019/uni00000013
/uni00000006/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f /uni00000006/uni00000029/uni0000004c/uni0000004f/uni00000057/uni00000048/uni00000055/uni00000048/uni00000047
/uni0000000b/uni00000046/uni0000000c/uni00000003/uni0000002a/uni00000055/uni00000044/uni0000004c/uni0000004f/uni00000034/uni00000024/uni0000001a/uni00000013/uni00000011/uni00000013/uni0000001a/uni00000015/uni00000011/uni00000018/uni0000001a/uni00000018/uni00000011/uni00000013/uni0000001a/uni0000001a/uni00000011/uni00000018/uni00000024/uni00000035/uni00000052/uni0000002a /uni0000005a/uni0000004c/uni00000057/uni0000004b/uni00000003/uni00000026/uni00000052/uni00000037 /uni0000005a/uni0000004c/uni00000057/uni0000004b/uni00000003/uni00000027/uni00000048/uni00000046 /uni0000005a/uni00000012/uni00000052/uni00000003/uni00000036/uni00000024
Figure 4: Performance (%) with different abstraction strate-
gies for the Structure-oriented Abstraction module.
We observe that ARoG demonstrates superior perfor-
mance across all conditions. The CoT succeeds by provid-ing necessary entities, while the Dec succeeds by leverag-
ing structured sub-questions. However, both strategies fail
to achieve optimal results since they do not incorporate ab-
stract concepts, which are crucial for the retrieval.
Performance in Different Privacy-Protected Scenarios
To further showcase ARoG’s robustness, we establish four
different privacy scenarios: full privacy-protected scenar-
ios (Private), exposing grounded entities information in re-
trieval phase (P-R), in generation phase (P-G), or both
phases (P-RAG). We select ToG to represent previous RAG-
based methods. We develop ARoG-R that uses the LLMs
to retrieve question-relevant entities, thereby replacing the
Abstraction-driven Retrieval module in ARoG. The experi-
mental results are illustrated in Figure 5.
P-RAG P-G P-R Private20406080Hits@1
WebQSP
P-RAG P-G P-R Private204060
CWQ
P-RAG P-G P-R Private20406080
GrailQAARoG (Total)
ARoG (Filtered)ARoG-R (Total)
ARoG-R (Filtered)ToG (Total)
ToG (Filtered)
Figure 5: Performance (%) of ARoG, ARoG-R and ToG in
different privacy-protected scenarios.
We observe that, when entities are anonymous in the
generation phase, ToG’s performance drops significantly
while ARoG and ARoG-R remain. The reason is that the
anonymous entities retrieved by ToG are meaningless, while
ARoG and ARoG-R address this issue based on the relation-
centric abstraction. Additionally, when entities are anony-
mous in the retrieval phase, ARoG-R drops significantly
while ARoG remains high. Although ToG also maintains
its performance, it shows a relatively poor result. The rea-
son is that ToG and ARoG-R retrieve the anonymous enti-
ties by feeding the question along with anonymous entities
to LLMs. This approach results in suboptimal performance.
In contrast, ARoG retrieve the question-relevant anonymous
entities based on the Abstraction-driven Retrieval module.
Overall, ARoG exhibits impressive robustness and high
performance across all three datasets and scenarios.
Case Study The detailed case study is shown in the Ap-
pendix C.
5 Conclusion
In this paper, we investigate the privacy-protected RAG sce-
nario in KGQA for the first time, where the entities within
KGs are anonymous to LLMs. To address the challenges
in this scenario, we propose a novel ARoG framework that
incorporates relation-centric and structure-oriented abstrac-
tion strategies. Based on these two strategies, ARoG is able
to effectively retrieve question-relevant knowledge triplets
while preventing data privacy. Extensive experiments con-
ducted on three datasets demonstrate the effectiveness and
robustness of ARoG, highlighting its potential to address the
challenging privacy-protected RAG scenario for KGQA.

References
Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013. Se-
mantic Parsing on Freebase from Question-Answer Pairs. In
Yarowsky, D.; Baldwin, T.; Korhonen, A.; Livescu, K.; and
Bethard, S., eds., Proceedings of the 2013 Conference on
Empirical Methods in Natural Language Processing , 1533–
1544. Seattle, Washington, USA: Association for Computa-
tional Linguistics.
Bollacker, K.; Evans, C.; Paritosh, P.; Sturge, T.; and Taylor,
J. 2008. Freebase: a collaboratively created graph database
for structuring human knowledge. In Proceedings of the
2008 ACM SIGMOD International Conference on Man-
agement of Data , SIGMOD ’08, 1247–1250. New York,
NY , USA: Association for Computing Machinery. ISBN
9781605581026.
Brown, T.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J. D.;
Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell,
A.; Agarwal, S.; Herbert-V oss, A.; Krueger, G.; Henighan,
T.; Child, R.; Ramesh, A.; Ziegler, D.; Wu, J.; Winter,
C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.;
Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford,
A.; Sutskever, I.; and Amodei, D. 2020. Language Mod-
els are Few-Shot Learners. In Larochelle, H.; Ranzato, M.;
Hadsell, R.; Balcan, M.; and Lin, H., eds., Advances in Neu-
ral Information Processing Systems , volume 33, 1877–1901.
Curran Associates, Inc.
Chen, L.; Tong, P.; Jin, Z.; Sun, Y .; Ye, J.; and Xiong, H.
2024. Plan-on-Graph: Self-Correcting Adaptive Planning of
Large Language Model on Knowledge Graphs. In Glober-
son, A.; Mackey, L.; Belgrave, D.; Fan, A.; Paquet, U.; Tom-
czak, J.; and Zhang, C., eds., Advances in Neural Informa-
tion Processing Systems , volume 37, 37665–37691. Curran
Associates, Inc.
Chen, X.; Bao, J.; Zhou, P.; and Liao, Y . 2023. Hierar-
chical Privacy-Preserved Knowledge Graph. In 2023 IEEE
43rd International Conference on Distributed Computing
Systems (ICDCS) , 1–2.
Gu, Y .; Kase, S.; Vanni, M.; Sadler, B.; Liang, P.; Yan, X.;
and Su, Y . 2021. Beyond I.I.D.: Three Levels of Gen-
eralization for Question Answering on Knowledge Bases.
InProceedings of the Web Conference 2021 , WWW ’21,
3477–3488. New York, NY , USA: Association for Comput-
ing Machinery. ISBN 9781450383127.
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y .; Ishii, E.;
Bang, Y . J.; Madotto, A.; and Fung, P. 2023. Survey of Hal-
lucination in Natural Language Generation. ACM Comput.
Surv. , 55(12).
Jiang, J.; Zhou, K.; Dong, Z.; Ye, K.; Zhao, X.; and Wen, J.-
R. 2023a. StructGPT: A General Framework for Large Lan-
guage Model to Reason over Structured Data. In Bouamor,
H.; Pino, J.; and Bali, K., eds., Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language Pro-
cessing , 9237–9251. Singapore: Association for Computa-
tional Linguistics.
Jiang, J.; Zhou, K.; Zhao, X.; and Wen, J.-R. 2023b.
UniKGQA: Unified Retrieval and Reasoning for Solving
Multi-hop Question Answering Over Knowledge Graph. InThe Eleventh International Conference on Learning Repre-
sentations .
Li, T.; Ma, X.; Zhuang, A.; Gu, Y .; Su, Y .; and Chen, W.
2023. Few-shot In-context Learning on Knowledge Base
Question Answering. In Rogers, A.; Boyd-Graber, J.; and
Okazaki, N., eds., Proceedings of the 61st Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers) , 6966–6980. Toronto, Canada: Association for
Computational Linguistics.
Nie, Z.; Zhang, R.; Wang, Z.; and Liu, X. 2024. Code-
Style In-Context Learning for Knowledge-Based Question
Answering. Proceedings of the AAAI Conference on Artifi-
cial Intelligence , 38(17): 18833–18841.
Qiu, Y .; Li, M.; Wang, Y .; Jia, Y .; and Jin, X. 2018.
Hierarchical Type Constrained Topic Entity Detection for
Knowledge Base Question Answering. In Companion Pro-
ceedings of the The Web Conference 2018 , WWW ’18,
35–36. Republic and Canton of Geneva, CHE: International
World Wide Web Conferences Steering Committee. ISBN
9781450356404.
Reimers, N.; and Gurevych, I. 2020. Making Monolingual
Sentence Embeddings Multilingual using Knowledge Distil-
lation. In Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing . Association for
Computational Linguistics.
Shi, W.; Min, S.; Yasunaga, M.; Seo, M.; James, R.; Lewis,
M.; Zettlemoyer, L.; and Yih, W.-t. 2024. REPLUG:
Retrieval-Augmented Black-Box Language Models. In Duh,
K.; Gomez, H.; and Bethard, S., eds., Proceedings of the
2024 Conference of the North American Chapter of the As-
sociation for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers) , 8371–8384. Mexico
City, Mexico: Association for Computational Linguistics.
Sun, J.; Xu, C.; Tang, L.; Wang, S.; Lin, C.; Gong, Y .; Ni,
L.; Shum, H.-Y .; and Guo, J. 2024. Think-on-Graph: Deep
and Responsible Reasoning of Large Language Model on
Knowledge Graph. In The Twelfth International Conference
on Learning Representations .
Talmor, A.; and Berant, J. 2018. The Web as a Knowledge-
Base for Answering Complex Questions. In Walker, M.;
Ji, H.; and Stent, A., eds., Proceedings of the 2018 Confer-
ence of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies,
Volume 1 (Long Papers) , 641–651. New Orleans, Louisiana:
Association for Computational Linguistics.
Wang, L.; Hu, Y .; He, J.; Xu, X.; Liu, N.; Liu, H.; and
Shen, H. T. 2024. T-SciQ: Teaching Multimodal Chain-of-
Thought Reasoning via Large Language Model Signals for
Science Question Answering. Proceedings of the AAAI Con-
ference on Artificial Intelligence , 38(17): 19162–19170.
Wang, X.; Wei, J.; Schuurmans, D.; Le, Q. V .; Chi, E. H.;
Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Self-
Consistency Improves Chain of Thought Reasoning in Lan-
guage Models. In The Eleventh International Conference on
Learning Representations .
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.;
Xia, F.; Chi, E.; Le, Q.; and Zhou, D. 2023. Chain-of-

Thought Prompting Elicits Reasoning in Large Language
Models. arXiv:2201.11903.
Xu, Y .; He, S.; Chen, J.; Wang, Z.; Song, Y .; Tong, H.;
Liu, G.; Zhao, J.; and Liu, K. 2024. Generate-on-Graph:
Treat LLM as both Agent and KG for Incomplete Knowl-
edge Graph Question Answering. In Al-Onaizan, Y .; Bansal,
M.; and Chen, Y .-N., eds., Proceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Process-
ing, 18410–18430. Miami, Florida, USA: Association for
Computational Linguistics.
Yih, W.-t.; Richardson, M.; Meek, C.; Chang, M.-W.; and
Suh, J. 2016. The Value of Semantic Parse Labeling for
Knowledge Base Question Answering. In Erk, K.; and
Smith, N. A., eds., Proceedings of the 54th Annual Meeting
of the Association for Computational Linguistics (Volume 2:
Short Papers) , 201–206. Berlin, Germany: Association for
Computational Linguistics.
Zhang, W.; Jin, L.; Zhu, Y .; Chen, J.; Huang, Z.; Wang, J.;
Hua, Y .; Liang, L.; and Chen, H. 2025. TrustUQA: A Trust-
ful Framework for Unified Structured Data Question An-
swering. Proceedings of the AAAI Conference on Artificial
Intelligence , 39(24): 25931–25939.

Appendix
A Experiment Setup
A.1 Datasets
Table 4 details the dataset splits and the number of sam-
ples selected for our experiments. A notable distinction ap-
plies to the CWQ dataset, where the original annotations
provide only single answer entity for each question. To fa-
cilitate a fair evaluation, we expand the answer set by exe-
cuting the provided SPARQL (SPARQL Protocol and RDF
Query Language) queries to retrieve all grounded entities.
This methodology differs from that of ToG, GoG, and PoG,
leading to a discrepancy in the reported performance on the
CWQ dataset.
To ensure a fair comparison, both our proposed
method and all baseline models are evaluated on the
same knowledge graph. Consistent with the ToG frame-
work, we utilize the Freebase public data dump (i.e.
https://developers.google.com/freebase?hl=en), which we
deploy using the Virtuoso RDF database system.
Dataset #(Train/Dev/Test) #Total #Filtered
WebQSP 3,098/-/1,639 1,639 535
CWQ 27,639/3,519/3,531 1,000 449
GrailQA 44,337/6,763/13,231 1,000 645
Table 4: Statistics of the datasets.
A.2 Settings for Baselines
IO, CoT and CoT-SC For a fair comparison, we use the
reproduced code for the IO, CoT, and CoT-SC methods from
the ToG repository to infer answers.
KB-BINDER We use the official repository and use KB-
BINDER (6) (with majority vote) to infer answers.
TrustUQA While we use the official repository for infer-
ence, we reconstruct the Condition Graph using the same
knowledge graph as our method and other baselines to en-
sure a fair comparison. It is important to note that our recon-
structed graph differs from the original in two significant
ways: it is considerably larger in scale and exhibits a higher
degree of noise.
ToG, GoG and PoG For the baseline methods ToG, GoG,
and PoG, we infer answers using their official repositories,
adhering strictly to the default settings provided. To ensure a
fair and consistent comparison, we evaluate all methods, in-
cluding our proposed ARoG, using the evaluation protocol
established in the official repository of ToG. Additionally, to
investigate performance under a privacy-protected RAG sce-
nario, we modify the Entity Name Search step, which pre-
vents the leakage of sensitive entity information.
A.3 Implementation Details
In our method and all baselines, we employ GPT-4O-MINI -
2024-07-18 as the underlying LLM for a fair comparison,accessed via the OpenAI official API. For the Structure-
oriented Abstraction module and the Generator model, the
temperature parameter is set to 0 to ensure deterministic out-
puts. In contrast, for the Relation-centric Abstraction mod-
ule, the temperature is set to 0.4 to introduce controlled vari-
ability. Both the frequency penalty and presence penalty are
set to 0. The maximum token length for generation is capped
at 256. Across all experiments, the width and depth of ex-
ploration are both set to 3, striking a balance between per-
formance and efficiency. We repeat the experiment 3 times
and report the average scores. All experiments are conducted
on a server equipped with two Intel(R) Xeon(R) Silver 4310
CPU @ 2.10GHz and 128 GB RAM memory.
B Generalization Assessment
To validate the Generalization across different LLMs, we
perform experiments using Q WEN 3-32B-FP8 to replace
GPT-4O-MINI -2024-07-18 for our method and all base-
lines. For a comprehensive evaluation, we benchmark ARoG
against Pure-LLM and RAG-based baselines on the KGQA
task. As detailed in Table 5, our method consistently outper-
forms these baselines.
The experiments yield findings that are consistent with the
conclusions in the Section 4.3 Performance Comparison of
main paper.
Type MethodWebQSP CWQ GrailQA
#Tot #Fil #Tot #Fil #Tot #Fil
PureIO 57.6 19.2 43.9 14.2 26.9 8.1
CoT 57.8 0.0 46.3 0.0 28.7 0.0
SC 57.7 3.3 46.1 1.3 29.1 2.4
SPKB-BINDER 20.2 9.1 - - 38.5 39.1
KB-BINDER-R 44.9 39.3 - - 60.5 60.9
TrustUQA 49.2 52.5 - - - -
TrustUQA-R 54.1 59.2 - - - -
RAGToG 57.6 22.1 49.5 15.4 37.4 18.4
GoG 54.7 26.1 40.2 15.1 27.9 14.0
PoG 39.5 22.3 32.3 9.6 51.8 41.8
ARoG 72.7 64.3 54.3 36.5 78.7 74.8
Imp. ↑14.9↑5.1↑4.8↑21.1↑18.2↑13.9
Table 5: Comparison results (%) of Pure-LLM methods
(Pure), SP-based methods (SP), and RAG-based methods
(RAG), under the #Total setting (#Tot) and #Filtered setting
(#Fil), using QWEN3-32B-FP8. The “-” indicates that the
SP-based method cannot be run due to lacking special for-
mal query annotations. “Imp.”: the absolute improvement of
ARoG relative to the second-best performer. Bold : the best
score. Underline : the second best score.
C Case Study
Figure 6 presents a typical case from the testing results on
the CWQ dataset. We compare the results of ARoG, CoT,
ToG, GoG and PoG in answering the question “What does
the artist that was nominated for ‘The Audacity of Hope’

have a degree in?”. The underlying LLMs used are all GPT-
4O-MINI .
The case study illustrates the distinct operational mech-
anisms of each approach. In the first iteration of ARoG,
it retrieves a set of question-relevant relations, such as
“book.written work.author”. It then identifies the cor-
responding entity cluster, which includes “m.02mjmr”.
For this entity, ARoG searches and filters its adjacent
relations, and then utilize an LLM to abstract it into
“m.02mjmr (person)”. Concurrently, the Structure-oriented
Abstraction module abstracts the question into the abstract
concept path: “Common (artist) →nominated for →
The Audacity of Hope (work); Common (artist) →has
degree in →Communications (field of study)”. After
that, the Abstraction-driven Retrieval module retrieves the
question-relevant triplets, such as “The Audacity of Hope,
book.written work.author, m.02mjmr (person)”. All of the
retrieved triplets are provided to the Generator module, and
it generates a negative flag. Then, the topic entity is updated
to “m.02mjmr (person)”. Starting from this new topic
entity, the Abstraction-driven Retrieval module successfully
retrieves “m.02mjmr (person), people.person.education-
education.education.major field ofstudy, m.062z7 (aca-
demic discipline)”. Finally, the Generator module infers
the correct answer “m.062z7 (academic discipline)”, i.e.
Political Science.
In contrast, CoT relies on the LLM’s internal knowledge,
which is incorrect in this case, it produces a wrong answer.
Besides, ToG fails to identify the relevant anonymous
entity “m.062z7” and instead retrieves incorrect triplets
“m.02mjmr, award.award nominee.award nominations-
award.award nomination.award, m.0nh4p7s”. This fail-
ure occurs because ToG can not distinguish the entity
“m.062z7” from other entities, such as “m.0nh4p7s”.
Ultimately, ToG can not infer an answer and defaults to the
result from the CoT approach. Similarly, although GoG and
PoG introduce LLMs’ internal knowledge to supplement or
modify the retrieved triplets, they introduce errors.
From this analysis, it is evident that ARoG outperforms
CoT, ToG, GoG and PoG. ARoG success in retrieving
question-relevant triplets and infer the correct answers, uti-
lizing two abstraction strategies.
D Prompts
Here, we provide all the prompts used in ARoG. The specific
in-context few-shot is shown in code files.
D.1 Relation-centric Abstraction
% The prompt used in the relation retrieval.
"""
Please retrieve %s relations that contribute
to the question from the following
relations (separated by semicolons).
Q: Name the president of the country whose
main spoken language was Brahui in 1980?
Topic Entity: Brahui Language
Relations: language.human_language.
main_country; language.human_language.
language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.
parent; language.human_language.
writing_system; base.rosetta.languoid.
languoid_class; language.human_language.
countries_spoken_in; kg.object_profile.
prominent_type; base.rosetta.languoid.
document; base.ontologies.
ontology_instance.equivalent_instances;
base.rosetta.languoid.local_name;
language.human_language.region
The output is:
[’language.human_language.main_country’,’
language.human_language.
countries_spoken_in’,’base.rosetta.
languoid.parent’]
Q: {}
Topic Entity: {}
Relations: {}
"""
% The prompt used in the entity abstraction.
The ‘‘{{OBJECT_RELATIONS}}’’ and ‘‘{{
SUBJECT_RELATIONS}}’’ are replaced by
the related relations generated by the
relation filtering.
"""
Please provide the type with a short
description of the ’ENTITY’ based on the
context.
Here is an example:
Context:
’ENTITY’ is the object of verbs: City/Town
’ENTITY’ is the subject of verbs: Newspapers
; Time zone(s); Area; Contained by;
Population
The output is: {"type": "geographic location
", "description": "a geographic location
or administrative area that is
associated with specific characteristics
, such as a newspaper circulation areas,
time zones, and population statistics
."}
Context:
’ENTITY’ is the object of verbs: {{
OBJECT_RELATIONS}}
’ENTITY’ is the subject of verbs: {{
SUBJECT_RELATIONS}}
Now you need to directly output its type
with a short description. Fill in the
format below without other information
or notes:
{"type": "the type of the entity", "
description": "the short description of
the entity"}
The output is:
"""
D.2 Structure-oriented Abstraction
"""

Question: What does the artist that was nominated for ''The Audacity of Hope'' have a degree in?
CoTAnalysis: The LLM's internal knowledge is incorrect, preventing a correct answer.
Reasoning: First, the artist nominated for "The Audacity of Hope" is Common. Second, Common has a 
degree in Communications. The answer is {Communications}.
Answer: Communications.
ToGAnalysis: A lack of semantic information of crucial entity m.062z7 prevents a successful retrieval, so it 
fails to answer the question and defaults to the CoT answer.
Retrieved Triplets: 
[The Audacity of Hope, book.written_work.author, m.02mjmr]; 
[m.02mjmr,  award.award_nominee.award_nominations|award.award_nomination.award, m.0nh4p7s].
Answer: Communications.
GoGAnalysis: Retrieved triplets are meaningless; generated triples supplement them but introduce errors.
Retrieved Triplets:
[m.02mjmr, author.works_written, The Audacity of Hope];
[m.02mjmr, person.education|education.degree, m.014mlp]
Generated Triplets:
[m.0c9w3gg, nomination.nominee, Barack Obama];
[m.0c9w3gg, person.education, Harvard Law School]
Answer: Harvard Law School
PoGAnalysis: Retrieved triplets are meaningless; reflection restores partial semantic, but errors remain.
Retrieved Triplets:
[m.02mjmr, author.works_written, The Audacity of Hope];
[m.0xn8125, education.honorary_degree_recipient.honorary_degrees, m.02mjmr]
Triplets After Reflection:
[Barack Obama, author.works_written, The Audacity of Hope];
[Law, education.honorary_degree_recipient.honorary_degrees, Barack Obama]
Answer: Law
ARoGAnalysis: Successfully retrieves the answer by abstracting from entities m.02mjmr and m.014mlp.
Abstract Concept Path: 1. Common (artist) → nominated for → The Audacity of Hope (work);
2. Common (artist) → has degree in → Communications (field of study).
Retrieved Triplets with abstracted entities: 
[The Audacity of Hope, book.written_work.author, m.02mjmr (person)]；
[m.02mjmr (person), people.person.education|education.education.major_field_of_study, m.062z7 
(academic discipline)].
Answer: m.062z7 (academic discipline). (i.e. Political Science)Figure 6: A typical case to compare different methods to answer the complex question under the privacy-protected RAG scenario
for KGQA. Incorrect entities and wrong answers are highlighted in red. The noun concept of entities in the abstract concept
path and retrieved triplets are highlighted in blue (question-relevant entities) or green (answer entities).

Please answer the questions step by step,
the format of the reasoning process
should be kept consistent with examples.
Question: What state is home to the
university that is represented in sports
by George Washington Colonials men’s
basketball?
Thought: First, the education institution
has a sports team named George
Washington Colonials men’s basketball in
is George Washington University. Second
, George Washington University is in
Washington D.C.. The answer is
Washington, D.C..
Reasoning Path: {George Washington
University (education institution) ->
represented in sports by -> George
Washington Colonials men’s basketball (
sport team)}; {George Washington
University (education institution) ->
located in -> Washington D.C. (state)}
Answer: Washington D.C. (state)
other in-context few-shot...
Question: {}
"""
D.3 Generator
% The prompt used to output the flag.
"""
Given a question and the associated
retrieved knowledge graph triplets (
subject entity, relation, object entity)
, you are asked to answer whether it is
sufficient for you to answer the
question with these triplets (Yes or No)
. Note that while several entities are
represented as MIDs and corresponding
type (e.g., ’Taylor Swift’ could be
represented by ’m.0011 (person)’), you
should assume the formal names are known
.
Q: Find the person who said "Taste cannot be
controlled by law", what did this
person die from?
Knowledge Triplets: Taste cannot be
controlled by law., media_common.
quotation.author, m.0wfjf99 (person)
A: {No}. The retrieved knowledge graph
triplet provides a quotation and
identifies the author as a person (m.0
wfjf99), but it does not provide any
information about the cause of death of
that person. To answer the question
about what the person died from,
additional information regarding the
individual’s death is required, which is
not present in the provided triplet.
Therefore, the triplet is insufficient
to answer the question.
other in-context few-shot...Q: {}
"""
% The prompt used to output the answers.
"""
Given a question and the associated
retrieved knowledge graph triplets (
entity, relation, entity), you are asked
to answer the question with these
triplets. Please note that while other
entities are represented as IDs (e.g., ’
m.0011’), you should assume you are
familiar with their formal names.
Q: The artist nominated for The Long Winter
lived where?
Knowledge Triplets: The Long Winter, book.
written_work.author, m.0bvl_7 (person)
m.0bvl_7 (person), people.person.
places_lived|people.place_lived.location
, m.0wfjc51 (place)
A: {m.0wfjc51 (place)}. The artist nominated
for The Long Winter, m.0bvl_7 (person),
lived in a location that is represented
by the entity m.0wfjc51.
other in-context few-shot...
Q: {}
"""
E Search SPARQL
To automatically process the KG data in ARoG, we prede-
fine the SPARQL for Freebase queries, which can be exe-
cuted by filling in the entity’s mid and relation. Notably, we
flatten Compound Value Type (CVT) nodes in Freebase into
multiple single-hop relations in order to narrow the gap be-
tween candidate triplets and abstract concept paths.
E.1 Relations Search
% search all adjacent relations of a
specific head entity.
"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
ns:%s ?relation ?x .
}
"""
% search all adjacent relations of a
specific tail entity.
"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?relation
WHERE {
?x ?relation ns:%s .
}
"""

E.2 Entity Cluster Search
% search all adjacent entity cluster of a
specific head entity-relation pair.
"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
ns:%s ns:%s ?tailEntity .
}
"""
% search all adjacent entity cluster of a
specific tail entity-relation pair.
"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
?tailEntity ns:%s ns:%s .
}
"""
"""
% search all adjacent entity cluster of a
specific head entity-relation pair,
where the relation is flattened.
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
ns:%s ns:%s ?mid_entity .
?mid_entity ns:%s ?tailEntity .
}
"""
"""
% search all adjacent entity cluster of a
specific tail entity-relation pair,
where the relation is flattened.
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?tailEntity
WHERE {
?tailEntity ns:%s ?mid_entity .
?mid_entity ns:%s ns:%s .
}
"""
E.3 Entity Name Search
"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
FILTER (!isLiteral(?tailEntity) OR lang(?
tailEntity) = ’’ OR langMatches(lang(?
tailEntity), ’en’))
{
?entity ns:type.object.name ?tailEntity .
FILTER(?entity = ns:%s)
}
}
"""