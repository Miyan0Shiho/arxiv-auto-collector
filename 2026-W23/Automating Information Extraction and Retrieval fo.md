# Automating Information Extraction and Retrieval for Industrial Spare Parts Pooling

**Authors**: Dyuman Bulloni, Rocco Felici, Oliver Avram, Anna Valente

**Published**: 2026-06-02 09:14:58

**PDF URL**: [https://arxiv.org/pdf/2606.03367v1](https://arxiv.org/pdf/2606.03367v1)

## Abstract
Maintenance organizations in manufacturing try to avoid downtime and unnecessary purchasing by reusing existing assets, but the main obstacle is not a lack of parts but a lack of actionable visibility across sites and partners. Inventories are distributed, described with inconsistent naming conventions, and contain duplicates and partially specified references, so the right part often exists somewhere but remains effectively undiscoverable. The paper proposes PhRAG, a hybrid Retrieval-Augmented Generation for Pooling this fragmented landscape into a Virtual Stock Pool (VSPool) that can be structured and searched as a single resource. Unstructured, heterogeneous spare part descriptions are structured via Named Entity Recognition (NER) into a shared virtual pool dataset and indexed to support robust retrieval even when users express needs in natural language rather than exact technical specifications. The proposed modular pipeline leverages the multitasking nature of generative language models to cover two dimensions that make industrial parts pooling challenging: (i) unstructured technical specifications from diverse data sources (e.g. new partners, catalogs, marketplace listings) are handled through an offline extraction and (ii) request variability at runtime (references, partial references, specifications, price/condition constraints) is handled through a hybrid RAG-based search engine capable of retrieving relevant components and justifying results. The framework demonstrates the potential of generative approaches compared with traditional NER approaches in the presence of data scarcity for technical specifications extraction and overcomes the opacity of standard information retrieval systems by generating justifications for retrieved components. The project's open-source code can be found at https://github.com/roccofelici/vspool.

## Full Text


<!-- PDF content starts -->

Automating Information Extraction and Retrieval
for Industrial Spare Parts Pooling
Dyuman Bulloni1,†, Rocco Felici1,†, Oliver Avram1, and Anna Valente1
Abstract— Maintenance organizations in manufacturing try
to avoid downtime and unnecessary purchasing by reusing
existing assets, but the main obstacle is not a lack of parts but a
lack of actionable visibility across sites and partners. Inventories
are distributed, described with inconsistent naming conventions,
and contain duplicates and partially specified references, so the
right part often exists somewhere but remains effectively undis-
coverable. The paper proposes PhRAG, a hybrid Retrieval-
Augmented Generation for Pooling this fragmented landscape
into a Virtual Stock Pool (VSPool) that can be structured
and searched as a single resource. Unstructured, heterogeneous
spare part descriptions are structured via Named Entity Recog-
nition (NER) into a shared virtual pool dataset and indexed
to support robust retrieval even when users express needs in
natural language rather than exact technical specifications. The
proposed modular pipeline leverages the multitasking nature
of generative language models to cover two dimensions that
make industrial parts pooling challenging: (i) unstructured
technical specifications from diverse data sources (e.g. new
partners, catalogs, marketplace listings) are handled through
an offline extraction and (ii) request variability at runtime
(references, partial references, specifications, price/condition
constraints) is handled through a hybrid RAG-based search
engine capable of retrieving relevant components and justifying
results. The framework demonstrates the potential of genera-
tive approaches compared with traditional NER approaches
in the presence of data scarcity for technical specifications
extraction and overcomes the opacity of standard information
retrieval systems by generating justifications for retrieved
components. The project’s open-source code can be found at
https://github.com/roccofelici/vspool.
I. INTRODUCTION
In the context of industrial maintenance, spare parts man-
agement aims to ensure the availability of components and
reduce idle stock. Sharing spare parts inventories between
industrial plants or within a community could reduce the size
of individual partners’ stocks and establish a policy of dor-
mant components circularity aimed at reducing overall waste.
This objective introduces several technological challenges,
in particular, (i) the need to create a common pool from
several inventories, i.e., multiple unstructured data sources,
and (ii) the ability to retrieve parts that meet the technical
specifications of plants requiring maintenance. The research
presented in this publication aims to evaluate such challenges
and proposes a framework to address them via a multitasking
architecture relying on generative language models.
1Automation, Robotics and Machines (ARM), Institute of Systems
and Technologies for Sustainable Production (ISTePS), University
of Applied Sciences and Arts of Southern Switzerland (SUPSI),
Lugano, Switzerland.{dyuman.bulloni, rocco.felici,
oliver.avram}@supsi.ch†Equal contribution.A virtually shared pool inventory is required to accurately
store relevant information, track quantities, and ensure that
all stakeholders joining the community can easily access
and order the exact parts needed. This gives rise to the
need to structure a vast amount of textual data referenced
with predefined entities from a highly technical domain
viatechnical specification extraction. NER is one of the
key techniques in Natural Language Processing (NLP) for
information extraction tasks. Its main goal is to organize
unstructured textual data into structured representations, en-
abling flexible interpretation and improving the usability of
such information. Although it is widely used in many con-
texts, the adaptation of such an approach for domain-specific
applications is too often a repetitive and tedious process that
comprises continuous iterations of data labeling, training, and
model evaluation [1], [2]. In particular, in the manufacturing
domain, the presence of highly specific terminology makes
it difficult to train models capable of extracting information
from unstructured textual data [3]. Furthermore, solutions
of this kind are heavily dependent on the creation of large
amounts of specially annotated data [4]. In this research,
we propose a language model-based approach that relies on
large pre-trained models and their autoregressive generation
of text to solve a NER task on data highly specific to the
manufacturing domain. To ensure reproducibility, the paper
aims to describe the applied methodology, provide an open-
source version of the implementation, and display findings
on FabNER [3], a publicly available benchmark dataset
within the same domain. The dataset addresses the task
of annotating industrial product descriptions with relevant
technical attributes under domain-specific constraints.
Regarding theretrieval of relevant spare parts, a search
engine tailored for this application has been developed to
map textual requests containing technical requirements of
the user’s needs to the data contained in the shared struc-
tured inventory. It is implemented by modularly reusing
components already developed for the extraction of technical
specifications, demonstrating the ability of language models
to adapt to different tasks within the same technical domain.
Its implementation focuses on efficiency and low-latency
inference suitable for industrial deployment and interactive
querying by the user. The evaluation of the retrieval capabil-
ities is conducted on VSPool, a proprietary dataset.
The whole research aims to evaluate the multitasking
capabilities of language models used as building blocks for
the task of technical information extraction and as part of
the retrieval of industrial components in the manufacturing
domain.arXiv:2606.03367v1  [cs.IR]  2 Jun 2026

Organization:We first discuss related work in Sec. II.
Then present our framework in Sec. III, discussing how
to solve the technical specifications extraction task and
the components retrieval problem in a modular fashion. In
Sec. IV we evaluate our method for the two tasks. Finally,
we conclude by proposing future developments in Sec. V.
II. RELATED WORK
A. Technical Specifications Extraction
Classical NER approaches [3] that tackle highly special-
ized domains are primarily supervised and rely on labeled
corpora to learn mappings between textual data and semantic
categories.
Early NER systems were predominantly rule-based and
were used to identify entities with well-defined lexical or
syntactic patterns, e.g., leveraging regular expressions [1].
Such approaches are still widely used in industrial settings,
particularly in constrained domains where entity formats are
stable, and domain knowledge can be explicitly encoded.
However, rule-based systems typically suffer from limited
scalability and poor generalization to unseen variations. To
overcome such limitations, the proposed approach described
in Sec.III relies on language models that, leveraging large-
scale pre-training, provide generalist capabilities on NLP
tasks and rapid adaptability to unseen linguistic problems.
With the availability of annotated data, statistical and
neural approaches became dominant. Recurrent neural ar-
chitectures such as Bidirectional Long Short-Term Mem-
ory networks (BiLSTMs), often combined with Conditional
Random Fields (CRFs), demonstrated strong performance
by modeling sequential dependencies and contextual infor-
mation [2]. Despite their success, supervised neural NER
models remain highly dependent on annotated data and
predefined entity schemas.
More recently, transformer-based models have achieved
state-of-the-art results in NER by leveraging deep contex-
tualized word representations learned through large-scale
pretraining. The latest paradigm leverages Large Language
Models (LLMs), which exhibit strong contextual reason-
ing capabilities and can handle unseen or loosely defined
entity types. Unlike traditional NER models, LLM-based
approaches are flexible with respect to data availability,
supporting zero-shot learning, few-shot learning, and stan-
dard supervised learning. These models can perform entity
extraction through prompting or lightweight fine-tuning, re-
ducing the need for task-specific annotation [5], [6], [7].
However, LLM-based approaches introduce new challenges,
including a trade-off between response time and output
quality, sensitivity to prompt design, and the occurrence
of hallucination-generated outputs that are not grounded in
the input data [8]. To this end, the framework proposed
includes adaptation to the manufacturing domain via a few-
shot in-context examples compatible with both small and
large language models. The latter could be used during
the offline phase, and the former during the deployment of
architectures designed for runtime and user interaction.NER in manufacturing operates in a different regime
from typical cases or even many domain-specific ones. The
texts are generally short and noisy, using technical domain
knowledge as context for abbreviations, metrics, and a lack
of cohesion for even the general categories. As a result of
that, models trained on standard datasets perform poorly due
to domain shift and a high rate of rare or unseen terms.
Entity types in this domain usually include components,
materials, processes, machines, parameters, defects, and
measurements, but also high technical tabular data in raw
format. Unlike traditional NER categories (e.g., PER, ORG,
LOC), these entities are highly specialized and often ap-
pear as long compound phrases, making boundary detection
more challenging. Research on the topic spans from fine-
tuned architectures for guideline-following [7] to continual
pretraining on curated manufacturing data [4]. To tackle
such challenges, we opted for a hybrid RAG approach to
gather in-context examples for a reliable NER in a specific
domain. Unlike semantic-only retrieval, the hybrid approach
is especially valuable in manufacturing because it preserves
details such as reference excerpts and manufacturer codes i.e.
identifiers with no intrinsic semantic meaning that semantic
embeddings typically fail to capture.
B. Components Retrieval
Language models are increasingly integrated into informa-
tion retrieval pipelines within RAG systems. In such archi-
tectures, language models are often employed as rerankers
to refine document ranking by assessing query–document
relevance through their semantic understanding [9]. Recent
works also explore more flexible and modular RAG frame-
works, in which specialized components, such as retrievers,
rerankers, and generators, are combined to improve retrieval
quality and adaptability to domain-specific tasks [10], [11].
Building on this line of research, we propose a search engine
adapted for the manufacturing domain that leverages a lan-
guage model within a modular RAG to both rerank retrieved
components and provide justifications for the results, thereby
improving transparency. The search engine is implemented
to reuse, as shared modules, the embedding models, rank
fusion method, and generative language model introduced in
thetechnical specifications extractioncomponent above.
III. PROPOSED APPROACH
The PhRAG framework proposed consists of two phases,
thetechnical specifications extraction(Sec. III-A) and the
search engine(Sec. III-B). As depicted in Fig. 1, the former
takes place offline and integrates unstructured data from var-
ious sources via NER into the Virtual Stock Pool (VSPool)
dataset. The engine instead employs the VSPool as a basis
for the retrieval of spare parts whenever a user, looking for
a component, interacts with the system. Both share the use
of pre-trained generative models and exclude fine-tuning,
proposing a few-shot approach whose examples are identified
and employed through a hybrid RAG [12].

examples of
requirements extraction
unstructured corpus
of components
semantic
embeddingstf-idf
encodings
most relevant
examples
generated
entitiesembedding
model
tf-idf
rank
fusion
generative
model
VSPooluser’s
querycomponents
descriptions
tf-idf
encodingssemantic
embeddings
ranking of top-k
retrieved components
final ranking
& justificationspecifications extraction (offline)
PhRAG modulessearch engine (runtime)
Fig. 1. PhRAG modular framework
A. Technical specifications extraction
One of the fundamental requirements of the proposed
framework is the definition of a common, consistent data
representation that remains compatible across heterogeneous
sources, namely the VSPool dataset. To this end, the tech-
nical specifications extraction is structured to enable the
systematic collection and organization of technical data.
Formally, we are given a corpusD={d i}N
i=1 of
unstructured textual descriptions of industrial components
originating from different sources, and a finite, pre-defined
collectionS={s j}M
j=1of technical specification types of
interest. The task of technical specification extraction can
be formulated as a NER problem. Each descriptiond∈ D
is represented as a sequence of tokensd= (t 1, t2, . . . , t |d|).
The goal is to assign to each tokent ka labels kfrom a label
set derived fromS, producing a corresponding sequence of
labelss∗= (s 1, s2, . . . , s |d|).
For the annotation of each descriptiondin the corpus, we
rely on a RAG. We retrieve a setV K={v k}K
k=1⊆ Vof
Kexamples from a cacheVof pre-annotated descriptions.
Such examples comprise pairs of tokens and correspondent
technical specifications,v={(t i, si)}|d|
i=1, ti∈d, s i∈ S.
The retrieval is defined as the pipelineϕ(V K|d)that,
given the descriptiondof a component, provides the set
of examples that are most relevant to that description. The
retrieval provides contextual information that, through few-
shot prompting [13], guides the generation towards more
reliable extraction.
The generation returns the sequence of technical require-
ments extracteds∗= (s 1, s2, . . . , s |d|)associated to thetokenized descriptiond= (t 1, t2, . . . , t |d|)as follows:
pη(s∗|d,V K) =|d|Y
i=1pη(si|d,V K, si−1).(1)
Examples retrieval:Retrieval, in the context of the NER
problem, refers to selecting the most relevant examples that
can guide, via few-shot prompting, the language model
in generating the appropriate sequence of labels. In the
proposed approach, retrieval is defined in a hybrid manner
that combines semantic and lexical retrievers. Given a set
Vof annotated descriptions, these are used to create tf-idf
sparse encodings and semantic dense encodings. Semantic
embeddings are createdx i=fθ(vi)∈Rn, vi∈ Vusing
a pretrained transformer-based embedding modelf θwith
parametersθ. The embedding modelf θused is all-MiniLM-
L6-v2 [14], an uncased sentence embedding model distilled
from BERT with12layers, hidden size of384, and33M
parameters.
For the retrieval of in-context examplesV K, the descrip-
tiondis embedded and compared using the Okapi Best
Matching (BM25) function [15] to provide a relevance score
to find more relevant examples via sparse embeddings.VBM25
K
is the set ofKannotated descriptions with the highest BM25
scores. A second set of relevant examplesVSEM
K is obtained
by comparing the dense embedding of the descriptiond
against embeddings of component descriptions inV. Com-
parison between embeddings is done viak-nearest neighbors
and similarity score between description embeddingsx, y,
L2-normalized asˆx=x
∥x∥2,ˆy=y
∥y∥2, is defined as
cosine similarity:S cos(x, y) = ˆx⊤ˆy. Both sets of annotated
descriptionsVBM25
K andVSEM
K are ranked from most to least
relevant. DefinedR={rBM25, rSEM}as the set of the
rank functions of bothVBM25
K andVSEM
K, to find the optimal
intersectionV Kbetween the two, Reciprocal Rank Fusion
(RRF) [16] has been employed:
RRF 
v∈ VBM25
K∪ VSEM
K
=X
r∈R1
b+r(v)
whereb= 60. The ranking provided by RRF scores defines
the examplesV K.
Entities generation:Examples inV Kcomprise sequences
of token-technical specification pairs(t, s)for all tokens
of each description. A decoder-only language model [17],
[18] is prompted with information about the NER task to
be solved, the definition of the entities to be generated,
theKexamples andd= (t 1, t2, . . . , t |d|), and the unseen
description to annotate. The generation returns the sequence
of technical specifications extracteds∗= (s 1, s2, . . . , s |d|)
associated to each token ofd. At each step, the generation
is constrained only to available technical specificationS,
masking generations of non-valid specifications with logit=
−∞. This guarantees the generated sequence always remains
valid.
B. Search engine
As presented in the previous Sec. III-A, technical speci-
fications extraction is crucial to join together unstructured

technical data from different plants. Once the categories
and technical specifications of interest are defined asS=
{sj}M
j=1, we construct the VSPool, a dataset of industrial
componentsC={c i}N
i=1. Each componentc iis associated
with a set of technical specificationsS i⊆ S, yielding the
dataset representation, VSPool={(c i,Si)}N
i=1. The pool of
components is shared among all partners, and the search
engine draws from such a large collection of structured
information regarding available components across the whole
community. The engine is implemented to allow the retrieval
of spare parts that a certain partner may need. Allowing any
partner to find components among the shared inventory is
necessary to ensure circularity of dormant stock across the
part-sharing community.
In practice, the search engine accepts as input a queryq
of the user in natural language, against which it provides
a ranked list of relevant spare partsC i⊆ Cthat meet
the technical needs of the user. The engine is based on a
search service and a graphical interface where a search bar,
filters, and facets are depicted. The objective is, in fact, an
information retrieval task and was addressed through the use
of a RAG.
The retrieval consists of a pipelineϕ(C i|q)that, given
the queryq, provides a preliminary ranked listC i. Unlike,
the generation for the technical specifications extraction
(Sec. III-A), here no example is given instead, in a zero-
shot manner, the ranked list is prepended to the prompt of
the language model that is employed as reranker to provide
(a) an optimal final rankingC∗and (b) transparency in the
form of an explicit justification of the provided ranking.
Retrieval:Given VSPool, in the form that will be detailed
in Sec. IV-A, the retrieval for the search engine focuses
on the following variables:reference,nameanddescription
of components. The retrieval consists of a hybrid approach,
i.e., during an offline phase, these data are encoded via tf-
idf and embedded in dense vectors of dimension384using
the pretrained model all-MiniLM-L6-v2 [14]. Data is then
indexed in an Elasticsearch cluster [19].
At runtime, the lexical search is focused on references
and keywords. The implementation is based on the BM25
function [15] to provide a relevance score function for the
ranking of the components retrieved. Given a queryqfrom
the user, a spare part name, reference, or descriptiondfrom
the set of dormant stock VSPool, the BM25 score for such
components is defined as:
BM25(d) =X
i∈qlogN
d fi×(k1+ 1)tf i
k1((1−b) +bdl
avdl) +tf i
wherelogN
d fiis the idf, i.e., the ratio between the total
number of spare parts in the whole inventory of dormant
stock and the number of spare parts descriptions that contain
thei-th term of the queryq;tf idefines the frequency of
thei-th term of queryqin the spare part descriptiond;
dl=P
i∈dtfiis the length of descriptiondof a component,
avdl, the average length of spare parts descriptions over the
whole inventory. Parametersb∈[0,1]andk∈R ≥0are setto0.75and1.5respectively. BM25, with respect to other
traditional approaches based on tf-idf encodings, integrates
document length into the scoring evaluation but essentially
still relies heavily on exact word matching. To overcome
this limitation, the Levenshtein edit distance with parameter
l= 2is introduced, allowing up to two editing operations
(namely substitution, insertion, or deletion of a character)
during word comparison.
Reference, name, and description of spare partsd i∈
Cfrom the inventory are also embedded in a latent em-
bedding, i.e., a dense vectorial representation, defined as
xi=fθ(di)∈Rdusing a pretrained language embedding
modelf θwith parametersθ[14]. The similarity between a
queryqof a user interacting with the search engine and a
descriptiond iof thei-th component is measured based on
the corresponding embeddingsx, yL2-normalized asˆx=
x
∥x∥2,ˆy=y
∥y∥2, is defined as cosine similarity:S cos(x, y) =
ˆx⊤ˆy. The comparison of embeddings is done via10-nearest
neighbors using cosine similarity as a distance measure
between vectors.
Both retrieval approaches output a ranking of top-nrele-
vant spare partsCLEX
n,CSEM
n; a combination of rankings from
the two is implemented defining RRF [16] as:
RRF 
c∈ CLEX
n∪ CSEM
n
=NX
ri∈Rwi
b+r i(c)(2)
whereR={rLEX, rSEM}is the set of ranking functions,
wi= 1,∀ito ensure equal contribution of the lexical-based
ranking and the semantic one, and the parameterbis set to
60.
Filtering:At run-time, the users can either interact with
the system through a search bar providing a free text query
qin natural language or selects filters and facets to explicitly
define preferences regarding the spare parts they are looking
for. Filtering mechanisms have been implemented to refine
the ranking output by the search modulesϕ(C i|q, f)only to
the subset of components that would meet the requisitesf
of the user. Filtersfare manually applied by the user or
automatically applied wheneverqcontains technical infor-
mation. Filters on VSPool are available for the following
variables:condition,brand,categories, andpricerange.
Automatic filtering is implemented with low-latency entities
extractions, finetuning the en core web sm [20] pipeline on
ad-hoc datasets [21] (1.371annotated queries for conditions,
100annotated instances for prices,373queries for brand and
categories).
Reranker:The set of ranked pertinent spare parts retrieved
is then refined by a language model prompted in a zero-
shot manner. A pre-trained model, Llama3.1[22], with
7B parameters running locally. The prompt is formed by
the query of the userq, the ranking retrievedC i, and a
detailed description of the task. The generation includes the
proposal of a definitive ranking,C∗, and a detailed justifi-
cation for the choice. Compared to traditional information
retrieval systems, the proposed approach retrieves relevant
results, providing technical justifications of the choice to the

user, discussing their requirements against the components
retrieved. Not only is the explanation useful to provide
transparency of the system for the user, but it also improves
the performance of the model in ranking the best parts first.
IV. RESULTS
A. Datasets
For the evaluation of the technical specifications extraction
introduced in Sec. III-A, it is considered FabNER [3], which
is a collection of datasets of manufacturing text comprising
more than350.000words for NER purposes. Unstructured
text is annotated for every word with entities representing
categories of industrial processes: Material (MATE), Manu-
facturing Process (MANP), Machine/Equipment (MACEQ),
Application (APPL), Features (FEAT), Mechanical Proper-
ties (PRO), Characterization (CHAR), Parameters (PARA),
Enabling Technology (ENAT), Concept/Principles (CON-
PRI), Manufacturing Standards (MANS), and BioMedical
(BIOP). In particular, for the validation, we referred to
the FabNER-simple dataset composed of9.435training
instances and2.064test ones. The dataset consists of an id
for each sentence (string), the list of tokens of the sentence
(list of strings), and the list of entity tags (list of integers).
The Virtual Stock Pool (VSPool) dataset has been con-
structed for validation purposes of the search engine. The
dataset comprises4.020components for industrial processes
across eleven brands. Spare parts were selected among the
most representative brands on the market, i.e., among the top
eleven brands in terms of the number of components avail-
able on component providers’ marketplaces. Selection has
been made to provide variability in terms of industrial pro-
cesses, reference systems across several distributors and con-
structors. Each component presents the following variables:
query (free text), reference (alphanumeric id, unique), cat-
egories (list of strings, keywords), brand (string, keyword),
condition (string, keyword), general discursive description
(free text), description embedding (dense vector), name or
short description (free text), name embedding (dense vector),
price (float, currency in euros), technical specifications (key,
field, unit, value) respectively in the format (string, string,
string, float), set of pictures, attachments. Among these
the query, in natural language, represents a query that an
operator could use to look for the component. Categories and
facets are defined a priori based on hierarchical classification
schemes. The degree of specialization increases as cate-
gory subsets become progressively narrower. For example,
the categoryelectrical equipmentincludes the subcategory
switches, which in turn contains the more specialized subset
toggle switches&accessories. Condition is a fixed string
corresponding to different levels of item conditions, e.g.,
refurbished,new,used. The description usually comprises the
possible application of the spare part, the general context of
its usage, and some details on its characteristics. The price in
euros is stored for each part without considering any taxation.
Technical specifications are saved as a tuple of key, value
pairs which include a description of the specification, e.g.,
maximum flowand the corresponding value and UoM, e.g.,1.8 liters per second. Attachments comprise links to valid
offers on marketplaces.
B. Metrics
The evaluation of the technical specification extraction
is based on the FabNER-simple [3] test set with2.064
annotated instances and relies on the following metrics:
Precision=TP
TP+FP,Recall=TP
TP+FN,
F1score= 2×Precision×Recall
Precision+Recall,
where true positive (TP) refers to the entities correctly
predicted, false positive (FP) to an entity predicted but not
present in the ground truth, and false negative (FN) to an
entity not predicted but present in the ground truth.
Instead, the assessment of the search engine is based on
the VSPool dataset (described in Sec. IV-A) and consists of
the evaluation of the ranked retrieval results that a search pro-
vides. In a supervised setting, the accuracy at k (or A@k) is
defined as the number of times where the correct components
are found among the top k results in the predicted ranking.
C. Evaluation
The evaluation is divided into two phases: the test of
the technical specification extraction (Sec. IV-C.1) on the
FabNER test set and the analysis of the performance of
the search engine (Sec. IV-C.2) on the VSPool dataset.
Experiments were conducted on a Lenovo 82WQ with a
dedicated NVIDIA GeForce RTX 4080 Laptop GPU running
Ubuntu 22.04.4 LTS.
1) Technical specification extraction evaluation:The first
experiment involves comparing generative language-based
models with their PhRAG version to quantify the added value
of the framework detailed in Sec. III-A. Small Language
Models (SLMs) [17] and an LLM1are selected. Base models
are prompted with the same task definition, entities defini-
tion, and a few static few-shot examples. The same models
are employed in the corresponding PhRAG. Temperature is
set to zero for all models to provide reproducibility of the
results. Up to five iterations with different pseudo-random
seeds are executed to evaluate variability. An absent standard
deviation indicates negligible variability between runs.
SLM LLM
Phi-3.5-mini PhRAG3.5BCode
Llama7BPhRAG7B GPT-5.2 PhRAG
F1score
avg±std0.04 0.10 0.06±0.01 0.19±0.2 0.36±0.016 0.59±0.012
TABLE I
COMPARISON BETWEEN BASE MODELS ANDPHRAG
FOR3.5B,7BAND100B+PARAMETERS ONFABNER.
Results are presented in Tab. I in terms of weighted aver-
age F1 scores across all entities in the FabNER test set. The
findings indicate performance improvements across language
1We consider small models with less than10B parameters and large
models with more than100B parameters.

models of all sizes. The improvements are more pronounced
for larger models. Notably, when PhRAG is integrated with
a state-of-the-art LLM, a substantial performance gain is
observed, achieving an improvement of up to23percentage
points over the base.
A second analysis focuses on evaluating the gap that
persists between few-shot prompted generalist models and
strictly supervised approaches that, fine-tuned on the task,
represent the state of the art. The evaluation is important
because, although supervised approaches generally achieve
superior performance, their effectiveness is constrained by
the availability of training data, which is not always acces-
sible in the manufacturing industry. Supervised approaches
are trained on9.44k documents annotated with technical
specifications. Different approaches have been evaluated on
the FabNER [3] test set described in Sec. IV-A. Selected
approaches for the validation are: small pretrained gener-
ative language models or SLMs (Phi-3.5-mini, Code Llama
7B [17]), large language models (a fine-tuned model GoLLIE
34B [7] for guidelines following and GPT-5.2-2025-12-11),
and an ad-hoc trained pipeline implemented or replicated
from state of the art to serve as a benchmark: SpaCy
encore web sm pipeline and BiLSTM+CRF [3]. Results in
SLM LLM Supervised training
Phi-3.5-miniCode
Llama7BGoLLIE34B GPT-5.2 PhRAG SpaCyBiLSTM
+CRF
F1score
avg±std0.04 0.06 26.3±0.4 0.36±0.016 0.59 0.82±0.01 0.88
TABLE II
COMPARISON BETWEEN SUPERVISED MODELS,GENERATIVE
APPROACHES,ANDPHRAGONFABNER.
Tab. II provide the overall weighted average F1 score for the
approaches in comparison. Scores are weighted with respect
to the unbalanced classes of the entities. Temperature and
number of iterations are kept as in the first experiment,
respectively, zero and five. Despite the enormous general-
ization capabilities of language models, the state of the art
in technical specification extraction for the manufacturing
domain remains predominantly dominated by approaches
trained ad hoc for the task in question. PhRAG, in its
biggest size, significantly reduces the gap between generative
and supervised approaches despite operating without task-
specific fine-tuning. While supervised models still achieve
the best performance, PhRAG shows that a modular retrieval-
augmented design can substantially improve unfinetuned,
LLM-based NER while also supporting the needs of the
search engine described in Sec. III-B.
In Fig. 2, are shown the results of Tab. II, in more
detail, with an overview of performances across all entities
comprised in the test set. It is possible to notice that few-
shot prompting approaches for SLMs still suffer from a large
performance gap compared to ad-hoc trained technologies
such as SpaCy for the task of technical information extrac-
tion: Code Llama7B presents, in the best case, a gap of36
percentage points in performance for the entity MANS and
83-points difference for the entity PARA, which is the worst0%20%40%60%80%100%training data percentage
MATE MANP MACEQ APPL FEAT PRO CHAR PARA ENAT CONPRI MANS BIOP0.00.20.40.60.81.0 f1-scoreCode Llama
GPT -3.5 TurboGPT -5.2 PhRAG SpaCy (fine-tuned)per entity percentage
of training data
Fig. 2. Entity-level comparison between supervised approaches, generative
approaches, and PhRAG on FabNER.
case scenario. LLMs reduce the performance gap relative
to smaller models and the state of the art. Nonetheless,
PhRAG furthermore narrows the difference between LLMs
and supervised approaches and, in some cases, matches the
benchmark supervised approaches. Data scarcity significantly
affects the performance of supervised learning approaches.
SpaCy’s performances go below0.60only for entities with
scarce training data: MANS,45annotated entities, and BIOP
107. In these extreme cases, e.g., MANS entity, PhRAG
presents an improvement to the state of the art.
To better evaluate the impact on performance of limited
data availability, we reduced the training dataset for SpaCy’s
training to10%of the original size, observing a decrease in
performance from0.82±0.01to0.67±0.00in the presence
of scarce training data. On the other hand, PhRAG, with
no training and a set of examples limited to10%, goes
from0.59±0.012to0.55in terms of weighted average
F1 score. PhRAG’s robustness in the face of variations in
available data, in contrast to traditional approaches, makes
it particularly useful in the manufacturing sector, where
annotated, highly specialized data is often missing.
Up to now, we have discussed only unfinetuned versions
of PhRAG, since in the manufacturing sector, curated data
is not always available for finetuning. As a final remark,
we underline that also finetuned language models can be
employed in PhRAG. In this regard, a fine-tuned version (on
FabNER training set) of Code Llama-7B model is obtained
via Low-Rank Adaptation (LoRA) [23], [24] and equipped
with PhRAG. It is observed a drastic improvement with
respect to the base model (Tab. II), by more than10times
the original F1 score, reaching a0.67weighted F1-score. An
approach that combines PhRAG and an efficient finetuning
on scarce data is left for future developments.
2) Search engine evaluation:The search engine intro-
duced in Sec. III-B has been evaluated in two respects.
The first is the measure of the impact on performances of
different embedding models; the second is an ablation study
to determine the contribution of the different parts of the
system.
Embedding models selection:Key aspects in the choice
of a suitable embedding model for the semantic search,
presented in Sec. III-B, are the dimension of the embeddings

and the impact they have on the quality of semantic search.
Larger embeddings have greater representational capacity but
suffer from longer inference times for embedding creation.
To balance these aspects, the impact of embedding models
in the semantic search is evaluated in a supervised setting
on the VSPool dataset described in Sec. IV-A. The dataset
comprises queries in natural language and corresponding
components. The queries are embedded and employed in
the semantic search to observe within how many positions
kin the ranking the spare part corresponding to the query
would have been retrieved. Such an evaluation has been
done fork= 1,5,10and averaging the results across these
three values ofk. Results are available in the Tab. III. The
set of models evaluated represents the current state of the
art [25]. Best results (underlined in the Tab. III) correspond
ModelEmbedding
dimensionA@1A@5A@10avg A@k
bge-m31024 0.716 0.836 0.875 0.809
all-MiniLM-L12-v2384 0.633 0.798 0.841 0.757
all-MiniLM-L6-v2384 0.627 0.8 0.842 0.756
gte-modernbert-base768 0.627 0.795 0.84 0.754
gtr-t5-large768 0.621 0.796 0.836 0.751
m3e-base768 0.625 0.789 0.839 0.751
all-mpnet-base-v1768 0.6 0.772 0.836 0.736
multi-qa-mpnet-base-cos-v1768 0.593 0.765 0.822 0.727
all-mpnet-base-v2768 0.554 0.734 0.804 0.697
paraphrase-MiniLM-L6-v2384 0.548 0.719 0.778 0.682
TABLE III
EMBEDDING MODELS EVALUATION:
SPARE PARTS FOUND WITHIN TOP-kRESULTS(k= 1,5,10)
to the model bge-m3 [26]; nonetheless, such a model has
embeddings of size1024, which slows down the process for
the creation of the vectors. The fastest model is all-MiniLM-
L6-v2 [14] that still presents optimal performance in terms
of parts found when employed in the semantic search.
Ablation study:To demonstrate the effectiveness in
retrieving relevant spare parts based on the textual input
of the user, the ranking of the proposed spare parts by
the search engine. Ordering of relevant spare parts output
by the search engine is evaluated in terms of top-k, i.e.,
the capability of including relevant spare parts among top-
kresults in the ranking. The evaluation has been done for
several values ofk= 1,5,10,15,20. To better understand
and quantify the contribution of each component to the
system, the evaluation involves the tests of all three search
modules, namely, lexical, semantic, hybrid, and the whole
PhRAG system, in a supervised way employing the VSPool
dataset. Results are presented in Tab. IV.
The several components of the search engine are detailed
in Sec. III-B. The lexical search approach based exclusively
on product reference ids effectively deals with complete,
partial, or partially incorrect references of spare parts, but
less than21%of the cases include the correct part among
the top-20positions of the provided rankings. The standalone
semantic approach is promising; it captures contextual infor-
mation regarding the needs of the operator but loses details
regarding references and specific technicalities. Extending
the lexical retrieval to product descriptions and technicalComponents
Lexical
(reference only)SemanticLexical
(benchmark)HybridPhRAG
(hybrid & reranker)
A@10.167 0.693 0.736 0.78 0.805
A@50.188 0.799 0.824 0.838 0.857
A@100.193 0.827 0.845 0.853 0.866
A@150.203 0.834 0.851 0.855 0.868
A@200.209 0.839 0.852 0.855 0.869
TABLE IV
ABLATION STUDY:
SPARE PARTS FOUND WITHIN TOP-kRESULTS(k= 1,5,10,15,20)
specifications, the search engine ensures to rank the correct
components at the first position in around73%of the cases,
and more than84%in the top-ten ranked spare parts. The
hybrid search encapsulates, via ranking fusion, the set of
spare parts given by the lexical and the semantic search.
Reranking via a generative language model not only im-
proves performance but also provides an explicit justification
of the retrieved components by reviewing the ranked spare
parts against the query and the operator’s requirements. This
latter capability, although not directly quantified, represents
a significant advance over the state of the art, as traditional
systems typically lack mechanisms to provide transparent
reasoning or explanatory validation for the retrieved results.
V. CONCLUSION
This paper introduces PhRAG, a hybrid RAG frame-
work designed to leverage the generative and multitasking
capabilities of language models for technical information
extraction and component retrieval in the manufacturing
domain. It integrates generative models with semantic and
lexical retrieval mechanisms to support two complementary
tasks: (i) the construction of a virtual spare-parts pool that
consolidates heterogeneous unstructured textual data into
a unified representation, and (ii) the implementation of a
search engine for component retrieval from unstructured user
queries. The framework, evaluated on benchmark datasets,
shows improvements in performance across models of dif-
ferent sizes. The results demonstrate that PhRAG effectively
addresses both objectives by employing shared components
within a modular framework, without the need for fine-
tuning. The method boosts base model performance up to
fine-tuned approach capabilities in scenarios with limited
annotated data (such as the case of the MANS entity)
and illustrate the value of generative models in refining
search engine results while also providing justifications of
the retrieved components for the user.
ACKNOWLEDGMENT
This research has been partially funded by the EIT man-
ufacturing project VSPOOL: Virtual Stock Pooling Platform
(grant ID: 24292).

REFERENCES
[1] D. Nadeau and S. Sekine, “A survey of named entity
recognition and classification,”Lingvisticae Investiga-
tiones, vol. 30, no. 1, pp. 3–26, 2007 (cit. on pp. 1,
2).
[2] G. Lample, M. Ballesteros, S. Subramanian, K.
Kawakami, and C. Dyer, “Neural architectures for
named entity recognition,” inProceedings of NAACL-
HLT, 2016, pp. 260–270 (cit. on pp. 1, 2).
[3] A. Kumar and B. Starly, “Fabner: Information ex-
traction from manufacturing process science domain
literature using named entity recognition,”Journal of
Intelligent Manufacturing, vol. 33, no. 8, pp. 2393–
2407, 2022, © 2021, The Author(s), under exclusive
licence to Springer Science+Business Media, LLC.
DOI:10.1007/s10845-021-01807-x(cit. on
pp. 1, 2, 5, 6).
[4] R. Armingaud and R. Besanc ¸on,Manufactubert: Ef-
ficient continual pretraining for manufacturing, 2025.
arXiv:2511.05135 [cs.CL]. [Online]. Available:
https : / / arxiv . org / abs / 2511 . 05135
(cit. on pp. 1, 2).
[5] S. Song, F. Shen, and J. Zhao, “Ropda: Robust
prompt-based data augmentation for low-resource
named entity recognition,” inProceedings of the AAAI
conference on artificial intelligence, vol. 38, 2024,
pp. 19 017–19 025 (cit. on p. 2).
[6] S. Wang et al., “Gpt-ner: Named entity recognition via
large language models,” inFindings of the association
for computational linguistics: NAACL 2025, 2025,
pp. 4257–4275 (cit. on p. 2).
[7] O. Sainz, I. Garc ´ıa-Ferrero, R. Agerri, O. L. de La-
calle, G. Rigau, and E. Agirre, “Gollie: Annotation
guidelines improve zero-shot information-extraction,”
arXiv preprint arXiv:2310.03668, 2023 (cit. on pp. 2,
6).
[8] D. Y .-B. Wang, Z. Shen, S. S. Mishra, Z. Xu, Y .
Teng, and H. Ding,Slot: Structuring the output of
large language models, 2025. arXiv:2505.04016
[cs.CL](cit. on p. 2).
[9] Y . Gao, T. Sheng, Y . Xiang, Y . Xiong, H. Wang, and J.
Zhang, “Chat-rec: Towards interactive and explainable
llms-augmented recommender system,”arXiv preprint
arXiv:2303.14524, 2023 (cit. on p. 2).
[10] W. Yu et al., “Generate rather than retrieve: Large
language models are strong context generators,”arXiv
preprint arXiv:2209.10063, 2022 (cit. on p. 2).
[11] Y . Gao et al.,Retrieval-augmented generation for
large language models: A survey, 2024. arXiv:2312.
10997 [cs.CL]. [Online]. Available:https://
arxiv.org/abs/2312.10997(cit. on p. 2).
[12] P. Lewis et al., “Retrieval-augmented generation for
knowledge-intensive nlp tasks,”Advances in neural in-
formation processing systems, vol. 33, pp. 9459–9474,
2020 (cit. on p. 2).[13] T. B. Brown et al., “Language models are few-shot
learners,” inAdvances in Neural Information Process-
ing Systems (NeurIPS), 2020 (cit. on p. 3).
[14] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and
M. Zhou, “Minilm: Deep self-attention distillation
for task-agnostic compression of pre-trained trans-
formers,” inProceedings of the 34th International
Conference on Neural Information Processing Sys-
tems, ser. NIPS ’20, Vancouver, BC, Canada: Curran
Associates Inc., 2020,ISBN: 9781713829546 (cit. on
pp. 3, 4, 7).
[15] S. Robertson and H. Zaragoza,The probabilistic rele-
vance framework: BM25 and beyond. Now Publishers
Inc, 2009, vol. 4 (cit. on pp. 3, 4).
[16] G. V . Cormack, C. L. Clarke, and S. Buettcher,
“Reciprocal rank fusion outperforms condorcet and
individual rank learning methods,” inProceedings of
the 32nd international ACM SIGIR conference on
Research and development in information retrieval,
2009, pp. 758–759 (cit. on pp. 3, 4).
[17] B. Roziere et al., “Code llama: Open foundation
models for code,”arXiv preprint arXiv:2308.12950,
2023 (cit. on pp. 3, 5, 6).
[18] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever,
et al.,Improving language understanding by genera-
tive pre-training, 2018. [Online]. Available:https:
//www.mikecaptain.com/resources/pdf/
GPT-1.pdf(cit. on p. 3).
[19] Elastic NV,Elasticsearch, 2026. [Online]. Avail-
able:https : / / www . elastic . co /
elasticsearch/(cit. on p. 4).
[20] M. Honnibal and I. Montani, “Spacy 2: Natural lan-
guage understanding with bloom embeddings, convo-
lutional neural networks and incremental parsing,”To
appear, 2017 (cit. on p. 4).
[21] J. Leskovec and A. Krevl,SNAP Datasets: Stanford
large network dataset collection,http://snap.
stanford.edu/data, Jun. 2014 (cit. on p. 4).
[22] A. Grattafiori et al., “The llama 3 herd of models,”
arXiv preprint arXiv:2407.21783, 2024 (cit. on p. 4).
[23] E. J. Hu et al., “Lora: Low-rank adaptation of large
language models,”CoRR, vol. abs/2106.09685, 2021.
arXiv:2106.09685. [Online]. Available:https:
//arxiv.org/abs/2106.09685(cit. on p. 6).
[24] A. D. Team,Axolotl: Easy lora fine-tuning for llms,
2023. [Online]. Available:https : / / github .
com/ironmussa/axolotl(cit. on p. 6).
[25] K. Enevoldsen et al.,Mmteb: Massive multilingual text
embedding benchmark, 2025. arXiv:2502.13595
[cs.CL]. [Online]. Available:https://arxiv.
org/abs/2502.13595(cit. on p. 7).
[26] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian,
and Z. Liu,Bge m3-embedding: Multi-lingual,
multi-functionality, multi-granularity text embeddings
through self-knowledge distillation, 2024. arXiv:
2402.03216 [cs.CL](cit. on p. 7).