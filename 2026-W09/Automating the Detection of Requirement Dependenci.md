# Automating the Detection of Requirement Dependencies Using Large Language Models

**Authors**: Ikram Darif, Feifei Niu, Manel Abdellatif, Lionel C. Briand, Ramesh S., Arun Adiththan

**Published**: 2026-02-25 22:33:27

**PDF URL**: [https://arxiv.org/pdf/2602.22456v1](https://arxiv.org/pdf/2602.22456v1)

## Abstract
Requirements are inherently interconnected through various types of dependencies. Identifying these dependencies is essential, as they underpin critical decisions and influence a range of activities throughout software development. However, this task is challenging, particularly in modern software systems, given the high volume of complex, coupled requirements. These challenges are further exacerbated by the ambiguity of Natural Language (NL) requirements and their constant change. Consequently, requirement dependency detection is often overlooked or performed manually. Large Language Models (LLMs) exhibit strong capabilities in NL processing, presenting a promising avenue for requirement-related tasks. While they have shown to enhance various requirements engineering tasks, their effectiveness in identifying requirement dependencies remains unexplored. In this paper, we introduce LEREDD, an LLM-based approach for automated detection of requirement dependencies that leverages Retrieval-Augmented Generation (RAG) and In-Context Learning (ICL). It is designed to identify diverse dependency types directly from NL requirements. We empirically evaluate LEREDD against two state-of-the-art baselines. The results show that LEREDD provides highly accurate classification of dependent and non-dependent requirements, achieving an accuracy of 0.93, and an F1 score of 0.84, with the latter averaging 0.96 for non-dependent cases. LEREDD outperforms zero-shot LLMs and baselines, particularly in detecting fine-grained dependency types, where it yields average relative gains of 94.87% and 105.41% in F1 scores for the Requires dependency over the baselines. We also provide an annotated dataset of requirement dependencies encompassing 813 requirement pairs across three distinct systems to support reproducibility and future research.

## Full Text


<!-- PDF content starts -->

Automating the Detection of Requirement
Dependencies Using Large Language Models
Ikram Darif∗, Feifei Niu∗, Manel Abdellatif†, Lionel C. Briand∗‡, Ramesh S§and Arun Adiththan§
∗University of Ottawa, Ottawa, Canada.
†École de technologie supérieure, Montreal, Canada.
‡Research Ireland Lero Centre for Software Research, University of Limerick, Ireland.
§General Motors, Detroit, Michigan, United States.
Email: idarif@uottawa.ca; fniu2@uottawa.ca; manel.abdellatif@etsmtl.ca; lbriand@uottawa.ca;
ramesh.s@gm.com; arun.adiththan@gm.com.
Abstract—Requirements are inherently interconnected through
various types of dependencies. Identifying these dependencies is
essential, as they underpin critical decisions and influence a range
of activities throughout software development. However, this task
is challenging, particularly in modern software systems, given the
high volume of complex, coupled requirements. These challenges
are further exacerbated by the ambiguity of Natural Language
(NL) requirements and their constant change. Consequently,
requirement dependency detection is often overlooked or per-
formed manually. Large Language Models (LLMs) exhibit strong
capabilities in NL processing, presenting a promising avenue for
requirement-related tasks. While they have shown to enhance
various requirements engineering tasks, their effectiveness in
identifying requirement dependencies remains unexplored. In
this paper, we introduce LEREDD, an LLM-based approach
for automated detection of requirement dependencies that lever-
ages Retrieval-Augmented Generation (RAG) and In-Context
Learning (ICL). It is designed to identify diverse dependency
types directly from NL requirements. We empirically evaluate
LEREDD against two state-of-the-art baselines. The results show
that LEREDD provides highly accurate classification of depen-
dent and non-dependent requirements, achieving an accuracy of
0.93, and anF 1score of 0.84, with the latter averaging 0.96 for
non-dependent cases. LEREDD outperforms zero-shot LLMs and
baselines, particularly in detecting fine-grained dependency types,
where it yields average relative gains of 94.87% and 105.41% in
F1scores for theRequiresdependency over the baselines. We
also provide an annotated dataset of requirement dependencies
encompassing 813 requirement pairs across three distinct systems
to support reproducibility and future research.
Index Terms—Requirements Engineering, Requirement De-
pendency Detection, Large Language Models.
I. INTRODUCTION
Modern software systems are becoming increasingly com-
plex, driven by the growing and advanced demands of stake-
holders and users. This complexity is directly reflected in their
Requirements Engineering (RE) processes, in which require-
ments are systematically identified, documented, analyzed, and
managed to accurately capture stakeholders’ needs [1]. As the
foundational artifacts of the software development life cycle,
requirements are arguably the most critical artifacts for project
success and the quality of the final product [2].
However, modern software systems are characterized by
large numbers of requirements, high requirement complexity,
and substantial inter-dependencies among them. Requirementsare not independent entities. They are inherently intercon-
nected through dependencies of varying natures [3]. Identify-
ing, classifying, and managing inter-requirement dependencies
is crucial, as they underpin critical software development
decisions and influence activities throughout the software
development life cycle [3]. For instance, they are essential for
conducting accurate impact analysis during system changes
and ensuring consistency by identifying conflicting require-
ments. Ignoring such dependencies not only has a detrimental
effect on project success but also compromises release quality
and leads to substantial rework [4], [5]. However, detecting re-
quirement dependencies remains error-prone, time-consuming,
and cognitively challenging [3]. These challenges are largely
attributable to the ambiguity introduced by the predominant
use of Natural Language (NL) to author requirements, as well
as to the reliance on manual effort for detection.
Existing approaches for detecting requirements dependen-
cies exhibit various limitations. Retrieval-based approaches
are restricted to pairwise classification and rely on fixed
representations, failing to account for domain-specific context.
Knowledge-based approaches use ontologies or graphs to
represent domain knowledge, but require substantial effort for
development and maintenance. Finally, ML-based approaches
require large training datasets and struggle with the class
imbalance inherent in dependency detection, as independent
requirements typically far outnumber dependent ones.
Large Language Models (LLMs) have significantly revolu-
tionized artificial intelligence and its applications, emerging as
advanced models with billions of parameters trained on vast
corpora. Their ability to be fine-tuned for specialized applica-
tions without exhaustive task-specific training has contributed
to their success [6], [7]. They are particularly renowned for
their capabilities in NL processing, reasoning, and generation,
rendering them especially relevant for NL RE tasks [8], [9].
While LLMs have been successfully applied to various RE-
related tasks, notably requirements elicitation and classifi-
cation, their utility for automated requirement dependency
detection remains largely unexplored [9], [10].
Driven by the NL processing capabilities of LLMs, we pro-
pose LEREDD: LLM-Enabled REquirement Dependency De-
tection, an automated approach for detecting various types ofarXiv:2602.22456v1  [cs.SE]  25 Feb 2026

direct dependencies between NL requirement pairs. LEREDD
leverages Retrieval-Augmented Generation (RAG) to extract
domain-specific context from the Software Requirement Spec-
ification (SRS) document, and utilizes In-Context Learning
(ICL) to dynamically retrieve relevant examples for each
dependency type and the no-dependency case. The retrieved
information provides a comprehensive, domain-specific con-
text that effectively guides the LLM during detection. For each
requirement pair, LEREDD generates: (1) a prediction speci-
fying the dependency type (e.g.,RequiresandImplements),
or the absence of a dependency, (2) the rationale behind
the prediction, and (3) a confidence score. We investigate
four LLMs, both proprietary and open-source, for detecting
requirement dependencies. We empirically evaluate LEREDD
on a set of 813 manually labeled requirement pairs across
three distinct systems, comparing its performance with two
state-of-the-art (SOTA) baselines.
The results show that LEREDD achieves highly accurate
classification of non-dependent requirements, with anF 1score
of 96%. This is particularly significant because non-dependent
pairs typically constitute the majority of requirement rela-
tionships in real-world SRS documents. By reliably filter-
ing these cases, LEREDD substantially reduces the manual
effort and time required for dependency analysis. LEREDD
also consistently outperforms the baselines, particularly in
detecting fine-grained dependency types. For example, it yields
average relative gains of 94.87% and 105.41% inF 1scores
for theRequiresdependency over baselines. Overall, LEREDD
achieves an average accuracy of 92.66% and anF 1score of
84.33%, across all dependency classes and the evaluated sys-
tems. Moreover, LEREDD effectively addresses the inherent
class imbalance among dependency types, achieving higher
accuracy andF 1scores than the baselines. It also demonstrates
strong robustness in cross-system evaluations, maintaining
superior F1 performance even when examples are retrieved
from different systems. This capability is particularly valuable
in real-world settings, where annotated data from the target
system is often unavailable. We release the annotated corpus
used in this study as an open-source dataset for benchmarking
and training, helping to mitigate the scarcity of public datasets.
The remainder of the paper is organized as follows. Sec-
tion II reviews related work on requirement dependency detec-
tion. Section III outlines the LEREDD framework. Section IV
and Section V describe the empirical setup and results of
our evaluations, respectively. Section VI provides an in-depth
discussion of the findings while Section VII examines the
threats to validity. Finally, Section VIII concludes the paper
and outlines future work.
II. RELATEDWORK
Several approaches have been proposed to support the
detection of requirement dependencies [6], [11]–[18]. Based
on the detection logic, such approaches can be classified into
four primary categories [3], [6]: information retrieval-based
approaches [11], [19]–[22], knowledge-based approaches [5],[12]–[14], [23]–[25], ML-based approaches [4], [6], [15],
[26]–[30], and LLM-based approaches [16]–[18].
Information Retrieval-based Approachesrely on vector
representations to build lexical- or semantic-based vector space
models for the requirements corpus [3]. Lexical approaches
identify dependencies using statistical methods, with TF-IDF
(Term Frequency-Inverse Document Frequency) commonly
applied to assess term importance relative to the corpus [3].
Lexical methods are often paired with semantic methods, such
as Latent Semantic Analysis (LSA), which generate embed-
dings and apply similarity measures to identify dependencies.
Liet al.[20] utilized TF-IDF paired with cosine similarity as
a baseline for requirement traceability detection. For binary
requirement dependency detection, Sameret al.[21] applied
a combination of TF-IDF and LSA, while Guanet al.[22]
implemented optimized variations of TF-IDF. Vector-based
approaches are limited to pairwise classification and rely on
fixed representations that fail to account for domain-specific
terminology. Furthermore, because requirements often contain
inconsistent terminology and dependencies are derived from
the underlying system architecture, detection requires inferen-
tial reasoning that these approaches lack.
Knowledge-based Approachesutilize structured repre-
sentations of domain knowledge to infer requirement de-
pendencies, which can be categorized into: (1) graph-based
approaches [12]–[14], [23], [24], which leverage structural
relationships within the SRS, and (2) ontology-based ap-
proaches [5], [25], which utilize formal ontologies for domain
representation. Priyadiet al.[12], Asyrofiet al.[13], and
Mokammelet al.[24] applied Natural Language Processing
(NLP) to extract requirement dependencies, such as “similar”
and “elaborate”, and model the requirement dependency graph.
Beyond simple linking, Guoet al.[14] applied finer semantic
analysis to graph structures to automatically detect “conflict”
dependencies. Similarly, Schlutteret al.[23] proposed an
NLP pipeline that maps requirements into a semantic relation
graph from which dependencies are identified. OpenReq-
DD automatically detects requirements dependencies by us-
ing ontologies to capture domain-specific term relationships,
combined with NLP and ML techniques [25]. Building on
this, Deshpandeet al.[5] compared OpenReq-DD with an
active learning approach for detecting “requires” and “refines”
dependencies, proposing a hybrid framework that combines
both approaches. While ontology-based approaches support
more effective domain modeling than graph-based approaches,
building and maintaining an ontology is labor-intensive, time-
consuming, and heavily dependent on expert knowledge [3].
ML-based Approachesutilize statistical algorithms to learn
patterns from training data for dependency detection [6].
Gräßleret al.[15] used Fine tuned-BERT models to identify
“refines” and “requires” dependencies, while Fischbachet
al.[26], [27] applied them to detect requirement causality.
Deshpandeet al.[4] and Ataset al.[28] relied on supervised
learning to classify dependencies and to identify “requires” de-
pendencies, respectively. Guanet al.[29] introduced an active
learning algorithm that iteratively selects relevant requirements

for manual annotation, which are continuously used to refine
the model. Abebaet al.[30] specifically targeted “conflict”
dependencies in non-functional requirements, demonstrating
that Bi-LSTM (bi-directional long short-term memory) with
pre-trained word2vec outperformed other classifiers.
Despite their potential, ML-based approaches are primarily
restricted to pattern recognition and lack the inference rea-
soning necessary for dependency detection. They rely heavily
on large, high-quality annotated data, which is scarce in real-
world contexts. Furthermore, traditional ML-based approaches
often struggle with data imbalance, as independent require-
ments typically far outnumber dependent ones.
LLM-based Approachesfor requirement dependency de-
tection are limited. Gärtneret al.[16] introduced ALICE,
which combines LLMs with formal logic to identify require-
ment contradictions. Using a contradiction taxonomy and a de-
cision tree, ALICE outperformed LLM-only approaches [16].
Similarly, Almoqrenet al.[17] integrated Knowledge Graphs
with LLMs to identify requirement dependencies from mobile
app reviews. Using BERT in an LLM-driven active learning
loop, they captured semantic and structural relations, achieving
high precision [17]. For trace links recovery, Niuet al.[18]
introduced TVR, an approach that leverages RAG-enhanced
LLMs to validate and recover traceability between stakeholder
and system requirements.
LLMs are particularly relevant for requirement dependency
detection, as they leverage inferential reasoning while facilitat-
ing automation. In contrast to ML approaches, LLMs reduce
or eliminate annotation overhead and are less sensitive to data
imbalance. Despite these advancements, LLM-based require-
ment dependency detection remains scarce and often confined
to specific dependency types (e.g., contradictions [16]). Re-
cent literature explicitly advocates for further investigation of
LLMs in this context [3]. Our research addresses this critical
gap by proposing an LLM-based approach that automates this
task while detecting a wide spectrum of dependency types.
III. LEREDD: LLM-ENABLEDREQUIREMENT
DEPENDENCYDETECTIONAPPROACH
A. Overview
LEREDD is an LLM-based dependency detection approach
that automatically identifies direct dependencies between pairs
of requirements. LEREDD takes as input an SRS document
(containing a list of requirements) and a dataset of annotated
requirement pairs, each annotated with their dependency type.
As output, it generates a prediction for each requirement pair
extracted from the SRS, specifying the dependency type (or no
dependency), along with a confidence score and a rationale.
The overall framework of LEREDD is illustrated in Fig-
ure 1. Given a set ofnNL requirements extracted from
the SRS document, a list ofn(n−1)
2requirement pairings is
generated, representing the complete set of unique requirement
pairs. These pairs are processed by a two-phase pipeline com-
prising aknowledge retrievalphase followed by adependency
inferencephase. Theknowledge retrievalphase applies a dual-
strategy process to facilitate in-context learning: (1)contextualretrieval, and (2)dynamic examples retrieval. Thecontextual
retrievalemploys RAG to extract domain-specific information
from the SRS document, serving as domain-specific context
within the prompt. Thedynamic examples retrievalidentifies
and retrieves similar requirement pairs for all dependency
types from the annotated dataset, serving as examples within
the prompt. The extracted data from theknowledge retrieval
phase informs thedependency inferencephase, in which an
LLM assesses the dependency links between the requirement
pair using the augmented prompt. To enhance the model’s rea-
soning and rigor, the prompt requires the LLM to perform self-
reflection, including an explanatory rationale and a confidence
score on a 5-point Likert scale for each prediction.
Knowledge 
Retrieval
 
Dependency 
I
nference
        
Requirement
 
pairs
     
Prediction
        
Confidence          
score
      
Rationale
       
Contextual  
      
Retrieval
        
Dynamic
 
Examples 
Retrieval
Domain 
context
Relevant 
examples 
SRS 
document
Annotated 
dataset
Fig. 1: LEREDD Framework.
B. Knowledge Retrieval
Theknowledge retrievalphase involves extracting contex-
tual information required for the detection task in two stages:
contextual retrievalanddynamic example retrieval.
Contextual retrievalemploys RAG, a technique that en-
ables LLMs to produce context-aware outputs by integrat-
ing domain-specific knowledge retrieved from relevant docu-
ments [8]. RAG is highly relevant to RE, as existing research
suggests that its limited adoption constitutes a missed opportu-
nity to leverage LLMs’ full reasoning and retrieval capabilities
for RE tasks [9]. In the context of dependency detection, RAG
supports the identification of structural relationships between
system components, which can subsequently be mapped as
dependencies between their corresponding requirements. For
instance, given the requirements:“The system shall include
the BCS”and“The system shall always stop the vehicle to
prevent collision with objects during the parking maneuver”,
a dependency is only apparent when the context–defining the
BCS as the subsystem responsible for braking–is provided.
Within the LEREDD framework, RAG is applied to the SRS
document, focusing on the system description sections and the
requirements list. The requirements list is incorporated into
the context pool because it provides fundamental information
about system architecture and component interactions. A fixed-
size chunking strategy is employed, wherein thek= 10
most semantically similar chunks are retrieved and provided
as domain context to the model. This configuration mitigates
noise and provides the LLM with focused context.
Complementarily,dynamic examples retrievalis employed
to facilitate ICL, a paradigm that enhances LLM predictions
by augmenting the input prompt with task-specific exam-
ples [31]. Building upon [18], [32], [33], LEREDD applies

a dynamic retrieval process to select semantically similar
examples from the annotated dataset, providing the model with
context-specific examples tailored to the target requirement
pair. Examples are provided for each dependency type and the
no-dependency case to provide a comprehensive context for
the classification task. To support example retrieval, LEREDD
leverages theSBERTmodel to generate embeddings for in-
dividual requirements in both the target pair(R 1, R2)and
the candidate example pair(R a, Rb). Semantic similaritysim
is then computed using the Euclidean similarity metric and
aggregated according to the following formula:
Score max_avg =max(sim(R 1,Ra),sim(R 1,Rb))+max(sim(R 2,Ra),sim(R 2,Rb))
2
(1)
This formula computes the mean of the maximum similarity
scores associated with each requirement in the target pair.
Subsequently, the topk= 4most similar examples to the
requirement pair to be annotated(R 1, R2)are retrieved for
each dependency type. Overall, the retrieval stage produces
domain-specific context and relevant examples, which are then
used as inputs to the subsequent dependency inference stage.
C. Dependency Inference
The dependency inference stage is the core of the LEREDD
framework, in which the LLM leverages contextual informa-
tion generated in the retrieval stage to identify dependencies
between requirement pairs. For this task, theGPT-4.1model
is employed as it demonstrated the best performance in our
evaluations (as reported in section V). The LLM prompt is
augmented with the top two retrieved context chunks and the
top four most similar examples for each dependency type.
Beyond contextual information, the prompt includes formal
definitions of each dependency type to ensure conceptual
clarity and to facilitate clear distinctions among types.
While LEREDD is extensible to accommodate diverse de-
pendency types, the current prompt supports seven types:Re-
quires,Implements,Conflicts,Contradicts,Details,Is similar,
andIs a variant. These types were derived from a SOTA
classification [3], except forImplements, which was introduced
specifically to address the needs of our industrial partner.
Regarding the definitions, some were reused from [3], while
others were refined to reduce ambiguity and improve clarity
for the LLM (e.g.,DetailsandRequires). In the following, we
report the definitions utilized in the prompt:
Requires: if the fulfillment of one requirement is a prereq-
uisite to the fulfillment of the other requirement.
Implements: if one is a higher-level requirement (e.g., a
system or subsystem level requirement) that is fulfilled by the
other lower-level requirement (e.g., a subsystem or component
level requirement).
Conflicts: if the fulfillment of one requirement restricts the
fulfillment of the other requirement.
Contradicts: if the two requirements are mutually exclusive,
then the fulfillment of one requirement violates the other.
Details: if both requirements describe the same action under
the same condition, and one requirement provides additional
details specifically regarding the shared action.Is similar: if one requirement replicates partially or totally
the content of the other requirement, resulting in redundancy.
Is a variant: if one requirement serves as an alternative to
the other.
Figure 2 illustrates the prompt employed for inference.
The prompt is structured into three primary segments: (1)
contextual scope, (2)input data, and (3)instructions. The first
segment instructs the model to adopt an expert requirements
engineer persona, specifies the domain and system from which
the requirements were derived, and defines the detection task.
Such information serves to ground the model, providing a
preliminary orientation for the detection. The second segment
includes the four inputs required for the task: the requirements
to be analyzed, the formal definitions of dependency types,
and the context and examples generated during theknowledge
retrievalstage. These inputs provide precise, granular infor-
mation, thereby facilitating accurate dependency detection.
You are an expert requirements engineer from the<domain of interest>. You will
be provided with a pair of requirements extracted from the software requirements
specification for<system name>.
Given the following requirement dependency types definitions, examples, and
context, your task is to analyze the pair of requirements and determine if a direct or
indirect dependency exists between them.
#Requirements to analyze:
Requirement A:<first requirement>
Requirement B:<second requirement>
#Dependency Definitions:
<Definitions of dependency types>
#Examples:
<Retrieved examples>
#Context:
<Retrieved domain-specific context>
#Instructions
- If a direct or indirect dependency exists between a pair, you should annotate it with
the type of dependency.
- If it does not fall into one of the above types of dependency, annotate it with
“No_dependency”.
- Explain the rationale behind your annotation.
- Provide a confidence score for the annotation. The score should range from 0 to 5,
with 0 indicating no confidence and 5 indicating the highest confidence.
- **The output MUST be structured using these exact labels, each on a new line:**
**Dependency_Status: [TYPE]**
**Rationale: [EXPLANATION]**
**Confidence Score: [SCORE]**
Fig. 2: Dependency Inference Prompt.
The final segment provides specific instructions for the
detection, the expected outputs, and their format. Beyond the
primary prediction (i.e., the dependency status and type), the
model is instructed to generate a rationale justifying the pre-
diction, and a confidence score, measured on a 5-point Likert
scale, reflecting its certainty in the generated prediction. These
additional outputs are designed to promote self-reflection in
the model, thereby reducing the risk of hallucinations. In
particular, prompting LLMs to generate rationales enhances
their reasoning capabilities while promoting transparency and
credibility [34], [35]. Moreover, providing confidence scores
improves the interpretability of the generated outputs [36].
Therefore, for each requirement pair, LEREDD generates
three outputs: (1) a prediction classifying the dependency type
or indicating the absence of a dependency, (2) a confidence
score, and (3) a rationale justifying the underlying reasoning.

IV. EMPIRICALEVALUATION
A. Research Questions
RQ.1: Which SOTA LLM is more effective for requirement
dependency detection?This RQ aims to empirically evaluate
the performance of SOTA LLMs on requirement dependency
detection under a zero-shot prompting setting, and to identify
the best-performing model.
RQ.2: What is the best prompting strategy for require-
ment dependency detection?This RQ aims to systematically
evaluate the performance of different prompting strategies,
including few-shot and RAG, for requirement dependency
detection, compare their performance to zero-shot prompting,
and identify the best-performing technique.
RQ.3: How does LEREDD compare to baseline ap-
proaches for intra-dataset dependency detection?This RQ
aims to evaluate and compare LEREDD with two SOTA
baselines, using requirements from the same dataset (1) to
select dynamic examples required by LEREDD and (2) to train
and test the selected baselines.
RQ.4: How does LEREDD compare to baseline ap-
proaches for cross-dataset dependency detection?This RQ
evaluates LEREDD against the two baselines in a cross-dataset
setting, where training and evaluation are performed on differ-
ent datasets. Specifically, we consider two datasetsD 1andD 2.
For LEREDD, example retrieval is performed using annotated
requirement pairs fromD 1, while the requirement pairs to
be classified are drawn fromD 2. In contrast, the baseline
models are trained onD 1and evaluated onD 2. This setup
reflects realistic deployment conditions, where annotated data
are typically available only for previously studied systems,
while the detection is performed on new systems.
B. Dataset Construction
1) Data Collection:To ensure the validity of our exper-
imental data, it must satisfy two criteria: (1) the require-
ments should be specified in NL and manually annotated for
direct dependencies to ensure their correctness, and (2) the
requirements should be recent to ensure that they were not
used for training LLMs. To the best of our knowledge, no
publicly available dataset meets both criteria. To mitigate these
limitations, we collected the SRS documents that were pub-
licly available on Michigan State University’s Requirements
Engineering course website1. These documents were produced
within a structured requirements engineering curriculum in
collaboration with industrial partners who guided their de-
velopment and validation. They underwent multiple rounds
of feedback and revision, resulting in relatively high-quality
specifications that accurately represent industrial practice in
regulated domains and are suitable for empirical analysis.
From the available materials, we selected three SRS doc-
uments corresponding to different automotive systems, each
representing a distinct functionality: Traffic Jam Assist (TJA),
Automated Parking Assist (APA), and Adaptive Driving Beam
(ADB). These systems are widely studied in the automotive
1https://www.cse.msu.edu/~cse435/#cinfoTABLE I: Distribution of Requirement Dependency Types
Dependency Type ADB TJA APA Total
# Requirements 413 200 200 813
Conflicts 14 - 4 18
Details 18 2 1 21
Implements 17 10 3 30
Is similar 3 1 3 7
Requires 32 18 45 95
No Dependency 329 169 144 642
domain and exhibit nontrivial interactions among require-
ments, making them appropriate for dependency analysis.
We curate the dataset by extracting NL requirements from
each SRS and omitting non-requirement sections, such as
introductions, abbreviations, and supplementary descriptions.
This process yields collections of 40, 25, and 50 requirement
statements for the ADB, TJA, and APA systems, respectively.
2) Ground Truth:To construct the ground truth for re-
quirement dependency analysis, two authors (each with more
than five years of experience in requirements engineering)
independently annotated the requirement pairs within each
system, following a structured, multi-step annotation process.
We integrated the set of requirement dependency types and
their formal definitions from Section III-C and augmented
them with representative examples, thereby forming the an-
notation guidelines. The two annotators then independently
analyzed and annotated requirement pairs within each system
in accordance with these guidelines. Each requirement pair
was labeled with a single dependency type or marked as having
no dependency. Because the annotation process is manual,
it remains both costly and time-consuming, even for SRS
documents containing a relatively small number of require-
ments. For instance, the ABD SRS comprises 40 requirements,
yielding 780 possible unique requirement pairs. Exhaustively
annotating all such pairs is therefore impractical.
To prioritize annotation effort on more likely dependency
cases, we embedded each requirement using all-MiniLM-L6-
v2, a pretrained BERT-based sentence encoder, and computed
the cosine similarity between all requirement pairs. The pairs
were then ranked in descending order of similarity. Manual
annotation was conducted after this ranking, beginning with
the most semantically similar requirement pairs. For the ADB
system, annotation was stopped after 413 requirement pairs,
as the number of dependent pairs reached a plateau. For
the TJA and APA systems, we annotated the top 200 most
similar requirement pairs for each system. This resulted in
813 annotated requirement pairs across the three systems.
After independent annotation, we measured inter-annotator
agreement using Cohen’s kappa score to assess consistency
beyond chance. The overall Cohen’s kappa is 0.43, indicating
moderate agreement, reflecting the task’s inherent subjectivity
and complexity. All disagreements were resolved by consen-
sus, resulting in a unified set of annotations that serves as the
gold standard for evaluation. The distributions of dependency
types across systems are summarized in Table I. The annotated
dataset is provided in our replication package.

C. Baselines
1) Selection Criteria:To evaluate the LEREDD framework,
we selected baseline approaches according to four main cri-
teria. First, the baselines should span different categories of
approaches to ensure diversity and to support rigorous, rep-
resentative comparisons with the broader research landscape.
Second, these approaches should be among the most prevalent
and well-established within their respective categories. This
criterion facilitates fair comparison across categories. Third,
the approaches should be applicable to different systems.
This criterion is essential because our evaluation encompasses
multiple systems. Finally, baselines should either provide
open-source implementations or comprehensive configuration
guidelines to ensure reproducibility of experimental results.
2) Selected Baselines:Based on the first and second cri-
teria, we considered three approaches: (1) a retrieval-based
approach usingTF-IDF & LSA, (2) a knowledge-based ap-
proach usingontologies, and (3) an ML-based approach using
fine-tuned BERT. Existing LLM-based methods were excluded
because they target specific scenarios or minority dependency
types underrepresented in our datasets. The selected ap-
proaches ensure diversity and represent prevalent approaches
within their respective categories. TF-IDF is a fundamental
lexical method, and its integration with LSA provides a
strong baseline for lexical and semantic retrieval [3], [21].
Ontologies provide more effective domain modeling compared
to graphs [3]. Fine-tuned BERT is a prominent ML approach
recognized for its accuracy in multi-class classification tasks,
outperforming conventional approaches [15], [37], [38]. Based
on the third criterion, the ontology-based approach was ex-
cluded. While it offers high precision through domain-specific
representation, it was omitted due to the high cost of ontology
development and limited scalability. Finally, based on the
fourth criterion, we retained theTF-IDF & LSAandfine-tuned
BERTbaselines, as the former is supported by standard Python
libraries and the latter by open-source implementations.
To evaluate LEREDD against the selected baselines, we
implementedTF-IDF & LSAfollowing the configuration de-
scribed by Sameret al.[21], where it serves as a recommender
forRequiresdependencies. For fine-tuned BERT, we adhered
to the configuration guidelines described by Gräßleret al.[15].
Their approach enables the detection ofRequiresandDetails
dependencies and uses oversampling and class weighting to
address class imbalance. To ensure a fair comparison, hyper-
parameters for both baselines were systematically tuned on the
ADB system dataset to identify their optimal configurations.
For LEREDD and fine-tuned BERT, we opted for an 80%/20%
split of the ground truth for intra-database experiments. For
BERT, the 80% was used for training, consistent with the
original approach [15], while for LEREDD, it served as the
example pool for few-shot retrieval.
V. EMPIRICALRESULTS
1) RQ1. Performance of SOTA LLMs for requirements de-
pendency detection:To evaluate the effectiveness of SOTA
LLMs in identifying requirement dependencies, we select fourrepresentative models that have demonstrated strong perfor-
mance in RE research [9], [39]:GPT-4.1,Llama 3.1,Gemma
20B, andMistral 7B, covering proprietary and open-source
models. All models were evaluated using a zero-shot setting,
without considering any in-context examples or task-specific
knowledge. In all experiments reported throughout this section,
the temperature was set to 0 to ensure consistent, reproducible
results. Table II reports the zero-shot performance of the four
LLMs across the three automotive systems. We should note
that we report the macro-average when calculating the metrics
(i.e., precision, recall, andF 1-score). This ensures that all de-
pendency classes are weighted equally, providing meaningful
results that are not biased by skewed class distributions.
As Table II shows, GPT-4.1 consistently outperforms the
evaluated models. It achieves the highestF 1scores across all
systems (ADB: 0.40, TJA: 0.29, APA: 0.47), with an average
of 0.39, indicating stronger generalization in zero-shot de-
pendency classification than other models. In contrast, Llama
3.1, Gemma 20B, and Mistral 7B exhibit lower and more
fluctuating performance, indicating that dependency detection
remains challenging for most open-source models. Mistral
achieves relatively competitive accuracy on ADB (0.77) and
TJA (0.84). However, itsF 1scores remain limited due to poor
performance on minority dependency types.
Across all models, theNo Dependencyclass consistently
achieves the highest performance. Most models achieve high
results for this class, particularly GPT-4.1, which obtainsF 1
Scores between 0.89 and 0.9 across systems. This result
suggests that distinguishing unrelated requirement pairs is
relatively easier for LLMs, likely because this class exhibits
clearer semantic separation. In contrast, models struggle with
fine-grained dependency types, such asImplements, frequently
obtaining near-zeroF 1scores. This suggests that such depen-
dencies require deeper semantic understanding and domain
knowledge, posing challenges for zero-shot LLMs without
task-specific guidance. Overall, GPT-4.1’s superior perfor-
mance led to its selection as the core engine for LEREDD.
Answer to RQ1:While GPT-4.1 tends to outperform
other models, zero-shot models typically over-predict
dependencies. Zero-shot GPT-4.1 can reliably identify
non-dependent requirement pairs, achieving an average
F1score of 0.87 for theNo dependencyclass across the
three systems. However, the model struggles with fine-
grained dependency types, yielding an average overallF 1
score of only 0.39, highlighting the limitations of zero-
shot dependency understanding.
2) RQ2. Comparative analysis of the performance of
prompting strategies for dependency detection:To address this
RQ, we evaluated the performance of various configurations
for dynamic few-shot prompting and RAG using GPT-4.1,
which was identified as the best-performing LLM based on
RQ1. We conducted a comprehensive suite of 216 experi-
ments across three automotive systems, covering all parameter

TABLE II: Experimental Results of Zero-Shot LLMs
LLMDependency ADB TJA APA
Type Acc P R F1 Acc P R F1 Acc P R F1
GPTNo Dep
0.770.95 0.86 0.90
0.800.95 0.86 0.90
0.810.98 0.81 0.89
Requires 0.31 0.72 0.43 0.32 0.67 0.43 0.56 0.93 0.70
Implements 0.50 0.29 0.37 0.60 0.30 0.40 0.00 0.00 0.00
Conflicts 0.43 0.43 0.43 N/A N/A N/A 1.00 0.75 0.86
Details 0.33 0.22 0.27 0.00 0.00 0.00 0.00 0.00 0.00
Is similar 0.00 0.00 0.00 0.00 0.00 0.00 0.50 0.33 0.40
Macro avg 0.42 0.42 0.40 0.310.300.29 0.51 0.47 0.47
LlamaNo Dep
0.550.96 0.57 0.72
0.720.96 0.76 0.85
0.430.90 0.31 0.46
Requires 0.15 0.91 0.25 0.25 0.83 0.38 0.28 0.89 0.43
Implements 1.00 0.12 0.21 0.00 0.00 0.00 0.00 0.00 0.00
Conflicts 0.39 0.50 0.44 N/A N/A N/A 0.14 0.25 0.18
Details 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Is similar 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Macro avg 0.420.35 0.27 0.20 0.27 0.21 0.22 0.24 0.18
GemmaNo Dep
0.480.98 0.50 0.66
0.660.97 0.69 0.81
0.410.97 0.27 0.42
Requires 0.14 1.00 0.25 0.20 0.89 0.32 0.27 0.93 0.42
Implements 0.14 0.12 0.13 0.00 0.00 0.00 0.00 0.00 0.00
Conflicts 0.33 0.21 0.26 N/A N/A N/A 0.00 0.00 0.00
Details 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Is similar 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Macro avg 0.27 0.30 0.22 0.230.320.23 0.21 0.20 0.14
MistralNo Dep
0.770.84 0.91 0.87
0.840.88 0.95 0.92
0.440.76 0.53 0.62
Requires 0.26 0.25 0.25 0.44 0.39 0.41 0.33 0.20 0.25
Implements 0.35 0.47 0.40 0.00 0.00 0.00 0.03 0.67 0.05
Conflicts 0.00 0.00 0.00 N/A N/A N/A 0.50 0.25 0.33
Details 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Is similar 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
Macro avg 0.24 0.27 0.25 0.26 0.27 0.27 0.27 0.27 0.21
variations for few-shot prompting. Specifically, we investigate
four parameters: (1) the embedding model, (2) the similarity
metric, (3) the formulas for calculating similarity between
requirement pairs, and (4) the number of examples (k). For em-
bedding generation, we tested SBERT (Sentence Bidirectional
Encoder Representations from Transformers), specifically the
all-mpnet-base-v2 model, and BGE-M3 (BAAI General Em-
bedding) for their proven efficacy in capturing sentence-level
semantic similarity. The similarity between these embeddings
is measured using Cosine or Euclidean distance, as they
remain the standard metrics for such tasks. To identify the
optimal method for calculating similarity between the target
requirement pair(R 1, R2)and the candidate example pair
(Ra, Rb), we compare two aggregation formulas. The first
formula averages the maximum similarity scores associated
with each requirement in the target pair (see equation (1)). The
second formula computes the arithmetic mean of the similarity
scores of all four requirement combinations.
We also varied the number of examples per dependency
type in the prompt from 1 to 9 and examinedF 1scores
to identify the optimal number to include in the LEREDD
prompt. Additionally, we conducted 24 experiments covering
two RAG-specific parameters: (1) chunk size, experimenting
with 500 and 1000 characters per chunk with 200 character
overlap, and (2) the number of chunks included in the prompt,
considering 2, 6, and 10 chunks, as well as the full document.
After determining the optimal configurations for RAG and
few-shot prompting, we compare the performance of the
different prompting strategies (zero-shot, few-shot, and few-
shot combined with RAG), as measured by the averageF 1
score. The results are reported in Table III.
Figure 3 presents the comparison ofF 1scores for different
few-shot configurations (with respect to similarity and the
number of examples) across the three systems. Due to space
constraints, a comprehensive table with full metrics is included
in our replication package. Table IV shows the results of RAG,
which are obtained by applying RAG to the identified optimal
few-shot configuration. It reports the overall accuracy, along
with precision, recall, andF 1scores, across all dependency
classes. The highest results are highlighted in bold.
1 2 3 4 5 6 7 8 9
Number of Examples per Dependency Type0.30.40.50.60.70.80.9F1 Score
ADB
1 2 3 4 5 6 7 8 9
Number of Examples per Dependency Type0.30.40.50.60.70.80.9F1 Score
TJA
1 2 3 4 5 6 7 8 9
Number of Examples per Dependency Type0.30.40.50.60.70.80.9F1 Score
APA
BGE-Cosine-Avg
BGE-Cosine-MaxBGE-Euclidean-Avg
BGE-Euclidean-MaxSBERT-Cosine-Avg
SBERT-Cosine-MaxSBERT-Euclidean-Avg
SBERT-Euclidean-MaxFig. 3:F 1score Comparison of Few-shot Configurations
Dynamic few-shot configuration.When comparing embed-
ding models,SBERTconsistently outperforms BGE, achiev-
ing the highest F1 scores across the three systems, as shown
in Figure 3. Specifically, it achieves higher bestF 1scores than
BGE, increasing from 0.67 to 0.69 for ADB, from 0.74 to 0.78
for TJA, and from 0.76 to 0.79 for APA. Moreover, SBERT
often demonstrates steeper improvements as the number of
examples increases, reflecting its ability to leverage additional
contextual examples. Regarding similarity metrics, the results
are different across systems. In ADB, Euclidean achieves the
highest peak performance, whereas in TJA and APA, Cosine
yields the best results. However, the difference between the
two metrics is generally small, suggesting that once high-
quality embeddings are available, retrieval effectiveness be-
comes less sensitive to the specific similarity function.
With regard to similarity aggregation, theMaxstrategy
generally produces higherF 1scores than the Average for
calculating similarity. This implies that having at least one
requirement in the example pair with high similarity to a
requirement in the target pair is critical for guiding the model.
In contrast, while Average is more stable, it is sensitive to non-
similar examples, thus lowering the overall similarity score.
Finally, among the number of examples, the highestF 1scores
are achieved with a moderate number. As Figure 3 shows,
all peak F1 score values are observed between three and
five examples for the three systems, after which additional
examples bring only marginal gains or slight fluctuations. This
indicates that providing a moderate number of examples, be-
tweenthree and five, is sufficient to convey most of the useful
task knowledge, while adding more examples introduces noise.
Overall, the optimal few-shot configuration across systems
usesSBERT, Euclidean distance, the maximum-similarity
aggregation, and 4 examples per dependency type. As Ta-

TABLE III:F 1score Comparison of Different Prompting
Strategies Across Three Systems
Dependency TypeZero-shot Few-shot Few-shot & RAG
No dependency0.90 0.94 0.95
Requires0.52 0.63 0.70
Implements0.385 0.68 0.70
TABLE IV: Experimental Results of RAG-based Contextual
Augmentation Across Three Systems
Dataset Metric2 Chunks 6 Chunks 10 Chunks Entire Doc
C1 C2 C1 C2 C1 C2 C1 C2
ADBAcc 0.87 0.89 0.90 0.90 0.900.90 0.90 0.90
P 0.73 0.76 0.79 0.80 0.800.83 0.80 0.79
R 0.54 0.63 0.64 0.67 0.630.68 0.63 0.64
F1 0.61 0.68 0.70 0.72 0.700.73 0.69 0.69
TJAAcc 0.92 0.91 0.940.93 0.93 0.91 0.93 0.93
P 0.800.89 0.84 0.81 0.82 0.76 0.88 0.88
R 0.74 0.64 0.780.75 0.780.76 0.73 0.75
F1 0.77 0.72 0.810.77 0.80 0.76 0.79 0.80
APAAcc 0.84 0.84 0.87 0.86 0.880.86 0.84 0.85
P 0.79 0.79 0.83 0.81 0.840.80 0.78 0.79
R 0.74 0.75 0.80 0.81 0.810.81 0.80 0.81
F1 0.76 0.77 0.82 0.81 0.830.81 0.79 0.80
C1 = 500 characters, C2 = 1000 characters.
ble III shows, this configuration consistently outperforms zero-
shot GPT-4.1, yielding substantial improvements in averageF 1
scores: from 0.90 to 0.94 forNo dependency, 0.52 to 0.63 for
Requires, and 0.385 to 0.68 forImplements.
RAG configuration.Building on the optimal few-shot con-
figuration, we integrate RAG with different numbers and sizes
of chunks. As Table IV shows, the effectiveness of RAG
depends on both chunk size and the number of retrieved
chunks. In general, including a moderate number of chunks,
specifically 6 or 10, yields the best performance: the highest
F1score is obtained with 10 chunks for ADB and APA, while
6 chunks work best for TJA. Fewer chunks limit the diversity
of retrieved context, whereas using the entire document tends
to introduce noise, hindering the identification of relevant
information. Furthermore, shorter contexts (500 characters)
often further improve performance. For instance, using six
500-character chunks instead of six 1000-character chunks
raises TJA’sF 1score from 0.77 to 0.81, likely because shorter
contexts reduce noise. Based on these observations, we useten
500-character chunkswhen integrating RAG into LEREDD.
Performance comparison of the different prompting strate-
gies.Building on the optimal few-shot configuration, inte-
grating RAG brings consistent and meaningful improvements
across all systems. Compared to a simple few-shot, combining
few-shot with RAG increases the averageF 1-score from 0.73
to 0.78 across all dependency types and systems, demonstrat-
ing that retrieved external knowledge effectively complements
in-context examples and helps resolve ambiguities that they
cannot address. As Table III shows, the averageF 1-score
increased from 0.94 to 0.95 forNo dependency, 0.63 to 0.70
forRequires, and 0.68 to 0.70 forImplements. When compared
against zero-shot GPT-4.1, the combined few-shot and RAG
approach provides substantial performance gains for both the
No dependencycases and granular dependency types, most
notably the latter. It achieves relative gains in averageF 1-
scores of 5.56%, 34.62%, and 81.82% forNo dependency,
Requires, andImplements, respectively. These results highlightTABLE V: Experimental Results of LEREDD Against Base-
lines for Intra-Dataset Evaluation across Three Systems
SystemDependency TF-IDF & LSA fine-tuned BERT LEREDDSuppType Acc P R F1 Acc P R F1 Acc P R F1
ADBNo Dep
0.810.93 0.85 0.89
0.820.92 0.86 0.89
0.970.98 0.980.98 66
Requires 0.17 0.33 0.22 0.30 0.50 0.38 0.86 1.000.92 6
Details N/A N/A N/A 0.50 0.50 0.50 1.00 0.750.86 4
Macro avg 0.55 0.59 0.56 0.57 0.62 0.59 0.95 0.91 0.92 76
TJANo Dep
0.890.94 0.94 0.94
0.370.92 0.32 0.48
0.920.94 0.970.96 34
Requires 0.50 0.50 0.50 0.12 0.75 0.20 0.67 0.500.57 4
Macro avg 0.72 0.72 0.72 0.52 0.54 0.34 0.80 0.74 0.76 38
APANo Dep
0.680.84 0.72 0.78
0.660.90 0.62 0.73
0.890.93 0.930.93 29
Requires 0.38 0.56 0.45 0.39 0.78 0.52 0.78 0.780.78 9
Macro avg 0.61 0.64 0.62 0.64 0.70 0.63 0.85 0.85 0.85 38
the critical importance of incorporating relevant examples
and domain-specific context to enhance the LLM’s predictive
performance. Furthermore, while the performance on granular
dependency types is interesting, the exceptionally highF 1-
scores forNo dependencyare of particular interest. Accurately
identifying and filtering these cases, which represent the vast
majority of instances within systems, can substantially reduce
the effort and time required for requirement dependency
analysis. Based on these findings, the few-shot configuration
employingSBERT, Euclidean distance, maximum similar-
ity aggregation, and four examples per dependency type,
integrated with a RAG configuration often 500-character
chunks, was adopted for LEREDD.
Answer to RQ2:Using GPT-4.1, both few-shot prompt-
ing and RAG consistently yield notable improvements
in dependency detection performance for bothNo de-
pendencyand granular dependency types. Optimal per-
formance is achieved with a few-shot configuration em-
ploying SBERT, Euclidean distance, maximum similarity
aggregation, and four examples per dependency type,
combined with a RAG configuration of ten 500-character
chunks. This setup achieves averageF 1scores of 0.95,
0.70, and 0.70 forNo dependency,Requires, andIm-
plements, yielding relative gains in 5.56%, 34.62%, and
81.82%, respectively, compared to zero-shot GPT-4.1.
3) RQ3. Comparison of LEREDD with Baselines for Intra-
Dataset Dependency Detection:Table V reports the intra-
dataset comparison between LEREDD and the two baselines.
The highest results are highlighted in bold, indicating that
LEREDD consistently achieves the best accuracy andF 1
scores across all dependency types relative to the two base-
lines. LEREDD achieves the highest accuracy andF 1scores
on all three systems, reaching 0.97 and 0.92 on ADB, 0.92
and 0.76 on TJA, and 0.89 and 0.85 on APA, respectively.
Compared with the strongest baseline on each dataset (which
varies), this corresponds to relativeF 1score improvements of
55.93% over fine-tuned BERT on ADB (from 0.59 to 0.92),
5.56% over TF-IDF & LSA on TJA (from 0.72 to 0.76), and
34.92% over fine-tuned BERT on APA (from 0.63 to 0.85).
The advantage of LEREDD is more evident when examin-
ing the more challenging dependency types. For theRequires
dependency, LEREDD improves theF 1score from 0.22 and
0.38 to 0.92 on ADB, yielding relative gains of 318% and
142% over TF-IDF & LSA and fine-tuned BERT, respectively.
Across systems, it yields average relative gains of 94.87%

TABLE VI: Experimental Results of LEREDD Against Base-
lines for Cross-Dataset Evaluation across Three Systems
Training TestingDependency TF-IDF & LSA fine-tuned BERT LEREDDSuppType Acc P R F1 Acc P R F1 Acc P R F1
TJA ADBNo Dep
0.830.93 0.88 0.90
0.550.90 0.57 0.69
0.930.97 0.95 0.96 329
Requires 0.20 0.31 0.24 0.07 0.34 0.12 0.59 0.75 0.66 32
Macro avg 0.56 0.59 0.57 0.48 0.45 0.41 0.78 0.85 0.81 361
APA ADBNo Dep
0.830.93 0.88 0.90
0.550.90 0.57 0.70
0.930.97 0.95 0.96 329
Requires 0.20 0.31 0.24 0.07 0.31 0.11 0.59 0.72 0.65 32
Macro avg 0.56 0.59 0.57 0.48 0.44 0.40 0.78 0.84 0.80 361
ADB TJANo Dep
0.870.93 0.92 0.93
0.380.95 0.33 0.49
0.960.98 0.99 0.99 169
Requires 0.32 0.33 0.32 0.14 0.83 0.23 0.87 0.72 0.79 18
Details N/A N/A N/A 0.00 0.00 0.00 0.50 0.50 0.50 2
Macro avg 0.62 0.63 0.63 0.36 0.39 0.24 0.78 0.74 0.76 189
APA TJANo Dep
0.870.93 0.92 0.93
0.480.90 0.48 0.63
0.930.97 0.95 0.96 169
Requires 0.32 0.33 0.32 0.09 0.50 0.16 0.59 0.72 0.65 18
Macro avg 0.62 0.63 0.63 0.50 0.49 0.39 0.78 0.83 0.80 187
ADB APANo Dep
0.650.83 0.68 0.75
0.310.72 0.18 0.29
0.860.92 0.910.91 144
Requires 0.35 0.56 0.43 0.22 0.71 0.33 0.71 0.710.71 45
Details N/A N/A N/A 0.00 0.00 0.00 0.00 0.00 0.00 1
Macro avg 0.59 0.62 0.59 0.31 0.30 0.21 0.54 0.54 0.54 190
TJA APANo Dep
0.650.83 0.68 0.75
0.240.67 0.01 0.03
0.880.92 0.92 0.92 144
Requires 0.35 0.56 0.43 0.24 0.98 0.38 0.76 0.76 0.76 45
Macro avg 0.59 0.62 0.59 0.45 0.50 0.20 0.84 0.84 0.84 189
and 105.41% inF 1scores for theRequiresdependency
over the baselines and provides a better balance between
precision and recall. For theDetailsdependency in ADB,
which contains only a few instances, LEREDD attains anF 1
score of 0.86, representing a 72% improvement over fine-
tuned BERT. Regarding theNo dependencyclass, LEREDD
consistently achieves superiorF 1scores, with the average
score increasing from 0.87 for TF-IDF & LSA and 0.7 for fine-
tuned BERT to 0.96 for LEREDD, yielding relative gains of
10.34% and 37.14%, respectively. Furthermore, the baselines
exhibit noticeable instability across datasets. For example, the
accuracy of fine-tuned BERT drops from 0.82 on ADB to 0.37
on TJA, whereas LEREDD remains consistently above 0.89.
Answer to RQ3:In the intra-dataset experimental setup,
LEREDD achieves higher accuracy andF 1-scores com-
pared to the baselines for bothNo dependencycases
and granular dependency types. It yields relative gains
of 33.33% over TF-IDF & LSA and 61.54% over fine-
tuned BERT across all dependency types and systems.
Furthermore, LEREDD consistently offers more stable
performance across different systems.
4) RQ4. Comparison of LEREDD with Baselines for Cross-
Dataset Dependency Detection:To address this RQ, we eval-
uate LEREDD’s performance against baselines using a cross-
dataset experimental setup: one dataset is used for testing,
while a different dataset serves as the training set for BERT
and as an example pool for LEREDD. As the TF-IDF & LSA
baseline does not require training, we report the results of
the same experiments conducted for RQ3, with performance
metrics computed over the entire requirements set. For fine-
tuned BERT and LEREDD, we run six experiments each to
cover all possible training–testing combinations across the
three systems, for a total of 12 experiments. Table VI reports
the results of the experiments for LEREDD and the baselines.
To facilitate direct comparison in Table VI, the TF-IDF &
LSA results are duplicated for combinations sharing the same
testing dataset. For each training–testing combination, the
highest-performing results are highlighted in bold.
The results of the TF-IDF & LSA baseline remain con-
sistent with those reported in RQ3, with a slight decrease
in performance in detecting theRequiresdependency, with
theF 1score averaging 0.33 across systems. Nevertheless,the baseline continues to perform well at identifyingNo
dependency, achieving an averageF 1score of 0.86.
The performance of fine-tuned BERT on theRequires
dependency decreased significantly compared to the results
observed in RQ3, with the averageF 1score dropping from
0.52 to 0.31. This decline is attributable to fine-tuned BERT,
like most ML models, being heavily dependent on the similar-
ities between training and testing data. Because these models
learn patterns and assign class weights based on the training
corpus, using different testing and training sets substantially
affects their performance. Notably, experiments using the ADB
system for training show a less severe performance drop. This
can be attributed to the larger size of the ADB training set,
which contains nearly twice the requirements of the other
systems, providing a slight advantage. Overall, the results
indicate that fine-tuned BERT struggles with cross-dataset
evaluations, reflecting a well-known limitation of ML models.
LEREDD continuously achieves the highest performance
across all six experiments. Although there is a slight per-
formance decline compared to RQ3, with average accuracy
andF 1scores decreasing by 1.61% and 9.52%, performance
remains high. LEREDD yields an average accuracy of 0.915
and an averageF 1score of 0.76 across all experiments. No-
tably, accuracy remained stable across datasets with differing
training sets, indicating LEREDD’s robustness to training data
variation. Regarding theDetailsdependency, accuracy was
constrained by its extreme sparsity in TJA and APA, resulting
inF 1scores of 0.50 and 0.00. These results are expected,
given the very low number of instances and the associated high
prediction risks. For theRequiresdependency, LEREDD sig-
nificantly outperforms the baselines, as the averageF 1scores
increased from 0.33 and 0.22 to 0.70, yielding relative gains
of 112.12% and 218.18%, respectively. LEREDD’s superior
performance is also evident in theNo dependencyclass, where
averageF 1scores improved from 0.86 and 0.47 to 0.95,
yielding relative gains of 10.47% and 102.13%, respectively.
Overall, LEREDD significantly outperforms the baselines in
cross-dataset evaluation. These results are particularly promis-
ing, given the scarcity of training data and lack of annotated
requirements from the same dataset in real-world settings.
Answer to RQ4:LEREDD significantly outperforms
baselines in cross-dataset evaluations for bothNo depen-
dencycases and granular dependency types. Furthermore,
its accuracy is comparable to that of the intra-dataset
evaluation, with the average decreasing by only 1.61%.
VI. DISCUSSION
In this paper, we introduce LEREDD. Through extensive
evaluation across multiple datasets and experimental settings,
we demonstrated that LEREDD consistently outperforms
SOTA baselines while maintaining strong generalization.
Zero-Shot Limitations.In our experiments, zero-shot GPT-
4.1 reliably detectsNo dependencybut performs poorly on
fine-grained relations (e.g.,Requires,Details). This indicates

that dependency detection is not merely semantic-similarity
matching but a structured form of reasoning that requires
domain context. In the absence of task-specific guidance,
zero-shot models rely on coarse semantic signals, tend to
over-predict dominant classes, and collapse on minority de-
pendencies. These findings indicate that zero-shot LLMs are
insufficient for industrial-grade dependency analysis.
Few-Shot Prompting: Precision over Quantity.We show
that few-shot prompting yields rapid performance gains that
saturate after three to five examples, indicating that asmall
number of relevant examplesis sufficient to establish effective
decision boundaries. The superiority of theMaxretrieval
strategy shows that a single highly relevant example is more
valuable than averaging multiple weak ones. SBERT consis-
tently outperforms BGE, while similarity metrics have a minor
impact. Thus, retrieval quality matters over retrieval scale.
RAG: Structured Context Augmentation.RAG consis-
tently improves performance over few-shot prompting alone,
confirming that examples provide task guidance while re-
trieved context supplies domain grounding. Optimal perfor-
mance under moderate chunk sizes highlights the trade-off
between contextual coverage and noise. Dependency detection,
therefore, benefits from structured, selective context augmen-
tation rather than unrestricted information injection.
LEREDD: Robustness Within and Across Datasets.
LEREDD maintains strong, stable performance in both intra-
and cross-dataset evaluations. It achieves superior results on
minority dependencies and onNo dependencycases. This
advantage persists under a distribution shift, where fine-tuned
BERT degrades substantially. LEREDD leverages dynamic
retrieval and inferential reasoning, reducing dependence on
static training distributions. Consequently, it maintains higher,
more balanced precision and recall across systems, demon-
strating stronger generalization in evolving industrial contexts.
Regarding fine-grained dependency types, while LEREDD
offers substantial improvements over baselines and zero-shot
LLMs, further refinement is necessary to achieve optimal
performance. LEREDD is particularly proficient in identifying
No dependencycases. Because these cases constitute the vast
majority of instances, effectively filtering them out substan-
tially reduces the overhead associated with the dependency
analysis task and is of high practical value.
LEREDD: Computational Cost.Regarding computational
time, LEREDD provides a good trade-off between BERT and
TF-IDF & LSA, while achieving a much higher and stable
accuracy. This is evidenced by our empirical results: TF-IDF
& LSA required an average of 2.48 seconds, while fine-tuned
BERT and LEREDD required an average of 4 minutes 3
seconds and 1 minute 48 seconds, respectively, across the three
systems during the intra-dataset evaluation.
Implications.Our findings suggest several important in-
sights regarding the use of LLMs for requirement dependency
detection. First, the accuracy of detectingNo dependencycases
is so high that it can be extremely useful for filtering out most
requirement pairs when analyzing inter-requirement dependen-
cies. Second, zero-shot LLMs are insufficient for accuratelyidentifying fine-grained dependency types and often yield low
F1scores even inNo dependencycases, underscoring the
need for task-specific guidance. Third, much better results can
be obtained with a small set of carefully selected, relevant
examples. Fourth, regarding RAG, retrieval precision plays a
more critical role than retrieval volume, as providing highly
relevant contextual information yields notable gains inF 1
Scores beyond those achieved with few-shot prompting alone.
Finally, the inferential capabilities of LLMs, combined with
their reduced reliance on dataset-specific distributions, con-
tribute to improved robustness across systems, enabling more
consistent performance in cross-dataset evaluation settings.
VII. THREATS TO VALIDITY
One threat to construct validity arises from dataset anno-
tation, which may be biased and error-prone, particularly for
fine-grained relations. To mitigate this risk, we followed pre-
defined annotation guidelines, and all annotations were cross-
validated by two independent annotators to reduce potential
bias. Another threat concerns external validity, as our datasets
are confined to the automotive domain, potentially limiting
generalizability to other application areas. To alleviate this
concern, we selected three systems that differ in size and
distribution and further conducted cross-dataset evaluations to
provide a more comprehensive assessment. Replicating base-
lines may also raise validity concerns if there are implementa-
tion differences. To reduce this risk, we reused the parameters
provided in the original approaches and tuned hyperparameters
to identify optimal configurations, ensuring fair and consistent
comparisons. Furthermore, LLM-based methods may produce
varying outputs due to their stochastic nature. To ensure re-
producibility and largely reduce randomness as a confounding
factor, we fixed the temperature to 0, enforcing deterministic
model behavior across all experiments. Finally, to mitigate
reliability threats and ensure reproducibility of our findings,
we provide a replication package that includes the source code,
configurations, and datasets used throughout the paper.
VIII. CONCLUSION
In this paper, we introduce LEREDD, an LLM-based ap-
proach for the automated detection of various types of depen-
dencies between NL requirements. LEREDD leverages RAG
and ICL to retrieve domain-specific contextual information
that guides the LLM during detection. Experimental results
show that LEREDD significantly outperforms SOTA baselines.
It achieves an average accuracy of 92.66% and anF 1score of
84.33% across three systems and all dependency types. The
performance is particularly notable for theNo dependency
class, which achieves an averageF 1-score of 96%. These
results are especially compelling asNo dependencycases ac-
count for the vast majority of requirement pairs. By effectively
filtering these cases, LEREDD can substantially reduce the
analysts’ overhead for dependency analysis. LEREDD also
demonstrates robustness in cross-system evaluations, achieving
significantly higher accuracy than baselines. Furthermore, the
annotated corpus provided in this paper can serve as a valuable

resource for training and benchmarking for future research.
As future work, we will extend LEREDD to capture indirect
and implicit requirement dependencies. In addition, we will
investigate the utility of predicted dependencies for impact
analysis when requirements evolve over time.
DATAAVAILABILITYSTATEMENT
Our replication package will be made available upon pub-
lication. It includes the annotated datasets, source code (zero-
shot, baselines, and LEREDD implementations), and a com-
prehensive table of few-shot results. Detailed experimental and
evaluation settings are provided to facilitate reproducibility.

REFERENCES
[1] “Iso/iec/ieee international standard - systems and software engineering
– life cycle processes – requirements engineering,”ISO/IEC/IEEE
29148:2018(E), 2018.
[2] H. Hofmann and F. Lehner, “Requirements engineering as a success
factor in software projects,”IEEE Software, vol. 18, no. 4, pp. 58–66,
2001.
[3] Q. Motger and X. Franch,Automated Requirements Relations Extraction.
Cham: Springer Nature Switzerland, 2025, pp. 177–206.
[4] G. Deshpande, C. Arora, and G. Ruhe, “Data-driven elicitation and
optimization of dependencies between requirements,” in2019 IEEE
27th International Requirements Engineering Conference (RE), 2019,
pp. 416–421.
[5] G. Deshpande, Q. Motger, C. Palomares, I. Kamra, K. Biesialska,
X. Franch, G. Ruhe, and J. Ho, “Requirements dependency extraction by
integrating active learning with ontology-based retrieval,” in2020 IEEE
28th International Requirements Engineering Conference (RE), 2020,
pp. 78–89.
[6] N. Mohamed, S. Mazen, and W. Helmy, “A comprehensive review of
software requirements dependencies analysis techniques,”Iraqi Journal
for Computer Science and Mathematics, vol. 6, no. 3, p. Article 5, 2025.
[7] T. Wu, L. Luo, Y .-F. Li, S. Pan, T.-T. Vu, and G. Haffari, “Continual
learning for large language models: A survey,” 2024. [Online].
Available: https://arxiv.org/abs/2402.01364
[8] H. Naveed, A. U. Khan, S. Qiu, M. Saqib, S. Anwar, M. Usman,
N. Akhtar, N. Barnes, and A. Mian, “A comprehensive overview
of large language models,”ACM Trans. Intell. Syst. Technol., 2025.
[Online]. Available: https://doi.org/10.1145/3744746
[9] M. A. Zadenoori, J. D ˛ abrowski, W. Alhoshan, L. Zhao, and
A. Ferrari, “Large language models (llms) for requirements engineering
(re): A systematic literature review,” 2025. [Online]. Available:
https://arxiv.org/abs/2509.11446
[10] N. Marques, R. R. Silva, and J. Bernardino, “Using chatgpt
in software requirements engineering: A comprehensive review,”
Future Internet, vol. 16, no. 6, 2024. [Online]. Available: https:
//www.mdpi.com/1999-5903/16/6/180
[11] Sarwosri, U. L. Yuhana, and S. Rochimah, “Conflict detection of
functional requirements based on clustering and rule-based system,”
IEEE Access, vol. 12, pp. 174 330–174 342, 2024.
[12] Y . Priyadi, A. Djunaidy, and D. Siahaan, “Requirements dependency
graph modeling on software requirements specification using text analy-
sis,” in2019 1st International Conference on Cybernetics and Intelligent
System (ICORIS), vol. 1, 2019, pp. 221–226.
[13] R. Asyrofi, D. O. Siahaan, and Y . Priyadi, “Extraction dependency based
on evolutionary requirement using natural language processing,” in2020
3rd International Seminar on Research of Information Technology and
Intelligent Systems (ISRITI), 2020, pp. 332–337.
[14] W. Guo, L. Zhang, and X. Lian, “Automatically detecting the
conflicts between software requirements based on finer semantic
analysis,”ArXiv, vol. abs/2103.02255, 2021. [Online]. Available:
https://api.semanticscholar.org/CorpusID:232104883
[15] I. Gräßler, C. Oleff, M. Hieb, and D. Preuß, “Automated requirement
dependency analysis for complex technical systems,”Proceedings of the
Design Society, vol. 2, p. 1865–1874, 2022.
[16] A. E. Gärtner and D. Göhlich, “Automated requirement contradiction
detection through formal logic and llms,”Automated Software
Engg., vol. 31, no. 2, Jun. 2024. [Online]. Available: https:
//doi.org/10.1007/s10515-024-00452-x
[17] N. Almoqren and M. Alrashoud, “Llm-driven active learning for
dependency analysis of mobile app requirements through contextual
reasoning and structural relationships,”Applied Sciences, vol. 15,
no. 18, 2025. [Online]. Available: https://www.mdpi.com/2076-3417/
15/18/9891
[18] F. Niu, R. Pan, L. C. Briand, and H. Hu, “Tvr: Automotive
system requirement traceability validation and recovery through
retrieval-augmented generation,” 2026. [Online]. Available: https:
//arxiv.org/abs/2504.15427
[19] T. OGAWA, A. Ohnishi, and H. Shimakawa, “A retrieval method of
software requirements from japanese requirements document with de-
pendency analysis and keywords,” in2022 IEEE Asia-Pacific Conference
on Computer Science and Data Engineering (CSDE), 2022, pp. 1–6.[20] Z. Li, M. Chen, L. Huang, and V . Ng, “Recovering traceability
links in requirements documents,” inProceedings of the Nineteenth
Conference on Computational Natural Language Learning. Beijing,
China: Association for Computational Linguistics, 2015, pp. 237–246.
[Online]. Available: https://aclanthology.org/K15-1024/
[21] R. Samer, M. Stettinger, M. Atas, A. Felfernig, G. Ruhe, and G. Desh-
pande, “New approaches to the identification of dependencies between
requirements,” in2019 IEEE 31st International Conference on Tools
with Artificial Intelligence (ICTAI), 2019, pp. 1265–1270.
[22] H. Guan, H. Xu, and L. Cai, “Requirement dependency extraction
based on improved stacking ensemble machine learning,”Mathematics,
vol. 12, no. 9, 2024. [Online]. Available: https://www.mdpi.com/
2227-7390/12/9/1272
[23] A. Schlutter and A. V ogelsang, “Improving trace link recovery using
semantic relation graphs and spreading activation,” inRequirements
Engineering: Foundation for Software Quality: 27th International
Working Conference, REFSQ 2021, Essen, Germany, April 12–15, 2021,
Proceedings. Berlin, Heidelberg: Springer-Verlag, 2021, p. 37–53.
[Online]. Available: https://doi.org/10.1007/978-3-030-73128-1_3
[24] F. Mokammel, E. Coatanéa, J. Coatanéa, V . Nenchev, E. Blanco, and
M. Pietola, “Automatic requirements extraction, analysis, and graph rep-
resentation using an approach derived from computational linguistics,”
Systems Engineering, vol. 21, no. 6, pp. 555–575, 2018.
[25] Q. Motger, R. Borrull, C. Palomares, and J. Marco, “Openreq-dd:
A. requirements dependency detection tool,” inREFSQ Workshops,
2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:
186206345
[26] J. Fischbach, J. Frattini, A. Spaans, M. Kummeth, A. V ogelsang,
D. Mendez, and M. Unterkalmsteiner, “Automatic detection of
causality in requirement artifacts: The cira approach,” inRequirements
Engineering: Foundation for Software Quality: 27th International
Working Conference, REFSQ 2021, Essen, Germany, April 12–15, 2021,
Proceedings. Berlin, Heidelberg: Springer-Verlag, 2021, p. 19–36.
[Online]. Available: https://doi.org/10.1007/978-3-030-73128-1_2
[27] J. Fischbach, J. Frattini, and A. V ogelsang, “Cira: A tool for
the automatic detection of causal relationships in requirements
artifacts,”ArXiv, vol. abs/2103.06768, 2021. [Online]. Available:
https://api.semanticscholar.org/CorpusID:232185494
[28] M. Atas, R. Samer, and A. Felfernig, “Automated identification of type-
specific dependencies between requirements,” in2018 IEEE/WIC/ACM
International Conference on Web Intelligence (WI), 2018, pp. 688–695.
[29] H. Guan, G. Cai, and H. Xu, “Automatic requirement dependency extrac-
tion based on integrated active learning strategies,”Machine Intelligence
Research, vol. 21, 02 2024.
[30] G. Abeba and E. Alemneh, “Identification of nonfunctional requirement
conflicts: Machine learning approach,” 1 2022.
[31] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu,
T. Liu, B. Chang, X. Sun, L. Li, and Z. Sui, “A survey on in-context
learning,” 2024. [Online]. Available: https://arxiv.org/abs/2301.00234
[32] X. Li, K. Lv, H. Yan, T. Lin, W. Zhu, Y . Ni, G. Xie, X. Wang, and
X. Qiu, “Unified demonstration retriever for in-context learning,” 2023.
[Online]. Available: https://arxiv.org/abs/2305.04320
[33] S. Wu, Y . Xiong, Y . Cui, H. Wu, C. Chen, Y . Yuan, L. Huang, X. Liu,
T.-W. Kuo, N. Guan, and C. J. Xue, “Retrieval-augmented generation
for natural language processing: A survey,” 2025. [Online]. Available:
https://arxiv.org/abs/2407.13193
[34] M. I, S. Saxena, S. Prasad, M. V . S. Prakash, A. Shankar, V . V ,
V . Vaddina, and S. Gopalakrishnan, “Minimizing factual inconsistency
and hallucination in large language models,” 2023. [Online]. Available:
https://arxiv.org/abs/2311.13878
[35] T. Xu, S. Wu, S. Diao, X. Liu, X. Wang, Y . Chen, and J. Gao, “Sayself:
Teaching llms to express confidence with self-reflective rationales,”
2024. [Online]. Available: https://arxiv.org/abs/2405.20974
[36] G. Detommaso, M. Bertran, R. Fogliato, and A. Roth, “Multicalibration
for confidence scoring in llms,” 2024. [Online]. Available: https:
//arxiv.org/abs/2404.04689
[37] S. Prabhu, M. Mohamed, and H. Misra, “Multi-class text classification
using bert-based active learning,”ArXiv, vol. abs/2104.14289,
2021. [Online]. Available: https://api.semanticscholar.org/CorpusID:
233444345
[38] Y . Arslan, K. Allix, L. Veiber, C. Lothritz, T. F. Bissyandé, J. Klein,
and A. Goujon, “A comparison of pre-trained language models for
multi-class text classification in the financial domain,” inCompanion
Proceedings of the Web Conference 2021, ser. WWW ’21. New York,

NY , USA: Association for Computing Machinery, 2021, p. 260–268.
[Online]. Available: https://doi.org/10.1145/3442442.3451375
[39] X. Hou, Y . Zhao, Y . Liu, Z. Yang, K. Wang, L. Li, X. Luo,
D. Lo, J. Grundy, and H. Wang, “Large language models for
software engineering: A systematic literature review,”ACM Trans.
Softw. Eng. Methodol., vol. 33, no. 8, Dec. 2024. [Online]. Available:
https://doi.org/10.1145/3695988