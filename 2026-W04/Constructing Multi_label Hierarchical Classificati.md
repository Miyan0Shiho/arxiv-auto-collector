# Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging

**Authors**: Andrew Crossman, Jonah Dodd, Viralam Ramamurthy Chaithanya Kumar, Riyaz Mohammed, Andrew R. Plummer, Chandra Sekharudu, Deepak Warrier, Mohammad Yekrangian

**Published**: 2026-01-21 00:41:34

**PDF URL**: [https://arxiv.org/pdf/2601.14556v1](https://arxiv.org/pdf/2601.14556v1)

## Abstract
MITRE ATT&CK is a cybersecurity knowledge base that organizes threat actor and cyber-attack information into a set of tactics describing the reasons and goals threat actors have for carrying out attacks, with each tactic having a set of techniques that describe the potential methods used in these attacks. One major application of ATT&CK is the use of its tactic and technique hierarchy by security specialists as a framework for annotating cyber-threat intelligence reports, vulnerability descriptions, threat scenarios, inter alia, to facilitate downstream analyses. To date, the tagging process is still largely done manually. In this technical note, we provide a stratified "task space" characterization of the MITRE ATT&CK text tagging task for organizing previous efforts toward automation using AIML methods, while also clarifying pathways for constructing new methods. To illustrate one of the pathways, we use the task space strata to stage-wise construct our own multi-label hierarchical classification models for the text tagging task via experimentation over general cyber-threat intelligence text -- using shareable computational tools and publicly releasing the models to the security community (via https://github.com/jpmorganchase/MITRE_models). Our multi-label hierarchical approach yields accuracy scores of roughly 94% at the tactic level, as well as accuracy scores of roughly 82% at the technique level. The models also meet or surpass state-of-the-art performance while relying only on classical machine learning methods -- removing any dependence on LLMs, RAG, agents, or more complex hierarchical approaches. Moreover, we show that GPT-4o model performance at the tactic level is significantly lower (roughly 60% accuracy) than our own approach. We also extend our baseline model to a corpus of threat scenarios for financial applications produced by subject matter experts.

## Full Text


<!-- PDF content starts -->

CONSTRUCTINGMULTI-LABELHIERARCHICALCLASSIFICATION
MODELS FORMITRE ATT&CK TEXTTAGGING
TECHNICALNOTE
Andrew Crossman∗
JPMorganChaseJonah Dodd
JPMorganChaseViralam Ramamurthy Chaithanya Kumar
JPMorganChaseRiyaz Mohammed
JPMorganChase
Andrew R. Plummer
JPMorganChaseChandra Sekharudu
JPMorganChaseDeepak Warrier
JPMorganChaseMohammad Yekrangian
JPMorganChase
ABSTRACT
MITRE ATT&CK is a cybersecurity knowledge base that organizes threat actor and cyber-attack
information into a set of tactics describing the reasons and goals threat actors have for carrying
out attacks, with each tactic having a set of techniques that describe the potential methods used in
these attacks. One major application of ATT&CK is the use of its tactic and technique hierarchy
by security specialists as a framework for annotating cyber-threat intelligence reports, vulnerability
descriptions, threat scenarios, inter alia, to facilitate downstream analyses. To date, the tagging
process is still largely done manually. In this technical note, we provide a stratified "task space"
characterization of the MITRE ATT&CK text tagging task for organizing previous efforts toward
automation using AIML methods, while also clarifying pathways for constructing new methods.
To illustrate one of the pathways, we use the task space strata to stage-wise construct our own
multi-label hierarchical classification models for the text tagging task via experimentation over
general cyber-threat intelligence text – using shareable computational tools and publicly releasing
the models to the security community (via https://github.com/jpmorganchase/MITRE_models). Our
multi-label hierarchical approach yields accuracy scores of roughly 94% at the tactic level, as well as
accuracy scores of roughly 82% at the technique level (when cast as multiclass transformations). The
models also meet or surpass state-of-the-art performance while relying only on classical machine
learning methods – removing any dependence on LLMs, RAG, agents, or more complex hierarchical
approaches (e.g., algorithm adaptation methods, or DAG-based methods). Moreover, we show that
GPT-4o model performance at the tactic level is significantly lower (roughly 60% accuracy) than
our own approach. We also extend our baseline model to a corpus of threat scenarios for financial
applications produced by subject matter experts.
KeywordsMITRE·Cybersecurity·Generative AI·Multi-label Classification·Hierarchical Classification
1 Introduction
Formal developments in cybersecurity began to take shape throughout the latter half of the 20th century (e.g., packet
switching, Diffie-Hellman key exchange, early antivirus software, inter alia) due largely to the dramatic increase in
the inter-connectivity of government, industry, and civilian information systems. The expansion and globalization
of the internet and related technologies over the last 30 years necessitated further advancement of security practices
(internet security protocols, cloud security protocols, etc.) to protect critical information resources from cyber-attacks
and their resulting cost. Indeed the widening scope of cybersecurity over the last few decades led organizations across
government and industry to formulate frameworks that support security specialists in the characterization, detection,
mitigation, and prevention of cyber-attacks.
∗Authors listed alphabetically.arXiv:2601.14556v1  [cs.LG]  21 Jan 2026

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Lockheed Martin produced a cybersecurity framework focusing on threat-based aspects of risk [Hutchins et al., 2011]
adapted from the DoD’s "kill chain" approach for targeting and engaging an adversary and the DoD’s "course of action"
model for disruption of adversary activities. The kill chain approach models the stages of an adversary’s progression
toward establishing an advanced persistent threat to a system together with potential detection and mitigation capabilities
at each stage. Analysis of multiple kill chains over time yields patterns of behavior, stratified into tactics and techniques,
that respectively characterize the "why" and "how" of adversary attacks.
The MITRE Corporation produces and maintains the ATT&CK framework [The MITRE Corporation, 2025] – a
knowledge base constructed from real-world observations of cyber-attacks that includes threat actor groups and their
known means of attack. The ATT&CK framework organizes cyber-attack information hierarchically, with the two main
levels being the tactics and techniques that characterize adversarial activities. In line with the kill chain, the tactics
correspond to the "why" of an attack – indicating the adversary’s goals or reasons, while the techniques correspond to
"how" the adversaries performed their attacks. The ATT&CK framework has been widely adopted and applied within
government and industry for threat modeling (see Crossman et al., 2025 and Section 3 for specific applications), threat
detection and hunting, vulnerability analysis, control validation, as well as broader cybersecurity intelligence analysis.
Note that the ATT&CK framework is not tied to a specific model of cyber-attack stages, allowing ATT&CK to easily
function as a cyber-threat intelligence annotation framework. Indeed, one of the major tasks of security analysts involves
reading cyber-threat intelligence reports and tagging the documents and their contents with MITRE ATT&CK tactics
and techniques to facilitate downstream analyses. To date, the tagging process is still largely carried out by analysts in
manual fashion. Over the last decade a collection of computational methods have taken shape that aim to automate (or
semi-automate) the tagging task to reduce analyst toil (see Section 2 for a review of established and emerging methods).
In this connection, our contributions in this technical note are the following:
•A stratified "task space" formulation of the MITRE ATT&CK text tagging task (Section 2) for organizing
existing AIML approaches and facilitating further developments;
•A "bottom-up" stage-wise construction of a baseline multi-label hierarchical tagging system for general cyber-
intelligence texts following the task space levels – the construction process is not beholden to canonical "top-
down" AIML modeling approaches or architectures, but rather, organically encapsulates the "Best Practices for
MITRE ATT&CK Mapping" specified in CISA’s guide for analysts [Cybersecurity and Infrastructure Security
Agency, 2023], building up from experimentation;
• A comparison of our model performance against GPT-4o during the construction process (Section 3);
•An example of the re-use of the baseline model to bootstrap modeling on new data sets – using a corpus of
threat scenarios for financial applications produced by security specialists within JPMC (Section 3);
•A release of a version of our tagging system that is publicly available for download and use by the security
community (via https://github.com/jpmorganchase/MITRE_models).
2 MITRE ATT&CK Text Tagging Task Formulation
We first provide a general formulation of the MITRE ATT&CK text tagging task in order to both organize existing
work on its (semi-)automation and clarify potential pathways for further development of its AIML-based modeling.
This "task space" formulation, shown in Table 1, provides clear strata for such comparisons, while also serving as a
road map for "bottom-up" model-building in low-resource-sparse-data settings (as we show in Section 3). Although a
comprehensive review of existing work is beyond the scope of this technical note (see Büchel et al., 2025, for a recent
survey), we follow with a brief AIML-focused review of related works established in the public space that exemplify
the different levels of instantiations of the general tagging task (listing references for each type).
The MITRE ATT&CK text tagging task takes the following general form:
ATT&CK:D7→T
where Dis a text document and Tis a formal representation of aspects of the ATT&CK knowledge base (restricted to
the Enterprise Matrix v14, herein). In the simplest form, Dis a short document, e.g., a sentence or phrase, and Tis a
single tactic or technique. However, both DandTcan be made complex. The input Dis extensible to paragraphs,
full documents, or even sets of documents, as well as text that varies in topic from general cybersecurity intelligence,
to threat scenario descriptions, threat reports, cyber-attack reports, vulnerability descriptions, etc. (see Della Penna
et al., 2025 for a review of existing annotated corpora corresponding to the types of D, as well as Alam et al., 2024
for benchmarking). The formal representations Tare extensible from a single tactic or technique to sets of tactics or
techniques, sets of tactics together with techniques, textual descriptions of tactics and techniques, hierarchical structures
2

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Table 1: Stratified Task Types for MITRE ATTA&CK Text Tagging.
Task ID Task TypeATT&CKMapping Form Output Details
1Multiclass Tactic
ClassificationD7→TTis a single tactic selected from
a set of tactics
2Multiclass Technique
ClassificationD7→TTis a single technique selected
from a set of techniques
3Multi-label Tactic
ClassificationD7→ {T 1, . . . , T n}Tiare tactics selected from a set
of tactics
4Multi-label Technique
ClassificationD7→ {T 1, . . . , T n}Tiare techniques selected from
a set of techniques
5Mixed-type Multi-label
ClassificationD7→ {T 1, . . . , T n}Tiare tactics and/or techniques
selected from a set of tactics and
a set of techniques
6Multiclass Hierarchical
ClassificationD7→(T 1, T2)T1is a tactic andT 2is a
technique forT 1
7Multi-label Hierarchical
ClassificationD7→ {T 1, . . . , T n}Tiis a tuple(T0
i, T1
i, . . . , Tk
i)
where T0
iis a tactic and each Tj
i
is a technique forT0
i(j >0)
8 Text-to-Text ClassificationD7→TTis a text description of tactics
or techniques (or both)
over tactics and techniques, etc. Task types that capture the nomenclature and descriptions of the common different
forms of Tare stratified (roughly) in terms of complexity and given IDs in Table 1. We step through descriptions of
existing works that exemplify each of these task types below.
Initial efforts to automate the tagging task following the advent of ATT&CK in 2013 relied primarily on expert-crafted
taxonomies and knowledge graphs over ATT&CK information as a basis for fuzzy string matching algorithms mapping
input texts of cyber-threat intel to known graph/taxonomy entries (see MITRE ATT&CK Extractor, MITRE D3FEND
for recent versions, roughly addressing Task IDs 1 and 2 in Table 1). While effective to a degree, the rigidity and
inflexibility of these methods (relying on classical NLP-based syntactic and semantic representations of cyber-threat
intel entities, relations, and concepts) led to the initiation (and later expansion) of efforts in the emerging machine
learning space (e.g., Ayoade et al., 2018, Ampel et al., 2021, Rahman et al., 2024).
The Threat Report ATT&CK Mapper (TRAM) project began as a cybersecurity community effort to drive machine
learning-based progress on the tagging task. The follow-up TRAM 2.0 (https://github.com/center-for-threat-informed-
defense/tram) extended the original project beyond basic machine learning methods via the incorporation of emerging
transformer-based text representations known to facilitate tasks akin to ATT&CK tagging (see also Alves et al., 2022,
You et al., 2022, Rani et al., 2023, Rani et al., 2024), while also adopting a multi-label approach (Task IDs 3 and 4 in
Table 1, see Mendsaikhan et al., 2020, Kuppa et al., 2021, Grigorescu et al., 2022 for more on multi-label methods).
Still, the tagging task in TRAM is restricted to just the technique level within the ATT&CK hierarchy.
In contrast, the Reports Classification by Adversarial Tactics and Techniques (rcATT) system (see Legoy et al., 2020
and https://github.com/vlegoy/rcATT) tags cyber-threat intelligence reports with both tactics and techniques (at the
document level) using machine learning models trained independently at the tactic and techniques levels of the ATT&CK
framework (falling within Task ID 5 in Table 1). The system is equipped with a UI for recording user feedback to the
automated tagging output and using it for updating the tagging models. The approach, however, does not capture or
utilize the known hierarchical relationships between tactics and techniques in the classification process (but rather as a
post-classification processing step).
3

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Figure 1: A multi-label hierarchical classification system for the MITRE ATT&CK text tagging task. Documents are
decomposed into sentences that are vectorized using TF-IDF. The system provides a hashing technique for encrypting
the text as a part of the vectorization process. The first level of hierarchical classification (a) uses a multi-label
classification model to predict the top ntactic labels. The second level (b) uses tactic-specific multi-label classification
models, conditioned on the predicted tactics, to provide the top mtechnique labels for each tactic. Output for the entire
system (c) is a structure of(n∗m)-many tactic-technique pairs.
TTPDrill [Husari et al., 2017] is an ontology-based approach (see Satvat et al., 2021, Li et al., 2022, and Alam et al.,
2023 for related approaches) to the tagging task that directly incorporates the hierarchical relationship between tactics
and techniques when mapping sentences in cyber-threat intelligence reports (Task ID 6 in Table 1). Its threat action
ontology is manually crafted with fields that hierarchically represent kill chain phases, tactics, and techniques, as well
as more specific information on threat action types. Sentences are mapped to the ontology first through a dependency
parser that creates threat action "candidates" from their constituent text, and the candidates are compared against
ontology entries via a semantic similarity computation. The tactic and technique of the best matching ontology entry is
assigned to each sentence in a threat report (above a learned threshold). One drawback is that the mapping produces
just one tactic-technique pair for each sentence when multiple such labels may be relevant for cyber-threat analysis.
A growing number of methods over the last few years attempt to directly address the multi-label nature of the ATT&CK
tagging task on the one hand, in addition to its hierarchical nature on the other (see Task ID 7 in Table 1). These
methods incorporate and integrate advances in the general fields of both multi-label classification (e.g., the development
of problem transformation versus algorithm adaptation methods, see Kassim et al., 2024) and hierarchical classification
(e.g., tree- versus DAG-based methods, see Ramírez-Corona et al., 2016 inter alia) that have taken shape over the last
20 years, cross-cut by concurrent advances in deep learning (see Liu et al., 2022, Li et al., 2024). In the next section, we
construct our multi-label hierarchical ATT&CK tagging models, building up along the task space strata in Table 1.
The recent Text-to-Text methods (Task ID 8 in Table 1) abstract away from the structure of the multi-label hierarchical
characterization of the output of the ATT&CK tagging task while attempting to preserve and extend the nature of the
output. Specifically, the output of the mapping, T, need not be a formal structure, but rather a text that encompasses the
information that a multi-label hierarchical structure would contain (e.g., the input document Dmaps to a set of tactics
and/or techniques, while Tmay also contain additional information). Many industry and research groups (see Branescu
et al., 2024, Fayyazi et al., 2024, Xu et al., 2024, Schwartz et al., 2025, Huang et al., 2024, Nir et al., 2025, Liu et al.,
2025) are moving in the direction of Text-to-Text classification for MITRE ATT&CK tagging. While we reserve plans
for extension of our multi-label hierarchical approach to Text-to-Text classification, we briefly comment on them in
Section 4.
4

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Figure 2: Tactic counts for the baseline cyber-intelligence text data set. Total (14405), with Defense Evasion (2642),
Discovery (2287), Command and Control (2072), Execution (1675), Persistence (1496), Credential Access (869),
Collection (820), Privilege Escalation (547), Initial Access (525), Resource Development (395), Impact (336), Lateral
Movement (265), Reconnaissance (240), Exfiltration (236).
3 Multi-label Hierarchical ATT&CK Tagging System Construction and Evaluation
Our full multi-label hierarchical classification model architecture for the MITRE ATT&CK tagging task is shown in
Figure 1. Rather than making a "top-down" architecture selection – an a priori choice of an AIML modeling architecture
– we took a "bottom-up" sequential approach in building up the architecture, progressing through the task space strata in
Table 1. Moreover, we motivated progression through the strata via the results of three experimental stages. The first
two experimental stages rely on a data set of 14405 general cyber-intelligence sentences each of which has a single
corresponding gold standard ATT&CK tactic and technique label. The data were compiled and curated by cybersecurity
specialist within JPMC to ensure data quality. The distribution of the data set by tactic is shown in Figure 2. The
third experimental stage relies on a second data set of 552 threat scenarios extracted from threat models produced by
cybersecurity specialists within JPMC for actual applications used within the bank. Of these data points, 486 have at
least one gold standard tactic label, while 66 have no tactic label (these are omitted from experiments). Of the 486 data
points that have tactic labels, 306 have a single ATT&CK tactic label2. The remaining 180 data points are multi-labeled.
We note the class imbalances and sparsity in the data sets and leave them as is to better replicate real-world data
conditions. Moreover, supporting experiments that accounted for the class imbalances showed similar results to our
main experiments.
Experimental Stage 1– We conducted a pilot study comparing the performance of a stochastic gradient descent
support vector machine (from scikit-learn, referred to as our baseline multiclass SGD model in the remainder of this
paper) against GPT-4o in multiclass tactic classification (Task ID 1 in Table 1). Specifically, we start with
ATT&CK:D7→T
limiting Tto a single ATT&CK tactic and restricting Dto sentences. The multiclass SGD model was selected based on
earlier experimental results showing that the model type outperformed other standard machine learning models on this
same multiclass task. Full parameter details of the model will accompany our public release. The selection of GPT-4o
for comparison was due to its being the latest release available to us when initiating our experiments. The temperature
is left at the default setting of 1.
2The tactic distribution is ’Initial Access’ (87), ’Impact’ (71) ’Collection’ (50), ’Defense Evasion’ (29), ’Exfiltration’ (29),
’Lateral Movement’ (24), ’Privilege Escalation’ (20), ’Credential Access’ (19), ’Discovery’ (15), ’Resource Development’ (14),
’Execution’ (8), ’Persistence’ (7), ’Reconnaissance’ (2), ’Command and Control’ (1).
5

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Table 2: Classification evaluation results for ATT&CK tactic tagging pilot study. Results show that a multiclass SGD
model generally outperforms GPT-4o in multiclass prediction for a given input cyber-intelligence sentence.
Evaluation Attribute Multiclass SGD Model GPT-4o
Accuracy0.81950.59
F10.77950.60
Accuracy parsed by Tactic
Defense evasion0.82720.6345
Discovery0.89690.6433
Persistence0.79030.5017
Initial access0.73070.6286
Collection0.84240.6402
Execution0.80550.5194
Lateral movement 0.59570.6226
Impact0.71420.6716
Command and control0.85610.5783
Credential access 0.76210.8046
Privilege escalation0.71300.2091
Reconnaissance 0.63880.6875
Resource development0.81170.5190
Exfiltration 0.58130.7234
The cyber-intelligence data was randomly split into a training set (80%) and a test set (20%) ensuring faithful
representation of the tactic distribution within each. The multiclass SGD model was trained on the former set
and evaluated on the later. All of the textual input to the multiclass SGD model was first transformed into vector
representations, in this case, TF-IDF for simplicity. The GPT-4o model was evaluated on the test set by saturating the
prompt below with the test sentences, one at a time, in their textual rather than vectorized forms.
Look at this cyber-intelligence text and label it with a mitre tag
from the selection provided to you in this message.
RETURN YOUR RESPONSE IN THE FOLLOWING JSON FORMAT WITHOUT MARKDOWN:
{{
"Tag": "YOUR MITRE TAG"
}}
IT IS EXTREMELY IMPORTANT THAT YOU RETURN THE EXACT "NAME" VALUE
FOR A MAXIMUM REWARD.
MITRE_TAGS:
* TA0006 - Credential Access * TA0002 - Execution * TA0003 - Persistence
* TA0001 - Initial Access * TA0005 - Defense Evasion * TA0007 - Discovery
* TA0008 - Lateral Movement * TA0009 - Collection * TA0010 - Exfiltration
* TA0043 - Reconnaissance * TA0040 - Impact * TA0042 - Resource Development
* TA0011 - Command and Control * TA0004 - Privilege Escalation
cyber-intelligence text:
{input-sentence}
Note that this GPT-4o tagging approach is technically Text-to-Text classification (Task ID 8 in Table 1), necessitating
that its output be normalized to ensure that generated tactic labels were comparable to the ground truth tactic labels.
Results in Table 2 show that our multiclass SGD model significantly outperformed GPT-4o over cyber-intelligence data
at the tactic level. Given the results, and the overall light footprint, share-ability, and extensibility of the multiclass SGD
model, we took it as the point of departure for our multi-label hierarchical classification system.
Experimental Stage 2– We next conducted a set of experiments to address three goals. The first goal concerned the
"problem transformation" for the multiclass SGD model, that is, making it behave more like a multi-label classification
model (moving up to Task ID 3 in Table 1). The second goal involved extending the transformed multi-label SGD
6

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Table 3: Multi-label classification evaluation results on the cyber-intelligence baseline data set. Results show that when
adopting a top- nlabeling method performance increases substantially at the tactic level. Moreover, the data hashing for
security does not impact model performance.
Evaluation AttributeMulti-label Classifier using
Multiclass SGD ModelsMulti-label Classifier using
Multiclass SGD Models (Hashing)
Topn= 3Accuracy 0.8264 0.8105
Tactic accuracy 0.9455 0.9427
Technique accuracy 0.8264 0.8105
Tactics correct 2724 2716
Techniques correct 2381 2335
Both correct predictions 2381 2335
Total predictions 2881 2881
Topn= 3Accuracy parsed by Tactic
Defense evasion 0.9424 0.9597
Discovery 0.9635 0.9700
Persistence 0.9537 0.9466
Initial access 0.8942 0.8654
Collection 0.9576 0.9455
Execution 0.9444 0.9306
Lateral movement 0.8723 0.8511
Impact 0.9107 0.8750
Command and control 0.9688 0.9736
Credential access 0.9351 0.9297
Privilege escalation 0.9130 0.9043
Reconnaissance 0.8333 0.8056
Resource development 0.9647 0.9412
Exfiltration 0.9070 0.8605
model to classification at the technique level, i.e., making it properly multi-label and hierarchical (moving up to Task ID
7 in Table 1). The third goal concerned ensuring the safety of data used to train the models in service of public release,
while not impeding model performance.
For the first goal, we simply modify the output of the multiclass SGD model to be the top npredicted tactics (choosing
n= 3 ), rather than the top 1 tactic (corresponding to (a) in Figure 1 with n= 3 ). The performance of this multi-label
ATT&CK tagging system is measured in terms of a standard subset operation. That is, suppose we are given an input
sentence Swith a ground truth tactic T, and let {T1, T2, T3}be the top 3 multi-label SGD tactic prediction. The
prediction for Sis considered correct if and only if {T} ⊆ {T 1, T2, T3}(known formally as top- naccuracy, where
n= 3 ). The multiclass SGD model was trained from scratch as a multiclass model over the cyber-intelligence data set,
but then evaluated using the-multi-label accuracy method. System performance for the multi-label accuracy evaluation
at the tactic level is shown in the top partition of Table 3, with accuracy reaching 94%, parsed out by tactic in the lower
partition. While the boost in performance for the SGD model is expected with the more general formulation of accuracy
relative to Experimental Stage 1, the improvement in results parsed out by tactic support the treatment of the tagging
task as multi-label rather than multiclass.
For the second goal, we extend our tactic-level SGD model to the technique level by simply training multiclass
SGD classifiers for the techniques associated with each tactic. That is, we first parse the cyber-intelligence data into
tactic-specific data sets, and then we train tactic-specific multiclass SGD models to make multiclass predictions over
the techniques for that tactic using a randomized 80-20 training-test split of the corresponding tactic-specific data sets.
The multi-label mapping at the technique level is again based on top 3 multiclass SGD classifier output (corresponding
to (b) in Figure 1 with m= 3 ). The final system prediction for a given input sentence is three tactics, each of which
is paired with three techniques (corresponding to (c) in Figure 1 with n=m= 3 ). Assuming the restriction of top
n= 3 predictions at both levels, multi-label hierarchical system accuracy is defined as follows. Let Sbe an input
sentence with ground truth tactic and technique labels (TaS, TeS). Let{T1, T2, T3}be the top 3 multi-label SGD tactic
prediction and for each Tilet{T1
i, T2
i, T3
i}be the technique predictions that follow. The tactic-technique predictions
are arranged into a set of nine pairs {(Ti, Tj
i)|for1≤i, j≤3} . The prediction for Sis considered correct if and only
if{(Ta S, TeS)} ⊆ {(T i, Tj
i)|for1≤i, j≤3} . Overall accuracy of the multi-label hierarchical system is shown in
7

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Table 4: Multi-label Tactic-Level Classification Evaluation Results for Threat Scenarios. Results show that the baseline
Multi-label SGD trained on cyber-intelligence data does not immediately generalize to threat scenario tagging, However,
the underlying model architecture is adaptable showing improvement with only a small amount of training data.
Evaluation AttributeMulti-label Classifier using
Multiclass SGD Models
Trained on Cyber-intel DataMulti-label Classifier using
Multiclass SGD Models trained on
Threat Scenario Data
Topn= 3Accuracy 0.410.66
Tactics correct 5488
Total predictions 132 132
Topn= 3Accuracy parsed by Tactic
Defense evasion 0.50.62
Discovery 0.420.57
Persistence0.750.25
Initial access 0.190.74
Collection 0.600.75
Execution 0.00 0.00
Lateral movement 0.16 0.16
Impact 0.540.87
Command and control 0.001.00
Credential access 0.70 0.70
Privilege escalation 0.000.40
Reconnaissance 0.00 0.00
Resource development 0.000.50
Exfiltration 0.62 0.62
the top partition of Table 3, reaching 82%. Moreover, the table shows that techniques are never predicted correctly
together with an incorrect tactic prediction (that is, "Techniques correct" is the same as "Both correct"), showing
the merit of the hierarchical approach. Specifically, the proper DAG structure of the MITRE ATT&CK hierarchy (a
technique can have multiple tactic parents) can be dealt with using multi-label hierarchical modeling.
For the third goal, the system includes a hashing option that encrypts the data used in SGD model training as a part
of the vectorization process. We tested our hashing option using MurmurHash3, though others are available through
scikit-learn. The hashed representations were also run through TfidfTransformer to ensure IDF weighting. We trained
two multiclass SGD models from scratch at the tactic level (corresponding to (a) in Figure 1 with n= 3 ) using the
cyber-intelligence data, one exposed to the standard TF-IDF vectors and the other exposed to the hashing-based vectors.
Both models were evaluated using the multi-label accuracy method for the tactic level described above. Results of
the overall comparison are shown in the top partition of Table 3, parsed out at the tactic level in the lower partition of
Table 3. Note that the encryption method does not significantly impact system performance. This allows us to share the
models out to the community with a high degree of security on the sensitive data used to train the models.
Experimental Stage 3– Our final set of experiments is two-fold, investigating how well the multi-label SGD models
worked on new data sets that differ in content from the general cyber-intelligence data on the one hand, and how
well the "problem transformed" multi-label SGD model type would perform on data points with actual gold standard
multi-labels. We note that both experiments were carried out on the threat scenario data set, which contains only
486 data points, sparse category counts, and labels only at the tactic level – limiting the interpretation of the results.
Moreover, since the data set contains true multi-label data points, we extend the multi-label accuracy definition for
tactics from Experimental Stage 2 as follows. Suppose we are given an input sentence Swith ground truth tactics
{TS
1, TS
2, . . . , TS
n}, and let {T1, T2, T3}be the top 3 multi-label SGD model tactic prediction. The number of correct
predictions for Sis the cardinality of the set intersection {TS
1, TS
2, . . . , TS
n} ∩ {T 1, T2, T3}. This formulation limits the
number of correct multi-label predictions to three for each sentence S, however there are only seven data points with
four or more multi-labels, so impact on performance is minimal.
The threat scenario data was randomly split into a training set ( ∼80%) and a test set ( ∼20%) ensuring faithful
representation of the tactic distribution within each as best as possible given the data sparsity. The split yielded a
test set consisting of 111 threat scenario sentences with a grand total of 132 tactic ground truth labels (due to the
test set containing multi-labeled threat scenarios). All of the textual data was again transformed into TF-IDF vector
representations. In the first experiment, the threat scenario test set was simply run through the multiclass SGD model
8

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
with tactic prediction accuracy computed using the defined set intersection cardinality. For the second experiment, a
multiclass SGD model was trained from scratch as a multiclass classifier on the training set and then evaluated as a
multi-label model on the test set using the defined set intersection accuracy. Results in Table 4 show that the baseline
Multi-label SGD trained on cyber-intelligence data does not immediately generalize to threat scenario tagging, However,
the underlying model architecture is adaptable, showing improvement with only a small amount of training data. While
more exploration is needed, the approach is in line with low-resource-sparse-data model building.
4 Review
In this technical note, we began by providing a general "task space" formulation of the MITRE ATT&CK text tagging
task for organizing existing AIML-related work and facilitating further developments. The formulation gave structure
to our "bottom-up" stage-wise construction of a baseline multi-label hierarchical tagging system for general cyber-
intelligence texts, as we leveled up through the task space strata based on experimental results. Our system construction
process eschewed the canonical "top-down" AIML modeling predispositions in favor of incorporating the "Best
Practices for MITRE ATT&CK Mapping" specified in CISA’s guide for analysts [Cybersecurity and Infrastructure
Security Agency, 2023]. During our system build-up we showed that our baseline models outperformed GPT-4o on
multiclass tactic prediction. We also showed how to re-use the baseline models to bootstrap modeling on new data sets –
exemplifying this re-use on a set of threat scenarios for financial applications produced by security specialists within
JPMC. We also implemented a model-performance-preserving hashing method in supporting our public release of a
tagging system for download and use by the security community.
We close with two main observations that came to light in producing this technical note. The first is, there are a great
many approaches to the MITRE ATT&CK text tagging task. Yet wide-spread adoption of any of these approaches
by security specialists seems to be rare, if not nonexistent. Usability of the models and systems in low-resource-
sparse-data settings (especially, if customizable) may likely be a prerequisite to system adoption, even when the more
advanced approaches have higher performance scores. Hence the reason for the public release of our system, which
is straightforward to set up and use. The second observation is, the cybersecurity community is deeply interested in
technological advancements, as they both impact and facilitate cybersecurity activities. Yet, to a large degree there is a
gap between the highly specialized activities of this community and a rich mathematical/technical literature, in AIML
and beyond, than can benefit community efforts. Hence our rigorous formulation of the ATT&CK tagging task and
mapping to existing cybersecurity works. We aim to further bridge this gap in later publications.
5 Acknowledgments
We thank the JPMorganChase Cybersecurity Community and appreciate your contributions and feedback.
This paper was prepared for informational purposes with contributions from the Cybersecurity and Technology Controls
organization of JPMorgan Chase & Co. This paper is not a product of the Research Department of JPMorgan Chase
& Co. or its affiliates. Neither JPMorgan Chase & Co. nor any of its affiliates makes any explicit or implied
representation or warranty and none of them accept any liability in connection with this paper, including, without
limitation, with respect to the completeness, accuracy, or reliability of the information contained herein and the potential
legal, compliance, tax, or accounting effects thereof. This document is not intended as investment research or investment
advice, or as a recommendation, offer, or solicitation for the purchase or sale of any security, financial instrument,
financial product or service, or to be used in any way for evaluating the merits of participating in any transaction.
References
Eric M Hutchins, Michael J Cloppert, Rohan M Amin, et al. Intelligence-driven computer network defense informed
by analysis of adversary campaigns and intrusion kill chains.Leading Issues in Information Warfare & Security
Research, 1(1):80, 2011.
The MITRE Corporation. MITRE ATT&CK, 2025. URLhttps://attack.mitre.org/.
Andrew Crossman, Andrew R. Plummer, Chandra Sekharudu, Deepak Warrier, and Mohammad Yekrangian. Auspex:
Building threat modeling tradecraft into an artificial intelligence-based copilot. In2025 IEEE Conference on Artificial
Intelligence (CAI), pages 1160–1167, 2025. doi:10.1109/CAI64502.2025.00201.
Cybersecurity and Infrastructure Security Agency. Best Practices for MITRE ATT&CK Mapping, 2023.
URL https://www.cisa.gov/sites/default/files/2023-01/Best%20Practices%20for%20MITRE%
20ATTCK%20Mapping.pdf.
9

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Marvin Büchel, Tommaso Paladini, Stefano Longari, Michele Carminati, Stefano Zanero, Hodaya Binyamini, Gal
Engelberg, Dan Klein, Giancarlo Guizzardi, Marco Caselli, Andrea Continella, Maarten van Steen, Andreas Peter, and
Thijs van Ede. Sok: Automated ttp extraction from cti reports – are we there yet? In34th USENIX Security Symposium,
Seattle, WA, USA, 2025. URL https://www.usenix.org/conference/usenixsecurity25/presentation/
buechel.
Sofia Della Penna, Roberto Natella, Vittorio Orbinato, Lorenzo Parracino, and Luciano Pianese. Cti-hal: A human-
annotated dataset for cyber threat intelligence analysis.arXiv preprint arXiv:2504.05866, 2025.
Md Tanvirul Alam, Dipkamal Bhusal, Le Nguyen, and Nidhi Rastogi. Ctibench: A benchmark for evaluating llms in
cyber threat intelligence. InAdvances in Neural Information Processing Systems 37. NeurIPS, 2024.
Gbadebo Ayoade, Swarup Chandra, Latifur Khan, Kevin Hamlen, and Bhavani Thuraisingham. Automated threat report
classification over multi-source data. In2018 IEEE 4th International Conference on Collaboration and Internet
Computing (CIC), pages 236–245. IEEE, 2018.
Benjamin Ampel, Sagar Samtani, Steven Ullman, and Hsinchun Chen. Linking common vulnerabilities and exposures
to the mitre att&ck framework: A self-distillation approach.arXiv preprint arXiv:2108.01696, 2021.
Fariha Ishrat Rahman, Sadaf Md Halim, Anoop Singhal, and Latifur Khan. Alert: A framework for efficient extraction
of attack techniques from cyber threat intelligence reports using active learning. InIFIP Annual Conference on Data
and Applications Security and Privacy, pages 203–220. Springer, 2024.
Paulo MMR Alves, PR Geraldo Filho, and Vinícius P Gonçalves. Leveraging bert’s power to classify ttp from
unstructured text. In2022 Workshop on Communication Networks and Power Systems (WCNPS), pages 1–7. IEEE,
2022.
Yizhe You, Jun Jiang, Zhengwei Jiang, Peian Yang, Baoxu Liu, Huamin Feng, Xuren Wang, and Ning Li. Tim: threat
context-enhanced ttp intelligence mining on unstructured threat data.Cybersecurity, 5(1):3, 2022.
Nanda Rani, Bikash Saha, Vikas Maurya, and Sandeep Kumar Shukla. Ttphunter: Automated extraction of actionable
intelligence as ttps from narrative threat reports. InProceedings of the 2023 australasian computer science week,
pages 126–134, 2023.
Nanda Rani, Bikash Saha, Vikas Maurya, and Sandeep Kumar Shukla. Ttpxhunter: Actionable threat intelligence
extraction as ttps from finished cyber threat reports.Digital Threats: Research and Practice, 5(4):1–19, 2024.
Otgonpurev Mendsaikhan, Hirokazu Hasegawa, Yukiko Yamaguchi, and Hajime Shimada. Automatic mapping of
vulnerability information to adversary techniques. InThe Fourteenth International Conference on Emerging Security
Information, Systems and Technologies SECUREWARE2020, 2020.
Aditya Kuppa, Lamine Aouad, and Nhien-An Le-Khac. Linking cve’s to mitre att&ck techniques. InProceedings of
the 16th International Conference on Availability, Reliability and Security, pages 1–12, 2021.
Octavian Grigorescu, Andreea Nica, Mihai Dascalu, and Razvan Rughinis. Cve2att&ck: Bert-based mapping of cves to
mitre att&ck techniques.Algorithms, 15(9):314, 2022.
Valentine Legoy, Marco Caselli, Christin Seifert, and Andreas Peter. Automated retrieval of att&ck tactics and
techniques for cyber threat reports.arXiv preprint arXiv:2004.14322, 2020.
Ghaith Husari, Ehab Al-Shaer, Mohiuddin Ahmed, Bill Chu, and Xi Niu. Ttpdrill: Automatic and accurate extraction of
threat actions from unstructured text of cti sources. InProceedings of the 33rd Annual Computer Security Applications
Conference, ACSAC ’17, page 103–115, New York, NY , USA, 2017. Association for Computing Machinery. ISBN
9781450353458. doi:10.1145/3134600.3134646. URLhttps://doi.org/10.1145/3134600.3134646.
Kiavash Satvat, Rigel Gjomemo, and VN Venkatakrishnan. Extractor: Extracting attack behavior from threat reports.
In2021 IEEE European Symposium on Security and Privacy (EuroS&P), pages 598–615. IEEE, 2021.
Zhenyuan Li, Jun Zeng, Yan Chen, and Zhenkai Liang. Attackg: Constructing technique knowledge graph from cyber
threat intelligence reports. InEuropean Symposium on Research in Computer Security, pages 589–609. Springer,
2022.
Md Tanvirul Alam, Dipkamal Bhusal, Youngja Park, and Nidhi Rastogi. Looking beyond iocs: Automatically extracting
attack patterns from external cti. InProceedings of the 26th international symposium on research in attacks, intrusions
and defenses, pages 92–108, 2023.
Mohammed Awal Kassim, Herna Viktor, and Wojtek Michalowski. Multi-label lifelong machine learning: A scoping
review of algorithms, techniques, and applications.IEEE Access, 12:74539–74557, 2024.
Mallinali Ramírez-Corona, L Enrique Sucar, and Eduardo F Morales. Hierarchical multilabel classification based on
path evaluation.International Journal of Approximate Reasoning, 68:179–193, 2016.
10

Constructing Multi-label Hierarchical Classification Models for MITRE ATT&CK Text Tagging
Chenjing Liu, Junfeng Wang, and Xiangru Chen. Threat intelligence att&ck extraction based on the attention
transformer hierarchical recurrent neural network.Applied Soft Computing, 122:108826, 2022. ISSN 1568-4946.
doi:https://doi.org/10.1016/j.asoc.2022.108826. URL https://www.sciencedirect.com/science/article/
pii/S1568494622002289.
Lingzi Li, Cheng Huang, and Junren Chen. Automated discovery and mapping att&ck tactics and techniques for
unstructured cyber threat intelligence.Computers & Security, 140:103815, 2024.
Ioana Branescu, Octavian Grigorescu, and Mihai Dascalu. Automated mapping of common vulnerabilities and
exposures to mitre att&ck tactics.Information, 15(4):214, 2024.
Reza Fayyazi, Rozhina Taghdimi, and Shanchieh Jay Yang. Advancing ttp analysis: Harnessing the power of large
language models with retrieval augmented generation. In2024 Annual Computer Security Applications Conference
Workshops (ACSAC Workshops), pages 255–261. IEEE, 2024.
Ming Xu, Hongtai Wang, Jiahao Liu, Yun Lin, Chenyang Xu Yingshi Liu, Hoon Wei Lim, and Jin Song Dong. Intelex:
A llm-driven attack-level threat intelligence extraction framework.arXiv preprint arXiv:2412.10872, 2024.
Yuval Schwartz, Lavi Ben-Shimol, Dudu Mimran, Yuval Elovici, and Asaf Shabtai. Llmcloudhunter: Harnessing llms
for automated extraction of detection rules from cloud-based cti. InProceedings of the ACM on Web Conference
2025, pages 1922–1941, 2025.
Yi-Ting Huang, R Vaitheeshwari, Meng-Chang Chen, Ying-Dar Lin, Ren-Hung Hwang, Po-Ching Lin, Yuan-Cheng
Lai, Eric Hsiao-Kuang Wu, Chung-Hsuan Chen, Zi-Jie Liao, et al. Mitretrieval: Retrieving mitre techniques from
unstructured threat reports by fusion of deep learning and ontology.IEEE Transactions on Network and Service
Management, 21(4):4871–4887, 2024.
Daniel Nir, Florian Klaus Kaiser, Shay Giladi, Sapir Sharabi, Raz Moyal, Shalev Shpolyansky, Andres Murillo, Aviad
Elyashar, and Rami Puzis. Labeling network intrusion detection system (nids) rules with mitre att&ck techniques:
Machine learning vs. large language models.Big Data and Cognitive Computing, 9(2):23, 2025.
Xiaoqun Liu, Jiacheng Liang, Qiben Yan, Jiyong Jang, Sicheng Mao, Muchao Ye, Jinyuan Jia, and Zhaohan Xi. CyLens:
Towards Reinventing Cyber Threat Intelligence in the Paradigm of Agentic Large Language Models.arXiv preprint
arXiv:2502.20791, 2025.
11