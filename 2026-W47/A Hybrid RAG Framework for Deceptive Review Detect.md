# A Hybrid RAG Framework for Deceptive Review Detection: Integrating RoBERTa-LoRA, FAISS-Based Retrieval, and LLM Reasoning

**Authors**: Andrei Kostin, José Ángel Diaz-Garcia, Karel Gutiérrez-Batista

**Published**: 2026-11-18 08:39:46

**PDF URL**: [https://arxiv.org/pdf/2611.24049v1](https://arxiv.org/pdf/2611.24049v1)

## Abstract
The growth of deceptive online opinions poses a critical risk of misinformation and fraud. This paper proposes a Retrieval-Augmented Generation (RAG) system to automatically distinguish spam reviews. The architecture integrates RoBERTa with Low-Rank Adaptation (LoRA) classification, FAISS semantic vector retrieval, and a Large Language Model (LLM) for final reasoning. The system was evaluated across balanced and severely imbalanced real-world scenarios. Results demonstrate that five epochs of fine-tuning enable robust domain transfer, achieving an F1-score of 0.90 in cross-domain evaluation, outperforming baselines (OpenAI Detector,F1 = 0.82) and demonstrating competitive performance against specialized in-domain SOTA models (fakeRoBERTa,F1 = 0.97). The proposed architecture provides evidence-backed explainability and statistically significant precision gains while maintaining comparable overall detection performance.

## Full Text


<!-- PDF content starts -->

A Hybrid RAG Framework for Deceptive Review
Detection: Integrating RoBERTa-LoRA, FAISS-Based
Retrieval, and LLM Reasoning
Andrei Kostina,∗, José Ángel Díaz Garcíab,1, Karel Gutiérrez-Batistac,1
aDepartment of Computer Science, University of Jaén, Campus Las
Lagunillas, Jaén, 23071, Spain
bDepartment of Computer Languages and Systems, University of Granada, Cortadura del
Valle, s/n, Ceuta, 51001, Spain
cDepartment of Computer Science and Artificial Intelligence, University of
Granada, Cortadura del Valle, s/n, Ceuta, 51001, Spain
Abstract
The growth of deceptive online opinions poses a critical risk of misinformation
and fraud. This paper proposes a Retrieval-Augmented Generation (RAG)
system to automatically distinguish spam reviews. The architecture integrates
RoBERTa with Low-Rank Adaptation (LoRA) classification, FAISS semantic
vector retrieval, and a Large Language Model (LLM) for final reasoning. The
system was evaluated across balanced and severely imbalanced real-world
scenarios. Results demonstrate that five epochs of fine-tuning enable robust
domain transfer, achieving an F1-score of0 .90in cross-domain evaluation,
outperforming baselines (OpenAI Detector, F1= 0.82) and demonstrating
competitive performance against specialized in-domain SOTA models (fakeR-
oBERTa, F1= 0.97). The proposed architecture provides evidence-backed
explainability and statistically significant precision gains while maintaining
comparable overall detection performance.
Keywords:Deceptive Opinion Detection, Retrieval-Augmented Generation,
Low-Rank Adaptation, Large Language Model, Semantic Search, Explainable
AI
∗Corresponding author
Email addresses:ak000171@red.ujaen.es(Andrei Kostin),
joseangeldiazg@ugr.es(José Ángel Díaz García),karel@decsai.ugr.es(Karel
Gutiérrez-Batista)
1These authors contributed equally to this work.
arXiv:2611.24049v1  [cs.CL]  18 Nov 2026

1. Introduction
The rapid growth of e-commerce has radically changed customer habits,
making online reviews one of the main factors influencing business reputation
and consumer decisions in the digital world. Nowadays, e-commerce platforms
(for example, Amazon, Walmart) and special services (such as Yelp, TripAdvi-
sor) generate enormous amounts of data. By the end of 2022, Yelp’s base had
more than 265 million reviews [ 1]. According to recent BrightLocal reports,
97% of consumers read reviews before making a purchase, and 41% of users
consult them regularly [ 2]. The economic impact of these results is enormous;
for example, a Harvard Business School study found that a one-star increase
in a restaurant’s Yelp rating leads to a 5–9% increase in revenue [3].
Finally, the significant financial impact of these numbers has led to the
proliferation of fake reviews and an increased risk of misinformation. Some
market participants follow a strategy known as astroturfing in which they
use fraudulent reviews to artificially inflate their own reputation and to
discredit competitors [ 4]. According to estimates by analytical resources, the
proportion of fake reviews on major platforms such as Amazon can reach
30% [5]. This problem is getting worse with the appearance of modern
large language models (LLMs) capable of mass-generating synthetic texts
that are linguistically indistinguishable from those written by real people
[6]. As a result, traditional filters and manual moderation are losing their
effectiveness, requiring the development of a new generation of intelligent
automated systems.
Recent advances in expert systems have further demonstrated the effec-
tiveness of integrating RAG with adaptive decision mechanisms for content
verification. For instance, in [ 7], the authors introduced SWIFT, a self-
reflective knowledge-switching framework that dynamically triggers external
retrieval only when parametric confidence drops below a threshold, signifi-
cantly optimising inference efficiency. To mitigate the risk of hallucinated
reasoning during arbitration, Yang et al. [8] proposed an agentic framework
designed for evidence-inconsistency filtering before feeding context into LLMs.
Additionally, Gao et al. [ 9] demonstrated with SEL-PMF that combining
Parameter-Efficient Fine-Tuning with LLM-generated background evidence
substantially enhances semantic representation in sparse textual contexts.
Building upon these expert architectural principles, our work proposes a
threshold-gated hybrid RAG system specifically engineered for deceptive
online review detection under domain shift and class imbalance. Our research
2

is guided by the following research questions:
•RQ1: How does retrieval-augmented generation affect the precision-
recall trade-off in deceptive review detection?
•RQ2: To what extent does LoRA enable efficient cross-domain adapta-
tion in deceptive review detection?
•RQ3: How robust is the proposed architecture under severe class
imbalance?
•RQ4: Does retrieval-based reasoning improve explainability without
degrading performance?
The contributions of this work are as follows:
•A hybrid RAG architecture that combines a supervised classifier
(RoBERTa-LoRA), a semantic retrieval module (FAISS) that returns
both similarity scores and the actual text of similar past reviews, and
a large language model (Llama 3.1) that uses all of this to make and
explain the final decision.
•A classifier that can adapt to a new domain and new class imbalance in
only five additional training epochs, instead of retraining from scratch,
and it was tested across three conditions: a balanced in-domain set,
a severely imbalanced set, and a cross-domain set with AI-generated
fakes.
•An ablation study that measures how much each part of the system
(classifier, retrieval, LLM reasoning) actually contributes to the final
result.
•Giving the language model the actual text of retrieved similar reviews,
not just distance scores, allows it to produce a short written justification
for each decision, giving the system a level of explainability that the
baseline classifier and simple k-NN approaches do not have.
From an operational perspective, automated moderation in modern
e-commerce platforms cannot rely solely on black-box probabilistic scoring.
Incorrectly penalizing legitimate seller reviews or missing sophisticated
computer-generated spam introduces severe reputational and economic
3

liabilities. To overcome this limitation, our hybrid architecture positions
the RAG framework not merely as a binary classifier, but as an auditable
expert decision-support system for online content moderation. By pairing
high-precision vector retrieval with LLM arbitration, the system provides
human safety teams with an inspectable evidence chain, transforming opaque
machine learning predictions into a transparent, expert-assisted auditing
workflow.
The rest of the paper is organised as follows. Section 2 provides an
overview of related work in the field of spam detection and the application of
deep learning. Section 3 describes the proposed system architecture in detail,
including preprocessing steps, the configuration of the FAISS semantic search
module, classifier operation, and the LLM inference mechanism. Section 4
details the LLM decision layer, prompt design, and confidence-threshold
routing mechanism. Section 5 presents the experimental design, quantitative
results, and ablation analysis. Section 6 discusses the findings in relation to
the research questions. Finally, Section 7 summarises the study and outlines
key directions for future work.
2. Related works
Over the past decade, researchers have been actively studying the problem
of detecting deceptive opinion spam. The evolution of approaches in this field
can be divided into three key stages: the use of traditional machine learning,
the application of deep learning models based on the transformer architecture,
and the implementation of modern systems augmented with external search,
such as RAG.
2.1. Classic Machine Learning
Early research in the field of spam review detection relied on lexical
analysis, n-grams, and psycholinguistic characteristics. A fundamental work
in this area was the study by Ott et al., which introduced the first large-scale
annotated dataset (the Deceptive Opinion Spam Corpus), consisting of 400
genuine and 400 fake hotel reviews [ 10]. The fraudulent reviews were written
by experts. The authors showed that humans can detect generated deception
only at the level of random guessing (about 50% accuracy), whereas automatic
text-classification algorithms achieve nearly 90% accuracy, confirming the
critical need to automate this process.
4

On the other hand, Jindal and Liu conducted an analysis of 5.8 million
reviews on the Amazon platform, focusing on identifying duplicate and near-
duplicate reviews written by the same reviewers or about the same products
as key indicators of spam activity [ 11]. Their approach demonstrated the
importance of combining text analysis with the study of user behaviour
patterns. However, classical methods (such as SVM, Logistic Regression, or
Random Forest) based on TF-IDF features have limited ability to capture
complex semantics and are vulnerable to modern generative models used by
spammers.
2.2. Deep Learning and Transformers
With the development of neural network technologies, researchers have
focused on deep learning models, such as CNNs and RNNs, and transformer-
based architectures (e.g., BERT, RoBERTa), which are capable of extracting
semantic patterns from text without the need for manual feature engineering
[12,13]. In particular, Raza et al. applied BERT with progressive training
and achieved strong results (95.3% accuracy, 0.953 F1-score) on the WELFake
dataset, which includes more than 72,000 articles, confirming the superiority
of transformers over traditional machine learning [14].
At the same time, Papageorgiou et al. have noted in their research on the
use of LLMs for dealing with fake content that modern neural networks play
a dual role [ 6]. In particular, they can act as high-quality synthetic spam
or, on the other hand, serve as a powerful tool for detecting it. However,
classic transformers are limited by the knowledge acquired during pretraining,
tend to overfit under severe class imbalance, and function as “black boxes”,
providing no evidence to support their decisions.
2.3. RAG architecture
To overcome the limitations of a lack of transparency and the external
knowledge used by models, RAG architectures have been actively researched
in recent years. The integration of RAG allows systems to dynamically
retrieve relevant text examples from external knowledge bases and use them as
additional context [ 15]. These factors improve the accuracy and explainability
of the final responses in tasks requiring fact verification or fraudulent detection.
This approach has already demonstrated high efficiency. In particular, Singal
et al. developed a system that extracts evidence phrases from a knowledge
base, which were then transferred to an LLM using context-aware training.
This approach improved the performance of baseline models by 22% with a
5

minimal number of training examples [ 16]. Also, the VeraCT Scan system
successfully integrated semantic web search for fake news classification using
GPT-4 Turbo and Llama-2 13B models, demonstrating state-of-the-art results
on the BuzzFeedNews and PolitiFact datasets [17].
Despite the achieved progress of RAG systems in the field of news fact-
checking, the application of this architecture to the detection of fake reviews
remains largely unexplored. Unlike fact-checking, where falsity is determined
by discrepancies with objective reality, fake reviews are identified through
linguistic anomalies, triteness, and contextual inconsistencies.
2.4. Existing Benchmarks and Baseline Models
A growing challenge is the detection of AI-generated spam reviews. Salmi-
nen et al. created a cross-domain dataset of 40,000 Amazon reviews, half of
which were generated using two language models, ULMFiT and GPT-2. In
their study, baseline classifiers, such as NBSVM and the specialised fakeR-
oBERTa model, were trained and tested within a single domain (in-domain),
achieving accuracies of up to 97%, while humans were only able to solve the
task 55% of the time [18].
The proposed architecture fills an existing research gap by adapting the
RAG methodology to the task of filtering online reviews. Unlike the baseline
models mentioned in the last study, our system is designed for effective cross-
domain transfer and resilience to class imbalance. The proposed system also
offers high explainability and decision transparency through the extraction of
semantic evidence.
For clarity, Table 1 summarises the key studies in spam detection and con-
tent verification. As shown in this table, although existing transformer-based
models achieve high accuracy in familiar domains, they lack explainability
and require resource-intensive retraining. The proposed RAG architecture
solves these issues.
6

Table 1: Summary of studies in the field of spam and fake content detection (Related
Work)
Author Description Dataset Outcomes Research Gap
Ott et al.
(2011) [10]First systematic study on
deceptive reviews; utilisation
of n-grams and
psycholinguistic analysis800 hotel
reviews of 20
Chicago hotels
(400 genuine,
400 fake)Algorithm
accuracy
89.8%,
whereas
humans
recognise
fakes only
50% of the
time.It needs to adapt to new
domains; vulnerable to
modern spammers.
Jindal &
Liu (2008)
[11]The study categorises review
spam into three types
(untruthful opinions,
brand-only reviews, ads).
They used duplicate and
near-duplicate review
patterns to identify and
analyse deceptive opinions
with a logistic regression
model5.8 million
reviews from
the Amazon
platformFailure of
text-only
approaches
(they obtained
a poor AUC
of only 63%)It has a limitation in finding
duplicates using a
2-gram-based content
comparison with the Jaccard
distance, and fails to capture
subtle semantic patterns
Raza et al.
(2025) [14]Application of BERT
architecture with progressive
episode-based training
strategy to capture linguistic
nuances of fake newsWELFake
dataset (72,134
articles).Accuracy:
95.3%,
F1-score:
0.953.The current model lacks
transparent explainability in
its decision-making. The
WELFake dataset has
potential imperfections
Lopez-Joya
et al.
(2025) [15]They proposed the
VERIFAID RAG system for
fake news detection. The
system was built on an
automatically generated,
dynamically growing
database. This design
provides traceable evidence to
support the final classificationVERIFAID-
dataset (33,337
entries)On
benchmark
datasets like
GossipCop,
CoAID, and
PolitiFact, the
VERIFAID
framework
achieved
competitive
F1-Macro and
accuracy
scoresThe framework is designed
specifically to detect general
misinformation, fake news,
and political or health-related
rumours, not tailored to
deceptive review spam
Singal et al.
(2024) [16]Automated fact-checking
system based on RAG (using
Mixtral 8x22B) and
in-context learning (Few-Shot
ICL) to provide
evidence-backed predictionsAVeriTeC
dataset (4,568
real-world
claims)Improved
baseline
metrics by
22% with a
minimal
number of
training
examplesThe extracted QA evidence is
susceptible to LLM
hallucinations. Tuned
exclusively for fact-checking,
not for review analysis
Niu et al.
(2024) [17]The VeraCT RAG system
integrating web search and
optimised query generation
for transparent decision
reasoning using GPT-4 and
Llama-2BuzzFeedNews,
Fakenewsnet,
LLMFake,
PolitiFact-
Snopes-2024
and
FakeNews2024
datasetsAchieved
state-of-the-
art results in
news
verification
tasksComplex web search
architecture is not adapted
for isolated review platforms
7

(A) Training & Indexing Phase
(B) Online Inference
(C) Domain AdaptationTraining Data
(OpSpam + Yelp
Labeled Reviews)
Vector Knowledge Base
(YelpZIP + YelpNYC)Embedding Extraction
(RoBERTa-base)RoBERTa-LoRA
Classifier
FAISS Indextrain
index
Unseen ReviewxFeature Extractor
e(x)+ RoBERTa-LoRApbase∈[τlow,τhigh]?k-NN Retrieval (FAISS)
closest distance,
fake ratio,
top-3 examplesLLM ArbitrationFinal Predictionˆy
+ Justification
Direct Label
ˆy=I(p base≥0.5)pbaseYes (Uncertain)
No (Confident)Retrieval Summarytrained LoRA weights
New-Domain Data
(other product
categories
or imbalance)LoRA Fine-Tuning
(5 epochs)domain shiftupdated
LoRA weights
Figure 1: Diagram of the proposed Hybrid RAG architecture.
3. System architecture
3.1. Overview
The proposed system for fake review detection is based on a three-level
hybrid RAG architecture. As compared with traditional approaches, our
architecture combines three independent prediction sources: a supervised
classifier based on RoBERTa and LoRA adapters, which gives a primary
spam probability; a semantic retrieval module based on a FAISS index, which
extracts some examples as evidence with retrieval confidence metrics; and,
finally, a decision layer based on an LLM (Llama-3.1-8B) for final decision
and logical inference based on all collected signals from the two previous
system layers.
As shown in Fig. 1, the proposed architecture has two pipelines plus a
domain-adaptation step. Firstly, the training pipeline includes data prepro-
cessing, embedding extraction, building a FAISS index based on labelled Yelp
HoReCa reviews, and training the RoBERTa-LoRA classifier on annotated
data. Second, the inference pipeline for new data vectorises inputs, obtains a
probability from the classifier, retrieves confidence metrics for nearest neigh-
8

bours from the knowledge-base FAISS index, and lets the LLM make the final
decision.
3.2. Datasets
In this study, different datasets were used for classifier training, knowledge
base construction, and diverse system evaluation. Every dataset plays a
strictly defined function in the architecture. The general descriptions and
roles of all datasets are presented in Table 2.
Table 2: Summary of datasets and their roles in the proposed system
Dataset Role in System Size and Details
Deceptive
Opinion Spam
Corpus v1.4 [10]Classifier training
(first dataset1)1,600 hotel reviews; 800 genuine / 800
fake. Split 70/30 (1,120 train / 480
validation).
Yelp Fake Review
Dataset [19]Classifier training
(second dataset1)9,929 train / 2,483 test instances.
Restaurant reviews with realistic class
skew based on Yelp’s filter.
YelpNYC +
YelpZIP [20]FAISS Knowledge
Base768,683 labelled reviews by Yelp
platform filter (86% genuine / 14%
fake). Used exclusively to build a
large-scale semantic index.
In-Domain Data
(Test set split)Evaluation
(Exp. 1: Baseline)1,309 instances from the combined
training data. Stratified split (70%
genuine / 30% fake) with fixed seed.
Hotel Reviews
[21]Evaluation
(Exp. 2 with
imbalance)Sample of 3,000 reviews. Severe class
imbalance: 2,700 genuine (90%) / 300
fake (10%).
Cross-domain
Reviews [18]Evaluation
(Exp. 3 with
cross-domain
reviews)Sample of 3,000 multi-domain reviews
(50% genuine / 50% LLM-generated
fakes).
1Note: After preprocessing, the annotated datasets (Opinion Spam and Yelp) were
combined into a single training corpus with a 70/30 stratified split to preserve class
proportions.
9

3.3. Preprocessing
During preprocessing of the selected datasets, a unified data normalisation
pipeline was implemented. Specifically, this includes basic cleaning of noise
symbols and the removal of rows containing empty opinion text and very
short reviews that could impair the performance of the spam detection model.
Likewise, words were not converted to lowercase to avoid losing potential
context in fraudulent or genuine opinions—such as the use of Caps Lock to
express emotions.
The decision to preserve the case of text characters is crucial from a
linguistic perspective of a review, since the use of uppercase characters
(“AMAZING”, “BEST EVER”) may be related to the underlying context of
the review as a whole and must be analyzed by the system.
The specific stages of data preprocessing include:
1. Removal of HTML tags, special characters, and null values.
2.Normalisation of whitespace: replace multiple spaces, tabs, and line
breaks with a single space.
3.Removal of text shorter than 30 characters, as it lacks sufficient linguistic
context for initial classification.
4.Removal of duplicate reviews to avoid overfitting on repetitive examples.
5. Label normalisation.
To train the supervised classifier, the Opinion Spam Corpus v1.4 and
Yelp Fake Reviews Dataset were preprocessed and merged into a single
training corpus. This combined dataset was partitioned using a stratified
70%/30% train-validation split. Stratification ensured that the class ratios
are maintained in both sets, which is crucial for an accurate comparison of
the results.
For the vector knowledge base, the YelpNYC and YelpZIP were combined
into a unified large-scale corpus. The text duplicates were removed from the
combined corpus to maintain index integrity, prevent memory redundancy,
and eliminate bias during FAISS k-NN semantic retrieval.
3.4. Feature Extraction
The next step after data preprocessing is to obtain a vector representation
of the text. To do this, the pre-trained RoBERTa (Robustly Optimised BERT
Pretraining Approach) model was used, which is better suited for detecting
fake opinions because it is more robust to context preservation and variability.
10

It also significantly improves results compared to more basic methods, such
as TF-IDF, as will be demonstrated later.
Specifically, 768-dimensional embeddings from the roberta-base model
were extracted from the merged training dataset (Yelp + Deceptive Opinion
Spam Corpus) to train the classifier, and were also obtained from the large
combined corpus (YelpNYC + YelpZIP) to build the FAISS index, which will
be described in the following section.
RoBERTa was selected as the base encoder model for several reasons. First,
compared to the original BERT, RoBERTa was trained for a longer period
and with more data, which improves the quality of contextual representations
[13]. Additionally, RoBERTa uses dynamic masking during training, which
increases its ability to adapt to the input text.
3.5. Supervised Classifier (RoBERTa-LoRA)
The initial review classification is performed using a RoBERTa model
fine-tuned with the Parameter-Efficient Fine-Tuning (PEFT) library [ 22].
The Low-Rank Adaptation (LoRA) mechanism was applied to reduce the
computational costs of retraining a large model [ 23]. LoRA freezes the model’s
initial weightsW 0and applies updates via low-rank decomposition matrices
(AandB). Mathematically, the forward pass is expressed as:
h=W 0x+ ∆Wx=W 0x+BAx(1)
where,B∈Rd×r,A∈Rr×k, andr≪min(d,k).
The adapter’s hyperparameters were set as follows: rank r= 8,α= 16,
anddropout = 0.1. It was trained for 10 epochs using the AdamW optimiser
with weight decay regularisation to prevent overfitting. A CustomLossTrainer
module for fine-tuning was developed with a weighted cross-entropy loss
function to effectively handle imbalanced real-world data. The class weights
(wk) were calculated dynamically using the following formula:
wk=N
K·n k(2)
where,Nis the total number of examples, Kis the number of classes, and nk
is the number of examples in class k. For example, with a 90/10 imbalance,
the model penalises an error on the minority class about 9 times more severely.
A weighted Cross-Entropy Loss function, mathematically, is calculated as:
11

Lweighted =−1
NN/summationdisplay
j=1[w1tjlog(p j) +w 0(1−t j) log(1−p j)](3)
whereNis the total number of objects in the sample, tjis the target label
for binary classification, pjis the model’s prediction that the opinion belongs
to class 1 (false), and w1andw0are the weights assigned to fake and genuine
reviews respectively.
During research, a linear interpolation of the classifier probabilities and
the k-NN probabilities from FAISS ( Pfinal=λPclf+ (1−λ)PkNN) was tested,
but this approach was rejected because it did not outperform the standalone
LoRA classifier in the F1 metric.
3.6. Retrieval Module (FAISS)
To provide the system with an evidence-search mechanism, a verified
vector database from the combined Yelp dataset (YelpNYC + YelpZIP) was
constructed. This base was implemented using the FAISS (Facebook AI
Similarity Search) library [ 24]. The IndexFlatL2 index was used, which
performs an exact search for nearest neighbours based on Euclidean distance
inR768dimensions. Embeddings were generated in batches of 512 examples
to optimise memory usage.
During inference, after receiving the input data, the system searches for
the top-k nearest neighbours. For each query, the system retrieves k= 20
nearest neighbours. This value was selected empirically to balance stability
and retrieval relevance. Too small values ( k= 3) give unstable results,
while too large values ( k= 100) increase the number of irrelevant outliers
and degrade the retrieval signal. Since the knowledge base is fully labelled,
the retrieval module extracts quantitative similarity metrics and qualitative
textual evidence, some of which are formatted into the retrieval summary for
the LLM:
•Distance to the nearest neighbor( top1_score ): the minimum
Euclidean distance dmin(p,q) =min(/radicalig/summationtextn
k=1(pk−qk)2). A low value
indicates that there is an almost identical example in the knowledge
base, which increases the reliability of the contextual signal.
•Mean distance to neighbours( mean_topk_score ): the average dis-
tance of the k neighbours found, dmean =1
k/summationtextk
i=1di. It demonstrates the
12

overall density of the neighbourhood through a query in the knowledge
base space.
•Difference between the average distance and the nearest neigh-
bor( score_gap ): The difference between the two previously calculated
measures is∆ d=dmean−dmin. A large difference indicates that the
first neighbour is significantly closer than the others. It means that the
query has entered a sparse area with a similar example. A low result
indicates a dense cluster of similar examples.
•Deceptive Ratio in Top-k Neighbors( fake_ratio ): the proportion
of verified deceptive reviews among the k retrieved neighbours, which
is calculated as Fake Ratio =1
k/summationtextk
i=1yi, y i∈{0,1}, whereyi= 1
represents a fake review and yi= 0represents a genuine review. A ratio
close to1.0signals that the target review falls into a dense cluster of
spam opinions.
•Top-3 Similar Textual Examples: The raw review texts of the 3
closest neighbours to provide the LLM with direct contextual evidence
for comparative analysis.
3.7. Inference and Domain Adaptation
One of the system’s key innovations is the domain adaptation during the
inference layer. When the system faces input data from a new domain (in
our case, a transition occurs from hotel/restaurant reviews to other product
categories), the RoBERTa-LoRA classifier undergoes additional retraining
for 5 epochs. Thanks to LoRA’s computational efficiency, this step allows
the model to quickly adapt to a new feature distribution and specific class
imbalances.
4. LLM Decision Layer
The locally deployed large language model Llama-3.1-8B-Instruct (quan-
tised to 8-bit, Q8) was used as the final decision-making layer [ 25]. We chose
this model for its strong instruction-following capabilities and high reasoning
quality, while keeping it computationally efficient for local inference. The
Llama model with 8 billion parameters can correctly handle strictly struc-
tured prompts and return responses in a specified format. We developed a
specialised two-part prompt to ensure strict output formatting and minimise
13

hallucinations. This approach includes aSystem Messageand aHuman
Template.
The prompt engineering process for the system message included several
iterations. In the initial phase, the model’s specific function and output format
were specified. Based on the results of the classification error analysis, clear
rules were added to the prompt that prohibit the model from interpreting
the positivity, length, or level of detail of the text as indications of fakeness.
Finally, arigorousoutputschemawithspecifiedfieldswasintroduced, ensuring
a deterministic syntactic analysis of the model’s responses. The text of the
system message is provided below:
You are an expert in deceptive opinion spam detection.
Classify the target review:
- 1 = fake/spam
- 0 = real
Rules:
- Final decision is yours.
- Classifier result is supportive, not decisive.
- Retrieval is highly supportive.
- Generic, templated, promotional, repetitive text -> LABEL 1.
- Grounded, situational, imperfect, experience-based text -> LABEL
0.
- Do not treat positivity, length, detail, or emotion as fake by
itself.
- Output only in the exact format below.
Output exactly:
LABEL: 0|1
REASONING: [Your reasoning, explicitly mentioning the retrieved
examples, fake ratio, and distance metric, and prioritizing them
over the classifier result]
The prompt positions the language model as an expert in spam detec-
tion. The algorithm instructs the LLM to summarise the target review’s
textual characteristics with external evidence, treating the classifier result
as supportive and the retrieval metrics as highly supportive. This was done
to ensure that noise in the extracted data would not distort the model’s
internal logic. The strict output format makes it easy to retrieve and analyse
responses from the LLM. Clear rules guided the model to identify formulaic
or promotional phrasing as spam, while treating experience-grounded details
as genuine. The strict output requires a binary label, and a brief explanation
14

ensures deterministic response parsing and explainable decision-making.
The Human Template is used to dynamically transmit the extracted data
(FAISS retrieval summary including distance metric, neighbour fake ratio,
and top-3 closest examples from the verified vector database, and classifier
probability), as well as the review text for each specific query. Its text is
provided below:
RETRIEVAL SUMMARY:
- Closest distance: {top1_score} (Lower is more similar to memory)
- Fake Ratio in Top-{neighbor_count} neighbors: {fake_ratio}
(1.0 = all closest are fake, 0.0 = all real)
TOP 3 SIMILAR EXAMPLES FROM VERIFIED DATABASE:
1. {retrieval_example_1}
2. {retrieval_example_2}
3. {retrieval_example_3}
CLASSIFIER RESULTS:
- Probability of spam: {p_clf}
TARGET REVIEW TEXT:
{text}
One of the system’s engineering solutions is a confidence threshold that
lets the system skip the LLM when the classifier shows high confidence in
its prediction. For a review xwith an embedding e, the RoBERTa-LoRA
classifier gives a spam probability pbase=P(spam|x ). Two fixed thresholds
are used:τ low= 0.3andτ high= 0.7.
•Ifpbase<0.3orpbase>0.7, the classifier is considered confident.
The LLM is not called, and the system returns the classifier’s label,
ˆybase=I(pbase≥0.5).
•If0.3≤pbase≤0.7, the review falls in the uncertainty window. The
system retrieves the evidence from FAISS and sends it to the LLM,
which makes the final decision.
In other words, the LLM only sees the reviews where the classifier is close
to a coin flip. This routing logic is shown in Algorithm 1. The same thresholds
are used in all experiments. The values0 .3and0.7were set symmetrically
around0.5as a simple choice and were not tuned. In our experiments, this
window contained about 48% of the zero-shot cross-domain reviews and about
21% after fine-tuning (Section 5). Threshold tuning is left for future work.
15

This approach achieves two primary objectives:
•Computational Resource Efficiency: By bypassing the generative
layer for high-confidence samples, the mechanism drastically reduces
GPU compute cycles, memory consumption, and overall inference la-
tency.
•Environmental Sustainability: By avoiding unnecessary LLM for-
ward passes on reviews that the classifier is confident about, the system
significantly reduces energy footprint.
Algorithm 1Confidence-Threshold LLM Routing
Require: Review text x, Embedding e, Base probability pbase, Thresholds
[τlow,τhigh], FAISS indexI, Neighborsk
Ensure:Final predictionˆy∈{0,1}
1:ˆybase⇐I(pbase≥0.5)
2:ifpbase<τlow∨pbase>τhighthen▷High confidence region: bypass LLM
3:returnˆy base
4:else▷Uncertainty region:τ low≤pbase≤τhigh
5:ifI̸=∅then
6:C⇐FAISS_Search(I,e,k)▷Retrieve similarity stats & top-3
examples
7:else
8:C⇐∅
9:end if
10:R⇐Invoke_LLM(x,p base,ˆybase,C)
11:ˆy llm⇐ExtractLabel(R)
12:ifˆy llm̸=Nonethen
13:returnˆy llm
14:else
15:returnˆy base ▷Parsing error of llm response
16:end if
17:end if
5. Results
5.1. Experimental Scenarios
As described in detail in Section 3.2, three experimental scenarios were
developed based on the prepared datasets to evaluate the RAG system:
16

1.Baseline Scenario: Testing on a familiar domain (hotels and restaurants)
with a 70/30 stratified imbalance.
2.ExtremeImbalanceScenario: TestingonanexternalsetofHotelReviews
with a realistic class imbalance (90/10).
3.Cross-Domain AI-Spam Scenario: Testing on a balanced (50/50) Cross-
domain Reviews dataset, where the fake reviews are generated by
artificial intelligence. It was a challenge for the system’s generalisation
ability.
5.2. Evaluation Metrics
Given the nature of the task, the model was evaluated using the following
metrics: Accuracy, Precision, Recall, F1-score, and PR-AUC. Particular
attention was paid to Precision and F1-score on the fake class. In content
moderation, a false positive causes direct reputational and commercial harm
to sellers, so Precision determines the practical applicability of the system.
F1-score as the balanced measure shows whether the gain in Precision is
achieved without a significant loss of overall detection quality.
5.3. Classifier Experiments
Before development of the final RAG pipeline, a comparison of approaches
for initial classification was conducted, ranging from baseline models (TF-IDF)
to fine-tuned contextual embeddings (RoBERTa-LoRA).
As summarised in Table 3, experiments showed that bag-of-words baselines
have noticeable limitations. While TF-IDF with Logistic Regression and
Linear SVM achieved moderate F1-scores (0 .6024and0.6040, respectively),
the complementary Naive Bayes baseline suffered from severe majority class
bias, failing to detect fake reviews (Recall of0.0897and F1-score of0.1639).
RoBERTa embedding representations with traditional classifiers improved
detection sensitivity in balanced settings; for example, Logistic Regression
achieved a Recall of0 .6923, but its low precision (0 .5590) degraded the overall
PR-AUC to0 .6617. Tree-based ensembles (Random Forest and LightGBM)
similarly struggled with the dense representation space under class skew.
In contrast, parameter-efficient fine-tuning of RoBERTa with Low-Rank
Adaptation (LoRA) and a weighted cross-entropy loss function demonstrated
significant performance across all metrics. This approach established an
optimal balance on the deceptive class: an Accuracy of0 .8130, Precision of
0.6862, Recall of0.6880, and an F1-score of0.6871.
17

Table 3: Comparison of different classification models on the test dataset
Model Acc F1-Score Prec Rec PR-AUC
TF-IDF + LogReg 0.7701 0.6024 0.6213 0.5846 0.7030
TF-IDF + LinearSVM 0.7907 0.6040 0.6921 0.53590.7172
TF-IDF + SVM 0.8067 0.5886 0.8044 0.4641 0.7076
TF-IDF + Naive Bayes 0.7273 0.16390.94590.0897 0.6327
RoBERTa + LogReg 0.7456 0.6186 0.55900.69230.6617
RoBERTa + LinearSVM 0.7441 0.6109 0.5584 0.6744 0.6382
RoBERTa + SVM 0.6914 0.5258 0.4848 0.5744 0.6019
RoBERTa + Random Forest 0.7823 0.4991 0.7933 0.3641 0.6629
RoBERTa + LightGBM 0.7807 0.5423 0.7173 0.4359 0.6798
RoBERTa-LoRA 0.8130 0.68710.6862 0.6880 0.6918
5.4. RAG System Evaluation
The final evaluation of the complete RAG architecture was conducted on
external datasets (the Hotel Reviews dataset and a cross-domain dataset) in
two modes: zero-shot (without retraining) and with lightweight retraining of
the classifier for 5 epochs.
For the 5-epoch domain adaptation experiments, each target dataset was
split into a 60% adaptation set and a 40% test set for evaluation using
stratified sampling and a fixed random seed. The RoBERTa-LoRA classifier
underwent lightweight fine-tuning on the adaptation split for 5 epochs, after
which the full RAG pipeline was evaluated strictly on the unseen test data.
The summary results for all scenarios are presented in Table 4.
18

Table 4: Performance comparison between baseline RoBERTa-LoRA and the full RAG
pipeline across all evaluation scenarios
Dataset Setup Acc Prec Rec F1-Score
In-Domain Baseline (70/30)
RoBERTa-LoRA only 0.8130 0.68620.6880 0.6871
Full RAG Pipeline0.8237 0.74240.6266 0.6796
Hotel Reviews (90/10, zero-shot)
RoBERTa-LoRA only 0.6712 0.16840.60070.2631
Full RAG Pipeline0.7656 0.19670.45390.2745
Hotel Reviews (90/10, 5-epoch adaptation)
RoBERTa-LoRA only 0.7083 0.21320.61360.3164
Full RAG Pipeline0.8592 0.35200.33330.3424
Cross-Domain Reviews (50/50, zero-shot)
RoBERTa-LoRA only 0.5790 0.56130.6819 0.6158
Full RAG Pipeline0.6300 0.63910.5788 0.6075
Cross-Domain Reviews (50/50, 5-epoch adaptation)
RoBERTa-LoRA only 0.8300 0.76510.92830.8389
Full RAG Pipeline0.8433 0.80670.88290.8431
19

The presented experimental comparison reveals the key patterns of the
proposed RAG architecture across various domains and under different class
distribution configurations. In all tested scenarios, the full RAG pipeline
improves precision and accuracy compared to the baseline RoBERTa-LoRA
classifier. Specifically, on the In-Domain dataset, precision increases from
0.6862to0.7424; on the fine-tuned Cross-Domain dataset, it increases from
0.7651to0.8067; and on the fine-tuned Hotel Reviews dataset, it rises from
0.2132to0.3520. This consistent improvement indicates that combining
retrieved examples with LLM reasoning acts as a highly effective verification
filter to eliminate false alarms.
This improvement in decision confidence is accompanied by a noticeable
trade-off between precision and recall. However, this compromise is advanta-
geous for practical e-commerce deployments, where falsely flagging genuine
user reviews causes greater reputational and commercial harm than missing
some fake reviews.
Furthermore, the results demonstrate the importance of a lightweight and
efficient adaptation. Evaluating the framework in a pure zero-shot setting
across new domains yields modest baseline results, whereas applying just five
additional epochs of LoRA fine-tuning boosts performance, elevating Cross-
Domain accuracy from0 .6300to0.8433and F1-score to0 .8431. Similarly,
undersevereclassimbalanceintheHotelReviewscase, the5-epochadaptation
improves overall accuracy from0 .7083to0.8592, increases precision from
0.2132to0.3520, and improves the F1-score from0.3164to0.3424.
5.4.1. Statistical Significance Analysis
To ensure that the performance changes resulting from the use of the RAG
architecture are statistically significant and not due to random sampling, a
paired bootstrap resampling test is performed between the Full RAG system
and RoBERTa-LoRA with a95%confidence level ( N= 2,000iterations).
The statistical evaluation was executed across the three primary benchmark
scenarios: In-Domain set, Fine-Tuned Hotel Reviews, and Fine-Tuned Cross-
Domain. Table 5 shows the results of the statistical analysis.
20

Table 5: Paired Bootstrap Significance Analysis ( N= 2,000). Asterisks (∗) denote
statistically significant differences in favour of our proposal, while double asterisks (∗∗)
mean significant differences in favour of RoBERTa-LoRA (p<0.05).
Scenario Metric∆(RAG−RoBERTa-LoRA)95% CI (p-value)
In-DomainPrecision+0.0562(+0.0271,+0.0873)∗
Recall−0.0614 (−0.0897,−0.0326)∗∗
F1-score−0.0075 (−0.0314,+0.0173)[p= 0.55]
Hotel ReviewsPrecision+0.1388(+0.0715,+0.2071)∗
(Fine-Tuned)Recall−0.2803 (−0.3688,−0.194)∗∗
F1-score+0.026 (−0.0456,+0.0913)[p= 0.46]
Cross-DomainPrecision+0.0416(+0.0213,+0.0608)∗
(Fine-Tuned)Recall−0.0455 (−0.066,−0.0254)∗∗
F1-score+0.0042 (−0.0115,+0.0195)[p= 0.59]
In all three evaluation scenarios, paired hypothesis testing reveals a con-
sistent and statistically significant trade-off between precision and recall.
The full RAG architecture provides a consistent and statistically significant
increase in precision across all tests. In particular, precision increases by
+0.0562(95%CI:[+0 .0271,+0.0873]) on the In-Domain scenario, by+0 .1388
(95%CI:[+0 .0715,+0.2071]) on the Fine-Tuned Hotel Reviews configuration,
and by+0 .0416(95%CI:[+0 .0213,+0.0608]) on the Fine-Tuned Cross-
Domain case. This behaviour confirms that the combination of FAISS’s vector
search and the LLM’s reasoning acts as a reliable filter for false positives.
At the same time, the conservative verification of the LLM layer leads
to a statistically significant reduction in recall across all configurations:
−0.0614(95%CI:[−0.0897,−0.0326]) for the In-Domain dataset, −0.2803
(95%CI:[−0.3688,−0.1940]) for hotel reviews, and −0.0455(95%CI:
[−0.0660,−0.0254]) for cross-domain testing. The most noticeable decrease
is observed when there is a strong class imbalance (Hotel Reviews dataset).
In this case, the LLM requires significantly more substantial textual evidence
before predicting the “fake” label.
The variation in the F1 metric remains statistically insignificant across all
three configurations: p= 0.547for the In-Domain set, p= 0.461for the hotel
reviews dataset, andp= 0.592for the cross-domain experiment.
For completeness, the detailed paired bootstrap forest plots and numerical
significancetablesforthezero-shotbaselineconfigurations(withoutparameter-
efficient domain adaptation) are provided in Appendix A.
21

5.4.2. Granular Error Analysis and Threshold Routing
To evaluate the performance of the confidence-threshold routing mecha-
nism, the decisions by the LLM layer were analysed across different probability
confidence intervals of the primary RoBERTa-LoRA classifier. In Figure 2,
thesecorrectionsareclassifiedas“Rescued”(incorrectbasepredictionssuccess-
fully corrected by the RAG pipeline) and “Broken” (correct base predictions
harmed by the LLM) across RoBERTa-LoRA confidence bins for all five
experimental setups: In-Domain, Hotel Reviews (Zero-Shot and Fine-Tuned),
and Cross-Domain AI-Spam (Zero-Shot and Fine-Tuned).
As expected from the routing rule in Section 4, the LLM made no changes
outside the uncertainty window ( pbase<0.3andpbase>0.7) in any of
the five setups. All corrections happen inside[0 .3,0.7]. Consequently, for
unambiguous samples, it is possible to eliminate the GPU compute overhead
and LLM latency.
The adjustments are strictly limited to the uncertainty window ([0 .3,0.7]).
The highest positive net gains are consistently observed in the probability
interval0.5–0.7for all datasets. In this region, where the main classifier
slightly favours a spam prediction with low confidence, the LLM, based
on the FAISS retrieval evidence, acts as a high-precision verification filter,
successfully correcting false alarms.
22

Figure2: EffectofRAGonModelPredictionsAcrossConfidenceBinsacrossallexperiments.
Applying domain adaptation with effective parameter tuning radically
changes the confidence distribution of the base classifier. In Zero-Shot settings
and the Fine-Tuned Hotel Reviews configuration, the model’s high uncertainty
causes a large number of samples to fall within the[0 .3,0.7]interval (e.g.,
n= 1065for Hotel Reviews Zero-Shot, n= 763for Fine-Tuned Hotel Reviews,
andn= 1463for Cross-Domain Zero-Shot). In contrast, after fine-tuning
in the cross-domain scenario, the number of uncertainty cases decreases
significantly ( n= 248uncertain samples vs n= 952samples with the model’s
high confidence). As a result, the RAG performs extensive corrections, leading
to a significant net increase of+283samples in the Zero-Shot Hotel Reviews
configuration,+181in the Fine-Tuned Hotel Reviews setting, and+153in
23

the Zero-Shot Cross-Domain scenario.
Fine-tuning in the Cross-Domain case increases the confidence level of
the base classifier, shifting the majority of samples into extreme confidence
intervals. Although this substantially reduces the number of samples entering
the LLM pipeline, the RAG architecture maintains a consistently positive net
correction rate.
5.5. Comparison with Cross-Domain SOTA Baselines
The performance of the proposed RAG system was compared with the
baseline models established by Salminen et al. on the same cross-domain
dataset to evaluate the robustness and competitiveness of the proposed RAG
architecture. Their study provides benchmarks using an OpenAI GPT-2
output detector, NBSVM, and a fine-tuned fakeRoBERTa model [18].
The baseline models were trained and evaluated strictly in-domain (using
an 80/20 split of the target 40,432-review corpus). In contrast, the proposed
RAG system was initially trained on a completely distinct domain (HoReCa
reviews). To ensure strict evaluation fairness, the exact test split of8 ,087
reviews was reproduced using a fixedrandom_state = 2021.
Table 6: Comparison of the proposed RAG system against baseline models on the cross-
domain dataset.
Detector Evaluation Mode Precision Recall F1-Score
OpenAI Detector1In-domain (8k test set) 0.83 0.82 0.82
NBSVM1In-domain (8k test set) 0.95 0.95 0.95
fakeRoBERTa1In-domain (8k test set) 0.97 0.97 0.97
Proposed RAG
System (No
retraining)zero-shot (8k test set) 0.76 0.42 0.54
Proposed RAG
System (With
retraining)5 epochs adaptation
(8k test set)0.88 0.92 0.90
1Note: Baseline metrics are extracted from Salminen et al. (2022) [18].
Table 6 demonstrates the strong performance of the proposed architec-
ture against target-domain baselines. Despite being initially trained on a
substantially different domain, lightweight 5-epoch adaptation on a small
sample of2,264reviews enables our system to achieve a competitive F1-score
24

of 0.90. Notably, the fine-tuned RAG system outperforms detection baselines
such as the OpenAI Detector (F1 =0 .82) and comes close to specialized
in-domain models like fakeRoBERTa (F1 =0 .97) and NBSVM (F1 =0 .95).
The proposed architecture achieves this level of accuracy using less than 7%
of the total training data for fine-tuning to a new domain (2 ,264vs.32,375
reviews).
5.6. Ablation Study
To evaluate the contribution of each component in the proposed architec-
ture, an ablation study was conducted on a test sample from the in-domain
dataset. During the experiments, four system configurations were compared
sequentially: the baseline logistic regression model applied to RoBERTa em-
beddings (RoBERTa + LogReg), the fine-tuned pretrained RoBERTa classifier
using Low-Rank Adaptation (RoBERTa-LoRA), the full RAG pipeline based
on a RoBERTa-LoRA classifier with structured memory evidence and explicit
vector metrics (Full RAG), and the full RAG pipeline using a simplified
prompt without explicit vector metrics and retrieved examples (Full RAG
with Light Prompt). The empirical results across all four configurations are
summarised in Table 7.
Table 7: Ablation analysis of pipeline components and prompting strategies on the in-
domain test dataset.
Configuration Accuracy Precision Recall F1-Score
RoBERTa + LogReg 0.7456 0.5590 0.6923 0.6186
RoBERTa-LoRA 0.8130 0.68620.6880 0.6871
Full RAG (light prompt) 0.8229 0.7331 0.6394 0.6831
Full RAG0.8237 0.74240.6266 0.6796
Comparative analysis of these configurations reveals several insights:
•Baseline vs Parameter-Efficient Fine-Tuning: Transitioning from
a linear classifier built on RoBERTa embeddings to LoRA fine-tuning of
the RoBERTa model significantly improves overall classification metrics,
raising Accuracy from0 .7456to0.8130, Precision from0 .5590to0.6862,
and F1-score from0.6186to0.6871.
•EffectivenessofFullRAGReasoning: IntegratingtheLLMdecision
layer with structured memory evidence to verify questionable reviews
25

reduces false positives by increasing Precision from0 .6862to0.7424
while maintaining strong overall Accuracy (0 .8237) compared with the
RoBERTa-LoRA classifier.
•Role of Explicit Similarity Metrics and Evidence: Implementing
a lightweight prompt that hides numerical FAISS vector distances and
retrieved examples does not worsen the metrics, but interpretability —
as will be shown in the following Section 5.7 — decreases significantly.
To demonstrate the systemic weaknesses of the light prompt configura-
tion, comprehensive cross-domain and imbalanced evaluations were added
to Appendix B (Table B.11). As detailed in Appendix B, using the light
prompt under severe class imbalance (fine-tuned Hotel Reviews case) causes
a sharp decline in the Precision metric from0 .3520to0.1981and in F1-score
from0.3424to0.2442. In contrast, the Precision score improves across all
evidence-backed settings. These findings confirm that providing structured
numerical distance metrics and real evidence within a full RAG prompt is
essential for model stability across diverse environments.
5.7. Qualitative Analysis & Case Studies
Evaluating the transparency and interpretability of generated decision
justifications is essential for deployment of an expert system in real-world
content moderation workflows. To systematically measure the quality of
LLM reasoning, the model’s explanations generated within the classifier’s
uncertainty window ( pbase∈[0.30; 0.70]) were analysed. A comparative
analysis of the proposed prompt for the RAG system, which includes explicit
numerical FAISS metrics and examples from the database, against the light
prompt, which omits explicit memory distances, was conducted.
Four key explainability and reasoning performance dimensions were calcu-
lated:
1.Evidence Grounding Rate (EGR): The proportion of generated
LLMexplanationsthathavereferencestoretrievedmetricsandexamples
from the vector database (e.g., neighbour distance, fake ratio);
2.AverageQuantitativeFactsCount: Theaveragenumberofverifiable
statistical parameters and vector metrics cited per explanation;
3.Positive Correction Rate (PCR): The frequency with which the
LLM successfully overrides an incorrect baseline classifier prediction;
26

Table 8: Grounding and reasoning metrics within the uncertainty window ( pbase∈
[0.30,0.70]).
Scenario Setup EGR (%) Avg. Facts Count PCR (%) Perplexity
In-Domain BaselineLight Prompt 0.5 1.0125.6 32.89
Full RAG100.0 3.5025.1 41.07
Hotel ReviewsLight Prompt 0.5 1.04 23.1 30.61
(Fine-Tuned)Full RAG99.5 6.32 31.3 27.55
Cross-DomainLight Prompt 0.0 1.08 19.327.61
(Fine-Tuned)Full RAG80.5 1.98 25.040.40
4.Mean Perplexity: The evaluation of fluency and linguistic predictabil-
ity of generated justifications using an external language model (Qwen-
2.5 0.5B). While lower values indicate a more traditional text structure,
local increases reflect the syntactic complexity associated with the
inclusion of statistical data and specific probabilities in natural text.
The empirical results across three setups (In-Domain dataset and fine-
tuned configurations of experimental data) are presented in Table 8.
As shown in Table 8, hiding the FAISS numerical distances in the configu-
ration with the light prompt obviously leads to an almost complete collapse
of grounding, reducing the EGR metric to 0.0-0.5 in all scenarios. In contrast,
the system with the proposed prompt achieves nearly perfect grounding jus-
tification (99 .5%–100.0%in in-domain and hotel reviews configurations, as
well as80.5%in cross-domain settings), consistently integrating an average
of1.98to6.32statistical parameters into each decision.
Furthermore, the positive correction rate (PCR) demonstrates that the
Full RAGpipeline successfully corrects between25 .0%and31.3%of baseline
classification errors. Under challenging conditions, such as an imbalanced
hotel reviews dataset and a cross-domain task, the proposed prompt achieves
a significantly higher rate of positive correction than theLight Promptvari-
ant (31.3%vs23.1%in hotel reviews,25 .0%vs19.3%in a cross-domain
environment).
With respect to the linguistic structure of LLM explanations, the Perplex-
ity metric, estimated using Qwen-2.5 (0.5B), provides important information
about the density of justifications. In the In-Domain (41 .07vs32.89) and
Cross-Domain (40 .40vs27.61) configurations, theFull RAG Promptexhibits
significantly higher perplexity than theLight Prompt. This increase in per-
27

plexity does not indicate poor readability, but is directly due to the dense
inclusion of non-standard numerical constructs, such as exact floating-point
vector distances (0 .1082), similarity thresholds, and specific classifier prob-
abilities (pbase= 0.615). These numerical tokens introduce local syntactic
complexity that differs from standard conversational pre-training text, which
reduces predictability for the Qwen evaluator. Curiously, in the imbalanced
Hotel Reviews dataset, the justifications generated usingFull RAGexhibit
both an exceptionally high fact density ( Avg. Facts Count = 6.32) and a lower
perplexity score (27 .55), indicating that specialised fine-tuning allows the
LLM to seamlessly integrate complex numerical evidence without sacrificing
natural sentence flow.
To show how evidence-based reasoning eliminates classification errors in
borderline cases, Table 9 qualitatively compares the explanations generated
for ambiguous cases.
28

Table 9: Side-by-side comparison of generated explanations on two uncertain test instances.
Bold highlights verifiable statistical evidence; italicised text indicates ungrounded specula-
tion.
Input Review &
MetadataLight Prompt (Ablation) Full RAG Prompt (Ours)
“GREAT CORN
Came here on a
late Monday night
and was one of
the last customers
I was very pleased
that the waitress
did not try to rush
us out”
Base Probabil-
ity:
pbase= 0.61
(Uncertain)
Retrieved Stat:
fake_ratio= 0.10
2/3examples - re-
als
Ground Truth:
y= 0(Genuine)Prediction:1(Deceptive)×
Output:
“LABEL: 1. REASONING: The
review contains generic and tem-
plated text (“GREAT CORN”) and
lacks situational context, making
it seem like a pre-written promo-
tional phrase. The mention of a
specific day and time (“late Mon-
day night”) is not enough to ground
the review in reality.”Prediction:0(Genuine)✓
Output:
“LABEL: 0. REASONING: The
retrieved examples are mostly real
(2out of3), with a fake ratio of
0.1in the top-20 neighbours. The
closest distance is0 .1082, indicat-
ing a moderate similarity to mem-
ory. The classifier result is0 .615,
which is not conclusive. The text is
grounded and situational, describ-
ing a specific experience at a restau-
rant. The mention of a late Mon-
day night and a specific interaction
with the waitress suggests a gen-
uine review. The use of “GREAT
CORN” in all caps may be a mi-
nor red flag, but it is not enough
to override the other evidence.”
“I feel like i need
to go back and
read this book
again I felt it was
very repetitive and
not applicable to
most Id love to be
able to quit work
move to the beach
and relax That
is simply not an
option”
Base Probabil-
ity:
pbase= 0.67
(Uncertain)
Retrieved Stat:
fake_ratio= 0.00
3/3examples -
reals
Ground Truth:
y= 0(Genuine)Prediction:1(Deceptive)×
Output:
“LABEL: 1 REASONING: The re-
view contains generic and tem-
plated language, such as “I feel like
I need to go back and read this book
again” and “Id love to be able to
quit work move to the beach and
relax”, which are common phrases
used in promotional or spam re-
views. The text also lacks spe-
cific details about the book, mak-
ing it seem like a generic complaint
rather than a genuine review.”Prediction:0(Genuine)✓
Output:
“LABEL: 0 REASONING: The re-
trieved examples are mostly real
(3/3), with a fake ratio of0 .00in
the top-20 neighbours. The clos-
est distance is0 .0937, indicating
a moderate similarity to memory.
The classifier result is a probabil-
ity of spam (0 .675), but I priori-
tise the retrieval evidence over the
classifier result. The target review
text is grounded and situational, ex-
pressing a personal opinion about
a book. The language is imperfect
and experience-based, describing a
feeling of repetition and inapplica-
bility. The tone is not overly pro-
motional or templated, and the text
does not exhibit generic or repeti-
tive language.”
29

The qualitative comparisons presented in Table 9 illustrate the fundamen-
tal difference between ungrounded generative reasoning (Light Prompt) and
vector-based reasoning (Full RAG Prompt):
1.Vulnerability to stylistic and structural artifacts: The LLM pro-
vided with a simplified prompt penalises stylistic informalities, specif-
ically, capitalised headings (“GREAT CORN”) or emotional and col-
loquial expressions (“I’d love to be able to quit work”), mistakenly
labelling them as formulaic deception. This results in false positives,
where genuine, situational customer feedback is wrongly penalised for
stylistic informalities.
2.Retrieval Neighborhood Geometry for Evidence Grounding:
Theclassical RAG promptprovides explicit quantitative memory signals
(fake_ratio = 0.10and0.00; nearestdistances0 .1082and0.0937). These
numerical metrics inform the LLM that the target review is within a
dense and statistically homogeneous cluster of verified genuine reviews.
Guided by this spatial evidence, the LLM correctly recognises that
minor stylistic noise does not override strong memory proximity.
3.Overcoming Baseline Classifier Bias: In borderline cases, when
the primary classifier outputs moderate spam probabilities ( pbase= 0.61
and0.67), ungrounded LLMs tend to confirm the classifier bias toward
fraud. In contrast, the structured RAG prompt lets the generative layer
actively overcome the classifier’s uncertainty whenever the retrieved
vector memory provides overwhelming evidence of authenticity, directly
preventing false positives.
6. Discussion
An empirical analysis of the experimental results demonstrates the per-
formance capabilities, structural trade-offs, and generalisation limits of the
hybrid Retrieval-Augmented Generation (RAG) architecture. This section
presents a comprehensive analysis of the results, which directly addresses the
main research questions (RQ1–RQ4).
6.1. Performance Improvement via Retrieval-Augmented Reasoning (RQ1)
The experimental data confirm that combining the RoBERTa-LoRA clas-
sifier with FAISS vector search and LLM-based reasoning shifts the classifier
toward a higher-precision operating point while maintaining statistically
30

competitive overall detection performance. Across all evaluated scenarios,
the full RAG pipeline achieves a statistically significant increase in Pre-
cision:+0.0562on the In-Domain dataset (rising from0 .6862to0.7424,
p-value< 0.05),+0.1388on the imbalanced Hotel Reviews dataset (rising
from0.2132to0.3520,p-value< 0.05), and+0 .0416on the Cross-Domain
dataset (rising from0.7651to0.8067, p-value<0.05).
This increase in Precision results in a statistically significant decrease in
Recall (−0.0614In-Domain,−0.2803Hotel Reviews, −0.0455Cross-Domain),
whereas changes in the overall F1-score remain statistically insignificant
(p-value> 0.05). In practical content moderation, this conservative verifica-
tion threshold is highly advantageous, because falsely labelling a genuine user
review as fake causes legitimate sellers greater commercial and reputational
damage than failing to detect an occasional fraudulent review. Further-
more, the threshold routing mechanism ( τlow= 0.3,τhigh= 0.7) restricts
the LLM outputs exclusively to borderline samples within the uncertainty
interval (pbase∈[0.3; 0.7]), yielding a net reduction in classification errors
(+14 within the domain, +181 for fine-tuned hotel reviews, +16 for fine-tuned
cross-domain reviews). Moreover, this approach completely avoids genera-
tive compute overhead for deterministic predictions, leading to substantial
reductions in energy consumption and the operational carbon footprint.
6.2. Cross-Domain Adaptation Efficiency via LoRA (RQ2)
Evaluating the system under zero-shot conditions on unfamiliar domains
yields modest performance (Cross-Domain Accuracy0 .6300, F1-score0 .6075)
due to domain shift and new AI-generated linguistic structures. However,
implementing parameter-efficient fine-tuning using LoRA adapters in just 5
additionalepochsenablesfastdomaintransfer, raisingCross-DomainAccuracy
to0.8433, Recall to0.8829, and F1-score to0.8431.
When evaluated against target-domain baselines established by Salminen
et al. (OpenAI Detector F1 = 0.82, NBSVM F1 = 0.95, fakeRoBERTa
F1 = 0.97), which were trained exclusively in-domain, the proposed RAG
system achieves competitive accuracy (F1 = 0.90, Precision = 0.88, Recall =
0.92) despite initial pre-training on an entirely different corpus of hotel and
restaurant reviews. It is important to note that the proposed methodology
achieves these results using less than 7% of the training data to adapt to
the target domain (2 ,264reviews compared to the32 ,375used in the SOTA
models within a single domain), which demonstrates both highly efficient data
utilization and strong cross-domain transferability. Additionally, adaptation
31

using LoRA improves the reliability of the base classifier predictions by
shifting the sample distributions toward high-confidence ranges and reducing
the number of ambiguous reviews requiring LLM evaluation, specifically, from
43% (zero-shot) of unclear cases to 12% (after fine-tuning), consequently
optimising computational throughput during inference.
6.3. Architecture Robustness Under Class Imbalance (RQ3)
Review distributions in e-commerce are often characterised by extreme
class imbalance similar to the Hotel Reviews scenario with 90% of genuine
and 10% of fake reviews. Traditional classifiers using the standard cross-
entropy loss function suffer from a strong bias toward the majority class,
as demonstrated by the TF-IDF + Naive Bayes baseline, which achieved a
Recall of only0 .0897. To mitigate this problem, the base RoBERTa-LoRA
classifier incorporates a dynamically weighted cross-entropy loss function that
penalises minority class misclassifications approximately 9 times more heavily
in the case of a 90%/10% imbalance.
Under a 90%/10% class skew, applying 5 epochs of LoRA adaptation
elevates overall Accuracy from0 .7083to0.8592, Precision from0 .2132to
0.3520, and F1-score from0 .3164to0.3424. The LLM generative layer in
the full RAG architecture maintains a conservative decision threshold under
high class imbalance, requiring stronger evidence from the vector knowledge
base. Although it results in a decrease in Recall, it provides a significant
improvement in Precision (+0 .1388,p-value< 0.05) and a net gain of+181
correctly reclassified reviews within the uncertainty window.
6.4. Interpretability Without Performance Degradation (RQ4)
Standard transformer-based classifiers operate as “black-box” models and
do not provide verifiable justifications for their predictions. Ablation analysis
demonstrates that integrating LLM reasoning with FAISS vector memory
allows ustomaintainbaselineclassification quality(Accuracy0 .8237vs0.8130,
F1-score0.6796vs0.6871on In-Domain data) while ensuring transparency in
decision-making.
It is important to note that providing explicit numerical vector metrics
(nearest neighbour distance, fake ratio among the 20 nearest neighbours)
and the top-3 examples from within the proposed RAG prompt yields an
Evidence Grounding Rate (EGR) of99 .5%–100.0%in In-Domain and Hotel
Reviews settings, providing an average of 1.98 to 6.32 statistical metrics per
explanation. In contrast, removing explicit distance metrics (Light prompt
32

experiment) obviously causes the EGR to drop to 0.0–0.5%, triggering LLM
hallucinations in which harmless stylistic features (such as capital letters
in headings or emotional phrasing) are mistakenly interpreted as deceptive
indicators. The spatial neighbourhood geometry provided by FAISS allows the
LLM to recognise when a target review falls within a dense cluster of verified
reviews, overriding base-classifier uncertainty and reducing false alarms.
6.5. Practical Implications and Limitations
From an operational perspective, the proposed hybrid RAG architecture
establishes a balanced, scalable framework for real-world automated modera-
tion platforms through three integrated capabilities. First, lightweight 5-epoch
LoRA fine-tuning enables rapid domain adaptation; when expanding to new
product categories (e.g., transitioning from hospitality to consumer electron-
ics), platform administrators can update only the low-rank adapter using a
minimal labelled sample, achieving substantial performance gains without
retraining the backbone network. Second, the threshold-gated routing mecha-
nism optimises inference latency and computational overhead, bypassing LLM
invocation when classifier confidence is high and thereby reducing operational
carbon footprint. Third, evidence-grounded reasoning provides trust-and-
safety teams with explicit, auditable decision traces retrieved directly from
vector memory.
Ultimately, this work demonstrates that Retrieval-Augmented Generation
transforms deceptive review detection from opaque binary classification into
an auditable expert content moderation system. Rather than serving purely as
a standalone predictor, the RAG architecture operates as an expert auditing
engine that reconciles high precision with verifiable textual evidence. For
platform administrators, trust-and-safety teams, and regulatory compliance
auditors, this transparent reasoning trail bridges the gap between automated
deep learning predictions and defensible human-in-the-loop decision-making
in real-world commercial environments.
Despite these strengths, the system has several limitations. First, under
severe class imbalance (90/10), the fine-tuned Full RAG pipeline reaches a
precision of0 .3520and a recall of0 .3333. This means that about two thirds of
the classified reviews are genuine and about two thirds of the fake reviews are
missed, so the system should not work alone in such conditions. Second, the
LLM layer provides the classifier with a higher precision and a lower recall,
while the F1-score does not change significantly. We did not test whether
a simple change of the classifier decision threshold gives the same effect, so
33

this research is still open. Third, explainability was measured with automatic
metrics only. The Evidence Grounding Rate shows that explanations cite the
retrieved evidence, but not that they use it correctly, and we did not run a
study with human moderators. Finally, all results come from a single run with
one LLM (Llama-3.1-8B, 8-bit), fixed thresholds, and labelled target-domain
data for the 5-epoch adaptation.
7. Conclusions and future work
This paper presents a hybrid architecture based on RAG designed for
the automatic detection and explanation of deceptive online reviews. By
structurally integrating a supervised RoBERTa-LoRA classifier, a FAISS
vector search module, and a large language model, the proposed system
overcomes the fundamental limitations of traditional “black-box” neural
network classifiers and traditional lexical models. The main contribution
of this research is transforming deceptive review detection into an evidence-
grounded, auditable decision-making pipeline in which each prediction is
accompanied by a human-understandable justification based on evidence
retrieved from vector memory.
The experimental evaluation across three different scenarios (in-domain,
a highly asymmetric distribution of90%/10%of hotel reviews, and a bal-
anced cross-domain AI-generated spam) demonstrates the robustness and
generalisation ability of the proposed architecture. The full RAG acts as a
high-precision verification filter, delivering statistically significant increases
in Precision across all evaluated setups (+0 .0562In-Domain,+0 .1388Im-
balanced data,+0 .0416Cross-Domain, p-value< 0.05). Lightweight LoRA
fine-tuning in just 5 additional epochs enables rapid cross-domain transfer, el-
evating F1-score to 0.90 on benchmark evaluation and outperforming OpenAI
Detector (F1 =0 .82) despite using minimal data for adaptation process. To
optimise efficiency, the confidence threshold routing mechanism strictly limits
LLM inference to borderline predictions, significantly reducing computational
overhead. Furthermore, providing explicit numerical vector metrics and re-
trieved examples achieves an Evidence Grounding Rate of80 .5%–100.0%,
effectively eliminating hallucinations. Although integrating the retrieval exact
numerical scores increases local perplexity (27 .55–41.07), this reflects the
structural density of quantitative evidence rather than degraded readability.
Future work will focus on several key directions. First, we plan to anal-
yse trade-offs between precision–recall trade-offs under different operational
34

settings and investigate deployment in agentic moderation environments. To
address current operational constraints, future extensions will explore mi-
nority class resampling techniques (such as SMOTE and ADASYN), expand
the vector knowledge base across broader commercial domains, integrate
higher-capacity LLM backbones, and extend the framework to multilingual
review environments.
Acknowledgments
The research reported in this paper was supported by the European
Union NextGenerationEU/PRTR, and FederaTrans project: Grant PID2024-
158373OB-I00 funded by MICIU/AEI/10.13039/501100011033 and by ERD-
F/EU. Finally, the research reported in this paper is also funded by the
European Union (CUSTOMAI project, grant agreement no. 101226029).
Appendix A. Zero-Shot Significance Analysis
To complement the main evaluation, this appendix presents the detailed
paired bootstrap significance analysis for the zero-shot baseline setups. Fig-
ure A.3 illustrates the bootstrap difference distributions, and Table A.10
details the precise numerical metrics and confidence intervals.
Table A.10: Paired Bootstrap Significance Analysis ( N= 2,000). Asterisks (∗) denote
statistically significant differences in favour of our proposal, while double asterisks (∗∗)
mean significant differences in favour of RoBERTa-LoRA (p<0.05).
Scenario Metric∆(RAG−Base)95% CI (p-value)
Hotel Reviews
(zero-shot)Precision+0.0283(+0.0103,+0.0457)∗
Recall−0.1468 (−0.1953,−0.1018)∗∗
F1-Score+0.0114 (−0.0145,+0.035)[p=.341]
Cross-Domain
(zero-shot)Precision+0.0779(+0.06113,+0.0954)∗
Recall−0.1031 (−0.1269,−0.0778)∗∗
F1-Score−0.0083 (−0.0262,+0.001)[p=.388]
35

Figure A.3: Bootstrap confidence intervals (95%CI) between Full RAG and Baseline across
zero-shot configurations.
Appendix B. Additional Experiments with Light Prompt
Table B.11: Ablation study of pipeline components and prompting strategies on the Hotel
Reviews and Cross-Domain scenarios.
Dataset Setup Acc Prec Rec F1-Score
Hotel Reviews (zero-shot)
RoBERTa-LoRA 0.6712 0.16840.60070.2631
Full RAG (light prompt) 0.7456 0.1807 0.4539 0.2585
Full RAG0.7656 0.19670.45390.2745
Hotel Reviews (fine-tuned)
RoBERTa-LoRA 0.7083 0.21320.61360.3164
Full RAG (light prompt) 0.7833 0.1981 0.3182 0.2442
Full RAG0.8592 0.35200.33330.3424
Cross-Domain (zero-shot)
RoBERTa-LoRA 0.5790 0.56130.68190.6158
Full RAG (light prompt) 0.6270 0.6156 0.65500.6347
RAG (RoBERTa-LoRA)0.6300 0.63910.5788 0.6075
Cross-Domain (fine-tuned)
RoBERTa-LoRA 0.8300 0.7651 0.9283 0.8389
Full RAG (light prompt) 0.8292 0.76030.93710.8395
Full RAG0.8433 0.80670.88290.8431
References
[1]Statista, Cumulative number of reviews submitted to
yelp, https://www.statista.com/statistics/278032/
cumulative-number-of-reviews-submitted-to-yelp/(2023).
36

[2]Brightlocal News, Local consumer review survey, https://www.
brightlocal.com/research/local-consumer-review-survey/
(2025).
[3]M. Luca, Reviews, reputation, and revenue: The case of yelp. com,
Com (March 15, 2016). Harvard Business School NOM Unit Working
Paper (12-016) (2016).
[4]U. Aslam, M. Jayabalan, H. Ilyas, A. Suhail, A survey on opinion spam
detection methods, International Journal of Scientific and Technology
Research 8 (9) (2019).
[5]Wise Review, Fake review statistics, https://wiserreview.com/blog/
fake-review-statistics/(2025).
[6]E. Papageorgiou, C. Chronis, I. Varlamis, Y. Himeur, A survey on the
use of large language models (llms) in fake news, Future Internet 16 (8)
(2024) 298.
[7]P. Lv, M. Dong, P. Hu, A lightweight framework for self-reflective
knowledge switching in misinformation detection, Expert Systems with
Applications (2026) 131502.
[8]C. Yang, S. Yu, Y. Chen, R. Huang, Y. Qin, C. Lin, A causality-aware
and hallucination-mitigation framework for faithful news summarization,
Expert Systems with Applications (2026) 132143.
[9]L. Gao, J. Meng, D. Zhao, B. Wang, Sel-pmf: Semantic enhancement via
llms and progressive multimodal fusion for fake news detection, Expert
Systems with Applications (2026) 133670.
[10]M. Ott, Y. Choi, C. Cardie, J. T. Hancock, Finding deceptive opinion
spam by any stretch of the imagination, in: Proceedings of the 49th
annual meeting of the association for computational linguistics: Human
language technologies, 2011, pp. 309–319.
[11]N. Jindal, B. Liu, Opinion spam and analysis, in: Proceedings of the
2008 international conference on web search and data mining, 2008, pp.
219–230.
37

[12]J.Devlin, M.-W.Chang, K.Lee, K.Toutanova, Bert: Pre-trainingofdeep
bidirectional transformers for language understanding, in: Proceedings
of the 2019 conference of the North American chapter of the association
for computational linguistics: human language technologies, volume 1
(long and short papers), 2019, pp. 4171–4186.
[13]Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis,
L. Zettlemoyer, V. Stoyanov, Roberta: A robustly optimized bert pre-
training approach, arXiv preprint arXiv:1907.11692 (2019).
[14]N. Raza, S. J. Abdulkadir, Y. A. Abid, S. S. Albouq, A. Alwadain, A. U.
Rehman, E. H. Sumiea, M. Farhan, Enhancing fake news detection with
transformer-based deep learning: A multidisciplinary approach, Plos one
20 (9) (2025) e0330954.
[15]S. Lopez-Joya, J. A. Diaz-Garcia, M. D. Ruiz, M. J. Martin-Bautista,
The blueprint of a new fact-checking system: A methodology to enrich
rag systems with new generated datasets, Computers and Electrical
Engineering 128 (2025) 110746.
[16]R. Singal, P. Patwa, P. Patwa, A. Chadha, A. Das, Evidence-backed
fact checking using rag and few-shot in-context learning with llms, in:
Proceedings of the Seventh Fact Extraction and VERification Workshop
(FEVER), 2024, pp. 91–98.
[17]C. Niu, Y. Guan, Y. Wu, J. Zhu, J. Song, R. Zhong, K. Zhu, S. Xu,
S. Diao, T. Zhang, Veract scan: Retrieval-augmented fake news detection
with justifiable reasoning, in: Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 3: System
Demonstrations), 2024, pp. 266–277.
[18]J. Salminen, C. Kandpal, A. M. Kamel, S.-g. Jung, B. J. Jansen, Creating
and detecting fake reviews of online products, Journal of Retailing and
Consumer Services 64 (2022) 102771.
[19]Kaggle, Yelp fake review dataset, https://www.kaggle.com/datasets/
virenraina/yelp-fake-review-dataset/data(2026).
[20] M. Liu, Yelp dataset (Apr. 2024).doi:10.57760/sciencedb.17086.
38

[21]Kaggle, Yelp hotel reviews dataset, https://www.kaggle.com/
datasets/abidmeeraj/yelp-labelled-dataset(2026).
[22]L. Xu, H. Xie, S. J. Qin, X. Tao, F. L. Wang, Parameter-efficient
fine-tuning methods for pretrained language models: A critical review
and assessment, IEEE Transactions on Pattern Analysis and Machine
Intelligence (2026).
[23]E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang,
W. Chen, et al., Lora: Low-rank adaptation of large language models.,
Vol. 1, 2022, p. 3.
[24]J. Johnson, M. Douze, H. Jégou, Billion-scale similarity search with gpus,
IEEE transactions on big data 7 (3) (2019) 535–547.
[25]Ollama, Llama3.1:8b-instruct, https://ollama.com/library/llama3.
1:8b-instruct-q8_0(2025).
39