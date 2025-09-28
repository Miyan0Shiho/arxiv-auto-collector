# Beyond Stars: Bridging the Gap Between Ratings and Review Sentiment with LLM

**Authors**: Najla Zuhir, Amna Mohammad Salim, Parvathy Premkumar, Moshiur Farazi

**Published**: 2025-09-25 09:39:12

**PDF URL**: [http://arxiv.org/pdf/2509.20953v1](http://arxiv.org/pdf/2509.20953v1)

## Abstract
We present an advanced approach to mobile app review analysis aimed at
addressing limitations inherent in traditional star-rating systems. Star
ratings, although intuitive and popular among users, often fail to capture the
nuanced feedback present in detailed review texts. Traditional NLP techniques
-- such as lexicon-based methods and classical machine learning classifiers --
struggle to interpret contextual nuances, domain-specific terminology, and
subtle linguistic features like sarcasm. To overcome these limitations, we
propose a modular framework leveraging large language models (LLMs) enhanced by
structured prompting techniques. Our method quantifies discrepancies between
numerical ratings and textual sentiment, extracts detailed, feature-level
insights, and supports interactive exploration of reviews through
retrieval-augmented conversational question answering (RAG-QA). Comprehensive
experiments conducted on three diverse datasets (AWARE, Google Play, and
Spotify) demonstrate that our LLM-driven approach significantly surpasses
baseline methods, yielding improved accuracy, robustness, and actionable
insights in challenging and context-rich review scenarios.

## Full Text


<!-- PDF content starts -->

Beyond Stars: Bridging the Gap Between Ratings
and Review Sentiment with LLM
Najla Zuhir, Amna Mohammad Salim, Parvathy Premkumar, Moshiur Farazi*
College of Computing & Information Technology
University of Doha for Science and Technology, Doha, Qatar
{najla.zuhir, amna.mohammad.salim, parvathypremkumar1901}@gmail.com, moshiur.farazi@udst.edu.qa
Abstract—We present an advanced approach to mobile app
review analysis aimed at addressing limitations inherent in
traditional star-rating systems. Star ratings, although intuitive
and popular among users, often fail to capture the nuanced
feedback present in detailed review texts. Traditional NLP
techniques—such as lexicon-based methods and classical machine
learning classifiers—struggle to interpret contextual nuances,
domain-specific terminology, and subtle linguistic features like
sarcasm. To overcome these limitations, we propose a modular
framework leveraging large language models (LLMs) enhanced
by structured prompting techniques. Our method quantifies
discrepancies between numerical ratings and textual sentiment,
extracts detailed, feature-level insights, and supports interactive
exploration of reviews through retrieval-augmented conversa-
tional question answering (RAG-QA). Comprehensive experi-
ments conducted on three diverse datasets (A WARE, Google Play,
and Spotify) demonstrate that our LLM-driven approach signif-
icantly surpasses baseline methods, yielding improved accuracy,
robustness, and actionable insights in challenging and context-
rich review scenarios.
Index Terms—App Reviews, Review Analysis, Discrepancy,
Natural Language Processing (NLP), Large Language Models
(LLMs), Prompt Engineering, Aspect-Based Sentiment Analysis
(ABSA), Topic Modeling, Retrieval-Augmented Question Answer-
ing (RAG-QA).
I. INTRODUCTION
The exponential growth of mobile applications has trans-
formed app ratings into critical success indicators, directly
influencing user adoption decisions and marketplace visibility.
Star ratings serve as the primary quality metric for millions
of users navigating crowded app stores because they are
easy to interpret and provide an immediate signal. However,
ratings alone do not provide developers with the detailed
feedback required to guide feature prioritization, bug fixes,
and overall product strategy for iterative improvement. App
store reviews, together with star ratings put thedeveloper-in-
the-loop. Review texts often reveal specific feature requests,
usability issues, and contextual concerns that numeric scores
cannot capture. When reviews are analyzed alongside ratings,
the combined signal uncoverswhatusers think andwhythey
think it, which provide contextual information essential for
guiding targeted, user-informed improvements.
Historically, sentiment extraction has depended on lexicon-
based methods and conventional machine learning classifiers
[1], [2]. Such techniques perform well on straightforward po-
larity tasks by identifying positive, negative and/or neutrality
Fig. 1. Comparison of traditional NLP pipelines and LLM-based sentiment
analysis. Traditional approaches involve multiple sequential steps and require
extensive validation, whereas LLM-based methods can perform sentiment
extraction in a single, prompt-driven step, yielding more reliable results.
of the review text. However, they struggle to capture con-
textual subtleties, domain-specific terminology, and complex
linguistic phenomena such as sarcasm—factors that undermine
the reliability of their predictions [3]. Furthermore, traditional
NLP-based models suffer even more from extracting relevant
keywords or information from reviews and communicating
effective feedback to the developer. To overcome these short-
comings, recent work has turned to large language models
(LLMs) in conjunction with sophisticated prompt-engineering
strategies [4], [5]. These models exhibit a deeper understand-
ing of context and can distill richer, more actionable insights
from unstructured review text.
Structured prompting has emerged as a powerful means of
steering LLMs toward fine-grained information extraction and
sentiment analysis in app reviews [4]. By defining explicit
prompt templates—ranging from zero-shot instructions (e.g.
‘Classify the sentiment of this review as positive, nega-
tive, or neutral.’) and few-shot examples to chain-of-thought
chains that require the model to articulate its reasoning—these
methods can dynamically guide model outputs into precise,
structured formats. For example, Wang et al. demonstrated that
five-shot prompting of ChatGPT not only rivals but can exceed
fine-tuned BERT classifiers on review sentiment tasks. In the
aspect-level setting, Shah et al. [6] employ an ‘explain-then-
annotate’ prompt to extract feature–sentiment pairs directly
from unstructured text and show a 23.6 percentage-point F1
gain over rule-based baselines in zero-shot mode and a further
6 point boost with five examples. Beyond example-basedarXiv:2509.20953v1  [cs.AI]  25 Sep 2025

prompting, instructive templates that define the model’s role
(e.g. ‘You are a sentiment analysis assistant. . . ’) and enforce
output schema (such as JSON) have been shown to improve
consistency and ease downstream parsing. Finally, hybrid
frameworks that layer domain-specific rules and heuristic
filters atop prompt templates enable developers to encode
ontological constraints—ensuring that extracted aspects and
sentiments align with real-world feature taxonomies and de-
veloper needs. Together, these structured prompting strategies
unlock rapid, low-overhead extraction of context-sensitive,
actionable insights from millions of raw app reviews—without
the time and expense of task-specific fine-tuning.
Despite these gains, structured prompting remains vul-
nerable to brittleness and hallucination across the pipeline.
Minor rephrasing can disrupt aspect–sentiment extraction and
topic labeling, and generative question-answering (QA) mod-
ules may invent spurious features or misinterpret queries. In
this work, we address these issues by embedding an auto-
mated prompt-optimization framework that adapts templates
to each app’s vocabulary and feedback patterns. At every
stage—discrepancy analysis, aspect extraction, topic modeling,
and retrieval-augmented QA—we layer in lightweight consis-
tency checks, grounded in V ADER-derived baselines and rule-
based heuristics, to detect and correct unlikely outputs. This
hybrid strategy delivers robust, accurate, and scalable review
analysis without the need for extensive manual prompt tuning.
The contributions of this paper are as follows:
•Baseline discrepancy analysis.We introduce a lexicon-
based V ADER pipeline that computes per-review polar-
ity, normalizes scores to the five-star scale, and visu-
alizes the gap between text-derived sentiment and user
ratings—establishing a reproducible baseline for down-
stream comparison.
•Robust LLM-driven review mining framework.We
develop a modular, prompt-based system—leveraging
few-shot examples, prompt chaining, and
meta-prompting—that automatically extracts
aspect–sentiment–recommendation triples, uncovers
thematic clusters via LLM-enhanced topic modeling,
and supports retrieval-augmented QA. An automated
prompt-optimization loop and lightweight, rule-based
consistency checks ensure stability and accuracy across
diverse app vocabularies.
•Comprehensive empirical validation.Through experi-
ments on three heterogeneous corpora (AWARE, Google
Play, and Spotify), we demonstrate that our LLM-based
pipeline significantly outperforms traditional lexicon- and
machine-learning baselines in sentiment accuracy, aspect-
extraction F1, topic coherence, and QA relevance.
II. RELATEDWORK
Traditional Sentiment Analysis Methods:Historically,
mobile app review analysis primarily utilized lexicon-based
sentiment analysis and classical machine learning (ML) tech-
niques, such as Support Vector Machines (SVM) and RandomForest classifiers [1]. These methods excelled in straight-
forward polarity classification tasks—categorizing reviews as
positive, negative, or neutral—but often struggled with lin-
guistic complexities such as sarcasm, context dependence,
and domain-specific jargon [2], [3]. Probabilistic topic mod-
eling methods, like Latent Dirichlet Allocation (LDA) and
Non-negative Matrix Factorization (NMF), were frequently
employed to uncover themes from unlabeled data, but these
approaches lacked scalability and robustness against evolving
language use [7]. To bridge this gap, our work leverages
advanced LLM prompting techniques, significantly reducing
the dependency on extensive labeled datasets and manual fea-
ture engineering, enabling more robust and scalable sentiment
analysis.
Aspect-Based Sentiment Analysis (ABSA):ABSA tech-
niques emerged to provide detailed insights by identifying
sentiments toward specific app features. Early approaches
combined rule-based feature extraction with lexicon-based
sentiment scoring, and later methods employed fine-tuned
BERT models, substantially improving aspect extraction per-
formance [8], [9]. Nevertheless, these supervised models re-
quired extensive domain-specific annotations, limiting their
adaptability and scalability. Recent research has demonstrated
the effectiveness of LLMs, notably GPT-4, outperforming
traditional ABSA methods without task-specific fine-tuning
[6]. Our approach extends this advancement by integrating
structured LLM prompts that extract not only aspect–sentiment
pairs but also actionable user recommendations, delivering
richer, developer-focused feedback.
Discrepancy Between Star Ratings and Review Text:
Identifying discrepancies between numerical star ratings and
textual sentiment remains a critical yet challenging aspect
of app review analysis. Approximately 20% of app reviews
exhibit inconsistencies that significantly affect user perception
and app marketability [10]. Traditional ML methods and
advanced deep learning techniques have attempted to address
these mismatches but often required extensive preprocessing
or lacked nuanced interpretability [11], [12]. Recently, the
potential of direct discrepancy detection via LLM prompting
was suggested but remains largely unexplored [4]. Our work
quantifies rating-review discrepancies by mapping V ADER
sentiment scores to star ratings and computing absolute dif-
ferences. This lexicon-based baseline enables future LLM-
enhanced analysis.
Topic Modeling and Retrieval-Augmented QA:Beyond
traditional sentiment analysis, topic modeling and retrieval-
augmented QA methods have seen significant improvements
through integration with LLMs. Techniques like BERT-Topic
have effectively clustered and summarized large-scale re-
view datasets [5]. Additionally, retrieval-augmented gener-
ative methods have enabled interactive querying of review
databases, providing developers immediate insights into user
feedback [13]. Despite these advances, existing methods often
lack robustness to minor linguistic variations and suffer from
potential hallucinations or inaccuracies in generated responses.
Our approach addresses these challenges by embedding auto-

Fig. 2. Modular LLM-based review analysis framework. Preprocessed reviews feed into three independent components: (1) Aspect–sentiment extraction with
recommendations, (2) topic modeling to surface high-level themes, and (3) retrieval-augmented QA to answer targeted questions.
mated prompt optimization loops tailored to app-specific vo-
cabularies and implementing consistency checks through rule-
based heuristics and lexicon-based baselines, thus ensuring
robust and reliable thematic summaries and conversational QA
interactions.
III. METHODS
We first outline theAspect Extractionmodule for fine-
grained feature and sentiment mining. We then detail theTopic
Modelingapproach used to uncover latent thematic structures
in large-scale review corpora. Finally, we present theRetrieval-
Augmented QAsystem that supports interactive querying and
evidence-backed summarization of user feedback.
Although we present the system as a complete LLM-based
pipeline for app review analytics, it is intentionally modular:
each component can operate independently. For this study,
we paired each module with the dataset best suited to its
task (AWARE for aspect-level extraction, Spotify for topic
modeling and retrieval, and Google Play for discrepancy
analysis), ensuring rigorous, task-appropriate evaluation rather
than forcing a single dataset through all stages.
A. LLM Backends
We experimented with the following three LLMs in all our
LLM-powered modules: aspect extraction, topic-label genera-
tion, and RAG-QA:
•GPT-4(via the OpenAI API), chosen for its state-of-the-
art instruction-tuned performance [14];
•LLaMA 2 7B-chat(via HuggingFace), selected as a
widely-used open-source alternative [15];
•Mistral 7B(via the official Mistral inference endpoint),
included for its strong quality-to-compute trade-off [16].After preliminary evaluations, GPT-4 consistently demon-
strated superior performance in capturing contextual nuance
and adhering to structured output formats. Therefore, to ensure
a clear and focused analysis of the maximum potential of our
framework, all experimental results reported in Section V are
generated using the GPT-4 model.
B. Aspect Extraction
Aspect extraction involves identifying specific features and
their associated sentiments expressed within app reviews.
To achieve accurate and scalable extraction, we employ a
structured LLM-based prompting approach. This module takes
individual review sentences from the AWARE dataset as input.
Using a few-shot prompt that includes exemplar aspect terms
drawn directly from AWARE, the model returns structured
outputs capturing precise aspect terms. Subsequently, a second
LLM prompt classifies the sentiment for each extracted aspect
as positive, neutral, or negative. Finally, the module mines
explicit user recommendations, outputting imperative phrases
for actionable feedback.
C. Topic Modeling
Topic modeling aims to discover and interpret meaningful
thematic clusters within large-scale review corpora. After
standard preprocessing steps—cleaning, deduplication, and
English-language filtering—the Spotify review dataset serves
as input. The processed text undergoes embedding generation
using the pre-trained transformer model (”all-mpnet-base-v2”)
from SentenceTransformers, yielding high-dimensional em-
beddingsh. BERTopic is applied to cluster these embeddings
using UMAP for dimensionality reduction and HDBSCAN for
clustering:
c=HDBSCAN(UMAP(h)),(1)

wherecdenotes cluster assignments. To enhance interpretabil-
ity, the top keywords from each topic cluster are further
processed using few-shot prompt, which generates descriptive
and intuitive topic labels and summaries, facilitating human
comprehension and analysis.
D. Retrieval-Augmented QA
The retrieval-augmented QA component enables interactive
querying of large-scale review datasets. Initially, the cleaned
review corpusDis segmented into overlapping text chunks,
encoded into vector embeddingsx iusing a sentence trans-
former model, and indexed within a FAISS-based vector store
V. When presented with a natural-language queryq, the
retriever computes its embeddingx qand retrieves thekmost
semantically similar chunks via cosine similarity scoring:
score(q,x i) =xq·xi
∥xq∥ ∥x i∥,x i∈ V.(2)
These top-ranked chunks populate a structured prompt, refined
by meta-prompting techniques, to instruct the answering LLM
to produce concise, evidence-backed responses. Thus, users
receive targeted, context-aware summaries and actionable in-
sights directly from large datasets without manual review.
IV. EXPERIMENTS
To evaluate the effectiveness of advanced review analy-
sis techniques, we conducted three LLM-Based experiments
across multiple app review datasets. This methodology is
designed to address the emerging capabilities of modern LLM-
driven approaches.
Each experimental module is described in detail in the
following subsections.
A. Data Collection & Pre-processing
We utilize three heterogeneous app review datasets:
•Google Play Store App Reviews:A comprehensive
collection of user reviews and corresponding star ratings
for various mobile applications. The dataset contains over
12k+ reviews [17].
•Spotify Reviews:This large corpus consists of user-
generated reviews, each accompanied by metadata such
as star rating, date, device, and app version. The Spotify
review dataset contains 80k+ reviews [18].
•A WARE (ABSA Dataset):human-annotated dataset that
contains 11k+ review sentences of apps in the domains
of Productivity, Social Networking, and Games. Each
sentence is labeled with aspect terms, aspect categories,
and aspect sentiment [19].
Pre-processing:Each dataset was processed through a
uniform preprocessing pipeline, which included standard text
cleaning, deduplication and filtering to retain only English-
language reviews. The resulting clean datasets were then used
as input for our experiments.B. Discrepancy Analysis (Rating vs. Sentiment)
We have experimented with multiple app review datasets,
to quantify how often and by what magnitude user reviews
diverge from star ratings. For rapid prototyping, we used
V ADER’s SentimentIntensityAnalyzer [20] to assign a polarity
score to each review. These polarity scores were then mapped
onto a 1–5 scale (the ‘Sentiment Rating’) to match the original
star-rating range. Finally, we computed a discrepancy feature
as the absolute difference between a review sentiment rating
and its original star rating, revealing systematic cases where
numerical ratings either overestimate or underestimate user
satisfaction (Fig. 3). This experiment underscores the impor-
tance of combining star ratings with textual analysis: relying
solely on numeric scores may misrepresent the true sentiment
of users.
Note on the V ADER baseline.We strictly use V ADER
as a fast, reproducible lexicon baseline. It is known to over-
predict neutral in domain-specific app text and to struggle with
sarcasm, litotes (e.g., “not bad”), and negation scope; it also
lacks coverage for app-specific vocabulary without lexicon
adaptation. Consequently, we treat V ADER as a conservative
floor to motivate discrepancy analysis rather than as an accu-
racy ceiling.
C. Aspect Extraction with Recommendation Mining
Using the AWARE dataset, the Aspect Extraction module
was tested through a structured pipeline involving few-shot
and chained prompts. Initially, few-shot prompting identified
concrete aspect terms within reviews. Subsequently, a chained
prompt classified each aspect’s sentiment polarity (positive,
neutral, or negative). Finally, recommendation mining ex-
tracted actionable feedback expressed by users. The detailed
example outputs from this pipeline are presented in Table II.
The experimental evaluation aims to measure precision and
recall of the extracted (aspect, sentiment) tuples to validate the
robustness and utility of our structured prompting methods.
D. LLM-Enhanced Topic Modeling
In this experiment, our aim was to apply BERTopic for
unsupervised topic modeling, enhanced with custom embed-
dings and post-processed using LLM. The dataset used is
spotify reviews, and sentence embeddings are implemented
by using a transformer model from the sentence-transformers
library (e.g., ”all-mpnet-base-v2”). These embeddings are fur-
ther clustered using HDBSCAN, and topics are extracted via
BERTopic. In an effort to improve topic interpretability, a
custom function sends the top keywords from each topic to the
LLM, using a structured prompt that asks the model to return
a short, specific label in title case. Along with this, another
LLM prompt—fed with sample reviews from each topic clus-
ter—generates summaries of representative topic documents.
The main aim with this integration of BERTopic with LLM-
generated labels and summaries is designed to enhance the
ease of understanding, readability and user relevance of the
discovered topics in the review domain. The detailed example
outputs from this pipeline are presented in Table III.

Fig. 3. All three datasets shows discrepancy, it suggests that numeric ratings may overestimate or underestimate user satisfaction, as they don’t fully capture
the nuanced feedback expressed in written reviews. V ADER-derived sentiment ratings, on the other hand, provide a more grounded perspective, highlighting
the importance of combining both metrics for an accurate assessment of user satisfaction.
TABLE I
SAMPLERETRIEVAL-AUGMENTEDQA EXAMPLES
Query & Answer Retrieved Reviews
Q1:What specific crashes or
freezes do users report most often?
A1:Users report frequent crashes
and freezes requiring restarts, of-
ten when navigating away, during
timeouts, or upon screen-lock.*Randomly crashes and freezes
despite latest version *Regular
crashes on Pixel when navigating
away *App unresponsive with fre-
quent freezes *Crashes more fre-
quent after updates.
Q2:What are the most common
user complaints?A2:Users cite
poor customer service, unresolved
billing issues, bugs, intrusive ads,
and weak podcast features.*Horrible customer service—still
charging my card *Very poor cus-
tomer experience; app full of bugs
*Constant emails and annoying
pop-ups *Podcasts switch too of-
ten; excessive commercials *Un-
skippable ads and a terrible podcast
player
Q3:Give me an overview of all
the reviews.A3:Despite frus-
tration over bugs and instability,
users still appreciate music recom-
mendations, though many consider
switching due to technical issues
and ads.*Still love the playlist suggestions,
but crashes make it unusable *The
ads are out of control—feel like
I’m paying for promotions *Great
discovery features, but app stability
needs fixes *Considering switching
to another service because of con-
stant bugs
E. Retrieval-Augmented QA
A retrieval-augmented QA system was constructed to enable
interactive queries on large-scale review datasets. Reviews
were segmented into overlapping text chunks and encoded
into vector embeddings indexed in a FAISS-based retriever.
Upon receiving a user query, the system employed cosine
similarity scoring to retrieve the most relevant text chunks,
as described in Eq. (3). A structured prompt—enhanced via
meta-prompting—guided the answering LLM to generate con-
cise, contextually relevant responses referencing the retrieved
chunks. Evaluation results are presented in Section V .
V. RESULTS
This section presents the results of the experiments de-
scribed in Section IV. All results reported in the following
subsections were generated using the GPT-4 model, selected
for its superior performance as detailed in Section III.A. Re-
sults are systematically reported for the key evaluation tasks:TABLE II
SAMPLE PIPELINE OUTPUT SHOWING THE ORIGINAL REVIEW SENTENCE,
EXTRACTED ASPECT–SENTIMENT PAIRS,AND USER RECOMMENDATIONS.
Sentence Aspect–Sentiment Reco.
don’t get me started on finding
old documents, a feature that was
said to have improved.document
finding– negativeimprove
document
search
feature
everything takes multiple steps
and functionality is now slower.functionality–
negative,speed–
negativereduce
steps;
improve
speed
i could not turn auto save off, and
it was not saving even though i
had a stable internet connection.auto-save
function–
negative,stable
internet– neutralfix
auto-save
feature
i use it for all my classes and it
saves me money on notebooks and
it’s way easier for organization.classes– positive,
organization–
positive—
it’s annoying that notability
doesn’t offer landscape page when
wider view is needed.landscape page–
negativeoffer
landscape
view
the new evernote home for my
desktop is amazing and
customizable!evernote home–
positive—
Aspect extraction, sentiment classification, topic modeling,
quality control and computational efficiency.
A. Aspect Extraction
Table V summarizes the aspect extraction performance on
the AWARE dataset, comparing the LLM-based approach
(GPT-4, prompt-based) to a fine-tuned DeBERTa-v3-large
model [21]. This evaluation leverages AWARE’s human-
annotated aspect terms and categories as ground truth.
The LLM-based approach shows a notable improvement
of approximately 5.1% in F1-score compared to the fine-
tuned transformer baseline. This indicates its superior ability
to capture implicit and nuanced aspects within reviews, as well
as handling domain-specific terminology effectively.
B. Sentiment Classification
Table VI presents the sentiment distributions predicted by
the LLM-based classifier and the baseline V ADER model

TABLE III
LLM-GENERATEDTOPICMODELINGRESULTS ON THESPOTIFYREVIEWSDATASET
Topic
IDCount Top Keywords LLM-Generated Topic Label Topic Summary
0 817 spotify, it, the, and, my, to,
app, on, is, haveUnexpected Playback and
Queue FailuresSpotify users are experiencing technical issues such as lag, the
app making its own order for the queue, logging out
unexpectedly, and problems with playing liked songs. Some users
are also having trouble with their library not loading correctly.
Despite these issues, other users appreciate Spotify for its value in
discovering new artists and podcasts.
1 367 spotify, music, to, and, the,
is, premium, you, for, ofPremium Subscription
FrustrationsSpotify users are expressing frustration with the premium
requirement to access lyrics, limited playlist customization, and
the high frequency of ads, which they find annoying and
intrusive. However, some users appreciate Spotify’s improved
features and extensive music library.
2 366 shuffle, smart, the, it, to,
songs, and, off, on, sameUnintuitive Shuffle Controls Spotify users are expressing frustration with the shuffle function
on the platform, stating that it prioritizes certain songs or artists,
does not work as expected, and cannot be disabled without
subscribing to premium, leading to a poor user experience and a
lack of control over their listening experience.
3 357 offline, downloaded, mode,
the, to, app, when, internet,
it, andOffline Playback Issues Spotify users are experiencing issues with the app’s offline mode,
as it incorrectly shows an offline notification even when the
device is connected to the internet, and has trouble loading
downloaded content after 24 hours, affecting usage in areas with
poor signal or for conserving data. Some users also report
difficulty using the app with mobile data.
4 296 podcast, podcasts, to, the,
it, and, for, episode, app, isPodcast Audio Stability and
Transcript NeedsSpotify receives positive reviews for its podcast feature, with
users appreciating its convenience for nightly listening and access
to specific podcasts like Joe Rogan. However, some users
experience issues such as intermittent audio playback on certain
devices, podcast playlist annoyances, and a lack of control over
unwanted podcast suggestions. A common request is for the
inclusion of transcripts under episodes.
TABLE IV
LLM-GENERATEDSUMMARY OFPOSITIVE/NEGATIVEFEEDBACK ON
SPOTIFYSERVICEASPECTS.
Aspect Positive Feedback Negative Feedback
Music Recommendations Praised for quality –
Service Reliability Some still recommend Considering cancellation
Competitive Position Better than competitors Significant technical problems
TABLE V
ASPECTEXTRACTIONPERFORMANCE ONAWARE DATASET
Model Precision Recall F1-Score
GPT-4 (Prompt-based) 0.892 0.892 0.892
DeBERTa-v3-large 0.847 0.835 0.841
on the AWARE dataset. These results are benchmarked
against AWARE’s manually labeled sentiment annotations
(positive/negative/neutral).
TABLE VI
PREDICTEDSENTIMENTDISTRIBUTION ONAWARE DATASET
Model Positive Negative Neutral
LLM-based 22.3% 29.8% 47.9%
V ADER 8.7% 3.1% 88.2%The LLM-based model generates a balanced sentiment
distribution, overcoming the strong neutral bias shown by
V ADER. However, detailed sentiment classification metrics
(see Table VII) reveal areas for improvement.
TABLE VII
SENTIMENTCLASSIFICATIONMETRICS ONAWARE DATASET
Sentiment Precision Recall F1-Score
Positive 0.061 0.589 0.110
Negative 0.174 0.443 0.250
Neutral 0.917 0.498 0.645
Weighted Avg 0.825 0.496 0.594
Positive sentiment classification suffers from low precision,
indicating frequent false positives, while negative detection
shows moderate recall yet limited precision. Neutral sentiment
is precise but less comprehensive. Overall, a weighted F1 of
0.594 highlights areas for further refinement.
C. Topic Modeling
The topic modeling results, presented in Table VIII, assess
the improvement gained by integrating an LLM into the
BERTopic pipeline on the Spotify dataset.
The LLM-enhanced approach improves silhouette from
negative to positive, indicating enhanced cluster separation

TABLE VIII
SILHOUETTECOEFFICIENTS FORTOPICMODELING
Metric BERTopic Only BERTopic + LLM
Silhouette Score -0.0313 0.0302
and topic distinctiveness, and validating the benefit of LLM-
generated labels for interpretability.
D. Question and Answering (QA)
For retrieval of reviews, we sampled five Spotify-centric
queries and retrieved the top K = 10 review chunks for each.
We measured two unsupervised metrics:
•Average Cosine Similarity: the mean cosine similarity
between each query embedding and its top-10 chunk
embeddings.
•Retrieval Diversity: the fraction of unique review IDs
among all retrieved chunks (distinct IDs / 10).
Our retriever achieved perfect diversity and cosine scores
from 0.618 to 0.754, demonstrating reliable, on-topic retrieval.
Table IX summarizes these proxy metrics.
For generation of answers, we randomly sampled 20 gen-
erated answers (each paired with its cited snippets) and anno-
tated them ourselves, confirming that each answer (1) reflected
the cited excerpts, (2) covered the main points of those
excerpts, and (3) was written in clear, reader-friendly prose.
We found the responses to be accurate and comprehensive.
TABLE IX
RETRIEVALPROXYMETRICS(K=10)FORSELECTEDSPOTIFYQUERIES
(HIGHER DIVERSITY IS BETTER)
Query Avg. Diversity
Cosine Sim.
What complaints do users have about
Spotify’s offline-mode buffering?0.713 1.0
What do listeners say about Spotify
crashing or freezing on startup?0.754 1.0
How do listeners describe the app’s offline
playback experience?0.696 1.0
How do users report errors or failures
when downloading songs for offline use?0.618 1.0
What do users say about Spotify’s
crossfade and track-transition experience?0.650 1.0
E. Computational Efficiency
Table X compares training time, inference speed, resource
requirements, scalability, and adaptability between traditional
methods and our LLM-based pipeline.
LLM-based methods deliver immediate adaptability and
high scalability, at the cost of increased resource demands and
moderate inference latency.
VI. CONCLUSION ANDFUTUREWORK
Our experimental evaluation demonstrates that the LLM-
based review analysis framework significantly outperforms tra-
ditional NLP baselines across multiple tasks. In particular, theTABLE X
COMPUTATIONALEFFICIENCYCOMPARISON
Aspect Traditional Methods LLM-based Approach
Training Time High Minimal (prompt design)
Inference Speed Fast Moderate
Resource Requirements Lower Higher
Scalability Limited High
Adaptability Requires retraining Immediate adaptation
prompt-driven GPT-4 model achieved a 5.1% F1 improvement
in aspect extraction compared to a fine-tuned DeBERTa-v3-
large system (Table V), produced a more balanced sentiment
distribution with a weighted F1 of 0.594 versus V ADER’s neu-
tral bias (Table VII), and markedly enhanced topic coherence
by shifting silhouette scores from –0.0313 to 0.0302 when
integrated with BERTopic (Table VIII).and enabled interactive
developer insights through retrieval-augmented QA, achieving
perfect retrieval diversity (100%) and high relevance (cosine:
0.618–0.754) with human-verified response accuracy. These
gains attest to the framework’s ability to capture nuanced,
context-specific information and generate semantically coher-
ent summaries, validating its utility for large-scale app review
mining.
Despite these performance advantages, the LLM-based
pipeline incurs higher computational overhead and moderate
inference latency relative to traditional methods (Table X).
To address these challenges, future work will investigate
techniques for model explainability—such as layer-wise at-
tribution and prompt-tuning diagnostics—to illuminate the
decision process of LLM-driven sentiment classifiers. We also
plan to explore agentic, interactive architectures that allow
dynamic adaptation of prompts based on real-time feedback,
as well as extend our approach to multilingual and cross-
domain scenarios to leverage LLMs’ broad linguistic capabili-
ties. Furthermore, integrating explainable topic discovery with
actionable reasoning promises to deliver richer, developer-
focused insights that align with evolving user priorities.
In conclusion, by bridging numeric star ratings with feature-
level sentiment and retrieval-augmented question answering,
our study offers a comprehensive, human-centered perspective
on user feedback. As app marketplaces continue to grow in
size and complexity, embedding explainable, adaptable AI
solutions within real-time development workflows will be
essential for anticipating user needs, prioritizing feature im-
provements, and ultimately crafting applications that resonate
more deeply with their audiences.
Ethical considerations. LLMs can be biased or invent facts.
We reduce risk by restricting QA answers to retrieved, cited
review snippets, keeping an audit trail of prompts/outputs,
returning “not stated” when evidence is missing, and requiring
human or rule-based checks before any real-world decisions.
REFERENCES
[1] S. Biswas, K. Young, and J. Griffith, “A comparison of automatic
labelling approaches for sentiment analysis,” 2022. [Online]. Available:
https://arxiv.org/pdf/2211.02976

[2] C. Kumaresan and P. Thangaraju, “Sentiment analysis in multiple lan-
guages: A review of current approaches and challenges,”REST Journal
on Data Analytics and Artificial Intelligence, vol. 2, no. 1, pp. 8–15,
2023.
[3] S. Yadav and M. Sarkar, “Enhancing sentiment analysis using domain-
specific lexicon: A case study on gst,” inProceedings of the 2018
International Conference on Advances in Computing, Communications
and Informatics (ICACCI). Bangalore, India: IEEE, Sep. 2018, pp.
1109–1114.
[4] H. Zhao, H. Chen, F. Yang, N. Liu, H. Deng, H. Cai, S. Wang, D. Yin,
and M. Du, “Explainability for large language models: A survey,”ACM
Transactions on Intelligent Systems and Technology, vol. 15, no. 2, pp.
1–38, 2024. [Online]. Available: https://doi.org/10.1145/3639372
[5] M. Agua, N. Ant ´onio, P. Carrasco, and C. Rassal, “Large language
models powered aspect-based sentiment analysis for enhanced customer
insights,”Tourism Management Studies, vol. 21, no. 1, pp. 1–19, 2025.
[6] F. A. Shah, A. Sabir, R. Sharma, and D. Pfahl, “How effectively do
llms extract feature-sentiment pairs from app reviews?” 2024. [Online].
Available: https://arxiv.org/pdf/2409.07162
[7] K. Tabianan, D. Arputharaj, M. N. B. Abd Rani, and S. Nagalingham,
“Data analysis and rating prediction on google play store using data
mining techniques,”Journal of Data Science, vol. 2022, no. 01, 2022.
[8] C. Gao, J. Zeng, Z. Wen, D. Lo, X. Xia, I. King, and M. R. Lyu,
“Emerging app issue identification via online joint sentiment-topic
tracing,” 2020. [Online]. Available: https://arxiv.org/abs/2008.09976
[9] A. Mahmood, “Identifying the influence of various factors of apps
on google play app ratings,”Journal of Data, Information and
Management, vol. 2, no. 1, pp. 15–23, 2020. [Online]. Available:
https://doi.org/10.1007/s42488-019-00015-w
[10] R. Aralikatte, G. Sridhara, N. Gantayat, and S. Mani, “Fault in your
stars: An analysis of android app reviews,” 2017. [Online]. Available:
https://arxiv.org/abs/1708.04968v1
[11] S. Ranjan and S. Mishra, “Comparative sentiment analysis of app
reviews,” 2020. [Online]. Available: https://arxiv.org/abs/2006.09739v1
[12] S. Sadiq, M. Umer, S. Ullah, S. Mirjalili, V . Rupapara, and
M. Nappi, “Discrepancy detection between user reviews and numeric
ratings of google app store using deep learning,”Expert Systems
with Applications, vol. 181, p. 115111, 2021. [Online]. Available:
https://doi.org/10.1016/j.eswa.2021.115111
[13] Y . Gaoet al., “Retrieval-augmented generation for large language
models: A survey,” 2024, arXiv preprint arXiv:2312.10997.
[14] OpenAI, “GPT-4 technical report,”CoRR, vol. abs/2303.08774, 2023.
[Online]. Available: https://doi.org/10.48550/arXiv.2303.08774
[15] H. Touvron, L. Martin, K. R. Stone, P. Albert, A. Almahairi,
Y . Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale,
D. Bikel, and et al., “Llama 2: Open foundation and fine-tuned chat
models,”arXiv preprint arXiv:2307.09288, 2023. [Online]. Available:
https://arxiv.org/abs/2307.09288
[16] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S.
Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample,
L. Saulnier, L. Renard Lavaud, M.-A. Lachaux, P. Stock, T. Le Scao,
T. Lavril, T. Wang, T. Lacroix, W. El Sayed, and et al., “Mistral
7B,”arXiv preprint arXiv:2310.06825, 2023. [Online]. Available:
https://arxiv.org/abs/2310.06825
[17] P. Rathi, “Google play store reviews,”
https://www.kaggle.com/datasets/prakharrathi25/google-play-store-
reviews, 2021, accessed: 2025-06-18.
[18] A. Kumar, “Spotify reviews playstore daily update,”
https://www.kaggle.com/datasets/ashishkumarak/spotify-reviews-
playstore-daily-update‘, 2022, accessed: 2025-06-18.
[19] C. R. Rivero, L. Alharahsheh, X. Yue, and C.-T. Lu, “Aware: A
large-scale dataset for mobile app review analysis,” 2021. [Online].
Available: https://zenodo.org/record/5528481
[20] C. Hutto and E. Gilbert, “Vader: A parsimonious rule-
based model for sentiment analysis of social media text,” in
Proceedings of the International AAAI Conference on Web and
Social Media (ICWSM), 2014, pp. 216–225. [Online]. Available:
https://ojs.aaai.org/index.php/ICWSM/article/view/14550
[21] Heng Yang and Biqing Zeng and Mayi Xu and Tianxing
Wang, “yangheng/deberta-v3-large-absa-v1.1: Deberta-v3
large model fine-tuned for aspect-based sentiment analysis,”
https://huggingface.co/yangheng/deberta-v3-large-absa-v1.1, Hugging
Face, 2024, accessed: 2025-06-18; MIT license; trained on
ABSADatasets, 435M parameters.