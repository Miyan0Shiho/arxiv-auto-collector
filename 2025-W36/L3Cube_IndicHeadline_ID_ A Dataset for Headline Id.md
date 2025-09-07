# L3Cube-IndicHeadline-ID: A Dataset for Headline Identification and Semantic Evaluation in Low-Resource Indian Languages

**Authors**: Nishant Tanksale, Tanmay Kokate, Darshan Gohad, Sarvadnyaa Barate, Raviraj Joshi

**Published**: 2025-09-02 16:54:30

**PDF URL**: [http://arxiv.org/pdf/2509.02503v1](http://arxiv.org/pdf/2509.02503v1)

## Abstract
Semantic evaluation in low-resource languages remains a major challenge in
NLP. While sentence transformers have shown strong performance in high-resource
settings, their effectiveness in Indic languages is underexplored due to a lack
of high-quality benchmarks. To bridge this gap, we introduce
L3Cube-IndicHeadline-ID, a curated headline identification dataset spanning ten
low-resource Indic languages: Marathi, Hindi, Tamil, Gujarati, Odia, Kannada,
Malayalam, Punjabi, Telugu, Bengali and English. Each language includes 20,000
news articles paired with four headline variants: the original, a semantically
similar version, a lexically similar version, and an unrelated one, designed to
test fine-grained semantic understanding. The task requires selecting the
correct headline from the options using article-headline similarity. We
benchmark several sentence transformers, including multilingual and
language-specific models, using cosine similarity. Results show that
multilingual models consistently perform well, while language-specific models
vary in effectiveness. Given the rising use of similarity models in
Retrieval-Augmented Generation (RAG) pipelines, this dataset also serves as a
valuable resource for evaluating and improving semantic understanding in such
applications. Additionally, the dataset can be repurposed for multiple-choice
question answering, headline classification, or other task-specific evaluations
of LLMs, making it a versatile benchmark for Indic NLP. The dataset is shared
publicly at https://github.com/l3cube-pune/indic-nlp

## Full Text


<!-- PDF content starts -->

L3Cube-IndicHeadline-ID: A Dataset for Headline Identification and
Semantic Evaluation in Low-Resource Indian Languages
Nishant Tanksale1,3Tanmay Kokate1,3Darshan Gohad1,3
Sarvadnyaa Barate1,3Raviraj Joshi2,3∗
1Department of Information Technology, PICT, Pune
2Indian Institute of Technology Madras, Chennai
3L3Cube Labs, Pune
Abstract
Semantic evaluation in low-resource languages
remains a major challenge in NLP. While sen-
tence transformers have shown strong perfor-
mance in high-resource settings, their effective-
ness in Indic languages is underexplored due to
a lack of high-quality benchmarks. To bridge
this gap, we introduce L3Cube-IndicHeadline-
ID, a curated headline identification dataset
spanning ten low-resource Indic languages:
Marathi, Hindi, Tamil, Gujarati, Odia, Kan-
nada, Malayalam, Punjabi, Telugu, Bengali and
English. Each language includes 20,000 news
articles paired with four headline variants: the
original, a semantically similar version, a lex-
ically similar version, and an unrelated one,
designed to test fine-grained semantic under-
standing. The task requires selecting the cor-
rect headline from the options using article-
headline similarity. We benchmark several sen-
tence transformers, including multilingual and
language-specific models, using cosine sim-
ilarity. Results show that multilingual mod-
els consistently perform well, while language-
specific models vary in effectiveness. Given
the rising use of similarity models in Retrieval-
Augmented Generation (RAG) pipelines, this
dataset also serves as a valuable resource for
evaluating and improving semantic understand-
ing in such applications. Additionally, the
dataset can be repurposed for multiple-choice
question answering, headline classification, or
other task-specific evaluations of LLMs, mak-
ing it a versatile benchmark for Indic NLP. The
dataset is shared publicly at https://github.
com/l3cube-pune/indic-nlp .
1 Introduction
Natural Language Processing (NLP) faces persis-
tent challenges in addressing low-resource lan-
guages, primarily due to a lack of standardized
datasets and methodologies. Despite their cultural
and linguistic significance, many Indian languages
∗Correspondence: ravirajoshi@gmail.comremain underrepresented in research, creating bar-
riers to developing effective solutions (Dongare,
2024; Magueresse et al., 2020).
Social media platforms, in particular, underscore
the pressing need for tailored NLP solutions. Text-
driven interactions in regional languages dominate
these platforms, underscoring the need for semantic
understanding technologies that can process and
analyse the growing volume of regional language
data. Concurrently, the emergence of Retrieval-
Augmented Generation (RAG) methods has driven
the need for high-quality sentence embeddings to
enhance information retrieval and generation tasks
(Panchal and Shah, 2024)
Sentence transformers, with their ability to gen-
erate semantically meaningful embeddings, pro-
vide a promising avenue for advancing NLP in
low-resource contexts. By incorporating pooling
layers over the outputs of BERT-based models,
these transformers produce fixed-size embeddings
that are both efficient and effective for downstream
tasks (Reimers and Gurevych, 2019).
Recent years have witnessed the development of
BERT-based models for Indic languages, reflect-
ing the growing need for NLP tools catering to
non-English-speaking populations (Deode et al.,
2023). However, a significant gap persists in cre-
ating robust datasets for assessing these models in
multilingual and low-resource contexts.
To address this gap, we introduce the L3Cube-
IndicHeadline-ID1: a novel benchmark comprising
20,000 news articles for each of ten major Indic
languages (Mirashi et al., 2024). Each article is
paired with four titles: the original, a semantically
similar title, a lexically similar title, and an unre-
lated title. This dataset evaluates models based
on their ability to identify the original title using
cosine similarity, offering a scalable and nuanced
evaluation approach for sentence-level semantic
understanding in Indic languages. We particularly
1l3cube-pune/IndicHeadline-IDarXiv:2509.02503v1  [cs.CL]  2 Sep 2025

focus on the semantic understanding aspect due
to the growing importance of similarity models in
recent Retrieval-Augmented Generation (RAG) ap-
plications. Beyond semantic similarity evaluation,
the dataset can also be used for headline identifi-
cation using classification models or framed as a
multiple-choice question answering task, making it
versatile for various natural language understand-
ing applications.
2 Related Work
Semantic evaluation in multilingual and low-
resource languages has gained significant atten-
tion in recent years. Foundational efforts such as
the Semantic Textual Similarity (STS) task and
the Semantic Textual Relatedness (STR) dataset
have provided valuable sentence pair annotations
across multiple languages (Ousidhoum et al., 2024;
Conneau et al., 2018). However, these resources
predominantly emphasize high-resource languages
and rely on extensive manual annotations, limiting
their applicability and scalability in low-resource
contexts.
For Indic languages, the IndicNLP Suite intro-
duced by (Kakwani et al., 2020) has made notable
strides by offering pre-trained language models and
evaluation benchmarks. However, its headline pre-
diction dataset is not publicly available, creating
a gap in publicly available resources for semantic
evaluation in Indian languages. Similarly, multilin-
gual datasets like MASSIVE have extended cover-
age to 51 languages but are primarily designed for
token-level tasks such as intent recognition and slot
filling, leaving sentence-level evaluation in Indic
languages largely unaddressed (FitzGerald et al.,
2023).
Efforts such as IndicSentEval by (Aravapalli
et al., 2024) further advance the field by evalu-
ating the linguistic properties encoded by multilin-
gual transformer models. Their dataset and probing
tasks focus on surface, syntactic, and semantic fea-
tures across six Indic languages, shedding light on
the strengths of both universal and Indic-specific
models. However, this work emphasizes probing
linguistic representations rather than sentence-level
semantic alignment.
Efforts such as IndicMT Eval (B et al., 2023)
have explored meta-evaluation frameworks for ma-
chine translation metrics, addressing the limitations
of standard MT evaluation methods for Indian lan-
guages. The dataset captures linguistic and culturalnuances across multiple Indic languages, providing
a benchmark for translation quality assessment.
Our work complements and builds upon these
initiatives by introducing the first publicly available
large-scale dataset specifically designed for head-
line identification across ten Indic languages (Mi-
rashi et al., 2024). By systematically generating
diverse candidate headlines, we address limitations
associated with manual annotations. Our frame-
work provides a robust, scalable mechanism for
evaluating sentence transformers, offering a unique
contribution to sentence-level semantic evaluation
in low-resource Indic languages.
3 Methodology
The dataset used in this study is based on L3Cube-
IndicNews (Mirashi et al., 2024) and includes
20,000 samples for each of ten Indic languages:
Marathi, Hindi, Tamil, Gujarati, Odia, Kannada,
Malayalam, Punjabi, Telugu, and Bengali. Each
sample comprises a news article paired with its
original headline.
To construct IndicHeadline-ID, a headline iden-
tification dataset, three additional candidate head-
lines were selected for each article, forming a set of
four options: the original headline, a semantically
similar headline, a lexically similar headline, and a
random headline. The original serves as the ground
truth. This setup enables a nuanced evaluation
of sentence transformers, especially for retrieval-
augmented generation (RAG) tasks, by testing their
ability to capture semantic meaning, distinguish
lexical overlap, and reject irrelevant content. The
candidate titles are selected as follows:
•Original Title : The true headline of the news
article, directly sourced from the IndicNews
dataset. It serves as the ground truth and is
used to assess the model’s ability to accurately
align embeddings with the article’s context.
•Semantically Similar Title : A different ti-
tle that expresses the same core meaning as
the original but might use different words
or phrasing. These titles are selected using
language-specific sentence embedding mod-
els developed by L3Cube Labs (Deode et al.,
2023). Cosine similarity is computed between
the embedding of the original title and all
other titles in the dataset. The most seman-
tically similar title is selected, excluding the

Figure 1: Methodology for creating the dataset with candidate titles
Original Title / Headline Semantic Title Lexical Title Random Title
Chhattisgarh Polls:
Congress Releases First
List, Fields CM Bhupesh
Baghel From PatanAhead Of Phase-1 Polling
In Chhattisgarh, CM
Baghel Faces The Heat
For Mahadev App Case.
Top PointsChhattisgarh CM Bhu-
pesh Baghel Resigns Af-
ter Congress Party’s Shock
DefeatChandrayaan-3 Launches
Today: 10 Interesting
Facts About ISRO’s Third
Moon Mission
Hyundai Ioniq 5 Electric
SUV With 631 Km Range
— First Look ReviewHyundai Ioniq 5 First
Drive Review: Lots Of
Tech, Space and Style —
Know If This EV Is Value-
For-MoneyBMW iX Electric SUV
First Look Review —
Know About Design And
InteriorsNorth India Likely To
Usher In New Year With
More Chill and Rain, No
Respite In Sight From Fog
Till Saturday
Table 1: Examples of English headlines with semantic, lexical, and random negatives.
original itself, for each instance. This can-
didate is included to challenge the model by
introducing a plausible yet incorrect option,
making the selection task more difficult and
testing the model’s ability to distinguish be-
tween closely related semantic content.
•Lexically Similar Title : A title that shares
significant word overlap with the original
headline but differs in meaning. These titles
are identified by computing word frequency-
based vector representations for all titles and
retrieving the one with the highest lexical sim-
ilarity to the original. This comparison cap-
tures similarity at the surface level of word
usage, without incorporating deeper semantic
relationships.•Random Title : An unrelated headline arbi-
trarily chosen from the dataset. These distrac-
tor titles are included to evaluate the model’s
robustness in rejecting irrelevant content and
simulating realistic conditions with unrelated
information.
The examples for English are provided in Ta-
ble 1.
During evaluation, each news article and its can-
didate headlines were encoded into embeddings
using sentence transformer models (Deode et al.,
2023). Cosine similarity scores were calculated
between the article embedding and each candidate
headline embedding, and the candidate with the
highest score was identified as the predicted match
(Lin et al., 2017; Mueller and Thyagarajan, 2016).
This dataset-based evaluation eliminates the need

x: language-specific model
Multilingual Models Language-Specific Models
indic-
sentence-
bert-nliindic-
sentence-
similarity-sbertmultilingual-
e5-basemuril base
cased x-bertx-sentence-
similarity-
sbert
Marathi 0.5756 0.5561 0.5747 0.5831 0.5948 0.5579
Hindi 0.8428 0.8514 0.9089 0.7449 0.7794 0.8625
Tamil 0.8415 0.8481 0.8815 0.7645 0.7543 0.8574
Gujarati 0.8096 0.8082 0.8390 0.7051 0.7392 0.8162
Odia 0.8210 0.7891 0.8468 0.6136 0.6732 0.8039
Kannada 0.8650 0.8730 0.8912 0.7953 0.7935 0.8918
Malayalam 0.8092 0.8102 0.8155 0.6905 0.6891 0.8239
Punjabi 0.9712 0.9609 0.9715 0.9257 0.9193 0.9728
Telugu 0.8075 0.8090 0.8400 0.7129 0.7270 0.6395
Bengali 0.8009 0.8123 0.8385 0.7115 0.6897 0.8305
English 0.6554 0.6312 0.7681 – – –
Table 2: Performance of Multilingual and Language-Specific Models on Different Languages (x:language).
for manual similarity annotations, which are chal-
lenging to scale for low-resource languages.
The algorithmic generation of semantic and
lexical candidates ensures consistency across lan-
guages and scalability. By including all headline
types, the dataset provides a detailed assessment of
model strengths and weaknesses, which cannot be
achieved with single-score metrics like BLEU or
METEOR (Denkowski and Lavie, 2014). This ap-
proach offers valuable insights into sentence trans-
formers’ performance on low-resource Indic lan-
guages and paves the way for further advancements
in the domain.
4 Models
We benchmarked a set of multilingual and Indic-
specific sentence transformer models to evaluate
semantic understanding in low-resource Indic lan-
guages. Multilingual models like multilingual-e5-
base (Wang et al., 2022) and google-muril-base-
cased (Khanuja et al., 2021) were chosen for
their strong cross-lingual transfer capabilities and
proven performance on semantic similarity tasks.
Indic-specific models such as IndicSBERT (Deode
et al., 2023)and language specific BERT (Joshi,
2022) were included as they are pre-trained or fine-
tuned on Indian language corpora, making them
more attuned to the linguistic characteristics of
the region. This diverse selection allowed us to
compare general-purpose multilingual models with
those explicitly designed for Indic languages. The
goal was to understand how well each model cap-
tures sentence-level semantics in scenarios withlimited language resources.
5 Results and Discussion
Among the multilingual models, indic-sentence-
bert-nli and indic-sentence-similarity-sbert demon-
strated consistent performance across most lan-
guages (see Table 2). However, multilingual-e5-
base outperformed other models for Hindi and
Tamil, achieving cosine similarity scores of 0.9089
and 0.8815, respectively, and also showed strong
results for English.
Language-specific models generally achieved
higher scores than multilingual counterparts for cer-
tain languages. For instance, the Kannada-specific
model obtained the highest score of 0.8914. In con-
trast, Marathi and Bengali exhibited comparatively
lower performance across all models, reflecting the
challenges of semantic evaluation in these contexts.
Overall, the results highlight the strengths of
language-specific models in capturing nuanced se-
mantics within individual languages, while multi-
lingual models provide a robust baseline for cross-
lingual evaluation.
6 Limitations
While the Indic Headline Identification Dataset en-
sures linguistic consistency by sourcing news ar-
ticles in formal dialects, it does not account for
informal or dialectal variations. This limitation
may impact the generalizability of findings to less
formal contexts, such as social media or colloquial
speech.

Additionally, the dataset’s reliance on algorithmi-
cally selected candidates, while scalable, may not
capture the full spectrum of semantic and lexical
diversity observed in real-world scenarios. Future
work could explore incorporating manually anno-
tated datasets or augmenting the framework with
synthetic data to address these limitations.
7 Conclusion
In this work, we present L3Cube-IndicHeadline-ID,
a comprehensive dataset and evaluation framework
for semantic similarity across ten Indic languages
and English, filling a crucial gap in the landscape of
low-resource NLP. Through extensive benchmark-
ing of state-of-the-art sentence transformers, we
offer key insights into the strengths and limitations
of both multilingual and language-specific models.
Our results demonstrate the strong performance
of models like multilingual-e5-base, while also un-
derscoring the benefits of targeted fine-tuning for
individual languages. These findings emphasize
the importance of balancing generalization with
specialization in multilingual NLP. Moreover, the
persistent challenges in low-resource settings call
for future research into more adaptable multilingual
architectures and equitable pre-training paradigms
that better serve diverse linguistic communities.
By releasing L3Cube-IndicHeadline-ID and as-
sociated benchmarks, we provide an essential re-
source to catalyze further research in Indic seman-
tic similarity. We hope this work will drive the
development of more inclusive and representative
language technologies, empowering NLP applica-
tions across the linguistically rich and diverse In-
dian subcontinent.
Acknowledgment
This work was carried out under the mentorship of
L3Cube Labs, Pune. We would like to express our
gratitude towards our mentor for his continuous
support and encouragement. This work is a part of
the L3Cube-IndicNLP Project.
References
A. Aravapalli et al. 2024. Indicsenteval: How effec-
tively do multilingual transformer models encode
linguistic properties for indic languages? ArXiv ,
abs/2410.02611.
Ananya Sai B, Tanay Dixit, Vignesh Nagarajan, Anoop
Kunchukuttan, Pratyush Kumar, Mitesh M. Khapra,and Raj Dabre. 2023. Indicmt eval: A dataset to
meta-evaluate machine translation metrics for indian
languages. In Proceedings of the 61st Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 14210–14228. Asso-
ciation for Computational Linguistics.
Alexis Conneau, Ruty Rinott, Guillaume Lample, Ad-
ina Williams, Samuel R. Bowman, Holger Schwenk,
and Veselin Stoyanov. 2018. XNLI: Evaluating cross-
lingual sentence representations. In Proceedings of
the 2018 Conference on Empirical Methods in Nat-
ural Language Processing , pages 2475–2485, Brus-
sels, Belgium. Association for Computational Lin-
guistics.
Michael Denkowski and Alon Lavie. 2014. Meteor
universal: Language specific translation evaluation
for any target language. In Proceedings of the Ninth
Workshop on Statistical Machine Translation , pages
376–380, Baltimore, Maryland, USA. Association
for Computational Linguistics.
S. Deode, J. Gadre, A. Kajale, A. Joshi, and R. Joshi.
2023. L3cube-indicsbert: A simple approach for
learning cross-lingual sentence representations using
multilingual bert. ArXiv , abs/2304.11434.
Pratibha Dongare. 2024. Creating corpus of low re-
source indian languages for natural language process-
ing: Challenges and opportunities. In Proceedings
of the 7th Workshop on Indian Language Data: Re-
sources and Evaluation , pages 54–58. ELRA and
ICCL.
Jack FitzGerald et al. 2023. Massive: A 1m-example
multilingual natural language understanding dataset
with 51 typologically-diverse languages. In Proceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 4277–4302. Association for Computational
Linguistics.
Raviraj Joshi. 2022. L3cube-hindbert and devbert:
Pre-trained bert transformer models for devanagari
based hindi and marathi languages. arXiv preprint
arXiv:2211.11418 .
Divyanshu Kakwani et al. 2020. Indicnlpsuite: Mono-
lingual corpora, evaluation benchmarks and pre-
trained multilingual language models for indian lan-
guages. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2020 , pages 4948–4961.
Association for Computational Linguistics.
Sanchit Khanuja, Krishnan Sankaran, Niket Nangia,
Manish Srivastava, Partha Gupta, Katharina Kann,
and Anil Kumar Sinha. 2021. Muril: Multilingual
representations for indian languages. In Findings
of the Association for Computational Linguistics:
EMNLP 2021 , pages 4902–4912. Association for
Computational Linguistics.
Zhouhan Lin, Minwei Feng, Cicero Nogueira dos San-
tos, Mo Yu, Bing Xiang, Bowen Zhou, and Yoshua

Bengio. 2017. A structured self-attentive sentence
embedding. arXiv preprint arXiv:1703.03130 .
A. Magueresse, V . Carles, and E. Heetderks. 2020. Low-
resource languages: A review of past work and future
challenges. ArXiv , abs/2006.07264.
A. Mirashi et al. 2024. L3cube-indicnews: News-based
short text and long document classification datasets
in indic languages. ArXiv , abs/2401.02254.
Jonas Mueller and Aditya Thyagarajan. 2016. Semantic
textual similarity using siamese recurrent architec-
tures. arXiv preprint arXiv:1606.05495 .
N.D. Ousidhoum et al. 2024. Semrel2024: A collection
of semantic textual relatedness datasets for 14 lan-
guages. In Proceedings of the Annual Meeting of the
Association for Computational Linguistics .
Brijeshkumar Panchal and Apurva Shah. 2024. Nlp
research: A historical survey and current trends in
global, indic, and gujarati languages. In Proceedings
of the 2024 International Conference on Ubiquitous
Intelligence and Systems (ICUIS) , pages 1263–1272.
IEEE.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
Conference on Empirical Methods in Natural Lan-
guage Processing .
Sheng Wang, Tianyi Ma, Yuan Xie, Ying Li, Yelong Li,
and Jimmy Lin. 2022. E5: Embedding for everyone.
arXiv preprint arXiv:2212.09543 .