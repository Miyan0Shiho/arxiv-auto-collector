# Hakim: Farsi Text Embedding Model

**Authors**: Mehran Sarmadi, Morteza Alikhani, Erfan Zinvandi, Zahra Pourbahman

**Published**: 2025-05-13 10:57:32

**PDF URL**: [http://arxiv.org/pdf/2505.08435v2](http://arxiv.org/pdf/2505.08435v2)

## Abstract
Recent advancements in text embedding have significantly improved natural
language understanding across many languages, yet Persian remains notably
underrepresented in large-scale embedding research. In this paper, we present
Hakim, a novel state-of-the-art Persian text embedding model that achieves a
8.5% performance improvement over existing approaches on the FaMTEB benchmark,
outperforming all previously developed Persian language models. As part of this
work, we introduce three new datasets - Corpesia, Pairsia-sup, and
Pairsia-unsup - to support supervised and unsupervised training scenarios.
Additionally, Hakim is designed for applications in chatbots and
retrieval-augmented generation (RAG) systems, particularly addressing retrieval
tasks that require incorporating message history within these systems. We also
propose a new baseline model built on the BERT architecture. Our language model
consistently achieves higher accuracy across various Persian NLP tasks, while
the RetroMAE-based model proves particularly effective for textual information
retrieval applications. Together, these contributions establish a new
foundation for advancing Persian language understanding.

## Full Text


<!-- PDF content starts -->

arXiv:2505.08435v2  [cs.CL]  14 May 2025Hakim: Farsi Text Embedding Model
Mehran Sarmadi‡§, Morteza Alikhani‡§, Erfan Zinvandi§, Zahra Pourbahman§
‡MCINEXT,§Sharif University of Technology
{mehran.sarmadi99, morteza.alikhani95, e.zeynvandi1376, zahra.pourbahman95}@sharif.edu
Abstract
Recent advancements in text embedding have
significantly improved natural language un-
derstanding across many languages, yet Per-
sian remains notably underrepresented in large-
scale embedding research. In this paper, we
present Hakim, a novel state-of-the-art Persian
text embedding model that achieves a 8.5%
performance improvement over existing ap-
proaches on the FaMTEB benchmark, outper-
forming all previously developed Persian lan-
guage models. As part of this work, we intro-
duce three new datasets—Corpesia, Pairsia-sup,
and Pairsia-unsup—to support supervised and
unsupervised training scenarios. Additionally,
Hakim is designed for applications in chatbots
and retrieval-augmented generation (RAG) sys-
tems, particularly addressing retrieval tasks that
require incorporating message history within
these systems. We also propose a new baseline
model built on the BERT architecture. Our lan-
guage model consistently achieves higher accu-
racy across various Persian NLP tasks, while
the RetroMAE-based model proves particularly
effective for textual information retrieval appli-
cations. Together, these contributions establish
a new foundation for advancing Persian lan-
guage understanding.
1 Introduction
Text embedding models play a pivotal role in mod-
ern Natural Language Processing (NLP) by trans-
forming textual data into numerical representa-
tions that capture semantic and contextual mean-
ing. While languages such as English have bene-
fited from extensive research in embedding method-
ologies, Persian remains significantly underrep-
resented. Existing multilingual models, such as
Multilingual E5 (Wang et al., 2024) and BGE-
M3 (Chen et al., 2024), struggle to capture the
intricacies of Persian grammar and semantics due
to the limited availability of high-quality training
data.In training a high-quality text embedding model,
the choice of a strong foundational language model
plays a crucial role. Several pre-trained language
models have been developed for Persian, including
ParsBERT (Farahani et al., 2020), FaBERT (Ma-
sumi et al., 2024), and TookaBERT (SadraeiJava-
heri et al., 2024). These models utilize the BERT
algorithm to train their language representations.
However, traditional BERT-based models do not ex-
plicitly optimize for embedding generation. Since
text embedding models aim to capture the semantic
meaning of a given text within the CLS token, mod-
els like the RetroMAE (Xiao et al., 2022), which
apply a dedicated loss function on the CLS to-
ken during training, are better suited for this task.
Therefore, there is a clear need to train a new foun-
dational language model specifically optimized for
text embeddings.
One of the most critical aspects of training a
foundational language model is ensuring access
to a clean and diverse dataset. To construct such
a dataset, we crawled a wide range of Persian-
language websites across different domains. We
then individually processed and denoised the text
from each website to curate a high-quality corpus,
which we named Corpesia. This dataset comprises
text from 46 websites across 21 broad topics and
contains over 11 billion tokens. We refer to this
dataset as Corpesia, and utilize it alongside two
other datasets, hmBlogs and Queries, for training
two models: BERT and RetroMAE.
To train a high-quality text embedding model,
having a large collection of semantically paired
texts is essential for understanding the relationship
between two pieces of text. To construct a robust
and comprehensive dataset, we collected a large
corpus of 50 million text pairs. Following a series
of filtering steps, this collection was refined and
reduced to 5 million high-quality pairs, resulting in
the final dataset, referred to as Pairsia-unsup. This
dataset encompasses a wide range of topics and

tasks within the Persian language, ensuring diverse
and representative coverage. By incorporating vari-
ous linguistic structures, contextual meanings, and
semantic relationships, our dataset aims to enhance
the performance and generalizability of the text
embedding model. The inclusion of such a vast
and varied corpus is essential for capturing the in-
tricacies of Persian text and improving the model’s
ability to generate meaningful and contextually rich
embeddings.
In the final phase of model training, we em-
ployed a carefully curated dataset comprising 1.3
million supervised instances. The construction of
this dataset, referred to as Pairsia-sup, was guided
by the objective of incorporating a diverse range of
NLP tasks that are well-suited to text embedding-
based solutions. This dataset served as the founda-
tion for supervised training of the model. Moreover,
task-specific instructions were integrated during
this stage to effectively guide the model across dif-
ferent NLP tasks, leading to a substantial increase
in performance, with accuracy improving by up to
5.71%.
FaMTEB (Zinvandi et al., 2025) is a bench-
mark developed based on MTEB for evaluating
text embedding models in the Persian language.
This benchmark comprises 7 tasks and 63 datasets
aimed at assessing the performance of text embed-
ding models. One of the novel aspects highlighted
in this benchmark is the evaluation of these models
in tasks related to chatbots and retrieval-augmented
generation (RAG).
In this work, we utilize the FaMTEB benchmark
to evaluate our model, Hakim . Additionally, for
training the model on chatbot- and RAG-related
tasks, we leverage the training data provided within
this benchmark in the supervised training stage.
The use of this data enables our model to support
functionalities such as search with follow-up capa-
bility.
We introduce a novel task on which our model is
trained, referred to as cross classification. This task
is inspired by the architecture of Cross-Encoders.
In cross classification, we concatenate a pair of
texts with an instruction that reflects the semantic
relationship between them, drawn from tasks such
as retrieval, classification, semantic textual similar-
ity (STS), or named entity recognition (NER). This
augmented pair is then provided to the model as
input. The label, in turn, specifies the nature of the
relationship—e.g., whether the document answers
the query (retrieval), whether the text belongs toa given class (classification), the similarity score
between two sentences (STS), or the entity type of
a highlighted token in the sentence (NER).
After training, our model achieves an average
accuracy of 88% on this task. One of the potential
applications of this capability is the verification
of LLM outputs across a range of tasks, enabling
more reliable and interpretable responses.
Our main contributions are as follows:
•Introduction of three new datasets— Corpesia ,
Pairsia-sup , and Pairsia-unsup —used in the
pretraining phase and for training the text em-
bedding model.
•Development of a state-of-the-art text embed-
ding model for the Persian language.
•Enabling the model’s integration into RAG
systems and chatbot applications.
•Demonstrating the model’s applicability to the
novel task of cross-classification.
2 Related Work
2.1 General Text Embedding Approaches
Text embeddings have evolved from static rep-
resentations such as Word2Vec (Mikolov et al.,
2013) and FastText (Bojanowski et al., 2017) to
deep learning-based contextualized models like
ELMo (Peters et al., 2018) and transformer-based
architectures such as BERT and its variants. These
advances have greatly improved the ability to cap-
ture rich semantic meanings across different lin-
guistic contexts.
Recent breakthroughs in contrastive learning,
particularly in models like SimCSE (Gao et al.,
2021) and Sentence-BERT (Reimers and Gurevych,
2019), have significantly advanced general-purpose
text representations. However, models like Sim-
CSE were primarily trained on single tasks such as
sentence similarity and are not inherently suitable
for broader applications like classification, question
answering, and other complex downstream tasks.
A new generation of text embedding models has
been developed to address the need for models that
generalize across a wider range of tasks. These
models are designed to produce representations
that are more adaptable and robust for diverse NLP
applications, such as retrieval, classification, and
QA, thereby overcoming the limitations of earlier
single-task-trained embeddings.

Among these models, the Bilingual Generative
Embeddings model(BGE) (Xiao et al., 2024) has
demonstrated significant advancements in text rep-
resentation through a two-step contrastive training
process. In the first step, BGE was trained on a
large-scale dataset comprising 100 million unla-
beled paired samples, which were widely collected
from the web and structured data sources across
various domains. This extensive pretraining phase
allowed the model to develop a broad understand-
ing of linguistic structures and semantic relation-
ships.
In the second step, BGE was fine-tuned using a
dataset containing 838,465 labeled samples, cov-
ering a diverse range of NLP tasks to enhance its
ability to generate high-quality text embeddings.
These tasks were specifically designed to assess
and improve different aspects of text representa-
tion, ensuring the model’s robustness and adaptabil-
ity. As a result of this two-stage training approach,
BGE achieved state-of-the-art performance in En-
glish and Chinese language processing, demonstrat-
ing superior effectiveness in multiple NLP bench-
marks.
Many other text embedding models, such as
GTE and Nomic (Nussbaum et al., 2025), apply a
similar strategy to train general-purpose text em-
bedding models. These approaches leverage large-
scale contrastive training and fine-tuning on di-
verse labeled datasets to enhance their performance
across various languages and tasks.
In addition to these approaches, NV-Embed (Lee
et al., 2025) introduces a high-performance text em-
bedding model based on decoder-only architectures.
Unlike traditional methods that rely on <EOS> to-
ken representations or mean pooling, NV-Embed
employs a latent attention layer to generate more
effective text embeddings.
The model follows a two-step training process:
first, it is pretrained on a large-scale retrieval
dataset to learn query-document relationships; sec-
ond, it is fine-tuned on diverse tasks such as clas-
sification and clustering to enhance generalization.
Additionally, NV-Embed incorporates instruction-
based learning to optimize performance across spe-
cific NLP tasks, demonstrating the potential of
decoder-only models for robust text representation.
2.2 Persian Text Embedding
In the Persian language, the only publicly available
text embedding model to date is Tooka SBERT, for
which detailed training information is not available.Among multilingual models that support Persian,
notable examples include BGE-m3, Jina-embed-
v3 (Sturua et al., 2024), and mE5.
Although these models can generate vector rep-
resentations for Persian text, Persian is not one of
the primary languages used in their training. For
example, Jina-embed-v3 supports 100 languages,
but its primary focus is on 30 languages, excluding
Persian. As a result, these models may not neces-
sarily achieve optimal performance in Persian text
representation.
2.3 FaMTEB
The Massive Text Embedding Benchmark (MTEB)
is a comprehensive suite designed to evaluate text-
embedding models across multiple NLP tasks, in-
cluding classification, clustering, pair classification,
reranking, retrieval, summarization, bitext mining,
and semantic textual similarity (STS). MTEB has
become the most popular benchmark for evaluat-
ing text embeddings. While MTEB primarily fo-
cuses on English, it lacks sufficient support for low-
resource languages like Persian. To address this,
FaMTEB extends MTEB by introducing a large-
scale Persian benchmark with 63 datasets covering
six of the same tasks as MTEB. However, instead
of bitext mining and summarization, FaMTEB eval-
uates models on summary retrieval.
3 Method
3.1 Data
3.1.1 Pre-training Data
Corpesia This dataset comprises textual data from
Iranian websites, categorized into 21 distinct cat-
egories—such as news, economy, and books, as
illustrated in Table 2—and sourced from 46 web-
sites. Each website, after being crawled using a
dedicated crawler, is extracted as a structured data
instance, preserving the hierarchical layout of ti-
tles, headings, and paragraphs in the stored file. We
used Selectolax , a fast HTML parser, to efficiently
parse the HTML content during this process. In
addition, various forms of textual noise—such as
advertisements, source citations, and site-specific
template phrases—are identified and filtered sepa-
rately.
To prepare data for pretraining HakimBERT , we
used a probabilistic strategy to combine text, rather
than treating each paragraph as a separate sample.
Specifically, 90% of webpages were dynamically
merged into longer text blocks until the model to-

Figure 1: This figure presents an overview of all the datasets utilized for training the Hakim network. The Corpesia
corpus is employed for training the base model. The Pairsia-unsup dataset is used during the unsupervised training
phase, while the Pairsia-sup dataset is incorporated in the final stage of training.
ken limit was reached. The remaining 10% fol-
lowed the standard approach, with paragraphs kept
separate. In the merged set, document headings
were retained in 50% of cases to preserve structural
cues. Additionally, 10% of merged samples were
allowed to slightly exceed the model token limit
before being split, helping the model generalize
better to longer sequences while still respecting
practical training limits.
Overall, the dataset amounts to approximately
11 billion tokens, which is significantly smaller
than larger datasets like Targoman (Ziabary, 2024)
(around 41 billion tokens). However, as discussed
in Section 4.1, pretraining our base model on this
dataset led to better downstream results compared
to TookaBERT, which is trained on Targoman and
two other datasets.
hmBlogs The hmBlogs dataset is a large-scale
Persian corpus built from approximately 20 million
blog posts, containing around 6.8 billion tokens.
Queries This dataset consists of 8.5 million
anonymized queries collected from a Persian-
language search engine.
3.1.2 Pairsia-unsup
The dataset comprises diverse types of pairs, includ-
ing title-document, question-answer, FAQ-style
pairs, and more. We construct this dataset throughCategory Web Urls Domains #tokens
isna.ir, khabaronline.ir, tasnimnews.com,
News yjc.ir, hamshahrionline.ir, mehrnews.com 4,748,358,290
tabnak.ir, asriran.com
Technology digiato.com, zoomit.ir, digikala.com/mag, 414,639,633
Science bigbangpage.com 8,786,319
Economy khanesarmaye.com, ecoiran.com, donya-e-eqtesad.com 822,676,697
Books & Literature taaghche.com, fidibo.com, ganjoor.net 162,292,381
Apps farsroid.com, soft98.ir, sarzamindownload.com 17,185,689
Health newmiind.com, doctoreto.com 35,301,238
Sports varzesh3.com 99,979,068
Music upmusics.com, music-fa.com 15,745,744
Movie vipofilm.com 4,533,569
E-commerce emalls.ir, khodro45.com 46,405,081
Travel hamgardi.com, alibaba.ir/mag 34,242,853
Food parsiday.com 6,658,426
Kids voolak.com, radiokodak.com 2,084,614
Religion wiki.ahlolbait.com, hawzah.net, fa.wikishia.net 461,472,962
Blog virgool.io, magerta.ir, motamem.org 669,092,816
Forum ninisite.com 977,392,614
Wiki fa.wikipedia.org 2,132,566,416
Encyclopedia abadis.ir 127,148,347
Entertainment beytoote.com, namnak.com 375,829,131
Law wikihoghoogh.net 3,369,729
SUM - 11,165,761,617
Table 1: The available data in Corpesia has been pre-
sented in detail in this table.
three primary sources: (1) structured text extracted
from websites, (2) machine-translated English cor-
pora, and (3) Persian datasets that already existed.
Due to the inherent noise in the collected data,
we apply the BGE-M3 text embedding model for
data cleaning. Given the varying distributions
across datasets and the fact that the embedding
model yields different similarity scores across do-
mains, we compute a separate similarity threshold
for each subset and filter accordingly. We also
balance the datasets across different task types to

prevent the model from becoming biased toward
any particular task. This process reduces the unsu-
pervised dataset from 50 million pairs to 5 million
processed pairs.
In the following, several of the datasets used in
this study are described.
News-fa This dataset consists of pairs collected
from news websites. The pairs are either formed
by matching the page title with the page sum-
mary (as most news sections include a summary
for each page), or by pairing the page summary
with the first paragraph of the page. The news
websites used to construct this dataset include
ten sources: Asriran, Donya-e-Eqtesad, Ecoiran,
Hamshahri Online,ISNA, Khabar Online, Tabnak,
Tasnim, Varzesh3, and YJC.
Web Page Most websites follow a general struc-
ture as shown below:
{H1,P,H2,P,H2,P, . . .}
Here, H1represents the main title of the page, fol-
lowed by a general explanation or overview P. This
is typically followed by several subsections, each
introduced by H2, which denotes a specific topic,
accompanied by corresponding passages.
We construct pairs from various websites across
different domains by forming pairs such as (H1,P)
and(H2,P). Examples of such websites include
khanesarmaye, javab24, farsroid, and parsiday,
which cover a wide range of topics, including eco-
nomics, cooking, technology, etc.
FAQ This dataset comprises frequently asked
questions (FAQs) and their corresponding answers
collected from various organizations and compa-
nies, including mobile network operators, energy
providers, and others.
Papers We also constructed a collection of ti-
tle–abstract pairs from papers by crawling two aca-
demic websites: SID and Irandoc.
NLI-fa This dataset comprises machine-
translated versions of the MNLI andSNLI datasets
in Persian. Models trained on this dataset, such as
SimCSE (Gao et al., 2021), are capable of produc-
ing high-quality embeddings.
Other translated datasets To train the model,
several valuable English datasets—whose equiv-
alents are scarce in Persian—were machine-
translated into Persian. These datasets include Ad-
versarialQA , which consists of questions, context
passages, and short answers derived from the con-
text; MS MARCO , consisting of query–passagepairs based on real user queries from the Bing
search engine; QNLI , a collection of question-
answer pairs each labeled to indicate whether the
answer correctly addresses the question; ROPES ,
which evaluates whether a conclusion can be in-
ferred from a paragraph; SAMSum , comprising
dialogue-summary pairs; and WikiLingua , col-
lected from WikiHow, containing instructional
texts paired with their summaries.
Other existing datasets In addition to our pro-
prietary data, we also leveraged several publicly
available datasets that have previously been used in
various Persian text embedding tasks. Among these
datasets are FarSICK (Ghasemi and Keyvanrad,
2021), a Persian translation of the English SICK
dataset for the Semantic Textual Similarity (STS)
task; FarsTAIL (Amirkhani et al., 2020), which
contains data for the textual entailment task in Per-
sian; The Persian Web Document Retrieval Cor-
pus(Zinvandi et al., 2024) is a collection consisting
of query-document pairs retrieved from the web;
andSynTran-fa (Farsi et al., 2024), a collection of
short question-answer pairs in Persian. We also
utilized other datasets that are applicable to related
text embedding tasks.
3.1.3 Pairsia-sup
For supervised fine-tuning, we curate a high-quality
and diverse set of clean Persian data across vari-
ous tasks relevant to text embedding, including
retrieval, classification, semantic textual similarity
(STS), question answering (QA), and others. To
the best of our knowledge, this is the first Persian
dataset of such scale and diversity in this domain.
Each example in Pairsia-sup is paired with nine
negative samples. These include: (1) three hard
negatives constructed from positive pairs within
the same dataset, (2) three hard negatives sampled
across all datasets, and (3) three random negatives.
To extract hard negatives, all relevant documents
are indexed using the model trained on Pairsia-
unsup. Then, for a given query, the top K retrieved
documents are discarded, since these negative pairs
may be too similar to the correct answer, and three
negatives are randomly selected from the range
between ranks K and L. The value of L serves as
an upper bound for the selection of candidates. The
parameters K and L are chosen separately for each
dataset to control the difficulty level of the negative
samples.
Some of the datasets included in Pairsia-Sup are:
FaMTEB We utilize the major datasets provided

in the training part of the FaMTEB datasets to ad-
dress various tasks. These include the train split
ofBEIR-fa , synthetic datasets, datasets specifically
developed for chatbot applications, and the Query-
to-Query dataset, which brings semantically similar
queries closer together.
Other existing datasets Some of the datasets
used for supervised training overlap with those
used in the unsupervised setting. However, the
key differences in this stage are the cleanliness of
the utilized data, the presence of sufficient exam-
ples for each task, and the fact that each pair also
received multiple negatives, as explained before.
3.2 Model
3.2.1 Pretraining
To pretrain our base model and train a new tok-
enizer, we leveraged two rich datasets: our propri-
etary collection, Corpesia , and the publicly avail-
able HmBlogs (Khansari and Shamsfard, 2021).
In addition to these, we incorporated user queries
sourced from an Iranian search engine, enhancing
the model’s robustness to conversational language,
misspellings, and incomplete intent expressions.
To train our tokenizer, we first preprocessed all
datasets. The preprocessing pipeline consisted of
the following steps:
•Character Filtering: We removed characters
not commonly used in Persian. Since the use
of Arabic and Latin characters is prevalent in
modern Persian text, we retained both. Addi-
tionally, we preserved emojis and punctuation
marks, as they are widely used across lan-
guages and can carry semantic information.
•Noise Removal: We eliminated URLs and
residual HTML tags from the text to reduce
noise and improve the quality of the training
data.
After preprocessing, we trained a tokenizer using
the WordPiece algorithm with a vocabulary size of
50,000 tokens. For normalization, we applied the
NFKC standard to simplify and unify the text. Fur-
thermore, we normalized Arabic letters that have
different forms in Persian to their standard Persian
equivalents. Numerical digits were also converted
to English numerals to ensure consistency through-
out the corpus.
The data was fed to the model line-by-line for
inputs exceeding 100 tokens. For shorter texts, we
used a grouped-token strategy: each piece of textwas tokenized and concatenated with others until
reaching the model’s maximum token capacity.
To improve the capability of the model in dense
retrieval and representation learning, we adopted a
Masked Autoencoder (MAE) approach inspired
by RetroMAE (Xiao et al., 2022). This method em-
ploys a lightweight decoder following the BERT
encoder, specifically applied to the [CLS] token, to
produce a more semantically enriched representa-
tion.
3.2.2 Stage 1 (Unsupervised)
In this stage, we train the model on a large-scale
corpus of semantically related text pairs, Pairsia-
Sup, to enable it to acquire general-purpose seman-
tic representations.
We pretrain the model using this corpus with
a contrastive learning objective based on the In-
foNCE loss. In this part of training, we bring the
query closer to its positive pair, while using in-
batch negatives for the negative samples. Further
experimental details are provided in Section 4.
3.2.3 Stage 2 (Supervised)
In this stage of training, we fine-tune the model
previously trained in Stage 1 using the Pairsia-sup
dataset along with the corresponding instructions.
Based on the specific task associated with each
data instance, we append a distinct instruction to
both elements of each pair. In general, we incor-
porate instructions into the dataset in three distinct
forms: first, for classification tasks; second, for
cross-classification tasks; and third, for cases where
the two elements of the pair share a semantic rela-
tionship or form a relative pair (Figure 2).
In classification tasks, the first element of the
pair typically contains a piece of text that must be
classified into a category represented by the sec-
ond element. As illustrated in Figure 2, we append
an instruction to each element of the pair accord-
ingly. In cross-classification tasks, the goal is to
assess the relationship between two textual compo-
nents embedded within a single instruction-driven
prompt. For instance, in a question answering (QA)
scenario, we assess whether a given question and
answer—each framed within an instructional tem-
plate—form a valid pair. In what we refer to as
relative pair tasks, the second element of the pair
bears a specific relationship to the first. This rela-
tionship may take the form of a question-answer
pair, semantic relevance, a retrieved document, or
similar associations.

Stage 1 Stage 2
Datasets digikala_mag,
farsroid, farsick,
irandoc-pair,
MSMARCO_fa,
news_fa and etcalpaca_fa,
HotpotQA_fa,
MeDiaPQA,
MIRACL_fa, Syn-
theticPersianQA,
Q2Q and etc
Size 5,371,643 1,302,659
Table 2: Sample of datasets used in stage1 and stage2
of training.
To ensure flexibility, the model is also exposed
to the same data without instructions, enabling it to
generalize to both instruction-based and instruction-
free usage scenarios.
Finally, the model is trained using a contrastive
learning objective similar to the pretraining stage.
Further experimental results and analysis are pro-
vided in Section 4.
3.2.4 Training Details
Pretraining loss: BERT is pretrained using the
Masked Language Modeling (MLM) objective.
Given a sequence of tokens X= [x1, x2, . . . , x n],
a subset of tokens {xi1, xi2, . . . , x ik}is randomly
selected and replaced with a special [MASK] token.
The goal is to predict the original tokens based on
the masked input.
Letˆxijbe the model’s prediction for the masked
token xij. The MLM loss is computed using cross-
entropy over the vocabulary for each masked posi-
tion:
LMLM=−kX
j=1logP(ˆxij=xij|Xmasked )
Here, Xmasked denotes the input sequence with
selected tokens replaced by [MASK] . The model
is trained to maximize the likelihood of the true
tokens given their masked context, enabling it to
capture deep bidirectional contextual representa-
tions.
The total training loss is averaged over all
masked positions and batch elements. This self-
supervised strategy allows BERT to learn rich lan-
guage representations from large unlabeled cor-
pora.
RetroMAE loss: RetroMAE enhances BERT-
style pretraining by introducing a lightweight de-
coder and modifying the loss function to improve
the representation of the [CLS] token for retrievaltasks. The model uses two distinct masking strate-
gies:
1.Encoding Stage : A moderately masked ver-
sion of the input, ˜Xenc, is processed by the encoder
Φencto produce the sentence embedding h˜X:
h˜X←Φenc(˜Xenc)
2. Decoding Stage : A more aggressively
masked version, ˜Xdec, is combined with h˜Xto
form the input for the decoder Φdec, which recon-
structs the original sentence:
H˜Xdec= [h˜X, ex1+p1, . . . , e xn+pn]
Ldec=X
xi∈maskedCE
xi|Φdec(H˜Xdec)
This setup forces the encoder to generate high-
quality, semantically rich sentence embeddings, as
the decoder alone cannot recover the full input with-
out meaningful representations. This strategy not
only increases the difficulty of the reconstruction
task but also ensures that nearly all input tokens
contribute learning signals, significantly boosting
data efficiency.
RetroMAE-v2 (DupMAE): Introduces a duplex
masked auto-encoder framework with two comple-
mentary decoding tasks:
•Reconstructing the original input sentence
based on the [CLS] embedding.
•Predicting the bag-of-words (BoW) features
of the input sentence using the embeddings of
the ordinary tokens. These two tasks are com-
bined to train a unified encoder, enhancing the
semantic representation capability by leverag-
ing both [CLS] and token-level embeddings.
InfoNCE loss: InfoNCE loss encourages similar
(positive) text pairs to have closer representations
while pushing dissimilar (negative) pairs apart. For
each query q, a relevant document d+, and a set
of irrelevant documents D−={d−
1, . . . , d−
n}, the
InfoNCE loss is commonly used. It maximizes the
similarity between qandd+while minimizing it
with respect to D−. The loss is given by:
Lcl=−loges(q,d+)/τ
es(q,d+)/τ+Pn
i=1es(q,d−
i)/τ
where s(·,·)is a similarity function like cosine
andτis a temperature parameter.

4 Experiments
4.1 Analysis
BERT Like Pretraining: We pretrained our BERT-
based model using the AdamW optimizer with
hyperparameters β1= 0.9,β2= 0.999, and
ϵ= 1e−8. The learning rate was set to 5×10−5,
and we employed a linear learning rate scheduler
with 1,000 warm-up steps. Pretraining was con-
ducted for 3 epochs with a batch size of 32 per
device. No weight decay was applied. Mixed-
precision training (fp16) was enabled to improve
computational efficiency. To ensure reproducibility,
all experiments were run with a fixed random seed
of 42.
RetroMAE: In the next stage, we continue pre-
training the initial model on the same plain-text cor-
pus using the RetroMAE-v2 loss for three epochs
with a batch size of 64. Since text embedding mod-
els represent sentences solely through the [CLS]
token, leveraging this model can substantially en-
hance the accuracy of downstream tasks.
Unsupervised: The initial dataset prepared for
this stage consisted of 51,817,807 data pairs. Af-
ter filtering out irrelevant entries and balancing
samples, the dataset was reduced to 5,371,643 in-
stances. A significant portion of the semantic pair
data was collected from the web, primarily in the
form of document-title pairs, which introduces a
bias in the model toward this specific structure. To
mitigate this bias, we perform sampling to reduce
the overrepresentation of such instances. We also
experimented on various subsets of the dataset and
chose the best dataset combination. Our model was
trained for five epochs with a batch size of 16 on
the filtered and rebalanced dataset.
Table 4 presents a comparative evaluation of the
Hakim-unsup model against other baseline models
on the FaMTEB benchmark. As shown, a single
stage of fine-tuning results in a substantial perfor-
mance gain for the Hakim model, which is among
the top 3 best-performing models by far.
Supervised: In the final stage of training, we
utilize a total of 1.3M data pairs (which increases
to approximately 4.5 million pairs after adding in-
structions in various forms) spanning various tasks
included in the FaMTEB benchmark. The model
is fine-tuned for 2 epochs with a batch size of 8,
under two settings: with instruction tuning and
without. Additionally, the instruction-tuned model
is exposed to both instructed and non-instructed
samples, enabling it to generalize across differentusage scenarios.
The instruction-free model outperforms the pre-
vious best-performing model by 2.8%, while the
instruction-tuned variant achieves an even higher
improvement of 8.5%, as shown in Tables 4 and
6. These results highlight the substantial benefit of
incorporating task instructions during fine-tuning,
demonstrating that instruction tuning can signif-
icantly enhance model accuracy across a diverse
range of tasks.
4.2 Ablation Study
4.2.1 Zero Shot
Since the model has been trained on diverse tasks
with various instructions, it possesses the ability to
perform zero-shot on different problems. To assess
the zero-shot capabilities of our model, we evalu-
ated it on a set of benchmark datasets that were not
seen during training. This evaluation aims to exam-
ine the model’s ability to generalize to new tasks
and domains without any additional fine-tuning.
As shown in Table 5, the model achieves strong
classification performance on the MassiveIntent-
Classification and HamshahriClustring datasets, de-
spite having no prior exposure to similar data dur-
ing training.
These results highlight the model’s effective gen-
eralization and support its ability to perform well
in zero-shot scenarios.
4.2.2 Different Instructions
To optimize the model’s performance, we experi-
mented with various instruction formulations dur-
ing training. Each variant reflects a different strat-
egy for instruction construction, and we refer to the
resulting models as Hakim-inst1 through Hakim-
inst5. Among these, Hakim-inst1 is the variant we
designate as our main model, Hakim, throughout
the paper.
•Inst1 uses the full instruction templates in-
troduced in Figure 2 without modifications,
and also the exact instructions used for each
task is in Figure 3, 4,and 5. This configuration
serves as the baseline for our instruction-tuned
model.
•Inst2 follows the same instruction format as
shown in Figure 2, with two key exception:
in classification and cross classification tasks,
the list of class names is included right after
task_prompt in the query instruction.

SA QA NER NLI
Model DeepSentiPers MirasOpinion PQuAD PCoQA ParsTwiner MULTICONER V2 FarsTail ParsiNLU QP
BERT 62.11 81.03 71.47 19.24 57.67 30.53 67.24 72.62
mBERT 67.21 83.71 86.25 47.25 76.95 56.88 82.14 79.43
XLM-RoBERTa 76.35 85.05 87.87 47.36 80.45 53.47 84.53 78.72
ParsBERT 78.03 84.68 86.67 42.18 82.69 60.31 82.38 79.45
FaBERT 78.77 85.33 87.22 42.98 84.55 51.08 83.64 81.75
TookaBERT-Base 78.09 84.83 87.88 45.57 84.54 61.20 83.04 79.79
HakimBERT 82.99 85.04 88.31 45.03 86.11 61.42 86.82 83.53
Table 3: F1-score comparison of BERT-based models on eight Persian natural language understanding tasks
spanning Sentiment Analysis (SA), Question Answering (QA), Named Entity Recognition (NER), and Natural
Language Inference (NLI).
Size (M) Avg. Class. Cluster. PairClass. Rerank. Retriv. STS SumRet.
Tooka-SBERT 353 60.65 59.40 56.45 87.04 58.29 27.86 76.42 59.06
GTE-multilingual-base 305 63.64 56.07 57.28 84.58 69.72 41.22 75.75 60.88
multilingual-e5-base 278 62.93 57.62 56.52 84.04 72.07 41.20 74.45 54.58
multilingual-e5-large 560 64.40 59.86 57.19 84.42 74.34 42.98 75.38 56.61
BGE-m3 567 65.29 58.75 57.73 85.21 74.56 43.38 76.35 61.07
Jina-embeddings-v3 572 64.53 59.93 59.15 83.71 61.26 43.51 78.65 65.50
Hakim-unsup. 124 64.56 60.65 58.89 86.41 67.56 37.71 79.36 61.34
Hakim-small. 38 70.45 80.19 66.31 87.41 67.30 38.05 75.53 78.40
Hakim 124 73.81 84.56 70.46 89.75 69.46 40.43 76.62 85.41
Table 4: Evaluation of various text embedding models on the FaMTEB benchmark.
Model Name MassiveIntentClassification HamshahriClustring
tooka-sbert 63.19 63.28
intfloat-multilingual-e5-large 65.49 67.42
Hakim 72.72 69.17
Table 5: Zero-shot performance on unseen benchmarks,
showing strong generalization across tasks.
•Inst3 provides a more concise version of the
instructions used in Inst1, aiming to reduce
verbosity while preserving task-relevant infor-
mation.
•Inst4 uses the same abbreviated instructions
as Inst3 but differs in its training regime. Un-
like Inst2–Inst3, Inst4 is not exposed to any
examples without instructions. Exposing the
model to both instructed and non-instructed
inputs during training enhances generalization.
Thus, Inst4 intentionally omits this exposure
to study its effect.
•Inst5 also uses the same abbreviated instruc-
tions as Inst1; however, no instruction is added
to the second pair, except in Classification
tasks.
•We also experiment with a No-Inst variant,
which, unlike other models, is trained without
any instruction and is only trained in the Rel-
ative Pair setting. This is because training inthe Cross and Classification settings without
an instruction is not meaningful.
As shown in Table 6, the model trained with Inst1
achieves higher accuracy compared to the other
models. These results also indicate that incorporat-
ing instructions into the second pair, as opposed to
the more common approach in Inst5—where the
instruction is added only to the query—leads to
improved accuracy.
4.2.3 RAG
To evaluate the model’s capability in tasks related
to Retrieval-Augmented Generation (RAG), we
measured the performance of the Hakim model
on a subset of the FaMTEB dataset consisting of
RAG data. As presented in Table 7, fine-tuning
the model on this dataset led to a significant im-
provement in accuracy, indicating its effectiveness
in handling dialogue-based retrieval tasks.
Additionally, Figure 6 illustrates a qualitative ex-
ample from the SynPerChatbotRAGFAQRetrieval
dataset from FaMTEB benchmark. In this case, the
user query requires understanding the preceding
chat history to retrieve a relevant response. We
compare the outputs of the Hakim with Jina and
multilingual-e5. The results show that the Hakim
model successfully retrieves the appropriate an-

Instruction Version Class. Cluster. PairClass. Rerank. Retriv. STS SumRet. Ave.
Inst2 84.66 66.94 84.91 70.52 40.04 74.11 86.15 72.48
Inst3 83.56 67.59 89.30 69.74 39.68 77.28 85.74 73.27
Inst4 84.25 66.87 87.87 69.71 37.74 77.09 85.67 72.74
Inst4 w.o. Inst 74.67 57.70 87.87 69.80 38.27 78.22 78.52 69.29
Inst5 84.60 70.42 89.84 69.32 38.26 78.47 84.82 73.67
No-Inst 63.03 54.51 88.80 70.78 37.72 78.79 83.41 68.10
Inst1 (Hakim) 84.56 70.46 89.75 69.46 40.43 76.62 85.41 73.81
Inst1 w.o. Inst 72.84 56.97 89.76 69.00 35.92 78.62 82.92 69.43
Table 6: Performance comparison of different instruction-tuning variants across multiple tasks. Inst1 (Hakim)
achieves the highest average score, demonstrating the effectiveness of full instruction templates.
Figure 2: The template of adding instructions to differ-
ent data types used in Hakim. In general, data is catego-
rized into three types: classification, cross-classification,
and Relative Pairs types. Instructions and data pairs are
added accordingly.
swer, demonstrating its ability to capture context
and perform effective conversational retrieval.
4.3 Cross Classification
For training on the cross classification task, we set
aside a portion of the dataset for evaluation. Table 8
reports the accuracy of the Hakim model compared
to other baseline models on this dataset. As shown,
the model is capable of handling tasks in a cross for-
mulation, allowing it to address problems that were
previously not directly solvable using traditional
text embedding models. For instance, semantic tex-
tual similarity (STS) can now be performed using
the Hakim model by encoding a pair of sentences
within an instruction, and associating the pair with
varying similarity scores, effectively transformingthe task into a classification or regression problem
within the cross classification framework.
4.4 HakimBERT Analysis
To evaluate our base model, we conducted a se-
ries of experiments. Our first experiment compares
the training data used for HakimBERT with the
Targoman dataset. In this experiment, we train
two BERT models with identical tokenizers and
hyperparameters on our dataset and the Targoman
dataset, respectively. As shown in Table 9, the
model trained on our dataset consistently outper-
forms the one trained on Targoman across most
tasks.
We also assess the effectiveness of our tok-
enizer by training two models on the same dataset:
one using our tokenizer and the other using the
TookaBERT tokenizer. The results, presented
in Table 10, demonstrate that, except for two
tasks—where the models perform comparably, the
model trained with the Hakim tokenizer achieves
higher accuracy in all other tasks.
To further investigate the effectiveness of Retro-
MAE training, we perform a second-stage pretrain-
ing (unsupervised finetuning) on two models: one
using the original HakimBERT and the other using
HakimBERT pretrained with the RetroMAE objec-
tive. As shown in Table 11, we observe accuracy
improvements across most datasets when using the
RetroMAE-enhanced model, indicating the benefit
of incorporating the RetroMAE objective into the
training process.
5 Conclusion
In this work, we introduced Hakim, a state-of-
the-art Persian text embedding model designed
to address the longstanding underrepresentation
of Persian in large-scale NLP research. By con-

Figure 3: The instructions employed for addressing classification tasks.
Dataset Hakim Hakim No-Inst BGE-m3 Jina-embeddings-v3
SynPerChatbotRAGFAQRetrieval 54.70 58.19 32.03 47.45
SynPerChatbotRAGFAQPC 93.39 89.63 64.42 62.05
SynPerChatbotConvSAClassification 89.84 78.84 61.02 71.57
SynPerChatbotTopicsRetrieval 52.15 18.80 19.18 18.75
SynPerChatbotRAGTopicsRetrieval 50.02 28.37 19.91 24.26
Table 7: Evaluation of retrieval-augmented generation performance on FaMTEB subsets. Hakim outperforms
baseline models, demonstrating superior contextual understanding in dialogue-based retrieval tasks.
structing and utilizing three new high-quality
datasets—Corpesia, Pairsia-unsup, and Pairsia-
sup—we established a comprehensive training
pipeline that supports both unsupervised and su-
pervised learning paradigms. Our model not only
surpasses previous approaches on the FaMTEB
benchmark with an 8.5% performance improve-
ment, but also demonstrates robust capabilities in
downstream applications such as chatbots, retrieval-
augmented generation (RAG), and a novel cross-
classification task. The integration of instruction-
based supervision and domain-specific corpus con-struction has proven effective in capturing the lin-
guistic nuances of Persian, paving the way for more
accurate and contextually aware embeddings. Over-
all, our contributions lay a strong foundation for
further advances in Persian NLP, and we anticipate
that our works in this paper will support a wide
range of future research and practical applications.

Figure 4: The instructions employed for addressing cross classification tasks.
Figure 5: The instructions employed for addressing Relative Pair tasks like retrieval, sts, and summarization.

Model
CExappc
ChatbotRagFAQ
FarsiParaphraseDetection
KeywordAndToneKeyword
Farstail
ParsABSA
ParsinluEntail
ParsinluQueryParaph
ParsiTwiNER
STSSyn
Wikiann
SyntheticQAFa
Hakim 99.45 92.58 97.22 98.84 93.79 89.26 57.12 83.58 94.23 80.6 95.74 96.87
Alibaba-NLP-gte-multilingual-base 87.63 51.91 59.16 50.27 33.6 49.34 34.09 52.8 30.52 30.54 66.18 50.62
BAAI-bge-m3 93.34 51.79 51.79 51.57 34.07 54.8 35.69 53.84 31.24 31.42 63.73 52.08
intfloat-multilingual-e5-large 94.5 52.47 67.87 51.11 33.75 48.38 36.08 53.0 29.91 33.31 69.75 51.4
tooka-sbert 86.18 53.4 53.81 50.34 33.64 54.84 35.85 51.09 23.06 30.36 55.57 50.5
sentence-transformers 85.69 52.47 56.0 50.43 33.66 45.94 34.49 53.23 31.55 29.58 60.98 50.48
Table 8: Accuracy comparison of Hakim and baseline models on 12 Persian cross classification tasks, highlighting
the impact of instruction tuning.
SA QA NER NLI
Model DeepSentiPers MirasOpinion PQuAD PCoQA ParsTwiner MULTICONER V2 FarsTail ParsiNLU QP
Targoman 81.96 85.10 87.87 45.29 85.42 59.63 81.14 80.34
Ours 82.17 84.23 88.47 47.83 83.69 60.66 85.40 80.39
Table 9: Comparison of BERT-based models trained on our dataset vs. the Targoman dataset across eight Persian
NLU tasks, showing the impact of training data on performance.
SA QA NER NLI
Model DeepSentiPers MirasOpinion PQuAD PCoQA ParsTwiner MULTICONER V2 FarsTail ParsiNLU QP
TookaBERT 81.80 84.74 88.46 45.26 84.65 60.68 85.23 81.15
Ours 82.99 85.04 88.31 45.03 86.11 61.42 86.82 83.53
Table 10: Performance comparison of models trained with different tokenizers (Tooka vs. Ours) on eight Persian
NLU tasks, showing the effect of tokenizer choice on downstream performance.
Prompt Version Class. Cluster. PairClass. Rerank. Retriv. STS SumRet. Ave.
Hakim-unsup w.o. Retro 58.75 60.12 85.75 65.81 35.56 78.90 50.85 62.25
Hakim-unsup 60.65 58.89 86.41 67.56 37.71 79.36 61.34 64.56
Table 11: Comparison of Hakim-unsup model performance with and without RetroMAE enhancement across seven
faMTEB tasks. Results show consistent gains from incorporating RetroMAE.

(a) Chat
 (b) Answers
Figure 6: Figure (a) illustrates a chat between a user
and a chatbot, where the user asks a follow-up ques-
tion that requires understanding of the prior conversa-
tion. Figure (b) presents the correct answer along with
the responses retrieved by different models through a
question-answering index. The Jina and E5 models re-
turn incorrect question-answer pairs, whereas the Hakim
model successfully retrieves the correct answer.

References
Hossein Amirkhani, Mohammad AzariJafari, Zohreh
Pourjafari, Soroush Faridan-Jahromi, Zeinab
Kouhkan, and Azadeh Amirak. 2020. Farstail: A
persian natural language inference dataset. CoRR ,
abs/2009.08820.
Piotr Bojanowski, Edouard Grave, Armand Joulin, and
Tomas Mikolov. 2017. Enriching word vectors with
subword information. Transactions of the Associa-
tion for Computational Linguistics , 5:135–146.
Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. In Findings of the Asso-
ciation for Computational Linguistics: ACL 2024 ,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Mehrdad Farahani, Mohammad Gharachorloo, Marzieh
Farahani, and Mohammad Manthouri. 2020. Pars-
bert: Transformer-based model for persian language
understanding. Neural Processing Letters , 53:3831 –
3847.
Farhan Farsi, Sadra Sabouri, Kian Kashfipour, Soroush
Gooran, Hossein Sameti, and Ehsaneddin Asgari.
2024. Syntran-fa: Generating comprehensive an-
swers for farsi qa pairs via syntactic transformation.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.
SimCSE: Simple contrastive learning of sentence em-
beddings. In Proceedings of the 2021 Conference
on Empirical Methods in Natural Language Process-
ing, pages 6894–6910, Online and Punta Cana, Do-
minican Republic. Association for Computational
Linguistics.
Zahra Ghasemi and Mohammad Ali Keyvanrad. 2021.
Farsick: A persian semantic textual similarity and
natural language inference dataset. In 2021 11th
International Conference on Computer Engineering
and Knowledge (ICCKE) , pages 194–199.
Hamzeh Motahari Khansari and Mehrnoush Shams-
fard. 2021. Hmblogs: A big general persian corpus.
CoRR , abs/2111.02362.
Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan
Raiman, Mohammad Shoeybi, Bryan Catanzaro, and
Wei Ping. 2025. Nv-embed: Improved techniques
for training llms as generalist embedding models.
Preprint , arXiv:2405.17428.
Mostafa Masumi, Seyed Soroush Majd, Mehrnoush
Shamsfard, and Hamid Beigy. 2024. Fabert:
Pre-training bert on persian blogs. ArXiv ,
abs/2402.06617.
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey
Dean. 2013. Efficient estimation of word representa-
tions in vector space. Preprint , arXiv:1301.3781.Zach Nussbaum, John X. Morris, Brandon Duderstadt,
and Andriy Mulyar. 2025. Nomic embed: Training a
reproducible long context text embedder. Preprint ,
arXiv:2402.01613.
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt
Gardner, Christopher Clark, Kenton Lee, and Luke
Zettlemoyer. 2018. Deep contextualized word repre-
sentations. In Proceedings of the 2018 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, Volume 1 (Long Papers) , pages 2227–2237,
New Orleans, Louisiana. Association for Computa-
tional Linguistics.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. In Proceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP) , pages
3982–3992, Hong Kong, China. Association for Com-
putational Linguistics.
MohammadAli SadraeiJavaheri, Ali Moghaddaszadeh,
Milad Molazadeh, Fariba Naeiji, Farnaz Aghaba-
baloo, Hamideh Rafiee, Zahra Amirmahani, Tohid
Abedini, Fatemeh Zahra Sheikhi, and Amirmoham-
mad Salehoof. 2024. Tookabert: A step forward for
persian nlu. Preprint , arXiv:2407.16382.
Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram,
Michael Günther, Bo Wang, Markus Krimmel, Feng
Wang, Georgios Mastrapas, Andreas Koukounas, An-
dreas Koukounas, Nan Wang, and Han Xiao. 2024.
jina-embeddings-v3: Multilingual embeddings with
task lora. Preprint , arXiv:2409.10173.
Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang,
Rangan Majumder, and Furu Wei. 2024. Multilingual
e5 text embeddings: A technical report. Preprint ,
arXiv:2402.05672.
Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao.
2022. RetroMAE: Pre-training retrieval-oriented lan-
guage models via masked auto-encoder. In Proceed-
ings of the 2022 Conference on Empirical Methods in
Natural Language Processing , pages 538–548, Abu
Dhabi, United Arab Emirates. Association for Com-
putational Linguistics.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval , SIGIR ’24, page 641–649, New
York, NY , USA. Association for Computing Machin-
ery.
S. Mehran M. Ziabary. 2024. Targo-
man/PersianWebScraper: An accurate scrapper
to scrape popular persian websites, mostly in-
tended to be used as a tool to create large corpora
for Persian language. — github.com. https:
//github.com/Targoman/PersianWebScraper .

Erfan Zinvandi, Morteza Alikhani, Zahra Pourbahman,
Reza Kazemi, and Arash Amini. 2024. Persian web
document retrieval corpus. In 2024 12th Iran Work-
shop on Communication and Information Theory
(IWCIT) , pages 1–3.
Erfan Zinvandi, Morteza Alikhani, Mehran Sarmadi,
Zahra Pourbahman, Sepehr Arvin, Reza Kazemi,
and Arash Amini. 2025. Famteb: Massive text em-
bedding benchmark in persian language. Preprint ,
arXiv:2502.11571.