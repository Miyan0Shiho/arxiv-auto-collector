# QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering

**Authors**: An Quang Tang, Xiuzhen Zhang, Minh Ngoc Dinh, Zhuang Li

**Published**: 2025-06-04 14:50:32

**PDF URL**: [http://arxiv.org/pdf/2506.04020v1](http://arxiv.org/pdf/2506.04020v1)

## Abstract
Review-based Product Question Answering (PQA) allows e-commerce platforms to
automatically address customer queries by leveraging insights from user
reviews. However, existing PQA systems generate answers with only a single
perspective, failing to capture the diversity of customer opinions. In this
paper we introduce a novel task Quantitative Query-Focused Summarization
(QQSUM), which aims to summarize diverse customer opinions into representative
Key Points (KPs) and quantify their prevalence to effectively answer user
queries. While Retrieval-Augmented Generation (RAG) shows promise for PQA, its
generated answers still fall short of capturing the full diversity of
viewpoints. To tackle this challenge, our model QQSUM-RAG, which extends RAG,
employs few-shot learning to jointly train a KP-oriented retriever and a KP
summary generator, enabling KP-based summaries that capture diverse and
representative opinions. Experimental results demonstrate that QQSUM-RAG
achieves superior performance compared to state-of-the-art RAG baselines in
both textual quality and quantification accuracy of opinions. Our source code
is available at: https://github.com/antangrocket1312/QQSUMM

## Full Text


<!-- PDF content starts -->

arXiv:2506.04020v1  [cs.CL]  4 Jun 2025QQSUM: A Novel Task and Model of Quantitative Query-Focused
Summarization for Review-based Product Question Answering
An Quang Tang and Xiuzhen Zhang*and Minh Ngoc Dinh and Zhuang Li
RMIT University, Australia
s3695273@rmit.edu.vn ,xiuzhen.zhang@rmit.edu.au
minh.dinh4@rmit.edu.vn ,zhuang.li@rmit.edu.au
Abstract
Review-based Product Question Answering
(PQA) allows e-commerce platforms to auto-
matically address customer queries by lever-
aging insights from user reviews. However,
existing PQA systems generate answers with
only a single perspective, failing to capture
the diversity of customer opinions. In this pa-
per we introduce a novel task Quantitative
Query-Focused Summarization ( QQSUM ),
which aims to summarize diverse customer
opinions into representative Key Points (KPs)
and quantify their prevalence to effectively an-
swer user queries. While Retrieval-Augmented
Generation (RAG) shows promise for PQA,
its generated answers still fall short of captur-
ing the full diversity of viewpoints. To tackle
this challenge, our model QQSUM-RAG ,
which extends RAG, employs few-shot learn-
ing to jointly train a KP-oriented retriever and
a KP summary generator, enabling KP-based
summaries that capture diverse and represen-
tative opinions. Experimental results demon-
strate that QQSUM-RAG achieves superior
performance compared to state-of-the-art RAG
baselines in both textual quality and quan-
tification accuracy of opinions. Our source
code is available at: https://github.com/
antangrocket1312/QQSUMM
1 Introduction
With the rapid expansion of e-commerce, con-
sumers increasingly rely on product reviews to in-
form their purchasing decisions. Automatic review-
based product question answering (PQA) systems
have emerged, leveraging user reviews to provide
immediate responses on e-commerce Q&A plat-
forms (McAuley and Yang, 2016; Gupta et al.,
2019). However, current PQA systems face a
key limitation: they typically generate a single
answer (Gupta et al., 2019), overlooking the fact
that many subjective e-commerce queries require
*Corresponding author.answers that reflect diverse viewpoints. For ex-
ample, when comparing camera lenses (Figure 1),
some shoppers prioritize versatility and afford-
ability, while others focus on image quality and
speed. Recent PQA approaches aim to improve an-
swer quality using retrieval-augmented generation
(RAG). These systems first retrieve reviews rele-
vant to the query and then use them as context for
large language models (LLMs) to generate answers.
Yet, LLMs often struggle to present multifaceted
perspectives (Sorensen et al., 2024), leading to an-
swers that primarily reflect dominant opinions from
the retrieved reviews (Deng et al., 2020, 2023).
Separately, opinion summarization has made
progress through Key Point Analysis (KPA), which
summarizes reviews into concise, representative
statements called key points (KPs) while also quan-
tifying their prevalence (Bar-Haim et al., 2020a,b,
2021; Tang et al., 2024a,b). However, these KPA
methods focus on general summarization rather
than answering specific queries. For tasks like prod-
uct comparison, summarization must incorporate
only query-focused KPs, making general KPA ap-
proaches insufficient for PQA.
While 
comparing 
the 
Nikon 
24-120mm 
F4 
lens 
with 
the 
24-70mm 
F2.8 
lens 
as 
a 
general 
walk-around 
lens:
?
 
11 
comments
 
says 
that 
the 
24-120mm 
F4 
lens
 
has 
a 
longer 
zoom 
range 
and 
is 
more 
affordable
 
than 
the 
24-70mm 
F2.8.
? 
9 
comments
 
prefer 
the 
24-70mm 
F2.8
 
for 
its 
better 
image 
quality 
and 
faster 
aperture
.
I'm 
seriously 
considering 
taking 
this 
lens 
instead 
of 
my 
AF-S 
24-70 
because 
of 
its 
size 
and 
zoom 
range.
Conventional 
Q&A
 
single 
answer 
of 
a 
random 
or 
major 
opinion
Much 
easier 
to 
for 
everyday 
use. 
I 
can 
use 
it 
for 
travel, 
chasing 
the 
kids 
around, 
or 
any 
other 
every 
day 
shooting 
Answer
QQSUMM
bullet-point 
quantitative 
summary 
of 
diverse 
opinions 
The 
24-120 
has 
good 
reach, 
good 
image 
quality, 
not 
heavy, 
not 
that 
expensive
This 
is 
probably 
not 
the 
best 
lens 
to 
use 
for 
portraits 
because 
it's 
just 
not 
fast 
enough 
(f-stop)
Text
Matched 
Comments
Matched 
Comments
Query:
 
How 
does 
this 
Nikon 
24-120mm 
F4 
lens
 
compared 
with 
the 
24-70mm 
F2.8
 
as 
a 
general 
walk 
around 
lense.
Figure 1: Comparison of conventional Q&A and QQ-
SUM . More details of QQSUM output are in Table 12.
In this paper, we introduce a novel task Quanti-
tative Query-Focused Summarization ( QQSUM ),
which generates comprehensive answers contain-
ing diverse KPs along with their quantified relative
importance (Figure 1). Our solution, QQSUM-
RAG , extends the RAG framework by integrating

KP-oriented retrieval and summarization. Specifi-
cally, QQSUM-RAG retrieves query-relevant re-
views, clusters them by distinct opinions, and sum-
marizes representative KPs from each cluster. This
approach provides broader coverage of key insights,
overcoming the single-perspective limitation of
conventional RAG-based systems.
A key challenge in implementing this approach
is scarcity of training data for such a specialized
task. To address this, we develop a co-training strat-
egy that jointly optimizes the retriever and LLM
through shared supervision signals, enhancing the
alignment between retrieved opinion clusters and
generated KPs. This strategy enables robust perfor-
mance of QQSUM-RAG even with limited train-
ing examples. To support few-shot learning, we
carefully curated a dataset of queries with KPs
and their prevalence quantification, through human-
LLM collaboration. Empirical results show that
QQSUM-RAG significantly outperforms RAG
baselines based on in-context learning and quanti-
tative summarization.
Our main contributions are:
•We introduce a novel task QQSUM . Unlike
traditional PQA, QQSUM generates answers
that capture diverse customer opinions with
their prevalence, addressing queries that re-
quire multiple viewpoints.
•We propose QQSUM-RAG , a RAG-based
framework with KP-oriented retrieval and
summarization. The framework is optimized
through a co-training strategy that improves
alignment between retrieved opinion clusters
and generated KPs in few-shot learning set-
ting. Our experiments show that QQSUM-
RAG significantly outperforms baselines with
up to 2.11 times improvement in textual
similarity with ground-truth KPs and up to
67.12% improvement in quantification perfor-
mance over state-of-the-art KPA system for
reviews (Tang et al., 2024b).
2 Related Work
2.1 Review-based PQA
Unlike domain-specific QA tasks such as biomedi-
cal or legal QA focusing on factual answers, review-
based PQA seeks to provide answers of consumers’
subjective opinions about a product. While ex-
tractive PQA approaches retrieve relevant review
snippets as answers (Chen et al., 2019a; Yu et al.,2012), it fails to provide precise responses since
the review might not be specifically written for an-
swering the given question. Recently, inspired by
the advances of seq-2-seq models, abstractive, i.e.,
generation-based, approaches can generate natural-
language answers from reviews (Chen et al., 2019c;
Gao et al., 2019). However, these approaches fre-
quently suffer from hallucinations and factual in-
consistencies, sometimes generating random an-
swers that misrepresent or contradict the prevalent
opinions (Deng et al., 2020, 2023). Existing review-
based PQA framework then cannot capture nor
quantify faithfully the diverse opinions of reviews
in its answer.
2.2 Key Point Analysis
Developed initially to summarize arguments (Bar-
Haim et al., 2020a,b), KPA was later adapted for
summarization of reviews (Bar-Haim et al., 2021;
Tang et al., 2024a,b). While Bar-Haim et al. (2021)
integrates sentiment analysis and collective key
point mining to select and match KPs from broader
domain with comments, Tang et al. (2024a) in-
tegrates aspect-based sentiment analysis (ABSA)
into extracting and matching of KPs to comments
for more unique KPs and precise quantification.
More recent abstractive KPA studies apply abstrac-
tive summarization to paraphrase and generate KPs
from comments (sentences) (Kapadnis et al., 2021;
Li et al., 2023; Tang et al., 2024b). Overall, whether
extractive or abstractive approaches, KPA can only
produce KPs for general and high-level opinions
without catering to specific queries.
2.3 Textual Summarization
Document summarization aims to produce concise
textual summaries capturing the salient informa-
tion in source documents. While extractive review
summarization approaches use surface features to
rank and extract salient opinions for summariza-
tion (Mihalcea and Tarau, 2004; Angelidis and Lap-
ata, 2018; Zhao and Chaturvedi, 2020), abstractive
techniques use sequence-to-sequence models (Chu
and Liu, 2019; Suhara et al., 2020; Bražinskas et al.,
2020b,a; Zhang et al., 2020a) to generate review-
like summaries containing only the most prevalent
opinions. Recently, prompted opinion summariza-
tion leveraging Large Language Models (LLMs)
was applied to generate fluent and concise review
summaries (Bhaskar et al., 2023). However, exist-
ing studies lack focus on presenting and quantify-
ing the diverse opinions in reviews.

3 Quantitative Query-Focused
Summarization
3.1 Task Formulation
Letqdenote a query, i.e., community question,
andRe={rj}|Re|
j=1denotes a set of review com-
ments on a product e,QQSUM aims to retrieve
relevant comments Dto answer qand generate a
KP-based summary Squantifying viewpoints pre-
sented in D. We formulate S={kp1, . . . , kp n}as
a bullet-point summary containing multiple KPs,
where each bullet-point represents a KP1and its
prevalence (Bar-Haim et al., 2021). For instance,
with the bullet-point “ 23 comments praise that the
headphone is very comfortable for long hours ”, the
KP is “ Comfortable for long hours ”, and the preva-
lence count is 23. Each key point kpi, is matched to
a subset of supporting comments Ci={c1, c2, . . .}
(where ci∈ D), with prevalence being measured
as|Ci|.
3.2 The QQSUM-RAG Framework
Figure 2 illustrates the architecture of QQSUM-
RAG .QQSUM-RAG is based on the retrieval-
augmented generation (RAG) paradigm and con-
sists of 2 stages: KP-Oriented Retrieval andKP
Summary Generation . It utilizes a Retriever to
retrieve and cluster query-relevant comments into
groups, and the LLM to generate the final KP sum-
mary based on the comment clusters. Importantly,
the retriever and LLM can be jointly trained with
shared supervision signals to ensure comment clus-
ters retrieved match KPs generated.
The following general loss function describes
every training step of QQSUM-RAG , whose pa-
rameters are updated at the cluster-KP level rather
than the query level:
L= (1−d)·(Lclus+gold_score ) +d· Lgen (1)
where Lclusis the retrieval loss for each comment
cluster, Lgenis the LLM’s generation loss com-
puted for the KP generated from the respective
cluster, and dis a damping factor to balance be-
tween the two. Notably, gold_score represents the
Perplexity Distillation loss (Izacard et al., 2023),
which transforms the supervisory signals of the
LLM to improve the Retriever. The intuition is that
within a cluster, comments that better contribute
to helping the LLM generate the KP with lower
perplexity should be ranked higher.
1unique and non-overlapping opinion at high level3.2.1 KP-Oriented Retrieval
Given a query q, the Retriever should retrieve rele-
vant review comments Rqthat emphasize opinions
focused on q. We utilize a shared encoder Ethat
can encode both the input query qand each review
comment rj∈Re. Comments are ranked by the
similarity score s(x,rj) = Ec(x)⊤Ed(rj)that is
calculated by taking the dot product of the embed-
dings of the query xand the comment rj. Only
comments with s(x,rj)≥1is selected for Rq.2
Different from standard RAG where generation
is based on the direct retrieval result, to ensure di-
verse and representative opinions for generation,
we enhance the Retriever with the clustering ob-
jective to produce distinctive comment groups that
conceptually match KPs for generation.
KP-Oriented Retrieval Loss Starting with an
empty list of clusters C, and iterate through every
comment in Rq, for every comment, we further
iterate through every existing cluster ci∈Cand
calculate its average cosine similarity score to all
comments of the cluster. Finally, we add the com-
ment to any clusters with average cosine similarity
score above a threshold ( λ= 1.2),3otherwise, a
new cluster is created. Importantly, a comment can
be mapped to multiple clusters. We empirically
showed that our proposed clustering algorithm is
more effective than HDBSCAN (McInnes et al.,
2017) and K-Means through an ablation experi-
ment in Appendix K.
To train the retriever for KP-oriented retrieval,
we align predicted comment clusters Cwith an-
notated clusters P, where Pgroups comments
matched to the same KP (annotation details in
§3.3). The centroid embedding of a cluster is the
mean embedding of its comments rk:¯Ec(ci) =
1
MPM
k=1E(rk). Because a cluster ci∈Cmay
contain mixed opinions represented by multiple
clusters from P, we map each cito the mean
embedding of Pmatch ⊂P:¯Ec(Pmatch ) =
1
MPM
j=1¯Ec(pj), where the semantic similarity
between ciand every pjissim(ci,pj) =
¯Ec(ci)⊤¯Ec(pj)≥threshold . The training objec-
tive minimizes the mean-squared-error (MSE) loss
between each comment rkinciand the average
center of the most similar clusters Pmatch .
Lclus=1
|ci||ci|X
k=1||¯Ec(Pmatch )−E(rk)||2
2.(2)
2the similarity threshold 1 is set empirically
3set empirically based on cluster quality

Comment
cluster 
1
cluster 
2
cluster 
3
KP
+
gold_score
Review
Comments
KP-oriented 
Retrieval
Query
Encoder
Comment-KP 
Matching 
Annotations
update
relevance
ranking
AmazonKP 
Dataset
Adapter
+
Open-source 
LLM
update
Query
Quantification 
Rules
+
Generate 
a 
KP 
summary 
capturing 
opinions 
to 
answer 
the 
query 
Comment
Cluster
Comment
Cluster
Comment
Cluster
+
+
+
Input
KP 
Summary 
Generation
+
+ 
Y 
comments 
say 
that 
[Key 
Point 
2]
+ 
X 
comments 
say 
that 
[Key 
Point 
1]
While 
answer 
[Query]:
Context
Bullet-point 
KP 
Summary
While 
answer 
[Query]:
+ 
X 
comments 
say 
that 
[Key 
Point 
1]
+ 
Z 
comments 
say 
that 
[Key 
Point 
3]
+ 
Y 
comments 
say 
that 
[Key 
Point 
2]
+ 
X 
comments 
say 
that 
[Key 
Point 
1]
While 
answer 
[Query]:
ContextFigure 2: The training architecture of the QQSUM-RAG framework.
3.2.2 KP Summary Generation
A key limitation of previous KPA studies is that
KPs may contain redundant opinions, due to that re-
view comments, possibly containing multiple opin-
ions, are mapped to individual KPs locally (Bar-
Haim et al., 2021; Tang et al., 2024b). To address
this limitation, we propose to generate KPs at the
global level, where the goal is to generate an overall
KP-based summary without redundancy. Our main
idea is that generated KPs are used as the context
for the LLM to better reason and generate the next
KP, which should be a unique, non-overlapping
opinion statement.
Prompting Strategies Following OpenAI’s
prompt engineering guidelines4, we format query-
relevant comment clusters from the Retriever
into a structured prompt with four parts (detailed
in Listing 3, Appendix F): 1)Context and input
structure, 2)Task definition and output require-
ments, 3)Summarization steps for identifying
representative KPs per cluster and generating the
final KP-based summary, and 4)Commonsense
quantification rules to prioritize clusters by size
and prevent overlapping KPs. To minimize
ambiguity and hallucination, we encode predicted
clusters Cas JSON objects and assign each a
unique ID, requiring the LLM to label generated
KPs accordingly.
Next-KP-Generation Training During training,
generating multiple KPs in a summary lacks align-
ment with Lclus, which is computed per comment
cluster. To address this, we introduce a Next-
KP-Generation objective, inspired by Next-Token
Prediction in LMs (Brown et al., 2020), to en-
hance the generation of salient, non-overlapping
4https://platform.openai.com/docs/guides/
prompt-engineeringKPs. This approach fine-tunes the LLM to itera-
tively generate KPs within the summary. Specif-
cally, let the final KP-based summary S=
{kp1, . . . , kp i, . . . , kp n}, each kpiis generated
with preceding KPs {kp1, . . . , kp i−1}as the con-
text, prompting the LLM to iteratively complete
S. The generation loss for each kpiofci∈C
is computed as the negative log-likelihood (NLL)
against the reference KP, annotated for the most
similar pi∈Pidentified during retrieval,
Lgen=−1
TTX
t=1logP(xt|x<t) (3)
where P(xt|x<t)represents the probability as-
signed by the model to the correct token xt, given
the preceding tokens x<t.
3.3 Human-LLM Key Point Annotation
From Section 3.2, to train our QQSUM-RAG
framework in the few-shot setting, annotation of
KPs for queries and relevant comments are nece-
sary. Prior KPA studies only include annotations
matching comments to KPs without queries (Bar-
Haim et al., 2020a,b). No datasets exist for match-
ing comments to KPs in PQA.
Query
Community 
Answer 
(A)
A1
A3
Stage 
1: 
Extracting 
unified 
KPs 
from 
Community 
Answers
Does 
the 
KP 
match 
the 
review 
comment?
KP
Comment
Stage 
2: 
Human-LLM 
collaborative 
comment-KP 
Matching
MTurk 
Workers
Very 
Well
Somewhat
Not 
Well
Not 
at 
all
Somewhat 
Well
No
Yes
KP
Comment
prevalence 
= 
10
prevalence 
= 
5
While 
answering 
[Query]
+ 
10 
comments 
say 
that 
+ 
5 
comments 
say 
that 
Stage 
3: 
Bullet-like 
Summary 
Generation
Comment-KP 
Matching
KP1/KP2/KP3
Figure 3: Illustration of the human-LLM collaborative
annotation pipeline for A MAZON KP.

Statistic Train Test
# Product Categories 17 17
# Instances (queries) Per Category 2 148
Total Instances 34 2516
# Reviews Per Query 71.18 72.70
# Review Comments Per Query 452.03 431.62
# Answers Per Query 7.53 6.45
# KPs Per Query (Stage 1) 9.26 6.90
# Relevant Comments Per Query (Stage 2) 24.50 –
# Comments (Prevalence) per KP (Stage 2) 6.37 –
Summary Length (Stage 3) 101.29 –
Table 1: Core statistics of the A MAZON KP dataset.
We leverage the popular PQA dataset Ama-
zonQ&A (Gupta et al., 2019) for our QQSUM
task, focusing on only answerable ,subjective (non-
factual) questions that have multiple answers. Out
of 17 product categories (e.g., Electronics, Video
Games), we only include businesses with 50-100
reviews, and sampling top 150 questions per cate-
gory based on answer count. For ease of reference
we name this curated dataset AMAZON KP. Details
on question classification for AMAZON KPare in
Appendix A, and their taxonomy in Appendix B.
Notably, the dominance of “ Scenario-based ” ques-
tions underscore the importance of QQSUM for
generating KP summary to answer user questions
on preferences and scenarios.
Manually summarizing and quantifying opinions
from comments is laborious and time-consuming,
if not impossible. Research shows LLM’s strong
annotation capabilities (He et al., 2024), and so
we design a three-stage human-LLM collaborative
annotation pipeline, shown in Figure 3.
Stage 1: KP Extraction from Gold Community
Answers Given a query qi, the AmazonQ&A
dataset provides multiple answers, i.e. responses,
from online users Ai={a1, a2, .}, serving as
ideal approximation of gold opinions. However,
these responses can contain overlapping opinions.
We therefore zero-shot prompted GPT-4-o-mini to
extract distinctive and non-overlapping KPs from
Ai. Empirical validation with human annotators
confirms that the extracted KPs are of high quality,
with 90% of community answers were represented
by KPs, while 87.5% of the extracted KPs are ver-
ified as valid (precision). Further details are in
Appendices C and D.
Stage 2: LLM-based and Manual Comment-KP
Matching Based on the annotation process in the
literature (Bar-Haim et al., 2020a), we further inte-
grate LLMs to reduce human effort and time. Us-
ing KPs extracted from gold answers (Stage 1), we
prompt GPT4-o-mini to annotate pairwise matchesbetween comments and KPs from all available
reviews of the product. LLM-matched pairs are
then validated by three Amazon Mechanical Turk
(MTurk) workers. Finally, comments from vali-
dated pairs are grouped by similar KPs, with KP
prevalence determined by the number of match-
ing comments. Further details on KP Matching
annotations are provided in Apppendix E.
Stage 3: KP-based Summary We utilize KPs
and their prevalence counts, discovered for every
query, to manually compose a bullet-point KP-
based summary, where each bullet point corre-
sponds to a KP and is annotated as “ |kpi|comments
say that kpi”.
The number of pairwise comment-KP matching
annotations required per query can be up to 2K-
3.5K. For training , to control annotation costs,
we conducted Stages 1, 2 and 3 annotations on a
small subset of 34 instances for few-shot training
ofQQSUM-RAG , randomly selecting two queries
per product category for supervised labeling. For
evaluating the KP-based summary , the remain-
ing examples with only Stage 1 annotations serve
as the test set. The core statistics of AMAZON KP
are shown in Table 1.
4 Experiments
We employ Atlas (Izacard et al., 2023), a pre-
trained efficient RAG model, as our backbone
model for QQSUM-RAG . We utilized Con-
triever (Izacard et al., 2022) as the retriever while
replacing the original language model with open-
source LLMs (e.g., Vicuna-7B5,Mistral-7B6)
for generation. For computational feasiblity, we
apply Low-Rank Adaptation (LoRA) (Hu et al.,
2021), which adds trainable parameters while freez-
ing the model’s original weights.
4.1 Baselines
We benchmark QQSUM-RAG against 3 RAG
baselines.
(Retriever + LLM) co-train We few-shot trained
Atlas (Izacard et al., 2023), with the standard
RAG architecture and Retriever-LLM generator
co-training, for the QQSUM task. The retriever
retrieves relevant comments, while letting the LLM
implicitly infer KPs’ matching comments and their
5https://huggingface.co/lmsys/vicuna-7b-v1.5
6https://huggingface.co/mistralai/
Mistral-7B-Instruct-v0.2

quantities during KP summary generation. For
training, we aggegrated matching comments across
KPs, per query, as the retrieval ground truth.
Frozen Retriever + Prompt LLM To assess
in-context learning (ICL) for QQSUM , we use a
frozen retriever and Vicuna-7B ,Mistral-7B , and
GPT-4-Turbo as the LLM for ICL. Few-shot train-
ing instances are concatenated with test instances,
with the number of few-shot examples optimized
for context length and cost: 4-shot for Mistral-7B
andGPT-4-Turbo , and 2-shot for Vicuna-7B .
Frozen Retriever + KPA We replace the LLM
of a standard RAG with existing KPA review sum-
marization systems to adapt KPA to the QQSUM
task. Comments were first retrieved by a frozen
retriever and then RKPA-Base (Bar-Haim et al.,
2021) utilizes a quality ranking model (Gretz et al.,
2020) to extract KP candidates before matching
comments to KPs using a KP Matching model (Bar-
Haim et al., 2020b) at threshold tmatch = 0.99.
PAKPA (Tang et al., 2024b) clusters comments
by aspect and sentiment before generating aspect-
oriented KPs.
All experiments were conducted at the KP
level, focusing on KPs in the summary outputs
ofQQSUM-RAG and baselines for fair compar-
ison. We post-process the output KP-based sum-
mary into KPs as JSON objects, where each object
covers the KP information of a bullet point in the
summary.7The baselines were implemented us-
ing either the PyTorch module or the Huggingface
transformers framework, and were trained on a
NVIDIA GeForce RTX 4090 GPU.
4.2 Evaluation Dimensions
We conducted experiments on the test set of AMA-
ZONKP(§3.3), consisting of questions from 17
product categories. For reasonable cost, we sample
8 questions from each category for evaluation.
4.2.1 KP Textual Quality
Automatic Evaluation Extracted KPs from gold
community answers in AmazonKP (Stage 1 of
§ 3.3) serves as the reference KPs for this auto-
matic evaluation. We first perform a lexical com-
parison between KPs in the generated summary
7We use a simple LLM-based post processor, prompting
gpt-4-o-mini with ’Format all key points and their
prevalences mentioned in the above bullet-point
summary in a JSON list, where each JSON object
format as: {’key_point’: <key point of a bullet>,
’prevalence’: <key point’s prevalence>}’and the ground truth by computing the highest
Rouge (Lin, 2004) score between generated and
reference KPs for each query and then average the
maxima. Then, following Li et al. (2023), we cal-
culate soft-Precision/Recall/F1 (denoted by sP, sR
and sF1, respectively), which measure the seman-
ticsimilarity between individual generated KP and
the reference KP. While sPfinds the reference KP
with the highest similarity score for each generated
KP,sRis vice-versa, and ( sF1) is the harmonic
mean between sPandsR.
sP=1
n×X
α
i∈Amax
β
j∈Bf(αi, βj) (4)
sR=1
m×X
β
i∈Bmax
α
j∈Af(αi, βj) (5)
Additionally, inspired by sP/sR/sF1 of Li et al.
(2023), we further propose RDto identify KP re-
dundancy . For each generated KP in the summary
for a query, RD finds the neighborhood KP with
the highest similarity score.
RD=1
n×X
α
i∈Amax
θ
j̸=α
i∈Af(αi, θj) (6)
where fcomputes similarities between two in-
dividual key points, A,Bis the set of generated
and reference KPs and n=|A|andm=|B|,
respectively. We use state-of-the-art semantic sim-
ilarity metrics BLEURT (Sellam et al., 2020) and
BERTScore (Zhang et al., 2020b), along with LLM-
based metric G-E VAL-4 (Liu et al., 2023) as fmax.
Note that G-E VAL scores are scaled from 1-5 to 0-
1 for comparability and its evaluation prompt was
also customized to fit our evaluation (Appendix G).
Human Evaluation We manually evaluated the
information quality of generated KPs in the sum-
mary considering 7 different dimensions utilized in
previous KPA studies (Kapadnis et al., 2021; Tang
et al., 2024b), including REDUNDANCY ,COVER -
AGE,FAITHFULNESS VALIDITY ,SENTIMENT ,IN-
FORMATIVENESS andSINGLE ASPECT . Details
of these dimensions are in Appendix H.
We conducted pairwise comparisons of KPs
from different systems using Amazon Mechanical
Turk (MTurk). Given a dimension for evaluation,
each comparison involved choosing the better one
from two summaries, each taken from a different
system. Using the Bradley-Terry model Friedman
et al. (2021), we calculated rankings from these
comparisons among the models. For an example of

P@5 P@10 P@20 P@all
QQSUM-RAG (Ours)
Contriever (Izacard et al., 2022)
+ Mistral 0.668 0.633 0.601 0.535
+ Vicuna 0.567 0.527 0.526 0.367
all-MiniLM-L12-v2 (Wang et al., 2021)
+ Mistral 0.590 0.538 0.500 0.440
+ Vicuna 0.569 0.507 0.468 0.362
(Retriever +LLM )co-train (Izacard et al., 2023)
Contriever (Izacard et al., 2022)
+ Mistral 0.544 0.511 0.459 0.345
+ Vicuna 0.444 0.467 0.442 0.328
all-MiniLM-L12-v2 (Wang et al., 2021)
+ Mistral 0.531 0.530 0.515 0.350
+ Vicuna 0.552 0.512 0.454 0.339
frozen Retriever +prompt LLM
Contriever 0.494 0.447 0.404 0.325
all-MiniLM-L12-v2 0.479 0.446 0.452 0.315
BM25 0.469 0.432 0.387 0.283
Table 2: Performance of retrieval models.
an annotation, see Appendix I. Note that for reason-
able cost, we sample and select only the popular
question (with the highest average KP prevalence),
each from 5 common categories8ofAMAZON KP.
4.2.2 KP Quantification Performance
We evaluate the KP quantification performance of
different systems for KP-comment matching and
factual alignment.
KP-comment matching We first assess the ac-
curacy of the KP comment matching by extend-
ing Bar-Haim et al. (2021) to measure both preci-
sion (correctness of predicted matches) and recall
(coverage of ground-truth matches). For each sys-
tem, we compute precision and recall by prompt-
inggpt-4-o-mini to annotate pairwise match /non-
match between generated KPs and retrieved com-
ments Rq. Additionally, leveraging annotated
comment-KP pairs, we introduce QuantErr , which
measures the mean absolute error between pre-
dicted and actual KP prevalence count. Empiri-
cal validation shows gpt-4-o-mini annotations
highly correlated with MTurk workers’ judgement
(Pearson’s r= 0.647) (Appendix J).
KP-comment factual alignment: We further
employed AlignScore (Zha et al., 2023) for au-
tomatic evaluation of factual alignment between
generated KPs and their corresponding comments.
4.3 Results
4.3.1 The Retrieval Model
The retriever is important for retrieving comments
relevant to queries and so we first evaluated the per-
formance of different backbone retrieval models.
8namely Home & Kitchen ,Sports & Outdoors ,Tools &
Home Improvement ,Health & Personal Care , and BeautyFor this we prompted gpt-4-o-mini to annotate
the relevance of retrieved comments to queries. Ta-
ble 2 reports the retrieval Precision@k (P@k), mea-
sured at different levels of top-k-ranked retrieved
comments ( [5,10,20, all]), using 3 different re-
trieval models: Contriever, all-MiniLM-L12-v2
and BM25. Note that BM25 is not a neural en-
coder and therefore can only be evaluated in the
frozen Retriever + Prompt LLM setting.
Overall, the trained retriever of QQSUM-RAG ,
as being co-trained with the LLM and extended
for KP-oriented retrieval, outperform all baselines.
Notably, co-training with stronger LM can also
contribute up to 45.78% improvement, as the su-
pervision signal from more query-focused KP gen-
eration helps train the Retriever to rank documents
more accurately. Contriever stood out as the best
performer regardless of the LM selection. Hereafter
we base all upcoming experiments with Contriever
as the retrieval model.
4.3.2 KP Quality
KPs produced by different systems in terms of tex-
tual quality, semantic quality and redundancy are
reported in Table 3. Scores of all systems are low
in general, as opinions in product reviews may
not cover all opinions from community answers
to questions. From Table 3, QQSUM-RAG out-
performs other systems in all quality dimensions. It
shows 2.11 times improvement in textual similarity
with reference KPs (0.256 vs. 0.121 in ROUGE-1),
0.23 point absolute improvement in semantic simi-
larity (0.39 vs. 0.16 in BERTScore) and 0.14 point
absolute reduction in Redundancy (0.37 vs. 0.51
using BERTScore for semantic similarity).
The high quality of KPs in QQSUM-RAG
can be attributed to the KP-oriented retrieval of
QQSUM-RAG . Notably, although (Retriever +
LLM) co-train shares the same backbone model and
co-training design with QQSUM-RAG , the lack
of (1) opinion-level clustering of retrieved com-
ments and (2) limited modeling capability of LLMs
makes this model unable to produce KPs as diverse,
unique and representative as QQSUM-RAG . The
weak reasoning capability of LLMs for diverse
opinion summarization is further exposed in the
frozen Retriever + prompt LLMs setting, where
LLMs even with strong modelling capability like
GPT-4-Turbo struggle to elaborate diverse and dis-
tinctive KPs from hundreds of comments.
It is worth noting that Mistral-7B broadly ex-
hibits higher performance than Vicuna-7B across

ROUGE BERTScore BLEURT G-Eval-4
R-1 R-2 R-L sP sR sF1 RD ↓ sP sR sF1 RD ↓ sP sR sF1 RD ↓
QQSUM −RAG (Ours)
+ Mistral 0.256 0.061 0.220 0.39 0.29 0.33 0.37 0.51 0.41 0.46 0.49 0.88 0.82 0.85 0.36
+ Vicuna 0.222 0.078 0.204 0.38 0.26 0.31 0.53 0.49 0.39 0.44 0.54 0.87 0.81 0.84 0.36
(Retriever +LLM )co−train (Izacard et al., 2023)
+ Mistral 0.209 0.057 0.194 0.37 0.28 0.32 0.43 0.49 0.40 0.44 0.55 0.81 0.82 0.81 0.41
+ Vicuna 0.174 0.041 0.161 0.37 0.26 0.31 0.48 0.48 0.38 0.42 0.58 0.78 0.78 0.78 0.41
Frozen Retriever +prompt LLM
+ Mistral 0.210 0.055 0.191 0.33 0.26 0.29 0.51 0.46 0.38 0.42 0.55 0.79 0.80 0.79 0.41
+ Vicuna 0.164 0.059 0.154 0.22 0.20 0.21 0.48 0.46 0.31 0.37 0.59 0.73 0.73 0.73 0.41
+ GPT-4-Turbo 0.197 0.048 0.174 0.32 0.25 0.28 0.44 0.45 0.38 0.41 0.54 0.77 0.77 0.77 0,38
Frozen Retriever +KPA
+ PAKPA (Tang et al., 2024b) 0.179 0.027 0.162 0.34 0.28 0.31 0.46 0.47 0.41 0.44 0.54 0.79 0.80 0.80 0.36
+ RKPA-Base (Bar-Haim et al., 2021) 0.121 0.016 0.106 0.16 0.14 0.14 0.50 0.43 0.36 0.39 0.61 0.69 0.70 0.69 0.51
Table 3: KP summary textual quality. sP, sR and sF1 refer to Soft-Precision, Soft-Recall, and Soft-F1 respectively
based on set-level evaluation method against reference KPs in gold answer.
CV FF RD VL SN IN SA
QQSUM-RAG (Ours) 28.44 26.56 25.34 35.23 31.11 25.9 24.8
(Retriever + LLM) co-train (Izacard et al., 2023) 11.06 11.17 14.7 9.99 9.54 13.49 17.52
Frozen Retriever + prompt LLM (GPT-4-Turbo) 15.12 12.84 15.73 10.36 14.6 12.59 10.79
Frozen Retriever + PAKPA (Tang et al., 2024b) 9.94 12.41 13.28 7.7 8.87 13.04 9.34
Frozen Retriever + RKPA-Base (Bar-Haim et al., 2021) 16.20 22.28 15.73 22.91 20.75 21.02 18.77
Table 4: Human evaluation of KP information quality by different dimensions. Reported are the Bradley Terry scores
of 7 dimensions, from left to right, COVERAGE ,FAITHFULNESS andREDUNDANCY ,VALIDITY ,SENTIMENT ,
INFORMATIVENESS ,SINGLE ASPECT . For reasonable cost, we only conducted manual evaluation on Mistral - the
best LM configuration of QQSUM-RAG and (Retriever + LLM) co-train , selected from Table 3.
all systems based on LLM generation and in all KP
quality measurement (up to 15.32%), largely due
to its stronger modeling capability.
Frozen Retreiver + KPA baselines, despite their
high performance for review summarization, is in-
effective for QQSUM . Not surprisingly PAKPA,
which generates KPs based on aspect-sentiment,
broadly shows better performance than RKPA-
Base, an extractive KPA system. It is possible
that multiple query-relevant opinions on the same
aspect are expected to answer a user query, thus
leading to the weak performance of PAKPA.
Our manual evaluation of KP information qual-
ity further validates the above findings, as shown
by the Bradley Terry scores in Table 4. Overall,
QQSUM-RAG achieves up to 4.58 times improve-
ments on all 7 dimensions, and are notably higher
onCOVERAGE (CV) (2.86 times), VALIDITY (VL)
(2.38 times), and S ENTIMENT (SN) (3.5 times).
4.3.3 KP Quantification
Table 5 presents the quantification performance
for different systems. F 1, combining Recall and
Precision, measures the overall performance of
KP-comment matching for all systems. QuantErr
(lower the better) directly measures KP quantifi-
cation errors. Overall, QQSUM-RAG shows the
best performance in terms of both F 1(0.792 vs.0.154) and QuantErr (4.24 vs. 30.13).
Comparing QQSUM-RAG against the Re-
triever+LLM generation systems, namely (Re-
triever + LLM) co-train and Frozen Retriever +
prompt LLM, we can see that, without clustering
comments, LLMs perform comment-KP matching
and KP quantification, showing extremely low Re-
call (0.185–0.249), in contrast to the high Recall
ofQQSUM-RAG (0.684-0.869). This can be at-
tributed to two main factors: (1) LLMs tend to
hallucinate when generating KPs from a large set
of retrieved comments, and (2) their limited context
window restricts their ability to effectively match
comments to KPs.
Comparing QQSUM-RAG against Retriever +
KPA systems, our model shows up to 67.12% im-
provement in quantification performance over state-
of-the-art KPA system for reviews (PAKPA) (Tang
et al., 2024b), with a 36.53% reduction in Quan-
tErr. Note that Frozen Retriever + PAKPA achieves
the highest matching precision due to aspect-level
opinion quantification. However, it has low recall,
possibly because it relies on aspect-based sentiment
analysis of comments, which can fail to identify
implicit opinions not explicitly including aspects.
As shown in Table 5, results for KP-Comment
Factual Alignment show that QQSUM-RAG and

KP-Comment Matching KP-Comment Factual Alignment
P R F1 QuantErr ↓ AlignScore
QQSUM-RAG (Ours)
+ Mistral 0.694 0.869 0.792 04.24 0.749
+ Vicuna 0.538 0.684 0.602 07.83 0.630
(Retriever +LLM )co−train (Izacard et al., 2023)
+ Mistral 0.567 0.249 0.346 18.10 0.653
+ Vicuna 0.442 0.094 0.154 30.13 0.394
Frozen Retriever +prompt LLM
+ GPT-4-Turbo 0.746 0.200 0.313 16.63 0.673
+ Mistral 0.498 0.214 0.300 19.14 0.624
+ Vicuna 0.439 0.185 0.260 21.52 0.531
Frozen Retriever +KPA
+ PAKPA (Tang et al., 2024b) 0.762 0.520 0.619 06.68 0.749
+ RKPA-Base (Bar-Haim et al., 2021) 0.371 0.314 0.340 15.62 0.354
Table 5: Performance for KP-Comment matching and factual alignment
Frozen Retriever + KPA (PAKPA) achieve high fac-
tual correctness in KP generation, outperforming
other systems (0.749 vs. 0.354). This result high-
lights that QQSUM-RAG generates KPs grounded
in the retrieved comments, and similarly PAKPA
generates KPs grounded in aspects.
4.4 Ablation Study
We evaluate the contribution of Next-KP-
Generation in QQSUM-RAG , with results in
Tables 10 and 11 (Appendix L). In particular, we
configure a variant QQSUM-RAG Single-KP that
replaces Next-KP-Generation with KP generation
for each comment cluster. Not including previously
generated KPs as context, QQSUM-RAG Single-KP
struggles to capture the truly representative
opinion of the cluster, likely generating KPs with
overlapping opinions, especially for comments
containing multiple opinions. Note that while
its KP quality underperforms RAG baselines, its
KP Quantification performance remain superior,
largely attributed to KP-oriented Retrieval.
4.5 Case studies
We conducted case studies to evaluate the redun-
dancy and specificity of generated KPs for a query
comparing camera lenses, presented in Table 14
(Appendix M). Overall, QQSUM-RAG stands
out for generating KPs with minimal redundancy,
high informativeness, and alignment with the query.
First, QQSUM-RAG reduces redundancy by effec-
tively capturing distinct product features relevant to
the user’s needs (e.g., faster aperture), whereas (Re-
triever + LLM) co-train , GPT-4-Turbo Prompt LLM,
and PAKPA tend to generate repetitive and generic
statements, such as “The 24-70mm f/2.8 is a bet-
ter lens overall.” Furthermore, QQSUM-RAG
expands feature coverage, capturing details such
as Vibration Reduction (VR) technology, whichseveral baselines fail to mention.
4.6 Error Analysis
Our analysis on a KP summary of QQSUM-RAG
reports two systematic error patterns, as shown in
Table 13. First, a KP can be falsely matched to
comments expressing similar opinions but on dif-
ferent targets. For instance, the comment “ For a
lens that is overall a rather mixed bag . . . it is very
expensive. ” was matched to KP “ The 24-120mm
F4 lens has a longer zoom range and is more afford-
able than the 24-70mm F2.8. ”. Since the comment
lacks an explicit product reference, it remains un-
clear whether it critiques the 24-120mm F4 or the
24-70mm F2.8 . The second type of errors stems
from the sentence-level quantification, where input
review sentences often contain co-occurring multi-
aspect opinions, making it difficult for the Retriever
to isolate distinct aspects into separate clusters.
5 Conclusion
In this paper, we studied a new task Quantitative
Query-focused Summarization, namely QQSUM ,
for capturing and quantifying diverse opinions from
online reviews for PQA. We propose QQSUM-
RAG , a few-shot summarization model based on
retrieval-augmented generation where summary is
generated by LLMs from groups of user opinions
relevant to a query. QQSUM-RAG addresses the
issue of existing RAG frameworks for providing
only random or major opinion in the answer. By ex-
tending the retriever with opinion-based clustering
of relevant comments, our model ensures captur-
ing more diverse and representative opinions in the
summary, along with accurate quantification. Ex-
perimental results show that our solution greatly
enhances both the quality and quantitative perfor-
mance of key point generation in summaries.

Acknowledgement
This research is supported in part by the
Australian Research Council Discovery Project
DP200101441 .
Limitations
We evaluated the textual quality of generated KPs
only on AmazonQ&A, as it is the only (to our
best knowledge) public dataset with abundance of
online community answers written by online users
usable as ground truth for our automatic evaluation.
Since we are leveraging answers from Ama-
zonQ&A to summarize and quantify the prevalence
of query-relevant opinions from reviews regarding
a query, an inevitable limitation is that key points
extracted from the Q&A answers might not fully
in line with viewpoints in reviews to answer ques-
tions. Similarly, opinions in product reviews also
may not sufficiently cover all opinions in commu-
nity answers.
Ethics Statement
We have applied ethical research standards in our
organization for data collection and processing
throughout our work.
The AmazonQ&A dataset used in our experi-
ments was publicly crowdsourced and released for
the research publication for the review-based prod-
uct question answering task (Gupta et al., 2019).
The dataset was published following their ethical
standard, after removing all personal information.
The answers to questions do not contain contents
that are harmful to readers.
We ensured fair compensation for crowd anno-
tators on Amazon Mechanical Turk. We setup and
conducted fair payment to workers on their annota-
tion tasks/assignments according to our organiza-
tion’s standards, with an estimation of the difficulty
and expected time required per task based on our
own experience. Especially, we also made bonus
rewards to annotators who exerted high-quality an-
notations in their assignments.
References
Stefanos Angelidis and Mirella Lapata. 2018. Sum-
marizing opinions: Aspect extraction meets senti-
ment prediction and they are both weakly supervised.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing , pages
3675–3686, Brussels, Belgium. Association for Com-
putational Linguistics.Roy Bar-Haim, Lilach Eden, Roni Friedman, Yoav Kan-
tor, Dan Lahav, and Noam Slonim. 2020a. From ar-
guments to key points: Towards automatic argument
summarization. In Proceedings of the 58th Annual
Meeting of the Association for Computational Lin-
guistics , pages 4029–4039, Online. Association for
Computational Linguistics.
Roy Bar-Haim, Lilach Eden, Yoav Kantor, Roni Fried-
man, and Noam Slonim. 2021. Every bite is an ex-
perience: Key Point Analysis of business reviews.
InProceedings of the 59th Annual Meeting of the
Association for Computational Linguistics and the
11th International Joint Conference on Natural Lan-
guage Processing (Volume 1: Long Papers) , pages
3376–3386, Online. Association for Computational
Linguistics.
Roy Bar-Haim, Yoav Kantor, Lilach Eden, Roni Fried-
man, Dan Lahav, and Noam Slonim. 2020b. Quanti-
tative argument summarization and beyond: Cross-
domain key point analysis. In Proceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP) , pages 39–49, On-
line. Association for Computational Linguistics.
Adithya Bhaskar, Alex Fabbri, and Greg Durrett. 2023.
Prompted opinion summarization with gpt-3.5. In
Findings of the Association for Computational Lin-
guistics: ACL 2023 , pages 9282–9300.
Ralph Allan Bradley and Milton E. Terry. 1952. RANK
ANALYSIS OF INCOMPLETE BLOCK DESIGNS:
THE METHOD OF PAIRED COMPARISONS.
Biometrika , 39(3-4):324–345.
Arthur Bražinskas, Mirella Lapata, and Ivan Titov.
2020a. Few-shot learning for opinion summarization.
InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP) ,
pages 4119–4135, Online. Association for Computa-
tional Linguistics.
Arthur Bražinskas, Mirella Lapata, and Ivan Titov.
2020b. Unsupervised opinion summarization as
copycat-review generation. In Proceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics , pages 5151–5169, Online. Asso-
ciation for Computational Linguistics.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-V oss,
Gretchen Krueger, Tom Henighan, Rewon Child,
Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens
Winter, Chris Hesse, Mark Chen, Eric Sigler, Ma-
teusz Litwin, Scott Gray, Benjamin Chess, Jack
Clark, Christopher Berner, Sam McCandlish, Alec
Radford, Ilya Sutskever, and Dario Amodei. 2020.
Language models are few-shot learners. In Ad-
vances in Neural Information Processing Systems ,
volume 33, pages 1877–1901. Curran Associates,
Inc.

Kathy Charmaz. 2015. Grounded theory. Qualitative
psychology: A practical guide to research methods ,
3:53–84.
Long Chen, Ziyu Guan, Wei Zhao, Wanqing Zhao, Xi-
aopeng Wang, Zhou Zhao, and Huan Sun. 2019a.
Answer identification from product reviews for user
questions by multi-task attentive networks. In Pro-
ceedings of the AAAI Conference on Artificial Intelli-
gence , volume 33, pages 45–52.
Qibin Chen, Junyang Lin, Yichang Zhang, Hongxia
Yang, Jingren Zhou, and Jie Tang. 2019b. Towards
knowledge-based personalized product description
generation in e-commerce. In KDD 2019 , pages
3040–3050.
Shiqian Chen, Chenliang Li, Feng Ji, Wei Zhou, and
Haiqing Chen. 2019c. Driven answer generation for
product-related questions in e-commerce. In Pro-
ceedings of the Twelfth ACM International Confer-
ence on Web Search and Data Mining , pages 411–
419.
Eric Chu and Peter Liu. 2019. Meansum: A neural
model for unsupervised multi-document abstractive
summarization. In International Conference on Ma-
chine Learning , pages 1223–1232. PMLR.
Yang Deng, Wenxuan Zhang, and Wai Lam. 2020.
Opinion-aware answer generation for review-driven
question answering in e-commerce. In Proceedings
of the 29th ACM International Conference on Infor-
mation & Knowledge Management , pages 255–264.
Yang Deng, Wenxuan Zhang, Qian Yu, and Wai Lam.
2023. Product question answering in E-commerce: A
survey. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 11951–11964, Toronto,
Canada. Association for Computational Linguistics.
Roni Friedman, Lena Dankin, Yufang Hou, Ranit
Aharonov, Yoav Katz, and Noam Slonim. 2021.
Overview of the 2021 key point analysis shared task.
InProceedings of the 8th Workshop on Argument
Mining , pages 154–164, Punta Cana, Dominican Re-
public. Association for Computational Linguistics.
Shen Gao, Zhaochun Ren, Yihong Zhao, Dongyan Zhao,
Dawei Yin, and Rui Yan. 2019. Product-aware an-
swer generation in e-commerce question-answering.
InProceedings of the Twelfth ACM International
Conference on Web Search and Data Mining , pages
429–437.
Shai Gretz, Roni Friedman, Edo Cohen-Karlik, As-
saf Toledo, Dan Lahav, Ranit Aharonov, and Noam
Slonim. 2020. A large-scale dataset for argument
quality ranking: Construction and analysis. In Pro-
ceedings of the AAAI Conference on Artificial Intelli-
gence , volume 34, pages 7805–7813.
Mansi Gupta, Nitish Kulkarni, Raghuveer Chanda,
Anirudha Rayasam, and Zachary C Lipton. 2019.
Amazonqa: A review-based question answering task.
arXiv preprint arXiv:1908.04364 .Xingwei He, Zhenghao Lin, Yeyun Gong, A-Long Jin,
Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan
Duan, and Weizhu Chen. 2024. AnnoLLM: Making
large language models to be better crowdsourced an-
notators. In Proceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies (Volume 6: Industry Track) , pages 165–190,
Mexico City, Mexico. Association for Computational
Linguistics.
Edward Hu, Yelong Shen, Phil Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Lu Wang, and Weizhu Chen. 2021.
Lora: Low-rank adaptation of large language models.
Preprint , arXiv:2106.09685.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2022. Unsupervised dense infor-
mation retrieval with contrastive learning.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models. Journal of Machine
Learning Research , 24(251):1–43.
Manav Kapadnis, Sohan Patnaik, Siba Panigrahi, Varun
Madhavan, and Abhilash Nandy. 2021. Team enigma
at ArgMining-EMNLP 2021: Leveraging pre-trained
language models for key point matching. In Pro-
ceedings of the 8th Workshop on Argument Mining ,
pages 200–205, Punta Cana, Dominican Republic.
Association for Computational Linguistics.
J Richard Landis and Gary G Koch. 1977. The mea-
surement of observer agreement for categorical data.
biometrics , pages 159–174.
Hao Li, Viktor Schlegel, Riza Batista-Navarro, and
Goran Nenadic. 2023. Do you hear the people sing?
key point analysis via iterative clustering and abstrac-
tive summarisation. In Proceedings of the 61st An-
nual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers) , pages 14064–
14080, Toronto, Canada. Association for Computa-
tional Linguistics.
Piji Li, Zihao Wang, Lidong Bing, and Wai Lam. 2019.
Persona-aware tips generation? In WWW 2019 ,
pages 1006–1016.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. In Text Summariza-
tion Branches Out , pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang,
Ruochen Xu, and Chenguang Zhu. 2023. G-eval:
NLG evaluation using gpt-4 with better human align-
ment. In Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing ,
pages 2511–2522, Singapore. Association for Com-
putational Linguistics.

Julian McAuley and Alex Yang. 2016. Addressing
complex and subjective product-related queries with
customer reviews. In Proceedings of the 25th In-
ternational Conference on World Wide Web , pages
625–635.
Leland McInnes, John Healy, Steve Astels, et al. 2017.
hdbscan: Hierarchical density based clustering. J.
Open Source Softw. , 2(11):205.
Rada Mihalcea and Paul Tarau. 2004. TextRank: Bring-
ing order into text. In Proceedings of the 2004 Con-
ference on Empirical Methods in Natural Language
Processing , pages 404–411, Barcelona, Spain. Asso-
ciation for Computational Linguistics.
Thibault Sellam, Dipanjan Das, and Ankur Parikh. 2020.
BLEURT: Learning robust metrics for text genera-
tion. In Proceedings of the 58th Annual Meeting of
the Association for Computational Linguistics , pages
7881–7892, Online. Association for Computational
Linguistics.
Taylor Sorensen, Jared Moore, Jillian Fisher,
Mitchell Gordon, Niloofar Mireshghallah,
Christopher Michael Rytting, Andre Ye, Li-
wei Jiang, Ximing Lu, Nouha Dziri, et al. 2024. A
roadmap to pluralistic alignment. arXiv preprint
arXiv:2402.05070 .
Yoshihiko Suhara, Xiaolan Wang, Stefanos Angelidis,
and Wang-Chiew Tan. 2020. OpinionDigest: A sim-
ple framework for opinion summarization. In Pro-
ceedings of the 58th Annual Meeting of the Asso-
ciation for Computational Linguistics , pages 5789–
5798, Online. Association for Computational Lin-
guistics.
An Tang, Xiuzhen Zhang, and Minh Dinh. 2024a.
Aspect-based key point analysis for quantitative sum-
marization of reviews. In 18th Conference of the
European Chapter of the Association for Computa-
tional Linguistics .
An Tang, Xiuzhen Zhang, Minh Dinh, and Erik Cam-
bria. 2024b. Prompted aspect key point analysis for
quantitative review summarization. In Proceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 10691–10708, Bangkok, Thailand. Association
for Computational Linguistics.
Wenhui Wang, Hangbo Bao, Shaohan Huang, Li Dong,
and Furu Wei. 2021. MiniLMv2: Multi-head self-
attention relation distillation for compressing pre-
trained transformers. In Findings of the Association
for Computational Linguistics: ACL-IJCNLP 2021 ,
pages 2140–2151, Online. Association for Computa-
tional Linguistics.
Jianxing Yu, Zheng-Jun Zha, and Tat-Seng Chua. 2012.
Answering opinion questions on products by exploit-
ing hierarchical organization of consumer reviews.
InProceedings of the 2012 Joint Conference on Em-
pirical Methods in Natural Language Processing and
Computational Natural Language Learning , pages391–401, Jeju Island, Korea. Association for Compu-
tational Linguistics.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. In Proceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) ,
pages 11328–11348, Toronto, Canada. Association
for Computational Linguistics.
Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Pe-
ter Liu. 2020a. Pegasus: Pre-training with extracted
gap-sentences for abstractive summarization. In In-
ternational Conference on Machine Learning , pages
11328–11339. PMLR.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020b. Bertscore: Eval-
uating text generation with bert. In International
Conference on Learning Representations .
Chao Zhao and Snigdha Chaturvedi. 2020. Weakly-
supervised opinion summarization by leveraging ex-
ternal information. In Proceedings of the AAAI Con-
ference on Artificial Intelligence , volume 34, pages
9644–9651.
AOpinionated Question Classification for
AMAZON KP Dataset
Existing online product-related questions can be
categorized into two groups: subjective (opinion-
ated) or objective (factal). While subjective ques-
tions asks about positive/negative feeling or stance
(e.g., whether a product is “good" or “bad"), objec-
tive questions confirms the actual product details
(e.g., products properties, specific use-cases). In
E-Commerce, questions are often subjective, i.e.,
asking for former buyer’s opinion, where differ-
ent customers often have certain preferences over
product aspects or information needs (Chen et al.,
2019b; Li et al., 2019), leading to various expecta-
tions for the provided answers.
We extract subjective, i.e., opinionated, question
from AmazonQ&A by prompting the Mistral-7B
open-source LLM to analyze the question and its
associated answers, published by the online com-
munity. In this case, leveraging answers helps to
understand the nature of the questions, thereby bet-
ter reasoning whether the question is seeking for
subjective information from users. We present the
few-shot prompt for classifying opinionated, i.e.,
subjective, questions from AmazonQ&A in List-
ing 1.

Listing 1: Few-shot prompt (2 examples) for prompting Mistral-7B on opinionated question classification.
You will be provided with a question and multiple answers to that question, delimited by triple quotes.
The question was taken from a Question Answering dataset of product reviews, and can either be an opinionated or factual
question.
You were tasked to classify whether the given question is an opinionated or factual question.
Factual questions ask for objective data, specifications, or information that can be definitively answered based on product facts
, manual, user experience, or specifications. Factual question tends to covers unique and consistent opinions/fact in its
answers.
Opinionated questions are subjective and seek insights that are based on personal use, feelings, preferences, judgments, or
evaluations about a product. Opinionated question has multiple and diverse opinions in its answers.
Formally, you should perform the following steps:
1. Identify unique opinions from the answers of the given question
2. Based on the question content and the amount of opinions in the question's answer, identify the question's type.
Note that you must briefly explain why the question is opinionated or factual before giving the final decision.
Below are some few−shot examples:
Questions: How well does it work with wireless charging
Answers: ['Unfortunately with this case installed it will not hold the phone vertically.', 'I use the case with the official wireless
charger and have had no problems at all.', 'Works great. Not a fan of the dimensions.']
Type: 'Opinionated Question'
Questions: Are the shelves removeable?
Answers: ['yes, they are removeable..', 'Yes they are, you can arrange them for the size of the shot glass.']
Type: 'Factual Question'
B Qualitative Data Analysis of
Opinionated Questions’ Categories in
AMAZON KP
We further studied the utility of the QQSUM task
and our by conducting qualitative data analysis
to categorize possible opinionated question’s type
inAMAZON KP. Based on the grounded theory
methodology (Charmaz, 2015), our analysis em-
ploy human-LLM collaborative annotation to itera-
tively code the fine-grained categories from opin-
ionated questions. We sampled a subset of 100
questions from AMAZON KPfor data coding and
intepretation. On the subset, we start by prompting
ChatGPT to identify potential categories of opin-
ionated questions, including the categories’ name
and their definitions (Step 1). Importantly, the data
coding process involves human validation, in which
we iteratively a human annotator iteratively evalu-
ate the representative of generated categories while
interacting with ChatGPT, and manually refine the
categories where possible9(Step 2). Then, we
prompted a gp4-o-mini to annotate the labels of
entire questions in the subset, before asking human
annotator again to validate the representative and
9On categories requiring more fine-grained categorization,
we further conduct another analysis cycle on the particular
coarse-grained category, by selecting questions and answers
from the specific category for analysis.suitability of the candidate categories on questions.
Categories with abnormal distribution, e.g., 5 times
higher than others, or with high unmatching cases
will be passed back to Step 2 for another iterative
analysis cycles.
As a result, our analysis reported 5 categories
commonly representative of question in AMA-
ZONKP, namely, Performance ,Quality ,Recom-
mendation ,Comparative andControversial , with
each the stating clearly the purpose of the users
asking the questions and expected answers. Finally,
We prompted gpt-4-o-mini to annotate such cat-
egories on AMAZON KP’s opinionated questions,
and reported their taxonomy and statistics in Ta-
ble 6. Notably, the dominance of “ Scenario-based ”
questions underscore the importance of QQSUM
for generating KP summary to answer user ques-
tions on preferences and scenarios.
C Human Validation of GPT4’s Key
Point Extraction from Gold
Community Answer of AmazonQ&A
In this experiment, we empirically validate
gpt-4-o-mini ’s performance and credibility in ex-
tracting KPs from gold community answers for
AmazonKP (Stage 1 of §3.3). Specifically, to
maintain reasonable cost, we sampled a question,

Category Description Example # Query
Performance Ask how well a product performs or func-
tions in general.How well does it work on carpet? 376
Quality Ask about the overall or aspect-specific qual-
ity of the product.Is this product worth the money? 265
Scenario-based Ask whether a product fits specific use cases,
sizes, or other products.Does this item really stop the glare at night
even in rain or snow?1402
Recommendation Ask for suggestions tailored to specific is-
sues or use cases.What do you use to spray this stuff on your
lawn?156
Comparative Seeks opinions about the relative advantages
or disadvantages of a product compared to
others.Would a wired keyboard/mouse be better
than wireless?227
Controversial Reflect dissatisfaction or complaint about a
product, likely to provoke debate or contro-
versy.Why does this need adjustment screws? If I
have to align the laser then what’s the point?124
Table 6: A taxonomy of opinion questions A MAZON KP
i.e., queries, from 5 common product categories
of AmazonKP10, totaling 5 questions, and hired
workers to annotate whether the extracted KPs
matches original gold community answers of the
sampled questions, which is inspired by the KP
Matching evaluation of Bar-Haim et al. (2021).
More specifically, for a given query, we asked
workers to perform pairwise annotation between ex-
tracted KPs and the query’s respective community
answers. While Precision calculates the fraction of
KPs matched to at least one gold answer, i.e., out of
all extracted KPs how many are correctly mapped,
Recall shows the fractions of gold answers matched
to at least one KP, i.e., out of all answers how many
are covered by KPs. We then macro-averaged Pre-
cision/Recall computed for every question to obtain
the final values.
For human annotation, we employed 3 MTurk
crowd workers on every answer-KP pair, selecting
only those with an 80% or higher approval rate and
at least 10 approved tasks. Following Bar-Haim
et al. (2021), we exclude annotators with Annotator-
κ <0for quality control. This score averages all
pairwise Cohen’s Kappa (Landis and Koch, 1977)
for a given annotator, for any annotator sharing at
least50judgments with at least 2other annotators.
For labelling correct matches, we applied a strict
threshold, in which 100% votes (3 out of 3) of the
annotators had to agree that the match was correct.
Otherwise, it is incorrect.
Table 7 presents the fraction of extracted KPs
matched to at least one gold answer (Precision)
and vice versa (Recall). Overall, the experiment
confirms that the extracted KPs are of high
quality, with 90.0% of community answers were
10namely Home_and_Kitchen ,Sports_and_Outdoors ,
Tools_and_Home_Improvement ,Health_and_Personal_Care
andBeautyPrecision 87.5%
Recall 90.0%
# Matched Answers Per KP 2.39
# Matched KPs Per Answer 2.61
Table 7: Performance validation of gpt-4-o-mini ’s
KP extraction from gold community answer. While
precision calculates the fraction of KPs matched to at
least one gold answer, recall shows the fractions of gold
answers matched to at least one KP.
represented with KPs (recall), while 87.5% of the
extracted KPs are verified as valid (precision).
Below are the match annotation guidelines for
(extracted KP, gold answer) pairs:
In this task you are presented with a question on
a product, a key point extracted from community
answers answering the question, and a community
answer for answering the query of that product.
You will be asked to answer the following ques-
tion:"Does the key point match, i.e., represent an
opinion in the community answer?"
A community answer might express opinions on
multiple aspects. A key point matches a community
answer if it captures the gist of the answer, or is di-
rectly supported by a point made in the community
answer.
The options are:
• Not At All
• Somewhat Not Well
• Somewhat Well
• Very Well

D Prompt for Key Point Extraction from
Gold Community Answer of
AmazonQ&A
We present the few-shot prompts for extracting key
points (KPs) from gold online community answers
of AmazonKP in Listing 2.
E Annotation Details of KP Matching for
AMAZON KP Dataset
We offer GPT-4-o-mini with 4 options for
labelling the matching status of given comment-KP
pairs. Pairs annotated as Very Well orSomewhat
Well by LLM then becomes candidate matching
pairs , which will be further validated by human
annotation for their correctness. For human
annotation, we employed 3 MTurk crowd workers
per comment-KP pair, selecting only those with
an 80% or higher approval rate and at least 10
approved tasks. Following Bar-Haim et al. (2021),
we exclude annotators with Annotator- κ <0for
quality control. This score averages all pairwise
Cohen’s Kappa (Landis and Koch, 1977) for a
given annotator, for any annotator sharing at least
50judgments with at least 2other annotators. For
labelling correct matches, at least 60% of the
annotators had to agree that the match is correct,
otherwise, it is incorrect. Comments from final
matching pairs, after confirmed by human, will
then be grouped by similar KPs, where the amount
of matching comments per KP is the prevalence of
the respective KP.
Below are the matching prompt for LLM and
the annotation guidelines for workers validating
(sentence, KP) pairs:
In this task, you are presented with a question
on a product, a key point taken from the summary
answering the question, and a sentence taken from
a review of that product.
You will be asked to answer the following ques-
tion: "Does the key point match, i.e, represent an
opinion in the review sentence?"
A review sentence might express opinions on
multiple aspects. A key point matches a sentence
if it captures the gist of the sentence, or is directly
supported by a point made in the sentence.
The options are:
• Not At All
• Somewhat Not Well• Somewhat Well
• Very Well
F Prompts for KP Summary Generation
of QQSUM-RAG
We present the instruction-finetuning prompts for
KP Summary Generation of QQSUM-RAG in
Listing 3.
G Prompts for G-E VAL Evaluation
For implementation of G-E VAL in our KP qual-
ity evaluation dimension (§4.2), we specifically
customize the model’s original prompt for evaluat-
ing summary’s relevance andredundancy . While
therelevance evaluation prompt is customized for
evaluating sP/sF/sF1 (Li et al., 2023) between indi-
vidual generated KPs and the reference KPs, redun-
dancy is customized for evaluating RDamong gen-
erated KPs. We presented our relevance evaluation
prompt in Listing 4 and the redundancy evaluation
prompt in Listing 5
H Dimensions of KP Quality Evaluation
This section provides detailed descriptions of tasks
and dimensions involved in our manual evaluation
of the KP textual quality. Annotators were asked to
perform a pairwise comparison between two sets of
KPs, each taken from a different model, generated
for a specific reviewed business entity considering
a specific dimension. The annotators must answer
a comparative question with respect to the evaluat-
ing dimension. (e.g., Which of the two summaries
captures better . . . ). For each dimension, follow-
ing Friedman et al. (2021), we calculate the ranking
using the Bradley-Terry model (Bradley and Terry,
1952), which predicts the probability of a given
participant winning a paired comparison, based
on previous paired comparison results of multiple
participants, and thus allows ranking them.
•VALIDITY : The key point in the summary
should be an understandable, well-written sen-
tence representing an opinion of the users to-
wards the question. This would filter out sen-
tences such as “It’s rare these days to find
that!” .
•SENTIMENT : The key point in the summary
should have a clear sentiment towards the
product being questioned (either positive or

Listing 2: One-shot prompt (1 example) for prompting GPT-4-o-mini on KP Extraction from community answers.
You will be provided with an opinionated question and multiple answers to that question, delimited by triple quotes.
An opinionated question seek insights of user opinions that are based on personal use, feelings, preferences, judgments, or
evaluations about a product, and was taken from a Question Answering dataset of product reviews.
You were tasked to extract a list of unique and concise key points from the list of answers to given opinionated question.
Key points are short and high quality sentences that expresses the main claims/viewpoints of users answering the opinionated
question
Note that the final extracted list of key points must directly relevant and can answer the input opinionated question.
Formally, you should perform the following steps:
1. In every answer from the list, extract all possible key point candidates.
2. From the extracted list of key point candidates, generate a list of only general and non−overlapping key points that are
relevant and can answer the input opinionated question.
Below are some few−shot examples:
Questions: Can I use these for running/working out? Do they handle sweat?
Answers: ['I have seen other people using these for running/working out. These are very comfortable in your ears for long
hours. As long you clean them after working out, you should be fine. These are built to last a long time.', 'I use them in
the gym and on the stair climber machine. They are fine. Not sure about running but would think they would work ok.', "
I don't know if I'll be any help, but I'll tell you about my experience nevertheless. I used it everyday in the gym & while I
go for work on my bike inside my helmet. In both cases, the sweat doesn't seem to have any effect. Even during long
rides, and when it rained heavily, the IE80 held up fine. The only issue you will have to worry about is the cable.
Though the cables are good quality, rough usage may affect the balance in volume levels between the two channels.
Though this doesn't affect the clarity, the balance can be disturbed. After a year of really rough usage, the IE80 right
volume was 1−2% lower than the left [I got mine replaced for free soon after]. But, this is an issue which affects every
IEM, and nothing to sweat over, since we can replace the cables if necessary. So if you don't give it a hard time, it
should hold up fine.[I can't even count the times it has fallen down or swung down and taken a hit against the gym
equipment, or when my phone/DAP slipped and yanked the cable]"]
Key Points: ['Comfortable for long hours', 'Built to last a long time', 'Suitable for gym and stair climber machine', 'Sweat
resistant during workouts', 'Cables may be affected by rough usage']
Listing 3: Prompt for instruction-finetuning QQSUM-RAG ’s LLM for KP Summary Generation. Please refer to
our released code for full prompts.
You will be provided with a question and a JSON list of relevant review comments, delimited by triple quotes.
The question asks the opinions of user reviews about a product, and can be answered by the list of comment clusters in the
provided JSON list. Each element in the JSON has been has been clustered to represent a common opinion answering
the question, accompanied by the quantity.
You were tasked to generate a quantitative summary that covers all opinions captured in the JSON list in answering the
questions.
Perform the following actions to solve this task:
− For every element in the JSON list, find the key point that represent the common opinion across the comments of the cluster
− Generate a long−form quantitative summary including all extracted key points and the cluster size, following the below
template:
'While answering about [Question]:
+ [Cluster size] of comments believe that [Key Point 1]
+ [Cluster size] of comments believe that [Key Point 2]
...'
Below are fundamental rules:
+ Larger cluster means higher support for the key point and with a bigger cluster size, the quantity must be higher
+ Only use number to report the cluster size for each key point, avoiding vague terms (e.g., some, most)
+ Ensure that each key point extracted from a cluster is distinctive and doesn't redundantly cover aspects mentioned in larger
clusters

Listing 4: Zero-shot prompt for G-E VALrelevancy evaluation between generated KPs and reference KPs, supporting
sP/sR/sF1 calculation.
You will be given one key point, short salient sentence, written to describe user opinion on a product.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and
refer to it as needed.
Evaluation Criteria:
Relevance (1−5) − selection of important content from the source. The summary should include only important information
from the source document. Annotators were instructed to penalize summaries which contained redundancies and excess
information.
Evaluation Steps:
1. Read the key point and the source key point carefully.
2. Compare the key point to the source key point and identify the main points.
3. Assess how well the key point covers the main points of the source key point, and how much irrelevant or redundant
information it contains.
4. Assign a relevance score from 1 to 5.
Listing 5: Zero-shot prompt for G-E VAL redundancy evaluation of generated KPs, supporting RDcalculation.
You will be given one key point, short salient sentence, written to describe user opinion on a product.
Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and
refer to it as needed.
Evaluation Criteria:
Redundancy (1−5) − overlapping opinion with the source. The summary should not include semantically similar opinion with
the source document. Annotators were instructed to penalize summaries which contained overlapping opinion with the
source.
Evaluation Steps:
1. Read the key point and the source key point carefully.
2. Compare the key point to the source key point and identify the main points.
3. Assess how much redundant opinion and information the key point covers that overlap with the source key point
4. Assign a redundancy score from 1 to 5.

negative). This would exclude sentences like
“I came for a company event” .
•INFORMATIVENESS : The key point in the
summary should discuss should discuss some
aspects of the reviewed product and contain
useful information. Any key point that is too
specific or only expresses sentiment cannot be
considered a good candidate.
•SINGLE ASPECT : The key point in the sum-
mary should not discuss multiple aspects (e.g.,
“Decent price, respectable portions, good fla-
vor” ).
•REDUNDANT : Each KP should express a dis-
tinct aspect. In other words, there should be
no overlap between the key points.
•COVERAGE : The summary, containing the set
of key points, should cover a wide diversity
of opinions relevant and representative to the
question.
•FAITHFULNESS : The key point in the sum-
mary should express reasonable and meaning-
ful opinions relevant to the question raised on
the product without hallucination. No conjec-
ture or unfounded claims should arise.
I Pairwise KP Quality Comparison
Annotation Guidelines
Below are the two summaries for a product ques-
tion in Tools_and_Home_Improvement , generated
by two different summarization frameworks. Each
summary contains several key points (i.e., salient
points) generated summarizing the user opinions
on different aspects. You are tasked to select which
summary you think is better according to the below
criteria.
Question: Does this tester accurately test AA
Lithium? The power drop off curve is so steep. It
seems unlikely...but I am hoping!.
Criteria: REDUNDANCY . Each key point in
the summary should express a distinct aspect. In
other words, there should be no overlap between
the key points.
Summary A: [’the tester accurately tests vari-
ous types of batteries, including AA Lithium, and
provides accurate readings’, ’there is uncertainty
about the accuracy of the percentage of charge re-
maining for AA Lithium batteries’, ’the tester does
not test a specific version of AA Lithium battery(L91)’, ’the tester is big and cumbersome, but ef-
fective in testing batteries under load’, ’the tester
requires four AA batteries to operate’, ’the tester
tests batteries by putting a load on them, making
the readings more accurate’, ’the tester tests batter-
ies quickly, with a test taking only 3-4 seconds for
a AA battery’, ’the tester is expensive but worth the
investment due to its accuracy and ability to save
money by testing old batteries’, ’the tester tests
batteries of various sizes, including AA, AAA, C’]
Summary B: [’I have compared the testers re-
sults to battery powered devices and found it does
give you the true useful state of a battery. ’, ’Now
that I found this tester I am happy, because it tests
a battery the way a battery should be tested.’, ’That
model also tests 6v lithium 2CR5 used in some
older cameras, which the current tester does not
since the times have moved on.’]
The options are:
• Summary A
• Summary B
J GPT4’s Comment-KP Matching
Annotation against Human Judgement
To validate gpt-4-o-mini ’s annotation perfor-
mance and credibility, we conduct an experiment to
measure LLM annotation judgement, as utilized for
the KP-comment matching evaluation in our main
experiment, in agreement with human (gold) pref-
erence. We sampled a subset of 5 queries from the
test set in our main experiment and hired workers
to annotate the correctness of comment-KP pairs
produced as the results of our framework’s quan-
tification outcome. Note that these sampled pairs
are part of the our main test set and have already
been annotated for LLM’s labels in our main ex-
periment. For human annotation, we employed 6
MTurk crowd workers on every comment-KP pair,
selecting only those with an 80% or higher approval
rate and at least 10 approved tasks. Following Bar-
Haim et al. (2021), we exclude annotators with
Annotator- κ < 0for quality control. This score
averages all pairwise Cohen’s Kappa (Landis and
Koch, 1977) for a given annotator, for any annotator
sharing at least 50judgments with at least 5other
annotators. For labelling correct matches, at least
60% of the annotators had to agree that the match
is correct, otherwise, it is incorrect. In this exper-
iment, we measured the accuracy, and conducted
a Pearson correlation ( r) test of gpt-4-o-mini ’s

annotation performance against human judgement,
with results reported in Table 8. For rtest, we set
the null hypothesis as gpt-4-o-mini ’s and Mturk
annotated labels are independent.
From Table 8, we saw signficant small p-value,
which indicates strong evidence against the null
hypothesis. Importantly, we also recorded Spear-
man’s rank correlation coefficient to be relatively
closed to 1. This implies that there is a sta-
tistically significant positive correlation between
gpt-4-o-mini and Mturk annotated labels, which
substantiates our decision of using gpt-4-o-mini
for comment-KP matching evaluation.
Pearson correlation ( r)0.647
p_value 5.342e-16
Accuracy 0.807
Table 8: Performance valiation of GPT4’s comment-KP
matching annotation against human judgement
Below are the match annotation guidelines for
(sentence, KP) pairs:
In this task, you are presented with a question
on a product, a key point taken from the summary
answering the question, and a sentence taken from
a review of that product.
You will be asked to answer the following ques-
tion: "Does the key point match, i.e, represent an
opinion in the review sentence?"
A review sentence might express opinions on
multiple aspects. A key point matches a sentence
if it captures the gist of the sentence, or is directly
supported by a point made in the sentence.
The options are:
• Not At All
• Somewhat Not Well
• Somewhat Well
• Very Well
K Clustering Algorithm of KP-Oriented
Retrieval in QQSUM-RAG
To validate other clustering techniques, we have de-
veloped an additional baseline that employs either
HDBSCAN (McInnes et al., 2017) or K-Means
clustering algorithm for grouping similar com-
ments by the Retriever, following our main exper-
imental setup and configuration in Section 4.2.2.Better than K-Means, HDBSCAN can automati-
cally detect the number of clusters without pre-
defined parameters and is used in a previous KPA
work (Li et al., 2023). We compare the factual
alignment of KP-comment pairs (measured by
AlignScore) across clustering methods in Table 9,
using our best model configuration (Contriever +
Mistral):
While both HDBSCAN and K-Means perform
reasonably, they are consistently outperformed by
our specialized clustering approach. More specifi-
cally, although HDBSCAN or K-Means achieves
relatively comparable matching Precision with our
clustering algorithm, our algorithm can capture
comments more sufficiently (much higher Recall)
than HDBSCAN and K-Means. This is mostly be-
cause our algorithm contains more tuneable cluster-
ing parameters and operations that are specifically
optimized for the QQSUM problem.
L Ablation Study: Single-KP Generation
vs KP Summary Generation in
QQSUM-RAG
We conducted an ablation study to evaluate the
impact of KP Summary Generation on QQSUM-
RAG , with KP quality and KP-comment matching
and factual consistency performance presented in
Table 10 and 11 respectively. To this end, we con-
figure QQSUM-RAG Single-KP , a variant that gen-
erates one KP at a time for each comment cluster
formed by KP-oriented Retrieval.
Overall, not including previously generated KPs
as context, QQSUM-RAG Single-KP struggles to
capture the truly representative opinion of the clus-
ter, likely generating KPs with overlapping opin-
ions, especially for comments containing multiple
opinions.
M Example output of QQSUM-RAG
and Baselines
We report the example output of query-relevant
comment clusters and KP summary produced by
QQSUM-RAG in Table 12 and 13, and further
compare top 5 key points, extracted from the sum-
mary of QQSUM-RAG and the baselines in Ta-
ble 14. Overall, QQSUM-RAG stands out for gen-
erating KPs with minimal redundancy, higher in-
formativeness, and better alignment with the query.

KP-Comment Matching KP-Comment Factual Alignment
P R F1 QuantErr ↓ AlignScore
Our proposed clustering algorithm 0.694 0.869 0.792 04.24 0.749
HDBSCAN clustering algorithm 0.682 0.507 0.582 11.47 0.718
K-Means clustering algorithm (n_clusters = 3) 0.677 0.424 0.522 15.50 0.681
Table 9: KP-Comment matching performance and factual consistency of generated summary between different
clustering methods applied for KP-oriented Retrieval of QQSUM-RAG . The experiment was conducted with the
Mistral configuration for QQSUM-RAG, proven to have superior performance than Vicuna from Table 3.
ROUGE BERTScore BLEURT G-Eval-4
R-1 R-2 R-L sP sR sF1 RD ↓Rel sP sR sF1 RD ↓Rel sP sR sF1 RD ↓Rel
QQSUM-RAG (Ours)
+ Mistral 0.256 0.061 0.220 0.39 0.29 0.33 0.37 0.27 0.51 0.41 0.46 0.49 0.45 4.52 4.29 4.40 2.43 4.05
+ Vicuna 0.222 0.078 0.204 0.38 0.26 0.31 0.53 0.25 0.49 0.39 0.44 0.54 0.41 4.47 4.25 4.36 2.45 3.68
QQSUM-RAG Single-KP
+ Mistral 0.191 0.035 0.160 0.29 0.22 0.25 0.48 0.22 0.48 0.39 0.43 0.62 0.39 4.21 4.22 4.22 2.51 3.14
+ Vicuna 0.171 0.045 0.154 0.22 0.17 0.19 0.57 0.20 0.48 0.38 0.42 0.66 0.36 4.10 4.12 4.11 2.60 2.87
Table 10: KP-level textual quality evaluation of generated summary between full implementation of QQSUM-
RAG and without (w/o) KP Summary Generation. sP, sR and sF1 refer to Soft-Precision, Soft-Recall, and Soft-F1
respectively based on set-level evaluation method against reference KPs in gold answer. G-E VAL-4 asks GPT-4 to
score a summary from 1-5.
KP-Comment Matching KP-Comment Factual Consistency
P R F1 QuantErr ↓AlignScore
(cluster-level)AlignScore
(retrieval-level)
QQSUM-RAG (Ours)
+ Mistral 0.694 0.869 0.792 04.24 0.749 0.826
+ Vicuna 0.538 0.684 0.602 07.83 0.630 0.690
QQSUM-RAG Single-KP
+ Mistral 0.640 0.520 0.574 17.84 0.682 0.741
+ Vicuna 0.598 0.471 0.527 22.63 0.601 0.660
Table 11: KP-Comment matching performance and factual consistency of generated summary between full
implementation of QQSUM-RAG and without (w/o) KP Summary Generation.

Query How does this Nikon 24-120mm F4 lens compared with the 24-70mm F2.8 as a general walk around lense?
Query-
Relevant
Comment
ClustersCluster1:
• I like the 24-70 better but this lens is a good all around and compact optic for everyday shooting.
•As has been said many times before: "the best lens is the one you will use", and I know I wouldn not use the 24-70mm F2.8 because it ´s too
heavy and bulky to take on backpacking/camping trips and when traveling abroad.
• This is the one lens which could replace 24-70 / 2.8, 70-200 2.8 VR II (up to some extent) for "everyday" use. ’
• . . .
Cluster2:
•I have an upcoming stay in Spain, and I’m seriously considering taking this lens instead of my AF-S 24-70 because of its size and zoom
range.
•My only complaint is the price tag: for a lens that is overall a rather mixed bag (depending on what you’re looking for you might be very
happy with it, or very disappointed) it is very expensive .
•The 24-120 has good reach, good image quality, not heavy, not that expensive for what it can do (constant f/4 in a zoom is very respectable)
and it’s also the only usable medium-telephoto FX zoom from Nikon with the VR technology.
•For a 5x zoom to be able to compete with a 3x zoom costing over $500 more(the Nikkor 24-70mm F2.8) should only mean that the 5x zoom
is a remarkable lens.
• . . .
Cluster3:
•For one thing, 24 70 is know to have better quality than this one.
•The range from 70 to 120 is not as important as a better overall quality.
•This is probably not the best lens to use for portraits because it’s just not fast enough (f-stop) , but for travel, chasing you kids around, or any
other every day shooting this lens is perfect.
•The biggest pro for the 24-70mm is the extra 1 stop of light, slightly quicker autofocus speed, and of course the corresponding softer bokeh
due to the 1 stop aperture opening.
• . . .
KP Summary While comparing the Nikon 24-120mm F4 lens with the 24-70mm F2.8 lens as a general walk-around lens:
+ 135 of comments believe that theNikon 24-120mm F4lensisrelatively lightweight andcompact, makingiteasy tocarry around anduse
forextended periodsoftime.
+ 11 of comments suggest that the24-120mm F4lenshasalonger zoom range andismore affordable than the24-70mm F2.8.
+ 9 of comments preferthe24-70mm F2.8 foritsbetterimagequalityandfaster aperture.
. . .
Table 12: Example output of query-relevant comment clusters and KP summary produced by QQSUM-RAG , given
a query, i.e., question, from AmazonQ&A. Comment clusters to a particular KP are marked in the same color as the
corresponding bullet in the summary. The relevant opinion in each comment that directly support the corresponding
KP is italicized .
Query : How does this Nikon 24-120mm F4 lens compared with the 24-70mm F2.8 as a general walk around lense?
Key Point Prevalence Matching Comments
The Nikon 24-120mm F4 lens is rela-
tively lightweight and compact, mak-
ing it easy to carry around and use for
everyday shooting.135 I like the 24-70 better but this lens is a good all around and
compact optic for everyday shooting.
As has been said many times before: "the best lens is the one
you will use", and I know I wouldn not use the 24-70mm F2.8
because it ´s too heavy and bulky to take on backpacking/camping
trips and when traveling abroad.
The 24-120mm F4 lens has a longer
zoom range and is more affordable
than the 24-70mm F2.8.11 I have an upcoming stay in Spain, and I’m seriously considering
taking this lens instead of my AF-S 24-70 because of its size and
zoom range.
My only complaint is the price tag: for a lens that is overall a
rather mixed bag (depending on what you’re looking for you
might be very happy with it, or very disappointed) it is very
expensive .
Prefer the 24-70mm F2.8 for its better
image quality, faster aperture and bet-
ter for wide shot.9 For one thing, 24 70 is know to have better quality than this one.
The range from 70 to 120 is not as important as a better overall
quality.
Table 13: Top 3 key points mentioned in the KP summary produced by QQSUM-RAG for answering a query from
AMAZON KP. For each key point, we show the prevalence, i.e., number of matching comments (with similar aspects
of the same cluster), and two top matching comments. The relevant opinion in each comment that directly support
the corresponding KP is italicized .

Query : How does this Nikon 24-120mm F4 lens compared with the 24-70mm F2.8 as a general walk around lense?
QQSUM-RAG (Retriever+
LLM) co-trainedContriever +
GPT-4-TurboContriever +
PAKPAContriever +
RKPA-Base
The Nikon 24-120mm
F4 lens is relatively
lightweight and com-
pact, making it easy to
carry around and use
for everyday shootingThe 24-120mm f/4 of-
fers more reach and
versatility than the 24-
70mm f/2.8.The 24-120mm lens
offers good versatility
and value for general
useThe 24-120 lens is
preferred over the
Nikkor 24-70mm
F2.8. due to its lighter
weight.The 24-120 is finally
at a stage where you
can carry it around on
your FX camera and
have no regrets.
The 24-120mm F4
lens has a longer
zoom range and is
more affordable than
the 24-70mm F2.8.The24-120mm f/4is
lighter and more af-
fordable than the 24-
70mm f/2.8.The 24-70mm lens
has superior image
quality and perfor-
manceBest 4+star walk -
around lens onthe
market.If you want a 4+ star
walk-around lens that
covers a great range ,
this is the best on the
market.
Prefer the 24-70mm
F2.8 for its better
image quality, faster
aperture and better for
wide shot.The24-70mm f/2.8 is
abetterlensoverall.the 24-70mm lens is
preferred for its opti-
calsuperiority.The 24-70mm lens is
highly recommended
for wide shots.The 24-70mm lens is
more expensive but
buy it if you need to
shoot wide.
The 24-120mm F4
lens has good image
quality, with sharp-
ness and contrast that
is comparable to the
24-70mm f/2.8The24-120mm f/4is
tooheavy.The 24-120mm lens
isamore practical
choice foreveryday
use.The Nikon 24-120
lens has good con-
trast compared to the
Nikon 24-70 lens.I briefly considered
the 24-70, but the
extra reach , vibra-
tion reduction, and
lower price point sold
me on this lens.
The 24-120mm F4
lens has good Vibra-
tion Reduction (VR)
technology that helps
to reduce camera
shake when taking
handheld shots.The 24-120mm f/4
has image stabiliza-
tion, which is a sig-
nificant advantage for
handheld shots.The 24-120mm f/4
has image stabiliza-
tion for handheld
shots.N/A N/A
. . .
Table 14: Top 5 key points, extracted from the summary of QQSUM-RAG and the baselines, ranked by their
prevalence on an example query from AMAZON KP. Overlapping opinions across KPs are highlighted red. KPs
lacking of informativeness are highlighted yellow