# Judging by Appearances? Auditing and Intervening Vision-Language Models for Bail Prediction

**Authors**: Sagnik Basu, Shubham Prakash, Ashish Maruti Barge, Siddharth D Jaiswal, Abhisek Dash, Saptarshi Ghosh, Animesh Mukherjee

**Published**: 2025-09-30 12:11:45

**PDF URL**: [http://arxiv.org/pdf/2510.00088v1](http://arxiv.org/pdf/2510.00088v1)

## Abstract
Large language models (LLMs) have been extensively used for legal judgment
prediction tasks based on case reports and crime history. However, with a surge
in the availability of large vision language models (VLMs), legal judgment
prediction systems can now be made to leverage the images of the criminals in
addition to the textual case reports/crime history. Applications built in this
way could lead to inadvertent consequences and be used with malicious intent.
In this work, we run an audit to investigate the efficiency of standalone VLMs
in the bail decision prediction task. We observe that the performance is poor
across multiple intersectional groups and models \textit{wrongly deny bail to
deserving individuals with very high confidence}. We design different
intervention algorithms by first including legal precedents through a RAG
pipeline and then fine-tuning the VLMs using innovative schemes. We demonstrate
that these interventions substantially improve the performance of bail
prediction. Our work paves the way for the design of smarter interventions on
VLMs in the future, before they can be deployed for real-world legal judgment
prediction.

## Full Text


<!-- PDF content starts -->

Judging by Appearances? Auditing and Intervening
Vision-Language Models for Bail Prediction
Sagnik Basu1Shubham Prakash1Ashish Maruti Barge1Siddharth D Jaiswal1
Abhisek Dash2Saptarshi Ghosh1Animesh Mukherjee1
1Indian Institute of Technology Kharagpur, India
2Max Planck Institute for Software Systems, Saarbruecken, Germany
1{basusagnik99.24, shubham.0313, ashishbarge.24, siddsjaiswal}@kgpian.iitkgp.ac.in
1{animeshm, saptarshi}@cse.iitkgp.ac.in
2adash@mpi-sws.org
Abstract
Large language models (LLMs) have been extensively used
for legal judgment prediction tasks based on case reports
and crime history. However, with a surge in the availabil-
ity of large vision language models (VLMs), legal judgment
prediction systems can now be made to leverage the images
of the criminals in addition to the textual case reports/crime
history. Applications built in this way could lead to inad-
vertent consequences and be used with malicious intent. In
this work, we run an audit to investigate the efficiency of
standalone VLMs in the bail decision prediction task. We
observe that the performance is poor across multiple inter-
sectional groups and models wrongly deny bail to deserving
individuals with very high confidence. We design different
intervention algorithms by first including legal precedents
through a RAG pipeline and then fine-tuning the VLMs us-
ing innovative schemes. We demonstrate that these inter-
ventions substantially improve the performance of bail pre-
diction. Our work paves the way for the design of smarter
interventions on VLMs in the future, before they can be de-
ployed for real-world legal judgment prediction.
1. Introduction
Artificial Intelligence (AI) is showing significant promise
for transforming the legal landscape, with research explor-
ing its application in automating tasks like legal document
summarization [2], legal information retrieval [36], legal
judgment prediction (LJP) [25] and many more. Among
the most consequential ones is LJP, where machine learn-
ing or deep learning models predict judicial outcomes —
such as bail decisions [26], conviction status, or sentenc-
ing [23, 25]— by reasoning over case facts, statutes (a
formal, written law) and precedents [44] (relevant previ-ous cases). LJP has been extensively studied across sev-
eral jurisdictions, with applications ranging from Supreme
People’s Court of China [47] to U.S. Supreme Court deci-
sions [19], as well as emerging applications in India [25]
and Australia [37]. In recent times, the complex reasoning
process required for LJP has been transformed by the ad-
vent of large language models (LLMs) which enable richer
analyses of these legal documents [17].
If we consider an LJP scenario in a real courtroom
setup (where a case is being argued in a court of Law),
there are multiple modalities involved with the decision
making process, such as images and videos as evidence
of the crime (e.g. face photographs, evidence from CCTV
footage), speech data from the witnesses, as well as the tex-
tual data (e.g. FIRs, judgments, witness statements in text).
However, existing studies in LJP rely heavily on textual
data, such as court opinions, statutes and case briefings for
modelling judicial outcomes. The days are not far when vi-
sion language models (VLMs) [20, 22, 39, 41], which can
understand and integrate information from both image and
text domains, would very likely be used to enhance AI-
assisted legal decision making. In fact, recently, VLMs
have been used to scan and process legal documents. For
instance, they have been used to understand first informa-
tion report (FIR) forms, which are often hand-written and
scanned [6, 43]. Another area where VLMs are being ex-
tensively used is in forensic image analysis, where it has
been observed that such AI tools cannot replace forensic
experts [11] and can at best work as an assistive tool.
While VLMs are being increasingly used for various
tasks in the legal and other domains, they have certain lim-
itations as well.Bias issuesin VLMs are an active area
of concern [35, 46], especially as these models are increas-
ingly applied in sensitive domains like healthcare [15], ed-
ucation [38] and law [6]. Continiet al. [8] discuss how the
1arXiv:2510.00088v1  [cs.AI]  30 Sep 2025

emergence of such generative AI tools has reduced the de-
livery of justice to mere statistical correlations among data,
thus completely suspending the emotive-cognitive process
that lies at the core of legal decision making. These is-
sues raise critical questions about fairness, accountability
and generalizability of AI systems in legal applications.
One of the specific tasks in AI assisted LJP isbail de-
cision prediction, wherein the model predicts whether an
accused person can be granted bail or not based on multiple
facts and evidences. This is a binary prediction task, where
the output 0/no indicates that the accused should be denied
bail, and 1/yes indicates that the accused should be granted
bail. As in a real-world scenario, multiple modalities are
involved, to further reduce workloads in courts, VLMs have
much potential to be used in these setups where a face im-
age is provided along with the case texts. This work tries to
“foresee” such a future scenario and evaluate the VLMs on
the basis of their known pitfalls. In this paper, we particu-
larly focus on three core research questions:
RQ1. How does a VLM behave in the presence of dif-
ferent modalities (i.e. image and text) in legal bail pre-
diction?This research question focuses on different VLMs
and their performance when it is presented with one or both
of the image and text modalities.
RQ2. Does incorporating previous case reports mirror-
ing the common law system help improve the models’
ability to make accurate and consistent decisions?Many
countries adopt the common law system, in which judges
issue their judgments based on statutes as well as prece-
dents. To address this question, we devise a RAG-based
framework that acts as a database containing previous case
reports, and VLMs base their predictions on the retrieved
relevant cases.
RQ3. How do the RAG setup and fine-tuning of VLMs
interact to influence the bail decision prediction?Since
fine-tuning is known to improve the prediction performance
of VLMs, we investigate whether vanilla or sophisticated
fine-tuning coupled with the already developed RAG can
improve the bail decision prediction performance.
Key contributions: The key contributions of our paper are
as follows.
1. We pair two datasets - (i) a dataset of mugshots of ac-
cused persons and (ii) a dataset of case reports with cor-
responding bail decisions to create input instances to au-
dit the performance of VLMs in legal judgment predic-
tion.
2. We audit the standalone VLMs to test their performance
in predicting bail decisions when the input is a pair con-
sisting of an image of an accused person and the associ-
ated case report.
3. We perform several interventions on the VLMs, which
are described in Sec. 4. We show that by applying the
correct interventions, we can bring out better perfor-mance from a VLM in a legal judgment prediction con-
text. These interventions primarily include RAG based
precedent inclusion and different fine-tuning schemes
for the VLMs.
4. Aside from the accuracy metric, we also evaluate the
chosen VLMs based on LR- and NPV , as these metrics
measure the likelihood of false negative predictions and
the trustworthiness of the models, which highly matters
in a legal context since it is considered more important to
ensure that an innocent individual is not denied bail, than
to ensure that a criminal is not granted bail. From our
comprehensive audit of multiple VLMs, we show that
all the models in their base form perform very poorly in
accuracy as well as in the LR- and NPV metrics. The
most alarming part is that, on average,these models are
highly confident in∼68%of their false negative predic-
tions. Nevertheless, suitably designed intervention tech-
niques result insteady and substantial gainsin terms of
all three metrics. This leads us to believe that if VLMs
are indeed to be used for LJP in future, effective inter-
vention techniques can be built before they are deployed
for this task.
2. Related work
In this section, we review the existing literature regarding
the innovations and contributions in LJP over the past few
years. We also show how the use of many AI tools and tech-
niques facilitate the legal process and law enforcement. At
the end of this section we also discuss recent advancements
in VLMs and their potential for future applications in legal
domain.
2.1. Text-focused LJP
The field of LJP has evolved from early machine learning
models using TF-IDF on case facts [3] to more sophisticated
deep learning approaches. A significant leap occurred with
the adoption of the transformer architectures, particularly
with the development of domain-specific pre-trained lan-
guage models like Legal-BERT [7] and InLegalBERT [31]
which demonstrate superior performance by pre-training on
large legal corpora. The current state-of-the-art is driven
by large language models (LLMs), with research shifting
toward fine-tuning both general-purpose models [17, 47]
and specialized legal LLMs like ChatLaw [9] to handle
the nuanced reasoning required for legal tasks. Nigamet
al. [27] show through their evaluation on Llama2 (70B and
13B) [40] and GPT3.5-Turbo [5] that prompting the LLMs
with the facts as well as statutes, precedents, rulings by
lower courts and arguments to mimic real-world scenarios
significantly enhances the quality of the predictions.
2

2.2. Dataset contributions in LJP
Progress in LJP has been heavily dependent on the cre-
ation of high-quality datasets. Foundational benchmarks
include the European Court of Human Rights (ECHR) cor-
pus [33] and the large-scale Chinese criminal case dataset,
CAIL [45]. For the Indian legal system, the Indian Legal
Documents Corpus (ILDC) [25] and the Hindi Legal Doc-
uments Corpus (HLDC) [18] are two of the most crucial
resources that provides a benchmark for understanding the
specifics of Indian case law, which we also leverage to build
our customised dataset. Our contribution extends this by
creating a novel evaluation setup that pairs legal text with
the Illinois DOC labeled faces dataset, enabling the study
of multimodal LJP in a way that existing text-only legal
datasets do not support. A very recent work by Nigamet
al. [28] proposes the largest and most diverse dataset of
Indian legal cases, alongside a specialised language model
fine-tuned for LJP and explainability. A very contemporary
work on NyayaRAG [29] builds the foundation by showing
that structured legal retrieval enhances both outcome accu-
racy and interpretability.
2.3. Recent applications of VLMs
VLMs have matured from establishing shared image-text
embeddings with CLIP [34] to full-fledged conversational
agents like LLaV A [21], which connect powerful vision
encoders to LLMs. The advanced models we employ –
LLaV A-NeXT [22], Qwen2.5-VL [39], Idefics3 [20], and
InternVL 3.5 [41] – represent this frontier, leveraging strong
LLM backbones like Mistral [16] and Llama3 [10]. While
these models have seen rapid adoption in high-stakes do-
mains such as medicine for analyzing radiological im-
ages [15], their application in the socio-technical and high-
stakes legal domain has only begun recently [6, 43]. Our re-
search is among the first to systematically audit these mod-
ern VLMs in the context of legal judgment prediction.
3. Models & Datasets
Our experiments are built upon a curated set of two distinct
datasets, each serving a specific role, and four state-of-the-
art VLMs in the 7-8B parameter range, selected for their
advanced capabilities in understanding both image and text
modalities.
3.1. Models
We select four powerful, open-source, instruction-tuned
VLMs that represent the current state-of-the-art.
•LLaV A-NeXT[22]: We use the 7B parameter variant,
specificallyllava-v1.6-mistral-7b-hf. It pairs
a strong CLIP-based vision encoder with theMistral-7B-
Instruct-v0.2LLM backbone.•Qwen2.5-VL[39]: We employ the
Qwen2.5-VL-7B-Instructmodel that is paired
with theQwen2.5-7B-InstructLLM.
•Idefics3[20]: We use theIdefics3-8B-Llama3
model that is built upon theLlama3.1-8B-InstructLLM
backbone, paired with the SigLip vision model.
•InternVL3.5[41]: We use theInternVL3 5-8B-HF
model that follows a “ViT-MLP-LLM” paradigm, pairing
a powerful InternViT vision encoder with the LLM from
theQwen3series.
3.2. Datasets
•Illinois DOC labeled faces dataset [12]:contains pris-
oner mugshots from the Illinois Department of Correc-
tions. Crucially for our audit, it provides associated meta-
data for each image, including sensitive Personally Iden-
tifiable Information (PII) such asracelabel (e.g. “Black”,
“White”, “Hispanic”, “Asian”etc.) andgenderlabel
(“male”, “female”). For this work, we have only chosen
images from four intersectional groups – “White Male”
(WM), “Black Male” (BM), “White Female” (WF) and
“Black Female” (BF) as Whites and African Americans
are the majority among other races. In this dataset, the
male-female ratio among the “Whites” is13370 : 1628
and among the “Blacks” is28156 : 1240. The dataset also
has information aboutoffense typesassociated with the
accused, which can be grouped into the following broad
categories –weapons violation, theft, battery, narcotics,
homicide, burglary, robbery, motor vehicle theft, intimi-
dation, stalking, criminal trespass, liquor law violation,
prostitution, human trafficking, public indecency, assault,
public peace violation.
•Legal documents corpus [18]:To obtain the case re-
ports, we utilize the development set of the HLDC [18],
containing 17K legal case documents. This corpus
is based on theExploration-Lab/IL-TURbench-
mark. The dataset contains case reports with the cor-
responding ground-truth bail decisions in Hindi. Note
that this is the only available comprehensive dataset of
bail decisions and thus for our purpose we translate the
dataset into English using IndicTrans2 [14] which is a
popular translation model between English and Indian
languages. We select only thedev allsplit (with
17,707 case reports) from the entire dataset for our use
case. Following Kapoor et al. [18] we consider only the
facts andargumentscolumn of the dataset as the
case report. Next, we perform the following preprocess-
ing steps.
1. We prompt GPT-4o [1] to give us a list of common
keywords found in criminal case reports, such as “sus-
pect”, “case number”, “arrest”etc. Following the
stopword removal procedure in NLP, we remove such
words from the dataset as they act as stopwords in a
3

Image Case facts
...The accused has no prior criminal history.< Name >has no independent< Name >...the
accused was carrying the animal in the said truck and when his truck was checked,two bun-
dles of two kilograms of charaswere recovered from it...
...Despite the occurrence of the incident, there are no independent public witnesses. They have
been in jail since the date of 29-12-2019. Around 130 kilograms of beef, bovine remains head,
hoof, legs, peacock skin and cow-slaughter tools were recovered from the spot....
...and not do any work in the name of marriage and used to talk for hours on the phone<
name >to which the applicant / accused objected and threatened that these days girls are
heard a lot,you will be framed in a false case and put in jail, false case filed...
...There isno recovery of any illegal asset from the possession of the candidatewhich is
alleged to have been recovered. It is false and fabricated...
Table 1. Combination of Illinois Images paired with case facts. Due to limited space, only a relevant portion of the case facts is shown.
legal context.
2. We tokenize each “facts and arguments” using
Mistral-7B-Instruct-v0.2[16] and remove
any cases that have a token length less than 50.
3. From thefacts andargumentscolumn we ex-
tract only the facts as having the arguments in the case
reports can introduce bias in the input to the VLM. To
this purpose, we select some keywords, such as “op-
pose”, “granted”, “rejected”etc. and remove the sen-
tences containing them. We check if the token length
becomes less than 50 and remove the corresponding
case reports from the dataset.
After this preprocessing step, the dataset size becomes
16,104. We split the data further into train (12,788 cases)
and test (3,316 cases) set in80 : 20ratio. Note that in
our RAG framework, the training samples serve as the
external knowledge base. Henceforth, we shall use the
termcase factinstead of case report as we extract only
the facts from the full case reports and use it for all our
experiments.
3.3. Pairing criminal images with case reports
For all our experiments, we pair the image dataset with the
case fact dataset since there is no benchmark dataset avail-
able with such paired information. In particular, we pair
each case fact with all the images in each intersectional
group. Thus, if there areNimages in all andMcase facts,
then the total number of pairs we have isN×M. Finally
if the case fact is from the training dataset then all the pairs
in which it participates are included into the training data;
similarly, if the case fact is from the test dataset then all the
pairs in which it participates become part of the test data.Tab. 1 shows some examples of image-case fact pairs from
our dataset.
4. Experimental setup
This section details the experimental innovations of our
work based on the three research questions. First we present
the framework for auditing VLMs in the presence of dif-
ferent modalities, such as image and text (RQ1). Next, to
address the second research question (RQ2) we evaluate
the models with a context-aware RAG setup. Finally, we
describe the comprehensive pipeline of various fine-tuning
schemes for the bail prediction task (RQ3).
4.1. Auditing VLMs for legal judgment prediction
We choose a VLM,M, to audit its prediction perfor-
mance across the intersectional group comprising gen-
der and race. Let the paired image (I) and case fact
(CTST) be denoted as[I:C TST]. As noted ear-
lier, the test set comprises all pairs in which an instance
ofCTST is a part of. We then pass this pair as in-
put toMand query it for the bail decision as follows.
M([I:C TST]) =yes|no
where yes (no) indicates thatMpredicts that bail should
(should not) be granted. These predictions are then com-
pared with the ground truth to compute various metrics dis-
cussed later in this section.
4.2. Intervention I: Precedent-aware VLMs
In many countries including India, the legal system fol-
lows theCommon Lawparadigm [13], which is grounded in
precedent and judicial decisions rather than solely codified
statutes. In this paradigm, the judges interpreting past cases
4

Figure 1. Overall fine-tuning design. Forvanillasetup typed-facts
are not added; however, inoffense type inducedsetup, the whole
architecture is in use.
that were similar to the current case, and apply the judge-
ments/principles in those prior cases to the new/current
case. Motivated by this, we investigatewhether incorpo-
rating similar past case facts as precedents improves model
predictions and, in turn, enhances fairness in bail decisions
across different intersectional groups.
To operationalize the common law paradigm in our set-
ting, we design a RAG framework [4, 42, 49]. Specif-
ically, we construct a vector store from the training set
(CTRN ) of the case facts, and make the model first re-
trieve relevant precedents and then generate responses tak-
ing into account the retrieved similar precedents. Using a
Euclidean distance–based similarity measure, we retrieve
the top three relevant case facts closest to the query case
fact and append them to the system prompt, along with
the query case fact in the user prompt, to obtain bail deci-
sion predictions from the model. This can be expressed as:
M([RAG(C TRN) :I:C TST]) =yes|no
4.3. Intervention II: Fine-tuned VLMs
To adapt VLMs for the bail prediction task, we develop a
comprehensive supervised fine-tuning (SFT) pipeline. The
primary motivation behind this step is to bridge the gap
between the generic pre-training objectives of VLMs and
the highly domain-specific nature of bail decision predic-
tion task. The fine-tuning process begins with input con-
struction, which we design in two ways: (1)vanilla: us-
ing only the case facts as the user prompt, and (2)offense
type induced: to begin with, we utilise the offense types in-
cluded in the metadata of the Illinois dataset as discussed in
Sec. 3.2. For each offense type, we obtain similar keywords
by querying GPT-4o [1]. Thus, for example, the offense
typehomicideexpands to the set of following semantically
similar keywords:{homicide, murder, manslaughter, first-
degree murder, second-degree murder,. . .}. Similarly, the
offense typetheftexpands to{theft, grand theft, shoplifting,
burglary, pickpocketing,. . .}while the offense typenar-
coticsexpands to{controlled substance, cocaine, heroin,methamphetamine, marijuana,. . .}. Now we consider each
case fact and search for the presence of keywords from each
of these expanded sets. If a case fact is found to contain one
or more keywords corresponding to an offense type, then
that offense type is associated with the case fact. Note that
more than one offense type might get associated with a case
fact in this process. We call such a type associated case
fact astyped-fact. In this way, all the case facts in all the
training and test pairs are converted to typed-facts. These
typed-facts are used to form the user prompt. Fig. 1 shows
the full fine-tuning setup. In thevanilladesign, we do not
add the typed-facts to the user prompt. In theoffense type
induceddesign, only the typed-facts are included in the user
prompt following the cases, as shown in Fig. 1.
In both setups, like earlier, the user prompt is paired with
an imageIfrom the Illinois dataset as visual evidence. For
loss computation, we frame the task as a binary decision-
making problem with a supervised textual target (yes/no),
which is the prediction output. During fine-tuning we make
the parameters of the vision tower frozen, and completely
mask the images from the input by setting the values in
the attention mask corresponding to the image tokens (i.e.
<image>) to 0. Through this approach, we want to ensure
thatthe models learn how the case facts lead to bail accep-
tance or rejection, not which person is linked with which
case fact. Since the image attentions are 0, for a given case
fact/typed-fact, while the training data containsNimages
with which it is paired up, we use one random pair out of
these, as the case fact only matters in the fine-tuning while
the pairing image does not.
Fine-tuning is performed with theSFTTrainerfrom
the TRL library, using theadamw torchoptimizer, a
learning rate of1×10−5, and an effective batch size of
8 through gradient accumulation. Model selection is guided
by validation loss minimization, where the validation set is
taken to be 10% of the train set described in Sec. 3.2. We
do a full fine-tuning on the models,i.e. updating all parame-
ters except the vision parameters, maximizing task-specific
adaptation. Let the vanilla fine-tuned model be denoted as
MVand the one based on offense type be denoted asMO.
The predictions on the test set are then obtained using the
two different models in the following ways.
•MV([I:C TST]) =yes|no
•MV([RAG(C TRN) :I:C TST]) =yes|no
•MO([RAG(CTY PE
TRN ) :I:CTY PE
TST ]) =yes|no
HereCTY PE
TRN andCTY PE
TST respectively correspond to the
typed-facts in the train and test sets.
4.4. Evaluation metrics
We evaluate the VLMs for their performance on the bail
prediction task across three metrics defined below. For each
of the metrics, the positive prediction is “bail granted” and
the negative prediction is “bail denied”.
5

•Accuracy.This metric is calculated asTP+TN
TP+FP+TN+FN
and captures a high-level perspective of how well the VLMs
perform for the task of bail prediction aggregated over all
the intersectional groups.
•Negative likelihood ratio (LR−).This is calculated as
FNR
TNR, whereFNR=FN
TP+FNandTNR=TN
TN+FP.
Here,LR−represents the likelihood of bail being denied
to individuals who are, in fact, eligible for release as per the
ground-truth.
•Negative predictive value (NPV).Defined asTN
TN+FN,
it measures the probability of bail denials being correct,
thereby indicating the trustworthiness of bail denial deci-
sions.
The choice of the metricsLR−and NPV are motivated
by the following argument – in tasks like granting of bail
in the legal domain, one should favor a decision making
process in which the false negative rate is low [48],i.e., it
is more important to ensure that a person deserving bail is
not denied of it than to ensure that a criminal is not granted
bail1.
5. Results
In this section, we discuss the results from our experiments.
Table Tab. 2 shows the results (described below) where each
metric is reported individually for the four intersectional
groups – “White Male” (WM), “Black Male” (BM), “White
Female” (WF) and “Black Female” (BF).
5.1. Results of auditing standalone VLMs (RQ1)
We present the results of auditing the standalone VLMs in
Table Tab. 2. The fourth column, labelled asM, notes the
accuracy, LR- and NPV values for this audit. For all the
VLMs, barring InternVL, the accuracy is below 50%. The
LR- values are very high for all the intersectional groups,
and the NPV values are no better than 45%. This means
that irrespective of the intersectional group,a large major-
ity of deserving individuals (as per the ground truth) are
denied bail. We further ask the VLM about its confidence
(high/medium/low) in the judgment predictions. On aver-
age across the models, out of all cases where a bail is incor-
rectly denied, in as many as∼68%cases, the VLM does
it with high confidence. This is a severely alarming trend,
making the base VLMs unsuitable for LJP.
5.2. Results of Intervention I (RQ2)
The results of the precedent-aware VLM are presented in
the fifth column of the Tab. 2 labelled asM[RAG]. For
all the VLMs, we observe a steady improvement in the ac-
curacy, while there is a remarkable increase of 16.14% for
Qwen. Overall, the best accuracy of 65.26% is obtained for
1“It is better that ten guilty persons escape than that one innocent suf-
fer” – William Blackstone.InternVL. More importantly, for all the models and across
all intersectional groups, theLR−values have declined,
and the NPV values have increased. Thus, Intervention I
has made the VLMs much more suitable for the bail predic-
tion task.
5.3. Results of Intervention II (RQ3)
We evaluate the fine-tuned VLMs in three setups – (a)MV,
i.e. using only the fine-tuned VLM without any precedents
(column 6, Tab. 2), (b)MV[RAG], retrieving relevant case
reports using a RAG and adding them into the user prompt
while querying the fine-tuned model (column 7, Tab. 2),
and (c)MO[RAG], retrieving relevant typed-facts using
the RAG and adding them in the user prompt while query-
ing the model that is also fine-tuned with typed-facts (col-
umn 8, Tab. 2). In all cases,MO[RAG]by far outper-
forms all the other intervention mechanisms in terms of ac-
curacy. For Llava-NeXT and Idefics3, the accuracy values
cross the 70% mark, with Llava-NeXT reporting as high as
75.72% accuracy. In terms of LR-, there is a drastic reduc-
tion in the case ofMO[RAG]compared to the other inter-
vention schemes for Qwen, Llava-NeXT and Idefics3. For
InternVL,MVhas the least LR- whileMO[RAG]is close
second. In terms of NPV , once againMO[RAG]achieves
the best values compared to the other intervention schemes
for Qwen, Llava-NeXT and Idefics3. For InternVL,MV
has the best NPV whileMO[RAG]is again close second.
Evidently, both our interventions are effective in making
VLMs more appropriate for the bail prediction task.
A general observation for the base models as well as for
the intervention schemes is that males experience slightly
higher false negative outputs than females,i.e.LR−val-
ues for females are marginally lower than those of males,
whereas the NPV values for females are marginally higher
than those for males. In other words, for all the setups,
males are denied bail slightly more often than females.
6. Discussion
In this work, we perform a comprehensiveanticipatory au-
ditof VLMs for the task of bail prediction (one of the criti-
cal tasks in the stack of LJP tasks). We prepare a large-scale
multimodal dataset of case facts [18] and mugshot images
of suspects [12] (across four intersectional groups), which
are then used to evaluate the VLMs under multiple settings.
6.1. Qualitative observations
The initial audit, on vanilla out-of-the-box VLMs reveals
that the models perform very poorly, with accuracy as low
as 42%. The fairness metrics, while not having any sta-
tistically significant differences between the intersectional
groups, report poor absolute values. We conclude that not
only are the models more likely to reject bail to those de-
6

Models MetricsAudit Intervention I Intervention II: Fine-tuning
M M[RAG] MVMV[RAG]MO[RAG]
QwenOverall accuracy (↑) 41.96% 58.10% 40.62% 65.92% 68.03%
LR- (↓)WM 0.97 0.68 0.96 0.53 0.47
BM 0.97 0.69 0.95 0.55 0.48
WF 0.96 0.68 0.96 0.52 0.49
BF 0.97 0.68 0.95 0.52 0.48
NPV (↑)WM 38.01% 46.70% 38.21% 52.53% 55.59%
BM 38.20% 46.37% 38.64% 52.34% 55.39%
WF 38.23% 46.86% 38.49% 53.34% 54.96%
BF 38.10% 46.81% 38.33% 53.02% 55.31%
Llava-NeXTOverall accuracy (↑) 48.08% 52.40% 72.44% 74.27% 75.72%
LR- (↓)WM 0.89 0.78 0.44 0.35 0.27
BM 0.89 0.79 0.44 0.35 0.27
WF 0.87 0.76 0.4 0.34 0.26
BF 0.86 0.76 0.4 0.33 0.26
NPV (↑)WM 40.12% 43.20% 57.45% 62.72% 68.49%
BM 40.13% 43.12% 57.85% 63.29% 68.75%
WF 40.60% 44.00% 59.97% 63.46% 69.73%
BF 41.01% 43.88% 59.80% 63.85% 69.20%
Idefics3Overall accuracy (↑) 46.79% 56.96% 56.78% 69.70% 72.74%
LR- (↓)WM 0.9 0.71 0.73 0.46 0.36
BM 0.92 0.72 0.72 0.45 0.35
WF 0.89 0.72 0.68 0.46 0.36
BF 0.9 0.73 0.7 0.44 0.32
NPV (↑)WM 39.55% 45.32% 44.87% 56.39% 62.01%
BM 39.34% 45.51% 45.41% 56.88% 63.20%
WF 40.20% 45.45% 46.89% 56.31% 62.75%
BF 39.55% 44.79% 45.69% 56.99% 64.76%
InternVLOverall accuracy (↑) 58.10% 65.26% 65.88% 68.18% 68.52%
LR- (↓)WM 0.75 0.54 0.42 0.47 0.46
BM 0.72 0.54 0.39 0.48 0.46
WF 0.7 0.54 0.35 0.44 0.43
BF 0.71 0.52 0.39 0.44 0.43
NPV (↑)WM 43.89% 52.18% 58.61% 55.61% 56.47%
BM 45.39% 52.62% 60.37% 55.70% 56.45%
WF 45.84% 52.62% 63.32% 57.47% 58.10%
BF 45.27% 53.04% 60.23% 57.35% 57.85%
Table 2. Results of the experiments. Column 4 presents the results of the audit of the VLMs. Column 5 presents the results of Intervention
I (precedent-aware VLMs), and columns 6-8 present the results of Intervention II (various types of fine-tuning). The best values for each
of the three metrics are color-coded as follows: Overall accuracy, Group-based LR-and Group-based NPV(best viewed in color).
↓: lower is better,↑: higher is better.
serving of it (high LR-), but also that these bail decisions
are not trustworthy (low NPV). Finally, a disturbing trend is
that in around 68% of cases, these models are highly con-
fident in rejecting bail to deserving candidates. Thus, us-
ing such models without domain knowledge baked into the
pipeline is highly dangerous in sensitive contexts like legalAI.
Next, we introduce two simple, but highly useful and rel-
evant, interventions into the pipeline – (i) using a precedent-
aware VLM where the precedents are brought in from a vec-
tor store of relevant case records and (ii) fine-tuned VLMs
under different fine-tuning schemes. We note an immedi-
7

Models Current case (excerpts) Precedent cases retrieved (excerpts)
notoyescorrections
MO[RAG] ... the accused objected and threatened that these
days girls are heard a lot, you will be framed in a
false case and put in jail, false case filed ....... FIR and medical are contradictory. The applicant / accused has no previous criminal history ...
... accused has been falsely implicated in the said false case by a conspiracy only to harass and humiliate him / her ...
... the plaintiff had borrowed two thousand rupees from the applicant / accused on a weekly basis. The plaintiff did not return the money and was pressurized ...
MV[RAG] ... It is not possible to kidnap any girl student from
the school from the said place ...... to have occurred at about 12 noon...<name>was alone with her father ... came to the house in the name of<name>and started to protest ...
There is an old enmity between the applicant and the plaintiff ... That is why a false case of the applicant / accused is written under pressure ...
yestonocorrections
MO[RAG] ... when his truck was checked, two bundles of
two kilograms of charas were recovered from it ...... does not have a criminal history. The applicant has been shown to have made a false recovery...
... a plastic foil was removed from the right side of the shirt pants ... The accused is stated to have recovered 500 grams of charas ...
...<name>selling ganja to the persons going to come in front of his house. When<name>was stopped, he behaved indecently with the policemen ...
MV[RAG] ... the accused is alleged to have killed<
name >... and thrown his body on the rail-
way line of Jungle Gram Kaili ...... We brought the injured< name >to the road< name >and from there went to the hospital ... Dr< name >saw that< name >was dead ...
... The family members slept in their house at night ... uncle of the plaintiff, gave information from his phone number to the plaintiff’s brother that someone had killed ...
... which he started looking for and met his friend< name >and< name >told him that his son had left home and had not yet reached home ...
Table 3. Examples of current cases and retrieved precedent cases, grouped by whether predictions correctly changed after an intervention
from rejection to granting of bail or from granting to rejection. Only small excerpts shown since case documents are very long.
ate improvement in all metrics for both interventions. First,
using only the precedents on a vanilla VLM not only im-
proves the accuracy by as much as 16%, but also has an
impact on the fairness metrics. Interestingly, the improve-
ment in fairness metrics is, again, (almost) independent of
the intersectional groups. Finally, the second intervention,
which involves a supervised fine-tuning of the VLM shows
a marked improvement on all metrics when the precedent-
aware RAG is incorporated into the pipeline. In Tab. 3, we
note some of the representative errors that get corrected for
each of the interventions. The 1stand 3rdrows show the
cases rectified after usingMO[RAG]and 2ndand 4throws
show rectifications after usingMV[RAG]. It is evident
from these examples that the retrieved cases are relevant,
which influences the predictions for the current case. We
also note that the retrieved cases after usingMO[RAG]are
very accurate and aligned with the offense type of the cur-
rent case. Such enhanced retrievals help the VLMs reach
the correct judgment, which is otherwise not possible for a
base model.
Finally, we caution here that while the models, upon in-
tervention, perform better compared to their standalone ver-
sions, the absolute accuracies are still at best 76%, and fur-
ther work is needed before they can be deployed in the real
world for sensitive legal AI tasks.
6.2. Position: Possibility of using VLMs for bail
prediction
Legal AI is a rapidly growing field with adoption rates in-
creasing throughout the world [19, 25, 37, 47]. Commonly,
AI tools are being used in the courtrooms to aid the legal
system through transcription [24], legal statute identifica-
tion [32],etc. These help not only reduce the workload and
burden of pending cases [30] but also automate and stan-
dardise a number of legal processes. It is especially useful
for countries like India, where documents are still unstruc-
tured and hand-written [31]. While such tools warrant care-
ful use, current evidence suggests they have limited direct
impact on the justice system itself. On the other hand, more
recently, there has been a rise in using AI for aiding judg-
ment decisions [6, 43]. We recognize bail prediction as ahighly sensitive application that requires continuous over-
sight (through lawmakers and independent technical audits)
as well as clear regulatory frameworks before any real-
world deployment. Our anticipatory audit represents, to our
knowledge, the first systematic evaluation of AI for bail pre-
diction that incorporates multimodal inputs, extending be-
yond prior unimodal analyses [18] and, thereby, simulating
more realistic aspects of courtroom use. Given the scarcity
of multimodal datasets for this domain, we curate our own
dataset by combining Illinois DOC face images [12] with
textual case facts from the HLDC dataset [18]. Our find-
ings suggest that such models, in their current form, are not
suitable for deployment in real-world settings. Specifically,
we observe that models are more likely to deny bail in cases
where experienced judges have granted it, underscoring the
importance of human-in-the-loop frameworks not only in
deployment but also in system design. These observations
highlight the need for carefully designed regulations, inde-
pendently verifiable oversight mechanisms, and a cautious
approach to the development and use of AI in sensitive do-
mains such as bail prediction. Having said that, we also
firmly believe that these models with proper interventions
in place can be used as very efficient and effectiveassistive
toolsacross courtrooms.
7. Conclusion
Through this investigation, we aimed to shed light on the
potentials and limitations of VLMs in real-world legal set-
tings. Our findings have implications not only for the tech-
nical development of legal AI systems but also for the
ethical and policy frameworks surrounding their deploy-
ment. Through detailed intervention and evaluation setups
we show thatby applying the correct interventions, we can
bring out better performance from a VLM in a legal judg-
ment prediction context. Our work emphasizes the neces-
sity of precedents in legal judgment prediction where AI
models do not see any other evidences besides the provided
case facts, restating the importance of this method imple-
mented by many other contemporary works. We note that
current VLMs along the suitable interventions can at best
8

act as assistive tools in the courtroom and the final human
emotive-cognitive delivery of justice is indispensable.
Though the inclusion of multiple modalities, in terms of
image, audio, video, text, is very imminent and practical
in the legal context, we should be extra-aware before
deploying such multimodal models. By introducing new
modalities, the AI models become harder to interpret and
easier to propagate and multiply the existing errors. While
our multitude of intervention methods show a thorough
improvement of the model behavior we believe that more
aggressive research is needed to develop stronger interven-
tions in the future before actual real-world deployment.
References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-
mad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,
Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
Gpt-4 technical report.arXiv preprint arXiv:2303.08774,
2023. 3, 5
[2] Mousumi Akter, Erion C ¸ ano, Erik Weber, Dennis Dobler,
and Ivan Habernal. A comprehensive survey on legal sum-
marization: Challenges and future directions.arXiv preprint
arXiv:2501.17830, 2025. 1
[3] Nikolaos Aletras, Dimitrios Tsarapatsanis, Daniel Preot ¸iuc-
Pietro, and Vasileios Lampos. Predicting judicial deci-
sions of the european court of human rights: A natural lan-
guage processing perspective.PeerJ computer science, 2:
e93, 2016. 2
[4] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Han-
naneh Hajishirzi. Self-rag: Self-reflective retrieval aug-
mented generation. InNeurIPS 2023 workshop on instruc-
tion tuning and instruction following, 2023. 5
[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Sub-
biah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakan-
tan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Lan-
guage models are few-shot learners.Advances in neural in-
formation processing systems, 33:1877–1901, 2020. 2
[6] Sagar Chakraborty, Gaurav Harit, and Saptarshi Ghosh. How
well do mllms understand handwritten legal documents? a
novel dataset for benchmarking.International Journal on
Document Analysis and Recognition (IJDAR), pages 1–17,
2025. 1, 3, 8
[7] Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasio-
tis, Nikolaos Aletras, and Ion Androutsopoulos. Legal-bert:
The muppets straight out of law school.arXiv preprint
arXiv:2010.02559, 2020. 2
[8] F. Contini, A. Minissale, and S. Bergman Blix. Artificial
intelligence and real decisions: predictive systems and gen-
erative ai vs. emotive-cognitive legal deliberations.Frontiers
in Sociology, 2024. 1
[9] Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan.
Chatlaw: Open-source legal large language model with inte-
grated external knowledge bases.CoRR, 2023. 2
[10] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. Thellama 3 herd of models.arXiv e-prints, pages arXiv–2407,
2024. 3
[11] S. Farber. Ai as a decision support tool in forensic image
analysis: A pilot study on integrating large language models
into crime scene investigation workflows.Journal of Foren-
sic Sciences, 70(3):932–943, 2025. 1
[12] David J. Fisher. Illinois doc labeled faces dataset. Kaggle.
3, 6, 8
[13] Sandra Fullerton Joireman. The evolution of the common
law: Legal development in kenya and india.Commonwealth
& Comparative Politics, 44(2):190–210, 2006. 4
[14] Jay Gala, Pranjal A Chitale, Raghavan AK, Varun Gumma,
Sumanth Doddapaneni, Aswanth Kumar, Janki Nawale,
Anupama Sujatha, Ratish Puduppully, Vivek Raghavan, et al.
Indictrans2: Towards high-quality and accessible machine
translation models for all 22 scheduled indian languages.
arXiv preprint arXiv:2305.16307, 2023. 3
[15] Iryna Hartsock and Ghulam Rasool. Vision-language mod-
els for medical report generation and visual question answer-
ing: A review.Frontiers in artificial intelligence, 7:1430984,
2024. 1, 3
[16] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch,
Chris Bamford, Devendra Singh Chaplot, Diego de las
Casas, Florian Bressand, Gianna Lengyel, Guillaume Lam-
ple, Lucile Saulnier, L ´elio Renard Lavaud, Marie-Anne
Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,
Thomas Wang, Timoth ´ee Lacroix, and William El Sayed.
Mistral 7b, 2023. 3, 4
[17] Cong Jiang and Xiaolei Yang. Legal syllogism prompting:
Teaching large language models for legal judgment predic-
tion. InProceedings of the nineteenth international confer-
ence on artificial intelligence and law, pages 417–421, 2023.
1, 2
[18] Arnav Kapoor, Mudit Dhawan, Anmol Goel, TH Arjun,
Akshala Bhatnagar, Vibhu Agrawal, Amul Agrawal, Arnab
Bhattacharya, Ponnurangam Kumaraguru, and Ashutosh
Modi. Hldc: Hindi legal documents corpus.arXiv preprint
arXiv:2204.00806, 2022. 3, 6, 8
[19] Daniel Martin Katz, Michael J Bommarito II, and Josh
Blackman. A general approach for predicting the behavior
of the supreme court of the united states.PloS one, 12(4):
e0174698, 2017. 1, 8
[20] Hugo Laurenc ¸on, Andr ´es Marafioti, Victor Sanh, and L ´eo
Tronchon. Building and better understanding vision-
language models: insights and future directions., 2024. 1,
3
[21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning.Advances in neural information
processing systems, 36:34892–34916, 2023. 3
[22] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved
reasoning, ocr, and world knowledge, 2024. 1, 3
[23] Yougang Lyu, Jitai Hao, Zihan Wang, Kai Zhao, Shen Gao,
Pengjie Ren, Zhumin Chen, Fang Wang, and Zhaochun Ren.
Multi-defendant legal judgment prediction via hierarchical
reasoning.arXiv preprint arXiv:2312.05762, 2023. 1
[24] Sayan Mahapatra, Debtanu Datta, Shubham Soni, Adrijit
Goswami, and Saptarshi Ghosh. Milpac: A novel benchmark
9

for evaluating translation of legal text to indian languages.
arXiv preprint arXiv:2310.09765, 2023. 8
[25] Vijit Malik, Rishabh Sanjay, Shubham Kumar Nigam, Kripa
Ghosh, Shouvik Kumar Guha, Arnab Bhattacharya, and
Ashutosh Modi. Ildc for cjpe: Indian legal documents cor-
pus for court judgment prediction and explanation.arXiv
preprint arXiv:2105.13562, 2021. 1, 3, 8
[26] Alexis Morin-Martel. Machine learning in bail decisions
and judges’ trustworthiness.Ai & Society, 39(4):2033–2044,
2024. 1
[27] Shubham Kumar Nigam, Aniket Deroy, Subhankar Maity,
and Arnab Bhattacharya. Rethinking legal judgement pre-
diction in a realistic scenario in the era of large language
models.arXiv preprint arXiv:2410.10542, 2024. 2
[28] Shubham Kumar Nigam, Balaramamahanthi Deepak Pat-
naik, Shivam Mishra, Noel Shallum, Kripabandhu Ghosh,
and Arnab Bhattacharya. Nyayaanumana & inlegalllama:
the largest indian legal judgment prediction dataset and spe-
cialized language model for enhanced decision analysis.
arXiv preprint arXiv:2412.08385, 2024. 3
[29] Shubham Kumar Nigam, Balaramamahanthi Deepak Pat-
naik, Shivam Mishra, Ajay Varghese Thomas, Noel Shallum,
Kripabandhu Ghosh, and Arnab Bhattacharya. Nyayarag:
Realistic legal judgment prediction with rag under the in-
dian common law system.arXiv preprint arXiv:2508.00709,
2025. 3
[30] District Court of India. National judicial data grid. 8
[31] Shounak Paul, Arpan Mandal, Pawan Goyal, and Saptarshi
Ghosh. Pre-trained language models for the legal domain:
A case study on indian law. InProceedings of 19th Interna-
tional Conference on Artificial Intelligence and Law - ICAIL
2023, 2023. 2, 8
[32] Shounak Paul, Rajas Bhatt, Pawan Goyal, and Saptarshi
Ghosh. Legal statute identification: A case study using state-
of-the-art datasets and methods. InProceedings of the 47th
International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, pages 2231–2240, 2024.
8
[33] Prakash Poudyal, Jarom ´ırˇSavelka, Aagje Ieven,
Marie Francine Moens, Teresa Goncalves, and Paulo
Quaresma. Echr: Legal corpus for argument mining. In
Proceedings of the 7th Workshop on Argument Mining,
pages 67–75, 2020. 3
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. InInternational conference on machine learning, pages
8748–8763. PmLR, 2021. 3
[35] Gabriele Ruggeri, Debora Nozza, et al. A multi-dimensional
study on bias in vision-language models. InFindings of the
Association for Computational Linguistics: ACL 2023. As-
sociation for Computational Linguistics, 2023. 1
[36] Carlo Sansone and Giancarlo Sperl ´ı. Legal information re-
trieval systems: state-of-the-art and open issues.Information
Systems, 106:101967, 2022. 1[37] Ehsan Shareghi, Jiuzhou Han, and Paul Burgess. Methods
for legal citation prediction in the age of llms: An australian
law case study.arXiv e-prints, pages arXiv–2412, 2024. 1, 8
[38] Markos Stamatakis, Joshua Berger, Christian Wartena, Ralph
Ewerth, and Anett Hoppe. Enhancing the learning experi-
ence: Using vision-language models to generate questions
for educational videos. InInternational Conference on Ar-
tificial Intelligence in Education, pages 305–319. Springer,
2025. 1
[39] Qwen Team. Qwen2.5-vl, 2025. 1, 3
[40] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert,
Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov,
Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.
Llama 2: Open foundation and fine-tuned chat models.arXiv
preprint arXiv:2307.09288, 2023. 2
[41] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long
Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Sheng-
long Ye, Jie Shao, et al. Internvl3.5: Advancing open-source
multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265, 2025. 1, 3
[42] Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven
Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang,
Anush Mattapalli, Ankur Taly, Jingbo Shang, et al. Specula-
tive rag: Enhancing retrieval augmented generation through
drafting.arXiv preprint arXiv:2407.08223, 2024. 5
[43] Hannes Westermann and Jaromir Savelka. Analyzing images
of legal documents: Toward multi-modal llms for access to
justice.arXiv preprint arXiv:2412.15260, 2024. 1, 3, 8
[44] Yiquan Wu, Siying Zhou, Yifei Liu, Weiming Lu, Xi-
aozhong Liu, Yating Zhang, Changlong Sun, Fei Wu, and
Kun Kuang. Precedent-enhanced legal judgment prediction
with llm and domain-model collaboration.arXiv preprint
arXiv:2310.09241, 2023. 1
[45] Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu,
Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han,
Zhen Hu, Heng Wang, et al. Cail2018: A large-scale
legal dataset for judgment prediction.arXiv preprint
arXiv:1807.02478, 2018. 3
[46] Yisong Xiao, Aishan Liu, QianJia Cheng, Zhenfei Yin,
Siyuan Liang, Jiapeng Li, Jing Shao, Xianglong Liu, and
Dacheng Tao. Genderbias-\emph{VL}: Benchmarking
gender bias in vision language models via counterfactual
probing.arXiv preprint arXiv:2407.00600, 2024. 1
[47] Lufeng Yuan, Jun Wang, Shifeng Fan, Yingying Bian, Bin-
ming Yang, Yueyue Wang, and Xiaobin Wang. Auto-
matic legal judgment prediction via large amounts of crim-
inal cases. In2019 IEEE 5th International Conference on
Computer and Communications (ICCC), pages 2087–2091.
IEEE, 2019. 1, 2, 8
[48] Muhammad Bilal Zafar, Isabel Valera, Manuel Gomez Ro-
driguez, and Krishna P. Gummadi. Fairness beyond disparate
treatment & disparate impact: Learning classification with-
out disparate mistreatment. InProceedings of the 26th Inter-
national Conference on World Wide Web, page 1171–1180,
Republic and Canton of Geneva, CHE, 2017. International
World Wide Web Conferences Steering Committee. 6
[49] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang,
Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao Zhang,
10

Jie Jiang, and Bin Cui. Retrieval-augmented genera-
tion for ai-generated content: A survey.arXiv preprint
arXiv:2402.19473, 2024. 5
11