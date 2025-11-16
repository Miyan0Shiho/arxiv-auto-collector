# Introducing A Bangla Sentence - Gloss Pair Dataset for Bangla Sign Language Translation and Research

**Authors**: Neelavro Saha, Rafi Shahriyar, Nafis Ashraf Roudra, Saadman Sakib, Annajiat Alim Rasel

**Published**: 2025-11-11 17:41:12

**PDF URL**: [https://arxiv.org/pdf/2511.08507v1](https://arxiv.org/pdf/2511.08507v1)

## Abstract
Bangla Sign Language (BdSL) translation represents a low-resource NLP task due to the lack of large-scale datasets that address sentence-level translation. Correspondingly, existing research in this field has been limited to word and alphabet level detection. In this work, we introduce Bangla-SGP, a novel parallel dataset consisting of 1,000 human-annotated sentence-gloss pairs which was augmented with around 3,000 synthetically generated pairs using syntactic and morphological rules through a rule-based Retrieval-Augmented Generation (RAG) pipeline. The gloss sequences of the spoken Bangla sentences are made up of individual glosses which are Bangla sign supported words and serve as an intermediate representation for a continuous sign. Our dataset consists of 1000 high quality Bangla sentences that are manually annotated into a gloss sequence by a professional signer. The augmentation process incorporates rule-based linguistic strategies and prompt engineering techniques that we have adopted by critically analyzing our human annotated sentence-gloss pairs and by working closely with our professional signer. Furthermore, we fine-tune several transformer-based models such as mBart50, Google mT5, GPT4.1-nano and evaluate their sentence-to-gloss translation performance using BLEU scores, based on these evaluation metrics we compare the model's gloss-translation consistency across our dataset and the RWTH-PHOENIX-2014T benchmark.

## Full Text


<!-- PDF content starts -->

Introducing A Bangla Sentence – Gloss Pair Dataset for Bangla
Sign Language Translation and Research
Neelavro Saha1, Rafi Shahriyar1, Nafis Ashraf Roudra1,
Saadman Sakib1, Annajiat Alim Rasel1
1Department of Computer Science and Engineering, BRAC University, Bangladesh
neelavro.saha@g.bracu.ac.bd, rafi.shahriyar@g.bracu.ac.bd, nafis.ashraf@g.bracu.ac.bd,
saadman.sakib1@g.bracu.ac.bd, annajiat@bracu.ac.bd
Abstract
Bangla Sign Language (BdSL) translation represents a low-resource NLP task due to the lack of large-scale datasets
that address sentence-level translation. Correspondingly, existing research in this field has been limited to word
and alphabet level detection. In this work, we introduce Bangla-SGP, a novel parallel dataset consisting of 1,000
human-annotated sentence–gloss pairs which was augmented with around 3,000 synthetically generated pairs using
syntactic and morphological rules through a rule-based Retrieval-Augmented Generation (RAG) pipeline. The gloss
sequences of the spoken Bangla sentences are made up of individual glosses which are Bangla sign supported
words and serve as an intermediate representation for a continuous sign. Our dataset consists of 1000 high quality
Bangla sentences that are manually annotated into a gloss sequence by a professional signer. The augmentation
process incorporates rule-based linguistic strategies and prompt engineering techniques that we have adopted by
critically analyzing our human annotated sentence-gloss pairs and by working closely with our professional signer.
Furthermore, we fine-tune several transformer-based models such as mBart50, Google mT5, GPT4.1-nano and
evaluate their sentence-to-gloss translation performance using BLEU scores, based on these evaluation metrics
wecomparethemodel’sgloss-translationconsistencyacrossourdatasetandtheRWTH-PHOENIX-2014Tbenchmark.
Keywords:NLP, Transformers, Bangla Sign Language, BdSL, Gloss Translation, Data Augmentation, Mor-
phological Transformation, RAG, Low-resource Languages.
1. Introduction
DevelopmentsinNaturalLanguageProcessinghas
made multiple spoken languages more accessible
through machine translation. Following the intro-
ductionofTransformerbasedarchitectures,models
can be trained to exhibit enhanced multilingual un-
derstanding, allowing information to be distributed
despite linguistic barriers (Conneau et al., 2019;
Liu et al., 2020). However, even in today’s world,
there still remains a huge accessibility challenge
for the deaf community - particularly when it comes
to reading and understanding text or spoken lan-
guage. Deaf people prefer sign language as a pri-
mary communication method. However, this mode
of communication is not incorporated in much of
the content of the modern day, which are mostly
voice or text-based. Research shows that when
people read, they associate words with their corre-
sponding pronunciations in order to comprehend
the text (Trezek and Mayer, 2019). Deaf people,
unable to hear, lack this phonological skill as they
have no idea what the words sound like. As such,
reading text is challenging for the deaf population.
To tackle these issues, ongoing research include
creating pipelines of different deep-learning and
transformer based models to convert spoken text
to its corresponding sign language translation in an
animated form (Moryossef et al., 2023; Zuo et al.,2024). These pipelines require extensive datasets
containing spoken sentences, their corresponding
videos,andglosssequencesthatserveasatypeof
structuredannotationsthatrepresentsignlanguage
in textual form. While text-to-gloss translation has
been explored in various languages globally, lit-
tle work has been done in the context of Bangla.
In Bangladesh, there are approximately 13 million
people with hearing difficulties, 3 million of whom
are deaf (Alauddin and Joarder, 2004). Many of
these people use Bangla sign language (BdSL) as
the primary method of communication. Therefore,
in this paper we present the following contributions:
Firstly, we introduce Bangla-SGP, the first phase
of our BdSL dataset, which could serve as a foun-
dational step for facilitating research in Continuous
Bangla Sign Language Recognition and transla-
tion, a field that remains vastly unexplored due to
a severe lack of publicly available resources. Sec-
ondly,weproposeanovelrule-basedRAGpipeline
for Bangla sentence-gloss pair generation, which
helps address the low-resource challenge in BdSL
by partially automating the gloss generation pro-
cess and reducing reliance on extensive manual
annotation. Finally,ourgeneratedglosssequences
can also be used as an intermediary step toward
3D animated representation of sign language sen-
tences, enabling real-world applications such as
education tools and virtual interpreters to supportarXiv:2511.08507v1  [cs.CL]  11 Nov 2025

the deaf community.
2. Related Works
2.1. Bangladeshi Sign Language (BdSL)
BdSL60, an extensive and one of the very few
word-level BdSL dataset consisting of 60 anno-
tated Bangla sign words and 9307 video trials
from 18 signers (Rubaiyeat et al., 2024). Vari-
ous preprocessing methods were applied to the
dataset. A PROLONGED version of the dataset
was created by duplicating frames to maintain uni-
formity for machine learning models. A FLIPPED
version was also created by transforming all left-
handed sign instances to right-handed. A Rel-
ative Quantization (RQ) dataset was also cre-
ated by quantizing the landmark points to re-
duce depth variations. Overall, the dataset
variations boil down to: i) PROLONGED and
FLIPPED, ii) PROLONGED and NON-FLIPPED,
iii) NON-PROLONGED and FLIPPED, iii) NON-
PROLONGED and NON-FLIPPED and finally v)
RQ. Mediapipe Hollistic was used to extract pose
keypoints and traditional SVM, SVM-DTW, and
Attention-based Bi-LSTM were used to benchmark
the dataset. Classical SVM provided a testing ac-
curacy of about 67.5%, while SVM-DTW scored
around 65.8%. On the other hand, attention-based
biLSTM had a testing accuracy of up to 75.1%.
Sams et al. (2023) introduced SignBD-Word a
word-level dataset in video format for Bangla sign
language. The dataset contains 6,000 video clips
of 200 distinct Bangla sign language words with
each sign recorded from both full-body and upper-
body perspectives. Additionally, annotations and
glossesareprovidedforeachvideo. Collectedwith
the help of 16 different signers, SignBD-word is a
datasetthatservesasabaselineforfutureresearch
in BdSL recognition and pose synthesis. To stan-
dardize the dimensions of all sign language video
clips, individual videos were sampled at 30 frames
and at consistent intervals. Individual frames were
then re-scaled to 224×224 pixels using bi-linear in-
terpolation. Theyalsocompare2DCNNwithLSTM
and 3D CNN approaches to determine which archi-
tectures best capture the spatio-temporal dynam-
ics of sign gestures. 3D CNN models such as I3D
and SlowFast outperform 2D CNN models signif-
icantly. Subsequently the authors explored GAN-
based methods for synthesizing sign languages
videos from the 2D body-pose skeletons. Three
models were used they are as follows, CycleGan,
pix2pixHDandSPADE.CycleGANmodelproduced
blurred and noisy results correspondingly SPADE
also struggled due to its reliance on semantic label
maps. Therefore, pix2pixHD stood out as the best
as outputs consisted of clear hand and face details.Realismofthesynthesizedvideosweredetermined
through a Mean Opinion Score (MOS) test. Here
20 university students rated the generated videos.
Out of the 3 models pix2pixHD was rated the high-
est. The primary limitations of this study includes
the limited number of signers, the restricted set of
sign words, and the need for further fine-tuning of
the GAN models for better performance.
2.2. Sign Language Datasets
Joze and Koller (2019) introduced a large dataset
for American Sign Language (ASL) recognition for
training robust deep learning models . Sign lan-
guage recognition is a challenging task due to its
multimodal nature which involves continuous hand
movement,differentbodyorientations,handshapes
and facial expressions. As such, a large scale
dataset is required to enable deep learning tech-
niques for sign language recognition. This was the
motivation behind this paper as it tries to bridge
the gap between ASL recognition and current com-
puter vision advancements. The dataset, MS-ASL,
hasover25thousandvideosofapproximately1000
ASL signs. Moreover, it has 222 distinct signers,
and the videos were shot in an uncontrolled envi-
ronment, under real-life conditions with variations
in lighting and background. These videos were col-
lected from public sources like educational ASL
videos from YouTube. Subsequently, the signs
wereprocessedtosegmentthesignsintoindividual
samples. OCR was used to extract text from the
videos and the subtitles were used for metadata.
Then it was reviewed manually to ensure quality,
particularlybycroppingthevideosthataretoolong.
Also, synonymswithASLvocabularywerehandled
manually. Afterwards the dataset was divided into
train, test and validation sets containing videos of
165, 20 and 37 signers respectively, and state of
the art models like I3D, 2D-CNN-LSTM and body
key-point recognition were used to benchmark the
dataset. Among these models, I3D had the best re-
sultsasitshowedanaccuracyof57.69%,whilethe
other models had much lower accuracy. I3D also
had the best results in the top-five accuracy results,
as it scored 81.08% in that metric. While this large
scale dataset offers significant advancements in
sign language recognition, it has some limitations.
There were not sufficient samples for some of the
signs which resulted in these classes having lower
performance. Additionally, some non-native an-
notators were involved in the manual annotation
process which may introduce some errors. Fur-
thermore, the dataset does not address regional
variations in ASL, which may affect the generaliz-
ability of the models.

3. Methodology
3.1. Dataset Creation
Firstly, we created a foundation dataset of 1000
Bangla sentence-gloss pairs. As shown in Fig-
ure 1, these sentences were collected from a va-
riety of sources, including Bangla newspaper cor-
pora, bangla literature, BdSL language develop-
ment book etc. The sentences were annotated to
their corresponding gloss by Mr. Ariful Islam who
is an experienced Bangla sign language presenter
at BTV.
Figure 1: Dataset Creation
3.2. Data Augmentation
Figure 2: Data Augmentation
Data augmentation plays a crucial role in ad-
dressing data scarcity, especially in low-resource
language tasks where annotated data is limited.
Figure 2 illustrates the workflow of our data aug-
mentation process. Specifically in our case of
Bangla Text-to-BdSL gloss translation, we curated
a dataset of 1000 sentence-gloss pairs annotated
byaprofessionalsigner. Whilethisdatasetislarger
than other similar works in the field of Bangla text
to gloss translation, it still remains insufficient in
achieving satisfactory results in model training. As
such, we applied several data augmentation strate-
giestoenhancethedatasetwhichinclude: (1)Rule-
based Morphological Transformation, (2) Masked-
TokenSubstitution,(3)Retrieval-AugmentedGener-
ation (RAG). With these methods, we generated a
total of 3000 new entries of text-gloss translations.
Our final dataset now has 4000 entries with 1:3
ratio for original to augmentation.
Figure 3: Rule Set Collection
3.2.1. Rule-based Morphological
Transformation
We collaborated closely with a professional signer
toanalyzeandextractcommontranslationpatterns
from our annotated dataset (Figure 3). A signifi-
cantportionofthesepatternswererelatedtotense.
Each tense follows consistent morphological rules
that map the verb of the text form into gloss. For
example, a common pattern in the future tense is
that the verb is translated to its root form followed
by the equivalent of “will be.” Based on such con-
sistent mappings, we, along with the help of the
professional signer, compiled a set of rules. After-
wards, we applied these rules to augment existing
sentences by systematically converting them into
other tenses.
An example of this transformation process is il-
lustrated in Figure 4. This method ensured that the
augmented sentence-gloss pairs remained linguis-
tically accurate and semantically faithful to the orig-
inal meaning. Furthermore, because the glosses
were generated using verified rules rather than ar-
bitrary transformations, the resulting data main-
tained a high level of consistency. With this, we
manually generated around 500 text-gloss pairs
and expanded the dataset in a controlled and in-
terpretable manner, which is particularly important
in low-resource tasks such as Bangla text-to-gloss
translation.
Figure 4: Example of verb tense transformations
used in rule-based morphological augmentation.
3.2.2. Masked-Token Substitution
The second method we applied is a form of contex-
tual augmentation known as Masked Token Sub-

Figure 5: RAG Based Augmentation
stitution. This technique creates lexical variants of
sentence-gloss pairs using known rules and the
Bangla-BERT model. We defined templates where
certain words (typically common nouns or verbs)
were masked. Bangla-BERT then predicted the
most contextually appropriate replacements for the
masked tokens. The newly generated sentences
preserved the grammatical structure and meaning
of the originals.
An illustrative example of this approach is shown
in Figure 6. By generating diverse sentence forms
throughthismethod,weaimedtoexposethemodel
to a broader range of linguistic patterns, improving
its generalization capability while maintaining the
rulesoftext-to-glosstranslation. Approximately500
text-gloss pairs were created using this approach.
Figure 6: Example of masked token substitution
where Bangla-BERT generates contextually valid
replacements for masked tokens, producing di-
verse yet grammatically consistent sentence vari-
ants.3.2.3.Retrieval-Augmented Generation (RAG)
based Few-Shot Prompting
Our third approach utilized a Retrieval-Augmented
Generation (RAG) pipeline for data augmentation.
We split the 1000 professionally annotated text-
gloss samples into 80%, 10%, and 10% for train-
ing, testing, and validation respectively. The train-
ing set was embedded using LaBSE and stored in
Pinecone, a vector database. This training data
servedasthegroundtruthreferenceforgenerating
additional synthetic samples. For retrieving supple-
mentary Bangla text samples, we used the publicly
available Bangla Corpus by Nuhash Afnan, which
contains sentences collected from newspapers on
varied topics. We selected this corpus because the
formalwritingstyleofnewspapercloselyalignswith
the professionally annotated training data used in
our experiments. Figure 5 illustrates the workflow
ofourRAGbasedaugmentationprocess. Weused
GPT-4.1-nanoforfew-shotpromptingandacustom
two-stage-prompting for gloss generation.
Retrieval-Augmented Generation (RAG), paired
with few-shot prompting, is particularly effective
for data augmentation in low-resource settings be-
cause it grounds a model’s generation by providing
contextually similar examples to learn from. Partic-
ularly in our case of Bangla text-to-gloss transla-
tion,itenablesthemodeltoproducemoreaccurate
translationsevenforsomerareorlessfrequentsen-
tence structures that are not covered by the com-
mon rule sets we compiled. For example, some
words do not have direct translations in BDSL. For

instance, the word “recipe” does not have a corre-
sponding sign and is instead translated to (khabar
ranna niyom) ’food cooking rules’. Without giving
such context during data augmentation using GPT,
these patterns may not show up in the output. As
such, weimplementedaRAGbasedapproachthat
first retrieves the similar occurrences from our pro-
fessionally annotated dataset. This way, if an input
sentence contains a word like “recipe”, the system
is likely to retrieve examples containing similar or
same terms. These retrieved examples, along with
their corresponding glosses, are then included in a
prompt passed as a few shot prompt. The model,
having access to these grounded examples, is bet-
ter guided to produce an accurate and contextually
appropriate gloss for the new sentence.
We set a similarity score threshold of 0.5, mean-
ing that if the score is above 0.5, that is considered
a similar sentence usable for few shot prompt. To
prevent overload, we also cap the number of re-
trieved examples at 20 per prompt. Now the prob-
lem arises if less or no similar sentences are found.
To address this, we implement a fallback mecha-
nismwhere,insteadofsendingnsimilarsentences,
it sends the rules in the prompt. We further im-
proved this approach by implementing a two-stage
prompting strategy. Instead of sending all the rules
at once, which may overwhelm the model due to
the complexity and volume of the input, we first
issueaprompttoidentifythetenseofthesentence.
Then,basedonthetense,weonlysendtherulesof
that particular tense along with the examples. This
way the prompt is more focused and likely to gener-
ate better results. This two-stage prompting strat-
egy draws on principles similar to chain-of-thought
prompting, where an intermediate reasoning step
(tense identification in our case) is used to refine
the final generation prompt. We generated 2000
text-gloss pairs using this approach, and applied
the Cohen’s kappa method to validate our results.
3.3. Validation using Cohen’s kappa
Cohen’s kappa is a widely used statistical measure
for evaluating how consistently different annota-
tors label the same data. It is particularly useful
in validating data augmentation techniques in low-
resourcesettingswheregroundtruthdataislimited.
Inourcase,wefirstgenerated1000text-glosspairs
using our RAG based approach. Then, we ran-
domlyselected15%(150samples)ofthetext-gloss
pairs and asked two professional BDSL signers to
independentlyreviewandvalidatetheglosstransla-
tions. The signers who validated these translations
are Mr. Ariful Islam and Mrs. Tanjila Tartushi, both
of whom are professional sign presenters at BTV.
They reviewed the samples independently and did
not consult each other. They were given two input
fields to validate each entry. The first field was aYes/Noquestionaskingiftheythinktheglossisun-
derstandable. The second field asked them to rate
the gloss on a scale of 1–5, where 1 indicated the
least understandable and most inaccurate transla-
tion, and 5 indicated most accurate translation. We
then calculated the Cohen’s kappa score between
the two signers’ judgments.
Metric Signer 1 Signer 2 Combined
Validation Rate (%) 74.7 76.0 75.3
Average Quality Score 2.96 3.41 3.19
Quality Distribution
High Quality (Score≥4) 35.3% 50.0% 42.7%
Acceptable (Score = 3) 26.0% 22.7% 24.3%
Low Quality (Score≤2) 38.7% 27.3% 33.0%
Inter-rater Reliability (Cohen’s Kappa)
Binary Agreementκ= 0.7489(Substantial)
Quality Agreementκ= 0.3496(Fair)
Table 1: Key evaluation metrics for RAG-based
data augmentation.
Table 1 shows the key evaluation metrics for the
effectiveness of our RAG-based data augmenta-
tionmethod. Itshowsthatthevalidationrate,which
measuresthepercentageofglossesmarkedascor-
rect by professional signers, was 74.7% for Signer
1 and 76.0% for Signer 2 and the average vali-
dation rate is 75.3%. The average quality score,
based on a 1–5 scale, was 2.96 from Signer 1 and
3.41 from Signer 2 and an average of 3.19 out of 5.
The quality distribution section breaks down how
many glosses were rated as high quality, accept-
able, or low quality (score ≤2). Overall, 42.7%
of glosses were rated as high quality and 24.3%
were acceptable, indicating that most glosses met
practical quality standards. Lastly, the reliability
is measured using Cohen’s kappa, which shows
substantial agreement ( κ= 0.7489) for binary vali-
dation and fair agreement ( κ= 0.3496) for quality
ratings. This supports our conclusion that the eval-
uations were consistent and trustworthy. As such,
wecanconfidentlyusetheaugmenteddataasare-
liable resource for Bangla text-to-gloss translation
tasks.
4. Experiments
This following work investigates the leveraging of
four separate transformer-based models, mBART-
50, NLLB-200, mT5 as well as GPT-4.1-nano
for translating Bengali-to-Bangla Sign Language
glosses, centered around a low-resource task.
These models all consist of an encoder-decoder ar-
chitectureandarepretrainedwithmultiplelanguage
corpora, ensuring their suitability in low-resource
data cases. The original dataset of 1,000 expert-
annotated sentence-gloss pairs was split into train-
ing, validation, and test sets in the proportions
of 80%, 10%, and 10%, respectively. The same

test set was used when evaluating the augmented
dataset to ensure a fair comparison.
4.1. Model Selection
The first model we have chosen is mT5-small (Xue
et al., 2021) of Google which is another sequence-
to-sequence Transformer model that is trained on
an extensive dataset called mC4 that covers 101
languages,includingBanglaandhasapproximately
300M. It follows the T5 architecture (Raffel et al.,
2019), featuring an encoder–decoder structure of
typicalTransformermodels. Thisarchitecturegives
mT5 its “Text-to-Text Transfer Transformer” desig-
nation, meaning it converts any NLP task into a
text-to-text format — taking text as input and gener-
ating text as output. This allows for a multitude of
tasks to be performed by the same model with the
same hyperparameters - sentence-to-gloss trans-
lation being one that mT5 excels at. Furthermore,
mT5 is already pretrained on the Bengali language
and is thus knowledgeable on Bengali sentence
structures and nuances. So this knowledge can
be leveraged to better aid a low-resource task like
sign language sentence to gloss conversion. Addi-
tionally the smaller parameter size enables training
even with our limited GPU resources. Taking into
account all these factors, along with its solid perfor-
mance when finetuned for downstream tasks, mT5
inherently feels like a great option for our specific
task.
The next selected model is mBART-50 which
is an extension of BART (Bidirectional and Auto-
Regressive Transformer). It is a multilingual de-
noisingsequence-to-sequencemodeltrainedondi-
versemonolingualcorporaacrossmanylanguages,
including low-resource ones such as Bangla, and
has a parameter count of approximately 610M (Liu
et al., 2020). Due to its multilingual pretraining,
mBART-50 effectively transfers knowledge from
high-resource to low-resource languages, which
results in improved translation quality whether it be
supervised or unsupervised. As such, it is particu-
larly suitable for Bangla text-to-gloss translation.
Another model that was useful for our task was
NLLB-200-1.3B. Costa-Jussà et al. (2022) ven-
tured heavily upon this robust multilingual trans-
lation model which is capable of handling over 200
languages and thousands of language pairs, in-
cluding many low-resource ones. It uses a Mixture-
of-Experts (MoE) architecture for adaptable mul-
tilingual translation. Though its large size risks
overfitting on limited data, Parameter-Efficient Fine-
Tuning (PEFT) mitigates this, enabling NLLB-200
to generalize well even with small low-resource
datasets—ideal for our Bangla text-to-gloss task.4.2. Model Fine-tuning
4.2.1. Fine-tuning mT5 for Bangla Gloss
Generation
AccordingtoXueetal.(2021)themT5-smallmodel
was pre-trained on a new Common Crawl-based
dataset covering 101 languages and has 300M
parameters. Therefore the model already under-
stands the underlying structure behind Bangla
words and sentences, allowing us to specialize it
in our Bangla sentence to gloss generation tasks.
We load our Bangla-sentence and gloss pairs with
a batch size of 16 because higher batch sizes like
32 and 64 would require a lot of memory for com-
putation. We set our learning rate to 0.001 with
warm-upstepssetto50allowingthemodeltograd-
ually linearly increase its learning rate to 0.001,
preventing large initial updates to the weights and
biaseswhichcandestabilizethemodel. Epochwas
selected at 20 with early stopping implemented, to
prevent overfitting and ensure optimal model selec-
tion. Additionally, we also used AdamW optimizer
for better regularization during training.
4.2.2. mBART-50 Fine-tuning Parameters
The mBART-50 model was pretrained on a diverse
set of 50 languages which included Bengali and
was primarily developed for fine tuning on trans-
lations tasks (Tang et al., 2020). Based on this
the model was fine tuned on the following hyper-
parameters. An effective batch size of 16 was se-
lected with the learning rate set to max 3e-5 with
warm-up steps set to 300. An epoch of 20 was
selected with early stopping implemented.
4.2.3. Parameter-efficient Fine-tuning of
NLLB-200
NLLB-200 was primarily optimized for machine
translation tasks. There are three variants of
this model: NLLB-200-distilled-600M, NLLB-200-
distilled-1.3B and NLLB-200-3.3B. The models are
pre trained on a massive multilingual corpus mak-
ing them eligible for the Bangla sentence to gloss
translation task. We evaluated the performance of
the NLLB-200-distilled-1.3B model to assess how
a billion-parameter architecture would perform on
our dataset. However, conducting a full parameter
finetuningofa1.3Bmodelwouldrequiresignificant
computational resources. Therefore, we applied
Parameter-EfficientFine-Tuning(PEFT)techniques,
specifically Low-Rank Adaptation (LoRA) which
enables effective model adaptation while main-
taining computational efficiency (Hu et al., 2021).
LoRa provides extra trainable parameters which
are known as low rank matrices, while preserving
the original pre-trained weights thus the original
pre-trained knowledge is less modified. Due to

Model Dataset BLEU-1 BLEU-2 BLEU-3 BLEU-4 COMET
mT5-small Base Dataset (1K) 51.79 36.26 25.73 17.56 0.8564
mT5-small Augmented Dataset (4K) 55.33 40.11 29.06 20.28 0.8602
mBART-50 Base Dataset (1K) 56.65 40.96 29.36 21.06 0.7829
mBART-50 Augmented Dataset (4K) 61.09 46.01 35.09 27.31 0.8261
GPT-4.1-nano Augmented Dataset (4K) 57.11 41.50 31.11 22.42 0.8913
NLLB-200-1.3B Base Dataset (1K) 67.99 40.82 26.49 16.67 0.907
Table 2: Model performance comparison using BLEU and COMET metrics on base (1K) and augmented
(4K) datasets.
our dataset being relatively small we set the rank
parameter(r)to8,whichdeterminesthedimension-
ality of the low-rank matrices and the LoRA. The
alpha parameter was set to 16 for a scaling factor
2 so that the learned weight updates are doubled
before being added to the original weights, thus al-
lowing the pre-trained model to learn task specific
patterns. A dropout rate of 0.05 was applied specif-
ically to LoRA layers to provide regularization and
prevent overfitting of the adapter weights. The tar-
get module section included q_proj, k_proj, v_proj,
out_proj of the self-attention layers, and fc1, fc2 in
MLPs. This allows the model to adapt in terms of
bothcontextunderstandingandfeaturetransforma-
tion for gloss sequence generation. The batch size
wassetto1withgradientaccumulationstepsof24,
creating an effective batch size of 24. Gradient ac-
cumulation maintains training stability and enables
effective parameter updates despite the small per-
device batch size. These steps were mainly taken
due to memory limitations, as larger batch sizes
would exceed available VRAM. The learning rate
was set to 3e-4 with the training epochs set to 5
and mixed precision training (FP16) was enabled
to reduce memory usage and accelerate training
whilemaintainingnumericalstability. Despitethese
efforts, we were only able to run the model on the
manual annotated part of our dataset.
4.3. Model Fine tuning Results
Table 2 shows the effectiveness of our data aug-
mentation strategies for Bangla text to gloss trans-
lation. Among these models, mBART-50 achieved
the best performance on both datasets, with the
augmented dataset (4K samples) showing consis-
tent improvements across all BLEU metrics com-
pared to the professional-only dataset of 1K sam-
ples. The model achieved a BLEU-4 score of
27.31 and COMET score of 0.8261 on the aug-
menteddataset. mT5alsobenefitedfromdataaug-
mentation, with BLEU-4 improving from 17.56 to
20.28 and achieving a COMET score of 0.8602 on
the augmented dataset. Unfortunately, NLLB-200
could not be evaluated on the augmented dataset
due to computational constraints, though it showed
promising results on the manual annotated datasetof 1000 Bangla sentences with a COMET score of
0.907. We also finetuned GPT-4.1-nano to com-
pare its performance to the other models, and it
delivered strong results on the augmented dataset,
achievingaBLEU-4scoreof22.42andtheCOMET
score of 0.8913 The consistent performance gains
across models validate our multi-faceted augmen-
tation approach combining RAG-based retrieval,
rule-based tense conversion, and masking tech-
niques.
4.4. Comparative Results with
RWTH-PHOENIX-2014T
Forcontextualcomparison,wealsoevaluateonthe
RWTH-PHOENIX-2014T dataset, a German Sign
Language benchmark widely used in sentence-to-
gloss translation. Although direct comparison is
not meaningful due to language and domain dif-
ferences, similar evaluation settings allow us to
observethatourmodelsachievecomparabletrans-
lation consistency within the Bangla domain. From
Table 3, we can observe that the mBART-50 model
achieved high BLEU-4 and comet score of 27.31
and 0.8261 on our dataset, compared to RWTH-
PHOENIX-2014T’s 21.49 and 0.6673 for mBART-
50. Similarly, for mT5, our dataset has also shown
BLEU-4 and COMET scores of 20.28 and 0.8602
compared to RWTH-PHOENIX-2014T’s 18.73 and
0.6205.
Dataset Model BLEU-1 BLEU-2 BLEU-3 BLEU-4 COMET
RWTH
PHOENIXmBART-50 48.49 35.01 26.85 21.49 0.6673
mT5-small 54.79 36.47 25.42 18.73 0.6205
Augmented
Dataset (4K)mBART-50 61.09 46.01 35.09 27.31 0.8261
mT5-small 55.33 40.11 29.06 20.28 0.8602
Table 3: Evaluation results on RWTH-PHOENIX-
2014T and our datasets.
5. Conclusion
In this paper, we have introduced Bangla-SGP,
a comprehensive, high-quality Bangla Sentence
Gloss pair dataset, which will contribute greatly
to this low-resource domain while also laying the

groundwork for promising future research in con-
tinuous sign language recognition and translation.
Furthermore, through our proposed novel RAG-
based pipeline, we have introduced and proved the
effectiveness of utilizing RAG in generating high-
qualitysyntheticglosssequences. Recognizingthe
potential of data augmentation in this low-resource
medium where expert manual annotation is quite
costly, we explored and devised various other data
augmentation strategies along with our proposed
RAG-based scheme. Our findings demonstrate
that data augmentation is a very effective and vi-
able solution for expanding gloss resources as well
as overcoming the limitations of low-resource set-
tings.
6. Dataset Availability Statement
TheBangla-SGPdatasetwillbemadepubliclyavail-
ableuponpublicationthroughourofficialrepository
at GitHub. The dataset will be released under the
CC BY-4.0 license and all augmentation rules, and
documentation will be included in the repository.
7. Discussion
7.1. Research Limitations
While our work lays down the foundation for future
work on Continuous Bangla Sign Language Recog-
nition and Translation, it also comes with several
limitations that need to be addressed. Firstly, al-
though our dataset of 4000 samples is a valuable
extension to such a low-resource setting, such as
sign language NLP tasks, it is still relatively limited
to ideally train modern transformer architectures
to perfectly capture the nuances and morpholog-
ical patterns that come with converting a Bangla
sign language sentence to its corresponding gloss
representation. Secondly, manual annotation of
Bangla sentence-gloss pairs by an expert is an
expensive procedure due to the limited availabil-
ity of Bangla sign language experts. Due to this,
we were only able to cooperate with one expert in
our work, so our dataset may include signer bias.
Thirdly, signlanguageisacomplexformofcommu-
nication that is not restricted to hand movements;
rather, it also involves facial expressions and body
movements to convey different words, emotions,
and tones. Glosses, however, as an intermedi-
ary, cannot capture facial expressions, and so our
dataset lacks a video component to capture these
aspects. Additionally, Bangla sign language is lim-
ited in vocabulary, due to which many words that
the hearing community uses may not have a sign
language/gloss representation. So, signers usually
break down that word into a set of glosses to ex-
press the word. For example, the word (shoptaho)‘week’ is broken down into (shat din) ‘seven days’
by signers. As such, our dataset needs to be ex-
pandeduponwithmoresuchexamplesformachine
translation to capture these special relationships.
Finally, names and landmarks do not typically have
adedicatedsinglesign;theyareusuallyspelledout
using alphabet-level signing. Glosses like these
don’t have a specific representation in our dataset
which may cause challenges during the training
of more advanced sign language translation sys-
tems. Finally, due to constraints in computational
resources, we were not able to evaluate NLLB-200-
1.3B on our augmented dataset. As a result, we
cannot empirically verify whether exposure to the
augmented sentence–gloss pairs would yield the
same improvements in the evaluation scores.
7.2. Future Works
The presented dataset is the Phase 1 of a multi-
modal BdSL dataset, consisting only of the gloss
sequences of spoken Bangla text. We acknowl-
edge that non-manual markers such as facial ex-
pressions, head movements and torso movements
play a critical role in sign language grammar and
meaning. In Phase 2, we will collect synchronized
sign video samples of the Bangla sentences and
add annotations for non-manual markers with the
help of multiple signers. Subsequently, through
inter-annotator agreement checks, we aim to de-
velop a comprehensive dataset that supports re-
search on both lexical and multimodal aspects of
sign language.
We plan to use the extensive dataset for creating
a pipeline that generates 3D sign language repre-
sentations from Bangla text. The planned pipeline
will build upon the gloss generation process from
Bangla text and involve constructing a gloss video
dictionary that consists of isolated gloss, which
serve as the key, and their corresponding RGB
video representations as their value. From these
videos, through the use of State-of-the-art 3D hu-
man reconstruction architectures such as SMPLer-
X, generation of 3D human meshes as obj files
is possible on a frame-by-frame basis (Cai et al.,
2024; Pavlakos et al., 2019). These meshes are
mappedbacktotheirrespectiveentriesinthegloss
video dictionary, and the resulting series of obj files
can be used to produce a continuous 3D anima-
tion of the signed sentence. We tried exploring
this pipeline on a preliminary level but several chal-
lenges remain, including the lack of a comprehen-
sive sign video dataset and unstable transitions
between consecutive signs during animation. We
were unable to solve these issues, which we will
also leave as future work or for other researchers
to address.

7.3. Ethical Consideration
All of the gloss sequences of the Bangla sentences
was annotated with explicit consent from a certi-
fied Bangla Sign Language (BdSL) interpreter. We
have obtained informed consent from them and
they are aware of their data being used for aca-
demic research purposes.
8. Bibliographical References
Mohammad Alauddin and Abul Hasnat Joarder.
2004.Deafness in Bangladesh, pages 64–69.
Springer Japan, Tokyo.
Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei,
Qingping Sun, Yanjun Wang, Hui En Pang, Haiyi
Mei, Mingyuan Zhang, Lei Zhang, Chen Change
Loy, Lei Yang, and Ziwei Liu. 2024. Smpler-x:
Scaling up expressive human pose and shape
estimation.
Alexis Conneau, Kartikay Khandelwal, Naman
Goyal, Vishrav Chaudhary, Guillaume Wenzek,
Francisco Guzmán, Edouard Grave, Myle Ott,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Unsupervised cross-lingual representation learn-
ing at scale.CoRR, abs/1911.02116.
Marta R Costa-Jussà, James Cross, Onur Çelebi,
Maha Elbayad, Kenneth Heafield, Kevin Heffer-
nan, Elahe Kalbassi, Janice Lam, Daniel Licht,
Jean Maillard, et al. 2022. No language left be-
hind: Scaling human-centered machine transla-
tion.arXiv preprint arXiv:2207.04672.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, YuanzhiLi, SheanWang, andWeizhu
Chen. 2021. Lora: Low-rank adaptation of large
language models.CoRR, abs/2106.09685.
Hamid Reza Vaezi Joze and Oscar Koller. 2019.
Ms-asl: A large-scale data set and benchmark
for understanding american sign language.
Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li,
Sergey Edunov, Marjan Ghazvininejad, Mike
Lewis, and Luke Zettlemoyer. 2020. Multilingual
denoising pre-training for neural machine trans-
lation.Transactions of the Association for Com-
putational Linguistics, 8:726–742.
Amit Moryossef, Mathias Müller, Anne Göhring,
Zifan Jiang, Yoav Goldberg, and Sarah Ebling.
2023. An open-source gloss-based baseline for
spoken to signed language translation.
Georgios Pavlakos, Vasileios Choutas, Nima Ghor-
bani, Timo Bolkart, Ahmed A. Osman, DimitriosTzionas, and Michael J. Black. 2019. Expres-
sive Body Capture: 3D Hands, Face, and Body
FromaSingleImage. In2019 IEEE/CVF Confer-
ence on Computer Vision and Pattern Recogni-
tion (CVPR), pages 10967–10977, Los Alamitos,
CA, USA. IEEE Computer Society.
Colin Raffel, Noam Shazeer, Adam Roberts,
Katherine Lee, Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li, and Peter Liu. 2019. Explor-
ing the limits of transfer learning with a unified
text-to-text transformer.
Husne Rubaiyeat, Hasan Mahmud, Ahsan Habib,
and Md Kamrul Hasan. 2024. Bdslw60: A word-
level bangla sign language dataset.
Ataher Sams, Ahsan Habib Akash, and S. M. Mah-
bubur Rahman. 2023. Signbd-word: Video-
basedbanglaword-levelsignlanguageandpose
translation. In2023 14th International Confer-
ence on Computing Communication and Net-
working Technologies (ICCCNT), pages 1–7.
Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen,
Naman Goyal, Vishrav Chaudhary, Jiatao Gu,
and Angela Fan. 2020. Multilingual translation
with extensible multilingual pretraining and fine-
tuning.CoRR, abs/2008.00401.
Beverly Trezek and Connie Mayer. 2019. Reading
and deafness: State of the evidence and impli-
cations for research and practice.Education
Sciences, 9(3).
Linting Xue, Noah Constant, Adam Roberts, Mi-
hir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya
Barua, and Colin Raffel. 2021. mT5: A mas-
sively multilingual pre-trained text-to-text trans-
former. InProceedings of the 2021 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies, pages 483–498, Online. Associa-
tion for Computational Linguistics.
Ronglai Zuo, Fangyun Wei, Zenggui Chen, Brian
Mak, Jiaolong Yang, and Xin Tong. 2024. A sim-
ple baseline for spoken language to sign lan-
guage translation with 3d avatars.