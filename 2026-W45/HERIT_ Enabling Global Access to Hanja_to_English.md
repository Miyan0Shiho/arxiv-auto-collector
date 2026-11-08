# HERIT: Enabling Global Access to Hanja-to-English Historical Translation by Mitigating Temporal Bias

**Authors**: Dayeon Jeong, Kyeongpil Kang

**Published**: 2026-11-03 14:10:59

**PDF URL**: [https://arxiv.org/pdf/2611.04949v1](https://arxiv.org/pdf/2611.04949v1)

## Abstract
Historical archives are invaluable for multidisciplinary research but remain inaccessible globally due to severe language barriers. While manual translation is prohibitively expensive and time-consuming, simply applying existing machine translation models is inadequate due to unique linguistic nuances and the temporal bias inherent in scarce human-labeled corpora. To address these challenges, we propose HERIT, a systematic domain-adaptation pipeline that leverages retrieval-augmented generation (RAG) to generate high-quality pseudo-labeled data from abundant Hanja-Korean corpora. This approach substantially expands the training dataset, mitigating both data scarcity and temporal overfitting across different historical periods. Extensive automated and human evaluations demonstrate that HERIT consistently outperforms baselines. Finally, we translate previously untranslated portions of the archives using our model, thereby enabling global access to these resources for researchers.

## Full Text


<!-- PDF content starts -->

HERIT: Enabling Global Access to Hanja-to-English Historical Translation
by Mitigating Temporal Bias
Dayeon Jeong1, Kyeongpil Kang2*
1Graduate School of Data Science, Kangwon National University
2Department of Computer Science and Engineering, Kangwon National University
{glue0440, rudvlf0413}@kangwon.ac.kr
Abstract
Historical archives are invaluable for multi-
disciplinary research but remain inaccessible
globally due to severe language barriers. While
manual translation is prohibitively expensive
and time-consuming, simply applying exist-
ing machine translation models is inadequate
due to unique linguistic nuances and the tem-
poral bias inherent in scarce human-labeled
corpora. To address these challenges, we pro-
pose HERIT, a systematic domain-adaptation
pipeline that leverages retrieval-augmented gen-
eration (RAG) to generate high-quality pseudo-
labeled data from abundant Hanja-Korean cor-
pora. This approach substantially expands the
training dataset, mitigating both data scarcity
and temporal overfitting across different his-
torical periods. Extensive automated and hu-
man evaluations demonstrate that HERIT con-
sistently outperforms baselines. Finally, we
translate previously untranslated portions of
the archives using our model, thereby enabling
global access to these resources for researchers.
1 Introduction
Historical records have been a vital source for
discovering knowledge and deriving insights. In
the humanities, these archives allow researchers
to analyze political dynamics and cultural evolu-
tion. In the natural sciences, they serve as crucial
sources for tracking natural phenomena, such as
solar eclipses, supernovae, and seismic activities.
The Annals of the Joseon Dynasty (AJD) and the
Journal of the Royal Secretariat (JRS) are regarded
as preeminent historical archives, recognized not
only in Korea but globally. They provide compre-
hensive records spanning over 500 years, compris-
ing a vast array of subjects. Extending beyond state
affairs and administration, they meticulously chron-
icle natural phenomena, including astronomical ob-
servations, seismic activity, famines, and epidemics.
*Corresponding authorAccordingly, researchers worldwide utilize these
archives across a broad spectrum of disciplines,
from the social to the natural sciences.
Given their significance, the translation of the
AJD from Hanja into Korean has been completed,
while the translation of the JRS is currently un-
derway. Furthermore, to promote the global acces-
sibility of these historical archives, the National
Institute of Korean History (NIKH) launched a
project to translate the AJD into English. How-
ever, relying on manual translation by experts pro-
ficient in both Hanja and English presents signif-
icant challenges; the project is projected to span
22 years with a budget of approximately 20 mil-
lion USD.1Furthermore, applying this manual ap-
proach to other archives such as the JRS, which is
five times more voluminous than the AJD, would
require prohibitive costs in both time and resources.
Consequently, existing human-translated English
corpora are not only scarce but also heavily skewed
towards specific reigns. Improving translation on
unseen historical periods requires a method that
generalizes beyond this temporal bias.
Since these archives are written in Hanja, lever-
aging Chinese-English LLMs appears to be an in-
tuitive solution for processing these corpora. How-
ever, such an approach encounters significant chal-
lenges. First, there is a fundamental script mis-
match: Chinese models typically utilize Simplified
Chinese, whereas Korean Hanja retains Traditional
Chinese forms, resulting in character incompati-
bility. Second, substantial semantic and syntactic
discrepancies exist. Even when characters are iden-
tical, semantic divergence is common, as homo-
graphs have often evolved distinct meanings in
the respective languages. Furthermore, the texts
contain Korea-specific Hanja usage that reflects
native grammatical structures, and the word order
1https://sillok.history.go.kr/eslk/about/tran
slationProjectInfo.do
arXiv:2611.04949v1  [cs.CL]  3 Nov 2026

frequently deviates from standard Chinese syntax.
Most critically, Chinese-English models translit-
erate proper nouns, such as personal names and
geographical locations, based on Chinese pronun-
ciation rather than Korean pronunciation. This mis-
alignment not only degrades translation quality but
also introduces factual errors.
To address these challenges, we propose a sys-
tematic domain-adaptation pipeline for training a
Hanja-to-English model specialized for historical
archives, harnessing human-translated corpora, as
shown in Fig. 1. To overcome the scarcity and tem-
poral overfitting of available expert data, we in-
troduce a retrieval-augmented in-context learning
for data augmentation strategy that generates high-
quality pseudo-labeled data. This effectively allevi-
ates data sparsity while minimizing overfitting to
specific eras. Subsequently, we presentHERIT: a
Hanja-EnglishRAG-basedIntelligentTranslation
model for historical archives. Extensive evalua-
tions, including quantitative metrics and human
expert assessments, demonstrate that HERIT signif-
icantly outperforms existing baselines. In addition,
we translate the remaining untranslated portions
of the JRS and AJD using our model. Our model
and the translated corpus are publicly available2to
facilitate future research.
2 Related Work
Historical archives have extensively contributed
to diverse research domains. In the field of social
sciences, they provide foundational data for ana-
lyzing international affairs (Thies, 2002), demog-
raphy (Alter, 2019), economic conditions (Clark,
2007), and linguistic transitions (Wei et al., 2025).
In parallel, the natural sciences leverage histori-
cal records to investigate natural phenomena such
as astronomical events (Wang et al., 2021; Wei
and Yan, 2024) and climate change (Chen et al.,
2020; Zhang et al., 2021). Thus, historical docu-
ments serve as invaluable resources across diverse
fields, offering a lens through which to understand
past events and derive critical insights for both the
present and the future.
Korean archives, such as AJD and JRS, are rec-
ognized as invaluable resources by the global aca-
demic community for their comprehensive scope,
spanning multiple centuries and covering a wide ar-
ray of subjects. Notably, the continuous and system-
atic management of these records makes them par-
2https://github.com/dandamdandam/HERITticularly advantageous for analyzing macroscopic
changes. For instance, they have been extensively
utilized in studies concerning astronomical ob-
servations and geological phenomena, as well as
disasters such as epidemics, and even economic
crises (Stephenson, 2011; Yang et al., 2012; Kim
et al., 2017; Wang et al., 2021; Wei and Yan, 2024;
Rhee, 2014; Hwang, 2022).
With advances in machine learning, such as OCR
and machine translation, research has been con-
ducted to effectively translate historical archives
into modern languages. Afli and Way (2016) pro-
posed a method to digitize historical documents
written in 17th-century French using OCR and
translate them into Modern English by applying
stochastic machine translation techniques. To trans-
late Ancient Chinese into Modern Chinese, Zhang
et al. (2019); Liu et al. (2019) developed passage-
level alignment methods and deep neural network-
based machine translation models.
Among Korean historical documents, the AJD
has been fully translated into Korean by the NIKH,
whereas the translation of the JRS remains ongoing
due to its vast volume. Addressing this, Kang et al.
(2021) developed a Hanja-to-Korean translation
model leveraging the parallel corpus derived from
both the AJD and JRS. Concurrently, a project to
translate the original Hanja text of the AJD into En-
glish is underway, with the records from the era of
King Sejong (approximately 2.5% of the total) cur-
rently completed. Using this partial Hanja-English
corpus alongside the Hanja-Korean data, Son et al.
(2022) proposed a Hanja-Korean-English multilin-
gual model. However, due to the scarcity of human-
translated Hanja-English data, the performance of
models trained from scratch remains limited.
Recent LLMs, such as Gemini-2.5 (Comanici
et al., 2025), Sonnet-4.5 (Anthropic, 2025), Kimi-
K2 (Team et al., 2025), GPT-5.1 (OpenAI, 2025),
and Qwen3 (Yang et al., 2025), have demonstrated
remarkable in-context learning capabilities. This
advancement enables them to utilize information
during the inference phase via few-shot learning,
even without having encountered it during the pre-
training or fine-tuning stages (Brown et al., 2020).
Specifically, RAG further improves performance
by incorporating retrieved examples relevant to the
input into the prompt (Lewis et al., 2020). Con-
sequently, there is active research into data aug-
mentation approaches for data-scarce domains that
leverage these few-shot learning and RAG strate-
gies (Seo et al., 2024; Song et al., 2024). Moreover,

Hanja Korean English
疫疾역질Epidemic disease
世宗세종King Sejong
江原道강원도Kangwon Province
Table 1: Examples of glossary information: Hanja
source, Korean pronunciation, and English translation.
methods such as Best-of-N (BoN) (Liu et al., 2024)
and Fusion-of-N (FusioN) (Khairi et al., 2025) have
been proposed to further enhance performance by
generating multiple candidate responses and aggre-
gating them to derive a final answer.
We address the scarcity of historical parallel
corpora by leveraging RAG-based augmentation.
Specifically, we construct a high-quality pseudo-
labeled dataset using Hanja-Korean corpora via
retrieval-augmented in-context learning and Fu-
sioN. Finally, we fine-tune a pre-trained LLM,
leveraging its inherent English proficiency and par-
tial Hanja understanding.
3 Datasets
As noted in Section 2, the translation of the AJD is
ongoing; currently, only the Annals of King Sejong
have been fully translated. For our experiments, we
obtained the AJD documents from NIKH3.
Using these human-translated documents, we
compiled a corpus yielding 350K Hanja-Korean
pairs and 17.7K Hanja-Korean-English triples4.
From the Hanja-English subset, we allocated 1,000
records each to the validation set Dvalidand the test
setDtest, with the remaining 15.7K samples con-
stituting the training set Dtrain. To mitigate period-
specific evaluation bias, we constructed an addi-
tional test set for human evaluation, Dtest
NT, by sam-
pling 80 records per king (2,080 entries total) from
the untranslated corpus. Crucially, Dtest
NTwas strictly
excluded from both the pseudo-labeling process
and the training data.
To prevent data leakage, we implemented a rig-
orous decontamination pipeline. Given that Hanja
is an ideographic script where individual charac-
ters carry dense semantic information, character-
level exact and edit-based matching is particularly
effective. Specifically, for each test instance, we
retrieved the top-10 candidate matches from Dtrain
3https://www.history.go.kr/en/main/main.do
4Since all AJD documents were originally translated from
Hanja to Korean, the Hanja-English data naturally forms
(Hanja, Korean, English) triples.using BM25 and computed their character-level
normalized Levenshtein similarity. We discarded
any candidate exceeding a similarity threshold of
0.9. We chose this threshold because historical
archives frequently feature formalized administra-
tive templates where changing a single entity or
action drastically alters the translation semantics;
lowering the threshold below 0.9would excessively
discard valid, semantically distinct evaluation in-
stances. This pipeline filtered out 83 samples from
Dtest, 3 from Dtest
NT, and 1 from the 200-sample hu-
man evaluation subset, leaving the evaluation splits
well-isolated from Dtrain. Detailed examples and
similarity distribution for the deduplication process
are provided in Appendix A.
Using Dtrainas an external knowledge base, we
applied retrieval-augmented in-context learning for
pseudo-labeling to the untranslated corpus (details
in Section 4.2). After filtering out documents longer
than 800 characters, we constructed an augmented
dataset, Daug, comprising 315K pseudo-labeled
samples from all reigns except King Sejong’s. Fi-
nally, both DtrainandDaugwere utilized to train
our model.
In addition, NIKH provides lexical resources,
including semantic meanings and English pronun-
ciations of proper nouns (e.g., personal and ge-
ographical names). Leveraging this data enables
the LLM to comprehend Hanja documents more
effectively and accurately translate proper nouns
into English based on their correct pronunciation.
Consequently, we compiled a glossary consisting
of 25K entries from these lexicons and utilized it
during the pseudo-labeling process.
4 Proposed Methods
This section presents our method for enhancing
pseudo-label quality, followed by an introduction to
our model fine-tuned on a blend of pseudo-labeled
and expert-translated datasets.
4.1 Selecting the Backbone LLM for
Pseudo-Labeling
Since the pseudo-labeling process can incur sub-
stantial API costs due to the large volume of
target historical documents, evaluating candidate
LLMs in terms of both performance and cost-
effectiveness is essential for selecting the optimal
backbone. To this end, we first assessed the trans-
lation performance of various LLMs on Dtestus-
ing base prompts. We employed standard metrics

1-1. Generate Drafts👉 T ask InstructionY our task is to translate the historical documents into English 
while preserving their original meaning and context. 
The documents were created in the Joseon era ..📃  Pr oposer  Conte x t### Input Document
- Hanja: 正朝使崔士儀, 副使柳漢回自京師 . 
- Korean: 정조사 최사의·부사 유한이 북경에서 돌아왔다 .
- Glossaries: [{"hanja": " 正朝使 ", "korean": " 정조사 ", "term": "envoy for 
the New Y ear ’ s greeting"}, {"hanja": " 崔士儀 ", "term": "Choe Saui"} ··· ]###  Ex am p les 
[Example 1]
Hanja: 正朝使工曹判書崔閏德回自京師 .
Korean: 정조사인 공조 판서 최윤덕이 북경에서 돌아왔다 .
English: Minister of W orks Choe Y undeok, envoy for of fering the ···[ Example 2]
H an j a :  謝恩使星原君 李正寧, 副使刑曹左參判崔士儀, 回自京師.
K orean :  사은사 성원군 이정녕과 부사 형조 좌참판 최사의가 북경으로부터 ···
English :  The Envo y  f or Expressing G ratitude f or I mperial B enevolence ···DraftDraftDraftDraft1. Data A ugmentationHanja- K o r 
CorpusRetrie v erInde x ingHanja- E n g 
Corpus1- 2 . F use DraftsF usorF inal S entenceC hoe S aui ,  envo y  f or 
o f f ering the N ew Y ear ' s ...👉 T ask InstructionB ased on the provided U ser I nput ,  f use them into a b etter 
translation that meets the criteria. ···📃  F usor  Conte x t### Input Document
- Hanja: 正朝使崔士儀, 副使柳漢回自京師 . 
- Korean: 정조사 최사의·부사 유한이 북경에서 돌아왔다 .
- Glossaries: [{"hanja": " 正朝使 ", "korean": " 정조사 ", "term": "envoy for 
the New Y ear ’ s greeting"}, {"hanja": " 崔士儀 ", "term": "Choe Saui"} ··· ]###  Ex am p les 
[Example 1]
Hanja: 正朝使工曹判書崔閏德回自京師 . ···[Example 2] 
Hanja: 正朝使 都摠制柳殷之,  副 使 漢城府尹沈道源,  回自京師 . 
Korean: 정조사 도총제  유은지와  부 사 한성부  윤 심도원 이 북경 으로부터  돌아왔다 .
English: Chief Commander G eneral Y u Eunji, the envoy for of fering ···### T ranslation Candidates
- candidate_1: Choe Saui, envoy for of fering the New Y ear's ... 
- candidate_2: Envoy for the New Y ear's greeting Choe Saui ...2 . F ine-T uning🪜  2 - S tage L earningSy nt h etic DataHuman- L a b eled Data
3. T ranslating✍️ Model T ranslation of 
Remaining Hanja CorpusPr oposerFigure 1: Overview of the proposed framework for translating historical documents into English. The framework
comprises three main stages: 1) pseudo-labeling for data augmentation, 2) fine-tuning the LLM by integrating
human-labeled and pseudo-labeled data, and 3) translation of the untranslated corpus using the trained model.
for machine translation, including BLEU (Post,
2018), METEOR (Banerjee and Lavie, 2005), and
chrF++ (Popovi ´c, 2017).
As shown in Table 2, Hunyuan-MT-7B, despite
being a leading Chinese translation model (Zheng
et al., 2025), underperforms on Hanja translation.
This result underscores the need for models specifi-
cally designed for Hanja. Results from the Qwen3
families demonstrate that translation quality gener-
ally improves as underlying model size increases.
Regarding reasoning capabilities, while Gemini-
2.5-Flash shows performance gains, Sonnet-4.5 ex-
periences a slight decline. This aligns with prior
surveys suggesting that reasoning does not signif-
icantly contribute to machine translation perfor-
mance (Zebaze et al., 2025). A similar trend is
observed with Gemini-2.5-Pro as shown in Table 4.
Although Gemini-2.5-Pro achieves the highest ac-
curacy, it incurs higher inference costs due to its
size and reasoning overhead. In contrast, Gemini-
2.5-Flash offers the most cost-effective trade-off,
achieving competitive performance at a fraction of
the cost. Therefore, we adopt Gemini-2.5-Flash for
the pseudo-labeling process.
4.2 Building a Pseudo-label Dataset for
Hanja-English Translation
To improve translation performance, we construct
a high-quality pseudo-labeled dataset by adopting
retrieval-augmented in-context learning and Fu-
sioN (Khairi et al., 2025) approaches, following
the process shown in Step 1 of Fig. 1. Specifically,Model BLEU chrF++ Cost $
Hunyuan-MT-7B 8.318 35.450 -
Qwen3 8B 10.631 38.738 -
Qwen3 32B 12.062 40.461 -
GPT-5.1 13.574 43.545 1.92
Kimi-K2 (R) 15.507 43.684 5.63
Gemini-2.5-Flash 15.720 43.971 0.42
Gemini-2.5-Flash (R)17.183 45.878 3.06
Sonnet-4.5 18.131 48.254 5.39
Sonnet-4.5 (R) 17.342 47.469 11.23
Gemini-2.5-Pro (R) 19.640 48.926 18.17
Table 2: Translation performance on Dtestusing basic
prompts and the total API cost. (R) denotes the integra-
tion of reasoning capabilities.
in Step 1-1, the Proposer takes untranslated source
documents to generate multiple translation drafts.
In Step 1-2, the Fusor takes these drafts along with
the original source document to generate the final
translation. Since this architecture generates the fi-
nal sentence by ensembling the Proposer’s drafts in
the Fusor, it not only mitigates mistranslation but
also enhances translation quality by leveraging ex-
panded information through a holistic examination
of the drafts.
As depicted in Fig. 1, our goal is to improve
the pseudo-label quality of the Proposer and Fu-
sor by enriching the input prompt with auxiliary
information, instead of relying solely on the Hanja
document. To address the limited proficiency of

LLMs in understanding Hanja, we augment the in-
put with the corresponding Korean text to enhance
contextual understanding. To ensure the accurate
translation of proper nouns, we utilize a curated
glossary by appending terms to the prompt when
exact matches are found in the source text. We
additionally utilize few-shot prompting to capture
the nuances, including contextual and lexical in-
formation, of Hanja-to-English translation. Specifi-
cally, we adopt RAG that retrieves the top- Ksim-
ilar documents S(h) fromDtrainfor a given in-
put Hanja document hvia an embedding-based re-
triever. These retrieved entries are fed into both the
Proposer and the Fusor as few-shot demonstrations
in the form of (Hanja, Korean, English) triples.
We optimized the hyperparameters for pseudo-
labeling based on the experiments described in Sec-
tion 5.1. To balance performance and inference
cost, we selected Gemini-2.5-Flash as the back-
bone for both the Proposer and Fusor, and em-
ployed Gemini-Embedding (Lee et al., 2025) as
the retriever. The retriever used the Korean com-
ponent of the input document as the query for the
Proposer, while utilizing both the input Hanja and
the English translation drafts as the query for the
Fusor. We employed 64 and 12 few-shot exam-
ples for the Proposer and the Fusor, respectively.
During pseudo-labeling, the Fusor synthesized the
final translation from five drafts generated by the
Proposer with a temperature of 0.3.
Finally, to construct the pseudo-labeled dataset
Daug, we applied the proposed augmentation strat-
egy to 317.4K Hanja-Korean pairs from corpora not
yet translated into English (excluding data from the
King Sejong era), incurring a cost of about $3,000.
This augmented dataset is utilized to mitigate data
sparsity and prevent potential overfitting toward a
specific historical period.
4.3 Model Training and Translation of the
Untranslated Corpus
We present our model, a Hanja-to-English his-
torical document translation model fine-tuned on
Qwen3 32B. Our model performs end-to-end trans-
lation from Hanja to English. Our training utilizes
two datasets: a pseudo-labeled dataset Daugand a
human-translated dataset Dtrain. We employ a two-
stage training strategy: the first stage uses Daug
to learn Hanja semantics and phonetic patterns
of proper nouns, while the second stage leverages
Dtrainto align the model with human-level stylistic
nuances. The entire training process was executedon eight H200 GPUs, taking 15.1 hours for the first
stage and 1.2 hours for the second stage.
Finally, we deployed our model to translate all
remaining untranslated AJD and JRS documents.
Using eight NVIDIA H200 GPUs, the translation
of these 2.09 million documents was completed in
94.6 hours, which is significantly faster and more
cost effective than manual translation. To facili-
tate global accessibility to these historical records,
both the dataset and the model are made publicly
available.
5 Experimental Results
In this section, we investigate optimal pseudo-
labeling settings and demonstrate our model’s supe-
rior performance against baselines including LLM-
based and human expert evaluations.
5.1 Ablation of the Pseudo-Labeling Pipeline
To comprehensively validate the effectiveness of
glossary integration, input augmentation, and few-
shot prompting strategies, we evaluated translation
performance on Dtestusing Gemini-2.5-Flash. All
experimental results with costs on Batch API are
summarized in Table 3.
First, we observe a significant improvement in
translation performance when a glossary is incorpo-
rated. This suggests that the glossary information
facilitates a better understanding of Hanja terms,
aiding in their translation into appropriate English
pronunciations. Similarly, providing pre-translated
Korean inputs alongside the Hanja source text en-
hances the understanding of the input documents,
indicating that Korean translations effectively sup-
port the pseudo-labeling process.
The latter part of the table presents the impact
of RAG and FusioN, and the input language on
the retriever. Including few-shot examples yields
substantial performance gains. In particular, while
static shots also improve performance, examples
that are relevant to the input document lead to much
larger gains, highlighting the effectiveness of RAG-
based methods for enhancing performance. For dy-
namic shots, using the Korean translation of the
input Hanja as the retriever query resulted in bet-
ter performance than using the Hanja input alone.
This is because the embedding model used for re-
trieval was trained on a larger corpus of Korean
text compared to Hanja, resulting in superior com-
prehension of Korean.
Our method using FusioN outperforms standard

BLEU METEOR chrF++ Cost $
Baseline 15.720 0.421 43.971 0.213
+ Glossary 23.747 0.504 51.478 0.236
+ Korean 25.457 0.520 52.637 0.255
+ RAG (K: 12)
•Static shots 30.387 0.565 56.385 0.981
•Dynamic shots (Hanja) 36.223 0.604 60.354 1.115
•Dynamic shots (Kor)36.430 0.606 60.5451.254
•Dynamic shots (Hanja + Kor) 35.945 0.602 60.219 1.272
RAG (Dynamic Shots, Kor,K: 64) 37.905 0.616 61.534 5.646
+ FusioN (K: 12,D: 4)
Proposer (Kor) Fusor (Kor) 36.869 0.614 61.474 2.713
Proposer (Kor) Fusor (Eng) 36.901 0.613 61.420 2.475
Proposer (Kor) Fusor (Hanja) 36.985 0.614 61.538 2.537
Proposer (Kor) Fusor (Hanja + Eng)37.017 0.614 61.5042.642
Proposer (Kor) Fusor (Hanja + Kor + Eng) 36.926 0.614 61.499 2.806
KProposer : 64,K Fusor: 12,D: 5
Proposer (Kor) Fusor (Hanja + Eng)38.462 0.625 62.5017.059
Table 3: Ablation study of pseudo-labeling. The language in parentheses denotes the query language used for
retrieval, whileKandDrepresent the number of few-shot examples and translation drafts, respectively.
single-stage translation pipelines. Experimental re-
sults with 12 few-shot examples and 4 drafts in-
dicate that concatenating the Hanja input with the
English draft sentences as query inputs for the re-
triever is effective when finding relevant documents
for the Fusor. This suggests that in the Fusor, which
refines drafts to generate the final translation out-
put, both the context of the target Hanja sentence
and that of the English candidate sentences are cru-
cial. Additionally, we conducted experiments on
the number of few-shots provided to the Proposer
and Fusor, the results of which are shown in Fig. 6.
Finally, we set the number of drafts to 5 and few-
shot examples for the Proposer and Fusor to 64 and
12, respectively.
As indicated in the bottom row of Table 3, the fi-
nal pseudo-labeling performance achieved a BLEU
score of 38.462, outperforming other baselines, in-
cluding the single translation approach that em-
ploys 64 dynamic few-shot examples (consistent
with the proposer). Note that this performance
does not reflect the direct translation of Hanja-only
documents into English. Instead, it represents the
pseudo-labeling performance facilitated by aux-
iliary information, such as Korean translations,
glossary data, and retrieved references from Dtrain.
Therefore, this rich auxiliary information enables
213214215216217218
Data Size36.036.537.037.538.038.5BLEU
Qwen3 8B
Qwen3 32BFigure 2: Translation performance with respect to
pseudo-labeling data size.
the construction of a high-quality pseudo-labeled
dataset, which subsequently improves model fine-
tuning.
5.2 Impact of Data Augmentation and
Two-Stage Learning
To verify the effectiveness of training with our aug-
mented data, we evaluated translation performance
using an expert-translated test dataset,Dtest.
First, we evaluate the model performance with
respect to the size of Daugused in the first train-
ing stage, as shown in Figure 2. We observe that

Model BLEU METEOR chrF++
H2KE 24.62 - -
Pivot-MT 35.539 0.596 59.804
With Glossary & RAG
Gemini-2.5-Flash 35.632 0.600 60.176
Gemini-2.5-Pro 34.847 0.598 60.139
Qwen3 32B (Dtrain) 38.117 0.622 62.338
Without Glossary & RAG
Qwen3 32B (Daug) 34.747 0.596 59.732
Qwen3 32B (Dtrain) 36.930 0.613 61.583
+ Mixed training:Daug∪Dtrain
Qwen3 32B 36.849 0.613 61.246
+ Two-stage learning:Daug→Dtrain(Ours)
Qwen3 8B37.849 0.623 62.187
Qwen3 32B38.432 0.626 62.525
Table 4: Translation performance on Dtest. Our method
is compared against baselines (H2KE (Son et al., 2022)
and Pivot-MT), zero-shot LLMs, and various ablation
models. For H2KE, the BLEU score reported in the
original paper is used.
translation performance improves as the volume of
pseudo-labeled data increases, demonstrating the
effectiveness of our proposed approach.
Furthermore, we compared models fine-tuned
exclusively on Dtrainagainst those trained on both
DtrainandDaug. To isolate the effects of fine-tuning,
we employed Gemini-2.5 Flash and Pro models
utilizing glossary information and RAG as base-
lines. As shown in Table 4, given sufficient glossary
context and relevant documents, Gemini-2.5-Pro
demonstrates no significant performance advantage
over Gemini-2.5-Flash. Although the model trained
only on Dtrainoutperforms the one trained only
onDaug, the latter achieves performance compara-
ble to Gemini-2.5-Pro augmented with a glossary
and RAG. More importantly, incorporating our con-
structed Daugleads to better performance than train-
ing on Dtrainalone, highlighting the effectiveness
of our pseudo-labeling approach in low-resource
historical document translation.
In addition, our two-stage training strategy out-
performs the naive alternative of jointly training on
a mixture of DaugandDtrain. We attribute this to
the severe size imbalance between the relatively
small Dtrainand the much larger Daug, prevent-
ing the model from effectively leveraging both
datasets when trained on them simultaneously. By
contrast, the two-stage strategy allows the model toModel Qwen3 8B Qwen3 32B
With Glossary & RAG
Dtrain6.539 7.003
Without Glossary & RAG
Dtrain6.312 6.883
Daug→Dtrain7.887 8.009
Table 5: LLM-based evaluation on the testsetDtest
NT.
better utilize their complementary strengths. Fur-
thermore, we fine-tuned an additional Qwen3-32B-
based pivot translation baseline (Pivot-MT) under
the same settings as our model. This baseline first
translates Hanja into Korean and then translates the
resulting Korean text into English. HERIT outper-
forms the Pivot-MT baseline, indicating that direct
Hanja-to-English translation is more effective than
pivot-based translation by avoiding error propaga-
tion across the two translation stages.
To examine whether pseudo-labeling can elimi-
nate the need for inference-time retrieval, we fine-
tuned Qwen3-32B on Dtrainwith a glossary and
RAG applied during both training and inference.
Although this configuration outperforms the Dtrain-
only baseline, it still falls short of HERIT. More
critically, retrieval augmentation incurs substan-
tial computational overhead—expanding the con-
text length to over 8K tokens (increasing the input
prompt length by 13 to 14 times) and scaling train-
ing computational cost by 9 times. By contrast,
HERIT effectively distills external knowledge di-
rectly into model parameters, achieving superior
translation quality while remaining retrieval-free
and highly efficient for large-scale historical text
translation.
5.3 LLM-Based and Human Evaluations
Although Dtestconsists of human-translated data, it
is restricted to specific reign periods, which limits
generalization across the entire chronological scope
of the AJD corpus. To address this, we performed
an LLM-based evaluation on the held-out dataset
Dtest
NTby sampling 80 documents from each of the
remaining reign periods excluded from pseudo-
labeling. Adopting the method proposed by Zheng
et al. (2023), models first generate translations for
Dtest
NT. Subsequently, the LLM Evaluator assesses
the translation quality score on a scale of 1 to
10, taking the source Hanja text and the model-
generated English translation as inputs. To enhance

1 2 3 4 5
ScoreMean: 3.93Mean: 3.22HERIT
Baseline
0 20 40 60 80 100
Preference Ratio (%)66.84% 33.16%Figure 3: Human expert evaluation scores and prefer-
ence ratios for our model versus the baseline.
the reliability of model-based evaluation, we pro-
vided the LLM Evaluator with the corresponding
human-translated Korean text and glossary infor-
mation as reference materials. We chose Sonnet-4.6
as the evaluator to avoid potential bias arising from
the training process.
As shown in Table 5, the LLM-based evalua-
tion demonstrates that for both the 8B and 32B
models, the two-stage learning approach combin-
ing pseudo-labeled and human-translated data out-
performs models trained exclusively on human-
translated data. Notably, given that Dtest
NTcovers the
entire AJD period disjoint from the training dataset,
these results confirm the effectiveness of pseudo-
labeling in enhancing generalization.
Furthermore, while incorporating the glossary
and RAG into both training and inference slightly
improves performance over the baseline trained
solely on Dtrain, the two-stage model trained on
Daugsubstantially outperforms the model fine-
tuned with the glossary and RAG. This high-
lights that pseudo-labeling provides significant ad-
ditional performance gains, demonstrating that our
approach effectively generalizes across the entire
AJD period rather than temporal overfitting to the
specific timeframe ofDtrain.
Additionally, using 200 examples sampled uni-
formly across different kings from Dtest
NT, we re-
cruited domain experts in Sinographic literature
and Korean history who are proficient in English
to evaluate our model against the baseline (Qwen3
32B trained solely on Dtrain). We asked the experts
to rate each translation on a scale of 1-5. Further, to
differentiate between models in cases where scores
were tied (excluding identical translations), we per-
formed a comparative evaluation where experts
were instructed to select the better response.
As shown in Fig. 3, our model achieved ascore of 3.93, outperforming the baseline score
of 3.22. In addition, HERIT achieved a selection
rate of 66.84% in the pairwise preference evalua-
tion, which is more than twice that of the baseline.
These results demonstrate the superior translation
quality of HERIT in human expert evaluation.
6 Analysis
This section evaluates the model’s robustness
against temporal overfitting and provides a qualita-
tive analysis of translation examples.
6.1 Temporal Overfitting
To investigate whether our method effectively mit-
igates temporal overfitting, we analyzed the hu-
man evaluation results. Since the evaluation sam-
ples were stratified by king, as detailed in Sec-
tion 5.3, we computed the average score for each
reign across both HERIT and the baseline, Qwen3
32B trained exclusively on human-labeled data. We
then determined the performance gap between the
two models for each reign and visualized these
gaps chronologically.
As noted in Section 2, the human-labeled cor-
pus is heavily concentrated on documents from
the reign of King Sejong. Consequently, Qwen3
trained on this temporally biased dataset performs
comparably to HERIT during this specific period,
as shown in Fig. 4. In contrast, HERIT consistently
achieves superior performance on documents from
other reigns. This indicates that training on data
concentrated in a single historical period leads to
temporal overfitting, while the proposed pseudo-
labeling strategy helps mitigate this bias and en-
hances overall translation robustness.
6.2 Qualitative Analysis of Translations
We also conducted a qualitative comparison of
HERIT against two baselines using samples from
the test set: Qwen3 32B fine-tuned exclusively
on human-translated data and Hunyuan-MT-7B,
a leading Chinese translation model.
As shown in the first example in Table 6, both
Qwen3 32B and HERIT correctly transliterated the
proper nouns according to Korean pronunciation
rules, whereas Hunyuan-MT-7B incorrectly relied
on Chinese phonetics. In the second case, the term
房子 carries distinct semantics depending on the
context: it denotes a “house” in standard Chinese
but refers to a “female palace servant” in the con-
text of the Joseon Dynasty. While Hunyuan-MT-7B

1400 1450 1500 1550 1600 1650 1700 1750 1800 1850 1900
Year0.4
0.00.40.81.21.62.0DifferenceKing Sejong
(1418 1450)
Figure 4: Differences in human expert scores between HERIT (ours) and Qwen3 32B trained solely on human-
labeled data, ordered chronologically by each king’s reign. Positive values indicate that HERIT outperforms the
baseline, demonstrating its robustness across different historical periods.
Input Hanja兩司啓趙應奎事,不允.
Hunyuan-MTa petition was submitted by the two departments regarding Zhao Yingkui’s case, but it
was denied.
Qwen3 32BThe two investigative bodies reported on the matter of Jo Eunggyu, but the king did not
approve their request.
HERIT The two offices reported on the matter of Jo Eunggyu, but the king did not approve.
Input Hanja 宜給宣飯及房子,炊飯,汲水人等
Hunyuan-MT It is appropriate to provide food, housing, cooking facilities, and water for the people.
Qwen3 32B They should be provided with food, a house, a female palace servant, and a water carrier.
HERIT We should provide them with meals, female palace servants, cooks, and water carriers.
Input Hanja 宥魚思漢,京外從便
Hunyuan-MT Yu Yu thinks of Han; outside the capital, do as you please.
Qwen3 32BThe king pardoned Eo Sahan and allowed him to reside in a place of his choice in the
capital or in the provinces.
HERITThe king pardoned Eo Sahan and allowed him to reside in a place of his choice outside
the capital.
Table 6: Qualitative examples of translations by different models using samples in Dtest. Red denotes incorrect
translations, while blue indicates correct ones.
defaulted to the standard meaning and Qwen3 32B
conflated the interpretations, HERIT identified the
historical context and translated the term correctly.
In the last example, Hunyuan-MT-7B failed to rec-
ognize the proper name 魚思漢 , and neither base-
line recognized the penal term 京外從便 (a form of
punishment). In contrast, HERIT accurately trans-
lated these domain-specific terms.
These results demonstrate HERIT’s superior
ability to deliver precise translations by capturing
both Sino-Korean lexical nuances and the unique
contexts of historical documents.
7 Conclusion
In this study, we presented HERIT, a systematic
domain-adaptation pipeline designed to address
the scarcity of parallel corpora in historical trans-lation. By leveraging abundant intermediate re-
sources through a RAG- and Fusion-based strategy,
we developed a robust method for generating high-
quality pseudo-labels without relying on costly hu-
man annotation. Extensive experiments, including
both human and LLM-based evaluations, demon-
strated that HERIT not only outperforms strong
baselines but also effectively alleviates temporal
overfitting, a persistent challenge in analyzing his-
torical archives. Furthermore, our results highlight
the potential of LLM-based data augmentation for
other low-resource classical languages where di-
rect translation pairs are limited. By enabling the
translation of previously inaccessible documents,
our work facilitates broader access to historical
archives, empowering researchers worldwide to en-
gage more deeply with historical texts.

Limitations
Due to budgetary constraints, we primarily em-
ployed Gemini-2.5-Flash for the Proposer and Fu-
sor components, focusing exclusively on AJD doc-
uments for pseudo-labeling. We anticipate that em-
ploying a more diverse set of LLMs as Proposers
to enhance translation diversity, as well as incor-
porating JRS documents as targets, would yield
higher-quality pseudo-labels. Furthermore, hard-
ware limitations necessitated restricting the maxi-
mum sequence length to 2,048 tokens, with a limit
of 800 Hanja characters. Consequently, approxi-
mately 6.7% of the training data was filtered out.
For inference, documents exceeding this limit were
processed using a chunking strategy. We expect
that extending the model’s maximum input length
will further improve translation performance.
Acknowledgments
This work was supported by the National Re-
search Foundation of Korea(NRF) grant funded
by the Korean government(MSIT) in 2026 (RS-
2025-00555320, Contribution Rate: 70%; RS-2023-
00242528, Regional Demand Oriented Data Sci-
ence Human Resource Development for Regional
Industry Innovation, Contribution Rate: 20%), the
ANCHOR program through the Gangwon AN-
CHOR Center, funded by the Ministry of Educa-
tion(MOE) and Gangwon State(G.S.), Republic of
Korea(2026-ANCHOR-10-002, Contribution Rate:
10%), and the Google Cloud Research Credits pro-
gram with the award GCP19980904.
References
Haithem Afli and Andy Way. 2016. Integrating optical
character recognition and machine translation of his-
torical documents. InProceedings of the Workshop
on Language Technology Resources and Tools for
Digital Humanities (LT4DH), pages 109–116.
George C Alter. 2019. The evolution of models in his-
torical demography.Journal of Interdisciplinary His-
tory, 50(3):325–362.
Anthropic. 2025. Introducing claude sonnet 4.5. https:
//www.anthropic.com/news/claude-sonnet-4
-5. Accessed: 2025-01-03.
Satanjeev Banerjee and Alon Lavie. 2005. Meteor: An
automatic metric for mt evaluation with improved cor-
relation with human judgments. InProceedings of
the acl workshop on intrinsic and extrinsic evaluation
measures for machine translation and/or summariza-
tion, pages 65–72.Tom Brown, Benjamin Mann, Nick Ryder, Melanie
Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind
Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, and 1 others. 2020. Language models are
few-shot learners.Advances in neural information
processing systems, 33:1877–1901.
Siying Chen, Yun Su, Xiuqi Fang, and Jia He. 2020. Cli-
mate records in ancient chinese diaries and their ap-
plication in historical climate reconstruction–a case
study of yunshan diary.Climate of the Past Discus-
sions, 2020:1–28.
Gregory Clark. 2007. The long march of history:
Farm wages, population, and economic growth, eng-
land 1209–1869 1.The Economic History Review,
60(1):97–135.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
and 1 others. 2025. Gemini 2.5: Pushing the fron-
tier with advanced reasoning, multimodality, long
context, and next generation agentic capabilities.
arXiv:2507.06261.
Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke
Zettlemoyer. 2022. 8-bit optimizers via block-wise
quantization.International Conference on Learning
Representations.
Kun Hwang. 2022. A disastrous situation of plague and
collapsed community depicted in a joseon dynasty
tale.Infection & Chemotherapy, 54(2):388.
Kyeongpil Kang, Kyohoon Jin, Soyoung Yang, Soojin
Jang, Jaegul Choo, and Youngbin Kim. 2021. Restor-
ing and mining the records of the joseon dynasty via
neural language modeling and machine translation.
InConference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 4031–4042.
Ammar Khairi, Daniel D’souza, Marzieh Fadaee, and
Julia Kreutzer. 2025. Making, not taking, the best of
n.arXiv:2510.00931.
Kwang-Hee Kim, Jung-Ho Park, Yongcheol Park, Tian-
Yao Hao, and Han-Joon Kim. 2017. Crustal struc-
ture beneath the southern korean peninsula from lo-
cal earthquakes.Geophysical Journal International,
209(2):969–978.
Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel
Cer, Madhuri Shanbhogue, Iftekhar Naim, Gus-
tavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Hen-
rique Schechter Vera, and 1 others. 2025. Gemini
embedding: Generalizable embeddings from gemini.
arXiv:2503.07891.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.

Dayiheng Liu, Kexin Yang, Qian Qu, and Jiancheng
Lv. 2019. Ancient–modern chinese translation with
a new large training dataset.ACM Transactions on
Asian and Low-Resource Language Information Pro-
cessing (TALLIP), 19(1):1–13.
Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman,
Mohammad Saleh, Peter J Liu, and Jialu Liu. 2024.
Statistical rejection sampling improves preference op-
timization. InInternational Conference on Learning
Representations.
OpenAI. 2025. Gpt-5.1: A smarter, more conversational
chatgpt.https://openai.com/index/gpt-5-1/ .
Accessed: 2026-01-03.
Maja Popovi ´c. 2017. chrf++: words helping character n-
grams. InProceedings of the conference on machine
translation, pages 612–618.
Matt Post. 2018. A call for clarity in reporting BLEU
scores. InProceedings of the Conference on Machine
Translation: Research Papers, pages 186–191.
Young Hoon Rhee. 2014. Economic stagnation and cri-
sis in korea during the eighteenth and nineteenth cen-
turies.Australian Economic History Review, 54(1):1–
13.
Minju Seo, Jinheon Baek, James Thorne, and
Sung Ju Hwang. 2024. Retrieval-augmented
data augmentation for low-resource domain tasks.
arXiv:2402.13482.
J Son, J Jin, H Yoo, J Bak, K Cho, and A Oh. 2022.
Translating hanja historical documents to contempo-
rary korean and english.Findings of the Association
for Computational Linguistics: EMNLP 2022, pages
1260–1272.
Fangzhou Song, Bin Zhu, Yanbin Hao, and Shuo Wang.
2024. Enhancing recipe retrieval with foundation
models: A data augmentation perspective. InEu-
ropean Conference on Computer Vision, pages 111–
127.
F Richard Stephenson. 2011. Historical eclipses and
earth’s rotation: 700 bc–ad 1600. InHighlighting
the History of Astronomy in the Asia-Pacific Region:
Proceedings of the ICOA Conference, pages 3–20.
Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen,
Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru
Chen, Yuankun Chen, Yutian Chen, and 1 oth-
ers. 2025. Kimi k2: Open agentic intelligence.
arXiv:2507.20534.
Cameron G Thies. 2002. A pragmatic guide to qualita-
tive historical analysis in the study of international re-
lations.International studies perspectives, 3(4):351–
372.
Yuqi Wang, Si Chen, Kaihua Xu, Limei Yan, Xinan
Yue, Fei He, and Yong Wei. 2021. Ancient auro-
ral records compiled from korean historical books.
Journal of Geophysical Research: Space Physics,
126(1):e2020JA028763.Yong Wei and LiMei Yan. 2024. Solar cycles during
the seventeenth century revealed by equatorial aurora
records.Earth and Planetary Physics, 9(1):182–187.
Yuting Wei, Meiling Li, Yangfu Zhu, Yuanxing Xu,
Yuqing Li, and Bin Wu. 2025. A diachronic language
model for long-time span classical chinese.Informa-
tion Processing & Management, 62(1):103925.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, and 1 others. 2025.
Qwen3 technical report.arXiv:2505.09388.
Hong-Jin Yang, Changbom Park, and Myeong-Gu
Park. 2012. Analysis of historical meteor and me-
teor shower records: Korea, china and japan.Pro-
ceedings of the International Astronomical Union,
10(H16):150–151.
Armel Zebaze, Rachel Bawden, and Benoît Sagot. 2025.
Llm reasoning for machine translation: Synthetic data
generation over thinking tokens.arXiv:2510.11919.
Can Zhang, Cheng Zhao, Aifeng Zhou, Haixia Zhang,
Weiguo Liu, Xiaoping Feng, Xiaoshuang Sun, Tian-
long Yan, Chengcheng Leng, Ji Shen, and 1 others.
2021. Quantification of temperature and precipi-
tation changes in northern china during the “5000-
year” chinese history.Quaternary Science Reviews,
255:106819.
Zhiyuan Zhang, Wei Li, and Qi Su. 2019. Automatic
translating between ancient chinese and contempo-
rary chinese with limited aligned corpora. InCCF
international conference on natural language pro-
cessing and chinese computing, pages 157–167.
Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo,
Chien-Chin Huang, Min Xu, Less Wright, Hamid
Shojanazeri, Myle Ott, Sam Shleifer, and 1 others.
2023. Pytorch fsdp: Experiences on scaling fully
sharded data parallel.Proceedings of the VLDB En-
dowment, 16(12):3848–3860.
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,
Zhuohan Li, Dacheng Li, Eric Xing, and 1 others.
2023. Judging llm-as-a-judge with mt-bench and
chatbot arena.Advances in neural information pro-
cessing systems, 36:46595–46623.
Mao Zheng, Zheng Li, Bingxin Qu, Mingyang Song,
Yang Du, Mingrui Sun, and Di Wang. 2025.
Hunyuan-mt technical report.arXiv:2509.05209.

0.0-0.6 0.6-0.7 0.7-0.8 0.8-0.9 0.9-1.0
Similarity Interval020406080Ratio (%)75.6%
5.8% 5.1% 4.8%8.7%Figure 5: Distribution of test set samples across similar-
ity intervals against the training data.
A Deduplication
To justify our similarity threshold, Table 7 presents
qualitative comparisons around the threshold. In
Example 1, the sentence essentially repeats a stan-
dardized tribute record with near-identical phrasing
and semantics, justifying its removal. Conversely,
Example 2 demonstrates why a lower threshold
would over-filter; replacing the ritual name ( 端
午vs.中秋 ) radically shifts the historical context,
date, and English translation despite high lexical
overlap. Manual inspection across intervals con-
firmed that 0.9 strictly eliminates near-duplicate
administrative phrasing while preserving semanti-
cally distinct formulaic records.
Figure 5 shows the distribution of maximum Lev-
enshtein similarity between DtestandDtrain. The
majority of evaluation samples ( 75.6% ) exhibit low
overlap ( ≤0.6 ). Crucially, no significant clustering
is observed near the 0.90 boundary, ensuring that
the benchmark remains well-isolated from training
data and genuinely assesses generalization rather
than template recall.
B Training Details
We trained the models for one epoch in each stage.
To mitigate overfitting and prevent a bias toward
the later stage, we employed stage-specific learning
rates. For the 32B model, the learning rates were set
to1×10−5for the first stage and 5×10−6for the
second; for the 8B model, we applied 2×10−5and
1×10−5, respectively. The batch size was set to 64
for the first stage and 16 for the second stage. All
models were trained with the PagedAdamW 8-bit
optimizer (Dettmers et al., 2022) and FSDP (Zhao
et al., 2023) on eight NVIDIA H200 GPUs.
0510152025
Cost
2 4 816 32 64128 256
Number of few-shot examples343536373839BLEU
3456
Cost
24 812 16 32
Number of few-shot examples37.337.437.537.637.737.837.9BLEU
Figure 6: Performance and cost of the Proposer and
Fusor across different numbers of few-shot examples.
C Experimental Results
C.1 Number of Few-shot Examples
Fig. 6 illustrates the impact of the number of few-
shot examples on the performance of both the Pro-
poser and Fusor. Notably, the Proposer exhibits
consistent performance gains up to 128 examples.
However, to balance performance against the in-
creased LLM inference costs incurred by larger
shot sizes, we adopted a 64-shot configuration for
the Proposer during pseudo-labeling. Additionally,
we identified that a temperature of 0.3 and a draft
count of 5 provided the best performance, and thus
utilized these parameters for pseudo-labeling. In
contrast, although the Fusor also improves with
more examples, its marginal gains are less pro-
nounced than those of the Proposer. Therefore, con-
sidering the performance-cost trade-off, we set the
few-shot size for the Fusor to 12.
C.2 Impact of FusioN
We also evaluated the pseudo-labeling performance
of multi-candidate aggregation methods, specifi-
cally FusioN and BoN. As shown in Table 8, BoN
outperforms the baseline, which generates transla-
tions in a single stage without candidate aggrega-
tion. However, FusioN demonstrates superior per-
formance compared to BoN, suggesting that synthe-

Case Language Sample inDtestSearched Sample inDtrain
Example 1
(Sim: 0.900)Hanja宗貞盛遣人來獻土宜.宗貞盛遣人來獻土物.
English S¯o Sadamori sent a messenger to
present local products.S¯o Sadamori sent a man to present his
local products to the king.
Example 2
(Sim: 0.857)Hanja上王詣健元陵,行端午別祭.上王詣健元陵,行中秋別祭.
English The abdicated king went to the royal
tomb of King Taejo and held a special
ancestral rite of the fifth day of the
fifth month.The abdicated king visited the royal
tomb of King Taejo to conduct the
special sacrificial rite for the Mid-
Autumn.
Table 7: Qualitative examples of test–train sample pairs with high Levenshtein similarity.
BLEU METEOR chrF++
No aggregation 37.905 0.616 61.534
BoN 38.337 0.622 62.117
FusioN38.462 0.625 62.501
Table 8: Comparison of pseudo-labeling performance
among FusioN, BoN, and Baseline.
Method BLEU METEOR chrF++ Cost ($)
Multiple 36.250 0.613 61.507 234.304
Single38.462 0.625 62.501 58.517
Table 9: Performance comparison between diverse-
model and single-model FusioN approaches.
sizing a final output by aggregating all candidates
is more effective than simply selecting the single
best candidate.
In addition, we investigated the impact of pro-
poser diversity on cost-effectiveness. The single-
proposer setting relied solely on Gemini-2.5-Flash,
whereas the diverse-proposer setting combined
Qwen3 235B-A22B, DeepSeek-V3.1, GPT-5.1,
Gemini-2.5-Flash, and Sonnet 4.5. Table 9 reports
the performance and cost of the single- and diverse-
proposer settings, taking prompt caching into ac-
count. The results show that the single-model con-
figuration achieves both better performance and
higher cost efficiency. This suggests that while in-
creasing model diversity yields a wider variety of
candidates, it does not necessarily translate into
better performance, making the single-model con-
figuration a more cost-effective choice.
To qualitatively verify that FusioN effectively
synthesizes proposer-generated translations during
the aggregation stage, we analyze the representa-tive examples shown in Table 10. In the first exam-
ple, while candidate (2) misinterprets the syntactic
structure, yielding a flawed translation, FusioN suc-
cessfully adopts the correct phrasing from candi-
date (1), preserving semantic accuracy. In the sec-
ond example, candidate (2) suffers from incomplete
generation. FusioN effectively filters out this noise
and consolidates valid information from the viable
candidates to produce a fluent and comprehensive
translation. In the third example, although multiple
candidates generate the generic term “white fal-
con,” FusioN selects the historically accurate term
“white goshawk” provided by a minority candidate,
demonstrating its capability for fine-grained lexical
selection. Overall, these cases illustrate that Fu-
sioN effectively aggregates candidate translations
to ensure both factual correctness and grammatical
fluency.
C.3 Translation Analysis for Proper Nouns
To assess whether our proposed pseudo-labeling
and two-stage training strategy effectively capture
historical and factual context, we conducted an
LLM-based evaluation. In particular, we examined
whether the model correctly renders proper nouns
using their actual Korean pronunciations rather
than Chinese readings. Specifically, using Claude
Sonnet-4.6, we evaluated the translation accuracy
of proper nouns—including historical figures, ge-
ographical names, and official titles—in Dtest. As
shown in Table 11, the proposed pseudo-labeling
method substantially improves proper-noun trans-
lation accuracy over the baseline model trained
only on Dtrain. Furthermore, the two-stage learn-
ing strategy provides additional gains beyond those
achieved by pseudo-labeling alone.

Input Hanja御勤政殿受朝.
Ground Truth The King proceeded to Geunjeong Hall and received officials in the morning court
assembly.
Translation
by Candidates(1) The king proceeded to Geunjeong Hall and received officials in the morning court
assembly.
(2) The king proceeded to Geunjeong Hall to receive the morning court assembly.
...
Translation
by FusorThe king proceeded to Geunjeong Hall and received officials in the morning court
assembly.
Input Hanja傳旨兵曹:自今司僕寺常養馬,冬節則三百匹,夏節則二百匹.
Ground TruthThe King had a royal directive transmitted to the Ministry of War: “Hereafter, the
number of horses to be reared continually by the Office of Royal Stables shall be 300
head in winter and 200 head in summer.”
Translation
by Candidates(1) The King transmitted a royal directive to the Ministry of War as follows: “From this
day forward, the Office of the Royal Stables shall always raise horses, with 300 horses
in winter and 200 horses in summer”
(2) The King transmitted a royal directive to the Ministry of War as follows:
...
Translation
by FusorThe King transmitted a royal directive to the Ministry of War, stating,“From this day
forward, the Office of the Royal Stables shall always raise horses, with 300 horses in
winter and 200 horses in summer”
Input Hanja咸吉道監司,進白鷹一連.
Ground Truth Governor of Hamgil Province presented the king with one white goshawk.
Translation
by Candidates(1) The governor of Hamgil Province presented the king with one white goshawk.
(2) The governor of Hamgil Province presented the king with one white falcon.
(3) The governor of Hamgil Province presented a white falcon to the king.
...
Translation
by FusorThe Governor of Hamgil Province presented the king with one white goshawk.
Table 10: Qualitative case studies showing how FusioN synthesizes candidate translations.
Model Accuracy
Qwen3 32B (Dtrain) 0.818
Qwen3 32B (Daug∪Dtrain) 0.875
Qwen3 32B (Daug→Dtrain)0.878
Table 11: LLM-based evaluation of proper noun accu-
racy across the baseline, pseudo-labeling, and training
strategies.
D Human Evaluation
To compare our model against the baseline, we re-
cruited three professional experts in Sinographic
literature and Korean history who are proficient in
English to evaluate 200 items. The evaluation was
conducted in a strict double-blind manner, ensuring
that the evaluators remained completely unawareof the identity of the generating models. Prior to
the task, the experts were provided with compre-
hensive instructions regarding the interface and the
evaluation rubric.
In addition, we developed a custom web-based
interface to facilitate this evaluation, as shown in
Fig. 7. The interface displays the source Hanja
text and the associated glossary at the top, while
the anonymized outputs from both models are pre-
sented in the center. Evaluators rated each trans-
lation on a 1-5 Likert scale. To ensure a rigorous
comparative analysis, a forced-choice mechanism
was implemented; whenever a tie occurred (i.e.,
identical scores were given to both models), evalua-
tors were required to select the superior translation
based on three core criteria:
•Semantic coherence: The degree of preserva-
tion of the original meaning without omission

Figure 7: Web-based evaluation tool for human experts.
or distortion.
•Terminological consistency: The correct and
uniform translation of historical terms and
proper nouns.
•Naturalness: The fluency and readability of
the translated English text.
Along with the quantitative scores, evaluators
provided qualitative comments justifying their pref-
erences or pointing out specific errors in the coun-
terpart translations, including semantic distortion,
information omission, terminology misuse, and un-
natural phrasing.
E Prompt Format
Fig. 8 shows the base prompt template used for
direct Hanja-to-English translation in the baseline
models discussed in Sections 4.1 and 5. The system
prompt defines the instructions for the translation
task and provides a description of Hanja within
the “Task” section. It also outlines “Guidelines for
translation” and briefly explains the user prompt
structure and expected output format. In the user
prompt, the source Hanja text and relevant glossary
information are provided in the “Input Document”
section. For RAG-based baselines, relevant “Exam-
ples” are included as (Hanja, English) pairs.
Similarly, Figs. 9 and 10 present the prompt
structures used in the pseudo-labeling process de-
scribed in Section 4.2. Unlike the base prompt,the Proposer and Fusor prompts incorporate Ko-
rean translations provided by human experts, which
serve as guidance for translating historical docu-
ments. The Fusor prompt largely follows the struc-
ture of the Proposer prompt, but differs in sev-
eral key respects. Following the implementation
of Khairi et al. (2025), it includes a “Task” section
describing the draft-fusion method and a “Fusion
Steps” section outlining the procedure. In the user
prompt, the task instructions and guidelines from
the Proposer stage are included under “Transla-
tion Instruction,” while the drafts generated by the
Proposer are listed under “Translation Candidates.”
The Fusor leverages these translation candidates to
generate the final translation.
F Use of AI Assistants
During the preparation of this manuscript, the
authors used AI assistants, such as Gemini and
Claude, for proofreading, grammar correction, and
improving readability. All core ideas, implementa-
tions, experimental evaluations, and analyses were
developed by the authors.

System Prompt
### Task
Your task is to translate the historical documents into English while preserving their original meaning and context. The
documents were created in the Joseon era in Korea (from the late 14th century to the early 20th century) and were written in
Hanja, a traditional writing system used at the time.
### Guidelines
Focus on accurately conveying the meaning, nuances, and context of the original texts.
### Input
The document contains the original Hanja text, and may optionally include glossaries or examples of expert translations.
### Output JSON Format
{“english_translation”: str}
User Prompt
### Examples
[Example 1]
Hanja:上王幸壤,觀離宮之役而還.離宮乃壤縣古基,去都城四十里.
English: The abdicated king visited Pungyang, observed the construction of a detached palace, and returned. The detached ...
[Example 2]
Hanja:上王幸離宮南郊,觀放鷹.
English: The abdicated king went to the southern outskirts of the temporary palace and viewed falconry.
...
### Input Document
• hanja:上王幸毋岳,觀離宮造成之役.
• glossaries: [{“hanja”:“上王”, “korean”: “상왕”, “term”: “Abdicated King”}, {“hanja”: “毋岳”, “term”: “Muak”}, ...]
Figure 8: An example of the base prompt.
System Prompt
### Task
Your task is to translate the historical documents into English while preserving their original meaning and context. The
documents were created in the Joseon era in Korea (from the late 14th century to the early 20th century) and were written in
Hanja, a traditional writing system used at the time.
### Guidelines
Focus on accurately conveying the meaning, nuances, and context of the original texts.
### Input
The document contains the original Hanja text, its Korean translation, and may optionally include glossaries or examples of
expert translations.
### Output JSON Format
{“english_translation”: str}
User Prompt
### Examples
[Example 1]
Hanja:正朝使工曹判書崔閏德回自京師.
Korean:정조사인공조판서최윤덕이북경에서돌아왔다.
English: Minister of Works Choe Yundeok, envoy for offering the New Year’s greetings to the Emperor of Ming China ...
[Example 2]
Hanja:謝恩使星原君李正寧,副使刑曹左參判崔士儀,回自京師.
Korean:사은사성원군이정녕과부사형조좌참판최사의가북경으로부터돌아왔다.
English: The Envoy for Expressing Gratitude for Imperial Benevolence, Lord of Seongwon Yi Jeongnyeong, and ...
...
### Input Document
• Hanja:正朝使崔士儀,副使柳漢回自京師.
• Korean:정조사최사의·부사유한이북경에서돌아왔다.
• Glossaries: [{“hanja”: “正朝使”, “korean”: “정조사”, “term”: “envoy for the New Year’s greeting”},
{“hanja”: “崔士儀”, “term”: “Choe Saui”} ... ]
Figure 9: An example of the prompt format used in the Proposer.

System Prompt
### Task
Based on the provided User Input, fuse them into a better translation that meets the criteria. The source texts are Joseon-era
documents originally written in Hanja between the late 14th and early 20th centuries. You must create a fused translation
that faithfully preserves the original meaning, nuances, and historical context.
### User Input
• Examples: (reference material) Certified example translations.
• Input Document
–Hanja: The original Hanja text.
–Korean: Its Korean translation.
–Glossaries: (reference material) Glossaries to clarify specific words and characters.
• Translation Instruction: Additional instructions for translation.
• Translation Candidates: Multiple English translation candidates.
### Fusion Steps
1. Read all source materials and translation candidates.
2. Compare strengths and weaknesses.
3. Generate a fused final translation.
### Output JSON Format
{“final_translation”: str}
User Prompt
### Examples
[Example 1]
Hanja:正朝使工曹判書崔閏德回自京師.
Korean:정조사인공조판서최윤덕이북경에서돌아왔다.
English: Minister of Works Choe Yundeok, envoy for offering the New Year’s greetings to the Emperor of Ming China ...
[Example 2]
Hanja:正朝使都摠制柳殷之,副使漢城府尹沈道源,回自京師.
Korean:정조사도총제유은지와부사한성부윤심도원이북경으로부터돌아왔다.
English: Chief Commander General Yu Eunji, the envoy for offering the New Year’s greetings to the Emperor of Ming ...
...
### Input Document
• Hanja:正朝使崔士儀,副使柳漢回自京師.
• Korean:정조사최사의·부사유한이북경에서돌아왔다.
• Glossaries: [{“hanja”: “正朝使”, “korean”: “정조사”, “term”: “envoy for the New Year’s greeting”},
{“hanja”: “崔士儀”, “term”: “Choe Saui”} ... ]
### Translation Instruction
Task: Translate the historical documents into English while preserving their original meaning and context.
Guidelines: Focus on accurately conveying the meaning, nuances, and context of the original texts.
### Translation Candidates
• Candidate_1: Envoy for the New Year’s greeting Choe Saui and Vice-Envoy Yu Han returned from Beijing.
• Candidate_2: Choe Saui, envoy for the New Year’s greeting, and Yu Han, vice-envoy, returned from Beijing.
• Candidate_3: Choe Saui, envoy for offering the New Year’s greetings to the Emperor of Ming China, and Yu Han ...
...
Figure 10: An example of the prompt format used in the Fusor.