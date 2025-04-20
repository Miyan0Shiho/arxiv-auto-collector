# AutoStyle-TTS: Retrieval-Augmented Generation based Automatic Style Matching Text-to-Speech Synthesis

**Authors**: Dan Luo, Chengyuan Ma, Weiqin Li, Jun Wang, Wei Chen, Zhiyong Wu

**Published**: 2025-04-14 15:18:59

**PDF URL**: [http://arxiv.org/pdf/2504.10309v1](http://arxiv.org/pdf/2504.10309v1)

## Abstract
With the advancement of speech synthesis technology, users have higher
expectations for the naturalness and expressiveness of synthesized speech. But
previous research ignores the importance of prompt selection. This study
proposes a text-to-speech (TTS) framework based on Retrieval-Augmented
Generation (RAG) technology, which can dynamically adjust the speech style
according to the text content to achieve more natural and vivid communication
effects. We have constructed a speech style knowledge database containing
high-quality speech samples in various contexts and developed a style matching
scheme. This scheme uses embeddings, extracted by Llama, PER-LLM-Embedder,and
Moka, to match with samples in the knowledge database, selecting the most
appropriate speech style for synthesis. Furthermore, our empirical research
validates the effectiveness of the proposed method. Our demo can be viewed at:
https://thuhcsi.github.io/icme2025-AutoStyle-TTS

## Full Text


<!-- PDF content starts -->

AutoStyle-TTS: Retrieval-Augmented Generation
based Automatic Style Matching Text-to-Speech
Synthesis
Dan Luo1,∗,1Chengyuan Ma1,∗, Weiqin Li1, Jun Wang2,†, Wei Chen1, Zhiyong Wu1,†
1Shenzhen International Graduate School, Tsinghua University, Shenzhen, China
2AI Lab, Tencent, Shenzhen, China
{luod23,mcy23 }@mails.tsinghua.edu.cn
Abstract —With the advancement of speech synthesis technol-
ogy, users have higher expectations for the naturalness and
expressiveness of synthesized speech. But previous research
ignores the importance of prompt selection. This study proposes
a text-to-speech (TTS) framework based on Retrieval-Augmented
Generation (RAG) technology, which can dynamically adjust
the speech style according to the text content to achieve more
natural and vivid communication effects. We have constructed a
speech style knowledge database containing high-quality speech
samples in various contexts and developed a style matching
scheme. This scheme uses embeddings, extracted by Llama,
PER-LLM-Embedder,and Moka, to match with samples in the
knowledge database, selecting the most appropriate speech style
for synthesis. Furthermore, our empirical research validates the
effectiveness of the proposed method. Our demo can be viewed
at: https://thuhcsi.github.io/icme2025-AutoStyle-TTS
Index Terms —Text-to-Speech, Retrieval-Augmented Genera-
tion, Automatic Style Matching
I. I NTRODUCTION
Existing language model (LM) based TTS models [1]–[5]
have achieved highly realistic results in speech generation
and voice cloning, and podcast synthesis is one of the most
important application scenarios [6]. Whether it is storytelling
or interview dialogue, an engaging podcast is often accom-
panied by changes of subject, interactions between speakers,
and emotional fluctuations [7]. As a result, speech synthesis
technology must not only be able to control the speaker’s
timbre and style of speech, but also be able to dynamically
adapt the style of speech according to different contextual
situations in order to create more natural and immersive user
experience.
Several studies have begun to explore how to achieve
accurate and fine control over speech style, which can be
implemented through text instructions or by modeling speech
styles from speech prompts. V oxinstruct [8] uses text to
describe the style, specifying information such as timbre and
mood as textual prompts for overall speech style control;
CosyV oice [3], U-style [9] and StyleFusion [10] extract speech
style representations from speech prompts with specific style
∗Equal Contribution
†Corresponding Authorextractors, such as redesigned U-net. One key feature of these
methods is that the generation quality of these GPT-like gen-
erative models depends heavily on the selection and design of
the prompt [11]. However, all of these approaches only focus
on controlling the style of the synthesized speech, ignoring the
effect of incorrect stylistic cues on the expressiveness of the
synthesized speech. [12]. Especially in podcasting scenarios,
the rich variation of styles makes the correctness of style cues
more demanding, and the automatic selection of appropriate
styles can also reduce the extra cost of manual selection.
More specifically, previous work suffers from the following
problems in prompt selection: (i) Limited availability. Con-
ventional TTS [8] systems tend to model the style provided by
text or speech and then apply this style to all generated speech.
These approaches expose obvious limitations The traditional
approach makes it difficult to realise the flexibility of adjusting
the tone, pace and emotional expression when facing long text-
generated and dialogue-generated podcast scenarios. In addi-
tion, manually selecting the appropriate voice style for each
sentence is not only inefficient but also leads to inconsistencies
between the content and the actual style of expression. (ii)
Content and style disharmony. The modeled speech styles
of these LLM-based speech synthesis approaches [3], [9], [10]
do not effectively take into account the specific content of the
text as well as the relevance of the contextual scenarios. This
leads to a lack of natural variation and coherence between
synthesized speech.
In the field of prompt enhancement, Retrieval-Augmented
Generation(RAG) significantly improves the model’s ability
to handle complex queries and generate more accurate infor-
mation by introducing an external knowledge database and
comprehensively analyzing specific scenario information [13],
[14], which has been proven effective in the domains of text,
image and audio generation [15]–[18]. Inspired by this, and in
order to solve the problem of limited availability, we introduce
the RAG technique to allow the model to automatically select
the appropriate speech style prompt to guide the style of the
synthesized speech without the need for manual selection. The
application of RAG also greatly enhances the flexibility of the
speech styles so that the synthesised speech can be flexiblyarXiv:2504.10309v1  [cs.SD]  14 Apr 2025

Database
RAG
Style prompt
Style and Timbre
Decoupled TTS
Speech prompt
Generated
Speech
-User input
Style  embeddingFig. 1. Overall Architecture
varied according to different scenarios and needs.
In this paper, we propose a novel TTS framework,
AutoStyle-TTS, which addresses the previous system’s short-
comings in usability and the disconnect between content and
style. Considering the ability to automatically select matching
style prompts, we introduced RAG. Due to the limitations of
existing speech style knowledge databases and the need to
consider the correlation between text style and content, we
designed a method for constructing a speech style knowledge
database and style matching. This external knowledge database
contains high-quality speech samples in multiple contexts,
covering different emotional expressions, tone changes, and
other features. Our experimental results show that our method
significantly improves the usability of the speech generation
system and the coherence of the generated speech, indicating
that our approach has high value in practical applications.
II. METHODOLOGY
The details of our proposed RAG-based automatic style
matching approach are presented in the following sections,
including the overall architecture as well as its key modules:
the style and timbre decoupled TTS module, the construction
and retrieval of the speech style knowledge database, and the
PER-LLM-Embedder,an embedding extractor we designed and
trained.
A. overall architecture
The overall model architecture, as shown in Fig. 1 , mainly
includes three modules:
•Speech Style Knowledge Database: The knowledge
database is constructed from a text and speech corpus,
consisting of high-quality speech segments containing
different emotional expressions, tone changes, and so
on. Specifically, it consists of data pairs composed of
extracted style embedding and corresponding audio.
•RAG Style Selection Module: This module is responsible
for analyzing style information based on user query input.
It retrieves the most relevant audio data from the retrieval
knowledge database and organizes the output as a prompt
for the next module.
Flow Matching
LLM
Speech Tokenizer Text EncoderSpeaker
EncoderS TE
timbre prompt text style promptgenerated speechFig. 2. Style and Timbre Decoupled TTS architecture
•Style and Timbre Decoupled TTS Module: This module
receives style prompts from RAG Style Selection Module
as well as user-specified zero-shot timbre prompts. It
extracts the specified style and timbre to generate the
target speech.
B. Style and Timbre Decoupled TTS
In this module, as shown in Fig. 2, we used CosyV oice as
the backbone and modified the input of some of its modules.
In the frontend, The timbre information is provided by the
speaker embedding obtained through the Speaker Encoder,
CAM++ [19], while the content and style information are
provided by the speech tokens obtained through the speech
tokenizer. The speech generation task is modeled in two major
parts: LLM modeling and flow matching stage. The style
information is injected only in the LLM modeling stage, while
the speaker embedding is added in both parts to manage the
global vector information.
In the LLM modeling stage, the input sequence is
constructed as [(S), v,{t}i∈[1:I],(T),{x}k∈[1:K],(E)], where
(S),(T), and (E)represent the start of the sequence,
the transition point between text tokens and speech tokens,
and the end of the sequence, respectively. vrepresents the
speaker embedding vector extracted from the prompt speech
by the speaker encoder, which stores more timbre information;
{t}i∈[1:I]represents the tokens obtained from the text to be
synthesized by a text encoder; {x}k∈[1:K]represents the tokens
extracted from the speech by the speech tokenizer, which
stores more style features. The objective function for training
the LLM uses cross-entropy loss:
LLLM =−1
K+ 1K+1X
k=1logq(xk) (1)
where q(xk)represents the posterior probability of vipredicted
by the LLM and the softmax layer.
The flow matching part models the generation of speech
mel-spectrograms conditioned on the aforementioned gener-
ated speech token sequences using a flow matching model.

denoise diarization VAD scoreChinese
ASR1
Chinese
ASR2
english
ASR1
English
ASR2language
identificationChinese
EnglishSpeaker IDpassedfailedfilter by
WER
filter by
WERFig. 3. Speech Preprocess Pipeline
The input conditions for the flow matching module include
speaker embedding vectors, masked Mel-spectrogram features,
and speech token output by the LLM module. Given these
input conditions, the flow matching model is trained to match
its parameters to the vector field of the speech data distribu-
tion [20].
C. Speech Style Knowledge Database and User Retrieval
In order to enable the support system to flexibly adjust the
speech style based on the input text content and provide a
more natural and context-appropriate auditory experience, we
designed and built a speech style knowledge database, which
provides a solid foundation for RAG. This knowledge database
stores high-quality speech samples in various contexts, cover-
ing features such as different emotional expressions and tone
variations.
When constructing the retrieval knowledge base, prepro-
cessing and the type of embedding representation will directly
impact the retrieval efficiency and matching accuracy of the
system [21]. Good preprocessing can improve data consistency
and quality, reducing the impact of noise on retrieval results.
Choosing an appropriate embedder model can capture the
desired style information for matching, ensuring that the most
contextually appropriate speech style samples can be found
quickly and accurately. Therefore, we constructed the style
speech knowledge base according to the following steps:
Firstly, to improve retrieval accuracy and avoid issues
caused by overly long speech segments, we performed chunk-
ing on the raw data, dividing each speech sample into short
segments of 5 to 10 seconds. This chunking strategy not only
helps enhance retrieval precision but also better accommodates
text inputs of varying lengths. Specifically, our preprocessing
workflow is shown in Fig. 3, which includes noise reduction,
speaker diarization, and other processing. Then, we trim based
on V oice Activity Detection (V AD), and segments with scores
higher than a preset quality threshold are transcribed into cor-
responding text data through Automatic Speech Recognition
(ASR) and saved into structured data. Next, we chunk the text
and audio corpus into smaller blocks.
Next, we designed an embedding extraction module, which
consists of the Llama3.2,PER-LLM-Embedder and Moka,
which is a industry-proven LLM-embedder. The embedding
extraction and user retrieval data flow can be seen in the
Fig. 4. To maintain style consistency, we introduced character
profiling as global information, which is extracted by the
Llama3.2 from the all of synthesized text. To match thetext content with the speech style, we introduced situational
emotional information, which is analyzed by the PER-LLM-
Embedder from the text and character profiling information.
Considering that users may specify their desired style during
actual use, we introduced user preference information, which
is encoded by Moka based on factors such as age, gender,
and region. Finally, the style embedding is extracted for the
text content and other preference information. The embedding
consists of three parts, as shown in the formula:
Estyle=Eprofile +Eemotion +Euser (2)
Use the combination of these three embeddings for style
matching, we can maintain the coherence of speech, change
accordingly with the change of speaking context, and provide
greater usability of the system.
These speech style embeddings and speech clips are or-
ganized into a searchable database, which is indexed using
Milvus. The data used in our speech style database includes
part of the Microsoft EXPRESSO dataset [22], which has
expressive english speech and other high-quality expressive
Chinese speech. And ultimately, it consists of high-quality
speech clips, including 30 speakers and more than 2000 speech
segments.
In the retrieval stage, we first judge the user’s query to
determine whether retrieval is necessary. Then we proceed to
further retrieval enhancement processing. The way we carry
out retrieval enhancement processing is to select the top k
most relevant to the query from the pre-constructed knowledge
database, based on the similarity of embeddings. The specific
processing method is: firstly, we rewrite the query based
on the user’s input information; , secondly,same as database
construction,use the Llama, PER-LLM-Embedder, Moka to
get embedding. Lastly, we use Max Inner Product Search
(MIPS) [23] to calculate the similarity and obtain the final
top-K prompts.
D. PER-LLM-Embedder Training
Since we want to obtain embeddings that represent situ-
ational emotions, and currently LLM perform exceptionally
well in understanding the semantics and emotions of text, we
fine-tune LLaMA3.2 on a specific dataset. We utilized the
IEMOCAP [24] dataset, which consists of daily conversational
scenarios and scripted emotional performances from ten actors,
as well as the Multimodal Multiscene Multilabel Emotion
Dialogue (M3ED) [25] dataset in Chinese, comprising 990
binary emotional dialogue video clips extracted from 56 dis-
tinct television dramas (500 episodes in total). These dataset
have some dialogue text with speaker-id and the emotion
labels. During the fine-tuning stage, we input dialogue text and
character profiles into the model with the goal of predicting
emotion labels. Specifically, the English data for the character
profiles is generated by LLaMA 3.2, while the Chinese data is
generated by Qwen2.5. Regarding the implementation details,
we employed a learning rate of 5e−4, a dropout rate of 0.2,
and set the number of epochs to 3. Additionally, the local
context window size (w)was set to 5.

RetrievalDatabase
Construction
Speech Preprocess
PipelinePER-LLM
Embedder
Age,Gender,...
Moka
Moka
PER-LLM
Embedder
Eprofile
Eemotion
EuserEstyleText Speech
Age,Gender,...
Text to synthesized 
Estyle
Eprofile EemotionEuser
matching style speechLlama
LlamaFig. 4. Speech Style Knowledge Database Construction and Retrieval Details
III. EXPERIMENTS AND RESULTS
A. Experimental Setup
Evaluation metrics. Our evaluation metrics are divided into
objective and subjective evaluations. For objective metrics,
we considered five evaluation indicators: Speaker Similarity
(SIM), Word Error Rate (WER), Virtual Speech Quality Ob-
jective Listener(VISQOL), KL divergence(KL), and Inception
score(IS). SIM evaluates the similarity between the generated
audio and the reference audio. WER calculates the word error
rate between the transcribed text of the generated audio and
the reference text. VISQOL is an objective, full-reference
metric for perceived audio quality. KL quantifies the difference
between generated speech and ground truth; IS evaluates the
diversity and realism of speech generated by a model.
For subjective metrics, we conducted two Mean Opinion
Score (MOS) subjective tests: Style Matching MOS(SM-
MOS) and Style Coherence MOS(SC-MOS). SM-MOS eval-
uates the style matching degree between the text and the syn-
thesized speech, while SC-MOS is used to compare whether
the overall style of the synthesized speech is harmonious and
whether the emotional fluctuations are appropriate. Addition-
ally, we set up an AB test to evaluate the effectiveness of our
method compared to manual selection.
B. Style and Timbre Decouple Experiment
To demonstrate that our model can transform speech styles
without affecting its ability to generate accurate timbre and
Fig. 5. Different Style Prompt Speech Synthesis Result Visualization
word correctness, we employ 1,00 samples from the Common
V oice dataset and the DiDiSpeech-2 dataset as the test set.
For Cosyvoice, we generate four sentences for each sample;
for our method, we generate four different style results for
each sample (visualization of the different style speech results
can be seen in Fig. 5). We use WER, SIM, VISQOL, KL and
IS metrics for evaluation.
TABLE I
STYLE AND TIMBRE DECOUPLE EXPERIMENT RESULT
Model metrics
name SIM↑ WER(%) ↓ VISQOL ↑ KL↓ IS↑
CosyV oice [3] 0.753 2.312 4.083 0.165 1.007
Ours 0.750 3.600 4.075 0.160 1.325

Fig. 6. AB Test Result
As illustrated in TABLE I ,the results indicate that our
method can achieve style control while maintaining timbre and
speech quality. Furthermore, the IS metric score demonstrates
that our method can generate results with more varied styles
and possesses more flexible generation capabilities.
C. RAG Impact Evaluation
We first conducted a subjective evaluation of the backbone
module of the TTS system, CosyV oice [3], another state-
of-the-art model in the TTS field, MaskGCT [26], and our
method, using the SM-MOS and SC-MOS metrics and inviting
15 volunteers to participate in the evaluation. The test set
was divided based on language, with data for each language
12 audio samples covering both story narration and dialogue
scenarios. For each sample, CosyV oice used the generated
speech from the previous sentence as the speech prompt for
the next sentence.
TABLE II
RAG IMPACT SUBJECTIVE EVAL
Model Language Metrics
name SM-MOS ↑ SC-MOS ↑
CosyV oice [3] 3.35±0.13 3.48±0.13
MaskGCT [26] English 3.85±0.13 3.81±0.12
AutoStyle-TTS(Ours) 3.85±0.13 3.81±0.12
CosyV oice [3] 3.38±0.14 3.51±0.14
MaskGCT [26] Chinese 3.85±0.13 3.81±0.12
AutoStyle-TTS(Ours) 3.90±0.12 3.83±0.13
According to the experimental results, our method signifi-
cantly outperformed Cosyvoice on both key metrics, indicating
that our technology has made substantial progress in text-style
matching and overall speech coherence.
Secondly, the evaluation also included an AB preference
test, where participants were asked to compare the speech
styles selected by our method with those chosen manually,
according to a certain requirement. Specifically, we per-
formed speech synthesis for 10 text passages(include story-
telling and dialogue) using both manually selected and model-
automatically selected speech styles. Thirty participants were
asked to choose their preferred samples. According to the
results of the AB test in Fig. 6, our method is comparable
to the manually selected prompt method in terms of user
preference. The experimental results demonstrate that our
proposed automated style matching mechanism can effectively
replace the time-consuming manual selection process whileensuring that the synthesized speech’s style is highly consistent
with the content, thereby providing a natural and coherent
auditory experience.
D. Ablation Experiment
To validate the effectiveness of the character profiling
and situational emotion in the style embeddings extracted
by Llama and our PER-LLM-Embedder, we conducted a
series of ablation experiments. The metrics, number of test
participants, and number of test samples per group are the
same as III-C. The profile-only and emotion-only represent
the cases where only character profiling or situational emotion
was used for similarity calculation to match the style prompt.
The K represents the number of retrieved style prompts, and
we concat the style prompts together and enter them into the
subsequent TTS module.
TABLE III
ABLATION EXPERIMENTS FOR EMBEDDING AND TOP-K
Ablation Retrieval metrics
Module Method SM-MOS ↑ SC-MOS ↑
only-profile 3.40±0.12 2.91±0.13
Embedding only-emotion 3.20±0.12 3.60±0.14
profile+emotion(ours) 3.85±0.13 3.81±0.12
K = 1 3.65±0.14 3.61±0.13
Top-K K = 3 3.85±0.12 3.81±0.13
K= 5 3.75±0.13 3.61±0.14
The results indicate that the model achieves the best per-
formance when matching with both character profiles and
situational emotions. The method that uses only character
profile embeddings for matching shows a decline in the SM-
MOS metric. This is because this method loses the situational
emotional information of individual sentences, making it dif-
ficult to match the emotions of the text with those of the
synthesized speech, leading to a decrease in the style matching
score. Also, This approach shows a decline in the SC-MOS
metric. This is because the loss of emotional matching leads
to disjointed transitions in the overall emotional fluctuations.
For the approach that uses only situational emotions, the main
decline is observed in the SC-MOS metric. The absence of
comprehensive character profile information for control results
in some loss of overall coherence. However, the decline is
not significant because the situational emotional information
is extracted with the aid of character profile information,
which carries some global information, helping to maintain
the coherence of the speech.
To evaluate the impact of the number of speech prompts
on the RAG method for speech style, we conducted ablation
experiments, as shown in the TABLE III. We found that as
the number of prompts increased, the speaker similarity also
gradually increased. This is because the TTS system attempts
to clone the speaking style, and longer style prompts provide
more appropriate style information. However, the results also
indicate that performance peaks at a prompt length of 3 and
diminishes with further increases in prompt length. We believe
that longer, inconsistent style prompts from different sources

may hinder the TTS system’s ability to generate coherent
speech.
IV. CONCLUSION
In this paper, we have developed a novel TTS framework
that integrates RAG technology. This framework is designed
to dynamically adjust speech styles in response to textual
prompts and user preference, resulting in a more natural and
engaging auditory experience. We construct a comprehensive
speech style database, combined with our style matching
scheme and embedding extractor, allowing for the precise
selection and application of speech styles that align with
the contextual requirements of the input text. Our empirical
evaluations have demonstrated that our get excellent effect in
terms of the style matching between text and speech and the
style coherence in speech.
V. A CKNOWLEDGMENT
This work is supported by National Natural Science Foun-
dation of China (62076144) and Shenzhen Science and Tech-
nology Program (JCYJ20220818101014030).
REFERENCES
[1] Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou,
Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei
He, Sheng Zhao, and Furu Wei, “Neural codec language models are
zero-shot text to speech synthesizers,” 2023.
[2] Tianxin Xie, Yan Rong, Pengfei Zhang, and Li Liu, “Towards control-
lable speech synthesis in the era of large language models: A survey,”
2024.
[3] Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang,
Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, et al., “Cosyvoice:
A scalable multilingual zero-shot text-to-speech synthesizer based on
supervised semantic tokens,” arXiv preprint arXiv:2407.05407 , 2024.
[4] Dongchao Yang, Rongjie Huang, Yuanyuan Wang, Haohan Guo, Dading
Chong, Songxiang Liu, Xixin Wu, and Helen Meng, “Simplespeech 2:
Towards simple and efficient text-to-speech with flow-based scalar latent
transformer diffusion models,” 2024.
[5] Dongchao Yang, Songxiang Liu, Rongjie Huang, Chao Weng, and Helen
Meng, “Instructtts: Modelling expressive tts in discrete latent space with
natural language style prompt,” 2023.
[6] Detai Xin, Sharath Adavanne, Federico Ang, Ashish Kulkarni, Shinno-
suke Takamichi, and Hiroshi Saruwatari, “Improving speech prosody of
audiobook text-to-speech synthesis with acoustic and textual contexts,”
inICASSP 2023 - 2023 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP) , 2023, pp. 1–5.
[7] W Lance Bennett and Alexandra B Segerberg, “The logic of connective
action: Digital media and the personalization of contentious politics,”
Information, Communication & Society , vol. 15, no. 5, pp. 739–760,
2013.
[8] Yixuan Zhou, Xiaoyu Qin, Zeyu Jin, Shuoyi Zhou, Shun Lei, Songtao
Zhou, Zhiyong Wu, and Jia Jia, “V oxinstruct: Expressive human
instruction-to-speech generation with unified multilingual codec lan-
guage modelling,” in ACM Multimedia 2024 , 2024.
[9] Tao Li, Zhichao Wang, Xinfa Zhu, Jian Cong, Qiao Tian, Yuping Wang,
and Lei Xie, “U-style: Cascading u-nets with multi-level speaker and
style modeling for zero-shot voice cloning,” IEEE/ACM Transactions
on Audio, Speech, and Language Processing , 2024.
[10] Zhiyong Chen, Xinnuo Li, Zhiqi Ai, and Shugong Xu, “Stylefusion
tts: Multimodal style-control and enhanced feature fusion for zero-shot
text-to-speech synthesis,” in Chinese Conference on Pattern Recognition
and Computer Vision (PRCV) . Springer, 2024, pp. 263–277.
[11] Haoyu Wang, Chunyu Qiang, Tianrui Wang, Cheng Gong, Qiuyu Liu,
Yu Jiang, Xiaobao Wang, Chenyang Wang, and Chen Zhang, “Emopro:
A prompt selection strategy for emotional expression in lm-based speech
synthesis,” 2024.[12] Thomas Bott, Florian Lux, and Ngoc Thang Vu, “Controlling emotion
in text-to-speech with natural language prompts,” 2024.
[13] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi,
Yi Dai, Jiawei Sun, and Haofen Wang, “Retrieval-augmented generation
for large language models: A survey,” arXiv preprint arXiv:2312.10997 ,
2023.
[14] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike Lewis, Wen tau Yih,
Tim Rockt ¨aschel, Sebastian Riedel, and Douwe Kiela, “Retrieval-
augmented generation for knowledge-intensive nlp tasks,” 2021.
[15] Philipp Christmann and Gerhard Weikum, “Rag-based question answer-
ing over heterogeneous data and text,” 2024.
[16] Yi Yuan, Haohe Liu, Xubo Liu, Qiushi Huang, Mark D. Plumbley, and
Wenwu Wang, “Retrieval-augmented text-to-audio generation,” 2024.
[17] Ho-Young Choi, Won-Gook Choi, and Joon-Hyuk Chang, “Retrieval-
augmented classifier guidance for audio generation,” in Interspeech
2024 , 2024, pp. 3310–3314.
[18] Jinlong Xue, Yayue Deng, Yingming Gao, and Ya Li, “Retrieval
augmented generation in prompt-based text-to-speech synthesis with
context-aware contrastive language-audio pretraining,” arXiv preprint
arXiv:2406.03714 , 2024.
[19] Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, and Qian Chen,
“Cam++: A fast and efficient network for speaker verification using
context-aware masking,” 2023.
[20] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel,
and Matt Le, “Flow matching for generative modeling,” 2023.
[21] Jie Yang, Thomas Drake, Andreas Damianou, and Yoelle Maarek,
“Leveraging crowdsourcing data for deep active learning an application:
Learning intents in alexa,” in Proceedings of the 2018 World Wide Web
Conference , Republic and Canton of Geneva, CHE, 2018, WWW ’18, p.
23–32, International World Wide Web Conferences Steering Committee.
[22] Tu Anh Nguyen, Wei-Ning Hsu, Antony D’Avirro, Bowen Shi, Itai
Gat, Maryam Fazel-Zarani, Tal Remez, Jade Copet, Gabriel Synnaeve,
Michael Hassid, Felix Kreuk, Yossi Adi, and Emmanuel Dupoux,
“Expresso: A benchmark and analysis of discrete expressive speech
resynthesis,” 2023.
[23] Alex Auvolat, Sarath Chandar, Pascal Vincent, Hugo Larochelle, and
Yoshua Bengio, “Clustering is efficient for approximate maximum inner
product search,” 2015.
[24] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily
Mower, Samuel Kim, Jeannette N Chang, Sungbok Lee, and Shrikanth S
Narayanan, “Iemocap: Interactive emotional dyadic motion capture
database,” Language resources and evaluation , vol. 42, pp. 335–359,
2008.
[25] Jinming Zhao, Tenggan Zhang, Jingwen Hu, Yuchen Liu, Qin Jin, Xin-
chao Wang, and Haizhou Li, “M3ED: Multi-modal multi-scene multi-
label emotional dialogue database,” in Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers) , Smaranda Muresan, Preslav Nakov, and Aline Villavi-
cencio, Eds., Dublin, Ireland, May 2022, pp. 5699–5710, Association
for Computational Linguistics.
[26] Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian
Guo, Jiachen Zheng, Qiang Zhang, Xueyao Zhang, Shunsi Zhang,
and Zhizheng Wu, “Maskgct: Zero-shot text-to-speech with masked
generative codec transformer,” arXiv preprint arXiv:2409.00750 , 2024.