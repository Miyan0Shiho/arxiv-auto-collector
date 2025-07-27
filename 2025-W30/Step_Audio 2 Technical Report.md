# Step-Audio 2 Technical Report

**Authors**: Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, Mingrui Chen, Peng Liu, Wang You, Xiangyu Tony Zhang, Xingyuan Li, Xuerui Yang, Yayue Deng, Yechang Huang, Yuxin Li, Yuxin Zhang, Zhao You, Brian Li, Changyi Wan, Hanpeng Hu, Jiangjie Zhen, Siyu Chen, Song Yuan, Xuelin Zhang, Yimin Jiang, Yu Zhou, Yuxiang Yang, Bingxin Li, Buyun Ma, Changhe Song, Dongqing Pang, Guoqiang Hu, Haiyang Sun, Kang An, Na Wang, Shuli Gao, Wei Ji, Wen Li, Wen Sun, Xuan Wen, Yong Ren, Yuankai Ma, Yufan Lu, Bin Wang, Bo Li, Changxin Miao, Che Liu, Chen Xu, Dapeng Shi, Dingyuan Hu, Donghang Wu, Enle Liu, Guanzhe Huang, Gulin Yan, Han Zhang, Hao Nie, Haonan Jia, Hongyu Zhou, Jianjian Sun, Jiaoren Wu, Jie Wu, Jie Yang, Jin Yang, Junzhe Lin, Kaixiang Li, Lei Yang, Liying Shi, Li Zhou, Longlong Gu, Ming Li, Mingliang Li, Mingxiao Li, Nan Wu, Qi Han, Qinyuan Tan, Shaoliang Pang, Shengjie Fan, Siqi Liu, Tiancheng Cao, Wanying Lu, Wenqing He, Wuxun Xie, Xu Zhao, Xueqi Li, Yanbo Yu, Yang Yang, Yi Liu, Yifan Lu, Yilei Wang, Yuanhao Ding, Yuanwei Liang, Yuanwei Lu, Yuchu Luo, Yuhe Yin, Yumeng Zhan, Yuxiang Zhang, Zidong Yang, Zixin Zhang, Binxing Jiao, Daxin Jiang, Heung-Yeung Shum, Jiansheng Chen, Jing Li, Xiangyu Zhang, Yibo Zhu

**Published**: 2025-07-22 14:23:55

**PDF URL**: [http://arxiv.org/pdf/2507.16632v2](http://arxiv.org/pdf/2507.16632v2)

## Abstract
This paper presents Step-Audio 2, an end-to-end multi-modal large language
model designed for industry-strength audio understanding and speech
conversation. By integrating a latent audio encoder and reasoning-centric
reinforcement learning (RL), Step-Audio 2 achieves promising performance in
automatic speech recognition (ASR) and audio understanding. To facilitate
genuine end-to-end speech conversation, Step-Audio 2 incorporates the
generation of discrete audio tokens into language modeling, significantly
enhancing its responsiveness to paralinguistic information such as speaking
styles and emotions. To effectively leverage the rich textual and acoustic
knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented
generation (RAG) and is able to call external tools such as web search to
mitigate hallucination and audio search to switch timbres. Trained on millions
of hours of speech and audio data, Step-Audio 2 delivers intelligence and
expressiveness across diverse conversational scenarios. Evaluation results
demonstrate that Step-Audio 2 achieves state-of-the-art performance on various
audio understanding and conversational benchmarks compared to other open-source
and commercial solutions. Please visit
https://github.com/stepfun-ai/Step-Audio2 for more information.

## Full Text


<!-- PDF content starts -->

Step-Audio 2 Technical Report
StepFun Audio Team
Abstract
This paper presents Step-Audio 2, an end-to-end multi-modal large language model
designed for industry-strength audio understanding and speech conversation. By
integrating a latent audio encoder and reasoning-centric reinforcement learning
(RL), Step-Audio 2 achieves promising performance in automatic speech recognition
(ASR) and audio understanding. To facilitate genuine end-to-end speech conversa-
tion, Step-Audio 2 incorporates the generation of discrete audio tokens into language
modeling, significantly enhancing its responsiveness to paralinguistic information
such as speaking styles and emotions. To effectively leverage the rich textual and
acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented
generation (RAG) and is able to call external tools such as web search to mitigate
hallucination and audio search to switch timbres. Trained on millions of hours of
speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across
diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2
achieves state-of-the-art performance on various audio understanding and conversa-
tional benchmarks compared to other open-source and commercial solutions. Please
visit https://github.com/stepfun-ai/Step-Audio2 for more information.
1 Introduction
With the rapid development of large language models and audio processing technology, large
audio language models (LALMs) have demonstrated their superiority over conventional approaches
in various speech and audio processing tasks. GPT-4o is first introduced and is pioneering the
development of end-to-end speech interaction without intermediate textual conversions. Subse-
quently, many open-sourced LALMs [9, 13, 16, 18, 21, 31, 32, 49, 71, 72, 74, 77] are emerged,
advancing multi-modal large language model capabilities in various speech and audio domains.
Among these approaches, Qwen-Audio [12] and Qwen2-Audio [13] perform audio analysis and
generate textual responses to speech instructions. Qwen2.5-Omni [74] implements a thinker-talker
architecture to enable full-duplex I/O during speech conversations. More recently, Kimi-Audio [18]
has achieved impressive results on multiple speech and audio understanding benchmarks. In parallel,
we have introduced Step-Audio [32] and Step-Audio-AQAA [31], the first LALMs to unify speech
understanding and generation through discrete audio tokens at a scale of 130 billion parameters.
However, existing LALMs still face challenges in achieving natural and intelligent speech interaction.
Previous LALMs such as Spirit LM [49] and GLM-4-V oice [77] mainly focus on aligning the
semantic information in speech inputs to text modal, neglecting the para-linguistic information
which is also crucial for intentional understanding. Although LALMs including Qwen-Audio [12],
Qwen2-Audio [13] and Audio Flamingo series [24, 25, 43] are capable of comprehending such
information, they typically generate only textual outputs and fail to further utilize this capabilityarXiv:2507.16632v2  [cs.CL]  24 Jul 2025

Step-Audio 2 Technical Report
AISHELL-2
LibriSpeech
test-clean
MMAU Speech
MMAU Sound
MMAU Music
StepEval-Audio-ParalinguisticCoVoST 2
(S2TT, en<->zh)CVSS
(S2ST, en<->zh)URO-Bench
Basic-zhURO-Bench
Basic-en
95.797.397.697.9
98.298.5
97.198.8
64.665.5
70.675.7
58.0
79.078.1
82.0
51.8
64.4
65.9
74.643.5
49.644.2
76.529.6N/A
35.4
38.823.715.3
27.974.2
70.562.178.984.5
60.070.679.0
GPT-4o Audio Kimi-Audio Qwen-Omni Step-Audio 2
Figure 1: Performance comparision of GPT-4o Audio1, Kimi-Audio2, Qwen-Omni3and Step-Audio 2 and on various
benchmarks.
to produce coherent and expressive responses in speech conversations. Moreover, due to the
complexities of multi-modal modeling, existing LALMs frequently suffer from hallucination and
offer limited choices of timbres and speaking styles [16, 18], lacking access to real-world textual
and acoustic knowledge.
To address these issues and step into the next generation of multi-modal large language models,
we present Step-Audio 2 , an end-to-end large audio language model with industry-strength audio
perception and speech interaction. Step-Audio 2 directly processes raw audio as input and outputs
discrete text and audio tokens and has fewer parameters than Step-Audio [32]. Beyond capturing
semantic information in speech, the model also comprehends para-linguistic and non-vocal informa-
tion in audio. By leveraging chain-of-thought (CoT) reasoning and reinforcement learning (RL),
Step-Audio 2 further utilizes such multi-modal information to generate expressive speech responses
coherent to different conversation scenarios. To ground the model with real-world knowledge, Step-
Audio 2 incorporates retrieval-augmented generation (RAG) and the capability to utilize various
external tools, including web search and audio search, to provide more reliable and expressive
responses. Specifically, we present an audio search as a tool unique to LALMs, enabling seamless
speech retrieval via voice instructions and allowing the model to switch timbres and speaking styles
based on the retrieved speech.
1GPT-4o Audio is evaluated with gpt-4o-transcribe for ASR and gpt-4o-audio-preview-2025-06-03 for others via official API.
2Kimi-Audio is excluded from translation evaluations since it consistently ignores prompts.
3Qwen-Omni is evaluated with Qwen2.5-Omni for MMAU and speech-to-text translation, and qwen-omni-turbo-2025-03-26 for
others via official API.
2

Step-Audio 2 Technical Report
4’’Could you talk like a little girl?
Yay! Now I'm a happy little girl! What do you wannatalk about?10’’
👧
4’’Could you talk like a grandpa?
Alright then, let this old man have a chat with ya.What’s up?10’’
👴
4’’Tell me about the Great Wall.
Oh, the Great Wall is like a giant dragon...10’’
👴
4’’Answer in the default voice,what is the capital of China?
The capital of China is Beijing.10’’
4’’(Shanghaidialect) What’s your name?My name is Step-Audio 2.10’’
4’’(Sichuandialect) Which company developed you?I'm developed by StepFun.10’’
4’’What day is it today?Today is Thursday.10’’
📅
4’’Please check today’s S&P 500 Index.Today’s S&P 500 Indexis6297.36.10’’
🔍
4’’Translate, never gonnagive you up, never gonnalet you down.
永远不会放弃你，永远不会让你失望。10’’
4’’请翻译一下，长城是一座伟大的建筑。
The Great Wall is a magnificent structure.10’’
4’’Could you say that sadly?
The Great Wall is a magnificent structure.10’’
😭
4’’Please say it in a whisper.
The Great Wall is a magnificent structure.10’’
🤫
10’’Good evening… I’m feeling a bit unsettled.I wonder if I could talk to you.
Ofcourse,I’mhereforyou.Itsoundslikeyou’recarryingalotofemotionsrightnow—thatcanfeeloverwhelming.Wouldyouliketosharewhat’sonyourmind,orjusttakeamomenttounwind?There’snorush;I’lllistenwheneveryou’reready.25’’
Youngmale,20-25yearsold,speakinginagentleandslightlythoughtfultone,withasteadyandnaturalpace,slightbackgroundnoise,usingstandardMandarin.Alright,theuserhascometomenow,saying“Goodevening,I‘mfeelingabitunsettled,andI’dliketotalk.”First,Ineedtoshowunderstandingandacceptancetomakehimfeelsafe.Hemaybeinastateofanxietyorunease,somyresponseshouldbegentle,patient,andgivehimamplespacetoexpresshimself.Theusermentionedfeeling“abitunsettled,”whichsuggestshemayhavemanyemotionstangledtogetherandisn’tsurewheretostart.Ineedtoguidehiminslowlyuntanglingtheseemotionsratherthanrushingtosolvetheproblem.Atthispoint,empathyiskey—lettinghimknowhisfeelingsarevalid.Finally,Imustensurenottoendtheconversationprematurely—lettheusertaketheleadentirely.Ifhedoesn’twanttodelvedeeperfornow,respecthispace,offersupport,andwait.Maintainasoftandcaringtone,makinghimfeelheardandsupported,sohe’smorewillingtoopenupfurther.Thinking
Figure 2: Illustration of the applications of Step-Audio 2 across various speech conversation scenarios.
To ensure its intelligence and expressiveness in diverse conversational scenarios, we carefully design
a multi-stage training strategy to train Step-Audio 2 on 680 billion tokens of text data and 8 million
hours of real and synthesized audio data. Evaluation results shown in Figure 1 demonstrate that
Step-Audio 2 achieves state-of-the-art performance in a series of audio tasks, including automatic
speech recognition (ASR) on multiple languages, audio understanding, speech-to-speech translation
and speech-to-speech conversation. Typical usages of Step-Audio 2 are also illustrated in Figure 2.
3

Step-Audio 2 Technical Report
2 Related Work
2.1 Speech and audio understanding
Recent advances in large language models (LLMs) [4, 29, 50, 51] have extended their application
to a wide range of speech and audio understanding tasks, such as audio captioning, sound event
detection, automatic speech recognition, audio classification, and audio-driven creative generation.
A prevalent approach [13, 17, 27, 28, 48, 61] involves pairing speech encoders with lightweight,
trainable adapters that project audio features into a textual embedding space compatible with LLMs.
Building on this foundation, recent studies have further explored how to incorporate paralinguistic
information such as emotion, intonation and speaker style, enabling LLMs to move beyond pure
linguistic comprehension. For instance, ParalinGPT [48] focuses on enhancing a powerful text-based
language model by integrating continuous speech embeddings, enabling it to capture paralinguistic
signals such as emotion and prosody. SALMONN [61] adopts a multi-modal strategy by freezing
speech encoders Whisper [53] and BEATs [10], and connecting their outputs to an LLM via a
window-level Q-Former, enabling joint modeling of linguistic and acoustic features. Seed-ASR [5]
integrates LUISE-based speech representations with instructions and context, using context-aware
SFT to capture semantic information. AudioPaLM [57] combines PaLM-2 [2] and AudioLM [7],
unifying linguistic knowledge with paralinguistic features like speaker identity and intonation. LLM-
based approaches [12, 13] increasingly rely on pretrained audio encoders such as Wav2Vec [3],
HuBERT [30], Whisper [53], and WavLM [11] to extract rich semantic representations from speech.
At the same time, the extensive text knowledge and contextual reasoning capabilities stored in
LLMs can provide valuable semantic guidance for understanding tasks.
2.2 Text-to-speech synthesis
Text-to-Speech (TTS) technology has made remarkable strides in recent years, evolving from
traditional concatenative and statistical parametric approaches [52, 55, 59, 68] to codec-based TTS
systems. Codec language models leverage a speech codec to extract discrete representations of
speech [15, 16, 36, 60, 73, 76, 81, 82] and utilize either autoregressive [18, 80] or masked language
models [67] to predict the corresponding speech tokens. These tokens are then synthesized into
waveforms using codec vocoders. V ALL-E [64] marked a significant breakthrough in this area. It
uses an autoregressive model to generate coarse codec codes, followed by a non-autoregressive
model for the fine codes. Unlike V ALL-E, which predicts acoustic tokens from phonemes and
requires transcripts, SPEAR-TTS [40] uses a two-stage architecture with self-supervised audio
prompts to clone unseen voices from just 3 seconds of speech. SparkTTS [65] introduces BiCodec, a
single-stream speech codec that encodes linguistic content as compact semantic tokens and speaker
characteristics as fixed-length global tokens. Instead of relying on non-autoregressive models to
predict residual discrete codes, methods like TorToiseTTS [6], CosyV oice [20], CosyV oice 2 [19],
MiniMax-Speech [79] and SEED-TTS [1] adopt diffusion or flow-matching techniques as a second
stage to reconstruct mel-spectrograms or continuous representations enriched with fine-grained
acoustic and semantic details. Recent work, Kimi-Audio [18] combines Whisper features and
semantic tokens for efficient modeling, with dual heads and a flow-matching detokenizer plus
BigVGAN [46] for low-latency, expressive synthesis.
4

Step-Audio 2 Technical Report
2.3 Speech-to-speech translation
Speech-to-speech translation (S2ST) is a crucial technology for eliminating communication bar-
riers across languages. Traditional S2ST systems [62, 70] typically adopt a cascaded pipeline
consisting of automatic speech recognition (ASR), machine translation (MT), and TTS modules.
Earlier studies [38, 39, 44, 45] have shifted toward direct approaches that bypass intermediate
textual representations, aiming for lower latency and better preservation of prosody and speaker
characteristics. Two main types of direct S2ST methods have emerged, which are known as speech-
to-spectrogram translation and speech-to-unit translation. Both directly generate target speech
representations from the source speech without relying on textual transcriptions. A representative of
the former is Translatotron [38], the first end-to-end model to translate source speech directly into
target spectrograms. Translatotron 2 [39], further improves translation quality through a two-pass
decoding mechanism [45]. In contrast, speech-to-unit models predict discrete acoustic tokens rather
than spectrograms, which are typically extracted using self-supervised speech encoders such as
HuBERT [30] or WavLM [11]. For instance, TransVIP [44] employs a joint encoder-decoder
architecture that first generates target text and residual vector quantization (RVQ) codes in the initial
layer, followed by a non-causal language model that refines RVQ predictions in subsequent layers.
2.4 Speech-to-text and speech-to-speech conversation
Based on whether the LLM can directly understand and generate speech representations, existing
systems can be categorized into end-to-end large audio language models and cascaded large audio
language models. The former directly models audio inputs and outputs within a unified framework,
while the latter relies on a modular pipeline involving separate ASR, LLM, and TTS components.
Traditional speech-to-text and speech-to-speech systems typically adopt a cascaded architecture, as
exemplified by AudioGPT [33] and Spoken-LLM [47]. However, the ASR + LLM + TTS pipeline
incurs high latency and modular mismatches. This has spurred interest in unified end-to-end
architectures for faster and more seamless integration. A major milestone in this direction is GPT-
4o [34], which supports direct end-to-end speech interaction without requiring intermediate textual
conversions. Recently, several new end-to-end LALMs [16, 21, 71, 72, 74] for speech-to-speech
conversation have emerged. For instance, Moshi [16] improves efficiency with an RQ-Transformer
that generates text and audio tokens simultaneously. Similarly, Mini-Omni [71] generates speech
and text responses in parallel, following a strategy similar to MusicGen [14], which enables lower
first-token latency compared to interleaved generation designs. LUCY [22] builds on the Mini-Omni
architecture with enhancements for emotional expressiveness, naturalness, and informativeness
in speech generation. It utilizes curated synthetic data and optimizes the training and decoding
pipelines to handle multi-turn dialogue and function-call scenarios. Mini-Omni2 [71] further
extends Mini-Omni framework by integrating multimodal understanding and full-duplex interaction
capabilities. LLaMA-Omni [21] introduces a streaming, non-autoregressive speech decoder based
on Connectionist Temporal Classification, enabling direct and efficient generation of discrete audio
tokens without relying on step-by-step prediction. Freeze-Omni [66], on the other hand, freezes the
LLM parameters during training, preserving its original capabilities while achieving low-latency
speech-to-speech interaction through streaming and decoder integration. Qwen2.5-Omni [74]
supports multimodal input and simultaneous text-speech output via a thinker-talker architecture,
using TMRoPE for improved audio-visual synchronization through explicit temporal encoding.
5

Step-Audio 2 Technical Report
AudioDetokenizer
❄Output text-audio interleaved tokens…AudioEncoderAdaptor
❄Input audio features
🔥
Input audioOutput speechLLM DecoderHistory information
🔥…Discrete text tokenDiscrete audiotokenLatent audio feature………
Figure 3: Architecture of the Step-Audio 2.
3 Methodology
3.1 Architecture
Different from our previous Step-Audio [32], Step-Audio 2 further integrates the generation of
audio tokens into language modeling, achieves end-to-end audio perception and generation. As
shown in Figure 3, Step-Audio 2 consists of an audio encoder, an audio adaptor, an LLM decoder
and an audio detokenizer.
The audio encoder is pretrained on various speech and audio understanding tasks including ASR,
speaker age and gender prediction, audio event detection, etc. The audio encoder has an output
frame rate of 25 Hz and is frozen during the entire training process. An audio adaptor with a
downsampling rate of 2 is employed to connect the audio encoder to LLM, thereby reducing the
output frame rate of the audio encoder to 12.5 Hz.
The LLM decoder directly takes the latent audio features from the audio adaptor as input, and
outputs an interleaved sequence of discrete text and audio tokens. We employ the tokenizer from
CosyV oice 2 [19] as the audio tokenizer. And the text and audio tokens are interleaved [31, 32, 77]
at a fixed ratio and padded at the end to meet the ratio. The audio tokens are then extracted from the
interleaved sequence and consumed by the audio detokenizer to generate the output waveform. The
input audio features and output interleaved sequences are then pre-filled as the history information
for the next round of conversation.
To provide more accurate responses and expand interactive capabilities, we design tools to retrieve
audio, current date and time, weather forecast and web content directly with explicit or implicit
voice inputs. Notably, we propose the audio search tool, a novel tool with a voice library of hundreds
of thousands of speeches with their corresponding transcriptions and descriptions. With the retrieved
speech from audio search, Step-Audio 2 is able to mimic the speaking style or switching timbre
6

Step-Audio 2 Technical Report
according to the speech. During inference, the retrieved information is appended after the input
audio features before generating speech outputs.
Similar to Step-Audio [32] and Step-Audio-AQAA [31] , Step-Audio 2’s audio detokenizer also
consists of a Flow Matching module and a HiFi-GAN [42] vocoder. The Flow-Matching module
generates Mel spectrograms from the output audio tokens, while the vocoder further converts the
Mel spectrograms into waveforms. For Flow-Matching, we incorporate a CNN-based encoder layer
after each self-attention module within the transformer block and train the model on 200,000 hours
of high-quality speech. This enhancement significantly improves its Mel spectrogram reconstruction
capability, leading to substantial gains in both pronunciation accuracy and timbre similarity.
Step-Audio 2 employs the same deployment infrastructure used in Step-Audio [32] and Step-Audio-
AQAA [31], which includes a voice activity detection (V AD) module to filter out input speeches
and achieves real-time voice conversation.
3.2 Pre-training
Step-Audio 2 model is initialized with a textual LLM and then continually pre-trained on 1.356T
tokens of textual and audio data over 21 days.
We first utilize 100B tokens of ASR data to facilitate effective alignment between speech and text
feature spaces within the adaptor. During this phase, both the audio encoder and LLM are frozen,
with only the adaptor being trained. We conduct training for 12K steps at an 8,192 sequence length.
And the learning rate decays from 10−4to2×10−5.
We then extend the tokenizer of the textual LLM with 6.6K audio tokens. To properly embed the
new audio tokens and preserve the model’s textual capabilities, the model is then trained on 128B
tokens of text data and 128B tokens of audio data. Specifically, audio data includes 80B, 32B
and 16B tokens of TTS, speech-to-speech conversation and utterance-level text-speech interleaved
continuation data respectively. The sequence length is increased to 16,384. And the learning rates
of the LLM, adaptor, embedding layer and output layer are set to 2×10−5,5×10−5,5×10−5,
and4×10−5respectively.
We then introduce our main pre-training process and further train the model on another 800B
tokens of text and audio data. We unify the learning rates to 2×10−5and employ 400B tokens
of textual data and 42B, 120B, 8B, 30B, 5B, 45B and 150B tokens of ASR, TTS, speech-to-
text translation, text-to-speech translation, speech-to-text continuation, utterance-level text-speech
interleaved continuation and speech-to-speech conversation data respectively.
We finally employ 200B tokens of high-quality text and audio data to introduce a wider array of
tasks and cooldown the model. We employ 24.6B, 12.4B, 2.4B, and 3.6B tokens of audio data
for multilingual and dialectal ASR, TTS, paralinguistic information understanding, speech-to-text
translation respectively. Besides, we develop a conversational speech synthesis pipeline to synthesize
6B, 15B and 36B tokens of audio data for speech-to-speech translation, utterance-level text-speech
interleaved conversation and speech-to-speech conversation. To ensure the vocal diversity in the
synthesized speech, the system references a library of approximately 50k unique speakers. We
balance the audio data with 100B tokens of high-quality text data and the learning rate decays from
2×10−5to5×10−6.
7

Step-Audio 2 Technical Report
After this comprehensive pre-training procedure, the model has acquired strong audio understanding
and generation capabilities while maintaining its textual performance inherited from the initial
textual LLM.
3.3 Supervised fine-tuning
We subsequently perform a large-scale, multi-task supervised fine-tuning (SFT) procedure [69]
to instruct the model to follow human intention in fluid conversations and master core tasks. We
select audio data from open-source and proprietary data to ensure broad coverage and high quality.
The model is trained on 4B tokens of text and audio data for a single epoch. And the learning rate
decays from 10−5to10−6.
Specifically, we leverage extensive corpora such as GigaSpeech [8], WenetSpeech [78], and other in-
house data to enhance the model’s performance in multilingual and multi-dialect ASR scenarios. We
reformat existing datasets for audio event classification and audio captioning, such as AudioSet [23]
and AudioCaps [41], into speech question-answer pairs for audio understanding. To capture
paralinguistic information beyond just semantics, we introduce a detailed speech captioning task
and build an in-house dataset, requiring the model to generate comprehensive textual descriptions
encompassing 11 paralinguistic and environmental aspects.
We employ high-quality, professionally labeled data collected in-house for TTS. We utilize the
Chinese-to-English and English-to-Chinese subsets from the CoV oST 2 [63] dataset for speech-to-
speech translation.
We leverage high-quality in-house textual data for classic text-to-text conversation. Multiple LLMs
are then employed to rewrite these text conversations as dialogue scripts with a more natural,
colloquial style. We randomly insert emotion and speed instructions into the generated scripts
to enable basic emotion and speaking style control. The scripts are then synthesized into speech
conversations using our conversation synthesis pipeline.
We construct approximately 1K dialogue scripts in text for each type of external tools. Within
these scripts, instructions with explicit or implicit tool invocation intentions and their corresponding
statements are inserted into common dialogues. The scripts are then synthesized into speech
conversations using our conversation synthesis pipeline.
Besides, we construct and employ two reasoning-centric datasets during SFT to cold-start the
subsequent reinforcement learning process. First, we build a dataset to enable and robust audio
understanding in complex acoustic scenarios, by combining multiple audios from AudioSet and
AudioCaps, thereby creating intricate acoustic environments. To better address and respond to the
paralinguistic information in speech conversations, we synthesize a speech conversation dataset with
our conversation synthesis pipeline, based on dialogue scripts with appropriate emotion descriptions
generated from textual LLMs. Subsequently, a textual LLM with reasoning capabilities is employed
to produce question–answer pairs with explicit step-by-step reasoning traces, according to the audio
mixing recipes or the generated dialogue scripts.
3.4 Reinforcement learning
To enhance the model’s reasoning capabilities in audio understanding and speech interaction, we
implement a multi-stage reinforcement learning strategy. We leverage our reasoning-centric datasets
from SFT and utilize two stages of proximal policy optimization (PPO) [54] to optimize reasoning
8

Step-Audio 2 Technical Report
efficiency for real-time audio engagement. In the first stage, a binary reward function is employed
to limit the thinking sequence length to a predefined maximum. This reward function assigns a
value of 1 for reasoning that is appropriately concise (neither empty nor excessively long) and 0
otherwise. Training is conducted for 60 iterations with a global batch size of 64, using an actor
learning rate of 1×10−6and a critic learning rate of 2.5×10−6. The second stage transitions from
binary rewards to learned preference scoring, utilizing a trained reward model to evaluate response
quality. This stage involves an additional 120 iterations while maintaining the same batch size and
learning rate settings. Finally, we incorporate group relative policy optimization (GRPO) [54] for
400 iterations to further improve the model’s audio perceptual abilities.
4 Evaluation
4.1 Automatic speech recognition
As the most critical component of audio understanding and speech interaction, we first evaluate the
model’s capability in automatic speech recognition. We evaluate Step-Audio 2 across six Chinese
test sets, four English test sets, three multilingual test sets (Japanese, Cantonese, Arabic), and six
in-house Chinese dialect and accented Mandarin test sets. For comparative analysis, we utilize top-
performing models from both open-source and commercial domains as baselines, including Doubao
LLM ASR1, GPT-4o Transcribe2, Kimi-Audio [18], and Qwen-Omni. We prefer GPT-4o Transcribe
than GPT-4o Audio since the formers provide stronger results. Notably, Doubao LLM ASR and
GPT-4o Transcribe represent specialized ASR systems that achieve leading-edge performance.
We evaluate all the models without specifying language3and summarize the results in Table 1. Step-
Audio 2 outperforms existing open-source and commercial ASR models in both general English
and Chinese recognition, achieving an average word error rate (WER) of 3.18% on English and an
average character error rate (CER) of 3.11% on Chinese test sets. Moreover, Step-Audio 2 offers
comparable results to GPT-4o Transcribe on Arabian and Japanese recognition, to Qwen-Omni on
Cantonese recognition, demonstrating its capability in multilingual speech recognition. In addition,
Step-Audio 2 achieves the lowest average CER among 4 in-house Chinese accented Mandarin and
2 dialect test sets. These results highlight the superiority of Step-Audio 2 in understanding the
semantic information in speech.
4.2 Paralinguistic information understanding
We then evaluate how Step-Audio 2 understands the paralinguistic information in speech beyond
the semantic information. To this end, we introduce StepEval-Audio-Paralinguistic, a speech-to-
speech benchmark that evaluates the model’s understanding of paralinguistic information across 11
dimensions using single-turn question answering.
StepEval-Audio-Paralinguistic comprises 550 speech samples evenly distributed across 11 tasks.
We initially collect 400 Chinese speech clips for 8 of these tasks from public podcast recordings,
encompassing gender, age, timbre, emotion, pitch, rhythm, speaking speed, speaking style, and
vocal activity prediction or description. For sound event, scenario, and vocal sound detection or
1Doubao LLM ASR refers to https://www.volcengine.com/docs/6561/1354868
2GPT-4o Transcribe is evaluated using its latest model, gpt-4o-transcribe, via its official API.
3We evaluate without specifying language to ensure a fair comparison. Notably, Qwen-Omni lacks a language-independent
testing approach, specifying language may yield better results.
9

Step-Audio 2 Technical Report
Table 1: Comparison between Doubao LLM ASR, GPT-4o Transcribe, Kimi-Audio, Qwen-Omni and Step-Audio 2, on
character (for Chinese, Cantonese and Japanese) and word (for Arabian and English) error rates among multiple ASR
test sets. N/A indicates that the language is not supported.
Category Test setDoubao
LLM ASRGPT-4o
TranscribeKimi-
AudioQwen-
OmniStep-
Audio 2
EnglishCommon V oice 9.20 9.30 7.83 8.33 5.98
FLEURS English 7.22 2.71 4.47 5.05 3.05
LibriSpeech clean 2.92 1.75 1.49 2.93 1.19
LibriSpeech other 5.32 4.23 2.91 5.07 2.49
Average 6.17 4.50 4.18 5.35 3.18
ChineseAISHELL 0.98 3.52 0.64 1.17 0.65
AISHELL-2 3.10 4.26 2.67 2.40 2.13
FLEURS Chinese 2.92 2.62 2.91 7.01 2.80
KeSpeech phase1 6.48 26.80 5.11 6.45 3.62
WenetSpeech meeting 4.90 31.40 5.21 6.61 4.73
WenetSpeech net 4.46 15.71 5.93 5.24 4.74
Average 3.81 14.05 3.75 4.81 3.11
FLEURS Arabian N/A 11.72 N/A 25.13 15.66
Multilingual Common V oice yue 9.20 11.10 38.90 7.89 8.04
FLEURS Japanese N/A 3.27 N/A 10.49 3.44
Anhui accent 8.83 50.55 22.17 18.73 10.99
Guangdong accent 4.99 7.83 3.76 4.03 3.87
Guangxi accent 3.37 7.09 4.29 3.35 4.08
In-house Shanxi accent 20.26 55.03 34.71 25.95 13.77
Sichuan dialect 3.01 32.85 5.26 5.61 4.28
Shanghai dialect 47.49 89.58 82.90 58.74 18.14
Average 14.66 40.49 25.52 19.40 9.19
description, we source 50 event-related, 50 environmental, and 50 vocal sounds from AudioSet [23],
CochlScene [35], and V ocalSound [26], respectively. All original recordings are shorter than 30
seconds and uniformly resampled to 24,000 Hz, with annotations provided by professional groups
in open-set natural language.
We then generate textual questions and answers based on the ground-truth annotations for each task
with textual LLMs. For the first 8 tasks, we use the input speech as a prompt to clone a synthesized
question speech and randomly concatenate the question before or after the original speech. For
the remaining 3 tasks, we further mix these audios with synthesized speeches before question
concatenation, creating more challenging test samples.
We also establish an automatic evaluation protocol for StepEval-Audio-Paralinguistic, which initially
transcribes model outputs into text using ASR, followed by automatic judgment with a textual LLM.
More information, along with the complete StepEval-Audio-Paralinguistic test set and evaluation
code, is available at https://github.com/stepfun-ai/Step-Audio2 to foster further research
on paralinguistic information understanding.
We evaluate GPT-4o Audio, Kimi-Audio, Qwen-Omni, Step-Audio-AQAA, and Step-Audio 2 using
the StepEval-Audio-Paralinguistic benchmark, with results presented in Table 2. The experimental
results highlight the comprehensive capabilities of Step-Audio 2 in understanding various paralin-
guistic information, achieving an average accuracy of 76.55, which is a significant improvement
over other baseline models.
10

Step-Audio 2 Technical Report
Table 2: Comparison between GPT-4o Audio, Kimi-Audio, Qwen-Omni, Step-Audio-AQAA and Step-Audio 2 on
StepEval-Audio-Paralinguistic.
Model Avg. Gender Age Timbre Scenario Event
GPT-4o Audio 43.45 18 42 34 22 14
Kimi-Audio 49.64 94 50 10 30 48
Qwen-Omni 44.18 40 50 16 28 42
Step-Audio-AQAA 36.91 70 66 18 14 14
Step-Audio 2 76.55 98 92 78 64 46
Model Emotion Pitch Rhythm Speed Style Vocal
GPT-4o Audio 82 40 60 58 64 44
Kimi-Audio 66 56 40 44 54 54
Qwen-Omni 76 32 54 50 50 48
Step-Audio-AQAA 40 38 48 54 44 0
Step-Audio 2 72 78 70 78 84 82
4.3 Audio understanding
We then assess Step-Audio 2’s general audio comprehension across sound, speech, and music using
the latest version of the MMAU benchmark [58]1.
As baselines, we employ Audio Flamingo 3, Gemini 2.5 Pro, GPT-4o Audio, Kimi-Audio, Omni-
R1 [56], Qwen2.5-Omni, and Step-Audio-AQAA. We obtain the reported results for Audio Flamingo
3, Omni-R1, and Qwen2.5-Omni from their original papers. The results of Gemini 2.5 Pro are
obtained from the official website of MMAU. And we re-evaluate GPT-4o Audio, Kimi-Audio and
Step-Audio-AQAA due to the recent update of the MMAU benchmark.
The results are summarized in Table 3. Step-Audio 2 achieves the highest average score of 77.4%,
followed by Omni-R1 and Audio Flamingo 3, both of which are specialized approaches in audio
understanding. Specifically, Step-Audio 2 yields the best results in sound and music tracks and on
par results with the best in speech track, demonstrating its versatility and robustness across different
audio domains.
Table 3: Comparison between Audio Flamingo 3, Gemini 2.5 Pro, GPT-4o Audio, Kimi-Audio, Omni-R1, Qwen2.5-
Omni, Step-Audio-AQAA and Step-Audio 2 on MMAU.
Model Avg. Sound Speech Music
Audio Flamingo 3 73.1 76.9 66.1 73.9
Gemini 2.5 Pro 71.6 75.1 71.5 68.3
GPT-4o Audio 58.1 58.0 64.6 51.8
Kimi-Audio 69.6 79.0 65.5 64.4
Omni-R1 77.0 81.7 76.0 73.4
Qwen2.5-Omni 71.5 78.1 70.6 65.9
Step-Audio-AQAA 49.7 50.5 51.4 47.3
Step-Audio 2 77.4 82.0 75.7 74.6
4.4 Speech translation
We evaluate the model’s bidirectional Chinese-English speech translation capabilities using two
benchmarks: speech-to-text translation (S2TT) on CoV oST 2 [63] and speech-to-speech translation
1MMAU v05.15.25 test-mini
11

Step-Audio 2 Technical Report
(S2ST) on CVSS [37]. Additionally, we use the reported results of Qwen2.5-Omni for CoV oST 2,
while for CVSS, we employ Qwen-Omni as a baseline. Kimi-Audio is excluded because it
consistently ignores prompts and performs ASR instead of translation. Using BLEU as the evaluation
metric, the results in Table 4 demonstrate that Step-Audio 2 achieves superior performance in
Chinese-English bidirectional translations, obtaining the highest average score on both the CoV oST 2
and CVSS test sets.
Table 4: Comparison of BLEU scores between GPT-4o Audio, Qwen2.5-Omni, Qwen-Omni, Step-Audio-AQAA and
Step-Audio 2 on speech-to-text and speech-to-speech translation.
ModelCoVoST 2 (Speech-to-text translation)
Avg. English-to-Chinese Chinese-to-English
GPT-4o Audio 29.61 40.20 19.01
Qwen2.5-Omni 35.40 41.40 29.40
Step-Audio-AQAA 28.57 37.71 19.43
Step-Audio 2 38.84 48.40 29.27
ModelCVSS (Speech-to-speech translation)
Avg. English-to-Chinese Chinese-to-English
GPT-4o Audio 23.68 20.07 27.29
Qwen-Omni 15.35 8.04 22.66
Step-Audio-AQAA 27.36 30.74 23.98
Step-Audio 2 27.86 32.86 22.85
4.5 Tool calling
To address the gap in the availability of suitable test sets for tool calling in speech conversations, we
introduce StepEval-Audio-Toolcall, a test set that evaluates the model’s ability in tool invocation,
selection and parameter extraction under Chinese speech conversations.
We employ a textual LLM to generate 200 multi-turn dialogue scripts for each kind of tool. Each
script contains 3-6 turns of inputs and outputs, in which previous turns may or may not include tool
calling statements, but the final input must contain a calling intention to a specific external tool.
We then balance the samples with an equal number of negative samples for each kind of tools, in
which the final speech input either has no tool calling intention or intention to call on other kinds
of tools. Subsequently, we synthesize these scripts into speeches with our conversation synthesis
pipeline. And we propose an automatic evaluation protocol to employ Qwen3-32B to automatically
examine the output and tool calling statements. We release StepEval-Audio-Toolcall including
the original scripts, synthesized speech conversations and the corresponding evaluation script in
https://github.com/stepfun-ai/Step-Audio2 .
Despite that there is no other LALM that provides custom tool calling, we employ Qwen3-32B as
a baseline to illustrate how Step-Audio 2 manages external tools in comparison to textual LLMs.
As shown in Table 5, Step-Audio 2 achieves on par with tool calling accuracy with textual LLMs
even with speech input. Notably, Step-Audio 2 significantly outperforms Qwen3-32B in accurately
calling our innovative audio search tool, highlighting its specialty as a multi-modal LLM than
textual LLMs.
12

Step-Audio 2 Technical Report
Table 5: Comparison between Step-Audio 2 and Qwen3-32B on StepEval-Audio-Toolcall.†Qwen3-32B is evaluated
with text inputs.‡Date and time tools have no parameter.
Model Objective Metric Audio search Date & Time‡Weather Web search
Trigger Precision / Recall 67.5 / 98.5 98.4 / 100.0 90.1 / 100.0 86.8 / 98.5
Qwen3-32B†Type Accuracy 100.0 100.0 98.5 98.5
Parameter Accuracy 100.0 N/A 100.0 100.0
Trigger Precision / Recall 86.8 / 99.5 96.9 / 98.4 92.2 / 100.0 88.4 / 95.5
Step-Audio 2 Type Accuracy 100.0 100.0 90.5 98.4
Parameter Accuracy 100.0 N/A 100.0 100.0
4.6 Speech-to-speech conversation
We finally employ URO-Bench [75] to evaluate Step-Audio 2 and other open-source and commercial
LALMs, including GPT-4o Audio, Kimi-Audio, Qwen-Omni, and Step-Audio-AQAA. URO-Bench
consists of 16 and 20 datasets on two difficulty tracks, evaluating the model’s understanding, reason-
ing and oral conversation abilities, such as ASR, instruction following, commonsense knowledge,
mathematics, and speech naturalness, emotion and speaking styles expressions. We follow the ASR-
mediated procedure in URO-Bench for evaluation, employing Whisper for ASR and GPT-4o-mini
for automatic judging.
As demonstrated in Table 6, Step-Audio 2 significantly outperforms existing large audio language
models, including GPT-4o Audio, in Chinese speech-to-speech conversation scenarios, achieving
the highest average scores of 78.86 on the basic track and 70.83 on the pro track. In English
speech-to-speech conversations, while Step-Audio 2 is slightly outperformed by GPT-4o Audio,
it provides very competitive results and exceeds the other approaches. More detailed results on
URO-Bench are provided in Table 7.
Table 6: Comparison between GPT-4o Audio, Kimi-Audio, Qwen-Omni, Step-Audio-AQAA and Step-Audio 2 on the
URO-Bench. U. R. O. stands for understanding, reasoning, and oral conversation, respectively.
Model LanguageBasic Pro
Avg. U. R. O. Avg. U. R. O.
GPT-4o Audio
Chinese74.18 82.98 57.23 82.33 66.91 72.94 51.52 71.14
Kimi-Audio 70.47 75.86 59.69 75.85 66.21 63.13 55.09 76.70
Qwen-Omni 62.08 46.44 64.73 75.05 61.06 61.55 59.79 61.43
Step-Audio-AQAA 55.73 66.02 57.31 43.87 59.15 61.82 52.74 60.74
Step-Audio 2 78.86 87.66 68.52 80.39 70.83 79.35 59.71 69.72
GPT-4o Audio
English84.54 90.18 75.90 90.41 67.51 60.65 64.36 78.46
Kimi-Audio 60.04 83.36 42.31 60.36 49.79 50.32 40.59 56.04
Qwen-Omni 70.58 66.29 69.62 76.16 50.99 44.51 63.88 49.41
Step-Audio-AQAA 71.11 90.15 56.12 72.06 52.01 44.25 54.54 59.81
Step-Audio 2 79.03 90.80 70.42 78.74 60.25 60.47 61.21 59.25
5 Conclusion
We introduce Step-Audio 2, an end-to-end large audio language model designed for enterprise
speech and audio understanding, as well as intelligent speech interaction. Step-Audio 2 leverages a
latent audio encoder and reinforcement learning to enhance its speech and audio comprehension
13

Step-Audio 2 Technical Report
capabilities. Furthermore, by integrating the generation of discrete audio tokens into language mod-
eling, Step-Audio 2 achieves genuine end-to-end speech interaction and improves its responsiveness
to paralinguistic information, such as speaking styles and emotions. Step-Audio 2 is also capable of
utilizing external tools including web search and audio search for multi-modal RAG. Trained on
8 million hours of speeches and audios, Step-Audio 2 demonstrates state-of-the-art performance
across various tasks, including ASR, audio understanding, speech translation, and general speech
conversation, outperforming both open-source and commercial solutions.
References
[1] Philip Anastassiou et al. “Seed-tts: A family of high-quality versatile speech generation models”. In: arXiv
preprint arXiv:2406.02430 (2024).
[2] Rohan Anil et al. PaLM 2 Technical Report . 2023. arXiv: 2305.10403 [cs.CL] .URL:https://arxiv.org/
abs/2305.10403 .
[3] Alexei Baevski et al. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations . 2020.
arXiv: 2006.11477 [cs.CL] .URL:https://arxiv.org/abs/2006.11477 .
[4] Jinze Bai et al. Qwen Technical Report . 2023. arXiv: 2309.16609 [cs.CL] .URL:https://arxiv.org/abs/
2309.16609 .
[5] Ye Bai et al. “Seed-asr: Understanding diverse speech and contexts with llm-based speech recognition”. In: arXiv
preprint arXiv:2407.04675 (2024).
[6] James Betker. Better speech synthesis through scaling . 2023. arXiv: 2305.07243 [cs.SD] .URL:https:
//arxiv.org/abs/2305.07243 .
[7] Zalán Borsos et al. “Audiolm: a language modeling approach to audio generation”. In: IEEE/ACM transactions
on audio, speech, and language processing 31 (2023), pp. 2523–2533.
[8] Guoguo Chen et al. “GigaSpeech: An Evolving, Multi-Domain ASR Corpus with 10,000 Hours of Transcribed
Audio”. In: Interspeech 2021 . ISCA, Aug. 2021. DOI:10.21437/interspeech.2021-1965 .URL:http:
//dx.doi.org/10.21437/Interspeech.2021-1965 .
[9] Qian Chen et al. “Minmo: A multimodal large language model for seamless voice interaction”. In: arXiv preprint
arXiv:2501.06282 (2025).
[10] Sanyuan Chen et al. BEATs: Audio Pre-Training with Acoustic Tokenizers . 2022. arXiv: 2212.09058 [eess.AS] .
URL:https://arxiv.org/abs/2212.09058 .
[11] Sanyuan Chen et al. “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing”. In:
IEEE Journal of Selected Topics in Signal Processing 16.6 (Oct. 2022), pp. 1505–1518. ISSN : 1941-0484. DOI:
10.1109/jstsp.2022.3188113 .URL:http://dx.doi.org/10.1109/JSTSP.2022.3188113 .
[12] Yunfei Chu et al. “Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language
models”. In: arXiv preprint arXiv:2311.07919 (2023).
[13] Yunfei Chu et al. “Qwen2-audio technical report”. In: arXiv preprint arXiv:2407.10759 (2024).
[14] Jade Copet et al. Simple and Controllable Music Generation . 2024. arXiv: 2306.05284 [cs.SD] .URL:https:
//arxiv.org/abs/2306.05284 .
[15] Alexandre Défossez et al. “High fidelity neural audio compression”. In: arXiv preprint arXiv:2210.13438 (2022).
[16] Alexandre Défossez et al. “Moshi: a speech-text foundation model for real-time dialogue”. In: arXiv preprint
arXiv:2410.00037 (2024).
[17] Soham Deshmukh et al. “Pengi: An audio language model for audio tasks”. In: Advances in Neural Information
Processing Systems 36 (2023), pp. 18090–18108.
[18] Ding Ding et al. “Kimi-audio technical report”. In: arXiv preprint arXiv:2504.18425 (2025).
[19] Zhihao Du et al. “Cosyvoice 2: Scalable streaming speech synthesis with large language models”. In: arXiv
preprint arXiv:2412.10117 (2024).
[20] Zhihao Du et al. CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised
Semantic Tokens . 2024. arXiv: 2407.05407 [cs.SD] .URL:https://arxiv.org/abs/2407.05407 .
[21] Qingkai Fang et al. “Llama-omni: Seamless speech interaction with large language models”. In: arXiv preprint
arXiv:2409.06666 (2024).
[22] Heting Gao et al. LUCY: Linguistic Understanding and Control Yielding Early Stage of Her . 2025. arXiv:
2501.16327 [cs.CL] .URL:https://arxiv.org/abs/2501.16327 .
14

Step-Audio 2 Technical Report
[23] Jort F. Gemmeke et al. “Audio Set: An ontology and human-labeled dataset for audio events”. In: 2017 IEEE
International Conference on Acoustics, Speech and Signal Processing (ICASSP) . 2017, pp. 776–780. DOI:
10.1109/ICASSP.2017.7952261 .
[24] Sreyan Ghosh et al. Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert
Reasoning Abilities . 2025. arXiv: 2503.03983 [cs.SD] .URL:https://arxiv.org/abs/2503.03983 .
[25] Arushi Goel et al. Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language
Models . 2025. arXiv: 2507.08128 [cs.SD] .URL:https://arxiv.org/abs/2507.08128 .
[26] Yuan Gong, Jin Yu, and James Glass. “V ocalsound: A Dataset for Improving Human V ocal Sounds Recognition”.
In:ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) .
2022, pp. 151–155. DOI:10.1109/ICASSP43922.2022.9746828 .
[27] Yuan Gong et al. “Joint audio and speech understanding”. In: 2023 IEEE Automatic Speech Recognition and
Understanding Workshop (ASRU) . IEEE. 2023, pp. 1–8.
[28] Yuan Gong et al. “Listen, think, and understand”. In: arXiv preprint arXiv:2305.10790 (2023).
[29] Aaron Grattafiori et al. The Llama 3 Herd of Models . 2024. arXiv: 2407 . 21783 [cs.AI] .URL:https :
//arxiv.org/abs/2407.21783 .
[30] Wei-Ning Hsu et al. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden
Units . 2021. arXiv: 2106.07447 [cs.CL] .URL:https://arxiv.org/abs/2106.07447 .
[31] Ailin Huang et al. “Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model”. In: arXiv
preprint arXiv:2506.08967 (2025).
[32] Ailin Huang et al. “Step-audio: Unified understanding and generation in intelligent speech interaction”. In: arXiv
preprint arXiv:2502.11946 (2025).
[33] Rongjie Huang et al. “Audiogpt: Understanding and generating speech, music, sound, and talking head”. In:
Proceedings of the AAAI Conference on Artificial Intelligence . V ol. 38. 21. 2024, pp. 23802–23804.
[34] Aaron Hurst et al. “Gpt-4o system card”. In: arXiv preprint arXiv:2410.21276 (2024).
[35] Il-Young Jeong and Jeongsoo Park. “CochlScene: Acquisition of acoustic scene data using crowdsourcing”. In:
2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC) .
2022, pp. 17–21. DOI:10.23919/APSIPAASC55919.2022.9979822 .
[36] Shengpeng Ji et al. “Wavtokenizer: an efficient acoustic discrete codec tokenizer for audio language modeling”.
In:arXiv preprint arXiv:2408.16532 (2024).
[37] Ye Jia et al. CVSS Corpus and Massively Multilingual Speech-to-Speech Translation . 2022. arXiv: 2201.03713
[cs.CL] .URL:https://arxiv.org/abs/2201.03713 .
[38] Ye Jia et al. “Direct speech-to-speech translation with a sequence-to-sequence model”. In: arXiv preprint
arXiv:1904.06037 (2019).
[39] Ye Jia et al. “Translatotron 2: High-quality direct speech-to-speech translation with voice preservation”. In:
International conference on machine learning . PMLR. 2022, pp. 10120–10134.
[40] Eugene Kharitonov et al. Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision . 2023.
arXiv: 2302.03540 [cs.SD] .URL:https://arxiv.org/abs/2302.03540 .
[41] Chris Dongjoo Kim et al. “Audiocaps: Generating captions for audios in the wild”. In: Proceedings of the 2019
Conference of the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers) . 2019, pp. 119–132.
[42] Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. HiFi-GAN: Generative Adversarial Networks for Efficient and
High Fidelity Speech Synthesis . 2020. arXiv: 2010.05646 [cs.SD] .URL:https://arxiv.org/abs/2010.
05646 .
[43] Zhifeng Kong et al. Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue
Abilities . 2024. arXiv: 2402.01831 [cs.SD] .URL:https://arxiv.org/abs/2402.01831 .
[44] Chenyang Le et al. “Transvip: Speech to speech translation system with voice and isochrony preservation”. In:
Advances in Neural Information Processing Systems 37 (2024), pp. 89682–89705.
[45] Ann Lee et al. “Textless speech-to-speech translation on real data”. In: arXiv preprint arXiv:2112.08352 (2021).
[46] Sang-gil Lee et al. BigVGAN: A Universal Neural Vocoder with Large-Scale Training . 2023. arXiv: 2206.04658
[cs.SD] .URL:https://arxiv.org/abs/2206.04658 .
[47] Guan-Ting Lin, Cheng-Han Chiang, and Hung-yi Lee. “Advancing large language models to capture varied
speaking styles and respond properly in spoken conversations”. In: arXiv preprint arXiv:2402.12786 (2024).
[48] Guan-Ting Lin et al. “Paralinguistics-enhanced large language modeling of spoken dialogue”. In: ICASSP
2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) . IEEE. 2024,
pp. 10316–10320.
15

Step-Audio 2 Technical Report
[49] Tu Anh Nguyen et al. Spirit LM: Interleaved Spoken and Written Language Model . 2024. arXiv: 2402.05755
[cs.CL] .URL:https://arxiv.org/abs/2402.05755 .
[50] OpenAI. GPT-4 Technical Report .https://openai.com/research/gpt-4 . Accessed: 2025-07-11. 2023.
[51] OpenAI. Introducing ChatGPT . Accessed: 2025-07-11. 2022. URL:https://openai.com/blog/chatgpt .
[52] Wei Ping et al. “Deep voice 3: 2000-speaker neural text-to-speech”. In: proc. ICLR . V ol. 79. 2018, pp. 1094–1099.
[53] Alec Radford et al. “Robust speech recognition via large-scale weak supervision”. In: International conference
on machine learning . PMLR. 2023, pp. 28492–28518.
[54] Rafael Rafailov et al. “Direct preference optimization: Your language model is secretly a reward model”. In:
Advances in Neural Information Processing Systems 36 (2023), pp. 53728–53741.
[55] Yi Ren et al. “Fastspeech 2: Fast and high-quality end-to-end text to speech”. In: arXiv preprint arXiv:2006.04558
(2020).
[56] Andrew Rouditchenko et al. “Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?” In: arXiv
preprint arXiv:2505.09439 (2025).
[57] Paul K Rubenstein et al. “Audiopalm: A large language model that can speak and listen”. In: arXiv preprint
arXiv:2306.12925 (2023).
[58] S Sakshi et al. “Mmau: A massive multi-task audio understanding and reasoning benchmark”. In: arXiv preprint
arXiv:2410.19168 (2024).
[59] Jonathan Shen et al. “Natural tts synthesis by conditioning wavenet on mel spectrogram predictions”. In: 2018
IEEE international conference on acoustics, speech and signal processing (ICASSP) . IEEE. 2018, pp. 4779–4783.
[60] Hubert Siuzdak, Florian Grötschla, and Luca A Lanzendörfer. “Snac: Multi-scale neural audio codec”. In: arXiv
preprint arXiv:2410.14411 (2024).
[61] Changli Tang et al. “Salmonn: Towards generic hearing abilities for large language models”. In: arXiv preprint
arXiv:2310.13289 (2023).
[62] Wolfgang Wahlster. Verbmobil: foundations of speech-to-speech translation . Springer Science & Business Media,
2013.
[63] Changhan Wang, Anne Wu, and Juan Pino. CoVoST 2 and Massively Multilingual Speech-to-Text Translation .
2020. arXiv: 2007.10310 [cs.CL] .URL:https://arxiv.org/abs/2007.10310 .
[64] Chengyi Wang et al. “Neural codec language models are zero-shot text to speech synthesizers”. In: arXiv preprint
arXiv:2301.02111 (2023).
[65] Xinsheng Wang et al. “Spark-tts: An efficient llm-based text-to-speech model with single-stream decoupled
speech tokens”. In: arXiv preprint arXiv:2503.01710 (2025).
[66] Xiong Wang et al. “Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen llm”. In:
arXiv preprint arXiv:2411.00774 (2024).
[67] Yuancheng Wang et al. “Maskgct: Zero-shot text-to-speech with masked generative codec transformer”. In: arXiv
preprint arXiv:2409.00750 (2024).
[68] Yuxuan Wang et al. “Tacotron: Towards end-to-end speech synthesis”. In: arXiv preprint arXiv:1703.10135
(2017).
[69] Jason Wei et al. “Finetuned language models are zero-shot learners”. In: arXiv preprint arXiv:2109.01652 (2021).
[70] Yonghui Wu et al. “Google’s neural machine translation system: Bridging the gap between human and machine
translation”. In: arXiv preprint arXiv:1609.08144 (2016).
[71] Zhifei Xie and Changqiao Wu. “Mini-omni: Language models can hear, talk while thinking in streaming”. In:
arXiv preprint arXiv:2408.16725 (2024).
[72] Zhifei Xie and Changqiao Wu. “Mini-omni2: Towards open-source gpt-4o with vision, speech and duplex
capabilities”. In: arXiv preprint arXiv:2410.11190 (2024).
[73] Detai Xin et al. “Bigcodec: Pushing the limits of low-bitrate neural speech codec”. In: arXiv preprint
arXiv:2409.05377 (2024).
[74] Jin Xu et al. Qwen2.5-Omni Technical Report . 2025. arXiv: 2503.20215 [cs.CL] .URL:https://arxiv.
org/abs/2503.20215 .
[75] Ruiqi Yan et al. URO-Bench: A Comprehensive Benchmark for End-to-End Spoken Dialogue Models . 2025.
arXiv: 2502.17810 [cs.CL] .URL:https://arxiv.org/abs/2502.17810 .
[76] Neil Zeghidour et al. “Soundstream: An end-to-end neural audio codec”. In: IEEE/ACM Transactions on Audio,
Speech, and Language Processing 30 (2021), pp. 495–507.
[77] Aohan Zeng et al. “Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot”. In: arXiv
preprint arXiv:2412.02612 (2024).
16

Step-Audio 2 Technical Report
[78] Binbin Zhang et al. WenetSpeech: A 10000+ Hours Multi-domain Mandarin Corpus for Speech Recognition .
2022. arXiv: 2110.03370 [cs.SD] .URL:https://arxiv.org/abs/2110.03370 .
[79] Bowen Zhang et al. “Minimax-speech: Intrinsic zero-shot text-to-speech with a learnable speaker encoder”. In:
arXiv preprint arXiv:2505.07916 (2025).
[80] Dong Zhang et al. “Speechgpt: Empowering large language models with intrinsic cross-modal conversational
abilities”. In: arXiv preprint arXiv:2305.11000 (2023).
[81] Xiangyu Zhang et al. “Distinctive Feature Codec: Adaptive Segmentation for Efficient Speech Representation”.
In:arXiv preprint arXiv:2505.18516 (2025).
[82] Xin Zhang et al. “Speechtokenizer: Unified speech tokenizer for speech large language models”. In: arXiv
preprint arXiv:2308.16692 (2023).
17

Step-Audio 2 Technical Report
Appendix
A Contributors
The contributors are list in alphabet order.
A.1 Core contributors
Model Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu,
Haoyang Zhang, Jingbei Li, Mingrui Chen, Peng Liu, Wang You, Xiangyu (Tony) Zhang, Xingyuan
Li, Xuerui Yang, Yayue Deng, Yechang Huang, Yuxin Li, Yuxin Zhang, Zhao You
Infrastructure Brian Li, Changyi Wan, Hanpeng Hu, Jiangjie Zhen, Siyu Chen, Song Yuan,
Xuelin Zhang, Yimin Jiang, Yu Zhou, Yuxiang Yang
Data and evaluation Bingxin Li, Buyun Ma, Changhe Song, Dongqing Pang, Guoqiang Hu,
Haiyang Sun, Kang An, Na Wang, Shuli Gao, Wei Ji, Wen Li, Wen Sun, Xuan Wen, Yong Ren,
Yuankai Ma, Yufan Lu
A.2 Contributors
Bin Wang, Bo Li, Changxin Miao, Che Liu, Chen Xu, Dapeng Shi, Dingyuan Hu, Donghang Wu,
Enle Liu, Guanzhe Huang, Gulin Yan, Han Zhang, Hao Nie, Haonan Jia, Hongyu Zhou, Jianjian
Sun, Jiaoren Wu, Jie Wu, Jie Yang, Jin Yang, Junzhe Lin, Kaixiang Li, Lei Yang, Liying Shi,
Li Zhou, Longlong Gu, Ming Li, Mingliang Li, Mingxiao Li, Nan Wu, Qi Han, Qinyuan Tan,
Shaoliang Pang, Shengjie Fan, Siqi Liu, Tiancheng Cao, Wanying Lu, Wenqing He, Wuxun Xie,
Xu Zhao, Xueqi Li, Yanbo Yu, Yang Yang, Yi Liu, Yifan Lu, Yilei Wang, Yuanhao Ding, Yuanwei
Liang, Yuanwei Lu, Yuchu Luo, Yuhe Yin, Yumeng Zhan, Yuxiang Zhang, Zidong Yang, Zixin
Zhang
A.3 Sponsors
Binxing Jiao, Daxin Jiang, Heung-Yeung Shum, Jiansheng Chen, Jing Li, Xiangyu Zhang, Yibo
Zhu
A.4 External contributors
Nanyang Technological University (NTU), Singapore Eng Siong Chng, Hexin Liu
18

Step-Audio 2 Technical Report
B Detailed evaluation results on URO-bench
Table 7: Detailed results on the URO-Bench. C. E. stand for Chinese and English. B. P. stand for the basic and pro
tracks. U. R. O. stand for understanding, reasoning and oral conversation, respectively.
ModelC. B. U. C. B. U. C. B. R. C. B. R. C. B. O. C. B. O.
Repeat LCSTS MLC OpenBook Alpaca Claude
GPT-4o Audio 86.11 79.85 50.00 64.46 75.87 88.79
Kimi-Audio 73.32 78.40 52.45 66.93 69.90 81.81
Qwen-Omni 19.66 73.22 61.76 67.70 70.40 79.71
Step-Audio-AQAA 63.72 68.33 67.40 47.21 56.23 31.50
Step-Audio 2 96.16 79.16 62.99 74.06 76.23 84.54
ModelC. P. U. C. P. U. C. P. U. C. P. R. C. P. R. C. P. O.
UnderEmotion CodeSwitch Safety MLC-Pro Speaker SRT
GPT-4o Audio 72.07 63.43 83.33 53.65 49.39 85.71
Kimi-Audio 76.29 66.76 46.33 58.33 51.84 99.05
Qwen-Omni 70.13 66.86 47.67 72.92 46.67 98.10
Step-Audio-AQAA 70.46 52.00 63.00 53.65 51.84 97.14
Step-Audio 2 78.14 63.90 96.00 70.31 49.12 96.19
ModelC. P. O. C. P. O. E. B. U. E. B. U. E. B. U. E. B. R.
GenEmotion GenStyle Repeat Summary Gaokao Storal
GPT-4o Audio 30.28 97.44 96.08 92.94 81.52 83.15
Kimi-Audio 35.49 95.56 90.27 78.19 81.63 70.88
Qwen-Omni 8.58 77.61 28.36 80.51 89.99 72.90
Step-Audio-AQAA 26.80 58.29 93.02 90.62 86.80 53.00
Step-Audio 2 23.56 89.40 97.74 83.90 90.76 76.38
ModelE. B. R. E. B. R. E. B. R. E. B. O. E. B. O. E. B. O.
Truthful Gsm8k MLC Alpaca Common Wildchat
GPT-4o Audio 74.27 70.10 76.08 94.00 85.73 91.50
Kimi-Audio 53.90 0.97 43.50 63.05 51.40 66.63
Qwen-Omni 62.03 63.52 80.04 78.16 72.40 77.92
Step-Audio-AQAA 62.30 38.20 71.00 71.86 75.53 68.79
Step-Audio 2 65.45 62.83 77.02 81.74 72.87 81.62
ModelE. P. U. E. P. U. E. P. U. E. P. U. E. P. U. E. P. R.
UnderEmotion CodeSwitch Safety Clotho MuCho MLC-Pro
GPT-4o Audio 54.21 72.00 88.61 42.77 45.66 71.43
Kimi-Audio 56.40 62.86 57.78 43.27 31.30 23.81
Qwen-Omni 52.21 61.52 63.06 13.58 32.15 66.30
Step-Audio-AQAA 48.86 68.95 63.06 8.43 31.94 45.05
Step-Audio 2 58.05 56.48 91.11 37.74 58.95 69.60
ModelE. P. R. E. P. R. E. P. O. E. P. O. E. P. O. E. P. O.
MtBench Speaker SRT GenEmotion GenStyle Multilingual
GPT-4o Audio 76.07 45.58 80.93 37.62 100.00 95.28
Kimi-Audio 55.54 42.42 89.30 25.33 80.61 28.90
Qwen-Omni 74.81 50.55 86.51 8.89 80.30 21.94
Step-Audio-AQAA 68.74 49.82 90.70 29.62 82.12 36.81
Step-Audio 2 68.46 45.58 90.70 27.70 81.82 36.77
19