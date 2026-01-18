# VoiceSculptor: Your Voice, Designed By You

**Authors**: Jingbin Hu, Huakang Chen, Linhan Ma, Dake Guo, Qirui Zhan, Wenhao Li, Haoyu Zhang, Kangxiang Xia, Ziyu Zhang, Wenjie Tian, Chengyou Wang, Jinrui Liang, Shuhan Guo, Zihang Yang, Bengu Wu, Binbin Zhang, Pengcheng Zhu, Pengyuan Xie, Chuan Xie, Qiang Zhang, Jie Liu, Lei Xie

**Published**: 2026-01-15 17:49:15

**PDF URL**: [https://arxiv.org/pdf/2601.10629v1](https://arxiv.org/pdf/2601.10629v1)

## Abstract
Despite rapid progress in text-to-speech (TTS), open-source systems still lack truly instruction-following, fine-grained control over core speech attributes (e.g., pitch, speaking rate, age, emotion, and style). We present VoiceSculptor, an open-source unified system that bridges this gap by integrating instruction-based voice design and high-fidelity voice cloning in a single framework. It generates controllable speaker timbre directly from natural-language descriptions, supports iterative refinement via Retrieval-Augmented Generation (RAG), and provides attribute-level edits across multiple dimensions. The designed voice is then rendered into a prompt waveform and fed into a cloning model to enable high-fidelity timbre transfer for downstream speech synthesis. VoiceSculptor achieves open-source state-of-the-art (SOTA) on InstructTTSEval-Zh, and is fully open-sourced, including code and pretrained models, to advance reproducible instruction-controlled TTS research.

## Full Text


<!-- PDF content starts -->

2026-01-16
VoiceSculptor: Your Voice, Designed By You
Jingbin Hu1, Huakang Chen1, Linhan Ma1, Dake Guo1, Qirui Zhan1, Wenhao Li1,
Haoyu Zhang1, Kangxiang Xia1, Ziyu Zhang1, Wenjie Tian1, Chengyou Wang1, Jinrui
Liang1, Shuhan Guo1, Zihang Yang1, Bengu Wu2, Binbin Zhang4, Pengcheng Zhu1,4,
Pengyuan Xie3, Chuan Xie3, Qiang Zhang3, Jie Liu3, Lei Xie1†
1Audio, Speech and Language Processing Group (ASLP@NPU), School of Computer
Science, Northwestern Polytechnical University
2Yutu Zhineng
3Shanghai Lingguang Zhaxian Technology
4WeNet Open Source Community
https://github.com/ASLP-lab/VoiceSculptor
https://huggingface.co/ASLP-lab/VoiceSculptor-VD
https://hujingbin1.github.io/VoiceSculptor-Demo
https://huggingface.co/spaces/ASLP-lab/VoiceSculptor
Abstract
Despite rapid progress in text-to-speech (TTS), open-source systems still lack truly
instruction-following, fine-grained control over core speech attributes (e.g., pitch, speak-
ing rate, age, emotion, and style). We present VoiceSculptor, an open-source unified
system that bridges this gap by integrating instruction-based voice design and high-
fidelity voice cloning in a single framework. It generates controllable speaker timbre
directly from natural-language descriptions, supports iterative refinement via Retrieval-
Augmented Generation (RAG), and provides attribute-level edits across multiple di-
mensions. The designed voice is then rendered into a prompt waveform and fed into a
cloning model to enable high-fidelity timbre transfer for downstream speech synthesis.
VoiceSculptor achieves open-source state-of-the-art (SOTA) on InstructTTSEval-Zh, and
is fully open-sourced, including code and pretrained models, to advance reproducible
instruction-controlled TTS research.
1 Introduction
In recent years, the rapid evolution of large-scale multimodal foundation models has fundamentally
reshaped the paradigm of generative artificial intelligence (AI), enabling unified generation across text,
speech, images, and videos. Commercial systems such as Gemini 2.5 Pro & Flash, and GPT-4o mini have
demonstrated strong instruction-following and multimodal reasoning capabilities, while recent text-to-
video and audio–visual generation models, including Veo 3, Wan 2.6, Seedance 1.5 pro and Kling 2.6,
have shown impressive progress in jointly synthesizing coherent visual content and synchronized audio.
These advances highlight a growing trend toward holistic, natural-language-driven content creation, in
which users expect to express complex intentions once and obtain rich and accurate multi-modal outputs.
In parallel, audio-centric foundation models such as MiMo-Audio (Zhang et al., 2025a) and Step-
Audio2 (Wu et al., 2025) have significantly improved the representation and generalization capacity
of speech models, laying the foundation for unified modeling of text and audio, and enabling speech
generation that can be directly controlled by textual instructions. Modern neural text-to-speech (TTS)
systems, including CosyVoice2 (Du et al., 2024) , LLaSA (Ye et al., 2025) , F5-TTS (Chen et al., 2025)
, SparkTTS (Wang et al., 2025) , and Index-TTS2 (Zhou et al., 2025) , can now generate highly natu-
ral speech and effectively mimic speaker timbre when reference audio is available. However, despite
1arXiv:2601.10629v1  [eess.AS]  15 Jan 2026

2026-01-16
Figure 1: The overview of VoiceSculptor, which is composed of two core components: voice design and
voice clone.
these advances, controllability over speech attributes remains limited, especially when compared to the
flexibility observed in recent multimodal generation systems.
Traditional non-LLM-based instruct TTS approaches, such as PromptStyle (Liu et al., 2023) and Prompt-
Speaker (Zhang et al., 2023) , rely on predefined prompts or learned style or speaker embeddings to
modulate speech attributes. While effective in constrained scenarios, these methods suffer from limited
scalability and expressiveness, as they depend on carefully designed templates and struggle to generalize
to open-ended natural language instructions. Recent LLM-based approaches attempt to leverage the se-
mantic modeling capability of large language models for instruction-driven speech synthesis. VoxInstruct
(Zhou et al., 2024) and FleSpeech (Li et al., 2025) represent this emerging direction by conditioning speech
generation on textual or multimodal representations. In particular, FleSpeech introduces a multimodal
prompt encoder and aligns text, audio, and visual representations into a shared latent space to guide
diffusion-based speech generation. Similarly, HiStyle (Zhang et al., 2025c) adopts a representation
alignment strategy by mapping text embeddings, style embeddings, and speaker embeddings into a
unified embedding space, enabling textual descriptions to implicitly steer the synthesis process. By
aligning heterogeneous modalities or control signals, these methods improve the consistency between
input instructions and generated speech styles. However, despite their effectiveness, both FleSpeech and
HiStyle rely on compact, continuous embeddings as the primary control interface. Such embedding-level
conditioning inevitably compresses rich and multi-dimensional voice attributes into low-bandwidth
representations, limiting the ability of large language models to perform explicit reasoning over fine-
grained acoustic properties. As a result, controllability remains coarse and implicit, making it difficult to
precisely manipulate individual attributes or faithfully interpret complex, compositional natural language
instructions.
Nevertheless, most existing systems still generate speech primarily conditioned on text and reference
audio, offering limited direct control over fine-grained acoustic attributes such as pitch, speaking rate,
age, emotional expression, speaking style, and others. This gap reveals a fundamental bottleneck in
current generative systems: although natural language has become the dominant interface for controlling
complex multimodal generation, speech synthesis still lacks a principled and flexible mechanism to
translate high-level linguistic intent into fine-grained acoustic realization. In contrast to visual and video
generation, voice generation remains heavily constrained by reference-based conditioning or rigid control
tokens.
To address this limitation, we propose VoiceSculptor, a unified and highly flexible speech synthesis
framework that bridges natural language intent and fine-grained voice generation. Unlike conventional
TTS systems that rely on fixed control tokens or reference audio alone, VoiceSculptor enables users to
design speaker timbre and manipulate multiple voice attributes directly via free-form natural language
instructions.
At the core of this capability, the voice design module introduces a chain-of-thought (CoT)-based (Wei
et al., 2022) , fine-grained attribute modeling mechanism that explicitly decomposes high-level natural
language instructions into structured intermediate reasoning steps across multiple acoustic and stylistic
2

2026-01-16
attributes. By modeling this reasoning process as auxiliary attribute tokens, the model is guided to
interpret abstract linguistic descriptions step by step and map them to concrete acoustic realizations,
enabling precise, disentangled control over prosody, style, and speaker-related characteristics.
To further enhance instruction understanding and robustness, the voice design module incorporates
Retrieval-Augmented Generation (RAG) (Lewis et al., 2021) , which retrieves semantically relevant
instruction examples and attribute knowledge to support iterative instruction refinement and general-
ization to out-of-domain descriptions. The framework integrates a voice design module with a voice
cloning module, enabling synthesized audio from descriptive instructions to serve as a prompt waveform
for downstream speech synthesis. By jointly leveraging CoT-based fine-grained attribute reasoning
and RAG-based instruction grounding, VoiceSculptor establishes a more expressive, intuitive, and scal-
able paradigm for personalized, highly controllable TTS, aligning speech generation with the broader
trajectory of multimodal generative systems.
2 Architecture
In-the-wild Data
 VAD Data Collection , Filtering and Annotation
Multispeaker
Detection
Volume
Speed PitchEnergy
Denoise
MOS Filter
Whisper-Large-V3FireredASR
SenseVoice
ASR
Kaldi 
Force Alignment
BiLSTM
Gemini 
Annotations
Deepseek
Translation、Caption
In-house Data
Advanced Acoustic 
Level
Prosodic Level
Data Pool
Age GenderEmotional LevelSenseVoice
Emotion2Vec
Qwen3-72B
Qwen3Omni
Cross-validation
Emotion
Distribution Statistics
Threshold adjustment 
based on human feedback
Correcting the real labels
Regular expressions  
filtering illusion text
Structured text generation
Annotated
Data Pool
Punctuation Prediction
Figure 2: The data pipeline of building VoiceSculptor.
2.1 Overview
As shown in Figure 1, VoiceSculptor adopts LLaSA-3B (Ye et al., 2025) as the voice design model and
CosyVoice2 (Du et al., 2024) as the voice cloning model. LLaSA is built upon the open-source LLaMA
(Touvron et al., 2023) family released by Meta and is fine-tuned to leverage the strong text understanding
and sequence modeling capabilities of large language models. To enable speech generation within
3

2026-01-16
an LLM framework, LLaSA incorporates an advanced neural audio codec, XCodec2, which converts
continuous speech waveforms into discrete audio tokens that resemble text tokens.
With the introduction of XCodec2, speech synthesis is reformulated as a sequence-to-sequence generation
problem. Given a natural language instruction, the LLM is responsible for interpreting the semantic
and stylistic intent of the text and predicting a corresponding sequence of discrete audio tokens. These
predicted tokens are then decoded by the audio codec to reconstruct high-quality speech waveforms.
This design allows the model to jointly reason over text and speech tokens using a unified autoregressive
generation paradigm.
Built upon this foundation, VoiceSculptor consists of three key contributions. First, we construct a com-
prehensive data processing pipeline that supports large-scale data collection, filtering, multi-dimensional
annotation, and human verification, providing high-quality supervision for instruction-driven voice
design. Second, we introduce CoT–based fine-grained attribute modeling, which enables precise and
interpretable control over prosodic and stylistic attributes by guiding the LLM to reason explicitly over
attribute-related semantics during audio token generation. Third, to improve robustness and generaliza-
tion to diverse and out-of-domain natural language instructions, we incorporate a RAG mechanism that
supplies semantically aligned in-domain instruction examples at inference time, effectively grounding
the instruction interpretation process.
Together, these components enable VoiceSculptor to translate natural language instructions into control-
lable and reusable voice representations, which can be seamlessly consumed by downstream speech
synthesis models such as CosyVoice2.
2.2 Data Processing Pipeline
The Figure 2 illustrates the end-to-end data collection, filtering, annotation, and validation pipeline
designed for VoiceSculptor. The pipeline starts from large-scale in-the-wild data and in-house data,
which are first subjected to a series of automatic preprocessing steps, including denoising1, voice activity
detection (VAD)2, multi-speaker detection3, and perceptual quality filtering4, to ensure basic acoustic
cleanliness and speaker consistency.
For linguistic alignment, automatic speech recognition (ASR) and forced alignment are applied to obtain
accurate transcriptions and punctuation information. Specifically, FireRedASR (Xu et al., 2025b) is used to
transcribe Chinese audio, Whisper (Radford et al., 2022) is employed for English audio, and SenseVoice
(An et al., 2024) is used for ASR cross-validation, as well as language and emotion recognition.The
resulting transcripts are further processed using the Kaidi5alignment tool to perform character-level (for
Chinese) or word-level forced alignment, producing precise timestamps and pause durations between
adjacent tokens. Based on these fine-grained temporal cues, a trained punctuation prediction model is
applied to restore punctuation marks, yielding linguistically coherent and temporally aligned text–audio
pairs.
Based on the aligned text–audio representations, a structured annotation process is subsequently con-
ducted across multiple levels. At the advanced acoustic level, each audio sample is first analyzed using
Gemini 2.5 Pro to obtain multi-dimensional annotations, including pitch, speaking rate, loudness, speaker
gender and age, emotional state, paralinguistic characteristics, and contextual attributes. Based on
these structured annotations, DeepSeek is subsequently employed to perform translation and caption
generation, producing natural-language descriptions of the audio style and vocal characteristics. To
mitigate hallucinations introduced by large language models, we apply a set of rule-based regular expres-
sion filters to remove inconsistent or unsupported content. This process yields relatively high-quality,
semantically grounded annotations and captions that serve as reliable supervision for instruction-driven
voice design. At the emotional level, speech is annotated through a cross-validated emotion labeling
process. Specifically, we employ multiple complementary models, including Emo2Vec (Ma et al., 2024) ,
Qwen3-8B (Yang et al., 2025) , SenseVoice, and Qwen3-Omni (Xu et al., 2025a), to independently predict
emotion-related attributes from the audio. The outputs from these models are then cross-validated to
resolve inconsistencies and improve label reliability, resulting in a final set of emotion annotations with
higher robustness and accuracy. At the prosodic level, fine-grained acoustic attributes are annotated
through a combination of automatic estimation, statistical analysis, and human verification. Specifically,
we employ the DataSpeech (Lyth & King, 2024) model to estimate continuous prosodic features, includ-
1https://github.com/Audio-WestlakeU/CleanMel
2https://github.com/wiseman/py-webrtcvad
3https://github.com/pyannote/pyannote-audio
4https://github.com/AndreevP/wvmos
5https://github.com/sunsetsonwheels/kaidi
4

2026-01-16
ing the mean and standard deviation of pitch, mean and standard deviation of energy, speaking rate,
and loudness. Speaker gender and age are annotated using VoxProfile (Feng et al., 2025) . Based on the
extracted attributes, we perform distributional analysis across the dataset and conduct targeted human
listening to calibrate attribute boundaries. The continuous prosodic features are discretized into five
intervals for each dimension, while age is categorized into four groups (child, youth, middle-aged, and
elderly). Finally, for samples originating from in-house datasets, we leverage available ground-truth
annotations to correct and refine the predicted age and gender labels, resulting in a set of reliable and
structured prosodic annotations.
All annotations are aggregated into a unified annotated data pool, where natural language descriptions
are structured and regularized to form consistent instruction-like text representations.
Through this comprehensive and iterative pipeline, VoiceSculptor constructs a high-quality, multi-
dimensionally annotated dataset that supports natural-language-driven voice control, fine-grained
attribute manipulation, and robust instruction understanding.
2.3 CoT–based Fine-grained Attribute Modeling
Various fine-grained acoustic attributes in speech signals—such as pitch, loudness, speaking rate, and
temporal dynamics—play a critical role in shaping prosodic structure, expressive rhythm, and overall
vocal style. However, directly conditioning speech synthesis models on explicit attribute tokens often
leads to brittle control and over-reliance on structured inputs, limiting the model’s ability to generalize to
diverse natural language instructions.
To address this challenge, we introduce a CoT–based fine-grained attribute modeling strategy that explic-
itly guides the model to reason over acoustic attributes through intermediate semantic representations.
Instead of treating fine-grained attributes as independent control signals, CoT organizes attribute infor-
mation into structured reasoning steps that bridge natural language instructions and acoustic realizations.
This design enables the model to interpret high-level textual descriptions, decompose them into attribute-
related semantics, and subsequently generate speech tokens that reflect the desired prosodic and stylistic
characteristics.
Based on this formulation, VoiceSculptor jointly models instruction text, CoT-based fine-grained at-
tribute tokens, and discrete speech tokens within a unified autoregressive framework. This allows the
model to explicitly control multiple acoustic dimensions during synthesis, supporting precise prosody
manipulation and flexible rendering of diverse vocal styles.
Furthermore, to prevent over-dependence on explicit attribute tokens and to encourage deeper instruction
understanding, we introduce a stochastic attribute token dropout strategy during training. Fine-grained
attribute tokens are randomly removed from the input with a predefined probability such as 0.2, forcing
the model to infer the intended acoustic attributes from natural language instructions and contextual
cues alone. This training strategy acts as an effective regularizer and improves the model’s robustness
and generalization in instruction-driven voice control.
2.4 Retrieval-augmented Instruction Generalization
To improve the model’s generalization ability and robustness when handling out-of-domain natural
language instructions, we incorporate a RAG mechanism into the inference pipeline. This design enables
the model to leverage prior knowledge encoded in semantically related in-domain instructions, thereby
reducing sensitivity to distributional shifts in input prompts.
Specifically, we construct a vector-based instruction repository by embedding a large-scale collection
of500K in-domain natural language instructions, which follow organizational and syntactic patterns
similar to those observed in the training data, using the Qwen3-Embedding-0.6B model (Zhang et al.,
2025b). The resulting high-dimensional semantic representations are stored in a Milvus6vector database,
enabling efficient large-scale similarity search during retrieval.
During inference, when retrieval augmentation is enabled, the incoming natural language instruction is
first converted into a dense vector representation using the same embedding model. A cosine similar-
ity–based semantic search is then performed against the vector database to identify the most relevant
in-domain instructions. The retrieved instructions, which are semantically aligned with the input query,
are subsequently injected into the model input, guiding the model toward more stable and accurate
interpretation of the user’s intent.
By grounding the generation process in semantically similar in-domain examples, the proposed retrieval-
6https://github.com/milvus-io/milvus
5

2026-01-16
augmented framework effectively mitigates the impact of unseen or structurally diverse instructions. This
approach enhances both the robustness and controllability of the model under open-ended instruction
scenarios, leading to more consistent and reliable generation performance across a wide range of natural
language inputs.
3 Experiments
We conduct extensive experiments to comprehensively evaluate the effectiveness, controllability, and
scalability of the proposed VoiceSculptor framework, with a particular focus on its voice design (VD)
module. Given that our training data primarily consists of Chinese instruction–speech pairs, we adopt the
InstructTTSEval-Zh benchmark (Huang et al., 2025) as the main evaluation protocol to assess instruction-
following performance in a controlled and fair setting. We compare VoiceSculptor against both strong
open-source baselines and representative commercial systems, and further analyze its performance under
different model sizes, data scales, and training strategies. In addition, we perform a series of ablation
studies to isolate the contributions of key components, including CoT-based fine-grained attribute tokens,
text-side cross-entropy supervision, and RAG. Human subjective evaluations are also conducted to
complement automated metrics and validate perceptual instruction adherence. Overall, the experimental
results consistently demonstrate that VoiceSculptor achieves state-of-the-art performance among open
instruction-following TTS systems, while exhibiting robust scalability and strong controllability across a
wide range of settings.
3.1 Evaluation On InstructTTSEval-Zh Benchmark
Our training data predominantly consists of Chinese instruction–speech pairs. Therefore, we focus our
evaluation on the model’s instruction-following and controllability performance in Chinese, which allows
for a fair and consistent assessment aligned with the training distribution. We emphasize that this choice
does not imply any inherent limitation of the proposed method to Chinese. Instead, Chinese is adopted
as a representative language to demonstrate the feasibility and effectiveness of our approach, which can
be naturally extended to other languages given appropriate instruction data.
To this end, we employ InstructTTSEval-Zh, a Chinese instruction-based TTS evaluation benchmark
designed to measure how well a model follows natural-language instructions in speech synthesis. The
benchmark evaluates multiple aspects of instruction controllability, including attribute perception and
synthesis accuracy (APS), description–speech consistency (DSD), and response precision (RP). These
metrics are computed by synthesizing speech from instruction prompts and assessing the generated
audio using a unified evaluation protocol with a large language model as the evaluator.
Table 1: Performance Comparison Across Different Models on InstructTTSEval-Zh Benchmark
Model APS (%) DSD (%) RP (%) AVG (%)
Gemini 2.5-Flash* 88.2 90.9 77.3 85.4
Gemini 2.5-Pro* 89.0 90.1 75.5 84.8
GPT-4o-Mini-TTS* 54.9 52.3 46.0 51.1
ElevenLabs* 42.8 50.9 59.1 50.9
VoxInstruct (Zhou et al., 2024) 47.5 52.3 42.6 47.5
MiMo-Audio-7B-Instruct (Zhang et al., 2025a) 70.166.157.1 64.5
VoiceSculptor-VD 75.7 64.761.5 67.6
VoiceSculptor-VD & VC77.265.1 59.6 67.3
* indicates commercial models, while the others are open-source. All metrics follow InstructTTSEval. For ElevenLabs
and MiMo-Audio-7B-Instruct, speech samples are generated using either the official APIs or the released open-source
models, and evaluated with Gemini 2.5 Pro, from which the reported results of the compared models are obtained.
As shown in the table 1, all results of VoiceSculptor are reported with RAG enabled, where an external
instruction text repository provides semantically aligned in-domain guidance during inference, ensuring
stable and robust instruction grounding for the voice design module.
Under this evaluation setting, VoiceSculptor-VD consistently outperforms all open and instruction-tuned
baselines across the majority of metrics. In particular, it achieves the best performance in APS and RP ,
indicating superior accuracy in attribute perception and instruction-to-acoustic rendering within the
voice design module. These results demonstrate that, when combined with retrieval-based instruction
grounding, the VD module of VoiceSculptor is highly effective at translating abstract natural language
6

2026-01-16
descriptions into fine-grained and controllable vocal characteristics, which constitutes the core objective
of voice design.
Although Gemini 2.5-Flash and Gemini 2.5-Pro achieve strong overall scores, they are commercial
proprietary models and rely on closed-source infrastructures. In contrast, VoiceSculptor achieves state-of-
the-art performance among open-source instruction-following TTS models, while maintaining a unified,
instruction-driven voice design framework.
Compared with MiMo-Audio-7B-Instruct, which achieves slightly higher performance on DSD, VoiceSculptor-
VD shows clear advantages in APS and RP , leading to the highest overall AVG score among open systems.
This indicates that the voice design module of VoiceSculptor prioritizes precise and consistent attribute
control over isolated stylistic similarity, resulting in more faithful and reliable instruction-following
behavior at the voice design stage. The substantial performance margin over VoxInstruct and GPT-4o-
Mini-TTS further highlights the effectiveness of the design choices specific to the VD module, including
fine-grained attribute modeling, instruction-aware training, and robust scaling strategies.
To further account for the requirements of downstream speech synthesis tasks, particularly the preser-
vation of stylistic characteristics conveyed by prompt waveforms, we additionally report results for
VoiceSculptor-VD and VC. In this setting, the prompt waveform generated by the voice design module,
together with the corresponding test text, is fed into the CosyVoice2 model to perform downstream
speech synthesis. This evaluation protocol allows us to directly assess whether the vocal style specified
and generated at the voice design stage can be faithfully retained when transferred to a subsequent
synthesis model.Experimental results indicate that the stylistic attributes encoded in the prompt wave-
forms produced by VoiceSculptor-VD are largely preserved in the downstream synthesized speech.
Despite the introduction of an additional synthesis stage and a different model architecture, the generated
speech maintains strong consistency with the intended vocal characteristics, demonstrating effective
style transfer and robustness of the designed prompt representations. These findings suggest that the
voice design outputs of VoiceSculptor-VD are not only effective in isolation, but also serve as reliable
and reusable conditioning signals for downstream text-to-speech systems, thereby supporting practical
deployment scenarios where voice design and speech synthesis are decoupled.
Overall, these results demonstrate that the voice design module of VoiceSculptor achieves state-of-the-art
performance among instruction-following TTS systems on the voice design benchmark. The consistent
improvements across multiple complementary metrics confirm that the VD module can accurately
interpret natural language instructions and reliably generate the desired vocal attributes, establishing
VoiceSculptor’s voice design component as a strong and practical solution for controllable voice design
in instruction-driven text-to-speech systems.
3.2 Scaling Study On Model Size And Data Size
Evaluating instruction-following performance using large-scale automated benchmarks requires repeated
calls to proprietary models such as Gemini, which significantly increases evaluation cost and latency. To
enable rapid and iterative validation of scaling trends, we construct an internal lightweight subjective
evaluation benchmark consisting of 100 carefully curated test instructions and their corresponding
synthesized speech outputs.
Table 2: Overview of Training Data Composition
Dataset Size Dataset Name Data Sources
1,000 hSFT Data1 In-the-wild data.
3,700 hSFT Data2 Combination of in-the-wild data and internal data.
4,000 hSFT Data3 Combination of more in-the-wild data and internal
data.
9,000 hCPT Data4 Combination of more in-the-wild data , emotion-
filtered samples from the open-sourceVoxBox(Wang
et al., 2025) dataset, and internal data.
We conduct a human listening study focusing on instruction-following capability, measured by Instruction-
following Mean Opinion Score (IMOS). IMOS specifically assesses how well the synthesized speech
adheres to the semantic and stylistic requirements expressed in the natural language instruction, rather
than overall audio quality alone. A total of 33 human listeners participated in the evaluation. Each
listener is randomly assigned 10 audio samples drawn from the 100-sample test set, ensuring that each
test sample is evaluated by multiple listeners while avoiding listener fatigue. Listeners are asked to
provide subjective scores based on how accurately the synthesized speech follows the given instruction,
7

2026-01-16
Table 3: Scaling Study on Model Size and Data Size
Training Configuration IMOS APS (%) DSD (%) RP (%) AVG (%)
1B /SFT Data1 , 3 epoch 3.09 51.3 48.2 35.7 45.1
3B /SFT Data1 , 3 epoch 3.24 59.2 53.1 39.4 50.6
1B /SFT Data2 , 3 epoch 3.35 61.5 55.9 45.9 54.4
3B /SFT Data2 , 3 epoch 3.58 72.4 60.0 52.8 61.8
3B /CPT Data4 , 2 epoch andSFT Data3 , 3 epoch 3.6775.7 64.7 61.5 67.6
using a standardized MOS-style rating scale. The final IMOS score is computed by averaging all listener
ratings across samples.
The results in 3 demonstrate that model performance consistently benefits from both increased model
capacity and enlarged, more diverse training data. Under identical Supervised Fine-Tuning (SFT) settings,
scaling the model from 1B to 3B parameters yields clear improvements across all metrics, indicating
stronger representation and generalization capabilities. For a fixed model size, expanding SFT data
from in-the-wild only to larger mixtures with internal data leads to substantial gains, highlighting the
importance of data scale and diversity. Finally, incorporating large-scale emotion-aware continual pre-
training (CPT) on CPT Data4 prior to SFT achieves the best overall performance, suggesting that CPT
provides a more favorable initialization and enables the model to better exploit downstream supervised
data for the voice design task.Figure 3 and Figure 4 further corroborate these findings by showing
consistently lower validation loss for larger models, and richer training data throughout training.
Overall, these results indicate that incorporating CoT fine-grained attribute tokens, together with stochas-
tic attribute dropout, provides a robust and scalable enhancement, yielding consistent validation loss
reduction across different model sizes.
3.3 Ablation Study On CoT-based Fine-grained Attribute Tokens
Table 4 presents the ablation results of the proposed CoT-based fine-grained attribute tokens on the
InstructTTSEval-Zh evaluation benchmark. Incorporating CoT leads to consistent and substantial im-
provements across all evaluation metrics, including IMOS, APS, DSD, RP , and the overall AVG score,
demonstrating the effectiveness of explicit chain-of-thought modeling for voice design tasks. These
gains are observed without altering the model architecture, indicating that CoT primarily enhances
controllability and attribute understanding rather than relying on increased model complexity.
Table 4: Ablation Study of CoT-based fine-grained attribute tokens
Setting IMOS APS (%) DSD (%) RP (%) AVG (%)
VoiceSculptor-VD 3.6775.7 64.7 61.5 67.6
VoiceSculptor-VD w/o CoT 3.59 71.6 61.9 58.9 63.5
Across both model scales, models equipped with CoT-based attribute tokens achieve better overall
performance, suggesting that the proposed approach generalizes well to different model capacities.
Figure 3 provides supporting evidence from validation loss trends, showing that CoT-enhanced models
exhibit more favorable optimization behavior during training.
Moreover, randomly dropping each attribute token with a probability of 0.2 during training does not
degrade performance. On the contrary, the consistent improvements observed in Table 4 indicate that
this stochastic token dropout strategy serves as an effective regularization mechanism, encouraging the
model to robustly integrate attribute information without overfitting to specific tokens.
The ablation study on CoT-based fine-grained attribute tokens validates the effectiveness of introducing
auxiliary attribute tokens into the voice design framework. As shown by the consistent improvements
across all evaluation metrics, these tokens enable the model to better capture and utilize fine-grained
attribute information, leading to more accurate and controllable speech generation. Beyond quantitative
gains, the use of CoT-based attribute tokens facilitates finer-grained control over voice characteristics,
allowing more precise manipulation of individual attributes. This enhanced controllability is further
demonstrated through qualitative examples on our demo page, where the impact of fine-grained attribute
conditioning can be clearly observed.
8

2026-01-16
3.4 Ablation Study Of Text Cross-Entropy Loss
Based on the results in Table 5, introducing the text-side cross-entropy (CE) loss during training proves to
be highly effective. By jointly modeling the instruction text and audio tokens within a unified training
objective, the model achieves consistent and significant improvements across all evaluation metrics.
Compared to the setting without text CE loss, incorporating text CE loss leads to notable gains in IMOS,
APS, DSD, RP , and the overall AVG score, indicating enhanced alignment between textual instructions
and generated speech.
Table 5: Ablation Study Of Text Cross Entropy Loss
Setting IMOS APS (%) DSD (%) RP (%) AVG (%)
VoiceSculptor-VD 3.6775.7 64.7 61.5 67.6
VoiceSculptor-VD w/o Text CE Loss 3.42 67.9 59.4 58.2 61.8
These results suggest that explicitly supervising the text modality encourages the model to better capture
long-range contextual dependencies and semantic intent conveyed by instructions, rather than treating
text merely as auxiliary conditioning. As a result, the model develops stronger context understanding and
instruction-following capabilities, which directly translate into improved controllability and perceptual
quality in the voice design task.
3.5 Ablation Study Of RAG
Table 6 demonstrates the necessity and effectiveness of incorporating RAG into the proposed VoiceSculptor-
VD framework. Enabling RAG leads to substantial improvements across all evaluation metrics, with
especially pronounced gains in APS (+7.1%), RP (+13.0%), and the overall AVG score (+8.2%). These
results indicate that external retrieval provides critical contextual and attribute-related information that
the model alone may fail to infer reliably from the instruction text, thereby significantly enhancing
controllability and instruction adherence in the voice design task.
Table 6: Ablation Study of RAG
Setting IMOS APS (%) DSD (%) RP (%) AVG (%)
VoiceSculptor-VD 3.6775.7 64.7 61.5 67.6
VoiceSculptor-VD w/o RAG 3.39 68.6 61.1 48.5 59.4
At the same time, the large performance gap between the RAG and non-RAG settings also reveals inherent
limitations of the current model. In particular, the model exhibits relatively limited text understanding
and generalization when relying solely on its internal representations, making it sensitive to instruction
phrasing and less robust to unseen or complex textual descriptions. By supplementing the model
with retrieved examples and structured attribute information, RAG effectively compensates for these
shortcomings.
In future work, we plan to address these limitations more fundamentally by strengthening the model’s
text understanding capabilities. Specifically, we will explore incorporating large-scale text data during
the pre-training stage to better preserve and enhance linguistic representations, performing instruction
data augmentation to improve robustness and generalization, and adopting more semantically expressive
audio representations by replacing the current xcodec2 with a more semantically expressive codec, thereby
alleviating the model’s reliance on external retrieval while maintaining strong instruction-following
performance.
4 Conclusion
In this work, we present VoiceSculptor, a unified instruction-driven voice design framework that enables
fine-grained and controllable speech synthesis through natural language. Extensive evaluations on the
InstructTTSEval-Zh benchmark demonstrate that VoiceSculptor-VD achieves state-of-the-art performance
among open-source instruction-following TTS systems, consistently outperforming strong baselines
across multiple complementary metrics. Our scaling study confirms that instruction-following capability
benefits predictably from increased model capacity, richer training data, and staged training strategies
with large-scale continual pre-training. Through systematic ablation studies, we further validate the effec-
tiveness of key design choices, including CoT-based fine-grained attribute tokens, text-side cross-entropy
9

2026-01-16
supervision, and retrieval-augmented generation, each of which contributes to improved instruction
understanding, controllability, and robustness. In particular, CoT-based attribute modeling enables more
precise and interpretable control over vocal characteristics, while RAG effectively compensates for the
model’s limited text generalization by providing semantically aligned in-domain guidance. Finally, we
demonstrate that the voice design outputs of VoiceSculptor can be reliably transferred to downstream
speech synthesis models, supporting practical deployment scenarios where voice design and speech
generation are decoupled. Together, these results establish VoiceSculptor as a scalable and effective
solution for instruction-driven voice design.
5 Ethics Statement
Do not use this model for unauthorized voice cloning, impersonation, fraud, scams, deepfakes, or any
illegal or malicious activities. Ensure compliance with local laws and regulations when using this model
and uphold ethical standards. The developers assume no liability for any misuse of this model. Important
clarification regarding generated voices: As a generative model, the voices produced by this system
are synthetic outputs inferred by the model, not recordings of real human voices. The generated voice
characteristics do not represent or reproduce any specific real individual, and are not derived from or
intended to imitate identifiable persons. We advocate for the responsible development and use of AI and
encourage the community to uphold safety and ethical principles in AI research and applications.
10

2026-01-16
6 Appendix
Figure 3: Validation loss curves for the ablation study on CoT and model scale.
For each model scale, the curves with lower validation loss correspond to models equipped with CoT-
based fine-grained attribute tokens, while the curves with higher validation loss denote the baseline
models without CoT attribute tokens, where each attribute token is randomly dropped with a probability
of 0.2 during training.The upper orange/green curves represent the 1B model trained on 8 ×L40 GPUs,
and the lower light-blue/light-orange curves represent the 3B model trained on 8 ×A100 GPUs.All curves
report validation loss as a function of training steps.Our group’s ablation experiments were validated
using 3700 hours of data.We selected the epoch with the lowest validation set loss, i.e., epoch 3, as the
final model for each of our experiments.
Figure 4: Validation loss curves of the SFT stage under different data configurations.
All models share the same 3B parameter architecture and are trained using identical optimization and
training settings on 8 ×A100 GPUs. The only difference lies in the training data scale and pretraining
strategy.From top to bottom, the curves correspond to 1000h SFT, 3700h SFT, and 9k h continued pre-
training (CPT) followed by 3700h SFT, respectively.Consistent with previous observations, increasing the
amount of supervised data reduces validation loss, while large-scale CPT further improves convergence
and generalization, yielding the lowest loss throughout the SFT stage.
11

2026-01-16
References
Keyu An, Qian Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Yue Gu, Ting He, Hangrui
Hu, Kai Hu, Shengpeng Ji, Yabin Li, Zerui Li, Heng Lu, Haoneng Luo, Xiang Lv, Bin Ma, Ziyang
Ma, Chongjia Ni, Changhe Song, Jiaqi Shi, Xian Shi, Hao Wang, Wen Wang, Yuxuan Wang, Zhangyu
Xiao, Zhijie Yan, Yexin Yang, Bin Zhang, Qinglin Zhang, Shiliang Zhang, Nan Zhao, and Siqi Zheng.
Funaudiollm: Voice understanding and generation foundation models for natural interaction between
humans and llms, 2024. URLhttps://arxiv.org/abs/2407.04051.
Yushen Chen, Zhikang Niu, Ziyang Ma, Keqi Deng, Chunhui Wang, JianZhao JianZhao, Kai Yu, and
Xie Chen. F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching. InProceedings
of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.
6255–6271, 2025.
Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang,
Changfeng Gao, Hui Wang, et al. Cosyvoice 2: Scalable streaming speech synthesis with large
language models.arXiv preprint arXiv:2412.10117, 2024.
Tiantian Feng, Jihwan Lee, Anfeng Xu, Yoonjeong Lee, Thanathai Lertpetchpun, Xuan Shi, Helin Wang,
Thomas Thebaud, Laureano Moro-Velazquez, Dani Byrd, Najim Dehak, and Shrikanth Narayanan.
Vox-profile: A speech foundation model benchmark for characterizing diverse speaker and speech
traits, 2025. URLhttps://arxiv.org/abs/2505.14648.
Kexin Huang, Qian Tu, Liwei Fan, Chenchen Yang, Dong Zhang, Shimin Li, Zhaoye Fei, Qinyuan Cheng,
and Xipeng Qiu. Instructttseval: Benchmarking complex natural-language instruction following in
text-to-speech systems, 2025. URLhttps://arxiv.org/abs/2506.16381.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https://arxiv.org/ab
s/2005.11401.
Hanzhao Li, Yuke Li, Xinsheng Wang, Jingbin Hu, Qicong Xie, Shan Yang, and Lei Xie. Flespeech: Flexibly
controllable speech generation with various prompts.arXiv preprint arXiv:2501.04644, 2025.
Guanghou Liu, Yongmao Zhang, Yi Lei, Yunlin Chen, Rui Wang, Zhifei Li, and Lei Xie. Prompt-
style: Controllable style transfer for text-to-speech with natural language descriptions.arXiv preprint
arXiv:2305.19522, 2023.
Dan Lyth and Simon King. Natural language guidance of high-fidelity text-to-speech with synthetic
annotations, 2024. URLhttps://arxiv.org/abs/2402.01912.
Ziyang Ma, Zhisheng Zheng, Jiaxin Ye, Jinchao Li, Zhifu Gao, Shiliang Zhang, and Xie Chen. emotion2vec:
Self-supervised pre-training for speech emotion representation. InFindings of the Association for
Computational Linguistics: ACL 2024, pp. 15747–15760, 2024.
Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust
speech recognition via large-scale weak supervision, 2022. URLhttps://arxiv.org/abs/2212.04356.
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix,
Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin,
Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023.
URLhttps://arxiv.org/abs/2302.13971.
Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi
Zheng, Rui Wang, Xiaoqin Feng, et al. Spark-tts: An efficient llm-based text-to-speech model with
single-stream decoupled speech tokens.arXiv preprint arXiv:2503.01710, 2025.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou,
et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in neural
information processing systems, 35:24824–24837, 2022.
Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang
Zhang, Jingbei Li, et al. Step-audio 2 technical report.arXiv preprint arXiv:2507.16632, 2025.
12

2026-01-16
Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting
He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang,
Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen,
Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou,
Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin. Qwen3-omni technical report, 2025a.
URLhttps://arxiv.org/abs/2509.17765.
Kai-Tuo Xu, Feng-Long Xie, Xu Tang, and Yao Hu. Fireredasr: Open-source industrial-grade mandarin
speech recognition models from encoder-decoder to llm integration, 2025b. URL https://arxiv.org/
abs/2501.14350.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
Zhen Ye, Xinfa Zhu, Chi-Min Chan, Xinsheng Wang, Xu Tan, Jiahe Lei, Yi Peng, Haohe Liu, Yizhu Jin,
Zheqi Dai, et al. Llasa: Scaling train-time and inference-time compute for llama-based speech synthesis.
arXiv preprint arXiv:2502.04128, 2025.
Dong Zhang, Gang Wang, Jinlong Xue, Kai Fang, Liang Zhao, Rui Ma, Shuhuai Ren, Shuo Liu, Tao
Guo, Weiji Zhuang, et al. Mimo-audio: Audio language models are few-shot learners.arXiv preprint
arXiv:2512.23808, 2025a.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text
embedding and reranking through foundation models, 2025b. URL https://arxiv.org/abs/2506.051
76.
Yongmao Zhang, Guanghou Liu, Yi Lei, Yunlin Chen, Hao Yin, Lei Xie, and Zhifei Li. Promptspeaker:
Speaker generation based on text descriptions. In2023 IEEE Automatic Speech Recognition and Under-
standing Workshop (ASRU), pp. 1–7. IEEE, 2023.
Ziyu Zhang, Hanzhao Li, Jingbin Hu, Wenhao Li, and Lei Xie. Histyle: Hierarchical style embedding
predictor for text-prompt-guided controllable speech synthesis. InNational Conference on Man-Machine
Speech Communication, pp. 522–535. Springer, 2025c.
Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, and Jingchen Shu. Indextts2: A
breakthrough in emotionally expressive and duration-controlled auto-regressive zero-shot text-to-
speech.arXiv preprint arXiv:2506.21619, 2025.
Yixuan Zhou, Xiaoyu Qin, Zeyu Jin, Shuoyi Zhou, Shun Lei, Songtao Zhou, Zhiyong Wu, and Jia
Jia. Voxinstruct: Expressive human instruction-to-speech generation with unified multilingual codec
language modelling. InProceedings of the 32nd ACM International Conference on Multimedia, pp. 554–563,
2024.
13