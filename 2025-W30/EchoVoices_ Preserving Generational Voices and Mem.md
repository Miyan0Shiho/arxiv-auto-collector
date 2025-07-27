# EchoVoices: Preserving Generational Voices and Memories for Seniors and Children

**Authors**: Haiying Xu, Haoze Liu, Mingshi Li, Siyu Cai, Guangxuan Zheng, Yuhuang Jia, Jinghua Zhao, Yong Qin

**Published**: 2025-07-21 03:47:45

**PDF URL**: [http://arxiv.org/pdf/2507.15221v1](http://arxiv.org/pdf/2507.15221v1)

## Abstract
Recent breakthroughs in intelligent speech and digital human technologies
have primarily targeted mainstream adult users, often overlooking the distinct
vocal patterns and interaction styles of seniors and children. These
demographics possess distinct vocal characteristics, linguistic styles, and
interaction patterns that challenge conventional ASR, TTS, and LLM systems. To
address this, we introduce EchoVoices, an end-to-end digital human pipeline
dedicated to creating persistent digital personas for seniors and children,
ensuring their voices and memories are preserved for future generations. Our
system integrates three core innovations: a k-NN-enhanced Whisper model for
robust speech recognition of atypical speech; an age-adaptive VITS model for
high-fidelity, speaker-aware speech synthesis; and an LLM-driven agent that
automatically generates persona cards and leverages a RAG-based memory system
for conversational consistency. Our experiments, conducted on the SeniorTalk
and ChildMandarin datasets, demonstrate significant improvements in recognition
accuracy, synthesis quality, and speaker similarity. EchoVoices provides a
comprehensive framework for preserving generational voices, offering a new
means of intergenerational connection and the creation of lasting digital
legacies.

## Full Text


<!-- PDF content starts -->

EchoVoices: Preserving Generational Voices and
Memories for Seniors and Children
Haiying Xu1*[0009−0004−1965−8416], Haoze Liu1*[0009−0008−5775−984X], Mingshi
Li1[0009−0009−4213−5409], Siyu Cai1[0009−0002−5687−8149], Guangxuan
Zheng1[0009−0007−2918−2855], Yuhang Jia1[0009−0001−2407−0789], Jinghua
Zhao1[0009−0008−9648−7373], and Yong Qin1[0009−0000−2748−3020]†
College of Computer Science, Nankai University
Correspondence: 2212180@mail.nankai.edu.cn, qinyong@nankai.edu.cn
Abstract. Recent breakthroughs in intelligent speech and digital hu-
man technologies have primarily targeted mainstream adult users, often
overlooking the distinct vocal patterns and interaction styles of seniors
and children. These demographics possess distinct vocal characteristics,
linguistic styles, and interaction patterns that challenge conventional
ASR, TTS, and LLM systems. To address this, we introduce EchoVoices,
anend-to-enddigitalhumanpipelinededicatedtocreatingpersistentdig-
ital personas for seniors and children, ensuring their voices and memories
are preserved for future generations. Our system integrates three core
innovations: a k-NN-enhanced Whisper model for robust speech recog-
nition of atypical speech; an age-adaptive VITS model for high-fidelity,
speaker-aware speech synthesis; and an LLM-driven agent that auto-
matically generates persona cards and leverages a RAG-based memory
system for conversational consistency. Our experiments, conducted on
the SeniorTalk and ChildMandarin datasets, demonstrate significant im-
provements in recognition accuracy, synthesis quality, and speaker sim-
ilarity. EchoVoices provides a comprehensive framework for preserving
generational voices, offering a new means of intergenerational connec-
tion and the creation of lasting digital legacies.1
Keywords: Senior and Child Speech ·Digital Human ·Memory Preser-
vation ·Speech Recognition ·Speech Synthesis.
1 Introduction
Recent advances in intelligent speech and digital human technologies have pro-
duced sophisticated conversational agents for mainstream adult users. These
systems, however, often fail to accommodate the unique needs of seniors and
children. Due to distinct vocal patterns, prosody, and linguistic styles inherent
to these age groups, conventional Automatic Speech Recognition (ASR), Text-
to-Speech (TTS), and Large Language Model (LLM) frameworks struggle with
accuracy and naturalness, limiting their accessibility and utility.
*These authors contributed equally to this work, sorted by surname.
†Corresponding author.
1Our project is available at: https://github.com/lhz191/EchoVoices.gitarXiv:2507.15221v1  [cs.SD]  21 Jul 2025

2 H. Xu, H. Liu et al.
This technological gap carries a significant cost: the rich oral histories of the
elderly and the critical linguistic development of the young risk being unrecorded
in our digital age. To address this, we propose EchoVoices, a system founded on
theprincipleofholisticpreservation.Ratherthanpursuingsimpletechnicalfixes,
our work aims to create persistent and authentic digital personas that capture
the full essence of an individual’s identity, including their memories, personality,
and unique way of speaking.
TheEchoVoicespipelineintegratesspecializedmodulesforage-adaptivespeech
recognition and synthesis with an LLM-driven agent for consistent memory man-
agement. By combining these components, we create expressive digital avatars
capableofmeaningful,context-awareinteraction.Thisapproachprovidesanovel
framework for preserving generational wisdom and creating lasting digital lega-
cies, offering a new technological medium for intergenerational connection.
Our main contributions are as follows:
–We developed EchoVoices , a holistic framework to generate expressive digi-
tal personas for seniors and children, which integrates specialized ASR, TTS,
andretrieval-augmentedLLMmodulestoauthenticallycaptureandpreserve
their unique voices, memories, and interaction styles.
–We introduce a k-NNaugmentation for Whisper that significantly improves
ASR on atypical speech, and a two-stage training strategy for VITS that
yields high-fidelity, age-adaptive speech synthesis.
–We deploy an LLM-driven agent that automatically distills a user’s per-
sona card from conversation and uses a RAG-based memory to maintain a
consistent and evolving digital identity.
2 Related Work
2.1 Speech Recognition and Synthesis
Large-scale ASR models like Whisper [1] show impressive general capabilities
but often falter on the atypical speech of seniors and children. The k-Nearest-
Neighbor (k-NN) paradigm, which augments model predictions by retrieving
similar examples from a datastore [2], offers a powerful method for domain adap-
tation without costly retraining. While k-NN has been applied to CTC-based
ASR [3], its use to enhance large encoder-decoder models like Whisper for the
distinct challenges of elderly and child speech remains underexplored. For speech
synthesis, while models like VITS [4] enable high-quality, multi-speaker gener-
ation, they require extensive resources for zero-shot adaptation [5]. Our work
addresses these gaps by adapting the k-NN framework to improve fine-tuned
Whisper models and employing a resource-efficient, two-stage training strategy
for VITS, specifically for these demographics.
2.2 LLM-Driven Digital Personas
The paradigm of Large Language Models (LLMs) now extends to powering so-
phisticated, autonomous agents, where maintaining a consistent persona is crit-
ical. Retrieval-Augmented Generation (RAG) [6] is a key methodology for this,
grounding LLMs in external knowledge to ensure factual consistency. Our work

EchoVoices 3
applies this by first using an LLM to distill a foundational ’persona card’ from
dialogue and then implementing a RAG-based memory system to ensure a co-
herent and evolving digital identity. This is situated within the broader field of
digital human generation, where ASR is fundamental for animating talking faces
using methods like Wav2Lip [7] or GeneFace [8]. Our work contributes to this
field by enhancing ASR for seniors and children, enabling more empathetic and
effective applications.
3 Method
3.1 System Overview
Our system enables personalized, cross-age speech interaction by integrating
speech recognition, language understanding, speech synthesis, and talking-face
generation in a single pipeline (Fig. 1). The input is a spoken query from a child
or elderly speaker, and the output is a synchronized talking-face video with
high-quality, speaker-aware synthetic speech.
We employ a fine-tuned Whisper model to transcribe the input audio into
text, adapting it to elderly and child speech domains. The recognized text is then
passed to a large language model (LLM) equipped with retrieval-augmented
generation (RAG) and Self-Talk prompting, which generates a context-aware,
informative response. The LLM output is then fed into a two-stage trained
VITS model, which synthesizes personalized speech for elderly or child voice
characteristics. Finally, we use Wav2Lip combined with GFPGAN to render a
synchronized, photorealistic talking-face video driven by the generated speech.
This modular design enables both speaker-specific adaptation and cross-
modal generation, supporting engaging and inclusive voice interaction for un-
derrepresented user groups.
3.2 k-NN Enhanced ASR for Specialized Demographics
To augment our fine-tuned Whisper model, we integrate a k-Nearest-Neighbors
(k-NN) mechanism that leverages instance-based knowledge at inference time.
First, we construct a domain-specific datastore by processing the training set
(e.g., elderly or child speech) with the fine-tuned model. For each token, the
final hidden state from the Whisper decoder is extracted as the key, and the
corresponding ground-truth token is the value. These key-value pairs are indexed
in a Faiss [9] datastore for efficient retrieval. During inference, at each step of the
beam search, the decoder’s current hidden state is used to query the datastore
for the k-nearest neighbors. Their corresponding tokens are used to form a non-
parametric probability distribution P kNN, which is interpolated with the model’s
original distribution P Whisperto produce the final prediction:
Pfinal = (1−λ)PWhisper +λPkNN
where λis a hyperparameter balancing the two distributions. This allows the
model to correct predictions for domain-specific or rare patterns by consulting
an explicit memory.

4 H. Xu, H. Liu et al.
Fig. 1.The EchoVoices System Pipeline. (a) A spoken query from a senior or child is
transcribed by our k-NN enhanced Whisper ASR model. (b) The text is processed by
an LLM-driven agent, which uses RAG to query a memory database and generate a
persona-consistent response. (c) The response text is synthesized into age-appropriate
speech by a two-stage fine-tuned VITS model.
3.3 Age-Adaptive TTS with VITS Fine-tuning
Tosynthesizeage-appropriatespeech,weemployatwo-stagetrainingstrategyon
the VITS architecture [4]. First, a model is pretrained on a large subset (90%) of
the target demographic data (SeniorTalk or ChildMandarin) to learn its general
acoustic and prosodic characteristics, such as vocal tremor in seniors or higher
fundamental frequencies in children. For adaptation to new, unseen speakers, we
thenre-initializethespeakerembeddinglayerwith e(0)
speaker∼ N (0, σ2I)andfine-
tune the model on the held-out 10% of speakers. This re-initialization prevents
overfitting to the pretrained identities and facilitates rapid adaptation to the
newspeaker’suniquevocalcharacteristics,ensuringnaturalanddemographically
appropriate speech synthesis.
3.4 LLM-Driven Persona Generation
To imbue the digital human with a consistent and evolving identity, we employ
an LLM-driven agent centered around a dynamic persona card. This card, which
functions as the agent’s foundational prompt, is a structured summary of the
user’sbackground,linguisticstyle,andkeymemories,automaticallydistilledand
updated by an LLM analyzing the conversation. For conversational continuity,
this core persona is augmented by a RAG-based memory system. Salient facts
from the dialogue are stored in a vector database and retrieved via similarity
search when generating a new response. These retrieved memories are then com-

EchoVoices 5
Fig. 2.The LLM-driven agent pipeline. (a) Spoken dialogue is first collected and tran-
scribed into text using the fintuned ASR model. (b) The transcribed text is processed
by a large language model, which extracts user-specific identity cards and retrieves rel-
evant memory information using a retrieval-augmented generation (RAG) mechanism.
(c) Based on the persona card and retrieved memory, the agent generates character-
consistent dialogue responses as illustrative examples.
bined with the persona card in the LLM’s prompt, ensuring every response is
both in-character and grounded in the history of the interaction.
4 Experiments
4.1 Dataset
We use two Mandarin speech datasets targeting underrepresented age groups:
SeniorTalk [10] for elderly speech and ChildMandarin [11] for young children.
SeniorTalk contains 55.53 hours of spontaneous speech from 202 speakers
aged 75–85, collected across 16 provinces. It is balanced in gender, region, and
age, and includes annotations such as speaker ID, timestamps, and accent tags.
ChildMandarin includes 41.25 hours of speech from 397 speakers aged 3–
5, recorded via smartphones in 22 provinces, with balanced gender distribution.
Each audio is paired with character-level transcriptions and metadata including
age, gender, and region.
4.2 Experimental Setup
ASR Fine-tuning with Whisper For the ASR task, we evaluate four sizes of
the Whisper model (tiny, base, small, and medium) on both datasets, measuring
performance in Character Error Rate (CER). We compare three settings: (1)
Zero-shot performance using the off-the-shelf models, (2) Fine-tuned models
adapted on each domain-specific dataset, and (3) our proposed Fine-tuned +
k-NNapproach, which augments the fine-tuned model with k-NN decoding.
Results We present the ASR results on the SeniorTalk and ChildMandarin
datasets in Table 1 and Table 2, respectively. The results are reported in terms
of CER (%).

6 H. Xu, H. Liu et al.
Table 1. CER (%) on the SeniorTalk dataset. Fine-tuning and k-NN augmentation
significantly improve performance.
Model Parameters Zero-shot Fine-tuned Fine-tuned + k-NN
Whisper-tiny 39M 82.85 29.73 26.69
Whisper-base 74M 67.54 22.70 18.85
Whisper-small 244M 56.34 17.59 15.84
Whisper-medium 769M 48.61 27.96 14.78
Table 2. CER (%) on the ChildMandarin dataset. The high CER for the medium fine-
tuned model reflects a high deletion rate, a failure mode our k-NN approach mitigates.
Model Parameters Zero-shot Fine-tuned Fine-tuned + k-NN
Whisper-tiny 39M 72.49 28.73 29.17
Whisper-base 74M 51.10 22.31 22.58
Whisper-small 244M 30.93 21.03 18.36
Whisper-medium 769M 24.45 81.49 17.66
The results demonstrate a clear and consistent trend: fine-tuning provides a
dramatic improvement over the zero-shot baseline across both datasets, under-
scoring the necessity of domain adaptation for these specialized demographics.
On the SeniorTalk dataset, our k-NN enhancement method consistently pushes
the performance further, providing an additional relative improvement across
all model sizes and achieving a best overall CER of 14.78% with the medium
model. This highlights the benefit of combining parametric adaptation with non-
parametric, instance-based knowledge retrieval.
On the ChildMandarin dataset, our method shows compelling results. Due to
insufficient training data, the fine-tuned medium model suffers a critical failure,
frequently producing no output and resulting in a prohibitive 81.49% CER. Our
k-NN augmentation not only improves performance but robustly rectifies this
failure, reducing the CER to 17.66%. This result powerfully demonstrates that
the k-NN mechanism can serve as a crucial guide for the model, ensuring it
produces a valid output where it would otherwise fail.
4.3 TTS Fine-tuning with VITS
Experimental Setup We evaluate our TTS approach using a speaker inde-
pendent protocol on both datasets, partitioning speakers into 90% for pretrain-
ing and 10% for fine-tuning. We compare direct fine-tuning on the 10% subset
against our proposed method which first pretrains on the 90% subset. Synthesis
quality is measured by the Character Error Rate (CER) from our ASR model on
synthesized audio (CER f) versus ground-truth audio (CER g). Speaker similarity
is evaluated using ECAPA-TDNN, x-vector, and pyannote.
Results As shown in Table 3, our two-stage training strategy consistently im-
proves speaker similarity across all metrics. For instance, on the SeniorTalk
dataset, the ECAPA-TDNN score increases from 0.5608 to 0.5644, and simi-
lar gains are observed for ChildMandarin. This indicates that pretraining on

EchoVoices 7
a larger, demographically-matched dataset allows the model to establish a ro-
bust prior of age-specific acoustic features. This robust prior not only enhances
speaker similarity but also enables faster convergence during the fine-tuning
stage for new, unseen speakers.
Furthermore, this pretraining significantly enhances the intelligibility of the
synthesized speech, as reflected by the reduction in Character Error Rate on the
synthesized audio (CER f). On the SeniorTalk dataset, CER fdrops from 50.52%
to 45.63%, with comparable improvements on ChildMandarin. The effectiveness
of this approach stems from the division of labor: the pretraining stage captures
general characteristics like vocal tremor in seniors or higher fundamental fre-
quencies in children, while the subsequent fine-tuning stage, aided by speaker
embedding re-initialization, can focus more effectively on capturing the unique
timbre of the target speaker. This results in more natural and demographically
appropriate synthetic speech.
Table 3. CER and speaker similarity for TTS fine-tuning on SeniorTalk and Child-
Mandarin datasets. “wi pt” denotes pretraining on 90% of speakers. Fine-tuning is
conducted on the remaining 10% of unseen speakers.
ModelCER Speaker Similarity
CER gCER fECAPA x-vector pyannote
SeniorTalk (wo pretrain) 42.60 50.52 0.5608 0.9503 0.5593
SeniorTalk (wi pretrain 36.11 45.63 0.5644 0.9512 0.5637
ChildMandarin (wo pretrain) 55.44 42.58 0.5999 0.9566 0.5899
ChildMandarin (wi pretrain) 48.93 39.18 0.6116 0.9583 0.5972
5 Limitation
While our current TTS system based on VITS effectively improves synthesis
quality through speaker-specific fine-tuning, it incurs additional computational
cost. Future work may explore zero-shot speaker adaptation techniques to im-
prove scalability and eliminate the need for per-speaker customization.Secondly,
though the k-NN enhancement proves effective, it introduces increased inference-
time latency and computational overhead, highlighting the need for future op-
timization strategies such as datastore pruning or vector quantization. Finally,
while we propose a cascaded system for elderly and child users, its modular
design may introduce latency and inefficiency. Future research may explore the
development of a unified end-to-end multimodal large model to improve effi-
ciency and streamline the interaction process.
6 Conclusion
In this paper, we presented EchoVoices, a holistic framework designed to create
authentic digital personas for seniors and children, addressing the shortcomings
of conventional speech technologies for these demographics. We demonstrated
the effectiveness of a modular approach that integrates three key innovations:
a k-NN augmented Whisper model for robust, domain-adapted ASR; a two-
stage training strategy for VITS that yields high-fidelity, age-appropriate syn-
thetic speech; and an LLM-driven agent that maintains a consistent persona

8 H. Xu, H. Liu et al.
through automatically generated persona cards and a RAG-based memory. Our
experiments validated this approach, showing not only significant improvements
in ASR accuracy—where the k-NN mechanism was crucial in rectifying catas-
trophic model failure on child speech—but also enhanced speaker similarity and
intelligibility in TTS. Ultimately, EchoVoices provides a robust and compre-
hensive blueprint for developing more inclusive AI, enabling the preservation of
generational voices and creating new pathways for digital legacy and intergen-
erational connection.
References
1. Alec Radford, Jongho Wu, Rewon Child, David Luan, Dario Amodei, and Ilya
Sutskever. Robust speech recognition via large-scale weak supervision. Interna-
tional Conference on Machine Learning (ICML 2023). PMLR , 2023.
2. Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Yejin Choi.
Generalization through memorization: Nearest neighbor language models. In In-
ternational Conference on Learning Representations (ICLR 2020) , 2020.
3. Jiaming Zhou, Shiwan Zhao, Yaqi Liu, Wenjia Zeng, Yong Chen, and Yong Qin.
Knn-ctc: Enhancing asr via retrieval of ctc pseudo labels. In Proc. Interspeech
2023, pages 5550–5554, 2023.
4. Jaehyeon Kim, Jungil Kong, and Juwon Son. Conditional variational autoencoder
with adversarial learning for end-to-end text-to-speech. In International Confer-
ence on Machine Learning (ICML 2021) , pages 5530–5540. PMLR, 2021.
5. Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqi Zhang, Liyang Zhou, Sanyuan Liu,
Zhiheng Chen, Yong Liu, Huazhe Wang, Jiatong Li, Lixing He, Sheng Zhao, and
Furu Wei. Neural codec language models are zero-shot text to speech synthesizers.
arXiv preprint arXiv:2301.02111 , 2023.
6. Patrick Lewis, Ethan Perez, Aleksa Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Kementchedjhiev, Rishabh Lovin, Luyu Ma, Marc Lewis,
et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances
in Neural Information Processing Systems 33 (NeurIPS 2020) , 2020.
7. K.R. Prajwal, V.P. Kumar, V. Muralidhar, R. Gundavarapu, V.P. Namboodiri,
and A. Jain. A lip sync expert is all you need for speech to lip generation in the
wild. In ACM International Conference on Multimedia , 2020.
8. Zhenhui Ye, Ziyue Jiang, Yi Ren, Jinglin Liu, Jinzheng He, and Zhou Zhao. Gene-
face: Generalized and high-fidelity audio-driven 3d talking face synthesis. In In-
ternational Conference on Learning Representations (ICLR) , 2023.
9. JeffJohnson,MatthijsDouze,andH’erveJégou. Billion-scalesimilaritysearchwith
gpus. IEEE Transactions on Knowledge and Data Engineering (TKDE) , 2019.
10. Yang Chen, Hui Wang, Shiyao Wang, Junyang Chen, Jiabei He, Jiaming Zhou,
Xi Yang, Yequan Wang, Yonghua Lin, and Yong Qin. Seniortalk: A chinese con-
versation dataset with rich annotations for super-aged seniors. arXiv preprint
arXiv:2405.12345 , 2024.
11. Jiaming Zhou, Shiyao Wang, Shiwan Zhao, Jiabei He, Haoqin Sun, Hui Wang,
Cheng Liu, Aobo Kong, Yujie Guo, Xi Yang, Yequan Wang, Yonghua Lin, and
Yong Qin. Childmandarin: A comprehensive mandarin speech dataset for young
children aged 3-5. arXiv preprint arXiv:2409.18584 , 2024.