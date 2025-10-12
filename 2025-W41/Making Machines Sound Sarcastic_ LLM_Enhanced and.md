# Making Machines Sound Sarcastic: LLM-Enhanced and Retrieval-Guided Sarcastic Speech Synthesis

**Authors**: Zhu Li, Yuqing Zhang, Xiyuan Gao, Shekhar Nayak, Matt Coler

**Published**: 2025-10-08 14:53:48

**PDF URL**: [http://arxiv.org/pdf/2510.07096v1](http://arxiv.org/pdf/2510.07096v1)

## Abstract
Sarcasm is a subtle form of non-literal language that poses significant
challenges for speech synthesis due to its reliance on nuanced semantic,
contextual, and prosodic cues. While existing speech synthesis research has
focused primarily on broad emotional categories, sarcasm remains largely
unexplored. In this paper, we propose a Large Language Model (LLM)-enhanced
Retrieval-Augmented framework for sarcasm-aware speech synthesis. Our approach
combines (1) semantic embeddings from a LoRA-fine-tuned LLaMA 3, which capture
pragmatic incongruity and discourse-level cues of sarcasm, and (2) prosodic
exemplars retrieved via a Retrieval Augmented Generation (RAG) module, which
provide expressive reference patterns of sarcastic delivery. Integrated within
a VITS backbone, this dual conditioning enables more natural and contextually
appropriate sarcastic speech. Experiments demonstrate that our method
outperforms baselines in both objective measures and subjective evaluations,
yielding improvements in speech naturalness, sarcastic expressivity, and
downstream sarcasm detection.

## Full Text


<!-- PDF content starts -->

MAKING MACHINES SOUND SARCASTIC: LLM-ENHANCED AND RETRIEVAL-GUIDED
SARCASTIC SPEECH SYNTHESIS
Zhu Li, Yuqing Zhang, Xiyuan Gao, Shekhar Nayak, Matt Coler
University of Groningen, The Netherlands
ABSTRACT
Sarcasm is a subtle form of non-literal language that poses signifi-
cant challenges for speech synthesis due to its reliance on nuanced
semantic, contextual, and prosodic cues. While existing speech syn-
thesis research has focused primarily on broad emotional categories,
sarcasm remains largely unexplored. In this paper, we propose
a Large Language Model (LLM)-enhanced Retrieval-Augmented
framework for sarcasm-aware speech synthesis. Our approach com-
bines (1) semantic embeddings from a LoRA-fine-tuned LLaMA
3, which capture pragmatic incongruity and discourse-level cues
of sarcasm, and (2) prosodic exemplars retrieved via a Retrieval
Augmented Generation (RAG) module, which provide expressive
reference patterns of sarcastic delivery. Integrated within a VITS
backbone, this dual conditioning enables more natural and contextu-
ally appropriate sarcastic speech. Experiments demonstrate that our
method outperforms baselines in both objective measures and sub-
jective evaluations, yielding improvements in speech naturalness,
sarcastic expressivity, and downstream sarcasm detection.
Index Terms—Sarcastic speech synthesis, sarcasm detection,
large language models, retrieval augmented generation
1. INTRODUCTION
Sarcasm is a common yet challenging form of non-literal language,
often marked by prosodic patterns such as exaggerated intonation
and prolonged syllables [1, 2]. Its proper interpretation relies on a
nuanced interplay of semantic, contextual, and prosodic cues, mak-
ing it difficult to capture computationally. While sarcasm is widely
used in everyday communication, it has received limited attention
in the field of speech synthesis. Most existing work on expressive
text-to-speech (TTS) has concentrated on broad emotional categories
such as happiness, anger, or sadness [3–5], whereas more subtle and
pragmatically grounded attitudes such as sarcasm remain largely un-
explored. While prior work has attempted to analyze acoustic corre-
lates of sarcastic speech, only a limited number of studies have ex-
plored directly manipulating prosodic features such as pitch, pace,
and loudness to generate sarcastic-sounding speech [6]. In addition,
current TTS systems, while successful at producing intelligible and
emotionally expressive speech, often lack the nuanced expressive-
ness needed to convey sarcastic intent [5, 7, 8].
Sarcastic speech synthesis holds considerable potential for im-
proving human-computer interaction in applications such as conver-
sational agents and entertainment systems [9]. However, the task
presents two major challenges. First, high-quality, annotated sarcas-
tic speech corpora remain scarce, limiting the applicability of purely
data-driven approaches. Existing corpora for sarcasm, such as MUS-
tARD [10] or its extensions [11], are primarily developed for detec-
tion rather than synthesis. Second, sarcasm is inherently more subtle
and context-dependent than conventional emotions such as anger orjoy, making it difficult to capture with handcrafted acoustic features
or simple prosodic controls. In contrast, research on sarcasm de-
tection has progressed significantly. Recent work has shown that
multi-modal approaches, incorporating textual, prosodic and visual
information, can significantly improve sarcasm recognition [10–14].
These advances highlight the feasibility of modeling sarcasm com-
putationally. Yet, most efforts have remained on the recognition side,
leaving the generation of sarcastic speech largely unaddressed. Mov-
ing from detection to synthesis is not only a natural next step but
also a crucial one, as it enables interactive systems to actively de-
ploy pragmatic phenomena rather than passively identify them.
To address these limitations, recent advances in natural language
processing offer promising directions. LLMs have demonstrated the
ability to capture high-level semantic and pragmatic features from
text. In the TTS domain, integrating LLM-derived embeddings has
been shown to enable more nuanced prosody control and improve
emotional expressiveness and contextual appropriateness [15–17].
Such capabilities are particularly relevant for sarcasm, which of-
ten relies on pragmatic incongruity between literal meaning and in-
tended attitude. Leveraging LLM-based semantic representations
could therefore provide a principled way of conditioning TTS mod-
els on sarcasm-relevant cues that handcrafted features fail to capture.
Complementary to this, RAG allows models to condition gen-
eration on external knowledge or examples retrieved from large
databases [18, 19]. While RAG has primarily been explored in text-
based tasks such as question answering and dialogue [20, 21], its
potential for speech synthesis is evidenced by related work on style
transfer and expressive TTS, where reference utterances are used to
guide prosody [18]. Existing approaches, however, typically rely
on manually selected or speaker-specific reference audio, limiting
scalability and contextual alignment. In the case of sarcasm, where
annotated corpora are scarce, automatic retrieval of semantically
similar sarcastic speech could provide prosodic exemplars that both
enrich training and guide generation. This suggests a paradigm in
which LLMs supply semantic-pragmatic cues while RAG retrieves
expressive references, together enabling a systematic modeling of
subtle pragmatic phenomena such as sarcasm in speech synthesis.
This study introduces a unified sarcasm-aware speech synthe-
sis framework that enriches the TTS system with LLM-extracted
semantic cues indicative of sarcasm and prosodically aligned au-
dio references for expressive modeling. Specifically, we first use
an LLM to extract high-level semantic representations capturing
features relevant to sarcastic intent. In parallel, a retrieval mecha-
nism based on RAG identifies semantically similar and contextually
aligned prosodic examples from a curated speech database, provid-
ing prosodic exemplars that guide the TTS model toward expressive
and contextually appropriate output. Both objective metrics and
subjective evaluations demonstrate that the proposed method out-
performs baseline models in naturalness, expressivity, and accurate
sarcasm expression.arXiv:2510.07096v1  [cs.CL]  8 Oct 2025

Sarcasm-aware VITSSarcasticspeech databaseInput TextSemantic EmbeddingPhoneme EmbeddingReference SpeechSarcasm-LoRA                Llama 3-8BPhoneme EncoderWavLMLlama 3-8BCross-attentionSarcastictext dataset
Index ListIndex ListSynthesized Speech
Semantic Embeddingfirst tokenlast token…Input TextSemantic EmbeddingSarcasm-LoRA                Llama 3-8BQuery
TopKInput Text
a. LLM Fine-tuningb. Sarcastic Speech Synthesisc. RetrievalFig. 1. Overview of the sarcasm-aware speech synthesis framework. Semantic cues from a LoRA fine-tuned LLaMA 3 and prosodic exemplars
retrieved via RAG are fused within a VITS-based model to synthesize sarcastic speech.
2. METHODS
To achieve high-quality sarcastic speech synthesis, we propose a
Retrieval-Augmented LLM-enhanced TTS framework. An overview
of the architecture is illustrated in Figure 1.
2.1. Sarcasm-Aware Semantic Encoding via LoRA
Low-Rank Adaptation (LoRA) has recently been widely adopted for
adapting LLMs across diverse domains such as emotion recognition,
sentiment analysis, and affective detection, due to its parameter ef-
ficiency and effectiveness [22]. Motivated by these advances, we
leverage LoRA to finetune the LLaMA 3-8B model for sarcasm-
aware semantic encoding. Fine-tuning is performed on a curated
sarcasm-labeled dataset, enabling the model to capture signals of
sarcastic intent, including pragmatic incongruity and discourse-level
cues.
Formally, given an input textx, the Sarcasm-LoRA encoder pro-
duces a semantic embedding:
Es=f LoRA(x)∈RTt×dt,(1)
whered tdenotes the dimensionality of the semantic embedding.
These embeddings capture both contextual and pragmatic aspects
of the input, providing a sarcasm-aware representation for further
processing.
2.2. RAG for Prosody Conditioning
To enhance prosodic expressiveness, we integrate a RAG module
into the synthesis pipeline. The key idea is to leverage semanti-
cally aligned sarcastic utterances as prosodic exemplars, enriching
the generated speech with realistic intonation patterns. Concretely,
we first construct an indexD={(u i,ai)}N
i=1of sarcastic utterances
uicurated from MUStARD++, wherea idenotes their pre-computed
semantic representations. For a given input text, the semantic em-
beddingE sproduced by the Sarcasm-LoRA encoder is used as aretrieval query to fetch the top-Ksemantically and pragmatically
relevant sarcastic utterances from the indexed database.
We first compute the cosine similarity betweenE sand each
database entry:
sim(E s,ai) =Es·ai
∥Es∥∥ai∥.(2)
We then retrieve the top-Kmost relevant utterances. Each re-
trieved utteranceu k∈ U top-K is encoded by WavLM [23] to obtain
a prosody embedding:
Ewk=Pool(f WavLM (uk))∈Rdw,(3)
where Pool(·)aggregates the frame-level features into a fixed-length
vector. These exemplars provide style references that reflect the
characteristic intonation and emphasis patterns of sarcastic speech.
Compared to conventional style transfer methods that rely on a
single manually chosen reference utterance, the proposed retrieval
mechanism offers two key advantages. First, it scales more naturally
to diverse input contexts, since the retrieval process automatically
identifies semantically aligned exemplars. Second, by conditioning
on multiple retrieved samples, the model can capture a richer variety
of sarcastic prosodic patterns, leading to more robust and contextu-
ally appropriate expressive synthesis.
2.3. LLM-Enhanced and Retrieval-Guided Sarcastic TTS
The system builds upon the VITS architecture [24], enriched with
semantic features extracted by a fine-tuned LLaMA 3 and guided by
prosodic exemplars retrieved via a RAG module.
LetE p∈RTp×dpdenote the phoneme embeddings andE s∈
RTt×dtthe sarcasm-aware semantic embeddings produced by the
fine-tuned LLaMA 3. In the cross-attention module, we treatE pas
the query andE sas key and value:
Q=E pWq,K=E sWk,V=E sWv,(4)
H=SoftmaxQK⊤
√dk
V,(5)

whereW q,Wk,Wvare learned projection matrices, andd kis
the dimensionality of the key.
Finally, the prosody exemplarsE w={E w1, . . . ,E wK}ex-
tracted by WavLM are linearly projected and added to the cross-
attention output to modulate the decoder:
Z=H+KX
k=1WwEwk,(6)
whereW wis a learned linear projection mapping the prosody
embeddings to the decoder dimension. This yields hidden statesZ
that are conditioned on phonemes, semantic content, and retrieved
prosodic cues, enabling the synthesis of sarcastic speech with ex-
pressive intonation.
3. EXPERIMENTS
To evaluate the effectiveness of our proposed approach to sarcas-
tic speech synthesis, we conducted a series of experiments involv-
ing baseline, LLM-enhanced, RAG-enhanced, and combined mod-
els. These experiments assess the impact of semantic guidance from
language models and reference-based prosodic conditioning on the
quality and sarcasm-awareness of synthesized speech1.
3.1. Data
For the initial pre-training phase of VITS, we use the HiFi-TTS
corpus [25], a high-fidelity audiobook-derived dataset with diverse
speakers and wideband, clean recordings.
To build the sarcastic speech database for retrieval purposes, we
use the MUStARD++ dataset [11]. MUStARD++ is a multimodal
sarcasm detection corpus compiled from popular sitcoms such as
Friends and The Big Bang Theory. It contains 1,202 audiovisual
utterances, equally divided into 601 sarcastic and 601 non-sarcastic
samples. Following standard practice, we partition the corpus into
training, validation, and test sets with an 8:1:1 ratio. Specifically,
10% of the data is held out for evaluation, while the remaining 80%
of sarcastic samples are used to construct the retrieval-based sarcas-
tic speech database. This setup ensures relatively sufficient coverage
of sarcastic prosody in the retrieval dataset.
To adapt LLMs as sarcasm-aware semantic encoders for speech
synthesis, we fine-tune them on the News Headlines Sarcasm dataset
[26]. This dataset contains over 28,000 headlines from sources such
asThe Onion, each annotated with a binary sarcasm label. The
class distribution is roughly balanced, making it suitable for learning
discourse-level markers of sarcastic intent. The LLM serves as a fea-
ture extractor that provides high-level semantic embeddings condi-
tioned on sarcastic intent. To validate the effectiveness of this adap-
tation, we also evaluate the fine-tuned model on a sarcasm detection
task on the whole MUStARD++ dataset [11], in order to test that the
learned embeddings capture sarcasm-relevant cues.
3.2. Experimental Setup
For training the baseline VITS model, we follow the standard config-
uration described in [24]. The model is pre-trained on the HiFi-TTS
corpus [25] using the open-source Amphion toolkit2.
To adapt LLMs for sarcasm-aware synthesis, the LoRA fine-
tuning of LLaMA 3 is performed with an expansion factor of 8 and
1Speech samples are available athttps://abel1802.github.io/
RAG-LLM-SarcasticTTS/
2https://github.com/open-mmlab/Amphiona learning rate of1e−4, updating only the low-rank adapters in the
attention layers while freezing the backbone weights3.
3.3. Compared Methods
We compare the proposed framework against a series of baseline
and enhanced variants to systematically evaluate the contribution of
different modules:
•VITS (Baseline): A standard variational text-to-speech
model serving as the baseline, without explicit semantic
or stylistic guidance.
•VITS + BERT (w/ BERT): Extends VITS with semantic em-
beddings from a pretrained BERT model [27], providing ad-
ditional semantic features during synthesis.
•VITS + LLaMA 3 (w/ LLaMA 3): Uses textual embeddings
generated by LLaMA 3 [28], enabling richer contextual and
pragmatic guidance.
•VITS + LLaMA 3-LoRA (w/ LoRA): Incorporates a LoRA-
fine-tuned version of LLaMA 3 trained on sarcasm-labeled
data, enhancing the ability to capture sarcasm-relevant se-
mantics.
•VITS + RAG (w/ RAG): Employs RAG to retrieve sarcas-
tic reference speech samples that are semantically similar to
the input text. These references are used to provide prosodic
guidance during synthesis.
•VITS + LLaMA 3-LoRA + RAG (Proposed): Our proposed
unified framework, combining sarcasm-aware semantic em-
beddings from fine-tuned LLaMA 3 with prosodic exemplars
retrieved through RAG. This model is designed to maximize
both semantic relevance and expressive alignment with sar-
castic intent.
4. RESULTS AND DISCUSSION
4.1. Sarcasm Detection Results
We evaluate sarcasm detection performance from text-only inputs on
the MUStARD++ dataset, comparing the original sarcasm detection
systems used in MUStARD++ [11], BERT, LLaMA 3, and LLaMA
3-LoRA. Models are evaluated with Precision (P), Recall (R), and
weighted F1-score (F1), with results summarized in Table 1.
Table 1. Performance of different models on sarcasm detection
(text-only input).
Method P (%) R (%) F1 (%)
MUStARD++ (text) 67.9 67.7 67.7
BERT 66.8 66.9 66.8
LLaMA 3 65.7 65.4 65.5
LLaMA 3-LoRA72.4 72.7 72.5
As shown in Table 1, both the MUStARD++ baseline and
BERT achieved moderate performance with F1-scores around
67%. LLaMA 3 performed slightly worse (65.5%), suggesting
that general-purpose pretraining alone is insufficient to capture the
nuanced cues of sarcasm. In contrast, parameter-efficient fine-
tuning with LoRA yielded a clear improvement, raising the F1-score
3https://github.com/hiyouga/LLaMA-Factory

Table 2. Evaluation results of TTS systems. MCD is in dB; pitch and energy are averaged with standard deviation; Sarcasm detection includes
precision (P), recall (R), and weighted F1-score (F1); MOS and SMOS denote naturalness and sarcasm MOS scores, respectively.
Objective Sarcasm Detection Subjective
Method MCD↓Pitch↓Energy↓P (%)↑R (%)↑F1 (%)↑MOS↑SMOS↑
GT – – – 62.8 62.3 62.33.8±0.1 4.5±0.2
Baseline9.8±3.0 261.9±148.8 4.4±1.460.3 60.5 59.92.6±0.1 3.2±0.2
w/ BERT9.6±2.8265.6±147.1 4.5±1.460.5 60.6 60.52.5±0.1 3.1±0.2
w/ LLaMA 310.4±3.1 261.6±135.1 4.4±1.059.6 59.7 59.02.0±0.1 2.6±0.2
w/ LoRA10.1±3.1 282.6±149.6 4.4±1.460.6 60.8 60.62.6±0.1 3.8±0.2
w/ RAG10.0±3.0 261.0±148.0 4.4±1.461.5 61.7 61.62.6±0.1 3.7±0.2
Proposed9.8±2.8259.6±152.3 4.4±1.462.7 62.9 62.52.7±0.1 3.8±0.2
to 72.5%. This demonstrates the effectiveness of the proposed
sarcasm-aware semantic encoding via LoRA (Section 2.1), and mo-
tivates its integration into the sarcasm-aware TTS framework to
further examine whether such improvements in semantic encoding
can also lead to more expressive synthesized speech.
4.2. Sarcastic Speech Synthesis Results
Table 2 summarizes the results of different TTS systems across
three perspectives: low-level acoustic metrics (MCD, pitch, en-
ergy), downstream sarcasm detection performance, and subjective
naturalness ratings (Natural MOS and Sarcasm MOS). To assess
downstream sarcasm detection performance, we adopt the detection
architecture used in MUStARD++ with collaborative gating [11],
using only thespeech modality. Features extracted from synthesized
audio are compared against target sarcasm labels, where higher
agreement indicates stronger sarcasm expressiveness.
4.2.1. Objective Evaluation
Overall, the proposed VITS + LLaMA 3-LoRA + RAG system
achieves the strongest performance across both dimensions. It ob-
tains the lowest distortion (9.83MCD) and the most stable prosodic
statistics, while also yielding the highest sarcasm detection weighted
F1-score (62.5%). In comparison, VITS + LLaMA 3 suffers from
degraded quality (higher MCD), suggesting that raw embeddings
from an untuned LLM do not align well with the synthesis task.
LoRA fine-tuning mitigates this issue, and retrieval-augmented
guidance further enhances both acoustic stability and sarcasm clas-
sification accuracy.
Across all systems, acoustic metrics such as MCD, pitch, and en-
ergy remain relatively stable, indicating that the improvements stem
less from low-level signal fidelity and more from high-level semantic
and prosodic conditioning.
4.2.2. Subjective Evaluation
In addition to objective metrics, we conducted a subjective evalu-
ation to assess the perceptual quality and sarcastic expressiveness
of the synthesized speech (Table 2). We recruited 30 listeners from
diverse backgrounds who were asked to rate randomly shuffled sam-
ples from each system on two dimensions: (1) naturalness, using the
standard Mean Opinion Score (MOS) on a 5-point Likert scale, and
(2) sarcasm expressiveness (SMOS), reflecting how well the output
conveyed sarcastic intent.
Pairwise comparison across systems reveals several important
trends. First, the baseline VITS system received moderate MOS(2.6±0.1) and SMOS (3.2±0.2), indicating intelligible but rel-
atively flat speech with limited sarcastic cues. When replacing the
text encoder with BERT, scores remained similar (2.5±0.1MOS,
3.1±0.2SMOS), suggesting that in our specific setting, BERT em-
beddings do not provide meaningful advantages for sarcasm condi-
tioning.
In addition, directly integrating LLaMA 3 degraded perfor-
mance (2.0±0.1MOS,2.6±0.2SMOS). Listeners frequently
reported that these samples sounded less natural and had flat intona-
tion patterns. This indicates that raw LLM embeddings are poorly
aligned with intended expressive speech.
LoRA fine-tuning substantially narrowed this gap, raising natu-
ralness to2.6±0.1and sarcasm expressiveness to3.8±0.2. Com-
pared to the baseline, listeners consistently noted clearer prosodic
cues (e.g., exaggerated intonation) that aligned with sarcastic intent.
This suggests that LoRA adaptation enabled LLaMA 3 embeddings
to contribute pragmatically relevant sarcastic cues. Incorporating
RAG provided a noticeable improvement (2.6±0.1MOS,3.7±0.2
SMOS). Pairwise comparisons highlighted that RAG samples exhib-
ited prosodic patterns more consistent with real sarcastic exemplars,
making sarcastic intent more immediately recognizable.
Finally, the proposed system achieved the strongest results with
2.7±0.1MOS and3.8±0.2SMOS. Taken together, the results
suggest three main insights: (i) raw LLM embeddings are insuffi-
cient and may harm synthesis quality; (ii) parameter-efficient adap-
tation with LoRA can effectively inject pragmatic knowledge into
the speech synthesis pipeline; and (iii) retrieval-augmented condi-
tioning further refines prosodic expressiveness by grounding synthe-
sis in real sarcastic exemplars. Together, these mechanisms enable
the proposed system to generate speech that is both natural and per-
ceptually recognizable as sarcastic.
5. CONCLUSION
We presented a unified framework for sarcasm-aware speech synthe-
sis that leverages semantic conditioning from finetuned LLMs and
prosodic guidance from retrieval-based exemplars. By fine-tuning
LLaMA 3, our system captures semantic-level markers of sarcastic
intent, while RAG provides prosodic references that enhance expres-
sive alignment. Our results demonstrate the feasibility of model-
ing subtle pragmatic phenomena with TTS by combining parameter-
efficient LLM adaptation and retrieval augmentation. In future work,
we aim to extend this approach to other pragmatic styles such as hu-
mor, and explore cross-lingual sarcasm synthesis in diverse cultural
contexts. We anticipate that this method could also serve as an ef-
fective data augmentation strategy for sarcasm detection tasks.

6. REFERENCES
[1] Henry S Cheang and Marc D Pell, “The sound of sarcasm,”
Speech communication, vol. 50, no. 5, pp. 366–381, 2008.
[2] Zhu Li, Xiyuan Gao, Yuqing Zhang, Shekhar Nayak, and Matt
Coler, “A functional trade-off between prosodic and semantic
cues in conveying sarcasm,” inProc. Interspeech 2024, 2024,
pp. 1070–1074.
[3] Yuxuan Wang, Daisy Stanton, Yu Zhang, RJ-Skerry Ryan, Eric
Battenberg, Joel Shor, Ying Xiao, Ye Jia, Fei Ren, and Rif A
Saurous, “Style tokens: Unsupervised style modeling, control
and transfer in end-to-end speech synthesis,” inInternational
conference on machine learning. PMLR, 2018, pp. 5180–5189.
[4] Kei Akuzawa, Yusuke Iwasawa, and Yutaka Matsuo, “Ex-
pressive speech synthesis via modeling expressions with vari-
ational autoencoder,” inProc. Interspeech 2018, 2018, pp.
3067–3071.
[5] Tao Li, Shan Yang, Liumeng Xue, and Lei Xie, “Controllable
emotion transfer for end-to-end speech synthesis,” in2021
12th International Symposium on Chinese Spoken Language
Processing (ISCSLP). IEEE, 2021, pp. 1–5.
[6] Sara Peters and Amit Almor, “Creating the sound of sarcasm,”
Journal of Language and Social Psychology, vol. 36, no. 2, pp.
241–250, 2017.
[7] Zhu Li, Yuqing Zhang, Xiyuan Gao, Devraj Raghuvanshi, Na-
gendra Kumar, Shekhar Nayak, and Matt Coler, “Integrat-
ing feedback loss from bi-modal sarcasm detector for sarcastic
speech synthesis,” inProc. SSW 2025, 2025, pp. 150–156.
[8] Zhu Li, Xiyuan Gao, Shekhar Nayak, and Matt Coler, “Sarcas-
ticspeech: Speech synthesis for sarcasm in low-resource sce-
narios,” inProc. SSW 2023, 2023, pp. 242–243.
[9] Hannes Ritschel, Ilhan Aslan, David Sedlbauer, and Elisabeth
Andr ´e, “Irony man: Augmenting a social robot with the abil-
ity to use irony in multimodal communication with humans,”
inProceedings of the 18th International Conference on Au-
tonomous Agents and MultiAgent Systems, 2019, pp. 86–94.
[10] Santiago Castro, Devamanyu Hazarika, Ver ´onica P ´erez-Rosas,
Roger Zimmermann, Rada Mihalcea, and Soujanya Poria, “To-
wards multimodal sarcasm detection (an Obviously perfect
paper),” inACL, Anna Korhonen, David Traum, and Llu ´ıs
M`arquez, Eds., Florence, Italy, July 2019, pp. 4619–4629, As-
sociation for Computational Linguistics.
[11] Anupama Ray, Shubham Mishra, Apoorva Nunna, and Push-
pak Bhattacharyya, “A multimodal corpus for emotion recog-
nition in sarcasm,” inLREC, Marseille, France, June 2022, pp.
6992–7003.
[12] Devraj Raghuvanshi, Xiyuan Gao, Zhu Li, Shubhi Bansal,
Matt Coler, Nagendra Kumar, and Shekhar Nayak, “Intra-
modal relation and emotional incongruity learning using graph
attention networks for multimodal sarcasm detection,” in
ICASSP. IEEE, 2025, pp. 1–5.
[13] Xiyuan Gao, Shubhi Bansal, Kushaan Gowda, Zhu Li, Shekhar
Nayak, Nagendra Kumar, and Matt Coler, “Amused: An at-
tentive deep neural network for multimodal sarcasm detection
incorporating bi-modal data augmentation,”arXiv preprint
arXiv:2412.10103, 2024.[14] Zhu Li, Xiyuan Gao, Yuqing Zhang, Shekhar Nayak, and Matt
Coler, “Evaluating multimodal large language models on spo-
ken sarcasm understanding,”arXiv preprint arXiv:2509.15476,
2025.
[15] Xincan Feng and Akifumi Yoshimoto, “Llama-vits: Enhanc-
ing tts synthesis with semantic awareness,” inLREC-COLING
2024, 2024, pp. 10642–10656.
[16] Shuhua Li, Qirong Mao, and Jiatong Shi, “PL-TTS: A Gen-
eralizable Prompt-based Diffusion TTS Augmented by Large
Language Model,” inInterspeech 2024. Sept. 2024, pp. 4888–
4892, ISCA.
[17] Maohao Shen, Shun Zhang, Jilong Wu, Zhiping Xiu, Ehab Al-
Badawy, Yiting Lu, Mike Seltzer, and Qing He, “Get large
language models ready to speak: A late-fusion approach for
speech generation,” inICASSP. IEEE, 2025, pp. 1–5.
[18] Jinlong Xue, Yayue Deng, Yingming Gao, and Ya Li,
“Retrieval Augmented Generation in Prompt-based Text-to-
Speech Synthesis with Context-Aware Contrastive Language-
Audio Pretraining,” inInterspeech 2024. Sept. 2024, pp. 1800–
1804, ISCA.
[19] Dan Luo, Chengyuan Ma, Weiqin Li, Jun Wang, Wei Chen, and
Zhiyong Wu, “AutoStyle-TTS: Retrieval-Augmented Genera-
tion based Automatic Style Matching Text-to-Speech Synthe-
sis,” Apr. 2025, arXiv:2504.10309 [cs].
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”
NeurIPS, vol. 33, pp. 9459–9474, 2020.
[21] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and
Jason Weston, “Retrieval augmentation reduces hallucination
in conversation,” inFindings of EMNLP 2021, 2021, pp. 3784–
3803.
[22] Yunrui Cai, Zhiyong Wu, Jia Jia, and Helen Meng, “Lora-mer:
Low-rank adaptation of pre-trained speech models for multi-
modal emotion recognition using mutual information,” inProc.
Interspeech 2024, 2024, pp. 4658–4662.
[23] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shu-
jie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yosh-
ioka, Xiong Xiao, et al., “Wavlm: Large-scale self-supervised
pre-training for full stack speech processing,”IEEE Journal of
Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–
1518, 2022.
[24] Jaehyeon Kim, Jungil Kong, and Juhee Son, “Conditional
variational autoencoder with adversarial learning for end-to-
end text-to-speech,” inInternational Conference on Machine
Learning. PMLR, 2021, pp. 5530–5540.
[25] Evelina Bakhturina, Vitaly Lavrukhin, Boris Ginsburg, and
Yang Zhang, “Hi-fi multi-speaker english tts dataset,” inProc.
Interspeech 2021, 2021, pp. 2776–2780.
[26] Rishabh Misra and Prahal Arora, “Sarcasm detection using
news headlines dataset,”AI Open, vol. 4, pp. 13–18, 2023.
[27] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina
Toutanova, “Bert: Pre-training of deep bidirectional transform-
ers for language understanding,” inNAACL, 2019.
[28] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al., “The
llama 3 herd of models,”arXiv e-prints, pp. arXiv–2407, 2024.