# NIM4-ASR: Towards Efficient, Robust, and Customizable Real-Time LLM-Based ASR

**Authors**: Yuan Xie, Jiaqi Song, Guang Qiu, Xianliang Wang, Kai Qiao, Junfeng Yuan, Shengqing Liu, Yi Zhang, Bowen Chen, Ming Lei, Jie Gao, Jie Wu

**Published**: 2026-04-20 11:21:06

**PDF URL**: [https://arxiv.org/pdf/2604.18105v1](https://arxiv.org/pdf/2604.18105v1)

## Abstract
Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a mainstream paradigm in recent years. Although existing LLM-based ASR models demonstrate impressive performance on public benchmarks, their training remains predominantly data-driven, leaving key practical challenges insufficiently addressed -- particularly limited downward scalability in resource-constrained deployments and hallucinations under acoustically challenging conditions. To address these issues, we present NIM4-ASR, a production-oriented LLM-based ASR framework optimized for both efficiency and robustness. Grounded in a principled delineation of functional roles between the encoder and the LLM, we redesign the multi-stage training paradigm to align each module with its intended capability boundary. Specifically, we reformulate the pre-training architecture and objective to mitigate the modality gap and improve parameter efficiency; introduce an iterative asynchronous SFT stage to preserve acoustic fidelity and constrain representation drift; and design an ASR-specialized reinforcement learning stage to further enhance recognition quality and robustness. We additionally incorporate a suite of production-oriented optimizations, including robustness under noisy and silent conditions, real-time streaming inference, and hotword customization via retrieval-augmented generation (RAG). Experiments show that NIM4-ASR achieves state-of-the-art performance on multiple public benchmarks with merely 2.3B parameters, while substantially outperforming larger-scale competitors on internal benchmarks -- particularly in entity-intensive real-world scenarios. NIM4-ASR further supports million-scale hotword customization via RAG with sub-millisecond retrieval latency, enabling efficient adaptation to emerging entities and personalized user requirements.

## Full Text


<!-- PDF content starts -->

NIM4-ASR: Towards Efficient, Robust, and
Customizable Real-Time LLM-Based ASR
Yuan Xie∗, Jiaqi Song∗, Guang Qiu, Xianliang Wang, Kai Qiao, Junfeng Yuan
Shengqing Liu, Yi Zhang, Bowen Chen, Ming Lei, Jie Gao, Jie Wu
Advanced Intelligent Systems Group, NIO
{ryan.xie2, jiaqi.song2}@nio.com
Abstract
Integrating large language models (LLMs) into automatic speech recognition
(ASR) has become a mainstream paradigm in recent years. Although existing
LLM-based ASR models demonstrate impressive performance on public bench-
marks, their training remains predominantly data-driven, leaving key practical
challenges insufficiently addressed—particularly limited downward scalability in
resource-constrained deployments and hallucinations under acoustically challeng-
ing conditions. To address these issues, we presentNIM4-ASR, a production-
oriented LLM-based ASR framework optimized for both efficiency and robustness.
Grounded in a principled delineation of functional roles between the encoder and
the LLM, we redesign the multi-stage training paradigm to align each module
with its intended capability boundary. Specifically, we reformulate the pre-training
architecture and objective to mitigate the modality gap and improve parameter
efficiency; introduce an iterative asynchronous SFT stage to preserve acoustic
fidelity and constrain representation drift; and design an ASR-specialized rein-
forcement learning stage to further enhance recognition quality and robustness. We
additionally incorporate a suite of production-oriented optimizations, including
robustness under noisy and silent conditions, real-time streaming inference, and hot-
word customization via retrieval-augmented generation (RAG). Experiments show
that NIM4-ASR achieves state-of-the-art performance on multiple public bench-
marks with merely 2.3B parameters, while substantially outperforming larger-scale
competitors on internal benchmarks—particularly in entity-intensive real-world sce-
narios. NIM4-ASR further supports million-scale hotword customization via RAG
with sub-millisecond retrieval latency, enabling efficient adaptation to emerging
entities and personalized user requirements.
1 Introduction
With the rapid advancement of large language models (LLMs), the prevailing paradigm of automatic
speech recognition (ASR) is undergoing a transition from classical architectures (Graves et al., 2006;
Chorowski et al., 2015; Chan et al., 2016; Graves, 2012) to the encoder–adaptor–LLM framework (Bai
et al., 2024a; An et al., 2025). Over the past two years, a series of LLM-based ASR models, including
Seed-ASR (Bai et al., 2024a), Fun-ASR (An et al., 2025), FireRedASR series (Xu et al., 2025b,
2026), V oxtral (Liu et al., 2025), Index-ASR (Song et al., 2025), and Qwen3-ASR (Shi et al., 2026),
have achieved promising performance on public ASR benchmarks.
Compared with classical ASR models that are primarily optimized for acoustic-to-lexical transduction,
LLM-based ASR benefits from the rich linguistic priors and contextual modeling capacity inherited
from large-scale language model pretraining (Fathullah et al., 2024). The LLM’s strong language
*Equal Contribution.arXiv:2604.18105v1  [eess.AS]  20 Apr 2026

modeling capacity and contextual coherence modeling help resolve acoustic and lexical ambiguities,
yielding transcriptions that are more fluent and semantically coherent. Furthermore, LLMs encode
extensive world knowledge during large-scale pretraining, substantially improving the recognition
of rare named entities, technical terminology, and domain-specific expressions that classical ASR
models frequently misrecognize (Wang et al., 2025). Overall, incorporating LLMs helps bridge
acoustic modeling with semantic understanding (An et al., 2025; Hono et al., 2024), leading to
enhanced robustness to acoustically ambiguous inputs such as noise and accent variations, as well as
improved cross-domain generalization. Despite these advantages, LLM-based ASR still faces several
key limitations in real-world scenarios.
1. Limited downward scalability.In deployment, particularly for real-time speech interfaces,
lightweight ASR models are favored for their lower inference latency and computational cost.
However, the downward scalability of LLM-based ASR appears disappointing: lightweight variants
such as Qwen3-ASR-0.6B and Fun-ASR-nano exhibit substantial performance gaps relative to their
full-scale counterparts. Beyond the degradation ordinarily expected from model downscaling, LLM-
based ASR models carry an additional structural cost from the modality tax (Aghajanyan et al.,
2023; Zhang et al., 2026): a non-trivial number of parameters are devoted to cross-modal alignment
rather than the ASR task itself. This overhead leaves lightweight LLMs with less effective capacity,
imposing a disproportionate performance degradation (Endo and Yeung-Levy, 2025).
2. Hallucination.Beyond the intrinsic hallucination tendencies of autoregressive LLMs, the encoder–
adaptor–LLM joint-training paradigm introduces additional risks (Bai et al., 2024b; Zhou et al., 2024;
Xie et al., 2026). During joint optimization, the encoder is progressively pulled toward the LLM’s
optimization objective under the influence of its stronger gradients and linguistic priors, causing
its representations to gradually shift toward the LLM’s text feature space (i.e., representation drift).
As a result, the encoder may increasingly rely on linguistic shortcuts at the expense of fine-grained
acoustic fidelity, exacerbating hallucinations under acoustically ambiguous conditions (Park et al.,
2025). In task-oriented in-vehicle speech interaction scenarios, hallucinations can cascade through
the downstream pipeline and trigger unintended actions (Tay et al., 2026).
3. Lack of production-ready hotword customization.Existing LLM-based ASR systems lack
mature hotword customization solutions comparable to N-gram language model rescoring meth-
ods (Song et al., 2019; Kuo and Chen, 2022) widely adopted in classical ASR systems. Such
customization support is indispensable for accurately transcribing personalized entities with similar
pronunciations (An et al., 2025; Lei et al., 2025), including homophonous location names, media
titles, and emerging proper nouns that often reside in the long tail of LLMs’ pretraining distribution.
To address the aforementioned limitations, we proposeNIM4-ASR (NOMI Intelligence Model
4.0-ASR), a production-oriented LLM-based ASR framework optimized for both efficiency and
robustness. NIM4-ASR adopts a redesigned multi-stage training paradigm that reduces the modality
gap between speech and text while explicitly delineating the functional roles of the encoder and the
LLM (Xie et al., 2026). Specifically, we redesign a module-aware pre-training scheme that aligns
training objectives with the intrinsic characteristics of each component. This encourages the encoder
to produce low-entropy, peaky representations that narrow the modality gap, reducing the LLM
capacity required for cross-modal alignment and improving parameter efficiency. We then develop an
Iterative Asynchronous SFT (IA-SFT) stage between alignment and joint SFT, which strengthens
cross-modal alignment while preserving functional decoupling across modules, thereby mitigating
representation drift and suppressing hallucinations. Additionally, we incorporate an ASR-specialized
reinforcement learning (RL) strategy to further enhance recognition quality and robustness. Beyond
the training-side design, NIM4-ASR also incorporates a series of production-oriented enhancements
for practical deployment, including robustness under noisy and silent conditions, real-time streaming
inference, and scalable hotword customization via retrieval-augmented generation (RAG). Finally,
we conduct extensive evaluations on diverse Mandarin and English benchmarks, demonstrating that
NIM4-ASR achieves state-of-the-art (SOTA) performance on several benchmarks with only 2.3B
parameters. Our key contributions are summarized as follows:
•Principled multi-stage training paradigm.We propose a principled multi-stage training paradigm
that reduces the modality gap and preserves module-specific functional specialization for improved
efficiency and robustness. We further introduce an ASR-specialized RL stage, which brings
additional gains in recognition accuracy and hallucination mitigation.
2

•Optimized streaming support.We cultivate the encoder’s native streaming capability from pre-
training and introduce a decoupled streaming inference strategy that separates encoder and LLM
execution. The inference strategy is further complemented by an incremental context extension
mechanism for efficient KV-cache reuse.
•Phoneme-level RAG for hotword customization.Building on Fun-ASR, we improve the
phoneme-level retrieval algorithm with an emphasis on retrieval precision and latency, enabling
million-scale hotword customization with sub-millisecond retrieval latency while preserving high
retrieval precision.
•Comprehensive evaluation.We conduct comprehensive evaluations across 25 benchmarks (15
public and 10 internal), showing that NIM4-ASR can achieve SOTA performance on multiple
benchmarks with only 2.3B parameters, validating its parameter efficiency and strong robustness.
2 Methodology
2.1 Architecture
StreamingSpeechEncoder
Speechadaptor
Phonemehead
LargeLanguageModelTranscribe the speech into text.Contextual hints:
Encoderrepresentations
HotworddatabaseSpeechembeddingsNIOHouseRetrieval“Give me your top picks for delicious lattes at NIO House”Transcript:Give me your top picks fordelicious lattesat NIO HouseGive…NIOHouse<eos><bos>Give…NIOHousePhonemehypos
Figure 1: The overall architecture of NIM4-ASR.
As shown in Figure 1, NIM4-ASR follows a modular encoder–adaptor–LLM architecture comprising
four components. Before being fed into the model, raw speech is first converted into 80-dimensional
log-Mel spectrograms using a 25 ms window with a 10 ms frame shift, followed by global mean and
variance normalization. The details of the four main components are described below:
•Streaming speech encoder.Our encoder adopts the Conformer encoder architecture from
FireRedASR-AED (Xu et al., 2025b), consisting of a 4x downsampling convolutional module fol-
lowed by a stack of Conformer blocks (Gulati et al., 2020), with approximately 600 M parameters
in total. The encoder converts acoustic features into continuous representations at a frame rate of
25 Hz (40 ms temporal resolution). To support low-latency online decoding, we convert it into a
chunk-based streaming encoder by simulating streaming constraints during training.
•Speech adaptor.The speech adaptor consists of a two-layer MLP that maps the encoder represen-
tations into the LLM’s input embedding space. Before projection, we apply a 4x downsampling by
concatenating four consecutive frames along the feature dimension to shorten the sequence length.
After downsampling, the frame rate is reduced to 6.25 Hz, corresponding to a temporal resolution
of 160 ms per token.
•Phoneme-level CTC head and RAG module.The phoneme-level CTC head (hereafter referred
to as the CTC head or phoneme head) serves as the acoustic front-end of the RAG module,
comprising a three-layer MLP. It decodes encoder representations into phoneme hypotheses via
greedy decoding. Based on these hypotheses, our retrieval algorithm searches the hotword database
to retrieve matching entries, which are then injected into the prompt as contextual hints for the
LLM. Further details of the RAG module are provided in Section 2.4.2.
•LLM decoder.The decoder is initialized from Qwen3-1.7B (Yang et al., 2025) and generates the
final transcription conditioned on both speech embeddings and optional retrieved hotword hints.
3

2.2 Training Recipe
In contrast to most prior work driven primarily by empirical fine-tuning, we begin with a princi-
pled analysis of the practical limitations of current LLM-based ASR systems and their underlying
causes (Xie et al., 2026), revealing that the cross-modal gap and representation drift remain insuf-
ficiently addressed. Based on these insights, we comprehensively redesign the training pipeline.
As illustrated in Figure 2, the methodological advances of NIM4-ASR center on four core training
stages: encoder pretraining, alignment, IA-SFT, and late joint SFT. Beyond this four-stage pipeline,
context SFT and RL are further incorporated after late joint SFT to strengthen contextual modeling
and robustness. The detailed procedures are described below.
Conventional LLM-based ASR
Stage 1: Pre-training
🔥 Encoder
🔥 Decoder
Stage 2: Alignment
❄/ 
🔥  EncoderFinal
🔥 Adaptor
❄ LLM
Stage 3: SFT
🔥 EncoderFinal
🔥 Adaptor
🔥 LLM
Stage 2: Alignment
❄ Encoderinit
🔥 Adaptor
❄ LLM
❄ Encodermid
🔥 Adaptor
🔥 LLMStage 3: IA-SFT w/ iterative updates
❄ EncoderFinal
🔥 AdaptorStage 1: Pre-training
🔥 CTC head
🔥 EncoderNIM4-ASR
🔥 CTC head
🔥 CTC head
🔥 CTC headInitializeUpdate
🔥 LLM
🔥 EncoderFinal
🔥 Adaptor
🔥 LLMStage 4: Late joint SFT
🔥 Encodermid
🔥 Encoderinit
🔥 EncoderFinal
❄ Encoder
🔥 Phoneme headFine-tune for RAG moduleInitialize
InitializeInitializeCKA-Triggered---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
❄ Frozen 
🔥 Trainable
❄ Encoderinit
🔥 Adaptor
🔥 LLMUpdateCKA-TriggeredFinal update
Figure 2: Comparison of training pipelines from encoder pretraining to joint SFT for conventional
LLM-based ASR and our NIM4-ASR.
2.2.1 Stage 1: Encoder Pre-training
To reduce the modality gap between encoder representations and the LLM embedding space, we adopt
an improved variant of Connectionist Temporal Classification (CTC) (Graves et al., 2006)—namely
CR-CTC (Yao et al., 2024)—as the pretraining objective. As illustrated in Figure 2, the model
architecture during pretraining consists of the encoder paired with a CTC head. In contrast to the
Attention-based Encoder-Decoder (AED) commonly used in prior work (Xu et al., 2025b, 2026),
CTC encourages the encoder to produce low-entropy, phoneme-discriminative representations that
align more naturally with the LLM’s embedding space, thereby reducing cross-modal alignment
overhead and reserving more model capacity for the ASR task.
Furthermore, we shift the supervision labels from character level to phoneme level (Yusuyin et al.,
2025), explicitly dedicating the encoder’s capacity to acoustic-to-phoneme mapping rather than
premature semantic anchoring, while encouraging the LLM to focus more on semantic reasoning.
This design achieves a cleaner decoupling of acoustic modeling from semantic reasoning, improving
role specialization of both modules. Moreover, adopting phoneme prediction as the pretraining
objective encourages the encoder to learn low-level acoustic representations with weak language
dependence, offering greater potential for extending to new languages and dialects.
To endow the encoder with native streaming capability, we incorporate the dynamic-chunk mechanism
during pretraining (Zhang et al., 2020). Specifically, the encoder processes full utterances under
chunk-wise streaming constraints, where the chunk size and the number of visible left-context chunks
are dynamically sampled for each batch. This exposes the encoder to a wide range of streaming
configurations, enabling flexible operation that accommodates varying latency budgets across different
deployment scenarios.
4

2.2.2 Stage 2: Alignment & Stage 3: IA-SFT
In conventional training paradigms, alignment and joint SFT are performed sequentially after pre-
training fully completes. As shown in Figure 2, we propose an encoder iteration mechanism for
NIM4-ASR that allows alignment to begin before pretraining completes, while IA-SFT is launched
upon alignment completion and proceeds asynchronously alongside the remaining pretraining process.
To decide when to initialize or update the encoder used by alignment and IA-SFT, we track encoder
representation dynamics using Centered Kernel Alignment (CKA) (Kornblith et al., 2019), which
compares the evolving encoder against the reference checkpoint that is initialized and periodically
updated throughout pretraining. Given two sets of encoder representations E(a), E(b)extracted from
the same evaluation set, CKA is defined as
CKA(E(a), E(b)) =⟨˜K(a),˜K(b)⟩Fq
⟨˜K(a),˜K(a)⟩F· ⟨˜K(b),˜K(b)⟩F,(1)
where ˜K(a)and ˜K(b)are centered Gram matrices calculated via ˜K(x)=CE(x)E(x)⊤C. The
centering matrix is defined as C=I L−1
LJL, where ILis the identity matrix and JLis the all-ones
matrix. CKA measures the geometric consistency of representation spaces, invariant to orthogonal
transformation and isotropic scaling.
Stage 2: Alignment.We start monitoring the encoder after pretraining reaches 500k steps, at which
point the encoder begins to exhibit a relatively stable optimization trend. The encoder at 500k
steps is snapshotted as the initial reference checkpoint, and CKA is evaluated every 10k pretraining
steps thereafter. When the CKA score between the evolving encoder and the current reference
checkpoint first falls below the predefined threshold1, we snapshot the corresponding encoder to
initialize alignment and simultaneously update the reference checkpoint. During alignment, both the
encoder and LLM are frozen, and only the adaptor is trained. In our setup, this first trigger occurs at
approximately 1.01M pretraining steps, and the alignment stage runs for 1.3M steps.
Stage 3: IA-SFT.After alignment completes, we perform IA-SFT as an intermediate stage before
joint SFT. IA-SFT keeps the encoder frozen and trains the adaptor–LLM stack across a sequence of
encoder snapshots produced by the asynchronous pretraining process. The procedure is as follows:
•(i) Initialization & monitoring.IA-SFT begins after alignment completes, training for 1M steps
with the encoder inherited from alignment, while encoder pretraining continues in parallel. The
CKA evaluation resumes from the previously updated reference checkpoint and continues every
10k pretraining steps, monitoring the representation shift.
•(ii) CKA-triggered update.Whenever the CKA score drops below the predefined threshold,
the snapshot of the current pretraining encoder is hot-swapped into the IA-SFT branch, and the
reference checkpoint is updated accordingly.
•(iii) Final update.The update cycle (ii) repeats until pretraining reaches its 2M-step maximum.
When pretraining completes, a final encoder update is applied regardless of the CKA score, and
IA-SFT runs for the final 2M steps.
In our implementation, IA-SFT trains for 1M steps using the encoder checkpoint at 1.01M pretraining
steps, another 1M steps using the encoder checkpoint at 1.32M pretraining steps, and a final 2M steps
using the fully pretrained encoder—totaling 4M steps across three encoder versions. During IA-SFT,
the encoder remains frozen but is periodically updated from the asynchronous pretraining process,
thus maintaining acoustic grounding. This allows the model to deepen cross-modal alignment without
the risk of representation drift. From a curriculum learning perspective, IA-SFT progressively exposes
the LLM to refined encoder representations, allowing it to learn invariant patterns and achieve greater
robustness to acoustic perturbations. Furthermore, since alignment and IA-SFT run asynchronously
alongside pretraining, the overall training pipeline remains time-efficient.
2.2.3 Stage 4: Late Joint SFT
After the completion of both encoder pretraining and IA-SFT, a robust initial cross-modal mapping
between speech representations and the LLM embedding space has been established. We then
1In this work, we empirically set this threshold to 0.975 based on global CKA dynamics during pretraining,
balancing meaningful representation changes against disruptive representation shifts.
5

perform late joint SFT, in which the encoder, adaptor, and LLM are jointly optimized in an end-to-end
manner. Compared with conventional joint training, the risk of representation drift induced by LLM
gradients is substantially reduced, as the preceding stages have already minimized the modality
gap. Consequently, these gradients serve primarily as fine-tuning signals that seamlessly refine
acoustic-to-phoneme mapping and phoneme-to-semantic grounding. From a geometric perspective,
the preceding alignment stages have established a stable cross-modal manifold, placing subsequent
optimization in a low-curvature region of the loss landscape. Within this regime, gradient updates act
as local refinements to decision boundaries and manifold geometry rather than inducing large-scale
topological restructuring.
Following late joint SFT, all subsequent training stages, including context SFT and RL, are conducted
in a fully end-to-end manner. With modality alignment concerns largely resolved in prior stages,
the model can devote its full capacity to refining complex cross-modal reasoning and long-context
interaction, progressively deepening the integration of acoustic perception and semantic modeling.
2.2.4 Stage 5: Context SFT
Following joint SFT, we introduce a context SFT stage (Bai et al., 2024a; An et al., 2025; Song et al.,
2025) to strengthen the model’s ability to leverage contextual information—a capability essential for
hotword customization in LLM-based ASR systems. In this stage, we first construct a keyword set
Sfrom the training corpus. All transcripts are parsed to extract candidate phrases, which are then
filtered by Qwen3-30B-A3B-Instruct (Yang et al., 2025) to retain named entities such as personal
names, POI (points of interest), media names and proper nouns. During training, we increase the
sampling ratio of long-duration utterances and probabilistically inject keywords sampled from Sinto
the prompt as contextual hints, following the template below:
NIM4-ASR Prompt Template
<|im_start|>system
You are a Multilingual Speech Recognition Model.<|im_end|>
<|im_start|>user
Transcribe the speech into text <speech>.
Contextual hints: {context}.<|im_end|>
<|im_start|>assistant
<transcription><|im_end|>
For each training instance, we first retrieve relevant keywords from Spresent in the transcript.
Additionally, for each keyword, we retrieve another keyword from Swith identical or highly similar
pronunciation to serve as a distractor context with a certain probability. Both relevant keywords and
distractors are concatenated and then added to the {context} field. The inclusion of distractors
discourages the LLM from over-relying on contextual cues at the expense of semantic plausibility.
During this stage, the encoder, adaptor and the LLM are jointly trained.
It is worth noting that our context SFT focuses on phrase-level contextual cues (An et al., 2025; Shi
et al., 2026) rather than the sentence- or dialogue-level context (Bai et al., 2024a), as this stage is
designed specifically for hotword customization rather than cross-turn dialogue consistency. For
multi-turn scenarios, keywords extracted from dialogue history can also be appended to the current
prompt. This strategy preserves critical contextual information in a compact form, while maintaining
lower inference latency than sentence-level alternatives.
2.2.5 Stage 6: ASR Specialized RL
To further improve transcription quality, we introduce an ASR-specialized RL stage based on Group
Relative Policy Optimization (GRPO) (Shao et al., 2024) that directly optimizes sequence-level
transcription behavior using verifiable rewards. In contrast to supervised objectives that rely on
token-level teacher-forcing, RL evaluates complete hypotheses and directly optimizes sequence-
level transcription behavior, improving recognition accuracy, hallucination robustness, and context-
sensitive keyword recognition.
6

Given an input audio qwith ground truth y, the policy model independently samples a group of K
candidate hypotheses {τ1, . . . , τ K} ∼π θ(· |q) , while each hypothesis is evaluated using a set of
ASR-specific reward functions.
•Accuracy reward:We apply a unified text normalization pipeline to both the generated hypotheses
and the ground truth, and then compute the character error rate (CER) of each hypothesis. The
reward function is defined as Racc(τ, y) = exp(−α·CER(τ,y)) , where αis set to 2.0in our
experiments. This reward is bounded within (0,1] , and its exponential form amplifies the differences
among low-CER hypotheses, encouraging fine-grained optimization on near-correct transcriptions.
For high-CER regions (e.g., CER>1.0 ), the function requires no clipping and still preserves
monotonic reward ordering, which is essential for computing within-group advantages in GRPO.
•Hallucination reward:We apply mixed-granularity tokenization (character-level for Chinese,
word-level for English) to both hypothesis and ground truth, then compute their lengths. The
hallucination reward Rhallu(τ, y) =−1 if the hypothesis length exceeds 2× or falls below 0.5× the
ground truth length; otherwiseR hallu(τ, y) = 0.
•Context reward:For each training sample, we use Qwen3-30B-A3B-Instruct to annotate 0–2
entity keywords per sample. During training, each sample randomly selects a subset of keywords
and injects them into the prompt as contextual hints. For each selected keyword, we check whether
it appears in the hypothesis: a hit yields a reward of +0.5 , while a miss incurs a penalty of −0.5 .
The cumulative score across all valid contexts is then averaged to obtain the sample-level context
reward Rcontext (τ, y) . Notably, we also define a list of important keywords. Whenever a sample
contains any important keyword, that keyword is included in the reward computation, regardless of
whether it is provided in the prompt.
Finally, the total reward is given by:
R(τ, y) =R acc(τ, y) + 0.5R hallu(τ, y) + 0.5R context (τ, y).(2)
Reinforcement learning algorithm.Following GRPO, we compute the group-normalized advan-
tage
ˆAi,t=R(τi, y)−mean({R(τ j, y)}K
j=1)
std({R(τ j, y)}K
j=1) +ϵ.(3)
where ϵis a small constant for numerical stability. Denote θoldas the policy parameters at the
beginning of each optimization step, εas the clipping range, and βas the KL penalty coefficient. The
GRPO objective is defined as
JGRPO =1
KKX
i=11
|τi||τi|X
t=1min
ri,t(θ)ˆAi,t,clip 
ri,t(θ),1−ε,1 +εˆAi,t
−β D KL(πθ∥πref),
(4)
where
ri,t(θ) =πθ(τi,t|q, τ i,<t)
πθold(τi,t|q, τ i,<t).(5)
RL training framework.We implement an RL training pipeline tailored for LLM-based ASR.
For each training batch, the policy model encodes input utterances into speech embeddings, which
are reused across both rollout generation and policy model log-probability computation to avoid
redundant computation. During policy rollout, we leverage vLLM (Kwon et al., 2023) to efficiently
sample Khypotheses conditioned on the speech embeddings and the instruction prompt. The sampled
hypotheses are then scored by the reward functions described above. Policy optimization is conducted
in a DeepSpeed ZeRO (Rajbhandari et al., 2020) distributed training setup, where token-level log-
probabilities are computed under both the policy model and the reference model. The reference model
remains frozen throughout training, providing a stable anchor for KL regularization and preventing
excessive policy drift. After each optimization step, the updated policy weights are synchronized to
the vLLM rollout engine, ensuring that hypothesis sampling remains on-policy.
Considering that ASR models typically employ deterministic decoding at inference time, we adopt a
cosine-annealed temperature schedule for rollout sampling, gradually decaying from 1.0 to 0.7. In the
early stages of RL training, the high temperature encourages diverse hypothesis generation, allowing
7

the reward signal to explore a broader range of transcription behaviors. As training progresses,
the temperature is smoothly reduced, progressively reinforcing top-1 path quality to ensure strong
performance under deterministic decoding at inference.
2.2.6 Additional Stage: Phoneme Head Training for RAG
After completing the RL stage, the main training pipeline is concluded. We then introduce an
additional stage to train the phoneme head required by the RAG module illustrated in Figure 1. In
this stage, the encoder inherits its structure and weights from the post-RL checkpoint and remains
frozen, while the phoneme head is initialized from the pretrained CTC head and remains trainable
(see Figure 2). The training objective and configuration are consistent with those used in pretraining.
After fine-tuning, the phoneme head can convert encoder representations into phoneme hypotheses
for the subsequent retrieval module.
2.3 Training Setup
This section presents additional implementation details, including training tricks and settings.
Robustness enhancement under noisy and silent conditions.In the first five training stages,
we apply several data augmentation tricks to improve model robustness. In addition to standard
SpecAugmentation (Park et al., 2019) and speed perturbation, we randomly inject realistic acoustic
disturbances, such as babble noise, vehicle noise, and background music, into 20% of clean training
samples to simulate challenging real-world environments. The Signal-to-Noise Ratio (SNR) for
these noise injections is randomly sampled from a normal distribution with mean 10 dB and standard
deviation 5 dB.
Furthermore, to improve the model’s robustness to silence, we adopt a padding-before-noise strat-
egy (An et al., 2025). Specifically, for the 20% training samples chosen for noise augmentation,
we prepend and append short silence segments to the utterance prior to noise injection, where the
duration of each silence segment is sampled from 0 to 1 second using a skewed Beta(1,3) distribution.
This strategy helps mitigate hallucinations in both offline and streaming inference. It is particularly
beneficial for streaming scenarios, where pauses between words or phrases may cause individual
chunks to contain a non-negligible proportion of non-speech frames that can trigger erroneous outputs.
By explicitly exposing the model to such cases during training, it learns to better distinguish speech
from non-speech content, thereby reducing the risk of hallucinations.
Training settings.The model is trained using the Adam optimizer (Kingma, 2014) with cosine
annealing and a 10k-step warm-up (except for RL). Our training corpus consists exclusively of
Mandarin, Chinese dialects, English, and code-switched Mandarin–English speech data. Table 1
details the training data scale and maximum learning rate for each stage.
Table 1: Training details for all stages.
Stage Training Data Maximum Learning Rate
Stage 1 Encoder pretraining 560k hours5×10−4
Stage 2 Alignment 50k hours1×10−3
Stage 3 IA-SFT 560k hours1×10−5
Stage 4 Late joint SFT 560k hours1×10−5
Stage 5 Context SFT 50k hours1×10−6
Stage 6 RL 20k samples2×10−6
2.4 Inference
2.4.1 Optimized Streaming Inference Pipeline for Real-time Speech Interactions
To achieve low-latency and high-throughput deployment in real-world streaming scenarios, NIM4-
ASR adopts a decoupled inference architecture, allowing different modules to be deployed on separate
accelerators to better utilize heterogeneous computing resources. The speech encoder is deployed on
8

Triton Inference Server2, enabling dynamic batching across concurrent audio streams and significantly
improving GPU utilization under high request concurrency. The adaptor and LLM decoder are served
using a vLLM-based inference engine3that provides efficient KV-cache management. During
inference, the encoder continuously processes incoming audio and transmits speech representations
to the vLLM server, where they are projected into speech embeddings and appended to the LLM
context. In addition, both the phoneme-level CTC head and the RAG module run on the CPU, where
the CTC head produces phoneme hypotheses that are used for hotword retrieval.
To make the decoding pipeline more streaming-friendly, the prompt structure follows a fixed ordering.
A static instruction prefix (e.g., system prompts and task instructions) is placed at the beginning of
the context and can therefore be pre-computed and cached in the KV-cache before inference begins.
Streaming speech embeddings are then appended incrementally as audio chunks arrive. Finally,
contextual information such as hotwords retrieved from the RAG module is injected as dynamic
textual context at the end of the prompt. This ordering allows the static prefix to be cached once,
while speech embeddings and hotword context can be prefetched incrementally during streaming
inference, reducing redundant KV-cache computation and improving decoding efficiency.
Existing LLM-based streaming ASR systems generally adopt chunk-based audio processing in the
encoder, but differ in their decoding paradigms. The first paradigm performs incremental streaming
recognition through periodic hypothesis refresh. In this design, the system repeatedly updates the
transcription hypothesis by decoding over the accumulated context as new audio becomes available.
The decoder conditions on previously generated text as a prefix prompt and may roll back several
recent tokens to allow local corrections when new acoustic evidence is observed (Liu et al., 2020).
Consequently, each incoming audio chunk triggers another decoding pass and requires re-prefilling the
LLM with the refreshed prefix. Such designs are well suited for long-form transcription scenarios (e.g.,
meetings and live streaming), where real-time on-screen display and iterative hypothesis refinement
are beneficial. However, as discussed in (Xia et al., 2026), this paradigm introduces two practical
inefficiencies: repeated decoding over the accumulated audio leading to substantial computational
redundancy, and unstable latency caused by hypothesis selection and revision mechanisms, which
increases the overall end-to-end latency despite enabling early partial outputs.
In contrast, NIM4-ASR adopts the second streaming inference paradigm with incremental context
extension tailored for real-time speech interaction. The streaming encoder processes incoming speech
chunk by chunk and incrementally extends the LLM context without repeatedly re-encoding the
entire audio history. To support this design, the streaming encoder operates with a chunk size of
640 ms. Each chunk is encoded immediately and the resulting speech representations are appended
to the LLM context through a streaming chunked prefill mechanism. During inference, the encoder
caches representations from the previous 4 chunks, allowing the current chunk to attend to a limited
left context while avoiding redundant computation over earlier audio segments. This design follows a
cache-aware streaming strategy, where intermediate representations are reused across chunks rather
than recomputed (Noroozi et al., 2024). Finally, the LLM decoder performs a single final decoding
pass when the voice activity detection (V AD) module detects the end of speech. Since the LLM
context is incrementally constructed through streaming chunked prefill during speech processing,
most KV-cache prefill computation is completed beforehand, allowing the decoder to start generation
immediately with minimal additional prefill overhead.
This design introduces a different trade-off compared with hypothesis-refresh pipelines. Refresh-
based streaming systems typically produce early partial hypotheses, enabling real-time on-screen
display. This design usually yields more favorable time-to-first-token (TTFT) and allows the hy-
potheses to be continuously refined as additional audio context becomes available. Such designs are
particularly effective for long-form transcription scenarios, where the primary objective is to enable
real-time display while maximizing recognition accuracy through iterative hypothesis refinement
over time. In contrast, NIM4-ASR performs a single continuous decoding process with incremental
context extension, avoiding repeated decoding passes over the accumulated audio. As a result, this
design prioritizes stable streaming inference rather than aggressive early hypothesis generation. It
maintains competitive initial response latency while substantially reducing tail latency. For typical
real-time speech interactions where utterances are short and instruction-oriented, repeated hypothesis
revision is often unnecessary. As a result, NIM4-ASR is better suited for real-time interactive speech
2https://github.com/triton-inference-server/server
3https://github.com/vllm-project/vllm
9

applications where end-to-end latency is critical, since downstream modules typically require a stable
complete sentence rather than frequently revised partial hypotheses.
2.4.2 Phoneme-based RAG for Hotword Customization
To enable efficient hotword customization, NIM4-ASR builds a phoneme-based hotword database
with a corresponding retrieval algorithm, as illustrated in Figure 1. Following prior work (An et al.,
2025), we preconvert each hotword text into a phoneme-token sequence and store it as a key-value
pair, where the key is the phoneme sequence and the value is the corresponding hotword text. These
phoneme sequences are first converted into discrete indices based on the phoneme vocabulary, and
then restructured into a trie augmented with failure links using the Aho-Corasick automaton (Aho and
Corasick, 1975) algorithm. During inference, the phoneme head attached to the encoder generates
phoneme hypotheses via greedy decoding, which are converted into index sequences and scanned by
the automaton in a single pass. When a partial match cannot be extended, the automaton follows the
failure link to the longest valid suffix state instead of restarting the search from scratch, enabling all
candidate hotwords to be retrieved with linear-time complexity in the hypothesis length.
To reduce redundant contextual hints, we apply a longest-match filtering strategy: shorter matches
fully covered by longer spans are discarded, retaining only the longest entity. For example, if both the
hotwords “NIO” and “NIO House” are matched in the same hypothesis, only “NIO House” is retained.
The retrieved hotword texts are then concatenated and injected into the LLM prompt as contextual
hints together with the speech embeddings, providing context-aware biasing for decoding. Owing to
the storage efficiency of index-level mapping and the linear-time complexity of the Aho-Corasick
automaton that depends only on query length rather than database size, the hotword database can
easily scale to millions of entries while maintaining sub-millisecond retrieval latency per query.
It is worth noting that our hotword customization is designed to optimize the recognition of named
entities such as location names and media titles, where the hotword database can be large and may
contain numerous phonetically similar or even homophonous entries. To ensure retrieval precision
under such large-scale settings, we adopt a hard-matching strategy in the RAG module, retrieving only
exact phoneme-sequence matches rather than approximate ones or those with minimal edit distance.
Empirically, retrieval misses are often less harmful than retrieval errors, since the LLM can still
recover the correct entity from internal linguistic knowledge and context. By contrast, soft matching
is more prone to introducing similar but incorrect hotwords, which can interfere with decoding even
if the model is robust to noisy contextual hints to some extent.
3 Evaluation
3.1 Evaluation Setup
We evaluate NIM4-ASR on both public benchmarks and internal benchmarks to assess its performance
across diverse domains.
Baseline Systems.We compare NIM4-ASR with several recent representative open-source LLM-
based ASR models, including Fun-ASR-Nano (An et al., 2025), GLM-ASR-Nano4, Qwen3-ASR-
1.7B (Shi et al., 2026), and FireRedASR2S-LLM (Xu et al., 2026). In addition, we also compare
NIM4-ASR against large audio language models (LALMs) and multimodal LLMs with strong ASR
capabilities, including Step-Audio2-Mini (Wu et al., 2025) and Qwen3-Omni-Instruct (Xu et al.,
2025a). While Fun-ASR, Qwen3-ASR and Qwen3-Omni support streaming inference, all baselines
are evaluated in the offline setting for fair comparison. For NIM4-ASR, we report results under both
offline and streaming inference settings.
Evaluation Metrics.We report Word Error Rate (WER) for English benchmarks, and Character
Error Rate (CER) for Mandarin, Chinese dialect, lyrics, and code-switched Chinese-English bench-
marks. As our internal benchmarks mainly consist of Mandarin speech, we use CER by default for
internal evaluation. To minimize the influence of surface-level variation such as numeric expression
formats and filler-word usage on evaluation statistics, we apply WeTextProcessing5, a WFST-based
4https://huggingface.co/zai-org/GLM-ASR-Nano-2512
5https://github.com/wenet-e2e/WeTextProcessing
10

toolkit for text normalization. This process may result in relatively lower absolute error rates across
models, but it enables a fairer comparison of their intrinsic recognition capabilities. All baselines
are reproduced following the official guidelines, and all transcriptions are normalized with the same
pipeline to ensure consistent cross-system evaluation.
Public Benchmarks.Public evaluation datasets cover a wide range of speech recognition sce-
narios. English benchmarks include LibriSpeech (Panayotov et al., 2015), V oxPopuli (Wang et al.,
2021), and MLS-English (Pratap et al., 2020). Mandarin benchmarks include AISHELL-1 (Bu
et al., 2017), AISHELL-2 (Du et al., 2018), AISHELL-2021-Eval6, WeNetSpeech (Zhang et al.,
2022a), and SpeechIO7. Chinese dialect evaluation includes WeNetSpeech-Chuan (Dai et al., 2025),
WeNetSpeech-Yue (Li et al., 2025), and KeSpeech (Tang et al., 2021). Additional challenging
benchmarks include Mandarin-English code-switching speech from CS-Dialogue (Zhou et al., 2025)
and ASCEND (Lovenia et al., 2022), as well as lyric transcription on M4Singer (Zhang et al., 2022b).
Internal Benchmarks.We further evaluate on a collection of internal benchmarks focused on
realistic in-car spontaneous speech scenarios—a setting that differs markedly from conventional
read-speech or conference-style corpora. These benchmarks mainly comprise instructional and
conversational utterances that reflect real-world user interaction patterns, offering a more practical
measure of ASR reliability in diverse in-car scenarios. All data were created by designing utterances
grounded in real-world cockpit scenarios, and then collected through crowdsourced speaker recording.
•Point of Interest (POI)data contains city-level POIs, which are derived from location names
across different cities.
•Mediadata involves media-related entities, including music titles, video titles, and radio program
names.
•Device Controldata contains in-car control commands, such as vehicle setting adjustments and
cockpit operation instructions.
•Conversationaldata includes two categories of conversational interactions: (1) Vehicle-domain
chat data focuses on vehicle-related conversations such as in-car knowledge queries and assistant
interactions; (2) Multi-domain chat data covers open-domain conversational queries across diverse
domains including media, sports, healthcare, history, arts, literature, ecology, tourism, technology,
science, culture, education, finance and entertainment.
3.2 Evaluation Results
3.2.1 Public Benchmarks
Table 2 reports the comparison results on public benchmarks. For NIM4-ASR, we report both offline
and streaming inference results. The offline setting reflects the upper-bound performance when full
acoustic context is available, while the streaming setting evaluates real-time recognition.
Overall, NIM4-ASR shows strong competitiveness in the offline setting. It consistently outperforms
baselines with smaller model sizes and achieves comparable or superior results against systems with
more than 8B parameters. Across open-source benchmarks, NIM4-ASR delivers robust performance
on Mandarin, dialectal speech, English, and code-switching. The main exception is meeting-style
benchmarks, such as WeNetSpeech Meeting, where it performs slightly worse than competing models.
This behavior is expected because NIM4-ASR is primarily optimized for streaming speech interaction
scenarios that require low-latency responses to short and medium-length utterances. In contrast, long-
form meeting transcription lies outside the primary design scope of the system and is correspondingly
less represented in the training data.
Beyond the offline comparison, we find that NIM4-ASR also achieves satisfactory performance
in the streaming mode, with only limited degradation relative to offline decoding. This can be
attributed to two factors: first, the strict local alignment induced by CTC helps maintain stable
acoustic representations under chunk-wise streaming inference; second, our dynamic chunk size and
context length streaming training strategy enables the model to make robust predictions even with
constrained acoustic context.
6https://aishelltech.com/aishell_2021_eval
7https://github.com/SpeechColab/Leaderboard
11

Table 2: Comparison with recent advanced baselines on public benchmarks. All baseline systems are
evaluated in offline mode. “N/A” denotes that a reliable result cannot be obtained under the official
inference interface.
Fun-ASR GLM-ASR Qwen3-ASR FireRedASR2S Step-Audio2 Qwen3-Omni NIM4-ASR
Nano Nano 1.7B LLM Mini Instruct Offline Stream
Model Size 0.8B 1.5B 2.0B 8B+ 8B+ 30B-A3B 2.3B 2.3B
Mandarin
AISHELL-1dev | test1.59 | 1.81 2.40 | 2.41 1.40 | 1.51 0.60 | 0.64 0.76 | 0.81 0.86 | 0.92 0.43 | 0.57 0.43 | 0.60
AISHELL-2-iosdev | test2.62 | 2.73 3.21 | 3.45 2.41 | 2.60 2.07 | 2.08 2.24 | 2.29 2.11 | 2.31 2.28 | 2.43 2.33 | 2.49
AISHELL-2021-EvalA | C | D4.75 | 4.29 | 2.33 7.25 | 9.48 | 3.40 4.22 | 3.51 | 1.82 13.40 | 3.92 | 4.68 4.54 | 3.69 | 2.34 5.19 | 3.34 | 1.66 3.12 | 1.51 | 1.81 3.28 | 1.63 | 2.22
WeNetSpeechmeeting | net4.68 | 5.22 6.87 | 5.72 4.00 | 4.13 3.36 | 3.52 4.23 | 4.63 3.92 | 3.85 4.91 | 4.72 5.71 | 5.00
SpeechIO 2.78 3.17 2.55 2.20 3.41 2.33 2.61 2.84
Chinese Dialect
WeNetSpeech-Chuaneasy | hard13.21 | 23.76 20.95 | 33.61 11.18 | 20.35 10.36 | 20.07 13.99 | 25.35 14.13 | 25.16 10.51 | 20.58 11.22 | 20.37
WeNetSpeech-Yueshort | long7.31 | 10.02 16.78 | 13.97 5.79 | 8.00 5.05 | 10.45 7.78 | 8.44 6.97 | 8.60 5.12 | 8.58 5.39 | 9.62
KeSpeech 7.18 9.59 4.98 3.05 3.98 6.00 4.40 5.08
English
LibriSpeech-devclean | other1.63 | 4.06 1.82 | 3.93 1.54 | 3.14 1.27 | 2.63 1.06 | 2.48 1.08 | 2.10 1.13 | 2.45 1.18 | 2.86
LibriSpeech-testclean | other1.63 | 4.35 1.96 | 4.29 1.56 | 3.49 1.29 | 2.97 1.22 | 2.61 1.15 | 2.38 1.19 | 2.53 1.29 | 2.92
V oxPopulidev | test7.86 | 7.70 8.78 | 8.52 7.58 | 7.42 9.38 | 9.24 8.86 | 8.37 6.86 | 6.75 6.18 | 6.08 6.26 | 6.22
MLS-English 6.80 5.32 4.93 4.71 4.37 4.04 4.77 5.04
Mandarin-English Code-switch
CS-Dialogue 5.37 6.15 5.44 4.63 9.46 8.51 4.70 4.91
ASCEND 11.91 12.29 10.87 10.22 13.50 18.68 11.46 11.85
Lyrics
M4Singer 5.25 18.45 5.72 N/A 9.68 8.40 6.39 6.94
NIM4-ASR offline vs. Baselines
Win : Lose 23:2 25:0 18:7 12:12 17:8 14:11 - -
3.2.2 Internal Benchmarks
Table 3: Comparison with recent advanced baselines on internal benchmarks. All baseline systems
are evaluated in offline mode. NIM4-ASR demonstrates consistent performance advantages on
most internal benchmarks, as the evaluated content largely consists of long-tail named entities that
open-source models rarely encounter during training.
Fun-ASR GLM-ASR Qwen3-ASR FireRedASR2S Step-Audio2 Qwen3-Omni NIM4-ASR
Nano Nano 1.7B LLM Mini Instruct Offline Stream
Model Size 0.8B 1.5B 2.0B 8B+ 8B+ 30B-A3B 2.3B 2.3B
Point of Interest (POI)
City A 7.07 14.68 9.14 8.54 9.41 9.67 3.86 3.85
City B 8.50 15.75 10.59 10.43 11.67 11.73 4.86 4.94
City C 7.60 17.55 10.01 10.17 11.35 12.18 3.77 3.81
City D 7.42 17.91 9.77 9.51 11.55 10.86 4.10 4.17
Media
Music 12.60 24.25 12.67 12.13 14.94 15.89 5.75 5.78
Video 8.27 20.35 9.69 9.38 12.30 15.33 2.99 3.03
Radio 13.69 19.82 10.51 11.84 14.21 17.91 1.21 1.17
Device Control
Vehicle control 4.74 8.78 5.31 4.52 4.97 4.18 1.88 1.78
Conversational
Vehicle-domain chateasy | hard3.75 | 5.92 5.63 | 10.12 3.31 | 5.96 2.93 | 5.61 2.35 | 7.63 5.98 | 6.60 2.70 | 4.88 2.76 | 4.83
Multi-domain chat 1.65 1.89 1.33 1.27 1.49 5.34 1.55 1.75
Table 3 reports results on our internal benchmarks. Two benchmarks, POI and Media, are entity-
intensive, comprising dense location names and media-related entities respectively. A key challenge
in these domains is that many entities share similar or identical pronunciations, requiring the model to
simultaneously resolve subtle acoustic differences and leverage contextual semantics to disambiguate
competing candidates. NIM4-ASR achieves particularly strong performance on these benchmarks,
driven primarily by comprehensive in-domain training data coverage, but also indicating that our
training strategy effectively preserves both the encoder’s fine-grained acoustic discriminability and
the LLM’s capacity for context-driven entity resolution.
Furthermore, NIM4-ASR also delivers clear improvements on both the vehicle control and vehicle-
domain chat benchmarks. We attribute this gap primarily to the long-tailed nature of domain
knowledge and terminology in general-purpose foundation models. By substantially increasing in-
domain data coverage, NIM4-ASR achieves more reliable recognition of vehicle control commands
and in-car assistant knowledge, thereby delivering a superior interaction experience within the vehicle
cockpit. By contrast, on the multi-domain chat benchmark, spanning open-domain topics without
vehicle-specific content, NIM4-ASR no longer leads but remains competitive. This demonstrates
12

the model’s strong generalization ability: despite limited training data coverage in domains such as
sports, healthcare, and finance, NIM4-ASR still maintains robust performance, indicating that its
gains are not solely driven by domain-specific data expansion.
3.2.3 Effectiveness of Hotword Customization
Table 4: Effectiveness of phoneme-based hotword RAG on internal entity-intensive POI benchmarks.
Recall here refers to the proportion of POI entities correctly recognized in the transcription output.
Streaming Streaming+RAG
CER / Recall CER / Recall
City A POI 3.85 / 82.63 3.33 / 88.07
City B POI 4.94 / 77.48 4.31 / 83.60
Beyond its strong fundamental recognition capability, NIM4-ASR also provides an effective hotword
customization mechanism. Through contextual hotword conditioning, NIM4-ASR can improve
recognition accuracy for acoustically similar entity names, domain-specific terminology, and newly
emerging expressions. To evaluate the effectiveness of the proposed hotword RAG mechanism, we
focus on entity-intensive POI recognition scenarios, selecting benchmarks from two major cities and
constructing city-specific retrieval databases, each comprising millions of location name–phoneme
pairs. As shown in Table 4, incorporating hotword context consistently improves streaming perfor-
mance, demonstrating the effectiveness of our phoneme-based RAG retrieval mechanism and its
practical benefit in entity-intensive recognition scenarios.
It is worth noting that, unlike previous work, we adopt exact matching rather than edit distance for
retrieval. We argue that for the RAG module in LLM-based ASR systems, retrieval precision is more
critical than recall, as the model’s strong inherent recognition capability already serves as a reliable
fallback when no hotword is retrieved. Moreover, pairing exact matching with the Aho-Corasick
algorithm allows the hotword database to scale to millions of entries without additional retrieval
overhead, avoiding the latency and precision degradation that typically follows vocabulary expansion.
3.2.4 Effectiveness on Hallucination Mitigation
Table 5: Hallucination rate on different benchmark scenarios. “w/o RL” and “w/ RL” denote model
after joint SFT and after the subsequent RL stage, respectively. For fair comparison, reported results
for NIM4-ASR are obtained under offline inference.
Model Mandarin Dialect English Code-switch Lyrics
Fun-ASR-nano 0.018% 0.217% 0.014% 0.397% 0.153%
GLM-ASR-nano 0.030% 0.201% 0.014% 0.315% 0.580%
Qwen3-ASR-1.7B 0.018% 0.120% 0.014% 0.345% 0.249%
FireRedASR2S-LLM 0.165% 0.298% 0.014% 0.335% 1.775%
Step-Audio2-mini 0.020% 0.194% 0.014% 1.255% 0.390%
Qwen3-Omni-Inst 0.013% 0.370%0.007%1.778% 0.129%
NIM4-ASR (offline w/o RL) 0.003% 0.122%0.007% 0.261%0.215%
NIM4-ASR (offline w/ RL)0.002% 0.117% 0.007% 0.261% 0.081%
Beyond recognition performance, NIM4-ASR demonstrates strong hallucination suppression. We
compare all baseline models and NIM4-ASR in terms of hallucination rate across five distinct
scenarios, where the rate for each scenario is defined as the ratio of hallucinated samples to total
samples aggregated over all benchmarks within that scenario. Specifically, a sample is classified
as hallucinated if its transcription exceeds the ground-truth length by over 50% with negligible
lexical overlap. Notably, we exclude three benchmarks: WeNetSpeech Meeting, SpeechIO, and
MLS-English from this evaluation, as no hallucinated samples are observed across any model; we
additionally exclude WeNetSpeech Net, as its prevalence of unreliably annotated short samples
inflates hallucination rates across all models.
13

As shown in Table 5, NIM4-ASR achieves substantially lower hallucination rates compared to
all baseline models. Attributed to our training paradigm design and noise data augmentation, the
model after joint SFT already exhibits a low hallucination rate: only marginally above the best-
performing baseline on Dialect and Lyrics benchmarks. After the RL stage, the hallucination rate is
further reduced, achieving the lowest average across all five scenarios; most notably, on Mandarin
benchmarks, NIM4-ASR attains a hallucination rate of 0.002%, substantially below all baselines.
3.2.5 Effectiveness of RL
Table 6: Effectiveness of the RL stage under different inference settings. “w/o RL” and “w/ RL”
correspond to the model after joint SFT and after the RL stage, respectively.
w/o RL w/ RL RL Gain (+/-)
Offline Streaming Offline Streaming Offline Streaming
Open-source Mandarin Avg. 2.71 2.80 2.44 2.65 -0.27 -0.15
Open-source English Avg. 3.55 3.76 3.48 3.68 -0.07 -0.08
Mandarin–English Code-switch Avg. 8.39 8.62 8.08 8.38 -0.31 -0.24
Internal Mandarin Avg. 3.57 3.69 3.41 3.44 -0.16 -0.25
We further ablate the RL stage to assess its contribution. As shown in Table 6, incorporating RL
yields consistent improvements under both offline and streaming settings, with the most substantial
gains observed on Mandarin and code-switching benchmarks. A key contributing factor is the high
prevalence of homophone and near-homophone confusions in these scenarios, which token-level
teacher-forcing does not directly penalize effectively. By contrast, RL optimizes sequence-level
rewards over complete decoding trajectories, explicitly penalizing sentence-level error propagation
induced by phonetic confusion, reinforcing entity and phrase-level consistency, and mitigating
exposure bias (Chen et al., 2025). In code-switching scenarios, acoustic ambiguity at language-
switch boundaries and cross-lingual entity competition are particularly pronounced; sequence-level
feedback from RL can more effectively suppress erroneous language continuation and improve overall
transcription consistency under mixed Mandarin–English conditions.
4 Conclusion
In this work, we revisit LLM-based ASR from a deployment-oriented perspective and identify three
obstacles that continue to hinder practical adoption: limited downward scalability arising from cross-
modal alignment overhead, hallucination induced by representation drift during joint optimization,
and the lack of production-ready mechanisms for hotword customization. NIM4-ASR addresses these
challenges through targeted architectural design and a multi-stage training paradigm. By explicitly
anchoring each training stage to the functional boundaries of its constituent modules, NIM4-ASR
improves parameter utilization efficiency and mitigates hallucinations under acoustically ambiguous
conditions, thus building a more stable foundation for LLM-based streaming speech recognition.
Building on this principle, NIM4-ASR further incorporates a real-time streaming inference pipeline
and phoneme-level RAG to enable million-scale hotword customization. Extensive evaluation on 25
benchmarks demonstrates that NIM4-ASR achieves SOTA performance on several benchmarks with
only 2.3B parameters, while maintaining low-latency streaming capability and clear advantages in
entity-intensive scenarios. Overall, these results suggest that advancing LLM-based ASR relies not
only on scaling model capacity, but more importantly on co-designing model architecture, training
objectives, and inference strategies. NIM4-ASR thus provides a practical solution for building
efficient, robust, and customizable LLM-based ASR systems for real-time speech interaction.
5 Limitations and Future Work
Although NIM4-ASR has demonstrated strong recognition performance and practical effectiveness,
several key issues remain to be addressed in the next stage of system iteration. First, the current model
supports only Mandarin, English, and a limited set of Chinese dialects, leaving broader multilingual
and dialectal coverage as an important direction for future work. Second, the current model uses
only retrieved hotwords as contextual input and does not yet incorporate conversation history, leaving
14

room for improvement in cross-turn transcription consistency in multi-turn interaction scenarios. In
addition, the gains brought by RL are not yet sufficiently stable, suggesting that further optimization
is needed in both algorithm design and reward formulation. In future work, we plan to focus on the
following directions:
•(1) Expanding support for more languages and Chinese dialects, and developing more adaptive
hotword customization mechanisms for dialectal and accented speech.
•(2) Incorporating conversation history as additional contextual information to improve cross-turn
transcription consistency in multi-turn interaction scenarios.
•(3) Further improving streaming inference efficiency and enabling scalable RAG acceleration under
high-concurrency deployment settings.
•(4) Refining the RL algorithm and reward design to further improve system robustness and reduce
hallucinations.
References
Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang,
Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative
mixed-modal language models. InInternational Conference on Machine Learning, pages 265–279.
PMLR, 2023.
Alfred V . Aho and Margaret J. Corasick. Efficient string matching: an aid to bibliographic search.
Communications of the ACM, 18(6):333–340, 1975.
Keyu An, Yanni Chen, Zhigao Chen, Chong Deng, Zhihao Du, Changfeng Gao, Zhifu Gao, Bo Gong,
Xiangang Li, Yabin Li, et al. Fun-asr technical report.arXiv preprint arXiv:2509.12508, 2025.
Ye Bai, Jingping Chen, Jitong Chen, Wei Chen, Zhuo Chen, Chuang Ding, Linhao Dong, Qianqian
Dong, Yujiao Du, Kepan Gao, et al. Seed-asr: Understanding diverse speech and contexts with
llm-based speech recognition.arXiv preprint arXiv:2407.04675, 2024a.
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou.
Hallucination of multimodal large language models: A survey.arXiv preprint arXiv:2404.18930,
2024b.
Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. Aishell-1: An open-source mandarin
speech corpus and a speech recognition baseline. In2017 20th conference of the oriental chapter
of the international coordinating committee on speech databases and speech I/O systems and
assessment (O-COCOSDA), pages 1–5. IEEE, 2017.
William Chan, Navdeep Jaitly, Quoc Le, and Oriol Vinyals. Listen, attend and spell: A neural network
for large vocabulary conversational speech recognition. In2016 IEEE international conference on
acoustics, speech and signal processing (ICASSP), pages 4960–4964. IEEE, 2016.
Chen Chen, Ke Hu, Chao-Han Huck Yang, Ankita Pasad, Edresson Casanova, Weiqing Wang,
Szu-Wei Fu, Jason Li, Zhehuai Chen, Jagadeesh Balam, et al. Reinforcement learning enhanced
full-duplex spoken dialogue language models for conversational interactions. InSecond Conference
on Language Modeling, 2025.
Jan K Chorowski, Dzmitry Bahdanau, Dmitriy Serdyuk, Kyunghyun Cho, and Yoshua Bengio.
Attention-based models for speech recognition.Advances in neural information processing
systems, 28, 2015.
Yuhang Dai, Ziyu Zhang, Shuai Wang, Longhao Li, Zhao Guo, Tianlun Zuo, Shuiyuan Wang, Hongfei
Xue, Chengyou Wang, Qing Wang, et al. Wenetspeech-chuan: A large-scale sichuanese corpus
with rich annotation for dialectal speech processing.arXiv preprint arXiv:2509.18004, 2025.
Jiayu Du, Xingyu Na, Xuechen Liu, and Hui Bu. Aishell-2: Transforming mandarin asr research into
industrial scale.arXiv preprint arXiv:1808.10583, 2018.
Mark Endo and Serena Yeung-Levy. Downscaling intelligence: Exploring perception and reasoning
bottlenecks in small multimodal models.arXiv preprint arXiv:2511.17487, 2025.
15

Yassir Fathullah, Chunyang Wu, Egor Lakomkin, Junteng Jia, Yuan Shangguan, Ke Li, Jinxi Guo,
Wenhan Xiong, Jay Mahadeokar, Ozlem Kalinli, et al. Prompting large language models with
speech recognition abilities. InICASSP 2024-2024 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pages 13351–13355. IEEE, 2024.
Alex Graves. Sequence transduction with recurrent neural networks.arXiv preprint arXiv:1211.3711,
2012.
Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. Connectionist temporal
classification: labelling unsegmented sequence data with recurrent neural networks. InProceedings
of the 23rd international conference on Machine learning, pages 369–376, 2006.
Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo
Wang, Zhengdong Zhang, Yonghui Wu, et al. Conformer: Convolution-augmented transformer for
speech recognition.arXiv preprint arXiv:2005.08100, 2020.
Yukiya Hono, Koh Mitsuda, Tianyu Zhao, Kentaro Mitsui, Toshiaki Wakatsuki, and Kei Sawada.
Integrating pre-trained speech and language models for end-to-end speech recognition. InFindings
of the Association for Computational Linguistics: ACL 2024, pages 13289–13305, 2024.
Diederik P Kingma. Adam: A method for stochastic optimization.arXiv preprint arXiv:1412.6980,
2014.
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural
network representations revisited. InInternational conference on machine learning, pages 3519–
3529. PMlR, 2019.
Chin-Hung Kuo and Kuan-Yu Chen. Correcting, rescoring and matching: An n-best list selection
framework for speech recognition. In2022 Asia-Pacific Signal and Information Processing
Association Annual Summit and Conference (APSIPA ASC), pages 729–734. IEEE, 2022.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. InProceedings of the 29th symposium on operating systems principles,
pages 611–626, 2023.
Zhihong Lei, Xingyu Na, Mingbin Xu, Ernest Pusateri, Christophe Van Gysel, Yuanyuan Zhang,
Shiyi Han, and Zhen Huang. Contextualization of asr with llm using phonetic retrieval-based
augmentation. InICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 1–5. IEEE, 2025.
Longhao Li, Zhao Guo, Hongjie Chen, Yuhang Dai, Ziyu Zhang, Hongfei Xue, Tianlun Zuo,
Chengyou Wang, Shuiyuan Wang, Jie Li, et al. Wenetspeech-yue: A large-scale cantonese speech
corpus with multi-dimensional annotation.arXiv preprint arXiv:2509.03959, 2025.
Alexander H Liu, Andy Ehrenberg, Andy Lo, Clément Denoix, Corentin Barreau, Guillaume Lample,
Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Mud-
direddy, et al. V oxtral.arXiv preprint arXiv:2507.13264, 2025.
Danni Liu, Gerasimos Spanakis, and Jan Niehues. Low-latency sequence-to-sequence speech
recognition and translation by partial hypothesis selection.arXiv preprint arXiv:2005.11185, 2020.
Holy Lovenia, Samuel Cahyawijaya, Genta Indra Winata, Peng Xu, Xu Yan, Zihan Liu, Rita Frieske,
Tiezheng Yu, Wenliang Dai, Elham J Barezi, et al. Ascend: A spontaneous chinese-english dataset
for code-switching in multi-turn conversation. InProceedings of the 13th Language Resources
and Evaluation Conference (LREC), 2022.
Vahid Noroozi, Somshubra Majumdar, Ankur Kumar, Jagadeesh Balam, and Boris Ginsburg. Stateful
conformer with cache-based inference for streaming automatic speech recognition. InICASSP
2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
pages 12041–12045. IEEE, 2024.
16

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: an asr corpus
based on public domain audio books. In2015 IEEE international conference on acoustics, speech
and signal processing (ICASSP), pages 5206–5210. IEEE, 2015.
Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and
Quoc V Le. Specaugment: A simple data augmentation method for automatic speech recognition.
arXiv preprint arXiv:1904.08779, 2019.
Hansol Park, Hoseong Ahn, Junwon Moon, Yejin Lee, and Kyuhong Shim. Evaluating hallucina-
tions in multimodal llms with spoken queries under diverse acoustic conditions.arXiv preprint
arXiv:2510.08581, 2025.
Vineel Pratap, Qiantong Xu, Anuroop Sriram, Gabriel Synnaeve, and Ronan Collobert. Mls: A
large-scale multilingual dataset for speech research.arXiv preprint arXiv:2012.03411, 2020.
Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations
toward training trillion parameter models. InSC20: international conference for high performance
computing, networking, storage and analysis, pages 1–16. IEEE, 2020.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathemat-
ical reasoning in open language models.arXiv preprint arXiv:2402.03300, 2024.
Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun
Hao, Yu Xi, Baosong Yang, et al. Qwen3-asr technical report.arXiv preprint arXiv:2601.21337,
2026.
Yuanfeng Song, Di Jiang, Xuefang Zhao, Qian Xu, Raymond Chi-Wing Wong, Lixin Fan, and Qiang
Yang. L2rs: a learning-to-rescore mechanism for automatic speech recognition.arXiv preprint
arXiv:1910.11496, 2019.
Zheshu Song, Lu Wang, Wei Deng, Zhuo Yang, Yong Wu, and Bin Xia. Index-asr technical report.
arXiv preprint arXiv:2601.00890, 2025.
Zhiyuan Tang, Dong Wang, Yanguang Xu, Jianwei Sun, Xiaoning Lei, Shuaijiang Zhao, Cheng Wen,
Xingjun Tan, Chuandong Xie, Shuran Zhou, et al. Kespeech: An open source speech dataset of
mandarin and its eight subdialects. InThirty-fifth Conference on Neural Information Processing
Systems Datasets and Benchmarks Track (Round 2), 2021.
Geeyang Tay, Wentao Ma, Jaewon Lee, Yuzhi Tang, Daniel Lee, Weisu Yin, Dongming Shen, Silin
Meng, Yi Zhu, Mu Li, et al. Back to basics: Revisiting asr in the age of voice agents.arXiv
preprint arXiv:2603.25727, 2026.
Changhan Wang, Morgane Riviere, Ann Lee, Anne Wu, Chaitanya Talnikar, Daniel Haziza, Mary
Williamson, Juan Pino, and Emmanuel Dupoux. V oxpopuli: A large-scale multilingual speech cor-
pus for representation learning, semi-supervised learning and interpretation. InProceedings of the
59th Annual Meeting of the Association for Computational Linguistics and the 11th International
Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 993–1003,
2021.
He Wang, Linhan Ma, Dake Guo, Xiong Wang, Lei Xie, Jin Xu, and Junyang Lin. Contextasr-bench:
A massive contextual speech recognition benchmark.arXiv preprint arXiv:2507.05727, 2025.
Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang
Zhang, Jingbei Li, et al. Step-audio 2 technical report.arXiv preprint arXiv:2507.16632, 2025.
Yinfeng Xia, Jian Tang, Junfeng Hou, Gaopeng Xu, and Haitao Yao. Uni-asr: Unified llm-based
architecture for non-streaming and streaming automatic speech recognition.arXiv preprint
arXiv:2603.11123, 2026.
Yuan Xie, Jiaqi Song, Guang Qiu, Xianliang Wang, Ming Lei, Jie Gao, and Jie Wu. Rethinking
entropy allocation in llm-based asr: Understanding the dynamics between speech encoders and
llms.arXiv preprint arXiv:2604.08003, 2026.
17

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi,
Ting He, Xinfa Zhu, et al. Qwen3-omni technical report.arXiv preprint arXiv:2509.17765, 2025a.
Kai-Tuo Xu, Feng-Long Xie, Xu Tang, and Yao Hu. Fireredasr: Open-source industrial-grade
mandarin speech recognition models from encoder-decoder to llm integration.arXiv preprint
arXiv:2501.14350, 2025b.
Kaituo Xu, Yan Jia, Kai Huang, Junjie Chen, Wenpeng Li, Kun Liu, Feng-Long Xie, Xu Tang, and
Yao Hu. Fireredasr2s: A state-of-the-art industrial-grade all-in-one automatic speech recognition
system.arXiv preprint arXiv:2603.10420, 2026.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388,
2025.
Zengwei Yao, Wei Kang, Xiaoyu Yang, Fangjun Kuang, Liyong Guo, Han Zhu, Zengrui Jin, Zhaoqing
Li, Long Lin, and Daniel Povey. Cr-ctc: Consistency regularization on ctc for improved speech
recognition.arXiv preprint arXiv:2410.05101, 2024.
Saierdaer Yusuyin, Te Ma, Hao Huang, Wenbo Zhao, and Zhijian Ou. Whistle: Data-efficient multi-
lingual and crosslingual speech recognition via weakly phonetic supervision.IEEE Transactions
on Audio, Speech and Language Processing, 2025.
Binbin Zhang, Di Wu, Zhuoyuan Yao, Xiong Wang, Fan Yu, Chao Yang, Liyong Guo, Yaguang Hu,
Lei Xie, and Xin Lei. Unified streaming and non-streaming two-pass end-to-end model for speech
recognition.arXiv preprint arXiv:2012.05481, 2020.
Binbin Zhang, Hang Lv, Pengcheng Guo, Qijie Shao, Chao Yang, Lei Xie, Xin Xu, Hui Bu, Xiaoyu
Chen, Chenchen Zeng, et al. Wenetspeech: A 10000+ hours multi-domain mandarin corpus for
speech recognition. InICASSP 2022-2022 IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), pages 6182–6186. IEEE, 2022a.
Lichao Zhang, Ruiqi Li, Shoutong Wang, Liqun Deng, Jinglin Liu, Yi Ren, Jinzheng He, Rongjie
Huang, Jieming Zhu, Xiao Chen, and Zhou Zhao. M4singer: A multi-style, multi-singer and
musical score provided mandarin singing corpus. In S. Koyejo, S. Mohamed, A. Agarwal,
D. Belgrave, K. Cho, and A. Oh, editors,Advances in Neural Information Processing Systems,
volume 35, pages 6914–6926. Curran Associates, Inc., 2022b.
Yu Zhang, Mufan Xu, Xuefeng Bai, Pengfei Zhang, Yang Xiang, Min Zhang, et al. Instruction
anchors: Dissecting the causal dynamics of modality arbitration.arXiv preprint arXiv:2602.03677,
2026.
Guanyu Zhou, Yibo Yan, Xin Zou, Kun Wang, Aiwei Liu, and Xuming Hu. Mitigating modality prior-
induced hallucinations in multimodal large language models via deciphering attention causality.
arXiv preprint arXiv:2410.04780, 2024.
Jiaming Zhou, Yujie Guo, Shiwan Zhao, Haoqin Sun, Hui Wang, Jiabei He, Aobo Kong, Shiyao Wang,
Xi Yang, Yequan Wang, et al. Cs-dialogue: A 104-hour dataset of spontaneous mandarin-english
code-switching dialogues for speech recognition.arXiv preprint arXiv:2502.18913, 2025.
18