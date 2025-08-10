# Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS

**Authors**: Vignesh Ethiraj, Ashwath David, Sidhanth Menon, Divya Vijay

**Published**: 2025-08-05 07:39:35

**PDF URL**: [http://arxiv.org/pdf/2508.04721v1](http://arxiv.org/pdf/2508.04721v1)

## Abstract
We introduce a low-latency telecom AI voice agent pipeline for real-time,
interactive telecommunications use, enabling advanced voice AI for call center
automation, intelligent IVR (Interactive Voice Response), and AI-driven
customer support. The solution is built for telecom, combining four specialized
models by NetoAI: TSLAM, a 4-bit quantized Telecom-Specific Large Language
Model (LLM); T-VEC, a Telecom-Specific Embedding Model; TTE, a Telecom-Specific
Automatic Speech Recognition (ASR) model; and T-Synth, a Telecom-Specific
Text-to-Speech (TTS) model. These models enable highly responsive,
domain-adapted voice AI agents supporting knowledge-grounded spoken
interactions with low latency. The pipeline integrates streaming ASR (TTE),
conversational intelligence (TSLAM), retrieval augmented generation (RAG) over
telecom documents, and real-time TTS (T-Synth), setting a new benchmark for
telecom voice assistants. To evaluate the system, we built a dataset of 500
human-recorded telecom questions from RFCs, simulating real telecom agent
queries. This framework allows analysis of latency, domain relevance, and
real-time performance across the stack. Results show that TSLAM, TTE, and
T-Synth deliver real-time factors (RTF) below 1.0, supporting enterprise,
low-latency telecom deployments. These AI agents -- powered by TSLAM, TTE, and
T-Synth -- provide a foundation for next-generation telecom AI, enabling
automated customer support, diagnostics, and more.

## Full Text


<!-- PDF content starts -->

Toward Low-Latency End-to-End Voice Agents for Telecommunications
Using Streaming ASR, Quantized LLMs, and Real-Time TTS
Vignesh Ethiraj∗Ashwath David∗Sidhanth Menon∗Divya Vijay∗
NetoAI
support@netoai.ai
Abstract
We propose a low-latency, end-to-end voice-to-
voice communication pipeline, purpose-built
for real-time, interactive telecom scenarios
such as call center automation and conversa-
tional IVR (Interactive V oice Response) sys-
tems. Our system integrates streaming auto-
matic speech recognition (ASR), a 4-bit quan-
tized large language model (LLM), retrieval-
augmented generation (RAG) over telecom doc-
uments, and real-time text-to-speech (TTS) to
enable responsive, knowledge-grounded spo-
ken interactions. To evaluate performance in
a realistic setting, we constructed a custom
dataset of 500 human-recorded utterances fea-
turing telecommunications-related questions
sourced from RFC documents. This bench-
mark simulates user queries to a telecom voice
agent and supports analysis of both latency
and domain relevance. The pipeline com-
bines sentence-level streaming, concurrent pro-
cessing, and vector-based document retrieval,
achieving real-time factors (RTF) below 1.0
across components. Results demonstrate the
system’s effectiveness for low-latency telecom
applications such as customer support and di-
agnostics.
1 Introduction
Real-time speech interfaces increasingly demand
low-latency processing across ASR, Natural Lan-
guage Understanding (NLU), and TTS. While sig-
nificant advances have been made in these areas
individually, integrating them into a single, low-
latency, end-to-end system remains challenging.
Naively chaining models sequentially results in cu-
mulative delays, limiting practical usability in con-
versational agents, voice summarization systems,
and assistive technologies.
This work addresses the challenge of minimiz-
ing end-to-end latency in a speech-input-to-speech-
output pipeline. We combine a pre-trained ASR
∗These authors contributed equally to this work.model, a quantized LLM for summarization, and
a real-time TTS system, connected via a multi-
threaded streaming architecture. Our contributions
include:
•Sentence-level streaming: The pipeline sup-
ports sentence-level streaming, allowing the
LLM to transmit generated sentences incre-
mentally to the TTS module for early and con-
tinuous audio output.
•4-bit LLM quantization: The LLM is quan-
tized to 4-bit precision, significantly reducing
GPU memory footprint and inference latency
while preserving generation quality.
•Concurrent module execution: ASR, LLM,
and TTS modules operate concurrently. They
are coordinated via a non-blocking producer-
consumer pattern, enabling seamless real-time
processing without blocking or unnecessary
wait times.
•Latency and performance analysis: The sys-
tem includes a detailed breakdown of latency
components, enabling understanding of per-
formance trade-offs and the effects of each
module and architectural choice on overall
responsiveness.
2 Related Work
Recent advances in streaming ASR, quantized
LLMs, and real-time TTS have enabled the de-
velopment of low-latency voice transformation
systems through careful architectural choices and
optimization techniques that directly inform our
pipeline implementation.
2.1 Conformer-Based Streaming ASR Models
Conformer-based ASR models have become the
main approach for streaming speech recognition.
They effectively capture both local and globalarXiv:2508.04721v1  [cs.SD]  5 Aug 2025

Input AudioStreaming ASR
T-Transcribe Engine
(TTE)
Embed T ranscript
FAISS Similarity
Sear ch
Documen t
Inde x
Retrieved
ChunksTranscript
Contextualized
PromptOutput AudioEncoding &
Decoding
(serialization)  Threading
T-VECTSLAM-Mini-2BLLM
TTS
T-SYNTHFigure 1: Overall pipeline architecture.
acoustic dependencies using convolutional and self-
attention layers (Gulati et al., 2020).
NVIDIA’s NeMo framework (NVIDIA, 2025)
offers optimized, pre-trained Conformer models
such as nvidia/stt_en_conformer_ctc_small .
These achieve real-time factors below 0.2 and com-
petitive word error rates (WER) on benchmarks
like LibriSpeech (Panayotov et al., 2015).
Such models use connectionist temporal classi-
fication (CTC) for alignment-free, frame-synchro-
nous output.
Other state-of-the-art toolkits and models, such
as AI4Bharat’s IndicConformer for multilingual
Indian ASR (AI4Bharat, 2024), AssemblyAI’s
Conformer-1 model (AssemblyAI, 2023), and the
open-source SpeechBrain toolkit (SpeechBrain,
2022), further expand Conformer adoption to large-
scale multilingual, noisy, and domain-specific
speech tasks. Our pipeline leverages our propri-
etary, telecom-optimized T-Transcribe Engine
(TTE) , based on a Conformer-CTC architecture.
Designed specifically for real-time conversational
and call-center scenarios, TTE balances recogni-
tion accuracy with low-latency inference to meet
the stringent requirements of telecom domain ap-
plications.
2.2 4-bit Quantization for LLM Deployment
Post-training quantization techniques have proven
effective for deploying large language models in
resource-constrained environments. Using the
BitsAndBytesConfig framework enables 4-bit
quantization with minimal performance degrada-
tion, achieving up to 40% latency reduction while
preserving generation quality (Gong et al., 2023).
Recent work on quantized conversational models
demonstrates that 4-bit precision maintains over
95% of original performance while reducing com-putational complexity by factors of 60 ×or more
(Biswas et al., 2025).
2.3 Streaming TTS and Parallel Synthesis
Modern neural vocoders have achieved real-time
synthesis through tensor-level optimizations and
chunked processing frameworks. Streaming TTS
systems that interleave text encoding and waveform
generation can reduce time-to-first-audio to under
50ms (Ellinas et al., 2020; Lee et al., 2023).
2.4 RAG Integration in Voice Systems
Retrieval-Augmented Generation (RAG) architec-
tures combine neural retrievers, such as dense em-
bedding models trained for semantic search with
generative language models, allowing new infor-
mation to be dynamically included in system re-
sponses without retraining the core model (Lewis
et al., 2020). Recent work demonstrates that RAG
techniques have been successfully extended from
text-only settings to voice and multimodal systems:
•WavRAG is a pioneering audio-integrated
RAG framework. It enables spoken dialogue
models to retrieve and utilize both audio and
text knowledge bases, with end-to-end audio
support for real-time, context-aware conversa-
tion (Chen et al., 2025).
•V oxRAG further enhances this approach by
implementing transcription-free, speech-to-
speech retrieval. This allows query and an-
swer generation entirely in the audio do-
main, showcasing the feasibility of RAG
for real-world, spoken question answering
tasks (Rackauckas and Hirschberg, 2025).
•RAG-based agents are already deployed in
voice assistants and IVR systems. These

combine speech-to-text, neural document re-
trieval, generative language models, and text-
to-speech synthesis for accurate, context-rich
spoken interactions, especially in customer
service and enterprise settings (Sambare et al.,
2025).
These developments have made RAG a founda-
tional technique for enriching voice systems with
up-to-date, domain-specific information at infer-
ence time.
3 Pipeline Architecture and
Implementation
Our end-to-end voice transformation pipeline inte-
grates streaming ASR, RAG, quantized LLM infer-
ence, and real-time TTS synthesis using a modular
multi-threaded architecture designed to minimize
end-to-end latency.
3.1 Streaming ASR Module
We employ our T-Transcribe Engine (TTE)
model, a Conformer-based architecture optimized
for real-time speech recognition with Connection-
ist Temporal Classification (CTC) training. This
model effectively balances low latency and tran-
scription accuracy with sub-0.2 real-time factor
(RTF) on GPU, making it suitable for streaming
applications. The ASR module transcribes audio
waveforms loaded via soundfile and generates
input text transcripts with precise timings recorded
for downstream metrics.
3.2 Retrieval-Augmented Generation (RAG)
To enhance factual grounding and contextual rele-
vance of the LLM responses, we integrate a RAG
submodule based on FAISS (Douze et al., 2024)
similarity search over document embeddings. Doc-
ument indexing is performed offline or at startup us-
ing the NetoAISolutions/T-VEC model (Ethiraj
et al., 2025a), which encodes documents into nor-
malized dense vectors. When a serialized FAISS
index and corresponding documents are cached,
these are loaded for efficiency; otherwise, the sys-
tem builds the index from text files in a config-
urable directory. Retrieval queries embed the ASR
transcript and perform inner-product search with
configurable kneighbors to provide relevant con-
text documents concatenated as input to the LLM
prompt. This design leverages efficient nearest
neighbor search algorithms to maintain sub-secondretrieval latency and integrates seamlessly with the
generation stage.
3.3 Quantized Large Language Model (LLM)
Inference
We utilize the NetoAISolutions/TSLAM-Mini-2B
(Ethiraj et al., 2025b) causal LLM, loaded via
Hugging Face Transformers (Wolf et al., 2020),
applying 4-bit post-training quantization via the
BitsAndBytes (Dettmers et al., 2022) library to
reduce GPU memory footprint and enable faster in-
ference without significant quality loss (Gong et al.,
2023). The tokenizer is configured with padding
tokens to support variable-length input sequences
safely. Streaming generation is implemented with
a custom PunctuatedBufferStreamer class that
segments output text into sentences in real-time us-
ing regex-based punctuation detection and places
serialized sentences in a thread-safe queue. This
streamer also captures fine-grained latency metrics
such as the time-to-first-token generation.
3.4 Real-Time Text-to-Speech Pipeline
The TTS submodule is implemented via our pro-
prietary, telecom-optimized T-SYNTH TTS model,
which leverages a vocal synthesis pipeline initial-
ized with a warmup routine on a reference voice to
reduce latency jitter by preloading required com-
ponents. The synthesis is performed in a dedi-
cated thread that consumes sentences serialized
by the streamer and converts text to waveform
chunks. These audio chunks are asynchronously
post-processed and concatenated to produce a sin-
gle WA V output file. Sentence-level synthesis tim-
ings are recorded, allowing for detailed analysis of
TTS synthesis overhead.
3.5 Multi-threading and Synchronization
Our implementation uses threading to parallelize
LLM generation and TTS synthesis, with custom
sentence streaming that segments LLM output in
real-time and feeds it to the TTS pipeline, achiev-
ing sub-second response times. The sentence queue
was given a timeout of 0.05 seconds to receive each
chunk of the LLM response as it came in, with ele-
ments in the queue being separated by appropriate
punctuations by the PunctuatedBufferStreamer .
The TTS thread was made to begin before the LLM
thread to let it load the necessary components be-
fore the first LLM response tokens started coming
in. A warm-up dummy TTS pipeline was also
introduced for the same purpose, as highlighted

previously. A technique of binary serialization was
also introduced between the LLM response and the
TTS generation to further reduce our pipeline time.
The response was packed into binary serials, and it
was then unpacked at the time of TTS generation.
msgpack was used to achieve this. These methods
helped reduce the end-to-end pipeline time signifi-
cantly, by about 0.8-1.0 seconds.
3.6 Metrics Reporting and Performance
Profiling
Performance and resource utilization metrics are
diligently tracked throughout the pipeline. The
MetricsReporter class collates timers for model
loading, ASR processing, RAG retrieval, LLM gen-
eration, and TTS synthesis. It reports latency break-
downs including ASR speed in words/sec, real-time
factors, LLM tokens/sec, and time-to-first-audio for
user experience insights. GPU details and memory
consumption are also included, enabling system-
level profiling.
The foundation of our low-latency voice trans-
formation pipeline is built upon a combination of
the following techniques:
• Conformer-based streaming ASR
• 4-bit quantized LLMs
• Parallel LLM response and TTS synthesis
• Binary serialization
• Efficient RAG retrieval
• Optimized threading
3.7 Model Initialization
Models are loaded once, and quantized models
are initialized using AutoModelForCausalLM with
BitsAndBytesConfig . TTS is warmed up with a
dummy sentence to reduce first-inference latency.
3.8 Streaming Sentence Generation
We introduce a PunctuatedBufferStreamer
class based on the HuggingFace TextStreamer . It
detects full sentences using punctuation and pushes
them into a thread-safe queue. This allows the
TTS system to begin audio synthesis while LLM
generation is ongoing.3.9 Multi-threaded Execution
LLM generation and TTS decoding run in parallel
threads. The LLM acts as a producer, and the TTS
system consumes sentences in FIFO order. ASR is
processed upfront due to its short execution time.
4 Experimental Setup
4.1 Dataset
To evaluate our telecom-oriented voice-to-voice
system, we constructed a custom dataset of 500
human-recorded utterances, each corresponding
to a spoken telecommunications-related question.
The prompts were sourced from open-access RFC
(Request for Comments) documents to ensure con-
tent relevance and compatibility with the down-
stream RAG-based retrieval system used by our
agent.
While large-scale voice datasets such as Lib-
riSpeech (Panayotov et al., 2015), Common V oice
(Ardila et al., 2020), and the Spoken Wikipedia
Corpora (Köhn et al., 2016) offer broad linguis-
tic coverage, they are either domain-agnostic or
narratively structured, making them suboptimal
for evaluating question-driven, real-time dialog
systems in specialized verticals like telecommu-
nications. Moreover, these corpora do not offer
fine-grained alignment with structured backend
knowledge (e.g., RFC indices), a critical feature for
retrieval-augmented generation (RAG) pipelines.
Our dataset bridges this gap by aligning natural
speech inputs with a vector index derived from the
same RFC documents. This alignment enables con-
trolled and realistic simulation of query-response
behavior in telecom use cases. Utterances were
recorded by two different speakers with varied ac-
cents and speaking rates to approximate deploy-
ment variability. Each file is stored in WA V format
and was manually verified for clarity and times-
tamp consistency. The dataset totals approximately
45 minutes of audio, with an average utterance du-
ration of 6.36 seconds.
This design supports robust benchmarking of
streaming latency, transcription accuracy, and
sentence-level retrieval performance in a setting
that closely mirrors the demands of real-world tele-
com voice agents.
4.2 Hardware
All experiments were conducted on a system
equipped with an NVIDIA H100 GPU (80GB)and

Stat ASR RAG LLM TTS Total ASR Speed LLM Speed Cosine TTFT TTFA
Processing Retrieval Generation Synthesis Time (words/sec) (tokens/sec) Similarity (%)
Mean 0.049 0.008 0.670 0.286 0.934 394.18 80.06 0.873 0.106 0.678
Min 0.029 0.008 0.218 0.106 0.417 134.24 58.60 0.659 0.077 0.412
Max 0.069 0.012 1.706 1.769 3.154 1010.15 86.97 1.000 0.181 1.482
Table 1: Latency and performance metrics across pipeline components. All times in seconds except where
mentioned.
Figure 2: Average latencies with min-max error bars
256 GB of RAM. The H100’s high memory band-
width and transformer-optimized architecture en-
abled real-time inference for quantized LLMs,
streaming ASR, and TTS components. All soft-
ware components were executed in a single-node
setup using mixed-precision inference where sup-
ported.
5 Evaluation and Results
5.1 Real-Time Performance
Our pipeline achieves near real-time performance
across all components. The average total latency
per utterance is 0.94 seconds , comfortably under
the 1-second threshold typically considered accept-
able for interactive systems. The ASR andTTS
modules operate with mean latencies of around
0.05s and0.28s , respectively, while the LLM is ex-
pectedly the most time-consuming, averaging 0.67s
per generation. Retrieval latency is negligible on
average ( 0.008s ).
5.2 Streaming Efficiency
To assess how the system handles input/output
streaming, we report time-to-first-token (TTFT)
and time-to-first-audio (TTFA) . The averageTTFT is 0.106s , suggesting that text generation be-
gins quickly after receiving input. TTFA is slightly
higher at 0.678 , which includes LLM latency and
the initial TTS processing. This shows that there is
an average gap of around 0.5 seconds between the
LLM generating its first set of tokens and the TTS
beginning its synthesis.
5.3 Model Throughput and Speed
The ASR operates at an average of 394 tokens/sec ,
and the LLM generates tokens at an average of 80
tokens/sec . These speeds indicate that the system
is well-suited for real-time applications, with ample
headroom for longer utterances or simultaneous
streams.
5.4 Semantic Preservation
Cosine similarity between the ASR transcript em-
beddings and the LLM-generated outputs averages
0.87, indicating strong semantic preservation dur-
ing transformation.
5.5 Latency Variability
While mean latencies are low, the worst-case to-
tal pipeline latency reached 3.154 seconds . Given
that the best case was at a low latency 0.417 sec-

onds , with the average being less than 1, it can be
inferred that GPU processing fluctuations were at
play in determining the average pipeline latency.
These outliers of 2+ seconds are rare, and this is fur-
ther supported by our general observations of the
pipeline usually settling comfortably at less than
0.7 seconds in most test runs.
6 Conclusion
We have introduced a low-latency, end-to-end voice
agent pipeline tailored for telecommunications ap-
plications, integrating streaming ASR, retrieval-
augmented generation with a quantized LLM, and
real-time TTS synthesis using a modular, multi-
threaded framework. Our detailed evaluation
demonstrates that streaming workflows, sentence-
level concurrency, aggressive quantization, and effi-
cient RAG significantly reduce total system latency
while preserving response quality and semantic rel-
evance. With an average response time below 1s
and strong performance across all components, the
pipeline meets the demanding requirements of real-
time interactive voice scenarios such as customer
support, diagnostics, and IVR replacement.
By open-sourcing our dataset and providing a
reproducible methodology, we lay the groundwork
for future research on scalable, low-latency spoken
dialog systems. We believe the techniques outlined
- spanning ASR architectures, LLM quantization,
document retrieval, and producer-consumer paral-
lelism - can be adapted to other verticals where
fast, knowledge-augmented voice interfaces are es-
sential. Future work will address scaling to more
diverse domains, supporting multilingual use cases,
and integrating adaptive learning for continual im-
provement in real-world deployments.
Limitations
A key limitation of our pipeline lies in the auto-
matic speech recognition (ASR) output used for
downstream processing. Inaccuracies in transcrip-
tion, especially in cases involving abbreviations,
proper nouns, or domain-specific terms can lead to
degraded performance in similarity computations
in RAG. Since our cosine similarity metric is com-
puted over these transcriptions, ASR errors directly
impact the final similarity scores.
Future work should consider adopting larger or
more specialized ASR models that are even better
adapted to the task domain. Improved ASR perfor-
mance would help mitigate transcription errors andthereby enhance the quality of semantic similarity
assessments.
These enhancements have the potential to im-
prove robustness and better capture the intended
meaning in noisy, imperfect, or domain-specific
transcriptions.
Ethics Statement
There are no ethical concerns to be discussed in
this implementation.
References
AI4Bharat. 2024. Indicconformer: Multilingual con-
former asr models. https://huggingface.co/ai4
bharat/indic-conformer-600m-multilingual .
Rosana Ardila, Megan Branson, Kelly Davis, Michael
Kohler, Josh Meyer, Michael Henretty, Reuben
Morais, Lindsay Saunders, Francis M. Tyers,
and Gregor Weber. 2020. Common V oice: A
massively-multilingual speech corpus. arXiv
preprint arXiv:1912.06670 .
AssemblyAI. 2023. Conformer-1: A robust speech
recognition model trained on 1 million hours of audio.
https://assemblyai.com/blog/conformer-1 .
Subrata Biswas, Mohammad Nur Hossain Khan, and
Bashima Islam. 2025. Quads: Quantized distillation
framework for efficient speech language understand-
ing.arXiv preprint arXiv:2505.14723 . Accepted at
INTERSPEECH 2025.
Yifu Chen, Shengpeng Ji, Haoxiao Wang, Ziqing Wang,
Siyu Chen, Jinzheng He, Jin Xu, and Zhou Zhao.
2025. Wavrag: Audio-integrated retrieval augmented
generation for spoken dialogue models. In Proceed-
ings of the 2025 Conference on Empirical Methods
in Natural Language Processing .
Tim Dettmers, Mike Lewis, Younes Belkada, and Luke
Zettlemoyer. 2022. Llm.int8(): 8-bit matrix multi-
plication for transformers at scale. arXiv preprint
arXiv:2208.07339 .
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The faiss library. arXiv preprint
arXiv:2401.08281 .
Nikolaos Ellinas, Georgios Vamvoukakis, Konstanti-
nos Markopoulos, Aimilios Chalamandaris, Georgia
Maniati, Panos Kakoulidis, Spyros Raptis, June Sig
Sung, Hyoungmin Park, and Pirros Tsiakoulis. 2020.
High quality streaming speech synthesis with low,
sentence-length-independent latency. In Interspeech
2020 , interspeech_2020, pages 2022–2026. ISCA.

Vignesh Ethiraj, Sidhanth Menon, and Divya Vijay.
2025a. T-vec: A telecom-specific vectorization
model with enhanced semantic understanding via
deep triplet loss fine-tuning.
Vignesh Ethiraj, Divya Vijay, Sidhanth Menon, and
Heblin Berscilla. 2025b. Efficient telecom specific
llm: Tslam-mini with qlora and digital twin data.
Zhuocheng Gong, Jiahao Liu, Qifan Wang, Yang Yang,
Jingang Wang, Wei Wu, Yunsen Xian, Dongyan
Zhao, and Rui Yan. 2023. Prequant: A task-agnostic
quantization approach for pre-trained language mod-
els. In Findings of the Association for Computational
Linguistics: ACL 2023 , pages 8065–8079.
Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki
Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang,
Zhengdong Zhang, Yonghui Wu, and Ruoming Pang.
2020. Conformer: Convolution-augmented trans-
former for speech recognition. In Proc. Interspeech .
Arne Köhn, Florian Stegen, and Timo Baumann. 2016.
Mining the spoken Wikipedia for speech data and
beyond. In Proceedings of the Tenth International
Conference on Language Resources and Evaluation
(LREC’16) , pages 4644–4647, Portorož, Slovenia.
European Language Resources Association (ELRA).
Jungil Lee, Wei Ping, Boris Ginsburg, and Bryan Catan-
zaro. 2023. Metavoc: Meta-learning for few-shot
text-to-speech with optimal transport. In Proc. IEEE
International Conference on Acoustics, Speech and
Signal Processing (ICASSP) , pages 1–5.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Kuttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. In Advances in Neural Infor-
mation Processing Systems , volume 33, pages 9459–
9474.
NVIDIA. 2025. Models: Nemo asr collection. https:
//docs.nvidia.com/nemo-framework/user-gui
de/latest/nemotoolkit/asr/models.html .
Vassil Panayotov, Guoguo Chen, Daniel Povey, and
Sanjeev Khudanpur. 2015. Librispeech: an asr cor-
pus based on public domain audio books. In 2015
IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP) , pages 5206–5210.
IEEE.
Zackary Rackauckas and Julia Hirschberg. 2025.
V oxrag: A step toward transcription-free rag sys-
tems in spoken question answering. arXiv preprint
arXiv:2505.17326 .
G. B. Sambare, Ganesh Kadam, Aditya Agre, Amay
Chandravanshi, Kanak Agrawal, and Parinitha Sam-
aga. 2025. Advancements in voice-activated systems:
A comprehensive survey on retrieval-augmented gen-
eration (rag) and large language model techniques.
IOSR Journal of Computer Engineering , 27(2):26–
40.SpeechBrain. 2022. Streaming speech recognition with
conformers. https://speechbrain.readthedoc
s.io/en/v1.0.2/tutorials/nn/conformer-str
eaming-asr.html .
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien
Chaumond, Clement Delangue, Anthony Moi, Pier-
ric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
Joe Davison, Sam Shleifer, Patrick von Platen, Clara
Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le
Scao, Sylvain Gugger, Mariama Drame, Quentin
Lhoest, and Alexander M. Rush. 2020. Transform-
ers: State-of-the-art natural language processing. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing: System
Demonstrations , pages 38–45, Online. Association
for Computational Linguistics.