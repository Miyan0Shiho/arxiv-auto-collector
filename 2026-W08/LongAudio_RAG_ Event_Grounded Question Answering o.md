# LongAudio-RAG: Event-Grounded Question Answering over Multi-Hour Long Audio

**Authors**: Naveen Vakada, Kartik Hegde, Arvind Krishna Sridhar, Yinyi Guo, Erik Visser

**Published**: 2026-02-16 10:15:22

**PDF URL**: [https://arxiv.org/pdf/2602.14612v1](https://arxiv.org/pdf/2602.14612v1)

## Abstract
Long-duration audio is increasingly common in industrial and consumer settings, yet reviewing multi-hour recordings is impractical, motivating systems that answer natural-language queries with precise temporal grounding and minimal hallucination. Existing audio-language models show promise, but long-audio question answering remains difficult due to context-length limits. We introduce LongAudio-RAG (LA-RAG), a hybrid framework that grounds Large Language Model (LLM) outputs in retrieved, timestamped acoustic event detections rather than raw audio. Multi-hour streams are converted into structured event records stored in an SQL database, and at inference time the system resolves natural-language time references, classifies intent, retrieves only the relevant events, and generates answers using this constrained evidence. To evaluate performance, we construct a synthetic long-audio benchmark by concatenating recordings with preserved timestamps and generating template-based question-answer pairs for detection, counting, and summarization tasks. Finally, we demonstrate the practicality of our approach by deploying it in a hybrid edge-cloud environment, where the audio grounding model runs on-device on IoT-class hardware while the LLM is hosted on a GPU-backed server. This architecture enables low-latency event extraction at the edge and high-quality language reasoning in the cloud. Experiments show that structured, event-level retrieval significantly improves accuracy compared to vanilla Retrieval-Augmented Generation (RAG) or text-to-SQL approaches.

## Full Text


<!-- PDF content starts -->

LongAudio-RAG: Event-Grounded Question Answering over Multi-Hour
Long Audio
Vakada Naveen, Kartik Hegde, Arvind Krishna Sridhar, Yinyi Guo, Erik Visser
Qualcomm Technologies Inc.,
nvakada@qti.qualcomm.com,karthegd@qti.qualcomm.com,arvisrid@qti.qualcomm.com,
yinyig@qti.qualcomm.com,evisser@qti.qualcomm.com
Abstract
Long-duration audio is increasingly com-
mon in industrial and consumer settings,
yet reviewing multi-hour recordings is im-
practical, motivating systems that answer
natural -language queries with precise tempo-
ral grounding and minimal hallucination. Ex-
isting audio–language models show promise,
but long -audio question answering remains
difficult due to context -length limits. We in-
troduce LongAudio-RAG (LA-RAG), a hy-
brid framework that grounds Large Language
Model (LLM) outputs in retrieved, times-
tamped acoustic event detections rather than
raw audio. Multi -hour streams are converted
into structured event records stored in an SQL
database, and at inference time the system re-
solves natural -language time references, clas-
sifies intent, retrieves only the relevant events,
and generates answers using this constrained
evidence. To evaluate performance, we con-
struct a synthetic long -audio benchmark by
concatenating recordings with preserved times-
tamps and generating template -based ques-
tion–answer pairs for detection, counting, and
summarization tasks. Finally, we demonstrate
the practicality of our approach by deploying
it in a hybrid edge–cloud environment, where
the audio grounding model runs on -device on
IoT-class hardware while the LLM is hosted on
a GPU -backed server. This architecture enables
low-latency event extraction at the edge and
high-quality language reasoning in the cloud.
Experiments show that structured, event -level
retrieval significantly improves accuracy com-
pared to vanilla Retrieval-Augmented Genera-
tion (RAG) or text-to-SQL approaches.
1 Introduction
Long-duration audio streams are now common not
only in industrial settings such as machine monitor-
ing and safety logging but also in homes through
smart assistants, baby monitors, and security sys-
tems. These recordings span many hours, making
manual review impractical and motivating systems
Figure 1: Chat Example for LongAudio-RAG
that can answer natural-language questions about
events and their timing. Unlike short-clip audio
tasks, long audio question answering (QA) must
handle time-bounded queries, aggregate counts,
and narrative summaries, requiring precise tem-
poral grounding and hallucination-free language
generation.
Recent progress in audio-language modeling has
enabled instruction-following and reasoning over
audio using large-scale (audio, question, answer)
training (Chu et al., 2023; Huang et al., 2024; Gong
et al., 2024; Ghosh et al., 2024; Goel et al., 2025).
Despite these advances, long audio QA remains
difficult in practice. First, raw multi-hour audio
cannot be directly ingested by most models due
to context-length and computation constraints, en-
couraging approaches that compress, segment, or
selectively attend to relevant evidence. Second,
natural-language time expressions are highly vari-
able (12-hour vs. 24-hour time, shift references,
relative phrases such as ‘before 5pm’), and errors
in time interpretation can invalidate downstream
reasoning. Third, open-ended generation over long
logs is prone to hallucination unless responses are
explicitly grounded in verifiable evidence.
Retrieval-augmented generation (RAG) reduces
hallucinations by grounding model outputs in re-
trieved evidence (Lewis et al., 2020; Gupta et al.,
2024). Although widely used for knowledge-
intensive QA and summarization, RAG perfor-
mance is tightly coupled to retrieval quality, es-
pecially for ambiguous or underspecified queries
(Karpukhin et al., 2020; Gupta et al., 2024). Re-arXiv:2602.14612v1  [eess.AS]  16 Feb 2026

cent work has extended these ideas beyond text:
audio-integrated RAG retrieves directly from au-
dio–text representations rather than relying solely
on transcripts (Chen et al., 2025), while long-
context video understanding similarly benefits from
retrieval-based selection of relevant segments to
manage large input spaces and improve faithful-
ness (Luo et al., 2024; Tan et al., 2025).
This work presents a hybrid framework forlong
audio question answeringthat grounds LLM out-
puts in timestamped acoustic detections rather than
generating responses directly from multi-hour au-
dio, for chatbot application as shown in Figure 1.
The system first converts long audio into a sequence
of detected events with event name, timestamps,
confidence scores and related attributes, stored in
a SQL database for efficient retrieval. At query
time, temporal expressions are resolved into con-
crete intervals, intent is classified (e.g., detection,
counting, summary), and only the relevant events
are passed to the LLM for answer generation as
shown in Figure 2.
To enable controlled evaluation of time-bounded
queries missing in existing short-audio datasets
(Wijngaard et al., 2025; Gemmeke et al., 2017)
we construct a synthetic long audio benchmark
by concatenating industrial and external record-
ings while preserving event timestamps. Template-
based question–answer pairs are generated across
detection, counting, and summary tasks with di-
verse temporal expressions, allowing systematic
testing of long audio reasoning.
Contributions.We present a comprehensive so-
lution to the open problem of performing reliable
question answering over multi -hour audio record-
ings that can span across days.
(i)We introduce a hybrid grounding framework
that anchors answers in timestamped acoustic
events rather than raw audio, enabling precise
temporal reasoning and reducing hallucina-
tion on hours-long content.
(ii)We proposed a complete, reproducible imple-
mentation stack, including event extraction,
SQL-backed retrieval, temporal reference res-
olution, intent classification, and evidence-
constrained generation.
(iii) We deploy the system in a hybrid edge-cloud
configuration: audio grounding runs on IoT
devices while the LLM is hosted on a GPU
server, a split that reduces bandwidth usage,
and scales language reasoning.(iv)We design the system to be latency-aware by
performing event detection and filtering at the
edge, ensuring that only compact, relevant ev-
idence is transmitted and enabling fast, stable
responses even for multi-hour queries.
(v)We proposed a synthetic long audio bench-
mark and demonstrated improvements over
text-to-SQL and RAG-based approaches.
To our knowledge, this is among the first event-
level, edge–cloud systems for multi-hour audio
question answering.
2 Related Work
Research on retrieval -augmented generation (RAG)
highlights the value of grounding LLM outputs in
external information to reduce hallucinations and
improve factuality. Canonical RAG pipelines pair
dense retrievers with generators to condition re-
sponses on retrieved passages (Lewis et al., 2020),
with surveys documenting challenges such as re-
trieval ambiguity, domain shift, and bias in retrieval
corpora (Gupta et al., 2024). Dense retrievers like
DPR enable semantic matching beyond token over-
lap, but often struggle when queries are underspec-
ified (Karpukhin et al., 2020), motivating hybrid
retrieval strategies and query refinement. Extend-
ing RAG beyond text has gained traction in au-
dio and video domains, where raw modalities ex-
ceed model context windows. Audio -focused ap-
proaches range from ASR -dependent pipelines to
end-to-end systems such as WavRAG, which re-
trieves from audio–text hybrid knowledge without
requiring transcription (Chen et al., 2025). Strong
pretrained encoders such as wav2vec 2.0 remain
crucial for robust audio retrieval and reasoning
(Baevski et al., 2020). In videos, RAG-like meth-
ods retrieve auxiliary OCR/ASR/object metadata
or select question -relevant temporal segments, as
in Video -RAG (Luo et al., 2024) and RAG -Adapter
(Tan et al., 2025). Efficient video -to-text conver-
sion frameworks like ViTA (Arefeen et al., 2024)
and spatiotemporal architectures including I3D
(Carreira and Zisserman, 2017) and TimeSformer
(Bertasius et al., 2021) further support long -video
retrieval and understanding.
Audio -language modeling and audio question an-
swering (AQA) focus on aligning acoustic events
with natural -language queries and often require
fine-grained temporal reasoning. Early diagnostic
datasets such as DAQA probe temporal and logical
understanding through synthetic event sequences
(Fayek and Johnson, 2020). Recent audio -language

Figure 2: LongAudio-RAG (LA-RAG): Proposed method for long audio question answering.
models (ALMs) combine audio encoders with
LLMs and rely on large -scale QA datasets to en-
able open -ended reasoning, with LTU trained on
OpenAQA -5M (Gong et al., 2024) and GAMA
introducing CompA -R for complex temporal meta-
data reasoning (Ghosh et al., 2024). Multi -turn
interaction datasets such as Audio Dialogues (Goel
et al., 2024) and multimodal instruction -tuning re-
sources like MULTIS (Zhao et al., 2023) expand
conversational and multimodal capabilities. Scal-
ing multimodal systems to handle long-duration
audio remains an open research challenge. Au-
dio Flamingo-3 (Goel et al., 2025; Yang et al.,
2024)demonstrates notable progress through cur-
riculum training and the use of curated long au-
dio datasets, enabling support for audio inputs of
up to 10 minutes , a limit that still falls short of
the extended-duration requirements of our target
use cases . In contrast, Qwen-3 Omni further ad-
vances long-context audio modeling by support-
ing audio inputs of up to 45 minutes, but still it
doesn’t scale to the length we support (Jin Xu et al.,
2025). Surveys cataloging audio -language corpora
note significant dataset overlap, bias, and language
imbalance (Wijngaard et al., 2025), with many re-
sources grounded in large event-labeled corpora
such as AudioSet (Gemmeke et al., 2017). Finally,
text-to-SQL systems (Hong et al., 2025; Deng et al.,
2021; Wang et al., 2025), can translate natural lan-
guage queries into SQL commands that can be
executed over traditional structured databases.
3 Methodology
3.1 Automated QA generation
We generate evaluation QA pairs using a two-tiered
approach that captures both controlled and realis-tic query variability for long -audio understanding
systems. Simple-QA pairs are created determin-
istically from ground -truth event annotations pro-
duced by the Audio Grounding Model (AGM) mod-
ule, covering detection, counting, and summary
queries expressed in an unambiguous HH:MM:SS
format. We created 3 types of queries - detection,
counting, summary . To better reflect real -world
usage, we also produce complex-QA pairs that in-
corporate diverse temporal expression types includ-
ing 24 -hour and 12 -hour formats,shift -based ref-
erences, before/after constraints, duration -based
phrasing, and other relative or segment-based inter-
vals. More details about the QA pairs can be found
in Appendix E.
3.2 Proposed approach for Long Audio
Question Answering
First we process a long audio file by running our
Audio Grounding Model (Section 3.2.1). using
shifted windows to generate a text log of metadata
( Eg: audio event name , start time etc) as shown in
Figure 2. This metadata is then inserted into a SQL
database that can be queried. Then we retrieve the
most relevant information from this database to an-
swer the query. The detailed step by step process to
generate the answering using the proposed method
is given below:
3.2.1 Audio Grounding Model (AGM)
To preserve the flexibility required for on-device
deployment across diverse acoustic environments
and application scenarios, we develop an open-
vocabulary Sound Event Detection (SED) model
based on text to audio grounding. Instead of rely-
ing on a fixed label set, the model localizes sound
events conditioned on free-form textual queries

Table 1: Evaluation results on the Simple-QA pairs ( n= 800 ) from the Home-IoT and Industrial-IoT datasets using
various baseline models.
DataAudio
Encoder ApproachDetection (%)
(n=300)Counting (%)
(n=300)Summary (%)
(n=200)Overall (%) Latency (s)
Home IoTAF3 RAG 49.87 53.73 23.60 44.75 5.77
AGM RAG 67.93 44.13 27.70 48.95 3.26
AGM Text2SQL 47.13 48.00 24.40 41.77 1.34
AGM LA-RAG (Ours)90.67 76.93 56.10 76.88 0.56
Industrial IoTAF3 RAG 46.51 49.83 24.80 42.17 5.43
AGM RAG 66.00 47.47 29.30 49.88 3.66
AGM Text2SQL 48.07 44.53 24.70 40.90 1.08
AGM LA-RAG (Ours)90.07 62.60 46.70 68.92 0.44
corresponding to the sounds of interest for a given
use case. Our Audio Grounding Model (AGM) is
trained following the phrase level WSTAG frame-
work described in (Xu et al., 2024), using the Au-
dioCaps (Kim et al., 2019) dataset from which
we extract phrase-level supervision. AGM adopts
the same architecture and hyperparameters as (Xu
et al., 2024): an audio encoder built from a CRNN
with eight convolutional layers followed by a bidi-
rectional GRU (BiGRU), and a text encoder com-
posed of a single word embedding layer with mean
pooling. Frame-level grounding scores are com-
puted via the cosine similarity between audio frame
embeddings and text tag embeddings, followed by
a sigmoid activation. The output of this module is
stored in JSON format which is then inserted into
a SQLite database for question answering. The
columns of the database are: audio event name,
start time , end time , confidence of the generated
class and other attributes like loudness.
3.2.2 Question Answering method
(1) Query Rephrasing:User queries are rewritten
into clearer and more explicit forms by incorporat-
ing chat history, clarifying missing details, and sta-
bilizing phrasing. This step reduces hallucinations,
improves consistency, and strengthens downstream
intent classification.
(2) Time Resolution:Natural-language time ex-
pressions are mapped to precise intervals through a
rule-based extractor that supports explicit ranges,
before/after constraints, duration-based phrasing,
and configurable shift references. Non-temporal re-
quests default to full-day intervals. When phrasing
falls outside the rule set, an LLM-prompting based
fallback method produces the start-end pair.
(3) Intent Classification:A hybrid classifier
identifies whether the query concerns summariza-
tion, detection, counting, or anomaly analysis.
Keyword-based detection provides fast matches,
while embedding based similarity captures indirect
or paraphrased intents, enabling accurate routing
across diverse query styles.(4) SQL based retrieval and LLM based Re-
sponse generation:After rephrasing, time reso-
lution, and intent detection, the system retrieves
relevant events from the database of audio events
metadata. Broad queries use the full event set,
whereas specific ones apply an embedding simi-
larity based (Tan et al., 2019) Top-k filter to use
the most relevant audio events . An intent-specific
prompt template (summary or detection/counting)
then conditions the LLM on filtered, timestamped
events to generate grounded and faithful responses.
Anomaly queries follow a separate specialized
workflow (Appendix I). More details on all these
steps along with the prompts used can be found in
Appendix F,G and M.
4 Results
We evaluate our system across multiple large lan-
guage models (LLMs), data-processing pipelines,
and query types to assess its effectiveness in real-
world Industrial IoT audio analytics. The evalu-
ation focuses on three downstream tasks derived
from machine logs:event detection,event counting
estimation, andnatural language summarization.
4.1 Dataset for evaluation
We built a controlled synthetic benchmark using
two 24-hour audio recordings, one for home IoT
(HIoT) and one for industrial IoT (IIoT) by concate-
nating short, labeled events to preserve timestamps
and vary density and noise (event classes listed in
Appendix B). To match evaluation scope, we re-
strict acoustic classes to these predefined lists and
pass the same closed set to downstream prompts,
enabling grounded reasoning over multi-hour time-
lines without drift. Background audio uses in-
dustrial white-noise machine recordings for IIoT
and low-hum ambience for HIoT, sourced from
Freesound (Fonseca et al., 2021). The final test
vectors have a global SNR of 6 dB and are nor-
malized to the Apple loudness target of –16 LUFS.

Table 2: Comparison Across Model Scales on Industrial-IoT dataset
Model Name # Active Params Detection (%) counting (%) Summary (%) Overall (%)
Phi-3-medium(Arah A, 2024) 14B 93.13 63.47 48.60 70.88
Minitron (Muralidharan et al., 2024) 8B 90.67 63.67 42.70 68.55
Qwen3-8B (Yang et al., 2025) 8B 86.20 43.53 36.90 57.88
Llama-3.1(Grattafiori et al., 2024) 8B 60.60 38.80 26.30 43.85
Qwen2.5 (Yang et al., 2024) 7B 90.73 64.60 38.50 67.88
Phi-3.5-MoE (Arah A, 2024) 6.6B (42B) 92.07 64.93 48.90 71.10
Phi-4-mini-instruct (Abouelenin et al., 2025) 3.8B 92.60 61.60 43.80 68.77
Llama-3.2 (Grattafiori et al., 2024) 3B 51.67 38.27 28.40 40.83
Qwen2.5 (Yang et al., 2024) 0.5B 70.80 41.67 38.20 51.74
Table 3: Time resolution module evaluation results
Difficulty Regex-only (%) LLM-only (%) Combined (%)
Easy 93.33 86.67 100.00
Medium 55.00 65.00 85.00
Hard 20.00 20.00 30.00
Overall 60.00 62.22 77.78
4.2 Evaluation metrics
We evaluated the system’s performance across
three question categories using GPT-4o as an auto-
mated judge with category-specific rubrics on a 1-5
scale. For detection questions (yes/no queries), the
evaluator primarily assessed answer correctness
while treating explanations as optional enhance-
ments, with correct answers scoring 4-5 and incor-
rect answers capped at 2 regardless of explanation
quality. Counting based questions were evaluated
based on numeric accuracy, where exact matches
scored 4-5, off-by-one or within-10% errors scored
3, and order-of-magnitude errors scored 1. Sum-
mary questions received the most nuanced evalu-
ation, considering content coverage, factual accu-
racy, completeness, and coherence, where scores
of 5 required all key points with accurate tempo-
ral/causal relationships, while scores of 3-4 indi-
cated partial coverage with minor omissions or in-
accuracies.
4.3 Baseline methods for comparison
RAG BaselineThe retrieval baseline encodes
each audio event as a concise text record containing
its tag, timestamp, duration, and loudness. At query
time, semantically relevant events are retrieved, and
the model answers using only these snippets, allow-
ing us to evaluate QA performance without explicit
structured reasoning. We compare two variants by
swapping our Audio Grounding Model with Au-
dio Flamingo 3 (AF3) as the audio encoder. In the
AF3-based RAG pipeline, we prompt the model to
generate event names, timestamps, and scene de-
scriptions, but it fails to produce reliably structured
outputs. Instead, audio is processed in 4-minute
chunks and stored in a vector database, with k= 5
used for all RAG experiments. We used sentence-
transformer(Reimers and Gurevych, 2019) based
all-MiniLM-L6-v2 as the embedding model for all
RAG experiments.Text-to-SQL BaselineThe text-to-SQL baseline
treats long -audio question answering as a struc-
tured querying task. An LLM is prompted to con-
vert the user query into an SQL query that can be
executed over the database. All audio events are
stored in a relational table, and natural -language
questions are translated into SQL that filters, ag-
gregates, or ranks events based on their tags and
temporal attributes. The final answer is gener-
ated by interpreting the resulting table, allowing
this baseline to test how well symbolic querying
captures fine -grained, time -specific information in
long-duration audio.
4.4 Audio Ground Model evaluation
We evaluate the standalone AGM on long form
audio using the Sound Event Detection (SED) task
using DESED dataset(Serizel et al., 2020) from
the AudioMarathon benchmark(He et al., 2025), a
recent suite designed to assess long context audio
understanding. For each multiple choice question
(MCQ), we pass the four candidate sound event
tags to AGM and compute a score for each tag by
averaging the model’s frame level similarity scores
across the entire audio clip. The predicted answer
corresponds to the tag with the highest aggregated
score. Under this simple inference setup, AGM
achieves an F1 score of 74.8, ranking just below the
top performing Qwen2.5 Omni 7B model, which
attains 78.4 on the leaderboard ( Appendix D ).
To obtain temporal event boundaries from AGM
output, we apply a global decision threshold of 0.8
to the framewise scores and refine the resulting
predictions using a 0.3 second median filter, which
we hold constant across all experiments.
4.5 Comparison of Querying Approaches
We evaluate four approaches for querying Indus-
trial IoT audio: (1) our method: AGM + longAudio-
RAG, (2) AGM + RAG, (3) AGM + Text2SQL over
event tables, and (4) AF3 + RAG. As shown in Ta-
ble 1, our approach consistently outperforms the
others across detection, counting, and summary
tasks. By using structured AGM logs, explicit
temporal resolution, and intent-aware processing,
the system delivers more accurate detections and

more reliable event-level reasoning than retrieval-
or SQL-based pipelines.
Standard RAG struggles on detection and count-
ing because retrieved text lacks precise temporal
structure, and LLMs are unreliable at boundary-
aware operations. Text2SQL also often fails, pro-
ducing invalid queries or misinterpreting time inter-
vals, resulting in incomplete or incorrect retrieval.
To isolate the impact of high-quality event ex-
traction, we compare against an AF3-based RAG
pipeline. AF3 + RAG performs substantially worse,
reinforcing that (i) well-structured AGM event logs
are essential for downstream temporal reasoning,
and (ii) retrieval grounded in event-level represen-
tations is more effective than operating on AF3
features or transcript-level content. We addition-
ally conducted a human evaluation study using a
curated set of 113 questions, divided into three cat-
egories: 67 detection, 35 counting, and 11 summa-
rization questions. A human evaluator rated system
responses on a 1–5 scale, on which our approach
with Phi-3.5-MoE LLM, achieved an overall aver-
age score of 4.28. Further details on the human
and GPT -based evaluation setup and analysis are
provided in Appendix H.
4.6 LLM Comparison Across Model Scales
Table 2 compares LLMs from 0.5B to 14B ac-
tive parameters, showing that larger models con-
sistently perform better, especially on detection
and summary tasks that require precision and con-
textual reasoning. The strongest results come
from Phi-3.5-MoE (71.10%; 42B, 6.6B active)
andPhi-3-medium (70.88%; 14B), these models
show strong reasoning and reliable interpretation
of the generated SQL tables. Models in the 7-
8B range, such as Minitron-8B andQwen2.5-7B ,
also perform competitively with scores near 68%.
In contrast, smaller models ( ≤3B), including
Llama-3.2-3B and Qwen2.5-0.5B , show clear
degradation across tasks, indicating that adequate
model capacity is essential for structured-context
understanding.
4.7 Time resolution module analysis
To evaluate our time -resolution methods, we cre-
ated 45 QA pairs spanning difficulty (easy, medium,
hard) and semantic type (details in Appendix F).
Easy questions use explicit, unambiguous expres-
sions (e.g., “between 2pm and 4pm”), medium
questions require interpreting relative phrases and
simple arithmetic (e.g., “first 2 hours of day shift”),
and hard questions involve ambiguous or collo-quial references, boundary cases, or multi -step con-
straints (e.g., “around lunchtime”). As shown in Ta-
ble 3, the combined regex+LLM approach achieves
the highest overall accuracy (77.78%), including
perfect accuracy on easy cases and strong improve-
ments on medium and hard queries.
4.8 Deployment Considerations
The system has been deployed in real-world envi-
ronments with all components running in produc-
tion. The Audio Grounding Model (AGM) operates
on the Qualcomm IQ-9075 platform, which deliv-
ers up to 100 TOPS of on-device AI performance,
enabling low-latency audio processing while keep-
ing raw audio local for privacy. Its octa-core Kryo
Gen6 CPU, Hexagon Tensor Processors, and indus-
trial operating range support continuous edge in-
ference. The LLM and embedding services run on
NVIDIA A100 GPU servers, providing the through-
put required for large-scale embedding generation
and model inference. Other details of the software
implementation can be found in Appendix K.
4.9 Inference Latency
Table 1 reports the end-to-end average latency per
query (seconds) for all Phi-4-mini-instruct config-
urations across Home IoT (HIoT) and Industrial
IoT (IIoT) scenarios. Our proposed method is the
fastest, Text-to-SQL incurs modest overhead, RAG
is slower, and AF3 prompt-based runs remain the
most expensive.
5 Conclusion
We presented an end-to-end framework for
multi-hour audio understanding that integrates a
lightweight AGM, structured event logs, and LLM-
based reasoning. Our approach outperforms stan-
dard RAG and Text2SQL pipelines by leverag-
ing temporally aligned AGM events for accurate
detection, counting, and summarization. Experi-
ments across diverse datasets show that mid-sized
instruction-tuned and MoE models, when paired
with high-quality event representations, deliver
strong performance without relying on very large
LLMs. Overall, the results demonstrate that struc-
tured event extraction combined with efficient re-
trieval forms a reliable foundation for machine-
centric audio analytics. Future work includes ex-
panding to additional sensing modalities and intro-
ducing agentic capabilities that leverage tools such
as text-to-SQL and code generation. We also aim
for fully end-to-end on-device deployment, run-
ning both the AGM and compact LLMs on edge
hardware for low-latency inference.

References
Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkin-
son, Hany Awadalla, Nguyen Bach, Jianmin Bao,
Alon Benhaim, Martin Cai, Vishrav Chaudhary, Con-
gcong Chen, and 1 others. 2025. Phi-4-mini tech-
nical report: Compact yet powerful multimodal lan-
guage models via mixture-of-loras.arXiv preprint
arXiv:2503.01743.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Hany Awadalla et al. Arah A, Jyoti Aneja. 2024. Phi-3
technical report: A highly capable language model
locally on your phone.Preprint, arXiv:2404.14219.
Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sar-
war Uddin, and Srimat Chakradhar. 2024. Vita: An
efficient video-to-text algorithm using vlm for rag-
based video analysis system. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition, pages 2266–2274.
Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed,
and Michael Auli. 2020. wav2vec 2.0: A framework
for self-supervised learning of speech representations.
Advances in neural information processing systems,
33:12449–12460.
Gedas Bertasius, Heng Wang, and Lorenzo Torresani.
2021. Is space-time attention all you need for video
understanding? InProceedings of the 38th Inter-
national Conference on Machine Learning (ICML).
PMLR.
Joao Carreira and Andrew Zisserman. 2017. Quo vadis,
action recognition? a new model and the kinetics
dataset. Inproceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages
6299–6308.
Yifu Chen, Shengpeng Ji, Haoxiao Wang, Ziqing
Wang, Siyu Chen, Jinzheng He, Jin Xu, and Zhou
Zhao. 2025. Wavrag: Audio-integrated retrieval
augmented generation for spoken dialogue models.
arXiv preprint arXiv:2502.14727.
Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shil-
iang Zhang, Zhijie Yan, Chang Zhou, and Jingren
Zhou. 2023. Qwen-audio: Advancing universal
audio understanding via unified large-scale audio-
language models.arXiv preprint arXiv:2311.07919.
Xiang Deng, Ahmed Hassan, Christopher Meek, Olek-
sandr Polozov, Huan Sun, and Matthew Richardson.
2021. Structure-grounded pretraining for text-to-sql.
InProceedings of the 2021 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
pages 1337–1350.Haytham M Fayek and Justin Johnson. 2020. Temporal
reasoning via audio question answering.IEEE/ACM
Transactions on Audio, Speech, and Language Pro-
cessing, 28:2283–2294.
Eduardo Fonseca, Xavier Favory, Jordi Pons, Frederic
Font, and Xavier Serra. 2021. Fsd50k: an open
dataset of human-labeled sound events.IEEE/ACM
Transactions on Audio, Speech, and Language Pro-
cessing, 30:829–852.
Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman,
Aren Jansen, Wade Lawrence, R Channing Moore,
Manoj Plakal, and Marvin Ritter. 2017. Audio set:
An ontology and human-labeled dataset for audio
events. In2017 IEEE international conference on
acoustics, speech and signal processing (ICASSP),
pages 776–780. IEEE.
Sreyan Ghosh, Sonal Kumar, Ashish Seth, Chandra Ki-
ran Reddy Evuru, Utkarsh Tyagi, S Sakshi, Oriol
Nieto, Ramani Duraiswami, and Dinesh Manocha.
2024. Gama: A large audio-language model with ad-
vanced audio understanding and complex reasoning
abilities. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 6288–6313.
Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Ku-
mar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck
Yang, Ramani Duraiswami, Dinesh Manocha, Rafael
Valle, and 1 others. 2025. Audio flamingo 3: Advanc-
ing audio intelligence with fully open large audio
language models.arXiv preprint arXiv:2507.08128.
Arushi Goel, Zhifeng Kong, Rafael Valle, and Bryan
Catanzaro. 2024. Audio dialogues: Dialogues
dataset for audio and music understanding.arXiv
preprint arXiv:2404.07616.
Yuan Gong, Hongyin Luo, Alexander H Liu, Leonid
Karlinsky, and James Glass. 2024. Listen, think, and
understand. InInternational Conference on Learning
Representations.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current
landscape and future directions.arXiv preprint
arXiv:2410.12837.
Peize He, Zichen Wen, Yubo Wang, Yuxuan Wang, Xi-
aoqian Liu, Jiajie Huang, Zehui Lei, Zhuangcheng
Gu, Xiangqi Jin, Jiabing Yang, and 1 others. 2025.
Audiomarathon: A comprehensive benchmark for
long-context audio understanding and efficiency in
audio llms.arXiv preprint arXiv:2510.07293.
Zijin Hong, Zheng Yuan, Qinggang Zhang, Hao Chen,
Junnan Dong, Feiran Huang, and Xiao Huang. 2025.

Next-generation database interfaces: A survey of llm-
based text-to-sql.IEEE Transactions on Knowledge
and Data Engineering.
Rongjie Huang, Mingze Li, Dongchao Yang, Jia-
tong Shi, Xuankai Chang, Zhenhui Ye, Yuning Wu,
Zhiqing Hong, Jiawei Huang, Jinglin Liu, and 1 oth-
ers. 2024. Audiogpt: Understanding and generating
speech, music, sound, and talking head. InProceed-
ings of the AAAI Conference on Artificial Intelligence,
volume 38, pages 23802–23804.
Hangrui Hu Jin Xu, Zhifang Guo and 1 others.
2025. Qwen3-omni technical report.arXiv preprint
arXiv:2509.17765.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage re-
trieval for open-domain question answering. In
EMNLP (1), pages 6769–6781.
Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee,
and Gunhee Kim. 2019. Audiocaps: Generating cap-
tions for audios in the wild. InProceedings of the
2019 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, Volume 1 (Long and
Short Papers), pages 119–132.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin,
Haojia Lin, Chaoyou Fu, Jinfa Huang, Jiayi Ji, Fei
Chao, Jiebo Luo, and 1 others. 2024. Video-rag:
Visually-aligned retrieval-augmented long video com-
prehension.arXiv preprint arXiv:2411.13093.
Saurav Muralidharan, Sharath Turuvekere Sreenivas,
Raviraj Joshi, Marcin Chochowski, Mostofa Patwary,
Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz,
and Pavlo Molchanov. 2024. Compact language mod-
els via pruning and knowledge distillation.Advances
in Neural Information Processing Systems, 37:41076–
41102.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
arXiv preprint arXiv:1908.10084.
Romain Serizel, Nicolas Turpault, Ankit Shah, and
Justin Salamon. 2020. Sound event detection in syn-
thetic domestic environments. InICASSP 2020-2020
IEEE International Conference on Acoustics, Speech
and Signal Processing (ICASSP), pages 86–90. IEEE.
Shulong Tan, Zhixin Zhou, Zhaozhuo Xu, and Ping Li.
2019. On efficient retrieval of top similarity vectors.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9thInternational Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pages 5236–5246.
Xichen Tan, Yunfan Ye, Yuanjing Luo, Qian Wan, Fang
Liu, and Zhiping Cai. 2025. Rag-adapter: A plug-
and-play rag-enhanced framework for long video un-
derstanding.arXiv preprint arXiv:2503.08576.
Bing Wang, Changyu Ren, Jian Yang, Xinnian Liang, Ji-
aqi Bai, Linzheng Chai, Zhao Yan, Qian-Wen Zhang,
Di Yin, Xing Sun, and 1 others. 2025. Mac-sql: A
multi-agent collaborative framework for text-to-sql.
InProceedings of the 31st International Conference
on Computational Linguistics, pages 540–557.
Gijs Wijngaard, Elia Formisano, Michele Esposito, and
Michel Dumontier. 2025. Audio-language datasets
of scenes and events: A survey.IEEE Access.
Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Tay-
lor Berg-Kirkpatrick, and Shlomo Dubnov. 2023.
Large-scale contrastive language-audio pretraining
with feature fusion and keyword-to-caption augmen-
tation. InICASSP 2023-2023 IEEE International
Conference on Acoustics, Speech and Signal Process-
ing (ICASSP), pages 1–5. IEEE.
Xuenan Xu, Ziyang Ma, Mengyue Wu, and Kai Yu.
2024. Towards weakly supervised text-to-audio
grounding.IEEE Transactions on Multimedia.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
An Yang, Baosong Yang, Beichen Zhang, and et al.
2024. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Zijia Zhao, Longteng Guo, Tongtian Yue, Sihan Chen,
Shuai Shao, Xinxin Zhu, Zehuan Yuan, and Jing
Liu. 2023. Chatbridge: Bridging modalities with
large language model as a language catalyst.arXiv
preprint arXiv:2305.16103.
A Ethical Considerations
This work involves processing long-duration audio,
which may contain sensitive personal or environ-
mental information. To mitigate privacy risks, all
raw audio is processed locally on edge devices,
and only structured event metadata is transmitted
to the server, reducing the likelihood of exposing
identifiable content. Our system does not attempt
to infer personal attributes, emotions, or identities,
and all event logs used for evaluation are synthetic
or derived from publicly available datasets to avoid
handling real user data. Finally, we ensure that
model outputs are grounded in retrieved evidence
to minimize hallucinations and reduce the risk of
generating misleading or harmful responses.

B Audio classes for Home-IoT and
Industrial IoT datasets
The home set coversalarms, sirens, door_bell,
door_knock, glass_breaking, car_crash,
door_close-open, baby_cry, gun_shot, cat,
car_honk, snoring, dog_bark; the industrial set
includestools clanking, hand saw, hand file,
workers talking, footsteps, arc welder, diesel
forklift, power hand drill, stamping machine,
walkie talkie, warning buzzer, factory whistle
C Custom sound enrollment and
detection
Our on-device long audio QA framework also sup-
ports custom sound enrollment, allowing users to
upload example clips of sound events of interest
and query for them in continuous audio. To enable
this capability, we train a prototypical network on
FSD50K (Wu et al., 2023) using CLAP (Fonseca
et al., 2021) audio embeddings, formulated as an
N way M shot classification task where the model
learns to discriminate among N classes given M
support samples. To preserve the CLAP audio en-
coder, we freeze all of its parameters and learn a
single 512 dimensional linear projection layer for
custom sound adaptation. Training uses negative
log-likelihood loss with softmax scoring, and clas-
sification is based on cosine similarity normalized
by a temperature scaled LogSoftmax. At inference
time, we apply a 5 second sliding window with
4 second overlap, yielding a per second score for
each enrolled sound; a global threshold of 0.5 is
then applied to these framewise scores to determine
temporal event boundaries.
D AGM evaluation
Table 4 reports Sound Event Detection (SED) per-
formance on the AudioMarathon benchmark , com-
paring leading long -form audio understanding mod-
els. Our AGM model, evaluated on the DESED
dataset, achieves a strong F1 score of74.8, ranking
just below the top Qwen2.5-Omni-7B system.
E Automatic QA pair generation
E.1 Simple-QA pair generation
The development of robust evaluation frameworks
for long audio based chatbot systems necessitates
the creation of comprehensive question-answer
(QA) datasets that can systematically assess modelperformance across diverse query types and tempo-
ral expressions. In this work, we present a simple
yet effective QA generation methodology that em-
ploys a deterministic approach to create evaluation
benchmarks for long-form audio understanding sys-
tems. Our framework generates three primary cat-
egories of queries: detection queries (binary pres-
ence/absence verification), counting queries (occur-
rence counting), and summary queries (temporal
and statistical aggregation). To ensure temporal
consistency and reproducibility, we adopt a stan-
dardized HH:MM:SS time format for all temporal
expressions, eliminating ambiguities inherent in
natural language time references such as "morn-
ing shift" or "12-hour format" variations. The
generation process leverages ground-truth audio
event annotations obtained using Audio Ground-
ing Model (AGM) to construct queries with verifi-
able answers, incorporating both positive samples
(events present in the audio) and negative samples
(absent events and unrelated event categories) to
evaluate false positive rates. Furthermore, we in-
troduce synonym-based query variations to assess
the semantic understanding capabilities of the tar-
get system, replacing canonical event labels with
contextually equivalent terms. This simple gen-
eration approach produces balanced datasets with
controlled distributions across query categories, en-
abling systematic evaluation of audio QA systems
while maintaining interpretability and reproducibil-
ity, critical requirements for benchmarking in the
audio understanding domain.
E.2 Complex-QA pair generation
While simple-QA generation provides a founda-
tional evaluation framework, real-world audio un-
derstanding systems must handle the inherent com-
plexity and variability of natural language temporal
expressions. To address this challenge, we pro-
pose a complex-QA generation methodology that
systematically incorporates diverse temporal refer-
ence formats to evaluate the robustness of audio
event detection systems under realistic query condi-
tions. Our approach extends beyond standardized
HH:MM:SS formats to encompass seven distinct
temporal expression types: (1) standard 24-hour
format (e.g., "between 08:00 and 16:00"), (2) 12-
hour format with meridiem indicators (e.g., "from
2:30 pm to 4:00 pm"), (3) shift-based references
aligned with industrial or operational contexts (e.g.,
"during the morning shift"), (4) before/after tem-
poral boundaries (e.g., "after 17:30"), (5) duration-

Table 4: Sound Event Detection (SED) Results on AudioMarathon Benchmark. The entry marks with ’*’ is our
result. The remaining entries are cited from (He et al., 2025)
.Rank Model SED (F1)
1 Qwen2.5-Omni-7B 78.4
2 AGM (Ours)*74.8
3 V oxtral-Mini-3B-2507 71.0
4 Qwen2.5-Omni-3B 70.2
5 Gemma-3n-E2B-it 50.2
6 Gemma-3n-E4B-it 50.2
7 Audio-Flamingo-3 59.5
8 Phi-4-Multimodal 55.1
9 Aero-1-Audio 55.0
10 Baichuan-Omni-1.5 45.7
11 Audio-Flamingo-2 27.1
based expressions relative to shifts or day bound-
aries (e.g., "first 2 hours of day shift"), (6) half-
based temporal divisions (e.g., "second half of after-
noon shift"), and (7) relative duration expressions
(e.g., "between 1 hour and 2 hours"). By system-
atically varying temporal expression complexity,
our QA generation facilitates the identification of
model weaknesses in time parsing, semantic in-
terpretation, and cross-format generalization, criti-
cal capabilities for deploying audio understanding
systems in production environments where users
employ diverse and unpredictable temporal refer-
ences.
E.3 Implementation Details and Dataset
Characteristics
Multi-Domain Support:The framework supports
two configurable IoT domains: Industrial IoT (12
event classes: arc welder, stamping machine, diesel
forklift, etc.) and Home IoT (13 event classes:
baby cry, doorbell, glass breaking, etc.), each with
domain-specific event taxonomies and synonym
mappings to ensure contextual relevance.
Dataset Scale:The generation process produces
100 QA pairs per section across five generation
phases: (1) detection & frequency with original
event labels, (2) detection & frequency with syn-
onym variations, (3) detection & frequency with un-
related events (negative samples), (4) specific event
summaries, and (5) generic summaries. Since de-
tection and frequency queries are generated in pairs
for the first three sections, this yields approximately
800 total QA pairs(600 detection/frequency + 200
summary) with balanced positive/negative distribu-
tions and a minimum 10-second temporal window.
Temporal Expression Distribution:Incomplex-QA mode, temporal expressions are
sampled using a probability-based strategy:
5% full-day queries, 15% shift-based (when
enabled), 10% before/after boundaries, 10%
duration-based, 5% half-based, 20% 12-hour
format, and the remainder as standard 24-hour
format, ensuring comprehensive coverage across
temporal expression types.
Shift Definitions:For industrial contexts, stan-
dardized shifts are defined as: Morning/Day
Shift (08:00–16:00), Afternoon/Evening Shift
(16:00–24:00), and Night Shift (00:00–08:00).
Shift-based queries can be toggled via a config-
uration flag, enabling evaluation with or without
shift-specific temporal reasoning.
Ground Truth Structure:Each QA pair
includes structured JSON metadata containing
query category/subcategory, temporal bounds
(HH:MM:SS), event tags, time expression type,
and relevant statistics (detection status, occurrence
counts, first/last event timestamps), enabling auto-
mated evaluation and error analysis.
Validation Mechanisms:The framework im-
plements safeguards including configuration vali-
dation with automatic fallbacks, maximum attempt
limits (10× target count) to prevent infinite loops,
minimum duration enforcement (10 seconds), and
automatic mode compatibility checks.
F Time resolution
The detailed category-level results in Appendix
Table 5 highlight the specific categories that influ-
ence model performance. Categories with well-
structured or explicitly defined temporal expres-
sions such asexplicit time ranges,shift-based cues,

andfull-day implicit referencesshow consistently
strong performance, particularly when using the
combined approach. In contrast, categories involv-
ing greater variability or ambiguity, includingrel-
ative durations,typos and variations, andedge
cases, yield lower accuracy across all methods,
reflecting their inherent complexity. This break-
down provides granular insight into error patterns
and complements the difficulty-based analysis pre-
sented in the main paper. Detailed flowchart and
prompt are shown in Figure 3 and 4.
G Query rephrasing module
The detailed flowchart and the prompt used for
query rephrasing module are shown in Figure 5
and 6.
H Dialogue evaluation using humans and
GPT-4o as judge
We evaluated a dialogue system on 113 single -turn
QA pairs across three task categories detection,
counting, and summarization using both a human
evaluator and a GPT-4o based evaluator (Achiam
et al., 2023). For every response, each judge as-
signed 1–5 scores for language quality, correctness,
reasoning coherence, and user experience; we av-
eraged these four scores to obtain a per -item com-
posite and then averaged those composites within
each category to produce the final category scores.
Human evaluation yielded category means of 4.36,
4.25, and 3.95 (detection, counting, summariza-
tion), while GPT-4o based evaluation yielded 4.46,
3.67, and 3.47 for the same categories. The Pear-
son correlation between human and GPT per -item
composites across all 114 examples was 0.57, indi-
cating moderate alignment. Despite comparatively
lower outcomes in counting and summarization,
performance is likely to improve with model scal-
ing or targeted fine -tuning within the size–accuracy
budget. Overall, the system delivers strong con-
versational performance and coherent execution
across categories
I Anomaly Detection Extension
Our framework naturally supports anomaly detec-
tion by treating anomalies as structured event-level
signals that can be retrieved, analyzed, and ex-
plained using the same time-resolution and intent-
routing components. When users ask anomaly-
related questions, the system first classifies the
query subtype (e.g., loudness anomaly, pitchanomaly, or abnormal start-time), retrieves the rele-
vant events, and computes anomalies using a plug-
in detector. In our implementation, we evaluate two
forms of attribute-based anomalies such as loud-
ness deviations from the normal range and tem-
poral anomalies, where event start times fall out-
side expected intervals derived from historical clus-
ters. The system produces a structured anomaly ta-
ble and, when needed, incorporates guidance from
user-provided instruction manuals before prompt-
ing the LLM to generate a grounded explanation.
This allows the method to flexibly support any
anomaly detection backend while maintaining faith-
ful, event-grounded audio question answer capabil-
ities.
We have implemented anomaly detection based
on the loudness attribute and also irregular sound
patterns. Figures 10 and 8 show an example with
the flowchart analysis pipeline that can be inte-
grated into our proposed system. Figure 9 shows
how we can integrate user manuals and other cus-
tom logs into our system for more personalized
responses to target the specific use-case.
J Limitations
Our system relies on the accuracy of the AGM
module, so missed or noisy detections can propa-
gate to downstream reasoning. The time-resolution
component may also struggle with rare or highly
ambiguous temporal expressions. Although our
synthetic benchmark enables controlled evaluation,
it does not fully capture the diversity of real-world
acoustic environments.
K Software Implementation
The system is implemented as a modular, service-
oriented stack comprising an edge AGM inference
service, a backend indexing and QA service, and
a lightweight Streamlit-based frontend. The AGM
service runs a compact PyTorch model via FastAPI,
producing JSON event logs with temporal meta-
data and supporting custom sound enrollment by
allowing users to register new classes from example
recordings. Its low computational footprint enables
deployment on resource-constrained edge devices,
reducing the need to transmit raw audio. The back-
end ingests these logs, stores them in a database,
and builds retrieval structures for downstream rea-
soning; it initializes an embedding model for se-
mantic indexing and an LLM for answer generation,
orchestrated through LlamaIndex and served effi-

Table 5: Category-wise accuracy results for the time resolution module.
Category Regex-only LLM-only Combined
explicit_time_ranges 100.00 100.00 100.00
shift_based 100.00 100.00 100.00
relative_durations 60.00 60.00 100.00
before_after 100.00 50.00 100.00
half_periods 100.00 50.00 100.00
full_day_implicit 100.00 100.00 100.00
typos_and_variations 27.27 72.73 72.73
edge_cases 20.00 20.00 30.00
ciently using vLLM on GPU machines. This back-
end also maintains conversational state for multi-
turn queries. The frontend provides a simple in-
terface for uploading audio, enrolling new sound
classes, and issuing natural-language queries, com-
municating with the backend and inference service
through RESTful APIs to allow each component to
scale independently.
L User Interface for LongAudio-RAG
We have implemented a web application front-end
which allows the users to start recording the audio
or upload an existing audio file and then chat over
its contents. The UI also provides the option to
enroll custom audio sounds that will be supported
by the system. The screenshot of the UI is shown
in Figure 11
M Prompts used for intent based
response generation
The prompts used for the response generation are
shown in Figures 12 and 13.

Figure 3: Time resolution module

Figure 4: Prompt used in time resolution module
Figure 5: Query Rephrasing Module

Figure 6: Prompt used in Query Rephrasing Module
Figure 7: Loudness anomaly detection

Figure 8: Frequency based anomaly detection
Figure 9: Anomaly detection processing for long audio QA

API
Serverendpoints
Web Page
AGM
Server
Server
Recorder
Wrapper
AGM LibraryAGM
controlendpointsbackend
AGM
controlREDIS
eventsQuery
Clear DB
eventsQuery
responsesFigure 10: Software Implementation Architecture
Figure 11: User Interface the long audio RAG
Figure 12: Prompt for Summary category

Figure 13: Prompt for detection/counting based questions