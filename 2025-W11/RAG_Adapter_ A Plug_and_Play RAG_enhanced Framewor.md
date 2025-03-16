# RAG-Adapter: A Plug-and-Play RAG-enhanced Framework for Long Video Understanding

**Authors**: Xichen Tan, Yunfan Ye, Yuanjing Luo, Qian Wan, Fang Liu, Zhiping Cai

**Published**: 2025-03-11 16:10:43

**PDF URL**: [http://arxiv.org/pdf/2503.08576v1](http://arxiv.org/pdf/2503.08576v1)

## Abstract
Multi-modal Large Language Models (MLLMs) capable of video understanding are
advancing rapidly. To effectively assess their video comprehension
capabilities, long video understanding benchmarks, such as Video-MME and MLVU,
are proposed. However, these benchmarks directly use uniform frame sampling for
testing, which results in significant information loss and affects the accuracy
of the evaluations in reflecting the true abilities of MLLMs. To address this,
we propose RAG-Adapter, a plug-and-play framework that reduces information loss
during testing by sampling frames most relevant to the given question.
Additionally, we introduce a Grouped-supervised Contrastive Learning (GCL)
method to further enhance sampling effectiveness of RAG-Adapter through
fine-tuning on our constructed MMAT dataset. Finally, we test numerous baseline
MLLMs on various video understanding benchmarks, finding that RAG-Adapter
sampling consistently outperforms uniform sampling (e.g., Accuracy of GPT-4o
increases by 9.3 percent on Video-MME), providing a more accurate testing
method for long video benchmarks.

## Full Text


<!-- PDF content starts -->

RAG-Adapter: A Plug-and-Play RAG-enhanced Framework
for Long Video Understanding
Xichen Tan
College of Computer Science and Technology,
National University of Defense Technology
Changsha, China
tanxc23@nudt.edu.cnYunfan Ye
School of Design,
Hunan University
Changsha, China
Yuanjing Luo
College of Computer and Mathematics,
Central South University of Forestry and Technology
Changsha, China
Qian Wan
Faculty of Artificial Intelligence in Education,
Central China Normal University
Wuhan, ChinaFang Liu
School of Design,
Hunan University
Changsha, China
Zhiping Cai
College of Computer Science and Technology,
National University of Defense Technology
Changsha, China
Abstract
Multi-modal Large Language Models (MLLMs) capable
of video understanding are advancing rapidly. To effec-
tively assess their video comprehension capabilities, long
video understanding benchmarks, such as Video-MME and
MLVU, are proposed. However, these benchmarks directly
use uniform frame sampling for testing, which results in
significant information loss and affects the accuracy of
the evaluations in reflecting the true abilities of MLLMs.
To address this, we propose RAG-Adapter , a plug-and-
play framework that reduces information loss during test-
ing by sampling frames most relevant to the given ques-
tion. Additionally, we introduce a Grouped-supervised
Contrastive Learning (GCL) method to further enhance
RAG-Adapter’s sampling effectiveness through fine-tuning
on our constructed MMAT dataset. Finally, we test nu-
merous baseline MLLMs on various video understanding
benchmarks, finding that RAG-Adapter sampling consis-
tently outperforms uniform sampling (e.g., GPT-4o’s accu-racy increases by 9.3% on Video-MME), providing a more
accurate testing method for long video benchmarks.
1. Introduction
In the field of video understanding, research on short
videos progresses earlier and more extensively than on
long videos, primarily due to the quadratic complexity con-
straint of transformer-based models in handling long se-
quences. To mitigate this, many long video models, such
as MovieChat [35] and LlamaVid [22], introduce input to-
ken compression algorithms to reduce computational costs.
To evaluate the long video understanding capabili-
ties of MLLMs, several specialized long video bench-
marks have been proposed, including Video-MME [12] and
MLVU [44]. However, these benchmarks do not standard-
ize the number of input frames during testing due to varia-
tions in models’ maximum frame capacities. Moreover, not
all MLLMs support one-frame-per-second sampling (as-
1arXiv:2503.08576v1  [cs.CV]  11 Mar 2025

MLLM Decoder
Projection
Image Encoder
Uniformly Sampled 
Video FramesQuestion
MLLM Decoder
Projection
Image Encoder
RAG-Adapter Sampled 
Video Frames
QuestionRAG-Adapter
(a) MLLM Architecture (b) MLLM + RAG-AdapterFigure 1. (a) and (b) show a comparison between scenarios with
and without the RAG-Adapter framework, respectively.
sumed sufficient to capture content). For these models, test-
ing relies on uniformly sampled frame subsets. In Video-
MME, for instance, the longest test video spans one hour,
yet only four uniformly sampled frames are used at mini-
mum, often omitting critical information. This leads to re-
sponses resembling random guesses and makes it challeng-
ing to accurately evaluate true model performance.
To address the testing challenges in existing long video
benchmarks, we propose a plug-and-play RAG-enhanced
(Retrieval Augmented Generation) optimization frame-
work, RAG-Adapter . As illustrated in Figure 1, RAG-
Adapter operates without modifying the internal architec-
ture of MLLMs, instead focusing on video frame input. By
retrieving the Top Kmost relevant video frames, it replaces
the uniform sampling method, significantly reducing infor-
mation loss. This straightforward yet effective approach
more accurately evaluates the true long video understand-
ing capabilities of MLLMs.
Although the approach is straightforward, research di-
rectly integrating RAG with MLLMs is limited. This is
mainly because RAG-Adapter’s retrieval performance de-
pends heavily on similarity matching between embeddings
generated by its text and image encoders ( Figure 2). The
embeddings produced by open-source encoders may be sub-
optimal for long video understanding tasks. Therefore, we
fine-tune these encoders through contrastive learning to bet-
ter align similar embeddings, thereby enhancing the re-
trieval effectiveness of RAG-Adapter.
Given the challenge of directly locating relevant frames
in long videos, we further construct a fine-tuning dataset,
MMAT , using short video understanding benchmarks. We
extract video frames and pair them with corresponding
questions to create positive pairs for fine-tuning.
Additionally, as a single video may correspond to mul-
tiple questions, the Self-supervised Contrastive Learning
(SCL) assumption that treats other questions as negative
samples may mislead the model during training. To ad-
dress this, we propose Grouped-supervised Contrastive
Learning (GCL) , where all positive pairs involving the
same video’s frame share a common group label. GCL en-
ables clearer differentiation between intra-group and inter-
group embeddings, thereby enhancing RAG-Adapter’s re-
trieval capabilities for video understanding tasks.Using retrieval results from RAG-Adapter fine-tuned
with GCL (RAG-Adapter, unless specified otherwise, is
GCL fine-tuned), we introduce two metrics: Average Sim-
ilarity Score (ASS) andNecessary Information Frame
(NIF) . ASS measures the average similarity between the
TopKframes retrieved by RAG-Adapter and the corre-
sponding questions, while NIF represents the average min-
imum number of frames containing essential information
needed to answer each question. The NIF reveals that, even
for long video understanding benchmarks, a small subset of
frames typically contains the required information, validat-
ing our approach of using a fixed number of frames (Top K)
across models for fair evaluation.
Notably, the ASS and NIF metrics offered by RAG-
Adapter serve as important indicators for evaluating bench-
mark quality. A lower ASS may indicate insufficient rele-
vance between video content and questions, suggesting po-
tential flaws in question formulation, while a lower NIF im-
plies that fewer frames are needed, indicating lower ques-
tion complexity.
In summary, the main contributions of this work are:
1) We propose RAG-Adapter , a plug-and-play enhance-
ment framework for MLLMs. By supplying input-level
video frames relevant to test questions, RAG-Adapter en-
hances the video understanding capabilities of MLLMs
without structural modifications.
2) We construct the MMAT fine-tuning dataset and pro-
pose Grouped-supervised Contrastive Learning (GCL)
for long video understanding scenarios, enhancing RAG-
Adapter’s retrieval performance.
3) We introduce two metrics through RAG-Adapter: Av-
erage Similarity Score (ASS) andNecessary Information
Frame (NIF) , as standards for evaluating benchmark qual-
ity and complexity in long video understanding. NIF fur-
ther confirms that RAG-Adapter provides information that
is both sufficient and effective.
4) Extensive experiments on open-source long video un-
derstanding benchmarks demonstrate the effectiveness of
RAG-Adapter in enhancing the video understanding capa-
bilities of existing MLLMs.
2. Related Work
2.1. Multi-model LLMs (MLLMs)
MLLMs extend traditional LLMs by incorporating a visual
encoder and projection layer, enabling image and video un-
derstanding. Video-based MLLMs [2, 5, 14, 25, 33, 38, 40],
process sampled video frames as input, is essentially equiv-
alent to image-based MLLMs [6, 10, 13, 19, 23] that sup-
port multiple images, even if not explicitly trained on video
data. To handle more frames for long video understanding,
many MLLMs reduce computational complexity by com-
pressing the number of visual tokens at the input level.
2

MovieChat [35] applies ToMe [7] methods to merge sim-
ilar tokens between adjacent frames. LLaMa-VID [22] re-
duces image tokens through average pooling, while Chat-
UniVi [17] uses the k-nearest-neighbor based density peaks
clustering algorithm (DPC-KNN) to segment videos into
events, and group tokens of each frame within these events.
Although these models can support inputs of up to thou-
sands of video frames, the NIF metric in Table 1 indicates
that the relevant information needed to answer questions re-
sides in only a small subset of frames. Furthermore, abla-
tion experiments in Table 6 show that using more uniformly
sampled frames can yield inferior performance compared to
using only frames directly relevant to the questions.
2.2. Long Video Understanding Benchmarks
To evaluate MLLMs’s long video understanding capabil-
ities, several benchmarks have been proposed, including
Video-MME [12], and MLVU [44]. These benchmarks con-
tain numerous manually annotated Q&A pairs, with aver-
age video lengths exceeding 10 minutes. The video content
covers a wide range of domains, spanning domains such as
daily life, art, sports, and television. They comprehensively
assess MLLMs’s abilities in cognition, reasoning, summa-
rization, and other aspects of long video comprehension.
Although these benchmarks provide a comprehensive
evaluation of different aspects, during the testing phase, a
uniform sampling of video frames is used for all questions.
Clearly, the information required for each question varies,
and there is a high likelihood that the relevant informa-
tion may not be included in the uniformly sampled frames.
Therefore, assessing the long video understanding capabil-
ities of MLLMs in this manner is not entirely reasonable.
2.3. Retrieval Augmented Generation (RAG)
RAG [18] was first introduced in NLP for retrieval aug-
mentation, and rapidly inspired advancements in text re-
trieval, with optimizations targeting various stages of the
RAG framework to enhance retrieval performance. For in-
stance, SPLADE [11] expands query with semantically sim-
ilar terms, Self-RAG [4] performs self-correction on re-
trievals, RAT [37] combines RAG with chain-of-thought
reasoning, and LoRAG [36] improves text generation qual-
ity via iterative looping. Toolformer [32] enables LLMs
to call different tool APIs, allowing information gathering
from diverse sources.
In the multi-modal domain, integrating RAG with LLMs
remains relatively underexplored. FairRAG [34] uses RAG
to promote fairness and diversity in image generation, and
RAR [24] leverages RAG to assist in image classification
and object detection. In the video domain, to our knowl-
edge, only iRAG [3] uses RAG by encoding video informa-
tion into contextual natural language descriptions, enabling
LLMs to interpret video content.These observations indicate that RAG’s application in
the video domain remains very limited. RAG-Adapter is
the first to directly integrate RAG with MLLMs, enhancing
long video understanding at the input frame level.
3. Method
3.1. RAG-Adapter Pipeline
RAG-Adapter is a simple yet effective plugin to enhance
MLLMs’ video understanding, with its main pipeline de-
tailed in Figure 2.
Video Preprocessing. For the test videos, frames are
sampled at one frame per second, forming {fi}N
i=1. Each
frame is then encoded into image embeddings {zfi}N
i=1
using the image encoder CLIP-L/14 [30]. As CLIP-L/14
primarily captures global features, which may miss fine-
grained details like objects and actions, we also employ the
open-source model CogVLM2 [15] to generate captions for
each frame, resulting in the set {ci}N
i=1. These captions are
encoded into text embeddings {zci}N
i=1using the text en-
coder BGE-M3 [9], accommodating CLIP’s text length lim-
itations. Here, fi,zfi,ci, and zcirepresent the ithframe,
its embedding, the caption, and its embedding, respectively.
Finally, {zfi}N
i=1and{zci}N
i=1are stored in the FramesDB
and CaptionsDB databases for retrieval.
Video Frames Retrieval. To address the dimensional dis-
crepancy between the text and image encoder embeddings
and avoid the added complexity and potential performance
issues of aligning these spaces, we employ a separate re-
trieval strategy. When a user submits a question, we encode
it using both the text and image encoders and independently
match it against the FramesDB and CaptionsDB, retrieving
the Top Mvideo frames {fi, sfi}M
i=1and Top Ncaptions
{ci, sci}N
i=1from each respective databases, where sfiand
scirepresent the similarity scores of the query with each
retrieced frame and caption, respectively.
To effectively integrate the retrieval results from both
databases, we introduce the Dual Reranker module, com-
prising two main steps:
1) We sum the similarity scores of the Top Mframes
and Top Ncaptions (noting that some captions may cor-
respond to frames outside the Top Mset), ranking them
by these summed scores to obtain the Top Xframes, their
corresponding captions and scores, where X is determined
jointly by M and N. The set {fX
i, cX
i, sX
i}X
i=1represents the
ithframe, its caption and summed score, respectively.
2) We find that frames ranked closely within the Top X
often exhibit high similarity, reducing diversity. To main-
tain relevance while enhancing diversity, we apply the Max-
imal Marginal Relevance (MMR) algorithm [8], commonly
used in recommendation systems. We begin with an ini-
tially selected set S=∅and an unselected set U=
3

Question: How many hats does 
the hamster change in this video?
Frames Captions
FramesDB
 CaptionsDBRaw Video
Question Embedding (I) Question Embedding (T)
Retrival RetrivalTopM
 TopN
TopK Relevant Frames
RAG -Adapter
Answer: (6)   
Figure 2. The RAG-Adapter pipeline framework. Given a video and a question, the video frames and corresponding captions are encoded
separately using image and text encoders and stored in databases. The question is encoded and retrieved using the same encoders. The Dual
Reranker module selects the Top Kframes relevant to the question. Details are provided in Section 3.1. To improve retrieval performance,
both encoders are fine-tuned using Grouped-supervised Contrastive Learning (GCL), as described in Section 3.2.
{fX
i, cX
i, sX
i}X
i=1. First, we add the frame with the high-
est summed score from UtoS. For each remaining frame in
U, the one with the highest Marginal Relevance (MR) score,
i⋆= arg max i∈UMRi, is then moved to S. This step is re-
peated K−1times, producing Kframes in S, representing
TopKrelevant frames selected by RAG-Adapter. The MRi
formula is as follow:
MRi=θ·sX
i−(1−θ)·max
j∈S[sim(fX
i, fX
j)+sim(cX
i, cX
j)]
(1)
θis a penalty coefficient to balance the weights of the
summed similarity score and diversity score, with sim()
computed via cosine similarity.
3.2. RAG-Adapter Fine-tuning
The text and image encoders in RAG-Adapter, BGE-M3
and CLIP-L/14, are trained on large-scale internet-based
corpora. However, their embedding spaces may not be fully
optimized for video understanding scenarios. To enhance
RAG-Adapter’s performance in this domain, we construct
a specialized dataset, MMAT , consisting of (Qi, Fi)and
(Qi, Ci)positive pairs for contrastive learning fine-tuning
of CLIP-L/14 and BGE-M3, respectively. Here, Qi,Fi, and
Cidenote the question, representative video frame, and cor-
responding caption for the ithvideo.
MMAT Construction. We employ a contrastive learning-
based fine-tuning method to better fit BGE-M3 and CLIP-
L/14’s embedding spaces to the requirements of video un-
derstanding benchmarks. Given the challenge of identifying
relevant frames in long videos, we start with widely used
short video understanding benchmarks, including MSVD-
QA [39], MSRVTT-QA [39], ActivityNet-QA [43], and
TGIF-QA [16], to construct the MMAT. To fully use the
available videos, the training and validation sets from these
question1:
what is sniffing a bunny?
question2:
what tries to subdue a 
bunny?
question3:
who is a kitten playing 
with?
question1:
who is coating a pork 
chop in flour?
question2:
who covers a piece of 
meat with flour?
question3:
what is a woman coating 
pieces of with flour?Group1:
Group2:
Positive Pair Negative PairFigure 3. Illustration of Grouped-supervised Contrastive Learning
(GCL) constructing positive and negative pairs.
benchmarks are combined to form the MMAT training set,
while their test sets create the MMAT test set.
Since the videos in these benchmarks are typically short
(usually under 10 seconds) with relatively consistent vi-
sual content, we sample frames at one frame per second
and select three representative frames from quartile posi-
tions within each video. For each question related to the
video, one of these frames is randomly chosen to construct
the(Qi, Fi)pairs. For each Fi, we use CogVLM2 to gener-
ate captions with detailed descriptions, thereby forming the
corresponding (Qi, Ci)pairs.
To ensure sampled frames align with questions despite
potential content inconsistencies, we use a script to auto-
4

matically exclude videos over 300 seconds and manually
filter out those with visibly inconsistent visuals.
We also observe occasional garbled text from CogVLM2
when generating captions for frames with repetitive charac-
ters. To address this, we use a script to detect and either
regenerate or manually correct such captions, followed by a
quick review to ensure semantic consistency with the video
frames. These measures ensure the quality of MMAT, re-
sulting in 417,993(Qi, Fi)and(Qi, Ci)pairs in the train-
ing set and 109,799 pairs in the test set.
Grouped-supervised Contrastive Learning (GCL). In
contrastive learning, a self-supervised loss is typically used,
where only pairs like (Qi, Fi)and(Qi, Ci)are treated as
positive samples, while all pairs (Qi, Fj)and(Qi, Cj){i̸=
j}are treated as negative samples by default.
However, in video understanding scenarios, a sin-
gle video may correspond to multiple questions. In
Self-supervised Contrastive Learning, pairs like (Qi, Fj),
(Qi, Cj){i̸=j}, which should be positive, may instead
be treated as negative samples, disrupting training. As for
Fully-supervised Contrastive Learning, label information is
incorporated, but it requires the manual construction of neg-
ative pairs.
To address this, we propose the Grouped-supervised
Contrastive Learning (GCL) , designed specifically for
video understanding scenarios. In GCL, pairs from the
same video are assigned a common group label. Within
each group “G”, all possible pairs, such as (QG
i, FG
j)and
(QG
i, CG
k), are treated as positive, while pairs from different
groups “G” and “ G′”, such as (QG
i, FG′
j)and(QG
i, CG′
k),
are treated as negative. GCL iterates through all combi-
nations, computes loss values for each, and then averages
them. Figure 3 presents an illustration of GCL, with the
loss function shown as follows:
Li=−1
|P(i)|X
p∈P(i)logexp( sim(Qi, Fp/Cp)/τ)P
a∈A′(i)exp( sim(Qi, Fa/Ca)/τ)
(2)
LGCL =1
|I|X
i∈ILi (3)
Here, Irepresents the set of all test questions, Lidenotes
the loss function for Qi, andP(i)is the set of positive sam-
ples associated with Qiwithin the same group. The nota-
tionA′(i) =A(i)\{p′∈P(i)\{p}}refers to the set of all
batch samples ( A(i)), excluding all positive samples except
the selected one, FporQp.τis the temperature coefficient.
This approach offers two main advantages: (1) it elim-
inates the need for manually constructing numerous nega-
tive pairs, and (2) by excludes all positive samples but the
selected one FporCpfrom Li’s denominator, the model
better captures detailed relationships between the questionTable 1. ASS and NIF metrics for evaluation benchmarks.
BenchmarksASS (0-2)NIFTop10 Top30 Top50
Video-MME 0.81 0.69 0.61 2.4
MLVU 0.76 0.61 0.52 2.9
Perception Test 0.74 0.69 0.68 1.9
EgoSchema 0.82 0.67 0.58 2.1
and each specific positive sample, facilitating a more refined
understanding of each sample’s unique characteristics.
4. Experiments and Analysis
4.1. Evaluation Benchmarks
We select two commonly used long video understanding
benchmarks, Video-MME and MLVU, as well as two rel-
atively shorter benchmarks focusing on human interaction
and perception in real-world scenarios: Perception Test [29]
and EgoSchema [26], for evaluation. This selection aims
to demonstrate RAG-Adapter’s generalization performance
across benchmarks with varying temporal spans (ranging
from approximately 0.5 minutes to several hours) and con-
texts. Due to the time-consuming process of generating cap-
tions for each video (as discussed in Table 5), we sample 90
videos from each benchmark to manage this. Video-MME
videos are categorized by length (short, medium, long) and
further divided into six content domains, from which we
randomly select five videos per domain. MLVU videos are
classified into nine types, from which ten videos are ran-
domly sampled per category. For Perception Test, the 90
longest videos from the Multiple-Choice Video Q&A task
are selected. In Egoschema, where all videos are 3 minutes
long, 90 videos are randomly sampled.
4.2. Statistics of ASS and NIF
Using the RAG-Adapter, we calculate the ASS and NIF
metrics for all evaluation benchmarks.
For ASS, we measure the average summed score,
ASS =1
nPK
i=1si(on a 0-2 scale), between all questions
and their Top Krelevant frames with captions, where Kset
to 10, 30, and 50. For NIF, we manually identify the mini-
mum number of frames containing essential visual informa-
tion needed to answer each question (note: for Video-MME,
some information is also found in subtitle files). The NIF
value is the average of these frame counts across all ques-
tions. Results are summarized in Table 1.
The ASS values for the four benchmarks are similar,
as the top Kframes retrieved by the RAG-Adapter in each
benchmark show little variation in relevance to the ques-
tions. Additionally, as K increases, the overall relevance
tends to decrease. MLVU has the highest NIF value due to
the greater frame requirement for Action Order and Video
Summarization tasks. Both MLVU and Video-MME show
5

Table 2. The test results for various MLLMs on Video-MME include accuracy metrics for 6 domains, along with the overall Avg. Acc.
(Average Accuracy). The highest accuracy is bolded , and the second highest is underlined .
Models Sampling MethodCategoryAvg. Acc. (%)Knowledge Film & Television Sports Competition Artistic Performance Life Record Multilingual
Image MLLMs
Otter-I [19]Uniform 28.9 28.9 33.3 33.3 24.4 33.3 30.4
RAG-Adapter 37.8 ( +8.9 ) 37.8 ( +8.9 ) 42.2 ( +8.9 ) 40.0 ( +6.7 ) 28.9 ( +4.5 ) 31.1 ( -2.2) 36.3 ( +5.9 )
LLaV A-1.6 [23]Uniform 33.3 22.2 33.3 44.4 26.7 24.4 30.7
RAG-Adapter 35.6 ( +2.3 ) 28.9 ( +6.7 ) 37.8 ( +4.5 ) 48.9 ( +4.5 ) 31.1 ( +4.4 ) 31.1 ( +6.7 ) 35.6 ( +4.9 )
GPT4-Turbo [1]Uniform 60.0 71.1 48.9 57.8 53.3 55.6 57.8
RAG-Adapter 71.1 (+11.1 ) 71.1 (+0.0 ) 51.1 ( +2.2 ) 60.0 ( +2.2 ) 53.3 ( +0.0 ) 60.0 (+4.4 ) 61.1 ( +3.3 )
Video MLLMs
Otter-V [19]Uniform 31.1 28.9 33.3 22.2 31.1 26.7 28.9
RAG-Adapter 37.8 ( +6.7 ) 31.1 ( +2.2 ) 35.6 ( +2.3 ) 28.9 ( +6.7 ) 37.8 ( +6.7 ) 31.1 ( +4.4 ) 33.7 ( +4.8 )
mPlug-Owl-V [41]Uniform 22.2 31.1 24.4 28.9 17.8 24.4 24.8
RAG-Adapter 35.6 ( +13.4 ) 37.8 ( +6.7 ) 28.9 ( +4.5 ) 31.1 ( +2.2 ) 26.7 ( +8.9 ) 28.9 ( +4.5 ) 31.5 ( +6.7 )
MovieChat [35]Uniform 24.4 28.9 22.2 31.1 22.2 28.9 26.3
RAG-Adapter 33.3 ( +8.9 ) 35.6 ( +6.7 ) 33.3 ( +11.1 ) 31.1 ( +0.0 ) 28.9 ( +6.7 ) 35.6 ( +6.7 ) 33.0 ( +6.7 )
VideoChat [20]Uniform 26.7 28.9 33.3 26.7 37.8 22.2 29.3
RAG-Adapter 31.1 ( +4.4 ) 35.6 ( +6.7 ) 33.3 ( +0.0 ) 33.3 ( +6.6 ) 40.0 ( +2.2 )33.3 ( +11.1 ) 34.4 ( +5.1 )
VideoChat2 [21]Uniform 33.3 17.8 24.4 35.6 44.4 28.9 30.7
RAG-Adapter 42.2 ( +8.9 ) 22.2 ( +4.4 ) 26.7 ( +2.3 ) 37.8 ( +2.2 ) 42.2 ( -2.2) 31.1 ( +2.2 ) 33.7 ( +3.0 )
LLaMA-VID [22]Uniform 31.1 17.8 24.4 37.8 22.2 26.7 26.7
RAG-Adapter 31.1 ( +0.0 ) 26.7 ( +8.9 ) 33.3 ( +8.9 ) 37.8 ( +0.0 ) 28.9 ( +6.7 ) 28.9 ( +2.2 ) 31.1 ( +4.4 )
TimeChat [31]Uniform 31.1 33.3 28.9 46.7 31.1 26.7 33.0
RAG-Adapter 33.3 ( +2.2 ) 42.2 ( +8.9 ) 31.1 ( +2.2 ) 48.9 ( +2.2 ) 33.3 ( +2.2 ) 33.3 ( +6.6 ) 37.0 ( +4.0 )
Chat-UniVi [17]Uniform 33.0 26.7 24.4 37.8 31.1 24.4 29.6
RAG-Adapter 42.2 ( +8.9 ) 35.6 ( +8.9 ) 35.6 ( +11.2 ) 46.7 ( +8.9 ) 42.2 ( +11.1 )35.6 ( +11.2 ) 39.7 ( +10.1 )
GPT-4o [28]Uniform 66.7 62.2 66.7 64.4 55.6 53.5 61.5
RAG-Adapter 77.8 (+11.1 ) 73.3 (+11.1 ) 68.9 (+2.2 ) 75.6 (+11.2 ) 68.9 (+13.3 )60.0 (+6.7 ) 70.8 (+9.3 )
Table 3. Comparison across more benchmarks.
Models Sampling Method MLVU Perception Test EgoSchema
MovieChat [35]Uniform 29.6 32.5 23.3
RAG-Adapter 41.5 ( +11.9 ) 37.8 ( +5.3 ) 28.9 ( +5.6 )
LLaMA-VID [22]Uniform 34.8 33.1 24.4
RAG-Adapter 43.0 (+8.2 ) 37.2 ( +4.1 ) 31.1 ( +6.7 )
TimeChat [31]Uniform 37.8 37.8 27.8
RAG-Adapter 45.2 (+7.4 ) 41.1 (+3.3 ) 32.2 (+4.4 )
Chat-UniVi [17]Uniform 32.6 38.1 32.2
RAG-Adapter 40.0 ( +7.4 ) 41.6 (+3.5 ) 41.1 (+8.9 )
slightly higher NIF values than the other two benchmarks,
given their longer average durations, though the number
of frames containing essential information remains limited.
Supplementary materials provide each test video’s ID, cor-
responding question (or ID), minimum frame count, frame
timestamps and identified issues in the benchmarks using
RAG-Adapter.
4.3. Baselines and Experimental Setups
We classify the baselines into two categories: image-
based MLLMs supporting multiple image inputs and video
MLLMs. Open-source models are tested locally on an
NVIDIA 4090 GPU, while proprietary models are accessed
via official APIs. Based on the NIF metrics (Table 1) and
the maximum number of key frames—Video-MME (9),
MLVU (20), Perception Test (8), and EgoSchema (6)—we
setK= 10 frames for Video-MME, Perception Test, and
EgoSchema, and K= 20 for MLVU to ensure sufficient
information retrieval by RAG-Adapter and maintain evalua-
tion fairness ( M,N, andθare set to 50,50, and 0.7, respec-
tively). Frames are input in chronological order to preserve
temporal information. We compare each MLLM’s perfor-
mance under identical frame input conditions, contrasting
uniform sampling with RAG-Adapter sampling. The accu-
racy (ranging from 0 to 100) for multiple-choice questions
across the four benchmarks is calculated by comparing theTable 4. Comparison under different fine-tuning methods.
Models Fine-Tuning Method Avg. Acc. (%)
TimeChat [31]No Fine-Tuning 33.7
SCL 32.6 ( -1.1)
CB 34.8 ( +1.1 )
GCL 37.0 (+3.3 )
GPT-4o [28]No Fine-Tuning 65.9
SCL 65.2 ( -0.7)
CB 66.7 ( +0.8 )
GCL 70.8 (+4.9 )
predicted results with the ground truth.
4.4. Results and Analysis
Table 2 compares the performance of uniform sampling and
RAG-Adapter sampling across six domains in the Video-
MME (without subtitle information). Table 3 compares the
performance on the remaining three benchmarks, and the
more comprehensive experimental results for MLVU are
provided in the supplementary materials. From the experi-
mental results, we draw the following key conclusions:
•Performance: RAG-Adapter sampling improves overall
performance across all models compared to uniform sam-
pling, indicating that the information loss from uniform
sampling adversely affects model performance in testing.
Thus, uniform sampling does not fully reflect MLLMs’
true long video understanding capabilities.
•Unified Improvement: In Table 2, while commercial
MLLMs outperform open-source models, RAG-Adapter
consistently enhances performance. For example, GPT-
4o shows accuracy gains exceeding 10% in the Knowl-
edge, Film & Television, Artistic Performance, and Life
Record domains, with an average accuracy increase of
9.3%. Models like mPLUG-Owl-V show a 13.4% im-
provement in Knowledge, with an overall accuracy in-
crease of 6.7%, while Chat-UniVi achieves over a 10%
6

Table 5. Ablation Study of the RAG-Adapter Components, where “T&E” denotes Text Encoder, “I&E” represents Image Encoder, and
“D&R” stands for the Dual Ranker.
Component Ablation
Fine-Tuning Method T&E I&E D&R Avg. Acc. (%) Preprocessing Retrieval Inference Recall@10
GCL✓ 33.3 48.8min 4.7s
0.88s19.7
✓ 34.5 11.4s 8.7s 18.7
✓✓ 35.2
48.8min13.5s 24.8
✓✓ ✓ 39.7
15.4s30.4
No Fine-Tuning ✓✓ ✓ 33.3 24.1
SCL ✓✓ ✓ 32.2 23.7
CB ✓✓ ✓ 35.6 25.1
Other Baselines
Sampling Method Avg. Acc. (%) Preprocessing Retrieval Inference Recall@10
Uniform 29.6 11.4s N/A0.88s5.5
Two-stage Retrieval 37.0 4.3min (8.5+2.5)s 25.8
Table 6. Comparison of different sampling strategies and input
frame counts.
Models Sampling Method Frames Count Avg. Acc. (%)
TimeChat [31]Uniform5 30.7
10 33.0
20 33.0
64 32.2
RAG-Adapter5 36.3
10 37.0
20 37.0
Chat-UniVi [17]Uniform5 27.8
10 29.6
20 32.6
256 31.9
RAG-Adapter5 35.6
10 39.7
20 40.0
Table 7. Comparison between no subtitles (w/o subs), subtitles
corresponding to RAG-Adapter sampled frames (w/ subs (Cor-
resp.)), and subtitles sampled by RAG-Adapter (w/ subs (RAG-
Adapter)).
Models Subtitles Avg. Acc. (%)
TimeChat [31]w/o subs 37.0
w/ subs (Corresp.) 38.2 ( +1.2 )
w/ subs (RAG-Adapter) 39.6 (+2.6 )
Chat-UniVi [17]w/o subs 39.7
w/ subs (Corresp.) 40.0 ( +0.3 )
w/ subs (RAG-Adapter) 41.5 (+1.8 )
improvement in Sports Competition, Life Record, Multi-
lingual, and overall accuracy increase. This demonstrates
the versatility of RAG-Adapter, as its effectiveness is not
directly linked to the intrinsic capabilities of the models.
•Generalization: In Table 3, Perception Test and
Egochema have shorter average durations (35s&3min)
compared to MLVU (12min), leading to a less pro-
nounced improvement of RAG-Adapter over uniform
sampling. Nonetheless, its performance across bench-
marks of varying lengths demonstrates the method’s ef-
fectiveness and generalization.
•Constraint: In Table 2, RAG-Adapter does not con-
sistently improve accuracy across all domains. GPT-
4 Turbo shows no improvement in Film & Television,
while MovieChat and LLaMA-VID remain unchanged in
Artistic Performance. Otter-I and VideoChat2 experience
a 2.2% accuracy drop in Multilingual and Life Record.This stability or slight decline is primarily due to RAG-
Adapter occasionally failing to retrieve all relevant infor-
mation, resulting in the omission of key frames. In such
cases, RAG-Adapter may mislead the model, affecting its
accuracy. We aim to further refine the retrieval process of
RAG-Adapter to minimize these issues.
4.5. Ablation Study
The following ablation experiments are conducted on the
Video-MME benchmark, with additional results provided
in the supplementary materials.
Effect of RAG-Adapter Fine-Tuning. In Table 4, we
evaluate RAG-Adapter’s performance across different
fine-tuning approaches, including No Fine-Tuning, Self-
supervised Contrastive Learning (SCL) fine-tuning, Cus-
tomizing Batch (CB) fine-tuning (where each question
in a batch belongs to a different video), and Grouped-
supervised Contrastive Learning (GCL) fine-tuning. The
results demonstrate that GCL fine-tuning achieves superior
performance for all models. This enhancement is primar-
ily due to GCL’s ability to train RAG-Adapter’s text and
image encoders to effectively learn rich positive sample
features within each group while avoiding the adverse ef-
fects of treating intra-group samples as negatives, as seen in
SCL. Moreover, GCL retains all inter-group negative sam-
ples from SCL and CB, ensuring robust learning.
Discussion on the effectiveness and efficiency of RAG-
Adapter components. In Table 5, we utilize Chat-UniVi
to conduct an ablation study on RAG-Adapter’s compo-
nents, evaluating pipeline efficiency (scaled to 10-minute
videos per query) and average recall. The preprocessing
phase involves frame sampling and captioning. Our method
achieves optimal accuracy and recall only with all compo-
nents and GCL fine-tuning. To address the impracticality
of captioning every frame in long videos, we propose a
Two-stage Retrieval method: initially retrieving the top 50
frames, then using captions to refine the selection to the
final top Kframes, achieving a favorable balance between
accuracy and efficiency. Also, retrieval accuracy using only
Image Encoder outperforms uniform sampling, offering an-
7

Video ID: s-lM2uwiwyQ     
Question ID: 257-3
Question: How many people are shown having lunch with the woman in the video?
Answer: (1)
(1) Uniform Sampling
(2) RAG-Adapter Sampling Relevant FramesFigure 4. Comparison of RAG-Adapter and uniform sampling results: RAG-Adapter accurately identifies two consecutive key frames
relevant to the question, whereas uniform sampling tends to miss them.
"What is the 
total number 
of people in 
the video?"
Answer: (7)
Figure 5. The relationship between the embedding spaces of video
frames sampled using different methods and that of the corre-
sponding questions. The frame embeddings are primarily grouped
into five clusters, each representing a set of consecutive shots, with
each cluster labeled by a representative frame.
other viable alternative in practice. Furthermore, inspired
by SeViLA [42], we employ the tested MLLM for frame
filtering but find it time-intensive and ineffective.
Comparison of Different Input Frame Counts. Since
some long-video MLLMs can support a larger number of
input frames, we compare uniform sampling (using 5, 10,
20 frames or the model’s maximum supported frames on a
single NVIDIA 4090) with RAG-Adapter sampling (using
5, 10, and 20 frames), as shown in Table 6. Results indi-
cate that, despite utilizing more frames, uniform sampling
does not outperform RAG-Adapter and even exhibits slight
performance degradation compared to using fewer frames.
For RAG-Adapter sampling, performance improves from
K= 5 toK= 10 , suggesting information loss at K= 5,
and stabilizes at K= 20 . This aligns with the NIF metric
for Video-MME, which is below 10, implying most ques-
tions require fewer than 10 frames to capture essential infor-
mation. Additionally, increasing uniformly sampled frames
does not guarantee inclusion of critical details and may in-troduce greater redundancy.
Impact of Subtitle Information. In the Video-MME
benchmark, subtitle files are available and contain some in-
formation relevant to certain questions. In Table 7, we ex-
amine the impact of subtitles on MLLMs using 10 frames
sampled by RAG-Adapter. We evaluate two subtitle inclu-
sion methods: providing subtitles directly correspond to the
sampled frames and using RAG-Adapter to select the 10
most relevant subtitles for each question.
Our experiments reveal two main insights. First, model
accuracy consistently improves with subtitle inclusion, as
subtitles often provide question-relevant information. Sec-
ond, subtitles filtered by RAG-Adapter outperform those di-
rectly tied to sampled frames, as critical subtitle information
may not align with key video content, and complex ques-
tions often rely more heavily on subtitle data.
5. Visualization
5.1. Visualization of Frame Sampling Methods
In Figure 4, we compare the results of uniform sampling
and RAG-Adapter sampling for the same question in the
Video-MME benchmark. The specific scene referenced in
the question - “How many people are shown having lunch
with the woman in the video?”, occurs only between 73-74
seconds in the original video. As a result, uniform sampling
fails to capture any relevant frames, whereas RAG-Adapter
successfully identifies the two pertinent frames (sampled
at one frame per second). Additional visualizations of the
video frames are provided in the supplementary materials.
5.2. Differences of Embedding spaces
In Figure 5, we reduce the embedding space of the question
and all video frames to two dimensions using UMAP [27]
(Uniform Manifold Approximation and Projection) to pre-
serve the global structure of the data. This visualization
illustrates the spatial relationship between the question em-
bedding and the embeddings of frames sampled by uniform
8

sampling, the non-fine-tuned RAG-Adapter, and the GCL
fine-tuned RAG-Adapter. It can be observed that the em-
beddings of uniformly sampled frames are highly scattered,
while the embeddings of frames sampled by the non-fine-
tuned RAG-Adapter cluster around a few similar frames. In
contrast, the embeddings from the GCL fine-tuned RAG-
Adapter exhibit greater diversity and are closer to the ques-
tion embedding.
6. Conclusion
In this paper, we integrate the RAG framework with
MLLMs, introducing RAG-Adapter, a plugin that enhances
the long video understanding capabilities of MLLMs
without modifying their internal structure. By provid-
ing question-relevant video frames during testing, RAG-
Adapter ensures that the evaluation of long video under-
standing benchmarks accurately reflects the model’s true
video comprehension capabilities. To better adapt RAG-
Adapter to the video question-answering context, we con-
struct a fine-tuning dataset, MMAT, and introduce Grouped-
supervised Contrastive Learning (GCL) to help RAG-
Adapter learn rich and relevant embedding between ques-
tions and video frames. Additionally, we proposed two met-
rics, ASS and NIF, to assess the benchmarks quality and
complexity, using NIF as a basis for determining the num-
ber of frames sampled by RAG-Adapter. Tests on Video-
MME, MLVU, Perception Test and EgoSchema demon-
strate that RAG-Adapter consistently improves accuracy
across all baseline MLLMs, demonstrating our approach’s
simplicity and effectiveness.
Limitations. RAG-Adapter does not always retrieve all
relevant frames, especially when key information is dis-
persed across multiple segments, often returning only a sub-
set. Additionally, complex tasks like sentiment analysis or
video summarization, which lack explicit visual cues, may
further constrain its effectiveness. Moreover, the substan-
tial preprocessing time required to encode video data into
the database makes RAG-Adapter unsuitable for real-time
video processing. While the proposed Two-Stage Retrieval
and purely visual retrieval strategies mitigate this issue, fu-
ture work will focus on further optimizing retrieval effi-
ciency.
References
[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-
mad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,
Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al.
Gpt-4 technical report. arXiv preprint arXiv:2303.08774 ,
2023. 6
[2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine
Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Men-
sch, Katherine Millican, Malcolm Reynolds, et al. Flamingo:
a visual language model for few-shot learning. Advancesin neural information processing systems , 35:23716–23736,
2022. 2
[3] Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Ud-
din, and Srimat Chakradhar. irag: An incremental retrieval
augmented generation system for videos. arXiv preprint
arXiv:2404.12309 , 2024. 3
[4] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. Self-rag: Learning to retrieve, gen-
erate, and critique through self-reflection. arXiv preprint
arXiv:2310.11511 , 2023. 3
[5] Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Es-
sam Sleiman, Deyao Zhu, Jian Ding, and Mohamed El-
hoseiny. Minigpt4-video: Advancing multimodal llms for
video understanding with interleaved visual-textual tokens.
arXiv preprint arXiv:2404.03413 , 2024. 2
[6] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang,
Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei
Huang, et al. Qwen technical report. arXiv preprint
arXiv:2309.16609 , 2023. 2
[7] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao
Zhang, Christoph Feichtenhofer, and Judy Hoffman. To-
ken merging: Your vit but faster. arXiv preprint
arXiv:2210.09461 , 2022. 3
[8] Jaime Carbonell and Jade Goldstein. The use of mmr,
diversity-based reranking for reordering documents and pro-
ducing summaries. In Proceedings of the 21st annual inter-
national ACM SIGIR conference on Research and develop-
ment in information retrieval , pages 335–336, 1998. 3
[9] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. Bge m3-embedding: Multi-
lingual, multi-functionality, multi-granularity text embed-
dings through self-knowledge distillation. arXiv preprint
arXiv:2402.03216 , 2024. 3
[10] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhang-
wei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng
Luo, Zheng Ma, et al. How far are we to gpt-4v? closing
the gap to commercial multimodal models with open-source
suites. arXiv preprint arXiv:2404.16821 , 2024. 2
[11] Thibault Formal, Benjamin Piwowarski, and St ´ephane Clin-
chant. Splade: Sparse lexical and expansion model for first
stage ranking. In Proceedings of the 44th International ACM
SIGIR Conference on Research and Development in Infor-
mation Retrieval , pages 2288–2292, 2021. 3
[12] Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren,
Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen,
Mengdan Zhang, et al. Video-mme: The first-ever compre-
hensive evaluation benchmark of multi-modal llms in video
analysis. arXiv preprint arXiv:2405.21075 , 2024. 1, 3
[13] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui
Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng,
Hanlin Zhao, et al. Chatglm: A family of large language
models from glm-130b to glm-4 all tools. arXiv preprint
arXiv:2406.12793 , 2024. 2
[14] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xue-
fei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam
Lim. Ma-lmm: Memory-augmented large multimodal model
for long-term video understanding. In Proceedings of the
9

IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 13504–13514, 2024. 2
[15] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu,
Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Jun-
hui Ji, Zhao Xue, et al. Cogvlm2: Visual language mod-
els for image and video understanding. arXiv preprint
arXiv:2408.16500 , 2024. 3
[16] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and
Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in
visual question answering. In Proceedings of the IEEE con-
ference on computer vision and pattern recognition , pages
2758–2766, 2017. 4
[17] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao,
and Li Yuan. Chat-univi: Unified visual representation em-
powers large language models with image and video un-
derstanding. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 13700–
13710, 2024. 3, 6, 7
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp
tasks. Advances in Neural Information Processing Systems ,
33:9459–9474, 2020. 3
[19] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang,
Jingkang Yang, and Ziwei Liu. Otter: a multi-modal model
with in-context instruction tuning. corr abs/2305.03726
(2023), 2023. 2, 6
[20] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai
Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao.
Videochat: Chat-centric video understanding. arXiv preprint
arXiv:2305.06355 , 2023. 6
[21] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang,
Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al.
Mvbench: A comprehensive multi-modal video understand-
ing benchmark. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 22195–
22206, 2024. 6
[22] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid:
An image is worth 2 tokens in large language models. In
European Conference on Computer Vision , pages 323–340.
Springer, 2025. 1, 3, 6
[23] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan
Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Im-
proved reasoning, ocr, and world knowledge, 2024. 2, 6
[24] Ziyu Liu, Zeyi Sun, Yuhang Zang, Wei Li, Pan Zhang, Xi-
aoyi Dong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Rar:
Retrieving and ranking augmented mllms for visual recogni-
tion. arXiv preprint arXiv:2403.13805 , 2024. 3
[25] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fa-
had Shahbaz Khan. Video-chatgpt: Towards detailed video
understanding via large vision and language models. arXiv
preprint arXiv:2306.05424 , 2023. 2
[26] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra
Malik. Egoschema: A diagnostic benchmark for very long-
form video language understanding. Advances in Neural In-
formation Processing Systems , 36:46212–46244, 2023. 5[27] Leland McInnes, John Healy, and James Melville. Umap:
Uniform manifold approximation and projection for dimen-
sion reduction. arXiv preprint arXiv:1802.03426 , 2018. 8
[28] OpenAI. Gpt-4o, 2024. 6
[29] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Re-
casens, Larisa Markeeva, Dylan Banarse, Skanda Koppula,
Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Per-
ception test: A diagnostic benchmark for multimodal video
models. Advances in Neural Information Processing Sys-
tems, 36:42748–42761, 2023. 5
[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning
transferable visual models from natural language supervi-
sion. In International conference on machine learning , pages
8748–8763. PMLR, 2021. 3
[31] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu
Hou. Timechat: A time-sensitive multimodal large lan-
guage model for long video understanding. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition , pages 14313–14323, 2024. 6, 7
[32] Timo Schick, Jane Dwivedi-Yu, Roberto Dess `ı, Roberta
Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer,
Nicola Cancedda, and Thomas Scialom. Toolformer: Lan-
guage models can teach themselves to use tools. Advances
in Neural Information Processing Systems , 36, 2024. 3
[33] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng
Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Bal-
akrishnan Varadarajan, Florian Bordes, et al. Longvu: Spa-
tiotemporal adaptive compression for long video-language
understanding. arXiv preprint arXiv:2410.17434 , 2024. 2
[34] Robik Shrestha, Yang Zou, Qiuyu Chen, Zhiheng Li,
Yusheng Xie, and Siqi Deng. Fairrag: Fair human gen-
eration via fair retrieval augmentation. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 11996–12005, 2024. 3
[35] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng
Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo,
Tian Ye, Yanting Zhang, et al. Moviechat: From dense token
to sparse memory for long video understanding. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 18221–18232, 2024. 1, 3, 6
[36] Ayush Thakur and Rashmi Vashisth. Loops on re-
trieval augmented generation (lorag). arXiv preprint
arXiv:2403.15450 , 2024. 3
[37] Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma,
and Yitao Liang. Rat: Retrieval augmented thoughts elicit
context-aware reasoning in long-horizon generation. arXiv
preprint arXiv:2403.05313 , 2024. 3
[38] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang,
and Bohan Zhuang. Longvlm: Efficient long video un-
derstanding via large language models. arXiv preprint
arXiv:2404.03384 , 2024. 2
[39] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang,
Xiangnan He, and Yueting Zhuang. Video question answer-
ing via gradually refined attention over appearance and mo-
tion. In Proceedings of the 25th ACM international confer-
ence on Multimedia , pages 1645–1653, 2017. 4
10

[40] Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, An-
toine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, and
Cordelia Schmid. Vid2seq: Large-scale pretraining of a vi-
sual language model for dense video captioning. In Proceed-
ings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition , pages 10714–10726, 2023. 2
[41] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan,
Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi,
Yaya Shi, et al. mplug-owl: Modularization empowers
large language models with multimodality. arXiv preprint
arXiv:2304.14178 , 2023. 6
[42] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal.
Self-chained image-language model for video localization
and question answering. Advances in Neural Information
Processing Systems , 36:76749–76771, 2023. 8
[43] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yuet-
ing Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for
understanding complex web videos via question answering.
InProceedings of the AAAI Conference on Artificial Intelli-
gence , pages 9127–9134, 2019. 4
[44] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi
Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng
Liu. Mlvu: A comprehensive benchmark for multi-task
long video understanding. arXiv preprint arXiv:2406.04264 ,
2024. 1, 3
11

RAG-Adapter: A Plug-and-Play RAG-enhanced Framework
for Long Video Understanding
Supplementary Material
Appendix Contents
A. Training Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . 1
B. Additional Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . 1
B.1. Comparison of more MLLMs under different
fine-tuning methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
B.2. Comparison of more MLLMs across different
input frame numbers and sampling strategies . . . . . . . .1
B.3. Comparison of more MLLMs under different
subtitle input conditions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
B.4. More Comparison Results on MLVU . . . . . . . . . 2
C. Unreasonable Questions in the Benchmark . . . . . . . 3
D. Additional Visualization Results . . . . . . . . . . . . . . . . . . 6
E. Detailed NIF Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
A. Training Hyperparameters
Table 8 provides the hyperparameters used for fine-tuning
the image encoder (BGE-M3) and text encoder (CLIP-L/14)
in RAG-Adapter. Both Self-supervised Contrastive Learn-
ing and Grouped-supervised Contrastive Learning use the
same hyperparameter configurations.
Table 8. Training Hyperparameters for BGE-M3 and CLIP-L/14.
HyperparameterEncoders
BGE-M3 CLIP-L/14
Batch size 32 32
Fine-tuning epochs 2 2
Fine-tuning iterations 26126 26126
Temperature 20 20
Weight decay 0.01 0.01
Learning rate 2e-5 1e-5
Warm-up iterations 2612 2612
Optimizer AdamW AdamW
Schedule linear decay cosine decay
AdamW β1 0.9 0.9
AdamW β2 0.999 0.98
AdamW ϵ 1e-6 1e-6B. Additional Experiments
B.1. Comparison of more MLLMs under different
fine-tuning methods
Table 9. Comparison under different fine-tuning methods.
Models Fine-Tuning Method Avg. Acc. (%)
MovieChat [35]No Fine-Tuning 29.2
SCL 28.5 ( -0.7)
CB 29.3 ( +0.1 )
GCL 33.0 (+3.8 )
LLaMA-VID [22]No Fine-Tuning 28.9
SCL 28.9 ( +0.0 )
CB 29.6 ( +0.7 )
GCL 31.1 (+2.2 )
B.2. Comparison of more MLLMs across different
input frame numbers and sampling strategies
Table 10. Comparison of different sampling strategies and input
frame counts.
Models Sampling Method Frames Count Avg. Acc. (%)
MovieChat [35]Uniform5 25.6
10 26.3
20 27.0
512 28.9
RAG-Adapter5 30.0
10 33.0
20 33.3
LLaMA-VID [22]Uniform5 26.7
10 26.7
20 27.1
512 27.8
RAG-Adapter5 28.9
10 31.1
20 30.8
B.3. Comparison of more MLLMs under different
subtitle input conditions
Table 11. Comparison between no subtitles (w/o subs), subtitles
corresponding to RAG-Adapter sampled frames (w/ subs (Cor-
resp.)), and subtitles sampled by RAG-Adapter (w/ subs (RAG-
Adapter)).
Models Subtitles Avg. Acc. (%)
MovieChat [35]w/o subs 33.0
w/ subs (Corresp.) 34.1 ( +1.1 )
w/ subs (RAG-Adapter) 34.8 (+1.8 )
LLaMA-VID [22][22] w/o subs 31.1
w/ subs (Corresp.) 31.9 ( +0.8 )
w/ subs (RAG-Adapter) 32.2 (+1.1 )
1

B.4. More Comparison Results on MLVU
Due to page limitations in the main body of the paper, we have included the comprehensive experiments on MLVU men-
tioned in Section 4.4 in the supplementary materials. Table 12 presents the performance of various MLLMs on the MLVU
benchmark using uniform sampling versus RAG-Adapter sampling, with all models evaluated using 20 frames. The results
are consistent with those on the Video-MME benchmark: RAG-Adapter sampling improves the performance of all MLLMs
compared to uniform sampling. For M-Avg, GPT-4o achieves the highest improvement of 12.6%, while VideoChat2 shows
the lowest gain of 6.6%. However, for G-Avg, the improvements are generally modest, with even a slight decline (e.g.,
VideoChat decreases by 0.03). This is because generative tasks require a more comprehensive understanding of the video
content, meaning that the sampled frames must adequately represent the entire video. In such scenarios, RAG-Adapter
sampling offers no significant advantage over uniform sampling.
Table 12. The test results for various MLLMs on MLVU. The evaluation includes nine types of tasks: PQA (Plot QA), NQA (Needle
QA), ER (Ego Reasoning), AC (Action Count), AO (Action Order), AR (Anomaly Recognition), TR (Topic Reasoning), SSC (Sub-Scene
Captioning), and VS (Video Summary). “M-Avg” (0-100) represents the average performance across multiple-choice tasks, while “G-Avg”
(0-10, marked by *) indicates the average performance for generative tasks.
Models Sampling MethodCategoryM-Avg G-AvgPQA NQA ER AC AO AR TR SSC* VS*
Image MLLMs
Otter-IUniform 31.4 20.8 34.4 20.0 40.0 40.0 40.0 2.15 1.10 31.8 1.63
RAG-Adapter 34.3 ( +2.9 )54.2 ( +33.4 )46.9 ( +12.5 )10.0 ( -10.0 )50.0 ( +10.0 )40.0 ( +0.0 )53.3 ( +13.3 ) 2.05 ( -0.1) 1.35 ( +0.25 )43.0 ( +11.2 )1.70 ( +0.07 )
LLaV A-1.6Uniform 34.3 29.2 34.4 20.0 20.0 50.0 73.3 1.30 1.05 37.0 1.18
RAG-Adapter 57.1 ( +22.8 )50.0 ( +20.8 )34.4 ( +0.0 ) 20.0 ( +0.0 )30.0 ( +10.0 )70.0 ( +20.0 ) 66.7 ( -6.6) 1.95 ( +0.65 )1.30 ( +0.25 )48.1 ( +11.1 )1.63 ( +0.45 )
Video MLLMs
Otter-VUniform 28.6 21.7 18.8 20.0 40.0 30.0 33.3 2.20 1.10 26.1 1.65
RAG-Adapter 31.4 ( +2.8 )37.5 ( +15.8 )28.1 ( +9.3 )40.0 ( +20.0 )40.0 ( +0.0 )40.0 ( +10.0 )40.0 ( +6.7 )2.25 ( +0.05 )1.20 ( +0.10 )34.8 ( +8.7 )1.73 ( +0.08 )
mPlug-Owl-VUniform 25.7 33.3 37.5 40.0 20.0 40.0 26.7 2.15 1.10 31.8 1.63
RAG-Adapter 34.3 ( +8.6 )50.0 ( +16.7 )40.6 ( +3.1 )50.0 ( +10.0 )40.0 ( +20.0 )50.0 ( +10.0 )40.0 ( +13.3 )2.80 ( +0.65 )1.15 ( +0.05 )42.2 ( +10.4 )1.98 ( +0.35 )
MovieChatUniform 25.7 29.2 25.0 30.0 30.0 20.0 53.3 1.40 1.05 29.6 1.23
RAG-Adapter 42.9 ( +17.2 )37.5 ( +8.3 ) 34.4 ( +9.4 )40.0 ( +10.0 )40.0 ( +10.0 )50.0 ( +30.0 )53.3 ( +0.0 )1.45 ( +0.05 )1.10 ( +0.05 )41.5 ( +11.9 )1.28 ( +0.05 )
VideoChatUniform 22.9 16.7 25.0 30.0 20.0 40.0 26.7 2.25 1.40 24.5 1.83
RAG-Adapter 31.4 ( +8.5 )33.3 ( +16.6 )25.0 ( +0.0 )40.0 ( +10.0 )40.0 ( +20.0 )50.0 ( +10.0 )33.3 ( +6.6 )2.30 ( +0.05 )1.30 ( -0.10 ) 33.3 ( +8.8 ) 1.80 ( -0.03 )
VideoChat2Uniform 34.3 29.2 21.9 30.0 20.0 40.0 26.7 2.25 1.15 28.9 1.70
RAG-Adapter 37.1 ( +2.8 ) 37.5 ( +8.3 ) 31.3 ( +9.4 )40.0 ( +10.0 )30.0 ( +10.0 )40.0 ( +0.0 ) 33.3 ( +6.6 )2.65 ( +0.40 )1.20 ( +0.05 )35.5 ( +6.6 )1.93 ( +0.23 )
LLaMA-VIDUniform 25.7 33.3 40.6 40.0 20.0 50.0 40.0 2.45 1.25 34.8 1.85
RAG-Adapter 31.4 ( +5.7 ) 37.5 ( +4.2 ) 43.8 ( +3.2 )50.0 ( +10.0 )20.0 ( +0.0 )70.0 ( +20.0 )66.7 ( +26.7 )2.65 ( +0.20 )1.25 ( +0.00 )43.0 ( +8.2 )1.95 ( +0.10 )
TimeChatUniform 34.3 41.7 40.6 40.0 40.0 20.0 40.0 1.69 1.10 37.8 1.40
RAG-Adapter 42.9 ( +8.6 )54.2 ( +12.5 )40.6 ( +0.0 )50.0 ( +10.0 )60.0 ( +20.0 )20.0 ( +0.0 ) 46.7 ( +6.7 )1.85 ( +0.16 )1.05 ( -0.05 ) 45.2 ( +7.4 )1.45 ( +0.05 )
Chat-UniViUniform 34.3 37.5 21.9 30.0 20.0 60.0 33.3 2.75 1.20 32.6 1.98
RAG-Adapter 37.1 ( +2.8 ) 45.8 ( +8.3 ) 28.1 ( +6.2 )50.0 ( +20.0 )30.0 ( +10.0 )60.0 ( +0.0 )46.7 ( +13.4 )2.95 ( +0.20 )1.20 ( +0.00 )40.0 ( +7.4 )2.08 ( +0.10 )
GPT-4oUniform 54.3 54.2 37.5 40.0 50.0 50.0 53.3 1.40 1.55 48.9 1.48
RAG-Adapter 62.9 ( +8.6 )70.8 ( +16.6 )50.0 ( +12.5 )50.0 ( +10.0 )60.0 ( +10.0 )80.0 ( +30.0 )60.0 ( +6.7 )2.35 ( +0.95 )1.85 ( +0.30 )61.5 ( +12.6 )2.10 ( +0.62 )
2

C. Unreasonable Questions in the Benchmark
While using RAG-Adapter to assist in calculating the NIF values for benchmarks, we identified a few unreasonable questions,
detailed in Figures 6 to 11. Despite these issues, both benchmarks maintain high overall quality. This suggests that RAG-
Adapter is an effective tool for evaluating and refining long video benchmarks, significantly reducing manual verification
effort and enhancing benchmark quality.
Video ID: dH8l--46j6s
Question ID: 214-3 
Question and Options: 
What does the male performer wear in this video?
A. Black pants and white shorts.   B. Black pants with a naked upper body.
C. Black pants with a naked upper body.   D. Black pants with a naked upper body.
Answer: (B)
Issue: Options B, C, and D are identical.Video-MME
Video ID: zNxi2s36tS0 
Question ID: 163-3 
Question and Options: 
What is the shortest time to reach the finish line in the video?
A. 9.5 seconds.   B. 10.06 seconds.   C. 8.7 seconds.   D. 8.5 seconds.
Answer: (B)
Issue: Options C and D are unreasonable; the world record for the 100-meter sprint is 9.58 seconds.
Figure 6. Issues identified in Video-MME during NIF calculations using RAG-Adapter.
MLVU
Video ID: subPlot_new_all_97
Question: 
Please describe the situation after a woman in white riding a horse is shot by an arrow and falls off the horse.
Answer: 
The man in front of the city gate looks back at the man in yellow clothes on the city tower, then turns back 
and runs towards the woman in white riding the horse.
Issue: The scene where "a woman in white riding a horse is shot by an arrow" does not appear in the video.
Video ID: needle_51
Question and Options: 
What nationality are the kids having fun in the paddy field?
A. American   B. Malays   C. Chinese   D. Indian
Answer: (B)
Issue: It is not possible to determine the children's nationality solely from the frames.
Figure 7. Issues identified in MLVU during NIF calculations using RAG-Adapter.
3

EgoSchema
Video ID:  51688142-10e7-48ab-adef-2caa5448b456 
Question and Options: 
How does the introduction of the palm frond contribute to the development of the final product, and why 
might c have chosen to include it?
   "option 0": "The palm frond contributes to the development of the final product by providing a structural 
element.",
   "option 1": "The versatile palm frond significantly contributes to the development of the final product by 
providing a highly functional and essential element.",
   "option 2": "The palm frond contributes to the development of the final product by providing a decorative 
element.",
   "option 3": "The palm frond significantly contributes to the ultimate development of the final product by 
reliably providing a crucial protective element.",
   "option 4": "The palm frond significantly contributes to the development of the final product by providing 
an essential nutritional element, enriching its value."
Answer: 2
Issue: Palm does not appear in the video.Figure 8. Issue 1 identified in EgoSchema during NIF calculations using RAG-Adapter.
EgoSchema
Video ID: dfb6c468-e124-40f6-9c4e-c13ee45a2ad9
Question and Options: 
What is the overarching goal of the actions taken by both c and the man throughout the video, and how do 
their techniques differ?
   "option 0": "The primary, overarching goal of the various actions taken by both c and the man in the video 
is to efficiently repair a broken light fixture together.",
   "option 1": "The primary, overarching goal driving the actions taken by both individual c and the man 
appearing throughout the entire video is to successfully remove a specific light fixture.",
   "option 2": "The overarching goal of the actions taken by both c and the man throughout the video is to 
clean a light fixture.",
   "option 3": "The primary, overarching goal of the various actions taken by both c and the man throughout 
the entire video is simply to replace a malfunctioning light bulb.",
   "option 4": "The overarching goal of the actions taken by both c and the man throughout the video is to 
install a new light fixture."
Answer: 4
Issue: The goal is to demolish a wall, not to install a new light fixture.
Figure 9. Issue 2 identified in EgoSchema during NIF calculations using RAG-Adapter.
4

EgoSchema
Video ID: ece69b04-2e67-434a-b923-1329feed590d
Question and Options: 
What is the primary objective c is trying to achieve in the video, and how does their interaction with various 
materials (ruler, cutter, craft pieces, glue, etc.) contribute to that objective?
   "option 0": "C is trying to build a model. they use the ruler to measure the craft pieces, the cutter to cut 
them out, the glue to put them together, and the paper to decorate them.",
   "option 1": "C is trying to make a craft. they use the ruler to measure the craft pieces, the cutter to cut 
them out, the glue to put them together, and the paper to decorate them.",
   "option 2": "C is diligently attempting to construct a structure. they skillfully utilize the ruler for measuring 
the various craft elements, employ the cutter to accurately shape them, apply the adhesive glue for securely 
assembling, and utilize the colorful paper to aesthetically decorate them.",
   "option 3": "Creatively, c is attempting to make an imaginative toy. diligently, they use the ruler to 
precisely measure the craft pieces, the cutter to skillfully cut them out, the glue to securely put them 
together, and the vibrant paper to beautifully decorate them.",
   "option 4": "C is attempting to create a thoughtful gift. they utilize the ruler to precisely measure the craft 
pieces, employ the cutter to shape them, the adhesive glue to assemble them securely, and the decorative 
paper to enhance their appearance."
Answer: 0
Issue: The video does not contain a cutter or decorative paper.Figure 10. Issue 3 identified in EgoSchema during NIF calculations using RAG-Adapter.
EgoSchema
Video ID: e67de76c-1058-49a7-a47e-12736da4ffc0
Question and Options: 
Based on the actions performed by the woman and c, determine the critical moments in the video when they 
accomplish their respective tasks and care for the bearded dragon and the playing cards. how do these 
moments illustrate their priorities?
   "option 0": "The critical moments in the video are when c picks up the bottle from the table, touches the 
cards on the table, and picks up the cards from the table. these moments illustrate his priority of playing 
with the cards.",
   "option 1": "The critical moments in the video are when the woman and c are both playing with the cards. 
these moments illustrate their shared priority of playing with the cards.",
   "option 2": "The critical moments in the video are when the woman and c are both caring for the bearded 
dragon. these moments illustrate their shared priority of caring for the bearded dragon.",
   "option 3": "The critical moments in the video are when the woman plays with the bearded dragon and c 
plays with the cards. these moments illustrate their different priorities.",
   "option 4": "The critical moments in the video are when the woman feeds the bearded dragon water, 
cleans it, and puts it in a container. these moments illustrate her priority of caring for the bearded dragon."
Answer: 4
Issue: The woman in the video does not place the bearded dragon into a container.
Figure 11. Issue 4 identified in EgoSchema during NIF calculations using RAG-Adapter.
5

D. Additional Visualization Results
Figures 12 to 16 present additional comparisons between uniform sampling of 10 frames and RAG-Adapter sampling of 10
frames on the Video-MME benchmark. The results demonstrate that RAG-Adapter more accurately identifies video frames
relevant to the question. In contrast, uniform sampling often misses these critical frames, resulting in MLLMs lacking
essential information when answering questions.
Video ID: 0JjgoicpYkU
Question ID: 599-3
Question: How much money can you save by purchasing the course through the link and 
promo code shared by the heroine in the video?
Answer: (11550 rubles)
(1) Uniform Sampling
(2) RAG-Adapter Sampling Relevant Frames
Figure 12. Comparison of RAG-Adapter and uniform sampling results: The answer to the question appears at the 527th and 528th seconds
of the video, showing the original course price as 16,500 rubles and the current price as 4,950 rubles, resulting in a total saving of 11,550
rubles. RAG-Adapter accurately identifies these two consecutive key frames, while uniform sampling tends to miss them.
Video ID: 1q-5IIyZL20
Question ID: 197-2
Question: What is the function of the black plastic bag shown in the video?
Answer: (To catch the dog's poo.)
(1) Uniform Sampling
(2) RAG-Adapter Sampling Relevant Frames
Figure 13. Comparison of RAG-Adapter and uniform sampling results: The answer to the question appears between the 68th and 73rd
seconds of the video, where a man picks up a black plastic bag to clean up after his dog. RAG-Adapter accurately identifies one key frame
depicting this action (the rest of the frames contain highly similar content), whereas uniform sampling misses it.
6

Video ID: 1wzgMHrkrys
Question ID: 779-2 
Question: Who is interviewed both before and after the race based on the video?
Answer: (Jake Gagne.)
(1) Uniform Sampling
(2) RAG-Adapter Sampling Relevant Frames
Figure 14. Comparison of RAG-Adapter and uniform sampling results: The answer to the question appears between 270-294 seconds and
2329-2414 seconds of the video, where a reporter interviews Jake Gagne both before and after the race. RAG-Adapter accurately identifies
frames at 280s, 284s, and 285s before the race, and 2334s after the race, showing the interview with Jake Gagne. The frames at 284s and
285s explicitly display Jake Gagne’s name. In contrast, uniform sampling only captures a frame at 2428s, which shows an interview with
another competitor, missing the key moments relevant to Jake Gagne.
Video ID: 8l1NyR6UvxU
Question ID: 307-2 
Question: Which of the following features can not describe Spartacus?
Answer: (Thick beard.)
(1) Uniform Sampling
(2) RAG-Adapter Sampling Relevant Frames
Figure 15. Comparison of RAG-Adapter and uniform sampling results: The answer to the question appears at the 39s and 40s of the video,
clearly displaying the name Spartacus and his appearance (notably without a thick beard). RAG-Adapter accurately identifies these two
key frames, whereas uniform sampling fails to capture them, missing essential visual details.
Video ID: 40BlVzjxu-I
Question ID: 006-1 
Question: What is one of the symbols of the festival that is introduced by the video?
Answer: (Shamrock.)
(1) Uniform Sampling
(2) RAG-Adapter Sampling Relevant Frames
Figure 16. Comparison of RAG-Adapter and uniform sampling results: The Shamrock logo appears in the video at 7-8 seconds, 23-25
seconds, 42-44 seconds, 53-55 seconds, and 95-105 seconds. Despite its frequent appearance, uniform sampling fails to capture any of
these key frames, whereas RAG-Adapter successfully identifies key frames at 42-43 seconds, 54-55 seconds, and the 101st second.
7

E. Detailed NIF Statistics
In Section 1, we propose the concept of the Necessary Information Frame (NIF), which represents the average minimum
number of frames containing essential information needed to answer each question. Table 1 in the paper presents the NIF
values for test benchmarks. To better validate the NIF metric, we manually collect the necessary frames (NIF value for each
question) and their corresponding timestamps (in seconds) for all questions across four benchmarks. Figures 17 to 22 illustrate
the statistics for Video-MME, including the question IDs and corresponding video IDs. Figures 23 to 27 show similar data for
MLVU, including partial question content, as MLVU lacks specific question IDs, along with the corresponding task and video
ID. Figures 28 to 34 and Figures 35 and 36 correspond to the relevant data for Perception Test and EgoSchema, respectively.
video_id timestamp
fFYNmVb3NCQ598-24 298;432;494;518
fFYNmVb3NCQ598-32 61;233
1O1TfTrEnss895-13 48;188;306
1O1TfTrEnss895-21 160
1O1TfTrEnss895-31 101
xIWaK92gRlo896-13 317;422;445
xIWaK92gRlo896-24 164;300;793;1254
xIWaK92gRlo896-34 158;330;656;1335
K9MQATj3894898-12 512;1340
K9MQATj3894898-22 19;28
K9MQATj3894898-32 1051;1155
uuCVnqV4cNc891-16 98;1187;1240;1872;1966;2963
uuCVnqV4cNc891-24 1422;2278;2334;2439
uuCVnqV4cNc891-36 11;104;405;1560;1873;2601
5KlS-p5eYH8900-12 1088;1509
5KlS-p5eYH8900-281;47;360;913;1043;1500;1973;2020
5KlS-p5eYH8900-35 338;342;1406;1507;2178
1wzgMHrkrys779-11 2015
1wzgMHrkrys779-22 286;2342
1wzgMHrkryquestion_id   NIF
s779-31 540
Figure 17. The NIF values for each question in Video-MME.
8

video_id timestamp
40BlVzjxu-I006-11 100
40BlVzjxu-I006-23 1;73;90
40BlVzjxu-I006-32 0;1
Qyg_91gNHCc051-13 6;73;78
Qyg_91gNHCc051-22 26;108
Qyg_91gNHCc051-31 75
HvjgQqNOq9A074-14 54;69;79;88
HvjgQqNOq9A074-21 9
HvjgQqNOq9A074-31 1
nYLMNQ77FjM052-11 28
nYLMNQ77FjM052-21 42
nYLMNQ77FjM052-31 78
WViSvPFUVd8079-12 0;1
WViSvPFUVd8079-21 39
WViSvPFUVd8079-36 8;12;18;28;45;50
DI6SemRT2iY357-11 29
DI6SemRT2iY357-21 73
DI6SemRT2iY357-31 270
kSBB5PsRV-k303-12 91;96
kSBB5PsRV-k303-23 86;170;251
kSBB5PsRV-k303-33 86;170;251
8l1NyR6UvxU307-12 40;45
8l1NyR6UvxU307-22 41;79
8l1NyR6UvxU307-31 224
t61Wl2HVwFo388-19 9;12;16;26;34;40;45;56;60
t61Wl2HVwFo388-21 336
t61Wl2HVwFo388-31 579
IaTaaNinolU316-11 12
IaTaaNinolU316-22 269;280
IaTaaNinolU316-31 301
j0J-favyUeQ688-14 61;916;1422;2861
j0J-favyUeQ688-23 924;1821;2854
j0J-favyUeQ688-33 731;1422;2226
zxKPjD8urG4607-13 103;209;1379
zxKPjD8urG4607-21 780
zxKPjD8urG4607-32 1643;1646
FQd5bo9nIZs632-13 98;419;1175
FQd5bo9nIZs632-22 150;199
FQd5bo9nIZs632-31 1175
9bbWYVrQgZ8636-15 103;158;337;538;1755
9bbWYVrQgZ8636-24 1452;1453;1454;1455
9bbWYVrQgZ8636-36 18;887;1022;1656;1745;2054
y2kg3MOk1sY680-15 326;739;825;1706;2281
y2kg3MOk1sY680-24 327;741;2149;2826
y2kg3MOk1sY680-32 1451;1452
tGdL-34L-GE118-11 52
tGdL-34L-GE118-21 40
tGdL-34L-GE118-31 33
43wqf_KhiUo121-18 43;44;45;46;47;48;49;50
43wqf_KhiUquestion_id   NIF
o121-21 1Figure 18. The NIF values for each question in Video-MME.
9

video_id timestamp
43wqf_KhiUo121-3 18;19;20;21;22;23
y6ReUXtm_VE126-1 3;18
y6ReUXtm_VE126-2 4
y6ReUXtm_VE126-3 21
drbi6HK1gSc093-1 3;7;11;21
drbi6HK1gSc093-2 16
drbi6HK1gSc093-3 76
PU-XOFIJMlg098-1 11;17;27
PU-XOFIJMlg098-2 91;96
PU-XOFIJMlg098-3 11
xr_nln2ZQw8412-1 240;243
xr_nln2ZQw8412-2 37;159;260
xr_nln2ZQw8412-3 63;90;326;573;649;717
-XpJeDGh8No395-1 124;131
-XpJeDGh8No395-2 270;338;358;413
-XpJeDGh8No395-3 565;586;598
V6ui161NyTg394-1 10;53;127
V6ui161NyTg394-2 145;152;166;170;316
V6ui161NyTg394-3 145;151;160;167
dTUaWnvIOp4427-1 9;33
dTUaWnvIOp4427-2 11;84;90
dTUaWnvIOp4427-3 146
dcYgBU4t98E393-1 41;42;43
dcYgBU4t98E393-2 199
dcYgBU4t98E393-3 32
p_4UPdFqgIQ725-1 627;686
p_4UPdFqgIQ725-2 627;723
p_4UPdFqgIQ725-3 141;313;416;463;627;921;964;1097
1NH5dJ9VRvU730-1 2343;2344;2406
1NH5dJ9VRvU730-2 66;473;1549
1NH5dJ9VRvU730-3 225
qd2ivr-5oEM721-1 775;830;2093
qd2ivr-5oEM721-2 536;594;603
qd2ivr-5oEM721-3 173;1666
Q8AZ16uBhr8697-1 412
Q8AZ16uBhr8697-2 1418
Q8AZ16uBhr8697-3 2851
xGcfBRkJSWQ708-1 1331;1337
xGcfBRkJSWQ708-2 1090;1102;1280
xGcfBRkJSWQ708-3 2357;2461;2592
rj6rJzs029A142-1 3
rj6rJzs029A142-2 17;18;19;20;21;22;23
rj6rJzs029A142-3 16;18;20;22
zNxi2s36tS0163-1 35
zNxi2s36tS0163-2 3
zNxi2s36tS0163-3 40
PJHkIJZGwKA143-1 10;15
PJHkIJZGwKA143-2 21;22;23;25
PJHkIJZGwKA143-3 4
ya2IXAREZhquestion_id   NIF
o155-1 1;5;6;32;76 
2 
1 
1 
4 
1 
1 
3 
2 
1 
2 
3 
6 
2 
4 
3 
3 
5 
4 
2 
3 
1 
3 
1 
1 
2 
2 
8 
3 
3 
1 
3 
3 
2 
1 
1 
1 
2 
3 
3 
1 
7 
4 
1 
1 
1 
2 
4 
1 
5 4Figure 19. The NIF values for each question in Video-MME.
10

video_id timestamp
ya2IXAREZho155-2 26;29;75
ya2IXAREZho155-3 14;39;63;88
fkJv7LRa6Pc179-1 4;22
fkJv7LRa6Pc179-2 22;30
fkJv7LRa6Pc179-3 50
0Y9-MQ44MdU445-1 34;125
0Y9-MQ44MdU445-2 1422;2278;2334;2439
0Y9-MQ44MdU445-3 193;437;573
Mxkg3qLIPC8451-1 10
Mxkg3qLIPC8451-2 130;156
Mxkg3qLIPC8451-3 286;290
H54zMD-9Q-8437-1 588
H54zMD-9Q-8437-2 33
H54zMD-9Q-8437-3 35
ZQIZx5Oqw88464-1 36;62;92
ZQIZx5Oqw88464-2 257;294
ZQIZx5Oqw88464-3 373;375
2Gg4OQo7-zA456-1 193;585;798
2Gg4OQo7-zA456-2 492;498
2Gg4OQo7-zA456-3 94
0k2ey_okQ4E754-1 79;1416;2197
0k2ey_okQ4E754-2 170;237;273;338
0k2ey_okQ4question_id   NIF
E754-3 562;1119;1522;2617
S_5vlPXLCRc731-1 243;435
S_5vlPXLCRc731-2 231
S_5vlPXLCRc731-3 1186
GV5CuB4zPTY774-1 22
GV5CuB4zPTY774-2 1832;1865;2276
GV5CuB4zPTY774-3 566
jdQ-20JEmgc757-1 739;1491
jdQ-20JEmgc757-2 319;354
jdQ-20JEmgc757-3 435;1557;2041;2166;2316;2563;2705
4H8hcvNeWtg206-1 49
4H8hcvNeWtg206-2 51;58
4H8hcvNeWtg206-3 50
fRf2aYYPrkc194-1 1;23
fRf2aYYPrkc194-2 5;22
fRf2aYYPrkc194-3 1;5;66
k74LDvXSnHM201-1 1
k74LDvXSnHM201-2 25;29
k74LDvXSnHM201-3 29
1q-5IIyZL20197-1 6
1q-5IIyZL20197-2 72
1q-5IIyZL20197-3 93
dH8l--46j6s214-1 64
dH8l--46j6s214-2 43
dH8l--46j6s214-3 11
QopYbLq-zIQ508-1 526;597;685
QopYbLq-zIQ508-2 747
QopYbLq-zIQ508-33 
4 
2 
2 
1 
2 
4 
3 
1 
2 
2 
1 
1 
1 
3 
2 
2 
3 
2 
1 
3 
4 
4 
2 
1 
1 
1 
3 
1 
2 
2 
7 
1 
2 
1 
2 
2 
3 
1 
2 
1 
1 
1 
1 
1 
1 
1 
3 
1 
1 0Figure 20. The NIF values for each question in Video-MME.
11

video_id timestamp
kq9Q9-U0vrc505-12 148;151
kq9Q9-U0vrc505-22 82;243
kq9Q9-U0vrc505-31 3
azZZZbSwLQght490-12 195;265
azZZZbSwLQght490-21 58
azZZZbSwLQght490-31 366
a7jZszvFpTY515-13 131;163;182
a7jZszvFpTY515-22 186;192
a7jZszvFpTquestion_id   NIF
Y515-31 3
323v_FtWqvo483-12 86;220
323v_FtWqvo483-23 79;81;129
323v_FtWqvo483-32 142;220
eQGSbBANfVg803-12 505;521
eQGSbBANfVg803-22 1603;1620
eQGSbBANfVg803-32 1934;1936
yh-EHgkFci4812-13 143;191;311
yh-EHgkFci4812-22 705;906
yh-EHgkFci4812-31 1161
D97vMwfWxvI795-18451;734;1057;1117;1677;1811;1944;2250
D97vMwfWxvI795-26 520;542;1160;1692;2385;2431
D97vMwfWxvI795-31 69
XDq08lk5GnQ794-14 821;1296;1472;2496
XDq08lk5GnQ794-22 1546;1621
XDq08lk5GnQ794-32 2443;2456
P69idA8JO98786-13 152;621;816
P69idA8JO98786-22 1762;1774
P69idA8JO98786-32 1884;1937
Kn10Jf1x24Q273-11 1
Kn10Jf1x24Q273-23 22;23;25
Kn10Jf1x24Q273-31 10
RP1AL2DU6vQ252-13 45;51;92
RP1AL2DU6vQ252-21 24
RP1AL2DU6vQ252-31 55
FlcVBKTSjRs222-11 34
FlcVBKTSjRs222-21 34
FlcVBKTSjRs222-34 1;14;36;38
VnvG08masio239-11 4
VnvG08masio239-21 94
VnvG08masio239-31 22
s-lM2uwiwyQ257-12 10;11
s-lM2uwiwyQ257-23 52;53;58
s-lM2uwiwyQ257-31 73
cFqLEwAvaHI575-11 35
cFqLEwAvaHI575-21 123
cFqLEwAvaHI575-31 159
ZBKUqc_ICpg534-13 33;288;309
ZBKUqc_ICpg534-21 52
ZBKUqc_ICpg534-32 18;19
-c8eATXUui8536-12 58;78
-c8eATXUui8536-23 381;385;416Figure 21. The NIF values for each question in Video-MME.
12

video_id timestamp
-c8eATXUui8536-33 512;521;582
tCRpDpDgBcE553-13 72;77;79
tCRpDpDgBcE553-21 315
tCRpDpDgBcE553-31 509
A30IuIjQYYg561-12 9;114
A30IuIjQYYg561-23 24;35;66
A30IuIjQYYg561-32 30;88
k3zNTrWrbOU827-12 1967;2033
k3zNTrWrbOU827-25 26;291;758;1150;1464
k3zNTrWrbOquestion_id   NIF
U827-33 423;1308;1838
wxff_4tDauo825-11 8
wxff_4tDauo825-24 7;630;1261;1977
wxff_4tDauo825-34 28;267;355;825
wTlERUE8LVw836-12 28;62
wTlERUE8LVw836-23 791;796;800
wTlERUE8LVw836-33 686;772;2297
aqFfjJrLkBA834-17 20;21;24;28;34;36;40
aqFfjJrLkBA834-21 11
aqFfjJrLkBA834-33 386;941;1419
elprD1hnDyU886-13 861;902;937
elprD1hnDyU886-26 514;640;694;1024;1085;1700
elprD1hnDyU886-32 2212;2610
AIs3LoU4JUo297-12 22;117
AIs3LoU4JUo297-22 2;93
AIs3LoU4JUo297-32 74;117
nbOXpuv7K4Q294-13 25;28;29
nbOXpuv7K4Q294-24 17;20;28;42
nbOXpuv7K4Q294-31 36
PYZSjin_Pe8293-12 59;60
PYZSjin_Pe8293-11 5
PYZSjin_Pe8293-12 79;103
bHt0Riqz0qo296-11 5
bHt0Riqz0qo296-23 35;54;64
bHt0Riqz0qo296-32 9;84
nVtTVt9csBc291-15 14;17;19;38;90
nVtTVt9csBc291-23 13;20;21
nVtTVt9csBc291-33 0;3;80
0JjgoicpYkU599-11 20
0JjgoicpYkU599-24 159;217;436;504
0JjgoicpYkU599-31 527
cVIfe0Gxa64595-11 0
cVIfe0Gxa64595-21 249
cVIfe0Gxa64595-31 245
InHaW59CmDw592-14 16;168;271;509
InHaW59CmDw592-22 391;659
InHaW59CmDw592-32 512;538
GrO5sxp3n0E594-11 28
GrO5sxp3n0E594-22 399;409
GrO5sxp3n0E594-31 534
fFYNmVb3NCQ598-13 193;236;422Figure 22. The NIF values for each question in Video-MME.
13

taskvideo_id question timestamp
1NIF
_plotQAmovie101_55What is the first expression of1 151
1_plotQAmovie101_55Why is the person in the gray s2 3;5
1_plotQAmovie101_56What color is the man's clothes1 164
1_plotQAmovie101_56What is on the woman's table th1 188
1_plotQAmovie101_56What is the man's emotion at th1 99
1_plotQAmovie101_10
4What is the girl's reaction aft1 4
1_plotQAmovie101_10What is the girl with glasses' 2 60;61
1_plotQAmovie101_10Why does the girl with glasses 3 276;300;314
1_plotQAxiaoliyu_3What did the cartoon dragon do 3 211;258;268
1_plotQAxiaoliyu_3What did the cartoon dragon tur1 301
1_plotQAxiaoliyu_3What did the cartoon snake plac1 36
1_plotQAhaimian_7What does the Cartoon Sponge do1 123
1_plotQAhaimian_7What do the other cartoon anima1 191
1_plotQAhaimian_7Why did the Cartoon Sponge turn2 180;181
1_plotQAhaimian_7Why is the Cartoon Octopus angr1 394
1_plotQAmovie101_14What color are the pants worn b1 160
1_plotQAmovie101_14What color is the bag on the ta1 155
1_plotQAmovie101_14What color is the long skirt wo1 108
1_plotQAmovie101_14What color is the suit worn by 1 19
1_plotQAmovie101_15What color is the animal that a1 1
1_plotQAmovie101_15What color is the hair of the p1 44
1_plotQAmovie101_15What color is the suit the girl1 395
1_plotQAmovie101_15What color is the table in the 1 106
1_plotQAmovie101_34How many candles are lit in the1 260
1_plotQAmovie101_34What color is the girl's hair i1 66
1_plotQAmovie101_34What color is the hair of the b1 622
1_plotQAmovie101_34What color is the hoodie that t1 187
1_plotQAmovie101_36In the scene where two people a1 120
1_plotQAmovie101_36What color is the hat worn by t1 136
1_plotQAmovie101_36What color is the scarf worn by1 58
1_plotQAmovie101_36What color is the top worn by t1 2
1_plotQAtomjerry_10What does the cartoon big mouse2 148;154
1_plotQAtomjerry_10What is the cartoon big mouse t2 176;243
1_plotQAtomjerry_10Why did the cartoon little mous2 246;248
1_plotQAtomjerry_10Why does the cartoon little mou3 33;56;61
2_needleneedle_1What is the backdrop of the bas1 148
2_needleneedle_1Where is the basketball court l1 148
2_needleneedle_13What are the volunteers doing i1 340
2_needleneedle_13What are the volunteers searchi1 340
2_needleneedle_13What are the volunteers doing t2 340;342
2_needleneedle_41What animal is sitting very sti1 416
2_needleneedle_41What is the American toad doing2 352;358
2_needleneedle_41What is the nature of the den w1 351
2_needleneedle_41What is the state of movement o3 351;368;376
2_needleneedle_51What animals are tied beside th1 475
2_needleneedle_51What are the two kids doing in 1 476
2_needleneedle_51What nationality are the kids h1 478
2_needleneedle_78What part of the doctor's face 1 506
2_needleneedle_86What is the direction of the bo1 132
2_needleneedle_86What is the kid doing with the 1 132Figure 23. The NIF values for each question in MLVU.
14

taskvideo_id question timestamp
2NIF
_needleneedle_86What is the mood of the boy wal2 134;147
2_needleneedle_109What does the engineer begin to1 803
2_needleneedle_113What are the volunteers searchi1 5264
2_needleneedle_119What is happening to the net on1 603
2_needleneedle_119What is the backdrop of the bas1 603
2_needleneedle_119What is the weather condition o1 603
2_needleneedle_119Where is the basketball court l1 603
2_needleneedle_149What logo is displayed on the s1 306
2_needleneedle_149Where is the Goldman Sachs Grou1 306
3_egoego_10What colour is the stool I sat 1 408
3_egoego_13How many green cups were on the1 310
3_egoego_13How many picture frames were on1 293
3_egoego_13In what location did I see the 1 205
3_egoego_13What color is the towel on the 1 319
3_egoego_13Where is the pack of Jenga game1 57
3_egoego_16Did I leave the door of the sec2 474;477
3_egoego_16In what room did I see the blac1 453
3_egoego_16Where was the blue chair? 1 377
3_egoego_16Where was the blue poly bag?1 377
3_egoego_16Where was the water bottle?1 427
3_egoego_17Did I cut the wood plank? 2 367;373
3_egoego_17What color was the measuring ta1 307
3_egoego_21Where did I put the jenga box?1 49
3_egoego_21Where was the dust pan? 3 168;171;178
3_egoego_28Did I attached the drill into t2 458;461
3_egoego_28Did I throw the drill on the gr2 453;454
3_egoego_28How many boxes did I pick up? 1 389
3_egoego_28Where did I keep the drill?2 454;457
3_egoego_28Where was the square bucket bef1 387
3_egoego_35Did I leave the front door open2 183;184
3_egoego_35What colour was the bottle I pr1 304
3_egoego_35What did I put in the orange tr2 169;171
3_egoego_35Where did I put the blue helmet1 193
3_egoego_35Where did I put the leftover pa2 270;277
3_egoego_35Where was the cat after I put f2 348;359
3_egoego_39Did I leave the car bonnet open1 34
3_egoego_39What did I remove from the box?2 260;261
3_egoego_53What did I put in Dustbin ?2 132;135
3_egoego_76What the green t-shirt man was 1 272
3_egoego_76Where was the wooden bamboo?1 252
3_egoego_76Who did I talk to at the garage1 285
4_countcount_1In this video, how many times d3 222;267;469
4_countcount_14In this video, how many instanc2 5;295
4_countcount_69In this video, how many instanc1 193
4_countcount_91In this video, how many instanc547;169;221;359;477
4_countcount_102Throughout this video, what is 4207;409;1220;1381
4_countcount_112In this video, how many times d3 84;304;321
4_countcount_115In this video, how many instanc1 94
4_countcount_117In this video, how many instanc2 212;314
4_countcount_130In this video, how many instanc2 216;455Figure 24. The NIF values for each question in MLVU.
15

taskvideo_id question timestamp
4_countcount_165Throughout this video, what is 31
5NIF
1
_orderorder_2Arrange the following events fr4 33;51;93;160
5_orderorder_10Arrange the following events fr6188;414;422;523;542;549
5_orderorder_14Arrange the following events fr5378;431;482;486;565
5_orderorder_41Arrange the following events fr6243;251;255;266;279;362
5_orderorder_114Arrange the following events fr5377;380;425;566;592
5_orderorder_128Arrange the following events fr4 98;130;148;214
5_orderorder_133Arrange the following events fr5525;529;591;635;636
5_orderorder_160Arrange the following events fr84572;4573;4576;4582;4625;
4630;4645;4691
5_orderorder_254Please identify the option that4 7;205;375;433
5_orderorder_260Can you tell me which option re4382;598;1761;2093
6_anomaly_
recosurveil_0Does this surveillance footage 3 216;424;504
6_anomaly_
recosurveil_28Is there any abnormality in thi694;136;177;324;754;872
6_anomaly_
recosurveil_98Are there any irregularities in3 53;72;301
6_anomaly_
recosurveil_101Does this surveillance footage 1 106
6_anomaly_
recosurveil_103Does this surveillance footage 6154;196;864;2980;3290;353
7
6_anomaly_
recosurveil_107Does this surveillance footage 3 79;137;163
6_anomaly_
recosurveil_115Is there any abnormality in thi2 108;522
6_anomaly_
recosurveil_125Does this surveillance footage 3 36;52;126
6_anomaly_
recosurveil_140Is there any abnormality in thi3 84;90;272
6_anomaly_
recosurveil_159Are there any irregularities in63;30;53;114;194;400
7_topic_re
asoning203What is the main scene in the v3 35;73;177
7_topic_re
asoning203What type of video is this?2 133;477
7_topic_re
asoning231What scenery is mainly shown in5102;128;135;190;216
7_topic_re
asoningAWA-16What is the setting of the vide2 8;37
7_topic_re
asoningAWD-1What is the background of the v3 29;79;340
7_topic_re
asoningAWD-1What type of video is this?4 75;222;226;345
7_topic_re
asoningen_tv_15In what kind of setting does th3 46;136;198
7_topic_re
asoningen_tv_15What genre of movie is the clip2 325;341
7_topic_re
asoninggame_14What is the object being built 1 10Figure 25. The NIF values for each question in MLVU.
16

taskvideo_id question timestam NIF p
7_topic_re
asoningmovie101_10What color is the hat worn by t1 15
7_topic_re
asoningmovie101_10What color is the scarf worn by1 251
7_topic_re
asoningmovie101_26What is the genre of this movie632;41;56;195;410;747
7_topic_re
asoningtomjerry_11What character appears the most756;74;107;109;116;131;169
7_topic_re
asoningxiaoliyu_2What is the background of the v4 85;271;279;293
7_topic_re
asoningxiaoliyu_2What is the protagonist in the 3 37;40;343
8_sub_scen
esubPlot_new
_all_0Please describe what happened w7842;843;844;860;862;866;8
98
8_sub_scen
esubPlot_new
_all_51Please describe the process of 8149;163;171;175;180;196;1
99;203
8_sub_scen
esubPlot_new
_all_97Please describe the situation a0 N/A
8_sub_scen
esubPlot_new
_all_103Please describe the situation w3 605;608;612
8_sub_scen
esubPlot_new
_all_108Please describe the action of t3 232;233;236
8_sub_scen
esubPlot_new
_all_113Please describe in detail what 3 37;39;115
8_sub_scen
esubPlot_new
_all_130Please describe what the man in41731;1733;1737;1740
8_sub_scen
esubPlot_new
_all_136What did the girl do after hail4 168;169;177;179
8_sub_scen
esubPlot_new
_all_172Can you describe how the man wi3 308;329;341
8_sub_scen
esubPlot_new
_all_196Please describe the situation w3 22;26;27
9_summary 9Can you summarize the main cont170;48;57;91;115;136;241;24
2;302;306;316;347;412;416
;442;447;468
9_summaryAWD-5Could you provide a summary of 140;12;20;68;89;108;115;140
;186;195;257;308;313;365
9_summaryAWH-4Can you provide a summary of th1411;15;20;23;28;64;123;137
;164;296;324;376;378;437
9_summaryen_tv_27Can you summarize the main cont101;4;55;57;87;89;234;237;4
08;415
9_summaryen_tv_29Could you provide a summary of 90;81;106;174;355;364;410;
424;425
9_summarymovie101_1Can you summarize the main cont1014;18;84;89;156;172;181;1
90;199;212
9_summarymovie101_58Can you provide a summary of th1517;44;57;68;212;239;290;3
64;421;443;459;468;487;52
3;547Figure 26. The NIF values for each question in MLVU.
taskvideo_id question timestamp
9NIF
_summarymovie101_10
8Can you provide a summary of th110;14;33;43;120;121;178;18
1;263;311;318
9_summarytomjerry_2Can you summarize the main cont1232;36;38;120;127;128;154;
168;224;235;255;262
9_summaryxiaoliyu_5Could you provide a summary of 200;1;8;55;57;58;77;83;143;
149;189;203;219;260;343;3
50;352;382;406;412
Figure 27. The NIF values for each question in MLVU.
17

video_idquestion_idNIF timestamp
video_222 0 1 13
video_312 0 3 11;17;23
video_312 1 3 11;17;23
video_312 2 1 23
video_312 3 1 11
video_312 4 2 2;8
video_312 5 1 13
video_557 0 1 0
video_557 1 3 11;12;13
video_557 2 3 11;12;13
video_557 0 3 10;13;14
video_557 1 1 32
video_557 2 1 27
video_557 3 3 10;13;14
video_766 0 1 11
video_766 1 2 10;12
video_812 0 1 29
video_812 1 1 29
video_812 2 3 4;9;15
video_812 3 3 4;9;15
video_921 0 72;7;11;16;20;25;31
video_921 1 72;7;11;16;20;25;31
video_921 2 1 2
video_921 3 1 7
video_921 4 1 11
video_980 0 2 0;1
video_980 1 5 11;13;17;19;28
video_1200 0 2 11;35
video_1200 1 1 35
video_1200 2 1 35
video_1200 3 1 21
video_1200 4 3 2;8;11
video_1261 0 1 33
video_1261 1 1 33
video_1261 2 2 6;33
video_1261 3 1 33
video_1261 4 6 8;9;10;12;13;14
video_1261 5 1 33
video_1261 6 1 13
video_1262 0 1 33
video_1262 1 1 22
video_1599 0 1 0
video_1830 0 1 18
video_2246 0 1 29
video_2246 1 1 29
video_2250 0 3 0;28;31
video_2250 1 2 0;31
video_2250 2 1 31
video_2250 3 1 0
video_2250 4 1 24Figure 28. The NIF values for each question in Perception Test.
18

video_idquestion_idNIF timestamp
video_2250 5 1 24
video_2280 0 1 0
video_2298 0 1 7
video_2369 0 2 30;32
video_2369 1 3 3;14;28
video_2959 0 1 0
video_2959 1 1 20
video_2959 2 1 10
video_2959 3 2 10;20
video_2959 4 2 10;20
video_2959 5 3 14;16;20
video_2959 6 1 10
video_2959 7 1 5
video_2959 8 1 10
video_2959 9 1 10
video_2999 0 1 0
video_2999 1 1 31
video_2999 2 1 31
video_3145 0 3 0;1;2
video_3145 1 1 2
video_3451 0 1 0
video_3451 1 1 18
video_3451 2 1 26
video_3464 0 1 0
video_3464 1 1 5
video_3619 0 2 15;22
video_3619 1 1 22
video_3768 0 3 0;2;4
video_3768 1 2 0;32
video_3768 2 1 32
video_3768 3 1 0
video_3768 4 1 32
video_3768 5 1 32
video_3928 0 1 24
video_4171 0 3 0;1;2
video_4171 1 1 1
video_4306 0 1 0
video_4306 1 1 1
video_4306 2 1 32
video_4306 3 1 0
video_4306 4 1 16
video_4306 5 1 16
video_4459 0 1 33
video_4459 1 1 20
video_4810 0 1 26
video_4810 1 1 26
video_4810 2 1 26
video_4846 0 4 0;5;14;31
video_4846 1 1 5
video_4846 2 1 5Figure 29. The NIF values for each question in Perception Test.
19

video_idquestion_idNIF timestamp
video_4846 3 1 5
video_4859 0 1 0
video_4859 1 1 13
video_4906 0 1 1
video_4991 0 3 0;1;2
video_4991 1 1 1
video_5015 0 3 0;1;2
video_5015 1 1 24
video_5015 2 1 14
video_5015 3 1 24
video_5308 0 1 5
video_5308 1 4 4;5;17;31
video_5308 2 1 5
video_5308 3 1 5
video_5308 4 1 5
video_5439 0 4 4;5;16;31
video_5439 1 1 5
video_5439 2 1 0
video_5439 3 1 5
video_5689 0 3 0;1;2
video_5875 0 1 1
video_5963 0 4 3;4;15;31
video_5963 1 1 4
video_5963 2 1 0
video_5963 3 1 4
video_5970 0 71;6;12;17;21;25;31
video_5970 1 71;6;12;17;21;25;31
video_5970 2 1 1
video_5970 3 1 1
video_5970 4 1 6
video_5970 5 1 12
video_6164 0 3 0;1;2
video_6164 1 1 1
video_6223 0 1 26
video_6223 1 1 17
video_6223 2 1 26
video_6249 0 82;5;9;13;17;23;27;33
video_6249 1 82;5;9;13;17;23;27;33
video_6249 2 1 2
video_6249 3 1 5
video_6249 4 1 9
video_6321 0 3 0;4;9
video_6321 1 2 0;9
video_6321 2 1 9
video_6321 3 1 0
video_6321 4 1 18
video_6321 5 1 18
video_6418 0 1 34
video_6418 1 1 0
video_6418 2 1 34Figure 30. The NIF values for each question in Perception Test.
20

video_idquestion_idNIF timestamp
video_6418 3 1 18
video_6418 4 2 18;34
video_6418 5 2 18;34
video_6418 6 5 23;25;29;32;34
video_6418 7 1 18
video_6418 8 1 9
video_6418 9 1 18
video_6418 10 1 10
video_6626 0 3 18;22;25
video_6736 0 6 3;8;15;21;26;31
video_6736 1 6 3;8;15;21;26;31
video_6736 2 1 3
video_6736 3 1 3
video_6736 4 1 8
video_6829 0 5 3;10;19;25;31
video_6829 1 5 3;10;19;25;31
video_6829 2 1 3
video_6829 3 1 3
video_6829 4 1 10
video_6829 5 1 19
video_6842 0 2 5;13
video_6842 1 1 33
video_6842 2 1 7
video_6896 0 71;6;11;15;20;25;29
video_6896 1 71;6;11;15;20;25;29
video_6896 2 1 1
video_6896 3 1 6
video_6896 4 1 11
video_7092 0 1 0
video_7092 1 1 8
video_7272 0 1 0
video_7318 0 3 0;1;2
video_7318 1 3 0;26;32
video_7318 2 2 0;32
video_7318 3 1 32
video_7318 4 1 0
video_7318 5 1 16
video_7318 6 1 16
video_7453 0 1 0
video_7453 1 1 30
video_7472 0 1 5
video_7609 0 3 0;1;2
video_7609 1 3 3;8;10
video_7609 2 3 3;8;10
video_7609 3 3 3;8;10
video_7609 4 2 3;30
video_7609 5 2 30;31
video_7609 6 1 3
video_7609 7 1 3
video_7704 0 71;7;12;17;23;28;33Figure 31. The NIF values for each question in Perception Test.
21

video_idquestion_idNIF timestamp
video_7704 1 71;7;12;17;23;28;33
video_7704 2 1 1
video_7704 3 1 1
video_7704 4 1 7
video_7704 5 1 12
video_8245 0 1 0
video_8245 1 1 1
video_8245 2 1 33
video_8245 3 1 0
video_8245 4 2 14;20
video_8245 5 1 21
video_8551 0 3 0;1;2
video_8551 1 1 22
video_8551 2 1 22
video_8635 0 3 0;18;33
video_8635 1 2 0;33
video_8635 2 1 33
video_8635 3 1 0
video_8635 4 1 22
video_8635 5 1 22
video_8679 0 1 2
video_8679 1 1 2
video_8679 2 1 2
video_8679 3 1 0
video_8732 0 1 0
video_8732 1 1 0
video_8732 2 2 12;16
video_8732 3 2 5;6
video_8732 4 1 12
video_8735 0 4 6;7;8;9
video_8735 1 3 17;21;26
video_8735 2 3 7;10;16
video_8735 3 1 4
video_8735 4 3 7;10;16
video_8888 0 2 10;11
video_8952 0 3 6;16;28
video_8952 1 1 6
video_8952 2 3 6;16;28
video_8952 3 1 0
video_8952 4 1 0
video_8952 5 1 33
video_8952 6 3 6;16;28
video_8975 0 1 0
video_8975 1 1 29
video_8975 2 1 16
video_8975 3 2 16;29
video_8975 4 2 16;29
video_8975 5 2 16;29
video_8975 6 3 23;26;29
video_8975 7 1 16Figure 32. The NIF values for each question in Perception Test.
22

video_idquestion_idNIF timestamp
video_8975 8 1 6
video_8975 9 1 16
video_8975 10 1 8
video_8995 0 3 0;1;2
video_8995 1 2 19;20
video_9008 0 1 0
video_9008 1 1 33
video_9008 2 1 20
video_9008 3 2 20;33
video_9008 4 2 20;33
video_9008 5 2 27;29
video_9008 6 1 20
video_9008 7 1 9
video_9008 8 1 20
video_9008 9 1 11
video_9342 0 3 0;1;2
video_9342 1 1 2
video_9431 0 3 0;1;2
video_9431 1 1 0
video_9431 2 2 14;20
video_9431 3 5 6;14;17;20;27
video_9431 4 1 24
video_9691 0 3 0;11;32
video_9691 1 2 0;32
video_9691 2 1 32
video_9691 3 1 0
video_9691 4 1 15
video_9691 5 1 15
video_9727 0 4 5;10;18;25
video_9727 1 1 5
video_9727 2 4 5;10;18;25
video_9727 3 1 0
video_9727 4 4 5;10;18;25
video_9727 5 4 5;10;18;25
video_9727 6 4 5;10;18;25
video_9772 0 1 33
video_9772 1 1 33
video_9772 2 1 33
video_9772 3 1 33
video_9772 4 1 33
video_9772 5 1 33
video_9799 0 1 22
video_9799 1 1 26
video_10028 0 3 0;1;2
video_10028 1 3 5;7;16
video_10076 0 1 0
video_10159 0 74;8;13;16;19;23;28
video_10159 1 74;8;13;16;19;23;28
video_10159 2 1 4
video_10159 3 1 8Figure 33. The NIF values for each question in Perception Test.
23

video_idquestion_idNIF timestamp
video_10159 4 1 13
video_10212 0 3 0;1;2
video_10212 1 1 0
video_10212 2 1 0
video_10212 3 1 0
video_10212 4 2 7;30
video_10212 5 1 30
video_10212 6 1 7
video_10212 7 1 0
video_10307 0 3 0;1;2
video_10307 1 1 17
video_10307 2 1 33
video_10386 0 4 7;15;17;23
video_10505 0 3 0;1;2
video_10505 1 1 0
video_10505 2 1 1
video_10505 3 1 28
video_10600 0 1 0
video_10600 1 1 6
video_10664 0 1 1
video_10664 1 6 1;3;5;7;11;18
video_10664 2 3 27;28;29
video_10690 0 2 9;10
video_10690 1 2 9;10
video_10944 0 1 0
video_10944 1 6 6;8;11;14;19;24
video_10944 2 6 6;8;11;14;19;24
video_10944 3 1 8
video_11253 0 3 0;1;2
video_11306 0 1 2
video_11306 1 1 23
video_11331 0 70;7;12;16;20;25;31
video_11331 1 70;7;12;16;20;25;31
video_11331 2 1 1
video_11331 3 1 0
video_11331 4 1 7
video_11331 5 1 12
video_11457 0 3 4;10;13
video_11457 1 3 25;30;33
video_11457 2 1 25
video_11457 3 3 4;10;13Figure 34. The NIF values for each question in Perception Test.
24

video_id NIF timestamp
0a8109fe-15b9-4f5c-b5f2-993013cb216b1 138
46853bef-9052-428d-8e61-df684147f4af2 38;131
b4cc9985-97e8-423a-9737-22e5d9b4dbce4 4;13;28;118
0a8b2c9d-b54c-4811-acf3-5977895d24451 32
e9322513-15b2-4b89-8ddb-7d1432beb8a14 20;25;28;32
b1a1280a-1f7f-4796-bca2-ba03f3fb93453 0;24;73
05f1fc03-0c9e-4fd4-9d85-bb7be4e692342 13;34
9de66400-05ec-4173-93c9-16c2cc9d881d3 27;56;153
55ea09ed-4590-4e59-8753-40a64d67abd91 166
6472e377-b65c-461a-a750-9b28a673dc863 19;40;139
901ae1fe-5be2-495b-9506-9ec2a28d8ae02 69;138
cd384dae-4229-4fa5-9fed-e4b5a1432f293 7;67;169
71d00225-7ea5-46a0-8015-9b5a667f619a4 3;49;120;143
425f2fb4-a2a7-4925-94b5-25f6b0b85f783 17;33;53
340a76ff-7144-4b31-906f-8a43ed866bc01 155
b58ab03f-3520-4916-81b8-2c42e3d0d31d1 39
0aadf5ce-07eb-4934-9e07-317a46bc0b211 16
93a92b6f-5ed2-4b2c-9f9c-a6307e1fb2563 4;12;136
cd2e4351-de59-4511-ab43-36c37b388a8d1 96
509e6545-5fff-4f73-ae0f-524dfa8b3c2c2 11;22
24f4b88e-2294-4017-a669-9e27c07d44e71 167
86d91c31-1bfe-4f99-a803-7466c8d801d11 66
d68218d2-5071-458d-8e4d-87f5707b7fbc3 34;38;120
a0a00d56-0f2d-4f3d-ae12-ee5bc4c7ba191 143
0688f66e-f115-49c6-85ff-712bf4f4a7583 13;30;125
1eb2f153-055f-4004-ad55-154359af80252 1;35
01a144a5-24d2-4a5a-af01-1f318d674bed1 39
824c2b85-b40b-4bbd-82bf-70468f9c042b3 16;35;37
5fe9843b-0b74-4505-b864-86eb53c25cc61 79
e27e9ec7-aaf3-4e5c-a387-1699fe66ea4f3 28;116;127
c66fe71d-e9c3-4983-ad77-26c0a8b1c0b91 109
e86bb89c-baf4-4463-8941-e296d1d4d62f1 140
b5867202-c87b-4ffa-8617-e8b2e9eba1a21 164
2faa1516-ed55-4a96-a4be-09c402cf2c761 74
13da1294-2b42-4ef6-8dd6-ff651ef4571f3 10;14;20
7ad240de-34ab-4694-a6be-05a47e14793f2 40;41
e614f1a5-7c7b-468f-9840-d7373f7402551 95
bcf9aba0-a4b6-4210-8823-025f43f2631f1 20
2e22aafd-1fbb-4e73-ab6f-d8f628b66ba11 29
84bd1e04-370a-4e4a-9255-776f8d8e38ad1 56Figure 35. The NIF values for each question in EgoSchema.
25

video_id NIF timestamp
90c3f31b-3b44-4b9a-a684-cef313a45c322 35;78
217fe8d0-dfc8-407b-86be-269378c5259a2 2;104
631682a5-5574-41a8-904f-7ee96fc936832 2;52
628e252f-e743-4054-92fd-b0ed6983571d2 22;91
51688142-10e7-48ab-adef-2caa5448b4561 180
45e313dd-9b30-444e-9577-b15a21cb59b44 10;31;66;91
86de4235-953e-458f-a951-40314e92a33b1 83
aebf3455-59b7-4071-a7d2-18d053c38f8f1 1
dfb6c468-e124-40f6-9c4e-c13ee45a2ad91 76
d0bbd7fa-2a15-4c25-ab06-adc7d04bce7b3 9;19;64
e48b8359-d35e-45f1-aa3b-eb1417e10dc83 11;12;29
97dc6bb7-7eed-45f7-bdb0-269ab8c2f6394 1;60;71;99
81ba0fd6-cc69-410d-9e2d-8317fd22cce83 12;37;60
b5d7c421-2b86-4ed0-b314-ce810c778c471 7
2b8d1e50-3ba7-492a-8a0b-104eb659c27b2 110;170
4b7495f1-e2c2-4d69-bbf4-c2dabeb5e6342 13;104
0d173aa3-9a94-4ba4-84bc-949d3254a63d4 41;45;48;134
cd882b7a-0766-4582-8388-3990b009b11b4 3;23;101;131
4070a4e2-14d5-4618-9889-dd18a416e2b52 4;148
68e0bb20-414d-42ef-a0a0-e821efbe8e064 20;22;37;106
9796f529-40ca-4e74-89ed-6a25efb24c8c3 76;78;79
223164c8-abed-4f1a-8f7c-4088c89d3ece1 26
97f80e1c-164d-4072-8ee9-980e366eec6c2 0;28
ece69b04-2e67-434a-b923-1329feed590d2 5;31
e00836f3-1506-4479-a028-e17f19cff0bf1 79
94f6e8bd-b65d-4fd0-b2a9-75b69397fe2e2 16;31
e04cd624-17e1-4986-b344-55aa92d7c0c31 50
5782e464-db9e-4bee-88f7-6c2f6727854b2 9;15
86cb525a-d61c-46f1-9c5f-cc72841849003 85;136;156
b3aaaea6-e6f6-499f-8b03-ddf85b3f62c41 12
22b86648-340c-4338-b46e-5eaba3a44b062 106;119
cc3abb84-1317-4718-b414-75f899c20ee32 12;30
e67de76c-1058-49a7-a47e-12736da4ffc02 0;27
f95e7f60-0f9a-40e7-bb60-55ecb287b2dc2 20;22
1bf1fdea-6f44-4d3a-b5c0-6852aaada71b2 35;36
9d83c60a-5d84-4bea-8325-56beed585df21 34
0e4804e0-85fa-48bc-ada3-a94167b06e531 4
6f42ba6a-cb2a-428f-bffb-7a15abd947273 9;22;114
04c51dba-1dcb-4b8f-a62c-efc363561d7b1 31
ee227b56-c12b-4725-89e9-aa29e0b4dbe81 14
803c8ecf-9448-48b6-8bf2-debd052dbe431 8
1bd933df-3575-4fa7-839f-765c7108259e2 20;106
c04da37a-b98f-4796-afe2-1b7d3af209116104;110;117;124;129;160
9748f410-2316-4a2f-9893-56f8f240dc673 34;88;152
0f181d98-5036-4990-b397-62e934e168ef3 2;12;21
640ad606-0376-48cb-bc6a-14bc34ec4eaa3 10;21;35
36424b90-8a32-44b3-8db1-ad8bc3b222d04 17;40;76;104
fcf8719c-b32d-463f-aa4c-6ac4149bb1c02 67;166
6c901d03-3413-41a8-9aa0-8f9f6ed6b6f13 2;6;97
5d15c716-f179-462c-84a6-fa94e9d14e941 86Figure 36. The NIF values for each question in EgoSchema.
26