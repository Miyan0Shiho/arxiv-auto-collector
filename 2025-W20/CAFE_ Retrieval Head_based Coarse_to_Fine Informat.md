# CAFE: Retrieval Head-based Coarse-to-Fine Information Seeking to Enhance Multi-Document QA Capability

**Authors**: Han Peng, Jinhao Jiang, Zican Dong, Wayne Xin Zhao, Lei Fang

**Published**: 2025-05-15 08:05:12

**PDF URL**: [http://arxiv.org/pdf/2505.10063v1](http://arxiv.org/pdf/2505.10063v1)

## Abstract
Advancements in Large Language Models (LLMs) have extended their input
context length, yet they still struggle with retrieval and reasoning in
long-context inputs. Existing methods propose to utilize the prompt strategy
and retrieval head to alleviate this limitation. However, they still face
challenges in balancing retrieval precision and recall, impacting their
efficacy in answering questions. To address this, we introduce $\textbf{CAFE}$,
a two-stage coarse-to-fine method to enhance multi-document question-answering
capacities. By gradually eliminating the negative impacts of background and
distracting documents, CAFE makes the responses more reliant on the evidence
documents. Initially, a coarse-grained filtering method leverages retrieval
heads to identify and rank relevant documents. Then, a fine-grained steering
method guides attention to the most relevant content. Experiments across
benchmarks show CAFE outperforms baselines, achieving up to 22.1% and 13.7%
SubEM improvement over SFT and RAG methods on the Mistral model, respectively.

## Full Text


<!-- PDF content starts -->

arXiv:2505.10063v1  [cs.CL]  15 May 2025CAFE: Retrieval Head-based Coarse-to-Fine Information Seeking to
Enhance Multi-Document QA Capability
Han Peng1*, Jinhao Jiang1*, Zican Dong1*, Wayne Xin Zhao1†, Lei Fang2,
1Gaoling School of Artificial Intelligence, Renmin University of China.
2DataCanvas Alaya NeW.
{panospeng, jiangjinhao, dongzican}@ruc.edu.cn
batmanfly@gmail.com
Abstract
Advancements in Large Language Models
(LLMs) have extended their input context
length, yet they still struggle with retrieval
and reasoning in long-context inputs. Existing
methods propose to utilize the prompt strategy
and retrieval head to alleviate this limitation.
However, they still face challenges in balancing
retrieval precision and recall, impacting their ef-
ficacy in answering questions. To address this,
we introduce CAFE , a two-stage coarse-to-fine
method to enhance multi-document question-
answering capacities. By gradually eliminating
the negative impacts of background and dis-
tracting documents, CAFE makes the responses
more reliant on the evidence documents. Ini-
tially, a coarse-grained filtering method lever-
ages retrieval heads to identify and rank rele-
vant documents. Then, a fine-grained steering
method guides attention to the most relevant
content. Experiments across benchmarks show
CAFE outperforms baselines, achieving up to
22.1% and 13.7% SubEM improvement over
SFT and RAG methods on the Mistral model,
respectively.
1 Introduction
Researchers have undertaken various efforts to ex-
tend the context length of Large Language Models
(LLMs), ranging from advancements in model ar-
chitectures (Yen et al., 2024; Munkhdalai et al.,
2024) to optimizations in training methods (Fu
et al., 2024b; An et al., 2024; Xiong et al., 2024;
Dong et al., 2025). These developments have
enabled some recently introduced LLMs to sup-
port relatively long context inputs ( i.e.,128K con-
text length for LLaMA-3.1 (Dubey et al., 2024)
and Qwen-2.5 (Yang et al., 2024), and even 10M
context length for Gemini (Reid et al., 2024)).
However, recent studies indicate that LLMs ex-
hibit limitations in retrieval and reasoning capabil-
*Equal Contribution.
†Corresponding author.
8k 16k 32k 64k 128k
Document Length4045505560657075SubEM Score (%)
Llama-3.1-8B-Instruct-128K
Mistral-3-7B-Instruct-32K
Phi-3.5-Mini-Instruct-128K
gold evidenceFigure 1: LLMs’ performance on HotpotQA varies with
the number of input documents. Solid lines represent
performance with the gold document, while dashed lines
show performance as more documents are added.
ity when processing the long context input (Liu
et al., 2024; Lee et al., 2024; Wang et al., 2024; Li
et al., 2024c), which poses significant challenges
for their effective application in downstream tasks,
including book summarization (Bai et al., 2024a),
multi-document question answering (Zhang et al.,
2024c), and code repository understanding (Bai
et al., 2024b).
To investigate performance bottlenecks, we con-
duct preliminary experiments on long-context re-
trieval and reasoning, focusing on multi-document
question answering, a representative task recently
studied (Zhu et al., 2024; Hsieh et al., 2024). Our
findings reveal that the performance of LLMs sig-
nificantly declines as the number of additional in-
put documents increases, compared to using only
the necessary documents ( i.e.,, gold documents), as
shown in Figure 1. This prompts a natural question:
how can we mitigate the impact of these additional
documents in long-context input?
To address this question, existing stud-
ies (Agrawal et al., 2024; Zhang et al., 2024b) pri-
marily employ external prompt strategies to guide

models in first extracting key information related
to a question from the long input and then utiliz-
ing them along with the original context, to an-
swer the question. However, its effectiveness is
constrained by the LLMs’ instruction-following
capabilities (Xu et al., 2024), which is amplified
in long inputs. To address this issue, subsequent
research (Qiu et al., 2025) analyses the model’s
internal attention mechanism, utilizing the retrieval
head to identify key information. As shown in
Table 1, our experiments also indicate that the at-
tention head outperforms the external prompt strat-
egy. Nonetheless, these methods face challenges in
balancing retrieval precision and recall, impacting
their efficacy in answering questions. Specifically,
enhancing recall introduces more irrelevant infor-
mation during the model’s reasoning process, while
increasing retrieval precision reduces recall.
To address these challenges, we first rethink that
humans typically do not search across all docu-
ments; instead, they would gradually identify the
relevant ones step-by-step and then reason over
them. For example, individuals first filter out irrel-
evant documents to create a manageable candidate
set, then further analyze and utilize the most rele-
vant documents to answer the question. Inspired
by this, we aim to design a strategy that guides
LLMs to gradually filter out irrelevant documents
from the long-context input, and then further en-
hance their ability to utilize the remaining relevant
documents to answer questions.
According to the above motivation, we pro-
pose CAFE , a novel two-stage coarse-to-fine
information-seeking method to enhance the
multi-document question-answering capabilities of
LLMs. Its core idea lies in a fine-grained utiliza-
tion of retrieval heads based on specific contexts in
multiple stages. Specifically, before information-
seeking, we pre-locate the retrieval heads for the
two stages respectively on the validation set. Then,
in the first stage, we implement a coarse-grained
filtering approach to filter out background docu-
ments. We identify relevant documents assigned
with high attention scores in each pre-detected re-
trieval head and further rerank these documents
according to the summed scores from all retrieval
heads. In the second stage, we guide the model
using a fine-grained steering approach. We utilize
another set of retrieval heads to further select rel-
evant documents from these reranked documents,
and employ attention steering on the most relevant
content to answer the final questions. In this way,we can guide the LLMs to gradually search for
evidence documents in the long context input and
utilize them to better answer the questions. Ad-
ditionally, the whole method is training-free and
applicable to a wide range of downstream tasks.
We conducted extensive experiments to evaluate
the proposed CAFE method using various LLMs.
The results demonstrate that our method consis-
tently outperforms existing strong baselines across
five benchmarks and three LLMs ( e.g., achieving
an 11.4% relative performance improvement com-
pared to the supervised fine-tuning method).
2 Related Work
Long-Context Utilization in Language Models.
Although extended the context length of LLMs suc-
cessfully (Dubey et al., 2024; Yang et al., 2024;
Dong et al., 2024), they still face significant chal-
lenges ( e.g., long-term decay (Chen et al., 2024)
and lost-in-the-middle (Liu et al., 2024)) in uti-
lizing long contexts effectively for complex tasks.
To enhance the long-context utilization capacities,
attention-based methods leverage the property of
attention heads and positional encodings, enlarging
the attention scores of the key tokens over the long
inputs (Wu et al., 2024; Gema et al., 2024). Dif-
ferent from previous methods, our work employs
a training-free two-stage framework, which iden-
tifies relevant documents and guild the response
more dependent on these documents.
Retrieval Head in Attention Mechanisms. Re-
cent studies have revealed specialized attention
heads in LLMs that exhibit retrieval capabilities
for locating critical information within long con-
texts, namely, retrieval heads (Wu et al., 2024). In
these heads, high attention values will be assigned
to the tokens most relevant to the current token in
the long inputs, achieving in-context retrieval of
previous information. Recently, some work retains
the full attention on the retrieval heads and employs
KV Cache compression on other heads to acceler-
ate the calculation (Fu et al., 2024a; Tang et al.,
2024; Xiao et al., 2024a). Different from them, our
method utilizes retrieval heads as a retrieval system
to identify evidence documents.
Retrieval-Augmented Generation. Retrieval-
Augmented Generation (RAG) has been widely
adopted to address various NLP tasks. For multi-
document question-answering tasks, traditional
RAG methods utilize external dense or sparse re-

Method HotpotQA-8k HotpotQA-16k HotpotQA-32k SQuAD Musique
ICR 0.64 0.51 0.38 0.91 0.58
Attention-Based 0.80 0.78 0.77 0.94 0.79
Table 1: Recall scores for different evidence selection strategies across various datasets using Llama-3.1-8B-Instruct.
trieval models to compute the similarity of docu-
ments with the question (Robertson and Zaragoza,
2009; Karpukhin et al., 2020). Then, relevant docu-
ments are retrieved as the input for models. Beyond
leveraging external models to retrieve documents,
several in-context retrieval methods have been pro-
posed (Agrawal et al., 2024; Li et al., 2024a). These
methods prompt the models to select the indices
of relevant documents. Unlike existing RAG ap-
proaches, our work leverages the model’s inherent
retrieval capabilities to perform a coarse-to-fine lo-
cation of evidence documents, effectively enhanc-
ing its retrieval and reasoning abilities.
3 Empirical Study
In this section, we conduct empirical studies to
analyze how to improve the retrieval and reasoning
capabilities of LLMs for multi-document question
answering from two aspects, i.e.,evidence selection
and attention intervention.
3.1 Evidence Selection
In this study, we examine document retrieval in
multi-document question answering using two pri-
mary methods: in-context retrieval and attention-
based retrieval. The in-context retrieval method
prompts LLMs to directly select the top- kdocu-
ments most relevant to the given question, while
attention-based retrieval method utilizes the at-
tention distribution over each document to per-
form selection. Subsequently, the selected doc-
uments are employed as condensed input in the
question-answering prompt. The results, presented
in Table 1, demonstrate that the attention-based
approach significantly outperforms the in-context
retrieval method, particularly as document length
increases (e.g., recall score decreases by 50.6% on
HotpotQA-32K). More experimental results can be
found in Appendix C.
3.2 Attention Intervention
Through the above experiments, we find that us-
ing attention heads can effectively help the model
to retrieve the most relevant documents from long
inputs. In this section, we further explore howMask Mode SubEM
No Mask 60.5
Evidence 2→Evidence 1 60.0
Question →Evidence 1 48.0
Question →Evidence 2 42.0
Question →Evidence 1, Evidence 2 29.5
Question →One Irrelevant Document 60.0
Question →Two Irrelevant Documents 59.5
Table 2: SubEM scores on HotpotQA-8K with various
masking strategies using Llama-3.1-8B-Instruct, where
Evidence 1andEvidence 2refer to the first and second
gold documents in the context.
to facilitate the model’s utilization of multiple re-
trieved documents. Specifically, we select Hot-
potQA test samples with input lengths within 8K
tokens. Based on the attention interaction, we mask
the attention between the two gold documents, as
well as mask the attention of the questions to the
two gold documents, respectively. We show the
results in Table 2. First, masking the attention
between the two gold documents has negligible
impact on performance compared to the unmasked
condition (60.0% for Evidence 2→Evidence 1vs.
60.5% for No Mask). This suggests that during
multi-hop question answering, the LLM does not
engage in implicit reasoning while encoding long
inputs, aligning with observations from recent stud-
ies (Yu and Ananiadou, 2024). Implicit reasoning
in our work denotes the model’s capacity to inte-
grate information across documents during the pre-
filling phase, before the question is posed. By using
attention interactions, the model embeds pertinent
content from earlier documents into later ones, al-
lowing it to retrieve the answer directly from the
final document without revisiting prior texts.
Moreover, when applying attention masks to
question-irrelevant documents, we observed min-
imal performance impact. Additionally, masking
the attention from the question to any gold docu-
ment results in a significant performance drops.
When all gold documents are masked simulta-
neously, the SubEM score even decreases to a
level similar to that observed when no document
is provided (29.5% for Question →Evidence 1,
Evidence 2vs. 23.0% for Closed Book). This

Retrieval Head Scor e : 0.5For Current Head
Doc1Doc2Doc3Doc4Doc5Doc6Doc7Doc8Doc90.20.3
 SUM
Question
Total Heads: Num of Layers * Num of HeadsLayer 15, Head 30
Question: Were Scott Derrickson and
Ed Wood of the same nationality?Retrieval Document Scor e
Doc1 Doc2 Doc3DociSorted by Retrieval
Document ScoresRetrieval Document Scor e
Question: Were Scott Derrickson and
Ed Wood of the same nationality?
Attention Steering
Final Response: Yes
Coarse-Grained Filtering Fine-Grained Steering
Doc1 Doc3 Doc2 Doc4
Layer
Head
Retrieval Heads (Stage-2)Layer
Head
Retrieval Heads (Stage-1)
Layer 15, Head 30
Layer 14, Head 13
Layer 13, Head 18
Layer 16, Head 21Layer 15, Head 30
Layer 10, Head 30
Layer 13, Head 18
Layer 10, Head 29Question: Onika Tanya Maraj is a
judge on a television show hosted
by whom?Retrieval Head Scor eFigure 2: Overall framework of our proposed CAFE approach. The red, blue, and yellow bar charts represent the
gold, distracting, and background documents, respectively.
demonstrates that the directly attention from the
question to the gold documents plays a critical role
in the LLM’s overall performance.
Building on the preceding experiments, attention
heads can be employed to assist the LLM in retriev-
ing pertinent documents from long inputs. In ad-
dition, By adjusting the attention directed towards
these documents, the model’s ability to utilize them
in answering questions is enhanced. These princi-
ples will guide the design of our method.
4 Method
4.1 Overall Framework
In multi-document question-answering tasks, there
are three categories of documents, i.e.,gold ev-
idence documents that contain information sup-
porting answering the questions, distracting docu-
ments that impede the model’s ability to generate
faithful answers, and background documents that
contain irrelevant information. Among them, The
latter two categories of documents increase the text
length and introduce noise, which further weaken
the model’s capabilities. Thus, we propose CAFE ,
a coarse-to-fine two-stage framework to enhance
the long-context question-answering capacities by
gradually eliminating the negative impacts of back-
ground and distracting documents. In our frame-work, we identify retrieval heads to locate back-
ground and distracting documents from the input
set. Given that the two stages rely on different con-
text information, we use different retrieval heads
for each. We first apply coarse-grained filtering
to remove background documents, then use fine-
grained attention steering to reduce the influence
of distracting ones. This two-stage process helps
the model focus on gold evidence. The overall
illustration is shown in Figure 2.
4.2 Retrieval Head Detection
In Section 3, we observe that the attention scores
can effectively identify evidence documents. Addi-
tionally, there exist some heads where the attention
scores of tokens in the question usually focus on to-
kens within the relevant content, namely, retrieval
heads (Wu et al., 2024). Leveraging the properties,
we first identify retrieval heads that can be further
employed to seek the relevant documents.
Retrieval Document Scores. Based on the analy-
sis in Section 3.2, we focus on the attention distribu-
tion from the question to the contextual documents.
Therefore, we first compute the retrieval docu-
ment score βh(di)by analyzing attention weight
distributions between the question qand each doc-

ument di:
βh(di) =αh(q, di)Pn
j=1αh(q, dj), (1)
where αh(q, di)represents the attention weight be-
tween the query qand document difor attention
headh, andnis the total number of documents in
the current sample.
Top-KRetrieval Heads Selection. To effectively
identify retrieval heads, we select Nsamples from
the validation set and calculate a retrieval head
score for each attention head hbased on the ev-
idence documents’ retrieval document scores on
these validation samples:
η(h) =NX
i=1X
e∈Eiβh(e), (2)
where Eiis the set of evidence documents for the
i-th sample. Subsequently, we select the Top- K
attention heads Hretwith the highest retrieval head
scores from the heads from all layers Has the
retrieval heads.
Hret=Top-K(η(h)), h∈ H. (3)
Notably, during the coarse-grained filtering and
fine-grained steering stages, we employ different
validation sets and select different retrieval heads
according to the properties of the two stages. The
distinction between the two types of retrieval heads
is detailed in Appendix D.
4.3 Coarse-Grained Filtering for Background
Documents
In Figure 1, we observe that a large amount of back-
ground documents leads to significant performance
degradation. Thus, we introduce a coarse-grained
filtering stage to filter background documents and
obtain a condensed input. Specifically, this stage
consists of two steps: background document filter-
ing and locality-based re-ranking.
Background Documents Filtering. To identify
background documents, we first compute the re-
trieval document scores of each document on se-
lected retrieval heads Hret. For each head h, we
select Top- M1documents based on the retrieval
document scores βh(d)from all documents Dand
consider them as relevant documents. Then, we
perform a union operation on these documents toobtain the relevant document set D∗and drop the
other documents.
D∗=[
h∈H retTop-M1(βh(d)), d∈ D.(4)
Locality-Based Re-Ranking. When processing
long context, LLMs usually demonstrate the prop-
erty of locality and lost-in-the-middle (Liu et al.,
2024; Su et al., 2024). This means when critical
information for answering the question is located
at the end of the long document, the model often
performs better. Thus, after obtaining the filtered
set of documents D∗, we apply a locality-based re-
ranking mechanism to rank these documents. For
the filtered candidate document set D∗, we com-
pute the document relevance score γh(d)for each
document as the sum of retrieval document scores
of all retrieval heads:
γh(d) =X
h∈H retβh(d), d∈ D∗. (5)
Subsequently, documents with higher document rel-
evance scores are positioned later in the sequence,
ensuring that more attention will be focused on the
documents that are more likely to contain critical
evidence during the generation of responses. Fi-
nally, we obtain the filtered and reranked document
sequence D′as the input of next stage:
D′={d′
1, . . . , d′
|D∗|},∀i < j, γ h(di)≤γh(dj).
(6)
4.4 Fine-Grained Steering for Distracting
Text
After the first stage of filtering background-
irrelevant documents, though the remaining docu-
ments usually contain question-relevant informa-
tion, there may still be distracting documents that
are not useful for answering the question. Thus, in
the fine-grained steering stage, we further identify
these distracting documents and steering the atten-
tion scores on them to reduce the influence of these
documents on the generation of responses.
Iterative Distracting Document Identification.
Similar to the coarse-grained filtering stage, to ef-
fectively identify and weaken the impact of these
distracting documents, we perform document iden-
tification by computing retrieval document scores
using another set of retrieval heads H′
ret:

LCLM BaselineSQuAD MuSiQue HotpotQA HotpotQA-16K HotpotQA-32K
SubEM F1 SubEM F1 SubEM F1 SubEM F1 SubEM F1
Llama-3.1-8BOracle RAG 92.5 86.4 39.0 39.3 76.5 76.8 76.5 76.8 76.5 76.8
Directly Answering 71.0 66.6 30.5 33.2 60.5 62.5 53.0 60.1 53.5 58.1
In-Context Retrieval 73.5 65.1 28.0 29.2 59.0 58.6 51.5 51.8 42.5 42.2
Vanilla RAG 84.5 76.6 28.0 29.7 64.0 64.8 63.0 63.6 61.5 62.4
SFT 69.0 70.1 33.5 38.9 63.0 69.8 62.5 68.0 61.5 67.4
CAFE (w/o FGS) 89.5 80.7 36.0 35.5 68.5 69.0 66.0 68.3 66.0 65.2
CAFE (ours) 89.5 82.6 36.5 36.5 70.0 70.4 69.0 69.0 68.5 68.1
Mistral-3-7BOracle RAG 84.0 80.1 40.5 38.9 67.0 71.3 67.0 71.3 67.0 71.3
Directly Answering 59.0 55.9 27.5 26.8 50.0 53.7 45.0 47.5 39.0 46.6
In-Context Retrieval 59.5 58.7 24.0 24.2 49.0 47.6 37.5 38.2 29.5 30.3
Vanilla RAG 69.5 69.2 27.5 26.2 53.5 55.9 53.5 55.4 51.0 54.7
SFT 60.0 60.1 30.5 33.1 57.5 61.9 52.5 56.7 47.5 53.6
CAFE (w/o FGS) 78.0 73.6 30.0 27.9 60.0 64.0 60.5 60.0 53.0 56.5
CAFE (ours) 78.5 75.2 31.0 29.9 61.5 65.2 60.5 61.7 58.0 61.7
Phi-3.5-MiniOracle RAG 85.0 80.0 35.0 38.1 73.0 75.8 73.0 75.8 73.0 75.8
Directly Answering 63.5 58.8 24.5 27.5 55.0 55.5 51.5 52.5 48.0 48.3
In-Context Retrieval 65.5 66.4 22.5 23.7 49.5 49.5 38.0 39.5 31.0 34.4
Vanilla RAG 76.0 72.5 25.5 26.1 58.5 60.2 56.0 58.8 55.0 58.7
SFT 64.5 65.1 34.5 40.9 60.5 71.8 61.0 71.8 58.0 67.3
CAFE (w/o FGS) 82.0 74.9 28.5 28.8 65.0 67.8 64.5 62.6 60.0 58.9
CAFE (ours) 84.5 75.8 30.0 31.9 66.5 68.0 66.5 64.8 61.5 60.1
Table 3: Evaluation results on three long-document question answering tasks. They are representative of single-hop
and multi-hop question-answering tasks. “CAFE (w/o FGS)” means that we only perform the fist stage without the
fine-grained steering for distracting text stage. The bold andunderline fonts denote the best and second best results
in each dataset. Notably, all models in the table are the instruct versions.
Dcand=[
h∈H′
retTop-M2(βh(d)), d∈ D′(7)
By identifying documents with high retrieval docu-
ment scores, we ultimately derive a candidate set
of evidence documents Dcand. Each document in
the candidate set is considered the golden evidence
while other documents are considered as distracting
documents during the following process of atten-
tion steering.
Inference-Time Attention Steering. After the
initial filtering stage, the number of remaining
documents is significantly reduced. In this stage,
directly removing detected distractors may result
in lower recall of evidence documents. Thus, in-
stead of only keeping the candidate set, we adopt
post-hoc attention steering (Zhang et al., 2024a),
an inference-only technique that reweights atten-
tion scores to guide the model’s focus toward user-
specified input spans. Specifically, given the can-
didate gold evidence set Dcand, our method empha-
sizes specific tokens by adding a constant attention
biasBhto the attention scores on tokens withinthese documents across all attention heads.
˜Ah= Softmax(( Qh⊺Kh+Bh)/√
d),(8)
Bh
ij=(
δifi∈qandj∈ C cand
0otherwise, (9)
where δis a positive constant that controls the
degree of attention adjustment. After applying
Softmax (·), the attention scores of tokens in Dcand
are enlarged while the attention scores of other to-
kens are reduced. This dynamic reweighting mech-
anism effectively enhances the model’s attention
toward tokens in Dcand, ensuring the responses are
more dependent on the critical evidence.
5 Experiments
5.1 Experimental Setup
Datasets. We evaluate the long-context perfor-
mance of our approach and baseline methods using
three question-answering datasets: SQuAD (Ra-
jpurkar et al., 2016), HotpotQA (Yang et al., 2018),
and MusiQue (Trivedi et al., 2022). These datasets
are collected from the RULER (Hsieh et al., 2024)

and LongBench (Bai et al., 2024a) benchmarks.
Additionally, we experiment with three versions of
HotpotQA that vary in context length to analyze
how model performance changes with text length.
To ensure consistency across all baselines and our
approach, we randomly select 200 samples from
each dataset to form the final test set. All experi-
ments are conducted using the same test sets.
Baselines and Metrics. For evaluation, we use
Substring Exact Match (SubEM) and F1 scores fol-
lowing existing work (Li et al., 2024b). SubEM
measures whether the gold answer appears as a
substring in the predictions, while the F1 score
evaluates the token-level overlap between predic-
tions and references. For compared baselines, we
select five types of methods, including Directly An-
swering ,In-Context Retrieval ,Oracle RAG ,Vanilla
RAG , and Supervised Fine-tuning . We present the
detailed description in Appendix B.
Implementation Details. We conduct our experi-
ments on three open-source models: Llama-3.1-8B-
Instruct, Mistral-3-7B-Instruct, and Phi-3.5-Mini-
Instruct. For coarse-grained filtering for back-
ground documents, we set the Top-M1to 4 and
Top-K1to 4. For fine-grained steering for distract-
ing text, we set the Top-M2to 2 and Top-K2to
{1,2,3,4} and we set δ= log 10 . As for the SFT
configuration, training is conducted with a batch
size of 64 and a learning rate of 1×10−5. We
set the number of training rounds to 1, as multiple
rounds resulted in overfitting with the limited data
available.
5.2 Main Results
Table 3 shows the results of our methods and other
baselines across three representative long context
question-answering datasets.
Firstly, our method achieves significantly better
multi-document question-answering performances
than other baselines. Across all three datasets,
our method consistently outperforms training-free
approaches and even surpasses the SFT method
in most settings. On single-hop SQuAD, our
method can achieve performances nearly the per-
formance ceiling introduced by Oracle RAG. On
more complex multi-hop question-answering tasks,
our method can still achieve a significant perfor-
mance improvement ( e.g., approximately 19.9%of
SubEM scores on the HotpotQA dataset compared
to the naive directly answering method).
Secondly, the two stages of our method work to-Method Llama Mistral Phi
CAFE 70.0 61.5 66.5
w/o CGF 62.5 52.0 55.0
w/o FGS 68.5 60.0 65.0
w/o Re-Ranking 68.0 59.5 65.5
Table 4: Ablation study on HotpotQA.
Granularity Recall SubEM F1
w/o Steering - 68.5 69.0
Sentence-Level 0.89 65.5 67.8
Document-Level 0.93 70.0 70.4
Table 5: Results with different steering granularities.
gether to prompt performance improvements. Com-
pared with in-context retrieval and vanilla RAG
which retrieve relevant documents via prompting
techniques or external models, only employing the
coarse-grained filtering stage can greatly boost the
performance, indicating that leveraging the inner re-
trieval heads can more effectively identify relevant
documents. Additionally, introducing fine-grained
attention steering can further boost long-context
question-answering capacities, which demonstrates
the necessity of introducing a fine-grained elimi-
nation of the negative impacts of distracting docu-
ments on multi-document question answering.
Finally, our method exhibits less performance
drop with longer input lengths. On the HotpotQA
dataset, we assess the performances across differ-
ent input lengths. Our method can preserve perfor-
mance to a greater extent when dealing with longer
texts ( e.g.,decreases 1.4%and2.1%SubEM scores
for Llama-3.1-8B on 16K and 32K). Instead, the
performances drop sharply with the length increas-
ing with other methods, especially in-context re-
trieval ( e.g., decreases 12.7%and28.0%SubEM
scores for Llama-3.1-8B on 16K and 32K). This
indicates that our method can effectively identify
the critical documents in the long input, scarcely
affected by the increased number of documents.
5.3 Further Analysis
Ablation Study. To assess the effectiveness of
our framework, we conduct ablation experiments
focusing on key steps within the pipeline. (1)
w/o Coarse-Grained Filtering (CGF) eliminates
the initial coarse-grained filtering of background
documents; (2) w/o Fine-Grained Steering (FGS)
omits the fine-grained steering of distracting text,

Model Method 1 10 20 30 40 50 Rand
LLaMA-3.1-8B-InstructDA 77.5/70.9 74.5/67.4 73.0/67.6 70.5/64.6 69.5/64.4 73.0/67.6 71.0/66.6
Ours 90.5/82.1 91.0/80.3 89.5/80.9 89.0/80.5 88.5/79.9 89.5/79.2 89.5/82.6
Mistral-3-7B-InstructDA 70.0/58.7 59.0/50.6 56.5/48.3 59.5/51.6 58.0/52.9 62.0/59.5 59.0/55.9
Ours 79.5/76.0 80.0/74.8 79.0/71.9 78.5/71.1 78.5/71.2 78.0/72.6 78.5/75.2
Table 6: Position-wise SubEM/F1 scores on two models. The column headers (1, 10, 20, etc) indicate the document
index where the gold document is inserted. DA denotes Directly Answering.
relying solely on documents D′for inference; (3)
w/o Locality-Based Re-Ranking bypasses locality-
based re-ranking in the first stage, resulting in the
use of filtered documents in a random order.
The results are presented in Table 4. All variants
show inferior performance compared to the origi-
nal method, underscoring the effectiveness of each
component in our framework. Notably, the absence
of Coarse-Grained Filtering ( w/o CGF ) results in a
substantial performance decline, highlighting the
critical role of first-stage filtering in excluding ir-
relevant background documents and preventing the
dilution of the model’s attention. Similarly, the re-
moval of Fine-Grained Steering ( w/o FGS ) leads to
decreased performance, indicating that the second
stage’s attention steering effectively mitigates the
impact of distracting documents. Furthermore, the
exclusion of Re-Ranking ( w/o Re-Ranking ) results
in significant performance degradation, demonstrat-
ing the effectiveness of putting the essential infor-
mation at the end of the input, to facilitate retrieval
and reasoning of models.
Impact of Hyperparameters. The choice of hyper-
parameters M(documents per head) and K(num-
ber of heads) during retrieval head selection has a
strong influence on both recall and overall perfor-
mance. As shown in Figure 3, increasing M1or
K1boosts recall by adding more candidates, but
can also introduce noise that limits final accuracy.
We therefore fix M1= 4 andK1= 4 for consis-
tency. A similar trade-off holds in the second stage,
though performance remains stable after attention
steering. The remaining hyperparameter details are
provided in Appendix E.
Granularity of Attention Steering. In the fine-
grained steering stage, we also evaluate the im-
pact of granularity of attention steering. Instead
of document-level, we identify relevant contexts
at sentence-level and steer the attention scores on
these sentences. As shown in Table 5, the recall
at sentence level is lower compared to document
K1=4K1=8M1=2 M1=40.89 0.92
0.97 0.99Recall(stage-1)
K1=4K1=8M1=2 M1=467.00 69.00
66.50 68.50SubEM(stage-1)
K2=1K2=2M2=2 M2=40.86 0.90
0.92 0.93Recall(stage-2)
K2=1K2=2M2=2 M2=469.50 70.00
69.50 70.50SubEM(stage-2)Figure 3: The impact of hyperparameters M(docu-
ments per retrieval head) and K(retrieval heads) on
Llama-3.1-8B-Instruct. The top row shows recall and
performance for coarse-grained filtering, while the bot-
tom row illustrates changes for fine-grained steering.
level. Additionally, the final performances degrade
significantly, even inferior to that before attention
steering. This indicates the importance of cover-
ing the golden evidence information as much as
possible during the attention steering stages.
Lost-in-the-Middle Performance. We investigate
the Lost-in-the-Middle phenomenon and the effec-
tiveness of our method in mitigating it. Experi-
ments are conducted on the SQuAD dataset using
LLaMA and Mistral, evaluating how the position
of the answer within a set of 50 documents af-
fects model performance. As shown in Table 6,
the Lost-in-the-Middle phenomenon significantly
degrades the baseline method’s performance, par-
ticularly when answers are in middle positions (e.g.,
Mistral’s SubEM score drops from 70% to 58%).
Our method effectively mitigates the issue, achiev-
ing stable and significantly improved performance
across all answer positions, consistently outper-
forming the baseline. This approach demonstrates

strong robustness and generalizability, requiring no
position-specific adjustments.
6 Conclusion
In this paper, we explored the challenges faced
by LLMs in handling long-context inputs, particu-
larly in multi-document question answering tasks.
Our findings revealed that the inclusion of irrele-
vant documents significantly hampers the retrieval
and reasoning capabilities of LLMs, motivating
the need for more effective long-context process-
ing strategies. To address this, we introduced
CAFE, a two-stage coarse-to-fine information-
seeking method that leverages retrieval head-based
filtering, document reranking, and fine-grained at-
tention steering to guide LLMs in processing long-
context inputs. Extensive experiments across mul-
tiple benchmarks and LLMs validate its effective-
ness, demonstrating its superiority over strong base-
lines, including supervised fine-tuning techniques.
Beyond its performance benefits, CAFE’s training-
free nature and broad applicability make it a practi-
cal solution for a wide range of downstream tasks.
Limitations
In this paper, we present a coarse-to-fine two-stage
framework to enhance retrieval and reasoning ca-
pacities of LLMs. Beyond multi-document ques-
tion answering tasks, we believe our framework can
be employed in broader tasks, e.g., long-document
reasoning, which have not been explored owing to
the computational costs. Additionally, our method
mainly focus on how to better identify evidence
documents to enhance performances. However,
though given the golden evidence, the LLMs can
still hardly to answer each question correctly. Ap-
proaches of improving the context-aware reasoning
capacities can be employed to further improve the
upper limit of our method.
References
Devanshu Agrawal, Shang Gao, and Martin Gajek. 2024.
Can’t remember details in long documents? you
need some r&r. In Findings of the Association for
Computational Linguistics: EMNLP 2024, Miami,
Florida, USA, November 12-16, 2024 , pages 12692–
12704. Association for Computational Linguistics.
Shengnan An, Zexiong Ma, Zeqi Lin, Nanning Zheng,
Jian-Guang Lou, and Weizhu Chen. 2024. Make
your LLM fully utilize the context. In Advances in
Neural Information Processing Systems 38: AnnualConference on Neural Information Processing Sys-
tems 2024, NeurIPS 2024, Vancouver, BC, Canada,
December 10 - 15, 2024 .
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024a. Longbench: A bilingual, mul-
titask benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers), ACL 2024, Bangkok, Thailand, Au-
gust 11-16, 2024 , pages 3119–3137. Association for
Computational Linguistics.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024b.
Longbench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks. CoRR ,
abs/2412.15204.
Yuhan Chen, Ang Lv, Jian Luan, Bin Wang, and Wei Liu.
2024. Hope: A novel positional encoding without
long-term decay for enhanced context awareness and
extrapolation. CoRR , abs/2410.21216.
Zican Dong, Junyi Li, Jinhao Jiang, Mingyu Xu,
Wayne Xin Zhao, Bingning Wang, and Weipeng
Chen. 2025. Longred: Mitigating short-text degra-
dation of long-context large language models via
restoration distillation. CoRR , abs/2502.07365.
Zican Dong, Junyi Li, Xin Men, Xin Zhao, Bingning
Wang, Zhen Tian, Weipeng Chen, and Ji-Rong Wen.
2024. Exploring context window of large language
models via decomposed positional vectors. In Ad-
vances in Neural Information Processing Systems
38: Annual Conference on Neural Information Pro-
cessing Systems 2024, NeurIPS 2024, Vancouver, BC,
Canada, December 10 - 15, 2024 .
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman,
Akhil Mathur, Alan Schelten, Amy Yang, Angela
Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang,
Archi Mitra, Archie Sravankumar, Artem Korenev,
Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien
Rodriguez, Austen Gregerson, Ava Spataru, Bap-
tiste Rozière, Bethany Biron, Binh Tang, Bobbie
Chern, Charlotte Caucheteux, Chaya Nayak, Chloe
Bi, Chris Marra, Chris McConnell, Christian Keller,
Christophe Touret, Chunyang Wu, Corinne Wong,
Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Al-
lonsius, Daniel Song, Danielle Pintz, Danny Livshits,
David Esiobu, Dhruv Choudhary, Dhruv Mahajan,
Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes,
Egor Lakomkin, Ehab AlBadawy, Elina Lobanova,
Emily Dinan, Eric Michael Smith, Filip Radenovic,
Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Geor-
gia Lewis Anderson, Graeme Nail, Grégoire Mialon,
Guan Pang, Guillem Cucurell, Hailey Nguyen, Han-
nah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov,
Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan
Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan

Geffert, Jana Vranes, Jason Park, Jay Mahadeokar,
Jeet Shah, Jelmer van der Linde, Jennifer Billock,
Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi,
Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu,
Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph
Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia,
Kalyan Vasuden Alwala, Kartikeya Upasani, Kate
Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and
et al. 2024. The llama 3 herd of models. CoRR ,
abs/2407.21783.
Tianyu Fu, Haofeng Huang, Xuefei Ning, Genghan
Zhang, Boju Chen, Tianqi Wu, Hongyi Wang, Zix-
iao Huang, Shiyao Li, Shengen Yan, Guohao Dai,
Huazhong Yang, and Yu Wang. 2024a. Moa: Mix-
ture of sparse attention for automatic large language
model compression. CoRR , abs/2406.14909.
Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Han-
naneh Hajishirzi, Yoon Kim, and Hao Peng. 2024b.
Data engineering for scaling language models to 128k
context. In Forty-first International Conference on
Machine Learning, ICML 2024, Vienna, Austria, July
21-27, 2024 . OpenReview.net.
Aryo Pradipta Gema, Chen Jin, Ahmed Abdulaal, Tom
Diethe, Philip Teare, Beatrice Alex, Pasquale Min-
ervini, and Amrutha Saseendran. 2024. Decore: De-
coding by contrasting retrieval heads to mitigate hal-
lucinations. CoRR , abs/2410.18860.
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shan-
tanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang,
and Boris Ginsburg. 2024. RULER: what’s the real
context size of your long-context language models?
CoRR , abs/2404.06654.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,
and Wen-tau Yih. 2020. Dense passage retrieval for
open-domain question answering. In Proceedings of
the 2020 Conference on Empirical Methods in Nat-
ural Language Processing, EMNLP 2020, Online,
November 16-20, 2020 , pages 6769–6781. Associa-
tion for Computational Linguistics.
Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua,
Devendra Singh Sachan, Michael Boratko, Yi Luan,
Sébastien M. R. Arnold, Vincent Perot, Siddharth
Dalmia, Hexiang Hu, Xudong Lin, Panupong Pasu-
pat, Aida Amini, Jeremy R. Cole, Sebastian Riedel,
Iftekhar Naim, Ming-Wei Chang, and Kelvin Guu.
2024. Can long-context language models subsume
retrieval, rag, sql, and more? CoRR , abs/2406.13121.
Huayang Li, Pat Verga, Priyanka Sen, Bowen Yang,
Vijay Viswanathan, Patrick Lewis, Taro Watanabe,
and Yixuan Su. 2024a. Alr2: A retrieve-then-
reason framework for long-context question answer-
ing. CoRR , abs/2410.03227.
Siheng Li, Cheng Yang, Zesen Cheng, Lemao Liu,
Mo Yu, Yujiu Yang, and Wai Lam. 2024b. Large
language models can self-improve in long-context
reasoning. CoRR , abs/2411.08147.Yanyang Li, Shuo Liang, Michael R. Lyu, and Liwei
Wang. 2024c. Making long-context language models
better multi-hop reasoners. In Proceedings of the
62nd Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), ACL
2024, Bangkok, Thailand, August 11-16, 2024 , pages
2462–2475. Association for Computational Linguis-
tics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language
models use long contexts. Trans. Assoc. Comput.
Linguistics , 12.
Tsendsuren Munkhdalai, Manaal Faruqui, and Sid-
dharth Gopal. 2024. Leave no context behind:
Efficient infinite context transformers with infini-
attention. CoRR , abs/2404.07143.
Yifu Qiu, Varun Embar, Yizhe Zhang, Navdeep Jaitly,
Shay B. Cohen, and Benjamin Han. 2025. Eliciting
in-context retrieval and reasoning for long-context
large language models. CoRR , abs/2501.08248.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100, 000+ questions
for machine comprehension of text. In Proceedings
of the 2016 Conference on Empirical Methods in
Natural Language Processing, EMNLP 2016, Austin,
Texas, USA, November 1-4, 2016 , pages 2383–2392.
The Association for Computational Linguistics.
Machel Reid, Nikolay Savinov, Denis Teplyashin,
Dmitry Lepikhin, Timothy P. Lillicrap, Jean-Baptiste
Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan
Firat, Julian Schrittwieser, Ioannis Antonoglou, Ro-
han Anil, Sebastian Borgeaud, Andrew M. Dai, Katie
Millican, Ethan Dyer, Mia Glaese, Thibault Sotti-
aux, Benjamin Lee, Fabio Viola, Malcolm Reynolds,
Yuanzhong Xu, James Molloy, Jilin Chen, Michael
Isard, Paul Barham, Tom Hennigan, Ross McIl-
roy, Melvin Johnson, Johan Schalkwyk, Eli Collins,
Eliza Rutherford, Erica Moreira, Kareem Ayoub,
Megha Goel, Clemens Meyer, Gregory Thornton,
Zhen Yang, Henryk Michalewski, Zaheer Abbas,
Nathan Schucher, Ankesh Anand, Richard Ives,
James Keeling, Karel Lenc, Salem Haykal, Siamak
Shakeri, Pranav Shyam, Aakanksha Chowdhery, Ro-
man Ring, Stephen Spencer, Eren Sezener, and et al.
2024. Gemini 1.5: Unlocking multimodal under-
standing across millions of tokens of context. CoRR ,
abs/2403.05530.
Stephen E. Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: BM25 and be-
yond. Found. Trends Inf. Retr. , pages 333–389.
Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng
Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: En-
hanced transformer with rotary position embedding.
Neurocomputing , 568:127063.
Hanlin Tang, Yang Lin, Jing Lin, Qingsen Han, Shikuan
Hong, Yiwu Yao, and Gongyi Wang. 2024. Razo-

rattention: Efficient KV cache compression through
retrieval heads. CoRR , abs/2407.15891.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. Musique: Multi-
hop questions via single-hop question composition.
Trans. Assoc. Comput. Linguistics , 10.
Minzheng Wang, Longze Chen, Fu Cheng, Shengyi
Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan
Xu, Lei Zhang, Run Luo, Yunshui Li, Min Yang, Fei
Huang, and Yongbin Li. 2024. Leave no document
behind: Benchmarking long-context llms with ex-
tended multi-doc QA. In Proceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing, EMNLP 2024, Miami, FL, USA,
November 12-16, 2024 , pages 5627–5646. Associa-
tion for Computational Linguistics.
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao
Peng, and Yao Fu. 2024. Retrieval head mecha-
nistically explains long-context factuality. CoRR ,
abs/2404.15574.
Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian
Guo, Shang Yang, Haotian Tang, Yao Fu, and Song
Han. 2024a. Duoattention: Efficient long-context
LLM inference with retrieval and streaming heads.
CoRR , abs/2410.10819.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024b. C-
pack: Packed resources for general chinese embed-
dings. In Proceedings of the 47th International ACM
SIGIR Conference on Research and Development in
Information Retrieval, SIGIR 2024, Washington DC,
USA, July 14-18, 2024 , pages 641–649. ACM.
Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang,
Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi
Rungta, Karthik Abinav Sankararaman, Barlas Oguz,
Madian Khabsa, Han Fang, Yashar Mehdad, Sharan
Narang, Kshitiz Malik, Angela Fan, Shruti Bhosale,
Sergey Edunov, Mike Lewis, Sinong Wang, and Hao
Ma. 2024. Effective long-context scaling of founda-
tion models. In Proceedings of the 2024 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), NAACL 2024,
Mexico City, Mexico, June 16-21, 2024 , pages 4643–
4663. Association for Computational Linguistics.
Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee,
Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina
Bakhturina, Mohammad Shoeybi, and Bryan Catan-
zaro. 2024. Retrieval meets long context large lan-
guage models. In The Twelfth International Con-
ference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net.
An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayi-
heng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian
Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang,
Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang,Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei
Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men,
Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren,
Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang,
Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and
Zihan Qiu. 2024. Qwen2.5 technical report. CoRR ,
abs/2412.15115.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W. Cohen, Ruslan Salakhutdinov, and
Christopher D. Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing. In Proceedings of the 2018 Conference on Em-
pirical Methods in Natural Language Processing,
Brussels, Belgium, October 31 - November 4, 2018 ,
pages 2369–2380. Association for Computational
Linguistics.
Howard Yen, Tianyu Gao, and Danqi Chen. 2024. Long-
context language modeling with parallel context en-
coding. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), ACL 2024, Bangkok, Thailand,
August 11-16, 2024 , pages 2588–2610. Association
for Computational Linguistics.
Zeping Yu and Sophia Ananiadou. 2024. How do large
language models learn in-context? query and key ma-
trices of in-context heads are two towers for metric
learning. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
EMNLP 2024, Miami, FL, USA, November 12-16,
2024 , pages 3281–3292. Association for Computa-
tional Linguistics.
Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong
Liu, Bin Yu, Jianfeng Gao, and Tuo Zhao. 2024a.
Tell your model where to attend: Post-hoc attention
steering for llms. In The Twelfth International Con-
ference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024 . OpenReview.net.
Qingru Zhang, Xiaodong Yu, Chandan Singh, Xiaodong
Liu, Liyuan Liu, Jianfeng Gao, Tuo Zhao, Dan Roth,
and Hao Cheng. 2024b. Model tells itself where to at-
tend: Faithfulness meets automatic attention steering.
CoRR , abs/2409.10790.
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zi-
hang Xu, Junhao Chen, Moo Khai Hao, Xu Han,
Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and
Maosong Sun. 2024c. ∞bench: Extending long
context evaluation beyond 100k tokens. CoRR ,
abs/2402.13718.
Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris
Callison-Burch. 2024. Fanoutqa: A multi-hop, multi-
document question answering benchmark for large
language models. In Proceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics, ACL 2024 - Short Papers, Bangkok, Thai-
land, August 11-16, 2024 , pages 18–37. Association
for Computational Linguistics.

A Performance Gap
We study how distracting documents affects model
performance in long-context settings. Starting with
only gold evidence, we gradually insert irrelevant
documents and observe a performance drop as
shown in Figure 1. This suggests that longer inputs
with more irrelevant content weaken the model’s
retrieval and reasoning. This motivates our design
of a retrieval strategy to filter out such noise.
B Baselines
We compare CAFE with the following baselines:
•Directly Answering. Asking LLMs to directly
answer the question by using the context.
•In-Context Retrieval. LLMs are initially
prompted to generate the key documents that sup-
port answering the question. Then, models are
prompted to answer the question with the key doc-
uments appended to the context.
•Oracle RAG. Asking LLMs to answer the
question only based on the ground-truth documents
to estimate an upper limit performance.
•Vanilla RAG. For Retrieval-Augmented Gen-
eration (RAG) over the documents, we employ
BGE-large-env-1.5 (Xiao et al., 2024b) as the em-
bedding model.
•Supervised Fine-tuning. The LLM is trained
on training sets of these datasets. We randomly
sample 2000, 5000, and 5000 training instances for
SQuAD, HotpotQA, and MusiQue, respectively.
C Evidence Selection Results
We conduct additional validations on larger and
different models, and the experimental results
are shown in Table 7. Even for Llama-3.1-70B-
Instruct, a model with significantly more parame-
ters, its ICR (In-Context Retrieval) capability still
declines sharply as the context length increases,
whereas the attention-based retrieval method re-
mains more stable. This suggests that ICR is con-
strained by the expansion capability of the context
window, whereas attention-based methods better
adapt to long-text settings. Additionally, we supple-
ment our experiments with Mistral and Phi, among
other models, to further validate the generalizabil-
ity of our findings. The results consistently demon-
strate that attention-based retrieval is more robust
than ICR in long-context scenarios.D Differences Between Coarse and Fine
Retrieval Heads
To better understand the behavior of retrieval heads
used in the two stages, we visualize their accumu-
lated attention scores over documents in Figure 4
and Figure 5. We observe that coarse-stage re-
trieval heads focus on a few documents to filter
background information, while fine-stage heads
attend more broadly to distinguish gold evidence
from distractors.
Figure 4: Attention distribution of retrieval heads used
in the coarse-grained filtering stage.
Figure 5: Attention distribution of retrieval heads used
in the fine-grained steering stage.
E Impact of Hyperparameters
We perform an ablation study on the fine-grained
retrieval parameter δ. As shown in Table 9, val-

Model Method HotpotQA-8k HotpotQA-16k HotpotQA-32k SQuAD Musique
Llama-3.1-70B-InstructICR 0.82 0.72 0.54 0.92 0.67
Attention-Based 0.85 0.81 0.77 0.95 0.81
Mistral-3-7B-InstructICR 0.65 0.49 0.33 0.78 0.46
Attention-Based 0.75 0.71 0.64 0.82 0.60
Phi-3.5-Mini-InstructICR 0.39 0.27 0.21 0.72 0.33
Attention-Based 0.54 0.52 0.43 0.73 0.36
Table 7: SubEM scores comparing In-Context Retrieval (ICR) and Attention-Based Retrieval methods across
different context lengths and datasets. The second column indicates the retrieval method; performance declines for
ICR as context length grows, while the attention-based approach remains more robust.
Method HotpotQA-8k HotpotQA-16k HotpotQA-32k SQuAD Musique
Directly Answering 637.02 1482.25 2867.17 592.65 1493.73
Ours (prefill) 840.95 1799.92 3880.35 719.55 1791.15
Table 8: TTFT (ms/token) comparison across datasets. “Ours (prefill)” refers to the inference time including the
prefill enhancement.
ues around log 10 yield stable performance, and
we use log 13 as the default setting in our main
experiments.
δ log 5 log 10 log 13 log 20
EM score 69.0 70.5 70.0 69.5
Table 9: Ablation study on the fine-grained retrieval
parameter δ.
F Inference Latency Evaluation
Our method requires multiple inferences (at least
two prefilling operations), which indeed increases
inference latency. In the first round, the prefill
length is the same as that of direct answer. As a
result, our method is slightly slower than the native
flash-attention used in direct answer. We report the
difference in prefill efficiency between our method
and the Directly Answering baseline in Table 8