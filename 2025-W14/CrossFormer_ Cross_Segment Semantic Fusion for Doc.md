# CrossFormer: Cross-Segment Semantic Fusion for Document Segmentation

**Authors**: Tongke Ni, Yang Fan, Junru Zhou, Xiangping Wu, Qingcai Chen

**Published**: 2025-03-31 02:27:49

**PDF URL**: [http://arxiv.org/pdf/2503.23671v2](http://arxiv.org/pdf/2503.23671v2)

## Abstract
Text semantic segmentation involves partitioning a document into multiple
paragraphs with continuous semantics based on the subject matter, contextual
information, and document structure. Traditional approaches have typically
relied on preprocessing documents into segments to address input length
constraints, resulting in the loss of critical semantic information across
segments. To address this, we present CrossFormer, a transformer-based model
featuring a novel cross-segment fusion module that dynamically models latent
semantic dependencies across document segments, substantially elevating
segmentation accuracy. Additionally, CrossFormer can replace rule-based chunk
methods within the Retrieval-Augmented Generation (RAG) system, producing more
semantically coherent chunks that enhance its efficacy. Comprehensive
evaluations confirm CrossFormer's state-of-the-art performance on public text
semantic segmentation datasets, alongside considerable gains on RAG benchmarks.

## Full Text


<!-- PDF content starts -->

CrossFormer: Cross-Segment Semantic Fusion for Document Segmentation
Tongke Ni1, Yang Fan1, Junru Zhou2, Xiangping Wu1, Qingcai Chen1
1Harbin Institute of Technology (Shenzhen), Shenzhen, China
2Tencent Inc. Guangzhou, China
{nee}@tanknee.cn {yfan}@stu.hit.edu.cn {doodlezhou}@tencent.com
{wuxiangping,qingcai.chen}@hit.edu.cn
Abstract
Text semantic segmentation involves partition-
ing a document into multiple paragraphs with
continuous semantics based on the subject
matter, contextual information, and document
structure. Traditional approaches have typi-
cally relied on preprocessing documents into
segments to address input length constraints,
resulting in the loss of critical semantic in-
formation across segments. To address this,
we present CrossFormer, a transformer-based
model featuring a novel cross-segment fusion
module that dynamically models latent seman-
tic dependencies across document segments,
substantially elevating segmentation accuracy.
Additionally, CrossFormer can replace rule-
based chunk methods within the Retrieval-
Augmented Generation (RAG) system, produc-
ing more semantically coherent chunks that en-
hance its efficacy. Comprehensive evaluations
confirm CrossFormer’s state-of-the-art perfor-
mance on public text semantic segmentation
datasets, alongside considerable gains on RAG
benchmarks.
1 Introduction
Text semantic segmentation constitutes a funda-
mental challenge in the field of document analy-
sis, focusing on the automated partition of docu-
ments into contiguous semantic units exhibiting
thematic or contextual coherence (Hearst, 1994).
This task serves as a foundational preprocessing
step for downstream applications spanning docu-
ment summarization (Xiao and Carenini, 2019; Ku-
piec et al., 1995), structured information extraction
(Chinchor et al., 1993), and Retrieval-Augmented
Generation (RAG) pipelines (Huang and Huang,
2024). Current methodologies bifurcate into multi-
ple operational paradigms, including hierarchical
segmentation (Bayomi and Lawless, 2018; Hazem
et al., 2020) and linear segmentation (Hearst, 1997).
Within the latter paradigm, CrossFormer focuseson the planar non-overlapping text semantic seg-
mentation task.
Text semantic segmentation necessitates that sen-
tences within the same paragraph cohesively re-
volve around a central topic, while maintaining
minimal semantic overlap between distinct para-
graphs. Early unsupervised methodologies iden-
tified segmentation boundaries through Bayesian
models (Chen et al., 2009; Riedl and Biemann,
2012) or graph-based methods, where sentences
were treated as nodes (Glavaš et al., 2016). On
the other hand, supervised methods have leveraged
pretrained language models (PLMs) derived from
extensive corpora, subsequently fine-tuning them
on annotated text semantic segmentation datasets.
This paradigm enables a more comprehensive uti-
lization of document content, yielding superior seg-
mentation results (Koshorek et al., 2018a; Lukasik
et al., 2020; Zhang et al., 2021; Yu et al., 2023).
However, the inherent constraints of transformer-
based models, particularly their maximum con-
text length, have led some approaches to partition
documents into fixed-length segments (Yu et al.,
2023). This strategy, while pragmatic, introduces
notable limitations. Chief among these is the ab-
sence of inter-segment correlations, which impedes
the model’s ability to capture sentence-level infor-
mation spanning multiple segments. Consequently,
the model’s capacity to comprehend the broader se-
mantic structure of the document is compromised.
To address this challenge, as illustrated in Figure
1, we propose CrossFormer, a novel model incorpo-
rating CSFM, which models interactions between
document segments, thereby enhancing segmen-
tation performance by integrating cross-segment
dependencies and facilitating a more robust under-
standing of the document’s overarching semantic
hierarchy.
Leveraging CrossFormer’s capacity for
document-level semantic segmentation, we
propose its integration into Retrieval-Augmented
1arXiv:2503.23671v2  [cs.CL]  2 Apr 2025

1 2 3 4 5 6 7 8
1 2 3 4 5 6 7 8
1 2 3 4 5 6 7 8CrossFormer1 2 3 4 7 8
1 2 2 3 ··· 7 8
1 2 3 4 5 6 7 8Neighboring  Sentence -based Method
Preprocessed
InputSemantic
Interaction
Document
Sentences1 2 3 4 5 6 7 8
1 2 3 4 5 6 7 8
1 2 3 4 5 6 7 8Document  Segment-based  Method
(a) (b) (c)6 5Figure 1: 1a illustrates methods that leverages neighboring sentences of the candidate segmentation boundary to
harness the contextual information (Lukasik et al., 2020). 1b presents approaches that divide the document into
segments, followed by the intra-segment semantic interaction. 1c introduces our proposed CrossFormer featuring
CSFM to extract cross-segment semantic interaction depicted by green lines.
Generation (RAG) systems. RAG has emerged
as a critical framework for mitigating limitations
inherent to large language models (LLMs), includ-
ing temporal data obsolescence, domain-specific
or proprietary data scarcity, and hallucination
(Izacard et al., 2023; Ram et al., 2023). A
conventional RAG pipeline comprises retrieval
and generation modules, where the retrieval
stage involves chunk segmentation, vectorization,
and query matching (Gao et al., 2023). Current
RAG implementations often employ rule-based
(Chase, 2022) or LLM-driven chunking methods
(Duarte et al., 2024; Zhao et al., 2024). However,
rule-based approaches frequently fail to preserve
intra-chunk semantic coherence. And LLM-based
segmentation methods, though effective in aligning
chunks with semantic boundaries (Duarte et al.,
2024; Zhao et al., 2024), incur substantial latency.
On this basis, we advocate for embedding Cross-
Former as the chunk splitter within RAG, which
can generate contextually coherent chunks while
maintaining computational efficiency, thereby
addressing the dual challenges of semantic fidelity
and processing speed.
The main contributions of our paper are as fol-
lows:
•We investigate the cross-segment semantic in-
formation loss caused by document prepro-
cessing and propose a Cross-Segment Fusion
Module (CSFM) to explicitly model latent
semantic dependencies across document seg-
ments, thereby improve the performance of
text semantic segmentation.
•We propose a novel model named Cross-
Former featuring CSFM, which achieves state-
of-the-art performance on the text semanticsegmentation benchmarks. Ablation studies
have demonstrated the effectiveness of the
CSFM module.
•We integrate CrossFormer as the text chunk
splitter into the RAG system and achieve supe-
rior performance compared to existing meth-
ods.
2 Related Work
2.1 Text Semantic Segmentation
Text semantic segmentation includes supervised
and unsupervised methods. Unsupervised methods,
like the Bayesian approach by (Chen et al., 2009),
use probabilistic generative models for document
segmentation; (Riedl and Biemann, 2012) applied
a Bayesian method based on LDA, identifying seg-
mentation boundaries through drops in coherence
scores between adjacent sentences; (Glavaš et al.,
2016) introduced an unsupervised graph method,
where sentences are nodes and edges represent se-
mantic similarity, segmenting text by finding the
maximum graph between adjacent sentences. Deep
learning-based supervised methods have improved
text semantic segmentation accuracy. (Koshorek
et al., 2018b) modeled document segmentation as
a sequence labeling task, predicting segmentation
boundaries at separators, and introduced the WIKI-
727k dataset, where supervised methods outper-
formed unsupervised ones. (Lukasik et al., 2020)
introduced hierarchical BERT and other models
to capture inter-sentence features, enhancing the
performance of pre-trained language models fine-
tuned on the WIKI-727k dataset.(Zhang et al.,
2021) proposed a model named SeqModel by in-
tegrating phonetic information and a sliding win-
dow to improve performance and lower resource
2

consumption, also providing the Wiki-zh dataset
for multilingual text semantic segmentation. (Yu
et al., 2023; Somasundaran et al., 2020; Arnold
et al., 2019) utilizes explicit coherence modeling to
enhance text semantic segmentation performance.
Unsupervised methods lack the capability to iter-
atively refine and optimize their performance by
labeled datasets. While current supervised learning
approaches have enhanced performance through
techniques such as phone embedding(Zhang et al.,
2021) and Contrastive Semantic Similarity Learn-
ing(Yu et al., 2023), we contend that greater focus
should be directed towards addressing the issue
of cross-segment information loss induced by pre-
processing strategies that partition long documents
into fixed-length segments.
2.2 RAG Text Chunk Splitter
Text chunking serves as a foundational component
of Retrieval-Augmented Generation (RAG) sys-
tems. Current methodologies bifurcate into rule-
based and LLM-based paradigms. Rule-based ap-
proaches—such as chunking by sentences, para-
graphs, or chapters—prioritize structural fidelity.
Prior work by (Chase, 2022) further combines sep-
arators, word counts, and character limits to define
chunk boundaries. While these strategies preserve
document structure, they inherently neglect seman-
tic relationships during chunking. LLM-based
methods, conversely, emphasize semantic coher-
ence. For instance, (Kamradt, 2024) cluster text via
embeddings to aggregate semantically related con-
tent, while (Duarte et al., 2024) employ question-
answering prompts to guide LLM-driven chunking
for narrative texts. (Zhao et al., 2024) advances
this by computing sentence-level perplexity and
deploying margin sampling to infer semantically
informed chunk boundaries. However, LLM-based
techniques face practical constraints: their parame-
terized architectures and autoregressive inference
mechanisms incur substantial latency. To balance
chunk accuracy and computational cost, we require
a lightweight model capable of accurately chunk
documents according to semantics.
3 Methodology
This section elucidates the architecture of Cross-
Former and its integration within the Retrieval-
Augmented Generation (RAG). The exposition
commences with a formal definition of the text
semantic segmentation task and the preprocessingmethodology of documents. Subsequently, we in-
troduce the proposed Cross-Segment Fusion Mod-
ule (CSFM), which enhances semantic coherence
across document segments. Finally, we provide
a comprehensive exposition of CrossFormer’s sys-
tem integration within the RAG pipeline, emphasiz-
ing its capacity to generate semantic chunks while
demonstrating enhancements in answer generation.
3.1 CrossFormer
Task Definition We frame the task of text se-
mantic segmentation as a sentence-level sequential
labeling problem (Lukasik et al., 2020; Koshorek
et al., 2018b; Zhang et al., 2021; Yu et al., 2023).
Consider a document D={s1, s2, . . . , s n}com-
prising nsentences, where sidenotes i-th sentence.
A binary segmentation label yi∈ {0,1}is assigned
to each sentence si, such that yi= 1denotes the ter-
minal boundary of a paragraph unified by semantic
coherence, while yi= 0signifies continuity within
the same topical segment. The objective is to train
a function f:D → { 0,1}ncapable of predict-
ingyifor each siaccording to the context. We
append a special separator [SENT ]to the end of
each sentence si, yielding a modified document
D′={s′
1, s′
2, . . . , s′
n}. The model subsequently
evaluates whether the [SENT ]token appended to
s′
idemarcates a segment boundary. Formally, let
T={T1, T2, . . . , T m}represent the complete set
of semantic paragraphs within D, where:
D=m[
i=1Tiand Ti∩Ti′=∅∀i̸=i′.(1)
Each paragraph Ti={s(i)
1, s(i)
2, . . . , s(i)
j, . . .}con-
stitutes an ordered sequence of sentences shar-
ing a cohesive semantic topic. The segmenta-
tion task thus reduces to partitioning Dinto the
setTof disjoint, topically homogeneous parti-
tions Ti, achieved through precise identification
of boundary-inducing [SENT ]tokens.
Document pre-processing Considering the
excessive length of documents typically involved
in text semantic segmentation tasks, it is essen-
tial to adopt appropriate methodologies for effec-
tively modeling long documents. In accordance
with established long-text modeling techniques, we
integrate truncation and segmentation approaches
(Dong et al., 2023). As shown in Algorithm 1,
given a document Dwith length N. We first seg-
mentDaccording to the separators specified by the
task (e.g., line breaks or periods). Denote the set of
3

Vanadium is a chemical element with symbol V and atomic number 23.……It is a hard, silvery grey, ductile, and malleable transition metal…
PretrainedLanguageModelDataPreprocessDoc	Segment!Doc	Segment"…Doc	Segment#
Cross-Segment Fusion ModuleLinear&Softmax𝐷𝑜𝑐𝐶ℎ𝑢𝑛𝑘!…𝐷𝑜𝑐𝐶ℎ𝑢𝑛𝑘$CrossFormerRawDocuments(a) CrossFormer Pipeline
[𝐶𝐿𝑆]Vanadium is a…𝑆𝐸𝑁𝑇!It is a hard, silvery...𝑆𝐸𝑁𝑇"…Del Río accepted…𝑆𝐸𝑁𝑇#[𝑆𝐸𝑃]
(b) Document Segment Example
[CLS]Sub&MaxPoolConcatℎ!"#[CLS][CLS]ℎ#$%Linear[CLS][CLS]	ℎ['$(!]
………
	ℎ#$*+!…	ℎ#$*+"	ℎ#$*+#	ℎ#$*+!…	ℎ#$*+"	ℎ#$*+#	ℎ#$*+!…	ℎ#$*+"	ℎ#$*+#… (c) Cross-Segment Fusion Module
Figure 2: Architecture of CrossFormer. 2a illustrates the pipeline of CrossFormer for text semantic segmentation
task and its architecture, which consists of a pre-trained language model, Cross-Segment Fusion Module (CSFM),
and a linear classifier. 2b shows an example of a preprocessed document segment as input to the model. 2c
demonstrates the detailed structure of CSFM.
sentences obtained as {si}n
i=1, where |si| ≤Lfor
alli= 1,···, nandPn
i=1|si|=N,Lis the maxi-
mum length of a single sentence. Then, we concate-
nate these sentences in sequence to form kdocu-
ment segments {dj}k
j=1, such thatPk
j=1|dj|=N
and|dj| ≤Mfor all j= 1,···, kandk < K ,
where Mis the maximum length of a document
segment. The parameters LandMare hyperparam-
eters with L < M . The parameter Krepresents
the maximum number of segments. A larger K
allows the model to capture longer-range semantic
information, but it also increases the demand on
GPU memory. These document segments are then
assembled into a batch and fed into the model for
training or inference. When generating document
segments, the special tokens [CLS ]and[SEP ]
will be appended at the beginning and end respec-
tively.
Cross-Segment Fusion Module (CSFM) Once
a document is splitted into multiple segments, these
segments are compiled into a batch. As depicted in
Figure 2, we propose CSFM that integrates global
and sentence-level information for classification.
hseg=h[CLS]−h[SEP] (2)
hglobal = max([ hseg-1, hseg-2, ..., h seg-k])(3)
In order to extract the semantic information
of segments, we select the pre-trained special
tokens [CLS ]and [SEP ]in the transformermodel. We construct hsegthrough formula (2).
Thus, we can obtain the semantic representation
hseg-1, hseg-2,···, hseg-k of k segments. Then, we
apply max-pooling to obtain the largest semantic
component from these representations and acquire
hglobal to represent the global semantic information
of all segments.
hconcat =Concat (hglobal, h[SENT] ) (4)
h[FEA] =Linear 2(Linear 1(hconcat)) (5)
Upon the global semantic embedding hglobal is
computed, we concatenate it with each separator
embedding h[SENT ]. Subsequently, two linear lay-
ers are applied to the concatenated vector hconcat
to get h[FEA] , the representations of separators in-
tegrated with global semantic information. This
process is mathematically encapsulated in equation
(4) and (5).
Following the acquisition of the output from the
CSFM, h[FEA] is input into the Linear layer and
applied the Softmax function to get the classifica-
tion results. Finally, We segment the document into
semantic paragraphs according to the classification
results .
During the training phase, the CrossFormer is
trained using a standard cross-entropy loss function
to maximize the likelihood of the training data,
thereby enhancing the model’s ability to accurately
4

Algorithm 1 Document Processing for Text Se-
mantic Segmentation
1:Input: A document Dconsisting of Ssentences. Max-
imum sentence length L, maximum document segment
length M. Maximum number of document segments K
2:Output: Batch Bof document segments for model M
3:{s1, s2, . . . , s n} ← Split and Truncate Dusing
separators such that max(|si|)≤Lfori= 1, . . . , n
4: Initialize an empty list B
5:j←1
6:segment ←empty string
7:fori= 1tondo
8: if|segment |+|si|+ 2> M then
9: segment ←segment + [SEP ]
10: Add segment toB
11: segment ←[CLS ] +si+ [SENT ]
12: j←j+ 1
13: else
14: segment ←segment +si+ [SENT ]
15: end if
16:end for
17:ifsegment is not empty then
18: Add segment + [SEP ]toB
19:end if
20:ifj > K then
21: B← {segment 1, . . . , segment K}
22:end if
23:return B
identify segmentation separators within the text
sequence.
3.2 CrossFormer as RAG Text Chunk Splitter
We integrate CrossFormer as the text chunk splitter
of the RAG system to get better semantic chunks.
As illustrated in Figure 3, we have devised a sys-
tematic process for splitting long documents. Ini-
tially, the CrossFormer is utilized to split the input
document, yielding a sequence of text chunks. Sub-
sequently, the length of each segment is evaluated.
When the length surpass a predefined threshold,
the text chunk will be input into the splitting queue
for further recursive processing until the model
determines that no additional segmentation is re-
quired or the text chunk length falls below the spec-
ified threshold. Upon achieving the segmented text
chunks, a retriever and a question prompt are em-
ployed to conduct relevance retrieval, generating
the context which is subsequently input into the
LLMs to get the final answer.
4 Experiment
4.1 Text Semantic Segmentation
Experimental Setup We performed testing and
evaluation on several widely used text semantic seg-
mentation datasets, including the English datasets
DocumentCross-Former𝐷𝑜𝑐𝐶ℎ𝑢𝑛𝑘!𝐷𝑜𝑐𝐶ℎ𝑢𝑛𝑘"𝐷𝑜𝑐𝐶ℎ𝑢𝑛𝑘#…𝐿𝑒𝑛𝐶ℎ𝑢𝑛𝑘!>thresholdQuestionPromptChunkRetrieverLargeLanguageModelGenerationAnswerFigure 3: A flowchart depicting the integration of Cross-
Former into the RAG system as a text chunk split-
ter.The process begins with document input, followed
by CrossFormer-based chunking. The top-k semanti-
cally relevant chunks are retrieved, and a large language
model generates the final answer using the retrieved
chunks.
WIKI-727k and WIKI-50 (Koshorek et al., 2018b),
and the Chinese dataset WIKI-zh (Zhang et al.,
2021). The CrossFormer was trained on the train-
ing subsets of these datasets and subsequently eval-
uated on their respective test subsets, with an infer-
ence batch size of 2.
Dataset statistic The WIKI-727k dataset is a
large-scale collection derived from Wikipedia. All
images, tables, and other non-textual elements have
been removed, and the documents have been tok-
enized into sentences using the PUNKT tokenizer
from NLTK (Bird, 2006). Semantic segmentation
of the documents was executed in accordance with
the table of contents. For the experiments, the train-
ing set provided by WIKI-727k was utilized to train
the model, and its performance was assessed on the
corresponding test set to ensure a fair comparison
with other models. The WIKI-50 dataset, a subset
derived from WIKI-727k, was employed for faster
evaluation of model performance. WIKI-zh, con-
structed similarly to WIKI-727k but utilizing data
from the Chinese Wikipedia, constitutes a larger
dataset. The statistical characteristics of these three
datasets are presented in Table 1, which illustrates
that WIKI-727k in English contains a greater num-
ber of sentences, a comparable number of topics,
yet fewer sentences per topic. The statistics of
WIKI-50 exhibit a close resemblance to those of
WIKI-727k.
Model training settings As discussed in Chap-
ter 3.1, to effectively model long documents within
a limited context, we developed a scheme that inte-
grates truncation and segmentation. In the experi-
5

Dataset Docs S/Doc SLen T/Doc S/T
WIKI-727k 727,746 52.65 129.20 6.18 8.52
WIKI-zh 820,773 11.77 43.28 5.57 2.11
WIKI-50 50 61.40 125.10 7.68 7.99
Table 1: Statistics of text semantic segmentation
datasets. Docs denotes the total number of documents.
S/Doc and T/Doc denote the average number of sen-
tences and topics per document, respectively. SLen
denotes the average number of characters per sentence.
S/T denotes the average number of sentences per topic.
Dataset ID Source Metric #Data
Single-Document QA
NarrativeQA 1-1 Literature, Film F1 200
Qasper 1-2 Science F1 200
MultiFieldQA-en 1-3 Multi-field F1 150
MultiFieldQA-zh 1-4 Multi-field F1 200
Multi-Document QA
HotpotQA 2-1 Wikipedia F1 200
2WikiMultihoopQA 2-2 Wikipedia F1 200
MuSiQue 2-3 Wikipedia F1 200
DuReader 2-4 Baidu Search Rouge-L 200
Table 2: Overview of the LongBench dataset (Bai et al.,
2024). Chinese datasets are highlighted in light green;
the Source column indicates the provenance of these
datasets, and the #Data column specifies the number of
samples contained within each dataset.
mental phase, we adapt the training and inference
parameters for various models and devices. For
models such as BERT (Devlin et al., 2019) and
RoBERTa (Liu et al., 2019), which have a context
length of 512 tokens, we established the maximum
length Mof document segments at 512 tokens and
did not impose a limit on the maximum number of
document segments K. For the Longformer (Belt-
agy et al., 2020) model, which supports a longer
context (up to 4096), we set the maximum number
of document segments to either 4 or 8, contingent
upon the available GPU memory, and accordingly
adjusted the maximum length of each document
segment.
4.2 RAG Text Chunk Splitter
Experiment Setup To explore whether replac-
ing the non-semantic chunking methods with a
text semantic segmentation model can bring end-
to-end performance improvements to the RAG
system, we followed (Bai et al., 2024), select-
ing the same eight datasets, including single-
document datasets NarrativeQA, MultiFieldQA-
en, Qasper, MultiFieldQA-zh, and multi-documentModelWIKI-727k
P R F1
BERT-Base + Bi-LSTM (2020) 67.3 53.9 59.9
Hier. BERT (24-Layers) (2020) 69.8 63.5 66.5
Cross-segment BERT-L 128 (2020) 69.1 63.2 66.0
Cross-segment BERT-L 256 (2020) 61.5 73.9 67.1
SeqModel:BERT-Base (2021) 70.6 65.9 68.2
Longformer-Base(TSSP+CSSL) (2023) - - 77.16
Longformer-Large(TSSP+CSSL)†82.40 74.81 78.42
CrossFormer:Longformer-Base(ours) 81.33 71.57 76.14
CrossFormer:Longformer-Large(ours) 82.02 75.98 78.88
Table 3: Performance results on the WIKI-727k test set.
Bold numbers indicate the best performance. †denotes
our reproduced results. Pmeans Precision, Rmeans
Recall.
datasets HotpotQA, MuSiQue, 2WikiMultihopQA,
DuReader. Notably, the MultiFieldQA-zh and
DuReader datasets are Chinese datasets, while the
others are in English. In the experiment, we use
CrossFormer:Roberta trained on the WIKI-727k
training set as the text chunk splitter.
Preprocessing and RAG Workflow In these
datasets of LongBench (Bai et al., 2024), the orig-
inal documents do not have ready-made sentence
segmentation results, and the addition style of sep-
arators varies across different datasets. Therefore,
we processed the original text, for example, by
adding line breaks after Chinese and English peri-
ods, and only at the end of sentences, to preserve
sentence integrity as much as possible. Then, we
split the original text into sentences based on sin-
gle line breaks. After obtaining the sentences, we
constructed batches according to Algorithm 1, and
input them into CrossFormer to get the semantic
segmentation results. Following the process illus-
trated in Figure 3, we split the long documents into
text chunks, and use bge-m3 (Chen et al., 2024)
embedding model to convert both the text chunks
and the question prompt into embeddings. We cal-
culated the cosine similarity scores between the
question embedding and the text chunk embed-
dings, selected the K Chunks as context, filled the
preset prompt template in LongBench, and then
input it into LLMs for answer generation.
5 Evaluation
5.1 Results of Text Semantic Segmentation
The performance of CrossFormer on the WIKI-
727k dataset is presented in Table 3. We evalu-
ate the performance of CrossFormer on the WIKI-
727k test set using both the Longformer-Base and
6

ModelWIKI-zh
P R F1
Cross-segment BERT-Base 128-128 61.2 80.2 69.4
SeqModel:BERT-Base 78.4 69.5 73.7
SeqModel:StructBERT-Base 79.2 72.7 75.8
SeqModel:RoBERTa-Base 74.6 73.7 74.2
SeqModel:ELECTRA-Base 73.5 76.6 75.0
CrossFormer:RoBERTa-Base (ours) 78.01 78.60 78.31
Table 4: The evaluation results on the WIKI-zh test set,
Bold numbers indicate the best performance. Pmeans
Precision, Rmeans Recall.
Longformer-Large models for training. Compared
to other text semantic segmentation models, Cross-
Former attains the highest F1 score.
The evaluation results on the Wiki-zh dataset are
presented in Table 4. We utilized the RoBERTa
(Liu et al., 2019) for this experiment. The experi-
mental findings reveal that CrossFormer achieved
superior performance compared to other leading
methods, with a notable improvement in the F1
metric by 2.51. These results suggest a significant
enhancement in performance.
5.2 Results on RAG benchmark
Eight datasets from the LongBench were employed
to evaluate retrieval and context compression. The
performance of CrossFormer as a chunk splitter
is presented in Table 7. The experiment was on
the number of input chunks generated by differ-
ent chunk splitter methods. We also explored the
impact of the number of input chunks on the per-
formance.
Moreover, the performance indicate that Cross-
Former demonstrates superior performance on En-
glish datasets in comparison to other methodolo-
gies. Conversely, its performance on Chinese
datasets is merely average, which may be attributed
to the utilization of only the English version of the
Wiki-727k corpus during the training phase.
In terms of performance metrics across each
dataset, CrossFormer outperformed other methods
in most settings, thereby demonstrating its effec-
tiveness. Although Lumber has achieved good re-
sults in some settings, due to its use of LLMs for
chunking, it has a certain disadvantage in terms of
speed. However the LC-C and LC-R methods have
good performance on some datasets, the chunk
method based on CrossFormer has achieved better
results on more datasets after integrating semantic
information.CrossFormer Base ModelWIKI-727k
Precision Recall F1
BERT-Base 67.16 73.42 70.15
RoBERTa-Base 73.96 72.66 73.31
Longformer-Base 81.33 71.57 76.14
Longformer-Large 82.02 75.98 78.88
Table 5: Performance on the Wiki-727K test set from
our proposed CrossFormer based on different PLMs.
Bold numbers denote the best performance.
CrossFormer Base ModelWIKI-727k
Precision Recall F1
Longformer-Base 80.05 71.87 75.74
w/o CSFM 80.43 70.68 75.24
Longformer-Large 82.02 75.98 78.88
w/o CSFM 80.53 75.31 77.83
BERT-Base 67.16 73.42 70.15
w/o CSFM 76.92 63.6 69.63
Table 6: Ablation study for CSFM on WIKI-727k test
set. Bold numbers denote the best performance.
5.3 Ablation Study
To validate the effectiveness of the proposed Cross-
Former method, ablation experiments were con-
ducted on the WIKI-727k test set utilizing vari-
ous models and the proposed CSFM. As illustrated
in Table 5, the method was trained and evaluated
on BERT-Base (Devlin et al., 2019), RoBERTa-
Base (Liu et al., 2019), Longformer-Base (Belt-
agy et al., 2020), and Longformer-Large (Beltagy
et al., 2020) models. The results indicate that the
CrossFormer method achieves the highest perfor-
mance with Longformer, likely due to its capacity
to model longer contexts at the model level, thereby
enabling the inclusion of more sentences within
the same document segment. This performance
advantage aligns with expectations. Additionally,
the performance variations observed between the
Longformer-Base and Longformer-Large models
confirm that an increase in model parameters re-
sults in substantial performance improvements.
As illustrated in Table 6, a series of compre-
hensive ablation experiments were conducted on
the WIKI-727k test set. The results indicate that
the module achieved performance enhancements
in both Recall and F1 metrics. Additionally, abla-
tion experiments were performed across different
models, demonstrating that the CSFM can gen-
erally augment the performance. In addition, as
depicted in Figure 4, experiments were also car-
ried out on the CrossFormer with different input
7

MethodSingle-Doc QA Multi-Doc QAAvg En-Avg Zh-Avg
1-1 1-2 1-3 1-4⋆2-1 2-2 2-3 2-4⋆
w/o retrieval 17.72 44.67 50.14 60.82 46.36 45.59 23.66 29.68 39.83 38.02 45.25
3 Chunks
LC-C 22.65 43.08 49.39 62.51 52.06 40.18 27.06 28.52 40.68 39.07 45.52
LC-R 20.32 40.16 48.27 59.12 51 42.19 28.17 28.51 39.72 38.35 43.82
Lumber 19.95 43.99 48.34 61.98 51.87 41.89 24.27 28.29 40.07 38.39 45.14
CrossFormer 23.5 40.36 47.25 63.61 51.24 47.65 26.25 28.52 41.05 39.38 46.07
5 Chunks
LC-C 22.28 42.14 48.22 62.66 53.32 47.67 29.08 28.68 41.76 40.45 45.67
LC-R 21.08 40.66 48.64 61.35 53.19 45.78 25.42 28.56 40.59 39.13 44.96
Lumber 20.94 41.64 49.75 62.88 52.31 44.95 25.43 29.59 40.94 39.17 46.24
CrossFormer 23.08 41.44 46.96 62.26 52.5 52.2 34.39 27.62 42.56 41.76 44.94
Table 7: Experiment results on LongBench dataset. The rows marked in gray indicate our proposed CrossFormer,
LC-C denotes Langchain’s CharacterTextSplitter , LC-R denotes Langchain’s RecursiveCharacterTextSplitter , and
Lumber indicates the LumberChunk (Duarte et al., 2024) method with Qwen2.5-7B-Instruct Model. K Chunks
denotes the number of the most relevant chunks selected as context. ⋆indicates the Chinese dataset. The name of
datasets corresponding to each ID are shown in Table 2.
500 1000 1500 2000 2500 3000 3500 4000
Max Input Length6668707274767880F1
CrossFormer:Longformer-Large
CrossFormer:Longformer-Base
Longformer-Base
Longformer-Large
Figure 4: The influence of max input length of Cross-
Former on the WIKI-50 dataset (Koshorek et al., 2018b).
Models without the prefix "CrossFormer" are ablation
experiments that do not contain CSFM.
lengths on the WIKI-50 dataset. The experimental
results revealed that the CrossFormer incorporating
CSFM exhibited improved performance in most in-
put lengths compared to the model without CSFM.
Secondly, it was discovered that when the input
length was less than approximately 1500, the per-
formance improved with the input length. When
the length was greater than approximately 1500, the
performance improvement it brought was relatively
minor.
6 Conclusion
In this paper, we introduce a new text semantic
segmentation model, termed CrossFormer, whichincorporates a novel Cross-Segment Fusion Mod-
ule (CSFM). This module is designed to deal with
the cross-segment information loss between docu-
ment segments by constructing document segment
embedding and global document embedding to alle-
viate the context information loss when predicting
segmentation boundaries. Furthermore, we inte-
grate the CrossFormer into the RAG system as a
semantic chunk splitter for documents, aiming to
overcome the limitations of previous rule-based
methods that fail to adequately leverage seman-
tic information within the text. Empirical results
indicate that our proposed CrossFormer achieves
state-of-the-art performance across several public
datasets. Additionally, the model demonstrates im-
provements in RAG performance.
7 Limitations
Despite its innovative Cross-Segment Fusion Mod-
ule and strong performance in text semantic seg-
mentation, CrossFormer has limitations. Since
CrossFormer is a deep learning method, when it
is used as a text chunk splitter, its running speed
is slower than rule-based methods but faster than
LLM-based methods. Secondly, CrossFormer can-
not precisely control the upper limit of the length of
text chunks. Therefore, it may be necessary to com-
bine rule-based methods to output an appropriate
length. Finally, as a linear text semantic segmen-
tation model, CrossFormer cannot output partially
overlapping text chunks, which is required in some
scenarios of the RAG system.
8

References
Sebastian Arnold, Rudolf Schneider, Philippe Cudré-
Mauroux, Felix A Gers, and Alexander Löser. 2019.
Sector: A neural model for coherent topic segmenta-
tion and classification. Transactions of the Associa-
tion for Computational Linguistics , 7:169–184.
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang,
and Juanzi Li. 2024. LongBench: A bilingual, multi-
task benchmark for long context understanding. In
Proceedings of the 62nd Annual Meeting of the As-
sociation for Computational Linguistics (Volume 1:
Long Papers) , pages 3119–3137, Bangkok, Thailand.
Association for Computational Linguistics.
Mostafa Bayomi and Séamus Lawless. 2018. C-HTS:
A concept-based hierarchical text segmentation ap-
proach. In Proceedings of the Eleventh International
Conference on Language Resources and Evaluation,
LREC 2018, Miyazaki, Japan, May 7-12, 2018 . Euro-
pean Language Resources Association (ELRA).
Iz Beltagy, Matthew E. Peters, and Arman Cohan. 2020.
Longformer: The long-document transformer. ArXiv ,
abs/2004.05150.
Steven Bird. 2006. Nltk: The natural language toolkit.
InAnnual Meeting of the Association for Computa-
tional Linguistics .
Harrison Chase. 2022. Langchain.
Harr Chen, S.R.K. Branavan, Regina Barzilay, and
David R Karger. 2009. Global models of document
structure using latent permutations. In Proceedings
of Human Language Technologies: The 2009 An-
nual Conference of the North American Chapter of
the Association for Computational Linguistics , pages
371–379, Boulder, Colorado. Association for Com-
putational Linguistics.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
InAnnual Meeting of the Association for Computa-
tional Linguistics .
Nancy Chinchor, Lynette Hirschman, and David D
Lewis. 1993. Evaluating message understanding sys-
tems: An analysis of the third message understand-
ing conference (muc-3). Computational linguistics ,
19(3):409–450.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. In North American Chapter of the Association
for Computational Linguistics .
Zican Dong, Tianyi Tang, Lunyi Li, and Wayne Xin
Zhao. 2023. A survey on long text modeling with
transformers. Preprint , arXiv:2302.14502.André V . Duarte, João Marques, Miguel Graça, Miguel
Freire, Lei Li, and Arlindo L. Oliveira. 2024. Lum-
berchunker: Long-form narrative document segmen-
tation. In Findings of the Association for Computa-
tional Linguistics: EMNLP 2024, Miami, Florida,
USA, November 12-16, 2024 , pages 6473–6486. As-
sociation for Computational Linguistics.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Qianyu Guo,
Meng Wang, and Haofen Wang. 2023. Retrieval-
augmented generation for large language models: A
survey. CoRR , abs/2312.10997.
Goran Glavaš, Federico Nanni, and Simone Paolo
Ponzetto. 2016. Unsupervised text segmentation us-
ing semantic relatedness graphs. In Proceedings of
the Fifth Joint Conference on Lexical and Computa-
tional Semantics , pages 125–130, Berlin, Germany.
Association for Computational Linguistics.
Amir Hazem, Béatrice Daille, Dominique Stutz-
mann, Christopher Kermorvant, and Louis Cheva-
lier. 2020. Hierarchical text segmentation for me-
dieval manuscripts. In Proceedings of the 28th Inter-
national Conference on Computational Linguistics,
COLING 2020, Barcelona, Spain (Online), Decem-
ber 8-13, 2020 , pages 6240–6251. International Com-
mittee on Computational Linguistics.
Marti A. Hearst. 1994. Multi-paragraph segmentation
expository text. In 32nd Annual Meeting of the As-
sociation for Computational Linguistics , pages 9–
16, Las Cruces, New Mexico, USA. Association for
Computational Linguistics.
Marti A. Hearst. 1997. Texttiling: Segmenting text into
multi-paragraph subtopic passages. Comput. Linguis-
tics, 23(1):33–64.
Yizheng Huang and Jimmy Huang. 2024. A survey
on retrieval-augmented text generation for large lan-
guage models. CoRR , abs/2404.10981.
Gautier Izacard, Patrick S. H. Lewis, Maria Lomeli,
Lucas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2023. Atlas: Few-shot learning
with retrieval augmented language models. J. Mach.
Learn. Res. , 24:251:1–251:43.
Greg Kamradt. 2024. Semantic chunking. https:
//github.com/FullStackRetrieval-com/
RetrievalTutorials/tree/main/tutorials/
LevelsOfTextSplitting .
Omri Koshorek, Adir Cohen, Noam Mor, Michael Rot-
man, and Jonathan Berant. 2018a. Text segmentation
as a supervised learning task. In Proceedings of
the 2018 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, NAACL-HLT, New
Orleans, Louisiana, USA, June 1-6, 2018, Volume
2 (Short Papers) , pages 469–473. Association for
Computational Linguistics.
9

Omri Koshorek, Adir Cohen, Noam Mor, Michael Rot-
man, and Jonathan Berant. 2018b. Text segmenta-
tion as a supervised learning task. arXiv preprint
arXiv:1803.09337 .
Julian Kupiec, Jan Pedersen, and Francine Chen. 1995.
A trainable document summarizer. In Proceedings
of the 18th annual international ACM SIGIR confer-
ence on Research and development in information
retrieval , pages 68–73.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Man-
dar Joshi, Danqi Chen, Omer Levy, Mike Lewis,
Luke Zettlemoyer, and Veselin Stoyanov. 2019.
Roberta: A robustly optimized bert pretraining ap-
proach. ArXiv , abs/1907.11692.
Michal Lukasik, Boris Dadachev, Kishore Papineni, and
Gonçalo Simões. 2020. Text segmentation by cross
segment attention. In Proceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP) , pages 4707–4716, Online. As-
sociation for Computational Linguistics.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Trans. Assoc. Comput. Linguistics ,
11:1316–1331.
Martin Riedl and Chris Biemann. 2012. TopicTiling:
A text segmentation algorithm based on LDA. In
Proceedings of ACL 2012 Student Research Work-
shop , pages 37–42, Jeju Island, Korea. Association
for Computational Linguistics.
Swapna Somasundaran et al. 2020. Two-level trans-
former and auxiliary coherence modeling for im-
proved text segmentation. In Proceedings of the
AAAI Conference on Artificial Intelligence , vol-
ume 34, pages 7797–7804.
Wen Xiao and Giuseppe Carenini. 2019. Extractive
summarization of long documents by combining
global and local context. In Proceedings of the
2019 Conference on Empirical Methods in Natu-
ral Language Processing and the 9th International
Joint Conference on Natural Language Processing,
EMNLP-IJCNLP 2019, Hong Kong, China, Novem-
ber 3-7, 2019 , pages 3009–3019. Association for
Computational Linguistics.
Hai Yu, Chong Deng, Qinglin Zhang, Jiaqing Liu, Qian
Chen, and Wen Wang. 2023. Improving long docu-
ment topic segmentation models with enhanced co-
herence modeling. In Proceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing , pages 5592–5605, Singapore. Associa-
tion for Computational Linguistics.
Qinglin Zhang, Qian Chen, Yali Li, Jiaqing Liu, and
Wen Wang. 2021. Sequence model with self-adaptive
sliding window for efficient spoken document seg-
mentation. In IEEE Automatic Speech Recognition
and Understanding Workshop, ASRU , pages 411–418,
Cartagena, Colombia. IEEE.Jihao Zhao, Zhiyuan Ji, Pengnian Qi, Simin Niu,
Bo Tang, Feiyu Xiong, and Zhiyu Li. 2024. Meta-
chunking: Learning efficient text segmentation via
logical perception. arXiv preprint arXiv:2410.12788 .
10