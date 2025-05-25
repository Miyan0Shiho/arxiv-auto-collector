# Reranking with Compressed Document Representation

**Authors**: Hervé Déjean, Stéphane Clinchant

**Published**: 2025-05-21 11:35:11

**PDF URL**: [http://arxiv.org/pdf/2505.15394v1](http://arxiv.org/pdf/2505.15394v1)

## Abstract
Reranking, the process of refining the output of a first-stage retriever, is
often considered computationally expensive, especially with Large Language
Models. Borrowing from recent advances in document compression for RAG, we
reduce the input size by compressing documents into fixed-size embedding
representations. We then teach a reranker to use compressed inputs by
distillation. Although based on a billion-size model, our trained reranker
using this compressed input can challenge smaller rerankers in terms of both
effectiveness and efficiency, especially for long documents. Given that text
compressors are still in their early development stages, we view this approach
as promising.

## Full Text


<!-- PDF content starts -->

May, 2025
Reranking with Compressed Document Representation
Hervé Déjean, Stéphane Clinchant
NAVER LABS Europe
https://github.com/naver/bergen
Abstract
Reranking,theprocessofrefiningtheoutputofafirst-stageretriever,isoftenconsideredcomputationally
expensive, especially with Large Language Models. Borrowing from recent advances in document
compression for RAG, we reduce the input size by compressing documents into fixed-size embedding
representations. We then teach a reranker to use compressed inputs by distillation. Although based on
a billion-size model, our trained reranker using this compressed input can challenge smaller rerankers
in terms of both effectiveness and efficiency, especially for long documents. Given that text compressors
are still in their early development stages, we view this approach as promising.
1. Introduction
Information Retrieval (IR) is typically understood as
a two-part process: a first-stage designed to swiftly
locatepertinentdocumentsforaspecificquery,followed
by a more costly refinement phase called reranking .
Initially performed with cross-encoder [ 10,20], recent
models have gradually been tested, including encoder-
decoder [ 21,32,33] or decoder-only models, namely
Large Language Models (LLM) [18,27].
While these LLM-based rerankers exhibit remarkable
capabilities (zero-shot scenarios, fewer relevance judge-
ments), they are much less efficient compared to tra-
ditional rerankers using cross-encoders [ 7,33] - which
are already deemed as slower components in IR sys-
tems. The efficiency of large language models remains
an open challenge [ 31], with relatively few studies ad-
dressing this issue. First, a late interaction architecture
has been proposed for cross-encoder-based rerankers,
but it comes at the cost in effectiveness [ 9,19]. More-
over, Liu et al. [16]presents an architecture which
allows the user to customize the architecture of their
reranker by configuring the depth and width of LLM,
achieving a 2 ×speed-up compared to the full-scale
model) with a sequence length of 1024. Gangi Reddy
et al.[8],Zhuang et al. [33]propose to leverage the
output logits of the first generated token to directly
obtain a ranked ordering of the candidates, lowering
the latency of listwise LLM rerankers by 50%.
We propose a more radical approach by signifi-
cantlyreducingthereranker’sinputsequencelength
through compressed representation, thereby achiev-
ing a 10x speed-up1compared to the usual textual
representation .
First-stage dense retriever makes use of this idea of
1assuming an input document length of 512 tokens.
Figure 1: Reranking processing time according to the
input length. Using compressed representation (RRK
models) enables our reranker to maintain constant effi-
ciency regardless of document length.
compressed input by representing a document with a
single embedding. In order to keep reranking effective,
a richer representation is required. Colbert [ 24] can
be considered as a reranker using a compact document
representation. Recently, prompt compression methods
have been developed to speed up LLMs facing long
contexts, dialogue or in retrieval augmented generation
[11]. Recent works like [ 23] and PISCO [ 17] are able
to learn compressed document representations in order
to optimise a Retrieval Augmented Generation (RAG)
systembyeffectivelyreplacingretrievedtextsbyamuch
shorter compressed representation in the LLM prompt.
This raises an interesting question for retrieval: could a
similar approach be used for reranking, relying solely
on a compressed document embedding?
Contribution: We show in this paper that using such
compressed representation in the context of reranking
Corresponding author(s): herve.dejean@naverlabs.comarXiv:2505.15394v1  [cs.IR]  21 May 2025

Reranking with Compressed Document Representation
solves the efficiency problem of LLM-based rerankers
while keeping comparable effectiveness. Additionally,
using a small, constant input length through the use of
compressed representation, enables reranking to main-
tain a constant efficiency regardless of document length
(Figure 12).
2. Reranking Compressed Representation
We build on a recent soft offlinecompression methods,
PISCO, to learn our reranking models called RRK3. A
RRKmodelconsistsintwomodels: afrozenPISCOcom-
pressor, mapping texts to embeddings, and a finetuned
decoder, mapping query-document pairs to a reranking
score.
First, each collection is compressed offlineusing the
PISCO compressor: for each document 𝑑𝑖, a set of
memory tokens(𝑚1,...,𝑚𝑙)is appended, forming
(𝑑𝑖;𝑚1,...,𝑚𝑙), which is fed to the compressor. The
final𝑙hidden states of these memory tokens represent
the document embeddings ci=(𝑐𝑠
𝑖)𝑠=1...𝑙. In our case,
we use𝑙=8memory tokens. To ensure consistent
embeddings for reranking and RAG, the PISCO model
remains frozen, with only the decoder being fine-
tuned using LoRA .
At reranking stage, the compressed documents embed-
dings care loaded and fed to the Decoder (Mistral-7B)
finetuned for reranking. To train the decoder, we use a
mean squared error (MSE) loss to match the scores of a
distilled teacher reranker, by adding a linear projection
layer that projects the last layer representation of the
end-of sequence token to a scalar.
We explore two RRK variants: one using all 32 layers of
the Mistral-7B decoder, and a lighter version using only
the first 16 layers for improved efficiency. In both cases,
8 memory tokens represent each document, and given a
maximum query length of 24 tokens, the total input size
remains fixed at 32 tokens , thus making the efficiency
of the RRK model very competitive (see Figure 1). No-
tably, our method shares conceptual similarities with
ColBERT [ 24], which stores one embedding per token;
however, RRKcompresses each document into just 8
embeddings. For the MSMARCO collection (8.8 million
documents), the PISCO index results in a storage size
of 270 GB, almost twice the size of the Colbert index
(154 GB). However, in our case, this drawback is miti-
gated by the fact that this index is also used during the
generation step in the RAG setting.
2The diagram illustrates the processing time for some models
based on the input length. We simulate a set of 50 queries with 50
documents each, using a batch size of 256 on a A100 80G.
3RRK: compressed version of ReRanKer3. Training
Training a high quality reranker depends on many fac-
tors such as labeled data with relevant documents and
carefully chosen negatives from multiple retrievers, [ 4].
All these choices make comparison hard and repro-
ducibility challenging due to varied and massive train-
ing sets. But, a simpler and more reproducible solu-
tion is to rely on distillation from existing rerankers.
In fact, our goal is to assess whether it is possible to
train rerankers from compressed representations and
see whether LLM-based rerankers could be made faster.
To do so, we take a state-of-the-art cross-encoder and
distill it into LLMs, RRK models but also a more recent
BERT baseline. Note that the distillation direction is
inverted: from a small model to a larger one.
In order to select our teacher, we performed a set of
evaluation using various first-stage and rerankers (see
Appendix A). Based on those results, and aiming at
building an efficient full RAG system, we choose the
SPLADE-V3 sparse model [ 14], faster than a dense
model, and its companion reranker Naver-DeBERTa
[12].
For our training collection, we use the MS MARCO (pas-
sage)dataset[ 1]Thetrainingcollection, whichconsists
of a set of queries and an appropriate document col-
lection (without the need for relevance judgments), is
processed using the selected first-stage retriever and
reranker. For each query, we identify the top 50 docu-
ments produced by the reranker used as teacher. For
the query set, we utilize the 0.5 million training queries,
pairingeachquerywith8documentsrandomlyselected
from the top 50 documents as provided by the retriever.
Weconducttrainingover2epochs, asadditionalepochs
did not yield significant improvements. The finetuning
takes 24h using 1 A100 GPU with a batch size of 8 and
a learning rate of 1×10−4. While stronger training con-
figurations could be developed, our focus is comparing
compressed vs. textual representations, making this
configuration suitable for a fair evaluation.
4. Evaluation
Forevaluation, weusetraditionalIRbenchmarksTREC-
DL19/20[ 5,6],BeIR[ 28],andduetospaceconstraints,
we have only included the pooled subset of the LoTTe
collection (forum and search) [ 24]. As mentioned in [ 3,
7],weignoretheBeIRArguanacollection,asthiscollec-
tion aims at finding counter-arguments. As for training,
the top 50 documents from the first-stage, SPLADE-v3,
are reranked. nDCG@10 is used as evaluation measure
for all datasets. We provide for all models their results
with a maximum document length of both 256 and 512
tokens.
2

Reranking with Compressed Document Representation
SPLADE-V3 Naver-DeBERTa ModernBERT Mistral 7B RRK RRK 16 Layers
(retriever) 256 512 256 512 256 512 256 512 256 512
TREC
DL 19 72.3 77.6 77.6 76.3 76.3 77.9 77.9 77.1 77.1 77.1 77.1
DL 20 75.4 75.4 75.4 76.7 76.7 76.9 76.9 77.0 77.0 75.8 75.8
BeIR
TREC-COVID 74.8 87.4 88.2 88.0 89.0 85.3 85.6 86.4 86.2 85.9 86.5
NFCorpus 35.7 37.8 37.6 38.2 38.1 37.9 38.5 38.0 38.7 38.3 38.9
NQ 58.6 65.7 65.7 65.9 66.0 67.1 67.2 66.5 66.8 65.1 65.2
HotpotQA 69.2 74.4 74.5 75.3 75.4 74.1 74.1 73.4 73.4 71.4 71.5
FIQA 37.4 47.1 47.8 46.7 47.6 47.0 48.2 46.7 47.1 45.1 46.1
Touché 2020-v2 29.3 31.2 33.5 32.9 35.2 31.6 31.6 28.6 28.9 29.4 28.7
Quora 81.4 84.3 84.3 86.0 86.0 86.4 86.4 87.0 87.0 87.0 87.0
DBPedia 45.0 48.8 48.8 50.1 50.1 50.5 50.5 49.4 49.4 49.1 49.1
SCIDOCS 15.8 19.3 19.2 19.5 19.2 20.1 19.2 19.4 19.5 19.4 19.5
FEVER 79.6 83.5 86.5 85.76 88.4 84.2 86.9 83.4 84.2 81.2 81.7
Climate-FEVER 23.3 25.0 27.4 23.2 25.3 23.9 2 7.5 25.9 25.9 25.8 25.9
SciFact 71.0 76.2 75.8 75.3 75.4 75.4 77.3 75.2 76.4 75.4 74.5
AVG 51.8 56.6 57.4 57.2 57.9 57.0 57.8 56.6 56.9 56.1 56.2
LoTTe
pooled search 53.3 62.2 62.6 62.1 62.5 62.5 62.9 62.4 62.6 61.4 61.1
pooled forum 36.0 45.8 46.4 45.4 45.9 45.9 46.4 45.9 46.3 45.3 45.2
Table 1: Evaluation ( 𝑛𝐷𝐶𝐺 @10∗100) of textual and compressed (RRK) rerankers . The top 50 of the SPLADE-v3
retrieved documents is used as input.
In order to compare the use of compression with re-
gard to the textual representation, we train with the
same setting two models using textual input: a Mistral-
7B as this model was used for the PISCO/RRK model,
and ModernBERT-large [ 29], a ’smaller, better, faster,
longer’[sic] bidirectional encoder, which shows very
competitive results in terms of effectiveness, and effi-
ciency (faster than DebertaV3). Appendix B shows that
the ModernBERT-base version is not able to reach the
teacher level.
The PISCO compressor needs an access to the com-
pressed representation during inference. This access
amount for less than 10% of the total reranking time.
When discussing efficiency, we always refer to Figure 1.
BeIR/TREC-NEWS Document Length Input
256 512 768 1024
Naver-DeBERTa-v3-Large 46.0 50.9 48.6 46.0
ModernBERT-Large 47.4 50.5 51.2 51.5
Mistral 7B 49.4 51.6 52.3 51.2
RRK 46.5 51.2 53.1 51.9
Table 2: Reranking "Long" documents. Increasing the
document length can enhance rerankers effectiveness
without affecting the efficiency of RRK models.5. Results
The results are presented in Table 1. The first col-
umn provides information about the average number of
wordsperqueryanddocument, whichwillbediscussed
later, and the second column evaluates our teacher
model (see Table C for the first stage evaluation).
All models perform similarly, with differences of less
than 1 point using nDCG@10 for the average score.
Notably, the main differences often originate from a
few specific datasets (esp FEVER and Touché). Ignoring
these datasets reduces the differences between Mod-
ernBERT and RRK by 50%. The recent ModernBERT
performs very well and is very efficient, which shows
that the reranking task can be performed with an en-
coder only architecture. The Mistral model fine-tuned
with textual representation achieves similar effective-
ness than ModernBERT, but is simply 20 times slower.
Compared with the RRK model, the differences among
datasets are less than 1 point, except for FEVER and
Touché. We can not explain yet this weakness of the
RRK model with these two datasets.
Letusfirstdrawsomesomegeneralobservations: distil-
lation is effective since all models replicate the teacher’s
results or improve it. Moreover, increasing input length
improves results by 1 point. This might seem marginal,
but the gain is more significant (>2 points) for collec-
3

Reranking with Compressed Document Representation
tions with long documents, like FEVER (full wikipedia
pages). While models utilizing textual representation
aredirectlyaffectedbyincreasedlength,theRRKmodel
is impacted only during offline compression, maintain-
ing constant efficiency in online reranking.
Secondly, we now discuss the key results on reranking
with compressed representation: RRK, our model with
a 32 token input length is up to 16 times faster than its
equivalent using textual representation with almost the
same effectiveness (-0.9 for BeIR, similar for LoTTe).
Furthermore, RRK is also faster than its teacher (Fig-
ure 1), for the same effectiveness. Compared to our
strong baseline, ModernBERT-Large, our RRK model is
a bit less effective for BeIR (-1 pt), but achieves similar
performance for LoTTe. Efficiency-wise, it is half as fast
(Figure 1), but becomes slightly faster with long inputs
(considering 768 tokens or more, see discussion below).
Regarding our RRK model with 16 layers, its efficiency
is remarkable, being faster than Modern-BERT, though
with a reduced effectiveness (-2 points). Besides, we
also attempted to train smaller PISCO models (like
Qwen 3B or Llama 1B), but all attempts have been
unsuccessful so far. Note that the RRK 16 layers is more
effectiveandasfastasthanthesmallModernBERT-base
model (Table 5).
Lastly, the use of the PISCO compressor, trained to
compress 128 tokens into 8 memory tokens, shows
promising results when applied to larger documents
(512tokensandbeyond),asshownfortheTREC-NEWS
collection [ 25] in Table 2. This table demonstrates that
the use of compressed representation is a natural set-
ting for processing long documents: the RRK model
achieves the best results with a 768 token input, while
other models suffers losses in effectiveness, and/or a
high increase of their efficiency (RRK is as fast as Mod-
ernBERT when the latter is fed with a 768 token input).
The decrease in performance observed with 1K tokens
for most models may be due to the PISCO compres-
sor being trained on relatively small documents (128
tokens or fewer). The reranker fine-tuning using this
same maximum document length may also contribute
to the decrease.
Overall, these results show that using compressed rep-
resentations enables a 7B parameter model to run 10
times faster than its textual counterpart, and only half
fast than a very recent 400M parameter model using
all accumulated recipes for improving its effectiveness
and efficiency. At this stage, we can see the glass as
half empty or half full. A good choice is to distill your
favorite reranker with ModernBERT. However the re-
sults we obtain with PISCO’s compressed representa-tion, trained for the generation part of a RAG system,
are very appealing.
6. Conclusion
In this work, we introduce a novel approach to rerank-
ing by utilizing compressed document representations,
significantly enhancing efficiency while maintaining
similar levels of effectiveness. Our results demonstrate
that employing compressed embeddings generated by
the PISCO model—an off-the-shelf compressor model
designed for Retrieval Augmented Generation—can
achieve acceptable performance with better latency
comparedtotraditionalrerankingmethods,particularly
whendocumentsaresufficientlylengthy. Thisefficiency
is achieved by simply reducing the input length for the
model through the use of a compressed representation
consisting of a small sequence of tokens.
Limitations
First, the efficiency of RRK is mostly due to its tiny input
length. This advantage holds as long as the query itself
isshort. UsingdatasetslikeSuetal. [26],wherequeries
lengthiscomparableto(BeIR)documentlength,breaks
this advantage and makes the RRK model slow. An eval-
uation with a query length of 48 tokens yields similar
results for the BeIR collection except for the Scifact
dataset, where the nDCG score rises from 75.1 to 76.4.
Secondly, it would be beneficial to employ smaller mod-
els instead of billion-sized ones as reranker. Unfortu-
nately, our initial attempts to use smaller models, such
as 1B parameter models, have not yet been successful
(to say the least, they failed miserably). Using smaller
models would lead to even better efficiency, and may
reducetheindexfootprint(usingsmallerhiddendimen-
sions).
Finally,thefactthatthePISCOcompressoraswellasthe
RRK model have been trained with a maximum input
length of 128 which certainly limits the effectiveness
of the RRK model for long documents.
References
[1]Payal Bajaj, Daniel Campos, Nick Craswell,
Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan
Majumder, Andrew McNamara, Bhaskar Mitra,
Tri Nguyen, Mir Rosenberg, Xia Song, Alina Sto-
ica, Saurabh Tiwary, and Tong Wang. 2018. Ms
marco: Ahumangeneratedmachinereadingcom-
prehension dataset. Preprint , arXiv:1611.09268.
2
[2]Sebastian Bruch, Franco Maria Nardini, Cosimo
Rulli, and Rossano Venturini. 2024. Efficient
inverted indexes for approximate retrieval over
4

Reranking with Compressed Document Representation
learned sparse representations. In Proceedings of
the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval ,
SIGIR 2024, page 152–162. ACM. 7
[3]Cesare Campagnano, Antonio Mallia, Jack
Pertschuk, and Fabrizio Silvestri. 2025. E2rank:
Efficient and effective layer-wise reranking. In
Advances in Information Retrieval , pages 417–426,
Cham. Springer Nature Switzerland. 2
[4]Hongliu Cao. 2024. Recent advances in text
embedding: A comprehensive review of top-
performing methods on the mteb benchmark.
Preprint , arXiv:2406.01607. 2
[5]Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and
Daniel Campos. 2021. Overview of the trec 2020
deep learning track. Preprint, arXiv:2102.07662.
2
[6]Nick Craswell, Bhaskar Mitra, Emine Yilmaz,
Daniel Campos, and Ellen M. Voorhees. 2020.
Overview of the trec 2019 deep learning track.
Preprint , arXiv:2003.07820. 2
[7]Hervé Déjean, Stéphane Clinchant, and Thibault
Formal. 2024. A thorough comparison of cross-
encoders and llms for reranking splade. Preprint,
arXiv:2403.10407. 1, 2
[8]Revanth Gangi Reddy, JaeHyeok Doo, Yifei Xu,
Md Arafat Sultan, Deevya Swain, Avirup Sil, and
Heng Ji. 2024. FIRST: Faster improved listwise
reranking with single token decoding. In Proceed-
ings of the 2024 Conference on Empirical Methods
inNaturalLanguageProcessing ,pages8642–8652,
Miami, Florida, USA. Association for Computa-
tional Linguistics. 1
[9]Luyu Gao, Zhuyun Dai, and Jamie Callan. 2020.
Modularized transfomer-based ranking frame-
work. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing
(EMNLP) , pages 4180–4190, Online. Association
for Computational Linguistics. 1
[10]Luyu Gao, Zhuyun Dai, and Jamie Callan.
2021. Rethink Training of BERT Rerankers in
Multi-Stage Retrieval Pipeline. arXiv preprint .
ArXiv:2101.08751 [cs]. 1
[11]Tao Ge, Jing Hu, Xun Wang, Si-Qing Chen, and
Furu Wei. 2023. In-context autoencoder for con-
textcompressioninalargelanguagemodel. arXiv
preprint arXiv:2307.06945 . 1
[12]Carlos Lassance and Stéphane Clinchant. 2023.
Naver labs europe (splade) @ trec deep learning
2022.Preprint , arXiv:2302.12574. 2, 8
[13]Carlos Lassance, Hervé Déjean, Stéphane Clin-
chant, and Nicola Tonellotto. 2024. Two-stepsplade: Simple, efficient and effective approxima-
tion of splade. Preprint , arXiv:2404.13357. 7
[14]Carlos Lassance, Hervé Déjean, Thibault Formal,
and Stéphane Clinchant. 2024. Splade-v3: New
baselines for splade. Preprint, arXiv:2403.06789.
2, 8
[15]Sean Lee, Aamir Shakir, Darius Koenig, and
Julius Lippand. 2023. Open source strikes bread
- new fluffy embedding model. https://www.
mixedbread.com/blog/mxbai-embed-large-v1 .
Accessed: 2025-05-05. 8
[16]Zheng Liu, Chaofan Li, Shitao Xiao, Chaozhuo
Li, Defu Lian, and Yingxia Shao. 2025. Ma-
tryoshka re-ranker: A flexible re-ranking architec-
ture with configurable depth and width. Preprint,
arXiv:2501.16302. 1
[17]Maxime Louis, Hervé Déjean, and Stéphane Clin-
chant. 2025. Pisco: Pretty simple compression
for retrieval-augmented generation. Preprint ,
arXiv:2501.16075. 1, 8, 9
[18]Xueguang Ma, Liang Wang, Nan Yang, Furu
Wei, and Jimmy Lin. 2023. Fine-Tuning LLaMA
for Multi-Stage Text Retrieval. arXiv preprint .
ArXiv:2310.08319 [cs]. 1
[19]Sean MacAvaney, Franco Maria Nardini, Raf-
faele Perego, Nicola Tonellotto, Nazli Goharian,
and Ophir Frieder. 2020. Efficient document re-
ranking for transformers by precomputing term
representations. In Proceedings of the 43rd Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval , SIGIR ’20,
page 49–58, New York, NY, USA. Association for
Computing Machinery. 1
[20]Rodrigo Nogueira and Kyunghyun Cho. 2020.
Passage re-ranking with bert. Preprint ,
arXiv:1901.04085. 1
[21]Ronak Pradeep, Rodrigo Nogueira, and Jimmy
Lin. 2021. The expando-mono-duo design pat-
tern for text ranking with pretrained sequence-
to-sequence models. Preprint, arXiv:2101.05667.
1
[22]David Rau, Hervé Déjean, Nadezhda Chirkova,
Thibault Formal, Shuai Wang, Vassilina Nikoulina,
and Stéphane Clinchant. 2024. Bergen: A bench-
marking library for retrieval-augmented genera-
tion.arXiv preprint arXiv:2407.01102 . 7
[23]David Rau, Shuai Wang, Hervé Déjean, and
Stéphane Clinchant. 2024. Context embeddings
for efficient answer generation in rag. Preprint ,
arXiv:2407.09252. 1
[24]Keshav Santhanam, O. Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei A. Zaharia. 2021.
Colbertv2: Effective and efficient retrieval via
5

Reranking with Compressed Document Representation
lightweight late interaction. In North American
Chapter of the Association for Computational Lin-
guistics. 1, 2
[25]Ian Soboroff, Shudong Huang, and Donna Har-
man. 2018. Trec 2018 news track overview. In
TREC, volume 409, page 410. 4
[26]Hongjin Su, Howard Yen, Mengzhou Xia, Weijia
Shi, Niklas Muennighoff, Han-yu Wang, Haisu
Liu, Quan Shi, Zachary S Siegel, Michael Tang,
Ruoxi Sun, Jinsung Yoon, Sercan O Arik, Danqi
Chen, and Tao Yu. 2024. Bright: A realistic and
challenging benchmark for reasoning-intensive
retrieval. 4
[27]Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie
Ren, Dawei Yin, and Zhaochun Ren. 2023. Is
ChatGPT Good at Search? Investigating Large
Language Models as Re-Ranking Agent. arXiv
preprint. ArXiv:2304.09542 [cs]. 1
[28]Nandan Thakur, Nils Reimers, Andreas Ruckl’e,
Abhishek Srivastava, and Iryna Gurevych. 2021.
Beir: A heterogenous benchmark for zero-shot
evaluation of information retrieval models. ArXiv,
abs/2104.08663. 2
[29]Benjamin Warner, Antoine Chaffin, Benjamin
Clavié, Orion Weller, Oskar Hallström, Said
Taghadouini, Alexis Gallagher, Raja Biswas, Faisal
Ladhak, Tom Aarsen, Nathan Cooper, Griffin
Adams, Jeremy Howard, and Iacopo Poli. 2024.
Smarter, better, faster, longer: A modern bidi-
rectional encoder for fast, memory efficient, and
long context finetuning and inference. Preprint ,
arXiv:2412.13663. 3, 8
[30]Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas
Muennighoff, Defu Lian, and Jian-Yun Nie. 2024.
C-pack: Packed resources for general chinese em-
beddings. Preprint , arXiv:2309.07597. 8
[31]Yutao Zhu, Huaying Yuan, Shuting Wang, Jiong-
nan Liu, Wenhan Liu, Chenlong Deng, Haonan
Chen,ZhengLiu,ZhichengDou,andJi-RongWen.
2024. Large language models for information re-
trieval: A survey. Preprint , arXiv:2308.07107.
1
[32]Honglei Zhuang, Zhen Qin, Rolf Jagerman, Kai
Hui, Ji Ma, Jing Lu, Jianmo Ni, Xuanhui Wang,
and Michael Bendersky. 2023. Rankt5: Fine-
tuning t5 for text ranking with ranking losses.
InProceedings of the 46th International ACM SI-
GIR Conference on Research and Development in
InformationRetrieval , SIGIR’23, page2308–2313,
New York, NY, USA. Association for Computing
Machinery. 1
[33]Shengyao Zhuang, Honglei Zhuang, Bevan Koop-
man, and Guido Zuccon. 2023. A Setwise Ap-proach for Effective and Highly Efficient Zero-
shot Ranking with Large Language Models. arXiv
preprint. ArXiv:2310.09497 [cs]. 1
6

Reranking with Compressed Document Representation
A. Selecting our Teacher
All the next experiments have been generated using the
BERGEN benchmark [ 22] and are easily reproducible.
Table 3 presents a combination of publicly available
first-stage retrievers and rerankers, including the tra-
ditional BM25 retriever (see Table 5 in Appendix for
more details about the models). Although it showcases
a limited number of models, this table demonstrates
how to select an appropriate teacher for the distilla-
tion process. Depending on the choice of your first-
stage retriever, which may be influenced by efficiency
constraints, certain available rerankers may be more
suitable than others.
Based on those results, MixBread seems like a vi-
able option as a retriever. However, given that the
Splade/DeBERTa combination offers similar perfor-
mance to MixBread/MixBread one, and prioritizing
efficiency , we choose the sparse model for its greater
speed than a dense one (comparable to BM25 with an
optimized implementation [ 2,13] ). Additionally, both
models have only been trained with the MSMARCO
data, which makes comparison and further investiga-
tion easier. Therefore, in this paper, we choose SPLADE-
V3 as the retriever and provide the top 50 documents
it retrieves to the Naver-DeBERTa reranker. We use
Naver-DeBERTa as our teacher model for distillation.
B. Rerankers
Since we were only using a Mistral 7B backbone in
this paper, it would be interesting to see whether other
backbone or size could change our current results. We
are currently only able to provide some results for a
non compression setting in Table 4. We first see that the
baseversionofMordernBERTisnotabletoreplicatethe
teacher results. Small LLMs as Llama3.2 1B is able to
replicate them, but as mentioned, we were not able to
train a PISCO using such backbone. For larger LLMs (7
billion parameters), no noticeable difference appears.
C. First-Stage Retriever and Rerankers
Table 5 provides the list of models used in this paper.
D. PISCO Architecture
Figure 2 shows how a PISCO model is trained.
7

Reranking with Compressed Document Representation
Retriever BM25 SPLADE-v3 Mix BGE BM25 BM25 BM25 SPLADE-v3 MIXB BGE
+ + + + + +
Reranker DeBERTa mix BGE DeBERTa mix BGE
BeIR
Trec-covid 0.59 0.75 0.76 0.75 0.82 0.81 0.73 0.85 0.87 0.79
SCIDOCS 0.15 0.16 0.23 0.22 0.19 0.19 0.17 0.19 0.20 0.18
NQ 0.31 0.59 0.56 0.55 0.54 0.53 0.53 0.66 0.64 0.50
SciFact 0.68 0.71 0.74 0.74 0.76 0.74 0.70 0.76 0.75 0.69
FIQA 0.24 0.37 0.45 0.44 0.40 0.37 0.34 0.45 0.44 0.39
QUORA 0.79 0.81 0.89 0.89 0.85 0.76 0.80 0.84 0.70 0.75
NFCorpus 0.34 0.36 0.38 0.37 0.38 0.39 0.33 0.38 0.40 0.31
HotpotQA 0.63 0.69 0.72 0.74 0.73 0.76 0.78 0.74 0.74 0.84
DPPedia 0.32 0.45 0.45 0.44 0.42 0.43 0.43 0.49 0.49 0.49
FEVER 0.65 0.80 0.86 0.84 0.80 0.78 0.87 0.78 0.81 0.92
Climate-FEVER 0.17 0.23 0.34 0.28 0.24 0.24 0.31 0.25 0.27 0.37
Touché v2 0.29 0.29 0.23 0.23 0.34 0.35 0.31 0.31 0.24 0.24
AVG 0.43 0.52 0.550.54 0.54 0.54 0.53 0.56 0.56 0.54
Table 3: Which reranker should you use? Rerankers may strongly depend on the choice of first-stage retriever.
NDCG@10. Rerankers consider the top 50 documents and a maximun document length of 256.
SPLADE-V3 DeBERTa-v3 ModernBERT-base ModernBERT-large llama 1B Mistral 7B LLama 8B Qwen 7B
(Retriever) (Teacher)
BeIR
TREC-COVID 0.75 0.87 0.88 0.88 0.86 0.85 0.85 0.85
NFCorpus 0.36 0.38 0.36 0.38 0.37 0.38 0.30.38 0.39
NQ 0.59 0.66 0.62 0.65 0.66 0.67 0.67 0.67
HotpotQA 0.69 0.74 0.72 0.74 0.75 0.74 0.74 0.75
FIQA 0.37 0.47 0.43 0.47 0.45 0.47 0.47 0.47
Touché 2020-v2 0.29 0.31 0.31 0.32 0.31 0.32 0.31 0.30
Quora 0.81 0.84 0.83 0.86 0.86 0.86 0.87 0.85
DBPedia 0.45 0.49 0.48 0.49 0.50 0.51 0.51 0.51
SCIDOCS 0.16 0.19 0.17 0.19 0.20 0.20 0.20 0.20
FEVER 0.80 0.83 0.88 0.85 0.86 0.84 0.85 0.85
Climate-FEVER 0.23 0.25 0.21 0.23 0.25 0.24 0.26 0.25
SciFact 0.71 0.76 0.63 0.76 0.75 0.75 0.76 0.77
AVG 0.52 0.56 0.54 0.57 0.57 0.57 0.57 0.57
Table 4: Evaluation (NDCG@10) of rerankers trained with no compression using Naver-DeBERTa as teacher. The
top 50 documents of the SPLADE-v3 retrieved documents is used as input. Maximum input length 256.
References Hugging Face model
Off-the-shelf Retriever
SPLADE-v3 Lassance et al. [14] naver/splade-v3
BGE Xiao et al. [30] BAAI/bge-large-en-v1.5
MIXBREAD Lee et al. [15] mixedbread-ai/mxbai-embed-large-v1
Off-the-self Reranker
Naver-Deberta Lassance and Clinchant [12] naver/trecdl22-crossencoder-debertav3
MIXBREAD Lee et al. [15] mixedbread-ai/mxbai-rerank-large-v1
BGE Xiao et al. [30] BAAI/bge-reranker-large
Trained Models (Reranker)
Mistral mistralai/Mistral-7B-v0.3
ModernBERT-large Warner et al. [29] answerdotai/ModernBERT-large
Compressor
PISCO Louis et al. [17] naver/pisco-mistral
Table 5: Models used in this article for retrieving, reranking and compressing.
8

Reranking with Compressed Document Representation
Quer y
In the mo vie Psy cho, who is
revealed t o be the true killer ?Document
collectionTeacher answer
In the mo vie Psy cho, the true
killer is r evealed t o be ...
Student answer
(teacher-for cing during tr aining)Cross-Entr opy
LossTeacher annotation
(No Gr ad) Retrie val
...Compr essed
Embeddings...Memor y
tokensCompr ession
(precomputed for inf erence)
......Psycho is a 1960 American
horror  lm pr oduced ...
...
At the police station, a
psychiatrist explains that ...
Inference
Figure 2: PISCO Architecture [ 17]: The compression process utilizes a language model with LoRA adapters,
appending memory tokens to each document to form embeddings, which control the compression rate through
optimization. Decoding involves fine-tuning the decoder to adapt generation with compressed representations
based on queries. The distillation objective employs Sequence-level Knowledge Distillation (SKD) to ensure
models give consistent answers whether inputs are compressed or not.
9