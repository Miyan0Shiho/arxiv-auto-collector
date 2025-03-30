# Dewey Long Context Embedding Model: A Technical Report

**Authors**: Dun Zhang, Panxiang Zou, Yudong Zhou

**Published**: 2025-03-26 09:55:00

**PDF URL**: [http://arxiv.org/pdf/2503.20376v1](http://arxiv.org/pdf/2503.20376v1)

## Abstract
This technical report presents the training methodology and evaluation
results of the open-source dewey_en_beta embedding model. The increasing demand
for retrieval-augmented generation (RAG) systems and the expanding context
window capabilities of large language models (LLMs) have created critical
challenges for conventional embedding models. Current approaches often struggle
to maintain semantic coherence when processing documents exceeding typical
sequence length limitations, significantly impacting retrieval performance in
knowledge-intensive applications. This paper presents dewey_en_beta, a novel
text embedding model that achieves excellent performance on MTEB (Eng, v2) and
LongEmbed benchmark while supporting 128K token sequences. Our technical
contribution centers on chunk alignment training, an innovative methodology
that enables the simultaneous generation of localized chunk embeddings and
global document-level representations through distillation. Information
regarding the model release can be found at
https://huggingface.co/infgrad/dewey_en_beta.

## Full Text


<!-- PDF content starts -->

Dewey Long Context Embedding Model: A Technical Report
Dun Zhang1∗, Panxiang Zou2, Yudong Zhou1
1PRIORSHAPE2RICHINFO
infgrad@163.com zoupanxiang@richinfo.cn
Abstract
This technical report presents the training
methodology and evaluation results of the open-
source dewey_en_beta embedding model. The
increasing demand for retrieval-augmented gen-
eration (RAG) systems and the expanding con-
text window capabilities of large language mod-
els (LLMs) have created critical challenges
for conventional embedding models. Current
approaches often struggle to maintain seman-
tic coherence when processing documents ex-
ceeding typical sequence length limitations,
significantly impacting retrieval performance
in knowledge-intensive applications. This pa-
per presents dewey_en_beta, a novel text em-
bedding model that achieves excellent perfor-
mance on MTEB (Eng, v2)(Enevoldsen et al.,
2025) and LongEmbed benchmark(Zhu et al.,
2024) while supporting 128K token sequences.
Our technical contribution centers on chunk
alignment training, an innovative methodol-
ogy that enables the simultaneous generation
of localized chunk embeddings and global
document-level representations through distil-
lation (Zhang et al., 2025). Information re-
garding the model release can be found at
https://huggingface.co/infgrad
/dewey_en_beta .
1 Introduction
Text embedding models serve as fundamental
components in contemporary information retrieval
systems and retrieval-enhanced language models.
While recent advancements in semantic represen-
tation learning have yielded considerable improve-
ments, practical challenges remain in processing
extended textual content. The sequence length ca-
pacity of embedding models presents an important
technical consideration, as enhanced context pro-
cessing could potentially improve document com-
prehension and facilitate more adaptable retrieval
implementations.
∗ ∗Dun Zhang is the corresponding author.In this technical report, inspired by late chunk-
ing(Günther et al., 2024) we present a preliminary
investigation into chunk-alignment training strate-
gies for text embedding models. This exploratory
approach attempts to address the representation
learning challenges associated with long-form doc-
uments through segment-level feature alignment.
Our experimental framework demonstrates the fea-
sibility of processing text sequences up to 128k to-
kens while preserving performance levels on stan-
dard evaluation benchmarks. The current imple-
mentation offers two operational modes: a conven-
tional single-vector encoding scheme and an exper-
imental multi-vector variant for extended context
processing.
The primary contribution lies in developing a
chunk-alignment training method (achieved by
knowledge distillation) that attempts to extend ex-
isting embedding architectures to longer textual
sequences. It should be emphasized that this repre-
sents an initial exploration rather than a definitive
solution, with the current results suggesting poten-
tial pathways for further investigation. Our com-
parative evaluations indicate relatively balanced
performance across conventional benchmarks and
preliminary long-context test sets, though we ac-
knowledge significant room for improvement in
both theoretical framework and practical imple-
mentation.
2 Training Methodology
Selection of Base Model
After investigating many models, we select
ModernBERT-Large(Warner et al., 2024) as the
base model of dewey_en_beta. ModernBERT is
a modernized bidirectional encoder-only Trans-
former model (BERT-style) pre-trained on 2 trillion
tokens of English and code data with a native con-
text length of up to 8,192 tokens. ModernBERT
leverages recent architectural improvements:arXiv:2503.20376v1  [cs.IR]  26 Mar 2025

•Rotary Positional Embeddings (RoPE)(Su
et al., 2023) for long-context support.
•Local-Global Alternating Attention(Team
et al., 2024) for efficiency on long inputs.
•Unpadding and Flash Attention (Portes et al.,
2023)(Zhang et al., 2024b)(Zeng et al., 2022)
for efficient inference.
To scale ModernBERT’s max length to 128k, we
change the global_rope_theta to 73780400 accord-
ing to (Men et al., 2024) and https://spaces
.ac.cn/archives/10122 .
Chunk-Alignment Training
Our model can generate three types of embed-
dings:
1.CLS embedding: A CLS embedding in a
BERT-like model. In our training process,
it will learn the teacher model’s embedding of
the whole text.
2.Chunk embeddings: The mean embeddings
of chunk token embeddings. In our training
process, it will learn the teacher model’s em-
bedding of each chunk.
3.Mean embedding: The mean embeddings of
all token embeddings (excluding CLS, SEP,
and prompt token embeddings). It is a special
case of chunk embedding (i.e. the chunk is
the whole text).
For a more comprehensive introduction of our
model and distillation framework, we make the
following definitions:
•teacher : a teacher embedding model with
the function encode to encode texts to embed-
dings
•cls_embed, chunk _embed i: CLS embed-
ding and chunk embedding of ithchunk in
student model(i.e. the model to be trained)
•cls_teacher _embed, chunk _teacher _embed i
as the whole text teacher embedding and
chunk teacher embeddings
•sx: The normalized vector representation of a
textxproduced by the student model.
•tx: The vector representation of the same text
x, produced by a teacher model.•SX: A matrix of normalized vector represen-
tations for a batch of text Xproduced by the
student model.
•TX: A corresponding matrix of vector repre-
sentations for the same batch of text X, gen-
erated by a teacher model.
cls_teacher _embed and
chunk _teacher _embed ican be obtained by
the following equations:
cls_teacher _embed =teacher.encode (wholetext )
chunk _embed i=teacher.encode (chunk i)
After getting cls and chunks embeddings of stu-
dent and teacher model, we calculate cosine _loss
1 and similarity _loss 2 as the final training loss.
Lcosine _loss=X
x1−sx·tx. (1)
Lsimilarity _loss=MSE (SXST
X, TXTT
X))(2)
Implementation Details
We use Linq-Embed-Mistral(Kim et al., 2024)
as our teacher model. We get unsupervised texts
from Infinity-Instruct(of Artificial Intelligence ,
BAAI)(Zhao et al., 2024)(Zhang et al., 2024a) and
fineweb-edu(Lozhkov et al., 2024). We take two
strategies to split text to chunks:
1. Split Text by Word
2.RecursiveCharacterTextSplitter in langchain
https://python.langchain.com/d
ocs/introduction/
We chose to use the RecursiveCharacter-
TextSplitter with 70% probability and the Split
Text by Word with 30% probability. All the two
strategies use a randomized chunk_size(from 64 to
500) and chunk_overlap(from 0.3∗chunk _size to
0.6∗chunk _size).
We are training our model with about 10 million
data (about 100 million chunks). The batch size
is set to 64, and the learning rate is 1e-4 with a
2000-step warmup and linear decay. We use the
StableAdamW optimizer(Wortsman et al., 2023),
which improves upon AdamW(Loshchilov and Hut-
ter, 2019) by adding Adafactor-style update clip-
ping as a per-parameter learning rate adjustment.
We total train 2 epochs. We set weight decay to
zero. The max length of training data is 2048.

Figure 1: Model architecture
3 Experimental Results
English Text Embedding Benchmark
MTEB(eng, v2)(Enevoldsen et al., 2025) is a
new English Massive Text Embedding Benchmark.
This benchmark was created to account for the fact
that many models have now been finetuned to tasks
in the original MTEB, and contains tasks that are
not as frequently used for model training. This
way the new benchmark and leaderboard can give
our users a more realistic expectation of models’
generalization performance.
We evaluated our model’s performance on this
benchmark. As shown in 1, while our model
supports a context length of 128k tokens, it still
achieves competitive results, outperforming most
models of comparable size and even some larger-
scale models on this particular benchmark.
LongEmbed Benchmark
LongEmbed(Zhu et al., 2024) is a benchmark
oriented at exploring models’ performance on long-
context retrieval. The benchmark comprises two
synthetic tasks and four carefully chosen real-world
tasks, featuring documents of varying length and
dispersed target information.
As can be observed from 2, our model achieves
reasonably good results with single-vector repre-
sentation, while the performance is further im-
proved to an optimal level when multi-vector rep-
resentation is employed.
LoCoV1 BenchmarkLoCoV1 (Saad-Falcon et al., 2024): a novel
12-tasks benchmark constructed to measure long-
context retrieval where chunking is not possible or
not effective.
4 Conclusion
In this concise technical report, we present prelimi-
nary insights into the dewey model and its training
methodology. The proposed approach employs
chunk-alignment techniques combined with knowl-
edge distillation, trained on extensive unsupervised
data. Our experimental results demonstrate promis-
ing performance across both long-text and short-
text evaluation benchmarks, though we acknowl-
edge these findings represent early-stage research
outcomes.
While observing moderate improvements
through chunk-alignment implementation, we
recognize substantial room for exploration and re-
finement in this methodology. This report records
current training details and empirical observations,
shared with the intention of inviting constructive
feedback and collaborative investigation. We
hope these preliminary findings might serve as a
discussion catalyst within the research community,
particularly regarding potential optimizations in
alignment strategies and scalability enhancements.

Model Zero-shot Parameters Dimensions Max Tokens Mean (Task) Mean (TaskType) Classification Clustering Pair Classification Reranking Retrieval STS Summarization
gemini-embedding-exp-03-07 95% Unknown 3072 8192 73.3 67.67 90.05 59.39 87.7 48.59 64.35 85.29 38.28
jasper_en_vision_language_v1 56% 1B 8960 131072 71.41 66.65 90.27 60.52 88.14 50 56.05 84.37 37.19
gte-Qwen2-7B-instruct NA 7B 3584 32768 70.72 65.77 88.52 58.97 85.9 50.47 58.09 82.69 35.74
stella_en_1.5B_v5 56% 1B 8960 131072 69.43 65.32 89.38 57.06 88.02 50.19 52.42 83.27 36.91
SFR-Embedding-2_R 85% 7B 4096 32768 69.82 65.31 90.54 59.39 88.09 48.99 53.75 80.86 35.54
Linq-Embed-Mistral 95% 7B 4096 32768 69.8 65.29 83 54.07 88.44 49.44 60.14 84.69 37.26
dewey_en_beta 95% 395M 2048 131072 0.68 63.30 81.83 51.75 86.82 46.35 56.32 84.21 35.79
gte-Qwen2-1.5B-instruct NA 1B 8960 32768 67.2 63.26 85.84 53.54 87.52 49.25 50.25 82.51 33.94
GritLM-7B 95% 7B 4096 4096 67.07 63.22 81.25 50.82 87.29 49.59 54.95 83.03 35.65
GritLM-8x7B 95% 57B 4096 4096 66.16 62.42 79.98 51.48 85.23 49.22 52.46 82.93 35.65
Table 1: MTEB(eng, v2) results, rows are sorted in descending order by column Mean (TaskType)
Model Zero-shot Number of Parameters Embedding Dimensions Max Tokens Mean (Task) Mean (TaskType) Retrieval
dewey_en_beta-MultiVectors 100% 395M 2048 131072 86.59 86.59 86.59
voyage-multilingual-2 100% Unknown 1024 32000 79.17 79.17 79.17
voyage-law-2 100% Unknown 1024 16000 78.85 78.85 78.85
dewey_en_beta-SingleVector 100% 395M 2048 131072 77.98 77.98 77.98
voyage-3 100% Unknown 1024 32000 74.06 74.06 74.06
inf-retriever-v1 100% 7B 3584 32768 73.19 73.19 73.19
Table 2: LongEmbed results, rows are sorted in descending order by column Mean (TaskType)
dataset-name bge-m3-8k gte-modernbert-base-8k Linq-Embed-Mistral-4k Linq-Embed-Mistral-8k SFR-Embedding-Mistral-8k e5-mistral-7b-instruct-8k dewey_en_beta-8k dewey_en_beta_64k dewey_en_beta_64k-multi-vectors
2wikimqa_test 0.9271 0.8658 0.8884 0.9067 0.8965 0.8901 0.8953 0.9051 0.9775
courtlistener_HTML_test 0.1933 0.2349 0.3551 0.3670 0.3647 0.3543 0.3415 0.3616 0.4775
courtlistener_Plain_Text_test 0.1888 0.2478 0.3675 0.3761 0.3679 0.3579 0.3377 0.3485 0.4426
gov_report_test 0.9869 0.9750 0.9832 0.9837 0.9816 0.9823 0.9855 0.9883 0.9853
legal_case_reports_test 0.3702 0.4476 0.5398 0.5432 0.5319 0.4850 0.5474 0.5875 0.6534
multifieldqa_test 0.9373 0.9341 0.9345 0.9327 0.9450 0.9321 0.9687 0.9564 0.9754
passage_retrieval_test 0.4493 0.5271 0.3470 0.3407 0.2902 0.3248 0.7562 0.7389 0.8550
qasper_abstract_test 1.0000 0.9806 0.9982 0.9982 0.9973 0.9965 0.9973 0.9982 0.9982
qasper_title_test 0.9860 0.8892 0.9838 0.9833 0.9861 0.9812 0.9742 0.9742 0.9840
qmsum_test 0.6668 0.6307 0.6816 0.7237 0.7169 0.7148 0.7438 0.7613 0.8154
stackoverflow_test 0.9634 0.9087 0.9760 0.9760 0.9766 0.9690 0.9362 0.9369 0.9443
summ_screen_fd_test 0.9320 0.9379 0.9747 0.9635 0.9656 0.9580 0.9796 0.9821 0.9788
Average 0.7168 0.7150 0.7525 0.7579 0.7517 0.7455 0.7886 0.7949 0.8406
Table 3: LoCoV1 ndcg@10 results
References
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua,
Márton Kardos, Ashwin Mathur, David Stap,
Jay Gala, Wissam Siblini, Dominik Krzemi ´nski,
Genta Indra Winata, Saba Sturua, Saiteja Utpala,
Mathieu Ciancone, Marion Schaeffer, Gabriel Se-
queira, Diganta Misra, Shreeya Dhakal, Jonathan
Rystrøm, Roman Solomatin, Ömer Ça ˘gatan, Akash
Kundu, Martin Bernstorff, Shitao Xiao, Akshita
Sukhlecha, Bhavish Pahwa, Rafał Po ´swiata, Kran-
thi Kiran GV , Shawon Ashraf, Daniel Auras, Björn
Plüster, Jan Philipp Harries, Loïc Magne, Isabelle
Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte
Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Kon-
rad Wojtasik, Taemin Lee, Marek Šuppa, Crystina
Zhang, Roberta Rocca, Mohammed Hamdy, Andri-
anos Michail, John Yang, Manuel Faysse, Aleksei
Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani,
Pranjal Chitale, Simone Tedeschi, Nguyen Tai,
Artem Snegirev, Michael Günther, Mengzhou Xia,
Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Kr-
ishnakumar, Anna Maksimova, Silvan Wehrli, Maria
Tikhonova, Henil Panchal, Aleksandr Abramov,
Malte Ostendorff, Zheng Liu, Simon Clematide,
Lester James Miranda, Alena Fenogenova, Guangyu
Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borgh-
ini, Federico Cassano, Hongjin Su, Jimmy Lin,
Howard Yen, Lasse Hansen, Sara Hooker, Cheng-
hao Xiao, Vaibhav Adlakha, Orion Weller, Siva
Reddy, and Niklas Muennighoff. 2025. Mmteb: Mas-
sive multilingual text embedding benchmark. arXiv
preprint arXiv:2502.13595 .Michael Günther, Isabelle Mohr, Daniel James Williams,
Bo Wang, and Han Xiao. 2024. Late chunking: Con-
textual chunk embeddings using long-context embed-
ding models.
Junseong Kim, Seolhwa Lee, Jihoon Kwon, Sangmo
Gu, Yejin Kim, Minkyung Cho, Jy yong Sohn, and
Chanyeol Choi. 2024. Linq-embed-mistral:elevating
text retrieval with improved gpt data through task-
specific control and quality refinement. Linq AI Re-
search Blog.
Ilya Loshchilov and Frank Hutter. 2019. Decoupled
weight decay regularization.
Anton Lozhkov, Loubna Ben Allal, Leandro von Werra,
and Thomas Wolf. 2024. Fineweb-edu: the finest
collection of educational content.
Xin Men, Mingyu Xu, Bingning Wang, Qingyu Zhang,
Hongyu Lin, Xianpei Han, and Weipeng Chen. 2024.
Base of rope bounds context length.
Beijing Academy of Artificial Intelligence (BAAI).
2024. Infinity instruct. arXiv preprint
arXiv:2406.XXXX .
Jacob Portes, Alexander Trott, Sam Havens, DANIEL
KING, Abhinav Venigalla, Moin Nadeem, Nikhil
Sardana, Daya Khudia, and Jonathan Frankle. 2023.
Mosaicbert: A bidirectional encoder optimized for
fast pretraining. In Advances in Neural Information
Processing Systems , volume 36, pages 3106–3130.
Curran Associates, Inc.

Jon Saad-Falcon, Daniel Y . Fu, Simran Arora, Neel
Guha, and Christopher Ré. 2024. Benchmarking and
building long-context retrieval models with loco and
m2-bert.
Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha,
Bo Wen, and Yunfeng Liu. 2023. Roformer: En-
hanced transformer with rotary position embedding.
Gemma Team, Morgane Riviere, Shreya Pathak,
Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupati-
raju, Léonard Hussenot, Thomas Mesnard, Bobak
Shahriari, Alexandre Ramé, Johan Ferret, Peter
Liu, Pouya Tafti, Abe Friesen, Michelle Casbon,
Sabela Ramos, Ravin Kumar, Charline Le Lan,
Sammy Jerome, Anton Tsitsulin, Nino Vieillard,
Piotr Stanczyk, Sertan Girgin, Nikola Momchev,
Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill,
Behnam Neyshabur, Olivier Bachem, Alanna Wal-
ton, Aliaksei Severyn, Alicia Parrish, Aliya Ah-
mad, Allen Hutchison, Alvin Abdagic, Amanda
Carl, Amy Shen, Andy Brock, Andy Coenen, An-
thony Laforge, Antonia Paterson, Ben Bastian, Bilal
Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu
Kumar, Chris Perry, Chris Welty, Christopher A.
Choquette-Choo, Danila Sinopalnikov, David Wein-
berger, Dimple Vijaykumar, Dominika Rogozi ´nska,
Dustin Herbison, Elisa Bandy, Emma Wang, Eric
Noland, Erica Moreira, Evan Senter, Evgenii Elty-
shev, Francesco Visin, Gabriel Rasskin, Gary Wei,
Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna
Klimczak-Pluci ´nska, Harleen Batra, Harsh Dhand,
Ivan Nardini, Jacinda Mein, Jack Zhou, James Svens-
son, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana
Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fer-
nandez, Joost van Amersfoort, Josh Gordon, Josh
Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mo-
hamed, Kartikeya Badola, Kat Black, Katie Mil-
lican, Keelin McDonell, Kelvin Nguyen, Kiranbir
Sodhia, Kish Greene, Lars Lowe Sjoesund, Lau-
ren Usui, Laurent Sifre, Lena Heuermann, Leti-
cia Lago, Lilly McNealus, Livio Baldini Soares,
Logan Kilpatrick, Lucas Dixon, Luciano Martins,
Machel Reid, Manvinder Singh, Mark Iverson, Mar-
tin Görner, Mat Velloso, Mateo Wirth, Matt Davi-
dow, Matt Miller, Matthew Rahtz, Matthew Watson,
Meg Risdal, Mehran Kazemi, Michael Moynihan,
Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi
Rahman, Mohit Khatwani, Natalie Dao, Nenshad
Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay
Chauhan, Oscar Wahltinez, Pankil Botarda, Parker
Barnes, Paul Barham, Paul Michel, Pengchong
Jin, Petko Georgiev, Phil Culliton, Pradeep Kup-
pala, Ramona Comanescu, Ramona Merhej, Reena
Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan
Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah
Cogan, Sarah Perrin, Sébastien M. R. Arnold, Se-
bastian Krause, Shengyang Dai, Shruti Garg, Shruti
Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan,
Ting Yu, Tom Eccles, Tom Hennigan, Tomas Ko-
cisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh
Meshram, Vishal Dharmadhikari, Warren Barkley,
Wei Wei, Wenming Ye, Woohyun Han, Woosuk
Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, ZichuanWei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh
Giang, Ludovic Peran, Tris Warkentin, Eli Collins,
Joelle Barral, Zoubin Ghahramani, Raia Hadsell,
D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov,
Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray
Kavukcuoglu, Clement Farabet, Elena Buchatskaya,
Sebastian Borgeaud, Noah Fiedel, Armand Joulin,
Kathleen Kenealy, Robert Dadashi, and Alek An-
dreev. 2024. Gemma 2: Improving open language
models at a practical size.
Benjamin Warner, Antoine Chaffin, Benjamin Clavié,
Orion Weller, Oskar Hallström, Said Taghadouini,
Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom
Aarsen, Nathan Cooper, Griffin Adams, Jeremy
Howard, and Iacopo Poli. 2024. Smarter, better,
faster, longer: A modern bidirectional encoder for
fast, memory efficient, and long context finetuning
and inference.
Mitchell Wortsman, Tim Dettmers, Luke Zettlemoyer,
Ari Morcos, Ali Farhadi, and Ludwig Schmidt. 2023.
Stable and low-precision training for large-scale
vision-language models.
Jinle Zeng, Min Li, Zhihua Wu, Jiaqi Liu, Yuang Liu,
Dianhai Yu, and Yanjun Ma. 2022. Boosting dis-
tributed training performance of the unpadded bert
model.
Bo-Wen Zhang, Yan Yan, Lin Li, and Guang Liu. 2024a.
Infinitymath: A scalable instruction tuning dataset in
programmatic mathematical reasoning.
Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong
Wang. 2025. Jasper and stella: distillation of sota
embedding models.
Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie,
Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang,
Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li,
and Min Zhang. 2024b. mGTE: Generalized long-
context text representation and reranking models for
multilingual text retrieval. In Proceedings of the 2024
Conference on Empirical Methods in Natural Lan-
guage Processing: Industry Track , pages 1393–1412,
Miami, Florida, US. Association for Computational
Linguistics.
Hanyu Zhao, Li Du, Yiming Ju, Chengwei Wu, and
Tengfei Pan. 2024. Beyond iid: Optimizing instruc-
tion learning from the perspective of instruction in-
teraction and dependency.
Dawei Zhu, Liang Wang, Nan Yang, Yifan Song, Wen-
hao Wu, Furu Wei, and Sujian Li. 2024. Longembed:
Extending embedding models for long context re-
trieval. arXiv preprint arXiv:2404.12096 .