# From Ranking to Selection: A Simple but Efficient Dynamic Passage Selector for Retrieval Augmented Generation

**Authors**: Siyuan Meng, Junming Liu, Yirong Chen, Song Mao, Pinlong Cai, Guohang Yan, Botian Shi, Ding Wang

**Published**: 2025-08-13 05:05:34

**PDF URL**: [http://arxiv.org/pdf/2508.09497v1](http://arxiv.org/pdf/2508.09497v1)

## Abstract
Retrieval-augmented generation (RAG) systems are often bottlenecked by their
reranking modules, which typically score passages independently and select a
fixed Top-K size. This approach struggles with complex multi-hop queries that
require synthesizing evidence across multiple documents, creating a trade-off
where small K values omit crucial information and large K values introduce
noise. To address this, we introduce the Dynamic Passage Selector (DPS), a
novel reranking framework that treats passage selection as a supervised
learning problem. Unlike traditional point-wise or list-wise methods, DPS is
fine-tuned to capture inter-passage dependencies and dynamically select the
most relevant set of passages for generation. As a seamless plug-and-play
module, DPS requires no modifications to the standard RAG pipeline.
Comprehensive evaluations on five benchmarks show that DPS consistently
outperforms state-of-the-art rerankers and fine-tuning methods. Notably, on the
challenging MuSiQue dataset, DPS improves the F1-score by 30.06% and 15.4% over
strong baselines like Qwen3-reranker and RankingGPT, respectively. Our results
demonstrate that by enabling adaptive evidence selection, DPS substantially
enhances reasoning capabilities in complex RAG scenarios.

## Full Text


<!-- PDF content starts -->

From Ranking to Selection: A Simple but Efficient
Dynamic Passage Selector for Retrieval Augmented Generation
Siyuan Meng1,2Junming Liu1, Yirong Chen1, Song Mao1, Pinlong Cai1, Guohang Yan1,
Botian Shi1, Ding Wang1*
1Shanghai Artificial Intelligence Laboratory
2East China Normal University
mengsiyuancm@gmail.com, wangding@pjlab.org.cn
Abstract
Retrieval-augmented generation (RAG) systems are often
bottlenecked by their reranking modules, which typically
score passages independently and select a fixed Top-K size.
This approach struggles with complex multi-hop queries that
require synthesizing evidence across multiple documents,
creating a trade-off where small K values omit crucial in-
formation and large K values introduce noise. To address
this, we introduce the Dynamic Passage Selector (DPS), a
novel reranking framework that treats passage selection as
a supervised learning problem. Unlike traditional point-wise
or list-wise methods, DPS is fine-tuned to capture inter-
passage dependencies and dynamically select the most rel-
evant set of passages for generation. As a seamless plug-
and-play module, DPS requires no modifications to the
standard RAG pipeline. Comprehensive evaluations on five
benchmarks show that DPS consistently outperforms state-
of-the-art rerankers and fine-tuning methods. Notably, on the
challenging MuSiQue dataset, DPS improves the F1-score
by 30.06% and 15.4% over strong baselines like Qwen3-
reranker and RankingGPT, respectively. Our results demon-
strate that by enabling adaptive evidence selection, DPS sub-
stantially enhances reasoning capabilities in complex RAG
scenarios.
Introdction
Information retrieval (IR) (V oorhees 1999) plays a funda-
mental role in many natural language processing (NLP)
tasks (Hambarde and Proenc ¸a 2023; Wang et al. 2024),
especially in retrieval-augmented generation (RAG) sys-
tems (Xia et al. 2025; Liu et al. 2025). These systems
retrieve relevant passages to support the generation pro-
cess, improving the factual accuracy and reducing halluci-
nations (Lewis et al. 2020). In typical RAG pipelines, a fast
retriever is often used to select a candidate set of passages
from a large corpus, which are then re-scored by a reranker
to prioritize passages that are more contextually relevant to
the input query.
Traditional reranking strategies, whether they operate at
the pointwise (Nogueira et al. 2020a; Liang et al. 2023),
pairwise (Yu et al. 2018; Qin et al. 2024), or listwise (Thonet
et al. 2022) level, typically follow a two-step routine: score
*Corresponding author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Top2 Top3 Top5 Top8 Top10
TopK0.00.20.40.60.8Score
HotpotQA
F1
EM
Recall
Top2 Top3 Top5 Top8 Top10
TopK0.00.10.20.30.4
MuSiQue
F1
EM
RecallFigure 1: Impact of Top-K passage retrieval on multi-hop
QA performance. F1, EM, and Recall scores are plotted for
HotpotQA and MuSiQue datasets.
each passage on its relevance to the query, and then pick
the top K candidates as the input for the generator (Li et al.
2023; Meng et al. 2024). Although effective for single-
hop retrieval, this approach overlooks the interdependence
among passages—a crucial factor in multi-hop and complex
reasoning scenarios, including QFS, long-context process-
ing, and high-level query reasoning (Edge et al. 2024; Qian
et al. 2024). In many cases, the information required to an-
swer a question is distributed across multiple sources, some
of which might not be highly ranked. Consequently, tradi-
tional reranking methods face an inherent trade-off: if Kis
too small, essential but lower-ranked evidence may be omit-
ted; if Kis too large, irrelevant or distracting content may
overwhelm the generator, leading to decreased performance.
To further illustrate this point, we conduct a prelimi-
nary analysis on two widely used multi-hop question an-
swering benchmarks, HotpotQA (Yang et al. 2018) and
MuSiQue (Trivedi et al. 2022). Our findings highlight the
key limitation of traditional reranking approaches: A fixed
Top-K retrieval strategy limits the ability of RAG sys-
tems to handle multi-hop and complex queries As illus-
trated in Figures 1, the optimal value of Kis highly query-
dependent, K=5for HotpotQA and K=3for MuSiQue. In-
creasing Kfrom 2 to 8 improves recall by up to 13.21% and
53.26% on HotpotQA and MuSiQue, respectively, demon-
strating the benefit of incorporating more contextual infor-
mation. However, the F1 score drops sharply beyond K= 5
on HotpotQA and K= 3on MuSiQue, indicating diminish-
ing or even negative returns as irrelevant content begins toarXiv:2508.09497v1  [cs.CL]  13 Aug 2025

dilute the useful evidence.
To address this problem, we propose to reframe rerank-
ing as a passage selection learning problem, where large
language models are fine-tuned through supervised learn-
ing (SFT) to identify the minimal set of passages that col-
lectively support answering a query. Rather than assigning
scalar relevance scores to each passage, the proposed frame-
work, Dynamic Passage Selector (DPS) , learns to reason
about the combinatorial relevance. This formulation allows
the model to explicitly capture the inter-passage dependen-
cies critical for multi-hop question answering, understand-
ing when a single passage would suffice or when multiple
passages must be selected together.
As illustrated in Figure 2, the number of passages re-
quired to answer the query varies between query A and
query B. Traditional reranking approaches retrieve a fixed
Top-K set of passages, which introduces redundant infor-
mation that may compromise answer quality or lead to in-
correct responses. In contrast, DPS dynamically selects the
minimal yet sufficient subset of passages based on query-
specific needs, thereby ensuring the quality of generated
content. Additionally, recent methods attempt to align re-
trieval and ranking in RAG pipelines (Li et al. 2024; Zhang
et al. 2025a; Zuccon, Zhuang, and Ma 2025), they often rely
on costly multi-stage API calls or require large-scale fine-
tuning. By contrast, the proposed DPS can be seen as a plug-
and-play component that integrates seamlessly into exist-
ing RAG workflows without fine-tuning process or pipeline
modifications.
We validate our approach on a range of question answer-
ing benchmarks with varying levels of difficulty, includ-
ing three popular multi-hop question answer datasets Hot-
potQA (Yang et al. 2018), MuSiQue (Trivedi et al. 2022),
2WikiMQA (Ho et al. 2020), as well as two out-of-domain
datasets Legal and CS (Qian et al. 2024). Experimental re-
sults show that our dynamic passage selector consistently
outperforms traditional point-wise rerankers and state-of-
the-art LLM-based list-wise rerankers, especially on com-
plex multi-hop queries. The results demonstrate that DPS
can dynamically identify contextually relevant passages,
surpassing the limitations of fixed top-K retrieval. By more
effectively aligning retrieval with generation, DPS leads to
more robust end-to-end performance. Our main contribu-
tions are as follows:
• To the best of our knowledge, this is the first work that
identifies and addresses a critical limitation in RAG sys-
tems: the fixed Top-K retrieval strategy severely impairs
RAG system’s performance on multi-hop and complex
queries.
• We treat reranking as a passage selection problem, fine-
tune LLMs to dynamically select a minimal set of suf-
ficient passages by modeling their interdependencies,
enabling accurate retrieval for multi-hop and complex
query, while remaining fully plug-and-play with existing
RAG systems.
• Through extensive evaluations across five diverse ques-
tion answering benchmarks, we demonstrate that DPS
achieves state-of-the-art performance on both challeng-ing multi-hop and out-of-domain datasets (e.g., 30.06%
F1 increase over Qwen3-reranker on MuSiQue), validat-
ing its capabilities and robustness for RAG systems.
Related Work
Traditional and Neural Reranking
Text ranking is a core task in IR, where the goal is to
identify and rank the most relevant documents for a user
query (Yates, Nogueira, and Lin 2021). Modern IR sys-
tems often adopt a two-stage architecture: a first-stage re-
triever (e.g., BM25 (Robertson et al. 1995) or dense retriev-
ers (Kang et al. 2025; Ma et al. 2025)) retrieves a broad
set of candidate documents, and a reranker refines this list
to ensure the top-ranked documents are semantically rel-
evant (Lin et al. 2021). Rerankers model fine-grained se-
mantic interactions and significantly boost performance over
first-stage retrieval. Recent reranking approaches predomi-
nantly rely on cross-encoder architectures, where the query
and each document are jointly encoded to compute seman-
tic similarity (Gao et al. 2025). Open-source models like
bge-reranker-large (Chen et al. 2024a) offer strong perfor-
mance and allow flexible fine-tuning across domains, rival-
ing proprietary models such as Cohere-reranker (Cohere Inc.
2024). Beyond cross-encoders, T5-based models such as
MonoT5 (Nogueira et al. 2020b) and RankT5 (Zhuang et al.
2023) reformulate ranking as a sequence generation task,
showing that encoder-only variants may suffice for many
reranking problems. However, these models typically treat
each document independently—following pointwise or pair-
wise ranking paradigms—which limits their ability to cap-
ture inter-document dependencies, especially for complex
reasoning tasks.
LLM-based Approaches
With the rise of LLMs, new efforts aim to leverage their
generative abilities for listwise ranking (Zhuang et al. 2024;
Podolak et al. 2025). For example, Listwise Ranking via
Language Models prompts GPT-3 to rank documents by
generating an ordered list of document IDs (Ma et al. 2023),
allowing global relevance reasoning over all candidates. De-
spite the promise of LLMs in reranking, Liu et al. observe
that vanilla prompting often fails, particularly in smaller-
scale LLMs, due to the misalignment between next-token
prediction objectives and ranking tasks (Qin et al. 2024).
Their proposed Pairwise Ranking Prompting (PRP) com-
bines prompting with scoring, improving ranking perfor-
mance even for open-source models like LLaMA and Mis-
tral. Similarly, RankingGPT (Zhang et al. 2023) addresses
this misalignment by introducing a two-stage training frame-
work: query generation pretraining followed by supervised
fine-tuning with hard negatives. Their results emphasize that
supervised fine-tuning is critical for aligning LLM behavior
with ranking objectives. Although these works explore list-
wise and generation-based paradigms, they often still oper-
ate at the level of individual documents and do not explic-
itly model the combinatorial relationships needed for multi-
hop reasoning. Our approach builds on this line of work
but moves beyond scalar relevance estimation by treating

LLM
Reranker
Retriever
Rank 1
Rank 2
Rank K...0.95
0.91
0.12Corpus Query
Rank 3 0.87Top-K ChunksQuery: Which club did Ronaldinho  and
Kaká  play for at the same time?
1. Tittle: 2006 Football World Cup
Context : Brazil's national team is one of the favorites to win
the championship, boasting legendary stars like Ronaldo ,
Ronaldinho , and Kaká ...N. Tittle: Ronaldinho
Context : On July 15, 2008 , Ronaldinho transferred
from FC Barcelona to AC Milan   for a transfer fee of
€21 million, with an annual salary of €6.5 million.
K. Tittle: Kaká
Context : During the 2008-09 season, Kaká  made 31
appearances in Serie A, scoring 16 goals to help AC
Milan  regain Champions League qualification. 3. Tittle: Ballon d'Or
Context : 2005-Ronaldinho -FC Barcelona- Brazil,  2006-
Cannavaro -Real Madrid CF -Italy , 2007-Kaká -AC Milan-
Brazil
1. Tittle: Kaká,
Context : During the  2008-09  season , Kaká made 31
appearances in Serie A, scoring 16 goals to help AC
Milan  regain Champions League qualification. 
2. Tittle: Ronaldinho
Context : On July 15, 2008, Ronaldinho  transferred from
FC Barcelona to AC Milan   for a transfer fee of €21
million, with an annual salary of €6.5 million.Generation
-Query-:
Which club did Ronaldinho and Kaká play for at the
same time?Dymnatic N Golden Chunks
Infer ence
You should gives helpful  and precise answers to the
user's ques-tions based on the context.
-Context-:
Tittle:Kaká, context:During the 2008-09 season, Kaká
made 31 appearances in Serie A, scoring 16 goals to
help AC Milan ...
-Answer -:
:  AC Milan
........
Rank 1
Rank 2
Rank K...0.95
0.91
0.12Rank 3 0.87Initial Retrieval
Rank 1
Rank 2
Rank K...0.95
Rank 3LLM-Reranker
0.85
0.75
0.2
A relevance score to each document
and then and then sorting documents
based on these scores. Identify minimal yet suf ficient sets of
documents that collectively yield a better 
results.
Knowledge 1 Fine-tuningEasy DifficultKnowledge 2 Fine-tuning
Knowledge 3 Fine-tuning
Knowledge n Fine-tuning
............Pretrain LLM
Curriculum Learning
Target ModelCurriculum-inspired Learning ProcessSelector
RerankerRetrieval
Top-N ChunksDynamic P Chunks
Fixed K Chunks
User
Query A
Query BCorpus
Embedding Vector Database
Golden Revelant Noisy
Query A
Query B
Query A
Query B
Wrong Answer
Correct AnswerFigure 2: Comparison between fixed- Kreranking and the proposed dynamic passage selector DPS. Given a user query, a
retrieval module fetches Top- Ncandidate passages from a large corpus. DPS adaptively selects a minimal yet sufficient subset
of passages, enabling more accurate multi-hop reasoning and generation.
reranking as a document selection problem guided by rea-
soning over document sets.
Methodology
This section presents the core methodology of DPS. As
shown in Figure 3, the DPS framework comprises two pri-
mary stages: offline supervised fine-tuning and online infer-
ence. During training, DPS formulates passage selection as a
conditional sequence modeling task, enabling the model to
learn the interdependencies among candidate passages and
to identify the most informative subset. At inference time,
DPS functions as a lightweight, plug-and-play component
that can be readily integrated into existing RAG pipelines
without requiring architectural modifications and additional
training.
Problem Formulation
Given a natural language query q∈Qand a large text corpus
C={c1, c2, . . .}, a fast retriever module retrieves a candi-
date set of passages:
P={p1, p2, . . . , p n} ⊆ C ,
where each pidenotes a retrieved passage relevant to the
query q, and nis the number of retrieved candidates (typi-
cally large, e.g., Top-100).
Traditional rerankers assign an independent relevance
score si=f(q, pi)to each passage using a scoring function
f(·), and select the top- Kpassages based on these scores:
Stop-K={pi|i∈argtop|S|=Ksi},
whereStop-K⊆ P denotes the fixed-size selected subset and
Kis a predefined constant.However, this standard approach has several key limita-
tions that we discussed before. First, the value of Kis query-
agnostic and fails to adapt to varying levels of query com-
plexity. For instance, single-hop questions may require only
one passage, whereas complex multi-hop questions may de-
mand a larger and more diverse evidence set. Second, the
scoring function f(q, pi)in traditional methods lacks ex-
plicit modeling of subset-level reasoning, and consequently
struggles to identify combinations of passages that jointly
offer comprehensive support for answer generation.
In contrast, our proposed DPS learns to select a query-
specific subset of passages of variable size, optimized to sup-
port accurate answer generation with minimal redundancy.
The selection process is formulated as a sequence prediction
task, which we describe in the following section.
Dynamic Passage Selection
We reformulate passage selection as a structured predic-
tion problem over variable-length subsets of retrieved can-
didates. The training and inference process of DPS is illus-
trated in Figure 3. Given a natural language query q∈Q
and a retrieved set of passages P={p1, p2, . . . , p n}, the
goal is to select a minimal yet sufficient subset S⊆ P
that supports accurate answer generation. We model this as
learning a conditional probability distribution over subsets:
Pθ(S|q,P),where S⊆ P,
where θdenotes model parameters, and the size |S|=kis
not fixed, but instead determined dynamically depending on
the complexity of the input query q.
To enable efficient modeling, we represent the subset S
as an ordered sequence of indices (i1, i2, . . . , i k), where
ij∈ {1, . . . , n }corresponds to the position of passage

Fine-tuningEasy DifficultFine-tuning
Fine-tuningPretrained LLM
Fine-tuining
DPSDPS  Training Stage
Alan Turing
UKMathematiciannationality profession
....Alan Turing Alan Turing
RunningAlan TuringUK
hobby 
UK EnigmabrokeEnigmabroke
broke
Enigmanationality invented 
Turing Machine
nation... Runninghobby 
typeCipher Machine
most_used_in
World War  II Who is the person who
 broke  the Enigma  cipher 
 machine?Query Documents
 What are the  hobbies  of 
 the person who  broke  the
 Enigma  code?Query
 What are the  hobbies  of the
 person who  broke  the  most- 
 used cipher machine  in
 World War IIQuery
Selected Document
Identical Knowledge
Across Documents
Unmatched Knowledge
Matched KnowledgeDocuments
DocumentsDPS
Retriever
Rank 1
Rank 2
Rank N...0.95
0.91
0.12Corpus Query
Rank 3 0.87Top-N Chunks
1. Tittle : 2006 Football World Cup
Context : Brazil's national team is one of the favorites to
win the championship, boasting legendary stars like Ronaldo ,
Ronaldinho , and Kaká ...K. Tittle : Ronaldinho
Context : On July 15, 2008 , Ronaldinho transferred
from FC Barcelona to AC Milan   for a transfer fee
of €21 million, with an annual salary of €6.5 million.
N. Tittle : Kaká
Context : During the 2008-09 season, Kaká  made 31
appearances in Serie A, scoring 16 goals to help  AC
Milan  regain Champions League qualification. 3. Tittle : Ballon d'Or
Context : 2005- Ronaldinho -FC Barcelona- Brazil,   2006-
Cannavaro -Real Madrid CF -Italy,  2007- Kaká -AC Milan-Brazil
1. Tittle : Kaká,
Context : During the  2008-09  season , Kaká made 31
appearances in Serie A, scoring 16 goals to help  AC
Milan  regain Champions League qualification. 
2. Tittle : Ronaldinho
Context : On July 15, 2008, Ronaldinho  transferred from FC
Barcelona to AC Milan   for a transfer fee of €21 million,
with an annual salary of €6.5 million.Generation
-Query-:
Which club did Ronaldinho and Kaká play for
at the same time?Dymnatic K  Golden Chunks
Inference
You should gives helpful  and precise answers to the
user's questions based on the context.
-Context-:
Tittle: Kaká
Context: During the 2008-09 season , Kaká made 31
appearances in Serie A, scoring 16 goals to help AC
Milan  regain Champions League 
Title:...
-Answer-:
:  AC Milan
........Query:  Which club did Ronaldinho  and Kaká  play
for at the same time?DPS  Inference Stage
Figure 3: During the training phase, we treat single-hop question answering data as easy samples, while multi-hop question
answering and complex queries as difficult samples. During the inference phase, DPS adaptively selects K passages from the N
passages retrieved by fast retriever to provide to the generator.
pij∈ P . The distribution is then factorized in an autore-
gressive manner:
Pθ(S|q,P) =kY
j=1Pθ(ij|q,P, i<j),
where i<j= (i1, . . . , i j−1)denotes the previously selected
indices. This formulation has two key benefits. First, each
decision P(ij|q,P, i<j)is explicitly conditioned on the
query q, ensuring alignment with the information need. Sec-
ondly, conditioning on i<jcaptures redundancy or comple-
mentarity among passages, promoting diversity and coher-
ence in the final subset. This autoregressive structure allows
the model to terminate generation once a sufficient set has
been selected, thus avoiding the need to predefine the subset
sizek.
We implement Pθusing a decoder-only large language
model fine-tuned for passage index prediction. The model
input is constructed by concatenating the query and the re-
trieved passages, where each passage piis prepended with a
unique index token:
Input =Query: q∥Passages: [1]p1[2]p2. . .[n]pn,
where [i]denotes the index token associated with passage
pi. The model is trained to generate the output sequence:
Output =i1, i2, . . . , i k.
where each ijindicates the index of a selected passage. This
formulation enables the model to directly predict informa-
tive subsets by leveraging the LLM’s strong sequence mod-
eling and in-context reasoning capabilities, all without re-
quiring architectural changes.
Supervised Fine-tuning
To effectively train the model, we construct a supervised
dataset containing tuples (q,P, S∗), where:•qis the input query,
•P={p1, . . . , p n}is the retrieved candidate passage set,
•S∗= (i∗
1, . . . , i∗
k)is the ground-truth ordered subset of
indices sufficient to answer q.
Given a training dataset D={(q(m),P(m), S∗(m))}M
m=1,
we minimize the sequence-level cross-entropy loss:
L(θ) =−1
MMX
m=1kmX
j=1logPθ 
i∗(m)
j|q(m),P(m), i∗(m)
<j
,
where mdenotes the m-th training instance in the dataset
of size M,i∗(m)
<j = (i∗(m)
1, . . . , i∗(m)
j−1)denotes the previ-
ously selected indices.
This training process encourages the model to: 1) predict
a minimal set of passages that jointly cover the required ev-
idence; 2) capture passage-level dependencies and avoid re-
dundancy; 3) adapt to variable query complexity by gener-
ating dynamic-length outputs.
Inference Strategy
At inference time, DPS receives a query qand a retrieved
candidate set P={p1, . . . , p n}obtained from a fast re-
triever (e.g., BM25 or embedding model). The model per-
forms autoregressive generation over passage indices to con-
struct a variable-sized subset S= (i1, i2, . . . , i k), where
each index ij∈ {1, . . . , n }denotes the position of a selected
passage in P.
The generation proceeds step-by-step, conditioned on the
query and previously selected indices:
ij∼Pθ(ij|q,P, i<j),forj= 1,2, . . . , k.
The final output is a dynamically determined subset S⊆
P, which is concatenated and used as the context input to
the downstream generator. Importantly, this process requires
no changes to the generator architecture or training pipeline,
making DPS a plug-and-play component compatible with
standard RAG systems.

Data Source Type Size
MS MARCO single-hop and multi-hop 100k
HotpotQA multi-hop 90k
MuSiQue multi-hop 20k
Synthetic data long context and high-level 5k
Table 1: Specification of training data.
Fine-tuning Data Construction
Fine-tuning the DPS model requires a corpus of accurately
labeled QA data, including high-quality examples of both
single-hop and multi-hop/complex queries, each paired with
its set of supporting passages (Detailed statistics are pro-
vided in the appendix.). Such data is crucial for training
the model to capture passage-level dependencies between a
query and the evidence needed to answer it. To this end, we
perform comprehensive data collection (shown as Table 1)
from two primary sources: existing labeled QA datasets and
synthetic generation.
We collect a diverse and high-quality set of fine-tuning
data from established benchmarks.For multi-hop QA, we in-
corporate data from HotpotQA and MuSiQue, which source
their corpora from Wikipedia. These datasets provide exam-
ples where multiple documents are needed to formulate an
answer. For single-hop QA, to bolster the model’s single-
hop reasoning capabilities and diversify our data distribu-
tion, we curated 100k question-passage pairs from the MS
MARCO (Bajaj et al. 2016) dataset. The vast majority of
these (approximately 95%) are single-evidence queries, re-
quiring only one passage for a complete answer.
We generate synthetic data to mitigate the shortage of
long context and high-level queries. We use GPT-4o to gen-
erate synthetic data, and the prompt is “You are a curious
AI assistant, please generate one specific and valuable ques-
tion based on the following passages. The generated ques-
tion should revolve around the core content of this text. Note
that the generated questions need to be answered by combin-
ing information from all the passages:”.
Experiment
Experitmental Setups
Datasets. We evaluate DPS on five widely used ques-
tion answering benchmarks covering both multi-hop
and domain-specific reasoning. HotpotQA (Yang et al.
2018) requires concatenating evidence from two support-
ing Wikipedia paragraphs to answer complex questions.
MuSiQue (Trivedi et al. 2022) similarly provides multi-
hop queries with paragraph-level annotations, but empha-
sizes compositional reasoning chains. 2WikiMQA (Ho et al.
2020) comprises factoid questions that may span one or two
Wikipedia articles. To evaluate long context and high-level
query, we include Legal (Qian et al. 2024), a collection of
legal-domain questions with specialized terminology, and
CS (Qian et al. 2024), which focuses on computer science
and programming knowledge.Baselines. We compare against popular pre-trained
models in zero-shot mode (GPT-4o, Gemini-1.5-flash,
Qwen2.5-72B, and Llama3.1-70B) to measure the ben-
efit of retrieval augmentation. In the fine-tuned RAG
category, we include DPA-RAG (Dong et al. 2024) and
RankRAG (Yu et al. 2024), both of which jointly optimize
retriever and generator components using passage-level
supervision and contrastive or distillation objectives.
For reranker-augmented methods, we evaluated seven
standard rerankers—BGE-reranker-v2-m3(Chen et al.
2024a), Qwen3-reranker-8B(Zhang et al. 2025b), Mono-
T5(Nogueira et al. 2020b), RankingGPT (Zhang et al. 2023),
RankLlama (Ma et al. 2024), RankZephyr(Pradeep, Shari-
fymoghaddam, and Lin 2023b) and RankVicuna (Pradeep,
Sharifymoghaddam, and Lin 2023a)—that apply a pre-
trained reranking model over fast retriever candidates before
feeding the top subset into an LLM.
Evaluation. Compared to previous reranking models that
focus on retrieval-based metrics, we place greater empha-
sis on whether the retrieved passages can improve the qual-
ity of the generated content. All methods are assessed using
token-level F1 and Exact Match (EM) metrics on multi-hop
QA dataset. For the CS and Legal datasets, the ground-truth
consists of long-form answers, so only F1 is used as the eval-
uation metric. The experiments are conducted on a single
NVIDIA A100 GPU with a fixed test batch size and tem-
perature. Each configuration is run three times with differ-
ent random seeds. During evaluation, for each dataset, we
vary the top-k parameter in baseline rerankers from 1 to 8
to measure performance across different evidence sizes, and
find K=5 is the optimal settings for most rerankers on five
datasets.
Implementation. We fine-tune DPS on Qwen2.5-7B us-
ing LoRA adapters on 4 NVIDIA A100 GPUs. We set train-
ing epoch I= 1 and use the DeepSpeed ZeRO strategy
during training, with learning rates of 1e−4and a warm ra-
tio of 0.05. To ensure fairness, we use the same vector re-
trieval method to retrieve 30 documents from the identical
corpus, serving as the candidate set for both the reranker and
DPS. We use one of the most advanced text-encoding mod-
els, BGE-M3 (Chen et al. 2024b), as the fast retriever across
all reranker methods. We use Qwen2.5-7B as the generator
(backbone LLMs) for all the baselines in main results. The
detail implementation of baseliens and our source code can
be found in the Appendix.
Main Results
As shown in Table 2, DPS achieves SOTA performance
across all five QA benchmarks, outperforming both fine-
tuned generation baselines (e.g., DPA-RAG, RankRAG)
and recent LLM-based reranking models (e.g., Ranking-
GPT, RankLlama). On average, DPS improves F1 scores
by 8.1% and EM scores by 7.9% over strong rerankers.
On multi-hop datasets such as HotpotQA, MuSiQue, and
2WikiMQA, DPS achieves consistent improvements over
strong reranking-based baselines. Specifically, DPS im-
proves F1 by 5.2 points on HotpotQA (66.34 vs. 61.12
against RankingGPT) and 3.87 points on MuSiQue (28.85

Method/Dataset HotpotQA Musique 2WikiMQA Legal CS
Metric F1 EM F1 EM F1 EM F1 EM F1 EM
w/o RAG
GPT-4o 41.72 30.74 14.07 05.97 34.31 28.42 18.27 - 20.51 -
Gemini1.5-flash 29.74 22.75 07.59 03.22 28.11 25.71 16.93 - 17.76 -
Qwen2.5-72B 23.20 18.47 03.32 01.53 14.42 12.91 14.00 - 14.60 -
Llama3.1-70B 40.06 28.66 13.75 04.56 31.04 25.39 17.36 - 17.86 -
Fine-tuning Method
DPA-RAG 57.39 46.72 20.45 17.36 39.82 39.22 - - - -
RankRAG 46.70 35.30 - - 36.90 31.40 - - - -
Reranker
BGE-reranker-v2-m3 58.17 45.46 19.14 10.49 23.44 19.35 23.92 - 19.47 -
Qwen3-reranker-8B 60.45 47.37 22.08 13.10 32.20 28.71 22.10 - 19.94 -
Mono-T5 55.31 42.11 18.23 11.14 21.93 19.21 20.10 - 19.01 -
RankingGPT 61.12 48.10 24.98 14.97 36.39 32.17 21.48 - 18.32 -
RankLlama 54.28 42.03 19.61 11.48 27.16 23.58 22.46 - 18.44 -
RankZephyr 53.15 41.49 19.73 11.90 34.15 30.11 22.37 - 18.59 -
RankVicuna 52.81 40.12 19.05 11.72 33.25 29.87 21.58 - 18.27 -
DPS 66.34 53.28 28.85 19.57 46.13 41.47 28.72 - 29.37 -
Table 2: Experimental results on five benchmarks, where we highlight the best results in bold . We compute the average EM and
F1 on three multi-hop QA datasets, and average F1 on CS and Legal.
Method HotpotQA Musique 2WikiMQA Legal CS
Metric F1 EM F1 EM F1 EM F1 EM F1 EM
-SFT 44.41 34.87 14.26 08.71 27.47 24.78 19.66 - 20.65 -
-MS 65.77 52.70 27.53 18.49 45.83 41.33 22.15 - 23.62 -
-Multihop 47.00 36.53 14.65 08.87 27.04 23.58 28.51 - 29.81 -
-Synthetic 64.21 52.05 28.02 18.67 46.02 40.39 28.20 - 29.21 -
DPS 66.34 53.28 28.85 19.57 46.13 41.47 28.72 - 29.37 -
Table 3: Ablation study on the effects of different training components on DPS performance across multiple datasets.
vs. 24.98). On 2WikiMQA, DPS reaches 46.13 F1, surpass-
ing RankingGPT by 9.74 points.
Beyond standard QA tasks, DPS shows strong generaliza-
tion to domain-specific datasets such as Legal and CS. De-
spite the absence of domain-specific fine-tuning, DPS out-
performs DPA-RAG and RankingGPT by up to 6.3% in F1
on the Legal and CS domains. This demonstrates DPS’s
strong capability in handling long contexts and high-level
queries. Moreover, this also suggests that dynamic passage
selection is more adaptable to unseen tasks and domains
compared to rigid Top-K reranking strategies.
These results confirm that the passage selection approach
generalizes well across both standard and multi-hop ques-
tion answering tasks.
Ablation Study
To investigate the impact of different training data com-
ponents on DPS performance, we conduct ablation exper-
iments by selectively removing parts of the training set.
Specifically, we compare the full DPS model fine-tuned with
supervised data including MS MARCO and multi-hop QAdatasets, with four ablated variants: 1) -SFT : DPS model
without supervised fine-tuning; 2) -MS: DPS trained with-
out MS MARCO data; 3) -Multihop : DPS trained without
multi-hop QA data; and 4) -Synthetic : excludes synthetic
QA data generated for pretraining. This analysis helps quan-
tify the impact of each training component on overall perfor-
mance across multiple QA benchmarks.
Table 3 summarizes the results on multiple experiments.
Removing supervised fine-tuning (-SFT) leads to a substan-
tial performance drop, confirming the importance of super-
vised learning for effective passage selection. Excluding MS
MARCO (-MS) data reduces general retrieval quality, re-
sulting in lower scores across datasets. Similarly, remov-
ing multi-hop data (-Multihop) affects the performance on
multi-hop benchmarks such as HotpotQA and MuSiQue, in-
dicating that exposure to multi-hop reasoning examples is
critical for DPS to capture inter-passage dependencies. Fi-
nally, excluding synthetic data (-Synthetic) also causes mod-
erate degradation, indicating that large-scale weak supervi-
sion from synthetic examples enhances the model’s ability
to handle long contexts and high-level queries, and provides

Method HotpotQA Musique 2WikiMQA Legal CS
Metric F1 EM F1 EM F1 EM F1 EM F1 EM
Qwen2.5-7B 15.53 12.69 01.05 00.37 13.38 13.03 16.38 - 18.36 -
Qwen2.5-72B 23.20 18.47 03.32 01.53 14.42 12.91 14.00 - 14.60 -
Llama3.1-8B 21.09 12.90 05.83 01.00 15.13 11.69 17.91 - 18.98 -
Gemini1.5-flash 29.74 22.75 07.59 03.22 28.11 25.71 16.93 - 17.76 -
GPT-4o 41.72 30.74 14.07 05.97 34.31 28.42 18.27 - 20.51 -
DPS
DPS+Qwen2.5-7B 66.34 53.28 28.85 19.57 46.13 41.47 28.72 - 29.37 -
DPS+Qwen2.5-14B 62.91 46.39 26.62 17.33 43.00 33.29 29.08 - 29.28 -
DPS+Qwen2.5-72B 69.02 54.14 30.03 19.53 47.18 42.03 23.62 - 22.22 -
DPS+Llama3.1-8B 62.80 49.09 25.91 15.96 33.44 24.54 29.68 - 31.62 -
DPS+Genemi1.5-flash 66.92 52.11 31.14 17.16 43.34 37.73 33.13 - 38.93 -
DPS+GPT-4o 68.65 53.07 32.40 21.06 46.23 41.78 30.67 32.27 -
Table 4: Performance of DPS combined with various backbone LLMs with different sizes.
valuable complementary signals for generalization. These
results demonstrate that each training data component con-
tributes to the final performance of DPS, and that a diverse,
comprehensive training set is essential for robust and gener-
alizable passage selection.
Further Experiment
Unlike many reranking methods that rely on large-scale
models or intricate multi-stage pipelines—such as Rank-
ingGPT and RankZephyr, which leverages GPT-3.5 ( 175B
parameters) or GPT-4. DPS functions independently of the
generation model and does not require fine-tuning of the an-
swer generator. Even when paired with smaller generation
models like Qwen2.5-7B, DPS delivers competitive results
compared to these substantially larger systems. This high-
lights DPS’s advantages in efficiency, scalability, and prac-
ticality for real-world applications where model size, infer-
ence cost, and ease of integration are critical.
We further evaluate the scalability and robustness of DPS
combined with various backbone LLMs of differing sizes
and capabilities. As shown in Table 4, DPS consistently
improves performance regardless of the underlying gen-
eration model. With smaller models such as Qwen2.5-7B
and Llama3.1-8B, DPS achieves substantial gains over their
standalone baselines, demonstrating the effectiveness of dy-
namic passage selection across different model scales. No-
tably, DPS paired with stronger models like Qwen2.5-72B
and GPT-4o attains the best overall results, achieving up to
69.02 F1 on HotpotQA and 32.40 F1 on MuSiQue. These
results underscore DPS’s strong adaptability and suggest it
can serve as a plug-and-play module to significantly enhance
retrieval-augmented reasoning performance across a wide
range of LLMs and domains.
Conclusion
We propose DPS, a novel dynamic passage selection frame-
work for information retrieval tasks. DPS explicitly mod-
els inter-document dependencies and adaptively selects aminimal subset of passages required for answer genera-
tion. Unlike traditional reranking approaches that assign
independent scores and rely on a fixed Top- Kstrategy,
DPS reformulates passage selection as a structured predic-
tion task. By leveraging large language models to reason
about combinatorial relevance, DPS provides a principled
and adaptive alternative to rigid ranking pipelines. Exten-
sive experiments across five diverse benchmarks—including
multi-hop and domain-specific datasets—demonstrate that
DPS consistently outperforms strong baselines and recent
rerankers, achieving new state-of-the-art results. Notably,
DPS achieves these gains without requiring generation
model retraining or pipeline modifications, highlighting its
practicality and scalability.
While DPS shows strong performance across settings,
several limitations remain for future research. DPS is limited
by the maximum input token length of the LLM, and when
the candidate passage set is too large, DPS needs to employ
a sliding window strategy for inference. Moreover, he ap-
proach still depends on the initial retrieval step; missing key
candidates can hinder downstream selection. Moreover, cur-
rent selection does not explicitly model reasoning chains or
intermediate steps, which may be necessary for more com-
plex queries. Although DPS does not require generator fine-
tuning, it still relies on supervised data to train the selector,
limiting its applicability in low-resource domains. Inference
cost, though moderate, is higher than pointwise rerankers,
which may impact latency in time-sensitive scenarios.
Despite these limitations, we view DPS as an important
step toward more adaptive, reasoning-aware retrieval in in-
formation retrieval. In future work, we plan to integrate ex-
plicit reasoning techniques to further improve evidence se-
lection. We also aim to explore lightweight or instruction-
tuned versions of DPS for zero-shot generalization, and
unify retrieval and selection under a single training objec-
tive. Beyond QA, we envision DPS benefiting broader ap-
plications including citation generation, open-domain dia-
logue, and multimodal grounding—paving the way for more
reliable and context-sensitive generation systems.

References
Bajaj, P.; Campos, D. F.; Craswell, N.; Deng, L.; Gao, J.;
Liu, X.; Majumder, R.; McNamara, A.; Mitra, B.; Nguyen,
T. M.; Rosenberg, M.; Song, X.; Stoica, A. M.; Tiwary, S.;
and Wang, T. 2016. MS MARCO: A Human Generated
MAchine Reading COmprehension Dataset. arXiv: Com-
putation and Language .
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.;
and Liu, Z. 2024a. Bge m3-embedding: Multi-
lingual, multi-functionality, multi-granularity text embed-
dings through self-knowledge distillation. arXiv preprint
arXiv:2402.03216 .
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and Liu,
Z. 2024b. BGE M3-Embedding: Multi-Lingual, Multi-
Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. In Annual Meeting of the As-
sociation for Computational Linguistics .
Cohere Inc. 2024. Rerank. https://cohere.com/rerank. Ac-
cessed: 2024-11-24.
Dong, G.; Zhu, Y .; Zhang, C.; Wang, Z.; Dou, Z.; and Wen,
J.-R. 2024. Understand What LLM Needs: Dual Preference
Alignment for Retrieval-Augmented Generation. Proceed-
ings of the ACM on Web Conference 2025 .
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody,
A.; Truitt, S.; and Larson, J. 2024. From Local to Global:
A Graph RAG Approach to Query-Focused Summarization.
ArXiv , abs/2404.16130.
Gao, J.; Chen, B.; Zhao, X.; Liu, W.; Li, X.; Wang, Y .; Wang,
W.; Guo, H.; and Tang, R. 2025. LLM4Rerank: LLM-based
Auto-Reranking Framework for Recommendations. In Pro-
ceedings of the ACM on Web Conference 2025 , WWW ’25,
228–239. New York, NY , USA: Association for Computing
Machinery. ISBN 9798400712746.
Hambarde, K. A.; and Proenc ¸a, H. 2023. Information Re-
trieval: Recent Advances and Beyond. IEEE Access , 11:
76581–76604.
Ho, X.; Nguyen, A.-K. D.; Sugawara, S.; and Aizawa,
A. 2020. Constructing a multi-hop qa dataset for com-
prehensive evaluation of reasoning steps. arXiv preprint
arXiv:2011.01060 .
Kang, J.; Li, R.; Liu, Q.; Huang, Z.; Zhang, Z.; Chen, Y .;
Zhu, L.; and Su, Y . 2025. Distribution-Driven Dense Re-
trieval: Modeling Many-to-One Query-Document Relation-
ship. Proceedings of the AAAI Conference on Artificial In-
telligence , 39(11): 11933–11941.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In
Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and
Lin, H., eds., Advances in Neural Information Processing
Systems , volume 33, 9459–9474. Curran Associates, Inc.
Li, X.; Zhu, Y .; Liu, S.; Ju, J.; Qu, Y .; and Cheng, G. 2023.
DyRRen: A Dynamic Retriever-Reranker-Generator Model
for Numerical Reasoning over Tabular and Textual Data.
Proceedings of the AAAI Conference on Artificial Intelli-
gence , 37(11): 13139–13147.Li, Y .; Yang, N.; Wang, L.; Wei, F.; and Li, W. 2024. Learn-
ing to Rank in Generative Retrieval. Proceedings of the
AAAI Conference on Artificial Intelligence , 38(8): 8716–
8723.
Liang, P.; Bommasani, R.; Lee, T.; Tsipras, D.; Soylu, D.;
Yasunaga, M.; Zhang, Y .; Narayanan, D.; Wu, Y .; Kumar,
A.; Newman, B.; Yuan, B.; Yan, B.; Zhang, C.; Cosgrove,
C.; Manning, C. D.; Re, C.; Acosta-Navas, D.; Hudson,
D. A.; Zelikman, E.; Durmus, E.; Ladhak, F.; Rong, F.; Ren,
H.; Yao, H.; WANG, J.; Santhanam, K.; Orr, L.; Zheng, L.;
Yuksekgonul, M.; Suzgun, M.; Kim, N.; Guha, N.; Chat-
terji, N. S.; Khattab, O.; Henderson, P.; Huang, Q.; Chi,
R. A.; Xie, S. M.; Santurkar, S.; Ganguli, S.; Hashimoto, T.;
Icard, T.; Zhang, T.; Chaudhary, V .; Wang, W.; Li, X.; Mai,
Y .; Zhang, Y .; and Koreeda, Y . 2023. Holistic Evaluation
of Language Models. Transactions on Machine Learning
Research . Featured Certification, Expert Certification, Out-
standing Certification.
Lin, J.; Ma, X.; Lin, S.-C.; Yang, J.-H.; Pradeep, R.; and
Nogueira, R. 2021. Pyserini: A Python Toolkit for Re-
producible Information Retrieval Research with Sparse and
Dense Representations. In Proceedings of the 44th Inter-
national ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval , SIGIR ’21, 2356–2362.
New York, NY , USA: Association for Computing Machin-
ery. ISBN 9781450380379.
Liu, J.; Meng, S.; Gao, Y .; Mao, S.; Cai, P.; Yan, G.; Chen,
Y .; Bian, Z.; Shi, B.; and Wang, D. 2025. Aligning vi-
sion to language: Text-free multimodal knowledge graph
construction for enhanced llms reasoning. arXiv preprint
arXiv:2503.12972 .
Ma, G.; Ma, Y .; Wu, X.; Su, Z.; Zhou, M.; and Hu, S. 2025.
Task-level Distributionally Robust Optimization for Large
Language Model-based Dense Retrieval. Proceedings of the
AAAI Conference on Artificial Intelligence , 39(23): 24759–
24767.
Ma, X.; Wang, L.; Yang, N.; Wei, F.; and Lin, J. 2024. Fine-
Tuning LLaMA for Multi-Stage Text Retrieval. In Pro-
ceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval , SI-
GIR ’24, 2421–2425. Association for Computing Machin-
ery. ISBN 9798400704314.
Ma, X.; Zhang, X.; Pradeep, R.; and Lin, J. 2023. Zero-shot
listwise document reranking with a large language model.
arXiv preprint arXiv:2305.02156 .
Meng, C.; Arabzadeh, N.; Askari, A.; Aliannejadi, M.; and
de Rijke, M. 2024. Ranked List Truncation for Large Lan-
guage Model-based Re-Ranking. In Proceedings of the 47th
International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval , SIGIR ’24, 141–151.
New York, NY , USA: Association for Computing Machin-
ery. ISBN 9798400704314.
Nogueira, R.; Jiang, Z.; Pradeep, R.; and Lin, J. 2020a. Doc-
ument Ranking with a Pretrained Sequence-to-Sequence
Model. In Cohn, T.; He, Y .; and Liu, Y ., eds., Findings
of the Association for Computational Linguistics: EMNLP
2020 , 708–718. Online: Association for Computational Lin-
guistics.

Nogueira, R.; Jiang, Z.; Pradeep, R.; and Lin, J. 2020b. Doc-
ument Ranking with a Pretrained Sequence-to-Sequence
Model. In Cohn, T.; He, Y .; and Liu, Y ., eds., Findings
of the Association for Computational Linguistics: EMNLP
2020 , 708–718. Online: Association for Computational Lin-
guistics.
Podolak, J.; Peri ´c, L.; Jani ´cijevi ´c, M.; and Petcu, R.
2025. Beyond Reproducibility: Advancing Zero-shot LLM
Reranking Efficiency with Setwise Insertion. In Proceed-
ings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval , SIGIR
’25, 3205–3213. New York, NY , USA: Association for Com-
puting Machinery. ISBN 9798400715921.
Pradeep, R.; Sharifymoghaddam, S.; and Lin, J. 2023a.
RankVicuna: Zero-Shot Listwise Document Reranking
with Open-Source Large Language Models. ArXiv ,
abs/2309.15088.
Pradeep, R.; Sharifymoghaddam, S.; and Lin, J. J. 2023b.
RankZephyr: Effective and Robust Zero-Shot Listwise
Reranking is a Breeze! ArXiv , abs/2312.02724.
Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Lian, D.; Dou, Z.; and
Huang, T. 2024. MemoRAG: Boosting Long Context Pro-
cessing with Global Memory-Enhanced Retrieval Augmen-
tation. Proceedings of the ACM on Web Conference 2025 .
Qin, Z.; Jagerman, R.; Hui, K.; Zhuang, H.; Wu, J.; Yan,
L.; Shen, J.; Liu, T.; Liu, J.; Metzler, D.; Wang, X.; and
Bendersky, M. 2024. Large Language Models are Effective
Text Rankers with Pairwise Ranking Prompting. In Duh,
K.; Gomez, H.; and Bethard, S., eds., Findings of the Asso-
ciation for Computational Linguistics: NAACL 2024 , 1504–
1518. Mexico City, Mexico: Association for Computational
Linguistics.
Robertson, S. E.; Walker, S.; Jones, S.; Hancock-Beaulieu,
M. M.; Gatford, M.; et al. 1995. Okapi at TREC-3 . British
Library Research and Development Department.
Thonet, T.; Cinar, Y . G.; Gaussier, E.; Li, M.; and Renders,
J.-M. 2022. Listwise Learning to Rank Based on Approxi-
mate Rank Indicators. Proceedings of the AAAI Conference
on Artificial Intelligence , 36(8): 8494–8502.
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
Question Composition. Transactions of the Association for
Computational Linguistics , 10: 539–554.
V oorhees, E. M. 1999. Natural Language Processing and
Information Retrieval. In Pazienza, M. T., ed., Information
Extraction , 32–48. Berlin, Heidelberg: Springer Berlin Hei-
delberg. ISBN 978-3-540-48089-1.
Wang, J.; Huang, J. X.; Tu, X.; Wang, J.; Huang, A. J.;
Laskar, M. T. R.; and Bhuiyan, A. 2024. Utilizing BERT
for Information Retrieval: Survey, Applications, Resources,
and Challenges. ACM Comput. Surv. , 56(7).
Xia, Y .; Zhou, J.; Shi, Z.; Chen, J.; and Huang, H. 2025. Im-
proving Retrieval Augmented Language Model with Self-
Reasoning. Proceedings of the AAAI Conference on Artifi-
cial Intelligence , 39(24): 25534–25542.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y .; Cohen, W. W.;
Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: Adataset for diverse, explainable multi-hop question answer-
ing. arXiv preprint arXiv:1809.09600 .
Yates, A.; Nogueira, R.; and Lin, J. 2021. Pretrained Trans-
formers for Text Ranking: BERT and Beyond. In Pro-
ceedings of the 14th ACM International Conference on Web
Search and Data Mining , WSDM ’21, 1154–1156. New
York, NY , USA: Association for Computing Machinery.
ISBN 9781450382977.
Yu, L.; Zhang, C.; Pei, S.; Sun, G.; and Zhang, X. 2018.
WalkRanker: A Unified Pairwise Ranking Model With Mul-
tiple Relations for Item Recommendation. Proceedings of
the AAAI Conference on Artificial Intelligence , 32(1).
Yu, Y .; Ping, W.; Liu, Z.; Wang, B.; You, J.; Zhang, C.;
Shoeybi, M.; and Catanzaro, B. 2024. RankRAG: Unify-
ing Context Ranking with Retrieval-Augmented Generation
in LLMs. ArXiv , abs/2407.02485.
Zhang, L.; Song, K.; Lee, Y . Q.; Guo, W.; Wang, H.; Li,
Y .; Guo, H.; Liu, Y .; Lian, D.; and Chen, E. 2025a. Killing
Two Birds with One Stone: Unifying Retrieval and Ranking
with a Single Generative Recommendation Model. In Pro-
ceedings of the 48th International ACM SIGIR Conference
on Research and Development in Information Retrieval , SI-
GIR ’25, 2224–2234. New York, NY , USA: Association for
Computing Machinery. ISBN 9798400715921.
Zhang, L.; Zhang, Y .; Long, D.; Xie, P.; Zhang, M.; and
Zhang, M. 2023. RankingGPT: Empowering Large Lan-
guage Models in Text Ranking with Progressive Enhance-
ment. CoRR .
Zhang, Y .; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J.
2025b. Qwen3 Embedding: Advancing Text Embedding and
Reranking Through Foundation Models. arXiv:2506.05176.
Zhuang, H.; Qin, Z.; Jagerman, R.; Hui, K.; Ma, J.; Lu, J.;
Ni, J.; Wang, X.; and Bendersky, M. 2023. RankT5: Fine-
Tuning T5 for Text Ranking with Ranking Losses. In Pro-
ceedings of the 46th International ACM SIGIR Conference
on Research and Development in Information Retrieval , SI-
GIR ’23, 2308–2313. New York, NY , USA: Association for
Computing Machinery. ISBN 9781450394086.
Zhuang, S.; Zhuang, H.; Koopman, B.; and Zuccon, G. 2024.
A Setwise Approach for Effective and Highly Efficient Zero-
shot Ranking with Large Language Models. In Proceed-
ings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval , SIGIR
’24, 38–47. New York, NY , USA: Association for Comput-
ing Machinery. ISBN 9798400704314.
Zuccon, G.; Zhuang, S.; and Ma, X. 2025. R2LLMs: Re-
trieval and Ranking with LLMs. In Proceedings of the 48th
International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval , SIGIR ’25, 4106–4109.
New York, NY , USA: Association for Computing Machin-
ery. ISBN 9798400715921.