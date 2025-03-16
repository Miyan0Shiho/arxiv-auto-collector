# Training Plug-n-Play Knowledge Modules with Deep Context Distillation

**Authors**: Lucas Caccia, Alan Ansell, Edoardo Ponti, Ivan Vulić, Alessandro Sordoni

**Published**: 2025-03-11 01:07:57

**PDF URL**: [http://arxiv.org/pdf/2503.08727v1](http://arxiv.org/pdf/2503.08727v1)

## Abstract
Dynamically integrating new or rapidly evolving information after (Large)
Language Model pre-training remains challenging, particularly in low-data
scenarios or when dealing with private and specialized documents. In-context
learning and retrieval-augmented generation (RAG) face limitations, including
their high inference costs and their inability to capture global document
information. In this paper, we propose a way of modularizing knowledge by
training document-level Knowledge Modules (KMs). KMs are lightweight components
implemented as parameter-efficient LoRA modules, which are trained to store
information about new documents and can be easily plugged into models on
demand. We show that next-token prediction performs poorly as the training
objective for KMs. We instead propose Deep Context Distillation: we learn KMs
parameters such as to simulate hidden states and logits of a teacher that takes
the document in context. Our method outperforms standard next-token prediction
and pre-instruction training techniques, across two datasets. Finally, we
highlight synergies between KMs and retrieval-augmented generation.

## Full Text


<!-- PDF content starts -->

Preprint.
TRAINING PLUG-N-PLAY KNOWLEDGE MODULES
WITH DEEPCONTEXT DISTILLATION
Lucas Caccia∗
Microsoft Research Montr ´ealAlan Ansell∗
University of Cambridge
Edoardo Ponti
University of EdinburghIvan Vuli ´c
University of CambridgeAlessandro Sordoni
Microsoft Research Montr ´eal
ABSTRACT
Dynamically integrating new or rapidly evolving information after (Large) Lan-
guage Model pre-training remains challenging, particularly in low-data scenarios
or when dealing with private and specialized documents. In-context learning and
retrieval-augmented generation (RAG) face limitations, including their high in-
ference costs and their inability to capture global document information. In this
paper, we propose a way of modularizing knowledge by training document-level
Knowledge Modules (KMs). KMs are lightweight components implemented as
parameter-efficient LoRA modules, which are trained to store information about
new documents and can be easily plugged into models on demand. We show
that next-token prediction performs poorly as the training objective for KMs. We
instead propose Deep Context Distillation: we learn KMs parameters such as to
simulate hidden states and logits of a teacher that takes the document in context.
Our method outperforms standard next-token prediction and pre-instruction train-
ing techniques, across two datasets. Finally, we highlight synergies between KMs
and retrieval-augmented generation.
1 I NTRODUCTION
Pre-training large language models (LLMs) on massive corpora has shown to be extremely effective at
capturing a wealth of general-purpose linguistic and factual knowledge in their parameters. Adapting
these models to incorporate new or rapidly evolving information remains challenging, particularly in
scenarios where private or specialized documents must be integrated post-hoc. This line of research
is most compelling when using LLMs in common enterprise scenarios when proprietary documents
often contain the latest instructions, policies, or product details. LLMs must integrate this private data
to support tasks such as question-answering (QA) or customer support; another scenario is supporting
scientific discovery, where LLMs could potentially propose new hypotheses or experiments if they
can ingest cutting-edge scientific publications. In both cases, we need a method that preserves the
model’s broad capabilities while efficiently encoding novel documents in low-data conditions. At the
same time, it is useful for this knowledge to be integrated on demand, in a plug-n-play fashion.
A standard solution to these problems is in-context learning , wherein new, up-to-date information
is provided in the context of the LLM, before the input prompt. Although many efforts have been
devoted to improving long-context models, this approach hits its limits in cases where documents are
extremely long. Retrieval-augmented generation (RAG) partially addresses these limitations by only
selecting the most relevant passages for a given prompt (Lewis et al., 2020). However, this comes at
the cost of: 1) lacking global document information, i.e. only local, or passage-level information is
presented in the context, and 2) increased inference costs (in terms of memory footprint and latency)
due to the context enlargement. Another alternative to capture global information is to continually
pre-train the LLM on new documents (e.g., via parameter-efficient methods like LoRA); however,
although next-token prediction is extremely effective during pre-training, it might not be effective in
low-data scenarios, as recent literature suggests (Jiang et al., 2024).
∗equal contribution.
1arXiv:2503.08727v1  [cs.LG]  11 Mar 2025

Preprint.
In this paper, we tackle the problem of integrating information about documents in a data-efficient
fashion, without knowledge of the downstream task at hand. We aim to train specialized knowledge
modules (KMs) that encode information about the new document into continuous parameters. Encod-
ing documents into continuous parameter is not new and traces back to Doc2Vec (Le & Mikolov,
2014) and more recently (Zhang et al., 2023). Here, we parameterize KMs as LoRA modules;
this allows for dynamically loading new knowledge on demand and aligns with recent approaches
to building more modular LLMs (Ostapenko et al., 2024; Muqeeth et al., 2024). Training KMs
with next-token prediction is challenging due to data scarcity. New documents contain orders of
magnitude fewer tokens compared to pre-training. To address this problem, we enhance the learning
signal through two synergistic techniques: knowledge distillation andsynthetic data generation .
With distillation, we optimize the parameter of each KM to reproduce the “behavior” of the LLM
when it is presented with the new document in context. This can be seen as a generalization of
Context Distillation (Snell et al., 2023). We conduct distillation from both the output probabilities
and the hidden states, similar to (Sanh et al., 2019). We dub this method “Deep Context Distillation”
(DCD). Through synthetic data generation, the model can infer additional facts or relationship entities,
which can be seen as intermediate reasoning steps, akin to Chain-of-Thought prompting (Wei et al.,
2022). One of the remaining questions is which inputs we should choose to elicit the behavior
of the context-conditioned model. We experiment with some variants and find that synthetically
generating summary data from the context-conditioned model to distill its behavior works better than
the alternatives.
Concretely, our contributions are as follows:
•We introduce Knowledge Modules trained with Deep Context Distillation, which effectively
condenses document knowledge into a plug-and-play parameter-efficient adapter.
•We show that DCD pairs well with synthetic data generation, thereby benefiting from
additional compute.
•We evaluate our method on two long-context question answering datasets (QuALITY and
NarrativeQA) and two base models (Phi-3 3B and Llama-3.1 8B) in two settings: open book
and closed book evaluation. In all settings, DCD-based Knowledge Modules outperform all
other methods, also displaying synergies with RAG.
2 K NOWLEDGE MODULES
The goal of knowledge modules is to encode information about a particular document Dinto a small
set of parameters such that they can be plugged on top of a base model on demand at inference time.
2.1 P ARAMETRIZATION
We parameterize KMs as parameter-efficient adapters using LoRA (Hu et al., 2022). For every linear
layer of the base model, LoRA modifies the computation by adding the outer product of two low-rank
learnable matrices AandB. LoRA adapters have been used in modular approaches due to their ease
of batching and merging (Ostapenko et al., 2024; Muqeeth et al., 2024).
2.2 N EXT-TOKEN PREDICTION LOSS
A straightforward way of learning a KM for a new document D={d1, . . . , d N}, where dtis a token,
is to use a language modeling (LM) loss such as next-token prediction. Let’s call θD
KMthe parameters
of the KM for document D, comprising A, B matrices for all linear layers in the model. The LM loss
trains θD
KMto be useful to predict the next token in the document:
LLM=−X
ilogp(dt|d<t;θD
KM). (1)
We will drop the superscript Dfrom the KM parameters for readability. Many variants of this loss
have been used in the past both in the original Doc2Vec (Le & Mikolov, 2014) and more recently as
span prediction (Xiao et al., 2023). While next-token prediction is the de-facto approach to knowledge
extraction when large amount of data is available, it suffers from the perplexity curse (Jiang et al.,
2024), where the LLM quickly overfits and minimizes the perplexity on the next token, but fails to
correctly extract all the relevant knowledge (Berglund et al., 2024).
2

Preprint.
2.3 D EEPCONTEXT DISTILLATION
In this work, we propose a more effective approach to learn KMs, by relying on distillation (Hinton,
2015), where the model’s output is trained to match the output of a teacher network that can have
access to additional information or knowledge. Our idea is based on Context Distillation (Snell et al.,
2023), originally proposed to internalize task instructions or reasoning steps into a LM. We propose
to distill the behavior of a teacher model that has access to document Dinto a student model that
doesn’t have access to Dbut can optimize the parameters of the KM θKM. In this way, we hope that
θKMwill encode the knowledge supporting powerful in-context inferences about D. In particular, we
perform distillation both in the output (probability) space and the hidden space of the teacher LM,
dubbing our loss Deep Context Distillation (DCD). A general form of DCD can be written as follows:
LDCD = min
θKMKL 
p(˜D|D)||p(˜D;θKM)
+X
l1
Zl∥hl
˜D|D−hl
˜D;θKM∥1, (2)
where ˜Ddenotes data over which the distillation loss is computed, p(˜D|D)is the teacher, and
p(˜D;θKM)is the student that doesn’t have access to D.hldenotes the hidden states of the base LM
for layer l, both in the teacher (˜D|D)and the student, and Zl=∥hl
˜D|D∥1a normalization factor that
depends on the L1norm of the target hidden states. The distillation at the output layer provides a
rich training signal as the full teacher distribution is available to the student, while the distillation in
the hidden layer provides more direct credit assignment to every LoRA layer. A similar hidden state
loss has been also used in DistilBERT (Sanh et al., 2019), albeit the authors using cosine similarity
instead of L1. We found our loss to perform well in practice.
A central element left to define to make DCD work in practice is the data ˜Dused to perform the
distillation process. We propose two versions of our context-distillation loss: the first is document
DCD (DDCD) and the second is summary DCD (SDCD).
Document DCD In DDCD, we sample random chunks of Ntokens from the document D, and
split it evenly in two. With CkandCk+1denoting these two contiguous chunks, we obtain
LDCD = min
θKMKL 
p(Ck+1|Ck
||p(Ck+1;θKM)) +X
l1
Zl∥hl
Ck+1|Ck−hl
Ck+1;θKM∥1.(3)
This approach allows us to leverage the available document as-is, without requiring any augmented
data to perform the distillation step.
Summary DCD In SDCD, we instead allow ourselves to generate synthetic data from the doc
ument D, and specifically summaries. To do so, we sample a chunk of Ntokens from document C,
Ckand ask the base LM to create a summary of the specific chunk, Sk. Then, we minimize:
LSDCD = min
θKMKL 
p(Sk|Ck)||p(Sk;θKM)
+X
l1
Zl∥hl
Sk|Ck−hl
Sk;θKM∥1. (4)
In practice, we create 16summary per chunk and use every chunk-summary pair for DCD. The idea
behind generating summaries is that summaries are likely to require a certain amount of inference
about the information contained in the document and therefore this might in turn help encode more
information into the KMs by providing a richer training signal.
2.4 T ASK ADAPTATION WITH KNOWLEDGE EXTRACTORS
The trained KM θKMis task-agnostic, as it is solely trained to reproduce the behavior of the teacher
model when Dis in context. When some supervised task data is available, we might want to train
additional parameters to use the knowledge stored in the KMs to maximize task performance. To
do so, we train a Knowledge Extractor (KE), θKE, a parameter-efficient module (LoRA) that can
combine with the document-specific KMs to maximize task performance. For example, in the context
of Question-Answering studied in this paper, if we have available a dataset of questions, answers and
supporting documents {(qi, ai, Di)}, we can train θKEto minimize the negative log-likelihood of the
answers when it is combined with every document KM trained separately:
LKM+KE= min
θKE,w−X
ilogp(ai|qi; [θDi
KM, θKE]w), (5)
3

Preprint.
where [.]denotes our combination function. To combine θKEandθDiKM, we use a learnable weighted
combination of the two LoRAs applied after the outer product: [θDiKM, θKE]w=wMADiBT
Di+
wEAEBT
E, where wM, wEare the weights given to the KM and KE parameters respectively. We
learn different combination parameters for every layer where LoRA is applied. We will show in the
experiments that, although the KE is trained in tandem with a set of documents during training, it can
generalize well to extracting knowledge from an unseen set of documents and corresponding KMs in
the held-out set.
To also experiment with the synergy between KMs and retrieval-augmented generation approaches,
we also extend Eq. 5, to the case where contextual information about the document is included in the
context. The extension of the loss is straightforward:
LRAG +KM+KE= min
θKE,w−X
ilogp(ai|qi, Pi
1, . . . , Pi
M,[θDi
KM, θKE]w), (6)
where Pi
1, . . . , Pi
Mare document passages retrieved conditioned on query qi.
2.5 T RAINING CONSIDERATIONS
Training knowledge modules on a set of documents D={Di}can be done in an embarassingly
parallel fashion. In contrast, training a KE requires having the set of KMs available for joint multi-
task training. KMs can be seen as akin to a continuous indexing mechanism that aims to compress
information about a document during training time (similar to GraphRAG (Edge et al., 2024)).
Therefore, we operate with the assumption that some of the computation happening at query time can
be moved into a more powerful index mechanism at training time. Training KMs requires gradient
descent. In the future, efficient ways of initialize KMs to decrease computation time during indexing
might be envisioned. SDCD also requires generating summaries from the document. This can be
done efficiently with serving libraries such as VLLM (Kwon et al., 2023).
3 E XPERIMENTS
Setup We experiment with two question-answering datasets QuALITY (Pang et al., 2022) and
NarrativeQA (Ko ˇcisk`y et al., 2018), and two base models, Phi-3 3B (Abdin et al., 2024) and Llama-3.1
8B (Grattafiori et al., 2024). QuALITY is a multi-choice question-answering dataset consisting of 150
training documents and 115 valid documents. The answers of the test documents are private, therefore
we report the results on the dev set. The dataset has a variable number of 4-way multiple-choice
questions for every document. The average length of documents in the dataset is ∼5,000 tokens.
NarrativeQA is a question-answering dataset where documents are longer, ∼60,000 tokens on average.
The dataset has 1,102 training documents, 115 dev documents and 355 test documents. For each
document, the question-answer pairs in the dataset are created based on a human-written summary
of the full document but the evaluation is performed only on the basis of the original documents.
Therefore, this dataset is especially challenging for models with limited context length as the gold
answers might require assembling information across the whole document. For NarrativeQA, we
evaluate performance using multi-reference Rouge-L with the gold answer, while for QuALITY we
use Accuracy (25% being random performance).
We experiment with different setups based on the amount of information available to the models
at test time. In the closed book setting, models do not have access to the document in context
while answering the questions, therefore all the models are tested on the basis of how well they
can incorporate information before solving the specific task at hand. In the open book setting, the
document is provided in the context of the models and therefore context processing can be potentially
conditioned on the question such as in RAG.
Baselines In closed book evaluation, we experiment with different ways of training KMs. The first
is using the standard LM loss KM LM, akin to continual pre-training. Then, we apply our document
DCD loss KM DDCD . Then, we benchmark our KM SDCD which uses generated summaries as
target for deep context distillation. Finally, we experiment with Pre-Instruction Tuning (PIT) (Jiang
et al., 2024), where they propose to concatenate task data (query/answer pairs) before the documents
during continual pre-training; to be fair with our methods that do not use task data during KM
4

Preprint.
Phi-3 (3B) QuALITY NQA
Closed Book
Base Model 26.6 12.1
KMLM 26.4 15.2
KMDDCD 28.7 21.2
KMPIT 29.4 11.5
KMSDCD 32.5 23.9
KMLM+ KE 36.0 20.7
KMDDCD + KE 40.4 28.0
KMSDCD + KE 47.1 32.2
Open Book
ICL 33.1 21.0
RAG 34.7 23.0
RAG + KM SDCD 35.1 23.1
RAG + KE 53.4 36.2
RAG + KM SDCD + KE 55.8 39.1Llama3.1 (8B) QuALITY NQA
Closed Book
Base Model 12.5 20.9
KMLM 26.1 17.9
KMSDCD 37.2 26.6
KMLM+ KE 40.9 22.3
KMSDCD + KE 57.2 32.5
Open Book
ICL 46.0 38.0
RAG 41.0 28.2
RAG + KM SDCD 40.1 27.5
RAG + KE 62.2 36.7
RAG + KM SDCD + KE 64.1 39.7
Table 1: Results for QuALITY and NarrativeQA and on Phi-3 3B (left) and Llama3.1 8B (right).
training. We concatenate our generated summaries before each document and we train on the
resulting concatenation. We denote this variant as KM PIT.
In the open book setting, we use RAG and ICL as baselines: for RAG, we split each document
into passages of 256 tokens and use SFR-embedding-2 (Meng et al., 2024) to embed passages and
questions to perform retrieval of the top-5 relevant passages as measured by the cosine similarity
with each question. Similarly to KM, we assume we know the document the questions relate to and
therefore we only retrieve passages from that document (we don’t search over the set of all documents
in the dataset). We report results of zero-shot RAG (just denoted as RAG) and a fine-tuned version of
RAG (RAG + KE), where a KE module (just a LoRA adapter) is fine-tuned on the task with the RAG
context. For all methods, KEs are always trained solely on the training documents, and never on the
dev/test documents. We experiment with combinations of RAG and KMs, RAG + KM (zero-shot)
and RAG + KE + KM (KE trained to combine RAG and KM information) to analyze the synergy
between RAG and KMs. Technically, ICL for decoder-only models can be considered as a closed
book approach if the KV document cache is stored in memory. However, this comes at an excessive
storage cost (for Llama3.1 8B, for a document of 60k tokens, it’s ∼30Gb). We experiment with KV
compression methods such as the recently proposed L2 compress (Devoto et al., 2024b) to analyze
tradeoffs between performance and storage cost.
KMs and KEs methods use LoRA as parameter-efficient adapter with a rank of 16, LoRA alpha of
16, learning rate of 1e-3, are trained with 1500 steps with batch size of 8 and cosine learning rate
scheduler. We use greedy decoding to sample the responses for NarrativeQA.
Results We report the results for the two base models and the two datasets in Table 1. Results
are consistent and show that SDCD performs best across the board in the closed book setting,
outperforming both LM, PIT and DDCD. In the open book setting, we see that KMs struggle at
zero-shot combination with RAG (RAG + KM lines in both table). For Llama-3.1, there are slight
signs of forgetting (compare RAG vs. RAG + KM), which might be due to the fact that KMs are
never trained to cope with long contexts. However, these are readily solved by training a specialized
KE, which highlights strongly synergistic behavior of RAG and KMs (compare RAG + KE and RAG
+ KM + KE; +3.4% R-L and +3% R-L in NQA and ∼2% Acc. on QuALITY).
In Figure 1, we relate performance on NQA vs. token cost at inference time, as measured as the
number of context tokens the model consumes to produce an answer. We denote by kthe number
of retrieved passages for RAG. We see that KM+KE (without any retrieval) outperform retrieving
RAG+KE with k= 1while only use the question tokens in the prompt (40 tokens in average vs 200
tokens for RAG+KE with k= 1). Scaling the number of retrieved passages kbenefit both RAG +
5

Preprint.
102103104
T oken Cost (# context tokens)25.027.530.032.535.037.540.042.5Rouge-L
RAG+KE
KM+KE
KMRAG+KM+KE
L2 PRESS~3x
k=1k=5k=16
k=1k=8
ICL
Figure 1: Number of tokens in the context for every model (on the x-axis) vs Rouge-L performance
(on the y-axis) on the NarrativeQA dataset. We report both zero-shot (diamond marker) and fine-tuned
models (circle marker, KE). We also benchmark a recent KV-cache compression method based on
the l2 norm of the key vectors (Devoto et al., 2024b).
KM + KE and RAG + KE, while introducing KM retains gains over RAG for similar values of k
and matches performance at a lower cost (42.4 obtained with k= 8vs 42.5 attained with k= 16 )
providing savings of 50% at inference time.
4 R ELATED WORK
In-Context Learning and RAG have been used to incorporate external knowledge at inference
time. While powerful, these techniques can impose high storage costs (e.g., storing large key-value
caches). Efforts to compress KV caches typically use heuristics such as norms of the key-value pairs
(Devoto et al., 2024a) or learned. Recent methods try to build more powerful RAG mechanisms by
scaling both indexing and generation (Yue et al., 2024; Edge et al., 2024). Yue et al. (2024) extend
RAG by retrieving passages, queries, and answers. They further scale inference time by decomposes
queries into sub-queries, incrementally building solutions. GraphRAG Edge et al. (2024) has a
costly indexing step which builds a knowledge graph from chunked documents by extracting (subject,
predicate, object) relations, then clusters and summarizes sub-graphs. This offline graph-based
pre-processing is reminiscent of KMs in reducing repeated computation at inference time. Much like
KMs, GraphRAG help capturing global document context. Data generated by GraphRAG at indexing
time could potentially be used as distillation data in KMs.
Knowledge Distillation typically transfers logits or intermediate representations from a teacher
model to a student (Xu et al., 2024). While effective at compressing large models, these approaches
usually differ from KMs in that they do not explicitly store domain knowledge for repeated use in
specific tasks. Context distillation (Snell et al., 2023) aims to “absorb” a context into a model in a way
that preserves its impact on generation. Early work focused on toxicity and safety contexts (Askell
et al., 2021) and for internalizing reasoning traces (Snell et al., 2023). Recently, Shin et al. (2024)
propose Generative Context Distillation, augmenting the distillation objective with a generative loss
to match model outputs more comprehensively.
Knowledge Injection with Modules Several works have similar goals of KMs. Xiao et al. (2023)
introduce “plug-and-play” document modules where an MLP transforms encoded document tokens
into soft prompts used across attention layers. Zhang et al. (2023) similarly train knowledge modules
in a task-agnostic or task-specific manner, caching the processed document representation for efficient
reuse. Amortization-based approaches train a “hypernetwork” to generate lightweight adaptations
(e.g., low-rank shifts or prefix-tuning parameters) from a given context. Chen et al. (2024) learn to
project context tokens into low-rank parameter shifts in a self-supervised manner, and Tack et al.
(2024) encode documents into prefix-tuning weights using a T5-based hypernetwork. These methods
train the hypernet on multi-task data, so at inference time it can produce task-specific or document-
6

Preprint.
specific modules in a single forward pass. KMs as described in this paper are trained with gradient
descent on single documents independent of any multi-task training. This increases per-document
costs but reduces the risk of domain shift. Future work might be devoted to efficient learning of KMs.
5 C ONCLUSION
In this paper, we proposed an alternative approach to the de-facto next-word prediction loss for
knowledge extraction of documents; we showed that using Deep Context Distillation, with synthetic
data, yields consistent improvements across several question-answering benchmarks. Furthermore,
we showed that our approach pairs nicely with RAG, offering practitioners multiple options to
maximize performance, depending on their inference budget.
REFERENCES
Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen
Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko,
Johan Bjorck, S ´ebastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong
Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai,
Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg,
Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J.
Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi, Xin Jin,
Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi, Dongwoo Kim, Young Jin Kim, Lev
Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui
Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo,
Piyush Madan, Ali Mahmoudzadeh, David Majercak, Matt Mazzola, Caio C ´esar Teodoro Mendes,
Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez-Becker,
Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang Ren, Gustavo de Rosa,
Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael
Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song,
Masahiro Tanaka, Andrea Tupini, Praneetha Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan
Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp Witte, Haiping
Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Jilong Xue, Sonali
Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu, Lu Yuan, Chenruidong
Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and
Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone, 2024.
URLhttps://arxiv.org/abs/2404.14219 .
Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones,
Nicholas Joseph, Benjamin Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny
Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom B. Brown,
Jack Clark, Sam McCandlish, Chris Olah, and Jared Kaplan. A general language assistant as a
laboratory for alignment. CoRR , abs/2112.00861, 2021. URL https://arxiv.org/abs/
2112.00861 .
Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak,
and Owain Evans. The reversal curse: Llms trained on ”a is b” fail to learn ”b is a”, 2024. URL
https://arxiv.org/abs/2309.12288 .
Tong Chen, Hao Fang, Patrick Xia, Xiaodong Liu, Benjamin Van Durme, Luke Zettlemoyer, Jianfeng
Gao, and Hao Cheng. Generative adapter: Contextualizing language models in parameters with a
single forward pass. arXiv preprint arXiv:2411.05877 , 2024.
Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. A simple and effective
L2norm-based strategy for KV cache compression. In Proceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing (EMNLP) , 2024a. URL https://arxiv.
org/abs/2406.11430 .
Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. A simple and effective
l2norm-based strategy for kv cache compression, 2024b. URL https://arxiv.org/abs/
2406.11430 .
7

Preprint.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130 , 2024.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models. arXiv preprint arXiv:2407.21783 , 2024.
Geoffrey Hinton. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 ,
2015.
Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International
Conference on Learning Representations , 2022. URL https://openreview.net/forum?
id=nZeVKeeFYf9 .
Zhengbao Jiang, Zhiqing Sun, Weijia Shi, Pedro Rodriguez, Chunting Zhou, Graham Neubig,
Xi Victoria Lin, Wen-tau Yih, and Srinivasan Iyer. Instruction-tuned language models are better
knowledge learners. arXiv preprint arXiv:2402.12847 , 2024.
Tom´aˇs Ko ˇcisk`y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G ´abor Melis,
and Edward Grefenstette. The narrativeqa reading comprehension challenge. Transactions of the
Association for Computational Linguistics , 6:317–328, 2018.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.
Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model
serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles , 2023.
Quoc V . Le and Tom ´as Mikolov. Distributed representations of sentences and documents. CoRR ,
abs/1405.4053, 2014. URL http://arxiv.org/abs/1405.4053 .
Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian Riedel, and Douwe
Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. CoRR , abs/2005.11401,
2020. URL https://arxiv.org/abs/2005.11401 .
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. Sfr-
embedding-2: Advanced text embedding with multi-stage training, 2024. URL https://
huggingface.co/Salesforce/SFR-Embedding-2_R .
Mohammed Muqeeth, Haokun Liu, Yufan Liu, and Colin Raffel. Learning to route among spe-
cialized experts for zero-shot generalization. In Forty-first International Conference on Ma-
chine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024 . OpenReview.net, 2024. URL
https://openreview.net/forum?id=r0qcGcFL4U .
Oleksiy Ostapenko, Zhan Su, Edoardo Maria Ponti, Laurent Charlin, Nicolas Le Roux, Matheus
Pereira, Lucas Caccia, and Alessandro Sordoni. Towards modular llms by building and reusing a
library of loras. arXiv preprint arXiv:2405.11157 , 2024.
Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi, Nikita Nangia, Jason Phang, Angelica Chen,
Vishakh Padmakumar, Johnny Ma, Jana Thompson, He He, and Samuel Bowman. QuALITY:
Question answering with long input texts, yes! In Marine Carpuat, Marie-Catherine de Marn-
effe, and Ivan Vladimir Meza Ruiz (eds.), Proceedings of the 2022 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technolo-
gies, pp. 5336–5358, Seattle, United States, July 2022. Association for Computational Linguis-
tics. doi: 10.18653/v1/2022.naacl-main.391. URL https://aclanthology.org/2022.
naacl-main.391 .
Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf, and Hugging Face. Distilbert, a
distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 ,
2019.
8

Preprint.
Haebin Shin, Lei Ji, Yeyun Gong, Sungdong Kim, Eunbi Choi, and Minjoon Seo. Generative context
distillation. arXiv preprint arXiv:2411.15927 , 2024.
Charlie Victor Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context, 2023. URL
https://openreview.net/forum?id=am22IukDiKf .
Jihoon Tack, Jaehyung Kim, Eric Mitchell, Jinwoo Shin, Yee Whye Teh, and Jonathan Richard
Schwarz. Online adaptation of language models with a memory of amortized contexts. arXiv
preprint arXiv:2403.04317 , 2024.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in
neural information processing systems , 35:24824–24837, 2022.
Chaojun Xiao, Zhengyan Zhang, Xu Han, Chi-Min Chan, Yankai Lin, Zhiyuan Liu, Xiangyang Li,
Zhonghua Li, Zhao Cao, and Maosong Sun. Plug-and-play document modules for pre-trained
models. arXiv preprint arXiv:2305.17660 , 2023.
Wenda Xu, Rujun Han, Zifeng Wang, Long T Le, Dhruv Madeka, Lei Li, William Yang Wang,
Rishabh Agarwal, Chen-Yu Lee, and Tomas Pfister. Speculative knowledge distillation: Bridging
the teacher-student gap through interleaved sampling. arXiv preprint arXiv:2410.11325 , 2024.
Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong
Wang, Xuanhui Wang, and Michael Bendersky. Inference scaling for long-context retrieval
augmented generation. arXiv preprint arXiv:2410.04343 , 2024.
Zhengyan Zhang, Zhiyuan Zeng, Yankai Lin, Huadong Wang, Deming Ye, Chaojun Xiao, Xu Han,
Zhiyuan Liu, Peng Li, Maosong Sun, et al. Plug-and-play knowledge injection for pre-trained
language models. arXiv preprint arXiv:2305.17691 , 2023.
9