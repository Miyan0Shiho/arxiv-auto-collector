# Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention

**Authors**: Sagie Dekel, Moshe Tennenholtz, Oren Kurland

**Published**: 2026-02-04 16:22:20

**PDF URL**: [https://arxiv.org/pdf/2602.04711v2](https://arxiv.org/pdf/2602.04711v2)

## Abstract
Retrieval Augmented Generation (RAG) is a highly effective paradigm for keeping LLM-based responses up-to-date and reducing the likelihood of hallucinations. Yet, RAG was recently shown to be quite vulnerable to corpus knowledge poisoning: an attacker injects misleading documents to the corpus to steer an LLM's output to an undesired response. We argue that the standard causal attention mechanism in LLMs enables harmful cross-document interactions, specifically in cases of attacks. Accordingly, we introduce a novel defense approach for RAG: Sparse Document Attention RAG (SDAG). This is a block-sparse attention mechanism that disallows cross-attention between retrieved documents. SDAG requires a minimal inference-time change to the attention mask; furthermore, no fine-tuning or additional architectural changes are needed. We present an empirical evaluation of LLM-based question answering (QA) with a variety of attack strategies on RAG. We show that our SDAG method substantially outperforms the standard causal attention mechanism in terms of attack success rate. We further demonstrate the clear merits of integrating SDAG with state-of-the-art RAG defense methods. Specifically, the integration results in performance that is statistically significantly better than the state-of-the-art.

## Full Text


<!-- PDF content starts -->

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse
Attention
Sagie Dekel1Moshe Tennenholtz1Oren Kurland1
Abstract
Retrieval Augmented Generation (RAG) is a
highly effective paradigm for keeping LLM-based
responses up-to-date and reducing the likelihood
of hallucinations. Yet, RAG was recently shown
to be quite vulnerable to corpus knowledge poi-
soning: an attacker injects misleading documents
to the corpus to steer an LLM’s output to an unde-
sired response. We argue that the standard causal
attention mechanism in LLMs enables harmful
cross-document interactions, specifically in cases
of attacks. Accordingly, we introduce a novel
defense approach for RAG: Sparse Document At-
tention RAG (SDAG). This is a block-sparse at-
tention mechanism that disallows cross-attention
between retrieved documents. SDAG requires a
minimal inference-time change to the attention
mask; furthermore, no fine-tuning or additional
architectural changes are needed. We present an
empirical evaluation of LLM-based question an-
swering (QA) with a variety of attack strategies on
RAG. We show that our SDAG method substan-
tially outperforms the standard causal attention
mechanism in terms of attack success rate. We
further demonstrate the clear merits of integrating
SDAG with state-of-the-art RAG defense meth-
ods. Specifically, the integration results in perfor-
mance that is statistically significantly better than
the state-of-the-art.
1. Introduction
Retrieval-Augmented Generation (RAG) has become a
widely adopted approach for ameliorating hallucinations
and providing fresh information in large language models’
(LLMs) responses (Lewis et al., 2020; Borgeaud et al., 2022).
In LLM-based question answering, for example, documents
retrieved from a corpus in response to the question are used
1Faculty of Data and Decision Sciences, Technion - Israel
Institute of Technology, Haifa, Israel. Correspondence to: Sagie
Dekel <sagie.dekel@campus.technion.ac.il>.
Preprint. February 6, 2026.together with the question as a prompt to the LLM, thereby
providing context.
The reliance on retrieval from a corpus introduces a vulner-
ability in RAG-based systems:corpus knowledge poisoning
attack(Xue et al., 2024; Chang et al., 2025). That is, an
attacker injects misleading adversarial documents into the
corpus to steer the system’s output (e.g., answer to a ques-
tion) towards the attacker-desired output. Even a single or a
small number of adversarial documents can have significant
impact on performance (Chang et al., 2025; Xi et al., 2025;
Zou et al., 2025). Accordingly, there is a growing body of
work on defending RAG systems against corpus knowledge
poisoning, and on detecting such attacks (Tan et al., 2025).
Defense methods are largely integrated as an additional com-
ponent in the RAG pipeline or alter the retrieval step (Hong
et al., 2024; Kim et al., 2025; Zhou et al., 2025).
We argue that an overlooked but fundamental aspect of
RAG’s vulnerability is thecausal attentionmechanism used
by decoder-only LLMs. Under causal attention, tokens in
one retrieved document can attend to — and thus be influ-
enced by — tokens from other retrieved documents. When
the retrieved document set contains conflicting information
— e.g., the correct information vs. the incorrect information
injected by an attacker — cross-document attention can fall
short.
Inspired by work on using sparse-attention in RAG to im-
prove efficiency and effectiveness during inference innon-
adversarialsettings (Ratner et al., 2023; Ma et al., 2025;
Zhu et al., 2025b), we introduce a novel defense method
for RAG against corpus poisoning attacks. The method,
termedSparse Document Attention RAG(SDAG), enforces
block-sparse-attention over retrieved documents. Specifi-
cally, tokens in a retrieved document attend only to previous
tokens in the same document; cross-attention between differ-
ent retrieved documents is disabled. The masked attention
is applied at inference time and hence, no fine-tuning or
architectural changes are called for during training.
We present a rigorous empirical evaluation of SDAG for
LLM-based question answering (QA) under corpus poison-
ing attacks. Various generators (LLMs), retrievers, datasets
and attack types are used. While previous work on de-
fending RAG against corpus poisoning focused mostly on
1arXiv:2602.04711v2  [cs.IR]  5 Feb 2026

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
attacks involving multiple adversarial documents (Hong
et al., 2024; Kim et al., 2025), we also consider the case of a
single document attack. Specifically, the attacker uses only
a single adversarial document to steer the system’s output.
This is a practical scenario where defense methods that are
based on the premise of multiple adversarial documents —
e.g., by utilizing their similarities — can fall short as we
show.
The empirical evaluation shows that SDAG substantially
reduces Attack Success Rate (ASR) — the ratio of ques-
tions for which the answer is that of the attacker — and
often improves question answering accuracy with respect to
the standard causal-attention mechanism. Furthermore, in a
single-document attack setting, SDAG consistently outper-
forms the state-of-the-art defense methods. When SDAG
is integrated with the state-of-the-art methods for multiple-
documents attacks, a new state-of-the-art is established.
We present an additional novel analysis based on an em-
bedding space of documents that sheds light on the effec-
tiveness of attacks and defenses. A case in point, we show
that the closer the adversarial documents are to the non-
adversarial in the embedding space, the more effective the
attack is. Furthermore, we show that SDAG “focuses” the
generation on a subset of documents that contain the correct
(non-adversarial) answer.
The code of SDAG’s implementation and evaluation is avail-
able at https://github.com/sagie-dekel/Sparse-Document-
Attention-RAG.
Our contributions can be summarized as follows:
•We point to the standard causal attention mechanism of
LLMs as an important aspect of RAG’s vulnerability
to corpus knowledge poisoning attacks. We present a
novel defense method, SDAG, which is based on block-
sparse-attention during inference time. SDAG does not
call for any training adjustments, and can be easily
integrated, as we show, with other defense methods.
•We empirically show that SDAG is substantially more
effective for coping with adversarial attacks than the
standard causal attention mechanism. SDAG also con-
sistently outperforms state-of-the-art defense methods
in single adversarial document settings. When inte-
grated with state-of-the-art defense methods in multi-
ple adversarial document settings, the resultant perfor-
mance is the new state-of-the-art.
•We present novel types of attacks and analysis based on
embedding spaces of documents. The analysis sheds
light on the effectiveness of attacks and defense meth-
ods.2. Related Work
Corpus Knowledge Poisoning Attacks and Defenses in
RAGThere is a growing body of work on adversarial at-
tack strategies aimed at manipulating the outputs of RAG
systems. A prominent class of attacks is poisoning attacks,
where an adversary injects malicious or misleading content
into the RAG corpus, aiming to steer the generator towards
attacker-chosen outputs (Zhong et al., 2023). Zou et al.
(2025) formulate knowledge poisoning attacks as an opti-
mization problem and propose an effective attack framework
that leverages LLMs to generate adversarial documents. Zhu
et al. (2025a) generate adversarial content guided by internal
neuron attributions of the RAG’s LLMs. This line of work
utilizes multiple adversarial documents in the attack.
Recently, Xi et al. (2025) employed reinforcement learning
to influence RAG system behavior, often using a single
document. Chang et al. (2025) demonstrated a realistic
threat model using only a single poisoning document to
steer RAG outputs.
To address the effects of corpus poisoning attacks, several
defense methods were proposed. These include: cluster-
ing the retrieved documents (Kim et al., 2025; Zhou et al.,
2025), fine-tuning the LLM (Hong et al., 2024), using a
graph-based approach (Shen et al., 2025), and applying
decoding-based methods (Xiang et al., 2024). Hong et al.
(2024) presented a highly effective defense method by fine-
tune a discriminator or prompt an LLM to leverage its dis-
criminative capabilities to identify adversarial documents.
Kim et al. (2025) presented a state-of-the-art defense ap-
proach: lightweight machine learning techniques that detect
and filter adversarial content after the retrieval phase and
before the generation phase.
In contrast to previously proposed defense methods, our
approach does not call for re-training, finetuning or addition
of components or steps to the RAG pipeline, and it does
not rely on filtering retrieved documents. Furthermore, our
method, which outperforms the state-of-the-art in a single-
document attack setting, can be integrated with current pre-
generation state-of-the-art defense methods to yield a new
state-of-the-art in multiple-document settings as we show.
Attention Mechanism in RAGIn standard RAG pipelines,
retrieved documents are concatenated and incorporated into
the generator prompt (Lewis et al., 2020; Izacard & Grave,
2021). This design incurs notable limitations due to the
finite context window of LLMs and the quadratic computa-
tional complexity of attention with respect to input length.
To address these challenges, a suite of sparse-attention ap-
proaches to utilizing retrieved documents were proposed.
The goal was to improve efficiency and effectiveness. For
example, Ratner et al. (2023), Ma et al. (2025) and Xiao et al.
(2025) showed that removing cross-attention between re-
2

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
trieved documents can improve effectiveness and efficiency
in non-adversarial settings. Zhu et al. (2025b) achieved im-
proved efficiency and effectiveness by training an LLM to
encode retrieved documents in parallel. The final response
was generated based on a subset of documents presumed rel-
evant by the LLM. While sparse attention was used in RAG
to improve efficiency and effectiveness in non-adversarial
settings, we are the first — to the best of our knowledge —
to apply it as a defense mechanism for corpus knowledge
positioning attacks.
3. Preliminaries
In this section we describe decoder-only LLM-based RAG
and the corpus knowledge poisoning attacks we consider.
3.1. Definitions and Notations
Given a question q, a retriever Rretrieves from a corpus Ca
setDofkdocuments, formally written as D:=R(q, C) =
{d1, d2, . . . , d k}. A RAG system then uses the question
qand the retrieved set Das an input to a generator Gto
produce an answer1y=G(p, q, D) , where pdenotes the
task instructions and generator’s additional context (e.g.,
few-shot in-context examples).
After tokenization of the input sequence, each retrieved
document di(∈D ) corresponds to a sequence of tokens,
referred to as ablockB i. BTand B Cdenote the block of
the task instructions and the generator’s additional context,
respectively. To maintain brevity, B Calso includes the
question’s tokens.
3.2. Threat Model: Corpus Knowledge Poisoning
Suppose an attacker aims to influence the behavior of a RAG
system. Specifically, the attacker’s goal is to steer the output
of the generator G, using adversarial documents, towards
an attacker-chosen target answer. Anadversarial document
is defined as any document containing deliberately crafted
misinformation designed to bias the RAG system towards
producing the attacker’s target answer (Kim et al., 2025)2.
Non-adversarial documents will be referred to asbenign
documents.
Inspired by prior work, we consider two settings:in-corpus
(Kim et al., 2025) andin-set(Xiang et al., 2024). In the
in-corpus setting, for each question qiand corresponding
target answer ˜ri, we assume that the attacker can inject
Nadversarial documents, denoted ˜d1,˜d2, . . . , ˜dN, into the
1RAG variants in which retrieved documents are incorporated
after an initial generation phase, e.g., Gao et al. (2023), are outside
the scope of this work.
2We do not consider black hat attacks whose primary objective
is to damage or disable the system.corpus C. After corpus poisoning, the retriever may re-
trieve adversarial documents as part of the retrieved set,
thereby influencing the generation outcome. Formally, the
retrieval and generation process under attack is given by
˜D:=R(q, C∪ { ˜di}N
i=1)→˜y=G(p, q, ˜D). Note that in
the in-corpus case, adversarial documents may not be in the
retrieved set D. Thus, in this setting, the attacker has two
goals: to have the adversarial documents highly ranked and
to steer the system’s response. Consequently, we also con-
sider the in-set setting which allows to focus on the second
goal. More specifically, assuming that the adversarial docu-
ments are in the retrieved set D, we measure the ability of
the attacker to steer the response. Considering both settings
enables a broad analysis of model behavior.
We define three attack strategies:Random,Near, andFar.
Each strategy specifies how the attacker selects (or gen-
erates) the Nadversarial documents. Under theRandom
strategy, the attacker uniformly samples Ndocuments from
apoolof candidate adversarial documents, which was cre-
ated using some approach (e.g., by prompting an LLM).
Under theNear(Far) strategy, the attacker selects the N
documents from the pool that are closest (farthest) to the
centroid of the benign retrieved documents in an embedding
space. Each strategy corresponds to a plausible real-world
approach an attacker might adopt when performing an attack.
The Near and Far strategies can be applied, for example, by
observing attribution outcomes in a RAG system (Li et al.,
2023). That is, it is often the case that the documents shown
by a RAG system to provide support for the generated re-
sponse are those in D. To the best of our knowledge, the
Near and Far strategies are novel to our study. We show
in Section 6.2 that the type of attack has an impact on the
attack effectiveness.
3.3. Decoder-only LLMs and Causal Attention
Decoder-only LLMs operate in an auto-regressive manner
using a causal attention mechanism, whereby each token is
permitted to attend only to preceding tokens in the sequence;
a.k.a., causal attention (Brown et al., 2020). As a result, the
hidden representation of a given token is influenced exclu-
sively by the representations of tokens that appear earlier in
the input (and itself). In recent years, state-of-the-art LLMs
are mostly based on decoder-only architectures with very
large model sizes (Minaee et al., 2025). This evolution has
enabled such models to process and reason over growing
amounts of textual context provided in the prompt. Conse-
quently, RAG systems that rely on concatenating the raw
text of retrieved documents into the generator’s input have
become widely used (Lewis et al., 2020; Izacard & Grave,
2021; Jiang et al., 2023b; Yoran et al., 2024). We refer to
this RAG architecture asCausal Attention RAG, orCARG
in short.
3

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
4. RAG Using Sparse Document Attention
We now turn to present our sparse-attention model for RAG.
The causal attention practice in LLMs is reasonable in set-
tings where there is contextual continuity across the token
sequence, such as natural language generation or text sum-
marization tasks. In the RAG framework, however, we
argue that such continuity does not necessarily hold across
documents. Retrieved documents are not guaranteed to be
contextually related, nor even relevant to the input question
(Cuconasu et al., 2024). The mismatch between the causal
attention practice and the RAG framework becomes partic-
ularly pronounced when aknowledge conflictarises in the
retrieved set; that is, when two or more retrieved documents
contain conflicting information (Cattan et al., 2025). Such
conflicts are especially common when the corpus is subject
to knowledge poisoning attacks. Under these conditions, the
representations of benign documents may be influenced by
adversarial documents through the attention mechanism, or
vice versa. Moreover, even in the absence of explicit knowl-
edge conflict, it is unclear whether enabling adversarial and
benign documents to influence one another through atten-
tion is warranted. Consequently, we question the suitability
of causal attention when the corpus is subject to knowledge
poisoning attacks.
The Model.We presentSparse Document Attention RAG
(SDAG) as a defense mechanism for RAG under corpus
knowledge poisoning attacks. The model is inspired by Rat-
ner et al. (2023), Ma et al. (2025), and Zhu et al. (2025b),
who use sparse attention to improve theeffectivenessand
efficiencyin non-adversarial settings. In contrast, we study
the merits of using sparse attention to address corpus poi-
soning attacks. Specifically, we propose a sparse document
attention approach where cross-attention between retrieved
documents is explicitly disallowed; that is, tokens from dif-
ferent documents are masked from one another in the atten-
tion mechanism. Concretely, for any two different retrieved
documents dianddj, every token in block B iis prevented
from attending to any token in block B j. Tokens in the
task-instruction block B T, the generator’s context block B C,
and generated tokens retain the standard causal attention, in
which a token attends to all previous tokens. Formally, we
define an attention mask matrix A(shared across all layers
and attention heads), for every pair of tokensr, c, as:
Ar,c=

1,(c∈B T∨r∈B C)∧r≥c
1,∃i∈ {1, . . . , k}, r, c∈B i∧r≥c
0,otherwise;
Ar,c= 1indicates that the token at position ris permitted
to attend to the token at position c, andAr,c= 0otherwise;
kis the number of retrieved documents in the set ˜D. We
provide a visual illustration of the resulting attention pattern
in Appendix A.SDAG can be used with any decoder-only LLM with mini-
mal modifications. Specifically, the only required change is
to the attention mask at inference time; no alterations to the
RAG pipeline or the underlying LLM architecture during
training time are necessary, and no additional fine-tuning
is required. As a result, SDAG can be easily used with any
pre-generation defense approach, as we show in Section 6.1.
5. Experimental Setting
We next describe the settings used to evaluate SDAG.
5.1. RAG Components
We consider multiple retriever–generator combinations. We
use three generators (LLMs): Llama-8B-Instruct (Grattafiori
et al., 2024), Qwen-7B-Instruct (Yang et al., 2024), and
Mistral-7B-Instruct (Jiang et al., 2023a). Notably, imple-
menting SDAG requires control over the attention mask,
which restricts our evaluation to open-source models. In
line with prior work (Zou et al., 2025), we set the temper-
ature of the generator to 0.1. In Appendix B, we study the
effect of temperature, model size, and reasoning capabilities
of the generator; the results and performance patterns are in
line with those we report in Section 6.
For retrieval, we use both dense and sparse rankers. For
dense retrieval, we employ E5-large-v2 (Wang et al., 2024)
and Contriever (Izacard et al., 2022). Specifically, the co-
sine between the embedding vectors of the question and
a document is used as the retrieval score. Okapi-BM25
(Robertson & Zaragoza, 2009) serves as the sparse-retrieval
method implemented via the Lucene-based Pyserini toolkit
(Lin et al., 2021). We set the number of retrieved documents,
k, to 5 and 10, following common practice (Xu et al., 2024;
Yu et al., 2024). The retrieved documents are then inserted
to the generator’s prompt, details of which are provided in
Appendix C.
5.2. Datasets
We evaluate our approach using three commonly used QA
benchmarks of varying complexity: HotpotQA (Yang et al.,
2018), which consist of multi-hop reasoning questions,
TriviaQA (Joshi et al., 2017) and Natural Questions (NQ)
(Kwiatkowski et al., 2019); the latter two include single-hop
questions. For each dataset, we randomly sample31,000
questions from the validation set to serve as our test set.
Following common practice in work on RAG (Chen et al.,
2017; Lee et al., 2019; Karpukhin et al., 2020; Cuconasu
et al., 2024), we use the English Wikipedia dump from De-
cember 20, 2018 as the corpus. As in prior work (Wang
et al., 2019; Karpukhin et al., 2020), each Wikipedia article
3Henceforth, we set a fixed random seed of 42 to ensure repro-
ducibility.
4

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
is segmented into non-overlapping passages of 100 words,
resulting in a total of 21,015,324 passages.
5.3. Baselines
We use three methods for reference comparison: (i) Causal
Attention RAG (CARG), as described in Section 3.3, which
allows us to analyze how the transition from the stan-
dard causal attention to sparse-document attention affects
RAG effectiveness under corpus poisoning attacks; (ii)
RAGDefender (Kim et al., 2025), a state-of-the-art method
which detects and filters adversarial documents at the post-
retrieval (and pre-generation) stage, prior to generation; and
(iii) a highly effective GPT-based approach, named Dis-
cern&Answer (Hong et al., 2024), which leverages the dis-
criminative capabilities of LLMs to identify and prioritize
reliable evidence among retrieved documents. We use the
GPT-based approach since we focus on decoder-only LLM-
based RAG. For the latter two, we follow the methodology
and hyperparameters reported in the original papers.
5.4. Attack Implementation
We generate adversarial documents using the PoisonedRAG
framework (Zou et al., 2025), a common practice in work on
RAG defense against adversarial attacks (Xiang et al., 2024;
Kim et al., 2025; Si et al., 2025). We employ GPT-4o (Ope-
nAI et al., 2024) to generate both the attacker’s target answer
and the corresponding adversarial documents that are meant
to steer the generator towards producing this answer. To
allow for diverse and meaningful attack strategies, we gen-
erate a pool of five distinct adversarial documents for each
target answer by setting the LLM sampling temperature to 1.
Unless otherwise stated, we select documents from the pool
using the Random attack strategy described in Section 3.2.
For the Near and Far strategies implementations, we use the
embeddings obtained by the retriever for dense retrieval, and
the embeddings obtained by BERT-base-uncased (Devlin
et al., 2019) for sparse retrieval.
Our main evaluation is performed using the in-set setting
to avoid retrieved sets with only benign documents (i.e.,
to isolate the effect of SDAG under knowledge poisoning
attacks). We provide results for the in-corpus setting in
Appendix D; the findings are in line with those we report in
Section 6 for the in-set setting.
We focus on a setting that is largely unexplored in prior
studies of RAG defenses: the attacker manages to have a
single document injected to the retrieved set (or to the cor-
pus in the in-corpus setting). We then evaluate SDAG in a
multiple adversarial documents setting, where the attacker
manages to have either two or three adversarial documents
injected to the retrieved set. Notably, the single adversarial
document setting can be more challenging than a setting
with multiple adversarial documents. As a case in point,consider approaches that rely on similarities between poten-
tially adversarial documents, e.g., RAGDefender (Kim et al.,
2025). Such methods can fall short in the single adversarial
document setting as we show in Section 6.1.
For the single-document in-set setting, we place the adver-
sarial document closest to the question (i.e., near the end of
the prompt). We study the effect of placing the adversarial
document at a random location in the prompt and at the
beginning of the prompt in Appendix E; the results are in
line with those we present in Section 6.1, where we position
it closest to the question. For the multiple-documents case,
we place the adversarial documents at random positions in
the prompt to avoid grouping them next to the question.
5.5. Evaluation Measures
Following prior work on RAG defense methods (Kim et al.,
2025; Si et al., 2025), we utilize two measures to evalu-
ate model performance:Accuracy(ACC) andAttack Suc-
cess Rate(ASR). Accuracy is the proportion of model an-
swers that contain the correct answer that is provided in
the dataset4. ASR is the proportion of model answers that
contain the attacker’s target answer.
As in prior work (Chen et al., 2017; Lee et al., 2019; Yu
et al., 2023), we determine whether a RAG output contains
an answer usingexact matchbetween strings after minor
normalization. Statistical significance of performance dif-
ferences is determined using a paired t-test withp≤0.05.
6. Results
We begin by presenting the research questions (RQs) that
we focus on in the results analysis:
•RQ1:Does SDAG outperform CARG under corpus
knowledge poisoning attacks? i.e., is block-based at-
tention (at the document level) more effective than the
standard (causal) attention in an adversarial setting?
•RQ2:Does SDAG outperform state-of-the-art RAG
defenses under corpus knowledge poisoning attacks?
•RQ3:How does the spatial positioning of an adversar-
ial document in the embedding space influences RAG
performance, and how do these effects differ between
SDAG and CARG?
6.1. RQ1 and RQ2: Effectiveness of SDAG
We first compare SDAG with CARG to study the effective-
ness of our block (document)-based attention in adversarial
4If multiple correct answers exist in the dataset, inclusion of at
least one correct answer is counted.
5

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 1.Performance comparison of SDAG with CARG. Higher
ACC and lower ASR values indicate better performance. Boldface
marks the best result for a generator, retriever, k, dataset, and
evaluation measure; ’∗’ marks a statistically significant difference
between SDAG and CARG for a generator, retriever, k, dataset,
and evaluation measure. LLM–Retr. denotes generator–retriever,
BM. denotes BM25, and Co. denotes Contriever.
HotpotQA TriviaQA NQ
LLM-Ret.k SDAG CARG SDAG CARG SDAG CARG
ACC ASR ACC ASR ACC ASR ACC ASR ACC ASR ACC ASR
Llama-E550.23∗0.41∗0.15 0.680.69∗0.20∗0.54 0.470.37 0.17∗0.33 0.41
100.26∗0.28∗0.18 0.660.74∗0.11∗0.57 0.410.37 0.10∗0.35 0.36
Llama-Co.50.13∗0.54∗0.07 0.780.49∗0.36∗0.34 0.640.23 0.28∗0.20 0.52
100.15∗0.33∗0.09 0.760.60∗0.21∗0.43 0.550.27 0.17∗0.24 0.48
Llama-BM.50.20∗0.40∗0.12 0.720.58∗0.26∗0.45 0.540.22 0.26∗0.19 0.53
100.21∗0.28∗0.13 0.690.65∗0.15∗0.50 0.480.26 0.16∗0.25 0.47
Qwen-E550.20∗0.52∗0.15 0.740.61∗0.32∗0.51 0.530.36∗0.25∗0.29 0.50
100.22∗0.39∗0.16 0.710.68∗0.21∗0.57 0.460.33 0.16∗0.330.43
Qwen-Co.50.10∗0.67∗0.07 0.810.45∗0.48∗0.35 0.700.21∗0.40∗0.15 0.61
100.13∗0.55∗0.09 0.800.54∗0.34∗0.41 0.630.25∗0.30∗0.19 0.57
Qwen-BM.50.18∗0.51∗0.11 0.770.56∗0.34∗0.43 0.610.17 0.46∗0.14 0.60
100.20∗0.40∗0.12 0.760.60∗0.25∗0.46 0.570.23∗0.36∗0.17 0.56
Mistral-E550.22∗0.52∗0.14 0.780.63∗0.35∗0.58 0.630.40∗0.26∗0.32 0.55
100.27∗0.37∗0.13 0.740.73∗0.22∗0.62 0.510.42∗0.19∗0.34 0.46
Mistral-Co.50.13∗0.65∗0.06 0.840.47∗0.53∗0.40 0.750.27∗0.39∗0.19 0.62
100.18∗0.47∗0.08 0.810.59∗0.35∗0.46 0.660.33∗0.26∗0.23 0.56
Mistral-BM.50.17∗0.54∗0.10 0.810.57∗0.37∗0.48 0.700.27∗0.33∗0.19 0.61
100.21∗0.40∗0.11 0.780.65∗0.25∗0.50 0.640.31∗0.23∗0.21 0.58
settings (RQ1). We then compare SDAG with state-of-the-
art defense methods (RQ2).
The Effect of SDAG on RAG Performance (RQ1).We
first compare in Table 1 the performance of SDAG and
CARG in the single adversarial document setting. We fur-
ther evaluate SDAG in the multiple-document setting below.
We see in Table 1 that in most relevant comparisons SDAG
statistically significantly outperforms CARG in terms of
both accuracy and ASR; many of the improvements are quite
substantial. These results attest to the fact that SDAG is
effective for both single-hop (the NQ and TriviaQA datasets)
and multi-hop (the Hotpot dataset) questions.
We can also see in Table 1, as expected, that reducing the
adversarial documents proportion in the retrieved set (by in-
creasing kfrom 5 to 10) improves the performance of SDAG
and CARG. However, SDAG attains larger improvements
than those attained by CARG with k= 10 in comparison to
k= 5 . In addition, Table 1 shows that in contrast to CARG,
substantial ASR improvements of SDAG using k= 10 in
comparison to k= 5 do not necessarily translate to corre-
sponding accuracy improvements. Furthermore, substantial
ASR improvements of SDAG over CARG do not necessarily
translate to corresponding accuracy improvements. Taken
together, these findings indicate that SDAG can sometimes
cause a shift in RAG output from the attacker’s target an-Table 2.Performance of SDAG and CARG for different attack
strategies on the NQ dataset, using E5 and k= 5 . Boldface
marks the best result for a generator, attack strategy, and evaluation
measure; ’∗’ marks a statistically significant difference between
SDAG and CARG for a generator, attack strategy, and evaluation
measure.
Generator MethodRandom Near Far
ACC ASR ACC ASR ACC ASR
LlamaSDAG 0.37 0.17∗0.35 0.23∗0.38 0.15∗
CARG 0.33 0.41 0.32 0.42 0.34 0.39
QwenSDAG 0.36 0.25∗0.34∗0.31∗0.36 0.22∗
CARG 0.29 0.50 0.27 0.51 0.32 0.46
MistralSDAG 0.40∗0.26∗0.37∗0.33∗0.41∗0.23∗
CARG 0.32 0.55 0.30 0.56 0.33 0.53
swer to incorrect answers. Still, as Table 1 shows, and as
discussed above, SDAG consistently and statistically signif-
icantly outperforms CARG for accuracy and ASR.
We further observe in Table 1 that the magnitude of improve-
ments of SDAG over CARG can vary across datasets and
generators. One possible explanation is that differences in
model architecture and training procedures lead to varying
sensitivity to the sparse attention employed by SDAG. In
particular, the inductive biases induced by each generator’s
pre-training and post-training may interact differently with
varying attention mechanisms and question distributions.
We can also see in Table 1 that E5 is more effective than
Contriever and BM25 as a retriever for both CARG and
SDAG. This finding is in line with those in past reports on
RAG for QA with CARG (Jin et al., 2025). Hence, in what
follows, we use E5 as the retriever.
Attack Strategies (RQ1).We next evaluate SDAG with the
different attack strategies defined in Section 3.2: Random,
Near, and Far. Recall that up to this point we used the
Random attack strategy. We focus on the NQ dataset. We
provide results using Contriever and BM25 in Appendix F;
the findings are consistent with those shown below for E5.
As shown in Table 2, SDAG consistently attains higher
accuracy and statistically significantly lower ASR than that
of CARG for all attack strategies. For instance, under the
Far attack strategy, SDAG reduces ASR by more than half
compared to CARG.
Table 2 also shows that the ASR performance of both SDAG
and CARG is consistently better for the Far strategy than
for the Random and Near strategies. This indicates that
adversarial documents that are far from the centroid of the
benign documents in the retrieved set constitute less effec-
tive attacks than adversarial documents close to the centroid.
We revisit this observation in Section 6.2.
Comparison of SDAG with Defense Baselines (RQ2).Ta-
ble 3 presents the results of SDAG, the baselines (CARG,
6

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Discern&answer and RAGDefender), and integration of
SDAG with the baselines in the single and the multiple
adversarial-document settings. Recall that SDAG can be
integrated with any pre-generation defense as it requires a
simple change in the generator’s attention mask. We pro-
vide additional results for the multiple-document setting in
Appendix F; which are consistent with those shown below.
We see in Table 3 that in most relevant comparisons in the
single-document setting SDAG statistically significantly out-
performs the baselines in terms of both accuracy and ASR.
In the single case where SDAG does not post lower ASR
than that of the baselines, its ASR is statistically signifi-
cantly indistinguishable. Moreover, in all cases, the inte-
gration of SDAG with a baseline statistically significantly
outperforms the baseline in terms of ASR and, in the vast
majority of cases, achieves higher accuracy.
We can see in Table 3 that in all cases in the multiple-
document setting, SDAG-RAGDefender statistically sig-
nificantly outperforms CARG and Discern&Answer in
both accuracy and ASR. In the majority of relevant
comparisons for the multiple document attack (2 or 3),
RAGDefender posts the best accuracy. However, its ac-
curacy is statistically significantly distinguishable from
that of SDAG-RAGDefender. Furthermore, in most cases,
SDAG-RAGDefender statistically significantly outperforms
RAGDefender in terms of ASR. In all cases SDAG and
SDAG-Discern&Answer statistically significantly outper-
form CARG and Discern&Answer, respectively, in terms
of ASR and achieve higher or on par accuracy. Notably,
RAGDefender, which relies on similarities between doc-
uments, shows substantial performance differences when
moving from single-document to multi-document settings.
All in all, the results presented above attest to the clear
merits of SDAG: it consistently outperforms the state-of-
the-art in terms of accuracy and ASR in the single-document
setting, and when integrated with RAGDefender, it becomes
a new state-of-the-art in the multiple-document setting.
6.2.RQ3: Spatial Positioning of Adversarial Documents
We turn to analyze how the spatial positioning of adversarial
documents affects the performance of RAG.
Spatial Positioning.In Section 6.1 we showed that attacks
based on adversarial documents distant from the benign
ones were less effective than those using adversarial docu-
ments that were close to the benign documents. To further
analyze the connection between an attack’s effectiveness
and the spatial positioning of an adversarial document in the
single-documents setting, we stratify the questions in the
dataset into two sets: (i)Distant Set(DS), which consists
of questions for which the distance between the adversar-
ial document and the centroid of the benign-document setTable 3.Performance comparison of SDAG-based methods with
CARG, Discern&Answer (D&A), and RAGDefender (RAGD) in
single and multiple adversarial document attack settings, using
Llama and E5. Boldface marks the best result for a #adv. docs,
k, dataset, and evaluation measure; ’ c’, ’d’, and ’ r’ mark a sta-
tistically significant difference between a SDAG-based method
and CARG, D&A, and RAGD, respectively, for a #adv. docs, k,
dataset, and evaluation measure.
#adv.
docskDefense
MethodHotpotQA TriviaQA NQ
ACC ASR ACC ASR ACC ASR
15CARG 0.15 0.68 0.54 0.47 0.33 0.41
D&A 0.16 0.64 0.59 0.40 0.34 0.38
RAGD 0.10 0.70 0.49 0.43 0.28 0.33
SDAG 0.23cd
r0.41cd
r0.69cd
r0.20cd
r 0.37r0.17cd
r
SDAG-D&A 0.23cd
r0.38cd
r0.71cd
r0.17cd
r 0.37r0.16cd
r
SDAG-RAGD 0.15d0.50cd
r 0.59c
r0.25cd
r 0.31 0.17cd
r
10CARG 0.18 0.66 0.57 0.41 0.35 0.36
D&A 0.18 0.62 0.62 0.35 0.35 0.32
RAGD 0.19 0.25 0.68 0.22 0.35 0.18
SDAG 0.26cd
r0.28cd0.74cd
r0.11cd
r 0.370.10cd
r
SDAG-D&A 0.26cd
r0.25cd0.75cd
r0.10cd
r 0.36 0.10cd
r
SDAG-RAGD 0.190.14cd
r0.69cd0.11cd
r 0.310.06cd
r
25CARG 0.11 0.77 0.43 0.62 0.25 0.52
D&A 0.12 0.74 0.51 0.52 0.26 0.47
RAGD 0.210.320.640.250.350.19
SDAG 0.13 0.63cd0.45 0.46cd0.23 0.39cd
SDAG-D&A 0.14c0.60cd0.54c0.36cd0.28 0.34cd
SDAG-RAGD 0.18cd0.28cd0.60cd0.23cd0.31cd0.16cd
10CARG 0.11 0.77 0.47 0.55 0.27 0.47
D&A 0.12 0.73 0.56 0.47 0.32 0.42
RAGD 0.210.28 0.65 0.240.350.19
SDAG 0.17cd0.50cd0.59 0.29cd0.28 0.24cd
SDAG-D&A 0.18cd0.45cd0.63cd0.28cd0.29 0.23cd
SDAG-RAGD 0.18cd0.19cd
r0.66cd0.14cd
r 0.310.12cd
r
310CARG 0.07 0.82 0.37 0.66 0.20 0.55
D&A 0.10 0.77 0.47 0.55 0.25 0.51
RAGD 0.210.380.670.260.380.20
SDAG 0.10 0.67cd0.41 0.49cd0.20 0.43cd
SDAG-D&A 0.11 0.63cd0.49c0.40cd0.23 0.37cd
SDAG-RAGD 0.21cd0.26cd
r0.65cd0.18cd
r0.34cd0.14cd
r
exceeds the benign set diameter5; and (ii)Near Set(NS),
which consists of questions for which this distance is less
than or equal to the diameter. Intuitively, when the dis-
tance exceeds the set diameter, the adversarial document
is geometrically separated from the benign-document set.
Here we measure distances in the embedding space induced
by the generator (i.e., we use the generator’s embeddings),
as we are interested in studying the effect of adversarial
document positioning on generation6. In Appendix G we
also report results based on the retriever’s embeddings; the
results and findings are consistent with those shown below
for the generator’s embeddings.
As in the attack strategies analysis, we focus on the NQ
dataset. We use Llama as the generator. In Appendix G
5The diameter is defined as the maximum distance between
any pair of items in the set.
6A document’s embedding is obtained by applying mean-
pooling over the generator’s token embeddings of a document.
7

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 4.Performance of SDAG and CARG on the NQ dataset,
using Llama and k= 5 , when stratifying questions into two sets:
Distant Set (DS) and Near Set (NS). Boldface marks the best
result for a retriever, question sets, and evaluation measure; ’∗’
marks a statistically significant difference between DS and NS
for a retriever and evaluation measure, as determined using an
unpaired t-test between the question sets.
RetrieverQuestion
SetAccuracy ASR
SDAG CARG SDAG CARG
E5DS 0.52∗0.40 0.08∗0.34
NS 0.35 0.33 0.16 0.39
ContrieverDS 0.31∗0.29∗0.21∗0.47
NS 0.21 0.19 0.31 0.50
BM25DS 0.34∗0.32∗0.190.53
NS 0.20 0.18 0.260.51
we report results for Qwen and Mistral which are in line
with those we report here for Llama. We use the Far attack
strategy, as other attack strategies predominantly induce
adversarial documents which are geometrically near the
benign documents, resulting in insufficient data for analysis.
Table 4 reports the performance of SDAG and CARG for
the two question sets (DS and NS). In the vast majority of
cases, SDAG achieves statistically significantly improved
performance on DS compared to NS, while CARG achieves
an improved accuracy and comparable ASR. Recall that we
observed a similar result in the attack strategy comparison
in Section 6.1, in which attacks using distant documents
were less effective. Taken together, these findings indicate
that adversarial documents that are geometrically closer to
the benign retrieved set yield more effective attacks.
We can also see in Table 4 that the relative ASR improve-
ments of SDAG for DS in comparison to NS is consistently
larger than that of CARG. As a result, we see in Table 4
that the relative ASR effectiveness of SDAG over CARG,
already observed in Table 1, widens when moving from NS
to DS. These observations suggest that the relative effec-
tiveness of SDAG over CARG is larger when the adversar-
ial documents are geometrically distant from the benign-
document set than when they are close. Still, we hasten
to point out that, as mentioned above, SDAG outperforms
CARG for adversarial documents that are near and distant
from the benign documents. The observed differences be-
tween DS and NS motivate the
Towards Dominant Set Selection.We next study the ques-
tion of which document subset, from the retrieved set, is
the one based on which the answer is generated when using
SDAG and CARG. We consider two types of (sub)sets in
the retrieved set: (i)Ground Truth Set(GTS), which con-
sists of documents containing the correct answer; and (ii)
Adversarial Set(AS), which consists of the adversarial docu-
ments of an attacker. Note that each set has a corresponding
answer (e.g., the correct answer for GTS and the attacker’sTable 5.Dominant-set-based generation of GTS for SDAG and
CARG, using E5. Boldface marks the best result for a k, generator,
dataset, and evaluation measure; ’∗’ marks a statistically significant
difference between SDAG and CARG for a k, generator, dataset,
and evaluation measure.
kGeneratorHotpotQA TriviaQA NQ
SDAG CARG SDAG CARG SDAG CARG
5Llama 0.67∗0.420.77∗0.600.70∗0.60
Qwen 0.510.420.690.660.660.59
Mistral 0.51∗0.400.81∗0.700.74∗0.64
10Llama 0.62∗0.380.85∗0.660.75∗0.63
Qwen 0.60∗0.430.76∗0.680.730.69
Mistral 0.75∗0.410.91∗0.760.92∗0.70
target incorrect answer for AS). We define thedominant set
as the one that includes the majority of documents in the
retrieved set. Exploring alternative dominance criteria or
additional set definitions is left for future work. We define
thedominant-set-based generationof a set as the proportion
of cases (i.e., questions) in which the RAG output corre-
sponds to the answer of the dominant set, out of all instances
in which the set is dominant. We focus on GTS analysis
because AS is never dominant in our experimental setting,
as it contains only a single document. In Appendix H we
present analysis of AS in the multiple-document setting.
Table 5 presents the dominant-set-based generation of GTS
for SDAG and CARG using E5. In all cases, SDAG achieves
a higher dominant-set-based generation than CARG, with
statistically significant improvements in most cases. This
finding attests to the fact that SDAG steers the generator to
attend more frequently to the GTS when it is dominant than
CARG, which helps to explain SDAG’s superiority over
CARG. These observations provide insight into the under-
lying mechanism of SDAG, which steers the RAG towards
deriving answers from a dominant (sub)set of documents.
7. Conclusions
We introduced a novel defense method against corpus knowl-
edge poisoning attacks: Sparse Document Attention RAG
(SDAG) that is based on a block-based attention mechanism.
Our extensive experiments showed that SDAG consistently
outperforms the standard causal-attention-based RAG. We
further compared SDAG with existing state-of-the-art RAG
defenses and showed that SDAG is a new state-of-the-art
defense for single-document attacks. Integrating SDAG
with existing defense methods yields the state-of-the-art for
multiple-documents attacks. Our findings show that SDAG
is a practical and effective approach to improve RAG perfor-
mance in terms of accuracy and ASR in adversarial settings.
Finally, we analyzed SDAG and its performance superiority
with respect to the induced embedding space of documents.
These findings lay the groundwork for future research on
sparse-attention in RAG systems.
8

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Impact Statement
LLMs are frequently used nowadays for question answer-
ing. RAG is applied in these settings to provide up-to-date
information and to reduce hallucinations. We pointed to a
vulnerability of RAG to corpus poisoning attacks due to the
standard causal-attention mechanism employed by LLMs.
We presented a novel, state-of-the-art defense method that
is based on block-based sparse attention. The method is
applied at inference time, does not call for finetuning, and
can be easily integrated with existing defense methods. We
also introduced new types of analysis, specifically based
on embedding spaces of documents, that shed light on the
effectiveness of attacks and defenses.
References
Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Ruther-
ford, E., Millican, K., Van Den Driessche, G. B., Lespiau,
J.-B., Damoc, B., Clark, A., De Las Casas, D., Guy, A.,
Menick, J., Ring, R., Hennigan, T., Huang, S., Maggiore,
L., Jones, C., Cassirer, A., Brock, A., Paganini, M., Irv-
ing, G., Vinyals, O., Osindero, S., Simonyan, K., Rae,
J., Elsen, E., and Sifre, L. Improving language models
by retrieving from trillions of tokens. In Chaudhuri, K.,
Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato,
S. (eds.),Proceedings of the 39th International Confer-
ence on Machine Learning, volume 162 ofProceedings
of Machine Learning Research, pp. 2206–2240. PMLR,
17–23 Jul 2022. URL https://proceedings.mlr.
press/v162/borgeaud22a.html.
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan,
J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G.,
Askell, A., Agarwal, S., Herbert-V oss, A., Krueger, G.,
Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu,
J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M.,
Gray, S., Chess, B., Clark, J., Berner, C., McCandlish,
S., Radford, A., Sutskever, I., and Amodei, D. Lan-
guage models are few-shot learners. InProceedings of
the 34th International Conference on Neural Informa-
tion Processing Systems, NIPS ’20, Red Hook, NY , USA,
2020. Curran Associates Inc. ISBN 9781713829546.
Cattan, A., Jacovi, A., Ram, O., Herzig, J., Aharoni, R.,
Goldshtein, S., Ofek, E., Szpektor, I., and Caciularu,
A. Dragged into conflicts: Detecting and addressing
conflicting sources in search-augmented llms, 2025. URL
https://arxiv.org/abs/2506.08500.
Chang, Z., Li, M., Jia, X., Wang, J., Huang, Y ., Jiang, Z.,
Liu, Y ., and Wang, Q. One shot dominance: Knowl-
edge poisoning attack on retrieval-augmented genera-
tion systems. In Christodoulopoulos, C., Chakraborty,
T., Rose, C., and Peng, V . (eds.),Findings of the Asso-
ciation for Computational Linguistics: EMNLP 2025,pp. 18811–18825, Suzhou, China, November 2025. As-
sociation for Computational Linguistics. ISBN 979-8-
89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.
1023. URL https://aclanthology.org/2025.
findings-emnlp.1023/.
Chen, D., Fisch, A., Weston, J., and Bordes, A. Reading
Wikipedia to answer open-domain questions. In Barzi-
lay, R. and Kan, M.-Y . (eds.),Proceedings of the 55th
Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pp. 1870–1879,
Vancouver, Canada, July 2017. Association for Compu-
tational Linguistics. doi: 10.18653/v1/P17-1171. URL
https://aclanthology.org/P17-1171/.
Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Cam-
pagnano, C., Maarek, Y ., Tonellotto, N., and Silvestri,
F. The power of noise: Redefining retrieval for rag sys-
tems. InProceedings of the 47th International ACM
SIGIR Conference on Research and Development in In-
formation Retrieval, SIGIR 2024, pp. 719–729. ACM,
July 2024. doi: 10.1145/3626772.3657834. URL http:
//dx.doi.org/10.1145/3626772.3657834.
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT:
Pre-training of deep bidirectional transformers for lan-
guage understanding. In Burstein, J., Doran, C., and
Solorio, T. (eds.),Proceedings of the 2019 Conference of
the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers), pp. 4171–4186, Min-
neapolis, Minnesota, June 2019. Association for Compu-
tational Linguistics. doi: 10.18653/v1/N19-1423. URL
https://aclanthology.org/N19-1423/.
Gao, L., Dai, Z., Pasupat, P., Chen, A., Chaganty, A. T.,
Fan, Y ., Zhao, V ., Lao, N., Lee, H., Juan, D.-C., and
Guu, K. RARR: Researching and revising what lan-
guage models say, using language models. In Rogers,
A., Boyd-Graber, J., and Okazaki, N. (eds.),Proceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
16477–16508, Toronto, Canada, July 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.
acl-long.910. URL https://aclanthology.org/
2023.acl-long.910/.
Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian,
A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A.,
Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn,
A., Yang, A., Mitra, A., Sravankumar, A., Korenev,
A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A.,
Gregerson, A., Spataru, A., Roziere, B., Biron, B., Tang,
B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra,
C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong,
C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D.,
9

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary,
D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes,
D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan,
E., Smith, E. M., Radenovic, F., Guzmán, F., Zhang, F.,
Synnaeve, G., Lee, G., Anderson, G. L., Thattai, G., Nail,
G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Ko-
revaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A.,
Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J.,
Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J.,
Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J.,
Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton,
J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia,
J., Alwala, K. V ., Prasad, K., Upasani, K., Plawiak, K., Li,
K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik,
K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary,
L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L.,
Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat,
L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh,
M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham,
M., Rita, M., Pavlova, M., Kambadur, M., Lewis, M.,
Si, M., Singh, M. K., Hassan, M., Goyal, N., Torabi, N.,
Bashlykov, N., Bogoychev, N., Chatterji, N., Zhang, N.,
Duchenne, O., Çelebi, O., Alrassy, P., Zhang, P., Li, P.,
Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan,
P., Koura, P. S., Xu, P., He, Q., Dong, Q., Srinivasan,
R., Ganapathy, R., Calderer, R., Cabral, R. S., Stojnic,
R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R.,
Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R., Silva,
R., Hou, R., Wang, R., Hosseini, S., Chennabasappa, S.,
Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie, S., Narang,
S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang,
S., Vandenhende, S., Batra, S., Whitman, S., Sootla, S.,
Collot, S., Gururangan, S., Borodinsky, S., Herman, T.,
Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speck-
bacher, T., Mihaylov, T., Xiao, T., Karn, U., Goswami, V .,
Gupta, V ., Ramanathan, V ., Kerkez, V ., Gonguet, V ., Do,
V ., V ogeti, V ., Albiero, V ., Petrovic, V ., Chu, W., Xiong,
W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang,
X., Tan, X. E., Xia, X., Xie, X., Jia, X., Wang, X., Gold-
schlag, Y ., Gaur, Y ., Babaei, Y ., Wen, Y ., Song, Y ., Zhang,
Y ., Li, Y ., Mao, Y ., Coudert, Z. D., Yan, Z., Chen, Z.,
Papakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey,
A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand,
A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A.,
Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A.,
Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poul-
ton, A., Ryan, A., Ramchandani, A., Dong, A., Franco,
A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A.,
Bharambe, A., Eisenman, A., Yazdan, A., James, B.,
Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola,
B. D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock,
B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B.,
Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu, C.,
Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H., Cai, C.,Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty,
D., Kreymer, D., Li, D., Adkins, D., Xu, D., Testuggine,
D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang,
D., Le, D., Holland, D., Dowling, E., Jamil, E., Mont-
gomery, E., Presani, E., Hahn, E., Wood, E., Le, E.-T.,
Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun,
F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Cag-
gioni, F., Kanayet, F., Seide, F., Florez, G. M., Schwarz,
G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov,
G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H.,
Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb, H.,
Rudolph, H., Suk, H., Aspegren, H., Goldman, H., Zhan,
H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I.,
Veliche, I.-E., Gat, I., Weissman, J., Geboski, J., Kohli,
J., Lam, J., Asher, J., Gaya, J.-B., Marcus, J., Tang, J.,
Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J.,
Jin, J., Yang, J., Cummings, J., Carvill, J., Shepard, J.,
McPhie, J., Torres, J., Ginsburg, J., Wang, J., Wu, K., U,
K. H., Saxena, K., Khandelwal, K., Zand, K., Matosich,
K., Veeraraghavan, K., Michelena, K., Li, K., Jagadeesh,
K., Huang, K., Chawla, K., Huang, K., Chen, L., Garg,
L., A, L., Silva, L., Bell, L., Zhang, L., Guo, L., Yu, L.,
Moshkovich, L., Wehrstedt, L., Khabsa, M., Avalani, M.,
Bhatt, M., Mankus, M., Hasson, M., Lennie, M., Reso,
M., Groshev, M., Naumov, M., Lathi, M., Keneally, M.,
Liu, M., Seltzer, M. L., Valko, M., Restrepo, M., Patel,
M., Vyatskov, M., Samvelyan, M., Clark, M., Macey,
M., Wang, M., Hermoso, M. J., Metanat, M., Rastegari,
M., Bansal, M., Santhanam, N., Parks, N., White, N.,
Bawa, N., Singhal, N., Egebo, N., Usunier, N., Mehta,
N., Laptev, N. P., Dong, N., Cheng, N., Chernoguz, O.,
Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P.,
Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P.,
Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P.,
Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy, R.,
Nayani, R., Mitra, R., Parthasarathy, R., Li, R., Hogan,
R., Battey, R., Wang, R., Howes, R., Rinott, R., Mehta,
S., Siby, S., Bondu, S. J., Datta, S., Chugh, S., Hunt, S.,
Dhillon, S., Sidorov, S., Pan, S., Mahajan, S., Verma,
S., Yamamoto, S., Ramaswamy, S., Lindsay, S., Lindsay,
S., Feng, S., Lin, S., Zha, S. C., Patil, S., Shankar, S.,
Zhang, S., Zhang, S., Wang, S., Agarwal, S., Sajuyigbe,
S., Chintala, S., Max, S., Chen, S., Kehoe, S., Satter-
field, S., Govindaprasad, S., Gupta, S., Deng, S., Cho,
S., Virk, S., Subramanian, S., Choudhury, S., Goldman,
S., Remez, T., Glaser, T., Best, T., Koehler, T., Robinson,
T., Li, T., Zhang, T., Matthews, T., Chou, T., Shaked,
T., V ontimitta, V ., Ajayi, V ., Montanez, V ., Mohan, V .,
Kumar, V . S., Mangla, V ., Ionescu, V ., Poenaru, V ., Mi-
hailescu, V . T., Ivanov, V ., Li, W., Wang, W., Jiang, W.,
Bouaziz, W., Constable, W., Tang, X., Wu, X., Wang, X.,
Wu, X., Gao, X., Kleinman, Y ., Chen, Y ., Hu, Y ., Jia, Y .,
Qi, Y ., Li, Y ., Zhang, Y ., Zhang, Y ., Adi, Y ., Nam, Y ., Yu,
Wang, Zhao, Y ., Hao, Y ., Qian, Y ., Li, Y ., He, Y ., Rait,
10

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z., Zhao,
Z., and Ma, Z. The llama 3 herd of models, 2024. URL
https://arxiv.org/abs/2407.21783.
Hong, G., Kim, J., Kang, J., Myaeng, S.-H., and Whang,
J. J. Why so gullible? enhancing the robustness
of retrieval-augmented models against counterfactual
noise. In Duh, K., Gomez, H., and Bethard, S.
(eds.),Findings of the Association for Computational
Linguistics: NAACL 2024, pp. 2474–2495, Mexico
City, Mexico, June 2024. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2024.findings-naacl.
159. URL https://aclanthology.org/2024.
findings-naacl.159/.
Izacard, G. and Grave, E. Leveraging passage retrieval
with generative models for open domain question answer-
ing. In Merlo, P., Tiedemann, J., and Tsarfaty, R. (eds.),
Proceedings of the 16th Conference of the European
Chapter of the Association for Computational Linguistics:
Main Volume, pp. 874–880, Online, April 2021. Associ-
ation for Computational Linguistics. doi: 10.18653/v1/
2021.eacl-main.74. URL https://aclanthology.
org/2021.eacl-main.74/.
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski,
P., Joulin, A., and Grave, E. Unsupervised dense infor-
mation retrieval with contrastive learning, 2022. URL
https://arxiv.org/abs/2112.09118.
Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C.,
Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel,
G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-
A., Stock, P., Scao, T. L., Lavril, T., Wang, T., Lacroix,
T., and Sayed, W. E. Mistral 7b, 2023a. URL https:
//arxiv.org/abs/2310.06825.
Jiang, Z., Xu, F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu,
J., Yang, Y ., Callan, J., and Neubig, G. Active retrieval
augmented generation. In Bouamor, H., Pino, J., and
Bali, K. (eds.),Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pp.
7969–7992, Singapore, December 2023b. Association
for Computational Linguistics. doi: 10.18653/v1/2023.
emnlp-main.495. URL https://aclanthology.
org/2023.emnlp-main.495/.
Jin, B., Yoon, J., Han, J., and Arik, S. Long-context
llms meet rag: Overcoming challenges for long
inputs in rag. In Yue, Y ., Garg, A., Peng, N., Sha,
F., and Yu, R. (eds.),International Conference on
Representation Learning, volume 2025, pp. 37784–
37822, 2025. URL https://proceedings.
iclr.cc/paper_files/paper/2025/file/
5df5b1f121c915d8bdd00db6aac20827-Paper-Conference.
pdf.Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. Trivi-
aQA: A large scale distantly supervised challenge dataset
for reading comprehension. In Barzilay, R. and Kan,
M.-Y . (eds.),Proceedings of the 55th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 1601–1611, Vancouver,
Canada, July 2017. Association for Computational Lin-
guistics. doi: 10.18653/v1/P17-1147. URL https:
//aclanthology.org/P17-1147/.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. In Web-
ber, B., Cohn, T., He, Y ., and Liu, Y . (eds.),Proceed-
ings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP), pp. 6769–
6781, Online, November 2020. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2020.emnlp-main.
550. URL https://aclanthology.org/2020.
emnlp-main.550/.
Kim, M., Lee, H., and Koo, H. Rescuing the unpoisoned:
Efficient defense against knowledge corruption attacks
on rag systems, 2025. URL https://arxiv.org/
abs/2511.01268.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I.,
Devlin, J., Lee, K., Toutanova, K., Jones, L., Kel-
cey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le,
Q., and Petrov, S. Natural questions: A benchmark
for question answering research.Transactions of the
Association for Computational Linguistics, 7:452–466,
2019. doi: 10.1162/tacl_a_00276. URL https://
aclanthology.org/Q19-1026/.
Lee, K., Chang, M.-W., and Toutanova, K. Latent retrieval
for weakly supervised open domain question answering.
In Korhonen, A., Traum, D., and Màrquez, L. (eds.),
Proceedings of the 57th Annual Meeting of the Asso-
ciation for Computational Linguistics, pp. 6086–6096,
Florence, Italy, July 2019. Association for Computa-
tional Linguistics. doi: 10.18653/v1/P19-1612. URL
https://aclanthology.org/P19-1612/.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel,
T., Riedel, S., and Kiela, D. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. In Larochelle, H.,
Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.),
Advances in Neural Information Processing Systems,
volume 33, pp. 9459–9474. Curran Associates, Inc.,
2020. URL https://proceedings.neurips.
cc/paper_files/paper/2020/file/
6b493230205f780e1bc26945df7481e5-Paper.
pdf.
11

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Li, D., Sun, Z., Hu, X., Liu, Z., Chen, Z., Hu, B., Wu,
A., and Zhang, M. A survey of large language models
attribution, 2023. URL https://arxiv.org/abs/
2311.03731.
Lin, J., Ma, X., Lin, S.-C., Yang, J.-H., Pradeep, R., and
Nogueira, R. Pyserini: A Python toolkit for reproducible
information retrieval research with sparse and dense rep-
resentations. InProceedings of the 44th Annual Inter-
national ACM SIGIR Conference on Research and De-
velopment in Information Retrieval (SIGIR 2021), pp.
2356–2362, 2021.
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts.Transactions of the
Association for Computational Linguistics, 12:157–173,
2024. doi: 10.1162/tacl_a_00638. URL https://
aclanthology.org/2024.tacl-1.9/.
Ma, D., Wang, Y ., and Lan, T. Block-attention for
efficient prefilling. In Yue, Y ., Garg, A., Peng, N.,
Sha, F., and Yu, R. (eds.),International Conference
on Representation Learning, volume 2025, pp. 63774–
63788, 2025. URL https://proceedings.
iclr.cc/paper_files/paper/2025/file/
a03037317560b8c5f2fb4b6466d4c439-Paper-Conference.
pdf.
Minaee, S., Mikolov, T., Nikzad, N., Chenaghlu, M., Socher,
R., Amatriain, X., and Gao, J. Large language models:
A survey, 2025. URL https://arxiv.org/abs/
2402.06196.
OpenAI, :, Hurst, A., Lerer, A., Goucher, A. P., Perelman,
A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A.,
Hayes, A., Radford, A., M ˛ adry, A., Baker-Whitcomb, A.,
Beutel, A., Borzunov, A., Carney, A., Chow, A., Kirillov,
A., Nichol, A., Paino, A., Renzin, A., Passos, A. T., Kir-
illov, A., Christakis, A., Conneau, A., Kamali, A., Jabri,
A., Moyer, A., Tam, A., Crookes, A., Tootoochian, A.,
Tootoonchian, A., Kumar, A., Vallone, A., Karpathy, A.,
Braunstein, A., Cann, A., Codispoti, A., Galu, A., Kon-
drich, A., Tulloch, A., Mishchenko, A., Baek, A., Jiang,
A., Pelisse, A., Woodford, A., Gosalia, A., Dhar, A., Pan-
tuliano, A., Nayak, A., Oliver, A., Zoph, B., Ghorbani, B.,
Leimberger, B., Rossen, B., Sokolowsky, B., Wang, B.,
Zweig, B., Hoover, B., Samic, B., McGrew, B., Spero, B.,
Giertler, B., Cheng, B., Lightcap, B., Walkin, B., Quinn,
B., Guarraci, B., Hsu, B., Kellogg, B., Eastman, B., Lu-
garesi, C., Wainwright, C., Bassin, C., Hudson, C., Chu,
C., Nelson, C., Li, C., Shern, C. J., Conger, C., Barette,
C., V oss, C., Ding, C., Lu, C., Zhang, C., Beaumont, C.,
Hallacy, C., Koch, C., Gibson, C., Kim, C., Choi, C.,
McLeavey, C., Hesse, C., Fischer, C., Winter, C., Czar-
necki, C., Jarvis, C., Wei, C., Koumouzelis, C., Sherburn,D., Kappler, D., Levin, D., Levy, D., Carr, D., Farhi, D.,
Mely, D., Robinson, D., Sasaki, D., Jin, D., Valladares,
D., Tsipras, D., Li, D., Nguyen, D. P., Findlay, D., Oiwoh,
E., Wong, E., Asdar, E., Proehl, E., Yang, E., Antonow, E.,
Kramer, E., Peterson, E., Sigler, E., Wallace, E., Brevdo,
E., Mays, E., Khorasani, F., Such, F. P., Raso, F., Zhang,
F., von Lohmann, F., Sulit, F., Goh, G., Oden, G., Salmon,
G., Starace, G., Brockman, G., Salman, H., Bao, H.,
Hu, H., Wong, H., Wang, H., Schmidt, H., Whitney, H.,
Jun, H., Kirchner, H., de Oliveira Pinto, H. P., Ren, H.,
Chang, H., Chung, H. W., Kivlichan, I., O’Connell, I.,
O’Connell, I., Osband, I., Silber, I., Sohl, I., Okuyucu,
I., Lan, I., Kostrikov, I., Sutskever, I., Kanitscheider, I.,
Gulrajani, I., Coxon, J., Menick, J., Pachocki, J., Aung, J.,
Betker, J., Crooks, J., Lennon, J., Kiros, J., Leike, J., Park,
J., Kwon, J., Phang, J., Teplitz, J., Wei, J., Wolfe, J., Chen,
J., Harris, J., Varavva, J., Lee, J. G., Shieh, J., Lin, J., Yu,
J., Weng, J., Tang, J., Yu, J., Jang, J., Candela, J. Q., Beut-
ler, J., Landers, J., Parish, J., Heidecke, J., Schulman, J.,
Lachman, J., McKay, J., Uesato, J., Ward, J., Kim, J. W.,
Huizinga, J., Sitkin, J., Kraaijeveld, J., Gross, J., Ka-
plan, J., Snyder, J., Achiam, J., Jiao, J., Lee, J., Zhuang,
J., Harriman, J., Fricke, K., Hayashi, K., Singhal, K.,
Shi, K., Karthik, K., Wood, K., Rimbach, K., Hsu, K.,
Nguyen, K., Gu-Lemberg, K., Button, K., Liu, K., Howe,
K., Muthukumar, K., Luther, K., Ahmad, L., Kai, L., Itow,
L., Workman, L., Pathak, L., Chen, L., Jing, L., Guy, L.,
Fedus, L., Zhou, L., Mamitsuka, L., Weng, L., McCal-
lum, L., Held, L., Ouyang, L., Feuvrier, L., Zhang, L.,
Kondraciuk, L., Kaiser, L., Hewitt, L., Metz, L., Doshi,
L., Aflak, M., Simens, M., Boyd, M., Thompson, M.,
Dukhan, M., Chen, M., Gray, M., Hudnall, M., Zhang, M.,
Aljubeh, M., Litwin, M., Zeng, M., Johnson, M., Shetty,
M., Gupta, M., Shah, M., Yatbaz, M., Yang, M. J., Zhong,
M., Glaese, M., Chen, M., Janner, M., Lampe, M., Petrov,
M., Wu, M., Wang, M., Fradin, M., Pokrass, M., Castro,
M., de Castro, M. O. T., Pavlov, M., Brundage, M., Wang,
M., Khan, M., Murati, M., Bavarian, M., Lin, M., Yesil-
dal, M., Soto, N., Gimelshein, N., Cone, N., Staudacher,
N., Summers, N., LaFontaine, N., Chowdhury, N., Ryder,
N., Stathas, N., Turley, N., Tezak, N., Felix, N., Kudige,
N., Keskar, N., Deutsch, N., Bundick, N., Puckett, N.,
Nachum, O., Okelola, O., Boiko, O., Murk, O., Jaffe, O.,
Watkins, O., Godement, O., Campbell-Moore, O., Chao,
P., McMillan, P., Belov, P., Su, P., Bak, P., Bakkum, P.,
Deng, P., Dolan, P., Hoeschele, P., Welinder, P., Tillet,
P., Pronin, P., Tillet, P., Dhariwal, P., Yuan, Q., Dias,
R., Lim, R., Arora, R., Troll, R., Lin, R., Lopes, R. G.,
Puri, R., Miyara, R., Leike, R., Gaubert, R., Zamani, R.,
Wang, R., Donnelly, R., Honsby, R., Smith, R., Sahai, R.,
Ramchandani, R., Huet, R., Carmichael, R., Zellers, R.,
Chen, R., Chen, R., Nigmatullin, R., Cheu, R., Jain, S.,
Altman, S., Schoenholz, S., Toizer, S., Miserendino, S.,
Agarwal, S., Culver, S., Ethersmith, S., Gray, S., Grove,
12

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
S., Metzger, S., Hermani, S., Jain, S., Zhao, S., Wu, S.,
Jomoto, S., Wu, S., Shuaiqi, Xia, Phene, S., Papay, S.,
Narayanan, S., Coffey, S., Lee, S., Hall, S., Balaji, S.,
Broda, T., Stramer, T., Xu, T., Gogineni, T., Christian-
son, T., Sanders, T., Patwardhan, T., Cunninghman, T.,
Degry, T., Dimson, T., Raoux, T., Shadwell, T., Zheng,
T., Underwood, T., Markov, T., Sherbakov, T., Rubin, T.,
Stasi, T., Kaftan, T., Heywood, T., Peterson, T., Walters,
T., Eloundou, T., Qi, V ., Moeller, V ., Monaco, V ., Kuo,
V ., Fomenko, V ., Chang, W., Zheng, W., Zhou, W., Man-
assra, W., Sheu, W., Zaremba, W., Patil, Y ., Qian, Y .,
Kim, Y ., Cheng, Y ., Zhang, Y ., He, Y ., Zhang, Y ., Jin, Y .,
Dai, Y ., and Malkov, Y . Gpt-4o system card, 2024. URL
https://arxiv.org/abs/2410.21276.
Ratner, N., Levine, Y ., Belinkov, Y ., Ram, O., Magar, I.,
Abend, O., Karpas, E., Shashua, A., Leyton-Brown, K.,
and Shoham, Y . Parallel context windows for large lan-
guage models. In Rogers, A., Boyd-Graber, J., and
Okazaki, N. (eds.),Proceedings of the 61st Annual Meet-
ing of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pp. 6383–6402, Toronto, Canada,
July 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.acl-long.352. URL https:
//aclanthology.org/2023.acl-long.352/.
Robertson, S. and Zaragoza, H. The probabilistic relevance
framework: Bm25 and beyond.Found. Trends Inf. Retr., 3
(4):333–389, April 2009. ISSN 1554-0669. doi: 10.1561/
1500000019. URL https://doi.org/10.1561/
1500000019.
Shen, Z., Imana, B. Y ., Wu, T., Xiang, C., Mittal, P., and Ko-
rolova, A. ReliabilityRAG: Effective and provably robust
defense for RAG-based web-search. InThe Thirty-ninth
Annual Conference on Neural Information Processing
Systems, 2025. URL https://openreview.net/
forum?id=D9JeNTs5Bu.
Si, X., Zhu, M., Qin, S., Yu, L., Zhang, L., Liu, S., Li, X.,
Duan, R., Liu, Y ., and Jia, X. Secon-RAG: A two-stage se-
mantic filtering and conflict-free framework for trustwor-
thy RAG. InThe Thirty-ninth Annual Conference on Neu-
ral Information Processing Systems, 2025. URL https:
//openreview.net/forum?id=tTwZhy8JqY.
Tan, X., Luan, H., Luo, M., Sun, X., Chen, P., and
Dai, J. RevPRAG: Revealing poisoning attacks in
retrieval-augmented generation through LLM activation
analysis. In Christodoulopoulos, C., Chakraborty, T.,
Rose, C., and Peng, V . (eds.),Findings of the Asso-
ciation for Computational Linguistics: EMNLP 2025,
pp. 12999–13011, Suzhou, China, November 2025. As-
sociation for Computational Linguistics. ISBN 979-8-
89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.698. URL https://aclanthology.org/2025.
findings-emnlp.698/.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal,
A. Interleaving retrieval with chain-of-thought reasoning
for knowledge-intensive multi-step questions. In Rogers,
A., Boyd-Graber, J., and Okazaki, N. (eds.),Proceed-
ings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pp.
10014–10037, Toronto, Canada, July 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.
acl-long.557. URL https://aclanthology.org/
2023.acl-long.557/.
Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R.,
and Wei, F. Improving text embeddings with large lan-
guage models. In Ku, L.-W., Martins, A., and Srikumar,
V . (eds.),Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pp. 11897–11916, Bangkok, Thailand,
August 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.acl-long.642. URL https:
//aclanthology.org/2024.acl-long.642/.
Wang, Z., Ng, P., Ma, X., Nallapati, R., and Xiang,
B. Multi-passage BERT: A globally normalized BERT
model for open-domain question answering. In Inui,
K., Jiang, J., Ng, V ., and Wan, X. (eds.),Proceed-
ings of the 2019 Conference on Empirical Methods
in Natural Language Processing and the 9th Interna-
tional Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pp. 5878–5882, Hong Kong, China,
November 2019. Association for Computational Lin-
guistics. doi: 10.18653/v1/D19-1599. URL https:
//aclanthology.org/D19-1599/.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B.,
Xia, F., Chi, E. H., Le, Q. V ., and Zhou, D. Chain-of-
thought prompting elicits reasoning in large language
models. InProceedings of the 36th International Confer-
ence on Neural Information Processing Systems, NIPS
’22, Red Hook, NY , USA, 2022. Curran Associates Inc.
ISBN 9781713871088.
Xi, M., Lv, S., Jin, Y ., Cheng, G., Wang, N., Li, Y ., and Yin,
J. Riprag: Hack a black-box retrieval-augmented gen-
eration question-answering system with reinforcement
learning, 2025. URL https://arxiv.org/abs/
2510.10008.
Xiang, C., Wu, T., Zhong, Z., Wagner, D., Chen, D., and Mit-
tal, P. Certifiably robust RAG against retrieval corruption.
InICML 2024 Next Generation of AI Safety Workshop,
2024. URL https://openreview.net/forum?
id=qsEeACAJjD.
13

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Xiao, E., Li, C.-J., Zhang, Y ., Neubig, G., and Bertsch,
A. Efficient many-shot in-context learning with dy-
namic block-sparse attention. In Che, W., Nabende, J.,
Shutova, E., and Pilehvar, M. T. (eds.),Proceedings of
the 63rd Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers), pp.
31946–31958, Vienna, Austria, July 2025. Association
for Computational Linguistics. ISBN 979-8-89176-251-0.
doi: 10.18653/v1/2025.acl-long.1542. URL https://
aclanthology.org/2025.acl-long.1542/.
Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z.,
Subramanian, S., Bakhturina, E., Shoeybi, M., and Catan-
zaro, B. Retrieval meets long context large language mod-
els, 2024. URL https://arxiv.org/abs/2310.
03025.
Xue, J., Zheng, M., Hu, Y ., Liu, F., Chen, X., and Lou,
Q. Badrag: Identifying vulnerabilities in retrieval aug-
mented generation of large language models.CoRR,
abs/2406.00083, 2024.
Yang, A., Yang, B., Hui, B., Zheng, B., Yu, B., Zhou, C.,
Li, C., Li, C., Liu, D., Huang, F., Dong, G., Wei, H.,
Lin, H., Tang, J., Wang, J., Yang, J., Tu, J., Zhang, J.,
Ma, J., Yang, J., Xu, J., Zhou, J., Bai, J., He, J., Lin,
J., Dang, K., Lu, K., Chen, K., Yang, K., Li, M., Xue,
M., Ni, N., Zhang, P., Wang, P., Peng, R., Men, R., Gao,
R., Lin, R., Wang, S., Bai, S., Tan, S., Zhu, T., Li, T.,
Liu, T., Ge, W., Deng, X., Zhou, X., Ren, X., Zhang,
X., Wei, X., Ren, X., Liu, X., Fan, Y ., Yao, Y ., Zhang,
Y ., Wan, Y ., Chu, Y ., Liu, Y ., Cui, Z., Zhang, Z., Guo,
Z., and Fan, Z. Qwen2 technical report, 2024. URL
https://arxiv.org/abs/2407.10671.
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng,
B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu,
D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin,
H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang,
J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang,
K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang,
P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo,
S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang,
X., Ren, X., Fan, Y ., Su, Y ., Zhang, Y ., Zhang, Y ., Wan,
Y ., Liu, Y ., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and
Qiu, Z. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W.,
Salakhutdinov, R., and Manning, C. D. HotpotQA: A
dataset for diverse, explainable multi-hop question an-
swering. In Riloff, E., Chiang, D., Hockenmaier, J.,
and Tsujii, J. (eds.),Proceedings of the 2018 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pp. 2369–2380, Brussels, Belgium, October-
November 2018. Association for Computational Lin-guistics. doi: 10.18653/v1/D18-1259. URL https:
//aclanthology.org/D18-1259/.
Yoran, O., Wolfson, T., Ram, O., and Berant, J. Making
retrieval-augmented language models robust to irrelevant
context. In Kim, B., Yue, Y ., Chaudhuri, S., Fragki-
adaki, K., Khan, M., and Sun, Y . (eds.),International
Conference on Representation Learning, volume 2024,
pp. 29862–29883, 2024.
Yu, W., Iter, D., Wang, S., Xu, Y ., Ju, M., Sanyal, S., Zhu,
C., Zeng, M., and Jiang, M. Generate rather than retrieve:
Large language models are strong context generators.
InThe Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.
net/forum?id=fB0hRu9GZUS.
Yu, Y ., Ping, W., Liu, Z., Wang, B., You, J., Zhang, C.,
Shoeybi, M., and Catanzaro, B. RankRAG: Unifying
context ranking with retrieval-augmented generation in
LLMs. InThe Thirty-eighth Annual Conference on Neu-
ral Information Processing Systems, 2024. URL https:
//openreview.net/forum?id=S1fc92uemC.
Zhong, Z., Huang, Z., Wettig, A., and Chen, D. Poi-
soning retrieval corpora by injecting adversarial pas-
sages. In Bouamor, H., Pino, J., and Bali, K. (eds.),
Proceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 13764–13775,
Singapore, December 2023. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2023.emnlp-main.
849. URL https://aclanthology.org/2023.
emnlp-main.849/.
Zhou, H., Lee, K.-H., Zhan, Z., Chen, Y ., Li, Z., Wang,
Z., Haddadi, H., and Yilmaz, E. Trustrag: Enhancing
robustness and trustworthiness in retrieval-augmented
generation, 2025. URL https://arxiv.org/abs/
2501.00879.
Zhu, H., Fiondella, L., Yuan, J., Zeng, K., and Jiao, L.
Neurogenpoisoning: Neuron-guided attacks on retrieval-
augmented generation of LLM via genetic optimiza-
tion of external knowledge. InThe Thirty-ninth Annual
Conference on Neural Information Processing Systems,
2025a. URL https://openreview.net/forum?
id=4kTpb8pITI.
Zhu, Y ., Gu, J.-C., Sikora, C., Ko, H., Liu, Y ., Lin, C.-C.,
Shu, L., Luo, L., Meng, L., Liu, B., and Chen, J.
Accelerating inference of retrieval-augmented generation
via sparse context selection. In Yue, Y ., Garg, A., Peng,
N., Sha, F., and Yu, R. (eds.),International Conference
on Learning Representations, volume 2025, pp. 25965–
25981, 2025b. URL https://proceedings.
iclr.cc/paper_files/paper/2025/file/
14

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
411fa9d368b5485be4c6bb62615b365e-Paper-Conference.
pdf.
Zou, W., Geng, R., Wang, B., and Jia, J. Poisonedrag:
knowledge corruption attacks to retrieval-augmented gen-
eration of large language models. InProceedings of the
34th USENIX Conference on Security Symposium, SEC
’25, USA, 2025. USENIX Association. ISBN 978-1-
939133-52-6.
Figure 1.Block-sparse attention patterns used in SDAG. The entry
(i, j) is colored blue if token iis allowed to attend token jand
white if not. Dark-blue represents task or context tokens, light-blue
represents retrieved documents tokens.
Table 6.Performance of SDAG and CARG using a generator
with reasoning abilities (Qwen3) and E5 as the retriever. Boldface
marks the best result for a k, dataset, and evaluation measure;
’∗’ marks a statistically significant difference between SDAG and
CARG for ak, dataset, and evaluation measure.
kMethodHotpotQA TriviaQA NQ
ACC ASR ACC ASR ACC ASR
5SDAG 0.36 0.44∗0.83 0.44∗0.29 0.36∗
CARG 0.33 0.84 0.80 0.81 0.26 0.51
10SDAG 0.34 0.27∗0.81 0.29∗0.41∗0.17∗
CARG 0.340.81 0.80 0.78 0.27 0.29
A. A Visual Illustration of SDAG Attention
Mask
Figure 1 provides a visual illustration of the block-sparse at-
tention mask SDAG. The figure shows the document-based
block patterns that are formed in the attention mask at infer-
ence time.
B. Effect of Several Factors on SDAG’s
Performance
We next evaluate SDAG using a generator with reasoning
abilities, using a generator with a large size (i.e., number of
parameters), and with varying temperature values. We use
the single adversarial document in-set setting and the Ran-
dom attack strategy, as in the comparison between SDAG
and CARG in Section 6.1.
Using a Reasoning Generator.In recent years, LLMs have
demonstrated improved effectiveness in a variety of tasks
15

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 7.Performance of SDAG and CARG using Llama-70B with
E5. Boldface marks the best result for a k, dataset, and evaluation
measure; ’∗’ marks a statistically significant difference between
SDAG and CARG for ak, dataset, and evaluation measure.
kMethodHotpotQA TriviaQA NQ
ACC ASR ACC ASR ACC ASR
5SDAG 0.26 0.40∗0.70 0.23∗0.44 0.18∗
CARG 0.23 0.59 0.69 0.42 0.42 0.39
10SDAG 0.29 0.19∗0.80∗0.09∗0.460.07∗
CARG 0.27 0.52 0.74 0.320.470.29
when leveraging reasoning capabilities (Wei et al., 2022),
specifically, when employing RAG using a generator with
reasoning capabilities (Trivedi et al., 2023). To evaluate the
effect of using RAG with a reasoning generator on SDAG
effectiveness, we use Qwen3-8B (Yang et al., 2025), a state-
of-the-art LLM equipped with reasoning abilities. Due to
computational resources, we limit the generation sequence
to 512 tokens; consequently, we filter out questions with
an open reasoning sequence (i.e., the generator has not
provided an answer within the sequence limit).
Table 6 reports the performance of SDAG and CARG us-
ing Qwen3 and E5. We can see that in all cases SDAG
statistically significantly outperforms CARG in terms of
ASR. In addition, SDAG always posts higher accuracy than
that of CARG. These results attest to the fact that SDAG is
also more effective against knowledge poisoning attacks for
RAG with a reasoning generator than CARG.
The Effect of Using Larger Generators.Heretofore, the
evaluation was based on LLMs of size ≈7B, as is common
practice in prior work on RAG for QA. We next evaluate
SDAG using Llama-3-70B-Instruct (Grattafiori et al., 2024)
to study the effect of increasing the generator size in terms of
the number of parameters. Table 7 presents the performance
of SDAG and CARG using E5 as the retriever. We can see
that SDAG in most relevant comparisons attains accuracy
higher and ASR statistically significantly lower than that of
CARG. This indicates that SDAG is also effective against
knowledge poisoning attacks with large generators. As was
the case for the smaller generators, we observe in Table 7
that the relative ASR effectiveness of SDAG over CARG is
larger in a small adversarial documents ratio ( k= 10 ) than
in a relatively large one (k= 5).
The Effect of Temperature on SDAG’s Performance.In
Section 6 we evaluated SDAG’s performance when the gen-
erator’s sampling temperature value was set to 0.1. In Ta-
ble 8 we report SDAG’s performance on the NQ dataset
with a wide variety of temperature values; E5 is used as
the retriever. We can see that SDAG’s performance is quite
stable with respect to different values of temperature. The
best accuracy is attained for a temperature of 0.1 (which was
used in Section 6) and the best ASR for a temperature of 1.0.Table 8.Performance of SDAG on the NQ dataset with different
values of temperature; Llama is used as the generator and E5 is
used as the retriever. Boldface marks the best result for a kand
evaluation measure.
kTemperature Accuracy ASR
50.0 0.330.23
0.1 0.330.25
0.3 0.330.24
0.5 0.330.23
0.8 0.310.21
1.0 0.320.21
100.0 0.33 0.15
0.1 0.350.17
0.3 0.31 0.15
0.5 0.32 0.15
0.8 0.32 0.15
1.0 0.320.13
The best attained performance is statistically significantly
indistinguishable from the other performance numbers for
temperature values. These results indicate a stable perfor-
mance of SDAG across varying values of the generator’s
sampling temperature.
C. RAG Prompt
Following prior work (Zou et al., 2025), we use the follow-
ing prompt for the generator:
You are a helpful assistant, below is a query
from a user and some relevant contexts.
Answer the question based on the following passages.
Your answer should be short and concise.
Passages: {retrieved documents}
Question: {query}
Answer:
D. SDAG’s Performance in the In-Corpus
Setting
We now turn to evaluate SDAG in the single-adversarial-
document in-corpus setting, where an attacker inserts an
adversarial document into the corpus, rather than into the
retrieved set. We use the Random attack strategy and E5
as the retriever, as in Section 6.1. Recall that we use the
PoisonedRAG framework to generate the adversarial docu-
ments (Zou et al., 2025), which promotes document ranking
by adding the question at the beginning of each document.
We begin by comparing the performance of SDAG with that
of CARG in the in-corpus setting. We can see in Table 9
that, in all cases, SDAG attains statistically significantly
lower ASR than than that of CARG, and in most cases a
higher accuracy. Notably, in the very few cases that SDAG
does not post higher accuracy, its accuracy is equal to that
16

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 9.Performance of SDAG and CARG in the in-corpus setting,
using E5 as the retriever. LLM denotes the generator. Boldface
marks the best result for an LLM, k, dataset, and evaluation mea-
sure; ’∗’ marks a statistically significant difference between SDAG
and CARG for an LLM,k, dataset, and evaluation measure.
LLMkHotpotQA TriviaQA NQ
SDAG CARG SDAG CARG SDAG CARG
ACC ASR ACC ASR ACC ASR ACC ASR ACC ASR ACC ASR
Llama50.20∗0.52∗0.16 0.670.62∗0.27∗0.57 0.450.36 0.22∗0.360.32
100.24∗0.36∗0.19 0.620.70∗0.17∗0.61 0.410.39 0.14∗0.390.36
Qwen50.19∗0.52∗0.15 0.700.59 0.35∗0.55 0.500.37 0.21∗0.36 0.35
100.24∗0.31∗0.17 0.670.70∗0.19∗0.59 0.470.37 0.13∗0.370.38
Mistral50.21∗0.53∗0.17 0.700.71∗0.36∗0.65 0.540.43 0.23∗0.41 0.36
100.27∗0.40∗0.20 0.670.77∗0.26∗0.68 0.500.42 0.19∗0.41 0.41
of CARG. These results indicate that the effectiveness of
SDAG we observed in Section 6 for the in-set setting (where
an adversarial document is always present in the retrieved
set) carries over to the in-corpus setting.
In Table 10 we compare the performance of SDAG with
that of the RAGDefender and the Discern-and-Answer base-
lines in the in-corpus setting. We also present the perfor-
mance of the integration of SDAG with the baselines. As
in Section 6.1, we use Llama as the generator. In the vast
majority of cases, SDAG achieves statistically significantly
lower ASR and higher accuracy than the baselines. No-
tably, SDAG always posts lower ASR than that of the base-
lines and in the few cases that SDAG does not post higher
accuracy, its accuracy is statistically significantly indistin-
guishable from that of the baselines. Furthermore, SDAG
integrated with a baseline achieves statistically significantly
lower ASR and higher accuracy than the baseline. These
findings attest to the effectiveness of SDAG when integrated
with an existing RAG defense. Overall, these results attest
to SDAG being a new state-of-the-art in terms of accuracy
and ASR also for the single-adversarial-document in-corpus
setting.
E. The Effect of Adversarial Document
Location in the Prompt on SDAG’s
Performance
We now study the effect of the location of the adversarial
document in the prompt on the performance of SDAG. Re-
call that throughout our experiments with the in-set setting
in Section 6, we inserted the adversarial document at the
Endof the prompt (closest document to the question).We
now consider two additional locations of the adversarial
document in the prompt: at theStartof the prompt (first
document in the prompt), and at aRandomposition in the
retrieved document list that is part of the prompt. As in
Section 6, we use the single-adversarial-document in-set set-Table 10.Performance comparison of SDAG-based defense meth-
ods with CARG, Discern&Answer (D&A), and RAGDefender
(RAGD) in the in-corpus single-adversarial-document setting, us-
ing Llama and E5. Boldface marks the best result for a k, dataset,
and evaluation measure; ’ c’, ’d’, and ’ r’ mark a statistically sig-
nificant difference between SDAG and CARG, D&A, RAGD,
respectively, for ak, dataset, and evaluation measure.
kDefense MethodHotpotQA TriviaQA NQ
ACC ASR ACC ASR ACC ASR
5CARG 0.16 0.67 0.57 0.45 0.36 0.32
D&A 0.18 0.62 0.62 0.370.370.31
RAGD 0.12 0.67 0.53 0.39 0.32 0.27
SDAG 0.20c
r0.52cd
r 0.62c
r0.27cd
r 0.36 0.22cd
r
SDAG-D&A 0.21c
r0.43cd
r 0.67cd
r 0.20cd
r 0.37r0.17cd
r
SDAG-RAGD 0.13 0.55cd
r 0.54 0.30cd
r 0.30 0.20cd
r
10CARG 0.18 0.61 0.61 0.410.390.36
D&A 0.20 0.58 0.66 0.340.390.33
RAGD 0.16 0.30 0.62 0.29 0.32 0.26
SDAG 0.23c
r0.29cd0.70c
r0.17cd
r 0.39r0.14cd
r
SDAG-D&A 0.23c
r0.28cd0.72cd
r 0.10cd
r 0.37r0.10cd
r
SDAG-RAGD 0.180.19cd
r 0.64 0.15cd
r 0.28 0.11cd
r
ting with the Random attack strategy. Due to computational
resources, we focus on the NQ dataset.
Table 11 presents the performance of SDAG and CARG
for the End, Start, and Random locations. We observe that
SDAG, in most relevant comparisons, attains higher accu-
racy and statistically significantly lower ASR than CARG
for all locations. In the few cases SDAG does not post the
best accuracy, its performance is statistically significantly
indistinguishable from that of CARG. These findings attest
to the robustness of SDAG’s performance with respect to
the positioning of the adversarial document in the prompt.
Furthermore, as can be seen in Table 11, there are no consis-
tent or substantial differences between the accuracy attained
by SDAG and CARG for the different locations of the ad-
versarial document.
It was shown in prior work that positioning of a document
containing the correct answer (to a question) next to the
question results in an improved accuracy of RAG systems
compared to the other positions (Cuconasu et al., 2024); as
we see in Table 11, this observation carries over to position-
ing of adversarial documents in terms of ASR, specifically,
to positioning of adversarial documents used for knowledge
poisoning attacks. In particular, in the vast majority of cases,
both SDAG and CARG attain higher ASR for the END lo-
cation than for the Start and Random locations. Notably,
the relative increase in ASR when moving from Random to
End, for both SDAG and CARG, is higher than the relative
increase in ASR when moving from Start to End. A possible
explanation could be the previously observed bias of LLMs
to the start and the end of the prompt (Liu et al., 2024). We
leave further research of this observation to future work.
17

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 11.Performance of SDAG and CARG on the NQ dataset, where the adversarial document is located in different positions in the
prompt: End, Start, and Random. Boldface marks the best result for a generator, retriever, k, location, and evaluation measure; ’∗’ marks a
statistically significant difference between SDAG and CARG for a generator, retriever,k, location, and evaluation measure.
RAG Configuration End Start Random
Generator RetrieverkSDAG CARG SDAG CARG SDAG CARG
ACC ASR ACC ASR ACC ASR ACC ASR ACC ASR ACC ASR
E550.33 0.27∗0.31 0.45 0.310.28∗0.320.420.31 0.27∗0.310.41
10 0.37 0.17∗0.35 0.40 0.360.13∗0.380.36 0.340.14∗0.360.36
Llama Contriever50.18 0.43∗0.17 0.570.17 0.45∗0.15 0.580.20 0.36∗0.19 0.52
10 0.25 0.27∗0.23 0.520.23 0.27∗0.22 0.530.24 0.23∗0.240.48
BM2550.19 0.32∗0.18 0.510.18 0.33∗0.180.500.19 0.33∗0.18 0.52
10 0.230.22∗0.240.460.25 0.19∗0.250.460.24 0.19∗0.240.46
E550.30 0.39∗0.29 0.540.33 0.32∗0.30 0.510.32 0.34∗0.30 0.47
10 0.290.28∗0.320.470.35 0.15∗0.32 0.48 0.310.23∗0.350.38
Qwen Contriever50.15 0.57∗0.12 0.670.17 0.54∗0.13 0.670.18 0.50∗0.16 0.59
10 0.20 0.47∗0.17 0.630.25∗0.34∗0.16 0.640.22 0.36∗0.21 0.53
BM2550.17 0.46∗0.14 0.600.19 0.41∗0.16 0.600.18 0.43∗0.16 0.57
100.23∗0.36∗0.17 0.560.27∗0.21∗0.20 0.560.24 0.29∗0.22 0.53
E550.36∗0.39∗0.30 0.550.41 0.33∗0.37 0.490.40 0.34∗0.36 0.50
100.40∗0.31∗0.33 0.480.43 0.23∗0.41 0.420.42 0.24∗0.38 0.45
Mistral Contriever50.23∗0.52∗0.16 0.630.25∗0.52∗0.20 0.610.26 0.48∗0.22 0.59
100.30∗0.44∗0.20 0.590.32∗0.39∗0.25 0.550.32∗0.34∗0.26 0.54
BM2550.24∗0.45∗0.19 0.610.28 0.37∗0.26 0.580.26 0.40∗0.24 0.58
100.29∗0.33∗0.21 0.580.32 0.27∗0.28 0.510.30∗0.29∗0.24 0.53
Table 12.Performance of SDAG and CARG for different attack
strategies on the NQ dataset, using Contriever and BM25 as the
retrievers, and k= 5 . Boldface marks the best result for a retriever,
generator, attack strategy, and evaluation measure; ’∗’ marks a
statistically significant difference between SDAG and CARG for a
retriever, generator, attack strategy, and evaluation measure.
Retriever Generator MethodRandom Near Far
ACC ASR ACC ASR ACC ASR
LlamaSDAG 0.23 0.28∗0.23 0.30∗0.23 0.30∗
CARG 0.20 0.52 0.19 0.52 0.21 0.49
Contriever QwenSDAG 0.21∗0.40∗0.20∗0.44∗0.19∗0.42∗
CARG 0.15 0.61 0.14 0.62 0.15 0.58
MistralSDAG 0.27∗0.39∗0.26∗0.41∗0.24∗0.41∗
CARG 0.19 0.62 0.18 0.63 0.20 0.60
LlamaSDAG 0.22 0.26∗0.20 0.29∗0.20 0.29∗
CARG 0.19 0.53 0.19 0.53 0.19 0.53
BM25 QwenSDAG 0.17 0.46∗0.19∗0.38∗0.21∗0.35∗
CARG 0.14 0.60 0.15 0.60 0.16 0.59
MistralSDAG 0.27∗0.33∗0.26∗0.36∗0.23∗0.33∗
CARG 0.19 0.61 0.18 0.62 0.17 0.52
F. Additional Results
We next present additional performance results and analysis
of SDAG in the in-set setting. We focus on the NQ dataset.
Attack Strategies.In Section 6.1 we evaluated SDAG us-
ing E5 as the retriever with the attack strategies defined in
Section 3.2:Random,Near, andFar. Here we provide per-
formance numbers when Contriever and BM25 are used asretrievers. As in Section 6.1, we use the single-adversarial-
document setting. The results are presented in Table 12.
Table 12 shows that, similar to the case in Section 6.1 where
we used E5, SDAG consistently achieves higher accuracy
and statistically significantly lower ASR than that of CARG
for all attack strategies. This finding attests to the robustness
of the effectiveness of SDAG over CARG to different strate-
gies of knowledge poisoning attacks and different retrievers.
SDAG’s performance under Multiple-Document Attack.
In Section 6.1 we presented SDAG performance in the mul-
tiple adversarial documents setting using Llama as the re-
triever. Here we provide performance numbers where Qwen
and Mistral are used as generators. We include the Llama re-
sults to allow comparison of SDAG’s performance between
generators. As in Section 6.1, we use E5 and the Random
attack strategy in the in-set setting. We insert either two or
three adversarial documents to the retrieved set. Note that
when we insert three adversarial documents and k= 5 , ad-
versarial documents constitute the majority of the retrieved
set; in this case, we do not expect the generator to produce
the correct answer, therefore, we exclude this setting from
our analysis.
We can see in Table 13 that in all cases, SDAG posts sta-
tistically significantly improved ASR compared to CARG.
In most cases SDAG achieves higher accuracy compared to
CARG; in the few cases where SDAG’s accuracy is lower,
it is statistically significantly indistinguishable from that of
CARG. These findings are in accordance with those reported
18

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 13.Performance of SDAG and CARG in the multiple-
adversarial-document setting on the NQ dataset, using E5. Bold-
face marks the best result for a #adv. docs, k, generator, and
evaluation measure; ’∗’ marks a statistically significant difference
between SDAG and CARG for a #adv. docs, k, generator, and
evaluation measure.
#adv.
docskGeneratorAccuracy ASR
SDAG CARG SDAG CARG
25Llama 0.230.25 0.39∗0.52
Qwen 0.230.21 0.50∗0.60
Mistral 0.33∗0.26 0.46∗0.63
10Llama 0.280.27 0.24∗0.47
Qwen 0.270.24 0.35∗0.54
Mistral 0.40∗0.29 0.35∗0.57
310Llama 0.20 0.20 0.43∗0.55
Qwen 0.210.19 0.53∗0.64
Mistral 0.34∗0.23 0.46∗0.63
in Section 6.1 for the Llama generator. Notably, when com-
paring the multiple-document setting results from Table 13
with those for the single-document setting in Table 1, we see
that, as could be expected, the attack effectiveness improves
in the multiple-document setting in terms of both (reduced)
accuracy and (increased) ASR.
G. Additional Analysis of the Spatial
Positioning of Adversarial Documents
In Section 6.2 we analyzed the effect of the spatial posi-
tioning of adversarial documents on the performance of
SDAG and CARG using the Llama generator; we used the
embedding space induced by the generator (i.e., we used
the generator’s embeddings). Here we provide additional
results using Qwen and Mistral as generators. We further
provide an analysis of the effect of the spatial positioning
of adversarial documents in the embedding space induced
by the retriever. We consider the two question sets defined
in Section 6.2:Distant Set(DS) andNear Set(NS). As
in Section 6.2, we focus on the NQ dataset and use the
Random attack strategy in the single-adversarial-document
in-set setting.
We begin with the embedding space induced by the gener-
ator, using Qwen and Mistral. Table 14 reports the Perfor-
mance of SDAG and CARG on the DS and NS question sets.
In all cases, SDAG attains better performance on DS than
on NS in terms of both accuracy and ASR; in most cases
the performance differences are statistically significantly
improved. CARG attains improved accuracy and on par
ASR on DS with respect to NS. We can also see in Table 14
that the relative effectiveness of SDAG compared to CARG
is more substantial when adversarial documents are geomet-
rically distant from the benign-document set than when they
are closer (i.e., DS vs. NS). The findings are in line withTable 14.Performance of SDAG and CARG on the NQ dataset
using Qwen and Mistral as generators, and k= 5 , when stratifying
questions into two sets: Distant Set (DS) and Near Set (NS). Bold-
face marks the best result for a generator, retriever, and evaluation
measure; ’∗’ marks a statistically significant difference between
the question sets for a generator, retriever, and evaluation measure,
using an unpaired t-test.
Generator RetrieverQuestion
SetAccuracy ASR
SDAG CARG SDAG CARG
E5DS 0.46∗0.41∗0.15∗0.43
NS 0.32 0.29 0.24 0.47
Qwen ContrieverDS 0.22 0.18 0.380.59
NS 0.18 0.14 0.430.57
BM25DS 0.29∗0.20 0.30∗0.62
NS 0.19 0.15 0.360.59
E5DS 0.48∗0.35 0.19∗0.55
NS 0.39 0.32 0.240.52
Mistral ContrieverDS 0.28 0.24 0.26∗0.57
NS 0.23 0.19 0.43 0.61
BM25DS 0.28∗0.20 0.280.54
NS 0.21 0.16 0.320.52
those in Section 6.2 (Table 4), which were for the Llama
generator.
We now turn to analyze the effect of the spatial positioning
in the embedding space induced by the retriever, rather
than the one induced by the generator. Table 15 reports
the Performance of SDAG and CARG on the DS and NS
question sets, using Llama as the generator. We observe
that in all, but one case, both SDAG and CARG attain better
performance on DS than on NS. This result indicates that
adversarial documents that are geometrically close to the
benign retrieved set yield more effective attacks also in the
embedding space induced by the retriever, and not only by
the generator as was the case in Section 6.2.
Notably, our findings from the analysis in the embedding
space induced by the retriever (Table 15) are consistent
with those in the embedding space induced by the generator
(Tables 4 and 14). In fact, we observe similar patterns in
RAG performance for both SDAG and CARG in terms of
both accuracy and ASR. We leave a deeper exploration of
this observation for future work.
H. Adversarial Document (Sub)Set Analysis
In Section 6.2 we showed that SDAG steers the genera-
tor to attend more frequently to dominant subsets of doc-
uments in the retrieved set that contain the correct answer
to a question. Here we study whether this finding holds
for subsets composed of adversarial documents. We use
the two types of (sub)sets defined in Section 6.2:Ground
Truth Set(GTS) andAdversarial Set(AS). We also use
the metricDominant-Set-based Generationof a set and the
dominant setdefinition from Section 6.2. We now turn
19

Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention
Table 15.Performance of SDAG and CARG on NQ dataset us-
ing Llama and k= 5 , when measuring distances in the retriever
embedding space and stratifying questions into two sets: Distant
Set (DS) and Near Set (NS). Boldface marks the best result for a
retriever and evaluation measure; ’∗’ marks a statistically signifi-
cant difference between question sets for a retriever and evaluation
measure, using an unpaired t-test.
RetrieverQuestion
SetAccuracy ASR
SDAG CARG SDAG CARG
E5DS 0.50∗0.41∗0.09∗0.32∗
NS 0.34 0.32 0.17 0.40
ContrieverDS 0.26 0.23 0.23∗0.47
NS 0.22 0.20 0.32 0.50
BM25DS 0.29∗0.25∗0.23 0.51
NS 0.18 0.16 0.270.51
Table 16.Dominant-set-based generation of AS for SDAG and
CARG, using E5 and k= 5 . Boldface marks the best result for a
generator and dataset; in all cases, SDAG and CARG are statisti-
cally significantly indistinguishable for a generator and dataset.
GeneratorHotpotQA TriviaQA NQ
SDAG CARG SDAG CARG SDAG CARG
Llama 0.840.820.900.840.600.55
Qwen 0.90 0.90 0.920.910.68 0.68
Mistral 0.90 0.900.910.94 0.67 0.67
to analyze the dominant-set-based generation of AS in the
multiple-adversarial-document in-corpus setting. Note that
in a single-document setting, the retrieved set includes one
adversarial document at most (i.e., the AS is never domi-
nant); therefore, we inject five adversarial documents into
the corpus. To ensure at least two sets for informative domi-
nant set analysis, we exclude questions where the retrieved
set consists only adversarial documents.
Table 16 reports the dominant-set-based generation of AS
for SDAG and CARG using E5. In contrast to the GTS
analysis presented in Section 6.2, we do not observe con-
sistent differences in dominant-set-based generation of AS
when comparing SDAG with CARG. In fact, in all cases,
the dominant-set-based generation of SDAG and CARG
is statistically significantly indistinguishable. This finding
indicates that SDAG’s more frequent generator steering to-
wards dominant GTS than CARG observed in Section 6.2
does not hold for subsets of adversarial documents.
I. Declaration of Generative AI Use
We used an LLM (OpenAI’s GPT-5)onlyfor linguistic re-
finement and grammar correction in the writeup and for
coding. We havenotused genAI tools for other purposes in
the work. Specifically, we wrote the paper and performed
all the research stages by ourselves: ideation, related work
coverage, model design, experimental setting design, evalu-
ation, analysis, conclusions, etc.
20