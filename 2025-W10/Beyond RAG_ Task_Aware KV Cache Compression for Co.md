# Beyond RAG: Task-Aware KV Cache Compression for Comprehensive Knowledge Reasoning

**Authors**: Giulio Corallo, Orion Weller, Fabio Petroni, Paolo Papotti

**Published**: 2025-03-06 21:07:41

**PDF URL**: [http://arxiv.org/pdf/2503.04973v1](http://arxiv.org/pdf/2503.04973v1)

## Abstract
Incorporating external knowledge in large language models (LLMs) enhances
their utility across diverse applications, but existing methods have
trade-offs. Retrieval-Augmented Generation (RAG) fetches evidence via
similarity search, but key information may fall outside top ranked results.
Long-context models can process multiple documents but are computationally
expensive and limited by context window size. Inspired by students condensing
study material for open-book exams, we propose task-aware key-value (KV) cache
compression, which compresses external knowledge in a zero- or few-shot setup.
This enables LLMs to reason efficiently over a compacted representation of all
relevant information. Experiments show our approach outperforms both RAG and
task-agnostic compression methods. On LongBench v2, it improves accuracy by up
to 7 absolute points over RAG with a 30x compression rate, while reducing
inference latency from 0.43s to 0.16s. A synthetic dataset highlights that RAG
performs well when sparse evidence suffices, whereas task-aware compression is
superior for broad knowledge tasks.

## Full Text


<!-- PDF content starts -->

Beyond RAG: Task-Aware KV Cache Compression
for Comprehensive Knowledge Reasoning
Giulio Corallo
SAP Labs,
EURECOM
giulio.corallo@sap.comOrion Weller
Johns Hopkins University
oweller@cs.jhu.eduFabio Petroni
Samaya AI
fabio@samaya.aiPaolo Papotti
EURECOM
papotti@eurecom.fr
Abstract
Incorporating external knowledge in large
language models (LLMs) enhances their
utility across diverse applications, but ex-
isting methods have trade-offs. Retrieval-
Augmented Generation (RAG) fetches evi-
dence via similarity search, but key infor-
mation may fall outside top ranked results.
Long-context models can process multiple
documents but are computationally expen-
sive and limited by context window size.
Inspired by students condensing study mate-
rial for open-book exams, we propose task-
aware key-value (KV) cache compression,
which compresses external knowledge in a
zero- or few-shot setup. This enables LLMs
to reason efficiently over a compacted rep-
resentation of all relevant information.
Experiments show our approach outper-
forms both RAG and task-agnostic com-
pression methods. On LongBench v2, it im-
proves accuracy by up to 7 absolute points
over RAG with a 30× compression rate,
while reducing inference latency from 0.43s
to 0.16s. A synthetic dataset highlights that
RAG performs well when sparse evidence
suffices, whereas task-aware compression is
superior for broad knowledge tasks.
1 Introduction
Incorporating external information into large lan-
guage models (LLMs) significantly enhances their
utility across various applications, enabling them
to generate more informed and accurate outputs.
Several methodologies have been devised to facil-
itate this integration, but each comes with limita-
tions that restrict its effectiveness.
Retrieval-Augmented Generation (RAG) is a
technique that enhances LLMs by leveraging ex-
ternal corpora to retrieve relevant chunks of in-
formation (Lewis et al., 2020). However, RAG
512 1024 2048 4096
Retrieval context length (token count)0102030405060Overall Score (%)
Overall Performance on 32k T oken Corpus
RAG KVCompress FS Full contextFigure 1: Overlap match of the words between ground
truth and predictions of various KV cache compres-
sion methods compared to RAG on a synthetic cor-
pus with 32k tokens. Our Few-Shot compression ap-
proach achieves results exceeding RAG when the con-
text length is much smaller than the corpus size.
is most effective in scenarios with narrow, fo-
cused queries that require a few pieces of evi-
dence. When dealing with broader queries that de-
mand synthesizing insights from multiple sources
across the corpus, retrieval mechanisms may fall
short (Barnett et al., 2024). This happens be-
cause retrieval typically relies on similarity-based
search, which may fail to capture implicit relation-
ships between distant pieces of evidence, making
it challenging to surface all relevant context and
often introducing noise or redundancy (Yu et al.,
2024). Researchers have begun to address these
issues, e.g., via improved chunking and pruning
strategies (Edge et al., 2024), but handling broad
queries with RAG remains challenging.
Recent advancements have extended LLMs’
ability to process longer contexts, pushing the
boundaries of how much information they can
handle simultaneously (Team et al., 2024; Li et al.,
2025a). This progress opens up the possibility
of processing entire corpora as input, offering
a compelling alternative to retrieval-based meth-arXiv:2503.04973v1  [cs.CL]  6 Mar 2025

A1What did Sam find amusing at the zoo?
Inference Q2
A2Sam found the penguins very amusing. Q1
When did they leave the zoo?
They left the zoo at 2:00 PM Q2
Ask anything … + …Compression Prompt
CONTEXT (C):
DOC 1 : Yesterday, the Thompson family visited the city zoo. 
They arrived at the zoo at 10:00 AM. [...] Her younger brother,
Sam, found the penguins very amusing as they waddled 
around. Later [...] They left the zoo at 2:00 PM, their day filled 
with happy memories.
DOC 2 : Atom form the basis of matter. […]
DOC N : […]
TASK (T) : Answer factual knowledge about the given text
FEW SHOT EXAMPLES (FS) :
Q1: What is the capital of France?
A1: Paris.
Q2: What is the chemical symbol for gold?
A2: Au.KV compression
guided by T+ FS
OFFLINE
128k tokenKV Cache
16k tokenKV Cache
Input
PROMPTENGINEERFINALUSER
LLMBEFORE
LLMAFTER
Multiple queries can be done ONLINEFigure 2: An illustration of our compression strategy that reduces the original context (C) from a KV cache of
128k tokens to 16k. This process is guided by task instructions (T) and few-shot examples (FS), condensing the
essential information needed for factual QA on the corpus documents. At inference time, the LLM can answer
multiple questions as if it had access to the entire (uncompressed) corpus.
ods. However, this approach comes with signif-
icant computational costs, as handling large in-
puts requires substantial memory resources, par-
ticularly on GPUs, which creates a scalability bot-
tleneck (Liu et al., 2023). Furthermore, as the
context grows, models often struggle to identify
the relevant pieces of information buried in all the
clutter (Liu et al., 2024).
To bridge the gap between massive corpora and
limited context windows, researchers are develop-
ing compression techniques that condense or filter
input text (Jha et al., 2024). Some of this strat-
egy focuses on optimizing the model’s key-value
(KV) cache, ensuring that essential information is
retained even within limited contexts. Existing ap-
proaches fall into two categories: query-agnostic
compression (Zhang et al., 2023; Devoto et al.,
2024; Xiao et al., 2023a; Feng et al., 2025), and
query-aware compression, which dynamically op-
timizes content based on the query during infer-
ence (Li et al., 2025b; Corallo and Papotti, 2024).
While the latter leads to highly relevant outputs,
it is computationally prohibitive, as it requires re-
compressing the input for each query, making it
impractical for real-world deployment.
In this study, we introduce a task-aware , query-
agnostic compressed cache, offering a balanced
trade-off between efficiency and relevance. In-
stead of recompressing the input for every query,
our approach precomputes a compressed cache
tailored to a broader task context. Our method
delivers performance that significantly surpasses
existing query-agnostic compression and closely
approaches the effectiveness of query-aware com-pression. Figure 1 shows the quality performance
of KV cache compression methods against RAG
on a 32k token corpus. With high compression
rates 64x and 32x (corpus compressed to 512 and
2048 tokens) our method outperforms RAG by
about 20 absolute points. With compression rates
between 16x and 8x, Few Shots performs on par or
better than fitting the original corpus in the model
context.
Figure 2 shows our compression strategy. The
task context can be defined through a succinct de-
scription (namely Zero Shot) or a small set of rep-
resentative examples in a Few-Shot setting. This
prompt produces a more compact representation
of the original context while preserving crucial de-
tails. Compression happens only once, creating
a representation that can be reused for any query
within the task domain. This eliminates the need
for repeated processing, streamlining inference by
bypassing real-time retrieval and prefilling. This
approach is not limited to QA but can be applied
to a wide range of tasks.
Experimental results across diverse tasks using
Llama 3.1, including the LongBench and Long-
Bench v2 benchmarks, demonstrate that our task-
aware compression method consistently outper-
forms both retrieval-based approaches and full-
context models in diverse evaluation settings, such
as Code Completion and Document QA. Further-
more, experiments on synthetic datasets highlight
the superior capability of our method in handling
broad, multifaceted queries. Notably, in scenar-
ios requiring the synthesis of widely distributed in-
formation, our approach significantly outperforms

RAG, establishing compression as a key enabler
for scaling LLM reasoning beyond retrieval-based
methods.
2 Background
LLMs built on the transformer architec-
ture (Vaswani et al., 2017) have become the
backbone of modern natural language processing.
Given a sequence of ntokens x∈Rn, each
transformer layer produces hidden representations
via a multi-head self-attention mechanism:
Attention( Q,K,V) = softmaxQK⊤
√dk
V,
where
Q=WQh,K=WKh,V=WVh,
withhrepresenting the hidden states (token em-
beddings) for the input sequence. The dimension
dkisd
Hwhere dis the hidden size and His the
number of attention heads.
In a knowledge-intensive task setup (Petroni
et al., 2020), many instruction-tuned LLMs orga-
nize their input as a context followed by a user
prompt. Formally, let xdenote a sequence of to-
kens. The input sequence can be expressed as:
x=h
x(cont),x(prompt )i
∈Rn(cont)+n(prompt ),(1)
Crucially, x(cont)serves as the knowledge the
model has access to when generating the final re-
sponse.
During inference, an LLM operates in two dis-
tinct phases:
Prefill Stage. The model processes the entire in-
put sequence xand caches the key-value (KV) ma-
trices for each layer:
K∈Rn×d,V∈Rn×d. (2)
Generation Stage. Tokens are generated autore-
gressively. For each new token yj, the model com-
putes:
qnew,knew,vnew∈Rd, (3)
and updates the cache as follows:
K←K
knew
,V←V
vnew
. (4)
Thanks to the cached KV matrices, self-attention
complexity reduces from O(n2d)toO(nd), sig-
nificantly improving efficiency. However, for
largen(cont), storing these matrices for every layer
can be prohibitively memory-intensive.2.1 KV-Cache Compression
To mitigate the memory load from very long con-
texts, a promising approach is KV-cache compres-
sion. Instead of retaining K,Vfor all ntokens,
one compresses them into smaller matrices:
eK∈Rk×dandeV∈Rk×dwith k≪n,
that still preserve the essential information needed
for generating the final response. Formally, one
seeks to minimize a divergence measure,
min
eK,eVh
dist 
y|K,V,y|eK,eVi
,
where yis the model’s output.
Previous work has also recognized that com-
pressing the KV cache of LMs allows for im-
proved performance at much smaller memory
costs (Qin et al., 2023; Ge et al., 2023; Corallo
and Papotti, 2024). In general there are three lev-
els of compression: (1) task-agnostic compres-
sion where no task information is present (Chari
et al.; Zhang et al., 2024a), (2) ad-hoc compres-
sion, where the compression is tailored for a sin-
gle specific task such as question-answering (Fu
et al., 2024; Cai et al., 2024), and (3) query-aware
compression where the compression happens w.r.t.
a specific example (Rehg, 2024; Xu et al., 2024).
Our work proposes a new compression technique
that works for any task specified in the prompt.
To explain how compression works, we detail
a query-aware iterative approach that compresses
the KV cache by retaining only the most relevant
Key-Value vectors for the query given at infer-
ence time (Corallo and Papotti, 2024). Let mbe
the chunk size, and let {c1,c2, . . .}be the seg-
ments obtained by slicing the context x(cont). The
segment-to-document ratio is defined as
s=n
m(5)
where nis the total document length. At itera-
tioni, the method takes as input
h
eKi−1,eVi−1|{z}
previous compressed cache,ci|{z}
current chunk,q|{z}
questioni
,
where eKi−1,eVi−1∈Rri−1×ddenote the com-
pressed cache from the previous iteration, ci∈
Rm×dis the chunk of context for the current itera-
tion, and q∈Rq×dis the question to be answered.

During the forward pass, the multi-head atten-
tion scores are computed. Crucially, the cross-
attention submatrix
W(q,c)∈Rq×(ri−1+m),
captures how each question token attends to both
the previous cache and the current chunk. The
online method then selects the top rtoken po-
sitions (according to the highest attention scores
inW(q,c)) to form eKi,eVi.After processing all
chunks, the final eK,eV∈Rk×dprovide a global
representation of the entire context, at a substan-
tially reduced size. Agnostic methods use similar
principles but in a single offline computation of the
cache, thus without making use of the query.
2.2 RAG and Knowledge Intensive Tasks
Knowledge-intensive tasks require models to uti-
lize external information to arrive at accurate an-
swers. This category encompasses tasks like
question-answering, summarization, and fact-
checking (Kwiatkowski et al., 2019; Petroni et al.,
2020). While larger models have demonstrated
improved capacity to store knowledge whitin their
parameters (Tirumala et al., 2022; Biderman et al.,
2023; Wang et al., 2024), they face limitations,
such as difficulties in updating or modifying the
embedded information (De Cao et al., 2021) and
hallucinations (Ji et al., 2023). To address this,
RAG integrates retrieval mechanisms with gen-
erative language models, enhancing the accu-
racy by explicitely incorporating external knowl-
edge (Lewis et al., 2020; Borgeaud et al., 2022;
Gao et al., 2023). However, RAG has its own chal-
lenges: crucial information might not be included
in the top-ranked retrieval results, preventing the
language model from reasoning over it.
3 Problem Formulation
A fundamental challenge in compressing long-
context representations is achieving query-
agnostic compression, where a precomputed
compressed cache eK,eVremains effective for
any query at inference time. However, empirical
results indicate that existing query-agnostic meth-
ods exhibit significant performance degradation,
particularly under high compression rates, often
falling behind both full-context processing and
RAG (NVIDIA, 2024).
On the other hand, query-aware compression,
circumvents the challenges of long-context mod-els by (i) processing documents in smaller seg-
ments while adjusting positional embeddings ac-
cordingly, and (ii) reducing the KV memory foot-
print in proportion to the compression rate. In
practice, query-aware compression can outper-
form retrieval-based methods like RAG. However,
a critical drawback is its reliance on a specific
user query at compression time. While effec-
tive for single-query scenarios, this assumption
becomes impractical when multiple queries need
to be processed, as rerunning the compressor for
each query is computationally prohibitive, thus
undermining the original goal of avoiding large-
scale retrieval or excessive context expansion.
This raises a key research question: Can we de-
sign a query-agnostic compression method that
preserves efficiency while maintaining qualita-
tive competitive performance?
4 Methodology
In this section, we present our task-aware, query-
agnostic compression strategy, motivated by the
remarkable in-context learning capabilities of
modern LLMs. We describe how we obtain a sin-
gle, reusable cache that covers an entire task’s ex-
ternal knowledge.
4.1 Motivation via In-Context Learning
LLMs demonstrate remarkable in-context learn-
ing capabilities (Brown et al., 2020; Dong et al.,
2022): once they read a sufficiently large prefix
of tokens, they often infer the nature of the task
without any parameter updates. In practice, tokens
later in the input are easier to predict because the
model has more context to condition on. An intu-
itive analogy is a student preparing for exams.
When asked a new question with no references
at hand (i.e., zero-shot inference ), both the human
student and the LLM rely solely on prior knowl-
edge. If a few solved examples are provided ( few-
shot learning ), they adapt by identifying solution
patterns from the examples, thereby reducing un-
certainty. Giving the student all available refer-
ence material (akin to letting an LLM observe the
entire corpus) can maximize accuracy. Our ob-
jective is to construct a compressed representa-
tionof the corpus in advance, much like a curated
“cheat sheet” that condenses the key information.
To explore how LLMs dynamically allocate atten-
tion within their input, we analyze cross-attention
patterns across different prompting strategies, as

(a) Last Token (b) Task (c) Task + Few -shots (d) Task + Few -shots + Q
Perplexity: 6.55 Perplexity: 6.13 Perplexity: 4.80 Perplexity: 4.66Figure 3: We examine how the model attends to context tokens when conditioned on the last token, a task descrip-
tion, a description with few-shot examples, and a description with both few-shot examples and a question. As we
increase the information in the prompt, the cross attention between the prompt and the context better discriminates
the context tokens that are relevant for decoding the answer. The perplexity is calculated on the loss for the answer.
shown in Figure 3. Notably, in (c), when a task
description and few examples are used, the model
commits to a subset of tokens in the latent space
that is similar to (d), the query-aware case, as ev-
idenced by the qualitative similarity in attention
distribution and the corresponding reduction in
perplexity on the final answer. This suggests that
a sufficiently structured prompt allows the model
to internally resolve much of the task-specific am-
biguity, even before an explicit question is in-
troduced. Guided by these insights, our goal is
to construct a task-aware, query-agnostic com-
pression strategy that enables efficient, reusable
caching of task-relevant knowledge.
4.2 Task-Aware, Query-Agnostic
Compression
Concretely, we create a compressed cache capa-
ble of supporting any query within a defined task
domain. The procedure is:
(1) Define a Task Description ( T).Rather than
targeting a single query, we incorporate a Task De-
scription specifying the type of questions to be an-
swered (e.g., “Answer factual questions about this
corpus”). When available, we add a few examples
to better illustrate the target task.
(2) Offline Compression. We adapt an exist-
ing query-aware method by replacing the query
with the Task Description T(Corallo and Pa-
potti, 2024). The corpus is split into chunks
{c1,c2, . . .}, and we iteratively compress these
chunks together with T:
eKi−1,eVi−1,ci,T
.
This yields a compressed cache (eK,eV)that cap-
tures essential domain knowledge. Crucially, this
compression is computed only once.(3) Efficient Query Resolution. When fac-
ing a new query qnewfrom the same domain, we
prepend the precomputed cache:
x(prompt )=eK,eV,qnew
.
No further compression or retrieval is necessary.
The LLM conditions on eKandeVto answer,
reusing the relevant external knowledge.
We develop two variants of this approach:
KVCompress ZS (Zero-Shot Task Description)
uses only broad task instructions, and KVCom-
press FS (Few-Shot Task Description) includes
sample task examples (such as QA pairs). Em-
pirically, this offline, query-agnostic compres-
sion speeds up inference while offering more
robust coverage of multi-hop or broad queries
than RAG’s similarity-based lookups. In contexts
where RAG struggles to gather widely dispersed
evidence or gets confused by near-identical en-
tity names, the global coverage of the compressed
cache avoids such pitfalls.
5 Experimental Setup
5.1 Models
Our experiments use L LAMA -3.1-8B-
INSTRUCT (Dubey et al., 2024) with 5 knowledge-
infusion baselines: RAG, KVC OMPRESS ZS,
KVC OMPRESS FS, KVC OMPRESS FS+Q, and a
Full Context upper bound. For the compression
baselines, we set s= 2 in all experiments, as
defined in Equation (5).
For RAG, we use BGE-LARGE -EN-V1.5 (Xiao
et al., 2023b) as the retriever, filling the entire
context with the top- kretrieved documents, each
containing 256 tokens. The same prompt is used
across all models.1
1Detailed prompt configurations for each

5.2 Datasets
We evaluate these methods on three datasets:
LongBench v2 (Bai et al., 2024) is a multiple-
choice benchmark designed to evaluate the reason-
ing capabilities of LLMs on realistic long-context
multitasks. It comprises questions distributed
across six major categories: single-document QA,
multi-document QA, long in-context learning,
long-dialogue history understanding, code repos-
itory understanding, and long structured data un-
derstanding. Context lengths vary significantly,
from 8,000 to 2 million words. State-of-the-
art models, without enhanced reasoning prompts,
attained only 50.1% accuracy. In our experi-
ments, we evaluate using Exact Match scores.
The ground truth for evaluation consists of se-
lecting one option among A, B, C, or D. For the
full context evaluation, we use the entire con-
text, truncating to keep the first and last halves
if it exceeds the context window length, and con-
strained generation over the provided options with
1 max_new_tokens (De Cao et al., 2020).
LongBench (Bai et al., 2023) is a benchmark
also designed for long-context understanding. It
covers 21 datasets across six tasks, including
single-document and multi-document QA, sum-
marization, code completion, few-shot learning,
and a synthetic task. The benchmark has an av-
erage context length of 6,711 words for English
tasks. LongBench uses automated evaluation met-
rics, such as F1 for question answering tasks (Ra-
jpurkar et al., 2016) and ROUGE-L for summa-
rization tasks (Lin, 2004), and Edit Similarity for
the Code Completion task (Svyatkovskiy et al.,
2020).
Our Synthetic Dataset is designed to enable
full control over the interconnectivity among text
chunks in the corpus for a QA task. We use two
query types: direct retrieval , which retrieval sys-
tems can handle easily, and join-like queries , re-
quiring broader comprehension of the entire cor-
pus. Ground truth answers are generated as lists
of entities so that model evaluation is straightfor-
ward: predictions and ground truths are normal-
ized (removing punctuation and converting to low-
ercase) and a score is obtained by computing word
dataset are available in our configuration files:
https://anonymous.4open.science/r/
context-compression-2-6B58/conf/custom_
datasets/ .
Figure 4: Overview of our synthetic dataset. In this
example, the connectivity level is set to 2.
Figure 5: Overview of our questions. In this example,
the connectivity level is set to 2.
overlap between the predicted and ground truth
entities. We describe the dataset creation next.
6 Synthetic Dataset Construction
We construct a synthetic dataset designed to pre-
cisely control corpus complexity and the connec-
tivity level between text chunks. By varying inter-
chunk connectivity, we are able to thoroughly
evaluate different methods, identifying the exact
scenarios where each technique performs well or
fails. Figure 4 illustrates the structured design of
our corpus. Our dataset will be publicly released
to support future research.

6.1 Structured Entities and Corpus Chunks
We define three entity types.
People. Each person is described through
template-structured biographies containing at-
tributes such as name ,age,occupation ,city, and
hobbies . To maintain uniformity and facilitate
controlled experiments, each biography text chunk
is standardized to a length of 256 tokens using ad-
ditional neutral filler text.
Projects. Each project has attributes including ti-
tle,domain ,sponsor ,year started , and a descrip-
tive summary. Like for people, each text chunk is
standardized to 256 tokens.
Memberships. A membership represents the rela-
tionship between people and projects and specifies
therole(e.g., Engineer ,Manager ) and department
(e.g., R&D ,Marketing ) that a person holds in a
project. These text chunks similarly include filler
text to meet the fixed-length criterion.
We generate multiple corpus instances with
varying connectivity levels , ranging from 1 to 8,
where level kmeans each person links to exactly
kprojects. Higher connectivity increases dataset
complexity by distributing relevant information
about a person across multiple membership and
project chunks, thus challenging the model’s abil-
ity to synthesize scattered information. Each cor-
pus at a given connectivity level comprises exactly
32k tokens, ensuring consistent corpus size across
experiments.
6.2 Controlled Question Types for Evaluation
To rigorously evaluate the performance of both
KV-cache compression and retrieval-based meth-
ods, we generate two primary question categories
(Figure 5).
Direct Retrieval Questions. These questions
require information localized within a single
Memberships chunk. Example templates include:
•Which projects does pname belong to?
•Which role does pname have in
projtitle ?
•Which department is pname part of?
Where variables like pname are instantiated at
generation time. Answering these queries does not
require cross-chunk synthesis.
Join-like Questions. Answering these queries
require combining information across multiple
Memberships andProjects chunks. For example:•What are pname ’s project domains?
•In which years did pname ’s projects begin?
•Who sponsors pname ’s projects?
Addressing these queries tests a model’s capa-
bility for multi-hop reasoning and synthesis across
distributed knowledge sources. As the connectiv-
ity level grows, these join-like questions become
increasingly complex, requiring the aggregation of
information from multiple chunks.
For each connectivity level (1 through 8), we
generate 50 distinct queries: 25 direct-retrieval
and 25 join-like, totaling 400 distinct queries
across all connectivity levels. Additionally, we
create a targeted variant of direct retrieval ques-
tions with highly similar entity names (e.g. Per-
son_01, Person_02, . . . ) to test embedding-based
retrieval robustness against closely related entities.
7 Results and Discussions
We structure our analysis around five research
questions over our synthetic dataset experiments
while also drawing connections to the LongBench
and LongBench v2 results.
When Does RAG Fail? RAG exhibits signifi-
cant limitations in multi-hop reasoning and high-
connectivity scenarios. Our synthetic dataset re-
veals a sharp performance decline for join-like
questions — those requiring the integration of dis-
persed information (Figure 6). This degradation
stems from RAG’s reliance on similarity-based re-
trieval and frequently omits crucial chunks con-
taining necessary details, thus limiting the model’s
ability to accurately answer these queries.
Additionally, RAG struggles with entity disam-
biguation, particularly for similarly named enti-
ties (Figure 7). Embedding-based retrieval fre-
quently misidentifies relevant passages and re-
trieves irrelevant chunks due to embedding prox-
imity among similarly named entities. These lim-
itations are further reflected in the results for the
LongBench v2 “hard questions” (Figure 8), where
KVC OMPRESS ZS already outperforms RAG, de-
spite the absence of few-shot examples. RAG
gains only marginal improvements with longer
contexts, highlighting the limitations of retrieval-
based methods in scenarios that require complex
long-context understanding.
In LongBench (Figure 9), KVC OMPRESS ZS
consistently surpasses RAG in summarization and
code completion tasks, which require synthesizing
multiple passages or maintaining a broader con-

1 2 3 4 5 6 7 8
Connectivity Level020406080Score (%)
Retrieval context length: 512
1 2 3 4 5 6 7 8
Connectivity Level
Retrieval context length: 1024
1 2 3 4 5 6 7 8
Connectivity Level
Retrieval context length: 2048
1 2 3 4 5 6 7 8
Connectivity Level
Retrieval context length: 4096Performance on Join-like Questions on 32k T oken Corpus
RAG KVCompress FS+Q KVCompress FS Evidence Coverage LimitFigure 6: Performance by increasing target context length (64x to 8x compression rate) and connectivity level for
Join-like questions in the synthetic dataset. The dashed line indicates for which connectivity level RAG gets the
needed chunk for a given context length.
256 512 1024 2048
Retrieval context length (token count)1530456075Overall Score (%)
Overall Performance on Similar-Named-Person Questions
RAG KVCompress FS+Q KVCompress FS Full context
Figure 7: Performance by retrieval context length size
(128x to 16x compression rate) for Direct Retrieval
questions with highly similar entity names in the syn-
thetic dataset.
text and is competitive in the other tasks, except
the QA, where it is the Few Shot setting that com-
petes with RAG.
Takeaway: RAG fails in multi-hop reasoning
and entity disambiguation, limiting its effective-
ness in synthesis tasks with long-context.
When Is RAG Effective? Experiments show that
RAG performs well when answers are localized
within a single document chunk. In direct-retrieval
scenarios, where each answer is self-contained,
RAG’s proves effective. This trend is evident in
our synthetic dataset (Figure 10) and in Long-
Bench results (Figure 9), where RAG achieves
competitive performance on tasks such as direct
question answering.
Takeaway: RAG excels when answers are self-
contained within the retrieved chunks, making
it effective for narrowly scoped questions.
2048 4096 8192 16384
Retrieval context length (token count)252627282930Exact Match (%)
Longbench-v2 Hard Questions on up to 128k T oken Corpus
RAG KVCompress ZS+Q KVCompress ZS Full contextFigure 8: Performance by retrieval context length size
(64x to 8x compression rate) for the Hard Questions in
LongBench v2.
What Are the Benefits of Our Task-Aware,
Query-Agnostic Compression Method? Our
proposed method departs from traditional re-
trieval and query-aware compression approaches
by precomputing a unified, task-aware com-
pressed cache that is reusable across multiple
queries. This approach offers several advantages
over both full-context processing and RAG: it dra-
matically reduces memory overhead and inference
time while preserving a global representation of
the corpus. Moreover, it overcomes RAG’s limi-
tations in broad query scenarios as shown in Fig-
ure 6 and in the LongBench v2 results both in Fig-
ure 9 and in Table 1.
Besides RAG, we evaluate our task-aware
compression strategies against several baseline
compression methods, including Expected Atten-
tion (NVIDIA, 2024), StreamingLLM (Xiao et al.,
2023a), and SnapKV (question-agnostic) (Li et al.,
2025b). As shown in Figure 11, our synthetic
dataset — configured with a high-connectivity
level of 8 to test multi-hop reasoning — reveals a
critical gap in the effectiveness of these compres-
sion techniques. These methods, which treat the

512 1000 2000
Retrieval context length20212223Edit Sim
Code Completion
512 1000 2000
Retrieval context length0.180.20Rouge-L
Summarization
512 1000 2000
Retrieval context length203040F1
Multi Document QA
512 1000 2000
Retrieval context length35404550Aggregated Score
Few-shot Learning
512 1000 2000
Retrieval context length404550Accuracy (EM)
Synthetic T ask
512 1000 2000
Retrieval context length25303540F1
Single Document QA
KVCompress ZS KVCompress ZS+Q KVCompress FS KVCompress FS+Q RAG Full contextFigure 9: Performance results on Longbench. Our FS variant is reported when examples are available (QA tasks).
For the QA tasks, the examples used in KVCompress FSt are taken (and removed) at random from the test set.
1 2 3 4 5 6 7 8
Connectivity Level20406080Score (%)
Retrieval context length: 256
RAG
KVCompress FS+Q
KVCompress FS
Evidence Coverage Limit
Figure 10: Performance by connectivity level (64x
compression rate) for the Direct Retrieval questions in
the synthetic dataset.
context in isolation and are “blind” to the specific
requirements of the query, consistently underper-
form. Moreover, the results underscore the util-
ity and sensitivity of our synthetic dataset, which
effectively captures nuances in method perfor-
mance, clearly differentiating between naive and
more sophisticated compression techniques. This
dataset serves as a robust benchmarking tool for
future research into KV-cache compression meth-
ods, particularly for evaluating their comprehen-
sive knowledge reasoning capabilities in challeng-
ing scenarios.
Takeaway: Task-aware query-agnostic com-
pression is scalable and efficient achieving bet-
ter performance when RAG fails.
How to Choose the Compression Variant? A
key decision is whether to use the zero-shot (ZS)
or few-shot (FS) variant. Our results suggest:
KVC OMPRESS ZS relies solely on a broad taskDataset (Metric) R. Length KVCompress ZS RAG
GovReport
(Rouge-L ↑)Full context 24.63
512 21.03 19.14
1000 21.57 19.92
2000 22.58 21.59
MultiNews
(Rouge-L ↑)Full context 18.45
512 16.79 9.35
1000 17.86 13.10
2000 18.42 17.39
PassageCount
(Accuracy ↑)Full context 3.25
512 5.77 1.50
1000 5.50 2.50
2000 5.00 4.00
LCC
(Edit Sim ↑)Full context 20.13
512 16.62 19.36
1000 19.35 17.89
2000 20.66 18.61
Table 1: Performance of KVCompress ZS and RAG
across all query-agnostic datasets in LongBench.
instruction—is optimal for query-agnostic tasks,
as shown in Table 1. In LongBench (Figure 9),
for non-QA tasks, it performs comparably or bet-
ter than RAG; in hard settings like LongBench v2
(Figure 8) it outperforms RAG.
KVC OMPRESS FS incorporates a few ex-
amples and excels in tasks such as question-
answering. As shown in Figures 1 and 9, FS sig-
nificantly reduces the performance gap with RAG,
whereas ZS struggles in these scenarios.
Takeaway: ZS is better suited for tasks that are
natively query-agnostic and FS is better when
dealing with examples such as in QA tasks.

512 1024 2048 4096
Retrieval context length015304560Score (%)
Comparison of Knowledge Infusion methods on 32k tokens corpus
Full context
KVCompress FSKVCompress FS+Q
RAGKVCompress ZS
Expected AttentionStreamingLLM
SnapKVFigure 11: Comparison against query-agnostic com-
pressors on our synthetic dataset for Join-like queries
with connectivity level=8 across decreasing compres-
sion levels (64x to 8x).
16k 32k 64k 128k
Corpus Length02468101214Time to First T oken (s)
Time T o First T oken (TTFT) Benchmark
Methods
Full context (Flash Attention 2)
KVCompress FS+Q (s=1)
KVCompress FS+Q (s=2)
KVCompress FS+Q (s=4)
KVCompress FS
RAG
Figure 12: Time to first token with increasing corpus
length (context length=8192 and question length=512).
What Are the Benefits of Integrating Query-
Aware Compression Signals? Integrating query-
aware signals (as in KVCompress FS+Q) yields
substantial performance gains, particularly in sce-
narios where computation time is less critical,
e.g., test-time settings like Deepseek r1 or Ope-
nAI o1 (Guo et al., 2025). By embedding explicit
query cues during compression, the model better
prioritizes critical information, as shown by im-
proved accuracy across all tasks and datasets.
Takeaway: Query-aware compression boosts
accuracy at the cost of an increase in processing
time. Query-agnostic methods like our proposal
are the fastest.
All results and takeaways are consistent and
held true when we applied the same methods
and datasets to Q WEN 2.5-7B-I NSTRUCT (Yang
et al., 2024), with its context window extended to
128k using the Yarn technique (Peng et al., 2023),
thereby reinforcing the robustness and generaliz-
ability of our approach across LLMs.8 Efficiency
Figure 12 reports inference efficiency for increas-
ing corpus size from 16k to 128k tokens, retrieval
context size fixed at 8k tokens, prompt length at
512 tokens, and chunk lengths determined by di-
viding the corpus length by the number of slices
(s) as defined in Equation (5).
For large corpus lengths, KVC OMPRESS FS+Q
shows better inference speed compared to process-
ing the entire context with a Flash Attention 2
implementation (Dao, 2023). As expected, KV-
COMPRESS FS exhibits the lowest inference la-
tency across all corpus lengths, being up to 2x
faster than RAG. This superior performance arises
from the offline precomputation of the KV cache
in KVC OMPRESS FS, which enable the model to
bypass tokenization and prefill at inference time.
9 Conclusion and Future work
In this paper, we presented a task-aware context
compression approach that enhances the ability
of LLMs to consume large corpora by efficiently
populating the KV cache with condensed contex-
tual representations. Our method enables multi-
step reasoning while mitigating the limitations of
traditional retrieval mechanisms. Empirical results
show that our approach outperforms RAG base-
lines, reduces infrastructure overhead, and im-
proves inference latency. By distilling raw text
into compact representations offline, our method
establishes a new paradigm for corpus-level rea-
soning, addressing critical constraints in existing
long-context models.
Future directions include integrating head-wise
and layer-wise compression strategies, leveraging
prior findings that certain heads and layers con-
tribute less to model performance and can be se-
lectively compressed in favor of more informative
ones (Feng et al., 2024; Zhang et al., 2024b). Ad-
ditionally, our results highlight a complementary
strength between KV compression and RAG: KV
compression excels in broad queries, while RAG
is more effective for narrow queries. This raises
the question of whether a hybrid approach could
further enhance retrieval—compressing the cor-
pus offline for global coverage while dynamically
fetching top-K chunks online to better address nar-
row queries. Exploring these strategies could un-
lock new efficiencies in long-context processing.

References
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du,
Xiao Liu, Aohan Zeng, Lei Hou, et al. 2023.
Longbench: A bilingual, multitask benchmark
for long context understanding. arXiv preprint
arXiv:2308.14508 .
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng,
Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng
Xu, Lei Hou, Yuxiao Dong, et al. 2024. Long-
bench v2: Towards deeper understanding and
reasoning on realistic long-context multitasks.
arXiv preprint arXiv:2412.15204 .
Scott Barnett, Stefanus Kurniawan, Srikanth
Thudumu, Zach Brannelly, and Mohamed Ab-
delrazek. 2024. Seven failure points when en-
gineering a retrieval augmented generation sys-
tem. In Proceedings of the IEEE/ACM 3rd
International Conference on AI Engineering-
Software Engineering for AI , pages 194–199.
Stella Biderman, Usvsn Prashanth, Lintang
Sutawika, Hailey Schoelkopf, Quentin An-
thony, Shivanshu Purohit, and Edward Raff.
2023. Emergent and predictable memorization
in large language models. Advances in Neu-
ral Information Processing Systems , 36:28072–
28090.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Mil-
lican, George Bm Van Den Driessche, Jean-
Baptiste Lespiau, Bogdan Damoc, Aidan Clark,
et al. 2022. Improving language models by re-
trieving from trillions of tokens. In Interna-
tional conference on machine learning , pages
2206–2240. PMLR.
Tom Brown, Benjamin Mann, Nick Ryder,
Melanie Subbiah, Jared D Kaplan, Prafulla
Dhariwal, Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell, et al. 2020.
Language models are few-shot learners. Ad-
vances in neural information processing sys-
tems, 33:1877–1901.
Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang
Liu, Tianyu Liu, Keming Lu, Wayne Xiong,
Yue Dong, Baobao Chang, Junjie Hu, et al.
2024. Pyramidkv: Dynamic kv cache compres-
sion based on pyramidal information funneling.
arXiv preprint arXiv:2406.02069 .Vivek Chari, Guanghui Qin, and Benjamin
Van Durme. Kv-distill: Nearly lossless context
compression for transformers.
Giulio Corallo and Paolo Papotti. 2024. Finch:
Prompt-guided key-value cache compression
for large language models. Transactions of
the Association for Computational Linguistics ,
12:1517–1532.
Tri Dao. 2023. Flashattention-2: Faster attention
with better parallelism and work partitioning.
arXiv preprint arXiv:2307.08691 .
N De Cao, W Aziz, and I Titov. 2021. Editing fac-
tual knowledge in language models. In EMNLP
2021-2021 Conference on Empirical Methods
in Natural Language Processing, Proceedings ,
pages 6491–6506.
Nicola De Cao, Gautier Izacard, Sebastian Riedel,
and Fabio Petroni. 2020. Autoregressive entity
retrieval. arXiv preprint arXiv:2010.00904 .
Alessio Devoto, Yu Zhao, Simone Scardapane,
and Pasquale Minervini. 2024. A sim-
ple and effective l_2norm-based strategy
for kv cache compression. arXiv preprint
arXiv:2406.11430 .
Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng,
Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing
Xu, and Zhifang Sui. 2022. A Survey on in-
Context Learning. arXiv:2301.00234 .
Abhimanyu Dubey, Abhinav Jauhri, Abhinav
Pandey, Abhishek Kadian, Ahmad Al-Dahle,
Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Amy Yang, Angela Fan, et al. 2024.
The llama 3 herd of models. arXiv preprint
arXiv:2407.21783 .
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven
Truitt, and Jonathan Larson. 2024. From
local to global: A graph rag approach to
query-focused summarization. arXiv preprint
arXiv:2404.16130 .
Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie,
and S Kevin Zhou. 2024. Ada-kv: Optimiz-
ing kv cache eviction by adaptive budget allo-
cation for efficient llm inference. arXiv preprint
arXiv:2407.11550 .

Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and
S Kevin Zhou. 2025. Identify critical kv cache
in llm inference from an output perturbation
perspective. arXiv preprint arXiv:2502.03805 .
Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong,
Yue Dong, and Wen Xiao. 2024. Not all heads
matter: A head-level kv cache compression
method with integrated retrieval and reasoning.
arXiv preprint arXiv:2410.19258 .
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxi-
ang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Haofen Wang, and Haofen Wang. 2023.
Retrieval-augmented generation for large lan-
guage models: A survey. arXiv preprint
arXiv:2312.10997 , 2.
Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia
Zhang, Jiawei Han, and Jianfeng Gao. 2023.
Model tells you what to discard: Adaptive kv
cache compression for llms. arXiv preprint
arXiv:2310.01801 .
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu,
Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025.
Deepseek-r1: Incentivizing reasoning capabil-
ity in llms via reinforcement learning. arXiv
preprint arXiv:2501.12948 .
Siddharth Jha, Lutfi Eren Erdogan, Sehoon
Kim, Kurt Keutzer, and Amir Gholami. 2024.
Characterizing prompt compression methods
for long context inference. arXiv preprint
arXiv:2407.08892 .
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu,
Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, An-
drea Madotto, and Pascale Fung. 2023. Survey
of hallucination in natural language generation.
ACM computing surveys , 55(12):1–38.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia
Redfield, Michael Collins, Ankur Parikh, Chris
Alberti, Danielle Epstein, Illia Polosukhin, Ja-
cob Devlin, Kenton Lee, et al. 2019. Natural
questions: a benchmark for question answering
research. Transactions of the Association for
Computational Linguistics , 7:453–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tauYih, Tim Rocktäschel, et al. 2020. Retrieval-
augmented generation for knowledge-intensive
nlp tasks. Advances in neural information pro-
cessing systems , 33:9459–9474.
Aonian Li, Bangwei Gong, Bo Yang, Boji
Shan, Chang Liu, Cheng Zhu, Chunhao Zhang,
Congchao Guo, Da Chen, Dong Li, et al.
2025a. Minimax-01: Scaling foundation mod-
els with lightning attention. arXiv preprint
arXiv:2501.08313 .
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat
Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle
Cai, Patrick Lewis, and Deming Chen. 2025b.
Snapkv: Llm knows what you are looking for
before generation. Advances in Neural Infor-
mation Processing Systems , 37:22947–22970.
Chin-Yew Lin. 2004. ROUGE: A package for
automatic evaluation of summaries. In Text
Summarization Branches Out , pages 74–81,
Barcelona, Spain. Association for Computa-
tional Linguistics.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin
Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. 2024. Lost in the mid-
dle: How language models use long contexts.
Transactions of the Association for Computa-
tional Linguistics , 12:157–173.
Zichang Liu, Aditya Desai, Fangshuo Liao,
Weitao Wang, Victor Xie, Zhaozhuo Xu, Anas-
tasios Kyrillidis, and Anshumali Shrivastava.
2023. Scissorhands: Exploiting the persistence
of importance hypothesis for LLM KV cache
compression at test time. In Thirty-seventh
Conference on Neural Information Processing
Systems .
NVIDIA. 2024. Llm kv cache compression made
easy.
Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and
Enrico Shippole. 2023. Yarn: Efficient con-
text window extension of large language mod-
els.arXiv preprint arXiv:2309.00071 .
Fabio Petroni, Aleksandra Piktus, Angela Fan,
Patrick Lewis, Majid Yazdani, Nicola De Cao,
James Thorne, Yacine Jernite, Vladimir
Karpukhin, Jean Maillard, et al. 2020. Kilt: a
benchmark for knowledge intensive language
tasks. arXiv preprint arXiv:2009.02252 .

Guanghui Qin, Corby Rosset, Ethan C Chau,
Nikhil Rao, and Benjamin Van Durme.
2023. Dodo: Dynamic contextual compres-
sion for decoder-only lms. arXiv preprint
arXiv:2310.02409 .
Pranav Rajpurkar, Jian Zhang, Konstantin Lopy-
rev, and Percy Liang. 2016. Squad: 100,000+
questions for machine comprehension of text.
arXiv preprint arXiv:1606.05250 .
Isaac Rehg. 2024. Kv-compress: Paged kv-
cache compression with variable compression
rates per attention head. arXiv preprint
arXiv:2410.00161 .
Alexey Svyatkovskiy, Shao Kun Deng, Shengyu
Fu, and Neel Sundaresan. 2020. Intellicode
compose: Code generation using transformer.
InProceedings of the 28th ACM joint meeting
on European software engineering conference
and symposium on the foundations of software
engineering , pages 1433–1443.
Gemini Team, Petko Georgiev, Ving Ian Lei,
Ryan Burnell, Libin Bai, Anmol Gulati, Garrett
Tanzer, Damien Vincent, Zhufeng Pan, Shibo
Wang, et al. 2024. Gemini 1.5: Unlocking mul-
timodal understanding across millions of tokens
of context. arXiv preprint arXiv:2403.05530 .
Kushal Tirumala, Aram Markosyan, Luke Zettle-
moyer, and Armen Aghajanyan. 2022. Mem-
orization without overfitting: Analyzing the
training dynamics of large language models.
Advances in Neural Information Processing
Systems , 35:38274–38290.
Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin. 2017. At-
tention is all you need. In Advances in Neu-
ral Information Processing Systems , volume 30.
Curran Associates, Inc.
Xinyi Wang, Antonis Antoniades, Yanai Elazar,
Alfonso Amayuelas, Alon Albalak, Kexun
Zhang, and William Yang Wang. 2024. Gen-
eralization vs memorization: Tracing language
models’ capabilities back to pretraining data.
arXiv preprint arXiv:2407.14985 .
Guangxuan Xiao, Yuandong Tian, Beidi Chen,
Song Han, and Mike Lewis. 2023a. Efficientstreaming language models with attention sinks.
arXiv preprint arXiv:2309.17453 .
Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas
Muennighoff. 2023b. C-pack: Packaged re-
sources to advance general chinese embedding.
Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang,
Xudong Lu, Aojun Zhou, Amrita Saha, Caim-
ing Xiong, and Doyen Sahoo. 2024. Think:
Thinner key cache by query-driven pruning.
arXiv preprint arXiv:2407.21018 .
An Yang, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Day-
iheng Liu, Fei Huang, Haoran Wei, et al. 2024.
Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115 .
Shuo Yu, Mingyue Cheng, Jiqian Yang, and Jie
Ouyang. 2024. A knowledge-centric bench-
marking framework and empirical study for
retrieval-augmented generation. arXiv preprint
arXiv:2409.13694 .
Rongzhi Zhang, Kuang Wang, Liyuan Liu, Shuo-
hang Wang, Hao Cheng, Chao Zhang, and
Yelong Shen. 2024a. Lorc: Low-rank com-
pression for llms kv cache with a progres-
sive compression strategy. arXiv preprint
arXiv:2410.03111 .
Xuan Zhang, Cunxiao Du, Chao Du, Tianyu Pang,
Wei Gao, and Min Lin. 2024b. Simlayerkv: A
simple framework for layer-level kv cache re-
duction. arXiv preprint arXiv:2410.13846 .
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tian-
long Chen, Lianmin Zheng, Ruisi Cai, Zhao
Song, Yuandong Tian, Christopher Ré, Clark
Barrett, et al. 2023. H2o: Heavy-hitter oracle
for efficient generative inference of large lan-
guage models. Advances in Neural Information
Processing Systems , 36:34661–34710.