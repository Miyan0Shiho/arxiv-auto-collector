# Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps

**Authors**: Jie Ou, Jinyu Guo, Shuaihong Jiang, Zhaokun Wang, Libo Qin, Shunyu Yao, Wenhong Tian

**Published**: 2025-05-19 05:39:38

**PDF URL**: [http://arxiv.org/pdf/2505.12731v1](http://arxiv.org/pdf/2505.12731v1)

## Abstract
Retrieval-augmented generation (RAG) has emerged as a pivotal method for
expanding the knowledge of large language models. To handle complex queries
more effectively, researchers developed Adaptive-RAG (A-RAG) to enhance the
generated quality through multiple interactions with external knowledge bases.
Despite its effectiveness, A-RAG exacerbates the pre-existing efficiency
challenges inherent in RAG, which are attributable to its reliance on multiple
iterations of generation. Existing A-RAG approaches process all retrieved
contents from scratch. However, they ignore the situation where there is a
significant overlap in the content of the retrieval results across rounds. The
overlapping content is redundantly represented, which leads to a large
proportion of repeated computations, thus affecting the overall efficiency. To
address this issue, this paper introduces a model-agnostic approach that can be
generally applied to A-RAG methods, which is dedicated to reducing the
redundant representation process caused by the overlapping of retrieval
results. Specifically, we use cache access and parallel generation to speed up
the prefilling and decoding stages respectively. Additionally, we also propose
an instruction-driven module to further guide the model to more effectively
attend to each part of the content in a more suitable way for LLMs. Experiments
show that our approach achieves 2.79 and 2.33 times significant acceleration on
average for prefilling and decoding respectively while maintaining equal
generation quality.

## Full Text


<!-- PDF content starts -->

arXiv:2505.12731v1  [cs.AI]  19 May 2025Accelerating Adaptive Retrieval Augmented Generation via
Instruction-Driven Representation Reduction of Retrieval Overlaps
Jie Ou1, Jinyu Guo1,∗, Shuaihong Jiang1, Zhaokun Wang1,
Libo Qin2, Shunyu Yao3and Wenhong Tian1
1School of Information and Software Engineering,
University of Electronic Science and Technology of China
2Central South University
3Big data and artificial intelligent institute, China Telecom Research Institute
* Corresponding Author: Jinyu Guo, Email: guojinyu@uestc.edu.cn
Abstract
Retrieval-augmented generation (RAG) has
emerged as a pivotal method for expanding
the knowledge of large language models. To
handle complex queries more effectively, re-
searchers developed Adaptive-RAG (A-RAG)
to enhance the generated quality through multi-
ple interactions with external knowledge bases.
Despite its effectiveness, A-RAG exacerbates
the pre-existing efficiency challenges inherent
in RAG, which are attributable to its reliance
on multiple iterations of generation. Existing
A-RAG approaches process all retrieved con-
tents from scratch. However, they ignore the
situation where there is a significant overlap
in the content of the retrieval results across
rounds. The overlapping content is redundantly
represented, which leads to a large proportion
of repeated computations, thus affecting the
overall efficiency. To address this issue, this
paper introduces a model-agnostic approach
that can be generally applied to A-RAG meth-
ods, which is dedicated to reducing the redun-
dant representation process caused by the over-
lapping of retrieval results. Specifically, we
use cache access and parallel generation to
speed up the prefilling and decoding stages re-
spectively. Additionally, we also propose an
instruction-driven module to further guide the
model to more effectively attend to each part of
the content in a more suitable way for LLMs.
Experiments show that our approach achieves
2.79 and 2.33 times significant acceleration on
average for prefilling and decoding respectively
while maintaining equal generation quality.
1 Introduction
Work smarter, not harder.
Allan F . Mogensen, 1930s
Large Language Models (LLMs) (e.g., LLaMA-
2, PaLM, and GPT-4 (Touvron et al., 2023; Anil
et al., 2023; Achiam et al., 2023)) have demon-
strated remarkable capabilities across various natu-
Figure 1: (a) The pipeline of A-RAG. (b) Analysis
of document overlap between first and later retrievals
(rounds 2-3) using 1000 2WikiMultihopQA samples.
ral language processing tasks. To address the grow-
ing demand for knowledge-intensive and factually
grounded responses, Retrieval-Augmented Genera-
tion (RAG) has emerged as a promising paradigm
to mitigate the inherent knowledge limitations of
LLMs (Lewis et al., 2020; Borgeaud et al., 2022;
Ram et al., 2023; Jiang et al., 2023; Asai et al.; Gao
et al., 2023; Gupta et al., 2024). It enhances the per-
formance by retrieving relevant information from
external knowledge base to generate contextually
appropriate and evidence-based responses.
Conventional RAG interacts with the external
knowledge bases once and combines the retrieved
documents with the query as input to LLMs, this
can also be called single-round RAG. Nevertheless,
single-round RAG encounters challenges in han-
dling complex user inquiries, as it is difficult to ex-
tract sufficient information by a single interaction.
Consequently, an increasing number of researchers
have begun to focus on and propose Adaptive-RAG
(A-RAG) (Borgeaud et al., 2022; Ram et al., 2023;
Mallen et al., 2023; Asai et al.; Trivedi et al., 2023;
Jiang et al., 2023; Zhao et al., 2023; Yang et al.,

2023b; Su et al., 2024; Zhang et al., 2024; Li et al.;
Yao et al., 2024; Jeong et al., 2024; Wang et al.,
2024a). Figure 1(a) illustrates the process of A-
RAG, which dynamically determines whether to
continue retrieval and adjusts the retrieval content
based on the quality of the generated responses.
Through iterative interactions with external knowl-
edge bases, A-RAG obtains more comprehensive
and valuable information to generate accurate and
comprehensive answers.
Although A-RAG enhances the performance, it
further exacerbates the inherent efficiency prob-
lems of the RAG due to external interactions and
increased computation. Within its post-retrieval
generation phase, existing methods process all re-
trieved contents from scratch. However, they over-
looked a situation where, during a single A-RAG
process, the high similarity among multiple rounds
of queries results in significant overlap in the con-
tent of the retrieved results across these rounds,
especially between adjacent rounds. Figure 1(b)
shows the overlap ratio of documents between
rounds with different settings for the number of
retrieved documents. These overlapping contents
are redundantly represented in each round, which
leads to a large proportion of repeated computa-
tions, thereby affecting the overall efficiency.
To tackle this issue, we propose Instruction-
Driven Representaion Reduction ( IDR 2), a model-
agnostic approach widely applicable to the A-RAG
methods. It aims to improve the efficiency of A-
RAG by eliminating repetitive representations of
overlapping content and redundant autoregressive
representation rounds. Concretely, we leverage rep-
resentation reduction techniques in both the prefill-
ing and decoding processes of the generation phase.
During prefilling, we propose the Cross-Iteration
Cache Sharing (CICS) module, which establishes
a shared memory repository for document repre-
sentations. It enables subsequent processing iter-
ations to bypass repetitive computations through
cached intermediate results, thereby reducing com-
putational overhead for overlapping content. In
addition, to further direct the model to more ef-
fectively attend to the various parts of the content
after prefilling, we introduce the Instruction-driven
Deduplication Guidance Reinforcement (IDGR)
module. It leverages the instruction-following ca-
pabilities of LLMs to implement context-aware
filtration, prioritizing semantically relevant cached
information while suppressing redundant content
through explicit linguistic guidance in a more suit-able way for LLMs. In the decoding process, we
propose a novel Information-Guided Parallel Gen-
eration (IGPG) module, which leverages the corre-
lation between retrieved documents and the gener-
ated results. By integrating phrasal fragments as
inputs at each autoregressive step, IGPG enables
parallel generation. It reduces the autoregressive
representation rounds to achieve acceleration.
We extensively evaluate our proposed method
on multiple datasets. The experimental results
show that our IDR 2significantly reduces the rep-
resentation process, which achieves 2.79 and 2.33
times acceleration for prefilling and decoding, con-
sequently accelerating the entire A-RAG workflow
by 2.0 times on average across various A-RAG ap-
proaches while maintaining the performance of
generation. In summary, our contributions are
mainly three-fold:
1.We propose IDR 2, an acceleration approach
for A-RAG based on representation reduction
techniques. It eliminates repetitive representa-
tions during the prefilling process and reduces
autoregressive representation redundancy in
the decoding phase, thereby speeding up the
entire A-RAG workflow almost without per-
formance loss.
2.We develop the Instruction-driven Deduplica-
tion Guidance Reinforcement module, which
leverages the instruction-following capabili-
ties of LLMs to further direct the model to
more effectively attend to the various parts of
the content after prefilling in a more suitable
way for LLMs.
3.Experimental results demonstrate that our ap-
proach significantly enhances the efficiency
of various A-RAG methods while exhibiting
robust adaptability across diverse scenarios
and various LLM scales.
2 Related Works
2.1 Adaptive-RAG
Generating satisfactory answers by single-round
RAG remains challenging (Komeili, 2021; Zhu
et al., 2021; Liu et al., 2024; Jiang et al., 2023; Yang
et al., 2023b; Ni et al., 2024). A-RAG was devel-
oped to address this limitation by enabling iterative
interactions with external knowledge bases, lead-
ing to more accurate and complete answers through
multiple retrieval rounds (Jiang et al., 2023; Yang

et al., 2023b; Ni et al., 2024). Current A-RAG ap-
proaches can be classified into two categories: 1)
Rule-based strategies, such as per-sentence itera-
tion (Trivedi et al., 2023), sliding window tokens
(Borgeaud et al., 2022; Ram et al., 2023), and con-
textual learning (Zhao et al., 2023; Zhang et al.,
2024; Li et al.). 2) Self-perception strategies evalu-
ate output confidence through internal states (Yao
et al., 2024), LLM output layer (Jiang et al., 2023;
Yang et al., 2023b; Su et al., 2024), or explicit
language-level feedback (Asai et al.). In contrast to
existing works that primarily focus on performance
enhancement, to the best of our knowledge, we are
the first to investigate the multi-turn document over-
lapping problem in A-RAG and improve efficiency
through its resolution.
2.2 Inference Acceleration
Efficiency of LLM. Research on improving LLM
efficiency can generally be classified into two cate-
gories: traditional optimization techniques (such as
quantization, pruning, knowledge distillation, etc.)
and LLM-specific optimizations which include: 1)
Early Exiting (Teerapittayanon et al., 2016; Xin
et al., 2020; Zhou et al., 2020; Kong et al., 2022;
Yang et al.; Bae et al., 2023), uses prediction heads
at different layers to enable early token exit when
confidence thresholds are met. 2) Token Pruning
(Goyal et al., 2020; Kim and Cho, 2021; Wang
et al., 2021; Kim et al., 2022; Hou et al., 2022;
Zhang et al., 2023b), retains only critical tokens
based on importance ranking.3) Speculative De-
coding (Chen et al., 2023; Leviathan et al., 2023;
Spector and Re; Yang et al., 2023a; Zhang et al.,
2023a; Kim et al., 2024; Miao et al., 2024), uses ef-
ficient smaller models to generate candidate tokens
for batch verification by LLM.
Efficiency of RAG. Speculative retrieval (Zhang
et al.) reduces retrieval overhead through local
retrieval and verification mechanisms. Parallel doc-
ument processing (Merth et al.) mitigates attention
complexity by processing multiple documents in-
dependently rather than concatenating them. Doc-
ument preprocessing (Lu et al., 2024) improves
prefilling efficiency by preprocessing and storing
document representation for the whole knowledge
base. Small expert LMs generate initial document-
specific drafts for LLM verification (Wang et al.,
2024b). In contrast to prior work, our IDR 2is a
general framework. It enhances A-RAG efficiency
by representation reduction with instruction-guided
information extraction.3 Methods
Figure 2 details the workflow of IDR 2, and we di-
vide the internal process of each iteration round
in A-RAG into three phases: retrieval, prefilling,
and decoding. When processing a query, cached
document representations in the CICS module are
first verified. These representations are loaded and
combined with uncached documents for prefilling,
where the IDGR module guides LLMs to prior-
itize relevant content while filtering noise. Be-
fore each autoregressive step, matching subsequent
phrase fragments are queried in the IGPG module
to enable parallel generation. Ultimately, these ap-
proaches enhance the overall efficiency of A-RAG.
Details of each module are given respectively in
the remainder of this section.
3.1 Cross-Iterative Cache Sharing (CICS)
A-RAG (Zhao et al., 2023; Jiang et al., 2023; Yang
et al., 2023b; Su et al., 2024; Zhang et al., 2024)
relies on multiple rounds of retrieval-generation in-
teractions to generate answers. We observe substan-
tial document overlap between adjacent retrieval
rounds. In light of this, duplicate representation for
overlapping documents introduces computational
redundancy, hampering the efficiency of A-RAG.
In order to avoid duplicate representation for
overlapping documents, we propose the cross-
iterative cache sharing (CICS) module. CICS ini-
tializes a cache space Cto store Key-Value pairs
corresponding to the documents retrieved in each
round for each query q. Because A-RAG may mod-
ify the query at the end of each iteration, we denote
the initial user query as q0. At round t, the retrieval
operation Dt=Retrieve (qt)returns a set of n
documents Dt={dt
1, dt
2, ..., dt
n}from the external
knowledge base. ndenotes the maximum number
of documents that can be retrieved in each round.
At round t, the LLM generates a sequence of tokens
At={a1
t, a2
t, ..., am
t}by:
a1
t, Kt, Vt=LLM P(qt, Dt, A<t) (1)
ai
t, ki
t, vi
t=LLM D(qt, Dt, A<t, a<i
t, Kt, Vt)
Kt=Concat (Kt, ki
t)
Vt=Concat (Vt, vi
t)(2)
where the inference process of LLM is divided into
two stages: prefilling and decoding, represented
with LLM PandLLM Drespectively. The Ktand
Vtrepresent the Key-Value pairs for DtandAt.

Figure 2: The pipeline of our IDR 2. the same color indicates the same document and representation.
Theai
tdenotes the ithgenerated token and mrep-
resents the number of generated tokens. The Eq.(2)
denoted the autoregression process, which can only
generate one token at each step. It needs to be exe-
cuted through multiple steps to obtain the complete
At. CICS stores the Kt, Vtfrom Eq.(1), which is
the representation for Dtas shown in Figure 2.
After receiving retrieved results Dt, CICS ex-
tracts existing representation from the cache to
avoid duplicate representation. Specifically, as the
example shown in Figure 2, CICS directly loads
the representation of documents #5881721 and
#12415199, then integrates them with the text of
document #12368478 for generation. This process
can be formalized as:
Ko
t, Vo
t, Do
t=Filter (Dt,C)
a1
t, Kt, Vt=LLM P(qt, Dn
t, A<t, Ko
t, Vo
t)(3)
where Do
tis the document set that has been pro-
cessed in the previous round and appears again at
the current round Dt. We can directly obtain its
corresponding representation Ko
tandVo
tfromC
without re-processing. For the new document set
Dn
t=Dt\Do
tat the current round, the prefilling
is still required.
Eq.(3) optimizes the prefilling phase by reusing
the cached representation of overlapping docu-
ments in CICS, avoiding redundant computation.
3.2 Instruction-driven Deduplication
Guidance Reinforcement (IDGR)
The CICS provides efficient representation reuse
across rounds. Notably, the Key-Value represen-
tation of each document dt
iincorporates informa-
tion from previously processed documents throughself-attention. As an example illustrated in Fig-
ure 2, at round 1, the representation of document
#12415199 contains information from documents
#5881721 and #10028469 due to the self-attention
mechanism.
The IDGR employs natural language instruc-
tionsItin the prompt to guide the LLM in filtering
redundant cached information. This module helps
the model focus on relevant content for the current
round, thereby ensuring high-quality generation.
At each iteration, the instruction Itis automatically
generated based on two rules and is subsequently
utilized during the prefilling phase, as:
a1
t, Kt, Vt=LLM P(qt, Dn
t, A<t, Ro
t, It)(4)
where Ro
tis the Ko
t, Vo
t. As shown in Figure 2,
the representation of document #10028469 is ex-
cluded from the computation. Furthermore, explicit
instructions guide the LLM to ignore the informa-
tion contained in the representation of document
#12415199.
The instruction Itis generated according to two
rules: First, we use document identifiers to explic-
itly tell the LLM which documents are relevant and
which are irrelevant for the current generation. This
helps the LLM focus on pertinent context while
filtering out redundant information. Second, we
provide document relevance rankings to the LLM.
In cases where we cannot directly adjust the input
order of documents based on their importance, ex-
plicit instructions guide the LLM to prioritize more
relevant documents.
In practice, Itcan be implemented with simple
natural language directives, leveraging the strong

semantic understanding capabilities of LLMs. For
instance, in the second round shown in Figure 2, the
instruction is formulated as: “ #5881721 ... are re-
lated docs. #10028469 is unrelated. The relevance
scores are ... ”. The relevance scores can be ob-
tained through either retriever rankings1or model-
based scoring methods2. While scoring methods
are not the focus of our study, they can be seam-
lessly integrated into our approach.
3.3 Information-Guided Parallel Generation
(IGPG)
Figure 3: The x-axis represents the length of consecutive
token combinations in the generated results. The y-
axis represents the proportion of all combinations in
the generated results that appear in the retrieved results.
(LLaMA2-7B, 2WikiMultihopQA).
RAG differs from standard LLM applications in
its access to extensive external context. For a given
query q, standard LLM inference can only utilize q
itself, whereas RAG leverages both qand retrieved
documents Dt. Furthermore, in different iterations
of A-RAG, the LLM-generated content maintains
high relevance to the retrieved documents Dt, as il-
lustrated in Figure 3. This relevance plays a crucial
role in determining the current round’s output. The
LLM generates a significant amount of existing
content. The autoregressive process of token-by-
token generation primarily contributes to the low
efficiency. In contrast to standard LLM generation,
we have the information of Dt, which contains a
substantial portion of the content that LLM must
generate. Consequently, we can utilize the informa-
tion from Dtto guide LLM in avoiding the inde-
pendent autoregressive representation process for
existing content. By constructing phrase fragments
and verifying multiple tokens during autoregres-
sion, parallel generation can be achieved.
1https://github.com/Muennighoff/sgpt
2https://huggingface.co/BAAI/bge-reranker-baseDifferent with traditional speculative decoding
approaches, IGPG requires neither the construc-
tion of small language models nor training. Dur-
ing each RAG iteration, IGPG uses Dtto con-
struct a approximate probabilistic language model
P(xt|x1, ..., x t−1)≈P(xt|xt−N+1, ..., x t−1),N
is the number of recent tokens.
Our IGPG has two steps as speculative decod-
ing approaches: draft generation and LLM paral-
lel generation. In the draft generation, by iterat-
ingMsteps, the module constructs a M-length
draft token sequence ˆAt,k={ˆa1
t,k,ˆa2
t,k, ...,ˆaM
t,k},
kdenotes the k-th autoregression step. During
LLM parallel generation, the LLM validates the
draft token sequence through a forward pass by
P(¯ai+M+1
t,k|a≤i
t,k−1,ˆa1
t,k, ...,ˆaM
t,k). The LLM exam-
ines whether each draft token aligns with the cur-
rent output, as Eq.(5). When a draft token ˆaj
t,kfails
validation, it is substituted with the LLM’s predic-
tion¯aj−1
t,k, and draft generation resumes from this
point until the complete sequence Atis produced.
aj
t,k=(
ˆaj
t, ifˆaj
t,k= ¯aj−1
t,k, j≥1
¯aj−1
t,kotherwise & stop(5)
where {¯a0
t,k, ...,¯ai+M+1
t,k}=LLM D(a≤i
t,k−1,ˆAt,k),
¯a0
t,kcorresponds to ai
t,k−1autoregressive.
If at least one token in the ˆAt,kis validated by the
LLM, combined with the token generated by the
LLM itself, the total number of generated tokens is
at least two, achieving a least 2 ×speedup.
4 Experiments
4.1 Language Models
We employed the LLaMA2-7B/13B (Touvron et al.,
2023) (L2-7B/13B) and the Vicuna-7B/13B (Chi-
ang et al., 2023) (V-7B/13B). The Vicuna model
is a chatbot fine-tuned on ShareGPT conversation
data. It contains different knowledge from LLaMA
and has an impact on A-RAG inference.
4.2 Downstream Task Datasets
We use the 2WikiMultihopQA (Ho et al., 2020),
HotpotQA (Yang et al., 2018), StrategyQA (Geva
et al., 2021) and IIRC (Ferguson et al., 2020) as
DRAGIN (Su et al., 2024). The datasets span multi-
hop question answering, common sense reason-
ing, and reading comprehension, covering different
types of reasoning tasks.

Methods LLMs2WikiMultihopQA HotpotQA StrategyQA IIRC
Pref.↑Deco. ↑E2E↑Pref.↑Deco. ↑E2E↑Pref.↑Deco. ↑E2E↑Pref.↑Deco. ↑E2E↑
FLRAG+IDR 2L2-7B 2.23× 2.37× 1.75× 2.27× 1.85× 1.53× 1.75× 1.49× 1.40× 3.08× 2.76× 2.10×
L2-13B 2.12× 2.36× 1.64× 2.29× 2.00× 1.54× 1.76× 1.62× 1.40× 2.70× 2.65× 1.83×
V-13B 2.05× 2.25× 1.60× 2.22× 1.97× 1.51× 1.65× 1.55× 1.31× 2.66× 2.72× 1.82×
FSRAG+IDR 2L2-7B 2.04× 2.64× 2.32× 2.41× 1.88× 1.69× 2.36× 1.97× 1.79× 2.92× 2.88× 2.20×
L2-13B 2.11× 2.47× 1.79× 2.16× 2.05× 1.59× 1.74× 1.59× 1.44× 2.16× 2.65× 1.74×
V-13B 2.23× 2.31× 1.75× 2.15× 2.31× 1.66× 2.02× 1.78× 1.53× 2.35× 2.69× 1.92×
FLARE+IDR 2L2-7B 2.81× 2.74× 2.60× 3.12× 2.61× 2.31× 2.71× 2.27× 1.77× 3.54× 2.97× 2.89×
L2-13B 1.98× 2.24× 2.16× 2.42× 1.87× 1.80× 2.45× 1.71× 1.69× 3.58× 3.15× 3.01×
V-13B 2.04× 2.14× 2.06× 2.27× 2.07× 1.98× 2.05× 1.93× 1.89× 2.70× 2.24× 2.22×
DRAGIN+IDR 2L2-7B 3.34× 2.77× 2.52× 4.09× 2.08× 2.09× 3.39× 1.98× 1.95× 4.49× 2.58× 2.58×
L2-13B 3.25× 3.21× 2.61× 3.97× 2.15× 2.01× 3.15× 1.98× 1.87× 4.25× 2.86× 2.65×
V-13B 3.57× 2.81× 2.55× 4.14× 2.29× 2.15× 3.25× 1.88× 1.82× 4.72× 4.00× 3.53×
Table 1: The average speedup ratios of IDR 2across different methods, models, and datasets. Pref. (prefilling)
demonstrates acceleration achieved through CICS and IDGR. Deco. (decoding) shows speedup from IGPG. E2E
(end-to-end) reflects the overall A-RAG acceleration, encompassing retrieval. ( n= 3, 1×= baseline speed)
LLMs Methods2WQA HQA SQA IIRC
EM↑F1↑EM↑F1↑Acc.↑EM↑F1↑
Llama2-7BDRAGIN † 22.5 28.68 22.6 33.02 65.10 15.93 20.24
DRAGIN+Ours 25.4 33.17 22.9 32.59 62.50 16.04 20.24
Llama2-13BDRAGIN † 30.4 39.91 31.6 42.60 66.10 18.5 22.59
DRAGIN+Ours 34.4 41.50 29.9 41.01 65.77 18.55 22.11
vicuna-13bDRAGIN † 25.4 34.90 30.1 42.50 66.20 23.90 28.11
DRAGIN+Ours 28.9 36.77 31.7 42.58 68.00 21.59 26.28
Table 2: The performance comparison on different datasets with different models. The †represents the reproduction.
4.3 Baselines
We selected four representative A-RAG methods
as baselines. These methods are characterized by
two key aspects: retrieval timing and query con-
struction. Retrieval timing determines when to
perform retrieval, while query construction decides
what content to use for retrieval. These features
directly impact both the quality and efficiency of
A-RAG workflows. FL-RAG (Khandelwal et al.,
2019; Borgeaud et al., 2022; Ram et al., 2023) re-
trieves every ztokens using previous tokens as
queries, FS-RAG (Trivedi et al., 2023) performs
retrieval after each sentence, FLARE (Jiang et al.,
2023) triggers retrieval for uncertain tokens, using
the last generated sentence as the query, and DRA-
GIN (Su et al., 2024) employs a dynamic approach
based on content importance and uncertainty, uti-
lizing attention distribution for query construction.
4.4 Implementation Details
For the retrieval modules, we select BM25 and fol-
low DRAGIN (Su et al., 2024). Additionally, we
also investigate replacing BM25 with SGPT (Muen-nighoff, 2022), a dense retrieval method. Our ex-
ternal knowledge is based on Wikipedia articles,
which are divided into passages containing 100 to-
kens each. For the 7B and 13B models, we use four
Nvidia 3090 GPUs and H800 GPUs, respectively.
nis the number of retrieved documents. Our IDR 2
based on the PyTorch (Paszke et al., 2019).
4.5 Main Results
Table 1 shows the results of our approach applied
to the different A-RAG methods, different models
with scales, and different scenarios. Table 1 shows
the acceleration ratios over baseline methods.
Overall, our method demonstrates end-to-end
speed improvements ranging from 1.31 ×to 3.53 ×.
During the prefilling phase, we observe accelera-
tion factors from 1.75 ×to 4.72 ×. These results
substantiate the efficacy of our proposed CICS and
IDGR mechanisms in effectively eliminating re-
dundant representations. For the decoding phase,
the IGPG technique achieves speedup ratios from
1.49×to 4.00 ×through the reduction of autoregres-
sive inference iterations. A comparative analysis
between the LLaMA2-7B and 13B architectures

(a) LLaMA2-7B prefilling cost.
 (b) LLaMA2-7B decoding cost.
 (c) LLaMA2-7B end-to-end cost.
(d) LLaMA2-13B prefilling cost.
 (e) LLaMA2-13B decoding cost.
 (f) LLaMA2-13B end-to-end cost.
Figure 4: The analysis of speedup for different numbers of retrieved documents.
(underlining indicates better results) shows slightly
better results for the 7B variant. We consider that
this effect stems from the fact that smaller mod-
els require less additional overhead to store and
load representations. Of the various A-RAG meth-
ods, the most significant performance improvement
(4.72×for prefilling) is achieved when applying
IDR 2to DRAGIN. We attribute this to the query re-
finement mechanism of DRAGIN which improves
query similarity, a property that effectively syner-
gizes with our representation reduction approach.
Models Methods Pref. (s) ↓Deco. (s) ↓E2E (s) ↓
L2-7BDR. 3.71 12.55 19.31
DR.+Ours 1.18 6.07 9.56
L2-13BDR. 3.84 19.26 26.31
DR.+Ours 1.24 8.06 12.34
V-13bDR. 4.34 25.03 33.47
DR.+Ours 1.29 10.49 15.18
Table 3: The runtime of IDR 2on 2WikiMultihopQA
dataset ( n= 3), DR. denotes DRAGIN (Su et al., 2024).
The specific runtime of the different phases in
the generation process are exhibited in Table 3. Our
experiments reveal that even on high-end NVIDIA
H800 GPUs, 13B-parameter models require 26.31-
33.47 seconds to process a single request. The
proposed method significantly reduces the overall
latency to 9.56-15.18 seconds, demonstrating sub-
stantial performance improvements and enhancedpracticality for real-world deployment.
Table 2 shows the performance of applying the
IDR 2to the DRAGIN method. The experimen-
tal results show that our approach effectively pre-
serves the original performance while exhibiting
strong adaptability and generalization across dif-
ferent models and tasks. Notably, our method also
carries some enhancements on 2WikiMultiHopQA.
4.6 Ablation studies
4.6.1 Impact of Retrieval Size
We analyze the impact of varying numbers of re-
trieved documents on A-RAG, with results visual-
ized in Figure 4. As the document count increases,
the context length grows monotonically, resulting
in progressively higher prefilling overhead propor-
tion. Our approach maintains consistent speedup
advantages across both prefilling and decoding.
4.6.2 Impact of Different Retrievers
Method Retriever EM↑E2E↑
DRAGIN BM25 30.4 1×
DRAGIN + IDR 2 BM25 34.4 2.61×
DRAGIN SGPT 27.3 1×
DRAGIN +IDR 2 SGPT 29.6 2.73×
Table 4: Analysis of different retrievers on the 2Wiki-
MultihopQA with LLaMA2-13B ( n= 3).
Table 4 demonstrates the effectiveness of IDR 2

Figure 5: The detailed process of A-RAG based on a specific example, with LLaMA-13B.
across different retrievers. Substituting the retriever
with SGPT results in an EM score of 29.6 and a
speedup ratio of 2.73 ×. The consistent perfor-
mance and acceleration above 2.6 ×across both
frequency-based and semantic retrievers confirms
the adaptability of IDR 2.
5 Analysis
5.1 Do these modules impact performance?
Table 5 shows how different modules affect the
generation quality. The redundant information
in the CICS module impacts the generation qual-
ity of LLM. This is evident in the performance
of LLaMA2-7B, where EM dropped from 22.0
to 20.3, with LLaMA2-13B showing similar de-
creases. The introduction of IDGR not only al-
leviates the adverse effects of redundant informa-
tion but also brings additional performance gains.
The LLaMA2-7B achieves EM scores of 25.4, and
LLaMA2-13B reaches 34.4. These results confirm
that IDGR successfully maintains inference quality.
5.2 Specific Case Studies
Figure 5 shows three iterative rounds. In Round
1(empty CICS), baseline prefilling takes 0.96s.
IGPG reduces decoding time from 4.24s to 2.16s.
After hallucination detection, the process enters
Round 2 . IDR 2reuses the cached representations
of #359518 in CICS, saving 0.53s (prefilling). Dur-
ing the decoding process, IGPG builds fragment
models, saving 2.31 seconds. For example, Nicos
Poulantzas requires six token-level autoregression:
_N,icos,_P,oul,ant,zas. After applying IGPG,
only one autoregression is needed after the prefix
"Nicos," is generated, achieving 4 ×local accelera-
tion. Similarly, dates like October 1979 ,September
1936 ,February 1568 achieve 4 ×acceleration by
retrieving corresponding fragments from IGPG us-
ing month information. At Round 3 , LLM furtherCICS IGPG IDGRLLaMA2-7B LLaMA2-13B
EM↑F1↑EM↑ F1↑
- - - 22.5 28.68 30.4 39.91
✓ - - 20.3 26.88 28.0 37.49
-✓ - 22.4 28.51 30.4 40.02
✓ ✓ - 20.2 26.78 28.3 37.62
✓ ✓ ✓ 25.4 33.17 34.4 41.50
Table 5: Analysis of the impact of different modules on
2WikiMultihopQA based on DRAGIN.
confirms Giuseppe Cesari’s lifespan and generates
the right answer. With all representations already in
CICS, the prefilling time reduces by 0.804 seconds
(about 80% of DRAGIN’s original time), achieving
5×speedup. Total savings reach 1.334s (prefilling)
and 6.16s (decoding), summing to 7.494s (48.47%
reduction) over DRAGIN’s 15.46s, achieving 1.95 ×
end-to-end acceleration.
6 Conclusion
This paper presents IDR 2, a model-agnostic ap-
proach that significantly improves the efficiency
of A-RAG by addressing redundant representation
processing across multiple iterations. To prevent
redundant computation, this paper introduces the
CICS module for caching document representa-
tions. The IDGR module then guides the genera-
tion process to focus on crucial information through
the instruction-following capability of LLMs. Fur-
thermore, the IGPG module leverages the corre-
lation between generated content and documents
to enable parallel generation of existing content,
reducing autoregressive rounds. Through extensive
experiments, IDR 2consistently demonstrates re-
markable efficiency improvements, achieving up
to 2.0×acceleration while maintaining generation
quality. Our future work aims to further compress
the KV cache while preserving its representation
capabilities.

7 Limitations
We acknowledge the limitations of this paper. Al-
though our method is a general approach, it re-
quires the corresponding LLM to be open-source
to apply the CICS and IGPG technologies in this
paper. It should be noted that the method in this pa-
per is not applicable to LLM APIs that only support
text-based interfaces. Therefore, our future work
also aims to develop more methods to overcome
the limitations of similar scenarios.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, et al. 2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 .
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin John-
son, Dmitry Lepikhin, Alexandre Passos, Siamak
Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng
Chen, et al. 2023. Palm 2 technical report. arXiv
preprint arXiv:2305.10403 .
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection. In The
Twelfth International Conference on Learning Repre-
sentations .
Sangmin Bae, Jongwoo Ko, Hwanjun Song, and Se-
Young Yun. 2023. Fast and robust early-exiting
framework for autoregressive language models with
synchronized parallel decoding. In Proceedings of
the 2023 Conference on Empirical Methods in Natu-
ral Language Processing , pages 5910–5924.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, et al. 2022.
Improving language models by retrieving from tril-
lions of tokens. In International conference on ma-
chine learning , pages 2206–2240. PMLR.
Charlie Chen, Sebastian Borgeaud, Geoffrey Irving,
Jean-Baptiste Lespiau, Laurent Sifre, and John
Jumper. 2023. Accelerating large language model
decoding with speculative sampling. arXiv preprint
arXiv:2302.01318 .
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng,
Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan
Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al.
2023. Vicuna: An open-source chatbot impressing
gpt-4 with 90%* chatgpt quality. See https://vicuna.
lmsys. org (accessed 14 April 2023) , 2(3):6.
James Ferguson, Matt Gardner, Hannaneh Hajishirzi,
Tushar Khot, and Pradeep Dasigi. 2020. Iirc: Adataset of incomplete information reading compre-
hension questions. In Proceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP) , pages 1137–1147.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen
Wang. 2023. Retrieval-augmented generation for
large language models: A survey. arXiv preprint
arXiv:2312.10997 .
Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot,
Dan Roth, and Jonathan Berant. 2021. Did aristotle
use a laptop? a question answering benchmark with
implicit reasoning strategies. Transactions of the
Association for Computational Linguistics , 9:346–
361.
Saurabh Goyal, Anamitra Roy Choudhury, Saurabh
Raje, Venkatesan Chakaravarthy, Yogish Sabharwal,
and Ashish Verma. 2020. Power-bert: Accelerating
bert inference via progressive word-vector elimina-
tion. In International Conference on Machine Learn-
ing, pages 3690–3699. PMLR.
Shailja Gupta, Rajesh Ranjan, and Surya Narayan
Singh. 2024. A comprehensive survey of retrieval-
augmented generation (rag): Evolution, current
landscape and future directions. arXiv preprint
arXiv:2410.12837 .
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps. In Proceedings of the 28th International Con-
ference on Computational Linguistics , pages 6609–
6625.
Le Hou, Richard Yuanzhe Pang, Tianyi Zhou, Yuexin
Wu, Xinying Song, Xiaodan Song, and Denny Zhou.
2022. Token dropping for efficient bert pretraining.
InProceedings of the 60th Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers) , pages 3774–3784.
Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju
Hwang, and Jong C Park. 2024. Adaptive-rag: Learn-
ing to adapt retrieval-augmented large language mod-
els through question complexity. In Proceedings of
the 2024 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers) , pages 7029–7043.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. In Proceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing , pages 7969–7992.
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke
Zettlemoyer, and Mike Lewis. 2019. Generalization
through memorization: Nearest neighbor language
models. In International Conference on Learning
Representations .

Gyuwan Kim and Kyunghyun Cho. 2021. Length-
adaptive transformer: Train once with length drop,
use anytime with search. In Joint Conference of the
59th Annual Meeting of the Association for Compu-
tational Linguistics and the 11th International Joint
Conference on Natural Language Processing, ACL-
IJCNLP 2021 , pages 6501–6511. Association for
Computational Linguistics (ACL).
Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Ji-
tendra Malik, Michael W Mahoney, Amir Gholami,
and Kurt Keutzer. 2024. Speculative decoding with
big little decoder. Advances in Neural Information
Processing Systems , 36.
Sehoon Kim, Sheng Shen, David Thorsley, Amir Gho-
lami, Woosuk Kwon, Joseph Hassoun, and Kurt
Keutzer. 2022. Learned token pruning for transform-
ers. In Proceedings of the 28th ACM SIGKDD Con-
ference on Knowledge Discovery and Data Mining ,
pages 784–794.
M Komeili. 2021. Internet-augmented dialogue genera-
tion. arXiv preprint arXiv:2107.07566 .
Jun Kong, Jin Wang, Liang-Chih Yu, and Xuejie Zhang.
2022. Accelerating inference for pretrained language
models by unified multi-perspective early exiting. In
Proceedings of the 29th International Conference on
Computational Linguistics , pages 4677–4686.
Yaniv Leviathan, Matan Kalman, and Yossi Matias.
2023. Fast inference from transformers via spec-
ulative decoding. In International Conference on
Machine Learning , pages 19274–19286. PMLR.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al. 2020. Retrieval-augmented generation
for knowledge-intensive nlp tasks. Advances in Neu-
ral Information Processing Systems , 33:9459–9474.
Xingxuan Li, Ruochen Zhao, Yew Ken Chia, Bosheng
Ding, Shafiq Joty, Soujanya Poria, and Lidong
Bing. Chain-of-knowledge: Grounding large lan-
guage models via dynamic knowledge adapting over
heterogeneous sources. In The Twelfth International
Conference on Learning Representations .
Huanshuo Liu, Bo Chen, Menghui Zhu, Jianghao Lin,
Jiarui Qin, Hao Zhang, Yang Yang, and Ruiming
Tang. 2024. Retrieval-oriented knowledge for click-
through rate prediction. In Proceedings of the 33rd
ACM International Conference on Information and
Knowledge Management , pages 1441–1451.
Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and
Yaohua Tang. 2024. Turborag: Accelerating retrieval-
augmented generation with precomputed kv caches
for chunked text. arXiv preprint arXiv:2410.07590 .
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das,
Daniel Khashabi, and Hannaneh Hajishirzi. 2023.
When not to trust language models: Investigatingeffectiveness of parametric and non-parametric mem-
ories. In Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers) , pages 9802–9822.
Thomas Merth, Qichen Fu, Mohammad Rastegari, and
Mahyar Najibi. Superposition prompting: Improv-
ing and accelerating retrieval-augmented generation.
InForty-first International Conference on Machine
Learning .
Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao
Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee
Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al.
2024. Specinfer: Accelerating large language model
serving with tree-based speculative inference and
verification. In Proceedings of the 29th ACM Interna-
tional Conference on Architectural Support for Pro-
gramming Languages and Operating Systems, Vol-
ume 3 , pages 932–949.
Niklas Muennighoff. 2022. Sgpt: Gpt sentence
embeddings for semantic search. arXiv preprint
arXiv:2202.08904 .
Shiyu Ni, Keping Bi, Jiafeng Guo, and Xueqi Cheng.
2024. When do llms need retrieval augmentation?
mitigating llms’ overconfidence helps retrieval aug-
mentation. arXiv preprint arXiv:2402.11457 .
Adam Paszke, Sam Gross, Francisco Massa, Adam
Lerer, James Bradbury, Gregory Chanan, Trevor
Killeen, Zeming Lin, Natalia Gimelshein, Luca
Antiga, et al. 2019. Pytorch: An imperative style,
high-performance deep learning library. Advances in
neural information processing systems , 32.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay,
Amnon Shashua, Kevin Leyton-Brown, and Yoav
Shoham. 2023. In-context retrieval-augmented lan-
guage models. Transactions of the Association for
Computational Linguistics , 11:1316–1331.
Benjamin Frederick Spector and Christopher Re. Accel-
erating llm inference with staged speculative decod-
ing. In Workshop on Efficient Systems for Foundation
Models@ ICML2023 .
Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu,
and Yiqun Liu. 2024. Dragin: Dynamic retrieval aug-
mented generation based on the real-time informa-
tion needs of large language models. arXiv preprint
arXiv:2403.10081 .
Surat Teerapittayanon, Bradley McDanel, and Hsiang-
Tsung Kung. 2016. Branchynet: Fast inference via
early exiting from deep neural networks. In 2016
23rd international conference on pattern recognition
(ICPR) , pages 2464–2469. IEEE.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Al-
bert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti
Bhosale, et al. 2023. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint
arXiv:2307.09288 .

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. In Proceedings of the
61st Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers) , pages
10014–10037.
Hanrui Wang, Zhekai Zhang, and Song Han. 2021. Spat-
ten: Efficient sparse attention architecture with cas-
cade token and head pruning. In 2021 IEEE Interna-
tional Symposium on High-Performance Computer
Architecture (HPCA) , pages 97–110. IEEE.
Ruobing Wang, Daren Zha, Shi Yu, Qingfei Zhao, Yux-
uan Chen, Yixuan Wang, Shuo Wang, Yukun Yan,
Zhenghao Liu, Xu Han, et al. 2024a. Retriever-
and-memory: Towards adaptive note-enhanced
retrieval-augmented generation. arXiv preprint
arXiv:2410.08821 .
Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven
Zheng, Swaroop Mishra, Vincent Perot, Yuwei
Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang,
et al. 2024b. Speculative rag: Enhancing retrieval
augmented generation through drafting. arXiv
preprint arXiv:2407.08223 .
Ji Xin, Raphael Tang, Jaejun Lee, Yaoliang Yu, and
Jimmy Lin. 2020. Deebert: Dynamic early exiting
for accelerating bert inference. In Proceedings of the
58th Annual Meeting of the Association for Compu-
tational Linguistics , pages 2246–2251.
Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin
Jiang, Linjun Yang, Rangan Majumder, and Furu
Wei. 2023a. Inference with reference: Lossless ac-
celeration of large language models. arXiv preprint
arXiv:2304.04487 .
Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dim-
itris Papailiopoulos, and Kangwook Lee. Predictive
pipelined decoding: A compute-latency trade-off for
exact llm decoding. Transactions on Machine Learn-
ing Research .
Yuchen Yang, Houqiang Li, Yanfeng Wang, and
Yu Wang. 2023b. Improving the reliability of large
language models by leveraging uncertainty-aware in-
context learning. arXiv preprint arXiv:2310.04782 .
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing . Associa-
tion for Computational Linguistics.
Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao,
Linmei Hu, Weichuan Liu, Lei Hou, and Juanzi
Li. 2024. Seakr: Self-aware knowledge retrieval
for adaptive retrieval augmented generation. arXiv
preprint arXiv:2406.19215 .Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen,
Gang Chen, and Sharad Mehrotra. 2023a. Draft
& verify: Lossless large language model accelera-
tion via self-speculative decoding. arXiv preprint
arXiv:2309.08168 .
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong
Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuan-
dong Tian, Christopher Ré, Clark Barrett, et al. 2023b.
H2o: Heavy-hitter oracle for efficient generative
inference of large language models. Advances in
Neural Information Processing Systems , 36:34661–
34710.
Zhihao Zhang, Alan Zhu, Lijie Yang, Yihua Xu, Lanting
Li, Phitchaya Mangpo Phothilimthana, and Zhihao
Jia. Accelerating iterative retrieval-augmented lan-
guage model serving with speculation. In Forty-first
International Conference on Machine Learning .
Zihan Zhang, Meng Fang, and Ling Chen. 2024. Re-
trievalqa: Assessing adaptive retrieval-augmented
generation for short-form open-domain question an-
swering. arXiv preprint arXiv:2402.16457 .
Ruochen Zhao, Xingxuan Li, Shafiq Joty, Chengwei
Qin, and Lidong Bing. 2023. Verify-and-edit: A
knowledge-enhanced chain-of-thought framework.
InThe 61st Annual Meeting Of The Association For
Computational Linguistics .
Wangchunshu Zhou, Canwen Xu, Tao Ge, Julian
McAuley, Ke Xu, and Furu Wei. 2020. Bert loses
patience: Fast and robust inference with early exit.
Advances in Neural Information Processing Systems ,
33:18330–18341.
Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming
Zheng, Soujanya Poria, and Tat-Seng Chua. 2021.
Retrieving and reading: A comprehensive survey on
open-domain question answering. arXiv preprint
arXiv:2101.00774 .