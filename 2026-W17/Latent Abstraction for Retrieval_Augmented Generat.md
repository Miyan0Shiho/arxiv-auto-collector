# Latent Abstraction for Retrieval-Augmented Generation

**Authors**: Ha Lan N. T, Minh-Anh Nguyen, Dung D. Le

**Published**: 2026-04-20 06:26:13

**PDF URL**: [https://arxiv.org/pdf/2604.17866v1](https://arxiv.org/pdf/2604.17866v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a standard approach for enhancing large language models (LLMs) with external knowledge, mitigating hallucinations, and improving factuality. However, existing systems rely on generating natural language queries at each hop and maintaining a strict architectural separation between retriever and generator, preventing them from leveraging the full representational capacity of the LLM. We propose \textbf{LAnR} (Latent Abstraction for RAG), a unified framework in which a single LLM jointly performs encoding, retrieval, and generation entirely within its own latent space. Rather than generating textual queries, LAnR produces dense retrieval vectors from the hidden states of a designated \texttt{[PRED]} token and uses them to match against encoded document representations from the same model. Furthermore, LAnR adaptively decides when sufficient evidence has been retrieved using a lightweight MLP control head over those same hidden states, eliminating both the separate retriever and explicit token-level stopping reasoning. This design is motivated by our empirical observation that answer token entropy reliably signals retrieval sufficiency. Extensive experiments on six QA benchmarks spanning single-hop and multi-hop settings demonstrate that LAnR outperforms existing RAG methods, while achieving improved inference efficiency through reduced number of retrieval calls and tighter model integration.

## Full Text


<!-- PDF content starts -->

Latent Abstraction for Retrieval-Augmented
Generation
Ha Lan N.T*, Minh-Anh Nguyen*, Dung D. Le
Center for AI Research, VinUniversity, Vietnam
{lan.nth, minh.na2, dung.ld}@vinuni.edu.vn
*Equal contribution
Abstract
Retrieval-Augmented Generation (RAG) has become a standard approach for
enhancing large language models (LLMs) with external knowledge, mitigating
hallucinations, and improving factuality. However, existing systems rely on gener-
ating natural language queries at each hop and maintaining a strict architectural
separation between retriever and generator, preventing them from leveraging the
full representational capacity of the LLM. We proposeLAnR(Latent Abstraction
for RAG), a unified framework in which a single LLM jointly performs encoding,
retrieval, and generation entirely within its own latent space. Rather than generating
textual queries, LAnR produces dense retrieval vectors from the hidden states of
a designated [PRED] token and uses them to match against encoded document
representations from the same model. Furthermore, LAnR adaptively decides when
sufficient evidence has been retrieved using a lightweight MLP control head over
those same hidden states, eliminating both the separate retriever and explicit token-
level stopping reasoning. This design is motivated by our empirical observation that
answer token entropy reliably signals retrieval sufficiency. Extensive experiments
on six QA benchmarks spanning single-hop and multi-hop settings demonstrate that
LAnR outperforms existing RAG methods, while achieving improved inference
efficiency through reduced number of retrieval calls and tighter model integration.
1 Introduction
Retrieval-Augmented Generation (RAG) has emerged as a standard paradigm for enhancing large
language models (LLMs) with external knowledge, improving factuality, and mitigating hallucinations
[Asai et al., 2023a, Ge et al., 2023, Ram et al., 2023]. By retrieving relevant documents from external
corpora and conditioning generation on this information, RAG systems enable LLMs to access up-
to-date and domain-specific knowledge beyond their parametric memory. Despite these advantages,
existing RAG frameworks exhibit several limitations. First, they rely heavily on explicit text-based
retrieval, where the model must generate natural language queries to interact with a separate retrieval
module. Second, they enforce a strict architectural separation between the retriever and the generator,
often requiring independently trained components or additional fine-tuning via reinforcement learning
(RL) or supervised fine-tuning (SFT) LLMs to work with retrieval modules [Shi et al., 2025b, Jin
et al., 2025, Trivedi et al., 2023, Asai et al., 2023b]. These design choices introduce substantial
computational overhead, limit the full utilization of the LLM’s shared representational capacity, and
increase inference latency due to the need for explicit token-level reasoning at each step, as illustrated
in Figure 1.
Recent advances in latent reasoning suggest an alternative paradigm, where LLMs perform reasoning
directly in hidden representation space rather than through fully verbalized intermediate steps [Yue
et al., 2025, Hao et al., 2024, Shi et al., 2025a, Xu et al., 2025]. Such approaches demonstrate
that latent trajectories in hidden states can encode rich reasoning processes without explicit token
Preprint. Work in progress.arXiv:2604.17866v1  [cs.CL]  20 Apr 2026

Think: First, I need to know the team David Beckham last
played for . 
Search: David Beckham Career
Beckham's professional club career began with Manchester
United, where he made his first-team debut in 1992 at age 17 …
He returned to Galaxy and won the MLS Cup in 2011 and 2012.
Following a four-month stint at Paris Saint-Germain, he retired
in May 2013 after a 20 year career .
𝑸: What is the home stadium of the team David
Beckham last played for?
Think: David Beckham's last professional team was Paris Saint-
Germain. I'll find the stadium of Paris Saint-Germain.
Search: Paris Saint-Germain home stadium
The Parc des Princes is an all-seater football stadium in Paris,
France. It is located in the southwest of the French capital,
within the 16th arrondissement, directly opposite the Stade
Jean-Bouin. The stadium has been the home of French
football club Paris Saint-Germain FC since July 1974 ….
External
Think: The home stadium for Paris Saint-Germain is the Parc
des Princes. I think I will answer now .
 Answer : Parc des Princes.𝑸: What is the home stadium of the team David
Beckham last played for?
[PRED]
Latent Query V ector
Beckham's professional club career began with Manchester
United, where he made his first-team debut in 1992 at age 17 …
He returned to Galaxy and won the MLS Cup in 2011 and 2012.
Following a four-month stint at Paris Saint-Germain, he retired
in May 2013 after a 20 year career .
Uncertain Certain[PRED]
The Parc des Princes is an all-seater football stadium in Paris,
France. It is located in the southwest of the French capital,
within the 16th arrondissement, directly opposite the Stade
Jean-Bouin. The stadium has been the home of French
football club Paris Saint-Germain FC since July 1974 ….
External
Uncertain Certain
 Answer : Parc des Princes.Figure 1:Comparison between conventional RAG and LAnR for multi-hop QA.Conventional
RAG performs explicit reasoning at each hop, including generating intermediate text, forming search
queries, and deciding whether to continue retrieval. In contrast, LAnR operates in latent space: a
special token [PRED] produces query vectors from hidden states, while a lightweight entropy-aware
MLP controls the retrieval process, enabling more efficient and integrated reasoning.
generation. However, this paradigm remains largely unexplored in the context of retrieval-augmented
systems [Zhang et al., 2025]. In contrast to conventional RAG pipelines, where models iteratively
generate search queries, retrieve documents, summarize evidence, and decide whether to continue
retrieval, we investigate whether retrieval and processing retrieved context can be integrated into
the latent reasoning process itself. To the best of our knowledge, this direction remains largely
underexplored, motivating two fundamental research questions:
1.Can LLMs implicitly learn to retrieve relevant documents without relying on explicit text
query generation?
2.Can LLMs implicitly determine how much information is sufficient, and thus decide when to
stop retrieval?
To address these questions, we propose a unifiedlatent abstraction retrieval-augmented generation
(LAnR) framework, in which retrieval is performed directly in the representation space of the LLM.
Instead of generating textual queries, the model produces latent query vectors derived from the hidden
states of designated special tokens, thereby leveraging the model’s reasoning capabilities without
being constrained by textual representations. These vectors are used to retrieve documents from a
vector database constructed using representations from the same LLM, enabling tight integration
between retrieval and generation, detailed in Section 3. To support multi-hop reasoning, we further
introduce a lightweight multi-layer perceptron (MLP) [Noriega, 2005] head that predicts whether
additional retrieval steps are necessary. This design is motivated by empirical observations that
uncertainty signals, such as entropy in the token distribution, correlate with the need for further
information acquisition, illustrated in Section 4.
Extensive experiments across multiple benchmarks demonstrate that our approach achieves superior
performance compared to state-of-the-art RAG methods, while significantly reducing inference
latency through fewer retrieval calls and eliminating the need for a separate retriever. We summarize
our main contributions as follows:
1.We introduce a novel latent RAG paradigm in which the LLM’s hidden representation space
is jointly used for document encoding, retrieval, and generation, unifying these components
within a single model.
2

2.We develop an implicit retrieval control mechanism, implemented as a lightweight head,
that determines when additional retrieval is needed without requiring explicit text-based
reasoning.
2 Related Works
2.1 Latent Reasoning
A key limitation of dominant reasoning paradigms such as explicit chain-of-thought (CoT) Wei
et al. [2022], Yao et al. [2023], Goyal et al. [2023] lies in their reliance on discrete token generation
during inference. In standard CoT decoding, the model commits to a single token at each step,
sampled from the predicted distribution. While this process enhances interpretability by verbalizing
intermediate reasoning steps, it inherently collapses the full probability distribution into a single
trajectory, discarding uncertainty and eliminating alternative reasoning paths that may be informative.
To address this limitation, recent work has explored latent reasoning Hao et al. [2024], Xu et al.
[2025], Shi et al. [2025a], Yue et al. [2025], Wang et al. [2025], where reasoning unfolds directly
in the continuous hidden-state space rather than through discrete text. This paradigm offers two
primary advantages over CoT: (1) increased representational capacity per step, as continuous vectors
can encode substantially richer information than individual tokens Chen et al. [2025c]; and (2) the
ability to implicitly preserve multiple reasoning hypotheses without prematurely collapsing them
into a single token sequence Li et al. [2025a]. However, despite these advances, prior work on latent
reasoning has primarily focused on improving reasoning quality in standalone language modeling
settings. Its application to RAG remains largely unexplored. In particular, existing methods do
not address how latent representations can be leveraged for retrieval itself, nor how they can guide
adaptive retrieval decisions.
2.2 Retrieval Augmented Generation
RAG enhances LLMs by incorporating external knowledge sources to mitigate hallucinations and
address knowledge gaps Lewis et al. [2020], Yue et al. [2024], Arslan et al. [2024]. A central
challenge in RAG systems lies in determining when and how to retrieve relevant information, as naive
single-step retrieval often introduces irrelevant or insufficient context Jiang et al. [2023], Jin et al.
[2024]. Early approaches rely on supervised fine-tuning (SFT) to train models for query generation
and retrieval integration Asai et al. [2023b], Trivedi et al. [2023], Yao et al. [2022], Schick et al. [2023],
but such methods depend on high-quality annotated trajectories and often struggle to generalize to
out-of-distribution scenarios. More recent work explores iterative and adaptive retrieval strategies,
where models interleave reasoning and retrieval in a multi-step process, commonly described as
"search-during-think" Jin et al. [2025], Shi et al. [2025b], Chen et al. [2025a]. However, existing
approaches primarily operate in the discrete text space, relying on explicit query generation and
token-level signals to control retrieval. Moreover, they often lack mechanisms for directly assessing
retrieval sufficiency or refining retrieved evidence beyond surface-level interactions.
3 Latent Query Construction
We begin by investigating latent query construction, where the LLM bypasses explicit textual query
generation and instead produces a dense retrieval vector directly from its hidden representations. To
enable this capability, the LLM is pre-trained to internalize query formulation, allowing it to infer
search intent from the input and map it directly into a retrieval vector. Formally, let x= (x 1, . . . , x T)
denote an input query token sequence. A standard autoregressive language modelMcomputes:
Ht= Transformer 
Et
,M(x t+1|x≤t) = softmax 
Wht
,
where Et= [e(x 1), . . . , e(x t)]∈Rt×dare token embeddings, Ht∈Rt×dare the last-layer hidden
states, ht=H t[t,:]∈Rd,e(·)is the embedding function, and W∈R|V|×dis the LM head. We
append a special token [PRED] to the input, forming ˜x= (x 1, . . . , x T,[PRED]) . Through causal
self-attention, the hidden state at this position attends to the full input context, yielding a latent query
vector:
q=h [PRED]∈Rd.(1)
3

Input quer y t ok ens[PRED] special t ok enR etrie v ed c ont e xt t ok ensLLM-g ener at ed t ok ensINPUT QUER Y" What is the home s t adium o f the t eam Da vid B eckham las t pla y ed f or?"1R etrie v al T urn 1Cont e xt: original ques tion onl y (x)LLMf or[PRED]Wha t isTransformer Layersq ¹C ONT INUE · RETRIEVE
A t each turn, [PRED] hidden s tat e yields lat ent quer y q, 
used f or r etrie v al and t o c ontr ol when t o s t op .Cont e xt:  c⁽ʳ⁾ = (x, D⁽¹⁾, D⁽²⁾, … , D⁽ʳ⁻¹⁾) q¹fθ( q¹ )  =  σ(MLP( q¹ ))ŷ  <  0 . 5ŷ  ≥  0 . 5C ONT INUE · RETRIEVERETRIEV A L C ONTR OL H E A D
q ⁽ʳ⁾RETRIEV A L C ONTR OL H E A Dfθ( q ⁽ʳ⁾ )  =  σ(MLP q ⁽ʳ⁾ )ŷ  <  0 . 5ŷ  ≥  0 . 5S T OP · G ENER A TEAppend R etrie v ed Docs t o c ont e xt → pr oc eed t o ne xt turn until 
the RETRIEV AL C ONTR OL HEAD r eturn S T OP signalq¹VECT OR INDEXv₁v₂v₃t op -KRETRIEVED DOCSd₁  t ok ensd₂ t o k ensc ont e xtd₁ ★B eckham ' s car eer b eg an at Man U t d …r etir ed at P SG  in Ma y 2013 .d ₂  P SG  w as f ounded in 1970  in P aris .c osine similarity⋮D ⁽¹⁾c¹ — INPUT T OKENS...pla y ed
S T OP · G ENER A TERLLMR etrie v al T urn RD ⁽²⁾D ⁽ʳ⁻¹⁾Transformer Layers[PRED]q ⁽ʳ⁾c ⁽ʳ⁾   — INPUT T OKENS...Wha t …A NS WER G ENER A T IONA ns w er :P ar c des P rinc es ....D ⁽¹⁾D ⁽ʳ⁻¹⁾D ⁽²⁾Wha t …D ⁽¹⁾ŷ  ≥  0 . 5  g ener a t e ans w erFigure 2:Overview of LAnR.Queries are injected into the LLM and combined with a [PRED]
token to form a latent query from hidden representations. This latent query is used for retrieval and to
decide whether further retrieval is needed via a lightweight MLP Retrieval Control Head. The LLM
then generates the answer from the retrieved context.
More generally, we can append N≥1 consecutive [PRED] tokens and obtain the last latent query
vectors qNfrom{q1, . . . , q N}, allowing the model to distribute the retrieval intent across multiple
embeddings for richer abstraction. We default to N=1 ; also, we can use N=0 as the last-token
hidden state of the original query serves as a baseline.
A key design principle of our framework is to derive document representations using thesame
language model. Concretely, for each document diin the corpus, we encode it with Mand extract
the hidden state of the final token as its representation. This simple yet effective strategy is consistent
with prior work [Cheng et al., 2024, Ge et al., 2023]:
vi=hlast
di∈Rd.(2)
By embedding both queries and documents within a shared representation space induced by the same
model parameters, our approach eliminates the need for a separate retriever. To enable latent query
construction, we train the model using a contrastive pre-training objective, similar to standard dense
retrieval methods [Wang et al., 2022, Chen et al., 2024], encouraging the [PRED] representations to
retrieve relevant documents. The loss is defined as:
LCL=−logexp 
sim(q, d+)/τ
exp 
sim(q, d+)/τ
+PN−
j=1exp 
sim(q, d−
j)/τ,(3)
where d+denotes the positive document embedding and {d−
j}N−
j=1are negative samples drawn
from hard distractors induced by the model’s own retrieval errors. These negatives are periodically
resampled, while document embeddings are refreshed according to Equation 2, following the standard
ANCE training strategy [Xiong et al., 2020]. Here, sim(·,·) represents cosine similarity and τis a
temperature parameter.
Preliminary results.We evaluate the effectiveness of the proposed latent query construction
mechanism, with results in Table 2 demonstrating that LLMs can learn effective implicit retrieval
directly from hidden-state representations, without relying on explicit query generation. These latent
queries are not constrained by textual forms and instead capitalize on the model’s internal reasoning
processes. This motivates our complete multi-turn retrieval method in Section 5.
4

Figure 3: Entropy distribution on HotPotQA under varying numbers of supporting documents.
4 Entropy-Guided Retrieval Control
Existing iterative RAG systems rely onexplicit token-level signalsto determine retrieval termination.
For instance, Search-R1 Jin et al. [2025] trains the model to emit a dedicated <answer> token, while
IRCoT Trivedi et al. [2023] halts retrieval upon generating phrases such asanswer is. In both cases,
the decision to stop is entangled with surface-level text generation rather than grounded in the model’s
internal assessment of evidence sufficiency. This raises a natural question: do the hidden states of an
LLM already encode a reliable signal of whether the retrieved context is adequate, even before any
answer tokens are produced?
To investigate this, we analyze the predictive entropy of the first generated token under an answer-
forcing setup, specifically immediately following the prompt prefixAnswer:. We vary the number of
gold documents included in the retrieved context and measure how this affects the entropy distribution
for the Qwen-2.5-3B model. This design isolates the model’s uncertainty at the onset of answer
generation, enabling a controlled assessment of whether entropy correlates with evidence sufficiency.
By comparing entropy trends across different levels of supporting evidence, we aim to determine
whether lower entropy reliably reflects more complete contextual grounding, thereby validating
entropy as a principled signal for adaptive retrieval decisions.
In the HotPotQA dataset, we observe a clear monotonic decrease in answer-token entropy as the
number of gold documents kin the context increases. When the context contains no gold documents
(k=0), the model exhibits relatively high entropy, reflecting substantial uncertainty over the answer
space. As kincreases, the entropy decreases steadily, with the distribution concentrating toward lower
values when sufficient supporting evidence is present. This trend is illustrated in the per-condition
box plots in Figure 3, where the distributions for k=0 and higher kvalues become progressively
more separable.
These results suggest that the model’s hidden states immediately preceding answer generation encode
a meaningful signal of retrieval sufficiency. In particular, lower entropy correlates with more complete
supporting evidence, indicating increased confidence in the generated answer. Motivated by this
observation, we propose a lightweight MLP-basedRetrieval Control Headthat operates on the
5

last-layer hidden representations of the [PRED] tokens to predict whether additional retrieval steps
are necessary. We detail the architecture and training of this component in Section 5.
5 End-to-End Framework and Training Objective
Building upon the latent query construction introduced in Section 3, we now present the complete
framework that extends single-step implicit retrieval to amulti-turnsetting. The key challenge is
determiningwhenthe model should perform additional retrieval andwhatit should retrieve at each
turn. We address both through: (1) an MLP-based retrieval control head that decides whether further
retrieval is necessary, and (2) an adaptive contrastive target mechanism that dynamically updates the
retrieval objective based on remaining unretrieved evidence, illustrated in Figure 2.
We extend the single-step pipeline of Section 3 to support iterative retrieval. At each retrieval turn
r= 1,2, . . . , R , the model constructs a latent query from the current context, which now includes
both the original question and any previously retrieved documents, and decides whether to retrieve
again or to proceed with answer generation. Letc(r)denote the accumulated context at turnr:
c(r)= 
x, D(1), D(2), . . . , D(r−1)
,
where D(s)denotes the set of top- Kdocuments retrieved at turn s. At each turn, we append [PRED]
toc(r)and extract the latent query vector: q(r)=h(r)
[PRED]∈Rd, which is used both for retrieval via
similarity matching against the document index (Equation 2) and as input to the retrieval control
head described below. The process repeats until the control head signals termination or a maximum
number of turns Ris reached, after which the model generates the final answer conditioned on the
full accumulated contextc(R+1).
MLP-based retrieval control head.A central component of our framework is a lightweight MLP
head that determines whether the currently retrieved evidence is sufficient to answer the query, or
whether additional retrieval is required, based on the observation in Section 4. At each turn r, the
retrieval control headf θtakesq(r)as input and produces a binary decision:
ˆy(r)=fθ 
q(r)
=σ 
MLP(q(r))
∈[0,1],(4)
where σis the sigmoid function. A prediction ˆy(r)≥0.5 indicates that all necessary evidence has
been retrieved and the model should proceed to generation; ˆy(r)<0.5 signals that additional retrieval
is needed. The training label is derived from the model’s own retrieval state. Let Pdenote the full
set of positive (gold) documents required to answer the query, and R(r)=D(1)∪ ··· ∪D(r)the
documents retrieved up to turnr. The ground-truth label is:
y(r)=1,ifP ⊆ R(r),
0,otherwise,(5)
i.e., the label is 1(stop) when all positive documents have been successfully retrieved, and 0
(continue) otherwise. This self-supervised formulation requires no external oracle: the signal is
generated entirely from the model’s own retrieval accuracy at training time. The control head is
trained with binary cross-entropy:
Lctrl=−RX
r=1h
y(r)log ˆy(r)+ (1−y(r)) log(1−ˆy(r))i
.(6)
In the multi-turn setting, naively using the same contrastive target at every turn is suboptimal: once a
positive document has been retrieved, it should no longer serve as the target for subsequent queries.
At turn r, the set of leftover positive documents is: P(r)=P \ R(r−1), where R(0)=∅. The
contrastive loss at turnruses a positive document sampled fromP(r):
L(r)
CL=−logexp 
sim(q(r), d+(r))/τ
exp 
sim(q(r), d+(r))/τ
+PN−
j=1exp 
sim(q(r), d−
j)/τ,(7)
where d+(r)is the embedding of a document drawn from P(r). This steers each successive turn
toward the missing evidence, avoiding redundant retrieval. The full training loss combines next-token
6

prediction, multi-turn contrastive retrieval, and the control head objective:
L=L NTP+λRX
r=1L(r)
CL+µL ctrl,(8)
where λandµare hyperparameters balancing the three objectives. The LNTP denotes the standard
next-token prediction loss, which maintains the generative capability of the LLM. We apply loss
masking to retrieved tokens so that the optimization objective is computed only over tokens generated
by the LLM, excluding retrieved content from gradient updates, following prior work [Jin et al., 2025,
Shi et al., 2025b].
Inference.At inference time, the model performs iterative retrieval starting from the input query: it
appends [PRED] , extracts q(r), retrieves the top- Kdocuments, and evaluates fθ(q(r)). If the control
head predicts continuation ( ˆy(r)<0.5 ), the retrieved documents are appended to the context and the
process repeats. Otherwise, the model generates the answer conditioned on the accumulated context,
using the following template:
Answer the given question: {question}. [PRED]
By using reference context: {context}.
Answer: {answer}
Table 1: Prompt template used in LAnR. {answer} is generated by the LLM, while {question} and
{context} are replaced with the input query and the retrieved top-Kdocuments.
A hard threshold of Rturns prevents unbounded retrieval. This procedure introduces minimal
overhead: the control head is a single MLP forward pass reusing the already-computed [PRED]
hidden state, and no additional LLM forward passes are required beyond those needed for retrieval
itself.
6 Experiments
6.1 Experimental Setting
Benchmarks.We evaluate LAnR on two benchmarks: HotpotQA (Yang et al., 2018) and Musique
(Trivedi et al., 2022b). These datasets encompass a diverse range of search with reasoning challenges,
enabling a comprehensive evaluation of LAnR.
Baselines.We compare LAnR against a diverse set of baselines spanning different retrieval and
training paradigms, as summarized in Table 2. (1)Without Retrieval: Direct generation and supervised
fine-tuning (SFT), which evaluate the model’s parametric knowledge without external context. (2)
Single-Hop Retrieval: Naive RAG Lewis et al. [2020], which performs a single round of retrieval
followed by generation. (3)Multi-Hop Retrieval: Iterative and search-based methods including
Search-o1 Li et al. [2025b], IRCoT Trivedi et al. [2023], ReSearch Chen et al. [2025b], Search-R1 Jin
et al. [2025], and AutoRefine Shi et al. [2025b], which interleave retrieval and reasoning across
multiple steps. These methods represent the current state of the art in adaptive retrieval and multi-hop
reasoning. To ensure fair comparison, all methods use the same retriever, identical knowledge corpus,
and the same maximum number of retrieved documents.
Experiment Setting.We conduct experiments using two types of models: Qwen-2.5-3B
(Base/Instruct) and Qwen-2.5-7B (Base/Instruct) Yang et al. [2025]. For retrieval, we use the
2018 Wikipedia dump Karpukhin et al. [2020] as the knowledge source. To ensure fair comparison,
we follow Jin et al. [2025] and set the number of retrieved passages to 3 across all retrieval-based
methods. For training, we merge the training sets of NQ and HotpotQA to form a unified dataset for
LAnR, following prior works Jin et al. [2025], Shi et al. [2025b]. Evaluation is conducted on the test
or validation sets of seven datasets to assess both in-domain and out-of-domain performance. Exact
Match (EM) is used as the evaluation metric.
6.2 Main Results
Table 2 presents the performance comparison across different retrieval settings. First, models without
retrieval perform poorly on both benchmarks, highlighting the limitations of relying solely on
7

Methods HotpotQA Musique
w/o Retrieval
Direct Generation 0.149 0.020
SFT 0.186 0.044
w/ Single-Hop Retrieval
Naive RAG Lewis et al. [2020] 0.255 0.047
w/ Multi-Hop Retrieval
Search-o1 Li et al. [2025b] 0.221 0.054
IRCoT Trivedi et al. [2023] 0.164 0.067
ReSearch-Instruct Chen et al. [2025b] 0.351 0.095
ReSearch-Base Chen et al. [2025b] 0.305 0.074
Search-R1-Instruct Jin et al. [2025] 0.331 0.124
Search-R1-Base Jin et al. [2025] 0.297 0.066
AutoRefine-Instruct Shi et al. [2025b] 0.404 0.169
AutoRefine-Base Shi et al. [2025b] 0.405 0.157
LAnR-Base 0.417 0.187
Table 2: Results on HotpotQA and Musique
parametric knowledge (e.g., Direct Generation achieves 0.149 / 0.020 on HotpotQA / MuSiQue).
Incorporating single-hop retrieval via naive RAG improves performance (0.255 / 0.047), but remains
insufficient for multi-hop reasoning tasks.
Multi-hop retrieval methods further boost performance, with recent approaches such as Search-R1
and AutoRefine demonstrating substantial gains by interleaving retrieval and reasoning. In particular,
AutoRefine achieves strong results (0.405 on HotpotQA and 0.169 on MuSiQue), establishing a
competitive baseline.
Our method, LAnR, achieves the best performance across both datasets, reaching0.417on HotpotQA
and0.187on MuSiQue. Notably, LAnR consistently outperforms prior state-of-the-art methods, with
larger improvements on the more challenging MuSiQue benchmark, suggesting its effectiveness in
complex multi-hop reasoning scenarios. These results demonstrate that performing retrieval in the
latent representation space provides a more effective mechanism for integrating external knowledge
compared to conventional text-based retrieval approaches.
7 Conclusion
We presentedLAnR, a unified latent retrieval-augmented generation framework that rethinks the
interaction between retrieval and reasoning in large language models. Departing from conventional
RAG pipelines that rely on explicit text-based query generation and modular retriever–generator
architectures, LAnR performs both retrieval and control directly in the model’s hidden representation
space. By introducing latent query vectors derived from [PRED] tokens and sharing representations
between queries and documents, our approach eliminates the need for a separate retriever while
enabling tighter integration between knowledge access and generation.
A key insight of this work is that LLM hidden states inherently encode signals useful not only for
reasoning, but also for retrieval and decision-making. We demonstrate that entropy in the model’s
predictive distribution serves as a reliable proxy for evidence sufficiency, motivating a lightweight
retrieval control head that adaptively determines when to continue or stop retrieval. Combined
with an adaptive multi-turn contrastive objective, this design enables efficient and targeted evidence
acquisition without explicit supervision over retrieval trajectories.
Empirical results show that LAnR achieves strong performance across multi-hop QA benchmarks
while reducing inference latency and retrieval overhead. Beyond performance gains, our findings
suggest a broader paradigm shift: retrieval can be treated as an internal capability of language models
rather than an external tool invoked through text. This opens new avenues for integrating reasoning,
memory, and control within a single model.
8

References
Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. A survey on rag with
llms.Procedia computer science, 246:3781–3790, 2024.
Akari Asai, Sewon Min, Zexuan Zhong, and Danqi Chen. Retrieval-based language models and
applications. InProceedings of the 61st Annual Meeting of the Association for Computational
Linguistics (Volume 6: Tutorial Abstracts), pages 41–46, 2023a.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to
retrieve, generate, and critique through self-reflection. InThe Twelfth International Conference on
Learning Representations, 2023b.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge
distillation.arXiv preprint arXiv:2402.03216, 4(5), 2024.
Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen
Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al. Learning to reason with search for llms via
reinforcement learning.arXiv preprint arXiv:2503.19470, 2025a.
Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen
Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, et al. Learning to reason with search for llms via
reinforcement learning.arXiv preprint arXiv:2503.19470, 2025b.
Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian
Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on
latent chain-of-thought reasoning.arXiv preprint arXiv:2505.16782, 2025c.
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and
Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one
token.Advances in Neural Information Processing Systems, 37:109487–109516, 2024.
Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for
context compression in a large language model.arXiv preprint arXiv:2307.06945, 2023.
Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh
Nagarajan. Think before you speak: Training language models with pause tokens.arXiv preprint
arXiv:2310.02226, 2023.
Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong
Tian. Training large language models to reason in a continuous latent space.arXiv preprint
arXiv:2412.06769, 2024.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 conference on empirical methods in natural language processing, pages 7969–7992, 2023.
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. Long-context llms meet rag: Overcoming
challenges for long inputs in rag.arXiv preprint arXiv:2410.05983, 2024.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and
Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement
learning.arXiv preprint arXiv:2503.09516, 2025.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In
Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP),
pages 6769–6781, 2020.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks.Advances in neural information processing systems, 33:
9459–9474, 2020.
9

Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, and
Rex Ying. Implicit reasoning in large language models: A comprehensive survey.arXiv preprint
arXiv:2509.02350, 2025a.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and
Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. InProceedings of
the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5420–5438,
2025b.
Leonardo Noriega. Multilayer perceptron tutorial.School of Computing. Staffordshire University, 4
(5):444, 2005.
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and
Yoav Shoham. In-context retrieval-augmented language models.Transactions of the Association
for Computational Linguistics, 11:1316–1331, 2023.
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke
Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach
themselves to use tools.Advances in neural information processing systems, 36:68539–68551,
2023.
Dachuan Shi, Abedelkadir Asi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, and Wen Xiao.
Swireasoning: Switch-thinking in latent and explicit for pareto-superior reasoning llms.arXiv
preprint arXiv:2510.05069, 2025a.
Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, and
Xiang Wang. Search and refine during think: Facilitating knowledge refinement for improved
retrieval-augmented reasoning.arXiv preprint arXiv:2505.11277, 2025b.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for computational linguistics (volume 1: long papers),
pages 10014–10037, 2023.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533, 2022.
Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, and Bang Liu. System-1.5 reasoning: Traversal in
language and latent spaces with dynamic shortcuts.arXiv preprint arXiv:2505.18962, 2025.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.Advances in
neural information processing systems, 35:24824–24837, 2022.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and
Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text
retrieval.arXiv preprint arXiv:2007.00808, 2020.
Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. Softcot: Soft chain-of-thought for efficient rea-
soning with llms. InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 23336–23351, 2025.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388,
2025.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan
Cao. React: Synergizing reasoning and acting in language models. InThe eleventh international
conference on learning representations, 2022.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
Tree of thoughts: Deliberate problem solving with large language models.Advances in neural
information processing systems, 36:11809–11822, 2023.
10

Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong
Wang, Xuanhui Wang, and Michael Bendersky. Inference scaling for long-context retrieval
augmented generation.arXiv preprint arXiv:2410.04343, 2024.
Zhenrui Yue, Bowen Jin, Huimin Zeng, Honglei Zhuang, Zhen Qin, Jinsung Yoon, Lanyu Shang,
Jiawei Han, and Dong Wang. Hybrid latent reasoning via reinforcement learning.arXiv preprint
arXiv:2505.18454, 2025.
Wenzheng Zhang, Xi Victoria Lin, Karl Stratos, Wen-tau Yih, and Mingda Chen. Imprag: Retrieval-
augmented generation with implicit queries.arXiv preprint arXiv:2506.02279, 2025.
11